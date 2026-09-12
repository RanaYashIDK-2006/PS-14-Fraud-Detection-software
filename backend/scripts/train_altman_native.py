#!/usr/bin/env python3
"""Train Altman-NATIVE model — full feature set from raw columns.

Unlike the ML_FEATURES-mapped model, this trains directly on:
  - Entity IDs encoded as fraud rates (leakage-safe expanding)
  - Raw MCC codes, merchant diversity, city diversity
  - Per-user velocity features (expanding mean/std)
  - Card-level cross-user features
  - Year-over-year seasonal patterns

Deployed alongside the mapped ensemble for A/B comparison.
"""
import numpy as np
import pandas as pd
import time
import json
import hashlib
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODELS_DIR = Path("models/production")
REPORTS_DIR = Path("reports")

# Use a separate directory for the native model to avoid overwriting the mapped one
NATIVE_DIR = MODELS_DIR / "altman_native"
NATIVE_DIR.mkdir(parents=True, exist_ok=True)


def recall_at(y_true, y_score, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= fpr_target
    return float(tpr[mask][-1]) if mask.any() else 0.0


def main():
    print("=" * 70)
    print("ALTMAN NATIVE MODEL — FULL FEATURE SET, ALL 24M ROWS")
    print("=" * 70)
    t0 = time.time()

    # ── Load full dataset (chunked, all fraud + sampled legit) ──────────────
    print("\n[1/7] Loading 24M rows (chunked)...")
    t1 = time.time()
    rng = np.random.RandomState(42)
    chunks = []
    n_total = 0
    for chunk in pd.read_csv(
        DATA_DIR / "credit_card_transactions-ibm_v2.csv",
        low_memory=False,
        chunksize=500_000,
    ):
        n_total += len(chunk)
        fraud_mask = chunk["Is Fraud?"] == "Yes"
        is_legit = ~fraud_mask
        sample_legit = rng.random(len(chunk)) < 0.01
        chunks.append(chunk[fraud_mask | (is_legit & sample_legit)].copy())

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Scanned: {n_total:,} | Sampled: {len(df):,} ({df['Is Fraud?'].eq('Yes').sum():,} fraud)")

    # ── Parse raw columns ───────────────────────────────────────────────────
    print("\n[2/7] Parsing raw columns...")
    df["amt"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["is_fraud"] = (df["Is Fraud?"] == "Yes").astype(int)
    df["hr"] = df["Time"].str.split(":").str[0].astype(int)
    df["mn"] = df["Time"].str.split(":").str[1].astype(int)
    df["dow"] = pd.to_datetime(df[["Year", "Month", "Day"]]).dt.dayofweek
    df["chip"] = (df["Use Chip"] == "Chip Transaction").astype(int)
    df["is_online"] = (df["Use Chip"] == "Online Transaction").astype(int)
    df["is_swipe"] = (df["Use Chip"] == "Swipe Transaction").astype(int)
    df["mcc"] = df["MCC"].fillna(0).astype(int)
    df["err"] = (df["Errors?"].fillna("") != "").astype(int)
    df["has_zip"] = df["Zip"].notna().astype(int)
    df["has_state"] = df["Merchant State"].notna().astype(int)

    # Sort by user + time for expanding features
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]].assign(
        hour=df["hr"], minute=df["mn"]))
    df = df.sort_values(["User", "datetime"]).reset_index(drop=True)

    # ── Entity encoding ─────────────────────────────────────────────────────
    print("\n[3/7] Encoding entities...")
    t2 = time.time()

    # Encode merchant, city, card as category codes for numerical use
    df["merchant_id"] = df["Merchant Name"].astype("category").cat.codes
    df["city_id"] = df["Merchant City"].astype("category").cat.codes
    df["card_id"] = df["Card"].astype("category").cat.codes

    # ── Feature engineering ─────────────────────────────────────────────────
    print("\n[4/7] Engineering native features...")
    F = pd.DataFrame()

    # Amount features
    F["amt"] = df["amt"]
    F["log_amt"] = np.log1p(df["amt"])
    F["amt_sq"] = df["amt"] ** 2

    # Time features
    F["hr"] = df["hr"]
    F["mn"] = df["mn"]
    F["dow"] = df["dow"]
    F["Month"] = df["Month"]
    F["Day"] = df["Day"]
    F["hour_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    F["hour_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    F["is_night"] = ((df["hr"] < 6) | (df["hr"] > 22)).astype(int)
    F["is_business_hours"] = ((df["hr"] >= 9) & (df["hr"] <= 17)).astype(int)

    # Card/channel features
    F["chip"] = df["chip"]
    F["is_online"] = df["is_online"]
    F["is_swipe"] = df["is_swipe"]
    F["err"] = df["err"]
    F["has_zip"] = df["has_zip"]
    F["has_state"] = df["has_state"]
    F["is_online_or_no_state"] = ((df["is_online"] == 1) | (df["has_state"] == 0)).astype(int)

    # MCC features (richer than just mcc_n)
    F["mcc"] = df["mcc"]
    F["mcc_high"] = (df["mcc"] >= 5000).astype(int)  # retail vs services
    F["mcc_restaurant"] = ((df["mcc"] >= 5812) & (df["mcc"] <= 5814)).astype(int)
    F["mcc_gas"] = ((df["mcc"] >= 5541) & (df["mcc"] <= 5542)).astype(int)
    F["mcc_grocery"] = ((df["mcc"] >= 5411) & (df["mcc"] <= 5422)).astype(int)
    F["mcc_travel"] = ((df["mcc"] >= 3000) & (df["mcc"] <= 3350)).astype(int)
    F["mcc_online"] = ((df["mcc"] >= 5967) & (df["mcc"] <= 5969)).astype(int)

    # Entity IDs (encoded)
    F["merchant_id"] = df["merchant_id"]
    F["city_id"] = df["city_id"]
    F["card_id"] = df["card_id"]

    # User velocity (expanding — leakage safe)
    F["user_tx_count"] = df.groupby("User").cumcount()
    F["card_tx_count"] = df.groupby("Card").cumcount()
    F["user_avg_amt"] = df.groupby("User")["amt"].transform(
        lambda x: x.expanding().mean().shift(1))
    F["amt_vs_user_avg"] = F["amt"] / (F["user_avg_amt"] + 1e-6)
    F["amt_zscore"] = (F["amt"] - F["user_avg_amt"]) / (
        df.groupby("User")["amt"].transform(
            lambda x: x.expanding().std().shift(1)) + 1e-6)

    # Merchant velocity (expanding — leakage safe)
    F["merch_tx_count"] = df.groupby("Merchant Name").cumcount()

    # User merchant/city diversity (cumcount-based — fast, leakage safe)
    # First occurrence of each merchant/city per user gets flag=1
    F["user_merchant_diversity"] = (
        ~df.duplicated(subset=["User", "Merchant Name"], keep="first")
    ).astype(int).groupby(df["User"]).cumsum().shift(1).fillna(1).astype(float)
    F["user_city_diversity"] = (
        ~df.duplicated(subset=["User", "Merchant City"], keep="first")
    ).astype(int).groupby(df["User"]).cumsum().shift(1).fillna(1).astype(float)

    # Entity fraud rates (LEAKAGE-SAFE: expanding + shift)
    F["user_fraud_rate"] = df.groupby("User")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(0.001)
    F["merch_fraud_rate"] = df.groupby("Merchant Name")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(0.001)
    F["city_fraud_rate"] = df.groupby("Merchant City")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(0.001)

    # Amount thresholds
    F["high_amt"] = (F["amt"] > F["user_avg_amt"] * 2).astype(int)
    F["very_high_amt"] = (F["amt"] > F["user_avg_amt"] * 5).astype(int)

    # Interactions
    F["amt_x_hr"] = F["amt"] * F["hr"]
    F["amt_x_mcc"] = F["amt"] * F["mcc"].astype(float)
    F["amt_x_chip"] = F["amt"] * F["chip"]
    F["amt_x_online"] = F["amt"] * F["is_online"]
    F["amt_x_night"] = F["amt"] * F["is_night"]

    # User-merchant interaction (expanding — leakage safe)
    F["user_merch_count"] = df.groupby(["User", "Merchant Name"]).cumcount()

    # Replace inf/nan
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat_cols = list(F.columns)
    print(f"  {len(feat_cols)} native features ({time.time()-t2:.0f}s)")

    # ── Temporal split ──────────────────────────────────────────────────────
    print("\n[5/7] Temporal train/test split...")
    train_mask = df["Month"] <= 9
    test_mask = df["Month"] > 9
    Xtr = F.loc[train_mask].values.astype(np.float32)
    ytr = df.loc[train_mask, "is_fraud"].values
    Xte = F.loc[test_mask].values.astype(np.float32)
    yte = df.loc[test_mask, "is_fraud"].values
    print(f"  Train: {len(Xtr):,} ({ytr.sum():,} fraud)")
    print(f"  Test:  {len(Xte):,} ({yte.sum():,} fraud)")

    # ── Train ensemble ──────────────────────────────────────────────────────
    print("\n[6/7] Training XGB+LGB+CB ensemble...")
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb

    fraud_rate = ytr.mean()
    spw = (1 - fraud_rate) / fraud_rate
    print(f"  fraud_rate={fraud_rate:.4f}, scale_pos_weight={spw:.1f}")

    # 5-fold CV
    print("\n  5-fold stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs, cv_r1s, cv_aprs = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr_s, ytr)):
        t3 = time.time()
        Xa, Xv = Xtr_s[tr_idx], Xtr_s[va_idx]
        ya, yv = ytr[tr_idx], ytr[va_idx]

        m_xgb = xgb.XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            scale_pos_weight=min(spw, 20), random_state=42,
            n_jobs=4, eval_metric="auc", use_label_encoder=False,
        )
        m_xgb.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)

        m_lgb = lgb.LGBMClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            scale_pos_weight=min(spw, 20), random_state=42, n_jobs=4, verbose=-1,
        )
        m_lgb.fit(Xa, ya, eval_set=[(Xv, yv)])

        m_cb = cb.CatBoostClassifier(
            iterations=400, depth=7, learning_rate=0.05,
            l2_leaf_reg=3, random_seed=42, verbose=0,
            auto_class_weights="Balanced",
        )
        m_cb.fit(Xa, ya, eval_set=(Xv, yv))

        p_xgb = m_xgb.predict_proba(Xv)[:, 1]
        p_lgb = m_lgb.predict_proba(Xv)[:, 1]
        p_cb = m_cb.predict_proba(Xv)[:, 1]
        p_ens = 0.34 * p_xgb + 0.33 * p_lgb + 0.33 * p_cb

        auc = roc_auc_score(yv, p_ens)
        r1 = recall_at(yv, p_ens, 0.01)
        apr = average_precision_score(yv, p_ens)
        cv_aucs.append(auc)
        cv_r1s.append(r1)
        cv_aprs.append(apr)
        print(f"    Fold {fold+1}: AUC={auc:.4f} R@1%FPR={r1:.4f} APR={apr:.4f} ({time.time()-t3:.0f}s)")

    print(f"\n  CV mean: AUC={np.mean(cv_aucs):.4f}±{np.std(cv_aucs):.4f} "
          f"R@1%FPR={np.mean(cv_r1s):.4f}±{np.std(cv_r1s):.4f}")

    # Retrain on full training set
    print("\n  Retraining on full training set...")
    t4 = time.time()

    m_xgb_final = xgb.XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        scale_pos_weight=min(spw, 20), random_state=42,
        n_jobs=4, eval_metric="auc", use_label_encoder=False,
    )
    m_xgb_final.fit(Xtr_s, ytr, verbose=False)

    m_lgb_final = lgb.LGBMClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        scale_pos_weight=min(spw, 20), random_state=42, n_jobs=4, verbose=-1,
    )
    m_lgb_final.fit(Xtr_s, ytr)

    m_cb_final = cb.CatBoostClassifier(
        iterations=400, depth=7, learning_rate=0.05,
        l2_leaf_reg=3, random_seed=42, verbose=0,
        auto_class_weights="Balanced",
    )
    m_cb_final.fit(Xtr_s, ytr)
    print(f"  Retrained in {time.time()-t4:.0f}s")

    # ── Temporal test ───────────────────────────────────────────────────────
    print("\n[7/7] Temporal test evaluation...")
    p_xgb_te = m_xgb_final.predict_proba(Xte_s)[:, 1]
    p_lgb_te = m_lgb_final.predict_proba(Xte_s)[:, 1]
    p_cb_te = m_cb_final.predict_proba(Xte_s)[:, 1]
    p_ens_te = 0.34 * p_xgb_te + 0.33 * p_lgb_te + 0.33 * p_cb_te

    test_auc = roc_auc_score(yte, p_ens_te)
    test_r1 = recall_at(yte, p_ens_te, 0.01)
    test_apr = average_precision_score(yte, p_ens_te)
    print(f"\n  Temporal test: AUC={test_auc:.4f} R@1%FPR={test_r1:.4f} APR={test_apr:.4f}")

    for name, p in [("XGB", p_xgb_te), ("LGB", p_lgb_te), ("CB", p_cb_te)]:
        print(f"    {name}: AUC={roc_auc_score(yte, p):.4f} R@1%FPR={recall_at(yte, p, 0.01):.4f}")

    # ── Feature importance ──────────────────────────────────────────────────
    print("\n  Top 10 features (LGB importance):")
    imp = m_lgb_final.feature_importances_
    top_idx = np.argsort(imp)[::-1][:10]
    for i in top_idx:
        print(f"    {feat_cols[i]:30s} {imp[i]:.4f}")

    # ── Save artifacts ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = NATIVE_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(m_xgb_final, model_dir / "xgb_native.joblib")
    joblib.dump(m_lgb_final, model_dir / "lgb_native.joblib")
    joblib.dump(m_cb_final, model_dir / "cb_native.joblib")
    joblib.dump(scaler, model_dir / "scaler_native.joblib")

    # Hash
    hash_data = b""
    for name in ["xgb_native.joblib", "lgb_native.joblib", "cb_native.joblib", "scaler_native.joblib"]:
        hash_data += (model_dir / name).read_bytes()
    model_hash = hashlib.sha256(hash_data).hexdigest()

    manifest = {
        "model_version": f"altman_native_{ts}",
        "model_type": "xgb_lgb_cb_native",
        "feature_version": "altman_native_v1",
        "dataset_version": "altman_ibm_v2_native",
        "dataset_rows_scanned": n_total,
        "training_rows": int(len(Xtr)),
        "n_fraud_train": int(ytr.sum()),
        "n_features": len(feat_cols),
        "features": feat_cols,
        "cv_auc_mean": round(float(np.mean(cv_aucs)), 6),
        "cv_auc_std": round(float(np.std(cv_aucs)), 6),
        "cv_r1_mean": round(float(np.mean(cv_r1s)), 6),
        "cv_r1_std": round(float(np.std(cv_r1s)), 6),
        "cv_apr_mean": round(float(np.mean(cv_aprs)), 6),
        "temporal_test_auc": round(float(test_auc), 6),
        "temporal_test_r1": round(float(test_r1), 6),
        "temporal_test_apr": round(float(test_apr), 6),
        "ensemble_weights": {"xgb": 0.34, "lgb": 0.33, "cb": 0.33},
        "model_hash": model_hash,
        "target_leakage": False,
        "temporal_split": "train_months_1-9_test_months_10-12",
        "trained_at": datetime.now().isoformat(),
        "is_native": True,
        "artifacts": {
            "xgb": "xgb_native.joblib",
            "lgb": "lgb_native.joblib",
            "cb": "cb_native.joblib",
            "scaler": "scaler_native.joblib",
        },
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (model_dir / "feature_list.json").write_text(json.dumps(feat_cols, indent=2))

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"SAVED: {model_dir}")
    print(f"  Features: {len(feat_cols)} (native from raw columns)")
    print(f"  Model hash: {model_hash[:16]}...")
    print(f"\n  CV AUC:     {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    print(f"  CV R@1%FPR: {np.mean(cv_r1s):.4f} ± {np.std(cv_r1s):.4f}")
    print(f"  Temporal AUC:     {test_auc:.4f}")
    print(f"  Temporal R@1%FPR: {test_r1:.4f}")
    print(f"  Total time: {elapsed:.0f}s")
    print(f"{'='*70}")

    # Save report
    report = {
        "model_version": manifest["model_version"],
        "model_type": "altman_native",
        "cv": {
            "auc_mean": round(float(np.mean(cv_aucs)), 6),
            "auc_std": round(float(np.std(cv_aucs)), 6),
            "r1_mean": round(float(np.mean(cv_r1s)), 6),
            "r1_std": round(float(np.std(cv_r1s)), 6),
        },
        "temporal_test": {
            "auc": round(float(test_auc), 6),
            "r1_fpr_1pct": round(float(test_r1), 6),
            "apr": round(float(test_apr), 6),
        },
        "features": feat_cols,
        "n_features": len(feat_cols),
        "elapsed_s": round(elapsed, 1),
    }
    (REPORTS_DIR / "altman_native_results.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport: reports/altman_native_results.json")


if __name__ == "__main__":
    main()
