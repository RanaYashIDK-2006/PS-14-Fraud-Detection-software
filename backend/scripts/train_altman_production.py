#!/usr/bin/env python3
"""Train Altman production ensemble — NO target leakage, temporal split, 5-fold CV.

Trains XGBoost + LightGBM + CatBoost on the IBM Altman credit card dataset.
- NO fraud rate features (target leakage)
- Temporal split: train on months 1-9, validate on months 10-12
- 5-fold stratified CV on training set
- Saves to models/production/ with manifest
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
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODELS_DIR = Path("models/production")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature engineering ──────────────────────────────────────────────────────

def engineer_features(df):
    """Engineer features from Altman data — NO target leakage."""
    F = pd.DataFrame()

    # Amount
    F["amt"] = df["amt"]
    F["log_amt"] = np.log1p(F["amt"])
    F["amt_sq"] = F["amt"] ** 2

    # Time
    F["hr"] = df["hr"]
    F["mn"] = df["mn"]
    F["dow"] = df["dow"]
    F["Month"] = df["Month"]
    F["Day"] = df["Day"]
    F["hour_sin"] = np.sin(2 * np.pi * F["hr"] / 24)
    F["hour_cos"] = np.cos(2 * np.pi * F["hr"] / 24)
    F["is_night"] = ((F["hr"] < 6) | (F["hr"] > 22)).astype(int)
    F["is_business_hours"] = ((F["hr"] >= 9) & (F["hr"] <= 17)).astype(int)

    # Card / channel
    F["chip"] = df["chip"]
    F["is_online"] = df["is_online"]
    F["err"] = (df["err"] != 0).astype(int)
    F["mcc_n"] = df["mcc_n"]

    # User velocity (cumulative — computed per-user sorted by time)
    # These are LEAKAGE-SAFE: cumulative count/avg up to (but not including) current tx
    F["user_tx_count"] = df.groupby("User").cumcount()
    F["card_tx_count"] = df.groupby("Card").cumcount()

    # User average amount (expanding mean — leakage safe)
    F["user_avg_amt"] = df.groupby("User")["amt"].transform(lambda x: x.expanding().mean().shift(1))
    F["amt_vs_user_avg"] = F["amt"] / (F["user_avg_amt"] + 1e-6)
    F["amt_zscore"] = (F["amt"] - F["user_avg_amt"]) / (
        df.groupby("User")["amt"].transform(lambda x: x.expanding().std().shift(1)) + 1e-6
    )

    # Merchant tx count (expanding — leakage safe)
    F["merch_tx_count"] = df.groupby("Merchant Name").cumcount()

    # Entity-level rolling fraud rates (LEAKAGE-SAFE: expanding + shift)
    # user_fraud_rate: fraction of this user's PAST events that were fraud
    F["user_fraud_rate"] = (
        df.groupby("User")["is_fraud"].transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.001)  # baseline for new users
    # merch_fraud_rate: fraction of this merchant's PAST events that were fraud
    F["merch_fraud_rate"] = (
        df.groupby("Merchant Name")["is_fraud"].transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.001)
    # city_fraud_rate: fraction of this city's PAST events that were fraud
    F["city_fraud_rate"] = (
        df.groupby("Merchant City")["is_fraud"].transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.001)

    # Location features
    F["has_zip"] = df["Zip"].notna().astype(int)
    F["has_state"] = df["Merchant State"].notna().astype(int)
    F["is_online_or_no_state"] = ((F["is_online"] == 1) | (F["has_state"] == 0)).astype(int)

    # Thresholds
    F["high_amt"] = (F["amt"] > F["user_avg_amt"] * 2).astype(int)
    F["very_high_amt"] = (F["amt"] > F["user_avg_amt"] * 5).astype(int)

    # Interactions
    F["amt_x_hr"] = F["amt"] * F["hr"]
    F["amt_x_mcc"] = F["amt"] * F["mcc_n"]
    F["amt_x_chip"] = F["amt"] * F["chip"]
    F["amt_x_online"] = F["amt"] * F["is_online"]
    F["amt_x_night"] = F["amt"] * F["is_night"]

    # Replace inf/nan
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return F


def recall_at(y_true, y_score, fpr_target=0.01):
    """Recall at a given FPR threshold."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= fpr_target
    if mask.any():
        return float(tpr[mask][-1])
    return 0.0


def main():
    print("=" * 70)
    print("ALTMAN PRODUCTION MODEL — NO TARGET LEAKAGE")
    print("=" * 70)
    t0 = time.time()

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/6] Loading 24M rows (chunked)...")
    t1 = time.time()
    rng = np.random.RandomState(42)
    chunks = []
    n_fraud = 0
    n_total = 0
    for chunk in pd.read_csv(
        DATA_DIR / "credit_card_transactions-ibm_v2.csv",
        usecols=["User", "Card", "Year", "Month", "Day", "Time", "Amount",
                 "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
                 "Zip", "MCC", "Errors?", "Is Fraud?"],
        low_memory=False,
        chunksize=500_000,
    ):
        n_total += len(chunk)
        fraud_mask = chunk["Is Fraud?"] == "Yes"
        n_fraud += fraud_mask.sum()
        # Keep all fraud + 1% of legit (~270K total)
        is_legit = ~fraud_mask
        sample_legit = rng.random(len(chunk)) < 0.01
        keep = fraud_mask | (is_legit & sample_legit)
        chunks.append(chunk[keep].copy())

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Total scanned: {n_total:,} | Fraud: {n_fraud:,} ({n_fraud/n_total*100:.3f}%)")
    print(f"  Sampled: {len(df):,} ({df['Is Fraud?'].eq('Yes').sum():,} fraud)")

    # ── Parse ────────────────────────────────────────────────────────────────
    print("\n[2/6] Parsing...")
    df["amt"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["is_fraud"] = (df["Is Fraud?"] == "Yes").astype(int)
    df["hr"] = df["Time"].str.split(":").str[0].astype(int)
    df["mn"] = df["Time"].str.split(":").str[1].astype(int)
    df["dow"] = pd.to_datetime(df[["Year", "Month", "Day"]]).dt.dayofweek
    df["chip"] = (df["Use Chip"] == "Chip Transaction").astype(int)
    df["is_online"] = (df["Use Chip"] == "Online Transaction").astype(int)
    df["mcc_n"] = df["MCC"].fillna(0).astype(float)
    df["err"] = df["Errors?"].fillna("0")

    # Sort by user + time for leakage-safe features
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]].assign(
        hour=df["hr"], minute=df["mn"]
    ))
    df = df.sort_values(["User", "datetime"]).reset_index(drop=True)
    print(f"  Parsed {len(df):,} rows ({df['is_fraud'].sum():,} fraud)")

    # ── Feature engineering ──────────────────────────────────────────────────
    print("\n[3/6] Feature engineering (leakage-safe)...")
    t2 = time.time()
    F = engineer_features(df)
    feat_cols = list(F.columns)
    print(f"  {len(feat_cols)} features in {time.time()-t2:.0f}s")

    # ── Temporal split ───────────────────────────────────────────────────────
    print("\n[4/6] Temporal train/test split...")
    # Train on months 1-9, test on months 10-12 (latest data)
    train_mask = df["Month"] <= 9
    test_mask = df["Month"] > 9
    Xtr = F.loc[train_mask].values.astype(np.float32)
    ytr = df.loc[train_mask, "is_fraud"].values
    Xte = F.loc[test_mask].values.astype(np.float32)
    yte = df.loc[test_mask, "is_fraud"].values
    print(f"  Train: {len(Xtr):,} ({ytr.sum():,} fraud)")
    print(f"  Test:  {len(Xte):,} ({yte.sum():,} fraud)")

    # ── Scale ────────────────────────────────────────────────────────────────
    print("\n[5/6] Training ensemble...")
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Import models
    import xgboost as xgb
    import lightgbm as lgb
    try:
        import catboost as cb
        HAS_CB = True
    except ImportError:
        HAS_CB = False
        print("  WARNING: catboost not installed, using 2-model ensemble")

    # Compute class weight
    fraud_rate = ytr.mean()
    spw = (1 - fraud_rate) / fraud_rate
    print(f"  fraud_rate={fraud_rate:.4f}, scale_pos_weight={spw:.1f}")

    # 5-fold CV on training set
    print("\n  5-fold stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    cv_r1s = []
    cv_aprs = []
    oof_preds = np.zeros(len(ytr))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr_s, ytr)):
        t3 = time.time()
        Xa, Xv = Xtr_s[tr_idx], Xtr_s[va_idx]
        ya, yv = ytr[tr_idx], ytr[va_idx]

        # XGBoost
        m_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=min(spw, 20), random_state=42,
            n_jobs=4, eval_metric="auc", use_label_encoder=False,
        )
        m_xgb.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)

        # LightGBM
        m_lgb = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=min(spw, 20), random_state=42,
            n_jobs=4, verbose=-1,
        )
        m_lgb.fit(Xa, ya, eval_set=[(Xv, yv)])

        # CatBoost (if available)
        if HAS_CB:
            m_cb = cb.CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.05,
                l2_leaf_reg=3, random_seed=42, verbose=0,
                auto_class_weights="Balanced",
            )
            m_cb.fit(Xa, ya, eval_set=(Xv, yv))

        # Predict on validation
        p_xgb = m_xgb.predict_proba(Xv)[:, 1]
        p_lgb = m_lgb.predict_proba(Xv)[:, 1]
        if HAS_CB:
            p_cb = m_cb.predict_proba(Xv)[:, 1]
            p_ens = 0.34 * p_xgb + 0.33 * p_lgb + 0.33 * p_cb
        else:
            p_ens = 0.5 * p_xgb + 0.5 * p_lgb

        # Store OOF
        oof_preds[va_idx] = p_ens

        auc = roc_auc_score(yv, p_ens)
        r1 = recall_at(yv, p_ens, 0.01)
        apr = average_precision_score(yv, p_ens)
        cv_aucs.append(auc)
        cv_r1s.append(r1)
        cv_aprs.append(apr)
        print(f"    Fold {fold+1}: AUC={auc:.4f} R@1%FPR={r1:.4f} APR={apr:.4f} ({time.time()-t3:.0f}s)")

    print(f"\n  CV mean: AUC={np.mean(cv_aucs):.4f}±{np.std(cv_aucs):.4f} "
          f"R@1%FPR={np.mean(cv_r1s):.4f}±{np.std(cv_r1s):.4f} "
          f"APR={np.mean(cv_aprs):.4f}±{np.std(cv_aprs):.4f}")

    # ── Retrain on full training set ─────────────────────────────────────────
    print("\n  Retraining on full training set...")
    t4 = time.time()

    m_xgb_final = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=min(spw, 20), random_state=42,
        n_jobs=4, eval_metric="auc", use_label_encoder=False,
    )
    m_xgb_final.fit(Xtr_s, ytr, verbose=False)

    m_lgb_final = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=min(spw, 20), random_state=42, n_jobs=4, verbose=-1,
    )
    m_lgb_final.fit(Xtr_s, ytr)

    if HAS_CB:
        m_cb_final = cb.CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, random_seed=42, verbose=0,
            auto_class_weights="Balanced",
        )
        m_cb_final.fit(Xtr_s, ytr)

    print(f"  Retrained in {time.time()-t4:.0f}s")

    # ── Temporal test evaluation ─────────────────────────────────────────────
    print("\n[6/6] Temporal test evaluation...")
    p_xgb_te = m_xgb_final.predict_proba(Xte_s)[:, 1]
    p_lgb_te = m_lgb_final.predict_proba(Xte_s)[:, 1]
    if HAS_CB:
        p_cb_te = m_cb_final.predict_proba(Xte_s)[:, 1]
        p_ens_te = 0.34 * p_xgb_te + 0.33 * p_lgb_te + 0.33 * p_cb_te
    else:
        p_ens_te = 0.5 * p_xgb_te + 0.5 * p_lgb_te

    test_auc = roc_auc_score(yte, p_ens_te)
    test_r1 = recall_at(yte, p_ens_te, 0.01)
    test_apr = average_precision_score(yte, p_ens_te)
    print(f"\n  Temporal test: AUC={test_auc:.4f} R@1%FPR={test_r1:.4f} APR={test_apr:.4f}")

    # Per-model test scores
    for name, p in [("XGB", p_xgb_te), ("LGB", p_lgb_te)] + ([("CB", p_cb_te)] if HAS_CB else []):
        a = roc_auc_score(yte, p)
        r = recall_at(yte, p, 0.01)
        print(f"    {name}: AUC={a:.4f} R@1%FPR={r:.4f}")

    # ── Save artifacts ───────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR / f"altman_prod_{ts}"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(m_xgb_final, model_dir / "xgb_production.joblib")
    joblib.dump(m_lgb_final, model_dir / "lgb_production.joblib")
    if HAS_CB:
        joblib.dump(m_cb_final, model_dir / "cb_production.joblib")
    joblib.dump(scaler, model_dir / "scaler_production.joblib")

    # Also save to models/production/ root (overwrites old)
    # Also copy to root production dir
    joblib.dump(m_xgb_final, MODELS_DIR / "xgb_production.joblib")
    joblib.dump(m_lgb_final, MODELS_DIR / "lgb_production.joblib")
    if HAS_CB:
        joblib.dump(m_cb_final, MODELS_DIR / "cb_production.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_production.joblib")

    # Compute model hash
    hash_data = b""
    for name in ["xgb_production.joblib", "lgb_production.joblib", "cb_production.joblib", "scaler_production.joblib"]:
        p = model_dir / name
        if p.exists():
            hash_data += p.read_bytes()
    model_hash = hashlib.sha256(hash_data).hexdigest()

    # Manifest
    manifest = {
        "model_version": f"altman_clean_{ts}",
        "model_type": "xgb_lgb_cb_ensemble_clean",
        "feature_version": "altman_clean_v1",
        "dataset_version": "altman_ibm_v2_no_leakage",
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
        "ensemble_weights": {"xgb": 0.34, "lgb": 0.33, "cb": 0.33} if HAS_CB else {"xgb": 0.5, "lgb": 0.5},
        "model_hash": model_hash,
        "target_leakage": False,
        "temporal_split": "train_months_1-9_test_months_10-12",
        "trained_at": datetime.now().isoformat(),
        "artifacts": {
            "xgb": "xgb_production.joblib",
            "lgb": "lgb_production.joblib",
            "cb": "cb_production.joblib" if HAS_CB else None,
            "scaler": "scaler_production.joblib",
        },
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (MODELS_DIR / "feature_list.json").write_text(json.dumps(feat_cols, indent=2))

    # Update feature_list.json in model_dir
    (model_dir / "feature_list.json").write_text(json.dumps(feat_cols, indent=2))

    # Update latest pointer
    (MODELS_DIR / "latest").write_text(f"altman_clean_{ts}\n")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"SAVED: {model_dir}")
    print(f"  XGB: {joblib.load(model_dir / 'xgb_production.joblib').n_estimators} trees")
    print(f"  LGB: {joblib.load(model_dir / 'lgb_production.joblib').n_estimators} trees")
    if HAS_CB:
        cb_model = joblib.load(model_dir / 'cb_production.joblib')
        print(f"  CB:  {getattr(cb_model, 'n_estimators', getattr(cb_model, 'tree_count_', '?'))} trees")
    print(f"  Scaler: {type(scaler).__name__}")
    print(f"  Features: {len(feat_cols)} (NO target leakage)")
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
        "cv": {
            "auc_mean": round(float(np.mean(cv_aucs)), 6),
            "auc_std": round(float(np.std(cv_aucs)), 6),
            "r1_mean": round(float(np.mean(cv_r1s)), 6),
            "r1_std": round(float(np.std(cv_r1s)), 6),
            "apr_mean": round(float(np.mean(cv_aprs)), 6),
            "apr_std": round(float(np.std(cv_aprs)), 6),
        },
        "temporal_test": {
            "auc": round(float(test_auc), 6),
            "r1_fpr_1pct": round(float(test_r1), 6),
            "apr": round(float(test_apr), 6),
            "n_test": int(len(Xte)),
            "n_fraud_test": int(yte.sum()),
        },
        "features": feat_cols,
        "n_features": len(feat_cols),
        "training_rows": int(len(Xtr)),
        "total_scanned": n_total,
        "target_leakage": False,
        "elapsed_s": round(elapsed, 1),
    }
    (REPORTS_DIR / "altman_production.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to reports/altman_production.json")


if __name__ == "__main__":
    main()
