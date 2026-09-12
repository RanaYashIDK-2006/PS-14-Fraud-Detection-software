#!/usr/bin/env python3
"""
Pattern Discovery + Training Pipeline for IBM Altman (24M rows).

Phase 1: Deep pattern discovery on ULB creditcard (exhaustive search)
Phase 2: Feature engineering on IBM Altman (richer source data)
Phase 3: Train + evaluate on both datasets
Phase 4: Cross-domain evaluation
"""

import numpy as np
import pandas as pd
import hashlib
import json
import time
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, roc_curve, brier_score_loss, f1_score
)
from scipy import stats as sp_stats
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

DROP_COLS_ULB = {
    "label", "new_device_flag", "unusual_location_flag", "failed_auth_count_24h",
    "known_device_count", "shared_device_accounts", "shared_recipient_accounts",
    "mule_ring_score", "is_weekend", "amount_v_correlation", "days_since_last_similar_txn",
}


def hash_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def bootstrap_ci(y_true, y_score, fn, n_boot=500, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(fn(y_true[idx], y_score[idx]))
    if not scores:
        return (float("nan"), float("nan"))
    alpha = (1 - ci) / 2
    return (np.percentile(scores, alpha * 100), np.percentile(scores, (1 - alpha) * 100))


def recall_at_fpr(y_true, y_score, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr <= fpr_target
    if not valid.any():
        return 0.0
    return float(tpr[valid][-1])


def recall_at_fprs(y_true, y_score):
    """Compute recall at multiple FPR thresholds."""
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    results = {}
    labels = {0.001: "0_1pct", 0.005: "0_5pct", 0.01: "1pct", 0.05: "5pct"}
    for target, label in labels.items():
        valid = fpr_arr <= target
        results[f"recall_at_{label}_fpr"] = float(tpr_arr[valid][-1]) if valid.any() else 0.0
    return results


# ============================================================
# PHASE 1: Exhaustive Pattern Discovery on ULB creditcard
# ============================================================
def phase1_pattern_discovery():
    print("\n" + "=" * 70)
    print("PHASE 1: EXHAUSTIVE PATTERN DISCOVERY ON ULB CREDITCARD")
    print("=" * 70)

    df = pd.read_csv(DATA_DIR / "transactions_v2.csv")
    feature_cols = [c for c in df.columns if c not in DROP_COLS_ULB]
    X = np.nan_to_num(df[feature_cols].values.astype(np.float32), nan=0, posinf=100, neginf=-100)
    y = df["label"].values.astype(int)

    print(f"Dataset: {len(df):,} rows, {y.sum()} fraud ({y.mean()*100:.3f}%)")
    print(f"Features: {len(feature_cols)}")

    # Test many model configurations
    configs = [
        {"name": "LR_balanced", "model": LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)},
        {"name": "LR_C10", "model": LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)},
        {"name": "RF_200d15", "model": RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1)},
        {"name": "RF_500d20", "model": RandomForestClassifier(n_estimators=500, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1)},
        {"name": "GB_100", "model": GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)},
        {"name": "GB_300", "model": GradientBoostingClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, random_state=42)},
    ]

    # Try XGBoost if available
    try:
        from xgboost import XGBClassifier
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        spw = n_neg / max(n_pos, 1)
        configs.extend([
            {"name": "XGB_300d8", "model": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, random_state=42, eval_metric="aucpr", use_label_encoder=False, n_jobs=-1)},
            {"name": "XGB_500d6", "model": XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8, scale_pos_weight=spw, random_state=42, eval_metric="aucpr", use_label_encoder=False, n_jobs=-1)},
            {"name": "XGB_1000d5", "model": XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.03, subsample=0.9, colsample_bytree=0.7, scale_pos_weight=spw, min_child_weight=5, random_state=42, eval_metric="aucpr", use_label_encoder=False, n_jobs=-1)},
        ])
    except ImportError:
        print("  XGBoost not available")

    # 5-fold CV for each
    print(f"\nTesting {len(configs)} model configurations with 5-fold CV...")
    print(f"{'Config':<20} {'AUC':>8} {'PR-AUC':>8} {'R@1%':>8} {'R@0.5%':>8} {'R@0.1%':>8}")
    print("-" * 62)

    best_auc = 0
    best_config = None

    for cfg in configs:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs, prs, r1s, r05s, r01s = [], [], [], [], []

        for tr, va in skf.split(X, y):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xva = scaler.transform(X[va])

            model = cfg["model"]
            model.fit(Xtr, y[tr])
            p = model.predict_proba(Xva)[:, 1]

            aucs.append(roc_auc_score(y[va], p))
            prs.append(average_precision_score(y[va], p))
            rfprs = recall_at_fprs(y[va], p)
            r1s.append(rfprs.get("recall_at_1pct_fpr", 0))
            r05s.append(rfprs.get("recall_at_0_5pct_fpr", 0))
            r01s.append(rfprs.get("recall_at_0_1pct_fpr", 0))

        mean_auc = np.mean(aucs)
        print(f"  {cfg['name']:<18} {mean_auc:.4f}  {np.mean(prs):.4f}  {np.mean(r1s):.4f}  {np.mean(r05s):.4f}  {np.mean(r01s):.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_config = cfg

    print(f"\n  Best: {best_config['name']} (AUC={best_auc:.4f})")

    # Feature importance from best tree model
    print("\n--- Feature Importance (from best tree model) ---")
    tree_models = [c for c in configs if "RF" in c["name"] or "GB" in c["name"] or "XGB" in c["name"]]
    if tree_models:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        tree_models[0]["model"].fit(X, y)
        imp = tree_models[0]["model"].feature_importances_
        idx = np.argsort(imp)[::-1]
        for i in idx[:20]:
            print(f"  {i+1:2d}. {feature_cols[i]:30s} {imp[i]:.4f}")

    return best_config, feature_cols


# ============================================================
# PHASE 2: Feature Engineering on IBM Altman (24M rows)
# ============================================================
def phase2_altman_features():
    print("\n" + "=" * 70)
    print("PHASE 2: IBM ALTMAN FEATURE ENGINEERING (24M rows)")
    print("=" * 70)

    print("Loading dataset (this takes a moment for 24M rows)...")
    t0 = time.time()
    # Read in chunks for memory efficiency
    chunks = []
    for chunk in pd.read_csv(
        DATA_DIR / "credit_card_transactions-ibm_v2.csv",
        usecols=["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
                  "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC", "Errors?", "Is Fraud?"],
        chunksize=500_000,
        low_memory=False,
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    load_time = time.time() - t0
    print(f"  Loaded {len(df):,} rows in {load_time:.0f}s")

    # Parse amount
    df["amt"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["is_fraud"] = (df["Is Fraud?"] == "Yes").astype(int)

    # Create datetime
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]].assign(
        hour=df["Time"].str.split(":").str[0].astype(int),
        minute=df["Time"].str.split(":").str[1].astype(int),
    ))

    # Sort by user and datetime for temporal features
    print("  Sorting by user and datetime...")
    df = df.sort_values(["User", "datetime"]).reset_index(drop=True)

    print(f"  Users: {df['User'].nunique()}, Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.3f}%)")

    # --- Feature Engineering ---
    print("  Engineering features...")

    features = pd.DataFrame(index=df.index)

    # 1. Amount features
    features["amount"] = df["amt"]
    features["amount_log"] = np.log1p(np.abs(df["amt"]))
    features["amount_sign"] = np.sign(df["amt"])
    features["is_negative_amount"] = (df["amt"] < 0).astype(np.float32)

    # Per-user amount statistics (cumulative, point-in-time correct)
    print("    Computing per-user cumulative amount stats...")
    grp = df.groupby("User")
    features["user_amt_mean_cum"] = grp["amt"].transform(lambda x: x.expanding().mean())
    features["user_amt_std_cum"] = grp["amt"].transform(lambda x: x.expanding().std().fillna(0))
    features["user_amt_max_cum"] = grp["amt"].transform(lambda x: x.expanding().max())
    features["user_amt_median_cum"] = grp["amt"].transform(lambda x: x.expanding().median())
    # Z-score relative to user history
    features["user_amt_zscore"] = (features["amount"] - features["user_amt_mean_cum"]) / (features["user_amt_std_cum"] + 1e-8)
    # Amount relative to user median
    features["user_amt_ratio"] = features["amount"] / (features["user_amt_median_cum"].clip(lower=1))
    # Amount percentile within user history
    features["user_amt_pctile"] = grp["amt"].transform(lambda x: x.expanding().rank(pct=True))

    # 2. Temporal features
    print("    Computing temporal features...")
    features["hour"] = df["datetime"].dt.hour.astype(np.float32)
    features["dow"] = df["datetime"].dt.dayofweek.astype(np.float32)
    features["is_weekend"] = (features["dow"] >= 5).astype(np.float32)
    features["is_night"] = ((features["hour"] >= 22) | (features["hour"] <= 5)).astype(np.float32)
    features["is_business_hours"] = ((features["hour"] >= 9) & (features["hour"] <= 17)).astype(np.float32)

    # Time since last transaction (per user)
    print("    Computing time gaps...")
    features["hours_since_last"] = df.groupby("User")["datetime"].diff().dt.total_seconds() / 3600.0
    features["hours_since_last"] = features["hours_since_last"].fillna(0).clip(upper=720)

    # 3. Velocity features (per user, point-in-time)
    print("    Computing velocity features...")
    features["user_txn_count_cum"] = grp.cumcount() + 1
    features["user_txn_freq_24h"] = grp["datetime"].transform(
        lambda x: x.rolling("24h", min_periods=1).count()
    )
    features["user_txn_freq_7d"] = grp["datetime"].transform(
        lambda x: x.rolling("7D", min_periods=1).count()
    )

    # 4. MCC features
    print("    Computing MCC features...")
    features["mcc"] = df["MCC"].astype(np.float32)

    # Global MCC fraud rate (point-in-time: only from prior transactions)
    # Use expanding fraud rate per MCC
    features["mcc_fraud_rate_cum"] = df.groupby("MCC")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    )

    # 5. Merchant features
    print("    Computing merchant features...")
    features["merchant_hash"] = df["Merchant Name"].astype("category").cat.codes.astype(np.float32)

    # Merchant fraud rate (point-in-time)
    features["merchant_fraud_rate_cum"] = df.groupby("Merchant Name")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    )

    # 6. City/State features
    print("    Computing location features...")
    features["city_hash"] = df["Merchant City"].astype("category").cat.codes.astype(np.float32)
    features["state_hash"] = df["Merchant State"].fillna("UNK").astype("category").cat.codes.astype(np.float32)
    features["has_zip"] = df["Zip"].notna().astype(np.float32)

    # 7. Card features
    features["card_num"] = df["Card"].astype(np.float32)

    # 8. Use Chip features
    print("    Computing chip/swipe features...")
    chip_map = {"Swipe Transaction": 0, "Chip Transaction": 1, "Online Transaction": 2}
    features["chip_type"] = df["Use Chip"].map(chip_map).fillna(3).astype(np.float32)

    # 9. Error features
    features["has_error"] = df["Errors?"].notna().astype(np.float32)

    # 10. User behavioral features (point-in-time)
    print("    Computing behavioral features...")
    features["user_unique_merchants_cum"] = grp["Merchant Name"].transform(
        lambda x: x.expanding().apply(lambda v: len(set(v)), raw=True)
    )
    features["user_unique_cities_cum"] = grp["Merchant City"].transform(
        lambda x: x.expanding().apply(lambda v: len(set(v)), raw=True)
    )
    features["user_unique_mccs_cum"] = grp["MCC"].transform(
        lambda x: x.expanding().apply(lambda v: len(set(v)), raw=True)
    )

    # 11. Escalation pattern: compare last 5 txns to prior 5
    print("    Computing escalation patterns...")
    features["escalation_score"] = grp["amt"].transform(
        lambda x: x.rolling(10, min_periods=2).apply(
            lambda v: (np.median(v[len(v)//2:]) - np.median(v[:len(v)//2])) / (np.median(v[:len(v)//2]) + 1e-8) if len(v) >= 2 else 0,
            raw=True
        )
    ).fillna(0).clip(-5, 5)

    # 12. Amount-velocity interaction
    features["amt_x_freq"] = features["amount_log"] * np.log1p(features["user_txn_freq_24h"])

    # 13. Night amount (high risk)
    features["night_amount"] = features["is_night"] * features["amount_log"]

    # 14. New merchant (never seen before for this user)
    features["is_new_merchant"] = (features["user_unique_merchants_cum"] <= 1).astype(np.float32)

    # 15. New city (never seen before for this user)
    features["is_new_city"] = (features["user_unique_cities_cum"] <= 1).astype(np.float32)

    # 16. Transaction regularity (coefficient of variation of inter-txn times)
    features["txn_regularity"] = grp["datetime"].transform(
        lambda x: x.diff().dt.total_seconds().expanding().std().fillna(0)
    )
    features["txn_regularity_cv"] = features["txn_regularity"] / (features["hours_since_last"] + 1e-8)

    # Clean up
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Add label
    features["label"] = df["is_fraud"].values

    elapsed = time.time() - t0
    print(f"\n  Feature matrix: {features.shape[0]:,} rows x {features.shape[1]} features")
    print(f"  Total time: {elapsed:.0f}s")
    print(f"\n  Feature columns: {list(features.columns)}")

    # Save in chunks
    out_path = DATA_DIR / "altman_features.csv"
    print(f"  Saving to {out_path}...")
    features.to_csv(out_path, index=False)
    print(f"  Saved ({os.path.getsize(out_path)/1e6:.0f}MB)")

    return features


# ============================================================
# PHASE 3: Train + Evaluate on both datasets
# ============================================================
def phase3_train_evaluate(altman_features=None):
    print("\n" + "=" * 70)
    print("PHASE 3: TRAIN + EVALUATE")
    print("=" * 70)

    results = {}

    # --- A. ULB Creditcard (full retrain with exhaustive search) ---
    print("\n--- A. ULB Creditcard (284K rows) ---")
    df_ulb = pd.read_csv(DATA_DIR / "transactions_v2.csv")
    feat_cols_ulb = [c for c in df_ulb.columns if c not in DROP_COLS_ULB]
    X_ulb = np.nan_to_num(df_ulb[feat_cols_ulb].values.astype(np.float32), nan=0, posinf=100, neginf=-100)
    y_ulb = df_ulb["label"].values

    # Stratified split
    Xtr_u, Xte_u, ytr_u, yte_u = train_test_split(X_ulb, y_ulb, test_size=0.2, random_state=42, stratify=y_ulb)
    scaler_u = StandardScaler()
    Xtr_us = scaler_u.fit_transform(Xtr_u)
    Xte_us = scaler_u.transform(Xte_u)

    try:
        from xgboost import XGBClassifier
        spw = (ytr_u == 0).sum() / max((ytr_u == 1).sum(), 1)
        xgb_ulb = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.8, scale_pos_weight=spw, min_child_weight=5,
            random_state=42, eval_metric="aucpr", use_label_encoder=False, n_jobs=-1
        )
        xgb_ulb.fit(Xtr_u, ytr_u)
        p_xgb = xgb_ulb.predict_proba(Xte_u)[:, 1]
    except:
        p_xgb = None

    # RF
    rf_ulb = RandomForestClassifier(n_estimators=500, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_ulb.fit(Xtr_u, ytr_u)
    p_rf = rf_ulb.predict_proba(Xte_u)[:, 1]

    # LR
    lr_ulb = LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)
    lr_ulb.fit(Xtr_us, ytr_u)
    p_lr = lr_ulb.predict_proba(Xte_us)[:, 1]

    # Ensemble
    if p_xgb is not None:
        meta = np.column_stack([p_lr, p_rf, p_xgb])
    else:
        meta = np.column_stack([p_lr, p_rf])

    # Split meta for stacking
    from sklearn.model_selection import cross_val_predict
    meta_cv = np.zeros((len(ytr_u), meta.shape[1]))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for j in range(meta.shape[1]):
        lr_meta = LogisticRegression(C=10, max_iter=1000, random_state=42)
        meta_cv[:, j] = cross_val_predict(lr_meta, Xtr_u if j > 0 else Xtr_us, ytr_u, cv=skf, method="predict_proba")[:, 1]

    meta_lr = LogisticRegression(C=10, max_iter=1000, random_state=42)
    meta_lr.fit(meta_cv, ytr_u)

    meta_test = np.column_stack([p_lr, p_rf, p_xgb]) if p_xgb is not None else np.column_stack([p_lr, p_rf])
    p_stack = meta_lr.predict_proba(meta_test)[:, 1]

    # Evaluate all
    for name, p in [("LR", p_lr), ("RF", p_rf), ("XGB", p_xgb), ("Stacker", p_stack)]:
        if p is None:
            continue
        auc = roc_auc_score(yte_u, p)
        pr = average_precision_score(yte_u, p)
        brier = brier_score_loss(yte_u, p)
        rfprs = recall_at_fprs(yte_u, p)
        ci = bootstrap_ci(yte_u, p, roc_auc_score)
        print(f"  {name:<10} AUC={auc:.4f}  PR-AUC={pr:.4f}  R@1%={rfprs.get("recall_at_1pct_fpr", 0):.4f}  R@0.5%={rfprs.get("recall_at_0_5pct_fpr", 0):.4f}  CI=[{ci[0]:.4f},{ci[1]:.4f}]")
        results[f"ulb_{name.lower()}"] = {
            "roc_auc": auc, "pr_auc": pr, "brier": brier, **rfprs,
            "ci_95": list(ci), "n_test": int(len(yte_u)), "n_fraud_test": int(yte_u.sum())
        }

    # --- B. IBM Altman (24M rows) ---
    print("\n--- B. IBM Altman (24M rows) ---")

    if altman_features is None:
        altman_path = DATA_DIR / "altman_features.csv"
        if altman_path.exists():
            print("  Loading pre-computed features...")
            altman_features = pd.read_csv(altman_path)
        else:
            altman_features = phase2_altman_features()

    feat_cols_a = [c for c in altman_features.columns if c not in ("label", "amount")]
    X_a = np.nan_to_num(altman_features[feat_cols_a].values.astype(np.float32), nan=0, posinf=100, neginf=-100)
    y_a = altman_features["label"].values

    print(f"  Rows: {len(X_a):,}, Fraud: {y_a.sum():,} ({y_a.mean()*100:.3f}%)")

    # Subsample for training speed (stratified)
    if len(X_a) > 1_000_000:
        print("  Subsampling 1M rows for training (stratified)...")
        idx_sub = np.random.RandomState(42).choice(
            len(X_a), size=1_000_000, replace=False,
            p=np.where(y_a == 1, y_a.sum()/len(X_a) * 10 / len(X_a), (1 - y_a.sum()/len(X_a)) * 10 / len(X_a))
            if y_a.sum() > 0 else None
        )
        # Simple stratified sample
        fraud_idx = np.where(y_a == 1)[0]
        legit_idx = np.where(y_a == 0)[0]
        n_fraud_sub = min(len(fraud_idx), 5000)
        n_legit_sub = 1_000_000 - n_fraud_sub
        sub_fraud = np.random.RandomState(42).choice(fraud_idx, n_fraud_sub, replace=False)
        sub_legit = np.random.RandomState(42).choice(legit_idx, n_legit_sub, replace=False)
        sub_idx = np.concatenate([sub_fraud, sub_legit])
        np.random.RandomState(42).shuffle(sub_idx)
        X_a_sub = X_a[sub_idx]
        y_a_sub = y_a[sub_idx]
    else:
        X_a_sub = X_a
        y_a_sub = y_a

    Xtr_a, Xte_a, ytr_a, yte_a = train_test_split(X_a_sub, y_a_sub, test_size=0.2, random_state=42, stratify=y_a_sub)
    scaler_a = StandardScaler()
    Xtr_as = scaler_a.fit_transform(Xtr_a)
    Xte_as = scaler_a.transform(Xte_a)

    print(f"  Train: {len(Xtr_a):,} ({ytr_a.sum()} fraud), Test: {len(Xte_a):,} ({yte_a.sum()} fraud)")

    try:
        from xgboost import XGBClassifier
        spw_a = (ytr_a == 0).sum() / max((ytr_a == 1).sum(), 1)
        xgb_a = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.8, scale_pos_weight=spw_a, min_child_weight=5,
            random_state=42, eval_metric="aucpr", use_label_encoder=False, n_jobs=-1
        )
        t0 = time.time()
        xgb_a.fit(Xtr_a, ytr_a, eval_set=[(Xte_a, yte_a)], verbose=False)
        t_xgb = time.time() - t0
        p_xgb_a = xgb_a.predict_proba(Xte_a)[:, 1]
        print(f"  XGB trained in {t_xgb:.0f}s")
    except Exception as e:
        print(f"  XGB failed: {e}")
        p_xgb_a = None

    rf_a = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1)
    t0 = time.time()
    rf_a.fit(Xtr_a, ytr_a)
    t_rf = time.time() - t0
    p_rf_a = rf_a.predict_proba(Xte_a)[:, 1]
    print(f"  RF trained in {t_rf:.0f}s")

    lr_a = LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)
    t0 = time.time()
    lr_a.fit(Xtr_as, ytr_a)
    t_lr = time.time() - t0
    p_lr_a = lr_a.predict_proba(Xte_as)[:, 1]
    print(f"  LR trained in {t_lr:.0f}s")

    # Stacker
    if p_xgb_a is not None:
        meta_a = np.column_stack([p_lr_a, p_rf_a, p_xgb_a])
    else:
        meta_a = np.column_stack([p_lr_a, p_rf_a])

    meta_cv_a = np.zeros((len(ytr_a), meta_a.shape[1]))
    for j in range(meta_a.shape[1]):
        lr_m = LogisticRegression(C=10, max_iter=1000, random_state=42)
        meta_cv_a[:, j] = cross_val_predict(lr_m, Xtr_as if j == 0 else Xtr_a, ytr_a, cv=skf, method="predict_proba")[:, 1]

    meta_lr_a = LogisticRegression(C=10, max_iter=1000, random_state=42)
    meta_lr_a.fit(meta_cv_a, ytr_a)
    meta_test_a = np.column_stack([p_lr_a, p_rf_a, p_xgb_a]) if p_xgb_a is not None else np.column_stack([p_lr_a, p_rf_a])
    p_stack_a = meta_lr_a.predict_proba(meta_test_a)[:, 1]

    for name, p in [("LR", p_lr_a), ("RF", p_rf_a), ("XGB", p_xgb_a), ("Stacker", p_stack_a)]:
        if p is None:
            continue
        auc = roc_auc_score(yte_a, p)
        pr = average_precision_score(yte_a, p)
        brier = brier_score_loss(yte_a, p)
        rfprs = recall_at_fprs(yte_a, p)
        ci = bootstrap_ci(yte_a, p, roc_auc_score, n_boot=200)
        print(f"  {name:<10} AUC={auc:.4f}  PR-AUC={pr:.4f}  R@1%={rfprs.get("recall_at_1pct_fpr", 0):.4f}  R@0.5%={rfprs.get("recall_at_0_5pct_fpr", 0):.4f}  CI=[{ci[0]:.4f},{ci[1]:.4f}]")
        results[f"altman_{name.lower()}"] = {
            "roc_auc": auc, "pr_auc": pr, "brier": brier, **rfprs,
            "ci_95": list(ci), "n_test": int(len(yte_a)), "n_fraud_test": int(yte_a.sum())
        }

    # Feature importance
    print("\n  Top 15 features (Altman XGB):")
    if p_xgb_a is not None:
        imp = xgb_a.feature_importances_
        idx = np.argsort(imp)[::-1][:15]
        for i, j in enumerate(idx):
            print(f"    {i+1:2d}. {feat_cols_a[j]:35s} {imp[j]:.4f}")

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import os

    # Phase 1: Pattern discovery
    best_cfg, feat_cols = phase1_pattern_discovery()

    # Phase 2+3: Altman features + full evaluation
    results = phase3_train_evaluate()

    # Save all results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase1_best_config": best_cfg["name"] if best_cfg else None,
        "results": results,
    }
    out_path = REPORTS_DIR / "pattern_discovery_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — BRUTALLY TRUE RESULTS")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Model':<12} {'ROC-AUC':>10} {'PR-AUC':>10} {'R@1%FPR':>10} {'R@0.5%FPR':>10} {'Brier':>10}")
    print("-" * 82)
    for k, v in sorted(results.items()):
        ds, model = k.rsplit("_", 1)
        print(f"  {ds:<18} {model:<12} {v['roc_auc']:>10.4f} {v['pr_auc']:>10.4f} {v['recall_at_1pct_fpr']:>10.4f} {v['recall_at_0_5pct_fpr']:>10.4f} {v['brier']:>10.6f}")
