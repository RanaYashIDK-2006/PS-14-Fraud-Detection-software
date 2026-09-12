#!/usr/bin/env python3
"""Retrain PS-14 production model on kartik2112 using exact ML_FEATURES.

Maps kartik2112 (1.85M real-like transactions) → 21 system ML_FEATURES
and trains the LR + RF + XGB ensemble + calibrator.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    brier_score_loss, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))
DATA = ROOT / "data"
ARTIFACTS = ROOT / "models" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
RESULTS = DATA / "real_data_evaluation"
RESULTS.mkdir(exist_ok=True)

# The exact 21 features the privacy layer outputs
ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
    "unusual_location_flag", "unusual_recipient_flag", "failed_auth_count_24h",
    "days_since_last_similar_txn", "gradual_escalation_score", "known_device_count",
    "account_tenure_days", "hour_of_day", "is_weekend", "shared_device_accounts",
    "shared_recipient_accounts", "mule_ring_score", "hour_deviation", "amount_zscore",
    "velocity_deviation", "recipient_novelty", "txn_regularity",
]


def map_kartik_to_ml_features(df: pd.DataFrame) -> np.ndarray:
    """Map kartik2112 columns to the exact 21 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, len(ML_FEATURES)), dtype=float)

    # Parse columns
    amt = pd.to_numeric(df["amt"], errors="coerce").fillna(0).values.astype(float)
    amt_mean = max(np.median(amt), 1)

    trans_time = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    hour = trans_time.dt.hour.fillna(12).values.astype(float)
    dow = trans_time.dt.dayofweek.fillna(0).values.astype(float)

    cc_num = df["cc_num"].values
    category = df["category"].fillna("unknown").values
    city = df["city"].fillna("Unknown").values
    merchant = df["merchant"].fillna("unknown").values

    lat = pd.to_numeric(df["lat"], errors="coerce").fillna(0).values
    lng = pd.to_numeric(df["long"], errors="coerce").fillna(0).values
    merch_lat = pd.to_numeric(df["merch_lat"], errors="coerce").fillna(0).values
    merch_long = pd.to_numeric(df["merch_long"], errors="coerce").fillna(0).values

    # amount_ratio: transaction amount / median amount
    X[:, ML_FEATURES.index("amount_ratio")] = amt / amt_mean

    # txn_freq_last_24h: cumulative transaction count per user / median (proxy)
    user_tx_count = pd.Series(cc_num).groupby(cc_num).cumcount().values
    X[:, ML_FEATURES.index("txn_freq_last_24h")] = np.clip(
        user_tx_count / max(np.median(user_tx_count), 1), 0, 100
    )

    # txn_time_unusual: how unusual the hour is (night=unusual)
    X[:, ML_FEATURES.index("txn_time_unusual")] = ((hour < 6) | (hour > 22)).astype(float)

    # new_device_flag: first transaction from this user-card combo (proxy)
    X[:, ML_FEATURES.index("new_device_flag")] = (user_tx_count < 1).astype(float)

    # unusual_location_flag: geo distance anomaly
    R = 6371
    dlat = np.radians(merch_lat - lat)
    dlon = np.radians(merch_long - lng)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(merch_lat)) * np.sin(dlon/2)**2
    geo_km = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    X[:, ML_FEATURES.index("unusual_location_flag")] = (geo_km > 500).astype(float)

    # unusual_recipient_flag: first time at this merchant
    user_merchant_first = pd.Series(merchant).groupby(cc_num).transform(
        lambda x: (pd.factorize(x)[0] == 0).astype(float)
    ).values
    X[:, ML_FEATURES.index("unusual_recipient_flag")] = user_merchant_first

    # failed_auth_count_24h: 0 (kartik2112 has no auth data)
    X[:, ML_FEATURES.index("failed_auth_count_24h")] = 0

    # days_since_last_similar_txn: cumulative position (proxy for recency)
    X[:, ML_FEATURES.index("days_since_last_similar_txn")] = np.clip(
        np.log1p(user_tx_count), 0, 30
    )

    # gradual_escalation_score: amount increase vs user average
    user_avg_amt = pd.Series(amt).groupby(cc_num).transform("mean").values
    X[:, ML_FEATURES.index("gradual_escalation_score")] = np.clip(
        (amt - user_avg_amt) / np.maximum(user_avg_amt, 1), -2, 10
    )

    # known_device_count: user's transaction count (proxy for known devices)
    X[:, ML_FEATURES.index("known_device_count")] = np.clip(user_tx_count, 0, 50)

    # account_tenure_days: day of year (proxy for account age)
    X[:, ML_FEATURES.index("account_tenure_days")] = trans_time.dt.dayofyear.fillna(150).values

    # hour_of_day
    X[:, ML_FEATURES.index("hour_of_day")] = hour

    # is_weekend
    X[:, ML_FEATURES.index("is_weekend")] = (dow >= 5).astype(float)

    # shared_device_accounts: random noise (no real link data)
    X[:, ML_FEATURES.index("shared_device_accounts")] = np.random.poisson(0.1, n).astype(float)

    # shared_recipient_accounts: random noise
    X[:, ML_FEATURES.index("shared_recipient_accounts")] = np.random.poisson(0.2, n).astype(float)

    # mule_ring_score: random noise (no real link data)
    X[:, ML_FEATURES.index("mule_ring_score")] = np.random.beta(1, 20, n)

    # hour_deviation: deviation from user's typical hour
    user_mean_hour = pd.Series(hour).groupby(cc_num).transform("mean").values
    X[:, ML_FEATURES.index("hour_deviation")] = np.abs(hour - user_mean_hour)

    # amount_zscore
    X[:, ML_FEATURES.index("amount_zscore")] = (amt - np.mean(amt)) / max(np.std(amt), 1e-6)

    # velocity_deviation: deviation from user's median tx count
    user_std_count = pd.Series(user_tx_count).groupby(cc_num).transform("std").fillna(1).values
    X[:, ML_FEATURES.index("velocity_deviation")] = np.clip(
        (user_tx_count - np.median(user_tx_count)) / max(np.median(user_std_count), 1), -5, 50
    )

    # recipient_novelty: first time at this category
    user_cat_first = pd.Series(category).groupby(cc_num).transform(
        lambda x: (pd.factorize(x)[0] == 0).astype(float)
    ).values
    X[:, ML_FEATURES.index("recipient_novelty")] = user_cat_first

    # txn_regularity: coefficient of variation of user's amount
    user_std_amt = pd.Series(amt).groupby(cc_num).transform("std").fillna(1).values
    X[:, ML_FEATURES.index("txn_regularity")] = np.clip(
        user_std_amt / np.maximum(user_avg_amt, 1), 0, 10
    )

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X


def compute_metrics(y_true, y_prob, name=""):
    """Full metrics with bootstrap CI."""
    n = len(y_true)
    n_fraud = int(y_true.sum())
    if n_fraud == 0:
        return {"n": n, "fraud": 0, "roc_auc": None}

    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
    def recall_at(fpr_target):
        idx = np.searchsorted(fpr_arr, fpr_target, side="left")
        return float(tpr_arr[min(idx, len(tpr_arr) - 1)])

    r01, r05, r1 = recall_at(0.001), recall_at(0.005), recall_at(0.01)

    j = tpr_arr - fpr_arr
    best = np.argmax(j)
    threshold = float(thresholds[best]) if best < len(thresholds) else 0.5
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    rng = np.random.RandomState(42)
    roc_cis, pr_cis = [], []
    for _ in range(50):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() > 0 and (1 - yt).sum() > 0:
            roc_cis.append(roc_auc_score(yt, yp))
            pr_cis.append(average_precision_score(yt, yp))

    return {
        "n": n, "fraud": n_fraud, "n_legit": n - n_fraud,
        "prev": round(n_fraud / n, 5),
        "roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
        "recall_at_01pct_fpr": round(r01, 4),
        "recall_at_05pct_fpr": round(r05, 4),
        "recall_at_1pct_fpr": round(r1, 4),
        "brier": round(brier, 4),
        "threshold": round(threshold, 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "ci_roc_auc_95": [
            round(np.percentile(roc_cis, 2.5), 4),
            round(np.percentile(roc_cis, 97.5), 4),
        ] if roc_cis else None,
    }


def main():
    t_total = time.time()

    print("=" * 60)
    print("PS-14 v3 Retrain — kartik2112 → 21 ML_FEATURES")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading kartik2112...")
    t0 = time.time()
    train_df = pd.read_csv(DATA / "kaggle_fraud" / "fraudTrain.csv")
    test_df = pd.read_csv(DATA / "kaggle_fraud" / "fraudTest.csv")
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    # Split: 80% train, 20% validation (from train set)
    split_idx = int(len(train_df) * 0.8)
    train_sub = train_df.iloc[:split_idx]
    val_sub = train_df.iloc[split_idx:]
    print(f"  Split: train={len(train_sub):,}  val={len(val_sub):,}  test={len(test_df):,}")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Map features
    print("\n[2/5] Mapping to 21 ML_FEATURES...")
    t0 = time.time()
    X_train = map_kartik_to_ml_features(train_sub)
    y_train = train_sub["is_fraud"].values.astype(int)
    X_val = map_kartik_to_ml_features(val_sub)
    y_val = val_sub["is_fraud"].values.astype(int)
    X_test = map_kartik_to_ml_features(test_df)
    y_test = test_df["is_fraud"].values.astype(int)
    print(f"  Train: {X_train.shape} fraud={y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.2f}%)")
    print(f"  Test:  {X_test.shape} fraud={y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.2f}%)")
    print(f"  Features mapped in {time.time()-t0:.1f}s")

    # Train
    print("\n[3/5] Training models...")
    t0 = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("  LR...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)
    lr.fit(X_train_s, y_train)

    n_pos = int(y_train.sum())
    spw = (len(y_train) - n_pos) / max(n_pos, 1)
    print(f"  XGBoost (scale_pos_weight={spw:.0f})...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", random_state=42, n_jobs=-1,
        eval_metric="aucpr",
    )
    xgb_model.fit(X_train, y_train, verbose=False)

    print("  RF (100 trees)...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    train_time = time.time() - t0
    print(f"  Trained in {train_time:.1f}s")

    # Validate
    print("\n[4/5] Validation metrics...")
    lr_val = lr.predict_proba(X_val_s)[:, 1]
    xgb_val = xgb_model.predict_proba(X_val)[:, 1]
    rf_val = rf.predict_proba(X_val)[:, 1]
    fused_val = 0.5 * rf_val + 0.3 * xgb_val + 0.2 * lr_val

    for name, preds in [("LR", lr_val), ("XGB", xgb_val), ("RF", rf_val), ("Fused", fused_val)]:
        m = compute_metrics(y_val, preds, name)
        print(f"  {name:6s}: ROC-AUC={m['roc_auc']}  R@1%FPR={m['recall_at_1pct_fpr']}  F1={m['f1']}")

    # Calibrate
    print("\n  Calibrating...")
    from src.risk_engine.calibration import PlattCalibration
    calibrator = PlattCalibration()
    calibrator.fit(fused_val, y_val)
    cal_val = calibrator.predict(fused_val)
    m_cal = compute_metrics(y_val, cal_val, "Calibrated")
    print(f"  Calibrated: ROC-AUC={m_cal['roc_auc']}  R@1%FPR={m_cal['recall_at_1pct_fpr']}  F1={m_cal['f1']}")

    # Test
    print("\n[5/5] Test set evaluation...")
    lr_test = lr.predict_proba(X_test_s)[:, 1]
    xgb_test = xgb_model.predict_proba(X_test)[:, 1]
    rf_test = rf.predict_proba(X_test)[:, 1]
    fused_test = 0.5 * rf_test + 0.3 * xgb_test + 0.2 * lr_test
    cal_test = calibrator.predict(fused_test)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    for name, preds in [("LR", lr_test), ("XGB", xgb_test), ("RF", rf_test),
                         ("Fused", fused_test), ("Calibrated", cal_test)]:
        m = compute_metrics(y_test, preds, name)
        ci = m.get("ci_roc_auc_95")
        ci_str = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        print(f"  {name:12s}: ROC-AUC={m['roc_auc']}{ci_str}  PR-AUC={m['pr_auc']}  "
              f"R@1%={m['recall_at_1pct_fpr']}  F1={m['f1']}")

    # Save artifacts
    print("\n[SAVING] Production artifacts...")
    joblib.dump(scaler, ARTIFACTS / "scaler.joblib")
    joblib.dump(lr, ARTIFACTS / "logistic_regression.joblib")
    joblib.dump(rf, ARTIFACTS / "random_forest.joblib")
    joblib.dump(xgb_model, ARTIFACTS / "xgboost.joblib")
    joblib.dump(calibrator, ARTIFACTS / "calibrator.joblib")
    joblib.dump(fused_val, ARTIFACTS / "fused_val_scores.joblib")
    joblib.dump(y_val, ARTIFACTS / "validation_labels.joblib")

    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    manifest = {
        "model_version": "3.0",
        "feature_version": "v3_21ml_features",
        "training_dataset": "kartik2112 (fraudTrain.csv)",
        "training_rows": int(len(train_sub)),
        "validation_rows": int(len(val_sub)),
        "test_rows": int(len(test_df)),
        "n_features": 21,
        "feature_names": ML_FEATURES,
        "training_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "weights": {"random_forest": 0.50, "xgboost": 0.30, "logistic_regression": 0.20},
        "random_seed": 42,
        "artifact_hashes": {
            "scaler": file_hash(ARTIFACTS / "scaler.joblib"),
            "xgboost": file_hash(ARTIFACTS / "xgboost.joblib"),
            "logistic_regression": file_hash(ARTIFACTS / "logistic_regression.joblib"),
            "random_forest": file_hash(ARTIFACTS / "random_forest.joblib"),
            "calibrator": file_hash(ARTIFACTS / "calibrator.joblib"),
        },
        "test_results": {
            "fused": compute_metrics(y_test, fused_test, "Fused"),
            "calibrated": compute_metrics(y_test, cal_test, "Calibrated"),
        },
    }
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    for f in ["scaler.joblib", "xgboost.joblib", "logistic_regression.joblib",
              "random_forest.joblib", "calibrator.joblib", "manifest.json"]:
        fp = ARTIFACTS / f
        if fp.exists():
            print(f"  {f}: {fp.stat().st_size:,} bytes  hash={manifest['artifact_hashes'].get(f.replace('.joblib',''), '?')}")

    # Save results
    results = {
        "dataset": "kartik2112",
        "n_features": 21,
        "feature_names": ML_FEATURES,
        "training_time_s": round(train_time, 1),
        "test_results": {
            "lr": compute_metrics(y_test, lr_test),
            "xgb": compute_metrics(y_test, xgb_test),
            "rf": compute_metrics(y_test, rf_test),
            "fused": compute_metrics(y_test, fused_test),
            "calibrated": compute_metrics(y_test, cal_test),
        },
        "manifest": manifest,
    }
    (RESULTS / "retrain_v3.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
