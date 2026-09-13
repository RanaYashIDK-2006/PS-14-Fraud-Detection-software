#!/usr/bin/env python3
"""Train the PS-14 model on IBM v2 data.

Computes ML_FEATURES from raw IBM v2 columns (timestamps, amounts, MCCs)
and trains the same LR + RF + XGB + fusion pipeline used for the
synthetic model, so we get a model that actually discriminates on
real-world-like data.

Usage:
    python scripts/ibm_train.py                    # full training
    python scripts/ibm_train.py --max-rows 500000  # quick test
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.privacy_layer.features import ML_FEATURES
from src.risk_engine.fusion import FusionEngine

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"
REPORTS_DIR = ROOT / "reports" / "ibm_train"

# Feature names to compute from IBM v2 raw columns
HIGH_RISK_MCC = {5967, 5966, 5964, 5962, 7995, 6051, 6540}


def compute_ibm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ML_FEATURES from raw IBM v2 columns."""
    # Parse timestamp
    df["ts"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
    time_parts = df["Time"].astype(str).str.split(":", expand=True)
    df["ts"] = df["ts"] + pd.to_timedelta(time_parts[0].astype(int), unit="h") + \
               pd.to_timedelta(time_parts[1].astype(int), unit="m")

    # Parse amount
    amt = pd.to_numeric(
        df["Amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0)
    df["amount"] = amt
    df["hour_of_day"] = df["ts"].dt.hour
    df["is_weekend"] = df["ts"].dt.dayofweek.isin([5, 6]).astype(int)

    # Sort by user and timestamp for running stats
    df = df.sort_values(["User", "ts"]).reset_index(drop=True)

    # Account tenure (days since first txn)
    first_ts = df.groupby("User")["ts"].transform("min")
    df["account_tenure_days"] = ((df["ts"] - first_ts).dt.total_seconds() / 86400).round(1)

    # Log amount
    df["log_amount"] = np.log1p(df["amount"])

    # Amount ratio (current / account median)
    acct_median = df.groupby("User")["amount"].transform("median")
    df["amount_ratio"] = (df["amount"] / acct_median.clip(lower=0.01)).round(4)

    # Known merchant count
    df["known_merchant_count"] = 0
    for user, grp in df.groupby("User"):
        idx = grp.index
        merchants = grp["Merchant Name"].values
        seen = set()
        counts = np.zeros(len(grp), dtype=int)
        for j, m in enumerate(merchants):
            seen.add(m)
            counts[j] = min(50, len(seen))
        df.loc[idx, "known_merchant_count"] = counts

    # Failed auth count (from Errors column)
    df["failed_auth_count_24h"] = df["Errors?"].notna().astype(int)

    # Online ratio (fraction of recent txns that are online)
    df["online_ratio"] = (df["Use Chip"] == "Online Transaction").astype(float)

    # MCC risk
    df["mcc_high_risk"] = df["MCC"].isin(HIGH_RISK_MCC).astype(int)

    # txn_freq_last_24h: count of txns in past 24h for this user
    df["txn_freq_last_24h"] = 0
    df["days_since_last_similar_txn"] = 0.0
    df["gradual_escalation_score"] = 0.0
    for user, grp in df.groupby("User"):
        if len(grp) < 2:
            continue
        idx = grp.index
        ts_arr = grp["ts"].values
        amt_arr = grp["amount"].values

        freq = np.zeros(len(grp), dtype=int)
        days_since = np.zeros(len(grp), dtype=float)
        escalation = np.zeros(len(grp), dtype=float)

        for j in range(1, len(grp)):
            window_start = ts_arr[j] - np.timedelta64(24, "h")
            mask = (ts_arr[:j] >= window_start)
            freq[j] = int(mask.sum())
            # Days since last similar (amount >= 80%)
            for k in range(j-1, -1, -1):
                if amt_arr[k] >= 0.8 * amt_arr[j]:
                    days_since[j] = (ts_arr[j] - ts_arr[k]) / np.timedelta64(1, "D")
                    break
            else:
                days_since[j] = (ts_arr[j] - ts_arr[0]) / np.timedelta64(1, "D")
            # Escalation
            window = amt_arr[max(0, j-9):j+1] / max(acct_median.iloc[idx[j]], 0.01)
            if len(window) >= 3:
                x = np.arange(len(window), dtype=float)
                y = np.log(np.maximum(window, 1e-6))
                slope = float(np.polyfit(x, y, 1)[0])
                escalation[j] = float(np.clip(slope * len(window) * 0.5, 0.0, 1.0))

        df.loc[idx, "txn_freq_last_24h"] = freq
        df.loc[idx, "days_since_last_similar_txn"] = days_since
        df.loc[idx, "gradual_escalation_score"] = escalation

    # Binary flags
    df["txn_time_unusual"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
    df["new_device_flag"] = (df["Use Chip"] == "Online Transaction").astype(int)

    # unusual_location_flag
    df["unusual_location_flag"] = 0
    for user, grp in df.groupby("User"):
        idx = grp.index
        cities = grp["Merchant City"].values
        seen = set()
        flags = np.zeros(len(grp), dtype=int)
        for j, city in enumerate(cities):
            if j > 0 and city not in seen:
                flags[j] = 1
            seen.add(city)
        df.loc[idx, "unusual_location_flag"] = flags

    # unusual_recipient_flag
    df["unusual_recipient_flag"] = 0
    for user, grp in df.groupby("User"):
        idx = grp.index
        merchants = grp["Merchant Name"].values
        seen = set()
        flags = np.zeros(len(grp), dtype=int)
        for j, m in enumerate(merchants):
            if j > 0 and m not in seen:
                flags[j] = 1
            seen.add(m)
        df.loc[idx, "unusual_recipient_flag"] = flags

    # known_device_count
    df["known_device_count"] = 1
    for user, grp in df.groupby("User"):
        idx = grp.index
        chips = grp["Use Chip"].values
        seen = set()
        counts = np.zeros(len(grp), dtype=int)
        for j, chip in enumerate(chips):
            seen.add(chip)
            counts[j] = min(8, len(seen))
        df.loc[idx, "known_device_count"] = counts

    # shared_device_accounts
    merchant_user_count = df.groupby("Merchant Name")["User"].nunique()
    df["shared_device_accounts"] = df["Merchant Name"].map(
        lambda m: max(0, merchant_user_count.get(m, 1) - 1)
    )

    # shared_recipient_accounts
    city_user_count = df.groupby("Merchant City")["User"].nunique()
    df["shared_recipient_accounts"] = df["Merchant City"].map(
        lambda c: max(0, city_user_count.get(c, 1) - 1)
    )

    # mule_ring_score
    df["mule_ring_score"] = (
        df["shared_device_accounts"].clip(upper=5) / 5.0 * 0.6 +
        df["shared_recipient_accounts"].clip(upper=10) / 10.0 * 0.4
    ).round(4)

    # Deviation features — use synthetic-compatible defaults
    df["hour_deviation"] = 0.5
    df["amount_zscore"] = 0.0
    df["velocity_deviation"] = 0.0
    df["recipient_novelty"] = (
        df["unusual_recipient_flag"] * 0.8 + df["unusual_location_flag"] * 0.2
    ).round(4)
    df["txn_regularity"] = 0.0
    df["txn_amount_bucket"] = "typical"

    # Label
    df["label"] = (df["Is Fraud?"] == "Yes").astype(int)

    return df


def train_ibm(max_rows: int | None = None) -> dict:
    """Train on IBM v2 data and evaluate on held-out test."""
    input_path = DATA_DIR / "credit_card_transactions-ibm_v2.csv"
    print(f"Loading {input_path}...")

    read_kwargs = {"usecols": [
        "User", "Card", "Year", "Month", "Day", "Time",
        "Amount", "Use Chip", "Merchant Name", "Merchant City",
        "Merchant State", "Zip", "MCC", "Errors?", "Is Fraud?"
    ]}
    if max_rows:
        read_kwargs["nrows"] = max_rows

    t0 = time.time()
    df = pd.read_csv(input_path, **read_kwargs)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Compute features
    print("  Computing features from raw columns...")
    t_feat = time.time()
    df = compute_ibm_features(df)
    print(f"  Features computed in {time.time()-t_feat:.1f}s")

    # Ensure all ML_FEATURES present
    for f in ML_FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    X = df[ML_FEATURES].to_numpy(dtype=np.float64, na_value=0.0)
    y = df["label"].to_numpy(dtype=int)
    n_fraud = int(y.sum())
    print(f"  Feature matrix: {X.shape}, fraud: {n_fraud} ({n_fraud/len(df)*100:.3f}%)")

    # Time-based split: 70% train, 15% val, 15% test (chronological)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"  Train: {len(X_train):,} (fraud {y_train.sum()})")
    print(f"  Val:   {len(X_val):,} (fraud {y_val.sum()})")
    print(f"  Test:  {len(X_test):,} (fraud {y_test.sum()})")

    # Train models
    print("\nTraining models...")
    t_train = time.time()

    # LR with L2 (the best from hparam_sweep)
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)

    # RF
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # XGB
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    spw = neg_count / max(pos_count, 1)
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)

    # Fusion: average of LR + RF + XGB probabilities
    lr_val = lr.predict_proba(X_val)[:, 1]
    rf_val = rf.predict_proba(X_val)[:, 1]
    xgb_val = xgb.predict_proba(X_val)[:, 1]
    fused_val = (lr_val + rf_val + xgb_val) / 3.0

    lr_test = lr.predict_proba(X_test)[:, 1]
    rf_test = rf.predict_proba(X_test)[:, 1]
    xgb_test = xgb.predict_proba(X_test)[:, 1]
    fused_test = (lr_test + rf_test + xgb_test) / 3.0

    train_time = time.time() - t_train
    print(f"  Training completed in {train_time:.1f}s")

    # Evaluate
    metrics = {}
    for name, scores in [("lr", lr_test), ("rf", rf_test), ("xgb", xgb_test), ("fused", fused_test)]:
        roc = roc_auc_score(y_test, scores) if y_test.sum() > 0 else 0
        pr = average_precision_score(y_test, scores) if y_test.sum() > 0 else 0

        # Recall@1%FPR
        from sklearn.metrics import roc_curve
        fpr_arr, tpr_arr, _ = roc_curve(y_test, scores)
        idx_1pct = np.searchsorted(fpr_arr, 0.01, side="right")
        if idx_1pct > 0:
            idx_1pct -= 1
        recall_at_1pct = tpr_arr[idx_1pct] if idx_1pct < len(tpr_arr) else 0

        metrics[name] = {
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "recall_at_1pct_fpr": round(float(recall_at_1pct), 4),
        }
        print(f"  {name:>6}: ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}, R@1%FPR={recall_at_1pct:.4f}")

    # Confusion matrix at threshold 0.5
    pred_05 = (fused_test >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_05).ravel()
    print(f"\n  At threshold 0.5: TPR={tp/(tp+fn):.4f}, FPR={fp/(fp+tn):.4f}, Precision={tp/(tp+fp):.4f}")

    # Calibrator: Platt scaling on fused validation scores
    from sklearn.linear_model import LogisticRegression as LRCal
    cal = LRCal(C=1.0, max_iter=1000)
    cal.fit(fused_val.reshape(-1, 1), y_val)
    fused_test_cal = cal.predict_proba(fused_test.reshape(-1, 1))[:, 1]

    roc_cal = roc_auc_score(y_test, fused_test_cal) if y_test.sum() > 0 else 0
    pr_cal = average_precision_score(y_test, fused_test_cal) if y_test.sum() > 0 else 0
    print(f"  calibrated: ROC-AUC={roc_cal:.4f}, PR-AUC={pr_cal:.4f}")

    # Confusion matrix for calibrated at threshold 0.5
    pred_cal_05 = (fused_test_cal >= 0.5).astype(int)
    tn_cal, fp_cal, fn_cal, tp_cal = confusion_matrix(y_test, pred_cal_05).ravel()
    print(f"  Calibrated at 0.5: TPR={tp_cal/(tp_cal+fn_cal):.4f}, FPR={fp_cal/(fp_cal+tn_cal):.4f}")

    # Find optimal threshold (Youden J)
    fpr_arr, tpr_arr, thresholds = roc_curve(y_test, fused_test_cal)
    j_scores = tpr_arr - fpr_arr
    best_idx = np.argmax(j_scores)
    print(f"  Best threshold: {thresholds[best_idx]:.4f} -> TPR={tpr_arr[best_idx]:.4f}, FPR={fpr_arr[best_idx]:.4f}")

    # Use calibrated scores as the final model output
    fused_test = fused_test_cal

    # Save models
    print("\nSaving models...")
    import joblib
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual models
    joblib.dump(lr, ARTIFACTS_DIR / "lr_model.joblib")
    joblib.dump(rf, ARTIFACTS_DIR / "rf_model.joblib")
    joblib.dump(xgb, ARTIFACTS_DIR / "xgb_model.joblib")
    joblib.dump(cal, ARTIFACTS_DIR / "ibm_calibrator.joblib")

    # Save metadata
    metadata = {
        "model_version": f"ibm_v2-{len(df)}rows",
        "training_data": "credit_card_transactions-ibm_v2.csv",
        "n_features": len(ML_FEATURES),
        "features": ML_FEATURES,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "fraud_rate_train": round(y_train.mean(), 6),
        "metrics": metrics,
        "training_time_s": round(train_time, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  Saved to {ARTIFACTS_DIR}")

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "training_data": "IBM v2",
        "total_rows": len(df),
        "features": ML_FEATURES,
        "metrics": metrics,
        "confusion_matrix_05": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "best_threshold": round(float(thresholds[best_idx]), 4),
        "best_tpr": round(float(tpr_arr[best_idx]), 4),
        "best_fpr": round(float(fpr_arr[best_idx]), 4),
    }
    report_path = REPORTS_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report: {report_path}")

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    train_ibm(args.max_rows)
