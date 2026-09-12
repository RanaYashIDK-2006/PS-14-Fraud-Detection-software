#!/usr/bin/env python3
"""Multi-dataset evaluation: train on kartik2112, test on all available datasets.

Imports all available datasets, maps them to the 21 ML_FEATURES,
and evaluates cross-dataset generalization.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DATA = ROOT / "data"
RESULTS = DATA / "real_data_evaluation"
RESULTS.mkdir(exist_ok=True)

# The 21 ML_FEATURES the system outputs
ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
    "unusual_location_flag", "unusual_recipient_flag", "failed_auth_count_24h",
    "days_since_last_similar_txn", "gradual_escalation_score", "known_device_count",
    "account_tenure_days", "hour_of_day", "is_weekend", "shared_device_accounts",
    "shared_recipient_accounts", "mule_ring_score", "hour_deviation", "amount_zscore",
    "velocity_deviation", "recipient_novelty", "txn_regularity",
]


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

    rng = np.random.RandomState(42)
    roc_cis = []
    for _ in range(50):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() > 0 and (1 - yt).sum() > 0:
            roc_cis.append(roc_auc_score(yt, yp))

    return {
        "n": n, "fraud": n_fraud, "prev": round(n_fraud / n, 5),
        "roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
        "recall_at_1pct_fpr": round(r1, 4),
        "brier": round(brier, 4),
        "ci_roc_auc_95": [
            round(np.percentile(roc_cis, 2.5), 4),
            round(np.percentile(roc_cis, 97.5), 4),
        ] if roc_cis else None,
    }


# --- Dataset Mappers ---

def map_kartik2112(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Map kartik2112 (fraudTrain/fraudTest) to 21 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, len(ML_FEATURES)), dtype=float)

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

    user_tx_count = pd.Series(cc_num).groupby(cc_num).cumcount().values
    user_avg_amt = pd.Series(amt).groupby(cc_num).transform("mean").values
    user_std_amt = pd.Series(amt).groupby(cc_num).transform("std").fillna(1).values

    X[:, ML_FEATURES.index("amount_ratio")] = amt / amt_mean
    X[:, ML_FEATURES.index("txn_freq_last_24h")] = np.clip(user_tx_count / max(np.median(user_tx_count), 1), 0, 100)
    X[:, ML_FEATURES.index("txn_time_unusual")] = ((hour < 6) | (hour > 22)).astype(float)
    X[:, ML_FEATURES.index("new_device_flag")] = (user_tx_count < 1).astype(float)

    R = 6371
    dlat = np.radians(merch_lat - lat)
    dlon = np.radians(merch_long - lng)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(merch_lat)) * np.sin(dlon/2)**2
    geo_km = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    X[:, ML_FEATURES.index("unusual_location_flag")] = (geo_km > 500).astype(float)

    user_merchant_first = pd.Series(merchant).groupby(cc_num).transform(
        lambda x: (pd.factorize(x)[0] == 0).astype(float)
    ).values
    X[:, ML_FEATURES.index("unusual_recipient_flag")] = user_merchant_first
    X[:, ML_FEATURES.index("failed_auth_count_24h")] = 0
    X[:, ML_FEATURES.index("days_since_last_similar_txn")] = np.clip(np.log1p(user_tx_count), 0, 30)
    X[:, ML_FEATURES.index("gradual_escalation_score")] = np.clip((amt - user_avg_amt) / np.maximum(user_avg_amt, 1), -2, 10)
    X[:, ML_FEATURES.index("known_device_count")] = np.clip(user_tx_count, 0, 50)
    X[:, ML_FEATURES.index("account_tenure_days")] = trans_time.dt.dayofyear.fillna(150).values
    X[:, ML_FEATURES.index("hour_of_day")] = hour
    X[:, ML_FEATURES.index("is_weekend")] = (dow >= 5).astype(float)
    X[:, ML_FEATURES.index("shared_device_accounts")] = np.random.poisson(0.1, n).astype(float)
    X[:, ML_FEATURES.index("shared_recipient_accounts")] = np.random.poisson(0.2, n).astype(float)
    X[:, ML_FEATURES.index("mule_ring_score")] = np.random.beta(1, 20, n)
    user_mean_hour = pd.Series(hour).groupby(cc_num).transform("mean").values
    X[:, ML_FEATURES.index("hour_deviation")] = np.abs(hour - user_mean_hour)
    X[:, ML_FEATURES.index("amount_zscore")] = (amt - np.mean(amt)) / max(np.std(amt), 1e-6)
    user_std_count = pd.Series(user_tx_count).groupby(cc_num).transform("std").fillna(1).values
    X[:, ML_FEATURES.index("velocity_deviation")] = np.clip((user_tx_count - np.median(user_tx_count)) / max(np.median(user_std_count), 1), -5, 50)
    user_cat_first = pd.Series(category).groupby(cc_num).transform(
        lambda x: (pd.factorize(x)[0] == 0).astype(float)
    ).values
    X[:, ML_FEATURES.index("recipient_novelty")] = user_cat_first
    X[:, ML_FEATURES.index("txn_regularity")] = np.clip(user_std_amt / np.maximum(user_avg_amt, 1), 0, 10)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["is_fraud"].values.astype(int)
    return X, y, ML_FEATURES


def map_ulb_creditcard(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Map ULB Credit Card (V1-V28 PCA) to 21 ML_FEATURES via aggregate stats."""
    n = len(df)
    X = np.zeros((n, len(ML_FEATURES)), dtype=float)

    amt = df["Amount"].values.astype(float)
    amt_mean = max(np.median(amt), 1)
    time_sec = df["Time"].values.astype(float)

    hour = ((time_sec % 86400) / 3600).astype(int)
    X[:, ML_FEATURES.index("amount_ratio")] = amt / amt_mean
    X[:, ML_FEATURES.index("amount_zscore")] = (amt - np.mean(amt)) / max(np.std(amt), 1e-6)
    X[:, ML_FEATURES.index("hour_of_day")] = hour
    X[:, ML_FEATURES.index("is_weekend")] = 0  # unknown

    # PCA components as aggregate stats
    v_cols = [c for c in df.columns if c.startswith("V")]
    if v_cols:
        v_data = df[v_cols].values
        X[:, ML_FEATURES.index("txn_freq_last_24h")] = np.clip(np.mean(np.abs(v_data), axis=1) * 10, 0, 100)
        X[:, ML_FEATURES.index("txn_time_unusual")] = (np.abs(v_data[:, 0]) > 2).astype(float)
        X[:, ML_FEATURES.index("new_device_flag")] = (np.abs(v_data[:, 1]) > 2).astype(float)
        X[:, ML_FEATURES.index("unusual_location_flag")] = (np.abs(v_data[:, 2]) > 2).astype(float)
        X[:, ML_FEATURES.index("unusual_recipient_flag")] = (np.abs(v_data[:, 3]) > 2).astype(float)
        X[:, ML_FEATURES.index("failed_auth_count_24h")] = 0
        X[:, ML_FEATURES.index("days_since_last_similar_txn")] = np.clip(np.abs(v_data[:, 4]), 0, 30)
        X[:, ML_FEATURES.index("gradual_escalation_score")] = np.clip(np.abs(v_data[:, 5]) / 5, 0, 1)
        X[:, ML_FEATURES.index("known_device_count")] = 1
        X[:, ML_FEATURES.index("account_tenure_days")] = time_sec / 86400
        X[:, ML_FEATURES.index("hour_deviation")] = np.abs(hour - 14)
        X[:, ML_FEATURES.index("velocity_deviation")] = np.clip(np.abs(v_data[:, 6]), -5, 50)
        X[:, ML_FEATURES.index("recipient_novelty")] = (np.abs(v_data[:, 7]) > 1.5).astype(float)
        X[:, ML_FEATURES.index("txn_regularity")] = np.clip(np.std(v_data, axis=1), 0, 10)

    X[:, ML_FEATURES.index("shared_device_accounts")] = 0
    X[:, ML_FEATURES.index("shared_recipient_accounts")] = 0
    X[:, ML_FEATURES.index("mule_ring_score")] = 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["Class"].values.astype(int)
    return X, y, ML_FEATURES


def map_paysim(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Map PaySim to 21 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, len(ML_FEATURES)), dtype=float)

    amt = df["amount"].values.astype(float)
    amt_mean = max(np.median(amt), 1)
    step = df.index.values.astype(float) if "step" not in df.columns else df["step"].values.astype(float)
    hour = (step % 24).astype(int)
    tx_type = df["type"].values
    old_balance = pd.to_numeric(df["oldbalanceOrg"], errors="coerce").fillna(0).values
    new_balance = pd.to_numeric(df["newbalanceOrig"], errors="coerce").fillna(0).values

    X[:, ML_FEATURES.index("amount_ratio")] = amt / max(amt_mean, 1)
    X[:, ML_FEATURES.index("amount_zscore")] = (amt - np.mean(amt)) / max(np.std(amt), 1e-6)
    X[:, ML_FEATURES.index("hour_of_day")] = hour
    X[:, ML_FEATURES.index("is_weekend")] = ((step // 24) % 7 >= 5).astype(float)
    X[:, ML_FEATURES.index("txn_time_unusual")] = ((hour < 6) | (hour > 22)).astype(float)
    X[:, ML_FEATURES.index("new_device_flag")] = (tx_type == "CASH_OUT").astype(float)
    X[:, ML_FEATURES.index("unusual_location_flag")] = (amt > 500000).astype(float)
    X[:, ML_FEATURES.index("unusual_recipient_flag")] = (tx_type == "TRANSFER").astype(float)
    X[:, ML_FEATURES.index("failed_auth_count_24h")] = 0
    X[:, ML_FEATURES.index("days_since_last_similar_txn")] = 0
    X[:, ML_FEATURES.index("gradual_escalation_score")] = np.clip((old_balance - new_balance) / np.maximum(old_balance, 1), 0, 1)
    X[:, ML_FEATURES.index("known_device_count")] = 1
    X[:, ML_FEATURES.index("account_tenure_days")] = step / 24
    X[:, ML_FEATURES.index("shared_device_accounts")] = 0
    X[:, ML_FEATURES.index("shared_recipient_accounts")] = 0
    X[:, ML_FEATURES.index("mule_ring_score")] = 0
    X[:, ML_FEATURES.index("txn_freq_last_24h")] = 0
    X[:, ML_FEATURES.index("hour_deviation")] = np.abs(hour - 14)
    X[:, ML_FEATURES.index("velocity_deviation")] = 0
    X[:, ML_FEATURES.index("recipient_novelty")] = 0
    X[:, ML_FEATURES.index("txn_regularity")] = 0

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["isFraud"].values.astype(int)
    return X, y, ML_FEATURES


def map_ealtman_chunk(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Map ealtman2019 chunk to 21 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, len(ML_FEATURES)), dtype=float)

    amt_str = df["Amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    amt = pd.to_numeric(amt_str, errors="coerce").fillna(0).values.astype(float)
    amt_mean = max(np.median(amt), 1)
    hour_raw = pd.to_numeric(df["Time"].str.split(":").str[0], errors="coerce").fillna(12).values.astype(float)
    dow = pd.to_datetime(df[["Year", "Month", "Day"]].rename(columns={"Year": "year", "Month": "month", "Day": "day"}), errors="coerce").dt.dayofweek.fillna(0).values
    chip = df["Use Chip"].fillna("Unknown").values
    user = df["User"].values

    user_tx_count = pd.Series(user).groupby(user).cumcount().values
    user_avg = pd.Series(amt).groupby(user).transform("mean").values
    user_std = pd.Series(amt).groupby(user).transform("std").fillna(1).values

    X[:, ML_FEATURES.index("amount_ratio")] = amt / amt_mean
    X[:, ML_FEATURES.index("amount_zscore")] = (amt - np.mean(amt)) / max(np.std(amt), 1e-6)
    X[:, ML_FEATURES.index("hour_of_day")] = hour_raw
    X[:, ML_FEATURES.index("is_weekend")] = (dow >= 5).astype(float)
    X[:, ML_FEATURES.index("txn_time_unusual")] = ((hour_raw < 6) | (hour_raw > 22)).astype(float)
    X[:, ML_FEATURES.index("new_device_flag")] = (chip == "Online Transaction").astype(float)
    X[:, ML_FEATURES.index("unusual_location_flag")] = (chip == "Swipe Transaction").astype(float)
    X[:, ML_FEATURES.index("unusual_recipient_flag")] = 0
    X[:, ML_FEATURES.index("failed_auth_count_24h")] = 0
    X[:, ML_FEATURES.index("days_since_last_similar_txn")] = np.clip(np.log1p(user_tx_count), 0, 30)
    X[:, ML_FEATURES.index("gradual_escalation_score")] = np.clip((amt - user_avg) / np.maximum(user_avg, 1), -2, 10)
    X[:, ML_FEATURES.index("known_device_count")] = np.clip(user_tx_count, 0, 50)
    X[:, ML_FEATURES.index("account_tenure_days")] = pd.to_datetime(df[["Year", "Month", "Day"]].rename(columns={"Year": "year", "Month": "month", "Day": "day"}), errors="coerce").dt.dayofyear.fillna(150).values
    X[:, ML_FEATURES.index("shared_device_accounts")] = 0
    X[:, ML_FEATURES.index("shared_recipient_accounts")] = 0
    X[:, ML_FEATURES.index("mule_ring_score")] = 0
    X[:, ML_FEATURES.index("txn_freq_last_24h")] = np.clip(user_tx_count / max(np.median(user_tx_count), 1), 0, 100)
    X[:, ML_FEATURES.index("hour_deviation")] = np.abs(hour_raw - 14)
    X[:, ML_FEATURES.index("velocity_deviation")] = np.clip((user_tx_count - np.median(user_tx_count)) / max(np.median(pd.Series(user_tx_count).groupby(user).transform("std").fillna(1).values), 1), -5, 50)
    X[:, ML_FEATURES.index("recipient_novelty")] = 0
    X[:, ML_FEATURES.index("txn_regularity")] = np.clip(user_std / np.maximum(user_avg, 1), 0, 10)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    is_fraud = df["Is Fraud?"].fillna("No").values
    y = (is_fraud == "Yes").astype(int)
    return X, y, ML_FEATURES


# --- Main ---

def main():
    t_total = time.time()
    print("=" * 70)
    print("PS-14 Multi-Dataset Cross-Evaluation")
    print("Train on kartik2112 → Test on all available datasets")
    print("=" * 70)

    # ===== STEP 1: Load and train on kartik2112 =====
    print("\n[1/4] Training on kartik2112...")
    t0 = time.time()
    train_df = pd.read_csv(DATA / "kaggle_fraud" / "fraudTrain.csv")
    test_df = pd.read_csv(DATA / "kaggle_fraud" / "fraudTest.csv")

    X_train, y_train, _ = map_kartik2112(train_df)
    X_test_k, y_test_k, _ = map_kartik2112(test_df)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_k_s = scaler.transform(X_test_k)

    print(f"  Train: {len(X_train):,} rows, {int(y_train.sum())} fraud")
    print(f"  Test:  {len(X_test_k):,} rows, {int(y_test_k.sum())} fraud")

    # Train models
    n_pos = int(y_train.sum())
    spw = (len(y_train) - n_pos) / max(n_pos, 1)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_s, y_train)

    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, min_child_weight=5,
        tree_method="hist", random_state=42,
        eval_metric="aucpr",
    )
    xgb_model.fit(X_train, y_train, verbose=False)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    train_time = time.time() - t0
    print(f"  Trained in {train_time:.1f}s")

    # ===== STEP 2: Evaluate on all datasets =====
    print("\n[2/4] Cross-dataset evaluation...")
    results = {}

    # Helper to evaluate
    def eval_dataset(name, X, y, subset=None):
        if subset is not None:
            X, y = X[subset], y[subset]
        if len(np.unique(y)) < 2:
            return {"name": name, "status": "SKIP (no fraud in subset)"}

        Xs = scaler.transform(X)
        p_lr = lr.predict_proba(Xs)[:, 1]
        p_xgb = xgb_model.predict_proba(X)[:, 1]
        p_rf = rf.predict_proba(X)[:, 1]
        p_fused = 0.5 * p_rf + 0.3 * p_xgb + 0.2 * p_lr

        m_xgb = compute_metrics(y, p_xgb, f"{name}-xgb")
        m_fused = compute_metrics(y, p_fused, f"{name}-fused")
        return {
            "name": name,
            "n": len(X),
            "fraud": int(y.sum()),
            "prev": round(y.mean(), 5),
            "xgb": m_xgb,
            "fused": m_fused,
        }

    # A. kartik2112 test (in-domain)
    print("  A. kartik2112 (in-domain test)...")
    m = eval_dataset("kartik2112-test", X_test_k, y_test_k)
    results["kartik2112_test"] = m
    print(f"     XGB: ROC={m['xgb']['roc_auc']}  Fused: ROC={m['fused']['roc_auc']}")

    # B. ULB Credit Card
    print("  B. ULB Credit Card...")
    t0 = time.time()
    ulb = pd.read_csv(DATA / "creditcard.csv")
    X_ulb, y_ulb, _ = map_ulb_creditcard(ulb)
    m = eval_dataset("ulb-creditcard", X_ulb, y_ulb)
    results["ulb_creditcard"] = m
    print(f"     {m['n']:,} rows, {m['fraud']} fraud, XGB ROC={m['xgb']['roc_auc']}, time={time.time()-t0:.1f}s")

    # C. PaySim 1M
    print("  C. PaySim 1M...")
    t0 = time.time()
    ps = pd.read_csv(DATA / "paysim_1m.csv")
    X_ps, y_ps, _ = map_paysim(ps)
    m = eval_dataset("paysim-1m", X_ps, y_ps)
    results["paysim_1m"] = m
    print(f"     {m['n']:,} rows, {m['fraud']} fraud, XGB ROC={m['xgb']['roc_auc']}, time={time.time()-t0:.1f}s")

    # D. ealtman2019 (first 300K for speed)
    print("  D. ealtman2019 (300K sample)...")
    t0 = time.time()
    chunk_size = 300_000
    ealtman_chunks = []
    n_chunks = 0
    for chunk in pd.read_csv(DATA / "ealtman2019" / "credit_card_transactions-ibm_v2.csv",
                              chunksize=chunk_size):
        if n_chunks >= 1:
            break
        ealtman_chunks.append(chunk)
        n_chunks += 1
    ealtman_df = pd.concat(ealtman_chunks)
    X_ealt, y_ealt, _ = map_ealtman_chunk(ealtman_df)
    m = eval_dataset("ealtman2019-300k", X_ealt, y_ealt)
    results["ealtman2019_300k"] = m
    print(f"     {m['n']:,} rows, {m['fraud']} fraud, XGB ROC={m['xgb']['roc_auc']}, time={time.time()-t0:.1f}s")

    # E. Elliptic Bitcoin
    print("  E. Elliptic Bitcoin...")
    t0 = time.time()
    elliptic_classes = pd.read_csv(DATA / "elliptic_bitcoin_dataset" / "elliptic_txs_classes.csv")
    # Features CSV has no headers — first column is txId
    elliptic_features = pd.read_csv(DATA / "elliptic_bitcoin_dataset" / "elliptic_txs_features.csv", header=None)
    elliptic_features.columns = ["txId"] + [f"local_feature_{i}" for i in range(elliptic_features.shape[1] - 1)]
    # Merge
    elliptic = elliptic_features.merge(elliptic_classes, on="txId", how="inner")
    elliptic["label_int"] = elliptic["class"].map({"1": 1, "2": 0}).fillna(-1)
    elliptic = elliptic[elliptic["label_int"] >= 0]
    # Use first 166 local features → map to ML_FEATURES via PCA-like aggregation
    v_cols = [c for c in elliptic.columns if c.startswith("local_feature_")]
    if not v_cols:
        v_cols = [c for c in elliptic.columns if c not in ("txId", "class", "label_int", "time_step")]
    if v_cols:
        v_data = elliptic[v_cols].fillna(0).values
        X_e = np.zeros((len(elliptic), len(ML_FEATURES)), dtype=float)
        X_e[:, ML_FEATURES.index("amount_zscore")] = (v_data[:, 0] - v_data[:, 0].mean()) / max(v_data[:, 0].std(), 1e-6) if v_data.shape[1] > 0 else 0
        X_e[:, ML_FEATURES.index("txn_freq_last_24h")] = np.clip(np.mean(np.abs(v_data[:, :10]), axis=1) * 5, 0, 100)
        X_e[:, ML_FEATURES.index("txn_time_unusual")] = (np.abs(v_data[:, 0]) > 2).astype(float)
        X_e[:, ML_FEATURES.index("new_device_flag")] = (np.abs(v_data[:, 1]) > 2).astype(float) if v_data.shape[1] > 1 else 0
        X_e[:, ML_FEATURES.index("unusual_location_flag")] = (np.abs(v_data[:, 2]) > 2).astype(float) if v_data.shape[1] > 2 else 0
        X_e[:, ML_FEATURES.index("unusual_recipient_flag")] = (np.abs(v_data[:, 3]) > 2).astype(float) if v_data.shape[1] > 3 else 0
        X_e[:, ML_FEATURES.index("days_since_last_similar_txn")] = np.clip(np.abs(v_data[:, 4]), 0, 30) if v_data.shape[1] > 4 else 0
        X_e[:, ML_FEATURES.index("gradual_escalation_score")] = np.clip(np.abs(v_data[:, 5]) / 5, 0, 1) if v_data.shape[1] > 5 else 0
        X_e[:, ML_FEATURES.index("hour_deviation")] = np.abs(v_data[:, 0] - v_data[:, 0].mean()) if v_data.shape[1] > 0 else 0
        X_e[:, ML_FEATURES.index("velocity_deviation")] = np.clip(np.abs(v_data[:, 6]), -5, 50) if v_data.shape[1] > 6 else 0
        X_e[:, ML_FEATURES.index("recipient_novelty")] = (np.abs(v_data[:, 7]) > 1.5).astype(float) if v_data.shape[1] > 7 else 0
        X_e[:, ML_FEATURES.index("txn_regularity")] = np.clip(np.std(v_data, axis=1), 0, 10)
        X_e = np.nan_to_num(X_e, nan=0.0, posinf=0.0, neginf=0.0)
        y_e = elliptic["label_int"].values.astype(int)
        m = eval_dataset("elliptic-bitcoin", X_e, y_e)
        results["elliptic_bitcoin"] = m
        print(f"     {m['n']:,} rows, {m['fraud']} fraud, XGB ROC={m['xgb']['roc_auc']}, time={time.time()-t0:.1f}s")
    else:
        print("     SKIP (no feature columns found)")

    # F. fraud_data.csv
    print("  F. fraud_data.csv...")
    t0 = time.time()
    fd = pd.read_csv(DATA / "fraud_data.csv")
    # Check columns
    if "is_fraud" in fd.columns:
        fraud_col = "is_fraud"
    elif "fraud" in fd.columns:
        fraud_col = "fraud"
    else:
        fraud_col = None
        print(f"     SKIP (no fraud label column, cols: {list(fd.columns)[:10]})")

    if fraud_col:
        # Try to map features
        X_fd = np.zeros((len(fd), len(ML_FEATURES)), dtype=float)
        amt_col = [c for c in fd.columns if "amt" in c.lower() or "amount" in c.lower()]
        if amt_col:
            amt = pd.to_numeric(fd[amt_col[0]], errors="coerce").fillna(0).values.astype(float)
            X_fd[:, ML_FEATURES.index("amount_ratio")] = amt / max(np.median(amt), 1)
            X_fd[:, ML_FEATURES.index("amount_zscore")] = (amt - np.mean(amt)) / max(np.std(amt), 1e-6)
        hour_col = [c for c in fd.columns if "hour" in c.lower() or "time" in c.lower()]
        if hour_col:
            hour = pd.to_numeric(fd[hour_col[0]], errors="coerce").fillna(12).values.astype(float) % 24
            X_fd[:, ML_FEATURES.index("hour_of_day")] = hour
            X_fd[:, ML_FEATURES.index("txn_time_unusual")] = ((hour < 6) | (hour > 22)).astype(float)
            X_fd[:, ML_FEATURES.index("hour_deviation")] = np.abs(hour - 14)
        X_fd = np.nan_to_num(X_fd, nan=0.0, posinf=0.0, neginf=0.0)
        y_fd = fd[fraud_col].values.astype(int)
        m = eval_dataset("fraud-data", X_fd, y_fd)
        results["fraud_data"] = m
        print(f"     {m['n']:,} rows, {m['fraud']} fraud, XGB ROC={m['xgb']['roc_auc']}, time={time.time()-t0:.1f}s")

    # ===== STEP 3: Aggregate results =====
    print("\n[3/4] Summary...")
    print("=" * 70)
    print(f"{'Dataset':<25} {'N':>10} {'Fraud':>7} {'Prev':>7} {'XGB ROC':>8} {'Fused ROC':>10} {'R@1%FPR':>8}")
    print("-" * 70)
    for key, r in results.items():
        if "status" in r:
            print(f"{r['name']:<25} {r['status']}")
            continue
        xgb_roc = r['xgb'].get('roc_auc', 'N/A')
        fused_roc = r['fused'].get('roc_auc', 'N/A')
        r1 = r['fused'].get('recall_at_1pct_fpr', 'N/A')
        print(f"{r['name']:<25} {r['n']:>10,} {r['fraud']:>7} {r['prev']:>7.3%} {xgb_roc:>8} {fused_roc:>10} {r1:>8}")

    # ===== STEP 4: Save results =====
    print("\n[4/4] Saving results...")
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_dataset": "kartik2112 (1.3M rows)",
        "n_features": 21,
        "results": results,
        "total_time_s": round(time.time() - t_total, 1),
    }
    out_path = RESULTS / "multi_dataset_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {out_path}")
    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
