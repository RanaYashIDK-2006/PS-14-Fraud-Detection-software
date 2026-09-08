#!/usr/bin/env python3
"""Optimized Authoritative Benchmark — PS-14 Fraud Detection.

Handles large datasets (1M+ rows) efficiently by pre-aggregating group
statistics instead of O(n²) expanding() transforms.

Usage:
    python scripts/run_benchmark.py [--category all|in_domain|temporal|cross_domain]

Output:
    benchmarks/results.json — all experiment results
    benchmarks/manifest.json — metadata
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    brier_score_loss,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
RESULTS_DIR = ROOT / "benchmarks"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true, y_prob):
    """Full metric suite with bootstrap 95% CIs."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    n_fraud = int(y_true.sum())
    n_legit = n - n_fraud
    prev = n_fraud / n if n > 0 else 0.0

    roc = roc_auc_score(y_true, y_prob) if n_fraud > 0 and n_legit > 0 else None
    pr = average_precision_score(y_true, y_prob) if n_fraud > 0 else None
    brier = brier_score_loss(y_true, y_prob) if n > 0 else None

    recall_at = {}
    if n_fraud > 0 and n_legit > 0:
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
        for target_fpr, label in [(0.001, "0.1pct"), (0.005, "0.5pct"), (0.01, "1pct")]:
            mask = fpr_arr <= target_fpr
            if mask.any():
                recall_at[label] = round(float(tpr_arr[np.where(mask)[0][-1]]), 4)
            else:
                recall_at[label] = 0.0
    else:
        for label in ("0.1pct", "0.5pct", "1pct"):
            recall_at[label] = None

    # Bootstrap CIs (sampled for speed on large datasets)
    ci = {}
    if n >= 50 and n_fraud >= 10:
        rng = np.random.default_rng(SEED)
        n_boot = 200
        boot_roc, boot_pr, boot_r1 = [], [], []
        sample_size = min(n, 50000)
        for _ in range(n_boot):
            idx = rng.choice(n, size=sample_size, replace=True)
            yt, yp = y_true[idx], y_prob[idx]
            if yt.sum() >= 2 and (1 - yt).sum() >= 2:
                boot_roc.append(roc_auc_score(yt, yp))
                boot_pr.append(average_precision_score(yt, yp))
                fpr_b, tpr_b, _ = roc_curve(yt, yp)
                mask = fpr_b <= 0.01
                boot_r1.append(float(tpr_b[np.where(mask)[0][-1]]) if mask.any() else 0.0)
        if boot_roc:
            ci["roc_auc"] = [round(float(np.percentile(boot_roc, 2.5)), 4),
                             round(float(np.percentile(boot_roc, 97.5)), 4)]
            ci["pr_auc"] = [round(float(np.percentile(boot_pr, 2.5)), 4),
                            round(float(np.percentile(boot_pr, 97.5)), 4)]
            ci["recall_1pct"] = [round(float(np.percentile(boot_r1, 2.5)), 4),
                                 round(float(np.percentile(boot_r1, 97.5)), 4)]

    return {
        "n": n, "n_fraud": n_fraud, "n_legit": n_legit,
        "prevalence": round(prev, 6),
        "roc_auc": round(float(roc), 4) if roc is not None else None,
        "pr_auc": round(float(pr), 4) if pr is not None else None,
        "brier": round(float(brier), 6) if brier is not None else None,
        "recall_at_0_1pct_fpr": recall_at.get("0.1pct"),
        "recall_at_0_5pct_fpr": recall_at.get("0.5pct"),
        "recall_at_1pct_fpr": recall_at.get("1pct"),
        "precision": None,
        "ci": ci,
    }


def hash_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# ─── Optimized Feature Pipelines ──────────────────────────────────────────

def _compute_card_stats(df):
    """Compute per-card aggregate statistics from a DataFrame.
    
    These stats represent the card's historical profile at the time of
    the latest transaction in df. For point-in-time correctness in
    temporal evaluation, call this on TRAINING data only, then pass
    the result to features_kaggle_from_stats() for test data.
    """
    # Median amount, mean, count, std, first unix, unique merchants/cities, median hour
    card_stats = df.groupby("cc_num").agg(
        card_median_amt=("amt", "median"),
        card_mean_amt=("amt", "mean"),
        card_count=("amt", "count"),
        card_std_amt=("amt", "std"),
        card_first_unix=("unix_time", "min"),
        card_n_unique_merchant=("merchant", "nunique"),
        card_n_unique_city=("city", "nunique"),
        card_median_hour=("trans_date_trans_time", lambda x: pd.to_datetime(x).dt.hour.median()),
    ).reset_index()
    card_stats["card_std_amt"] = card_stats["card_std_amt"].fillna(1.0)
    # Merchants seen per card (set of merchant names)
    card_merchants = df.groupby("cc_num")["merchant"].apply(set).reset_index()
    card_merchants.columns = ["cc_num", "merchants"]
    # Categories seen per card
    card_cats = df.groupby("cc_num")["category"].apply(set).reset_index()
    card_cats.columns = ["cc_num", "categories"]
    # Merchant encounter counts (how many times each card saw each merchant)
    merch_counts = df.groupby(["cc_num", "merchant"]).size().reset_index(name="count")
    merch_count_dict = {}
    for _, row in merch_counts.iterrows():
        merch_count_dict.setdefault(row["cc_num"], {})[row["merchant"]] = row["count"]
    card_stats = card_stats.merge(card_merchants, on="cc_num", how="left")
    card_stats = card_stats.merge(card_cats, on="cc_num", how="left")
    card_stats["merch_counts"] = card_stats["cc_num"].map(merch_count_dict).fillna({})
    # Per-card median distance
    lat1, lon1 = df["lat"].values, df["long"].values
    lat2, lon2 = df["merch_lat"].values, df["merch_long"].values
    dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    dist_df = pd.DataFrame({"cc_num": df["cc_num"], "dist": dist})
    card_dist_med = dist_df.groupby("cc_num")["dist"].median().reset_index()
    card_dist_med.columns = ["cc_num", "card_dist_med"]
    card_stats = card_stats.merge(card_dist_med, on="cc_num", how="left")
    # Per-card merchant encounter count in df
    card_merchant_first = df.groupby(["cc_num", "merchant"]).cumcount()
    mf_df = pd.DataFrame({"cc_num": df["cc_num"], "mf": card_merchant_first})
    # Not needed for stats — used inline in feature build
    # Merch-level stats
    merch_card_count = df.groupby("merchant")["cc_num"].nunique().reset_index()
    merch_card_count.columns = ["merchant", "mc_count"]
    return {
        "card_stats": card_stats,
        "merch_card_count": merch_card_count,
        "n_rows": len(df),
    }


def features_kaggle_from_stats(df, stats):
    """Build 21 features using pre-computed training-period stats.
    
    This is the correct way to build features for temporal evaluation:
    per-card statistics come from the training period, so the test set
    uses the same feature distributions as training.
    """
    from src.privacy_layer.features import ML_FEATURES
    n = len(df)
    X = np.zeros((n, len(ML_FEATURES)), dtype=np.float32)
    amt = df["amt"].values.astype(np.float32)
    hours = pd.to_datetime(df["trans_date_trans_time"]).dt.hour.values.astype(np.float32)
    dow = pd.to_datetime(df["trans_date_trans_time"]).dt.dayofweek.values.astype(np.float32)
    unix = df["unix_time"].values.astype(np.float64)

    card_stats = stats["card_stats"]
    merch_card_count = stats["merch_card_count"]

    # Merge training stats onto this dataframe
    merged = df[["cc_num"]].merge(card_stats, on="cc_num", how="left")
    card_median = merged["card_median_amt"].fillna(float(np.median(amt))).values
    card_mean = merged["card_mean_amt"].fillna(float(np.mean(amt))).values
    card_count = merged["card_count"].fillna(0).values
    card_std = merged["card_std_amt"].fillna(1.0).values
    card_first = merged["card_first_unix"].fillna(float(np.min(unix))).values
    card_n_merch = merged["card_n_unique_merchant"].fillna(0).values
    card_n_city = merged["card_n_unique_city"].fillna(0).values
    card_med_hour = merged["card_median_hour"].fillna(12.0).values
    hist_merchants_raw = merged["merchants"].values
    hist_cats_raw = merged["categories"].values
    hist_merchants = [x if isinstance(x, set) else set() for x in hist_merchants_raw]
    hist_cats = [x if isinstance(x, set) else set() for x in hist_cats_raw]
    merch_counts_row_raw = merged["merch_counts"].values
    merch_counts_row = [x if isinstance(x, dict) else {} for x in merch_counts_row_raw]
    card_dist_med = merged["card_dist_med"].fillna(1.0).values

    # Feature 0: amount_ratio
    X[:, ML_FEATURES.index("amount_ratio")] = np.where(card_median > 0, amt / card_median, 0.0)
    # Feature 1: txn_freq_last_24h (historical count)
    X[:, ML_FEATURES.index("txn_freq_last_24h")] = card_count.astype(np.float32)
    # Feature 2: txn_time_unusual
    X[:, ML_FEATURES.index("txn_time_unusual")] = np.abs(hours - card_med_hour) / 12.0
    # Feature 3: new_device_flag (category not seen in training)
    new_cat = np.zeros(n, dtype=np.float32)
    for i in range(n):
        cats = hist_cats[i]
        if isinstance(cats, set) and len(cats) > 0:
            new_cat[i] = 0.0 if df["category"].iloc[i] in cats else 1.0
        else:
            new_cat[i] = 1.0  # new card, treat as new device
    X[:, ML_FEATURES.index("new_device_flag")] = new_cat
    # Feature 4: unusual_location_flag
    lat1, lon1 = df["lat"].values, df["long"].values
    lat2, lon2 = df["merch_lat"].values, df["merch_long"].values
    dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    X[:, ML_FEATURES.index("unusual_location_flag")] = (dist > card_dist_med * 2).astype(np.float32)
    # Feature 5: unusual_recipient_flag (merchant not seen in training)
    new_merch = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ms = hist_merchants[i]
        if isinstance(ms, set) and len(ms) > 0:
            new_merch[i] = 0.0 if df["merchant"].iloc[i] in ms else 1.0
        else:
            new_merch[i] = 1.0
    X[:, ML_FEATURES.index("unusual_recipient_flag")] = new_merch
    # Feature 6: failed_auth_count_24h
    X[:, ML_FEATURES.index("failed_auth_count_24h")] = 0.0
    # Feature 7: days_since_last_similar_txn (use training last_tx as anchor)
    last_tx_arr = merged["card_first_unix"].fillna(0).values  # approximate with first
    X[:, ML_FEATURES.index("days_since_last_similar_txn")] = np.clip(
        (unix - last_tx_arr) / 86400.0, 0, 365).astype(np.float32)
    # Feature 8: gradual_escalation_score
    X[:, ML_FEATURES.index("gradual_escalation_score")] = np.where(
        card_mean > 0, amt / card_mean, 0.0).astype(np.float32)
    # Feature 9: known_device_count (merchants seen in training)
    X[:, ML_FEATURES.index("known_device_count")] = card_n_merch.astype(np.float32)
    # Feature 10: account_tenure_days
    X[:, ML_FEATURES.index("account_tenure_days")] = ((unix - card_first) / 86400.0).astype(np.float32)
    # Feature 11: hour_of_day
    X[:, ML_FEATURES.index("hour_of_day")] = hours
    # Feature 12: is_weekend
    X[:, ML_FEATURES.index("is_weekend")] = (dow >= 5).astype(np.float32)
    # Feature 13: shared_device_accounts
    mc_merged = df[["merchant"]].merge(merch_card_count, on="merchant", how="left")
    X[:, ML_FEATURES.index("shared_device_accounts")] = mc_merged["mc_count"].fillna(1).values.astype(np.float32)
    X[:, ML_FEATURES.index("shared_recipient_accounts")] = mc_merged["mc_count"].fillna(1).values.astype(np.float32)
    # Feature 15: mule_ring_score (cities seen in training)
    X[:, ML_FEATURES.index("mule_ring_score")] = card_n_city.astype(np.float32)
    # Feature 16: hour_deviation
    X[:, ML_FEATURES.index("hour_deviation")] = np.abs(hours - card_med_hour).astype(np.float32)
    # Feature 17: amount_zscore (using training mean/std)
    X[:, ML_FEATURES.index("amount_zscore")] = np.where(
        card_std > 0, (amt - card_mean) / card_std, 0.0).astype(np.float32)
    # Feature 18: velocity_deviation (training count / 24)
    X[:, ML_FEATURES.index("velocity_deviation")] = card_count.astype(np.float32) / 24.0
    # Feature 19: recipient_novelty
    novelty = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mc = merch_counts_row[i]
        merchant = df["merchant"].iloc[i]
        novelty[i] = 1.0 / (mc.get(merchant, 0) + 1) if isinstance(mc, dict) else 1.0
    X[:, ML_FEATURES.index("recipient_novelty")] = novelty
    # Feature 20: txn_regularity (training-based gap std, approximate globally)
    X[:, ML_FEATURES.index("txn_regularity")] = 0.0  # will be computed separately if needed

    return X


def features_kaggle_fast(df):
    """Vectorized 21-feature pipeline — O(n) instead of O(n²).
    
    NOTE: This computes stats from df itself. For temporal evaluation,
    use _compute_card_stats() + features_kaggle_from_stats() instead.
    """
    from src.privacy_layer.features import ML_FEATURES
    n = len(df)
    X = np.zeros((n, len(ML_FEATURES)), dtype=np.float32)
    amt = df["amt"].values.astype(np.float32)
    hours = pd.to_datetime(df["trans_date_trans_time"]).dt.hour.values.astype(np.float32)
    dow = pd.to_datetime(df["trans_date_trans_time"]).dt.dayofweek.values.astype(np.float32)
    unix = df["unix_time"].values.astype(np.float64)

    # Pre-aggregate per card (O(n) merge)
    card_stats = df.groupby("cc_num").agg(
        card_median_amt=("amt", "median"),
        card_mean_amt=("amt", "mean"),
        card_count=("amt", "count"),
        card_std_amt=("amt", "std"),
        card_first_unix=("unix_time", "min"),
        card_n_unique_merchant=("merchant", "nunique"),
        card_n_unique_city=("city", "nunique"),
        card_median_hour=("trans_date_trans_time", lambda x: pd.to_datetime(x).dt.hour.median()),
    ).reset_index()
    card_stats["card_std_amt"] = card_stats["card_std_amt"].fillna(1.0)

    # Also compute card-merchant and card-category counts
    card_merchant_counts = df.groupby(["cc_num", "merchant"]).size().reset_index(name="cm_count")
    card_cat_first = df.groupby(["cc_num", "category"]).cumcount().values
    card_merchant_first = df.groupby(["cc_num", "merchant"]).cumcount().values

    # Merge card stats
    merged = df[["cc_num"]].merge(card_stats, on="cc_num", how="left")
    cm = merged.values  # just use index arrays directly
    card_median = merged["card_median_amt"].values
    card_mean = merged["card_mean_amt"].values
    card_count = merged["card_count"].values
    card_std = merged["card_std_amt"].values
    card_first = merged["card_first_unix"].values
    card_n_merch = merged["card_n_unique_merchant"].values
    card_n_city = merged["card_n_unique_city"].values
    card_med_hour = merged["card_median_hour"].values

    X[:, ML_FEATURES.index("amount_ratio")] = np.where(card_median > 0, amt / card_median, 0.0)
    X[:, ML_FEATURES.index("txn_freq_last_24h")] = card_count.astype(np.float32)
    X[:, ML_FEATURES.index("txn_time_unusual")] = hours / 23.0
    X[:, ML_FEATURES.index("new_device_flag")] = (card_cat_first == 0).astype(np.float32)

    lat1, lon1 = df["lat"].values, df["long"].values
    lat2, lon2 = df["merch_lat"].values, df["merch_long"].values
    dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

    # Per-card median distance (vectorized)
    dist_df = pd.DataFrame({"cc_num": df["cc_num"], "dist": dist})
    card_dist_med = dist_df.groupby("cc_num")["dist"].median().reset_index()
    card_dist_med.columns = ["cc_num", "card_dist_med"]
    dist_merged = df[["cc_num"]].merge(card_dist_med, on="cc_num", how="left")["card_dist_med"].values
    X[:, ML_FEATURES.index("unusual_location_flag")] = (dist > dist_merged * 2).astype(np.float32)
    X[:, ML_FEATURES.index("unusual_recipient_flag")] = (card_merchant_first == 0).astype(np.float32)
    X[:, ML_FEATURES.index("failed_auth_count_24h")] = 0.0

    # Time gap — use sorting trick
    df_sorted_idx = np.argsort(unix, kind="mergesort")
    inv = np.empty_like(df_sorted_idx)
    inv[df_sorted_idx] = np.arange(n)
    sorted_unix = unix[df_sorted_idx]
    gaps_sorted = np.diff(sorted_unix, prepend=sorted_unix[0] - 86400)
    gaps = gaps_sorted[inv] / 86400.0
    X[:, ML_FEATURES.index("days_since_last_similar_txn")] = np.clip(gaps, 0, 365).astype(np.float32)

    # Escalation: ratio to card mean (pre-aggregated, O(n))
    X[:, ML_FEATURES.index("gradual_escalation_score")] = np.where(card_mean > 0, amt / card_mean, 0.0).astype(np.float32)

    X[:, ML_FEATURES.index("known_device_count")] = card_n_merch.astype(np.float32)
    X[:, ML_FEATURES.index("account_tenure_days")] = ((unix - card_first) / 86400.0).astype(np.float32)
    X[:, ML_FEATURES.index("hour_of_day")] = hours
    X[:, ML_FEATURES.index("is_weekend")] = (dow >= 5).astype(np.float32)

    # Shared features (pre-aggregated)
    merch_card_count = df.groupby("merchant")["cc_num"].nunique().reset_index()
    merch_card_count.columns = ["merchant", "mc_count"]
    mc_merged = df[["merchant"]].merge(merch_card_count, on="merchant", how="left")["mc_count"].values
    X[:, ML_FEATURES.index("shared_device_accounts")] = mc_merged.astype(np.float32)
    X[:, ML_FEATURES.index("shared_recipient_accounts")] = mc_merged.astype(np.float32)
    X[:, ML_FEATURES.index("mule_ring_score")] = card_n_city.astype(np.float32)
    X[:, ML_FEATURES.index("hour_deviation")] = np.abs(hours - card_med_hour).astype(np.float32)

    # Z-score
    X[:, ML_FEATURES.index("amount_zscore")] = np.where(card_std > 0, (amt - card_mean) / card_std, 0.0).astype(np.float32)
    X[:, ML_FEATURES.index("velocity_deviation")] = card_count.astype(np.float32) / 24.0

    # Recipient novelty (pre-aggregated)
    seen = (card_merchant_first + 1).astype(np.float32)
    X[:, ML_FEATURES.index("recipient_novelty")] = 1.0 / seen

    # Txn regularity — approximate with per-card std of gaps
    gap_df = pd.DataFrame({"cc_num": df["cc_num"], "gap": gaps})
    card_gap_std = gap_df.groupby("cc_num")["gap"].std().reset_index()
    card_gap_std.columns = ["cc_num", "gap_std"]
    card_gap_std["gap_std"] = card_gap_std["gap_std"].fillna(0)
    gap_std_merged = df[["cc_num"]].merge(card_gap_std, on="cc_num", how="left")["gap_std"].values
    X[:, ML_FEATURES.index("txn_regularity")] = np.clip(gap_std_merged, 0, 100).astype(np.float32)

    return X


def features_ulb(df):
    """ULB: PCA components V1-V28 + Amount + Time."""
    cols = [c for c in df.columns if c.startswith("V")]
    X = df[cols].values.astype(np.float32)
    if "Amount" in df.columns:
        X = np.column_stack([X, df["Amount"].values.astype(np.float32)])
    if "Time" in df.columns:
        X = np.column_stack([X, df["Time"].values.astype(np.float32)])
    return X


def features_paysim(df):
    """PaySim: type dummies + amounts + balances."""
    type_dummies = pd.get_dummies(df["type"], prefix="type").values.astype(np.float32)
    amt = df["amount"].values.astype(np.float32).reshape(-1, 1)
    bal_cols = []
    for c in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        if c in df.columns:
            bal_cols.append(df[c].values.astype(np.float32).reshape(-1, 1))
    if bal_cols:
        return np.hstack([type_dummies, amt] + bal_cols)
    return np.hstack([type_dummies, amt])


def features_fraud_data(df):
    """fraud_data: PCA V1-V28 + Amount."""
    v_cols = [c for c in df.columns if c.startswith("V")]
    X = df[v_cols].values.astype(np.float32)
    if "Amount" in df.columns:
        X = np.column_stack([X, df["Amount"].values.astype(np.float32)])
    return X


def train_xgb(X_train, y_train, n_estimators=400, max_depth=8):
    """Train XGB with class balancing."""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=max(n_neg / max(n_pos, 1), 1.0),
        eval_metric="auc", random_state=SEED, n_jobs=-1, verbosity=0,
        min_child_weight=3, gamma=0.05, reg_alpha=0.1,
    )
    model.fit(X_train, y_train)
    return model


def prepare_split(X, y, split_ratio=0.8):
    """Prepare train/test split with imputer and scaler."""
    split = int(len(X) * split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    imp = SimpleImputer(strategy="median").fit(X_train)
    Xc_train = np.nan_to_num(imp.transform(X_train), nan=0.0, posinf=10.0, neginf=-10.0)
    Xc_test = np.nan_to_num(imp.transform(X_test), nan=0.0, posinf=10.0, neginf=-10.0)
    scaler = StandardScaler().fit(Xc_train)
    return scaler.transform(Xc_train), scaler.transform(Xc_test), y_train, y_test, split


def eval_experiment(X, y, name, dataset, data_type, split_ratio=0.8, **extra):
    """Full train/eval pipeline returning result dict."""
    Xs_train, Xs_test, y_train, y_test, train_n = prepare_split(X, y, split_ratio)
    model = train_xgb(Xs_train, y_train)
    prob = model.predict_proba(Xs_test)[:, 1]
    mets = compute_metrics(y_test, prob)
    return {
        "experiment_id": name,
        "dataset": dataset,
        "data_type": data_type,
        "train_n": train_n,
        "test_n": mets["n"],
        **mets,
        **extra,
    }


# ─── Experiment Runners ───────────────────────────────────────────────────

def run_in_domain():
    """IN_DOMAIN: kaggle_fraud 80/20 temporal split."""
    print("\n=== IN_DOMAIN: kaggle_fraud 80/20 temporal ===")
    t0 = time.time()
    df = pd.concat([
        pd.read_csv("data/kaggle_fraud/fraudTrain.csv"),
        pd.read_csv("data/kaggle_fraud/fraudTest.csv"),
    ], ignore_index=True)
    df = df.drop(columns=["first", "last", "street", "state", "zip", "dob", "job", "trans_num", "Unnamed: 0"], errors="ignore")
    df = df.sort_values("unix_time").reset_index(drop=True)
    y = df["is_fraud"].values.astype(int)
    X = features_kaggle_fast(df)
    print(f"  Features: {X.shape} in {time.time()-t0:.1f}s")
    return eval_experiment(X, y, "IN_DOMAIN_v1", "kaggle_fraud", "REAL_PUBLIC_DATASET",
                           split_ratio=0.8, split="80/20 temporal")


def run_temporal():
    """TEMPORAL: Train 2019+Q1, test Apr-Sep 2020.

    CRITICAL: per-card stats are computed from TRAINING data only,
    then applied to test data. This matches production behavior and
    prevents feature distribution mismatch between train/test.
    """
    print("\n=== TEMPORAL: train 2019+Q1 → test Apr-Sep 2020 ===")
    t0 = time.time()
    df = pd.concat([
        pd.read_csv("data/kaggle_fraud/fraudTrain.csv"),
        pd.read_csv("data/kaggle_fraud/fraudTest.csv"),
    ], ignore_index=True)
    df = df.drop(columns=["first", "last", "street", "state", "zip", "dob", "job", "trans_num", "Unnamed: 0"], errors="ignore")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("unix_time").reset_index(drop=True)

    train_mask = (df["trans_date_trans_time"].dt.year == 2019) | \
                 ((df["trans_date_trans_time"].dt.year == 2020) & (df["trans_date_trans_time"].dt.month <= 3))
    test_mask = (df["trans_date_trans_time"].dt.year == 2020) & \
                (df["trans_date_trans_time"].dt.month >= 4) & (df["trans_date_trans_time"].dt.month <= 9)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    print(f"  Train: {len(df_train):,} rows, Test: {len(df_test):,} rows")

    # Step 1: Compute per-card stats from TRAINING data only
    print("  Computing training-period card stats...")
    stats = _compute_card_stats(df_train)
    print(f"  {len(stats['card_stats']):,} unique cards in training")

    # Step 2: Build train features using training stats
    t1 = time.time()
    X_train = features_kaggle_from_stats(df_train, stats)
    print(f"  Train features: {X_train.shape} in {time.time()-t1:.1f}s")

    # Step 3: Build test features using SAME training stats (point-in-time correct)
    t2 = time.time()
    X_test = features_kaggle_from_stats(df_test, stats)
    print(f"  Test features: {X_test.shape} in {time.time()-t2:.1f}s")

    y_train = df_train["is_fraud"].values.astype(int)
    y_test = df_test["is_fraud"].values.astype(int)

    imp = SimpleImputer(strategy="median").fit(X_train)
    Xc_train = np.nan_to_num(imp.transform(X_train), nan=0.0, posinf=10.0, neginf=-10.0)
    Xc_test = np.nan_to_num(imp.transform(X_test), nan=0.0, posinf=10.0, neginf=-10.0)
    scaler = StandardScaler().fit(Xc_train)
    Xs_train = scaler.transform(Xc_train)
    Xs_test = scaler.transform(Xc_test)

    model = train_xgb(Xs_train, y_train)
    prob = model.predict_proba(Xs_test)[:, 1]
    mets = compute_metrics(y_test, prob)
    return {
        "experiment_id": "TEMPORAL_v1",
        "category": "TEMPORAL",
        "dataset": "kaggle_fraud",
        "data_type": "REAL_PUBLIC_DATASET",
        "split": "train: 2019+Q1, test: Apr-Sep 2020",
        "train_n": len(df_train), "test_n": len(df_test),
        **mets,
    }


def run_cross_domain():
    """CROSS_DOMAIN: Per-domain training + evaluation."""
    print("\n=== CROSS_DOMAIN: per-domain evaluation ===")
    results = []

    # ULB
    print("  ULB (284K rows)...")
    t0 = time.time()
    df_ulb = pd.read_csv("data/creditcard.csv")
    y_ulb = df_ulb["Class"].values.astype(int)
    X_ulb = features_ulb(df_ulb)
    print(f"    Features: {X_ulb.shape} in {time.time()-t0:.1f}s")
    r = eval_experiment(X_ulb, y_ulb, "CROSS_DOMAIN_ulb_v1", "ulb_creditcard", "REAL_PUBLIC_DATASET")
    r["category"] = "CROSS_DOMAIN"
    results.append(r)

    # PaySim
    print("  PaySim (1.2M rows)...")
    t0 = time.time()
    df_ps = pd.read_csv("data/paysim_1m.csv")
    y_ps = df_ps["isFraud"].values.astype(int)
    X_ps = features_paysim(df_ps)
    print(f"    Features: {X_ps.shape} in {time.time()-t0:.1f}s")
    r = eval_experiment(X_ps, y_ps, "CROSS_DOMAIN_paysim_v1", "paysim_1m", "SYNTHETIC")
    r["category"] = "CROSS_DOMAIN"
    results.append(r)

    # fraud_data
    print("  fraud_data (22K rows)...")
    t0 = time.time()
    df_fd = pd.read_csv("data/fraud_data.csv")
    label_col = "Class" if "Class" in df_fd.columns else "is_fraud" if "is_fraud" in df_fd.columns else df_fd.columns[-1]
    y_fd = df_fd[label_col].values.astype(int)
    X_fd = features_fraud_data(df_fd)
    print(f"    Features: {X_fd.shape} in {time.time()-t0:.1f}s")
    r = eval_experiment(X_fd, y_fd, "CROSS_DOMAIN_frauddata_v1", "fraud_data", "REAL_PUBLIC_DATASET")
    r["category"] = "CROSS_DOMAIN"
    results.append(r)

    return results


def run_real_public():
    """REAL_PUBLIC: Evaluate on truly external datasets."""
    print("\n=== REAL_PUBLIC: external dataset evaluation ===")
    results = []

    # Ealtman (if downloaded)
    ealtman_path = ROOT / "data" / "ealtman_creditcard.csv"
    if ealtman_path.exists() and ealtman_path.stat().st_size > 100:
        print("  Ealtman credit card...")
        t0 = time.time()
        df_e = pd.read_csv(ealtman_path)
        # Try to find label column
        for lc in ["Class", "is_fraud", "isFraud", "fraud", "label"]:
            if lc in df_e.columns:
                label_col = lc
                break
        else:
            label_col = df_e.columns[-1]
        y_e = df_e[label_col].values.astype(int)
        X_e = features_ulb(df_e) if any(c.startswith("V") for c in df_e.columns) else df_e.select_dtypes(include=[np.number]).values.astype(np.float32)
        print(f"    Features: {X_e.shape} in {time.time()-t0:.1f}s")
        if int(y_e.sum()) > 0 and (1 - y_e).sum() > 0:
            r = eval_experiment(X_e, y_e, "REAL_PUBLIC_ealtman_v1", "ealtman_creditcard", "REAL_PUBLIC_DATASET")
            r["category"] = "REAL_PUBLIC"
            results.append(r)

    return results


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="all",
                        choices=["all", "in_domain", "temporal", "cross_domain", "real_public"])
    args = parser.parse_args()

    print("=" * 70)
    print("PS-14 AUTHORITATIVE BENCHMARK (Optimized)")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Seed: {SEED}")
    print("=" * 70)

    all_results = []

    if args.category in ("all", "in_domain"):
        all_results.append(run_in_domain())

    if args.category in ("all", "temporal"):
        all_results.append(run_temporal())

    if args.category in ("all", "cross_domain"):
        all_results.extend(run_cross_domain())

    if args.category in ("all", "real_public"):
        all_results.extend(run_real_public())

    # Save
    manifest = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "python_version": sys.version.split()[0],
        "n_experiments": len(all_results),
        "categories": list(set(r.get("category", "IN_DOMAIN") for r in all_results)),
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(all_results, indent=2))
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Summary
    print("\n" + "=" * 70)
    print("AUTHORITATIVE RESULTS")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Dataset':<15} {'Type':<18} {'N':>8} {'Fraud':>6} {'Prev':>8} {'ROC-AUC':>8} {'PR-AUC':>8} {'R@1%':>7} {'Brier':>8}")
    print("-" * 130)
    for r in all_results:
        prev_str = f"{r['prevalence']*100:.3f}%"
        roc_str = f"{r['roc_auc']:.4f}" if r['roc_auc'] else "N/A"
        pr_str = f"{r['pr_auc']:.4f}" if r['pr_auc'] else "N/A"
        r1_str = f"{r['recall_at_1pct_fpr']:.4f}" if r['recall_at_1pct_fpr'] is not None else "N/A"
        brier_str = f"{r['brier']:.6f}" if r.get('brier') is not None else "N/A"
        print(f"{r['experiment_id']:<30} {r.get('dataset','?'):<15} {r.get('data_type','?'):<18} "
              f"{r['n']:>8,} {r['n_fraud']:>6,} {prev_str:>8} {roc_str:>8} {pr_str:>8} {r1_str:>7} {brier_str:>8}")

    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
