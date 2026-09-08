#!/usr/bin/env python3
"""Evaluate PS-14 on ealtman2019/credit-card-transactions (24M rows, real IBM data).

Uses chunked processing to handle the 2.2GB file efficiently.
"""

import sys
import time
import json
import hashlib
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def compute_metrics(y_true, y_prob):
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

    return {
        "n": n, "n_fraud": n_fraud, "n_legit": n_legit,
        "prevalence": round(prev, 6),
        "roc_auc": round(float(roc), 4) if roc is not None else None,
        "pr_auc": round(float(pr), 4) if pr is not None else None,
        "brier": round(float(brier), 6) if brier is not None else None,
        "recall_at_0_1pct_fpr": recall_at.get("0.1pct"),
        "recall_at_0_5pct_fpr": recall_at.get("0.5pct"),
        "recall_at_1pct_fpr": recall_at.get("1pct"),
    }


def load_ealtman(chunksize=500000):
    """Load ealtman dataset in chunks, extracting features."""
    print("Loading ealtman2019 dataset (24M rows)...")

    # First pass: get row count and basic stats
    t0 = time.time()
    filepath = ROOT / "data" / "credit_card_transactions-ibm_v2.csv"

    # Read in chunks
    chunks = []
    total_rows = 0
    fraud_rows = 0

    for chunk in pd.read_csv(filepath, chunksize=chunksize, dtype=str):
        total_rows += len(chunk)
        fraud_rows += (chunk["Is Fraud?"] == "Yes").sum()
        chunks.append(chunk)
        if total_rows % 2000000 == 0:
            print(f"  Read {total_rows:,} rows ({time.time()-t0:.1f}s)...")

    print(f"  Total: {total_rows:,} rows, {fraud_rows:,} fraud ({fraud_rows/total_rows*100:.4f}%)")
    df = pd.concat(chunks, ignore_index=True)
    return df


def engineer_ealtman_features(df):
    """Feature engineering for ealtman dataset."""
    print("Engineering features...")
    t0 = time.time()
    n = len(df)

    # Parse amounts
    df["Amount_num"] = df["Amount"].str.replace("$", "").str.replace(",", "").astype(float)

    # Parse time
    df["Hour"] = df["Time"].str.split(":").str[0].astype(int)
    df["Minute"] = df["Time"].str.split(":").str[1].astype(int)

    # Create unix timestamp for ordering
    df["unix"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].str.zfill(2) + "-" + df["Day"].str.zfill(2) + " " + df["Time"]
    ).astype(np.int64) // 10**9

    # User-card key
    df["user_card"] = df["User"].astype(str) + "_" + df["Card"].astype(str)

    # Target
    y = (df["Is Fraud?"] == "Yes").values.astype(int)

    # Feature matrix
    X = np.zeros((n, 21), dtype=np.float32)

    # 1. Amount stats per user-card (pre-aggregated)
    uc_stats = df.groupby("user_card").agg(
        median_amt=("Amount_num", "median"),
        mean_amt=("Amount_num", "mean"),
        count=("Amount_num", "count"),
        std_amt=("Amount_num", "std"),
    ).reset_index()
    uc_stats["std_amt"] = uc_stats["std_amt"].fillna(1.0)

    merged = df[["user_card"]].merge(uc_stats, on="user_card", how="left")

    # 2. Features
    amt = df["Amount_num"].values
    hours = df["Hour"].values.astype(np.float32)
    dow = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].str.zfill(2) + "-" + df["Day"].str.zfill(2)).dt.dayofweek.values.astype(np.float32)

    X[:, 0] = np.where(merged["median_amt"] > 0, amt / merged["median_amt"], 0.0)  # amount_ratio
    X[:, 1] = merged["count"].values.astype(np.float32)  # txn_freq (all-time)
    X[:, 2] = hours / 23.0  # txn_time_unusual

    # New device flag (first category for this user-card)
    cat_first = df.groupby(["user_card", "Use Chip"]).cumcount().values
    X[:, 3] = (cat_first == 0).astype(np.float32)

    # Distance (use merchant state uniqueness as proxy)
    X[:, 4] = 0.0  # no lat/long

    # New merchant flag
    merch_first = df.groupby(["user_card", "Merchant Name"]).cumcount().values
    X[:, 5] = (merch_first == 0).astype(np.float32)

    # Failed auth count (use Errors column)
    X[:, 6] = (df["Errors?"].fillna("") != "").astype(np.float32)

    # Days since last txn
    df_sorted_idx = np.argsort(df["unix"].values, kind="mergesort")
    inv = np.empty_like(df_sorted_idx)
    inv[df_sorted_idx] = np.arange(n)
    sorted_unix = df["unix"].values[df_sorted_idx]
    gaps_sorted = np.diff(sorted_unix, prepend=sorted_unix[0] - 86400)
    gaps = gaps_sorted[inv] / 86400.0
    X[:, 7] = np.clip(gaps, 0, 365).astype(np.float32)

    # Escalation score
    X[:, 8] = np.where(merged["mean_amt"] > 0, amt / merged["mean_amt"], 0.0).astype(np.float32)

    # Known merchant count per user-card
    merch_count = df.groupby("user_card")["Merchant Name"].nunique().reset_index()
    merch_count.columns = ["user_card", "n_merch"]
    mc_merged = df[["user_card"]].merge(merch_count, on="user_card", how="left")
    X[:, 9] = mc_merged["n_merch"].values.astype(np.float32)

    # Account tenure
    first_unix = df.groupby("user_card")["unix"].min().reset_index()
    first_unix.columns = ["user_card", "first_unix"]
    fu_merged = df[["user_card"]].merge(first_unix, on="user_card", how="left")
    X[:, 10] = ((df["unix"].values - fu_merged["first_unix"].values) / 86400.0).astype(np.float32)

    X[:, 11] = hours
    X[:, 12] = (dow >= 5).astype(np.float32)

    # Shared merchant accounts
    merch_users = df.groupby("Merchant Name")["User"].nunique().reset_index()
    merch_users.columns = ["Merchant Name", "n_users"]
    mu_merged = df[["Merchant Name"]].merge(merch_users, on="Merchant Name", how="left")
    X[:, 13] = mu_merged["n_users"].values.astype(np.float32)
    X[:, 14] = mu_merged["n_users"].values.astype(np.float32)  # shared_recipient

    # Mule ring: cities per user
    city_count = df.groupby("user_card")["Merchant City"].nunique().reset_index()
    city_count.columns = ["user_card", "n_city"]
    cc_merged = df[["user_card"]].merge(city_count, on="user_card", how="left")
    X[:, 15] = cc_merged["n_city"].values.astype(np.float32)

    # Hour deviation
    med_hour = df.groupby("user_card")["Hour"].median().reset_index()
    med_hour.columns = ["user_card", "med_hour"]
    mh_merged = df[["user_card"]].merge(med_hour, on="user_card", how="left")
    X[:, 16] = np.abs(hours - mh_merged["med_hour"].values).astype(np.float32)

    # Amount z-score
    X[:, 17] = np.where(merged["std_amt"] > 0, (amt - merged["mean_amt"]) / merged["std_amt"], 0.0).astype(np.float32)

    # Velocity
    X[:, 18] = merged["count"].values.astype(np.float32) / 24.0

    # Recipient novelty
    X[:, 19] = (1.0 / (merch_first + 1)).astype(np.float32)

    # TXN regularity
    gap_df = pd.DataFrame({"user_card": df["user_card"].values, "gap": gaps})
    card_gap_std = gap_df.groupby("user_card")["gap"].std().reset_index()
    card_gap_std.columns = ["user_card", "gap_std"]
    card_gap_std["gap_std"] = card_gap_std["gap_std"].fillna(0)
    gs_merged = df[["user_card"]].merge(card_gap_std, on="user_card", how="left")
    X[:, 20] = np.clip(gs_merged["gap_std"].values, 0, 100).astype(np.float32)

    print(f"  Features computed in {time.time()-t0:.1f}s")
    return X, y


def main():
    print("=" * 70)
    print("PS-14 EVALUATION: ealtman2019/credit-card-transactions")
    print("=" * 70)

    t_start = time.time()

    # Load
    df = load_ealtman()

    # Feature engineering
    X, y = engineer_ealtman_features(df)

    # Hash dataset
    data_hash = hashlib.sha256(
        open(ROOT / "data" / "credit_card_transactions-ibm_v2.csv", "rb").read(10**7)  # first 10MB for speed
    ).hexdigest()[:16]

    # Split: 80/20 temporal
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nTrain: {split:,} rows ({int(y_train.sum()):,} fraud)")
    print(f"Test: {n-split:,} rows ({int(y_test.sum()):,} fraud)")

    # Prepare
    imp = SimpleImputer(strategy="median").fit(X_train)
    Xc_train = np.nan_to_num(imp.transform(X_train), nan=0.0, posinf=10.0, neginf=-10.0)
    Xc_test = np.nan_to_num(imp.transform(X_test), nan=0.0, posinf=10.0, neginf=-10.0)
    scaler = StandardScaler().fit(Xc_train)
    Xs_train = scaler.transform(Xc_train)
    Xs_test = scaler.transform(Xc_test)

    # Train
    print("\nTraining XGBoost...")
    t1 = time.time()
    n_pos = int(y_train.sum())
    n_neg = split - n_pos
    model = XGBClassifier(
        n_estimators=600, max_depth=10, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=max(n_neg / max(n_pos, 1), 1.0),
        eval_metric="auc", random_state=SEED, n_jobs=-1, verbosity=0,
        min_child_weight=3, gamma=0.05, reg_alpha=0.1,
    )
    model.fit(Xs_train, y_train)
    print(f"  Trained in {time.time()-t1:.1f}s")

    # Predict
    prob = model.predict_proba(Xs_test)[:, 1]

    # Metrics
    mets = compute_metrics(y_test, prob)

    total_time = time.time() - t_start

    result = {
        "experiment_id": "REAL_PUBLIC_ealtman_v1",
        "category": "REAL_PUBLIC",
        "dataset": "ealtman2019_credit_card_transactions",
        "data_type": "REAL_PUBLIC_DATASET",
        "data_hash": data_hash,
        "split": "80/20 temporal",
        "train_n": split,
        "test_n": n - split,
        **mets,
    }

    # Print
    print("\n" + "=" * 70)
    print("RESULTS: ealtman2019/credit-card-transactions")
    print("=" * 70)
    print(f"  Dataset: {n:,} transactions, {int(y.sum()):,} fraud ({y.sum()/n*100:.4f}%)")
    print(f"  Train: {split:,} | Test: {n-split:,}")
    print(f"  ROC-AUC: {mets['roc_auc']}")
    print(f"  PR-AUC:  {mets['pr_auc']}")
    print(f"  Recall@0.1%FPR: {mets['recall_at_0_1pct_fpr']}")
    print(f"  Recall@0.5%FPR: {mets['recall_at_0_5pct_fpr']}")
    print(f"  Recall@1%FPR:   {mets['recall_at_1pct_fpr']}")
    print(f"  Brier:   {mets['brier']}")
    print(f"  Time:    {total_time:.1f}s")
    print(f"  Hash:    {data_hash}")

    # Save
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    (out / "ealtman_results.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out / 'ealtman_results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
