#!/usr/bin/env python3
"""Split ealtman2019 into 300K chunks and evaluate on each.

Strategy:
- Load full 24.4M dataset
- Split sequentially into ~300K chunks
- Train on chunk 0 (first 300K)
- Evaluate on every subsequent chunk
- Report per-chunk metrics to measure consistency
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = DATA / "real_data_evaluation"
RESULTS.mkdir(exist_ok=True)

ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "hour_of_day", "is_weekend", "shared_device_accounts",
    "shared_recipient_accounts", "mule_ring_score", "hour_deviation",
    "amount_zscore", "velocity_deviation", "recipient_novelty", "txn_regularity",
]
N_F = len(ML_FEATURES)
CHUNK_SIZE = 300_000


def map_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map ealtman2019 chunk to PS-14 features."""
    n = len(df)
    rng = np.random.RandomState(42)
    features = np.zeros((n, N_F))

    amt_str = df["Amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    amt = pd.to_numeric(amt_str, errors="coerce").fillna(0).values.astype(float)
    amt_mean = max(amt.mean(), 1)

    hour_raw = pd.to_numeric(df["Time"].str.split(":").str[0], errors="coerce").fillna(12).values.astype(float)
    labels = (df["Is Fraud?"] == "Yes").astype(int).values

    dates = pd.to_datetime(df[["Year", "Month", "Day"]].rename(columns={"Year": "year", "Month": "month", "Day": "day"}))
    dow = dates.dt.dayofweek.values
    doy = dates.dt.dayofyear.values

    features[:, ML_FEATURES.index("hour_of_day")] = hour_raw
    features[:, ML_FEATURES.index("is_weekend")] = (dow >= 5).astype(float)
    features[:, ML_FEATURES.index("amount_ratio")] = amt / amt_mean

    chip = df["Use Chip"].fillna("Unknown").values
    features[:, ML_FEATURES.index("new_device_flag")] = (chip == "Online Transaction").astype(float)
    features[:, ML_FEATURES.index("unusual_location_flag")] = (chip == "Online Transaction").astype(float)

    user_city_count = df.groupby("User")["Merchant City"].transform("nunique").values
    features[:, ML_FEATURES.index("unusual_recipient_flag")] = (user_city_count > np.percentile(user_city_count, 90)).astype(float)

    user_tx_count = df.groupby("User").cumcount().values
    features[:, ML_FEATURES.index("txn_freq_last_24h")] = np.clip(user_tx_count / max(np.median(user_tx_count), 1), 0, 50)
    features[:, ML_FEATURES.index("txn_time_unusual")] = hour_raw / 24.0

    mcc = pd.to_numeric(df["MCC"], errors="coerce").fillna(0).values
    user_mcc_mode = df.groupby("User")["MCC"].transform(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0).values
    features[:, ML_FEATURES.index("failed_auth_count_24h")] = (mcc != user_mcc_mode).astype(float)

    features[:, ML_FEATURES.index("days_since_last_similar_txn")] = np.clip(doy / 365, 0, 10)

    user_amt_max = df.groupby("User")["Amount"].apply(
        lambda x: pd.to_numeric(x.str.replace("$", "").str.replace(",", ""), errors="coerce").expanding().max()
    ).values
    features[:, ML_FEATURES.index("gradual_escalation_score")] = amt / np.maximum(user_amt_max, 1)

    features[:, ML_FEATURES.index("known_device_count")] = df.groupby("User")["Card"].transform("nunique").values
    features[:, ML_FEATURES.index("account_tenure_days")] = doy.clip(0, 3650)

    user_city_seen = df.groupby("User")["Merchant City"].transform(lambda x: pd.factorize(x)[0]).values
    features[:, ML_FEATURES.index("unusual_recipient_flag")] = np.maximum(
        features[:, ML_FEATURES.index("unusual_recipient_flag")], (user_city_seen < 2).astype(float)
    )

    features[:, ML_FEATURES.index("shared_device_accounts")] = rng.poisson(0.1, n).astype(float)
    features[:, ML_FEATURES.index("shared_recipient_accounts")] = rng.poisson(0.2, n).astype(float)
    features[:, ML_FEATURES.index("mule_ring_score")] = (user_city_count > np.percentile(user_city_count, 95)).astype(float) * 2

    features[:, ML_FEATURES.index("hour_deviation")] = np.abs(hour_raw - 14) / 14
    features[:, ML_FEATURES.index("amount_zscore")] = (amt - amt_mean) / max(amt.std(), 1e-6)
    features[:, ML_FEATURES.index("velocity_deviation")] = np.clip(user_tx_count / max(np.median(user_tx_count), 1), 0, 5)
    features[:, ML_FEATURES.index("recipient_novelty")] = np.clip(user_city_seen / max(user_city_seen.max(), 1), 0, 1)
    features[:, ML_FEATURES.index("txn_regularity")] = rng.uniform(0, 1, n)

    return features, labels


def chunk_metrics(y_true, y_prob):
    n = len(y_true)
    n_fraud = int(y_true.sum())
    n_legit = n - n_fraud
    if n_fraud == 0 or n_legit == 0:
        return {"n": n, "fraud": n_fraud, "roc_auc": None, "pr_auc": None, "r1": None}

    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(fpr_arr, 0.01, side="left")
    r1 = float(tpr_arr[min(idx, len(tpr_arr) - 1)])
    brier = brier_score_loss(y_true, y_prob)
    return {
        "n": n, "fraud": n_fraud, "prev": round(n_fraud / n, 4),
        "roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
        "r1_fpr": round(r1, 4), "brier": round(brier, 4),
    }


def main():
    csv_path = DATA / "ealtman2019" / "credit_card_transactions-ibm_v2.csv"
    print(f"Loading {csv_path.name}...")

    # Count total first
    print("  Counting rows...")
    total = 0
    for chunk in pd.read_csv(csv_path, usecols=["Is Fraud?"], chunksize=1_000_000):
        total += len(chunk)
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  Total: {total:,} rows -> {n_chunks} chunks of {CHUNK_SIZE:,}")

    # Load all data into memory (24M rows ~ 2.2GB CSV → ~1GB DataFrame)
    print("  Loading full dataset...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Split into chunks
    print(f"\nSplitting into {n_chunks} chunks of {CHUNK_SIZE:,}...")
    chunks = []
    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = min((i + 1) * CHUNK_SIZE, len(df))
        chunks.append(df.iloc[start:end].copy())
        fraud = chunks[-1]["Is Fraud?"].value_counts().get("Yes", 0)
        print(f"  Chunk {i+1:2d}: rows {start:>10,}-{end:>10,}  fraud={fraud:,}")

    # Map features for all chunks
    print(f"\nMapping features for {n_chunks} chunks...")
    t0 = time.time()
    chunk_data = []
    for i, chunk_df in enumerate(chunks):
        X, y = map_features(chunk_df)
        chunk_data.append((X, y))
        print(f"  Chunk {i+1:2d}: {len(y):,} rows, {int(y.sum()):,} fraud ({time.time()-t0:.0f}s)")

    del df  # free memory

    # Train on chunk 0
    print(f"\nTraining on chunk 0 ({len(chunk_data[0][1]):,} rows)...")
    t0 = time.time()
    X_tr, y_tr = chunk_data[0]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    lr.fit(X_tr_s, y_tr)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr_s, y_tr)

    try:
        from xgboost import XGBClassifier
        n_fraud = int(y_tr.sum())
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            scale_pos_weight=max(n_fraud / max(len(y_tr) - n_fraud, 1), 1),
                            random_state=42, eval_metric="aucpr", verbosity=0,
                            tree_method="hist", n_jobs=-1)
        xgb.fit(X_tr_s, y_tr)
        has_xgb = True
    except ImportError:
        has_xgb = False

    print(f"  Trained in {time.time()-t0:.1f}s")

    # Evaluate on all chunks
    print(f"\nEvaluating on {n_chunks} chunks...")
    results = []
    for i, (X_te, y_te) in enumerate(chunk_data):
        if int(y_te.sum()) < 5:
            print(f"  Chunk {i+1:2d}: SKIP (only {int(y_te.sum())} fraud)")
            results.append({"chunk": i + 1, "status": "skip"})
            continue

        X_te_s = scaler.transform(X_te)
        lr_scores = lr.predict_proba(X_te_s)[:, 1]
        rf_scores = rf.predict_proba(X_te_s)[:, 1]
        if has_xgb:
            xgb_scores = xgb.predict_proba(X_te_s)[:, 1]
        else:
            xgb_scores = (lr_scores + rf_scores) / 2

        # Stacker: average of 3 models (simple, no val set needed)
        fused = (lr_scores + rf_scores + xgb_scores) / 3

        lr_m = chunk_metrics(y_te, lr_scores)
        rf_m = chunk_metrics(y_te, rf_scores)
        xgb_m = chunk_metrics(y_te, xgb_scores)
        fuse_m = chunk_metrics(y_te, fused)

        r = {
            "chunk": i + 1,
            "rows": len(y_te),
            "fraud": int(y_te.sum()),
            "lr": lr_m, "rf": rf_m, "xgb": xgb_m, "fused": fuse_m,
        }
        results.append(r)

        roc = fuse_m.get("roc_auc", "?")
        r1 = fuse_m.get("r1_fpr", "?")
        pr = fuse_m.get("pr_auc", "?")
        label = "TRAIN" if i == 0 else f"TEST "
        print(f"  Chunk {i+1:2d} [{label}]  ROC={roc}  PR={pr}  R@1%={r1}  fraud={fuse_m.get('fraud',0):,}")

    # Aggregate stats across test chunks (exclude chunk 0 = training)
    test_chunks = [r for r in results if r.get("chunk", 0) > 1 and "fused" in r]
    if test_chunks:
        rocs = [r["fused"]["roc_auc"] for r in test_chunks if r["fused"].get("roc_auc")]
        prs = [r["fused"]["pr_auc"] for r in test_chunks if r["fused"].get("pr_auc")]
        r1s = [r["fused"]["r1_fpr"] for r in test_chunks if r["fused"].get("r1_fpr")]

        print(f"\n{'='*70}")
        print(f"AGGREGATE ACROSS {len(test_chunks)} TEST CHUNKS")
        print(f"{'='*70}")
        print(f"  ROC-AUC:  mean={np.mean(rocs):.4f}  std={np.std(rocs):.4f}  min={np.min(rocs):.4f}  max={np.max(rocs):.4f}")
        print(f"  PR-AUC:   mean={np.mean(prs):.4f}  std={np.std(prs):.4f}  min={np.min(prs):.4f}  max={np.max(prs):.4f}")
        print(f"  R@1%FPR:  mean={np.mean(r1s):.4f}  std={np.std(r1s):.4f}  min={np.min(r1s):.4f}  max={np.max(r1s):.4f}")

    # Save
    out = RESULTS / "ealtman2019_chunks.json"
    with open(out, "w") as f:
        json.dump({"total_rows": total, "chunk_size": CHUNK_SIZE, "n_chunks": n_chunks, "results": results}, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
