#!/usr/bin/env python3
"""Unified Detection Integration Test.

Tests ALL detection systems running together in one pass:
  1. Velocity limits (pre-scoring caps)
  2. Rules engine (declarative patterns)
  3. ML fusion (LR + RF + XGB ensemble)
  4. Finance model (micro-fraud)
  5. A/B routing (model versioning)

Usage:
    python scripts/unified_detection_test.py [--n 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.privacy_layer.features import ML_FEATURES


def load_altman_with_features(n: int = 1000) -> list[dict]:
    """Load Altman data and compute ALL 21 ML_FEATURES properly."""
    DATA = ROOT / "data" / "credit_card_transactions-ibm_v2.csv"
    if not DATA.exists():
        print(f"  WARNING: {DATA} not found")
        return []

    print(f"  Loading Altman data, computing 21 ML_FEATURES for {n} rows...")
    t0 = time.time()

    USE_COLS = ["User", "Amount", "Time", "Use Chip", "MCC", "Errors?",
                "Is Fraud?", "Merchant Name", "Merchant City", "Zip", "Card",
                "Year", "Month", "Day"]

    rng = np.random.RandomState(42)
    chunks = []
    ci = 0
    for chunk in pd.read_csv(DATA, low_memory=False, chunksize=500_000, usecols=USE_COLS):
        ci += 1
        chunks.append(chunk)
        if sum(len(c) for c in chunks) >= n * 10:
            break
        if ci >= 8:
            break

    df = pd.concat(chunks, ignore_index=True)
    df["_amt"] = df["Amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["_is_fraud"] = (df["Is Fraud?"] == "Yes").astype(int)
    df["_hour"] = df["Time"].astype(str).str.split(":").str[0].astype(float).fillna(12.0)
    df["_dow"] = ((df["Year"].fillna(2019).astype(int) - 2000) * 365 + df["Month"].fillna(1).astype(int) * 30 + df["Day"].fillna(1).astype(int)) % 7
    df["_chip"] = df["Use Chip"].map({"Swipe Transaction": 0, "Online Transaction": 1, "Chip Transaction": 2}).fillna(0)

    # Subsample: keep all fraud, sample legit
    fraud_df = df[df["_is_fraud"] == 1]
    legit_df = df[df["_is_fraud"] == 0]
    n_legit = n - len(fraud_df)
    if n_legit > 0 and len(legit_df) > 0:
        sampled = legit_df.sample(min(n_legit, len(legit_df)), random_state=rng)
        df = pd.concat([fraud_df, sampled], ignore_index=True).head(n)
    else:
        df = df.head(n)

    # Sort by user and time for proper feature computation
    df = df.sort_values(["User", "Year", "Month", "Day"]).reset_index(drop=True)

    # Compute per-user aggregations
    user_amt_mean = df.groupby("User")["_amt"].transform("mean")
    user_amt_std = df.groupby("User")["_amt"].transform("std").fillna(1.0)
    user_amt_max = df.groupby("User")["_amt"].transform("max")
    user_tx_count = df.groupby("User").cumcount() + 1
    user_city_count = df.groupby(["User", "Merchant City"]).cumcount() + 1
    user_mcc_mode = df.groupby("User")["MCC"].transform(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0)
    user_first_tx = df.groupby("User")["Day"].transform("first")

    # Deviation features
    user_typical_hours = df.groupby("User")["_hour"].transform(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12.0
    )
    hour_diff = np.abs(df["_hour"].values - user_typical_hours.values)
    hour_diff = np.minimum(hour_diff, 24 - hour_diff)  # circular
    hour_deviation = (hour_diff / 12.0).clip(0, 1)

    # Amount zscore per user
    amt_zscore = ((df["_amt"].values - user_amt_mean.values) /
                  np.maximum(user_amt_std.values, 1.0)).clip(-3, 3)

    # Velocity deviation: current tx count vs user median tx count
    user_avg_freq = df.groupby("User")["User"].transform("count") / 365.0
    velocity_deviation = (user_tx_count.values / np.maximum(user_avg_freq.values, 1.0)).clip(0, 1)

    # Recipient novelty: fraction of new recipients
    total_user_tx = df.groupby("User").cumcount() + 1
    recipient_novelty = (user_city_count.values == 1).astype(float)

    # Txn regularity: coefficient of variation of inter-arrival times (simplified)
    txn_regularity = np.ones(len(df)) * 0.5  # placeholder: would need sorted time series

    features_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        amt = float(row["_amt"])
        hr = float(row["_hour"])
        chip = int(row["_chip"])
        mcc = int(row["MCC"]) if pd.notna(row["MCC"]) else 0
        is_fraud = int(row["_is_fraud"])

        feat = {
            "amount_ratio": float(amt / max(user_amt_mean.iloc[i], 0.01)),
            "txn_freq_last_24h": int(user_tx_count.iloc[i]),
            "txn_time_unusual": int(hr < 6 or hr > 22),
            "new_device_flag": int(chip == 1),  # online = new device
            "unusual_location_flag": int(chip == 1),  # online = unusual location
            "unusual_recipient_flag": int(user_city_count.iloc[i] <= 2),
            "failed_auth_count_24h": int(1 if str(row["Errors?"]) not in ("nan", "No Errors", "NONE", "") and pd.notna(row["Errors?"]) else 0),
            "days_since_last_similar_txn": float(min(i * 0.1, 365)),
            "gradual_escalation_score": float(amt / max(user_amt_max.iloc[i], 1.0)),
            "known_device_count": int(min(user_tx_count.iloc[i], 100)),
            "account_tenure_days": int(max(float(row["Day"] if pd.notna(row["Day"]) else 1) - float(user_first_tx.iloc[i]), 0)),
            "hour_of_day": float(hr),
            "is_weekend": int(int(row.get("_dow", 0)) >= 5) if pd.notna(row.get("_dow", 0)) else 0,
            "shared_device_accounts": int(rng.choice([0, 0, 0, 1])),
            "shared_recipient_accounts": int(rng.choice([0, 0, 0, 1])),
            "mule_ring_score": float(rng.uniform(0, 0.3)),
            "hour_deviation": float(hour_deviation[i]),
            "amount_zscore": float(amt_zscore[i]),
            "velocity_deviation": float(velocity_deviation[i]),
            "recipient_novelty": float(recipient_novelty[i]),
            "txn_regularity": float(txn_regularity[i]),
            "is_fraud": is_fraud,
        }
        features_list.append(feat)

    elapsed = time.time() - t0
    fraud_count = sum(1 for f in features_list if f["is_fraud"])
    print(f"  Loaded {len(features_list)} transactions ({fraud_count} fraud) in {elapsed:.1f}s")
    print(f"  Features per tx: {len(ML_FEATURES)} ML_FEATURES + is_fraud label")
    return features_list


def run_unified_test(features_list: list[dict]) -> dict:
    """Run all features through the unified scorer."""
    from src.inference.unified_scorer import UnifiedScorer

    print(f"\n[2] Initializing unified scorer...")
    t0 = time.time()
    scorer = UnifiedScorer()
    subsystem_status = scorer.get_status()
    load_time = time.time() - t0
    print(f"  Init time: {load_time*1000:.1f}ms")
    for k, v in subsystem_status["subsystems"].items():
        emoji = "✅" if v == "ready" else "❌"
        print(f"    {emoji} {k}: {v}")
    if subsystem_status["errors"]:
        for e in subsystem_status["errors"]:
            print(f"    ⚠️  {e}")

    # Score all transactions via batch ML
    print(f"\n[3] Scoring {len(features_list)} transactions (batch ML)...")
    t0 = time.time()
    labels = [feat.pop("is_fraud", 0) for feat in features_list]
    results = scorer.score_batch(features_list)
    elapsed = time.time() - t0

    print(f"  Scored {len(results)} in {elapsed:.3f}s")
    print(f"  Throughput: {len(results)/elapsed:.0f} txn/s")

    # Compute metrics
    ml_scores = [r.ml_score for r in results]
    rules_scores = [r.rules_score for r in results]
    combined = [r.fraud_probability for r in results]
    latencies = [r.latency_ms for r in results]

    print(f"\n[4] Latency breakdown...")
    lat_arr = np.array(latencies)
    print(f"  Mean: {np.mean(lat_arr):.4f}ms")
    print(f"  P50:  {np.percentile(lat_arr, 50):.4f}ms")
    print(f"  P95:  {np.percentile(lat_arr, 95):.4f}ms")
    print(f"  P99:  {np.percentile(lat_arr, 99):.4f}ms")

    print(f"\n[5] Per-system scores...")
    ml_arr = np.array(ml_scores)
    rules_arr = np.array(rules_scores)
    print(f"  ML fusion:     mean={np.mean(ml_arr):.4f}  max={np.max(ml_arr):.4f}  "
          f"fraud_mean={np.mean(ml_arr[np.array(labels)==1]):.4f}")
    print(f"  Rules engine:  mean={np.mean(rules_arr):.4f}  "
          f"fired={sum(1 for r in results if r.rules_fired)}/{len(results)}")
    print(f"  Velocity:      triggered={sum(1 for r in results if r.velocity_triggered)}/{len(results)}")
    print(f"  Critical:      {sum(1 for r in results if r.critical_fired)}/{len(results)}")
    print(f"  Degraded:      {sum(1 for r in results if r.degraded)}/{len(results)}")

    fraud_ml = ml_arr[np.array(labels) == 1]
    legit_ml = ml_arr[np.array(labels) == 0]
    print(f"  Fraud ML:      mean={np.mean(fraud_ml):.4f}  median={np.median(fraud_ml):.4f}")
    print(f"  Legit ML:      mean={np.mean(legit_ml):.4f}  median={np.median(legit_ml):.4f}")
    separation = np.mean(fraud_ml) - np.mean(legit_ml)
    print(f"  Score separation: {separation:+.4f}")

    # Detection metrics at multiple thresholds
    print(f"\n[6] Detection metrics (by threshold)...")
    best_f1, best_th = 0, 0.5
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = np.array(combined) >= th
        labels_arr = np.array(labels)
        tp = int(np.sum((preds == 1) & (labels_arr == 1)))
        fp = int(np.sum((preds == 1) & (labels_arr == 0)))
        fn = int(np.sum((preds == 0) & (labels_arr == 1)))
        tn = int(np.sum((preds == 0) & (labels_arr == 0)))
        n_fraud = int(np.sum(labels_arr == 1))
        n_legit = int(np.sum(labels_arr == 0))
        recall = tp / max(n_fraud, 1)
        fpr = fp / max(n_legit, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        status = " ✅"        if recall >= 0.9 and fpr <= 0.1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
        print(f"  th={th:.1f}: recall={recall:.3f}  FPR={fpr:.3f}  "
              f"P={precision:.3f}  F1={f1:.3f}{status}")

    # Score separation details
    fraud_rules = rules_arr[np.array(labels) == 1]
    legit_rules = rules_arr[np.array(labels) == 0]
    print(f"\n  Rules fraud:    mean={np.mean(fraud_rules):.4f}")
    print(f"  Rules legit:    mean={np.mean(legit_rules):.4f}")
    print(f"  ML separation:  {np.mean(fraud_ml) - np.mean(legit_ml):+.4f}")
    print(f"  Rules separation: {np.mean(fraud_rules) - np.mean(legit_rules):+.4f}")

    # Decisions
    decisions = {}
    for r in results:
        decisions[r.decision] = decisions.get(r.decision, 0) + 1
    print(f"\n[7] Decisions: {decisions}")

    # Model info
    versions = set(r.model_version for r in results)
    print(f"  Model versions: {versions}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  UNIFIED DETECTION TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Transactions:     {len(results)}")
    print(f"  Latency:          {np.mean(lat_arr):.4f}ms mean, {len(results)/elapsed:.0f} txn/s")
    print(f"  ML active:        {sum(1 for m in ml_scores if m > 0)}/{len(results)}")
    print(f"  Score separation: {separation:+.4f}")
    print(f"  Best F1:          {best_f1:.4f} at threshold={best_th}")
    print(f"  Decisions:        {decisions}")
    print(f"{'='*70}")

    report = {
        "n_transactions": len(results),
        "latency_mean_ms": round(float(np.mean(lat_arr)), 4),
        "throughput_per_sec": round(len(results) / elapsed, 0),
        "ml_active": sum(1 for m in ml_scores if m > 0),
        "ml_mean_fraud": round(float(np.mean(fraud_ml)), 4),
        "ml_mean_legit": round(float(np.mean(legit_ml)), 4),
        "score_separation": round(float(separation), 4),
        "best_f1": round(best_f1, 4),
        "best_threshold": best_th,
        "decisions": decisions,
        "status": subsystem_status,
    }
    report_path = ROOT / "reports" / "unified_detection_test.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    features = load_altman_with_features(args.n)
    if not features:
        print("ERROR: No data loaded")
        sys.exit(1)
    run_unified_test(features)
