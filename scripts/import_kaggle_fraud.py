#!/usr/bin/env python3
"""Import the Kaggle ULB Credit Card Fraud dataset and map to PS-14 §16 features.

The Kaggle dataset (284,807 transactions, 492 frauds, 0.172% prevalence)
contains PCA-transformed features (V1-V28) plus Time and Amount. We map
these to PS-14's §16 behavioral features using heuristic rules, then
evaluate the trained models against this external benchmark.

License: Dataset is under CC BY 4.0 (Kaggle) / academic use (original paper).
Citation: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca
Bontempi. "Calibrating Probability with Undersampling for Unbalanced
Classification." IEEE Symposium on Computational Intelligence in Data
Mining (CIDM), 2015.

Mapping strategy:
- amount_ratio: Amount / median Amount (per-class normalization)
- txn_freq_last_24h: derived from Time gaps between consecutive transactions
- txn_time_unusual: hour-of-day from Time (night hours = unusual)
- new_device_flag: V14 as proxy (strong fraud discriminator)
- unusual_location_flag: V4 as proxy (location-like PCA component)
- unusual_recipient_flag: V12 as proxy (recipient-like PCA component)
- failed_auth_count_24h: 0 (not in dataset)
- days_since_last_similar_txn: derived from Time gaps
- shared_device_accounts: 1 (default, not in dataset)
- shared_recipient_accounts: 1 (default, not in dataset)
- mule_ring_score: 0 (not in dataset)
- account_tenure_days: derived from Time position in dataset
- account_daily_spend_ratio: Amount / median Amount
- device_daily_count: derived from Time-windowed count
- hour_of_day: derived from Time

This mapping is HEURISTIC — it's designed to test whether the model
generalizes to a different data distribution, not to be perfectly accurate.
"""

from __future__ import annotations

import argparse
import hashlib
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


def map_to_ps14_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map Kaggle features to PS-14 §16 feature format."""
    print(f"Mapping {len(df)} rows...")

    # Normalize Time to hours (dataset Time is seconds from first transaction)
    df["hours"] = df["Time"] / 3600.0
    df["hour_of_day"] = (df["hours"] % 24).astype(int)

    # Sort by time for gap calculations
    df = df.sort_values("Time").reset_index(drop=True)

    # Compute time gaps between consecutive transactions
    df["time_gap"] = df["Time"].diff().fillna(0).clip(lower=0)

    # ── PS-14 §16 features ──

    # amount_ratio: Amount / median Amount (behavioral deviation)
    median_amount = df["Amount"].median()
    df["amount_ratio"] = (df["Amount"] / max(median_amount, 0.01)).clip(0, 10)

    # txn_freq_last_24h: count of transactions in rolling 24h window
    # Use a simplified approach: count transactions with time_gap < 86400
    df["txn_freq_last_24h"] = 0
    window_24h = 86400  # seconds
    for i in range(len(df)):
        start = df["Time"].iloc[i] - window_24h
        count = ((df["Time"] >= start) & (df["Time"] <= df["Time"].iloc[i])).sum()
        df.loc[df.index[i], "txn_freq_last_24h"] = min(count - 1, 20)  # cap at 20

    # txn_time_unusual: 1 if hour is between 0-5 or 22-23
    df["txn_time_unusual"] = df["hour_of_day"].apply(
        lambda h: 1 if h < 6 or h >= 22 else 0
    )

    # new_device_flag: use V14 as proxy (strong fraud signal in PCA space)
    # If V14 is more than 2 std dev from mean, flag as unusual
    v14_mean = df["V14"].mean()
    v14_std = df["V14"].std()
    df["new_device_flag"] = (np.abs(df["V14"] - v14_mean) > 2 * v14_std).astype(int)

    # unusual_location_flag: use V4 as proxy
    v4_mean = df["V4"].mean()
    v4_std = df["V4"].std()
    df["unusual_location_flag"] = (np.abs(df["V4"] - v4_mean) > 2 * v4_std).astype(int)

    # unusual_recipient_flag: use V12 as proxy
    v12_mean = df["V12"].mean()
    v12_std = df["V12"].std()
    df["unusual_recipient_flag"] = (np.abs(df["V12"] - v12_mean) > 2 * v12_std).astype(int)

    # failed_auth_count_24h: not in dataset, default to 0
    df["failed_auth_count_24h"] = 0

    # days_since_last_similar_txn: time gap in days
    df["days_since_last_similar_txn"] = (df["time_gap"] / 86400).clip(0, 30)

    # shared_device_accounts: not in dataset, default to 1
    df["shared_device_accounts"] = 1

    # shared_recipient_accounts: not in dataset, default to 1
    df["shared_recipient_accounts"] = 1

    # mule_ring_score: not in dataset, default to 0
    df["mule_ring_score"] = 0.0

    # account_tenure_days: position in dataset (normalized to days)
    total_time_hours = df["Time"].max() / 3600
    total_days = max(total_time_hours / 24, 1)
    df["account_tenure_days"] = ((df["Time"] / df["Time"].max()) * 90 + 7).astype(int)

    # account_daily_spend_ratio: Amount / median daily spend
    median_daily = df["Amount"].sum() / total_days
    df["account_daily_spend_ratio"] = (df["Amount"] / max(median_daily, 0.01)).clip(0, 10)

    # device_daily_count: transactions per device in 24h window
    df["device_daily_count"] = df["txn_freq_last_24h"].clip(0, 20)

    # hour_of_day (already computed)

    # Labels
    df["label"] = df["Class"].astype(int)

    # Archetype
    df["archetype"] = df["label"].map({0: "legit_kaggle", 1: "fraud_kaggle"})

    # event_id
    df["event_id"] = [f"kaggle-{uuid.uuid4().hex[:8]}" for _ in range(len(df))]

    # ts
    df["ts"] = pd.to_datetime(df["Time"], unit="s", origin="2026-01-01")

    # Select PS-14 columns
    ps14_cols = [
        "event_id", "ts",
        "txn_amount_bucket", "amount_ratio",
        "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
        "unusual_location_flag", "unusual_recipient_flag",
        "failed_auth_count_24h", "days_since_last_similar_txn",
        "shared_device_accounts", "shared_recipient_accounts",
        "mule_ring_score", "account_tenure_days",
        "account_daily_spend_ratio", "device_daily_count",
        "hour_of_day", "label", "archetype",
    ]

    # Create txn_amount_bucket from Amount
    df["txn_amount_bucket"] = pd.cut(
        df["Amount"],
        bins=[0, 10, 50, 200, float("inf")],
        labels=["small", "medium", "large", "extreme"],
    ).astype(str)

    result = df[ps14_cols].copy()

    fraud_count = (result["label"] == 1).sum()
    legit_count = (result["label"] == 0).sum()
    print(f"  Mapped: {len(result)} rows ({fraud_count} fraud, {legit_count} legit)")
    print(f"  Fraud rate: {fraud_count/len(result)*100:.3f}%")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Kaggle Credit Card Fraud dataset")
    parser.add_argument("--input", default="data/creditcard.csv", help="Input CSV")
    parser.add_argument("--output", default="data/validation_kaggle.csv", help="Output CSV")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows (0=all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  kaggle datasets download -d mlg-ulb/creditcardfraud")
        print(f"  Extract to {input_path}")
        return

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Raw: {len(df)} rows, {df.columns.tolist()}")

    if args.max_rows > 0:
        df = df.head(args.max_rows)
        print(f"  Trimmed to {len(df)} rows")

    # Map to PS-14 format
    result = map_to_ps14_features(df)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    # Print summary statistics
    print("\n=== Feature Statistics ===")
    for col in ["amount_ratio", "txn_freq_last_24h", "hour_of_day",
                 "new_device_flag", "unusual_location_flag",
                 "account_tenure_days", "account_daily_spend_ratio"]:
        fraud_mean = result[result["label"] == 1][col].mean()
        legit_mean = result[result["label"] == 0][col].mean()
        print(f"  {col:30s} fraud={fraud_mean:8.3f}  legit={legit_mean:8.3f}")


if __name__ == "__main__":
    main()
