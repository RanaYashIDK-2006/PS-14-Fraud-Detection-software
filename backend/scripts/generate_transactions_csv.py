#!/usr/bin/env python3
"""Generate data/transactions.csv with ML_FEATURES columns from kaggle_fraud.

The rules gate and rules simulator expect columns matching ML_FEATURES
(amount_ratio, txn_freq_last_24h, etc.) plus a 'label' column.
This script reads kaggle_fraud, computes the features, and writes the CSV.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DATA_DIR = ROOT / "data"
OUT = DATA_DIR / "transactions.csv"


def generate():
    print("Loading kaggle_fraud training data...")
    df = pd.read_csv(DATA_DIR / "kaggle_fraud" / "fraudTrain.csv", nrows=200_000)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour_of_day"] = df["trans_date_trans_time"].dt.hour
    df["is_weekend"] = df["trans_date_trans_time"].dt.dayofweek.isin([5, 6]).astype(int)
    df["unix"] = df["unix_time"].values.astype(float)

    amt = df["amt"].values.astype(float)
    hours = df["hour_of_day"].values.astype(float)

    # Per-card stats
    card_group = df.groupby("cc_num")
    card_amt_median = card_group["amt"].transform("median")
    card_amt_std = card_group["amt"].transform("std").fillna(0)
    card_count = card_group.cumcount() + 1

    # Feature 1: amount_ratio
    amount_ratio = amt / np.maximum(card_amt_median.values, 1)

    # Feature 2: txn_freq_last_24h
    txn_freq = card_count.values.astype(float)

    # Feature 3: txn_time_unusual
    typical_hours = set(range(8, 22))
    txn_time_unusual = np.array([1 if int(h) not in typical_hours else 0 for h in hours], dtype=float)

    # Feature 4: new_device_flag (0 - no device info)
    new_device_flag = np.zeros(len(df))

    # Feature 5: unusual_location_flag
    lat = df["lat"].values.astype(float)
    lon = df["long"].values.astype(float)
    merch_lat = df["merch_lat"].values.astype(float)
    merch_long = df["merch_long"].values.astype(float)
    dlat = np.radians(merch_lat - lat)
    dlon = np.radians(merch_long - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(merch_lat)) * np.sin(dlon / 2) ** 2
    distance_km = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    unusual_location_flag = np.zeros(len(df))  # no baseline to compare

    # Feature 6: unusual_recipient_flag (first merchant for card)
    df["_merch_seen"] = df.groupby(["cc_num", "merchant"]).cumcount()
    unusual_recipient_flag = (df["_merch_seen"].values == 0).astype(float)

    # Feature 7: failed_auth_count_24h (no auth info)
    failed_auth = np.zeros(len(df))

    # Feature 8: days_since_last_similar_txn
    time_gaps = card_group["unix"].diff().fillna(86400).values / 3600
    days_since = np.clip(time_gaps / 24, 0, 30)

    # Feature 9: gradual_escalation_score
    hist_mean = card_group["amt"].transform("mean").values
    escalation = np.clip((amt - hist_mean) / np.maximum(hist_mean, 1), -1, 5)

    # Feature 10: known_device_count
    known_device_count = np.ones(len(df)) * 2

    # Feature 11: account_tenure_days
    account_tenure = card_count.values.astype(float) * 0.5

    # Feature 12: hour_of_day
    hour_of_day = hours

    # Feature 13: is_weekend
    is_weekend = df["is_weekend"].values.astype(float)

    # Feature 14: shared_device_accounts (0)
    shared_device = np.zeros(len(df))

    # Feature 15: shared_recipient_accounts (0)
    shared_recipient = np.zeros(len(df))

    # Feature 16: mule_ring_score
    mule_ring = np.zeros(len(df))

    # Feature 17: hour_deviation
    hour_deviation = np.abs(hours - 12) / 12.0

    # Feature 18: amount_zscore
    amount_zscore = (amt - card_amt_median.values) / np.maximum(card_amt_std.values, 0.01)

    # Feature 19: velocity_deviation
    velocity_deviation = np.clip(time_gaps / np.maximum(
        pd.Series(time_gaps).groupby(df["cc_num"].values).transform(lambda x: x.expanding().mean()).values, 0.1
    ), 0, 10)

    # Feature 20: recipient_novelty
    recipient_novelty = unusual_recipient_flag

    # Feature 21: txn_regularity
    txn_regularity = card_group["amt"].transform(lambda x: x.std() / max(x.mean(), 0.01)).values

    # Label
    label = df["is_fraud"].values.astype(int)

    # Build output DataFrame — include 'ts' column for temporal ordering
    out = pd.DataFrame({
        "ts": df["unix"].values.astype(float),
        "amount_ratio": np.round(amount_ratio, 4),
        "txn_freq_last_24h": txn_freq.astype(int),
        "txn_time_unusual": txn_time_unusual.astype(int),
        "new_device_flag": new_device_flag.astype(int),
        "unusual_location_flag": unusual_location_flag.astype(int),
        "unusual_recipient_flag": unusual_recipient_flag.astype(int),
        "failed_auth_count_24h": failed_auth.astype(int),
        "days_since_last_similar_txn": np.round(days_since, 2),
        "gradual_escalation_score": np.round(escalation, 4),
        "known_device_count": known_device_count.astype(int),
        "account_tenure_days": np.round(account_tenure, 2),
        "hour_of_day": hour_of_day.astype(int),
        "is_weekend": is_weekend.astype(int),
        "shared_device_accounts": shared_device.astype(int),
        "shared_recipient_accounts": shared_recipient.astype(int),
        "mule_ring_score": np.round(mule_ring, 4),
        "hour_deviation": np.round(hour_deviation, 4),
        "amount_zscore": np.round(amount_zscore, 4),
        "velocity_deviation": np.round(velocity_deviation, 4),
        "recipient_novelty": np.round(recipient_novelty, 4),
        "txn_regularity": np.round(txn_regularity, 4),
        "label": label,
    })

    # Clip NaN/inf
    out = out.replace([np.inf, -np.inf], 0).fillna(0)

    # Move label to last column
    cols = [c for c in out.columns if c != "label"] + ["label"]
    out = out[cols]
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out):,} rows to {OUT}")
    print(f"Fraud: {int(out['label'].sum()):,} ({out['label'].mean()*100:.3f}%)")
    print(f"Columns: {list(out.columns)}")


if __name__ == "__main__":
    generate()
