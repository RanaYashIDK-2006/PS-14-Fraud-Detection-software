#!/usr/bin/env python3
"""Generate a realistic validation dataset for ML evaluation.

Creates a more diverse set of fraud patterns than the synthetic training
data, including:
- New fraud archetypes not in training
- Edge cases (tiny amounts, huge amounts, rare time patterns)
- Mimicry attacks (fraud that looks like legit behavior)
- Coordinated multi-account attacks
- Synthetic identity fraud (new accounts with suspicious patterns)

This is NOT real-world data — it's a more challenging synthetic benchmark
that tests generalization beyond the training archetypes.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


# Archetypes not in the training set
OOD_ARCHETYPES = [
    "mimicry_small",          # Small amounts that look legit but are rapid
    "velocity_slow_burn",     # Slow accumulation over weeks
    "device_sharing_ring",    # Multiple accounts on same device
    "recipient_farming",      # Many small txns to new recipients
    "temporal_anomaly",       # Transactions at unusual times
    "amount_laundering",      # Just-under-threshold amounts
    "geographic_impossible",  # Impossible travel speeds
    "session_hijack",         # Auth anomaly without device change
]


def generate_mimicry_small(rng: np.random.Generator, n: int = 20) -> pd.DataFrame:
    """Small amounts that look like normal transactions but are rapid."""
    rows = []
    for _ in range(n):
        base_amount = rng.uniform(5, 50)
        for i in range(rng.integers(3, 8)):
            rows.append({
                "txn_amount_bucket": rng.choice(["small", "medium"]),
                "amount_ratio": rng.uniform(0.3, 1.2),
                "txn_freq_last_24h": i + 1,
                "txn_time_unusual": 0,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 1 if i > 2 else 0,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(0, 1),
                "shared_device_accounts": 1,
                "shared_recipient_accounts": rng.integers(1, 3),
                "mule_ring_score": 0.0,
                "account_tenure_days": rng.integers(30, 365),
                "account_daily_spend_ratio": rng.uniform(0.5, 1.5),
                "device_daily_count": i + 1,
                "hour_of_day": rng.integers(8, 20),
                "label": 1,
                "archetype": "mimicry_small",
            })
    return pd.DataFrame(rows)


def generate_velocity_slow_burn(rng: np.random.Generator, n: int = 15) -> pd.DataFrame:
    """Slow accumulation over weeks — hard to detect."""
    rows = []
    for _ in range(n):
        for day in range(rng.integers(14, 30)):
            rows.append({
                "txn_amount_bucket": rng.choice(["small", "medium"]),
                "amount_ratio": rng.uniform(0.4, 1.0),
                "txn_freq_last_24h": rng.integers(1, 3),
                "txn_time_unusual": 0,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 0,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": 1,
                "shared_device_accounts": 1,
                "shared_recipient_accounts": rng.integers(1, 2),
                "mule_ring_score": 0.0,
                "account_tenure_days": 60 + day,
                "account_daily_spend_ratio": rng.uniform(0.3, 0.8),
                "device_daily_count": 1,
                "hour_of_day": rng.integers(9, 18),
                "label": 1,
                "archetype": "velocity_slow_burn",
            })
    return pd.DataFrame(rows)


def generate_device_sharing_ring(rng: np.random.Generator, n: int = 12) -> pd.DataFrame:
    """Multiple accounts sharing a device — coordinated attack."""
    rows = []
    n_accounts = rng.integers(3, 6)
    for acct in range(n_accounts):
        for _ in range(rng.integers(2, 5)):
            rows.append({
                "txn_amount_bucket": rng.choice(["medium", "large"]),
                "amount_ratio": rng.uniform(0.8, 2.0),
                "txn_freq_last_24h": rng.integers(1, 4),
                "txn_time_unusual": 0,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(0, 2),
                "shared_device_accounts": n_accounts,
                "shared_recipient_accounts": rng.integers(2, 5),
                "mule_ring_score": rng.uniform(0.3, 0.7),
                "account_tenure_days": rng.integers(7, 30),
                "account_daily_spend_ratio": rng.uniform(0.5, 1.5),
                "device_daily_count": rng.integers(3, 8),
                "hour_of_day": rng.integers(8, 22),
                "label": 1,
                "archetype": "device_sharing_ring",
            })
    return pd.DataFrame(rows)


def generate_amount_laundering(rng: np.random.Generator, n: int = 18) -> pd.DataFrame:
    """Just-under-threshold amounts to avoid velocity limits."""
    rows = []
    for _ in range(n):
        n_txns = rng.integers(5, 10)
        for i in range(n_txns):
            rows.append({
                "txn_amount_bucket": "small",
                "amount_ratio": rng.uniform(0.2, 0.9),
                "txn_freq_last_24h": i + 1,
                "txn_time_unusual": 0,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(0, 0.5),
                "shared_device_accounts": 1,
                "shared_recipient_accounts": rng.integers(1, 3),
                "mule_ring_score": rng.uniform(0.0, 0.3),
                "account_tenure_days": rng.integers(14, 90),
                "account_daily_spend_ratio": rng.uniform(0.8, 2.5),
                "device_daily_count": 1,
                "hour_of_day": rng.integers(10, 22),
                "label": 1,
                "archetype": "amount_laundering",
            })
    return pd.DataFrame(rows)


def generate_geographic_impossible(rng: np.random.Generator, n: int = 10) -> pd.DataFrame:
    """Transactions that require impossible travel speeds."""
    rows = []
    for _ in range(n):
        # Two transactions minutes apart in different locations
        for loc in [0, 1]:
            rows.append({
                "txn_amount_bucket": rng.choice(["medium", "large"]),
                "amount_ratio": rng.uniform(1.0, 3.0),
                "txn_freq_last_24h": 1,
                "txn_time_unusual": 1,
                "new_device_flag": 0,
                "unusual_location_flag": 1,
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": 0,
                "shared_device_accounts": 1,
                "shared_recipient_accounts": 1,
                "mule_ring_score": 0.0,
                "account_tenure_days": rng.integers(30, 180),
                "account_daily_spend_ratio": rng.uniform(1.0, 3.0),
                "device_daily_count": 1,
                "hour_of_day": rng.integers(0, 24),
                "label": 1,
                "archetype": "geographic_impossible",
            })
    return pd.DataFrame(rows)


# Also generate legit transactions for FPR measurement
def generate_legit_edge_cases(rng: np.random.Generator, n: int = 50) -> pd.DataFrame:
    """Legitimate edge cases that should NOT be flagged."""
    rows = []
    for _ in range(n):
        rows.append({
            "txn_amount_bucket": rng.choice(["small", "medium", "large"]),
            "amount_ratio": rng.uniform(0.1, 3.0),
            "txn_freq_last_24h": rng.integers(0, 8),
            "txn_time_unusual": rng.choice([0, 0, 0, 1]),
            "new_device_flag": rng.choice([0, 0, 0, 1]),
            "unusual_location_flag": rng.choice([0, 0, 1]),
            "unusual_recipient_flag": rng.choice([0, 0, 1]),
            "failed_auth_count_24h": rng.integers(0, 3),
            "days_since_last_similar_txn": rng.uniform(0, 7),
            "shared_device_accounts": rng.integers(1, 3),
            "shared_recipient_accounts": rng.integers(1, 3),
            "mule_ring_score": 0.0,
            "account_tenure_days": rng.integers(7, 365),
            "account_daily_spend_ratio": rng.uniform(0.2, 2.0),
            "device_daily_count": rng.integers(1, 5),
            "hour_of_day": rng.integers(6, 23),
            "label": 0,
            "archetype": "legit_edge_case",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic validation dataset")
    parser.add_argument("--n", type=int, default=1, help="Multiply each archetype by N")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="data/validation_realistic.csv", help="Output path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    generators = [
        ("mimicry_small", generate_mimicry_small),
        ("velocity_slow_burn", generate_velocity_slow_burn),
        ("device_sharing_ring", generate_device_sharing_ring),
        ("amount_laundering", generate_amount_laundering),
        ("geographic_impossible", generate_geographic_impossible),
    ]

    frames = []
    for name, gen in generators:
        for _ in range(args.n):
            df = gen(n=20, rng=rng)
            frames.append(df)
            print(f"  {name}: {len(df)} rows")

    # Add legit edge cases for FPR measurement
    legit = generate_legit_edge_cases(n=100, rng=rng)
    frames.append(legit)
    print(f"  legit_edge_case: {len(legit)} rows")

    df = pd.concat(frames, ignore_index=True)

    # Add event_id and ts columns
    df["event_id"] = [f"ood-{uuid.uuid4().hex[:8]}" for _ in range(len(df))]
    df["ts"] = pd.date_range("2026-01-01", periods=len(df), freq="1min")

    # Reorder columns
    feature_cols = [
        "event_id", "ts", "txn_amount_bucket", "amount_ratio",
        "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
        "unusual_location_flag", "unusual_recipient_flag",
        "failed_auth_count_24h", "days_since_last_similar_txn",
        "shared_device_accounts", "shared_recipient_accounts",
        "mule_ring_score", "account_tenure_days",
        "account_daily_spend_ratio", "device_daily_count",
        "hour_of_day", "label", "archetype",
    ]
    df = df[feature_cols]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    fraud = df[df["label"] == 1]
    legit_count = df[df["label"] == 0]
    print(f"\nTotal: {len(df)} rows ({len(fraud)} fraud, {len(legit_count)} legit)")
    print(f"Archetypes: {df['archetype'].nunique()}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
