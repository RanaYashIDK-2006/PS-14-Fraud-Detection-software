#!/usr/bin/env python3
"""Import UCI Credit Card Default dataset and map to PS-14 §16 features.

Dataset: "Default of Credit Card Clients" (UCI ML Repository #350)
- 30,000 clients, 24 months of credit history
- Target: default payment next month (binary)
- Features: credit limit, payment history, bill amounts, payment amounts
- License: CC BY 4.0

Maps to PS-14 §16 features using heuristic rules based on payment patterns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DATA = ROOT / "data"

ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "hour_of_day", "is_weekend",
    "shared_device_accounts", "shared_recipient_accounts", "mule_ring_score",
]


def map_to_ps16(df: pd.DataFrame) -> pd.DataFrame:
    """Map UCI features to PS-14 §16 format.

    UCI columns:
    - LIMIT_BAL: credit limit
    - PAY_0..PAY_6: repayment status (-2=no consumption, -1=paid in full,
      0=revolving credit, 1=payment delay 1mo, ..., 9=payment delay 9mo+)
    - BILL_AMT1..6: bill statement amount
    - PAY_AMT1..6: previous payment amount
    - SEX, EDUCATION, MARRIAGE, AGE: demographics (we ignore for privacy)
    """
    print(f"Mapping {len(df)} rows from UCI Credit Card Default...")
    out = pd.DataFrame(index=df.index)

    # amount_ratio: ratio of recent payment to credit limit
    # Higher ratio = more responsible (lower fraud risk)
    recent_bill = df["BILL_AMT1"].clip(lower=1)
    recent_pay = df["PAY_AMT1"]
    out["amount_ratio"] = (recent_pay / recent_bill).clip(0, 10).round(4)

    # txn_freq_last_24h: count of months with payment > 0 in last 6 months
    pay_cols = [f"PAY_AMT{i}" for i in [1, 2, 3, 4, 5, 6]]
    out["txn_freq_last_24h"] = (df[pay_cols] > 0).sum(axis=1).astype(int)

    # txn_time_unusual: payment delay pattern
    # PAY_0 is most recent: 1 = delayed 1 month, 9 = delayed 9+ months
    out["txn_time_unusual"] = (df["PAY_0"] >= 2).astype(int)

    # new_device_flag: first-time delinquency
    pay_0 = df["PAY_0"]
    pay_prev = df[["PAY_2", "PAY_3", "PAY_4"]].max(axis=1)
    out["new_device_flag"] = ((pay_0 >= 1) & (pay_prev <= 0)).astype(int)

    # unusual_location_flag: bill amount spike (>2x credit limit)
    out["unusual_location_flag"] = (df["BILL_AMT1"] > df["LIMIT_BAL"] * 2).astype(int)

    # unusual_recipient_flag: payment pattern anomaly
    # Large gap between bill and payment
    out["unusual_recipient_flag"] = ((df["BILL_AMT1"] - df["PAY_AMT1"]) > df["LIMIT_BAL"] * 0.8).astype(int)

    # failed_auth_count_24h: count of severe delays (>= 2 months)
    delay_cols = [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
    out["failed_auth_count_24h"] = (df[delay_cols] >= 2).sum(axis=1).astype(int)

    # days_since_last_similar_txn: months since last good payment
    # Check PAY columns for 0 (paid in full) or -1 (paid full)
    paid_mask = df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] <= 0
    last_paid = paid_mask.idxmax(axis=1).str.extract(r'(\d+)')[0].astype(float)
    out["days_since_last_similar_txn"] = last_paid.fillna(6).astype(int)

    # gradual_escalation_score: payment deterioration over time
    recent_delays = (df["PAY_0"] >= 1).astype(int)
    older_delays = (df[["PAY_2", "PAY_3", "PAY_4"]].max(axis=1) >= 1).astype(int)
    out["gradual_escalation_score"] = (recent_delays * 0.5 + older_delays * 0.3 + (df["PAY_0"] / 9).clip(0, 0.2)).round(4)

    # known_device_count: credit limit tier (proxy for account maturity)
    out["known_device_count"] = pd.cut(df["LIMIT_BAL"], bins=[0, 50000, 100000, 200000, 1e9], labels=[1, 2, 3, 4]).astype(int)

    # account_tenure_days: months of credit history (approx)
    out["account_tenure_days"] = 24  # Dataset spans 24 months

    # hour_of_day: use bill/payment pattern as proxy
    out["hour_of_day"] = 12  # Neutral (no time data in this dataset)

    # is_weekend: payment pattern (proxy)
    out["is_weekend"] = 0

    # Link-analysis features (not available in this dataset)
    out["shared_device_accounts"] = 0
    out["shared_recipient_accounts"] = 0
    out["mule_ring_score"] = 0.0

    # Target: default payment next month
    out["label"] = df["default payment next month"].values

    return out[ML_FEATURES + ["label"]]


def main():
    # Load UCI dataset
    xlsx_path = DATA / "uci_temp" / "default of credit card clients.xls"
    if not xlsx_path.exists():
        print(f"ERROR: Dataset not found at {xlsx_path}")
        print("Download from: https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip")
        return

    print("Loading UCI Credit Card Default dataset...")
    df = pd.read_excel(xlsx_path, header=1)
    # First column is row ID, skip it
    df = df.iloc[:, 1:]  # Drop ID column
    print(f"  Raw: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Default rate: {df['default payment next month'].mean()*100:.2f}%")
    print(f"  Defaults: {df['default payment next month'].sum():,} / {len(df):,}")

    # Map to PS-16 features
    mapped = map_to_ps16(df)
    print(f"\n  Mapped: {mapped.shape[0]} rows, {mapped.shape[1]} columns")
    print(f"  Label distribution: {mapped['label'].value_counts().to_dict()}")

    # Save
    out_path = DATA / "validation_uci_default.csv"
    mapped.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")
    print(f"  Columns: {list(mapped.columns)}")

    # Quick stats
    print(f"\n  Feature statistics:")
    for col in ML_FEATURES:
        vals = mapped[col]
        print(f"    {col:<35} mean={vals.mean():.4f}  std={vals.std():.4f}  range=[{vals.min():.4f}, {vals.max():.4f}]")


if __name__ == "__main__":
    main()
