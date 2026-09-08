#!/usr/bin/env python3
"""Analyze ealtman2019 fraud patterns to find detection gaps."""

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

print("Loading 2M rows...")
t0 = time.time()
df = pd.read_csv(ROOT / "data" / "credit_card_transactions-ibm_v2.csv", nrows=2_000_000, dtype=str)
n = len(df)

# Parse
df["amt"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
df["hour"] = df["Time"].str.split(":").str[0].astype(int)
df["uc"] = df["User"] + "_" + df["Card"]
y = (df["Is Fraud?"] == "Yes").values
fraud = df[y].copy()
legit = df[~y].copy()
print(f"  {n:,} rows, {int(y.sum()):,} fraud ({y.sum()/n*100:.4f}%) [{time.time()-t0:.0f}s]")

# ── 1. Amount patterns ────────────────────────────────────────────────────
print("\n=== AMOUNT PATTERNS ===")
print("Fraud amount distribution:")
print(fraud["amt"].describe())
print("\nLegit amount distribution:")
print(legit["amt"].describe())

print("\nFraud amount by Use Chip:")
print(fraud.groupby("Use Chip")["amt"].agg(["mean", "median", "count"]))

# ── 2. Time patterns ──────────────────────────────────────────────────────
print("\n=== TIME PATTERNS ===")
print("Fraud hour distribution:")
print(fraud["hour"].value_counts().sort_index())
print("\nLegit hour distribution:")
print(legit["hour"].value_counts().sort_index())

# ── 3. Merchant patterns ──────────────────────────────────────────────────
print("\n=== MERCHANT PATTERNS ===")
fraud_merchants = fraud["Merchant Name"].value_counts().head(10)
print("Top 10 fraud merchants:")
print(fraud_merchants)

# ── 4. Use Chip patterns ──────────────────────────────────────────────────
print("\n=== USE CHIP PATTERNS ===")
print("Fraud by Use Chip:")
print(fraud["Use Chip"].value_counts())
print("\nLegit by Use Chip:")
print(legit["Use Chip"].value_counts())

# ── 5. MCC patterns ──────────────────────────────────────────────────────
print("\n=== MCC PATTERNS ===")
print("Top 10 fraud MCCs:")
print(fraud["MCC"].value_counts().head(10))

# ── 6. Error patterns ────────────────────────────────────────────────────
print("\n=== ERROR PATTERNS ===")
fraud_errors = fraud["Errors?"].fillna("").replace("", "no_error")
print("Fraud errors:")
print(fraud_errors.value_counts().head(10))

# ── 7. Geographic patterns ────────────────────────────────────────────────
print("\n=== GEOGRAPHIC PATTERNS ===")
print("Top 10 fraud states:")
print(fraud["Merchant State"].value_counts().head(10))
print("\nTop 10 fraud cities:")
print(fraud["Merchant City"].value_counts().head(10))

# ── 8. Card-level patterns ────────────────────────────────────────────────
print("\n=== CARD-LEVEL PATTERNS ===")
card_fraud = fraud.groupby("uc").agg(
    fraud_count=("amt", "count"),
    fraud_amt_mean=("amt", "mean"),
    fraud_amt_max=("amt", "max"),
).sort_values("fraud_count", ascending=False)
print(f"Cards with fraud: {len(card_fraud)}")
print(f"Avg fraud per card: {card_fraud['fraud_count'].mean():.1f}")
print(f"Max fraud per card: {card_fraud['fraud_count'].max()}")

# ── 9. Sequential patterns (within same user-card) ────────────────────────
print("\n=== SEQUENTIAL PATTERNS ===")
# Check if fraud follows a specific pattern
uc_fraud = df[df["uc"].isin(fraud["uc"].unique())].sort_values(["uc", "Year", "Month", "Day", "Time"])
uc_fraud["_is_fraud"] = (uc_fraud["Is Fraud?"] == "Yes").astype(int)
# Find transactions right before fraud
uc_fraud["_prev_is_fraud"] = uc_fraud.groupby("uc")["_is_fraud"].shift(1)
print("What happens BEFORE fraud:")
print(uc_fraud[uc_fraud["_is_fraud"] == 1]["_prev_is_fraud"].value_counts())

# ── 10. MCC-specific fraud rates ──────────────────────────────────────────
print("\n=== MCC FRAUD RATES ===")
mcc_stats = df.groupby("MCC").agg(
    total=("amt", "count"),
    fraud=("Is Fraud?", lambda x: (x == "Yes").sum()),
).reset_index()
mcc_stats["rate"] = mcc_stats["fraud"] / mcc_stats["total"]
mcc_stats = mcc_stats[mcc_stats["total"] >= 100].sort_values("rate", ascending=False)
print(mcc_stats.head(10).to_string())

print("\n=== ANALYSIS COMPLETE ===")
