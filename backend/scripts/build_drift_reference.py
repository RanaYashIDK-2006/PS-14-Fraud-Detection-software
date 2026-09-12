#!/usr/bin/env python3
"""Build reference distribution for drift detection from Altman training data.

Generates the same 32 features as the clean Altman model and saves
per-feature statistics + raw arrays for PSI comparison.
"""
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.drift_detector import ReferenceDistribution

DATA_DIR = Path("data")
MODELS_DIR = Path("models/production")


def engineer_features(df):
    """Same 32 clean features as train_altman_production.py."""
    F = pd.DataFrame()
    F["amt"] = df["amt"]
    F["log_amt"] = np.log1p(F["amt"])
    F["amt_sq"] = F["amt"] ** 2
    F["hr"] = df["hr"]
    F["mn"] = df["mn"]
    F["dow"] = df["dow"]
    F["Month"] = df["Month"]
    F["Day"] = df["Day"]
    F["hour_sin"] = np.sin(2 * np.pi * F["hr"] / 24)
    F["hour_cos"] = np.cos(2 * np.pi * F["hr"] / 24)
    F["is_night"] = ((F["hr"] < 6) | (F["hr"] > 22)).astype(int)
    F["is_business_hours"] = ((F["hr"] >= 9) & (F["hr"] <= 17)).astype(int)
    F["chip"] = df["chip"]
    F["is_online"] = df["is_online"]
    F["err"] = (df["err"] != 0).astype(int)
    F["mcc_n"] = df["mcc_n"]
    F["user_tx_count"] = df.groupby("User").cumcount()
    F["card_tx_count"] = df.groupby("Card").cumcount()
    F["user_avg_amt"] = df.groupby("User")["amt"].transform(lambda x: x.expanding().mean().shift(1))
    F["amt_vs_user_avg"] = F["amt"] / (F["user_avg_amt"] + 1e-6)
    F["amt_zscore"] = (F["amt"] - F["user_avg_amt"]) / (
        df.groupby("User")["amt"].transform(lambda x: x.expanding().std().shift(1)) + 1e-6
    )
    F["merch_tx_count"] = df.groupby("Merchant Name").cumcount()
    F["has_zip"] = df["Zip"].notna().astype(int)
    F["has_state"] = df["Merchant State"].notna().astype(int)
    F["is_online_or_no_state"] = ((F["is_online"] == 1) | (F["has_state"] == 0)).astype(int)
    F["high_amt"] = (F["amt"] > F["user_avg_amt"] * 2).astype(int)
    F["very_high_amt"] = (F["amt"] > F["user_avg_amt"] * 5).astype(int)
    F["amt_x_hr"] = F["amt"] * F["hr"]
    F["amt_x_mcc"] = F["amt"] * F["mcc_n"]
    F["amt_x_chip"] = F["amt"] * F["chip"]
    F["amt_x_online"] = F["amt"] * F["is_online"]
    F["amt_x_night"] = F["amt"] * F["is_night"]
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return F


def main():
    print("=" * 60)
    print("Building Drift Reference Distribution")
    print("=" * 60)
    t0 = time.time()

    # Load Altman data (same sampling as training)
    print("\n[1/3] Loading Altman data (1% legit + all fraud)...")
    rng = np.random.RandomState(42)
    chunks = []
    for chunk in pd.read_csv(
        DATA_DIR / "credit_card_transactions-ibm_v2.csv",
        usecols=["User", "Card", "Year", "Month", "Day", "Time", "Amount",
                 "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
                 "Zip", "MCC", "Errors?", "Is Fraud?"],
        low_memory=False,
        chunksize=500_000,
    ):
        fraud_mask = chunk["Is Fraud?"] == "Yes"
        is_legit = ~fraud_mask
        sample_legit = rng.random(len(chunk)) < 0.01
        keep = fraud_mask | (is_legit & sample_legit)
        chunks.append(chunk[keep].copy())

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Sampled: {len(df):,} rows")

    # Parse
    df["amt"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["is_fraud"] = (df["Is Fraud?"] == "Yes").astype(int)
    df["hr"] = df["Time"].str.split(":").str[0].astype(int)
    df["mn"] = df["Time"].str.split(":").str[1].astype(int)
    df["dow"] = pd.to_datetime(df[["Year", "Month", "Day"]]).dt.dayofweek
    df["chip"] = (df["Use Chip"] == "Chip Transaction").astype(int)
    df["is_online"] = (df["Use Chip"] == "Online Transaction").astype(int)
    df["mcc_n"] = df["MCC"].fillna(0).astype(float)
    df["err"] = df["Errors?"].fillna("0")
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]].assign(
        hour=df["hr"], minute=df["mn"]
    ))
    df = df.sort_values(["User", "datetime"]).reset_index(drop=True)

    # Feature engineering
    print("\n[2/3] Engineering 32 clean features...")
    t1 = time.time()
    F = engineer_features(df)
    feature_names = list(F.columns)
    X = F.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    print(f"  {len(feature_names)} features, {len(X):,} rows ({time.time()-t1:.0f}s)")

    # Build reference distribution
    print("\n[3/3] Building reference distribution...")
    ref = ReferenceDistribution.from_training_data(X, feature_names, keep_raw=True)

    # Save to models/production/
    ref_path = MODELS_DIR / "drift_reference.json"
    ref.save(ref_path)
    print(f"  Saved: {ref_path}")
    print(f"  Also: {str(ref_path).replace('.json', '.npy')}")

    # Print summary stats
    print(f"\n  Reference stats:")
    print(f"    Samples: {ref.n_samples:,}")
    print(f"    Features: {ref.n_features}")
    for name in feature_names[:5]:
        s = ref.feature_stats[name]
        print(f"    {name}: mean={s['mean']:.4f}, std={s['std']:.4f}, p50={s['p50']:.4f}")
    print(f"    ... ({len(feature_names)} total)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
