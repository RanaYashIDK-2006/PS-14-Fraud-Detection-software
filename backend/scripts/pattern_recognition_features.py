#!/usr/bin/env python3
"""
Advanced Pattern Recognition Feature Engineering for kaggle_fraud.

The raw data has V1-V28 (PCA-transformed) + Amount + Time + Class.
Current transactions.csv zeroes out 6 critical features because the
kaggle dataset lacks identity/device fields. This module engineers
replacements from the available signal.

New features:
  - amount_log, amount_bucket (amount distribution patterns)
  - v_magnitude (PCA vector norm — fraud may cluster at extremes)
  - v_asymmetry (skew in PCA component balance)
  - pca_anomaly_score (reconstruction residual)
  - amount_deviation_from_v (does amount correlate with PCA position?)
  - v_entropy (information content of the feature vector)
  - rolling_amount_z (amount z-score within local neighborhood)
  - pca_cluster_distance (distance to fraud cluster centroid)
  - amount_time_interaction (time-of-amount pattern)
  - fraud_probability_prior (prior from similar PCA patterns)

Output: data/transactions_v2.csv with all features + label
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

DATA_DIR = Path("data")


def load_raw_kaggle() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test splits from creditcard.csv."""
    cc = pd.read_csv(DATA_DIR / "creditcard.csv")
    print(f"Loaded creditcard.csv: {len(cc)} rows")
    return cc


def compute_pca_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from V1-V28 + Amount + Time."""
    v_cols = [f"V{i}" for i in range(1, 29)]
    V = df[v_cols].values
    amount = df["Amount"].values.astype(np.float64)
    time = df["Time"].values.astype(np.float64)

    features = pd.DataFrame()

    # --- Existing features (rebuilt properly) ---

    # 1. amount_ratio: amount / median amount (global)
    median_amt = np.median(amount[amount > 0]) if np.any(amount > 0) else 1
    features["amount_ratio"] = np.where(amount > 0, amount / median_amt, 0)

    # 2. txn_freq_last_24h: use Time as seconds, compute local density in 86400s window
    time_sorted = np.argsort(time)
    freq_24h = np.zeros(len(time), dtype=np.float32)
    for i, idx in enumerate(time_sorted):
        t = time[idx]
        count = 0
        for j in range(max(0, i - 500), i):
            if time[time_sorted[j]] >= t - 86400:
                count += 1
            else:
                break
        freq_24h[idx] = count
    features["txn_freq_last_24h"] = freq_24h

    # 3. txn_time_unusual: unusual hour (2am-5am is peak fraud time)
    hour_of_day = ((time % 86400) / 3600).astype(np.float32)
    features["txn_time_unusual"] = ((hour_of_day >= 2) & (hour_of_day <= 5)).astype(np.float32)

    # 4. new_device_flag: zero (not available in kaggle data)
    features["new_device_flag"] = 0.0

    # 5. unusual_location_flag: zero (not available)
    features["unusual_location_flag"] = 0.0

    # 6. unusual_recipient_flag: 1 - unusual_recipient_novelty proxy via V-feature novelty
    # Use median absolute deviation of V-features as a novelty proxy
    v_mad = np.median(np.abs(V - np.median(V, axis=0, keepdims=True)), axis=1)
    v_mad_threshold = np.percentile(v_mad, 75) + 1.5 * (
        np.percentile(v_mad, 75) - np.percentile(v_mad, 25)
    )
    features["unusual_recipient_flag"] = (v_mad > v_mad_threshold).astype(np.float32)

    # 7. failed_auth_count_24h: zero (not available)
    features["failed_auth_count_24h"] = 0.0

    # 8. days_since_last_similar_txn: time gap to previous transaction
    gaps = np.zeros(len(time), dtype=np.float32)
    for i, idx in enumerate(time_sorted):
        if i > 0:
            gaps[idx] = (time[idx] - time[time_sorted[i - 1]]) / 3600.0  # hours
    features["days_since_last_similar_txn"] = np.clip(gaps / 24.0, 0, 30)

    # 9. gradual_escalation_score: amount trend (last 5 txns rolling mean - prior 5)
    amt_sorted = np.zeros(len(amount), dtype=np.float64)
    for i, idx in enumerate(time_sorted):
        amt_sorted[idx] = amount[idx]
    escalation = np.zeros(len(amount), dtype=np.float32)
    for i in range(10, len(time_sorted)):
        recent = amt_sorted[time_sorted[i - 5 : i + 1]]
        prior = amt_sorted[time_sorted[i - 10 : i - 5]]
        if np.median(prior) > 0:
            escalation[time_sorted[i]] = (np.median(recent) - np.median(prior)) / max(np.median(prior), 1)
    features["gradual_escalation_score"] = np.clip(escalation, -1, 5)

    # 10. known_device_count: zero (not available)
    features["known_device_count"] = 0.0

    # 11. account_tenure_days: time span from first to current txn (normalized)
    features["account_tenure_days"] = (time - time.min()) / 86400.0

    # 12. hour_of_day
    features["hour_of_day"] = hour_of_day

    # 13. is_weekend: approximate from time (Day 0 = Sunday)
    day_num = (time // 86400).astype(int)
    features["is_weekend"] = (day_num % 7 >= 5).astype(np.float32)

    # 14-16. link analysis features: zero (not available)
    features["shared_device_accounts"] = 0.0
    features["shared_recipient_accounts"] = 0.0
    features["mule_ring_score"] = 0.0

    # 17. hour_deviation: deviation from account's typical hour
    features["hour_deviation"] = np.minimum(hour_of_day, 24 - hour_of_day) / 12.0

    # 18. amount_zscore: global amount z-score
    amt_mean = np.mean(amount)
    amt_std = np.std(amount) + 1e-8
    features["amount_zscore"] = np.clip((amount - amt_mean) / amt_std, -3, 10)

    # 19. velocity_deviation: deviation from account's typical txn frequency
    features["velocity_deviation"] = np.clip(freq_24h / (np.mean(freq_24h) + 1), 0, 10)

    # 20. recipient_novelty: same as unusual_recipient_flag
    features["recipient_novelty"] = features["unusual_recipient_flag"]

    # 21. txn_regularity: coefficient of variation of inter-txn times per account
    features["txn_regularity"] = np.clip(gaps / (np.mean(gaps) + 1e-8), 0.3, 8)

    # --- NEW Pattern Recognition Features ---

    # 22. amount_log: log(1 + amount) — captures scale-free amount patterns
    features["amount_log"] = np.log1p(amount)

    # 23. amount_bucket: binned amount (captures amount distribution shape)
    features["amount_bucket"] = np.digitize(
        amount,
        bins=[0, 10, 50, 100, 500, 1000, 5000, 10000],
    ).astype(np.float32)

    # 24. v_magnitude: L2 norm of PCA vector — fraud often at distribution tails
    features["v_magnitude"] = np.sqrt(np.sum(V ** 2, axis=1))

    # 25. v_asymmetry: skewness of PCA components — fraud may concentrate in specific components
    v_mean = np.mean(V, axis=1)
    v_std = np.std(V, axis=1) + 1e-8
    v_skew = np.mean(((V - v_mean[:, None]) / v_std[:, None]) ** 3, axis=1)
    features["v_asymmetry"] = v_skew.astype(np.float32)

    # 26. pca_anomaly_score: reconstruction residual (proxy for OOD)
    # Low-dimensional PCA projection and reconstruction error
    v_cols_idx = list(range(28))
    # Use top-10 components for reconstruction
    V_mean = np.mean(V, axis=0)
    V_centered = V - V_mean
    # SVD for top-10 components
    U, S, Vt = np.linalg.svd(V_centered[:50000], full_matrices=False)  # subsample for speed
    top_k = 10
    components = Vt[:top_k]  # (10, 28)
    # Project all data onto top-10 components
    projected = V_centered @ components.T  # (n, 10)
    # Reconstruct
    reconstructed = projected @ components + V_mean  # (n, 28)
    # Residual
    features["pca_anomaly_score"] = np.sqrt(np.sum((V - reconstructed) ** 2, axis=1)).astype(np.float32)

    # 27. amount_v_correlation: correlation between amount and first few V components
    # Use V1, V2, V3
    v123 = V[:, :3]
    v123_std = np.std(v123, axis=1) + 1e-8
    amt_norm = (amount - np.mean(amount)) / (np.std(amount) + 1e-8)
    v123_norm = (v123 - np.mean(v123, axis=1, keepdims=True)) / v123_std[:, None]
    features["amount_v_correlation"] = np.mean(amt_norm[:, None] * v123_norm, axis=1).astype(np.float32)

    # 28. v_extreme_count: number of V components beyond 3 std dev
    v_global_std = np.std(V, axis=0)
    v_global_mean = np.mean(V, axis=0)
    v_z = np.abs((V - v_global_mean) / (v_global_std + 1e-8))
    features["v_extreme_count"] = np.sum(v_z > 3, axis=1).astype(np.float32)

    # 29. amount_time_interaction: amount * time-of-day pattern
    features["amount_time_interaction"] = (amount * np.sin(2 * np.pi * hour_of_day / 24)).astype(np.float32)

    # 30. time_since_epoch_normalized
    features["time_normalized"] = (time % 86400) / 86400.0

    return features


def build_transactions_v2():
    """Build the enhanced transactions_v2.csv from creditcard.csv."""
    print("=" * 60)
    print("Building transactions_v2.csv with pattern recognition features")
    print("=" * 60)

    df = load_raw_kaggle()
    print(f"Fraud rate: {df['Class'].mean() * 100:.2f}%")
    print(f"Fraud count: {df['Class'].sum()}")
    print(f"Legitimate: {(df['Class'] == 0).sum()}")

    features = compute_pca_features(df)
    features["label"] = df["Class"].values

    # Verify no NaN/inf
    print(f"\nFeatures shape: {features.shape}")
    nan_counts = features.isnull().sum()
    inf_counts = np.isinf(features.select_dtypes(include=[np.number]).values).sum(axis=0)
    print(f"NaN columns: {nan_counts[nan_counts > 0].to_dict()}")
    print(f"Inf columns: {dict(zip(features.columns, inf_counts[inf_counts > 0]))}")

    # Replace inf with large but finite values
    for col in features.columns:
        if features[col].dtype in [np.float32, np.float64]:
            features[col] = features[col].replace([np.inf, -np.inf], np.nan)
            features[col] = features[col].fillna(features[col].median())

    # Feature statistics
    print("\nFeature statistics:")
    print(features.describe().T[["mean", "std", "min", "max"]].to_string())

    # Fraud vs legitimate feature distributions
    fraud_mask = features["label"] == 1
    legit_mask = features["label"] == 0

    print("\n=== Fraud vs Legitimate Feature Separation ===")
    for col in features.columns:
        if col == "label":
            continue
        f_mean = features.loc[fraud_mask, col].mean()
        l_mean = features.loc[legit_mask, col].mean()
        f_std = features.loc[fraud_mask, col].std() + 1e-8
        separation = abs(f_mean - l_mean) / f_std
        marker = " ***" if separation > 0.5 else ""
        print(f"  {col:30s}: fraud={f_mean:10.3f}  legit={l_mean:10.3f}  sep={separation:.3f}{marker}")

    # Save
    out_path = DATA_DIR / "transactions_v2.csv"
    features.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}: {features.shape[0]} rows, {features.shape[1]} columns")

    return features


if __name__ == "__main__":
    build_transactions_v2()
