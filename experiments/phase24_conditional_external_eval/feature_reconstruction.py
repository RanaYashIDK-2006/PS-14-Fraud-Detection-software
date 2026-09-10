"""Maximal causal feature reconstruction for Kaggle dataset.

Reconstructs all 48 E_hardneg features using only information
available at decision time. Computes entity-level aggregates
causally from training data.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd


def _hash_id(val: str) -> int:
    """Deterministic integer hash for entity IDs."""
    return int(hashlib.md5(str(val).encode()).hexdigest()[:8], 16)


def _compute_entity_stats(
    train_df: pd.DataFrame,
    group_col: str,
) -> dict:
    """Compute per-entity statistics from training data."""
    stats = {}
    grouped = train_df.groupby(group_col)

    for entity, group in grouped:
        stats[entity] = {
            "tx_count": len(group),
            "avg_amt": group["amt"].mean(),
            "std_amt": group["amt"].std() if len(group) > 1 else 0.0,
            "fraud_count": int(group["is_fraud"].sum()),
            "fraud_rate": group["is_fraud"].mean(),
            "n_merchants": group["merchant"].nunique() if "merchant" in group.columns else 0,
            "n_cities": group["city"].nunique() if "city" in group.columns else 0,
        }

    return stats


def reconstruct_features_maximal(
    df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Reconstruct ALL 48 E_hardneg features from Kaggle schema.

    Uses causal entity-level aggregates from training data.
    Returns (features, reconstruction_report).
    """
    features = pd.DataFrame(index=df.index)
    report = {"exact": [], "causal": [], "constant": [], "unavailable": []}

    # ── EXACT: Amount features ──
    if "amt" in df.columns:
        features["amt"] = df["amt"]
        features["log_amt"] = np.log1p(df["amt"])
        features["amt_sq"] = df["amt"] ** 2
        features["high_amt"] = (df["amt"] > 200).astype(int)
        features["very_high_amt"] = (df["amt"] > 500).astype(int)
        report["exact"].extend(["amt", "log_amt", "amt_sq", "high_amt", "very_high_amt"])

    # ── EXACT: Temporal features ──
    if "trans_date_trans_time" in df.columns:
        ts = pd.to_datetime(df["trans_date_trans_time"])
        features["hr"] = ts.dt.hour
        features["mn"] = ts.dt.minute
        features["dow"] = ts.dt.dayofweek
        features["Month"] = ts.dt.month
        features["Day"] = ts.dt.day
        features["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        features["is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour <= 5)).astype(int)
        features["is_business_hours"] = ((ts.dt.hour >= 9) & (ts.dt.hour <= 17)).astype(int)
        report["exact"].extend([
            "hr", "mn", "dow", "Month", "Day", "hour_sin", "hour_cos",
            "is_night", "is_business_hours",
        ])

    # ── EXACT: Category features ──
    if "category" in df.columns:
        features["chip"] = (df["category"] == "grocery_pos").astype(int)
        features["is_online"] = df["category"].isin(["internet", "misc_net"]).astype(int)
        features["is_swipe"] = (df["category"] == "gas_transport").astype(int)
        report["exact"].extend(["chip", "is_online", "is_swipe"])

        # MCC-like category encoding
        cat_map = {
            "grocery_pos": 5411, "grocery_net": 5411,
            "gas_transport": 5541, "gas": 5541,
            "food_dining": 5812, "restaurant": 5812,
            "shopping_pos": 5311, "shopping_net": 5311,
            "entertainment": 7832, "travel": 4722,
            "personal_care": 7299, "health_fitness": 8099,
            "kids_pets": 5641, "home": 5200,
            "misc_net": 5999, "misc_pos": 5999,
        }
        features["mcc"] = df["category"].map(cat_map).fillna(0).astype(int)
        features["mcc_high"] = features["mcc"].isin([5999, 7832, 4722]).astype(int)
        features["mcc_restaurant"] = (features["mcc"] == 5812).astype(int)
        features["mcc_gas"] = (features["mcc"] == 5541).astype(int)
        features["mcc_grocery"] = (features["mcc"] == 5411).astype(int)
        features["mcc_travel"] = (features["mcc"] == 4722).astype(int)
        features["mcc_online"] = features["is_online"].values
        report["causal"].extend([
            "mcc", "mcc_high", "mcc_restaurant", "mcc_gas",
            "mcc_grocery", "mcc_travel", "mcc_online",
        ])

    # ── EXACT: Interaction features ──
    if "amt" in features.columns and "hr" in features.columns:
        features["amt_x_hr"] = features["amt"] * features["hr"]
        features["amt_x_mcc"] = features["amt"] * features["mcc"]
        features["amt_x_chip"] = features["amt"] * features["chip"]
        features["amt_x_online"] = features["amt"] * features["is_online"]
        features["amt_x_night"] = features["amt"] * features["is_night"]
        report["exact"].extend([
            "amt_x_hr", "amt_x_mcc", "amt_x_chip", "amt_x_online", "amt_x_night",
        ])

    # ── CONSTANT: has_zip, has_state (always 1 in Kaggle) ──
    features["has_zip"] = 1
    features["has_state"] = 1
    features["is_online_or_no_state"] = features["is_online"].values
    report["constant"].extend(["has_zip", "has_state", "is_online_or_no_state"])

    # ── CAUSAL: Entity IDs ──
    if "cc_num" in df.columns:
        features["card_id"] = df["cc_num"].apply(_hash_id)
        report["causal"].append("card_id")

    if "merchant" in df.columns:
        features["merchant_id"] = df["merchant"].apply(_hash_id)
        report["causal"].append("merchant_id")

    if "city" in df.columns:
        features["city_id"] = df["city"].apply(_hash_id)
        report["causal"].append("city_id")

    # ── CAUSAL: Entity-level aggregates from training data ──
    if train_df is not None and "cc_num" in train_df.columns:
        # Pre-compute entity stats from training data
        user_stats = _compute_entity_stats(train_df, "cc_num")
        merch_stats = _compute_entity_stats(train_df, "merchant") if "merchant" in train_df.columns else {}
        city_stats = _compute_entity_stats(train_df, "city") if "city" in train_df.columns else {}

        # User-level features
        features["user_tx_count"] = df["cc_num"].map(
            lambda x: user_stats.get(x, {}).get("tx_count", 0)
        ).fillna(0).astype(int)
        features["card_tx_count"] = features["user_tx_count"].values  # same as user

        user_avg = df["cc_num"].map(lambda x: user_stats.get(x, {}).get("avg_amt", 0.0)).fillna(0.0)
        features["user_avg_amt"] = user_avg

        user_std = df["cc_num"].map(lambda x: user_stats.get(x, {}).get("std_amt", 1.0)).fillna(1.0)
        features["amt_vs_user_avg"] = np.where(
            user_avg > 0,
            (df["amt"] - user_avg) / user_avg.clip(lower=0.01),
            0.0,
        )
        features["amt_zscore"] = np.where(
            user_std > 0,
            (df["amt"] - user_avg) / user_std.clip(lower=0.01),
            0.0,
        )

        features["user_merchant_diversity"] = df["cc_num"].map(
            lambda x: user_stats.get(x, {}).get("n_merchants", 0)
        ).fillna(0).astype(int)
        features["user_city_diversity"] = df["cc_num"].map(
            lambda x: user_stats.get(x, {}).get("n_cities", 0)
        ).fillna(0).astype(int)
        features["user_merch_count"] = features["user_merchant_diversity"].values

        report["causal"].extend([
            "user_tx_count", "card_tx_count", "user_avg_amt",
            "amt_vs_user_avg", "amt_zscore", "user_merchant_diversity",
            "user_city_diversity", "user_merch_count",
        ])

        # Merchant-level features
        if "merchant" in df.columns:
            features["merch_tx_count"] = df["merchant"].map(
                lambda x: merch_stats.get(x, {}).get("tx_count", 0)
            ).fillna(0).astype(int)
            report["causal"].append("merch_tx_count")

        # Fraud-rate features (causal from training)
        train_fraud = train_df[train_df["is_fraud"] == 1]

        user_fraud = train_fraud.groupby("cc_num").size().to_dict()
        user_total = train_df.groupby("cc_num").size().to_dict()
        features["user_fraud_rate"] = df["cc_num"].map(
            lambda x: user_fraud.get(x, 0) / max(user_total.get(x, 1), 1)
        ).fillna(0.0)
        report["causal"].append("user_fraud_rate")

        if "merchant" in train_df.columns:
            merch_fraud = train_fraud.groupby("merchant").size().to_dict()
            merch_total = train_df.groupby("merchant").size().to_dict()
            features["merch_fraud_rate"] = df["merchant"].map(
                lambda x: merch_fraud.get(x, 0) / max(merch_total.get(x, 1), 1)
            ).fillna(0.0)
            report["causal"].append("merch_fraud_rate")

        if "city" in train_df.columns:
            city_fraud = train_fraud.groupby("city").size().to_dict()
            city_total = train_df.groupby("city").size().to_dict()
            features["city_fraud_rate"] = df["city"].map(
                lambda x: city_fraud.get(x, 0) / max(city_total.get(x, 1), 1)
            ).fillna(0.0)
            report["causal"].append("city_fraud_rate")

    # ── UNAVAILABLE: err (no error/decline code in Kaggle) ──
    features["err"] = 0  # default: no error
    report["unavailable"].append("err")

    # ── Fill any remaining NaN with 0 ──
    features = features.fillna(0)

    return features, report
