"""Decision-time feature engine — reconstructs PS-14 features from real-world data."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureReconstructionResult:
    """Result of attempting to reconstruct the feature vector."""
    n_features: int
    n_reconstructed: int
    n_missing: int
    missing_features: list[str]
    status: str  # "PASS" | "PARTIAL" | "FAIL"
    reconstruction_details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_reconstructed": self.n_reconstructed,
            "n_missing": self.n_missing,
            "missing_features": self.missing_features,
            "status": self.status,
            "reconstruction_details": self.reconstruction_details,
        }


def reconstruct_p20_features(
    df: pd.DataFrame,
    schema_map: dict[str, str],
    timestamp_field: str,
) -> FeatureReconstructionResult:
    """Attempt to reconstruct P20's 45 features from real-world data.

    This is a best-effort reconstruction using available fields.
    Features that cannot be derived from a single transaction row
    are marked as requiring historical context.
    """
    # P20 features grouped by derivation complexity
    SIMPLE_DERIVABLE = {
        # Direct from fields
        "amount": lambda df, sm: df.get(sm.get("amount", "amount")),
        "hour_of_day": lambda df, sm: pd.to_datetime(df[timestamp_field], errors="coerce").dt.hour,
        "is_weekend": lambda df, sm: pd.to_datetime(df[timestamp_field], errors="coerce").dt.dayofweek.isin([5, 6]).astype(int),
    }

    DERIVABLE_WITH_CONTEXT = {
        # Can be derived with rolling windows
        "txn_freq_last_24h": "requires_rolling_window",
        "txn_time_unusual": "requires_typical_hours",
        "new_device_flag": "requires_device_history",
        "unusual_location_flag": "requires_location_profile",
        "unusual_recipient_flag": "requires_recipient_history",
        "failed_auth_count_24h": "requires_auth_log",
        "days_since_last_similar_txn": "requires_transaction_history",
        "gradual_escalation_score": "requires_amount_history",
        "known_device_count": "requires_device_history",
        "account_tenure_days": "requires_account_creation_date",
        "shared_device_accounts": "requires_device_graph",
        "shared_recipient_accounts": "requires_recipient_graph",
        "mule_ring_score": "requires_graph_analysis",
        "hour_deviation": "requires_typical_hours",
        "amount_zscore": "requires_amount_history",
        "velocity_deviation": "requires_velocity_baseline",
        "recipient_novelty": "requires_recipient_history",
        "txn_regularity": "requires_transaction_history",
    }

    MODEL_INPUT_FEATURES = [
        "amt", "log_amt", "amt_sq", "hr", "mn", "is_weekend",
        "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
        "unusual_location_flag", "unusual_recipient_flag",
        "failed_auth_count_24h", "days_since_last_similar_txn",
        "gradual_escalation_score", "known_device_count",
        "account_tenure_days", "hour_of_day",
        "shared_device_accounts", "shared_recipient_accounts",
        "mule_ring_score", "hour_deviation", "amount_zscore",
        "velocity_deviation", "recipient_novelty", "txn_regularity",
        "txn_amount_bucket", "amount_ratio", "category_encoded",
        "merchant_risk_score", "device_age_days", "location_distance",
        "amount_pct_of_avg", "txn_count_7d", "txn_count_30d",
        "avg_amount_7d", "avg_amount_30d", "max_amount_7d",
        "unique_merchants_7d", "unique_merchants_30d",
        "night_txn_ratio", "weekend_txn_ratio",
        "same_merchant_count_24h", "cross_border_flag",
        "card_age_days", "prev_fraud_flag",
        "merchant_fraud_history_30d", "city_population_bucket",
    ]

    reconstructed = set()
    missing = []

    # Check which features can be derived from available columns
    available_cols = set(df.columns)

    for feat in MODEL_INPUT_FEATURES:
        if feat in available_cols:
            reconstructed.add(feat)
        elif feat in SIMPLE_DERIVABLE:
            reconstructed.add(feat)
        elif feat in DERIVABLE_WITH_CONTEXT:
            missing.append(feat)
        else:
            missing.append(feat)

    n_total = len(MODEL_INPUT_FEATURES)
    n_recon = len(reconstructed)

    status = "PASS" if len(missing) == 0 else "PARTIAL" if n_recon > n_total * 0.5 else "FAIL"

    return FeatureReconstructionResult(
        n_features=n_total,
        n_reconstructed=n_recon,
        n_missing=len(missing),
        missing_features=missing,
        status=status,
        reconstruction_details={
            "available_columns": sorted(available_cols),
            "derivable_with_context": sorted(DERIVABLE_WITH_CONTEXT.keys()),
            "simple_derivable": sorted(SIMPLE_DERIVABLE.keys()),
        },
    )


def check_e_hardneg_label_features(
    df: pd.DataFrame,
    has_historical_labels: bool,
) -> dict:
    """Check whether E_hardneg's 3 fraud-rate features can be reconstructed."""
    features = ["user_fraud_rate", "merch_fraud_rate", "city_fraud_rate"]
    reconstructable = has_historical_labels

    return {
        "features": features,
        "has_historical_labels": has_historical_labels,
        "reconstructable": reconstructable,
        "status": "RECONSTRUCTABLE" if reconstructable else "NOT_RECONSTRUCTABLE",
        "e_hardneg_real_world_evaluation": "VALID" if reconstructable else "CONDITIONAL",
    }
