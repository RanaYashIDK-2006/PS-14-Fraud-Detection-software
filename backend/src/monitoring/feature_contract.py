"""Decision-time feature contract — classification, validation rules, and versioning.

Every model feature is classified into a category, assigned validation rules,
and checked for decision-time availability. This module is the single source
of truth for what the model expects and what is legitimate at inference time.

Feature categories:
  A. TRANSACTION-DERIVED      — from the current transaction
  B. HISTORICAL/BEHAVIORAL   — from account history (computed before this txn)
  C. DEVICE/CHANNEL          — from device/channel context
  D. AGGREGATED              — rolling counts, averages
  E. EXTERNAL/REFERENCE      — from external lookups
  F. LABEL/OUTCOME-DERIVED   — from fraud labels (NEVER at decision time)
  G. FUTURE/POST-EVENT       — from future information (NEVER at decision time)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FeatureCategory(str, Enum):
    TRANSACTION_DERIVED = "transaction_derived"
    HISTORICAL_BEHAVIORAL = "historical_behavioral"
    DEVICE_CHANNEL = "device_channel"
    AGGREGATED = "aggregated"
    EXTERNAL_REFERENCE = "external_reference"
    LABEL_OUTCOME = "label_outcome"        # NEVER at decision time
    FUTURE_POST_EVENT = "future_post_event"  # NEVER at decision time


class MissingPolicy(str, Enum):
    REJECT = "reject"                    # reject the request
    FAIL_CLOSED = "fail_closed"          # return high-risk fallback
    USE_NEUTRAL = "use_neutral"          # documented safe default
    USE_IMPUTED = "use_imputed"          # model-compatible imputation


class FeatureStatus(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    STALE = "stale"
    INVALID = "invalid"
    UNAVAILABLE_AT_DECISION_TIME = "unavailable_at_decision_time"


@dataclass
class FeatureSpec:
    """Complete specification for one model feature."""
    name: str
    category: FeatureCategory
    datatype: str  # "int", "float", "bool"
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None  # for categoricals
    missing_policy: MissingPolicy = MissingPolicy.USE_NEUTRAL
    neutral_value: Any = 0.0
    max_age_seconds: float | None = None  # freshness requirement
    description: str = ""
    available_at_decision_time: bool = True
    depends_on_label: bool = False
    depends_on_future: bool = False
    version: str = "1.0"


# ── ML_FEATURES contract (PS-14 §16 features) ─────────────────────────

ML_FEATURE_CONTRACT: dict[str, FeatureSpec] = {
    # A. Transaction-derived
    "amount_ratio": FeatureSpec(
        name="amount_ratio", category=FeatureCategory.TRANSACTION_DERIVED,
        datatype="float", min_value=0.0, max_value=1000.0,
        missing_policy=MissingPolicy.REJECT,
        description="ratio of transaction amount to account median",
    ),
    "txn_time_unusual": FeatureSpec(
        name="txn_time_unusual", category=FeatureCategory.TRANSACTION_DERIVED,
        datatype="int", min_value=0, max_value=1,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0,
        description="1 if hour is outside typical hours",
    ),
    "hour_of_day": FeatureSpec(
        name="hour_of_day", category=FeatureCategory.TRANSACTION_DERIVED,
        datatype="int", min_value=0, max_value=23,
        missing_policy=MissingPolicy.REJECT,
        description="hour of transaction (0-23)",
    ),
    "is_weekend": FeatureSpec(
        name="is_weekend", category=FeatureCategory.TRANSACTION_DERIVED,
        datatype="int", min_value=0, max_value=1,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0,
    ),
    "failed_auth_count_24h": FeatureSpec(
        name="failed_auth_count_24h", category=FeatureCategory.TRANSACTION_DERIVED,
        datatype="int", min_value=0, max_value=100,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0,
    ),

    # B. Historical/behavioral
    "days_since_last_similar_txn": FeatureSpec(
        name="days_since_last_similar_txn", category=FeatureCategory.HISTORICAL_BEHAVIORAL,
        datatype="float", min_value=0.0, max_value=3650.0,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=30.0,
        description="days since a similar transaction (0 for new accounts)",
    ),
    "gradual_escalation_score": FeatureSpec(
        name="gradual_escalation_score", category=FeatureCategory.HISTORICAL_BEHAVIORAL,
        datatype="float", min_value=0.0, max_value=1.0,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0.0,
    ),
    "known_device_count": FeatureSpec(
        name="known_device_count", category=FeatureCategory.HISTORICAL_BEHAVIORAL,
        datatype="int", min_value=0, max_value=100,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=1,
    ),
    "account_tenure_days": FeatureSpec(
        name="account_tenure_days", category=FeatureCategory.HISTORICAL_BEHAVIORAL,
        datatype="float", min_value=0.0, max_value=36500.0,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=1.0,
    ),

    # C. Device/channel
    "new_device_flag": FeatureSpec(
        name="new_device_flag", category=FeatureCategory.DEVICE_CHANNEL,
        datatype="int", min_value=0, max_value=1,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0,
    ),
    "unusual_location_flag": FeatureSpec(
        name="unusual_location_flag", category=FeatureCategory.DEVICE_CHANNEL,
        datatype="int", min_value=0, max_value=1,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0,
    ),
    "unusual_recipient_flag": FeatureSpec(
        name="unusual_recipient_flag", category=FeatureCategory.DEVICE_CHANNEL,
        datatype="int", min_value=0, max_value=1,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0,
    ),

    # D. Aggregated
    "txn_freq_last_24h": FeatureSpec(
        name="txn_freq_last_24h", category=FeatureCategory.AGGREGATED,
        datatype="int", min_value=0, max_value=1000,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0,
    ),
    "shared_device_accounts": FeatureSpec(
        name="shared_device_accounts", category=FeatureCategory.AGGREGATED,
        datatype="int", min_value=0, max_value=8,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0,
    ),
    "shared_recipient_accounts": FeatureSpec(
        name="shared_recipient_accounts", category=FeatureCategory.AGGREGATED,
        datatype="int", min_value=0, max_value=8,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0,
    ),
    "mule_ring_score": FeatureSpec(
        name="mule_ring_score", category=FeatureCategory.AGGREGATED,
        datatype="float", min_value=0.0, max_value=1.0,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0.0,
    ),

    # Deviation features
    "hour_deviation": FeatureSpec(
        name="hour_deviation", category=FeatureCategory.HISTORICAL_BEHAVIORAL,
        datatype="float", min_value=0.0, max_value=1.0,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0.0,
    ),
    "amount_zscore": FeatureSpec(
        name="amount_zscore", category=FeatureCategory.HISTORICAL_BEHAVIORAL,
        datatype="float", min_value=-10.0, max_value=10.0,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0.0,
    ),
    "velocity_deviation": FeatureSpec(
        name="velocity_deviation", category=FeatureCategory.HISTORICAL_BEHAVIORAL,
        datatype="float", min_value=0.0, max_value=1.0,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0.0,
    ),
    "recipient_novelty": FeatureSpec(
        name="recipient_novelty", category=FeatureCategory.HISTORICAL_BEHAVIORAL,
        datatype="float", min_value=0.0, max_value=1.0,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0.0,
    ),
    "txn_regularity": FeatureSpec(
        name="txn_regularity", category=FeatureCategory.HISTORICAL_BEHAVIORAL,
        datatype="float", min_value=0.0, max_value=1.0,
        missing_policy=MissingPolicy.USE_NEUTRAL, neutral_value=0.5,
    ),
}

# ── Vector-level contract (ML_FEATURES ordering) ────────────────────────

# This is the canonical ordering the model expects.
# Adding features here requires retraining.
ML_FEATURE_VERSION = "v1"
ML_FEATURE_ORDER = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "hour_of_day", "is_weekend",
    "shared_device_accounts", "shared_recipient_accounts", "mule_ring_score",
    "hour_deviation", "amount_zscore", "velocity_deviation",
    "recipient_novelty", "txn_regularity",
]

# ── Validation functions ────────────────────────────────────────────────

def validate_feature(name: str, value: Any, spec: FeatureSpec | None = None) -> tuple[FeatureStatus, str]:
    """Validate a single feature value against its contract.

    Returns (status, detail).
    """
    if spec is None:
        spec = ML_FEATURE_CONTRACT.get(name)
    if spec is None:
        return FeatureStatus.AVAILABLE, "no contract — unchecked"

    # Check nullability
    if value is None:
        if spec.nullable:
            return FeatureStatus.AVAILABLE, "null is allowed"
        if spec.missing_policy == MissingPolicy.REJECT:
            return FeatureStatus.MISSING, f"required feature '{name}' is null"
        return FeatureStatus.MISSING, f"feature '{name}' is null (policy: {spec.missing_policy.value})"

    # Check datatype
    if spec.datatype == "int":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return FeatureStatus.INVALID, f"expected int, got {type(value).__name__}"
        if value != int(value):
            return FeatureStatus.INVALID, f"expected int, got {value}"
        value = int(value)
    elif spec.datatype == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return FeatureStatus.INVALID, f"expected float, got {type(value).__name__}"
        value = float(value)
    elif spec.datatype == "bool":
        if not isinstance(value, bool):
            return FeatureStatus.INVALID, f"expected bool, got {type(value).__name__}"

    # Check NaN/Inf
    if isinstance(value, float) and not math.isfinite(value):
        return FeatureStatus.INVALID, f"non-finite value: {value}"

    # Check range
    if spec.min_value is not None and value < spec.min_value:
        return FeatureStatus.INVALID, f"value {value} < min {spec.min_value}"
    if spec.max_value is not None and value > spec.max_value:
        return FeatureStatus.INVALID, f"value {value} > max {spec.max_value}"

    # Check allowed values
    if spec.allowed_values is not None and value not in spec.allowed_values:
        return FeatureStatus.INVALID, f"value {value} not in allowed {spec.allowed_values}"

    return FeatureStatus.AVAILABLE, "ok"


def validate_feature_vector(features: dict[str, Any]) -> dict[str, tuple[FeatureStatus, str]]:
    """Validate all features in a vector against the contract.

    Returns dict mapping feature name -> (status, detail).
    """
    results: dict[str, tuple[FeatureStatus, str]] = {}
    for name in ML_FEATURE_ORDER:
        value = features.get(name)
        spec = ML_FEATURE_CONTRACT.get(name)
        results[name] = validate_feature(name, value, spec)
    return results


def check_feature_ordering(features: dict[str, Any]) -> tuple[bool, list[str]]:
    """Verify feature names match the expected ordering.

    Returns (is_correct, list of mismatches).
    """
    actual_keys = [k for k in features.keys() if k in ML_FEATURE_CONTRACT]
    if actual_keys == ML_FEATURE_ORDER:
        return True, []
    mismatches = []
    for i, expected in enumerate(ML_FEATURE_ORDER):
        if i >= len(actual_keys):
            mismatches.append(f"missing feature at position {i}: {expected}")
        elif actual_keys[i] != expected:
            mismatches.append(f"position {i}: expected {expected}, got {actual_keys[i]}")
    if len(actual_keys) > len(ML_FEATURE_ORDER):
        mismatches.append(f"extra features: {actual_keys[len(ML_FEATURE_ORDER):]}")
    return len(mismatches) == 0, mismatches


def classify_decision_time_availability(features: dict[str, Any]) -> dict[str, FeatureStatus]:
    """Check which features are legitimately available at decision time.

    Features that depend on labels or future information are flagged.
    """
    results: dict[str, FeatureStatus] = {}
    for name, value in features.items():
        spec = ML_FEATURE_CONTRACT.get(name)
        if spec is None:
            results[name] = FeatureStatus.AVAILABLE
            continue
        if spec.depends_on_label:
            results[name] = FeatureStatus.UNAVAILABLE_AT_DECISION_TIME
        elif spec.depends_on_future:
            results[name] = FeatureStatus.UNAVAILABLE_AT_DECISION_TIME
        elif not spec.available_at_decision_time:
            results[name] = FeatureStatus.UNAVAILABLE_AT_DECISION_TIME
        else:
            results[name] = FeatureStatus.AVAILABLE
    return results
