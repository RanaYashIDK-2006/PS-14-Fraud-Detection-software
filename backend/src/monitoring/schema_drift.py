"""Schema and feature-availability drift detection.

Detects:
  - missing features (expected but absent)
  - unexpected features (present but not in schema)
  - datatype mismatches
  - high missingness / null rates
  - stale timestamps
  - feature availability changes

Thresholds are configurable per-feature with sensible defaults.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SchemaStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class FeatureAvailability(str, Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"      # >20% null
    UNAVAILABLE = "unavailable"  # >80% null or missing entirely
    STALE = "stale"            # timestamp older than threshold


@dataclass
class FeatureDriftResult:
    """Result for a single feature's availability/schema check."""
    feature: str
    status: FeatureAvailability
    null_rate: float = 0.0
    missing_from_input: bool = False
    unexpected: bool = False
    type_mismatch: bool = False
    stale: bool = False
    detail: str = ""


@dataclass
class SchemaDriftReport:
    """Complete schema/availability drift report for one observation window."""
    timestamp: float
    n_observations: int
    expected_features: list[str]
    observed_features: list[str]
    feature_results: dict[str, FeatureDriftResult]
    overall_status: SchemaStatus
    missing_features: list[str]
    unexpected_features: list[str]
    degraded_features: list[str]
    unavailable_features: list[str]
    stale_features: list[str]
    type_mismatches: list[str]
    n_missing: int = 0
    n_unexpected: int = 0
    n_degraded: int = 0
    n_unavailable: int = 0
    n_stale: int = 0
    n_type_mismatches: int = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "n_observations": self.n_observations,
            "n_expected_features": len(self.expected_features),
            "n_observed_features": len(self.observed_features),
            "overall_status": self.overall_status.value,
            "missing_features": self.missing_features,
            "unexpected_features": self.unexpected_features,
            "degraded_features": self.degraded_features,
            "unavailable_features": self.unavailable_features,
            "stale_features": self.stale_features,
            "type_mismatches": self.type_mismatches,
            "n_missing": self.n_missing,
            "n_unexpected": self.n_unexpected,
            "n_degraded": self.n_degraded,
            "n_unavailable": self.n_unavailable,
            "n_stale": self.n_stale,
            "n_type_mismatches": self.n_type_mismatches,
        }


# Default thresholds
DEFAULT_NULL_WARN = 0.20      # >20% null -> degraded
DEFAULT_NULL_CRIT = 0.80      # >80% null -> unavailable
DEFAULT_STALE_SECONDS = 86400  # 24 hours
DEFAULT_MIN_OBSERVATIONS = 10


def check_schema(
    expected_features: list[str],
    observed_columns: list[str],
) -> SchemaDriftReport:
    """Check if observed columns match the expected feature schema.

    This is a static schema check (no data values, just column presence).
    """
    expected_set = set(expected_features)
    observed_set = set(observed_columns)

    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)

    feature_results: dict[str, FeatureDriftResult] = {}

    for f in expected_features:
        if f in missing:
            feature_results[f] = FeatureDriftResult(
                feature=f,
                status=FeatureAvailability.UNAVAILABLE,
                missing_from_input=True,
                detail="feature missing from input schema",
            )
        else:
            feature_results[f] = FeatureDriftResult(
                feature=f,
                status=FeatureAvailability.AVAILABLE,
            )

    for f in unexpected:
        feature_results[f] = FeatureDriftResult(
            feature=f,
            status=FeatureAvailability.AVAILABLE,
            unexpected=True,
            detail="unexpected feature present",
        )

    overall = SchemaStatus.OK
    if missing:
        overall = SchemaStatus.CRITICAL
    elif unexpected:
        overall = SchemaStatus.WARNING

    return SchemaDriftReport(
        timestamp=time.time(),
        n_observations=0,
        expected_features=expected_features,
        observed_features=observed_columns,
        feature_results=feature_results,
        overall_status=overall,
        missing_features=missing,
        unexpected_features=unexpected,
        degraded_features=[],
        unavailable_features=missing,
        stale_features=[],
        type_mismatches=[],
        n_missing=len(missing),
        n_unexpected=len(unexpected),
    )


def check_feature_availability(
    data: dict[str, list],
    expected_features: list[str],
    null_warn: float = DEFAULT_NULL_WARN,
    null_crit: float = DEFAULT_NULL_CRIT,
    timestamp_features: dict[str, float] | None = None,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> SchemaDriftReport:
    """Check feature availability in a batch of observations.

    Args:
        data: dict mapping feature_name -> list of values (all same length)
        expected_features: canonical feature list
        null_warn: null rate threshold for WARNING
        null_crit: null rate threshold for CRITICAL
        timestamp_features: optional dict mapping feature_name -> latest timestamp (epoch)
        stale_seconds: how old a timestamp must be to flag as stale
        min_observations: minimum observations for meaningful check
    """
    if not data:
        return SchemaDriftReport(
            timestamp=time.time(),
            n_observations=0,
            expected_features=expected_features,
            observed_features=[],
            feature_results={},
            overall_status=SchemaStatus.INSUFFICIENT_DATA,
            missing_features=[],
            unexpected_features=[],
            degraded_features=[],
            unavailable_features=[],
            stale_features=[],
            type_mismatches=[],
        )

    observed_columns = list(data.keys())
    n_obs = max(len(v) for v in data.values()) if data else 0

    if n_obs < min_observations:
        return SchemaDriftReport(
            timestamp=time.time(),
            n_observations=n_obs,
            expected_features=expected_features,
            observed_features=observed_columns,
            feature_results={},
            overall_status=SchemaStatus.INSUFFICIENT_DATA,
            missing_features=[],
            unexpected_features=[],
            degraded_features=[],
            unavailable_features=[],
            stale_features=[],
            type_mismatches=[],
        )

    expected_set = set(expected_features)
    observed_set = set(observed_columns)

    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)

    feature_results: dict[str, FeatureDriftResult] = {}
    degraded = []
    unavailable = []
    stale = []
    type_mismatches = []

    for f in expected_features:
        if f in missing:
            feature_results[f] = FeatureDriftResult(
                feature=f,
                status=FeatureAvailability.UNAVAILABLE,
                missing_from_input=True,
                null_rate=1.0,
                detail="feature missing from input",
            )
            unavailable.append(f)
            continue

        values = data.get(f, [])
        n_total = len(values)
        if n_total == 0:
            feature_results[f] = FeatureDriftResult(
                feature=f,
                status=FeatureAvailability.UNAVAILABLE,
                null_rate=1.0,
                detail="feature present but empty",
            )
            unavailable.append(f)
            continue

        n_null = sum(1 for v in values if v is None or (isinstance(v, float) and v != v))
        null_rate = n_null / n_total

        status = FeatureAvailability.AVAILABLE
        detail = ""

        if null_rate >= null_crit:
            status = FeatureAvailability.UNAVAILABLE
            detail = f"null rate {null_rate:.1%} >= {null_crit:.1%}"
            unavailable.append(f)
        elif null_rate >= null_warn:
            status = FeatureAvailability.DEGRADED
            detail = f"null rate {null_rate:.1%} >= {null_warn:.1%}"
            degraded.append(f)

        # Check type consistency
        non_null_types = set()
        for v in values:
            if v is not None:
                non_null_types.add(type(v).__name__)
        if len(non_null_types) > 1:
            status = FeatureAvailability.DEGRADED
            type_mismatches.append(f)
            detail = f"mixed types: {non_null_types}"

        # Check staleness
        if timestamp_features and f in timestamp_features:
            ts = timestamp_features[f]
            if time.time() - ts > stale_seconds:
                stale.append(f)
                if status == FeatureAvailability.AVAILABLE:
                    status = FeatureAvailability.STALE
                    detail = f"last updated {time.time() - ts:.0f}s ago (>{stale_seconds}s)"

        feature_results[f] = FeatureDriftResult(
            feature=f,
            status=status,
            null_rate=null_rate,
            type_mismatch=f in type_mismatches,
            stale=f in stale,
            detail=detail,
        )

    for f in unexpected:
        feature_results[f] = FeatureDriftResult(
            feature=f,
            status=FeatureAvailability.AVAILABLE,
            unexpected=True,
            detail="unexpected feature in input",
        )

    # Overall status: worst severity across all features
    overall = SchemaStatus.OK
    if unavailable or missing:
        overall = SchemaStatus.CRITICAL
    elif degraded or unexpected or stale or type_mismatches:
        overall = SchemaStatus.WARNING

    return SchemaDriftReport(
        timestamp=time.time(),
        n_observations=n_obs,
        expected_features=expected_features,
        observed_features=observed_columns,
        feature_results=feature_results,
        overall_status=overall,
        missing_features=missing,
        unexpected_features=unexpected,
        degraded_features=degraded,
        unavailable_features=unavailable,
        stale_features=stale,
        type_mismatches=type_mismatches,
        n_missing=len(missing),
        n_unexpected=len(unexpected),
        n_degraded=len(degraded),
        n_unavailable=len(unavailable),
        n_stale=len(stale),
        n_type_mismatches=len(type_mismatches),
    )
