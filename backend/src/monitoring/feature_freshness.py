"""Feature freshness and staleness detection.

Validates that time-sensitive features are not stale. A stale feature
must not silently appear current.

Tracks:
  - feature computation timestamp
  - feature maximum acceptable age
  - staleness classification
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class FreshnessStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"  # no timestamp available


@dataclass
class FreshnessResult:
    """Result of a freshness check for one feature."""
    feature: str
    status: FreshnessStatus
    age_seconds: float | None = None
    max_age_seconds: float | None = None
    detail: str = ""


# Default freshness requirements per feature category
# (seconds since computation; features older than this are STALE)
FRESHNESS_REQUIREMENTS: dict[str, float] = {
    # Transaction-derived: computed at ingest, valid for the current window
    "amount_ratio": 3600,          # 1 hour
    "txn_freq_last_24h": 3600,     # 1 hour
    "txn_time_unusual": 86400,     # 24 hours (hour-of-day doesn't change fast)
    "new_device_flag": 3600,       # 1 hour
    "unusual_location_flag": 3600, # 1 hour
    "unusual_recipient_flag": 3600, # 1 hour
    "failed_auth_count_24h": 3600, # 1 hour
    "hour_of_day": 86400,          # 24 hours (stable within a day)
    "is_weekend": 86400,           # 24 hours

    # Historical/behavioral: updated on commit
    "days_since_last_similar_txn": 86400,    # 24 hours
    "gradual_escalation_score": 86400,       # 24 hours
    "known_device_count": 86400,             # 24 hours
    "account_tenure_days": 86400,            # 24 hours (changes slowly)

    # Aggregated: rolling counts
    "shared_device_accounts": 86400,         # 24 hours
    "shared_recipient_accounts": 86400,      # 24 hours
    "mule_ring_score": 86400,                # 24 hours

    # Deviation features
    "hour_deviation": 86400,         # 24 hours
    "amount_zscore": 86400,          # 24 hours
    "velocity_deviation": 86400,     # 24 hours
    "recipient_novelty": 86400,      # 24 hours
    "txn_regularity": 86400,         # 24 hours
}


def check_feature_freshness(
    feature_timestamps: dict[str, float],
    current_time: float | None = None,
    custom_requirements: dict[str, float] | None = None,
) -> list[FreshnessResult]:
    """Check freshness of features based on their computation timestamps.

    Args:
        feature_timestamps: dict mapping feature_name -> computation timestamp (epoch)
        current_time: current timestamp (defaults to time.time())
        custom_requirements: override default freshness requirements

    Returns:
        List of FreshnessResult for each feature with a timestamp.
    """
    if current_time is None:
        current_time = time.time()

    requirements = dict(FRESHNESS_REQUIREMENTS)
    if custom_requirements:
        requirements.update(custom_requirements)

    results: list[FreshnessResult] = []

    for feature, ts in feature_timestamps.items():
        max_age = requirements.get(feature)

        if max_age is None:
            results.append(FreshnessResult(
                feature=feature,
                status=FreshnessStatus.UNKNOWN,
                detail="no freshness requirement defined",
            ))
            continue

        age = current_time - ts

        if age <= max_age:
            results.append(FreshnessResult(
                feature=feature,
                status=FreshnessStatus.FRESH,
                age_seconds=round(age, 1),
                max_age_seconds=max_age,
                detail=f"age {age:.0f}s <= max {max_age:.0f}s",
            ))
        else:
            results.append(FreshnessResult(
                feature=feature,
                status=FreshnessStatus.STALE,
                age_seconds=round(age, 1),
                max_age_seconds=max_age,
                detail=f"age {age:.0f}s > max {max_age:.0f}s",
            ))

    return results


def check_vector_freshness(
    features: dict[str, any],
    feature_timestamps: dict[str, float],
    current_time: float | None = None,
) -> tuple[bool, list[FreshnessResult]]:
    """Check if a feature vector is fresh enough for inference.

    Returns (is_fresh_enough, list_of_results).
    A vector is NOT fresh enough if any required feature is stale.
    """
    results = check_feature_freshness(feature_timestamps, current_time)
    stale = [r for r in results if r.status == FreshnessStatus.STALE]
    return len(stale) == 0, results


def get_stale_features(results: list[FreshnessResult]) -> list[str]:
    """Extract names of stale features from freshness results."""
    return [r.feature for r in results if r.status == FreshnessStatus.STALE]
