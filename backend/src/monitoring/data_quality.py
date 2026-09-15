"""Data quality gate — combines all Phase 41 checks into a single assessment.

Wires together:
  - Feature contract validation
  - Numerical robustness
  - Feature freshness
  - Temporal causality

Produces a single DATA_QUALITY_OK / WARNING / BLOCKED status per vector.

DRIFT (Phase 40) is distinct from DATA_QUALITY. A feature can be in-distribution
but still be stale, malformed, or causally invalid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.monitoring.feature_contract import (
    ML_FEATURE_CONTRACT,
    ML_FEATURE_ORDER,
    ML_FEATURE_VERSION,
    FeatureStatus,
    validate_feature_vector,
    check_feature_ordering,
)
from src.monitoring.numerical_robustness import (
    process_feature_vector,
    has_rejections,
    get_quality_summary,
    RobustnessResult,
)
from src.monitoring.feature_freshness import (
    check_vector_freshness,
    FreshnessResult,
    FreshnessStatus,
)
from src.monitoring.temporal_safeguards import (
    check_timestamp_safety,
    TemporalCheckResult,
    TemporalCheckStatus,
)


class DataQualityStatus(str, Enum):
    OK = "data_quality_ok"
    WARNING = "data_quality_warning"
    BLOCKED = "data_quality_blocked"


@dataclass
class DataQualityReport:
    """Complete data quality assessment for one feature vector."""
    timestamp: float
    overall_status: DataQualityStatus
    feature_version: str
    n_features: int
    n_valid: int
    n_missing: int
    n_invalid: int
    n_stale: int
    n_clamped: int
    n_imputed: int
    n_rejected: int
    validation_results: dict[str, tuple[FeatureStatus, str]]
    robustness_results: list[RobustnessResult]
    freshness_results: list[FreshnessResult]
    temporal_results: list[TemporalCheckResult]
    ordering_correct: bool
    ordering_mismatches: list[str]
    blocked_reasons: list[str]
    warning_reasons: list[str]
    processed_features: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "feature_version": self.feature_version,
            "n_features": self.n_features,
            "n_valid": self.n_valid,
            "n_missing": self.n_missing,
            "n_invalid": self.n_invalid,
            "n_stale": self.n_stale,
            "n_clamped": self.n_clamped,
            "n_imputed": self.n_imputed,
            "n_rejected": self.n_rejected,
            "ordering_correct": self.ordering_correct,
            "blocked_reasons": self.blocked_reasons,
            "warning_reasons": self.warning_reasons,
            "quality_summary": {
                r.action_taken: sum(1 for x in self.robustness_results if x.action_taken == r.action_taken)
                for r in self.robustness_results
            } if self.robustness_results else {},
        }


def assess_data_quality(
    features: dict[str, Any],
    feature_timestamps: dict[str, float] | None = None,
    event_timestamp: datetime | None = None,
    require_correct_ordering: bool = False,
    feature_version: str = ML_FEATURE_VERSION,
) -> DataQualityReport:
    """Run the complete data quality assessment pipeline.

    Args:
        features: raw feature vector from the Privacy Layer
        feature_timestamps: optional dict mapping feature -> computation timestamp
        event_timestamp: optional event timestamp for temporal checks
        require_correct_ordering: if True, incorrect ordering is a BLOCK
        feature_version: expected feature version

    Returns:
        DataQualityReport with overall status and detailed results.
    """
    blocked_reasons: list[str] = []
    warning_reasons: list[str] = []

    # 1. Feature ordering check
    ordering_correct, ordering_mismatches = check_feature_ordering(features)
    if not ordering_correct and require_correct_ordering:
        blocked_reasons.append(f"feature ordering incorrect: {ordering_mismatches}")
    elif not ordering_correct:
        warning_reasons.append(f"feature ordering differs from contract: {ordering_mismatches}")

    # 2. Contract validation
    validation_results = validate_feature_vector(features)

    n_valid = sum(1 for s, _ in validation_results.values() if s == FeatureStatus.AVAILABLE)
    n_missing = sum(1 for s, _ in validation_results.values() if s == FeatureStatus.MISSING)
    n_invalid = sum(1 for s, _ in validation_results.values() if s == FeatureStatus.INVALID)

    # Missing required features block inference
    for name, (status, detail) in validation_results.items():
        spec = ML_FEATURE_CONTRACT.get(name)
        if spec and status == FeatureStatus.MISSING and spec.missing_policy.value == "reject":
            blocked_reasons.append(f"required feature '{name}' is missing")
        elif status == FeatureStatus.INVALID:
            warning_reasons.append(f"feature '{name}' invalid: {detail}")

    # 3. Numerical robustness (process the vector)
    processed_features, robustness_results = process_feature_vector(features)
    quality_summary = get_quality_summary(robustness_results)

    n_clamped = sum(1 for r in robustness_results if r.action_taken == "clamped")
    n_imputed = sum(1 for r in robustness_results if r.action_taken == "imputed")
    n_rejected = sum(1 for r in robustness_results if r.action_taken == "rejected")

    if has_rejections(robustness_results):
        blocked_reasons.append(f"{n_rejected} features rejected (required but null)")

    # 4. Feature freshness
    freshness_results: list[FreshnessResult] = []
    n_stale = 0
    if feature_timestamps:
        is_fresh, freshness_results = check_vector_freshness(features, feature_timestamps)
        n_stale = sum(1 for r in freshness_results if r.status == FreshnessStatus.STALE)
        if not is_fresh:
            stale_names = [r.feature for r in freshness_results if r.status == FreshnessStatus.STALE]
            warning_reasons.append(f"{n_stale} stale features: {stale_names}")

    # 5. Temporal checks
    temporal_results: list[TemporalCheckResult] = []
    if event_timestamp is not None:
        temporal_results = check_timestamp_safety(event_timestamp)
        for tr in temporal_results:
            if tr.status == TemporalCheckStatus.FAIL:
                blocked_reasons.append(f"temporal check failed: {tr.detail}")
            elif tr.status == TemporalCheckStatus.WARNING:
                warning_reasons.append(f"temporal warning: {tr.detail}")

    # 6. Determine overall status
    if blocked_reasons:
        overall = DataQualityStatus.BLOCKED
    elif warning_reasons:
        overall = DataQualityStatus.WARNING
    else:
        overall = DataQualityStatus.OK

    return DataQualityReport(
        timestamp=time.time(),
        overall_status=overall,
        feature_version=feature_version,
        n_features=len(ML_FEATURE_ORDER),
        n_valid=n_valid,
        n_missing=n_missing,
        n_invalid=n_invalid,
        n_stale=n_stale,
        n_clamped=n_clamped,
        n_imputed=n_imputed,
        n_rejected=n_rejected,
        validation_results=validation_results,
        robustness_results=robustness_results,
        freshness_results=freshness_results,
        temporal_results=temporal_results,
        ordering_correct=ordering_correct,
        ordering_mismatches=ordering_mismatches,
        blocked_reasons=blocked_reasons,
        warning_reasons=warning_reasons,
        processed_features=processed_features,
    )
