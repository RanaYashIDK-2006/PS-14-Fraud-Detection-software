"""Runtime enforcement layer — wires Phase 41 safeguards into the inference path.

This module provides a single function that the Risk Engine calls BEFORE
model inference. It validates the feature vector against the Phase 41
contract and returns an enforcement result that the caller uses to decide
whether to proceed with inference or fall back to safe behavior.

Enforcement is CENTRALIZED: every inference path goes through this function.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    RobustnessResult,
)
from src.monitoring.feature_freshness import (
    check_vector_freshness,
    FreshnessResult,
    FreshnessStatus,
)


class EnforcementVerdict(str, Enum):
    """What the enforcement layer decides."""
    PROCEED = "proceed"           # all checks passed, safe to infer
    PROCEED_WITH_WARNINGS = "proceed_with_warnings"  # non-critical issues
    BLOCK_INFERENCE = "block_inference"  # critical issue, do not infer
    FALLBACK_RULES = "fallback_rules"   # ML unavailable, use rules-only


@dataclass
class EnforcementResult:
    """Result of the runtime enforcement check."""
    verdict: EnforcementVerdict
    processed_features: dict[str, float]
    validation_issues: list[str]
    numerical_issues: list[str]
    freshness_issues: list[str]
    ordering_correct: bool
    n_valid: int
    n_missing: int
    n_invalid: int
    n_clamped: int
    n_imputed: int
    enforcement_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "validation_issues": self.validation_issues,
            "numerical_issues": self.numerical_issues,
            "freshness_issues": self.freshness_issues,
            "ordering_correct": self.ordering_correct,
            "n_valid": self.n_valid,
            "n_missing": self.n_missing,
            "n_invalid": self.n_invalid,
            "n_clamped": self.n_clamped,
            "n_imputed": self.n_imputed,
            "enforcement_time_ms": round(self.enforcement_time_ms, 3),
        }


def enforce_before_inference(
    features: dict[str, Any],
    feature_timestamps: dict[str, float] | None = None,
    strict_ordering: bool = False,
) -> EnforcementResult:
    """Run Phase 41 enforcement checks BEFORE model inference.

    This is the single entry point that every inference path should call.
    It does NOT modify the original features dict — it returns processed_features
    that the caller should use for inference.

    Args:
        features: raw feature dict from the Privacy Layer / API
        feature_timestamps: optional dict mapping feature -> computation time (epoch)
        strict_ordering: if True, wrong ordering is a BLOCK (default: warning)

    Returns:
        EnforcementResult with verdict and processed features.
    """
    t0 = time.monotonic()
    validation_issues: list[str] = []
    numerical_issues: list[str] = []
    freshness_issues: list[str] = []

    # 1. Feature ordering check
    ordering_correct, ordering_mismatches = check_feature_ordering(features)
    if not ordering_correct:
        msg = f"feature ordering mismatch: {ordering_mismatches[:3]}"
        if strict_ordering:
            validation_issues.append(f"BLOCK: {msg}")
        else:
            validation_issues.append(f"WARN: {msg}")

    # 2. Contract validation (catches missing required, invalid types, out-of-range)
    validation_results = validate_feature_vector(features)
    n_valid = 0
    n_missing = 0
    n_invalid = 0

    for name, (status, detail) in validation_results.items():
        spec = ML_FEATURE_CONTRACT.get(name)
        if status == FeatureStatus.AVAILABLE:
            n_valid += 1
        elif status == FeatureStatus.MISSING:
            n_missing += 1
            if spec and spec.missing_policy.value == "reject":
                validation_issues.append(f"BLOCK: required feature '{name}' is missing")
            else:
                validation_issues.append(f"WARN: feature '{name}' is missing")
        elif status == FeatureStatus.INVALID:
            n_invalid += 1
            # Invalid values are BLOCKED — they indicate corrupt/malicious input
            validation_issues.append(f"BLOCK: feature '{name}' invalid: {detail}")

    # 3. Numerical robustness (process the vector)
    processed_features, robustness_results = process_feature_vector(features)
    n_clamped = sum(1 for r in robustness_results if r.action_taken == "clamped")
    n_imputed = sum(1 for r in robustness_results if r.action_taken == "imputed")

    # Check for rejections (required features that are null)
    rejections = [r for r in robustness_results if r.action_taken == "rejected"]
    if rejections:
        for r in rejections:
            numerical_issues.append(f"BLOCK: feature '{r.feature}' rejected (required, null)")

    # Check for non-finite values that survived to processed output
    for r in robustness_results:
        if r.was_non_finite:
            numerical_issues.append(f"WARN: feature '{r.feature}' had non-finite input, replaced with {r.processed_value}")

    # 4. Feature freshness (if timestamps available)
    if feature_timestamps:
        is_fresh, freshness_results = check_vector_freshness(features, feature_timestamps)
        stale = [r for r in freshness_results if r.status == FreshnessStatus.STALE]
        if stale:
            for r in stale:
                freshness_issues.append(f"WARN: feature '{r.feature}' is stale (age {r.age_seconds:.0f}s > max {r.max_age_seconds:.0f}s)")

    # 5. Determine verdict
    has_blocks = (
        any("BLOCK:" in i for i in validation_issues)
        or any("BLOCK:" in i for i in numerical_issues)
    )

    if has_blocks:
        verdict = EnforcementVerdict.BLOCK_INFERENCE
    elif validation_issues or numerical_issues or freshness_issues:
        verdict = EnforcementVerdict.PROCEED_WITH_WARNINGS
    else:
        verdict = EnforcementVerdict.PROCEED

    enforcement_time_ms = (time.monotonic() - t0) * 1000

    return EnforcementResult(
        verdict=verdict,
        processed_features=processed_features,
        validation_issues=validation_issues,
        numerical_issues=numerical_issues,
        freshness_issues=freshness_issues,
        ordering_correct=ordering_correct,
        n_valid=n_valid,
        n_missing=n_missing,
        n_invalid=n_invalid,
        n_clamped=n_clamped,
        n_imputed=n_imputed,
        enforcement_time_ms=enforcement_time_ms,
    )


def get_feature_contract_version() -> str:
    """Return the current feature contract version."""
    return ML_FEATURE_VERSION


def get_expected_feature_count() -> int:
    """Return the expected number of features."""
    return len(ML_FEATURE_ORDER)


def get_expected_feature_names() -> list[str]:
    """Return the expected feature names in order."""
    return list(ML_FEATURE_ORDER)
