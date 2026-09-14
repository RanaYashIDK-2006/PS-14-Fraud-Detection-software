"""Distribution-shift report — compares external vs training data distributions.

Provides interpretable warnings for significant mismatches without manufacturing
a single "distribution shift score."

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ShiftWarning:
    """A single distribution shift warning."""
    shift_type: str  # e.g., "PREVALENCE_SHIFT", "FEATURE_RANGE_SHIFT"
    feature: str = ""
    detail: str = ""
    severity: str = "warning"  # "warning", "critical"


@dataclass
class DistributionShiftReport:
    """Result of distribution shift analysis."""
    warnings: list[ShiftWarning] = field(default_factory=list)
    has_critical_shift: bool = False

    # Specific shift statistics
    prevalence_external: float = 0.0
    prevalence_training: float = 0.0
    prevalence_shift: float = 0.0

    feature_range_shifts: dict[str, dict] = field(default_factory=dict)
    missingness_shifts: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "has_critical_shift": self.has_critical_shift,
            "warning_count": len(self.warnings),
            "warnings": [
                {"type": w.shift_type, "feature": w.feature, "detail": w.detail, "severity": w.severity}
                for w in self.warnings
            ],
            "prevalence": {
                "external": round(self.prevalence_external, 6),
                "training": round(self.prevalence_training, 6),
                "shift": round(self.prevalence_shift, 6),
            },
        }


def check_prevalence_shift(
    y_external: np.ndarray,
    y_training: np.ndarray | None,
    threshold: float = 0.1,
) -> ShiftWarning | None:
    """Check if class prevalence differs significantly."""
    if y_training is None or len(y_training) == 0:
        return None

    prev_ext = np.mean(y_external == 1)
    prev_train = np.mean(y_training == 1)

    if prev_train == 0:
        return None

    relative_shift = abs(prev_ext - prev_train) / prev_train

    if relative_shift > threshold:
        return ShiftWarning(
            shift_type="PREVALENCE_SHIFT",
            detail=f"External prevalence {prev_ext:.4%} vs training {prev_train:.4%} "
                   f"(relative shift: {relative_shift:.1%})",
            severity="critical" if relative_shift > 0.5 else "warning",
        )
    return None


def check_feature_range_shifts(
    X_external: np.ndarray,
    X_training: np.ndarray | None,
    feature_names: list[str],
    threshold: float = 0.5,
) -> list[ShiftWarning]:
    """Check if feature ranges differ significantly."""
    if X_training is None or len(X_training) == 0:
        return []

    warnings = []
    for i, name in enumerate(feature_names):
        if i >= X_external.shape[1] or i >= X_training.shape[1]:
            continue

        ext_min, ext_max = np.nanmin(X_external[:, i]), np.nanmax(X_external[:, i])
        train_min, train_max = np.nanmin(X_training[:, i]), np.nanmax(X_training[:, i])

        ext_range = ext_max - ext_min
        train_range = train_max - train_min

        if train_range > 0:
            # Check if external range extends beyond training range
            if ext_min < train_min - threshold * train_range:
                warnings.append(ShiftWarning(
                    shift_type="FEATURE_RANGE_SHIFT",
                    feature=name,
                    detail=f"External min {ext_min:.4f} below training min {train_min:.4f}",
                ))
            if ext_max > train_max + threshold * train_range:
                warnings.append(ShiftWarning(
                    shift_type="FEATURE_RANGE_SHIFT",
                    feature=name,
                    detail=f"External max {ext_max:.4f} above training max {train_max:.4f}",
                ))

    return warnings


def check_missingness_shifts(
    X_external: np.ndarray,
    X_training: np.ndarray | None,
    feature_names: list[str],
    threshold: float = 0.1,
) -> list[ShiftWarning]:
    """Check if missingness patterns differ significantly."""
    if X_training is None or len(X_training) == 0:
        return []

    warnings = []
    for i, name in enumerate(feature_names):
        if i >= X_external.shape[1] or i >= X_training.shape[1]:
            continue

        ext_missing = np.mean(np.isnan(X_external[:, i]))
        train_missing = np.mean(np.isnan(X_training[:, i]))

        if abs(ext_missing - train_missing) > threshold:
            warnings.append(ShiftWarning(
                shift_type="MISSINGNESS_SHIFT",
                feature=name,
                detail=f"External missingness {ext_missing:.2%} vs training {train_missing:.2%}",
            ))

    return warnings


def run_distribution_shift_analysis(
    X_external: np.ndarray,
    y_external: np.ndarray,
    X_training: np.ndarray | None = None,
    y_training: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> DistributionShiftReport:
    """Run full distribution shift analysis.

    Args:
        X_external: External dataset features.
        y_external: External dataset labels.
        X_training: Training dataset features (if available).
        y_training: Training dataset labels (if available).
        feature_names: Feature names for reporting.

    Returns:
        DistributionShiftReport with all shift warnings.
    """
    report = DistributionShiftReport()

    # Prevalence shift
    if y_training is not None and len(y_training) > 0:
        report.prevalence_external = float(np.mean(y_external == 1))
        report.prevalence_training = float(np.mean(y_training == 1))
        report.prevalence_shift = abs(report.prevalence_external - report.prevalence_training)

    prevalence_warning = check_prevalence_shift(y_external, y_training)
    if prevalence_warning:
        report.warnings.append(prevalence_warning)
        if prevalence_warning.severity == "critical":
            report.has_critical_shift = True

    # Feature range shifts
    if feature_names:
        range_warnings = check_feature_range_shifts(X_external, X_training, feature_names)
        report.warnings.extend(range_warnings)

        # Missingness shifts
        missing_warnings = check_missingness_shifts(X_external, X_training, feature_names)
        report.warnings.extend(missing_warnings)

    return report
