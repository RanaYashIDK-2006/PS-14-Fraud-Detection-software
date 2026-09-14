"""Label validation — verifies fraud labels are present, valid, and well-documented.

Checks that labels exist, have expected values, positive/negative classes are
identifiable, unknown labels are handled, and label meaning is documented.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metadata_schema import ExternalDatasetMetadata


@dataclass
class LabelValidationReport:
    """Result of label validation."""
    label_exists: bool = False
    label_valid: bool = False
    unique_values: list[Any] = field(default_factory=list)
    num_positive: int = 0
    num_negative: int = 0
    num_unknown: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    label_timing_documented: bool = False
    label_leakage_detected: bool = False

    def to_dict(self) -> dict:
        return {
            "label_exists": self.label_exists,
            "label_valid": self.label_valid,
            "unique_values": [str(v) for v in self.unique_values],
            "num_positive": self.num_positive,
            "num_negative": self.num_negative,
            "num_unknown": self.num_unknown,
            "warnings": self.warnings,
            "errors": self.errors,
            "label_timing_documented": self.label_timing_documented,
            "label_leakage_detected": self.label_leakage_detected,
        }


def validate_labels(
    y: np.ndarray,
    meta: ExternalDatasetMetadata,
) -> LabelValidationReport:
    """Validate the label column of an external dataset.

    Args:
        y: Label values as numpy array.
        meta: External dataset metadata with label configuration.

    Returns:
        LabelValidationReport with validation results.
    """
    report = LabelValidationReport()

    # Check label exists
    if y is None or len(y) == 0:
        report.errors.append("No label values provided")
        return report

    report.label_exists = True

    # Get unique values
    unique_vals = np.unique(y)
    report.unique_values = unique_vals.tolist()

    # Check expected values match metadata
    pos_val = meta.positive_class_value
    neg_val = meta.negative_class_value

    # Count classes
    report.num_positive = int(np.sum(y == pos_val))
    report.num_negative = int(np.sum(y == neg_val))

    # Check for unknown values
    known_vals = {pos_val, neg_val}
    unknown_mask = ~np.isin(y, list(known_vals))
    report.num_unknown = int(np.sum(unknown_mask))

    if report.num_unknown > 0:
        report.warnings.append(
            f"{report.num_unknown} labels have values different from expected "
            f"positive ({pos_val}) and negative ({neg_val})"
        )

    # Build class distribution
    for val in unique_vals:
        count = int(np.sum(y == val))
        report.class_distribution[str(val)] = count

    # Check class balance
    total = len(y)
    if report.num_positive == 0:
        report.errors.append("No positive (fraud) cases found")
        return report

    if report.num_negative == 0:
        report.errors.append("No negative (legitimate) cases found")
        return report

    prevalence = report.num_positive / total
    if prevalence < 0.0001:
        report.warnings.append(
            f"Very low prevalence ({prevalence:.4%}) — statistical instability likely"
        )
    elif prevalence > 0.5:
        report.warnings.append(
            f"Very high prevalence ({prevalence:.2%}) — unusual for fraud detection"
        )

    # Check label timing documentation
    report.label_timing_documented = meta.fraud_label_timing not in ("", "unknown")
    if not report.label_timing_documented:
        report.warnings.append("Fraud label timing not documented — labels may be post-event")

    # Mark as valid if no errors
    report.label_valid = len(report.errors) == 0

    return report
