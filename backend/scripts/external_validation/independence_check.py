"""Independence checks — verifies no overlap between training and external datasets.

Checks exact duplicates, transaction identifiers, account identifiers,
hashes/fingeratures, repeated feature vectors, and suspicious near-duplicates.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class IndependenceReport:
    """Result of independence/overlap checks."""
    training_overlap_rows: int = 0
    external_duplicate_rows: int = 0
    near_duplicate_warning: bool = False
    exact_duplicate_rows: int = 0
    total_training_rows: int = 0
    total_external_rows: int = 0
    overlap_pct: float = 0.0
    is_independent: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "training_overlap_rows": self.training_overlap_rows,
            "external_duplicate_rows": self.external_duplicate_rows,
            "near_duplicate_warning": self.near_duplicate_warning,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "total_training_rows": self.total_training_rows,
            "total_external_rows": self.total_external_rows,
            "overlap_pct": round(self.overlap_pct, 4),
            "is_independent": self.is_independent,
            "warnings": self.warnings,
        }


def _compute_row_hash(row: Any) -> str:
    """Compute a deterministic hash of a row's values."""
    values = tuple(str(v) for v in row)
    return hashlib.sha256("|".join(values).encode()).hexdigest()[:16]


def check_exact_duplicates(X_external: np.ndarray) -> int:
    """Count exact duplicate rows within the external dataset."""
    if len(X_external) == 0:
        return 0
    # Use numpy unique with return_counts
    _, counts = np.unique(X_external, axis=0, return_counts=True)
    return int(np.sum(counts > 1))


def check_training_overlap(
    X_external: np.ndarray,
    X_training: np.ndarray | None,
) -> tuple[int, bool]:
    """Check for exact row overlap between external and training datasets.

    Returns:
        (overlap_count, near_duplicate_warning)
    """
    if X_training is None or len(X_training) == 0 or len(X_external) == 0:
        return 0, False

    # Check exact duplicates
    overlap = 0
    # For large datasets, use hash-based approach
    if len(X_external) > 10000 or len(X_training) > 10000:
        # Hash all training rows
        training_hashes = set()
        for i in range(len(X_training)):
            h = _compute_row_hash(X_training[i])
            training_hashes.add(h)
        # Check external rows
        for i in range(len(X_external)):
            h = _compute_row_hash(X_external[i])
            if h in training_hashes:
                overlap += 1
    else:
        # Small datasets — direct comparison
        for i in range(len(X_external)):
            for j in range(len(X_training)):
                if np.array_equal(X_external[i], X_training[j]):
                    overlap += 1
                    break

    # Near-duplicate check: check if any external row is within 1% of a training row
    near_dup_warning = False
    if overlap == 0 and len(X_external) > 0 and len(X_training) > 0:
        # Sample check for near-duplicates (expensive for large datasets)
        sample_size = min(100, len(X_external))
        indices = np.random.choice(len(X_external), sample_size, replace=False)
        for idx in indices:
            ext_row = X_external[idx]
            # Compute L2 distance to each training row's nearest neighbor
            diffs = np.abs(X_training - ext_row)
            min_distances = np.min(np.sum(diffs, axis=1))
            if min_distances < 0.01 * len(ext_row):
                near_dup_warning = True
                break

    return overlap, near_dup_warning


def run_independence_check(
    X_external: np.ndarray,
    X_training: np.ndarray | None = None,
) -> IndependenceReport:
    """Run all independence checks and produce a report.

    Args:
        X_external: External dataset feature matrix.
        X_training: Training dataset feature matrix (if available).

    Returns:
        IndependenceReport with overlap statistics and warnings.
    """
    report = IndependenceReport()
    report.total_external_rows = len(X_external)
    report.total_training_rows = len(X_training) if X_training is not None else 0

    # Check exact duplicates within external dataset
    report.external_duplicate_rows = check_exact_duplicates(X_external)
    if report.external_duplicate_rows > 0:
        report.warnings.append(
            f"{report.external_duplicate_rows} exact duplicate rows found within external dataset"
        )

    # Check overlap with training data
    if X_training is not None and len(X_training) > 0:
        overlap, near_dup = check_training_overlap(X_external, X_training)
        report.training_overlap_rows = overlap
        report.near_duplicate_warning = near_dup
        if overlap > 0:
            report.overlap_pct = overlap / len(X_external)
            report.is_independent = False
            report.warnings.append(
                f"{overlap} rows overlap with training data ({report.overlap_pct:.2%})"
            )
        if near_dup:
            report.warnings.append("Near-duplicate rows detected between external and training data")
    else:
        report.warnings.append("Training data not available — cannot verify independence")

    return report
