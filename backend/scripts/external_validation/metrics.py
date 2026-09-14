"""Metrics calculation — computes all supported metrics for external evaluation.

Where labels and sample size permit, calculates ROC-AUC, PR-AUC, recall,
precision, FPR, confusion matrix, recall at selected FPR operating points,
prevalence, and sample sizes.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvaluationMetrics:
    """Complete metrics for an external evaluation."""
    # Basic counts
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    prevalence: float = 0.0

    # Core metrics
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    fpr: float = 0.0
    fnr: float = 0.0
    f1: float = 0.0

    # Confusion matrix
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    # Operating-point metrics
    recall_at_1pct_fpr: float = 0.0
    recall_at_5pct_fpr: float = 0.0

    # Threshold used
    threshold: float = 0.5

    # Statistical warnings
    warnings: list[str] = field(default_factory=list)
    small_sample: bool = False

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "prevalence": round(self.prevalence, 6),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "fpr": round(self.fpr, 4),
            "fnr": round(self.fnr, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "recall_at_1pct_fpr": round(self.recall_at_1pct_fpr, 4),
            "recall_at_5pct_fpr": round(self.recall_at_5pct_fpr, 4),
            "threshold": self.threshold,
            "warnings": self.warnings,
            "small_sample": self.small_sample,
        }


def _compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC using numpy (no sklearn dependency)."""
    # Sort by score descending
    desc_order = np.argsort(-y_score)
    y_true_sorted = y_true[desc_order]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Compute TPR and FPR at each threshold
    tpr_values = np.cumsum(y_true_sorted) / n_pos
    fpr_values = np.cumsum(1 - y_true_sorted) / n_neg

    # Add origin
    tpr_values = np.concatenate([[0], tpr_values])
    fpr_values = np.concatenate([[0], fpr_values])

    # AUC = area under the ROC curve (trapezoidal)
    auc = np.trapezoid(tpr_values, fpr_values)
    return float(abs(auc))


def _compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC using numpy."""
    desc_order = np.argsort(-y_score)
    y_true_sorted = y_true[desc_order]

    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0.0

    tp_cumsum = np.cumsum(y_true_sorted)
    total_cumsum = np.arange(1, len(y_true_sorted) + 1)

    precision_values = tp_cumsum / total_cumsum
    recall_values = tp_cumsum / n_pos

    # Add origin
    precision_values = np.concatenate([[1.0], precision_values])
    recall_values = np.concatenate([[0.0], recall_values])

    # AUC = area under the PR curve
    auc = np.trapezoid(precision_values, recall_values)
    return float(abs(auc))


def _compute_recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float
) -> float:
    """Compute recall at a specific FPR operating point."""
    n_neg = np.sum(y_true == 0)
    n_pos = np.sum(y_true == 1)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    desc_order = np.argsort(-y_score)
    y_true_sorted = y_true[desc_order]

    fp_count = 0
    tp_count = 0
    best_recall = 0.0

    for label in y_true_sorted:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
            current_fpr = fp_count / n_neg
            current_recall = tp_count / n_pos
            if current_fpr <= target_fpr:
                best_recall = current_recall
            elif current_fpr > target_fpr:
                break

    return best_recall


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """Compute all supported evaluation metrics.

    Args:
        y_true: True labels (0/1).
        y_pred: Predicted labels (0/1).
        y_prob: Predicted probabilities (optional, for AUC metrics).
        threshold: Classification threshold used.

    Returns:
        EvaluationMetrics with all computed metrics.
    """
    metrics = EvaluationMetrics()
    metrics.threshold = threshold

    # Basic counts
    metrics.n_samples = len(y_true)
    metrics.n_positive = int(np.sum(y_true == 1))
    metrics.n_negative = int(np.sum(y_true == 0))
    metrics.prevalence = metrics.n_positive / metrics.n_samples if metrics.n_samples > 0 else 0.0

    # Statistical warnings
    if metrics.n_positive < 30:
        metrics.warnings.append(
            f"Very few positive cases ({metrics.n_positive}) — metrics may be unreliable"
        )
        metrics.small_sample = True
    elif metrics.n_positive < 100:
        metrics.warnings.append(
            f"Few positive cases ({metrics.n_positive}) — interpret with caution"
        )

    # Confusion matrix
    metrics.tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    metrics.fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    metrics.tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    metrics.fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    # Basic metrics
    metrics.recall = metrics.tp / (metrics.tp + metrics.fn) if (metrics.tp + metrics.fn) > 0 else 0.0
    metrics.precision = metrics.tp / (metrics.tp + metrics.fp) if (metrics.tp + metrics.fp) > 0 else 0.0
    metrics.fpr = metrics.fp / (metrics.fp + metrics.tn) if (metrics.fp + metrics.tn) > 0 else 0.0
    metrics.fnr = metrics.fn / (metrics.tp + metrics.fn) if (metrics.tp + metrics.fn) > 0 else 0.0
    metrics.f1 = (
        2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        if (metrics.precision + metrics.recall) > 0
        else 0.0
    )

    # AUC metrics (require probabilities)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics.roc_auc = _compute_roc_auc(y_true, y_prob)
        metrics.pr_auc = _compute_pr_auc(y_true, y_prob)
        metrics.recall_at_1pct_fpr = _compute_recall_at_fpr(y_true, y_prob, 0.01)
        metrics.recall_at_5pct_fpr = _compute_recall_at_fpr(y_true, y_prob, 0.05)

    return metrics
