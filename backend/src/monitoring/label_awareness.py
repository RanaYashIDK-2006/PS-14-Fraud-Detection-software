"""Label-awareness monitoring — distinguishes prediction monitoring from outcome monitoring.

When verified labels are unavailable:
  - ALLOW: feature drift, schema drift, feature availability, prediction drift
  - DO NOT: pretend prediction drift = fraud-rate drift

When verified labels become available:
  - compute observed fraud prevalence
  - compare against reference period
  - calculate outcome-based metrics where sample size permits

Key rule: monitoring must never use future outcomes as decision-time features.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LabelAvailability:
    """Tracks whether outcome labels are available for a monitoring window."""
    available: bool
    n_labeled: int = 0
    n_total: int = 0
    fraud_count: int = 0
    fraud_rate: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "n_labeled": self.n_labeled,
            "n_total": self.n_total,
            "fraud_count": self.fraud_count,
            "fraud_rate": round(self.fraud_rate, 6),
            "detail": self.detail,
        }


@dataclass
class OutcomeMonitoringReport:
    """Report when labels ARE available — outcome-based monitoring."""
    timestamp: float
    label_availability: LabelAvailability
    observed_fraud_rate: float
    reference_fraud_rate: float | None
    fraud_rate_shift: float
    n_labeled: int
    outcome_metrics: dict[str, float]  # precision, recall, etc. if computable
    overall_status: str  # "stable", "warning", "critical", "insufficient_data"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "label_availability": self.label_availability.to_dict(),
            "observed_fraud_rate": round(self.observed_fraud_rate, 6),
            "reference_fraud_rate": round(self.reference_fraud_rate, 6) if self.reference_fraud_rate is not None else None,
            "fraud_rate_shift": round(self.fraud_rate_shift, 6),
            "n_labeled": self.n_labeled,
            "outcome_metrics": {k: round(v, 6) for k, v in self.outcome_metrics.items()},
            "overall_status": self.overall_status,
        }


def check_label_availability(
    labels: list[int | None],
    min_labels: int = 30,
) -> LabelAvailability:
    """Check if verified outcome labels are available for this window.

    Args:
        labels: list of labels (1=fraud, 0=legit, None=unknown/pending)
        min_labels: minimum labeled observations for meaningful outcome monitoring
    """
    n_total = len(labels)
    n_labeled = sum(1 for l in labels if l is not None)

    if n_labeled < min_labels:
        return LabelAvailability(
            available=False,
            n_labeled=n_labeled,
            n_total=n_total,
            detail=f"only {n_labeled} labeled (need >= {min_labels})",
        )

    fraud_count = sum(1 for l in labels if l == 1)
    fraud_rate = fraud_count / n_labeled if n_labeled > 0 else 0.0

    return LabelAvailability(
        available=True,
        n_labeled=n_labeled,
        n_total=n_total,
        fraud_count=fraud_count,
        fraud_rate=fraud_rate,
        detail=f"{n_labeled}/{n_total} labeled, {fraud_count} fraud",
    )


def check_outcome_drift(
    current_labels: list[int | None],
    reference_fraud_rate: float | None = None,
    fraud_rate_warn_shift: float = 0.02,
    fraud_rate_crit_shift: float = 0.05,
    min_labels: int = 30,
) -> OutcomeMonitoringReport:
    """Check if observed fraud rate has drifted from the reference period.

    This ONLY runs when labels are actually available. It does NOT
    fabricate outcome metrics from predictions.
    """
    label_info = check_label_availability(current_labels, min_labels)

    if not label_info.available:
        return OutcomeMonitoringReport(
            timestamp=time.time(),
            label_availability=label_info,
            observed_fraud_rate=0.0,
            reference_fraud_rate=reference_fraud_rate,
            fraud_rate_shift=0.0,
            n_labeled=label_info.n_labeled,
            outcome_metrics={},
            overall_status="insufficient_data",
        )

    observed_rate = label_info.fraud_rate
    fraud_rate_shift = abs(observed_rate - (reference_fraud_rate or 0))

    overall = "stable"
    if fraud_rate_shift >= fraud_rate_crit_shift:
        overall = "critical"
    elif fraud_rate_shift >= fraud_rate_warn_shift:
        overall = "warning"

    return OutcomeMonitoringReport(
        timestamp=time.time(),
        label_availability=label_info,
        observed_fraud_rate=observed_rate,
        reference_fraud_rate=reference_fraud_rate,
        fraud_rate_shift=fraud_rate_shift,
        n_labeled=label_info.n_labeled,
        outcome_metrics={},  # precision/recall only if predictions also available
        overall_status=overall,
    )


def compute_outcome_metrics(
    labels: list[int],
    predictions: list[int],
    scores: list[float] | None = None,
) -> dict[str, float]:
    """Compute outcome metrics when both labels AND predictions are available.

    This is outcome monitoring, not prediction monitoring. It answers:
    "given that we now know the truth, how did our predictions perform?"
    """
    labels_arr = np.array(labels, dtype=int)
    preds_arr = np.array(predictions, dtype=int)

    n = len(labels_arr)
    if n == 0:
        return {}

    tp = int(np.sum((preds_arr == 1) & (labels_arr == 1)))
    fp = int(np.sum((preds_arr == 1) & (labels_arr == 0)))
    tn = int(np.sum((preds_arr == 0) & (labels_arr == 0)))
    fn = int(np.sum((preds_arr == 0) & (labels_arr == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": n,
        "prevalence": float(np.mean(labels_arr)),
    }

    if scores is not None and len(scores) == n:
        scores_arr = np.array(scores, dtype=float)
        fraud_scores = scores_arr[labels_arr == 1]
        legit_scores = scores_arr[labels_arr == 0]
        if len(fraud_scores) > 0:
            metrics["mean_fraud_score"] = float(np.mean(fraud_scores))
        if len(legit_scores) > 0:
            metrics["mean_legit_score"] = float(np.mean(legit_scores))

    return metrics
