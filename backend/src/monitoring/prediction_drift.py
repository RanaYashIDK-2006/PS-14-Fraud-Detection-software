"""Prediction / risk-score drift monitoring.

Tracks changes in:
  - risk-score distribution (mean, median, quantiles)
  - decision-band proportions (allow / step-up / verify rates)
  - score histogram shift

This is DISTINCT from feature drift. Prediction drift indicates the model's
output behavior is changing, which may or may not be caused by input drift.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Decision bands (matching risk_engine thresholds)
BAND_ALLOW = "allow"        # score < 0.30
BAND_STEP_UP = "step_up"    # 0.30 <= score < 0.60
BAND_VERIFY = "verify"      # score >= 0.60

DEFAULT_BAND_ALLOW_THRESHOLD = 0.30
DEFAULT_BAND_VERIFY_THRESHOLD = 0.60


@dataclass
class PredictionDriftReport:
    """Report of prediction-distribution drift for one window."""
    timestamp: float
    n_scores: int
    mean_score: float
    median_score: float
    p10_score: float
    p90_score: float
    std_score: float
    band_distribution: dict[str, float]  # band -> proportion
    reference_band_distribution: dict[str, float] | None
    band_shift: dict[str, float]  # band -> absolute shift
    max_band_shift: float
    mean_score_shift: float  # difference from reference mean
    overall_drift: str  # "stable", "warning", "critical"
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "n_scores": self.n_scores,
            "mean_score": round(self.mean_score, 6),
            "median_score": round(self.median_score, 6),
            "p10_score": round(self.p10_score, 6),
            "p90_score": round(self.p90_score, 6),
            "std_score": round(self.std_score, 6),
            "band_distribution": {k: round(v, 4) for k, v in self.band_distribution.items()},
            "band_shift": {k: round(v, 4) for k, v in self.band_shift.items()},
            "max_band_shift": round(self.max_band_shift, 4),
            "mean_score_shift": round(self.mean_score_shift, 6),
            "overall_drift": self.overall_drift,
        }


def compute_score_distribution(
    scores: np.ndarray,
) -> dict[str, float]:
    """Compute summary statistics for a score array."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return {"mean": 0, "median": 0, "p10": 0, "p90": 0, "std": 0, "n": 0}
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "p10": float(np.percentile(scores, 10)),
        "p90": float(np.percentile(scores, 90)),
        "std": float(np.std(scores)),
        "n": int(len(scores)),
    }


def compute_band_distribution(
    scores: np.ndarray,
    allow_threshold: float = DEFAULT_BAND_ALLOW_THRESHOLD,
    verify_threshold: float = DEFAULT_BAND_VERIFY_THRESHOLD,
) -> dict[str, float]:
    """Compute proportion of scores in each decision band."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return {BAND_ALLOW: 0.0, BAND_STEP_UP: 0.0, BAND_VERIFY: 0.0}
    allow = float(np.mean(scores < allow_threshold))
    verify = float(np.mean(scores >= verify_threshold))
    step_up = 1.0 - allow - verify
    return {BAND_ALLOW: allow, BAND_STEP_UP: step_up, BAND_VERIFY: verify}


def check_prediction_drift(
    current_scores: np.ndarray,
    reference_scores: np.ndarray | None = None,
    reference_band_distribution: dict[str, float] | None = None,
    band_warn_shift: float = 0.10,
    band_crit_shift: float = 0.25,
    mean_warn_shift: float = 0.05,
    mean_crit_shift: float = 0.15,
    min_scores: int = 30,
) -> PredictionDriftReport:
    """Check if prediction-score distribution has drifted.

    Args:
        current_scores: scores from the current observation window
        reference_scores: scores from the training/reference period (optional)
        reference_band_distribution: pre-computed reference band proportions (optional)
        band_warn_shift: absolute shift in any band proportion that triggers WARNING
        band_crit_shift: absolute shift in any band proportion that triggers CRITICAL
        mean_warn_shift: absolute shift in mean score that triggers WARNING
        mean_crit_shift: absolute shift in mean score that triggers CRITICAL
        min_scores: minimum scores for a meaningful check
    """
    current_scores = np.asarray(current_scores, dtype=float)
    current_scores = current_scores[np.isfinite(current_scores)]
    n = len(current_scores)

    # Default empty report
    empty_stats = {"mean": 0, "median": 0, "p10": 0, "p90": 0, "std": 0, "n": 0}
    empty_bands = {BAND_ALLOW: 0.0, BAND_STEP_UP: 0.0, BAND_VERIFY: 0.0}

    if n < min_scores:
        return PredictionDriftReport(
            timestamp=time.time(),
            n_scores=n,
            mean_score=0, median_score=0, p10_score=0, p90_score=0, std_score=0,
            band_distribution=empty_bands,
            reference_band_distribution=reference_band_distribution,
            band_shift={},
            max_band_shift=0.0,
            mean_score_shift=0.0,
            overall_drift="insufficient_data",
            detail=f"only {n} scores (need >= {min_scores})",
        )

    # Current distribution
    curr_stats = compute_score_distribution(current_scores)
    curr_bands = compute_band_distribution(current_scores)

    # Reference distribution
    if reference_scores is not None and len(reference_scores) >= min_scores:
        ref_stats = compute_score_distribution(reference_scores)
        ref_bands = compute_band_distribution(reference_scores)
        ref_band_dist = ref_bands
    elif reference_band_distribution is not None:
        ref_stats = None
        ref_band_dist = reference_band_distribution
    else:
        ref_stats = None
        ref_band_dist = None

    # Compute shifts
    band_shift: dict[str, float] = {}
    max_band_shift = 0.0
    mean_shift = 0.0

    if ref_band_dist is not None:
        for band in [BAND_ALLOW, BAND_STEP_UP, BAND_VERIFY]:
            shift = abs(curr_bands.get(band, 0) - ref_band_dist.get(band, 0))
            band_shift[band] = shift
            max_band_shift = max(max_band_shift, shift)

    if ref_stats is not None:
        mean_shift = abs(curr_stats["mean"] - ref_stats["mean"])

    # Determine overall drift
    overall = "stable"
    detail_parts = []

    if max_band_shift >= band_crit_shift:
        overall = "critical"
        worst_band = max(band_shift, key=band_shift.get)
        detail_parts.append(
            f"band {worst_band} shifted {max_band_shift:.3f} (crit >= {band_crit_shift})"
        )
    elif max_band_shift >= band_warn_shift:
        overall = "warning"
        worst_band = max(band_shift, key=band_shift.get)
        detail_parts.append(
            f"band {worst_band} shifted {max_band_shift:.3f} (warn >= {band_warn_shift})"
        )

    if mean_shift >= mean_crit_shift:
        overall = "critical"
        detail_parts.append(
            f"mean shifted {mean_shift:.4f} (crit >= {mean_crit_shift})"
        )
    elif mean_shift >= mean_warn_shift and overall == "stable":
        overall = "warning"
        detail_parts.append(
            f"mean shifted {mean_shift:.4f} (warn >= {mean_warn_shift})"
        )

    return PredictionDriftReport(
        timestamp=time.time(),
        n_scores=n,
        mean_score=curr_stats["mean"],
        median_score=curr_stats["median"],
        p10_score=curr_stats["p10"],
        p90_score=curr_stats["p90"],
        std_score=curr_stats["std"],
        band_distribution=curr_bands,
        reference_band_distribution=ref_band_dist,
        band_shift=band_shift,
        max_band_shift=max_band_shift,
        mean_score_shift=mean_shift,
        overall_drift=overall,
        detail="; ".join(detail_parts) if detail_parts else "no significant shift",
    )
