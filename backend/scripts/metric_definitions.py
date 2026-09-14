"""Phase 39: Authoritative metric definitions and single-source calculators.

One implementation per metric — all evaluation scripts and reports must use
these functions rather than recomputing metrics inline. Semantics are pinned
in docs/metric_definitions.md (human-readable) and here in code (authoritative,
``METRIC_DEFINITIONS_VERSION = "1.0"``).

Definitions (all binary classification, positive class = fraud = 1):

* ROC-AUC  — probability a uniformly chosen positive is ranked above a
  uniformly chosen negative; ties count 0.5. Implementation: sklearn
  ``roc_auc_score`` (trapezoidal over the empirical ROC curve).
* PR-AUC   — area under the precision-recall curve, computed by
  ``average_precision_score`` (step-preserving, not trapezoidal interpolation).
  Expected value for a random classifier = prevalence.
* Recall   — TP / (TP + FN) at a stated threshold; 0.0 when no positives.
* Precision— TP / (TP + FP) at a stated threshold; 0.0 when no predicted
  positives.
* FPR      — FP / (FP + TN); 0.0 when no negatives.
* Recall@1%FPR — recall at the threshold that achieves the target FPR
  (default 1%) **on the validation split** (``threshold_at_fpr``). The
  threshold is NEVER selected using test labels; it is applied unchanged to
  the test split. Ties in the score grid are broken toward the lower
  threshold (higher recall) among those within tolerance of the target FPR.
* Confusion matrix — counts at the stated threshold, ``score >= threshold``
  predicted positive.
* Prevalence — positives / total in the evaluated split.

Threshold semantics: positive prediction is ``score >= threshold`` everywhere
in this repository's calculators. Missing/NaN predictions are invalid input —
callers must drop or impute them explicitly before calling; the calculators
raise on NaN rather than silently skewing results.

Minimum sample conditions: CI functions emit a warning (and return
``ci_low = ci_high = None``) when a metric's conditioning cell is too small
(see ``MIN_CI_CELLS``) instead of producing spuriously tight intervals.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

METRIC_DEFINITIONS_VERSION = "1.0"
METRIC_DEFINITIONS_DOC = "docs/metric_definitions.md"

# Below this many observations in the conditioning cell (e.g. n_positives for
# recall, n_negatives for FPR), confidence intervals are withheld as
# statistically unstable.
MIN_CI_CELLS = 30
# Bootstrap settings: resampling unit is the individual observation
# (stratified by class), percentile method, documented seed.
BOOTSTRAP_DEFAULT_SAMPLES = 1000
BOOTSTRAP_DEFAULT_LEVEL = 0.95


@dataclass
class OperatingPoint:
    """Explicit operating-point report (threshold + confusion counts)."""
    threshold: float
    threshold_source: str          # "validation" | "fixed" | "none"
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    @property
    def n_alerts(self) -> int:
        return self.tp + self.fp

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "fpr": round(self.fpr, 4),
            "n_alerts": self.n_alerts,
        }


def confusion_counts(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """(tp, fp, tn, fn) at ``score >= threshold``. Raises on NaN input."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if np.isnan(scores).any() or np.isnan(y_true).any():
        raise ValueError("NaN in labels or scores — drop or impute explicitly before calling")
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    return tp, fp, tn, fn


def operating_point(y_true: np.ndarray, scores: np.ndarray, threshold: float,
                    threshold_source: str = "fixed") -> OperatingPoint:
    tp, fp, tn, fn = confusion_counts(y_true, scores, threshold)
    return OperatingPoint(threshold=float(threshold), threshold_source=threshold_source,
                          tp=tp, fp=fp, tn=tn, fn=fn)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC (ties = 0.5 credit via sklearn trapezoid)."""
    y_true, scores = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC via average_precision_score (random baseline = prevalence)."""
    y_true, scores = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def threshold_at_fpr_on_validation(y_val: np.ndarray, scores_val: np.ndarray,
                                   target_fpr: float = 0.01) -> float:
    """Select the threshold achieving ``target_fpr`` on VALIDATION labels only.

    Authoritative Recall@1%FPR discipline: the returned threshold is applied
    unchanged to the test split. Selecting it against test labels and then
    reporting the same split's recall as unbiased performance is test-set
    threshold tuning and is forbidden; ``assert_not_test_tuned`` in the
    record module records ``threshold_source`` so reviews can verify this.
    """
    y_val = np.asarray(y_val, dtype=int)
    scores_val = np.asarray(scores_val, dtype=float)
    grid = np.unique(np.quantile(scores_val, np.linspace(0.0, 1.0, 2001)))
    best_t, best_err = grid[0], np.inf
    for t in grid:
        _, fp, tn, _ = confusion_counts(y_val, scores_val, t)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        err = abs(fpr - target_fpr)
        if err < best_err:
            best_err, best_t = err, t
    return float(best_t)


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Recall at a PRE-SELECTED threshold (chosen on validation, not here)."""
    tp, _, _, fn = confusion_counts(y_true, scores, threshold)
    return tp / (tp + fn) if (tp + fn) else 0.0


def prevalence_report(y_true: np.ndarray) -> dict[str, Any]:
    """Class-imbalance block required in every evaluation report.

    Prevalence context matters because with heavy imbalance accuracy is
    dominated by the negative class (a never-fraud classifier scores
    1 - prevalence) and PR-AUC baselines equal prevalence — so raw accuracy
    must never be the headline metric.
    """
    y_true = np.asarray(y_true, dtype=int)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    total = n_pos + n_neg
    return {
        "total": total,
        "positive_count": n_pos,
        "negative_count": n_neg,
        "prevalence": n_pos / total if total else float("nan"),
        "note": ("Accuracy is reported (if at all) as a SECONDARY metric: with fraud "
                 "prevalence p, a trivial always-legit classifier achieves accuracy "
                 "1-p, so accuracy is dominated by the majority class and must not "
                 "be read as detection effectiveness."),
    }


@dataclass
class ConfidenceInterval:
    metric: str
    point_estimate: float
    ci_low: float | None
    ci_high: float | None
    level: float
    n_bootstrap: int
    seed: int
    resampling_unit: str = "observation (stratified by class)"
    method: str = "percentile bootstrap"
    withheld_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "point_estimate": round(self.point_estimate, 4) if np.isfinite(self.point_estimate) else None,
            "ci_low": round(self.ci_low, 4) if self.ci_low is not None else None,
            "ci_high": round(self.ci_high, 4) if self.ci_high is not None else None,
            "level": self.level,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "resampling_unit": self.resampling_unit,
            "method": self.method,
            "withheld_reason": self.withheld_reason,
        }


def bootstrap_ci(metric_fn, y_true: np.ndarray, scores: np.ndarray, *,
                 n_bootstrap: int = BOOTSTRAP_DEFAULT_SAMPLES,
                 level: float = BOOTSTRAP_DEFAULT_LEVEL,
                 seed: int = 42,
                 min_cell: int = MIN_CI_CELLS) -> ConfidenceInterval:
    """Percentile bootstrap CI for any metric_fn(y, s) -> float.

    Resampling unit: individual observations, stratified by class (each
    bootstrap sample keeps the original positive/negative counts so
    prevalence-dependent metrics stay interpretable). Intervals are withheld
    (``ci_low/ci_high = None`` + reason) when a conditioning cell is below
    ``min_cell`` — no fabricated precision on tiny samples.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    point = float(metric_fn(y_true, scores))
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    limiting = min(n_pos, n_neg)
    if limiting < min_cell or not np.isfinite(point):
        reason = (f"conditioning cell too small (n_pos={n_pos}, n_neg={n_neg}; "
                  f"minimum {min_cell} required)") if limiting < min_cell else \
                 "metric undefined on this data"
        return ConfidenceInterval(metric=metric_fn.__name__ if hasattr(metric_fn, "__name__") else "metric",
                                  point_estimate=point, ci_low=None, ci_high=None,
                                  level=level, n_bootstrap=n_bootstrap, seed=seed,
                                  withheld_reason=reason)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        take = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        vals[b] = metric_fn(y_true[take], scores[take])
    vals = vals[np.isfinite(vals)]
    if len(vals) < n_bootstrap * 0.5:
        return ConfidenceInterval(metric=metric_fn.__name__ if hasattr(metric_fn, "__name__") else "metric",
                                  point_estimate=point, ci_low=None, ci_high=None,
                                  level=level, n_bootstrap=n_bootstrap, seed=seed,
                                  withheld_reason="majority of bootstrap replicates undefined")
    alpha = (1.0 - level) / 2.0
    name = metric_fn.__name__ if hasattr(metric_fn, "__name__") else "metric"
    return ConfidenceInterval(metric=name, point_estimate=point,
                              ci_low=float(np.quantile(vals, alpha)),
                              ci_high=float(np.quantile(vals, 1 - alpha)),
                              level=level, n_bootstrap=n_bootstrap, seed=seed)


def majority_baseline(y_true: np.ndarray) -> dict[str, Any]:
    """Baseline: always predict the majority class (legit)."""
    y_true = np.asarray(y_true, dtype=int)
    n_pos = int((y_true == 1).sum())
    total = len(y_true)
    return {
        "baseline": "majority_class",
        "description": "always predicts the majority class (legitimate)",
        "recall": 0.0,
        "precision": 0.0 if n_pos else float("nan"),
        "fpr": 0.0,
        "f1": 0.0 if n_pos else float("nan"),
        "accuracy": (total - n_pos) / total if total else float("nan"),
        "roc_auc": 0.5,
        "pr_auc": n_pos / total if total else float("nan"),
    }


def random_baseline(y_true: np.ndarray, seed: int = 42) -> dict[str, Any]:
    """Baseline: uniform random scores — ROC-AUC 0.5, PR-AUC ≈ prevalence."""
    y_true = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(seed)
    scores = rng.random(len(y_true))
    n_pos = int((y_true == 1).sum())
    prev = n_pos / len(y_true) if len(y_true) else float("nan")
    tp, fp, tn, fn = confusion_counts(y_true, scores, 0.5)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "baseline": "random_scores",
        "description": "uniform random scores in [0,1); threshold 0.5",
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "fpr": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        "accuracy": round(float((y_true == (scores >= 0.5).astype(int)).mean()), 4) if len(y_true) else float("nan"),
        "roc_auc": round(roc_auc(y_true, scores), 4) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": round(pr_auc(y_true, scores), 4) if n_pos else float("nan"),
        "expected_roc_auc": 0.5,
        "expected_pr_auc": round(prev, 6) if np.isfinite(prev) else None,
    }


def compare_to_baseline(model_metrics: dict[str, float], baseline: dict[str, Any]) -> dict[str, Any]:
    """Fair same-split comparison; states improvement without claiming
    significance (significance would require the documented statistical test
    and is NOT claimed automatically)."""
    return {
        "model_vs_baseline": {
            "roc_auc_model": model_metrics.get("roc_auc"),
            "roc_auc_baseline": baseline.get("roc_auc"),
            "pr_auc_model": model_metrics.get("pr_auc"),
            "pr_auc_baseline": baseline.get("pr_auc"),
            "note": ("Both evaluated on the same eligible split with the same "
                     "metric definitions. No statistical-significance claim is "
                     "made by this comparison alone."),
        }
    }


def small_sample_warning(y_true: np.ndarray, min_per_class: int = 50) -> list[str]:
    """Warnings that must accompany reports on small/imbalanced data."""
    y_true = np.asarray(y_true, dtype=int)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    out: list[str] = []
    if n_pos < min_per_class:
        out.append(f"SMALL_POSITIVE_CLASS: only {n_pos} positive cases — point "
                   f"estimates of recall/PR-AUC are statistically unstable; "
                   f"confidence intervals may be withheld.")
    if n_neg < min_per_class:
        out.append(f"SMALL_NEGATIVE_CLASS: only {n_neg} negative cases — FPR and "
                   f"ROC-AUC estimates are unstable.")
    if n_pos and n_neg and (n_pos / (n_pos + n_neg)) < 0.01:
        out.append("HEAVY_IMBALANCE: prevalence below 1% — accuracy is "
                   "meaningless as a headline metric; rely on PR-AUC and "
                   "recall at a stated FPR operating point.")
    return out


@dataclass
class FullEvaluationMetrics:
    """The standardized metrics block every evaluation report must carry."""
    y_true: np.ndarray
    scores: np.ndarray
    threshold: float
    threshold_source: str = "validation"
    bootstrap: bool = False
    seed: int = 42
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        op = operating_point(self.y_true, self.scores, self.threshold, self.threshold_source)
        out: dict[str, Any] = {
            "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
            "ranking_metrics": {
                "roc_auc": roc_auc(self.y_true, self.scores),
                "pr_auc": pr_auc(self.y_true, self.scores),
            },
            "operating_point": op.to_dict(),
            "class_imbalance": prevalence_report(self.y_true),
            "warnings": self.warnings + small_sample_warning(self.y_true),
        }
        if self.bootstrap:
            out["confidence_intervals"] = {
                "roc_auc": bootstrap_ci(roc_auc, self.y_true, self.scores, seed=self.seed).to_dict(),
                "pr_auc": bootstrap_ci(pr_auc, self.y_true, self.scores, seed=self.seed).to_dict(),
                "recall_at_threshold": bootstrap_ci(
                    lambda y, s: recall_at_fpr(y, s, self.threshold),
                    self.y_true, self.scores, seed=self.seed).to_dict(),
            }
        out["baselines"] = {
            "majority_class": majority_baseline(self.y_true),
            "random_scores": random_baseline(self.y_true, seed=self.seed),
        }
        return out
