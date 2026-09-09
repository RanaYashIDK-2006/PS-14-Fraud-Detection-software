"""Model evaluation — frozen model scoring and metric computation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report,
)


@dataclass
class EvaluationMetrics:
    """Primary metrics for a model evaluation."""
    n_total: int = 0
    n_fraud: int = 0
    n_legit: int = 0
    prevalence: float = 0.0

    roc_auc: float = 0.0
    pr_auc: float = 0.0
    recall: float = 0.0
    fpr: float = 0.0
    precision: float = 0.0
    alert_rate: float = 0.0

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    alerts_per_1k: float = 0.0
    alerts_per_10k: float = 0.0
    alerts_per_100k: float = 0.0
    alerts_per_1M: float = 0.0

    threshold: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_predictions(
        cls,
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float,
    ) -> "EvaluationMetrics":
        """Compute metrics from true labels and predicted scores."""
        y_pred = (y_score >= threshold).astype(int)

        n_total = len(y_true)
        n_fraud = int(y_true.sum())
        n_legit = n_total - n_fraud
        prevalence = n_fraud / n_total if n_total > 0 else 0.0

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        alert_rate = (tp + fp) / n_total if n_total > 0 else 0.0

        # AUC metrics (need at least both classes present)
        try:
            roc_auc = roc_auc_score(y_true, y_score) if n_fraud > 0 and n_legit > 0 else 0.0
        except ValueError:
            roc_auc = 0.0

        try:
            pr_auc = average_precision_score(y_true, y_score) if n_fraud > 0 else 0.0
        except ValueError:
            pr_auc = 0.0

        return cls(
            n_total=n_total,
            n_fraud=n_fraud,
            n_legit=n_legit,
            prevalence=prevalence,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            recall=recall,
            fpr=fpr,
            precision=precision,
            alert_rate=alert_rate,
            tp=tp, fp=fp, fn=fn, tn=tn,
            alerts_per_1k=alert_rate * 1000,
            alerts_per_10k=alert_rate * 10000,
            alerts_per_100k=alert_rate * 100000,
            alerts_per_1M=alert_rate * 1000000,
            threshold=threshold,
        )


@dataclass
class UncertaintyEstimate:
    """Confidence intervals for key metrics."""
    method: str = "wilson_score"
    confidence_level: float = 0.95

    recall_ci_lower: float = 0.0
    recall_ci_upper: float = 0.0
    fpr_ci_lower: float = 0.0
    fpr_ci_upper: float = 0.0
    precision_ci_lower: float = 0.0
    precision_ci_upper: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def wilson_score_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    from scipy import stats
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0, center - spread), min(1, center + spread))


def compute_uncertainty(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    confidence: float = 0.95,
) -> UncertaintyEstimate:
    """Compute confidence intervals using Wilson score method."""
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n_pos = tp + fn
    n_neg = fp + tn

    recall_ci = wilson_score_ci(tp, n_pos, confidence) if n_pos > 0 else (0, 0)
    fpr_ci = wilson_score_ci(fp, n_neg, confidence) if n_neg > 0 else (0, 0)
    prec_ci = wilson_score_ci(tp, tp + fp, confidence) if (tp + fp) > 0 else (0, 0)

    return UncertaintyEstimate(
        confidence_level=confidence,
        recall_ci_lower=recall_ci[0],
        recall_ci_upper=recall_ci[1],
        fpr_ci_lower=fpr_ci[0],
        fpr_ci_upper=fpr_ci[1],
        precision_ci_lower=prec_ci[0],
        precision_ci_upper=prec_ci[1],
    )


@dataclass
class TemporalWindowResult:
    """Metrics for a single chronological window."""
    window_name: str
    start: str
    end: str
    n_total: int
    n_fraud: int
    prevalence: float
    recall: float
    fpr: float
    precision: float
    alert_rate: float
    pr_auc: float

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def evaluate_temporal_windows(
    df: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    timestamp_col: str,
    threshold: float,
    n_windows: int = 4,
) -> list[TemporalWindowResult]:
    """Evaluate metrics across chronological windows."""
    df_sorted = df.sort_values(timestamp_col).copy()
    df_sorted["_score"] = scores
    df_sorted["_label"] = labels

    windows = np.array_split(df_sorted.index, n_windows)
    results = []

    for i, idx in enumerate(windows):
        window_df = df_sorted.loc[idx]
        y_true = window_df["_label"].values
        y_score = window_df["_score"].values

        metrics = EvaluationMetrics.from_predictions(y_true, y_score, threshold)

        ts_values = pd.to_datetime(window_df[timestamp_col], errors="coerce")
        start = str(ts_values.min()) if not ts_values.isna().all() else "N/A"
        end = str(ts_values.max()) if not ts_values.isna().all() else "N/A"

        results.append(TemporalWindowResult(
            window_name=f"window_{i+1}",
            start=start,
            end=end,
            n_total=metrics.n_total,
            n_fraud=metrics.n_fraud,
            prevalence=metrics.prevalence,
            recall=metrics.recall,
            fpr=metrics.fpr,
            precision=metrics.precision,
            alert_rate=metrics.alert_rate,
            pr_auc=metrics.pr_auc,
        ))

    return results


def evaluate_by_channel(
    df: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    channel_col: str,
    threshold: float,
) -> list[dict]:
    """Evaluate metrics by channel."""
    df_eval = df.copy()
    df_eval["_score"] = scores
    df_eval["_label"] = labels

    results = []
    for channel, group in df_eval.groupby(channel_col):
        y_true = group["_label"].values
        y_score = group["_score"].values
        metrics = EvaluationMetrics.from_predictions(y_true, y_score, threshold)

        insufficient = metrics.n_fraud < 30
        results.append({
            "channel": channel,
            "n_total": metrics.n_total,
            "n_fraud": metrics.n_fraud,
            "recall": metrics.recall,
            "fpr": metrics.fpr,
            "precision": metrics.precision,
            "alert_rate": metrics.alert_rate,
            "statistical_power": "INSUFFICIENT" if insufficient else "SUFFICIENT",
        })

    return results
