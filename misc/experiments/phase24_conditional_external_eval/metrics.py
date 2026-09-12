"""Metrics computation with confidence intervals.

Computes ROC-AUC, PR-AUC, recall, FPR, precision, and Wilson
confidence intervals for classification metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    auc,
)


def wilson_ci(
    successes: int,
    trials: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Returns (lower, upper) bounds.
    """
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    model_name: str = "model",
) -> dict[str, Any]:
    """Compute full classification metrics at a fixed threshold."""
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n_total = len(y_true)
    n_fraud = int(y_true.sum())
    n_legit = n_total - n_fraud
    prevalence = n_fraud / n_total if n_total > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    alert_rate = (tp + fp) / n_total if n_total > 0 else 0.0

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = None

    # PR-AUC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall_curve, precision_curve)
    except ValueError:
        pr_auc = None

    # Confidence intervals
    recall_ci = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)
    fpr_ci = wilson_ci(fp, fp + tn) if (fp + tn) > 0 else (0.0, 0.0)
    precision_ci = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)

    return {
        "model": model_name,
        "threshold": threshold,
        "n_total": n_total,
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "prevalence": round(prevalence, 6),
        "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 6) if pr_auc is not None else None,
        "recall": round(recall, 6),
        "fpr": round(fpr, 6),
        "precision": round(precision, 6),
        "alert_rate": round(alert_rate, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "alerts_per_1k": round(alert_rate * 1000, 2),
        "alerts_per_10k": round(alert_rate * 10000, 2),
        "recall_ci_lower": round(recall_ci[0], 6),
        "recall_ci_upper": round(recall_ci[1], 6),
        "fpr_ci_lower": round(fpr_ci[0], 6),
        "fpr_ci_upper": round(fpr_ci[1], 6),
        "precision_ci_lower": round(precision_ci[0], 6),
        "precision_ci_upper": round(precision_ci[1], 6),
        "method": "wilson_score_interval",
        "z_score": 1.96,
    }


def compute_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 20,
) -> dict[str, Any]:
    """Compute score distribution for fraud and legit classes."""
    fraud_scores = y_score[y_true == 1]
    legit_scores = y_score[y_true == 0]

    bins = np.linspace(0, 1, n_bins + 1)
    fraud_hist, _ = np.histogram(fraud_scores, bins=bins)
    legit_hist, _ = np.histogram(legit_scores, bins=bins)

    bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)]

    return {
        "bin_labels": bin_labels,
        "fraud_counts": fraud_hist.tolist(),
        "legit_counts": legit_hist.tolist(),
        "fraud_mean": round(float(fraud_scores.mean()), 4) if len(fraud_scores) > 0 else None,
        "fraud_median": round(float(np.median(fraud_scores)), 4) if len(fraud_scores) > 0 else None,
        "legit_mean": round(float(legit_scores.mean()), 4) if len(legit_scores) > 0 else None,
        "legit_median": round(float(np.median(legit_scores)), 4) if len(legit_scores) > 0 else None,
    }


def write_metrics_artifacts(
    output_dir: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    model_name: str,
) -> dict[str, Any]:
    """Compute and write all metrics artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_classification_metrics(y_true, y_score, threshold, model_name)
    score_dist = compute_score_distribution(y_true, y_score)

    prefix = model_name.lower().replace(" ", "_")
    (output_dir / f"05_{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / f"06_{prefix}_score_distribution.json").write_text(
        json.dumps(score_dist, indent=2, default=str), encoding="utf-8"
    )

    return {"metrics": metrics, "score_distribution": score_dist}
