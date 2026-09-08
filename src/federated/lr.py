"""PS-14 federated package (architecture section 19).

Federated averaging (FedAvg) prototype: institutions train locally on their
own transaction data; only model weight vectors cross the boundary to a
coordinator, which averages them weighted by local data size. No raw data,
features, or labels ever leave an institution.

This module provides the pieces: a minimal balanced logistic regression
trainable with explicit epochs from a given weight init (so FedAvg can
control the loop), the FedAvg averaging rule, and evaluation metrics.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class LocalLR:
    """Logistic regression trained with class-balanced gradient descent.

    The point of hand-rolling it (instead of sklearn) is FedAvg: we need
    explicit control over (a) the initial weights (the global model) and
    (b) the number of local epochs per round.
    """

    def __init__(self, n_features: int, seed: int = 0) -> None:
        self.w: np.ndarray = np.zeros(n_features)
        self.b: float = 0.0
        self._seed = seed

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        lr: float,
        pos_weight: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Balanced gradient descent from the CURRENT weights for `epochs`
        passes. Returns the updated (w, b)."""
        n = len(y)
        if pos_weight is None:
            pos = int(y.sum())
            neg = n - pos
            pos_weight = max(neg, 1) / max(pos, 1)
        sw = np.where(y == 1, pos_weight, 1.0).astype(float)
        w, b = self.w.copy(), self.b
        for _ in range(epochs):
            p = sigmoid(X @ w + b)
            err = (p - y) * sw
            w -= lr * (X.T @ err) / n
            b -= lr * err.sum() / n
        self.w, self.b = w, b
        return w, b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = sigmoid(X @ self.w + self.b)
        return np.column_stack([1.0 - p, p])


def fedavg(updates: list[tuple[np.ndarray, float, int]]) -> tuple[np.ndarray, float]:
    """Weighted average of local (w, b, n_train) — the FedAvg aggregation."""
    total = sum(n for _, _, n in updates)
    w = sum(u * n for u, _, n in updates) / total
    b = sum(b_ * n for _, b_, n in updates) / total
    return w, b


def fedavg_params(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """Weighted index-wise average of per-layer parameter LISTS (the MLP
    FedAvg aggregation) — each layer is averaged across clients weighted by
    local data size, exactly like FedAvg does for a single weight vector."""
    total = sum(n for _, n in updates)
    n_params = len(updates[0][0])
    out: list[np.ndarray] = []
    for i in range(n_params):
        acc = sum(p[i] * n for p, n in updates) / total
        out.append(np.asarray(acc).astype(float))
    return out


def eval_metrics(y_true: np.ndarray, p1: np.ndarray) -> dict:
    """PR-AUC / ROC-AUC (primary, threshold-free) + P/R/F1 at 0.5.

    A test split with NO positives (or no negatives) has no meaningful
    ranking AUC: pr_auc / roc_auc are NaN there instead of a misleading
    0.0, and the heterogeneity sweep filters those institutions out of its
    macro (their test data simply cannot measure the model)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    pr_auc = float(average_precision_score(y_true, p1)) if n_pos > 0 else float("nan")
    roc_auc = float(roc_auc_score(y_true, p1)) if n_pos > 0 and n_neg > 0 else float("nan")
    preds = (p1 >= 0.5).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
