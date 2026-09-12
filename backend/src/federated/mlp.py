"""Small multi-layer perceptron for the federated prototype (architecture
section 19).

The counterpart to `src/federated/lr.py`'s `LocalLR`, built with the same
contract so FedAvg can control the loop: explicit initial params (the global
model), explicit epoch count per round, and a params getter/setter so only
weight vectors cross the process boundary. A one-hidden-layer ReLU MLP is
the smallest model that can learn the NONLINEAR boundaries the synthetic
fraud patterns contain (e.g. the AND-of-flags interaction rules like CNP
testing or account takeover), which is exactly the "separability benefit"
the LR-vs-MLP comparison quantifies.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


class LocalMLP:
    """Two-layer ReLU MLP (input -> hidden -> 1 sigmoid output), trained
    with class-balanced batch gradient descent from the CURRENT params for
    `epochs` passes. Params are [W1, b1, W2, b2]; init is small-random
    (a zero init would kill backprop through ReLU)."""

    def __init__(self, n_features: int, hidden: int = 16, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1: np.ndarray = rng.normal(0.0, 1.0 / np.sqrt(n_features), (n_features, hidden))
        self.b1: np.ndarray = np.zeros(hidden)
        self.W2: np.ndarray = rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, 1))
        self.b2: float = 0.0

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: list[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2 = params

    def train(self, X: np.ndarray, y: np.ndarray, *,
              epochs: int, lr: float, pos_weight: float | None = None) -> list[np.ndarray]:
        n = len(y)
        if pos_weight is None:
            pos = int(y.sum())
            neg = n - pos
            pos_weight = max(neg, 1) / max(pos, 1)
        sw = np.where(y == 1, pos_weight, 1.0).astype(float).reshape(-1, 1)
        yc = y.reshape(-1, 1)
        W1, b1, W2, b2 = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2  # copy scalars
        for _ in range(epochs):
            z1 = X @ W1 + b1
            a1 = relu(z1)
            z2 = a1 @ W2 + b2
            p = sigmoid(z2)
            err = (p - yc) * sw                       # dL/dz2 (n,1)
            dW2 = a1.T @ err / n
            db2 = err.sum(axis=0) / n
            da1 = (err @ W2.T) * (z1 > 0)             # ReLU backprop (n,hidden)
            dW1 = X.T @ da1 / n
            db1 = da1.sum(axis=0) / n
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        return self.params()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        p = sigmoid(a1 @ self.W2 + self.b2).ravel()
        return np.column_stack([1.0 - p, p])
