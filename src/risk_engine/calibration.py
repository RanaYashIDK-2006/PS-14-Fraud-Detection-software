"""Platt scaling for the fused risk score.

Replaces the isotonic calibrator: a parametric sigmoid fit on the raw
stacker output (1-D logistic regression), so it is smooth and monotonic and
cannot produce the step-function mapping an isotonic fit on near-separable
data does (hard 0.0 floor for most events, overconfident top decile).

Fit by `src/train_compare.py` on OUT-OF-ARCHETYPE cross-fit predictions
(the leave-one-archetype-out folds), so the raw->probability mapping comes
from patterns the fitting model never saw rather than leaky validation.

Pickled into `models/artifacts/calibrator.joblib`; the Risk Engine's
FusionEngine loads it by this class, so the file must stay importable
here.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibration:
    """p(y=1 | raw) = sigmoid(a * raw + b), fit as a 1-D logistic regression.

    Two parameters - cannot overfit the way an isotonic step function can.
    `predict` mirrors the IsotonicRegression interface (1-D array in,
    1-D array out) so FusionEngine is unchanged.
    """

    def __init__(self, C: float = 1e4):
        # High C ~ unregularized: we want the sigmoid through the data cloud
        # (the conditional fraud rate per raw score), not a shrunken fit.
        self.lr = LogisticRegression(C=C, max_iter=5000)

    def fit(self, raw: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "PlattCalibration":
        self.lr.fit(np.asarray(raw, dtype=float).reshape(-1, 1), y,
                    sample_weight=sample_weight)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=float).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]
