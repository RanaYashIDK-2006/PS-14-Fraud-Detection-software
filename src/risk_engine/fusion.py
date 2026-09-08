"""Model fusion (architecture section 5: "weighted ensemble or stacked
meta-learner").

Supports two modes:
1. Stacked mode (v2): LR + RF + XGB + IsolationForest → stacker → calibrator
2. Weighted mode (v3): LR + RF + XGB → weighted average → calibrator

Weighted mode is used when stacker.joblib / iso_train_scores.joblib are
absent (newer training runs). It uses tuned weights based on cross-validation
results: RF is primary (best generalization), XGB secondary, LR tertiary.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.privacy_layer.features import ML_FEATURES
from src.risk_engine.calibration import PlattCalibration  # noqa: F401 - resolves the pickled calibrator class

# v2 stacked mode models
MODEL_NAMES_STACKED = ("logistic_regression", "random_forest", "xgboost", "isolation_forest")
# v3 weighted mode models
MODEL_NAMES_WEIGHTED = ("logistic_regression", "random_forest", "xgboost")
# Tuned weights: RF primary (best test generalization 0.970), XGB secondary, LR tertiary
WEIGHTS = {"random_forest": 0.50, "xgboost": 0.30, "logistic_regression": 0.20}

# v5 finance-fraud model: 27 features (21 base + 6 subscription/micro-fraud detection)
FINANCE_FEATURES = ML_FEATURES + [
    "subscription_pattern_score",  # Regularity of charges to same merchant
    "micro_fraud_flag",            # Small amount + high balance (death by a thousand cuts)
    "balance_drain_ratio",         # Amount / account balance
    "merchant_fraud_concentration", # How suspicious this merchant category is
    "card_velocity_ratio",         # Current speed vs historical speed
    "amount_cluster_distance",     # Distance from typical amount clusters
]
# Finance detection thresholds
FINANCE_BOOST_THRESHOLD = 0.10  # If finance model scores above this, boost main score
FINANCE_BOOST_WEIGHT = 0.30    # How much to boost the main score


class FusionEngine:
    def __init__(self, artifacts_dir: Path):
        # v4 model uses an imputer for NaN handling
        imputer_path = artifacts_dir / "imputer.joblib"
        self.imputer = joblib.load(imputer_path) if imputer_path.exists() else None
        self.scaler = joblib.load(artifacts_dir / "scaler.joblib")

        # v5 finance-fraud model: specialized micro-fraud / subscription detection (27 features)
        finance_model_path = artifacts_dir / "xgboost_finance.joblib"
        if finance_model_path.exists():
            self.finance_model = joblib.load(finance_model_path)
            self.finance_scaler = joblib.load(artifacts_dir / "scaler_finance.joblib")
            self.finance_imputer = joblib.load(artifacts_dir / "imputer_finance.joblib")
            self._has_finance_model = True
        else:
            self.finance_model = None
            self.finance_scaler = None
            self.finance_imputer = None
            self._has_finance_model = False

        # Detect mode: stacked (v2) vs weighted (v3)
        stacker_path = artifacts_dir / "stacker.joblib"
        iso_path = artifacts_dir / "iso_train_scores.joblib"
        self._stacked_mode = stacker_path.exists() and iso_path.exists()

        if self._stacked_mode:
            # v2: full stack (LR + RF + XGB + ISO → stacker)
            self.models = {name: joblib.load(artifacts_dir / f"{name}.joblib")
                           for name in MODEL_NAMES_STACKED}
            self.stacker = joblib.load(stacker_path)
            self.iso_train = joblib.load(iso_path)
            self._iso_train_sorted = np.sort(self.iso_train)
        else:
            # v3: weighted ensemble (LR + RF + XGB → weighted avg)
            self.models = {name: joblib.load(artifacts_dir / f"{name}.joblib")
                           for name in MODEL_NAMES_WEIGHTED}
            self.stacker = None
            self.iso_train = None
            self._iso_train_sorted = None

        cal_path = artifacts_dir / "calibrator.joblib"
        self.calibrator = joblib.load(cal_path) if cal_path.exists() else None

        # Legacy fallback models (ignored in v3 weighted mode)
        combined_xgb_path = artifacts_dir / "xgb_combined.joblib"
        combined_scaler_path = artifacts_dir / "scaler_combined.joblib"
        self.combined_xgb = joblib.load(combined_xgb_path) if combined_xgb_path.exists() else None
        self.combined_scaler = joblib.load(combined_scaler_path) if combined_scaler_path.exists() else None
        improved_xgb_path = artifacts_dir / "xgb_improved.joblib"
        improved_scaler_path = artifacts_dir / "scaler_improved.joblib"
        self.improved_xgb = joblib.load(improved_xgb_path) if improved_xgb_path.exists() else None
        self.improved_scaler = joblib.load(improved_scaler_path) if improved_scaler_path.exists() else None

    def predict(self, features: dict) -> tuple[float, dict]:
        """CALIBRATED fused fraud probability in [0, 1] for one feature vector.

        Returns (probability, uncertainty_info) where uncertainty_info contains:
        - model_variance: variance of the base learner outputs
        - model_disagreement: max - min spread
        - individual_outputs: dict of model name -> probability
        """
        comp = self.components(features)
        outputs = comp["outputs"]
        vals = list(outputs.values())
        uncertainty = {
            "model_variance": round(float(np.var(vals)), 6),
            "model_disagreement": round(float(max(vals) - min(vals)), 6),
            "individual_outputs": {k: round(v, 4) for k, v in outputs.items()},
        }
        if self._stacked_mode:
            p_lr = outputs["logistic_regression"]
            p_rf = outputs["random_forest"]
            p_xgb = outputs["xgboost"]
            iso_pct = outputs["isolation_forest"]
            X_stack = np.array([[p_lr, p_rf, p_xgb, iso_pct]])
            raw = float(self.stacker.predict_proba(X_stack)[0, 1])
        else:
            raw = sum(WEIGHTS.get(k, 0) * v for k, v in outputs.items())
        if self.calibrator is None:
            return float(np.clip(raw, 0.0, 1.0)), uncertainty
        return float(np.clip(self.calibrator.predict(np.array([raw]))[0], 0.0, 1.0)), uncertainty

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        """Impute NaN/Inf then scale."""
        if self.imputer is not None:
            X = self.imputer.transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        return self.scaler.transform(X)

    def components(self, features: dict) -> dict:
        """Per-model outputs + weights for one event.

        Used by the internal SHAP-style attribution endpoint."""
        X = np.array([[features[f] for f in ML_FEATURES]], dtype=float)
        Xs = self._prepare(X)
        outs = {
            "logistic_regression": float(self.models["logistic_regression"].predict_proba(Xs)[0, 1]),
            "random_forest": float(self.models["random_forest"].predict_proba(Xs)[0, 1]),
            "xgboost": float(self.models["xgboost"].predict_proba(Xs)[0, 1]),
        }
        if self._stacked_mode:
            outs["isolation_forest"] = float((self.iso_train < -self.models["isolation_forest"].decision_function(Xs)[0]).mean())
            coefs = {n: float(c) for n, c in zip(MODEL_NAMES_STACKED, self.stacker.coef_[0])}
            intercept = float(self.stacker.intercept_[0])
        else:
            coefs = dict(WEIGHTS)
            intercept = 0.0
        return {
            "outputs": outs,
            "coefficients": coefs,
            "intercept": intercept,
            "X": X,  # cached feature array for reuse by predict_combined
        }

    def predict_raw(self, features: dict) -> float:
        """Uncalibrated stacker output (used internally / for calibration)."""
        return float(self.predict_raw_many([features])[0])

    def predict_raw_many(self, rows: list[dict]) -> np.ndarray:
        """Uncalibrated fused outputs for many feature dicts, batched.

        Batched prediction matters for offline workloads (backtesting,
        operating-point tuning)."""
        X = np.array([[r[f] for f in ML_FEATURES] for r in rows], dtype=float)
        Xs = self._prepare(X)

        p_lr = self.models["logistic_regression"].predict_proba(Xs)[:, 1]
        p_rf = self.models["random_forest"].predict_proba(Xs)[:, 1]
        p_xgb = self.models["xgboost"].predict_proba(Xs)[:, 1]

        if self._stacked_mode:
            iso_score = -self.models["isolation_forest"].decision_function(Xs)
            iso_pct = np.searchsorted(self._iso_train_sorted, iso_score, side='left') / len(self._iso_train_sorted)
            X_stack = np.column_stack([p_lr, p_rf, p_xgb, iso_pct])
            return self.stacker.predict_proba(X_stack)[:, 1]
        else:
            return (WEIGHTS["random_forest"] * p_rf
                    + WEIGHTS["xgboost"] * p_xgb
                    + WEIGHTS["logistic_regression"] * p_lr)

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        """Calibrated fused probabilities from a pre-built numpy matrix.

        Skips the dict→array conversion, giving maximum throughput for
        offline scoring / batch pipelines where features are already in
        matrix form.
        """
        raw = self.predict_raw_matrix(X)
        if self.calibrator is None:
            return np.clip(raw, 0.0, 1.0)
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def predict_raw_matrix(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated fused outputs from a pre-built numpy matrix.

        X must be shape (n, len(ML_FEATURES)) with features in ML_FEATURES order.
        This is the fastest scoring path — no Python loops, no dict conversion.
        """
        Xs = self._prepare(X)
        p_lr = self.models["logistic_regression"].predict_proba(Xs)[:, 1]
        p_rf = self.models["random_forest"].predict_proba(Xs)[:, 1]
        p_xgb = self.models["xgboost"].predict_proba(Xs)[:, 1]

        if self._stacked_mode:
            iso_score = -self.models["isolation_forest"].decision_function(Xs)
            iso_pct = np.searchsorted(self._iso_train_sorted, iso_score, side='left') / len(self._iso_train_sorted)
            X_stack = np.column_stack([p_lr, p_rf, p_xgb, iso_pct])
            return self.stacker.predict_proba(X_stack)[:, 1]
        else:
            return (WEIGHTS["random_forest"] * p_rf
                    + WEIGHTS["xgboost"] * p_xgb
                    + WEIGHTS["logistic_regression"] * p_lr)

    def predict_many(self, rows: list[dict]) -> np.ndarray:
        """Calibrated fused probabilities for many feature dicts (batched)."""
        raw = self.predict_raw_many(rows)
        if self.calibrator is None:
            return np.clip(raw, 0.0, 1.0)
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def predict_combined(self, features: dict) -> tuple[float, float, dict]:
        """Single-pass calibrated + weighted prediction.

        Returns (calibrated_score, weighted_score, uncertainty).
        """
        comp = self.components(features)
        outputs = comp["outputs"]
        vals = list(outputs.values())
        uncertainty = {
            "model_variance": round(float(np.var(vals)), 6),
            "model_disagreement": round(float(max(vals) - min(vals)), 6),
            "individual_outputs": {k: round(v, 4) for k, v in outputs.items()},
        }
        if self._stacked_mode:
            p_lr = outputs["logistic_regression"]
            p_rf = outputs["random_forest"]
            p_xgb = outputs["xgboost"]
            iso_pct = outputs["isolation_forest"]
            X_stack = np.array([[p_lr, p_rf, p_xgb, iso_pct]])
            raw = float(self.stacker.predict_proba(X_stack)[0, 1])
        else:
            raw = sum(WEIGHTS.get(k, 0) * v for k, v in outputs.items())
        if self.calibrator is not None:
            calibrated = float(np.clip(self.calibrator.predict(np.array([raw]))[0], 0.0, 1.0))
        else:
            calibrated = float(np.clip(raw, 0.0, 1.0))
        # weighted_score: same as calibrated in v3 mode
        weighted = calibrated
        return calibrated, weighted, uncertainty

    def predict_weighted(self, features: dict) -> float:
        """Precision-optimized ensemble score (0-1).

        Uses tuned weights: RF primary, XGB secondary, LR tertiary."""
        X = np.array([[features[f] for f in ML_FEATURES]], dtype=float)
        Xs = self._prepare(X)
        p_lr = float(self.models["logistic_regression"].predict_proba(Xs)[0, 1])
        p_rf = float(self.models["random_forest"].predict_proba(Xs)[0, 1])
        p_xgb = float(self.models["xgboost"].predict_proba(Xs)[0, 1])
        return (WEIGHTS["random_forest"] * p_rf
                + WEIGHTS["xgboost"] * p_xgb
                + WEIGHTS["logistic_regression"] * p_lr)

    def predict_weighted_many(self, rows: list[dict]) -> np.ndarray:
        """Batched precision-optimized ensemble scores."""
        X = np.array([[r[f] for f in ML_FEATURES] for r in rows], dtype=float)
        Xs = self._prepare(X)
        p_lr = self.models["logistic_regression"].predict_proba(Xs)[:, 1]
        p_rf = self.models["random_forest"].predict_proba(Xs)[:, 1]
        p_xgb = self.models["xgboost"].predict_proba(Xs)[:, 1]
        return (WEIGHTS["random_forest"] * p_rf
                + WEIGHTS["xgboost"] * p_xgb
                + WEIGHTS["logistic_regression"] * p_lr)

    def odds_of(self, p: float) -> float:
        """Odds for a calibrated probability: p / (1 - p), inf at p == 1."""
        if p >= 1.0:
            return float("inf")
        return float(p / (1.0 - p))

    def predict_with_finance(self, features: dict) -> tuple[float, dict]:
        """Finance-enhanced prediction: combines base model with micro-fraud detection.

        If the finance model is available and the transaction has subscription/micro-fraud
        indicators, it boosts the main fraud score. This catches small recurring charges
        that the base model might miss.

        Returns (probability, metadata) where metadata contains:
        - base_score: the base model score before finance boost
        - finance_score: the finance model score for micro-fraud detection
        - finance_boost: how much the score was boosted
        - is_micro_fraud_suspect: True if finance model flagged this as micro-fraud
        """
        # Get base score
        base_score, uncertainty = self.predict(features)

        metadata = {
            "base_score": base_score,
            "finance_score": 0.0,
            "finance_boost": 0.0,
            "is_micro_fraud_suspect": False,
        }

        if not self._has_finance_model:
            return base_score, metadata

        # Build 27-feature vector for finance model
        finance_vec = np.zeros((1, len(FINANCE_FEATURES)), dtype=float)
        for i, feat in enumerate(FINANCE_FEATURES):
            finance_vec[0, i] = features.get(feat, 0.0)

        # Impute and scale
        finance_vec = self.finance_imputer.transform(finance_vec)
        finance_vec = np.nan_to_num(finance_vec, nan=0.0, posinf=10.0, neginf=-10.0)
        finance_scaled = self.finance_scaler.transform(finance_vec)

        # Finance model prediction
        finance_score = float(self.finance_model.predict_proba(finance_scaled)[0, 1])
        metadata["finance_score"] = round(finance_score, 4)

        # Apply boost ONLY if finance-specific features are non-zero.
        # When all 6 finance features are zero (privacy layer doesn't compute them),
        # the model scores ~0.99 on all-zeros which is a false positive factory.
        finance_feats = FINANCE_FEATURES[-6:]  # the 6 finance-specific features
        has_finance_data = any(features.get(f, 0.0) != 0.0 for f in finance_feats)
        metadata["has_finance_data"] = has_finance_data

        if finance_score > FINANCE_BOOST_THRESHOLD and has_finance_data:
            boost = FINANCE_BOOST_WEIGHT * (finance_score - FINANCE_BOOST_THRESHOLD)
            boosted = float(np.clip(base_score + boost, 0.0, 1.0))
            metadata["finance_boost"] = round(boost, 4)
            metadata["is_micro_fraud_suspect"] = True
            return boosted, metadata

        return base_score, metadata
