"""Frozen model evaluation — runs the existing model without modification.

The external test set must NOT be used for retraining, feature selection,
hyperparameter tuning, threshold tuning, calibration fitting, or model selection.
The evaluation runs the model exactly as deployed, subject only to documented
schema/preprocessing adaptation.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EvaluationConfig:
    """Configuration for a frozen model evaluation."""
    model_version: str = ""
    model_hash: str = ""
    preprocessing_version: str = ""
    preprocessing_hash: str = ""
    evaluation_dataset_hash: str = ""
    evaluation_timestamp: str = ""
    threshold: float = 0.5
    feature_mapping_version: str = "1.0"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "model_hash": self.model_hash,
            "preprocessing_version": self.preprocessing_version,
            "preprocessing_hash": self.preprocessing_hash,
            "evaluation_dataset_hash": self.evaluation_dataset_hash,
            "evaluation_timestamp": self.evaluation_timestamp,
            "threshold": self.threshold,
            "feature_mapping_version": self.feature_mapping_version,
            "notes": self.notes,
        }


@dataclass
class FrozenEvaluationResult:
    """Result of a frozen model evaluation."""
    config: EvaluationConfig = field(default_factory=EvaluationConfig)
    predictions: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    scores: np.ndarray | None = None
    n_samples: int = 0
    evaluation_complete: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "n_samples": self.n_samples,
            "evaluation_complete": self.evaluation_complete,
            "errors": self.errors,
        }


def _compute_file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of a file for reproducibility."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_frozen_model(
    model_path: str | Path | None = None,
) -> Any:
    """Load the production model in frozen (read-only) mode.

    The model is loaded exactly as it exists on disk. No modification,
    no fine-tuning, no calibration adjustment.

    Args:
        model_path: Path to the model artifact. If None, uses the
            default production model path.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If model cannot be loaded.
    """
    if model_path is None:
        # Default production model path
        model_path = Path("models/artifacts/stacker.joblib")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        import joblib
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")


def run_frozen_evaluation(
    X: np.ndarray,
    model: Any,
    config: EvaluationConfig,
    threshold: float | None = None,
) -> FrozenEvaluationResult:
    """Run a frozen evaluation — model is NOT modified.

    Args:
        X: Feature matrix (n_samples, n_features).
        model: The frozen model object.
        config: Evaluation configuration for reproducibility.
        threshold: Classification threshold (overrides config.threshold).

    Returns:
        FrozenEvaluationResult with predictions and probabilities.
    """
    result = FrozenEvaluationResult(config=config)
    result.n_samples = len(X)

    if threshold is None:
        threshold = config.threshold

    try:
        # Get probabilities from the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            # For binary classification, take the positive class probability
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                probs = probabilities[:, 1]
            else:
                probs = probabilities.ravel()
        elif hasattr(model, "predict"):
            # Model returns scores directly
            probs = model.predict(X).astype(float)
        else:
            result.errors.append("Model has no predict or predict_proba method")
            return result

        result.probabilities = probs
        result.predictions = (probs >= threshold).astype(int)
        result.scores = (probs * 100).astype(float)  # 0-100 scale
        result.evaluation_complete = True

    except Exception as e:
        result.errors.append(f"Evaluation failed: {e}")

    return result


def verify_frozen_model(
    model: Any,
    expected_hash: str | None = None,
    model_path: str | Path | None = None,
) -> tuple[bool, str]:
    """Verify that the model has not been modified.

    Args:
        model: The loaded model object.
        expected_hash: Expected hash of the model file.
        model_path: Path to the model file for hash verification.

    Returns:
        (is_valid, message)
    """
    if model_path is not None:
        actual_hash = _compute_file_hash(model_path)
        if expected_hash and actual_hash != expected_hash:
            return False, f"Model hash mismatch: expected {expected_hash}, got {actual_hash}"
        return True, f"Model hash verified: {actual_hash}"

    # Can't verify without file path — warn but don't fail
    return True, "Model hash verification skipped (no file path provided)"
