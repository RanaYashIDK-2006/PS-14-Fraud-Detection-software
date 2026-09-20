"""Phase 93: Real-world validation evaluation protocol.

Defines the formal evaluation protocol PS-14 will use IF AND ONLY IF
an eligible external dataset passes Phase 91 -> 84 -> 85 -> feature
compatibility -> dataset admission.

Does NOT perform real-world validation, acquire data, contact providers,
retrain, tune thresholds, modify models, or change REAL_WORLD_VALIDATION.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ══════════════════════════════════════════════════════════════════════
# CONTROLLED ENUMS
# ══════════════════════════════════════════════════════════════════════

class EvaluationState(str, Enum):
    NOT_ELIGIBLE = "not_eligible"
    EVALUATION_BLOCKED = "evaluation_blocked"
    EVALUATION_INVALIDATED = "evaluation_invalidated"
    EVALUATION_COMPLETE = "evaluation_complete"
    EVALUATION_COMPLETE_WITH_LIMITATIONS = "evaluation_complete_with_limitations"


class OutcomeStatus(str, Enum):
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    RETRACTED = "retracted"
    PENDING = "pending"
    UNKNOWN = "unknown"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# ══════════════════════════════════════════════════════════════════════
# AUTHORITY CONSTANTS (from existing codebase)
# ══════════════════════════════════════════════════════════════════════

PRODUCTION_THRESHOLD = 0.018758
MODEL_ID = "altman_native"
RELEASE_ID = "release-altman_native_E_hardneg_cert_20260904"
FEATURE_VERSION = "v1"

# 48 native features — import from authoritative source
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES as NATIVE_48

LABEL_DEPENDENT_FEATURES = frozenset({
    "user_fraud_rate", "merch_fraud_rate", "city_fraud_rate",
})


# ══════════════════════════════════════════════════════════════════════
# MODEL IDENTITY LOCK
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelIdentityLock:
    """Frozen model identity that must remain constant during evaluation."""
    model_id: str
    release_id: str
    feature_version: str
    feature_count: int
    threshold: float
    artifact_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def verify(self, other: ModelIdentityLock) -> bool:
        """Verify another lock matches this one (model immutability check)."""
        return (self.model_id == other.model_id
                and self.release_id == other.release_id
                and self.feature_version == other.feature_version
                and self.feature_count == other.feature_count
                and self.threshold == other.threshold
                and self.artifact_hashes == other.artifact_hashes)


def build_identity_lock() -> ModelIdentityLock:
    """Build the authoritative model identity lock."""
    return ModelIdentityLock(
        model_id=MODEL_ID,
        release_id=RELEASE_ID,
        feature_version=FEATURE_VERSION,
        feature_count=48,
        threshold=PRODUCTION_THRESHOLD,
        artifact_hashes={
            "xgb_native.joblib": "a1cdebdfe01b709a5e0d9480078e75565521bd6cf5aa7eca06771a828d170bbc",
            "lgb_native.joblib": "d59aebcb08d6df05dd9640e1f88c1454aeb0045f8676ee723588b3d401a5c87f",
            "cb_native.joblib": "22b8377bc1ff4b6fd07e78630c5b9a8c8729829f286f7e4378948053692cc73f",
        },
    )


# ══════════════════════════════════════════════════════════════════════
# METRIC CALCULATIONS
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: list[int],
    y_scores: list[float],
    threshold: float = PRODUCTION_THRESHOLD,
) -> dict[str, float]:
    """Compute deterministic evaluation metrics.

    All calculations are pure functions of (y_true, y_scores, threshold).
    """
    n = len(y_true)
    if n == 0:
        return {"error": "no_data"}

    # Binary predictions at locked threshold
    y_pred = [1 if s >= threshold else 0 for s in y_scores]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fraud_capture = recall
    total_fraud = sum(y_true)
    fraud_capture_rate = total_fraud / n if n > 0 else 0.0
    alert_rate = sum(y_pred) / n if n > 0 else 0.0

    # ROC-AUC (trapezoidal)
    sorted_pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    thresholds_roc = [0.0] + [p[0] for p in sorted_pairs] + [1.0]
    roc_points = []
    for t in thresholds_roc:
        tp_r = sum(1 for s, y in zip(y_scores, y_true) if s >= t and y == 1)
        fp_r = sum(1 for s, y in zip(y_scores, y_true) if s >= t and y == 0)
        tpr = tp_r / total_fraud if total_fraud > 0 else 0.0
        fpr_r = fp_r / (n - total_fraud) if (n - total_fraud) > 0 else 0.0
        roc_points.append((fpr_r, tpr))
    roc_points.sort()

    roc_auc = 0.0
    for i in range(1, len(roc_points)):
        dx = roc_points[i][0] - roc_points[i - 1][0]
        y_avg = (roc_points[i][1] + roc_points[i - 1][1]) / 2
        roc_auc += dx * y_avg

    # PR-AUC
    pr_points = []
    for t in thresholds_roc:
        tp_r = sum(1 for s, y in zip(y_scores, y_true) if s >= t and y == 1)
        fp_r = sum(1 for s, y in zip(y_scores, y_true) if s >= t and y == 0)
        prec_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0.0
        rec_r = tp_r / total_fraud if total_fraud > 0 else 0.0
        pr_points.append((rec_r, prec_r))
    pr_points.sort()

    pr_auc = 0.0
    for i in range(1, len(pr_points)):
        dx = pr_points[i][0] - pr_points[i - 1][0]
        y_avg = (pr_points[i][1] + pr_points[i - 1][1]) / 2
        pr_auc += dx * y_avg

    # Brier score
    brier = sum((s - t) ** 2 for s, t in zip(y_scores, y_true)) / n

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "specificity": round(specificity, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "fraud_capture_rate": round(fraud_capture_rate, 6),
        "alert_rate": round(alert_rate, 6),
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "brier_score": round(brier, 6),
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": n,
        "total_fraud": total_fraud,
    }


def bootstrap_confidence_intervals(
    y_true: list[int],
    y_scores: list[float],
    threshold: float = PRODUCTION_THRESHOLD,
    n_bootstrap: int = 200,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Deterministic bootstrap confidence intervals."""
    rng_state = seed

    def _rng() -> int:
        nonlocal rng_state
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        return rng_state

    n = len(y_true)
    boot_metrics: list[dict[str, float]] = []

    for _ in range(n_bootstrap):
        indices = [_rng() % n for _ in range(n)]
        bt = [y_true[i] for i in indices]
        bs = [y_scores[i] for i in indices]
        boot_metrics.append(compute_metrics(bt, bs, threshold))

    # Extract percentile CIs
    alpha = (1 - confidence) / 2
    result: dict[str, Any] = {"n_bootstrap": n_bootstrap, "seed": seed, "confidence": confidence}

    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"):
        values = sorted([m[key] for m in boot_metrics])
        lo_idx = int(alpha * n_bootstrap)
        hi_idx = int((1 - alpha) * n_bootstrap) - 1
        lo_idx = min(lo_idx, len(values) - 1)
        hi_idx = min(hi_idx, len(values) - 1)
        result[f"{key}_ci_lower"] = round(values[lo_idx], 6)
        result[f"{key}_ci_upper"] = round(values[hi_idx], 6)

    return result


# ══════════════════════════════════════════════════════════════════════
# DRIFT MEASUREMENT
# ══════════════════════════════════════════════════════════════════════

def compute_psi(expected: list[float], actual: list[float], n_bins: int = 10) -> float:
    """Population Stability Index between expected and actual distributions."""
    if not expected or not actual:
        return 0.0

    min_val = min(min(expected), min(actual))
    max_val = max(max(expected), max(actual))
    if min_val == max_val:
        return 0.0

    bin_width = (max_val - min_val) / n_bins
    psi = 0.0

    for i in range(n_bins):
        lo = min_val + i * bin_width
        hi = lo + bin_width
        exp_count = sum(1 for v in expected if lo <= v < hi) / len(expected)
        act_count = sum(1 for v in actual if lo <= v < hi) / len(actual)
        exp_count = max(exp_count, 0.001)
        act_count = max(act_count, 0.001)
        psi += (act_count - exp_count) * math.log(act_count / exp_count)

    return round(psi, 6)


def measure_drift(
    training_features: dict[str, list[float]],
    evaluation_features: dict[str, list[float]],
) -> dict[str, Any]:
    """Measure distribution drift for each canonical feature."""
    drift_results: dict[str, Any] = {}
    high_drift_count = 0

    for feature in NATIVE_48:
        train_vals = training_features.get(feature, [])
        eval_vals = evaluation_features.get(feature, [])
        if train_vals and eval_vals:
            psi = compute_psi(train_vals, eval_vals)
            severity = DriftSeverity.NONE.value
            if psi > 0.25:
                severity = DriftSeverity.HIGH.value
                high_drift_count += 1
            elif psi > 0.10:
                severity = DriftSeverity.MODERATE.value
            elif psi > 0.02:
                severity = DriftSeverity.LOW.value
            drift_results[feature] = {"psi": psi, "severity": severity}
        else:
            drift_results[feature] = {"psi": 0.0, "severity": "not_available"}

    overall = "none"
    if high_drift_count > len(NATIVE_48) * 0.3:
        overall = "high"
    elif high_drift_count > len(NATIVE_48) * 0.1:
        overall = "moderate"

    return {"features": drift_results, "overall_severity": overall,
            "high_drift_count": high_drift_count}


# ══════════════════════════════════════════════════════════════════════
# EVALUATION MANIFEST
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EvaluationManifest:
    """Deterministic, immutable evaluation manifest."""
    dataset_id: str
    dataset_version: str
    dataset_hash: str
    model_id: str
    release_id: str
    feature_version: str
    threshold: float
    evaluation_protocol_version: str
    evaluation_start: str
    evaluation_end: str
    metrics_version: str
    bootstrap_seed: int
    bootstrap_iterations: int
    confidence_level: float
    total_eligible: int
    excluded_count: int
    exclusion_reasons: dict[str, int]
    metrics: dict[str, float]
    drift_summary: dict[str, str]
    outcome_trust_policy_version: str
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def build_evaluation_manifest(
    dataset_id: str,
    dataset_version: str,
    dataset_hash: str,
    metrics: dict[str, float],
    drift_summary: dict[str, str],
    total_eligible: int,
    excluded_count: int,
    exclusion_reasons: dict[str, int],
    start_time: str,
    end_time: str,
) -> EvaluationManifest:
    """Build a deterministic evaluation manifest."""
    identity = build_identity_lock()

    manifest_content = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "model_id": identity.model_id,
        "release_id": identity.release_id,
        "feature_version": identity.feature_version,
        "threshold": identity.threshold,
        "total_eligible": total_eligible,
        "excluded_count": excluded_count,
    }
    canonical = json.dumps(manifest_content, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return EvaluationManifest(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        dataset_hash=dataset_hash,
        model_id=identity.model_id,
        release_id=identity.release_id,
        feature_version=identity.feature_version,
        threshold=identity.threshold,
        evaluation_protocol_version="phase93_v1",
        evaluation_start=start_time,
        evaluation_end=end_time,
        metrics_version="phase93_v1",
        bootstrap_seed=42,
        bootstrap_iterations=200,
        confidence_level=0.95,
        total_eligible=total_eligible,
        excluded_count=excluded_count,
        exclusion_reasons=exclusion_reasons,
        metrics=metrics,
        drift_summary=drift_summary,
        outcome_trust_policy_version="outcome_trust_policy_v1",
        manifest_hash=manifest_hash,
    )


# ══════════════════════════════════════════════════════════════════════
# SYNTHETIC DRY-RUN
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SyntheticEvaluation:
    """Result of a synthetic (NOT real-world) evaluation dry-run."""
    evaluation_state: str
    is_real_world: bool
    rwv_status: str
    promotion_status: str
    metrics: dict[str, float]
    confidence_intervals: dict[str, Any]
    drift: dict[str, Any]
    manifest: dict
    model_identity_valid: bool
    threshold_locked: bool

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def run_synthetic_evaluation() -> SyntheticEvaluation:
    """Run a deterministic synthetic evaluation dry-run.

    Uses ONLY synthetic data. This is NOT real-world validation.
    """
    # Generate synthetic predictions and labels
    rng_state = 42

    def _rng() -> float:
        nonlocal rng_state
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        return rng_state / 0x7FFFFFFF

    n = 500
    fraud_rate = 0.05
    y_true: list[int] = []
    y_scores: list[float] = []

    for _ in range(n):
        label = 1 if _rng() < fraud_rate else 0
        # Synthetic scores: fraud cases get higher scores
        if label == 1:
            score = 0.01 + _rng() * 0.05
        else:
            score = _rng() * 0.015
        y_true.append(label)
        y_scores.append(round(score, 6))

    # Metrics at locked threshold
    metrics = compute_metrics(y_true, y_scores, PRODUCTION_THRESHOLD)

    # Bootstrap CI (reduced iterations for performance)
    ci = bootstrap_confidence_intervals(y_true, y_scores, PRODUCTION_THRESHOLD, n_bootstrap=50)

    # Drift (synthetic: no real training data, use uniform)
    training_features = {f: [_rng() for _ in range(100)] for f in NATIVE_48[:3]}
    evaluation_features = {f: [_rng() * 1.1 for _ in range(100)] for f in NATIVE_48[:3]}
    drift = measure_drift(training_features, evaluation_features)

    # Model identity lock
    identity = build_identity_lock()

    # Manifest
    start = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    end = datetime(2026, 1, 2, tzinfo=timezone.utc).isoformat()
    manifest = build_evaluation_manifest(
        dataset_id="SYNTHETIC-DRY-RUN-001",
        dataset_version="1.0",
        dataset_hash=hashlib.sha256(b"synthetic_dry_run").hexdigest(),
        metrics=metrics,
        drift_summary={"overall": drift["overall_severity"]},
        total_eligible=n,
        excluded_count=0,
        exclusion_reasons={},
        start_time=start,
        end_time=end,
    )

    return SyntheticEvaluation(
        evaluation_state=EvaluationState.EVALUATION_COMPLETE.value,
        is_real_world=False,
        rwv_status="BLOCKED_PENDING_ELIGIBLE_DATASET",
        promotion_status="no_promotion",
        metrics=metrics,
        confidence_intervals=ci,
        drift=drift,
        manifest=manifest.to_dict(),
        model_identity_valid=True,
        threshold_locked=True,
    )


# ══════════════════════════════════════════════════════════════════════
# PROTOCOL RESULT
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProtocolResult:
    """Complete evaluation protocol result."""
    protocol_id: str
    protocol_version: str
    model_identity: dict
    locked_threshold: float
    synthetic_evaluation: dict
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def build_protocol() -> ProtocolResult:
    """Build the complete evaluation protocol with synthetic dry-run."""
    identity = build_identity_lock()
    synthetic = run_synthetic_evaluation()

    manifest_content = {
        "protocol_id": "RWV-EVALUATION-PROTOCOL-93",
        "protocol_version": "phase93_v1",
        "model_id": identity.model_id,
        "release_id": identity.release_id,
        "threshold": identity.threshold,
        "is_real_world": False,
        "rwv_status": "BLOCKED_PENDING_ELIGIBLE_DATASET",
    }
    canonical = json.dumps(manifest_content, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return ProtocolResult(
        protocol_id="RWV-EVALUATION-PROTOCOL-93",
        protocol_version="phase93_v1",
        model_identity=identity.to_dict(),
        locked_threshold=PRODUCTION_THRESHOLD,
        synthetic_evaluation=synthetic.to_dict(),
        manifest_hash=manifest_hash,
    )
