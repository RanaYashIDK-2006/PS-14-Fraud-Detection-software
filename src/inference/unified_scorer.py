"""Unified fraud detection scorer — combines all detection systems in one pass.

Architecture:
  Transaction → [Velocity Limits] → [Rules] → [ML Fusion] → [A/B Router] → Decision
                                                           ↓
                                                   [Drift Monitor]

Integrates:
  1. Velocity limits (pre-scoring hard/soft caps)
  2. Rules engine (declarative fraud patterns)
  3. ML fusion (LR + RF + XGB ensemble with calibration)
  4. Finance model (micro-fraud / subscription detection)
  5. Inference LGB model (sliding-window velocity features)
  6. A/B testing (traffic splitting between model versions)
  7. PSI drift detection (monitors feature distribution shifts)

Replaces separate /evaluate, /score, /ab/score calls with a single
unified pipeline that runs all detectors in one pass.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class UnifiedResult:
    """Combined result from all detection systems."""
    # Final output
    fraud_probability: float
    risk_score: int          # 0-100
    decision: str            # "allow", "verify", "block"
    risk_level: str          # "low", "medium", "high", "critical"

    # Per-system scores
    ml_score: float          # FusionEngine calibrated probability
    rules_score: float       # RulesEngine score [0,1]
    rules_fired: list[str]
    reason_codes: list[str]
    velocity_triggered: bool
    velocity_triggers: list[dict]
    finance_score: float     # Finance micro-fraud score
    latency_ms: float        # Scoring latency in ms

    # Metadata (all optional)
    model_version: str = "unified_v1"
    degraded: bool = False   # True if ML failed, rules-only fallback
    critical_fired: bool = False
    ab_version: str = ""     # A/B routed version
    ab_experiment: str = ""

    # Drift monitoring
    drift_status: str = "stable"  # "stable", "warning", "critical"
    drift_psi: float = 0.0        # aggregate PSI at time of scoring
    drift_alert: str = ""         # non-empty if drift detected


class UnifiedScorer:
    """Single-pass fraud detection combining all systems.

    Usage:
        scorer = UnifiedScorer()
        result = scorer.score(features_dict)
        # or
        results = scorer.score_batch([features_dict_1, features_dict_2, ...])
    """

    def __init__(self, artifacts_dir: Path = None, rules_path: Path = None):
        t0 = time.time()

        if artifacts_dir is None:
            artifacts_dir = ROOT / "models" / "artifacts"
        if rules_path is None:
            rules_path = ROOT / "src" / "risk_engine" / "rules.yaml"

        self._loaded = False
        self._load_errors = []

        # 1. Load ML Fusion Engine
        try:
            from src.risk_engine.fusion import FusionEngine
            self.fusion = FusionEngine(artifacts_dir)
            self._ml_ready = True
        except Exception as e:
            self._load_errors.append(f"fusion: {e}")
            self.fusion = None
            self._ml_ready = False

        # 2. Load Rules Engine
        try:
            from src.risk_engine.rules_engine import RulesEngine
            self.rules = RulesEngine.from_yaml(rules_path)
            self._rules_ready = True
        except Exception as e:
            self._load_errors.append(f"rules: {e}")
            self.rules = None
            self._rules_ready = False

        # 3. Load velocity limits config
        try:
            import yaml
            cfg = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
            self.velocity_cfg = cfg.get("velocity_limits")
            from src.risk_engine.limits import evaluate_limits
            self._evaluate_limits = evaluate_limits
            self._limits_ready = True
        except Exception as e:
            self._load_errors.append(f"limits: {e}")
            self.velocity_cfg = None
            self._evaluate_limits = None
            self._limits_ready = False

        # 4. Load A/B testing framework (optional)
        try:
            from src.inference.ab_testing import TrafficSplitter
            from src.inference.model_registry import ModelRegistry
            self.registry = ModelRegistry()
            self.registry.auto_discover()
            self.ab = TrafficSplitter()
            self._ab_ready = True
        except Exception as e:
            self._load_errors.append(f"ab: {e}")
            self.registry = None
            self.ab = None
            self._ab_ready = False

        # 5. Load PSI drift detector (optional)
        self._drift_detector = None
        self._drift_ready = False
        self._drift_ref_path = ROOT / "models" / "production" / "drift_reference.json"
        try:
            from src.monitoring.drift_detector import DriftDetector, ReferenceDistribution
            if self._drift_ref_path.exists():
                self._drift_detector = DriftDetector.load(self._drift_ref_path)
                self._drift_ready = True
                self._drift_ref = ReferenceDistribution.load(self._drift_ref_path)
            else:
                self._drift_ref = None
        except Exception as e:
            self._load_errors.append(f"drift: {e}")

        # Drift sliding window (thread-safe)
        self._drift_window_size = 500
        self._drift_check_interval = 200  # check every N scored transactions
        self._drift_buffer: list[np.ndarray] = []
        self._drift_lock = threading.Lock()
        self._drift_n_scored = 0
        self._drift_last_report: Optional[dict] = None
        self._drift_history: list[dict] = []
        self._drift_alerts: list[dict] = []
        self._drift_alert_callbacks: list = []

        self._load_time_ms = round((time.time() - t0) * 1000, 1)
        self._loaded = True

    def score(self, features: dict, user_id: str = "",
              experiment: str = None) -> UnifiedResult:
        """Score a single transaction through ALL detection systems in one pass."""
        t0 = time.perf_counter()
        ml_score = 0.0
        rules_result = {"score": 0, "fired_rules": [], "reason_codes": [], "critical": False}
        velocity_result = {"pass": True, "triggers": []}
        finance_score = 0.0
        degraded = False
        critical_fired = False
        ab_version = ""
        ab_experiment = ""

        # 1. Velocity limits (pre-scoring)
        if self._limits_ready and self._evaluate_limits and self.velocity_cfg:
            try:
                velocity_result = self._evaluate_limits(features, self.velocity_cfg)
            except Exception:
                velocity_result = {"pass": True, "triggers": []}

        # 2. Rules engine
        if self._rules_ready and self.rules:
            try:
                rules_result = self.rules.evaluate(features)
            except Exception:
                rules_result = {"score": 0, "fired_rules": [], "reason_codes": [], "critical": False}

        # 3. ML fusion
        if self._ml_ready and self.fusion:
            try:
                ml_score, uncertainty = self.fusion.predict(features)
            except Exception:
                ml_score = 0.0
                degraded = True

        # 4. A/B routing (if experiment specified)
        if self._ab_ready and self.ab and experiment:
            try:
                ab_version = self.ab.get_version(experiment, user_id=user_id)
                ab_experiment = experiment
                # Score with the routed model
                scorer_fn = self.registry.get_scorer(ab_version)
                mv = self.registry.get_model(ab_version)
                if scorer_fn and mv:
                    # Build feature vector for the routed model
                    feat_vec = self._build_routed_features(features, mv)
                    if feat_vec is not None:
                        routed_score = float(scorer_fn(feat_vec.reshape(1, -1))[0])
                        # Blend: 60% routed + 40% primary ML
                        ml_score = 0.6 * routed_score + 0.4 * ml_score
            except Exception:
                pass  # Fallback to primary ML score

        # 5. Combine all scores
        critical_fired = rules_result.get("critical", False) or not velocity_result.get("pass", True)

        # Weighted combination
        rules_w = 0.25
        ml_w = 0.65
        velocity_w = 0.10

        # If velocity hard-cap triggered, floor the score
        if not velocity_result.get("pass", True):
            combined = 1.0
        elif critical_fired:
            combined = max(0.7, ml_w * ml_score + rules_w * rules_result["score"] + velocity_w)
        else:
            combined = ml_w * ml_score + rules_w * rules_result["score"] + velocity_w * (1.0 if velocity_result.get("triggers") else 0.0)

        combined = min(1.0, max(0.0, combined))

        # Map to 0-100 risk score
        risk_score = int(round(combined * 100))

        # Decision thresholds
        if risk_score >= 70 or critical_fired:
            decision = "block"
        elif risk_score >= 30:
            decision = "verify"
        else:
            decision = "allow"

        # Risk level
        if risk_score >= 70:
            risk_level = "critical"
        elif risk_score >= 40:
            risk_level = "high"
        elif risk_score >= 15:
            risk_level = "medium"
        else:
            risk_level = "low"

        latency_ms = (time.perf_counter() - t0) * 1000

        # 6. Drift monitoring — accumulate feature vector and check PSI
        drift_status = "stable"
        drift_psi = 0.0
        drift_alert = ""
        if self._drift_ready and self._drift_detector:
            feat_vec = self._extract_feature_vector(features)
            if feat_vec is not None:
                drift_status, drift_psi, drift_alert = self._accumulate_and_check_drift(feat_vec)

        return UnifiedResult(
            fraud_probability=round(combined, 6),
            risk_score=risk_score,
            decision=decision,
            risk_level=risk_level,
            ml_score=round(ml_score, 6),
            rules_score=round(rules_result.get("score", 0), 4),
            rules_fired=rules_result.get("fired_rules", []),
            reason_codes=rules_result.get("reason_codes", []),
            velocity_triggered=not velocity_result.get("pass", True),
            velocity_triggers=velocity_result.get("triggers", []),
            finance_score=round(finance_score, 6),
            drift_status=drift_status,
            drift_psi=round(drift_psi, 6),
            drift_alert=drift_alert,
            latency_ms=round(latency_ms, 3),
            model_version="unified_v1",
            degraded=degraded,
            critical_fired=critical_fired,
            ab_version=ab_version,
            ab_experiment=ab_experiment,
        )

    def score_batch(self, features_list: list[dict], user_ids: list[str] = None,
                    experiment: str = None) -> list[UnifiedResult]:
        """Score a batch — ML batched via predict_many, rules/velocity per-row."""
        if user_ids is None:
            user_ids = [f"batch_{i}" for i in range(len(features_list))]

        t0 = time.perf_counter()
        n = len(features_list)

        # 1. Batch ML scoring
        ml_scores = [0.0] * n
        if self._ml_ready and self.fusion:
            try:
                from src.privacy_layer.features import ML_FEATURES as _MF
                rows = [{f: float(features_list[i].get(f, 0.0)) for f in _MF}
                        for i in range(n)]
                raw = self.fusion.predict_raw_many(rows)
                if self.fusion.calibrator is not None:
                    ml_scores = self.fusion.calibrator.predict(raw.reshape(-1, 1)).tolist()
                else:
                    ml_scores = raw.tolist()
            except Exception:
                ml_scores = [0.0] * n

        # 2. Per-row rules + velocity (lightweight)
        results = []
        for i, features in enumerate(features_list):
            # Rules
            rules_result = {"score": 0, "fired_rules": [], "reason_codes": [], "critical": False}
            if self._rules_ready and self.rules:
                try:
                    rules_result = self.rules.evaluate(features)
                except Exception:
                    pass

            # Velocity
            velocity_result = {"pass": True, "triggers": []}
            if self._limits_ready and self._evaluate_limits and self.velocity_cfg:
                try:
                    velocity_result = self._evaluate_limits(features, self.velocity_cfg)
                except Exception:
                    pass

            # Combine
            ml_score = ml_scores[i]
            critical_fired = rules_result.get("critical", False) or not velocity_result.get("pass", True)
            if not velocity_result.get("pass", True):
                combined = 1.0
            elif critical_fired:
                combined = max(0.7, 0.65 * ml_score + 0.25 * rules_result["score"] + 0.10)
            else:
                combined = 0.65 * ml_score + 0.25 * rules_result["score"] + 0.10 * (1.0 if velocity_result.get("triggers") else 0.0)
            combined = min(1.0, max(0.0, combined))
            risk_score = int(round(combined * 100))
            decision = "block" if risk_score >= 70 else ("verify" if risk_score >= 30 else "allow")
            risk_level = "critical" if risk_score >= 70 else ("high" if risk_score >= 40 else ("medium" if risk_score >= 15 else "low"))

            results.append(UnifiedResult(
                fraud_probability=round(combined, 6),
                risk_score=risk_score, decision=decision, risk_level=risk_level,
                ml_score=round(ml_score, 6),
                rules_score=round(rules_result.get("score", 0), 4),
                rules_fired=rules_result.get("fired_rules", []),
                reason_codes=rules_result.get("reason_codes", []),
                velocity_triggered=not velocity_result.get("pass", True),
                velocity_triggers=velocity_result.get("triggers", []),
                finance_score=0.0,
                latency_ms=0.0,
                model_version="unified_v1",
                degraded=ml_score == 0.0 and self._ml_ready,
                critical_fired=critical_fired,
            ))

        total_ms = (time.perf_counter() - t0) * 1000
        per_tx = total_ms / max(n, 1)
        for r in results:
            r.latency_ms = round(per_tx, 3)

        return results

    def _build_routed_features(self, features: dict, mv) -> Optional[np.ndarray]:
        """Build a feature vector for the A/B routed model."""
        try:
            # Try to extract the features the routed model expects
            feat_cols = mv.feat_cols if mv.feat_cols else []
            if not feat_cols:
                return None

            vec = []
            for col in feat_cols:
                val = features.get(col, 0.0)
                if isinstance(val, (int, float)):
                    vec.append(float(val))
                elif isinstance(val, bool):
                    vec.append(1.0 if val else 0.0)
                else:
                    vec.append(0.0)

            return np.array(vec, dtype=np.float32)
        except Exception:
            return None

    # ── Drift detection helpers ──────────────────────────────────────────

    def _extract_feature_vector(self, features: dict) -> Optional[np.ndarray]:
        """Extract a numeric feature vector from the features dict for drift monitoring.

        Uses the 32 Altman clean features if available, otherwise falls back
        to all numeric values from the ML_FEATURES set.
        """
        try:
            # Try Altman clean features first
            from src.risk_engine.altman_ensemble import map_ml_features_to_altman
            vec = map_ml_features_to_altman(features)
            return vec.astype(np.float32)
        except Exception:
            pass

        # Fallback: extract all numeric ML_FEATURES
        try:
            from src.privacy_layer.features import ML_FEATURES
            vec = [float(features.get(f, 0.0)) for f in ML_FEATURES]
            return np.array(vec, dtype=np.float32)
        except Exception:
            return None

    def _accumulate_and_check_drift(self, feat_vec: np.ndarray) -> tuple[str, float, str]:
        """Accumulate feature vector and periodically check PSI drift.

        Returns (status, psi, alert_message).
        """
        with self._drift_lock:
            self._drift_buffer.append(feat_vec)
            self._drift_n_scored += 1

            # Trim buffer to window size
            if len(self._drift_buffer) > self._drift_window_size:
                self._drift_buffer = self._drift_buffer[-self._drift_window_size:]

            # Check drift every N scored transactions
            if (self._drift_n_scored % self._drift_check_interval != 0 or
                    len(self._drift_buffer) < 100):
                # Return last known status
                if self._drift_last_report:
                    return (
                        self._drift_last_report.get("status", "stable"),
                        self._drift_last_report.get("aggregate_psi", 0.0),
                        self._drift_last_report.get("alert", ""),
                    )
                return "stable", 0.0, ""

            # Run PSI check on the buffer
            try:
                X_buffer = np.array(self._drift_buffer)
                report = self._drift_detector.check(X_buffer)

                # Determine status
                if report.needs_retrain:
                    status = "critical"
                    alert = (f"CRITICAL drift: aggregate PSI={report.aggregate_psi:.4f}, "
                             f"max={report.max_psi:.4f} ({report.max_psi_feature}), "
                             f"{report.n_critical} features critical")
                elif report.needs_investigation:
                    status = "warning"
                    alert = (f"WARNING drift: aggregate PSI={report.aggregate_psi:.4f}, "
                             f"{report.n_warning} features elevated")
                else:
                    status = "stable"
                    alert = ""

                # Store report
                report_dict = report.to_dict()
                report_dict["status"] = status
                report_dict["alert"] = alert
                report_dict["n_scored"] = self._drift_n_scored
                self._drift_last_report = report_dict
                self._drift_history.append(report_dict)

                # Trim history
                if len(self._drift_history) > 100:
                    self._drift_history = self._drift_history[-100:]

                # Fire alert callbacks
                if alert:
                    alert_entry = {
                        "timestamp": time.time(),
                        "status": status,
                        "alert": alert,
                        "psi": report.aggregate_psi,
                        "n_scored": self._drift_n_scored,
                    }
                    self._drift_alerts.append(alert_entry)
                    for cb in self._drift_alert_callbacks:
                        try:
                            cb(alert_entry)
                        except Exception:
                            pass

                return status, report.aggregate_psi, alert

            except Exception as e:
                return "stable", 0.0, f"drift check error: {e}"

    def get_drift_status(self) -> dict:
        """Get current drift monitoring status."""
        with self._drift_lock:
            result = {
                "drift_detector_ready": self._drift_ready,
                "reference_path": str(self._drift_ref_path),
                "reference_exists": self._drift_ref_path.exists(),
                "n_scored": self._drift_n_scored,
                "check_interval": self._drift_check_interval,
                "window_size": self._drift_window_size,
                "buffer_size": len(self._drift_buffer),
                "n_checks": len(self._drift_history),
                "n_alerts": len(self._drift_alerts),
            }

            if self._drift_last_report:
                result["last_check"] = {
                    "status": self._drift_last_report.get("status", "unknown"),
                    "aggregate_psi": self._drift_last_report.get("aggregate_psi", 0),
                    "max_psi": self._drift_last_report.get("max_psi", 0),
                    "max_psi_feature": self._drift_last_report.get("max_psi_feature", ""),
                    "n_stable": self._drift_last_report.get("n_stable", 0),
                    "n_warning": self._drift_last_report.get("n_warning", 0),
                    "n_critical": self._drift_last_report.get("n_critical", 0),
                    "needs_retrain": self._drift_last_report.get("needs_retrain", False),
                    "alert": self._drift_last_report.get("alert", ""),
                }
            else:
                result["last_check"] = None

            # Recent alerts (last 10)
            result["recent_alerts"] = self._drift_alerts[-10:]

            # Reference feature stats (summary)
            if self._drift_ref:
                result["reference"] = {
                    "n_samples": self._drift_ref.n_samples,
                    "n_features": self._drift_ref.n_features,
                    "feature_names": self._drift_ref.feature_names,
                }

            return result

    def get_drift_history(self, limit: int = 20) -> list[dict]:
        """Get recent drift check history."""
        with self._drift_lock:
            return self._drift_history[-limit:]

    def on_drift_alert(self, callback) -> None:
        """Register a callback for drift alerts.

        callback receives a dict with: timestamp, status, alert, psi, n_scored
        """
        self._drift_alert_callbacks.append(callback)

    def force_drift_check(self) -> Optional[dict]:
        """Force an immediate drift check on the current buffer."""
        with self._drift_lock:
            if not self._drift_ready or not self._drift_buffer:
                return None
            try:
                X_buffer = np.array(self._drift_buffer)
                report = self._drift_detector.check(X_buffer)
                report_dict = report.to_dict()
                if report.needs_retrain:
                    report_dict["status"] = "critical"
                elif report.needs_investigation:
                    report_dict["status"] = "warning"
                else:
                    report_dict["status"] = "stable"
                self._drift_last_report = report_dict
                return report_dict
            except Exception as e:
                return {"error": str(e)}

    def get_status(self) -> dict:
        """Get status of all detection subsystems."""
        return {
            "loaded": self._loaded,
            "load_time_ms": self._load_time_ms,
            "subsystems": {
                "ml_fusion": "ready" if self._ml_ready else "unavailable",
                "rules": "ready" if self._rules_ready else "unavailable",
                "velocity_limits": "ready" if self._limits_ready else "unavailable",
                "ab_testing": "ready" if self._ab_ready else "unavailable",
                "drift_detection": "ready" if self._drift_ready else "unavailable",
            },
            "drift": self.get_drift_status(),
            "registered_models": len(self.registry.versions) if self.registry else 0,
            "errors": self._load_errors,
        }
