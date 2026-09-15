"""Comprehensive drift monitoring orchestrator (Phase 40).

Ties together:
  - Feature drift (PSI via existing psi.py / drift_detector.py)
  - Schema drift (schema_drift.py)
  - Feature-availability drift (schema_drift.py)
  - Prediction/risk-score drift (prediction_drift.py)
  - Alert lifecycle (alert_manager.py)
  - Safeguards (safeguards.py)
  - Label awareness (label_awareness.py)
  - Baseline governance (baseline_governance.py)

This module is the single entry point for monitoring status. It does NOT
replace the existing runtime drift detector (risk_engine/drift_detector.py)
which runs in-process on every evaluate() call. Instead, it provides a
higher-level orchestration layer for dashboarding and governance.

DRIFT DETECTED != AUTOMATIC RETRAINING
DRIFT DETECTED != AUTOMATIC PROMOTION
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.monitoring.schema_drift import (
    SchemaDriftReport,
    SchemaStatus,
    FeatureAvailability,
    check_schema,
    check_feature_availability,
)
from src.monitoring.prediction_drift import (
    PredictionDriftReport,
    check_prediction_drift,
)
from src.monitoring.alert_manager import (
    AlertManager,
    AlertState,
    Alert,
)
from src.monitoring.safeguards import PromotionSafeguard
from src.monitoring.label_awareness import (
    LabelAvailability,
    OutcomeMonitoringReport,
    check_label_availability,
    check_outcome_drift,
)
from src.monitoring.baseline_governance import (
    BaselineMetadata,
    BaselineStatus,
)


@dataclass
class MonitoringStatus:
    """Complete monitoring status for a single check run."""
    timestamp: float
    overall_status: str  # "healthy", "degraded", "critical", "monitoring_unavailable"
    feature_drift_state: str  # from existing PSI detector
    feature_drift_max_psi: float
    schema_drift: SchemaDriftReport | None
    prediction_drift: PredictionDriftReport | None
    label_availability: LabelAvailability | None
    outcome_report: OutcomeMonitoringReport | None
    alert_state: str  # from AlertManager
    active_alerts: int
    n_features_monitored: int
    n_features_drifting: int
    n_features_unavailable: int
    n_schema_issues: int
    baseline_loaded: bool
    model_version: str
    monitoring_healthy: bool = True
    detail: str = ""

    def to_dict(self) -> dict:
        result = {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "feature_drift": {
                "state": self.feature_drift_state,
                "max_psi": round(self.feature_drift_max_psi, 6),
            },
            "schema_drift": self.schema_drift.to_dict() if self.schema_drift else None,
            "prediction_drift": self.prediction_drift.to_dict() if self.prediction_drift else None,
            "label_availability": self.label_availability.to_dict() if self.label_availability else None,
            "outcome_monitoring": self.outcome_report.to_dict() if self.outcome_report else None,
            "alert_state": self.alert_state,
            "active_alerts": self.active_alerts,
            "summary": {
                "n_features_monitored": self.n_features_monitored,
                "n_features_drifting": self.n_features_drifting,
                "n_features_unavailable": self.n_features_unavailable,
                "n_schema_issues": self.n_schema_issues,
                "baseline_loaded": self.baseline_loaded,
                "model_version": self.model_version,
                "monitoring_healthy": self.monitoring_healthy,
            },
        }
        if self.detail:
            result["detail"] = self.detail
        return result


class DriftMonitor:
    """Comprehensive drift monitoring orchestrator.

    Usage:
        monitor = DriftMonitor(
            expected_features=ML_FEATURES,
            baseline_path=Path("data/drift_baseline.json"),
        )

        # On each monitoring window:
        status = monitor.check_window(
            feature_data={...},
            scores=np.array([...]),
            labels=[...] if available else None,
        )

        # Get status for dashboard:
        status_json = status.to_dict()

        # Check if retrain is eligible:
        safeguard = monitor.check_retrain_eligible()

        # Check if promotion is eligible:
        promotion_check = monitor.check_promotion_eligible(...)
    """

    def __init__(
        self,
        expected_features: list[str],
        baseline_path: Path | None = None,
        model_version: str = "",
        alert_manager: AlertManager | None = None,
    ):
        self.expected_features = expected_features
        self.baseline_path = baseline_path
        self.model_version = model_version
        self.alert_manager = alert_manager or AlertManager()
        self.safeguard = PromotionSafeguard()

        # Baseline metadata
        self.baseline_metadata: BaselineMetadata | None = None
        if baseline_path and baseline_path.exists():
            meta_path = baseline_path.parent / f"{baseline_path.stem}_metadata.json"
            if meta_path.exists():
                try:
                    self.baseline_metadata = BaselineMetadata.load(meta_path)
                except Exception:
                    pass

        # Reference data for prediction drift
        self._reference_scores: np.ndarray | None = None
        self._reference_band_distribution: dict[str, float] | None = None

        # History
        self._status_history: list[MonitoringStatus] = []
        self._last_check_ts: float = 0.0

    def set_reference_scores(self, scores: np.ndarray) -> None:
        """Set reference score distribution for prediction drift comparison."""
        self._reference_scores = scores.copy()

    def check_window(
        self,
        feature_data: dict[str, list] | None = None,
        scores: np.ndarray | None = None,
        labels: list[int | None] | None = None,
        reference_fraud_rate: float | None = None,
        feature_drift_state: str = "normal",
        feature_drift_max_psi: float = 0.0,
        feature_drift_alerting: list[str] | None = None,
        feature_psi_values: dict[str, float] | None = None,
    ) -> MonitoringStatus:
        """Run a comprehensive monitoring check on the current window.

        Args:
            feature_data: dict mapping feature_name -> list of values
            scores: array of risk scores from this window
            labels: list of verified labels (1=fraud, 0=legit, None=unknown)
            reference_fraud_rate: reference fraud rate for outcome comparison
            feature_drift_state: from existing PSI detector ("normal"/"warning"/"critical")
            feature_drift_max_psi: max PSI from existing detector
            feature_drift_alerting: list of features currently alerting on PSI
            feature_psi_values: per-feature PSI values
        """
        now = time.time()

        # 1. Schema/feature-availability drift
        schema_report = None
        n_schema_issues = 0
        n_unavailable = 0
        if feature_data:
            schema_report = check_feature_availability(
                data=feature_data,
                expected_features=self.expected_features,
            )
            n_schema_issues = (
                schema_report.n_missing
                + schema_report.n_unexpected
                + schema_report.n_type_mismatches
                + schema_report.n_stale
            )
            n_unavailable = schema_report.n_unavailable

        # 2. Prediction drift
        prediction_report = None
        if scores is not None and len(scores) > 0:
            prediction_report = check_prediction_drift(
                current_scores=scores,
                reference_scores=self._reference_scores,
                reference_band_distribution=self._reference_band_distribution,
            )

        # 3. Label awareness
        label_info = None
        outcome_report = None
        if labels is not None:
            label_info = check_label_availability(labels)
            if label_info.available:
                outcome_report = check_outcome_drift(
                    current_labels=labels,
                    reference_fraud_rate=reference_fraud_rate,
                )

        # 4. Determine overall status
        overall = "healthy"
        detail_parts = []

        # Feature drift
        if feature_drift_state == "critical":
            overall = "critical"
            detail_parts.append("feature drift CRITICAL")
        elif feature_drift_state == "warning" and overall != "critical":
            overall = "degraded"
            detail_parts.append("feature drift WARNING")

        # Schema drift
        if schema_report and schema_report.overall_status == SchemaStatus.CRITICAL:
            overall = "critical"
            detail_parts.append(f"schema CRITICAL: {schema_report.n_missing} missing, {schema_report.n_unavailable} unavailable")
        elif schema_report and schema_report.overall_status == SchemaStatus.WARNING:
            if overall != "critical":
                overall = "degraded"
            detail_parts.append(f"schema WARNING: {schema_report.n_unexpected} unexpected, {schema_report.n_degraded} degraded")

        # Prediction drift
        if prediction_report and prediction_report.overall_drift == "critical":
            overall = "critical"
            detail_parts.append("prediction drift CRITICAL")
        elif prediction_report and prediction_report.overall_drift == "warning":
            if overall != "critical":
                overall = "degraded"
            detail_parts.append("prediction drift WARNING")

        # Label outcome drift
        if outcome_report and outcome_report.overall_status == "critical":
            overall = "critical"
            detail_parts.append("outcome drift CRITICAL")
        elif outcome_report and outcome_report.overall_status == "warning":
            if overall != "critical":
                overall = "degraded"
            detail_parts.append("outcome drift WARNING")

        # 5. Alert lifecycle
        n_features_drifting = len(feature_drift_alerting) if feature_drift_alerting else 0
        worst_level = "alert" if feature_drift_state == "critical" else ("warn" if feature_drift_state == "warning" else "ok")

        # Also consider schema and prediction drift for alerting
        if schema_report and schema_report.overall_status == SchemaStatus.CRITICAL:
            worst_level = "alert"
        elif schema_report and schema_report.overall_status == SchemaStatus.WARNING and worst_level != "alert":
            worst_level = "warn"

        if prediction_report and prediction_report.overall_drift == "critical":
            worst_level = "alert"
        elif prediction_report and prediction_report.overall_drift == "warning" and worst_level != "alert":
            worst_level = "warn"

        new_alerts = self.alert_manager.record_window(
            worst_level=worst_level,
            features_alerting=feature_drift_alerting,
            source="feature_drift",
            model_version=self.model_version,
            window_start=now - 3600,  # placeholder
            window_end=now,
            n_observations=len(feature_data[list(feature_data.keys())[0]]) if feature_data else 0,
            feature_psi_values=feature_psi_values,
        )

        # 6. Build status
        status = MonitoringStatus(
            timestamp=now,
            overall_status=overall,
            feature_drift_state=feature_drift_state,
            feature_drift_max_psi=feature_drift_max_psi,
            schema_drift=schema_report,
            prediction_drift=prediction_report,
            label_availability=label_info,
            outcome_report=outcome_report,
            alert_state=self.alert_manager.state.value,
            active_alerts=len(self.alert_manager.active_alerts),
            n_features_monitored=len(self.expected_features),
            n_features_drifting=n_features_drifting,
            n_features_unavailable=n_unavailable,
            n_schema_issues=n_schema_issues,
            baseline_loaded=self.baseline_metadata is not None,
            model_version=self.model_version,
            monitoring_healthy=True,
            detail="; ".join(detail_parts) if detail_parts else "all monitoring nominal",
        )

        self._status_history.append(status)
        self._last_check_ts = now

        return status

    def check_retrain_eligible(self) -> dict:
        """Check if retraining is eligible (does NOT auto-authorize)."""
        latest_status = self._status_history[-1] if self._status_history else None
        drift_state = latest_status.feature_drift_state if latest_status else "unknown"
        has_critical = len(self.alert_manager.active_alerts) > 0
        healthy = latest_status.monitoring_healthy if latest_status else False

        results = self.safeguard.check_retrain_eligible(
            drift_state=drift_state,
            has_active_critical_alerts=has_critical,
            monitoring_healthy=healthy,
        )
        return self.safeguard.summarize(results)

    def check_promotion_eligible(self, **gate_kwargs) -> dict:
        """Check if promotion is eligible (does NOT auto-authorize)."""
        latest_status = self._status_history[-1] if self._status_history else None
        drift_state = latest_status.feature_drift_state if latest_status else "unknown"
        has_critical = len(self.alert_manager.active_alerts) > 0

        results = self.safeguard.check_promotion_eligible(
            drift_state=drift_state,
            has_active_critical_alerts=has_critical,
            **gate_kwargs,
        )
        return self.safeguard.summarize(results)

    def status(self) -> dict:
        """Complete monitoring status for dashboard/API."""
        latest = self._status_history[-1].to_dict() if self._status_history else {}
        latest["alert_manager"] = self.alert_manager.status()
        latest["baseline_metadata"] = self.baseline_metadata.to_dict() if self.baseline_metadata else None
        latest["history_size"] = len(self._status_history)
        latest["last_check_ts"] = self._last_check_ts

        # Safety disclaimer
        latest["safeguards"] = {
            "drift_does_not_mean_retrain": True,
            "drift_does_not_mean_promote": True,
            "real_world_validation": "BLOCKED",
        }

        return latest

    def save_status(self, path: Path) -> None:
        """Save current status to JSON for persistence."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.status(), indent=2, default=str), encoding="utf-8")
