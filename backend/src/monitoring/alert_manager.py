"""Alert lifecycle manager with hysteresis, dedup, and cooldown.

Prevents alert flapping by requiring consecutive windows to confirm
a state transition before emitting an alert. Implements:

  - Alert fingerprinting (dedup identical alerts)
  - Consecutive-window confirmation (hysteresis)
  - Cooldown period after alert emission
  - Recovery tracking (CRITICAL -> NORMAL requires sustained improvement)
  - Alert storm protection (aggregate systemic issues)

Alert states:
  NORMAL -> WATCH (consecutive warnings >= watch_threshold)
         -> WARNING (consecutive warnings >= warning_threshold)
         -> CRITICAL (consecutive criticals >= critical_threshold)

Recovery:
  CRITICAL -> RECOVERING (consecutive ok windows >= recovery_threshold)
           -> NORMAL (sustained ok)

A single noisy window does NOT trigger an alert.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlertState(str, Enum):
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"
    RECOVERING = "recovering"


class AlertSeverity(str, Enum):
    INFO = "info"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A single alert event."""
    alert_id: str
    fingerprint: str
    severity: AlertSeverity
    source: str  # "feature_drift", "schema", "prediction_drift", "data_quality"
    feature: str | None = None
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    model_version: str = ""
    baseline_id: str = ""
    window_start: float = 0.0
    window_end: float = 0.0
    n_observations: int = 0
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "fingerprint": self.fingerprint,
            "severity": self.severity.value,
            "source": self.source,
            "feature": self.feature,
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 6),
            "threshold": round(self.threshold, 6),
            "message": self.message,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "baseline_id": self.baseline_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "n_observations": self.n_observations,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class AlertManager:
    """Manages alert lifecycle with hysteresis and dedup.

    Parameters:
        watch_consecutive: consecutive warning-level events before WATCH
        warning_consecutive: consecutive warning-level events before WARNING
        critical_consecutive: consecutive critical-level events before CRITICAL
        recovery_consecutive: consecutive normal events to recover from CRITICAL
        cooldown_seconds: minimum seconds between identical alert emissions
        max_active_alerts: max active alerts before storm protection kicks in
        max_alert_history: max alerts to retain in history
    """
    watch_consecutive: int = 2
    warning_consecutive: int = 3
    critical_consecutive: int = 2
    recovery_consecutive: int = 3
    cooldown_seconds: float = 300.0  # 5 minutes
    max_active_alerts: int = 50
    max_alert_history: int = 500

    # Internal state
    _consecutive_warnings: int = field(default=0, init=False)
    _consecutive_criticals: int = field(default=0, init=False)
    _consecutive_normals: int = field(default=0, init=False)
    _current_state: AlertState = field(default=AlertState.NORMAL, init=False)
    _active_alerts: dict[str, Alert] = field(default_factory=dict, init=False)
    _alert_history: list[Alert] = field(default_factory=list, init=False)
    _fingerprint_last_emitted: dict[str, float] = field(default_factory=dict, init=False)
    _state_transitions: list[dict] = field(default_factory=list, init=False)

    def _make_fingerprint(self, source: str, feature: str | None, metric_name: str) -> str:
        """Generate a dedup fingerprint from alert identity."""
        key = f"{source}:{feature or 'global'}:{metric_name}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _make_alert_id(self) -> str:
        """Generate a unique alert ID."""
        ts = int(time.time() * 1000)
        return f"alert-{ts}-{len(self._alert_history)}"

    def _can_emit(self, fingerprint: str) -> bool:
        """Check if cooldown has elapsed for this fingerprint."""
        last = self._fingerprint_last_emitted.get(fingerprint, 0)
        return (time.time() - last) >= self.cooldown_seconds

    def _record_emission(self, fingerprint: str) -> None:
        self._fingerprint_last_emitted[fingerprint] = time.time()

    def _record_state_transition(self, from_state: AlertState, to_state: AlertState) -> None:
        self._state_transitions.append({
            "from": from_state.value,
            "to": to_state.value,
            "timestamp": time.time(),
        })

    def record_window(
        self,
        worst_level: str,  # "ok", "warn", "alert"
        features_alerting: list[str] | None = None,
        source: str = "feature_drift",
        model_version: str = "",
        baseline_id: str = "",
        window_start: float = 0.0,
        window_end: float = 0.0,
        n_observations: int = 0,
        feature_psi_values: dict[str, float] | None = None,
        warn_threshold: float = 0.10,
        alert_threshold: float = 0.25,
    ) -> list[Alert]:
        """Record a monitoring window result and manage alert state.

        Returns a list of newly emitted alerts (may be empty).
        """
        new_alerts: list[Alert] = []
        prev_state = self._current_state

        if worst_level == "alert":
            self._consecutive_criticals += 1
            self._consecutive_warnings = 0
            self._consecutive_normals = 0
        elif worst_level == "warn":
            self._consecutive_warnings += 1
            self._consecutive_criticals = 0
            self._consecutive_normals = 0
        else:
            self._consecutive_normals += 1
            self._consecutive_criticals = 0
            self._consecutive_warnings = 0

        # State transitions with hysteresis
        if self._current_state == AlertState.NORMAL:
            if self._consecutive_criticals >= self.critical_consecutive:
                self._current_state = AlertState.CRITICAL
            elif self._consecutive_warnings >= self.warning_consecutive:
                self._current_state = AlertState.WARNING
            elif self._consecutive_warnings >= self.watch_consecutive:
                self._current_state = AlertState.WATCH

        elif self._current_state == AlertState.WATCH:
            if self._consecutive_criticals >= self.critical_consecutive:
                self._current_state = AlertState.CRITICAL
            elif self._consecutive_warnings >= self.warning_consecutive:
                self._current_state = AlertState.WARNING
            elif self._consecutive_normals >= 1:
                self._current_state = AlertState.NORMAL
                self._consecutive_warnings = 0

        elif self._current_state == AlertState.WARNING:
            if self._consecutive_criticals >= self.critical_consecutive:
                self._current_state = AlertState.CRITICAL
            elif self._consecutive_normals >= 1:
                self._current_state = AlertState.NORMAL
                self._consecutive_warnings = 0

        elif self._current_state == AlertState.CRITICAL:
            if self._consecutive_normals >= self.recovery_consecutive:
                self._current_state = AlertState.RECOVERING
                self._consecutive_normals = 0
            # stay CRITICAL if still alerting

        elif self._current_state == AlertState.RECOVERING:
            if self._consecutive_criticals >= 1:
                self._current_state = AlertState.CRITICAL
                self._consecutive_normals = 0
            elif self._consecutive_normals >= self.recovery_consecutive:
                self._current_state = AlertState.NORMAL
                # Resolve all active alerts
                for alert in self._active_alerts.values():
                    alert.resolved = True
                self._active_alerts.clear()
                self._consecutive_normals = 0

        # Record state transition
        if self._current_state != prev_state:
            self._record_state_transition(prev_state, self._current_state)

        # Emit alerts on state escalation
        if self._current_state != prev_state and self._current_state in (
            AlertState.WARNING, AlertState.CRITICAL
        ):
            if features_alerting:
                # Emit per-feature alerts, but storm-protect by grouping
                if len(features_alerting) > 5:
                    # Aggregate into a single systemic alert
                    fp = self._make_fingerprint(source, None, "systemic_drift")
                    if self._can_emit(fp):
                        alert = Alert(
                            alert_id=self._make_alert_id(),
                            fingerprint=fp,
                            severity=AlertSeverity.CRITICAL if self._current_state == AlertState.CRITICAL else AlertSeverity.WARNING,
                            source=source,
                            metric_name="systemic_drift",
                            metric_value=float(len(features_alerting)),
                            threshold=5.0,
                            message=f"systemic drift affecting {len(features_alerting)} features",
                            model_version=model_version,
                            baseline_id=baseline_id,
                            window_start=window_start,
                            window_end=window_end,
                            n_observations=n_observations,
                        )
                        new_alerts.append(alert)
                        self._active_alerts[fp] = alert
                        self._record_emission(fp)
                else:
                    for feat in features_alerting:
                        psi_val = (feature_psi_values or {}).get(feat, 0.0)
                        fp = self._make_fingerprint(source, feat, "psi")
                        if self._can_emit(fp):
                            alert = Alert(
                                alert_id=self._make_alert_id(),
                                fingerprint=fp,
                                severity=AlertSeverity.CRITICAL if self._current_state == AlertState.CRITICAL else AlertSeverity.WARNING,
                                source=source,
                                feature=feat,
                                metric_name="psi",
                                metric_value=psi_val,
                                threshold=alert_threshold if self._current_state == AlertState.CRITICAL else warn_threshold,
                                message=f"feature {feat} PSI={psi_val:.4f}",
                                model_version=model_version,
                                baseline_id=baseline_id,
                                window_start=window_start,
                                window_end=window_end,
                                n_observations=n_observations,
                            )
                            new_alerts.append(alert)
                            self._active_alerts[fp] = alert
                            self._record_emission(fp)
            else:
                # Generic alert without feature specificity
                fp = self._make_fingerprint(source, None, "general")
                if self._can_emit(fp):
                    alert = Alert(
                        alert_id=self._make_alert_id(),
                        fingerprint=fp,
                        severity=AlertSeverity.CRITICAL if self._current_state == AlertState.CRITICAL else AlertSeverity.WARNING,
                        source=source,
                        metric_name="overall",
                        message=f"monitoring state escalated to {self._current_state.value}",
                        model_version=model_version,
                        baseline_id=baseline_id,
                        window_start=window_start,
                        window_end=window_end,
                        n_observations=n_observations,
                    )
                    new_alerts.append(alert)
                    self._active_alerts[fp] = alert
                    self._record_emission(fp)

        # Trim history
        self._alert_history.extend(new_alerts)
        if len(self._alert_history) > self.max_alert_history:
            self._alert_history = self._alert_history[-self.max_alert_history:]

        return new_alerts

    @property
    def state(self) -> AlertState:
        return self._current_state

    @property
    def active_alerts(self) -> list[Alert]:
        return [a for a in self._active_alerts.values() if not a.resolved]

    @property
    def alert_history(self) -> list[Alert]:
        return list(self._alert_history)

    @property
    def state_transitions(self) -> list[dict]:
        return list(self._state_transitions)

    def acknowledge(self, alert_id: str) -> bool:
        for alert in self._active_alerts.values():
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def status(self) -> dict:
        return {
            "state": self._current_state.value,
            "active_alerts": len(self.active_alerts),
            "total_alerts_emitted": len(self._alert_history),
            "consecutive_warnings": self._consecutive_warnings,
            "consecutive_criticals": self._consecutive_criticals,
            "consecutive_normals": self._consecutive_normals,
            "state_transitions": len(self._state_transitions),
        }
