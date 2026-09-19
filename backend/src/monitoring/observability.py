"""Phase 70: Production observability, monitoring & security event detection.

Provides structured logging, request correlation, operational metrics,
model telemetry, security event detection, alerting, and health/readiness
semantics for the PS-14 production risk path.

Architecture:
    A. Application logs       — machine-readable structured events
    B. Operational metrics    — counters, latencies, distributions
    C. Security events        — authentication, tampering, anomalies
    D. Fraud decision/audit   — DB-4 hash chain (existing)
    E. Model telemetry        — score distributions, release identity
    F. Incident/forensic      — forensic_release_history.py (existing)

Privacy:
    NEVER log: raw PII, auth secrets, internal tokens, full feature payloads.
    Use hashes/correlation IDs for traceability.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import time
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# ── Security Event Types ─────────────────────────────────────────────────

class SecurityEventType(str, Enum):
    AUTH_FAILURE = "AUTH_FAILURE"
    REPEATED_AUTH_FAILURE = "REPEATED_AUTH_FAILURE"
    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    REPEATED_MALFORMED = "REPEATED_MALFORMED"
    SCHEMA_REJECTION = "SCHEMA_REJECTION"
    FEATURE_ENFORCEMENT_BLOCK = "FEATURE_ENFORCEMENT_BLOCK"
    RELEASE_INTEGRITY_FAILURE = "RELEASE_INTEGRITY_FAILURE"
    MODEL_ARTIFACT_TAMPER = "MODEL_ARTIFACT_TAMPER"
    AUDIT_CHAIN_FAILURE = "AUDIT_CHAIN_FAILURE"
    RUNTIME_STATE_UNEXPECTED = "RUNTIME_STATE_UNEXPECTED"
    IDEMPOTENCY_ABUSE = "IDEMPOTENCY_ABUSE"
    REQUEST_BURST = "REQUEST_BURST"
    UNEXPECTED_ADMIN_OP = "UNEXPECTED_ADMIN_OP"
    DEGRADED_MODE_ENTRY = "DEGRADED_MODE_ENTRY"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class SecuritySeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ── Security Event Record ────────────────────────────────────────────────

@dataclass
class SecurityEvent:
    """A single security-relevant event."""
    event_id: str
    timestamp: float
    event_type: str  # SecurityEventType value
    severity: str  # SecuritySeverity value
    service: str
    correlation_id: str = ""
    source_ip: str = ""
    attempted_operation: str = ""
    outcome: str = ""
    related_event_ids: list[str] = field(default_factory=list)
    containment_status: str = "OPEN"
    evidence_hash: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    event_hash: str = ""

    def compute_hash(self, previous_hash: str = "") -> str:
        """Tamper-evident hash binding to chain."""
        payload = json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "service": self.service,
            "correlation_id": self.correlation_id,
            "attempted_operation": self.attempted_operation,
            "outcome": self.outcome,
            "containment_status": self.containment_status,
            "previous_hash": previous_hash,
        }, sort_keys=True)
        self.event_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.event_hash

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type
        d["severity"] = self.severity
        return d


# ── Structured Log Entry ─────────────────────────────────────────────────

@dataclass
class StructuredLogEntry:
    """Machine-readable structured log entry."""
    timestamp: float
    service: str
    severity: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    correlation_id: str
    endpoint: str
    operation: str
    outcome: str
    latency_ms: float = 0.0
    model_id: str = ""
    release_id: str = ""
    feature_version: str = ""
    runtime_state: str = ""
    error_category: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "ts": round(self.timestamp, 6),
            "svc": self.service,
            "sev": self.severity,
            "cid": self.correlation_id,
            "ep": self.endpoint,
            "op": self.operation,
            "out": self.outcome,
            "lat_ms": round(self.latency_ms, 3),
            "mid": self.model_id,
            "rid": self.release_id,
            "fv": self.feature_version,
            "rt": self.runtime_state,
            "err": self.error_category,
        }, separators=(",", ":"))


# ── Operational Metrics ──────────────────────────────────────────────────

@dataclass
class OperationalMetrics:
    """Collects and exposes operational counters and distributions."""
    # Requests
    total_requests: int = 0
    successful_requests: int = 0
    rejected_requests: int = 0
    error_count: int = 0

    # Latency (in ms)
    _latency_buffer: deque = field(default_factory=lambda: deque(maxlen=10000))
    _inference_buffer: deque = field(default_factory=lambda: deque(maxlen=10000))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Model
    prediction_count: int = 0
    score_band_low: int = 0
    score_band_medium: int = 0
    score_band_high: int = 0
    decision_allow: int = 0
    decision_review: int = 0
    decision_block: int = 0

    # Data quality
    schema_rejections: int = 0
    feature_enforcement_blocks: int = 0
    freshness_failures: int = 0
    drift_detections: int = 0

    # Idempotency
    new_events: int = 0
    replayed_events: int = 0
    duplicate_attempts: int = 0

    # Security
    auth_failures: int = 0
    malformed_requests: int = 0

    # System
    degraded_mode_entries: int = 0

    def record_request(self, latency_ms: float, success: bool) -> None:
        with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.rejected_requests += 1
            self._latency_buffer.append(latency_ms)

    def record_error(self) -> None:
        with self._lock:
            self.error_count += 1

    def record_prediction(self, ml_score: float, risk_band: str, decision: str) -> None:
        with self._lock:
            self.prediction_count += 1
            if risk_band == "low":
                self.score_band_low += 1
            elif risk_band == "medium":
                self.score_band_medium += 1
            elif risk_band == "high":
                self.score_band_high += 1
            decision_key = decision.lower()
            if decision_key == "allow":
                self.decision_allow += 1
            elif decision_key == "review":
                self.decision_review += 1
            elif decision_key == "block":
                self.decision_block += 1

    def record_inference_latency(self, ms: float) -> None:
        with self._lock:
            self._inference_buffer.append(ms)

    def record_idempotency(self, is_new: bool) -> None:
        with self._lock:
            if is_new:
                self.new_events += 1
            else:
                self.replayed_events += 1

    def record_schema_rejection(self) -> None:
        with self._lock:
            self.schema_rejections += 1

    def record_feature_block(self) -> None:
        with self._lock:
            self.feature_enforcement_blocks += 1

    def record_auth_failure(self) -> None:
        with self._lock:
            self.auth_failures += 1

    def record_malformed(self) -> None:
        with self._lock:
            self.malformed_requests += 1

    def latency_percentiles(self) -> dict[str, float]:
        with self._lock:
            if not self._latency_buffer:
                return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
            data = sorted(self._latency_buffer)
            n = len(data)
            return {
                "count": n,
                "p50": round(data[n // 2], 3),
                "p95": round(data[int(n * 0.95)], 3),
                "p99": round(data[int(n * 0.99)], 3),
                "max": round(data[-1], 3),
                "avg": round(sum(data) / n, 3),
            }

    def inference_latency_percentiles(self) -> dict[str, float]:
        with self._lock:
            if not self._inference_buffer:
                return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
            data = sorted(self._inference_buffer)
            n = len(data)
            return {
                "count": n,
                "p50": round(data[n // 2], 3),
                "p95": round(data[int(n * 0.95)], 3),
                "p99": round(data[int(n * 0.99)], 3),
                "max": round(data[-1], 3),
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "rejected": self.rejected_requests,
                "errors": self.error_count,
            },
            "latency": self.latency_percentiles(),
            "inference_latency": self.inference_latency_percentiles(),
            "model": {
                "predictions": self.prediction_count,
                "band_low": self.score_band_low,
                "band_medium": self.score_band_medium,
                "band_high": self.score_band_high,
                "decision_allow": self.decision_allow,
                "decision_review": self.decision_review,
                "decision_block": self.decision_block,
            },
            "data_quality": {
                "schema_rejections": self.schema_rejections,
                "feature_enforcement_blocks": self.feature_enforcement_blocks,
                "freshness_failures": self.freshness_failures,
                "drift_detections": self.drift_detections,
            },
            "idempotency": {
                "new_events": self.new_events,
                "replayed_events": self.replayed_events,
                "duplicate_attempts": self.duplicate_attempts,
            },
            "security": {
                "auth_failures": self.auth_failures,
                "malformed_requests": self.malformed_requests,
            },
            "system": {
                "degraded_mode_entries": self.degraded_mode_entries,
            },
        }


# ── Model Telemetry ──────────────────────────────────────────────────────

@dataclass
class ModelTelemetry:
    """Tracks model behavior without exposing sensitive transaction info."""
    model_id: str = ""
    release_id: str = ""
    feature_version: str = ""

    # Distribution tracking
    _score_buffer: deque = field(default_factory=lambda: deque(maxlen=10000))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    prediction_count: int = 0
    feature_validation_failures: int = 0
    drift_signals: int = 0
    degradation_events: int = 0

    def record_prediction(
        self, ml_score: float, risk_band: str, decision: str,
        model_variance: float = 0.0, model_disagreement: float = 0.0,
    ) -> None:
        with self._lock:
            self.prediction_count += 1
            self._score_buffer.append({
                "score": ml_score,
                "band": risk_band,
                "decision": decision,
                "variance": model_variance,
                "disagreement": model_disagreement,
            })

    def score_distribution(self) -> dict[str, Any]:
        with self._lock:
            if not self._score_buffer:
                return {"count": 0}
            scores = [e["score"] for e in self._score_buffer]
            variances = [e["variance"] for e in self._score_buffer]
            disagreements = [e["disagreement"] for e in self._score_buffer]
            return {
                "count": len(scores),
                "score_min": round(min(scores), 6),
                "score_max": round(max(scores), 6),
                "score_mean": round(sum(scores) / len(scores), 6),
                "avg_variance": round(sum(variances) / len(variances), 6),
                "avg_disagreement": round(sum(disagreements) / len(disagreements), 6),
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "release_id": self.release_id,
            "feature_version": self.feature_version,
            "prediction_count": self.prediction_count,
            "score_distribution": self.score_distribution(),
            "feature_validation_failures": self.feature_validation_failures,
            "drift_signals": self.drift_signals,
            "degradation_events": self.degradation_events,
        }


# ── Alert Conditions ─────────────────────────────────────────────────────

@dataclass
class AlertCondition:
    """A deterministic alert condition."""
    condition_id: str
    name: str
    description: str
    metric_source: str
    metric_name: str
    threshold: float
    window_seconds: float
    severity: str  # SecuritySeverity value
    enabled: bool = True


# Default alert conditions (configurable example thresholds)
DEFAULT_ALERT_CONDITIONS: list[AlertCondition] = [
    AlertCondition(
        condition_id="auth-fail-5-min",
        name="Repeated authentication failures",
        description="5+ auth failures within 5 minutes",
        metric_source="security",
        metric_name="auth_failures",
        threshold=5,
        window_seconds=300,
        severity=SecuritySeverity.HIGH.value,
    ),
    AlertCondition(
        condition_id="malformed-10-min",
        name="Repeated malformed requests",
        description="10+ malformed requests within 10 minutes",
        metric_source="security",
        metric_name="malformed_requests",
        threshold=10,
        window_seconds=600,
        severity=SecuritySeverity.MEDIUM.value,
    ),
    AlertCondition(
        condition_id="error-rate-5pct",
        name="High error rate",
        description="Error rate exceeds 5% of requests",
        metric_source="requests",
        metric_name="error_rate_pct",
        threshold=5.0,
        window_seconds=300,
        severity=SecuritySeverity.HIGH.value,
    ),
    AlertCondition(
        condition_id="degraded-mode",
        name="Service entered degraded mode",
        description="Runtime state changed to degraded/DRIFTED",
        metric_source="system",
        metric_name="degraded_mode_entries",
        threshold=1,
        window_seconds=60,
        severity=SecuritySeverity.CRITICAL.value,
    ),
    AlertCondition(
        condition_id="feature-block-burst",
        name="Feature enforcement burst",
        description="20+ feature enforcement blocks within 5 minutes",
        metric_source="data_quality",
        metric_name="feature_enforcement_blocks",
        threshold=20,
        window_seconds=300,
        severity=SecuritySeverity.MEDIUM.value,
    ),
    AlertCondition(
        condition_id="schema-reject-burst",
        name="Schema rejection burst",
        description="10+ schema rejections within 5 minutes",
        metric_source="data_quality",
        metric_name="schema_rejections",
        threshold=10,
        window_seconds=300,
        severity=SecuritySeverity.MEDIUM.value,
    ),
    AlertCondition(
        condition_id="drift-detected",
        name="Drift detected",
        description="Any drift detection event",
        metric_source="data_quality",
        metric_name="drift_detections",
        threshold=1,
        window_seconds=60,
        severity=SecuritySeverity.HIGH.value,
    ),
    AlertCondition(
        condition_id="high-rejection-rate",
        name="High rejection rate",
        description="Rejection rate exceeds 20% of requests",
        metric_source="requests",
        metric_name="rejection_rate_pct",
        threshold=20.0,
        window_seconds=300,
        severity=SecuritySeverity.HIGH.value,
    ),
]


# ── Health / Readiness Semantics ─────────────────────────────────────────

class HealthLevel(str, Enum):
    """Health check granularity."""
    LIVENESS = "liveness"        # Process alive
    READINESS = "readiness"      # Can process requests
    MODEL_READY = "model_ready"  # Verified model loaded
    FULLY_HEALTHY = "fully_healthy"  # All systems go


@dataclass
class HealthReport:
    """Structured health report with clear semantics."""
    liveness: str = "alive"
    readiness: str = "ready"
    model_readiness: str = "not_loaded"
    runtime_state: str = "UNKNOWN"
    db_status: str = "unknown"
    audit_chain_status: str = "unknown"
    release_id: str = ""
    model_id: str = ""
    feature_version: str = ""
    uptime_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def overall_status(self) -> str:
        if self.liveness != "alive":
            return "dead"
        if self.readiness != "ready":
            return "not_ready"
        if self.model_readiness != "loaded":
            return "degraded"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.overall_status(),
            "liveness": self.liveness,
            "readiness": self.readiness,
            "model_readiness": self.model_readiness,
            "runtime_state": self.runtime_state,
            "db": self.db_status,
            "audit_chain": self.audit_chain_status,
            "release_id": self.release_id,
            "model_id": self.model_id,
            "feature_version": self.feature_version,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "details": self.details,
        }


# ── Security Incident Ledger ─────────────────────────────────────────────

class SecurityIncidentLedger:
    """Tamper-evident security incident/event ledger.

    Integrates with existing forensic architecture concepts.
    Events are hash-chained for integrity detection.
    """

    def __init__(self) -> None:
        self._events: list[SecurityEvent] = []
        self._chain_head: str = ""
        self._lock = threading.Lock()
        # Rate tracking for burst detection
        self._auth_failure_times: deque = deque(maxlen=100)
        self._malformed_times: deque = deque(maxlen=100)
        self._request_times: deque = deque(maxlen=1000)

    def record_event(
        self,
        event_type: SecurityEventType | str,
        severity: SecuritySeverity | str,
        service: str,
        correlation_id: str = "",
        source_ip: str = "",
        attempted_operation: str = "",
        outcome: str = "",
        details: dict[str, Any] | None = None,
    ) -> SecurityEvent:
        """Record a security event with hash chain binding."""
        with self._lock:
            ev = SecurityEvent(
                event_id=f"sev-{uuid.uuid4().hex[:12]}",
                timestamp=time.time(),
                event_type=event_type.value if isinstance(event_type, SecurityEventType) else event_type,
                severity=severity.value if isinstance(severity, SecuritySeverity) else severity,
                service=service,
                correlation_id=correlation_id,
                source_ip=source_ip,
                attempted_operation=attempted_operation,
                outcome=outcome,
                details=details or {},
            )
            ev.compute_hash(self._chain_head)
            self._chain_head = ev.event_hash
            self._events.append(ev)
            return ev

    def detect_bursts(self, window_seconds: float = 60.0) -> list[SecurityEvent]:
        """Detect request bursts using sliding window."""
        now = time.time()
        with self._lock:
            self._request_times.append(now)
            recent = [t for t in self._request_times if now - t <= window_seconds]
            self._request_times.clear()
            self._request_times.extend(recent)
            count = len(recent)
        if count >= 50:  # configurable threshold
            return [self.record_event(
                SecurityEventType.REQUEST_BURST,
                SecuritySeverity.MEDIUM,
                "observability",
                details={"count": count, "window_seconds": window_seconds},
            )]
        return []

    def detect_auth_failures(self, window_seconds: float = 300.0, threshold: int = 5) -> list[SecurityEvent]:
        """Detect repeated authentication failures."""
        now = time.time()
        with self._lock:
            recent = [t for t in self._auth_failure_times if now - t <= window_seconds]
            self._auth_failure_times.clear()
            self._auth_failure_times.extend(recent)
            count = len(recent)
        if count >= threshold:
            return [self.record_event(
                SecurityEventType.REPEATED_AUTH_FAILURE,
                SecuritySeverity.HIGH,
                "observability",
                details={"failure_count": count, "window_seconds": window_seconds},
            )]
        return []

    def track_auth_failure(self) -> None:
        with self._lock:
            self._auth_failure_times.append(time.time())

    def track_malformed(self) -> None:
        with self._lock:
            self._malformed_times.append(time.time())

    def verify_chain(self) -> tuple[bool, str]:
        """Verify the hash chain integrity."""
        prev_hash = ""
        for i, ev in enumerate(self._events):
            expected = hashlib.sha256(json.dumps({
                "event_id": ev.event_id,
                "timestamp": ev.timestamp,
                "event_type": ev.event_type,
                "severity": ev.severity,
                "service": ev.service,
                "correlation_id": ev.correlation_id,
                "attempted_operation": ev.attempted_operation,
                "outcome": ev.outcome,
                "containment_status": ev.containment_status,
                "previous_hash": prev_hash,
            }, sort_keys=True).encode()).hexdigest()
            if ev.event_hash != expected:
                return False, f"Chain broken at event {i} ({ev.event_id})"
            prev_hash = ev.event_hash
        return True, "Chain intact"

    def get_events(
        self, event_type: SecurityEventType | str | None = None,
        severity: SecuritySeverity | str | None = None,
        limit: int = 100,
    ) -> list[SecurityEvent]:
        """Get events with optional filtering."""
        et = event_type.value if isinstance(event_type, SecurityEventType) else event_type
        sev = severity.value if isinstance(severity, SecuritySeverity) else severity
        result = self._events
        if et:
            result = [e for e in result if e.event_type == et]
        if sev:
            result = [e for e in result if e.severity == sev]
        return result[-limit:]

    def get_event_count(self) -> int:
        return len(self._events)


# ── Alert Evaluator ──────────────────────────────────────────────────────

class AlertEvaluator:
    """Evaluates alert conditions against operational metrics."""

    def __init__(
        self,
        conditions: list[AlertCondition] | None = None,
        ledger: SecurityIncidentLedger | None = None,
    ) -> None:
        self.conditions = conditions or list(DEFAULT_ALERT_CONDITIONS)
        self.ledger = ledger
        self._fired: dict[str, float] = {}  # condition_id -> last_fire_time
        self._lock = threading.Lock()

    def evaluate(self, metrics: OperationalMetrics) -> list[dict[str, Any]]:
        """Evaluate all alert conditions and return fired alerts."""
        now = time.time()
        fired = []
        m_dict = metrics.to_dict()
        with self._lock:
            for cond in self.conditions:
                if not cond.enabled:
                    continue
                # Cooldown: don't re-fire within window_seconds
                last = self._fired.get(cond.condition_id, 0)
                if now - last < cond.window_seconds:
                    continue

                value = self._get_metric_value(m_dict, cond)
                if value >= cond.threshold:
                    self._fired[cond.condition_id] = now
                    alert_info = {
                        "condition_id": cond.condition_id,
                        "name": cond.name,
                        "description": cond.description,
                        "severity": cond.severity,
                        "threshold": cond.threshold,
                        "current_value": value,
                        "timestamp": now,
                    }
                    fired.append(alert_info)
                    # Record in ledger if available
                    if self.ledger:
                        self.ledger.record_event(
                            SecurityEventType.DEGRADED_MODE_ENTRY
                            if cond.severity == SecuritySeverity.CRITICAL.value
                            else SecurityEventType.REQUEST_BURST,
                            cond.severity,
                            "alert_evaluator",
                            attempted_operation=cond.name,
                            outcome="FIRED",
                            details=alert_info,
                        )
        return fired

    def _get_metric_value(self, m_dict: dict, cond: AlertCondition) -> float:
        """Extract the relevant metric value for an alert condition."""
        source = cond.metric_source
        name = cond.metric_name

        if source == "requests" and name == "error_rate_pct":
            total = m_dict["requests"]["total"]
            errors = m_dict["requests"]["errors"]
            return (errors / total * 100) if total > 0 else 0.0

        if source == "requests" and name == "rejection_rate_pct":
            total = m_dict["requests"]["total"]
            rejected = m_dict["requests"]["rejected"]
            return (rejected / total * 100) if total > 0 else 0.0

        # Direct mapping
        source_dict = m_dict.get(source, {})
        return float(source_dict.get(name, 0))


# ── Correlation ID Manager ───────────────────────────────────────────────

def generate_correlation_id() -> str:
    """Generate a correlation ID for request tracing."""
    return f"cid-{uuid.uuid4().hex[:16]}"


# ── Central Observability Store ───────────────────────────────────────────

class ObservabilityStore:
    """Central in-process observability store.

    Thread-safe.  Persists to JSON on demand for export.
    Does NOT persist by default (in-memory only for this phase).
    """

    def __init__(self, service_name: str = "risk-engine") -> None:
        self.service_name = service_name
        self.metrics = OperationalMetrics()
        self.model_telemetry = ModelTelemetry()
        self.security_ledger = SecurityIncidentLedger()
        self.alert_evaluator = AlertEvaluator(ledger=self.security_ledger)
        self._log_buffer: deque[StructuredLogEntry] = deque(maxlen=10000)
        self._lock = threading.Lock()
        self._started_at = time.time()

    def log(
        self, severity: str, endpoint: str, operation: str, outcome: str,
        correlation_id: str = "", latency_ms: float = 0.0,
        model_id: str = "", release_id: str = "", feature_version: str = "",
        runtime_state: str = "", error_category: str = "",
        details: dict[str, Any] | None = None,
    ) -> StructuredLogEntry:
        """Write a structured log entry."""
        entry = StructuredLogEntry(
            timestamp=time.time(),
            service=self.service_name,
            severity=severity,
            correlation_id=correlation_id or generate_correlation_id(),
            endpoint=endpoint,
            operation=operation,
            outcome=outcome,
            latency_ms=latency_ms,
            model_id=model_id,
            release_id=release_id,
            feature_version=feature_version,
            runtime_state=runtime_state,
            error_category=error_category,
        )
        with self._lock:
            self._log_buffer.append(entry)
        return entry

    def get_logs(self, limit: int = 100) -> list[StructuredLogEntry]:
        with self._lock:
            return list(self._log_buffer)[-limit:]

    def evaluate_alerts(self) -> list[dict[str, Any]]:
        """Evaluate alert conditions against current metrics."""
        return self.alert_evaluator.evaluate(self.metrics)

    def health_report(
        self, db_ok: bool = True, model_ok: bool = True,
        runtime_state: str = "UNKNOWN", release_id: str = "",
        model_id: str = "", feature_version: str = "",
        audit_chain_ok: bool = True,
    ) -> HealthReport:
        """Generate a structured health report."""
        return HealthReport(
            liveness="alive",
            readiness="ready" if db_ok else "not_ready",
            model_readiness="loaded" if model_ok else "not_loaded",
            runtime_state=runtime_state,
            db_status="ok" if db_ok else "error",
            audit_chain_status="ok" if audit_chain_ok else "error",
            release_id=release_id,
            model_id=model_id,
            feature_version=feature_version,
            uptime_seconds=time.time() - self._started_at,
        )

    def monitor_event(self, operation: str, endpoint: str = "", **kwargs: Any) -> _RequestMonitor:
        """Context manager for monitoring a request."""
        return _RequestMonitor(self, endpoint, operation, **kwargs)

    def export_state(self) -> dict[str, Any]:
        """Export complete observability state (no secrets)."""
        return {
            "service": self.service_name,
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "metrics": self.metrics.to_dict(),
            "model_telemetry": self.model_telemetry.to_dict(),
            "security_events": {
                "total": self.security_ledger.get_event_count(),
                "chain_valid": self.security_ledger.verify_chain()[0],
            },
            "recent_logs": len(self._log_buffer),
        }


class _RequestMonitor:
    """Context manager for monitoring a single request."""

    def __init__(self, store: ObservabilityStore, endpoint: str, operation: str, **kwargs: Any) -> None:
        self.store = store
        self.endpoint = endpoint
        self.operation = operation
        self.correlation_id = kwargs.get("correlation_id", generate_correlation_id())
        self.model_id = kwargs.get("model_id", "")
        self.release_id = kwargs.get("release_id", "")
        self.feature_version = kwargs.get("feature_version", "")
        self._start = 0.0
        self._outcome = "unknown"

    def __enter__(self) -> _RequestMonitor:
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        latency_ms = (time.monotonic() - self._start) * 1000
        if exc_type is not None:
            self._outcome = "error"
            self.store.metrics.record_error()
        self.store.log(
            severity="ERROR" if self._outcome == "error" else "INFO",
            endpoint=self.endpoint,
            operation=self.operation,
            outcome=self._outcome,
            correlation_id=self.correlation_id,
            latency_ms=latency_ms,
            model_id=self.model_id,
            release_id=self.release_id,
            feature_version=self.feature_version,
        )

    def set_outcome(self, outcome: str) -> None:
        self._outcome = outcome
