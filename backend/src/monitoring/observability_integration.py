"""Phase 72: Runtime observability integration for the PS-14 risk engine.

Wires Phase 70 ObservabilityStore into the actual production request path.
Provides thin functions called from main.py at each critical point.

This module does NOT duplicate Phase 70 infrastructure. It bridges it
into the live runtime with minimal code surface.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import time
import threading
from typing import Any

from src.monitoring.observability import (
    ObservabilityStore,
    SecurityEventType,
    SecuritySeverity,
    generate_correlation_id,
)


_store: ObservabilityStore | None = None
_lock = threading.Lock()

# Observability must never cause a fraud decision to fail.
# Every public function wraps in try/except to guarantee fail-open.


def init_observability(service_name: str = "risk-engine") -> ObservabilityStore:
    """Initialize the global observability store. Called once in lifespan."""
    global _store
    with _lock:
        _store = ObservabilityStore(service_name=service_name)
        return _store


def get_store() -> ObservabilityStore | None:
    """Get the current observability store (may be None before init)."""
    return _store


def on_request_received(
    event_id: str = "",
    fraud_id: str = "",
    correlation_id: str = "",
) -> dict[str, Any]:
    """Called at the very start of /internal/evaluate.

    Returns monitoring context dict to be threaded through the request.
    """
    try:
        store = _store
        if store is None:
            return {"cid": correlation_id or generate_correlation_id()}
        cid = correlation_id or generate_correlation_id()
        return {"cid": cid}
    except Exception:
        return {"cid": correlation_id or generate_correlation_id()}


def on_auth_failure(
    correlation_id: str = "",
    source_ip: str = "",
) -> None:
    """Record an authentication failure."""
    try:
        store = _store
        if store is None:
            return
        store.metrics.record_auth_failure()
        store.security_ledger.track_auth_failure()
        store.security_ledger.record_event(
            SecurityEventType.AUTH_FAILURE,
            SecuritySeverity.MEDIUM,
            "risk-engine",
            correlation_id=correlation_id,
            source_ip=source_ip,
            attempted_operation="/internal/evaluate",
            outcome="REJECTED",
        )
        # Check for repeated failures
        store.security_ledger.detect_auth_failures(threshold=5)
    except Exception:
        pass


def on_auth_success(correlation_id: str = "") -> None:
    """Record a successful authentication (for metrics baseline)."""
    pass  # No security event on success; just a no-op for API consistency


def on_validation_failure(
    correlation_id: str = "",
    reason: str = "",
    details: dict | None = None,
) -> None:
    """Record a schema/validation rejection."""
    try:
        store = _store
        if store is None:
            return
        store.metrics.record_malformed()
        store.metrics.record_request(0, False)
        store.security_ledger.record_event(
            SecurityEventType.MALFORMED_REQUEST,
            SecuritySeverity.LOW,
            "risk-engine",
            correlation_id=correlation_id,
            attempted_operation="/internal/evaluate",
            outcome="VALIDATION_REJECTED",
            details={"reason": reason, **(details or {})},
        )
    except Exception:
        pass


def on_idempotent_replay(
    correlation_id: str = "",
    event_id: str = "",
) -> None:
    """Record an idempotent replay."""
    try:
        store = _store
        if store is None:
            return
        store.metrics.record_idempotency(False)
        store.metrics.record_request(0.1, True)
        store.log(
            severity="INFO",
            endpoint="/internal/evaluate",
            operation="idempotent_replay",
            outcome="REPLAYED",
            correlation_id=correlation_id,
        )
    except Exception:
        pass


def on_feature_block(
    correlation_id: str = "",
    enforcement_result: dict | None = None,
) -> None:
    """Record a feature enforcement block."""
    try:
        store = _store
        if store is None:
            return
        store.metrics.record_feature_block()
        store.security_ledger.record_event(
            SecurityEventType.FEATURE_ENFORCEMENT_BLOCK,
            SecuritySeverity.MEDIUM,
            "risk-engine",
            correlation_id=correlation_id,
            attempted_operation="/internal/evaluate",
            outcome="BLOCKED",
            details=enforcement_result or {},
        )
    except Exception:
        pass


def on_degraded_mode(
    correlation_id: str = "",
    reason: str = "",
    runtime_state: str = "",
) -> None:
    """Record entry into degraded mode."""
    try:
        store = _store
        if store is None:
            return
        store.metrics.degraded_mode_entries += 1
        store.model_telemetry.degradation_events += 1
        store.security_ledger.record_event(
            SecurityEventType.DEGRADED_MODE_ENTRY,
            SecuritySeverity.HIGH,
            "risk-engine",
            correlation_id=correlation_id,
            attempted_operation="/internal/evaluate",
            outcome="DEGRADED",
            details={"reason": reason, "runtime_state": runtime_state},
        )
    except Exception:
        pass


def on_drift_detected(
    correlation_id: str = "",
    drift_state: str = "",
) -> None:
    """Record a drift detection event."""
    try:
        store = _store
        if store is None:
            return
        store.metrics.drift_detections += 1
        store.model_telemetry.drift_signals += 1
        store.security_ledger.record_event(
            SecurityEventType.RUNTIME_STATE_UNEXPECTED,
            SecuritySeverity.HIGH,
            "risk-engine",
            correlation_id=correlation_id,
            attempted_operation="/internal/evaluate",
            outcome="DRIFT_DETECTED",
            details={"drift_state": drift_state},
        )
    except Exception:
        pass


def on_evaluation_complete(
    correlation_id: str = "",
    event_id: str = "",
    fraud_id: str = "",
    ml_score: float = 0.0,
    risk_band: str = "",
    decision: str = "",
    risk_score: int = 0,
    degraded: bool = False,
    model_variance: float = 0.0,
    model_disagreement: float = 0.0,
    latency_ms: float = 0.0,
    model_id: str = "altman_native",
    release_id: str = "",
    feature_version: str = "altman_native_v1",
    idempotent_replay: bool = False,
    reason_codes: list | None = None,
    runtime_state: str = "READY",
) -> None:
    """Record the completion of a successful evaluation."""
    try:
        store = _store
        if store is None:
            return

        # Request metrics
        store.metrics.record_request(latency_ms, True)
        if idempotent_replay:
            store.metrics.record_idempotency(False)
        else:
            store.metrics.record_idempotency(True)

        # Model prediction metrics
        store.metrics.record_prediction(ml_score, risk_band, decision)

        # Model telemetry
        store.model_telemetry.record_prediction(
            ml_score=ml_score,
            risk_band=risk_band,
            decision=decision,
            model_variance=model_variance,
            model_disagreement=model_disagreement,
        )

        # Structured log
        store.log(
            severity="INFO",
            endpoint="/internal/evaluate",
            operation="evaluate",
            outcome="COMPLETED",
            correlation_id=correlation_id,
            latency_ms=latency_ms,
            model_id=model_id,
            release_id=release_id,
            feature_version=feature_version,
            runtime_state=runtime_state,
            details={
                "event_id": event_id,
                "risk_score": risk_score,
                "risk_band": risk_band,
                "decision": decision,
                "degraded": degraded,
                "idempotent_replay": idempotent_replay,
            },
        )
    except Exception:
        pass


def on_evaluation_error(
    correlation_id: str = "",
    error_category: str = "",
    latency_ms: float = 0.0,
) -> None:
    """Record an evaluation error."""
    try:
        store = _store
        if store is None:
            return
        store.metrics.record_request(latency_ms, False)
        store.metrics.record_error()
        store.log(
            severity="ERROR",
            endpoint="/internal/evaluate",
            operation="evaluate",
            outcome="ERROR",
            correlation_id=correlation_id,
            latency_ms=latency_ms,
            error_category=error_category,
        )
    except Exception:
        pass


def on_release_integrity_failure(
    correlation_id: str = "",
    details: dict | None = None,
) -> None:
    """Record a release integrity failure."""
    try:
        store = _store
        if store is None:
            return
        store.security_ledger.record_event(
            SecurityEventType.RELEASE_INTEGRITY_FAILURE,
            SecuritySeverity.CRITICAL,
            "risk-engine",
            correlation_id=correlation_id,
            attempted_operation="/internal/evaluate",
            outcome="INTEGRITY_FAILURE",
            details=details or {},
        )
    except Exception:
        pass


def evaluate_alerts() -> list[dict[str, Any]]:
    """Evaluate alert conditions against current metrics."""
    try:
        store = _store
        if store is None:
            return []
        return store.evaluate_alerts()
    except Exception:
        return []


def get_health_report(
    db_ok: bool = True,
    model_ok: bool = True,
    runtime_state: str = "UNKNOWN",
    release_id: str = "",
    model_id: str = "",
    feature_version: str = "",
    audit_chain_ok: bool = True,
) -> dict[str, Any]:
    """Generate a health report integrating Phase 70 semantics."""
    try:
        store = _store
        if store is None:
            return {
                "status": "unknown",
                "liveness": "alive",
                "readiness": "unknown",
                "model_readiness": "unknown",
                "error": "observability not initialized",
            }
        report = store.health_report(
            db_ok=db_ok,
            model_ok=model_ok,
            runtime_state=runtime_state,
            release_id=release_id,
            model_id=model_id,
            feature_version=feature_version,
            audit_chain_ok=audit_chain_ok,
        )
        return report.to_dict()
    except Exception:
        return {"status": "error", "error": "health report generation failed"}


def get_observability_summary() -> dict[str, Any]:
    """Get a summary of observability state for export."""
    try:
        store = _store
        if store is None:
            return {"error": "not initialized"}
        return store.export_state()
    except Exception:
        return {"error": "export failed"}


def get_metrics_summary() -> dict[str, Any]:
    """Get current metrics summary."""
    try:
        store = _store
        if store is None:
            return {}
        return store.metrics.to_dict()
    except Exception:
        return {}


def get_model_telemetry_summary() -> dict[str, Any]:
    """Get current model telemetry summary."""
    try:
        store = _store
        if store is None:
            return {}
        return store.model_telemetry.to_dict()
    except Exception:
        return {}


def get_security_events_summary() -> dict[str, Any]:
    """Get security events summary."""
    try:
        store = _store
        if store is None:
            return {"total": 0, "chain_valid": True}
        ok, msg = store.security_ledger.verify_chain()
        return {
            "total": store.security_ledger.get_event_count(),
            "chain_valid": ok,
            "chain_message": msg,
        }
    except Exception:
        return {"total": 0, "chain_valid": False, "error": "query failed"}
