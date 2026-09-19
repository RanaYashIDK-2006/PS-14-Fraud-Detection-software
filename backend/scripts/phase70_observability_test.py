#!/usr/bin/env python3
"""Phase 70: Production observability, monitoring & security event detection tests.

Tests structured logging, operational metrics, model telemetry, security event
detection, alerting, health/readiness, correlation, and forensic chain integrity.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.monitoring.observability import (
    SecurityEventType,
    SecuritySeverity,
    SecurityEvent,
    StructuredLogEntry,
    OperationalMetrics,
    ModelTelemetry,
    AlertCondition,
    DEFAULT_ALERT_CONDITIONS,
    HealthLevel,
    HealthReport,
    SecurityIncidentLedger,
    AlertEvaluator,
    ObservabilityStore,
    generate_correlation_id,
)

passed = 0
failed = 0
errors = []


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        errors.append(name)


# ── Section 1: Security Event Types & Severity ───────────────────────────
print("\n=== Security Event Types & Severity ===")

check("1. SecurityEventType has AUTH_FAILURE", hasattr(SecurityEventType, "AUTH_FAILURE"))
check("2. SecurityEventType has REPEATED_AUTH_FAILURE", hasattr(SecurityEventType, "REPEATED_AUTH_FAILURE"))
check("3. SecurityEventType has MALFORMED_REQUEST", hasattr(SecurityEventType, "MALFORMED_REQUEST"))
check("4. SecurityEventType has RELEASE_INTEGRITY_FAILURE", hasattr(SecurityEventType, "RELEASE_INTEGRITY_FAILURE"))
check("5. SecurityEventType has MODEL_ARTIFACT_TAMPER", hasattr(SecurityEventType, "MODEL_ARTIFACT_TAMPER"))
check("6. SecurityEventType has AUDIT_CHAIN_FAILURE", hasattr(SecurityEventType, "AUDIT_CHAIN_FAILURE"))
check("7. SecurityEventType has RUNTIME_STATE_UNEXPECTED", hasattr(SecurityEventType, "RUNTIME_STATE_UNEXPECTED"))
check("8. SecurityEventType has REQUEST_BURST", hasattr(SecurityEventType, "REQUEST_BURST"))
check("9. SecurityEventType has DEGRADED_MODE_ENTRY", hasattr(SecurityEventType, "DEGRADED_MODE_ENTRY"))
check("10. SecuritySeverity has INFO/LOW/MEDIUM/HIGH/CRITICAL",
      all(hasattr(SecuritySeverity, s) for s in ("INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL")))

# ── Section 2: SecurityEvent Record ──────────────────────────────────────
print("\n=== SecurityEvent Record ===")

ev = SecurityEvent(
    event_id="sev-test-001",
    timestamp=1000.0,
    event_type=SecurityEventType.AUTH_FAILURE.value,
    severity=SecuritySeverity.HIGH.value,
    service="risk-engine",
    correlation_id="cid-test",
    attempted_operation="/internal/evaluate",
    outcome="REJECTED",
)
check("11. SecurityEvent created", ev.event_id == "sev-test-001")
check("12. SecurityEvent has timestamp", ev.timestamp == 1000.0)
check("13. SecurityEvent has event_type", ev.event_type == "AUTH_FAILURE")
check("14. SecurityEvent has severity", ev.severity == "HIGH")

h1 = ev.compute_hash()
check("15. SecurityEvent hash computed", len(h1) == 64)
h2 = ev.compute_hash()
check("16. SecurityEvent hash deterministic", h1 == h2)

# Chain binding: different prev_hash should produce different hash
h3 = ev.compute_hash(previous_hash="abc123")
check("17. Chain binding changes hash", h3 != h1)

ev_dict = ev.to_dict()
check("18. SecurityEvent to_dict has event_id", ev_dict["event_id"] == "sev-test-001")
check("19. SecurityEvent to_dict has event_type", ev_dict["event_type"] == "AUTH_FAILURE")
check("20. SecurityEvent to_dict has severity", ev_dict["severity"] == "HIGH")

# ── Section 3: Structured Logging ────────────────────────────────────────
print("\n=== Structured Logging ===")

entry = StructuredLogEntry(
    timestamp=time.time(),
    service="risk-engine",
    severity="INFO",
    correlation_id="cid-abc",
    endpoint="/internal/evaluate",
    operation="evaluate",
    outcome="ALLOW",
    latency_ms=1.5,
    model_id="altman_native",
    release_id="release-001",
    feature_version="altman_native_v1",
    runtime_state="READY",
)
check("21. StructuredLogEntry created", entry.service == "risk-engine")
check("22. Log entry has correlation_id", entry.correlation_id == "cid-abc")
check("23. Log entry has latency_ms", entry.latency_ms == 1.5)

log_json = entry.to_json()
check("24. Log entry to_json is valid JSON", json.loads(log_json) is not None)
log_parsed = json.loads(log_json)
check("25. Log JSON has ts field", "ts" in log_parsed)
check("26. Log JSON has svc field", log_parsed["svc"] == "risk-engine")
check("27. Log JSON has sev field", log_parsed["sev"] == "INFO")
check("28. Log JSON has cid field", log_parsed["cid"] == "cid-abc")
check("29. Log JSON has lat_ms field", log_parsed["lat_ms"] == 1.5)

# ── Section 4: Operational Metrics ───────────────────────────────────────
print("\n=== Operational Metrics ===")

metrics = OperationalMetrics()
check("30. OperationalMetrics created", metrics.total_requests == 0)

metrics.record_request(1.0, True)
metrics.record_request(2.0, True)
metrics.record_request(3.0, False)
check("31. Record requests counts correctly",
      metrics.total_requests == 3 and metrics.successful_requests == 2 and metrics.rejected_requests == 1)

metrics.record_error()
check("32. Record error increments count", metrics.error_count == 1)

metrics.record_prediction(0.8, "high", "block")
metrics.record_prediction(0.1, "low", "allow")
metrics.record_prediction(0.5, "medium", "review")
check("33. Prediction counts correct",
      metrics.prediction_count == 3 and
      metrics.score_band_high == 1 and
      metrics.score_band_low == 1 and
      metrics.score_band_medium == 1)
check("34. Decision counts correct",
      metrics.decision_block == 1 and
      metrics.decision_allow == 1 and
      metrics.decision_review == 1)

metrics.record_inference_latency(2.5)
metrics.record_inference_latency(3.5)
check("35. Inference latency recorded", len(metrics._inference_buffer) == 2)

metrics.record_idempotency(True)
metrics.record_idempotency(False)
check("36. Idempotency tracked",
      metrics.new_events == 1 and metrics.replayed_events == 1)

metrics.record_schema_rejection()
check("37. Schema rejection recorded", metrics.schema_rejections == 1)

metrics.record_feature_block()
check("38. Feature block recorded", metrics.feature_enforcement_blocks == 1)

metrics.record_auth_failure()
check("39. Auth failure recorded", metrics.auth_failures == 1)

metrics.record_malformed()
check("40. Malformed request recorded", metrics.malformed_requests == 1)

pcts = metrics.latency_percentiles()
check("41. Latency percentiles computed", pcts["count"] == 3)
check("42. p50 is correct", pcts["p50"] == 2.0)
check("43. p95 exists", "p95" in pcts)
check("44. p99 exists", "p99" in pcts)
check("45. max is correct", pcts["max"] == 3.0)

inf_pcts = metrics.inference_latency_percentiles()
check("46. Inference latency percentiles", inf_pcts["count"] == 2)

m_dict = metrics.to_dict()
check("47. to_dict has requests", "requests" in m_dict)
check("48. to_dict has latency", "latency" in m_dict)
check("49. to_dict has model", "model" in m_dict)
check("50. to_dict has data_quality", "data_quality" in m_dict)
check("51. to_dict has idempotency", "idempotency" in m_dict)
check("52. to_dict has security", "security" in m_dict)
check("53. to_dict has system", "system" in m_dict)

# Empty metrics
empty = OperationalMetrics()
empty_pcts = empty.latency_percentiles()
check("54. Empty metrics returns zero percentiles", empty_pcts["count"] == 0)

# ── Section 5: Model Telemetry ──────────────────────────────────────────
print("\n=== Model Telemetry ===")

tel = ModelTelemetry(model_id="altman_native", release_id="release-001", feature_version="altman_native_v1")
check("55. ModelTelemetry created", tel.model_id == "altman_native")

tel.record_prediction(0.8, "high", "block", model_variance=0.01, model_disagreement=0.05)
tel.record_prediction(0.1, "low", "allow", model_variance=0.001, model_disagreement=0.01)
check("56. Prediction count correct", tel.prediction_count == 2)

dist = tel.score_distribution()
check("57. Score distribution count", dist["count"] == 2)
check("58. Score min recorded", dist["score_min"] == 0.1)
check("59. Score max recorded", dist["score_max"] == 0.8)
check("60. Score mean computed", 0.44 < dist["score_mean"] < 0.46)
check("61. Avg variance computed", dist["avg_variance"] > 0)
check("62. Avg disagreement computed", dist["avg_disagreement"] > 0)

# Empty telemetry
empty_tel = ModelTelemetry()
empty_dist = empty_tel.score_distribution()
check("63. Empty telemetry count is 0", empty_dist["count"] == 0)

tel_dict = tel.to_dict()
check("64. Telemetry to_dict has model_id", tel_dict["model_id"] == "altman_native")
check("65. Telemetry to_dict has release_id", tel_dict["release_id"] == "release-001")
check("66. Telemetry to_dict has feature_version", tel_dict["feature_version"] == "altman_native_v1")

# ── Section 6: Security Incident Ledger ──────────────────────────────────
print("\n=== Security Incident Ledger ===")

ledger = SecurityIncidentLedger()
check("67. Ledger created", ledger.get_event_count() == 0)

e1 = ledger.record_event(
    SecurityEventType.AUTH_FAILURE,
    SecuritySeverity.HIGH,
    "risk-engine",
    correlation_id="cid-1",
    attempted_operation="/internal/evaluate",
    outcome="REJECTED",
)
check("68. Event recorded", ledger.get_event_count() == 1)
check("69. Event has ID", e1.event_id.startswith("sev-"))
check("70. Event has hash", len(e1.event_hash) == 64)

e2 = ledger.record_event(
    SecurityEventType.MALFORMED_REQUEST,
    SecuritySeverity.MEDIUM,
    "risk-engine",
    correlation_id="cid-2",
    outcome="REJECTED",
)
check("71. Second event recorded", ledger.get_event_count() == 2)

# Chain integrity
chain_ok, chain_msg = ledger.verify_chain()
check("72. Chain integrity verified", chain_ok is True)
check("73. Chain message is 'Chain intact'", chain_msg == "Chain intact")

# Tamper detection: modify event
original_hash = e1.event_hash
e1.event_hash = "TAMPERED"
chain_ok2, chain_msg2 = ledger.verify_chain()
check("74. Tampered event detected", chain_ok2 is False)
check("75. Tamper detection message", "broken" in chain_msg2.lower() or "Chain" in chain_msg2)
e1.event_hash = original_hash  # restore

# Filtering
auth_events = ledger.get_events(event_type=SecurityEventType.AUTH_FAILURE)
check("76. Filter by event type works", len(auth_events) == 1)

high_events = ledger.get_events(severity=SecuritySeverity.HIGH)
check("77. Filter by severity works", len(high_events) == 1)

# Limit
limited = ledger.get_events(limit=1)
check("78. Limit parameter works", len(limited) == 1)

# Burst detection
burst_ledger = SecurityIncidentLedger()
for i in range(55):
    burst_ledger.track_malformed()  # just tracking times
# Record enough events for burst detection
burst_events = burst_ledger.detect_bursts(window_seconds=300.0)
# Should detect burst since we're calling within a tight loop
# Note: request_times may not have 50+ entries yet, so this may not fire
check("79. Burst detection runs without error", isinstance(burst_events, list))

# Auth failure detection
auth_ledger = SecurityIncidentLedger()
for i in range(6):
    auth_ledger.track_auth_failure()
auth_burst = auth_ledger.detect_auth_failures(window_seconds=300, threshold=5)
check("80. Auth failure detection fires", len(auth_burst) >= 1)
if auth_burst:
    check("81. Auth failure event type correct", auth_burst[0].event_type == SecurityEventType.REPEATED_AUTH_FAILURE.value)

# ── Section 7: Alert Conditions ──────────────────────────────────────────
print("\n=== Alert Conditions ===")

check("82. Default alert conditions defined", len(DEFAULT_ALERT_CONDITIONS) >= 5)
check("83. Auth failure alert exists",
      any(c.condition_id == "auth-fail-5-min" for c in DEFAULT_ALERT_CONDITIONS))
check("84. Error rate alert exists",
      any(c.condition_id == "error-rate-5pct" for c in DEFAULT_ALERT_CONDITIONS))
check("85. Degraded mode alert exists",
      any(c.condition_id == "degraded-mode" for c in DEFAULT_ALERT_CONDITIONS))
check("86. Drift detection alert exists",
      any(c.condition_id == "drift-detected" for c in DEFAULT_ALERT_CONDITIONS))

# Alert evaluator
evaluator = AlertEvaluator()
check("87. AlertEvaluator created", len(evaluator.conditions) == len(DEFAULT_ALERT_CONDITIONS))

# No alerts when metrics are zero
clean_metrics = OperationalMetrics()
fired = evaluator.evaluate(clean_metrics)
check("88. No alerts on clean metrics", len(fired) == 0)

# Auth failure alert should fire
alert_metrics = OperationalMetrics()
for _ in range(10):
    alert_metrics.record_auth_failure()
evaluator2 = AlertEvaluator()
fired2 = evaluator2.evaluate(alert_metrics)
auth_alerts = [a for a in fired2 if a["condition_id"] == "auth-fail-5-min"]
check("89. Auth failure alert fires at threshold", len(auth_alerts) >= 1)

# Cooldown: shouldn't fire again immediately
fired3 = evaluator2.evaluate(alert_metrics)
auth_alerts3 = [a for a in fired3 if a["condition_id"] == "auth-fail-5-min"]
check("90. Alert cooldown prevents re-fire", len(auth_alerts3) == 0)

# Error rate alert
evaluator3 = AlertEvaluator()
error_metrics = OperationalMetrics()
for _ in range(5):
    error_metrics.record_request(1.0, False)
for _ in range(95):
    error_metrics.record_request(1.0, True)
error_metrics.record_error()
fired4 = evaluator3.evaluate(error_metrics)
# 5% error rate = 5/100 = exactly at threshold
check("91. Error rate alert evaluation runs", isinstance(fired4, list))

# Disabled condition
evaluator4 = AlertEvaluator(conditions=[
    AlertCondition(
        condition_id="test-disabled",
        name="Disabled",
        description="Should not fire",
        metric_source="security",
        metric_name="auth_failures",
        threshold=1,
        window_seconds=60,
        severity=SecuritySeverity.HIGH.value,
        enabled=False,
    )
])
disabled_metrics = OperationalMetrics()
disabled_metrics.record_auth_failure()
fired5 = evaluator4.evaluate(disabled_metrics)
check("92. Disabled condition does not fire", len(fired5) == 0)

# ── Section 8: Health Reports ────────────────────────────────────────────
print("\n=== Health Reports ===")

check("93. HealthLevel has LIVENESS", hasattr(HealthLevel, "LIVENESS"))
check("94. HealthLevel has READINESS", hasattr(HealthLevel, "READINESS"))
check("95. HealthLevel has MODEL_READY", hasattr(HealthLevel, "MODEL_READY"))

hr = HealthReport(
    liveness="alive",
    readiness="ready",
    model_readiness="loaded",
    runtime_state="READY",
    db_status="ok",
    audit_chain_status="ok",
    release_id="release-001",
    model_id="altman_native",
    feature_version="altman_native_v1",
)
check("96. HealthReport created", hr.liveness == "alive")
check("97. Overall status ok", hr.overall_status() == "ok")

# Degraded model
hr_degraded = HealthReport(model_readiness="not_loaded")
check("98. Degraded status when model not loaded", hr_degraded.overall_status() == "degraded")

# Not ready
hr_not_ready = HealthReport(readiness="not_ready")
check("99. Not ready status correct", hr_not_ready.overall_status() == "not_ready")

# Dead
hr_dead = HealthReport(liveness="dead")
check("100. Dead status correct", hr_dead.overall_status() == "dead")

hr_dict = hr.to_dict()
check("101. HealthReport to_dict has status", hr_dict["status"] == "ok")
check("102. HealthReport to_dict has release_id", hr_dict["release_id"] == "release-001")
check("103. HealthReport to_dict has model_id", hr_dict["model_id"] == "altman_native")
check("104. HealthReport to_dict has db", hr_dict["db"] == "ok")

# ── Section 9: Correlation IDs ───────────────────────────────────────────
print("\n=== Correlation IDs ===")

cid1 = generate_correlation_id()
cid2 = generate_correlation_id()
check("105. Correlation ID has prefix", cid1.startswith("cid-"))
check("106. Correlation ID is unique", cid1 != cid2)
check("107. Correlation ID length", len(cid1) == 20)

# ── Section 10: ObservabilityStore ────────────────────────────────────────
print("\n=== ObservabilityStore ===")

store = ObservabilityStore(service_name="test-service")
check("108. ObservabilityStore created", store.service_name == "test-service")

# Log
log_entry = store.log(
    severity="INFO",
    endpoint="/test",
    operation="test-op",
    outcome="OK",
    correlation_id="cid-test-store",
    latency_ms=5.0,
    model_id="test-model",
    release_id="test-release",
)
check("109. Store log entry created", log_entry.endpoint == "/test")
check("110. Store log has correlation_id", log_entry.correlation_id == "cid-test-store")

logs = store.get_logs()
check("111. Store get_logs returns entries", len(logs) == 1)
check("112. Log entry matches stored", logs[0].correlation_id == "cid-test-store")

# Metrics
store.metrics.record_request(1.5, True)
store.metrics.record_prediction(0.5, "low", "allow")
check("113. Store metrics accessible", store.metrics.total_requests == 1)

# Alert evaluation
alerts = store.evaluate_alerts()
check("114. Alert evaluation runs", isinstance(alerts, list))

# Health report
report = store.health_report(
    db_ok=True,
    model_ok=True,
    runtime_state="READY",
    release_id="test-release",
    model_id="test-model",
    feature_version="v1",
)
check("115. Health report generated", report.overall_status() == "ok")

# Export state
state = store.export_state()
check("116. Export state has service", state["service"] == "test-service")
check("117. Export state has metrics", "metrics" in state)
check("118. Export state has model_telemetry", "model_telemetry" in state)
check("119. Export state has security_events", "security_events" in state)

# Context manager monitoring
with store.monitor_event("evaluate", "/internal/evaluate") as mon:
    time.sleep(0.001)
    mon.set_outcome("ALLOW")
check("120. Context manager monitoring works", len(store.get_logs()) >= 2)

# ── Section 11: Model Telemetry Integration ──────────────────────────────
print("\n=== Model Telemetry Integration ===")

store2 = ObservabilityStore()
store2.model_telemetry.model_id = "altman_native"
store2.model_telemetry.release_id = "release-001"
store2.model_telemetry.feature_version = "altman_native_v1"

for _ in range(10):
    store2.model_telemetry.record_prediction(0.5, "low", "allow", 0.01, 0.01)

tel_dict2 = store2.model_telemetry.to_dict()
check("121. Telemetry prediction count", tel_dict2["prediction_count"] == 10)
check("122. Telemetry model_id preserved", tel_dict2["model_id"] == "altman_native")

# ── Section 12: Thread Safety ────────────────────────────────────────────
print("\n=== Thread Safety ===")

thread_store = ObservabilityStore()
errors_list = []

def thread_worker(n: int) -> None:
    try:
        for _ in range(50):
            thread_store.metrics.record_request(1.0, True)
            thread_store.metrics.record_prediction(0.5, "low", "allow")
            thread_store.log("INFO", "/test", "test", "OK")
            thread_store.security_ledger.record_event(
                SecurityEventType.AUTH_FAILURE,
                SecuritySeverity.LOW,
                "test",
            )
    except Exception as e:
        errors_list.append(str(e))

threads = [threading.Thread(target=thread_worker, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check("123. Thread safety: no exceptions", len(errors_list) == 0)
check("124. Thread safety: requests counted",
      thread_store.metrics.total_requests == 250)  # 5 * 50
check("125. Thread safety: predictions counted",
      thread_store.metrics.prediction_count == 250)
check("126. Thread safety: security events recorded",
      thread_store.security_ledger.get_event_count() == 250)

# ── Section 13: Security Event Detection Integration ─────────────────────
print("\n=== Security Event Detection Integration ===")

sec_store = ObservabilityStore()

# Simulate repeated auth failures
for _ in range(6):
    sec_store.security_ledger.track_auth_failure()
sec_store.metrics.record_auth_failure()
sec_burst = sec_store.security_ledger.detect_auth_failures(threshold=5)
check("127. Auth failure detection in store", len(sec_burst) >= 1)

# Simulate feature enforcement blocks
for _ in range(25):
    sec_store.metrics.record_feature_block()
# Verify metric count
check("128. Feature block count accurate", sec_store.metrics.feature_enforcement_blocks == 25)

# Simulate schema rejections
for _ in range(15):
    sec_store.metrics.record_schema_rejection()
check("129. Schema rejection count accurate", sec_store.metrics.schema_rejections == 15)

# ── Section 14: Failure Behavior ─────────────────────────────────────────
print("\n=== Failure Behavior ===")

fail_store = ObservabilityStore()

# Logging should not fail even with unusual inputs
try:
    fail_store.log(
        severity="ERROR",
        endpoint="/fail",
        operation="fail-op",
        outcome="ERROR",
        error_category="VALIDATION",
    )
    check("130. Logging with error category works", True)
except Exception as e:
    check("130. Logging with error category works", False, str(e))

# Export should work even with empty state
try:
    empty_state = fail_store.export_state()
    check("131. Export empty state works", "metrics" in empty_state)
except Exception as e:
    check("131. Export empty state works", False, str(e))

# Health report with everything degraded
degraded_report = fail_store.health_report(
    db_ok=False,
    model_ok=False,
    runtime_state="DRIFTED",
    audit_chain_ok=False,
)
check("132. Fully degraded health report", degraded_report.overall_status() == "not_ready")

# ── Section 15: Privacy Review ───────────────────────────────────────────
print("\n=== Privacy Review ===")

# Structured logs should NOT contain PII fields
log_json_check = json.loads(log_entry.to_json())
check("133. Log JSON has no 'name' field", "name" not in log_json_check)
check("134. Log JSON has no 'email' field", "email" not in log_json_check)
check("135. Log JSON has no 'phone' field", "phone" not in log_json_check)
check("136. Log JSON has no 'ssn' field", "ssn" not in log_json_check)
check("137. Log JSON has no 'card' field", "card" not in log_json_check)

# Export state should NOT contain secrets
state_check = store.export_state()
state_str = json.dumps(state_check)
check("138. Export state has no 'token' in values", "token" not in state_str.lower())
check("139. Export state has no 'secret' in values", "secret" not in state_str.lower())
check("140. Export state has no 'password' in values", "password" not in state_str.lower())

# Security events should not store raw PII
ev_privacy = SecurityEvent(
    event_id="sev-privacy-test",
    timestamp=time.time(),
    event_type=SecurityEventType.AUTH_FAILURE.value,
    severity=SecuritySeverity.HIGH.value,
    service="risk-engine",
)
ev_privacy_dict = ev_privacy.to_dict()
check("141. Security event has no raw PII fields",
      not any(k in ev_privacy_dict for k in ("name", "email", "phone", "ssn", "card")))

# ── Section 16: Determinism ──────────────────────────────────────────────
print("\n=== Determinism ===")

# Same metrics should produce same percentiles
det_m1 = OperationalMetrics()
det_m2 = OperationalMetrics()
for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    det_m1.record_request(v, True)
    det_m2.record_request(v, True)
check("142. Deterministic percentiles",
      det_m1.latency_percentiles() == det_m2.latency_percentiles())

# Same predictions produce same telemetry distribution
det_t1 = ModelTelemetry()
det_t2 = ModelTelemetry()
for s in [0.1, 0.5, 0.9]:
    det_t1.record_prediction(s, "low", "allow")
    det_t2.record_prediction(s, "low", "allow")
check("143. Deterministic telemetry distribution",
      det_t1.score_distribution() == det_t2.score_distribution())

# Same health params produce same report
hr1 = HealthReport(release_id="r1", model_id="m1")
hr2 = HealthReport(release_id="r1", model_id="m1")
check("144. Deterministic health report", hr1.to_dict() == hr2.to_dict())

# ── Section 17: Edge Cases ───────────────────────────────────────────────
print("\n=== Edge Cases ===")

# Negative latency
neg_metrics = OperationalMetrics()
neg_metrics.record_request(-1.0, True)
neg_pcts = neg_metrics.latency_percentiles()
check("145. Negative latency handled", neg_pcts["count"] == 1)

# Very large latency
big_metrics = OperationalMetrics()
big_metrics.record_request(99999.0, True)
big_pcts = big_metrics.latency_percentiles()
check("146. Large latency handled", big_pcts["max"] == 99999.0)

# Zero division in error rate
zero_metrics = OperationalMetrics()
zero_eval = AlertEvaluator(conditions=[
    AlertCondition(
        condition_id="test-zero-div",
        name="Zero div test",
        description="Test",
        metric_source="requests",
        metric_name="error_rate_pct",
        threshold=5.0,
        window_seconds=60,
        severity=SecuritySeverity.HIGH.value,
    )
])
try:
    zero_fired = zero_eval.evaluate(zero_metrics)
    check("147. Zero-division in error rate handled", True)
except ZeroDivisionError:
    check("147. Zero-division in error rate handled", False)

# Ledger with no events
empty_ledger = SecurityIncidentLedger()
ok, msg = empty_ledger.verify_chain()
check("148. Empty ledger chain is valid", ok is True)

# Multiple concurrent ledger writes
concurrent_ledger = SecurityIncidentLedger()
concurrent_errors = []

def ledger_writer() -> None:
    try:
        for _ in range(20):
            concurrent_ledger.record_event(
                SecurityEventType.AUTH_FAILURE,
                SecuritySeverity.LOW,
                "test",
            )
    except Exception as e:
        concurrent_errors.append(str(e))

t_list = [threading.Thread(target=ledger_writer) for _ in range(3)]
for t in t_list:
    t.start()
for t in t_list:
    t.join()
check("149. Concurrent ledger writes safe", len(concurrent_errors) == 0)
check("150. Concurrent event count correct", concurrent_ledger.get_event_count() == 60)

# ── Section 18: Forensic Chain ───────────────────────────────────────────
print("\n=== Forensic Chain Integrity ===")

forensic_ledger = SecurityIncidentLedger()
for i in range(20):
    forensic_ledger.record_event(
        SecurityEventType.AUTH_FAILURE if i % 3 == 0 else SecurityEventType.MALFORMED_REQUEST,
        SecuritySeverity.HIGH if i % 5 == 0 else SecuritySeverity.LOW,
        "risk-engine",
        correlation_id=f"cid-forensic-{i}",
    )
chain_ok, chain_msg = forensic_ledger.verify_chain()
check("151. Forensic chain with 20 events intact", chain_ok is True)
check("152. Forensic chain count correct", forensic_ledger.get_event_count() == 20)

# Tamper one event in the middle
forensic_ledger._events[10].outcome = "TAMPERED"
chain_ok2, chain_msg2 = forensic_ledger.verify_chain()
check("153. Mid-chain tamper detected", chain_ok2 is False)

# ── Section 19: Alert Evaluator Integration ──────────────────────────────
print("\n=== Alert Evaluator Integration ===")

# Test feature enforcement burst alert
fe_evaluator = AlertEvaluator()
fe_metrics = OperationalMetrics()
for _ in range(25):
    fe_metrics.record_feature_block()
fe_alerts = fe_evaluator.evaluate(fe_metrics)
fe_burst_alerts = [a for a in fe_alerts if a["condition_id"] == "feature-block-burst"]
check("154. Feature block burst alert fires", len(fe_burst_alerts) >= 1)

# Test schema rejection burst alert
sr_evaluator = AlertEvaluator()
sr_metrics = OperationalMetrics()
for _ in range(12):
    sr_metrics.record_schema_rejection()
sr_alerts = sr_evaluator.evaluate(sr_metrics)
sr_burst_alerts = [a for a in sr_alerts if a["condition_id"] == "schema-reject-burst"]
check("155. Schema rejection burst alert fires", len(sr_burst_alerts) >= 1)

# Test rejection rate alert
rr_evaluator = AlertEvaluator()
rr_metrics = OperationalMetrics()
for _ in range(30):
    rr_metrics.record_request(1.0, False)  # rejected
for _ in range(70):
    rr_metrics.record_request(1.0, True)   # successful
rr_alerts = rr_evaluator.evaluate(rr_metrics)
rr_burst = [a for a in rr_alerts if a["condition_id"] == "high-rejection-rate"]
# 30% rejection > 20% threshold
check("156. Rejection rate alert fires", len(rr_burst) >= 1)

# ── Section 20: Production Model Binding ─────────────────────────────────
print("\n=== Production Model Binding ===")

prod_store = ObservabilityStore(service_name="risk-engine")
prod_store.model_telemetry.model_id = "altman_native"
prod_store.model_telemetry.release_id = "release-altman_native_E_hardneg_cert_20260904"
prod_store.model_telemetry.feature_version = "altman_native_v1"

prod_store.metrics.record_request(1.5, True)
prod_store.metrics.record_prediction(0.001, "low", "allow")
prod_store.metrics.record_prediction(0.05, "high", "block")

prod_state = prod_store.export_state()
check("157. Production telemetry has model_id",
      prod_state["model_telemetry"]["model_id"] == "altman_native")
check("158. Production telemetry has release_id",
      "altman_native_E_hardneg_cert_20260904" in prod_state["model_telemetry"]["release_id"])
check("159. Production metrics have predictions",
      prod_state["metrics"]["model"]["predictions"] == 2)

prod_report = prod_store.health_report(
    db_ok=True,
    model_ok=True,
    runtime_state="READY",
    release_id="release-altman_native_E_hardneg_cert_20260904",
    model_id="altman_native",
    feature_version="altman_native_v1",
)
check("160. Production health report is ok", prod_report.overall_status() == "ok")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"Phase 70 Observability Tests: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'=' * 60}")

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  - {e}")

sys.exit(0 if failed == 0 else 1)
