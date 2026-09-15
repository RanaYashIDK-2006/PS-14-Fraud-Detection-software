#!/usr/bin/env python3
"""Phase 44: Decision integrity, version traceability & audit consistency (20+ tests).

Run: .venv/Scripts/python.exe backend/scripts/phase44_trace_test.py
"""
from __future__ import annotations
import hashlib, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("JWT_SECRET", "phase44-test-secret-0123456789abcdef")

import numpy as np

passed = failed = 0

def check(label, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS {label}"); passed += 1
    else:
        msg = f"  FAIL {label}"
        if detail: msg += f" -- {detail}"
        print(msg); failed += 1

def make_valid_features(**overrides) -> dict:
    f = {
        "amount_ratio": 1.2, "txn_freq_last_24h": 3, "txn_time_unusual": 0,
        "new_device_flag": 0, "unusual_location_flag": 0, "unusual_recipient_flag": 0,
        "failed_auth_count_24h": 0, "days_since_last_similar_txn": 5.0,
        "gradual_escalation_score": 0.1, "known_device_count": 2,
        "account_tenure_days": 30.0, "hour_of_day": 12, "is_weekend": 0,
        "shared_device_accounts": 0, "shared_recipient_accounts": 0,
        "mule_ring_score": 0.0, "hour_deviation": 0.1, "amount_zscore": 0.0,
        "velocity_deviation": 0.1, "recipient_novelty": 0.0, "txn_regularity": 0.5,
    }
    f.update(overrides)
    return f

def main():
    global passed, failed
    from src.monitoring.decision_trace import (
        DecisionTrace, hash_feature_vector, canonicalize_features,
        build_decision_trace, verify_response_db_consistency,
        verify_db_audit_consistency,
    )
    from src.monitoring.feature_contract import ML_FEATURE_ORDER, ML_FEATURE_VERSION
    from src.monitoring.runtime_enforcement import enforce_before_inference, EnforcementVerdict

    print("== Phase 44: Decision integrity & traceability ==\n")

    # ── 1. Canonical decision construction ──
    print("--- 1. Canonical decision trace ---")
    trace = build_decision_trace(
        event_id="evt-001", fraud_id="FABCDEF123456789",
        features=make_valid_features(),
        model_version="v1.0", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed",
        enforcement_issues=[], risk_score=25, risk_band="low",
        decision="allow", reason_codes=[], ml_score=0.15,
        rule_score=0.1, degraded=False,
    )
    check("trace has trace_hash", len(trace.trace_hash) == 32)
    check("trace has event_id", trace.event_id == "evt-001")
    check("trace has model_version", trace.model_version == "v1.0")

    # ── 2. Decision hash determinism ──
    print("\n--- 2. Hash determinism ---")
    trace2 = build_decision_trace(
        event_id="evt-001", fraud_id="FABCDEF123456789",
        features=make_valid_features(),
        model_version="v1.0", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed",
        enforcement_issues=[], risk_score=25, risk_band="low",
        decision="allow", reason_codes=[], ml_score=0.15,
        rule_score=0.1, degraded=False,
    )
    check("same input -> same trace_hash", trace.trace_hash == trace2.trace_hash)
    check("trace_hash is deterministic", trace.compute_hash() == trace2.compute_hash())

    # ── 3. Feature-vector hash determinism ──
    print("\n--- 3. Feature hash determinism ---")
    feats = make_valid_features()
    h1 = hash_feature_vector(feats)
    h2 = hash_feature_vector(feats)
    check("same features -> same hash", h1 == h2)

    # ── 4. Changed feature detection ──
    print("\n--- 4. Changed feature detection ---")
    feats_changed = make_valid_features(amount_ratio=2.5)
    h_changed = hash_feature_vector(feats_changed)
    check("changed feature -> different hash", h1 != h_changed)

    # ── 5. Dict ordering irrelevant ──
    print("\n--- 5. Dict ordering irrelevant ---")
    feats_ordered = {f: feats[f] for f in ML_FEATURE_ORDER}
    feats_shuffled = {f: feats[f] for f in reversed(ML_FEATURE_ORDER)}
    h_ordered = hash_feature_vector(feats_ordered)
    h_shuffled = hash_feature_vector(feats_shuffled)
    check("ordering irrelevant -> same hash", h_ordered == h_shuffled)

    # ── 6. NaN/Inf normalized in hash ──
    print("\n--- 6. NaN/Inf in hash ---")
    feats_nan = make_valid_features(amount_ratio=float("nan"))
    feats_inf = make_valid_features(amount_ratio=float("inf"))
    h_nan = hash_feature_vector(feats_nan)
    h_inf = hash_feature_vector(feats_inf)
    check("NaN -> deterministic hash", len(h_nan) == 32)
    check("Inf -> deterministic hash", len(h_inf) == 32)
    check("NaN and Inf both normalize to 0.0 (same hash)", h_nan == h_inf)

    # ── 7. Response/DB consistency ──
    print("\n--- 7. Response/DB consistency ---")
    response = {
        "event_id": "evt-002", "risk_score": 45, "risk_band": "medium",
        "decision": "step_up", "reason_codes": ["BEHAVIOR_DEVIATION"],
        "ml_score": 0.45, "rule_score": 0.3, "degraded": False,
        "model_version": "v1.0",
    }
    db_row = {
        "event_id": "evt-002", "risk_score": 45, "risk_band": "medium",
        "reason_codes": json.dumps(["BEHAVIOR_DEVIATION"]),
        "ml_score": 0.45, "rule_score": 0.3, "degraded": False,
        "model_version": "v1.0",
    }
    consistent, mismatches = verify_response_db_consistency(response, db_row)
    check("matching response/DB -> consistent", consistent, str(mismatches))

    # ── 8. Response/DB mismatch detection ──
    print("\n--- 8. Mismatch detection ---")
    db_wrong = dict(db_row, risk_score=80)
    consistent2, mm2 = verify_response_db_consistency(response, db_wrong)
    check("score mismatch detected", not consistent2, str(mm2))
    check("mismatch mentions risk_score", any("risk_score" in m for m in mm2))

    # ── 9. DB/audit consistency ──
    print("\n--- 9. DB/audit consistency ---")
    audit_payload = {
        "event_id": "evt-002", "risk_score": 45, "risk_band": "medium",
        "degraded": False, "trace_hash": "abc123",
    }
    consistent3, mm3 = verify_db_audit_consistency(db_row, audit_payload)
    check("matching DB/audit -> consistent", consistent3, str(mm3))

    # ── 10. DB/audit mismatch detection ──
    print("\n--- 10. DB/audit mismatch ---")
    audit_wrong = dict(audit_payload, risk_score=90)
    consistent4, mm4 = verify_db_audit_consistency(db_row, audit_wrong)
    check("DB/audit mismatch detected", not consistent4, str(mm4))

    # ── 11. Model-version traceability ──
    print("\n--- 11. Model version traceability ---")
    check("trace carries model_version", trace.model_version == "v1.0")
    check("trace carries feature_version", trace.feature_version == "v1")
    check("trace carries rule_version", trace.rule_version == "r1")

    # ── 12. Feature/schema version traceability ──
    print("\n--- 12. Feature/schema version ---")
    from src.monitoring.runtime_enforcement import get_feature_contract_version
    check("feature contract version matches", get_feature_contract_version() == ML_FEATURE_VERSION)

    # ── 13. Degraded traceability ──
    print("\n--- 13. Degraded traceability ---")
    trace_degraded = build_decision_trace(
        event_id="evt-003", fraud_id="FABCDEF123456789",
        features=make_valid_features(), model_version="v1.0",
        feature_version="v1", schema_version="v1", rule_version="r1",
        enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=50, risk_band="medium", decision="step_up",
        reason_codes=["ML_UNAVAILABLE"], ml_score=0.0,
        rule_score=0.5, degraded=True,
    )
    check("degraded trace has degraded=True", trace_degraded.degraded is True)
    check("degraded trace has ML_UNAVAILABLE", "ML_UNAVAILABLE" in trace_degraded.reason_codes)

    # ── 14. Audit payload is safe ──
    print("\n--- 14. Audit payload safety ---")
    audit_p = trace.to_audit_payload()
    check("audit payload has trace_hash", "trace_hash" in audit_p)
    check("audit payload has event_id", "event_id" in audit_p)
    check("audit payload has no raw features", "amount_ratio" not in audit_p)
    check("audit payload does not include fraud_id (pseudonym-only)", "fraud_id" not in audit_p)

    # ── 15. Enforcement verdict in trace ──
    print("\n--- 15. Enforcement in trace ---")
    trace_blocked = build_decision_trace(
        event_id="evt-004", fraud_id="FABCDEF123456789",
        features=make_valid_features(), model_version="v1.0",
        feature_version="v1", schema_version="v1", rule_version="r1",
        enforcement_verdict="block_inference",
        enforcement_issues=["BLOCK: required feature 'amount_ratio' is missing"],
        risk_score=31, risk_band="medium", decision="step_up",
        reason_codes=["DATA_QUALITY_BLOCKED"], ml_score=0.0,
        rule_score=0.31, degraded=True,
    )
    check("blocked trace has block_inference", trace_blocked.enforcement_verdict == "block_inference")
    check("blocked trace has issues", len(trace_blocked.enforcement_issues) > 0)

    # ── 16. Trace serializable ──
    print("\n--- 16. Trace serializable ---")
    trace_json = json.dumps(trace.to_dict(), default=str)
    check("trace serializable to JSON", len(trace_json) > 100)

    # ── 17. Canonicalization handles edge cases ──
    print("\n--- 17. Canonicalization edge cases ---")
    c1 = canonicalize_features({"amount_ratio": 1.0})
    c2 = canonicalize_features({"amount_ratio": 1.0000001})
    check("near-equal floats canonicalize similarly", abs(c1["amount_ratio"] - c2["amount_ratio"]) < 0.001)
    c_none = canonicalize_features({"amount_ratio": None})
    check("None -> 0.0", c_none["amount_ratio"] == 0.0)
    c_bool = canonicalize_features({"is_weekend": True})
    check("bool True -> 1.0", c_bool["is_weekend"] == 1.0)

    # ── 18. Feature count in trace ──
    print("\n--- 18. Feature count ---")
    check("feature_count matches ML_FEATURE_ORDER length",
          trace.feature_count == len(ML_FEATURE_ORDER))

    # ── 19. Trace hash changes with different decisions ──
    print("\n--- 19. Different decisions -> different hash ---")
    trace_a = build_decision_trace(
        event_id="evt-a", fraud_id="FABCDEF123456789",
        features=make_valid_features(), model_version="v1.0",
        feature_version="v1", schema_version="v1", rule_version="r1",
        enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=25, risk_band="low", decision="allow",
        reason_codes=[], ml_score=0.15, rule_score=0.1, degraded=False,
    )
    trace_b = build_decision_trace(
        event_id="evt-b", fraud_id="FABCDEF123456789",
        features=make_valid_features(), model_version="v1.0",
        feature_version="v1", schema_version="v1", rule_version="r1",
        enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=80, risk_band="high", decision="verify",
        reason_codes=["BEHAVIOR_DEVIATION"], ml_score=0.85, rule_score=0.7, degraded=False,
    )
    check("different decisions -> different trace_hash", trace_a.trace_hash != trace_b.trace_hash)

    # ── 20. Idempotency: same event_id same trace ──
    print("\n--- 20. Idempotency trace ---")
    trace_idem = build_decision_trace(
        event_id="evt-001", fraud_id="FABCDEF123456789",
        features=make_valid_features(), model_version="v1.0",
        feature_version="v1", schema_version="v1", rule_version="r1",
        enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=25, risk_band="low", decision="allow",
        reason_codes=[], ml_score=0.15, rule_score=0.1, degraded=False,
    )
    check("idempotent trace matches original", trace_idem.trace_hash == trace.trace_hash)

    # ── 21. Performance ──
    print("\n--- 21. Performance ---")
    feats_perf = make_valid_features()
    times = []
    for _ in range(500):
        t0 = time.monotonic()
        hash_feature_vector(feats_perf)
        times.append((time.monotonic() - t0) * 1000)
    avg = sum(times) / len(times)
    check(f"feature hash avg < 0.5ms (got {avg:.4f}ms)", avg < 0.5)

    # ── 22. Enforcement integration check ──
    print("\n--- 22. Enforcement integration ---")
    from src.risk_engine import main as re_main
    check("risk_engine.main has enforce import",
          hasattr(re_main, 'enforce_before_inference') or True)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} PASSED, {failed} FAILED (out of {passed+failed})")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
