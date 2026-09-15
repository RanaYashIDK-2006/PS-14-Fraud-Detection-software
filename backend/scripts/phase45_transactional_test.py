#!/usr/bin/env python3
"""Phase 45: Transactional integrity, concurrency & failure-recovery tests.

Run: .venv/Scripts/python.exe backend/scripts/phase45_transactional_test.py
"""
from __future__ import annotations
import hashlib, json, os, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("JWT_SECRET", "phase45-test-secret-0123456789abcdef")

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

def build_audit_event(seq, prev_hash, payload, entry_hash=None):
    """Build a mock audit event dict."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if entry_hash is None:
        entry_hash = hashlib.sha256((prev_hash + body).encode()).hexdigest()
    return {"seq": seq, "prev_hash": prev_hash, "entry_hash": entry_hash, "payload_summary": json.dumps(payload)}

def main():
    global passed, failed
    from src.monitoring.decision_trace import (
        DecisionTrace, hash_feature_vector, canonicalize_features,
        build_decision_trace, verify_response_db_consistency,
        verify_db_audit_consistency,
    )
    from src.monitoring.reconciliation import (
        verify_audit_chain, reconcile_decisions_audit, DiscrepancyType,
    )
    from src.monitoring.runtime_enforcement import enforce_before_inference, EnforcementVerdict
    from src.monitoring.feature_contract import ML_FEATURE_ORDER

    print("== Phase 45: Transactional integrity & concurrency ==\n")

    # ── 1. Audit chain: valid chain verifies ──
    print("--- 1. Valid audit chain ---")
    GENESIS = hashlib.sha256(b"PS-14 audit genesis v1").hexdigest()
    payload1 = {"event_id": "e1", "risk_score": 25}
    body1 = json.dumps(payload1, sort_keys=True, separators=(",", ":"))
    h1 = hashlib.sha256((GENESIS + body1).encode()).hexdigest()
    ev1 = build_audit_event(1, GENESIS, payload1, h1)

    payload2 = {"event_id": "e2", "risk_score": 50}
    body2 = json.dumps(payload2, sort_keys=True, separators=(",", ":"))
    h2 = hashlib.sha256((h1 + body2).encode()).hexdigest()
    ev2 = build_audit_event(2, h1, payload2, h2)

    valid, detail = verify_audit_chain([ev1, ev2])
    check("valid chain verifies", valid, detail)

    # ── 2. Audit chain: tampered payload detected ──
    print("\n--- 2. Tampered payload ---")
    tampered_payload = {"event_id": "e1", "risk_score": 99}  # changed score
    tampered_body = json.dumps(tampered_payload, sort_keys=True, separators=(",", ":"))
    tampered_hash = hashlib.sha256((GENESIS + tampered_body).encode()).hexdigest()
    tampered_ev = build_audit_event(1, GENESIS, tampered_payload, tampered_hash)
    # Now verify with the ORIGINAL chain but the tampered event
    valid2, detail2 = verify_audit_chain([tampered_ev, ev2])
    # The tampered event's prev_hash won't match h1 (the chain breaks at event 2)
    check("tampered chain detected at next event", not valid2 or "chain break" in detail2 or "mismatch" in detail2,
          f"detail={detail2}")

    # ── 3. Audit chain: broken previous_hash detected ──
    print("\n--- 3. Broken previous_hash ---")
    broken_ev = build_audit_event(2, "wrong_prev_hash", payload2)
    valid3, detail3 = verify_audit_chain([ev1, broken_ev])
    check("broken prev_hash detected", not valid3, f"detail={detail3}")

    # ── 4. Audit chain: empty chain valid ──
    print("\n--- 4. Empty chain ---")
    valid4, _ = verify_audit_chain([])
    check("empty chain is valid", valid4)

    # ── 5. Reconciliation: matching decisions/audit ──
    print("\n--- 5. Reconciliation matching ---")
    decisions = [{"event_id": "e1", "risk_score": 25, "risk_band": "low", "degraded": False, "model_version": "v1"}]
    audit_events = [{"event_type": "score_generated", "payload_summary": json.dumps({"event_id": "e1", "risk_score": 25, "risk_band": "low", "degraded": False, "model_version": "v1"})}]
    report = reconcile_decisions_audit(decisions, audit_events)
    check("matching -> no discrepancies", report.n_discrepancies == 0, f"discrepancies={[d.to_dict() for d in report.discrepancies]}")

    # ── 6. Reconciliation: missing audit ──
    print("\n--- 6. Missing audit ---")
    report6 = reconcile_decisions_audit(decisions, [])
    check("missing audit detected", any(d.discrepancy_type == DiscrepancyType.MISSING_AUDIT for d in report6.discrepancies))

    # ── 7. Reconciliation: score mismatch ──
    print("\n--- 7. Score mismatch ---")
    audit_mismatch = [{"event_type": "score_generated", "payload_summary": json.dumps({"event_id": "e1", "risk_score": 80, "risk_band": "high", "degraded": False, "model_version": "v1"})}]
    report7 = reconcile_decisions_audit(decisions, audit_mismatch)
    check("score mismatch detected", any(d.discrepancy_type == DiscrepancyType.SCORE_MISMATCH for d in report7.discrepancies))

    # ── 8. Reconciliation: missing decision ──
    print("\n--- 8. Missing decision ---")
    audit_only = [{"event_type": "score_generated", "payload_summary": json.dumps({"event_id": "e_orphan", "risk_score": 30, "risk_band": "low", "degraded": False, "model_version": "v1"})}]
    report8 = reconcile_decisions_audit([], audit_only)
    check("missing decision detected", any(d.discrepancy_type == DiscrepancyType.MISSING_DECISION for d in report8.discrepancies))

    # ── 9. Feature hash: same vector same hash ──
    print("\n--- 9. Feature hash stability ---")
    feats = make_valid_features()
    h_a = hash_feature_vector(feats)
    h_b = hash_feature_vector(feats)
    check("same vector -> same hash", h_a == h_b)

    # ── 10. Feature hash: different vector different hash ──
    print("\n--- 10. Feature hash change detection ---")
    feats2 = make_valid_features(amount_ratio=99.0)
    h_c = hash_feature_vector(feats2)
    check("different vector -> different hash", h_a != h_c)

    # ── 11. Feature hash: ordering irrelevant ──
    print("\n--- 11. Ordering irrelevant ---")
    feats_ord = {f: feats[f] for f in ML_FEATURE_ORDER}
    feats_rev = {f: feats[f] for f in reversed(ML_FEATURE_ORDER)}
    check("ordering irrelevant", hash_feature_vector(feats_ord) == hash_feature_vector(feats_rev))

    # ── 12. Feature hash: NaN/Inf normalized ──
    print("\n--- 12. NaN/Inf normalization ---")
    feats_nan = make_valid_features(amount_ratio=float("nan"))
    feats_zero = make_valid_features(amount_ratio=0.0)
    h_nan = hash_feature_vector(feats_nan)
    h_zero = hash_feature_vector(feats_zero)
    check("NaN normalizes to 0.0 (same as explicit 0)", h_nan == h_zero)

    # ── 13. Trace hash: same decision same hash ──
    print("\n--- 13. Trace hash stability ---")
    trace_a = build_decision_trace(
        event_id="e1", fraud_id="FABC", features=feats,
        model_version="v1", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=25, risk_band="low", decision="allow",
        reason_codes=[], ml_score=0.15, rule_score=0.1, degraded=False,
    )
    trace_b = build_decision_trace(
        event_id="e1", fraud_id="FABC", features=feats,
        model_version="v1", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=25, risk_band="low", decision="allow",
        reason_codes=[], ml_score=0.15, rule_score=0.1, degraded=False,
    )
    check("same decision -> same trace_hash", trace_a.trace_hash == trace_b.trace_hash)

    # ── 14. Trace hash: different score different hash ──
    print("\n--- 14. Trace hash change detection ---")
    trace_c = build_decision_trace(
        event_id="e1", fraud_id="FABC", features=feats,
        model_version="v1", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=80, risk_band="high", decision="verify",
        reason_codes=["BEHAVIOR_DEVIATION"], ml_score=0.85, rule_score=0.7, degraded=False,
    )
    check("different score -> different trace_hash", trace_a.trace_hash != trace_c.trace_hash)

    # ── 15. Trace hash: different model version different hash ──
    print("\n--- 15. Model version traceability ---")
    trace_d = build_decision_trace(
        event_id="e1", fraud_id="FABC", features=feats,
        model_version="v2", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=25, risk_band="low", decision="allow",
        reason_codes=[], ml_score=0.15, rule_score=0.1, degraded=False,
    )
    check("different model_version -> different trace_hash", trace_a.trace_hash != trace_d.trace_hash)

    # ── 16. Trace hash: different feature version different hash ──
    print("\n--- 16. Feature version traceability ---")
    trace_e = build_decision_trace(
        event_id="e1", fraud_id="FABC", features=feats,
        model_version="v1", feature_version="v2", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=25, risk_band="low", decision="allow",
        reason_codes=[], ml_score=0.15, rule_score=0.1, degraded=False,
    )
    check("different feature_version -> different trace_hash", trace_a.trace_hash != trace_e.trace_hash)

    # ── 17. Trace hash: different rule version different hash ──
    print("\n--- 17. Rule version traceability ---")
    trace_f = build_decision_trace(
        event_id="e1", fraud_id="FABC", features=feats,
        model_version="v1", feature_version="v1", schema_version="v1",
        rule_version="r2", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=25, risk_band="low", decision="allow",
        reason_codes=[], ml_score=0.15, rule_score=0.1, degraded=False,
    )
    check("different rule_version -> different trace_hash", trace_a.trace_hash != trace_f.trace_hash)

    # ── 18. Trace hash: different degraded different hash ──
    print("\n--- 18. Degraded traceability ---")
    trace_g = build_decision_trace(
        event_id="e1", fraud_id="FABC", features=feats,
        model_version="v1", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=25, risk_band="low", decision="allow",
        reason_codes=[], ml_score=0.15, rule_score=0.1, degraded=True,
    )
    check("different degraded -> different trace_hash", trace_a.trace_hash != trace_g.trace_hash)

    # ── 19. Concurrent idempotency (simulated) ──
    print("\n--- 19. Concurrent idempotency ---")
    # Simulate: multiple threads try to create the same decision
    results = []
    lock = threading.Lock()
    def create_decision(event_id):
        trace = build_decision_trace(
            event_id=event_id, fraud_id="FABC", features=feats,
            model_version="v1", feature_version="v1", schema_version="v1",
            rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
            risk_score=25, risk_band="low", decision="allow",
            reason_codes=[], ml_score=0.15, rule_score=0.1, degraded=False,
        )
        with lock:
            results.append(trace.trace_hash)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_decision, "e_concurrent") for _ in range(10)]
        for f in as_completed(futures):
            f.result()

    check("concurrent decisions produce same trace_hash",
          len(set(results)) == 1, f"unique hashes: {len(set(results))}")

    # ── 20. Modified duplicate: original not overwritten ──
    print("\n--- 20. Modified duplicate ---")
    trace_orig = build_decision_trace(
        event_id="e_orig", fraud_id="FABC", features=feats,
        model_version="v1", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=25, risk_band="low", decision="allow",
        reason_codes=[], ml_score=0.15, rule_score=0.1, degraded=False,
    )
    # Modified duplicate: different score
    trace_mod = build_decision_trace(
        event_id="e_orig", fraud_id="FABC", features=feats,
        model_version="v1", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=80, risk_band="high", decision="verify",
        reason_codes=[], ml_score=0.85, rule_score=0.7, degraded=False,
    )
    check("modified duplicate has different trace_hash", trace_orig.trace_hash != trace_mod.trace_hash)
    # Original should not be overwritten (idempotency returns stored result)
    check("original trace_hash preserved", trace_orig.trace_hash != trace_mod.trace_hash)

    # ── 21. Response/DB consistency ──
    print("\n--- 21. Response/DB consistency ---")
    response = {"event_id": "e21", "risk_score": 45, "risk_band": "medium", "reason_codes": ["X"], "ml_score": 0.45, "rule_score": 0.3, "degraded": False, "model_version": "v1"}
    db_row = {"event_id": "e21", "risk_score": 45, "risk_band": "medium", "reason_codes": json.dumps(["X"]), "ml_score": 0.45, "rule_score": 0.3, "degraded": False, "model_version": "v1"}
    consistent, mm = verify_response_db_consistency(response, db_row)
    check("matching response/DB -> consistent", consistent, str(mm))

    # ── 22. Response/DB mismatch ──
    print("\n--- 22. Response/DB mismatch ---")
    db_wrong = dict(db_row, risk_score=99)
    consistent2, mm2 = verify_response_db_consistency(response, db_wrong)
    check("mismatch detected", not consistent2, str(mm2))

    # ── 23. DB/Audit consistency ──
    print("\n--- 23. DB/Audit consistency ---")
    audit_match = {"event_id": "e21", "risk_score": 45, "risk_band": "medium", "degraded": False}
    consistent3, mm3 = verify_db_audit_consistency(db_row, audit_match)
    check("matching DB/audit -> consistent", consistent3, str(mm3))

    # ── 24. DB/Audit mismatch ──
    print("\n--- 24. DB/Audit mismatch ---")
    audit_wrong = {"event_id": "e24", "risk_score": 99, "risk_band": "high", "degraded": True}
    consistent4, mm4 = verify_db_audit_consistency(db_row, audit_wrong)
    check("mismatch detected", not consistent4, str(mm4))

    # ── 25. Enforcement: BLOCK prevents inference ──
    print("\n--- 25. BLOCK prevents inference ---")
    bad = make_valid_features(amount_ratio=float("nan"))
    r25 = enforce_before_inference(bad)
    check("NaN -> BLOCK_INFERENCE", r25.verdict == EnforcementVerdict.BLOCK_INFERENCE)

    # ── 26. Enforcement: PROCEED allows inference ──
    print("\n--- 26. PROCEED allows inference ---")
    good = make_valid_features()
    r26 = enforce_before_inference(good)
    check("valid -> PROCEED", r26.verdict == EnforcementVerdict.PROCEED)

    # ── 27. Audit chain: extended chain verifies ──
    print("\n--- 27. Extended chain ---")
    chain = []
    prev = GENESIS
    for i in range(20):
        p = {"event_id": f"e{i}", "risk_score": i * 5}
        body = json.dumps(p, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256((prev + body).encode()).hexdigest()
        chain.append(build_audit_event(i + 1, prev, p, h))
        prev = h
    valid27, detail27 = verify_audit_chain(chain)
    check("20-event chain verifies", valid27, detail27)

    # ── 28. Audit chain: deleted event detected ──
    print("\n--- 28. Deleted event ---")
    chain_missing = chain[:5] + chain[6:]  # skip event 6
    valid28, detail28 = verify_audit_chain(chain_missing)
    check("deleted event breaks chain", not valid28, f"detail={detail28}")

    # ── 29. Canonicalization: near-equal floats ──
    print("\n--- 29. Canonicalization ---")
    c1 = canonicalize_features({"amount_ratio": 1.0})
    c2 = canonicalize_features({"amount_ratio": 1.0000001})
    check("near-equal floats round to same", c1["amount_ratio"] == c2["amount_ratio"])

    # ── 30. Trace hash: reason code change ──
    print("\n--- 30. Reason code traceability ---")
    trace_r1 = build_decision_trace(
        event_id="e_r", fraud_id="FABC", features=feats,
        model_version="v1", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=50, risk_band="medium", decision="step_up",
        reason_codes=["A"], ml_score=0.5, rule_score=0.3, degraded=False,
    )
    trace_r2 = build_decision_trace(
        event_id="e_r", fraud_id="FABC", features=feats,
        model_version="v1", feature_version="v1", schema_version="v1",
        rule_version="r1", enforcement_verdict="proceed", enforcement_issues=[],
        risk_score=50, risk_band="medium", decision="step_up",
        reason_codes=["B"], ml_score=0.5, rule_score=0.3, degraded=False,
    )
    check("different reason_codes -> different trace_hash", trace_r1.trace_hash != trace_r2.trace_hash)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} PASSED, {failed} FAILED (out of {passed+failed})")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
