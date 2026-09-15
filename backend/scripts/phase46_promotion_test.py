#!/usr/bin/env python3
"""Phase 46: Promotion gate adversarial tests.

Proves that the centralized promotion gate prevents ineligible models
from becoming active.  Tests both the gate module directly and the
wired paths in ModelRegistry and model_deploy.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("JWT_SECRET", "phase46-test-secret-0123456789abcdef")

from src.monitoring.promotion_gate import (
    GateStatus,
    PromotionVerdict,
    PromotionBlockedError,
    evaluate_promotion,
    assert_promotion_allowed,
    evaluate_real_world_validation,
    evaluate_model_artifact_binding,
    evaluate_feature_schema_binding,
    evaluate_governance_gates,
    evaluate_security_gates,
    evaluate_drift_status,
    evaluate_external_dataset_eligibility,
    evaluate_rollback_availability,
    MODEL_GOVERNANCE_GATES,
)

passed = 0
failed = 0
total = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")
    return condition


# ── 1. REAL_WORLD_VALIDATION is always BLOCKED ───────────────────────────
print("\n--- 1. Real-world validation gate ---")
g = evaluate_real_world_validation()
check("1a. status is BLOCKED", g.status == GateStatus.BLOCKED)
check("1b. blocking=True", g.blocking is True)
check("1c. reason mentions BLOCKED", "BLOCKED" in g.reason)


# ── 2. Promotion blocked when RWV is BLOCKED ────────────────────────────
print("\n--- 2. Centralized gate blocks promotion when RWV blocked ---")
d = evaluate_promotion(candidate_model_id="test_model_v1")
check("2a. verdict is BLOCKED", d.verdict == PromotionVerdict.BLOCKED)
check("2b. REAL_WORLD_VALIDATION in blocking gates",
      "REAL_WORLD_VALIDATION" in d.blocking_gates)
check("2c. at least 1 blocking gate", len(d.blocking_gates) >= 1)


# ── 3. assert_promotion_allowed raises ───────────────────────────────────
print("\n--- 3. assert_promotion_allowed raises on BLOCKED ---")
raised = False
try:
    assert_promotion_allowed(d)
except PromotionBlockedError as e:
    raised = True
    check("3a. PromotionBlockedError raised", True)
    check("3b. error mentions gate", "gate" in str(e).lower())
except Exception as e:
    check("3a. wrong exception type", False, f"got {type(e).__name__}")
check("3c. exception was raised", raised)


# ── 4. Model artifact binding — mismatch blocks ──────────────────────────
print("\n--- 4. Model artifact binding ---")
g = evaluate_model_artifact_binding("aaa111", "bbb222")
check("4a. mismatch = FAIL", g.status == GateStatus.FAIL)
check("4b. blocking", g.blocking is True)
check("4c. reason mentions mismatch", "differs" in g.reason.lower() or "mismatch" in g.reason.lower())


# ── 5. Model artifact binding — match passes ─────────────────────────────
print("\n--- 5. Model artifact binding — match ---")
g = evaluate_model_artifact_binding("aaa111", "aaa111")
check("5a. match = PASS", g.status == GateStatus.PASS)


# ── 6. Feature schema binding — mismatch blocks ──────────────────────────
print("\n--- 6. Feature schema binding ---")
g = evaluate_feature_schema_binding("v1", "v2")
check("6a. mismatch = FAIL", g.status == GateStatus.FAIL)
check("6b. blocking", g.blocking is True)


# ── 7. Governance gates — failing gates block ────────────────────────────
print("\n--- 7. Governance gates ---")
bad_gates = {g: "UNVERIFIED" for g in MODEL_GOVERNANCE_GATES}
g = evaluate_governance_gates(bad_gates)
check("7a. all UNVERIFIED = FAIL", g.status == GateStatus.FAIL)
check("7b. blocking", g.blocking is True)


# ── 8. Governance gates — all PASS ───────────────────────────────────────
print("\n--- 8. Governance gates — all PASS ---")
good_gates = {g: "PASS" for g in MODEL_GOVERNANCE_GATES}
g = evaluate_governance_gates(good_gates)
check("8a. all PASS = PASS", g.status == GateStatus.PASS)


# ── 9. Governance gates — one FAIL blocks ────────────────────────────────
print("\n--- 9. Governance gates — one FAIL blocks ---")
mixed_gates = {g: "PASS" for g in MODEL_GOVERNANCE_GATES}
mixed_gates["security"] = "FAIL"
g = evaluate_governance_gates(mixed_gates)
check("9a. one FAIL = FAIL", g.status == GateStatus.FAIL)
check("9b. blocking", g.blocking is True)


# ── 10. Security gate — failure blocks ───────────────────────────────────
print("\n--- 10. Security gate ---")
g = evaluate_security_gates(security_passed=False)
check("10a. FAIL", g.status == GateStatus.FAIL)
check("10b. blocking", g.blocking is True)


# ── 11. Security gate — pass ─────────────────────────────────────────────
print("\n--- 11. Security gate — pass ---")
g = evaluate_security_gates(security_passed=True)
check("11a. PASS", g.status == GateStatus.PASS)
check("11b. evidence notes self-authored", "self-authored" in g.evidence.lower())


# ── 12. Drift status — critical blocks ───────────────────────────────────
print("\n--- 12. Drift status ---")
g = evaluate_drift_status(drift_critical=True)
check("12a. FAIL on critical drift", g.status == GateStatus.FAIL)


# ── 13. External dataset — ineligible blocks ─────────────────────────────
print("\n--- 13. External dataset eligibility ---")
g = evaluate_external_dataset_eligibility(eligible=False)
check("13a. FAIL when ineligible", g.status == GateStatus.FAIL)
check("13b. blocking", g.blocking is True)


# ── 14. Full gate — all gates evaluated ──────────────────────────────────
print("\n--- 14. Full gate evaluation ---")
d = evaluate_promotion(
    candidate_model_id="test_v1",
    candidate_artifact_hash="abc123",
    evaluated_artifact_hash="abc123",
    candidate_feature_version="v1",
    evaluated_feature_version="v1",
    governance_gates={g: "PASS" for g in MODEL_GOVERNANCE_GATES},
    security_passed=True,
    drift_critical=False,
    external_dataset_eligible=None,
    has_rollback_source=True,
)
check("14a. still BLOCKED due to RWV", d.verdict == PromotionVerdict.BLOCKED)
check("14b. only RWV is blocking", d.blocking_gates == ["REAL_WORLD_VALIDATION"])
check("14c. all gates present", len(d.gates) == 8)


# ── 15. Serialization roundtrip ──────────────────────────────────────────
print("\n--- 15. Serialization roundtrip ---")
import json
data = d.to_dict()
check("15a. serializable", isinstance(data, dict))
check("15b. has verdict", "verdict" in data)
check("15c. has gates", "gates" in data and len(data["gates"]) == 8)
check("15d. has blocking_gates", "blocking_gates" in data)


# ── 16. ModelRegistry.promote() blocked without gate ─────────────────────
print("\n--- 16. ModelRegistry.promote() with gate decision ---")
from src.risk_engine.model_registry import ModelRegistry, ModelCandidate
with tempfile.TemporaryDirectory() as tmpdir:
    reg = ModelRegistry(Path(tmpdir))
    # Register a candidate
    cand_path = Path(tmpdir) / "cand.joblib"
    cand_path.write_bytes(b"fake model")
    meta_path = Path(tmpdir) / "cand_meta.json"
    meta_path.write_text('{"version": "test"}')
    reg.register_candidate(str(cand_path), str(meta_path))
    check("16a. candidate registered", reg.state.candidate is not None)
    # Try to promote with BLOCKED gate
    promoted = False
    try:
        reg.promote(gate_decision=d)
        promoted = True
    except RuntimeError as e:
        check("16b. RuntimeError raised", "BLOCKED" in str(e))
    check("16c. promotion did NOT happen", not promoted)
    check("16d. candidate still exists", reg.state.candidate is not None)


# ── 17. ModelRegistry.promote() blocked with PromotionDecision ───────────
print("\n--- 17. ModelRegistry.promote() with PromotionDecision ---")
from src.monitoring.promotion_gate import PromotionDecision
with tempfile.TemporaryDirectory() as tmpdir:
    reg = ModelRegistry(Path(tmpdir))
    cand_path = Path(tmpdir) / "cand.joblib"
    cand_path.write_bytes(b"fake model")
    meta_path = Path(tmpdir) / "cand_meta.json"
    meta_path.write_text('{"version": "test"}')
    reg.register_candidate(str(cand_path), str(meta_path))
    blocked_d = PromotionDecision(verdict=PromotionVerdict.BLOCKED,
                                   blocking_gates=["REAL_WORLD_VALIDATION"])
    raised = False
    try:
        reg.promote(gate_decision=blocked_d)
    except RuntimeError:
        raised = True
    check("17a. RuntimeError on BLOCKED", raised)
    check("17b. candidate not promoted", reg.state.candidate is not None)


# ── 18. Artifact mismatch blocks ─────────────────────────────────────────
print("\n--- 18. Artifact mismatch blocks full gate ---")
d2 = evaluate_promotion(
    candidate_model_id="test_v2",
    candidate_artifact_hash="aaa111",
    evaluated_artifact_hash="bbb222",
)
check("18a. BLOCKED", d2.verdict == PromotionVerdict.BLOCKED)
check("18b. MODEL_ARTIFACT_BINDING in blocking",
      "MODEL_ARTIFACT_BINDING" in d2.blocking_gates)


# ── 19. Feature version mismatch blocks ──────────────────────────────────
print("\n--- 19. Feature version mismatch blocks ---")
d3 = evaluate_promotion(
    candidate_feature_version="v1",
    evaluated_feature_version="v2",
)
check("19a. BLOCKED", d3.verdict == PromotionVerdict.BLOCKED)
check("19b. FEATURE_SCHEMA_BINDING in blocking",
      "FEATURE_SCHEMA_BINDING" in d3.blocking_gates)


# ── 20. Governance failure blocks ────────────────────────────────────────
print("\n--- 20. Governance failure blocks ---")
d4 = evaluate_promotion(
    governance_gates={g: "UNVERIFIED" for g in MODEL_GOVERNANCE_GATES},
)
check("20a. BLOCKED", d4.verdict == PromotionVerdict.BLOCKED)
check("20b. MODEL_GOVERNANCE in blocking",
      "MODEL_GOVERNANCE" in d4.blocking_gates)


# ── 21. Security failure blocks ──────────────────────────────────────────
print("\n--- 21. Security failure blocks ---")
d5 = evaluate_promotion(security_passed=False)
check("21a. BLOCKED", d5.verdict == PromotionVerdict.BLOCKED)
check("21b. SECURITY_CI in blocking", "SECURITY_CI" in d5.blocking_gates)


# ── 22. Multiple simultaneous failures ───────────────────────────────────
print("\n--- 22. Multiple simultaneous failures ---")
d6 = evaluate_promotion(
    candidate_artifact_hash="aaa",
    evaluated_artifact_hash="bbb",
    governance_gates={g: "FAIL" for g in MODEL_GOVERNANCE_GATES},
    security_passed=False,
)
check("22a. BLOCKED", d6.verdict == PromotionVerdict.BLOCKED)
check("22b. >= 3 blocking gates", len(d6.blocking_gates) >= 3)


# ── 23. NOT_EVALUATED does NOT equal PASS ────────────────────────────────
print("\n--- 23. NOT_EVALUATED != PASS ---")
g = evaluate_governance_gates(None)
check("23a. NOT_EVALUATED when no record", g.status == GateStatus.NOT_EVALUATED)
check("23b. blocking when not evaluated", g.blocking is True)


# ── 24. Backward-compatible promote() without gate still works ───────────
print("\n--- 24. Backward-compatible promote() without gate ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg = ModelRegistry(Path(tmpdir))
    cand_path = Path(tmpdir) / "cand.joblib"
    cand_path.write_bytes(b"fake model")
    meta_path = Path(tmpdir) / "cand_meta.json"
    meta_path.write_text('{"version": "test"}')
    reg.register_candidate(str(cand_path), str(meta_path))
    # promote() without gate_decision = backward-compatible
    reg.promote()  # should not raise
    check("24a. promote() without gate succeeded", reg.state.candidate is None)
    check("24b. mode is direct", reg.state.mode == "direct")


# ── 25. File save roundtrip ──────────────────────────────────────────────
print("\n--- 25. File save roundtrip ---")
with tempfile.TemporaryDirectory() as tmpdir:
    p = Path(tmpdir) / "decision.json"
    d.save(p)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    check("25a. file saved", p.exists())
    check("25b. verdict preserved", loaded["verdict"] == "PROMOTION_BLOCKED")
    check("25c. gates count", len(loaded["gates"]) == 8)
    check("25d. blocking_gates preserved",
          "REAL_WORLD_VALIDATION" in loaded["blocking_gates"])


# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Phase 46 promotion gate tests: {passed}/{total} passed")
if failed:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("  ALL PASSED")
    sys.exit(0)
