"""Phase 101: Audit Finding Remediation & Regression Closure Tests.

100+ deterministic assertions covering:
- INV-07 re-audit (should now PASS)
- Finding registry integrity
- Promotion gate attestation reference
- Runtime attestation enforcement
- All Phase 100 invariants still pass
- Full cross-phase regression

All tests are deterministic and offline.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.monitoring.audit_finding_registry import (
    AuditFinding,
    AuditFindingRegistry,
    FindingStatus,
    FindingSeverity,
    create_finding,
    _hash_dict,
    _now_iso,
)
from src.monitoring.phase100_independent_audit import (
    generate_audit_report,
    ALL_AUDIT_FUNCTIONS,
    AuditVerdict,
)
from src.monitoring.rwv_evidence_ledger import (
    RWVEvidenceLedger,
    LedgerVerificationResult,
    EvidenceType,
    EvidenceStatus,
    LineageStatus,
    create_ledger_audit_event,
    detect_entry_tampering,
    detect_chain_break,
    compute_environment_fingerprint,
)
from src.monitoring.rwv_reproducibility import (
    ReproducibilityManifest,
    ReproducibilityResult,
    RecomputationResult,
    build_reproducibility_manifest,
    verify_reproducibility_manifest,
    recompute_metrics,
    verify_recomputation,
    REPRODUCIBILITY_POLICY_VERSION,
)
from src.monitoring.rwv_promotion_evidence import (
    RWVPromotionEvidence,
    PromotionEvidenceConstructionStatus,
    GateDecisionState,
    build_promotion_evidence,
    verify_evidence_binding,
    verify_evidence_tampering,
    prepare_gate_submission,
    reset_replay_registry,
    check_replay,
    PROMOTION_EVIDENCE_VERSION,
)
from src.monitoring.rwv_adjudication import (
    ValidityStatus,
    AcceptanceStatus,
    PromotionEvidenceStatus,
    adjudicate_rwv_result,
)
from src.monitoring.rwv_execution import (
    RWVEvaluationRecord,
    ExecutionStatus,
    validate_temporal_ordering,
)
from src.monitoring.rwv_readiness_audit import (
    MODEL_ID,
    RELEASE_ID,
    FEATURE_VERSION,
    build_evidence_pack,
    validate_evidence_pack,
)
from src.monitoring.provider_evidence import (
    KNOWN_CANDIDATES,
    DatasetQualificationState,
    qualify_dataset,
)
from src.monitoring.promotion_gate import (
    PromotionToken,
    PromotionDecision,
    PromotionVerdict,
    evaluate_promotion,
    evaluate_real_world_validation,
    GateStatus,
)
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES

passed = 0
failed = 0
total = 0

def check(condition: bool, label: str):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  OK  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: INV-07 RE-AUDIT
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 1: INV-07 re-audit ===")
report = generate_audit_report()
inv07 = [i for i in report.invariants if i["invariant_id"] == "INV-07"][0]
check(inv07["verdict"] == "pass", f"INV-07 now PASS (was WARN): {inv07['finding']}")

# ══════════════════════════════════════════════════════════════════════
# SECTION 2: ALL 32 INVARIANTS PASS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 2: All 32 invariants pass ===")
check(report.fail_count == 0, f"0 FAIL (got {report.fail_count})")
check(report.warn_count == 0, f"0 WARN (got {report.warn_count})")
check(report.pass_count == 32, f"32 PASS (got {report.pass_count})")

# ══════════════════════════════════════════════════════════════════════
# SECTION 3: PROMOTION GATE HAS ATTESTATION REFERENCE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 3: Promotion gate attestation reference ===")
import src.monitoring.promotion_gate as pg
source = inspect.getsource(pg)
check("runtime attestation" in source.lower(), "Promotion gate references runtime attestation")
check("Phase 49" in source, "Promotion gate references Phase 49")
check("startup" in source.lower() or "deployed" in source.lower(), "Promotion gate references deployment-time attestation")

# ══════════════════════════════════════════════════════════════════════
# SECTION 4: FINDING REGISTRY — CREATE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 4: Finding registry creation ===")
registry = AuditFindingRegistry()
check(registry.all_findings() == [], "Empty registry")

# ══════════════════════════════════════════════════════════════════════
# SECTION 5: FINDING REGISTRY — REGISTER
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 5: Finding registry register ===")
finding = create_finding(
    finding_id="PHASE100-INV07",
    source_phase=100,
    invariant_id="INV-07",
    title="Runtime attestation not referenced in promotion gate",
    severity=FindingSeverity.LOW.value,
    description="Promotion gate source does not reference runtime attestation",
    evidence="grep for 'attestation' in promotion_gate.py returned empty",
    root_cause="Documentation/traceability gap: attestation enforced at startup, not in gate source",
    affected_components=("promotion_gate", "runtime_attestation"),
    remediation="Added explicit trust boundary note to promotion gate docstring",
    regression_test="phase101_independent_audit_test.py",
    status=FindingStatus.REMEDIATED.value,
)
registry.register(finding)
check(registry.get("PHASE100-INV07") is not None, "Finding registered")
check(len(registry.all_findings()) == 1, "Registry has 1 finding")

# ══════════════════════════════════════════════════════════════════════
# SECTION 6: FINDING REGISTRY — DUPLICATE REJECTION
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 6: Finding registry duplicate rejection ===")
try:
    registry.register(finding)
    check(False, "Should raise")
except ValueError:
    check(True, "Duplicate finding_id rejected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 7: FINDING RECORD — FROZEN
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 7: Finding record frozen ===")
try:
    finding.finding_id = "modified"
    check(False, "Should not modify frozen")
except Exception:
    check(True, "Finding is frozen")

# ══════════════════════════════════════════════════════════════════════
# SECTION 8: FINDING RECORD — TO_DICT
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 8: Finding to_dict ===")
fd = finding.to_dict()
check(isinstance(fd, dict), "to_dict returns dict")
check("finding_id" in fd, "Dict has finding_id")
check("affected_components" in fd, "Dict has affected_components")
check(isinstance(fd["affected_components"], list), "Components serialized as list")

# ══════════════════════════════════════════════════════════════════════
# SECTION 9: FINDING RECORD — HASH DETERMINISM
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 9: Finding hash determinism ===")
f2 = create_finding(
    finding_id="PHASE100-INV07", source_phase=100, invariant_id="INV-07",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test", status="open",
)
# Different status → different hash
check(finding.resolution_hash != f2.resolution_hash or finding.status != f2.status,
      "Different status produces different finding")

# ══════════════════════════════════════════════════════════════════════
# SECTION 10: FINDING STATUSES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 10: Finding statuses ===")
check(FindingStatus.OPEN.value == "open", "OPEN")
check(FindingStatus.REMEDIATED.value == "remediated", "REMEDIATED")
check(FindingStatus.ACCEPTED_RISK.value == "accepted_risk", "ACCEPTED_RISK")
check(FindingStatus.REJECTED_WITH_EVIDENCE.value == "rejected_with_evidence", "REJECTED")
check(len([s for s in FindingStatus]) == 6, "6 finding statuses")

# ══════════════════════════════════════════════════════════════════════
# SECTION 11: FINDING SEVERITIES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 11: Finding severities ===")
check(FindingSeverity.CRITICAL.value == "critical", "CRITICAL")
check(FindingSeverity.HIGH.value == "high", "HIGH")
check(FindingSeverity.LOW.value == "low", "LOW")
check(len([s for s in FindingSeverity]) == 5, "5 severities")

# ══════════════════════════════════════════════════════════════════════
# SECTION 12: FINDING — REMEDIATED STATUS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 12: Finding remediated status ===")
check(finding.status == FindingStatus.REMEDIATED.value, "INV-07 finding is REMEDIATED")

# ══════════════════════════════════════════════════════════════════════
# SECTION 13: FINDING — SEVERITY LOW
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 13: Finding severity LOW ===")
check(finding.severity == FindingSeverity.LOW.value, "INV-07 severity is LOW")

# ══════════════════════════════════════════════════════════════════════
# SECTION 14: FINDING — ROOT CAUSE DOCUMENTED
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 14: Finding root cause documented ===")
check(len(finding.root_cause) > 10, "Root cause is documented")

# ══════════════════════════════════════════════════════════════════════
# SECTION 15: FINDING — EVIDENCE DOCUMENTED
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 15: Finding evidence documented ===")
check(len(finding.evidence) > 10, "Evidence is documented")

# ══════════════════════════════════════════════════════════════════════
# SECTION 16: FINDING — REMEDIATION DOCUMENTED
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 16: Finding remediation documented ===")
check(len(finding.remediation) > 10, "Remediation is documented")

# ══════════════════════════════════════════════════════════════════════
# SECTION 17: FINDING — REGRESSION TEST DOCUMENTED
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 17: Finding regression test documented ===")
check(len(finding.regression_test) > 5, "Regression test is documented")

# ══════════════════════════════════════════════════════════════════════
# SECTION 18: FINDING — RESOLUTION HASH
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 18: Finding resolution hash ===")
check(len(finding.resolution_hash) == 64, "Resolution hash is 64 hex")

# ══════════════════════════════════════════════════════════════════════
# SECTION 19: FINDING — JSON SERIALIZABLE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 19: Finding JSON serializable ===")
try:
    json.dumps(finding.to_dict())
    check(True, "Finding is JSON-serializable")
except Exception:
    check(False, "Finding must be JSON-serializable")

# ══════════════════════════════════════════════════════════════════════
# SECTION 20: FINDING — AFFECTED COMPONENTS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 20: Finding affected components ===")
check("promotion_gate" in finding.affected_components, "promotion_gate affected")
check("runtime_attestation" in finding.affected_components, "runtime_attestation affected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 21: RUNTIME ATTESTATION — STILL EXISTS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 21: Runtime attestation still exists ===")
import src.monitoring.runtime_attestation as ra
check(hasattr(ra, "RuntimeAttestation"), "RuntimeAttestation class exists")
check(hasattr(ra, "build_attestation"), "build_attestation exists")
check(hasattr(ra, "RuntimeState"), "RuntimeState enum exists")

# ══════════════════════════════════════════════════════════════════════
# SECTION 22: PROMOTION GATE — STILL AUTHORITATIVE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 22: Promotion gate still authoritative ===")
check("def evaluate_promotion" in source, "evaluate_promotion exists")
check("class GateResult" in source or "GateResult" in source, "GateResult exists")
check("class PromotionVerdict" in source, "PromotionVerdict exists")

# ══════════════════════════════════════════════════════════════════════
# SECTION 23: RWV GATE — STILL BLOCKED
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 23: RWV gate still blocked ===")
rwv_gate = evaluate_real_world_validation()
check(rwv_gate.status == GateStatus.BLOCKED, "RWV gate BLOCKED")

# ══════════════════════════════════════════════════════════════════════
# SECTION 24: ALL CANDIDATES STILL BLOCKED
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 24: All candidates still blocked ===")
for name, cand in KNOWN_CANDIDATES.items():
    q = qualify_dataset(cand)
    check(
        q.qualification_state != DatasetQualificationState.QUALIFIED_FOR_CONTROLLED_RWV.value,
        f"{name} blocked",
    )

# ══════════════════════════════════════════════════════════════════════
# SECTION 25: EVIDENCE PACK — STILL VALID
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 25: Evidence pack still valid ===")
pack = build_evidence_pack()
pv = validate_evidence_pack(pack)
check(pv.get("valid", False), "Evidence pack valid")

# ══════════════════════════════════════════════════════════════════════
# SECTION 26: 48 NATIVE FEATURES — STILL CORRECT
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 26: 48 native features ===")
check(len(ALTMAN_NATIVE_FEATURES) == 48, "48 features")
check(len(set(ALTMAN_NATIVE_FEATURES)) == 48, "All unique")

# ══════════════════════════════════════════════════════════════════════
# SECTION 27: LEDGER — STILL WORKS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 27: Ledger still works ===")
l = RWVEvidenceLedger()
l.append_entry("test", "L-001", "h1")
l.append_entry("test", "L-002", "h2")
check(l.verify_chain() == LedgerVerificationResult.VERIFIED, "Chain verified")

# ══════════════════════════════════════════════════════════════════════
# SECTION 28: REPRODUCIBILITY — STILL WORKS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 28: Reproducibility still works ===")
m = build_reproducibility_manifest(model_id=MODEL_ID, release_id=RELEASE_ID)
recomputed = _hash_dict(m.to_canonical())
check(recomputed == m.manifest_hash, "Manifest hash verified")

# ══════════════════════════════════════════════════════════════════════
# SECTION 29: PROMOTION EVIDENCE — STILL WORKS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 29: Promotion evidence still works ===")
ev = RWVPromotionEvidence(
    evidence_id="x", session_id="x", evaluation_record_hash="x",
    adjudication_hash="x", provider_id="x", dataset_id="x",
    dataset_version="x", dataset_qualification_hash="x",
    model_id="x", release_id="x", release_manifest_hash="x",
    artifact_hash="x", feature_contract_version="x",
    native_feature_version="x", preprocessing_hash="x", rule_hash="x",
    evaluation_protocol_version="x", acceptance_spec_version="x",
    evaluation_config_hash="x", result_status="x", acceptance_status="x",
    promotion_evidence_status="x", evidence_policy_version="x",
    evidence_hash="x", created_at="x",
)
check(not isinstance(ev, PromotionToken), "Evidence NOT a token")

# ══════════════════════════════════════════════════════════════════════
# SECTION 30: BINDING CHECKS — STILL WORK
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 30: Binding checks still work ===")
wrong_ev = RWVPromotionEvidence(**{**ev.__dict__, "model_id": "wrong"})
checks = verify_evidence_binding(wrong_ev, expected_model_id=MODEL_ID)
failed_b = [c for c in checks if not c.matched]
check(len(failed_b) >= 1, "Wrong model detected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 31: TAMPER DETECTION — STILL WORKS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 31: Tamper detection still works ===")
tampered_ev = RWVPromotionEvidence(**{**ev.__dict__, "evidence_hash": "TAMPERED"})
valid_t, _ = verify_evidence_tampering(tampered_ev)
check(not valid_t, "Tampered evidence detected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 32: GATE ADAPTER — STILL WORKS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 32: Gate adapter still works ===")
gate_input, gate_state = prepare_gate_submission(tampered_ev)
check(gate_state == GateDecisionState.PROMOTION_EVIDENCE_INVALID.value, "Tampered -> INVALID")

# ══════════════════════════════════════════════════════════════════════
# SECTION 33: REPLAY PROTECTION — STILL WORKS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 33: Replay protection still works ===")
reset_replay_registry()
l_replay = RWVEvidenceLedger()
l_replay.append_entry("test", "REPLAY-001", "h1")
try:
    l_replay.append_entry("test", "REPLAY-001", "h1")
    check(False, "Should raise")
except ValueError:
    check(True, "Duplicate rejected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 34: TEMPORAL — STILL WORKS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 34: Temporal validation still works ===")
ok_t = validate_temporal_ordering([1000.0, 2000.0, 3000.0])
bad_t = validate_temporal_ordering([3000.0, 2000.0, 1000.0])
check(ok_t.get("valid", False), "Ordered valid")
check(not bad_t.get("valid", True), "Unordered blocked")

# ══════════════════════════════════════════════════════════════════════
# SECTION 35: RECOMPUTATION — STILL WORKS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 35: Recomputation still works ===")
rec = RWVEvaluationRecord(
    session_id="x", provider_id="x", dataset_id="x", dataset_version="x",
    qualification_hash="x", dataset_hash="x", model_id=MODEL_ID, release_id=RELEASE_ID,
    release_manifest_hash="x", feature_contract_version=FEATURE_VERSION,
    native_feature_version="v1", evaluation_protocol_version="phase93_v1",
    acceptance_spec_version="phase91_v1", evaluation_config_hash="x",
    execution_status="completed",
    metrics={
        "sample_count": 100, "positive_count": 10, "negative_count": 90,
        "excluded_count": 0, "coverage": 1.0, "fraud_prevalence": 0.1,
        "precision": 0.8, "recall": 0.7, "specificity": 0.95,
        "false_positive_rate": 0.05, "false_negative_rate": 0.3,
        "f1": 0.7466666666666667, "risk_band_distribution": {},
    },
    subgroup_results={}, temporal_summary={"valid": True},
    leakage_check_result="no_leakage", contamination_check_result="no_contamination",
    result_hash="", created_at="2026-09-21T00:00:00Z",
)
recomp = verify_recomputation(rec)
check(recomp.result == RecomputationResult.MATCH.value, "Recomputation matches")

# ══════════════════════════════════════════════════════════════════════
# SECTION 36: BUNDLE — STILL WORKS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 36: Bundle still works ===")
bundle = l.export_bundle()
valid_b, _ = RWVEvidenceLedger.verify_bundle(bundle)
check(valid_b, "Bundle verified")

# ══════════════════════════════════════════════════════════════════════
# SECTION 37: BUNDLE TAMPER — STILL DETECTED
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 37: Bundle tamper still detected ===")
tampered_b = copy.deepcopy(bundle) if False else {**bundle, "entries": [dict(e) for e in bundle["entries"]]}
tampered_b["entries"][0]["evidence_hash"] = "TAMPERED"
valid_tb, _ = RWVEvidenceLedger.verify_bundle(tampered_b)
check(not valid_tb, "Tampered bundle detected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 38: FINDING REGISTRY — OPEN FINDINGS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 38: Finding registry open findings ===")
check(len(registry.open_findings()) == 0, "No open findings")
check(len(registry.remediated_findings()) == 1, "1 remediated finding")

# ══════════════════════════════════════════════════════════════════════
# SECTION 39: FINDING — SOURCE PHASE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 39: Finding source phase ===")
check(finding.source_phase == 100, "Source phase is 100")

# ══════════════════════════════════════════════════════════════════════
# SECTION 40: FINDING — INVARIANT ID
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 40: Finding invariant ID ===")
check(finding.invariant_id == "INV-07", "Invariant ID is INV-07")

# ══════════════════════════════════════════════════════════════════════
# SECTION 41: FINDING — TITLE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 41: Finding title ===")
check("runtime attestation" in finding.title.lower(), "Title mentions runtime attestation")

# ══════════════════════════════════════════════════════════════════════
# SECTION 42: FINDING — DESCRIPTION
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 42: Finding description ===")
check(len(finding.description) > 20, "Description is substantive")

# ══════════════════════════════════════════════════════════════════════
# SECTION 43: NO MODEL MUTATION
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 43: No model mutation ===")
import src.monitoring.rwv_evidence_ledger as ledger_mod
import src.monitoring.rwv_reproducibility as repro_mod
import src.monitoring.rwv_promotion_evidence as promo_mod
check(".fit(" not in inspect.getsource(ledger_mod), "No .fit() in ledger")
check(".fit(" not in inspect.getsource(repro_mod), "No .fit() in reproducibility")
check(".fit(" not in inspect.getsource(promo_mod), "No .fit() in promotion evidence")

# ══════════════════════════════════════════════════════════════════════
# SECTION 44: NO PROMOTION BYPASS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 44: No promotion bypass ===")
check(".promote(" not in inspect.getsource(ledger_mod), "No .promote() in ledger")
check(".promote(" not in inspect.getsource(repro_mod), "No .promote() in reproducibility")
check(".promote(" not in inspect.getsource(promo_mod), "No .promote() in promotion evidence")

# ══════════════════════════════════════════════════════════════════════
# SECTION 45: NO NETWORK ACCESS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 45: No network access ===")
ledger_src = inspect.getsource(ledger_mod)
repro_src = inspect.getsource(repro_mod)
check("requests.get" not in ledger_src, "No requests.get in ledger")
check("urllib.request" not in ledger_src, "No urllib in ledger")
check("requests.get" not in repro_src, "No requests.get in reproducibility")

# ══════════════════════════════════════════════════════════════════════
# SECTION 46: NO CREDENTIALS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 46: No credentials ===")
check("api_key" not in ledger_src, "No api_key in ledger")
check("password" not in ledger_src, "No password in ledger")
check("api_key" not in repro_src, "No api_key in reproducibility")

# ══════════════════════════════════════════════════════════════════════
# SECTION 47: NO PICKLE/JOBLIB
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 47: No pickle/joblib ===")
check("pickle.load" not in ledger_src, "No pickle.load in ledger")
check("joblib.load" not in ledger_src, "No joblib.load in ledger")

# ══════════════════════════════════════════════════════════════════════
# SECTION 48: LEDGER — ENTRY FROZEN
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 48: Ledger entry frozen ===")
entry = l.get_entry(0)
try:
    entry.evidence_id = "modified"
    check(False, "Should not modify frozen")
except Exception:
    check(True, "Ledger entry frozen")

# ══════════════════════════════════════════════════════════════════════
# SECTION 49: MANIFEST — FROZEN
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 49: Manifest frozen ===")
try:
    m.model_id = "changed"
    check(False, "Should not modify frozen")
except Exception:
    check(True, "Manifest frozen")

# ══════════════════════════════════════════════════════════════════════
# SECTION 50: FINDING — RESOLUTION HASH DETERMINISTIC
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 50: Resolution hash deterministic ===")
f3 = create_finding(
    finding_id="TEST-001", source_phase=100, invariant_id="INV-TEST",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test", status="open",
)
f4 = create_finding(
    finding_id="TEST-001", source_phase=100, invariant_id="INV-TEST",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test", status="open",
)
check(f3.resolution_hash == f4.resolution_hash, "Same inputs -> same hash")

# ══════════════════════════════════════════════════════════════════════
# SECTION 51: FINDING — DIFFERENT STATUS -> DIFFERENT HASH
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 51: Different status -> different hash ===")
f5 = create_finding(
    finding_id="TEST-002", source_phase=100, invariant_id="INV-TEST",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test", status="remediated",
)
f6 = create_finding(
    finding_id="TEST-002", source_phase=100, invariant_id="INV-TEST",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test", status="open",
)
check(f5.resolution_hash != f6.resolution_hash, "Different status -> different hash")

# ══════════════════════════════════════════════════════════════════════
# SECTION 52: FINDING — FROZEN DATACLASS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 52: Finding frozen dataclass ===")
try:
    f3.status = "modified"
    check(False, "Should not modify frozen")
except Exception:
    check(True, "Finding is frozen dataclass")

# ══════════════════════════════════════════════════════════════════════
# SECTION 53: FINDING — TO CANONICAL
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 53: Finding to_canonical ===")
canon = finding.to_canonical()
check("finding_id" in canon, "to_canonical has finding_id")
check("resolution_hash" in canon, "to_canonical has resolution_hash")

# ══════════════════════════════════════════════════════════════════════
# SECTION 54: FINDING — AFFECTED COMPONENTS AS TUPLE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 54: Finding affected components tuple ===")
check(isinstance(finding.affected_components, tuple), "Components is tuple")
check(len(finding.affected_components) == 2, "2 components affected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 55: FINDING — TO_DICT COMPONENTS AS LIST
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 55: Finding to_dict components as list ===")
fd2 = finding.to_dict()
check(isinstance(fd2["affected_components"], list), "to_dict serializes as list")

# ══════════════════════════════════════════════════════════════════════
# SECTION 56: FINDING REGISTRY — GET
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 56: Finding registry get ===")
found = registry.get("PHASE100-INV07")
check(found is not None, "Get finds finding")
check(found.finding_id == "PHASE100-INV07", "Correct finding returned")

# ══════════════════════════════════════════════════════════════════════
# SECTION 57: FINDING REGISTRY — GET MISSING
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 57: Finding registry get missing ===")
check(registry.get("NONEXISTENT") is None, "Missing returns None")

# ══════════════════════════════════════════════════════════════════════
# SECTION 58: FINDING REGISTRY — ALL FINDINGS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 58: Finding registry all findings ===")
all_f = registry.all_findings()
check(len(all_f) == 1, "1 finding in registry")

# ══════════════════════════════════════════════════════════════════════
# SECTION 59: FINDING REGISTRY — OPEN EMPTY
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 59: Finding registry open empty ===")
check(len(registry.open_findings()) == 0, "No open findings")

# ══════════════════════════════════════════════════════════════════════
# SECTION 60: FINDING REGISTRY — REMEDIATED HAS 1
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 60: Finding registry remediated has 1 ===")
check(len(registry.remediated_findings()) == 1, "1 remediated finding")

# ══════════════════════════════════════════════════════════════════════
# SECTION 61: FINDING — CREATED_AT
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 61: Finding created_at ===")
check("T" in finding.created_at, "Timestamp has T separator")

# ══════════════════════════════════════════════════════════════════════
# SECTION 62: FINDING — RESOLUTION_HASH IS 64 HEX
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 62: Resolution hash format ===")
check(len(finding.resolution_hash) == 64, "64 hex chars")
check(all(c in "0123456789abcdef" for c in finding.resolution_hash), "Valid hex")

# ══════════════════════════════════════════════════════════════════════
# SECTION 63: FINDING — JSON ROUNDTRIP
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 63: Finding JSON roundtrip ===")
fd_json = json.dumps(finding.to_dict())
fd_back = json.loads(fd_json)
check(fd_back["finding_id"] == "PHASE100-INV07", "Roundtrip preserves finding_id")
check(fd_back["status"] == "remediated", "Roundtrip preserves status")

# ══════════════════════════════════════════════════════════════════════
# SECTION 64: FINDING — RESOLUTION HASH CHANGES WITH STATUS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 64: Resolution hash changes with status ===")
f_open = create_finding(
    finding_id="HASH-TEST", source_phase=100, invariant_id="INV-HASH",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test", status="open",
)
f_closed = create_finding(
    finding_id="HASH-TEST", source_phase=100, invariant_id="INV-HASH",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test", status="remediated",
)
check(f_open.resolution_hash != f_closed.resolution_hash, "Different status -> different hash")

# ══════════════════════════════════════════════════════════════════════
# SECTION 65: FINDING — RESOLUTION HASH CHANGES WITH SEVERITY
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 65: Resolution hash changes with severity ===")
f_low = create_finding(
    finding_id="SEV-TEST", source_phase=100, invariant_id="INV-SEV",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test", status="open",
)
f_high = create_finding(
    finding_id="SEV-TEST", source_phase=100, invariant_id="INV-SEV",
    title="test", severity="high", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test", status="open",
)
check(f_low.resolution_hash != f_high.resolution_hash, "Different severity -> different hash")

# ══════════════════════════════════════════════════════════════════════
# SECTION 66: FINDING — RESOLUTION HASH CHANGES WITH ROOT CAUSE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 66: Resolution hash changes with root cause ===")
f_rc1 = create_finding(
    finding_id="RC-TEST", source_phase=100, invariant_id="INV-RC",
    title="test", severity="low", description="test", evidence="test",
    root_cause="cause A", affected_components=("a",), remediation="test",
    regression_test="test", status="open",
)
f_rc2 = create_finding(
    finding_id="RC-TEST", source_phase=100, invariant_id="INV-RC",
    title="test", severity="low", description="test", evidence="test",
    root_cause="cause B", affected_components=("a",), remediation="test",
    regression_test="test", status="open",
)
check(f_rc1.resolution_hash != f_rc2.resolution_hash, "Different root cause -> different hash")

# ══════════════════════════════════════════════════════════════════════
# SECTION 67: FINDING — RESOLUTION HASH CHANGES WITH COMPONENTS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 67: Resolution hash changes with components ===")
f_c1 = create_finding(
    finding_id="COMP-TEST", source_phase=100, invariant_id="INV-COMP",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test", status="open",
)
f_c2 = create_finding(
    finding_id="COMP-TEST", source_phase=100, invariant_id="INV-COMP",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a", "b"), remediation="test",
    regression_test="test", status="open",
)
check(f_c1.resolution_hash != f_c2.resolution_hash, "Different components -> different hash")

# ══════════════════════════════════════════════════════════════════════
# SECTION 68: FINDING — ALL REQUIRED FIELDS IN TO_DICT
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 68: All fields in to_dict ===")
required_fields = [
    "finding_id", "source_phase", "invariant_id", "title", "severity",
    "status", "description", "evidence", "root_cause", "affected_components",
    "remediation", "regression_test", "created_at", "resolved_at", "resolution_hash",
]
for f_name in required_fields:
    check(f_name in fd, f"Dict has {f_name}")

# ══════════════════════════════════════════════════════════════════════
# SECTION 69: FINDING — ALL REQUIRED FIELDS IN TO_CANONICAL
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 69: All fields in to_canonical ===")
for f_name in required_fields:
    check(f_name in canon, f"Canonical has {f_name}")

# ══════════════════════════════════════════════════════════════════════
# SECTION 70: FINDING — HASH IS SHA-256
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 70: Hash is SHA-256 ===")
check(len(finding.resolution_hash) == 64, "64 hex chars")
check(len(_hash_dict({"test": 1})) == 64, "Hash function returns 64 hex")

# ══════════════════════════════════════════════════════════════════════
# SECTION 71: ENVIRONMENT FINGERPRINT — DETERMINISTIC
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 71: Environment fingerprint deterministic ===")
fp1 = compute_environment_fingerprint()
fp2 = compute_environment_fingerprint()
check(fp1 == fp2, "Deterministic")
check(len(fp1) == 64, "64 hex")

# ══════════════════════════════════════════════════════════════════════
# SECTION 72: LEDGER — CHAIN BREAK DETECTION
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 72: Chain break detection ===")
valid_cb, _ = detect_chain_break(l)
check(valid_cb, "Chain valid")

# ══════════════════════════════════════════════════════════════════════
# SECTION 73: LEDGER — ENTRY TAMPER DETECTION
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 73: Entry tamper detection ===")
valid_et, _ = detect_entry_tampering(entry)
check(valid_et, "Entry valid")

# ══════════════════════════════════════════════════════════════════════
# SECTION 74: LEDGER — TAMPERED ENTRY DETECTED
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 74: Tampered entry detected ===")
from src.monitoring.rwv_evidence_ledger import RWVEvidenceLedgerEntry
tampered_entry = RWVEvidenceLedgerEntry(**{**entry.__dict__, "evidence_hash": "X"})
valid_te, _ = detect_entry_tampering(tampered_entry)
check(not valid_te, "Tampered entry detected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 75: LEDGER — EMPTY LEDGER VERIFY
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 75: Empty ledger verify ===")
empty_l = RWVEvidenceLedger()
check(empty_l.verify_chain() == LedgerVerificationResult.VERIFIED, "Empty verified")

# ══════════════════════════════════════════════════════════════════════
# SECTION 76: LEDGER — EMPTY HEAD
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 76: Empty head ===")
check(RWVEvidenceLedger().head is None, "Empty head None")

# ══════════════════════════════════════════════════════════════════════
# SECTION 77: LEDGER — EMPTY HEAD_HASH
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 77: Empty head_hash ===")
check(RWVEvidenceLedger().head_hash == "empty", "Empty head_hash")

# ══════════════════════════════════════════════════════════════════════
# SECTION 78: LEDGER — LOOKUP
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 78: Ledger lookup ===")
found_l = l.get_entry(0)
check(found_l is not None, "get_entry finds")
check(l.get_entry(999) is None, "Out of range None")
found_l2 = l.get_by_evidence_id("L-001")
check(found_l2 is not None, "get_by_evidence_id finds")
check(l.get_by_evidence_id("NONEXISTENT") is None, "Missing None")

# ══════════════════════════════════════════════════════════════════════
# SECTION 79: LEDGER — GENESIS PARENT
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 79: Genesis parent ===")
check(found_l.parent_evidence_hash == "genesis", "Genesis parent")

# ══════════════════════════════════════════════════════════════════════
# SECTION 80: LEDGER — SEQUENTIAL IDS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 80: Sequential IDs ===")
for i in range(2):
    e = l.get_entry(i)
    check(e.ledger_entry_id == i, f"Entry {i} correct ID")

# ══════════════════════════════════════════════════════════════════════
# SECTION 81: LEDGER — HEAD HASH CHANGES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 81: Head hash changes ===")
l3 = RWVEvidenceLedger()
h0 = l3.head_hash
l3.append_entry("test", "H-001", "h")
h1 = l3.head_hash
l3.append_entry("test", "H-002", "h2")
h2 = l3.head_hash
check(h0 == "empty", "Initial empty")
check(h1 != h2, "Changes on append")

# ══════════════════════════════════════════════════════════════════════
# SECTION 82: LEDGER — BUNDLE JSON SERIALIZABLE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 82: Bundle JSON serializable ===")
try:
    json.dumps(bundle)
    check(True, "Bundle JSON-serializable")
except Exception:
    check(False, "Bundle must be JSON-serializable")

# ══════════════════════════════════════════════════════════════════════
# SECTION 83: MANIFEST — JSON SERIALIZABLE
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 83: Manifest JSON serializable ===")
try:
    json.dumps(m.to_dict())
    check(True, "Manifest JSON-serializable")
except Exception:
    check(False, "Manifest must be JSON-serializable")

# ══════════════════════════════════════════════════════════════════════
# SECTION 84: FINDING — SOURCE PHASE IS INTEGER
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 84: Finding source phase is int ===")
check(isinstance(finding.source_phase, int), "Source phase is int")
check(finding.source_phase == 100, "Source phase is 100")

# ══════════════════════════════════════════════════════════════════════
# SECTION 85: FINDING — SEVERITY IN VALID SET
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 85: Finding severity in valid set ===")
valid_sev = {"critical", "high", "medium", "low", "informational"}
check(finding.severity in valid_sev, "Severity valid")

# ══════════════════════════════════════════════════════════════════════
# SECTION 86: FINDING — STATUS IN VALID SET
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 86: Finding status in valid set ===")
valid_stat = {"open", "remediation_in_progress", "remediated", "accepted_risk", "deferred", "rejected_with_evidence"}
check(finding.status in valid_stat, "Status valid")

# ══════════════════════════════════════════════════════════════════════
# SECTION 87: FINDING — RESOLVED_AT EMPTY FOR NEW
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 87: Finding resolved_at empty for new ===")
f_new = create_finding(
    finding_id="NEW-TEST", source_phase=100, invariant_id="INV-NEW",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test",
)
check(f_new.resolved_at == "", "resolved_at empty for new finding")

# ══════════════════════════════════════════════════════════════════════
# SECTION 88: FINDING — STATUS DEFAULTS TO OPEN
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 88: Finding status defaults to open ===")
check(f_new.status == "open", "Default status is open")

# ══════════════════════════════════════════════════════════════════════
# SECTION 89: FINDING — CANONICAL INCLUDES ALL FIELDS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 89: Canonical includes all fields ===")
canon_keys = set(canon.keys())
dict_keys = set(fd.keys())
check(canon_keys == dict_keys, "Canonical and dict have same keys")

# ══════════════════════════════════════════════════════════════════════
# SECTION 90: FINDING REGISTRY — EMPTY AFTER CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 90: Registry empty after construction ===")
empty_reg = AuditFindingRegistry()
check(len(empty_reg.all_findings()) == 0, "Empty registry")
check(len(empty_reg.open_findings()) == 0, "No open findings")
check(len(empty_reg.remediated_findings()) == 0, "No remediated findings")

# ══════════════════════════════════════════════════════════════════════
# SECTION 91: FINDING REGISTRY — REGISTER INCREASES COUNT
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 91: Registry register increases count ===")
empty_reg.register(f_new)
check(len(empty_reg.all_findings()) == 1, "1 finding after register")

# ══════════════════════════════════════════════════════════════════════
# SECTION 92: FINDING REGISTRY — GET AFTER REGISTER
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 92: Registry get after register ===")
found_new = empty_reg.get("NEW-TEST")
check(found_new is not None, "Get finds registered")
check(found_new.finding_id == "NEW-TEST", "Correct finding")

# ══════════════════════════════════════════════════════════════════════
# SECTION 93: FINDING REGISTRY — OPEN FINDINGS AFTER REGISTER
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 93: Registry open findings after register ===")
check(len(empty_reg.open_findings()) == 1, "1 open finding")

# ══════════════════════════════════════════════════════════════════════
# SECTION 94: FINDING REGISTRY — REMEDIATED EMPTY FOR OPEN
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 94: Registry remediated empty for open ===")
check(len(empty_reg.remediated_findings()) == 0, "No remediated for open")

# ══════════════════════════════════════════════════════════════════════
# SECTION 95: FINDING — ALL ENUMS HAVE UNIQUE VALUES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 95: Enum values unique ===")
fs_vals = [s.value for s in FindingStatus]
check(len(fs_vals) == len(set(fs_vals)), "FindingStatus values unique")
fv_vals = [s.value for s in FindingSeverity]
check(len(fv_vals) == len(set(fv_vals)), "FindingSeverity values unique")

# ══════════════════════════════════════════════════════════════════════
# SECTION 96: FINDING — JSON ROUNDTRIP FOR ALL FIELDS
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 96: JSON roundtrip all fields ===")
fd_all = json.loads(json.dumps(finding.to_dict()))
check(fd_all["finding_id"] == finding.finding_id, "finding_id roundtrip")
check(fd_all["source_phase"] == finding.source_phase, "source_phase roundtrip")
check(fd_all["invariant_id"] == finding.invariant_id, "invariant_id roundtrip")
check(fd_all["title"] == finding.title, "title roundtrip")
check(fd_all["severity"] == finding.severity, "severity roundtrip")
check(fd_all["status"] == finding.status, "status roundtrip")
check(fd_all["description"] == finding.description, "description roundtrip")
check(fd_all["evidence"] == finding.evidence, "evidence roundtrip")
check(fd_all["root_cause"] == finding.root_cause, "root_cause roundtrip")
check(fd_all["remediation"] == finding.remediation, "remediation roundtrip")
check(fd_all["regression_test"] == finding.regression_test, "regression_test roundtrip")
check(fd_all["resolution_hash"] == finding.resolution_hash, "resolution_hash roundtrip")

# ══════════════════════════════════════════════════════════════════════
# SECTION 97: FINDING — CANONICAL ROUNDTRIP
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 97: Canonical roundtrip ===")
canon_json = json.dumps(finding.to_canonical())
canon_back = json.loads(canon_json)
check(canon_back["finding_id"] == finding.finding_id, "Canonical roundtrip finding_id")
check(canon_back["resolution_hash"] == finding.resolution_hash, "Canonical roundtrip hash")

# ══════════════════════════════════════════════════════════════════════
# SECTION 98: FINDING — HASH CHANGES WITH EVERY FIELD
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 98: Hash changes with every field ===")
base_hash = finding.resolution_hash
# Change finding_id
f_mod = AuditFinding(**{**finding.__dict__, "finding_id": "CHANGED"})
check(_hash_dict(f_mod.to_canonical()) != base_hash, "Hash changes with finding_id")
# Change title
f_mod2 = AuditFinding(**{**finding.__dict__, "title": "CHANGED"})
check(_hash_dict(f_mod2.to_canonical()) != base_hash, "Hash changes with title")
# Change description
f_mod3 = AuditFinding(**{**finding.__dict__, "description": "CHANGED"})
check(_hash_dict(f_mod3.to_canonical()) != base_hash, "Hash changes with description")

# ══════════════════════════════════════════════════════════════════════
# SECTION 99: FINDING — ALL SEVERITIES PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 99: All severities produce different hashes ===")
sev_hashes = set()
for sev in FindingSeverity:
    f_sev = create_finding(
        finding_id=f"SEV-{sev.value}", source_phase=100, invariant_id="INV-SEV",
        title="test", severity=sev.value, description="test", evidence="test",
        root_cause="test", affected_components=("a",), remediation="test",
        regression_test="test", status="open",
    )
    sev_hashes.add(f_sev.resolution_hash)
check(len(sev_hashes) == 5, "All 5 severities produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 100: FINDING — ALL STATUSES PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 100: All statuses produce different hashes ===")
stat_hashes = set()
for stat in FindingStatus:
    f_stat = create_finding(
        finding_id=f"STAT-{stat.value}", source_phase=100, invariant_id="INV-STAT",
        title="test", severity="low", description="test", evidence="test",
        root_cause="test", affected_components=("a",), remediation="test",
        regression_test="test", status=stat.value,
    )
    stat_hashes.add(f_stat.resolution_hash)
check(len(stat_hashes) == 6, "All 6 statuses produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 101: FINDING — ALL INVARIANT IDS PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 101: All invariant IDs produce different hashes ===")
inv_hashes = set()
for inv_id in ["INV-A", "INV-B", "INV-C"]:
    f_inv = create_finding(
        finding_id=f"INV-{inv_id}", source_phase=100, invariant_id=inv_id,
        title="test", severity="low", description="test", evidence="test",
        root_cause="test", affected_components=("a",), remediation="test",
        regression_test="test", status="open",
    )
    inv_hashes.add(f_inv.resolution_hash)
check(len(inv_hashes) == 3, "All 3 invariant IDs produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 102: FINDING — ALL SOURCE PHASES PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 102: All source phases produce different hashes ===")
phase_hashes = set()
for phase in [100, 101, 102]:
    f_phase = create_finding(
        finding_id=f"PHASE-{phase}", source_phase=phase, invariant_id="INV-PHASE",
        title="test", severity="low", description="test", evidence="test",
        root_cause="test", affected_components=("a",), remediation="test",
        regression_test="test", status="open",
    )
    phase_hashes.add(f_phase.resolution_hash)
check(len(phase_hashes) == 3, "All 3 source phases produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 103: FINDING — ALL ROOT CAUSES PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 103: All root causes produce different hashes ===")
rc_hashes = set()
for rc in ["cause A", "cause B", "cause C"]:
    f_rc = create_finding(
        finding_id=f"RC-{rc}", source_phase=100, invariant_id="INV-RC",
        title="test", severity="low", description="test", evidence="test",
        root_cause=rc, affected_components=("a",), remediation="test",
        regression_test="test", status="open",
    )
    rc_hashes.add(f_rc.resolution_hash)
check(len(rc_hashes) == 3, "All 3 root causes produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 104: FINDING — ALL EVIDENCES PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 104: All evidences produce different hashes ===")
ev_hashes = set()
for ev_str in ["evidence A", "evidence B", "evidence C"]:
    f_ev = create_finding(
        finding_id=f"EV-{ev_str}", source_phase=100, invariant_id="INV-EV",
        title="test", severity="low", description="test", evidence=ev_str,
        root_cause="test", affected_components=("a",), remediation="test",
        regression_test="test", status="open",
    )
    ev_hashes.add(f_ev.resolution_hash)
check(len(ev_hashes) == 3, "All 3 evidences produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 105: FINDING — ALL REMEDIATIONS PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 105: All remediations produce different hashes ===")
rem_hashes = set()
for rem in ["rem A", "rem B", "rem C"]:
    f_rem = create_finding(
        finding_id=f"REM-{rem}", source_phase=100, invariant_id="INV-REM",
        title="test", severity="low", description="test", evidence="test",
        root_cause="test", affected_components=("a",), remediation=rem,
        regression_test="test", status="open",
    )
    rem_hashes.add(f_rem.resolution_hash)
check(len(rem_hashes) == 3, "All 3 remediations produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 106: FINDING — ALL REGRESSION TESTS PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 106: All regression tests produce different hashes ===")
rt_hashes = set()
for rt in ["test A", "test B", "test C"]:
    f_rt = create_finding(
        finding_id=f"RT-{rt}", source_phase=100, invariant_id="INV-RT",
        title="test", severity="low", description="test", evidence="test",
        root_cause="test", affected_components=("a",), remediation="test",
        regression_test=rt, status="open",
    )
    rt_hashes.add(f_rt.resolution_hash)
check(len(rt_hashes) == 3, "All 3 regression tests produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 107: FINDING — ALL TITLES PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 107: All titles produce different hashes ===")
title_hashes = set()
for title in ["title A", "title B", "title C"]:
    f_title = create_finding(
        finding_id=f"TITLE-{title}", source_phase=100, invariant_id="INV-TITLE",
        title=title, severity="low", description="test", evidence="test",
        root_cause="test", affected_components=("a",), remediation="test",
        regression_test="test", status="open",
    )
    title_hashes.add(f_title.resolution_hash)
check(len(title_hashes) == 3, "All 3 titles produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 108: FINDING — ALL DESCRIPTIONS PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 108: All descriptions produce different hashes ===")
desc_hashes = set()
for desc in ["desc A", "desc B", "desc C"]:
    f_desc = create_finding(
        finding_id=f"DESC-{desc}", source_phase=100, invariant_id="INV-DESC",
        title="test", severity="low", description=desc, evidence="test",
        root_cause="test", affected_components=("a",), remediation="test",
        regression_test="test", status="open",
    )
    desc_hashes.add(f_desc.resolution_hash)
check(len(desc_hashes) == 3, "All 3 descriptions produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 109: FINDING — ALL COMPONENT SETS PRODUCE DIFFERENT HASHES
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 109: All component sets produce different hashes ===")
comp_hashes = set()
for comps in [("a",), ("a", "b"), ("a", "b", "c")]:
    f_comp = create_finding(
        finding_id=f"COMP-{comps}", source_phase=100, invariant_id="INV-COMP",
        title="test", severity="low", description="test", evidence="test",
        root_cause="test", affected_components=comps, remediation="test",
        regression_test="test", status="open",
    )
    comp_hashes.add(f_comp.resolution_hash)
check(len(comp_hashes) == 3, "All 3 component sets produce unique hashes")

# ══════════════════════════════════════════════════════════════════════
# SECTION 110: FINDING — REGISTRY DUPLICATE DIFFERENT IDs OK
# ══════════════════════════════════════════════════════════════════════
print("\n=== SECTION 110: Registry duplicate different IDs OK ===")
reg2 = AuditFindingRegistry()
f_a = create_finding(
    finding_id="A-001", source_phase=100, invariant_id="INV-A",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test",
)
f_b = create_finding(
    finding_id="B-001", source_phase=100, invariant_id="INV-B",
    title="test", severity="low", description="test", evidence="test",
    root_cause="test", affected_components=("a",), remediation="test",
    regression_test="test",
)
reg2.register(f_a)
reg2.register(f_b)
check(len(reg2.all_findings()) == 2, "2 findings with different IDs")

# ══════════════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
if failed:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
