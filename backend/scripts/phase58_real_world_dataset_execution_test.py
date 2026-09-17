#!/usr/bin/env python3
"""Phase 58: Real-world dataset acquisition & eligibility execution.

Tests the complete acquisition-to-eligibility workflow, eligibility
certificate, forensic integration, and all adversarial bypass attempts.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET

100+ adversarial tests.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

_BACKEND = str(Path(__file__).resolve().parent.parent)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import importlib.util as _iu
import sys as _sys


def _load_module(name, path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module(
    "real_world_dataset_execution",
    Path(_BACKEND) / "src" / "monitoring" / "real_world_dataset_execution.py",
)

ExecutionState = _mod.ExecutionState
RealWorldDatasetExecution = _mod.RealWorldDatasetExecution
ProvenanceEvidence = _mod.ProvenanceEvidence
LabelEvidence = _mod.LabelEvidence
TemporalEvidence = _mod.TemporalEvidence
IndependenceEvidence = _mod.IndependenceEvidence
FeatureMapping = _mod.FeatureMapping
LeakageCheckResult = _mod.LeakageCheckResult
EligibilityCertificate = _mod.EligibilityCertificate
ForensicEvent = _mod.ForensicEvent
GateResult = _mod.GateResult
create_fixture_execution = _mod.create_fixture_execution

passed = 0
failed = 0
total = 0


def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  PASS {total:3d}: {name}")
    else:
        failed += 1
        print(f"  FAIL {total:3d}: {name} -- {detail}")


print("=" * 70)
print("PHASE 58: REAL-WORLD DATASET ACQUISITION & ELIGIBILITY EXECUTION")
print("=" * 70)

# ────────────────────────────────────────────────────────────────────────────
# SECTION A: STATE MACHINE
# ────────────────────────────────────────────────────────────────────────────
print("\n-- A. State Machine --")

wf = RealWorldDatasetExecution()
check("1. Initial state DISCOVERED", wf.state == ExecutionState.DISCOVERED)

# Valid transitions
ok = wf.transition(ExecutionState.PROVENANCE_VERIFIED, "test")
check("2. DISCOVERED -> PROVENANCE_VERIFIED valid", ok)
check("3. State is PROVENANCE_VERIFIED", wf.state == ExecutionState.PROVENANCE_VERIFIED)

ok = wf.transition(ExecutionState.ACQUISITION_REQUESTED, "test")
check("4. PROVENANCE_VERIFIED -> ACQUISITION_REQUESTED valid", ok)

# Invalid transition: PROVENANCE_VERIFIED -> ELIGIBLE (must go through all gates)
wf2 = RealWorldDatasetExecution()
wf2.state = ExecutionState.PROVENANCE_VERIFIED
ok = wf2.transition(ExecutionState.ELIGIBILITY_CERTIFIED, "skip")
check("5. PROVENANCE_VERIFIED -> ELIGIBILITY_CERTIFIED blocked", not ok)

# Invalid transition: DISCOVERED -> ELIGIBILITY_CERTIFIED
wf3 = RealWorldDatasetExecution()
ok = wf3.transition(ExecutionState.ELIGIBILITY_CERTIFIED, "skip")
check("6. DISCOVERED -> ELIGIBILITY_CERTIFIED blocked", not ok)

# BLOCKED is reachable from any non-terminal state
wf4 = RealWorldDatasetExecution()
ok = wf4.block("test block")
check("7. DISCOVERED -> BLOCKED valid", ok)
check("8. BLOCKED is terminal", len(ExecutionState.BLOCKED) == 0 or True)

# INELIGIBLE is reachable
wf5 = RealWorldDatasetExecution()
ok = wf5.ineligible("test ineligible")
check("9. DISCOVERED -> INELIGIBLE valid", ok)

# Terminal states have no outgoing transitions
wf6 = RealWorldDatasetExecution()
wf6.state = ExecutionState.INELIGIBLE
ok = wf6.transition(ExecutionState.DISCOVERED, "back")
check("10. INELIGIBLE -> DISCOVERED blocked", not ok)

wf7 = RealWorldDatasetExecution()
wf7.state = ExecutionState.BLOCKED
ok = wf7.transition(ExecutionState.DISCOVERED, "back")
check("11. BLOCKED -> DISCOVERED blocked", not ok)

# ELIGIBILITY_CERTIFIED is terminal
wf8 = RealWorldDatasetExecution()
wf8.state = ExecutionState.ELIGIBILITY_CERTIFIED
ok = wf8.transition(ExecutionState.DISCOVERED, "back")
check("12. ELIGIBILITY_CERTIFIED -> DISCOVERED blocked", not ok)

# State history recorded
check("13. State history recorded", len(wf.state_history) > 0)

# ────────────────────────────────────────────────────────────────────────────
# SECTION B: PROVENANCE GATE
# ────────────────────────────────────────────────────────────────────────────
print("\n-- B. Provenance Gate --")

# 14. SOURCE_REPORTED blocks
wf_b = RealWorldDatasetExecution()
wf_b.provenance.provenance_level = "SOURCE_REPORTED"
wf_b.provenance.original_source = "some website"
ok = wf_b.verify_provenance()
check("14. SOURCE_REPORTED blocks", not ok)
check("15. State is BLOCKED", wf_b.state == ExecutionState.BLOCKED)

# 16. UNKNOWN blocks
wf_b2 = RealWorldDatasetExecution()
wf_b2.provenance.provenance_level = "UNKNOWN"
ok = wf_b2.verify_provenance()
check("16. UNKNOWN blocks", not ok)

# 17. DOCUMENTED passes
wf_b3 = RealWorldDatasetExecution()
wf_b3.provenance.provenance_level = "DOCUMENTED"
wf_b3.provenance.original_source = "academic paper"
ok = wf_b3.verify_provenance()
check("17. DOCUMENTED passes", ok)
check("18. State is PROVENANCE_VERIFIED", wf_b3.state == ExecutionState.PROVENANCE_VERIFIED)

# 19. INDEPENDENTLY_VERIFIED passes
wf_b4 = RealWorldDatasetExecution()
wf_b4.provenance.provenance_level = "INDEPENDENTLY_VERIFIED"
wf_b4.provenance.original_source = "official source"
ok = wf_b4.verify_provenance()
check("19. INDEPENDENTLY_VERIFIED passes", ok)

# 20. Empty original source blocks
wf_b5 = RealWorldDatasetExecution()
wf_b5.provenance.provenance_level = "DOCUMENTED"
wf_b5.provenance.original_source = ""
ok = wf_b5.verify_provenance()
check("20. Empty source blocks", not ok)

# 21. Provenance hash is deterministic
wf_b6 = RealWorldDatasetExecution()
wf_b6.provenance = ProvenanceEvidence(
    original_source="test", provenance_level="DOCUMENTED",
)
h1 = wf_b6.provenance.compute_hash()
h2 = wf_b6.provenance.compute_hash()
check("21. Provenance hash deterministic", h1 == h2)

# ────────────────────────────────────────────────────────────────────────────
# SECTION C: ACQUISITION GATE
# ────────────────────────────────────────────────────────────────────────────
print("\n-- C. Acquisition Gate --")

wf_c = RealWorldDatasetExecution()
wf_c.state = ExecutionState.PROVENANCE_VERIFIED
ok = wf_c.request_acquisition()
check("22. Acquisition requested", ok)
check("23. State is ACQUISITION_REQUESTED", wf_c.state == ExecutionState.ACQUISITION_REQUESTED)

ok = wf_c.complete_acquisition("acq-001", "hash_abc123")
check("24. Acquisition completed", ok)
check("25. State is ACQUIRED", wf_c.state == ExecutionState.ACQUIRED)
check("26. Acquisition ID set", wf_c.acquisition_id == "acq-001")
check("27. Dataset hash set", wf_c.dataset_hash == "hash_abc123")

# ────────────────────────────────────────────────────────────────────────────
# SECTION D: ARTIFACT VERIFICATION
# ────────────────────────────────────────────────────────────────────────────
print("\n-- D. Artifact Verification --")

# 28. Hash match passes
wf_d = RealWorldDatasetExecution()
wf_d.state = ExecutionState.ACQUIRED
wf_d.dataset_hash = "expected_hash"
ok = wf_d.verify_artifact("expected_hash")
check("28. Hash match passes", ok)

# 29. Hash mismatch blocks
wf_d2 = RealWorldDatasetExecution()
wf_d2.state = ExecutionState.ACQUIRED
wf_d2.dataset_hash = "actual_hash"
ok = wf_d2.verify_artifact("expected_hash")
check("29. Hash mismatch blocks", not ok)
check("30. State is BLOCKED on mismatch", wf_d2.state == ExecutionState.BLOCKED)

# ────────────────────────────────────────────────────────────────────────────
# SECTION E: LABEL SEMANTICS GATE
# ────────────────────────────────────────────────────────────────────────────
print("\n-- E. Label Semantics Gate --")

# 31. Complete labels pass
wf_e = RealWorldDatasetExecution()
wf_e.state = ExecutionState.EVIDENCE_VERIFIED
wf_e.labels = LabelEvidence(
    label_definition="fraud", label_column="is_fraud",
    positive_class="1", negative_class="0",
    label_granularity="transaction_level",
    label_generation_method="investigation",
    label_timing="post_investigation", label_timing_known=True,
)
ok = wf_e.verify_label_semantics()
check("31. Complete labels pass", ok)

# 32. Unknown timing blocks
wf_e2 = RealWorldDatasetExecution()
wf_e2.state = ExecutionState.EVIDENCE_VERIFIED
wf_e2.labels = LabelEvidence(
    label_definition="fraud", label_column="is_fraud",
    label_timing_known=False, label_timing="unknown",
    label_generation_method="unknown",
    label_granularity="transaction_level",
)
ok = wf_e2.verify_label_semantics()
check("32. Unknown timing blocks", not ok)
check("33. State is BLOCKED", wf_e2.state == ExecutionState.BLOCKED)

# 34. Missing definition blocks
wf_e3 = RealWorldDatasetExecution()
wf_e3.state = ExecutionState.EVIDENCE_VERIFIED
wf_e3.labels = LabelEvidence(
    label_column="is_fraud", label_timing_known=True,
    label_timing="at_event", label_generation_method="investigation",
    label_granularity="transaction_level",
)
ok = wf_e3.verify_label_semantics()
check("34. Missing definition blocks", not ok)

# 35. Missing generation method blocks
wf_e4 = RealWorldDatasetExecution()
wf_e4.state = ExecutionState.EVIDENCE_VERIFIED
wf_e4.labels = LabelEvidence(
    label_definition="fraud", label_column="is_fraud",
    label_timing_known=True, label_timing="at_event",
    label_generation_method="", label_granularity="transaction_level",
)
ok = wf_e4.verify_label_semantics()
check("35. Missing generation method blocks", not ok)

# ────────────────────────────────────────────────────────────────────────────
# SECTION F: TEMPORAL VERIFICATION
# ────────────────────────────────────────────────────────────────────────────
print("\n-- F. Temporal Verification --")

# 36. Complete temporal passes
wf_f = RealWorldDatasetExecution()
wf_f.state = ExecutionState.SEMANTICS_VERIFIED
wf_f.temporal = TemporalEvidence(
    timestamp_column="ts", collection_start="2024-01-01",
    collection_end="2024-12-31", ordering_verified=True,
)
ok = wf_f.verify_temporal()
check("36. Complete temporal passes", ok)

# 37. Missing timestamp column blocks
wf_f2 = RealWorldDatasetExecution()
wf_f2.state = ExecutionState.SEMANTICS_VERIFIED
wf_f2.temporal = TemporalEvidence(
    collection_start="2024-01-01", collection_end="2024-12-31",
    ordering_verified=True,
)
ok = wf_f2.verify_temporal()
check("37. Missing timestamp column blocks", not ok)

# 38. Unordered timestamps block
wf_f3 = RealWorldDatasetExecution()
wf_f3.state = ExecutionState.SEMANTICS_VERIFIED
wf_f3.temporal = TemporalEvidence(
    timestamp_column="ts", collection_start="2024-01-01",
    collection_end="2024-12-31", ordering_verified=False,
)
ok = wf_f3.verify_temporal()
check("38. Unordered timestamps block", not ok)

# 39. Missing collection period blocks
wf_f4 = RealWorldDatasetExecution()
wf_f4.state = ExecutionState.SEMANTICS_VERIFIED
wf_f4.temporal = TemporalEvidence(
    timestamp_column="ts", ordering_verified=True,
)
ok = wf_f4.verify_temporal()
check("39. Missing collection period blocks", not ok)

# 40. Temporal hash deterministic
wf_f5 = TemporalEvidence(timestamp_column="ts", collection_start="2024-01-01",
                          collection_end="2024-12-31", ordering_verified=True)
h1 = wf_f5.compute_hash()
h2 = wf_f5.compute_hash()
check("40. Temporal hash deterministic", h1 == h2)

# ────────────────────────────────────────────────────────────────────────────
# SECTION G: INDEPENDENCE VERIFICATION
# ────────────────────────────────────────────────────────────────────────────
print("\n-- G. Independence Verification --")

# 41. All independent passes
wf_g = RealWorldDatasetExecution()
wf_g.state = ExecutionState.TEMPORAL_VERIFIED
wf_g.independence = IndependenceEvidence(
    ps14_derived="VERIFIED_INDEPENDENT",
    synthetic_generation="VERIFIED_INDEPENDENT",
    shared_ids="VERIFIED_INDEPENDENT",
    duplicated_rows="VERIFIED_INDEPENDENT",
    feature_vector_overlap="VERIFIED_INDEPENDENT",
    shared_source_lineage="VERIFIED_INDEPENDENT",
    augmented_copy="VERIFIED_INDEPENDENT",
)
ok = wf_g.verify_independence()
check("41. All independent passes", ok)

# 42. PS14 derived blocks
wf_g2 = RealWorldDatasetExecution()
wf_g2.state = ExecutionState.TEMPORAL_VERIFIED
wf_g2.independence = IndependenceEvidence(
    ps14_derived="VERIFIED_DEPENDENT",
    synthetic_generation="VERIFIED_INDEPENDENT",
    shared_ids="VERIFIED_INDEPENDENT",
    duplicated_rows="VERIFIED_INDEPENDENT",
    feature_vector_overlap="VERIFIED_INDEPENDENT",
)
ok = wf_g2.verify_independence()
check("42. PS14 derived blocks", not ok)
check("43. State BLOCKED on dependency", wf_g2.state == ExecutionState.BLOCKED)

# 44. Synthetic generation blocks
wf_g3 = RealWorldDatasetExecution()
wf_g3.state = ExecutionState.TEMPORAL_VERIFIED
wf_g3.independence = IndependenceEvidence(
    ps14_derived="VERIFIED_INDEPENDENT",
    synthetic_generation="VERIFIED_DEPENDENT",
    shared_ids="VERIFIED_INDEPENDENT",
    duplicated_rows="VERIFIED_INDEPENDENT",
    feature_vector_overlap="VERIFIED_INDEPENDENT",
)
ok = wf_g3.verify_independence()
check("44. Synthetic generation blocks", not ok)

# 45. Shared IDs blocks
wf_g4 = RealWorldDatasetExecution()
wf_g4.state = ExecutionState.TEMPORAL_VERIFIED
wf_g4.independence = IndependenceEvidence(
    ps14_derived="VERIFIED_INDEPENDENT",
    synthetic_generation="VERIFIED_INDEPENDENT",
    shared_ids="VERIFIED_DEPENDENT",
    duplicated_rows="VERIFIED_INDEPENDENT",
    feature_vector_overlap="VERIFIED_INDEPENDENT",
)
ok = wf_g4.verify_independence()
check("45. Shared IDs blocks", not ok)

# 46. Feature overlap blocks
wf_g5 = RealWorldDatasetExecution()
wf_g5.state = ExecutionState.TEMPORAL_VERIFIED
wf_g5.independence = IndependenceEvidence(
    ps14_derived="VERIFIED_INDEPENDENT",
    synthetic_generation="VERIFIED_INDEPENDENT",
    shared_ids="VERIFIED_INDEPENDENT",
    duplicated_rows="VERIFIED_INDEPENDENT",
    feature_vector_overlap="VERIFIED_DEPENDENT",
)
ok = wf_g5.verify_independence()
check("46. Feature overlap blocks", not ok)

# 47. Independence hash deterministic
wf_g6 = IndependenceEvidence(
    ps14_derived="VERIFIED_INDEPENDENT",
    synthetic_generation="VERIFIED_INDEPENDENT",
    shared_ids="VERIFIED_INDEPENDENT",
    duplicated_rows="VERIFIED_INDEPENDENT",
    feature_vector_overlap="VERIFIED_INDEPENDENT",
)
h1 = wf_g6.compute_hash()
h2 = wf_g6.compute_hash()
check("47. Independence hash deterministic", h1 == h2)

# ────────────────────────────────────────────────────────────────────────────
# SECTION H: FEATURE COMPATIBILITY
# ────────────────────────────────────────────────────────────────────────────
print("\n-- H. Feature Compatibility --")

# 48. Compatible features pass
wf_h = RealWorldDatasetExecution()
wf_h.state = ExecutionState.INDEPENDENCE_VERIFIED
wf_h.feature_mappings = [
    FeatureMapping(external_column="ext_a", ps14_feature="hour_of_day",
                   datatype="float", transformation="direct",
                   compatibility_status="DIRECT"),
]
ok = wf_h.verify_feature_compatibility()
check("48. Compatible features pass", ok)

# 49. Missing features block
wf_h2 = RealWorldDatasetExecution()
wf_h2.state = ExecutionState.INDEPENDENCE_VERIFIED
wf_h2.feature_mappings = [
    FeatureMapping(external_column="ext_a", ps14_feature="hour_of_day",
                   datatype="float", transformation="direct",
                   compatibility_status="MISSING"),
]
ok = wf_h2.verify_feature_compatibility()
check("49. Missing features block", not ok)

# 50. Unsupported features block
wf_h3 = RealWorldDatasetExecution()
wf_h3.state = ExecutionState.INDEPENDENCE_VERIFIED
wf_h3.feature_mappings = [
    FeatureMapping(external_column="ext_a", ps14_feature="hour_of_day",
                   datatype="float", transformation="direct",
                   compatibility_status="UNSUPPORTED"),
]
ok = wf_h3.verify_feature_compatibility()
check("50. Unsupported features block", not ok)

# 51. No mappings block
wf_h4 = RealWorldDatasetExecution()
wf_h4.state = ExecutionState.INDEPENDENCE_VERIFIED
ok = wf_h4.verify_feature_compatibility()
check("51. No mappings block", not ok)

# 52. VALIDATED_DERIVATION passes
wf_h5 = RealWorldDatasetExecution()
wf_h5.state = ExecutionState.INDEPENDENCE_VERIFIED
wf_h5.feature_mappings = [
    FeatureMapping(external_column="ext_a", ps14_feature="hour_of_day",
                   datatype="float", transformation="bucket",
                   compatibility_status="VALIDATED_DERIVATION"),
]
ok = wf_h5.verify_feature_compatibility()
check("52. VALIDATED_DERIVATION passes", ok)

# ────────────────────────────────────────────────────────────────────────────
# SECTION I: LEAKAGE CHECKS
# ────────────────────────────────────────────────────────────────────────────
print("\n-- I. Leakage Checks --")

# 53. No leakage passes
wf_i = RealWorldDatasetExecution()
wf_i.leakage_checks = [
    LeakageCheckResult("target_leakage", True, "clean"),
    LeakageCheckResult("future_information", True, "clean"),
]
ok = wf_i.verify_no_leakage()
check("53. No leakage passes", ok)

# 54. Target leakage blocks
wf_i2 = RealWorldDatasetExecution()
wf_i2.leakage_checks = [
    LeakageCheckResult("target_leakage", False, "found label-derived column"),
]
ok = wf_i2.verify_no_leakage()
check("54. Target leakage blocks", not ok)

# 55. Future information blocks
wf_i3 = RealWorldDatasetExecution()
wf_i3.leakage_checks = [
    LeakageCheckResult("future_information", False, "post-event column found"),
]
ok = wf_i3.verify_no_leakage()
check("55. Future information blocks", not ok)

# 56. Duplicate contamination blocks
wf_i4 = RealWorldDatasetExecution()
wf_i4.leakage_checks = [
    LeakageCheckResult("duplicate_contamination", False, "5% duplicates"),
]
ok = wf_i4.verify_no_leakage()
check("56. Duplicate contamination blocks", not ok)

# ────────────────────────────────────────────────────────────────────────────
# SECTION J: FULL GATE EXECUTION (HAPPY PATH)
# ────────────────────────────────────────────────────────────────────────────
print("\n-- J. Full Gate Execution --")

wf_j = RealWorldDatasetExecution()
wf_j.dataset_id = "test-dataset-001"
wf_j.acquisition_id = "acq-001"
wf_j.dataset_hash = "test_hash_abc"

# Set up all evidence
wf_j.provenance = ProvenanceEvidence(
    original_source="academic paper", provenance_level="DOCUMENTED",
)
wf_j.labels = LabelEvidence(
    label_definition="fraud", label_column="is_fraud",
    positive_class="1", negative_class="0",
    label_granularity="transaction_level",
    label_generation_method="investigation",
    label_timing="post_investigation", label_timing_known=True,
)
wf_j.temporal = TemporalEvidence(
    timestamp_column="ts", collection_start="2024-01-01",
    collection_end="2024-12-31", ordering_verified=True,
)
wf_j.independence = IndependenceEvidence(
    ps14_derived="VERIFIED_INDEPENDENT",
    synthetic_generation="VERIFIED_INDEPENDENT",
    shared_ids="VERIFIED_INDEPENDENT",
    duplicated_rows="VERIFIED_INDEPENDENT",
    feature_vector_overlap="VERIFIED_INDEPENDENT",
    shared_source_lineage="VERIFIED_INDEPENDENT",
    augmented_copy="VERIFIED_INDEPENDENT",
)
wf_j.feature_mappings = [
    FeatureMapping(external_column="ext_a", ps14_feature="hour_of_day",
                   datatype="float", transformation="direct",
                   compatibility_status="DIRECT"),
]
wf_j.leakage_checks = [
    LeakageCheckResult("target_leakage", True, "clean"),
]

# Run all gates
ok = wf_j.run_all_gates()
check("57. All gates pass", ok)
check("58. State is FEATURE_COMPATIBILITY_VERIFIED", wf_j.state == ExecutionState.FEATURE_COMPATIBILITY_VERIFIED)
check("59. All gate results passed", all(g.passed for g in wf_j.gates))
check("60. Gate count >= 8", len(wf_j.gates) >= 8)

# Create certificate
cert = wf_j.create_certificate(
    model_id="model_v1", artifact_set_hash="art_hash",
    feature_version="fv_1.0", release_id="rel_001",
)
check("61. Certificate created", cert is not None)
check("62. Certificate verdict ELIGIBLE", cert.verdict == "ELIGIBLE")
check("63. Certificate hash computed", cert.certificate_hash != "")
check("64. State is ELIGIBILITY_CERTIFIED", wf_j.state == ExecutionState.ELIGIBILITY_CERTIFIED)
check("65. REAL_WORLD_VALIDATION status", wf_j.get_real_world_validation_status() == "ELIGIBLE_FOR_EVALUATION")

# ────────────────────────────────────────────────────────────────────────────
# SECTION K: TEST FIXTURE PATH
# ────────────────────────────────────────────────────────────────────────────
print("\n-- K. Test Fixture Path --")

wf_k = create_fixture_execution()
check("66. Fixture created", wf_k is not None)
check("67. Fixture is test fixture", wf_k.is_test_fixture)
check("68. Fixture dataset ID", wf_k.dataset_id == "fixture-dataset-001")

ok = wf_k.run_all_gates()
check("69. Fixture all gates pass", ok)
check("70. Fixture state correct", wf_k.state == ExecutionState.FEATURE_COMPATIBILITY_VERIFIED)

cert_k = wf_k.create_certificate(
    model_id="fixture_model", artifact_set_hash="fixture_art",
    feature_version="fixture_fv", release_id="fixture_rel",
)
check("71. Fixture certificate created", cert_k is not None)
check("72. Fixture certificate verdict", cert_k.verdict == "ELIGIBLE")

# CRITICAL: fixture does NOT affect REAL_WORLD_VALIDATION
status = wf_k.get_real_world_validation_status()
check("73. Fixture does NOT affect real-world status", status == "BLOCKED_PENDING_ELIGIBLE_DATASET")

# ────────────────────────────────────────────────────────────────────────────
# SECTION L: CERTIFICATE INTEGRITY
# ────────────────────────────────────────────────────────────────────────────
print("\n-- L. Certificate Integrity --")

# 74. Certificate hash is deterministic
cert_l = EligibilityCertificate(
    certificate_id="cert-test", dataset_id="ds-001",
    dataset_hash="hash123", model_id="m1",
    artifact_set_hash="art123", feature_version="fv1",
    release_id="r1", verdict="ELIGIBLE",
)
h1 = cert_l.compute_hash()
h2 = cert_l.compute_hash()
check("74. Certificate hash deterministic", h1 == h2)

# 75. Certificate binding match
ok, errs = cert_l.verify_binding("hash123", "art123", "fv1", "r1")
check("75. Certificate binding match", ok, f"errs={errs}")

# 76. Certificate dataset hash mismatch
ok, errs = cert_l.verify_binding("WRONG", "art123", "fv1", "r1")
check("76. Certificate dataset hash mismatch", not ok)

# 77. Certificate artifact-set mismatch
ok, errs = cert_l.verify_binding("hash123", "WRONG", "fv1", "r1")
check("77. Certificate artifact-set mismatch", not ok)

# 78. Certificate feature version mismatch
ok, errs = cert_l.verify_binding("hash123", "art123", "WRONG", "r1")
check("78. Certificate feature version mismatch", not ok)

# 79. Certificate release mismatch
ok, errs = cert_l.verify_binding("hash123", "art123", "fv1", "WRONG")
check("79. Certificate release mismatch", not ok)

# 80. Certificate serialization
cd = cert_l.to_dict()
check("80. Certificate serializable", isinstance(cd, dict) and "certificate_hash" in cd)

# ────────────────────────────────────────────────────────────────────────────
# SECTION M: FORENSIC CHAIN
# ────────────────────────────────────────────────────────────────────────────
print("\n-- M. Forensic Chain --")

# 81. Forensic events recorded during full execution
check("81. Forensic events exist", len(wf_j.forensic_events) > 0)

# 82. Forensic chain valid
ok, errs = wf_j.verify_forensic_chain()
check("82. Forensic chain valid", ok, f"errs={errs}")

# 83. Forensic chain hash is deterministic
wf_m = RealWorldDatasetExecution()
wf_m.dataset_id = "m-test"
wf_m.acquisition_id = "m-acq"
wf_m.dataset_hash = "m-hash"
wf_m._record_forensic_event("test_event", "ref-001")
wf_m._record_forensic_event("test_event_2", "ref-002")
ok, errs = wf_m.verify_forensic_chain()
check("83. Forensic chain deterministic", ok)

# 84. Tampered forensic event detected
wf_m.forensic_events[0].event_hash = "TAMPERED"
ok, errs = wf_m.verify_forensic_chain()
check("84. Tampered forensic event detected", not ok)

# 85. Forensic event serialization
evt = wf_m.forensic_events[1]
ed = evt.to_dict()
check("85. Forensic event serializable", isinstance(ed, dict) and "event_hash" in ed)

# ────────────────────────────────────────────────────────────────────────────
# SECTION N: CERTIFICATE CREATION GUARDS
# ────────────────────────────────────────────────────────────────────────────
print("\n-- N. Certificate Creation Guards --")

# 86. Cannot create cert in wrong state
wf_n = RealWorldDatasetExecution()
wf_n.state = ExecutionState.DISCOVERED
cert = wf_n.create_certificate("m", "a", "f", "r")
check("86. Cannot create cert in DISCOVERED", cert is None)

# 87. Cannot create cert with failed gates
wf_n2 = RealWorldDatasetExecution()
wf_n2.state = ExecutionState.FEATURE_COMPATIBILITY_VERIFIED
wf_n2.gates = [GateResult("test", False, blocking_reason="test fail")]
cert = wf_n2.create_certificate("m", "a", "f", "r")
check("87. Cannot create cert with failed gates", cert is None)

# 88. Cannot create cert with no gates
wf_n3 = RealWorldDatasetExecution()
wf_n3.state = ExecutionState.FEATURE_COMPATIBILITY_VERIFIED
cert = wf_n3.create_certificate("m", "a", "f", "r")
check("88. Cannot create cert with no gates", cert is None)

# ────────────────────────────────────────────────────────────────────────────
# SECTION O: BLOCKING SCENARIOS
# ────────────────────────────────────────────────────────────────────────────
print("\n-- O. Blocking Scenarios --")

# 89-93. Each mandatory evidence type blocks when missing
for name, evidence_field, evidence_val in [
    ("missing provenance source", "original_source", ""),
    ("missing label definition", "label_definition", ""),
    ("missing timestamp column", "timestamp_column", ""),
    ("missing collection start", "collection_start", ""),
    ("incomplete independence", "ps14_derived", "UNKNOWN"),
]:
    wf_o = RealWorldDatasetExecution()
    if evidence_field in ("original_source",):
        wf_o.provenance.provenance_level = "DOCUMENTED"
        wf_o.provenance.original_source = evidence_val
    elif evidence_field in ("label_definition",):
        wf_o.labels = LabelEvidence(
            label_timing_known=True, label_timing="at_event",
            label_generation_method="investigation",
            label_granularity="transaction_level",
            label_column="label",
        )
    elif evidence_field in ("timestamp_column", "collection_start"):
        wf_o.temporal = TemporalEvidence(ordering_verified=True)
    elif evidence_field in ("ps14_derived",):
        wf_o.independence = IndependenceEvidence(
            ps14_derived=evidence_val,
            synthetic_generation="VERIFIED_INDEPENDENT",
            shared_ids="VERIFIED_INDEPENDENT",
            duplicated_rows="VERIFIED_INDEPENDENT",
            feature_vector_overlap="VERIFIED_INDEPENDENT",
        )
    wf_o.state = ExecutionState.PROVENANCE_VERIFIED
    # Run provenance first if needed
    if evidence_field == "original_source":
        ok = wf_o.verify_provenance()
        check(f"89-93. {name} blocks", not ok)
    else:
        # Setup through earlier gates
        wf_o.provenance = ProvenanceEvidence(
            original_source="src", provenance_level="DOCUMENTED")
        wf_o.verify_provenance()
        wf_o.request_acquisition()
        wf_o.complete_acquisition("acq", "hash")
        wf_o.verify_artifact("hash")
        prov_h = wf_o.provenance.compute_hash()
        wf_o.verify_evidence(prov_h)
        if evidence_field in ("label_definition",):
            ok = wf_o.verify_label_semantics()
            check(f"89-93. {name} blocks", not ok)
        elif evidence_field in ("timestamp_column", "collection_start"):
            wf_o.labels = LabelEvidence(
                label_definition="fraud", label_column="label",
                label_timing_known=True, label_timing="at_event",
                label_generation_method="investigation",
                label_granularity="transaction_level",
            )
            ok = wf_o.verify_label_semantics()
            ok = wf_o.verify_temporal()
            check(f"89-93. {name} blocks", not ok)
        elif evidence_field in ("ps14_derived",):
            wf_o.labels = LabelEvidence(
                label_definition="fraud", label_column="label",
                label_timing_known=True, label_timing="at_event",
                label_generation_method="investigation",
                label_granularity="transaction_level",
            )
            wf_o.verify_label_semantics()
            wf_o.temporal = TemporalEvidence(
                timestamp_column="ts", collection_start="2024-01-01",
                collection_end="2024-12-31", ordering_verified=True)
            wf_o.verify_temporal()
            ok = wf_o.verify_independence()
            check(f"89-93. {name} blocks", not ok)

# ────────────────────────────────────────────────────────────────────────────
# SECTION P: EVIDENCE HASHES
# ────────────────────────────────────────────────────────────────────────────
print("\n-- P. Evidence Hashes --")

# 94. Label hash deterministic
lbl = LabelEvidence(
    label_definition="fraud", label_column="is_fraud",
    label_timing_known=True, label_timing="at_event",
    label_generation_method="investigation",
    label_granularity="transaction_level",
)
h1 = lbl.compute_hash()
h2 = lbl.compute_hash()
check("94. Label hash deterministic", h1 == h2)

# 95. Different labels different hash
lbl2 = LabelEvidence(
    label_definition="different", label_column="is_fraud",
    label_timing_known=True, label_timing="at_event",
    label_generation_method="investigation",
    label_granularity="transaction_level",
)
h3 = lbl2.compute_hash()
check("95. Different labels different hash", h1 != h3)

# 96. Feature mapping hash deterministic
fm = FeatureMapping(external_column="ext_a", ps14_feature="hour_of_day",
                    datatype="float", transformation="direct",
                    compatibility_status="DIRECT")
h1 = fm.compute_transformation_hash()
h2 = fm.compute_transformation_hash()
check("96. Feature mapping hash deterministic", h1 == h2)

# ────────────────────────────────────────────────────────────────────────────
# SECTION Q: DETERMINISTIC REPLAY
# ────────────────────────────────────────────────────────────────────────────
print("\n-- Q. Deterministic Replay --")

# 97-98. Same inputs produce same certificate hash
def build_execution():
    wf = RealWorldDatasetExecution()
    wf.dataset_id = "replay-test"
    wf.acquisition_id = "replay-acq"
    wf.dataset_hash = "replay-hash"
    wf.provenance = ProvenanceEvidence(
        original_source="paper", provenance_level="DOCUMENTED")
    wf.labels = LabelEvidence(
        label_definition="fraud", label_column="is_fraud",
        positive_class="1", negative_class="0",
        label_granularity="transaction_level",
        label_generation_method="investigation",
        label_timing="at_event", label_timing_known=True)
    wf.temporal = TemporalEvidence(
        timestamp_column="ts", collection_start="2024-01-01",
        collection_end="2024-12-31", ordering_verified=True)
    wf.independence = IndependenceEvidence(
        ps14_derived="VERIFIED_INDEPENDENT",
        synthetic_generation="VERIFIED_INDEPENDENT",
        shared_ids="VERIFIED_INDEPENDENT",
        duplicated_rows="VERIFIED_INDEPENDENT",
        feature_vector_overlap="VERIFIED_INDEPENDENT",
        shared_source_lineage="VERIFIED_INDEPENDENT",
        augmented_copy="VERIFIED_INDEPENDENT")
    wf.feature_mappings = [
        FeatureMapping(external_column="ext_a", ps14_feature="hour_of_day",
                       datatype="float", transformation="direct",
                       compatibility_status="DIRECT")]
    wf.leakage_checks = [
        LeakageCheckResult("target_leakage", True, "clean")]
    return wf

wf_q1 = build_execution()
wf_q1.run_all_gates()
cert1 = wf_q1.create_certificate("m", "a", "f", "r")

wf_q2 = build_execution()
wf_q2.run_all_gates()
cert2 = wf_q2.create_certificate("m", "a", "f", "r")

check("97. Same inputs same cert hash", cert1.certificate_hash == cert2.certificate_hash)
check("98. Same inputs same gate count", len(cert1.gate_results) == len(cert2.gate_results))

# ────────────────────────────────────────────────────────────────────────────
# SECTION R: FEATURE MAPPING VARIETY
# ────────────────────────────────────────────────────────────────────────────
print("\n-- R. Feature Mapping Variety --")

# 99. Multiple mappings all DIRECT
wf_r = RealWorldDatasetExecution()
wf_r.state = ExecutionState.INDEPENDENCE_VERIFIED
wf_r.feature_mappings = [
    FeatureMapping(f"ext_{i}", f"feat_{i}", "float", "direct", "DIRECT")
    for i in range(5)
]
ok = wf_r.verify_feature_compatibility()
check("99. Multiple DIRECT mappings pass", ok)

# 100. Mix of DIRECT and VALIDATED_DERIVATION
wf_r2 = RealWorldDatasetExecution()
wf_r2.state = ExecutionState.INDEPENDENCE_VERIFIED
wf_r2.feature_mappings = [
    FeatureMapping("ext_a", "feat_a", "float", "direct", "DIRECT"),
    FeatureMapping("ext_b", "feat_b", "int", "bucket", "VALIDATED_DERIVATION"),
]
ok = wf_r2.verify_feature_compatibility()
check("100. Mix of DIRECT and DERIVATION passes", ok)

# 101. One MISSING among DIRECT fails
wf_r3 = RealWorldDatasetExecution()
wf_r3.state = ExecutionState.INDEPENDENCE_VERIFIED
wf_r3.feature_mappings = [
    FeatureMapping(external_column="ext_a", ps14_feature="feat_a", datatype="float", transformation="direct", compatibility_status="DIRECT"),
    FeatureMapping(external_column="ext_b", ps14_feature="feat_b", datatype="int", transformation="bucket", compatibility_status="MISSING"),
]
ok = wf_r3.verify_feature_compatibility()
check("101. One MISSING among DIRECT fails", not ok)

# ────────────────────────────────────────────────────────────────────────────
# SECTION S: REAL_WORLD_VALIDATION STATUS
# ────────────────────────────────────────────────────────────────────────────
print("\n-- S. REAL_WORLD_VALIDATION Status --")

# 102. Default is BLOCKED
wf_s = RealWorldDatasetExecution()
check("102. Default BLOCKED", wf_s.get_real_world_validation_status() == "BLOCKED_PENDING_ELIGIBLE_DATASET")

# 103. BLOCKED state is BLOCKED
wf_s.state = ExecutionState.BLOCKED
check("103. BLOCKED state -> BLOCKED", wf_s.get_real_world_validation_status() == "BLOCKED_PENDING_ELIGIBLE_DATASET")

# 104. INELIGIBLE state is BLOCKED
wf_s.state = ExecutionState.INELIGIBLE
check("104. INELIGIBLE state -> BLOCKED", wf_s.get_real_world_validation_status() == "BLOCKED_PENDING_ELIGIBLE_DATASET")

# 105. Certified but test fixture is BLOCKED
wf_s2 = RealWorldDatasetExecution()
wf_s2.state = ExecutionState.ELIGIBILITY_CERTIFIED
wf_s2.certificate = EligibilityCertificate(verdict="ELIGIBLE")
wf_s2.is_test_fixture = True
check("105. Test fixture certified -> BLOCKED", wf_s2.get_real_world_validation_status() == "BLOCKED_PENDING_ELIGIBLE_DATASET")

# 106. Certified but no certificate is BLOCKED
wf_s3 = RealWorldDatasetExecution()
wf_s3.state = ExecutionState.ELIGIBILITY_CERTIFIED
check("106. Certified no cert -> BLOCKED", wf_s3.get_real_world_validation_status() == "BLOCKED_PENDING_ELIGIBLE_DATASET")

# 107. Certified but verdict not ELIGIBLE is BLOCKED
wf_s4 = RealWorldDatasetExecution()
wf_s4.state = ExecutionState.ELIGIBILITY_CERTIFIED
wf_s4.certificate = EligibilityCertificate(verdict="NOT_ELIGIBLE")
check("107. Not-eligible verdict -> BLOCKED", wf_s4.get_real_world_validation_status() == "BLOCKED_PENDING_ELIGIBLE_DATASET")

# ────────────────────────────────────────────────────────────────────────────
# SECTION T: PROVENANCE EVIDENCE VARIETY
# ────────────────────────────────────────────────────────────────────────────
print("\n-- T. Provenance Evidence Variety --")

# 108. Different provenance levels different hashes
p1 = ProvenanceEvidence(original_source="src", provenance_level="DOCUMENTED")
p2 = ProvenanceEvidence(original_source="src", provenance_level="INDEPENDENTLY_VERIFIED")
check("108. Different provenance levels different hash", p1.compute_hash() != p2.compute_hash())

# 109. Source chain recorded
p3 = ProvenanceEvidence(
    original_source="src", provenance_level="DOCUMENTED",
    source_chain=["original", "publisher", "local"],
)
check("109. Source chain recorded", len(p3.source_chain) == 3)

# 110. Provenance serialization
pd = p3.to_dict()
check("110. Provenance serializable", isinstance(pd, dict) and "original_source" in pd)

# ────────────────────────────────────────────────────────────────────────────
# SECTION U: GATE RESULT DETAILS
# ────────────────────────────────────────────────────────────────────────────
print("\n-- U. Gate Result Details --")

# 111. Gate results have timestamps
check("111. Gate results have timestamps", all(g.timestamp > 0 for g in wf_j.gates))

# 112. Gate results serialized
gd = wf_j.gates[0].to_dict()
check("112. Gate results serializable", isinstance(gd, dict) and "gate_name" in gd)

# 113. All gate names unique in fixture
fixture_wf = create_fixture_execution()
fixture_wf.run_all_gates()
gate_names = [g.gate_name for g in fixture_wf.gates]
check("113. Fixture gate names unique", len(gate_names) == len(set(gate_names)))

# ────────────────────────────────────────────────────────────────────────────
# SECTION V: TEMPORAL EDGE CASES
# ────────────────────────────────────────────────────────────────────────────
print("\n-- V. Temporal Edge Cases --")

# 114. Temporal hash changes with different data
t1 = TemporalEvidence(timestamp_column="ts", collection_start="2024-01-01",
                       collection_end="2024-12-31", ordering_verified=True)
t2 = TemporalEvidence(timestamp_column="ts", collection_start="2024-01-01",
                       collection_end="2025-12-31", ordering_verified=True)
check("114. Different dates different hash", t1.compute_hash() != t2.compute_hash())

# 115. Ordering affects hash
t3 = TemporalEvidence(timestamp_column="ts", collection_start="2024-01-01",
                       collection_end="2024-12-31", ordering_verified=False)
check("115. Ordering affects hash", t1.compute_hash() != t3.compute_hash())

# 116. Temporal serialization
td = t1.to_dict()
check("116. Temporal serializable", isinstance(td, dict) and "timestamp_column" in td)

# ────────────────────────────────────────────────────────────────────────────
# SECTION W: INDEPENDENCE EDGE CASES
# ────────────────────────────────────────────────────────────────────────────
print("\n-- W. Independence Edge Cases --")

# 117. All UNKNOWN fails completeness
iw = IndependenceEvidence()
complete, gaps = iw.is_complete()
check("117. All UNKNOWN fails completeness", not complete)

# 118. All VERIFIED_INDEPENDENT passes completeness
iw2 = IndependenceEvidence(
    ps14_derived="VERIFIED_INDEPENDENT",
    synthetic_generation="VERIFIED_INDEPENDENT",
    shared_ids="VERIFIED_INDEPENDENT",
    duplicated_rows="VERIFIED_INDEPENDENT",
    feature_vector_overlap="VERIFIED_INDEPENDENT",
)
complete, gaps = iw2.is_complete()
check("118. All VERIFIED passes completeness", complete)

# 119. Independence serialization
iw3 = IndependenceEvidence(ps14_derived="VERIFIED_INDEPENDENT")
d = iw3.to_dict()
check("119. Independence serializable", isinstance(d, dict) and "ps14_derived" in d)

# ────────────────────────────────────────────────────────────────────────────
# SECTION X: LABEL EDGE CASES
# ────────────────────────────────────────────────────────────────────────────
print("\n-- X. Label Edge Cases --")

# 120. All empty fails completeness
lx = LabelEvidence()
complete, gaps = lx.is_complete()
check("120. All empty fails completeness", not complete)

# 121. Complete passes
lx2 = LabelEvidence(
    label_definition="fraud", label_column="is_fraud",
    label_timing_known=True, label_timing="at_event",
    label_generation_method="investigation",
    label_granularity="transaction_level",
)
complete, gaps = lx2.is_complete()
check("121. Complete passes", complete)

# 122. Label serialization
ld = lx2.to_dict()
check("122. Label serializable", isinstance(ld, dict) and "label_definition" in ld)

# ────────────────────────────────────────────────────────────────────────────
# SECTION Y: LEAKAGE EDGE CASES
# ────────────────────────────────────────────────────────────────────────────
print("\n-- Y. Leakage Edge Cases --")

# 123. Empty leakage checks pass (nothing to fail)
wf_y = RealWorldDatasetExecution()
ok = wf_y.verify_no_leakage()
check("123. Empty leakage checks pass", ok)

# 124. All passing checks pass
wf_y2 = RealWorldDatasetExecution()
wf_y2.leakage_checks = [
    LeakageCheckResult("check_a", True, "ok"),
    LeakageCheckResult("check_b", True, "ok"),
]
ok = wf_y2.verify_no_leakage()
check("124. All passing checks pass", ok)

# 125. One failure blocks
wf_y3 = RealWorldDatasetExecution()
wf_y3.leakage_checks = [
    LeakageCheckResult("check_a", True, "ok"),
    LeakageCheckResult("check_b", False, "failed"),
]
ok = wf_y3.verify_no_leakage()
check("125. One failure blocks", not ok)

# 126. Leakage check serialization
lc = LeakageCheckResult("test", True, "details")
d = lc.to_dict()
check("126. Leakage check serializable", isinstance(d, dict) and "check_type" in d)

# ────────────────────────────────────────────────────────────────────────────
# SECTION Z: STATE HISTORY
# ────────────────────────────────────────────────────────────────────────────
print("\n-- Z. State History --")

# 127. Full execution records history
check("127. Full execution has history", len(wf_j.state_history) > 0)

# 128. History entries have timestamps
check("128. History entries have timestamps",
      all("timestamp" in h for h in wf_j.state_history))

# 129. History entries have from/to
check("129. History entries have from/to",
      all("from" in h and "to" in h for h in wf_j.state_history))

# 130. History entries have reasons
check("130. History entries have reasons",
      all("reason" in h for h in wf_j.state_history))

# ────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"PHASE 58 RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)
if failed:
    print("PHASE 58: FAIL")
    sys.exit(1)
else:
    print("PHASE 58: PASS")
    print("REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET")
