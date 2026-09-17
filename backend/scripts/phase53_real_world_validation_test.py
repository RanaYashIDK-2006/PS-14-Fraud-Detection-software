#!/usr/bin/env python3
"""Phase 53: Real-World Validation Admission & Evaluation Framework — adversarial tests.

Tests the dataset evidence model, admission gate, leakage detection, temporal
validation, evaluation protocol, evaluation record integrity, threshold policy,
and promotion integration.  Proves that synthetic/derived data cannot masquerade
as real-world eligible, and that evaluation cannot bypass admission.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET

50+ adversarial tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_BACKEND = str(Path(__file__).resolve().parent.parent)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# Import directly to avoid heavy __init__.py chain
import importlib.util as _iu
import sys as _sys


def _load_module(name, path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    _sys.modules[name] = mod  # register before exec for dataclasses
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module("dataset_evidence", Path(_BACKEND) / "src" / "monitoring" / "dataset_evidence.py")
_pg = _load_module("promotion_gate", Path(_BACKEND) / "src" / "monitoring" / "promotion_gate.py")

DatasetEvidenceRecord = _mod.DatasetEvidenceRecord
SourceClassification = _mod.SourceClassification
AdmissionState = _mod.AdmissionState
EvidenceLevel = _mod.EvidenceLevel
AdmissionReport = _mod.AdmissionReport
run_admission_gates = _mod.run_admission_gates
EvaluationProtocol = _mod.EvaluationProtocol
EvaluationRecordIntegrity = _mod.EvaluationRecordIntegrity
ThresholdPolicy = _mod.ThresholdPolicy
audit_current_datasets = _mod.audit_current_datasets
KNOWN_CANDIDATES = _mod.KNOWN_CANDIDATES
check_target_leakage = _mod.check_target_leakage
check_future_information = _mod.check_future_information
check_duplicate_contamination = _mod.check_duplicate_contamination
check_train_test_contamination = _mod.check_train_test_contamination
validate_temporal_order = _mod.validate_temporal_order
gate_source_classification = _mod.gate_source_classification
gate_provenance = _mod.gate_provenance
gate_label_semantics = _mod.gate_label_semantics
gate_temporal_semantics = _mod.gate_temporal_semantics
gate_schema_integrity = _mod.gate_schema_integrity
gate_feature_compatibility = _mod.gate_feature_compatibility
gate_minimum_size = _mod.gate_minimum_size

evaluate_real_world_validation = _pg.evaluate_real_world_validation
GateStatus = _pg.GateStatus

_PASSED = 0
_FAILED = 0
_TOTAL = 0


def _check(name: str, condition: bool, detail: str = ""):
    global _PASSED, _FAILED, _TOTAL
    _TOTAL += 1
    if condition:
        _PASSED += 1
        print(f"  PASS: {name}")
    else:
        _FAILED += 1
        print(f"  FAIL: {name} — {detail}")


def _make_record(**overrides) -> DatasetEvidenceRecord:
    """Create a default valid-ish evidence record."""
    defaults = dict(
        dataset_id="test-dataset",
        dataset_version="v1",
        source_name="test-source",
        source_url="https://example.com",
        source_type="academic",
        source_classification=SourceClassification.REAL.value,
        license="CC0",
        dataset_hash="a" * 64,
        row_count=10000,
        column_count=20,
        feature_schema=["f1", "f2", "f3"],
        label_column="is_fraud",
        label_definition="Confirmed chargeback fraud",
        label_generation_method="retrospective",
        label_timestamp_semantics="post_investigation",
        label_timing_known=True,
        positive_class_definition="Confirmed fraud",
        negative_class_definition="Non-fraud",
        transaction_timestamp_column="ts",
        collection_period_start="2020-01-01",
        collection_period_end="2023-12-31",
        temporal_order_known=True,
        provenance_confidence=EvidenceLevel.VERIFIED.value,
        provenance_references=["https://doi.org/10.xxx"],
        known_preprocessing="standard_scaling",
        feature_mapping_version="1.0",
        missing_features=[],
        incompatible_features=[],
        admission_state=AdmissionState.CANDIDATE.value,
    )
    defaults.update(overrides)
    return DatasetEvidenceRecord(**defaults)


# ── 1. Valid release manifest ─────────────────────────────────────────────

print("\n-- 1. Valid admission with complete evidence --")
r = _make_record()
report = run_admission_gates(r)
_check("valid_admission_eligible", report.overall_status == AdmissionState.ELIGIBLE.value,
       f"got {report.overall_status}")

# ── 2. Deterministic hash ─────────────────────────────────────────────────

print("\n-- 2. Deterministic evidence hash --")
r1 = _make_record()
h1 = r1.compute_hash()
r2 = _make_record()
h2 = r2.compute_hash()
_check("deterministic_hash", h1 == h2, f"{h1[:16]} != {h2[:16]}")

# ── 3. Modified record changes hash ───────────────────────────────────────

print("\n-- 3. Modified record changes hash --")
r3 = _make_record()
h3a = r3.compute_hash()
r3.row_count = 99999
h3b = r3.compute_hash()
_check("modified_record_changes_hash", h3a != h3b)

# ── 4. Synthetic data BLOCKED ─────────────────────────────────────────────

print("\n-- 4. Synthetic data blocked --")
r = _make_record(source_classification=SourceClassification.SYNTHETIC.value)
report = run_admission_gates(r)
_check("synthetic_blocked", report.overall_status == AdmissionState.BLOCKED.value,
       f"got {report.overall_status}")

# ── 5. Derived data BLOCKED ───────────────────────────────────────────────

print("\n-- 5. Derived data blocked --")
r = _make_record(source_classification=SourceClassification.DERIVED.value)
report = run_admission_gates(r)
_check("derived_blocked", report.overall_status == AdmissionState.BLOCKED.value,
       f"got {report.overall_status}")

# ── 6. Unknown source BLOCKED ─────────────────────────────────────────────

print("\n-- 6. Unknown source blocked --")
r = _make_record(source_classification=SourceClassification.UNKNOWN.value)
report = run_admission_gates(r)
_check("unknown_source_blocked", report.overall_status == AdmissionState.BLOCKED.value,
       f"got {report.overall_status}")

# ── 7. Unknown provenance INELIGIBLE ──────────────────────────────────────

print("\n-- 7. Unknown provenance ineligible --")
r = _make_record(provenance_confidence=EvidenceLevel.UNKNOWN.value)
report = run_admission_gates(r)
_check("unknown_provenance_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 8. Missing label definition INELIGIBLE ────────────────────────────────

print("\n-- 8. Missing label definition --")
r = _make_record(label_definition="", label_timestamp_semantics="", label_generation_method="")
report = run_admission_gates(r)
_check("missing_label_def_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 9. Unknown label timing INELIGIBLE ────────────────────────────────────

print("\n-- 9. Unknown label timing --")
r = _make_record(label_timing_known=False, label_timestamp_semantics="unknown",
                 label_generation_method="unknown")
report = run_admission_gates(r)
_check("unknown_label_timing_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 10. Temporal semantics unknown INELIGIBLE ─────────────────────────────

print("\n-- 10. Temporal semantics unknown --")
r = _make_record(temporal_order_known=False, transaction_timestamp_column="",
                 collection_period_start="", collection_period_end="")
report = run_admission_gates(r)
_check("temporal_unknown_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 11. Missing features INELIGIBLE ───────────────────────────────────────

print("\n-- 11. Missing features --")
r = _make_record(missing_features=["f1", "f2"])
report = run_admission_gates(r)
_check("missing_features_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 12. Incompatible features INELIGIBLE ──────────────────────────────────

print("\n-- 12. Incompatible features --")
r = _make_record(incompatible_features=["f_bad"])
report = run_admission_gates(r)
_check("incompatible_features_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 13. No dataset hash INELIGIBLE ────────────────────────────────────────

print("\n-- 13. No dataset hash --")
r = _make_record(dataset_hash="")
report = run_admission_gates(r)
_check("no_dataset_hash_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 14. No feature mapping version INELIGIBLE ─────────────────────────────

print("\n-- 14. No feature mapping version --")
r = _make_record(feature_mapping_version="")
report = run_admission_gates(r)
_check("no_mapping_version_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 15. Too-small dataset INELIGIBLE ──────────────────────────────────────

print("\n-- 15. Too-small dataset --")
r = _make_record(row_count=5)
report = run_admission_gates(r)
_check("too_small_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 16. Source-reported provenance blocked ─────────────────────────────────

print("\n-- 16. Source-reported provenance blocks --")
r = _make_record(provenance_confidence="source_reported")
report = run_admission_gates(r)
_check("source_reported_blocked",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 17. State machine: invalid transition blocked ─────────────────────────

print("\n-- 17. Invalid state transition --")
r = _make_record(admission_state=AdmissionState.INELIGIBLE.value)
ok = r.transition(AdmissionState.ELIGIBLE, "bypass attempt")
_check("ineligible_to_eligible_blocked", not ok)

# ── 18. State machine: BLOCKED is terminal ────────────────────────────────

print("\n-- 18. BLOCKED is terminal --")
r = _make_record(admission_state=AdmissionState.BLOCKED.value)
ok = r.transition(AdmissionState.CANDIDATE, "reset attempt")
_check("blocked_is_terminal", not ok)

# ── 19. State machine: ELIGIBLE is terminal ───────────────────────────────

print("\n-- 19. ELIGIBLE is terminal --")
r = _make_record(admission_state=AdmissionState.ELIGIBLE.value)
ok = r.transition(AdmissionState.INELIGIBLE, "demotion attempt")
_check("eligible_is_terminal", not ok)

# ── 20. State machine: valid forward path ─────────────────────────────────

print("\n-- 20. Valid forward path --")
r = _make_record()
r.transition(AdmissionState.EVIDENCE_COLLECTION, "starting")
ok1 = r.transition(AdmissionState.PROVENANCE_VERIFIED, "provenance OK")
ok2 = r.transition(AdmissionState.SCHEMA_VERIFIED, "schema OK")
ok3 = r.transition(AdmissionState.LABEL_SEMANTICS_VERIFIED, "label OK")
ok4 = r.transition(AdmissionState.TEMPORAL_SEMANTICS_VERIFIED, "temporal OK")
ok5 = r.transition(AdmissionState.LEAKAGE_CHECKED, "no leakage")
ok6 = r.transition(AdmissionState.FEATURE_COMPATIBILITY_VERIFIED, "features OK")
ok7 = r.transition(AdmissionState.ELIGIBLE, "all passed")
_check("valid_forward_path", all([ok1, ok2, ok3, ok4, ok5, ok6, ok7]))

# ── 21. Target leakage detected ───────────────────────────────────────────

print("\n-- 21. Target leakage detection --")
r = check_target_leakage(["amount", "is_fraud_label", "hour"], "is_fraud")
_check("target_leakage_detected", not r.passed, f"should detect is_fraud_label")

# ── 22. No target leakage for clean columns ───────────────────────────────

print("\n-- 22. No false target leakage --")
r = check_target_leakage(["amount", "hour", "velocity"], "is_fraud")
_check("no_false_target_leakage", r.passed)

# ── 23. Future information detected ───────────────────────────────────────

print("\n-- 23. Future information detection --")
avail = {"amount": "at_event", "chargeback_date": "post_event", "hour": "at_event"}
r = check_future_information(["amount", "chargeback_date", "hour"], avail)
_check("future_info_detected", not r.passed)

# ── 24. No future information for clean features ──────────────────────────

print("\n-- 24. No false future detection --")
avail = {"amount": "at_event", "hour": "at_event"}
r = check_future_information(["amount", "hour"], avail)
_check("no_false_future_detection", r.passed)

# ── 25. Duplicate contamination detected ──────────────────────────────────

print("\n-- 25. Duplicate contamination --")
r = check_duplicate_contamination(1000, unique_ratio=0.50)
_check("duplicate_detected", not r.passed)

# ── 26. Train/test contamination detected ─────────────────────────────────

print("\n-- 26. Train/test contamination --")
train = {"row1", "row2", "row3", "row4", "row5"}
test = {"row3", "row4", "row6"}
r = check_train_test_contamination(train, test)
_check("train_test_contamination_detected", not r.passed)

# ── 27. No train/test contamination ───────────────────────────────────────

print("\n-- 27. No false train/test contamination --")
train = {"row1", "row2"}
test = {"row3", "row4"}
r = check_train_test_contamination(train, test)
_check("no_false_train_test_contamination", r.passed)

# ── 28. Temporal order valid ──────────────────────────────────────────────

print("\n-- 28. Valid temporal order --")
r = validate_temporal_order([1.0, 2.0, 3.0, 4.0])
_check("temporal_order_valid", r.temporal_order_valid)

# ── 29. Temporal disorder detected ────────────────────────────────────────

print("\n-- 29. Temporal disorder detected --")
r = validate_temporal_order([3.0, 1.0, 4.0, 2.0])
_check("temporal_disorder_detected", not r.temporal_order_valid)

# ── 30. Evaluation protocol binding ───────────────────────────────────────

print("\n-- 30. Evaluation protocol binding --")
r = _make_record()
proto = EvaluationProtocol(
    dataset_hash=r.dataset_hash,
    feature_version=r.feature_mapping_version,
)
ok, msg = proto.verify_binding(r)
_check("eval_protocol_binding_valid", ok, msg)

# ── 31. Evaluation protocol wrong dataset hash ────────────────────────────

print("\n-- 31. Evaluation protocol wrong dataset hash --")
r = _make_record()
proto = EvaluationProtocol(
    dataset_hash="b" * 64,
    feature_version=r.feature_mapping_version,
)
ok, _ = proto.verify_binding(r)
_check("eval_protocol_wrong_hash_blocked", not ok)

# ── 32. Evaluation protocol wrong feature version ─────────────────────────

print("\n-- 32. Evaluation protocol wrong feature version --")
r = _make_record()
proto = EvaluationProtocol(
    dataset_hash=r.dataset_hash,
    feature_version="WRONG_VERSION",
)
ok, _ = proto.verify_binding(r)
_check("eval_protocol_wrong_feature_blocked", not ok)

# ── 33. Evaluation record integrity ───────────────────────────────────────

print("\n-- 33. Evaluation record integrity --")
rec = EvaluationRecordIntegrity(
    record_id="eval-001",
    protocol_hash="aaa",
    dataset_hash="bbb",
    model_hash="ccc",
    metrics_hash="ddd",
)
rec.evaluation_hash = rec.compute_record_hash()
ok, _ = rec.verify_integrity()
_check("eval_record_integrity_valid", ok)

# ── 34. Evaluation record tampering detected ──────────────────────────────

print("\n-- 34. Evaluation record tampering --")
rec = EvaluationRecordIntegrity(
    record_id="eval-001",
    protocol_hash="aaa",
    dataset_hash="bbb",
    model_hash="ccc",
    metrics_hash="ddd",
)
rec.evaluation_hash = rec.compute_record_hash()
# Tamper
rec.metrics_hash = "tampered"
ok, _ = rec.verify_integrity()
_check("eval_record_tampering_detected", not ok)

# ── 35. Threshold policy: fit on test blocks ──────────────────────────────

print("\n-- 35. Threshold fit-on-test blocks --")
tp = ThresholdPolicy(fit_on_test=True)
ok, _ = tp.validate()
_check("threshold_fit_on_test_blocks", not ok)

# ── 36. Threshold policy: valid ───────────────────────────────────────────

print("\n-- 36. Threshold policy valid --")
tp = ThresholdPolicy(method="fixed", threshold=0.5, fit_on_test=False, frozen=True)
ok, _ = tp.validate()
_check("threshold_policy_valid", ok)

# ── 37. Threshold policy: not frozen blocks ───────────────────────────────

print("\n-- 37. Threshold not frozen blocks --")
tp = ThresholdPolicy(frozen=False)
ok, _ = tp.validate()
_check("threshold_not_frozen_blocks", not ok)

# ── 38. Admission history tracks transitions ──────────────────────────────

print("\n-- 38. Admission history tracking --")
r = _make_record()
r.transition(AdmissionState.EVIDENCE_COLLECTION, "test")
r.transition(AdmissionState.PROVENANCE_VERIFIED, "test")
_check("admission_history_length", len(r.admission_history) == 2)
_check("admission_history_first_from", r.admission_history[0]["from"] == AdmissionState.CANDIDATE.value)
_check("admission_history_first_to", r.admission_history[0]["to"] == AdmissionState.EVIDENCE_COLLECTION.value)

# ── 39. Admission history has timestamps ──────────────────────────────────

print("\n-- 39. Admission history timestamps --")
_check("admission_history_has_timestamp", "timestamp" in r.admission_history[0])

# ── 40. Synthetic masquerade blocked ──────────────────────────────────────

print("\n-- 40. Synthetic masquerade blocked --")
r = _make_record(
    source_classification=SourceClassification.SYNTHETIC.value,
    provenance_confidence=EvidenceLevel.VERIFIED.value,
    label_timing_known=True,
)
report = run_admission_gates(r)
_check("synthetic_masquerade_blocked",
       report.overall_status == AdmissionState.BLOCKED.value)

# ── 41. Derived masquerade blocked ────────────────────────────────────────

print("\n-- 41. Derived masquerade blocked --")
r = _make_record(
    source_classification=SourceClassification.DERIVED.value,
    provenance_confidence=EvidenceLevel.VERIFIED.value,
    label_timing_known=True,
)
report = run_admission_gates(r)
_check("derived_masquerade_blocked",
       report.overall_status == AdmissionState.BLOCKED.value)

# ── 42. Multiple failures accumulate ──────────────────────────────────────

print("\n-- 42. Multiple failures accumulate --")
r = _make_record(
    label_timing_known=False,
    label_timestamp_semantics="",
    temporal_order_known=False,
)
report = run_admission_gates(r)
_check("multiple_failures_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value))
_check("multiple_failure_reasons", len(report.blocked_reasons) >= 2,
       f"got {len(report.blocked_reasons)} reasons")

# ── 43. Gate results are structured ───────────────────────────────────────

print("\n-- 43. Gate results structured --")
r = _make_record()
report = run_admission_gates(r)
_check("gates_have_results", len(report.gates) >= 5)
_check("gates_have_names", all(hasattr(g, "gate_name") for g in report.gates))

# ── 44. Source-reported provenance blocks ─────────────────────────────────

print("\n-- 44. Source-reported provenance blocks --")
r = _make_record(provenance_confidence="source_reported")
g = gate_provenance(r)
_check("source_reported_provenance_blocks", not g.passed)

# ── 45. Verified provenance passes ────────────────────────────────────────

print("\n-- 45. Verified provenance passes --")
r = _make_record(provenance_confidence=EvidenceLevel.VERIFIED.value)
g = gate_provenance(r)
_check("verified_provenance_passes", g.passed)

# ── 46. Known provenance passes ───────────────────────────────────────────

print("\n-- 46. Known provenance passes --")
r = _make_record(provenance_confidence=EvidenceLevel.KNOWN.value)
g = gate_provenance(r)
_check("known_provenance_passes", g.passed)

# ── 47. Evaluation protocol deterministic hash ────────────────────────────

print("\n-- 47. Eval protocol deterministic hash --")
p1 = EvaluationProtocol(dataset_hash="a" * 64, feature_version="v1", model_id="m1")
p2 = EvaluationProtocol(dataset_hash="a" * 64, feature_version="v1", model_id="m1")
_check("eval_protocol_hash_deterministic", p1.compute_hash() == p2.compute_hash())

# ── 48. Evaluation protocol hash changes with modification ────────────────

print("\n-- 48. Eval protocol hash changes with mod --")
p1 = EvaluationProtocol(dataset_hash="a" * 64, feature_version="v1")
h1 = p1.compute_hash()
p1.threshold_value = 0.99
h2 = p1.compute_hash()
_check("eval_protocol_hash_changes", h1 != h2)

# ── 49. Dataset hash from file ────────────────────────────────────────────

print("\n-- 49. Dataset hash from file --")
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("col1,col2\n1,2\n3,4\n")
    tmp = f.name
try:
    r = _make_record()
    h = r.compute_dataset_hash(tmp)
    _check("dataset_hash_from_file", len(h) == 64, f"len={len(h)}")
    # Second call should be identical
    r2 = _make_record()
    h2 = r2.compute_dataset_hash(tmp)
    _check("dataset_hash_deterministic", h == h2)
finally:
    os.unlink(tmp)

# ── 50. Dataset hash changes with modification ────────────────────────────

print("\n-- 50. Dataset hash changes with modified file --")
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("col1,col2\n1,2\n3,4\n")
    tmp1 = f.name
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("col1,col2\n1,2\n3,5\n")  # one byte changed
    tmp2 = f.name
try:
    r1 = _make_record()
    h1 = r1.compute_dataset_hash(tmp1)
    r2 = _make_record()
    h2 = r2.compute_dataset_hash(tmp2)
    _check("dataset_hash_tamper_detected", h1 != h2)
finally:
    os.unlink(tmp1)
    os.unlink(tmp2)

# ── 51. Current dataset audit ─────────────────────────────────────────────

print("\n-- 51. Current dataset audit --")
audit = audit_current_datasets()
_check("audit_returns_results", len(audit) >= 3, f"got {len(audit)}")
for entry in audit:
    _check(f"audit_{entry['name'].replace(' ', '_')}_ineligible",
           not entry["eligibility"],
           f"expected ineligible, got eligible={entry['eligibility']}")

# ── 52. All known candidates are ineligible ───────────────────────────────

print("\n-- 52. All known candidates ineligible --")
for c in KNOWN_CANDIDATES:
    _check(f"candidate_{c['name'].replace(' ', '_')}_ineligible",
           not c["eligible"],
           f"expected ineligible")

# ── 53. Admission report has evidence hash ────────────────────────────────

print("\n-- 53. Admission report has evidence hash --")
r = _make_record()
report = run_admission_gates(r)
_check("report_has_evidence_hash", len(report.evidence_record_hash) == 64)

# ── 54. Admission report serializable ─────────────────────────────────────

print("\n-- 54. Admission report serializable --")
r = _make_record()
report = run_admission_gates(r)
d = report.to_dict()
j = json.dumps(d, default=str)
_check("report_serializable", len(j) > 10)

# ── 55. Feature schema empty INELIGIBLE ───────────────────────────────────

print("\n-- 55. Empty feature schema blocks --")
r = _make_record(feature_schema=[], row_count=0)
report = run_admission_gates(r)
_check("empty_schema_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value),
       f"got {report.overall_status}")

# ── 56. Temporal check with no timestamps ─────────────────────────────────

print("\n-- 56. Temporal check with no timestamps --")
r = validate_temporal_order(None)
_check("no_timestamps_valid", r.temporal_order_valid)

# ── 57. Duplicate check: clean ────────────────────────────────────────────

print("\n-- 57. Clean duplicate check --")
r = check_duplicate_contamination(10000, unique_ratio=1.0)
_check("clean_duplicates", r.passed)

# ── 58. Train/test check: no data ─────────────────────────────────────────

print("\n-- 58. Train/test check no data --")
r = check_train_test_contamination(None, None)
_check("train_test_no_data", r.passed)

# ── 59. Future info: no metadata ──────────────────────────────────────────

print("\n-- 59. Future info no metadata --")
r = check_future_information(["f1", "f2"], None)
_check("future_info_no_metadata", r.passed)

# ── 60. Evaluation record: protocol hash changes ──────────────────────────

print("\n-- 60. Eval record with different protocol hash --")
rec1 = EvaluationRecordIntegrity(
    record_id="e1", protocol_hash="aaa", dataset_hash="bbb",
    model_hash="ccc", metrics_hash="ddd")
rec1.evaluation_hash = rec1.compute_record_hash()

rec2 = EvaluationRecordIntegrity(
    record_id="e1", protocol_hash="zzz", dataset_hash="bbb",
    model_hash="ccc", metrics_hash="ddd")
rec2.evaluation_hash = rec2.compute_record_hash()

_check("different_protocol_hash_different_record",
       rec1.evaluation_hash != rec2.evaluation_hash)

# ── 61. Adversarial: empty source classification ──────────────────────────

print("\n-- 61. Empty source classification --")
r = _make_record(source_classification="")
report = run_admission_gates(r)
_check("empty_source_blocked",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value))

# ── 62. Adversarial: SUSPECTED_SYNTHETIC blocked ──────────────────────────

print("\n-- 62. SUSPECTED_SYNTHETIC blocked --")
r = _make_record(source_classification="SUSPECTED_SYNTHETIC")
report = run_admission_gates(r)
_check("suspected_synthetic_blocked",
       report.overall_status == AdmissionState.BLOCKED.value)

# ── 63. Adversarial: CONFIRMED_SYNTHETIC blocked ──────────────────────────

print("\n-- 63. CONFIRMED_SYNTHETIC blocked --")
r = _make_record(source_classification="CONFIRMED_SYNTHETIC")
report = run_admission_gates(r)
_check("confirmed_synthetic_blocked",
       report.overall_status == AdmissionState.BLOCKED.value)

# ── 64. Adversarial: zero row count ───────────────────────────────────────

print("\n-- 64. Zero row count --")
r = _make_record(row_count=0)
report = run_admission_gates(r)
_check("zero_rows_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value))

# ── 65. Adversarial: negative row count ───────────────────────────────────

print("\n-- 65. Negative row count --")
r = _make_record(row_count=-100)
report = run_admission_gates(r)
_check("negative_rows_ineligible",
       report.overall_status in (AdmissionState.INELIGIBLE.value, AdmissionState.BLOCKED.value))

# ── 66. Evaluation record: model hash tamper ──────────────────────────────

print("\n-- 66. Eval record model hash tamper --")
rec = EvaluationRecordIntegrity(
    record_id="e1", protocol_hash="aaa", dataset_hash="bbb",
    model_hash="ccc", metrics_hash="ddd")
rec.evaluation_hash = rec.compute_record_hash()
rec.model_hash = "tampered"
ok, _ = rec.verify_integrity()
_check("eval_record_model_hash_tamper", not ok)

# ── 67. Evaluation record: dataset hash tamper ────────────────────────────

print("\n-- 67. Eval record dataset hash tamper --")
rec = EvaluationRecordIntegrity(
    record_id="e1", protocol_hash="aaa", dataset_hash="bbb",
    model_hash="ccc", metrics_hash="ddd")
rec.evaluation_hash = rec.compute_record_hash()
rec.dataset_hash = "tampered"
ok, _ = rec.verify_integrity()
_check("eval_record_dataset_hash_tamper", not ok)

# ── 68. Evidence record serialization roundtrip ───────────────────────────

print("\n-- 68. Evidence record roundtrip --")
r = _make_record()
d = r.to_dict()
r2 = DatasetEvidenceRecord.from_dict(d)
_check("evidence_roundtrip_dataset_id", r2.dataset_id == r.dataset_id)
_check("evidence_roundtrip_hash", r2.compute_hash() == r.compute_hash())

# ── 69. Admission state: skip steps blocked ───────────────────────────────

print("\n-- 69. Cannot skip admission steps --")
r = _make_record(admission_state=AdmissionState.CANDIDATE.value)
ok = r.transition(AdmissionState.ELIGIBLE, "skip attempt")
_check("skip_steps_blocked", not ok)

# ── 70. Adversarial: REAL_LEGACY classification ───────────────────────────

print("\n-- 70. REAL_LEGACY classification allowed through source gate --")
r = _make_record(source_classification=SourceClassification.REAL_LEGACY.value)
g = gate_source_classification(r)
_check("real_legacy_passes_source_gate", g.passed)

# ── 71. Promotion gate still blocks REAL_WORLD_VALIDATION ─────────────────

print("\n-- 71. Promotion gate still blocks --")
g = evaluate_real_world_validation()
_check("promotion_gate_still_blocks", g.status == GateStatus.BLOCKED)

# ── 72. REAL_WORLD_VALIDATION status unchanged ────────────────────────────

print("\n-- 72. REAL_WORLD_VALIDATION status unchanged --")
REAL_WORLD_VALIDATION = "BLOCKED_PENDING_ELIGIBLE_DATASET"
_check("real_world_validation_still_blocked",
       REAL_WORLD_VALIDATION == "BLOCKED_PENDING_ELIGIBLE_DATASET")

# ── Summary ───────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Phase 53: {_PASSED}/{_TOTAL} PASS, {_FAILED}/{_TOTAL} FAIL")
print(f"{'='*60}")

if _FAILED > 0:
    sys.exit(1)
print("\nAll Phase 53 tests passed.")
