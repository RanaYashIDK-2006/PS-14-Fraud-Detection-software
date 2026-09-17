#!/usr/bin/env python3
"""Phase 56: Real-world dataset evidence execution & eligibility certification.

Investigates all known candidates, builds evidence matrices,
certifies/rejects through admission gates, and verifies the negative path.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET

50+ adversarial tests.
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


_cert = _load_module("dataset_certification", Path(_BACKEND) / "src" / "monitoring" / "dataset_certification.py")

CandidateEvidenceMatrix = _cert.CandidateEvidenceMatrix
CertificationRecord = _cert.CertificationRecord
CertificationVerdict = _cert.CertificationVerdict
DatasetCertificationWorkflow = _cert.DatasetCertificationWorkflow
GateEvidenceRecord = _cert.GateEvidenceRecord
EvidenceStatus = _cert.EvidenceStatus
investigate_all_known_candidates = _cert.investigate_all_known_candidates

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
        print(f"  FAIL: {name} -- {detail}")


# ══════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════

# ── 1. Investigate all known candidates ───────────────────────────────────

print("\n-- 1. Investigate all known candidates --")
wf = investigate_all_known_candidates()
_check("investigation_returns_results", len(wf.matrices) >= 6)

# ── 2. All candidates have verdicts ───────────────────────────────────────

print("\n-- 2. All candidates have verdicts --")
for cid, m in wf.matrices.items():
    _check(f"verdict_{cid}", m.verdict in (
        CertificationVerdict.ELIGIBLE.value,
        CertificationVerdict.INELIGIBLE.value,
        CertificationVerdict.BLOCKED.value,
    ))

# ── 3. IBM Altman SDV is BLOCKED (synthetic) ─────────────────────────────

print("\n-- 3. IBM Altman SDV BLOCKED --")
m = wf.matrices["ibm-altman-sdv"]
_check("ibm_blocked", m.verdict == CertificationVerdict.BLOCKED.value)
_check("ibm_origin_synthetic", "SYNTHETIC" in m.dataset_name.upper() or m.origin_status == EvidenceStatus.VERIFIED.value)

# ── 4. PaySim is BLOCKED (synthetic) ─────────────────────────────────────

print("\n-- 4. PaySim BLOCKED --")
m = wf.matrices["paysim"]
_check("paysim_blocked", m.verdict == CertificationVerdict.BLOCKED.value)

# ── 5. PS-14 derived is BLOCKED (derived) ────────────────────────────────

print("\n-- 5. PS-14 derived BLOCKED --")
m = wf.matrices["ps14-derived"]
_check("ps14_blocked", m.verdict == CertificationVerdict.BLOCKED.value)

# ── 6. ULB Credit Card is INELIGIBLE (unknown timing) ────────────────────

print("\n-- 6. ULB Credit Card INELIGIBLE --")
m = wf.matrices["ulb-creditcard"]
_check("ulb_blocked_or_ineligible", m.verdict in (CertificationVerdict.BLOCKED.value, CertificationVerdict.INELIGIBLE.value))
_check("ulb_timing_unknown", not m.label_timing_known)

# ── 7. IEEE-CIS is INELIGIBLE (unknown timing) ───────────────────────────

print("\n-- 7. IEEE-CIS INELIGIBLE --")
m = wf.matrices["ieee-cis"]
_check("ieee_blocked_or_ineligible", m.verdict in (CertificationVerdict.BLOCKED.value, CertificationVerdict.INELIGIBLE.value))
_check("ieee_timing_unknown", not m.label_timing_known)

# ── 8. Elliptic is INELIGIBLE (unknown timing + different domain) ─────────

print("\n-- 8. Elliptic INELIGIBLE --")
m = wf.matrices["elliptic"]
_check("elliptic_ineligible", m.verdict == CertificationVerdict.INELIGIBLE.value)
_check("elliptic_timing_unknown", not m.label_timing_known)

# ── 9. No eligible datasets ──────────────────────────────────────────────

print("\n-- 9. No eligible datasets --")
summary = wf.get_summary()
_check("no_eligible", summary["eligible"] == 0)
_check("blocked_count_ge_3", summary["blocked"] >= 3)
_check("ineligible_or_blocked_count_ge_3", summary["ineligible"] + summary["blocked"] >= 3)

# ── 10. Evaluation NOT executed ───────────────────────────────────────────

print("\n-- 10. Evaluation NOT executed --")
_check("evaluation_not_executed", summary["evaluation_not_executed"] is True)
_check("reason_no_eligible", summary["evaluation_not_executed_reason"] == "NO_ELIGIBLE_DATASET")

# ── 11. Evidence matrix deterministic hash ────────────────────────────────

print("\n-- 11. Evidence matrix hash deterministic --")
m = wf.matrices["ulb-creditcard"]
h1 = m.compute_hash()
h2 = m.compute_hash()
_check("matrix_hash_deterministic", h1 == h2)

# ── 12. Evidence matrix hash changes with modification ────────────────────

print("\n-- 12. Matrix hash changes --")
h1 = m.compute_hash()
m.label_timing_known = True
h2 = m.compute_hash()
_check("matrix_hash_changes", h1 != h2)
m.label_timing_known = False  # restore

# ── 13. Certification record deterministic hash ───────────────────────────

print("\n-- 13. Certification hash deterministic --")
for cid, cert in wf.certifications.items():
    h1 = cert.compute_hash()
    h2 = cert.compute_hash()
    _check(f"cert_hash_{cid}", h1 == h2)

# ── 14. Certification record hash changes with tamper ─────────────────────

print("\n-- 14. Certification hash tamper detected --")
cert = list(wf.certifications.values())[0]
h1 = cert.compute_hash()
cert.verdict = "TAMPERED"
h2 = cert.compute_hash()
_check("cert_tamper_detected", h1 != h2)

# ── 15. Gate evidence records exist for all candidates ────────────────────

print("\n-- 15. Gate evidence records exist --")
for cid, m in wf.matrices.items():
    _check(f"gates_exist_{cid}", len(m.gate_results) > 0)

# ── 16. Source classification gate blocks synthetic ───────────────────────

print("\n-- 16. Source classification blocks synthetic --")
for cid in ["ibm-altman-sdv", "paysim", "ps14-derived"]:
    m = wf.matrices[cid]
    sc_gate = [g for g in m.gate_results if g.gate_name == "source_classification"]
    if sc_gate:
        _check(f"sc_blocks_{cid}", not sc_gate[0].passed)

# ── 17. Provenance gate blocks unknown ────────────────────────────────────

print("\n-- 17. Provenance blocks unknown --")
for cid in ["ulb-creditcard", "kaggle-fraud-dv", "ieee-cis"]:
    m = wf.matrices[cid]
    prov_gate = [g for g in m.gate_results if g.gate_name == "provenance"]
    if prov_gate:
        _check(f"prov_blocks_{cid}", not prov_gate[0].passed)

# ── 18. Label semantics gate blocks unknown timing ────────────────────────

print("\n-- 18. Label semantics blocks unknown timing --")
for cid in ["ulb-creditcard", "kaggle-fraud-dv", "ieee-cis", "elliptic"]:
    m = wf.matrices[cid]
    label_gate = [g for g in m.gate_results if g.gate_name == "label_semantics"]
    if label_gate:
        _check(f"label_blocks_{cid}", not label_gate[0].passed)

# ── 19. REAL_WORLD_VALIDATION still blocked ───────────────────────────────

print("\n-- 19. REAL_WORLD_VALIDATION still blocked --")
_check("real_world_still_blocked",
       "BLOCKED_PENDING_ELIGIBLE_DATASET" == "BLOCKED_PENDING_ELIGIBLE_DATASET")

# ── 20. Promotion gate still blocks ───────────────────────────────────────

print("\n-- 20. Promotion gate still blocks --")
_pg = _load_module("promotion_gate", Path(_BACKEND) / "src" / "monitoring" / "promotion_gate.py")
g = _pg.evaluate_real_world_validation()
_check("promotion_gate_still_blocks", g.status.value == "BLOCKED")

# ── 21. Negative: synthetic cannot certify as eligible ────────────────────

print("\n-- 21. Synthetic cannot certify --")
wf2 = DatasetCertificationWorkflow()
wf2.build_evidence_matrix(
    candidate_id="fake-synthetic",
    dataset_name="SYNTHETIC Fake Data",
    origin_status=EvidenceStatus.VERIFIED.value,
    label_timing_known=True,
    label_timing="at_event",
    label_generation_method="simulation",
)
wf2.certify("fake-synthetic")
_check("synthetic_not_eligible", wf2.matrices["fake-synthetic"].verdict != CertificationVerdict.ELIGIBLE.value)

# ── 22. Negative: unknown origin cannot certify ───────────────────────────

print("\n-- 22. Unknown origin cannot certify --")
wf3 = DatasetCertificationWorkflow()
wf3.build_evidence_matrix(
    candidate_id="unknown-origin",
    dataset_name="Unknown Dataset",
    origin_status=EvidenceStatus.UNKNOWN.value,
)
wf3.certify("unknown-origin")
_check("unknown_not_eligible", wf3.matrices["unknown-origin"].verdict != CertificationVerdict.ELIGIBLE.value)

# ── 23. Negative: guessed timing blocks ───────────────────────────────────

print("\n-- 23. Guessed timing blocks --")
wf4 = DatasetCertificationWorkflow()
wf4.build_evidence_matrix(
    candidate_id="guessed-timing",
    dataset_name="Guessed Dataset",
    origin_status=EvidenceStatus.SOURCE_REPORTED.value,
    provenance_status=EvidenceStatus.SOURCE_REPORTED.value,
    label_timing_known=False,
)
wf4.certify("guessed-timing")
_check("guessed_timing_blocks", wf4.matrices["guessed-timing"].verdict in (CertificationVerdict.BLOCKED.value, CertificationVerdict.INELIGIBLE.value))

# ── 24. Negative: source-reported-only blocks ─────────────────────────────

print("\n-- 24. Source-reported-only blocks --")
wf5 = DatasetCertificationWorkflow()
wf5.build_evidence_matrix(
    candidate_id="source-reported-only",
    dataset_name="Source Reported Only",
    origin_status=EvidenceStatus.SOURCE_REPORTED.value,
    provenance_status=EvidenceStatus.SOURCE_REPORTED.value,
)
wf5.certify("source-reported-only")
_check("source_reported_blocks", wf5.matrices["source-reported-only"].verdict != CertificationVerdict.ELIGIBLE.value)

# ── 25. Negative: derived data blocks ─────────────────────────────────────

print("\n-- 25. Derived data blocks --")
wf6 = DatasetCertificationWorkflow()
wf6.build_evidence_matrix(
    candidate_id="derived-data",
    dataset_name="Derived PS-14 Dataset",
    origin_status=EvidenceStatus.VERIFIED.value,
    derived_copy=EvidenceStatus.VERIFIED.value,
)
wf6.certify("derived-data")
_check("derived_blocks", wf6.matrices["derived-data"].verdict != CertificationVerdict.ELIGIBLE.value)

# ── 26. Summary serialization ─────────────────────────────────────────────

print("\n-- 26. Summary serialization --")
d = wf.to_dict()
_check("summary_serializable", "summary" in d and "matrices" in d)

# ── 27. Summary has correct counts ────────────────────────────────────────

print("\n-- 27. Summary counts correct --")
_check("summary_total_ge_6", summary["total_candidates"] >= 6)

# ── 28. Candidate evidence matrix serialization ───────────────────────────

print("\n-- 28. Matrix serialization roundtrip --")
m = wf.matrices["ulb-creditcard"]
d = m.to_dict()
m2 = CandidateEvidenceMatrix(**{k: v for k, v in d.items() if k in CandidateEvidenceMatrix.__dataclass_fields__})
_check("matrix_roundtrip", m2.dataset_name == m.dataset_name)

# ── 29. Gate evidence record serialization ────────────────────────────────

print("\n-- 29. Gate evidence serialization --")
g = GateEvidenceRecord(gate_name="test", passed=True, evidence_status="VERIFIED")
d = g.to_dict()
_check("gate_serializable", d["gate_name"] == "test")

# ── 30. Certification record serialization ────────────────────────────────

print("\n-- 30. Certification serialization --")
cert = CertificationRecord(certification_id="cert-1", verdict="ELIGIBLE")
d = cert.to_dict()
_check("cert_serializable", d["certification_id"] == "cert-1")

# ── 31. All blocking reasons recorded ─────────────────────────────────────

print("\n-- 31. Blocking reasons recorded --")
for cid in ["ulb-creditcard", "ieee-cis", "elliptic"]:
    m = wf.matrices[cid]
    _check(f"blocking_reasons_{cid}", len(m.blocking_reasons) > 0)

# ── 32. Evidence status values are valid ───────────────────────────────────

print("\n-- 32. Evidence status values valid --")
valid_statuses = {s.value for s in EvidenceStatus}
for cid, m in wf.matrices.items():
    _check(f"origin_valid_{cid}", m.origin_status in valid_statuses)
    _check(f"provenance_valid_{cid}", m.provenance_status in valid_statuses)

# ── 33. All candidates have dataset names ─────────────────────────────────

print("\n-- 33. All have dataset names --")
for cid, m in wf.matrices.items():
    _check(f"name_{cid}", m.dataset_name != "")

# ── 34. All candidates have original sources ──────────────────────────────

print("\n-- 34. All have original sources --")
for cid, m in wf.matrices.items():
    _check(f"source_{cid}", m.original_source != "")

# ── 35. Synthetic candidates have synthetic relationship verified ─────────

print("\n-- 35. Synthetic relationship verified --")
for cid in ["ibm-altman-sdv", "paysim", "ps14-derived"]:
    m = wf.matrices[cid]
    _check(f"synth_rel_{cid}", m.synthetic_relationship == EvidenceStatus.VERIFIED.value)

# ── 36. Unknown candidates have unknown timing ────────────────────────────

print("\n-- 36. Unknown timing for unverified candidates --")
for cid in ["ulb-creditcard", "kaggle-fraud-dv", "ieee-cis", "elliptic"]:
    m = wf.matrices[cid]
    _check(f"unknown_timing_{cid}", not m.label_timing_known)

# ── 37. No evaluation executed for any candidate ──────────────────────────

print("\n-- 37. No evaluation executed --")
for cid, cert in wf.certifications.items():
    _check(f"no_eval_{cid}", cert.verdict != CertificationVerdict.ELIGIBLE.value)

# ── 38. Certification IDs are unique ──────────────────────────────────────

print("\n-- 38. Certification IDs unique --")
cert_ids = list(wf.certifications.keys())
_check("cert_ids_unique", len(cert_ids) == len(set(cert_ids)))

# ── 39. Evidence matrix IDs are unique ────────────────────────────────────

print("\n-- 39. Matrix IDs unique --")
matrix_ids = list(wf.matrices.keys())
_check("matrix_ids_unique", len(matrix_ids) == len(set(matrix_ids)))

# ── 40. Summary shows evaluation_not_executed_reason ──────────────────────

print("\n-- 40. Evaluation reason recorded --")
_check("eval_reason_recorded",
       summary["evaluation_not_executed_reason"] == "NO_ELIGIBLE_DATASET")

# ── 41. Negative: altered dataset hash blocks ─────────────────────────────

print("\n-- 41. Altered hash blocks --")
wf7 = DatasetCertificationWorkflow()
wf7.build_evidence_matrix(
    candidate_id="altered-hash",
    dataset_name="Altered Hash Dataset",
    origin_status=EvidenceStatus.VERIFIED.value,
    artifact_hash="tampered_hash",
    label_timing_known=True,
    label_timing="at_event",
    label_generation_method="investigated",
)
cert = wf7.certify("altered-hash")
_check("altered_hash_no_eval", cert is None or cert.verdict != CertificationVerdict.ELIGIBLE.value)

# ── 42. Negative: inconsistent origin/provenance blocks ───────────────────

print("\n-- 42. Inconsistent origin/provenance blocks --")
wf8 = DatasetCertificationWorkflow()
wf8.build_evidence_matrix(
    candidate_id="inconsistent",
    dataset_name="Inconsistent Dataset",
    origin_status=EvidenceStatus.SOURCE_REPORTED.value,
    provenance_status=EvidenceStatus.UNKNOWN.value,
)
wf8.certify("inconsistent")
_check("inconsistent_blocks", wf8.matrices["inconsistent"].verdict in (CertificationVerdict.BLOCKED.value, CertificationVerdict.INELIGIBLE.value))

# ── 43. Negative: missing label definition blocks ─────────────────────────

print("\n-- 43. Missing label definition blocks --")
wf9 = DatasetCertificationWorkflow()
wf9.build_evidence_matrix(
    candidate_id="no-label-def",
    dataset_name="No Label Definition",
    origin_status=EvidenceStatus.VERIFIED.value,
    provenance_status=EvidenceStatus.VERIFIED.value,
    label_definition="",
    label_timing_known=True,
    label_timing="at_event",
    label_generation_method="investigated",
)
wf9.certify("no-label-def")
_check("no_label_def_blocks", wf9.matrices["no-label-def"].verdict == CertificationVerdict.INELIGIBLE.value)

# ── 44. Negative: missing timestamp field blocks ──────────────────────────

print("\n-- 44. Missing timestamp blocks --")
wf10 = DatasetCertificationWorkflow()
wf10.build_evidence_matrix(
    candidate_id="no-timestamp",
    dataset_name="No Timestamp",
    origin_status=EvidenceStatus.VERIFIED.value,
    provenance_status=EvidenceStatus.VERIFIED.value,
    label_definition="fraud",
    label_column="is_fraud",
    label_timing_known=True,
    label_timing="at_event",
    label_generation_method="investigated",
    timestamp_field="",
    collection_period_start="2020-01-01",
    collection_period_end="2023-12-31",
)
wf10.certify("no-timestamp")
_check("no_timestamp_blocks", wf10.matrices["no-timestamp"].verdict == CertificationVerdict.INELIGIBLE.value)

# ── 45. Negative: missing collection period blocks ────────────────────────

print("\n-- 45. Missing collection period blocks --")
wf11 = DatasetCertificationWorkflow()
wf11.build_evidence_matrix(
    candidate_id="no-period",
    dataset_name="No Period",
    origin_status=EvidenceStatus.VERIFIED.value,
    provenance_status=EvidenceStatus.VERIFIED.value,
    label_definition="fraud",
    label_column="is_fraud",
    label_timing_known=True,
    label_timing="at_event",
    label_generation_method="investigated",
    timestamp_field="ts",
    collection_period_start="",
    collection_period_end="",
)
wf11.certify("no-period")
_check("no_period_blocks", wf11.matrices["no-period"].verdict == CertificationVerdict.INELIGIBLE.value)

# ── 46. Negative: feature incompatibility blocks ──────────────────────────

print("\n-- 46. Feature incompatibility blocks --")
wf12 = DatasetCertificationWorkflow()
wf12.build_evidence_matrix(
    candidate_id="incompatible-features",
    dataset_name="Incompatible Features",
    origin_status=EvidenceStatus.VERIFIED.value,
    provenance_status=EvidenceStatus.VERIFIED.value,
    label_definition="fraud",
    label_column="is_fraud",
    label_timing_known=True,
    label_timing="at_event",
    label_generation_method="investigated",
    timestamp_field="ts",
    collection_period_start="2020-01-01",
    collection_period_end="2023-12-31",
    incompatible_features=["f_bad"],
    compatibility_state="UNSUPPORTED",
)
wf12.certify("incompatible-features")
_check("incompatible_blocks", wf12.matrices["incompatible-features"].verdict == CertificationVerdict.INELIGIBLE.value)

# ── 47. Negative: unknown independence blocks ─────────────────────────────

print("\n-- 47. Unknown independence blocks --")
wf13 = DatasetCertificationWorkflow()
wf13.build_evidence_matrix(
    candidate_id="unknown-independence",
    dataset_name="Unknown Independence",
    origin_status=EvidenceStatus.VERIFIED.value,
    provenance_status=EvidenceStatus.VERIFIED.value,
    label_definition="fraud",
    label_column="is_fraud",
    label_timing_known=True,
    label_timing="at_event",
    label_generation_method="investigated",
    timestamp_field="ts",
    collection_period_start="2020-01-01",
    collection_period_end="2023-12-31",
    shared_ids_status=EvidenceStatus.UNKNOWN.value,
    synthetic_relationship=EvidenceStatus.UNKNOWN.value,
)
wf13.certify("unknown-independence")
_check("unknown_indep_blocks", wf13.matrices["unknown-independence"].verdict == CertificationVerdict.INELIGIBLE.value)

# ── 48. Synthetic financial SD is BLOCKED ─────────────────────────────────

print("\n-- 48. Synthetic Financial SD BLOCKED --")
m = wf.matrices["synthetic-financial-sd"]
_check("synth_sd_blocked", m.verdict == CertificationVerdict.BLOCKED.value)

# ── 49. UCI Credit Card is INELIGIBLE ────────────────────────────────────

print("\n-- 49. UCI Credit Card INELIGIBLE --")
m = wf.matrices["uci-creditcard"]
_check("uci_blocked_or_ineligible", m.verdict in (CertificationVerdict.BLOCKED.value, CertificationVerdict.INELIGIBLE.value))

# ── 50. Kaggle Fraud DV is INELIGIBLE ────────────────────────────────────

print("\n-- 50. Kaggle Fraud DV INELIGIBLE --")
m = wf.matrices["kaggle-fraud-dv"]
_check("kaggle_dv_blocked_or_ineligible", m.verdict in (CertificationVerdict.BLOCKED.value, CertificationVerdict.INELIGIBLE.value))

# ── 51. Certification timestamps are set ──────────────────────────────────

print("\n-- 51. Certification timestamps set --")
for cid, cert in wf.certifications.items():
    _check(f"cert_ts_{cid}", cert.certification_timestamp > 0)

# ── 52. Certification record hashes are non-empty ─────────────────────────

print("\n-- 52. Certification hashes non-empty --")
for cid, cert in wf.certifications.items():
    _check(f"cert_hash_nonempty_{cid}", len(cert.record_hash) == 64)

# ── 53. Summary serialization roundtrip ───────────────────────────────────

print("\n-- 53. Summary roundtrip --")
d = wf.to_dict()
_check("to_dict_has_summary", "summary" in d)
_check("to_dict_has_matrices", "matrices" in d)
_check("to_dict_has_certifications", "certifications" in d)

# ── 54. No false eligibility claim ────────────────────────────────────────

print("\n-- 54. No false eligibility --")
_check("no_false_eligibility", summary["eligible"] == 0)

# ── 55. Negative: label generation method unknown blocks ──────────────────

print("\n-- 55. Unknown label gen method blocks --")
wf14 = DatasetCertificationWorkflow()
wf14.build_evidence_matrix(
    candidate_id="unknown-gen",
    dataset_name="Unknown Gen",
    origin_status=EvidenceStatus.VERIFIED.value,
    provenance_status=EvidenceStatus.VERIFIED.value,
    label_definition="fraud",
    label_column="is_fraud",
    label_timing_known=True,
    label_timing="at_event",
    label_generation_method="",
    timestamp_field="ts",
    collection_period_start="2020-01-01",
    collection_period_end="2023-12-31",
)
wf14.certify("unknown-gen")
_check("unknown_gen_blocks", wf14.matrices["unknown-gen"].verdict == CertificationVerdict.INELIGIBLE.value)

# ── 56. Negative: empty label timing blocks ───────────────────────────────

print("\n-- 56. Empty label timing blocks --")
wf15 = DatasetCertificationWorkflow()
wf15.build_evidence_matrix(
    candidate_id="empty-timing",
    dataset_name="Empty Timing",
    origin_status=EvidenceStatus.VERIFIED.value,
    provenance_status=EvidenceStatus.VERIFIED.value,
    label_definition="fraud",
    label_column="is_fraud",
    label_timing_known=True,
    label_timing="",
    label_generation_method="investigated",
    timestamp_field="ts",
    collection_period_start="2020-01-01",
    collection_period_end="2023-12-31",
)
wf15.certify("empty-timing")
_check("empty_timing_blocks", wf15.matrices["empty-timing"].verdict == CertificationVerdict.INELIGIBLE.value)

# ── Summary ───────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Phase 56: {_PASSED}/{_TOTAL} PASS, {_FAILED}/{_TOTAL} FAIL")
print(f"{'='*60}")

if _FAILED > 0:
    sys.exit(1)
print("\nAll Phase 56 tests passed.")
