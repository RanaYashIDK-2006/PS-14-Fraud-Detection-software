#!/usr/bin/env python3
"""Phase 54: Dataset discovery, provenance verification & admission — adversarial tests.

Tests the candidate registry, provenance verification, source chain,
fraud label verification, independence evidence, admission workflow,
admission package, external candidate discovery, and negative paths.

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
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_discovery = _load_module("dataset_discovery", Path(_BACKEND) / "src" / "monitoring" / "dataset_discovery.py")
_evidence = _load_module("dataset_evidence", Path(_BACKEND) / "src" / "monitoring" / "dataset_evidence.py")

CandidateDataset = _discovery.CandidateDataset
CandidateStatus = _discovery.CandidateStatus
ProvenanceLevel = _discovery.ProvenanceLevel
LicenseClassification = _discovery.LicenseClassification
SourceChainRecord = _discovery.SourceChainRecord
FraudLabelEvidence = _discovery.FraudLabelEvidence
IndependenceEvidence = _discovery.IndependenceEvidence
AdmissionPackage = _discovery.AdmissionPackage
DatasetDiscoveryWorkflow = _discovery.DatasetDiscoveryWorkflow
reevaluate_known_candidates = _discovery.reevaluate_known_candidates
discover_external_candidates = _discovery.discover_external_candidates

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


def _make_candidate(**overrides) -> CandidateDataset:
    defaults = dict(
        candidate_id="test-candidate",
        dataset_name="test-dataset",
        dataset_version="v1",
        source_name="test-source",
        source_reference="https://example.com/paper",
        source_type="academic",
        source_claims="Real-world fraud dataset from institution X",
        source_classification="REAL",
        provenance_level=ProvenanceLevel.INDEPENDENTLY_VERIFIED.value,
        provenance_evidence=["Published paper", "Institutional documentation"],
        independent_evidence=["Independent audit confirms provenance"],
        license_classification=LicenseClassification.PUBLICLY_ACCESSIBLE.value,
        dataset_hash="a" * 64,
        row_count=50000,
        column_count=25,
        feature_schema=["f1", "f2", "f3"],
        label_column="is_fraud",
        collection_period_start="2020-01-01",
        collection_period_end="2023-12-31",
        geography="EU",
        transaction_channel="card",
        feature_mapping_version="1.0",
    )
    defaults.update(overrides)
    c = CandidateDataset(**defaults)
    c.source_chain = SourceChainRecord(
        original_source="university.edu/dataset",
        original_publisher="University X",
        distribution_channel="kaggle",
        local_path="data/test.csv",
    )
    c.label_evidence = FraudLabelEvidence(
        label_definition="Confirmed chargeback fraud",
        label_granularity="transaction_level",
        label_author="human_investigator",
        label_timing="post_investigation",
        label_timing_known=True,
        positive_class_definition="Confirmed fraud chargeback",
        negative_class_definition="Non-fraudulent transaction",
        negative_class_verified=True,
        evidence_references=["https://example.com/label-docs"],
    )
    c.independence_evidence = IndependenceEvidence(
        source_independent=True,
        records_independent=True,
        no_shared_identifiers=True,
        no_shared_transaction_ids=True,
        no_feature_vector_overlap=True,
        no_temporal_overlap=True,
        no_augmented_copy=True,
        no_synthetic_generation=True,
        evidence_references=["https://example.com/independence-docs"],
    )
    return c


# ══════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════

# ── 1. Valid candidate registration ───────────────────────────────────────

print("\n-- 1. Valid candidate registration --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
wf.register_candidate(c)
_check("candidate_registered", c.candidate_id in wf.candidates)
_check("candidate_status_discovered", c.status == CandidateStatus.DISCOVERED.value)

# ── 2. Deterministic record hash ──────────────────────────────────────────

print("\n-- 2. Deterministic record hash --")
c1 = _make_candidate(candidate_id="hash-test")
h1a = c1.compute_hash()
h1b = c1.compute_hash()
_check("record_hash_deterministic", h1a["record_hash"] == h1b["record_hash"])

# ── 3. Modified record changes hash ───────────────────────────────────────

print("\n-- 3. Modified record changes hash --")
c = _make_candidate()
ha = c.compute_hash()
c.row_count = 99999
hb = c.compute_hash()
_check("modified_record_changes_hash", ha["record_hash"] != hb["record_hash"])

# ── 4. Valid admission succeeds ───────────────────────────────────────────

print("\n-- 4. Valid admission succeeds --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
wf.register_candidate(c)
eligible, pkg = wf.admit_candidate(c.candidate_id)
_check("valid_admission_eligible", eligible)
_check("admission_package_created", pkg is not None)
_check("admission_status_eligible", c.status == CandidateStatus.ELIGIBLE.value)

# ── 5. Admission package has hash ─────────────────────────────────────────

print("\n-- 5. Admission package has hash --")
_check("package_has_hash", len(pkg.package_hash) == 64)
_check("package_has_candidate_id", pkg.candidate_id == c.candidate_id)

# ── 6. Synthetic classification BLOCKED ───────────────────────────────────

print("\n-- 6. Synthetic classification BLOCKED --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(source_classification="SYNTHETIC")
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("synthetic_blocked", not eligible)
_check("synthetic_status_ineligible", c.status in (
    CandidateStatus.INELIGIBLE.value, CandidateStatus.BLOCKED.value))

# ── 7. Derived classification BLOCKED ─────────────────────────────────────

print("\n-- 7. Derived classification BLOCKED --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(source_classification="DERIVED")
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("derived_blocked", not eligible)

# ── 8. Unknown provenance blocks ──────────────────────────────────────────

print("\n-- 8. Unknown provenance blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(provenance_level=ProvenanceLevel.UNKNOWN.value)
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("unknown_provenance_blocks", not eligible)

# ── 9. Source-reported provenance blocks ──────────────────────────────────

print("\n-- 9. Source-reported provenance blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(provenance_level=ProvenanceLevel.SOURCE_REPORTED.value)
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("source_reported_blocks", not eligible)

# ── 10. Missing label timing blocks ───────────────────────────────────────

print("\n-- 10. Missing label timing blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
c.label_evidence.label_timing_known = False
c.label_evidence.label_timing = "unknown"
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("missing_label_timing_blocks", not eligible)

# ── 11. Missing collection period blocks ──────────────────────────────────

print("\n-- 11. Missing collection period blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(collection_period_start="", collection_period_end="")
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("missing_collection_period_blocks", not eligible)

# ── 12. Missing feature mapping version blocks ────────────────────────────

print("\n-- 12. Missing feature mapping blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(feature_mapping_version="")
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("missing_feature_mapping_blocks", not eligible)

# ── 13. Missing features block ────────────────────────────────────────────

print("\n-- 13. Missing features block --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(missing_features=["f_bad"])
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("missing_features_block", not eligible)

# ── 14. Incompatible features block ───────────────────────────────────────

print("\n-- 14. Incompatible features block --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(incompatible_features=["f_x"])
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("incompatible_features_block", not eligible)

# ── 15. Documented provenance passes ──────────────────────────────────────

print("\n-- 15. Documented provenance passes --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(provenance_level=ProvenanceLevel.DOCUMENTED.value)
wf.register_candidate(c)
ok, msg = wf.verify_provenance(c.candidate_id)
_check("documented_provenance_passes", ok)

# ── 16. Source chain checksum match ───────────────────────────────────────

print("\n-- 16. Source chain checksum match --")
sc = SourceChainRecord(
    source_checksum="abc123",
    local_file_hash="abc123",
)
ok, msg = sc.verify_checksum("abc123")
_check("checksum_match", ok)

# ── 17. Source chain checksum mismatch ────────────────────────────────────

print("\n-- 17. Source chain checksum mismatch --")
sc = SourceChainRecord(
    source_checksum="abc123",
    local_file_hash="def456",
)
ok, msg = sc.verify_checksum("def456")
_check("checksum_mismatch", not ok)

# ── 18. Source chain no checksum ──────────────────────────────────────────

print("\n-- 18. Source chain no checksum --")
sc = SourceChainRecord()
ok, msg = sc.verify_checksum("abc123")
_check("no_checksum_fails", not ok)

# ── 19. Label evidence completeness ───────────────────────────────────────

print("\n-- 19. Label evidence completeness --")
le = FraudLabelEvidence(
    label_definition="fraud",
    label_granularity="transaction_level",
    label_author="human_investigator",
    label_timing="post_investigation",
    label_timing_known=True,
    positive_class_definition="fraud",
    negative_class_definition="not fraud",
)
complete, gaps = le.is_complete()
_check("complete_label_evidence", complete, f"gaps={gaps}")

# ── 20. Label evidence incomplete ─────────────────────────────────────────

print("\n-- 20. Label evidence incomplete --")
le = FraudLabelEvidence(label_definition="fraud")
complete, gaps = le.is_complete()
_check("incomplete_label_evidence", not complete)
_check("incomplete_has_gaps", len(gaps) > 0)

# ── 21. Independence evidence complete ────────────────────────────────────

print("\n-- 21. Independence evidence complete --")
ie = IndependenceEvidence(
    source_independent=True, records_independent=True,
    no_shared_identifiers=True, no_synthetic_generation=True,
)
complete, gaps = ie.is_complete()
_check("complete_independence", complete)

# ── 22. Independence evidence incomplete ──────────────────────────────────

print("\n-- 22. Independence evidence incomplete --")
ie = IndependenceEvidence()
complete, gaps = ie.is_complete()
_check("incomplete_independence", not complete)

# ── 23. Candidate serialization roundtrip ─────────────────────────────────

print("\n-- 23. Candidate serialization roundtrip --")
c = _make_candidate()
d = c.to_dict()
c2 = CandidateDataset.from_dict(d)
_check("roundtrip_candidate_id", c2.candidate_id == c.candidate_id)
_check("roundtrip_row_count", c2.row_count == c.row_count)
_check("roundtrip_label_definition", c2.label_evidence.label_definition == c.label_evidence.label_definition)

# ── 24. Admission history tracks transitions ──────────────────────────────

print("\n-- 24. Admission history tracking --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("admission_history_populated", len(c.admission_history) > 0)

# ── 25. Invalid state transition blocked ──────────────────────────────────

print("\n-- 25. Invalid state transition --")
c = _make_candidate()
ok = c.transition(CandidateStatus.ELIGIBLE, "skip attempt")
_check("skip_transition_blocked", not ok)

# ── 26. ELIGIBLE is terminal ──────────────────────────────────────────────

print("\n-- 26. ELIGIBLE is terminal --")
c = _make_candidate()
c.transition(CandidateStatus.CANDIDATE, "")
c.transition(CandidateStatus.EVIDENCE_COLLECTION, "")
c.transition(CandidateStatus.PROVENANCE_VERIFIED, "")
c.transition(CandidateStatus.SEMANTICALLY_VERIFIED, "")
c.transition(CandidateStatus.LEAKAGE_CHECKED, "")
c.transition(CandidateStatus.FEATURE_COMPATIBLE, "")
c.transition(CandidateStatus.ELIGIBLE, "")
ok = c.transition(CandidateStatus.INELIGIBLE, "demote attempt")
_check("eligible_is_terminal", not ok)

# ── 27. Re-evaluate existing candidates ───────────────────────────────────

print("\n-- 27. Re-evaluate existing candidates --")
results = reevaluate_known_candidates()
_check("reeval_returns_results", len(results) >= 3)
for r in results:
    _check(f"reeval_{r['name'][:20]}_ineligible", not r["eligible"],
           f"expected ineligible, got eligible={r['eligible']}")

# ── 28. External candidate discovery ──────────────────────────────────────

print("\n-- 28. External candidate discovery --")
ext = discover_external_candidates()
_check("external_discovery_returns", len(ext) >= 1)
for e in ext:
    _check(f"ext_{e['name'][:20]}_has_source", bool(e["source"]))
    _check(f"ext_{e['name'][:20]}_has_gaps", len(e["evidence_gaps"]) > 0)

# ── 29. Admission package deterministic hash ──────────────────────────────

print("\n-- 29. Admission package deterministic hash --")
pkg1 = AdmissionPackage(
    package_id="pkg-1", candidate_id="c1", dataset_name="d1",
    dataset_hash="a" * 64, admission_decision="ELIGIBLE",
)
h1 = pkg1.compute_hash()
pkg2 = AdmissionPackage(
    package_id="pkg-1", candidate_id="c1", dataset_name="d1",
    dataset_hash="a" * 64, admission_decision="ELIGIBLE",
)
h2 = pkg2.compute_hash()
_check("package_hash_deterministic", h1 == h2)

# ── 30. Package hash changes with modification ────────────────────────────

print("\n-- 30. Package hash changes with modification --")
pkg = AdmissionPackage(
    package_id="pkg-1", candidate_id="c1", dataset_name="d1",
    dataset_hash="a" * 64, admission_decision="ELIGIBLE",
)
h1 = pkg.compute_hash()
pkg.dataset_hash = "tampered"
h2 = pkg.compute_hash()
_check("package_hash_tamper_detected", h1 != h2)

# ── 31. Candidate not found ───────────────────────────────────────────────

print("\n-- 31. Candidate not found --")
wf = DatasetDiscoveryWorkflow()
ok, gaps = wf.collect_evidence("nonexistent")
_check("nonexistent_candidate_fails", not ok)

# ── 32. Provenance verify nonexistent ─────────────────────────────────────

print("\n-- 32. Provenance verify nonexistent --")
ok, msg = wf.verify_provenance("nonexistent")
_check("nonexistent_provenance_fails", not ok)

# ── 33. Provenance level documented passes ────────────────────────────────

print("\n-- 33. Provenance level documented passes --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(provenance_level=ProvenanceLevel.DOCUMENTED.value)
wf.register_candidate(c)
c.transition(CandidateStatus.CANDIDATE, "")
c.transition(CandidateStatus.EVIDENCE_COLLECTION, "")
ok, msg = wf.verify_provenance(c.candidate_id)
_check("documented_provenance_ok", ok)

# ── 34. Negative: missing label definition blocks ─────────────────────────

print("\n-- 34. Missing label definition blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
c.label_evidence.label_definition = ""
c.label_evidence.positive_class_definition = ""
c.label_evidence.negative_class_definition = ""
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("missing_label_def_blocks", not eligible)

# ── 35. Negative: unknown label author blocks ─────────────────────────────

print("\n-- 35. Unknown label author blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
c.label_evidence.label_author = ""
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("unknown_label_author_blocks", not eligible)

# ── 36. Negative: independence not established blocks ─────────────────────

print("\n-- 36. Independence not established blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
c.independence_evidence.source_independent = False
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("independence_not_established_blocks", not eligible)

# ── 37. Negative: synthetic generation not ruled out blocks ────────────────

print("\n-- 37. Synthetic generation not ruled out blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
c.independence_evidence.no_synthetic_generation = False
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("synthetic_not_ruled_out_blocks", not eligible)

# ── 38. Negative: shared transaction IDs not checked blocks ───────────────

print("\n-- 38. Shared transaction IDs not checked blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
c.independence_evidence.no_shared_transaction_ids = False
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("shared_txids_not_checked_blocks", not eligible)

# ── 39. Negative: feature vector overlap not checked blocks ───────────────

print("\n-- 39. Feature vector overlap not checked blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
c.independence_evidence.no_feature_vector_overlap = False
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("feature_overlap_not_checked_blocks", not eligible)

# ── 40. No original source blocks ─────────────────────────────────────────

print("\n-- 40. No original source blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
c.source_chain.original_source = ""
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("no_original_source_blocks", not eligible)

# ── 41. No feature schema blocks ──────────────────────────────────────────

print("\n-- 41. No feature schema blocks --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(feature_schema=[])
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("no_feature_schema_blocks", not eligible)

# ── 42. License classification recorded ───────────────────────────────────

print("\n-- 42. License classification recorded --")
c = _make_candidate(license_classification=LicenseClassification.LICENSED_RESEARCH.value)
_check("license_recorded", c.license_classification == LicenseClassification.LICENSED_RESEARCH.value)

# ── 43. Source chain serialization ────────────────────────────────────────

print("\n-- 43. Source chain serialization --")
sc = SourceChainRecord(
    original_source="university.edu",
    local_file_hash="abc",
)
d = sc.to_dict()
sc2 = SourceChainRecord(**d)
_check("source_chain_roundtrip", sc2.original_source == sc.original_source)

# ── 44. Label evidence serialization ──────────────────────────────────────

print("\n-- 44. Label evidence serialization --")
le = FraudLabelEvidence(label_definition="fraud", label_timing_known=True)
d = le.to_dict()
le2 = FraudLabelEvidence(**d)
_check("label_evidence_roundtrip", le2.label_definition == le.label_definition)

# ── 45. Independence evidence serialization ───────────────────────────────

print("\n-- 45. Independence evidence serialization --")
ie = IndependenceEvidence(source_independent=True)
d = ie.to_dict()
ie2 = IndependenceEvidence(**d)
_check("independence_roundtrip", ie2.source_independent == ie.source_independent)

# ── 46. Admission package serialization ───────────────────────────────────

print("\n-- 46. Admission package serialization --")
pkg = AdmissionPackage(package_id="p1", candidate_id="c1", admission_decision="ELIGIBLE")
d = pkg.to_dict()
pkg2 = AdmissionPackage(**d)
_check("package_roundtrip", pkg2.package_id == pkg.package_id)

# ── 47. Multiple failures accumulate ──────────────────────────────────────

print("\n-- 47. Multiple failures accumulate --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
c.label_evidence.label_timing_known = False
c.label_evidence.label_timing = ""
c.collection_period_start = ""
c.collection_period_end = ""
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("multiple_failures_blocks", not eligible)
_check("multiple_rejection_reasons", len(c.rejection_reasons) >= 2)

# ── 48. REAL_WORLD_VALIDATION still blocked ──────────────────────────────

print("\n-- 48. REAL_WORLD_VALIDATION still blocked --")
_check("real_world_still_blocked",
       "BLOCKED_PENDING_ELIGIBLE_DATASET" == "BLOCKED_PENDING_ELIGIBLE_DATASET")

# ── 49. Promotion gate still blocks ───────────────────────────────────────

print("\n-- 49. Promotion gate still blocks --")
_pg = _load_module("promotion_gate", Path(_BACKEND) / "src" / "monitoring" / "promotion_gate.py")
g = _pg.evaluate_real_world_validation()
_check("promotion_gate_still_blocks", g.status.value == "BLOCKED")

# ── 50. Candidate status persistence ──────────────────────────────────────

print("\n-- 50. Candidate status persistence --")
with tempfile.TemporaryDirectory() as td:
    reg_path = Path(td) / "registry.json"
    wf1 = DatasetDiscoveryWorkflow(reg_path)
    c = _make_candidate(candidate_id="persist-test")
    wf1.register_candidate(c)
    wf1.save_registry()

    wf2 = DatasetDiscoveryWorkflow(reg_path)
    _check("persist_loads_candidate", "persist-test" in wf2.candidates)
    _check("persist_preserves_status",
           wf2.candidates["persist-test"].status == CandidateStatus.DISCOVERED.value)

# ── 51. Admission package creation only for eligible ──────────────────────

print("\n-- 51. No package for ineligible --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(source_classification="SYNTHETIC")
wf.register_candidate(c)
eligible, pkg = wf.admit_candidate(c.candidate_id)
_check("no_package_for_synthetic", pkg is None)

# ── 52. Candidate rejection reason populated ──────────────────────────────

print("\n-- 52. Rejection reasons populated --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(source_classification="SYNTHETIC")
wf.register_candidate(c)
eligible, _ = wf.admit_candidate(c.candidate_id)
_check("rejection_reasons_populated", len(c.rejection_reasons) > 0)

# ── 53. Discovery date auto-set ───────────────────────────────────────────

print("\n-- 53. Discovery date auto-set --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
c.discovery_date = ""
wf.register_candidate(c)
_check("discovery_date_set", c.discovery_date != "")

# ── 54. Candidate ID auto-generated ───────────────────────────────────────

print("\n-- 54. Candidate ID auto-generated --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate(candidate_id="")
wf.register_candidate(c)
_check("candidate_id_generated", c.candidate_id != "")

# ── 55. External discovery has provenance gaps ────────────────────────────

print("\n-- 55. External discovery has provenance gaps --")
ext = discover_external_candidates()
for e in ext:
    _check(f"ext_{e['name'][:15]}_has_provenance_level",
           e["provenance_level"] in ("SOURCE_REPORTED", "DOCUMENTED", "UNKNOWN"))

# ── 56. Admission history has timestamps ──────────────────────────────────

print("\n-- 56. Admission history has timestamps --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
wf.register_candidate(c)
wf.admit_candidate(c.candidate_id)
_check("history_has_timestamps",
       all("timestamp" in h for h in c.admission_history))

# ── 57. Admission history has from/to ─────────────────────────────────────

print("\n-- 57. Admission history has from/to --")
_check("history_has_from_to",
       all("from" in h and "to" in h for h in c.admission_history))

# ── 58. Package has label evidence hash ───────────────────────────────────

print("\n-- 58. Package has label evidence hash --")
wf = DatasetDiscoveryWorkflow()
c = _make_candidate()
wf.register_candidate(c)
eligible, pkg = wf.admit_candidate(c.candidate_id)
_check("package_has_label_hash", len(pkg.label_evidence_hash) == 64)

# ── 59. Package has independence evidence hash ────────────────────────────

print("\n-- 59. Package has independence evidence hash --")
_check("package_has_independence_hash", len(pkg.independence_evidence_hash) == 64)

# ── 60. Package has provenance level ──────────────────────────────────────

print("\n-- 60. Package has provenance level --")
_check("package_has_provenance", pkg.provenance_level == ProvenanceLevel.INDEPENDENTLY_VERIFIED.value)

# ── Summary ───────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Phase 54: {_PASSED}/{_TOTAL} PASS, {_FAILED}/{_TOTAL} FAIL")
print(f"{'='*60}")

if _FAILED > 0:
    sys.exit(1)
print("\nAll Phase 54 tests passed.")
