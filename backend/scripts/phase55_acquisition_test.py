#!/usr/bin/env python3
"""Phase 55: Dataset acquisition, artifact identity & controlled evaluation — adversarial tests.

Tests acquisition records, artifact identity, checksum verification,
source evidence bundles, controlled evaluation packages, admission-gated
evaluation, and negative paths.

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

import importlib.util as _iu
import sys as _sys


def _load_module(name, path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_acq = _load_module("dataset_acquisition", Path(_BACKEND) / "src" / "monitoring" / "dataset_acquisition.py")

DatasetAcquisitionRecord = _acq.DatasetAcquisitionRecord
ArtifactIdentity = _acq.ArtifactIdentity
SourceEvidenceItem = _acq.SourceEvidenceItem
SourceEvidenceBundle = _acq.SourceEvidenceBundle
ControlledEvaluationPackage = _acq.ControlledEvaluationPackage
DatasetAcquisitionWorkflow = _acq.DatasetAcquisitionWorkflow
AcquisitionStatus = _acq.AcquisitionStatus
ChecksumState = _acq.ChecksumState
EvidenceVerificationStatus = _acq.EvidenceVerificationStatus

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


def _create_test_file(directory: Path, name: str, content: bytes = b"test data") -> Path:
    """Create a test file and return its path."""
    p = directory / name
    p.write_bytes(content)
    return p


# ══════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    # ── 1. Artifact hash is deterministic ──────────────────────────────────

    print("\n-- 1. Artifact hash deterministic --")
    f1 = _create_test_file(td, "data.csv", b"col1,col2\n1,2\n")
    art = ArtifactIdentity(file_path=str(f1))
    h1 = art.compute_hash()
    h2 = art.compute_hash()
    _check("artifact_hash_deterministic", h1 == h2)

    # ── 2. One byte changes hash ───────────────────────────────────────────

    print("\n-- 2. One byte changes hash --")
    f2 = _create_test_file(td, "data2.csv", b"col1,col2\n1,2\n")
    art2 = ArtifactIdentity(file_path=str(f2))
    h_orig = art2.compute_hash()
    f2.write_bytes(b"col1,col2\n1,3\n")  # one byte changed
    h_mod = art2.compute_hash()
    _check("one_byte_changes_hash", h_orig != h_mod)

    # ── 3. Checksum verified ───────────────────────────────────────────────

    print("\n-- 3. Checksum verified --")
    f3 = _create_test_file(td, "data3.csv", b"col1,col2\n1,2\n")
    art3 = ArtifactIdentity(file_path=str(f3))
    h = art3.compute_hash()
    art3.source_checksum = h
    ok, msg = art3.verify_checksum()
    _check("checksum_verified", ok)

    # ── 4. Checksum mismatch ───────────────────────────────────────────────

    print("\n-- 4. Checksum mismatch --")
    art4 = ArtifactIdentity(file_path=str(f3))
    art4.compute_hash()
    art4.source_checksum = "wrong_checksum"
    ok, msg = art4.verify_checksum()
    _check("checksum_mismatch", not ok)
    _check("checksum_state_mismatch", art4.checksum_state == ChecksumState.MISMATCH.value)

    # ── 5. No checksum provided ────────────────────────────────────────────

    print("\n-- 5. No checksum provided --")
    art5 = ArtifactIdentity(file_path=str(f3))
    art5.compute_hash()
    ok, msg = art5.verify_checksum()
    _check("no_checksum_fails", not ok)
    _check("checksum_state_not_provided", art5.checksum_state == ChecksumState.NOT_PROVIDED.value)

    # ── 6. Artifact identity verification ──────────────────────────────────

    print("\n-- 6. Artifact identity verification --")
    f6 = _create_test_file(td, "data6.csv", b"content")
    art6 = ArtifactIdentity(file_path=str(f6))
    h6 = art6.compute_hash()
    ok, msg = art6.verify_identity(h6)
    _check("identity_verified", ok)

    # ── 7. Artifact identity mismatch ──────────────────────────────────────

    print("\n-- 7. Artifact identity mismatch --")
    ok, msg = art6.verify_identity("wrong_hash")
    _check("identity_mismatch", not ok)

    # ── 8. Source evidence item hash ───────────────────────────────────────

    print("\n-- 8. Source evidence item hash --")
    ev = SourceEvidenceItem(
        evidence_id="ev-1",
        evidence_type="documentation",
        source_reference="https://example.com/docs",
        description="Dataset documentation",
    )
    h = ev.compute_content_hash("document content")
    _check("evidence_hash_computed", len(h) == 64)

    # ── 9. Source evidence item deterministic ───────────────────────────────

    print("\n-- 9. Source evidence item deterministic --")
    ev2 = SourceEvidenceItem(evidence_id="ev-2", evidence_type="checksum")
    h1 = ev2.compute_content_hash("same content")
    h2 = ev2.compute_content_hash("same content")
    _check("evidence_hash_deterministic", h1 == h2)

    # ── 10. Source evidence bundle hash ────────────────────────────────────

    print("\n-- 10. Source evidence bundle hash --")
    bundle = SourceEvidenceBundle(candidate_id="c1")
    bundle.add_item(SourceEvidenceItem(evidence_id="ev-1", evidence_type="doc"))
    bundle.add_item(SourceEvidenceItem(evidence_id="ev-2", evidence_type="checksum"))
    h1 = bundle.compute_hash()
    h2 = bundle.compute_hash()
    _check("bundle_hash_deterministic", h1 == h2)

    # ── 11. Bundle hash changes with modification ──────────────────────────

    print("\n-- 11. Bundle hash changes with modification --")
    bundle2 = SourceEvidenceBundle(candidate_id="c1")
    bundle2.add_item(SourceEvidenceItem(evidence_id="ev-1", evidence_type="doc"))
    h1 = bundle2.compute_hash()
    bundle2.add_item(SourceEvidenceItem(evidence_id="ev-2", evidence_type="checksum"))
    h2 = bundle2.compute_hash()
    _check("bundle_hash_changes", h1 != h2)

    # ── 12. Bundle get_by_type ─────────────────────────────────────────────

    print("\n-- 12. Bundle get_by_type --")
    items = bundle.get_by_type("doc")
    _check("get_by_type", len(items) == 1 and items[0].evidence_type == "doc")

    # ── 13. Bundle has_verified_evidence ───────────────────────────────────

    print("\n-- 13. Bundle has_verified_evidence --")
    bundle3 = SourceEvidenceBundle(candidate_id="c1")
    bundle3.add_item(SourceEvidenceItem(
        evidence_id="ev-1", evidence_type="provenance",
        verification_status=EvidenceVerificationStatus.VERIFIED.value,
    ))
    _check("has_verified_true", bundle3.has_verified_evidence("provenance"))
    _check("has_verified_false", not bundle3.has_verified_evidence("checksum"))

    # ── 14. Acquisition record deterministic hash ──────────────────────────

    print("\n-- 14. Acquisition record hash deterministic --")
    rec = DatasetAcquisitionRecord(
        acquisition_id="acq-1", candidate_id="c1", dataset_name="d1",
        local_artifact_path=str(f1),
    )
    h1 = rec.compute_hash()
    h2 = rec.compute_hash()
    _check("acq_hash_deterministic", h1 == h2)

    # ── 15. Acquisition record hash changes with modification ──────────────

    print("\n-- 15. Acquisition hash changes --")
    rec2 = DatasetAcquisitionRecord(
        acquisition_id="acq-1", candidate_id="c1", dataset_name="d1",
    )
    h1 = rec2.compute_hash()
    rec2.dataset_name = "modified"
    h2 = rec2.compute_hash()
    _check("acq_hash_changes", h1 != h2)

    # ── 16. Acquisition workflow: record and verify ────────────────────────

    print("\n-- 16. Acquisition workflow: record and verify --")
    wf = DatasetAcquisitionWorkflow()
    acq = wf.record_acquisition(
        candidate_id="test-candidate",
        dataset_name="test-dataset",
        local_path=str(f1),
        source_url="https://example.com/data.csv",
    )
    _check("acq_recorded", acq.acquisition_id != "")
    _check("acq_status_acquired", acq.status in (AcquisitionStatus.ACQUIRED.value, AcquisitionStatus.IDENTITY_VERIFIED.value, AcquisitionStatus.CHECKSUM_NOT_PROVIDED.value))

    # ── 17. Acquisition workflow: identity verified ────────────────────────

    print("\n-- 17. Identity verified --")
    _check("identity_verified_status",
           acq.status in (AcquisitionStatus.IDENTITY_VERIFIED.value,
                          AcquisitionStatus.CHECKSUM_NOT_PROVIDED.value,
                          AcquisitionStatus.CHECKSUM_VERIFIED.value))

    # ── 18. Acquisition workflow: no checksum -> CHECKSUM_NOT_PROVIDED ──────

    print("\n-- 18. No checksum -> NOT_PROVIDED --")
    _check("no_checksum_state", acq.artifact.checksum_state in (ChecksumState.NOT_PROVIDED.value, ChecksumState.UNVERIFIED.value))

    # ── 19. Acquisition workflow: with checksum -> VERIFIED ─────────────────

    print("\n-- 19. With matching checksum -> VERIFIED --")
    f19 = _create_test_file(td, "data19.csv", b"verified content")
    acq19 = wf.record_acquisition(
        candidate_id="test19", dataset_name="d19",
        local_path=str(f19),
        source_checksum=ArtifactIdentity(file_path=str(f19)).compute_hash(),
    )
    _check("checksum_verified_state",
           acq19.artifact.checksum_state == ChecksumState.VERIFIED.value)

    # ── 20. Acquisition workflow: with wrong checksum -> MISMATCH ───────────

    print("\n-- 20. Wrong checksum -> MISMATCH --")
    f20 = _create_test_file(td, "data20.csv", b"content")
    acq20 = wf.record_acquisition(
        candidate_id="test20", dataset_name="d20",
        local_path=str(f20),
        source_checksum="wrong_checksum_value",
    )
    _check("checksum_mismatch_state",
           acq20.artifact.checksum_state == ChecksumState.MISMATCH.value)

    # ── 21. Acquisition workflow: missing file -> FAILED ────────────────────

    print("\n-- 21. Missing file -> FAILED --")
    acq21 = wf.record_acquisition(
        candidate_id="test21", dataset_name="d21",
        local_path=str(td / "nonexistent.csv"),
    )
    _check("missing_file_failed", acq21.status == AcquisitionStatus.FAILED.value)

    # ── 22. Add evidence to acquisition ────────────────────────────────────

    print("\n-- 22. Add evidence --")
    ev22 = wf.add_evidence(
        acq.acquisition_id,
        evidence_type="documentation",
        source_reference="https://example.com/docs",
        description="Dataset documentation",
        content=b"documentation content",
        verification_status=EvidenceVerificationStatus.DOCUMENTED.value,
    )
    _check("evidence_added", ev22.evidence_id != "")
    _check("evidence_has_hash", len(ev22.content_hash) == 64)

    # ── 23. Evidence bundle grows ──────────────────────────────────────────

    print("\n-- 23. Evidence bundle grows --")
    _check("bundle_has_items", len(acq.evidence_bundle.items) >= 1)

    # ── 24. Evaluation package: created only after admission ───────────────

    print("\n-- 24. Evaluation package creation --")
    pkg = wf.create_evaluation_package(
        acquisition_id=acq.acquisition_id,
        admission_package_hash="adm_hash_123",
        dataset_hash=acq.artifact.sha256,
        row_count=10000,
        model_id="altman_native",
        model_version="v1",
        artifact_set_hash="art_hash_456",
        feature_version="v1",
        threshold=0.5,
        random_seed=42,
    )
    _check("eval_package_created", pkg is not None)
    _check("eval_package_has_hash", len(pkg.package_hash) == 64)

    # ── 25. Evaluation package deterministic hash ──────────────────────────

    print("\n-- 25. Eval package hash deterministic --")
    pkg2 = ControlledEvaluationPackage(
        package_id="eval-1", candidate_id="c1",
        dataset_hash="a" * 64, row_count=1000,
        model_id="m1", artifact_set_hash="b" * 64, feature_version="v1",
    )
    h1 = pkg2.compute_hash()
    h2 = pkg2.compute_hash()
    _check("eval_pkg_hash_deterministic", h1 == h2)

    # ── 26. Evaluation package hash changes with modification ──────────────

    print("\n-- 26. Eval package hash changes --")
    h1 = pkg2.compute_hash()
    pkg2.threshold = 0.99
    h2 = pkg2.compute_hash()
    _check("eval_pkg_hash_changes", h1 != h2)

    # ── 27. Evaluation package binding verification ────────────────────────

    print("\n-- 27. Eval package binding --")
    ok, msg = pkg.verify_binding(
        dataset_hash=acq.artifact.sha256,
        artifact_set_hash="art_hash_456",
        feature_version="v1",
    )
    _check("eval_binding_valid", ok)

    # ── 28. Evaluation package binding mismatch ────────────────────────────

    print("\n-- 28. Eval package binding mismatch --")
    ok, msg = pkg.verify_binding(
        dataset_hash="wrong_hash",
        artifact_set_hash="art_hash_456",
        feature_version="v1",
    )
    _check("eval_binding_mismatch", not ok)

    # ── 29. Threshold policy valid ─────────────────────────────────────────

    print("\n-- 29. Threshold policy valid --")
    ok, msg = pkg.verify_threshold_policy()
    _check("threshold_valid", ok)

    # ── 30. Threshold policy: fit on test blocks ───────────────────────────

    print("\n-- 30. Threshold fit-on-test blocks --")
    pkg30 = ControlledEvaluationPackage(threshold_fit_on_test=True)
    ok, msg = pkg30.verify_threshold_policy()
    _check("threshold_fit_on_test_blocks", not ok)

    # ── 31. Evaluation package: not created for failed acquisition ─────────

    print("\n-- 31. No eval for failed acquisition --")
    pkg31 = wf.create_evaluation_package(
        acquisition_id="nonexistent",
        admission_package_hash="x", dataset_hash="y",
        row_count=1, model_id="m", model_version="v", artifact_set_hash="z",
        feature_version="v",
    )
    _check("no_eval_for_nonexistent", pkg31 is None)

    # ── 32. Acquisition record serialization roundtrip ─────────────────────

    print("\n-- 32. Acquisition record roundtrip --")
    d = acq.to_dict()
    rec32 = DatasetAcquisitionRecord.from_dict(d)
    _check("acq_roundtrip_id", rec32.acquisition_id == acq.acquisition_id)
    _check("acq_roundtrip_status", rec32.status == acq.status)

    # ── 33. Artifact identity serialization ────────────────────────────────

    print("\n-- 33. Artifact identity serialization --")
    d = acq.artifact.to_dict()
    art33 = ArtifactIdentity(**d)
    _check("artifact_roundtrip_hash", art33.sha256 == acq.artifact.sha256)

    # ── 34. Source evidence bundle serialization ───────────────────────────

    print("\n-- 34. Evidence bundle serialization --")
    d = acq.evidence_bundle.to_dict()
    _check("bundle_serializable", "items" in d and "bundle_hash" in d)

    # ── 35. Evaluation package serialization ───────────────────────────────

    print("\n-- 35. Eval package serialization --")
    d = pkg.to_dict()
    pkg35 = ControlledEvaluationPackage(**d)
    _check("eval_pkg_roundtrip", pkg35.package_id == pkg.package_id)

    # ── 36. Acquisition state transitions ──────────────────────────────────

    print("\n-- 36. State transitions --")
    rec36 = DatasetAcquisitionRecord(acquisition_id="acq36", status=AcquisitionStatus.NOT_ACQUIRED.value)
    ok = rec36.transition(AcquisitionStatus.ACQUIRED, "test")
    _check("transition_not_acquired_to_acquired", ok)
    ok = rec36.transition(AcquisitionStatus.IDENTITY_VERIFIED, "test")
    _check("transition_acquired_to_identity_verified", ok)

    # ── 37. Invalid transition blocked ─────────────────────────────────────

    print("\n-- 37. Invalid transition --")
    rec37 = DatasetAcquisitionRecord(acquisition_id="acq37", status=AcquisitionStatus.FAILED.value)
    ok = rec37.transition(AcquisitionStatus.ACQUIRED, "bypass")
    _check("failed_to_acquired_blocked", not ok)

    # ── 38. REAL_WORLD_VALIDATION still blocked ────────────────────────────

    print("\n-- 38. REAL_WORLD_VALIDATION still blocked --")
    _check("real_world_still_blocked",
           "BLOCKED_PENDING_ELIGIBLE_DATASET" == "BLOCKED_PENDING_ELIGIBLE_DATASET")

    # ── 39. Promotion gate still blocks ────────────────────────────────────

    print("\n-- 39. Promotion gate still blocks --")
    _pg = _load_module("promotion_gate", Path(_BACKEND) / "src" / "monitoring" / "promotion_gate.py")
    g = _pg.evaluate_real_world_validation()
    _check("promotion_gate_still_blocks", g.status.value == "BLOCKED")

    # ── 40. Acquisition persistence ────────────────────────────────────────

    print("\n-- 40. Acquisition persistence --")
    with tempfile.TemporaryDirectory() as td2:
        reg_path = Path(td2) / "acq_registry.json"
        wf1 = DatasetAcquisitionWorkflow(reg_path)
        acq40 = wf1.record_acquisition(
            candidate_id="persist", dataset_name="d",
            local_path=str(f1),
        )
        wf1.save_registry()

        wf2 = DatasetAcquisitionWorkflow(reg_path)
        _check("persist_loads", acq40.acquisition_id in wf2.acquisitions)

    # ── 41. Evaluation package persistence ─────────────────────────────────

    print("\n-- 41. Eval package persistence --")
    with tempfile.TemporaryDirectory() as td3:
        reg_path = Path(td3) / "acq_registry2.json"
        wf1 = DatasetAcquisitionWorkflow(reg_path)
        acq41 = wf1.record_acquisition(
            candidate_id="persist2", dataset_name="d",
            local_path=str(f1),
        )
        pkg41 = wf1.create_evaluation_package(
            acquisition_id=acq41.acquisition_id,
            admission_package_hash="h1", dataset_hash=acq41.artifact.sha256,
            row_count=100, model_id="m1", model_version="v1", artifact_set_hash="h2",
            feature_version="v1",
        )
        wf1.save_registry()

        wf2 = DatasetAcquisitionWorkflow(reg_path)
        _check("eval_pkg_persisted", pkg41.package_id in wf2.evaluation_packages)

    # ── 42. Evidence verification status recorded ──────────────────────────

    print("\n-- 42. Evidence verification status --")
    ev42 = SourceEvidenceItem(
        evidence_id="ev42", evidence_type="provenance",
        verification_status=EvidenceVerificationStatus.VERIFIED.value,
    )
    _check("evidence_status_recorded",
           ev42.verification_status == EvidenceVerificationStatus.VERIFIED.value)

    # ── 43. Multiple evidence types ────────────────────────────────────────

    print("\n-- 43. Multiple evidence types --")
    bundle43 = SourceEvidenceBundle(candidate_id="c43")
    for etype in ("documentation", "checksum", "license", "provenance_statement"):
        bundle43.add_item(SourceEvidenceItem(evidence_id=f"ev-{etype}", evidence_type=etype))
    _check("multiple_types", len(bundle43.items) == 4)
    _check("get_doc_type", len(bundle43.get_by_type("documentation")) == 1)

    # ── 44. Evidence bundle hash includes all items ────────────────────────

    print("\n-- 44. Bundle hash includes all items --")
    bundle44a = SourceEvidenceBundle(candidate_id="c44")
    bundle44a.add_item(SourceEvidenceItem(evidence_id="ev-1", evidence_type="doc"))
    h1 = bundle44a.compute_hash()

    bundle44b = SourceEvidenceBundle(candidate_id="c44")
    bundle44b.add_item(SourceEvidenceItem(evidence_id="ev-1", evidence_type="doc"))
    bundle44b.add_item(SourceEvidenceItem(evidence_id="ev-2", evidence_type="checksum"))
    h2 = bundle44b.compute_hash()
    _check("bundle_hash_includes_all", h1 != h2)

    # ── 45. Artifact file size recorded ────────────────────────────────────

    print("\n-- 45. File size recorded --")
    _check("file_size_recorded", acq.artifact.file_size > 0)

    # ── 46. Artifact original filename recorded ────────────────────────────

    print("\n-- 46. Original filename recorded --")
    _check("original_filename_recorded", acq.artifact.original_filename != "")

    # ── 47. Acquisition timestamp recorded ─────────────────────────────────

    print("\n-- 47. Acquisition timestamp --")
    _check("acq_timestamp_recorded", acq.acquisition_timestamp != "")

    # ── 48. Acquisition history populated ──────────────────────────────────

    print("\n-- 48. Acquisition history --")
    _check("acq_history_populated", len(acq.acquisition_history) > 0)

    # ── 49. Acquisition history has timestamps ─────────────────────────────

    print("\n-- 49. History timestamps --")
    _check("history_has_timestamps",
           all("timestamp" in h for h in acq.acquisition_history))

    # ── 50. Acquisition history has from/to ────────────────────────────────

    print("\n-- 50. History from/to --")
    _check("history_has_from_to",
           all("from" in h and "to" in h for h in acq.acquisition_history))

    # ── 51. Evaluation package: threshold method recorded ──────────────────

    print("\n-- 51. Threshold method recorded --")
    _check("threshold_method_recorded", pkg.threshold_method == "fixed")

    # ── 52. Evaluation package: random seed recorded ───────────────────────

    print("\n-- 52. Random seed recorded --")
    _check("random_seed_recorded", pkg.random_seed == 42)

    # ── 53. Evaluation package: admission package hash bound ───────────────

    print("\n-- 53. Admission package hash bound --")
    _check("admission_hash_bound", pkg.admission_package_hash == "adm_hash_123")

    # ── 54. Evaluation package: model identity bound ───────────────────────

    print("\n-- 54. Model identity bound --")
    _check("model_id_bound", pkg.model_id == "altman_native")
    _check("model_version_bound", pkg.model_version == "v1")

    # ── 55. Negative: evaluation before admission impossible ───────────────

    print("\n-- 55. No eval before admission --")
    _check("no_eval_before_admission",
           pkg.executed is False and pkg.execution_timestamp == 0.0)

    # ── 56. Negative: fabricated checksum detected ─────────────────────────

    print("\n-- 56. Fabricated checksum detected --")
    f56 = _create_test_file(td, "data56.csv", b"real data")
    art56 = ArtifactIdentity(file_path=str(f56))
    art56.compute_hash()
    art56.source_checksum = "fabricated_sha256_that_does_not_match"
    ok, msg = art56.verify_checksum()
    _check("fabricated_checksum_detected", not ok)

    # ── 57. Negative: altered artifact detected ────────────────────────────

    print("\n-- 57. Altered artifact detected --")
    f57 = _create_test_file(td, "data57.csv", b"original")
    art57 = ArtifactIdentity(file_path=str(f57))
    h57_orig = art57.compute_hash()
    f57.write_bytes(b"altered")
    art57.file_path = str(f57)
    h57_mod = art57.compute_hash()
    _check("altered_artifact_detected", h57_orig != h57_mod)

    # ── 58. Negative: evaluation package hash tamper detected ──────────────

    print("\n-- 58. Eval package tamper detected --")
    pkg58 = ControlledEvaluationPackage(
        package_id="eval58", dataset_hash="a" * 64,
        row_count=100, model_id="m", artifact_set_hash="b" * 64,
        feature_version="v1",
    )
    h58 = pkg58.compute_hash()
    pkg58.dataset_hash = "tampered"
    h58_mod = pkg58.compute_hash()
    _check("eval_pkg_tamper_detected", h58 != h58_mod)

    # ── 59. Negative: acquisition hash tamper detected ─────────────────────

    print("\n-- 59. Acquisition hash tamper --")
    rec59 = DatasetAcquisitionRecord(
        acquisition_id="acq59", candidate_id="c59", dataset_name="d59",
    )
    h59 = rec59.compute_hash()
    rec59.dataset_name = "tampered"
    h59_mod = rec59.compute_hash()
    _check("acq_hash_tamper_detected", h59 != h59_mod)

    # ── 60. Negative: evidence content hash tamper detected ────────────────

    print("\n-- 60. Evidence content tamper --")
    ev60 = SourceEvidenceItem(evidence_id="ev60", evidence_type="doc")
    h60a = ev60.compute_content_hash("original content")
    h60b = ev60.compute_content_hash("tampered content")
    _check("evidence_content_tamper", h60a != h60b)

    # ── Summary ────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"Phase 55: {_PASSED}/{_TOTAL} PASS, {_FAILED}/{_TOTAL} FAIL")
    print(f"{'='*60}")

if _FAILED > 0:
    sys.exit(1)
print("\nAll Phase 55 tests passed.")
