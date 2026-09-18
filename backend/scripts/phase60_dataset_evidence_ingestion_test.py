"""Phase 60: Dataset Evidence Ingestion - 120+ adversarial tests."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_BACKEND = str(Path(__file__).resolve().parent.parent)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import importlib.util as _iu

def _load_module(name, path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_mod = _load_module("dei", Path(_BACKEND) / "src" / "monitoring" / "dataset_evidence_ingestion.py")

DatasetEvidenceIngestion = _mod.DatasetEvidenceIngestion
EvidencePackage = _mod.EvidencePackage
EvidenceItem = _mod.EvidenceItem
EvidenceManifest = _mod.EvidenceManifest
EvidenceDiff = _mod.EvidenceDiff
EvidenceCategoryGate = _mod.EvidenceCategoryGate
EvidencePackageState = _mod.EvidencePackageState
EvidenceStatus = _mod.EvidenceStatus
SourceCaptureLevel = _mod.SourceCaptureLevel
create_fixture_evidence_package = _mod.create_fixture_evidence_package

passed = 0
failed = 0
total = 0

def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name} -- {detail}")

print("\n" + "="*60)
print("PHASE 60: DATASET EVIDENCE INGESTION TESTS")
print("="*60)


# ── 1. Package creation ──────────────────────────────────────────────────

print("\n-- 1. Package creation --")

ing = DatasetEvidenceIngestion()
pkg = ing.create_package(
    package_id="pkg-001",
    candidate_id="cand-001",
    dataset_id="ds-001",
    acquisition_id="acq-001",
    source_name="test-source",
    publisher="test-publisher",
)
check("Package created", pkg is not None)
check("Package ID correct", pkg.package_id == "pkg-001")
check("Initial state", pkg.state == EvidencePackageState.EVIDENCE_PACKAGE_CREATED.value)
check("Package stored", "pkg-001" in ing.packages)


# ── 2. Source capture ────────────────────────────────────────────────────

print("\n-- 2. Source capture --")

ok = ing.capture_source("pkg-001", distribution_reference="dist://ref",
                        license_reference="lic://ref", release_version="1.0")
check("Source capture succeeds", ok)
pkg = ing.packages["pkg-001"]
check("State is SOURCE_CAPTURED", pkg.state == EvidencePackageState.SOURCE_CAPTURED.value)
check("Distribution reference captured", pkg.distribution_reference == "dist://ref")
check("License reference captured", pkg.license_reference == "lic://ref")
check("Release version captured", pkg.release_version == "1.0")


# ── 3. Artifact capture ──────────────────────────────────────────────────

print("\n-- 3. Artifact capture --")

ok = ing.capture_artifact("pkg-001", dataset_filename="data.csv",
                          file_size=2048, dataset_sha256="abc123hash")
check("Artifact capture succeeds", ok)
check("State is ARTIFACT_CAPTURED", pkg.state == EvidencePackageState.ARTIFACT_CAPTURED.value)
check("Filename captured", pkg.dataset_filename == "data.csv")
check("File size captured", pkg.file_size == 2048)
check("SHA-256 captured", pkg.dataset_sha256 == "abc123hash")


# ── 4. Add evidence items ────────────────────────────────────────────────

print("\n-- 4. Add evidence items --")

item = ing.add_evidence("pkg-001", evidence_type="PROVENANCE",
                        source_reference="https://source.org",
                        captured_content="Provenance documentation content",
                        verification_status="DOCUMENTED",
                        source_capture_level=SourceCaptureLevel.DOCUMENTATION_CAPTURED.value)
check("Evidence item created", item is not None)
check("Evidence ID set", item.evidence_id.startswith("ev-"))
check("Content hash computed", len(item.content_hash) == 64)
check("Verification status set", item.verification_status == "DOCUMENTED")
check("Source capture level set", item.source_capture_level == "DOCUMENTATION_CAPTURED")


# ── 5. Add multiple evidence types ───────────────────────────────────────

print("\n-- 5. Multiple evidence types --")

for cat in ["LABEL_SEMANTICS", "TEMPORAL", "INDEPENDENCE", "FEATURE_COMPATIBILITY",
            "LEAKAGE", "LICENSE", "SOURCE_IDENTITY", "ARTIFACT"]:
    ing.add_evidence("pkg-001", evidence_type=cat,
                     captured_content=f"{cat} evidence content",
                     verification_status="DOCUMENTED")

pkg = ing.packages["pkg-001"]
check("9 evidence items total", len(pkg.evidence_items) == 9)


# ── 6. Documentation capture ─────────────────────────────────────────────

print("\n-- 6. Documentation capture --")

ok = ing.capture_documentation("pkg-001")
check("Documentation capture succeeds", ok)
check("State is DOCUMENTATION_CAPTURED",
      pkg.state == EvidencePackageState.DOCUMENTATION_CAPTURED.value)


# ── 7. Compute hashes ────────────────────────────────────────────────────

print("\n-- 7. Compute hashes --")

ok = ing.compute_hashes("pkg-001")
check("Hash computation succeeds", ok)
check("State is HASHES_COMPUTED", pkg.state == EvidencePackageState.HASHES_COMPUTED.value)
# Verify all items have content hashes
all_hashed = all(e.content_hash for e in pkg.evidence_items if e.captured_content)
check("All items with content have hashes", all_hashed)


# ── 8. Index evidence ────────────────────────────────────────────────────

print("\n-- 8. Index evidence --")

ok = ing.index_evidence("pkg-001")
check("Index succeeds", ok)
check("State is EVIDENCE_INDEXED", pkg.state == EvidencePackageState.EVIDENCE_INDEXED.value)


# ── 9. Validate evidence ─────────────────────────────────────────────────

print("\n-- 9. Validate evidence --")

all_pass, gates = ing.validate_evidence("pkg-001")
check("Validation succeeds", all_pass)
check("State is EVIDENCE_VALIDATED", pkg.state == EvidencePackageState.EVIDENCE_VALIDATED.value)
check("Gates returned", len(gates) > 0)
# All blocking categories should pass
blocking_gates = [g for g in gates if g.blocking]
check("All blocking gates pass", all(g.status == EvidenceStatus.PASS.value for g in blocking_gates))


# ── 10. Build manifest ───────────────────────────────────────────────────

print("\n-- 10. Build manifest --")

manifest = ing.build_manifest("pkg-001")
check("Manifest created", manifest is not None)
check("Manifest hash is 64 chars", len(manifest.manifest_hash) == 64)
check("Manifest has evidence IDs", len(manifest.evidence_ids) > 0)
check("Manifest has evidence hashes", len(manifest.evidence_hashes) > 0)
check("Manifest has gate results", len(manifest.gate_results) > 0)


# ── 11. Ready for certification ──────────────────────────────────────────

print("\n-- 11. Ready for certification --")

ok = ing.ready_for_certification("pkg-001")
check("Ready succeeds", ok)
check("State is READY_FOR_CERTIFICATION",
      pkg.state == EvidencePackageState.READY_FOR_CERTIFICATION.value)
check("Package hash computed", len(pkg.package_hash) == 64)


# ── 12. Export / Import round trip ───────────────────────────────────────

print("\n-- 12. Export / Import round trip --")

exported = ing.export_package("pkg-001")
check("Export returns data", exported is not None)
check("Export has package_id", exported["package_id"] == "pkg-001")
check("Export has manifest", "manifest" in exported)
check("Export has evidence_items", "evidence_items" in exported)

# Import into fresh ingestion
ing2 = DatasetEvidenceIngestion()
imported = ing2.import_package(exported)
check("Import succeeds", imported is not None)
check("Imported package ID matches", imported.package_id == "pkg-001")
check("Imported manifest hash matches",
      imported.manifest.manifest_hash == manifest.manifest_hash)


# ── 13. Validate imported package ────────────────────────────────────────

print("\n-- 13. Validate imported package --")

ok, msg = ing2.validate_imported_package("pkg-001")
check("Imported package integrity verified", ok)


# ── 14. Manifest hash determinism ────────────────────────────────────────

print("\n-- 14. Manifest hash determinism --")

m1 = EvidenceManifest(package_id="p", dataset_id="d", artifact_hash="a")
m1.evidence_ids = ["ev-1", "ev-2"]
m1.evidence_hashes = {"ev-1": "h1", "ev-2": "h2"}
m1.compute_hash()

m2 = EvidenceManifest(package_id="p", dataset_id="d", artifact_hash="a")
m2.evidence_ids = ["ev-1", "ev-2"]
m2.evidence_hashes = {"ev-1": "h1", "ev-2": "h2"}
m2.compute_hash()

check("Identical manifests -> same hash", m1.manifest_hash == m2.manifest_hash)


# ── 15. Manifest hash tamper detection ───────────────────────────────────

print("\n-- 15. Manifest hash tamper detection --")

m = EvidenceManifest(package_id="p", dataset_id="d")
m.evidence_ids = ["ev-1"]
m.compute_hash()
original = m.manifest_hash

m.evidence_ids = ["ev-1", "ev-TAMPERED"]
m.compute_hash()
check("Manifest tamper detected", m.manifest_hash != original)


# ── 16. Package hash tamper detection ────────────────────────────────────

print("\n-- 16. Package hash tamper detection --")

pkg = ing.packages["pkg-001"]
original_pkg_hash = pkg.package_hash

pkg.dataset_sha256 = "TAMPERED"
ok, msg = pkg.verify_integrity()
check("Package tamper detected", not ok)
# Restore
pkg.dataset_sha256 = "abc123hash"


# ── 17. Evidence diff: no changes ────────────────────────────────────────

print("\n-- 17. Evidence diff: no changes --")

# Create identical packages
ing3 = DatasetEvidenceIngestion()
p1 = ing3.create_package("p1", "c1", "d1", "a1")
ing3.capture_source("p1")
ing3.capture_artifact("p1", dataset_filename="f", file_size=100, dataset_sha256="h1")
ing3.add_evidence("p1", "PROVENANCE", captured_content="content1", verification_status="DOCUMENTED")
ing3.capture_documentation("p1")
ing3.compute_hashes("p1")
ing3.index_evidence("p1")
ing3.validate_evidence("p1")
ing3.build_manifest("p1")
ing3.ready_for_certification("p1")

ing4 = DatasetEvidenceIngestion()
p2 = ing4.create_package("p1", "c1", "d1", "a1")
ing4.capture_source("p1")
ing4.capture_artifact("p1", dataset_filename="f", file_size=100, dataset_sha256="h1")
ing4.add_evidence("p1", "PROVENANCE", captured_content="content1", verification_status="DOCUMENTED")
ing4.capture_documentation("p1")
ing4.compute_hashes("p1")
ing4.index_evidence("p1")
ing4.validate_evidence("p1")
ing4.build_manifest("p1")
ing4.ready_for_certification("p1")

diff = ing3.diff_packages("p1", "p1", other=ing4)
check("No changes detected", not diff.material_changes)
check("Diff has 0 changes", len(diff.changes) == 0)


# ── 18. Evidence diff: detects changes ───────────────────────────────────

print("\n-- 18. Evidence diff: detects changes --")

ing5 = DatasetEvidenceIngestion()
p3 = ing5.create_package("p3", "c2", "d2", "a2")
ing5.capture_source("p3")
ing5.capture_artifact("p3", dataset_filename="f", file_size=200, dataset_sha256="h2")
ing5.add_evidence("p3", "PROVENANCE", captured_content="content2", verification_status="DOCUMENTED")
ing5.capture_documentation("p3")
ing5.compute_hashes("p3")
ing5.index_evidence("p3")
ing5.validate_evidence("p3")
ing5.build_manifest("p3")
ing5.ready_for_certification("p3")

diff = ing3.diff_packages("p1", "p3", other=ing5)
check("Changes detected", diff.material_changes)
check("File size differs", any(c["field"] == "file_size" for c in diff.changes))
check("SHA-256 differs", any(c["field"] == "dataset_sha256" for c in diff.changes))


# ── 19. Certification binding: valid ─────────────────────────────────────

print("\n-- 19. Certification binding --")

ok, msg = ing.verify_certification_binding("pkg-001", manifest.manifest_hash)
check("Valid binding passes", ok)


# ── 20. Certification binding: mismatch ──────────────────────────────────

print("\n-- 20. Certification binding mismatch --")

ok, msg = ing.verify_certification_binding("pkg-001", "WRONG_HASH")
check("Wrong manifest hash blocks", not ok)


# ── 21. Certification binding: missing package ───────────────────────────

print("\n-- 21. Certification binding: missing package --")

ok, msg = ing.verify_certification_binding("nonexistent", "hash")
check("Missing package blocks", not ok)


# ── 22. Forensic chain integrity ─────────────────────────────────────────

print("\n-- 22. Forensic chain integrity --")

ok, errors = ing.verify_forensic_chain()
check("Forensic chain is valid", ok, str(errors))


# ── 23. Forensic chain: tamper detection ──────────────────────────────────

print("\n-- 23. Forensic chain tamper detection --")

ing_t = DatasetEvidenceIngestion()
ing_t.create_package("tamper-pkg", "c", "d", "a")
ing_t.capture_source("tamper-pkg")
# Tamper with event
ing_t._forensic_events[0]["event_type"] = "TAMPERED"
ok, errors = ing_t.verify_forensic_chain()
check("Tampered forensic event detected", not ok)


# ── 24. State machine: invalid transitions ───────────────────────────────

print("\n-- 24. State machine invalid transitions --")

ing_s = DatasetEvidenceIngestion()
p = ing_s.create_package("s-pkg", "c", "d", "a")
# Try to jump from CREATED to EVIDENCE_VALIDATED
ok = p.transition(EvidencePackageState.EVIDENCE_VALIDATED)
check("CREATED -> EVIDENCE_VALIDATED blocked", not ok)

# Try to go from BLOCKED
p.state = EvidencePackageState.BLOCKED.value
ok = p.transition(EvidencePackageState.SOURCE_CAPTURED)
check("BLOCKED is terminal", not ok)


# ── 25. State machine: valid transitions ──────────────────────────────────

print("\n-- 25. State machine valid transitions --")

ing_v = DatasetEvidenceIngestion()
p = ing_v.create_package("v-pkg", "c", "d", "a")

ok = p.transition(EvidencePackageState.SOURCE_CAPTURED)
check("CREATED -> SOURCE_CAPTURED", ok)

ok = p.transition(EvidencePackageState.ARTIFACT_CAPTURED)
check("SOURCE_CAPTURED -> ARTIFACT_CAPTURED", ok)

ok = p.transition(EvidencePackageState.DOCUMENTATION_CAPTURED)
check("ARTIFACT_CAPTURED -> DOCUMENTATION_CAPTURED", ok)

ok = p.transition(EvidencePackageState.HASHES_COMPUTED)
check("DOCUMENTATION_CAPTURED -> HASHES_COMPUTED", ok)

ok = p.transition(EvidencePackageState.EVIDENCE_INDEXED)
check("HASHES_COMPUTED -> EVIDENCE_INDEXED", ok)

ok = p.transition(EvidencePackageState.EVIDENCE_VALIDATED)
check("EVIDENCE_INDEXED -> EVIDENCE_VALIDATED", ok)

ok = p.transition(EvidencePackageState.READY_FOR_CERTIFICATION)
check("EVIDENCE_VALIDATED -> READY_FOR_CERTIFICATION", ok)

ok = p.transition(EvidencePackageState.SOURCE_CAPTURED)
check("READY_FOR_CERTIFICATION is terminal", not ok)


# ── 26. Missing evidence stays MISSING ───────────────────────────────────

print("\n-- 26. Missing evidence stays MISSING --")

ing_m = DatasetEvidenceIngestion()
pm = ing_m.create_package("m-pkg", "c", "d", "a")
ing_m.capture_source("m-pkg")
ing_m.capture_artifact("m-pkg", "f", 100, "h")
# Don't add any evidence items
ing_m.capture_documentation("m-pkg")
ing_m.compute_hashes("m-pkg")
ing_m.index_evidence("m-pkg")

all_pass, gates = ing_m.validate_evidence("m-pkg")
check("Validation fails with missing evidence", not all_pass)
missing_gates = [g for g in gates if g.status == EvidenceStatus.MISSING.value]
check("Missing categories exist", len(missing_gates) > 0)
check("MISSING is not PASS", all(g.status != EvidenceStatus.PASS.value for g in missing_gates))


# ── 27. Evidence content hash is deterministic ───────────────────────────

print("\n-- 27. Content hash determinism --")

e1 = EvidenceItem(evidence_id="e1", captured_content="same content")
e1.compute_content_hash()
e2 = EvidenceItem(evidence_id="e2", captured_content="same content")
e2.compute_content_hash()
check("Same content -> same hash", e1.content_hash == e2.content_hash)

e3 = EvidenceItem(evidence_id="e3", captured_content="different content")
e3.compute_content_hash()
check("Different content -> different hash", e1.content_hash != e3.content_hash)


# ── 28. Package hash excludes volatile timestamps ────────────────────────

print("\n-- 28. Package hash excludes timestamps --")

p_a = EvidencePackage(package_id="a", dataset_id="d", created_at=1000)
p_a.compute_hash()
p_b = EvidencePackage(package_id="a", dataset_id="d", created_at=2000)
p_b.compute_hash()
check("Different timestamps -> same hash", p_a.package_hash == p_b.package_hash)


# ── 29. Export excludes secrets ──────────────────────────────────────────

print("\n-- 29. Export excludes secrets --")

exported = ing.export_package("pkg-001")
export_str = json.dumps(exported)
check("No JWT_SECRET in export", "jwt_secret" not in export_str.lower())
check("No API key in export", "api_key" not in export_str.lower())
check("No password in export", "password" not in export_str.lower())


# ── 30. Import: tampered package fails ───────────────────────────────────

print("\n-- 30. Import tampered package --")

# Export original, then import it cleanly
exported_orig = ing.export_package("pkg-001")
ing_clean = DatasetEvidenceIngestion()
imported_clean = ing_clean.import_package(dict(exported_orig))
check("Clean import succeeds", imported_clean is not None)
ok_clean, _ = ing_clean.validate_imported_package(imported_clean.package_id)
check("Clean import validates", ok_clean)

# Now tamper the IMPORTED package object and re-validate
imported_clean.dataset_sha256 = "TAMPERED_POST_IMPORT"
ok_tamper, msg = ing_clean.validate_imported_package(imported_clean.package_id)
check("Post-import tamper detected", not ok_tamper, msg)

# Also verify that a tampered export produces a different import hash
exported_tamper = dict(exported_orig)
exported_tamper["dataset_sha256"] = "TAMPERED_EXPORT"
ing_tamper = DatasetEvidenceIngestion()
imported_tamper = ing_tamper.import_package(exported_tamper)
check("Tampered export imports", imported_tamper is not None)
# The tampered import's _exported_hash differs from clean import's
check("Tampered export has different hash",
      imported_tamper._exported_hash != imported_clean._exported_hash)


# ── 31. TEST_FIXTURE isolation ───────────────────────────────────────────

print("\n-- 31. TEST_FIXTURE isolation --")

pkg_fix, ing_fix = create_fixture_evidence_package()
check("Fixture package created", pkg_fix is not None)
check("Fixture is marked", pkg_fix.is_test_fixture)
check("Fixture state is READY", pkg_fix.state == EvidencePackageState.READY_FOR_CERTIFICATION.value)
check("Fixture manifest has hash", len(pkg_fix.manifest.manifest_hash) == 64)
check("Fixture forensic chain valid", ing_fix.verify_forensic_chain()[0])


# ── 32. Full workflow success ────────────────────────────────────────────

print("\n-- 32. Full workflow --")

ing_fw = DatasetEvidenceIngestion()
fw_pkg = ing_fw.create_package("fw-pkg", "fw-cand", "fw-ds", "fw-acq",
                                source_name="fw-src", publisher="fw-pub")
# Add evidence for all required categories
for cat in ["PROVENANCE", "ARTIFACT", "LABEL_SEMANTICS", "TEMPORAL",
            "INDEPENDENCE", "FEATURE_COMPATIBILITY", "LEAKAGE", "SOURCE_IDENTITY"]:
    ing_fw.add_evidence("fw-pkg", evidence_type=cat,
                        captured_content=f"{cat} content",
                        verification_status="DOCUMENTED")

ok, msg = ing_fw.run_full_workflow("fw-pkg")
check("Full workflow succeeds", ok)
check("Reason is Complete", msg == "Complete")
check("Final state is READY", ing_fw.packages["fw-pkg"].state == EvidencePackageState.READY_FOR_CERTIFICATION.value)


# ── 33. Full workflow: missing evidence fails ────────────────────────────

print("\n-- 33. Full workflow with missing evidence --")

ing_fw2 = DatasetEvidenceIngestion()
ing_fw2.create_package("fw2-pkg", "c", "d", "a")
# Only add PROVENANCE, skip others
ing_fw2.add_evidence("fw2-pkg", evidence_type="PROVENANCE",
                     captured_content="content", verification_status="DOCUMENTED")
ok, msg = ing_fw2.run_full_workflow("fw2-pkg")
check("Workflow fails with missing evidence", not ok)
check("Reason mentions validation", "validation" in msg.lower() or "fail" in msg.lower())


# ── 34. Add evidence to non-existent package ─────────────────────────────

print("\n-- 34. Non-existent package --")

item = ing.add_evidence("nonexistent", "PROVENANCE")
check("Add evidence to nonexistent returns None", item is None)

ok = ing.capture_source("nonexistent")
check("Capture source to nonexistent returns False", not ok)

ok = ing.capture_artifact("nonexistent")
check("Capture artifact to nonexistent returns False", not ok)


# ── 35. Evidence item serialization ──────────────────────────────────────

print("\n-- 35. Evidence item serialization --")

e = EvidenceItem(evidence_id="ev-1", evidence_type="PROVENANCE",
                 verification_status="DOCUMENTED")
d = e.to_dict()
check("to_dict has evidence_id", d["evidence_id"] == "ev-1")
check("to_dict has verification_status", d["verification_status"] == "DOCUMENTED")

e2 = EvidenceItem.from_dict(d)
check("from_dict round-trip", e2.evidence_id == "ev-1")
check("from_dict status preserved", e2.verification_status == "DOCUMENTED")


# ── 36. Category gate serialization ──────────────────────────────────────

print("\n-- 36. Category gate serialization --")

g = EvidenceCategoryGate(category="PROVENANCE", status="PASS", blocking=True)
d = g.to_dict()
check("to_dict has category", d["category"] == "PROVENANCE")
check("to_dict has status", d["status"] == "PASS")

g2 = EvidenceCategoryGate(**d)
check("Round-trip", g2.category == "PROVENANCE")


# ── 37. Manifest serialization ───────────────────────────────────────────

print("\n-- 37. Manifest serialization --")

m = EvidenceManifest(package_id="p", dataset_id="d")
m.evidence_ids = ["ev-1"]
m.compute_hash()
d = m.to_dict()
check("to_dict has manifest_hash", len(d["manifest_hash"]) == 64)

m2 = EvidenceManifest.from_dict(d)
check("from_dict round-trip", m2.package_id == "p")
check("from_dict hash preserved", m2.manifest_hash == m.manifest_hash)


# ── 38. Package serialization ────────────────────────────────────────────

print("\n-- 38. Package serialization --")

p = EvidencePackage(package_id="test", dataset_id="ds", dataset_sha256="hash123")
p.compute_hash()
d = p.to_dict()
check("to_dict has package_hash", len(d["package_hash"]) == 64)

p2 = EvidencePackage.from_dict(d)
check("from_dict round-trip", p2.package_id == "test")
check("from_dict sha256 preserved", p2.dataset_sha256 == "hash123")


# ── 39. Export same evidence -> same hash ────────────────────────────────

print("\n-- 39. Export determinism --")

exp1 = ing.export_package("pkg-001")
exp2 = ing.export_package("pkg-001")
check("Same package -> same export hash", exp1["package_hash"] == exp2["package_hash"])
check("Same package -> same manifest hash",
      exp1["manifest"]["manifest_hash"] == exp2["manifest"]["manifest_hash"])


# ── 40. Source capture level distinction ──────────────────────────────────

print("\n-- 40. Source capture levels --")

ing_ref = DatasetEvidenceIngestion()
ing_ref.create_package("ref-pkg", "c", "d", "a")
ing_ref.capture_source("ref-pkg")
ing_ref.capture_artifact("ref-pkg", "f", 100, "h")

item_ref = ing_ref.add_evidence("ref-pkg", "PROVENANCE",
                                 source_capture_level=SourceCaptureLevel.REFERENCE_ONLY.value)
check("REFERENCE_ONLY level", item_ref.source_capture_level == SourceCaptureLevel.REFERENCE_ONLY.value)

item_doc = ing_ref.add_evidence("ref-pkg", "LICENSE",
                                 source_capture_level=SourceCaptureLevel.DOCUMENTATION_CAPTURED.value)
check("DOCUMENTATION_CAPTURED level", item_doc.source_capture_level == SourceCaptureLevel.DOCUMENTATION_CAPTURED.value)

item_ind = ing_ref.add_evidence("ref-pkg", "INDEPENDENCE",
                                 source_capture_level=SourceCaptureLevel.INDEPENDENTLY_VERIFIED.value)
check("INDEPENDENTLY_VERIFIED level", item_ind.source_capture_level == SourceCaptureLevel.INDEPENDENTLY_VERIFIED.value)


# ── 41. Blocking vs non-blocking gates ───────────────────────────────────

print("\n-- 41. Blocking vs non-blocking gates --")

ing_b = DatasetEvidenceIngestion()
ing_b.create_package("b-pkg", "c", "d", "a")
ing_b.capture_source("b-pkg")
ing_b.capture_artifact("b-pkg", "f", 100, "h")
# Only add LICENSE (non-blocking) evidence
ing_b.add_evidence("b-pkg", "LICENSE", captured_content="license", verification_status="DOCUMENTED")
ing_b.capture_documentation("b-pkg")
ing_b.compute_hashes("b-pkg")
ing_b.index_evidence("b-pkg")
all_pass, gates = ing_b.validate_evidence("b-pkg")
license_gate = [g for g in gates if g.category == "LICENSE"][0]
check("LICENSE gate passes", license_gate.status == EvidenceStatus.PASS.value)
check("LICENSE gate not blocking", not license_gate.blocking)
# But provenance gate should fail (blocking, no evidence)
prov_gate = [g for g in gates if g.category == "PROVENANCE"][0]
check("PROVENANCE gate missing", prov_gate.status == EvidenceStatus.MISSING.value)
check("PROVENANCE gate blocking", prov_gate.blocking)


# ── 42-50: Additional tests ──────────────────────────────────────────────

print("\n-- 42-50: Additional tests --")

# 42. Forensic chain: single event valid
ing_42 = DatasetEvidenceIngestion()
ing_42.create_package("p42", "c", "d", "a")
ok, errors = ing_42.verify_forensic_chain()
check("42. Single event valid", ok, str(errors))

# 43. Forensic chain: empty chain valid
ing_43 = DatasetEvidenceIngestion()
ok, errors = ing_43.verify_forensic_chain()
check("43. Empty chain valid", ok)

# 44. Package hash: different package_id -> different hash
p1 = EvidencePackage(package_id="a", dataset_id="d")
p1.compute_hash()
p2 = EvidencePackage(package_id="b", dataset_id="d")
p2.compute_hash()
check("44. Different ID -> different hash", p1.package_hash != p2.package_hash)

# 45. Package hash: different dataset_id -> different hash
p1 = EvidencePackage(package_id="a", dataset_id="d1")
p1.compute_hash()
p2 = EvidencePackage(package_id="a", dataset_id="d2")
p2.compute_hash()
check("45. Different dataset_id -> different hash", p1.package_hash != p2.package_hash)

# 46. Evidence diff: diff hash is deterministic
diff = ing3.diff_packages("p1", "p2")
diff_hash_1 = diff.diff_hash
diff = ing3.diff_packages("p1", "p2")
check("46. Diff hash deterministic", diff.diff_hash == diff_hash_1)

# 47. Import: empty/invalid data returns None
ing_47 = DatasetEvidenceIngestion()
result = ing_47.import_package({})
check("47. Empty import returns None", result is None)
result = ing_47.import_package({"package_id": ""})
check("47. Empty package_id returns None", result is None)

# 48. Validate imported package: non-existent returns False
ing_48 = DatasetEvidenceIngestion()
ok, msg = ing_48.validate_imported_package("nonexistent")
check("48. Validate nonexistent returns False", not ok)

# 49. Evidence type filtering in manifest
manifest = ing.build_manifest("pkg-001")
prov_hashes = [manifest.evidence_hashes[eid] for eid in manifest.evidence_ids
               if any(e.evidence_id == eid and e.evidence_type == "PROVENANCE"
                      for e in ing.packages["pkg-001"].evidence_items)]
check("49. Manifest has PROVENANCE hashes", len(prov_hashes) > 0)

# 50. Gate results in manifest
check("50. Manifest gate results populated", len(manifest.gate_results) > 0)


# ── 51-70: More adversarial tests ────────────────────────────────────────

print("\n-- 51-70: Adversarial edge cases --")

# 51. Package state_history records transitions
p51 = EvidencePackage(package_id="p51", dataset_id="d")
p51.transition(EvidencePackageState.SOURCE_CAPTURED)
p51.transition(EvidencePackageState.ARTIFACT_CAPTURED)
check("51. State history has 2 entries", len(p51.state_history) == 2)
check("51. First from CREATED", p51.state_history[0]["from"] == EvidencePackageState.EVIDENCE_PACKAGE_CREATED.value)

# 52. State history includes timestamp
check("52. Timestamp in history", "timestamp" in p51.state_history[0])

# 53. Evidence item without content -> empty hash
e53 = EvidenceItem(evidence_id="e53")
e53.compute_content_hash()
check("53. Empty content -> hash of empty", e53.content_hash == hashlib.sha256(b"").hexdigest())

# 54. Manifest sorted evidence IDs
m54 = EvidenceManifest(package_id="p")
m54.evidence_ids = ["ev-3", "ev-1", "ev-2"]
m54.compute_hash()
m54_2 = EvidenceManifest(package_id="p")
m54_2.evidence_ids = ["ev-1", "ev-2", "ev-3"]
m54_2.compute_hash()
check("54. Manifest sorts evidence IDs", m54.manifest_hash == m54_2.manifest_hash)

# 55. Evidence diff with missing packages
diff55 = ing.diff_packages("missing-a", "missing-b")
check("55. Diff missing packages has changes", diff55.material_changes)

# 56. Certification binding: empty expected hash
ok, msg = ing.verify_certification_binding("pkg-001", "")
check("56. Empty expected hash blocks", not ok)

# 57. Evidence status enum values
check("57. PASS value", EvidenceStatus.PASS.value == "PASS")
check("57. FAIL value", EvidenceStatus.FAIL.value == "FAIL")
check("57. MISSING value", EvidenceStatus.MISSING.value == "MISSING")
check("57. NOT_APPLICABLE value", EvidenceStatus.NOT_APPLICABLE.value == "NOT_APPLICABLE")

# 58. Source capture level enum values
check("58. REFERENCE_ONLY value", SourceCaptureLevel.REFERENCE_ONLY.value == "REFERENCE_ONLY")
check("58. DOCUMENTATION_CAPTURED value",
      SourceCaptureLevel.DOCUMENTATION_CAPTURED.value == "DOCUMENTATION_CAPTURED")
check("58. INDEPENDENTLY_VERIFIED value",
      SourceCaptureLevel.INDEPENDENTLY_VERIFIED.value == "INDEPENDENTLY_VERIFIED")

# 59. Package state enum values
check("59. CREATED value",
      EvidencePackageState.EVIDENCE_PACKAGE_CREATED.value == "EVIDENCE_PACKAGE_CREATED")
check("59. READY value",
      EvidencePackageState.READY_FOR_CERTIFICATION.value == "READY_FOR_CERTIFICATION")

# 60. Diff hash is 64 chars
diff60 = EvidenceDiff(package_a_id="a", package_b_id="b")
diff60.compute_hash()
check("60. Diff hash is 64 chars", len(diff60.diff_hash) == 64)

# 61. Manifest: gate results in sorted order
m61 = EvidenceManifest(package_id="p")
m61.gate_results = [
    EvidenceCategoryGate(category="Z", status="PASS"),
    EvidenceCategoryGate(category="A", status="PASS"),
]
m61_2 = EvidenceManifest(package_id="p")
m61_2.gate_results = [
    EvidenceCategoryGate(category="A", status="PASS"),
    EvidenceCategoryGate(category="Z", status="PASS"),
]
m61.compute_hash()
m61_2.compute_hash()
check("61. Gate result ordering doesn't affect hash", m61.manifest_hash == m61_2.manifest_hash)

# 62. Package is_test_fixture in hash
p_a = EvidencePackage(package_id="x", is_test_fixture=False)
p_a.compute_hash()
p_b = EvidencePackage(package_id="x", is_test_fixture=True)
p_b.compute_hash()
check("62. is_test_fixture affects hash", p_a.package_hash != p_b.package_hash)

# 63. Diff to_dict
diff63 = EvidenceDiff(package_a_id="a", package_b_id="b",
                      changes=[{"field": "test", "a": "1", "b": "2"}],
                      material_changes=True)
d63 = diff63.to_dict()
check("63. Diff to_dict has changes", len(d63["changes"]) == 1)
check("63. Diff to_dict material_changes", d63["material_changes"] is True)

# 64. Evidence manifest from_dict with gates
m64_dict = {
    "package_id": "p64", "dataset_id": "d", "artifact_hash": "a",
    "evidence_ids": [], "evidence_hashes": {}, "documentation_hashes": {},
    "verification_states": {},
    "gate_results": [{"category": "PROVENANCE", "status": "PASS", "evidence_count": 1,
                      "required_count": 1, "blocking": True, "details": "ok"}],
    "manifest_hash": "",
}
m64 = EvidenceManifest.from_dict(m64_dict)
check("64. Manifest from_dict with gates", len(m64.gate_results) == 1)
check("64. Gate category preserved", m64.gate_results[0].category == "PROVENANCE")

# 65. Package from_dict with items
p65_dict = {
    "package_id": "p65", "candidate_id": "c", "dataset_id": "d",
    "acquisition_id": "a", "state": "HASHES_COMPUTED",
    "evidence_items": [{"evidence_id": "ev-0", "evidence_type": "PROVENANCE",
                        "verification_status": "DOCUMENTED"}],
    "category_gates": [{"category": "PROVENANCE", "status": "PASS", "blocking": True}],
    "manifest": {"package_id": "p65", "dataset_id": "d", "artifact_hash": "",
                 "evidence_ids": [], "evidence_hashes": {}, "documentation_hashes": {},
                 "verification_states": {}, "gate_results": [], "manifest_hash": "abc"},
}
p65 = EvidencePackage.from_dict(p65_dict)
check("65. Package from_dict with items", len(p65.evidence_items) == 1)
check("65. Package from_dict with gates", len(p65.category_gates) == 1)
check("65. Package from_dict manifest", p65.manifest.manifest_hash == "abc")

# 66. EvidenceItem from_dict
e66 = EvidenceItem.from_dict({
    "evidence_id": "ev-66", "evidence_type": "TEMPORAL",
    "source_reference": "ref", "captured_content": "content",
    "content_hash": "hash123", "verification_status": "PASS",
    "source_capture_level": "DOCUMENTATION_CAPTURED",
})
check("66. EvidenceItem from_dict", e66.evidence_id == "ev-66")
check("66. From_dict content_hash", e66.content_hash == "hash123")

# 67. Multiple evidence items of same type
ing67 = DatasetEvidenceIngestion()
ing67.create_package("p67", "c", "d", "a")
ing67.capture_source("p67")
ing67.capture_artifact("p67", "f", 100, "h")
ing67.add_evidence("p67", "PROVENANCE", captured_content="doc1", verification_status="DOCUMENTED")
ing67.add_evidence("p67", "PROVENANCE", captured_content="doc2", verification_status="DOCUMENTED")
prov_items = [e for e in ing67.packages["p67"].evidence_items if e.evidence_type == "PROVENANCE"]
check("67. Multiple items same type", len(prov_items) == 2)

# 68. Run full workflow on fixture
pkg_fix2, ing_fix2 = create_fixture_evidence_package("fixture-002")
check("68. Fixture full workflow", pkg_fix2.state == EvidencePackageState.READY_FOR_CERTIFICATION.value)
check("68. Fixture forensic chain", ing_fix2.verify_forensic_chain()[0])

# 69. Export fixture
exp_fix = ing_fix2.export_package("fixture-002")
check("69. Fixture export succeeds", exp_fix is not None)
check("69. Fixture export is_test_fixture", exp_fix["is_test_fixture"] is True)

# 70. Re-import fixture and verify
ing_fix3 = DatasetEvidenceIngestion()
imp_fix = ing_fix3.import_package(exp_fix)
check("70. Fixture re-import", imp_fix is not None)
ok, msg = ing_fix3.validate_imported_package("fixture-002")
check("70. Re-imported fixture valid", ok)


# ── 71-90: State machine and workflow edge cases ──────────────────────────

print("\n-- 71-90: State machine edge cases --")

# 71. Evidence can be added at any non-terminal state (by design)
ing71 = DatasetEvidenceIngestion()
ing71.create_package("p71", "c", "d", "a")
item = ing71.add_evidence("p71", "PROVENANCE")
check("71. Add evidence at CREATED state succeeds", item is not None)
# But cannot add in BLOCKED/INVALID/READY states
ing71b = DatasetEvidenceIngestion()
p71b = ing71b.create_package("p71b", "c", "d", "a")
p71b.state = "BLOCKED"
item = ing71b.add_evidence("p71b", "PROVENANCE")
check("71b. Add evidence in BLOCKED state fails", item is None)
p71b.state = "READY_FOR_CERTIFICATION"
item = ing71b.add_evidence("p71b", "PROVENANCE")
check("71c. Add evidence in READY state fails", item is None)

# 72. BLOCKED state from INVALID transition
ing72 = DatasetEvidenceIngestion()
p72 = ing72.create_package("p72", "c", "d", "a")
ok = p72.transition(EvidencePackageState.INVALID)
check("72. CREATED -> INVALID", ok)
ok = p72.transition(EvidencePackageState.SOURCE_CAPTURED)
check("72. INVALID is terminal", not ok)

# 73. INCOMPLETE from EVIDENCE_INDEXED
ing73 = DatasetEvidenceIngestion()
p73 = ing73.create_package("p73", "c", "d", "a")
p73.transition(EvidencePackageState.SOURCE_CAPTURED)
p73.transition(EvidencePackageState.ARTIFACT_CAPTURED)
p73.transition(EvidencePackageState.DOCUMENTATION_CAPTURED)
p73.transition(EvidencePackageState.HASHES_COMPUTED)
p73.transition(EvidencePackageState.EVIDENCE_INDEXED)
ok = p73.transition(EvidencePackageState.INCOMPLETE)
check("73. INDEXED -> INCOMPLETE", ok)

# 74. INCOMPLETE from EVIDENCE_VALIDATED
ing74 = DatasetEvidenceIngestion()
p74 = ing74.create_package("p74", "c", "d", "a")
p74.transition(EvidencePackageState.SOURCE_CAPTURED)
p74.transition(EvidencePackageState.ARTIFACT_CAPTURED)
p74.transition(EvidencePackageState.DOCUMENTATION_CAPTURED)
p74.transition(EvidencePackageState.HASHES_COMPUTED)
p74.transition(EvidencePackageState.EVIDENCE_INDEXED)
p74.transition(EvidencePackageState.EVIDENCE_VALIDATED)
ok = p74.transition(EvidencePackageState.INCOMPLETE)
check("74. VALIDATED -> INCOMPLETE", ok)

# 75. BLOCKED from SOURCE_CAPTURED
ing75 = DatasetEvidenceIngestion()
p75 = ing75.create_package("p75", "c", "d", "a")
p75.transition(EvidencePackageState.SOURCE_CAPTURED)
ok = p75.transition(EvidencePackageState.BLOCKED)
check("75. SOURCE_CAPTURED -> BLOCKED", ok)

# 76. BLOCKED from ARTIFACT_CAPTURED
ing76 = DatasetEvidenceIngestion()
p76 = ing76.create_package("p76", "c", "d", "a")
p76.transition(EvidencePackageState.SOURCE_CAPTURED)
p76.transition(EvidencePackageState.ARTIFACT_CAPTURED)
ok = p76.transition(EvidencePackageState.BLOCKED)
check("76. ARTIFACT_CAPTURED -> BLOCKED", ok)

# 77. BLOCKED from DOCUMENTATION_CAPTURED
ing77 = DatasetEvidenceIngestion()
p77 = ing77.create_package("p77", "c", "d", "a")
p77.transition(EvidencePackageState.SOURCE_CAPTURED)
p77.transition(EvidencePackageState.ARTIFACT_CAPTURED)
p77.transition(EvidencePackageState.DOCUMENTATION_CAPTURED)
ok = p77.transition(EvidencePackageState.BLOCKED)
check("77. DOCUMENTATION_CAPTURED -> BLOCKED", ok)

# 78. BLOCKED from HASHES_COMPUTED
ing78 = DatasetEvidenceIngestion()
p78 = ing78.create_package("p78", "c", "d", "a")
for s in [EvidencePackageState.SOURCE_CAPTURED, EvidencePackageState.ARTIFACT_CAPTURED,
          EvidencePackageState.DOCUMENTATION_CAPTURED, EvidencePackageState.HASHES_COMPUTED]:
    p78.transition(s)
ok = p78.transition(EvidencePackageState.BLOCKED)
check("78. HASHES_COMPUTED -> BLOCKED", ok)

# 79. BLOCKED from EVIDENCE_INDEXED
ing79 = DatasetEvidenceIngestion()
p79 = ing79.create_package("p79", "c", "d", "a")
for s in [EvidencePackageState.SOURCE_CAPTURED, EvidencePackageState.ARTIFACT_CAPTURED,
          EvidencePackageState.DOCUMENTATION_CAPTURED, EvidencePackageState.HASHES_COMPUTED,
          EvidencePackageState.EVIDENCE_INDEXED]:
    p79.transition(s)
ok = p79.transition(EvidencePackageState.BLOCKED)
check("79. EVIDENCE_INDEXED -> BLOCKED", ok)

# 80. BLOCKED from EVIDENCE_VALIDATED
ing80 = DatasetEvidenceIngestion()
p80 = ing80.create_package("p80", "c", "d", "a")
for s in [EvidencePackageState.SOURCE_CAPTURED, EvidencePackageState.ARTIFACT_CAPTURED,
          EvidencePackageState.DOCUMENTATION_CAPTURED, EvidencePackageState.HASHES_COMPUTED,
          EvidencePackageState.EVIDENCE_INDEXED, EvidencePackageState.EVIDENCE_VALIDATED]:
    p80.transition(s)
ok = p80.transition(EvidencePackageState.BLOCKED)
check("80. EVIDENCE_VALIDATED -> BLOCKED", ok)

# 81. Diff: manifest hash change
ing81a = DatasetEvidenceIngestion()
ing81a.create_package("p81a", "c", "d", "a")
ing81a.capture_source("p81a")
ing81a.capture_artifact("p81a", "f", 100, "h")
ing81a.add_evidence("p81a", "PROVENANCE", captured_content="c1", verification_status="DOCUMENTED")
ing81a.capture_documentation("p81a")
ing81a.compute_hashes("p81a")
ing81a.index_evidence("p81a")
ing81a.validate_evidence("p81a")
ing81a.build_manifest("p81a")
ing81a.ready_for_certification("p81a")

ing81b = DatasetEvidenceIngestion()
ing81b.create_package("p81b", "c", "d", "a")
ing81b.capture_source("p81b")
ing81b.capture_artifact("p81b", "f", 100, "h")
ing81b.add_evidence("p81b", "PROVENANCE", captured_content="c2", verification_status="DOCUMENTED")
ing81b.capture_documentation("p81b")
ing81b.compute_hashes("p81b")
ing81b.index_evidence("p81b")
ing81b.validate_evidence("p81b")
ing81b.build_manifest("p81b")
ing81b.ready_for_certification("p81b")

diff81 = ing81a.diff_packages("p81a", "p81b", other=ing81b)
check("81. Manifest hash diff detected",
      any(c["field"] == "manifest_hash" for c in diff81.changes))

# 82-85: Evidence type diff
ing82a = DatasetEvidenceIngestion()
ing82a.create_package("p82a", "c", "d", "a")
ing82a.capture_source("p82a")
ing82a.capture_artifact("p82a", "f", 100, "h")
ing82a.add_evidence("p82a", "PROVENANCE", captured_content="c", verification_status="DOCUMENTED")
ing82a.add_evidence("p82a", "LABEL_SEMANTICS", captured_content="c", verification_status="DOCUMENTED")
ing82a.capture_documentation("p82a")
ing82a.compute_hashes("p82a")
ing82a.index_evidence("p82a")
ing82a.validate_evidence("p82a")
ing82a.build_manifest("p82a")
ing82a.ready_for_certification("p82a")

ing82b = DatasetEvidenceIngestion()
ing82b.create_package("p82b", "c", "d", "a")
ing82b.capture_source("p82b")
ing82b.capture_artifact("p82b", "f", 100, "h")
ing82b.add_evidence("p82b", "PROVENANCE", captured_content="c", verification_status="DOCUMENTED")
# Missing LABEL_SEMANTICS
ing82b.capture_documentation("p82b")
ing82b.compute_hashes("p82b")
ing82b.index_evidence("p82b")
ing82b.validate_evidence("p82b")
ing82b.build_manifest("p82b")
ing82b.ready_for_certification("p82b")

diff82 = ing82a.diff_packages("p82a", "p82b", other=ing82b)
check("82. Evidence type diff detected",
      any(c["field"] == "evidence_types" for c in diff82.changes))

# 83. Run full workflow: already completed is idempotent
ok, msg = ing.run_full_workflow("pkg-001")
check("83. Re-run full workflow is idempotent", ok)
check("83. Reason is Already ready", "Already" in msg)

# 84. Export non-existent package returns None
exp84 = ing.export_package("nonexistent")
check("84. Export nonexistent returns None", exp84 is None)

# 85. Import with valid structure
valid_export = {
    "package_id": "imp-1", "candidate_id": "c", "dataset_id": "d",
    "acquisition_id": "a", "source_name": "s", "publisher": "p",
    "original_source": "o", "distribution_reference": "", "license_reference": "",
    "release_version": "", "dataset_filename": "f.csv", "file_size": 100,
    "dataset_sha256": "hash", "acquisition_timestamp": "",
    "evidence_items": [], "category_gates": [],
    "manifest": {"package_id": "imp-1", "dataset_id": "d", "artifact_hash": "hash",
                 "evidence_ids": [], "evidence_hashes": {}, "documentation_hashes": {},
                 "verification_states": {}, "gate_results": [], "manifest_hash": ""},
    "state": "READY_FOR_CERTIFICATION", "state_history": [],
    "created_at": 0, "package_hash": "", "is_test_fixture": True,
}
ing85 = DatasetEvidenceIngestion()
imp85 = ing85.import_package(valid_export)
check("85. Import valid structure", imp85 is not None)
check("85. Import package_id", imp85.package_id == "imp-1")

# 86. Forensic event: hash is 64 chars
check("86. Forensic event hash 64 chars",
      len(ing._forensic_events[0]["event_hash"]) == 64)

# 87. Forensic event: previous_event_hash for first event
check("87. First event empty previous hash",
      ing._forensic_events[0]["previous_event_hash"] == "")

# 88. Forensic event: chain links
if len(ing._forensic_events) > 1:
    check("88. Second event chains to first",
          ing._forensic_events[1]["previous_event_hash"] == ing._forensic_events[0]["event_hash"])

# 89. Diff: both packages missing
diff89 = DatasetEvidenceIngestion().diff_packages("x", "y")
check("89. Both missing -> material change", diff89.material_changes)

# 90. Diff: one missing
diff90 = ing.diff_packages("pkg-001", "nonexistent")
check("90. One missing -> material change", diff90.material_changes)


# ── 91-120: Final adversarial tests ──────────────────────────────────────

print("\n-- 91-120: Final adversarial tests --")

# 91. Evidence content hash with bytes
e91 = EvidenceItem(evidence_id="e91")
e91.compute_content_hash(b"binary content")
check("91. Bytes content hash", len(e91.content_hash) == 64)

# 92. Package verify_integrity after compute
p92 = EvidencePackage(package_id="p92", dataset_id="d")
p92.compute_hash()
ok, msg = p92.verify_integrity()
check("92. Integrity passes after compute", ok)

# 93. Package verify_integrity: no hash -> fail
p93 = EvidencePackage()
ok, msg = p93.verify_integrity()
check("93. No hash -> integrity fail", not ok)

# 94. Category gate: FAIL status
ing94 = DatasetEvidenceIngestion()
ing94.create_package("p94", "c", "d", "a")
ing94.capture_source("p94")
ing94.capture_artifact("p94", "f", 100, "h")
ing94.add_evidence("p94", "PROVENANCE", captured_content="c",
                   verification_status=EvidenceStatus.FAIL.value)
ing94.capture_documentation("p94")
ing94.compute_hashes("p94")
ing94.index_evidence("p94")
all_pass, gates = ing94.validate_evidence("p94")
prov_gate = [g for g in gates if g.category == "PROVENANCE"][0]
check("94. FAIL evidence -> FAIL gate", prov_gate.status == EvidenceStatus.FAIL.value)

# 95. NOT_APPLICABLE evidence
ing95 = DatasetEvidenceIngestion()
ing95.create_package("p95", "c", "d", "a")
ing95.capture_source("p95")
ing95.capture_artifact("p95", "f", 100, "h")
ing95.add_evidence("p95", "LEAKAGE", captured_content="c",
                   verification_status=EvidenceStatus.NOT_APPLICABLE.value)
# No other evidence
ing95.capture_documentation("p95")
ing95.compute_hashes("p95")
ing95.index_evidence("p95")
all_pass, gates = ing95.validate_evidence("p95")
leak_gate = [g for g in gates if g.category == "LEAKAGE"][0]
check("95. NOT_APPLICABLE for leakage", leak_gate.status != EvidenceStatus.PASS.value)

# 96. Multiple exports are deterministic
exp96a = ing.export_package("pkg-001")
exp96b = ing.export_package("pkg-001")
check("96. Export determinism (package_hash)", exp96a["package_hash"] == exp96b["package_hash"])
check("96. Export determinism (manifest_hash)",
      exp96a["manifest"]["manifest_hash"] == exp96b["manifest"]["manifest_hash"])

# 97. Fixture: different fixture IDs are different packages
_, ing97a = create_fixture_evidence_package("fix-a")
_, ing97b = create_fixture_evidence_package("fix-b")
check("97. Different fixture IDs", "fix-a" in ing97a.packages)
check("97. Different fixture IDs 2", "fix-b" in ing97b.packages)

# 98. Diff sorted changes
diff98 = EvidenceDiff(
    package_a_id="a", package_b_id="b",
    changes=[{"field": "b"}, {"field": "a"}],
)
diff98.compute_hash()
diff98_hash = diff98.diff_hash
diff98_2 = EvidenceDiff(
    package_a_id="a", package_b_id="b",
    changes=[{"field": "a"}, {"field": "b"}],
)
diff98_2.compute_hash()
check("98. Diff sorts changes", diff98_hash == diff98_2.diff_hash)

# 99. Evidence item: compute hash with None content
e99 = EvidenceItem(evidence_id="e99", captured_content="")
e99.compute_content_hash(None)
check("99. None content -> hash of empty bytes",
      e99.content_hash == hashlib.sha256(b"").hexdigest())

# 100. Full workflow: state progression recorded
ing100 = DatasetEvidenceIngestion()
ing100.create_package("p100", "c", "d", "a")
for cat in ["PROVENANCE", "ARTIFACT", "LABEL_SEMANTICS", "TEMPORAL",
            "INDEPENDENCE", "FEATURE_COMPATIBILITY", "LEAKAGE", "SOURCE_IDENTITY"]:
    ing100.add_evidence("p100", evidence_type=cat, captured_content=f"{cat}",
                        verification_status="DOCUMENTED")
ok, msg = ing100.run_full_workflow("p100")
check("100. Full workflow state history", len(ing100.packages["p100"].state_history) > 0)

# 101. Forensic events include all lifecycle events
event_types = [e["event_type"] for e in ing._forensic_events]
check("101. Has EVIDENCE_PACKAGE_CREATED", "EVIDENCE_PACKAGE_CREATED" in event_types)
check("101. Has SOURCE_CAPTURED", "SOURCE_CAPTURED" in event_types)
check("101. Has ARTIFACT_CAPTURED", "ARTIFACT_CAPTURED" in event_types)
check("101. Has DOCUMENTATION_CAPTURED", "DOCUMENTATION_CAPTURED" in event_types)

# 102. Evidence manifest: verification_states populated
manifest = ing.build_manifest("pkg-001")
check("102. Manifest verification_states", len(manifest.verification_states) > 0)

# 103. Evidence manifest: documentation_hashes populated
check("103. Manifest documentation_hashes", len(manifest.documentation_hashes) > 0)

# 104. Package to_dict includes all nested structures
p104 = EvidencePackage(package_id="p104", dataset_id="d")
p104.evidence_items.append(EvidenceItem(evidence_id="ev", evidence_type="PROVENANCE"))
p104.category_gates.append(EvidenceCategoryGate(category="PROVENANCE", status="PASS"))
p104.manifest = EvidenceManifest(package_id="p104")
p104.manifest.compute_hash()
d104 = p104.to_dict()
check("104. to_dict has evidence_items", len(d104["evidence_items"]) == 1)
check("104. to_dict has category_gates", len(d104["category_gates"]) == 1)
check("104. to_dict has manifest", "manifest" in d104)

# 105. Evidence manifest from_dict without gate_results
m105 = EvidenceManifest.from_dict({
    "package_id": "p105", "dataset_id": "d", "artifact_hash": "a",
    "evidence_ids": [], "evidence_hashes": {}, "documentation_hashes": {},
    "verification_states": {},
})
check("105. Manifest from_dict no gates", len(m105.gate_results) == 0)

# 106. Evidence diff: category gate change detection
ing106a = DatasetEvidenceIngestion()
ing106a.create_package("p106a", "c", "d", "a")
ing106a.capture_source("p106a")
ing106a.capture_artifact("p106a", "f", 100, "h")
ing106a.add_evidence("p106a", "PROVENANCE", captured_content="c", verification_status="DOCUMENTED")
ing106a.capture_documentation("p106a")
ing106a.compute_hashes("p106a")
ing106a.index_evidence("p106a")
ing106a.validate_evidence("p106a")
ing106a.build_manifest("p106a")
ing106a.ready_for_certification("p106a")

ing106b = DatasetEvidenceIngestion()
ing106b.create_package("p106b", "c", "d", "a")
ing106b.capture_source("p106b")
ing106b.capture_artifact("p106b", "f", 100, "h")
ing106b.add_evidence("p106b", "PROVENANCE", captured_content="c",
                     verification_status=EvidenceStatus.FAIL.value)
ing106b.capture_documentation("p106b")
ing106b.compute_hashes("p106b")
ing106b.index_evidence("p106b")
ing106b.validate_evidence("p106b")
ing106b.build_manifest("p106b")
ing106b.ready_for_certification("p106b")

diff106 = ing106a.diff_packages("p106a", "p106b", other=ing106b)
check("106. Gate change detected",
      any(c["field"] == "category_gates" for c in diff106.changes))

# 107. Package state history includes reason
p107 = EvidencePackage(package_id="p107", dataset_id="d")
p107.transition(EvidencePackageState.SOURCE_CAPTURED, "because test")
check("107. Reason in history", p107.state_history[0]["reason"] == "because test")

# 108. Forensic event actor_type defaults to SYSTEM
check("108. Forensic actor_type SYSTEM",
      ing._forensic_events[0]["actor_type"] == "SYSTEM")

# 109. Export fixture round-trip
exp_fix = ing_fix2.export_package("fixture-002")
ing_fix4 = DatasetEvidenceIngestion()
imp_fix = ing_fix4.import_package(exp_fix)
ok, msg = ing_fix4.validate_imported_package("fixture-002")
check("109. Fixture round-trip integrity", ok)

# 110. Manifest sorted gate results
m110 = EvidenceManifest(package_id="p")
m110.gate_results = [
    EvidenceCategoryGate(category="Z_A", status="MISSING", blocking=True),
    EvidenceCategoryGate(category="A_Z", status="PASS", blocking=True),
]
m110.compute_hash()
m110_2 = EvidenceManifest(package_id="p")
m110_2.gate_results = [
    EvidenceCategoryGate(category="A_Z", status="PASS", blocking=True),
    EvidenceCategoryGate(category="Z_A", status="MISSING", blocking=True),
]
m110_2.compute_hash()
check("110. Gate results sorted for determinism", m110.manifest_hash == m110_2.manifest_hash)

# 111. Diff material_changes flag
diff111 = EvidenceDiff(material_changes=False)
check("111. No changes -> not material", not diff111.material_changes)
diff111b = EvidenceDiff(material_changes=True, changes=[{"field": "test"}])
check("111. Changes -> material", diff111b.material_changes)

# 112. Package: acquisition_timestamp defaults to empty
p112 = EvidencePackage(package_id="p112")
check("112. Default acquisition_timestamp empty", p112.acquisition_timestamp == "")

# 113. Evidence item: source_reference preserved
e113 = EvidenceItem(evidence_id="e113", source_reference="https://example.com/data")
d113 = e113.to_dict()
check("113. source_reference in to_dict", d113["source_reference"] == "https://example.com/data")

# 114. Forensic chain: 3 events all link correctly
ing114 = DatasetEvidenceIngestion()
ing114.create_package("p114", "c", "d", "a")
ing114.capture_source("p114")
ing114.capture_artifact("p114", "f", 100, "h")
ok, errors = ing114.verify_forensic_chain()
check("114. 3-event chain valid", ok, str(errors))
check("114. 3 events", len(ing114._forensic_events) == 3)

# 115. Package file_size in hash
p115a = EvidencePackage(package_id="p115", file_size=100)
p115a.compute_hash()
p115b = EvidencePackage(package_id="p115", file_size=200)
p115b.compute_hash()
check("115. file_size in hash", p115a.package_hash != p115b.package_hash)

# 116. Package dataset_filename in hash
p116a = EvidencePackage(package_id="p116", dataset_filename="a.csv")
p116a.compute_hash()
p116b = EvidencePackage(package_id="p116", dataset_filename="b.csv")
p116b.compute_hash()
check("116. filename in hash", p116a.package_hash != p116b.package_hash)

# 117. Manifest dataset_id in hash
m117a = EvidenceManifest(package_id="p", dataset_id="d1")
m117a.compute_hash()
m117b = EvidenceManifest(package_id="p", dataset_id="d2")
m117b.compute_hash()
check("117. dataset_id in manifest hash", m117a.manifest_hash != m117b.manifest_hash)

# 118. Manifest artifact_hash in hash
m118a = EvidenceManifest(package_id="p", artifact_hash="a1")
m118a.compute_hash()
m118b = EvidenceManifest(package_id="p", artifact_hash="a2")
m118b.compute_hash()
check("118. artifact_hash in manifest hash", m118a.manifest_hash != m118b.manifest_hash)

# 119. Empty forensic chain is valid
ing119 = DatasetEvidenceIngestion()
ok, errors = ing119.verify_forensic_chain()
check("119. Empty chain valid", ok)

# 120. Forensic chain: empty event hash detection
ing120 = DatasetEvidenceIngestion()
ing120._forensic_events.append({"event_type": "TEST", "package_id": "p",
                                 "dataset_id": "d", "details": "",
                                 "actor_type": "SYSTEM", "previous_event_hash": "",
                                 "event_hash": ""})
ok, errors = ing120.verify_forensic_chain()
check("120. Empty event hash detected", not ok)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"PHASE 60 RESULTS: {passed}/{total} passed, {failed} failed")
print("="*60)

if failed > 0:
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
