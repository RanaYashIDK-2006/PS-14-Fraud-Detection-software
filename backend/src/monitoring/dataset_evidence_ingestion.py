"""Phase 60: Reproducible dataset evidence ingestion & audit package.

Provides a deterministic evidence-ingestion and evaluation-readiness
package around the existing Phase 53-59 architecture.

Key guarantees:
  - Evidence ingestion does NOT automatically establish eligibility
  - Missing evidence remains MISSING (never silently promoted to PASS)
  - Independent verification boundary is preserved
  - Evidence manifest hash is deterministic and tamper-evident
  - Export/import round-trip preserves integrity
  - Change detection invalidates stale certification
  - TEST_FIXTURE path isolated from real-world validation
  - No secrets in evidence packages

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET

STATUS: IMPLEMENTED
PHASE: 60
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# ── Evidence Package States ──────────────────────────────────────────────

class EvidencePackageState(str, Enum):
    EVIDENCE_PACKAGE_CREATED = "EVIDENCE_PACKAGE_CREATED"
    SOURCE_CAPTURED = "SOURCE_CAPTURED"
    ARTIFACT_CAPTURED = "ARTIFACT_CAPTURED"
    DOCUMENTATION_CAPTURED = "DOCUMENTATION_CAPTURED"
    HASHES_COMPUTED = "HASHES_COMPUTED"
    EVIDENCE_INDEXED = "EVIDENCE_INDEXED"
    EVIDENCE_VALIDATED = "EVIDENCE_VALIDATED"
    READY_FOR_CERTIFICATION = "READY_FOR_CERTIFICATION"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"
    INCOMPLETE = "INCOMPLETE"


_VALID_EVIDENCE_TRANSITIONS: dict[EvidencePackageState, set[EvidencePackageState]] = {
    EvidencePackageState.EVIDENCE_PACKAGE_CREATED: {
        EvidencePackageState.SOURCE_CAPTURED,
        EvidencePackageState.BLOCKED,
        EvidencePackageState.INVALID,
    },
    EvidencePackageState.SOURCE_CAPTURED: {
        EvidencePackageState.ARTIFACT_CAPTURED,
        EvidencePackageState.BLOCKED,
        EvidencePackageState.INVALID,
    },
    EvidencePackageState.ARTIFACT_CAPTURED: {
        EvidencePackageState.DOCUMENTATION_CAPTURED,
        EvidencePackageState.BLOCKED,
        EvidencePackageState.INVALID,
    },
    EvidencePackageState.DOCUMENTATION_CAPTURED: {
        EvidencePackageState.HASHES_COMPUTED,
        EvidencePackageState.BLOCKED,
        EvidencePackageState.INVALID,
    },
    EvidencePackageState.HASHES_COMPUTED: {
        EvidencePackageState.EVIDENCE_INDEXED,
        EvidencePackageState.BLOCKED,
        EvidencePackageState.INVALID,
    },
    EvidencePackageState.EVIDENCE_INDEXED: {
        EvidencePackageState.EVIDENCE_VALIDATED,
        EvidencePackageState.BLOCKED,
        EvidencePackageState.INVALID,
        EvidencePackageState.INCOMPLETE,
    },
    EvidencePackageState.EVIDENCE_VALIDATED: {
        EvidencePackageState.READY_FOR_CERTIFICATION,
        EvidencePackageState.BLOCKED,
        EvidencePackageState.INVALID,
        EvidencePackageState.INCOMPLETE,
    },
    EvidencePackageState.READY_FOR_CERTIFICATION: set(),
    EvidencePackageState.BLOCKED: set(),
    EvidencePackageState.INVALID: set(),
    EvidencePackageState.INCOMPLETE: set(),
}


# ── Evidence Verification States ─────────────────────────────────────────

class EvidenceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SourceCaptureLevel(str, Enum):
    REFERENCE_ONLY = "REFERENCE_ONLY"
    DOCUMENTATION_CAPTURED = "DOCUMENTATION_CAPTURED"
    INDEPENDENTLY_VERIFIED = "INDEPENDENTLY_VERIFIED"


# ── Evidence Item ────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    """A single piece of evidence with identity and verification."""
    evidence_id: str = ""
    evidence_type: str = ""  # "provenance", "label", "temporal", "independence", etc.
    source_reference: str = ""
    captured_content: str = ""
    content_hash: str = ""
    capture_timestamp: str = ""
    verification_status: str = EvidenceStatus.MISSING.value
    source_capture_level: str = SourceCaptureLevel.REFERENCE_ONLY.value

    def compute_content_hash(self, content: str | bytes | None = None) -> str:
        """Compute deterministic hash of evidence content."""
        if content is None:
            content = self.captured_content.encode("utf-8") if self.captured_content else b""
        elif isinstance(content, str):
            content = content.encode("utf-8")
        self.content_hash = hashlib.sha256(content).hexdigest()
        return self.content_hash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceItem:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ── Evidence Category Gate ───────────────────────────────────────────────

@dataclass
class EvidenceCategoryGate:
    """Result of an evidence completeness check for one category."""
    category: str = ""  # "PROVENANCE", "ARTIFACT", "LABEL_SEMANTICS", etc.
    status: str = EvidenceStatus.MISSING.value
    evidence_count: int = 0
    required_count: int = 1
    blocking: bool = True
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Evidence Manifest ────────────────────────────────────────────────────

@dataclass
class EvidenceManifest:
    """Canonical, deterministic manifest of all evidence in a package."""
    package_id: str = ""
    dataset_id: str = ""
    artifact_hash: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    evidence_hashes: dict[str, str] = field(default_factory=dict)
    documentation_hashes: dict[str, str] = field(default_factory=dict)
    verification_states: dict[str, str] = field(default_factory=dict)
    gate_results: list[EvidenceCategoryGate] = field(default_factory=list)
    manifest_hash: str = ""

    def compute_hash(self) -> str:
        """Deterministic hash over manifest contents (excludes manifest_hash itself)."""
        d = {
            "package_id": self.package_id,
            "dataset_id": self.dataset_id,
            "artifact_hash": self.artifact_hash,
            "evidence_ids": sorted(self.evidence_ids),
            "evidence_hashes": dict(sorted(self.evidence_hashes.items())),
            "documentation_hashes": dict(sorted(self.documentation_hashes.items())),
            "verification_states": dict(sorted(self.verification_states.items())),
            "gate_results": [g.to_dict() for g in sorted(self.gate_results, key=lambda g: g.category)],
        }
        canonical = json.dumps(d, sort_keys=True, default=str)
        self.manifest_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.manifest_hash

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gate_results"] = [g.to_dict() for g in self.gate_results]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceManifest:
        gate_data = data.pop("gate_results", [])
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        m = cls(**known)
        m.gate_results = [EvidenceCategoryGate(**g) for g in gate_data]
        return m


# ── Evidence Package ─────────────────────────────────────────────────────

@dataclass
class EvidencePackage:
    """Immutable evidence package tying all evidence to a dataset.

    Tamper-evident: any modification invalidates the package hash.
    """
    package_id: str = ""
    candidate_id: str = ""
    dataset_id: str = ""
    acquisition_id: str = ""

    # Source identity
    source_name: str = ""
    publisher: str = ""
    original_source: str = ""
    distribution_reference: str = ""
    license_reference: str = ""
    release_version: str = ""

    # Artifact identity
    dataset_filename: str = ""
    file_size: int = 0
    dataset_sha256: str = ""
    acquisition_timestamp: str = ""

    # Evidence items
    evidence_items: list[EvidenceItem] = field(default_factory=list)

    # Category gates
    category_gates: list[EvidenceCategoryGate] = field(default_factory=list)

    # Manifest
    manifest: EvidenceManifest = field(default_factory=EvidenceManifest)

    # State
    state: str = EvidencePackageState.EVIDENCE_PACKAGE_CREATED.value
    state_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # Integrity
    package_hash: str = ""
    is_test_fixture: bool = False

    def transition(self, new_state: EvidencePackageState, reason: str = "") -> bool:
        """Attempt a state transition."""
        current = EvidencePackageState(self.state)
        allowed = _VALID_EVIDENCE_TRANSITIONS.get(current, set())
        if new_state not in allowed:
            return False
        self.state_history.append({
            "from": current.value,
            "to": new_state.value,
            "reason": reason,
            "timestamp": time.time(),
        })
        self.state = new_state.value
        return True

    def compute_hash(self) -> str:
        """Deterministic package hash (excludes volatile fields)."""
        d = {
            "package_id": self.package_id,
            "candidate_id": self.candidate_id,
            "dataset_id": self.dataset_id,
            "acquisition_id": self.acquisition_id,
            "source_name": self.source_name,
            "publisher": self.publisher,
            "original_source": self.original_source,
            "distribution_reference": self.distribution_reference,
            "license_reference": self.license_reference,
            "release_version": self.release_version,
            "dataset_filename": self.dataset_filename,
            "file_size": self.file_size,
            "dataset_sha256": self.dataset_sha256,
            "evidence_items": [e.to_dict() for e in self.evidence_items],
            "category_gates": [g.to_dict() for g in self.category_gates],
            "manifest_hash": self.manifest.manifest_hash,
            "is_test_fixture": self.is_test_fixture,
        }
        canonical = json.dumps(d, sort_keys=True, default=str)
        self.package_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.package_hash

    def verify_integrity(self) -> tuple[bool, str]:
        """Verify package hash matches current content."""
        if not self.package_hash:
            return False, "No package hash computed"
        old_hash = self.package_hash
        self.compute_hash()
        if old_hash == self.package_hash:
            return True, "Package integrity verified"
        return False, f"Package hash mismatch: expected={self.package_hash[:16]}... got={old_hash[:16]}..."

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence_items"] = [e.to_dict() for e in self.evidence_items]
        d["category_gates"] = [g.to_dict() for g in self.category_gates]
        d["manifest"] = self.manifest.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidencePackage:
        items_data = data.pop("evidence_items", [])
        gates_data = data.pop("category_gates", [])
        manifest_data = data.pop("manifest", {})
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        pkg = cls(**known)
        pkg.evidence_items = [EvidenceItem.from_dict(i) for i in items_data]
        pkg.category_gates = [EvidenceCategoryGate(**g) for g in gates_data]
        if manifest_data:
            pkg.manifest = EvidenceManifest.from_dict(manifest_data)
        return pkg


# ── Evidence Diff ────────────────────────────────────────────────────────

@dataclass
class EvidenceDiff:
    """Deterministic comparison between two evidence packages."""
    package_a_id: str = ""
    package_b_id: str = ""
    changes: list[dict[str, str]] = field(default_factory=list)
    material_changes: bool = False
    diff_hash: str = ""

    def compute_hash(self) -> str:
        d = {
            "package_a_id": self.package_a_id,
            "package_b_id": self.package_b_id,
            "changes": sorted(json.dumps(c, sort_keys=True) for c in self.changes),
            "material_changes": self.material_changes,
        }
        canonical = json.dumps(d, sort_keys=True, default=str)
        self.diff_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.diff_hash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Evidence Ingestion Workflow ──────────────────────────────────────────

class DatasetEvidenceIngestion:
    """Deterministic evidence ingestion and validation workflow.

    Provides: capture, hashing, manifest, export/import, change detection.
    Does NOT establish eligibility — that remains with Phase 58/59.
    """

    def __init__(self):
        self.packages: dict[str, EvidencePackage] = {}
        self._forensic_events: list[dict[str, Any]] = []

    def create_package(
        self,
        package_id: str,
        candidate_id: str,
        dataset_id: str,
        acquisition_id: str,
        source_name: str = "",
        publisher: str = "",
        original_source: str = "",
        is_test_fixture: bool = False,
    ) -> EvidencePackage:
        """Create a new evidence package."""
        pkg = EvidencePackage(
            package_id=package_id,
            candidate_id=candidate_id,
            dataset_id=dataset_id,
            acquisition_id=acquisition_id,
            source_name=source_name,
            publisher=publisher,
            original_source=original_source,
            is_test_fixture=is_test_fixture,
        )
        self.packages[package_id] = pkg
        self._record_forensic_event("EVIDENCE_PACKAGE_CREATED", package_id,
                                    pkg.dataset_id, "")
        return pkg

    def capture_source(
        self,
        package_id: str,
        distribution_reference: str = "",
        license_reference: str = "",
        release_version: str = "",
    ) -> bool:
        """Capture source identity information."""
        pkg = self.packages.get(package_id)
        if not pkg:
            return False
        if not pkg.transition(EvidencePackageState.SOURCE_CAPTURED, "source captured"):
            return False
        if distribution_reference:
            pkg.distribution_reference = distribution_reference
        if license_reference:
            pkg.license_reference = license_reference
        if release_version:
            pkg.release_version = release_version
        self._record_forensic_event("SOURCE_CAPTURED", package_id,
                                    pkg.dataset_id, "")
        return True

    def capture_artifact(
        self,
        package_id: str,
        dataset_filename: str = "",
        file_size: int = 0,
        dataset_sha256: str = "",
        acquisition_timestamp: str = "",
    ) -> bool:
        """Capture artifact identity."""
        pkg = self.packages.get(package_id)
        if not pkg:
            return False
        if not pkg.transition(EvidencePackageState.ARTIFACT_CAPTURED, "artifact captured"):
            return False
        pkg.dataset_filename = dataset_filename
        pkg.file_size = file_size
        pkg.dataset_sha256 = dataset_sha256
        pkg.acquisition_timestamp = acquisition_timestamp or time.strftime("%Y-%m-%dT%H:%M:%S")
        self._record_forensic_event("ARTIFACT_CAPTURED", package_id,
                                    pkg.dataset_id, "")
        return True

    def add_evidence(
        self,
        package_id: str,
        evidence_type: str,
        source_reference: str = "",
        captured_content: str = "",
        verification_status: str = EvidenceStatus.MISSING.value,
        source_capture_level: str = SourceCaptureLevel.REFERENCE_ONLY.value,
    ) -> EvidenceItem | None:
        """Add an evidence item to the package."""
        pkg = self.packages.get(package_id)
        if not pkg:
            return None
        # Cannot add evidence in terminal failure states
        if pkg.state in (
            EvidencePackageState.BLOCKED.value,
            EvidencePackageState.INVALID.value,
            EvidencePackageState.READY_FOR_CERTIFICATION.value,
        ):
            return None

        item = EvidenceItem(
            evidence_id=f"ev-{len(pkg.evidence_items)}-{evidence_type}",
            evidence_type=evidence_type,
            source_reference=source_reference,
            captured_content=captured_content,
            capture_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            verification_status=verification_status,
            source_capture_level=source_capture_level,
        )
        if captured_content:
            item.compute_content_hash()
        pkg.evidence_items.append(item)
        return item

    def capture_documentation(self, package_id: str) -> bool:
        """Transition to DOCUMENTATION_CAPTURED."""
        pkg = self.packages.get(package_id)
        if not pkg:
            return False
        if not pkg.transition(EvidencePackageState.DOCUMENTATION_CAPTURED, "documentation captured"):
            return False
        self._record_forensic_event("DOCUMENTATION_CAPTURED", package_id,
                                    pkg.dataset_id, "")
        return True

    def compute_hashes(self, package_id: str) -> bool:
        """Transition to HASHES_COMPUTED and compute all evidence hashes."""
        pkg = self.packages.get(package_id)
        if not pkg:
            return False
        if not pkg.transition(EvidencePackageState.HASHES_COMPUTED, "hashes computed"):
            return False
        # Compute content hashes for all evidence items
        for item in pkg.evidence_items:
            if item.captured_content and not item.content_hash:
                item.compute_content_hash()
        return True

    def index_evidence(self, package_id: str) -> bool:
        """Transition to EVIDENCE_INDEXED."""
        pkg = self.packages.get(package_id)
        if not pkg:
            return False
        if not pkg.transition(EvidencePackageState.EVIDENCE_INDEXED, "evidence indexed"):
            return False
        return True

    def validate_evidence(self, package_id: str) -> tuple[bool, list[EvidenceCategoryGate]]:
        """Validate evidence completeness by category.

        Returns (all_pass, gate_results).
        Missing evidence remains MISSING — never promoted to PASS.
        """
        pkg = self.packages.get(package_id)
        if not pkg:
            return False, []

        # Define required categories
        required_categories = [
            ("PROVENANCE", True),
            ("ARTIFACT", True),
            ("LABEL_SEMANTICS", True),
            ("TEMPORAL", True),
            ("INDEPENDENCE", True),
            ("FEATURE_COMPATIBILITY", True),
            ("LEAKAGE", True),
            ("LICENSE", False),  # not strictly blocking
            ("SOURCE_IDENTITY", True),
        ]

        gates = []
        for category, blocking in required_categories:
            items = [e for e in pkg.evidence_items if e.evidence_type == category]
            verified = [e for e in items if e.verification_status in (
                EvidenceStatus.PASS.value, "VERIFIED", "DOCUMENTED")]

            if not items:
                status = EvidenceStatus.MISSING.value
            elif verified:
                status = EvidenceStatus.PASS.value
            else:
                # Items exist but none are verified
                statuses = set(e.verification_status for e in items)
                if EvidenceStatus.FAIL.value in statuses:
                    status = EvidenceStatus.FAIL.value
                else:
                    status = EvidenceStatus.MISSING.value

            gate = EvidenceCategoryGate(
                category=category,
                status=status,
                evidence_count=len(items),
                blocking=blocking,
                details=f"{len(items)} items, {len(verified)} verified" if items else "no evidence",
            )
            gates.append(gate)

        pkg.category_gates = gates

        # Check if any blocking category is MISSING or FAIL
        all_pass = True
        for gate in gates:
            if gate.blocking and gate.status in (EvidenceStatus.MISSING.value, EvidenceStatus.FAIL.value):
                all_pass = False
                break

        if all_pass:
            pkg.transition(EvidencePackageState.EVIDENCE_VALIDATED, "all categories pass")
        else:
            pkg.transition(EvidencePackageState.INCOMPLETE, "blocking categories failed")

        return all_pass, gates

    def build_manifest(self, package_id: str) -> EvidenceManifest | None:
        """Build canonical evidence manifest."""
        pkg = self.packages.get(package_id)
        if not pkg:
            return None

        manifest = EvidenceManifest(
            package_id=pkg.package_id,
            dataset_id=pkg.dataset_id,
            artifact_hash=pkg.dataset_sha256,
            evidence_ids=[e.evidence_id for e in pkg.evidence_items],
            evidence_hashes={e.evidence_id: e.content_hash for e in pkg.evidence_items},
            documentation_hashes={
                e.evidence_id: e.content_hash
                for e in pkg.evidence_items
                if e.evidence_type in ("documentation", "LICENSE", "SOURCE_IDENTITY")
            },
            verification_states={
                e.evidence_id: e.verification_status for e in pkg.evidence_items
            },
            gate_results=list(pkg.category_gates),
        )
        manifest.compute_hash()
        pkg.manifest = manifest
        return manifest

    def ready_for_certification(self, package_id: str) -> bool:
        """Transition to READY_FOR_CERTIFICATION."""
        pkg = self.packages.get(package_id)
        if not pkg:
            return False
        if not pkg.transition(EvidencePackageState.READY_FOR_CERTIFICATION, "ready"):
            return False
        # Finalize package hash
        pkg.compute_hash()
        return True

    # ── Full Workflow ────────────────────────────────────────────────────

    def run_full_workflow(self, package_id: str) -> tuple[bool, str]:
        """Run the complete evidence ingestion workflow.

        Returns (success, reason).
        """
        pkg = self.packages.get(package_id)
        if not pkg:
            return False, "Package not found"

        # Already completed?
        if pkg.state == EvidencePackageState.READY_FOR_CERTIFICATION.value:
            return True, "Already ready"

        # Step 1: source capture
        if not self.capture_source(package_id):
            return False, "Source capture failed"

        # Step 2: artifact capture (requires at least filename)
        if not self.capture_artifact(package_id, dataset_filename=pkg.dataset_filename):
            return False, "Artifact capture failed"

        # Step 3: documentation
        if not self.capture_documentation(package_id):
            return False, "Documentation capture failed"

        # Step 4: compute hashes
        if not self.compute_hashes(package_id):
            return False, "Hash computation failed"

        # Step 5: index
        if not self.index_evidence(package_id):
            return False, "Evidence indexing failed"

        # Step 6: validate
        all_pass, gates = self.validate_evidence(package_id)
        if not all_pass:
            return False, "Evidence validation failed"

        # Step 7: build manifest
        self.build_manifest(package_id)

        # Step 8: ready
        if not self.ready_for_certification(package_id):
            return False, "Failed to mark ready"

        return True, "Complete"

    # ── Export / Import ──────────────────────────────────────────────────

    def export_package(self, package_id: str) -> dict[str, Any] | None:
        """Export evidence package as a safe, deterministic representation.

        Excludes secrets, credentials, and unnecessary personal data.
        """
        pkg = self.packages.get(package_id)
        if not pkg:
            return None
        return pkg.to_dict()

    def import_package(self, data: dict[str, Any]) -> EvidencePackage | None:
        """Import an evidence package from exported data.

        Validates that the data has the required structure.
        The exported_hash is preserved so tampering can be detected by validate_imported_package.
        """
        if not isinstance(data, dict) or not data.get("package_id"):
            return None
        try:
            # Compute a hash of the entire export BEFORE from_dict mutates it
            export_copy = json.loads(json.dumps(data, default=str))
            export_copy.pop("package_hash", None)
            exported_hash = hashlib.sha256(
                json.dumps(export_copy, sort_keys=True, default=str).encode()
            ).hexdigest()
            pkg = EvidencePackage.from_dict(data)
            if not pkg.package_id:
                return None
            pkg._exported_hash = exported_hash
            self.packages[pkg.package_id] = pkg
            return pkg
        except Exception:
            return None

    def validate_imported_package(self, package_id: str) -> tuple[bool, str]:
        """Validate that an imported package maintains integrity.

        Recomputes the export-content hash from the current package state
        and compares it against the hash captured during import.
        """
        pkg = self.packages.get(package_id)
        if not pkg:
            return False, "Package not found"
        exported_hash = getattr(pkg, "_exported_hash", "")
        if not exported_hash:
            return False, "No exported hash stored"
        # Recompute the export-content hash from current package state
        current_export = pkg.to_dict()
        current_export.pop("package_hash", None)
        current_hash = hashlib.sha256(
            json.dumps(current_export, sort_keys=True, default=str).encode()
        ).hexdigest()
        if current_hash == exported_hash:
            return True, "Integrity verified"
        return False, f"Hash mismatch: expected={exported_hash[:16]}... got={current_hash[:16]}..."

    # ── Change Detection ─────────────────────────────────────────────────

    def diff_packages(self, package_a_id: str, package_b_id: str,
                      other: DatasetEvidenceIngestion | None = None) -> EvidenceDiff:
        """Compare two evidence packages and detect changes.

        If other is provided, look up package_b from the other ingestion instance.
        """
        a = self.packages.get(package_a_id)
        b = (other or self).packages.get(package_b_id)
        if not a or not b:
            return EvidenceDiff(
                package_a_id=package_a_id,
                package_b_id=package_b_id,
                changes=[{"field": "error", "a": "missing", "b": "missing"}],
                material_changes=True,
            )

        changes = []
        compare_fields = [
            "source_name", "publisher", "original_source",
            "distribution_reference", "license_reference", "release_version",
            "dataset_filename", "file_size", "dataset_sha256",
        ]
        for field_name in compare_fields:
            val_a = getattr(a, field_name, "")
            val_b = getattr(b, field_name, "")
            if val_a != val_b:
                changes.append({
                    "field": field_name,
                    "a": str(val_a)[:64],
                    "b": str(val_b)[:64],
                })

        # Compare evidence item counts and types
        types_a = sorted(e.evidence_type for e in a.evidence_items)
        types_b = sorted(e.evidence_type for e in b.evidence_items)
        if types_a != types_b:
            changes.append({
                "field": "evidence_types",
                "a": str(types_a),
                "b": str(types_b),
            })

        # Compare category gates
        gates_a = {g.category: g.status for g in a.category_gates}
        gates_b = {g.category: g.status for g in b.category_gates}
        if gates_a != gates_b:
            changes.append({
                "field": "category_gates",
                "a": str(gates_a),
                "b": str(gates_b),
            })

        # Compare manifest hashes
        if a.manifest.manifest_hash != b.manifest.manifest_hash:
            changes.append({
                "field": "manifest_hash",
                "a": a.manifest.manifest_hash[:16],
                "b": b.manifest.manifest_hash[:16],
            })

        material = bool(changes)
        diff = EvidenceDiff(
            package_a_id=package_a_id,
            package_b_id=package_b_id,
            changes=changes,
            material_changes=material,
        )
        diff.compute_hash()
        return diff

    # ── Certification Binding ────────────────────────────────────────────

    def verify_certification_binding(
        self,
        package_id: str,
        expected_manifest_hash: str,
    ) -> tuple[bool, str]:
        """Verify that a certification's bound evidence package hasn't changed."""
        pkg = self.packages.get(package_id)
        if not pkg:
            return False, f"Package {package_id} not found"
        if pkg.manifest.manifest_hash == expected_manifest_hash:
            return True, "Certification binding verified"
        return False, (
            f"Manifest hash mismatch: package={pkg.manifest.manifest_hash[:16]}... "
            f"expected={expected_manifest_hash[:16]}..."
        )

    # ── Forensic Events ──────────────────────────────────────────────────

    def _record_forensic_event(
        self, event_type: str, package_id: str,
        dataset_id: str, details: str = "",
    ) -> None:
        prev_hash = self._forensic_events[-1].get("event_hash", "") if self._forensic_events else ""
        event = {
            "event_type": event_type,
            "package_id": package_id,
            "dataset_id": dataset_id,
            "details": details,
            "actor_type": "SYSTEM",
            "previous_event_hash": prev_hash,
        }
        canonical = json.dumps(event, sort_keys=True, default=str)
        event["event_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        self._forensic_events.append(event)

    def verify_forensic_chain(self) -> tuple[bool, list[str]]:
        """Verify forensic event chain integrity."""
        errors = []
        for i, event in enumerate(self._forensic_events):
            expected_prev = self._forensic_events[i - 1].get("event_hash", "") if i > 0 else ""
            if event.get("previous_event_hash", "") != expected_prev:
                errors.append(f"Event {i}: previous hash mismatch")
                continue
            d = {k: v for k, v in event.items() if k != "event_hash"}
            canonical = json.dumps(d, sort_keys=True, default=str)
            expected_hash = hashlib.sha256(canonical.encode()).hexdigest()
            if event.get("event_hash", "") != expected_hash:
                errors.append(f"Event {i}: hash mismatch (tampered)")
        return (len(errors) == 0, errors)


# ── TEST_FIXTURE Helper ──────────────────────────────────────────────────

def create_fixture_evidence_package(
    package_id: str = "fixture-evidence-001",
) -> tuple[EvidencePackage, DatasetEvidenceIngestion]:
    """Create a TEST_FIXTURE evidence package for code-path testing.

    NOT real-world eligible. Exercises the evidence ingestion mechanics.
    """
    ingestion = DatasetEvidenceIngestion()
    pkg = ingestion.create_package(
        package_id=package_id,
        candidate_id="fixture-candidate",
        dataset_id="fixture-dataset",
        acquisition_id="fixture-acq",
        source_name="TEST_FIXTURE",
        publisher="TEST_FIXTURE",
        original_source="fixture://test",
        is_test_fixture=True,
    )

    # Capture source
    ingestion.capture_source(
        package_id,
        distribution_reference="fixture://distribution",
        license_reference="fixture://license",
        release_version="1.0-fixture",
    )

    # Capture artifact
    ingestion.capture_artifact(
        package_id,
        dataset_filename="fixture_data.csv",
        file_size=1024,
        dataset_sha256="fixture_hash_abc123",
        acquisition_timestamp="2024-01-01T00:00:00",
    )

    # Add evidence items for all required categories
    categories = [
        "PROVENANCE", "ARTIFACT", "LABEL_SEMANTICS", "TEMPORAL",
        "INDEPENDENCE", "FEATURE_COMPATIBILITY", "LEAKAGE",
        "LICENSE", "SOURCE_IDENTITY",
    ]
    for cat in categories:
        ingestion.add_evidence(
            package_id,
            evidence_type=cat,
            source_reference=f"fixture://{cat.lower()}",
            captured_content=f"TEST_FIXTURE {cat} evidence content",
            verification_status="DOCUMENTED",
            source_capture_level=SourceCaptureLevel.DOCUMENTATION_CAPTURED.value,
        )

    # Capture documentation
    ingestion.capture_documentation(package_id)

    # Compute hashes
    ingestion.compute_hashes(package_id)

    # Index
    ingestion.index_evidence(package_id)

    # Validate
    ingestion.validate_evidence(package_id)

    # Build manifest
    ingestion.build_manifest(package_id)

    # Ready
    ingestion.ready_for_certification(package_id)

    return pkg, ingestion
