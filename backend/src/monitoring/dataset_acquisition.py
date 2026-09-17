"""Phase 55: Dataset acquisition, artifact identity & controlled evaluation.

Extends Phase 53/54 with:

  - DatasetAcquisitionRecord (deterministic acquisition metadata)
  - ArtifactIdentity (SHA-256 integrity, checksum verification)
  - SourceEvidenceBundle (structured evidence items with hashes)
  - ControlledEvaluationPackage (evaluation ONLY after admission)
  - Acquisition workflow integration with discovery

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET

This module does NOT fabricate datasets, eligibility, or evaluation results.

STATUS: IMPLEMENTED
PHASE: 55
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


# ── Acquisition Status ────────────────────────────────────────────────────

class AcquisitionStatus(str, Enum):
    NOT_ACQUIRED = "NOT_ACQUIRED"
    ACQUIRED = "ACQUIRED"
    IDENTITY_VERIFIED = "IDENTITY_VERIFIED"
    CHECKSUM_VERIFIED = "CHECKSUM_VERIFIED"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    CHECKSUM_NOT_PROVIDED = "CHECKSUM_NOT_PROVIDED"
    EXTRACTION_COMPLETE = "EXTRACTION_COMPLETE"
    FAILED = "FAILED"


class ChecksumState(str, Enum):
    VERIFIED = "CHECKSUM_VERIFIED"
    NOT_PROVIDED = "CHECKSUM_NOT_PROVIDED"
    MISMATCH = "CHECKSUM_MISMATCH"
    UNVERIFIED = "CHECKSUM_UNVERIFIED"


class EvidenceVerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    DOCUMENTED = "DOCUMENTED"
    SOURCE_REPORTED = "SOURCE_REPORTED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CONTRADICTED = "CONTRADICTED"


# ── Artifact Identity ─────────────────────────────────────────────────────

@dataclass
class ArtifactIdentity:
    """Cryptographic identity of an acquired dataset artifact."""
    file_path: str = ""
    original_filename: str = ""
    file_size: int = 0
    sha256: str = ""
    hash_algorithm: str = "sha256"
    source_checksum: str = ""
    source_checksum_algorithm: str = ""
    checksum_state: str = ChecksumState.UNVERIFIED.value
    archive_format: str = ""  # "zip", "tar.gz", "csv", etc.
    extracted_files: dict[str, str] = field(default_factory=dict)  # name -> sha256
    extraction_method: str = ""
    extraction_version: str = ""

    def compute_hash(self, file_path: str | Path | None = None) -> str:
        """Compute SHA-256 of the artifact file."""
        path = Path(file_path) if file_path else Path(self.file_path)
        if not path.exists():
            return ""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        self.sha256 = h.hexdigest()
        self.file_size = path.stat().st_size
        return self.sha256

    def verify_checksum(self) -> tuple[bool, str]:
        """Verify artifact matches source-provided checksum."""
        if not self.source_checksum:
            self.checksum_state = ChecksumState.NOT_PROVIDED.value
            return False, "No source checksum provided"
        if not self.sha256:
            self.checksum_state = ChecksumState.UNVERIFIED.value
            return False, "No artifact hash computed"
        if self.source_checksum == self.sha256:
            self.checksum_state = ChecksumState.VERIFIED.value
            return True, "Checksum matches"
        self.checksum_state = ChecksumState.MISMATCH.value
        return False, (
            f"Checksum mismatch: source={self.source_checksum[:16]}… "
            f"artifact={self.sha256[:16]}…"
        )

    def verify_identity(self, expected_hash: str) -> tuple[bool, str]:
        """Verify artifact identity against expected hash."""
        if not self.sha256:
            return False, "No hash computed"
        if self.sha256 == expected_hash:
            return True, "Identity verified"
        return False, (
            f"Identity mismatch: expected={expected_hash[:16]}… "
            f"got={self.sha256[:16]}…"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Source Evidence Item ───────────────────────────────────────────────────

@dataclass
class SourceEvidenceItem:
    """A single piece of evidence about a dataset's provenance or properties."""
    evidence_id: str = ""
    evidence_type: str = ""  # "documentation", "checksum", "license", "provenance_statement", etc.
    source_reference: str = ""  # URL, file path, or citation
    captured_at: str = ""
    description: str = ""
    content_hash: str = ""  # SHA-256 of the evidence document/content
    verification_status: str = EvidenceVerificationStatus.UNKNOWN.value
    notes: str = ""

    def compute_content_hash(self, content: str | bytes) -> str:
        """Compute hash of evidence content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.content_hash = hashlib.sha256(content).hexdigest()
        return self.content_hash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Source Evidence Bundle ────────────────────────────────────────────────

@dataclass
class SourceEvidenceBundle:
    """Structured collection of evidence items for a dataset candidate."""
    candidate_id: str = ""
    items: list[SourceEvidenceItem] = field(default_factory=list)
    bundle_hash: str = ""

    def add_item(self, item: SourceEvidenceItem) -> None:
        """Add an evidence item to the bundle."""
        self.items.append(item)

    def compute_hash(self) -> str:
        """Compute deterministic hash of the entire evidence bundle."""
        items_data = [item.to_dict() for item in self.items]
        canonical = json.dumps(items_data, sort_keys=True, default=str)
        self.bundle_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.bundle_hash

    def get_by_type(self, evidence_type: str) -> list[SourceEvidenceItem]:
        """Get all evidence items of a given type."""
        return [item for item in self.items if item.evidence_type == evidence_type]

    def has_verified_evidence(self, evidence_type: str) -> bool:
        """Check if there is verified evidence of a given type."""
        items = self.get_by_type(evidence_type)
        return any(
            item.verification_status in (EvidenceVerificationStatus.VERIFIED.value,
                                          EvidenceVerificationStatus.DOCUMENTED.value)
            for item in items
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "items": [item.to_dict() for item in self.items],
            "bundle_hash": self.bundle_hash,
        }


# ── Dataset Acquisition Record ────────────────────────────────────────────

@dataclass
class DatasetAcquisitionRecord:
    """Deterministic record of dataset acquisition."""
    acquisition_id: str = ""
    candidate_id: str = ""
    dataset_name: str = ""
    dataset_version: str = ""
    acquisition_timestamp: str = ""
    acquisition_method: str = ""  # "download", "api", "manual", "ssh", etc.
    source_url: str = ""
    publisher: str = ""
    source_distribution: str = ""
    original_source: str = ""
    local_artifact_path: str = ""
    artifact: ArtifactIdentity = field(default_factory=ArtifactIdentity)
    evidence_bundle: SourceEvidenceBundle = field(default_factory=SourceEvidenceBundle)
    status: str = AcquisitionStatus.NOT_ACQUIRED.value
    acquisition_history: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def compute_hash(self) -> str:
        """Compute deterministic hash of the acquisition record."""
        d = asdict(self)
        # Exclude mutable audit fields
        d.pop("acquisition_history", None)
        d.pop("notes", None)
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def transition(self, new_status: AcquisitionStatus, reason: str = "") -> bool:
        """Attempt an acquisition status transition."""
        valid = {
            AcquisitionStatus.NOT_ACQUIRED: {
                AcquisitionStatus.ACQUIRED, AcquisitionStatus.FAILED,
            },
            AcquisitionStatus.ACQUIRED: {
                AcquisitionStatus.IDENTITY_VERIFIED, AcquisitionStatus.CHECKSUM_VERIFIED,
                AcquisitionStatus.CHECKSUM_MISMATCH, AcquisitionStatus.CHECKSUM_NOT_PROVIDED,
                AcquisitionStatus.EXTRACTION_COMPLETE, AcquisitionStatus.FAILED,
            },
            AcquisitionStatus.IDENTITY_VERIFIED: {
                AcquisitionStatus.CHECKSUM_VERIFIED, AcquisitionStatus.CHECKSUM_NOT_PROVIDED,
                AcquisitionStatus.CHECKSUM_MISMATCH, AcquisitionStatus.EXTRACTION_COMPLETE,
                AcquisitionStatus.FAILED,
            },
        }
        allowed = valid.get(AcquisitionStatus(self.status), set())
        if new_status not in allowed:
            return False
        self.acquisition_history.append({
            "from": self.status,
            "to": new_status.value,
            "reason": reason,
            "timestamp": time.time(),
        })
        self.status = new_status.value
        return True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetAcquisitionRecord:
        artifact_data = data.pop("artifact", None)
        evidence_data = data.pop("evidence_bundle", None)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        record = cls(**known)
        if artifact_data and isinstance(artifact_data, dict):
            record.artifact = ArtifactIdentity(**artifact_data)
        if evidence_data and isinstance(evidence_data, dict):
            items = [SourceEvidenceItem(**i) for i in evidence_data.get("items", [])]
            record.evidence_bundle = SourceEvidenceBundle(
                candidate_id=evidence_data.get("candidate_id", ""),
                items=items,
                bundle_hash=evidence_data.get("bundle_hash", ""),
            )
        return record


# ── Controlled Evaluation Package ─────────────────────────────────────────

@dataclass
class ControlledEvaluationPackage:
    """Evaluation package created ONLY after successful admission.

    Cryptographically bound to the exact admitted dataset and model.
    """
    package_id: str = ""
    candidate_id: str = ""
    acquisition_id: str = ""
    admission_package_hash: str = ""

    # Frozen dataset identity
    dataset_hash: str = ""
    dataset_schema_hash: str = ""
    row_count: int = 0

    # Frozen model identity
    model_id: str = ""
    model_version: str = ""
    artifact_set_hash: str = ""
    feature_version: str = ""
    preprocessing_hash: str = ""

    # Frozen evaluation protocol
    evaluation_protocol_version: str = "1.0"
    threshold: float = 0.5
    threshold_method: str = "fixed"
    threshold_fit_on_test: bool = False
    random_seed: int = 0

    # Evaluation results (filled after execution)
    executed: bool = False
    execution_timestamp: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: dict[str, int] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)

    # Integrity
    package_hash: str = ""
    created_at: float = field(default_factory=time.time)

    def compute_hash(self) -> str:
        """Compute deterministic hash of the evaluation package."""
        d = asdict(self)
        d.pop("package_hash", None)
        canonical = json.dumps(d, sort_keys=True, default=str)
        self.package_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.package_hash

    def verify_binding(
        self,
        dataset_hash: str,
        artifact_set_hash: str,
        feature_version: str,
    ) -> tuple[bool, str]:
        """Verify this package is bound to the exact dataset and model."""
        errors = []
        if self.dataset_hash != dataset_hash:
            errors.append(
                f"Dataset hash mismatch: package={self.dataset_hash[:16]}… "
                f"expected={dataset_hash[:16]}…"
            )
        if self.artifact_set_hash != artifact_set_hash:
            errors.append(
                f"Artifact-set hash mismatch: package={self.artifact_set_hash[:16]}… "
                f"expected={artifact_set_hash[:16]}…"
            )
        if self.feature_version != feature_version:
            errors.append(
                f"Feature version mismatch: package={self.feature_version} "
                f"expected={feature_version}"
            )
        if errors:
            return False, "; ".join(errors)
        return True, "Evaluation package bound to dataset and model"

    def verify_threshold_policy(self) -> tuple[bool, str]:
        """Verify threshold policy is sound."""
        if self.threshold_fit_on_test:
            return False, "Threshold was fit on test data — test-set leakage"
        if self.threshold_method not in ("fixed", "validation_set"):
            return False, f"Unknown threshold method: {self.threshold_method}"
        return True, f"Threshold policy valid: {self.threshold_method}={self.threshold}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Acquisition Workflow ──────────────────────────────────────────────────

class DatasetAcquisitionWorkflow:
    """Manages the dataset acquisition and evaluation workflow.

    Integrates with Phase 54's DatasetDiscoveryWorkflow to provide:
      - Deterministic artifact acquisition
      - Artifact identity verification
      - Source evidence collection
      - Admission-gated evaluation
      - Forensic integration
    """

    def __init__(self, registry_path: str | Path | None = None):
        self.acquisitions: dict[str, DatasetAcquisitionRecord] = {}
        self.evaluation_packages: dict[str, ControlledEvaluationPackage] = {}
        self._registry_path = Path(registry_path) if registry_path else None
        self._load_registry()

    def _load_registry(self) -> None:
        """Load acquisition registry from disk if available."""
        if self._registry_path and self._registry_path.exists():
            try:
                data = json.loads(self._registry_path.read_text(encoding="utf-8"))
                for aid, adata in data.get("acquisitions", {}).items():
                    self.acquisitions[aid] = DatasetAcquisitionRecord.from_dict(adata)
                for pid, pdata in data.get("evaluation_packages", {}).items():
                    self.evaluation_packages[pid] = ControlledEvaluationPackage(**pdata)
            except Exception:
                pass

    def save_registry(self) -> None:
        """Save acquisition registry to disk."""
        if self._registry_path:
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "acquisitions": {aid: a.to_dict() for aid, a in self.acquisitions.items()},
                "evaluation_packages": {pid: p.to_dict() for pid, p in self.evaluation_packages.items()},
            }
            self._registry_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )

    def record_acquisition(
        self,
        candidate_id: str,
        dataset_name: str,
        local_path: str,
        source_url: str = "",
        publisher: str = "",
        source_checksum: str = "",
        source_checksum_algorithm: str = "",
        acquisition_method: str = "download",
    ) -> DatasetAcquisitionRecord:
        """Record a dataset acquisition and verify artifact identity."""
        aid = f"acq-{candidate_id}-{hashlib.sha256(local_path.encode()).hexdigest()[:8]}"
        record = DatasetAcquisitionRecord(
            acquisition_id=aid,
            candidate_id=candidate_id,
            dataset_name=dataset_name,
            acquisition_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            acquisition_method=acquisition_method,
            source_url=source_url,
            publisher=publisher,
            local_artifact_path=local_path,
            status=AcquisitionStatus.NOT_ACQUIRED.value,
        )
        record.artifact.file_path = local_path
        record.artifact.original_filename = Path(local_path).name
        record.artifact.source_checksum = source_checksum
        record.artifact.source_checksum_algorithm = source_checksum_algorithm

        # Compute artifact hash
        path = Path(local_path)
        if path.exists():
            record.artifact.compute_hash(local_path)
            record.transition(AcquisitionStatus.ACQUIRED, f"file exists: {local_path}")

            # Verify identity
            if record.artifact.sha256:
                record.transition(AcquisitionStatus.IDENTITY_VERIFIED,
                                  f"hash={record.artifact.sha256[:16]}…")

            # Verify checksum
            if source_checksum:
                ok, msg = record.artifact.verify_checksum()
                if ok:
                    record.transition(AcquisitionStatus.CHECKSUM_VERIFIED, msg)
                else:
                    record.transition(AcquisitionStatus.CHECKSUM_MISMATCH, msg)
            else:
                record.transition(AcquisitionStatus.CHECKSUM_NOT_PROVIDED,
                                  "No source checksum to verify against")
        else:
            record.transition(AcquisitionStatus.FAILED, f"File not found: {local_path}")

        self.acquisitions[aid] = record
        return record

    def add_evidence(
        self,
        acquisition_id: str,
        evidence_type: str,
        source_reference: str,
        description: str,
        content: str | bytes | None = None,
        verification_status: str = EvidenceVerificationStatus.SOURCE_REPORTED.value,
    ) -> SourceEvidenceItem:
        """Add an evidence item to an acquisition's evidence bundle."""
        record = self.acquisitions.get(acquisition_id)
        if not record:
            raise ValueError(f"Acquisition {acquisition_id} not found")

        item = SourceEvidenceItem(
            evidence_id=f"ev-{len(record.evidence_bundle.items)}",
            evidence_type=evidence_type,
            source_reference=source_reference,
            captured_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            description=description,
            verification_status=verification_status,
        )
        if content:
            item.compute_content_hash(content)

        record.evidence_bundle.add_item(item)
        return item

    def create_evaluation_package(
        self,
        acquisition_id: str,
        admission_package_hash: str,
        dataset_hash: str,
        row_count: int,
        model_id: str,
        model_version: str,
        artifact_set_hash: str,
        feature_version: str,
        preprocessing_hash: str = "",
        threshold: float = 0.5,
        random_seed: int = 42,
    ) -> ControlledEvaluationPackage | None:
        """Create an evaluation package ONLY after successful admission.

        Returns None if the acquisition is not in a valid state.
        """
        record = self.acquisitions.get(acquisition_id)
        if not record:
            return None

        # Must have verified identity
        if record.status in (AcquisitionStatus.NOT_ACQUIRED.value,
                              AcquisitionStatus.FAILED.value):
            return None

        pid = f"eval-{acquisition_id}"
        pkg = ControlledEvaluationPackage(
            package_id=pid,
            candidate_id=record.candidate_id,
            acquisition_id=acquisition_id,
            admission_package_hash=admission_package_hash,
            dataset_hash=dataset_hash,
            row_count=row_count,
            model_id=model_id,
            model_version=model_version,
            artifact_set_hash=artifact_set_hash,
            feature_version=feature_version,
            preprocessing_hash=preprocessing_hash,
            threshold=threshold,
            random_seed=random_seed,
        )
        pkg.compute_hash()
        self.evaluation_packages[pid] = pkg
        return pkg

    def verify_evaluation_package(
        self,
        package_id: str,
        dataset_hash: str,
        artifact_set_hash: str,
        feature_version: str,
    ) -> tuple[bool, str]:
        """Verify an evaluation package is still bound to the correct artifacts."""
        pkg = self.evaluation_packages.get(package_id)
        if not pkg:
            return False, f"Package {package_id} not found"
        return pkg.verify_binding(dataset_hash, artifact_set_hash, feature_version)


# ── Integration: Extend CandidateDataset with acquisition ─────────────────

def attach_acquisition_to_candidate(
    candidate_dict: dict[str, Any],
    acquisition: DatasetAcquisitionRecord,
) -> dict[str, Any]:
    """Attach acquisition metadata to a candidate record dict."""
    candidate_dict["acquisition_id"] = acquisition.acquisition_id
    candidate_dict["artifact_hash"] = acquisition.artifact.sha256
    candidate_dict["artifact_size"] = acquisition.artifact.file_size
    candidate_dict["checksum_state"] = acquisition.artifact.checksum_state
    return candidate_dict
