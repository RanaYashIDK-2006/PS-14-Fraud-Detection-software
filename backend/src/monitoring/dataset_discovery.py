"""Phase 54: Dataset discovery, provenance verification & admission workflow.

Extends Phase 53's admission framework with:

  - CandidateDataset record with full provenance tracking
  - Provenance verification levels (SOURCE_REPORTED, DOCUMENTED, INDEPENDENTLY_VERIFIED)
  - Source chain tracking (original → release → distribution → local)
  - Fraud label verification (who, what, when, how)
  - Dataset independence verification
  - Admission package generation (cryptographically bound)
  - External candidate discovery placeholder
  - Re-evaluation of all known candidates

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET

This module does NOT fabricate datasets or eligibility.

STATUS: IMPLEMENTED
PHASE: 54
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

# ── Provenance Verification Levels ────────────────────────────────────────

class ProvenanceLevel(str, Enum):
    SOURCE_REPORTED = "SOURCE_REPORTED"
    DOCUMENTED = "DOCUMENTED"
    INDEPENDENTLY_VERIFIED = "INDEPENDENTLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    UNKNOWN = "UNKNOWN"


class CandidateStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    CANDIDATE = "CANDIDATE"
    EVIDENCE_COLLECTION = "EVIDENCE_COLLECTION"
    PROVENANCE_VERIFIED = "PROVENANCE_VERIFIED"
    SEMANTICALLY_VERIFIED = "SEMANTICALLY_VERIFIED"
    LEAKAGE_CHECKED = "LEAKAGE_CHECKED"
    FEATURE_COMPATIBLE = "FEATURE_COMPATIBLE"
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"


class LicenseClassification(str, Enum):
    PUBLICLY_ACCESSIBLE = "PUBLICLY_ACCESSIBLE"
    LICENSED_RESEARCH = "LICENSED_RESEARCH"
    LICENSED_COMMERCIAL = "LICENSED_COMMERCIAL"
    LICENSE_UNKNOWN = "LICENSE_UNKNOWN"
    RESTRICTIVE = "RESTRICTIVE"


# ── Source Chain ───────────────────────────────────────────────────────────

@dataclass
class SourceChainRecord:
    """Tracks the provenance chain from original source to local artifact."""
    original_source: str = ""
    original_publisher: str = ""
    original_publication_date: str = ""
    dataset_release_identifier: str = ""
    distribution_channel: str = ""
    distribution_url: str = ""
    local_path: str = ""
    local_file_hash: str = ""
    local_file_size: int = 0
    hash_algorithm: str = "sha256"
    source_checksum: str = ""
    source_checksum_algorithm: str = ""
    checksum_verified: bool = False
    transformations: list[str] = field(default_factory=list)

    def verify_checksum(self, local_hash: str) -> tuple[bool, str]:
        """Verify local artifact matches source-provided checksum."""
        if not self.source_checksum:
            return False, "No source checksum provided — cannot verify"
        if not self.local_file_hash:
            return False, "No local file hash computed"
        if self.source_checksum == local_hash:
            self.checksum_verified = True
            return True, "Checksum matches"
        return False, (
            f"Checksum mismatch: source={self.source_checksum[:16]}… "
            f"local={local_hash[:16]}…"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Fraud Label Evidence ───────────────────────────────────────────────────

@dataclass
class FraudLabelEvidence:
    """Structured evidence about fraud label semantics."""
    label_definition: str = ""
    label_granularity: str = ""  # "transaction_level", "account_level", "batch_level"
    label_author: str = ""  # "human_investigator", "automated_system", "unknown"
    label_timing: str = ""  # "at_event", "post_investigation", "retrospective", "unknown"
    label_timing_known: bool = False
    label_latency_hours: float | None = None  # hours between event and label
    positive_class_definition: str = ""
    negative_class_definition: str = ""
    negative_class_verified: bool = False  # are negatives verified or just "not fraud"?
    ambiguous_records_present: bool = False
    unknown_labels_present: bool = False
    label_source_documentation: str = ""
    evidence_references: list[str] = field(default_factory=list)

    def is_complete(self) -> tuple[bool, list[str]]:
        """Check if all mandatory label evidence is provided."""
        gaps = []
        if not self.label_definition:
            gaps.append("No label definition")
        if not self.label_granularity:
            gaps.append("Label granularity unknown")
        if not self.label_author:
            gaps.append("Label author unknown — who assigned the label?")
        if self.label_timing in ("", "unknown"):
            gaps.append("Label timing unknown")
        if not self.label_timing_known:
            gaps.append("Label timing not confirmed as known")
        if not self.positive_class_definition:
            gaps.append("Positive class definition missing")
        if not self.negative_class_definition:
            gaps.append("Negative class definition missing")
        return len(gaps) == 0, gaps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Independence Evidence ─────────────────────────────────────────────────

@dataclass
class IndependenceEvidence:
    """Evidence that a dataset is independent from PS-14 training data."""
    source_independent: bool = False
    records_independent: bool = False
    no_shared_identifiers: bool = False
    no_shared_transaction_ids: bool = False
    no_feature_vector_overlap: bool = False
    no_temporal_overlap: bool = False
    no_augmented_copy: bool = False
    no_synthetic_generation: bool = False
    evidence_references: list[str] = field(default_factory=list)
    notes: str = ""

    def is_complete(self) -> tuple[bool, list[str]]:
        """Check if independence evidence is sufficient."""
        gaps = []
        if not self.source_independent:
            gaps.append("Source independence not established")
        if not self.records_independent:
            gaps.append("Record independence not established")
        if not self.no_shared_identifiers:
            gaps.append("Shared identifiers not checked")
        if not self.no_synthetic_generation:
            gaps.append("Synthetic generation not ruled out")
        return len(gaps) == 0, gaps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Candidate Dataset Record ──────────────────────────────────────────────

@dataclass
class CandidateDataset:
    """Complete candidate dataset record with provenance tracking.

    Every field must be explicitly populated or left at its default.
    Unknown/guessed values must NOT be substituted.
    """
    # Identity
    candidate_id: str = ""
    dataset_name: str = ""
    dataset_version: str = ""
    discovery_date: str = ""
    discovered_by: str = "system"  # "system", "operator", "automation"

    # Source
    source_name: str = ""
    source_reference: str = ""  # URL, DOI, or citation
    source_type: str = ""  # "public_research", "proprietary", "academic", "government"
    source_claims: str = ""  # what the source claims about the dataset

    # Classification
    source_classification: str = "UNKNOWN"
    license_classification: str = LicenseClassification.LICENSE_UNKNOWN.value
    license_details: str = ""

    # Provenance
    provenance_level: str = ProvenanceLevel.UNKNOWN.value
    provenance_evidence: list[str] = field(default_factory=list)
    independent_evidence: list[str] = field(default_factory=list)

    # Source chain
    source_chain: SourceChainRecord = field(default_factory=SourceChainRecord)

    # Cryptographic identity
    dataset_hash: str = ""
    file_hashes: dict[str, str] = field(default_factory=dict)
    row_count: int = 0
    column_count: int = 0

    # Schema
    feature_schema: list[str] = field(default_factory=list)
    label_column: str = ""

    # Fraud label evidence
    label_evidence: FraudLabelEvidence = field(default_factory=FraudLabelEvidence)

    # Independence
    independence_evidence: IndependenceEvidence = field(default_factory=IndependenceEvidence)

    # Known transformations
    known_preprocessing: str = ""
    known_transformations: list[str] = field(default_factory=list)
    known_sampling: str = ""

    # Feature compatibility
    feature_mapping_version: str = ""
    incompatible_features: list[str] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)

    # Admission
    status: str = CandidateStatus.DISCOVERED.value
    admission_history: list[dict[str, Any]] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    admission_timestamp: float = 0.0

    # Temporal
    collection_period_start: str = ""
    collection_period_end: str = ""
    geography: str = ""
    transaction_channel: str = ""

    def compute_hash(self) -> dict[str, str]:
        """Compute deterministic hashes for the candidate record."""
        d = asdict(self)
        # Exclude mutable audit fields
        for k in ("status", "admission_history", "rejection_reasons", "admission_timestamp"):
            d.pop(k, None)
        canonical = json.dumps(d, sort_keys=True, default=str)
        record_hash = hashlib.sha256(canonical.encode()).hexdigest()

        # Also compute file hash if source_chain has local_path
        file_hash = ""
        if self.source_chain.local_path and Path(self.source_chain.local_path).exists():
            h = hashlib.sha256()
            with open(self.source_chain.local_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            file_hash = h.hexdigest()

        return {"record_hash": record_hash, "file_hash": file_hash}

    def transition(self, new_status: CandidateStatus, reason: str = "") -> bool:
        """Attempt a candidate status transition."""
        valid = {
            CandidateStatus.DISCOVERED: {CandidateStatus.CANDIDATE, CandidateStatus.REJECTED, CandidateStatus.BLOCKED, CandidateStatus.INELIGIBLE},
            CandidateStatus.CANDIDATE: {CandidateStatus.EVIDENCE_COLLECTION, CandidateStatus.REJECTED},
            CandidateStatus.EVIDENCE_COLLECTION: {CandidateStatus.PROVENANCE_VERIFIED, CandidateStatus.INELIGIBLE, CandidateStatus.REJECTED},
            CandidateStatus.PROVENANCE_VERIFIED: {CandidateStatus.SEMANTICALLY_VERIFIED, CandidateStatus.INELIGIBLE},
            CandidateStatus.SEMANTICALLY_VERIFIED: {CandidateStatus.LEAKAGE_CHECKED, CandidateStatus.INELIGIBLE},
            CandidateStatus.LEAKAGE_CHECKED: {CandidateStatus.FEATURE_COMPATIBLE, CandidateStatus.INELIGIBLE},
            CandidateStatus.FEATURE_COMPATIBLE: {CandidateStatus.ELIGIBLE, CandidateStatus.INELIGIBLE},
        }
        allowed = valid.get(CandidateStatus(self.status), set())
        if new_status not in allowed:
            return False
        self.admission_history.append({
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
    def from_dict(cls, data: dict[str, Any]) -> CandidateDataset:
        # Handle nested dataclasses
        sc_data = data.pop("source_chain", None)
        le_data = data.pop("label_evidence", None)
        ie_data = data.pop("independence_evidence", None)

        # Filter to known fields
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}

        candidate = cls(**known)
        if sc_data and isinstance(sc_data, dict):
            candidate.source_chain = SourceChainRecord(**sc_data)
        if le_data and isinstance(le_data, dict):
            candidate.label_evidence = FraudLabelEvidence(**le_data)
        if ie_data and isinstance(ie_data, dict):
            candidate.independence_evidence = IndependenceEvidence(**ie_data)
        return candidate


# ── Admission Package ─────────────────────────────────────────────────────

@dataclass
class AdmissionPackage:
    """Cryptographically bound admission package for an eligible candidate."""
    package_id: str = ""
    candidate_id: str = ""
    dataset_name: str = ""
    dataset_hash: str = ""
    file_hashes: dict[str, str] = field(default_factory=dict)
    provenance_level: str = ""
    provenance_evidence: list[str] = field(default_factory=list)
    label_evidence_hash: str = ""
    independence_evidence_hash: str = ""
    feature_mapping_version: str = ""
    admission_decision: str = ""
    blocking_reasons: list[str] = field(default_factory=list)
    approval_reasons: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    package_hash: str = ""

    def compute_hash(self) -> str:
        """Compute deterministic hash of the admission package."""
        d = asdict(self)
        d.pop("package_hash", None)
        canonical = json.dumps(d, sort_keys=True, default=str)
        self.package_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.package_hash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Discovery Workflow ────────────────────────────────────────────────────

class DatasetDiscoveryWorkflow:
    """Manages the dataset discovery and admission workflow.

    Provides:
      - Candidate registration
      - Evidence collection
      - Provenance verification
      - Admission gate execution
      - Admission package generation
      - Candidate re-evaluation
    """

    def __init__(self, registry_path: str | Path | None = None):
        self.candidates: dict[str, CandidateDataset] = {}
        self.packages: dict[str, AdmissionPackage] = {}
        self._registry_path = Path(registry_path) if registry_path else None
        self._load_registry()

    def _load_registry(self) -> None:
        """Load candidate registry from disk if available."""
        if self._registry_path and self._registry_path.exists():
            try:
                data = json.loads(self._registry_path.read_text(encoding="utf-8"))
                for cid, cdata in data.get("candidates", {}).items():
                    self.candidates[cid] = CandidateDataset.from_dict(cdata)
                for pid, pdata in data.get("packages", {}).items():
                    self.packages[pid] = AdmissionPackage(**pdata)
            except Exception:
                pass

    def save_registry(self) -> None:
        """Save candidate registry to disk."""
        if self._registry_path:
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "candidates": {cid: c.to_dict() for cid, c in self.candidates.items()},
                "packages": {pid: p.to_dict() for pid, p in self.packages.items()},
            }
            self._registry_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )

    def register_candidate(self, candidate: CandidateDataset) -> CandidateDataset:
        """Register a new candidate dataset."""
        if not candidate.candidate_id:
            candidate.candidate_id = f"candidate-{hashlib.sha256(candidate.dataset_name.encode()).hexdigest()[:12]}"
        candidate.status = CandidateStatus.DISCOVERED.value
        candidate.discovery_date = candidate.discovery_date or time.strftime("%Y-%m-%d")
        self.candidates[candidate.candidate_id] = candidate
        return candidate

    def collect_evidence(self, candidate_id: str) -> tuple[bool, list[str]]:
        """Move candidate to EVIDENCE_COLLECTION and verify evidence completeness."""
        c = self.candidates.get(candidate_id)
        if not c:
            return False, ["Candidate not found"]

        # Check label evidence
        label_complete, label_gaps = c.label_evidence.is_complete()
        # Check independence evidence
        indep_complete, indep_gaps = c.independence_evidence.is_complete()

        all_gaps = label_gaps + indep_gaps
        if not c.source_chain.original_source:
            all_gaps.append("No original source identified")
        if not c.feature_schema:
            all_gaps.append("No feature schema defined")

        if all_gaps:
            c.rejection_reasons = all_gaps
            return False, all_gaps

        c.transition(CandidateStatus.CANDIDATE, "evidence sufficient")
        c.transition(CandidateStatus.EVIDENCE_COLLECTION, "evidence collected")
        return True, []

    def verify_provenance(self, candidate_id: str) -> tuple[bool, str]:
        """Verify provenance of a candidate."""
        c = self.candidates.get(candidate_id)
        if not c:
            return False, "Candidate not found"

        if c.provenance_level == ProvenanceLevel.INDEPENDENTLY_VERIFIED.value:
            c.transition(CandidateStatus.PROVENANCE_VERIFIED, "provenance independently verified")
            return True, "Provenance independently verified"
        if c.provenance_level == ProvenanceLevel.DOCUMENTED.value:
            c.transition(CandidateStatus.PROVENANCE_VERIFIED, "provenance documented")
            return True, "Provenance documented"
        if c.provenance_level == ProvenanceLevel.SOURCE_REPORTED.value:
            c.rejection_reasons.append("Provenance only source-reported — needs independent verification")
            return False, "Provenance only source-reported"
        c.rejection_reasons.append(f"Provenance level: {c.provenance_level}")
        return False, f"Provenance level: {c.provenance_level}"

    def verify_semantics(self, candidate_id: str) -> tuple[bool, list[str]]:
        """Verify label and temporal semantics."""
        c = self.candidates.get(candidate_id)
        if not c:
            return False, ["Candidate not found"]

        gaps = []
        if not c.label_evidence.label_timing_known:
            gaps.append("Label timing unknown")
        if c.label_evidence.label_timing in ("", "unknown"):
            gaps.append("Label timing not specified")
        if not c.collection_period_start or not c.collection_period_end:
            gaps.append("Collection period unknown")

        if gaps:
            c.rejection_reasons.extend(gaps)
            return False, gaps

        c.transition(CandidateStatus.SEMANTICALLY_VERIFIED, "semantics verified")
        return True, []

    def check_leakage(self, candidate_id: str) -> tuple[bool, list[str]]:
        """Run leakage checks on a candidate."""
        c = self.candidates.get(candidate_id)
        if not c:
            return False, ["Candidate not found"]

        gaps = []
        if c.independence_evidence.no_shared_transaction_ids is False:
            gaps.append("Shared transaction IDs not checked")
        if c.independence_evidence.no_feature_vector_overlap is False:
            gaps.append("Feature vector overlap not checked")

        if gaps:
            c.rejection_reasons.extend(gaps)
            return False, gaps

        c.transition(CandidateStatus.LEAKAGE_CHECKED, "leakage checks passed")
        return True, []

    def check_feature_compatibility(self, candidate_id: str) -> tuple[bool, list[str]]:
        """Check feature compatibility with PS-14."""
        c = self.candidates.get(candidate_id)
        if not c:
            return False, ["Candidate not found"]

        gaps = []
        if c.missing_features:
            gaps.append(f"Missing features: {c.missing_features}")
        if c.incompatible_features:
            gaps.append(f"Incompatible features: {c.incompatible_features}")
        if not c.feature_mapping_version:
            gaps.append("No feature mapping version")

        if gaps:
            c.rejection_reasons.extend(gaps)
            return False, gaps

        c.transition(CandidateStatus.FEATURE_COMPATIBLE, "features compatible")
        return True, []

    def admit_candidate(self, candidate_id: str) -> tuple[bool, AdmissionPackage | None]:
        """Run the complete admission workflow for a candidate.

        Returns (is_eligible, admission_package_or_none).
        """
        c = self.candidates.get(candidate_id)
        if not c:
            return False, None

        # Step 0: Check source classification FIRST (fast fail for SYNTHETIC/DERIVED)
        sc = c.source_classification
        if sc in ("SYNTHETIC", "SUSPECTED_SYNTHETIC", "CONFIRMED_SYNTHETIC"):
            c.rejection_reasons.append(f"Source is {sc} — not eligible")
            c.transition(CandidateStatus.BLOCKED, f"source is {sc}")
            return False, None
        if sc == "DERIVED":
            c.rejection_reasons.append("Source is DERIVED — not independent")
            c.transition(CandidateStatus.BLOCKED, "source is DERIVED")
            return False, None
        if sc in ("UNKNOWN", ""):
            c.rejection_reasons.append("Source classification UNKNOWN")
            c.transition(CandidateStatus.BLOCKED, "source is UNKNOWN")
            return False, None

        # Step 1: Collect evidence
        c.transition(CandidateStatus.CANDIDATE, "source classification acceptable")
        ok, gaps = self.collect_evidence(candidate_id)
        if not ok:
            c.transition(CandidateStatus.INELIGIBLE, "; ".join(gaps))
            return False, None

        # Step 2: Verify provenance
        ok, msg = self.verify_provenance(candidate_id)
        if not ok:
            c.transition(CandidateStatus.INELIGIBLE, msg)
            return False, None

        # Step 3: Verify semantics
        ok, gaps = self.verify_semantics(candidate_id)
        if not ok:
            c.transition(CandidateStatus.INELIGIBLE, "; ".join(gaps))
            return False, None

        # Step 4: Check leakage
        ok, gaps = self.check_leakage(candidate_id)
        if not ok:
            c.transition(CandidateStatus.INELIGIBLE, "; ".join(gaps))
            return False, None

        # Step 5: Check feature compatibility
        ok, gaps = self.check_feature_compatibility(candidate_id)
        if not ok:
            c.transition(CandidateStatus.INELIGIBLE, "; ".join(gaps))
            return False, None

        # Step 6: Mark as eligible
        c.transition(CandidateStatus.ELIGIBLE, "all gates passed")
        c.admission_timestamp = time.time()

        # Generate admission package
        hashes = c.compute_hash()
        pkg = AdmissionPackage(
            package_id=f"pkg-{candidate_id}",
            candidate_id=candidate_id,
            dataset_name=c.dataset_name,
            dataset_hash=c.dataset_hash or hashes.get("file_hash", ""),
            file_hashes=c.file_hashes,
            provenance_level=c.provenance_level,
            provenance_evidence=c.provenance_evidence,
            label_evidence_hash=hashlib.sha256(
                json.dumps(c.label_evidence.to_dict(), sort_keys=True, default=str).encode()
            ).hexdigest(),
            independence_evidence_hash=hashlib.sha256(
                json.dumps(c.independence_evidence.to_dict(), sort_keys=True, default=str).encode()
            ).hexdigest(),
            feature_mapping_version=c.feature_mapping_version,
            admission_decision="ELIGIBLE",
            approval_reasons=["All admission gates passed"],
        )
        pkg.compute_hash()
        self.packages[pkg.package_id] = pkg
        return True, pkg


# ── Existing Candidate Re-evaluation ──────────────────────────────────────

def reevaluate_known_candidates() -> list[dict[str, Any]]:
    """Re-evaluate all known candidates through the Phase 54 workflow.

    Returns structured results for each candidate.
    """
    from src.monitoring.dataset_evidence import KNOWN_CANDIDATES

    results = []
    for c in KNOWN_CANDIDATES:
        candidate = CandidateDataset(
            candidate_id=f"reeval-{c['name'].lower().replace(' ', '-')[:30]}",
            dataset_name=c["name"],
            source_classification=c["classification"],
        )
        workflow = DatasetDiscoveryWorkflow()
        workflow.register_candidate(candidate)

        # Attempt admission
        eligible, pkg = workflow.admit_candidate(candidate.candidate_id)

        results.append({
            "name": c["name"],
            "source_classification": c["classification"],
            "admission_status": candidate.status,
            "eligible": eligible,
            "blocking_reasons": candidate.rejection_reasons,
            "original_reason": c["reason"],
            "evidence_gaps": c["evidence_gaps"],
        })

    return results


# ── External Discovery Placeholder ────────────────────────────────────────

def discover_external_candidates() -> list[dict[str, Any]]:
    """Discover potential fraud datasets from external sources.

    Returns list of discovered candidates with evidence gaps.
    If external search is unavailable, returns empty list with marker.
    """
    # This is a placeholder for external discovery.
    # In a real deployment, this would search academic repositories,
    # government data portals, and industry data sharing initiatives.
    # For now, we return discovered candidates from existing knowledge.

    candidates = [
        {
            "name": "IEEE-CIS Fraud Detection",
            "source": "https://www.kaggle.com/competitions/ieee-fraud-detection",
            "source_type": "competition_dataset",
            "claims": "Real e-commerce transactions from Vesta Corporation",
            "provenance_level": "SOURCE_REPORTED",
            "known_issues": [
                "PCA-anonymized features — original feature meanings unknown",
                "Label timing not documented",
                "Transaction channel limited to e-commerce",
                "Competition context may introduce selection bias",
            ],
            "evidence_gaps": [
                "independent_provenance_verification",
                "label_timing_documentation",
                "feature_semantics_documentation",
            ],
        },
        {
            "name": "Synthetic Financial Datasets (SD)",
            "source": "https://www.kaggle.com/datasets/ntnu-testimon/paysim1",
            "source_type": "simulator",
            "claims": "PaySim mobile money simulator output",
            "provenance_level": "DOCUMENTED",
            "known_issues": [
                "Simulated data — not real-world fraud",
                "Generator rules may not reflect real fraud patterns",
            ],
            "evidence_gaps": ["synthetic_by_design"],
        },
        {
            "name": "Credit Card Fraud Detection (UCI)",
            "source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            "source_type": "academic",
            "claims": "European cardholder transactions, 2 days in Sept 2013",
            "provenance_level": "SOURCE_REPORTED",
            "known_issues": [
                "PCA-transformed — original features unknown",
                "Label timing not documented",
                "Only 2 days of data",
                "No transaction channel metadata",
            ],
            "evidence_gaps": [
                "independent_provenance_verification",
                "label_timing_documentation",
                "original_feature_documentation",
            ],
        },
    ]

    return candidates
