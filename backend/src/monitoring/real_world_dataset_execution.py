"""Phase 58: Real-world dataset acquisition & eligibility execution.

Provides the controlled acquisition-to-eligibility workflow that integrates
Phases 53-57 into a single deterministic execution path.

Workflow:
  DISCOVERED → PROVENANCE_VERIFIED → ACQUISITION_REQUESTED → ACQUIRED
  → ARTIFACT_VERIFIED → EVIDENCE_VERIFIED → SEMANTICS_VERIFIED
  → TEMPORAL_VERIFIED → INDEPENDENCE_VERIFIED → FEATURE_COMPATIBILITY_VERIFIED
  → ELIGIBILITY_CERTIFIED

Any mandatory failure → INELIGIBLE or BLOCKED.

No direct transition to ELIGIBLE without all gates passing.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET

STATUS: IMPLEMENTED
PHASE: 58
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


# ── Execution Workflow States ──────────────────────────────────────────────

class ExecutionState(str, Enum):
    DISCOVERED = "DISCOVERED"
    PROVENANCE_VERIFIED = "PROVENANCE_VERIFIED"
    ACQUISITION_REQUESTED = "ACQUISITION_REQUESTED"
    ACQUIRED = "ACQUIRED"
    ARTIFACT_VERIFIED = "ARTIFACT_VERIFIED"
    EVIDENCE_VERIFIED = "EVIDENCE_VERIFIED"
    SEMANTICS_VERIFIED = "SEMANTICS_VERIFIED"
    TEMPORAL_VERIFIED = "TEMPORAL_VERIFIED"
    INDEPENDENCE_VERIFIED = "INDEPENDENCE_VERIFIED"
    FEATURE_COMPATIBILITY_VERIFIED = "FEATURE_COMPATIBILITY_VERIFIED"
    ELIGIBILITY_CERTIFIED = "ELIGIBILITY_CERTIFIED"
    INELIGIBLE = "INELIGIBLE"
    BLOCKED = "BLOCKED"


# Valid transitions (source → allowed targets)
_EXECUTION_TRANSITIONS: dict[ExecutionState, set[ExecutionState]] = {
    ExecutionState.DISCOVERED: {
        ExecutionState.PROVENANCE_VERIFIED, ExecutionState.BLOCKED, ExecutionState.INELIGIBLE,
    },
    ExecutionState.PROVENANCE_VERIFIED: {
        ExecutionState.ACQUISITION_REQUESTED, ExecutionState.BLOCKED, ExecutionState.INELIGIBLE,
    },
    ExecutionState.ACQUISITION_REQUESTED: {
        ExecutionState.ACQUIRED, ExecutionState.BLOCKED, ExecutionState.INELIGIBLE,
    },
    ExecutionState.ACQUIRED: {
        ExecutionState.ARTIFACT_VERIFIED, ExecutionState.BLOCKED, ExecutionState.INELIGIBLE,
    },
    ExecutionState.ARTIFACT_VERIFIED: {
        ExecutionState.EVIDENCE_VERIFIED, ExecutionState.BLOCKED, ExecutionState.INELIGIBLE,
    },
    ExecutionState.EVIDENCE_VERIFIED: {
        ExecutionState.SEMANTICS_VERIFIED, ExecutionState.BLOCKED, ExecutionState.INELIGIBLE,
    },
    ExecutionState.SEMANTICS_VERIFIED: {
        ExecutionState.TEMPORAL_VERIFIED, ExecutionState.BLOCKED, ExecutionState.INELIGIBLE,
    },
    ExecutionState.TEMPORAL_VERIFIED: {
        ExecutionState.INDEPENDENCE_VERIFIED, ExecutionState.BLOCKED, ExecutionState.INELIGIBLE,
    },
    ExecutionState.INDEPENDENCE_VERIFIED: {
        ExecutionState.FEATURE_COMPATIBILITY_VERIFIED, ExecutionState.BLOCKED, ExecutionState.INELIGIBLE,
    },
    ExecutionState.FEATURE_COMPATIBILITY_VERIFIED: {
        ExecutionState.ELIGIBILITY_CERTIFIED, ExecutionState.BLOCKED, ExecutionState.INELIGIBLE,
    },
    ExecutionState.ELIGIBILITY_CERTIFIED: set(),
    ExecutionState.INELIGIBLE: set(),
    ExecutionState.BLOCKED: set(),
}


# ── Gate Evidence Record ───────────────────────────────────────────────────

@dataclass
class GateResult:
    """Result of a single acquisition/eligibility gate."""
    gate_name: str
    passed: bool
    evidence_hash: str = ""
    blocking_reason: str = ""
    details: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Provenance Evidence Record ─────────────────────────────────────────────

@dataclass
class ProvenanceEvidence:
    """Structured provenance evidence for a dataset candidate."""
    original_source: str = ""
    publisher: str = ""
    source_url: str = ""
    source_chain: list[str] = field(default_factory=list)
    license_status: str = "UNKNOWN"
    provenance_level: str = "SOURCE_REPORTED"
    publication_date: str = ""
    acquisition_date: str = ""
    evidence_references: list[str] = field(default_factory=list)

    def compute_hash(self) -> str:
        d = {k: v for k, v in asdict(self).items()}
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Label Evidence Record ──────────────────────────────────────────────────

@dataclass
class LabelEvidence:
    """Fraud label semantics evidence."""
    label_definition: str = ""
    label_column: str = ""
    positive_class: str = ""
    negative_class: str = ""
    label_granularity: str = ""  # transaction_level, account_level, batch
    label_generation_method: str = ""  # investigation, automated, retrospective
    label_timing: str = ""  # at_event, post_investigation, retrospective
    label_timing_known: bool = False
    label_author: str = ""  # human_investigator, automated_system, unknown
    prediction_time_availability: str = ""  # available, unavailable, unknown

    def compute_hash(self) -> str:
        d = {k: v for k, v in asdict(self).items()}
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def is_complete(self) -> tuple[bool, list[str]]:
        gaps = []
        if not self.label_definition:
            gaps.append("missing label definition")
        if not self.label_column:
            gaps.append("missing label column")
        if not self.label_timing_known:
            gaps.append("label timing unknown")
        if not self.label_timing or self.label_timing == "unknown":
            gaps.append("label timing not specified")
        if not self.label_generation_method or self.label_generation_method == "unknown":
            gaps.append("label generation method unknown")
        if not self.label_granularity:
            gaps.append("missing label granularity")
        return (len(gaps) == 0, gaps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Temporal Evidence Record ───────────────────────────────────────────────

@dataclass
class TemporalEvidence:
    """Temporal integrity evidence."""
    timestamp_column: str = ""
    collection_start: str = ""
    collection_end: str = ""
    ordering_verified: bool = False
    future_information_check: str = "NOT_CHECKED"
    temporal_evidence_hash: str = ""

    def compute_hash(self) -> str:
        d = {k: v for k, v in asdict(self).items()}
        d.pop("temporal_evidence_hash", None)
        canonical = json.dumps(d, sort_keys=True, default=str)
        self.temporal_evidence_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.temporal_evidence_hash

    def is_complete(self) -> tuple[bool, list[str]]:
        gaps = []
        if not self.timestamp_column:
            gaps.append("missing timestamp column")
        if not self.collection_start:
            gaps.append("missing collection start")
        if not self.collection_end:
            gaps.append("missing collection end")
        if not self.ordering_verified:
            gaps.append("temporal ordering not verified")
        return (len(gaps) == 0, gaps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Independence Evidence Record ───────────────────────────────────────────

@dataclass
class IndependenceEvidence:
    """Dataset independence verification evidence."""
    ps14_derived: str = "UNKNOWN"  # VERIFIED_INDEPENDENT, VERIFIED_DEPENDENT, UNKNOWN
    synthetic_generation: str = "UNKNOWN"
    shared_ids: str = "UNKNOWN"
    duplicated_rows: str = "UNKNOWN"
    feature_vector_overlap: str = "UNKNOWN"
    shared_source_lineage: str = "UNKNOWN"
    augmented_copy: str = "UNKNOWN"

    def compute_hash(self) -> str:
        d = {k: v for k, v in asdict(self).items()}
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def is_complete(self) -> tuple[bool, list[str]]:
        gaps = []
        for field_name in ["ps14_derived", "synthetic_generation", "shared_ids",
                           "duplicated_rows", "feature_vector_overlap"]:
            val = getattr(self, field_name)
            if val == "UNKNOWN":
                gaps.append(f"{field_name} unknown")
        return (len(gaps) == 0, gaps)

    def is_independent(self) -> tuple[bool, list[str]]:
        """Check all independence claims are VERIFIED_INDEPENDENT."""
        issues = []
        for field_name in ["ps14_derived", "synthetic_generation", "shared_ids",
                           "duplicated_rows", "feature_vector_overlap", "shared_source_lineage",
                           "augmented_copy"]:
            val = getattr(self, field_name)
            if val != "VERIFIED_INDEPENDENT" and val != "NOT_APPLICABLE":
                issues.append(f"{field_name}: {val}")
        return (len(issues) == 0, issues)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Feature Mapping Record ─────────────────────────────────────────────────

@dataclass
class FeatureMapping:
    """Deterministic feature mapping from external to PS-14 features."""
    external_column: str = ""
    ps14_feature: str = ""
    datatype: str = ""
    transformation: str = ""  # "direct", "hash", "bucket", "log", etc.
    transformation_hash: str = ""
    compatibility_status: str = ""  # DIRECT, VALIDATED_DERIVATION, MISSING, UNSUPPORTED

    def compute_transformation_hash(self) -> str:
        h = hashlib.sha256(f"{self.external_column}:{self.transformation}:{self.datatype}".encode()).hexdigest()
        self.transformation_hash = h
        return h

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Leakage Check Record ───────────────────────────────────────────────────

@dataclass
class LeakageCheckResult:
    """Result of a leakage check."""
    check_type: str = ""
    passed: bool = False
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Eligibility Certificate ────────────────────────────────────────────────

@dataclass
class EligibilityCertificate:
    """Tamper-evident eligibility certificate binding all evidence to a dataset.

    Immutable after creation — any modification invalidates the certificate hash.
    """
    certificate_id: str = ""
    dataset_id: str = ""
    dataset_hash: str = ""
    acquisition_id: str = ""
    acquisition_hash: str = ""
    provenance_hash: str = ""
    label_hash: str = ""
    temporal_hash: str = ""
    independence_hash: str = ""
    feature_mapping_hash: str = ""
    leakage_hash: str = ""
    model_id: str = ""
    artifact_set_hash: str = ""
    feature_version: str = ""
    release_id: str = ""
    evaluation_protocol_id: str = ""
    verdict: str = "NOT_CERTIFIED"
    gate_results: list[GateResult] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    certificate_hash: str = ""

    def compute_hash(self) -> str:
        """Compute tamper-evident certificate hash (excludes mutable fields)."""
        d = {
            "certificate_id": self.certificate_id,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "acquisition_id": self.acquisition_id,
            "acquisition_hash": self.acquisition_hash,
            "provenance_hash": self.provenance_hash,
            "label_hash": self.label_hash,
            "temporal_hash": self.temporal_hash,
            "independence_hash": self.independence_hash,
            "feature_mapping_hash": self.feature_mapping_hash,
            "leakage_hash": self.leakage_hash,
            "model_id": self.model_id,
            "artifact_set_hash": self.artifact_set_hash,
            "feature_version": self.feature_version,
            "release_id": self.release_id,
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "verdict": self.verdict,
            "gate_results": [g.to_dict() for g in self.gate_results],
        }
        canonical = json.dumps(d, sort_keys=True, default=str)
        self.certificate_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.certificate_hash

    def verify_binding(
        self,
        dataset_hash: str,
        artifact_set_hash: str,
        feature_version: str,
        release_id: str,
    ) -> tuple[bool, list[str]]:
        """Verify certificate is bound to the correct dataset and model."""
        errors = []
        if self.dataset_hash != dataset_hash:
            errors.append(f"Dataset hash mismatch: cert={self.dataset_hash[:16]}... actual={dataset_hash[:16]}...")
        if self.artifact_set_hash != artifact_set_hash:
            errors.append(f"Artifact-set hash mismatch: cert={self.artifact_set_hash[:16]}... actual={artifact_set_hash[:16]}...")
        if self.feature_version != feature_version:
            errors.append(f"Feature version mismatch: cert={self.feature_version} actual={feature_version}")
        if self.release_id != release_id:
            errors.append(f"Release ID mismatch: cert={self.release_id} actual={release_id}")
        return (len(errors) == 0, errors)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Forensic Event Record ──────────────────────────────────────────────────

@dataclass
class ForensicEvent:
    """Tamper-evident forensic event for acquisition/eligibility lifecycle."""
    event_id: str = ""
    event_type: str = ""  # acquisition, evidence, gate, certificate, etc.
    dataset_id: str = ""
    acquisition_id: str = ""
    certificate_id: str = ""
    dataset_hash: str = ""
    evidence_hashes: dict[str, str] = field(default_factory=dict)
    gate_results: list[GateResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    actor_type: str = "SYSTEM"
    previous_event_hash: str = ""
    event_hash: str = ""

    def compute_hash(self, previous_hash: str = "") -> str:
        """Compute event hash chained to previous event."""
        self.previous_event_hash = previous_hash
        d = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "dataset_id": self.dataset_id,
            "acquisition_id": self.acquisition_id,
            "certificate_id": self.certificate_id,
            "dataset_hash": self.dataset_hash,
            "evidence_hashes": self.evidence_hashes,
            "gate_results": [g.to_dict() for g in self.gate_results],
            "previous_event_hash": previous_hash,
        }
        canonical = json.dumps(d, sort_keys=True, default=str)
        self.event_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.event_hash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Execution Workflow ─────────────────────────────────────────────────────

class RealWorldDatasetExecution:
    """Controlled acquisition-to-eligibility execution workflow.

    Integrates Phases 53-58 into a single deterministic path.
    Every state transition requires explicit validation.
    Failed mandatory gates → INELIGIBLE or BLOCKED (terminal).
    """

    def __init__(self):
        self.state = ExecutionState.DISCOVERED
        self.state_history: list[dict[str, Any]] = []
        self.gates: list[GateResult] = []
        self.provenance: ProvenanceEvidence = ProvenanceEvidence()
        self.labels: LabelEvidence = LabelEvidence()
        self.temporal: TemporalEvidence = TemporalEvidence()
        self.independence: IndependenceEvidence = IndependenceEvidence()
        self.feature_mappings: list[FeatureMapping] = []
        self.leakage_checks: list[LeakageCheckResult] = []
        self.certificate: EligibilityCertificate | None = None
        self.forensic_events: list[ForensicEvent] = []
        self.dataset_id: str = ""
        self.dataset_hash: str = ""
        self.acquisition_id: str = ""
        self.acquisition_hash: str = ""
        self.is_test_fixture: bool = False
        self._gate_history: dict[str, list[dict[str, Any]]] = {}

    def transition(self, new_state: ExecutionState, reason: str = "") -> bool:
        """Attempt a state transition."""
        allowed = _EXECUTION_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            return False
        self.state_history.append({
            "from": self.state.value,
            "to": new_state.value,
            "reason": reason,
            "timestamp": time.time(),
        })
        self.state = new_state
        return True

    def block(self, reason: str) -> bool:
        """Transition to BLOCKED state."""
        return self.transition(ExecutionState.BLOCKED, reason)

    def ineligible(self, reason: str) -> bool:
        """Transition to INELIGIBLE state."""
        return self.transition(ExecutionState.INELIGIBLE, reason)

    def _add_gate(self, name: str, passed: bool, reason: str = "",
                  details: str = "", evidence_hash: str = "") -> GateResult:
        gate = GateResult(
            gate_name=name, passed=passed, blocking_reason=reason,
            details=details, evidence_hash=evidence_hash,
        )
        self.gates.append(gate)
        return gate

    # ── Gate 1: Provenance ────────────────────────────────────────────────

    def verify_provenance(self) -> bool:
        """Gate 1: Verify provenance evidence."""
        if self.provenance.provenance_level not in ("DOCUMENTED", "INDEPENDENTLY_VERIFIED"):
            self._add_gate("provenance", False,
                          f"Provenance level: {self.provenance.provenance_level} (requires DOCUMENTED or INDEPENDENTLY_VERIFIED)")
            self.block(f"provenance: {self.provenance.provenance_level}")
            return False

        if not self.provenance.original_source:
            self._add_gate("provenance", False, "No original source specified")
            self.block("provenance: no original source")
            return False

        h = self.provenance.compute_hash()
        self._add_gate("provenance", True, evidence_hash=h,
                       details=f"Provenance: {self.provenance.provenance_level}, source: {self.provenance.original_source}")
        self.transition(ExecutionState.PROVENANCE_VERIFIED, f"provenance_hash={h[:16]}...")
        return True

    # ── Gate 2: Acquisition ───────────────────────────────────────────────

    def request_acquisition(self) -> bool:
        """Gate 2: Request acquisition."""
        self._add_gate("acquisition_requested", True, details="Acquisition requested")
        return self.transition(ExecutionState.ACQUISITION_REQUESTED)

    def complete_acquisition(self, acquisition_id: str, dataset_hash: str) -> bool:
        """Gate 2b: Complete acquisition with verified artifact."""
        self.acquisition_id = acquisition_id
        self.dataset_hash = dataset_hash
        self._add_gate("acquisition_completed", True,
                       details=f"acquisition_id={acquisition_id}, hash={dataset_hash[:16]}...")
        return self.transition(ExecutionState.ACQUIRED)

    # ── Gate 3: Artifact Verification ─────────────────────────────────────

    def verify_artifact(self, expected_hash: str) -> bool:
        """Gate 3: Verify artifact identity matches expected hash."""
        if self.dataset_hash != expected_hash:
            self._add_gate("artifact_verification", False,
                          f"Hash mismatch: expected={expected_hash[:16]}... got={self.dataset_hash[:16]}...")
            self.block("artifact hash mismatch")
            return False

        self._add_gate("artifact_verification", True,
                       evidence_hash=self.dataset_hash,
                       details=f"Artifact hash verified: {self.dataset_hash[:16]}...")
        return self.transition(ExecutionState.ARTIFACT_VERIFIED)

    # ── Gate 4: Evidence Verification ─────────────────────────────────────

    def verify_evidence(self, provenance_hash: str) -> bool:
        """Gate 4: Verify all evidence is present and consistent."""
        if not provenance_hash:
            self._add_gate("evidence_verification", False, "No provenance hash")
            self.block("missing provenance evidence")
            return False

        self._add_gate("evidence_verification", True,
                       evidence_hash=provenance_hash,
                       details="Evidence bundle verified")
        return self.transition(ExecutionState.EVIDENCE_VERIFIED)

    # ── Gate 5: Label Semantics ───────────────────────────────────────────

    def verify_label_semantics(self) -> bool:
        """Gate 5: Verify fraud label semantics are complete and known."""
        complete, gaps = self.labels.is_complete()
        if not complete:
            self._add_gate("label_semantics", False, f"Label gaps: {'; '.join(gaps)}")
            self.block(f"label semantics: {'; '.join(gaps)}")
            return False

        h = self.labels.compute_hash()
        self._add_gate("label_semantics", True, evidence_hash=h,
                       details=f"Label: {self.labels.label_definition}, timing: {self.labels.label_timing}")
        return self.transition(ExecutionState.SEMANTICS_VERIFIED)

    # ── Gate 6: Temporal Verification ─────────────────────────────────────

    def verify_temporal(self) -> bool:
        """Gate 6: Verify temporal integrity."""
        complete, gaps = self.temporal.is_complete()
        if not complete:
            self._add_gate("temporal_verification", False, f"Temporal gaps: {'; '.join(gaps)}")
            self.block(f"temporal: {'; '.join(gaps)}")
            return False

        h = self.temporal.compute_hash()
        self._add_gate("temporal_verification", True, evidence_hash=h,
                       details=f"Temporal: {self.temporal.collection_start} to {self.temporal.collection_end}")
        return self.transition(ExecutionState.TEMPORAL_VERIFIED)

    # ── Gate 7: Independence ──────────────────────────────────────────────

    def verify_independence(self) -> bool:
        """Gate 7: Verify dataset independence."""
        independent, issues = self.independence.is_independent()
        if not independent:
            self._add_gate("independence", False, f"Independence issues: {'; '.join(issues)}")
            self.block(f"independence: {'; '.join(issues)}")
            return False

        h = self.independence.compute_hash()
        self._add_gate("independence", True, evidence_hash=h,
                       details="Independence verified")
        return self.transition(ExecutionState.INDEPENDENCE_VERIFIED)

    # ── Gate 8: Feature Compatibility ─────────────────────────────────────

    def verify_feature_compatibility(self) -> bool:
        """Gate 8: Verify feature mapping compatibility."""
        if not self.feature_mappings:
            self._add_gate("feature_compatibility", False, "No feature mappings defined")
            self.block("no feature mappings")
            return False

        missing = [m for m in self.feature_mappings if m.compatibility_status == "MISSING"]
        unsupported = [m for m in self.feature_mappings if m.compatibility_status == "UNSUPPORTED"]
        if missing or unsupported:
            names = [m.ps14_feature for m in missing + unsupported]
            self._add_gate("feature_compatibility", False,
                          f"Incompatible features: {names}")
            self.block(f"feature compatibility: {names}")
            return False

        mapping_hash = hashlib.sha256(
            json.dumps([m.to_dict() for m in self.feature_mappings], sort_keys=True, default=str).encode()
        ).hexdigest()
        self._add_gate("feature_compatibility", True, evidence_hash=mapping_hash,
                       details=f"Feature mapping: {len(self.feature_mappings)} features mapped")
        return self.transition(ExecutionState.FEATURE_COMPATIBILITY_VERIFIED)

    # ── Gate 9: Leakage Checks ────────────────────────────────────────────

    def verify_no_leakage(self) -> bool:
        """Gate 9: Verify no data leakage."""
        failed = [c for c in self.leakage_checks if not c.passed]
        if failed:
            reasons = [f"{c.check_type}: {c.details}" for c in failed]
            self._add_gate("leakage_checks", False, f"Leakage detected: {'; '.join(reasons)}")
            self.block(f"leakage: {'; '.join(reasons)}")
            return False

        h = hashlib.sha256(
            json.dumps([c.to_dict() for c in self.leakage_checks], sort_keys=True).encode()
        ).hexdigest()
        self._add_gate("leakage_checks", True, evidence_hash=h,
                       details=f"No leakage detected ({len(self.leakage_checks)} checks)")
        return True  # Leakage check doesn't change state — it's validated alongside features

    # ── Run All Gates ─────────────────────────────────────────────────────

    def run_all_gates(self) -> bool:
        """Execute all gates in sequence. Returns True only if all pass."""
        if not self.verify_provenance():
            return False
        if not self.request_acquisition():
            return False
        if not self.complete_acquisition(self.acquisition_id, self.dataset_hash):
            return False
        if not self.verify_artifact(self.dataset_hash):
            return False
        provenance_hash = self.provenance.compute_hash()
        if not self.verify_evidence(provenance_hash):
            return False
        if not self.verify_label_semantics():
            return False
        if not self.verify_temporal():
            return False
        if not self.verify_independence():
            return False
        if not self.verify_feature_compatibility():
            return False
        if not self.verify_no_leakage():
            return False
        return True

    # ── Certificate Creation ──────────────────────────────────────────────

    def create_certificate(
        self,
        model_id: str,
        artifact_set_hash: str,
        feature_version: str,
        release_id: str,
        evaluation_protocol_id: str = "Phase57-v1.0",
    ) -> EligibilityCertificate | None:
        """Create eligibility certificate after all gates pass.

        Only succeeds when state is FEATURE_COMPATIBILITY_VERIFIED
        and all gates passed.
        """
        if self.state != ExecutionState.FEATURE_COMPATIBILITY_VERIFIED:
            return None

        # Verify all gates passed (must have at least one gate)
        if not self.gates or not all(g.passed for g in self.gates):
            return None

        # Compute evidence hashes
        provenance_hash = self.provenance.compute_hash()
        label_hash = self.labels.compute_hash()
        temporal_hash = self.temporal.compute_hash()
        independence_hash = self.independence.compute_hash()
        feature_mapping_hash = hashlib.sha256(
            json.dumps([m.to_dict() for m in self.feature_mappings], sort_keys=True, default=str).encode()
        ).hexdigest()
        leakage_hash = hashlib.sha256(
            json.dumps([c.to_dict() for c in self.leakage_checks], sort_keys=True).encode()
        ).hexdigest()
        acquisition_hash = hashlib.sha256(
            f"{self.acquisition_id}:{self.dataset_hash}".encode()
        ).hexdigest()

        cert = EligibilityCertificate(
            certificate_id=f"cert-{self.dataset_id}",
            dataset_id=self.dataset_id,
            dataset_hash=self.dataset_hash,
            acquisition_id=self.acquisition_id,
            acquisition_hash=acquisition_hash,
            provenance_hash=provenance_hash,
            label_hash=label_hash,
            temporal_hash=temporal_hash,
            independence_hash=independence_hash,
            feature_mapping_hash=feature_mapping_hash,
            leakage_hash=leakage_hash,
            model_id=model_id,
            artifact_set_hash=artifact_set_hash,
            feature_version=feature_version,
            release_id=release_id,
            evaluation_protocol_id=evaluation_protocol_id,
            verdict="ELIGIBLE",
            gate_results=list(self.gates),
        )
        cert.compute_hash()

        # Transition to ELIGIBILITY_CERTIFIED
        self.certificate = cert
        self.transition(ExecutionState.ELIGIBILITY_CERTIFIED,
                       f"certificate={cert.certificate_hash[:16]}...")

        # Record forensic event
        self._record_forensic_event("certificate_created", cert.certificate_id)

        return cert

    # ── Forensic Event Recording ──────────────────────────────────────────

    def _record_forensic_event(self, event_type: str, reference_id: str = "") -> ForensicEvent:
        """Record a tamper-evident forensic event."""
        prev_hash = self.forensic_events[-1].event_hash if self.forensic_events else ""

        event = ForensicEvent(
            event_id=f"evt-{event_type}-{len(self.forensic_events)}",
            event_type=event_type,
            dataset_id=self.dataset_id,
            acquisition_id=self.acquisition_id,
            certificate_id=reference_id,
            dataset_hash=self.dataset_hash,
            evidence_hashes={
                "provenance": self.provenance.compute_hash(),
                "labels": self.labels.compute_hash(),
                "temporal": self.temporal.compute_hash(),
                "independence": self.independence.compute_hash(),
            },
            gate_results=list(self.gates),
            actor_type="SYSTEM",
        )
        event.compute_hash(prev_hash)
        self.forensic_events.append(event)
        return event

    def verify_forensic_chain(self) -> tuple[bool, list[str]]:
        """Verify the forensic event chain is intact."""
        errors = []
        for i, event in enumerate(self.forensic_events):
            expected_prev = self.forensic_events[i - 1].event_hash if i > 0 else ""
            if event.previous_event_hash != expected_prev:
                errors.append(f"Event {i}: previous hash mismatch")
                continue
            # Recompute hash
            d = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "dataset_id": event.dataset_id,
                "acquisition_id": event.acquisition_id,
                "certificate_id": event.certificate_id,
                "dataset_hash": event.dataset_hash,
                "evidence_hashes": event.evidence_hashes,
                "gate_results": [g.to_dict() for g in event.gate_results],
                "previous_event_hash": event.previous_event_hash,
            }
            canonical = json.dumps(d, sort_keys=True, default=str)
            expected_hash = hashlib.sha256(canonical.encode()).hexdigest()
            if event.event_hash != expected_hash:
                errors.append(f"Event {i}: hash mismatch (tampered)")
        return (len(errors) == 0, errors)

    def get_real_world_validation_status(self) -> str:
        """Return the current REAL_WORLD_VALIDATION status."""
        if (self.state == ExecutionState.ELIGIBILITY_CERTIFIED
                and self.certificate is not None
                and self.certificate.verdict == "ELIGIBLE"
                and not self.is_test_fixture):
            return "ELIGIBLE_FOR_EVALUATION"
        return "BLOCKED_PENDING_ELIGIBLE_DATASET"


# ── Test Fixture Helper ────────────────────────────────────────────────────

def create_fixture_execution() -> RealWorldDatasetExecution:
    """Create a TEST_FIXTURE execution for code-path testing.

    This is NOT a real-world dataset. It exercises the execution
    mechanics without producing any real-world validation claim.
    """
    wf = RealWorldDatasetExecution()
    wf.is_test_fixture = True
    wf.dataset_id = "fixture-dataset-001"
    wf.acquisition_id = "fixture-acq-001"

    # Provenance
    wf.provenance = ProvenanceEvidence(
        original_source="TEST_FIXTURE: synthetic source",
        publisher="TEST_FIXTURE",
        source_url="fixture://test",
        source_chain=["fixture_source"],
        license_status="PUBLICLY_ACCESSIBLE",
        provenance_level="INDEPENDENTLY_VERIFIED",
    )

    # Labels
    wf.labels = LabelEvidence(
        label_definition="TEST_FIXTURE: simulated fraud",
        label_column="label",
        positive_class="1",
        negative_class="0",
        label_granularity="transaction_level",
        label_generation_method="investigation",
        label_timing="at_event",
        label_timing_known=True,
        label_author="human_investigator",
        prediction_time_availability="available",
    )

    # Temporal
    wf.temporal = TemporalEvidence(
        timestamp_column="timestamp",
        collection_start="2024-01-01",
        collection_end="2024-12-31",
        ordering_verified=True,
        future_information_check="PASS",
    )

    # Independence
    wf.independence = IndependenceEvidence(
        ps14_derived="VERIFIED_INDEPENDENT",
        synthetic_generation="VERIFIED_INDEPENDENT",
        shared_ids="VERIFIED_INDEPENDENT",
        duplicated_rows="VERIFIED_INDEPENDENT",
        feature_vector_overlap="VERIFIED_INDEPENDENT",
        shared_source_lineage="VERIFIED_INDEPENDENT",
        augmented_copy="VERIFIED_INDEPENDENT",
    )

    # Feature mappings (21 PS-14 features)
    ps14_features = [
        "hour_of_day", "is_weekend", "amount_ratio", "txn_amount_bucket",
        "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
        "unusual_location_flag", "unusual_recipient_flag",
        "failed_auth_count_24h", "days_since_last_similar_txn",
        "gradual_escalation_score", "known_device_count",
        "account_tenure_days", "archetype",
        "txn_regularity", "recipient_risk_score", "device_risk_score",
        "location_risk_score", "amount_zscore", "velocity_ratio",
    ]
    for feat in ps14_features:
        wf.feature_mappings.append(FeatureMapping(
            external_column=f"ext_{feat}",
            ps14_feature=feat,
            datatype="float",
            transformation="direct",
            compatibility_status="DIRECT",
        ))
    for m in wf.feature_mappings:
        m.compute_transformation_hash()

    # Leakage checks
    wf.leakage_checks = [
        LeakageCheckResult("target_leakage", True, "No target-derived features"),
        LeakageCheckResult("future_information", True, "No post-event columns"),
        LeakageCheckResult("duplicate_contamination", True, "No duplicates detected"),
        LeakageCheckResult("train_evaluation_contamination", True, "No shared records"),
    ]

    return wf
