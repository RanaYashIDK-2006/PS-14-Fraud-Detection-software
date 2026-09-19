"""Phase 85: External dataset eligibility evidence & readiness assessment.

Builds the minimum auditable infrastructure for documenting an externally
acquired dataset candidate, verifying its provenance and eligibility
evidence, determining whether it can be evaluated by PS-14, preserving
a deterministic evaluation manifest, and explicitly keeping
REAL_WORLD_VALIDATION blocked when eligibility is not established.

Does NOT download datasets, bypass authentication, fabricate provenance,
create feature mappings, modify the model, or change the RWV gate.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.monitoring.dataset_admission import (
    FeatureCompatibilityState,
    FeatureSchemaDescriptor,
    evaluate_feature_compatibility,
    KNOWN_EXTERNAL_SCHEMAS,
    ML_FEATURE_CONTRACT,
    ML_FEATURE_ORDER,
    ML_FEATURE_VERSION,
)
from src.monitoring.dataset_ingestion import (
    DatasetClassification,
    DatasetProvenance,
    LabelProvenance,
    KNOWN_DATASETS,
)


# ── Eligibility / readiness enums ─────────────────────────────────────

class EligibilityStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    INCOMPATIBLE = "incompatible"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


class ReadinessStatus(str, Enum):
    READY_FOR_EXTERNAL_EVALUATION = "ready_for_external_evaluation"
    NOT_READY = "not_ready"
    BLOCKED = "blocked"


class IndependenceStatus(str, Enum):
    INDEPENDENT = "independent"
    DERIVED_FROM_PS14 = "derived_from_ps14"
    DERIVED_FROM_TRAINING = "derived_from_training"
    DERIVED_FROM_BENCHMARK = "derived_from_benchmark"
    UNKNOWN = "unknown"


class ProvenanceStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FORGED = "forged"
    UNKNOWN = "unknown"


# ── Evidence record ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ExternalDatasetEvidence:
    """Immutable evidence record for an externally acquired dataset.

    Captures only auditable metadata.  Does NOT store raw data or PII.
    """
    dataset_id: str
    dataset_name: str
    dataset_version: str
    source_url: str
    publisher: str
    acquisition_timestamp: str
    source_hash: str
    row_count: int
    schema_hash: str
    schema_description: str
    feature_schema_id: str
    feature_compatibility: str  # FeatureCompatibilityState value
    label_provenance: str  # LabelProvenance value
    label_definition: str
    temporal_coverage: str
    geography: str
    independence_status: str  # IndependenceStatus value
    provenance_status: str  # ProvenanceStatus value
    eligibility_status: str  # EligibilityStatus value
    provenance_reference: str
    evidence_notes: str
    policy_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_url": self.source_url,
            "publisher": self.publisher,
            "acquisition_timestamp": self.acquisition_timestamp,
            "source_hash": self.source_hash,
            "row_count": self.row_count,
            "schema_hash": self.schema_hash,
            "schema_description": self.schema_description,
            "feature_schema_id": self.feature_schema_id,
            "feature_compatibility": self.feature_compatibility,
            "label_provenance": self.label_provenance,
            "label_definition": self.label_definition,
            "temporal_coverage": self.temporal_coverage,
            "geography": self.geography,
            "independence_status": self.independence_status,
            "provenance_status": self.provenance_status,
            "eligibility_status": self.eligibility_status,
            "provenance_reference": self.provenance_reference,
            "evidence_notes": self.evidence_notes,
            "policy_version": self.policy_version,
        }


# ── Eligibility assessment ────────────────────────────────────────────

@dataclass(frozen=True)
class EligibilityAssessment:
    """Deterministic eligibility assessment of an external dataset."""
    provenance_status: str
    label_provenance_status: str
    independence_status: str
    temporal_validity: str
    feature_compatibility_status: str
    evaluation_suitability: str
    overall_eligibility: str  # EligibilityStatus value
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance_status": self.provenance_status,
            "label_provenance_status": self.label_provenance_status,
            "independence_status": self.independence_status,
            "temporal_validity": self.temporal_validity,
            "feature_compatibility_status": self.feature_compatibility_status,
            "evaluation_suitability": self.evaluation_suitability,
            "overall_eligibility": self.overall_eligibility,
            "blocking_reasons": list(self.blocking_reasons),
        }


def assess_eligibility(
    evidence: ExternalDatasetEvidence,
) -> EligibilityAssessment:
    """Deterministic eligibility assessment from evidence metadata.

    Evaluates each eligibility dimension independently.  Does NOT
    modify the evidence record.
    """
    blocking: list[str] = []

    # A. Provenance
    provenance_ok = evidence.provenance_status == ProvenanceStatus.VERIFIED.value
    if not provenance_ok:
        blocking.append(f"provenance_not_verified: {evidence.provenance_status}")

    # B. Label provenance
    label_ok = evidence.label_provenance in (
        LabelProvenance.INDEPENDENT_HUMAN.value,
        LabelProvenance.INDEPENDENT_INSTITUTIONAL.value,
    )
    if not label_ok:
        blocking.append(f"label_provenance_not_independent: {evidence.label_provenance}")

    # C. Independence
    indep_ok = evidence.independence_status == IndependenceStatus.INDEPENDENT.value
    if not indep_ok:
        blocking.append(f"independence_not_established: {evidence.independence_status}")

    # D. Temporal validity
    temporal_ok = bool(evidence.temporal_coverage and evidence.temporal_coverage != "unknown")
    if not temporal_ok:
        blocking.append("temporal_coverage_missing")

    # E. Feature compatibility
    compat_ok = evidence.feature_compatibility == FeatureCompatibilityState.EXACT_COMPATIBLE.value
    if not compat_ok:
        blocking.append(f"feature_incompatible: {evidence.feature_compatibility}")

    # F. Evaluation suitability
    suitable_rows = evidence.row_count >= 100  # minimum viable dataset
    suitable_label = bool(evidence.label_definition)
    suitable = suitable_rows and suitable_label
    if not suitable:
        reasons = []
        if not suitable_rows:
            reasons.append(f"insufficient_rows: {evidence.row_count}")
        if not suitable_label:
            reasons.append("label_definition_missing")
        blocking.append(f"evaluation_unsuitable: {'; '.join(reasons)}")

    # Overall
    if blocking:
        overall = EligibilityStatus.INELIGIBLE
    elif provenance_ok and label_ok and indep_ok and temporal_ok and compat_ok and suitable:
        overall = EligibilityStatus.VERIFIED
    else:
        overall = EligibilityStatus.UNKNOWN

    return EligibilityAssessment(
        provenance_status=evidence.provenance_status,
        label_provenance_status=evidence.label_provenance,
        independence_status=evidence.independence_status,
        temporal_validity="valid" if temporal_ok else "invalid",
        feature_compatibility_status=evidence.feature_compatibility,
        evaluation_suitability="suitable" if suitable else "unsuitable",
        overall_eligibility=overall.value,
        blocking_reasons=tuple(blocking),
    )


# ── Readiness assessment ──────────────────────────────────────────────

@dataclass(frozen=True)
class ReadinessAssessment:
    """Deterministic readiness assessment for external evaluation."""
    readiness: str  # ReadinessStatus value
    eligibility: str  # EligibilityStatus value
    blocking_reasons: tuple[str, ...]
    assessment_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "readiness": self.readiness,
            "eligibility": self.eligibility,
            "blocking_reasons": list(self.blocking_reasons),
            "assessment_timestamp": self.assessment_timestamp,
        }


def assess_external_dataset_readiness(
    evidence: ExternalDatasetEvidence,
) -> ReadinessAssessment:
    """Deterministic readiness assessment.

    READY requires actual evidence satisfying the established project gate.
    If no eligible dataset is present, the result is BLOCKED.
    """
    eligibility = assess_eligibility(evidence)

    if eligibility.overall_eligibility == EligibilityStatus.VERIFIED.value:
        readiness = ReadinessStatus.READY_FOR_EXTERNAL_EVALUATION
    elif eligibility.overall_eligibility == EligibilityStatus.INELIGIBLE.value:
        readiness = ReadinessStatus.BLOCKED
    else:
        readiness = ReadinessStatus.NOT_READY

    return ReadinessAssessment(
        readiness=readiness.value,
        eligibility=eligibility.overall_eligibility,
        blocking_reasons=eligibility.blocking_reasons,
        assessment_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Evaluation manifest ───────────────────────────────────────────────

@dataclass(frozen=True)
class EvaluationManifest:
    """Deterministic evaluation manifest for an external dataset."""
    dataset_id: str
    dataset_name: str
    dataset_version: str
    source_hash: str
    schema_hash: str
    feature_contract_version: str
    model_id: str
    model_release: str
    evaluation_protocol_version: str
    eligibility_status: str
    readiness_status: str
    provenance_status: str
    label_provenance: str
    independence_status: str
    feature_compatibility: str
    manifest_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_hash": self.source_hash,
            "schema_hash": self.schema_hash,
            "feature_contract_version": self.feature_contract_version,
            "model_id": self.model_id,
            "model_release": self.model_release,
            "evaluation_protocol_version": self.evaluation_protocol_version,
            "eligibility_status": self.eligibility_status,
            "readiness_status": self.readiness_status,
            "provenance_status": self.provenance_status,
            "label_provenance": self.label_provenance,
            "independence_status": self.independence_status,
            "feature_compatibility": self.feature_compatibility,
            "manifest_version": self.manifest_version,
        }

    def compute_hash(self) -> str:
        """Deterministic SHA-256 hash of the manifest."""
        d = self.to_dict()
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_evaluation_manifest(
    evidence: ExternalDatasetEvidence,
    readiness: ReadinessAssessment,
) -> EvaluationManifest:
    """Build a deterministic evaluation manifest."""
    return EvaluationManifest(
        dataset_id=evidence.dataset_id,
        dataset_name=evidence.dataset_name,
        dataset_version=evidence.dataset_version,
        source_hash=evidence.source_hash,
        schema_hash=evidence.schema_hash,
        feature_contract_version=ML_FEATURE_VERSION,
        model_id="altman_native",
        model_release="release-altman_native_E_hardneg_cert_20260904",
        evaluation_protocol_version="v1",
        eligibility_status=readiness.eligibility,
        readiness_status=readiness.readiness,
        provenance_status=evidence.provenance_status,
        label_provenance=evidence.label_provenance,
        independence_status=evidence.independence_status,
        feature_compatibility=evidence.feature_compatibility,
        manifest_version="phase85_v1",
    )


# ── Convenience: build evidence from ingestion ────────────────────────

def build_evidence_from_ingestion(
    ingestion_result: dict[str, Any],
    publisher: str = "",
    source_url: str = "",
    provenance_status: str = ProvenanceStatus.UNVERIFIED.value,
    independence_status: str = IndependenceStatus.UNKNOWN.value,
    label_provenance_override: str = "",
    temporal_coverage: str = "unknown",
    geography: str = "unknown",
    provenance_reference: str = "",
    evidence_notes: str = "",
) -> ExternalDatasetEvidence:
    """Build ExternalDatasetEvidence from an ingestion result dict.

    Does NOT fabricate provenance or independence.  Defaults to
    UNVERIFIED/UNKNOWN for safety.
    """
    prov = ingestion_result.get("provenance", {})
    compat = ingestion_result.get("feature_compatibility", {})
    cand = ingestion_result.get("candidate", {})

    label_prov = label_provenance_override or prov.get("label_provenance", LabelProvenance.UNKNOWN.value)

    return ExternalDatasetEvidence(
        dataset_id=cand.get("dataset_id", ""),
        dataset_name=prov.get("dataset_name", ""),
        dataset_version=cand.get("dataset_version", ""),
        source_url=source_url or prov.get("source_url", ""),
        publisher=publisher,
        acquisition_timestamp=ingestion_result.get("ingestion_time", ""),
        source_hash=prov.get("source_hash", ""),
        row_count=prov.get("row_count", 0),
        schema_hash=prov.get("schema_hash", ""),
        schema_description=str(prov.get("schema_columns", [])),
        feature_schema_id=cand.get("feature_schema_id", ""),
        feature_compatibility=compat.get("compatibility", "unknown"),
        label_provenance=label_prov,
        label_definition=prov.get("label_field", ""),
        temporal_coverage=temporal_coverage,
        geography=geography,
        independence_status=independence_status,
        provenance_status=provenance_status,
        eligibility_status=EligibilityStatus.UNKNOWN.value,
        provenance_reference=provenance_reference,
        evidence_notes=evidence_notes,
        policy_version="outcome_trust_policy_v1",
    )
