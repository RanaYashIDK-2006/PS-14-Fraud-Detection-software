"""Phase 105: Provider Evidence Intake & Verification Boundary.

Strict, machine-checkable intake for externally supplied provider or
institution evidence, feeding EXISTING Phase 104 qualification only:

    External provider/institution evidence
            ↓
    Phase 105 evidence intake  (submit → validate → canonicalize → hash)
            ↓
    Evidence authenticity / integrity / provenance checks
            ↓
    Evidence normalization → Phase 104 EvidencePackage
            ↓
    Phase 104 evaluate_dataset_evidence()   (the ONLY qualification authority)
            ↓
    DATASET_QUALIFIED_FOR_CONTROLLED_RWV → Phase 96 RWV

Integrity is deliberately SEPARATE from truth: a document hash proves a
supplied document has not changed; it never makes a provider claim true.
INTEGRITY / AUTHENTICITY / PROVENANCE / CONTENT_CLAIM are four distinct
verification states — never collapsed into one "trusted" flag — and a
system-generated hash never becomes a provider attestation.

This module: NO network, NO provider contact, NO dataset acquisition, NO
raw transaction records, NO secrets/PII storage, NO model access, NO RWV,
NO promotion, NO duplicate qualification authority, NO bypass parameters
(force / allow_unverified / skip_validation / override / admin_override).

STATUS: IMPLEMENTED
SYSTEM_READINESS: SYSTEM_READY_PENDING_ELIGIBLE_DATASET
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
PROMOTION: PROMOTION_GATE_REQUIRED
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Sequence

from src.monitoring.external_dataset_contract import (
    ATTESTED_ORIGINS,
    CONTEXT_ID,
    FORBIDDEN_CONTENT_PATTERNS,
    MANDATORY_DIMENSIONS,
    PROVIDER_FACT_DIMENSIONS,
    UNTRUSTED_ORIGINS,
    ContractDimension,
    EvidenceItem,
    EvidenceOrigin,
    EvidencePackage,
    EvidenceStatus,
    ExternalDatasetContract,
    build_evidence_package,
    canonical_json,
    make_evidence_item,
    worst_status,
)
from src.monitoring.external_dataset_qualification import (
    QualificationResult,
    evaluate_dataset_evidence,
)
# Phase 95 weak-evidence detector: a reference string starting with
# "UNVERIFIED" / "TBD" / ... is not verified evidence.
from src.monitoring.provider_evidence import _is_evidence_verified
# Fixed assessment instant (deterministic; shared with Phase 104 fixtures).
from src.monitoring.external_dataset_contract import EVIDENCE_AS_OF
import re

# ══════════════════════════════════════════════════════════════════════
# VERSIONS
# ══════════════════════════════════════════════════════════════════════

INTAKE_VERSION = "phase105_intake_v1"
PACKAGE_VERSION = "phase105_evidence_package_v1"

# No expiry exists ⇒ preserved verbatim, never auto-expired (spec §9).
NO_EXPIRY_DECLARED = "no_expiry_declared"

# Deterministic document types accepted by the metadata-only intake.
SUPPORTED_DOCUMENT_TYPES: frozenset[str] = frozenset({
    "application/pdf", "text/plain", "text/csv", "application/json",
    "application/xml", "text/xml",
})

# The explicit evidence chain (spec §6) — every stage traceable.
EVIDENCE_CHAIN_STAGES: tuple[str, ...] = (
    "evidence_artifact",
    "evidence_submission",
    "attestation_scope",
    "evidence_dimension",
    "phase104_evidence",
    "dataset_qualification",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _parse_iso(value: str) -> Any:
    from datetime import datetime
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(
        canonical_json(obj).encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# CONTROLLED ENUMS
# ══════════════════════════════════════════════════════════════════════

class AttestationScope(str, Enum):
    """What an attestation actually covers (spec §4).

    An attestation is NEVER accepted for a dimension it does not cover:
    a document confirming dataset identity does not satisfy label
    provenance, and a usage agreement does not prove feature compatibility.
    """
    PROVIDER_IDENTITY = "provider_identity"
    DATASET_IDENTITY = "dataset_identity"
    DATASET_VERSION = "dataset_version"
    DATASET_PERIOD = "dataset_period"
    LABEL_PROVENANCE = "label_provenance"
    TIMESTAMP_SEMANTICS = "timestamp_semantics"
    FEATURE_SCHEMA = "feature_schema"
    ENTITY_CONTINUITY = "entity_continuity"
    INDEPENDENCE = "independence"
    CONTAMINATION = "contamination"
    USAGE_PERMISSION = "usage_permission"
    ACCESS_AUTHORITY = "access_authority"
    EVALUATION_SUITABILITY = "evaluation_suitability"


# Which mandatory Phase 104 dimension(s) each attestation scope may cover.
# EVALUATION_SUITABILITY maps to the label-provenance claim only, where
# "suitability for evaluation" is recorded (spec §1.D) — it never covers
# access authority, usage permission, or feature compatibility.
SCOPE_DIMENSIONS: dict[AttestationScope, frozenset[str]] = {
    AttestationScope.PROVIDER_IDENTITY: frozenset({
        ContractDimension.PROVIDER_IDENTITY.value}),
    AttestationScope.DATASET_IDENTITY: frozenset({
        ContractDimension.DATASET_IDENTITY.value}),
    AttestationScope.DATASET_VERSION: frozenset({
        ContractDimension.DATASET_IDENTITY.value}),
    AttestationScope.DATASET_PERIOD: frozenset({
        ContractDimension.DATASET_IDENTITY.value}),
    AttestationScope.LABEL_PROVENANCE: frozenset({
        ContractDimension.LABEL_PROVENANCE.value}),
    AttestationScope.TIMESTAMP_SEMANTICS: frozenset({
        ContractDimension.TEMPORAL_SEMANTICS.value}),
    AttestationScope.FEATURE_SCHEMA: frozenset({
        ContractDimension.FEATURE_COMPATIBILITY.value}),
    AttestationScope.ENTITY_CONTINUITY: frozenset({
        ContractDimension.ENTITY_CONTINUITY.value}),
    AttestationScope.INDEPENDENCE: frozenset({
        ContractDimension.INDEPENDENCE.value}),
    AttestationScope.CONTAMINATION: frozenset({
        ContractDimension.CONTAMINATION.value}),
    AttestationScope.USAGE_PERMISSION: frozenset({
        ContractDimension.USAGE_PERMISSION.value}),
    AttestationScope.ACCESS_AUTHORITY: frozenset({
        ContractDimension.ACCESS_AUTHORITY.value}),
    AttestationScope.EVALUATION_SUITABILITY: frozenset({
        ContractDimension.LABEL_PROVENANCE.value}),
}


def scope_covers_dimension(scope: str, dimension: str) -> bool:
    """True only when the attestation scope explicitly covers the
    dimension (spec §4)."""
    try:
        return dimension in SCOPE_DIMENSIONS[AttestationScope(scope)]
    except ValueError:
        return False


class UsagePermissionState(str, Enum):
    """Explicit usage-permission states (spec §10).  Permission is NEVER
    inferred from public availability, papers, repositories, or
    popularity; UNKNOWN fails closed."""
    AUTHORIZED_FOR_RESEARCH = "authorized_for_research"
    AUTHORIZED_FOR_EVALUATION = "authorized_for_evaluation"
    AUTHORIZED_FOR_INTERNAL_VALIDATION = "authorized_for_internal_validation"
    AUTHORIZED_FOR_MODEL_EVALUATION = "authorized_for_model_evaluation"
    AUTHORIZED_FOR_TRAINING = "authorized_for_training"
    AUTHORIZED_FOR_PUBLICATION = "authorized_for_publication"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


# Permission states that cover the intended use (model evaluation).
EVALUATION_PERMISSION_STATES: frozenset[str] = frozenset({
    UsagePermissionState.AUTHORIZED_FOR_RESEARCH.value,
    UsagePermissionState.AUTHORIZED_FOR_EVALUATION.value,
    UsagePermissionState.AUTHORIZED_FOR_INTERNAL_VALIDATION.value,
    UsagePermissionState.AUTHORIZED_FOR_MODEL_EVALUATION.value,
})


class VerificationState(str, Enum):
    """Four INDEPENDENT verification axes (spec §2) — never collapsed.

    Integrity proves a document has not changed; authenticity ties an
    attestation to a known authority; provenance identifies where
    evidence came from; the content claim is the actual provider fact.
    """
    INTEGRITY_VERIFIED = "integrity_verified"
    INTEGRITY_UNVERIFIED = "integrity_unverified"
    AUTHENTICITY_VERIFIED = "authenticity_verified"
    AUTHENTICITY_UNVERIFIED = "authenticity_unverified"
    PROVENANCE_VERIFIED = "provenance_verified"
    PROVENANCE_UNVERIFIED = "provenance_unverified"
    CONTENT_CLAIM_VERIFIED = "content_claim_verified"
    CONTENT_CLAIM_UNVERIFIED = "content_claim_unverified"


# ══════════════════════════════════════════════════════════════════════
# INTAKE REASON CODES (deterministic, sorted at output)
# ══════════════════════════════════════════════════════════════════════

class IntakeReasonCode:
    """Deterministic provider-evidence intake reason codes."""
    # Identity / binding (spec §7/§8)
    EVIDENCE_ID_MISSING = "evidence_id_missing"
    PROVIDER_ID_MISSING = "provider_id_missing"
    PROVIDER_ID_MISMATCH = "provider_id_mismatch"
    DATASET_ID_MISSING = "dataset_id_missing"
    DATASET_ID_MISMATCH = "dataset_id_mismatch"
    DATASET_VERSION_MISSING = "dataset_version_missing"
    DATASET_VERSION_MISMATCH = "dataset_version_mismatch"
    CROSS_CONTEXT_EVIDENCE = "cross_context_evidence"
    # Document integrity (spec §5)
    EVIDENCE_HASH_MISMATCH = "evidence_hash_mismatch"
    DOCUMENT_HASH_MALFORMED = "document_hash_malformed"
    DOCUMENT_SIZE_INVALID = "document_size_invalid"
    UNSUPPORTED_DOCUMENT_TYPE = "unsupported_document_type"
    FORBIDDEN_CONTENT = "evidence_contains_forbidden_content"
    # Provenance / freshness (spec §9)
    SUPPLIED_BY_MISSING = "supplied_by_missing"
    SOURCE_REFERENCE_MISSING = "source_reference_missing"
    EVIDENCE_TIMESTAMP_INVALID = "evidence_timestamp_invalid"
    EVIDENCE_STALE = "evidence_stale"
    EVIDENCE_FUTURE_DATED = "evidence_future_dated"
    ATTESTATION_TIMESTAMP_MISSING = "attestation_timestamp_missing"
    ATTESTATION_TIMESTAMP_INVALID = "attestation_timestamp_invalid"
    ATTESTATION_FUTURE_DATED = "attestation_future_dated"
    ATTESTATION_EXPIRED = "attestation_expired"
    ATTESTATION_INTERVAL_INVALID = "attestation_interval_invalid"
    # Origin rules (spec §3)
    EVIDENCE_ORIGIN_UNKNOWN = "evidence_origin_unknown"
    EVIDENCE_ORIGIN_UNTRUSTED = "evidence_origin_untrusted"
    ORIGIN_CANNOT_ATTEST = "origin_cannot_attest"
    ORIGIN_ATTESTATION_MISMATCH = "origin_attestation_mismatch"
    PROVIDER_FACT_WITHOUT_ATTESTATION = "provider_fact_without_attestation"
    PUBLIC_DOCUMENTATION_NOT_AUTHORIZATION = (
        "public_documentation_not_authorization")
    # Attestation / authenticity (spec §4)
    ATTESTATION_REFERENCE_MISSING = "attestation_reference_missing"
    INSTITUTIONAL_ATTESTATION_REFERENCE_MISSING = (
        "institutional_attestation_reference_missing")
    ATTESTATION_REFERENCE_UNVERIFIED = "attestation_reference_unverified"
    ATTESTATION_REFERENCE_MISMATCH = "attestation_reference_mismatch"
    ATTESTATION_SCOPE_MISSING = "attestation_scope_missing"
    ATTESTATION_SCOPE_DIMENSION_MISMATCH = "attestation_scope_dimension_mismatch"
    # Usage permission (spec §10)
    USAGE_PERMISSION_UNKNOWN = "usage_permission_unknown"
    USAGE_PERMISSION_RESTRICTED = "usage_permission_restricted"
    USAGE_PERMISSION_NOT_COVERING_EVALUATION = (
        "usage_permission_not_covering_evaluation")
    USAGE_PERMISSION_EXPIRED = "usage_permission_expired"
    # Status / content claim
    EVIDENCE_STATUS_NOT_VERIFIED = "evidence_status_not_verified"
    # Conflict / duplication codes (spec §5/§12) — emitted by
    # detect_evidence_conflicts(); CONTRADICTED types never resolve.
    CONFLICTING_EVIDENCE_ID = "conflicting_evidence_id"
    DUPLICATE_EVIDENCE_ID = "duplicate_evidence_id"
    CONFLICTING_ARTIFACT_HASH = "conflicting_artifact_hash"
    DUPLICATE_ARTIFACT_HASH = "duplicate_artifact_hash"
    CONFLICTING_DOCUMENT_CLAIMS = "conflicting_document_claims"
    CONFLICTING_EFFECTIVE_INTERVAL = "conflicting_effective_interval"


# Conflict codes that force CONTRADICTED (never silently resolved).
CONTRADICTED_CONFLICT_CODES: frozenset[str] = frozenset({
    IntakeReasonCode.CONFLICTING_EVIDENCE_ID,
    IntakeReasonCode.CONFLICTING_ARTIFACT_HASH,
    IntakeReasonCode.CONFLICTING_DOCUMENT_CLAIMS,
    IntakeReasonCode.CONFLICTING_EFFECTIVE_INTERVAL,
    IntakeReasonCode.ATTESTATION_INTERVAL_INVALID,
})

# Conflict codes that are duplicates: integrity defects → UNVERIFIED.
DUPLICATE_CONFLICT_CODES: frozenset[str] = frozenset({
    IntakeReasonCode.DUPLICATE_EVIDENCE_ID,
    IntakeReasonCode.DUPLICATE_ARTIFACT_HASH,
})


# ══════════════════════════════════════════════════════════════════════
# IMMUTABLE INTAKE STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EvidenceArtifact:
    """Metadata-only document reference — never the raw document."""
    source_reference: str = ""
    source_document_hash: str = ""
    source_document_size: int = 0
    source_document_type: str = ""

    def artifact_hash(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True)
class EvidenceSubmission:
    """One immutable, metadata-only, externally supplied evidence item.

    No PAN, credentials, passwords, API keys, secrets, unnecessary PII,
    or raw transaction records — document references, hashes, and
    attestation statements only (spec §1/§11).
    """
    evidence_id: str = ""
    provider_id: str = ""
    dataset_id: str = ""
    dataset_version: str = ""
    dimension: str = ""
    evidence_type: str = ""
    evidence_origin: str = EvidenceOrigin.UNKNOWN.value
    evidence_status: str = EvidenceStatus.UNVERIFIED.value
    statement: str = ""
    supplied_by: str = ""
    supplied_at: str = ""
    source_reference: str = ""
    source_document_hash: str = ""
    source_document_size: int = 0
    source_document_type: str = ""
    provider_attestation_reference: str = ""
    institutional_attestation_reference: str = ""
    attestation_timestamp: str = ""
    attestation_scope: tuple[str, ...] = ()
    effective_from: str = ""
    effective_until: str = ""
    claim_key: str = ""
    claim_value: str = ""
    permission_state: str = UsagePermissionState.UNKNOWN.value
    context_id: str = CONTEXT_ID
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "attestation_scope",
                           tuple(self.attestation_scope))
        if self.dimension not in MANDATORY_DIMENSIONS:
            raise ValueError(f"unknown evidence dimension: {self.dimension}")
        if self.evidence_origin not in {o.value for o in EvidenceOrigin}:
            raise ValueError(f"unknown evidence origin: {self.evidence_origin}")
        if self.evidence_status not in {s.value for s in EvidenceStatus}:
            raise ValueError(f"unknown evidence status: {self.evidence_status}")
        if self.permission_state not in {p.value for p in UsagePermissionState}:
            raise ValueError(f"unknown usage permission: {self.permission_state}")
        for scope in self.attestation_scope:
            if scope not in {s.value for s in AttestationScope}:
                raise ValueError(f"unknown attestation scope: {scope}")
        if not self.evidence_hash:
            object.__setattr__(self, "evidence_hash", self.evidence_hash_for())

    def canonical_serialization(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k != "evidence_hash"}
        return canonical_json(d)

    def evidence_hash_for(self) -> str:
        return _stable_hash(
            json.loads(self.canonical_serialization()))

    def verify_hash(self) -> bool:
        return self.evidence_hash == self.evidence_hash_for()

    def artifact(self) -> EvidenceArtifact:
        return EvidenceArtifact(
            source_reference=self.source_reference,
            source_document_hash=self.source_document_hash,
            source_document_size=self.source_document_size,
            source_document_type=self.source_document_type,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["attestation_scope"] = list(self.attestation_scope)
        return d


def submit_provider_evidence(**fields: Any) -> EvidenceSubmission:
    """Deterministic, offline intake of one evidence submission.

    Constructs the immutable submission and computes its SHA-256 evidence
    hash.  Unknown/dimension/origin/status/scope values fail closed with
    ValueError; unknown keyword arguments (including force / override /
    bypass style bypasses) fail closed with TypeError.  No network, no
    model access, no side effects.
    """
    return EvidenceSubmission(**fields)


def canonicalize_provider_evidence(submission: EvidenceSubmission) -> str:
    """Canonical, order-stable serialization of one submission."""
    return submission.canonical_serialization()


def hash_provider_evidence(submission: EvidenceSubmission) -> str:
    """Deterministic SHA-256 evidence hash."""
    return submission.evidence_hash_for()


# ══════════════════════════════════════════════════════════════════════
# VALIDATION REPORT (four independent verification axes)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ValidationReport:
    """Deterministic intake validation result for one submission.

    valid == True only when integrity, provenance, and the content claim
    are all VERIFIED and no reason code fired.  Authenticity is required
    for attested origins; public documentation is never authenticable as
    a provider attestation and is judged on its content claim alone.
    """
    evidence_id: str
    valid: bool
    integrity_state: str
    authenticity_state: str
    provenance_state: str
    content_claim_state: str
    freshness_state: str
    binding_state: str
    scope_state: str
    reason_codes: tuple[str, ...]
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["reason_codes"] = list(self.reason_codes)
        return d


def validate_provider_evidence(
    submission: EvidenceSubmission,
    contract: ExternalDatasetContract,
) -> ValidationReport:
    """Fail-closed intake validation of one submission against the Phase
    104 contract it claims to support (spec §7/§9/§10/§12).

    Deterministic and offline: provider/dataset/version/context binding,
    document integrity, origin rules, attestation scope, attestation
    freshness, and explicit usage permission.  No bypass parameters.
    """
    reasons: list[str] = []

    # ── 1. Document integrity (independence: never proves truth) ──
    if submission.verify_hash():
        integrity = VerificationState.INTEGRITY_VERIFIED.value
    else:
        integrity = VerificationState.INTEGRITY_UNVERIFIED.value
        reasons.append(IntakeReasonCode.EVIDENCE_HASH_MISMATCH)
    if not _HEX64.match(submission.source_document_hash or ""):
        integrity = VerificationState.INTEGRITY_UNVERIFIED.value
        reasons.append(IntakeReasonCode.DOCUMENT_HASH_MALFORMED)
    if submission.source_document_size <= 0:
        integrity = VerificationState.INTEGRITY_UNVERIFIED.value
        reasons.append(IntakeReasonCode.DOCUMENT_SIZE_INVALID)
    if submission.source_document_type not in SUPPORTED_DOCUMENT_TYPES:
        integrity = VerificationState.INTEGRITY_UNVERIFIED.value
        reasons.append(IntakeReasonCode.UNSUPPORTED_DOCUMENT_TYPE)

    # Content boundary: no secrets / PII / raw records (spec §5/§11).
    blob = "\n".join(str(v) for k, v in submission.to_dict().items()
                     if isinstance(v, str))
    for pattern in FORBIDDEN_CONTENT_PATTERNS:
        if re.search(pattern, blob, flags=re.IGNORECASE):
            reasons.append(IntakeReasonCode.FORBIDDEN_CONTENT)
            integrity = VerificationState.INTEGRITY_UNVERIFIED.value
            break

    # ── 2. Provenance (where it came from) ──
    provenance = VerificationState.PROVENANCE_VERIFIED.value
    if not submission.supplied_by:
        provenance = VerificationState.PROVENANCE_UNVERIFIED.value
        reasons.append(IntakeReasonCode.SUPPLIED_BY_MISSING)
    if not submission.source_reference:
        provenance = VerificationState.PROVENANCE_UNVERIFIED.value
        reasons.append(IntakeReasonCode.SOURCE_REFERENCE_MISSING)
    supplied = _parse_iso(submission.supplied_at)
    valid_from = _parse_iso(contract.evidence_valid_from)
    cutoff = _parse_iso(contract.evidence_cutoff)
    if supplied is None:
        provenance = VerificationState.PROVENANCE_UNVERIFIED.value
        reasons.append(IntakeReasonCode.EVIDENCE_TIMESTAMP_INVALID)
    else:
        if valid_from is not None and supplied < valid_from:
            provenance = VerificationState.PROVENANCE_UNVERIFIED.value
            reasons.append(IntakeReasonCode.EVIDENCE_STALE)
        if cutoff is not None and supplied > cutoff:
            provenance = VerificationState.PROVENANCE_UNVERIFIED.value
            reasons.append(IntakeReasonCode.EVIDENCE_FUTURE_DATED)

    # ── 3. Provider/dataset/version/context binding (spec §7/§8/§17) ──
    binding_failed = False
    if not submission.evidence_id:
        reasons.append(IntakeReasonCode.EVIDENCE_ID_MISSING)
        binding_failed = True
    if not submission.provider_id:
        reasons.append(IntakeReasonCode.PROVIDER_ID_MISSING)
        binding_failed = True
    elif submission.provider_id != contract.provider_reference:
        reasons.append(IntakeReasonCode.PROVIDER_ID_MISMATCH)
        binding_failed = True
    if not submission.dataset_id:
        reasons.append(IntakeReasonCode.DATASET_ID_MISSING)
        binding_failed = True
    elif submission.dataset_id != contract.dataset_id:
        reasons.append(IntakeReasonCode.DATASET_ID_MISMATCH)
        binding_failed = True
    contract_version = contract.dataset_identity.version
    if contract_version:
        if not submission.dataset_version:
            reasons.append(IntakeReasonCode.DATASET_VERSION_MISSING)
            binding_failed = True
        elif submission.dataset_version != contract_version:
            reasons.append(IntakeReasonCode.DATASET_VERSION_MISMATCH)
            binding_failed = True
    if submission.context_id != contract.context_id:
        reasons.append(IntakeReasonCode.CROSS_CONTEXT_EVIDENCE)
        binding_failed = True

    # ── 4. Evidence origin rules (spec §3) ──
    origin = submission.evidence_origin
    attested_origin = origin in ATTESTED_ORIGINS
    provider_fact = submission.dimension in PROVIDER_FACT_DIMENSIONS
    authenticity = VerificationState.AUTHENTICITY_UNVERIFIED.value

    if origin == EvidenceOrigin.UNKNOWN.value:
        reasons.append(IntakeReasonCode.EVIDENCE_ORIGIN_UNKNOWN)
    if origin in UNTRUSTED_ORIGINS:
        # Local/system evidence never proves external/provider facts.
        reasons.append(IntakeReasonCode.EVIDENCE_ORIGIN_UNTRUSTED)
        if submission.evidence_type and "attestation" in submission.evidence_type.lower():
            reasons.append(IntakeReasonCode.ORIGIN_ATTESTATION_MISMATCH)
    if origin in (EvidenceOrigin.LOCAL_SYNTHETIC.value,
                  EvidenceOrigin.LOCAL_RESEARCH.value,
                  EvidenceOrigin.SYSTEM_GENERATED.value,
                  EvidenceOrigin.PUBLIC_DOCUMENTATION.value):
        if (submission.provider_attestation_reference
                or submission.institutional_attestation_reference):
            # A document hash or local artifact never becomes a
            # provider/institution attestation.
            reasons.append(IntakeReasonCode.ORIGIN_CANNOT_ATTEST)
            if evidence_type_implies_provider(submission.evidence_type, origin):
                reasons.append(IntakeReasonCode.ORIGIN_ATTESTATION_MISMATCH)

    if origin == EvidenceOrigin.PROVIDER_ATTESTED.value:
        declared = contract.provider_identity.provider_attestation_reference
        ref = submission.provider_attestation_reference
        if not ref:
            reasons.append(IntakeReasonCode.ATTESTATION_REFERENCE_MISSING)
        elif not _is_evidence_verified(ref):
            reasons.append(IntakeReasonCode.ATTESTATION_REFERENCE_UNVERIFIED)
        elif not declared or ref != declared:
            # Contract knows no provider attestation, or a different one —
            # a fabricated reference never verifies.
            reasons.append(IntakeReasonCode.ATTESTATION_REFERENCE_MISMATCH)
    elif origin == EvidenceOrigin.INSTITUTIONALLY_ATTESTED.value:
        anchor = _institutional_anchor(submission.dimension, contract)
        ref = submission.institutional_attestation_reference
        if not ref:
            reasons.append(
                IntakeReasonCode.INSTITUTIONAL_ATTESTATION_REFERENCE_MISSING)
        elif not _is_evidence_verified(ref):
            reasons.append(IntakeReasonCode.ATTESTATION_REFERENCE_UNVERIFIED)
        elif not anchor or ref != anchor:
            reasons.append(IntakeReasonCode.ATTESTATION_REFERENCE_MISMATCH)

    if provider_fact and origin not in ATTESTED_ORIGINS:
        # Public/system/local evidence never establishes provider-side
        # facts (identity, authority, dataset identity, continuity,
        # permission).
        if origin == EvidenceOrigin.PUBLIC_DOCUMENTATION.value:
            if submission.dimension == ContractDimension.ACCESS_AUTHORITY.value:
                reasons.append(
                    IntakeReasonCode.PUBLIC_DOCUMENTATION_NOT_AUTHORIZATION)
            else:
                reasons.append(
                    IntakeReasonCode.PROVIDER_FACT_WITHOUT_ATTESTATION)
        elif origin not in UNTRUSTED_ORIGINS:
            reasons.append(IntakeReasonCode.PROVIDER_FACT_WITHOUT_ATTESTATION)

    # ── 5. Attestation scope coverage (spec §4) ──
    if not submission.attestation_scope:
        scope_state = "missing"
        reasons.append(IntakeReasonCode.ATTESTATION_SCOPE_MISSING)
    elif not any(scope_covers_dimension(s, submission.dimension)
                 for s in submission.attestation_scope):
        scope_state = "mismatch"
        reasons.append(IntakeReasonCode.ATTESTATION_SCOPE_DIMENSION_MISMATCH)
    else:
        scope_state = "covered"

    # ── 6. Attestation freshness (spec §9) — authenticity is computed
    #    LAST for attested origins so it also covers the attestation
    #    timestamp checks (missing/invalid/future-dated ⇒ unverified). ──
    if attested_origin:
        if not submission.attestation_timestamp:
            reasons.append(IntakeReasonCode.ATTESTATION_TIMESTAMP_MISSING)
        else:
            att_ts = _parse_iso(submission.attestation_timestamp)
            if att_ts is None:
                reasons.append(IntakeReasonCode.ATTESTATION_TIMESTAMP_INVALID)
            elif cutoff is not None and att_ts > cutoff:
                reasons.append(IntakeReasonCode.ATTESTATION_FUTURE_DATED)
        authenticity = _authenticity_for_attestation(
            submission, contract, reasons)
    freshness_state = _freshness_state(submission, contract, reasons)

    # ── 7. Explicit usage permission (spec §10) ──
    if submission.dimension == ContractDimension.USAGE_PERMISSION.value:
        perm = submission.permission_state
        if perm == UsagePermissionState.UNKNOWN.value:
            reasons.append(IntakeReasonCode.USAGE_PERMISSION_UNKNOWN)
        elif perm == UsagePermissionState.RESTRICTED.value:
            reasons.append(IntakeReasonCode.USAGE_PERMISSION_RESTRICTED)
        elif perm not in EVALUATION_PERMISSION_STATES:
            reasons.append(
                IntakeReasonCode.USAGE_PERMISSION_NOT_COVERING_EVALUATION)
        if freshness_state == "expired":
            reasons.append(IntakeReasonCode.USAGE_PERMISSION_EXPIRED)

    # ── 8. Content claim (the actual provider fact) ──
    content_claim = _content_claim_state(submission, integrity, authenticity,
                                         scope_state, reasons)

    # ── 9. Overall validity ──
    reasons_sorted = tuple(sorted(set(reasons)))
    valid = (not reasons_sorted
             and integrity == VerificationState.INTEGRITY_VERIFIED.value
             and provenance == VerificationState.PROVENANCE_VERIFIED.value
             and content_claim == VerificationState.CONTENT_CLAIM_VERIFIED.value)
    binding_state = "failed" if binding_failed else "bound"

    report = ValidationReport(
        evidence_id=submission.evidence_id,
        valid=valid,
        integrity_state=integrity,
        authenticity_state=authenticity,
        provenance_state=provenance,
        content_claim_state=content_claim,
        freshness_state=freshness_state,
        binding_state=binding_state,
        scope_state=scope_state,
        reason_codes=reasons_sorted,
        report_hash="",
    )
    payload = {k: v for k, v in report.to_dict().items()
               if k != "report_hash"}
    return ValidationReport(
        **{**report.__dict__,
           "report_hash": _stable_hash(payload)})


def evidence_type_implies_provider(evidence_type: str, origin: str) -> bool:
    """An evidence type claiming provider/institution attestation while
    the origin cannot attest is impersonation (spec §3)."""
    low = (evidence_type or "").lower()
    return "attestation" in low and origin not in ATTESTED_ORIGINS


def _institutional_anchor(dimension: str,
                          contract: ExternalDatasetContract) -> str:
    """The contract-declared institutional reference the submission's
    attestation must match for its dimension (never invented)."""
    if dimension == ContractDimension.USAGE_PERMISSION.value:
        return contract.usage_permission.permission_reference
    if dimension == ContractDimension.ACCESS_AUTHORITY.value:
        return contract.access_authority.basis_reference
    return contract.provider_identity.authoritative_provenance_reference


def _authenticity_for_attestation(
    submission: EvidenceSubmission,
    contract: ExternalDatasetContract,
    reasons: list[str],
) -> str:
    """Authenticity of an ATTESTED origin requires the attestation
    reference to exist, be non-weak, and match the contract's declared
    reference — plus a valid attestation timestamp."""
    if (IntakeReasonCode.ATTESTATION_REFERENCE_MISSING in reasons
            or IntakeReasonCode.INSTITUTIONAL_ATTESTATION_REFERENCE_MISSING
            in reasons
            or IntakeReasonCode.ATTESTATION_REFERENCE_UNVERIFIED in reasons
            or IntakeReasonCode.ATTESTATION_REFERENCE_MISMATCH in reasons
            or IntakeReasonCode.ATTESTATION_TIMESTAMP_MISSING in reasons
            or IntakeReasonCode.ATTESTATION_TIMESTAMP_INVALID in reasons
            or IntakeReasonCode.ATTESTATION_FUTURE_DATED in reasons):
        return VerificationState.AUTHENTICITY_UNVERIFIED.value
    return VerificationState.AUTHENTICITY_VERIFIED.value


def _freshness_state(submission: EvidenceSubmission,
                     contract: ExternalDatasetContract,
                     reasons: list[str]) -> str:
    """Attestation effective-interval policy (spec §9):
    - no declared expiry ⇒ NO_EXPIRY_DECLARED, never auto-expired
    - inverted/unparseable interval ⇒ CONTRADICTED (reason)
    - explicitly expired vs the evidence cutoff ⇒ expired (reason)
    """
    start, until = submission.effective_from, submission.effective_until
    if not until and not start:
        return NO_EXPIRY_DECLARED
    t_start = _parse_iso(start) if start else None
    t_until = _parse_iso(until) if until else None
    if (start and t_start is None) or (until and t_until is None):
        reasons.append(IntakeReasonCode.ATTESTATION_INTERVAL_INVALID)
        return "interval_invalid"
    if t_start is not None and t_until is not None and t_start > t_until:
        reasons.append(IntakeReasonCode.ATTESTATION_INTERVAL_INVALID)
        return "interval_invalid"
    if t_until is None:
        return NO_EXPIRY_DECLARED
    cutoff = _parse_iso(contract.evidence_cutoff)
    if cutoff is not None and t_until <= cutoff:
        reasons.append(IntakeReasonCode.ATTESTATION_EXPIRED)
        return "expired"
    return "fresh"


def _content_claim_state(
    submission: EvidenceSubmission,
    integrity: str,
    authenticity: str,
    scope_state: str,
    reasons: list[str],
) -> str:
    """Content-claim verification — independent of integrity.

    A perfect hash never verifies a claim on its own (spec §2/§89–§92).
    """
    origin = submission.evidence_origin
    declared_ok = submission.evidence_status == EvidenceStatus.VERIFIED.value
    if not declared_ok:
        reasons.append(IntakeReasonCode.EVIDENCE_STATUS_NOT_VERIFIED)
        return VerificationState.CONTENT_CLAIM_UNVERIFIED.value
    if integrity != VerificationState.INTEGRITY_VERIFIED.value:
        return VerificationState.CONTENT_CLAIM_UNVERIFIED.value
    if origin in UNTRUSTED_ORIGINS:
        return VerificationState.CONTENT_CLAIM_UNVERIFIED.value
    if origin in ATTESTED_ORIGINS:
        if authenticity != VerificationState.AUTHENTICITY_VERIFIED.value:
            return VerificationState.CONTENT_CLAIM_UNVERIFIED.value
        if scope_state != "covered":
            return VerificationState.CONTENT_CLAIM_UNVERIFIED.value
        return VerificationState.CONTENT_CLAIM_VERIFIED.value
    # PUBLIC_DOCUMENTATION: content claim only ever follows the declared
    # status for NON provider-side facts; provider facts already failed
    # closed in the origin rules above.
    if submission.dimension in PROVIDER_FACT_DIMENSIONS:
        return VerificationState.CONTENT_CLAIM_UNVERIFIED.value
    if scope_state != "covered":
        return VerificationState.CONTENT_CLAIM_UNVERIFIED.value
    return VerificationState.CONTENT_CLAIM_VERIFIED.value


# ══════════════════════════════════════════════════════════════════════
# CONFLICT DETECTION (spec §5/§12) — contradictions never resolve
# ══════════════════════════════════════════════════════════════════════

def _conflict_index(
    submissions: Sequence[EvidenceSubmission],
) -> dict[str, tuple[str, ...]]:
    """Map evidence_id → conflict codes affecting it (deterministic)."""
    codes: dict[str, set[str]] = {}

    def add(eid: str, code: str) -> None:
        codes.setdefault(eid, set()).add(code)

    # Same evidence_id: identical hash ⇒ duplicate; different ⇒ CONTRADICTED.
    by_id: dict[str, list[EvidenceSubmission]] = {}
    for s in submissions:
        by_id.setdefault(s.evidence_id, []).append(s)
    for eid, group in sorted(by_id.items()):
        if len(group) > 1:
            # Compare the RECOMPUTED content hashes: an item whose stored
            # hash no longer matches its content conflicts with an item
            # under the same evidence_id (post-hash alteration).
            hashes = {s.evidence_hash_for() for s in group}
            code = (IntakeReasonCode.CONFLICTING_EVIDENCE_ID
                    if len(hashes) > 1
                    else IntakeReasonCode.DUPLICATE_EVIDENCE_ID)
            for s in group:
                add(s.evidence_id, code)

    # Same source_reference with different document hashes ⇒ CONTRADICTED.
    by_ref: dict[str, set[str]] = {}
    ref_ids: dict[str, set[str]] = {}
    for s in submissions:
        if s.source_reference:
            by_ref.setdefault(s.source_reference, set()).add(
                s.source_document_hash)
            ref_ids.setdefault(s.source_reference, set()).add(s.evidence_id)
    for ref, hashes in sorted(by_ref.items()):
        if len(hashes) > 1:
            for eid in ref_ids[ref]:
                add(eid, IntakeReasonCode.CONFLICTING_ARTIFACT_HASH)

    # Same document hash: identical content under different IDs ⇒
    # duplicate; same claim key with different values ⇒ CONTRADICTED.
    by_hash: dict[str, list[EvidenceSubmission]] = {}
    for s in submissions:
        if s.source_document_hash:
            by_hash.setdefault(s.source_document_hash, []).append(s)
    for doc_hash, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        ids = {s.evidence_id for s in group}
        signatures = {(s.dimension, s.claim_key, s.claim_value, s.statement)
                      for s in group}
        if len(ids) > 1 and len(signatures) == 1:
            for s in group:
                add(s.evidence_id, IntakeReasonCode.DUPLICATE_ARTIFACT_HASH)
        claim_values: dict[tuple[str, str], set[str]] = {}
        for s in group:
            if s.claim_key:
                claim_values.setdefault(
                    (s.dimension, s.claim_key), set()).add(s.claim_value)
        for (dim, key), values in sorted(claim_values.items()):
            if len(values) > 1:
                for s in group:
                    if s.claim_key == key and s.dimension == dim:
                        add(s.evidence_id,
                            IntakeReasonCode.CONFLICTING_DOCUMENT_CLAIMS)

    # Claim conflicts within the same dataset+dimension (spec §12):
    # periods, label provenance, timestamps, schema, continuity,
    # independence, contamination — contradictions stay visible.
    by_claim: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    for s in submissions:
        if s.claim_key:
            by_claim.setdefault(
                (s.dataset_id, s.dimension, s.claim_key),
                {}).setdefault(s.claim_value, set()).add(s.evidence_id)
    for (dataset_id, dim, key), value_map in sorted(by_claim.items()):
        if len(value_map) > 1:
            code = f"{dim}:{key}_contradicted"
            for ids in value_map.values():
                for eid in ids:
                    add(eid, code)

    # Conflicting effective intervals for the same dataset+dimension.
    by_interval: dict[tuple[str, str], dict[tuple[str, str], set[str]]] = {}
    for s in submissions:
        by_interval.setdefault(
            (s.dataset_id, s.dimension), {}).setdefault(
            (s.effective_from, s.effective_until), set()).add(s.evidence_id)
    for key, interval_map in sorted(by_interval.items()):
        if len(interval_map) > 1:
            for ids in interval_map.values():
                for eid in ids:
                    add(eid, IntakeReasonCode.CONFLICTING_EFFECTIVE_INTERVAL)

    return {eid: tuple(sorted(cs)) for eid, cs in sorted(codes.items())}


def detect_evidence_conflicts(
    submissions: Sequence[EvidenceSubmission],
) -> tuple[str, ...]:
    """Deterministic conflict codes across a submission set (spec §13).

    Contradictory evidence fails closed — never resolved toward
    whichever source is convenient.
    """
    index = _conflict_index(submissions)
    return tuple(sorted({code for codes in index.values() for code in codes}))


def conflicting_evidence_ids(
    submissions: Sequence[EvidenceSubmission],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Evidence_id → conflict codes (traceable per submission)."""
    return tuple(sorted(_conflict_index(submissions).items()))


def evidence_conflict_status(
    submission: EvidenceSubmission,
    submissions: Sequence[EvidenceSubmission],
) -> str | None:
    """Effective conflict-driven status override for one submission:
    CONTRADICTED for contradictions, UNVERIFIED for duplicates, None if
    uninvolved."""
    codes = _conflict_index(submissions).get(submission.evidence_id, ())
    if any(c in CONTRADICTED_CONFLICT_CODES or c.endswith("_contradicted")
           for c in codes):
        return EvidenceStatus.CONTRADICTED.value
    if any(c in DUPLICATE_CONFLICT_CODES for c in codes):
        return EvidenceStatus.UNVERIFIED.value
    return None


# ══════════════════════════════════════════════════════════════════════
# NORMALIZATION → PHASE 104 (single qualification authority)
# ══════════════════════════════════════════════════════════════════════

def to_phase104_item(
    submission: EvidenceSubmission,
    effective_status: str,
) -> EvidenceItem:
    """One submission → one Phase 104 EvidenceItem.

    Origin, provider, dataset, and context are copied VERBATIM — no
    evidence may silently change them (spec §6).  Only the STATUS is
    recomputed from intake validation (never upgraded).
    """
    return make_evidence_item(
        evidence_id=submission.evidence_id,
        dimension=submission.dimension,
        origin=submission.evidence_origin,
        status=effective_status,
        statement=submission.statement,
        provider_reference=submission.provider_id,
        dataset_reference=submission.dataset_id,
        context_id=submission.context_id,
        issued_at=submission.supplied_at,
        claim_key=submission.claim_key,
        claim_value=submission.claim_value,
        provenance_reference=submission.source_reference,
        supports_provider_fact=submission.evidence_origin in ATTESTED_ORIGINS,
    )


def normalize_submissions(
    contract: ExternalDatasetContract,
    submissions: Sequence[EvidenceSubmission],
) -> tuple[EvidenceItem, ...]:
    """Evidence normalization: validate + de-conflict + convert.

    - integrity-failed submissions are EXCLUDED (never contribute)
    - contradicted submissions keep status CONTRADICTED (visible)
    - duplicates are downgraded to UNVERIFIED
    - any other validation defect downgrades VERIFIED → UNVERIFIED
    - statuses are never upgraded
    """
    conflict_map = dict(_conflict_index(submissions))
    items: list[EvidenceItem] = []
    for s in submissions:
        report = validate_provider_evidence(s, contract)
        if report.integrity_state != VerificationState.INTEGRITY_VERIFIED.value:
            continue  # tampered / malformed metadata never contributes
        computed: str | None = None
        codes = conflict_map.get(s.evidence_id, ())
        if (any(c in CONTRADICTED_CONFLICT_CODES
                or c.endswith("_contradicted") for c in codes)
                or IntakeReasonCode.ATTESTATION_INTERVAL_INVALID
                in report.reason_codes):
            computed = EvidenceStatus.CONTRADICTED.value
        elif any(c in DUPLICATE_CONFLICT_CODES for c in codes):
            computed = EvidenceStatus.UNVERIFIED.value
        elif report.reason_codes:
            computed = EvidenceStatus.UNVERIFIED.value
        statuses = [x for x in (s.evidence_status, computed) if x]
        effective = worst_status(statuses)
        items.append(to_phase104_item(s, effective))
    return tuple(items)


def build_intake_package(
    contract: ExternalDatasetContract,
    submissions: Sequence[EvidenceSubmission],
) -> EvidencePackage:
    """Normalized Phase 104 evidence package from intake submissions."""
    return build_evidence_package(
        package_id=f"phase105-{contract.dataset_id.lower()}-intake",
        provider_reference=contract.provider_reference,
        dataset_reference=contract.dataset_id,
        evidence_version=contract.provider_identity.evidence_version,
        context_id=contract.context_id,
        package_created_at=EVIDENCE_AS_OF,
        items=normalize_submissions(contract, submissions),
    )


def qualify_submissions(
    contract: ExternalDatasetContract,
    submissions: Sequence[EvidenceSubmission],
) -> QualificationResult:
    """Intake → validation → normalization → PHASE 104 qualification.

    Phase 104 remains the ONLY qualification authority: this function
    returns Phase 104's QualificationResult unchanged.  It creates no
    RWV session, no promotion evidence, and no PromotionToken.
    """
    package = build_intake_package(contract, submissions)
    return evaluate_dataset_evidence(contract, package)


def qualification_path_signature() -> tuple[str, ...]:
    """Parameter surface of the Phase 104 qualification engine as reached
    through Phase 105 — proves no bypass parameters (force /
    allow_unverified / skip_validation / override / admin_override /
    bypass) exist anywhere on the intake→qualification path."""
    return tuple(inspect.signature(evaluate_dataset_evidence).parameters)


# ══════════════════════════════════════════════════════════════════════
# EVIDENCE CHAIN (spec §6) — every transformation traceable
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EvidenceChain:
    """Traceability record across the full evidence chain:
    EvidenceArtifact → EvidenceSubmission → AttestationScope →
    EvidenceDimension → Phase 104 Evidence → Dataset Qualification.
    """
    evidence_id: str
    artifact_hash: str
    submission_hash: str
    attestation_scope: tuple[str, ...]
    dimension: str
    phase104_content_hash: str
    origin: str
    provider_id: str
    dataset_id: str
    dataset_version: str


def trace_evidence_chain(
    submission: EvidenceSubmission,
    item: EvidenceItem | None = None,
) -> EvidenceChain:
    """Trace one submission through every chain stage.  Binding fields
    (origin, provider_id, dataset_id, dataset_version, scope) are read
    from the submission itself — the chain never silently rewrites them
    (spec §6: no origin/provider/dataset/version/scope changes)."""
    return EvidenceChain(
        evidence_id=submission.evidence_id,
        artifact_hash=submission.artifact().artifact_hash(),
        submission_hash=submission.evidence_hash,
        attestation_scope=submission.attestation_scope,
        dimension=submission.dimension,
        phase104_content_hash=item.content_hash if item is not None else "",
        origin=submission.evidence_origin,
        provider_id=submission.provider_id,
        dataset_id=submission.dataset_id,
        dataset_version=submission.dataset_version,
    )


# ══════════════════════════════════════════════════════════════════════
# EVIDENCE PACKAGE EXPORT (spec §16) — deterministic, metadata-only
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EvidencePackageExport:
    """Deterministic metadata-only evidence package export.

    Contains references, hashes, scopes, statuses, the contradiction
    state, and the Phase 104 qualification result — never credentials,
    API keys, passwords, PAN, unnecessary PII, raw transaction records,
    or model artifacts.
    """
    package_id: str
    package_version: str
    provider_references: tuple[str, ...]
    dataset_references: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    attestation_scopes: tuple[str, ...]
    evidence_origins: tuple[str, ...]
    evidence_statuses: tuple[str, ...]
    document_hashes: tuple[str, ...]
    contradiction_state: tuple[str, ...]
    qualification_result: str
    phase104_result_hash: str
    package_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("provider_references", "dataset_references",
                     "evidence_ids", "attestation_scopes",
                     "evidence_origins", "evidence_statuses",
                     "document_hashes", "contradiction_state"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.package_hash:
            object.__setattr__(self, "package_hash", self.package_hash_for())

    def payload(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("package_hash", None)
        return d

    def package_hash_for(self) -> str:
        return _stable_hash(self.payload())

    def verify_hash(self) -> bool:
        return self.package_hash == self.package_hash_for()

    def canonical_serialization(self) -> str:
        return canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for name in ("provider_references", "dataset_references",
                     "evidence_ids", "attestation_scopes",
                     "evidence_origins", "evidence_statuses",
                     "document_hashes", "contradiction_state"):
            d[name] = list(getattr(self, name))
        return d


def export_evidence_package(
    contract: ExternalDatasetContract,
    submissions: Sequence[EvidenceSubmission],
) -> EvidencePackageExport:
    """Deterministic metadata-only package export bound to the Phase 104
    qualification result (spec §16).  Order-insensitive: the same
    submissions in any order export to the same package hash."""
    result = qualify_submissions(contract, submissions)
    export = EvidencePackageExport(
        package_id=f"phase105-{contract.dataset_id.lower()}-intake",
        package_version=PACKAGE_VERSION,
        provider_references=tuple(sorted({s.provider_id for s in submissions})),
        dataset_references=tuple(sorted({s.dataset_id for s in submissions})),
        evidence_ids=tuple(sorted(s.evidence_id for s in submissions)),
        attestation_scopes=tuple(sorted(
            {sc for s in submissions for sc in s.attestation_scope})),
        evidence_origins=tuple(sorted(
            {s.evidence_origin for s in submissions})),
        evidence_statuses=tuple(sorted(
            {i.status for i in normalize_submissions(contract, submissions)})),
        document_hashes=tuple(sorted(
            {s.source_document_hash for s in submissions})),
        contradiction_state=detect_evidence_conflicts(submissions),
        qualification_result=result.qualification_state,
        phase104_result_hash=result.result_hash,
        package_hash="",
    )
    return export


__all__ = [
    "INTAKE_VERSION", "PACKAGE_VERSION", "NO_EXPIRY_DECLARED",
    "SUPPORTED_DOCUMENT_TYPES", "EVIDENCE_CHAIN_STAGES",
    "EVALUATION_PERMISSION_STATES", "SCOPE_DIMENSIONS",
    "CONTRADICTED_CONFLICT_CODES", "DUPLICATE_CONFLICT_CODES",
    "AttestationScope", "UsagePermissionState", "VerificationState",
    "IntakeReasonCode",
    "EvidenceArtifact", "EvidenceSubmission", "ValidationReport",
    "EvidenceChain", "EvidencePackageExport",
    "submit_provider_evidence", "validate_provider_evidence",
    "canonicalize_provider_evidence", "hash_provider_evidence",
    "detect_evidence_conflicts", "conflicting_evidence_ids",
    "evidence_conflict_status", "scope_covers_dimension",
    "evidence_type_implies_provider",
    "to_phase104_item", "normalize_submissions", "build_intake_package",
    "qualify_submissions", "qualification_path_signature",
    "trace_evidence_chain", "export_evidence_package",
]
