"""Phase 95: Provider evidence & dataset qualification gate.

Defines exactly what evidence must be supplied and verified before an
external dataset can become eligible for controlled Real-World Validation.

Does NOT acquire data, contact providers, modify models, retrain,
tune thresholds, or change REAL_WORLD_VALIDATION.

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


# ══════════════════════════════════════════════════════════════════════
# CONTROLLED ENUMS
# ══════════════════════════════════════════════════════════════════════

class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    MISSING = "missing"
    CONTRADICTED = "contradicted"
    NOT_APPLICABLE = "not_applicable"


class DatasetQualificationState(str, Enum):
    NOT_ASSESSED = "not_assessed"
    PENDING_EVIDENCE = "pending_evidence"
    BLOCKED = "blocked"
    QUALIFIED_FOR_CONTROLLED_RWV = "qualified_for_controlled_rwv"


class SourceType(str, Enum):
    PUBLIC_REAL_WORLD = "public_real_world"
    INSTITUTIONAL_REAL_WORLD = "institutional_real_world"
    SYNTHETIC = "synthetic"
    RESEARCH_DERIVED = "research_derived"
    UNKNOWN = "unknown"


class AccessClass(str, Enum):
    PUBLIC_DOWNLOADABLE = "public_downloadable"
    INSTITUTIONAL_ACCESS_REQUIRED = "institutional_access_required"
    CONFIDENTIAL = "confidential"
    UNKNOWN = "unknown"


# ══════════════════════════════════════════════════════════════════════
# QUALIFICATION DIMENSIONS
# ══════════════════════════════════════════════════════════════════════

QUALIFICATION_DIMENSIONS = (
    "provider_provenance",
    "dataset_identity",
    "access_authority",
    "label_provenance",
    "timestamp_semantics",
    "feature_schema",
    "entity_continuity",
    "independence",
    "contamination",
    "temporal_validity",
    "feature_compatibility",
    "leakage_risk",
    "evaluation_suitability",
    "usage_permission",
)


# ══════════════════════════════════════════════════════════════════════
# EVIDENCE QUALITY DETECTION
# ══════════════════════════════════════════════════════════════════════

# Prefixes that indicate evidence is NOT actually verified
_EVIDENCE_WEAK_PREFIXES = (
    "UNVERIFIED",
    "MISSING",
    "UNKNOWN",
    "NOT DOCUMENTED",
    "NOT PUBLICLY",
    "NOT CONFIRMED",
    "NOT AVAILABLE",
    "UNAVAILABLE",
    "TBD",
    "PENDING",
    "N/A",
)

_EVIDENCE_WEAK_SUFFIXES = (
    "UNVERIFIED",
    "NOT PUBLICLY DOCUMENTED",
    "NOT DOCUMENTED",
    "NOT CONFIRMED",
    "NOT AVAILABLE",
    "NO DATA-USE AGREEMENT IN PLACE",
    "NO ACCESS IN PLACE",
)


def _is_evidence_verified(text: str) -> bool:
    """Determine if an evidence string represents actual verified evidence.

    A non-empty string that contains 'UNVERIFIED', 'MISSING', 'UNKNOWN',
    etc. is NOT treated as verified evidence.  The mere presence of a
    reference string does not constitute verification.
    """
    if not text or not text.strip():
        return False
    upper = text.upper().strip()
    # Check prefix patterns
    for prefix in _EVIDENCE_WEAK_PREFIXES:
        if upper.startswith(prefix):
            return False
    # Check suffix / containment patterns
    for pattern in _EVIDENCE_WEAK_SUFFIXES:
        if pattern in upper:
            return False
    return True


def _check_evidence_field(text: str) -> EvidenceStatus:
    """Classify an evidence field into an EvidenceStatus."""
    if not text or not text.strip():
        return EvidenceStatus.MISSING
    upper = text.upper().strip()
    for prefix in ("CONTRADICTED", "INCONSISTENT", "REFUTED"):
        if upper.startswith(prefix) or f" {prefix} " in upper:
            return EvidenceStatus.CONTRADICTED
    if _is_evidence_verified(text):
        return EvidenceStatus.VERIFIED
    # Has text but it's weak/unverified
    for prefix in _EVIDENCE_WEAK_PREFIXES:
        if upper.startswith(prefix):
            if prefix in ("MISSING",):
                return EvidenceStatus.MISSING
            return EvidenceStatus.UNVERIFIED
    return EvidenceStatus.UNVERIFIED


# ══════════════════════════════════════════════════════════════════════
# PROVIDER EVIDENCE RECORD
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProviderEvidence:
    """Immutable provider evidence record for a dataset candidate."""
    provider_id: str
    dataset_id: str
    dataset_version: str
    provider_name: str
    dataset_title: str
    source_type: str  # SourceType value
    access_class: str  # AccessClass value
    ownership_or_controller: str
    acquisition_method: str
    acquisition_date: str
    documentation_reference: str
    schema_reference: str
    label_method_reference: str
    timestamp_reference: str
    feature_dictionary_reference: str
    entity_identifier_reference: str
    usage_permission_reference: str
    independence_statement: str
    contamination_statement: str
    provenance_statement: str
    evidence_version: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def compute_evidence_hash(evidence: ProviderEvidence) -> str:
    """Deterministic SHA-256 hash of provider evidence."""
    d = evidence.to_dict()
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# QUALIFICATION DIMENSION RESULT
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DimensionResult:
    """Result of one qualification dimension."""
    dimension: str
    status: str  # EvidenceStatus value
    evidence_reference: str
    is_required_for_rwv: bool
    is_blocking: bool
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ══════════════════════════════════════════════════════════════════════
# QUALIFICATION REPORT
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class QualificationReport:
    """Deterministic qualification report for a dataset candidate."""
    provider_id: str
    dataset_id: str
    dataset_version: str
    qualification_state: str
    overall_eligible: bool
    dimensions: list[dict]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    feature_compatibility_result: str
    label_provenance_result: str
    temporal_result: str
    leakage_result: str
    contamination_result: str
    usage_authorization_result: str
    evidence_hash: str
    policy_version: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["blockers"] = list(d["blockers"])
        d["warnings"] = list(d["warnings"])
        return d


# ══════════════════════════════════════════════════════════════════════
# QUALIFICATION ENGINE
# ══════════════════════════════════════════════════════════════════════

def qualify_dataset(evidence: ProviderEvidence) -> QualificationReport:
    """Deterministic qualification assessment for a dataset candidate.

    Every mandatory dimension must pass. Missing/unknown/UNVERIFIED
    evidence is treated as BLOCKED, never as verified.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    dimensions: list[DimensionResult] = []

    def _dim(name: str, status: EvidenceStatus, ref: str,
             required: bool, details: str) -> DimensionResult:
        is_blocking = required and status in (
            EvidenceStatus.MISSING,
            EvidenceStatus.UNVERIFIED,
            EvidenceStatus.CONTRADICTED,
        )
        if is_blocking:
            blockers.append(f"{name}: {status.value} -- {details}")
        dim = DimensionResult(name, status.value, ref, required, is_blocking, details)
        dimensions.append(dim)
        return dim

    # A. Provider provenance
    prov_status = _check_evidence_field(evidence.provenance_statement)
    _dim("provider_provenance", prov_status,
         evidence.documentation_reference, True,
         "provider name and provenance statement must be documented and verified")

    # B. Dataset identity
    id_status = EvidenceStatus.VERIFIED if (
        evidence.dataset_id and evidence.dataset_version and
        _is_evidence_verified(evidence.schema_reference)
    ) else EvidenceStatus.MISSING
    _dim("dataset_identity", id_status,
         evidence.schema_reference, True,
         "dataset_id, version, and schema must be documented and verified")

    # C. Access authority
    access_status = _check_evidence_field(evidence.usage_permission_reference)
    _dim("access_authority", access_status,
         evidence.usage_permission_reference, True,
         "access authority and usage permission must be documented and verified")

    # D. Label provenance
    label_status = _check_evidence_field(evidence.label_method_reference)
    _dim("label_provenance", label_status,
         evidence.label_method_reference, True,
         "label generation method must be documented and verified")

    # E. Timestamp semantics
    ts_status = _check_evidence_field(evidence.timestamp_reference)
    _dim("timestamp_semantics", ts_status,
         evidence.timestamp_reference, True,
         "timestamp semantics must be documented and verified")

    # F. Feature schema
    schema_field = _check_evidence_field(evidence.schema_reference)
    feat_dict_field = _check_evidence_field(evidence.feature_dictionary_reference)
    if schema_field == EvidenceStatus.VERIFIED and feat_dict_field == EvidenceStatus.VERIFIED:
        schema_status = EvidenceStatus.VERIFIED
    elif schema_field in (EvidenceStatus.MISSING, EvidenceStatus.CONTRADICTED):
        schema_status = schema_field
    elif feat_dict_field in (EvidenceStatus.MISSING, EvidenceStatus.CONTRADICTED):
        schema_status = feat_dict_field
    else:
        schema_status = EvidenceStatus.UNVERIFIED
    _dim("feature_schema", schema_status,
         evidence.feature_dictionary_reference, True,
         "schema and feature dictionary must be documented and verified")

    # G. Entity continuity
    ent_status = _check_evidence_field(evidence.entity_identifier_reference)
    _dim("entity_continuity", ent_status,
         evidence.entity_identifier_reference, True,
         "entity identifier semantics and stability must be documented and verified")

    # H. Independence
    indep_status = _check_evidence_field(evidence.independence_statement)
    _dim("independence", indep_status,
         evidence.independence_statement, True,
         "independence from training data must be documented and verified")

    # I. Contamination
    cont_status = _check_evidence_field(evidence.contamination_statement)
    _dim("contamination", cont_status,
         evidence.contamination_statement, True,
         "contamination status must be documented and verified")

    # J. Temporal validity
    temporal_status = ts_status  # depends on timestamp documentation
    _dim("temporal_validity", temporal_status,
         evidence.timestamp_reference, True,
         "temporal validity requires verified timestamp semantics")

    # K. Feature compatibility (requires schema)
    feat_compat_status = schema_status  # requires schema documentation
    _dim("feature_compatibility", feat_compat_status,
         evidence.feature_dictionary_reference, True,
         "feature compatibility requires verified schema documentation")

    # L. Leakage risk (requires both timestamp and label docs)
    if ts_status == EvidenceStatus.VERIFIED and label_status == EvidenceStatus.VERIFIED:
        leakage_status = EvidenceStatus.VERIFIED
    elif ts_status in (EvidenceStatus.MISSING, EvidenceStatus.CONTRADICTED):
        leakage_status = ts_status
    elif label_status in (EvidenceStatus.MISSING, EvidenceStatus.CONTRADICTED):
        leakage_status = label_status
    else:
        leakage_status = EvidenceStatus.UNVERIFIED
    _dim("leakage_risk", leakage_status,
         f"{evidence.timestamp_reference}; {evidence.label_method_reference}", True,
         "leakage risk assessment requires verified timestamp and label documentation")

    # M. Evaluation suitability
    id_ok = bool(evidence.dataset_id and evidence.dataset_version)
    if id_ok and label_status == EvidenceStatus.VERIFIED and schema_status == EvidenceStatus.VERIFIED:
        suit_status = EvidenceStatus.VERIFIED
    elif label_status in (EvidenceStatus.MISSING, EvidenceStatus.CONTRADICTED):
        suit_status = label_status
    elif schema_status in (EvidenceStatus.MISSING, EvidenceStatus.CONTRADICTED):
        suit_status = schema_status
    else:
        suit_status = EvidenceStatus.UNVERIFIED
    _dim("evaluation_suitability", suit_status,
         evidence.schema_reference, True,
         "evaluation suitability requires verified identity, labels, and schema")

    # N. Usage permission
    perm_status = _check_evidence_field(evidence.usage_permission_reference)
    _dim("usage_permission", perm_status,
         evidence.usage_permission_reference, True,
         "usage permission must be documented and verified")

    # Determine qualification state
    if blockers:
        state = DatasetQualificationState.BLOCKED.value
        eligible = False
    elif all(d.status == EvidenceStatus.VERIFIED.value for d in dimensions):
        state = DatasetQualificationState.QUALIFIED_FOR_CONTROLLED_RWV.value
        eligible = True
    else:
        state = DatasetQualificationState.PENDING_EVIDENCE.value
        eligible = False

    # Compute evidence hash
    evidence_hash = compute_evidence_hash(evidence)

    # Build sub-results
    schema_ok = (schema_status == EvidenceStatus.VERIFIED)
    label_ok = (label_status == EvidenceStatus.VERIFIED)
    ts_ok = (ts_status == EvidenceStatus.VERIFIED)
    leakage_ok = (leakage_status == EvidenceStatus.VERIFIED)
    cont_ok = (cont_status == EvidenceStatus.VERIFIED)
    perm_ok = (perm_status == EvidenceStatus.VERIFIED)

    return QualificationReport(
        provider_id=evidence.provider_id,
        dataset_id=evidence.dataset_id,
        dataset_version=evidence.dataset_version,
        qualification_state=state,
        overall_eligible=eligible,
        dimensions=[d.to_dict() for d in dimensions],
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        feature_compatibility_result="requires_actual_schema_check" if schema_ok else "pending_schema_verification",
        label_provenance_result="verified" if label_ok else "unverified",
        temporal_result="verified" if ts_ok else "unverified",
        leakage_result="verified" if leakage_ok else "unverified",
        contamination_result="verified" if cont_ok else "unverified",
        usage_authorization_result="verified" if perm_ok else "unverified",
        evidence_hash=evidence_hash,
        policy_version="phase95_v1",
    )


# ══════════════════════════════════════════════════════════════════════
# KNOWN CANDIDATE EVIDENCE RECORDS
# ══════════════════════════════════════════════════════════════════════

WORLDLINE_2017_EVIDENCE = ProviderEvidence(
    provider_id="WORLDLINE",
    dataset_id="WORLDLINE_ECOM_2017_NAG",
    dataset_version="unknown_pending_provider_schema",
    provider_name="Worldline",
    dataset_title="Worldline Belgium Real-World E-Commerce Fraud Dataset (2017)",
    source_type=SourceType.INSTITUTIONAL_REAL_WORLD.value,
    access_class=AccessClass.CONFIDENTIAL.value,
    ownership_or_controller="Worldline",
    acquisition_method="institutional_research_access",
    acquisition_date="not_acquired",
    documentation_reference="NAG paper (published research)",
    schema_reference="UNVERIFIED -- exact schema not publicly documented",
    label_method_reference="Published: human investigators with expert rules",
    timestamp_reference="Published: timestamps documented; absolute vs relative UNVERIFIED",
    feature_dictionary_reference="UNVERIFIED -- feature dictionary not publicly documented",
    entity_identifier_reference="Published: customer IDs, terminal IDs documented; stability UNVERIFIED",
    usage_permission_reference="MISSING -- no data-use agreement in place",
    independence_statement="Independent from PS-14 training data (no contamination evidence)",
    contamination_statement="No contamination evidence found in inspected PS-14 artifacts",
    provenance_statement="Published research documents real-world e-commerce transactions Jan-Jul 2017",
    evidence_version="phase95_v1",
)

WORLDLINE_2018_EVIDENCE = ProviderEvidence(
    provider_id="WORLDLINE",
    dataset_id="WORLDLINE_ONLINE_2018",
    dataset_version="unknown_pending_provider_schema",
    provider_name="Worldline",
    dataset_title="Worldline Online International Transactions (2018)",
    source_type=SourceType.INSTITUTIONAL_REAL_WORLD.value,
    access_class=AccessClass.CONFIDENTIAL.value,
    ownership_or_controller="Worldline",
    acquisition_method="institutional_research_access",
    acquisition_date="not_acquired",
    documentation_reference="Published research",
    schema_reference="UNVERIFIED -- exact schema not publicly documented",
    label_method_reference="UNVERIFIED -- label methodology not fully documented",
    timestamp_reference="UNVERIFIED -- timestamp semantics not confirmed",
    feature_dictionary_reference="UNVERIFIED -- feature dictionary not publicly documented",
    entity_identifier_reference="UNVERIFIED -- entity identifiers not confirmed",
    usage_permission_reference="MISSING -- no data-use agreement in place",
    independence_statement="Independent from PS-14 training data (no contamination evidence)",
    contamination_statement="No contamination evidence found",
    provenance_statement="Published research documents online international transactions May-Sep 2018",
    evidence_version="phase95_v1",
)

NOVATTI_EVIDENCE = ProviderEvidence(
    provider_id="NOVATTI",
    dataset_id="NOVATTI",
    dataset_version="unknown_pending_provider_schema",
    provider_name="Novatti Group Ltd",
    dataset_title="Novatti Group Ltd Real-World Transaction Dataset",
    source_type=SourceType.INSTITUTIONAL_REAL_WORLD.value,
    access_class=AccessClass.CONFIDENTIAL.value,
    ownership_or_controller="Novatti Group Ltd",
    acquisition_method="institutional_research_access",
    acquisition_date="not_acquired",
    documentation_reference="Published research on Novatti fraud detection",
    schema_reference="UNVERIFIED -- exact schema not publicly documented",
    label_method_reference="Published: chargeback-derived fraud labels",
    timestamp_reference="Published: Jan 2023 - Jun 2025 temporal range",
    feature_dictionary_reference="UNVERIFIED -- feature dictionary not publicly documented",
    entity_identifier_reference="Published: customer/card identifiers substantially removed",
    usage_permission_reference="MISSING -- no data-use agreement in place",
    independence_statement="Independent from PS-14 training data",
    contamination_statement="No contamination evidence found",
    provenance_statement="Published research documents 126K real merchant transactions with 394 fraud cases",
    evidence_version="phase95_v1",
)

IEEE_CIS_EVIDENCE = ProviderEvidence(
    provider_id="IEEE_CIS",
    dataset_id="IEEE_CIS",
    dataset_version="competition_2019",
    provider_name="Vesta Corporation / IEEE CIS",
    dataset_title="IEEE-CIS Fraud Detection Dataset",
    source_type=SourceType.PUBLIC_REAL_WORLD.value,
    access_class=AccessClass.PUBLIC_DOWNLOADABLE.value,
    ownership_or_controller="Vesta Corporation",
    acquisition_method="public_kaggle_download",
    acquisition_date="not_acquired",
    documentation_reference="Kaggle competition page, IEEE DataPort",
    schema_reference="VERIFIED -- schema documented in competition and EDA writeups",
    label_method_reference="UNVERIFIED -- exact label methodology not fully documented",
    timestamp_reference="VERIFIED -- TransactionDT is relative timedelta (NOT absolute clock time)",
    feature_dictionary_reference="VERIFIED -- Vesta-engineered features documented as masked",
    entity_identifier_reference="VERIFIED -- card1-card6 present but anonymisation scheme undocumented",
    usage_permission_reference="Kaggle competition terms (not institutional research DUA)",
    independence_statement="Independent from PS-14 training data",
    contamination_statement="No contamination evidence found",
    provenance_statement="Real e-commerce transactions from Vesta Corporation",
    evidence_version="phase95_v1",
)


KNOWN_CANDIDATES: dict[str, ProviderEvidence] = {
    "WORLDLINE_ECOM_2017_NAG": WORLDLINE_2017_EVIDENCE,
    "WORLDLINE_ONLINE_2018": WORLDLINE_2018_EVIDENCE,
    "NOVATTI": NOVATTI_EVIDENCE,
    "IEEE_CIS": IEEE_CIS_EVIDENCE,
}


# ══════════════════════════════════════════════════════════════════════
# BATCH QUALIFICATION
# ══════════════════════════════════════════════════════════════════════

def qualify_all_known_candidates() -> dict[str, QualificationReport]:
    """Qualify all known dataset candidates."""
    return {name: qualify_dataset(ev) for name, ev in KNOWN_CANDIDATES.items()}
