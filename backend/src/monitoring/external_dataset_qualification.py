"""Phase 104: External Dataset Qualification Engine.

Deterministic, offline evaluation of an external dataset evidence contract
against its provider evidence package:

    evaluate_dataset_evidence(contract, evidence) -> QualificationResult

Validates: provider identity, evidence origin, access authority, dataset
identity, label provenance, temporal semantics, feature compatibility,
entity continuity, independence, contamination, usage permission,
contradictions, and evidence integrity.

Fail-closed by construction:
  - UNKNOWN / MISSING / CONTRADICTED mandatory evidence blocks qualification
  - SYSTEM_GENERATED / LOCAL_SYNTHETIC / LOCAL_RESEARCH evidence can never
    prove provider facts (origin is separate from status)
  - No bypass parameters exist (no force / allow_unverified /
    skip_validation / override / admin_override)
  - Deterministic reason codes and a deterministic result hash

This function does NOT: access the network, load external data, load model
artifacts, use credentials, execute RWV, create promotion evidence, create
PromotionToken, or promote a model.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
PROMOTION: PROMOTION_GATE_REQUIRED
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.monitoring.external_dataset_contract import (
    AMBIGUOUS_TIMEZONE_TOKENS,
    CANONICAL_TRANSFORMATION,
    CONTRACT_VERSION,
    CONTAMINATION_FIELDS,
    ENTITY_FACET_ATTR,
    EVIDENCE_POLICY_VERSION,
    MANDATORY_DIMENSIONS,
    OVERLAP_REASON,
    ATTESTED_ORIGINS,
    PROVIDER_FACT_DIMENSIONS,
    TRUSTED_LABEL_ORIGINS,
    UNTRUSTED_ORIGINS,
    AccessAuthority,
    ContractDimension,
    DatasetQualificationState,
    EvidenceOrigin,
    EvidencePackage,
    EvidenceStatus,
    ExternalDatasetContract,
    FeatureAvailability,
    FeaturePreflightItem,
    detect_contradictions,
    detect_missing_evidence,
    preflight_feature_compatibility,
    verify_package_integrity,
    worst_status,
)

# Reuse the Phase 95 identifier authorities — never duplicated here.
from src.monitoring.rwv_readiness_audit import MODEL_ID, RELEASE_ID
from src.monitoring.rwv_reproducibility import (
    NATIVE_FEATURE_VERSION,
    PREPROCESSING_HASH,
    RULE_HASH,
)


# ══════════════════════════════════════════════════════════════════════
# REASON CODES (deterministic, sorted at output)
# ══════════════════════════════════════════════════════════════════════

class ReasonCode:
    """Deterministic qualification reason codes."""
    # Contract-level
    CONTRACT_NOT_ASSESSED = "contract_not_assessed"
    MODEL_IDENTITY_MISMATCH = "model_identity_mismatch"
    RELEASE_IDENTITY_MISMATCH = "release_identity_mismatch"
    # Provider identity
    PROVIDER_IDENTITY_MISSING = "provider_identity_missing"
    PROVIDER_IDENTITY_UNVERIFIED = "provider_identity_unverified"
    PROVIDER_IDENTITY_UNKNOWN = "provider_identity_unknown"
    # Evidence origin
    EVIDENCE_ORIGIN_UNTRUSTED = "evidence_origin_untrusted"
    EVIDENCE_ORIGIN_UNKNOWN = "evidence_origin_unknown"
    EVIDENCE_ORIGIN_IMPERSONATION = "evidence_origin_impersonation"
    PROVIDER_FACT_WITHOUT_ATTESTATION = "provider_fact_without_attestation"
    # Access authority
    ACCESS_AUTHORITY_UNKNOWN = "access_authority_unknown"
    ACCESS_AUTHORITY_INFERRED_FROM_PUBLIC = (
        "access_authority_inferred_from_public_availability")
    ACCESS_AUTHORITY_UNVERIFIED = "access_authority_unverified"
    # Dataset identity
    DATASET_IDENTITY_MISSING = "dataset_identity_missing"
    DATASET_IDENTITY_UNVERIFIED = "dataset_identity_unverified"
    # Labels
    LABEL_PROVENANCE_MISSING = "label_provenance_missing"
    LABEL_PROVENANCE_UNKNOWN = "label_provenance_unknown"
    LABEL_PROVENANCE_UNTRUSTED = "label_provenance_untrusted"
    SYNTHETIC_LABELS = "synthetic_labels"
    LABEL_LEAKAGE = "label_leakage"
    FUTURE_LABELS = "future_labels"
    # Temporal
    TIMESTAMP_SEMANTICS_MISSING = "timestamp_semantics_missing"
    TIMESTAMP_SEMANTICS_AMBIGUOUS = "timestamp_semantics_ambiguous"
    TEMPORAL_INVERSION = "temporal_inversion"
    FUTURE_INFORMATION_LEAKAGE = "future_information_leakage"
    # Features
    FEATURE_CONTRACT_VERSION_MISMATCH = "feature_contract_version_mismatch"
    NATIVE_FEATURE_VERSION_MISMATCH = "native_feature_version_mismatch"
    FEATURE_ORDER_ALTERED = "feature_order_altered"
    NONCANONICAL_TRANSFORMATION = "noncanonical_transformation"
    DIRECT_DOMAIN_FEATURE_INFERENCE = "direct_domain_feature_inference"
    FEATURE_UNAVAILABLE = "feature_unavailable"
    FEATURE_AMBIGUOUS = "feature_ambiguous"
    FEATURE_LEAKAGE_RISK = "feature_leakage_risk"
    FEATURE_INCOMPATIBLE = "feature_incompatible"
    # Entity continuity
    ENTITY_CONTINUITY_UNKNOWN = "entity_continuity_unknown"
    ENTITY_CONTINUITY_UNSTABLE = "entity_continuity_unstable"
    ENTITY_CONTINUITY_MISSING_HISTORY = "entity_continuity_missing_history"
    ENTITY_CONTINUITY_INSUFFICIENT = "entity_continuity_insufficient"
    # Independence
    INDEPENDENCE_NOT_ESTABLISHED = "independence_not_established"
    LOCAL_DATASET_NOT_REAL_WORLD = "local_dataset_not_real_world"
    # Contamination
    CONTAMINATION_UNKNOWN = "contamination_unknown"
    # Usage permission
    USAGE_PERMISSION_MISSING = "usage_permission_missing"
    USAGE_PERMISSION_AMBIGUOUS = "usage_permission_ambiguous"
    USAGE_PERMISSION_EXPIRED = "usage_permission_expired"
    USAGE_PERMISSION_PROVIDER_MISMATCH = "usage_permission_provider_mismatch"
    # Evidence integrity
    EVIDENCE_INTEGRITY_FAILED = "evidence_integrity_failed"
    EVIDENCE_ITEM_TAMPERED = "evidence_item_hash_mismatch"
    EVIDENCE_MISSING = "evidence_missing"
    EVIDENCE_CONTRADICTED = "evidence_contradicted"
    EVIDENCE_STALE = "evidence_stale"
    EVIDENCE_REPLAYED = "evidence_replayed"
    EVIDENCE_CROSS_CONTEXT = "evidence_cross_context"
    EVIDENCE_VERSION_MISMATCH = "evidence_version_mismatch"
    # Qualification outcome
    QUALIFIED_FOR_CONTROLLED_RWV = "qualified_for_controlled_rwv"
    NOT_QUALIFIED = "not_qualified"


# ══════════════════════════════════════════════════════════════════════
# DIMENSION RESULT / QUALIFICATION RESULT
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DimensionOutcome:
    """Deterministic result for one mandatory qualification dimension."""
    dimension: str
    status: str                # EvidenceStatus value
    reason_codes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    origins: tuple[str, ...]
    blocking: bool

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["reason_codes"] = list(self.reason_codes)
        d["evidence_ids"] = list(self.evidence_ids)
        d["origins"] = list(self.origins)
        return d


@dataclass(frozen=True)
class QualificationResult:
    """Deterministic qualification outcome for one dataset candidate."""
    contract_version: str
    evidence_policy_version: str
    provider_reference: str
    dataset_id: str
    qualification_state: str       # DatasetQualificationState value
    qualified: bool
    dimensions: tuple[DimensionOutcome, ...]
    reason_codes: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    contradictory_evidence: tuple[str, ...]
    feature_preflight: tuple[FeaturePreflightItem, ...]
    origin_classification: tuple[tuple[str, int], ...]
    contract_hash: str
    package_hash: str
    result_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["dimensions"] = [x.to_dict() for x in self.dimensions]
        d["reason_codes"] = list(self.reason_codes)
        d["missing_evidence"] = list(self.missing_evidence)
        d["contradictory_evidence"] = list(self.contradictory_evidence)
        d["feature_preflight"] = [x.to_dict() for x in self.feature_preflight]
        d["origin_classification"] = [list(x) for x in self.origin_classification]
        return d


def _stable_hash(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# EVIDENCE COLLECTION & ORIGIN GATING
# ══════════════════════════════════════════════════════════════════════

def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _collect(
    contract: ExternalDatasetContract,
    package: EvidencePackage | None,
    dimension: str,
) -> tuple[list[Any], list[str]]:
    """Integrity-passing, context-bound evidence items for one dimension.

    Items from a different dataset/provider/context, stale items, future
    (replayed) items, and items whose evidence version disagrees with the
    contract are excluded and reported as reason codes.
    """
    reasons: list[str] = []
    if package is None:
        return [], reasons
    cutoff = _parse_iso(contract.evidence_cutoff)
    valid_from = _parse_iso(contract.evidence_valid_from)
    items = []
    for item in package.items:
        if item.dimension != dimension:
            continue
        # Tampered items are isolated: they never contribute status.
        if not item.verify_hash():
            reasons.append(ReasonCode.EVIDENCE_ITEM_TAMPERED)
            continue
        # Cross-context evidence never contributes.
        if item.dataset_reference != contract.dataset_id:
            reasons.append(ReasonCode.EVIDENCE_CROSS_CONTEXT)
            continue
        if item.provider_reference != contract.provider_reference:
            reasons.append(ReasonCode.EVIDENCE_CROSS_CONTEXT)
            continue
        if item.context_id != contract.context_id:
            reasons.append(ReasonCode.EVIDENCE_CROSS_CONTEXT)
            continue
        # Stale / replayed (future-dated) evidence never contributes.
        issued = _parse_iso(item.issued_at)
        if issued is None:
            reasons.append(ReasonCode.EVIDENCE_STALE)
            continue
        if valid_from is not None and issued < valid_from:
            reasons.append(ReasonCode.EVIDENCE_STALE)
            continue
        if cutoff is not None and issued > cutoff:
            reasons.append(ReasonCode.EVIDENCE_REPLAYED)
            continue
        items.append(item)
    return items, reasons


def _origin_status(items: list[Any], dimension: str) -> tuple[str | None, list[str]]:
    """Effective status contributed by evidence origins for a dimension.

    Returns None when no items exist.  Untrusted origins can never verify a
    dimension; provider-fact dimensions additionally require attested origin.
    """
    reasons: list[str] = []
    if not items:
        return None, reasons
    statuses: list[str] = []
    for item in items:
        origin = item.origin
        status = item.status
        if origin in UNTRUSTED_ORIGINS:
            # Local/system evidence never proves provider (or any) facts.
            if origin == EvidenceOrigin.UNKNOWN.value:
                reasons.append(ReasonCode.EVIDENCE_ORIGIN_UNKNOWN)
            elif item.supports_provider_fact or origin in (
                EvidenceOrigin.LOCAL_SYNTHETIC.value,
                EvidenceOrigin.LOCAL_RESEARCH.value,
            ):
                reasons.append(ReasonCode.EVIDENCE_ORIGIN_IMPERSONATION)
            else:
                reasons.append(ReasonCode.EVIDENCE_ORIGIN_UNTRUSTED)
            # Downgrade: untrusted origin can never be VERIFIED.
            if status == EvidenceStatus.VERIFIED.value:
                status = EvidenceStatus.UNVERIFIED.value
        elif dimension in PROVIDER_FACT_DIMENSIONS and origin not in ATTESTED_ORIGINS:
            # Public documentation never proves provider-side facts.
            reasons.append(ReasonCode.PROVIDER_FACT_WITHOUT_ATTESTATION)
            if status == EvidenceStatus.VERIFIED.value:
                status = EvidenceStatus.UNVERIFIED.value
        statuses.append(status)
    return worst_status(statuses), reasons


# ══════════════════════════════════════════════════════════════════════
# PER-DIMENSION CONTRACT CHECKS
# ══════════════════════════════════════════════════════════════════════

def _check_provider_identity(contract: ExternalDatasetContract) -> tuple[str, list[str]]:
    p = contract.provider_identity
    reasons: list[str] = []
    required = (
        p.provider_name, p.provider_organization, p.dataset_owner_controller,
        p.authoritative_provenance_reference, p.provider_attestation_reference,
        p.evidence_timestamp, p.evidence_version,
    )
    if not p.provider_name or not p.provider_organization:
        reasons.append(ReasonCode.PROVIDER_IDENTITY_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    if not p.provider_attestation_reference:
        reasons.append(ReasonCode.PROVIDER_IDENTITY_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    lower = " ".join(required).lower()
    if "unknown" in lower:
        reasons.append(ReasonCode.PROVIDER_IDENTITY_UNKNOWN)
        return EvidenceStatus.MISSING.value, reasons
    if any(not v for v in required):
        reasons.append(ReasonCode.PROVIDER_IDENTITY_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    return EvidenceStatus.VERIFIED.value, reasons


def _check_access_authority(contract: ExternalDatasetContract) -> tuple[str, list[str]]:
    a = contract.access_authority
    reasons: list[str] = []
    if a.authority == AccessAuthority.UNKNOWN.value:
        reasons.append(ReasonCode.ACCESS_AUTHORITY_UNKNOWN)
        return EvidenceStatus.MISSING.value, reasons
    if not a.basis_reference:
        # Never infer authorization merely from public downloadability.
        if a.publicly_downloadable:
            reasons.append(ReasonCode.ACCESS_AUTHORITY_INFERRED_FROM_PUBLIC)
        else:
            reasons.append(ReasonCode.ACCESS_AUTHORITY_UNVERIFIED)
        return EvidenceStatus.MISSING.value, reasons
    lower = a.basis_reference.lower()
    if "unknown" in lower or "unverified" in lower or "missing" in lower:
        reasons.append(ReasonCode.ACCESS_AUTHORITY_UNVERIFIED)
        return EvidenceStatus.UNVERIFIED.value, reasons
    return EvidenceStatus.VERIFIED.value, reasons


def _check_dataset_identity(contract: ExternalDatasetContract) -> tuple[str, list[str]]:
    d = contract.dataset_identity
    reasons: list[str] = []
    if not d.canonical_dataset_id or not d.version:
        reasons.append(ReasonCode.DATASET_IDENTITY_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    if not d.schema_descriptor:
        reasons.append(ReasonCode.DATASET_IDENTITY_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    if not d.record_count_evidence or d.record_count <= 0:
        reasons.append(ReasonCode.DATASET_IDENTITY_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    if not d.collection_period or not d.extraction_sampling_description:
        reasons.append(ReasonCode.DATASET_IDENTITY_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    fields = " ".join([
        d.canonical_dataset_id, d.version, d.schema_descriptor,
        d.collection_period, d.extraction_sampling_description,
    ]).lower()
    if "unknown" in fields or "unverified" in fields:
        reasons.append(ReasonCode.DATASET_IDENTITY_UNVERIFIED)
        return EvidenceStatus.UNVERIFIED.value, reasons
    return EvidenceStatus.VERIFIED.value, reasons


def _check_label_provenance(contract: ExternalDatasetContract) -> tuple[str, list[str]]:
    lab = contract.label_provenance
    reasons: list[str] = []
    required = (
        lab.label_definition, lab.label_origin, lab.labeling_authority,
        lab.labeling_process, lab.confirmation_timing,
        lab.label_availability_time,
    )
    if not all(required):
        reasons.append(ReasonCode.LABEL_PROVENANCE_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    if lab.labels_retrospective is None or lab.independently_adjudicated is None:
        reasons.append(ReasonCode.LABEL_PROVENANCE_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    if lab.suitable_for_evaluation is None:
        reasons.append(ReasonCode.LABEL_PROVENANCE_UNKNOWN)
        return EvidenceStatus.MISSING.value, reasons
    if not lab.suitable_for_evaluation:
        reasons.append(ReasonCode.LABEL_PROVENANCE_UNKNOWN)
        return EvidenceStatus.MISSING.value, reasons
    origin_lower = lab.label_origin.lower()
    if "synthetic" in origin_lower or "simulated" in origin_lower:
        reasons.append(ReasonCode.SYNTHETIC_LABELS)
        return EvidenceStatus.MISSING.value, reasons
    if "leak" in origin_lower or "future" in origin_lower:
        reasons.append(ReasonCode.LABEL_LEAKAGE)
        return EvidenceStatus.MISSING.value, reasons
    if lab.label_origin.lower() not in TRUSTED_LABEL_ORIGINS:
        reasons.append(ReasonCode.LABEL_PROVENANCE_UNTRUSTED)
        return EvidenceStatus.MISSING.value, reasons
    # Labels must be available at or before the evaluation cutoff.
    avail = _parse_iso(lab.label_availability_time)
    cutoff = _parse_iso(contract.temporal_semantics.evaluation_cutoff)
    if avail is None or cutoff is None:
        reasons.append(ReasonCode.LABEL_PROVENANCE_UNKNOWN)
        return EvidenceStatus.MISSING.value, reasons
    if avail > cutoff:
        reasons.append(ReasonCode.FUTURE_LABELS)
        return EvidenceStatus.MISSING.value, reasons
    joined = " ".join(required).lower()
    if "unknown" in joined or "unverified" in joined:
        reasons.append(ReasonCode.LABEL_PROVENANCE_UNKNOWN)
        return EvidenceStatus.UNVERIFIED.value, reasons
    return EvidenceStatus.VERIFIED.value, reasons


def _check_temporal(contract: ExternalDatasetContract) -> tuple[str, list[str]]:
    t = contract.temporal_semantics
    reasons: list[str] = []
    if not t.event_timestamp_semantics or not t.timezone_semantics:
        reasons.append(ReasonCode.TIMESTAMP_SEMANTICS_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    if not t.evaluation_cutoff or not t.temporal_ordering_rules:
        reasons.append(ReasonCode.TIMESTAMP_SEMANTICS_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    if not t.observation_time:
        reasons.append(ReasonCode.TIMESTAMP_SEMANTICS_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    tz_lower = t.timezone_semantics.lower()
    if any(tok in tz_lower for tok in AMBIGUOUS_TIMEZONE_TOKENS):
        reasons.append(ReasonCode.TIMESTAMP_SEMANTICS_AMBIGUOUS)
        return EvidenceStatus.MISSING.value, reasons
    ts_lower = t.event_timestamp_semantics.lower()
    if "unknown" in ts_lower or "ambiguous" in ts_lower or "relative" in ts_lower:
        # Relative-only timestamps (e.g. IEEE-CIS TransactionDT) are
        # ambiguous for absolute-time feature derivation.
        reasons.append(ReasonCode.TIMESTAMP_SEMANTICS_AMBIGUOUS)
        return EvidenceStatus.MISSING.value, reasons
    if t.contains_future_information:
        reasons.append(ReasonCode.FUTURE_INFORMATION_LEAKAGE)
        return EvidenceStatus.MISSING.value, reasons
    # Where applicable verify effective_at <= observed_at.
    if t.effective_at and t.observed_at:
        eff = _parse_iso(t.effective_at)
        obs = _parse_iso(t.observed_at)
        if eff is None or obs is None:
            # Unparseable timestamps are ambiguous semantics, rejected.
            reasons.append(ReasonCode.TIMESTAMP_SEMANTICS_AMBIGUOUS)
            return EvidenceStatus.MISSING.value, reasons
        if eff > obs:
            reasons.append(ReasonCode.TEMPORAL_INVERSION)
            return EvidenceStatus.MISSING.value, reasons
    return EvidenceStatus.VERIFIED.value, reasons


def _check_feature_compatibility(
    contract: ExternalDatasetContract,
    preflight: tuple[FeaturePreflightItem, ...],
) -> tuple[str, list[str]]:
    fc = contract.feature_compatibility
    reasons: list[str] = []
    from src.monitoring.feature_contract import ML_FEATURE_VERSION
    from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES

    if fc.feature_contract_version != ML_FEATURE_VERSION:
        reasons.append(ReasonCode.FEATURE_CONTRACT_VERSION_MISMATCH)
    if fc.native_feature_version != NATIVE_FEATURE_VERSION:
        reasons.append(ReasonCode.NATIVE_FEATURE_VERSION_MISMATCH)
    if fc.transformation != CANONICAL_TRANSFORMATION:
        reasons.append(ReasonCode.NONCANONICAL_TRANSFORMATION)
    if tuple(fc.native_feature_order) != tuple(ALTMAN_NATIVE_FEATURES):
        reasons.append(ReasonCode.FEATURE_ORDER_ALTERED)
    if fc.direct_domain_feature_inference:
        reasons.append(ReasonCode.DIRECT_DOMAIN_FEATURE_INFERENCE)

    unavailable = [
        i.feature for i in preflight
        if i.availability == FeatureAvailability.UNAVAILABLE.value
    ]
    ambiguous = [
        i.feature for i in preflight
        if i.availability == FeatureAvailability.AMBIGUOUS.value
    ]
    leakage = [i.feature for i in preflight if i.leakage_risk]
    if unavailable:
        reasons.append(ReasonCode.FEATURE_UNAVAILABLE)
    if ambiguous:
        reasons.append(ReasonCode.FEATURE_AMBIGUOUS)
    if leakage:
        reasons.append(ReasonCode.FEATURE_LEAKAGE_RISK)

    if reasons:
        # Any preflight/version/order/transformation failure is blocking.
        reasons.append(ReasonCode.FEATURE_INCOMPATIBLE)
        return EvidenceStatus.MISSING.value, reasons
    return EvidenceStatus.VERIFIED.value, reasons


def _check_entity_continuity(contract: ExternalDatasetContract) -> tuple[str, list[str]]:
    e = contract.entity_continuity
    reasons: list[str] = []
    unstable = False
    missing_history = False
    unknown = False
    for facet, attr in ENTITY_FACET_ATTR.items():
        value = getattr(e, attr)
        if value is True:
            continue
        if value is False:
            if attr in ("historical_transaction", "temporal_history", "transaction"):
                missing_history = True
            else:
                unstable = True
        else:
            unknown = True
    # Anonymized identifiers are never assumed to preserve continuity.
    if e.identifiers_anonymized and not (unstable or missing_history):
        # Claimed stability without explicit continuity evidence → unknown.
        any_true = any(getattr(e, a) is True for a in ENTITY_FACET_ATTR.values())
        if any_true:
            unknown = True

    if unstable:
        reasons.append(ReasonCode.ENTITY_CONTINUITY_UNSTABLE)
    if missing_history:
        reasons.append(ReasonCode.ENTITY_CONTINUITY_MISSING_HISTORY)
    if unknown:
        reasons.append(ReasonCode.ENTITY_CONTINUITY_UNKNOWN)
    if unstable or missing_history:
        reasons.append(ReasonCode.ENTITY_CONTINUITY_INSUFFICIENT)
    if reasons:
        status = (EvidenceStatus.MISSING.value if (unstable or missing_history)
                  else EvidenceStatus.UNVERIFIED.value)
        return status, reasons
    return EvidenceStatus.VERIFIED.value, reasons


def _check_independence(contract: ExternalDatasetContract) -> tuple[str, list[str]]:
    ind = contract.independence
    reasons: list[str] = []
    if ind.dataset_origin in ("local_synthetic", "synthetic", "local_research",
                              "research_derived"):
        reasons.append(ReasonCode.LOCAL_DATASET_NOT_REAL_WORLD)
        return EvidenceStatus.MISSING.value, reasons
    if ind.dataset_origin != "real_world_external":
        reasons.append(ReasonCode.INDEPENDENCE_NOT_ESTABLISHED)
        return EvidenceStatus.MISSING.value, reasons
    # Overlap in any protected axis blocks.
    for key in ind.known_overlaps:
        reasons.append(OVERLAP_REASON.get(key, f"{key}_overlap"))
    if ind.known_overlaps:
        reasons.append(ReasonCode.INDEPENDENCE_NOT_ESTABLISHED)
        return EvidenceStatus.MISSING.value, reasons
    from src.monitoring.external_dataset_contract import INDEPENDENCE_KEYS
    confirmed = set(ind.confirmed_independent)
    if set(INDEPENDENCE_KEYS) - confirmed:
        reasons.append(ReasonCode.INDEPENDENCE_NOT_ESTABLISHED)
        return EvidenceStatus.MISSING.value, reasons
    return EvidenceStatus.VERIFIED.value, reasons


def _check_contamination(contract: ExternalDatasetContract) -> tuple[str, list[str]]:
    c = contract.contamination
    reasons: list[str] = []
    detected: list[str] = []
    unknown: list[str] = []
    for field_name in CONTAMINATION_FIELDS:
        value = getattr(c, field_name)
        if value is True:
            detected.append(f"{field_name}_detected")
        elif value is None:
            unknown.append(field_name)
    if detected:
        # Contamination present ⇒ fails closed.
        reasons.extend(detected)
        return EvidenceStatus.MISSING.value, reasons
    if unknown:
        # UNKNOWN contamination status fails closed.
        reasons.append(ReasonCode.CONTAMINATION_UNKNOWN)
        return EvidenceStatus.MISSING.value, reasons
    return EvidenceStatus.VERIFIED.value, reasons


def _check_usage_permission(contract: ExternalDatasetContract) -> tuple[str, list[str]]:
    u = contract.usage_permission
    reasons: list[str] = []
    if not u.permission_reference or not u.permission_scope:
        reasons.append(ReasonCode.USAGE_PERMISSION_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    if u.permitted_use is None:
        reasons.append(ReasonCode.USAGE_PERMISSION_AMBIGUOUS)
        return EvidenceStatus.MISSING.value, reasons
    if not u.permitted_use:
        reasons.append(ReasonCode.USAGE_PERMISSION_MISSING)
        return EvidenceStatus.MISSING.value, reasons
    if not u.valid_from or not u.valid_until:
        reasons.append(ReasonCode.USAGE_PERMISSION_AMBIGUOUS)
        return EvidenceStatus.MISSING.value, reasons
    valid_until = _parse_iso(u.valid_until)
    cutoff = _parse_iso(contract.evidence_cutoff)
    if valid_until is None:
        reasons.append(ReasonCode.USAGE_PERMISSION_AMBIGUOUS)
        return EvidenceStatus.MISSING.value, reasons
    if cutoff is not None and valid_until < cutoff:
        reasons.append(ReasonCode.USAGE_PERMISSION_EXPIRED)
        return EvidenceStatus.MISSING.value, reasons
    # Permission must be held by the very provider the contract names.
    holder = u.permission_holder.strip().lower()
    expected = contract.provider_identity.provider_name.strip().lower()
    if not holder or holder != expected:
        reasons.append(ReasonCode.USAGE_PERMISSION_PROVIDER_MISMATCH)
        return EvidenceStatus.MISSING.value, reasons
    lower = " ".join([u.permission_reference, u.permission_scope]).lower()
    if "unknown" in lower or "ambiguous" in lower or "unverified" in lower:
        reasons.append(ReasonCode.USAGE_PERMISSION_AMBIGUOUS)
        return EvidenceStatus.UNVERIFIED.value, reasons
    return EvidenceStatus.VERIFIED.value, reasons


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

_CONTRACT_CHECKS = {
    ContractDimension.PROVIDER_IDENTITY.value: _check_provider_identity,
    ContractDimension.ACCESS_AUTHORITY.value: _check_access_authority,
    ContractDimension.DATASET_IDENTITY.value: _check_dataset_identity,
    ContractDimension.LABEL_PROVENANCE.value: _check_label_provenance,
    ContractDimension.TEMPORAL_SEMANTICS.value: _check_temporal,
    ContractDimension.ENTITY_CONTINUITY.value: _check_entity_continuity,
    ContractDimension.INDEPENDENCE.value: _check_independence,
    ContractDimension.CONTAMINATION.value: _check_contamination,
    ContractDimension.USAGE_PERMISSION.value: _check_usage_permission,
}


def evaluate_dataset_evidence(
    contract: ExternalDatasetContract,
    evidence: EvidencePackage | None,
) -> QualificationResult:
    """Deterministic qualification of an external dataset evidence contract.

    Fail-closed: every mandatory dimension must be VERIFIED with trusted,
    attested, integrity-valid evidence.  No bypass parameters exist — the
    signature accepts exactly (contract, evidence).
    """
    preflight = preflight_feature_compatibility(contract)
    missing = detect_missing_evidence(contract, evidence)
    contradictions = detect_contradictions(contract, evidence)
    integrity_ok, integrity_reasons = verify_package_integrity(evidence)

    # Origin classification summary (origin is separate from status).
    origin_counts: dict[str, int] = {}
    if evidence is not None:
        for item in evidence.items:
            origin_counts[item.origin] = origin_counts.get(item.origin, 0) + 1
    origin_classification = tuple(sorted(origin_counts.items()))

    contract_hash = contract.contract_hash()
    package_hash = evidence.package_hash if evidence is not None else ""

    # Evidence-version binding: the package must carry the same evidence
    # version the contract's provider identity attests to.
    version_ok = (
        evidence is None
        or not contract.provider_identity.evidence_version
        or evidence.evidence_version == contract.provider_identity.evidence_version
    )

    outcomes: list[DimensionOutcome] = []
    all_reasons: set[str] = set()

    # ── Contract-level identity binding (Phase 94/96 authorities) ──
    if not contract.assessed:
        all_reasons.add(ReasonCode.CONTRACT_NOT_ASSESSED)
    if contract.model_id and contract.model_id != MODEL_ID:
        all_reasons.add(ReasonCode.MODEL_IDENTITY_MISMATCH)
    if contract.release_id and contract.release_id != RELEASE_ID:
        all_reasons.add(ReasonCode.RELEASE_IDENTITY_MISMATCH)

    # ── Evidence integrity (fail closed before any dimension) ──
    if not integrity_ok:
        all_reasons.add(ReasonCode.EVIDENCE_INTEGRITY_FAILED)
        all_reasons.update(integrity_reasons)
    if evidence is None:
        all_reasons.add(ReasonCode.EVIDENCE_MISSING)
    if not version_ok:
        all_reasons.add(ReasonCode.EVIDENCE_VERSION_MISMATCH)
    for dim in missing:
        all_reasons.add(ReasonCode.EVIDENCE_MISSING)
    for code in contradictions:
        all_reasons.add(code)
        all_reasons.add(ReasonCode.EVIDENCE_CONTRADICTED)

    # ── Per-dimension evaluation ──
    for dimension in MANDATORY_DIMENSIONS:
        dim_reasons: list[str] = []
        # 1. Contract-side check
        if dimension == ContractDimension.FEATURE_COMPATIBILITY.value:
            c_status, c_reasons = _check_feature_compatibility(contract, preflight)
        else:
            check = _CONTRACT_CHECKS.get(dimension)
            if check is None:
                c_status, c_reasons = EvidenceStatus.MISSING.value, []
            else:
                c_status, c_reasons = check(contract)
        dim_reasons.extend(c_reasons)
        # 2. Evidence-side origin-gated status
        items, collect_reasons = _collect(contract, evidence, dimension)
        dim_reasons.extend(collect_reasons)
        e_status, origin_reasons = _origin_status(items, dimension)
        dim_reasons.extend(origin_reasons)
        if e_status is None:
            # No integrity-passing, context-valid evidence for this
            # dimension — including the case where every candidate item
            # was filtered (cross-context / stale / replayed / tampered).
            # Contract-only status must never carry a dimension alone.
            e_status = EvidenceStatus.MISSING.value
            dim_reasons.append(ReasonCode.EVIDENCE_MISSING)
        # 3. Contradictions for this dimension (codes are dimension-tagged
        #    as "<dimension>:<code>").
        dim_contra = [c for c in contradictions if c.startswith(f"{dimension}:")]
        contradicted_items = [
            i for i in items if i.status == EvidenceStatus.CONTRADICTED.value
        ]
        if dim_contra or contradicted_items:
            c_status = EvidenceStatus.CONTRADICTED.value
            dim_reasons.extend(dim_contra)
            if contradicted_items:
                dim_reasons.append(ReasonCode.EVIDENCE_CONTRADICTED)
        # 4. Deterministic precedence fold
        statuses = [s for s in (c_status, e_status) if s]
        status = worst_status(statuses)
        blocking = status != EvidenceStatus.VERIFIED.value
        if blocking and status == EvidenceStatus.MISSING.value and not dim_reasons:
            dim_reasons.append(ReasonCode.EVIDENCE_MISSING)
        unique = tuple(sorted(set(dim_reasons)))
        all_reasons.update(unique)
        outcomes.append(DimensionOutcome(
            dimension=dimension,
            status=status,
            reason_codes=unique,
            evidence_ids=tuple(i.evidence_id for i in items),
            origins=tuple(sorted({i.origin for i in items})),
            blocking=blocking,
        ))

    # ── Qualification state (Phase 95 precedent: MISSING / UNVERIFIED /
    #    CONTRADICTED on a mandatory dimension ⇒ BLOCKED; integrity
    #    failure or absent evidence ⇒ BLOCKED; a single mandatory
    #    failing dimension prevents qualification) ──
    any_missing = [o.dimension for o in outcomes
                   if o.status == EvidenceStatus.MISSING.value]
    any_contradicted = [o.dimension for o in outcomes
                        if o.status == EvidenceStatus.CONTRADICTED.value]
    any_unverified_dim = any(
        o.status != EvidenceStatus.VERIFIED.value for o in outcomes)
    contract_blocking = bool(all_reasons & {
        ReasonCode.CONTRACT_NOT_ASSESSED,
        ReasonCode.MODEL_IDENTITY_MISMATCH,
        ReasonCode.RELEASE_IDENTITY_MISMATCH,
    })

    if not contract.assessed:
        state = DatasetQualificationState.NOT_ASSESSED.value
    elif (contract_blocking or not integrity_ok or not version_ok
          or evidence is None or any_unverified_dim):
        state = DatasetQualificationState.BLOCKED.value
    elif all_reasons:
        # All dimensions VERIFIED but stale/replayed/cross-context items
        # were filtered ⇒ pending evidence review, never qualified.
        state = DatasetQualificationState.PENDING_EVIDENCE.value
    else:
        state = DatasetQualificationState.QUALIFIED_FOR_CONTROLLED_RWV.value

    qualified = state == DatasetQualificationState.QUALIFIED_FOR_CONTROLLED_RWV.value
    reason_codes = tuple(sorted(all_reasons))
    if qualified:
        reason_codes = (ReasonCode.QUALIFIED_FOR_CONTROLLED_RWV,)

    result_hash = _stable_hash({
        "contract_version": CONTRACT_VERSION,
        "evidence_policy_version": EVIDENCE_POLICY_VERSION,
        "dataset_id": contract.dataset_id,
        "state": state,
        "dimensions": [(o.dimension, o.status) for o in outcomes],
        "reason_codes": list(reason_codes),
        "contract_hash": contract_hash,
        "package_hash": package_hash,
        "preflight": [(i.feature, i.availability) for i in preflight],
    })

    return QualificationResult(
        contract_version=CONTRACT_VERSION,
        evidence_policy_version=EVIDENCE_POLICY_VERSION,
        provider_reference=contract.provider_reference,
        dataset_id=contract.dataset_id,
        qualification_state=state,
        qualified=qualified,
        dimensions=tuple(outcomes),
        reason_codes=reason_codes,
        missing_evidence=tuple(sorted(set(missing) | set(any_missing))),
        contradictory_evidence=tuple(sorted(set(contradictions)
                                             | set(any_contradicted))),
        feature_preflight=preflight,
        origin_classification=origin_classification,
        contract_hash=contract_hash,
        package_hash=package_hash,
        result_hash=result_hash,
    )


def evaluate_known_candidates() -> dict[str, QualificationResult]:
    """Evaluate all known dataset candidates — none can qualify today."""
    from src.monitoring.external_dataset_contract import (
        KNOWN_DATASET_IDS,
        build_known_candidate_contract_and_evidence,
    )
    results: dict[str, QualificationResult] = {}
    for dataset_id in KNOWN_DATASET_IDS:
        contract, evidence = build_known_candidate_contract_and_evidence(dataset_id)
        results[dataset_id] = evaluate_dataset_evidence(contract, evidence)
    return results


def qualification_signature() -> tuple[str, ...]:
    """Parameter names of evaluate_dataset_evidence — proves no bypass
    parameters (force / allow_unverified / skip_validation / override /
    admin_override) exist."""
    return tuple(inspect.signature(evaluate_dataset_evidence).parameters)


__all__ = [
    "ReasonCode", "DimensionOutcome", "QualificationResult",
    "evaluate_dataset_evidence", "evaluate_known_candidates",
    "qualification_signature",
]
