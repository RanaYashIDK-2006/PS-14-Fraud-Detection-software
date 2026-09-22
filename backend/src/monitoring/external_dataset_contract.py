"""Phase 104: External Dataset Evidence Contract & Evidence Intake Boundary.

Defines the exact evidence contract an external dataset/provider must satisfy
before a dataset could enter the existing Phase 96 controlled RWV process.

This phase creates the CONTRACT, not the external dataset:
  - NO external dataset acquired
  - NO provider contacted
  - NO real-world validation performed
  - NO model modified / retrained / tuned
  - NO promotion path created, NO gate weakened
  - NO duplicate promotion / RWV / trust-policy authority

Evidence origin is ALWAYS separate from evidence status.  SYSTEM_GENERATED,
LOCAL_SYNTHETIC and LOCAL_RESEARCH evidence can never prove provider facts
and never impersonates provider evidence.  UNKNOWN fails closed.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from src.monitoring.provider_evidence import (
    DatasetQualificationState,
    EvidenceStatus,
    KNOWN_CANDIDATES,
    _check_evidence_field,
)
from src.monitoring.feature_contract import ML_FEATURE_VERSION, ML_FEATURE_ORDER
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES

# ══════════════════════════════════════════════════════════════════════
# VERSIONS
# ══════════════════════════════════════════════════════════════════════

CONTRACT_VERSION = "phase104_v1"
EVIDENCE_POLICY_VERSION = "phase104_evidence_policy_v1"

# Canonical pipeline (NOT redefined here — bound to authoritative sources):
#   21 domain/runtime features (ML_FEATURE_ORDER)
#           ↓  existing deterministic map_raw_to_native()
#   48 native features (ALTMAN_NATIVE_FEATURES)
#           ↓
#   Altman-Native ensemble
DOMAIN_FEATURE_COUNT = len(ML_FEATURE_ORDER)   # 21
NATIVE_FEATURE_COUNT = len(ALTMAN_NATIVE_FEATURES)  # 48
CANONICAL_TRANSFORMATION = "map_raw_to_native"

# Evidence intake window for known candidates (contract parameter, not
# evidence): items outside [valid_from, cutoff] are stale / replayed.
EVIDENCE_VALID_FROM = "2026-01-01T00:00:00+00:00"
EVIDENCE_CUTOFF = "2026-09-22T00:00:00+00:00"

# Fixed assessment instant for deterministic known-candidate packages.
EVIDENCE_AS_OF = "2026-09-22T00:00:00+00:00"


# ══════════════════════════════════════════════════════════════════════
# CONTROLLED ENUMS  (status semantics reuse Phase 95/81)
# ══════════════════════════════════════════════════════════════════════

class EvidenceOrigin(str, Enum):
    """Where evidence comes from — ALWAYS separate from evidence status."""
    PROVIDER_ATTESTED = "provider_attested"
    INSTITUTIONALLY_ATTESTED = "institutionally_attested"
    PUBLIC_DOCUMENTATION = "public_documentation"
    LOCAL_SYNTHETIC = "local_synthetic"
    LOCAL_RESEARCH = "local_research"
    SYSTEM_GENERATED = "system_generated"
    UNKNOWN = "unknown"


class AccessAuthority(str, Enum):
    PUBLIC_DOCUMENTED = "public_documented"
    PROVIDER_AUTHORIZED = "provider_authorized"
    INSTITUTIONAL_AGREEMENT = "institutional_agreement"
    CONFIDENTIAL_PROVIDER_ACCESS = "confidential_provider_access"
    UNKNOWN = "unknown"


class FeatureAvailability(str, Enum):
    """Per-native-feature classification (spec F)."""
    DIRECTLY_AVAILABLE = "directly_available"
    DETERMINISTICALLY_DERIVABLE = "deterministically_derivable"
    UNAVAILABLE = "unavailable"
    LEAKAGE_RISK = "leakage_risk"
    AMBIGUOUS = "ambiguous"


# Qualification states reuse Phase 95 (DatasetQualificationState):
#   not_assessed / pending_evidence / blocked / qualified_for_controlled_rwv
# Evidence status reuses Phase 95 (EvidenceStatus):
#   verified / unverified / missing / contradicted / not_applicable

# Deterministic precedence: CONTRADICTED > MISSING > UNVERIFIED > VERIFIED
# (NOT_APPLICABLE is excluded unless it is the only status present.)
STATUS_PRECEDENCE: tuple[str, ...] = (
    EvidenceStatus.CONTRADICTED.value,
    EvidenceStatus.MISSING.value,
    EvidenceStatus.UNVERIFIED.value,
    EvidenceStatus.VERIFIED.value,
)
_STATUS_RANK = {s: i for i, s in enumerate(reversed(STATUS_PRECEDENCE))}
# reversed => verified=0 ... contradicted=3  (higher = worse)


def worst_status(statuses: list[str]) -> str:
    """Deterministic precedence fold over evidence statuses."""
    usable = [s for s in statuses if s != EvidenceStatus.NOT_APPLICABLE.value]
    if not usable:
        return EvidenceStatus.MISSING.value
    return max(usable, key=lambda s: _STATUS_RANK.get(s, 3))


# ══════════════════════════════════════════════════════════════════════
# QUALIFICATION DIMENSIONS (spec A–J)
# ══════════════════════════════════════════════════════════════════════

class ContractDimension(str, Enum):
    PROVIDER_IDENTITY = "provider_identity"
    ACCESS_AUTHORITY = "access_authority"
    DATASET_IDENTITY = "dataset_identity"
    LABEL_PROVENANCE = "label_provenance"
    TEMPORAL_SEMANTICS = "temporal_semantics"
    FEATURE_COMPATIBILITY = "feature_compatibility"
    ENTITY_CONTINUITY = "entity_continuity"
    INDEPENDENCE = "independence"
    CONTAMINATION = "contamination"
    USAGE_PERMISSION = "usage_permission"


MANDATORY_DIMENSIONS: tuple[str, ...] = tuple(d.value for d in ContractDimension)

# Dimensions asserting PROVIDER-side facts: only provider/institution
# attestation can verify them.  Public documentation alone never proves
# identity, authority, permission, or identifier stability.
PROVIDER_FACT_DIMENSIONS: frozenset[str] = frozenset({
    ContractDimension.PROVIDER_IDENTITY.value,
    ContractDimension.ACCESS_AUTHORITY.value,
    ContractDimension.DATASET_IDENTITY.value,
    ContractDimension.ENTITY_CONTINUITY.value,
    ContractDimension.USAGE_PERMISSION.value,
})

# Origins that can NEVER verify any dimension.
UNTRUSTED_ORIGINS: frozenset[str] = frozenset({
    EvidenceOrigin.LOCAL_SYNTHETIC.value,
    EvidenceOrigin.LOCAL_RESEARCH.value,
    EvidenceOrigin.SYSTEM_GENERATED.value,
    EvidenceOrigin.UNKNOWN.value,
})

# Origins that can assert provider-side facts.
ATTESTED_ORIGINS: frozenset[str] = frozenset({
    EvidenceOrigin.PROVIDER_ATTESTED.value,
    EvidenceOrigin.INSTITUTIONALLY_ATTESTED.value,
})

# Origins that impersonate provider evidence if they claim provider facts.
IMPERSONATING_ORIGINS: frozenset[str] = UNTRUSTED_ORIGINS


# ══════════════════════════════════════════════════════════════════════
# FEATURE SOURCE REQUIREMENTS (spec F/G — binds to the 48 native features)
# ══════════════════════════════════════════════════════════════════════

REQUIREMENT_KEYS: tuple[str, ...] = (
    "amount", "absolute_time", "channel", "errors", "zip", "state", "mcc",
    "merchant_id", "city_id", "card_id",
    "user_history", "card_history", "merchant_history",
    "labeled_user_history", "labeled_merchant_history", "labeled_city_history",
)

# Every canonical native feature → the raw/history sources it requires.
# (Derived from src/privacy_layer/native_features.derive_native_features
#  and src/risk_engine.altman_native_ensemble.map_raw_to_native.)
FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "amt": ("amount",),
    "log_amt": ("amount",),
    "amt_sq": ("amount",),
    "hr": ("absolute_time",),
    "mn": ("absolute_time",),
    "dow": ("absolute_time",),
    "Month": ("absolute_time",),
    "Day": ("absolute_time",),
    "hour_sin": ("absolute_time",),
    "hour_cos": ("absolute_time",),
    "is_night": ("absolute_time",),
    "is_business_hours": ("absolute_time",),
    "chip": ("channel",),
    "is_online": ("channel",),
    "is_swipe": ("channel",),
    "err": ("errors",),
    "has_zip": ("zip",),
    "has_state": ("state",),
    "is_online_or_no_state": ("channel", "state"),
    "mcc": ("mcc",),
    "mcc_high": ("mcc",),
    "mcc_restaurant": ("mcc",),
    "mcc_gas": ("mcc",),
    "mcc_grocery": ("mcc",),
    "mcc_travel": ("mcc",),
    "mcc_online": ("mcc",),
    "merchant_id": ("merchant_id",),
    "city_id": ("city_id",),
    "card_id": ("card_id",),
    "user_tx_count": ("user_history",),
    "card_tx_count": ("card_history",),
    "user_avg_amt": ("user_history", "amount"),
    "amt_vs_user_avg": ("user_history", "amount"),
    "amt_zscore": ("user_history", "amount"),
    "merch_tx_count": ("merchant_history",),
    "user_merchant_diversity": ("user_history", "merchant_id"),
    "user_city_diversity": ("user_history", "city_id"),
    "user_fraud_rate": ("labeled_user_history",),
    "merch_fraud_rate": ("labeled_merchant_history",),
    "city_fraud_rate": ("labeled_city_history",),
    "high_amt": ("amount",),
    "very_high_amt": ("amount",),
    "amt_x_hr": ("amount", "absolute_time"),
    "amt_x_mcc": ("amount", "mcc"),
    "amt_x_chip": ("amount", "channel"),
    "amt_x_online": ("amount", "channel"),
    "amt_x_night": ("amount", "absolute_time"),
    "user_merch_count": ("user_history", "merchant_id"),
}

BLOCKING_AVAILABILITY: frozenset[str] = frozenset({
    FeatureAvailability.UNAVAILABLE.value,
    FeatureAvailability.LEAKAGE_RISK.value,
    FeatureAvailability.AMBIGUOUS.value,
})


# ══════════════════════════════════════════════════════════════════════
# CONTRACT SUB-STRUCTURES (immutable / frozen — spec A–J)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProviderIdentityClaim:
    """A. Provider identity — unknown identity fails closed."""
    provider_name: str = ""
    provider_organization: str = ""
    dataset_owner_controller: str = ""
    authoritative_provenance_reference: str = ""
    provider_attestation_reference: str = ""
    evidence_timestamp: str = ""
    evidence_version: str = ""


@dataclass(frozen=True)
class AccessAuthorityClaim:
    """B. Access authority — UNKNOWN never qualifies; public download alone
    is never inferred as authorization."""
    authority: str = AccessAuthority.UNKNOWN.value
    basis_reference: str = ""
    publicly_downloadable: bool = False


@dataclass(frozen=True)
class DatasetIdentityClaim:
    """C. Dataset identity — missing values are never invented."""
    canonical_dataset_id: str = ""
    provider_dataset_id: str = ""
    version: str = ""
    collection_period: str = ""
    release_period: str = ""
    fingerprint: str = ""
    fingerprint_available: bool = False
    schema_descriptor: str = ""
    record_count: int = 0
    record_count_evidence: str = ""
    extraction_sampling_description: str = ""


@dataclass(frozen=True)
class LabelProvenanceClaim:
    """D. Label provenance — unknown label provenance fails closed."""
    label_definition: str = ""
    label_origin: str = ""           # human_investigation / chargebacks / ...
    labeling_authority: str = ""
    labeling_process: str = ""
    confirmation_timing: str = ""
    labels_retrospective: bool | None = None
    independently_adjudicated: bool | None = None
    suitable_for_evaluation: bool | None = None
    label_availability_time: str = ""  # ISO; compared vs evaluation cutoff


@dataclass(frozen=True)
class TemporalSemanticsClaim:
    """E. Temporal semantics — ambiguous timestamps rejected;
    verifies effective_at <= observed_at where applicable.
    (Collection period lives in DatasetIdentityClaim; label availability
     time lives in LabelProvenanceClaim and is cross-checked here.)"""
    event_timestamp_semantics: str = ""
    timezone_semantics: str = ""
    observation_time: str = ""
    evaluation_cutoff: str = ""
    temporal_ordering_rules: str = ""
    effective_at: str = ""
    observed_at: str = ""
    contains_future_information: bool = False


@dataclass(frozen=True)
class FeatureCompatibilityClaim:
    """F. Feature compatibility — binds to the canonical 21→48 pipeline.
    No dataset-specific transformations, no direct 21-feature inference."""
    feature_contract_version: str = ""
    native_feature_version: str = ""
    transformation: str = CANONICAL_TRANSFORMATION
    native_feature_order: tuple[str, ...] = ()
    direct_domain_feature_inference: bool = False
    # (requirement_key, FeatureAvailability.value) — undeclared requirement
    # keys are treated as AMBIGUOUS (fail closed, never imputed).
    requirement_availability: tuple[tuple[str, str], ...] = ()
    leakage_risk_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntityContinuityClaim:
    """G. Entity continuity — anonymized identifiers are NEVER assumed to
    preserve continuity.  True=stable established, False=unstable,
    None=unknown/missing (fails closed)."""
    user: bool | None = None
    card: bool | None = None
    merchant: bool | None = None
    recipient: bool | None = None
    city: bool | None = None
    transaction: bool | None = None
    historical_transaction: bool | None = None
    temporal_history: bool | None = None
    identifiers_anonymized: bool = False


@dataclass(frozen=True)
class IndependenceClaim:
    """H. Independence — local synthetic/research datasets never qualify."""
    dataset_origin: str = "unknown"   # real_world_external/local_* /unknown
    confirmed_independent: tuple[str, ...] = ()
    known_overlaps: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContaminationClaim:
    """I. Contamination — UNKNOWN contamination status fails closed."""
    training_overlap: bool | None = None
    duplicate_transactions: bool | None = None
    duplicate_entities: bool | None = None
    benchmark_contamination: bool | None = None
    prior_model_exposure: bool | None = None
    feature_leakage: bool | None = None
    label_leakage: bool | None = None
    preprocessing_leakage: bool | None = None


@dataclass(frozen=True)
class UsagePermissionClaim:
    """J. Usage permission — never inferred from public availability."""
    permission_reference: str = ""
    permission_scope: str = ""
    permitted_use: bool | None = None
    valid_from: str = ""
    valid_until: str = ""
    permission_holder: str = ""


INDEPENDENCE_KEYS: tuple[str, ...] = (
    "model_training", "model_tuning", "threshold_selection",
    "feature_engineering", "synthetic_generation",
    "previous_benchmark_evaluation", "prior_model_exposure",
)

OVERLAP_REASON: dict[str, str] = {
    "model_training": "training_overlap",
    "model_tuning": "tuning_overlap",
    "threshold_selection": "threshold_selection_overlap",
    "feature_engineering": "feature_engineering_dependence",
    "synthetic_generation": "synthetic_generation_dependence",
    "previous_benchmark_evaluation": "benchmark_contamination",
    "prior_model_exposure": "prior_model_exposure",
}

CONTAMINATION_FIELDS: tuple[str, ...] = (
    "training_overlap", "duplicate_transactions", "duplicate_entities",
    "benchmark_contamination", "prior_model_exposure", "feature_leakage",
    "label_leakage", "preprocessing_leakage",
)

ENTITY_FACETS: tuple[str, ...] = (
    "user", "card", "merchant", "recipient", "city", "transaction",
    "historical", "temporal_history",
)
# contract attribute name for each facet code stem
ENTITY_FACET_ATTR: dict[str, str] = {
    "user": "user",
    "card": "card",
    "merchant": "merchant",
    "recipient": "recipient",
    "city": "city",
    "transaction": "transaction",
    "historical": "historical_transaction",
    "temporal_history": "temporal_history",
}

TRUSTED_LABEL_ORIGINS: frozenset[str] = frozenset({
    "human_investigation", "chargebacks", "chargeback",
    "provider_adjudication", "institutional_adjudication",
    "regulatory_referral", "law_enforcement_referral",
})

AMBIGUOUS_TIMEZONE_TOKENS: tuple[str, ...] = (
    "ambiguous", "unknown", "mixed", "unspecified", "local time", "local_time",
)


# ══════════════════════════════════════════════════════════════════════
# TOP-LEVEL CONTRACT
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExternalDatasetContract:
    """Immutable evidence contract for one external dataset candidate."""
    contract_version: str
    context_id: str
    assessed: bool
    provider_reference: str
    dataset_id: str
    model_id: str
    release_id: str
    evidence_valid_from: str
    evidence_cutoff: str
    provider_identity: ProviderIdentityClaim
    access_authority: AccessAuthorityClaim
    dataset_identity: DatasetIdentityClaim
    label_provenance: LabelProvenanceClaim
    temporal_semantics: TemporalSemanticsClaim
    feature_compatibility: FeatureCompatibilityClaim
    entity_continuity: EntityContinuityClaim
    independence: IndependenceClaim
    contamination: ContaminationClaim
    usage_permission: UsagePermissionClaim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def contract_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    def declared_claims(self) -> dict[str, str]:
        """Contract-declared values that evidence claims are checked against."""
        claims: dict[str, str] = {}
        if self.provider_identity.provider_name:
            claims["provider_name"] = self.provider_identity.provider_name
        if self.dataset_id:
            claims["dataset_id"] = self.dataset_id
        if self.dataset_identity.version:
            claims["dataset_version"] = self.dataset_identity.version
        if self.dataset_identity.fingerprint_available and self.dataset_identity.fingerprint:
            claims["fingerprint"] = self.dataset_identity.fingerprint
        if self.dataset_identity.record_count > 0:
            claims["record_count"] = str(self.dataset_identity.record_count)
        return claims


# ══════════════════════════════════════════════════════════════════════
# CANONICAL SERIALIZATION / HASHING
# ══════════════════════════════════════════════════════════════════════

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# Forbidden content in evidence packages (spec 5): no PAN, credentials,
# passwords, API keys, secrets, unnecessary PII, raw sensitive records.
FORBIDDEN_CONTENT_PATTERNS: tuple[str, ...] = (
    r"\bpassword\b", r"\bpasswd\b", r"\bapi[_-]?key\b", r"\bsecret\b",
    r"\bprivate[_-]?key\b", r"\bcredential\b", r"\bcvv\b", r"\bcvc\b",
    r"\bssn\b", r"\bpan\b", r"\bbearer\b", r"\bsk-[A-Za-z0-9]{8,}\b",
)


def find_forbidden_content(package: "EvidencePackage") -> tuple[str, ...]:
    """Return forbidden-content terms found anywhere in the package."""
    hits: list[str] = []
    blobs: list[str] = [
        package.package_id, package.provider_reference,
        package.dataset_reference, package.evidence_version,
        package.context_id,
    ]
    for item in package.items:
        blobs.extend([
            item.evidence_id, item.statement, item.claim_key,
            item.claim_value, item.provenance_reference,
        ])
    joined = "\n".join(blobs)
    for pattern in FORBIDDEN_CONTENT_PATTERNS:
        for m in re.finditer(pattern, joined, flags=re.IGNORECASE):
            hits.append(m.group(0).lower())
    return tuple(sorted(set(hits)))


# ══════════════════════════════════════════════════════════════════════
# EVIDENCE ITEM & PACKAGE (metadata-only — no PII, no raw records)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EvidenceItem:
    """One immutable, metadata-only piece of evidence."""
    evidence_id: str
    dimension: str
    origin: str                     # EvidenceOrigin value (≠ status)
    status: str                     # EvidenceStatus value (≠ origin)
    statement: str
    provider_reference: str
    dataset_reference: str
    context_id: str
    issued_at: str
    claim_key: str = ""
    claim_value: str = ""
    provenance_reference: str = ""
    supports_provider_fact: bool = False
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.dimension not in MANDATORY_DIMENSIONS:
            raise ValueError(f"unknown evidence dimension: {self.dimension}")
        if self.origin not in {o.value for o in EvidenceOrigin}:
            raise ValueError(f"unknown evidence origin: {self.origin}")
        if self.status not in {s.value for s in EvidenceStatus}:
            raise ValueError(f"unknown evidence status: {self.status}")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", self.content_hash_for())

    def content_hash_for(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k != "content_hash"}
        return _canonical_hash(d)

    def verify_hash(self) -> bool:
        return self.content_hash == self.content_hash_for()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_evidence_item(**kwargs: Any) -> EvidenceItem:
    """Factory that computes the item content hash."""
    return EvidenceItem(**kwargs)


@dataclass(frozen=True)
class EvidencePackage:
    """Deterministic metadata-only evidence package with SHA-256 hash."""
    package_id: str
    provider_reference: str
    dataset_reference: str
    evidence_version: str
    context_id: str
    package_created_at: str
    items: tuple[EvidenceItem, ...]
    package_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        if not self.package_hash:
            object.__setattr__(self, "package_hash", self.package_hash_for())

    def package_hash_for(self) -> str:
        d = {
            "package_id": self.package_id,
            "provider_reference": self.provider_reference,
            "dataset_reference": self.dataset_reference,
            "evidence_version": self.evidence_version,
            "context_id": self.context_id,
            "package_created_at": self.package_created_at,
            "item_hashes": sorted(i.content_hash for i in self.items),
        }
        return _canonical_hash(d)

    def verify_hash(self) -> bool:
        return self.package_hash == self.package_hash_for()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items"] = [i.to_dict() for i in self.items]
        return d


def build_evidence_package(
    *,
    package_id: str,
    provider_reference: str,
    dataset_reference: str,
    evidence_version: str,
    context_id: str,
    package_created_at: str,
    items: tuple[EvidenceItem, ...],
) -> EvidencePackage:
    """Build a package and compute its canonical SHA-256 hash."""
    return EvidencePackage(
        package_id=package_id,
        provider_reference=provider_reference,
        dataset_reference=dataset_reference,
        evidence_version=evidence_version,
        context_id=context_id,
        package_created_at=package_created_at,
        items=items,
    )


def verify_package_integrity(package: EvidencePackage | None) -> tuple[bool, tuple[str, ...]]:
    """Detect altered evidence, altered packages, altered hashes, duplicates
    and replayed (same-content, new-id) evidence.  Fail closed."""
    if package is None:
        return False, ("evidence_package_missing",)
    reasons: list[str] = []
    if not package.verify_hash():
        reasons.append("evidence_package_hash_mismatch")
    seen_ids: set[str] = set()
    content_index: dict[tuple[str, str, str], str] = {}
    for item in package.items:
        if not item.verify_hash():
            reasons.append("evidence_item_hash_mismatch")
        if item.evidence_id in seen_ids:
            reasons.append("duplicate_evidence")
        seen_ids.add(item.evidence_id)
        triple = (item.dimension, item.claim_key, item.statement)
        prior = content_index.get(triple)
        if prior is not None and prior != item.evidence_id:
            reasons.append("duplicate_evidence_content")
        else:
            content_index.setdefault(triple, item.evidence_id)
    if find_forbidden_content(package):
        reasons.append("evidence_contains_forbidden_content")
    return (not reasons), tuple(sorted(set(reasons)))


# ══════════════════════════════════════════════════════════════════════
# CONTRADICTION & MISSING-EVIDENCE DETECTION (spec 5/6)
# ══════════════════════════════════════════════════════════════════════

def detect_contradictions(
    contract: ExternalDatasetContract,
    package: EvidencePackage | None,
) -> tuple[str, ...]:
    """Detect item↔item and item↔contract claim conflicts.

    Example: "merchant identifiers remain stable" vs "merchant identifiers
    are randomized per transaction" ⇒ ENTITY_CONTINUITY contradicted.
    Conflicting evidence is NEVER silently resolved.
    """
    reasons: list[str] = []
    if package is None:
        return ()
    declared = contract.declared_claims()
    by_key: dict[tuple[str, str], set[str]] = {}
    for item in package.items:
        if not item.claim_key:
            continue
        # item vs contract
        if item.claim_key in declared:
            a = str(declared[item.claim_key]).strip()
            b = str(item.claim_value).strip()
            if a and b and a != b:
                # Dimension-tagged so the engine attaches the
                # contradiction to the right dimension (never dropped).
                reasons.append(
                    f"{item.dimension}:{_claim_conflict_code(item.claim_key)}")
        # item vs item
        if item.claim_value:
            by_key.setdefault((item.dimension, item.claim_key), set()).add(
                item.claim_value.strip()
            )
    for (dim, key), values in sorted(by_key.items()):
        if key and len(values) >= 2:
            reasons.append(f"{dim}:{key}_contradicted")
    return tuple(sorted(set(reasons)))


def _claim_conflict_code(claim_key: str) -> str:
    return {
        "provider_name": "provider_identity_contradicted",
        "dataset_id": "dataset_id_mismatch",
        "dataset_version": "dataset_version_mismatch",
        "fingerprint": "fingerprint_mismatch",
        "record_count": "record_count_inconsistent",
    }.get(claim_key, f"{claim_key}_mismatch")


def detect_missing_evidence(
    contract: ExternalDatasetContract,
    package: EvidencePackage | None,
) -> tuple[str, ...]:
    """Dimensions with no integrity-passing evidence items at all."""
    present: set[str] = set()
    if package is not None:
        for item in package.items:
            present.add(item.dimension)
    return tuple(
        d for d in MANDATORY_DIMENSIONS
        if d not in present and contract.assessed
    )


# ══════════════════════════════════════════════════════════════════════
# FEATURE PREFLIGHT (spec 7) — deterministic, against the canonical 48
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FeaturePreflightItem:
    """One canonical native feature's compatibility preflight result."""
    feature: str
    source_requirements: tuple[str, ...]
    availability: str              # FeatureAvailability value
    derivability: str
    leakage_risk: bool
    reason: str
    qualification_impact: str      # "blocking" | "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preflight_feature_compatibility(
    contract: ExternalDatasetContract,
) -> tuple[FeaturePreflightItem, ...]:
    """Classify every canonical native feature.  Never alters the canonical
    48-feature list, never adds features, never imputes unavailable ones."""
    claim = contract.feature_compatibility
    declared = {k: v for k, v in claim.requirement_availability}
    items: list[FeaturePreflightItem] = []
    for feature in ALTMAN_NATIVE_FEATURES:
        requirements = FEATURE_REQUIREMENTS.get(feature, ())
        reasons: list[str] = []
        if feature in claim.leakage_risk_features:
            availability = FeatureAvailability.LEAKAGE_RISK.value
            reasons.append("feature declared leakage-risk")
        elif not requirements:
            # A feature with no declared source requirement cannot be
            # imputed — treat as ambiguous (fail closed).
            availability = FeatureAvailability.AMBIGUOUS.value
            reasons.append("no source requirement declared for feature")
        else:
            per_req: list[str] = []
            for req in requirements:
                value = declared.get(req)
                if value is None:
                    per_req.append(FeatureAvailability.AMBIGUOUS.value)
                    reasons.append(f"requirement '{req}' not declared")
                elif value not in {a.value for a in FeatureAvailability}:
                    per_req.append(FeatureAvailability.AMBIGUOUS.value)
                    reasons.append(f"requirement '{req}' has invalid value")
                else:
                    per_req.append(value)
                    if value != FeatureAvailability.DIRECTLY_AVAILABLE.value:
                        reasons.append(f"requirement '{req}'={value}")
            # Deterministic severity fold across requirements:
            # UNAVAILABLE > LEAKAGE_RISK > AMBIGUOUS > DERIVABLE > DIRECT.
            severity = [
                FeatureAvailability.UNAVAILABLE.value,
                FeatureAvailability.LEAKAGE_RISK.value,
                FeatureAvailability.AMBIGUOUS.value,
                FeatureAvailability.DETERMINISTICALLY_DERIVABLE.value,
                FeatureAvailability.DIRECTLY_AVAILABLE.value,
            ]
            availability = next(
                (s for s in severity if s in per_req),
                FeatureAvailability.AMBIGUOUS.value,
            )
        if availability == FeatureAvailability.DIRECTLY_AVAILABLE.value:
            derivability = "direct"
        elif availability == FeatureAvailability.DETERMINISTICALLY_DERIVABLE.value:
            derivability = CANONICAL_TRANSFORMATION
        else:
            derivability = "none"
        blocking = availability in BLOCKING_AVAILABILITY
        if not reasons:
            reasons.append("all source requirements satisfied")
        items.append(FeaturePreflightItem(
            feature=feature,
            source_requirements=requirements,
            availability=availability,
            derivability=derivability,
            leakage_risk=availability == FeatureAvailability.LEAKAGE_RISK.value,
            reason="; ".join(dict.fromkeys(reasons)),
            qualification_impact="blocking" if blocking else "none",
        ))
    return tuple(items)


def blocking_features(
    preflight: tuple[FeaturePreflightItem, ...],
) -> tuple[str, ...]:
    return tuple(i.feature for i in preflight if i.qualification_impact == "blocking")


# ══════════════════════════════════════════════════════════════════════
# KNOWN-CANDIDATE CONTRACTS & EVIDENCE (spec 9 — no fabrication)
# ══════════════════════════════════════════════════════════════════════
# Built ONLY from the Phase 95 published-record fields.  Missing knowledge
# stays missing/unknown; nothing is invented to make a candidate qualify.

CONTEXT_ID = "phase104_external_dataset_evaluation"

_A = FeatureAvailability.AMBIGUOUS.value
_D = FeatureAvailability.DETERMINISTICALLY_DERIVABLE.value
_X = FeatureAvailability.UNAVAILABLE.value
_R = FeatureAvailability.DIRECTLY_AVAILABLE.value


def _avail(**overrides: str) -> tuple[tuple[str, str], ...]:
    base = {k: _A for k in REQUIREMENT_KEYS}
    base.update(overrides)
    return tuple(sorted(base.items()))


# Worldline 2017: Phase 103 records feature_compatible=True for this
# candidate; Phase 95's feature dictionary stays UNVERIFIED so the
# evidence side still fails closed even while requirements are derivable.
_WORLDLINE_2017_AVAILABILITY = _avail(**{k: _D for k in REQUIREMENT_KEYS})
_WORLDLINE_2017_RATIONALE = (
    "Phase 103 records feature_compatible=True (requirements "
    "deterministically derivable pending provider schema); Phase 95 feature "
    "dictionary UNVERIFIED so the dimension still cannot verify."
)
# Worldline 2018: same authoritative Phase 103 record.
_WORLDLINE_2018_AVAILABILITY = _avail(**{k: _D for k in REQUIREMENT_KEYS})
_WORLDLINE_2018_RATIONALE = (
    "Phase 103 records feature_compatible=True; Phase 95 schema and entity "
    "identifiers UNVERIFIED so the dimension still cannot verify."
)
# Novatti: customer/card identifiers substantially removed ⇒ entity and
# labeled-history features unavailable.
_NOVATTI_AVAILABILITY = _avail(
    amount=_D, absolute_time=_D,
    card_id=_X, user_history=_X, card_history=_X,
    labeled_user_history=_X, labeled_merchant_history=_X,
    labeled_city_history=_X,
)
_NOVATTI_RATIONALE = (
    "Phase 95/103: customer/card identifiers substantially removed ⇒ "
    "entity-bound and labeled-history features UNAVAILABLE; schema beyond "
    "amounts/temporal range not publicly documented."
)
# IEEE-CIS: documented schema has TransactionAmt (direct) but relative-only
# TransactionDT, no mcc/merchant/city/zip/state/channel/errors columns,
# card1-6 present with undocumented anonymisation, labels unverified.
_IEEE_AVAILABILITY = _avail(
    amount=_R,
    absolute_time=_X, channel=_X, errors=_X, zip=_X, state=_X, mcc=_X,
    merchant_id=_X, city_id=_X,
    user_history=_X, merchant_history=_X,
    labeled_user_history=_X, labeled_merchant_history=_X,
    labeled_city_history=_X,
    card_id=_A, card_history=_A,
)
_IEEE_RATIONALE = (
    "Phase 95/88 documented schema: TransactionDT is a relative timedelta "
    "(NOT absolute clock time); no mcc/merchant/city/zip/state/channel/"
    "error columns; card1-6 anonymisation scheme undocumented; label "
    "methodology unverified."
)


# Per-dimension known-candidate profiles (documentation, not evidence).
_PROFILE_RATIONALE: dict[str, dict[str, Any]] = {
    "WORLDLINE_ECOM_2017_NAG": {
        "features": _WORLDLINE_2017_RATIONALE,
        "availability": _WORLDLINE_2017_AVAILABILITY,
        "label_origin": "human_investigation",
        "entities": {k: None for k in ENTITY_FACET_ATTR.values()},
        "anonymized": False,
        "access_authority": AccessAuthority.UNKNOWN.value,
        "publicly_downloadable": False,
    },
    "WORLDLINE_ONLINE_2018": {
        "features": _WORLDLINE_2018_RATIONALE,
        "availability": _WORLDLINE_2018_AVAILABILITY,
        "label_origin": "unknown",
        "entities": {k: None for k in ENTITY_FACET_ATTR.values()},
        "anonymized": False,
        "access_authority": AccessAuthority.UNKNOWN.value,
        "publicly_downloadable": False,
    },
    "NOVATTI": {
        "features": _NOVATTI_RATIONALE,
        "availability": _NOVATTI_AVAILABILITY,
        "label_origin": "chargebacks",
        "entities": {k: None for k in ENTITY_FACET_ATTR.values()},
        # Explicit authoritative values (facet keys, not attribute names):
        # identifiers substantially removed ⇒ unstable user/card and no
        # historical/temporal per-entity history.
        "entities_overrides": {
            "user": False, "card": False,
            "historical": False, "temporal_history": False,
        },
        "anonymized": True,
        "access_authority": AccessAuthority.UNKNOWN.value,
        "publicly_downloadable": False,
    },
    "IEEE_CIS": {
        "features": _IEEE_RATIONALE,
        "availability": _IEEE_AVAILABILITY,
        "label_origin": "unknown",
        "entities": {k: None for k in ENTITY_FACET_ATTR.values()},
        "anonymized": True,
        "access_authority": AccessAuthority.PUBLIC_DOCUMENTED.value,
        "publicly_downloadable": True,
    },
}


def _derive_label_origin(label_method_reference: str) -> str:
    low = label_method_reference.lower()
    if "human investigator" in low:
        return "human_investigation"
    if "chargeback" in low:
        return "chargebacks"
    return "unknown"


def build_known_candidate_contract(dataset_id: str) -> ExternalDatasetContract:
    """Derive a Phase 104 contract from the Phase 95 published record.

    Only documented facts are copied; everything undocumented stays empty /
    unknown so the candidate fails closed.  No evidence is fabricated.
    """
    ev = KNOWN_CANDIDATES[dataset_id]
    profile = _PROFILE_RATIONALE[dataset_id]

    label_origin = profile["label_origin"]
    # Keep origin aligned with the actual published statement when the
    # statement itself documents the origin.
    derived = _derive_label_origin(ev.label_method_reference)
    if derived != "unknown":
        label_origin = derived

    entities = dict(profile["entities"])
    entities.update(profile.get("entities_overrides", {}))
    entity_kwargs = {
        attr: entities.get(facet)
        for facet, attr in ENTITY_FACET_ATTR.items()
    }

    provider_identity = ProviderIdentityClaim(
        provider_name=ev.provider_name,
        provider_organization=ev.provider_name,
        dataset_owner_controller=ev.ownership_or_controller,
        authoritative_provenance_reference=(
            ev.provenance_statement
            if _check_evidence_field(ev.provenance_statement).value == "verified"
            else ""
        ),
        # No provider attestation exists for any known candidate — this is
        # exactly why Worldline is blocked on "provider evidence missing".
        provider_attestation_reference="",
        evidence_timestamp="",           # no provider evidence timestamp
        evidence_version=ev.evidence_version,
    )
    access_authority = AccessAuthorityClaim(
        authority=profile["access_authority"],
        basis_reference=(
            ev.usage_permission_reference
            if _check_evidence_field(ev.usage_permission_reference).value == "verified"
            else ""
        ),
        publicly_downloadable=profile["publicly_downloadable"],
    )
    dataset_identity = DatasetIdentityClaim(
        canonical_dataset_id=ev.dataset_id,
        provider_dataset_id="",          # not publicly documented
        version=ev.dataset_version,      # "unknown_pending_provider_schema"
        collection_period="",            # not separately documented
        release_period="",
        fingerprint="",                  # not available — never invented
        fingerprint_available=False,
        schema_descriptor=ev.schema_reference,
        record_count=0,                  # exact count not established
        record_count_evidence=ev.provenance_statement,
        extraction_sampling_description="",  # not documented
    )
    label_provenance = LabelProvenanceClaim(
        label_definition="",             # never publicly documented
        label_origin=label_origin,
        labeling_authority=ev.label_method_reference,
        labeling_process=ev.label_method_reference,
        confirmation_timing="",          # not documented
        labels_retrospective=None,       # unknown — fails closed
        independently_adjudicated=None,  # unknown — fails closed
        suitable_for_evaluation=None,    # unknown — fails closed
        label_availability_time="",      # not documented
    )
    temporal_semantics = TemporalSemanticsClaim(
        event_timestamp_semantics=ev.timestamp_reference,
        timezone_semantics="",           # never documented
        observation_time="",             # not documented
        evaluation_cutoff=EVIDENCE_CUTOFF,
        temporal_ordering_rules="",      # not documented
        effective_at="",
        observed_at="",
        contains_future_information=False,
    )
    feature_compatibility = FeatureCompatibilityClaim(
        feature_contract_version=ML_FEATURE_VERSION,
        native_feature_version=ML_FEATURE_VERSION,
        transformation=CANONICAL_TRANSFORMATION,
        native_feature_order=tuple(ALTMAN_NATIVE_FEATURES),
        direct_domain_feature_inference=False,
        requirement_availability=profile["availability"],
        leakage_risk_features=(),
    )
    entity_continuity = EntityContinuityClaim(
        identifiers_anonymized=profile["anonymized"],
        **entity_kwargs,
    )
    independence = IndependenceClaim(
        dataset_origin="real_world_external",
        confirmed_independent=(),
        known_overlaps=(),
    )
    contamination = ContaminationClaim()   # all unknown — fails closed
    usage_permission = UsagePermissionClaim(
        permission_reference=ev.usage_permission_reference,
        permission_scope="",              # scope never documented
        permitted_use=None,               # unknown — fails closed
        valid_from="",
        valid_until="",
        permission_holder="",
    )

    return ExternalDatasetContract(
        contract_version=CONTRACT_VERSION,
        context_id=CONTEXT_ID,
        assessed=True,
        provider_reference=ev.provider_id,
        dataset_id=ev.dataset_id,
        model_id="",                      # bound by the engine, not here
        release_id="",
        evidence_valid_from=EVIDENCE_VALID_FROM,
        evidence_cutoff=EVIDENCE_CUTOFF,
        provider_identity=provider_identity,
        access_authority=access_authority,
        dataset_identity=dataset_identity,
        label_provenance=label_provenance,
        temporal_semantics=temporal_semantics,
        feature_compatibility=feature_compatibility,
        entity_continuity=entity_continuity,
        independence=independence,
        contamination=contamination,
        usage_permission=usage_permission,
    )


def build_known_candidate_evidence(dataset_id: str) -> EvidencePackage:
    """Evidence package for a known candidate: ONLY the Phase 95 published
    statements, with their honest origins.  No provider attestation exists,
    so no PROVIDER_ATTESTED items are ever created here."""
    ev = KNOWN_CANDIDATES[dataset_id]
    contract = build_known_candidate_contract(dataset_id)
    pub = EvidenceOrigin.PUBLIC_DOCUMENTATION.value
    sysgen = EvidenceOrigin.SYSTEM_GENERATED.value
    seq = [0]

    def item(dim: str, statement: str, origin: str,
             claim_key: str = "", claim_value: str = "") -> EvidenceItem:
        seq[0] += 1
        return make_evidence_item(
            evidence_id=f"{dataset_id}:e{seq[0]:02d}",
            dimension=dim,
            origin=origin,
            status=_check_evidence_field(statement).value,
            statement=statement,
            provider_reference=ev.provider_id,
            dataset_reference=ev.dataset_id,
            context_id=CONTEXT_ID,
            issued_at=EVIDENCE_AS_OF,
            claim_key=claim_key,
            claim_value=claim_value,
            provenance_reference=statement,
            supports_provider_fact=False,
        )

    items: list[EvidenceItem] = [
        item(ContractDimension.DATASET_IDENTITY.value, ev.provenance_statement,
             pub, claim_key="dataset_id", claim_value=ev.dataset_id),
        item(ContractDimension.DATASET_IDENTITY.value, ev.schema_reference, pub),
        item(ContractDimension.ACCESS_AUTHORITY.value,
             ev.usage_permission_reference, pub),
        item(ContractDimension.USAGE_PERMISSION.value,
             ev.usage_permission_reference, pub),
        item(ContractDimension.LABEL_PROVENANCE.value,
             ev.label_method_reference, pub),
        item(ContractDimension.TEMPORAL_SEMANTICS.value,
             ev.timestamp_reference, pub),
        item(ContractDimension.FEATURE_COMPATIBILITY.value,
             ev.feature_dictionary_reference, pub),
        item(ContractDimension.ENTITY_CONTINUITY.value,
             ev.entity_identifier_reference, pub),
        # Local-inspection statements about OUR side: system-generated,
        # never proof of provider facts, never able to reach VERIFIED.
        item(ContractDimension.INDEPENDENCE.value,
             ev.independence_statement, sysgen),
        item(ContractDimension.CONTAMINATION.value,
             ev.contamination_statement, sysgen),
        # NOTE: no PROVIDER_IDENTITY items — no provider attestation exists.
    ]
    return build_evidence_package(
        package_id=f"phase104-{dataset_id.lower()}-evidence",
        provider_reference=ev.provider_id,
        dataset_reference=ev.dataset_id,
        evidence_version=ev.evidence_version,
        context_id=CONTEXT_ID,
        package_created_at=EVIDENCE_AS_OF,
        items=tuple(items),
    )


KNOWN_DATASET_IDS: tuple[str, ...] = (
    "WORLDLINE_ECOM_2017_NAG",
    "WORLDLINE_ONLINE_2018",
    "NOVATTI",
    "IEEE_CIS",
)


def build_known_candidate_contract_and_evidence(
    dataset_id: str,
) -> tuple[ExternalDatasetContract, EvidencePackage]:
    return (build_known_candidate_contract(dataset_id),
            build_known_candidate_evidence(dataset_id))


__all__ = [
    "CONTRACT_VERSION", "EVIDENCE_POLICY_VERSION", "CONTEXT_ID",
    "DOMAIN_FEATURE_COUNT", "NATIVE_FEATURE_COUNT",
    "CANONICAL_TRANSFORMATION", "ML_FEATURE_VERSION",
    "EVIDENCE_VALID_FROM", "EVIDENCE_CUTOFF", "EVIDENCE_AS_OF",
    "EvidenceOrigin", "AccessAuthority", "FeatureAvailability",
    "ContractDimension", "MANDATORY_DIMENSIONS",
    "PROVIDER_FACT_DIMENSIONS", "UNTRUSTED_ORIGINS", "ATTESTED_ORIGINS",
    "STATUS_PRECEDENCE", "worst_status",
    "ProviderIdentityClaim", "AccessAuthorityClaim", "DatasetIdentityClaim",
    "LabelProvenanceClaim", "TemporalSemanticsClaim",
    "FeatureCompatibilityClaim", "EntityContinuityClaim",
    "IndependenceClaim", "ContaminationClaim", "UsagePermissionClaim",
    "ExternalDatasetContract",
    "INDEPENDENCE_KEYS", "OVERLAP_REASON", "CONTAMINATION_FIELDS",
    "ENTITY_FACETS", "ENTITY_FACET_ATTR", "TRUSTED_LABEL_ORIGINS",
    "AMBIGUOUS_TIMEZONE_TOKENS",
    "REQUIREMENT_KEYS", "FEATURE_REQUIREMENTS", "BLOCKING_AVAILABILITY",
    "EvidenceItem", "EvidencePackage", "make_evidence_item",
    "build_evidence_package", "verify_package_integrity",
    "detect_contradictions", "detect_missing_evidence",
    "canonical_json", "find_forbidden_content",
    "FeaturePreflightItem", "preflight_feature_compatibility",
    "blocking_features",
    "build_known_candidate_contract", "build_known_candidate_evidence",
    "build_known_candidate_contract_and_evidence", "KNOWN_DATASET_IDS",
    # re-exported Phase 95 authorities
    "EvidenceStatus", "DatasetQualificationState", "KNOWN_CANDIDATES",
]
