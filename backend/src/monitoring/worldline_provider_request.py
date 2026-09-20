"""Phase 91: Worldline provider data request & acceptance package validator.

Defines exactly what an authorized provider must supply and validates
whether the received package is documented well enough to enter the
existing Phase 84/85 admission pipeline.

Does NOT acquire data, contact providers, authenticate, download,
modify models, or change REAL_WORLD_VALIDATION.

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

class ValidatorResult(str, Enum):
    PACKAGE_MISSING = "package_missing"
    IDENTITY_INVALID = "identity_invalid"
    AUTHORIZATION_MISSING = "authorization_missing"
    PROVENANCE_INCOMPLETE = "provenance_incomplete"
    SCHEMA_MISSING = "schema_missing"
    ENTITY_CONTINUITY_FAILED = "entity_continuity_failed"
    TIMESTAMP_EVIDENCE_FAILED = "timestamp_evidence_failed"
    LABEL_EVIDENCE_FAILED = "label_evidence_failed"
    FEATURE_MAPPING_INCOMPLETE = "feature_mapping_incomplete"
    LEAKAGE_CONTROL_FAILED = "leakage_control_failed"
    PRIVACY_REVIEW_REQUIRED = "privacy_review_required"
    INTEGRITY_FAILED = "integrity_failed"
    TRANSFORMATION_EVIDENCE_FAILED = "transformation_evidence_failed"
    PACKAGE_VALID_FOR_PHASE85 = "package_valid_for_phase85"


class MappingStatus(str, Enum):
    DIRECT = "direct"
    DERIVED = "derived"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"
    INCOMPATIBLE = "incompatible"


class FieldRequired(str, Enum):
    REQUIRED = "required"
    CONDITIONAL = "conditional"
    OPTIONAL = "optional"


class PrivacyClass(str, Enum):
    SAFE = "safe"
    PSEUDONYMOUS = "pseudonymous"
    SENSITIVE = "sensitive"
    PII = "pii"
    PROHIBITED = "prohibited"


# ══════════════════════════════════════════════════════════════════════
# CANONICAL 48 FEATURES (from Phase 90)
# ══════════════════════════════════════════════════════════════════════

CANONICAL_48 = (
    "amt", "log_amt", "amt_sq", "hr", "mn", "dow", "Month", "Day",
    "hour_sin", "hour_cos", "is_night", "is_business_hours",
    "chip", "is_online", "is_swipe", "err", "has_zip", "has_state",
    "is_online_or_no_state",
    "mcc", "mcc_high", "mcc_restaurant", "mcc_gas", "mcc_grocery",
    "mcc_travel", "mcc_online",
    "merchant_id", "city_id", "card_id",
    "user_tx_count", "card_tx_count", "user_avg_amt", "amt_vs_user_avg",
    "amt_zscore", "merch_tx_count",
    "user_merchant_diversity", "user_city_diversity",
    "user_fraud_rate", "merch_fraud_rate", "city_fraud_rate",
    "high_amt", "very_high_amt",
    "amt_x_hr", "amt_x_mcc", "amt_x_chip", "amt_x_online", "amt_x_night",
    "user_merch_count",
)

LABEL_DEPENDENT_FEATURES = frozenset({
    "user_fraud_rate", "merch_fraud_rate", "city_fraud_rate",
    "card_fraud_rate", "user_recipient_fraud_rate",
})


# ══════════════════════════════════════════════════════════════════════
# REQUIRED SOURCE FIELDS
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RequiredSourceField:
    """Specification for one required source field."""
    field_name: str
    semantic_description: str
    required: str  # FieldRequired value
    entity: str
    expected_type: str
    timestamp_semantics: str
    label_dependency: str
    historical_dependency: bool
    privacy_class: str  # PrivacyClass value
    acceptable_anonymisation: str
    feature_dependencies: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["feature_dependencies"] = list(d["feature_dependencies"])
        return d


REQUIRED_SOURCE_FIELDS: list[RequiredSourceField] = [
    RequiredSourceField(
        "transaction_amount", "Payment amount in local currency",
        FieldRequired.REQUIRED.value, "none", "numeric_positive",
        "transaction_time", "none", False, PrivacyClass.SAFE.value,
        "none (amount is not PII)", ("amt", "log_amt", "amt_sq"),
    ),
    RequiredSourceField(
        "transaction_timestamp", "When the transaction occurred",
        FieldRequired.REQUIRED.value, "none", "datetime_or_numeric",
        "transaction_time", "none", False, PrivacyClass.SAFE.value,
        "none (timestamp is not PII)", ("hr", "mn", "dow", "Month", "Day"),
    ),
    RequiredSourceField(
        "customer_identifier", "Stable pseudonymous customer identifier",
        FieldRequired.REQUIRED.value, "user_customer", "string_or_numeric",
        "none", "none", False, PrivacyClass.PSEUDONYMOUS.value,
        "SHA-256 or tokenisation; must be stable across transactions",
        ("user_tx_count", "user_avg_amt", "user_fraud_rate", "amt_vs_user_avg",
         "amt_zscore", "high_amt", "very_high_amt", "user_merchant_diversity",
         "user_city_diversity", "user_merch_count"),
    ),
    RequiredSourceField(
        "merchant_identifier", "Stable pseudonymous merchant identifier",
        FieldRequired.REQUIRED.value, "merchant", "string_or_numeric",
        "none", "none", False, PrivacyClass.PSEUDONYMOUS.value,
        "SHA-256 or tokenisation; must be stable across transactions",
        ("merch_tx_count", "merchant_avg_amt", "merchant_fraud_rate",
         "merchant_city_count", "merchant_user_count", "user_merchant_diversity",
         "user_merch_count"),
    ),
    RequiredSourceField(
        "fraud_label", "Binary fraud indicator (1=fraud, 0=legitimate)",
        FieldRequired.REQUIRED.value, "none", "binary",
        "label_effective_time", "label_source", False, PrivacyClass.SAFE.value,
        "none (binary label is not PII)",
        ("user_fraud_rate", "merch_fraud_rate", "city_fraud_rate"),
    ),
    RequiredSourceField(
        "card_identifier", "Stable pseudonymous card identifier",
        FieldRequired.CONDITIONAL.value, "card", "string_or_numeric",
        "none", "none", False, PrivacyClass.PSEUDONYMOUS.value,
        "SHA-256 or tokenisation; required if card-level features evaluated",
        ("card_tx_count", "card_avg_amt", "card_fraud_rate", "card_city_count",
         "card_user_count"),
    ),
    RequiredSourceField(
        "city_identifier", "Stable geographic/city identifier",
        FieldRequired.CONDITIONAL.value, "city_geography", "string_or_numeric",
        "none", "none", False, PrivacyClass.PSEUDONYMOUS.value,
        "Stable geographic code; required if city-level features evaluated",
        ("city_avg_amt", "city_fraud_rate", "city_merchant_count",
         "city_user_count", "user_city_diversity"),
    ),
    RequiredSourceField(
        "merchant_category_code", "MCC or equivalent category",
        FieldRequired.CONDITIONAL.value, "none", "numeric_or_categorical",
        "none", "none", False, PrivacyClass.SAFE.value,
        "Standard MCC or documented equivalent",
        ("mcc", "mcc_high", "mcc_restaurant", "mcc_gas", "mcc_grocery",
         "mcc_travel", "mcc_online", "user_mcc_diversity", "amt_x_mcc"),
    ),
    RequiredSourceField(
        "transaction_channel", "Chip/Online/Swipe indicator",
        FieldRequired.CONDITIONAL.value, "none", "categorical",
        "none", "none", False, PrivacyClass.SAFE.value,
        "Channel type (chip/online/swipe) or documented equivalent",
        ("chip", "is_online", "is_swipe", "is_online_or_no_state",
         "amt_x_chip", "amt_x_online"),
    ),
    RequiredSourceField(
        "transaction_error", "Transaction error indicator",
        FieldRequired.OPTIONAL.value, "none", "categorical",
        "none", "none", False, PrivacyClass.SAFE.value,
        "Error description or indicator",
        ("err",),
    ),
]


# ══════════════════════════════════════════════════════════════════════
# FEATURE MAPPING ENTRY
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FeatureMappingEntry:
    """Required feature mapping for one PS-14 native feature."""
    native_feature: str
    source_fields: tuple[str, ...]
    transformation: str
    aggregation_window: str
    entity: str
    temporal_cutoff: str
    label_dependency: str
    leakage_controls: str
    evidence: str
    status: str  # MappingStatus value

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["source_fields"] = list(d["source_fields"])
        return d


# ══════════════════════════════════════════════════════════════════════
# ACCEPTANCE PACKAGE SCHEMA
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DatasetIdentityEvidence:
    """Evidence of dataset identity."""
    dataset_id: str
    provider: str
    dataset_name: str
    date_range: str
    approximate_row_count: int
    transaction_domain: str
    is_original: bool

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class AuthorizationEvidence:
    """Evidence of authorized access."""
    authorized: bool
    authorization_reference: str
    data_use_agreement_reference: str
    access_date: str
    authorizing_institution: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class ProvenanceEvidence:
    """Provenance evidence for the dataset."""
    provider_name: str
    data_owner: str
    data_custodian: str
    collection_period: str
    collection_method: str
    anonymisation_method: str
    label_generation_method: str
    transformation_history: str
    publication_reference: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class SchemaEvidence:
    """Schema evidence for the dataset."""
    columns: tuple[str, ...]
    row_count: int
    column_count: int
    column_types: dict[str, str]
    null_rates: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["columns"] = list(d["columns"])
        return d


@dataclass(frozen=True)
class EntityEvidence:
    """Entity continuity evidence."""
    customer_id_present: bool
    customer_id_stable: bool
    customer_id_anonymised: bool
    merchant_id_present: bool
    merchant_id_stable: bool
    merchant_id_anonymised: bool
    card_id_present: bool
    card_id_stable: bool
    city_id_present: bool
    city_id_stable: bool
    terminal_id_present: bool
    terminal_id_stable: bool

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class TimestampEvidence:
    """Timestamp evidence."""
    has_timestamps: bool
    timestamp_is_absolute: bool
    timezone_documented: bool
    timezone_value: str
    timestamp_precision: str
    chronological_ordering: bool

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class LabelEvidence:
    """Label provenance evidence."""
    label_field: str
    label_values: tuple[str, ...]
    label_definition: str
    label_source: str
    label_generation_process: str
    label_effective_time_documented: bool
    label_retraction_documented: bool
    label_dispute_documented: bool

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["label_values"] = list(d["label_values"])
        return d


@dataclass(frozen=True)
class PrivacyEvidence:
    """Privacy evidence."""
    pii_inventory: tuple[str, ...]
    anonymisation_description: str
    identifier_tokenisation_method: str
    retention_policy: str
    data_minimisation_statement: str
    prohibited_fields_present: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["pii_inventory"] = list(d["pii_inventory"])
        d["prohibited_fields_present"] = list(d["prohibited_fields_present"])
        return d


@dataclass(frozen=True)
class IntegrityEvidence:
    """Data integrity evidence."""
    raw_dataset_hash: str
    schema_hash: str
    manifest_hash: str
    file_list: tuple[str, ...]
    file_sizes: dict[str, int]
    dataset_version: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["file_list"] = list(d["file_list"])
        return d


@dataclass(frozen=True)
class TransformationEvidence:
    """Transformation history evidence."""
    transformations_applied: bool
    transformation_description: str
    transformation_code_version: str
    source_hash: str
    transformed_hash: str
    operator_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class WorldlineAcceptancePackage:
    """Complete acceptance package from an authorized provider."""
    dataset_identity: dict
    authorization_evidence: dict
    provenance_evidence: dict
    schema_evidence: dict
    entity_evidence: dict
    timestamp_evidence: dict
    label_evidence: dict
    feature_mapping: list[dict]
    privacy_evidence: dict
    integrity_evidence: dict
    transformation_evidence: dict

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ══════════════════════════════════════════════════════════════════════
# PACKAGE BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_acceptance_package(
    dataset_identity: DatasetIdentityEvidence,
    authorization: AuthorizationEvidence,
    provenance: ProvenanceEvidence,
    schema: SchemaEvidence,
    entity: EntityEvidence,
    timestamp: TimestampEvidence,
    label: LabelEvidence,
    feature_mapping: list[FeatureMappingEntry],
    privacy: PrivacyEvidence,
    integrity: IntegrityEvidence,
    transformation: TransformationEvidence,
) -> WorldlineAcceptancePackage:
    """Build a deterministic acceptance package."""
    return WorldlineAcceptancePackage(
        dataset_identity=dataset_identity.to_dict(),
        authorization_evidence=authorization.to_dict(),
        provenance_evidence=provenance.to_dict(),
        schema_evidence=schema.to_dict(),
        entity_evidence=entity.to_dict(),
        timestamp_evidence=timestamp.to_dict(),
        label_evidence=label.to_dict(),
        feature_mapping=[fm.to_dict() for fm in feature_mapping],
        privacy_evidence=privacy.to_dict(),
        integrity_evidence=integrity.to_dict(),
        transformation_evidence=transformation.to_dict(),
    )


def canonicalize_acceptance_package(pkg: WorldlineAcceptancePackage) -> str:
    """Deterministic canonical JSON of the acceptance package."""
    return json.dumps(pkg.to_dict(), sort_keys=True, separators=(",", ":"))


def hash_acceptance_package(pkg: WorldlineAcceptancePackage) -> str:
    """Deterministic SHA-256 hash of the acceptance package."""
    canonical = canonicalize_acceptance_package(pkg)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# VALIDATOR
# ══════════════════════════════════════════════════════════════════════

def validate_worldline_acceptance_package(
    pkg: WorldlineAcceptancePackage | None,
) -> dict[str, Any]:
    """Deterministic validator for a Worldline acceptance package.

    Returns a dict with:
    - result: ValidatorResult value
    - blocking_reasons: list of specific blockers
    - feature_mapping_coverage: dict with counts
    - next_action: recommended next step
    """
    if pkg is None:
        return {
            "result": ValidatorResult.PACKAGE_MISSING.value,
            "blocking_reasons": ["no package provided"],
            "feature_mapping_coverage": {},
            "next_action": "SUPPLY_ACCEPTANCE_PACKAGE",
        }

    blocking: list[str] = []

    # 1. Identity
    di = pkg.dataset_identity
    if di.get("dataset_id") != "WORLDLINE_ECOM_2017_NAG":
        blocking.append(f"identity_invalid: {di.get('dataset_id', 'missing')}")

    # 2. Authorization
    auth = pkg.authorization_evidence
    if not auth.get("authorized"):
        blocking.append("authorization_missing: not authorized")
    if not auth.get("authorization_reference"):
        blocking.append("authorization_missing: no authorization reference")
    if not auth.get("data_use_agreement_reference"):
        blocking.append("authorization_missing: no data-use agreement")

    # 3. Provenance
    prov = pkg.provenance_evidence
    required_prov = ["provider_name", "data_owner", "label_generation_method",
                     "collection_period", "anonymisation_method"]
    for field in required_prov:
        if not prov.get(field):
            blocking.append(f"provenance_incomplete: missing {field}")

    # 4. Schema
    schema = pkg.schema_evidence
    if not schema.get("columns") or schema.get("column_count", 0) == 0:
        blocking.append("schema_missing: no columns documented")
    if schema.get("row_count", 0) < 1000:
        blocking.append(f"schema_incompatible: only {schema.get('row_count', 0)} rows")

    # 5. Entity continuity
    ent = pkg.entity_evidence
    if not ent.get("customer_id_present"):
        blocking.append("entity_continuity_failed: customer_id not present")
    if not ent.get("customer_id_stable"):
        blocking.append("entity_continuity_failed: customer_id not stable")
    if not ent.get("merchant_id_present"):
        blocking.append("entity_continuity_failed: merchant_id not present")
    if not ent.get("merchant_id_stable"):
        blocking.append("entity_continuity_failed: merchant_id not stable")

    # 6. Timestamps
    ts = pkg.timestamp_evidence
    if not ts.get("has_timestamps"):
        blocking.append("timestamp_evidence_failed: no timestamps")
    if not ts.get("chronological_ordering"):
        blocking.append("timestamp_evidence_failed: no chronological ordering")

    # 7. Labels
    lbl = pkg.label_evidence
    if not lbl.get("label_field"):
        blocking.append("label_evidence_failed: no label field")
    if not lbl.get("label_generation_process"):
        blocking.append("label_evidence_failed: no label generation process")
    if not lbl.get("label_effective_time_documented"):
        blocking.append("label_evidence_failed: label effective time not documented")

    # 8. Feature mapping
    fm = pkg.feature_mapping
    if len(fm) == 0:
        blocking.append("feature_mapping_incomplete: no feature mappings")
    else:
        mapped_features = {m.get("native_feature") for m in fm}
        missing = set(CANONICAL_48) - mapped_features
        if missing:
            blocking.append(f"feature_mapping_incomplete: missing {sorted(missing)}")
        # Check no duplicates
        if len(mapped_features) < len(fm):
            blocking.append("feature_mapping_incomplete: duplicate features in mapping")

    # 9. Leakage controls
    for m in fm:
        if m.get("label_dependency") and not m.get("leakage_controls"):
            if m.get("native_feature") in LABEL_DEPENDENT_FEATURES:
                blocking.append(
                    f"leakage_control_failed: {m.get('native_feature')} "
                    f"requires leakage controls"
                )

    # 10. Privacy
    priv = pkg.privacy_evidence
    prohibited = priv.get("prohibited_fields_present", [])
    if prohibited:
        blocking.append(f"privacy_review_required: prohibited fields {prohibited}")

    # 11. Integrity
    integ = pkg.integrity_evidence
    if not integ.get("raw_dataset_hash"):
        blocking.append("integrity_failed: no raw dataset hash")
    if not integ.get("schema_hash"):
        blocking.append("integrity_failed: no schema hash")

    # 12. Transformation
    transf = pkg.transformation_evidence
    if not transf.get("transformation_description") and transf.get("transformations_applied"):
        blocking.append("transformation_evidence_failed: transformations applied but not documented")

    # Determine result
    if blocking:
        # Pick the most specific result
        first_blocker = blocking[0]
        if "identity" in first_blocker:
            result = ValidatorResult.IDENTITY_INVALID
        elif "authorization" in first_blocker:
            result = ValidatorResult.AUTHORIZATION_MISSING
        elif "provenance" in first_blocker:
            result = ValidatorResult.PROVENANCE_INCOMPLETE
        elif "schema" in first_blocker:
            result = ValidatorResult.SCHEMA_MISSING
        elif "entity" in first_blocker:
            result = ValidatorResult.ENTITY_CONTINUITY_FAILED
        elif "timestamp" in first_blocker:
            result = ValidatorResult.TIMESTAMP_EVIDENCE_FAILED
        elif "label" in first_blocker:
            result = ValidatorResult.LABEL_EVIDENCE_FAILED
        elif "feature_mapping" in first_blocker:
            result = ValidatorResult.FEATURE_MAPPING_INCOMPLETE
        elif "leakage" in first_blocker:
            result = ValidatorResult.LEAKAGE_CONTROL_FAILED
        elif "privacy" in first_blocker:
            result = ValidatorResult.PRIVACY_REVIEW_REQUIRED
        elif "integrity" in first_blocker:
            result = ValidatorResult.INTEGRITY_FAILED
        elif "transformation" in first_blocker:
            result = ValidatorResult.TRANSFORMATION_EVIDENCE_FAILED
        else:
            result = ValidatorResult.PROVENANCE_INCOMPLETE
    else:
        result = ValidatorResult.PACKAGE_VALID_FOR_PHASE85

    # Feature mapping coverage
    mapped = {m.get("native_feature") for m in fm} if fm else set()
    coverage = {
        "total": len(CANONICAL_48),
        "mapped": len(mapped & set(CANONICAL_48)),
        "missing": len(set(CANONICAL_48) - mapped),
        "unknown": sum(1 for m in fm if m.get("status") == "unknown"),
        "incompatible": sum(1 for m in fm if m.get("status") == "incompatible"),
    }

    next_action = (
        "SUPPLY_MISSING_EVIDENCE" if blocking
        else "PROCEED_TO_PHASE84_INGESTION"
    )

    return {
        "result": result.value,
        "blocking_reasons": blocking,
        "feature_mapping_coverage": coverage,
        "next_action": next_action,
    }


# ══════════════════════════════════════════════════════════════════════
# PROVIDER CHECKLIST EXPORT
# ══════════════════════════════════════════════════════════════════════

PROVIDER_CHECKLIST = {
    "dataset_id": "WORLDLINE_ECOM_2017_NAG",
    "required_sections": [
        {"section": "Authorization", "items": [
            "Data-use agreement reference",
            "Authorization reference from data owner",
            "Access date and scope",
            "Authorizing institution",
        ]},
        {"section": "Provenance", "items": [
            "Provider name and data owner",
            "Data custodian",
            "Collection period (Jan-Jul 2017)",
            "Collection method",
            "Publication/reference identifier",
        ]},
        {"section": "Schema", "items": [
            "Complete column list with types",
            "Row count",
            "Null rates per column",
            "Sample data (first 10 rows, redacted if needed)",
        ]},
        {"section": "Entities", "items": [
            "Customer identifier: present, stable, anonymised",
            "Merchant identifier: present, stable, anonymised",
            "Card identifier: present, stable (if card-level features needed)",
            "City/geography identifier: present, stable (if city features needed)",
            "Terminal identifier: document if present (do NOT equate to merchant)",
        ]},
        {"section": "Timestamps", "items": [
            "Transaction timestamp field",
            "Timezone or reference time",
            "Precision (second/minute/hour)",
            "Chronological ordering confirmed",
        ]},
        {"section": "Labels", "items": [
            "Fraud label field and values",
            "Label generation process (human investigators, expert rules)",
            "Label effective timestamp",
            "Label retraction/dispute process",
        ]},
        {"section": "Feature Mapping", "items": [
            "Mapping for all 48 canonical PS-14 features",
            "Source field(s) for each feature",
            "Transformation/aggregation method",
            "Temporal cutoff for historical features",
            "Entity required for each feature",
            "Leakage controls for label-dependent features",
        ]},
        {"section": "Privacy", "items": [
            "PII inventory (list of all fields with privacy classification)",
            "Anonymisation/tokenisation method",
            "Data minimisation statement",
            "Retention policy",
            "Confirmation: no raw PAN, CVV, passwords, or government IDs",
        ]},
        {"section": "Integrity", "items": [
            "SHA-256 hash of raw dataset",
            "Schema hash",
            "Package manifest hash",
            "File list with sizes",
            "Dataset version",
        ]},
        {"section": "Transformation", "items": [
            "List of all transformations applied to raw data",
            "Transformation code version",
            "Source and transformed data hashes",
        ]},
    ],
    "optional_sections": [
        "Terminal identifier documentation",
        "Recipient identifier documentation",
        "Additional metadata",
    ],
    "explicit_note": "Missing information is not treated as compatible. "
                     "Unknown fields remain UNKNOWN. Fields not documented "
                     "cannot silently become ACCEPTED.",
}


def generate_provider_checklist() -> dict[str, Any]:
    """Deterministic export of the provider checklist."""
    return PROVIDER_CHECKLIST


# ══════════════════════════════════════════════════════════════════════
# PROVIDER REQUEST SPECIFICATION
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProviderRequestSpec:
    """Immutable provider request specification."""
    request_id: str
    request_version: str
    target_dataset: str
    required_source_fields: list[dict]
    required_sections: tuple[str, ...]
    canonical_48_features: tuple[str, ...]
    label_dependent_features: tuple[str, ...]
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["canonical_48_features"] = list(d["canonical_48_features"])
        d["label_dependent_features"] = list(d["label_dependent_features"])
        return d


def build_provider_request() -> ProviderRequestSpec:
    """Build the deterministic provider request specification."""
    manifest_content = {
        "request_id": "WORLDLINE-PROVIDER-REQUEST-91",
        "request_version": "phase91_v1",
        "target_dataset": "WORLDLINE_ECOM_2017_NAG",
        "source_field_count": len(REQUIRED_SOURCE_FIELDS),
        "canonical_feature_count": len(CANONICAL_48),
        "label_dependent_count": len(LABEL_DEPENDENT_FEATURES),
        "required_sections": [
            "Authorization", "Provenance", "Schema", "Entities",
            "Timestamps", "Labels", "Feature Mapping", "Privacy",
            "Integrity", "Transformation",
        ],
    }
    canonical = json.dumps(manifest_content, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return ProviderRequestSpec(
        request_id="WORLDLINE-PROVIDER-REQUEST-91",
        request_version="phase91_v1",
        target_dataset="WORLDLINE_ECOM_2017_NAG",
        required_source_fields=[f.to_dict() for f in REQUIRED_SOURCE_FIELDS],
        required_sections=manifest_content["required_sections"],
        canonical_48_features=CANONICAL_48,
        label_dependent_features=tuple(LABEL_DEPENDENT_FEATURES),
        manifest_hash=manifest_hash,
    )
