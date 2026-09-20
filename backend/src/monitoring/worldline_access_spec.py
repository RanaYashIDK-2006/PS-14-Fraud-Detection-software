"""Phase 90: Worldline authorized data access & acceptance specification.

Defines what PS-14 would require from an authorized Worldline dataset
before it could enter the Phase 84/85 admission pipeline.

Does NOT download data, connect to Worldline, authenticate, modify
models, or change REAL_WORLD_VALIDATION.

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
# CONTROLLED STATUS VALUES
# ══════════════════════════════════════════════════════════════════════

class DatasetIdentityStatus(str, Enum):
    VERIFIED = "verified"
    IDENTITY_UNVERIFIED = "identity_unverified"
    AMBIGUOUS = "ambiguous"
    NOT_RECEIVED = "not_received"


class ProvenanceEvidenceStatus(str, Enum):
    DOCUMENTED = "documented"
    PROVIDER_CONFIRMED = "provider_confirmed"
    UNVERIFIED = "unverified"
    UNKNOWN = "unknown"


class SchemaFieldStatus(str, Enum):
    ACCEPTED = "accepted"
    ACCEPTED_AFTER_SCHEMA_VERIFICATION = "accepted_after_schema_verification"
    CONDITIONAL = "conditional"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class AcceptanceGateResult(str, Enum):
    NOT_RECEIVED = "not_received"
    IDENTITY_UNVERIFIED = "identity_unverified"
    PROVENANCE_INCOMPLETE = "provenance_incomplete"
    SCHEMA_INCOMPATIBLE = "schema_incompatible"
    ENTITY_INCOMPATIBLE = "entity_incompatible"
    TEMPORAL_INCOMPATIBLE = "temporal_incompatible"
    LEAKAGE_RISK = "leakage_risk"
    LABEL_PROVENANCE_INSUFFICIENT = "label_provenance_insufficient"
    PRIVACY_REVIEW_REQUIRED = "privacy_review_required"
    SECURITY_REVIEW_REQUIRED = "security_review_required"
    ELIGIBLE_FOR_PHASE85_ADMISSION = "eligible_for_phase85_admission"


class EntityRequirementStatus(str, Enum):
    REQUIRED_STABLE = "required_stable"
    REQUIRED_DOCUMENTED = "required_documented"
    OPTIONAL = "optional"
    NOT_REQUIRED = "not_required"
    UNKNOWN = "unknown"


class LeakageRiskLevel(str, Enum):
    SAFE = "safe"
    SAFE_WITH_TEMPORAL_WINDOW = "safe_with_temporal_window"
    REQUIRES_LABEL_HISTORY = "requires_label_history"
    LEAKAGE_RISK = "leakage_risk"
    NOT_DERIVABLE = "not_derivable"
    UNKNOWN = "unknown"


# ══════════════════════════════════════════════════════════════════════
# DATASET IDENTITY SPECIFICATIONS
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WorldlineDatasetIdentity:
    """Immutable identity specification for a Worldline dataset."""
    dataset_id: str
    dataset_name: str
    provider: str
    publication_reference: str
    date_range: str
    transaction_domain: str
    approximate_scale: str
    source_institution: str
    is_original: bool
    label_methodology: str
    identity_status: str  # DatasetIdentityStatus

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# Primary target: 2017 NAG paper dataset
WORLDLINE_ECOM_2017_NAG = WorldlineDatasetIdentity(
    dataset_id="WORLDLINE_ECOM_2017_NAG",
    dataset_name="Worldline Belgium Real-World E-Commerce Fraud Dataset (2017)",
    provider="Worldline",
    publication_reference=(
        "NAG: neural feature aggregation framework for credit card "
        "fraud detection (published research paper). Dataset provided "
        "by Worldline for the study."
    ),
    date_range="January 2017 - July 2017",
    transaction_domain="real_world_ecommerce_credit_card",
    approximate_scale="60,000,000+ transactions",
    source_institution="Worldline / collaborating research institution",
    is_original=True,
    label_methodology=(
        "Fraud labels assigned by a team of human investigators "
        "based on regularly updated expert rules. Labels are NOT "
        "synthetic or model-generated."
    ),
    identity_status=DatasetIdentityStatus.IDENTITY_UNVERIFIED.value,
)

# Secondary: separate 2018 dataset (DIFFERENT dataset, do NOT conflate)
WORLDLINE_ONLINE_2018 = WorldlineDatasetIdentity(
    dataset_id="WORLDLINE_ONLINE_2018",
    dataset_name="Worldline Online International Transactions (2018)",
    provider="Worldline",
    publication_reference=(
        "Published research on adversarial fraud detection using "
        "Worldline online international transactions."
    ),
    date_range="May 2018 - September 2018",
    transaction_domain="online_international_transactions",
    approximate_scale="60,000,000+ transactions",
    source_institution="Worldline",
    is_original=True,
    label_methodology="Not fully documented in available public sources",
    identity_status=DatasetIdentityStatus.IDENTITY_UNVERIFIED.value,
)

# All known Worldline dataset identities (for collision detection)
WORLDLINE_DATASETS: dict[str, WorldlineDatasetIdentity] = {
    "WORLDLINE_ECOM_2017_NAG": WORLDLINE_ECOM_2017_NAG,
    "WORLDLINE_ONLINE_2018": WORLDLINE_ONLINE_2018,
}


# ══════════════════════════════════════════════════════════════════════
# REQUIRED PROVENANCE FIELDS
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProvenanceRequirement:
    """One required provenance field."""
    field_name: str
    description: str
    required: bool
    evidence_status: str  # ProvenanceEvidenceStatus
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


PROVENANCE_REQUIREMENTS: list[ProvenanceRequirement] = [
    ProvenanceRequirement("provider_name", "Name of the data provider", True,
        ProvenanceEvidenceStatus.DOCUMENTED.value, "Worldline"),
    ProvenanceRequirement("data_owner", "Legal owner of the data", True,
        ProvenanceEvidenceStatus.UNKNOWN.value, "To be confirmed by provider"),
    ProvenanceRequirement("data_custodian", "Organization responsible for data", True,
        ProvenanceEvidenceStatus.UNKNOWN.value, "To be confirmed by provider"),
    ProvenanceRequirement("dataset_id", "Unique identifier matching a known identity", True,
        ProvenanceEvidenceStatus.DOCUMENTED.value),
    ProvenanceRequirement("dataset_version", "Version identifier", True,
        ProvenanceEvidenceStatus.UNKNOWN.value),
    ProvenanceRequirement("acquisition_date", "Date data was obtained", True,
        ProvenanceEvidenceStatus.UNKNOWN.value),
    ProvenanceRequirement("authorized_access_reference", "Reference to access authorization", True,
        ProvenanceEvidenceStatus.UNKNOWN.value),
    ProvenanceRequirement("data_use_agreement_reference", "Reference to data use agreement", True,
        ProvenanceEvidenceStatus.UNKNOWN.value),
    ProvenanceRequirement("publication_reference", "Research paper or documentation reference", True,
        ProvenanceEvidenceStatus.DOCUMENTED.value),
    ProvenanceRequirement("original_collection_period", "Period during which data was collected", True,
        ProvenanceEvidenceStatus.DOCUMENTED.value, "Jan-Jul 2017"),
    ProvenanceRequirement("collection_method", "How the data was collected", True,
        ProvenanceEvidenceStatus.UNKNOWN.value),
    ProvenanceRequirement("transformation_history", "Any transformations applied to raw data", True,
        ProvenanceEvidenceStatus.UNKNOWN.value),
    ProvenanceRequirement("anonymisation_method", "How identifiers were anonymised", True,
        ProvenanceEvidenceStatus.UNKNOWN.value),
    ProvenanceRequirement("label_generation_method", "How fraud labels were generated", True,
        ProvenanceEvidenceStatus.DOCUMENTED.value, "Human investigators + expert rules"),
    ProvenanceRequirement("schema_version", "Schema version of the received data", True,
        ProvenanceEvidenceStatus.UNKNOWN.value),
    ProvenanceRequirement("integrity_hash", "Hash of the received dataset", True,
        ProvenanceEvidenceStatus.UNKNOWN.value),
]


# ══════════════════════════════════════════════════════════════════════
# ENTITY REQUIREMENTS
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EntityRequirement:
    """Requirement for a specific entity type."""
    entity_type: str
    required: str  # EntityRequirementStatus
    ps14_features_using_it: tuple[str, ...]
    minimum_properties: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["ps14_features_using_it"] = list(d["ps14_features_using_it"])
        d["minimum_properties"] = list(d["minimum_properties"])
        return d


ENTITY_REQUIREMENTS: list[EntityRequirement] = [
    EntityRequirement(
        "user_customer",
        EntityRequirementStatus.REQUIRED_STABLE.value,
        ("user_tx_count", "user_avg_amt", "amt_vs_user_avg", "amt_zscore",
         "user_fraud_rate", "high_amt", "very_high_amt", "user_merchant_diversity",
         "user_city_diversity", "user_merch_count"),
        ("stable_pseudonymous_id", "consistent_across_transactions",
         "documented_anonymisation"),
        "Customer/user identity is required for user-level behavioral features",
    ),
    EntityRequirement(
        "card",
        EntityRequirementStatus.REQUIRED_DOCUMENTED.value,
        ("card_tx_count", "card_avg_amt", "card_fraud_rate", "card_city_count",
         "card_user_count"),
        ("stable_pseudonymous_id", "consistent_across_transactions"),
        "Card identity required if card-level features are evaluated",
    ),
    EntityRequirement(
        "merchant",
        EntityRequirementStatus.REQUIRED_STABLE.value,
        ("merch_tx_count", "merchant_avg_amt", "merchant_fraud_rate",
         "merchant_city_count", "merchant_user_count", "merchant_avg_age",
         "user_merchant_diversity"),
        ("stable_pseudonymous_id", "consistent_across_transactions",
         "documented_anonymisation"),
        "Merchant identity is required for merchant-level features",
    ),
    EntityRequirement(
        "city_geography",
        EntityRequirementStatus.REQUIRED_DOCUMENTED.value,
        ("city_avg_amt", "city_fraud_rate", "city_merchant_count",
         "city_user_count", "user_city_diversity", "merchant_city_count",
         "card_city_count"),
        ("stable_identifier", "documented_semantics"),
        "City/geographic identity required for city-level features",
    ),
    EntityRequirement(
        "terminal",
        EntityRequirementStatus.OPTIONAL.value,
        (),
        ("stable_id_if_present", "do_not_equate_to_merchant"),
        "Terminal != merchant by default. Must not be automatically mapped.",
    ),
    EntityRequirement(
        "recipient",
        EntityRequirementStatus.OPTIONAL.value,
        ("user_recipient_count", "user_avg_recipient_amt",
         "user_recipient_fraud_rate", "recipient_novelty"),
        ("stable_id_if_present", "documented_relationship"),
        "Recipient identity only needed if recipient-based features are retained",
    ),
]


# ══════════════════════════════════════════════════════════════════════
# 48-FEATURE ACCEPTANCE MATRIX
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FeatureAcceptanceSpec:
    """Acceptance specification for one PS-14 native feature."""
    feature_name: str
    feature_index: int
    required_source: str
    acceptable_source_types: str
    entity_required: str
    timestamp_required: bool
    label_required: bool
    historical_required: bool
    leakage_risk: str  # LeakageRiskLevel value
    minimum_evidence: str
    acceptance_status: str  # SchemaFieldStatus value

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _build_acceptance_matrix() -> list[FeatureAcceptanceSpec]:
    """Build the 48-feature acceptance matrix.

    Before actual schema inspection, most features are:
    - ACCEPTED_AFTER_SCHEMA_VERIFICATION (if documented as available)
    - UNKNOWN (if not documented)
    - REJECTED (if incompatible with the dataset's documented structure)
    """
    s = SchemaFieldStatus
    l = LeakageRiskLevel

    return [
        FeatureAcceptanceSpec("amt", 0,
            "Transaction amount field", "direct_numeric",
            "none", False, False, False, l.SAFE.value,
            "Published: transaction amounts available",
            s.ACCEPTED.value),
        FeatureAcceptanceSpec("log_amt", 1,
            "Transaction amount", "deterministic_derivation",
            "none", False, False, False, l.SAFE.value,
            "log1p(amt); derivable from documented amount",
            s.ACCEPTED.value),
        FeatureAcceptanceSpec("amt_sq", 2,
            "Transaction amount", "deterministic_derivation",
            "none", False, False, False, l.SAFE.value,
            "amt^2; derivable from documented amount",
            s.ACCEPTED.value),
        FeatureAcceptanceSpec("hr", 3,
            "Transaction timestamp (absolute)", "timestamp_derivation",
            "none", True, False, False, l.SAFE.value,
            "Requires absolute timestamp with clock semantics",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("mn", 4,
            "Transaction timestamp (absolute)", "timestamp_derivation",
            "none", True, False, False, l.SAFE.value,
            "Requires absolute timestamp",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("dow", 5,
            "Transaction timestamp (absolute)", "timestamp_derivation",
            "none", True, False, False, l.SAFE.value,
            "Requires calendar date from absolute timestamp",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("Month", 6,
            "Transaction timestamp (absolute)", "timestamp_derivation",
            "none", True, False, False, l.SAFE.value,
            "Requires calendar date",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("Day", 7,
            "Transaction timestamp (absolute)", "timestamp_derivation",
            "none", True, False, False, l.SAFE.value,
            "Requires calendar date",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("hour_sin", 8,
            "hr", "derived_from_hr",
            "none", True, False, False, l.SAFE.value,
            "sin(2pi*hr/24)",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("hour_cos", 9,
            "hr", "derived_from_hr",
            "none", True, False, False, l.SAFE.value,
            "cos(2pi*hr/24)",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("is_night", 10,
            "hr", "derived_from_hr",
            "none", True, False, False, l.SAFE.value,
            "1 if hr<6 or hr>22",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("is_business_hours", 11,
            "hr", "derived_from_hr",
            "none", True, False, False, l.SAFE.value,
            "1 if 9<=hr<=17",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("chip", 12,
            "Transaction channel type", "categorical",
            "none", False, False, False, l.UNKNOWN.value,
            "Channel (chip/online/swipe) not documented for Worldline",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("is_online", 13,
            "Transaction channel type", "categorical",
            "none", False, False, False, l.UNKNOWN.value,
            "Online indicator not documented",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("is_swipe", 14,
            "Transaction channel type", "categorical",
            "none", False, False, False, l.UNKNOWN.value,
            "Swipe indicator not documented",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("err", 15,
            "Transaction error field", "categorical",
            "none", False, False, False, l.UNKNOWN.value,
            "Transaction errors not documented",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("has_zip", 16,
            "Postal/ZIP code presence", "boolean_derivation",
            "none", False, False, False, l.UNKNOWN.value,
            "ZIP code not documented",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("has_state", 17,
            "Merchant state presence", "boolean_derivation",
            "none", False, False, False, l.UNKNOWN.value,
            "Merchant state not documented",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("is_online_or_no_state", 18,
            "is_online + has_state", "composite_derivation",
            "none", False, False, False, l.UNKNOWN.value,
            "Composite of undocumented fields",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("mcc", 19,
            "Merchant Category Code", "direct_numeric",
            "none", False, False, False, l.UNKNOWN.value,
            "MCC not documented for Worldline",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("mcc_high", 20,
            "mcc", "derived_from_mcc",
            "none", False, False, False, l.UNKNOWN.value,
            "1 if mcc>=5000; depends on mcc",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("mcc_restaurant", 21,
            "mcc", "derived_from_mcc",
            "none", False, False, False, l.UNKNOWN.value,
            "1 if 5812<=mcc<=5814; depends on mcc",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("mcc_gas", 22,
            "mcc", "derived_from_mcc",
            "none", False, False, False, l.UNKNOWN.value,
            "1 if 5541<=mcc<=5542; depends on mcc",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("mcc_grocery", 23,
            "mcc", "derived_from_mcc",
            "none", False, False, False, l.UNKNOWN.value,
            "1 if 5411<=mcc<=5422; depends on mcc",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("mcc_travel", 24,
            "mcc", "derived_from_mcc",
            "none", False, False, False, l.UNKNOWN.value,
            "1 if 3000<=mcc<=3350; depends on mcc",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("mcc_online", 25,
            "mcc", "derived_from_mcc",
            "none", False, False, False, l.UNKNOWN.value,
            "1 if 5967<=mcc<=5969; depends on mcc",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("merchant_id", 26,
            "Stable merchant identifier", "pseudonymous_id",
            "merchant", False, False, False, l.SAFE.value,
            "Required for merchant-level features; must be stable",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("city_id", 27,
            "Stable geographic identifier", "pseudonymous_id",
            "city_geography", False, False, False, l.SAFE.value,
            "Required for city-level features",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("card_id", 28,
            "Stable card identifier", "pseudonymous_id",
            "card", False, False, False, l.SAFE.value,
            "Required if card-level features evaluated",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("user_tx_count", 29,
            "customer_id + transaction history", "historical_aggregation",
            "user_customer", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Count of customer transactions before t",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("card_tx_count", 30,
            "card_id + transaction history", "historical_aggregation",
            "card", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Count of card transactions before t",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("user_avg_amt", 31,
            "customer_id + amounts + history", "historical_aggregation",
            "user_customer", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Average amount for customer before t",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("amt_vs_user_avg", 32,
            "amt + user_avg_amt", "ratio_derivation",
            "user_customer", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "amt / user_avg_amt",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("amt_zscore", 33,
            "amt + user_avg_amt", "zscore_derivation",
            "user_customer", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "(amt - user_avg_amt) / user_avg_amt",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("merch_tx_count", 34,
            "merchant_id + transaction history", "historical_aggregation",
            "merchant", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Count of merchant transactions before t",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("user_merchant_diversity", 35,
            "customer_id + merchant_id + history", "historical_diversity",
            "user_customer + merchant", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Unique merchants per customer before t",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("user_city_diversity", 36,
            "customer_id + city_id + history", "historical_diversity",
            "user_customer + city_geography", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Unique cities per customer before t",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("user_fraud_rate", 37,
            "customer_id + fraud labels + history", "historical_label_aggregation",
            "user_customer", True, True, True, l.REQUIRES_LABEL_HISTORY.value,
            "Fraction of customer fraud before t (only historical labels)",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("merch_fraud_rate", 38,
            "merchant_id + fraud labels + history", "historical_label_aggregation",
            "merchant", True, True, True, l.REQUIRES_LABEL_HISTORY.value,
            "Fraction of merchant fraud before t",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("city_fraud_rate", 39,
            "city_id + fraud labels + history", "historical_label_aggregation",
            "city_geography", True, True, True, l.REQUIRES_LABEL_HISTORY.value,
            "Fraction of city fraud before t",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("high_amt", 40,
            "amt + user_avg_amt", "threshold_derivation",
            "user_customer", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "1 if amt > 2 * user_avg_amt",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("very_high_amt", 41,
            "amt + user_avg_amt", "threshold_derivation",
            "user_customer", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "1 if amt > 5 * user_avg_amt",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("amt_x_hr", 42,
            "amt * hr", "interaction_derivation",
            "none", True, False, False, l.SAFE.value,
            "Amount * hour interaction",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("amt_x_mcc", 43,
            "amt * mcc", "interaction_derivation",
            "none", False, False, False, l.UNKNOWN.value,
            "Amount * MCC; mcc availability unknown",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("amt_x_chip", 44,
            "amt * chip", "interaction_derivation",
            "none", False, False, False, l.UNKNOWN.value,
            "Amount * chip; chip availability unknown",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("amt_x_online", 45,
            "amt * is_online", "interaction_derivation",
            "none", False, False, False, l.UNKNOWN.value,
            "Amount * is_online; is_online availability unknown",
            s.UNKNOWN.value),
        FeatureAcceptanceSpec("amt_x_night", 46,
            "amt * is_night", "interaction_derivation",
            "none", True, False, False, l.SAFE.value,
            "Amount * night indicator",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
        FeatureAcceptanceSpec("user_merch_count", 47,
            "customer_id + merchant_id + history", "historical_count",
            "user_customer + merchant", True, False, True, l.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Unique merchant count per customer before t",
            s.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value),
    ]


ACCEPTANCE_MATRIX: list[FeatureAcceptanceSpec] = _build_acceptance_matrix()


# ══════════════════════════════════════════════════════════════════════
# TEMPORAL LEAKAGE REJECTION RULES
# ══════════════════════════════════════════════════════════════════════

LEAKAGE_REJECTION_RULES: tuple[str, ...] = (
    "REJECT if current transaction label appears in its own features",
    "REJECT if future transaction labels are used for historical features",
    "REJECT if future transactions are used for historical features",
    "REJECT if full-dataset fraud rates are used (must be per-entity historical)",
    "REJECT if post-outcome information is used for pre-transaction features",
    "REJECT if target-derived fields are included without temporal controls",
    "Fraud-rate features must use only labels available strictly before t",
    "Historical features must use only transactions with timestamp < t",
)

HISTORICAL_LABEL_DEPENDENT_FEATURES: tuple[str, ...] = (
    "user_fraud_rate", "merch_fraud_rate", "city_fraud_rate",
    "card_fraud_rate", "user_recipient_fraud_rate",
)


# ══════════════════════════════════════════════════════════════════════
# PRIVACY REJECTION RULES
# ══════════════════════════════════════════════════════════════════════

PRIVACY_REJECTION_RULES: tuple[str, ...] = (
    "REJECT raw card numbers (PAN, card number, card token)",
    "REJECT CVV/CVC codes",
    "REJECT passwords or authentication credentials",
    "REJECT unnecessary personally identifying information",
    "REJECT unnecessary names, email addresses, phone numbers",
    "REJECT unnecessary addresses or location data beyond what is required",
    "REJECT for privacy review if unauthorized PII is present",
    "Prefer stable pseudonymous identifiers over direct identifiers",
)


# ══════════════════════════════════════════════════════════════════════
# POST-ACQUISITION WORKFLOW
# ══════════════════════════════════════════════════════════════════════

POST_ACQUISITION_STEPS: tuple[str, ...] = (
    "1. Verify access authorization",
    "2. Record provenance",
    "3. Hash raw dataset (SHA-256)",
    "4. Freeze immutable raw copy",
    "5. Validate dataset identity (match WORLDLINE_ECOM_2017_NAG)",
    "6. Validate schema against acceptance matrix",
    "7. Validate entity continuity (user, card, merchant, city)",
    "8. Validate timestamps (absolute vs relative, ordering)",
    "9. Validate label provenance (human investigator, expert rules)",
    "10. Build temporal feature derivation plan",
    "11. Run leakage audit (temporal separation, label independence)",
    "12. Build 48-feature compatibility matrix",
    "13. Run Phase 84 dataset ingestion",
    "14. Run Phase 85 external dataset evidence",
    "15. Run feature compatibility (Phase 83)",
    "16. Run evaluation suitability check",
    "17. Only then consider real-world evaluation",
    "18. Never automatically promote the model",
)


# ══════════════════════════════════════════════════════════════════════
# ACCEPTANCE GATE
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AccessPackage:
    """The minimum information an authorized data provider must supply."""
    dataset_id: str
    provenance: dict[str, str]
    schema_columns: tuple[str, ...]
    row_count: int
    has_timestamps: bool
    timestamp_is_absolute: bool | None
    has_customer_ids: bool
    has_merchant_ids: bool
    has_card_ids: bool
    has_city_ids: bool
    has_mcc: bool
    has_channel_type: bool
    has_fraud_labels: bool
    label_methodology: str
    source_hash: str
    data_use_agreement: bool

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["schema_columns"] = list(d["schema_columns"])
        return d


def evaluate_access_package(package: AccessPackage) -> dict[str, Any]:
    """Deterministic acceptance gate for a Worldline access package.

    Returns a dict with:
    - gate_result: AcceptanceGateResult value
    - blocking_reasons: list of specific blockers
    - accepted_features: count of features with ACCEPTED status
    - unknown_features: count of features with UNKNOWN status
    - rejected_features: count of features with REJECTED status
    - next_action: recommended next step
    """
    blocking: list[str] = []
    matrix = ACCEPTANCE_MATRIX

    # 1. Identity verification
    identity_known = package.dataset_id in WORLDLINE_DATASETS
    if not identity_known:
        blocking.append(f"identity_unverified: dataset_id '{package.dataset_id}' not recognized")

    # 2. Provenance completeness
    required_prov = [r for r in PROVENANCE_REQUIREMENTS if r.required]
    provided_prov = package.provenance
    missing_prov = [r.field_name for r in required_prov if r.field_name not in provided_prov or not provided_prov[r.field_name]]
    if missing_prov:
        blocking.append(f"provenance_incomplete: missing {missing_prov}")

    # 3. Access authorization
    if not package.data_use_agreement:
        blocking.append("data_use_agreement_missing")

    # 4. Schema field assessment
    accepted = 0
    unknown = 0
    rejected = 0
    conditional = 0

    for spec in matrix:
        if spec.acceptance_status == SchemaFieldStatus.ACCEPTED.value:
            accepted += 1
        elif spec.acceptance_status == SchemaFieldStatus.ACCEPTED_AFTER_SCHEMA_VERIFICATION.value:
            conditional += 1
        elif spec.acceptance_status == SchemaFieldStatus.UNKNOWN.value:
            unknown += 1
        elif spec.acceptance_status == SchemaFieldStatus.REJECTED.value:
            rejected += 1

    # 5. Entity compatibility
    if not package.has_customer_ids:
        blocking.append("entity_incompatible: customer IDs not available (required for user-level features)")
    if not package.has_merchant_ids:
        blocking.append("entity_incompatible: merchant IDs not available (required for merchant-level features)")

    # 6. Temporal compatibility
    if not package.has_timestamps:
        blocking.append("temporal_incompatible: no transaction timestamps")
    if package.timestamp_is_absolute is False:
        blocking.append("temporal_incompatible: timestamps are relative (clock features NOT_DERIVABLE)")

    # 7. Label provenance
    if not package.has_fraud_labels:
        blocking.append("label_provenance_insufficient: no fraud labels")
    elif "human" not in package.label_methodology.lower() and "investigator" not in package.label_methodology.lower():
        # Not necessarily blocking, but flag
        pass

    # 8. Privacy
    for col in package.schema_columns:
        lower = col.lower()
        if any(term in lower for term in ("pan", "card_number", "cardnumber", "cvv", "cvc", "password", "ssn")):
            blocking.append(f"privacy_review_required: suspicious column '{col}'")

    # 9. Schema size check
    if package.row_count < 1000:
        blocking.append(f"schema_incompatible: only {package.row_count} rows (need >1000)")

    # Determine gate result (check specific blocking categories)
    blocking_str = " ".join(blocking)
    if not package.has_timestamps or not package.has_fraud_labels:
        gate = AcceptanceGateResult.TEMPORAL_INCOMPATIBLE
    elif any("entity_incompatible" in b for b in blocking):
        gate = AcceptanceGateResult.ENTITY_INCOMPATIBLE
    elif any("privacy_review_required" in b for b in blocking):
        gate = AcceptanceGateResult.PRIVACY_REVIEW_REQUIRED
    elif any("identity_unverified" in b for b in blocking):
        gate = AcceptanceGateResult.IDENTITY_UNVERIFIED
    elif any("provenance_incomplete" in b for b in blocking):
        gate = AcceptanceGateResult.PROVENANCE_INCOMPLETE
    elif blocking:
        gate = AcceptanceGateResult.SCHEMA_INCOMPATIBLE
    else:
        gate = AcceptanceGateResult.ELIGIBLE_FOR_PHASE85_ADMISSION

    next_action = (
        "AUTHORIZED_SCHEMA_AND_DATA_ACCESS_REQUEST"
        if gate != AcceptanceGateResult.ELIGIBLE_FOR_PHASE85_ADMISSION
        else "PROCEED_TO_PHASE85_ADMISSION"
    )

    return {
        "gate_result": gate.value,
        "blocking_reasons": blocking,
        "accepted_features": accepted,
        "conditional_features": conditional,
        "unknown_features": unknown,
        "rejected_features": rejected,
        "total_features": len(matrix),
        "next_action": next_action,
    }


# ══════════════════════════════════════════════════════════════════════
# SPECIFICATION RESULT
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AccessSpecResult:
    """Immutable specification result."""
    spec_id: str
    spec_version: str
    primary_target: dict
    secondary_target: dict
    dataset_count: int
    provenance_requirements_count: int
    entity_requirements_count: int
    acceptance_matrix_count: int
    leakage_rules_count: int
    privacy_rules_count: int
    post_acquisition_steps_count: int
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def build_access_spec() -> AccessSpecResult:
    """Build the deterministic access specification."""
    manifest_content = {
        "spec_id": "WORLDLINE-ACCESS-SPEC-90",
        "spec_version": "phase90_v1",
        "primary_target_id": WORLDLINE_ECOM_2017_NAG.dataset_id,
        "secondary_target_id": WORLDLINE_ONLINE_2018.dataset_id,
        "dataset_count": len(WORLDLINE_DATASETS),
        "provenance_requirements": len(PROVENANCE_REQUIREMENTS),
        "entity_requirements": len(ENTITY_REQUIREMENTS),
        "acceptance_matrix": len(ACCEPTANCE_MATRIX),
        "leakage_rules": len(LEAKAGE_REJECTION_RULES),
        "privacy_rules": len(PRIVACY_REJECTION_RULES),
        "post_acquisition_steps": len(POST_ACQUISITION_STEPS),
    }
    canonical = json.dumps(manifest_content, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return AccessSpecResult(
        spec_id="WORLDLINE-ACCESS-SPEC-90",
        spec_version="phase90_v1",
        primary_target=WORLDLINE_ECOM_2017_NAG.to_dict(),
        secondary_target=WORLDLINE_ONLINE_2018.to_dict(),
        dataset_count=len(WORLDLINE_DATASETS),
        provenance_requirements_count=len(PROVENANCE_REQUIREMENTS),
        entity_requirements_count=len(ENTITY_REQUIREMENTS),
        acceptance_matrix_count=len(ACCEPTANCE_MATRIX),
        leakage_rules_count=len(LEAKAGE_REJECTION_RULES),
        privacy_rules_count=len(PRIVACY_REJECTION_RULES),
        post_acquisition_steps_count=len(POST_ACQUISITION_STEPS),
        manifest_hash=manifest_hash,
    )
