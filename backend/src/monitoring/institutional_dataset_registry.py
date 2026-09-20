"""Phase 89: Institutional real-world fraud dataset discovery,
forensic preflight & access registry.

Deterministic, read-only registry of institutional real-world fraud
dataset candidates with 48-feature native preflight assessment.

Does NOT download datasets, modify models, bypass admission, or
change REAL_WORLD_VALIDATION.

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


# ── Controlled status enums ───────────────────────────────────────────

class ProvenanceStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNKNOWN = "unknown"


class LabelProvenanceType(str, Enum):
    HUMAN_INVESTIGATOR = "human_investigator"
    CHARGEBACK = "chargeback"
    INSTITUTIONAL_RULE_SYSTEM = "institutional_rule_system"
    MODEL_GENERATED = "model_generated"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class AccessStatus(str, Enum):
    PUBLIC = "public"
    INSTITUTIONAL_ACCESS_REQUIRED = "institutional_access_required"
    CONFIDENTIAL = "confidential"
    UNKNOWN = "unknown"


class EntityAvailability(str, Enum):
    AVAILABLE = "available"
    ANONYMISED_STABLE = "anonymised_stable"
    REMOVED = "removed"
    UNKNOWN = "unknown"


class CompatibilityStatus(str, Enum):
    PROMISING_UNVERIFIED = "promising_unverified"
    PARTIAL = "partial"
    INCOMPATIBLE = "incompatible"
    VERIFIED = "verified"


class AcquisitionStatus(str, Enum):
    NOT_STARTED = "not_started"
    ACCESS_REQUEST_REQUIRED = "access_request_required"
    AUTHORIZED_PENDING_DATA = "authorized_pending_data"
    DATA_RECEIVED = "data_received"
    BLOCKED = "blocked"


class ContaminationStatus(str, Enum):
    NO_CONTAMINATION_EVIDENCE_FOUND = "no_contamination_evidence_found"
    UNKNOWN = "unknown"
    CONTAMINATED = "contaminated"


class FeatureClassification(str, Enum):
    DIRECT = "direct"
    DOCUMENTED_DERIVABLE = "documented_derivable"
    HISTORICAL_DERIVABLE = "historical_derivable"
    UNKNOWN = "unknown"
    NOT_DERIVABLE = "not_derivable"
    LEAKAGE_RISK = "leakage_risk"


class TemporalLeakageClassification(str, Enum):
    SAFE_PRE_TRANSACTION = "safe_pre_transaction"
    SAFE_WITH_TEMPORAL_WINDOW = "safe_with_temporal_window"
    REQUIRES_LABEL_HISTORY = "requires_label_history"
    LEAKAGE_RISK = "leakage_risk"
    NOT_DERIVABLE = "not_derivable"
    UNKNOWN = "unknown"


# ── Canonical 48-feature list (authoritative) ─────────────────────────

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


# ── Feature preflight entry ──────────────────────────────────────────

@dataclass(frozen=True)
class FeaturePreflight:
    """Deterministic feature compatibility assessment for one PS-14
    native feature against a candidate dataset."""
    feature_name: str
    feature_index: int
    candidate_id: str
    documented_source: str
    derivation_method: str
    entity_required: str
    temporal_requirement: str
    label_dependency: str
    classification: str  # FeatureClassification value
    leakage_classification: str  # TemporalLeakageClassification value
    evidence: str
    uncertainty: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ── Entity continuity entry ──────────────────────────────────────────

@dataclass(frozen=True)
class EntityContinuity:
    """Assessment of entity availability for a candidate."""
    entity_type: str  # user, card, merchant, terminal, city, recipient
    candidate_id: str
    documented_field: str
    availability: str  # EntityAvailability value
    supports_repeated_observations: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ── Dataset registry record ──────────────────────────────────────────

@dataclass(frozen=True)
class InstitutionalDatasetRecord:
    """Immutable registry record for an institutional dataset candidate."""
    dataset_id: str
    dataset_name: str
    provider: str
    institution: str
    provenance_status: str
    transaction_type: str
    date_range: str
    approximate_row_count: int
    fraud_count: int
    fraud_rate: str
    label_type: str  # LabelProvenanceType value
    label_provenance: str
    label_confidence: str
    customer_entity_status: str  # EntityAvailability value
    card_entity_status: str
    merchant_entity_status: str
    terminal_entity_status: str
    timestamp_status: str
    amount_status: str
    merchant_category_status: str
    geographic_status: str
    channel_status: str
    historical_behavior_status: str
    public_access_status: str  # AccessStatus value
    institutional_access_required: bool
    feature_compatibility_status: str  # CompatibilityStatus value
    evidence_sources: tuple[str, ...]
    evidence_timestamp: str
    contamination_status: str  # ContaminationStatus value
    acquisition_status: str  # AcquisitionStatus value
    blockers: tuple[str, ...]
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["evidence_sources"] = list(d["evidence_sources"])
        d["blockers"] = list(d["blockers"])
        return d


# ── Registry result ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RegistryResult:
    """Immutable, deterministic registry result."""
    registry_id: str
    registry_version: str
    records: list[dict]
    feature_preflights: list[dict]
    entity_continuities: list[dict]
    contamination_assessment: dict
    rwv_gate_status: str
    manifest_hash: str
    assessment_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ══════════════════════════════════════════════════════════════════════
# WORLDLINE CANDIDATE
# ══════════════════════════════════════════════════════════════════════

WORLDLINE_RECORD = InstitutionalDatasetRecord(
    dataset_id="INST-WORLDLINE-001",
    dataset_name="Worldline Belgium Real-World E-Commerce Fraud Dataset",
    provider="Worldline",
    institution="Worldline / University合作 (published research)",
    provenance_status=ProvenanceStatus.PARTIALLY_VERIFIED.value,
    transaction_type="real_world_ecommerce",
    date_range="January 2017 - July 2017",
    approximate_row_count=60_000_000,
    fraud_count=0,  # exact count not published
    fraud_rate="unknown (exact rate not published)",
    label_type=LabelProvenanceType.HUMAN_INVESTIGATOR.value,
    label_provenance="Published research states fraud labels were assigned by human investigators",
    label_confidence="partially_verified (methodology not fully documented publicly)",
    customer_entity_status=EntityAvailability.AVAILABLE.value,
    card_entity_status=EntityAvailability.UNKNOWN.value,
    merchant_entity_status=EntityAvailability.UNKNOWN.value,
    terminal_entity_status=EntityAvailability.AVAILABLE.value,
    timestamp_status="absolute_timestamps_documented",
    amount_status="available",
    merchant_category_status=EntityAvailability.UNKNOWN.value,
    geographic_status=EntityAvailability.UNKNOWN.value,
    channel_status=EntityAvailability.UNKNOWN.value,
    historical_behavior_status="potentially_available (60M+ transactions over 7 months)",
    public_access_status=AccessStatus.CONFIDENTIAL.value,
    institutional_access_required=True,
    feature_compatibility_status=CompatibilityStatus.PROMISING_UNVERIFIED.value,
    evidence_sources=(
        "Published research on Worldline fraud detection (peer-reviewed)",
        "Worldline corporate documentation",
        "Academic papers describing the dataset structure",
    ),
    evidence_timestamp=datetime.now(timezone.utc).isoformat(),
    contamination_status=ContaminationStatus.NO_CONTAMINATION_EVIDENCE_FOUND.value,
    acquisition_status=AcquisitionStatus.ACCESS_REQUEST_REQUIRED.value,
    blockers=(
        "Dataset is confidential/not publicly downloadable",
        "Institutional/research access required",
        "Exact schema not publicly documented",
        "Feature compatibility unverified (requires authorized schema inspection)",
        "Exact fraud-label methodology not fully documented",
        "MCC, merchant_id, city_id, card_id availability unknown",
        "channel (chip/online/swipe) availability unknown",
    ),
    next_action="AUTHORIZED_SCHEMA_AND_DATA_ACCESS_REQUEST",
)

# ══════════════════════════════════════════════════════════════════════
# NOVATTI CANDIDATE
# ══════════════════════════════════════════════════════════════════════

NOVATTI_RECORD = InstitutionalDatasetRecord(
    dataset_id="INST-NOVATTI-001",
    dataset_name="Novatti Group Ltd Real-World Transaction Dataset",
    provider="Novatti Group Ltd",
    institution="Novatti Group Ltd",
    provenance_status=ProvenanceStatus.PARTIALLY_VERIFIED.value,
    transaction_type="real_world_merchant_transactions",
    date_range="January 2023 - June 2025",
    approximate_row_count=126_184,
    fraud_count=394,
    fraud_rate="0.31% (394 / 126184)",
    label_type=LabelProvenanceType.CHARGEBACK.value,
    label_provenance="Chargeback validation/rejection outcomes",
    label_confidence="partially_verified (chargeback-derived, not independently adjudicated)",
    customer_entity_status=EntityAvailability.REMOVED.value,
    card_entity_status=EntityAvailability.REMOVED.value,
    merchant_entity_status=EntityAvailability.AVAILABLE.value,
    terminal_entity_status=EntityAvailability.UNKNOWN.value,
    timestamp_status="absolute_timestamps_documented",
    amount_status="available",
    merchant_category_status=EntityAvailability.UNKNOWN.value,
    geographic_status=EntityAvailability.UNKNOWN.value,
    channel_status=EntityAvailability.UNKNOWN.value,
    historical_behavior_status="limited (126K transactions, customer/card identifiers removed)",
    public_access_status=AccessStatus.CONFIDENTIAL.value,
    institutional_access_required=True,
    feature_compatibility_status=CompatibilityStatus.PARTIAL.value,
    evidence_sources=(
        "Published research on Novatti fraud detection",
        "Novatti Group Ltd corporate documentation",
    ),
    evidence_timestamp=datetime.now(timezone.utc).isoformat(),
    contamination_status=ContaminationStatus.NO_CONTAMINATION_EVIDENCE_FOUND.value,
    acquisition_status=AcquisitionStatus.ACCESS_REQUEST_REQUIRED.value,
    blockers=(
        "Dataset is confidential/private",
        "Institutional/research access required",
        "Customer/card/network identifiers substantially removed",
        "Cannot support PS-14 user-level behavioral features (user_tx_count, user_avg_amt, user_fraud_rate, etc.)",
        "Cannot support card-level features (card_tx_count, card_avg_amt, card_fraud_rate, etc.)",
        "Exact schema not publicly documented",
        "MCC, city, channel availability unknown",
        "Label provenance: chargeback-derived (not human investigator)",
    ),
    next_action="AUTHORIZED_SCHEMA_AND_DATA_ACCESS_REQUEST",
)


# ══════════════════════════════════════════════════════════════════════
# 48-FEATURE PREFLIGHT MATRICES
# ══════════════════════════════════════════════════════════════════════

def _build_worldline_preflights() -> list[FeaturePreflight]:
    """48-feature preflight for Worldline Belgium dataset.

    Evidence basis: published research describes customer IDs,
    terminal IDs, transaction timestamps, amounts, and fraud labels
    assigned by human investigators.  MCC, merchant_id, city_id,
    card_id, use_chip/channel are NOT documented.
    """
    p = "INST-WORLDLINE-001"
    return [
        # ── Amount features ──────────────────────────────────────────
        FeaturePreflight("amt", 0, p,
            "Transaction amount (documented)", "DIRECT",
            "none", "none", "none",
            FeatureClassification.DIRECT.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Published research documents transaction amounts",
            "Exact field name not confirmed"),
        FeaturePreflight("log_amt", 1, p,
            "Transaction amount (documented)", "DOCUMENTED_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "log1p of documented transaction amount",
            "Requires confirmed amount field"),
        FeaturePreflight("amt_sq", 2, p,
            "Transaction amount (documented)", "DOCUMENTED_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Square of documented transaction amount",
            "Requires confirmed amount field"),

        # ── Time features ────────────────────────────────────────────
        FeaturePreflight("hr", 3, p,
            "Transaction timestamps (documented as absolute)", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Published research states timestamps are available; assumed absolute",
            "Timestamp semantics (absolute vs relative) not confirmed from primary source"),
        FeaturePreflight("mn", 4, p,
            "Transaction timestamps", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Minute derivable from absolute timestamps",
            "Depends on timestamp being absolute"),
        FeaturePreflight("dow", 5, p,
            "Transaction timestamps", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Day-of-week derivable from calendar date",
            "Depends on timestamp being absolute"),
        FeaturePreflight("Month", 6, p,
            "Transaction timestamps", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Calendar month derivable from date",
            "Depends on timestamp being absolute"),
        FeaturePreflight("Day", 7, p,
            "Transaction timestamps", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Calendar day derivable from date",
            "Depends on timestamp being absolute"),
        FeaturePreflight("hour_sin", 8, p,
            "hr (derivable from timestamps)", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "sin(2pi*hr/24); hr derivable if timestamps absolute",
            "Depends on timestamp being absolute"),
        FeaturePreflight("hour_cos", 9, p,
            "hr (derivable from timestamps)", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "cos(2pi*hr/24); hr derivable if timestamps absolute",
            "Depends on timestamp being absolute"),
        FeaturePreflight("is_night", 10, p,
            "hr (derivable from timestamps)", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Derived from hr; hr derivable if timestamps absolute",
            "Depends on timestamp being absolute"),
        FeaturePreflight("is_business_hours", 11, p,
            "hr (derivable from timestamps)", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Derived from hr; hr derivable if timestamps absolute",
            "Depends on timestamp being absolute"),

        # ── Transaction type features ────────────────────────────────
        FeaturePreflight("chip", 12, p,
            "no documented equivalent", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Channel (chip/online/swipe) not documented",
            "Cannot determine availability without schema inspection"),
        FeaturePreflight("is_online", 13, p,
            "no documented equivalent", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Online indicator not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("is_swipe", 14, p,
            "no documented equivalent", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Swipe indicator not documented",
            "Cannot determine without schema inspection"),

        # ── Error / address features ─────────────────────────────────
        FeaturePreflight("err", 15, p,
            "no documented equivalent", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Transaction errors not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("has_zip", 16, p,
            "no documented equivalent", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "ZIP/postal code not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("has_state", 17, p,
            "no documented equivalent", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Merchant state not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("is_online_or_no_state", 18, p,
            "depends on chip + has_state", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Composite depends on undocumented fields",
            "Cannot determine without schema inspection"),

        # ── MCC features ─────────────────────────────────────────────
        FeaturePreflight("mcc", 19, p,
            "no documented equivalent", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "MCC not documented in published research",
            "Cannot determine without schema inspection"),
        FeaturePreflight("mcc_high", 20, p,
            "depends on mcc", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Derived from mcc; mcc availability unknown",
            "Depends on mcc"),
        FeaturePreflight("mcc_restaurant", 21, p,
            "depends on mcc", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Derived from mcc; mcc availability unknown",
            "Depends on mcc"),
        FeaturePreflight("mcc_gas", 22, p,
            "depends on mcc", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Derived from mcc; mcc availability unknown",
            "Depends on mcc"),
        FeaturePreflight("mcc_grocery", 23, p,
            "depends on mcc", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Derived from mcc; mcc availability unknown",
            "Depends on mcc"),
        FeaturePreflight("mcc_travel", 24, p,
            "depends on mcc", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Derived from mcc; mcc availability unknown",
            "Depends on mcc"),
        FeaturePreflight("mcc_online", 25, p,
            "depends on mcc", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Derived from mcc; mcc availability unknown",
            "Depends on mcc"),

        # ── Entity identity features ─────────────────────────────────
        FeaturePreflight("merchant_id", 26, p,
            "terminal_id documented (not merchant_id)", "UNKNOWN",
            "merchant_entity", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Terminal IDs documented but merchant IDs not confirmed",
            "Terminal_id != merchant_id; cannot assume equivalence"),
        FeaturePreflight("city_id", 27, p,
            "no documented equivalent", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "City/geographic identifier not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("card_id", 28, p,
            "card entity status unknown", "UNKNOWN",
            "card_entity", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Card identifier availability not documented",
            "Cannot determine without schema inspection"),

        # ── Historical velocity features ─────────────────────────────
        FeaturePreflight("user_tx_count", 29, p,
            "customer_id documented", "HISTORICAL_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.HISTORICAL_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Customer IDs documented; transaction count derivable with temporal window",
            "Requires confirmed customer_id field and chronological ordering"),
        FeaturePreflight("card_tx_count", 30, p,
            "card entity unknown", "UNKNOWN",
            "card_entity", "pre_transaction_history", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Card identifier availability not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("user_avg_amt", 31, p,
            "customer_id documented", "HISTORICAL_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.HISTORICAL_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Customer IDs + amounts documented; average derivable with temporal window",
            "Requires confirmed customer_id and chronological ordering"),
        FeaturePreflight("amt_vs_user_avg", 32, p,
            "amt + user_avg_amt (both potentially available)", "HISTORICAL_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.HISTORICAL_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Ratio of amount to user historical average",
            "Requires confirmed user_avg_amt derivation"),
        FeaturePreflight("amt_zscore", 33, p,
            "amt + user_avg_amt (both potentially available)", "HISTORICAL_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.HISTORICAL_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_WITH_TEMPORAL_WINDOW.value,
            "Z-score relative to user historical average",
            "Requires confirmed user_avg_amt derivation"),
        FeaturePreflight("merch_tx_count", 34, p,
            "terminal_id documented (not merchant_id)", "UNKNOWN",
            "merchant_entity", "pre_transaction_history", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Terminal IDs exist but merchant equivalence unknown",
            "Terminal_id may group differently than merchant_id"),
        FeaturePreflight("user_merchant_diversity", 35, p,
            "customer_id + terminal_id (partial)", "UNKNOWN",
            "user_entity + merchant_entity", "pre_transaction_history", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Requires user-merchant interaction; merchant entity unclear",
            "Terminal != merchant; cannot confirm"),
        FeaturePreflight("user_city_diversity", 36, p,
            "city_id not documented", "NOT_DERIVABLE",
            "user_entity + city_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "City identifier not documented",
            "No city entity available from published evidence"),

        # ── Fraud rate features ──────────────────────────────────────
        FeaturePreflight("user_fraud_rate", 37, p,
            "customer_id + fraud labels documented", "LEAKAGE_RISK",
            "user_entity", "pre_transaction_label_history", "fraud_label",
            FeatureClassification.LEAKAGE_RISK.value,
            TemporalLeakageClassification.LEAKAGE_RISK.value,
            "Customer IDs + fraud labels exist; rate computable with strict temporal window but requires label-history separation",
            "Must use only labels from transactions strictly before t; requires confirmed customer_id and label availability"),
        FeaturePreflight("merch_fraud_rate", 38, p,
            "terminal_id + fraud labels (partial)", "LEAKAGE_RISK",
            "merchant_entity", "pre_transaction_label_history", "fraud_label",
            FeatureClassification.LEAKAGE_RISK.value,
            TemporalLeakageClassification.LEAKAGE_RISK.value,
            "Terminal IDs + fraud labels may exist; merchant equivalence uncertain",
            "Terminal != merchant; label-history window required"),
        FeaturePreflight("city_fraud_rate", 39, p,
            "city_id not documented", "NOT_DERIVABLE",
            "city_entity", "pre_transaction_label_history", "fraud_label",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "City identifier not documented",
            "No city entity available"),

        # ── Amount interaction features ──────────────────────────────
        FeaturePreflight("high_amt", 40, p,
            "amt + user_avg_amt (potentially available)", "HISTORICAL_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.HISTORICAL_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_WITH_TEMPORAL_WINDOW.value,
            "1 if amt > 2*user_avg_amt; both potentially available",
            "Requires confirmed user_avg_amt derivation"),
        FeaturePreflight("very_high_amt", 41, p,
            "amt + user_avg_amt (potentially available)", "HISTORICAL_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.HISTORICAL_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_WITH_TEMPORAL_WINDOW.value,
            "1 if amt > 5*user_avg_amt; both potentially available",
            "Requires confirmed user_avg_amt derivation"),
        FeaturePreflight("amt_x_hr", 42, p,
            "amt * hr (both potentially available)", "HISTORICAL_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.HISTORICAL_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Amount * hour interaction; both derivable if timestamps absolute",
            "Depends on timestamp being absolute"),
        FeaturePreflight("amt_x_mcc", 43, p,
            "mcc not documented", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "MCC not documented",
            "Cannot derive without MCC"),
        FeaturePreflight("amt_x_chip", 44, p,
            "chip not documented", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Channel indicator not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("amt_x_online", 45, p,
            "is_online not documented", "UNKNOWN",
            "none", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Online indicator not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("amt_x_night", 46, p,
            "amt * is_night (hr derivable if timestamps absolute)", "HISTORICAL_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.HISTORICAL_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Amount * night indicator; hr derivable if timestamps absolute",
            "Depends on timestamp being absolute"),

        # ── User merch count ─────────────────────────────────────────
        FeaturePreflight("user_merch_count", 47, p,
            "customer_id + terminal_id (partial)", "UNKNOWN",
            "user_entity + merchant_entity", "pre_transaction_history", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Requires user-merchant interaction; merchant entity uncertain",
            "Terminal != merchant; cannot confirm"),
    ]


def _build_novatti_preflights() -> list[FeaturePreflight]:
    """48-feature preflight for Novatti Group dataset.

    Evidence basis: 126K transactions, chargeback-derived labels,
    merchant-level history, but customer/card/network identifiers
    substantially removed.
    """
    p = "INST-NOVATTI-001"
    return [
        # ── Amount features ──────────────────────────────────────────
        FeaturePreflight("amt", 0, p,
            "Transaction amounts (documented)", "DIRECT",
            "none", "none", "none",
            FeatureClassification.DIRECT.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Published research documents transaction amounts",
            "Exact field name not confirmed"),
        FeaturePreflight("log_amt", 1, p,
            "Transaction amounts", "DOCUMENTED_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "log1p of documented amount",
            "Requires confirmed amount field"),
        FeaturePreflight("amt_sq", 2, p,
            "Transaction amounts", "DOCUMENTED_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Square of documented amount",
            "Requires confirmed amount field"),

        # ── Time features ────────────────────────────────────────────
        FeaturePreflight("hr", 3, p,
            "Transaction timestamps (Jan 2023-Jun 2025)", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Published research describes temporal range; assumed absolute",
            "Timestamp semantics not confirmed from primary source"),
        FeaturePreflight("mn", 4, p,
            "Transaction timestamps", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Minute derivable from absolute timestamps",
            "Depends on timestamp being absolute"),
        FeaturePreflight("dow", 5, p,
            "Transaction timestamps", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Day-of-week derivable from date",
            "Depends on timestamp being absolute"),
        FeaturePreflight("Month", 6, p,
            "Transaction timestamps", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Calendar month derivable from date",
            "Depends on timestamp being absolute"),
        FeaturePreflight("Day", 7, p,
            "Transaction timestamps", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Calendar day derivable from date",
            "Depends on timestamp being absolute"),
        FeaturePreflight("hour_sin", 8, p,
            "hr (derivable from timestamps)", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "sin(2pi*hr/24); hr derivable if timestamps absolute",
            "Depends on timestamp being absolute"),
        FeaturePreflight("hour_cos", 9, p,
            "hr (derivable from timestamps)", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "cos(2pi*hr/24); hr derivable if timestamps absolute",
            "Depends on timestamp being absolute"),
        FeaturePreflight("is_night", 10, p,
            "hr (derivable from timestamps)", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Derived from hr",
            "Depends on timestamp being absolute"),
        FeaturePreflight("is_business_hours", 11, p,
            "hr (derivable from timestamps)", "DOCUMENTED_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.DOCUMENTED_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Derived from hr",
            "Depends on timestamp being absolute"),

        # ── Transaction type features ────────────────────────────────
        FeaturePreflight("chip", 12, p,
            "no documented equivalent", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Channel (chip/online/swipe) not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("is_online", 13, p,
            "no documented equivalent", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Online indicator not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("is_swipe", 14, p,
            "no documented equivalent", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Swipe indicator not documented",
            "Cannot determine without schema inspection"),

        # ── Error / address features ─────────────────────────────────
        FeaturePreflight("err", 15, p,
            "no documented equivalent", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Transaction errors not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("has_zip", 16, p,
            "no documented equivalent", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "ZIP/postal code not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("has_state", 17, p,
            "no documented equivalent", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Merchant state not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("is_online_or_no_state", 18, p,
            "depends on chip + has_state", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Composite depends on undocumented fields",
            "Cannot determine without schema inspection"),

        # ── MCC features ─────────────────────────────────────────────
        FeaturePreflight("mcc", 19, p,
            "no documented equivalent", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "MCC not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("mcc_high", 20, p,
            "depends on mcc", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Derived from mcc; mcc unknown",
            "Depends on mcc"),
        FeaturePreflight("mcc_restaurant", 21, p,
            "depends on mcc", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Derived from mcc; mcc unknown",
            "Depends on mcc"),
        FeaturePreflight("mcc_gas", 22, p,
            "depends on mcc", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Derived from mcc; mcc unknown",
            "Depends on mcc"),
        FeaturePreflight("mcc_grocery", 23, p,
            "depends on mcc", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Derived from mcc; mcc unknown",
            "Depends on mcc"),
        FeaturePreflight("mcc_travel", 24, p,
            "depends on mcc", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Derived from mcc; mcc unknown",
            "Depends on mcc"),
        FeaturePreflight("mcc_online", 25, p,
            "depends on mcc", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Derived from mcc; mcc unknown",
            "Depends on mcc"),

        # ── Entity identity features ─────────────────────────────────
        FeaturePreflight("merchant_id", 26, p,
            "merchant-level info documented (anonymised)", "UNKNOWN",
            "merchant_entity", "none", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Merchant-level history documented but anonymised",
            "Anonymisation scheme unknown; cannot confirm stable merchant_id"),
        FeaturePreflight("city_id", 27, p,
            "no documented equivalent", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "City identifier not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("card_id", 28, p,
            "card identifiers substantially removed", "NOT_DERIVABLE",
            "card_entity", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Card identifiers substantially removed per documentation",
            "Cannot derive card features without card identifiers"),

        # ── Historical velocity features ─────────────────────────────
        FeaturePreflight("user_tx_count", 29, p,
            "customer identifiers substantially removed", "NOT_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Customer identifiers substantially removed",
            "Cannot derive user-level features without customer_id"),
        FeaturePreflight("card_tx_count", 30, p,
            "card identifiers substantially removed", "NOT_DERIVABLE",
            "card_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Card identifiers substantially removed",
            "Cannot derive card-level features without card_id"),
        FeaturePreflight("user_avg_amt", 31, p,
            "customer identifiers substantially removed", "NOT_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Customer identifiers substantially removed",
            "Cannot derive user-level features without customer_id"),
        FeaturePreflight("amt_vs_user_avg", 32, p,
            "depends on user_avg_amt", "NOT_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Depends on user_avg_amt which is not derivable",
            "Customer identifiers removed"),
        FeaturePreflight("amt_zscore", 33, p,
            "depends on user_avg_amt", "NOT_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Depends on user_avg_amt which is not derivable",
            "Customer identifiers removed"),
        FeaturePreflight("merch_tx_count", 34, p,
            "merchant-level info documented (anonymised)", "UNKNOWN",
            "merchant_entity", "pre_transaction_history", "none",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Merchant-level history documented; anonymisation unknown",
            "Cannot confirm stable merchant entity without schema inspection"),
        FeaturePreflight("user_merchant_diversity", 35, p,
            "customer identifiers removed", "NOT_DERIVABLE",
            "user_entity + merchant_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Cannot compute without customer identifiers",
            "Customer identifiers substantially removed"),
        FeaturePreflight("user_city_diversity", 36, p,
            "customer + city identifiers both missing", "NOT_DERIVABLE",
            "user_entity + city_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Neither customer nor city identifiers available",
            "Both entities missing"),

        # ── Fraud rate features ──────────────────────────────────────
        FeaturePreflight("user_fraud_rate", 37, p,
            "customer identifiers removed", "NOT_DERIVABLE",
            "user_entity", "pre_transaction_label_history", "fraud_label",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Cannot compute user fraud rate without customer identifiers",
            "Customer identifiers substantially removed"),
        FeaturePreflight("merch_fraud_rate", 38, p,
            "merchant-level info documented (anonymised)", "UNKNOWN",
            "merchant_entity", "pre_transaction_label_history", "fraud_label",
            FeatureClassification.UNKNOWN.value,
            TemporalLeakageClassification.UNKNOWN.value,
            "Merchant-level history + fraud labels may exist",
            "Cannot confirm without schema inspection; label-history window required"),
        FeaturePreflight("city_fraud_rate", 39, p,
            "city identifiers not documented", "NOT_DERIVABLE",
            "city_entity", "pre_transaction_label_history", "fraud_label",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "City identifier not documented",
            "No city entity available"),

        # ── Amount interaction features ──────────────────────────────
        FeaturePreflight("high_amt", 40, p,
            "depends on user_avg_amt", "NOT_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Depends on user_avg_amt which is not derivable",
            "Customer identifiers removed"),
        FeaturePreflight("very_high_amt", 41, p,
            "depends on user_avg_amt", "NOT_DERIVABLE",
            "user_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Depends on user_avg_amt which is not derivable",
            "Customer identifiers removed"),
        FeaturePreflight("amt_x_hr", 42, p,
            "amt * hr (both potentially available)", "HISTORICAL_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.HISTORICAL_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Amount * hour; both derivable if timestamps absolute",
            "Depends on timestamp being absolute"),
        FeaturePreflight("amt_x_mcc", 43, p,
            "mcc not documented", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "MCC not documented",
            "Cannot derive without MCC"),
        FeaturePreflight("amt_x_chip", 44, p,
            "chip not documented", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Channel indicator not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("amt_x_online", 45, p,
            "is_online not documented", "NOT_DERIVABLE",
            "none", "none", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Online indicator not documented",
            "Cannot determine without schema inspection"),
        FeaturePreflight("amt_x_night", 46, p,
            "amt * is_night (hr derivable if timestamps absolute)", "HISTORICAL_DERIVABLE",
            "none", "absolute_timestamp_needed", "none",
            FeatureClassification.HISTORICAL_DERIVABLE.value,
            TemporalLeakageClassification.SAFE_PRE_TRANSACTION.value,
            "Amount * night indicator; hr derivable if timestamps absolute",
            "Depends on timestamp being absolute"),

        # ── User merch count ─────────────────────────────────────────
        FeaturePreflight("user_merch_count", 47, p,
            "customer identifiers removed", "NOT_DERIVABLE",
            "user_entity + merchant_entity", "pre_transaction_history", "none",
            FeatureClassification.NOT_DERIVABLE.value,
            TemporalLeakageClassification.NOT_DERIVABLE.value,
            "Cannot compute without customer identifiers",
            "Customer identifiers substantially removed"),
    ]


# ══════════════════════════════════════════════════════════════════════
# ENTITY CONTINUITY ASSESSMENTS
# ══════════════════════════════════════════════════════════════════════

WORLDLINE_ENTITIES = [
    EntityContinuity("user", "INST-WORLDLINE-001", "customer_id",
        EntityAvailability.AVAILABLE.value, True,
        "Customer IDs documented in published research"),
    EntityContinuity("card", "INST-WORLDLINE-001", "unknown",
        EntityAvailability.UNKNOWN.value, False,
        "Card identifier availability not documented"),
    EntityContinuity("merchant", "INST-WORLDLINE-001", "terminal_id (not merchant_id)",
        EntityAvailability.UNKNOWN.value, False,
        "Terminal IDs documented but merchant_id equivalence unknown"),
    EntityContinuity("terminal", "INST-WORLDLINE-001", "terminal_id",
        EntityAvailability.AVAILABLE.value, True,
        "Terminal IDs documented in published research"),
    EntityContinuity("city", "INST-WORLDLINE-001", "unknown",
        EntityAvailability.UNKNOWN.value, False,
        "Geographic/city identifier not documented"),
    EntityContinuity("recipient", "INST-WORLDLINE-001", "unknown",
        EntityAvailability.REMOVED.value, False,
        "No recipient relationship documented"),
]

NOVATTI_ENTITIES = [
    EntityContinuity("user", "INST-NOVATTI-001", "customer identifiers substantially removed",
        EntityAvailability.REMOVED.value, False,
        "Customer/card/network identifiers substantially removed per documentation"),
    EntityContinuity("card", "INST-NOVATTI-001", "card identifiers substantially removed",
        EntityAvailability.REMOVED.value, False,
        "Card identifiers substantially removed per documentation"),
    EntityContinuity("merchant", "INST-NOVATTI-001", "merchant-level (anonymised)",
        EntityAvailability.ANONYMISED_STABLE.value, True,
        "Merchant-level history documented but anonymised"),
    EntityContinuity("terminal", "INST-NOVATTI-001", "unknown",
        EntityAvailability.UNKNOWN.value, False,
        "Terminal identifier not documented"),
    EntityContinuity("city", "INST-NOVATTI-001", "unknown",
        EntityAvailability.UNKNOWN.value, False,
        "Geographic/city identifier not documented"),
    EntityContinuity("recipient", "INST-NOVATTI-001", "unknown",
        EntityAvailability.REMOVED.value, False,
        "No recipient relationship documented"),
]


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def get_preflight_summary(candidate_id: str, preflights: list[FeaturePreflight]) -> dict[str, int]:
    """Count features by classification for a candidate."""
    counts: dict[str, int] = {}
    for pf in preflights:
        c = pf.classification
        counts[c] = counts.get(c, 0) + 1
    return counts


def run_registry() -> RegistryResult:
    """Run the complete deterministic institutional dataset registry.

    Produces an immutable result with registry records, 48-feature
    preflight matrices, entity continuity assessments, contamination
    audit, and a deterministic manifest hash.
    """
    worldline_preflights = _build_worldline_preflights()
    novatti_preflights = _build_novatti_preflights()

    wl_summary = get_preflight_summary("INST-WORLDLINE-001", worldline_preflights)
    nv_summary = get_preflight_summary("INST-NOVATTI-001", novatti_preflights)

    # Contamination assessment
    contamination = {
        "worldline": ContaminationStatus.NO_CONTAMINATION_EVIDENCE_FOUND.value,
        "novatti": ContaminationStatus.NO_CONTAMINATION_EVIDENCE_FOUND.value,
        "note": (
            "NO CONTAMINATION EVIDENCE FOUND IN INSPECTED PS-14 ARTIFACTS. "
            "PS-14 training uses synthetic PaySim-derived data. Neither "
            "Worldline nor Novatti appears in training scripts, release "
            "manifests, feature engineering, validation, or test fixtures."
        ),
    }

    # Build deterministic manifest
    manifest_content = {
        "registry_id": "INSTITUTIONAL-DATASET-REGISTRY-89",
        "registry_version": "phase89_v1",
        "candidate_count": 2,
        "worldline_compatibility": CompatibilityStatus.PROMISING_UNVERIFIED.value,
        "novatti_compatibility": CompatibilityStatus.PARTIAL.value,
        "worldline_derivable": wl_summary.get("direct", 0) + wl_summary.get("documented_derivable", 0) + wl_summary.get("historical_derivable", 0),
        "worldline_not_derivable": wl_summary.get("not_derivable", 0),
        "worldline_unknown": wl_summary.get("unknown", 0),
        "worldline_leakage": wl_summary.get("leakage_risk", 0),
        "novatti_derivable": nv_summary.get("direct", 0) + nv_summary.get("documented_derivable", 0) + nv_summary.get("historical_derivable", 0),
        "novatti_not_derivable": nv_summary.get("not_derivable", 0),
        "novatti_unknown": nv_summary.get("unknown", 0),
        "novatti_leakage": nv_summary.get("leakage_risk", 0),
        "rwv_gate": "BLOCKED_PENDING_ELIGIBLE_DATASET",
    }
    canonical = json.dumps(manifest_content, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return RegistryResult(
        registry_id="INSTITUTIONAL-DATASET-REGISTRY-89",
        registry_version="phase89_v1",
        records=[WORLDLINE_RECORD.to_dict(), NOVATTI_RECORD.to_dict()],
        feature_preflights=[pf.to_dict() for pf in worldline_preflights] + [pf.to_dict() for pf in novatti_preflights],
        entity_continuities=[e.to_dict() for e in WORLDLINE_ENTITIES + NOVATTI_ENTITIES],
        contamination_assessment=contamination,
        rwv_gate_status="BLOCKED_PENDING_ELIGIBLE_DATASET",
        manifest_hash=manifest_hash,
        assessment_timestamp=datetime.now(timezone.utc).isoformat(),
    )
