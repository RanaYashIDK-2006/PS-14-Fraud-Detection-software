"""Phase 106: Public Multi-Dataset External Benchmark Discovery & Registry.

Governed, reproducible registry of legitimately documented PUBLIC fraud
detection datasets suitable for candidate benchmark evaluation.  This phase
is NOT real-world validation:

  - NO external dataset acquired, NO provider contacted, NO data transfer
  - NO model modified / retrained / tuned, NO threshold changed
  - NO PromotionToken, NO RWVPromotionEvidence, NO RWV session
  - NO alternate feature mapping: compatibility is measured by REUSING the
    authoritative Phase 104 preflight over the canonical pipeline
        21 domain/runtime features -> map_raw_to_native() -> 48 native
        features -> Altman-Native ensemble
  - NO bypass parameters (force / allow_unverified / skip_validation /
    override / admin_override or equivalents)

The benchmark manifest produced here is a BenchmarkManifest ONLY.  It is
NOT ProviderEvidence, NOT a QualificationResult, NOT RWVPromotionEvidence
and NOT a PromotionToken; no existing RWV or promotion gate accepts it.
Phase 104 remains the sole dataset-qualification authority, Phase 96 the
sole RWV execution harness, Phase 46 the sole promotion authority.

Multi-source rows are ALWAYS described as a "multi-source public benchmark"
— never as one financial institution's dataset.  Missing metadata is never
invented; unestablished license/provenance/semantics fails closed into
GROUP_D.

STATUS: IMPLEMENTED
SYSTEM_READINESS: SYSTEM_READY_PENDING_ELIGIBLE_DATASET
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
PROMOTION: PROMOTION_GATE_REQUIRED
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from src.monitoring.external_dataset_contract import (
    CANONICAL_TRANSFORMATION,
    CONTRACT_VERSION,
    AccessAuthorityClaim,
    ContaminationClaim,
    DatasetIdentityClaim,
    EntityContinuityClaim,
    ExternalDatasetContract,
    FeatureCompatibilityClaim,
    FeaturePreflightItem,
    IndependenceClaim,
    LabelProvenanceClaim,
    ProviderIdentityClaim,
    REQUIREMENT_KEYS,
    TemporalSemanticsClaim,
    UsagePermissionClaim,
    blocking_features,
    build_known_candidate_contract,
    canonical_json,
    preflight_feature_compatibility,
)
from src.monitoring.feature_contract import ML_FEATURE_ORDER, ML_FEATURE_VERSION
from src.monitoring.phase103_production_readiness_closure import (
    PROMOTION_STATE,
    REAL_WORLD_VALIDATION,
    SYSTEM_READINESS,
)
from src.monitoring.real_world_evaluation_protocol import (
    PRODUCTION_THRESHOLD,
    compute_metrics,
)
from src.monitoring.rwv_readiness_audit import MODEL_ID, RELEASE_ID
from src.monitoring.rwv_reproducibility import (
    NATIVE_FEATURE_VERSION,
    PREPROCESSING_HASH,
    RULE_HASH,
)
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES

# ══════════════════════════════════════════════════════════════════════
# VERSIONS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════

REGISTRY_VERSION = "phase106_registry_v1"
MANIFEST_VERSION = "phase106_benchmark_manifest_v1"
EVAL_CONFIG_VERSION = "phase106_eval_config_v1"

BENCHMARK_ID = "phase106_multi_source_public_benchmark"
BENCHMARK_CONTEXT_ID = "phase106_public_benchmark_registry"
# Fixed manifest instant — deterministic, never a wall-clock read.
MANIFEST_CREATED_AT = "2026-09-22T00:00:00+00:00"

BENCHMARK_DESCRIPTOR = "multi-source public benchmark"
DOMAIN_FEATURE_COUNT = len(ML_FEATURE_ORDER)      # 21
NATIVE_FEATURE_COUNT = len(ALTMAN_NATIVE_FEATURES)  # 48

# Manual acquisition only — no automated transfer exists anywhere here.
EXPECTED_LOCAL_ROOT = "data/external"

# The ONLY cross-source field with defensible common semantics.  Amount,
# time and entity columns are deliberately NOT harmonized: currency units,
# clock semantics and identifier spaces are not established across sources.
COMMON_BENCHMARK_FIELDS: tuple[str, ...] = ("normalized_label",)
COMMON_FIELD_RATIONALE = (
    "Only the normalized label has defensible common semantics across "
    "sources.  Amount units/currency, timestamp anchors and entity "
    "identifier spaces are NOT established across sources, so those "
    "columns stay source-local and are never pooled cross-source."
)

# Phrases that would misrepresent the multi-source benchmark as a single
# institution's data (spec: NEVER imply one financial institution).
FORBIDDEN_SINGLE_SOURCE_CLAIMS: tuple[str, ...] = (
    "single financial institution",
    "one financial institution",
    "one bank's transactions",
    "our institution's transactions",
    "single-institution dataset",
    "single institution dataset",
    "one provider's dataset",
)

# Label-derived / post-outcome / evaluation-derived column patterns.
# Any matched column is excluded from benchmark features by construction.
LEAKAGE_FIELD_PATTERNS: tuple[str, ...] = (
    r"^isfraud$",
    r"^is_fraud$",
    r"^fraudbool$",
    r"^fraudrisk$",
    r"^class$",
    r"^label$",
    r"^target$",
    r"isflaggedfraud",
    r"flagged",
    r"^chargeback",
    r"^outcome",
    r"^reversed",
    r"^newbalance",
    r"_after$",
    r"^post_",
    r"model_score",
    r"predicted",
    r"^prediction",
)

# Synthetic-augmentation markers: rows carrying these are leakage-risk.
SYNTHETIC_AUGMENTATION_MARKERS: tuple[str, ...] = (
    "smote",
    "synthetic_augmented",
    "augmented_synthetic",
    "gan_generated",
    "generated_by_model",
)

# Parameters that must NEVER appear on any public function here.
FORBIDDEN_BYPASS_PARAMETERS: frozenset[str] = frozenset({
    "force", "allow_unverified", "skip_validation", "override",
    "admin_override", "bypass",
})

# The benchmark manifest is never accepted by these gate authorities.
BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE: tuple[str, ...] = (
    "ProviderEvidence",
    "QualificationResult",
    "RWVPromotionEvidence",
    "PromotionToken",
)

NOT_DOCUMENTED = "not_documented"


def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# CONTROLLED ENUMS
# ══════════════════════════════════════════════════════════════════════

class DatasetClassification(str, Enum):
    """Spec §DATASET CLASSIFICATION — simulated is never called real."""
    PUBLIC_REAL_ANONYMIZED = "public_real_anonymized"
    PUBLIC_RESEARCH_DERIVED = "public_research_derived"
    PUBLIC_SIMULATED = "public_simulated"
    PUBLIC_SYNTHETIC = "public_synthetic"
    UNKNOWN = "unknown"


# Classifications that may never be described as real-world data.
NON_REAL_WORLD_CLASSIFICATIONS: frozenset[str] = frozenset({
    DatasetClassification.PUBLIC_SIMULATED.value,
    DatasetClassification.PUBLIC_SYNTHETIC.value,
    DatasetClassification.PUBLIC_RESEARCH_DERIVED.value,
    DatasetClassification.UNKNOWN.value,
})


class BenchmarkGroup(str, Enum):
    """Spec §BENCHMARK GROUPS."""
    GROUP_A = "group_a"   # fully compatible with the 21->48 contract
    GROUP_B = "group_b"   # partial: limited component/feature evaluation
    GROUP_C = "group_c"   # reference-only: incompatible with native model
    GROUP_D = "group_d"   # rejected: provenance/license/semantics/leakage


class HarmonizationStatus(str, Enum):
    """Row-level harmonization outcome (spec §MULTI-DATASET HARMONIZATION)."""
    HARMONIZED = "harmonized"
    EXCLUDED_UNMAPPED_LABEL = "excluded_unmapped_label"
    EXCLUDED_FOREIGN_SOURCE = "excluded_foreign_source"


class EvaluationReadiness(str, Enum):
    """Spec §EVALUATION readiness states."""
    READY_NATIVE = "ready_for_native_evaluation"
    COMPONENT_ONLY = "component_evaluation_only"
    REFERENCE_ONLY = "reference_only"
    REJECTED_NOT_ESTABLISHED = "rejected_governance_not_established"


GROUP_READINESS: dict[str, str] = {
    BenchmarkGroup.GROUP_A.value: EvaluationReadiness.READY_NATIVE.value,
    BenchmarkGroup.GROUP_B.value: EvaluationReadiness.COMPONENT_ONLY.value,
    BenchmarkGroup.GROUP_C.value: EvaluationReadiness.REFERENCE_ONLY.value,
    BenchmarkGroup.GROUP_D.value: (
        EvaluationReadiness.REJECTED_NOT_ESTABLISHED.value),
}


# ══════════════════════════════════════════════════════════════════════
# DATASET REGISTRY RECORD (spec §DATASET DISCOVERY — never invent values)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BenchmarkDataset:
    """Immutable discovery record for one public benchmark candidate.

    Undocumented metadata stays at the documented sentinel (empty string /
    None / NOT_DOCUMENTED); nothing is inferred to make a dataset look
    better than its documentation supports.
    """
    dataset_id: str
    canonical_name: str
    source_reference: str
    provider_author: str
    publication_reference: str
    license_terms: str            # "" when not established
    license_documented: bool
    license_reference: str
    access_type: str              # descriptive only — never authorization
    dataset_description: str
    transaction_count: int | None  # None = not transcribed from source
    fraud_count: int | None
    fraud_rate: float | None
    counts_reference: str
    timestamp_semantics: str
    timestamp_supports_ordering: bool
    timestamp_supports_absolute_time: bool
    entity_identifiers: tuple[str, ...]
    merchant_information: str
    amount_information: str
    transaction_type_information: str
    geography_information: str
    device_information: str
    mcc_information: str
    historical_aggregate_information: str
    label_column: str
    label_semantics: str
    label_provenance: str
    classification: str           # DatasetClassification value
    known_transformations: tuple[str, ...]
    known_leakage_risks: tuple[str, ...]
    documentation_confidence: str  # high | medium | low
    provenance_confidence: str     # high | medium | low
    semantics_established: bool
    leakage_clear: bool           # identified risks excluded from scope
    schema_columns: tuple[str, ...]
    requirement_availability: tuple[tuple[str, str], ...]
    expected_local_path: str
    acquisition_status: str       # always "not_acquired" in this phase

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        valid = {c.value for c in DatasetClassification}
        if self.classification not in valid:
            raise ValueError(f"invalid classification: {self.classification}")
        keys = {k for k, _ in self.requirement_availability}
        if keys != set(REQUIREMENT_KEYS):
            raise ValueError(
                f"{self.dataset_id}: requirement_availability must declare "
                f"exactly the {len(REQUIREMENT_KEYS)} canonical requirement keys")
        if self.license_documented and not self.license_terms:
            raise ValueError(f"{self.dataset_id}: documented license needs terms")
        if not self.license_documented and self.license_terms:
            raise ValueError(
                f"{self.dataset_id}: undocumented license must not carry terms")
        if self.acquisition_status != "not_acquired":
            raise ValueError("phase 106 never acquires datasets")

    @property
    def schema_hash(self) -> str:
        return _stable_hash(list(self.schema_columns))

    @property
    def timestamp_columns_present(self) -> bool:
        """Documented schema includes a timestamp column (never assumed)."""
        col = TIMESTAMP_COLUMNS.get(self.dataset_id)
        return bool(col) and col in self.schema_columns

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _avail(base: str = "unavailable",
           **overrides: str) -> tuple[tuple[str, str], ...]:
    """Requirement availability tuple: canonical keys only (fail closed)."""
    values = {k: base for k in REQUIREMENT_KEYS}
    unknown = set(overrides) - set(REQUIREMENT_KEYS)
    if unknown:
        raise ValueError(f"unknown requirement keys: {sorted(unknown)}")
    values.update(overrides)
    return tuple(sorted(values.items()))


_X = "unavailable"
_D = "deterministically_derivable"
_R = "directly_available"
_AM = "ambiguous"

# IEEE-CIS: authoritative availability REUSED from the Phase 104 contract
# built on the Phase 95 published record — single source of truth.
_IEEE_AVAILABILITY = build_known_candidate_contract(
    "IEEE_CIS").feature_compatibility.requirement_availability

# ULB: documented schema = Time, Amount, V1..V28 (PCA), Class only.
# Everything else is documented-absent; Time is a relative offset only.
_ULB_AVAILABILITY = _avail(_X, amount=_R)

# PaySim: step/type/amount/nameOrig balances/nameDest/isFraud/isFlaggedFraud.
# Per-account history and prior-label rates are derivable from ordered
# rows; balance columns and isFlaggedFraud are leakage and never mapped.
_PAYSIM_AVAILABILITY = _avail(
    _X, amount=_R, user_history=_D, labeled_user_history=_D)

# BankSim: step/age/gender/category/amount/merchant/fraudRisk.
# 'category' semantics are NOT established as MCC (ambiguous, not imputed);
# no customer identifier is documented so user-side history stays absent.
_BANKSIM_AVAILABILITY = _avail(
    _X, amount=_R, merchant_id=_R,
    merchant_history=_D, labeled_merchant_history=_D, mcc=_AM)

# BAF: full column dictionary not transcribed in this offline phase, so
# every source requirement is ambiguous — nothing is assumed available.
_BAF_AVAILABILITY = _avail(_AM)

# Label / timestamp / amount source columns per dataset (documented only).
LABEL_COLUMNS: dict[str, str] = {
    "IEEE_CIS": "isFraud",
    "ULB_CREDIT_CARD_FRAUD": "Class",
    "PAYSIM": "isFraud",
    "BANKSIM": "fraudRisk",
    "BAF_BANK_ACCOUNT_FRAUD": "fraud_bool",
}
TIMESTAMP_COLUMNS: dict[str, str | None] = {
    "IEEE_CIS": "TransactionDT",
    "ULB_CREDIT_CARD_FRAUD": "Time",
    "PAYSIM": "step",
    "BANKSIM": "step",
    "BAF_BANK_ACCOUNT_FRAUD": None,
}
AMOUNT_COLUMNS: dict[str, str | None] = {
    "IEEE_CIS": "TransactionAmt",
    "ULB_CREDIT_CARD_FRAUD": "Amount",
    "PAYSIM": "amount",
    "BANKSIM": "amount",
    "BAF_BANK_ACCOUNT_FRAUD": None,
}


def _ulb_schema() -> tuple[str, ...]:
    return tuple(["Time", "Amount"] +
                 [f"V{i}" for i in range(1, 29)] + ["Class"])


BENCHMARK_DATASETS: tuple[BenchmarkDataset, ...] = (
    BenchmarkDataset(
        dataset_id="IEEE_CIS",
        canonical_name="IEEE-CIS Fraud Detection (Kaggle competition 2019)",
        source_reference="Kaggle competition: ieee-fraud-detection",
        provider_author="Vesta Corporation via IEEE CIS competition",
        publication_reference="Kaggle competition description; Phase 95/88 "
                              "in-project documentation records",
        license_terms="Kaggle competition terms (not an institutional "
                      "research data use agreement)",
        license_documented=True,
        license_reference="Phase 95 usage_permission_reference: Kaggle "
                          "competition terms",
        access_type="public_repository_page",
        dataset_description="Real e-commerce card transactions with "
                            "identity tables; identifiers anonymized.",
        transaction_count=590540,
        fraud_count=None,
        fraud_rate=0.035,
        counts_reference="Kaggle competition description (train "
                         "transaction rows; ~3.5% fraud rate)",
        timestamp_semantics="TransactionDT = seconds elapsed from an "
                            "undisclosed anchor (relative only; no "
                            "absolute clock time; no timezone)",
        timestamp_supports_ordering=True,
        timestamp_supports_absolute_time=False,
        entity_identifiers=("TransactionID", "card1", "card2", "card3",
                            "card4", "card5", "card6", "addr1", "addr2",
                            "DeviceType", "DeviceInfo"),
        merchant_information="no merchant identifier column documented "
                             "(Phase 104: merchant_id unavailable)",
        amount_information="TransactionAmt present (currency not stated "
                           "in source description)",
        transaction_type_information="no channel column documented "
                                     "(Phase 104: channel unavailable)",
        geography_information="addr1/addr2 anonymized region codes only; "
                              "no state/zip/city columns documented",
        device_information="DeviceType/DeviceInfo present in identity "
                           "tables",
        mcc_information="no MCC column documented (Phase 104: mcc "
                        "unavailable)",
        historical_aggregate_information="no per-entity history columns "
                                         "documented",
        label_column="isFraud",
        label_semantics="1 = fraudulent transaction per competition "
                        "definition",
        label_provenance="adjudication methodology not publicly "
                         "documented (Phase 95: label method UNVERIFIED)",
        classification=DatasetClassification.PUBLIC_REAL_ANONYMIZED.value,
        known_transformations=(
            "identifiers anonymized by provider",
            "Vesta-engineered count/velocity features with masked values",
        ),
        known_leakage_risks=(
            "relative-only timestamp: absolute-time features cannot be "
            "derived and must never be synthesized",
        ),
        documentation_confidence="high",
        provenance_confidence="high",
        semantics_established=True,
        leakage_clear=True,
        schema_columns=("TransactionID", "TransactionDT", "TransactionAmt",
                        "isFraud", "card1", "card2", "card3", "card4",
                        "card5", "card6", "addr1", "addr2", "dist",
                        "DeviceType", "DeviceInfo", "P_emaildomain",
                        "R_emaildomain"),
        requirement_availability=_IEEE_AVAILABILITY,
        expected_local_path=f"{EXPECTED_LOCAL_ROOT}/IEEE_CIS/",
        acquisition_status="not_acquired",
    ),
    BenchmarkDataset(
        dataset_id="ULB_CREDIT_CARD_FRAUD",
        canonical_name="ULB Credit Card Fraud Detection (creditcard.csv)",
        source_reference="Kaggle dataset: mlg-ulb/creditcardfraud",
        provider_author="ULB Machine Learning Group research release",
        publication_reference="Dal Pozzolo et al. credit card fraud "
                              "release accompanying the Kaggle dataset",
        license_terms="CC BY-NC-SA 4.0 (as published on the source page)",
        license_documented=True,
        license_reference="Kaggle dataset page license field "
                          "(mlg-ulb/creditcardfraud)",
        access_type="public_repository_page",
        dataset_description="284,807 European cardholder transactions "
                            "over two days; original features replaced "
                            "by PCA components V1..V28.",
        transaction_count=284807,
        fraud_count=492,
        fraud_rate=0.001727,
        counts_reference="Kaggle dataset description: 284,807 rows, "
                         "492 frauds (0.172%)",
        timestamp_semantics="Time = seconds between the transaction and "
                            "the first transaction in the file "
                            "(relative only; no absolute anchor; no "
                            "timezone)",
        timestamp_supports_ordering=True,
        timestamp_supports_absolute_time=False,
        entity_identifiers=(),
        merchant_information="no merchant identifier (original columns "
                             "not disclosed)",
        amount_information="Amount present (currency not stated in the "
                           "source description)",
        transaction_type_information="no channel/transaction-type column "
                                     "documented",
        geography_information="no geography columns (PCA components only)",
        device_information="no device columns (PCA components only)",
        mcc_information="no MCC column (original semantics undisclosed)",
        historical_aggregate_information="no entity identifiers, so no "
                                         "history can be constructed",
        label_column="Class",
        label_semantics="1 = fraudulent transaction, 0 otherwise",
        label_provenance="labeling process beyond class semantics not "
                         "documented by the source",
        classification=DatasetClassification.PUBLIC_REAL_ANONYMIZED.value,
        known_transformations=(
            "PCA applied: V1..V28 principal components, original "
            "features undisclosed",
            "Time recorded as seconds offset from file start",
        ),
        known_leakage_risks=(
            "PCA component semantics undisclosed: components must never "
            "be mapped to canonical requirements",
        ),
        documentation_confidence="high",
        provenance_confidence="high",
        semantics_established=True,
        leakage_clear=True,
        schema_columns=_ulb_schema(),
        requirement_availability=_ULB_AVAILABILITY,
        expected_local_path=f"{EXPECTED_LOCAL_ROOT}/ULB_CREDIT_CARD_FRAUD/",
        acquisition_status="not_acquired",
    ),
    BenchmarkDataset(
        dataset_id="PAYSIM",
        canonical_name="PaySim mobile money payment simulation",
        source_reference="Mendeley Data: Fraud detection payment dataset "
                         "(PaySim)",
        provider_author="Ajayi et al. research release (simulator "
                        "calibrated on real mobile-money logs)",
        publication_reference="Ajayi et al. (2018), Data in Brief — "
                              "PaySim simulation dataset",
        license_terms="CC BY 4.0 (as published on the Mendeley Data "
                      "record)",
        license_documented=True,
        license_reference="Mendeley Data record license field (PaySim)",
        access_type="public_direct_download",
        dataset_description="Agent-based simulation of mobile money "
                            "transfers with simulator ground-truth fraud "
                            "labels.",
        transaction_count=1048576,
        fraud_count=None,
        fraud_rate=0.00605,
        counts_reference="Source dataset description: 1,048,576 rows "
                         "(sample), ~0.605% isFraud",
        timestamp_semantics="step = one simulated hour from simulation "
                            "start (simulation clock; not wall-clock; "
                            "no timezone)",
        timestamp_supports_ordering=True,
        timestamp_supports_absolute_time=False,
        entity_identifiers=("nameOrig", "nameDest"),
        merchant_information="no merchant identifier (nameDest is an "
                             "account, not a merchant)",
        amount_information="amount present (simulated units)",
        transaction_type_information="type: PAYMENT/TRANSFER/CASH_IN/"
                                     "CASH_OUT/DEBIT (transaction type, "
                                     "NOT channel semantics)",
        geography_information="no geography columns documented",
        device_information="no device columns documented",
        mcc_information="no MCC column documented",
        historical_aggregate_information="per-account sequences and "
                                         "prior-label rates derivable "
                                         "from ordered rows",
        label_column="isFraud",
        label_semantics="1 = fraudulent transaction per simulator "
                        "ground truth",
        label_provenance="simulator ground truth (retrospective within "
                         "the simulation), not human adjudication",
        classification=DatasetClassification.PUBLIC_SIMULATED.value,
        known_transformations=(
            "agent-based simulation calibrated on real logs",
            "one-hour simulation steps instead of calendar time",
        ),
        known_leakage_risks=(
            "isFlaggedFraud is rules-engine output (evaluation-derived) "
            "— excluded from features",
            "newbalanceOrig/newbalanceDest are post-event state and the "
            "simulator skips balance updates on fraudulent rows — "
            "label leakage, excluded from features",
        ),
        documentation_confidence="high",
        provenance_confidence="high",
        semantics_established=True,
        leakage_clear=True,
        schema_columns=("step", "type", "amount", "nameOrig",
                        "oldbalanceOrg", "newbalanceOrig", "nameDest",
                        "oldbalanceDest", "newbalanceDest", "isFraud",
                        "isFlaggedFraud"),
        requirement_availability=_PAYSIM_AVAILABILITY,
        expected_local_path=f"{EXPECTED_LOCAL_ROOT}/PAYSIM/",
        acquisition_status="not_acquired",
    ),
    BenchmarkDataset(
        dataset_id="BANKSIM",
        canonical_name="BankSim bank transaction simulation",
        source_reference="Kaggle repository: BankSim dataset",
        provider_author="Research simulation release (calibrated on "
                        "Spanish banking patterns)",
        publication_reference="BankSim simulation publication referenced "
                              "by the source repository page",
        license_terms="",   # NOT established from available documentation
        license_documented=False,
        license_reference=NOT_DOCUMENTED,
        access_type="public_repository_page",
        dataset_description="Agent-based simulation of bank transfers "
                            "with per-merchant fraud patterns.",
        transaction_count=None,
        fraud_count=None,
        fraud_rate=None,
        counts_reference="row/fraud counts not transcribed from the "
                         "source in this offline phase",
        timestamp_semantics="step = simulated time step (simulation "
                            "clock; not wall-clock; no timezone)",
        timestamp_supports_ordering=True,
        timestamp_supports_absolute_time=False,
        entity_identifiers=("merchant",),
        merchant_information="merchant identifiers present (fixed "
                             "merchant set)",
        amount_information="amount present (simulated currency)",
        transaction_type_information="category: merchant category labels "
                                     "(NOT MCC codes)",
        geography_information="no geography columns in the published "
                              "schema list",
        device_information="no device columns documented",
        mcc_information="category exists but its semantics are NOT "
                        "established as MCC — ambiguous, never mapped",
        historical_aggregate_information="merchant sequences derivable; "
                                         "no customer identifier "
                                         "documented",
        label_column="fraudRisk",
        label_semantics="1 = fraudulent transaction per simulator "
                        "ground truth",
        label_provenance="simulator ground truth, not human "
                         "adjudication",
        classification=DatasetClassification.PUBLIC_SIMULATED.value,
        known_transformations=(
            "agent-based simulation calibrated on reported Spanish "
            "banking patterns",
        ),
        known_leakage_risks=(
            "fraudRisk is the label column — never a feature",
        ),
        documentation_confidence="medium",
        provenance_confidence="high",
        semantics_established=True,
        leakage_clear=True,
        schema_columns=("step", "age", "gender", "category", "amount",
                        "merchant", "fraudRisk"),
        requirement_availability=_BANKSIM_AVAILABILITY,
        expected_local_path=f"{EXPECTED_LOCAL_ROOT}/BANKSIM/",
        acquisition_status="not_acquired",
    ),
    BenchmarkDataset(
        dataset_id="BAF_BANK_ACCOUNT_FRAUD",
        canonical_name="Bank Account Fraud (BAF) suite",
        source_reference="NeurIPS 2022 Datasets & Benchmarks: Bank "
                         "Account Fraud suite (research portal)",
        provider_author="NeurIPS 2022 publication consortium",
        publication_reference="NeurIPS 2022 Datasets & Benchmarks — "
                              "Bank Account Fraud (BAF) suite",
        license_terms="",   # NOT established from available documentation
        license_documented=False,
        license_reference=NOT_DOCUMENTED,
        access_type="research_portal",
        dataset_description="Bank account/application fraud benchmark "
                            "variants published for research; full "
                            "column dictionary not transcribed in this "
                            "offline phase.",
        transaction_count=None,
        fraud_count=None,
        fraud_rate=None,
        counts_reference="row/fraud counts not transcribed from the "
                         "source in this offline phase",
        timestamp_semantics=NOT_DOCUMENTED,
        timestamp_supports_ordering=False,
        timestamp_supports_absolute_time=False,
        entity_identifiers=(),
        merchant_information=NOT_DOCUMENTED,
        amount_information=NOT_DOCUMENTED,
        transaction_type_information=NOT_DOCUMENTED,
        geography_information=NOT_DOCUMENTED,
        device_information=NOT_DOCUMENTED,
        mcc_information=NOT_DOCUMENTED,
        historical_aggregate_information=NOT_DOCUMENTED,
        label_column="fraud_bool",
        label_semantics="1 = fraudulent per source definition "
                        "(column confirmed by publication)",
        label_provenance="labeling methodology not transcribed in this "
                         "offline phase",
        classification=DatasetClassification.PUBLIC_SYNTHETIC.value,
        known_transformations=(
            "synthetic variants generated for research publication",
        ),
        known_leakage_risks=(
            "column dictionary not established: no column may be mapped "
            "to a canonical requirement",
        ),
        documentation_confidence="low",
        provenance_confidence="high",
        semantics_established=False,
        leakage_clear=True,
        schema_columns=(),
        requirement_availability=_BAF_AVAILABILITY,
        expected_local_path=f"{EXPECTED_LOCAL_ROOT}/BAF_BANK_ACCOUNT_FRAUD/",
        acquisition_status="not_acquired",
    ),
)

DATASET_IDS: tuple[str, ...] = tuple(d.dataset_id
                                     for d in BENCHMARK_DATASETS)

_BY_ID: dict[str, BenchmarkDataset] = {
    d.dataset_id: d for d in BENCHMARK_DATASETS
}


def get_dataset(dataset_id: str) -> BenchmarkDataset:
    """Look up a registered dataset; unknown ids fail closed."""
    if dataset_id not in _BY_ID:
        raise KeyError(f"dataset not registered: {dataset_id}")
    return _BY_ID[dataset_id]


def detect_duplicate_datasets(
    entries: Iterable[BenchmarkDataset],
) -> tuple[str, ...]:
    """Duplicate dataset ids / canonical names / source references.

    The real registry must return (); fixtures prove detection works.
    """
    issues: list[str] = []
    seen_id: set[str] = set()
    seen_name: set[str] = set()
    seen_source: set[str] = set()
    for e in entries:
        if e.dataset_id in seen_id:
            issues.append(f"duplicate_dataset_id:{e.dataset_id}")
        if e.canonical_name.casefold() in seen_name:
            issues.append(f"duplicate_canonical_name:{e.canonical_name}")
        if e.source_reference in seen_source:
            issues.append(f"duplicate_source_reference:{e.source_reference}")
        seen_id.add(e.dataset_id)
        seen_name.add(e.canonical_name.casefold())
        seen_source.add(e.source_reference)
    return tuple(issues)


def manual_acquisition_instructions(dataset_id: str) -> str:
    """Documented MANUAL acquisition steps — never automated.

    Phase 106 provides instructions and expected local paths only; no
    code path in this repository performs an outbound transfer.
    """
    ds = get_dataset(dataset_id)
    return (
        f"MANUAL ACQUISITION (human action outside this codebase):\n"
        f" 1. Open the source page: {ds.source_reference}\n"
        f" 2. Read and record the license terms; if license_documented "
        f"is False, STOP and establish the license before any use.\n"
        f" 3. Download the files through the browser from the official "
        f"page only.\n"
        f" 4. Place them under: {ds.expected_local_path}\n"
        f" 5. Record SHA-256 hashes of every file and re-run Phase 106 "
        f"to bind them into the manifest.\n"
        f"Never script this acquisition; never acquire when "
        f"license_documented is False."
    )


# ══════════════════════════════════════════════════════════════════════
# FEATURE COMPATIBILITY — authoritative Phase 104 preflight, no redefinition
# ══════════════════════════════════════════════════════════════════════

def build_benchmark_compat_contract(dataset_id: str) -> ExternalDatasetContract:
    """Feature-compatibility ONLY contract for the Phase 104 preflight.

    assessed=False: this contract is NEVER submitted to the Phase 104
    qualification engine — Phase 104 remains the sole qualification
    authority and this module never qualifies anything.
    """
    ds = get_dataset(dataset_id)
    return ExternalDatasetContract(
        contract_version=CONTRACT_VERSION,
        context_id=BENCHMARK_CONTEXT_ID,
        assessed=False,
        provider_reference=ds.provider_author,
        dataset_id=ds.dataset_id,
        model_id=MODEL_ID,
        release_id=RELEASE_ID,
        evidence_valid_from="",
        evidence_cutoff="",
        provider_identity=ProviderIdentityClaim(),
        access_authority=AccessAuthorityClaim(),
        dataset_identity=DatasetIdentityClaim(
            canonical_dataset_id=ds.dataset_id),
        label_provenance=LabelProvenanceClaim(
            label_definition=ds.label_semantics),
        temporal_semantics=TemporalSemanticsClaim(
            event_timestamp_semantics=ds.timestamp_semantics),
        feature_compatibility=FeatureCompatibilityClaim(
            feature_contract_version=ML_FEATURE_VERSION,
            native_feature_version=NATIVE_FEATURE_VERSION,
            transformation=CANONICAL_TRANSFORMATION,
            native_feature_order=tuple(ALTMAN_NATIVE_FEATURES),
            direct_domain_feature_inference=False,
            requirement_availability=ds.requirement_availability,
            leakage_risk_features=(),
        ),
        entity_continuity=EntityContinuityClaim(),
        independence=IndependenceClaim(),
        contamination=ContaminationClaim(),
        usage_permission=UsagePermissionClaim(),
    )


def dataset_preflight(dataset_id: str) -> tuple[FeaturePreflightItem, ...]:
    """Classify all 48 native features via the authoritative preflight."""
    return preflight_feature_compatibility(
        build_benchmark_compat_contract(dataset_id))


def usable_features(preflight: Sequence[FeaturePreflightItem]) -> tuple[str, ...]:
    return tuple(i.feature for i in preflight
                 if i.qualification_impact == "none")


def assign_benchmark_group(
    dataset_id: str,
) -> tuple[str, tuple[str, ...]]:
    """Deterministic Group A/B/C/D assignment (governance veto first).

    D fires when license, provenance, semantics or leakage cannot be
    established (or classification is unknown).  Otherwise the feature
    preflight decides: no blocking features -> A, some usable -> B,
    none usable -> C.
    """
    ds = get_dataset(dataset_id)
    veto: list[str] = []
    if not ds.license_documented:
        veto.append("license_not_established")
    if ds.provenance_confidence == "low":
        veto.append("provenance_not_established")
    if ds.classification == DatasetClassification.UNKNOWN.value:
        veto.append("classification_unknown")
    if not ds.semantics_established:
        veto.append("schema_semantics_not_established")
    if not ds.leakage_clear:
        veto.append("leakage_risk_unresolved")
    if veto:
        return BenchmarkGroup.GROUP_D.value, tuple(veto)

    preflight = dataset_preflight(dataset_id)
    blocking = blocking_features(preflight)
    usable = usable_features(preflight)
    if not blocking:
        return BenchmarkGroup.GROUP_A.value, ()
    if usable:
        return BenchmarkGroup.GROUP_B.value, (
            f"{len(blocking)}/{len(preflight)} native features blocking",
            f"{len(usable)} native features usable for component evaluation",
        )
    return BenchmarkGroup.GROUP_C.value, (
        "no native feature usable; reference data only",
    )


def evaluation_scope(dataset_id: str) -> str:
    group, _ = assign_benchmark_group(dataset_id)
    return GROUP_READINESS[group]


def source_specific_compatibility_report() -> tuple[dict[str, Any], ...]:
    """Per-dataset compatibility report (spec: source-specific report)."""
    rows: list[dict[str, Any]] = []
    for ds in BENCHMARK_DATASETS:
        preflight = dataset_preflight(ds.dataset_id)
        group, reasons = assign_benchmark_group(ds.dataset_id)
        rows.append({
            "dataset_id": ds.dataset_id,
            "classification": ds.classification,
            "group": group,
            "group_reasons": list(reasons),
            "license_documented": ds.license_documented,
            "provenance_confidence": ds.provenance_confidence,
            "documentation_confidence": ds.documentation_confidence,
            "usable_features": list(usable_features(preflight)),
            "blocking_features": list(blocking_features(preflight)),
            "requirement_availability": [list(p)
                                         for p in ds.requirement_availability],
        })
    return tuple(rows)


def native_feature_eligibility_report() -> tuple[dict[str, Any], ...]:
    """Native 48-feature eligibility per dataset (spec report item 8)."""
    rows: list[dict[str, Any]] = []
    for ds in BENCHMARK_DATASETS:
        preflight = dataset_preflight(ds.dataset_id)
        blocking = blocking_features(preflight)
        group, _ = assign_benchmark_group(ds.dataset_id)
        eligible = (group == BenchmarkGroup.GROUP_A.value
                    and not blocking
                    and ds.acquisition_status == "acquired")
        reasons: list[str] = []
        if blocking:
            reasons.append(f"{len(blocking)} blocking native features")
        if group == BenchmarkGroup.GROUP_D.value:
            reasons.append("governance_not_established")
        if ds.acquisition_status != "acquired":
            reasons.append("file_not_acquired")
        rows.append({
            "dataset_id": ds.dataset_id,
            "group": group,
            "native_feature_count": len(preflight),
            "usable_count": len(usable_features(preflight)),
            "blocking_count": len(blocking),
            "status": ("ELIGIBLE_FOR_NATIVE_EVALUATION" if eligible
                       else "INCOMPATIBLE_FOR_NATIVE_EVALUATION"),
            # No file exists in Phase 106, so eligible rows are exactly 0.
            "eligible_rows": 0,
            "reasons": reasons,
        })
    return tuple(rows)


def native_eligible_row_count(dataset_id: str) -> int:
    """Rows eligible for native evaluation — always 0 in Phase 106.

    Eligibility requires Group A feature compatibility AND an acquired,
    hash-verified local file.  No dataset is Group A and no file is ever
    acquired in this phase, so the count is precisely zero for every
    registered dataset (never estimated, never imputed).
    """
    ds = get_dataset(dataset_id)
    group, _ = assign_benchmark_group(ds.dataset_id)
    if group != BenchmarkGroup.GROUP_A.value:
        return 0
    if ds.acquisition_status != "acquired":
        return 0
    return 0  # row counting happens only after manual acquisition


# ══════════════════════════════════════════════════════════════════════
# HARMONIZATION LAYER (spec §MULTI-DATASET HARMONIZATION)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HarmonizedRow:
    """One harmonized row — source identity ALWAYS preserved."""
    source_dataset_id: str
    source_record_id: str          # "" when the source has no record id
    original_label: str            # verbatim source value
    normalized_label: int | None   # 0/1, None when unmappable
    timestamp: str                 # verbatim source value, "" if absent
    original_schema_metadata: tuple[tuple[str, str], ...]
    harmonization_status: str      # HarmonizationStatus value

    def __post_init__(self) -> None:
        valid = {s.value for s in HarmonizationStatus}
        if self.harmonization_status not in valid:
            raise ValueError(
                f"invalid harmonization status: {self.harmonization_status}")
        if (self.harmonization_status == HarmonizationStatus.HARMONIZED.value
                and self.normalized_label not in (0, 1)):
            raise ValueError("harmonized rows require normalized label 0/1")
        if (self.harmonization_status != HarmonizationStatus.HARMONIZED.value
                and self.normalized_label is not None):
            raise ValueError("excluded rows must not carry a label")

    @property
    def schema_metadata_dict(self) -> dict[str, str]:
        return dict(self.original_schema_metadata)


def normalize_label(dataset_id: str, raw: Any) -> int | None:
    """Deterministic label normalization; unknown values NEVER guessed."""
    get_dataset(dataset_id)  # unknown dataset fails closed
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int) and raw in (0, 1):
        return raw
    if isinstance(raw, str) and raw.strip() in ("0", "1"):
        return int(raw.strip())
    return None


def harmonize_rows(
    dataset_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[HarmonizedRow, ...]:
    """Normalize rows of ONE source into HarmonizedRows.

    Source identity, original label, original schema metadata and the raw
    timestamp are preserved verbatim.  A row declaring a foreign
    source_dataset_id is excluded, never silently re-attributed.
    """
    ds = get_dataset(dataset_id)
    label_col = LABEL_COLUMNS[dataset_id]
    ts_col = TIMESTAMP_COLUMNS[dataset_id]
    out: list[HarmonizedRow] = []
    for row in rows:
        foreign = row.get("source_dataset_id")
        if foreign is not None and str(foreign) != dataset_id:
            status = HarmonizationStatus.EXCLUDED_FOREIGN_SOURCE.value
            norm = None
        else:
            norm = normalize_label(dataset_id, row.get(label_col))
            status = (HarmonizationStatus.HARMONIZED.value
                      if norm is not None
                      else HarmonizationStatus.EXCLUDED_UNMAPPED_LABEL.value)
        raw_label = row.get(label_col)
        ts_val = row.get(ts_col) if ts_col else None
        metadata = tuple(
            sorted((str(k), str(v)) for k, v in row.items()
                   if k != "record_id"))
        out.append(HarmonizedRow(
            source_dataset_id=dataset_id,
            source_record_id=str(row.get("record_id", "")),
            original_label="" if raw_label is None else str(raw_label),
            normalized_label=norm,
            timestamp="" if ts_val is None else str(ts_val),
            original_schema_metadata=metadata,
            harmonization_status=status,
        ))
    return tuple(out)


def find_single_source_claims(text: str) -> tuple[str, ...]:
    """Detect forbidden single-institution representations."""
    low = text.casefold()
    return tuple(p for p in FORBIDDEN_SINGLE_SOURCE_CLAIMS if p in low)


def describe_benchmark(entries: Sequence[Mapping[str, Any]]) -> str:
    """The ONLY approved descriptor for combined sources."""
    _ = entries
    return BENCHMARK_DESCRIPTOR


@dataclass(frozen=True)
class MultiSourceBenchmark:
    """Combined rows — always a multi-source public benchmark."""
    benchmark_id: str
    descriptor: str
    rows: tuple[HarmonizedRow, ...]
    source_dataset_ids: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        if self.descriptor != BENCHMARK_DESCRIPTOR:
            raise ValueError(
                f"descriptor must be {BENCHMARK_DESCRIPTOR!r}; single-"
                "institution representation is forbidden")
        if find_single_source_claims(self.descriptor):
            raise ValueError("descriptor contains forbidden claim")
        sources = tuple(sorted({r.source_dataset_id for r in self.rows}))
        for s in sources:
            get_dataset(s)  # unknown/unregistered source fails closed
        object.__setattr__(self, "source_dataset_ids", sources)

    def harmonized_rows(self) -> tuple[HarmonizedRow, ...]:
        return tuple(r for r in self.rows
                     if r.harmonization_status
                     == HarmonizationStatus.HARMONIZED.value)

    def excluded_rows(self) -> tuple[HarmonizedRow, ...]:
        return tuple(r for r in self.rows
                     if r.harmonization_status
                     != HarmonizationStatus.HARMONIZED.value)


def build_multisource_benchmark(
    row_groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> MultiSourceBenchmark:
    """Harmonize several sources into one benchmark, identity preserved."""
    all_rows: list[HarmonizedRow] = []
    for dataset_id, rows in row_groups:
        all_rows.extend(harmonize_rows(dataset_id, rows))
    return MultiSourceBenchmark(
        benchmark_id=BENCHMARK_ID,
        descriptor=BENCHMARK_DESCRIPTOR,
        rows=tuple(all_rows),
    )


# ══════════════════════════════════════════════════════════════════════
# LEAKAGE & CONTAMINATION CONTROLS (spec §LEAKAGE CONTROLS)
# ══════════════════════════════════════════════════════════════════════

def scan_leakage_fields(field_names: Iterable[str]) -> tuple[str, ...]:
    """Flag label-derived / post-outcome / evaluation-derived columns."""
    hits: list[str] = []
    for name in field_names:
        low = name.casefold()
        for pattern in LEAKAGE_FIELD_PATTERNS:
            if re.search(pattern, low):
                hits.append(name)
                break
    return tuple(sorted(set(hits)))


def dataset_leakage_findings() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Leakage columns detected in each documented schema."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for ds in BENCHMARK_DATASETS:
        flagged = scan_leakage_fields(ds.schema_columns)
        out.append((ds.dataset_id, flagged))
    return tuple(out)


def row_fingerprint(row: HarmonizedRow) -> str:
    """Semantic fingerprint over (label, timestamp, amount) only —
    column naming differs across sources, semantics are compared."""
    meta = row.schema_metadata_dict
    amt_col = AMOUNT_COLUMNS.get(row.source_dataset_id)
    amount = meta.get(amt_col) if amt_col else None
    return _stable_hash({
        "y": row.normalized_label,
        "ts": row.timestamp,
        "amt": amount,
    })


def detect_cross_source_duplicates(
    rows: Sequence[HarmonizedRow],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Same (label, time, amount) content under different sources."""
    by_fp: dict[str, set[str]] = {}
    for r in rows:
        if r.harmonization_status != HarmonizationStatus.HARMONIZED.value:
            continue
        by_fp.setdefault(row_fingerprint(r), set()).add(r.source_dataset_id)
    return tuple(
        (fp, tuple(sorted(sources)))
        for fp, sources in sorted(by_fp.items())
        if len(sources) > 1
    )


def detect_source_contamination(
    rows: Sequence[HarmonizedRow],
    expected_sources: frozenset[str],
) -> tuple[str, ...]:
    """Rows attributed to a source outside the expected set."""
    return tuple(sorted({
        r.source_dataset_id for r in rows
        if r.source_dataset_id not in expected_sources
    }))


def detect_train_evaluation_overlap(
    bench_rows: Sequence[HarmonizedRow],
    training_fingerprints: Iterable[str],
) -> tuple[str, ...]:
    """Detectable train/evaluation overlap (fingerprint intersection)."""
    train = set(training_fingerprints)
    if not train:
        return ()
    return tuple(sorted({
        row_fingerprint(r) for r in bench_rows
        if r.harmonization_status == HarmonizationStatus.HARMONIZED.value
        and row_fingerprint(r) in train
    }))


def detect_synthetic_augmentation(
    rows: Sequence[HarmonizedRow],
) -> tuple[str, ...]:
    """Rows carrying synthetic-augmentation markers (leakage risk)."""
    hits: list[str] = []
    for i, r in enumerate(rows):
        blob = " ".join(f"{k}={v}" for k, v in r.original_schema_metadata)
        low = blob.casefold()
        if any(m in low for m in SYNTHETIC_AUGMENTATION_MARKERS):
            hits.append(f"{r.source_dataset_id}:{i}")
    return tuple(hits)


def source_contamination_report() -> dict[str, Any]:
    """Contamination status for the (empty) real benchmark."""
    return {
        "acquired_rows": 0,
        "cross_source_duplicates": [],
        "train_evaluation_overlap": [],
        "synthetic_augmentation_rows": [],
        "status": "no rows acquired — detectors armed and fixture-verified",
    }


# ══════════════════════════════════════════════════════════════════════
# EVALUATION INFRASTRUCTURE (spec §EVALUATION — assessment, never tuning)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BenchmarkEvaluationConfig:
    """Locked evaluation configuration — threshold is read-only."""
    config_version: str
    threshold: float                 # = PRODUCTION_THRESHOLD (locked)
    metrics: tuple[str, ...]
    stratification: tuple[str, ...]
    temporal_scope: str
    purpose: str
    native_scope_requires: str

    def __post_init__(self) -> None:
        if self.threshold != PRODUCTION_THRESHOLD:
            raise ValueError("threshold must equal the locked production "
                             "threshold")


def build_evaluation_config() -> BenchmarkEvaluationConfig:
    return BenchmarkEvaluationConfig(
        config_version=EVAL_CONFIG_VERSION,
        threshold=PRODUCTION_THRESHOLD,
        metrics=("precision", "recall", "f1", "pr_auc", "roc_auc",
                 "mcc", "brier_score", "confusion"),
        stratification=("source_dataset_id",),
        temporal_scope="within_source_ordering_only",
        purpose="external_assessment_only_no_tuning_no_threshold_selection",
        native_scope_requires=EvaluationReadiness.READY_NATIVE.value,
    )


def compute_mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    """Matthews correlation coefficient from a confusion matrix."""
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0:
        return 0.0
    return round((tp * tn - fp * fn) / denom, 6)


def class_imbalance(y_true: Sequence[int]) -> dict[str, Any]:
    n = len(y_true)
    fraud = sum(1 for y in y_true if y == 1)
    return {
        "rows": n,
        "fraud_count": fraud,
        "fraud_rate": round(fraud / n, 6) if n else 0.0,
    }


def compute_benchmark_metrics(
    y_true: Sequence[int],
    y_scores: Sequence[float],
) -> dict[str, Any]:
    """Metrics at the LOCKED production threshold.

    Deliberately accepts NO threshold argument: benchmark results can
    never select or shift a threshold (spec §EVALUATION).
    """
    base = compute_metrics(list(y_true), list(y_scores))
    if "error" in base:
        return base
    base["mcc"] = compute_mcc(int(base["tp"]), int(base["fp"]),
                              int(base["fn"]), int(base["tn"]))
    base["class_imbalance"] = class_imbalance(y_true)
    base["threshold_source"] = "PRODUCTION_THRESHOLD (locked, read-only)"
    return base


def evaluation_coverage(
    rows: Sequence[HarmonizedRow],
) -> dict[str, Any]:
    """Coverage / eligible-row counts per source."""
    per_source: dict[str, int] = {}
    harmonized = 0
    for r in rows:
        per_source[r.source_dataset_id] = (
            per_source.get(r.source_dataset_id, 0) + 1)
        if r.harmonization_status == HarmonizationStatus.HARMONIZED.value:
            harmonized += 1
    return {
        "total_rows": len(rows),
        "harmonized_rows": harmonized,
        "excluded_rows": len(rows) - harmonized,
        "per_source": dict(sorted(per_source.items())),
        "native_eligible_rows": 0,
    }


def stratified_evaluation(
    rows: Sequence[HarmonizedRow],
    y_scores: Sequence[float],
) -> dict[str, Any]:
    """Source-stratified metrics; GROUP_D sources are refused."""
    if len(rows) != len(y_scores):
        raise ValueError("rows and scores must align")
    per_source_y: dict[str, list[int]] = {}
    per_source_s: dict[str, list[float]] = {}
    refusals: dict[str, str] = {}
    for r, s in zip(rows, y_scores):
        if r.harmonization_status != HarmonizationStatus.HARMONIZED.value:
            continue
        scope = evaluation_scope(r.source_dataset_id)
        if scope == EvaluationReadiness.REJECTED_NOT_ESTABLISHED.value:
            refusals[r.source_dataset_id] = (
                "source_rejected_governance_not_established")
            continue
        per_source_y.setdefault(r.source_dataset_id, []).append(
            int(r.normalized_label))  # type: ignore[arg-type]
        per_source_s.setdefault(r.source_dataset_id, []).append(float(s))
    metrics = {
        ds: compute_benchmark_metrics(per_source_y[ds],
                                      per_source_s[ds])
        for ds in sorted(per_source_y)
    }
    return {"per_source": metrics, "refusals": dict(sorted(refusals.items()))}


def cross_dataset_evaluation(
    rows: Sequence[HarmonizedRow],
    y_scores: Sequence[float],
) -> dict[str, Any]:
    """Pooled metrics over COMMON fields only; GROUP_D refused.

    Pooled evaluation uses only the harmonized label — feature vectors
    are never concatenated across sources (semantics not established).
    """
    if len(rows) != len(y_scores):
        raise ValueError("rows and scores must align")
    ys: list[int] = []
    ss: list[float] = []
    refusals: dict[str, str] = {}
    for r, s in zip(rows, y_scores):
        if r.harmonization_status != HarmonizationStatus.HARMONIZED.value:
            continue
        if evaluation_scope(r.source_dataset_id) == (
                EvaluationReadiness.REJECTED_NOT_ESTABLISHED.value):
            refusals[r.source_dataset_id] = (
                "source_rejected_governance_not_established")
            continue
        ys.append(int(r.normalized_label))  # type: ignore[arg-type]
        ss.append(float(s))
    return {
        "field_scope": list(COMMON_BENCHMARK_FIELDS),
        "pooled_metrics": compute_benchmark_metrics(ys, ss),
        "refusals": dict(sorted(refusals.items())),
        "note": "label-scope pooling only; no cross-source feature merge",
    }


def temporal_evaluation(
    rows: Sequence[HarmonizedRow],
    y_scores: Sequence[float],
    holdout_fraction: float = 0.2,
) -> dict[str, Any]:
    """Within-source temporal split; cross-source alignment refused."""
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in (0, 1)")
    if len(rows) != len(y_scores):
        raise ValueError("rows and scores must align")
    by_source: dict[str, list[tuple[HarmonizedRow, float]]] = {}
    for r, s in zip(rows, y_scores):
        if r.harmonization_status != HarmonizationStatus.HARMONIZED.value:
            continue
        by_source.setdefault(r.source_dataset_id, []).append((r, float(s)))

    result: dict[str, Any] = {
        "scope": "within_source_ordering_only",
        "refusals": {},
        "segments": {},
    }
    for source, pairs in sorted(by_source.items()):
        ds = get_dataset(source)
        if (not ds.timestamp_supports_ordering
                or not ds.timestamp_columns_present):
            result["refusals"][source] = "timestamps_not_orderable"
            continue
        def _key(pair: tuple[HarmonizedRow, float]) -> tuple[int, Any]:
            raw = pair[0].timestamp
            try:
                return (0, float(raw))
            except (TypeError, ValueError):
                return (1, raw)
        ordered = sorted(pairs, key=_key)
        split = int(len(ordered) * (1.0 - holdout_fraction))
        split = min(max(split, 1), len(ordered) - 1) if len(ordered) > 1 else 0
        early, late = ordered[:split], ordered[split:]
        result["segments"][source] = {
            "early": compute_benchmark_metrics(
                [int(r.normalized_label) for r, _ in early],  # type: ignore[misc]
                [s for _, s in early]) if early else {"error": "no_data"},
            "late": compute_benchmark_metrics(
                [int(r.normalized_label) for r, _ in late],  # type: ignore[misc]
                [s for _, s in late]) if late else {"error": "no_data"},
        }
    return result


# ══════════════════════════════════════════════════════════════════════
# BENCHMARK MANIFEST (spec §PROVENANCE — NEVER gate evidence)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BenchmarkManifest:
    """Deterministic, immutable benchmark manifest.

    manifest_kind pins the type: this is NOT ProviderEvidence, NOT a
    QualificationResult, NOT RWVPromotionEvidence and NOT a
    PromotionToken.  No RWV or promotion gate accepts it.
    """
    manifest_kind: str
    manifest_version: str
    benchmark_id: str
    created_at: str
    descriptor: str
    dataset_ids: tuple[str, ...]
    source_references: tuple[tuple[str, str], ...]
    license_metadata: tuple[tuple[str, str, bool], ...]
    source_hashes: tuple[tuple[str, str], ...]   # no files => empty
    row_counts: tuple[tuple[str, int | None], ...]
    schema_hashes: tuple[tuple[str, str], ...]
    harmonization_spec_hash: str
    feature_compatibility_results: tuple[tuple[str, str, int, int], ...]
    evaluation_config: tuple[tuple[str, str], ...]
    model_id: str
    release_id: str
    feature_contract_version: str
    native_feature_version: str
    domain_feature_count: int
    native_feature_count: int
    canonical_transformation: str
    preprocessing_hash: str
    rule_hash: str
    production_threshold: float
    manifest_hash: str

    def __post_init__(self) -> None:
        # Identity/threshold immutability: production model identity,
        # release identity, the canonical mapping and the threshold are
        # frozen, so any substituted identity fails CLOSED at
        # construction.  Hash self-consistency stays with verify_hash(),
        # so mutation of any other field is DETECTED, never rebuilt.
        if self.manifest_kind != "benchmark_manifest":
            raise ValueError(
                "manifest_kind must be 'benchmark_manifest'; gate-evidence "
                "kinds are forbidden here")
        if self.descriptor != BENCHMARK_DESCRIPTOR:
            raise ValueError(
                f"descriptor must be {BENCHMARK_DESCRIPTOR!r}; "
                "single-institution representation is forbidden")
        if self.model_id != MODEL_ID:
            raise ValueError("manifest model identity is immutable")
        if self.release_id != RELEASE_ID:
            raise ValueError("manifest release identity is immutable")
        if self.feature_contract_version != ML_FEATURE_VERSION:
            raise ValueError("manifest feature contract version is immutable")
        if self.native_feature_version != NATIVE_FEATURE_VERSION:
            raise ValueError("manifest native feature version is immutable")
        if self.domain_feature_count != DOMAIN_FEATURE_COUNT:
            raise ValueError("manifest domain feature count must be the canonical 21")
        if self.native_feature_count != NATIVE_FEATURE_COUNT:
            raise ValueError("manifest native feature count must be the canonical 48")
        if self.production_threshold != PRODUCTION_THRESHOLD:
            raise ValueError("manifest threshold is immutable")
        if self.canonical_transformation != CANONICAL_TRANSFORMATION:
            raise ValueError("manifest canonical mapping is immutable")

    def payload(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("manifest_hash", None)
        return d

    def manifest_hash_for(self) -> str:
        return _stable_hash(self.payload())

    def verify_hash(self) -> bool:
        return self.manifest_hash == self.manifest_hash_for()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_benchmark_manifest() -> BenchmarkManifest:
    """Build the deterministic benchmark manifest (offline, no clock)."""
    harmonization_spec_hash = _stable_hash({
        "common_fields": list(COMMON_BENCHMARK_FIELDS),
        "rationale": COMMON_FIELD_RATIONALE,
        "label_columns": dict(sorted(LABEL_COLUMNS.items())),
        "timestamp_columns": {k: v for k, v in
                              sorted(TIMESTAMP_COLUMNS.items())},
        "statuses": sorted(s.value for s in HarmonizationStatus),
        "descriptor": BENCHMARK_DESCRIPTOR,
    })
    compat_rows = []
    for ds in BENCHMARK_DATASETS:
        group, _ = assign_benchmark_group(ds.dataset_id)
        preflight = dataset_preflight(ds.dataset_id)
        compat_rows.append((
            ds.dataset_id, group,
            len(usable_features(preflight)),
            len(blocking_features(preflight)),
        ))
    compat = tuple(compat_rows)
    cfg = build_evaluation_config()
    evaluation_config = (
        ("config_version", cfg.config_version),
        ("threshold", repr(cfg.threshold)),
        ("metrics", ",".join(cfg.metrics)),
        ("stratification", ",".join(cfg.stratification)),
        ("temporal_scope", cfg.temporal_scope),
        ("purpose", cfg.purpose),
        ("native_scope_requires", cfg.native_scope_requires),
    )
    partial = {
        "manifest_kind": "benchmark_manifest",
        "manifest_version": MANIFEST_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "created_at": MANIFEST_CREATED_AT,
        "descriptor": BENCHMARK_DESCRIPTOR,
        "dataset_ids": DATASET_IDS,
        "source_references": tuple(
            (d.dataset_id, d.source_reference)
            for d in BENCHMARK_DATASETS),
        "license_metadata": tuple(
            (d.dataset_id, d.license_terms, d.license_documented)
            for d in BENCHMARK_DATASETS),
        "source_hashes": (),   # no files legitimately available yet
        "row_counts": tuple(
            (d.dataset_id, d.transaction_count)
            for d in BENCHMARK_DATASETS),
        "schema_hashes": tuple(
            (d.dataset_id, d.schema_hash)
            for d in BENCHMARK_DATASETS),
        "harmonization_spec_hash": harmonization_spec_hash,
        "feature_compatibility_results": compat,
        "evaluation_config": evaluation_config,
        "model_id": MODEL_ID,
        "release_id": RELEASE_ID,
        "feature_contract_version": ML_FEATURE_VERSION,
        "native_feature_version": NATIVE_FEATURE_VERSION,
        "domain_feature_count": DOMAIN_FEATURE_COUNT,
        "native_feature_count": NATIVE_FEATURE_COUNT,
        "canonical_transformation": CANONICAL_TRANSFORMATION,
        "preprocessing_hash": PREPROCESSING_HASH,
        "rule_hash": RULE_HASH,
        "production_threshold": PRODUCTION_THRESHOLD,
    }
    manifest_hash = _stable_hash(partial)
    return BenchmarkManifest(**partial, manifest_hash=manifest_hash)


def registry_integrity_hash() -> str:
    """Stable hash over the whole registry (reproducibility anchor)."""
    return _stable_hash([d.to_dict() for d in BENCHMARK_DATASETS])


def group_summary() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {g.value: [] for g in BenchmarkGroup}
    for ds in BENCHMARK_DATASETS:
        group, _ = assign_benchmark_group(ds.dataset_id)
        out[group].append(ds.dataset_id)
    return {k: tuple(v) for k, v in out.items()}


def documented_row_totals() -> dict[str, Any]:
    documented = [d for d in BENCHMARK_DATASETS
                  if d.transaction_count is not None]
    unknown = [d.dataset_id for d in BENCHMARK_DATASETS
               if d.transaction_count is None]
    total = sum(d.transaction_count or 0 for d in documented)
    return {
        "documented_rows_total": total,
        "documented_count_datasets": tuple(d.dataset_id
                                           for d in documented),
        "count_not_transcribed_datasets": tuple(unknown),
        "native_eligible_rows_total": 0,
        "rejected_rows_total": total,   # all rejected for native scope
    }


__all__ = [
    "REGISTRY_VERSION", "MANIFEST_VERSION", "EVAL_CONFIG_VERSION",
    "BENCHMARK_ID", "BENCHMARK_CONTEXT_ID", "MANIFEST_CREATED_AT",
    "BENCHMARK_DESCRIPTOR", "EXPECTED_LOCAL_ROOT",
    "COMMON_BENCHMARK_FIELDS", "COMMON_FIELD_RATIONALE",
    "FORBIDDEN_SINGLE_SOURCE_CLAIMS", "LEAKAGE_FIELD_PATTERNS",
    "SYNTHETIC_AUGMENTATION_MARKERS", "FORBIDDEN_BYPASS_PARAMETERS",
    "BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE", "NOT_DOCUMENTED",
    "DOMAIN_FEATURE_COUNT", "NATIVE_FEATURE_COUNT",
    "LABEL_COLUMNS", "TIMESTAMP_COLUMNS", "AMOUNT_COLUMNS",
    "DATASET_IDS", "BENCHMARK_DATASETS",
    "DatasetClassification", "NON_REAL_WORLD_CLASSIFICATIONS",
    "BenchmarkGroup", "HarmonizationStatus", "EvaluationReadiness",
    "GROUP_READINESS",
    "BenchmarkDataset", "HarmonizedRow", "MultiSourceBenchmark",
    "BenchmarkEvaluationConfig", "BenchmarkManifest",
    "get_dataset", "detect_duplicate_datasets",
    "manual_acquisition_instructions",
    "build_benchmark_compat_contract", "dataset_preflight",
    "usable_features", "assign_benchmark_group", "evaluation_scope",
    "source_specific_compatibility_report",
    "native_feature_eligibility_report", "native_eligible_row_count",
    "normalize_label", "harmonize_rows", "find_single_source_claims",
    "describe_benchmark", "build_multisource_benchmark",
    "scan_leakage_fields", "dataset_leakage_findings", "row_fingerprint",
    "detect_cross_source_duplicates", "detect_source_contamination",
    "detect_train_evaluation_overlap", "detect_synthetic_augmentation",
    "source_contamination_report",
    "build_evaluation_config", "compute_mcc", "class_imbalance",
    "compute_benchmark_metrics", "evaluation_coverage",
    "stratified_evaluation", "cross_dataset_evaluation",
    "temporal_evaluation",
    "build_benchmark_manifest", "registry_integrity_hash",
    "group_summary", "documented_row_totals",
    # re-exported Phase 103 global state (read-only)
    "SYSTEM_READINESS", "REAL_WORLD_VALIDATION", "PROMOTION_STATE",
    # re-exported locked identity constants (read-only)
    "MODEL_ID", "RELEASE_ID", "ML_FEATURE_VERSION",
    "NATIVE_FEATURE_VERSION", "PREPROCESSING_HASH", "RULE_HASH",
    "PRODUCTION_THRESHOLD",
]
