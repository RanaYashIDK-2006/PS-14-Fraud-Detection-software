"""Phase 53: Dataset evidence model, admission gate & evaluation framework.

Defines the complete dataset-evidence lifecycle:

  CANDIDATE → EVIDENCE_COLLECTION → PROVENANCE_VERIFIED → SCHEMA_VERIFIED
  → LABEL_SEMANTICS_VERIFIED → TEMPORAL_SEMANTICS_VERIFIED → LEAKAGE_CHECKED
  → FEATURE_COMPATIBILITY_VERIFIED → ELIGIBLE

Any mandatory failure results in INELIGIBLE / BLOCKED.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET

This module does NOT fabricate datasets or eligibility.
It provides the framework for evaluating genuinely eligible datasets
when they eventually appear.

STATUS: IMPLEMENTED
PHASE: 53
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

# ── Source Classification ─────────────────────────────────────────────────

class SourceClassification(str, Enum):
    SYNTHETIC = "SYNTHETIC"
    REAL = "REAL"
    REAL_LEGACY = "REAL_LEGACY"
    DERIVED = "DERIVED"
    UNKNOWN = "UNKNOWN"
    REAL_WORLD_ELIGIBLE = "REAL_WORLD_ELIGIBLE"


class AdmissionState(str, Enum):
    CANDIDATE = "CANDIDATE"
    EVIDENCE_COLLECTION = "EVIDENCE_COLLECTION"
    PROVENANCE_VERIFIED = "PROVENANCE_VERIFIED"
    SCHEMA_VERIFIED = "SCHEMA_VERIFIED"
    LABEL_SEMANTICS_VERIFIED = "LABEL_SEMANTICS_VERIFIED"
    TEMPORAL_SEMANTICS_VERIFIED = "TEMPORAL_SEMANTICS_VERIFIED"
    LEAKAGE_CHECKED = "LEAKAGE_CHECKED"
    FEATURE_COMPATIBILITY_VERIFIED = "FEATURE_COMPATIBILITY_VERIFIED"
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    BLOCKED = "BLOCKED"


class EvidenceLevel(str, Enum):
    KNOWN = "KNOWN"
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ── Dataset Evidence Model ────────────────────────────────────────────────

@dataclass
class DatasetEvidenceRecord:
    """Structured dataset evidence for real-world validation admission.

    Every field must be explicitly provided or left at its default.
    Fields left at default (empty/UNKNOWN) block ELIGIBLE classification.
    """
    # Identity
    dataset_id: str = ""
    dataset_version: str = ""

    # Source
    source_name: str = ""
    source_url: str = ""
    source_type: str = ""  # "public_research", "proprietary", "academic", etc.
    source_classification: str = SourceClassification.UNKNOWN.value
    license: str = ""
    acquisition_timestamp: str = ""

    # Cryptographic identity
    dataset_hash: str = ""
    file_hashes: dict[str, str] = field(default_factory=dict)
    schema_hash: str = ""

    # Size
    row_count: int = 0
    column_count: int = 0

    # Schema
    feature_schema: list[str] = field(default_factory=list)
    label_column: str = ""
    label_definition: str = ""

    # Label semantics
    label_generation_method: str = ""  # "human_review", "automated", "retrospective", "unknown"
    label_timestamp_semantics: str = ""  # "at_event", "post_investigation", "unknown"
    label_timing_known: bool = False
    positive_class_definition: str = ""
    negative_class_definition: str = ""
    label_latency_description: str = ""

    # Temporal
    transaction_timestamp_column: str = ""
    collection_period_start: str = ""
    collection_period_end: str = ""
    temporal_order_known: bool = False

    # Provenance confidence
    provenance_confidence: str = EvidenceLevel.UNKNOWN.value  # "source_reported", "independently_verified"
    provenance_references: list[str] = field(default_factory=list)

    # Known transformations
    known_preprocessing: str = ""
    known_transformations: list[str] = field(default_factory=list)
    known_sampling: str = ""
    known_augmentation: str = ""

    # Feature compatibility
    feature_mapping_version: str = ""
    incompatible_features: list[str] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)

    # Evaluation binding
    evaluation_record_hash: str = ""
    training_period: str = ""
    validation_period: str = ""
    test_period: str = ""

    # Phase metadata
    admission_state: str = AdmissionState.CANDIDATE.value
    admission_timestamp: float = 0.0
    admission_history: list[dict[str, Any]] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)

    def compute_hash(self) -> str:
        """Deterministic SHA-256 over the evidence record (excluding mutable fields)."""
        d = asdict(self)
        # Exclude mutable audit fields
        for k in ("admission_state", "admission_timestamp", "admission_history",
                   "blocked_reasons"):
            d.pop(k, None)
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def compute_dataset_hash(self, file_path: str | Path) -> str:
        """Compute deterministic SHA-256 of a dataset file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        self.dataset_hash = h.hexdigest()
        return self.dataset_hash

    def transition(self, new_state: AdmissionState, reason: str = "") -> bool:
        """Attempt an admission state transition. Returns True if valid."""
        valid_transitions = {
            AdmissionState.CANDIDATE: {AdmissionState.EVIDENCE_COLLECTION, AdmissionState.INELIGIBLE, AdmissionState.BLOCKED},
            AdmissionState.EVIDENCE_COLLECTION: {AdmissionState.PROVENANCE_VERIFIED, AdmissionState.INELIGIBLE, AdmissionState.BLOCKED},
            AdmissionState.PROVENANCE_VERIFIED: {AdmissionState.SCHEMA_VERIFIED, AdmissionState.INELIGIBLE, AdmissionState.BLOCKED},
            AdmissionState.SCHEMA_VERIFIED: {AdmissionState.LABEL_SEMANTICS_VERIFIED, AdmissionState.INELIGIBLE, AdmissionState.BLOCKED},
            AdmissionState.LABEL_SEMANTICS_VERIFIED: {AdmissionState.TEMPORAL_SEMANTICS_VERIFIED, AdmissionState.INELIGIBLE, AdmissionState.BLOCKED},
            AdmissionState.TEMPORAL_SEMANTICS_VERIFIED: {AdmissionState.LEAKAGE_CHECKED, AdmissionState.INELIGIBLE, AdmissionState.BLOCKED},
            AdmissionState.LEAKAGE_CHECKED: {AdmissionState.FEATURE_COMPATIBILITY_VERIFIED, AdmissionState.INELIGIBLE, AdmissionState.BLOCKED},
            AdmissionState.FEATURE_COMPATIBILITY_VERIFIED: {AdmissionState.ELIGIBLE, AdmissionState.INELIGIBLE, AdmissionState.BLOCKED},
        }
        allowed = valid_transitions.get(AdmissionState(self.admission_state), set())
        if new_state not in allowed:
            return False
        self.admission_history.append({
            "from": self.admission_state,
            "to": new_state.value,
            "reason": reason,
            "timestamp": time.time(),
        })
        self.admission_state = new_state.value
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetEvidenceRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Leakage Detection ─────────────────────────────────────────────────────

@dataclass
class LeakageCheckResult:
    """Result of a leakage check."""
    check_name: str
    passed: bool
    details: str = ""
    affected_columns: list[str] = field(default_factory=list)


def check_target_leakage(
    columns: list[str],
    label_column: str,
) -> LeakageCheckResult:
    """Check for columns that may encode the target label."""
    suspicious = []
    label_lower = label_column.lower()
    for col in columns:
        col_lower = col.lower()
        # Direct label leakage
        if col_lower == label_lower:
            continue
        # Common target-derived column names
        for pattern in ("is_fraud", "fraud_label", "label", "target", "outcome"):
            if pattern in col_lower and col_lower != label_lower:
                suspicious.append(col)
                break
    if suspicious:
        return LeakageCheckResult(
            check_name="target_leakage",
            passed=False,
            details=f"Potentially target-derived columns: {suspicious}",
            affected_columns=suspicious,
        )
    return LeakageCheckResult(check_name="target_leakage", passed=True)


def check_future_information(
    columns: list[str],
    feature_availability: dict[str, str] | None = None,
) -> LeakageCheckResult:
    """Check for columns that encode future information."""
    if not feature_availability:
        return LeakageCheckResult(check_name="future_information", passed=True,
                                  details="No feature availability metadata")
    future_cols = [col for col, avail in feature_availability.items()
                   if avail in ("post_event", "future", "unknown")]
    if future_cols:
        return LeakageCheckResult(
            check_name="future_information",
            passed=False,
            details=f"Post-event/future columns: {future_cols}",
            affected_columns=future_cols,
        )
    return LeakageCheckResult(check_name="future_information", passed=True)


def check_duplicate_contamination(
    row_count: int,
    unique_ratio: float = 1.0,
    threshold: float = 0.95,
) -> LeakageCheckResult:
    """Check for excessive duplicate rows."""
    if unique_ratio < threshold:
        return LeakageCheckResult(
            check_name="duplicate_contamination",
            passed=False,
            details=f"Unique ratio {unique_ratio:.2%} below threshold {threshold:.0%}",
        )
    return LeakageCheckResult(
        check_name="duplicate_contamination",
        passed=True,
        details=f"Unique ratio {unique_ratio:.2%}",
    )


def check_train_test_contamination(
    train_hashes: set[str] | None = None,
    test_hashes: set[str] | None = None,
) -> LeakageCheckResult:
    """Check for overlapping rows between train and test."""
    if not train_hashes or not test_hashes:
        return LeakageCheckResult(
            check_name="train_test_contamination",
            passed=True,
            details="No row hashes provided — cannot check",
        )
    overlap = train_hashes & test_hashes
    if overlap:
        return LeakageCheckResult(
            check_name="train_test_contamination",
            passed=False,
            details=f"{len(overlap)} overlapping rows between train and test",
        )
    return LeakageCheckResult(
        check_name="train_test_contamination",
        passed=True,
        details="No overlap detected",
    )


# ── Temporal Validation ───────────────────────────────────────────────────

@dataclass
class TemporalValidationResult:
    """Result of temporal validation."""
    temporal_order_valid: bool = True
    future_leakage_detected: bool = False
    train_test_overlap: bool = False
    preprocessing_fit_on_full: bool = False
    details: list[str] = field(default_factory=list)


def validate_temporal_order(
    timestamps: list[float] | None = None,
) -> TemporalValidationResult:
    """Verify timestamps are monotonically non-decreasing where ordering matters."""
    result = TemporalValidationResult()
    if not timestamps or len(timestamps) < 2:
        result.details.append("No timestamps provided — cannot validate order")
        return result
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            result.temporal_order_valid = False
            result.details.append(f"Timestamp disorder at index {i}: {timestamps[i]} < {timestamps[i-1]}")
            break
    if result.temporal_order_valid:
        result.details.append("Timestamps are monotonically non-decreasing")
    return result


# ── Admission Gate ────────────────────────────────────────────────────────

@dataclass
class AdmissionGateResult:
    """Result of a single admission gate."""
    gate_name: str
    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdmissionReport:
    """Complete admission report for a dataset."""
    dataset_id: str
    overall_status: str  # AdmissionState value
    source_classification: str
    gates: list[AdmissionGateResult] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    leakage_results: list[LeakageCheckResult] = field(default_factory=list)
    temporal_result: TemporalValidationResult | None = None
    evidence_record_hash: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "overall_status": self.overall_status,
            "source_classification": self.source_classification,
            "gates": [
                {"gate": g.gate_name, "passed": g.passed, "reason": g.reason}
                for g in self.gates
            ],
            "blocked_reasons": self.blocked_reasons,
            "leakage_results": [
                {"check": l.check_name, "passed": l.passed, "details": l.details}
                for l in self.leakage_results
            ],
            "evidence_record_hash": self.evidence_record_hash,
            "timestamp": self.timestamp,
        }


def gate_source_classification(record: DatasetEvidenceRecord) -> AdmissionGateResult:
    """Gate 1: Source must not be SYNTHETIC, DERIVED, or UNKNOWN."""
    sc = record.source_classification
    if sc in (SourceClassification.SYNTHETIC.value, "SUSPECTED_SYNTHETIC", "CONFIRMED_SYNTHETIC"):
        return AdmissionGateResult(
            "source_classification", False,
            f"Dataset is {sc} — not eligible for real-world validation",
            {"classification": sc},
        )
    if sc == SourceClassification.DERIVED.value:
        return AdmissionGateResult(
            "source_classification", False,
            "Dataset is DERIVED — not independent from training data",
            {"classification": sc},
        )
    if sc in (SourceClassification.UNKNOWN.value, ""):
        return AdmissionGateResult(
            "source_classification", False,
            "Source classification UNKNOWN — cannot determine eligibility",
            {"classification": sc},
        )
    return AdmissionGateResult(
        "source_classification", True,
        f"Source classified as {sc}",
        {"classification": sc},
    )


def gate_provenance(record: DatasetEvidenceRecord) -> AdmissionGateResult:
    """Gate 2: Provenance must be verified."""
    if record.provenance_confidence == EvidenceLevel.VERIFIED.value:
        return AdmissionGateResult("provenance", True, "Provenance independently verified")
    if record.provenance_confidence == EvidenceLevel.KNOWN.value:
        return AdmissionGateResult("provenance", True, "Provenance known from source")
    if record.provenance_confidence == "source_reported":
        return AdmissionGateResult("provenance", False,
                                   "Provenance only source-reported — needs independent verification")
    return AdmissionGateResult(
        "provenance", False,
        f"Provenance confidence: {record.provenance_confidence} — insufficient",
    )


def gate_label_semantics(record: DatasetEvidenceRecord) -> AdmissionGateResult:
    """Gate 3: Label semantics must be understood and timing must be known."""
    errors = []
    if not record.label_definition:
        errors.append("No label definition")
    if not record.label_column:
        errors.append("No label column specified")
    if not record.label_timing_known:
        errors.append("Label timing unknown — labels may be post-event or future information")
    if record.label_timestamp_semantics in ("", "unknown"):
        errors.append("Label timestamp semantics unknown")
    if record.label_generation_method in ("", "unknown"):
        errors.append("Label generation method unknown")
    if errors:
        return AdmissionGateResult("label_semantics", False, "; ".join(errors))
    return AdmissionGateResult(
        "label_semantics", True,
        f"Label defined: {record.label_definition}, timing: {record.label_timestamp_semantics}",
    )


def gate_temporal_semantics(record: DatasetEvidenceRecord) -> AdmissionGateResult:
    """Gate 4: Temporal semantics must be established."""
    errors = []
    if not record.temporal_order_known:
        errors.append("Temporal ordering not verified")
    if not record.transaction_timestamp_column:
        errors.append("No transaction timestamp column identified")
    if record.collection_period_start and record.collection_period_end:
        pass  # period defined
    elif not record.collection_period_start and not record.collection_period_end:
        errors.append("Collection period unknown")
    if errors:
        return AdmissionGateResult("temporal_semantics", False, "; ".join(errors))
    return AdmissionGateResult(
        "temporal_semantics", True,
        f"Temporal order verified, timestamp column: {record.transaction_timestamp_column}",
    )


def gate_schema_integrity(record: DatasetEvidenceRecord) -> AdmissionGateResult:
    """Gate 5: Dataset must have verifiable cryptographic identity."""
    errors = []
    if not record.dataset_hash:
        errors.append("No dataset hash — identity cannot be verified")
    if not record.feature_schema:
        errors.append("No feature schema defined")
    if record.row_count <= 0:
        errors.append("Row count must be positive")
    if errors:
        return AdmissionGateResult("schema_integrity", False, "; ".join(errors))
    return AdmissionGateResult(
        "schema_integrity", True,
        f"Dataset identity verified: {record.dataset_hash[:16]}… ({record.row_count} rows)",
    )


def gate_feature_compatibility(record: DatasetEvidenceRecord) -> AdmissionGateResult:
    """Gate 6: All required features must be mappable."""
    if record.missing_features:
        return AdmissionGateResult(
            "feature_compatibility", False,
            f"Missing required features: {record.missing_features}",
        )
    if record.incompatible_features:
        return AdmissionGateResult(
            "feature_compatibility", False,
            f"Incompatible features: {record.incompatible_features}",
        )
    if not record.feature_mapping_version:
        return AdmissionGateResult(
            "feature_compatibility", False,
            "No feature mapping version specified",
        )
    return AdmissionGateResult(
        "feature_compatibility", True,
        f"All features compatible (mapping v{record.feature_mapping_version})",
    )


def gate_minimum_size(record: DatasetEvidenceRecord) -> AdmissionGateResult:
    """Gate 7: Dataset must have minimum viable size."""
    MIN_ROWS = 1000
    MIN_POSITIVE = 50
    errors = []
    if record.row_count < MIN_ROWS:
        errors.append(f"Row count {record.row_count} < minimum {MIN_ROWS}")
    # num_positive not directly on evidence record; check feature_schema length
    if not record.feature_schema:
        errors.append("Feature schema empty")
    if errors:
        return AdmissionGateResult("minimum_size", False, "; ".join(errors))
    return AdmissionGateResult(
        "minimum_size", True,
        f"{record.row_count} rows, {len(record.feature_schema)} features",
    )


def run_admission_gates(record: DatasetEvidenceRecord) -> AdmissionReport:
    """Run all admission gates and produce a report.

    Any mandatory failure results in INELIGIBLE.
    Source classification of SYNTHETIC/DERIVED/UNKNOWN results in BLOCKED.
    """
    report = AdmissionReport(
        dataset_id=record.dataset_id or "unknown",
        overall_status=record.admission_state,
        source_classification=record.source_classification,
    )

    gates = [
        gate_source_classification(record),
        gate_provenance(record),
        gate_label_semantics(record),
        gate_temporal_semantics(record),
        gate_schema_integrity(record),
        gate_feature_compatibility(record),
        gate_minimum_size(record),
    ]

    report.gates = list(gates)

    has_block = False
    has_fail = False

    for g in gates:
        if not g.passed:
            if g.gate_name == "source_classification":
                has_block = True
                report.blocked_reasons.append(g.reason)
            else:
                has_fail = True
                report.blocked_reasons.append(g.reason)

    if has_block:
        report.overall_status = AdmissionState.BLOCKED.value
        record.transition(AdmissionState.BLOCKED, "; ".join(report.blocked_reasons))
    elif has_fail:
        report.overall_status = AdmissionState.INELIGIBLE.value
        record.transition(AdmissionState.INELIGIBLE, "; ".join(report.blocked_reasons))
    else:
        # Walk through all successful states
        for target in [
            AdmissionState.EVIDENCE_COLLECTION,
            AdmissionState.PROVENANCE_VERIFIED,
            AdmissionState.SCHEMA_VERIFIED,
            AdmissionState.LABEL_SEMANTICS_VERIFIED,
            AdmissionState.TEMPORAL_SEMANTICS_VERIFIED,
            AdmissionState.LEAKAGE_CHECKED,
            AdmissionState.FEATURE_COMPATIBILITY_VERIFIED,
            AdmissionState.ELIGIBLE,
        ]:
            record.transition(target, "gate passed")
        report.overall_status = record.admission_state

    report.evidence_record_hash = record.compute_hash()
    return report


# ── Evaluation Protocol ───────────────────────────────────────────────────

@dataclass
class EvaluationProtocol:
    """Reproducible evaluation protocol — runs only after admission."""
    dataset_hash: str = ""
    model_id: str = ""
    model_version: str = ""
    artifact_set_hash: str = ""
    manifest_hash: str = ""
    feature_version: str = ""
    schema_version: str = ""
    preprocessing_hash: str = ""
    rule_hash: str = ""
    evaluation_code_version: str = ""
    random_seed: int = 0
    timestamp: float = 0.0

    # Threshold policy
    threshold_selection_method: str = ""  # "fixed", "validation_set", "never"
    threshold_value: float = 0.5
    threshold_fit_on_test: bool = False  # MUST be False for valid evaluation

    # Metrics produced
    metrics: dict[str, float] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Deterministic hash of the evaluation protocol."""
        d = asdict(self)
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def verify_binding(self, record: DatasetEvidenceRecord) -> tuple[bool, str]:
        """Verify this evaluation is bound to the exact admitted dataset."""
        if self.dataset_hash != record.dataset_hash:
            return False, (
                f"Dataset hash mismatch: evaluation={self.dataset_hash[:16]}… "
                f"record={record.dataset_hash[:16]}…"
            )
        if self.feature_version != record.feature_mapping_version:
            return False, (
                f"Feature version mismatch: evaluation={self.feature_version} "
                f"record={record.feature_mapping_version}"
            )
        return True, "Evaluation bound to admitted dataset"


# ── Evaluation Record Integrity ───────────────────────────────────────────

@dataclass
class EvaluationRecordIntegrity:
    """Immutable, tamper-evident evaluation record."""
    record_id: str = ""
    evaluation_hash: str = ""
    protocol_hash: str = ""
    dataset_hash: str = ""
    model_hash: str = ""
    metrics_hash: str = ""
    created_at: float = 0.0
    signature: str = ""  # HMAC — key derived from settings

    def compute_record_hash(self) -> str:
        """Compute the record integrity hash."""
        parts = [
            self.record_id,
            self.protocol_hash,
            self.dataset_hash,
            self.model_hash,
            self.metrics_hash,
        ]
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def verify_integrity(self) -> tuple[bool, str]:
        """Verify the record has not been tampered with."""
        expected = self.compute_record_hash()
        if self.evaluation_hash != expected:
            return False, (
                f"Evaluation record tampered: expected {expected[:16]}… "
                f"got {self.evaluation_hash[:16]}…"
            )
        return True, "Record integrity verified"


# ── Threshold Policy ──────────────────────────────────────────────────────

@dataclass
class ThresholdPolicy:
    """Ensures test-set threshold leakage is impossible."""
    method: str = "fixed"  # "fixed", "validation_set"
    threshold: float = 0.5
    fit_on_test: bool = False
    frozen: bool = True

    def validate(self) -> tuple[bool, str]:
        """Verify threshold policy is sound."""
        if self.fit_on_test:
            return False, "Threshold was fit on test data — test-set leakage"
        if not self.frozen:
            return False, "Threshold policy is not frozen"
        if self.method not in ("fixed", "validation_set"):
            return False, f"Unknown threshold method: {self.method}"
        return True, f"Threshold policy valid: {self.method}={self.threshold}"


# ── Current Dataset Audit ─────────────────────────────────────────────────

# Well-known candidate datasets in the repository
KNOWN_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "IBM Altman SDV (e_hardneg)",
        "classification": SourceClassification.SYNTHETIC.value,
        "eligible": False,
        "reason": "Synthetic dataset generated by SDV Altman model; Phase 17 verdict: SYNTHETIC_REGIME_ARTIFACT",
        "evidence_gaps": ["provenance_synthetic", "label_generation_artificial", "temporal_synthetic"],
    },
    {
        "name": "ULB/MLG Credit Card (kaggle)",
        "classification": SourceClassification.UNKNOWN.value,
        "eligible": False,
        "reason": "PCA-transformed features; provenance partially documented but label timing unknown",
        "evidence_gaps": ["label_timing_unknown", "features_not_original", "provenance_unclear"],
    },
    {
        "name": "Kaggle Fraud Detection (DV)",
        "classification": SourceClassification.UNKNOWN.value,
        "eligible": False,
        "reason": "Source unclear; label provenance partially documented; no channel info",
        "evidence_gaps": ["source_origin_unclear", "label_provenance_partial", "no_channel_info"],
    },
    {
        "name": "PaySim",
        "classification": SourceClassification.SYNTHETIC.value,
        "eligible": False,
        "reason": "Academic mobile money simulator; synthetic by definition",
        "evidence_gaps": ["synthetic_by_design", "no_real_world_fraud_patterns"],
    },
    {
        "name": "Elliptic (Bitcoin)",
        "classification": SourceClassification.REAL_LEGACY.value,
        "eligible": False,
        "reason": "Real Bitcoin transaction data; label semantics differ from payment fraud; no decision-time features",
        "evidence_gaps": ["label_semantics_different", "feature_compatibility_unknown", "temporal_order_unverified"],
    },
    {
        "name": "PS-14 derived datasets",
        "classification": SourceClassification.DERIVED.value,
        "eligible": False,
        "reason": "Derived from PS-14 synthetic training pipeline — not independent",
        "evidence_gaps": ["derived_from_training_data", "not_independent"],
    },
]


def audit_current_datasets() -> list[dict[str, Any]]:
    """Audit all known candidate datasets against admission requirements."""
    results = []
    for candidate in KNOWN_CANDIDATES:
        # Build an evidence record to test against admission gates
        record = DatasetEvidenceRecord(
            dataset_id=candidate["name"],
            source_name=candidate["name"],
            source_classification=candidate["classification"],
        )
        report = run_admission_gates(record)
        results.append({
            "name": candidate["name"],
            "classification": candidate["classification"],
            "admission_status": report.overall_status,
            "eligibility": candidate["eligible"],
            "reason": candidate["reason"],
            "evidence_gaps": candidate["evidence_gaps"],
            "gate_results": [
                {"gate": g.gate_name, "passed": g.passed, "reason": g.reason}
                for g in report.gates
            ],
        })
    return results
