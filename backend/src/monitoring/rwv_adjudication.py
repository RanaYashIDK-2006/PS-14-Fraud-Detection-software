"""Phase 97: RWV Result Adjudication & Acceptance Gate.

Deterministic fail-closed adjudication of RWV evaluation records.
Distinguishes VALID_EVALUATION from ACCEPTED_RWV_RESULT from
PROMOTION_ELIGIBLE_EVIDENCE -- these are NOT interchangeable.

Does NOT acquire data, contact providers, modify models, retrain,
tune thresholds, or change REAL_WORLD_VALIDATION.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.monitoring.rwv_execution import (
    RWVEvaluationRecord,
    EvaluationMetrics,
    ExecutionStatus,
    build_evaluation_config,
    compute_metrics,
)
from src.monitoring.provider_evidence import (
    ProviderEvidence,
    QualificationReport,
    qualify_dataset,
    compute_evidence_hash,
    KNOWN_CANDIDATES,
    DatasetQualificationState,
)
from src.monitoring.rwv_readiness_audit import (
    MODEL_ID,
    RELEASE_ID,
    PRODUCTION_THRESHOLD,
    FEATURE_VERSION,
    build_evidence_pack,
    validate_evidence_pack,
)
from src.monitoring.real_world_evaluation_protocol import (
    build_protocol,
    build_identity_lock,
    NATIVE_48,
)
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

ADJUDICATION_POLICY_VERSION = "phase97_v1"
EVAL_PROTOCOL_VERSION = "phase93_v1"
ACCEPTANCE_SPEC_VERSION = "phase91_v1"


# ══════════════════════════════════════════════════════════════════════
# VALIDITY STATES
# ══════════════════════════════════════════════════════════════════════

class ValidityStatus(str, Enum):
    EVALUATION_INVALID = "evaluation_invalid"
    EVALUATION_VALID = "evaluation_valid"
    EVALUATION_VALID_WITH_WARNINGS = "evaluation_valid_with_warnings"


# ══════════════════════════════════════════════════════════════════════
# ACCEPTANCE STATES
# ══════════════════════════════════════════════════════════════════════

class AcceptanceStatus(str, Enum):
    NOT_ADJUDICATED = "rwv_result_not_adjudicated"
    REJECTED = "rwv_result_rejected"
    ACCEPTED = "rwv_result_accepted"
    ACCEPTED_WITH_LIMITATIONS = "rwv_result_accepted_with_limitations"
    CRITERIA_INCOMPLETE = "acceptance_criteria_incomplete"


# ══════════════════════════════════════════════════════════════════════
# PROMOTION EVIDENCE STATES
# ══════════════════════════════════════════════════════════════════════

class PromotionEvidenceStatus(str, Enum):
    NOT_PROMOTION_ELIGIBLE = "not_promotion_eligible"
    PROMOTION_EVIDENCE_ELIGIBLE = "promotion_evidence_eligible"


# ══════════════════════════════════════════════════════════════════════
# VALIDITY REASON CODES
# ══════════════════════════════════════════════════════════════════════

class ValidityReason(str, Enum):
    HASH_MISMATCH = "hash_mismatch"
    WRONG_MODEL = "wrong_model"
    WRONG_RELEASE = "wrong_release"
    WRONG_THRESHOLD = "wrong_threshold"
    WRONG_FEATURE_VERSION = "wrong_feature_version"
    WRONG_PROTOCOL = "wrong_protocol"
    WRONG_ACCEPTANCE_SPEC = "wrong_acceptance_spec"
    WRONG_DATASET_VERSION = "wrong_dataset_version"
    WRONG_DATASET_QUALIFICATION = "wrong_dataset_qualification"
    MISSING_METRIC = "missing_metric"
    MALFORMED_RESULT = "malformed_result"
    INCONSISTENT_SAMPLE_COUNTS = "inconsistent_sample_counts"
    INVALID_EXCLUSION_ACCOUNTING = "invalid_exclusion_accounting"
    TEMPORAL_VIOLATION = "temporal_violation"
    LEAKAGE_VIOLATION = "leakage_violation"
    CONTAMINATION_VIOLATION = "contamination_violation"
    TAMPERED_EVIDENCE = "tampered_evidence"
    EXECUTION_FAILED = "execution_failed"
    METRIC_CONSISTENCY_FAILURE = "metric_consistency_failure"


# ══════════════════════════════════════════════════════════════════════
# CRITERION EVALUATION
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CriterionResult:
    """Deterministic result for one acceptance criterion."""
    criterion_id: str
    description: str
    observed_value: str
    required_value: str
    status: str  # "pass" | "fail" | "incomplete"
    evidence_reference: str
    protocol_version: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ══════════════════════════════════════════════════════════════════════
# ADJUDICATION RECORD
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RWVAdjudication:
    """Immutable RWV adjudication result."""
    session_id: str
    evaluation_record_hash: str
    model_id: str
    release_id: str
    release_manifest_hash: str
    dataset_id: str
    dataset_version: str
    qualification_hash: str
    evaluation_protocol_version: str
    acceptance_spec_version: str
    evaluation_config_hash: str
    validity_status: str
    acceptance_status: str
    promotion_evidence_status: str
    metric_results: dict[str, Any]
    criteria_results: tuple[dict[str, Any], ...]
    coverage_result: dict[str, Any]
    temporal_result: dict[str, Any]
    leakage_result: dict[str, Any]
    contamination_result: dict[str, Any]
    exclusion_result: dict[str, Any]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    adjudication_hash: str
    policy_version: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["criteria_results"] = list(d["criteria_results"])
        d["blockers"] = list(d["blockers"])
        d["warnings"] = list(d["warnings"])
        return d


# ══════════════════════════════════════════════════════════════════════
# AUDIT EVENT
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AdjudicationAuditEvent:
    """Immutable audit event for adjudication lifecycle."""
    event_type: str
    session_id: str
    timestamp: str
    details: str
    event_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ══════════════════════════════════════════════════════════════════════
# HASHING UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _hash_dict(d: dict[str, Any]) -> str:
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════
# AUDIT EVENTS
# ══════════════════════════════════════════════════════════════════════

def create_adjudication_audit_event(
    event_type: str,
    session_id: str,
    details: str = "",
) -> AdjudicationAuditEvent:
    ts = _now_iso()
    event_dict = {"event_type": event_type, "session_id": session_id,
                  "timestamp": ts, "details": details}
    return AdjudicationAuditEvent(
        event_type=event_type, session_id=session_id, timestamp=ts,
        details=details, event_hash=_hash_dict(event_dict),
    )


# ══════════════════════════════════════════════════════════════════════
# VALIDITY CHECKS
# ══════════════════════════════════════════════════════════════════════

def _check_validity(
    record: RWVEvaluationRecord,
    expected_model_id: str = MODEL_ID,
    expected_release_id: str = RELEASE_ID,
    expected_threshold: float = PRODUCTION_THRESHOLD,
    expected_feature_version: str = FEATURE_VERSION,
    expected_protocol: str = EVAL_PROTOCOL_VERSION,
    expected_spec: str = ACCEPTANCE_SPEC_VERSION,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Check evaluation record validity. Returns (status, blockers, warnings)."""
    blockers: list[str] = []
    warnings: list[str] = []

    # Execution status
    if record.execution_status != ExecutionStatus.COMPLETED.value:
        blockers.append(ValidityReason.EXECUTION_FAILED.value)

    # Model identity
    if record.model_id != expected_model_id:
        blockers.append(ValidityReason.WRONG_MODEL.value)

    # Release identity
    if record.release_id != expected_release_id:
        blockers.append(ValidityReason.WRONG_RELEASE.value)

    # Feature version
    if record.feature_contract_version != expected_feature_version:
        blockers.append(ValidityReason.WRONG_FEATURE_VERSION.value)

    # Native feature version
    if record.native_feature_version != expected_feature_version:
        blockers.append(ValidityReason.WRONG_FEATURE_VERSION.value)

    # Protocol version
    if record.evaluation_protocol_version != expected_protocol:
        blockers.append(ValidityReason.WRONG_PROTOCOL.value)

    # Acceptance spec
    if record.acceptance_spec_version != expected_spec:
        blockers.append(ValidityReason.WRONG_ACCEPTANCE_SPEC.value)

    # Metrics present
    required_metrics = [
        "sample_count", "positive_count", "negative_count",
        "precision", "recall", "f1",
    ]
    for m in required_metrics:
        if m not in record.metrics:
            blockers.append(f"{ValidityReason.MISSING_METRIC.value}:{m}")

    # Metric consistency
    metrics = record.metrics
    if "sample_count" in metrics and "positive_count" in metrics and "negative_count" in metrics:
        sc = metrics["sample_count"]
        pc = metrics["positive_count"]
        nc = metrics["negative_count"]
        ec = metrics.get("excluded_count", 0)

        if sc > 0 and (pc + nc + ec) != sc:
            # Allow coverage gap but flag it
            if pc + nc > sc:
                blockers.append(ValidityReason.INCONSISTENT_SAMPLE_COUNTS.value)
            else:
                warnings.append("coverage_gap_in_sample_counts")

        # Recompute precision
        if "precision" in metrics:
            tp = metrics.get("tp", metrics.get("positive_count", 0) - metrics.get("fn", 0))
            fp = metrics.get("fp", 0)
            expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if abs(metrics["precision"] - expected_prec) > 1e-9:
                warnings.append("precision_may_be_inconsistent")

    # Temporal result
    if record.temporal_summary.get("valid") is False:
        blockers.append(ValidityReason.TEMPORAL_VIOLATION.value)

    # Leakage check
    leakage = record.leakage_check_result.lower()
    leakage_safe = "no_leakage" in leakage or "synthetic" in leakage or "safe" in leakage
    if not leakage_safe and ("violation" in leakage or "leak" in leakage):
        blockers.append(ValidityReason.LEAKAGE_VIOLATION.value)

    # Contamination check
    contamination = record.contamination_check_result.lower()
    if "contaminat" in contamination and "no" not in contamination:
        blockers.append(ValidityReason.CONTAMINATION_VIOLATION.value)

    # Sample count > 0
    if metrics.get("sample_count", 0) == 0:
        blockers.append(ValidityReason.MALFORMED_RESULT.value)

    # Fraud prevalence consistency
    if metrics.get("sample_count", 0) > 0 and "fraud_prevalence" in metrics:
        expected_prev = metrics["positive_count"] / metrics["sample_count"]
        if abs(metrics["fraud_prevalence"] - expected_prev) > 1e-9:
            warnings.append("fraud_prevalence_may_be_inconsistent")

    # Coverage consistency
    if metrics.get("sample_count", 0) > 0 and "coverage" in metrics:
        if metrics["coverage"] < 0.0 or metrics["coverage"] > 1.0:
            blockers.append(ValidityReason.MALFORMED_RESULT.value)

    if blockers:
        status = ValidityStatus.EVALUATION_INVALID.value
    elif warnings:
        status = ValidityStatus.EVALUATION_VALID_WITH_WARNINGS.value
    else:
        status = ValidityStatus.EVALUATION_VALID.value

    return status, tuple(blockers), tuple(warnings)


# ══════════════════════════════════════════════════════════════════════
# ACCEPTANCE CRITERIA
# ══════════════════════════════════════════════════════════════════════

def _evaluate_acceptance_criteria(
    record: RWVEvaluationRecord,
) -> tuple[str, tuple[dict[str, Any], ...], tuple[str, ...]]:
    """Evaluate acceptance criteria. Returns (status, criteria, blockers).

    Acceptance criteria come exclusively from Phase 91/93.
    Do NOT invent numerical thresholds.
    """
    criteria: list[dict[str, Any]] = []
    blockers: list[str] = []
    metrics = record.metrics

    def _eval(cid: str, desc: str, observed: str, required: str,
              passed: bool, evidence: str) -> dict[str, Any]:
        c = {
            "criterion_id": cid,
            "description": desc,
            "observed_value": observed,
            "required_value": required,
            "status": "pass" if passed else "fail",
            "evidence_reference": evidence,
            "protocol_version": ADJUDICATION_POLICY_VERSION,
        }
        criteria.append(c)
        if not passed:
            blockers.append(cid)
        return c

    # C1: Evaluation is valid
    _eval(
        "c1_evaluation_validity",
        "Evaluation record must be valid",
        "valid" if record.execution_status == ExecutionStatus.COMPLETED.value else "invalid",
        "valid",
        record.execution_status == ExecutionStatus.COMPLETED.value,
        f"execution_status={record.execution_status}",
    )

    # C2: Model identity
    _eval(
        "c2_model_identity",
        "Model must match production model",
        record.model_id,
        MODEL_ID,
        record.model_id == MODEL_ID,
        f"model_id={record.model_id}",
    )

    # C3: Release identity
    _eval(
        "c3_release_identity",
        "Release must match production release",
        record.release_id,
        RELEASE_ID,
        record.release_id == RELEASE_ID,
        f"release_id={record.release_id}",
    )

    # C4: Feature contract
    _eval(
        "c4_feature_contract",
        "Feature version must match production",
        record.feature_contract_version,
        FEATURE_VERSION,
        record.feature_contract_version == FEATURE_VERSION,
        f"feature_version={record.feature_contract_version}",
    )

    # C5: Evaluation protocol
    _eval(
        "c5_evaluation_protocol",
        "Protocol version must match",
        record.evaluation_protocol_version,
        EVAL_PROTOCOL_VERSION,
        record.evaluation_protocol_version == EVAL_PROTOCOL_VERSION,
        f"protocol={record.evaluation_protocol_version}",
    )

    # C6: Acceptance specification
    _eval(
        "c6_acceptance_specification",
        "Acceptance spec version must match",
        record.acceptance_spec_version,
        ACCEPTANCE_SPEC_VERSION,
        record.acceptance_spec_version == ACCEPTANCE_SPEC_VERSION,
        f"spec={record.acceptance_spec_version}",
    )

    # C7: Sample count
    sc = metrics.get("sample_count", 0)
    _eval(
        "c7_sample_count",
        "Must have non-zero samples",
        str(sc),
        "> 0",
        sc > 0,
        f"sample_count={sc}",
    )

    # C8: Coverage
    cov = metrics.get("coverage", 0.0)
    _eval(
        "c8_coverage",
        "Coverage must be positive",
        f"{cov:.4f}",
        "> 0",
        cov > 0,
        f"coverage={cov}",
    )

    # C9: Labels present
    pc = metrics.get("positive_count", 0)
    nc = metrics.get("negative_count", 0)
    _eval(
        "c9_labels_present",
        "Must have both positive and negative labels",
        f"pos={pc},neg={nc}",
        "both > 0",
        pc > 0 and nc > 0,
        f"positive={pc},negative={nc}",
    )

    # C10: Precision computable
    prec = metrics.get("precision", -1.0)
    _eval(
        "c10_precision",
        "Precision must be valid [0, 1]",
        f"{prec:.6f}",
        "in [0, 1]",
        0.0 <= prec <= 1.0,
        f"precision={prec}",
    )

    # C11: Recall computable
    rec = metrics.get("recall", -1.0)
    _eval(
        "c11_recall",
        "Recall must be valid [0, 1]",
        f"{rec:.6f}",
        "in [0, 1]",
        0.0 <= rec <= 1.0,
        f"recall={rec}",
    )

    # C12: F1 computable
    f1 = metrics.get("f1", -1.0)
    _eval(
        "c12_f1",
        "F1 must be valid [0, 1]",
        f"{f1:.6f}",
        "in [0, 1]",
        0.0 <= f1 <= 1.0,
        f"f1={f1}",
    )

    # C13: No NaN/Inf in metrics
    has_nan = False
    for k, v in metrics.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            has_nan = True
            break
    _eval(
        "c13_no_nan_inf",
        "Metrics must not contain NaN or Inf",
        "nan_inf_found" if has_nan else "clean",
        "clean",
        not has_nan,
        f"has_nan={has_nan}",
    )

    # C14: Temporal validity
    temporal_valid = record.temporal_summary.get("valid", False)
    _eval(
        "c14_temporal_validity",
        "Temporal ordering must be valid",
        "valid" if temporal_valid else "invalid",
        "valid",
        temporal_valid,
        f"temporal={record.temporal_summary}",
    )

    # C15: Leakage check
    leakage_safe = "no_leakage" in record.leakage_check_result.lower() or "synthetic" in record.leakage_check_result.lower()
    _eval(
        "c15_leakage_check",
        "No leakage detected",
        record.leakage_check_result,
        "no leakage",
        leakage_safe,
        f"leakage={record.leakage_check_result}",
    )

    # C16: Contamination check
    contam_safe = "no_contamination" in record.contamination_check_result.lower() or "synthetic" in record.contamination_check_result.lower()
    _eval(
        "c16_contamination_check",
        "No contamination detected",
        record.contamination_check_result,
        "no contamination",
        contam_safe,
        f"contamination={record.contamination_check_result}",
    )

    # C17: Exclusion accounting
    ec = metrics.get("excluded_count", 0)
    _eval(
        "c17_exclusion_accounting",
        "Exclusion count must be non-negative",
        str(ec),
        ">= 0",
        ec >= 0,
        f"excluded={ec}",
    )

    # C18: Metric consistency
    tp = metrics.get("tp", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)
    tn = metrics.get("tn", 0)
    total_labeled = tp + fp + fn + tn
    if sc > 0 and total_labeled > 0:
        consistent = abs(total_labeled - (pc + nc)) <= 0 or total_labeled == sc
    else:
        consistent = True
    _eval(
        "c18_metric_consistency",
        "Confusion matrix must be consistent",
        f"tp={tp},fp={fp},fn={fn},tn={tn}",
        "consistent with counts",
        consistent,
        f"total_labeled={total_labeled}",
    )

    if blockers:
        status = AcceptanceStatus.REJECTED.value
    else:
        status = AcceptanceStatus.ACCEPTED.value

    return status, tuple(criteria), tuple(blockers)


# ══════════════════════════════════════════════════════════════════════
# PROMOTION EVIDENCE CHECK
# ══════════════════════════════════════════════════════════════════════

def _check_promotion_evidence(
    validity_status: str,
    acceptance_status: str,
    record: RWVEvaluationRecord,
) -> str:
    """Determine if the result is promotion-evidence eligible."""
    if validity_status != ValidityStatus.EVALUATION_VALID.value:
        return PromotionEvidenceStatus.NOT_PROMOTION_ELIGIBLE.value

    if acceptance_status not in (
        AcceptanceStatus.ACCEPTED.value,
        AcceptanceStatus.ACCEPTED_WITH_LIMITATIONS.value,
    ):
        return PromotionEvidenceStatus.NOT_PROMOTION_ELIGIBLE.value

    # Must have valid leakage and contamination
    leakage_safe = "no_leakage" in record.leakage_check_result.lower() or "synthetic" in record.leakage_check_result.lower()
    contam_safe = "no_contamination" in record.contamination_check_result.lower() or "synthetic" in record.contamination_check_result.lower()
    temporal_valid = record.temporal_summary.get("valid", False)

    if not (leakage_safe and contam_safe and temporal_valid):
        return PromotionEvidenceStatus.NOT_PROMOTION_ELIGIBLE.value

    # Must match production model and release
    if record.model_id != MODEL_ID or record.release_id != RELEASE_ID:
        return PromotionEvidenceStatus.NOT_PROMOTION_ELIGIBLE.value

    # Evidence pack must be valid
    pack = build_evidence_pack()
    pack_valid = validate_evidence_pack(pack).get("valid", False)
    if not pack_valid:
        return PromotionEvidenceStatus.NOT_PROMOTION_ELIGIBLE.value

    return PromotionEvidenceStatus.PROMOTION_EVIDENCE_ELIGIBLE.value


# ══════════════════════════════════════════════════════════════════════
# MAIN ADJUDICATION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def adjudicate_rwv_result(
    record: RWVEvaluationRecord,
    qualification: QualificationReport | None = None,
) -> RWVAdjudication:
    """Deterministic fail-closed adjudication of an RWV evaluation record.

    The result distinguishes:
      - VALID_EVALUATION (the record is internally consistent)
      - ACCEPTED_RWV_RESULT (the record satisfies acceptance criteria)
      - PROMOTION_ELIGIBLE_EVIDENCE (the result can support promotion)

    These are NOT interchangeable.
    """
    # 1. Validity check
    validity_status, validity_blockers, validity_warnings = _check_validity(record)

    # 2. Acceptance criteria
    acceptance_status, criteria, acceptance_blockers = _evaluate_acceptance_criteria(record)

    # 3. Promotion evidence
    promotion_status = _check_promotion_evidence(validity_status, acceptance_status, record)

    # 4. Coverage result
    metrics = record.metrics
    coverage_result = {
        "input_records": metrics.get("sample_count", 0) + metrics.get("excluded_count", 0),
        "evaluated_records": metrics.get("sample_count", 0),
        "excluded_records": metrics.get("excluded_count", 0),
        "coverage": metrics.get("coverage", 0.0),
        "fraud_prevalence": metrics.get("fraud_prevalence", 0.0),
    }

    # 5. Temporal result
    temporal_result = record.temporal_summary

    # 6. Leakage result
    leakage_result = {
        "status": record.leakage_check_result,
        "safe": "no_leakage" in record.leakage_check_result.lower()
                or "synthetic" in record.leakage_check_result.lower(),
    }

    # 7. Contamination result
    contamination_result = {
        "status": record.contamination_check_result,
        "safe": "no_contamination" in record.contamination_check_result.lower()
                or "synthetic" in record.contamination_check_result.lower(),
    }

    # 8. Exclusion result
    ec = metrics.get("excluded_count", 0)
    exclusion_result = {
        "excluded_count": ec,
        "exclusion_accounted": ec >= 0,
    }

    # 9. Metric results
    metric_results = {
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "f1": metrics.get("f1", 0.0),
        "specificity": metrics.get("specificity", 0.0),
        "false_positive_rate": metrics.get("false_positive_rate", 0.0),
        "false_negative_rate": metrics.get("false_negative_rate", 0.0),
        "sample_count": metrics.get("sample_count", 0),
        "positive_count": metrics.get("positive_count", 0),
        "negative_count": metrics.get("negative_count", 0),
        "excluded_count": metrics.get("excluded_count", 0),
        "coverage": metrics.get("coverage", 0.0),
        "fraud_prevalence": metrics.get("fraud_prevalence", 0.0),
    }

    # 10. Blockers
    all_blockers = list(validity_blockers) + list(acceptance_blockers)

    # 11. Evidence pack hash
    evidence_pack = build_evidence_pack()
    release_manifest_hash = evidence_pack.get("pack_hash", "")

    # 12. Qualification hash
    qual_hash = ""
    provider_id = ""
    dataset_version = ""
    if qualification is not None:
        qual_hash = qualification.evidence_hash
        provider_id = qualification.provider_id
        dataset_version = qualification.dataset_version

    # 13. Config hash
    eval_config = build_evaluation_config()

    # 14. Build adjudication
    adj_dict = {
        "session_id": record.session_id,
        "evaluation_record_hash": record.result_hash,
        "model_id": record.model_id,
        "release_id": record.release_id,
        "release_manifest_hash": release_manifest_hash,
        "dataset_id": record.dataset_id,
        "dataset_version": record.dataset_version,
        "qualification_hash": qual_hash,
        "evaluation_protocol_version": record.evaluation_protocol_version,
        "acceptance_spec_version": record.acceptance_spec_version,
        "evaluation_config_hash": eval_config.config_hash,
        "validity_status": validity_status,
        "acceptance_status": acceptance_status,
        "promotion_evidence_status": promotion_status,
    }
    adjudication_hash = _hash_dict(adj_dict)

    return RWVAdjudication(
        session_id=record.session_id,
        evaluation_record_hash=record.result_hash,
        model_id=record.model_id,
        release_id=record.release_id,
        release_manifest_hash=release_manifest_hash,
        dataset_id=record.dataset_id,
        dataset_version=record.dataset_version,
        qualification_hash=qual_hash,
        evaluation_protocol_version=record.evaluation_protocol_version,
        acceptance_spec_version=record.acceptance_spec_version,
        evaluation_config_hash=eval_config.config_hash,
        validity_status=validity_status,
        acceptance_status=acceptance_status,
        promotion_evidence_status=promotion_status,
        metric_results=metric_results,
        criteria_results=criteria,
        coverage_result=coverage_result,
        temporal_result=temporal_result,
        leakage_result=leakage_result,
        contamination_result=contamination_result,
        exclusion_result=exclusion_result,
        blockers=tuple(all_blockers),
        warnings=validity_warnings,
        adjudication_hash=adjudication_hash,
        policy_version=ADJUDICATION_POLICY_VERSION,
        created_at=_now_iso(),
    )


# ══════════════════════════════════════════════════════════════════════
# ADJUDICATION HASH VERIFICATION
# ══════════════════════════════════════════════════════════════════════

def verify_adjudication_hash(adj: RWVAdjudication) -> bool:
    """Verify the adjudication hash is consistent with its contents."""
    adj_dict = {
        "session_id": adj.session_id,
        "evaluation_record_hash": adj.evaluation_record_hash,
        "model_id": adj.model_id,
        "release_id": adj.release_id,
        "release_manifest_hash": adj.release_manifest_hash,
        "dataset_id": adj.dataset_id,
        "dataset_version": adj.dataset_version,
        "qualification_hash": adj.qualification_hash,
        "evaluation_protocol_version": adj.evaluation_protocol_version,
        "acceptance_spec_version": adj.acceptance_spec_version,
        "evaluation_config_hash": adj.evaluation_config_hash,
        "validity_status": adj.validity_status,
        "acceptance_status": adj.acceptance_status,
        "promotion_evidence_status": adj.promotion_evidence_status,
    }
    return _hash_dict(adj_dict) == adj.adjudication_hash


# ══════════════════════════════════════════════════════════════════════
# SYNTHETIC FIXTURES FOR TESTING
# ══════════════════════════════════════════════════════════════════════

def _make_valid_record(**overrides: Any) -> RWVEvaluationRecord:
    """Create a valid evaluation record for testing."""
    defaults = {
        "session_id": "test-session-001",
        "provider_id": "SYNTHETIC_TEST",
        "dataset_id": "SYNTHETIC_DATA",
        "dataset_version": "1.0",
        "qualification_hash": "abc123",
        "dataset_hash": "def456",
        "model_id": MODEL_ID,
        "release_id": RELEASE_ID,
        "release_manifest_hash": "ghi789",
        "feature_contract_version": FEATURE_VERSION,
        "native_feature_version": FEATURE_VERSION,
        "evaluation_protocol_version": EVAL_PROTOCOL_VERSION,
        "acceptance_spec_version": ACCEPTANCE_SPEC_VERSION,
        "evaluation_config_hash": "jkl012",
        "metrics": {
            "sample_count": 1000,
            "positive_count": 50,
            "negative_count": 950,
            "excluded_count": 0,
            "coverage": 1.0,
            "fraud_prevalence": 0.05,
            "precision": 0.80,
            "recall": 0.60,
            "specificity": 0.95,
            "false_positive_rate": 0.05,
            "false_negative_rate": 0.40,
            "f1": 0.6857,
            "tp": 30,
            "fp": 7,
            "fn": 20,
            "tn": 943,
            "risk_band_distribution": {},
            "metrics_hash": "",
        },
        "subgroup_results": {"amount_band": "not_available"},
        "temporal_summary": {"valid": True, "count": 1000},
        "leakage_check_result": "synthetic_no_leakage_possible",
        "contamination_check_result": "synthetic_no_contamination",
        "execution_status": ExecutionStatus.COMPLETED.value,
        "result_hash": "",
        "created_at": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    rec = RWVEvaluationRecord(**defaults)
    # Compute result hash
    from src.monitoring.rwv_execution import compute_result_hash
    rh = compute_result_hash(rec)
    return RWVEvaluationRecord(**{**defaults, "result_hash": rh})
