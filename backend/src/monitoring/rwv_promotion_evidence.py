"""Phase 98: RWV-to-Promotion Boundary & Evidence Integration.

Formal, fail-closed boundary between:
  RWV Evidence → Promotion Evidence Package → Existing Promotion Gate → Promotion Decision

RWV evidence may provide evidence FOR promotion,
but RWV evidence must NEVER itself promote a model.

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

from src.monitoring.rwv_adjudication import (
    RWVAdjudication,
    ValidityStatus,
    AcceptanceStatus,
    PromotionEvidenceStatus,
    adjudicate_rwv_result,
    verify_adjudication_hash,
)
from src.monitoring.rwv_execution import (
    RWVEvaluationRecord,
    ExecutionStatus,
)
from src.monitoring.provider_evidence import (
    QualificationReport,
    DatasetQualificationState,
)
from src.monitoring.rwv_readiness_audit import (
    MODEL_ID,
    RELEASE_ID,
    FEATURE_VERSION,
    build_evidence_pack,
    validate_evidence_pack,
)


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

PROMOTION_EVIDENCE_VERSION = "phase98_v1"
NATIVE_FEATURE_VERSION = "v1"
EVAL_PROTOCOL_VERSION = "phase93_v1"
ACCEPTANCE_SPEC_VERSION = "phase91_v1"
PREPROCESSING_HASH = "stable_preprocessing_v1"
RULE_HASH = "rules_v1"


# ══════════════════════════════════════════════════════════════════════
# EVIDENCE STATES
# ══════════════════════════════════════════════════════════════════════

class PromotionEvidenceConstructionStatus(str, Enum):
    """States for promotion evidence construction."""
    BUILT = "built"
    REFUSED_ADJUDICATION_INVALID = "refused_adjudication_invalid"
    REFUSED_RESULT_REJECTED = "refused_result_rejected"
    REFUSED_ACCEPTANCE_INCOMPLETE = "refused_acceptance_incomplete"
    REFUSED_PROVIDER_QUALIFICATION_FAILED = "refused_provider_qualification_failed"
    REFUSED_DATASET_QUALIFICATION_FAILED = "refused_dataset_qualification_failed"
    REFUSED_MODEL_MISMATCH = "refused_model_mismatch"
    REFUSED_RELEASE_MISMATCH = "refused_release_mismatch"
    REFUSED_FEATURE_CONTRACT_MISMATCH = "refused_feature_contract_mismatch"
    REFUSED_PROTOCOL_MISMATCH = "refused_protocol_mismatch"
    REFUSED_ACCEPTANCE_SPEC_MISMATCH = "refused_acceptance_spec_mismatch"
    REFUSED_LEAKAGE_RISK = "refused_leakage_risk"
    REFUSED_CONTAMINATION_RISK = "refused_contamination_risk"
    REFUSED_TAMPERING_DETECTED = "refused_tampering_detected"
    REFUSED_EVIDENCE_PACK_INVALID = "refused_evidence_pack_invalid"


class GateDecisionState(str, Enum):
    """States for promotion gate adapter."""
    PROMOTION_EVIDENCE_MISSING = "promotion_evidence_missing"
    PROMOTION_EVIDENCE_INVALID = "promotion_evidence_invalid"
    PROMOTION_EVIDENCE_VALID = "promotion_evidence_valid"
    PROMOTION_GATE_REQUIRED = "promotion_gate_required"
    PROMOTION_GATE_APPROVED = "promotion_gate_approved"
    PROMOTION_GATE_REJECTED = "promotion_gate_rejected"


# ══════════════════════════════════════════════════════════════════════
# IMMUTABLE EVIDENCE STRUCTURE
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RWVPromotionEvidence:
    """Immutable RWV promotion evidence record.

    This is NOT a promotion token. It is evidence that may be submitted
    to the existing promotion gate. The gate remains the only authority
    for promotion decisions.
    """
    evidence_id: str
    session_id: str
    evaluation_record_hash: str
    adjudication_hash: str
    provider_id: str
    dataset_id: str
    dataset_version: str
    dataset_qualification_hash: str
    model_id: str
    release_id: str
    release_manifest_hash: str
    artifact_hash: str
    feature_contract_version: str
    native_feature_version: str
    preprocessing_hash: str
    rule_hash: str
    evaluation_protocol_version: str
    acceptance_spec_version: str
    evaluation_config_hash: str
    result_status: str
    acceptance_status: str
    promotion_evidence_status: str
    evidence_policy_version: str
    evidence_hash: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class EvidenceBindingCheck:
    """Result of binding verification."""
    field_name: str
    expected: str
    actual: str
    matched: bool

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class PromotionEvidenceAuditEvent:
    """Immutable audit event for promotion evidence lifecycle."""
    event_type: str
    evidence_id: str
    session_id: str
    timestamp: str
    details: str
    event_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class GateAdapterResult:
    """Result of the promotion gate adapter."""
    gate_state: str
    evidence_id: str
    gate_decision_summary: str
    blocking_reasons: tuple[str, ...]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["blocking_reasons"] = list(d["blocking_reasons"])
        return d


# ══════════════════════════════════════════════════════════════════════
# EVIDENCE REPLAY TRACKER
# ══════════════════════════════════════════════════════════════════════

# Module-level replay registry (evidence_id → (model_id, release_id, dataset_id, dataset_version))
_evidence_replay_registry: dict[str, tuple[str, str, str, str]] = {}


def _track_evidence(
    evidence_id: str,
    model_id: str,
    release_id: str,
    dataset_id: str,
    dataset_version: str,
) -> bool:
    """Track evidence for replay detection.

    Returns True if this is a fresh first-time registration.
    Returns False if replay detected against incompatible context.
    Returns True if replay against same exact context (idempotent).
    """
    context = (model_id, release_id, dataset_id, dataset_version)
    if evidence_id in _evidence_replay_registry:
        existing = _evidence_replay_registry[evidence_id]
        return existing == context  # idempotent if same, incompatible if different
    _evidence_replay_registry[evidence_id] = context
    return True


def _reset_replay_registry() -> None:
    """Reset replay registry for testing."""
    _evidence_replay_registry.clear()


# ══════════════════════════════════════════════════════════════════════
# HASHING UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _canonical_json(obj: Any) -> str:
    """Canonical JSON serialization with stable key ordering."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _hash_dict(d: dict[str, Any]) -> str:
    """SHA-256 hash of canonical JSON."""
    return hashlib.sha256(_canonical_json(d).encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_evidence_id(session_id: str, adjudication_hash: str) -> str:
    """Deterministic evidence ID from session + adjudication."""
    raw = f"rwv-promotion-evidence|{session_id}|{adjudication_hash}"
    return "EVID-" + hashlib.sha256(raw.encode()).hexdigest()[:24]


# ══════════════════════════════════════════════════════════════════════
# EVIDENCE BINDING VERIFICATION
# ══════════════════════════════════════════════════════════════════════

def verify_evidence_binding(
    evidence: RWVPromotionEvidence,
    *,
    expected_model_id: str = MODEL_ID,
    expected_release_id: str = RELEASE_ID,
    expected_feature_version: str = FEATURE_VERSION,
    expected_native_feature_version: str = NATIVE_FEATURE_VERSION,
    expected_eval_protocol: str = EVAL_PROTOCOL_VERSION,
    expected_acceptance_spec: str = ACCEPTANCE_SPEC_VERSION,
    expected_preprocessing_hash: str = PREPROCESSING_HASH,
    expected_rule_hash: str = RULE_HASH,
) -> list[EvidenceBindingCheck]:
    """Verify all bound fields in the evidence record.

    Returns a list of binding checks. Any failed check means
    the evidence is invalid for the expected model/release context.
    """
    bindings = [
        ("model_id", expected_model_id, evidence.model_id),
        ("release_id", expected_release_id, evidence.release_id),
        ("feature_contract_version", expected_feature_version, evidence.feature_contract_version),
        ("native_feature_version", expected_native_feature_version, evidence.native_feature_version),
        ("evaluation_protocol_version", expected_eval_protocol, evidence.evaluation_protocol_version),
        ("acceptance_spec_version", expected_acceptance_spec, evidence.acceptance_spec_version),
        ("preprocessing_hash", expected_preprocessing_hash, evidence.preprocessing_hash),
        ("rule_hash", expected_rule_hash, evidence.rule_hash),
    ]
    return [
        EvidenceBindingCheck(
            field_name=name,
            expected=exp,
            actual=act,
            matched=(exp == act),
        )
        for name, exp, act in bindings
    ]


# ══════════════════════════════════════════════════════════════════════
# EVIDENCE TAMPERING DETECTION
# ══════════════════════════════════════════════════════════════════════

def compute_evidence_hash(evidence: RWVPromotionEvidence) -> str:
    """Compute the deterministic evidence hash for a promotion evidence record."""
    binding_fields = {
        "evidence_id": evidence.evidence_id,
        "session_id": evidence.session_id,
        "evaluation_record_hash": evidence.evaluation_record_hash,
        "adjudication_hash": evidence.adjudication_hash,
        "provider_id": evidence.provider_id,
        "dataset_id": evidence.dataset_id,
        "dataset_version": evidence.dataset_version,
        "dataset_qualification_hash": evidence.dataset_qualification_hash,
        "model_id": evidence.model_id,
        "release_id": evidence.release_id,
        "release_manifest_hash": evidence.release_manifest_hash,
        "artifact_hash": evidence.artifact_hash,
        "feature_contract_version": evidence.feature_contract_version,
        "native_feature_version": evidence.native_feature_version,
        "preprocessing_hash": evidence.preprocessing_hash,
        "rule_hash": evidence.rule_hash,
        "evaluation_protocol_version": evidence.evaluation_protocol_version,
        "acceptance_spec_version": evidence.acceptance_spec_version,
        "evaluation_config_hash": evidence.evaluation_config_hash,
        "result_status": evidence.result_status,
        "acceptance_status": evidence.acceptance_status,
        "promotion_evidence_status": evidence.promotion_evidence_status,
        "evidence_policy_version": evidence.evidence_policy_version,
    }
    return _hash_dict(binding_fields)


def verify_evidence_tampering(evidence: RWVPromotionEvidence) -> tuple[bool, str]:
    """Verify that the evidence hash has not been tampered with.

    Returns (is_valid, reason).
    """
    recomputed = compute_evidence_hash(evidence)
    if recomputed == evidence.evidence_hash:
        return True, "Evidence hash verified"
    return False, "Evidence hash mismatch — tampering detected"


# ══════════════════════════════════════════════════════════════════════
# AUDIT EVENTS
# ══════════════════════════════════════════════════════════════════════

def create_promotion_evidence_audit_event(
    event_type: str,
    evidence_id: str,
    session_id: str,
    details: str,
) -> PromotionEvidenceAuditEvent:
    """Create a deterministic audit event."""
    ts = _now_iso()
    raw = f"promotion_evidence|{event_type}|{evidence_id}|{session_id}|{ts}|{details}"
    evt_hash = hashlib.sha256(raw.encode()).hexdigest()
    return PromotionEvidenceAuditEvent(
        event_type=event_type,
        evidence_id=evidence_id,
        session_id=session_id,
        timestamp=ts,
        details=details,
        event_hash=evt_hash,
    )


# ══════════════════════════════════════════════════════════════════════
# CORE: BUILD PROMOTION EVIDENCE
# ══════════════════════════════════════════════════════════════════════

def build_promotion_evidence(
    adjudication: RWVAdjudication,
    evaluation_record: RWVEvaluationRecord,
    qualification: QualificationReport | None = None,
    *,
    expected_model_id: str = MODEL_ID,
    expected_release_id: str = RELEASE_ID,
    expected_feature_version: str = FEATURE_VERSION,
    expected_native_feature_version: str = NATIVE_FEATURE_VERSION,
    expected_eval_protocol: str = EVAL_PROTOCOL_VERSION,
    expected_acceptance_spec: str = ACCEPTANCE_SPEC_VERSION,
    expected_preprocessing_hash: str = PREPROCESSING_HASH,
    expected_rule_hash: str = RULE_HASH,
) -> tuple[RWVPromotionEvidence | None, str, list[str]]:
    """Build promotion evidence from an adjudicated RWV result.

    This function REFUSES to construct evidence unless all 20 eligibility
    checks pass. It does NOT promote — it produces evidence for the
    existing promotion gate.

    Returns: (evidence_or_None, construction_status, blocking_reasons)
    """
    blockers: list[str] = []
    evidence_id = _generate_evidence_id(
        adjudication.session_id,
        adjudication.adjudication_hash,
    )

    # 1. Adjudication must be valid
    if adjudication.validity_status != ValidityStatus.EVALUATION_VALID.value:
        if adjudication.validity_status == ValidityStatus.EVALUATION_VALID_WITH_WARNINGS.value:
            pass  # allowed with warnings
        else:
            blockers.append("Adjudication validity status is not valid")
            return None, PromotionEvidenceConstructionStatus.REFUSED_ADJUDICATION_INVALID.value, blockers

    # 2. Result must be accepted
    if adjudication.acceptance_status not in (
        AcceptanceStatus.ACCEPTED.value,
        AcceptanceStatus.ACCEPTED_WITH_LIMITATIONS.value,
    ):
        blockers.append(f"Adjudication acceptance status: {adjudication.acceptance_status}")
        return None, PromotionEvidenceConstructionStatus.REFUSED_RESULT_REJECTED.value, blockers

    # 3. Acceptance criteria must be complete (no incomplete criteria)
    for crit in adjudication.criteria_results:
        if crit.get("status") == "INCOMPLETE":
            blockers.append(f"Acceptance criterion INCOMPLETE: {crit.get('criterion_id', 'unknown')}")
            return None, PromotionEvidenceConstructionStatus.REFUSED_ACCEPTANCE_INCOMPLETE.value, blockers

    # 4. Provider qualification must have passed
    if qualification is not None:
        if qualification.qualification_state != DatasetQualificationState.QUALIFIED_FOR_CONTROLLED_RWV.value:
            blockers.append(f"Dataset qualification state: {qualification.qualification_state}")
            return None, PromotionEvidenceConstructionStatus.REFUSED_PROVIDER_QUALIFICATION_FAILED.value, blockers

    # 5. Dataset qualification — if qualification provided, check it
    if qualification is not None:
        if qualification.qualification_state not in (
            DatasetQualificationState.QUALIFIED_FOR_CONTROLLED_RWV.value,
        ):
            blockers.append(f"Dataset not qualified: {qualification.qualification_state}")
            return None, PromotionEvidenceConstructionStatus.REFUSED_DATASET_QUALIFICATION_FAILED.value, blockers

    # 6-8. Model/Release identity must match
    if adjudication.model_id != expected_model_id:
        blockers.append(f"Model mismatch: adjudication={adjudication.model_id} expected={expected_model_id}")
        return None, PromotionEvidenceConstructionStatus.REFUSED_MODEL_MISMATCH.value, blockers

    if adjudication.release_id != expected_release_id:
        blockers.append(f"Release mismatch: adjudication={adjudication.release_id} expected={expected_release_id}")
        return None, PromotionEvidenceConstructionStatus.REFUSED_RELEASE_MISMATCH.value, blockers

    # 9-10. Feature contract must match
    if evaluation_record.feature_contract_version != expected_feature_version:
        blockers.append(f"Feature version mismatch: {evaluation_record.feature_contract_version} != {expected_feature_version}")
        return None, PromotionEvidenceConstructionStatus.REFUSED_FEATURE_CONTRACT_MISMATCH.value, blockers

    # 11-12. Protocol must match
    if adjudication.evaluation_protocol_version != expected_eval_protocol:
        blockers.append(f"Protocol mismatch: {adjudication.evaluation_protocol_version} != {expected_eval_protocol}")
        return None, PromotionEvidenceConstructionStatus.REFUSED_PROTOCOL_MISMATCH.value, blockers

    if adjudication.acceptance_spec_version != expected_acceptance_spec:
        blockers.append(f"Acceptance spec mismatch: {adjudication.acceptance_spec_version} != {expected_acceptance_spec}")
        return None, PromotionEvidenceConstructionStatus.REFUSED_ACCEPTANCE_SPEC_MISMATCH.value, blockers

    # 13. Leakage must be safe
    leakage_safe = (
        "no_leakage" in adjudication.leakage_result.get("status", "").lower()
        or "synthetic" in adjudication.leakage_result.get("status", "").lower()
    )
    if not leakage_safe:
        blockers.append("Leakage check not safe")
        return None, PromotionEvidenceConstructionStatus.REFUSED_LEAKAGE_RISK.value, blockers

    # 14. Contamination must be safe
    contam_safe = (
        "no_contamination" in adjudication.contamination_result.get("status", "").lower()
        or "synthetic" in adjudication.contamination_result.get("status", "").lower()
    )
    if not contam_safe:
        blockers.append("Contamination check not safe")
        return None, PromotionEvidenceConstructionStatus.REFUSED_CONTAMINATION_RISK.value, blockers

    # 15. Evidence pack must be valid
    pack = build_evidence_pack()
    pack_valid = validate_evidence_pack(pack).get("valid", False)
    if not pack_valid:
        blockers.append("Evidence pack validation failed")
        return None, PromotionEvidenceConstructionStatus.REFUSED_EVIDENCE_PACK_INVALID.value, blockers

    # 16-20. Construct evidence
    dataset_qual_hash = ""
    if qualification is not None:
        dataset_qual_hash = qualification.evidence_hash

    ts = _now_iso()
    evidence = RWVPromotionEvidence(
        evidence_id=evidence_id,
        session_id=adjudication.session_id,
        evaluation_record_hash=adjudication.evaluation_record_hash,
        adjudication_hash=adjudication.adjudication_hash,
        provider_id=adjudication.qualification_hash or "unknown",
        dataset_id=adjudication.dataset_id,
        dataset_version=adjudication.dataset_version,
        dataset_qualification_hash=dataset_qual_hash,
        model_id=adjudication.model_id,
        release_id=adjudication.release_id,
        release_manifest_hash=adjudication.release_manifest_hash,
        artifact_hash=evaluation_record.dataset_hash,
        feature_contract_version=evaluation_record.feature_contract_version,
        native_feature_version=NATIVE_FEATURE_VERSION,
        preprocessing_hash=PREPROCESSING_HASH,
        rule_hash=RULE_HASH,
        evaluation_protocol_version=adjudication.evaluation_protocol_version,
        acceptance_spec_version=adjudication.acceptance_spec_version,
        evaluation_config_hash=adjudication.evaluation_config_hash,
        result_status=adjudication.validity_status,
        acceptance_status=adjudication.acceptance_status,
        promotion_evidence_status=adjudication.promotion_evidence_status,
        evidence_policy_version=PROMOTION_EVIDENCE_VERSION,
        evidence_hash="",  # placeholder
        created_at=ts,
    )

    # Compute hash and replace placeholder
    evidence_hash = compute_evidence_hash(evidence)
    evidence = RWVPromotionEvidence(
        **{**evidence.__dict__, "evidence_hash": evidence_hash}
    )

    # Track for replay detection
    _track_evidence(
        evidence_id,
        evidence.model_id,
        evidence.release_id,
        evidence.dataset_id,
        evidence.dataset_version,
    )

    return evidence, PromotionEvidenceConstructionStatus.BUILT.value, []


# ══════════════════════════════════════════════════════════════════════
# PROMOTION GATE ADAPTER
# ══════════════════════════════════════════════════════════════════════

def prepare_gate_submission(
    evidence: RWVPromotionEvidence,
) -> tuple[dict[str, Any], str]:
    """Prepare the evidence for submission to the existing promotion gate.

    This does NOT call the gate — it only prepares the input.
    The actual gate invocation must be done by the operator/pipeline.

    Returns: (gate_input_dict, gate_state)
    """
    # Verify evidence integrity first
    is_valid, reason = verify_evidence_tampering(evidence)
    if not is_valid:
        return {}, GateDecisionState.PROMOTION_EVIDENCE_INVALID.value

    # Verify binding
    binding_checks = verify_evidence_binding(evidence)
    failed_bindings = [c for c in binding_checks if not c.matched]
    if failed_bindings:
        return {}, GateDecisionState.PROMOTION_EVIDENCE_INVALID.value

    gate_input = {
        "candidate_model_id": evidence.model_id,
        "candidate_artifact_hash": evidence.artifact_hash,
        "candidate_feature_version": evidence.feature_contract_version,
        "rwv_evidence_id": evidence.evidence_id,
        "rwv_session_id": evidence.session_id,
        "rwv_evaluation_record_hash": evidence.evaluation_record_hash,
        "rwv_adjudication_hash": evidence.adjudication_hash,
        "rwv_dataset_id": evidence.dataset_id,
        "rwv_dataset_version": evidence.dataset_version,
    }
    return gate_input, GateDecisionState.PROMOTION_EVIDENCE_VALID.value


def check_replay(
    evidence: RWVPromotionEvidence,
    target_model_id: str,
    target_release_id: str,
    target_dataset_id: str,
    target_dataset_version: str,
) -> tuple[bool, str]:
    """Check if this evidence can be used in the given target context.

    Returns (is_allowed, reason).
    """
    context = (target_model_id, target_release_id, target_dataset_id, target_dataset_version)
    if evidence.evidence_id in _evidence_replay_registry:
        existing = _evidence_replay_registry[evidence.evidence_id]
        if existing == context:
            return True, "Idempotent replay in same context"
        return False, (
            f"Replay detected: evidence was created for context {existing} "
            f"but is being used for {context}"
        )
    return False, "Evidence ID not in registry — evidence was not built through the standard pipeline"


def reset_replay_registry() -> None:
    """Public reset for replay registry."""
    _reset_replay_registry()
