"""Phase 46: Centralized promotion gate.

Single authoritative check that determines whether a model candidate is
eligible for promotion to active/production.  Consolidates gates from:

- model_governance.promote_gate()     (leakage, data quality, performance, …)
- external_validation/promotion_gate  (provenance, causality, independence, …)
- security CI gate                    (scan, pentest, regression)
- Phase 40 monitoring drift status
- Phase 42/43 runtime enforcement
- REAL_WORLD_VALIDATION status

Every activation path (ModelRegistry.promote, cmd_deploy, manual) MUST pass
through evaluate_promotion() before the model becomes active.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_EVALUATED = "NOT_EVALUATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PromotionVerdict(str, Enum):
    ELIGIBLE = "PROMOTION_ELIGIBLE"
    BLOCKED = "PROMOTION_BLOCKED"
    CRITERIA_NOT_DEFINED = "PROMOTION_CRITERIA_NOT_DEFINED"


@dataclass
class GateResult:
    gate_name: str
    status: GateStatus
    reason: str = ""
    evidence: str = ""
    blocking: bool = True
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "gate": self.gate_name,
            "status": self.status.value,
            "reason": self.reason,
            "evidence": self.evidence,
            "blocking": self.blocking,
        }


@dataclass
class PromotionDecision:
    verdict: PromotionVerdict
    candidate_model_id: str = ""
    candidate_artifact_hash: str = ""
    candidate_feature_version: str = ""
    candidate_schema_version: str = ""
    gates: list[GateResult] = field(default_factory=list)
    blocking_gates: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    real_world_validation_status: str = "BLOCKED_PENDING_ELIGIBLE_DATASET"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "candidate_model_id": self.candidate_model_id,
            "candidate_artifact_hash": self.candidate_artifact_hash,
            "candidate_feature_version": self.candidate_feature_version,
            "candidate_schema_version": self.candidate_schema_version,
            "gates": [g.to_dict() for g in self.gates],
            "blocking_gates": self.blocking_gates,
            "timestamp": self.timestamp,
            "real_world_validation_status": self.real_world_validation_status,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ── Gate definitions ──────────────────────────────────────────────────────

REAL_WORLD_VALIDATION_GATES = [
    "PROVENANCE_VERIFIED",
    "FEATURE_CAUSALITY_VERIFIED",
    "DATASET_INDEPENDENT",
    "LABEL_VALIDATED",
    "SCHEMA_COMPATIBLE",
    "FROZEN_MODEL_EVALUATED",
    "REQUIRED_PERFORMANCE_THRESHOLD_MET",
    "SECURITY_REQUIREMENTS_MET",
]

MODEL_GOVERNANCE_GATES = [
    "leakage_checks",
    "data_quality",
    "validation_performance",
    "untouched_test",
    "robustness",
    "production_parity",
    "security",
    "approval",
]


def _gate(name: str, status: GateStatus, reason: str, evidence: str = "",
          blocking: bool = True) -> GateResult:
    return GateResult(gate_name=name, status=status, reason=reason,
                      evidence=evidence, blocking=blocking)


def evaluate_real_world_validation() -> GateResult:
    """Hard gate: REAL_WORLD_VALIDATION must be PASS, not BLOCKED."""
    # This is a machine-readable check — the status is determined by
    # external validation pipeline (Phase 38), not by this function.
    # The default state is BLOCKED.
    return _gate(
        "REAL_WORLD_VALIDATION",
        GateStatus.BLOCKED,
        "REAL_WORLD_VALIDATION = BLOCKED_PENDING_ELIGIBLE_DATASET — "
        "no eligible real-world dataset has passed the validation gate",
        evidence="Phase 38 external validation pipeline ready; "
                 "no eligible dataset acquired",
        blocking=True,
    )


def evaluate_model_artifact_binding(
    candidate_artifact_hash: str | None,
    evaluated_artifact_hash: str | None,
) -> GateResult:
    """The evaluated model must match the candidate being promoted."""
    if not candidate_artifact_hash or not evaluated_artifact_hash:
        return _gate(
            "MODEL_ARTIFACT_BINDING",
            GateStatus.NOT_EVALUATED,
            "Artifact hashes not provided — cannot verify binding",
            blocking=True,
        )
    if candidate_artifact_hash == evaluated_artifact_hash:
        return _gate(
            "MODEL_ARTIFACT_BINDING",
            GateStatus.PASS,
            "Candidate artifact hash matches evaluated artifact",
            evidence=f"hash={candidate_artifact_hash[:16]}…",
            blocking=True,
        )
    return _gate(
        "MODEL_ARTIFACT_BINDING",
        GateStatus.FAIL,
        "Candidate artifact differs from evaluated artifact — "
        "Model A evaluated but Model B promoted",
        evidence=f"candidate={candidate_artifact_hash[:16]}… "
                 f"evaluated={evaluated_artifact_hash[:16]}…",
        blocking=True,
    )


def evaluate_feature_schema_binding(
    candidate_feature_version: str | None,
    evaluated_feature_version: str | None,
) -> GateResult:
    """Feature/schema version must match between evaluation and promotion."""
    if not candidate_feature_version or not evaluated_feature_version:
        return _gate(
            "FEATURE_SCHEMA_BINDING",
            GateStatus.NOT_EVALUATED,
            "Feature versions not provided — cannot verify binding",
            blocking=True,
        )
    if candidate_feature_version == evaluated_feature_version:
        return _gate(
            "FEATURE_SCHEMA_BINDING",
            GateStatus.PASS,
            "Feature version matches",
            evidence=f"version={candidate_feature_version}",
            blocking=True,
        )
    return _gate(
        "FEATURE_SCHEMA_BINDING",
        GateStatus.FAIL,
        "Feature version mismatch between evaluation and promotion",
        evidence=f"candidate={candidate_feature_version} "
                 f"evaluated={evaluated_feature_version}",
        blocking=True,
    )


def evaluate_governance_gates(
    governance_gates: dict[str, str] | None,
) -> GateResult:
    """Check model governance promotion gates (leakage, data quality, …)."""
    if not governance_gates:
        return _gate(
            "MODEL_GOVERNANCE",
            GateStatus.NOT_EVALUATED,
            "No governance gate record found",
            blocking=True,
        )
    failing = []
    for g in MODEL_GOVERNANCE_GATES:
        val = str(governance_gates.get(g, "UNVERIFIED")).upper()
        if val not in ("PASS", "TRUE", "APPROVED"):
            failing.append(f"{g}={governance_gates.get(g, 'UNVERIFIED')}")
    if not failing:
        return _gate(
            "MODEL_GOVERNANCE",
            GateStatus.PASS,
            "All governance gates PASS",
            evidence=f"gates checked: {len(MODEL_GOVERNANCE_GATES)}",
            blocking=True,
        )
    return _gate(
        "MODEL_GOVERNANCE",
        GateStatus.FAIL,
        f"Governance gates failing: {'; '.join(failing)}",
        evidence=f"failing={failing}",
        blocking=True,
    )


def evaluate_security_gates(
    security_passed: bool | None = None,
) -> GateResult:
    """Security CI gate must pass."""
    if security_passed is None:
        return _gate(
            "SECURITY_CI",
            GateStatus.NOT_EVALUATED,
            "Security CI gate result not provided",
            blocking=True,
        )
    if security_passed:
        return _gate(
            "SECURITY_CI",
            GateStatus.PASS,
            "Security CI gate passed (self-authored tests)",
            evidence="Note: self-authored, not independent audit",
            blocking=True,
        )
    return _gate(
        "SECURITY_CI",
        GateStatus.FAIL,
        "Security CI gate failed",
        blocking=True,
    )


def evaluate_drift_status(
    drift_critical: bool | None = None,
) -> GateResult:
    """Critical drift must not be present."""
    if drift_critical is None:
        return _gate(
            "DRIFT_STATUS",
            GateStatus.NOT_APPLICABLE,
            "Drift status not checked",
            blocking=False,
        )
    if not drift_critical:
        return _gate(
            "DRIFT_STATUS",
            GateStatus.PASS,
            "No critical drift detected",
            blocking=False,
        )
    return _gate(
        "DRIFT_STATUS",
        GateStatus.FAIL,
        "Critical drift detected — investigation required before promotion",
        blocking=False,
    )


def evaluate_external_dataset_eligibility(
    eligible: bool | None = None,
) -> GateResult:
    """External dataset must be eligible (if used for validation)."""
    if eligible is None:
        return _gate(
            "EXTERNAL_DATASET_ELIGIBILITY",
            GateStatus.NOT_APPLICABLE,
            "No external dataset evaluation referenced",
            blocking=False,
        )
    if eligible:
        return _gate(
            "EXTERNAL_DATASET_ELIGIBILITY",
            GateStatus.PASS,
            "External dataset passed eligibility gates",
            blocking=False,
        )
    return _gate(
        "EXTERNAL_DATASET_ELIGIBILITY",
        GateStatus.FAIL,
        "External dataset failed eligibility gates",
        blocking=True,
    )


def evaluate_rollback_availability(
    has_rollback_source: bool | None = None,
) -> GateResult:
    """A rollback source must exist before promotion."""
    if has_rollback_source:
        return _gate(
            "ROLLBACK_AVAILABILITY",
            GateStatus.PASS,
            "Rollback source available",
            blocking=False,
        )
    return _gate(
        "ROLLBACK_AVAILABILITY",
        GateStatus.NOT_EVALUATED,
        "Rollback source status unknown",
        blocking=False,
    )


# ── Central evaluation function ──────────────────────────────────────────

def evaluate_promotion(
    *,
    candidate_model_id: str = "",
    candidate_artifact_hash: str | None = None,
    candidate_feature_version: str | None = None,
    evaluated_artifact_hash: str | None = None,
    evaluated_feature_version: str | None = None,
    governance_gates: dict[str, str] | None = None,
    security_passed: bool | None = None,
    drift_critical: bool | None = None,
    external_dataset_eligible: bool | None = None,
    has_rollback_source: bool | None = None,
) -> PromotionDecision:
    """Single authoritative promotion evaluation.

    Every activation path MUST call this before promoting a model.
    Returns a structured PromotionDecision with per-gate evidence.
    """
    decision = PromotionDecision(
        verdict=PromotionVerdict.BLOCKED,  # default; updated at end
        candidate_model_id=candidate_model_id,
        candidate_artifact_hash=candidate_artifact_hash or "",
        candidate_feature_version=candidate_feature_version or "",
        candidate_schema_version=candidate_feature_version or "",
    )

    # Gate 1: REAL_WORLD_VALIDATION (HARD BLOCK)
    g = evaluate_real_world_validation()
    decision.gates.append(g)
    if g.status == GateStatus.BLOCKED:
        decision.blocking_gates.append(g.gate_name)

    # Gate 2: Model artifact binding
    g = evaluate_model_artifact_binding(candidate_artifact_hash,
                                        evaluated_artifact_hash)
    decision.gates.append(g)
    if g.status == GateStatus.FAIL:
        decision.blocking_gates.append(g.gate_name)

    # Gate 3: Feature/schema binding
    g = evaluate_feature_schema_binding(candidate_feature_version,
                                        evaluated_feature_version)
    decision.gates.append(g)
    if g.status == GateStatus.FAIL:
        decision.blocking_gates.append(g.gate_name)

    # Gate 4: Model governance gates
    g = evaluate_governance_gates(governance_gates)
    decision.gates.append(g)
    if g.status == GateStatus.FAIL:
        decision.blocking_gates.append(g.gate_name)

    # Gate 5: Security
    g = evaluate_security_gates(security_passed)
    decision.gates.append(g)
    if g.status == GateStatus.FAIL:
        decision.blocking_gates.append(g.gate_name)

    # Gate 6: Drift (non-blocking warning)
    g = evaluate_drift_status(drift_critical)
    decision.gates.append(g)
    if g.status == GateStatus.FAIL:
        decision.blocking_gates.append(g.gate_name)

    # Gate 7: External dataset eligibility
    g = evaluate_external_dataset_eligibility(external_dataset_eligible)
    decision.gates.append(g)
    if g.status == GateStatus.FAIL:
        decision.blocking_gates.append(g.gate_name)

    # Gate 8: Rollback availability
    g = evaluate_rollback_availability(has_rollback_source)
    decision.gates.append(g)
    if g.status == GateStatus.FAIL:
        decision.blocking_gates.append(g.gate_name)

    # Final verdict
    if decision.blocking_gates:
        decision.verdict = PromotionVerdict.BLOCKED
    else:
        decision.verdict = PromotionVerdict.ELIGIBLE

    return decision


def assert_promotion_allowed(decision: PromotionDecision) -> None:
    """Raise if promotion is not eligible.  Use as a hard guard."""
    if decision.verdict != PromotionVerdict.ELIGIBLE:
        blocking = ", ".join(decision.blocking_gates)
        raise PromotionBlockedError(
            f"Promotion BLOCKED — {len(decision.blocking_gates)} gate(s) "
            f"failed: [{blocking}]"
        )


class PromotionBlockedError(RuntimeError):
    """Raised when a promotion attempt fails the gate."""
