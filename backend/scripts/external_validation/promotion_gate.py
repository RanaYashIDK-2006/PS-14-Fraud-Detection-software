"""Promotion gate integration — ensures no automatic promotion after evaluation.

The system must NOT allow external dataset evaluation to automatically
promote a model to production. Instead, all necessary gates must pass.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .metadata_schema import ProvenanceStatus


@dataclass
class PromotionGateResult:
    """Result of the promotion gate check."""
    promotion_eligible: bool = False
    promotion_blocked: bool = True
    criteria_not_defined: bool = False
    gates: list[dict[str, Any]] = field(default_factory=list)
    blocking_gates: list[str] = field(default_factory=list)
    status_message: str = "PROMOTION_BLOCKED"

    def to_dict(self) -> dict:
        return {
            "promotion_eligible": self.promotion_eligible,
            "promotion_blocked": self.promotion_blocked,
            "criteria_not_defined": self.criteria_not_defined,
            "gates": self.gates,
            "blocking_gates": self.blocking_gates,
            "status_message": self.status_message,
        }


# Required gates for promotion (all must PASS)
REQUIRED_PROMOTION_GATES = [
    "PROVENANCE_VERIFIED",
    "FEATURE_CAUSALITY_VERIFIED",
    "DATASET_INDEPENDENT",
    "LABEL_VALIDATED",
    "SCHEMA_COMPATIBLE",
    "FROZEN_MODEL_EVALUATED",
    "REQUIRED_PERFORMANCE_THRESHOLD_MET",
    "SECURITY_REQUIREMENTS_MET",
]


def check_promotion_gates(
    provenance_verified: bool = False,
    causality_verified: bool = False,
    dataset_independent: bool = False,
    label_validated: bool = False,
    schema_compatible: bool = False,
    frozen_model_evaluated: bool = False,
    performance_threshold_met: bool = False,
    security_requirements_met: bool = False,
) -> PromotionGateResult:
    """Check all promotion gates and determine if promotion is eligible.

    Args:
        provenance_verified: Whether dataset provenance is verified.
        causality_verified: Whether feature causality is verified.
        dataset_independent: Whether dataset is independent from training data.
        label_validated: Whether labels are validated.
        schema_compatible: Whether schema is compatible.
        frozen_model_evaluated: Whether frozen model evaluation completed.
        performance_threshold_met: Whether performance meets required threshold.
        security_requirements_met: Whether security requirements are met.

    Returns:
        PromotionGateResult with gate status.
    """
    result = PromotionGateResult()

    gate_results = [
        ("PROVENANCE_VERIFIED", provenance_verified),
        ("FEATURE_CAUSALITY_VERIFIED", causality_verified),
        ("DATASET_INDEPENDENT", dataset_independent),
        ("LABEL_VALIDATED", label_validated),
        ("SCHEMA_COMPATIBLE", schema_compatible),
        ("FROZEN_MODEL_EVALUATED", frozen_model_evaluated),
        ("REQUIRED_PERFORMANCE_THRESHOLD_MET", performance_threshold_met),
        ("SECURITY_REQUIREMENTS_MET", security_requirements_met),
    ]

    for gate_name, gate_status in gate_results:
        result.gates.append({
            "gate": gate_name,
            "status": "PASS" if gate_status else "FAIL",
        })
        if not gate_status:
            result.blocking_gates.append(gate_name)

    # Determine promotion eligibility
    if len(result.blocking_gates) == 0:
        result.promotion_eligible = True
        result.promotion_blocked = False
        result.status_message = "PROMOTION_ELIGIBLE — all gates pass"
    else:
        result.promotion_eligible = False
        result.promotion_blocked = True
        result.status_message = f"PROMOTION_BLOCKED — {len(result.blocking_gates)} gate(s) failed"

    return result


def check_promotion_from_report(report: dict) -> PromotionGateResult:
    """Check promotion gates from an audit report dictionary.

    This is a convenience function that extracts gate results from a report.

    Args:
        report: Audit report as dictionary.

    Returns:
        PromotionGateResult.
    """
    # Extract eligibility status
    eligibility = report.get("eligibility", {})
    elig_status = eligibility.get("overall_status", "UNKNOWN")

    # Extract causality
    causality = report.get("causality", {})
    causality_clear = causality.get("all_clear", False)

    # Extract independence
    independence = report.get("independence", {})
    is_independent = independence.get("is_independent", False)

    # Extract label validation
    # (not directly in report — infer from eligibility)
    label_valid = elig_status in ("ELIGIBLE", "CONDITIONALLY_ELIGIBLE")

    # Extract schema
    schema = report.get("schema", {})
    schema_compatible = schema.get("schema_compatible", False)

    # Extract model evaluation
    model = report.get("model", {})
    eval_complete = bool(model.get("model_hash"))

    # Extract performance
    performance = report.get("performance", {})
    # No default threshold — report PROMOTION_CRITERIA_NOT_DEFINED if not set

    return check_promotion_gates(
        provenance_verified=(elig_status == "ELIGIBLE"),
        causality_verified=causality_clear,
        dataset_independent=is_independent,
        label_validated=label_valid,
        schema_compatible=schema_compatible,
        frozen_model_evaluated=eval_complete,
        performance_threshold_met=False,  # No threshold defined
        security_requirements_met=False,  # No security gate in this pipeline
    )
