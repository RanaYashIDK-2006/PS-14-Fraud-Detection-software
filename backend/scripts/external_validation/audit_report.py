"""Audit report generation — produces reproducible evaluation reports.

Contains dataset info, eligibility, schema, causality, independence,
distribution, model, performance, and final decision.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metadata_schema import ExternalDatasetMetadata
from .eligibility_gates import EligibilityReport
from .causality_audit import CausalityReport
from .schema_mapper import SchemaMappingReport
from .independence_check import IndependenceReport
from .label_validation import LabelValidationReport
from .frozen_evaluator import EvaluationConfig, FrozenEvaluationResult
from .metrics import EvaluationMetrics
from .distribution_shift import DistributionShiftReport


@dataclass
class AuditReport:
    """Complete audit report for an external dataset evaluation."""
    # Dataset
    dataset_name: str = ""
    dataset_source: str = ""
    dataset_hash: str = ""
    num_rows: int = 0
    num_positive: int = 0
    num_negative: int = 0
    date_range: str = ""
    geography: str = ""
    provenance_status: str = ""

    # Eligibility
    eligibility: dict = field(default_factory=dict)

    # Schema
    schema: dict = field(default_factory=dict)

    # Causality
    causality: dict = field(default_factory=dict)

    # Independence
    independence: dict = field(default_factory=dict)

    # Distribution
    distribution: dict = field(default_factory=dict)

    # Model
    model: dict = field(default_factory=dict)

    # Performance
    performance: dict = field(default_factory=dict)

    # Final decision
    final_decision: str = "NOT_READY"
    decision_reasons: list[str] = field(default_factory=list)

    # Metadata
    report_version: str = "1.0"
    generated_at: str = ""
    pipeline_version: str = "phase38"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> None:
        """Save the report to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")


def generate_audit_report(
    meta: ExternalDatasetMetadata,
    eligibility: EligibilityReport,
    causality: CausalityReport,
    schema: SchemaMappingReport,
    independence: IndependenceReport,
    label_validation: LabelValidationReport,
    distribution: DistributionShiftReport,
    eval_config: EvaluationConfig,
    eval_result: FrozenEvaluationResult,
    metrics: EvaluationMetrics,
    dataset_hash: str = "",
) -> AuditReport:
    """Generate a complete audit report from all pipeline components.

    Args:
        meta: External dataset metadata.
        eligibility: Eligibility gate results.
        causality: Causality audit results.
        schema: Schema mapping results.
        independence: Independence check results.
        label_validation: Label validation results.
        distribution: Distribution shift analysis results.
        eval_config: Evaluation configuration.
        eval_result: Frozen evaluation results.
        metrics: Computed metrics.
        dataset_hash: Hash of the dataset file.

    Returns:
        Complete AuditReport.
    """
    report = AuditReport()

    # Dataset section
    report.dataset_name = meta.name
    report.dataset_source = meta.source
    report.dataset_hash = dataset_hash
    report.num_rows = meta.num_rows
    report.num_positive = meta.num_positive
    report.num_negative = meta.num_negative
    report.date_range = f"{meta.date_range_start} to {meta.date_range_end}"
    report.geography = meta.geography
    report.provenance_status = meta.provenance_status

    # Sub-reports
    report.eligibility = eligibility.to_dict()
    report.schema = schema.to_dict()
    report.causality = causality.to_dict()
    report.independence = independence.to_dict()
    report.distribution = distribution.to_dict()
    report.model = eval_config.to_dict()
    report.performance = metrics.to_dict()

    # Determine final decision
    decision_reasons = []

    # Check eligibility
    if eligibility.overall_status in ("INELIGIBLE", "BLOCKED"):
        report.final_decision = "INELIGIBLE"
        decision_reasons.append(f"Dataset eligibility: {eligibility.overall_status}")
        for reason in eligibility.blocked_reasons:
            decision_reasons.append(reason)
    elif eligibility.overall_status == "UNKNOWN":
        report.final_decision = "NOT_READY"
        decision_reasons.append("Dataset eligibility: UNKNOWN — cannot proceed")
    elif eligibility.overall_status == "CONDITIONALLY_ELIGIBLE":
        report.final_decision = "CONDITIONALLY_ELIGIBLE"
        decision_reasons.append("Dataset is conditionally eligible — some gates need resolution")
    elif eligibility.overall_status == "ELIGIBLE":
        # Check independence
        if not independence.is_independent:
            report.final_decision = "INELIGIBLE"
            decision_reasons.append("Dataset overlaps with training data — not independent")
        elif not label_validation.label_valid:
            report.final_decision = "NOT_READY"
            decision_reasons.append("Label validation failed")
        elif not schema.schema_compatible:
            report.final_decision = "NOT_READY"
            decision_reasons.append("Schema incompatible — required features missing or wrong type")
        elif not eval_result.evaluation_complete:
            report.final_decision = "NOT_READY"
            decision_reasons.append("Frozen model evaluation did not complete")
        else:
            # All gates pass — evaluation complete but promotion still blocked
            report.final_decision = "EVALUATED"
            decision_reasons.append("Evaluation complete — promotion requires additional gates")

    # Always note that promotion is blocked
    if report.final_decision in ("EVALUATED", "CONDITIONALLY_ELIGIBLE"):
        decision_reasons.append("REAL_WORLD_VALIDATION = BLOCKED — promotion requires all gates to pass")
        decision_reasons.append("PROMOTION = BLOCKED — see promotion_gate.py for required criteria")

    report.decision_reasons = decision_reasons
    report.generated_at = datetime.now(timezone.utc).isoformat()

    return report
