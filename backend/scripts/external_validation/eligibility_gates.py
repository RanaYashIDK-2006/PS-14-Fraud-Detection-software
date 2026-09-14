"""Dataset eligibility gates — deterministic classifier for external datasets.

Classifies a dataset into states: ELIGIBLE, CONDITIONALLY_ELIGIBLE, INELIGIBLE,
UNKNOWN, BLOCKED. Critical unknown fields prevent ELIGIBLE classification.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .metadata_schema import (
    ExternalDatasetMetadata,
    DatasetEligibility,
    GateStatus,
    ProvenanceStatus,
)


@dataclass
class GateResult:
    """Result of a single eligibility gate."""
    gate_name: str
    status: str  # GateStatus value
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EligibilityReport:
    """Complete eligibility report for a dataset."""
    overall_status: str  # DatasetEligibility value
    gates: list[GateResult] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status,
            "gates": [
                {"gate": g.gate_name, "status": g.status, "reason": g.reason}
                for g in self.gates
            ],
            "blocked_reasons": self.blocked_reasons,
        }


def check_provenance(meta: ExternalDatasetMetadata) -> GateResult:
    """Gate A: Provenance verification."""
    status = meta.provenance_status
    if status == ProvenanceStatus.VERIFIED.value:
        return GateResult("provenance", GateStatus.PASS.value, "Provenance verified")
    elif status == ProvenanceStatus.UNKNOWN.value:
        return GateResult("provenance", GateStatus.UNKNOWN.value, "Provenance status is UNKNOWN — cannot proceed")
    elif status in (
        ProvenanceStatus.SUSPECTED_SYNTHETIC.value,
        ProvenanceStatus.CONFIRMED_SYNTHETIC.value,
    ):
        return GateResult("provenance", GateStatus.FAIL.value, f"Dataset is {status}")
    else:
        return GateResult("provenance", GateStatus.FAIL.value, f"Provenance: {status}")


def check_license(meta: ExternalDatasetMetadata) -> GateResult:
    """Gate B: License/usage permission."""
    if not meta.license:
        return GateResult("license", GateStatus.UNKNOWN.value, "No license information provided")
    license_lower = meta.license.lower()
    if any(w in license_lower for w in ["research", "academic", "mit", "apache", "cc0", "public domain", "open"]):
        return GateResult("license", GateStatus.PASS.value, f"License: {meta.license}")
    if any(w in license_lower for w in ["proprietary", "confidential", "restricted", "nda"]):
        return GateResult("license", GateStatus.FAIL.value, f"Restrictive license: {meta.license}")
    return GateResult("license", GateStatus.UNKNOWN.value, f"License unclear: {meta.license}")


def check_fraud_label_definition(meta: ExternalDatasetMetadata) -> GateResult:
    """Gate C: Fraud-label definition."""
    if not meta.label_definition:
        return GateResult("label_definition", GateStatus.FAIL.value, "No label definition provided")
    if meta.fraud_label_timing in ("", "unknown"):
        return GateResult(
            "label_definition",
            GateStatus.UNKNOWN.value,
            "Label timing unknown — fraud labels may be post-event",
        )
    return GateResult("label_definition", GateStatus.PASS.value, f"Label: {meta.label_definition}")


def check_feature_provenance(meta: ExternalDatasetMetadata) -> GateResult:
    """Gate D: Feature provenance — are features available at decision time?"""
    if not meta.feature_definitions:
        return GateResult(
            "feature_provenance",
            GateStatus.UNKNOWN.value,
            "No feature provenance metadata provided",
        )
    suspicious = [f for f in meta.feature_definitions if f.post_event_possible]
    if suspicious:
        names = [f.feature_name for f in suspicious]
        return GateResult(
            "feature_provenance",
            GateStatus.FAIL.value,
            f"Post-event features detected: {names}",
            {"suspicious_features": names},
        )
    unknown = [f for f in meta.feature_definitions if f.availability_time == "unknown"]
    if unknown:
        names = [f.feature_name for f in unknown]
        return GateResult(
            "feature_provenance",
            GateStatus.UNKNOWN.value,
            f"Feature availability unknown for: {names}",
            {"unknown_features": names},
        )
    return GateResult("feature_provenance", GateStatus.PASS.value, "All features available at decision time")


def check_temporal_validity(meta: ExternalDatasetMetadata) -> GateResult:
    """Gate E: Temporal validity — dataset covers a meaningful time range."""
    if not meta.date_range_start or not meta.date_range_end:
        return GateResult("temporal_validity", GateStatus.UNKNOWN.value, "Date range not specified")
    return GateResult("temporal_validity", GateStatus.PASS.value, f"Range: {meta.date_range_start} to {meta.date_range_end}")


def check_feature_availability(meta: ExternalDatasetMetadata) -> GateResult:
    """Gate F: Decision-time feature availability."""
    if meta.feature_availability_at_decision_time in ("", "unknown"):
        return GateResult(
            "feature_availability",
            GateStatus.UNKNOWN.value,
            "Feature availability at decision time not specified",
        )
    if meta.feature_availability_at_decision_time == "all_available":
        return GateResult("feature_availability", GateStatus.PASS.value, "All features available at decision time")
    return GateResult(
        "feature_availability",
        GateStatus.UNKNOWN.value,
        f"Feature availability: {meta.feature_availability_at_decision_time}",
    )


def check_duplicate_contamination(meta: ExternalDatasetMetadata) -> GateResult:
    """Gate G: Duplicate contamination."""
    if meta.duplicate_pct > 0.05:
        return GateResult(
            "duplicate_contamination",
            GateStatus.FAIL.value,
            f"High duplicate rate: {meta.duplicate_pct:.1%}",
        )
    return GateResult("duplicate_contamination", GateStatus.PASS.value, "Duplicate rate acceptable")


def check_class_label_integrity(meta: ExternalDatasetMetadata) -> GateResult:
    """Gate L: Class-label integrity."""
    if meta.num_rows <= 0:
        return GateResult("class_label_integrity", GateStatus.FAIL.value, "No rows in dataset")
    if meta.num_positive <= 0:
        return GateResult("class_label_integrity", GateStatus.FAIL.value, "No positive (fraud) cases")
    if meta.num_negative <= 0:
        return GateResult("class_label_integrity", GateStatus.FAIL.value, "No negative (legitimate) cases")
    prevalence = meta.num_positive / meta.num_rows
    if prevalence < 0.0001:
        return GateResult(
            "class_label_integrity",
            GateStatus.UNKNOWN.value,
            f"Very low prevalence: {prevalence:.4%} — statistical instability likely",
        )
    return GateResult(
        "class_label_integrity",
        GateStatus.PASS.value,
        f"{meta.num_positive}/{meta.num_rows} positive ({prevalence:.2%})",
    )


def check_dataset_independence(meta: ExternalDatasetMetadata) -> GateResult:
    """Gate M: Dataset independence from training data.

    This gate requires a separate independence check (see independence_check.py).
    Here we only check if the flag indicates the dataset is derived.
    """
    if meta.is_derived:
        return GateResult(
            "dataset_independence",
            GateStatus.FAIL.value,
            "Dataset is derived from PS-14 data — not independent",
        )
    return GateResult("dataset_independence", GateStatus.UNKNOWN.value, "Independence not verified — run independence_check")


def run_all_gates(meta: ExternalDatasetMetadata) -> EligibilityReport:
    """Run all eligibility gates and produce a report.

    The overall status is:
    - ELIGIBLE: all gates PASS
    - CONDITIONALLY_ELIGIBLE: some gates UNKNOWN but none FAIL
    - INELIGIBLE: any gate FAIL
    - UNKNOWN: all gates UNKNOWN
    - BLOCKED: provenance is CONFIRMED_SYNTHETIC or dataset is derived
    """
    gates = [
        check_provenance(meta),
        check_license(meta),
        check_fraud_label_definition(meta),
        check_feature_provenance(meta),
        check_temporal_validity(meta),
        check_feature_availability(meta),
        check_duplicate_contamination(meta),
        check_class_label_integrity(meta),
        check_dataset_independence(meta),
    ]

    blocked_reasons = []
    has_fail = False
    has_unknown = False
    has_pass = False

    for gate in gates:
        if gate.status == GateStatus.FAIL.value:
            has_fail = True
            blocked_reasons.append(f"{gate.gate_name}: {gate.reason}")
        elif gate.status == GateStatus.UNKNOWN.value:
            has_unknown = True
        elif gate.status == GateStatus.PASS.value:
            has_pass = True

    # Determine overall status
    if meta.provenance_status in (
        ProvenanceStatus.CONFIRMED_SYNTHETIC.value,
        ProvenanceStatus.SUSPECTED_SYNTHETIC.value,
    ):
        overall = DatasetEligibility.BLOCKED.value
        blocked_reasons.append("Dataset is synthetic — not eligible for real-world validation")
    elif meta.is_derived:
        overall = DatasetEligibility.BLOCKED.value
        blocked_reasons.append("Dataset is derived from PS-14 data — not independent")
    elif has_fail:
        overall = DatasetEligibility.INELIGIBLE.value
    elif has_unknown and not has_fail:
        overall = DatasetEligibility.CONDITIONALLY_ELIGIBLE.value
    elif has_pass and not has_unknown and not has_fail:
        overall = DatasetEligibility.ELIGIBLE.value
    else:
        overall = DatasetEligibility.UNKNOWN.value

    return EligibilityReport(
        overall_status=overall,
        gates=gates,
        blocked_reasons=blocked_reasons,
    )
