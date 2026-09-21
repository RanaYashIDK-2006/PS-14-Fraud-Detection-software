"""Phase 102: Production-Readiness Reconciliation Report.

Deterministic audit report for Phase 102 reconciliation.
STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _stable_hash(d: Any) -> str:
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stable_json(d: Any) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class Phase102Report:
    """Deterministic Phase 102 readiness report."""
    report_id: str
    status: str
    total_invariants: int
    pass_count: int
    fail_count: int
    warn_count: int
    unknown_count: int
    critical_findings: tuple[str, ...]
    high_findings: tuple[str, ...]
    medium_findings: tuple[str, ...]
    low_findings: tuple[str, ...]
    informational_findings: tuple[str, ...]
    unresolved_findings: tuple[str, ...]
    claim_matrix_summary: str
    documentation_issues: tuple[str, ...]
    model_release_consistent: bool
    feature_contract_consistent: bool
    dataset_inventory_consistent: bool
    rwv_state_consistent: bool
    promotion_boundary_consistent: bool
    runtime_attestation_consistent: bool
    evidence_lineage_consistent: bool
    report_hash: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def generate_phase102_report(reconciliation_report: Any) -> Phase102Report:
    """Generate a deterministic Phase 102 readiness report."""
    invariants = reconciliation_report.invariants

    critical = []
    high = []
    medium = []
    low = []
    informational = []

    for inv in invariants:
        if inv["verdict"] == "fail":
            sev = inv.get("severity", "medium")
            entry = f"{inv['invariant_id']}: {inv['finding']}"
            if sev == "critical":
                critical.append(entry)
            elif sev == "high":
                high.append(entry)
            elif sev == "medium":
                medium.append(entry)
            else:
                low.append(entry)
        elif inv["verdict"] == "warn":
            informational.append(f"{inv['invariant_id']}: {inv['finding']}")

    state_checks = {
        "state_consistency", "rwv_consistency",
    }
    identity_checks = {"identity_consistency", "threshold_consistency"}
    feature_checks = {"feature_consistency", "feature_safety"}
    dataset_checks = {"dataset_consistency"}
    promotion_checks = {"promotion_separation", "promotion_safety", "gate_safety"}
    attestation_checks = {"model_safety", "temporal_safety"}
    evidence_checks = {"evidence_consistency", "ledger_consistency",
                        "reproducibility_consistency", "immutability",
                        "tamper_safety", "replay_safety", "binding_safety"}

    def check_pass(categories: set) -> bool:
        for inv in invariants:
            if inv.get("category", "") in categories and inv["verdict"] == "fail":
                return False
        return True

    model_release = check_pass(identity_checks)
    feature_contract = check_pass(feature_checks)
    dataset_inventory = check_pass(dataset_checks)
    rwv_state = check_pass(state_checks)
    promotion_boundary = check_pass(promotion_checks)
    runtime_att = check_pass(attestation_checks)
    evidence_lineage = check_pass(evidence_checks)

    unresolved = []
    for inv in invariants:
        if inv["verdict"] == "fail":
            unresolved.append(f"{inv['invariant_id']}: {inv['finding']} — {inv.get('remediation', 'N/A')}")

    report = Phase102Report(
        report_id=f"PHASE102-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        status="COMPLETE" if reconciliation_report.fail_count == 0 else "FINDINGS",
        total_invariants=reconciliation_report.total_invariants,
        pass_count=reconciliation_report.pass_count,
        fail_count=reconciliation_report.fail_count,
        warn_count=reconciliation_report.warn_count,
        unknown_count=reconciliation_report.unknown_count,
        critical_findings=tuple(critical),
        high_findings=tuple(high),
        medium_findings=tuple(medium),
        low_findings=tuple(low),
        informational_findings=tuple(informational),
        unresolved_findings=tuple(unresolved),
        claim_matrix_summary=f"{reconciliation_report.total_invariants} invariants across {len(set(i.get('category', '') for i in invariants))} categories",
        documentation_issues=tuple(),
        model_release_consistent=model_release,
        feature_contract_consistent=feature_contract,
        dataset_inventory_consistent=dataset_inventory,
        rwv_state_consistent=rwv_state,
        promotion_boundary_consistent=promotion_boundary,
        runtime_attestation_consistent=runtime_att,
        evidence_lineage_consistent=evidence_lineage,
        report_hash="",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    report_hash = _stable_hash(report.to_dict())
    return Phase102Report(**{**report.__dict__, "report_hash": report_hash})
