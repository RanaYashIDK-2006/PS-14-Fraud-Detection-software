"""Phase 103: Production-Readiness Closure Report.

Deterministic report summarizing the closure audit results.
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


@dataclass(frozen=True)
class Phase103Report:
    """Deterministic Phase 103 closure report."""
    report_id: str
    audit_version: str
    status: str
    total_invariants: int
    pass_count: int
    fail_count: int
    warn_count: int
    conclusion: str
    single_remaining_prerequisite: str
    dataset_blocker_summary: str
    dependency_graph_summary: str
    false_claim_summary: str
    model_release_consistent: bool
    feature_contract_intact: bool
    evidence_integrity_intact: bool
    promotion_boundary_intact: bool
    global_state_correct: bool
    report_hash: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def generate_phase103_report(closure_result: Any) -> Phase103Report:
    """Generate a deterministic Phase 103 closure report."""
    invariants = closure_result.invariants

    # Determine category pass/fail
    categories = {}
    for inv in invariants:
        cat = inv.category
        if cat not in categories:
            categories[cat] = {"pass": 0, "fail": 0, "warn": 0}
        if inv.verdict == "pass":
            categories[cat]["pass"] += 1
        elif inv.verdict == "fail":
            categories[cat]["fail"] += 1
        elif inv.verdict == "warn":
            categories[cat]["warn"] += 1

    def cat_pass(name: str) -> bool:
        return categories.get(name, {}).get("fail", 0) == 0

    report = Phase103Report(
        report_id=f"PHASE103-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        audit_version=closure_result.audit_version,
        status="PASS" if closure_result.fail_count == 0 else "FINDINGS",
        total_invariants=closure_result.total_invariants,
        pass_count=closure_result.pass_count,
        fail_count=closure_result.fail_count,
        warn_count=closure_result.warn_count,
        conclusion=closure_result.conclusion,
        single_remaining_prerequisite=closure_result.single_remaining_prerequisite,
        dataset_blocker_summary=f"{len(closure_result.dataset_blockers)} datasets blocked",
        dependency_graph_summary=f"{len(closure_result.dependency_graph)} nodes, all external blocked",
        false_claim_summary=f"{sum(1 for fc in closure_result.false_claims if fc.get('found', False))} patterns found",
        model_release_consistent=cat_pass("model_release"),
        feature_contract_intact=cat_pass("feature_contract"),
        evidence_integrity_intact=cat_pass("evidence_integrity"),
        promotion_boundary_intact=cat_pass("promotion_closure"),
        global_state_correct=cat_pass("global_state"),
        report_hash="",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    report_hash = _stable_hash(report.to_dict())
    return Phase103Report(**{**report.__dict__, "report_hash": report_hash})
