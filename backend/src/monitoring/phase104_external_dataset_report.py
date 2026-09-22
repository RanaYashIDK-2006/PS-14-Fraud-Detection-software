"""Phase 104: External Dataset Qualification Report.

Deterministic, metadata-only report over the Phase 104 evidence contract
and the qualification results for the four known candidates.

NO EXTERNAL DATASET WAS ACQUIRED.
NO PROVIDER WAS CONTACTED.
NO REAL-WORLD VALIDATION WAS PERFORMED.

Reuses the authoritative global state from Phase 103 and the authoritative
model/release/feature identities from Phases 94-96 — no duplicate
promotion, RWV, or trust-policy authority is created here.

STATUS: IMPLEMENTED
SYSTEM_READINESS: SYSTEM_READY_PENDING_ELIGIBLE_DATASET
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
PROMOTION: PROMOTION_GATE_REQUIRED
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.monitoring.external_dataset_contract import (
    CONTRACT_VERSION,
    DOMAIN_FEATURE_COUNT,
    EVIDENCE_POLICY_VERSION,
    NATIVE_FEATURE_COUNT,
    CANONICAL_TRANSFORMATION,
    KNOWN_DATASET_IDS,
    MANDATORY_DIMENSIONS,
    build_known_candidate_contract_and_evidence,
)
from src.monitoring.external_dataset_qualification import (
    QualificationResult,
    evaluate_dataset_evidence,
)
from src.monitoring.feature_contract import ML_FEATURE_VERSION
from src.monitoring.phase103_production_readiness_closure import (
    PROMOTION_STATE as GLOBAL_PROMOTION_STATE,
    REAL_WORLD_VALIDATION as GLOBAL_REAL_WORLD_VALIDATION,
    SYSTEM_READINESS as GLOBAL_SYSTEM_READINESS,
)
from src.monitoring.rwv_readiness_audit import MODEL_ID, RELEASE_ID, FEATURE_VERSION
from src.monitoring.rwv_reproducibility import (
    NATIVE_FEATURE_VERSION,
    PREPROCESSING_HASH,
    RULE_HASH,
)
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES

REPORT_ID = "PHASE104-EXTERNAL-DATASET-QUALIFICATION"

DECLARATIONS: tuple[str, ...] = (
    "NO EXTERNAL DATASET WAS ACQUIRED.",
    "NO PROVIDER WAS CONTACTED.",
    "NO REAL-WORLD VALIDATION WAS PERFORMED.",
)

EXPECTED_CONCLUSION = "READY_WITH_EXTERNAL_PREREQUISITE"


def _stable_hash(d: Any) -> str:
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateQualificationSummary:
    """Per-candidate deterministic qualification summary."""
    dataset_id: str
    qualification_state: str
    qualified: bool
    dimension_statuses: tuple[tuple[str, str], ...]
    reason_codes: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    contradictory_evidence: tuple[str, ...]
    origin_classification: tuple[tuple[str, int], ...]
    feature_preflight_counts: tuple[tuple[str, int], ...]
    blocking_features: tuple[str, ...]
    entity_continuity_status: str
    independence_status: str
    contamination_status: str
    usage_permission_status: str
    contract_hash: str
    package_hash: str
    result_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        for key in (
            "dimension_statuses", "reason_codes", "missing_evidence",
            "contradictory_evidence", "origin_classification",
            "feature_preflight_counts", "blocking_features",
        ):
            d[key] = [list(x) if isinstance(x, tuple) else x for x in d[key]]
        return d


@dataclass(frozen=True)
class Phase104Report:
    """Deterministic Phase 104 external dataset qualification report."""
    report_id: str
    contract_version: str
    evidence_policy_version: str
    model_id: str
    release_id: str
    feature_contract_version: str
    native_feature_version: str
    feature_version: str
    preprocessing_hash: str
    rule_hash: str
    domain_feature_count: int
    native_feature_count: int
    canonical_transformation: str
    canonical_native_feature_order: tuple[str, ...]
    mandatory_dimensions: tuple[str, ...]
    system_readiness: str
    real_world_validation: str
    promotion_state: str
    candidates: tuple[CandidateQualificationSummary, ...]
    any_qualified: bool
    qualified_datasets: tuple[str, ...]
    missing_evidence_union: tuple[str, ...]
    contradictory_evidence_union: tuple[str, ...]
    declarations: tuple[str, ...]
    conclusion: str
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["candidates"] = [c.to_dict() for c in self.candidates]
        d["canonical_native_feature_order"] = list(self.canonical_native_feature_order)
        d["mandatory_dimensions"] = list(self.mandatory_dimensions)
        d["qualified_datasets"] = list(self.qualified_datasets)
        d["missing_evidence_union"] = list(self.missing_evidence_union)
        d["contradictory_evidence_union"] = list(self.contradictory_evidence_union)
        d["declarations"] = list(self.declarations)
        return d


def _summarize(dataset_id: str, result: QualificationResult) -> CandidateQualificationSummary:
    dim_status = tuple((o.dimension, o.status) for o in result.dimensions)
    preflight_counts: dict[str, int] = {}
    for item in result.feature_preflight:
        preflight_counts[item.availability] = preflight_counts.get(item.availability, 0) + 1

    def _dim_status(name: str) -> str:
        for o in result.dimensions:
            if o.dimension == name:
                return o.status
        return "missing"

    return CandidateQualificationSummary(
        dataset_id=dataset_id,
        qualification_state=result.qualification_state,
        qualified=result.qualified,
        dimension_statuses=dim_status,
        reason_codes=result.reason_codes,
        missing_evidence=result.missing_evidence,
        contradictory_evidence=result.contradictory_evidence,
        origin_classification=result.origin_classification,
        feature_preflight_counts=tuple(sorted(preflight_counts.items())),
        blocking_features=tuple(
            i.feature for i in result.feature_preflight
            if i.qualification_impact == "blocking"
        ),
        entity_continuity_status=_dim_status("entity_continuity"),
        independence_status=_dim_status("independence"),
        contamination_status=_dim_status("contamination"),
        usage_permission_status=_dim_status("usage_permission"),
        contract_hash=result.contract_hash,
        package_hash=result.package_hash,
        result_hash=result.result_hash,
    )


def generate_phase104_report() -> Phase104Report:
    """Generate the deterministic Phase 104 qualification report.

    Pure offline evaluation over metadata-only evidence packages — no
    network, no external data, no model artifacts, no RWV, no promotion.
    """
    summaries: list[CandidateQualificationSummary] = []
    for dataset_id in KNOWN_DATASET_IDS:
        contract, evidence = build_known_candidate_contract_and_evidence(dataset_id)
        result = evaluate_dataset_evidence(contract, evidence)
        summaries.append(_summarize(dataset_id, result))

    qualified = tuple(s.dataset_id for s in summaries if s.qualified)
    missing_union = tuple(sorted({
        d for s in summaries for d in s.missing_evidence
    }))
    contradictory_union = tuple(sorted({
        d for s in summaries for d in s.contradictory_evidence
    }))

    report = Phase104Report(
        report_id=REPORT_ID,
        contract_version=CONTRACT_VERSION,
        evidence_policy_version=EVIDENCE_POLICY_VERSION,
        model_id=MODEL_ID,
        release_id=RELEASE_ID,
        feature_contract_version=ML_FEATURE_VERSION,
        native_feature_version=NATIVE_FEATURE_VERSION,
        feature_version=FEATURE_VERSION,
        preprocessing_hash=PREPROCESSING_HASH,
        rule_hash=RULE_HASH,
        domain_feature_count=DOMAIN_FEATURE_COUNT,
        native_feature_count=NATIVE_FEATURE_COUNT,
        canonical_transformation=CANONICAL_TRANSFORMATION,
        canonical_native_feature_order=tuple(ALTMAN_NATIVE_FEATURES),
        mandatory_dimensions=MANDATORY_DIMENSIONS,
        system_readiness=GLOBAL_SYSTEM_READINESS,
        real_world_validation=GLOBAL_REAL_WORLD_VALIDATION,
        promotion_state=GLOBAL_PROMOTION_STATE,
        candidates=tuple(summaries),
        any_qualified=bool(qualified),
        qualified_datasets=qualified,
        missing_evidence_union=missing_union,
        contradictory_evidence_union=contradictory_union,
        declarations=DECLARATIONS,
        conclusion=EXPECTED_CONCLUSION,
        report_hash="",
    )
    report_hash = _stable_hash(report.to_dict())
    return Phase104Report(**{**report.__dict__, "report_hash": report_hash})


__all__ = [
    "REPORT_ID", "DECLARATIONS", "EXPECTED_CONCLUSION",
    "CandidateQualificationSummary", "Phase104Report",
    "generate_phase104_report",
]
