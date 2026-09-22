"""Phase 106: Public Multi-Dataset External Benchmark Report.

Deterministic, offline report over the Phase 106 public benchmark
registry: candidate discovery, provenance/licensing, classification,
feature compatibility, harmonization, leakage/contamination findings,
benchmark manifest hash and evaluation readiness.

THIS IS NOT REAL-WORLD VALIDATION.
NO EXTERNAL DATASET WAS ACQUIRED.
NO PROVIDER WAS CONTACTED.
NO DATA WAS DOWNLOADED.
NO MODEL WAS MODIFIED, RETRAINED, OR TUNED.
NO PROMOTION PATH WAS CREATED.

The benchmark manifest is metadata about a multi-source public
benchmark.  It is NOT ProviderEvidence, NOT a QualificationResult, NOT
RWVPromotionEvidence and NOT a PromotionToken — no RWV or promotion
gate accepts it.  Phase 104 remains the sole dataset-qualification
authority, Phase 96 the sole RWV harness, Phase 46 the sole promotion
authority.

Reuses the authoritative global state from Phase 103 and the
authoritative model/release/feature identities from Phases 94–96 — no
duplicate promotion, RWV, or trust-policy authority is created here.

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
    EVIDENCE_POLICY_VERSION,
    KNOWN_DATASET_IDS,
)
from src.monitoring.external_dataset_qualification import (
    evaluate_known_candidates,
)
from src.monitoring.feature_contract import ML_FEATURE_ORDER, ML_FEATURE_VERSION
from src.monitoring.phase103_production_readiness_closure import (
    PROMOTION_STATE as GLOBAL_PROMOTION_STATE,
    REAL_WORLD_VALIDATION as GLOBAL_REAL_WORLD_VALIDATION,
    SYSTEM_READINESS as GLOBAL_SYSTEM_READINESS,
)
from src.monitoring.phase106_public_benchmark_registry import (
    BENCHMARK_DATASETS,
    BENCHMARK_DESCRIPTOR,
    BENCHMARK_ID,
    COMMON_BENCHMARK_FIELDS,
    DATASET_IDS,
    EVAL_CONFIG_VERSION,
    MANIFEST_VERSION,
    REGISTRY_VERSION,
    assign_benchmark_group,
    build_benchmark_manifest,
    dataset_preflight,
    dataset_leakage_findings,
    documented_row_totals,
    group_summary,
    native_feature_eligibility_report,
    registry_integrity_hash,
    source_contamination_report,
    source_specific_compatibility_report,
    usable_features,
)
from src.monitoring.real_world_evaluation_protocol import PRODUCTION_THRESHOLD
from src.monitoring.rwv_readiness_audit import MODEL_ID, RELEASE_ID, FEATURE_VERSION
from src.monitoring.rwv_reproducibility import (
    NATIVE_FEATURE_VERSION,
    PREPROCESSING_HASH,
    RULE_HASH,
)

REPORT_ID = "PHASE106-PUBLIC-BENCHMARK-DISCOVERY"

DECLARATIONS: tuple[str, ...] = (
    "THIS IS NOT REAL-WORLD VALIDATION.",
    "NO EXTERNAL DATASET WAS ACQUIRED.",
    "NO PROVIDER WAS CONTACTED.",
    "NO DATA WAS DOWNLOADED.",
    "NO MODEL WAS MODIFIED, RETRAINED, OR TUNED.",
    "NO PROMOTION PATH WAS CREATED.",
    "CURRENT REAL_WORLD_VALIDATION REMAINS "
    "BLOCKED_PENDING_ELIGIBLE_DATASET.",
    "PRODUCTION PROMOTION REMAINS PROMOTION_GATE_REQUIRED.",
    "BENCHMARK RESULTS CONSTITUTE EXTERNAL ASSESSMENT ONLY, "
    "NEVER REAL-WORLD VALIDATION.",
)

EXPECTED_CONCLUSION = "READY_WITH_EXTERNAL_PREREQUISITE"


def _stable_hash(d: Any) -> str:
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"),
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetCompatibilitySummary:
    """Per-dataset discovery + compatibility summary."""
    dataset_id: str
    canonical_name: str
    classification: str
    group: str
    license_documented: bool
    license_terms: str
    documentation_confidence: str
    provenance_confidence: str
    transaction_count: int | None
    usable_features: tuple[str, ...]
    blocking_features: tuple[str, ...]
    eligible_rows: int
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["usable_features"] = list(self.usable_features)
        d["blocking_features"] = list(self.blocking_features)
        d["rejection_reasons"] = list(self.rejection_reasons)
        return d


@dataclass(frozen=True)
class Phase106Report:
    """Deterministic Phase 106 public benchmark discovery report."""
    report_id: str
    phase: int
    registry_version: str
    manifest_version: str
    eval_config_version: str
    contract_version: str
    evidence_policy_version: str
    benchmark_id: str
    descriptor: str
    model_id: str
    release_id: str
    feature_contract_version: str
    native_feature_version: str
    feature_version: str
    preprocessing_hash: str
    rule_hash: str
    domain_feature_count: int
    native_feature_count: int
    production_threshold: float
    system_readiness: str
    real_world_validation: str
    promotion_state: str
    candidates: tuple[DatasetCompatibilitySummary, ...]
    candidate_count: int
    group_summary: tuple[tuple[str, tuple[str, ...]], ...]
    common_harmonized_fields: tuple[str, ...]
    leakage_findings: tuple[tuple[str, tuple[str, ...]], ...]
    contamination_findings: dict[str, Any]
    native_eligibility: tuple[dict[str, Any], ...]
    row_totals: dict[str, Any]
    registry_hash: str
    manifest_hash: str
    evaluation_readiness: str
    known_candidate_states: tuple[tuple[str, str], ...]
    any_dataset_qualified: bool
    qualified_datasets: tuple[str, ...]
    declarations: tuple[str, ...]
    conclusion: str
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["candidates"] = [c.to_dict() for c in self.candidates]
        for key in ("group_summary", "common_harmonized_fields",
                    "known_candidate_states", "qualified_datasets",
                    "declarations", "leakage_findings"):
            d[key] = [list(x) if isinstance(x, tuple) else x
                      for x in d[key]]
        d["native_eligibility"] = [dict(x) for x in self.native_eligibility]
        return d


def generate_phase106_report() -> Phase106Report:
    """Generate the deterministic Phase 106 benchmark report.

    Pure offline evaluation over registry metadata — no network, no
    provider contact, no external data transfer, no model artifacts,
    no RWV, no promotion.  Zero rows are eligible for native evaluation
    because no file is acquired and no candidate is Group A.
    """
    manifest = build_benchmark_manifest()
    contamination = source_contamination_report()
    eligibility = native_feature_eligibility_report()
    row_totals = documented_row_totals()
    compat_src = source_specific_compatibility_report()
    compat_by_id = {c["dataset_id"]: c for c in compat_src}

    summaries: list[DatasetCompatibilitySummary] = []
    for ds in BENCHMARK_DATASETS:
        c = compat_by_id[ds.dataset_id]
        preflight = dataset_preflight(ds.dataset_id)
        group, reasons = assign_benchmark_group(ds.dataset_id)
        rejection: list[str] = list(reasons)
        if c["blocking_features"]:
            rejection.append(
                f"{len(c['blocking_features'])}/48 native features blocking")
        rejection.append("file_not_acquired")
        summaries.append(DatasetCompatibilitySummary(
            dataset_id=ds.dataset_id,
            canonical_name=ds.canonical_name,
            classification=ds.classification,
            group=group,
            license_documented=ds.license_documented,
            license_terms=ds.license_terms or "not_documented",
            documentation_confidence=ds.documentation_confidence,
            provenance_confidence=ds.provenance_confidence,
            transaction_count=ds.transaction_count,
            usable_features=tuple(usable_features(preflight)),
            blocking_features=tuple(
                i.feature for i in preflight
                if i.qualification_impact == "blocking"),
            eligible_rows=0,
            rejection_reasons=tuple(rejection),
        ))

    # Known RWV candidates stay governed by Phase 104/95 — never upgraded
    # by benchmark membership.
    known = evaluate_known_candidates()
    known_states = tuple(
        (k, v.qualification_state) for k, v in sorted(known.items()))
    qualified = tuple(k for k, v in known.items() if v.qualified)

    groups = group_summary()
    evaluation_readiness = (
        "NOT_READY_NO_DATASET_FULLY_COMPATIBLE_AND_NO_FILE_ACQUIRED"
        if all(s.eligible_rows == 0 for s in summaries)
        else "READY")

    report = Phase106Report(
        report_id=REPORT_ID,
        phase=106,
        registry_version=REGISTRY_VERSION,
        manifest_version=MANIFEST_VERSION,
        eval_config_version=EVAL_CONFIG_VERSION,
        contract_version=CONTRACT_VERSION,
        evidence_policy_version=EVIDENCE_POLICY_VERSION,
        benchmark_id=BENCHMARK_ID,
        descriptor=BENCHMARK_DESCRIPTOR,
        model_id=MODEL_ID,
        release_id=RELEASE_ID,
        feature_contract_version=ML_FEATURE_VERSION,
        native_feature_version=NATIVE_FEATURE_VERSION,
        feature_version=FEATURE_VERSION,
        preprocessing_hash=PREPROCESSING_HASH,
        rule_hash=RULE_HASH,
        domain_feature_count=len(ML_FEATURE_ORDER),
        native_feature_count=(
            len(compat_by_id[DATASET_IDS[0]]["usable_features"]) +
            len(compat_by_id[DATASET_IDS[0]]["blocking_features"])),
        production_threshold=PRODUCTION_THRESHOLD,
        system_readiness=GLOBAL_SYSTEM_READINESS,
        real_world_validation=GLOBAL_REAL_WORLD_VALIDATION,
        promotion_state=GLOBAL_PROMOTION_STATE,
        candidates=tuple(summaries),
        candidate_count=len(summaries),
        group_summary=tuple((g, groups[g]) for g in sorted(groups)),
        common_harmonized_fields=COMMON_BENCHMARK_FIELDS,
        leakage_findings=dataset_leakage_findings(),
        contamination_findings=contamination,
        native_eligibility=eligibility,
        row_totals=row_totals,
        registry_hash=registry_integrity_hash(),
        manifest_hash=manifest.manifest_hash,
        evaluation_readiness=evaluation_readiness,
        known_candidate_states=known_states,
        any_dataset_qualified=bool(qualified),
        qualified_datasets=qualified,
        declarations=DECLARATIONS,
        conclusion=EXPECTED_CONCLUSION,
        report_hash="",
    )
    report_hash = _stable_hash(report.to_dict())
    return Phase106Report(**{**report.__dict__, "report_hash": report_hash})


__all__ = [
    "REPORT_ID", "DECLARATIONS", "EXPECTED_CONCLUSION",
    "DatasetCompatibilitySummary", "Phase106Report",
    "generate_phase106_report",
]
