"""Phase 107: Public Multi-Dataset Benchmark Report.

Deterministic, offline report over the Phase 107 local-only ingestion
of the Phase 106 registered public benchmark candidates: what was
acquired, what was verified, schemas/labels/duplicates/contamination
findings, the admitted source partitions, component-evaluation
eligibility, and the benchmark manifest hash.

THIS IS PUBLIC BENCHMARK TESTING.  THIS IS NOT REAL-WORLD VALIDATION.

NO EXTERNAL DATA WAS DOWNLOADED — every file this report describes was
already present locally (manual user acquisition) and was verified
locally, offline, deterministically.
NO PROVIDER WAS CONTACTED.
NO MODEL WAS PROMOTED, RETRAINED OR MODIFIED; NO THRESHOLD, FEATURE
CONTRACT OR CANONICAL PIPELINE WAS CHANGED.
Genuine institutional/provider-attested evidence for any candidate
dataset remains UNRESOLVED, so qualified datasets stay empty and every
state in the Phase 103 closure remains exactly as it was.

The report is generated only from local ingestion records and the
existing authorities; it never touches the network, model artifacts or
any gate.  Its benchmark manifest reuses
BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE (Phase 106): bookkeeping only,
accepted by no existing gate.

Run:  ../.venv/Scripts/python.exe scripts/phase107_public_dataset_ingestion_test.py
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.monitoring.external_dataset_contract import canonical_json
from src.monitoring.external_dataset_qualification import (
    evaluate_known_candidates,
)
from src.monitoring.phase103_production_readiness_closure import (
    PROMOTION_STATE,
    REAL_WORLD_VALIDATION,
    SYSTEM_READINESS,
)
from src.monitoring.phase106_public_benchmark_registry import (
    BENCHMARK_DESCRIPTOR,
    COMMON_BENCHMARK_FIELDS,
    DATASET_IDS,
    REGISTRY_VERSION,
    get_dataset,
    native_eligible_row_count,
    source_contamination_report,
)
from src.monitoring.phase107_public_dataset_ingestion import (
    BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
    BENCHMARK_NAME,
    EXPECTED_PRODUCTION_THRESHOLD,
    INGESTION_VERSION,
    MANIFEST_VERSION,
    BenchmarkPartition,
    BenchmarkReadiness,
    IngestionRecord,
    Phase107BenchmarkManifest,
    build_component_evaluation_views,
    build_phase107_manifest,
    build_partitions,
    ingest_all,
)
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES
from src.monitoring.feature_contract import ML_FEATURE_ORDER
from src.monitoring.rwv_readiness_audit import FEATURE_VERSION, MODEL_ID, RELEASE_ID
from src.monitoring.rwv_reproducibility import NATIVE_FEATURE_VERSION

REPORT_VERSION = "phase107_report_v1"
EXPECTED_CONCLUSION = "READY_WITH_EXTERNAL_PREREQUISITE"

DECLARATIONS: tuple[str, ...] = (
    "PUBLIC BENCHMARK TESTING ONLY - THIS IS NOT REAL-WORLD VALIDATION.",
    "NO DATA WAS DOWNLOADED AND NO PROVIDER WAS CONTACTED: every file "
    "was already present locally (manual user acquisition) and was "
    "verified offline, locally and deterministically.",
    "NO MODEL WAS PROMOTED, RETRAINED OR MODIFIED; NO THRESHOLD, "
    "FEATURE CONTRACT OR CANONICAL 21->48 PIPELINE WAS CHANGED.",
    "INSTITUTIONAL/PROVIDER EVIDENCE REMAINS UNRESOLVED: no candidate "
    "dataset has genuine provider-attested evidence, qualified "
    "datasets remain empty, and every Phase 103 closure state stays "
    "exactly as it was.",
    "PRODUCTION PROMOTION REMAINS PROMOTION_GATE_REQUIRED; REAL-WORLD "
    "VALIDATION REMAINS BLOCKED_PENDING_ELIGIBLE_DATASET; NOTHING IN "
    "THIS PHASE MOVES EITHER STATE.",
    "INGESTED PUBLIC BENCHMARK ROWS NEVER CONSTITUTE PROVIDER-ATTESTED "
    "EVIDENCE AND CANNOT UNBLOCK ANY EXISTING GATE.",
    "THE BENCHMARK MAY REACH READY_FOR_COMPONENT_EVALUATION BUT MUST "
    "NEVER BE READ AS READY_FOR_REAL_WORLD_VALIDATION.",
    "PUBLIC DATASET ROWS ARE MULTI-SOURCE PUBLIC RESEARCH DATA - NEVER "
    "INSTITUTIONAL, WORLDLINE, OR PROVIDER-VALIDATED TRANSACTIONS.",
)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 over the canonical JSON encoding."""
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Phase107BenchmarkReport:
    """Immutable, hash-stable summary of the Phase 107 outcome."""

    phase: int
    report_version: str
    registry_version: str
    ingestion_version: str
    manifest_version: str
    model_id: str
    release_id: str
    feature_version: str
    native_feature_version: str
    domain_feature_count: int
    native_feature_count: int
    production_threshold: float
    canonical_mapping: str
    system_readiness: str
    real_world_validation: str
    promotion_state: str
    # --- acquisition / verification (spec report items 1-7) ----------
    datasets_acquired: tuple[str, ...]
    datasets_verified: tuple[str, ...]
    datasets_failed_verification: tuple[str, ...]
    datasets_absent: tuple[str, ...]
    acquisition_statuses: tuple[tuple[str, str], ...]
    file_hashes: tuple[tuple[str, str, str], ...]
    dataset_versions: tuple[tuple[str, str], ...]
    schema_verification: tuple[tuple[str, str, tuple[str, ...],
                                     tuple[tuple[str, str], ...],
                                     tuple[str, ...],
                                     tuple[tuple[str, int], ...],
                                     str, str,
                                     tuple[str, ...],
                                     tuple[str, ...]], ...]
    row_counts: tuple[tuple[str, int], ...]
    column_counts: tuple[tuple[str, int], ...]
    label_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    # --- governance / labels / leakage (items 8-10) -------------------
    label_semantics: tuple[tuple[str, str], ...]
    license_provenance: tuple[tuple[str, bool, str], ...]
    excluded_leakage_fields: tuple[tuple[str, tuple[str, ...]], ...]
    # --- findings (items 11-12) ---------------------------------------
    duplicate_findings: tuple[tuple[str, int, int, int, int], ...]
    source_contamination_findings: tuple[str, ...]
    # --- schema / partitions (items 13-14) ----------------------------
    benchmark_name: str
    benchmark_descriptor: str
    harmonized_fields: tuple[str, ...]
    canonical_field_mappings: tuple[
        tuple[str, tuple[tuple[str, str, str], ...]], ...]
    partitions: tuple[tuple[str, str, int, int, str, str, str, str], ...]
    component_views: tuple[tuple[str, str, tuple[str, ...],
                                 tuple[str, ...]], ...]
    # --- native boundary (items 15-17) --------------------------------
    native_compatibility: tuple[tuple[str, str, int, int], ...]
    native_evaluation_eligible: bool
    rows_eligible_native: int
    rows_eligible_component: tuple[tuple[str, int], ...]
    rows_eligible_component_total: int
    # --- manifest / reproducibility (items 18-19) ---------------------
    benchmark_manifest_sha256: str
    reproducibility_status: str
    # --- Phase 104 authority + states ---------------------------------
    known_candidate_states: tuple[tuple[str, str, bool], ...]
    qualified_datasets: tuple[str, ...]
    any_dataset_qualified: bool
    benchmark_readiness: str
    not_gate_evidence: str
    declarations: tuple[str, ...]
    conclusion: str
    report_hash: str

    def __post_init__(self) -> None:
        if self.phase != 107:
            raise ValueError("report phase is fixed")
        if self.report_version != REPORT_VERSION:
            raise ValueError("unexpected report version")
        if self.model_id != MODEL_ID:
            raise ValueError("production model identity is locked")
        if self.release_id != RELEASE_ID:
            raise ValueError("production release identity is locked")
        if self.feature_version != FEATURE_VERSION:
            raise ValueError("feature version is locked")
        if self.native_feature_version != NATIVE_FEATURE_VERSION:
            raise ValueError("native feature version is locked")
        if self.production_threshold != EXPECTED_PRODUCTION_THRESHOLD:
            raise ValueError("production threshold is locked")
        if self.canonical_mapping.count("map_raw_to_native") != 1:
            raise ValueError("canonical mapping identity is locked")
        if self.system_readiness != SYSTEM_READINESS:
            raise ValueError("system readiness must be reported as-is")
        if self.real_world_validation != REAL_WORLD_VALIDATION:
            raise ValueError("real-world validation must be reported as-is")
        if self.promotion_state != PROMOTION_STATE:
            raise ValueError("promotion state must be reported as-is")
        if self.domain_feature_count != len(ML_FEATURE_ORDER):
            raise ValueError("domain feature count is locked")
        if self.native_feature_count != len(ALTMAN_NATIVE_FEATURES):
            raise ValueError("native feature count is locked")
        if self.rows_eligible_native != 0:
            raise ValueError(
                "no dataset may become native-eligible in this phase")
        if self.native_evaluation_eligible:
            raise ValueError(
                "native evaluation stays ineligible (Group A is empty)")
        if self.any_dataset_qualified or self.qualified_datasets:
            raise ValueError(
                "no candidate may report as qualified without genuine "
                "provider-attested evidence")
        if self.not_gate_evidence != BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE:
            raise ValueError("manifest declaration must be carried")
        if self.conclusion != EXPECTED_CONCLUSION:
            raise ValueError("conclusion is fixed by the closure state")
        if self.declarations != DECLARATIONS:
            raise ValueError("declarations must not be altered")
        if self.benchmark_readiness not in {
                s.value for s in BenchmarkReadiness}:
            raise ValueError("invalid benchmark readiness")
        if self.rows_eligible_component_total != sum(
                n for _, n in self.rows_eligible_component):
            raise ValueError("component rows must reconcile")
        object.__setattr__(self, "report_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        payload = dataclasses.asdict(self)
        payload["report_hash"] = ""
        return _stable_hash(payload)

    def verify_hash(self) -> bool:
        return self.report_hash == self._compute_hash()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def generate_phase107_report(
    records: Mapping[str, IngestionRecord] | None = None,
    partitions: Sequence[BenchmarkPartition] | None = None,
    manifest: Phase107BenchmarkManifest | None = None,
) -> Phase107BenchmarkReport:
    """Build the deterministic Phase 107 report from local state.

    Default inputs re-run the local ingestion (offline, deterministic);
    callers may pass precomputed records to avoid the re-read.  The
    manifest is rebuilt a second time to state reproducibility as a
    measured fact rather than an assumption.
    """
    if records is None:
        records = ingest_all()
    if partitions is None:
        partitions = build_partitions(records)
    if manifest is None:
        manifest = build_phase107_manifest(records, partitions)

    rebuild = build_phase107_manifest(records, partitions)
    reproducibility = ("DETERMINISTIC_REBUILD_IDENTICAL"
                       if rebuild.manifest_hash == manifest.manifest_hash
                       else "REBUILD_MISMATCH_FAIL_CLOSED")

    acquired = tuple(d for d in DATASET_IDS
                     if d in records
                     and records[d].status != "not_acquired")
    verified = tuple(d for d in DATASET_IDS
                     if d in records and records[d].verified)
    failed = tuple(d for d in DATASET_IDS
                   if d in records
                   and records[d].status
                   in ("acquired_verification_failed",
                       "incomplete_acquisition"))
    absent = tuple(d for d in DATASET_IDS
                   if d not in records
                   or records[d].status == "not_acquired")
    statuses = tuple((d, records[d].status) for d in DATASET_IDS
                     if d in records)
    file_hashes = tuple(
        (d, f.path, f.sha256) for d in DATASET_IDS if d in records
        for f in records[d].file_records)
    versions = tuple((d, records[d].version) for d in DATASET_IDS
                     if d in records)
    schema_verification = tuple(
        (d, records[d].status, records[d].missing_columns,
         records[d].renamed_columns, records[d].extra_columns,
         records[d].type_mismatch_counts,
         records[d].row_count_check, records[d].fraud_count_check,
         records[d].verification_errors,
         records[d].verification_warnings)
        for d in DATASET_IDS if d in records)
    row_counts = tuple((d, records[d].row_count) for d in DATASET_IDS
                       if d in records)
    column_counts = tuple((d, records[d].column_count)
                          for d in DATASET_IDS if d in records)
    label_counts = tuple((d, records[d].label_counts)
                         for d in DATASET_IDS if d in records)
    label_semantics = tuple(
        (d, get_dataset(d).label_semantics) for d in DATASET_IDS)
    license_provenance = tuple(
        (d, get_dataset(d).license_documented,
         get_dataset(d).provenance_confidence) for d in DATASET_IDS)
    excluded_leakage = tuple(
        (p.dataset_id, p.excluded_fields) for p in partitions)
    duplicate_findings = tuple(
        (p.dataset_id,
         p.duplicate_findings.within_source_duplicate_rows,
         p.duplicate_findings.cross_source_duplicate_rows,
         p.duplicate_findings.normalized_collision_rows,
         p.duplicate_findings.unique_rows)
        for p in partitions)
    contamination = tuple(
        f"{key}:{canonical_json(value)}"
        for key, value in sorted(source_contamination_report().items()))
    partition_rows = tuple(
        (p.partition_id, p.dataset_id, p.row_count, p.supervised_rows,
         p.group, p.evaluation_readiness, p.primary_sha256,
         p.partition_hash)
        for p in partitions)
    views = build_component_evaluation_views(partitions)
    view_rows = tuple(
        (v.view_id, v.status, v.source_partitions, v.included_fields)
        for v in views)
    native_compat = manifest.native_compatibility
    component_rows = tuple((p.dataset_id, p.supervised_rows)
                           for p in partitions)
    component_total = sum(n for _, n in component_rows)
    native_rows = sum(native_eligible_row_count(d) for d in DATASET_IDS)

    candidates = evaluate_known_candidates()
    candidate_states = tuple(
        (d, candidates[d].qualification_state, candidates[d].qualified)
        for d in sorted(candidates))
    qualified = tuple(d for d in sorted(candidates)
                      if candidates[d].qualified)

    readiness = (BenchmarkReadiness.READY_FOR_COMPONENT_EVALUATION.value
                 if partitions and component_total > 0
                 else
                 BenchmarkReadiness.NOT_READY_NO_VERIFIED_SOURCE.value)

    return Phase107BenchmarkReport(
        phase=107,
        report_version=REPORT_VERSION,
        registry_version=REGISTRY_VERSION,
        ingestion_version=INGESTION_VERSION,
        manifest_version=MANIFEST_VERSION,
        model_id=MODEL_ID,
        release_id=RELEASE_ID,
        feature_version=FEATURE_VERSION,
        native_feature_version=NATIVE_FEATURE_VERSION,
        domain_feature_count=len(ML_FEATURE_ORDER),
        native_feature_count=len(ALTMAN_NATIVE_FEATURES),
        production_threshold=EXPECTED_PRODUCTION_THRESHOLD,
        canonical_mapping=manifest.canonical_mapping,
        system_readiness=SYSTEM_READINESS,
        real_world_validation=REAL_WORLD_VALIDATION,
        promotion_state=PROMOTION_STATE,
        datasets_acquired=acquired,
        datasets_verified=verified,
        datasets_failed_verification=failed,
        datasets_absent=absent,
        acquisition_statuses=statuses,
        file_hashes=file_hashes,
        dataset_versions=versions,
        schema_verification=schema_verification,
        row_counts=row_counts,
        column_counts=column_counts,
        label_counts=label_counts,
        label_semantics=label_semantics,
        license_provenance=license_provenance,
        excluded_leakage_fields=excluded_leakage,
        duplicate_findings=duplicate_findings,
        source_contamination_findings=contamination,
        benchmark_name=BENCHMARK_NAME,
        benchmark_descriptor=BENCHMARK_DESCRIPTOR,
        harmonized_fields=COMMON_BENCHMARK_FIELDS,
        canonical_field_mappings=manifest.canonical_field_mappings,
        partitions=partition_rows,
        component_views=view_rows,
        native_compatibility=native_compat,
        native_evaluation_eligible=native_rows > 0,
        rows_eligible_native=native_rows,
        rows_eligible_component=component_rows,
        rows_eligible_component_total=component_total,
        benchmark_manifest_sha256=manifest.manifest_hash,
        reproducibility_status=reproducibility,
        known_candidate_states=candidate_states,
        qualified_datasets=qualified,
        any_dataset_qualified=bool(qualified),
        benchmark_readiness=readiness,
        not_gate_evidence=BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
        declarations=DECLARATIONS,
        conclusion=EXPECTED_CONCLUSION,
        report_hash="",
    )


__all__ = [
    "DECLARATIONS", "EXPECTED_CONCLUSION", "REPORT_VERSION",
    "Phase107BenchmarkReport", "generate_phase107_report",
]
