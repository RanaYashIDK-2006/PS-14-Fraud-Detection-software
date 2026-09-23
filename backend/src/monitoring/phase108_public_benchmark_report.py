"""Phase 108: Verified Public Benchmark Execution Report.

Deterministic, offline report over the FIRST real, reproducible
measurements of the CURRENT production system against the Phase 107
verified ULB benchmark file: dataset identity and integrity, the Mode A
baseline audit, the Mode B component evaluation results and metric
block, temporal/error/robustness/leakage findings, the diagnostic
threshold sweep, production artifact integrity before/after, and the
benchmark evaluation manifest hash.

THIS IS PUBLIC EXTERNAL BENCHMARK EVALUATION.  IT IS NOT REAL-WORLD
VALIDATION, NOT INSTITUTIONAL VALIDATION, NOT PROVIDER-ATTESTED
VALIDATION.

NO EXTERNAL DATASET WAS ACQUIRED BY THIS PHASE (the file was verified
locally against the Phase 107 SHA-256; no download occurred).
NO PROVIDER WAS CONTACTED.
NO REAL-WORLD VALIDATION WAS PERFORMED.
NO MODEL WAS PROMOTED, RETRAINED, FIT OR MODIFIED; NO THRESHOLD,
RELEASE OR FEATURE CONTRACT WAS CHANGED.
Genuine institutional/provider-attested evidence for any candidate
dataset remains UNRESOLVED, so qualified datasets stay empty and every
state in the Phase 103 closure remains exactly as it was.

The report embeds the full BenchmarkExecution (spec report items 7-18
and 22-25, 28) and carries the report-level statements (items 1-6,
19-21, 26-27, 29-36) as locked fields.  Run-dependent regression
results are reported by the test suite and the phase summary; the
security results recorded here are facts of THIS run (measured) or
structural properties of the Phase 108 modules (no fitting, promotion,
RWV, network or audit code path exists).

Run:  ../.venv/Scripts/python.exe scripts/phase108_public_benchmark_execution_test.py
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from src.monitoring.external_dataset_contract import canonical_json
from src.monitoring.external_dataset_qualification import (
    evaluate_known_candidates,
)
from src.monitoring.feature_contract import ML_FEATURE_ORDER
from src.monitoring.phase103_production_readiness_closure import (
    PROMOTION_STATE,
    REAL_WORLD_VALIDATION,
    SYSTEM_READINESS,
)
from src.monitoring.phase107_public_dataset_ingestion import (
    BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
    EXPECTED_PRODUCTION_THRESHOLD,
)
from src.monitoring.phase108_public_benchmark_execution import (
    CALIBRATION_STATUS,
    EXPECTED_FRAUD,
    EXPECTED_ROWS,
    EXPECTED_SHA256,
    NATIVE_MODEL_EVALUATION,
    DATASET_CANONICAL_NAME,
    DATASET_ID,
    EvaluationMode,
    BenchmarkExecution,
    execute_evaluation,
    verify_manifest_hash,
    verify_result_hash,
)
from src.monitoring.rwv_readiness_audit import FEATURE_VERSION, MODEL_ID, RELEASE_ID
from src.monitoring.rwv_reproducibility import NATIVE_FEATURE_VERSION
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES

REPORT_VERSION = "phase108_report_v1"
REPORT_CREATED_AT = "2026-09-23T00:00:00+00:00"  # fixed: stable reruns
EXPECTED_CONCLUSION = "READY_WITH_EXTERNAL_PREREQUISITE"
BENCHMARK_RESULT_STATE = "PUBLIC_EXTERNAL_BENCHMARK_RESULTS_AVAILABLE"
RESULTS_CLASSIFICATION = "PUBLIC_EXTERNAL_BENCHMARK_RESULTS"

# States the benchmark is FORBIDDEN to ever report.
FORBIDDEN_RESULT_STATES: tuple[str, ...] = (
    "REAL_WORLD_VALIDATION_COMPLETE",
    "PROMOTION_ELIGIBLE",
    "PRODUCTION_VALIDATED",
    "INSTITUTIONALLY_VALIDATED",
)

DECLARATIONS: tuple[str, ...] = (
    "PUBLIC EXTERNAL BENCHMARK RESULTS ONLY - THIS IS NOT REAL-WORLD "
    "VALIDATION, NOT INSTITUTIONAL VALIDATION, NOT PROVIDER-ATTESTED "
    "VALIDATION.",
    "ULB IS USED FOR EXTERNAL/COMPONENT BENCHMARK EVALUATION ONLY; IT "
    "IS NOT A NATIVE 48-FEATURE EVALUATION AND DOES NOT VALIDATE THE "
    "ALTMAN-NATIVE ARCHITECTURE.",
    "THE PRODUCTION THRESHOLD 0.018758 WAS NOT CHANGED; NO MODEL WAS "
    "PROMOTED, RETRAINED, FIT OR MODIFIED; NO RELEASE WAS CREATED AND "
    "NO PROMOTION TOKEN OR RWV EVIDENCE EXISTS.",
    "INSTITUTIONAL/PROVIDER EVIDENCE REMAINS UNRESOLVED: no candidate "
    "dataset has genuine provider-attested evidence, qualified "
    "datasets remain empty, and every Phase 103 closure state stays "
    "exactly as it was.",
    "PRODUCTION PROMOTION REMAINS PROMOTION_GATE_REQUIRED; REAL-WORLD "
    "VALIDATION REMAINS BLOCKED_PENDING_ELIGIBLE_DATASET; NOTHING IN "
    "THIS PHASE MOVES EITHER STATE.",
    "MEASURED RESULTS ARE REPORTED WITH THEIR LIMITATIONS: NO "
    "GOOD/BAD JUDGMENT, NO RANKING, NO SUPERIORITY, NO "
    "PRODUCTION-READINESS AND NO GENERALIZATION CLAIM IS MADE.",
    "THE FAIL-CLOSED CONSTANT SCORE VECTOR MEASURES DECISION BEHAVIOR "
    "UNDER ABSENT INPUTS, NOT DETECTION CAPABILITY.",
)

LIMITATIONS: tuple[str, ...] = (
    "no overall good/bad judgment of the model is made",
    "the system is not ranked against other models and no superiority "
    "is claimed",
    "benchmark results do not establish production readiness",
    "no generalization to real financial institutions is claimed",
    "ULB does not validate the 48-feature Altman-Native architecture "
    "(Group B: 5/48 native features usable; native evaluation "
    "NOT_APPLICABLE)",
    "the scored component fail-closes to a constant score because 0/17 "
    "rule features are derivable from ULB: the metrics measure "
    "fail-closed decision behavior, not detection",
    "ULB Time is dataset-relative seconds: no calendar period is "
    "claimed for any temporal slice",
    "raw accuracy is misleading at 0.1727% fraud prevalence; the "
    "imbalance warning travels with the metric block",
    "threshold-sweep rows are DIAGNOSTIC_ONLY descriptive analysis, "
    "never threshold selection",
)

SECURITY_RESULTS: tuple[tuple[str, str], ...] = (
    ("network_access",
     "none: execution performs local file reads only (source scan in "
     "the test suite confirms no network code path)"),
    ("production_artifact_integrity",
     "12/12 artifacts hashed before and after, byte-identical "
     "(measured in this run)"),
    ("production_threshold",
     "0.018758 unchanged, read-only before/after (measured in this "
     "run)"),
    ("retrain_fit_tune",
     "not performed: no fitting or optimization code path exists in "
     "the Phase 108 modules"),
    ("promotion_or_release",
     "not performed: no promotion token, RWV promotion-evidence object "
     "or release creation is constructed anywhere in Phase 108"),
    ("rwv_session",
     "not performed: no controlled RWV session is created; Phase 96 "
     "entry requires Phase 104 qualification first"),
    ("audit_chain",
     "not written: benchmark results are confined to "
     "reports/external_benchmark/phase108/ulb/"),
    ("bypass_parameters",
     "absent from every public entry point (signature assertions in "
     "the test suite reject force/allow_unverified/skip_validation/"
     "override/admin_override/bypass)"),
)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 over the canonical JSON encoding."""
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Phase108BenchmarkReport:
    """Immutable, hash-stable summary of the Phase 108 outcome."""

    phase: int
    report_version: str
    created_at: str
    # --- locked production identity ------------------------------------
    model_id: str
    release_id: str
    feature_version: str
    native_feature_version: str
    domain_feature_count: int
    native_feature_count: int
    production_threshold: float
    canonical_mapping: str
    # --- global states (reported exactly as-is) ------------------------
    system_readiness: str
    real_world_validation: str
    promotion_state: str
    # --- spec report items 1-6: dataset identity + integrity -----------
    dataset_id: str
    dataset_canonical_name: str
    dataset_sha256: str
    row_count: int
    fraud_count: int
    fraud_prevalence: float
    integrity_result: str
    # --- items 7-9 headline (full detail inside execution) -------------
    modes_executed: tuple[str, ...]
    native_model_evaluation: str
    mode_c_status: str
    # --- items 19-21: threshold + calibration --------------------------
    threshold_used: float
    production_threshold_changed: bool
    calibration_status: str
    # --- items 26-27: manifest + model/release identity ----------------
    benchmark_evaluation_manifest_sha256: str
    benchmark_results_hash: str
    # --- classification / governance -----------------------------------
    benchmark_result_state: str
    results_classification: str
    known_candidate_states: tuple[tuple[str, str, bool], ...]
    qualified_datasets: tuple[str, ...]
    any_dataset_qualified: bool
    reproducibility_status: str
    security_results: tuple[tuple[str, str], ...]
    limitations: tuple[str, ...]
    not_gate_evidence: tuple[str, ...]
    declarations: tuple[str, ...]
    conclusion: str
    # --- items 7-18, 22-25, 28 in full ---------------------------------
    execution: BenchmarkExecution
    report_hash: str

    def __post_init__(self) -> None:
        if self.phase != 108:
            raise ValueError("report phase is fixed")
        if self.report_version != REPORT_VERSION:
            raise ValueError("unexpected report version")
        if self.created_at != REPORT_CREATED_AT:
            raise ValueError("report timestamp is a fixed constant")
        if self.model_id != MODEL_ID:
            raise ValueError("production model identity is locked")
        if self.release_id != RELEASE_ID:
            raise ValueError("production release identity is locked")
        if self.feature_version != FEATURE_VERSION:
            raise ValueError("feature version is locked")
        if self.native_feature_version != NATIVE_FEATURE_VERSION:
            raise ValueError("native feature version is locked")
        if self.domain_feature_count != len(ML_FEATURE_ORDER):
            raise ValueError("domain feature count is locked")
        if self.native_feature_count != len(ALTMAN_NATIVE_FEATURES):
            raise ValueError("native feature count is locked")
        if self.production_threshold != EXPECTED_PRODUCTION_THRESHOLD:
            raise ValueError("production threshold is locked at 0.018758")
        if self.threshold_used != EXPECTED_PRODUCTION_THRESHOLD:
            raise ValueError("the reported threshold must be production's")
        if self.production_threshold_changed:
            raise ValueError(
                "reporting a threshold change is forbidden in Phase 108")
        if self.canonical_mapping.count("map_raw_to_native") != 1:
            raise ValueError("canonical mapping identity is locked")
        if self.system_readiness != SYSTEM_READINESS:
            raise ValueError("system readiness must be reported as-is")
        if self.real_world_validation != REAL_WORLD_VALIDATION:
            raise ValueError("real-world validation must be reported as-is")
        if self.promotion_state != PROMOTION_STATE:
            raise ValueError("promotion state must be reported as-is")
        # --- items 1-6 bind the executed dataset exactly ---------------
        if self.dataset_id != DATASET_ID:
            raise ValueError("dataset identity is pinned to the Phase 107 ID")
        if self.dataset_canonical_name != DATASET_CANONICAL_NAME:
            raise ValueError("dataset name must match the execution")
        if self.dataset_sha256 != EXPECTED_SHA256:
            raise ValueError("dataset hash must be the Phase 107 digest")
        if self.row_count != EXPECTED_ROWS:
            raise ValueError("row count must stay pinned at 284807")
        if self.fraud_count != EXPECTED_FRAUD:
            raise ValueError("fraud count must stay pinned at 492")
        if round(self.fraud_prevalence, 6) != round(
                EXPECTED_FRAUD / EXPECTED_ROWS, 6):
            raise ValueError("fraud prevalence must reconcile")
        if self.integrity_result != "VERIFIED":
            raise ValueError("only a VERIFIED execution may be reported")
        # --- items 7-9 / 19-21 headline ---------------------------------
        expected_modes = (
            EvaluationMode.MODE_A_BASELINE_DATASET_AUDIT.value,
            EvaluationMode.MODE_B_COMPONENT_EVALUATION.value)
        if self.modes_executed != expected_modes:
            raise ValueError("only Mode A and Mode B may be reported")
        if self.native_model_evaluation != NATIVE_MODEL_EVALUATION:
            raise ValueError("native model evaluation stays NOT_APPLICABLE")
        if self.mode_c_status != NATIVE_MODEL_EVALUATION:
            raise ValueError("Mode C stays NOT_APPLICABLE")
        if self.calibration_status != CALIBRATION_STATUS:
            raise ValueError("calibration stays NOT_APPLICABLE")
        # --- items 26-27 bind the embedded execution --------------------
        if self.benchmark_evaluation_manifest_sha256 != \
                self.execution.manifest.manifest_hash:
            raise ValueError("manifest hash must match the execution")
        if self.benchmark_results_hash != self.execution.result_hash:
            raise ValueError("results hash must match the execution")
        # --- forbidden states never appear ------------------------------
        reported = (
            self.system_readiness, self.real_world_validation,
            self.promotion_state, self.benchmark_result_state,
            self.results_classification, self.conclusion)
        for token in FORBIDDEN_RESULT_STATES:
            if token in reported:
                raise ValueError(f"forbidden result state: {token}")
        if self.benchmark_result_state != BENCHMARK_RESULT_STATE:
            raise ValueError(
                "the benchmark reports results available, nothing more")
        if self.results_classification != RESULTS_CLASSIFICATION:
            raise ValueError("results are public external benchmark only")
        # --- Phase 104 authority stays empty ----------------------------
        if self.any_dataset_qualified or self.qualified_datasets:
            raise ValueError(
                "no candidate may report as qualified without genuine "
                "provider-attested evidence")
        for _d, _state, _q in self.known_candidate_states:
            if _q:
                raise ValueError(
                    "no known candidate may report as qualified")
        if self.not_gate_evidence != BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE:
            raise ValueError("not-gate-evidence declaration must be carried")
        if self.declarations != DECLARATIONS:
            raise ValueError("declarations must not be altered")
        if self.limitations != LIMITATIONS:
            raise ValueError("limitations must not be altered")
        if not self.security_results or any(
                not k or not v for k, v in self.security_results):
            raise ValueError("security results must be stated concretely")
        if self.reproducibility_status not in {
                "DETERMINISTIC_HASHES_VERIFIED",
                "HASH_MISMATCH_FAIL_CLOSED"}:
            raise ValueError("invalid reproducibility status")
        if self.conclusion != EXPECTED_CONCLUSION:
            raise ValueError("conclusion is fixed by the closure state")
        # --- embedded execution is self-consistent ----------------------
        if self.execution.dataset_sha256 != self.dataset_sha256:
            raise ValueError("execution must bind the reported dataset")
        if self.execution.threshold_used != self.threshold_used:
            raise ValueError("execution and report thresholds must agree")
        if not verify_result_hash(self.execution):
            raise ValueError("embedded execution result hash is invalid")
        object.__setattr__(self, "report_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        payload = dataclasses.asdict(self)
        payload["report_hash"] = ""
        return _stable_hash(payload)

    def verify_hash(self) -> bool:
        return self.report_hash == self._compute_hash()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def generate_phase108_report(
    execution: BenchmarkExecution | None = None,
) -> Phase108BenchmarkReport:
    """Build the deterministic Phase 108 report from a real execution.

    The default runs execute_evaluation() (offline, deterministic,
    ~10s over the verified local file); callers may pass a
    precomputed execution to avoid the re-read.
    """
    if execution is None:
        execution = execute_evaluation()

    manifest_ok = verify_manifest_hash(execution.manifest)
    result_ok = verify_result_hash(execution)
    reproducibility = ("DETERMINISTIC_HASHES_VERIFIED"
                       if manifest_ok and result_ok
                       else "HASH_MISMATCH_FAIL_CLOSED")

    candidates = evaluate_known_candidates()
    candidate_states = tuple(
        (d, candidates[d].qualification_state, candidates[d].qualified)
        for d in sorted(candidates))
    qualified = tuple(d for d in sorted(candidates)
                      if candidates[d].qualified)

    return Phase108BenchmarkReport(
        phase=108,
        report_version=REPORT_VERSION,
        created_at=REPORT_CREATED_AT,
        model_id=MODEL_ID,
        release_id=RELEASE_ID,
        feature_version=FEATURE_VERSION,
        native_feature_version=NATIVE_FEATURE_VERSION,
        domain_feature_count=len(ML_FEATURE_ORDER),
        native_feature_count=len(ALTMAN_NATIVE_FEATURES),
        production_threshold=EXPECTED_PRODUCTION_THRESHOLD,
        canonical_mapping=execution.manifest.canonical_mapping,
        system_readiness=SYSTEM_READINESS,
        real_world_validation=REAL_WORLD_VALIDATION,
        promotion_state=PROMOTION_STATE,
        dataset_id=execution.dataset_id,
        dataset_canonical_name=execution.dataset_canonical_name,
        dataset_sha256=execution.dataset_sha256,
        row_count=execution.row_count,
        fraud_count=execution.fraud_count,
        fraud_prevalence=execution.fraud_prevalence,
        integrity_result=execution.integrity_result,
        modes_executed=execution.modes_executed,
        native_model_evaluation=execution.native_model_evaluation,
        mode_c_status=execution.mode_c_status,
        threshold_used=execution.threshold_used,
        production_threshold_changed=not execution.threshold_unchanged,
        calibration_status=execution.calibration_status,
        benchmark_evaluation_manifest_sha256=execution.manifest.manifest_hash,
        benchmark_results_hash=execution.result_hash,
        benchmark_result_state=BENCHMARK_RESULT_STATE,
        results_classification=RESULTS_CLASSIFICATION,
        known_candidate_states=candidate_states,
        qualified_datasets=qualified,
        any_dataset_qualified=bool(qualified),
        reproducibility_status=reproducibility,
        security_results=SECURITY_RESULTS,
        limitations=LIMITATIONS,
        not_gate_evidence=execution.manifest.not_gate_evidence,
        declarations=DECLARATIONS,
        conclusion=EXPECTED_CONCLUSION,
        execution=execution,
        report_hash="",
    )
