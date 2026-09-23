"""Phase 108: Verified Public Benchmark Execution — test suite.

Deterministic, offline, adversarial coverage for the Phase 108
execution engine, metric authority parity, Mode A/B evaluation,
threshold policy, temporal/error analyses, contamination and artifact
integrity, the evaluation manifest, the deterministic report, and the
untouched RWV/promotion boundaries.  Never touches the network, model
artifacts (read-only hashing only), DB-4 or any authority.

Run:  ../.venv/Scripts/python.exe scripts/phase108_public_benchmark_execution_test.py
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
REPO_ROOT = os.path.dirname(BACKEND)

from src.monitoring import phase108_public_benchmark_execution as EXC
from src.monitoring import phase108_public_benchmark_report as REP
from src.monitoring import real_world_evaluation_protocol as PROTO
from src.monitoring.external_dataset_contract import (
    CANONICAL_TRANSFORMATION,
    canonical_json,
)
from src.monitoring.external_dataset_qualification import (
    QualificationResult,
    evaluate_known_candidates,
)
from src.monitoring.feature_contract import ML_FEATURE_ORDER, ML_FEATURE_VERSION
from src.monitoring.phase103_production_readiness_closure import (
    PROMOTION_STATE,
    REAL_WORLD_VALIDATION,
    SYSTEM_READINESS,
    check_global_state,
)
from src.monitoring.phase106_public_benchmark_registry import (
    assign_benchmark_group,
    dataset_preflight,
    evaluation_scope,
    harmonize_rows,
    native_eligible_row_count,
    normalize_label,
    row_fingerprint,
    usable_features,
)
from src.monitoring.phase107_public_dataset_ingestion import (
    BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
    EXPECTED_PRODUCTION_THRESHOLD,
    FORBIDDEN_BYPASS_PARAMETERS,
    ingest_dataset,
    sha256_file,
)
from src.monitoring.phase108_public_benchmark_execution import (
    AMOUNT_BUCKET_EDGES,
    EXPECTED_COLUMNS,
    EXPECTED_FRAUD,
    EXPECTED_INGESTION_STATUS,
    EXPECTED_LEGIT,
    EXPECTED_PRODUCTION_THRESHOLD as EXEC_PINNED_THRESHOLD,
    EXPECTED_ROWS,
    EXPECTED_SCHEMA,
    EXPECTED_SCHEMA_HASH,
    EXPECTED_SHA256,
    EXECUTION_VERSION,
    IMBALANCE_WARNING,
    MANIFEST_CREATED_AT,
    MANIFEST_VERSION,
    NATIVE_MODEL_EVALUATION,
    PRODUCTION_ARTIFACT_PATHS,
    PROBE_SPECS,
    RESULTS_NAMESPACE,
    RULE_FEATURE_UNAVAILABILITY,
    SWEEP_LABELS,
    SWEEP_STATEMENT,
    SWEEP_THRESHOLDS,
    BenchmarkExecution,
    BaselineAudit,
    ComponentResult,
    ContaminationCheck,
    DatasetIntegrityError,
    DiagnosticSweep,
    ErrorAnalysis,
    EvaluationManifest,
    EvaluationMode,
    ProductionArtifactError,
    TemporalAnalysis,
    TemporalSlice,
    BucketResult,
    ArtifactIntegrity,
    _artifact_path,
    _row_fingerprint,
    _validate,
    compare_artifacts,
    confusion_at_threshold,
    evaluation_metrics,
    execute_evaluation,
    probe_metrics,
    resolve_benchmark_file,
    run_input_validation_probes,
    snapshot_artifacts,
    verify_manifest_hash,
    verify_result_hash,
    verify_source_integrity,
    write_evaluation_results,
)
from src.monitoring.phase108_public_benchmark_report import (
    BENCHMARK_RESULT_STATE,
    DECLARATIONS,
    EXPECTED_CONCLUSION,
    FORBIDDEN_RESULT_STATES,
    LIMITATIONS,
    REPORT_CREATED_AT,
    REPORT_VERSION,
    RESULTS_CLASSIFICATION,
    SECURITY_RESULTS,
    Phase108BenchmarkReport,
    generate_phase108_report,
)
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES
from src.monitoring.promotion_gate import (
    GateStatus,
    PromotionBlockedError,
    PromotionToken,
    PromotionVerdict,
    assert_promotion_allowed,
    evaluate_promotion,
    evaluate_real_world_validation,
)
from src.monitoring.provider_evidence import KNOWN_CANDIDATES, qualify_dataset
from src.monitoring.real_world_evaluation_protocol import (
    PRODUCTION_THRESHOLD,
    compute_metrics,
)
from src.risk_engine.altman_native_ensemble import map_raw_to_native
from src.risk_engine.main import FeatureVector, band_of
from src.monitoring.rwv_execution import (
    SessionState,
    check_rwv_execution_eligibility,
    create_session,
)
from src.monitoring.rwv_promotion_evidence import RWVPromotionEvidence
from src.monitoring.rwv_readiness_audit import FEATURE_VERSION, MODEL_ID, RELEASE_ID
from src.monitoring.rwv_reproducibility import NATIVE_FEATURE_VERSION

# ══════════════════════════════════════════════════════════════════════
# HARNESS
# ══════════════════════════════════════════════════════════════════════

ASSERTIONS = 0
INVARIANT_COUNT = 0
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    global ASSERTIONS
    ASSERTIONS += 1
    if not cond:
        FAILURES.append(msg)
        print(f"FAIL  {msg}")


def inv(cond: bool, msg: str) -> None:
    global INVARIANT_COUNT
    INVARIANT_COUNT += 1
    check(cond, f"[inv] {msg}")


def section(title: str) -> None:
    print(f"\n== SECTION {title}")


def expect_failclosed(fn, msg: str) -> None:
    try:
        fn()
    except Exception:
        check(True, msg)
        return
    check(False, msg + " (no exception raised)")


def expect_valueerror(fn, msg: str) -> None:
    try:
        fn()
    except ValueError:
        check(True, msg)
        return
    except Exception as exc:  # wrong exception type
        check(False, msg + f" (wrong exception: {type(exc).__name__})")
        return
    check(False, msg + " (no exception raised)")


def expect_typeerror(fn, msg: str) -> None:
    try:
        fn()
    except TypeError:
        check(True, msg)
        return
    except Exception as exc:
        check(False, msg + f" (wrong exception: {type(exc).__name__})")
        return
    check(False, msg + " (no exception raised)")


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS & AUTHORITATIVE BINDINGS
# ══════════════════════════════════════════════════════════════════════

section("1. Module constants & authoritative bindings")

inv(MODEL_ID == "altman_native", "model identity pinned")
inv(RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904",
    "release identity pinned")
inv(FEATURE_VERSION == "v1" and ML_FEATURE_VERSION == "v1",
    "domain feature version pinned to v1")
inv(NATIVE_FEATURE_VERSION == "v1", "native feature version pinned")
check(len(ML_FEATURE_ORDER) == 21, "21 domain features")
check(len(ALTMAN_NATIVE_FEATURES) == 48, "48 native features")
inv(CANONICAL_TRANSFORMATION == "map_raw_to_native",
    "canonical transformation identity")
check(callable(map_raw_to_native), "canonical mapping importable")
inv(PRODUCTION_THRESHOLD == 0.018758, "production threshold 0.018758")
inv(PRODUCTION_THRESHOLD == EXPECTED_PRODUCTION_THRESHOLD
    == EXEC_PINNED_THRESHOLD == PROTO.PRODUCTION_THRESHOLD,
    "every threshold source agrees on 0.018758")
check(EXPECTED_SHA256.startswith("76274b691b16"),
      "Phase 107 SHA-256 prefix matches the spec")
check(len(EXPECTED_SHA256) == 64, "full 64-hex digest recorded")
check(EXPECTED_ROWS == 284_807 and EXPECTED_FRAUD == 492
      and EXPECTED_LEGIT == 284_315, "row/fraud/legit counts pinned")
check(EXPECTED_FRAUD + EXPECTED_LEGIT == EXPECTED_ROWS,
      "class counts reconcile to the row count")
check(EXPECTED_COLUMNS == 31 and len(EXPECTED_SCHEMA) == 31,
      "31-column schema pinned")
check(EXPECTED_SCHEMA[0] == "Time" and EXPECTED_SCHEMA[-2:] == ("Amount",
                                                                "Class"),
      "schema order: Time ... Amount, Class")
check(tuple(f"V{i}" for i in range(1, 29)) == EXPECTED_SCHEMA[1:29],
      "V1..V28 contiguous in the pinned schema")
check(EXPECTED_INGESTION_STATUS == "acquired_verified",
      "ingestion status pinned")
check(EXECUTION_VERSION == "phase108_execution_v1"
      and MANIFEST_VERSION == "phase108_evaluation_manifest_v1"
      and REPORT_VERSION == "phase108_report_v1", "version strings pinned")
check(MANIFEST_CREATED_AT == REPORT_CREATED_AT
      == "2026-09-23T00:00:00+00:00", "fixed creation timestamps")
check(RESULTS_NAMESPACE == "reports/external_benchmark/phase108/ulb",
      "dedicated results namespace")
check(SWEEP_THRESHOLDS == (0.0, 0.005, 0.018758, 0.05, 0.5),
      "sweep grid fixed")
check(SWEEP_LABELS == ("DIAGNOSTIC_ONLY", "NON_PRODUCTION",
                       "NO_THRESHOLD_CHANGE"), "sweep labels fixed")
check("descriptive analysis only" in SWEEP_STATEMENT
      and "does not constitute threshold selection" in SWEEP_STATEMENT,
      "sweep statement carries the mandated sentence")
check("0.1727" in IMBALANCE_WARNING and "99.83" in IMBALANCE_WARNING,
      "imbalance warning explains the accuracy trap")
check(EXC.CALIBRATION_STATUS == "NOT_APPLICABLE", "calibration N/A pinned")
check(AMOUNT_BUCKET_EDGES == (0.0, 10.0, 50.0, 100.0, 500.0, 1000.0,
                              5000.0, float("inf")),
      "amount bucket edges fixed")
check(NATIVE_MODEL_EVALUATION == "NOT_APPLICABLE",
      "native evaluation stays NOT_APPLICABLE")
check(len(EXC.NATIVE_MODEL_EVALUATION_REASONS) >= 4
      and all(EXC.NATIVE_MODEL_EVALUATION_REASONS),
      "native-evaluation reasons stated concretely")
check(len(RULE_FEATURE_UNAVAILABILITY) == 17, "17 rule features declared")
inv(len(PRODUCTION_ARTIFACT_PATHS) == 12, "12 production artifacts watched")
for rel in PRODUCTION_ARTIFACT_PATHS:
    check(_artifact_path(rel).is_file(), f"artifact exists: {rel}")
inv(EXPECTED_SCHEMA_HASH == "eb5a7f0652502a72971b81fb206b7f4aadc87247"
    "37f5368d6390f9d057ac3b9a", "registered schema hash pinned")

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — PINNED SIGNATURES & BYPASS-PARAMETER REJECTION
# ══════════════════════════════════════════════════════════════════════

section("2. Pinned signatures & bypass-parameter rejection")

EXC_SIGNATURES = {
    "execute_evaluation": ("dataset_path",),
    "resolve_benchmark_file": ("dataset_path",),
    "verify_source_integrity": ("path",),
    "snapshot_artifacts": (),
    "compare_artifacts": ("before", "after"),
    "run_input_validation_probes": (),
    "probe_metrics": ("outcomes",),
    "evaluation_metrics": ("y_true", "y_scores", "threshold"),
    "confusion_at_threshold": ("y_true", "y_scores", "threshold"),
    "write_evaluation_results": ("result", "out_dir"),
    "verify_manifest_hash": ("manifest",),
    "verify_result_hash": ("result",),
    "_resolve_results_dir": ("out_dir",),
}
REP_SIGNATURES = {
    "generate_phase108_report": ("execution",),
}
for name, expected in EXC_SIGNATURES.items():
    got = tuple(inspect.signature(getattr(EXC, name)).parameters)
    inv(got == expected, f"EXC.{name} signature pinned {got}")
for name, expected in REP_SIGNATURES.items():
    got = tuple(inspect.signature(getattr(REP, name)).parameters)
    inv(got == expected, f"REP.{name} signature pinned {got}")

# no function anywhere in either module may expose a bypass parameter
for module, alias in ((EXC, "EXC"), (REP, "REP")):
    offenders = []
    for name, obj in vars(module).items():
        if (inspect.isfunction(obj)
                and getattr(obj, "__module__", "").startswith(
                    "src.monitoring.phase108")):
            clash = (set(inspect.signature(obj).parameters)
                     & FORBIDDEN_BYPASS_PARAMETERS)
            if clash:
                offenders.append((name, sorted(clash)))
    inv(not offenders, f"{alias}: no bypass parameters exist ({offenders})")

BYPASS_FNS = [
    ("execute_evaluation", EXC.execute_evaluation),
    ("write_evaluation_results", EXC.write_evaluation_results),
    ("run_input_validation_probes", EXC.run_input_validation_probes),
    ("verify_source_integrity", EXC.verify_source_integrity),
    ("snapshot_artifacts", EXC.snapshot_artifacts),
    ("resolve_benchmark_file", EXC.resolve_benchmark_file),
    ("compare_artifacts", EXC.compare_artifacts),
    ("evaluation_metrics", EXC.evaluation_metrics),
    ("generate_phase108_report", REP.generate_phase108_report),
]
for bypass in sorted(FORBIDDEN_BYPASS_PARAMETERS):
    for fn_name, fn in BYPASS_FNS:
        expect_typeerror(lambda f=fn, b=bypass: f(**{b: True}),
                         f"{bypass}=True rejected by {fn_name}")
inv(True, "all six bypass classes rejected by every entry point")

inv(not any(n.startswith("qualify") for n in vars(EXC))
    and not any(n.startswith("qualify") for n in vars(REP)),
    "no qualification authority is defined here")
inv(not any(n.startswith("promote") for n in vars(EXC))
    and not any(n.startswith("promote") for n in vars(REP)),
    "no promotion authority is defined here")
inv("map_raw_to_native" not in vars(EXC)
    and "map_raw_to_native" not in vars(REP),
    "canonical transformation is referenced, never redefined")

# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — SECURITY SOURCE SCANS
# ══════════════════════════════════════════════════════════════════════

section("3. Security source scans (offline, no-eval, no-gate, no-fabrication)")

# Mirrors the Phase 107 list with two documented deviations:
#  - "model artifact path" (r"models/production") is DROPPED because
#    Phase 108 is REQUIRED to hash production artifacts read-only for
#    the before/after integrity check (§12 proves all hashes equal);
#  - "retrain" narrows to the call form r"\bretrain\s*\(" because the
#    report's structural security field is literally named
#    "retrain_fit_tune" and prose mentions retrain/fit/tune.
FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("pickle use", r"pickle\.(load|loads)|import pickle"),
    ("joblib use", r"joblib\.load|import joblib"),
    ("urllib", r"\burllib\b"),
    ("requests lib", r"\brequests\b"),
    ("httpx", r"\bhttpx\b"),
    ("aiohttp", r"\baiohttp\b"),
    ("socket", r"\bsocket\b"),
    ("subprocess", r"\bsubprocess\b"),
    ("url openers", r"urlretrieve|urlopen"),
    ("curl", r"\bcurl\b"),
    ("wget", r"\bwget\b"),
    ("model fit", r"\.fit\s*\("),
    ("tabular loader", r"read_csv|read_table|read_excel|read_parquet"),
    ("eval call", r"\beval\s*\("),
    ("exec call", r"\bexec\s*\("),
    ("dunder import", r"__import__"),
    ("os.system", r"os\.system"),
    ("os.popen", r"os\.popen"),
    ("os.environ", r"os\.environ"),
    ("getenv", r"getenv"),
    ("extract-all", r"extractall"),
    ("tarfile", r"\btarfile\b"),
    ("session creation", r"create_session"),
    ("session run", r"run_session"),
    ("e2e run", r"execute_e2e"),
    ("promotion evaluation", r"evaluate_promotion"),
    ("promotion assert", r"assert_promotion_allowed"),
    ("audit append", r"append_audit_event"),
    ("rwv execution module", r"rwv_execution"),
    ("promotion evidence module", r"rwv_promotion_evidence"),
    ("promotion gate module", r"promotion_gate"),
    ("provider evidence module", r"provider_evidence"),
    ("promotion token", r"PromotionToken"),
    ("rwv evidence class", r"RWVPromotionEvidence"),
    ("qualification state literal",
     r"DATASET_QUALIFIED_FOR_CONTROLLED_RWV"),
    ("retrain call", r"\bretrain\s*\("),
    ("promote call", r"\bpromote\s*\("),
    ("train call", r"\btrain\s*\("),
    ("release manifest path", r"release_manifest"),
    ("network download call", r"\.download\s*\("),
    ("secrets module", r"secrets\."),
    ("credentials access", r"credential[s]?\s*[:=]"),
    ("native ensemble call", r"FusionEngine|predict_many|predict_raw_many"),
)


def _read_source(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


SOURCE_TEXTS = {
    "execution": _read_source(EXC.__file__),
    "report": _read_source(REP.__file__),
}
for label, pattern in FORBIDDEN_PATTERNS:
    for source_name, text in SOURCE_TEXTS.items():
        check(re.search(pattern, text) is None,
              f"{source_name} contains no {label}")
inv(True, f"{len(FORBIDDEN_PATTERNS)} forbidden patterns x 2 sources")

for source_name, text in SOURCE_TEXTS.items():
    check("import socket" not in text and "from socket" not in text,
          f"{source_name}: no socket import")
    check("import subprocess" not in text,
          f"{source_name}: no subprocess import")

# ══════════════════════════════════════════════════════════════════════
# SHARED REAL EXECUTION (one ingest + one execution reused everywhere)
# ══════════════════════════════════════════════════════════════════════

section("4. Dataset integrity, real execution & Mode A baseline audit")

REC = ingest_dataset(EXC.DATASET_ID)
EXEC = execute_evaluation()
METRIC = dict(EXEC.metric_block)
BASE = EXEC.baseline_audit
COMPONENTS = {c.name: c for c in EXEC.components}
REPORT = generate_phase108_report(EXEC)

# --- Phase 107 record reuse ------------------------------------------
inv(REC.status == EXPECTED_INGESTION_STATUS, "record: acquired_verified")
inv(REC.verified is True, "record: verified flag")
inv(REC.row_count == EXPECTED_ROWS, "record row count 284807")
inv(REC.label_counts == (("0", EXPECTED_LEGIT), ("1", EXPECTED_FRAUD)),
    "record label counts")
inv(set(REC.schema_expected) == set(EXPECTED_SCHEMA)
    and len(REC.schema_expected) == 31,
    "record expected schema: same 31-column set as the file schema")
inv(REC.schema_actual == EXPECTED_SCHEMA, "record actual schema == expected")
inv(REC.missing_columns == () and REC.renamed_columns == ()
    and REC.extra_columns == (), "no missing/renamed/extra columns")
inv(REC.type_mismatch_counts == () and REC.ragged_rows == 0,
    "no type mismatches, no ragged rows")
inv(REC.row_count_check == "match" and REC.fraud_count_check == "match",
    "record count checks match")
check(REC.file_records and REC.file_records[0].sha256 == EXPECTED_SHA256,
      "record file digest == pinned Phase 107 digest")

# --- execution identity + integrity ----------------------------------
inv(EXEC.integrity_result == "VERIFIED", "execution integrity VERIFIED")
inv(EXEC.dataset_sha256 == EXPECTED_SHA256, "execution binds Phase 107 hash")
inv(EXEC.row_count == EXPECTED_ROWS and EXEC.fraud_count == EXPECTED_FRAUD,
    "execution pins row/fraud counts")
inv(round(EXEC.fraud_prevalence, 6)
    == round(EXPECTED_FRAUD / EXPECTED_ROWS, 6), "prevalence reconciles")
inv(EXEC.schema_hash == EXPECTED_SCHEMA_HASH, "schema hash matches registry")
inv(EXEC.dataset_status == EXPECTED_INGESTION_STATUS,
    "dataset status carried as acquired_verified")
inv(EXEC.phase == 108 and EXEC.execution_version == EXECUTION_VERSION,
    "execution phase/version pinned")
inv(EXEC.modes_executed == (
    EvaluationMode.MODE_A_BASELINE_DATASET_AUDIT.value,
    EvaluationMode.MODE_B_COMPONENT_EVALUATION.value),
    "only Mode A and Mode B execute")

# --- file location + hash gate ---------------------------------------
real_path = resolve_benchmark_file()
inv(real_path.is_file() and real_path.name == "creditcard.csv",
    "benchmark file resolves to the real local creditcard.csv")
inv(verify_source_integrity(real_path) == EXPECTED_SHA256,
    "correct SHA-256 accepted")
with tempfile.TemporaryDirectory() as tmp:
    forged = os.path.join(tmp, "creditcard.csv")
    with open(forged, "w", encoding="utf-8") as fh:
        fh.write("Time,V1,Class\n0,0.0,0\n")
    expect_failclosed(
        lambda: verify_source_integrity(Path(forged)),
        "changed SHA-256 rejected (DatasetIntegrityError)")
    expect_failclosed(
        lambda: execute_evaluation(dataset_path=forged),
        "execute refuses a forged file before any metric")
expect_failclosed(
    lambda: resolve_benchmark_file(os.path.join(REPO_ROOT, "nope.csv")),
    "absent dataset path rejected")
inv(not EXC.verify_source_integrity.__doc__.count("relax")
    and "Phase 107" in EXC.verify_source_integrity.__doc__,
    "hash gate documents its Phase 107 binding")

# --- Mode A pins ------------------------------------------------------
inv(BASE.total_rows == EXPECTED_ROWS, "baseline: total rows")
inv(BASE.fraud_rows == EXPECTED_FRAUD and BASE.non_fraud_rows == EXPECTED_LEGIT,
    "baseline: fraud/non-fraud split")
inv(BASE.fraud_pct == 0.172749, "baseline: fraud percentage 0.172749")
inv(BASE.missing_total == 0 and BASE.missing_by_column == (),
    "baseline: zero missing values")
inv(BASE.class_distribution == (("0", EXPECTED_LEGIT), ("1", EXPECTED_FRAUD)),
    "baseline: class distribution")
inv(BASE.duplicate_findings == dataclasses.astuple(REC.duplicate_findings),
    "baseline: duplicate statistics REUSED from the Phase 107 record")
inv(BASE.duplicate_findings
    == (284807, 282953, 1854, 0, 6940, 3129),
    "baseline: exact six-field duplicate findings")
inv(BASE.verification_warnings == ("leakage_fields_present",),
    "baseline: Phase 107 warnings carried verbatim")
inv(BASE.unmapped_labels == 0 and BASE.ragged_rows == 0
    and BASE.type_mismatch_total == 0,
    "baseline: no unmapped labels, no ragged rows, no type mismatches")
inv(BASE.amount == {"count": 284807.0, "min": 0.0, "max": 25691.16,
                    "mean": 88.349619, "median": 22.0, "p95": 365.0},
    "baseline: amount statistics measured")
inv(BASE.time_stats == {"min": 0.0, "max": 172792.0, "mean": 94813.859575,
                        "calendar_conversion": 0.0},
    "baseline: relative Time statistics, no calendar conversion")

# --- baseline fail-closed constructors --------------------------------
expect_valueerror(
    lambda: dataclasses.replace(BASE, total_rows=EXPECTED_ROWS + 1),
    "baseline rejects a wrong row count")
expect_valueerror(
    lambda: dataclasses.replace(BASE, fraud_rows=EXPECTED_FRAUD + 1),
    "baseline rejects a wrong fraud count")
expect_valueerror(
    lambda: dataclasses.replace(BASE, unmapped_labels=1),
    "baseline rejects unmapped labels")
expect_valueerror(
    lambda: dataclasses.replace(BASE, missing_total=7),
    "baseline rejects inconsistent missing counts")
expect_valueerror(
    lambda: dataclasses.replace(BASE, fraud_pct=50.0),
    "baseline rejects a drifted fraud percentage")
inv(True, "every baseline tamper fails closed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — METRIC AUTHORITY PARITY
# ══════════════════════════════════════════════════════════════════════

section("5. Metric parity against real_world_evaluation_protocol.compute_metrics")

AUTHORITY_KEYS = {
    "alert_rate", "brier_score", "f1", "fn", "fnr", "fp", "fpr",
    "fraud_capture_rate", "pr_auc", "precision", "recall", "roc_auc",
    "specificity", "threshold", "tn", "total", "total_fraud", "tp",
}
MY_EXTRAS = {
    "accuracy", "balanced_accuracy", "class_imbalance", "mcc",
    "support_fraud", "support_legit", "threshold_source",
}
inv(set(compute_metrics([0, 1], [0.0, 1.0], threshold=0.5))
    == AUTHORITY_KEYS, "authority key set pinned")

PARITY_FIXTURES = [
    ("mixed_n10", [0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
     [0.0, 0.0, 0.2, 0.4, 0.0, 0.9, 0.7, 0.0, 0.0, 0.0], 0.5),
    ("constant_prod_thr", [0, 1, 0, 1, 1, 0],
     [0.0] * 6, 0.018758),
    ("all_alert", [1, 0, 1, 0, 1, 0, 1, 0],
     [1.0] * 8, 0.5),
    ("ties", [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
     [0.5] * 12, 0.5),
    ("no_fraud", [0] * 10,
     [i / 10 for i in range(10)], 0.5),
    ("all_fraud", [1] * 10,
     [i / 10 for i in range(10)], 0.5),
    ("deterministic_200",
     [1 if i % 7 == 0 else 0 for i in range(200)],
     [((i * 37) % 101) / 100.0 for i in range(200)], 0.5),
]
for label, y, s, thr in PARITY_FIXTURES:
    auth = compute_metrics(y, s, threshold=thr)
    mine = evaluation_metrics(y, s, threshold=thr)
    equal = True
    for k, v in auth.items():
        ok = mine.get(k) == v
        equal = equal and ok
        check(ok, f"parity[{label}].{k}: auth={v} mine={mine.get(k)}")
    check(set(mine) - set(auth) == MY_EXTRAS,
          f"parity[{label}]: documented extras only")
    inv(equal, f"parity exact on fixture {label}")
check(evaluation_metrics([], [], threshold=0.5) == {"error": "no_data"},
      "empty input yields the explicit no_data marker")
expect_valueerror(
    lambda: evaluation_metrics([0, 1], [0.0], threshold=0.5),
    "misaligned score vector rejected")

# --- the ULB block itself ---------------------------------------------
check(METRIC["tp"] == 0 and METRIC["fp"] == 0, "ULB: tp=0 fp=0 (fail-closed)")
check(METRIC["fn"] == EXPECTED_FRAUD, "ULB: every fraud row is FN")
check(METRIC["tn"] == EXPECTED_LEGIT, "ULB: every legit row is TN")
check(METRIC["precision"] == 0.0 and METRIC["recall"] == 0.0
      and METRIC["f1"] == 0.0, "ULB: precision/recall/F1 are 0.0")
check(METRIC["specificity"] == 1.0 and METRIC["fpr"] == 0.0
      and METRIC["fnr"] == 1.0, "ULB: specificity/FPR/FNR measured")
check(METRIC["accuracy"] == 0.998273, "ULB: raw accuracy 0.998273")
check(METRIC["balanced_accuracy"] == 0.5, "ULB: balanced accuracy 0.5")
check(METRIC["mcc"] == 0.0, "ULB: MCC 0.0 (no predictions)")
check(METRIC["roc_auc"] == 0.5, "ULB: ROC-AUC 0.5 (constant scores)")
check(METRIC["pr_auc"] == 0.000864, "ULB: PR-AUC 0.000864")
check(METRIC["brier_score"] == 0.001727, "ULB: Brier == prevalence")
check(METRIC["fraud_capture_rate"] == 0.001727
      and METRIC["alert_rate"] == 0.0, "ULB: capture/alert rates")
check(METRIC["threshold"] == 0.018758, "ULB: production threshold used")
check(METRIC["total"] == EXPECTED_ROWS and METRIC["total_fraud"] == EXPECTED_FRAUD,
      "ULB: totals reconcile")
check(METRIC["support_legit"] == EXPECTED_LEGIT
      and METRIC["support_fraud"] == EXPECTED_FRAUD, "ULB: supports")
inv(METRIC["class_imbalance"]["fraud_rate"] is not None
    or "fraud" in METRIC["class_imbalance"], "class-imbalance block present")
inv("PRODUCTION_THRESHOLD" in METRIC["threshold_source"],
    "metric block cites the locked threshold source")

# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — THRESHOLD POLICY & DIAGNOSTIC SWEEP
# ══════════════════════════════════════════════════════════════════════

section("6. Threshold policy & diagnostic-only sweep")

check(evaluation_metrics([0, 1], [0.0, 1.0], threshold=0.5)["threshold"]
      == 0.5, "diagnostic call at an arbitrary threshold works")
inv(PROTO.PRODUCTION_THRESHOLD == 0.018758,
    "authority module threshold untouched by any call above")
inv(EXEC.threshold_used == 0.018758,
    "primary analysis uses the production threshold")
inv(EXEC.threshold_unchanged is True, "threshold immutability flag set")
inv(dict(EXEC.metric_block)["threshold"] == 0.018758,
    "metric block threshold == 0.018758")
inv(EXEC.artifact_integrity.threshold_before
    == EXEC.artifact_integrity.threshold_after == 0.018758,
    "before/after threshold reads agree")

sw = EXEC.sweep
inv(sw.labels == SWEEP_LABELS, "sweep labels carried verbatim")
inv(sw.statement == SWEEP_STATEMENT, "sweep statement carried verbatim")
inv(sw.production_threshold == 0.018758, "sweep reports the locked threshold")
inv(tuple(r[0] for r in sw.rows) == SWEEP_THRESHOLDS,
    "sweep rows follow the fixed grid")
inv(sw.rows[0] == (0.0, 0.001727, 1.0, 0.003449, 0.0, 0.000864),
    "sweep row at t=0 alerts everything (measured)")
for row in sw.rows[1:]:
    check(row[1] == 0.0 and row[2] == 0.0 and row[3] == 0.0
          and row[4] == 0.0, f"sweep row t={row[0]} stays all-negative")
for row in sw.rows:
    check(row[5] == 0.000864,
          f"sweep row t={row[0]} carries the score-property PR-AUC")
inv("score vector" in sw.pr_auc_note and "threshold" in sw.pr_auc_note,
    "PR-AUC sweep note explains threshold-independence")

expect_valueerror(
    lambda: dataclasses.replace(sw, labels=("OK",)),
    "sweep rejects altered labels")
expect_valueerror(
    lambda: dataclasses.replace(sw, statement="changed"),
    "sweep rejects an altered statement")
expect_valueerror(
    lambda: dataclasses.replace(sw, production_threshold=0.5),
    "sweep rejects a non-production threshold")
expect_valueerror(
    lambda: dataclasses.replace(sw, rows=((0.1,) + (0.0,) * 5,)),
    "sweep rejects an off-grid row")
expect_valueerror(
    lambda: dataclasses.replace(EXEC, threshold_used=0.5),
    "execution rejects a non-production threshold")
inv(True, "sweep cannot become threshold selection")

# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — MODE B COMPONENTS & INPUT-VALIDATION PROBES
# ══════════════════════════════════════════════════════════════════════

section("7. Mode B components & probes against the real validation boundary")

inv(tuple(c.name for c in EXEC.components) == (
    "input_validation", "amount_processing", "timestamp_processing",
    "class_label_handling", "rules_engine_fail_closed",
    "decision_infrastructure", "data_quality_enforcement",
    "audit_trace_generation"), "eight components in declared order")
for comp in EXEC.components:
    check(comp.status in {"evaluated", "not_applicable"},
          f"{comp.name}: valid status")
    check(isinstance(comp.input_features, tuple),
          f"{comp.name}: input features declared (tuple)")
    check(bool(comp.limitations), f"{comp.name}: limitations declared")
    check(bool(comp.metric_scope), f"{comp.name}: metric scope declared")
    if comp.status == "evaluated":
        check(bool(comp.findings), f"{comp.name}: findings stated")
    else:
        check(not comp.metrics,
              f"{comp.name}: not-applicable carries no metrics")
inv(sum(1 for c in EXEC.components if c.status == "evaluated") == 7
    and sum(1 for c in EXEC.components if c.status == "not_applicable") == 1,
    "7 evaluated + 1 not-applicable (audit/trace never written)")

expect_valueerror(
    lambda: dataclasses.replace(COMPONENTS["amount_processing"],
                                status="unknown"),
    "component rejects an unknown status")
expect_valueerror(
    lambda: dataclasses.replace(COMPONENTS["amount_processing"],
                                limitations=()),
    "component rejects empty limitations")
expect_valueerror(
    lambda: dataclasses.replace(COMPONENTS["audit_trace_generation"],
                                metrics=(("tp", 1),)),
    "not-applicable component rejects metrics")
inv(True, "component structure guards fail closed")

# --- probes ------------------------------------------------------------
outs = run_input_validation_probes()
inv(len(outs) == len(PROBE_SPECS) == 22, "22 probes executed")
inv(all(exp == act for _p, exp, act in outs),
    "every probe behaves exactly as its contract documents")
inv(EXEC.probe_mismatches == 0, "execution records zero probe mismatches")
out_map = {p: (e, a) for p, e, a in outs}
for rejected in ("nan_amount_ratio", "inf_amount_ratio", "neg_amount_ratio",
                 "hour_24", "hour_neg1", "unusual_2", "escalation_11",
                 "overflow_1e309", "str_garbage", "short_event_id",
                 "long_event_id", "bad_fraud_id", "missing_required_field",
                 "nan_amount", "neg_amount", "mule_ring_1p5",
                 "shared_device_neg"):
    check(out_map[rejected] == (False, False),
          f"probe {rejected} rejected as expected")
for accepted in ("valid_base", "huge_finite_1e308", "str_numeric",
                 "label_and_pca_injection", "duplicate_event_id"):
    check(out_map[accepted] == (True, True),
          f"probe {accepted} accepted as expected")

# probe metrics recomputed independently (probe_metrics applies the
# documented decision-threshold override; confusion is pinned below)
probe_y = [0 if e else 1 for _p, e, _a in outs]
probe_s = [0.0 if a else 1.0 for _p, _e, a in outs]
recomputed = EXC.probe_metrics(outs)
manual = evaluation_metrics(probe_y, probe_s, threshold=0.5)
for k, v in manual.items():
    if k != "threshold_source":
        check(recomputed.get(k) == v,
              f"probe_metrics.{k} matches the manual recomputation")
embedded = dict(COMPONENTS["input_validation"].metrics)
for k, v in recomputed.items():
    check(embedded.get(k) == v,
          f"probe metrics.{k} recomputed identically")
check((embedded["tp"], embedded["fp"], embedded["fn"], embedded["tn"])
      == (17, 0, 0, 5), "probe confusion: 17 invalid rejected, 5 valid kept")
check(embedded["precision"] == 1.0 and embedded["recall"] == 1.0
      and embedded["f1"] == 1.0 and embedded["accuracy"] == 1.0,
      "probe conformance rates all 1.0")
check(embedded["threshold"] == 0.5,
      "probe block uses its documented decision threshold 0.5")
check("probe conformance" in embedded["threshold_source"],
      "probe threshold_source distinguishes itself from production")
missing_payload = EXC._base_payload()
del missing_payload["features"]["amount_ratio"]
inv(_validate(missing_payload) is False,
    "_validate helper: payload missing a required field rejected")
inv(_validate(EXC._base_payload()) is True,
    "_validate helper: the documented base payload accepted")

# --- feature boundary ---------------------------------------------------
pca_like = {"Class"} | {f"V{i}" for i in range(1, 29)}
inv(not (set(ML_FEATURE_ORDER) & pca_like),
    "no PCA/Class column exists in the 21-feature contract")
inv(not (set(FeatureVector.model_fields) & pca_like),
    "no PCA/Class column exists on the runtime FeatureVector")
inv(EXEC.probe_mismatches == 0
    and "Class" not in FeatureVector.model_fields,
    "Class is the label, never a feature")

# --- rule-input walk (real rules.yaml) ---------------------------------
def _walk_features(node, acc: set) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "feature" and isinstance(value, str):
                acc.add(value)
            else:
                _walk_features(value, acc)
    elif isinstance(node, list):
        for item in node:
            _walk_features(item, acc)


with open(os.path.join(BACKEND, "src", "risk_engine", "rules.yaml"),
          encoding="utf-8") as fh:
    RULES_CFG = yaml.safe_load(fh)
rule_feats: set = set()
for rule in RULES_CFG["rules"]:
    _walk_features(rule.get("condition"), rule_feats)
    _walk_features(rule.get("conditions"), rule_feats)
_walk_features(RULES_CFG.get("velocity_limits"), rule_feats)
inv(rule_feats == set(RULE_FEATURE_UNAVAILABILITY),
    "the 17-feature unavailability table == the real rules.yaml inputs")
inv(len(rule_feats) == 17, "rules.yaml reads exactly 17 features")

rules_comp = COMPONENTS["rules_engine_fail_closed"]
inv("derivable_features=0" in rules_comp.findings,
    "zero features derivable from ULB")
inv(f"score_vector=constant_0 over {EXPECTED_ROWS} rows"
    in rules_comp.findings,
    "constant fail-closed score vector recorded")
inv("distinct_scores=1" in rules_comp.findings,
    "exactly one distinct score")
inv("0 of 17 rule features" in " ".join(rules_comp.limitations),
    "rules limitation states 0/17 explicitly")

# --- decision banding ---------------------------------------------------
band_findings = [f for f in COMPONENTS["decision_infrastructure"].findings
                 if f.startswith("band[")]
band_total = sum(int(re.search(r"=(\d+)$", f).group(1))
                 for f in band_findings)
inv(band_total == EXPECTED_ROWS, "band counts cover every row")
expected_band_prefix = (
    f"band_of({round(100.0 * 0.0):g}) -> {band_of(0)}")
inv(any(f.startswith(expected_band_prefix)
        for f in COMPONENTS["decision_infrastructure"].findings),
    "band_of(0) finding recorded dynamically")

# --- data quality / amount / timestamp / class findings -----------------
dq = " ".join(COMPONENTS["data_quality_enforcement"].findings)
inv("missing_cells=0" in dq, "data quality: zero missing cells")
inv("within_source_duplicates=1854, cross_source=0" in dq,
    "data quality: Phase 107 duplicate statistics reused")
ap = " ".join(COMPONENTS["amount_processing"].findings)
inv(f"rows_checked={EXPECTED_ROWS}" in ap, "amount: all rows checked")
inv("constraint_violations=0" in ap, "amount: no constraint violations")
tsf = " ".join(COMPONENTS["timestamp_processing"].findings)
inv("calendar conversion: none" in tsf, "timestamp: no calendar conversion")
cl = " ".join(COMPONENTS["class_label_handling"].findings)
inv("unmapped_labels=0" in cl, "class labels: all mapped")
inv("never mapped" in cl and "PCA MAPPING DETECTED" not in cl,
    "class handling: PCA never mapped")
inv("492" in cl, "class handling: fraud count visible")

# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — MODE C BOUNDARY (native evaluation NOT_APPLICABLE)
# ══════════════════════════════════════════════════════════════════════

section("8. Mode C boundary: native 48-feature evaluation stays forbidden")

inv(EXEC.native_model_evaluation == NATIVE_MODEL_EVALUATION == "NOT_APPLICABLE",
    "native_model_evaluation = NOT_APPLICABLE")
inv(EXEC.mode_c_status == "NOT_APPLICABLE", "Mode C status NOT_APPLICABLE")
inv(EXEC.group == "group_b", "ULB stays Group B")
inv(EXEC.scope == "component_evaluation_only", "scope stays component-only")
inv(EXEC.native_eligible_rows == 0, "zero native-eligible rows")
inv(EXEC.usable_native_features == ("amt", "log_amt", "amt_sq", "high_amt",
                                    "very_high_amt"),
    "five usable native features (measured by Phase 106)")
inv(EXEC.usable_native_features
    == usable_features(dataset_preflight(EXC.DATASET_ID)),
    "usability is the Phase 106 preflight authority")
inv(assign_benchmark_group(EXC.DATASET_ID)[0] == "group_b",
    "Phase 106 grouping still group_b")
inv(evaluation_scope(EXC.DATASET_ID) == "component_evaluation_only",
    "Phase 106 scope still component-only")
inv(native_eligible_row_count(EXC.DATASET_ID) == 0,
    "Phase 106 native eligibility still zero")
inv(EXEC.usable_native_features != tuple(ALTMAN_NATIVE_FEATURES)
    and len(EXEC.usable_native_features) == 5,
    "5 != 48: Group B never becomes Group A here")
exec_text = _read_source(EXC.__file__)
inv("FusionEngine" not in exec_text and "predict_many" not in exec_text,
    "execution module never touches the native ensemble")
expect_valueerror(
    lambda: dataclasses.replace(EXEC, native_model_evaluation="APPLICABLE"),
    "execution rejects an applicable native evaluation")
expect_valueerror(
    lambda: dataclasses.replace(EXEC, native_eligible_rows=1),
    "execution rejects any native-eligible row")
inv(True, "Mode C boundary fail-closed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════

section("9. Temporal analysis (relative Time, no calendar claims)")

temp = EXEC.temporal
inv(temp.timestamp_status == "ordering_only_relative",
    "timestamp status: ordering only")
inv(temp.calendar_conversion == "none", "no calendar conversion performed")
inv(temp.ordering_deterministic is True, "ordering deterministic")
inv(sum(s.rows for s in temp.slices) == EXPECTED_ROWS,
    "slices cover every row")
inv(sum(s.fraud for s in temp.slices) == EXPECTED_FRAUD,
    "slice fraud counts reconcile")
check([s.rows for s in temp.slices] == [94935, 94935, 94937],
      "tertile sizes measured")
check([s.fraud for s in temp.slices] == [217, 153, 122],
      "tertile fraud distribution measured")
check([s.name for s in temp.slices] == ["early", "middle", "late"],
      "slice names ordered")
early, middle, late = temp.slices
inv(early.time_max <= middle.time_min
    and middle.time_max <= late.time_min,
    "time ranges non-decreasing across slices (ties allowed)")
inv(early.time_min == 0.0 and late.time_max == 172792.0,
    "global time range spans the file")
for s in temp.slices:
    check(0.0 < s.fraud_rate <= 0.01,
          f"{s.name}: fraud rate within (0, 1%]")
    check(s.time_min <= s.time_max, f"{s.name}: time range ordered")
inv("tertiles" in temp.split_method and "relative" in temp.split_method
    and "no calendar period" in temp.split_method,
    "split method documents construction + relative seconds")
inv("no future information leaks" in temp.future_leakage_note,
    "future-leakage note stated")
serialized = canonical_json(dataclasses.asdict(temp))
check(re.search(r"\d{4}-\d{2}-\d{2}", serialized) is None,
      "no calendar dates anywhere in the temporal analysis")
expect_valueerror(
    lambda: dataclasses.replace(temp, calendar_conversion="utc"),
    "temporal rejects a calendar conversion")
expect_valueerror(
    lambda: dataclasses.replace(temp, timestamp_status="absolute"),
    "temporal rejects an absolute timestamp claim")
expect_valueerror(
    lambda: dataclasses.replace(temp, slices=temp.slices[:2]),
    "temporal rejects slices that do not cover every row")
inv(True, "temporal guards fail closed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════

section("10. Error analysis (aggregates only, reconciled)")

err = EXEC.error_analysis
inv((err.tp, err.fp, err.fn, err.tn)
    == (METRIC["tp"], METRIC["fp"], METRIC["fn"], METRIC["tn"]),
    "error analysis reconciles with the metric block")
inv(err.fp == 0 and err.fn == EXPECTED_FRAUD,
    "fail-closed: no FPs, all fraud is FN")
inv(err.fn_amount == err.fraud_amount,
    "FN amount distribution == fraud amount distribution")
inv(err.fn_amount["count"] == float(EXPECTED_FRAUD),
    "FN amount count == 492")
inv(err.legit_amount["count"] == float(EXPECTED_LEGIT),
    "legit amount count == 284315")
inv(err.fp_amount == "NOT_APPLICABLE",
    "FP amount stats: NOT_APPLICABLE, not invented")
inv([b.label for b in err.amount_buckets]
    == ["[0, 10)", "[10, 50)", "[50, 100)", "[100, 500)",
        "[500, 1000)", "[1000, 5000)", "[5000, inf)"],
    "seven amount buckets with fixed labels")
check([b.rows for b in err.amount_buckets]
      == [97314, 92390, 37718, 47893, 6423, 3014, 55],
      "amount-bucket row counts measured")
check([b.fraud for b in err.amount_buckets]
      == [249, 56, 57, 95, 26, 9, 0], "amount-bucket fraud counts measured")
inv(sum(b.rows for b in err.amount_buckets) == EXPECTED_ROWS,
    "amount buckets cover every row")
inv(sum(b.fraud for b in err.amount_buckets) == EXPECTED_FRAUD,
    "amount-bucket fraud reconciles")
for b in err.amount_buckets:
    check(b.tp + b.fp + b.fn + b.tn == b.rows,
          f"bucket {b.label}: confusion reconciles")
    check(b.fn == b.fraud and b.fp == 0,
          f"bucket {b.label}: fail-closed confusion")
inv(sum(b.rows for b in err.temporal_buckets) == EXPECTED_ROWS
    and sum(b.fraud for b in err.temporal_buckets) == EXPECTED_FRAUD,
    "temporal buckets reconcile")
inv([b.label for b in err.temporal_buckets] == ["early", "middle", "late"],
    "temporal buckets reuse the tertile names")
inv(err.reconciliation == (
    ("tp+fp+fn+tn==rows", True),
    ("fn==fraud_count", True),
    ("fp==0", True),
    ("fn_amount_count==fraud_count", True),
    ("amount_bucket_rows_cover", True),
), "five reconciliations, all true, none vacuous")
inv(err.identifier_exposure == "aggregate_amount_and_time_only",
    "aggregate-only exposure declared")
err_keys = set(dataclasses.asdict(err))
inv(err_keys == {"tp", "fp", "fn", "tn", "fraud_amount", "legit_amount",
                 "fn_amount", "fp_amount", "amount_buckets",
                 "temporal_buckets", "reconciliation",
                 "identifier_exposure"},
    "error analysis field whitelist exact")


def _collect_keys(node, acc: set) -> None:
    if isinstance(node, dict):
        acc.update(node)
        for v in node.values():
            _collect_keys(v, acc)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_keys(item, acc)


err_text = canonical_json(dataclasses.asdict(err))
all_keys: set = set()
_collect_keys(dataclasses.asdict(err), all_keys)
allowed_keys = err_keys | {"label", "lo", "hi", "rows", "fraud",
                           "count", "min", "max", "mean", "median", "p95"}
inv(all_keys <= allowed_keys,
    f"no identifier keys leak into the analysis ({sorted(all_keys - allowed_keys)})")
check("event_id" not in err_text and "fraud_id" not in err_text
      and '"V1"' not in err_text and "customer" not in err_text,
      "no sensitive-looking identifiers exposed")

expect_valueerror(
    lambda: dataclasses.replace(err, fn=1),
    "error analysis rejects fn != fraud count")
expect_valueerror(
    lambda: dataclasses.replace(err, fp=1),
    "error analysis rejects any false positive under constant scores")
expect_valueerror(
    lambda: dataclasses.replace(err, reconciliation=(("x", False),)),
    "error analysis rejects a failed reconciliation")
expect_valueerror(
    lambda: dataclasses.replace(err,
                                identifier_exposure="raw_rows"),
    "error analysis rejects raw-row exposure")
expect_valueerror(
    lambda: dataclasses.replace(err, amount_buckets=err.amount_buckets[:6]),
    "error analysis rejects incomplete amount buckets")
inv(True, "error-analysis guards fail closed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 11 — CONTAMINATION & FINGERPRINT FORMULA
# ══════════════════════════════════════════════════════════════════════

section("11. Source contamination: measured, never assumed")

cont = EXEC.contamination
inv(cont.train_unchanged is True, "training data unchanged across the run")
inv(cont.train_sha256_before == cont.train_sha256_after
    and cont.train_sha256_before != "absent",
    "training hash identical before/after, actually hashed")
inv(cont.benchmark_wrote_training is False,
    "benchmark never wrote into training data")
inv(cont.train_fingerprint_overlap == 0,
    "zero row-fingerprint overlap with training data")
inv(cont.ts_domains_disjoint is True,
    "timestamp domains disjoint (MEASURED per row)")
train_path = os.path.join(REPO_ROOT, "data", "transactions.csv")
inv(cont.train_sha256_before == sha256_file(train_path),
    "training hash independently reproduced")
with open(train_path, encoding="utf-8", newline="") as fh:
    train_rows = sum(1 for _ in csv.DictReader(fh))
inv(cont.train_rows == train_rows, "training row count independently counted")
feedback_path = os.path.join(REPO_ROOT, "data", "feedback_labeled.csv")
inv(cont.feedback_sha256 == sha256_file(feedback_path),
    "feedback hash independently reproduced")

# fingerprint formula equivalence: my helper == the Phase 106 authority
ulb_path = os.path.join(REPO_ROOT, "data", "creditcard.csv")
with open(ulb_path, encoding="utf-8", newline="") as fh:
    reader = csv.reader(fh)
    header = next(reader)
    first_row = dict(zip(header, next(reader)))
hrows = harmonize_rows(EXC.DATASET_ID, [first_row])
inv(len(hrows) == 1, "first ULB row harmonized")
mine_fp = _row_fingerprint(
    int(normalize_label(EXC.DATASET_ID, first_row["Class"])),
    str(first_row["Time"]), str(first_row["Amount"]))
inv(row_fingerprint(hrows[0]) == mine_fp,
    "row-fingerprint formula == the Phase 106 authority (live row)")

expect_valueerror(
    lambda: dataclasses.replace(cont,
                                train_sha256_after="0" * 64),
    "contamination rejects a changed training hash")
expect_valueerror(
    lambda: dataclasses.replace(cont, train_unchanged=False),
    "contamination rejects an unchanged=False claim")
expect_valueerror(
    lambda: dataclasses.replace(cont, benchmark_wrote_training=True),
    "contamination rejects benchmark writes into training data")
inv(True, "contamination guards fail closed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 12 — PRODUCTION ARTIFACT INTEGRITY (read-only hashing)
# ══════════════════════════════════════════════════════════════════════

section("12. Production artifact integrity before/after (12/12 hashed)")

ai = EXEC.artifact_integrity
inv(len(ai.entries) == len(PRODUCTION_ARTIFACT_PATHS) == 12,
    "twelve artifact entries")
for rel, before, after in ai.entries:
    check(before != "absent", f"artifact actually hashed: {rel}")
    check(before == after, f"artifact unchanged: {rel}")
    check(len(before) == 64, f"artifact digest is sha256: {rel}")
inv({p for p, _b, _a in ai.entries} == set(PRODUCTION_ARTIFACT_PATHS),
    "entry paths == the watched set")
inv(ai.unchanged is True, "overall unchanged flag")
inv(ai.threshold_before == ai.threshold_after == 0.018758,
    "threshold reads identical")
inv(ai.threshold_unchanged is True, "threshold unchanged flag")
# independent re-hash of two entries (the entries are not lying)
for probe_rel in ("src/risk_engine/rules.yaml",
                  "models/feature_contract.json"):
    with open(_artifact_path(probe_rel), "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    recorded = dict((p, b) for p, b, _a in ai.entries)[probe_rel]
    check(digest == recorded, f"independent re-hash matches: {probe_rel}")

expect_failclosed(
    lambda: compare_artifacts(
        {rel: "a" * 64 for rel in PRODUCTION_ARTIFACT_PATHS},
        {rel: ("b" * 64 if rel == "src/risk_engine/rules.yaml"
               else "a" * 64)
         for rel in PRODUCTION_ARTIFACT_PATHS}),
    "compare_artifacts raises on any content drift")
expect_failclosed(
    lambda: dataclasses.replace(ai, unchanged=False),
    "ArtifactIntegrity rejects unchanged=False")
expect_failclosed(
    lambda: dataclasses.replace(ai, threshold_before=0.5),
    "ArtifactIntegrity rejects a drifted threshold")
inv(True, "artifact guards fail closed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 13 — MANIFEST & RESULT HASHING / TYPE IDENTITY
# ══════════════════════════════════════════════════════════════════════

section("13. Evaluation manifest & result: deterministic, tamper-evident")

mani = EXEC.manifest
inv(verify_manifest_hash(mani), "manifest hash verifies")
inv(len(mani.manifest_hash) == 64
    and re.fullmatch(r"[0-9a-f]{64}", mani.manifest_hash),
    "manifest hash is 64-hex sha256")
inv(verify_result_hash(EXEC), "result hash verifies")
inv(len(EXEC.result_hash) == 64
    and re.fullmatch(r"[0-9a-f]{64}", EXEC.result_hash),
    "result hash is 64-hex sha256")
inv(mani.dataset_sha256 == EXPECTED_SHA256
    and EXEC.manifest.dataset_sha256 == EXEC.dataset_sha256,
    "manifest binds the executed dataset")
inv(mani.model_id == MODEL_ID and mani.release_id == RELEASE_ID,
    "manifest binds production identity")
inv(mani.production_threshold == 0.018758, "manifest binds the threshold")
inv(mani.evaluation_modes == (
    EvaluationMode.MODE_A_BASELINE_DATASET_AUDIT.value,
    EvaluationMode.MODE_B_COMPONENT_EVALUATION.value),
    "manifest records Mode A + Mode B only")
inv(mani.mode_c_status == "NOT_APPLICABLE", "manifest Mode C NOT_APPLICABLE")
inv(mani.not_gate_evidence == BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
    "manifest carries the not-gate-evidence declaration")
inv(mani.row_count == EXPECTED_ROWS
    and mani.label_counts == (("0", EXPECTED_LEGIT), ("1", EXPECTED_FRAUD)),
    "manifest pins rows and labels")
inv(mani.schema_hash == EXPECTED_SCHEMA_HASH, "manifest pins the schema hash")
inv(mani.canonical_mapping.count("map_raw_to_native") == 1,
    "manifest pins the canonical mapping identity")
inv(mani.domain_feature_count == 21 and mani.native_feature_count == 48,
    "manifest pins 21 -> 48")

# tamper detection: mutate behind the constructor's back
tampered = dataclasses.replace(mani)
object.__setattr__(tampered, "row_count", mani.row_count + 1)
inv(not verify_manifest_hash(tampered), "tampered manifest detected")

for kwargs in (
        {"dataset_sha256": "0" * 64},
        {"model_id": "altman_native_v2"},
        {"production_threshold": 0.5},
        {"mode_c_status": "NOT_APPLICABLE_OK"},
        {"created_at": "2027-01-01T00:00:00+00:00"},
        {"row_count": EXPECTED_ROWS + 1},
        {"schema_hash": "0" * 64},
):
    expect_valueerror(lambda k=kwargs: dataclasses.replace(mani, **k),
                      f"manifest rejects {list(kwargs)[0]} tampering")
expect_valueerror(
    lambda: dataclasses.replace(
        mani, evaluation_modes=(EvaluationMode.MODE_A_BASELINE_DATASET_AUDIT.value,
                                EvaluationMode.MODE_B_COMPONENT_EVALUATION.value,
                                EvaluationMode.MODE_C_BENCHMARK_MODEL_EVALUATION.value)),
    "manifest rejects a Mode C execution claim")
expect_valueerror(
    lambda: dataclasses.replace(
        mani, not_gate_evidence=("something else",)),
    "manifest rejects a stripped not-gate-evidence declaration")
inv(True, "manifest guards fail closed")

# type identity: bookkeeping only, never gate evidence
inv(isinstance(mani, EvaluationManifest), "manifest has exactly its own type")
inv(not isinstance(mani, QualificationResult),
    "manifest is not a QualificationResult")
inv(not isinstance(mani, RWVPromotionEvidence),
    "manifest is not RWV promotion evidence")
inv(not isinstance(EXEC, QualificationResult)
    and not isinstance(EXEC, RWVPromotionEvidence),
    "execution result is not gate evidence of any kind")
inv(not isinstance(REPORT, QualificationResult)
    and not isinstance(REPORT, RWVPromotionEvidence),
    "report is not gate evidence of any kind")
expect_failclosed(lambda: PromotionToken(mani),
                  "PromotionToken from a manifest rejected")
expect_failclosed(
    lambda: PromotionToken(dataclasses.replace(mani)),
    "PromotionToken from an untouched-copy manifest rejected")
expect_valueerror(
    lambda: dataclasses.replace(EXEC, integrity_result="FAILED"),
    "execution rejects a non-VERIFIED integrity claim")
expect_valueerror(
    lambda: dataclasses.replace(EXEC, row_count=1),
    "execution rejects a wrong row count")
inv(True, "type identity & execution guards fail closed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 14 — REPRODUCIBILITY & RESULT STORAGE
# ══════════════════════════════════════════════════════════════════════

section("14. Reproducibility: identical rerun, dedicated namespace")

REP2 = generate_phase108_report()   # fresh execution inside
inv(REP2.execution.result_hash == EXEC.result_hash,
    "independent rerun reproduces the result hash")
inv(REP2.execution.manifest.manifest_hash == mani.manifest_hash,
    "independent rerun reproduces the manifest hash")
inv(REP2.report_hash == REPORT.report_hash,
    "independent rerun reproduces the report hash")
inv(REP2.to_dict() == REPORT.to_dict(),
    "independent rerun reproduces byte-level report content")
inv(REP2.reproducibility_status == "DETERMINISTIC_HASHES_VERIFIED",
    "reproducibility recorded as measured")
inv(REPORT.verify_hash(), "report hash verifies")
inv(len(REPORT.report_hash) == 64, "report hash is 64-hex")

with tempfile.TemporaryDirectory() as tmp:
    out_ns = os.path.join(tmp, "reports", "external_benchmark",
                           "phase108", "ulb")
    files1 = write_evaluation_results(EXEC, out_ns)
    bytes1 = [open(os.path.join(tmp, f), "rb").read() if os.path.exists(
        os.path.join(tmp, f)) else open(f, "rb").read() for f in files1]
    files2 = write_evaluation_results(EXEC, out_ns)
    bytes2 = [open(f, "rb").read() for f in files2]
    inv(files1 == files2, "writer output paths are deterministic")
    inv(bytes1 == bytes2, "writer output bytes are deterministic")
    inv(all(RESULTS_NAMESPACE in f.replace("\\", "/") for f in files2),
        "results confined to the dedicated namespace")
    parsed = json.loads(bytes2[1])
    inv(parsed["result_hash"] == EXEC.result_hash,
        "stored results carry the verified result hash")
    parsed_m = json.loads(bytes2[0])
    inv(parsed_m["manifest_hash"] == mani.manifest_hash,
        "stored manifest carries the verified manifest hash")
    expect_failclosed(
        lambda: write_evaluation_results(
            EXEC, os.path.join(tmp, "somewhere_else")),
        "writes outside the Phase 108 namespace are rejected")

# committed artifacts (written deterministically during this phase)
committed_dir = os.path.join(REPO_ROOT, "reports", "external_benchmark",
                             "phase108", "ulb")
committed_manifest = os.path.join(committed_dir, "evaluation_manifest.json")
committed_results = os.path.join(committed_dir, "evaluation_results.json")
check(os.path.isfile(committed_manifest),
      "evaluation_manifest.json stored in the namespace")
check(os.path.isfile(committed_results),
      "evaluation_results.json stored in the namespace")
if os.path.isfile(committed_manifest):
    with open(committed_manifest, encoding="utf-8") as fh:
        inv(json.load(fh)["manifest_hash"] == mani.manifest_hash,
            "stored manifest matches the verified hash")
if os.path.isfile(committed_results):
    with open(committed_results, encoding="utf-8") as fh:
        inv(json.load(fh)["result_hash"] == EXEC.result_hash,
            "stored results match the verified hash")

# ══════════════════════════════════════════════════════════════════════
# SECTION 15 — DETERMINISTIC REPORT (spec items 1-36)
# ══════════════════════════════════════════════════════════════════════

section("15. Deterministic report: states, statements, spec coverage")

inv(isinstance(REPORT, Phase108BenchmarkReport), "report has its own type")
inv(REPORT.phase == 108 and REPORT.report_version == REPORT_VERSION
    and REPORT.created_at == REPORT_CREATED_AT, "report identity pinned")
# items 1-6
inv(REPORT.dataset_id == EXC.DATASET_ID, "item: dataset identity")
inv(REPORT.dataset_sha256 == EXPECTED_SHA256, "item: dataset SHA-256")
inv(REPORT.row_count == 284807, "item: row count")
inv(REPORT.fraud_count == 492, "item: fraud count")
inv(REPORT.fraud_prevalence == 0.001727, "item: fraud prevalence")
inv(REPORT.integrity_result == "VERIFIED", "item: integrity result")
# items 7-9 headline
inv(REPORT.modes_executed == EXEC.modes_executed, "item: modes executed")
inv(REPORT.native_model_evaluation == "NOT_APPLICABLE",
    "item: native evaluation NOT_APPLICABLE")
inv(REPORT.mode_c_status == "NOT_APPLICABLE", "item: Mode C status")
# items 19-21
inv(REPORT.threshold_used == 0.018758, "item: threshold used")
inv(REPORT.production_threshold_changed is False,
    "item: production threshold NOT changed")
inv(REPORT.calibration_status == "NOT_APPLICABLE", "item: calibration N/A")
# items 26-27
inv(REPORT.benchmark_evaluation_manifest_sha256 == mani.manifest_hash,
    "item: benchmark evaluation manifest hash")
inv(REPORT.benchmark_results_hash == EXEC.result_hash,
    "item: results hash")
# states
inv(REPORT.system_readiness == SYSTEM_READINESS
    == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET", "system readiness as-is")
inv(REPORT.real_world_validation == REAL_WORLD_VALIDATION
    == "BLOCKED_PENDING_ELIGIBLE_DATASET", "RWV state as-is")
inv(REPORT.promotion_state == PROMOTION_STATE
    == "PROMOTION_GATE_REQUIRED", "promotion state as-is")
inv(REP.SYSTEM_READINESS == SYSTEM_READINESS
    and REP.REAL_WORLD_VALIDATION == REAL_WORLD_VALIDATION
    and REP.PROMOTION_STATE == PROMOTION_STATE,
    "report module re-exports read-only state")
# classification + governance
inv(REPORT.benchmark_result_state == BENCHMARK_RESULT_STATE
    == "PUBLIC_EXTERNAL_BENCHMARK_RESULTS_AVAILABLE",
    "item: results-available state (the only permitted one)")
inv(REPORT.results_classification == RESULTS_CLASSIFICATION,
    "item: public external benchmark results classification")
inv(REPORT.qualified_datasets == () and not REPORT.any_dataset_qualified,
    "no dataset qualified")
inv(REPORT.known_candidate_states == (
    ("IEEE_CIS", "blocked", False),
    ("NOVATTI", "blocked", False),
    ("WORLDLINE_ECOM_2017_NAG", "blocked", False),
    ("WORLDLINE_ONLINE_2018", "blocked", False)),
    "all four candidates reported blocked")
inv(REPORT.security_results == SECURITY_RESULTS
    and len(REPORT.security_results) == 8,
    "item: eight security results stated")
inv(REPORT.limitations == LIMITATIONS and len(REPORT.limitations) == 9,
    "item: nine limitations stated")
inv(REPORT.declarations == DECLARATIONS and len(REPORT.declarations) == 7,
    "seven declarations locked verbatim")
inv(REPORT.conclusion == EXPECTED_CONCLUSION
    == "READY_WITH_EXTERNAL_PREREQUISITE", "conclusion fixed")
inv(REPORT.not_gate_evidence == BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
    "not-gate-evidence carried")
inv(REPORT.execution is EXEC, "report embeds the exact execution object")
inv(REPORT.model_id == MODEL_ID and REPORT.release_id == RELEASE_ID
    and REPORT.production_threshold == 0.018758,
    "report locks production identity + threshold")
inv(REPORT.canonical_mapping == mani.canonical_mapping,
    "report carries the manifest mapping identity")

# spec statement phrases
decl_text = " ".join(DECLARATIONS)
for phrase in ("NOT REAL-WORLD VALIDATION",
               "NOT A NATIVE 48-FEATURE EVALUATION",
               "EXTERNAL/COMPONENT BENCHMARK EVALUATION ONLY",
               "0.018758 WAS NOT CHANGED",
               "INSTITUTIONAL/PROVIDER EVIDENCE REMAINS UNRESOLVED",
               "PROMOTION_GATE_REQUIRED",
               "BLOCKED_PENDING_ELIGIBLE_DATASET"):
    check(phrase in decl_text, f"declaration states: {phrase}")

# forbidden states never appear anywhere in the serialized report
serialized_report = json.dumps(REPORT.to_dict(), sort_keys=True)
for token in FORBIDDEN_RESULT_STATES:
    check(token not in serialized_report,
          f"forbidden state absent from report: {token}")
inv(True, "all four forbidden result states absent")

# spec items reachable through the embedded execution
for field_name in ("baseline_audit", "components", "metric_block",
                   "imbalance_warning", "temporal", "error_analysis",
                   "robustness_findings", "leakage_findings",
                   "contamination", "artifact_integrity", "sweep",
                   "manifest", "probe_mismatches"):
    check(getattr(REPORT.execution, field_name) is not None,
          f"report reaches spec field: {field_name}")
inv(bool(REPORT.execution.robustness_findings), "robustness findings stated")
inv(bool(REPORT.execution.leakage_findings), "leakage findings stated")

# report fail-closed construction
for kwargs, label in (
        ({"benchmark_result_state": "PROMOTION_ELIGIBLE"},
         "forbidden result state"),
        ({"qualified_datasets": ("X",)}, "non-empty qualified datasets"),
        ({"production_threshold_changed": True}, "threshold-change claim"),
        ({"declarations": DECLARATIONS[:1]}, "altered declarations"),
        ({"limitations": ()}, "stripped limitations"),
        ({"conclusion": "PRODUCTION_READY"}, "altered conclusion"),
        ({"dataset_sha256": "0" * 64}, "wrong dataset hash"),
        ({"threshold_used": 0.5}, "wrong reported threshold"),
        ({"security_results": ()}, "stripped security results"),
        ({"reproducibility_status": "MAYBE"}, "invalid reproducibility"),
):
    expect_valueerror(lambda k=kwargs: dataclasses.replace(REPORT, **k),
                      f"report rejects {label}")
inv(True, "report guards fail closed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 16 — RWV / PROMOTION GATES REMAIN UNCHANGED AND BLOCKING
# ══════════════════════════════════════════════════════════════════════

section("16. RWV / promotion gates remain unchanged and blocking")

closure = check_global_state()
inv(bool(closure) and all(i.verdict == "pass" for i in closure),
    f"Phase 103 closure invariants all pass ({len(closure)})")
rwv_gate = evaluate_real_world_validation()
inv(rwv_gate.status == GateStatus.BLOCKED
    and rwv_gate.gate_name == "REAL_WORLD_VALIDATION",
    "RWV gate still BLOCKED under its authoritative name")
decision = evaluate_promotion()
inv(decision.verdict == PromotionVerdict.BLOCKED,
    "promotion decision still BLOCKED")
inv("REAL_WORLD_VALIDATION" in decision.blocking_gates,
    "blocking gate lists REAL_WORLD_VALIDATION")
expect_failclosed(lambda: assert_promotion_allowed(decision),
                  "assert_promotion_allowed raises on blocked decision")
ev = KNOWN_CANDIDATES["IEEE_CIS"]
qual = qualify_dataset(ev)
elig = check_rwv_execution_eligibility(ev, qual)
inv(elig["eligible"] is False and bool(elig["blocking_gates"]),
    "RWV execution eligibility False with blocking gates")
sess = create_session(ev, qual, session_id="phase108-test")
inv(sess.status == SessionState.BLOCKED.value,
    f"RWV session created BLOCKED ({sess.status})")
expect_failclosed(lambda: PromotionToken(decision),
                  "PromotionToken from blocked decision rejected")
expect_failclosed(lambda: PromotionToken(REPORT),
                  "PromotionToken from the benchmark report rejected")
inv(MODEL_ID == "altman_native"
    and RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904",
    "production identities still authoritative")
inv(len(ALTMAN_NATIVE_FEATURES) == 48 and len(ML_FEATURE_ORDER) == 21
    and PRODUCTION_THRESHOLD == 0.018758,
    "model contract and threshold untouched by Phase 108")
inv(SYSTEM_READINESS == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET",
    "SYSTEM_READINESS unchanged")
inv(REAL_WORLD_VALIDATION == "BLOCKED_PENDING_ELIGIBLE_DATASET",
    "REAL_WORLD_VALIDATION unchanged")
inv(PROMOTION_STATE == "PROMOTION_GATE_REQUIRED",
    "PROMOTION unchanged")

# ══════════════════════════════════════════════════════════════════════
# SECTION 17 — KNOWN CANDIDATES REMAIN PHASE 104 BLOCKED
# ══════════════════════════════════════════════════════════════════════

section("17. Known institutional candidates remain Phase 104 BLOCKED")

CANDIDATE_RESULTS = evaluate_known_candidates()
inv(set(CANDIDATE_RESULTS) == set(KNOWN_CANDIDATES),
    "all four known candidates still evaluated")
for dataset_id in ("WORLDLINE_ECOM_2017_NAG", "WORLDLINE_ONLINE_2018",
                   "NOVATTI", "IEEE_CIS"):
    result = CANDIDATE_RESULTS[dataset_id]
    inv(not result.qualified and result.qualification_state == "blocked",
        f"{dataset_id}: BLOCKED and not qualified")
inv(all(not r.qualified for r in CANDIDATE_RESULTS.values()),
    "qualified_datasets stays empty without provider-attested evidence")
inv(not REPORT.any_dataset_qualified, "report agrees: nothing qualified")
inv(isinstance(REPORT.execution.manifest, EvaluationManifest),
    "evaluation manifest is bookkeeping only")
inv(REPORT.benchmark_evaluation_manifest_sha256
    == REPORT.execution.manifest.manifest_hash,
    "report pins the manifest hash (bookkeeping chain only)")

# ══════════════════════════════════════════════════════════════════════
# SECTION 18 — SELF-COUNT & SUMMARY
# ══════════════════════════════════════════════════════════════════════

section("18. Self-count")

inv(INVARIANT_COUNT >= 100,
    f"invariant floor 100+ reached ({INVARIANT_COUNT})")
inv(ASSERTIONS >= 500,
    f"assertion floor 500+ reached ({ASSERTIONS})")
inv(not FAILURES, f"zero failures ({len(FAILURES)})")

print("\n" + "=" * 70)
print(f"PHASE 108 SUITE: {ASSERTIONS} assertions, "
      f"{INVARIANT_COUNT} invariants, {len(FAILURES)} failures")
if FAILURES:
    print("FAILURES:")
    for f in FAILURES:
        print(f"  - {f}")
print("=" * 70)
sys.exit(1 if FAILURES else 0)
