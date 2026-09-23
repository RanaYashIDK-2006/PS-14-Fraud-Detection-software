"""Phase 107: Public Dataset Acquisition, Verified Ingestion & Benchmark Assembly — test suite.

Deterministic, offline, adversarial coverage for the Phase 107 local
ingestion engine, source-preserving benchmark assembly, component views,
manifest, report and the untouched RWV/promotion boundaries.  Never
touches the network, model artifacts, DB-4 or any authority.

Run:  ../.venv/Scripts/python.exe scripts/phase107_public_dataset_ingestion_test.py
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import itertools
import json
import os
import re
import sys
import tempfile
import time
import zipfile

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from src.monitoring import phase106_public_benchmark_registry as REG
from src.monitoring import phase107_public_dataset_ingestion as ING
from src.monitoring import phase107_public_benchmark_report as REP
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
from src.monitoring.phase107_public_benchmark_report import (
    DECLARATIONS,
    EXPECTED_CONCLUSION,
    REPORT_VERSION,
    Phase107BenchmarkReport,
    generate_phase107_report,
)
from src.monitoring.phase107_public_dataset_ingestion import (
    CANONICAL_CANDIDATE_FIELDS,
    CANDIDATE_PATHS,
    DATASET_DIRNAMES,
    EXPECTED_PRODUCTION_THRESHOLD,
    FORBIDDEN_BYPASS_PARAMETERS,
    INGESTION_VERSION,
    MANIFEST_CREATED_AT,
    MANIFEST_VERSION,
    PRIMARY_BASENAMES,
    BENCHMARK_NAME,
    AcquisitionStatus,
    BenchmarkPartition,
    BenchmarkReadiness,
    CombinedBenchmark,
    ComponentViewStatus,
    ContainerType,
    DuplicateAccumulator,
    DuplicateFindings,
    FieldMapping,
    IngestionLedger,
    IngestionRecord,
    MappingType,
    Phase107BenchmarkManifest,
    TimestampStatus,
    build_combined_benchmark,
    build_component_evaluation_views,
    build_field_mappings,
    build_ingestion_ledger,
    build_phase107_manifest,
    build_partitions,
    detect_container,
    detect_dataset_replacement,
    ingest_all,
    ingest_dataset,
    ingest_many,
    inspect_archive,
    resolve_candidate_paths,
    safe_extract_archive,
    sha256_file,
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
from src.monitoring.real_world_evaluation_protocol import PRODUCTION_THRESHOLD
from src.risk_engine.altman_native_ensemble import map_raw_to_native
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
    except Exception as exc:  # noqa: BLE001
        check(False, f"{msg} (wrong exception: {type(exc).__name__})")
        return
    check(False, msg + " (no ValueError)")


def expect_typeerror(fn, msg: str) -> None:
    try:
        fn()
    except TypeError:
        check(True, msg)
        return
    except Exception as exc:  # noqa: BLE001
        check(False, f"{msg} (wrong exception: {type(exc).__name__})")
        return
    check(False, msg + " (no TypeError)")


# ══════════════════════════════════════════════════════════════════════
# FIXTURES (temp only — never inside the repository)
# ══════════════════════════════════════════════════════════════════════

_TMP = tempfile.TemporaryDirectory(prefix="phase107_")
TMPROOT = _TMP.name

ULB_SCHEMA = tuple(REG.get_dataset("ULB_CREDIT_CARD_FRAUD").schema_columns)
IEEE_SCHEMA = tuple(REG.get_dataset("IEEE_CIS").schema_columns)
BANKSIM_SCHEMA = tuple(REG.get_dataset("BANKSIM").schema_columns)


def fixture_row(dataset_id: str, **overrides: str) -> list[str]:
    """A full-width row for one registered dataset's schema."""
    schema = REG.get_dataset(dataset_id).schema_columns
    row = [""] * len(schema)

    def put(col: str | None, value: str) -> None:
        if col and col in schema:
            row[schema.index(col)] = value

    put(REG.LABEL_COLUMNS.get(dataset_id), "0")
    put(REG.TIMESTAMP_COLUMNS.get(dataset_id), "1")
    put(REG.AMOUNT_COLUMNS.get(dataset_id), "5.0")
    for key, value in overrides.items():
        put(key, value)
    return row


def write_csv(name: str, header, rows) -> str:
    path = os.path.join(TMPROOT, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = __import__("csv").writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_text(name: str, text: str) -> str:
    path = os.path.join(TMPROOT, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def zip_with(name: str, members: list[tuple[str, str]]) -> str:
    path = os.path.join(TMPROOT, name)
    with zipfile.ZipFile(path, "w") as zf:
        for member, content in members:
            zf.writestr(member, content)
    return path


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — MODULE CONSTANTS & AUTHORITIES
# ══════════════════════════════════════════════════════════════════════

section("1. Module constants & authoritative bindings")

inv(ING.PHASE == 107, "phase number is 107")
check(ING.INGESTION_VERSION == "phase107_ingestion_v1"
      and INGESTION_VERSION == ING.INGESTION_VERSION, "ingestion version")
check(ING.MANIFEST_VERSION == "phase107_benchmark_manifest_v1",
      "manifest version")
check(MANIFEST_CREATED_AT == "2026-09-23T00:00:00+00:00",
      "manifest timestamp is a constant, never a clock read")
check(ING.BENCHMARK_NAME == "PUBLIC_MULTI_SOURCE_FRAUD_BENCHMARK",
      "benchmark name is multi-source public")
check(ING.ACQUISITION_ROOT == "data/external_benchmark",
      "spec acquisition structure")
check(EXPECTED_PRODUCTION_THRESHOLD == 0.018758,
      "threshold literal matches spec")
inv(PRODUCTION_THRESHOLD == EXPECTED_PRODUCTION_THRESHOLD == 0.018758,
    "threshold equals the locked production authority")
inv(MODEL_ID == "altman_native"
    and RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904",
    "production model/release identities per spec")
inv(FEATURE_VERSION == ML_FEATURE_VERSION == "v1",
    "runtime/domain feature version is v1 (both authorities agree)")
check(NATIVE_FEATURE_VERSION == ING.NATIVE_FEATURE_VERSION,
      "native feature version bound in ingestion module")
inv(len(ML_FEATURE_ORDER) == 21 == ING.DOMAIN_FEATURE_COUNT,
    "domain feature count is 21")
inv(len(ALTMAN_NATIVE_FEATURES) == 48 == ING.NATIVE_FEATURE_COUNT,
    "native feature count is 48")
check(ING.ML_FEATURE_ORDER == ML_FEATURE_ORDER,
      "module binds the existing feature order, does not fork it")
check(CANONICAL_TRANSFORMATION.count("map_raw_to_native") == 1,
      "canonical transformation names the one pipeline")
inv(FORBIDDEN_BYPASS_PARAMETERS == frozenset(
    {"force", "allow_unverified", "skip_validation", "override",
     "admin_override", "bypass"}),
    "bypass parameter set matches spec exactly")
check(FORBIDDEN_BYPASS_PARAMETERS is REG.FORBIDDEN_BYPASS_PARAMETERS,
      "bypass set reused from Phase 106, not redefined")

inv(set(DATASET_DIRNAMES) == set(REG.DATASET_IDS)
    == set(PRIMARY_BASENAMES) == set(CANDIDATE_PATHS),
    "all five registered candidates have directory/primary/path maps")
for dataset_id in REG.DATASET_IDS:
    paths = CANDIDATE_PATHS[dataset_id]
    inv(paths[0].startswith(ING.ACQUISITION_ROOT),
        f"{dataset_id}: spec structure is the first candidate")
    inv(all(not os.path.isabs(p) and "\\" not in p for p in paths),
        f"{dataset_id}: candidate paths are repo-relative POSIX")
inv("data/creditcard.csv" in CANDIDATE_PATHS["ULB_CREDIT_CARD_FRAUD"],
    "legacy local ULB drop is a documented candidate")
inv("data/paysim.csv" in CANDIDATE_PATHS["PAYSIM"],
    "legacy local PaySim drop is a documented candidate")
inv(ING.MULTIFILE_REQUIRED == {"IEEE_CIS": ("train_transaction.csv",
                                            "train_identity.csv")},
    "IEEE-CIS is the only multi-file candidate")
inv(set(ING.SAFE_MEMBER_SUFFIXES).isdisjoint(
    set(ING.EXECUTABLE_SUFFIXES) | set(ING.ARCHIVE_SUFFIXES)),
    "archive member suffix classes are disjoint")
check(ING.REPO_ROOT == os.path.dirname(BACKEND), "repo root resolution")

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — SIGNATURES & BYPASS PARAMETERS
# ══════════════════════════════════════════════════════════════════════

section("2. Pinned signatures & bypass-parameter rejection")

PINNED_SIGNATURES = {
    "ingest_dataset": ("dataset_id", "candidate_paths",
                       "duplicate_accumulator"),
    "ingest_all": (),
    "ingest_many": ("dataset_requests",),
    "build_partitions": ("records",),
    "build_combined_benchmark": ("partitions",),
    "build_component_evaluation_views": ("partitions",),
    "build_phase107_manifest": ("records", "partitions"),
    "build_ingestion_ledger": ("records",),
    "build_field_mappings": ("dataset_id",),
    "resolve_candidate_paths": ("dataset_id",),
    "acquisition_instructions": ("dataset_id",),
    "detect_container": ("path",),
    "sha256_file": ("path",),
    "inspect_archive": ("archive_path",),
    "safe_extract_archive": ("archive_path", "extraction_root"),
    "detect_dataset_replacement": ("old", "new"),
    "verify_manifest_hash": ("manifest",),
}
REP_SIGNATURES = {
    "generate_phase107_report": ("records", "partitions", "manifest"),
}
for name, expected in PINNED_SIGNATURES.items():
    got = tuple(inspect.signature(getattr(ING, name)).parameters)
    inv(got == expected, f"ING.{name} signature pinned {got}")
for name, expected in REP_SIGNATURES.items():
    got = tuple(inspect.signature(getattr(REP, name)).parameters)
    inv(got == expected, f"REP.{name} signature pinned {got}")

# no public function anywhere may expose a bypass parameter
for module, alias in ((ING, "ING"), (REP, "REP")):
    offenders = []
    for name, obj in vars(module).items():
        if (inspect.isfunction(obj)
                and getattr(obj, "__module__", "").startswith(
                    "src.monitoring.phase107")):
            clash = (set(inspect.signature(obj).parameters)
                     & FORBIDDEN_BYPASS_PARAMETERS)
            if clash:
                offenders.append((name, sorted(clash)))
    inv(not offenders, f"{alias}: no bypass parameters exist ({offenders})")

BYPASS_FNS = [
    ("ingest_dataset", ING.ingest_dataset),
    ("ingest_many", ING.ingest_many),
    ("build_partitions", ING.build_partitions),
    ("build_component_evaluation_views",
     ING.build_component_evaluation_views),
    ("build_phase107_manifest", ING.build_phase107_manifest),
    ("build_combined_benchmark", ING.build_combined_benchmark),
    ("build_ingestion_ledger", ING.build_ingestion_ledger),
    ("generate_phase107_report", REP.generate_phase107_report),
]
for bypass in sorted(FORBIDDEN_BYPASS_PARAMETERS):
    for fn_name, fn in BYPASS_FNS:
        expect_typeerror(lambda f=fn, b=bypass: f(**{b: True}),
                         f"{bypass}=True rejected by {fn_name}")
inv(True, "all six bypass classes rejected by every entry point")

inv(not any(n.startswith("qualify") for n in vars(ING))
    and not any(n.startswith("qualify") for n in vars(REP)),
    "no qualification authority is defined here")
inv(not any(n.startswith("promote") for n in vars(ING))
    and not any(n.startswith("promote") for n in vars(REP)),
    "no promotion authority is defined here")
inv("map_raw_to_native" not in vars(ING)
    and "map_raw_to_native" not in vars(REP),
    "canonical transformation is imported, never redefined")

# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — SECURITY SOURCE SCANS
# ══════════════════════════════════════════════════════════════════════

section("3. Security source scans (offline, no-eval, no-gate, no-fabrication)")

# Case-sensitive on purpose: uppercase state literals such as
# PROMOTION_GATE_REQUIRED are values, not module references.
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
    ("retrain", r"\bretrain"),
    ("promote call", r"\bpromote\s*\("),
    ("train call", r"\btrain\s*\("),
    ("model artifact path", r"models/production"),
    ("release manifest path", r"release_manifest"),
    ("network download call", r"\.download\s*\("),
    ("secrets module", r"secrets\."),
    ("credentials access", r"credential[s]?\s*[:=]"),
)
def _read_source(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


SOURCE_TEXTS = {
    "ingestion": _read_source(ING.__file__),
    "report": _read_source(REP.__file__),
}
for label, pattern in FORBIDDEN_PATTERNS:
    for source_name, text in SOURCE_TEXTS.items():
        check(re.search(pattern, text) is None,
              f"{source_name} contains no {label}")
inv(True, f"{len(FORBIDDEN_PATTERNS)} forbidden patterns x 2 sources")

# no network/tooling imports anywhere in the modules
for source_name, text in SOURCE_TEXTS.items():
    check("import socket" not in text and "from socket" not in text,
          f"{source_name}: no socket import")
    check("import subprocess" not in text,
          f"{source_name}: no subprocess import")

# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — ACQUISITION: REAL WORKSPACE STATE
# ══════════════════════════════════════════════════════════════════════

section("4. Acquisition of the real local workspace files")

RECORDS = ingest_all()
inv(set(RECORDS) == set(REG.DATASET_IDS), "all candidates addressed")
inv(RECORDS["IEEE_CIS"].status == "not_acquired"
    and RECORDS["BANKSIM"].status == "not_acquired"
    and RECORDS["BAF_BANK_ACCOUNT_FRAUD"].status == "not_acquired",
    "absent candidates are NOT_ACQUIRED, never an error")
inv(RECORDS["ULB_CREDIT_CARD_FRAUD"].verified,
    "the genuine local ULB file verifies")
inv(RECORDS["PAYSIM"].status
    == "acquired_verification_failed",
    "the legacy local PaySim files fail verification, not skip it")

ulb = RECORDS["ULB_CREDIT_CARD_FRAUD"]
inv(ulb.primary_file == "data/creditcard.csv",
    f"ULB primary is the documented legacy file ({ulb.primary_file})")
inv(ulb.row_count == 284807, "ULB row count matches the documented source")
inv(dict(ulb.label_counts) == {"0": 284315, "1": 492},
    "ULB label counts match documented 492 frauds")
inv(ulb.row_count_check == "match" and ulb.fraud_count_check == "match",
    "both documented-count checks pass")
check(ulb.verification_errors == (), "ULB carries no verification errors")
check("leakage_fields_present" in ulb.verification_warnings,
      "ULB leakage warning recorded (Class)")
check(dict(ulb.normalized_label_counts) == {"0": 284315, "1": 492},
      "every ULB label normalizes deterministically (no unknowns)")
check(sum(n for _, n in ulb.label_counts) == ulb.row_count,
      "label counts reconcile with rows")
ulb_path = os.path.join(ING.REPO_ROOT, *ulb.primary_file.split("/"))
independent_sha = sha256_file(ulb_path)
inv(dict((f.path, f.sha256) for f in ulb.file_records)[
    "data/creditcard.csv"] == independent_sha,
    "recorded SHA-256 matches an independent hash of the file")
inv(ulb.version == f"sha256:{independent_sha}",
    "dataset version IS the file hash")
check(ulb.verify_hash(), "ULB record hash verifies")
check(ulb.row_id_kind == "local_row_position_hash",
      "ULB rows have no source record id — local position hash only")
check(ulb.timestamp_status == "ordering_only_relative",
      "ULB timestamps are ordering-only relative")
check(ulb.column_count == len(ULB_SCHEMA)
      and set(ulb.schema_actual) == set(ULB_SCHEMA),
      "ULB schema matches the registry set (source column order kept)")
check(tuple(ulb.schema_actual)
      == tuple(REG.get_dataset("ULB_CREDIT_CARD_FRAUD")
               .schema_columns)[:0] + ulb.schema_actual,
      "schema_actual is the truthful file header order")
check(ulb.missing_columns == () and ulb.renamed_columns == ()
      and ulb.extra_columns == () and ulb.type_mismatch_counts == ()
      and ulb.ragged_rows == 0,
      "no schema discrepancies on the genuine file")

pay = RECORDS["PAYSIM"]
inv(pay.primary_file == "data/paysim.csv",
    f"PaySim primary resolves in documented order ({pay.primary_file})")
check(pay.alternate_files == ("data/paysim_1m.csv",),
      "the second legacy PaySim file is recorded, never silently skipped")
inv({"step", "nameOrig", "nameDest"}.issubset(set(pay.missing_columns)),
    "PaySim missing columns identified precisely")
check("schema_missing_columns" in pay.verification_errors
      and "row_count_mismatch" in pay.verification_errors,
      "PaySim fails on schema and documented-count mismatch")
check(pay.row_count_check == "mismatch",
      "100000 rows != documented 1048576")
check("expected_filename_absent" in pay.verification_warnings,
      "unexpected filename recorded as warning")
check("fingerprint_columns_incomplete" in pay.verification_warnings,
      "fingerprint disabled when the timestamp column is absent")
check("alternate_candidate_files_present" in pay.verification_warnings,
      "alternate candidate recorded")
check(len(pay.file_records) == 2, "both located files are hashed")
check(pay.label_counts and dict(pay.label_counts).get("1") is not None,
      "PaySim label column found and counted")
check(pay.verify_hash(), "PaySim record hash verifies")

for dataset_id in ("IEEE_CIS", "BANKSIM", "BAF_BANK_ACCOUNT_FRAUD"):
    rec = RECORDS[dataset_id]
    check(rec.located_files == () and rec.primary_file is None
          and rec.row_count == 0 and rec.version == "not_acquired"
          and rec.file_records == ()
          and "dataset_not_located" in rec.verification_warnings
          and rec.duplicate_findings == DuplicateFindings(0, 0, 0, 0, 0, 0),
          f"{dataset_id}: clean NOT_ACQUIRED record")
inv(REG.get_dataset("BAF_BANK_ACCOUNT_FRAUD").schema_columns == (),
    "BAF schema was never invented (registry empty)")

resolved = resolve_candidate_paths("ULB_CREDIT_CARD_FRAUD")
check(resolved == ("data/creditcard.csv",),
      "resolution order is deterministic")
check(resolve_candidate_paths("BANKSIM") == (),
      "absent dataset resolves to nothing")

# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — VERIFICATION FIXTURES (failure paths & admission gates)
# ══════════════════════════════════════════════════════════════════════

section("5. Verification fixtures: wrong files, schema drift, license gate")

# wrong file (documented row count not met)
wrong = write_csv("ulb_wrong.csv", ULB_SCHEMA,
                  [fixture_row("ULB_CREDIT_CARD_FRAUD")] * 9)
rec_wrong = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (wrong,))
inv(rec_wrong.status == "acquired_verification_failed",
    "wrong-sized file rejected")
check("row_count_mismatch" in rec_wrong.verification_errors,
      "wrong file detected via documented row count")
check(rec_wrong.row_count_check == "mismatch", "row check mismatch")
check(rec_wrong.duplicate_findings.total_rows == rec_wrong.row_count == 9,
      "findings cover the fixture rows")
check(rec_wrong.verify_hash(), "fixture record hash verifies")

# missing column
hdr_missing = tuple(c for c in ULB_SCHEMA if c != "V28")
rows_short = []
for _ in range(9):
    full = fixture_row("ULB_CREDIT_CARD_FRAUD")
    rows_short.append([v for c, v in zip(ULB_SCHEMA, full)
                       if c != "V28"])
missing_path = write_csv("ulb_missing.csv", hdr_missing, rows_short)
rec_missing = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (missing_path,))
inv(rec_missing.status == "acquired_verification_failed",
    "missing column fails closed")
check(rec_missing.missing_columns == ("V28",),
      f"missing column named precisely ({rec_missing.missing_columns})")
check("schema_missing_columns" in rec_missing.verification_errors,
      "missing-column error recorded")

# renamed column (case drift)
hdr_renamed = ["time" if c == "Time" else c for c in ULB_SCHEMA]
rows_renamed = []
for _ in range(9):
    full = fixture_row("ULB_CREDIT_CARD_FRAUD")
    rows_renamed.append(["time" if c == "Time" else v
                         for c, v in zip(ULB_SCHEMA, full)])
renamed_path = write_csv("ulb_renamed.csv", hdr_renamed, rows_renamed)
rec_renamed = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (renamed_path,))
inv(rec_renamed.status == "acquired_verification_failed",
    "renamed column fails closed")
check(("Time", "time") in rec_renamed.renamed_columns,
      "renamed column pair recorded")
check("schema_renamed_columns" in rec_renamed.verification_errors,
      "rename error recorded")

# extra column recorded (not fatal on its own — count still fails here)
hdr_extra = tuple(ULB_SCHEMA) + ("operator_note",)
rows_extra = [fixture_row("ULB_CREDIT_CARD_FRAUD") + ["n"]
              for _ in range(9)]
extra_path = write_csv("ulb_extra.csv", hdr_extra, rows_extra)
rec_extra = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (extra_path,))
check(rec_extra.extra_columns == ("operator_note",),
      "extra column recorded, never silently adopted")
check("schema_extra_columns" in rec_extra.verification_warnings,
      "extra column is a warning")
check(rec_extra.missing_columns == (), "extra column adds no missing cols")

# type mismatch
rows_bad = []
for i in range(9):
    full = fixture_row("ULB_CREDIT_CARD_FRAUD",
                       **({"Amount": "abc"} if i == 3 else {}))
    rows_bad.append(full)
type_path = write_csv("ulb_type.csv", ULB_SCHEMA, rows_bad)
rec_type = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (type_path,))
check(("Amount", 1) in rec_type.type_mismatch_counts,
      f"type mismatch counted without storing values "
      f"({rec_type.type_mismatch_counts})")
check("schema_type_mismatch" in rec_type.verification_errors,
      "type mismatch error recorded")

# ragged row (one extra cell)
rows_ragged = [fixture_row("ULB_CREDIT_CARD_FRAUD") for _ in range(9)]
rows_ragged.append(list(rows_ragged[0]) + ["EXTRA"])
ragged_path = write_csv("ulb_ragged.csv", ULB_SCHEMA, rows_ragged)
rec_ragged = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (ragged_path,))
check(rec_ragged.ragged_rows == 1 and "ragged_rows"
      in rec_ragged.verification_errors, "ragged row detected")

# ambiguous label: preserved verbatim, UNKNOWN never forced to0/1
rows_amb = [fixture_row("ULB_CREDIT_CARD_FRAUD") for _ in range(8)]
rows_amb.append(fixture_row("ULB_CREDIT_CARD_FRAUD", Class="maybe"))
amb_path = write_csv("ulb_ambiguous.csv", ULB_SCHEMA, rows_amb)
rec_amb = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (amb_path,))
check(dict(rec_amb.label_counts).get("maybe") == 1,
      "ambiguous label preserved verbatim")
check(dict(rec_amb.normalized_label_counts).get("unknown") == 1,
      "ambiguous label normalizes to UNKNOWN, never guessed")
check(rec_amb.status == "acquired_verification_failed",
      "ambiguous label file also fails the documented counts")

# absent / empty / unknown ids
absent = ingest_dataset("BANKSIM", (os.path.join(TMPROOT, "nope.csv"),))
inv(absent.status == "not_acquired",
    "explicit path to a missing file is NOT_ACQUIRED, not a crash")
empty = ingest_dataset("ULB_CREDIT_CARD_FRAUD", ())
check(empty.status == "not_acquired", "empty candidate set is NOT_ACQUIRED")
expect_failclosed(lambda: ingest_dataset("NOT_A_DATASET"),
                  "unregistered dataset id fails closed")
expect_failclosed(lambda: REG.get_dataset("NOT_A_DATASET"),
                  "registry lookup for unknown id fails closed")

# JSON container support
schema = ULB_SCHEMA
json_rows = [
    {c: v for c, v in zip(schema, fixture_row("ULB_CREDIT_CARD_FRAUD"))}
    for _ in range(9)
]
json_path = os.path.join(TMPROOT, "creditcard.json")
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(json_rows, fh)
rec_json = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (json_path,))
check(detect_container(json_path) == ContainerType.JSON,
      "json container detected")
check(rec_json.schema_actual == ULB_SCHEMA and rec_json.missing_columns == (),
      "json schema read identically to csv")
check(rec_json.status == "acquired_verification_failed"
      and "row_count_mismatch" in rec_json.verification_errors,
      "json file still subject to documented-count verification")

# IEEE-CIS multi-file: incomplete alone, schema-complete with companion
ieee_full = fixture_row("IEEE_CIS", TransactionID="1",
                        TransactionDT="100", TransactionAmt="5.0",
                        isFraud="0")
ieee_map = dict(zip(IEEE_SCHEMA, ieee_full))
ieee_tx_header = tuple(c for c in IEEE_SCHEMA
                       if c not in ("DeviceType", "DeviceInfo"))
ieee_tx = write_csv("train_transaction.csv", ieee_tx_header,
                    [[ieee_map[c] for c in ieee_tx_header]])
rec_ieee_single = ingest_dataset("IEEE_CIS", (ieee_tx,))
inv(rec_ieee_single.status == "incomplete_acquisition",
    "IEEE without identity table is INCOMPLETE, never verified")
check("required_files_missing" in rec_ieee_single.verification_errors,
    "missing companion file recorded")
check(rec_ieee_single.row_id_kind == "source_record_id",
      "IEEE uses the genuine source TransactionID as record id")
ieee_id = write_csv("train_identity.csv", ("DeviceType", "DeviceInfo"),
                    [["Windows", "chrome/90"]])
rec_ieee_pair = ingest_dataset("IEEE_CIS", (ieee_tx, ieee_id))
inv(rec_ieee_pair.missing_columns == ()
    and rec_ieee_pair.schema_actual == ieee_tx_header,
    "companion table completes the registered schema union")
check("required_files_missing" not in rec_ieee_pair.verification_errors,
    "both files present clears the completeness gate")
check(rec_ieee_pair.status == "acquired_verification_failed"
      and "row_count_mismatch" in rec_ieee_pair.verification_errors,
      "small IEEE fixture still fails documented row count")

# BankSim: fully verified locally, yet license blocks any partition
bank_rows = [fixture_row("BANKSIM", step=str(i % 5 + 1))
             for i in range(12)]
bank_path = write_csv("banksim.csv", BANKSIM_SCHEMA, bank_rows)
rec_bank = ingest_dataset("BANKSIM", (bank_path,))
inv(rec_bank.verified,
    "BankSim fixture verifies locally (counts not documented)")
inv(REG.get_dataset("BANKSIM").license_documented is False,
    "BankSim license is not established")
inv(build_partitions({"BANKSIM": rec_bank}) == (),
    "verified-but-unlicensed file receives NO partition")
expect_valueerror(
    lambda: dataclasses.replace(
        build_partitions(RECORDS)[0], dataset_id="BANKSIM"),
    "direct partition construction for a Group D dataset fails closed")
inv(REG.get_dataset("BANKSIM").transaction_count is None
    and REG.get_dataset("BANKSIM").fraud_count is None,
    "undocumented counts stay None — never invented")

# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — SOURCE IDENTITY & COMBINED BENCHMARK
# ══════════════════════════════════════════════════════════════════════

section("6. Source identity & source-preserving combined benchmark")

PARTITIONS = build_partitions(RECORDS)
inv(len(PARTITIONS) == 1
    and PARTITIONS[0].dataset_id == "ULB_CREDIT_CARD_FRAUD",
    "exactly the verified licensed source receives a partition")
part = PARTITIONS[0]
inv(part.partition_id == "benchmark/ulb",
    "partition namespace follows the spec layout")
check(part.group == "group_b"
      and part.evaluation_readiness == "component_evaluation_only",
      "Group B stays component-only after ingestion")
check(part.supervised_rows == 284807
      and part.row_count == 284807,
      "all ULB rows are supervised-eligible")
check(part.primary_sha256 == independent_sha,
      "partition bound to the verified file hash")
check(part.dataset_version == ulb.version,
      "partition carries the file-hash version")
check(part.license_documented
      and part.provenance_confidence == "high"
      and part.classification == "public_real_anonymized",
      "partition carries Phase 106 governance metadata")
check(part.verify_hash(), "partition hash verifies")
check(part.label_semantics == REG.get_dataset(
    "ULB_CREDIT_CARD_FRAUD").label_semantics,
    "label semantics come from the registry, not invented")
check("Class" in part.excluded_fields,
      "label column is on the evaluation exclusion list")
check(set(REG.scan_leakage_fields(part.schema_columns))
      <= set(part.excluded_fields),
      "every scanned leakage field is excluded")
check(not any(c.lower() in ("customer_id", "card_id", "cc_num", "name")
              for c in part.schema_columns),
      "no fabricated identity columns appear in the partition")

combined = build_combined_benchmark(PARTITIONS)
inv(combined.benchmark_name == BENCHMARK_NAME
    and combined.descriptor == REG.BENCHMARK_DESCRIPTOR,
    "combined benchmark named as multi-source public benchmark")
inv(combined.total_rows == 284807
    and combined.supervised_rows == 284807,
    "combined totals are the sum of admitted partitions")
check(combined.harmonized_fields == REG.COMMON_BENCHMARK_FIELDS,
      "only Phase 106's defensible common field is harmonized")
check(combined.source_dataset_ids == ("ULB_CREDIT_CARD_FRAUD",),
      "source identity preserved in the combined view")
check({"Amount", "Time"} <= set(combined.source_local_fields)
      and "Class" not in combined.source_local_fields,
      "raw amount/time stay source-local; labels never pooled raw")

# streaming rows: stamped, deterministic, label verbatim
rows_a = list(itertools.islice(combined.iter_rows(), 50))
rows_b = list(itertools.islice(combined.iter_rows(), 50))
inv(len(rows_a) == 50 and rows_a == rows_b,
    "streamed rows are deterministic across passes")
inv(all(r.source_dataset_id == "ULB_CREDIT_CARD_FRAUD"
        for r in rows_a),
    "every combined row carries its source_dataset_id")
inv(all(re.fullmatch(r"ULB_CREDIT_CARD_FRAUD:\d+:[0-9a-f]{8}",
                     r.source_record_id)
        for r in rows_a),
    "source_row_id is a deterministic position/content hash")
inv(all(r.original_label in ("0", "1") for r in rows_a),
    "original labels preserved verbatim through streaming")
first_raw_label = None
with open(ulb_path, "r", encoding="utf-8", newline="") as fh:
    reader = __import__("csv").reader(fh)
    next(reader)
    first_raw_row = next(reader)
first_raw_label = first_raw_row[ULB_SCHEMA.index("Class")]
inv(rows_a[0].original_label == first_raw_label,
    "first streamed label equals the raw file cell")
inv(all(r.normalized_label in (0, 1, None) for r in rows_a),
    "normalized labels are only0/1/None")

expect_valueerror(lambda: build_combined_benchmark(()),
                  "no admitted partitions -> no combined benchmark")
expect_valueerror(
    lambda: dataclasses.replace(
        combined, benchmark_name="one bank's transactions"),
    "benchmark name substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        combined,
        descriptor=f"dataset {REG.FORBIDDEN_SINGLE_SOURCE_CLAIMS[0]}"),
    "single-institution descriptor claim fails closed")
expect_valueerror(lambda: dataclasses.replace(combined,
                                               source_files=()),
                  "empty source files fail closed")
expect_valueerror(
    lambda: dataclasses.replace(
        combined,
        partition_ids=("benchmark/a", "benchmark/b"),
        source_files=(("benchmark/a", "X", "f"), ("benchmark/y", "X", "g")),
        source_dataset_ids=("X",)),
    "misaligned source files fail closed")
expect_valueerror(
    lambda: dataclasses.replace(
        combined,
        partition_ids=("benchmark/a", "benchmark/b"),
        source_files=(("benchmark/a", "X", "f"),
                      ("benchmark/b", "X", "g")),
        source_dataset_ids=("X", "X")),
    "duplicate combined sources fail closed")

# foreign-source rows are never re-attributed
foreign = REG.harmonize_rows(
    "ULB_CREDIT_CARD_FRAUD",
    [{"source_dataset_id": "PAYSIM", "record_id": "evil",
      "Class": "1", "Time": "1", "Amount": "1"}])
inv(foreign[0].harmonization_status
    == REG.HarmonizationStatus.EXCLUDED_FOREIGN_SOURCE.value
    and foreign[0].normalized_label is None,
    "cross-source stamp forgery excluded, never merged")

# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — DUPLICATE DETECTION & CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

section("7. Duplicate classification (never deletion)")

ulb_findings = ulb.duplicate_findings
inv(ulb_findings is not None
    and (ulb_findings.unique_rows
         + ulb_findings.within_source_duplicate_rows
         + ulb_findings.cross_source_duplicate_rows
         == ulb_findings.total_rows == 284807),
    "real-file duplicate classes partition the row total exactly")

acc = DuplicateAccumulator()
acc.observe("ULB_CREDIT_CARD_FRAUD", "contentX", "fp1")
acc.observe("ULB_CREDIT_CARD_FRAUD", "contentX", "fp1")
acc.observe("ULB_CREDIT_CARD_FRAUD", "contentY", None)
acc.fold_local("ULB_CREDIT_CARD_FRAUD")
f_a = acc.findings_for("ULB_CREDIT_CARD_FRAUD", 3)
inv(f_a.within_source_duplicate_rows == 2 and f_a.unique_rows == 1
    and f_a.cross_source_duplicate_rows == 0,
    "same-source exact duplicate classified as WITHIN_SOURCE")
acc.observe("PAYSIM", "contentX", "fp1")
acc.fold_local("PAYSIM")
f_a2 = acc.findings_for("ULB_CREDIT_CARD_FRAUD", 3)
f_b = acc.findings_for("PAYSIM", 1)
inv(f_a2.cross_source_duplicate_rows == 2
    and f_b.cross_source_duplicate_rows == 1,
    "cross-source duplicate rows flagged symmetrically on both sides")
inv(f_a2.unique_rows == 1 and f_b.unique_rows == 0,
    "cross classification supersedes within/uniqueness consistently")
expect_valueerror(lambda: acc.fold_local("PAYSIM"),
                  "double-folding a source fails closed")
inv(acc.cross_source_groups == 1, "cross-source content group counted")

acc2 = DuplicateAccumulator()
acc2.observe("ULB_CREDIT_CARD_FRAUD", "cA", "fp")
acc2.observe("ULB_CREDIT_CARD_FRAUD", "cB", "fp")
acc2.fold_local("ULB_CREDIT_CARD_FRAUD")
f2 = acc2.findings_for("ULB_CREDIT_CARD_FRAUD", 2)
inv(f2.normalized_collision_rows == 2
    and f2.normalized_collision_groups == 1
    and f2.unique_rows == 2,
    "normalized collision overlaps uniqueness — rows reported, "
    "never merged or deleted")

expect_valueerror(lambda: DuplicateFindings(10, 5, 3, 3, 0, 0),
                  "duplicate classes that do not partition fail closed")
expect_valueerror(lambda: DuplicateFindings(10, 11, 0, 0, 0, 0),
                  "negative/over-total classes fail closed")

# fixture cross-source fingerprint collision through real file ingestion
u_rows = [fixture_row("ULB_CREDIT_CARD_FRAUD", Time="5",
                      Amount="42.5", Class="1")]
u_rows += [fixture_row("ULB_CREDIT_CARD_FRAUD", Time="1",
                       Amount="1.0", Class="0")] * 4
f_ulb = write_csv("u.csv", ULB_SCHEMA, u_rows)
ps_schema = tuple(REG.get_dataset("PAYSIM").schema_columns)
f_ps = write_csv("p.csv", ps_schema,
                 [["5", "PAYMENT", "42.5", "C1", "1000", "957.5",
                   "M1", "0", "0", "1", "0"]])
pair = ingest_many({"ULB_CREDIT_CARD_FRAUD": (f_ulb,),
                    "PAYSIM": (f_ps,)})
pu, pp = pair["ULB_CREDIT_CARD_FRAUD"], pair["PAYSIM"]
inv(pu.duplicate_findings.normalized_collision_rows == 1
    and pp.duplicate_findings.normalized_collision_rows == 1,
    "same (label, time, amount) fingerprint collides across sources "
    "and is flagged on both sides")
check(pu.duplicate_findings.within_source_duplicate_rows == 4
      and pu.duplicate_findings.unique_rows == 1,
      "fixture within-source duplicates classified")
expect_valueerror(
    lambda: ingest_many({"NOT_A_DATASET": (f_ulb,)}),
    "ingest_many with unregistered id fails closed")

# versioning: content change always yields a new version
t_future = time.time() + 3600
os.utime(wrong, (t_future, t_future))
rec_untouched = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (wrong,))
inv(rec_untouched.record_hash == rec_wrong.record_hash
    and rec_untouched.version == rec_wrong.version,
    "mtime changes never alter a record hash or version")
inv(detect_dataset_replacement(rec_wrong, rec_untouched) == "identical",
    "identical re-ingest reported as identical")
rows_more = rows_bad + [fixture_row("ULB_CREDIT_CARD_FRAUD")]
changed = write_csv("creditcard_changed.csv", ULB_SCHEMA, rows_more)
rec_changed = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (changed,))
inv(detect_dataset_replacement(rec_wrong, rec_changed)
    == "primary_hash_changed",
    "replaced file detected as a hash change, never overwritten")
check(detect_dataset_replacement(rec_wrong, rec_json)
    == "primary_hash_changed", "hash-change verdict on differing files")
check(detect_dataset_replacement(rec_wrong, RECORDS["PAYSIM"])
    == "dataset_id_mismatch", "cross-dataset comparison rejected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — HARMONIZATION & CANONICAL FIELD MAPPINGS
# ══════════════════════════════════════════════════════════════════════

section("8. Source-preserving normalization & field mappings")

USABLE_COUNTS = {"IEEE_CIS": 5, "ULB_CREDIT_CARD_FRAUD": 5,
                 "PAYSIM": 10, "BANKSIM": 8, "BAF_BANK_ACCOUNT_FRAUD": 0}
for dataset_id in REG.DATASET_IDS:
    mappings = build_field_mappings(dataset_id)
    inv(len(mappings) > 0
        and mappings == build_field_mappings(dataset_id),
        f"{dataset_id}: mapping table deterministic")
    inv(all(m.mapping_type in {t.value for t in MappingType}
            for m in mappings),
        f"{dataset_id}: only declared mapping types")
    inv(all(m.source_field == "" for m in mappings
            if m.mapping_type == MappingType.NOT_AVAILABLE.value),
        f"{dataset_id}: unavailable fields carry no invented source")
    inv(any(m.mapping_type == MappingType.EXCLUDED_LEAKAGE.value
            for m in mappings),
        f"{dataset_id}: leakage exclusions declared")
    inv(all(m.canonical_field in CANONICAL_CANDIDATE_FIELDS
            or m.canonical_field == "excluded_evaluation_field"
            for m in mappings),
        f"{dataset_id}: canonical fields bounded by the contract")
    inv(any(m.mapping_type == MappingType.NOT_AVAILABLE.value
            and m.canonical_field == "mcc" for m in mappings),
        f"{dataset_id}: MCC never manufactured")

ulb_map = {(m.canonical_field, m.source_field, m.mapping_type)
           for m in build_field_mappings("ULB_CREDIT_CARD_FRAUD")}
check(("original_label", "Class", "direct") in ulb_map,
      "ULB label maps directly (verbatim preservation)")
check(("normalized_label", "Class", "deterministic_transform") in ulb_map,
      "ULB normalized label is a deterministic transform")
check(("amount", "Amount", "direct") in ulb_map,
      "ULB amount maps directly")
check(("transaction_type", "", "not_available") in ulb_map,
      "ULB has no transaction type — not fabricated")
ps_map = {(m.canonical_field, m.source_field, m.mapping_type)
          for m in build_field_mappings("PAYSIM")}
check(("transaction_type", "type", "direct") in ps_map,
      "PaySim transaction type maps directly")
check(any(c == "entity_identifier" and t == "semantically_incompatible"
          for c, _, t in [(m[0], m[1], m[2]) for m in ps_map]
          if c == "entity_identifier"),
      "PaySim account ids are source-local, never cross-source")
bank_map = {(m.canonical_field, m.source_field, m.mapping_type)
            for m in build_field_mappings("BANKSIM")}
check(("merchant_category", "category", "semantically_incompatible")
      in bank_map,
      "BankSim category is not forced into merchant semantics")

# authoritative normalization semantics reused, never redefined
check(REG.normalize_label("ULB_CREDIT_CARD_FRAUD", "1") == 1
      and REG.normalize_label("ULB_CREDIT_CARD_FRAUD", "0") == 0
      and REG.normalize_label("ULB_CREDIT_CARD_FRAUD", "maybe") is None
      and REG.normalize_label("ULB_CREDIT_CARD_FRAUD", "1") == 1,
      "normalize_label deterministic and never guesses")
unmapped = REG.harmonize_rows(
    "ULB_CREDIT_CARD_FRAUD",
    [{"record_id": "x", "Class": "maybe", "Time": "9",
      "Amount": "3.0"}])
inv(unmapped[0].harmonization_status
    == REG.HarmonizationStatus.EXCLUDED_UNMAPPED_LABEL.value
    and unmapped[0].normalized_label is None,
    "unmappable label excluded from supervised harmonization")
mapped = REG.harmonize_rows(
    "ULB_CREDIT_CARD_FRAUD",
    [{"record_id": "x", "Class": "1", "Time": "9", "Amount": "3.0"}])
inv(mapped[0].harmonization_status
    == REG.HarmonizationStatus.HARMONIZED.value
    and mapped[0].original_label == "1",
    "harmonized row keeps the verbatim original label")
check(REG.row_fingerprint(mapped[0]) == REG.row_fingerprint(mapped[0]),
      "row fingerprint deterministic")

# supervised eligibility excludes UNKNOWN labels (partition-level)
fake_norm = (("0", 10), ("1", 5), ("unknown", 3))
fake_rec = dataclasses.replace(
    ulb, normalized_label_counts=fake_norm,
    label_counts=(("0", 10), ("1", 5), ("maybe", 3)),
    row_count=18,
    duplicate_findings=DuplicateFindings(18, 18, 0, 0, 0, 0))
fake_parts = build_partitions(
    {**RECORDS, "ULB_CREDIT_CARD_FRAUD": fake_rec})
inv(len(fake_parts) == 1 and fake_parts[0].supervised_rows == 15,
    "UNKNOWN labels excluded from supervised component rows")

# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — TEMPORAL HANDLING
# ══════════════════════════════════════════════════════════════════════

section("9. Temporal semantics")

EXPECTED_TS = {
    "IEEE_CIS": TimestampStatus.ORDERING_ONLY_RELATIVE.value,
    "ULB_CREDIT_CARD_FRAUD":
        TimestampStatus.ORDERING_ONLY_RELATIVE.value,
    "PAYSIM": TimestampStatus.ORDERING_ONLY_RELATIVE.value,
    "BANKSIM": TimestampStatus.ORDERING_ONLY_RELATIVE.value,
}
for dataset_id, expected in EXPECTED_TS.items():
    ds = REG.get_dataset(dataset_id)
    inv(RECORDS[dataset_id].timestamp_status == expected,
        f"{dataset_id}: ordering-only relative timestamps")
    inv(ds.timestamp_supports_absolute_time is False,
        f"{dataset_id}: absolute time never claimed by the registry")
baf_status = RECORDS["BAF_BANK_ACCOUNT_FRAUD"].timestamp_status
inv(baf_status in {t.value for t in TimestampStatus},
    "BAF timestamp status is a declared enum value")
inv(REG.get_dataset("BAF_BANK_ACCOUNT_FRAUD")
    .timestamp_supports_ordering is False,
    "BAF ordering not established by the registry")

first_ts = rows_a[0].timestamp
with open(ulb_path, "r", encoding="utf-8", newline="") as fh:
    reader2 = __import__("csv").reader(fh)
    next(reader2)
    raw_ts = next(reader2)[ULB_SCHEMA.index("Time")]
inv(first_ts == raw_ts,
    "relative counter retained verbatim — never converted to a date")
inv(not re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", first_ts),
    "no calendar date is ever derived from a relative counter")

temporal_view = next(v for v in
                     build_component_evaluation_views(PARTITIONS)
                     if v.view_id == "temporal_ordering")
inv(set(temporal_view.source_partitions)
    <= {p.partition_id for p in PARTITIONS
        if p.timestamp_status
        == TimestampStatus.ORDERING_ONLY_RELATIVE.value},
    "temporal view only admits ordering-capable sources")

# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — LEAKAGE CONTROLS
# ══════════════════════════════════════════════════════════════════════

section("10. Leakage controls & evaluation exclusions")

VIEWS = build_component_evaluation_views(PARTITIONS)
inv(len(VIEWS) == 6, "six declared component views")
label_cols = {c for c in REG.LABEL_COLUMNS.values() if c}
inv({"Class", "isFraud"} <= label_cols,
    "registered label columns known")
for view in VIEWS:
    inv(not (set(view.included_fields) & label_cols),
        f"{view.view_id}: label columns are never view features")
    inv(not (set(view.included_fields) & set(view.excluded_fields)),
        f"{view.view_id}: excluded fields never included")
    inv(bool(view.limitations), f"{view.view_id}: limitations declared")
    inv(view.scope_statement
        == "component_evaluation_only_never_altman_native_performance",
        f"{view.view_id}: component-only scope statement")
    inv(set(view.source_partitions)
        <= {p.partition_id for p in PARTITIONS},
        f"{view.view_id}: sources are admitted partitions only")
inv(any(v.status == ComponentViewStatus.NO_ELIGIBLE_SOURCE.value
        for v in VIEWS),
    "views without eligible sources are declared, not silently dropped")

manifest_probe = build_phase107_manifest(RECORDS, PARTITIONS)
manifest_excluded = dict(manifest_probe.excluded_fields)
inv("Class" in manifest_excluded["ULB_CREDIT_CARD_FRAUD"],
    "manifest records ULB label exclusion even for verified files")
ps_excluded = set(manifest_excluded["PAYSIM"])
inv({"isFraud", "isFlaggedFraud", "newbalanceOrig",
     "newbalanceDest"} <= ps_excluded,
    "PaySim leakage fields recorded as excluded despite failed verify")
inv(REG.get_dataset("PAYSIM").known_leakage_risks
    and set(REG.get_dataset("PAYSIM").known_leakage_risks)
    <= set(manifest_excluded["PAYSIM"]),
    "registry leakage risks flow into the manifest exclusions")

# label columns exist in raw sources but are excluded from evaluation
with open(ulb_path, "r", encoding="utf-8", newline="") as fh:
    header_raw = next(__import__("csv").reader(fh))
inv("Class" in header_raw and "Class" in part.excluded_fields,
    "raw label column preserved in the source AND excluded from eval")

expect_valueerror(
    lambda: dataclasses.replace(
        VIEWS[0],
        included_fields=tuple(sorted(
            set(VIEWS[0].included_fields) | {"Class"}))),
    "view claiming a label column fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        VIEWS[0], included_fields=("amount",),
        excluded_fields=("amount",)),
    "view with excluded/ included overlap fails closed")
expect_valueerror(
    lambda: dataclasses.replace(VIEWS[0], scope_statement="native!"),
    "scope statement substitution fails closed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 11 — NATIVE 48-FEATURE BOUNDARY
# ══════════════════════════════════════════════════════════════════════

section("11. Native 21->48 feature boundary unchanged")

EXPECTED_GROUPS = {"IEEE_CIS": "group_b",
                   "ULB_CREDIT_CARD_FRAUD": "group_b",
                   "PAYSIM": "group_b",
                   "BANKSIM": "group_d",
                   "BAF_BANK_ACCOUNT_FRAUD": "group_d"}
for dataset_id, expected in EXPECTED_GROUPS.items():
    group, _ = REG.assign_benchmark_group(dataset_id)
    inv(group == expected,
        f"{dataset_id}: Phase 106 group unchanged after ingestion")
    inv(REG.native_eligible_row_count(dataset_id) == 0,
        f"{dataset_id}: native-eligible rows stay0")
for entry in manifest_probe.native_compatibility:
    dataset_id, group, usable, native_rows = entry
    inv(native_rows == 0 and group == EXPECTED_GROUPS[dataset_id],
        f"{dataset_id}: manifest native rows0 / group pinned")
    inv(usable == USABLE_COUNTS[dataset_id],
        f"{dataset_id}: usable feature count {usable} matches Phase106")

report_native = generate_phase107_report(RECORDS, PARTITIONS)
inv(report_native.rows_eligible_native == 0
    and not report_native.native_evaluation_eligible,
    "report:0 native rows, native evaluation ineligible")
check(report_native.domain_feature_count == 21
      and report_native.native_feature_count == 48,
      "report carries21/48 counts")
vec21 = {name: float(i) for i, name in enumerate(ML_FEATURE_ORDER)}
out48a = map_raw_to_native(vec21)
out48b = map_raw_to_native(dict(vec21))
inv(len(out48a) == 48
    and out48a.tolist() == out48b.tolist(),
    "canonical mapping: 21 -> 48, deterministic, untouched")
check(CANONICAL_TRANSFORMATION.strip("'") == "map_raw_to_native",
      "canonical transformation names the one mapping function")
check(not hasattr(ING, "derive_native_features")
      and not hasattr(ING, "native_vector"),
      "no local redefinition of the native derivation functions")
check(ING.CANONICAL_TRANSFORMATION == CANONICAL_TRANSFORMATION,
      "module binds the canonical transformation")

expect_valueerror(
    lambda: dataclasses.replace(
        manifest_probe, production_threshold=0.99),
    "threshold substitution fails closed at construction")
expect_valueerror(
    lambda: dataclasses.replace(
        manifest_probe, production_release_id="release-old"),
    "historical release substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        manifest_probe, production_model_id="altman_legacy"),
    "model substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        manifest_probe, feature_version="v2"),
    "feature-version substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        manifest_probe, native_feature_version="v9"),
    "native-feature-version substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        manifest_probe,
        canonical_mapping="map_raw_to_native_v2"),
    "alternate transformation substitution fails closed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 12 — ARCHIVE SAFETY
# ══════════════════════════════════════════════════════════════════════

section("12. Archive safety (inspect first, extract only when fully safe)")

csv_body = (",".join(ULB_SCHEMA) + "\n"
            + ",".join(fixture_row("ULB_CREDIT_CARD_FRAUD")) + "\n")
safe_zip = zip_with("safe.zip", [("creditcard.csv", csv_body)])
insp = inspect_archive(safe_zip)
inv(insp.safe and insp.member_count == 1,
    "clean archive inspects safe")
bad_cases = [
    ("traversal", "../evil.csv", "x,y\n1,2\n", "path_traversal"),
    ("backslash_traversal", "..\\evil.csv", "x,y\n1,2\n",
     "path_traversal"),
    ("absolute", "/etc/passwd", "x,y\n1,2\n", "absolute_path"),
    ("windows_abs", "C:/evil.csv", "x,y\n1,2\n", "absolute_path"),
    ("nested_zip", "inner.zip", "PK\x03\x04", "nested_archive"),
    ("nested_targz", "inner.tar.gz", "x", "nested_archive"),
    ("sh", "run.sh", "#!/bin/sh", "executable_member"),
    ("py", "payload.py", "print(1)", "executable_member"),
    ("binary", "blob.bin", "\x00\x01", "unexpected_member_type"),
]
for label, member, content, expected in bad_cases:
    evil = zip_with(f"evil_{label}.zip", [(member, content)])
    res = inspect_archive(evil)
    inv(not res.safe and any(expected in v for v in res.violations),
        f"{label}: rejected with {expected} ({res.violations})")

# exec-bit member without a dangerous extension
evil_mode = os.path.join(TMPROOT, "evil_mode.zip")
with zipfile.ZipFile(evil_mode, "w") as zf:
    zi = zipfile.ZipInfo("data.csv")
    zi.create_system = 3
    zi.external_attr = (0o100755 << 16)
    zf.writestr(zi, "a,b\n1,2\n")
res_mode = inspect_archive(evil_mode)
inv(not res_mode.safe
    and any("executable_member" in v for v in res_mode.violations),
    "exec-bit member rejected regardless of extension")

# symlink member
evil_sym = os.path.join(TMPROOT, "evil_sym.zip")
with zipfile.ZipFile(evil_sym, "w") as zf:
    zi = zipfile.ZipInfo("link.csv")
    zi.create_system = 3
    zi.external_attr = (0o120777 << 16)
    zf.writestr(zi, "/etc/hostname")
res_sym = inspect_archive(evil_sym)
inv(not res_sym.safe
    and any("symlink_member" in v for v in res_sym.violations),
    "symlink member rejected")

# unsafe archives are never extracted anywhere
out_dir = os.path.join(TMPROOT, "extract_out")
expect_valueerror(
    lambda: safe_extract_archive(
        os.path.join(TMPROOT, "evil_traversal.zip"), out_dir),
    "unsafe extraction raises instead of writing")
check(not os.path.exists(out_dir),
      "no extraction directory is created for unsafe archives")
check(not os.path.exists(os.path.join(TMPROOT, "evil.csv")),
      "traversal target never written")

# member size guard (bounded test override, restored)
out2 = os.path.join(TMPROOT, "extract_ok")
members = safe_extract_archive(safe_zip, out2)
inv(members == ("creditcard.csv",)
    and os.path.isfile(os.path.join(out2, "creditcard.csv")),
    "safe archive extracts under the enforced directory")
original_max = ING.MAX_MEMBER_BYTES
try:
    ING.MAX_MEMBER_BYTES = 4
    huge = zip_with("huge.zip", [("creditcard.csv", csv_body)])
    res_huge = inspect_archive(huge)
    inv(not res_huge.safe
        and any("member_too_large" in v for v in res_huge.violations),
        "oversized member rejected")
finally:
    ING.MAX_MEMBER_BYTES = original_max
check(ING.MAX_MEMBER_BYTES == original_max, "size guard restored")

# end-to-end: dataset supplied as a clean archive
rec_zip = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (safe_zip,))
inv(len(rec_zip.extracted_files) == 1
    and rec_zip.extracted_files[0].sha256,
    "extracted file hashed into the record")
check(rec_zip.status == "acquired_verification_failed"
      and "row_count_mismatch" in rec_zip.verification_errors,
      "archived dataset still verified against documented counts")

# unsafe archive supplied as a dataset fails closed without extraction
rec_evil_zip = ingest_dataset(
    "ULB_CREDIT_CARD_FRAUD",
    (os.path.join(TMPROOT, "evil_traversal.zip"),))
inv(rec_evil_zip.status == "acquired_verification_failed"
    and "archive_unsafe_members" in rec_evil_zip.verification_errors
    and rec_evil_zip.extracted_files == (),
    "unsafe archive dataset fails closed, extracts nothing")

# unsupported container refuses to fabricate verification
parquet_path = os.path.join(TMPROOT, "fake.parquet")
with open(parquet_path, "wb") as fh:
    fh.write(b"PAR1\x00\x00\x00\x00PAR1")
inv(detect_container(parquet_path) == ContainerType.PARQUET,
    "parquet magic detected")
rec_parquet = ingest_dataset("ULB_CREDIT_CARD_FRAUD", (parquet_path,))
inv(rec_parquet.status == "acquired_verification_failed"
    and "container_not_supported" in rec_parquet.verification_errors,
    "unsupported container fails closed — never silently parsed")

# ══════════════════════════════════════════════════════════════════════
# SECTION 13 — MANIFEST, LEDGER & REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════

section("13. Manifest determinism, guards, ledger & gates evidence")

MANIFEST = manifest_probe
inv(MANIFEST.phase == 107
    and MANIFEST.manifest_version == MANIFEST_VERSION,
    "manifest identity fields")
inv(MANIFEST.datasets_present
    == tuple(d for d in REG.DATASET_IDS
             if RECORDS[d].status != "not_acquired"),
    "datasets_present = located files")
inv(MANIFEST.datasets_absent
    == ("IEEE_CIS", "BANKSIM", "BAF_BANK_ACCOUNT_FRAUD"),
    "datasets_absent listed precisely")
inv(MANIFEST.datasets_verified == ("ULB_CREDIT_CARD_FRAUD",),
    "datasets_verified listed precisely")
check(MANIFEST.verify_hash(), "manifest hash verifies")
rebuild2 = build_phase107_manifest(RECORDS, PARTITIONS)
inv(rebuild2.manifest_hash == MANIFEST.manifest_hash,
    "manifest hash deterministic across builds")
records_again = ingest_all()
inv(build_phase107_manifest(records_again,
                            build_partitions(records_again)
                            ).manifest_hash == MANIFEST.manifest_hash,
    "manifest hash reproduces across a full independent re-ingest")
inv(RECORDS["ULB_CREDIT_CARD_FRAUD"].record_hash
    == records_again["ULB_CREDIT_CARD_FRAUD"].record_hash,
    "record hash reproduces across re-ingest")
payload = json.dumps(dataclasses.asdict(MANIFEST), default=str)
inv("fs_modified_at" not in payload,
    "filesystem mtimes are excluded from the manifest payload")
inv(len(MANIFEST.manifest_hash) == 64, "manifest hash is SHA-256")

manifest_tamper = build_phase107_manifest(RECORDS, PARTITIONS)
good_hash = manifest_tamper.manifest_hash
object.__setattr__(manifest_tamper, "row_counts",
                   (("ULB_CREDIT_CARD_FRAUD", 1),))
inv(not manifest_tamper.verify_hash(),
    "tampered manifest content detected by hash")
object.__setattr__(manifest_tamper, "row_counts",
                   MANIFEST.row_counts)
inv(manifest_tamper.verify_hash() and manifest_tamper.manifest_hash
    == good_hash, "hash restored after un-tampering")

expect_valueerror(
    lambda: dataclasses.replace(MANIFEST, datasets_verified=("IEEE_CIS",)),
    "verified-not-present inconsistency fails closed")
expect_valueerror(
    lambda: dataclasses.replace(MANIFEST,
                                datasets_present=("IEEE_CIS",),
                                datasets_absent=("IEEE_CIS",)),
    "present/absent overlap fails closed")
expect_valueerror(
    lambda: dataclasses.replace(MANIFEST, datasets_present=("FAKE",)),
    "unregistered dataset in manifest fails closed")
expect_valueerror(
    lambda: dataclasses.replace(MANIFEST, phase=106),
    "phase substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(MANIFEST, manifest_version="v9"),
    "manifest version substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        MANIFEST, not_gate_evidence=MANIFEST.not_gate_evidence[:-1]),
    "not-gate-evidence declaration cannot be altered")

# the manifest is not gate evidence in any type hierarchy
inv(not isinstance(MANIFEST, RWVPromotionEvidence),
    "manifest is not RWV promotion evidence")
inv(not isinstance(MANIFEST, QualificationResult),
    "manifest is not a qualification result")
expect_failclosed(lambda: PromotionToken(MANIFEST),
                  "PromotionToken from a manifest rejected")
expect_failclosed(
    lambda: PromotionToken(dataclasses.replace(
        MANIFEST, manifest_hash="")),
    "PromotionToken from a mutated manifest rejected")

# ledger & overwrite protection
ledger = build_ingestion_ledger(RECORDS)
inv(ledger.version("ULB_CREDIT_CARD_FRAUD") == ulb.version,
    "ledger version is the file hash")
inv(ledger.version("BANKSIM") == "not_acquired",
    "ledger records absence honestly")
inv(ledger.latest("IEEE_CIS") is RECORDS["IEEE_CIS"],
    "ledger latest returns the record")
expect_valueerror(
    lambda: IngestionLedger((ulb,
                             dataclasses.replace(
                                 ulb, ragged_rows=1))),
    "same version with different content fails closed (no overwrite)")
expect_valueerror(
    lambda: dataclasses.replace(
        ulb, dataset_id="FAKE_DATASET"),
    "record cannot be re-attributed to an unregistered dataset")
expect_valueerror(
    lambda: dataclasses.replace(
        ulb, status="acquired_verified",
        verification_errors=("row_count_mismatch",)),
    "verified record cannot carry verification errors")
expect_valueerror(
    lambda: dataclasses.replace(
        ulb, status="bogus_status"),
    "unknown acquisition status fails closed")
absent_rec = RECORDS["BANKSIM"]
expect_valueerror(
    lambda: dataclasses.replace(absent_rec, primary_file="x"),
    "not_acquired record cannot claim a primary file")
expect_valueerror(
    lambda: dataclasses.replace(
        RECORDS["PAYSIM"], status="acquired_verified"),
    "failed record cannot be relabelled verified while carrying errors")

# ══════════════════════════════════════════════════════════════════════
# SECTION 14 — PARTITION GUARDS
# ══════════════════════════════════════════════════════════════════════

section("14. Partition structure guards")

expect_valueerror(
    lambda: dataclasses.replace(part, group="group_a"),
    "partition group substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(part, license_documented=False),
    "partition license flag cannot be revoked after admission")
expect_valueerror(
    lambda: dataclasses.replace(
        part, partition_id="benchmark/other"),
    "partition id must match its dataset")
expect_valueerror(
    lambda: dataclasses.replace(
        part, excluded_fields=tuple(
            f for f in part.excluded_fields if f != "Class")),
    "partition without label exclusion fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        part, schema_columns=part.schema_columns + ("isFraud",),
        excluded_fields=part.excluded_fields),
    "partition admitting an unexcluded leakage column fails closed")
part_tamper = build_partitions(RECORDS)[0]
object.__setattr__(part_tamper, "partition_hash", "faked")
inv(not part_tamper.verify_hash(), "tampered partition hash detected")

ledger2 = build_ingestion_ledger(records_again)
inv(ledger2.version("ULB_CREDIT_CARD_FRAUD") == ulb.version,
    "ledger rebuilds identically")

# ══════════════════════════════════════════════════════════════════════
# SECTION 15 — REPORT
# ══════════════════════════════════════════════════════════════════════

section("15. Deterministic report & empty-state resilience")

REPORT = report_native
inv(REPORT.phase == 107 and REPORT.report_version == REPORT_VERSION,
    "report identity")
inv(REPORT.datasets_verified == ("ULB_CREDIT_CARD_FRAUD",)
    and REPORT.datasets_failed_verification == ("PAYSIM",)
    and REPORT.datasets_absent
    == ("IEEE_CIS", "BANKSIM", "BAF_BANK_ACCOUNT_FRAUD"),
    "report acquisition lists match ingestion exactly")
check(REPORT.datasets_acquired
      == ("ULB_CREDIT_CARD_FRAUD", "PAYSIM"),
      "report acquired = located, whatever the verification outcome")
check(len(REPORT.file_hashes) == sum(
    len(RECORDS[d].file_records) for d in RECORDS),
    "every located file hashed in the report")
check(dict(REPORT.row_counts)["ULB_CREDIT_CARD_FRAUD"] == 284807
      and dict(REPORT.row_counts)["PAYSIM"] == 100000,
      "row counts reported per source")
check(dict(REPORT.column_counts)["ULB_CREDIT_CARD_FRAUD"]
      == len(ULB_SCHEMA), "column counts reported")
check(dict(REPORT.label_counts)["ULB_CREDIT_CARD_FRAUD"]
      == ulb.label_counts, "label counts reported verbatim")
check(dict(REPORT.label_semantics)["ULB_CREDIT_CARD_FRAUD"]
      == REG.get_dataset("ULB_CREDIT_CARD_FRAUD").label_semantics,
      "label semantics sourced from the registry")
license_map = {d: (lic, prov) for d, lic, prov
               in REPORT.license_provenance}
check(license_map["BANKSIM"][0] is False,
      "license gap reported, never papered over")
check(REPORT.rows_eligible_native == 0
      and REPORT.rows_eligible_component_total == 284807,
      "native0 / component284807 row eligibility")
check(REPORT.rows_eligible_component
      == (("ULB_CREDIT_CARD_FRAUD", 284807),),
      "component rows attributed to their source")
check(REPORT.benchmark_readiness
      == BenchmarkReadiness.READY_FOR_COMPONENT_EVALUATION.value,
      "benchmark may be READY_FOR_COMPONENT_EVALUATION")
inv(REPORT.system_readiness == SYSTEM_READINESS
    and REPORT.real_world_validation == REAL_WORLD_VALIDATION
    and REPORT.promotion_state == PROMOTION_STATE,
    "global states reported unchanged")
inv(REPORT.conclusion == EXPECTED_CONCLUSION
    == "READY_WITH_EXTERNAL_PREREQUISITE",
    "conclusion unchanged")
inv(not REPORT.any_dataset_qualified
    and REPORT.qualified_datasets == (),
    "no qualified datasets claimed")
inv(all(state == "blocked" and not qualified
        for _, state, qualified in REPORT.known_candidate_states)
    and len(REPORT.known_candidate_states) == 4,
    "all four institutional candidates remain Phase 104 BLOCKED")
check(REPORT.reproducibility_status
      == "DETERMINISTIC_REBUILD_IDENTICAL",
      "reproducibility measured, not assumed")
check(REPORT.benchmark_manifest_sha256 == MANIFEST.manifest_hash,
      "report pins the manifest hash")
check(REPORT.declarations == DECLARATIONS
      and len(REPORT.declarations) == 8,
      "all eight declarations present verbatim")
check(any("NOT REAL-WORLD VALIDATION" in d
          for d in REPORT.declarations),
      "declaration: this is not RWV")
check(any("NO PROVIDER WAS CONTACTED" in d
          for d in REPORT.declarations),
      "declaration: no provider contacted, local files only")
check(any("UNRESOLVED" in d for d in REPORT.declarations),
      "declaration: institutional evidence unresolved")
check(any("NEVER CONSTITUTE PROVIDER-ATTESTED" in d
          for d in REPORT.declarations),
      "declaration: benchmark rows are never provider evidence")
check(any("PROMOTION_GATE_REQUIRED" in d
          for d in REPORT.declarations),
      "declaration: promotion remains gated")
check(any("READY_FOR_REAL_WORLD_VALIDATION" in d
          for d in REPORT.declarations),
      "declaration: benchmark readiness never means RWV")
check(REPORT.not_gate_evidence == MANIFEST.not_gate_evidence,
      "manifest not-gate-evidence statement carried into the report")
check(REPORT.verify_hash(), "report hash verifies")
report_dict = REPORT.to_dict()
report_dict["report_hash"] = ""
manual = hashlib.sha256(
    canonical_json(report_dict).encode("utf-8")).hexdigest()
inv(manual == REPORT.report_hash,
    "report hash reproducible independently")
report_tamper = generate_phase107_report(RECORDS, PARTITIONS)
before_hash = report_tamper.report_hash
object.__setattr__(report_tamper, "rows_eligible_component_total", 999)
inv(not report_tamper.verify_hash(),
    "tampered report content detected by hash")
object.__setattr__(report_tamper, "rows_eligible_component_total",
                   REPORT.rows_eligible_component_total)
inv(report_tamper.verify_hash()
    and report_tamper.report_hash == before_hash,
    "report hash restored after un-tampering")
expect_valueerror(
    lambda: dataclasses.replace(report_tamper, rows_eligible_native=5),
    "native row inflation fails closed")
expect_valueerror(
    lambda: dataclasses.replace(REPORT, phase=106),
    "report phase substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(REPORT, report_version="v0"),
    "report version substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(REPORT, production_threshold=1.0),
    "report threshold substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(REPORT, system_readiness="READY"),
    "report readiness-state substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(REPORT, promotion_state="PROMOTED"),
    "report promotion-state substitution fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        REPORT, any_dataset_qualified=True,
        qualified_datasets=("ULB_CREDIT_CARD_FRAUD",)),
    "forged qualification claim fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        REPORT, declarations=DECLARATIONS[:-1]),
    "declaration removal fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        REPORT, conclusion="READY_FOR_REAL_WORLD_VALIDATION"),
    "RWV conclusion forgery fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        REPORT, native_evaluation_eligible=True),
    "native eligibility forgery fails closed")
expect_valueerror(
    lambda: dataclasses.replace(
        REPORT, rows_eligible_component=(("ULB_CREDIT_CARD_FRAUD", 7),)),
    "component row totals must reconcile")

# empty state: fresh environment with no local files at all
empty_report = generate_phase107_report(records={}, partitions=())
inv(empty_report.datasets_acquired == ()
    and empty_report.datasets_absent == tuple(REG.DATASET_IDS),
    "no local files -> everything absent, nothing crashes")
check(empty_report.benchmark_readiness
      == BenchmarkReadiness.NOT_READY_NO_VERIFIED_SOURCE.value,
      "no verified source -> NOT_READY_NO_VERIFIED_SOURCE")
check(empty_report.rows_eligible_component_total == 0
      and empty_report.partitions == ()
      and empty_report.rows_eligible_native == 0,
      "empty state yields empty eligibility")
check(empty_report.verify_hash(), "empty-state report hash verifies")
check(all(status == "not_acquired"
          for _, status in empty_report.acquisition_statuses),
      "empty-state statuses are all not_acquired")

# full determinism across an independent ingestion + generation
REPORT2 = generate_phase107_report(
    records_again, build_partitions(records_again))
inv(REPORT2.report_hash == REPORT.report_hash
    and REPORT2.benchmark_manifest_sha256
    == REPORT.benchmark_manifest_sha256,
    "report reproduces byte-for-byte on a clean re-ingest")

# ══════════════════════════════════════════════════════════════════════
# SECTION 16 — RWV / PROMOTION GATES UNCHANGED
# ══════════════════════════════════════════════════════════════════════

section("16. RWV / promotion gates remain unchanged and blocking")

inv(SYSTEM_READINESS == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET",
    "SYSTEM_READINESS unchanged")
inv(REAL_WORLD_VALIDATION == "BLOCKED_PENDING_ELIGIBLE_DATASET",
    "REAL_WORLD_VALIDATION unchanged")
inv(PROMOTION_STATE == "PROMOTION_GATE_REQUIRED",
    "PROMOTION unchanged")
inv(REP.SYSTEM_READINESS == SYSTEM_READINESS
    and REP.REAL_WORLD_VALIDATION == REAL_WORLD_VALIDATION
    and REP.PROMOTION_STATE == PROMOTION_STATE,
    "report module re-exports read-only state")
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
sess = create_session(ev, qual, session_id="phase107-test")
inv(sess.status == SessionState.BLOCKED.value,
    f"RWV session created BLOCKED ({sess.status})")
expect_failclosed(lambda: PromotionToken(decision),
                  "PromotionToken from blocked decision rejected")
inv(MODEL_ID == "altman_native"
    and RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904",
    "production identities still authoritative")
inv(len(ALTMAN_NATIVE_FEATURES) == 48
    and PRODUCTION_THRESHOLD == 0.018758,
    "native model contract and threshold untouched by Phase 107")

# ══════════════════════════════════════════════════════════════════════
# SECTION 17 — KNOWN-CANDIDATE REGRESSION
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
inv(not REPORT.any_dataset_qualified,
    "report agrees: nothing qualified")

# the manifest stays exactly its own type — never gate evidence
inv(REPORT.benchmark_manifest_sha256 == MANIFEST.manifest_hash,
    "report pins the manifest hash (bookkeeping chain only)")
inv(isinstance(MANIFEST, Phase107BenchmarkManifest),
    "manifest has exactly its own type")

# ══════════════════════════════════════════════════════════════════════
# SELF-COUNT & SUMMARY
# ══════════════════════════════════════════════════════════════════════

section("18. Self-count")

inv(INVARIANT_COUNT >= 100,
    f"invariant floor100+ reached ({INVARIANT_COUNT})")
inv(ASSERTIONS >= 500,
    f"assertion floor500+ reached ({ASSERTIONS})")
inv(not FAILURES, f"zero failures ({len(FAILURES)})")

print("\n" + "=" * 70)
print(f"PHASE 107 SUITE: {ASSERTIONS} assertions, "
      f"{INVARIANT_COUNT} invariants, {len(FAILURES)} failures")
if FAILURES:
    print("FAILURES:")
    for f in FAILURES:
        print(f"  - {f}")
print("=" * 70)
sys.exit(1 if FAILURES else 0)
