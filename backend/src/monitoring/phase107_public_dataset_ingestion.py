"""Phase 107: Public Dataset Acquisition, Verified Ingestion & Benchmark Assembly.

Deterministic, local-only ingestion of the Phase 106 registered public
benchmark candidates from files the user has ALREADY acquired manually,
followed by source-preserving benchmark assembly.

    manual local files -> verification -> partitions -> combined benchmark
    -> component-evaluation views -> Phase 108 evaluation

THIS IS PUBLIC BENCHMARK TESTING.  THIS IS NOT REAL-WORLD VALIDATION.

Hard boundaries (enforced by this module's structure and by the Phase 107
suite's source scans):

- NO network access, NO downloads: files must already exist locally at
  documented acquisition paths; this module never fetches anything.
- Local file reads only (stdlib csv/json/zipfile).  No pickle, no joblib,
  no trusted-deserialization of external artifacts.  Archives are
  inspected member-by-member and only extracted after every safety check
  passes (path traversal, absolute paths, symlinks, executables, nested
  archives, unexpected types, oversized members are all rejected first;
  extraction is per-member with containment checks — never extract-all).
- Source identity is preserved on every row (source_dataset_id +
  deterministic source_row_id).  No customer/merchant/geography/MCC/
  history identity is ever invented.  Local row ids are labelled as
  local position hashes, never as original identities.
- Original labels are preserved verbatim; normalized labels are produced
  only by the Phase 106 authoritative normalize_label(); unmappable label
  values become UNKNOWN and are excluded from supervised component views.
- Known leakage fields (Phase 106) stay in the raw source representation
  and are recorded on an explicit evaluation exclusion list; they are
  never model features.
- The canonical feature pipeline (21 domain/runtime features ->
  map_raw_to_native() -> derive_native_features() -> native_vector() ->
  48 Altman-Native features) is BOUND, never redefined.  Missing native
  features are never fabricated or imputed; Group B/D datasets are never
  promoted to Group A by this phase.
- The Phase 107 benchmark manifest is metadata for the public benchmark
  only.  It reuses BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE: it is not any
  form of qualification evidence and no existing gate accepts it.
- No bypass parameters (force / allow_unverified / skip_validation /
  override / admin_override / bypass) exist anywhere in this module;
  the parameter list of every public function is pinned by the suite.
- No second qualification authority: Phase 104 stays authoritative for
  dataset qualification, Phase 105 for provider evidence intake, Phase
  106 for candidate discovery/compatibility classification.

Run:  ../.venv/Scripts/python.exe scripts/phase107_public_dataset_ingestion_test.py
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Mapping, Sequence

from src.monitoring.external_dataset_contract import (
    CANONICAL_TRANSFORMATION,
    canonical_json,
)
from src.monitoring.feature_contract import ML_FEATURE_ORDER
from src.monitoring.phase106_public_benchmark_registry import (
    AMOUNT_COLUMNS,
    BENCHMARK_DESCRIPTOR,
    BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
    DATASET_IDS,
    FORBIDDEN_BYPASS_PARAMETERS,
    GROUP_READINESS,
    LABEL_COLUMNS,
    TIMESTAMP_COLUMNS,
    BenchmarkEvaluationConfig,
    BenchmarkGroup,
    HarmonizationStatus,
    HarmonizedRow,
    assign_benchmark_group,
    build_evaluation_config,
    dataset_preflight,
    COMMON_BENCHMARK_FIELDS,
    find_single_source_claims,
    get_dataset,
    harmonize_rows,
    manual_acquisition_instructions,
    normalize_label,
    row_fingerprint,
    scan_leakage_fields,
    usable_features,
)
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES
from src.monitoring.rwv_readiness_audit import FEATURE_VERSION, MODEL_ID, RELEASE_ID
from src.monitoring.rwv_reproducibility import NATIVE_FEATURE_VERSION

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS (read-only bindings to the existing authorities)
# ══════════════════════════════════════════════════════════════════════

INGESTION_VERSION = "phase107_ingestion_v1"
MANIFEST_VERSION = "phase107_benchmark_manifest_v1"
MANIFEST_CREATED_AT = "2026-09-23T00:00:00+00:00"
BENCHMARK_NAME = "PUBLIC_MULTI_SOURCE_FRAUD_BENCHMARK"
ACQUISITION_ROOT = "data/external_benchmark"
PHASE = 107

# Read-only copy of the locked production threshold.  The suite cross-
# checks this value against the protocol authority; nothing in this
# module may change it.
EXPECTED_PRODUCTION_THRESHOLD = 0.018758

DOMAIN_FEATURE_COUNT = len(ML_FEATURE_ORDER)      # 21
NATIVE_FEATURE_COUNT = len(ALTMAN_NATIVE_FEATURES)  # 48

DATASET_DIRNAMES: dict[str, str] = {
    "IEEE_CIS": "ieee_cis",
    "ULB_CREDIT_CARD_FRAUD": "ulb",
    "PAYSIM": "paysim",
    "BANKSIM": "banksim",
    "BAF_BANK_ACCOUNT_FRAUD": "baf",
}

# Ordered acquisition candidates per dataset (repo-root-relative).
# Spec structure first, then the Phase 106 documented location, then
# legacy manual drops already present in this workspace.  Only local
# paths; nothing here is ever fetched.
CANDIDATE_PATHS: dict[str, tuple[str, ...]] = {
    "IEEE_CIS": (
        "data/external_benchmark/ieee_cis/train_transaction.csv",
        "data/external_benchmark/ieee_cis/train_identity.csv",
        "data/external/IEEE_CIS/train_transaction.csv",
        "data/external/IEEE_CIS/train_identity.csv",
    ),
    "ULB_CREDIT_CARD_FRAUD": (
        "data/external_benchmark/ulb/creditcard.csv",
        "data/external/ULB_CREDIT_CARD_FRAUD/creditcard.csv",
        "data/creditcard.csv",
    ),
    "PAYSIM": (
        "data/external_benchmark/paysim/payment_sim.csv",
        "data/external_benchmark/paysim/"
        "ps_20174392719_1491204439457_log.csv",
        "data/external/PAYSIM/payment_sim.csv",
        "data/external/PAYSIM/"
        "ps_20174392719_1491204439457_log.csv",
        "data/paysim.csv",
        "data/paysim_1m.csv",
    ),
    "BANKSIM": (
        "data/external_benchmark/banksim/banksim.csv",
        "data/external/BANKSIM/banksim.csv",
    ),
    "BAF_BANK_ACCOUNT_FRAUD": (
        "data/external_benchmark/baf/base.csv",
        "data/external/BAF_BANK_ACCOUNT_FRAUD/base.csv",
    ),
}

PRIMARY_BASENAMES: dict[str, str] = {
    "IEEE_CIS": "train_transaction.csv",
    "ULB_CREDIT_CARD_FRAUD": "creditcard.csv",
    "PAYSIM": "payment_sim.csv",
    "BANKSIM": "banksim.csv",
    "BAF_BANK_ACCOUNT_FRAUD": "base.csv",
}

# Datasets whose registered schema spans multiple files; ingestion is
# incomplete (and cannot be verified) until every file is present.
MULTIFILE_REQUIRED: dict[str, tuple[str, ...]] = {
    "IEEE_CIS": ("train_transaction.csv", "train_identity.csv"),
}

# Phase 107 column->canonical mapping declarations.  These only state
# WHERE a documented source column sits relative to the canonical
# candidate fields; they are not feature transformations.
TRANSACTION_TYPE_COLUMNS: dict[str, tuple[str, ...]] = {
    "PAYSIM": ("type",),
}
MERCHANT_CATEGORY_COLUMNS: dict[str, tuple[str, ...]] = {
    "BANKSIM": ("category",),
}
CROSS_SOURCE_ENTITY_COLUMNS: dict[str, tuple[str, ...]] = {
    "IEEE_CIS": ("cards", "addr"),
    "PAYSIM": ("nameOrig", "nameDest"),
}

CANONICAL_CANDIDATE_FIELDS: tuple[str, ...] = (
    "original_label",
    "normalized_label",
    "timestamp",
    "amount",
    "transaction_type",
    "merchant_category",
    "entity_identifier",
    "mcc",
)

SAFE_MEMBER_SUFFIXES: tuple[str, ...] = (
    ".csv", ".json", ".jsonl", ".txt", ".md", ".parquet",
)
ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".zst",
)
EXECUTABLE_SUFFIXES: tuple[str, ...] = (
    ".sh", ".bat", ".cmd", ".exe", ".dll", ".so", ".dylib",
    ".py", ".pyc", ".pyo", ".jar", ".ps1", ".msi", ".apk",
)
MAX_MEMBER_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB per archive member

_BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(_BACKEND)


# ══════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════

class AcquisitionStatus(str, Enum):
    """Spec §ACQUISITION: absence never fails the phase; presence is
    always fully verified, never silently skipped."""
    NOT_ACQUIRED = "not_acquired"
    INCOMPLETE_ACQUISITION = "incomplete_acquisition"
    ACQUIRED_VERIFIED = "acquired_verified"
    ACQUIRED_VERIFICATION_FAILED = "acquired_verification_failed"


class ContainerType(str, Enum):
    CSV = "csv"
    JSON = "json"
    ZIP = "zip"
    PARQUET = "parquet"
    UNSUPPORTED = "unsupported"
    MISSING = "missing"


class MappingType(str, Enum):
    """Spec §SOURCE-PRESERVING NORMALIZATION mapping types."""
    DIRECT = "direct"
    DETERMINISTIC_TRANSFORM = "deterministic_transform"
    NOT_AVAILABLE = "not_available"
    SEMANTICALLY_INCOMPATIBLE = "semantically_incompatible"
    EXCLUDED_LEAKAGE = "excluded_leakage"


class TimestampStatus(str, Enum):
    ORDERING_ONLY_RELATIVE = "ordering_only_relative"
    UNKNOWN = "unknown"
    NOT_AVAILABLE = "not_available"


class ComponentViewStatus(str, Enum):
    ELIGIBLE = "eligible"
    NO_ELIGIBLE_SOURCE = "no_eligible_source"
    VALIDITY_NOT_ESTABLISHED = "validity_not_established"


class BenchmarkReadiness(str, Enum):
    READY_FOR_COMPONENT_EVALUATION = "READY_FOR_COMPONENT_EVALUATION"
    NOT_READY_NO_VERIFIED_SOURCE = "NOT_READY_NO_VERIFIED_SOURCE"


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _stable(payload: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 over the canonical JSON encoding."""
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")).hexdigest()


def _strip_keys(obj: Any, keys: frozenset[str]) -> Any:
    """Recursively drop non-content keys before hashing."""
    if isinstance(obj, dict):
        return {k: _strip_keys(v, keys) for k, v in obj.items()
                if k not in keys}
    if isinstance(obj, (list, tuple)):
        return [_strip_keys(v, keys) for v in obj]
    return obj


def _resolve(path: str) -> str:
    """Repo-relative paths resolve against the repository root; absolute
    paths (test fixtures) are used as given."""
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, *path.split("/"))


def _display(path: str) -> str:
    """Repo-root-relative POSIX path when possible, else absolute."""
    abs_path = _resolve(path)
    rel = os.path.relpath(abs_path, REPO_ROOT)
    if not rel.startswith(".."):
        return rel.replace(os.sep, "/")
    return abs_path.replace(os.sep, "/")


def sha256_file(path: str) -> str:
    """Streaming SHA-256 of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_container(path: str) -> ContainerType:
    """Extension + magic-byte container detection (never executes)."""
    if not os.path.isfile(path):
        return ContainerType.MISSING
    with open(path, "rb") as fh:
        magic = fh.read(4)
    lower = path.lower()
    if magic == b"PAR1":
        return ContainerType.PARQUET
    if magic[:2] == b"PK":
        return ContainerType.ZIP
    if lower.endswith((".csv", ".tsv")):
        return ContainerType.CSV
    if lower.endswith((".json", ".jsonl", ".ndjson")):
        return ContainerType.JSON
    if lower.endswith(".parquet"):
        return ContainerType.PARQUET
    if lower.endswith(ARCHIVE_SUFFIXES):
        return ContainerType.UNSUPPORTED
    return ContainerType.UNSUPPORTED


def acquisition_instructions(dataset_id: str) -> str:
    """Manual acquisition instructions: Phase 106 guidance plus the
    Phase 107 expected local paths.  Manual only — never automated."""
    get_dataset(dataset_id)  # unknown dataset fails closed
    paths = "\n".join(f"  {_p}" for _p in CANDIDATE_PATHS[dataset_id])
    return (
        f"{manual_acquisition_instructions(dataset_id)}\n"
        f"Phase 107 expected local acquisition paths (first existing "
        f"candidate is verified):\n{paths}"
    )


def resolve_candidate_paths(dataset_id: str) -> tuple[str, ...]:
    """Located (existing) candidate files for a dataset, in documented
    order.  Absence is NOT_ACQUIRED, never an error."""
    get_dataset(dataset_id)  # unknown dataset fails closed
    return tuple(
        p for p in CANDIDATE_PATHS[dataset_id]
        if os.path.isfile(_resolve(p))
    )


def _primary_path(dataset_id: str, located: Sequence[str]) -> str:
    """Prefer the registered primary basename, else the first located."""
    wanted = PRIMARY_BASENAMES[dataset_id]
    for p in located:
        if os.path.basename(p) == wanted:
            return p
    return located[0]


# ── table readers (local files only; no external loaders) ─────────────

def _iter_csv(path: str) -> tuple[tuple[str, ...], Iterator[list[str]]]:
    fh = open(path, "r", encoding="utf-8-sig", newline="")  # noqa: SIM115
    reader = csv.reader(fh)
    try:
        header = next(reader)
    except StopIteration:
        fh.close()
        raise ValueError(f"empty data file: {os.path.basename(path)}")
    def _rows() -> Iterator[list[str]]:
        try:
            for row in reader:
                yield row
        finally:
            fh.close()
    return tuple(header), _rows()


def _iter_json(path: str) -> tuple[tuple[str, ...], Iterator[list[str]]]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.lower().endswith((".jsonl", ".ndjson")):
        objects = [json.loads(line) for line in text.splitlines()
                   if line.strip()]
    else:
        objects = json.loads(text)
        if isinstance(objects, dict):
            objects = objects.get("records", [objects])
    if not objects or not isinstance(objects[0], dict):
        raise ValueError("JSON dataset must be an array of objects")
    header = tuple(objects[0].keys())
    def _rows() -> Iterator[list[str]]:
        for obj in objects:
            if not isinstance(obj, dict):
                yield []
                continue
            yield ["" if obj.get(k) is None else str(obj.get(k))
                   for k in header]
    return header, _rows()


# ── archive safety (inspect first, extract only when fully safe) ─────

@dataclass(frozen=True)
class ArchiveInspection:
    """Deterministic member-level safety verdict for a local archive."""
    archive_path: str
    member_count: int
    violations: tuple[str, ...]
    safe: bool


def _member_violation(name: str, info: zipfile.ZipInfo) -> str | None:
    """One violation reason for a member, or None when the member is
    acceptable.  Fail closed on anything unexpected."""
    if not name or name.strip() == "":
        return "empty_member_name"
    norm = name.replace("\\", "/")
    if norm.startswith("/") or (len(norm) > 1 and norm[1] == ":"):
        return f"absolute_path:{name}"
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if ".." in parts:
        return f"path_traversal:{name}"
    lower = norm.lower()
    if lower.endswith(ARCHIVE_SUFFIXES):
        return f"nested_archive:{name}"
    if lower.endswith(EXECUTABLE_SUFFIXES):
        return f"executable_member:{name}"
    if not lower.endswith(SAFE_MEMBER_SUFFIXES):
        return f"unexpected_member_type:{name}"
    mode = (info.external_attr >> 16) & 0xFFFF
    if (mode & 0o170000) == 0o120000:
        return f"symlink_member:{name}"
    if mode & 0o111:
        return f"executable_member:{name}"
    if info.file_size > MAX_MEMBER_BYTES:
        return f"member_too_large:{name}"
    return None


def inspect_archive(archive_path: str) -> ArchiveInspection:
    """Member-by-member safety inspection; extraction never happens as a
    side effect of inspecting."""
    rel = _display(archive_path)
    if detect_container(archive_path) != ContainerType.ZIP:
        return ArchiveInspection(rel, 0,
                                 ("not_a_supported_archive",), False)
    violations: list[str] = []
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        for info in infos:
            reason = _member_violation(info.filename, info)
            if reason is not None:
                violations.append(reason)
    return ArchiveInspection(rel, len(infos), tuple(violations),
                             not violations)


def safe_extract_archive(archive_path: str,
                         extraction_root: str) -> tuple[str, ...]:
    """Extract a fully-validated zip under extraction_root only.

    Raises ValueError when ANY member is unsafe — nothing is written in
    that case.  Per-member containment checks; never extract-all; never
    executes member content.  Returns extracted member names.
    """
    inspection = inspect_archive(archive_path)
    if not inspection.safe:
        raise ValueError(
            f"unsafe archive rejected: {inspection.violations}")
    root = os.path.realpath(extraction_root)
    extracted: list[str] = []
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            norm = info.filename.replace("\\", "/")
            dest = os.path.realpath(os.path.join(root, *norm.split("/")))
            if dest != root and not dest.startswith(root + os.sep):
                raise ValueError(
                    f"extraction escapes target dir: {info.filename}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as out:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    out.write(chunk)
            extracted.append(norm)
    return tuple(extracted)


# ══════════════════════════════════════════════════════════════════════
# INGESTION RECORDS
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FileRecord:
    """Identity of one verified local file (hash is the dataset
    version).  fs_modified_at is provenance only and is excluded from
    every deterministic hash so reports rebuild identically anywhere."""
    path: str
    sha256: str
    size_bytes: int
    container: str
    fs_modified_at: str


@dataclass(frozen=True)
class DuplicateFindings:
    """Deterministic duplicate classification for one source.

    unique_rows partitions exactly with within/cross rows:
        total = unique + within_source + cross_source
    normalized_collision_rows overlap that partition by design — rows
    sharing (label, timestamp, amount) fingerprints with differing raw
    content are reported, never merged and never deleted.
    """
    total_rows: int
    unique_rows: int
    within_source_duplicate_rows: int
    cross_source_duplicate_rows: int
    normalized_collision_rows: int
    normalized_collision_groups: int

    def __post_init__(self) -> None:
        if min(self.total_rows, self.unique_rows,
               self.within_source_duplicate_rows,
               self.cross_source_duplicate_rows,
               self.normalized_collision_rows,
               self.normalized_collision_groups) < 0:
            raise ValueError("duplicate finding counts must be >= 0")
        if (self.unique_rows + self.within_source_duplicate_rows
                + self.cross_source_duplicate_rows
                != self.total_rows):
            raise ValueError(
                "duplicate classes must partition the row total")


_ZERO_DUPLICATES = DuplicateFindings(0, 0, 0, 0, 0, 0)


@dataclass(frozen=True)
class IngestionRecord:
    """Immutable outcome of verifying one candidate dataset."""
    dataset_id: str
    status: str                       # AcquisitionStatus value
    located_files: tuple[str, ...]
    primary_file: str | None
    alternate_files: tuple[str, ...]
    file_records: tuple[FileRecord, ...]
    extracted_files: tuple[FileRecord, ...]
    archive_violations: tuple[str, ...]
    schema_expected: tuple[str, ...]
    schema_actual: tuple[str, ...]
    missing_columns: tuple[str, ...]
    renamed_columns: tuple[tuple[str, str], ...]
    extra_columns: tuple[str, ...]
    type_mismatch_counts: tuple[tuple[str, int], ...]
    ragged_rows: int
    row_count: int
    column_count: int
    label_counts: tuple[tuple[str, int], ...]
    normalized_label_counts: tuple[tuple[str, int], ...]
    row_count_check: str   # match | mismatch | not_documented | not_located | not_checked
    fraud_count_check: str # match | mismatch | not_documented | not_located | not_checked
    verification_errors: tuple[str, ...]
    verification_warnings: tuple[str, ...]
    row_id_kind: str
    timestamp_status: str
    duplicate_findings: DuplicateFindings | None
    record_hash: str

    def __post_init__(self) -> None:
        valid = {s.value for s in AcquisitionStatus}
        if self.status not in valid:
            raise ValueError(f"invalid acquisition status: {self.status}")
        if self.dataset_id not in DATASET_IDS:
            raise ValueError(
                f"unregistered dataset id: {self.dataset_id}")
        if self.status == AcquisitionStatus.NOT_ACQUIRED.value:
            if self.located_files or self.primary_file is not None:
                raise ValueError("not_acquired records carry no files")
        else:
            if not self.located_files or self.primary_file is None:
                raise ValueError(
                    "acquired records require located files")
        if (self.status == AcquisitionStatus.ACQUIRED_VERIFIED.value
                and self.verification_errors):
            raise ValueError("verified records cannot carry errors")
        if (self.status
                == AcquisitionStatus.ACQUIRED_VERIFICATION_FAILED.value
                and not self.verification_errors):
            raise ValueError("failed records require errors")
        if (self.status
                == AcquisitionStatus.INCOMPLETE_ACQUISITION.value
                and "required_files_missing"
                not in self.verification_errors):
            raise ValueError(
                "incomplete records require required_files_missing")
        if (self.status != AcquisitionStatus.NOT_ACQUIRED.value
                and self.duplicate_findings is None):
            raise ValueError("acquired records carry duplicate findings")
        if self.duplicate_findings is not None:
            if self.duplicate_findings.total_rows != self.row_count:
                raise ValueError(
                    "duplicate findings must cover the row total")
        # always derived from the validated content: replace() re-runs
        # this after re-validation, so a stale carried hash is never kept.
        object.__setattr__(self, "record_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return _stable(self._payload())

    def _payload(self) -> dict[str, Any]:
        return _strip_keys(
            dataclasses.asdict(self),
            frozenset({"record_hash", "fs_modified_at"}))

    def verify_hash(self) -> bool:
        return self.record_hash == self._compute_hash()

    @property
    def verified(self) -> bool:
        return (self.status
                == AcquisitionStatus.ACQUIRED_VERIFIED.value)

    @property
    def version(self) -> str:
        """Dataset version identity — the file hash.  A replaced file
        therefore always yields a NEW version."""
        if self.primary_file is None:
            return "not_acquired"
        primary = next(
            (f for f in self.file_records
             if f.path == _display(self.primary_file)), None)
        if primary is None:
            return "unhashed"
        return f"sha256:{primary.sha256}"


class DuplicateAccumulator:
    """Shared duplicate-detection state across per-dataset passes.

    observe() collects per-source maps, fold_local() moves them into
    the global view, and findings_for() classifies one source against
    the fully folded global view — classification therefore sees every
    source (ingest_many folds everything before classifying).  Nothing
    is ever deleted from the row totals: duplicates are classified,
    not removed.
    """

    def __init__(self) -> None:
        self._local_content: dict[str, dict[str, int]] = {}
        self._local_fp: dict[str, dict[str, list[Any]]] = {}
        self._global_content: dict[str, list[tuple[str, int]]] = {}
        # fingerprint -> [first_content, {dataset: count}, ambiguous]
        self._global_fp: dict[str, list[Any]] = {}
        self._folded: set[str] = set()

    def observe(self, dataset_id: str, content_hash: str,
                fingerprint: str | None) -> None:
        local_c = self._local_content.setdefault(dataset_id, {})
        local_c[content_hash] = local_c.get(content_hash, 0) + 1
        if fingerprint is None:
            return
        local_f = self._local_fp.setdefault(dataset_id, {})
        entry = local_f.get(fingerprint)
        if entry is None:
            local_f[fingerprint] = [content_hash, 1, False]
        else:
            entry[1] += 1
            if entry[0] != content_hash:
                entry[2] = True

    def fold_local(self, dataset_id: str) -> None:
        """Move one source's local maps into the global view.  Once
        per source (guarded); classification happens afterwards."""
        if dataset_id in self._folded:
            raise ValueError(f"duplicate fold twice: {dataset_id}")
        self._folded.add(dataset_id)
        local_c = self._local_content.pop(dataset_id, {})
        local_f = self._local_fp.pop(dataset_id, {})
        for content_hash, count in local_c.items():
            entry = self._global_content.get(content_hash)
            if entry is None:
                self._global_content[content_hash] = [
                    (dataset_id, count)]
            else:
                owners = [o for o, _ in entry]
                if dataset_id in owners:
                    idx = owners.index(dataset_id)
                    entry[idx] = (dataset_id, entry[idx][1] + count)
                else:
                    entry.append((dataset_id, count))
        for fp, (ref, count, differs) in local_f.items():
            entry = self._global_fp.get(fp)
            if entry is None:
                self._global_fp[fp] = [ref, {dataset_id: count},
                                       differs]
            else:
                if entry[0] != ref or differs:
                    entry[2] = True
                by_ds = entry[1]
                by_ds[dataset_id] = by_ds.get(dataset_id, 0) + count

    def findings_for(self, dataset_id: str, total_rows: int
                     ) -> DuplicateFindings:
        """Classify ONE source against the fully folded global view.
        Cross-source classification supersedes within-source, so both
        sides of a cross-source duplicate pair are marked."""
        within = 0
        cross = 0
        for owners in self._global_content.values():
            mine = next((c for o, c in owners if o == dataset_id),
                        None)
            if mine is None:
                continue
            if len(owners) > 1:
                cross += mine
            elif mine > 1:
                within += mine
        collision_rows = 0
        collision_groups = 0
        for _ref, by_ds, ambiguous in self._global_fp.values():
            if ambiguous and dataset_id in by_ds:
                collision_rows += by_ds[dataset_id]
                collision_groups += 1
        unique = total_rows - within - cross
        if unique < 0:
            raise ValueError("duplicate classification exceeds rows")
        return DuplicateFindings(
            total_rows=total_rows,
            unique_rows=unique,
            within_source_duplicate_rows=within,
            cross_source_duplicate_rows=cross,
            normalized_collision_rows=collision_rows,
            normalized_collision_groups=collision_groups,
        )

    @property
    def cross_source_groups(self) -> int:
        return sum(1 for owners in self._global_content.values()
                   if len(owners) > 1)


def detect_dataset_replacement(old: "IngestionRecord",
                               new: "IngestionRecord") -> str:
    """Re-ingestion verdict: a replaced local file must surface as a
    new version, never as a silent overwrite."""
    if old.dataset_id != new.dataset_id:
        return "dataset_id_mismatch"
    if old.version == new.version and old.record_hash == new.record_hash:
        return "identical"
    if old.version != new.version:
        return "primary_hash_changed"
    return "record_content_changed"


@dataclass(frozen=True)
class IngestionLedger:
    """Append-only view over ingestion records: the same dataset at two
    versions may coexist; a version is always its file hash."""
    entries: tuple[IngestionRecord, ...]

    def __post_init__(self) -> None:
        seen: dict[tuple[str, str], str] = {}
        for e in self.entries:
            key = (e.dataset_id, e.version)
            prior = seen.get(key)
            if prior is None:
                seen[key] = e.record_hash
            elif prior != e.record_hash:
                raise ValueError(
                    "reingest attempts to overwrite an existing version")
            elif e.status == AcquisitionStatus.NOT_ACQUIRED.value:
                raise ValueError("duplicate not_acquired ledger entry")

    def latest(self, dataset_id: str) -> IngestionRecord | None:
        matches = [e for e in self.entries
                   if e.dataset_id == dataset_id]
        return matches[-1] if matches else None

    def version(self, dataset_id: str) -> str | None:
        rec = self.latest(dataset_id)
        return rec.version if rec is not None else None


# ══════════════════════════════════════════════════════════════════════
# INGESTION
# ══════════════════════════════════════════════════════════════════════

def _file_record(path: str) -> FileRecord:
    st = os.stat(path)
    return FileRecord(
        path=_display(path),
        sha256=sha256_file(path),
        size_bytes=st.st_size,
        container=detect_container(path).value,
        fs_modified_at=str(int(st.st_mtime)),
    )


def _not_acquired(dataset_id: str) -> IngestionRecord:
    ds = get_dataset(dataset_id)
    return IngestionRecord(
        dataset_id=dataset_id,
        status=AcquisitionStatus.NOT_ACQUIRED.value,
        located_files=(),
        primary_file=None,
        alternate_files=(),
        file_records=(),
        extracted_files=(),
        archive_violations=(),
        schema_expected=tuple(ds.schema_columns),
        schema_actual=(),
        missing_columns=(),
        renamed_columns=(),
        extra_columns=(),
        type_mismatch_counts=(),
        ragged_rows=0,
        row_count=0,
        column_count=0,
        label_counts=(),
        normalized_label_counts=(),
        row_count_check="not_documented"
        if ds.transaction_count is None else "not_located",
        fraud_count_check="not_documented"
        if ds.fraud_count is None else "not_located",
        verification_errors=(),
        verification_warnings=("dataset_not_located",),
        row_id_kind="none",
        timestamp_status=_timestamp_status(dataset_id),
        duplicate_findings=_ZERO_DUPLICATES,
        record_hash="",
    )


def _timestamp_status(dataset_id: str) -> str:
    ds = get_dataset(dataset_id)
    if TIMESTAMP_COLUMNS.get(dataset_id) is None:
        return TimestampStatus.NOT_AVAILABLE.value
    if not ds.timestamp_supports_ordering:
        return TimestampStatus.UNKNOWN.value
    return TimestampStatus.ORDERING_ONLY_RELATIVE.value


def _check_header(dataset_id: str, header: tuple[str, ...]
                  ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...],
                             tuple[str, ...]]:
    """Missing (fatal), renamed (fatal, case/name drift) and extra
    (recorded) columns against the Phase 106 registered schema."""
    expected = get_dataset(dataset_id).schema_columns
    actual = list(header)
    actual_lower = {c.lower(): c for c in actual}
    missing: list[str] = []
    renamed: list[tuple[str, str]] = []
    for col in expected:
        if col in actual:
            continue
        hit = actual_lower.get(col.lower())
        if hit is not None and hit != col:
            renamed.append((col, hit))
        else:
            missing.append(col)
    renamed_set = {r[1] for r in renamed}
    extra = tuple(c for c in actual
                  if c not in expected and c not in renamed_set)
    return tuple(missing), tuple(renamed), extra


def ingest_dataset(
    dataset_id: str,
    candidate_paths: Sequence[str] | None = None,
    duplicate_accumulator: DuplicateAccumulator | None = None,
) -> IngestionRecord:
    """Verify one candidate dataset from local files.  Fail closed.

    candidate_paths overrides the documented acquisition locations for
    fixture/testing use; the file is still fully verified — an override
    path never relaxes a check.
    """
    ds = get_dataset(dataset_id)  # unknown dataset fails closed
    # explicit candidate paths are still existence-filtered: a supplied
    # path to a missing file is NOT_ACQUIRED, never a crash.
    raw = (tuple(candidate_paths) if candidate_paths is not None
           else CANDIDATE_PATHS[dataset_id])
    located = tuple(p for p in raw if os.path.isfile(_resolve(p)))
    if not located:
        return _not_acquired(dataset_id)

    errors: list[str] = []
    warnings: list[str] = []
    acc = (duplicate_accumulator if duplicate_accumulator is not None
           else DuplicateAccumulator())

    # multi-file completeness (Phase 106 registered schema spans files)
    required = MULTIFILE_REQUIRED.get(dataset_id, ())
    located_basenames = {os.path.basename(p) for p in located}
    missing_required = tuple(r for r in required
                             if r not in located_basenames)

    primary_rel = _primary_path(dataset_id, located)
    primary_abs = _resolve(primary_rel)
    primary_basename = os.path.basename(primary_rel)
    if primary_basename != PRIMARY_BASENAMES[dataset_id]:
        warnings.append("expected_filename_absent")
    if len(located) > 1:
        warnings.append("alternate_candidate_files_present")

    file_records = tuple(_file_record(_resolve(p)) for p in located)
    extracted_records: list[FileRecord] = []
    archive_violations: tuple[str, ...] = ()

    # container handling — data source may be an archive that must be
    # inspected and only then extracted under a contained directory.
    container = detect_container(primary_abs)
    data_paths: list[str] = []
    if container == ContainerType.ZIP:
        inspection = inspect_archive(primary_abs)
        if not inspection.safe:
            archive_violations = inspection.violations
            errors.append("archive_unsafe_members")
            errors.append("schema_not_verifiable")
        else:
            extraction_root = os.path.join(
                os.path.dirname(primary_abs),
                ".extracted", os.path.splitext(primary_basename)[0])
            safe_extract_archive(primary_abs, extraction_root)
            for member in sorted(os.listdir(extraction_root)):
                member_abs = os.path.join(extraction_root, member)
                if os.path.isfile(member_abs) and \
                        detect_container(member_abs) in (
                            ContainerType.CSV, ContainerType.JSON):
                    extracted_records.append(_file_record(member_abs))
                    data_paths.append(member_abs)
            if not data_paths:
                errors.append("archive_contains_no_data_file")
    elif container == ContainerType.PARQUET:
        errors.append("container_not_supported")
    elif container == ContainerType.UNSUPPORTED:
        errors.append("container_not_supported")
    elif container in (ContainerType.CSV, ContainerType.JSON):
        data_paths.append(primary_abs)
        # companion tables of a multi-file registered schema (e.g. the
        # IEEE-CIS identity table) join the schema union only — they
        # never replace the primary table or its count checks.
        for extra_path in located:
            resolved_extra = _resolve(extra_path)
            if resolved_extra == primary_abs:
                continue
            if detect_container(resolved_extra) in (
                    ContainerType.CSV, ContainerType.JSON):
                data_paths.append(resolved_extra)

    label_counts: dict[str, int] = {}
    normalized_counts: dict[str, int] = {}
    schema_actual: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    renamed_columns: tuple[tuple[str, str], ...] = ()
    extra_columns: tuple[str, ...] = ()
    type_mismatches: tuple[tuple[str, int], ...] = ()
    ragged_rows = 0
    row_count = 0
    column_count = 0
    fingerprint_ready = False

    if data_paths and "schema_not_verifiable" not in errors:
        # choose the primary data table: the registered basename wins
        # (extracted archives included), else the first located table;
        # companion tables extend the schema union only.
        primary_data = next(
            (p for p in data_paths
             if os.path.basename(p) == PRIMARY_BASENAMES[dataset_id]),
            data_paths[0])
        headers: list[tuple[str, ...]] = []
        for p in data_paths:
            if p == primary_data:
                continue
            if detect_container(p) in (ContainerType.CSV,
                                       ContainerType.JSON):
                try:
                    h, _ = (_iter_csv(p) if detect_container(p)
                            == ContainerType.CSV else _iter_json(p))
                    headers.append(h)
                except ValueError:
                    errors.append("companion_unreadable")
        try:
            primary_header, primary_rows = (
                _iter_csv(primary_data)
                if detect_container(primary_data) == ContainerType.CSV
                else _iter_json(primary_data))
        except ValueError:
            errors.append("data_file_unreadable")
            primary_header, primary_rows = (), iter(())
        if primary_header:
            schema_actual = primary_header
            column_count = len(primary_header)
            union = list(primary_header)
            for h in headers:
                union.extend(c for c in h if c not in union)
            missing_columns, renamed_columns, extra_columns = \
                _check_header(dataset_id, tuple(union))
            if missing_columns:
                errors.append("schema_missing_columns")
            if renamed_columns:
                errors.append("schema_renamed_columns")
            if extra_columns:
                warnings.append("schema_extra_columns")

            label_col = LABEL_COLUMNS[dataset_id]
            amount_col = AMOUNT_COLUMNS.get(dataset_id)
            ts_col = TIMESTAMP_COLUMNS.get(dataset_id)
            try:
                idx = {c: i for i, c in enumerate(primary_header)}
                label_i = idx[label_col]
            except KeyError:
                errors.append("label_column_missing")
                label_i = None
            amount_i = idx.get(amount_col) if amount_col else None
            ts_i = idx.get(ts_col) if ts_col else None
            fingerprint_ready = (label_i is not None
                                 and amount_i is not None
                                 and ts_i is not None)
            numeric_idx = [(c, idx[c]) for c in (amount_col, ts_col)
                           if c and c in idx]
            type_bad = {c: 0 for c, _ in numeric_idx}
            known_labels: dict[str, int] = {}

            for row in primary_rows:
                row_count += 1
                if len(row) != len(primary_header):
                    ragged_rows += 1
                    row = (row + [""] * len(primary_header))[
                        :len(primary_header)]
                for col, i in numeric_idx:
                    value = row[i] if i < len(row) else ""
                    if value == "":
                        type_bad[col] += 1
                        continue
                    try:
                        float(value)
                    except ValueError:
                        type_bad[col] += 1
                raw_label = row[label_i] if label_i is not None else ""
                key = str(raw_label)
                known_labels[key] = known_labels.get(key, 0) + 1
                norm = (normalize_label(dataset_id, raw_label)
                        if label_i is not None else None)
                nkey = "unknown" if norm is None else str(norm)
                normalized_counts[nkey] = \
                    normalized_counts.get(nkey, 0) + 1
                content = hashlib.sha256(
                    json.dumps(list(row), separators=(",", ":"),
                               ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                fingerprint: str | None = None
                if fingerprint_ready:
                    metadata = ((amount_col, row[amount_i]),) \
                        if amount_col is not None else ()
                    hrow = HarmonizedRow(
                        source_dataset_id=dataset_id,
                        source_record_id=(
                            f"{dataset_id}:{row_count - 1}"),
                        original_label=key,
                        normalized_label=norm,
                        timestamp=row[ts_i] if ts_i is not None else "",
                        original_schema_metadata=metadata,
                        harmonization_status=(
                            HarmonizationStatus.HARMONIZED.value
                            if norm is not None else
                            HarmonizationStatus
                            .EXCLUDED_UNMAPPED_LABEL.value),
                    )
                    fingerprint = row_fingerprint(hrow)
                acc.observe(dataset_id, content, fingerprint)

            type_mismatches = tuple(
                (c, n) for c, n in sorted(type_bad.items()) if n > 0)
            if type_mismatches:
                errors.append("schema_type_mismatch")
            if ragged_rows:
                errors.append("ragged_rows")
            label_counts = tuple(sorted(known_labels.items()))
            if not fingerprint_ready:
                warnings.append("fingerprint_columns_incomplete")

    normalized_label_counts = tuple(sorted(normalized_counts.items()))

    # documented-count checks (wrong-file detection; never relaxed)
    if data_paths and row_count:
        if ds.transaction_count is None:
            row_count_check = "not_documented"
        elif row_count == ds.transaction_count:
            row_count_check = "match"
        else:
            row_count_check = "mismatch"
            errors.append("row_count_mismatch")
        positives = normalized_counts.get("1", 0)
        if ds.fraud_count is None:
            if ds.fraud_rate is None:
                fraud_count_check = "not_documented"
            else:
                observed = positives / row_count
                if abs(observed - ds.fraud_rate) <= 1e-3:
                    fraud_count_check = "match"
                else:
                    fraud_count_check = "mismatch"
                    errors.append("fraud_count_mismatch")
        elif positives == ds.fraud_count:
            fraud_count_check = "match"
        else:
            fraud_count_check = "mismatch"
            errors.append("fraud_count_mismatch")
    else:
        row_count_check = "not_documented" \
            if ds.transaction_count is None else "not_checked"
        fraud_count_check = "not_documented" \
            if ds.fraud_count is None else "not_checked"

    # leakage scan over the actual schema (Phase 106 authoritative)
    if schema_actual:
        hits = scan_leakage_fields(schema_actual)
        if hits:
            warnings.append("leakage_fields_present")

    if missing_required:
        errors.append("required_files_missing")

    # status classification — never silently skip a supplied dataset
    if missing_required:
        status = AcquisitionStatus.INCOMPLETE_ACQUISITION.value
        if "required_files_missing" not in errors:
            errors.append("required_files_missing")
    elif errors:
        status = AcquisitionStatus.ACQUIRED_VERIFICATION_FAILED.value
    else:
        status = AcquisitionStatus.ACQUIRED_VERIFIED.value

    acc.fold_local(dataset_id)
    findings = acc.findings_for(dataset_id, row_count)

    row_id_kind = ("source_record_id"
                   if "TransactionID" in schema_actual
                   else "local_row_position_hash")

    return IngestionRecord(
        dataset_id=dataset_id,
        status=status,
        located_files=tuple(_display(p) for p in located),
        primary_file=_display(primary_rel),
        alternate_files=tuple(
            _display(p) for p in located
            if p != primary_rel),
        file_records=file_records,
        extracted_files=tuple(extracted_records),
        archive_violations=archive_violations,
        schema_expected=get_dataset(dataset_id).schema_columns,
        schema_actual=schema_actual,
        missing_columns=missing_columns,
        renamed_columns=renamed_columns,
        extra_columns=extra_columns,
        type_mismatch_counts=type_mismatches,
        ragged_rows=ragged_rows,
        row_count=row_count,
        column_count=column_count,
        label_counts=label_counts,
        normalized_label_counts=normalized_label_counts,
        row_count_check=row_count_check,
        fraud_count_check=fraud_count_check,
        verification_errors=tuple(errors),
        verification_warnings=tuple(warnings),
        row_id_kind=row_id_kind,
        timestamp_status=_timestamp_status(dataset_id),
        duplicate_findings=findings,
        record_hash="",
    )


def ingest_many(
    dataset_requests: Mapping[str, Sequence[str] | None],
) -> dict[str, IngestionRecord]:
    """Ingest several registered candidates under ONE shared duplicate
    accumulator, then classify every source against the fully folded
    global view — cross-source duplicate rows are symmetric on both
    sources.  Unknown dataset keys fail closed; candidate-path
    overrides never relax a single verification check."""
    unknown = [d for d in dataset_requests if d not in DATASET_IDS]
    if unknown:
        raise ValueError(f"unregistered dataset ids: {unknown}")
    acc = DuplicateAccumulator()
    records: dict[str, IngestionRecord] = {
        dataset_id: ingest_dataset(
            dataset_id,
            candidate_paths=dataset_requests[dataset_id],
            duplicate_accumulator=acc,
        )
        for dataset_id in DATASET_IDS if dataset_id in dataset_requests
    }
    # every source folded — classification now sees the complete
    # cross-source picture before the records are returned
    return {
        dataset_id: dataclasses.replace(
            rec,
            duplicate_findings=acc.findings_for(
                dataset_id, rec.row_count))
        for dataset_id, rec in records.items()
    }


def ingest_all() -> dict[str, IngestionRecord]:
    """Ingest every registered candidate from the documented locations,
    sharing one duplicate accumulator so cross-source detection works.
    Absent datasets yield NOT_ACQUIRED — never an error."""
    return ingest_many({d: None for d in DATASET_IDS})


def build_ingestion_ledger(
    records: Mapping[str, IngestionRecord],
) -> IngestionLedger:
    """Ordered, append-only ledger over an ingestion result."""
    return IngestionLedger(entries=tuple(
        records[d] for d in DATASET_IDS if d in records))


# ══════════════════════════════════════════════════════════════════════
# CANONICAL FIELD MAPPINGS
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FieldMapping:
    """canonical_field <- source_field with an explicit mapping type.
    No heuristic semantic merging: absent source columns stay
    NOT_AVAILABLE, non-equivalent columns stay SEMANTICALLY_INCOMPATIBLE,
    leakage columns stay EXCLUDED_LEAKAGE."""
    canonical_field: str
    source_field: str
    mapping_type: str                # MappingType value


def build_field_mappings(dataset_id: str) -> tuple[FieldMapping, ...]:
    """Deterministic canonical mapping table for one registered dataset,
    derived ONLY from Phase 106 registry metadata (works for absent
    datasets too — missing fields are recorded, never invented)."""
    ds = get_dataset(dataset_id)  # unknown dataset fails closed
    out: list[FieldMapping] = []
    label_col = LABEL_COLUMNS[dataset_id]
    out.append(FieldMapping("original_label", label_col,
                            MappingType.DIRECT.value))
    out.append(FieldMapping("normalized_label", label_col,
                            MappingType.DETERMINISTIC_TRANSFORM.value))
    ts_col = TIMESTAMP_COLUMNS.get(dataset_id)
    out.append(FieldMapping(
        "timestamp", ts_col or "",
        MappingType.DIRECT.value if ts_col
        else MappingType.NOT_AVAILABLE.value))
    amount_col = AMOUNT_COLUMNS.get(dataset_id)
    out.append(FieldMapping(
        "amount", amount_col or "",
        MappingType.DIRECT.value if amount_col
        else MappingType.NOT_AVAILABLE.value))
    ttype = TRANSACTION_TYPE_COLUMNS.get(dataset_id, ())
    out.append(FieldMapping(
        "transaction_type", ttype[0] if ttype else "",
        MappingType.DIRECT.value if ttype
        else MappingType.NOT_AVAILABLE.value))
    merchant = MERCHANT_CATEGORY_COLUMNS.get(dataset_id, ())
    if merchant:
        # documented category column is not established as cross-source
        # equivalent to the canonical MCC semantics
        out.append(FieldMapping("merchant_category", merchant[0],
                                MappingType.SEMANTICALLY_INCOMPATIBLE
                                .value))
    else:
        out.append(FieldMapping("merchant_category", "",
                                MappingType.NOT_AVAILABLE.value))
    entity = CROSS_SOURCE_ENTITY_COLUMNS.get(dataset_id, ())
    if entity:
        # identifier spaces are source-local; cross-source pooling is
        # forbidden by Phase 106 semantics
        for col in entity:
            out.append(FieldMapping("entity_identifier", col,
                                    MappingType.SEMANTICALLY_INCOMPATIBLE
                                    .value))
    else:
        out.append(FieldMapping("entity_identifier", "",
                                MappingType.NOT_AVAILABLE.value))
    # no registered dataset documents MCC — never manufacture it
    out.append(FieldMapping("mcc", "",
                            MappingType.NOT_AVAILABLE.value))
    for risk in ds.known_leakage_risks:
        out.append(FieldMapping("excluded_evaluation_field", risk,
                                MappingType.EXCLUDED_LEAKAGE.value))
    return tuple(out)


# ══════════════════════════════════════════════════════════════════════
# PARTITIONS AND THE COMBINED BENCHMARK
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BenchmarkPartition:
    """One source's benchmark partition: metadata, schema, counts,
    exclusions, compatibility and a hash — bound to the verified local
    file by its hash.  Logical namespace `benchmark/<dir>`; row data
    stays in the source file (source-preserving, never copied)."""
    partition_id: str                 # "benchmark/<dirname>"
    dataset_id: str
    acquisition_status: str
    primary_file: str
    primary_sha256: str
    dataset_version: str
    schema_columns: tuple[str, ...]
    row_count: int
    column_count: int
    label_counts: tuple[tuple[str, int], ...]
    normalized_label_counts: tuple[tuple[str, int], ...]
    supervised_rows: int
    label_semantics: str
    excluded_fields: tuple[str, ...]
    canonical_mappings: tuple[FieldMapping, ...]
    duplicate_findings: DuplicateFindings
    group: str
    evaluation_readiness: str
    row_id_kind: str
    timestamp_status: str
    classification: str
    license_documented: bool
    provenance_confidence: str
    partition_hash: str

    def __post_init__(self) -> None:
        ds = get_dataset(self.dataset_id)
        if self.dataset_id not in DATASET_IDS:
            raise ValueError("partition dataset must be registered")
        group, _ = assign_benchmark_group(self.dataset_id)
        if group == BenchmarkGroup.GROUP_D.value:
            raise ValueError(
                "group_d datasets never receive a benchmark partition")
        if not ds.license_documented:
            raise ValueError(
                "undocumented license never receives a partition")
        if self.license_documented != ds.license_documented:
            raise ValueError(
                "license flag must match the registry truth")
        if self.group != group:
            raise ValueError("partition group drift")
        label_col = LABEL_COLUMNS[self.dataset_id]
        if label_col not in self.excluded_fields:
            raise ValueError("label column must be evaluation-excluded")
        leaked = set(scan_leakage_fields(self.schema_columns))
        if not leaked.issubset(set(self.excluded_fields)):
            raise ValueError("leakage fields must be excluded")
        if self.partition_id != (
                f"benchmark/{DATASET_DIRNAMES[self.dataset_id]}"):
            raise ValueError("partition id must name its dataset")
        object.__setattr__(self, "partition_hash",
                           self._compute_hash())

    def _compute_hash(self) -> str:
        return _stable(_strip_keys(
            dataclasses.asdict(self), frozenset({"partition_hash"})))

    def verify_hash(self) -> bool:
        return self.partition_hash == self._compute_hash()


def build_partitions(
    records: Mapping[str, IngestionRecord],
) -> tuple[BenchmarkPartition, ...]:
    """Admission: source-level verification SUCCESS plus established
    license/provenance (Phase 106 groups).  Verified-but-unlicensed
    (Group D) files never receive a partition."""
    partitions: list[BenchmarkPartition] = []
    for dataset_id in DATASET_IDS:
        rec = records.get(dataset_id)
        if rec is None or not rec.verified:
            continue
        ds = get_dataset(dataset_id)
        group, _rationale = assign_benchmark_group(dataset_id)
        readiness = GROUP_READINESS[group]
        if group == BenchmarkGroup.GROUP_D.value:
            continue
        if not ds.license_documented:
            continue
        label_col = LABEL_COLUMNS[dataset_id]
        leakage = set(scan_leakage_fields(rec.schema_actual))
        excluded = tuple(sorted(leakage | {label_col}
                                | set(ds.known_leakage_risks)))
        supervised = 0
        for value, count in rec.normalized_label_counts:
            if value in ("0", "1"):
                supervised += count
        primary = rec.primary_file or ""
        primary_sha = next(
            (f.sha256 for f in rec.file_records
             if f.path == _display(primary)), "")
        partitions.append(BenchmarkPartition(
            partition_id=f"benchmark/{DATASET_DIRNAMES[dataset_id]}",
            dataset_id=dataset_id,
            acquisition_status=rec.status,
            primary_file=primary,
            primary_sha256=primary_sha,
            dataset_version=rec.version,
            schema_columns=rec.schema_actual,
            row_count=rec.row_count,
            column_count=rec.column_count,
            label_counts=rec.label_counts,
            normalized_label_counts=rec.normalized_label_counts,
            supervised_rows=supervised,
            label_semantics=ds.label_semantics,
            excluded_fields=excluded,
            canonical_mappings=build_field_mappings(dataset_id),
            duplicate_findings=rec.duplicate_findings or _ZERO_DUPLICATES,
            group=group,
            evaluation_readiness=readiness,
            row_id_kind=rec.row_id_kind,
            timestamp_status=rec.timestamp_status,
            classification=ds.classification,
            license_documented=ds.license_documented,
            provenance_confidence=ds.provenance_confidence,
            partition_hash="",
        ))
    return tuple(partitions)


@dataclass(frozen=True)
class CombinedBenchmark:
    """PUBLIC_MULTI_SOURCE_FRAUD_BENCHMARK — the source-preserving
    combined view.  Every emitted row carries source_dataset_id and a
    deterministic source_row_id; source identity is never pooled away."""
    benchmark_name: str
    descriptor: str
    partition_ids: tuple[str, ...]
    source_files: tuple[tuple[str, str, str], ...]
    source_dataset_ids: tuple[str, ...]
    total_rows: int
    supervised_rows: int
    harmonized_fields: tuple[str, ...]
    source_local_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.benchmark_name != BENCHMARK_NAME:
            raise ValueError("unexpected benchmark name")
        claims = find_single_source_claims(
            f"{self.benchmark_name} {self.descriptor}")
        if claims:
            raise ValueError(
                f"single-institution claims in benchmark: {claims}")
        if not self.partition_ids:
            raise ValueError(
                "combined benchmark requires at least one admitted "
                "partition")
        if len(set(self.source_dataset_ids)) != len(
                self.source_dataset_ids):
            raise ValueError("combined benchmark sources must be unique")
        if not self.source_files:
            raise ValueError("combined benchmark requires source files")
        if (tuple(f[0] for f in self.source_files)
                != self.partition_ids
                or tuple(f[1] for f in self.source_files)
                != self.source_dataset_ids):
            raise ValueError("source files must align with partitions")

    def iter_rows(self) -> Iterator[HarmonizedRow]:
        """Stream every combined row through the authoritative Phase
        106 harmonize_rows(): source_dataset_id stamped on the row,
        original label verbatim, deterministic local source_row_id
        (position + content hash — never an original identity).  Rows
        are produced on demand and never materialized or copied."""
        for _partition_id, dataset_id, primary in self.source_files:
            path = _resolve(primary)
            container = detect_container(path)
            if container == ContainerType.CSV:
                header, rows = _iter_csv(path)
            elif container == ContainerType.JSON:
                header, rows = _iter_json(path)
            else:
                continue
            label_col = LABEL_COLUMNS[dataset_id]
            if label_col not in header:
                continue
            for idx, row in enumerate(rows):
                content = hashlib.sha256(
                    json.dumps(list(row), separators=(",", ":"),
                               ensure_ascii=False).encode("utf-8")
                ).hexdigest()[:8]
                rowdict = {col: (row[i] if i < len(row) else "")
                           for i, col in enumerate(header)}
                rowdict["record_id"] = f"{dataset_id}:{idx}:{content}"
                yield harmonize_rows(dataset_id, [rowdict])[0]


def build_combined_benchmark(
    partitions: Sequence[BenchmarkPartition],
) -> CombinedBenchmark:
    """Combine only admitted (verified + licensed) partitions.  Group
    B/D datasets are combined as themselves — never upgraded."""
    if not partitions:
        raise ValueError("no admitted partitions to combine")
    source_local = tuple(sorted({
        m.source_field for p in partitions
        for m in p.canonical_mappings
        if m.mapping_type in (MappingType.DIRECT.value,
                              MappingType.DETERMINISTIC_TRANSFORM.value)
        and m.canonical_field not in ("original_label",
                                       "normalized_label")
        and m.source_field}))
    return CombinedBenchmark(
        benchmark_name=BENCHMARK_NAME,
        descriptor=BENCHMARK_DESCRIPTOR,
        partition_ids=tuple(p.partition_id for p in partitions),
        source_files=tuple(
            (p.partition_id, p.dataset_id, p.primary_file)
            for p in partitions),
        source_dataset_ids=tuple(p.dataset_id for p in partitions),
        total_rows=sum(p.row_count for p in partitions),
        supervised_rows=sum(p.supervised_rows for p in partitions),
        harmonized_fields=COMMON_BENCHMARK_FIELDS,
        source_local_fields=source_local,
    )


# ══════════════════════════════════════════════════════════════════════
# COMPONENT-EVALUATION VIEWS (prepared for Phase 108)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ComponentEvaluationView:
    """A declared, clearly-scoped component-evaluation view.  Never
    Altman-Native model performance — always component-only."""
    view_id: str
    status: str                       # ComponentViewStatus value
    source_partitions: tuple[str, ...]
    included_fields: tuple[str, ...]
    excluded_fields: tuple[str, ...]
    label_semantics: str
    intended_metrics: tuple[str, ...]
    limitations: tuple[str, ...]
    scope_statement: str

    def __post_init__(self) -> None:
        if self.scope_statement != _COMPONENT_SCOPE:
            raise ValueError("component views carry one fixed scope")
        overlap = set(self.included_fields) & set(self.excluded_fields)
        if overlap:
            raise ValueError(
                f"view includes excluded fields: {sorted(overlap)}")
        for col in LABEL_COLUMNS.values():
            if col in self.included_fields:
                raise ValueError(
                    "label columns are never view features")


_COMPONENT_SCOPE = ("component_evaluation_only_never_altman_native_"
                    "performance")


def build_component_evaluation_views(
    partitions: Sequence[BenchmarkPartition],
) -> tuple[ComponentEvaluationView, ...]:
    """Prepare component-evaluation views from admitted partitions.
    Each view declares sources, fields, metrics and limitations; views
    with no eligible source are declared but marked ineligible."""
    excluded = tuple(sorted({
        f for p in partitions for f in p.excluded_fields}))
    by_id = {p.dataset_id: p for p in partitions}

    def _with_amount() -> tuple[str, ...]:
        return tuple(
            p.partition_id for p in partitions
            if AMOUNT_COLUMNS.get(p.dataset_id)
            and AMOUNT_COLUMNS[p.dataset_id] in p.schema_columns)

    def _temporal() -> tuple[str, ...]:
        return tuple(
            p.partition_id for p in partitions
            if p.timestamp_status
            == TimestampStatus.ORDERING_ONLY_RELATIVE.value)

    def _typed() -> tuple[str, ...]:
        out = []
        for dataset_id, cols in TRANSACTION_TYPE_COLUMNS.items():
            p = by_id.get(dataset_id)
            if p and all(c in p.schema_columns for c in cols):
                out.append(p.partition_id)
        return tuple(out)

    views: list[ComponentEvaluationView] = [
        ComponentEvaluationView(
            view_id="amount_based",
            status=(ComponentViewStatus.ELIGIBLE.value
                    if _with_amount()
                    else ComponentViewStatus.NO_ELIGIBLE_SOURCE.value),
            source_partitions=_with_amount(),
            included_fields=("amount", "normalized_label"),
            excluded_fields=excluded,
            label_semantics="per-source documented label semantics; "
                            "unmapped labels excluded",
            intended_metrics=("precision", "recall", "f1"),
            limitations=(
                "currency units are not documented cross-source; "
                "amount pooling across sources is forbidden",
                "scores are component statistics, never the "
                "production model's output",
            ),
            scope_statement=_COMPONENT_SCOPE,
        ),
        ComponentEvaluationView(
            view_id="temporal_ordering",
            status=(ComponentViewStatus.ELIGIBLE.value
                    if _temporal()
                    else ComponentViewStatus.NO_ELIGIBLE_SOURCE.value),
            source_partitions=_temporal(),
            included_fields=("timestamp", "normalized_label"),
            excluded_fields=excluded,
            label_semantics="per-source documented label semantics",
            intended_metrics=("precision", "recall",
                              "temporal_ordering_consistency"),
            limitations=(
                "timestamps are relative/ordering-only; absolute "
                "calendar dates are never derived from them",
                "no causal or wall-clock claims",
            ),
            scope_statement=_COMPONENT_SCOPE,
        ),
        ComponentEvaluationView(
            view_id="class_imbalance",
            status=(ComponentViewStatus.ELIGIBLE.value
                    if partitions
                    else ComponentViewStatus.NO_ELIGIBLE_SOURCE.value),
            source_partitions=tuple(p.partition_id for p in partitions),
            included_fields=("normalized_label",),
            excluded_fields=excluded,
            label_semantics="per-source documented label semantics",
            intended_metrics=("pr_auc", "mcc", "f1"),
            limitations=(
                "prevalence differs across sources; pooled prevalence "
                "is not an institutional fraud statistic",
            ),
            scope_statement=_COMPONENT_SCOPE,
        ),
        ComponentEvaluationView(
            view_id="transaction_type",
            status=(ComponentViewStatus.ELIGIBLE.value
                    if _typed()
                    else ComponentViewStatus.NO_ELIGIBLE_SOURCE.value),
            source_partitions=_typed(),
            included_fields=("transaction_type", "normalized_label"),
            excluded_fields=excluded,
            label_semantics="per-source documented label semantics",
            intended_metrics=("recall_by_transaction_type",),
            limitations=(
                "only sources documenting a transaction-type column "
                "participate",
            ),
            scope_statement=_COMPONENT_SCOPE,
        ),
        ComponentEvaluationView(
            view_id="input_validation",
            status=(ComponentViewStatus.ELIGIBLE.value
                    if partitions
                    else ComponentViewStatus.NO_ELIGIBLE_SOURCE.value),
            source_partitions=tuple(p.partition_id for p in partitions),
            included_fields=("schema_columns", "label_counts"),
            excluded_fields=excluded,
            label_semantics="raw label distributions only",
            intended_metrics=("schema_conformance_rate",
                              "label_conformance_rate"),
            limitations=(
                "conformance of locally ingested files against the "
                "Phase 106 registry; not a model metric",
            ),
            scope_statement=_COMPONENT_SCOPE,
        ),
        ComponentEvaluationView(
            view_id="drift_monitoring",
            status=(ComponentViewStatus.VALIDITY_NOT_ESTABLISHED.value),
            source_partitions=(),
            included_fields=(),
            excluded_fields=excluded,
            label_semantics="n/a",
            intended_metrics=("population_stability_index",),
            limitations=(
                "population equivalence with the production training "
                "distribution is not established for public datasets; "
                "no drift claim is valid yet",
            ),
            scope_statement=_COMPONENT_SCOPE,
        ),
    ]
    return tuple(views)


# ══════════════════════════════════════════════════════════════════════
# BENCHMARK MANIFEST
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Phase107BenchmarkManifest:
    """Deterministic, tamper-evident manifest for the public benchmark.

    Reuses BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE from Phase 106: this
    manifest is benchmark bookkeeping, never qualification evidence,
    and no existing gate treats it as authorization.  Identity fields
    (model, release, feature contract, threshold, canonical mapping)
    are construction-locked: a substituted identity fails immediately.
    """
    benchmark_id: str
    manifest_version: str
    phase: int
    created_at: str
    datasets_present: tuple[str, ...]
    datasets_absent: tuple[str, ...]
    datasets_verified: tuple[str, ...]
    dataset_file_hashes: tuple[tuple[str, str, str], ...]
    extracted_file_hashes: tuple[tuple[str, str, str], ...]
    schemas: tuple[tuple[str, tuple[str, ...]], ...]
    row_counts: tuple[tuple[str, int], ...]
    column_counts: tuple[tuple[str, int], ...]
    label_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    source_classifications: tuple[tuple[str, str], ...]
    license_provenance: tuple[tuple[str, bool, str], ...]
    excluded_fields: tuple[tuple[str, tuple[str, ...]], ...]
    canonical_field_mappings: tuple[
        tuple[str, tuple[tuple[str, str, str], ...]], ...]
    duplicate_findings: tuple[tuple[str, int, int, int, int], ...]
    native_compatibility: tuple[tuple[str, str, int, int], ...]
    evaluation_config: BenchmarkEvaluationConfig
    production_model_id: str
    production_release_id: str
    feature_version: str
    native_feature_version: str
    production_threshold: float
    canonical_mapping: str
    not_gate_evidence: str
    manifest_hash: str

    def __post_init__(self) -> None:
        if self.phase != PHASE:
            raise ValueError("manifest phase is fixed")
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError("unexpected manifest version")
        if self.production_model_id != MODEL_ID:
            raise ValueError("production model identity is locked")
        if self.production_release_id != RELEASE_ID:
            raise ValueError("production release identity is locked")
        if self.feature_version != FEATURE_VERSION:
            raise ValueError("domain/runtime feature version is locked")
        if self.native_feature_version != NATIVE_FEATURE_VERSION:
            raise ValueError("native feature version is locked")
        if self.production_threshold != EXPECTED_PRODUCTION_THRESHOLD:
            raise ValueError("production threshold is locked")
        if self.canonical_mapping != CANONICAL_TRANSFORMATION:
            raise ValueError("canonical 21->48 mapping is locked")
        if self.not_gate_evidence != BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE:
            raise ValueError("manifest must carry the not-gate-evidence "
                             "declaration")
        unknown = (set(self.datasets_present)
                   | set(self.datasets_absent)
                   | set(self.datasets_verified))
        if not unknown.issubset(set(DATASET_IDS)):
            raise ValueError("manifest datasets must be registered")
        if set(self.datasets_present) & set(self.datasets_absent):
            raise ValueError("present/absent must be disjoint")
        if not set(self.datasets_verified).issubset(
                set(self.datasets_present)):
            raise ValueError("verified datasets must be present")
        object.__setattr__(self, "manifest_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return _stable(_strip_keys(
            dataclasses.asdict(self), frozenset({"manifest_hash"})))

    def verify_hash(self) -> bool:
        return self.manifest_hash == self._compute_hash()


def build_phase107_manifest(
    records: Mapping[str, IngestionRecord],
    partitions: Sequence[BenchmarkPartition],
) -> Phase107BenchmarkManifest:
    """Deterministic manifest over an ingestion result.  Read-only with
    respect to every production identity; carries no row data."""
    present = tuple(d for d in DATASET_IDS
                    if d in records
                    and records[d].status
                    != AcquisitionStatus.NOT_ACQUIRED.value)
    absent = tuple(d for d in DATASET_IDS
                   if d not in records
                   or records[d].status
                   == AcquisitionStatus.NOT_ACQUIRED.value)
    verified = tuple(d for d in DATASET_IDS
                     if d in records and records[d].verified)
    file_hashes: list[tuple[str, str, str]] = []
    extracted_hashes: list[tuple[str, str, str]] = []
    schemas: list[tuple[str, tuple[str, ...]]] = []
    row_counts: list[tuple[str, int]] = []
    column_counts: list[tuple[str, int]] = []
    label_counts: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    excluded: list[tuple[str, tuple[str, ...]]] = []
    for dataset_id in DATASET_IDS:
        rec = records.get(dataset_id)
        if rec is None:
            continue
        for f in rec.file_records:
            file_hashes.append((dataset_id, f.path, f.sha256))
        for f in rec.extracted_files:
            extracted_hashes.append((dataset_id, f.path, f.sha256))
        if rec.schema_actual:
            schemas.append((dataset_id, rec.schema_actual))
        row_counts.append((dataset_id, rec.row_count))
        column_counts.append((dataset_id, rec.column_count))
        label_counts.append((dataset_id, rec.label_counts))
        excluded.append((dataset_id, tuple(sorted(
            set(scan_leakage_fields(rec.schema_actual))
            | ({LABEL_COLUMNS[dataset_id]}
               if rec.schema_actual else set())
            | set(get_dataset(dataset_id).known_leakage_risks)))))
    duplicate_rows = tuple(
        (p.dataset_id,
         p.duplicate_findings.within_source_duplicate_rows,
         p.duplicate_findings.cross_source_duplicate_rows,
         p.duplicate_findings.normalized_collision_rows,
         p.duplicate_findings.unique_rows)
        for p in partitions)
    native_compat = []
    for dataset_id in DATASET_IDS:
        group, _ = assign_benchmark_group(dataset_id)
        usable = len(usable_features(dataset_preflight(dataset_id)))
        # Group A is empty by Phase 106 authority — ingestion never
        # re-promotes a group, so native-eligible rows stay zero.
        native_compat.append((dataset_id, group, usable, 0))
    mappings = tuple(
        (d, tuple((m.canonical_field, m.source_field, m.mapping_type)
                  for m in build_field_mappings(d)))
        for d in DATASET_IDS)
    classifications = tuple(
        (d, get_dataset(d).classification) for d in DATASET_IDS)
    license_prov = tuple(
        (d, get_dataset(d).license_documented,
         get_dataset(d).provenance_confidence)
        for d in DATASET_IDS)
    return Phase107BenchmarkManifest(
        benchmark_id=f"phase107-{BENCHMARK_NAME.lower()}",
        manifest_version=MANIFEST_VERSION,
        phase=PHASE,
        created_at=MANIFEST_CREATED_AT,
        datasets_present=present,
        datasets_absent=absent,
        datasets_verified=verified,
        dataset_file_hashes=tuple(file_hashes),
        extracted_file_hashes=tuple(extracted_hashes),
        schemas=tuple(schemas),
        row_counts=tuple(row_counts),
        column_counts=tuple(column_counts),
        label_counts=tuple(label_counts),
        source_classifications=classifications,
        license_provenance=license_prov,
        excluded_fields=tuple(excluded),
        canonical_field_mappings=mappings,
        duplicate_findings=duplicate_rows,
        native_compatibility=tuple(native_compat),
        evaluation_config=build_evaluation_config(),
        production_model_id=MODEL_ID,
        production_release_id=RELEASE_ID,
        feature_version=FEATURE_VERSION,
        native_feature_version=NATIVE_FEATURE_VERSION,
        production_threshold=EXPECTED_PRODUCTION_THRESHOLD,
        canonical_mapping=CANONICAL_TRANSFORMATION,
        not_gate_evidence=BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
        manifest_hash="",
    )


def verify_manifest_hash(manifest: Phase107BenchmarkManifest) -> bool:
    return manifest.verify_hash()


__all__ = [
    "ACQUISITION_ROOT", "BENCHMARK_NAME", "CANONICAL_CANDIDATE_FIELDS",
    "CANDIDATE_PATHS", "DOMAIN_FEATURE_COUNT", "DATASET_DIRNAMES",
    "EXPECTED_PRODUCTION_THRESHOLD", "FORBIDDEN_BYPASS_PARAMETERS",
    "INGESTION_VERSION", "MANIFEST_CREATED_AT", "MANIFEST_VERSION",
    "MULTIFILE_REQUIRED", "NATIVE_FEATURE_COUNT", "PHASE",
    "PRIMARY_BASENAMES", "REPO_ROOT", "TRANSACTION_TYPE_COLUMNS",
    "AcquisitionStatus", "ArchiveInspection", "BenchmarkPartition",
    "BenchmarkReadiness", "CombinedBenchmark",
    "ComponentEvaluationView", "ComponentViewStatus",
    "ContainerType", "DuplicateAccumulator", "DuplicateFindings",
    "FieldMapping", "FileRecord", "IngestionLedger", "IngestionRecord",
    "MappingType", "Phase107BenchmarkManifest", "TimestampStatus",
    "acquisition_instructions", "build_combined_benchmark",
    "build_component_evaluation_views", "build_field_mappings",
    "build_ingestion_ledger", "build_partitions",
    "build_phase107_manifest",
    "detect_container", "detect_dataset_replacement", "ingest_all",
    "ingest_dataset", "ingest_many", "inspect_archive",
    "resolve_candidate_paths",
    "safe_extract_archive", "sha256_file", "verify_manifest_hash",
]
