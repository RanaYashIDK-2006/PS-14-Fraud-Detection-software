"""Phase 39: Immutable evaluation records — traceable ML provenance.

Every evaluation run produces an EvaluationRecord that binds a metric set to
the exact model files, dataset bytes, configuration, code state, and seed that
produced it. A reviewer must be able to determine from the record alone which
model and which data generated the reported numbers, without relying on
filenames (which can be reused or silently overwritten).

Hashing conventions (documented in docs/metric_definitions.md and mirrored
here in code — the code is authoritative):

* Model/artifact hash: SHA-256 over the raw bytes of each artifact file,
  reported per file. The overall ``model_hash`` is the SHA-256 of the
  lexicographically sorted "<name>:<file-sha256>" lines joined with "\\n".
  Any byte change in any artifact changes the evaluation identity.
* Dataset fingerprint: SHA-256 over the file's raw bytes (``sha256:<hex>``).
  Deterministic, independent of compression metadata; two files with the
  same bytes always produce the same fingerprint.
* Git commit: ``git rev-parse HEAD`` at run time, or ``None`` outside a
  repository. Not inferred from filenames or previous records.
* Records are APPEND-ONLY: ``EvaluationLedger`` never mutates an existing
  record; a re-run appends a new record with a new evaluation_id. Failed or
  unfavorable evaluations are preserved exactly like favorable ones.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALUATION_RECORD_VERSION = "1.0"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 over a file's raw bytes, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(paths: list[Path]) -> dict[str, Any]:
    """Per-file SHA-256 for model artifacts + an aggregate model_hash.

    The aggregate is the SHA-256 of the sorted "<name>:<sha256>" lines so the
    identity is independent of dictionary/file-listing order.
    """
    files: dict[str, str] = {}
    for p in paths:
        if p.exists():
            files[p.name] = sha256_file(p)
    digest_src = "\n".join(f"{k}:{v}" for k, v in sorted(files.items()))
    return {
        "files": files,
        "model_hash": hashlib.sha256(digest_src.encode("utf-8")).hexdigest()
        if files else None,
    }


def dataset_fingerprint(path: Path) -> dict[str, Any]:
    """Deterministic dataset identity: SHA-256 over raw bytes.

    What is hashed: exactly the file bytes on disk — no header reinterpretation,
    no dtype normalization, no row reordering. Any change to the evaluation
    data (even a single character) changes the fingerprint.
    """
    if not path.exists():
        return {"path": str(path), "sha256": None, "size_bytes": None, "exists": False}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "exists": True,
    }


def git_commit() -> str | None:
    """HEAD commit SHA at run time, or None outside a git repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def software_versions() -> dict[str, str]:
    """Environment versions 'where practical' — best-effort, never fabricated."""
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "scipy"):
        try:
            versions[mod] = __import__(mod).__version__  # type: ignore[union-attr]
        except Exception:
            versions[mod] = "not-installed"
    return versions


@dataclass
class EvaluationRecord:
    """Immutable provenance binding for one evaluation run.

    metric_definitions_version pins the authoritative metric semantics (see
    docs/metric_definitions.md): a record is only comparable to other records
    produced under the same metric semantics.
    """
    evaluation_id: str
    timestamp_utc: str
    model_identifier: str
    model_hash: str | None
    artifact_file_hashes: dict[str, str]
    preprocessing_version: str
    feature_schema_version: str
    dataset: dict[str, Any]            # dataset_fingerprint() output
    training_dataset: dict[str, Any] | None
    seed: int | None
    threshold: float | None
    threshold_source: str              # "validation" | "fixed" | "none"
    evaluation_config: dict[str, Any]
    git_commit: str | None
    software: dict[str, str]
    metric_definitions_version: str
    metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    status: str = "COMPLETED"          # COMPLETED | FAILED — failures are preserved
    record_version: str = EVALUATION_RECORD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def new_evaluation_id(timestamp_utc: str, model_hash: str | None, dataset_sha: str | None,
                      content_digest: str | None = None) -> str:
    """Deterministic, collision-resistant id.

    Binds the UTC second, model identity, dataset identity, and (when given)
    a digest of the record's own content (metrics/config/status) so two runs
    in the same second on identical artifacts+data with DIFFERENT outcomes
    still get distinct ids.
    """
    h = hashlib.sha256(
        f"{timestamp_utc}|{model_hash}|{dataset_sha}|{content_digest or ''}".encode("utf-8")
    ).hexdigest()[:12]
    stamp = timestamp_utc.replace("-", "").replace(":", "").replace("+00:00", "Z")
    return f"eval-{stamp}-{h}"


def create_evaluation_record(
    *,
    model_identifier: str,
    artifact_paths: list[Path],
    dataset_path: Path,
    training_dataset_path: Path | None = None,
    preprocessing_version: str = "1.0",
    feature_schema_version: str = "1.0",
    seed: int | None = None,
    threshold: float | None = None,
    threshold_source: str = "none",
    evaluation_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    status: str = "COMPLETED",
) -> EvaluationRecord:
    """Build a complete provenance record for an evaluation run."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    arts = artifact_hashes(artifact_paths)
    ds = dataset_fingerprint(dataset_path)
    tr = dataset_fingerprint(training_dataset_path) if training_dataset_path else None
    cfg = dict(evaluation_config or {})
    cfg.setdefault("metric_definitions_version", "1.0")
    content_digest = hashlib.sha256(json.dumps(
        {"metrics": metrics or {}, "config": cfg, "status": status,
         "warnings": warnings or [], "threshold": threshold,
         "threshold_source": threshold_source, "seed": seed},
        sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return EvaluationRecord(
        evaluation_id=new_evaluation_id(ts, arts["model_hash"], ds.get("sha256"), content_digest),
        timestamp_utc=ts,
        model_identifier=model_identifier,
        model_hash=arts["model_hash"],
        artifact_file_hashes=arts["files"],
        preprocessing_version=preprocessing_version,
        feature_schema_version=feature_schema_version,
        dataset=ds,
        training_dataset=tr,
        seed=seed,
        threshold=threshold,
        threshold_source=threshold_source,
        evaluation_config=cfg,
        git_commit=git_commit(),
        software=software_versions(),
        metric_definitions_version=cfg["metric_definitions_version"],
        metrics=dict(metrics or {}),
        warnings=list(warnings or []),
        status=status,
    )


class EvaluationLedger:
    """Append-only store of evaluation records (JSONL).

    Never rewrites or removes existing lines. A re-run appends a new record
    with a fresh evaluation_id — the previous record (including failed
    evaluations and their provenance) remains on disk for audit.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: EvaluationRecord) -> str:
        line = json.dumps(record.to_dict(), indent=None, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return record.evaluation_id

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def latest_by_model(self, model_identifier: str) -> dict[str, Any] | None:
        matching = [r for r in self.records() if r.get("model_identifier") == model_identifier]
        return matching[-1] if matching else None


def record_from_run(  # noqa: ANN201 - dict for JSON reports
    *, model_identifier: str, artifacts_dir: Path,
    dataset_path: Path, train_data_path: Path | None = None,
    seed: int | None = None, threshold: float | None = None,
    threshold_source: str = "validation",
    metrics: dict[str, Any] | None = None,
    extra_config: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    status: str = "COMPLETED",
) -> EvaluationRecord:
    """Convenience constructor for train_compare-style runs.

    Hashes every *.joblib in artifacts_dir as the model identity.
    """
    return create_evaluation_record(
        model_identifier=model_identifier,
        artifact_paths=sorted(artifacts_dir.glob("*.joblib")),
        dataset_path=dataset_path,
        training_dataset_path=train_data_path,
        seed=seed,
        threshold=threshold,
        threshold_source=threshold_source,
        evaluation_config={"artifacts_dir": str(artifacts_dir), **(extra_config or {})},
        metrics=metrics,
        warnings=warnings,
        status=status,
    )
