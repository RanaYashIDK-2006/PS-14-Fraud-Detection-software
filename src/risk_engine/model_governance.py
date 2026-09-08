"""Model versioning & governance (check #17).

Every model gets a permanent, immutable record (models/model_records/<id>.json)
capturing dataset hash, code/feature-schema version, feature list, algorithm,
hyperparameters, thresholds and metrics, plus a SHA-256 hash of every artifact
file. `verify()` compares what the app loads against the recorded hashes and
feature schema and fails safely on any mismatch.

Promotion requires passing the full gate (leakage, data quality, untouched
test, robustness, parity, security) — a higher AUC alone never promotes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

RECORDS_DIR_NAME = "model_records"


class ModelIntegrityError(RuntimeError):
    """Raised when a loaded model fails integrity/schema verification."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelRecord:
    model_id: str                       # e.g. altman_lean_15feat_20260830_200346
    created_at: str
    training_dataset_path: str
    training_dataset_sha256: str = ""
    training_dataset_rows: int | None = None
    code_version: str = ""              # feature-schema / code version string
    feature_schema_version: str = ""
    features: list[str] = field(default_factory=list)
    algorithm: str = ""
    hyperparameters: dict = field(default_factory=dict)
    training_seed: int | None = None
    training_period: str = ""
    validation_period: str = ""
    final_test_period: str = ""
    selected_threshold: float | None = None
    calibration_version: str = ""
    performance_metrics: dict = field(default_factory=dict)
    artifact_files: dict = field(default_factory=dict)   # name -> {sha256, size}
    promotion: dict = field(default_factory=dict)        # gate results + reason
    parent_model_id: str = ""            # what this model replaces / was trained from

    def save(self, records_dir: Path) -> Path:
        records_dir.mkdir(parents=True, exist_ok=True)
        p = records_dir / f"{self.model_id}.json"
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, records_dir: Path, model_id: str) -> "ModelRecord":
        p = records_dir / f"{model_id}.json"
        return cls(**json.loads(p.read_text(encoding="utf-8")))

    @classmethod
    def load_file(cls, path: Path) -> "ModelRecord":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def make_record(
    model_id: str,
    artifact_dir: Path,
    *,
    training_dataset_path: str = "",
    feature_schema_version: str = "",
    features: list[str] | None = None,
    algorithm: str = "",
    hyperparameters: dict | None = None,
    training_seed: int | None = None,
    selected_threshold: float | None = None,
    performance_metrics: dict | None = None,
    code_version: str = "unknown",
    dataset_sha256: str = "",
    dataset_rows: int | None = None,
    training_period: str = "",
    validation_period: str = "",
    final_test_period: str = "",
    calibration_version: str = "",
    parent_model_id: str = "",
) -> ModelRecord:
    """Hash every artifact file in artifact_dir into an immutable record."""
    files = {}
    for p in sorted(artifact_dir.iterdir()):
        if p.is_file() and not p.name.endswith((".bak", ".tmp")):
            files[p.name] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    rec = ModelRecord(
        model_id=model_id,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        training_dataset_path=training_dataset_path,
        training_dataset_sha256=dataset_sha256,
        training_dataset_rows=dataset_rows,
        code_version=code_version,
        feature_schema_version=feature_schema_version,
        features=list(features or []),
        algorithm=algorithm,
        hyperparameters=dict(hyperparameters or {}),
        training_seed=training_seed,
        training_period=training_period,
        validation_period=validation_period,
        final_test_period=final_test_period,
        selected_threshold=selected_threshold,
        calibration_version=calibration_version,
        performance_metrics=dict(performance_metrics or {}),
        artifact_files=files,
        parent_model_id=parent_model_id,
    )
    return rec


def verify_artifacts(record: ModelRecord, artifact_dir: Path) -> list[dict]:
    """Compare on-disk artifact files against the record's hashes.

    Returns a list of per-file checks {name, pass, detail}. Any missing file,
    size change or hash mismatch is a FAIL.
    """
    checks = []
    for name, rec in record.artifact_files.items():
        p = artifact_dir / name
        if not p.exists():
            checks.append({"name": name, "pass": False, "detail": "MISSING on disk"})
            continue
        if p.stat().st_size != rec["size"]:
            checks.append({"name": name, "pass": False,
                           "detail": f"size {p.stat().st_size} != recorded {rec['size']}"})
            continue
        h = sha256_file(p)
        checks.append({"name": name, "pass": h == rec["sha256"],
                       "detail": "hash match" if h == rec["sha256"] else f"hash {h[:16]} != recorded {rec['sha256'][:16]}"})
    # Files on disk not in the record are informational, not failures
    return checks


def verify_feature_schema(record: ModelRecord, loaded_features: list[str]) -> list[dict]:
    """Compare loaded feature order/names against the recorded schema."""
    return [{
        "pass": list(loaded_features) == list(record.features),
        "recorded": list(record.features),
        "loaded": list(loaded_features),
        "detail": "match" if list(loaded_features) == list(record.features)
                  else f"schema mismatch: recorded {len(record.features)} feats, loaded {len(loaded_features)}",
    }]


def verify(record: ModelRecord, artifact_dir: Path, loaded_features: list[str] | None = None) -> dict:
    """Full integrity + schema verification. Raises ModelIntegrityError on FAIL."""
    checks = verify_artifacts(record, artifact_dir)
    if loaded_features is not None:
        checks += verify_feature_schema(record, loaded_features)
    failures = [c for c in checks if not c["pass"]]
    result = {
        "model_id": record.model_id,
        "pass": len(failures) == 0,
        "n_checks": len(checks),
        "n_failures": len(failures),
        "checks": checks,
    }
    if not result["pass"]:
        first = failures[0]
        first_name = first.get("name") or "feature-schema"
        raise ModelIntegrityError(
            f"Model {record.model_id} integrity FAIL: {len(failures)}/{len(checks)} checks failed. "
            f"First: {first_name} {first.get('detail', '')}"
        )
    return result


def promote_gate(candidate: ModelRecord) -> dict:
    """Evaluate the documented promotion gate (gates that have records).

    The gate list is the required sequence from the audit; fields present in the
    record's promotion block are evaluated, absent gates are reported as
    UNVERIFIED so a promotion is never silently approved on AUC alone.
    """
    required = ["leakage_checks", "data_quality", "validation_performance",
                "untouched_test", "robustness", "production_parity", "security", "approval"]
    gate = candidate.promotion.get("gates", {})
    rows = []
    for g in required:
        if g in gate:
            rows.append({"gate": g, "status": str(gate[g])})
        else:
            rows.append({"gate": g, "status": "UNVERIFIED"})
    fails = [r for r in rows if str(r["status"]).upper() not in ("PASS", "TRUE", "APPROVED")]
    return {
        "model_id": candidate.model_id,
        "gate": rows,
        "approved": len(fails) == 0,
        "reason": candidate.promotion.get("reason", ""),
        "higher_auc_alone_is_insufficient": True,
    }


def rollback(records_dir: Path, from_dir: Path, to_dir: Path,
             bad_model_id: str, good_model_id: str) -> dict:
    """Rollback helper: copy the recorded-good artifacts back into place.

    to_dir is the active model directory; the good model's artifact files are
    re-copied from from_dir (which must verify against the good record first).
    """
    good = ModelRecord.load(records_dir, good_model_id)
    v = verify_artifacts(good, from_dir)
    if any(not c["pass"] for c in v):
        raise ModelIntegrityError(f"cannot rollback: source {from_dir} does not verify as {good_model_id}")
    for name in good.artifact_files:
        shutil.copy2(from_dir / name, to_dir / name)
    # Verify after rollback
    after = verify_artifacts(good, to_dir)
    return {
        "rolled_back_from": bad_model_id,
        "rolled_back_to": good_model_id,
        "files_copied": len(good.artifact_files),
        "post_rollback_verify_pass": all(c["pass"] for c in after),
        "checks": after,
    }
