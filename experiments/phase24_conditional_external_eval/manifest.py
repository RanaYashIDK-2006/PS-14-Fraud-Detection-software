"""Manifest and hash handling for model artifact verification.

Verifies model integrity before evaluation and records hashes
for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_artifacts(
    model_dir: Path,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify model artifact integrity.

    Checks that all expected files exist and hashes match.
    """
    results = {
        "model_dir": str(model_dir),
        "artifacts": {},
        "all_match": True,
        "missing_files": [],
        "hash_mismatches": [],
    }

    if not model_dir.exists():
        results["status"] = "MODEL_DIR_NOT_FOUND"
        results["all_match"] = False
        return results

    # Find all model files
    model_files = list(model_dir.glob("*.joblib")) + list(model_dir.glob("*.json"))

    for f in model_files:
        actual_hash = compute_file_hash(f)
        artifact_info = {
            "path": str(f.name),
            "size_bytes": f.stat().st_size,
            "hash": actual_hash,
        }

        if expected_hashes and f.name in expected_hashes:
            expected = expected_hashes[f.name]
            artifact_info["expected_hash"] = expected
            artifact_info["match"] = actual_hash == expected
            if not actual_hash == expected:
                results["hash_mismatches"].append(f.name)
                results["all_match"] = False
        else:
            artifact_info["expected_hash"] = None
            artifact_info["match"] = None

        results["artifacts"][f.name] = artifact_info

    # Check for missing expected files
    if expected_hashes:
        for fname in expected_hashes:
            if fname not in results["artifacts"]:
                results["missing_files"].append(fname)
                results["all_match"] = False

    results["status"] = "VERIFIED" if results["all_match"] else "VERIFICATION_FAILED"
    return results


def load_production_manifest(project_root: Path) -> dict[str, Any]:
    """Load the production model manifest."""
    manifest_path = project_root / "models" / "production" / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"error": "Production manifest not found"}


def load_e_hardneg_manifest(project_root: Path) -> dict[str, Any]:
    """Load the E_hardneg certified model manifest."""
    manifest_path = (
        project_root / "models" / "production" / "altman_native" / "manifest.json"
    )
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"error": "E_hardneg manifest not found"}


def load_p20_feature_list(project_root: Path) -> list[str]:
    """Load the P20 feature list."""
    feature_path = project_root / "reports" / "phase20" / "model_artifacts" / "feature_list.json"
    if feature_path.exists():
        return json.loads(feature_path.read_text(encoding="utf-8"))
    return []


def write_manifest_artifacts(
    output_dir: Path,
    project_root: Path,
    e_hardneg_verification: dict,
    p20_features: list[str],
) -> dict[str, Any]:
    """Write manifest verification artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    e_hardneg_manifest = load_e_hardneg_manifest(project_root)
    prod_manifest = load_production_manifest(project_root)

    artifact = {
        "e_hardneg_manifest": e_hardneg_manifest,
        "production_manifest": prod_manifest,
        "e_hardneg_verification": e_hardneg_verification,
        "p20_feature_count": len(p20_features),
        "p20_features": p20_features,
    }

    (output_dir / "07_manifest_verification.json").write_text(
        json.dumps(artifact, indent=2, default=str), encoding="utf-8"
    )

    return artifact
