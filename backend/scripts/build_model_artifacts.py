#!/usr/bin/env python3
"""Reproducible model artifact generation pipeline.

Generates ALL required production artifacts from scratch:
  1. Training data → synthetic (if real data unavailable)
  2. Model training → train_compare.py
  3. Calibration → PlattCalibration on validation data
  4. Model manifest → metadata.json with hashes
  5. Artifact verification → loads every artifact

Usage:
    python scripts/build_model_artifacts.py [--skip-training]
    python scripts/build_model_artifacts.py               # full build
    python scripts/build_model_artifacts.py --skip-training  # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
PY = sys.executable
ARTIFACTS = ROOT / "models" / "artifacts"

# Required artifacts for production
REQUIRED_ARTIFACTS = [
    "scaler.joblib",
    "logistic_regression.joblib",
    "random_forest.joblib",
    "xgboost.joblib",
    "isolation_forest.joblib",
    "stacker.joblib",
    "iso_train_scores.joblib",
    "calibrator.joblib",
]


def file_hash(path: Path) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def verify_artifacts() -> dict[str, bool]:
    """Verify all required artifacts exist and load."""
    import joblib
    results = {}
    for name in REQUIRED_ARTIFACTS:
        path = ARTIFACTS / name
        if not path.exists():
            print(f"  MISSING: {name}")
            results[name] = False
            continue
        try:
            obj = joblib.load(path)
            # Basic sanity: non-None, non-empty
            results[name] = obj is not None
            status = "OK" if results[name] else "EMPTY"
            print(f"  {status}: {name} ({path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            results[name] = False
    return results


def update_manifest():
    """Add artifact hashes to metadata.json."""
    meta_path = ARTIFACTS / "metadata.json"
    if not meta_path.exists():
        print("  metadata.json not found — skipping manifest update")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Add artifact hashes
    hashes = {}
    for name in REQUIRED_ARTIFACTS:
        path = ARTIFACTS / name
        if path.exists():
            hashes[name] = file_hash(path)

    # Add rules hash
    rules_path = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
    if rules_path.exists():
        hashes["rules.yaml"] = file_hash(rules_path)

    meta["artifact_hashes"] = hashes
    meta["feature_version"] = "v1"

    # Make data paths relative
    for key in ("data", "feedback_source"):
        if key in meta:
            val = meta[key]
            if isinstance(val, str) and Path(val).is_absolute():
                try:
                    meta[key] = str(Path(val).relative_to(ROOT))
                except ValueError:
                    pass

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Updated metadata.json with {len(hashes)} artifact hashes")


def train_pipeline(timeout: int = 600):
    """Run the full training pipeline."""
    import subprocess
    data_path = ROOT / "data" / "transactions.csv"
    if not data_path.exists():
        print(f"  Training data not found: {data_path}")
        print("  Generating synthetic data first...")
        result = subprocess.run(
            [PY, str(ROOT / "backend" / "src" / "generate_synthetic_data.py")],
            cwd=str(ROOT), timeout=120,
        )
        if result.returncode != 0:
            print("  FATAL: Synthetic data generation failed")
            sys.exit(1)

    print(f"  Running training pipeline (timeout: {timeout}s)...")
    try:
        result = subprocess.run(
            [PY, str(ROOT / "backend" / "src" / "train_compare.py"),
             "--data", "data/transactions.csv",
             "--outdir", "models/artifacts"],
            cwd=str(ROOT), timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  TRAINING TIMEOUT after {timeout}s")
        print(f"  Increase timeout: python scripts/build_model_artifacts.py --train-timeout {timeout * 2}")
        sys.exit(1)
    if result.returncode != 0:
        print(f"  Training pipeline failed (exit {result.returncode})")
        sys.exit(1)
    print("  Training pipeline completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Build/rebuild model artifacts")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, verify existing artifacts only")
    parser.add_argument("--train-timeout", type=int, default=600,
                        help="Training timeout in seconds (default: 600)")
    args = parser.parse_args()

    print("=" * 60)
    print("PS-14 MODEL ARTIFACT BUILD PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Train (unless skipped)
    if not args.skip_training:
        print("[1] Training pipeline")
        train_pipeline(timeout=args.train_timeout)
        print()
    else:
        print("[1] Training — SKIPPED (--skip-training)")
        print()

    # Step 2: Verify artifacts
    print("[2] Verify artifacts")
    results = verify_artifacts()
    all_ok = all(results.values())
    print(f"  {'ALL ARTIFACTS VERIFIED' if all_ok else 'SOME ARTIFACTS MISSING/FAILED'}")
    print()

    # Step 3: Update manifest
    print("[3] Update model manifest")
    update_manifest()
    print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("BUILD COMPLETE — all artifacts present and loadable")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"BUILD INCOMPLETE — {len(failed)} artifact(s) missing: {failed}")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
