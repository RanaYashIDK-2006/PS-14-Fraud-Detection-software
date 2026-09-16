#!/usr/bin/env python3
"""Phase 50: Reconstruct a ReleaseManifest for the legacy altman_native model.

Produces a truthful, signed ReleaseManifest from evidence ALREADY present in the
repository — existing artifact bytes, the altman_native/manifest.json, the model
record at models/model_records/, rules.yaml, and feature_contract.py.  No
fabricated evaluation scores, no invented provenance, no promotion bypass.

The resulting manifest makes the currently served legacy artifact
cryptographically identifiable and runtime-attested, while REAL_WORLD_VALIDATION
remains BLOCKED_PENDING_ELIGIBLE_DATASET.

Run from the project root:
  ./.venv/Scripts/python.exe backend/scripts/reconstruct_legacy_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

os.environ.setdefault("JWT_SECRET", os.environ.get("JWT_SECRET", "reconstruct-legacy-secret-0123456789abcdef"))

from src.monitoring.release_manifest import (  # noqa: E402
    ReleaseManifest,
    create_release_manifest,
    sha256_file,
    artifact_set_hash,
)
from src.monitoring.feature_contract import ML_FEATURE_VERSION  # noqa: E402
from src.monitoring.runtime_enforcement import enforce_before_inference  # noqa: E402

# ── Paths ───────────────────────────────────────────────────────────────

PRODUCTION_DIR = ROOT / "models" / "production"
ARTIFACT_DIR = PRODUCTION_DIR / "altman_native"
MODEL_RECORD_PATH = ROOT / "models" / "model_records" / "altman_native_E_hardneg_cert_20260904.json"
NATIVE_MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
RULES_PATH = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
RELEASE_MANIFEST_PATH = PRODUCTION_DIR / "release_manifest.json"
SCALER_PATH = ARTIFACT_DIR / "scaler_native.joblib"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    print("== Phase 50: Legacy release reconstruction ==\n")

    # ── 1. Read existing evidence ───────────────────────────────────────
    if not MODEL_RECORD_PATH.exists():
        sys.exit(f"Model record not found: {MODEL_RECORD_PATH}")
    if not ARTIFACT_DIR.is_dir():
        sys.exit(f"Artifact directory not found: {ARTIFACT_DIR}")
    if not RULES_PATH.exists():
        sys.exit(f"Rules not found: {RULES_PATH}")

    model_record = _read_json(MODEL_RECORD_PATH)
    native_manifest = _read_json(NATIVE_MANIFEST_PATH)

    model_id = native_manifest["model_version"]  # "altman_native_E_hardneg_cert_20260904"
    model_version = model_id
    training_seed = model_record.get("training_seed", 42)
    training_config = model_record.get("training", {})

    print(f"  model_id:          {model_id}")
    print(f"  model_version:     {model_version}")
    print(f"  training_seed:     {training_seed}")

    # ── 2. Compute artifact hashes ──────────────────────────────────────
    # artifact_set_hash skips .json, .bak, .tmp — only .joblib files hashed.
    art = artifact_set_hash(ARTIFACT_DIR)
    print(f"  artifact_hash:     {art['model_hash'][:32]}…")
    print(f"  artifact_files:    {list(art['files'].keys())}")

    # ── 3. Compute preprocessing hash (scaler only) ─────────────────────
    preprocessing_hash = sha256_file(SCALER_PATH)
    print(f"  preprocessing_hash: {preprocessing_hash[:32]}…")

    # ── 4. Compute rule hash ────────────────────────────────────────────
    rule_hash = hashlib.sha256(RULES_PATH.read_bytes()).hexdigest()[:32]
    print(f"  rule_hash:         {rule_hash}")

    # ── 5. Evaluation record hash ───────────────────────────────────────
    # Use the model record itself as the evaluation evidence — it contains
    # the actual training metrics, validation metrics, test metrics, and
    # the promotion recommendation.  Hash the canonical JSON.
    eval_hash = hashlib.sha256(
        json.dumps(model_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    print(f"  eval_record_hash:  {eval_hash[:32]}…")

    # ── 6. Training config hash ─────────────────────────────────────────
    training_config_hash = hashlib.sha256(
        json.dumps(training_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    print(f"  training_cfg_hash: {training_config_hash[:32]}…")

    # ── 7. Source git SHA ───────────────────────────────────────────────
    git_sha = ""
    try:
        git_sha = subprocess_capture(["git", "rev-parse", "--short", "HEAD"])
    except Exception:
        pass
    print(f"  source_git_sha:    {git_sha or 'unknown'}")

    # ── 8. Feature/schema version ───────────────────────────────────────
    feature_version = ML_FEATURE_VERSION  # "v1"
    schema_version = feature_version
    print(f"  feature_version:   {feature_version}")
    print(f"  schema_version:    {schema_version}")

    # ── 9. Create and sign the manifest ─────────────────────────────────
    manifest = create_release_manifest(
        release_id=f"legacy-{model_id}",
        model_id=model_id,
        model_version=model_version,
        artifact_dir=ARTIFACT_DIR,
        feature_version=feature_version,
        schema_version=schema_version,
        preprocessing_hash=preprocessing_hash,
        rule_hash=rule_hash,
        evaluation_record_hash=eval_hash,
        training_config_hash=training_config_hash,
        source_git_sha=git_sha,
        python_version=sys.version.split()[0],
        dependency_fingerprint="",
        training_seed=training_seed,
        gate_verdict="LEGACY_ATTESTED",
        gate_timestamp=0.0,
    )

    # ── 10. Verify the manifest we just created ─────────────────────────
    print("\n--- Verification ---")
    ok, reason = manifest.verify_signature()
    print(f"  signature:         {'OK' if ok else 'FAIL'} — {reason}")

    ok, reason = manifest.verify_artifacts(ARTIFACT_DIR)
    print(f"  artifacts:         {'OK' if ok else 'FAIL'} — {reason}")

    # ── 11. Save ────────────────────────────────────────────────────────
    manifest.save(RELEASE_MANIFEST_PATH)
    print(f"\n  Saved: {RELEASE_MANIFEST_PATH}")
    print(f"  manifest_hash:     {manifest.compute_manifest_hash()[:32]}…")

    # ── 12. Verify artifact_set_hash matches native manifest hashes ─────
    native_hashes = native_manifest.get("artifacts", {})
    mismatches = []
    for name, expected in native_hashes.items():
        actual = art["files"].get(name)
        if actual != expected:
            mismatches.append(f"{name}: expected={expected[:16]}… actual={actual[:16] if actual else 'MISSING'}…")
    if mismatches:
        print(f"\n  WARNING: native manifest hash mismatches: {mismatches}")
    else:
        print("\n  All artifact hashes match native manifest ✓")

    # ── 13. Quick inference test ────────────────────────────────────────
    print("\n--- Quick inference smoke test ---")
    test_vector = {
        "amount_ratio": 1.0, "txn_freq_last_24h": 1, "txn_time_unusual": 0,
        "new_device_flag": 0, "unusual_location_flag": 0, "unusual_recipient_flag": 0,
        "failed_auth_count_24h": 0, "days_since_last_similar_txn": 3.0,
        "gradual_escalation_score": 0.0, "known_device_count": 3,
        "account_tenure_days": 90.0, "hour_of_day": 12, "is_weekend": 0,
        "shared_device_accounts": 0, "shared_recipient_accounts": 0,
        "mule_ring_score": 0, "hour_deviation": 0.5, "amount_zscore": 0,
        "velocity_deviation": 0, "recipient_novelty": 0, "txn_regularity": 0,
    }
    enforcement = enforce_before_inference(test_vector)
    print(f"  enforcement verdict: {enforcement.verdict}")

    print("\n== Reconstruction complete ==")
    print(f"   release_id:    {manifest.release_id}")
    print(f"   gate_verdict:  {manifest.gate_verdict}")
    print(f"   REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET")


def subprocess_capture(cmd: list[str]) -> str:
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=10)
    return r.stdout.strip()


if __name__ == "__main__":
    main()
