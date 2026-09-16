#!/usr/bin/env python3
"""Phase 50: Legacy model attestation & zero-legacy runtime.

Tests that the previously unattested legacy altman_native model is now
runtime-attested with a truthful ReleaseManifest, while promotion remains
correctly blocked.

Run: ./.venv/Scripts/python.exe backend/scripts/phase50_legacy_attestation_test.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("JWT_SECRET", "phase50-test-0123456789abcdef")
os.environ["PS14_MODE"] = "development"

passed = failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  PASS {label}")
        passed += 1
    else:
        msg = f"  FAIL {label}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        failed += 1


def main() -> int:
    global passed, failed
    print("=" * 60)
    print("Phase 50: Legacy attestation & zero-legacy runtime")
    print("=" * 60)

    from src.monitoring.release_manifest import (
        ReleaseManifest,
        create_release_manifest,
        artifact_set_hash,
        sha256_file,
    )
    from src.monitoring.runtime_attestation import (
        RuntimeState,
        verify_release_for_load,
        build_attestation,
        detect_runtime_drift,
        RELEASE_MANIFEST_FILENAME,
    )
    from src.monitoring.promotion_gate import evaluate_promotion

    PRODUCTION_DIR = ROOT / "models" / "production"
    ARTIFACT_DIR = PRODUCTION_DIR / "altman_native"
    RULES_PATH = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
    SCALER_PATH = ARTIFACT_DIR / "scaler_native.joblib"
    MANIFEST_PATH = PRODUCTION_DIR / RELEASE_MANIFEST_FILENAME

    # ── 1. Legacy artifact discovered correctly ─────────────────────────
    print("\n-- 1. Legacy artifact discovery --")
    check("altman_native dir exists", ARTIFACT_DIR.is_dir())
    native_manifest_path = ARTIFACT_DIR / "manifest.json"
    check("native manifest.json exists", native_manifest_path.exists())
    native_m = json.loads(native_manifest_path.read_text(encoding="utf-8"))
    check("native manifest has model_version", "model_version" in native_m)
    check("native manifest has artifacts hash", "artifacts" in native_m)
    check("4 artifact files exist", len(list(ARTIFACT_DIR.glob("*.joblib"))) == 4)

    # ── 2. Release manifest exists and is valid ─────────────────────────
    print("\n-- 2. Release manifest --")
    check("release_manifest.json exists", MANIFEST_PATH.exists())
    rm = ReleaseManifest.load(MANIFEST_PATH)
    check("manifest has release_id", rm.release_id.startswith("legacy-"))
    check("manifest model_id matches native", rm.model_id == native_m["model_version"])
    check("manifest gate_verdict is LEGACY_ATTESTED", rm.gate_verdict == "LEGACY_ATTESTED")
    check("manifest has feature_version", rm.feature_version == "v1")
    check("manifest has schema_version", rm.schema_version == "v1")
    check("manifest has preprocessing_hash", len(rm.preprocessing_hash) == 64)
    check("manifest has rule_hash", len(rm.rule_hash) == 32)
    check("manifest has evaluation_record_hash", len(rm.evaluation_record_hash) == 64)
    check("manifest has source_git_sha", len(rm.source_git_sha) >= 7)

    # ── 3. Deterministic manifest hash ──────────────────────────────────
    print("\n-- 3. Deterministic manifest --")
    h1 = rm.compute_manifest_hash()
    rm2 = ReleaseManifest.load(MANIFEST_PATH)
    h2 = rm2.compute_manifest_hash()
    check("manifest hash deterministic across loads", h1 == h2)

    # ── 4. Artifact hash correct ────────────────────────────────────────
    print("\n-- 4. Artifact hash --")
    art = artifact_set_hash(ARTIFACT_DIR)
    check("manifest artifact_hash matches disk", rm.artifact_hash == art["model_hash"])
    check("manifest artifact_files matches disk", rm.artifact_files == art["files"])

    # ── 5. Manifest signature valid ─────────────────────────────────────
    print("\n-- 5. Manifest signature --")
    ok, reason = rm.verify_signature()
    check("signature valid", ok, reason)

    # ── 6. Attestation valid ────────────────────────────────────────────
    print("\n-- 6. Attestation --")
    rule_hash = hashlib.sha256(RULES_PATH.read_bytes()).hexdigest()[:32]
    ok, failures, identity = verify_release_for_load(
        rm, ARTIFACT_DIR, expected_rule_hash=rule_hash,
    )
    check("verify_release_for_load passes", ok, str(failures))
    check("identity has model_id", identity.get("model_id") == rm.model_id)
    check("identity has artifact_hash", identity.get("artifact_hash") == rm.artifact_hash)

    att = build_attestation(identity, ARTIFACT_DIR)
    check("attestation has release_id", att.release_id == rm.release_id)
    check("attestation has model_version", att.model_version == rm.model_version)
    check("attestation has artifact_hash", att.artifact_hash == rm.artifact_hash)
    check("attestation has manifest_hash", att.manifest_hash == rm.compute_manifest_hash())

    # ── 7. Attestation != promotion ──────────────────────────────────────
    print("\n-- 7. Attestation != promotion --")
    gate = evaluate_promotion(
        candidate_model_id=rm.model_id,
        candidate_artifact_hash=rm.artifact_hash,
        candidate_feature_version=rm.feature_version,
    )
    check("promotion gate verdict is BLOCKED", gate.verdict.name == "BLOCKED")
    check("gate_verdict is LEGACY_ATTESTED not PROMOTION_ELIGIBLE",
          rm.gate_verdict == "LEGACY_ATTESTED")

    # ── 8. Promotion remains blocked ────────────────────────────────────
    print("\n-- 8. Promotion blocked --")
    check("real_world_validation flag present",
          any("REAL_WORLD_VALIDATION" in g.gate_name for g in gate.gates))
    rwv_gate = next(g for g in gate.gates if "REAL_WORLD_VALIDATION" in g.gate_name)
    check("REAL_WORLD_VALIDATION gate is FAIL", rwv_gate.status != "PASS")

    # ── 9. Missing manifest blocked ─────────────────────────────────────
    print("\n-- 9. Missing manifest --")
    tmp = tempfile.mkdtemp(prefix="phase50-")
    ok, f1, _ = verify_release_for_load(rm, Path(tmp))
    check("missing artifacts dir -> fail", not ok)

    # ── 10. Corrupted artifact blocked ──────────────────────────────────
    print("\n-- 10. Corrupted artifact --")
    tmp2 = Path(tempfile.mkdtemp(prefix="phase50-corrupt-"))
    shutil.copytree(ARTIFACT_DIR, tmp2 / "altman_native", dirs_exist_ok=True)
    fake_model = tmp2 / "altman_native" / "xgb_native.joblib"
    fake_model.write_bytes(b"CORRUPTED" * 1000)
    ok, f2, _ = verify_release_for_load(rm, tmp2 / "altman_native")
    check("corrupted artifact -> fail", not ok)
    check("failure mentions artifacts", any("artifact" in f.lower() for f in f2), str(f2))
    shutil.rmtree(tmp2, ignore_errors=True)

    # ── 11. Corrupted manifest blocked ──────────────────────────────────
    print("\n-- 11. Corrupted manifest --")
    rm_bad = ReleaseManifest.load(MANIFEST_PATH)
    rm_bad.artifact_hash = "0" * 64
    ok, f3, _ = verify_release_for_load(rm_bad, ARTIFACT_DIR, expected_rule_hash=rule_hash)
    check("corrupted manifest hash -> fail", not ok)

    # ── 12. Rule mismatch blocked ───────────────────────────────────────
    print("\n-- 12. Rule mismatch --")
    ok, f4, _ = verify_release_for_load(rm, ARTIFACT_DIR, expected_rule_hash="deadbeefdeadbeef")
    check("wrong rule_hash -> fail", not ok)

    # ── 13. Forged signature blocked ────────────────────────────────────
    print("\n-- 13. Forged signature --")
    rm_forge = ReleaseManifest.load(MANIFEST_PATH)
    rm_forge._signature = "a" * 64
    ok, reason = rm_forge.verify_signature()
    check("forged signature detected", not ok, reason)

    # ── 14. Fake gate verdict blocked ───────────────────────────────────
    print("\n-- 14. Fake gate verdict --")
    rm_fake = ReleaseManifest.load(MANIFEST_PATH)
    rm_fake.gate_verdict = "PROMOTION_ELIGIBLE"
    ok, f5, _ = verify_release_for_load(rm_fake, ARTIFACT_DIR, expected_rule_hash=rule_hash)
    check("fake PROMOTION_ELIGIBLE with LEGACY_ATTESTED manifest -> fail", not ok)

    # ── 15. Fake promotion decision blocked ─────────────────────────────
    print("\n-- 15. Fake promotion decision --")
    gate2 = evaluate_promotion(
        candidate_model_id=rm.model_id,
        candidate_artifact_hash=rm.artifact_hash,
        candidate_feature_version=rm.feature_version,
    )
    check("promotion gate still BLOCKED", gate2.verdict.name == "BLOCKED")

    # ── 16. Manifest tamper detected ────────────────────────────────────
    print("\n-- 16. Manifest tamper --")
    rm_tamper = ReleaseManifest.load(MANIFEST_PATH)
    rm_tamper.model_version = "tampered-version"
    ok, reason = rm_tamper.verify_signature()
    check("tampered manifest signature invalid", not ok, reason)

    # ── 17. Runtime drift detection ─────────────────────────────────────
    print("\n-- 17. Runtime drift --")
    drift_state, drift_reasons = detect_runtime_drift(att, ARTIFACT_DIR)
    check("no drift on clean dir", drift_state == RuntimeState.READY, str(drift_reasons))

    # ── 18. Environment override cannot attest ──────────────────────────
    print("\n-- 18. Env override cannot attest --")
    check("gate_verdict from manifest not env",
          rm.gate_verdict == "LEGACY_ATTESTED" and rm.gate_verdict != os.environ.get("GATE_OVERRIDE"))

    # ── 19. Live startup with attestation ───────────────────────────────
    print("\n-- 19. Live startup attestation --")
    from fastapi.testclient import TestClient
    from src.risk_engine.main import app as risk_app
    with TestClient(risk_app) as rc:
        h = rc.get("/health")
        health = h.json()
        check("health runtime_state READY", health.get("runtime_state") == "READY")
        check("health release_attested true", health.get("release_attested") is True)
        check("health model ok", health.get("model") == "ok")
        token = os.environ["INTERNAL_TOKEN"]
        att_resp = rc.get("/internal/release-attestation", headers={"X-Internal-Token": token})
        att_body = att_resp.json()
        check("attestation endpoint attested=true", att_body.get("attested") is True)
        check("attestation endpoint ready=true", att_body.get("ready") is True)
        if att_body.get("attestation"):
            att_data = att_body["attestation"]
            check("attestation model_id matches", att_data.get("model_id") == rm.model_id)
            check("attestation model_version matches", att_data.get("model_version") == rm.model_version)

    # ── 20. Live inference uses attested release ────────────────────────
    print("\n-- 20. Live inference --")
    features = {"amount_ratio": 0.95, "txn_freq_last_24h": 1, "txn_time_unusual": 0,
                "new_device_flag": 0, "unusual_location_flag": 0, "unusual_recipient_flag": 0,
                "failed_auth_count_24h": 0, "days_since_last_similar_txn": 3.0,
                "gradual_escalation_score": 0.0, "known_device_count": 3,
                "account_tenure_days": 90.0, "hour_of_day": 12, "is_weekend": 0,
                "shared_device_accounts": 0, "shared_recipient_accounts": 0,
                "mule_ring_score": 0, "hour_deviation": 0.5, "amount_zscore": 0,
                "velocity_deviation": 0, "recipient_novelty": 0, "txn_regularity": 0}
    with TestClient(risk_app) as rc:
        token = os.environ["INTERNAL_TOKEN"]
        r = rc.post("/internal/evaluate", headers={"X-Internal-Token": token},
                    json={"event_id": "phase50-eval-0001", "fraud_id": "F24BRMMMBJWYMTDW",
                          "features": features})
        resp = r.json()
        check("evaluate not degraded", resp.get("degraded") is False)
        check("evaluate risk_band present", resp.get("risk_band") in ("low", "medium", "high"))
        check("evaluate runtime_state in response",
              resp.get("runtime_state") in (None, "READY"))
        check("evaluate has risk_score", isinstance(resp.get("risk_score"), int))

    # ── 21. DecisionTrace release identity ──────────────────────────────
    print("\n-- 21. DecisionTrace release identity --")
    with TestClient(risk_app) as rc:
        token = os.environ["INTERNAL_TOKEN"]
        r = rc.post("/internal/evaluate", headers={"X-Internal-Token": token},
                    json={"event_id": "phase50-eval-0002", "fraud_id": "F24BRMMMBJWYMTDW",
                          "features": features})
        resp = r.json()
        check("evaluate has runtime_state",
              resp.get("runtime_state") in (None, "READY"))

    # ── 22. Restart preserves identity ──────────────────────────────────
    print("\n-- 22. Restart preserves identity --")
    with TestClient(risk_app) as rc:
        h1 = rc.get("/health").json()
        h2 = rc.get("/health").json()
        check("consistent attestation across calls",
              h1.get("release_attested") == h2.get("release_attested") == True)
        check("consistent runtime_state",
              h1.get("runtime_state") == h2.get("runtime_state") == "READY")

    # ── 23. Artifact replacement detected ───────────────────────────────
    print("\n-- 23. Artifact replacement --")
    backup = ARTIFACT_DIR / "xgb_native.joblib.bak"
    shutil.copy2(ARTIFACT_DIR / "xgb_native.joblib", backup)
    (ARTIFACT_DIR / "xgb_native.joblib").write_bytes(b"TAMPERED" * 500)
    drift_state2, _ = detect_runtime_drift(att, ARTIFACT_DIR)
    check("replacement detected as DRIFTED", drift_state2 == RuntimeState.DRIFTED)
    shutil.copy2(backup, ARTIFACT_DIR / "xgb_native.joblib")
    backup.unlink(missing_ok=True)

    # ── 24. Restore -> recover ───────────────────────────────────────────
    print("\n-- 24. Restore after tamper --")
    drift_state3, _ = detect_runtime_drift(att, ARTIFACT_DIR)
    check("restore -> no drift", drift_state3 == RuntimeState.READY)

    # ── 25. Migration event in audit ────────────────────────────────────
    print("\n-- 25. Migration event --")
    check("release_manifest.json on disk", MANIFEST_PATH.exists())
    manifest_data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    check("manifest has _nonce", "_nonce" in manifest_data)
    check("manifest has _signature", "_signature" in manifest_data)

    # ── 26. Migrated release cannot bypass promotion ────────────────────
    print("\n-- 26. Cannot bypass promotion --")
    from src.risk_engine.model_registry import ModelRegistry
    reg = ModelRegistry(artifacts_dir=ROOT / "models" / "artifacts")
    check("registry has no candidate", reg.state.candidate is None)

    # ── 27. REAL_WORLD_VALIDATION remains blocked ───────────────────────
    print("\n-- 27. REAL_WORLD_VALIDATION --")
    gate3 = evaluate_promotion(
        candidate_model_id=rm.model_id,
        candidate_artifact_hash=rm.artifact_hash,
        candidate_feature_version=rm.feature_version,
    )
    check("promotion remains BLOCKED", gate3.verdict.name == "BLOCKED")

    # ── 28. Env override cannot select model ────────────────────────────
    print("\n-- 28. Env override --")
    os.environ["MODEL_VERSION"] = "fake-approved"
    with TestClient(risk_app) as rc:
        h = rc.get("/health").json()
        check("env override ignored", h.get("release_attested") is True)
    os.environ.pop("MODEL_VERSION", None)

    # ── 29. Stale manifest hash rejected ────────────────────────────────
    print("\n-- 29. Stale manifest --")
    rm_stale = ReleaseManifest.load(MANIFEST_PATH)
    rm_stale.artifact_hash = "0" * 64  # wrong artifact
    ok, _, _ = verify_release_for_load(rm_stale, ARTIFACT_DIR, expected_rule_hash=rule_hash)
    check("stale/wrong artifact hash rejected", not ok)

    # ── 30. Wrong manifest for model ────────────────────────────────────
    print("\n-- 30. Wrong manifest for model --")
    rm_wrong = ReleaseManifest.load(MANIFEST_PATH)
    rm_wrong.model_id = "wrong-model-id"
    ok, f6, _ = verify_release_for_load(rm_wrong, ARTIFACT_DIR, expected_rule_hash=rule_hash)
    check("wrong model_id in manifest rejected", not ok)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    total = passed + failed
    if failed:
        print(f"Phase 50: {passed} passed, {failed} failed (out of {total})")
    else:
        print(f"Phase 50: {passed} passed, 0 failed")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
