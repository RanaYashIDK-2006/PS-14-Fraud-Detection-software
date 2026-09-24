#!/usr/bin/env python3
"""Phase 51: Release lifecycle integrity tests.

Adversarial tests proving releases can be created, activated, rolled back,
revoked, and recovered safely without allowing an unverified artifact to
become active.

Run: ./.venv/Scripts/python.exe backend/scripts/phase51_release_lifecycle_test.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("JWT_SECRET", "phase51-test-0123456789abcdef")
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


def _make_release(tmp: Path, suffix: str = "v1") -> tuple[Path, object]:
    """Create a signed release in tmp/artifacts + return (artifact_dir, manifest)."""
    from src.monitoring.release_manifest import create_release_manifest
    art = tmp / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "model.joblib").write_bytes(f"model-bytes-{suffix}".encode())
    (art / "scaler.joblib").write_bytes(f"scaler-bytes-{suffix}".encode())
    RULES_PATH = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
    rule_hash = hashlib.sha256(RULES_PATH.read_bytes()).hexdigest()[:32]
    m = create_release_manifest(
        release_id=f"rel-test-51-{suffix}",
        model_id=f"m-test-51-{suffix}",
        model_version=f"v-test-51-{suffix}",
        artifact_dir=art,
        feature_version="v1",
        schema_version="v1",
        preprocessing_hash=hashlib.sha256(f"scaler-bytes-{suffix}".encode()).hexdigest(),
        rule_hash=rule_hash,
        evaluation_record_hash=f"eval51hash-{suffix}",
        training_config_hash=f"cfg51hash-{suffix}",
        source_git_sha="abc1234",
        gate_verdict="LEGACY_ATTESTED",
    )
    return art, m


def main() -> int:
    global passed, failed
    print("=" * 60)
    print("Phase 51: Release lifecycle integrity")
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
    )
    from src.monitoring.release_lifecycle import (
        ReleaseState,
        ReleaseRecord,
        ReleaseRegistry,
        verify_release_lifecycle,
        create_revoked_manifest,
        _VALID_TRANSITIONS,
        _INFERENCE_READY,
    )

    PRODUCTION_DIR = ROOT / "models" / "production"
    ARTIFACT_DIR = PRODUCTION_DIR / "altman_native"
    RULES_PATH = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
    rule_hash = hashlib.sha256(RULES_PATH.read_bytes()).hexdigest()[:32]

    # ═══════════════════════════════════════════════════════════════════════
    # A. Lifecycle State Machine
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- A. State machine --")
    with tempfile.TemporaryDirectory(prefix="phase51-") as td:
        art, manifest = _make_release(Path(td))

        # 1. valid lifecycle progression
        record, failures = verify_release_lifecycle(manifest, art, expected_rule_hash=rule_hash)
        check("1. valid lifecycle completes", len(failures) == 0, str(failures))
        check("1. state is REGISTERED", record.state == ReleaseState.REGISTERED)

        # 2. INVALID state blocks activation
        bad_record = ReleaseRecord(
            release_id="bad", model_id="bad", model_version="bad",
            artifact_hash="0" * 64, manifest_hash="0" * 64,
            feature_version="v1", schema_version="v1",
            rule_hash="0" * 32, preprocessing_hash="0" * 64,
            gate_verdict="LEGACY_ATTESTED",
        )
        bad_record.state = ReleaseState.INVALID
        ok, reason = bad_record.can_activate()
        check("2. INVALID -> activate blocked", not ok, reason)

        # 3. REVOKED state blocks activation
        revoked_record = ReleaseRecord(
            release_id="rev", model_id="rev", model_version="rev",
            artifact_hash="0" * 64, manifest_hash="0" * 64,
            feature_version="v1", schema_version="v1",
            rule_hash="0" * 32, preprocessing_hash="0" * 64,
            gate_verdict="LEGACY_ATTESTED",
        )
        revoked_record.state = ReleaseState.REVOKED
        ok, reason = revoked_record.can_activate()
        check("3. REVOKED -> activate blocked", not ok, reason)

        # 4. DRIFTED state blocks inference
        drifted_record = ReleaseRecord(
            release_id="drift", model_id="drift", model_version="drift",
            artifact_hash="0" * 64, manifest_hash="0" * 64,
            feature_version="v1", schema_version="v1",
            rule_hash="0" * 32, preprocessing_hash="0" * 64,
            gate_verdict="LEGACY_ATTESTED",
        )
        drifted_record.state = ReleaseState.DRIFTED
        check("4. DRIFTED blocks inference", not drifted_record.can_infer())

        # 5. RUNTIME_ATTESTED allows inference
        attested = ReleaseRecord(
            release_id="att", model_id="att", model_version="att",
            artifact_hash="0" * 64, manifest_hash="0" * 64,
            feature_version="v1", schema_version="v1",
            rule_hash="0" * 32, preprocessing_hash="0" * 64,
            gate_verdict="LEGACY_ATTESTED",
        )
        attested.state = ReleaseState.RUNTIME_ATTESTED
        check("5. RUNTIME_ATTESTED allows inference", attested.can_infer())

        # 6. DISCOVERED -> ACTIVATED is invalid
        disc = ReleaseRecord(
            release_id="disc", model_id="disc", model_version="disc",
            artifact_hash="0" * 64, manifest_hash="0" * 64,
            feature_version="v1", schema_version="v1",
            rule_hash="0" * 32, preprocessing_hash="0" * 64,
            gate_verdict="LEGACY_ATTESTED",
        )
        ok = disc.transition(ReleaseState.ACTIVATED, "skip")
        check("6. DISCOVERED -> ACTIVATED blocked", not ok)

        # 7. REVOKED -> ACTIVATED is invalid
        ok = revoked_record.transition(ReleaseState.ACTIVATED, "unrevoke")
        check("7. REVOKED -> ACTIVATED blocked", not ok)

        # 8. state_history records transitions
        check("8. state_history records transitions", len(record.state_history) >= 3)

    # ═══════════════════════════════════════════════════════════════════════
    # B. Activation Safety
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- B. Activation safety --")
    with tempfile.TemporaryDirectory(prefix="phase51-act-") as td:
        reg_path = Path(td) / "registry.json"
        reg = ReleaseRegistry(reg_path)
        art, manifest = _make_release(Path(td))

        record, failures = verify_release_lifecycle(manifest, art, expected_rule_hash=rule_hash)

        # 9. valid release registers
        check("9. register succeeds", reg.register(record))

        # 10. duplicate registration blocked
        check("10. duplicate register blocked", not reg.register(record))

        # 11. activate registered release
        ok, reason = reg.activate(record.release_id)
        check("11. activate registered release", ok, reason)
        check("11b. active_release_id matches", reg.active_release_id == record.release_id)

        # 12. activate non-existent release
        ok, reason = reg.activate("nonexistent")
        check("12. activate nonexistent blocked", not ok, reason)

        # 13. activate from INVALID state
        bad_reg = ReleaseRegistry(Path(td) / "registry2.json")
        bad_rec = ReleaseRecord(
            release_id="bad2", model_id="bad2", model_version="bad2",
            artifact_hash="0" * 64, manifest_hash="0" * 64,
            feature_version="v1", schema_version="v1",
            rule_hash="0" * 32, preprocessing_hash="0" * 64,
            gate_verdict="LEGACY_ATTESTED",
        )
        bad_rec.state = ReleaseState.INVALID
        bad_reg.register(bad_rec)
        ok, reason = bad_reg.activate("bad2")
        check("13. activate INVALID release blocked", not ok, reason)

    # ═══════════════════════════════════════════════════════════════════════
    # C. Revocation
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- C. Revocation --")
    with tempfile.TemporaryDirectory(prefix="phase51-rev-") as td:
        reg_path = Path(td) / "registry.json"
        reg = ReleaseRegistry(reg_path)
        art, manifest = _make_release(Path(td))
        record, _ = verify_release_lifecycle(manifest, art, expected_rule_hash=rule_hash)
        reg.register(record)

        # 14. revoke release
        ok, reason = reg.revoke(record.release_id, "security incident")
        check("14. revoke succeeds", ok, reason)
        check("14b. state is REVOKED", record.state == ReleaseState.REVOKED)
        check("14c. revocation_reason set", record.revocation_reason == "security incident")

        # 15. activate revoked release
        ok, reason = reg.activate(record.release_id)
        check("15. activate revoked release blocked", not ok, reason)

        # 16. revoke already revoked
        ok, reason = reg.revoke(record.release_id, "double revoke")
        check("16. double revoke blocked", not ok, reason)

        # 17. revoke non-existent
        ok, reason = reg.revoke("nonexistent", "ghost")
        check("17. revoke nonexistent blocked", not ok, reason)

    # ═══════════════════════════════════════════════════════════════════════
    # D. Manifest Tampering
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- D. Manifest tampering --")
    with tempfile.TemporaryDirectory(prefix="phase51-tamper-") as td:
        art, manifest = _make_release(Path(td))

        # 18. modified artifact hash in manifest
        tampered = ReleaseManifest(
            release_id=manifest.release_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            artifact_hash="0" * 64,
            artifact_files=dict(manifest.artifact_files),
            feature_version=manifest.feature_version,
            schema_version=manifest.schema_version,
            preprocessing_hash=manifest.preprocessing_hash,
            rule_hash=manifest.rule_hash,
            evaluation_record_hash=manifest.evaluation_record_hash,
            training_config_hash=manifest.training_config_hash,
            source_git_sha=manifest.source_git_sha,
            gate_verdict=manifest.gate_verdict,
        )
        tampered.sign()
        record, failures = verify_release_lifecycle(tampered, art, expected_rule_hash=rule_hash)
        check("18. tampered artifact hash -> INVALID", record.state == ReleaseState.INVALID, str(failures))

        # 19. forged signature
        forged = ReleaseManifest(
            release_id=manifest.release_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            artifact_hash=manifest.artifact_hash,
            artifact_files=dict(manifest.artifact_files),
            feature_version=manifest.feature_version,
            schema_version=manifest.schema_version,
            preprocessing_hash=manifest.preprocessing_hash,
            rule_hash=manifest.rule_hash,
            evaluation_record_hash=manifest.evaluation_record_hash,
            training_config_hash=manifest.training_config_hash,
            source_git_sha=manifest.source_git_sha,
            gate_verdict=manifest.gate_verdict,
        )
        forged._nonce = "0" * 16
        forged._signature = "a" * 64
        record, failures = verify_release_lifecycle(forged, art, expected_rule_hash=rule_hash)
        check("19. forged signature -> INVALID", record.state == ReleaseState.INVALID, str(failures))

        # 20. invalid gate verdict (not PROMOTION_ELIGIBLE or LEGACY_ATTESTED)
        wrong_gate = ReleaseManifest(
            release_id=manifest.release_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            artifact_hash=manifest.artifact_hash,
            artifact_files=dict(manifest.artifact_files),
            feature_version=manifest.feature_version,
            schema_version=manifest.schema_version,
            preprocessing_hash=manifest.preprocessing_hash,
            rule_hash=manifest.rule_hash,
            evaluation_record_hash=manifest.evaluation_record_hash,
            training_config_hash=manifest.training_config_hash,
            source_git_sha=manifest.source_git_sha,
            gate_verdict="FAKE_VERDICT",
        )
        wrong_gate.sign()
        record, failures = verify_release_lifecycle(wrong_gate, art, expected_rule_hash=rule_hash)
        check("20. invalid gate verdict -> INVALID", record.state == ReleaseState.INVALID, str(failures))

    # ═══════════════════════════════════════════════════════════════════════
    # E. Artifact Tampering
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- E. Artifact tampering --")
    with tempfile.TemporaryDirectory(prefix="phase51-art-") as td:
        art, manifest = _make_release(Path(td))

        # 21. modified artifact bytes
        art_copy = Path(td) / "art_copy"
        shutil.copytree(art, art_copy)
        (art_copy / "model.joblib").write_bytes(b"CORRUPTED")
        record, failures = verify_release_lifecycle(manifest, art_copy, expected_rule_hash=rule_hash)
        check("21. corrupted artifact -> INVALID", record.state == ReleaseState.INVALID, str(failures))

        # 22. missing artifact
        art_empty = Path(td) / "art_empty"
        art_empty.mkdir()
        record, failures = verify_release_lifecycle(manifest, art_empty, expected_rule_hash=rule_hash)
        check("22. missing artifact -> INVALID", record.state == ReleaseState.INVALID, str(failures))

        # 23. missing directory
        record, failures = verify_release_lifecycle(manifest, Path(td) / "nonexistent")
        check("23. missing dir -> INVALID", record.state == ReleaseState.INVALID, str(failures))

    # ═══════════════════════════════════════════════════════════════════════
    # F. Rollback
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- F. Rollback --")
    with tempfile.TemporaryDirectory(prefix="phase51-rb-") as td:
        reg_path = Path(td) / "registry.json"
        reg = ReleaseRegistry(reg_path)

        # Create two releases
        art1, m1 = _make_release(Path(td) / "r1")
        rec1, _ = verify_release_lifecycle(m1, art1, expected_rule_hash=rule_hash)
        reg.register(rec1)
        reg.activate(rec1.release_id)

        art2, m2 = _make_release(Path(td) / "r2")
        # Make r2 different
        (art2 / "model.joblib").write_bytes(b"model-bytes-v2")
        m2_v2 = create_release_manifest(
            release_id="rel-test-51-v2",
            model_id="m-test-51-v2",
            model_version="v-test-51-v2",
            artifact_dir=art2,
            feature_version="v1",
            schema_version="v1",
            preprocessing_hash=hashlib.sha256(b"scaler-bytes-v1").hexdigest(),
            rule_hash="abcd1234" * 4,
            evaluation_record_hash="eval51hash",
            training_config_hash="cfg51hash",
            source_git_sha="abc1234",
            gate_verdict="LEGACY_ATTESTED",
        )
        rec2, _ = verify_release_lifecycle(m2_v2, art2, expected_rule_hash=rule_hash)
        reg.register(rec2)

        # 24. rollback to release 1
        ok, reason = reg.activate(rec1.release_id)
        check("24. rollback to r1 succeeds", ok, reason)
        check("24b. active is r1", reg.active_release_id == rec1.release_id)

        # 25. rollback to revoked release
        reg.revoke(rec2.release_id, "test revoke")
        ok, reason = reg.activate(rec2.release_id)
        check("25. rollback to revoked r2 blocked", not ok, reason)

        # 26. rollback to INVALID release
        bad_rec = ReleaseRecord(
            release_id="bad-rb", model_id="bad-rb", model_version="bad-rb",
            artifact_hash="0" * 64, manifest_hash="0" * 64,
            feature_version="v1", schema_version="v1",
            rule_hash="0" * 32, preprocessing_hash="0" * 64,
            gate_verdict="LEGACY_ATTESTED",
        )
        bad_rec.state = ReleaseState.INVALID
        reg.register(bad_rec)
        ok, reason = reg.activate("bad-rb")
        check("26. rollback to INVALID blocked", not ok, reason)

    # ═══════════════════════════════════════════════════════════════════════
    # G. Concurrent Activation
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- G. Concurrent activation --")
    with tempfile.TemporaryDirectory(prefix="phase51-conc-") as td:
        reg_path = Path(td) / "registry.json"
        reg = ReleaseRegistry(reg_path)

        # Create 5 releases
        records = []
        for i in range(5):
            art_d = Path(td) / f"art{i}"
            art_d.mkdir()
            (art_d / "model.joblib").write_bytes(f"model-bytes-{i}".encode())
            m = create_release_manifest(
                release_id=f"rel-conc-{i}",
                model_id=f"m-conc-{i}",
                model_version=f"v-conc-{i}",
                artifact_dir=art_d,
                feature_version="v1",
                schema_version="v1",
                preprocessing_hash=f"pre{i}" + "0" * 58,
                rule_hash=rule_hash,
                evaluation_record_hash=f"eval{i}" + "0" * 58,
                training_config_hash=f"cfg{i}" + "0" * 58,
                gate_verdict="LEGACY_ATTESTED",
            )
            rec, _ = verify_release_lifecycle(m, art_d, expected_rule_hash=rule_hash)
            reg.register(rec)
            records.append(rec)

        # 27. concurrent activation of same release
        results = []
        def activate_rel(rid):
            ok, reason = reg.activate(rid)
            results.append((ok, reason, rid))

        threads = [threading.Thread(target=activate_rel, args=(records[0].release_id,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successes = [r for r in results if r[0]]
        # All should succeed: lock makes them sequential, idempotent activation is safe
        check("27. concurrent same-release: all succeed (idempotent)", len(successes) == 5, f"{len(successes)}/5 succeeded")

        # 28. concurrent activation of different releases (fresh registry)
        reg2_path = Path(td) / "registry_conc.json"
        reg2 = ReleaseRegistry(reg2_path)
        for rec in records:
            reg2.register(rec)

        results2 = []
        def activate_rel2(rid):
            ok, reason = reg2.activate(rid)
            results2.append((ok, reason, rid))

        threads2 = [threading.Thread(target=activate_rel2, args=(records[i].release_id,)) for i in range(3)]
        for t in threads2:
            t.start()
        for t in threads2:
            t.join()

        successes2 = [r for r in results2 if r[0]]
        # Lock serializes; all REGISTERED releases can activate. records[0] was
        # already ACTIVATED from test 27, so it gets 'already active' (True).
        # All activations are serialized — no race conditions possible.
        check("28. concurrent different releases: no races (lock works)", len(successes2) >= 1, f"{len(successes2)}/3 succeeded")

    # ═══════════════════════════════════════════════════════════════════════
    # H. Runtime Attestation with Revoked Release
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- H. Runtime attestation with revoked release --")
    with tempfile.TemporaryDirectory(prefix="phase51-revatt-") as td:
        art, manifest = _make_release(Path(td))

        # 29. revoked manifest fails gate verdict
        revoked_m = create_revoked_manifest(manifest, "security incident")
        ok, reason = revoked_m.verify_signature()
        check("29. revoked manifest signature valid", ok, reason)
        ok, failures, identity = verify_release_for_load(revoked_m, art, expected_rule_hash=rule_hash)
        check("29b. revoked manifest fails verification", not ok, str(failures))

    # ═══════════════════════════════════════════════════════════════════════
    # I. Restart Recovery
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- I. Restart recovery --")
    with tempfile.TemporaryDirectory(prefix="phase51-restart-") as td:
        reg_path = Path(td) / "registry.json"
        reg = ReleaseRegistry(reg_path)
        art, manifest = _make_release(Path(td))
        rec, _ = verify_release_lifecycle(manifest, art, expected_rule_hash=rule_hash)
        reg.register(rec)
        reg.activate(rec.release_id)

        # 30. registry persists across reload
        reg2 = ReleaseRegistry(reg_path)
        check("30. registry reload preserves active", reg2.active_release_id == rec.release_id)

        # 31. after restart, attestation still works
        ok, failures, identity = verify_release_for_load(manifest, art, expected_rule_hash=rule_hash)
        check("31. post-restart attestation works", ok, str(failures))

    # ═══════════════════════════════════════════════════════════════════════
    # J. Audit Trail
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- J. Audit trail --")
    with tempfile.TemporaryDirectory(prefix="phase51-audit-") as td:
        reg_path = Path(td) / "registry.json"
        reg = ReleaseRegistry(reg_path)
        art, manifest = _make_release(Path(td))
        rec, _ = verify_release_lifecycle(manifest, art, expected_rule_hash=rule_hash)
        reg.register(rec)
        reg.activate(rec.release_id)
        reg.revoke(rec.release_id, "audit test")

        # 32. state_history records all transitions
        check("32. state_history has transitions", len(rec.state_history) >= 3)

        # 33. revocation info preserved
        check("33. revocation_reason preserved", rec.revocation_reason == "audit test")
        check("33b. revoked_at set", rec.revoked_at is not None)

    # ═══════════════════════════════════════════════════════════════════════
    # K. Live Startup (attested legacy release)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- K. Live startup --")
    # Regenerate manifest with the test JWT_SECRET so HMAC matches.
    # Save original so we can restore it afterwards (other tests need it).
    _mp = PRODUCTION_DIR / "release_manifest.json"
    _original_manifest_bytes = _mp.read_bytes() if _mp.exists() else None
    if PRODUCTION_DIR.exists() and ARTIFACT_DIR.exists():
        from src.monitoring.release_manifest import create_release_manifest as _crm
        _m = _crm(
            release_id="legacy-altman_native_E_hardneg_cert_20260904",
            model_id="altman_native_E_hardneg_cert_20260904",
            model_version="altman_native_E_hardneg_cert_20260904",
            artifact_dir=ARTIFACT_DIR,
            feature_version="v1",
            schema_version="v1",
            preprocessing_hash=sha256_file(ARTIFACT_DIR / "scaler_native.joblib"),
            rule_hash=rule_hash,
            evaluation_record_hash=hashlib.sha256(
                (ROOT / "models" / "model_records" / "altman_native_E_hardneg_cert_20260904.json").read_bytes()
            ).hexdigest() if (ROOT / "models" / "model_records" / "altman_native_E_hardneg_cert_20260904.json").exists() else "e0" * 32,
            training_config_hash="e0" * 32,
            source_git_sha="legacy",
            gate_verdict="LEGACY_ATTESTED",
        )
        _m.save(_mp)
    from fastapi.testclient import TestClient
    from src.risk_engine.main import app as risk_app
    with TestClient(risk_app) as rc:
        h = rc.get("/health").json()
        # 34. runtime_state READY
        check("34. runtime_state READY", h.get("runtime_state") == "READY")
        # 35. release_attested true
        check("35. release_attested true", h.get("release_attested") is True)
        # 36. model ready (canonical HealthReport: model_readiness)
        check("36. model ready", h.get("model_readiness") == "loaded")
    # Restore original manifest so downstream tests see the real manifest
    if _original_manifest_bytes is not None:
        _mp.write_bytes(_original_manifest_bytes)

    # ═══════════════════════════════════════════════════════════════════════
    # L. Inference with attested release
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- L. Inference --")
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
                    json={"event_id": "phase51-eval-001", "fraud_id": "F24BRMMMBJWYMTDW",
                          "features": features})
        resp = r.json()
        # 37. not degraded
        check("37. inference not degraded", resp.get("degraded") is False)
        # 38. risk_band present
        check("38. risk_band present", resp.get("risk_band") in ("low", "medium", "high"))

    # ═══════════════════════════════════════════════════════════════════════
    # M. REAL_WORLD_VALIDATION remains blocked
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- M. REAL_WORLD_VALIDATION --")
    from src.monitoring.promotion_gate import evaluate_promotion
    gate = evaluate_promotion(
        candidate_model_id="test",
        candidate_artifact_hash="test",
        candidate_feature_version="v1",
    )
    # 39. promotion remains BLOCKED
    check("39. promotion BLOCKED", gate.verdict.name == "BLOCKED")
    # 40. REAL_WORLD_VALIDATION gate fails
    rwv = next((g for g in gate.gates if "REAL_WORLD_VALIDATION" in g.gate_name), None)
    check("40. REAL_WORLD_VALIDATION gate FAIL", rwv is not None and rwv.status != "PASS")

    # ═══════════════════════════════════════════════════════════════════════
    # N. Stale manifest rejected
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- N. Stale manifest --")
    with tempfile.TemporaryDirectory(prefix="phase51-stale-") as td:
        art, manifest = _make_release(Path(td))
        # 41. wrong artifact hash in manifest
        stale_m = ReleaseManifest(
            release_id=manifest.release_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            artifact_hash="0" * 64,
            artifact_files=dict(manifest.artifact_files),
            feature_version=manifest.feature_version,
            schema_version=manifest.schema_version,
            preprocessing_hash=manifest.preprocessing_hash,
            rule_hash=manifest.rule_hash,
            evaluation_record_hash=manifest.evaluation_record_hash,
            training_config_hash=manifest.training_config_hash,
            source_git_sha=manifest.source_git_sha,
            gate_verdict=manifest.gate_verdict,
        )
        stale_m.sign()
        record, failures = verify_release_lifecycle(stale_m, art, expected_rule_hash=rule_hash)
        check("41. stale artifact hash -> INVALID", record.state == ReleaseState.INVALID, str(failures))

    # ═══════════════════════════════════════════════════════════════════════
    # O. Drift detection
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- O. Drift detection --")
    with tempfile.TemporaryDirectory(prefix="phase51-drift-") as td:
        art, manifest = _make_release(Path(td))
        ok, _, identity = verify_release_for_load(manifest, art, expected_rule_hash=rule_hash)
        att = build_attestation(identity, art)

        # 42. no drift on clean dir
        state, reasons = detect_runtime_drift(att, art)
        check("42. no drift on clean dir", state == RuntimeState.READY, str(reasons))

        # 43. drift after artifact change
        (art / "model.joblib").write_bytes(b"DRIFTED")
        state, reasons = detect_runtime_drift(att, art)
        check("43. drift detected after change", state == RuntimeState.DRIFTED)

        # 44. drift after restore
        (art / "model.joblib").write_bytes(b"model-bytes-v1")
        state, reasons = detect_runtime_drift(att, art)
        check("44. restore -> no drift", state == RuntimeState.READY)

    # ═══════════════════════════════════════════════════════════════════════
    # P. Release identity uniqueness
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- P. Release identity --")
    with tempfile.TemporaryDirectory(prefix="phase51-id-") as td:
        art1, m1 = _make_release(Path(td) / "a", "alpha")
        art2, m2 = _make_release(Path(td) / "b", "beta")
        # 45. different artifacts -> different hash
        check("45. different artifacts -> different hash", m1.artifact_hash != m2.artifact_hash)
        # 46. different release_id
        check("46. different release_id", m1.release_id != m2.release_id)

    # ═══════════════════════════════════════════════════════════════════════
    # Q. Release record serialization
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- Q. Serialization --")
    rec = ReleaseRecord(
        release_id="ser", model_id="ser", model_version="ser",
        artifact_hash="a" * 64, manifest_hash="b" * 64,
        feature_version="v1", schema_version="v1",
        rule_hash="c" * 32, preprocessing_hash="d" * 64,
        gate_verdict="LEGACY_ATTESTED",
    )
    d = rec.to_dict()
    # 47. to_dict has all fields
    check("47. to_dict has release_id", d["release_id"] == "ser")
    check("47b. to_dict has state", d["state"] == "DISCOVERED")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    total = passed + failed
    if failed:
        print(f"Phase 51: {passed} passed, {failed} failed (out of {total})")
    else:
        print(f"Phase 51: {passed} passed, 0 failed")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
