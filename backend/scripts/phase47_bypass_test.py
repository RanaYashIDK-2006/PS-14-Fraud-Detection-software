#!/usr/bin/env python3
"""Phase 47-48: Promotion bypass-proof activation tests.

Proves that NO caller — direct, None, fake, stale, or concurrent —
can activate an ineligible model through ModelRegistry.promote().

Phase 48 adds: release_manifest is now also mandatory.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("JWT_SECRET", "phase47-test-secret-0123456789abcdef")

from src.monitoring.promotion_gate import (
    GateStatus,
    PromotionVerdict,
    PromotionBlockedError,
    PromotionToken,
    PromotionDecision,
    GateResult,
    evaluate_promotion,
    evaluate_real_world_validation,
)
from src.monitoring.release_manifest import (
    ReleaseManifest,
    create_release_manifest,
    artifact_set_hash,
)
from src.risk_engine.model_registry import ModelRegistry, ModelCandidate

passed = 0
failed = 0
total = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")
    return condition


def _make_registry(tmpdir: str) -> tuple[ModelRegistry, Path, Path]:
    """Create a registry + fake candidate model + metadata."""
    td = Path(tmpdir)
    reg = ModelRegistry(td)
    cand = td / "cand.joblib"
    cand.write_bytes(b"fake model bytes for testing")
    meta = td / "cand_meta.json"
    meta.write_text('{"version": "test_v1"}')
    reg.register_candidate(str(cand), str(meta))
    return reg, cand, meta


def _aggregate_hash(path: Path) -> str:
    """Compute aggregate artifact hash (same as ReleaseManifest)."""
    art = artifact_set_hash(path.parent)
    return art["model_hash"] or ""


def _make_manifest(tmpdir: str, cand_path: Path, **overrides) -> ReleaseManifest:
    """Create a signed release manifest for testing."""
    defaults = dict(
        release_id="test_release_001",
        model_id=str(cand_path),
        model_version="test_v1",
        feature_version="v1",
        schema_version="v1",
        gate_verdict="PROMOTION_ELIGIBLE",
        gate_timestamp=time.time(),
    )
    defaults.update(overrides)
    return create_release_manifest(
        artifact_dir=cand_path.parent,
        **defaults,
    )


def _fake_token(model_id: str, artifact_hash: str, feature_version: str) -> PromotionToken:
    """Construct a PromotionToken directly (bypassing evaluate_promotion)."""
    t = PromotionToken.__new__(PromotionToken)
    t.model_id = model_id
    t.artifact_hash = artifact_hash
    t.feature_version = feature_version
    t.timestamp = time.time()
    t._nonce = "fake_nonce_0000"
    t._signature = "fake_signature_" * 4  # invalid signature
    return t


def _eligible_decision(model_id: str = "", artifact_hash: str = "",
                       feature_version: str = "") -> PromotionDecision:
    """Create a PromotionDecision with ELIGIBLE verdict (hand-constructed)."""
    return PromotionDecision(
        verdict=PromotionVerdict.ELIGIBLE,
        candidate_model_id=model_id,
        candidate_artifact_hash=artifact_hash,
        candidate_feature_version=feature_version,
    )


def _blocked_decision() -> PromotionDecision:
    return PromotionDecision(verdict=PromotionVerdict.BLOCKED)


# ═══════════════════════════════════════════════════════════════════════════
# 1. promote(None) MUST be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 1. promote(None) must be blocked ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    raised = False
    try:
        reg.promote(gate_decision=None)
    except RuntimeError as e:
        raised = True
        check("1a. RuntimeError raised", "required" in str(e).lower() or "blocked" in str(e).lower())
    check("1b. promotion did NOT happen", reg.state.candidate is not None)
    check("1c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 2. promote() with no args must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 2. promote() with no args must be blocked ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    raised = False
    try:
        reg.promote()  # no gate_decision
    except RuntimeError:
        raised = True
    check("2a. RuntimeError raised on promote()", raised)
    check("2b. candidate not promoted", reg.state.candidate is not None)


# ═══════════════════════════════════════════════════════════════════════════
# 3. promote with BLOCKED decision must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 3. promote with BLOCKED decision ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    raised = False
    try:
        reg.promote(gate_decision=_blocked_decision())
    except RuntimeError:
        raised = True
    check("3a. RuntimeError on BLOCKED", raised)
    check("3b. candidate not promoted", reg.state.candidate is not None)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Hand-constructed ELIGIBLE decision without token must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 4. Hand-constructed ELIGIBLE without token ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    fake_d = _eligible_decision()
    raised = False
    try:
        reg.promote(gate_decision=fake_d)  # no token!
    except RuntimeError as e:
        raised = True
        check("4a. RuntimeError on missing token", "token" in str(e).lower())
    check("4b. candidate not promoted", reg.state.candidate is not None)
    check("4c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Fake token with invalid signature must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 5. Fake token with invalid signature ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    fake_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    token = _fake_token(str(cand), cand_hash, "v1")
    manifest = _make_manifest(tmpdir, cand)
    raised = False
    try:
        reg.promote(gate_decision=fake_d, promotion_token=token,
                    release_manifest=manifest)
    except RuntimeError as e:
        raised = True
        check("5a. RuntimeError on invalid signature",
              "signature" in str(e).lower() or "forged" in str(e).lower() or "token" in str(e).lower())
    check("5b. candidate not promoted", reg.state.candidate is not None)
    check("5c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Valid token + ELIGIBLE decision WITHOUT manifest must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 6. Valid token + ELIGIBLE without manifest ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    good_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    token = PromotionToken(good_d)
    raised = False
    try:
        reg.promote(gate_decision=good_d, promotion_token=token)
    except RuntimeError as e:
        raised = True
        check("6a. RuntimeError on missing manifest", "manifest" in str(e).lower())
    check("6b. candidate not promoted", reg.state.candidate is not None)
    check("6c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Fake manifest with invalid signature must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 7. Fake manifest with invalid signature ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    good_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    token = PromotionToken(good_d)
    # Create a manifest but corrupt its signature
    manifest = _make_manifest(tmpdir, cand)
    manifest._signature = "tampered_signature"
    raised = False
    try:
        reg.promote(gate_decision=good_d, promotion_token=token,
                    release_manifest=manifest)
    except RuntimeError as e:
        raised = True
        check("7a. RuntimeError on tampered manifest",
              "signature" in str(e).lower() or "manifest" in str(e).lower())
    check("7b. candidate not promoted", reg.state.candidate is not None)
    check("7c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Manifest with wrong gate verdict must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 8. Manifest with wrong gate verdict ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    good_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    token = PromotionToken(good_d)
    manifest = _make_manifest(tmpdir, cand, gate_verdict="PROMOTION_BLOCKED")
    raised = False
    try:
        reg.promote(gate_decision=good_d, promotion_token=token,
                    release_manifest=manifest)
    except RuntimeError as e:
        raised = True
        check("8a. RuntimeError on wrong gate verdict", "verdict" in str(e).lower())
    check("8b. candidate not promoted", reg.state.candidate is not None)
    check("8c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Non-ReleaseManifest object must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 9. Non-ReleaseManifest object ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    good_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    token = PromotionToken(good_d)
    raised = False
    try:
        reg.promote(gate_decision=good_d, promotion_token=token,
                    release_manifest="not_a_manifest")
    except RuntimeError as e:
        raised = True
        check("9a. RuntimeError on fake manifest object",
              "manifest" in str(e).lower())
    check("9b. candidate not promoted", reg.state.candidate is not None)
    check("9c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Stale token must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 10. Stale token (expired) ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    good_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    token = PromotionToken(good_d)
    token.timestamp = time.time() - 7200  # 2 hours ago
    manifest = _make_manifest(tmpdir, cand)
    raised = False
    try:
        reg.promote(gate_decision=good_d, promotion_token=token,
                    release_manifest=manifest)
    except RuntimeError as e:
        raised = True
        check("10a. RuntimeError on stale token",
              "expired" in str(e).lower() or "signature" in str(e).lower() or "token" in str(e).lower())
    check("10b. candidate not promoted", reg.state.candidate is not None)
    check("10c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Token for wrong model must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 11. Token for wrong model ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    good_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    # Token for a different model
    other_d = _eligible_decision(
        model_id="/nonexistent/other.joblib", artifact_hash=cand_hash,
        feature_version="v1"
    )
    token = PromotionToken(other_d)
    manifest = _make_manifest(tmpdir, cand)
    raised = False
    try:
        reg.promote(gate_decision=good_d, promotion_token=token,
                    release_manifest=manifest)
    except RuntimeError as e:
        raised = True
        check("11a. RuntimeError on wrong model token",
              "mismatch" in str(e).lower() or "token" in str(e).lower())
    check("11b. candidate not promoted", reg.state.candidate is not None)
    check("11c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 12. Token for wrong artifact must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 12. Token for wrong artifact ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    good_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    token = PromotionToken(good_d)
    # Modify the artifact after token creation
    cand.write_bytes(b"MODIFIED model bytes")
    manifest = _make_manifest(tmpdir, cand)
    raised = False
    try:
        reg.promote(gate_decision=good_d, promotion_token=token,
                    release_manifest=manifest)
    except RuntimeError as e:
        raised = True
        check("12a. RuntimeError on artifact mismatch",
              "mismatch" in str(e).lower() or "artifact" in str(e).lower() or "manifest" in str(e).lower())
    check("12b. candidate not promoted", reg.state.candidate is not None)
    check("12c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 13. Manifest artifact mismatch must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 13. Manifest artifact mismatch ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    good_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    token = PromotionToken(good_d)
    # Create manifest, then modify artifact
    manifest = _make_manifest(tmpdir, cand)
    cand.write_bytes(b"TAMPERED after manifest creation")
    raised = False
    try:
        reg.promote(gate_decision=good_d, promotion_token=token,
                    release_manifest=manifest)
    except RuntimeError as e:
        raised = True
        check("13a. RuntimeError on manifest artifact mismatch",
              "mismatch" in str(e).lower() or "manifest" in str(e).lower() or "binding" in str(e).lower())
    check("13b. candidate not promoted", reg.state.candidate is not None)
    check("13c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 14. Valid token + ELIGIBLE + valid manifest = promotion succeeds
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 14. Valid token + ELIGIBLE + valid manifest ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    good_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    token = PromotionToken(good_d)
    manifest = _make_manifest(tmpdir, cand)
    raised = False
    try:
        reg.promote(gate_decision=good_d, promotion_token=token,
                    release_manifest=manifest)
    except Exception as e:
        raised = True
        print(f"    unexpected error: {e}")
    check("14a. promotion succeeded", not raised)
    check("14b. candidate is now None (promoted)", reg.state.candidate is None)
    check("14c. mode is direct", reg.state.mode == "direct")


# ═══════════════════════════════════════════════════════════════════════════
# 15. promote() with no candidate
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 15. promote() with no candidate ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg = ModelRegistry(Path(tmpdir))
    raised = False
    try:
        reg.promote(gate_decision=_eligible_decision())
    except Exception as e:
        raised = True
    check("15a. no error when no candidate (just returns)", not raised)
    check("15b. mode unchanged", reg.state.mode == "direct")


# ═══════════════════════════════════════════════════════════════════════════
# 16. promote_to_canary does NOT make model active
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 16. promote_to_canary does NOT activate ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    reg.promote_to_canary(0.5)
    check("16a. mode is canary", reg.state.mode == "canary")
    check("16b. candidate still exists (not promoted)", reg.state.candidate is not None)
    check("16c. candidate status is canary", reg.state.candidate.status == "canary")


# ═══════════════════════════════════════════════════════════════════════════
# 17. Rollback does NOT require gate
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 17. Rollback works without gate ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    result = reg.rollback()
    check("17a. rollback returned True", result is True)
    check("17b. candidate status is rolled_back",
          reg.state.candidate.status == "rolled_back")
    check("17c. mode is direct", reg.state.mode == "direct")


# ═══════════════════════════════════════════════════════════════════════════
# 18. PromotionToken rejects BLOCKED decision
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 18. PromotionToken rejects BLOCKED decision ---")
try:
    t = PromotionToken(_blocked_decision())
    check("18a. should have raised", False)
except PromotionBlockedError:
    check("18a. PromotionBlockedError raised", True)


# ═══════════════════════════════════════════════════════════════════════════
# 19. Token serialization roundtrip preserves validity
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 19. Token serialization roundtrip ---")
good_d = _eligible_decision(
    model_id="test_model", artifact_hash="abc123", feature_version="v1"
)
token = PromotionToken(good_d)
data = token.to_dict()
token2 = PromotionToken.from_dict(data)
ok, _ = token2.verify("test_model", "abc123", "v1")
check("19a. roundtrip preserves validity", ok)


# ═══════════════════════════════════════════════════════════════════════════
# 20. ReleaseManifest: valid manifest creation + signing + verification
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 20. ReleaseManifest: create + sign + verify ---")
with tempfile.TemporaryDirectory() as tmpdir:
    td = Path(tmpdir)
    (td / "model.joblib").write_bytes(b"model bytes")
    manifest = create_release_manifest(
        release_id="rel_001",
        model_id="test_model",
        model_version="v1",
        artifact_dir=td,
        feature_version="f_v1",
        gate_verdict="PROMOTION_ELIGIBLE",
    )
    ok, reason = manifest.verify_signature()
    check("20a. signature valid", ok)
    check("20b. has artifact_hash", bool(manifest.artifact_hash))
    check("20c. gate_verdict is ELIGIBLE", manifest.gate_verdict == "PROMOTION_ELIGIBLE")


# ═══════════════════════════════════════════════════════════════════════════
# 21. ReleaseManifest: tampered manifest detected
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 21. ReleaseManifest: tampered manifest detected ---")
with tempfile.TemporaryDirectory() as tmpdir:
    td = Path(tmpdir)
    (td / "model.joblib").write_bytes(b"model bytes")
    manifest = create_release_manifest(
        release_id="rel_002",
        model_id="test_model",
        model_version="v1",
        artifact_dir=td,
        feature_version="f_v1",
        gate_verdict="PROMOTION_ELIGIBLE",
    )
    # Tamper with the manifest
    manifest.artifact_hash = "tampered_hash"
    ok, reason = manifest.verify_signature()
    check("21a. tampered manifest detected", not ok)
    check("21b. reason mentions tampered/forged",
          any(w in reason.lower() for w in ["tampered", "forged", "invalid", "signature"]))


# ═══════════════════════════════════════════════════════════════════════════
# 22. ReleaseManifest: artifact verification
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 22. ReleaseManifest: artifact verification ---")
with tempfile.TemporaryDirectory() as tmpdir:
    td = Path(tmpdir)
    (td / "model.joblib").write_bytes(b"model bytes")
    manifest = create_release_manifest(
        release_id="rel_003",
        model_id="test_model",
        model_version="v1",
        artifact_dir=td,
        feature_version="f_v1",
        gate_verdict="PROMOTION_ELIGIBLE",
    )
    # Verify against same directory
    ok, reason = manifest.verify_artifacts(td)
    check("22a. artifacts verify OK", ok)
    # Modify artifact
    (td / "model.joblib").write_bytes(b"TAMPERED")
    ok2, reason2 = manifest.verify_artifacts(td)
    check("22b. tampered artifact detected", not ok2)
    check("22c. reason mentions mismatch", "mismatch" in reason2.lower() or "changed" in reason2.lower())


# ═══════════════════════════════════════════════════════════════════════════
# 23. ReleaseManifest: binding verification
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 23. ReleaseManifest: binding verification ---")
with tempfile.TemporaryDirectory() as tmpdir:
    td = Path(tmpdir)
    (td / "model.joblib").write_bytes(b"model bytes")
    manifest = create_release_manifest(
        release_id="rel_004",
        model_id="test_model",
        model_version="v1",
        artifact_dir=td,
        feature_version="f_v1",
        gate_verdict="PROMOTION_ELIGIBLE",
    )
    ok, _ = manifest.verify_binding(expected_model_id="test_model")
    check("23a. binding OK with correct model_id", ok)
    ok2, reason2 = manifest.verify_binding(expected_model_id="wrong_model")
    check("23b. binding fails with wrong model_id", not ok2)
    check("23c. reason mentions mismatch", "mismatch" in reason2.lower())


# ═══════════════════════════════════════════════════════════════════════════
# 24. ReleaseManifest: serialization roundtrip
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 24. ReleaseManifest: serialization roundtrip ---")
with tempfile.TemporaryDirectory() as tmpdir:
    td = Path(tmpdir)
    (td / "model.joblib").write_bytes(b"model bytes")
    manifest = create_release_manifest(
        release_id="rel_005",
        model_id="test_model",
        model_version="v1",
        artifact_dir=td,
        feature_version="f_v1",
        gate_verdict="PROMOTION_ELIGIBLE",
    )
    data = manifest.to_dict()
    manifest2 = ReleaseManifest.from_dict(data)
    ok, _ = manifest2.verify_signature()
    check("24a. roundtrip preserves signature", ok)
    check("24b. roundtrip preserves model_id", manifest2.model_id == "test_model")
    check("24c. roundtrip preserves artifact_hash", manifest2.artifact_hash == manifest.artifact_hash)


# ═══════════════════════════════════════════════════════════════════════════
# 25. REAL_WORLD_VALIDATION blocks centralized gate
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 25. REAL_WORLD_VALIDATION blocks centralized gate ---")
g = evaluate_real_world_validation()
check("25a. RWV is BLOCKED", g.status == GateStatus.BLOCKED)
d = evaluate_promotion(candidate_model_id="test")
check("25b. full gate returns BLOCKED", d.verdict == PromotionVerdict.BLOCKED)
check("25c. RWV in blocking gates", "REAL_WORLD_VALIDATION" in d.blocking_gates)


# ═══════════════════════════════════════════════════════════════════════════
# 26. Concurrent promotion attempts
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 26. Concurrent promotion attempts ---")
import shutil as _shutil
with tempfile.TemporaryDirectory() as tmpdir:
    shared_dir = Path(tmpdir) / "shared"
    shared_dir.mkdir()
    (shared_dir / "cand.joblib").write_bytes(b"shared model bytes")
    (shared_dir / "cand_meta.json").write_text('{"version": "v1"}')
    reg = ModelRegistry(shared_dir)
    reg.register_candidate(str(shared_dir / "cand.joblib"),
                          str(shared_dir / "cand_meta.json"))
    cand_hash = _aggregate_hash(shared_dir / "cand.joblib")
    good_d = _eligible_decision(
        model_id=str(shared_dir / "cand.joblib"),
        artifact_hash=cand_hash,
        feature_version="v1",
    )
    errors = []

    def try_promote():
        try:
            t = PromotionToken(good_d)
            m = create_release_manifest(
                release_id="concurrent_test",
                model_id=str(shared_dir / "cand.joblib"),
                model_version="v1",
                artifact_dir=shared_dir,
                feature_version="v1",
                gate_verdict="PROMOTION_ELIGIBLE",
            )
            reg.promote(gate_decision=good_d, promotion_token=t,
                        release_manifest=m)
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=try_promote) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check("26a. candidate is None after concurrent promotion",
          reg.state.candidate is None)
    check("26b. mode is direct", reg.state.mode == "direct")
    check("26c. no exceptions from any thread", len(errors) == 0)


# ═══════════════════════════════════════════════════════════════════════════
# 27. canary -> promote must pass gate
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 27. canary -> promote must pass gate ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    reg.promote_to_canary(0.1)
    check("27a. in canary mode", reg.state.mode == "canary")
    raised = False
    try:
        reg.promote()  # no gate — must fail
    except RuntimeError:
        raised = True
    check("27b. promote() without gate fails", raised)
    check("27c. still in canary", reg.state.mode == "canary")


# ═══════════════════════════════════════════════════════════════════════════
# 28. Manifest with artifact hash mismatch must be blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 28. Manifest with artifact hash mismatch ---")
with tempfile.TemporaryDirectory() as tmpdir:
    reg, cand, meta = _make_registry(tmpdir)
    cand_hash = _aggregate_hash(cand)
    good_d = _eligible_decision(
        model_id=str(cand), artifact_hash=cand_hash, feature_version="v1"
    )
    token = PromotionToken(good_d)
    # Create manifest with WRONG artifact hash
    manifest = _make_manifest(tmpdir, cand)
    manifest.artifact_hash = "wrong_hash_00000000000000000000000000"
    # Re-sign after tampering so signature check passes but binding fails
    manifest.sign()
    raised = False
    try:
        reg.promote(gate_decision=good_d, promotion_token=token,
                    release_manifest=manifest)
    except RuntimeError as e:
        raised = True
        check("28a. RuntimeError on artifact hash mismatch",
              "mismatch" in str(e).lower() or "manifest" in str(e).lower() or "binding" in str(e).lower())
    check("28b. candidate not promoted", reg.state.candidate is not None)
    check("28c. exception was raised", raised)


# ═══════════════════════════════════════════════════════════════════════════
# 29. Full adversarial matrix: all invalid paths blocked
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 29. Full adversarial matrix ---")
all_blocked = True
test_cases = [
    ("None gate", lambda r, c, h: r.promote(gate_decision=None)),
    ("No args", lambda r, c, h: r.promote()),
    ("BLOCKED decision", lambda r, c, h: r.promote(gate_decision=_blocked_decision())),
    ("ELIGIBLE no token", lambda r, c, h: r.promote(
        gate_decision=_eligible_decision(str(c), h, "v1"))),
    ("ELIGIBLE + token no manifest", lambda r, c, h: r.promote(
        gate_decision=_eligible_decision(str(c), h, "v1"),
        promotion_token=PromotionToken(_eligible_decision(str(c), h, "v1")))),
    ("Fake token", lambda r, c, h: r.promote(
        gate_decision=_eligible_decision(str(c), h, "v1"),
        promotion_token=_fake_token(str(c), h, "v1"),
        release_manifest=_make_manifest(str(Path(c).parent), c))),
    ("Wrong model token", lambda r, c, h: r.promote(
        gate_decision=_eligible_decision(str(c), h, "v1"),
        promotion_token=PromotionToken(_eligible_decision(
            "/wrong/path", h, "v1")),
        release_manifest=_make_manifest(str(Path(c).parent), c))),
]
for label, fn in test_cases:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg, cand, meta = _make_registry(tmpdir)
        cand_hash = _aggregate_hash(cand)
        raised = False
        try:
            fn(reg, cand, cand_hash)
        except Exception:
            raised = True
        if not raised:
            all_blocked = False
            print(f"    [FAIL] {label} — promotion succeeded!")
        else:
            print(f"    [PASS] {label} — blocked")
check("29a. ALL adversarial paths blocked", all_blocked)


# ═══════════════════════════════════════════════════════════════════════════
# 30. Manifest determinism: same inputs => same hash
# ═══════════════════════════════════════════════════════════════════════════
print("\n--- 30. Manifest determinism ---")
with tempfile.TemporaryDirectory() as tmpdir:
    td = Path(tmpdir)
    (td / "model.joblib").write_bytes(b"model bytes")
    # Create manifests directly with fixed created_at for determinism
    from src.monitoring.release_manifest import ReleaseManifest, artifact_set_hash
    art = artifact_set_hash(td)
    def _det_manifest(**overrides):
        defaults = dict(
            release_id="det_test", model_id="m", model_version="v1",
            artifact_hash=art["model_hash"], artifact_files=art["files"],
            feature_version="f1", schema_version="f1",
            gate_verdict="PROMOTION_ELIGIBLE", gate_timestamp=1000000.0,
            created_at=1000000.0,
        )
        defaults.update(overrides)
        m = ReleaseManifest(**defaults)
        m.sign()
        return m
    m1 = _det_manifest()
    m2 = _det_manifest()
    check("30a. same inputs produce same manifest hash",
          m1.compute_manifest_hash() == m2.compute_manifest_hash())
    # Different artifact => different hash
    (td / "model.joblib").write_bytes(b"DIFFERENT model bytes")
    art2 = artifact_set_hash(td)
    m3 = _det_manifest(artifact_hash=art2["model_hash"], artifact_files=art2["files"])
    check("30b. different artifact produces different hash",
          m1.compute_manifest_hash() != m3.compute_manifest_hash())


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Phase 47-48 bypass-proof tests: {passed}/{total} passed")
if failed:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("  ALL PASSED")
    sys.exit(0)
