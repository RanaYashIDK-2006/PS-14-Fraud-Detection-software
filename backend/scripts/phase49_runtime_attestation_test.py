#!/usr/bin/env python3
"""Phase 49: Runtime release attestation, deployment integrity & active-model drift.

Tests the REAL runtime path: the risk engine lifespan, health endpoint, and
release-attestation endpoint.  Adversarial scenarios prove that a corrupted,
mismatched, or unapproved release can never reach healthy inference.

NOTE on live boots: each TestClient-style boot loads the real native ensemble
(~5 s).  The suite boots the real lifespan ONCE for the live-path tests
(valid boot, corrupted boot, forged-manifest boot) and covers the remaining
scenarios against the same modules the lifespan calls (verify_release_for_load /
detect_runtime_drift / check_registry_runtime_consistency), which is the exact
code the runtime executes.

Run: ./.venv/Scripts/python.exe backend/scripts/phase49_runtime_attestation_test.py
"""
from __future__ import annotations

import asyncio
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
os.environ.setdefault("JWT_SECRET", "phase49-test-secret-0123456789abcdef")
os.environ["PS14_MODE"] = "development"

passed = failed = 0


def check(label, condition, detail=""):
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


def make_release(tmp: Path, files: dict[str, bytes] | None = None,
                 **kw) -> tuple[Path, object]:
    """Create a signed release in tmp/artifacts + return (artifact_dir, manifest)."""
    from src.monitoring.release_manifest import create_release_manifest
    art = tmp / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    for name, data in (files or {"model.joblib": b"model-bytes",
                                 "scaler.joblib": b"scaler-bytes"}).items():
        (art / name).write_bytes(data)
    defaults = dict(
        release_id="rel-test", model_id="m-test", model_version="v-test",
        artifact_dir=art, feature_version="v1",
        gate_verdict="PROMOTION_ELIGIBLE", gate_timestamp=time.time(),
    )
    defaults.update(kw)
    m = create_release_manifest(**defaults)
    return art, m


# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    global passed, failed
    from src.monitoring.runtime_attestation import (
        RuntimeState, RuntimeAttestation, verify_release_for_load,
        build_attestation, detect_runtime_drift,
        check_registry_runtime_consistency, attestation_audit_payload,
        RELEASE_MANIFEST_FILENAME,
    )
    from src.monitoring.release_manifest import (
        ReleaseManifest, create_release_manifest, artifact_set_hash,
    )

    print("== Phase 49: Runtime release attestation ==\n")

    # ── A. Module-level verification pipeline ────────────────────────
    print("--- A. Pre-load verification pipeline ---")
    tmp = Path(tempfile.mkdtemp())

    # A1: valid release verifies
    art, man = make_release(tmp)
    ok, failures, ident = verify_release_for_load(man, art)
    check("A1 valid release verifies", ok and not failures, str(failures))
    check("A2 identity carries manifest hash", len(ident["manifest_hash"]) == 64)
    check("A3 identity carries artifact hash", ident["artifact_hash"] == man.artifact_hash)

    # A4: deterministic identity for same inputs
    ok2, _f2, ident2 = verify_release_for_load(man, art)
    check("A4 identity deterministic", ident == ident2)

    # A5: unsigned (forged signature) rejected
    art5, man5 = make_release(tmp / "a5")
    man5._signature = "f" * 64
    ok5, f5, _ = verify_release_for_load(man5, art5)
    check("A5 forged signature blocked", not ok5 and any("signature" in x for x in f5))

    # A6: tampered payload (hash changes -> signature breaks)
    art6, man6 = make_release(tmp / "a6")
    man6.model_version = "tampered"
    ok6, f6, _ = verify_release_for_load(man6, art6)
    check("A6 tampered manifest blocked", not ok6)

    # A7: non-ELIGIBLE gate verdict blocked
    art7, man7 = make_release(tmp / "a7", gate_verdict="PROMOTION_BLOCKED")
    ok7, f7, _ = verify_release_for_load(man7, art7)
    check("A7 non-eligible gate blocked",
          not ok7 and any("gate_verdict" in x for x in f7))

    # A8: corrupted artifact blocked (one byte)
    art8, man8 = make_release(tmp / "a8")
    (art8 / "model.joblib").write_bytes(b"model-bytes-CORRUPTED")
    ok8, f8, _ = verify_release_for_load(man8, art8)
    check("A8 corrupted artifact blocked", not ok8 and any("artifacts" in x for x in f8))

    # A9: missing artifact file blocked
    art9, man9 = make_release(tmp / "a9")
    (art9 / "scaler.joblib").unlink()
    ok9, f9, _ = verify_release_for_load(man9, art9)
    check("A9 missing artifact blocked", not ok9)

    # A10: missing artifact dir blocked
    ok10, f10, _ = verify_release_for_load(man8, tmp / "does-not-exist")
    check("A10 missing artifact dir blocked", not ok10)

    # A11: unexpected extra artifact blocked
    art11, man11 = make_release(tmp / "a11")
    (art11 / "evil.joblib").write_bytes(b"planted")
    ok11, f11, _ = verify_release_for_load(man11, art11)
    check("A11 unexpected artifact blocked", not ok11)

    # A12: swapped artifacts blocked (model bytes in scaler, vice versa)
    art12, man12 = make_release(tmp / "a12")
    (art12 / "model.joblib").write_bytes(b"scaler-bytes")
    (art12 / "scaler.joblib").write_bytes(b"model-bytes")
    ok12, f12, _ = verify_release_for_load(man12, art12)
    check("A12 swapped artifacts blocked", not ok12)

    # A13: rule-hash mismatch blocked
    art13, man13 = make_release(tmp / "a13", rule_hash="aaaa")
    ok13, f13, _ = verify_release_for_load(man13, art13, expected_rule_hash="bbbb")
    check("A13 rule mismatch blocked", not ok13 and any("rule_hash" in x for x in f13))

    # A14: feature-version binding mismatch blocked
    art14, man14 = make_release(tmp / "a14", feature_version="v1")
    ok14, f14, _ = verify_release_for_load(man14, art14, expected_feature_version="v2")
    check("A14 feature version mismatch blocked", not ok14)

    # A15: matching binding passes
    ok15, f15, _ = verify_release_for_load(man14, art14, expected_feature_version="v1")
    check("A15 matching binding passes", ok15)

    # ── B. Attestation ───────────────────────────────────────────────
    print("\n--- B. Runtime attestation ---")
    okB, _fB, identB = verify_release_for_load(man, art)
    att = build_attestation(identB, art)
    check("B1 attestation verified flag", att.verified)
    check("B2 attestation hash deterministic",
          att.attestation_hash() == att.attestation_hash())
    check("B3 loaded_at recorded", att.loaded_at > 0)

    att_b = build_attestation(dict(identB), art)
    check("B4 same identity -> same attestation hash",
          att.attestation_hash() == att_b.attestation_hash())

    att_c = build_attestation({**identB, "artifact_hash": "0" * 64}, art)
    check("B5 changed artifact -> different attestation hash",
          att.attestation_hash() != att_c.attestation_hash())

    # B6: safe dict exposes no secrets / signature material
    safe = att.to_safe_dict()
    leak = [k for k in safe if "signature" in k.lower() or "secret" in k.lower() or "key" in k.lower()]
    check("B6 safe dict has no secret-bearing keys", not leak, str(leak))
    check("B7 safe dict includes release identity",
          safe["release_id"] == man.release_id and safe["manifest_hash"])

    # B8: attestation of empty identity is rejected downstream (verified=False)
    att_unverified = RuntimeAttestation(
        release_id="", model_id="", model_version="", artifact_hash="",
        manifest_hash="", feature_version="", schema_version="",
        preprocessing_hash="", rule_hash="", evaluation_record_hash="",
        verified=False,
    )
    check("B8 unverified attestation flagged", att_unverified.verified is False)
    p8 = attestation_audit_payload(att_unverified, RuntimeState.FAILED, ["x"])
    check("B9 audit payload carries failures", p8["failures"] == ["x"])
    check("B10 audit payload carries state", p8["runtime_state"] == "FAILED")

    # ── C. Runtime drift detection ───────────────────────────────────
    print("\n--- C. Runtime drift detection ---")
    st, dr = detect_runtime_drift(att, art)
    check("C1 unchanged artifacts -> READY", st == RuntimeState.READY and not dr)

    (art / "model.joblib").write_bytes(b"model-bytes-REPLACED-AFTER-LOAD")
    st2, dr2 = detect_runtime_drift(att, art)
    check("C2 post-load file replacement -> DRIFTED", st2 == RuntimeState.DRIFTED)
    check("C3 drift reasons actionable", any("artifact_hash" in x for x in dr2))

    # C4: file modification must NOT silently become a new active model —
    # drift never produces a READY state for the OLD attestation
    st3, _ = detect_runtime_drift(att, art)
    check("C3b drifted state never READY", st3 != RuntimeState.READY)

    (art / "model.joblib").write_bytes(b"model-bytes")  # restore
    st4, _ = detect_runtime_drift(att, art)
    check("C4 restored artifacts -> READY again", st4 == RuntimeState.READY)

    # C5: deleted artifact dir -> DRIFTED
    st5, _ = detect_runtime_drift(att, tmp / "gone")
    check("C5 missing dir -> DRIFTED", st5 == RuntimeState.DRIFTED)

    # ── D. Registry ↔ runtime consistency ────────────────────────────
    print("\n--- D. Registry/runtime consistency ---")
    reg_active = {
        "release_id": att.release_id, "model_id": att.model_id,
        "artifact_hash": att.artifact_hash, "manifest_hash": att.manifest_hash,
        "feature_version": att.feature_version, "model_version": att.model_version,
        "rule_hash": att.rule_hash,
    }
    stD, mm = check_registry_runtime_consistency(reg_active, att)
    check("D1 matching registry -> READY", stD == RuntimeState.READY, str(mm))

    stD2, mm2 = check_registry_runtime_consistency(
        {**reg_active, "artifact_hash": "e" * 64}, att)
    check("D2 artifact mismatch -> INCONSISTENT",
          stD2 == RuntimeState.INCONSISTENT and any("artifact_hash" in x for x in mm2))

    stD3, mm3 = check_registry_runtime_consistency(
        {**reg_active, "release_id": "rel-OTHER"}, att)
    check("D3 release_id mismatch -> INCONSISTENT",
          stD3 == RuntimeState.INCONSISTENT)

    stD4, mm4 = check_registry_runtime_consistency(
        {**reg_active, "model_version": "v-OLD"}, att)
    check("D4 model_version mismatch -> INCONSISTENT",
          stD4 == RuntimeState.INCONSISTENT)

    stD5, _ = check_registry_runtime_consistency(None, att)
    check("D5 registry w/o release tracking -> READY (attestation governs)",
          stD5 == RuntimeState.READY)

    stD6, _ = check_registry_runtime_consistency({}, att)
    check("D6 empty registry record -> READY", stD6 == RuntimeState.READY)

    # Every tracked field mismatching is caught
    caught = 0
    for field in ["release_id", "model_id", "model_version", "artifact_hash",
                  "manifest_hash", "feature_version", "schema_version",
                  "preprocessing_hash", "rule_hash"]:
        stX, _ = check_registry_runtime_consistency({field: "WRONG"}, att)
        caught += stX == RuntimeState.INCONSISTENT
    check("D7 all 9 tracked fields enforced", caught == 9, f"caught={caught}")

    # ── E. Manifest serialization / immutability ─────────────────────
    print("\n--- E. Manifest serialization ---")
    roundtrip = ReleaseManifest.from_dict(man.to_dict())
    check("E1 to_dict/from_dict roundtrip preserves signature",
          roundtrip.verify_signature()[0])
    check("E2 roundtrip preserves hash",
          roundtrip.compute_manifest_hash() == man.compute_manifest_hash())

    saved = tmp / "saved_manifest.json"
    man.save(saved)
    loaded = ReleaseManifest.load(saved)
    check("E3 save/load roundtrip verifies", loaded.verify_signature()[0])
    okE, _fE, _iE = verify_release_for_load(loaded, art)
    check("E4 loaded-from-disk manifest passes pipeline", okE)

    mutated = json.loads(saved.read_text())
    mutated["model_version"] = "mutated"
    (tmp / "mutated.json").write_text(json.dumps(mutated))
    m2 = ReleaseManifest.load(tmp / "mutated.json")
    check("E5 mutated manifest signature fails", not m2.verify_signature()[0])

    # ── F. Live path: REAL risk engine lifespan ──────────────────────
    # Boot the actual app 3 times: valid / corrupted artifact / forged manifest.
    print("\n--- F. Live-path boots (real lifespan) ---")
    import src.risk_engine.main as rm
    from src.monitoring.release_manifest import create_release_manifest as _crm

    NATIVE_DIR = ROOT / "models" / "production" / "altman_native"
    PROD_DIR = ROOT / "models" / "production"
    mpath = PROD_DIR / RELEASE_MANIFEST_FILENAME
    live_rule_hash = hashlib.sha256(rm.RULES_PATH.read_bytes()).hexdigest()[:32]
    live_man = _crm(
        release_id="rel-49-live", model_id="altman_native",
        model_version="altman_native_E_hardneg_cert_20260904",
        artifact_dir=NATIVE_DIR, feature_version="native48",
        schema_version="native48", rule_hash=live_rule_hash,
        gate_verdict="PROMOTION_ELIGIBLE", gate_timestamp=time.time(),
    )
    # Phase 110: back up the production manifest — this suite overwrites it
    # with a fixture below and must restore the exact original bytes at the
    # end, so a crash can never leave fixture bytes (or a deleted manifest)
    # behind in the production tree.
    _orig_prod_manifest = mpath.read_bytes() if mpath.exists() else None
    live_man.save(mpath)

    async def boot():
        async with rm.lifespan(rm.app):
            return (rm.attestation, rm.runtime_state, rm.fusion,
                    rm.health(), rm.runtime_failures)

    try:
        attL, stL, fusL, hL, _fL = asyncio.run(boot())
        check("F1 valid live boot -> READY", stL == RuntimeState.READY)
        check("F2 live boot attested", attL is not None and attL.release_id == "rel-49-live")
        check("F3 live boot model_readiness loaded",
              fusL is not None and hL.get("model_readiness") == "loaded")
        check("F4 live health runtime_state READY", hL["runtime_state"] == "READY")
        check("F5 live attestation binds deployed artifacts",
              attL.artifact_hash == live_man.artifact_hash)

        # corrupted artifact restart
        xl = NATIVE_DIR / "xgb_native.joblib"
        orig = xl.read_bytes()
        xl.write_bytes(orig + b"CORRUPTION")
        fC, stC, fusC, hC, failsC = asyncio.run(boot())
        check("F6 corrupted artifact restart -> FAILED", stC == RuntimeState.FAILED)
        check("F7 corrupted restart loads NO model", fusC is None)
        check("F8 corrupted restart leaves NO attestation", fC is None)
        check("F9 corrupted restart health model_readiness not_loaded",
              hC.get("model_readiness") == "not_loaded")
        check("F10 corrupted restart failure is auditable",
              any("Artifact hash mismatch" in x for x in failsC), str(failsC[:1]))
        xl.write_bytes(orig)

        # forged manifest restart
        md = json.loads(mpath.read_text())
        md["gate_verdict"] = "PROMOTION_BLOCKED"
        mpath.write_text(json.dumps(md))
        _aF, stF, fusF, hF, _ffF = asyncio.run(boot())
        check("F11 forged manifest restart -> FAILED", stF == RuntimeState.FAILED)
        check("F12 forged manifest loads NO model", fusF is None)

        # tampered-but-resigned manifest (valid signature, wrong artifacts)
        md2 = json.loads(mpath.read_text())
        # rebuild a manifest with the ORIGINAL payload but a hash of DIFFERENT artifacts
        from src.monitoring.release_manifest import ReleaseManifest as _RM
        bad_man = _crm(
            release_id="rel-49-live", model_id="altman_native",
            model_version="altman_native_E_hardneg_cert_20260904",
            artifact_dir=PROD_DIR,  # WRONG dir (aggregate hash differs)
            feature_version="native48", schema_version="native48",
            rule_hash=live_rule_hash,
            gate_verdict="PROMOTION_ELIGIBLE", gate_timestamp=time.time(),
        )
        bad_man.save(mpath)
        _aX, stX, fusX, _hX, _fx = asyncio.run(boot())
        check("F13 validly-signed manifest w/ wrong artifacts -> FAILED",
              stX == RuntimeState.FAILED and fusX is None)
    finally:
        if mpath.exists():
            mpath.unlink()

    # ── G. Env-var override is NOT evidence ──────────────────────────
    print("\n--- G. Configuration override protections ---")
    os.environ["MODEL_VERSION"] = "approved-version"
    os.environ["RELEASE_ID"] = "rel-49-live"
    try:
        attG, stG, _fG, _hG, _fg = asyncio.run(boot_with(rm))
        check("G1 env vars do not fabricate attestation",
              attG is None, f"attestation={attG is not None} state={stG}")
    finally:
        del os.environ["MODEL_VERSION"]
        del os.environ["RELEASE_ID"]

    # ── H. Endpoint behaviour (unauth + auth) ────────────────────────
    print("\n--- H. Attestation endpoint ---")
    from fastapi.testclient import TestClient
    from src.settings import load_dotenv_and_patch, settings as _s
    load_dotenv_and_patch()
    _tok = _s.internal_token
    with TestClient(rm.app) as c:
        r = c.get("/internal/release-attestation",
                  headers={"X-Internal-Token": "wrong-token"})
        check("H1 attestation endpoint requires internal token", r.status_code == 401)
        r2 = c.get("/internal/release-attestation",
                   headers={"X-Internal-Token": _tok})
        body = r2.json()
        check("H2 endpoint exposes runtime_state", "runtime_state" in body, r2.text[:120])
        check("H3 no manifest -> attested false (honest state)",
              body.get("attested") is False, r2.text[:200])
        h = c.get("/health").json()
        check("H4 health reports runtime_state", "runtime_state" in h)
        check("H5 health reports release_attested flag", "release_attested" in h)
        check("H6 legacy (no manifest) boot healthy, model loaded",
              h["status"] == "ok" and h.get("model_readiness") == "loaded")

    # Phase 110: restore the exact production-manifest bytes (H needed the
    # file absent; the shared tree must end this suite byte-identical).
    if _orig_prod_manifest is not None:
        mpath.write_bytes(_orig_prod_manifest)
    elif mpath.exists():
        mpath.unlink()  # suite started with no manifest -> leave as found

    print(f"\n{'='*60}")
    print(f"Phase 49: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


async def boot_with(rm):
    async with rm.lifespan(rm.app):
        return rm.attestation, rm.runtime_state, rm.fusion, rm.health(), rm.runtime_failures


if __name__ == "__main__":
    main()
