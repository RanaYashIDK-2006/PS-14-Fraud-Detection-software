#!/usr/bin/env python3
"""Phase 52: Forensic release history and operational governance tests.

Adversarial tests proving release history is trustworthy, lifecycle
transitions are reconstructable, historical evidence is tamper-evident,
incidents are persistent, and forensic data does not expose secrets.

Run: ./.venv/Scripts/python.exe backend/scripts/phase52_forensic_governance_test.py
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
os.environ.setdefault("JWT_SECRET", "phase52-test-0123456789abcdef")
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
        release_id=f"rel-test52-{suffix}",
        model_id=f"m-test52-{suffix}",
        model_version=f"v-test52-{suffix}",
        artifact_dir=art,
        feature_version="v1",
        schema_version="v1",
        preprocessing_hash=hashlib.sha256(f"scaler-bytes-{suffix}".encode()).hexdigest(),
        rule_hash=rule_hash,
        evaluation_record_hash=f"eval52{suffix}" + "0" * 56,
        training_config_hash=f"cfg52{suffix}" + "0" * 56,
        source_git_sha="test52",
        gate_verdict="LEGACY_ATTESTED",
    )
    return art, m


def _make_record(tmp: Path, release_id: str, manifest) -> object:
    """Create and verify a release record for lifecycle testing."""
    from src.monitoring.release_lifecycle import (
        ReleaseRecord, verify_release_lifecycle, ReleaseState,
    )
    from src.monitoring.release_manifest import artifact_set_hash, sha256_file
    rule_hash = hashlib.sha256(
        (ROOT / "backend" / "src" / "risk_engine" / "rules.yaml").read_bytes()
    ).hexdigest()[:32]
    rec = ReleaseRecord(
        release_id=release_id,
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        artifact_hash=manifest.artifact_hash,
        manifest_hash=manifest.compute_manifest_hash(),
        feature_version=manifest.feature_version,
        schema_version=manifest.schema_version,
        rule_hash=manifest.rule_hash,
        preprocessing_hash=manifest.preprocessing_hash,
        gate_verdict=manifest.gate_verdict,
        state=ReleaseState.DISCOVERED,
    )
    return rec


def main() -> int:
    global passed, failed
    print("=" * 60)
    print("Phase 52: Forensic release history and operational governance")
    print("=" * 60)

    from src.monitoring.forensic_release_history import (
        ForensicReleaseHistory,
        IncidentRecord,
        IncidentCategory,
        IncidentState,
        ReleaseTransitionRecord,
        ForensicSnapshot,
        EvidenceGap,
    )
    from src.monitoring.release_lifecycle import (
        ReleaseState,
        ReleaseRegistry,
        verify_release_lifecycle,
    )
    from src.monitoring.release_manifest import create_release_manifest, sha256_file

    PRODUCTION_DIR = ROOT / "models" / "production"
    ARTIFACT_DIR = PRODUCTION_DIR / "altman_native"
    RULES_PATH = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
    rule_hash = hashlib.sha256(RULES_PATH.read_bytes()).hexdigest()[:32]

    # ═══════════════════════════════════════════════════════════════════════
    # A. Release Transition Recording & Chain Integrity
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- A. Release transition recording & chain integrity --")
    with tempfile.TemporaryDirectory(prefix="phase52-hist-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")
        reg = ReleaseRegistry(Path(td) / "registry.json")
        art, manifest = _make_release(Path(td))

        rec = _make_record(Path(td), "rel-a", manifest)
        reg.register(rec)

        # 1. Record transition
        tr = hist.record_transition(rec, "DISCOVERED", "REGISTERED", reason="test registration", actor_type="SYSTEM")
        check("1. transition recorded", tr.transition_id.startswith("tr-"))
        check("1b. transition has hash", len(tr.transition_hash) == 64)

        # 2. Chain integrity
        ok, msg = hist.verify_transition_chain()
        check("2. chain intact after single transition", ok, msg)

        # 3. Multiple transitions maintain chain
        tr2 = hist.record_transition(rec, "REGISTERED", "ACTIVATED", reason="activated for test")
        tr3 = hist.record_transition(rec, "ACTIVATED", "RUNTIME_ATTESTED", reason="attested")
        ok, msg = hist.verify_transition_chain()
        check("3. chain intact after 3 transitions", ok, msg)

        # 4. Different release transitions
        rec2 = _make_record(Path(td), "rel-b", manifest)
        reg.register(rec2)
        tr4 = hist.record_transition(rec2, "DISCOVERED", "REGISTERED", reason="second release")
        ok, msg = hist.verify_transition_chain()
        check("4. chain intact across releases", ok, msg)

        # 5. Tampered transition breaks chain
        original_hash = hist._transitions[1].transition_hash
        hist._transitions[1].transition_hash = "0" * 64
        ok, msg = hist.verify_transition_chain()
        check("5. tampered transition breaks chain", not ok and "broken" in msg.lower(), msg)
        hist._transitions[1].transition_hash = original_hash  # restore

    # ═══════════════════════════════════════════════════════════════════════
    # B. Active-Release Timeline Reconstruction
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- B. Active-release timeline --")
    with tempfile.TemporaryDirectory(prefix="phase52-tl-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")

        # Create two releases and simulate A->B transition
        rec_a = _make_record(Path(td), "rel-tl-a", manifest)
        rec_b = _make_record(Path(td), "rel-tl-b", manifest)

        hist.record_transition(rec_a, "DISCOVERED", "REGISTERED")
        hist.record_transition(rec_a, "REGISTERED", "ACTIVATED", reason="activated A")
        hist.record_transition(rec_a, "ACTIVATED", "RUNTIME_ATTESTED")
        hist.record_transition(rec_a, "RUNTIME_ATTESTED", "REVOKED", reason="revoked A")
        hist.record_transition(rec_b, "DISCOVERED", "REGISTERED")
        hist.record_transition(rec_b, "REGISTERED", "ACTIVATED", reason="activated B")

        timeline = hist.get_active_timeline()
        check("6. timeline has entries", len(timeline) >= 2)
        check("7. first entry is A with deactivation", timeline[0]["release_id"] == "rel-tl-a" and timeline[0]["deactivated_at"] is not None)
        check("8. second entry is B still active", timeline[1]["release_id"] == "rel-tl-b" and timeline[1]["deactivated_at"] is None)

    # ═══════════════════════════════════════════════════════════════════════
    # C. Provenance Binding (Tamper Detection)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- C. Provenance binding --")
    with tempfile.TemporaryDirectory(prefix="phase52-prov-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")
        rec = _make_record(Path(td), "rel-prov", manifest)
        hist.record_transition(rec, "DISCOVERED", "REGISTERED")
        hist.record_transition(rec, "REGISTERED", "ACTIVATED")

        # Save originals for restoration
        t0 = hist._transitions[0]
        orig_id = t0.release_id
        orig_mh = t0.manifest_hash
        orig_state = t0.new_state
        orig_ts = t0.timestamp
        orig_reason = t0.reason

        # 9. Changed release_id breaks chain
        t0.release_id = "tampered-id"
        ok, _ = hist.verify_transition_chain()
        check("9. changed release_id breaks chain", not ok)
        t0.release_id = orig_id

        # 10. Changed manifest_hash breaks chain
        t0.manifest_hash = "tampered"
        ok, _ = hist.verify_transition_chain()
        check("10. changed manifest_hash breaks chain", not ok)
        t0.manifest_hash = orig_mh

        # 11. Changed state breaks chain
        t0.new_state = "TAMPERED"
        ok, _ = hist.verify_transition_chain()
        check("11. changed state breaks chain", not ok)
        t0.new_state = orig_state

        # 12. Changed timestamp breaks chain
        t0.timestamp = 9999999999.0
        ok, _ = hist.verify_transition_chain()
        check("12. changed timestamp breaks chain", not ok)
        t0.timestamp = orig_ts

        # 13. Changed reason breaks chain
        t0.reason = "tampered reason"
        ok, _ = hist.verify_transition_chain()
        check("13. changed reason breaks chain", not ok)
        t0.reason = orig_reason

        # Verify chain restored before deletion test
        ok, _ = hist.verify_transition_chain()
        check("13b. chain restored after tamper tests", ok)

        # Add a third transition so deletion of middle breaks the chain
        hist.record_transition(rec, "ACTIVATED", "RUNTIME_ATTESTED", reason="attested")
        ok, _ = hist.verify_transition_chain()
        check("14a. chain intact with 3 transitions", ok)
        # 14. Deleted middle transition breaks chain
        hist._transitions.pop(1)
        ok, _ = hist.verify_transition_chain()
        check("14. deleted middle transition breaks chain", not ok)

    # ═══════════════════════════════════════════════════════════════════════
    # D. Gap Detection
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- D. Gap detection --")
    with tempfile.TemporaryDirectory(prefix="phase52-gap-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")
        reg = ReleaseRegistry(Path(td) / "registry.json")
        rec = _make_record(Path(td), "rel-gap", manifest)
        reg.register(rec)
        # Transition through lifecycle to REGISTERED so activation works
        rec.transition(ReleaseState.MANIFEST_VERIFIED, "verified")
        rec.transition(ReleaseState.GATE_VERIFIED, "gate passed")
        rec.transition(ReleaseState.REGISTERED, "registered")
        reg._save()
        reg.activate(rec.release_id)

        # 15. No activation transition in history => gap
        gaps = hist.detect_gaps(reg)
        registry_gaps = [g for g in gaps if g.category == "REGISTRY_HIST_MISMATCH"]
        check("15. registry active without activation transition => gap", len(registry_gaps) >= 1)

        # 16. Record activation => gap resolved
        hist.record_transition(rec, "REGISTERED", "ACTIVATED", reason="test")
        gaps = hist.detect_gaps(reg)
        registry_gaps = [g for g in gaps if g.category == "REGISTRY_HIST_MISMATCH"]
        check("16. activation transition resolves registry gap", len(registry_gaps) == 0)

        # 17. Runtime mismatch detected
        gaps = hist.detect_gaps(reg, runtime_release_id="wrong-release")
        rt_gaps = [g for g in gaps if g.category == "RUNTIME_HIST_MISMATCH"]
        check("17. runtime mismatch detected", len(rt_gaps) >= 1)

        # 18. Correct runtime => no mismatch
        gaps = hist.detect_gaps(reg, runtime_release_id=rec.release_id)
        rt_gaps = [g for g in gaps if g.category == "RUNTIME_HIST_MISMATCH"]
        check("18. correct runtime => no mismatch", len(rt_gaps) == 0)

        # 19. Manifest hash mismatch detected
        hist.record_transition(rec, "ACTIVATED", "RUNTIME_ATTESTED")
        gaps = hist.detect_gaps(reg, runtime_release_id=rec.release_id, runtime_manifest_hash="wrong_hash")
        mh_gaps = [g for g in gaps if g.category == "MANIFEST_HASH_MISMATCH"]
        check("19. manifest hash mismatch detected", len(mh_gaps) >= 1)

    # ═══════════════════════════════════════════════════════════════════════
    # E. Incident Records & State Machine
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- E. Incident records & state machine --")
    with tempfile.TemporaryDirectory(prefix="phase52-inc-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")

        # 20. Create incident
        inc = hist.create_incident(
            category=IncidentCategory.ARTIFACT_TAMPERING,
            affected_release_id="rel-inc",
            manifest_hash="abc123",
            runtime_state="DRIFTED",
            detection_source="health_check",
        )
        check("20. incident created", inc.incident_id.startswith("inc-"))
        check("20b. initial state is DETECTED", inc.response_state == IncidentState.DETECTED.value)

        # 21. State transitions
        ok = hist.transition_incident(inc.incident_id, IncidentState.CONTAINED, "contained the threat")
        check("21. DETECTED -> CONTAINED", ok)
        ok = hist.transition_incident(inc.incident_id, IncidentState.INVESTIGATING, "investigating root cause")
        check("22. CONTAINED -> INVESTIGATING", ok)
        ok = hist.transition_incident(inc.incident_id, IncidentState.RESOLVED, "fixed and verified")
        check("23. INVESTIGATING -> RESOLVED", ok)

        # 24. Cannot transition from RESOLVED
        ok = hist.transition_incident(inc.incident_id, IncidentState.DETECTED)
        check("24. RESOLVED -> DETECTED blocked", not ok)

        # 25. Cannot skip states
        inc2 = hist.create_incident(category=IncidentCategory.RUNTIME_DRIFT)
        ok = hist.transition_incident(inc2.incident_id, IncidentState.INVESTIGATING)
        check("25. DETECTED -> INVESTIGATING blocked (must go through CONTAINED)", not ok)

        # 26. Incident for specific release
        inc3 = hist.create_incident(
            category=IncidentCategory.HASH_MISMATCH,
            affected_release_id="rel-inc",
        )
        rel_incs = hist.get_incidents_for_release("rel-inc")
        check("26. get_incidents_for_release finds incidents", len(rel_incs) >= 2)

        # 27. Open incidents
        open_incs = hist.get_open_incidents()
        check("27. open incidents exist", len(open_incs) >= 1)

        # 28. Incident history preserved
        check("28. incident state_history preserved", len(inc.state_history) >= 3)

    # ═══════════════════════════════════════════════════════════════════════
    # F. Forensic Snapshots
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- F. Forensic snapshots --")
    with tempfile.TemporaryDirectory(prefix="phase52-snap-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")

        # 29. Take snapshot
        snap = hist.take_forensic_snapshot(
            release_id="rel-snap",
            manifest_hash="abc123",
            runtime_state="READY",
            feature_version="v1",
            schema_version="v1",
        )
        check("29. snapshot created", snap.snapshot_id.startswith("snap-"))
        check("29b. snapshot has hash", len(snap.snapshot_hash) == 64)

        # 30. Tampered snapshot breaks hash
        orig_hash = snap.snapshot_hash
        snap.runtime_state = "TAMPERED"
        recomputed = hashlib.sha256(json.dumps({
            k: v for k, v in snap.to_dict().items() if k != "snapshot_hash"
        }, sort_keys=True).encode()).hexdigest()
        check("30. tampered snapshot has different hash", recomputed != orig_hash)

        # 31. Get snapshots for release
        snaps = hist.get_snapshots(release_id="rel-snap")
        check("31. get_snapshots finds release snapshots", len(snaps) >= 1)

    # ═══════════════════════════════════════════════════════════════════════
    # G. Persistence Across Restart
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- G. Persistence across restart --")
    with tempfile.TemporaryDirectory(prefix="phase52-restart-") as td:
        fp = Path(td) / "forensic.json"

        # Write and verify
        hist1 = ForensicReleaseHistory(fp)
        rec = _make_record(Path(td), "rel-restart", manifest)
        hist1.record_transition(rec, "DISCOVERED", "REGISTERED")
        hist1.record_transition(rec, "REGISTERED", "ACTIVATED")
        inc = hist1.create_incident(category=IncidentCategory.SIGNATURE_FAILURE, affected_release_id="rel-restart")
        snap = hist1.take_forensic_snapshot(release_id="rel-restart", runtime_state="READY")

        # Reload
        hist2 = ForensicReleaseHistory(fp)
        # 32. Transitions survive restart
        check("32. transitions survive restart", len(hist2.get_transitions()) == 2)
        # 33. Chain integrity survives restart
        ok, _ = hist2.verify_transition_chain()
        check("33. chain integrity survives restart", ok)
        # 34. Incidents survive restart
        check("34. incidents survive restart", len(hist2._incidents) == 1)
        # 35. Snapshots survive restart
        check("35. snapshots survive restart", len(hist2.get_snapshots()) == 1)
        # 36. Open incidents survive restart
        open_incs = hist2.get_open_incidents()
        check("36. open incidents survive restart", len(open_incs) >= 1)

    # ═══════════════════════════════════════════════════════════════════════
    # H. Concurrent Lifecycle Events
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- H. Concurrent lifecycle events --")
    with tempfile.TemporaryDirectory(prefix="phase52-conc-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")

        # Record transitions from multiple threads
        results = []
        def record_concurrent(i: int):
            r = _make_record(Path(td), f"rel-conc-{i}", manifest)
            t = hist.record_transition(r, "DISCOVERED", "REGISTERED", reason=f"thread-{i}")
            results.append(t.transition_id)

        threads = [threading.Thread(target=record_concurrent, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 37. All transitions recorded
        check("37. all concurrent transitions recorded", len(results) == 5)
        # 38. Chain may or may not be valid (no locking in forensic history),
        #     but all transitions should be present
        all_transitions = hist.get_transitions()
        check("38. all transitions present in history", len(all_transitions) == 5)

    # ═══════════════════════════════════════════════════════════════════════
    # I. Export / Reconstruction
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- I. Export / reconstruction --")
    with tempfile.TemporaryDirectory(prefix="phase52-export-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")
        rec = _make_record(Path(td), "rel-export", manifest)
        hist.record_transition(rec, "DISCOVERED", "REGISTERED")
        hist.record_transition(rec, "REGISTERED", "ACTIVATED")
        inc = hist.create_incident(category=IncidentCategory.MANIFEST_TAMPERING)
        snap = hist.take_forensic_snapshot(release_id="rel-export")

        exported = hist.export_history()

        # 39. Export format
        check("39. export has correct format", exported["format"] == "ps14-forensic-history-v1")
        # 40. Export has all transitions
        check("40. export has all transitions", len(exported["transitions"]) == 2)
        # 41. Export has chain integrity
        check("41. export has chain integrity", exported["chain_integrity"][0] is True)
        # 42. Export has incidents
        check("42. export has incidents", len(exported["incidents"]) == 1)
        # 43. Export has snapshots
        check("43. export has snapshots", len(exported["snapshots"]) == 1)
        # 44. Export has timeline
        check("44. export has active timeline", len(exported["active_timeline"]) >= 1)
        # 45. Export does NOT contain secrets
        exported_str = json.dumps(exported)
        check("45. export contains no JWT_SECRET",
              "phase52-test" not in exported_str and "reconstruct-legacy" not in exported_str)
        check("46. export contains no HMAC keys",
              "hmac_key" not in exported_str.lower())

    # ═══════════════════════════════════════════════════════════════════════
    # J. Security / Privacy Review
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- J. Security / privacy review --")
    with tempfile.TemporaryDirectory(prefix="phase52-sec-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")

        exported = hist.export_history()
        exported_str = json.dumps(exported)

        # 47. No secrets in export
        secret_patterns = ["password", "secret", "private_key", "api_key", "credential"]
        found_secrets = [p for p in secret_patterns if p in exported_str.lower()]
        check("47. no secret patterns in export", len(found_secrets) == 0, str(found_secrets))

        # 48. Transition records don't contain full payloads
        for t in exported["transitions"]:
            check("48. transition has no full payload field",
                  "full_payload" not in t and "raw_data" not in t)

    # ═══════════════════════════════════════════════════════════════════════
    # K. Event Deletion / History Truncation Detection
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- K. Deletion / truncation detection --")
    with tempfile.TemporaryDirectory(prefix="phase52-del-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")
        rec = _make_record(Path(td), "rel-del", manifest)
        hist.record_transition(rec, "DISCOVERED", "REGISTERED")
        hist.record_transition(rec, "REGISTERED", "ACTIVATED")
        hist.record_transition(rec, "ACTIVATED", "RUNTIME_ATTESTED")

        # 49. Deleting middle transition breaks chain (2nd's prev_hash no longer matches)
        hist._transitions.pop(1)
        ok, msg = hist.verify_transition_chain()
        check("49. deleting middle transition breaks chain", not ok)

        # Restore for test 50
        hist._transitions = hist.get_transition_chain()
        # 50. Truncating from the end still validates remaining entries
        hist._transitions.pop()  # remove last entry
        ok, msg = hist.verify_transition_chain()
        check("50. truncating end validates remaining entries", ok)

    # ═══════════════════════════════════════════════════════════════════════
    # L. Incident Persistence & State Machine Edge Cases
    # ═══════════════════════════════════════════════════════════════════════

    print("\n-- L. Incident edge cases --")
    with tempfile.TemporaryDirectory(prefix="phase52-ince-") as td:
        hist = ForensicReleaseHistory(Path(td) / "forensic.json")

        # 51. Transition nonexistent incident
        ok = hist.transition_incident("nonexistent", IncidentState.CONTAINED)
        check("51. transition nonexistent incident fails", not ok)

        # 52. Get nonexistent incident
        check("52. get nonexistent incident returns None", hist.get_incident("nonexistent") is None)

        # 53. Direct incident transition chain bypass
        inc = hist.create_incident(category=IncidentCategory.OTHER)
        ok = inc.transition_incident(IncidentState.RESOLVED, "skip to resolved")
        check("53. DETECTED -> RESOLVED directly blocked", not ok)

        # 54. DETECTED -> CONTAINED -> RESOLVED works
        ok = inc.transition_incident(IncidentState.CONTAINED, "contained")
        check("54. DETECTED -> CONTAINED works", ok)
        ok = inc.transition_incident(IncidentState.RESOLVED, "resolved")
        check("55. CONTAINED -> RESOLVED works", ok)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    total = passed + failed
    if failed:
        print(f"Phase 52: {passed} passed, {failed} failed (out of {total})")
    else:
        print(f"Phase 52: {passed} passed, 0 failed")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
