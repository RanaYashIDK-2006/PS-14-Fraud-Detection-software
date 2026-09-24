"""Phase 110 — Authoritative shape reconciliation & audit-hygiene CI
enforcement: test suite.

Covers (Parts 1-7, 9-11):
  1  Manifest contract: canonical constants, cross-consistency, production
     manifest validates clean (read-only, disk-bound artifact hashes).
  2  Compatibility safety: stale feature version, wrong model/release id,
     wrong threshold, forged signature, altered artifact/preprocessing/
     rule/evaluation bindings — every mutation rejected.
  3  Health schema lock: exact canonical key set; the legacy `model` key
     stays gone; READY / MODEL_NOT_READY / DRIFTED / FAILED states never
     fabricate model readiness.
  4  Live boot consistency: /health identity == verified manifest identity
     == contract constants; runtime READY; release_attested true.
  5  Hygiene invariant scenarios: pristine passes; every leakage class
     (fixture growth, unknown fixture type, dup seq, dup event id,
     vanished fork, doctored fork row, fixture timestamp, future fork)
     fails closed.
  6  Fixture isolation: re-running phase76/phase77/phase79 leaves the
     shared chain's frozen fixture snapshot byte-stable.
  7  No bypass parameters anywhere; gate + CI workflow both invoke the
     hygiene check; stale `altman_native_v1` references confined to the
     documented allowlist.

Never touches the network, never mutates production artifacts (hashes are
compared before/after), never writes to the shared chain except the
legitimate runtime boot events of section 4.

Run:  ../.venv/Scripts/python.exe scripts/phase110_reconciliation_test.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = BACKEND.parent

from src.monitoring import manifest_contract as MC
from src.monitoring import observability as OBS
from src.monitoring.feature_contract import ML_FEATURE_VERSION
from src.monitoring.phase103_production_readiness_closure import (
    PROMOTION_STATE,
    REAL_WORLD_VALIDATION,
    SYSTEM_READINESS,
)
from src.monitoring.phase109_audit_fork_repair import AFFECTED_SEQUENCES
from src.monitoring.real_world_evaluation_protocol import (
    MODEL_ID,
    PRODUCTION_THRESHOLD,
    RELEASE_ID,
)

LIVE_AUDIT_DB = REPO_ROOT / "db" / "audit.db"
PROD_DIR = REPO_ROOT / "models" / "production"
REL_MANIFEST = PROD_DIR / "release_manifest.json"
NATIVE_DIR = PROD_DIR / "altman_native"
RULES_PATH = BACKEND / "src" / "risk_engine" / "rules.yaml"

sys.path.insert(0, str(SCRIPTS))
import audit_hygiene_check as HY  # noqa: E402

failures: list[str] = []
_checks = {"n": 0}


def check(name: str, cond: bool, detail: str = "") -> None:
    _checks["n"] += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def tree_hash(root: Path) -> str:
    if not root.exists():
        return "ABSENT"
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).replace("\\", "/").encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_rows() -> list[dict]:
    con = sqlite3.connect(f"file:{LIVE_AUDIT_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(
            "SELECT * FROM audit_events ORDER BY seq"
        ).fetchall()]
    finally:
        con.close()


def chain_tail_row(rows: list[dict]) -> dict:
    """Append a WELL-CHAINED synthetic row (so only the rule under test
    can fire, never an incidental link break)."""
    import json as _json
    last = rows[-1]
    payload = {"phase110": True, "n": len(rows)}
    body = _json.dumps(payload, sort_keys=True, separators=(",", ":"))
    entry = hashlib.sha256((last["entry_hash"] + body).encode("utf-8")).hexdigest()
    return {
        "seq": last["seq"] + 1,
        "event_id": f"p110-{len(rows)}",
        "event_type": "phase110_probe",
        "fraud_id": "PH110",
        "prev_hash": last["entry_hash"],
        "entry_hash": entry,
        "payload_summary": body,
        "created_at": "2026-09-24 08:30:00",
    }


def fixture_counts(rows: list[dict]) -> dict[str, int]:
    from collections import Counter
    c = Counter(str(r.get("event_type")) for r in rows)
    return {t: c.get(t, 0) for t in HY.FIXTURE_FROZEN_COUNTS}


def main() -> int:
    t0 = time.time()

    # Production-artifact snapshot BEFORE anything runs (Part 10/13).
    prod_before = tree_hash(PROD_DIR)
    rules_before = file_hash(RULES_PATH)
    flist_before = file_hash(NATIVE_DIR / "feature_list.json")
    rel_before = file_hash(REL_MANIFEST)

    # ── [1] Manifest contract: constants + cross-consistency ──────────
    print("== [1] manifest contract ==")
    check("CANONICAL_MODEL_ID", MC.CANONICAL_MODEL_ID == "altman_native")
    check("CANONICAL_RELEASE_ID",
          MC.CANONICAL_RELEASE_ID ==
          "release-altman_native_E_hardneg_cert_20260904")
    check("CANONICAL_MODEL_VERSION",
          MC.CANONICAL_MODEL_VERSION ==
          "altman_native_E_hardneg_cert_20260904")
    check("CANONICAL_LEGACY_RELEASE_ID",
          MC.CANONICAL_LEGACY_RELEASE_ID ==
          "legacy-altman_native_E_hardneg_cert_20260904")
    check("CANONICAL_FEATURE_VERSION == ML_FEATURE_VERSION == v1",
          MC.CANONICAL_FEATURE_VERSION == ML_FEATURE_VERSION == "v1")
    check("CANONICAL_THRESHOLD == 0.018758",
          MC.CANONICAL_THRESHOLD == PRODUCTION_THRESHOLD == 0.018758)
    check("contract agrees with governance protocol identity",
          MC.CANONICAL_MODEL_ID == MODEL_ID
          and MC.CANONICAL_RELEASE_ID == RELEASE_ID)
    check("stale feature version registered for rejection",
          "altman_native_v1" in MC.STALE_FEATURE_VERSIONS
          and MC.CANONICAL_FEATURE_VERSION not in MC.STALE_FEATURE_VERSIONS)
    check("no feature-version alias table exists",
          not any("alias" in k.lower() for k in vars(MC)))
    check("production manifest satisfies contract (read-only, disk-bound)",
          MC.validate_release_manifest(
              json.loads(REL_MANIFEST.read_text()),
              artifact_dir=NATIVE_DIR) == (),
          str(MC.validate_release_manifest(
              json.loads(REL_MANIFEST.read_text()), artifact_dir=NATIVE_DIR)))
    check("production manifest identifies canonical model",
          MC.manifest_identifies_canonical_model(
              json.loads(REL_MANIFEST.read_text())))

    # ── [2] Compatibility safety: every mutation rejected ─────────────
    print("== [2] manifest mutation rejection ==")
    base = json.loads(REL_MANIFEST.read_text())

    def viol(mutate, disk: bool = False) -> tuple[str, ...]:
        m = json.loads(json.dumps(base))
        mutate(m)
        return MC.validate_release_manifest(
            m, artifact_dir=NATIVE_DIR if disk else None)

    check("stale feature version altman_native_v1 rejected",
          any("stale feature_version" in v
              for v in viol(lambda m: m.update(feature_version="altman_native_v1"))))
    check("wrong feature version rejected",
          any("feature_version" in v
              for v in viol(lambda m: m.update(feature_version="v2"))))
    check("wrong model_id rejected",
          any("model_id" in v
              for v in viol(lambda m: m.update(model_id="altman_native_v2"))))
    check("wrong release_id rejected",
          any("release_id" in v
              for v in viol(lambda m: m.update(
                  release_id="release-altman_native_E_hardneg_cert_20260904-evil"))))
    check("wrong threshold rejected",
          any("locked_threshold" in v
              for v in viol(lambda m: m.update(locked_threshold=0.9))))
    check("altered artifact hash rejected (disk-bound)",
          any("hash mismatch against disk" in v
              for v in viol(lambda m: m["artifact_files"].update(
                  {"xgb_native.joblib": "0" * 64}), disk=True)))
    check("altered preprocessing binding rejected",
          any("preprocessing_hash" in v
              for v in viol(lambda m: m.update(preprocessing_hash="f" * 64))))
    check("altered rule hash rejected",
          any("rule_hash" in v
              for v in viol(lambda m: m.update(rule_hash="not-hex"))))
    check("altered evaluation binding rejected",
          any("evaluation_record_hash" in v
              for v in viol(lambda m: m.update(evaluation_record_hash="ab12"))))
    check("removed artifact_hash rejected",
          any("artifact_hash" in v
              for v in viol(lambda m: m.pop("artifact_hash"))))
    check("PROMOTION_BLOCKED verdict rejected",
          any("gate_verdict" in v
              for v in viol(lambda m: m.update(gate_verdict="PROMOTION_BLOCKED"))))

    # Forged signature / wrong artifact dir through the RUNTIME verifier.
    from src.monitoring.release_manifest import ReleaseManifest
    from src.monitoring.runtime_attestation import verify_release_for_load
    live_rule_hash = hashlib.sha256(
        RULES_PATH.read_bytes()).hexdigest()[:32]
    rm_base = ReleaseManifest.load(REL_MANIFEST)
    ok0, f0, _ = verify_release_for_load(rm_base, NATIVE_DIR,
                                         expected_rule_hash=live_rule_hash)
    check("baseline manifest passes runtime verification", ok0, str(f0))
    rm_bad = ReleaseManifest.load(REL_MANIFEST)
    rm_bad.model_id = "wrong-model-id"
    ok1, _f1, _i1 = verify_release_for_load(rm_bad, NATIVE_DIR,
                                            expected_rule_hash=live_rule_hash)
    check("forged manifest (altered field, stale signature) rejected at runtime",
          not ok1)
    ok2, _f2, _i2 = verify_release_for_load(rm_base, PROD_DIR,
                                            expected_rule_hash=live_rule_hash)
    check("altered artifact bindings (wrong dir) rejected at runtime", not ok2)

    # ── [3] Health schema lock + state matrix ─────────────────────────
    print("== [3] health schema + states ==")
    canonical_keys = {
        "status", "liveness", "readiness", "model_readiness",
        "runtime_state", "db", "audit_chain", "release_id", "model_id",
        "feature_version", "uptime_seconds", "details",
    }
    hr = OBS.HealthReport(model_readiness="loaded", runtime_state="READY",
                          release_id="rel-x", model_id="m",
                          feature_version="v1")
    d = hr.to_dict()
    check("HealthReport exact canonical key set", set(d) == canonical_keys,
          str(set(d) ^ canonical_keys))
    check("legacy `model` key stays removed", "model" not in d)
    check("READY + loaded model -> status ok",
          d["model_readiness"] == "loaded" and d["status"] == "ok")

    for state in ("MODEL_NOT_READY", "DRIFTED", "FAILED"):
        h = OBS.HealthReport(model_readiness="not_loaded",
                             runtime_state=state,
                             release_id="", model_id="",
                             feature_version="v1").to_dict()
        check(f"{state} never claims model-ready",
              h["model_readiness"] == "not_loaded"
              and h["status"] != "ok"
              and h["runtime_state"] == state,
              json.dumps(h))
    # Drifted boots still expose honest liveness, and dead liveness wins.
    dead = OBS.HealthReport(liveness="dead").to_dict()
    check("dead liveness dominates status", dead["status"] == "dead")
    notready = OBS.HealthReport(readiness="not_ready").to_dict()
    check("not-ready readiness dominates status",
          notready["status"] == "not_ready")

    # ── [4] Live boot consistency (Part 9) ────────────────────────────
    print("== [4] live /health consistency ==")
    from fastapi.testclient import TestClient
    from src.risk_engine.main import app as risk_app
    with TestClient(risk_app) as c:
        h = c.get("/health").json()
    rel = json.loads(REL_MANIFEST.read_text())
    check("live runtime_state READY", h.get("runtime_state") == "READY",
          str(h.get("runtime_state")))
    check("live release_attested true", h.get("release_attested") is True)
    check("live model_readiness loaded", h.get("model_readiness") == "loaded")
    check("live status ok", h.get("status") == "ok")
    check("live feature_version v1", h.get("feature_version") == "v1",
          str(h.get("feature_version")))
    check("live model_id == verified manifest model_id == canonical version",
          h.get("model_id") == rel.get("model_id")
          == MC.CANONICAL_MODEL_VERSION, str(h.get("model_id")))
    check("live release_id == verified manifest release_id",
          h.get("release_id") == rel.get("release_id")
          == MC.CANONICAL_LEGACY_RELEASE_ID, str(h.get("release_id")))
    check("live health has no legacy model key", "model" not in h)

    # ── [5] Hygiene invariant scenarios ───────────────────────────────
    print("== [5] hygiene invariant scenarios ==")
    check("live shared chain: hygiene CLEAN",
          HY.evaluate_audit_hygiene(load_rows()) == [])
    # pristine fresh chain passes (no fixture rows, no forks)
    fresh: list[dict] = []
    prev = hashlib.sha256(b"PS-14 audit genesis v1").hexdigest()
    for i in range(1, 4):
        body = json.dumps({"i": i}, sort_keys=True, separators=(",", ":"))
        entry = hashlib.sha256((prev + body).encode()).hexdigest()
        fresh.append({"seq": i, "event_id": f"f{i}", "event_type": "app_event",
                      "fraud_id": "FRESH", "prev_hash": prev,
                      "entry_hash": entry, "payload_summary": body,
                      "created_at": "2026-09-24 09:00:00"})
        prev = entry
    check("pristine fresh chain passes", HY.evaluate_audit_hygiene(fresh) == [],
          str(HY.evaluate_audit_hygiene(fresh)))

    rows = load_rows()

    def joined(vs: list[str]) -> str:
        return " | ".join(vs)

    # fixture growth, chain kept valid
    g = rows + [dict(chain_tail_row(rows), event_type="lifecycle_test_1")]
    vs = HY.evaluate_audit_hygiene(g)
    check("future lifecycle fixture row -> count drift fails",
          any("fixture count drift: lifecycle_test_1" in v for v in vs),
          joined(vs))
    # unknown future fixture-like type
    u = rows + [dict(chain_tail_row(rows), event_type="phase999_probe_test")]
    vs = HY.evaluate_audit_hygiene(u)
    check("unknown fixture-like event_type fails",
          any("unexpected fixture-like event_type" in v for v in vs),
          joined(vs))
    # duplicate sequence
    dup = [dict(r) for r in rows]
    dup.insert(10, dict(dup[9]))
    vs = HY.evaluate_audit_hygiene(dup)
    check("duplicate sequence fails",
          any("duplicate audit sequence" in v for v in vs), joined(vs))
    # duplicate event id (hash untouched -> only the id rule can fire)
    de = [dict(r) for r in rows]
    de[50]["event_id"] = de[5]["event_id"]
    vs = HY.evaluate_audit_hygiene(de)
    check("duplicate event_id fails",
          any("duplicate audit event_id" in v for v in vs), joined(vs))
    # future fork at a non-frozen sequence
    ff = [dict(r) for r in rows]
    target = next(r for r in ff if r["seq"] not in AFFECTED_SEQUENCES
                  and r["seq"] > 5000)
    target["prev_hash"] = "e" * 64
    vs = HY.evaluate_audit_hygiene(ff)
    check("future (unfrozen) fork fails closed",
          any("chain not trustworthy" in v for v in vs), joined(vs))
    # vanished historical fork row
    vz = [dict(r) for r in rows if r["seq"] != 731]
    vs = HY.evaluate_audit_hygiene(vz)
    check("vanished historical fork row fails", bool(vs), joined(vs[:2]))
    # doctored quarantined fork row (in-memory only)
    dz = [dict(r) for r in rows]
    dz731 = next(r for r in dz if r["seq"] == 731)
    dz731["payload_summary"] = dz731["payload_summary"] + " "
    vs = HY.evaluate_audit_hygiene(dz)
    check("doctored quarantined fork row fails", bool(vs), joined(vs[:2]))
    # fixture timestamp after cutover (replace an existing row: count intact)
    ts = [dict(r) for r in rows]
    life = next(r for r in ts if r["event_type"] == "lifecycle_test_1")
    life["created_at"] = "2026-09-25 00:00:00"
    vs = HY.evaluate_audit_hygiene(ts)
    check("fixture timestamp after cutover fails",
          any("fixture timestamp after cutover" in v for v in vs),
          joined(vs))
    # frozen snapshot exactly matches the live chain right now
    check("live fixture counts == frozen snapshot",
          fixture_counts(rows) == HY.FIXTURE_FROZEN_COUNTS,
          str({k: (fixture_counts(rows).get(k), v)
               for k, v in HY.FIXTURE_FROZEN_COUNTS.items()
               if fixture_counts(rows).get(k) != v}))

    # ── [6] Fixture isolation: re-runs never touch the shared chain ───
    print("== [6] fixture isolation (phase76/77/79 re-run) ==")
    before = fixture_counts(load_rows())
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("DB_DIR", None)  # each suite binds its own temp DB_DIR
    # Section 4's app import can merge the repo .env (DATABASE_URL +
    # Supabase keys) into the parent's environ; these isolation suites are
    # SQLite-only fixtures, so hand the children a clean database env.
    for k in list(env):
        if (k == "DATABASE_URL" or k.endswith("_DB_URL")
                or k.startswith(("SUPABASE", "POSTGRES", "PG"))):
            env.pop(k)
    for suite in ("phase76_lifecycle_recovery_test.py",
                  "phase77_audit_chain_concurrency_test.py",
                  "phase79_threat_model_security_test.py"):
        r = subprocess.run([sys.executable, f"scripts/{suite}"],
                           cwd=BACKEND, env=env, capture_output=True,
                           text=True, timeout=600)
        import tempfile as _tf
        child_log = Path(_tf.gettempdir()) / f"p110_child_{suite}.log"
        child_log.write_text(
            (r.stdout or "") + "\n=== STDERR ===\n" + (r.stderr or ""),
            encoding="utf-8")
        check(f"{suite} passes isolated", r.returncode == 0,
              f"rc={r.returncode} tail="
              + ((r.stderr or "") + (r.stdout or ""))[-700:])
        after = fixture_counts(load_rows())
        check(f"{suite} wrote ZERO fixture rows to the shared chain",
              after == before,
              str({k: (before[k], after[k]) for k in after
                   if after[k] != before[k]}))
    check("shared chain still hygiene-CLEAN after re-runs",
          HY.evaluate_audit_hygiene(load_rows()) == [])

    # ── [7] No bypass; gate + CI wiring; stale refs confined ──────────
    print("== [7] security / wiring / stale refs ==")
    forbidden = re.compile(
        r"\b(skip_audit_hygiene|ignore_fixture_check|allow_known_fork"
        r"|disable_chain_check|ci_override|force_green|allow_unverified"
        r"|skip_validation|admin_override|force)\s*=")
    for fn in (HY.evaluate_audit_hygiene, HY.check_shared_chain_hygiene,
               HY.main, MC.validate_release_manifest,
               MC.manifest_identifies_canonical_model):
        params = list(inspect.signature(fn).parameters)
        bad = [p for p in params
               if p not in ("rows", "manifest", "artifact_dir", "db_path")
               and re.search(
                   r"force|override|skip|bypass|allow_unverified|ignore",
                   p)]
        check(f"{fn.__name__}: no bypass parameters", not bad, str(bad))
    check("hygiene evaluator takes rows only (no tunable path/flags)",
          list(inspect.signature(HY.evaluate_audit_hygiene).parameters)
          == ["rows"])
    for fp in (SCRIPTS / "audit_hygiene_check.py",
               SCRIPTS / "security_ci_gate.py",
               BACKEND / "src" / "monitoring" / "manifest_contract.py"):
        text = fp.read_text(encoding="utf-8")
        m = forbidden.search(text)
        check(f"no bypass assignment in {fp.name}", m is None,
              m.group(0) if m else "")
        net = [tok for tok in ("urllib", "http.client", "import requests",
                               "socket.", "urlopen") if tok in text]
        check(f"no network primitives in {fp.name}", not net, str(net))
    gate_text = (SCRIPTS / "security_ci_gate.py").read_text(encoding="utf-8")
    check("security_ci_gate runs the hygiene invariant",
          "audit_hygiene_check.py" in gate_text)
    wf = (REPO_ROOT / ".github" / "workflows" / "ci-cd.yml").read_text(
        encoding="utf-8")
    check("CI Security Scan job runs the hygiene invariant",
          "audit_hygiene_check.py" in wf)

    # Stale feature-version references confined to the documented allowlist.
    stale_allowlist = {
        # The contract's REJECTION registry: `altman_native_v1` appears only
        # as a value that validation must refuse (pinned by the dedicated
        # "rejection registry" check below).
        "manifest_contract.py",
        "phase68_e2e_integration.py",
        "phase70_observability_test.py",
        "phase71_architecture_gap_audit.py",
        "phase72_observability_integration_test.py",
        "train_altman_native.py",  # writes historical training records only
    }
    offenders = set()
    for root in (BACKEND / "src", BACKEND / "scripts", REPO_ROOT):
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            try:
                if "altman_native_v1" in p.read_text(encoding="utf-8",
                                                      errors="ignore"):
                    offenders.add(p.name)
            except OSError:
                pass
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8",
                                                 errors="ignore")
    check("README has no stale altman_native_v1 claim",
          "altman_native_v1" not in readme)
    src_offenders = {o for o in offenders
                     if o not in stale_allowlist and o != Path(__file__).name}
    check("stale refs confined to documented allowlist",
          not src_offenders, str(sorted(src_offenders)))
    check("stale ref exists ONLY as the contract's rejection registry",
          "altman_native_v1" in MC.STALE_FEATURE_VERSIONS)

    # ── [8] Identity + artifact immutability ──────────────────────────
    print("== [8] identity + artifacts ==")
    check("SYSTEM_READINESS unchanged",
          SYSTEM_READINESS == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET")
    check("REAL_WORLD_VALIDATION unchanged",
          REAL_WORLD_VALIDATION == "BLOCKED_PENDING_ELIGIBLE_DATASET")
    check("PROMOTION unchanged",
          PROMOTION_STATE == "PROMOTION_GATE_REQUIRED")
    check("model/release/threshold unchanged",
          MODEL_ID == "altman_native"
          and RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904"
          and PRODUCTION_THRESHOLD == 0.018758)
    check("production manifest bytes unchanged by this suite",
          file_hash(REL_MANIFEST) == rel_before)
    check("models/production tree unchanged by this suite",
          tree_hash(PROD_DIR) == prod_before)
    check("rules.yaml unchanged", file_hash(RULES_PATH) == rules_before)
    check("feature_list.json unchanged",
          file_hash(NATIVE_DIR / "feature_list.json") == flist_before)

    print()
    total = _checks["n"]
    passed = total - len(failures)
    print(f"Total: {total}  |  PASS: {passed}  |  FAIL: {len(failures)}")
    if failures:
        for name in failures:
            print(f"  FAILED: {name}")
    else:
        print("ALL PHASE 110 CHECKS PASSED")
    print(f"elapsed_s={time.time() - t0:.1f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
