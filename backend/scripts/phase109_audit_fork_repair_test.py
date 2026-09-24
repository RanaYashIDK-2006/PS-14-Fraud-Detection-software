"""Phase 109: Audit-chain fork forensic repair — test suite.

Deterministic, offline coverage for the Phase 109 repair:
  1  Forensic re-derivation matches the frozen findings (read-only).
  2  Findings are immutable and self-validating.
  3  evaluate_chain: live chain is trustworthy under evidence-bound
     quarantine while the strict verdict stays False (no weakening).
  4  Tamper battery: every attack fails closed (synthetic rows only).
  5  Quarantine is evidence-bound (in-memory drift detection only —
     the live DB is never written by this suite).
  6  verify_chain (strict authority) is unchanged and still fails
     the live chain.
  7  Concurrency: legacy stale-max race REPRODUCES the fork; the
     repaired writer matrix (1/2/5/10 processes) stays strict-valid;
     crash rolls back atomically; contention retries land.
  8  Fixture isolation: phase76 runs on a temp DB; the shared chain's
     lifecycle-row count does not move.
  9  backup/restore suite passes (incl. corrupted-backup rejection).
 10  No bypass parameters or repair endpoints exist anywhere.
 11  Manifest/report are deterministic and cannot claim RWV/promotion.
 12  Production identity, threshold and global state are unchanged;
     the models/ trees hash identically before and after.

Run:  ../.venv/Scripts/python.exe scripts/phase109_audit_fork_repair_test.py
Worker modes (--init-worker/--legacy-worker/--repaired-worker/
--crash-worker) are dispatched BEFORE any src import so a fresh
interpreter can bind DB_DIR to a temporary database.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
REPO_ROOT = os.path.dirname(BACKEND)
PY = sys.executable
GENESIS = hashlib.sha256(b"PS-14 audit genesis v1").hexdigest()
LIVE_LIFECYCLE_ROWS = 75  # frozen 2026-09-19 fixture pollution (Part 7)

WORKER_MODES = {
    "--init-worker",
    "--legacy-worker",
    "--repaired-worker",
    "--crash-worker",
}


# ── Worker implementations (stdlib only; src imported lazily) ─────────
def _canon_local(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def _w_init(dbdir: str) -> int:
    os.environ["DB_DIR"] = dbdir
    if BACKEND not in sys.path:
        sys.path.insert(0, BACKEND)
    from src.audit_service.db import engine
    from src.audit_service.models import Base

    Base.metadata.create_all(bind=engine)
    return 0


def _w_legacy(db: str, n: int) -> int:
    """The pre-Phase-109 algorithm: read max (no reservation) -> gap -> insert."""
    con = sqlite3.connect(db, timeout=10)
    wid = os.getpid()
    for i in range(n):
        cur = con.execute(
            "SELECT entry_hash FROM audit_events ORDER BY seq DESC LIMIT 1"
        )
        row = cur.fetchone()
        prev = row[0] if row else GENESIS
        payload = {"w": wid, "i": i}
        entry = hashlib.sha256((prev + _canon_local(payload)).encode()).hexdigest()
        time.sleep(0.005)  # the stale-read window (query -> hash -> insert)
        con.execute(
            "INSERT INTO audit_events (event_id, fraud_id, event_type,"
            " prev_hash, entry_hash, payload_summary)"
            " VALUES (?, 'RACE', 'legacy_race', ?, ?, ?)",
            (f"{wid}-{i}", prev, entry, json.dumps(payload)),
        )
        con.commit()
    con.close()
    return 0


def _w_repaired(dbdir: str, n: int, tag: str) -> int:
    os.environ["DB_DIR"] = dbdir
    if BACKEND not in sys.path:
        sys.path.insert(0, BACKEND)
    from src.audit_service.writer import append_audit_event, flush_audit_queue

    for i in range(n):
        append_audit_event(f"REPAIRED-{tag}", "phase109_repro", {"tag": tag, "i": i})
    flush_audit_queue(timeout=15.0)
    # flush()'s in-flight grace is not deterministic under contention —
    # poll until this worker's rows have actually landed.
    audit_file = str(Path(dbdir) / "audit.db")
    deadline = time.time() + 20
    while time.time() < deadline:
        con = sqlite3.connect(f"file:{audit_file}?mode=ro", uri=True)
        cnt = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE fraud_id = ?",
            (f"REPAIRED-{tag}",),
        ).fetchone()[0]
        con.close()
        if cnt >= n:
            return 0
        time.sleep(0.05)
    return 3


def _w_crash(dbdir: str) -> int:
    """Open an IMMEDIATE tx, insert a chained row, die without committing."""
    con = sqlite3.connect(str(Path(dbdir) / "audit.db"), timeout=10)
    con.isolation_level = None
    con.execute("BEGIN IMMEDIATE")
    row = con.execute(
        "SELECT entry_hash FROM audit_events ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    prev = row[0] if row else GENESIS
    payload = {"crash": True}
    entry = hashlib.sha256((prev + _canon_local(payload)).encode()).hexdigest()
    con.execute(
        "INSERT INTO audit_events (event_id, fraud_id, event_type,"
        " prev_hash, entry_hash, payload_summary)"
        " VALUES ('crash-row', 'CRASH', 'crash_probe', ?, ?, ?)",
        (prev, entry, json.dumps(payload)),
    )
    os._exit(1)  # no COMMIT — the reservation must roll back atomically


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in WORKER_MODES:
    # Runs BEFORE any src import so DB_DIR binds to the temp database.
    _mode = sys.argv[1]
    if _mode == "--init-worker":
        raise SystemExit(_w_init(sys.argv[2]))
    if _mode == "--legacy-worker":
        raise SystemExit(_w_legacy(sys.argv[2], int(sys.argv[3])))
    if _mode == "--repaired-worker":
        raise SystemExit(_w_repaired(sys.argv[2], int(sys.argv[3]), sys.argv[4]))
    if _mode == "--crash-worker":
        raise SystemExit(_w_crash(sys.argv[2]))

# ── Parent: normal test run (safe to import src below this line) ──────
from src.audit_service.writer import canonical, verify_chain  # noqa: E402
from src.monitoring import phase109_audit_fork_repair as P109  # noqa: E402
from src.monitoring import phase109_audit_fork_report as REP  # noqa: E402
from src.monitoring.phase103_production_readiness_closure import (  # noqa: E402
    PROMOTION_STATE,
    REAL_WORLD_VALIDATION,
    SYSTEM_READINESS,
)
from src.monitoring.real_world_evaluation_protocol import (  # noqa: E402
    MODEL_ID,
    PRODUCTION_THRESHOLD,
    RELEASE_ID,
)

LIVE_DB = Path(REPO_ROOT) / "db" / "audit.db"
failures: list[str] = []
_checks = {"n": 0}


def check(name: str, cond: bool, detail: str = "") -> None:
    _checks["n"] += 1
    status = "PASS" if cond else "FAIL"
    line = f"  [{status}] {name}"
    if detail and (not cond or len(detail) < 120):
        line += f"  ({detail})"
    print(line)
    if not cond:
        failures.append(name)


def spawn(args: list[str], dbdir: str) -> subprocess.Popen:
    env = dict(os.environ, DB_DIR=dbdir, PYTHONIOENCODING="utf-8")
    return subprocess.Popen(
        [PY, os.path.abspath(__file__)] + args,
        env=env, cwd=BACKEND,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def link_breaks(db: Path) -> list[int]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM audit_events ORDER BY seq").fetchall()
    con.close()
    breaks, prev = [], GENESIS
    for r in rows:
        if r["prev_hash"] != prev:
            breaks.append(r["seq"])
        prev = r["entry_hash"]
    return breaks


def strict_verify(db: Path) -> tuple[bool, int | None]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM audit_events ORDER BY seq").fetchall()
    con.close()
    prev = GENESIS
    for r in rows:
        try:
            body = canonical(json.loads(r["payload_summary"]))
        except Exception:
            return False, r["seq"]
        recomputed = hashlib.sha256((prev + body).encode()).hexdigest()
        if r["prev_hash"] != prev or r["entry_hash"] != recomputed:
            return False, r["seq"]
        prev = r["entry_hash"]
    return True, None


def live_rows() -> list[dict]:
    con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM audit_events ORDER BY seq"
    ).fetchall()]
    con.close()
    return rows


def make_chain(n: int = 6, start_seq: int = 1) -> list[dict]:
    rows: list[dict] = []
    prev = GENESIS
    for i in range(start_seq, start_seq + n):
        body = canonical({"i": i, "probe": "phase109"})
        entry = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
        rows.append({
            "seq": i,
            "event_id": f"ev{i}",
            "event_type": "p109_probe",
            "fraud_id": "P109-TEST",
            "prev_hash": prev,
            "entry_hash": entry,
            "payload_summary": body,
            "created_at": "2026-09-24 00:00:00",
        })
        prev = entry
    return rows


def tree_hash(root: Path) -> str:
    if not root.exists():
        return "ABSENT"
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root)).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    # Production trees hashed BEFORE anything heavy runs (Part 12/15).
    prod_before = tree_hash(Path(REPO_ROOT) / "models" / "production")
    art_before = tree_hash(Path(REPO_ROOT) / "models" / "artifacts")

    # ── [1] Forensic re-derivation vs frozen findings ─────────────────
    print("== [1] forensic re-derivation (read-only) ==")
    check("live audit.db exists", LIVE_DB.exists(), str(LIVE_DB))
    derived = P109.derive_findings_from_db(str(LIVE_DB))
    by_seq = {d["seq"]: d for d in derived}
    check("exactly five link breaks on disk",
          sorted(by_seq) == list(P109.AFFECTED_SEQUENCES),
          str(sorted(by_seq)))
    field_map = [
        ("expected", "expected_previous_hash"),
        ("actual", "actual_previous_hash"),
        ("pred_event_id", "pred_event_id"),
        ("offender_event_id", "offender_event_id"),
        ("pred_ts", "pred_ts"),
        ("offender_ts", "offender_ts"),
        ("event_type", "event_type"),
        ("fraud_id", "fraud_id"),
        ("entry", "entry"),
        ("payload_sha", "payload_sha"),
    ]
    for raw in P109._RAW_FINDINGS:
        d = by_seq.get(raw["seq"])
        bad = [
            rk for rk, dk in field_map
            if d is None or str(d.get(dk)) != str(raw[rk])
        ]
        check(f"seq {raw['seq']}: derived evidence == frozen finding",
              not bad, f"mismatched {bad}")
    check("classification primary = CONCURRENT_WRITER_ORDERING",
          P109.ROOT_CAUSE_PRIMARY == "CONCURRENT_WRITER_ORDERING")
    check("classification secondary includes TEST_FIXTURE_POLLUTION",
          "TEST_FIXTURE_POLLUTION" in P109.ROOT_CAUSE_SECONDARY)

    # ── [2] Findings immutable + self-validating ──────────────────────
    print("== [2] finding immutability ==")
    f0 = P109.PRIMARY_FINDING
    check("dataclass is frozen", hasattr(P109.AuditForkFinding, "__dataclass_fields__")
          and P109.AuditForkFinding.__dataclass_params__.frozen)
    try:
        f0.first_bad_sequence = 1  # type: ignore[misc]
        mutated = True
    except dataclasses.FrozenInstanceError:
        mutated = False
    check("field mutation raises FrozenInstanceError", not mutated)
    check("evidence_hash is 64-hex and non-empty",
          bool(re.fullmatch(r"[0-9a-f]{64}", f0.evidence_hash)))
    try:
        dataclasses.replace(f0, evidence_hash="0" * 64)
        bad_hash_rejected = False
    except ValueError:
        bad_hash_rejected = True
    check("forged evidence_hash rejected at construction", bad_hash_rejected)
    recomputed = P109._evidence_digest(f0.binding_fields())
    check("evidence_hash reproducible from binding fields",
          recomputed == f0.evidence_hash)

    # ── [3] evaluate_chain over the live chain ────────────────────────
    print("== [3] evaluate_chain (live) ==")
    rows = live_rows()
    res = P109.evaluate_chain(rows)
    check("live chain trustworthy (ok=True)", res["ok"], str(res["reason"]))
    check("strict verdict still False (no weakening)", res["strict_ok"] is False)
    check("first_bad_seq == 731", res["first_bad_seq"] == 731,
          str(res["first_bad_seq"]))
    check("all five forks quarantined",
          res["quarantined_breaks"] == list(P109.AFFECTED_SEQUENCES),
          str(res["quarantined_breaks"]))
    check("finding ids recorded",
          res["finding_ids"] == sorted(f"P109-FORK-{s}" for s in P109.AFFECTED_SEQUENCES),
          str(res["finding_ids"]))
    check("n_entries == row count", res["n_entries"] == len(rows))

    # ── [4] Tamper battery (synthetic rows only) ──────────────────────
    print("== [4] tamper battery (synthetic) ==")
    clean = make_chain()
    r = P109.evaluate_chain(clean)
    check("clean synthetic chain: ok & strict_ok", r["ok"] and r["strict_ok"])

    tampered = make_chain()
    tampered[2]["payload_summary"] = canonical({"i": 999})
    r = P109.evaluate_chain(tampered)
    check("modified payload -> fail closed (self-consistency)",
          not r["ok"] and "self-consistency" in (r["reason"] or ""),
          str(r["reason"]))

    forged = make_chain()
    forged[4]["prev_hash"] = "f" * 64
    forged = rechain_keep_prev(forged, from_index=4)
    r = P109.evaluate_chain(forged)
    check("forged prev_hash (non-731) -> fail closed (unquarantined)",
          not r["ok"] and "unquarantined" in (r["reason"] or ""),
          str(r["reason"]))

    genesis_forged = make_chain()
    genesis_forged[0]["prev_hash"] = "a" * 64
    genesis_forged = rechain_keep_prev(genesis_forged, from_index=0)
    r = P109.evaluate_chain(genesis_forged)
    check("forged genesis link -> fail closed",
          not r["ok"], str(r["reason"]))

    ev = make_chain(n=2, start_seq=730)  # rows 730, 731
    ev[1]["prev_hash"] = "0" * 64
    ev = rechain_keep_prev(ev, from_index=1)
    r = P109.evaluate_chain(ev)
    check("break AT seq 731 with wrong evidence -> fail closed (quarantine is evidence-bound)",
          not r["ok"] and r["first_bad_seq"] == 731,
          str(r["reason"]))

    # The genuine quarantined row passes its own binding:
    live731 = next(x for x in rows if x["seq"] == 731)
    check("genuine seq-731 row matches its frozen finding",
          P109.FINDING_BY_SEQ[731].matches_row(live731))

    dropped = make_chain(start_seq=730)
    del dropped[1]
    r = P109.evaluate_chain(dropped)
    check("deleted row -> fail closed (gap break)", not r["ok"], str(r["reason"]))

    reordered = list(reversed(make_chain()))
    r = P109.evaluate_chain(reordered)
    check("reordered rows -> fail closed", not r["ok"], str(r["reason"]))

    dup = make_chain()
    dup.insert(2, dict(dup[1]))
    r = P109.evaluate_chain(dup)
    check("duplicate seq row -> fail closed", not r["ok"], str(r["reason"]))

    # ── [5] Evidence binding: in-memory drift only (disk untouched) ───
    print("== [5] quarantine evidence binding (in-memory) ==")
    drift = [dict(x) for x in rows]
    for d in drift:
        if d["seq"] == 731:
            d["payload_summary"] = d["payload_summary"] + " "
    r = P109.evaluate_chain(drift)
    check("drift on quarantined row -> quarantine void, fail closed",
          not r["ok"], str(r["reason"]))
    drift2 = [dict(x) for x in rows]
    normal = next(x for x in drift2 if x["seq"] not in P109.AFFECTED_SEQUENCES)
    body = json.loads(normal["payload_summary"])
    body["drifted"] = True
    normal["payload_summary"] = canonical(body)
    r = P109.evaluate_chain(drift2)
    check("drift on a normal row's payload -> fail closed", not r["ok"],
          str(r["reason"]))
    check("live DB still byte-stable after in-memory tamper tests",
          P109.evaluate_chain(live_rows())["ok"])

    # ── [6] Strict authority unchanged ────────────────────────────────
    print("== [6] verify_chain strict authority ==")
    ns_rows = [
        SimpleNamespace(seq=x["seq"], event_id=x["event_id"],
                        prev_hash=x["prev_hash"], entry_hash=x["entry_hash"],
                        payload_summary=x["payload_summary"])
        for x in rows
    ]
    strict_live = verify_chain(ns_rows)
    check("verify_chain STILL fails the live chain at 731 (unchanged, not weakened)",
          strict_live.get("ok") is False and strict_live.get("first_bad_seq") == 731,
          json.dumps(strict_live, default=str))
    ns_clean = [
        SimpleNamespace(seq=x["seq"], event_id=x["event_id"],
                        prev_hash=x["prev_hash"], entry_hash=x["entry_hash"],
                        payload_summary=x["payload_summary"])
        for x in make_chain()
    ]
    check("verify_chain passes a clean chain", verify_chain(ns_clean).get("ok") is True)

    # ── [7] Concurrency reproduction + repaired matrix ────────────────
    print("== [7] concurrency: legacy race + repaired writer matrix ==")
    tmp = Path(tempfile.mkdtemp(prefix="ps14_p109_"))

    legacy_db_dir = tmp / "legacy"
    legacy_db_dir.mkdir()
    p = spawn(["--init-worker", str(legacy_db_dir)], str(legacy_db_dir))
    p.wait(timeout=60)
    legacy_db = legacy_db_dir / "audit.db"
    p1 = spawn(["--legacy-worker", str(legacy_db), "8"], str(legacy_db_dir))
    p2 = spawn(["--legacy-worker", str(legacy_db), "8"], str(legacy_db_dir))
    p1.wait(timeout=60)
    p2.wait(timeout=60)
    breaks = link_breaks(legacy_db)
    v_ok, v_bad = strict_verify(legacy_db)
    check("legacy algorithm FORKS the chain (reproduced)",
          len(breaks) >= 1, str(breaks))
    check("strict verifier reports the first legacy break",
          (not v_ok) and v_bad == breaks[0], str(v_bad))
    con = sqlite3.connect(f"file:{legacy_db}?mode=ro", uri=True)
    shared = con.execute(
        "SELECT COUNT(*) FROM (SELECT prev_hash FROM audit_events"
        " GROUP BY prev_hash HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    con.close()
    check("stale-read signature: duplicate prev_hash across rows",
          shared >= 1, str(shared))

    for writers, per in ((1, 15), (2, 15), (5, 10), (10, 6)):
        dbdir = tmp / f"rep_{writers}"
        dbdir.mkdir()
        p = spawn(["--init-worker", str(dbdir)], str(dbdir))
        p.wait(timeout=60)
        procs = [
            spawn(["--repaired-worker", str(dbdir), str(per), f"w{writers}i{i}"],
                  str(dbdir))
            for i in range(writers)
        ]
        for pr in procs:
            out, _ = pr.communicate(timeout=120)
            if pr.returncode != 0:
                print((out or "")[-600:])
        db = dbdir / "audit.db"
        v_ok, v_bad = strict_verify(db)
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        n = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT seq), COUNT(DISTINCT event_id)"
            " FROM audit_events"
        ).fetchone()
        con.close()
        check(f"{writers} writer(s) x {per}: strict chain VALID",
              v_ok, f"bad={v_bad}")
        check(f"{writers} writer(s): {n[0]} rows, unique seq & event_id",
              n[0] == writers * per and n[1] == n[0] and n[2] == n[0],
              str(tuple(n)))

    dbdir10 = tmp / "rep_10"
    procs = [
        spawn(["--repaired-worker", str(dbdir10), "3", f"restart{i}"], str(dbdir10))
        for i in range(2)
    ]
    for pr in procs:
        pr.communicate(timeout=60)
    v_ok, v_bad = strict_verify(dbdir10 / "audit.db")
    check("process-restart continuation chains onto prior writers",
          v_ok, f"bad={v_bad}")

    db = dbdir10 / "audit.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    before = con.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    con.close()
    p = spawn(["--crash-worker", str(dbdir10)], str(dbdir10))
    p.wait(timeout=30)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    after = con.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    con.close()
    v_ok, v_bad = strict_verify(db)
    check("crash inside IMMEDIATE tx: row rolled back atomically",
          after == before, f"{before} -> {after}")
    check("chain strict-valid after crash", v_ok, f"bad={v_bad}")

    holder = sqlite3.connect(str(db))
    holder.isolation_level = None
    holder.execute("BEGIN IMMEDIATE")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    before = con.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    con.close()
    p = spawn(["--repaired-worker", str(dbdir10), "2", "retry"], str(dbdir10))
    time.sleep(0.4)  # hold the reservation while the writer blocks
    holder.execute("COMMIT")
    holder.close()
    p.communicate(timeout=60)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    after = con.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    con.close()
    v_ok, v_bad = strict_verify(db)
    check("contention: blocked writer retried and landed both events",
          after == before + 2, f"{before} -> {after}")
    check("chain strict-valid after contention", v_ok, f"bad={v_bad}")

    # ── [8] Test-fixture isolation (Part 7) ───────────────────────────
    print("== [8] fixture isolation ==")
    lc_sql = ("SELECT COUNT(*),"
              " SUM(CASE WHEN fraud_id LIKE 'CHAIN%' THEN 1 ELSE 0 END),"
              " MIN(date(created_at)), MAX(date(created_at))"
              " FROM audit_events WHERE event_type LIKE 'lifecycle_test%'")
    con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    lc_before, lc_chain, lc_min, lc_max = con.execute(lc_sql).fetchone()
    con.close()
    check(f"live lifecycle rows frozen at {LIVE_LIFECYCLE_ROWS} (pre-run)",
          lc_before == LIVE_LIFECYCLE_ROWS, str(lc_before))
    check("every lifecycle row is a CHAIN-* test fixture (class B/C)",
          lc_chain == lc_before and lc_before > 0,
          f"{lc_chain}/{lc_before} fixture ids, {lc_min}..{lc_max}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r76 = subprocess.run(
        [PY, "scripts/phase76_lifecycle_recovery_test.py"],
        cwd=BACKEND, env=env, capture_output=True, text=True, timeout=600,
    )
    check("phase76 lifecycle suite passes with isolation",
          r76.returncode == 0,
          (r76.stdout or "")[-200:].replace("\n", " "))
    con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    lc_after = con.execute(lc_sql).fetchone()[0]
    con.close()
    check("phase76 wrote ZERO lifecycle rows to the shared chain",
          lc_after == lc_before, f"{lc_before} -> {lc_after}")
    p76 = (Path(BACKEND) / "scripts" / "phase76_lifecycle_recovery_test.py").read_text(
        encoding="utf-8"
    )
    check("phase76 source keeps temp-DB isolation (no shared DB_DIR)",
          "mkdtemp" in p76 and "DB_DIR" in p76)

    # ── [9] Backup/restore suite (incl. corruption rejection) ─────────
    print("== [9] backup/restore ==")
    rbr = subprocess.run(
        [PY, "scripts/backup_restore_test.py"],
        cwd=BACKEND, env=env, capture_output=True, text=True, timeout=600,
    )
    check("backup_restore_test passes (strict + quarantine + corruption)",
          rbr.returncode == 0 and "ALL BACKUP/RESTORE CHECKS PASSED" in (rbr.stdout or ""),
          (rbr.stdout or "")[-200:].replace("\n", " "))
    check("backup suite quarantined-fork check present",
          "Restore preserves all five quarantined forks" in (rbr.stdout or ""))

    # ── [10] Security: no bypass anywhere ─────────────────────────────
    print("== [10] security / no-bypass scans ==")
    forbidden_re = re.compile(
        r"\b(force|allow_unverified|skip_validation|admin_override"
        r"|ignore_chain|skip_verification)\s*="
    )
    param_names_bad = re.compile(
        r"^(force|allow_unverified|skip_validation|override|admin_override"
        r"|bypass|ignore_chain|skip_verification)$"
    )
    for fn in (P109.evaluate_chain, P109.derive_findings_from_db,
               REP.build_manifest, REP.generate_phase109_report,
               REP.manifest_sha256, REP.write_manifest):
        hit = [p for p in inspect.signature(fn).parameters
               if param_names_bad.match(p)]
        check(f"{fn.__name__}: no bypass parameters", not hit, str(hit))
    scan_files = [
        Path(BACKEND) / "src" / "monitoring" / "phase109_audit_fork_repair.py",
        Path(BACKEND) / "src" / "monitoring" / "phase109_audit_fork_report.py",
        Path(BACKEND) / "src" / "audit_service" / "writer.py",
        Path(BACKEND) / "src" / "audit_service" / "main.py",
    ]
    for fp in scan_files:
        text = fp.read_text(encoding="utf-8")
        m = forbidden_re.search(text)
        check(f"no bypass assignment in {fp.name}", m is None,
              m.group(0) if m else "")
    audit_main = (Path(BACKEND) / "src" / "audit_service" / "main.py").read_text(
        encoding="utf-8"
    )
    check("integrity endpoint uses evaluate_chain (quarantine-aware)",
          "evaluate_chain(rows)" in audit_main)
    check("no repair/override route added to audit service",
          not re.search(r"@(app|router)\.(get|post)\(\"[^\"]*(repair|override)",
                        audit_main))
    writer_src = (Path(BACKEND) / "src" / "audit_service" / "writer.py").read_text(
        encoding="utf-8"
    )
    check("writer takes a BEGIN IMMEDIATE reservation",
          'db.execute(text("BEGIN IMMEDIATE"))' in writer_src)
    import src.audit_service.writer as W
    check("append_audit_event signature unchanged",
          list(inspect.signature(W.append_audit_event).parameters)
          == ["fraud_id", "event_type", "payload"])
    for fp in scan_files[:2]:
        text = fp.read_text(encoding="utf-8")
        bad = [tok for tok in ("urllib", "http.client", "import requests",
                               "socket.", "urlopen")
               if tok in text]
        check(f"no network primitives in {fp.name}", not bad, str(bad))
    for fp in scan_files[:2]:
        text = fp.read_text(encoding="utf-8")
        bad = [tok for tok in ("PromotionToken", "RWVPromotionEvidence",
                               "promote(", "pickle", "joblib")
               if tok in text]
        check(f"no promotion/pickling constructs in {fp.name}", not bad, str(bad))

    # ── [11] Manifest + report determinism ────────────────────────────
    print("== [11] manifest determinism ==")
    m1 = REP.build_manifest()
    m2 = REP.build_manifest()
    check("manifest build deterministic", m1 == m2)
    check("manifest sha256 deterministic",
          REP.manifest_sha256(m1) == REP.manifest_sha256(m2)
          and bool(re.fullmatch(r"[0-9a-f]{64}", REP.manifest_sha256(m1))))
    dump = json.dumps(
        {k: v for k, v in m1.items() if k != "forbidden_result_states"},
        sort_keys=True,
    )
    check("manifest cannot claim RWV/promotion states",
          not any(s in dump for s in REP.FORBIDDEN_RESULT_STATES))
    check("manifest carries not_authorizes list",
          m1["not_authorizes"] == ["RWV", "promotion", "release creation",
                                   "model modification", "gate bypass"])
    check("manifest global state matches the authoritative state",
          m1["global_state"]["SYSTEM_READINESS"] == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET"
          and m1["global_state"]["REAL_WORLD_VALIDATION"] == "BLOCKED_PENDING_ELIGIBLE_DATASET"
          and m1["global_state"]["PROMOTION"] == "PROMOTION_GATE_REQUIRED"
          and m1["global_state"]["qualified_datasets"] == [])
    check("manifest production identity locked",
          m1["production_identity"]["model_id"] == "altman_native"
          and m1["production_identity"]["release_id"]
          == "release-altman_native_E_hardneg_cert_20260904"
          and m1["production_identity"]["production_threshold"] == 0.018758)
    rep = REP.generate_phase109_report()
    check("report state = AUDIT_CHAIN_INTEGRITY_RESTORED",
          rep.repair_state == "AUDIT_CHAIN_INTEGRITY_RESTORED")
    check("report frozen", rep.__dataclass_params__.frozen)
    try:
        dataclasses.replace(rep, manifest_sha256="0" * 64)
        forged_hash_rejected = False
    except ValueError:
        forged_hash_rejected = True
    check("forged report manifest hash rejected", forged_hash_rejected)
    with tempfile.TemporaryDirectory() as td:
        p_a = Path(td) / "a.json"
        p_b = Path(td) / "b.json"
        h_a = REP.write_manifest(str(p_a))
        h_b = REP.write_manifest(str(p_b))
        check("manifest file write deterministic",
              h_a == h_b and p_a.read_bytes() == p_b.read_bytes())
    # On-disk canonical manifest (reports/audit_repair/phase109/)
    out_dir = Path(REPO_ROOT) / "reports" / "audit_repair" / "phase109"
    out_dir.mkdir(parents=True, exist_ok=True)
    disk_h = REP.write_manifest(str(out_dir / "phase109_repair_manifest.json"))
    check("canonical manifest written with stable hash",
          disk_h == REP.manifest_sha256())

    # ── [12] Production invariants ────────────────────────────────────
    print("== [12] production invariants ==")
    check("threshold unchanged (0.018758)",
          PRODUCTION_THRESHOLD == 0.018758, str(PRODUCTION_THRESHOLD))
    check("model identity unchanged", MODEL_ID == "altman_native", MODEL_ID)
    check("release identity unchanged",
          RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904", RELEASE_ID)
    check("SYSTEM_READINESS unchanged",
          SYSTEM_READINESS == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET", SYSTEM_READINESS)
    check("REAL_WORLD_VALIDATION unchanged",
          REAL_WORLD_VALIDATION == "BLOCKED_PENDING_ELIGIBLE_DATASET",
          REAL_WORLD_VALIDATION)
    check("PROMOTION unchanged",
          PROMOTION_STATE == "PROMOTION_GATE_REQUIRED", PROMOTION_STATE)
    prod_after = tree_hash(Path(REPO_ROOT) / "models" / "production")
    art_after = tree_hash(Path(REPO_ROOT) / "models" / "artifacts")
    check("models/production tree untouched by this suite",
          prod_after == prod_before,
          f"{prod_before[:12]} -> {prod_after[:12]}")
    check("models/artifacts tree untouched by this suite",
          art_after == art_before,
          f"{art_before[:12]} -> {art_after[:12]}")
    con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    lc_final = con.execute(lc_sql).fetchone()[0]
    con.close()
    check(f"shared chain lifecycle count still {LIVE_LIFECYCLE_ROWS} at exit",
          lc_final == LIVE_LIFECYCLE_ROWS, str(lc_final))
    check("shared chain still trustworthy at exit",
          P109.evaluate_chain(live_rows())["ok"])

    print()
    total = _checks["n"]
    passed = total - len(failures)
    print(f"Total: {total}  |  PASS: {passed}  |  FAIL: {len(failures)}")
    if failures:
        for name in failures:
            print(f"  FAILED: {name}")
    else:
        print("ALL PHASE 109 CHECKS PASSED")
    print(f"manifest_sha256={disk_h}")
    print(f"elapsed_s={time.time() - t0:.1f}")
    return 1 if failures else 0


def rechain_keep_prev(rows: list[dict], from_index: int) -> list[dict]:
    """Recompute entry_hash for rows[from_index:] so each row stays
    self-consistent with its (possibly forged) prev_hash, leaving the
    LINK relationship broken exactly where it was forged."""
    for i in range(from_index, len(rows)):
        body = canonical(json.loads(rows[i]["payload_summary"]))
        rows[i]["entry_hash"] = hashlib.sha256(
            (rows[i]["prev_hash"] + body).encode("utf-8")
        ).hexdigest()
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
