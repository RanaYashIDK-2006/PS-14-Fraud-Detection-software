#!/usr/bin/env python3
"""Tests for the versioned feedback pool (section 7).

Covers the pool mechanics without touching the live DBs:
  1. every export writes a dated snapshot whose content equals the pool,
  2. snapshots accumulate (latest is the newest by timestamp),
  3. resolve_feedback: explicit file, "latest", at-or-before date
     resolution, no-snapshot errors, and mutual exclusion, and
  4. the trainer's resolved snapshot content matches the pool file that
     export wrote.

Run from the project root:
  python scripts/feedback_test.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Isolated DBs (the trigger appends to a throwaway DB-4, never the dev store).
TMP_DB = tempfile.mkdtemp(prefix="ps14-fbdb-")
os.environ["DB_DIR"] = TMP_DB
os.environ["JWT_SECRET"] = "feedback-test-secret-0123456789abcdef"
os.environ["INTERNAL_TOKEN"] = "feedback-test-internal"

import pandas as pd  # noqa: E402

import export_feedback as exp  # noqa: E402
from src.audit_service.writer import GENESIS_HASH, flush_audit_queue, verify_chain  # noqa: E402
from src.settings import get_settings  # noqa: E402
from src.train_compare import resolve_feedback  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def make_pool(rows: list[tuple[str, int]]) -> pd.DataFrame:
    """A minimal pool in the training schema (only the columns that matter)."""
    return pd.DataFrame(
        [{"event_id": eid, "label": lab, "ts": "2026-08-16 10:00:00"} for eid, lab in rows],
        columns=exp.COLUMNS,
    )


def main() -> int:
    print("== versioned feedback pool tests ==")
    tmp = Path(tempfile.mkdtemp(prefix="ps14-fb-"))
    snap_dir = tmp / "snapshots"

    # ---- 1. snapshot write: dated name, content == pool -------------------
    pool = make_pool([("ev-1", 0), ("ev-2", 1)])
    p1 = exp.write_snapshot(pool, snap_dir, stamp="20260816_120000")
    check("snapshot written with dated name",
          p1.name == "feedback_20260816_120000.csv", p1.name)
    check("snapshot content equals the pool",
          p1.read_bytes().decode("utf-8") == pool.to_csv(index=False))
    check("snapshot columns match the training schema",
          pd.read_csv(p1).columns.tolist() == exp.COLUMNS)

    # ---- 2. accumulation: newer snapshot is the latest ---------------------
    p2 = exp.write_snapshot(make_pool([("ev-1", 0), ("ev-2", 1), ("ev-3", 1)]),
                            snap_dir, stamp="20260816_133000")
    check("accumulates a second snapshot", p2.name == "feedback_20260816_133000.csv")
    check("latest == newest by timestamp",
          sorted(snap_dir.glob("feedback_*.csv"))[-1] == p2)

    # ---- 3. resolve_feedback ----------------------------------------------
    path, tag = resolve_feedback(str(p1), None, snap_dir)
    check("explicit file resolves as-is", path == p1.resolve() and tag is None)

    path, tag = resolve_feedback(None, "latest", snap_dir)
    check("'latest' picks the newest snapshot", path == p2.resolve() and tag == "20260816_133000",
          str(path.name))

    path, tag = resolve_feedback(None, "20260816", snap_dir)
    check("date resolves the last snapshot on/before that day",
          path == p2.resolve() and tag == "20260816_133000", str(path.name))

    path, tag = resolve_feedback(None, "2026-08-16", snap_dir)  # normalized form
    check("date accepts YYYY-MM-DD form", path == p2.resolve(), str(path.name))

    path, tag = resolve_feedback(None, "20260816_120000", snap_dir)
    check("timestamp resolution is inclusive of that moment",
          path == p1.resolve() and tag == "20260816_120000", str(path.name))

    path, tag = resolve_feedback(None, "20260817", snap_dir)
    check("a later date still resolves the newest snapshot", path == p2.resolve())

    try:
        resolve_feedback(None, "20260801", snap_dir)
        check("date before the earliest snapshot raises", False)
    except SystemExit as e:
        check("date before the earliest snapshot raises", "no feedback snapshot" in str(e), str(e))

    try:
        resolve_feedback(str(p1), "latest", snap_dir)
        check("--feedback + --feedback-date are mutually exclusive", False)
    except SystemExit as e:
        check("--feedback + --feedback-date are mutually exclusive",
              "mutually exclusive" in str(e), str(e))

    empty = tmp / "empty"
    try:
        resolve_feedback(None, "latest", empty)
        check("missing snapshot dir raises", False)
    except SystemExit as e:
        check("missing snapshot dir raises", "run scripts/export_feedback.py" in str(e), str(e))

    path, tag = resolve_feedback(None, None, snap_dir)
    check("no pool requested -> None", path is None and tag is None)

    # ---- 4. trainer consumes the snapshot the export wrote ----------------
    pool_file = tmp / "feedback_labeled.csv"
    pool.to_csv(pool_file, index=False)
    snap3 = exp.write_snapshot(pool, snap_dir, stamp="20260817_090000")
    path, tag = resolve_feedback(None, "latest", snap_dir)
    check("resolved snapshot is the export's own snapshot", path == snap3.resolve())
    check("resolved content matches the export pool",
          pd.read_csv(path).equals(pd.read_csv(pool_file)))

    # ---- 5. section-6 gate + retrain-queue trigger -------------------------
    def audit_rows() -> list[tuple]:
        con = sqlite3.connect(get_settings().audit_db_path)
        rows = con.execute(
            "SELECT seq, event_type, prev_hash, entry_hash FROM audit_events ORDER BY seq"
        ).fetchall()
        con.close()
        return rows

    def audit_as_objects(rows: list[tuple]):
        from types import SimpleNamespace
        con = sqlite3.connect(get_settings().audit_db_path)
        payloads = {r[0]: r[1] for r in con.execute(
            "SELECT seq, payload_summary FROM audit_events ORDER BY seq")}
        con.close()
        return [SimpleNamespace(seq=r[0], event_type=r[1], prev_hash=r[2],
                                entry_hash=r[3], payload_summary=payloads[r[0]])
                for r in rows]

    check("latest_snapshot returns the newest", exp.latest_snapshot(snap_dir) == snap3, snap3.name)
    check("latest_snapshot None on missing dir", exp.latest_snapshot(tmp / "nodir") is None)

    same = make_pool([("ev-1", 0), ("ev-2", 1)])
    check("pool_unchanged flags an identical pool", exp.pool_unchanged(same, snap_dir) == snap3, snap3.name)
    diff = make_pool([("ev-1", 0), ("ev-2", 1), ("ev-9", 1)])
    check("pool_unchanged None on a changed pool", exp.pool_unchanged(diff, snap_dir) is None)
    check("pool_unchanged None with no snapshots", exp.pool_unchanged(diff, tmp / "nodir") is None)

    before = len(audit_rows())
    real_small = make_pool([("ev-1", 0), ("ev-2", 0), ("ev-3", 0)])  # 0 disputed
    check("below the disputed threshold triggers nothing",
          exp.emit_retrain_trigger(real_small, "20260817_090000", 5) is None
          and len(audit_rows()) == before)

    real_trigger = make_pool([("ev-1", 0), ("ev-2", 1), ("ev-3", 1),
                              ("ev-4", 1), ("ev-5", 1), ("ev-6", 1)])  # 5 disputed
    payload = exp.emit_retrain_trigger(real_trigger, "20260817_090000", 5)
    check("threshold breach returns the payload",
          payload is not None and payload["disputed"] == 5, str(payload))
    flush_audit_queue()
    rows = audit_rows()
    check("retrain_trigger appended to DB-4",
          len(rows) == before + 1 and rows[-1][1] == "retrain_trigger", str(rows[-1][:2]))
    linked = rows[-1][2] == (GENESIS_HASH if len(rows) == 1 else rows[-2][3])
    check("retrain_trigger hash-chained", linked,
          f"prev={rows[-1][2][:12]}… genesis={GENESIS_HASH[:12]}…" if len(rows) == 1
          else f"prev={rows[-1][2][:12]}… <- {rows[-2][3][:12]}…")
    con = sqlite3.connect(get_settings().audit_db_path)
    pl = json.loads(con.execute(
        "SELECT payload_summary FROM audit_events WHERE seq = ?", (rows[-1][0],)
    ).fetchone()[0])
    con.close()
    check("trigger payload pseudonymous + actionable",
          pl["monitor"] == "feedback_queue"
          and pl["action"] == "retrain_queue_per_section_6"
          and pl["snapshot"] == "20260817_090000"
          and pl["confirmed"] == 1 and pl["disputed"] == 5
          and "fraud_id" not in pl and "event_id" not in pl, str(pl))
    check("chain integrity holds after trigger",
          verify_chain(audit_as_objects(audit_rows()))["ok"] is True)

    before2 = len(audit_rows())
    payload_off = exp.emit_retrain_trigger(real_trigger, "20260817_090000", 5, emit=False)
    check("--no-alert suppresses the append but still reports",
          payload_off is not None and len(audit_rows()) == before2)

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
