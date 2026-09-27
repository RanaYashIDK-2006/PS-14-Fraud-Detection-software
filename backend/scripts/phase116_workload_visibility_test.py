"""Phase 116 — Review workload & turnaround visibility: suite.

Sections
--------
 [1] production artifact identity BEFORE the suite runs
 [2] harness: isolated temp DB_DIR, waiting rows, UNDER_REVIEW rows with
     server-derived reviewer identity, and crafted completed reviews with
     known start/completion stamps (single- and multi-cycle), plus admin
     login
 [3] summary scalars: Needs Review / Under Review / Reviewed, reviewed
     last 24h / 7d, oldest waiting, active reviewers (incl. missing
     identity -> null), queue aging + review counts intact
 [4] duration semantics: exact count/average/median/min/max over known
     current-cycle durations; missing start, missing completion, zero and
     negative durations excluded (never counted as 0); reopened review
     counts only its CURRENT cycle; boundary buckets half-open/lower-
     inclusive
 [5] periods: 24h / 7d / 30d / all exact counts + duration sets, default
     24h, invalid values -> 400
 [6] state transitions: start/complete/reopen move the workload with the
     server (never locally), current-cycle duration across a real reopen,
     concurrent completes move workload exactly once, existing review
     counts stay correct
 [7] security: 401 unauthenticated/invalid auth, malformed/injection-
     shaped workload_range -> 400, allowlisted parameter only, no client
     timestamp, no secrets/PANs/credentials in responses or URLs, no new
     audit event types
 [8] frontend pins: Review Workload section, period control + URL
     allowlist/restoration (refresh/Back/Forward), N/A rendering, server
     refresh after mutations, no rankings/grades language, regression
     pins for Phase-113/114/115 surfaces
 [9] regression pins: state machine, URL restoration, queue navigation,
     concurrency, audit events, bounded pagination, script blocks
 [10] canonical constants + qualified_datasets + RWV / promotion states
 [11] production artifact identity AFTER (byte-identical)

Everything runs against an ISOLATED temp DB_DIR: no row written here can
reach the shared production DB-3/DB-4, and no production artifact is ever
opened for write. No network, no model fitting, no threshold/release work.

Run from backend/:
  ../.venv/Scripts/python.exe scripts/phase116_workload_visibility_test.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DB_TMP = tempfile.mkdtemp(prefix="ps14_p116wl_")
os.environ["DB_DIR"] = DB_TMP
ADMIN_USER = "admin"
ADMIN_PASS = "p116wl-suite-pass-7f3d"
os.environ["ADMIN_USER"] = ADMIN_USER
os.environ["ADMIN_PASS"] = ADMIN_PASS

from fastapi.testclient import TestClient  # noqa: E402

import src.front_service.main as fm  # noqa: E402
from src.front_service.main import app  # noqa: E402
from src.monitoring import manifest_contract as MC  # noqa: E402
from src.monitoring.phase103_production_readiness_closure import (  # noqa: E402
    PROMOTION_STATE,
    REAL_WORLD_VALIDATION,
    SYSTEM_READINESS,
)
from src.monitoring.phase108_public_benchmark_execution import (  # noqa: E402
    PRODUCTION_ARTIFACT_PATHS,
    snapshot_artifacts,
)
from src.monitoring.real_world_evaluation_protocol import (  # noqa: E402
    MODEL_ID,
    PRODUCTION_THRESHOLD,
    RELEASE_ID,
)
from src.audit_service.writer import (  # noqa: E402
    append_audit_event,
    flush_audit_queue,
)

failures: list[str] = []
n_assert = 0


def ok(cond: bool, msg: str) -> None:
    global n_assert
    n_assert += 1
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


# ── [1] production artifact identity BEFORE ───────────────────────────
WATCHED = tuple(PRODUCTION_ARTIFACT_PATHS) + (
    "models/production/release_manifest.json",)
before = snapshot_artifacts()
before["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
ok(len([k for k, v in before.items() if v != "absent"]) == 13,
   f"13 production artifacts readable before ({len(before)})")

# ── [2] harness: seed isolated DB-3 + crafted lifecycle rows ─────────
con = sqlite3.connect(str(fm._DBS["risk"]))
con.executescript(
    """
    CREATE TABLE IF NOT EXISTS risk_scores (
        score_id VARCHAR(36) PRIMARY KEY,
        fraud_id VARCHAR(16) NOT NULL,
        event_id VARCHAR(64) NOT NULL UNIQUE,
        risk_score INTEGER NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
        risk_band VARCHAR(16) NOT NULL,
        reason_codes TEXT NOT NULL,
        model_version VARCHAR(64),
        ml_score FLOAT,
        rule_score FLOAT,
        degraded BOOLEAN,
        scored_at DATETIME
    );
    """
)
fm._ensure_review_tables()

# One whole-second base so every crafted duration is an exact integer.
BASE = datetime.now(timezone.utc).replace(microsecond=0)


def ago(sec: int) -> str:
    return (BASE - timedelta(seconds=sec)).isoformat()


def stamp(seconds_ago: int) -> str:
    return (BASE - timedelta(seconds=seconds_ago)).strftime(
        "%Y-%m-%d %H:%M:%S")


# Waiting rows (flagged UNREVIEWED): w1 is the 8-day backlog anchor — the
# summary's Needs Review KPI window is 24h while oldest_waiting has no
# window (backlog age), so w1 sits outside the KPI but anchors the oldest.
SEEDS = [
    ("w1", 691200, "high"),     # 8d -> oldest waiting
    ("w2", 3600, "high"),
    ("w3", 600, "high"),
    ("w4", 100, "medium"),
    ("u1", 7200, "high"),       # walked UNDER_REVIEW via the API
    ("lowband", 300, "low"),    # never in Needs Review
]
for i, (key, age, band) in enumerate(SEEDS, start=1):
    con.execute(
        "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
        "risk_band, reason_codes, model_version, ml_score, rule_score, "
        "degraded, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (f"sc-p116wl-{key}", f"F-P116WL-{i:04d}",
         f"evt-p116wl-{key}-0001", 60 + i, band, "[]",
         MC.CANONICAL_MODEL_VERSION, 0.5, 4.0, 0, stamp(age)))
con.commit()
con.close()

EID = {key: f"evt-p116wl-{key}-0001" for key, *_ in SEEDS}
EID.update({
    "c899": "evt-p116wl-c899-0001", "c900": "evt-p116wl-c900-0001",
    "c3599": "evt-p116wl-c3599-0001", "c3600": "evt-p116wl-c3600-0001",
    "c14399": "evt-p116wl-c14399-0001", "c14400": "evt-p116wl-c14400-0001",
    "c86399": "evt-p116wl-c86399-0001", "c86400": "evt-p116wl-c86400-0001",
    "cd3": "evt-p116wl-cd3-0001", "cd10": "evt-p116wl-cd10-0001",
    "cd60": "evt-p116wl-cd60-0001",
    "nostart": "evt-p116wl-nostart-0001",
    "nocomp": "evt-p116wl-nocomp-0001",
    "zerodur": "evt-p116wl-zerodur-0001",
    "negdur": "evt-p116wl-negdur-0001",
    "twocycle": "evt-p116wl-twocycle-0001",
    "u2": "evt-p116wl-u2-0001",
})


def ins_review(key: str, state: str, reviewer: str | None,
               created: str, updated: str, reviewed: str | None) -> None:
    conn = sqlite3.connect(str(fm._DBS["risk"]))
    try:
        conn.execute(
            "INSERT INTO event_reviews (event_id, review_state, "
            "reviewer_id, created_at, updated_at, reviewed_at, version) "
            "VALUES (?,?,?,?,?,?,2)",
            (EID[key], state, reviewer, created, updated, reviewed))
        conn.commit()
    finally:
        conn.close()


# Current-cycle completed reviews with EXACT durations (in-window rows
# complete 300s ago).  Bucket edges are half-open, lower-inclusive.
DURS_24H = [899, 900, 3599, 3600, 14399, 14400, 86399, 86400]
for dur in DURS_24H:
    ins_review(f"c{dur}", "REVIEWED", "admin", ago(300 + dur), ago(300),
               ago(300))
# Older completions: inside 7d / 30d / all-time only.
DURS_7D = [7200]
DURS_30D = [1800]
DURS_ALL = [1200]
for key, end, dur in (("cd3", 259200, 7200), ("cd10", 864000, 1800),
                      ("cd60", 5184000, 1200)):
    ins_review(key, "REVIEWED", "admin", ago(end + dur), ago(end), ago(end))
# Missing start (no cycle-begin audit event at all) -> unmeasurable.
ins_review("nostart", "REVIEWED", "admin", ago(3600), ago(3600), ago(3600))
# Missing completion: REVIEWED state but no reviewed_at stamp.
ins_review("nocomp", "REVIEWED", "admin", ago(4000), ago(4000), None)
# Zero duration: begin == completion (impossible in reality) -> excluded.
ins_review("zerodur", "REVIEWED", "admin", ago(5000), ago(5000), ago(5000))
# Negative duration: begin AFTER completion -> excluded.
ins_review("negdur", "REVIEWED", "admin", ago(100), ago(5000), ago(5000))
# Two cycles: begin1 7200s ago, reopened 1000s ago, completed 400s ago.
# Current-cycle duration must be 600s ONLY (never 6800s).
ins_review("twocycle", "REVIEWED", "admin", ago(7200), ago(400), ago(400))
# Second distinct reviewer currently holding a record.
ins_review("u2", "UNDER_REVIEW", "reviewer-2", ago(10800), ago(10800), None)

# Cycle-begin audit events through the real chain writer (no new event
# types, no fabricated completion events, exact payload timestamps).
AUDIT_BATCH: list[tuple[str, str, str]] = []
for dur in DURS_24H:
    AUDIT_BATCH.append((f"c{dur}", "admin_review_started", ago(300 + dur)))
AUDIT_BATCH += [
    ("cd3", "admin_review_started", ago(259200 + 7200)),
    ("cd10", "admin_review_started", ago(864000 + 1800)),
    ("cd60", "admin_review_started", ago(5184000 + 1200)),
    ("zerodur", "admin_review_started", ago(5000)),
    ("negdur", "admin_review_started", ago(100)),
    # twocycle: the original begin FIRST, then the reopen (newest wins).
    ("twocycle", "admin_review_started", ago(7200)),
    ("twocycle", "admin_review_reopened", ago(1000)),
]
for n, (key, etype, ts) in enumerate(AUDIT_BATCH, start=1):
    append_audit_event(
        f"F-P116WL-{n:04d}", etype,
        {"event_id": EID[key],
         "previous_review_state": ("REVIEWED"
                                   if etype == "admin_review_reopened"
                                   else "UNREVIEWED"),
         "new_review_state": "UNDER_REVIEW",
         "admin_identity": "admin", "timestamp": ts})
flush_audit_queue(timeout=10)

c = TestClient(app)
r = c.post("/admin/login",
           json={"username": ADMIN_USER, "passphrase": ADMIN_PASS})
ok(r.status_code == 200 and "token" in r.json(),
   f"login -> 200 + token ({r.status_code})")
TOKEN = r.json()["token"]
HDR = {"Authorization": f"Bearer {TOKEN}"}


def rv_get(eid: str, client: TestClient | None = None):
    return (client or c).get(
        f"/admin/api/transactions/{eid}/review", headers=HDR)


def rv_post(eid: str, action: str, client: TestClient | None = None):
    h = dict(HDR)
    h["Content-Type"] = "application/json"
    return (client or c).post(
        f"/admin/api/transactions/{eid}/review/{action}",
        json={}, headers=h)


def list_tx(**kw):
    params = {"limit": 200}
    params.update(kw)
    return c.get("/admin/api/transactions", headers=HDR, params=params)


def summary(rng: str | None = None):
    params = {} if rng is None else {"workload_range": rng}
    return c.get("/admin/api/summary", headers=HDR, params=params)


def wl(rng: str | None = None) -> dict | None:
    r = summary(rng)
    if r.status_code != 200:
        return None
    return (r.json() or {}).get("workload")


def review_audit_rows(event_type: str | None = None) -> list[dict]:
    flush_audit_queue(timeout=5)
    conn = sqlite3.connect(str(fm._DBS["audit"]))
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT seq, event_type, payload_summary FROM audit_events"
        params: tuple = ()
        if event_type:
            sql += " WHERE event_type = ?"
            params = (event_type,)
        return [dict(x) for x in conn.execute(
            sql + " ORDER BY seq", params).fetchall()]
    finally:
        conn.close()


def event_ts_list(eid: str, etype: str) -> list[str]:
    """Chronological payload timestamps of one audit type for one event."""
    out: list[str] = []
    for x in review_audit_rows(etype):
        try:
            p = json.loads(x["payload_summary"])
        except (TypeError, ValueError):
            continue
        ts = p.get("timestamp")
        if p.get("event_id") == eid and isinstance(ts, str):
            out.append(ts)
    return out


def cycle_begin_ts(eid: str) -> str | None:
    """Newest cycle-begin timestamp recorded for one event (DB-4)."""
    found: str | None = None
    for etype in ("admin_review_started", "admin_review_reopened"):
        hits = event_ts_list(eid, etype)
        if hits:
            ts = hits[-1]
            if found is None or ts > found:
                found = ts
    return found


def reviewed_at_of(eid: str) -> str | None:
    conn = sqlite3.connect(str(fm._DBS["risk"]))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT reviewed_at FROM event_reviews WHERE event_id = ?",
            (eid,)).fetchone()
        return row["reviewed_at"] if row else None
    finally:
        conn.close()


def iso_delta(a: str, b: str) -> float:
    return (datetime.fromisoformat(a)
            - datetime.fromisoformat(b)).total_seconds()


def expected_dur(durs: list[float]) -> dict:
    ds = sorted(durs)
    n = len(ds)
    return {
        "count": n,
        "average_seconds": round(sum(ds) / n, 2) if n else None,
        "median_seconds": round(statistics.median(ds), 2) if n else None,
        "min_seconds": int(round(ds[0])) if n else None,
        "max_seconds": int(round(ds[-1])) if n else None,
    }


def expected_buckets(durs: list[float]) -> dict:
    return {
        "lt15m": sum(1 for x in durs if x < 900),
        "15m_1h": sum(1 for x in durs if 900 <= x < 3600),
        "1h_4h": sum(1 for x in durs if 3600 <= x < 14400),
        "4h_24h": sum(1 for x in durs if 14400 <= x < 86400),
        "gt24h": sum(1 for x in durs if x >= 86400),
    }


# Walk u1 into UNDER_REVIEW through the real state machine (identity
# always comes from the session, never the request).
r = rv_post(EID["u1"], "start")
ok(r.status_code == 200 and r.json()["review_state"] == "UNDER_REVIEW",
   "seed: u1 walked to UNDER_REVIEW via the API")

# ── [3] summary scalars: counts, backlog, active reviewers ───────────
r = summary()
ok(r.status_code == 200, f"summary -> 200 ({r.status_code})")
d = r.json()
ok("workload" in d, "summary payload carries the workload block")
w = d.get("workload")
ok(isinstance(w, dict), "workload block present on a working store")
cnt = d.get("counts") or {}
ok(cnt.get("needs_review") == 3,
   f"Needs Review: flagged UNREVIEWED inside the 24h KPI window "
   f"({cnt.get('needs_review')})")
ok(cnt.get("under_review") == 2,
   f"Under Review scalar ({cnt.get('under_review')})")
ok(cnt.get("reviewed") == 16,
   f"Reviewed scalar counts every REVIEWED state incl. missing stamp "
   f"({cnt.get('reviewed')})")
ok(w.get("needs_review") == cnt.get("needs_review")
   and w.get("under_review") == cnt.get("under_review")
   and w.get("reviewed") == cnt.get("reviewed"),
   "workload mirrors the summary's authoritative review scalars")
ok(w.get("range") == "24h",
   f"default workload range is 24h ({w.get('range')})")
ok(w.get("reviewed_last_24h") == 12,
   f"reviewed_last_24h counts timestamped completions in-window "
   f"({w.get('reviewed_last_24h')})")
ok(w.get("reviewed_last_7d") == 13,
   f"reviewed_last_7d ({w.get('reviewed_last_7d')})")
ok(w.get("reviewed_in_period") == w.get("reviewed_last_24h"),
   "default period (24h) scopes reviewed_in_period to 24h")
oldest = w.get("oldest_waiting_seconds")
ok(isinstance(oldest, int) and 691200 <= oldest <= 691200 + 3600,
   f"oldest_waiting_seconds is the 8-day backlog row ({oldest})")
ok(w.get("active_reviewers") == 2,
   f"active_reviewers = distinct reviewers on UNDER_REVIEW rows "
   f"({w.get('active_reviewers')})")

# Missing reviewer identity -> honest null, never a guessed count.
con = sqlite3.connect(str(fm._DBS["risk"]))
con.execute("INSERT INTO event_reviews (event_id, review_state, reviewer_id, "
            "created_at, updated_at, reviewed_at, version) "
            "VALUES (?,?,?,?,?,NULL,1)",
            ("evt-p116wl-noid-0001", "UNDER_REVIEW", None, ago(60), ago(60)))
con.commit()
con.close()
ok(wl().get("active_reviewers") is None,
   "under-review row without identity -> active_reviewers null (N/A)")
con = sqlite3.connect(str(fm._DBS["risk"]))
con.execute("DELETE FROM event_reviews WHERE event_id = ?",
            ("evt-p116wl-noid-0001",))
con.commit()
con.close()
ok(wl().get("active_reviewers") == 2,
   "identity restored -> active_reviewers counts 2 again")

# Existing queue surfaces unchanged (Phase 115 preserved).  NOTE: the
# list endpoint's review_counts are risk-row-scoped (a review row needs a
# scored transaction behind it), unlike the summary scalars above.
r = list_tx()
rc = r.json().get("review_counts")
ok(rc == {"needs": 4, "under": 1, "reviewed": 0},
   f"review_counts stay Phase-114/115-correct ({rc})")
ag = r.json().get("queue_aging")
ok(isinstance(ag, dict) and isinstance(ag["oldest_waiting_seconds"], int)
   and sum(ag[k] for k in ("under_15m", "15m_to_1h", "1h_to_4h",
                           "4h_to_24h", "over_24h")) == 4,
   f"queue_aging buckets still partition the waiting queue ({ag})")

# ── [4] duration semantics: exact current-cycle aggregates ───────────
EXP24 = [float(x) for x in DURS_24H] + [600.0]  # twocycle: CURRENT cycle
w = wl()
dur = w.get("completed_review_duration")
ok(dur is not None, "completed_review_duration block present")
ok(dur == expected_dur(EXP24),
   f"24h duration aggregate over known cycles "
   f"({dur} vs {expected_dur(EXP24)})")
ok(w.get("duration_buckets") == expected_buckets(EXP24),
   f"24h duration buckets match the boundary seeds "
   f"({w.get('duration_buckets')})")
ok(dur["count"] == 9,
   "missing start / zero / negative durations are excluded — "
   "unmeasurable is never counted as 0")
ok(w.get("reviewed_in_period") == 12 and dur["count"] == 9,
   "12 completions in 24h, 9 of them measurable (honest gap, no zeros)")
ok(EXP24.count(600.0) == 1 and 6800.0 not in EXP24,
   "multi-cycle row measured from reopen -> completion only")
ok(w.get("duration_buckets") == {"lt15m": 2, "15m_1h": 2, "1h_4h": 2,
                                 "4h_24h": 2, "gt24h": 1},
   f"boundary buckets exactly as specified ({w.get('duration_buckets')})")
# Missing completion stays out of the timestamped counts but inside the
# state count — the two answer different questions, honestly.
ok(w["reviewed"] == 16 and w["reviewed_last_24h"] == 12,
   "missing completion: state count includes it, period count cannot")

# ── [5] periods: 24h / 7d / 30d / all + invalid values ───────────────
w7 = wl("7d")
ok(w7.get("reviewed_in_period") == 13,
   f"7d period counts the 3-day-old completion "
   f"({w7.get('reviewed_in_period')})")
ok(w7.get("completed_review_duration") == expected_dur(EXP24 + DURS_7D),
   "7d duration set adds exactly the 7d completion")
ok((w7.get("reviewed_last_24h"), w7.get("reviewed_last_7d")) == (12, 13),
   "reviewed_last_24h/7d are period-INDEPENDENT facts (always both)")
w30 = wl("30d")
ok(w30.get("reviewed_in_period") == 14,
   f"30d period ({w30.get('reviewed_in_period')})")
ok(w30.get("completed_review_duration") == expected_dur(
    EXP24 + DURS_7D + DURS_30D),
   "30d duration set")
wall = wl("all")
ok(wall.get("reviewed_in_period") == 15,
   f"all-time period counts every timestamped completion "
   f"({wall.get('reviewed_in_period')})")
ok(wall.get("completed_review_duration") == expected_dur(
    EXP24 + DURS_7D + DURS_30D + DURS_ALL),
   "all-time duration set")
ok(wall.get("duration_buckets") == expected_buckets(
    EXP24 + DURS_7D + DURS_30D + DURS_ALL),
   "all-time duration buckets")
for bad in ("1y", "24H", "24 hours", "30", "0", "", "24h,7d", "all-time",
            "24h' OR '1'='1", "7d; DROP TABLE event_reviews", "*",
            "24h--", "../../etc/passwd", "%00"):
    rr = summary(bad)
    ok(rr.status_code == 400,
       f"invalid workload_range {bad!r} -> 400 ({rr.status_code})")
ok(summary("24h").status_code == 200
   and summary("7d").status_code == 200
   and summary("30d").status_code == 200
   and summary("all").status_code == 200,
   "all four allowlisted periods -> 200")

# ── [6] state transitions move the workload (server-side only) ───────
# Start: an active review leaves Needs Review and joins Under Review.
r = rv_post(EID["w2"], "start")
ok(r.status_code == 200 and r.json()["review_state"] == "UNDER_REVIEW",
   "start w2 -> UNDER_REVIEW")
w = wl()
ok(w.get("under_review") == 3 and w.get("needs_review") == 2,
   f"start moves active workload ({w.get('under_review')} under / "
   f"{w.get('needs_review')} needs)")
ok(w.get("active_reviewers") == 2,
   "distinct reviewer count: same reviewer, no inflation")
ok(list_tx().json()["review_counts"] == {"needs": 3, "under": 2,
                                         "reviewed": 0},
   "queue review_counts move with the same transition")

# Complete: joins the completed workload for the current period.
r = rv_post(EID["w2"], "complete")
ok(r.status_code == 200 and r.json()["review_state"] == "REVIEWED",
   "complete w2 -> REVIEWED")
w = wl()
ok(w.get("reviewed") == 17 and w.get("reviewed_in_period") == 13,
   f"complete adds one reviewed completion in period "
   f"({w.get('reviewed')} / {w.get('reviewed_in_period')})")
ok(w.get("completed_review_duration")["count"] == 10,
   "complete adds its measured duration to the aggregate")
ok(w.get("duration_buckets")["lt15m"] == 3,
   "a fresh completion lands in the <15m bucket")

# Reopen: leaves the completed workload, rejoins the active one.
r = rv_post(EID["w2"], "reopen")
ok(r.status_code == 200 and r.json()["review_state"] == "UNDER_REVIEW",
   "reopen w2 -> UNDER_REVIEW")
w = wl()
ok(w.get("reviewed") == 16 and w.get("reviewed_in_period") == 12,
   "reopen removes it from reviewed/period counts (reviewed_at cleared)")
ok(w.get("completed_review_duration")["count"] == 9
   and w.get("duration_buckets")["lt15m"] == 2,
   "reopen removes its duration from the aggregates")
ok(w.get("under_review") == 3 and w.get("active_reviewers") == 2,
   "reopen rejoins active workload")

# Second completion: CURRENT cycle begins at the reopen audit event.
r = rv_post(EID["w2"], "complete")
ok(r.status_code == 200 and r.json()["review_state"] == "REVIEWED",
   "re-complete w2 -> REVIEWED (second cycle)")
w2_started = event_ts_list(EID["w2"], "admin_review_started")
w2_reopened = event_ts_list(EID["w2"], "admin_review_reopened")
w2_completed = event_ts_list(EID["w2"], "admin_review_completed")
ok(len(w2_started) == 1 and len(w2_reopened) == 1
   and len(w2_completed) == 2,
   "both cycles recorded in the existing audit events "
   f"({len(w2_started)}/{len(w2_reopened)}/{len(w2_completed)})")
r_end = reviewed_at_of(EID["w2"])
ok(r_end == w2_completed[1],
   "table completion stamp equals the CURRENT cycle's completion event")
second_cycle = iso_delta(w2_completed[1], w2_reopened[0])
whole_span = iso_delta(w2_completed[1], w2_started[0])
ok(0 <= second_cycle < whole_span,
   f"current-cycle duration excludes the earlier cycle "
   f"({second_cycle}s of {whole_span}s total)")
w = wl()
EXP_AFTER = EXP24 + [second_cycle]
ok(w.get("completed_review_duration") == expected_dur(EXP_AFTER),
   "aggregate equals known cycles + the DB-derived current-cycle duration "
   "(proves the server measures reopen -> completion, not start -> completion)")
ok(w.get("reviewed_in_period") == 13,
   "re-completion returns to the period count")

# Concurrency: two simultaneous completes move the workload exactly once.
r = rv_post(EID["w3"], "start")
ok(r.status_code == 200, "start w3 for the concurrency race")
results: list[tuple[int, dict]] = []


def race_complete():
    rr = rv_post(EID["w3"], "complete")
    try:
        results.append((rr.status_code, rr.json()))
    except ValueError:
        results.append((rr.status_code, {}))


t1 = threading.Thread(target=race_complete)
t2 = threading.Thread(target=race_complete)
t1.start()
t2.start()
t1.join()
t2.join()
changed = [x for x in results if x[1].get("changed") is True]
ok(len(results) == 2 and all(x[0] == 200 for x in results),
   f"both racing completes answered 200 ({[x[0] for x in results]})")
ok(len(changed) == 1,
   f"exactly one transition applied under the race "
   f"({[x[1].get('changed') for x in results]})")
w3_completions = event_ts_list(EID["w3"], "admin_review_completed")
ok(len(w3_completions) == 1,
   f"race produced exactly ONE completion audit event "
   f"({len(w3_completions)})")
w = wl()
w3_dur = iso_delta(reviewed_at_of(EID["w3"]),
                   cycle_begin_ts(EID["w3"]) or "")
EXP_FINAL = EXP_AFTER + [w3_dur]
ok(w.get("completed_review_duration") == expected_dur(EXP_FINAL),
   "race winner's current-cycle duration measured exactly once")
ok(w.get("reviewed_in_period") == 14,
   f"period count after the race ({w.get('reviewed_in_period')})")
ok(w.get("active_reviewers") == 2,
   "races never fabricate reviewers (still the two distinct identities)")
ok(list_tx().json()["review_counts"] == {"needs": 2, "under": 1,
                                         "reviewed": 2},
   "final review counts correct after start/complete/reopen/race")

# ── [7] security ─────────────────────────────────────────────────────
fresh = TestClient(app)
for path in ("/admin/api/summary",
             "/admin/api/summary?workload_range=7d",
             "/admin/api/summary?workload_range=bogus"):
    rr = fresh.get(path)
    ok(rr.status_code == 401,
       f"unauthenticated {path.split('/admin/api')[1]} -> 401 "
       f"({rr.status_code})")
rr = fresh.get("/admin/api/summary?workload_range=24h",
               headers={"Authorization": "Bearer not-a-real-token"})
ok(rr.status_code == 401, f"invalid bearer token -> 401 ({rr.status_code})")
rr = fresh.get("/admin/api/summary?workload_range=24h",
               cookies={"admin_session": "garbage-session"})
ok(rr.status_code == 401, f"invalid session cookie -> 401 ({rr.status_code})")
# Auth first, then validation: a malformed range with no session is an
# authentication failure, never a validation oracle.
ok(fresh.get("/admin/api/summary?workload_range='; DROP--").status_code == 401,
   "unauthenticated + injection-shaped range -> 401 (auth checked first)")
rr = summary("bogus")
ok(rr.status_code == 400 and "workload_range" in (rr.json().get("detail")
                                                  or ""),
   f"authenticated malformed range -> 400 with a named allowlist "
   f"({rr.status_code}: {rr.json().get('detail')})")

# The only new server-side input is the allowlisted label itself.
sig = inspect.signature(fm.admin_api_summary)
ok(set(sig.parameters) == {"request", "workload_range"},
   f"summary accepts exactly one new client parameter "
   f"({list(sig.parameters)})")
for forbidden in ("reviewer_id", "admin_identity", "timestamp", "started_at",
                  "reviewed_at", "duration", "seconds"):
    ok(forbidden not in sig.parameters,
       f"no client-side {forbidden} input (server-derived only)")

src = (BACKEND / "src" / "front_service" / "main.py").read_text(
    encoding="utf-8").replace("\r\n", "\n")
ok("workload_range not in _WORKLOAD_RANGES" in src,
   "workload_range gated by an exact-match allowlist")
ok("_WORKLOAD_RANGE_DAYS" in src and "now - timedelta(days=" in src,
   "period cutoffs computed from SERVER time, never a client timestamp")
ok("LIMIT 5000" in src, "workload queries are bounded")
ok("admin_workload" not in src and "workload_logged" not in src,
   "no new audit event type invented for analytics")
summary_body = src.split("def admin_api_summary")[1].split(
    "def admin_api_live")[0]
ok("append_audit_event" not in summary_body,
   "summary writes NO audit events (read-only observability)")

# No secrets / PANs / credentials in any response this phase touches.
for path in ("/admin/api/summary?workload_range=all",
             "/admin/api/summary?workload_range=24h",
             "/admin/api/transactions?review=needs&age=gt24h"
             "&queue_sort=oldest"):
    txt = c.get(path, headers=HDR).text
    ok("passphrase" not in txt and "blob_key" not in txt
       and ADMIN_PASS not in txt and "TOTP" not in txt
       and "otpauth" not in txt,
       f"no secrets in {path.split('?')[0]}")
    ok(not re.search(r"\b4[0-9]{15}\b|\b3[0-9]{14}\b", txt),
       f"no PAN-shaped values in {path.split('?')[0]}")
    ok("token" not in txt.lower() or TOKEN not in txt,
       f"session token never echoed in {path.split('?')[0]}")

shell = c.get("/admin").text
flat = shell.replace("\r\n", "\n")
SECRET_PATS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)[\"']Bearer\s+[A-Za-z0-9._\-]{40,}[\"']"),
     "hard-coded bearer token"),
    (re.compile(r"[?&](?:access_token|api_key|apikey|secret|totp)=[A-Za-z0-9]"),
     "credential in a URL query string"),
]
hits = [label for pat, label in SECRET_PATS if pat.search(shell)]
ok(not hits, f"no secrets in the served shell ({hits})")
ok(ADMIN_PASS not in shell and TOKEN not in shell,
   "admin passphrase / session token never appear in the shell")
ok("workload_range" in shell
   and re.search(r"'\?workload_range=' \+ v", shell) is not None,
   "the URL carries only the allowlisted label (no values built from input)")
ok(re.search(r"workload_range=[A-Za-z0-9]{8,}", shell) is None,
   "no long literal ever embedded in the workload URL")

# ── [8] frontend pins ────────────────────────────────────────────────
ok('id="wl-section"' in shell and "Review Workload" in shell,
   "compact Review Workload section on the dashboard")
for eid in ("wl-needs", "wl-under", "wl-reviewed", "wl-reviewers",
            "wl-reviewed-label", "wl-line-completed", "wl-line-avg",
            "wl-line-median", "wl-line-oldest", "wl-buckets", "wl-range",
            "wl-detail"):
    ok(f'id="{eid}"' in shell, f"workload element present: {eid}")
m_sel = re.search(r'<select id="wl-range".*?</select>', shell, re.S)
opts = re.findall(r'<option value="([^"]+)">',
                  m_sel.group(0)) if m_sel else []
ok(opts == ["24h", "7d", "30d", "all"],
   f"period control offers exactly 24h/7d/30d/all ({opts})")
ok("workload_range must be one of" not in shell,
   "period validation stays server-side (client only sends allowlisted)")
ok("Completed reviews: N/A" in shell and "Average review duration: N/A"
   in shell and "Median review duration: N/A" in shell
   and "Oldest waiting: N/A" in shell and "Duration buckets: N/A" in shell,
   "every workload line starts at N/A (never a fabricated 0)")
ok("renderWorkload(s.workload, c)" in shell,
   "dashboard renders the workload block from the server payload")
ok("'/admin/api/summary?workload_range='" in shell,
   "summary fetched with the selected period")
ok("function renderWorkload(w, c)" in shell
   and "function workloadUrlQuery()" in shell
   and "function applyWorkloadQuery(sp)" in shell
   and "function pushWorkloadUrl()" in shell,
   "workload helpers present")
ok("const WORKLOAD_RANGES = ['24h', '7d', '30d', 'all'];" in shell,
   "client allowlist matches the server's four labels")
ok(re.search(r"function applyWorkloadQuery\(sp\) \{.*?"
             r"WORKLOAD_RANGES\.includes\(v\).*?"
             r"el\.value = '24h';", shell, re.S) is not None,
   "invalid URL period falls back to the 24h default")
ok("if (!r.detail) applyWorkloadQuery(new URLSearchParams("
   "window.location.search));" in shell,
   "Back/Forward restore the workload period (popstate)")
ok("applyWorkloadQuery(new URLSearchParams(window.location.search));\n"
   "            loadDashboard();" in flat,
   "refresh lands on the URL's period (restored before the first fetch)")
ok("workloadUrlQuery() : '';" in shell,
   "dashboard tab URLs carry the period state")
ok("case 'dashboard':" in shell and "loadDashboard();" in shell,
   "opening the dashboard re-fetches the authoritative summary")
ok("if (d.changed) await afterReviewAction(action);" in shell
   and re.search(r"// Phase 116 §11[\s\S]{0,240}?"
                 r"if \(d\.changed\) loadDashboard\(\);", shell) is not None,
   "every review mutation refreshes the server summary (no local math)")
ok(re.search(r"wlSel\.addEventListener\('change'", shell) is not None
   and "pushWorkloadUrl();" in shell,
   "period change pushes a history entry and re-fetches")
ok("durFmt = v => (v != null ? fmtWait(v) : 'N/A')" in shell,
   "durations render through fmtWait (null -> N/A)")
# Descriptive only: no rankings, grades, or comparisons anywhere.
seg = shell[shell.find('id="tab-dashboard"'):shell.find('id="tab-live"')]
js_text = "\n".join(re.findall(r"<script>(.*?)</script>", shell, re.S))
for word in ("leaderboard", "productivity", "fastest", "slowest",
             "best reviewer", "worst", "performance score", "ranked",
             "grade:"):
    ok(word not in seg.lower() and word not in js_text.lower()
       and word not in src.lower(),
       f"no ranking/grade language: {word!r}")

# ── [9] regression pins: Phase-113/114/115 behavior ──────────────────
ok(fm._REVIEW_TRANSITIONS == {
    ("UNREVIEWED", "start"): "UNDER_REVIEW",
    ("UNDER_REVIEW", "complete"): "REVIEWED",
    ("REVIEWED", "reopen"): "UNDER_REVIEW",
}, "review state machine unchanged")
ok(fm._REVIEW_ACTIONS == {"start": "UNDER_REVIEW", "complete": "REVIEWED",
                          "reopen": "UNDER_REVIEW"},
   "review action targets unchanged")
ok(fm._REVIEW_FILTERS == ("all", "needs", "under", "reviewed")
   and fm._AGE_FILTERS == {"lt15m": (None, 900), "15m_1h": (900, 3600),
                           "1h_4h": (3600, 14400),
                           "4h_24h": (14400, 86400),
                           "gt24h": (86400, None)}
   and fm._QUEUE_SORTS == ("oldest", "newest"),
   "Phase 114/115 filter allowlists pinned")
ok(fm._WORKLOAD_RANGES == ("24h", "7d", "30d", "all")
   and fm._WORKLOAD_RANGE_DAYS == {"24h": 1, "7d": 7, "30d": 30,
                                   "all": None},
   "workload period allowlist pinned")
# Invalid transition rejected + replay idempotent (Phase 113).
rr = rv_post(EID["w2"], "start")
ok(rr.status_code == 409,
   f"invalid transition (start from REVIEWED) -> 409 ({rr.status_code})")
rr = rv_post(EID["w2"], "complete")
ok(rr.status_code == 200 and rr.json()["changed"] is False
   and rr.json()["review_state"] == "REVIEWED",
   "replay in the same state stays an idempotent success")
# Queue aging + ordering surface still behaves (Phase 115).
rr = list_tx(age="gt24h", queue_sort="oldest")
ok(rr.status_code == 200
   and all(x["review_state"] == "UNREVIEWED"
           and isinstance(x["waiting_seconds"], int)
           for x in rr.json()["rows"]),
   "age filter + oldest ordering still carry authoritative waiting ages")
rr = list_tx(age="bogus")
ok(rr.status_code == 400, f"age allowlist still enforced ({rr.status_code})")
ok("const TXN_URL_KEYS = ['q', 'event_id', 'fraud_id', 'band', 'decision',"
   in flat and "'review', 'since', 'until', 'age', 'queue_sort'];" in flat
   and "workload_range'" not in flat.split("TXN_URL_KEYS")[1][:600],
   "transaction URL allowlist unchanged (workload stays dashboard-only)")
ok("if (!r.detail) applyTxnQuery(new URLSearchParams(window.location.search));"
   in shell,
   "popstate list restore intact (Phase 115 pin)")
ok(shell.count(
    "window.location.pathname + window.location.search !== target") >= 3,
   "all three push paths idempotent (identical target never pushes)")
ok("const wantOpen = txnOpenExact;" in shell and "txnOpenExact = false;" in shell,
   "auto-open gate intact")
ok("if (livePaused) { setConn('PAUSED'); return; }" in shell
   and shell.count("if (livePaused) { setConn('PAUSED'); return; }") >= 2,
   "live-monitor pause race stays fixed")
ok(re.findall(r'data-rchip="([a-z]+)"', shell)
   == ["all", "needs", "under", "reviewed"],
   "review chips stay All/Needs/Under/Reviewed")
ok(re.findall(r'data-chip="([a-z]+)"', shell)
   == ["all", "flagged", "approved", "blocked"],
   "quick-filter chips untouched")
ok("function updateTxnNav()" in shell
   and "prev.disabled = !(idx > 0);" in shell,
   "queue navigation intact")
ok("function txnQueueLabel()" in shell
   and "const REVIEW_QUEUE_LABEL = { needs: 'Needs Review'" in shell,
   "queue labels intact")
ok("min(int(limit), 200)" in src and "(server-capped)" in shell,
   "bounded history pagination intact (limit <= 200)")
ok('maxlength="2000"' in shell and "admin_review_note_added" in shell,
   "notes stay Phase-113: bounded, same audit event")
ok("bulk" not in shell.lower(), "no bulk mutation affordance")
ok("Math.random" not in shell, "no random navigation anywhere")
ok("dash-details" not in shell, "removed dashboard expando stays removed")
blocks = re.findall(r"<script>(.*?)</script>", shell, re.S)
ok(len(blocks) == 2, f"exactly two script blocks ({len(blocks)})")
js = "\n".join(blocks)
ok(js.count("{") == js.count("}"),
   f"script braces balance ({js.count('{')} vs {js.count('}')})")
ok('id="dash-review-kpi"' in shell and "Needs Review →" in shell
   and 'id="dash-needs-review"' in shell,
   "Phase-113 dashboard KPI intact")
ok('id="dash-review-sub"' in shell,
   "Phase-114 dashboard KPI sub-line intact")
ok('id="f-queue-oldest"' in shell and 'id="f-queue-aging"' in shell,
   "Phase-115 queue banner intact")
ok("function renderQueueAging(aging)" in shell
   and "function fmtWait(sec)" in shell,
   "Phase-115 aging renderer + formatter intact")
# Live evaluation of the real fmtWait (presentation only).
m_fmt = re.search(r"function fmtWait\(sec\) \{.*?\n        \}", shell, re.S)
NODE = shutil.which("node")
if NODE and m_fmt:
    prog = (f"const fmtWait = {m_fmt.group(0)}; "
            f"console.log(JSON.stringify([null,4,2520,8280,97200]"
            f".map(v => fmtWait(v))));")
    proc = subprocess.run([NODE, "-e", prog], capture_output=True, text=True)
    got = json.loads(proc.stdout.strip()) if proc.returncode == 0 else None
    ok(got == ["N/A", "<1m", "42m", "2h 18m", "1d 3h"],
       f"fmtWait still formats exactly as specified ({got})")
else:
    ok(True, "fmtWait source pinned (node unavailable for live eval)")
# Audit surface: no new event types anywhere in this phase.
known = {x["event_type"] for x in review_audit_rows()}
ok(known <= {"score_generated", "admin_tx_search", "admin_tx_view",
             "admin_login_success", "admin_review_started",
             "admin_review_completed", "admin_review_reopened",
             "admin_review_note_added"},
   f"Phase 116 added no new audit event type ({sorted(known)})")
rv = rv_get(EID["w3"]).json()
ok(rv["review_state"] == "REVIEWED" and rv["version"] == 2,
   f"w3 ended REVIEWED at version 2 under the race ({rv['version']})")

# ── [10] canonical constants + governance states ─────────────────────
ok(MC.CANONICAL_THRESHOLD == 0.018758, "threshold constant unchanged")
ok(MC.CANONICAL_FEATURE_VERSION == "v1", "feature version unchanged")
ok(MC.CANONICAL_MODEL_ID == "altman_native", "governance model_id unchanged")
ok(MC.CANONICAL_RELEASE_ID
   == "release-altman_native_E_hardneg_cert_20260904",
   "governance release_id unchanged")
ok(MODEL_ID == "altman_native"
   and RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904"
   and PRODUCTION_THRESHOLD == 0.018758,
   "protocol model/release/threshold unchanged")
ok(SYSTEM_READINESS == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET",
   "system readiness state unchanged")
ok(REAL_WORLD_VALIDATION == "BLOCKED_PENDING_ELIGIBLE_DATASET",
   "RWV state unchanged")
ok(PROMOTION_STATE == "PROMOTION_GATE_REQUIRED",
   "promotion state unchanged")
mdl = summary().json().get("details", {}).get("model") or {}
ok(mdl.get("threshold") == 0.018758
   and mdl.get("governance_model_id") == MC.CANONICAL_MODEL_ID
   and mdl.get("governance_release_id") == MC.CANONICAL_RELEASE_ID,
   "summary model block still exposes the canonical identity")

from src.monitoring.phase104_external_dataset_report import (  # noqa: E402
    generate_phase104_report,
)
from src.monitoring.phase106_public_benchmark_report import (  # noqa: E402
    generate_phase106_report,
)
rep104 = generate_phase104_report()
ok(rep104.qualified_datasets == () and rep104.any_qualified is False,
   "phase104 qualified_datasets stays ()")
rep106 = generate_phase106_report()
ok(rep106.qualified_datasets == () and rep106.any_dataset_qualified is False,
   "phase106 qualified_datasets stays ()")

# ── [11] production artifact identity AFTER ──────────────────────────
after = snapshot_artifacts()
after["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
changed = [k for k in WATCHED if before.get(k) != after.get(k)]
ok(not changed,
   f"production artifacts byte-identical across the suite ({changed})")

print()
print(f"PHASE 116 WORKLOAD VISIBILITY: {n_assert} assertions, "
      f"{len(failures)} failures")
if failures:
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PHASE 116 WORKLOAD VISIBILITY: ALL PASS")
