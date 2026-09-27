"""Phase 115 — Review queue aging & operator workload visibility: suite.

Sections
--------
 [1] production artifact identity BEFORE the suite runs
 [2] harness: isolated temp DB_DIR, seeded rows across every aging bucket
     (with exact 15m/1h/4h/24h boundary rows), three review states, a
     missing-timestamp row, and admin login
 [3] aging calculations: per-row waiting_seconds/waiting_since from the
     row's OWN scored_at against server time (UTC), missing stamp -> null
     (never zero), detail view, human-readable fmtWait formatting
 [4] aging counts: all five buckets, empty buckets, honest zeros, oldest
     waiting, scope = needs under the OTHER active filters, independent
     of the active review chip, unavailable -> key absent (N/A)
 [5] age filters: five valid values, invalid -> 400, combinations with
     review/other filters, pagination, rows always carry an age
 [6] queue ordering: default unchanged, oldest/newest, deterministic
     ties, never score-sorted, bounded page stability, invalid -> 400
 [7] frontend pins: banner oldest/breakdown, Waiting column, aging
     controls, URL allowlist + restore, detail line, server-authoritative
     refresh after review actions
 [8] review transitions: start/replay/complete/reopen move the aging
     summary with the server, stale -> 409, concurrency race moves aging
     exactly once, exactly one audit event, no client-side count drift
 [9] security: 401/403/400 surface, no client timestamp is authoritative,
     parameterized SQL, no secrets/PANs/token, no new audit event types
 [10] regression pins: Phase-113/114 behavior (state machine, URL
      restoration, queue navigation, history clamps, live feed, two
      script blocks, known bug fixes stay fixed)
 [11] canonical constants + qualified_datasets + RWV / promotion states
 [12] production artifact identity AFTER (byte-identical)

Everything runs against an ISOLATED temp DB_DIR: no row written here can
reach the shared production DB-3/DB-4, and no production artifact is ever
opened for write. No network, no model fitting, no threshold/release work.

Run from backend/:
  ../.venv/Scripts/python.exe scripts/phase115_queue_aging_test.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import sqlite3
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

DB_TMP = tempfile.mkdtemp(prefix="ps14_p115qy_")
os.environ["DB_DIR"] = DB_TMP
ADMIN_USER = "admin"
ADMIN_PASS = "p115qy-suite-pass-7f3d"
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
from src.audit_service.writer import flush_audit_queue  # noqa: E402

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

# ── [2] harness: seed isolated DB-3 aging rows ───────────────────────
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
NOW0 = datetime.now(timezone.utc).replace(microsecond=0)


def stamp(seconds_ago: int) -> str:
    """UTC wall-clock stamp, the format the backend stores scored_at in."""
    return (NOW0 - timedelta(seconds=seconds_ago)).strftime(
        "%Y-%m-%d %H:%M:%S")


# (key, age-seconds, band, score, initial transition)
# Bucket rows are flagged + UNREVIEWED = the Needs Review queue (13):
#   under_15m  = u1, b15lo          (2)
#   15m_1h     = b15*, h1lo         (2)
#   1h_4h      = h1*, tiea, tieb, q4lo (4)
#   4h_24h     = q4*, d24lo         (2)
#   over_24h   = d24*, old          (2)   * = exact boundary, lands in the
#   nul has scored_at = NULL -> outside every bucket (needs 13, sum 12)
# Higher bucket (age >= edge), which only upward drift can confirm.
SEEDS = [
    ("u1", 300, "high", 96, None),
    ("b15lo", 480, "high", 64, None),
    ("b15", 900, "high", 72, None),
    ("h1lo", 2700, "medium", 58, None),
    ("h1", 3600, "high", 88, None),
    ("tiea", 7200, "high", 96, None),
    ("tieb", 7200, "medium", 31, None),
    ("q4lo", 10800, "medium", 61, None),
    ("q4", 14400, "high", 77, None),
    ("d24lo", 72000, "high", 91, None),
    ("d24", 86400, "high", 84, None),
    ("old", 172800, "medium", 40, None),
    ("und", 108000, "high", 88, "start"),
    ("rev", 108000, "high", 84, "complete"),
    ("low", 108000, "low", 4, None),
]
for i, (key, age, band, score, _) in enumerate(SEEDS, start=1):
    con.execute(
        "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
        "risk_band, reason_codes, model_version, ml_score, rule_score, "
        "degraded, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (f"sc-p115qy-{key}", f"F-P115QY-{key.upper()}",
         f"evt-p115qy-{key}-{i:04d}", score, band, "[]",
         MC.CANONICAL_MODEL_VERSION, 0.5, 4.0, 0, stamp(age)))
# Missing-timestamp row: flagged + UNREVIEWED but scored_at is NULL ->
# waiting_seconds must be null (never zero), outside every bucket.
con.execute(
    "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
    "risk_band, reason_codes, model_version, ml_score, rule_score, "
    "degraded, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,NULL)",
    ("sc-p115qy-nul", "F-P115QY-NUL", "evt-p115qy-nul-0016", 79, "high",
     "[]", MC.CANONICAL_MODEL_VERSION, 0.5, 4.0, 0))
con.commit()
con.close()

EID = {key: f"evt-p115qy-{key}-{i:04d}"
       for i, (key, *_rest) in enumerate(SEEDS, start=1)}
EID["nul"] = "evt-p115qy-nul-0016"
AGE = {key: age for key, age, *_ in SEEDS}

# The queue in default (Phase-114) order: scored_at DESC = age ASC,
# NULLs last, ties by event_id ASC — 13 flagged UNREVIEWED rows.
QUEUE_DESC = ["u1", "b15lo", "b15", "h1lo", "h1", "tiea", "tieb",
              "q4lo", "q4", "d24lo", "d24", "old", "nul"]
QUEUE_ASC = ["nul", "old", "d24", "d24lo", "q4", "q4lo", "tiea", "tieb",
             "h1", "h1lo", "b15", "b15lo", "u1"]
BUCKETS0 = {"under_15m": 2, "15m_to_1h": 2, "1h_to_4h": 4,
            "4h_to_24h": 2, "over_24h": 2}

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


def rv_post(eid: str, action: str, body: dict | None = None,
            client: TestClient | None = None):
    h = dict(HDR)
    h["Content-Type"] = "application/json"
    return (client or c).post(
        f"/admin/api/transactions/{eid}/review/{action}",
        json={} if body is None else body, headers=h)


def list_tx(**kw):
    params = {"limit": 200}
    params.update(kw)
    return c.get("/admin/api/transactions", headers=HDR, params=params)


def eids(r) -> list[str]:
    return [x["event_id"] for x in r.json()["rows"]]


def rows_by_key(r) -> dict:
    out = {}
    for x in r.json()["rows"]:
        for k, v in EID.items():
            if x["event_id"] == v:
                out[k] = x
    return out


def aging(r) -> dict | None:
    return r.json().get("queue_aging")


def buckets(ag: dict | None) -> dict | None:
    """Bucket counts only — oldest_waiting_seconds grows every second,
    so cross-request comparisons must never include it (§5 measures
    against live server time)."""
    return None if ag is None else {k: ag[k] for k in BUCKETS0}


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


# Establish the non-needs states: und UNDER_REVIEW, rev REVIEWED.
r = rv_post(EID["und"], "start")
ok(r.status_code == 200 and r.json()["review_state"] == "UNDER_REVIEW",
   "seed: und walked to UNDER_REVIEW")
r = rv_post(EID["rev"], "start")
r = rv_post(EID["rev"], "complete")
ok(rv_get(EID["rev"]).json()["review_state"] == "REVIEWED",
   "seed: rev walked to REVIEWED")

# ── [3] aging calculations ────────────────────────────────────────────
r = list_tx()
ok(r.status_code == 200, f"transactions list -> 200 ({r.status_code})")
by = rows_by_key(r)
u1 = by["u1"]
ok(isinstance(u1["waiting_seconds"], int)
   and 300 <= u1["waiting_seconds"] <= 300 + 1800,
   f"valid waiting timestamp -> precise integer seconds "
   f"({u1['waiting_seconds']})")
ok(u1["waiting_since"] == u1["scored_at"],
   "waiting-start is the row's own authoritative scored_at")
ok(re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
                u1["waiting_since"] or "") is not None,
   f"waiting-start is a UTC wall-clock stamp ({u1['waiting_since']})")
ok(u1["scored_at"] == stamp(300),
   "stored stamp is UTC as seeded (no local offset shift)")
for key in ("und", "rev"):
    ok(by[key]["waiting_seconds"] is None
       and by[key]["waiting_since"] is None,
       f"{key}: not awaiting review -> waiting is null, not zero")
ok(by["nul"]["waiting_seconds"] is None
   and by["nul"]["waiting_since"] is None,
   "missing scored_at -> waiting_seconds null (never fabricated 0)")
ok(isinstance(by["low"]["waiting_seconds"], int),
   "unreviewed low-band row still has a state-honest waiting age")

# Detail view (§12): same authority, same null rules.
d = c.get(f"/admin/api/transactions/{EID['u1']}", headers=HDR).json()
ok(isinstance(d["waiting_seconds"], int)
   and d["waiting_since"] == d["score"]["scored_at"],
   "detail carries the authoritative waiting age for an awaiting row")
d = c.get(f"/admin/api/transactions/{EID['und']}", headers=HDR).json()
ok(d["waiting_seconds"] is None,
   "detail waiting is null once review started")
d = c.get(f"/admin/api/transactions/{EID['nul']}", headers=HDR).json()
ok(d["waiting_seconds"] is None,
   "detail waiting is null for the missing-timestamp row")

# Human-readable formatting (§11): presentation-only, executed against
# the REAL fmtWait extracted from the served shell.
shell = c.get("/admin").text
m_fmt = re.search(r"function fmtWait\(sec\) \{.*?\n        \}", shell, re.S)
ok(bool(m_fmt), "fmtWait helper present in the served shell")
NODE = shutil.which("node")
if NODE and m_fmt:
    prog = (f"const fmtWait = {m_fmt.group(0)}; "
            f"const cases = [null,4,59,60,240,2520,3600,3660,8280,"
            f"86400,97200]; "
            f"console.log(JSON.stringify(cases.map(v => fmtWait(v))));")
    proc = subprocess.run([NODE, "-e", prog], capture_output=True, text=True)
    got = json.loads(proc.stdout.strip()) if proc.returncode == 0 else None
    want = ["N/A", "<1m", "<1m", "1m", "4m", "42m", "1h", "1h 1m",
            "2h 18m", "1d", "1d 3h"]
    ok(got == want,
       f"fmtWait formats exactly as specified ({got} vs {want})")
else:
    ok(True, "fmtWait source pinned (node unavailable for live eval)")

# ── [4] aging counts: buckets, scope, honesty ─────────────────────────
r = list_tx()
ag = aging(r)
rc = r.json()["review_counts"]
ok(rc == {"needs": 13, "under": 1, "reviewed": 1},
   f"review_counts untouched by Phase 115 ({rc})")
ok(ag is not None and all(ag[k] == v for k, v in BUCKETS0.items()),
   f"all five aging buckets from the server ({ag})")
ok(sum(ag[k] for k in BUCKETS0) == 12,
   "buckets partition 12 rows; the missing-stamp row sits outside "
   "every bucket (never fabricated into <15m)")
ok(rc["needs"] - sum(ag[k] for k in BUCKETS0) == 1,
   "needs - buckets == the one row with no authoritative timestamp")
ok(isinstance(ag["oldest_waiting_seconds"], int)
   and 172800 <= ag["oldest_waiting_seconds"] <= 172800 + 1800,
   f"oldest_waiting_seconds is the 48h row ({ag['oldest_waiting_seconds']})")

# Independent of the ACTIVE review chip (§6): switching chips never
# reshapes the aging summary — it always describes the needs queue.
base_ag = ag
base_rc = rc
for chip in ("needs", "under", "reviewed"):
    rr = list_tx(review=chip)
    ok(buckets(aging(rr)) == buckets(base_ag),
       f"queue_aging unchanged under review={chip}")
    ok(rr.json()["review_counts"] == base_rc,
       f"review_counts unchanged under review={chip}")
rr = list_tx()
ok(buckets(aging(rr)) == buckets(base_ag)
   and rr.json()["review_counts"] == base_rc,
   "queue_aging + counts identical with no chip active")

# Scoped by the OTHER active filters (same semantics as the counts).
rr = list_tx(event_id=EID["tiea"])
ag1 = aging(rr)
ok(ag1 == {"under_15m": 0, "15m_to_1h": 0, "1h_to_4h": 1,
           "4h_to_24h": 0, "over_24h": 0,
           "oldest_waiting_seconds": ag1["oldest_waiting_seconds"]}
   and 7200 <= ag1["oldest_waiting_seconds"] <= 7200 + 1800,
   f"single-row scope: one bucket = 1, empty buckets = honest 0 ({ag1})")
ok(rr.json()["review_counts"] == {"needs": 1, "under": 0, "reviewed": 0},
   "counts follow the same single-row scope")
rr = list_tx(band="low")
aglow = aging(rr)
ok(aglow == {"under_15m": 0, "15m_to_1h": 0, "1h_to_4h": 0,
             "4h_to_24h": 0, "over_24h": 0,
             "oldest_waiting_seconds": None},
   f"empty queue -> all-zero buckets, oldest null ({aglow})")
rr = list_tx(since=stamp(100000))
ag2 = aging(rr)
ok(ag2["over_24h"] == 1 and ag2["oldest_waiting_seconds"] is not None
   and ag2["oldest_waiting_seconds"] <= 86400 + 1800,
   f"period scope narrows aging with the other filters ({ag2})")
rr = list_tx(flagged="true")
ok(buckets(aging(rr)) == buckets(base_ag),
   "flagged scope keeps the same needs-queue aging")

# Unavailable -> key absent (the console renders N/A, never 0).
r = c.get("/admin/api/transactions", headers=HDR,
          params={"limit": 50, "decision": "allow"})
ok(r.status_code == 200 and "queue_aging" not in r.json()
   and "review_counts" not in r.json(),
   "early empty return carries no aging summary (N/A, not fabricated 0)")

# ── [5] age filters (backend-side) ────────────────────────────────────
AGE_CASES = [
    ("lt15m", ["u1", "b15lo"]),
    ("15m_1h", ["b15", "h1lo"]),
    ("1h_4h", ["h1", "tiea", "tieb", "q4lo"]),
    ("4h_24h", ["q4", "d24lo"]),
    ("gt24h", ["d24", "low", "old"]),
]
total_hits = 0
for aval, keys in AGE_CASES:
    rr = list_tx(age=aval)
    ok(rr.status_code == 200, f"age={aval} -> 200 ({rr.status_code})")
    got = {k for k, v in EID.items() if v in eids(rr)}
    total_hits += rr.json()["total"]
    ok(got == set(keys), f"age={aval} selects exactly {keys} ({sorted(got)})")
    ok(all(x["waiting_seconds"] is not None for x in rr.json()["rows"]),
       f"age={aval}: every returned row carries an authoritative age")
ok(total_hits == 13,
   f"the five age filters partition every waiting row ({total_hits})")
ok(list_tx(age="lt15m").json()["total"] == 2
   and list_tx(review="needs", age="lt15m").json()["total"] == 2,
   "age combines with review=needs without changing meaning")
rr = list_tx(review="under", age="gt24h")
ok(rr.json()["total"] == 0,
   "under-review rows never match an age filter (no waiting age)")
ok(list_tx(age="lt15m", band="medium").json()["total"] == 0,
   "age combines with band filter (medium has no <15m rows)")
rr = list_tx(event_id=EID["old"], age="gt24h")
ok(rr.json()["total"] == 1 and eids(rr) == [EID["old"]],
   "age combines with an exact event_id filter")
rr = list_tx(event_id=EID["old"], age="lt15m")
ok(rr.json()["total"] == 0,
   "age mismatch under an exact ID stays empty")
rr = list_tx(age="gt24h", limit=1, offset=0)
ok(rr.json()["total"] == 3 and eids(rr) == [EID["d24"]]
   and rr.json()["has_more"] is True,
   "age filter paginates server-side (page 1 = d24)")
rr = list_tx(age="gt24h", limit=1, offset=1)
ok(eids(rr) == [EID["low"]] and rr.json()["has_more"] is True,
   "age filter page 2 continues the authoritative order (low)")
rr = list_tx(age="gt24h", limit=1, offset=2)
ok(eids(rr) == [EID["old"]] and rr.json()["has_more"] is False,
   "age filter page 3 ends at the oldest row (old)")
for bad in ("bogus", "15m", "LT15M", "gt24h' OR '1'='1", "gt24h;DROP"):
    rr = list_tx(age=bad)
    ok(rr.status_code == 400,
       f"malformed age {bad!r} -> 400 ({rr.status_code})")

# ── [6] queue ordering ────────────────────────────────────────────────
def keys_of(rr) -> list[str]:
    inv = {v: k for k, v in EID.items()}
    return [inv.get(x, x) for x in eids(rr)]


r_default = list_tx(review="needs")
ok(keys_of(r_default) == QUEUE_DESC,
   f"default order is EXACTLY the Phase-114 ordering ({keys_of(r_default)[:4]}...)")
r_newest = list_tx(review="needs", queue_sort="newest")
ok(keys_of(r_newest) == QUEUE_DESC,
   "queue_sort=newest == the default scored_at DESC order")
r_oldest = list_tx(review="needs", queue_sort="oldest")
ok(keys_of(r_oldest) == QUEUE_ASC,
   f"queue_sort=oldest -> authoritative waiting age, oldest first "
   f"({keys_of(r_oldest)[:4]}...)")
ok(keys_of(r_oldest)[:2] == ["nul", "old"],
   "oldest-first leads with the 48h row, never a score-driven pick")
ok(keys_of(r_default)[12] == "nul",
   "default order keeps the NULL-stamp row last (deterministic)")
ok(keys_of(r_oldest)[6:8] == ["tiea", "tieb"]
   and keys_of(r_default)[5:7] == ["tiea", "tieb"],
   "identical scored_at ties resolve by event_id ASC in BOTH orders")
ok(EID["tiea"] != EID["tieb"] and "tiea" in keys_of(r_oldest),
   "tie rows exist and stay deterministic")
# Score must not drive order: tiea (96) vs tieb (31) share a stamp, and
# old (score 40) must still lead oldest-first over u1 (score 96).
ok(keys_of(r_oldest)[-1] == "u1",
   "highest-score row is LAST when sorting oldest-first (not score-sorted)")
# Bounded pages stitch into the exact full order.
p1 = keys_of(list_tx(review="needs", queue_sort="oldest", limit=5, offset=0))
p2 = keys_of(list_tx(review="needs", queue_sort="oldest", limit=5, offset=5))
p3 = keys_of(list_tx(review="needs", queue_sort="oldest", limit=5, offset=10))
ok(p1 + p2 + p3 == QUEUE_ASC and len(set(p1 + p2 + p3)) == 13,
   "oldest-first pages are stable and gap/duplicate-free")
rr = list_tx(review="needs", age="gt24h", queue_sort="oldest")
ok(keys_of(rr) == ["old", "d24"],
   "age filter + review=needs + oldest order combine server-side "
   "(low-band never enters the needs queue)")
ok(buckets(aging(r_oldest)) == buckets(base_ag),
   "ordering never changes the aging summary")
for bad in ("Older", "oldest, event_id", "score", "oldest;DROP"):
    rr = list_tx(queue_sort=bad)
    ok(rr.status_code == 400,
       f"malformed queue_sort {bad!r} -> 400 ({rr.status_code})")

# ── [7] frontend pins (§9-§15) ────────────────────────────────────────
ok('id="f-queue-oldest"' in shell
   and "Oldest waiting: N/A" in shell,
   "banner has an Oldest waiting line defaulting to N/A")
ok('id="f-queue-aging"' in shell and "Aging breakdown: N/A" in shell,
   "banner has an aging breakdown defaulting to N/A")
i_rqa = shell.find("function renderQueueAging(aging)")
rqa_txt = shell[i_rqa:shell.find("function renderQueueHeader", i_rqa)]
ok(i_rqa > 0 and "Oldest waiting: " in rqa_txt
   and "oldest_waiting_seconds" in rqa_txt,
   "renderQueueAging derives Oldest from the server field only")
ok("aging['15m_to_1h']" in rqa_txt and "Date" not in rqa_txt,
   "breakdown renders server numbers with NO client clock")
ok("renderQueueAging(d.queue_aging);" in shell
   and shell.count("renderQueueAging(null);") >= 2
   and "renderQueueAging(fresh.queue_aging);" in shell,
   "aging renders on load, on failure (N/A) and after review actions")
ok("queue_aging: dj.queue_aging || null" in shell,
   "queue context re-fetch carries the fresh aging summary")
ok('id="f-age"' in shell and 'id="f-qsort"' in shell,
   "Waiting age + Queue order controls exist in the filter UI")
age_opts = re.findall(r'<option value="([^"]*)">[^<]*</option>',
                      shell.split('id="f-age"')[1].split("</select>")[0])
ok(age_opts == ["", "lt15m", "15m_1h", "1h_4h", "4h_24h", "gt24h"],
   f"Waiting age options are All + the five buckets ({age_opts})")
qsort_opts = re.findall(r'<option value="([^"]*)">',
                        shell.split('id="f-qsort"')[1].split("</select>")[0])
ok(qsort_opts == ["", "oldest", "newest"],
   f"Queue order options are Default/Oldest/Newest ({qsort_opts})")
ok("<th>Waiting</th>" in shell and 'data-label="Waiting"' in shell,
   "results table gained the Waiting column")
ok(shell.count('colspan="7"') >= 4,
   f"transaction-table placeholders match 7 columns "
   f"({shell.count('colspan=\"7\"')})")
ok("row.waiting_seconds == null" in shell
   and "fmtWait(row.waiting_seconds)" in shell,
   "row Waiting cell: server value or N/A, formatting is presentation-only")
ok('id="d-waiting-val"' in shell
   and "d.waiting_seconds == null" in shell,
   "detail shows Waiting for review (N/A when not awaiting)")
ok("if (wEl && state !== 'UNREVIEWED') wEl.innerHTML = naVal(null);" in shell
   and "const wEl = $('d-waiting-val');" in shell,
   "after a transition the waiting field follows the server state (N/A, "
   "never a stale duration)")
ok("function fmtWait(sec)" in shell and "isNaN(sec)" in shell
   and "'<1m'" in shell,
   "fmtWait handles null/unknown honestly, never 0")
# URL allowlist + restore (§14).
i_keys = shell.find("const TXN_URL_KEYS")
keys_txt = shell[i_keys:shell.find("]", i_keys)]
listed = re.findall(r"'([a-z_]+)'", keys_txt)
ok(listed == ['q', 'event_id', 'fraud_id', 'band', 'decision', 'min_score',
              'max_score', 'degraded', 'data_quality', 'review', 'since',
              'until', 'age', 'queue_sort'],
   f"URL allowlist extended with age + queue_sort, nothing dropped ({listed})")
ok("set('age', $('f-age') ? $('f-age').value : '');" in shell
   and "set('queue_sort', $('f-qsort') ? $('f-qsort').value : '');" in shell,
   "both controls travel in the query string")
ok("'f-review', 'f-range', 'f-age', 'f-qsort']" in shell,
   "URL restore resets both queue controls before applying")
i_apply = shell.find("function applyTxnQuery")
i_apply_end = shell.find("function pushTxnUrl")
apply_txt = shell[i_apply:i_apply_end]
ok("['lt15m', '15m_1h', '1h_4h', '4h_24h', 'gt24h'].includes(ageV)" in apply_txt,
   "restoring an age value validates against the allowlist")
ok("['oldest', 'newest'].includes(sortV)" in apply_txt,
   "restoring a queue_sort validates against the allowlist")
ok("txnOpenExact = false;" in apply_txt,
   "restoring list state can never auto-open a detail (Phase 114)")
ok("p.delete('age'); p.delete('queue_sort');" in shell,
   "aging controls are orthogonal to the status chips")
ok("'age', 'queue_sort'].includes(k)).length === 1" in shell
   and "'age', 'queue_sort'].includes(k));" in shell,
   "aging params never masquerade as an exact-ID search (§15)")
# Server-authoritative refresh after transitions (§13).
ok("async function afterReviewAction(action)" in shell
   and "const fresh = await fetchTxnCtx(txnCtx.query);" in shell
   and "renderQueueAging(fresh.queue_aging);" in shell,
   "after an action the queue + aging are re-fetched FIRST (server truth)")
ok(not re.search(r"(review_counts|queue_aging)\.\w+\s*--", shell),
   "no client-side count/aging decrement anywhere")
ok(not re.search(r"oldest_waiting_seconds\s*[:=]\s*\d", shell),
   "oldest-waiting is never hardcoded client-side")

# ── [8] review transitions move the aging summary ─────────────────────
r = list_tx()
counts0 = r.json()["review_counts"]
ag0 = aging(r)
ok(counts0["needs"] == 13 and ag0["under_15m"] == 2,
   "precondition: needs 13, under_15m 2")
rv = rv_post(EID["u1"], "start")
ok(rv.status_code == 200 and rv.json()["changed"] is True
   and rv.json()["version"] == 1,
   "start u1: UNREVIEWED -> UNDER_REVIEW (v1), server authoritative")
r = list_tx()
ok(r.json()["review_counts"] == {"needs": 12, "under": 2, "reviewed": 1},
   "counts follow the server transition (needs 13 -> 12)")
ag = aging(r)
ok(ag["under_15m"] == 1 and sum(ag[k] for k in BUCKETS0) == 11,
   f"aging summary refreshed with the transition ({ag})")
rr = list_tx(event_id=EID["u1"])
ok(rows_by_key(rr)["u1"]["waiting_seconds"] is None,
   "started row no longer waiting (null, not stale seconds)")
rv = rv_post(EID["u1"], "start")   # replay
ok(rv.status_code == 200 and rv.json()["changed"] is False,
   "replayed start is idempotent (changed=false)")
r = list_tx()
ok(r.json()["review_counts"] == {"needs": 12, "under": 2, "reviewed": 1}
   and aging(r)["under_15m"] == 1,
   "replay moves no counts and no buckets (no drift)")

# Stale mutation -> 409, record + aging untouched.
rv = rv_post(EID["b15lo"], "complete")
ok(rv.status_code == 409
   and "invalid review transition" in rv.json()["detail"],
   f"stale complete-from-UNREVIEWED -> 409 ({rv.status_code})")
d = rv_get(EID["b15lo"]).json()
ok(d["review_state"] == "UNREVIEWED" and d["version"] == 0,
   "rejected stale mutation left the record untouched")
r = list_tx()
ok(r.json()["review_counts"]["needs"] == 12
   and aging(r)["under_15m"] == 1,
   "rejected mutation moved no counts and no buckets")

# Concurrency: 6-way race on start settles aging exactly once.
before_cc = r.json()["review_counts"]


def hammer(eid: str, action: str, n: int = 6) -> list[dict]:
    results: list[dict] = []
    lock = threading.Lock()
    errors: list[str] = []

    def worker():
        try:
            tc = TestClient(app)
            rr2 = tc.post(
                f"/admin/api/transactions/{eid}/review/{action}",
                json={}, headers={**HDR,
                                  "Content-Type": "application/json"})
            with lock:
                if rr2.status_code != 200:
                    errors.append(f"{action}: HTTP {rr2.status_code}")
                else:
                    results.append(rr2.json())
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"{action}: {exc}")

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    ok(not errors, f"concurrent {action}: no failed requests ({errors})")
    return results


res = hammer(EID["b15"], "start")
ok(len(res) == 6 and sum(1 for x in res if x["changed"]) == 1,
   "6 concurrent starts -> exactly one change, rest are replays")
r = list_tx()
ok(r.json()["review_counts"] == {"needs": 11, "under": 3, "reviewed": 1},
   "raced start moved exactly one row between buckets")
ag = aging(r)
ok(ag["15m_to_1h"] == 1 and sum(ag[k] for k in BUCKETS0) == 10
   and r.json()["review_counts"]["needs"] - 10 == 1,
   f"aging summary moved exactly once under the race ({ag})")
starts = [json.loads(x["payload_summary"])
          for x in review_audit_rows("admin_review_started")]
ok(sum(1 for p in starts if p["event_id"] == EID["b15"]) == 1
   and sum(1 for p in starts if p["event_id"] == EID["u1"]) == 1,
   "races + replays produced exactly ONE audit event per transition")

# Complete + reopen keep following the server.
rv = rv_post(EID["b15"], "complete")
ok(rv.status_code == 200 and rv.json()["review_state"] == "REVIEWED"
   and rv.json()["version"] == 2,
   "complete b15: UNDER_REVIEW -> REVIEWED (v2)")
rv = rv_post(EID["b15"], "reopen")
ok(rv.status_code == 200 and rv.json()["review_state"] == "UNDER_REVIEW"
   and rv.json()["version"] == 3,
   "reopen b15: REVIEWED -> UNDER_REVIEW (v3)")
r = list_tx()
ok(r.json()["review_counts"] == {"needs": 11, "under": 3, "reviewed": 1},
   "reopen returns it to under; needs stays 11 (no phantom waiting row)")
ok(aging(r)["15m_to_1h"] == 1,
   "aging unchanged by complete/reopen (row left the queue at start)")

# ── [9] security ──────────────────────────────────────────────────────
fresh = TestClient(app)
for path in ("/admin/api/transactions?review=needs&age=gt24h"
             "&queue_sort=oldest",
             "/admin/api/summary",
             f"/admin/api/transactions/{EID['u1']}",
             f"/admin/api/transactions/{EID['u1']}/review"):
    rr = fresh.get(path)
    ok(rr.status_code == 401,
       f"unauthenticated {path.split('?')[0].split('/admin/api')[1]} "
       f"-> 401 ({rr.status_code})")
rr = fresh.get("/admin/api/transactions?age=lt15m",
               headers={"Authorization": "Bearer not-a-real-token"})
ok(rr.status_code == 401, f"invalid bearer token -> 401 ({rr.status_code})")
rr = fresh.get("/admin/api/transactions?age=lt15m",
               cookies={"admin_session": "garbage-session"})
ok(rr.status_code == 401, f"invalid session cookie -> 401 ({rr.status_code})")

nohead = TestClient(app)
nohead.cookies.update(c.cookies)
rr = nohead.post(f"/admin/api/transactions/{EID['und']}/review/start")
ok(rr.status_code == 403 and "X-Requested-With" in rr.json()["detail"],
   f"cookie POST without X-Requested-With -> 403 ({rr.status_code})")
rr = nohead.post(f"/admin/api/transactions/{EID['und']}/review/start",
                 headers={"X-Requested-With": "XMLHttpRequest"})
ok(rr.status_code == 200 and rr.json()["changed"] is False,
   f"cookie POST with the header succeeds as a harmless replay ({rr.status_code})")

for bad in ("lt15m' OR '1'='1", "gt24h--", "<15m", "999"):
    rr = list_tx(age=bad)
    ok(rr.status_code == 400,
       f"injection-shaped age {bad!r} -> 400 ({rr.status_code})")
for bad in ("oldest'--", "1; SELECT 1", "auto"):
    rr = list_tx(queue_sort=bad)
    ok(rr.status_code == 400,
       f"injection-shaped queue_sort {bad!r} -> 400 ({rr.status_code})")

# No client-supplied waiting timestamp is ever authoritative (§4/§17).
sig = inspect.signature(fm.admin_api_transactions)
ok("waiting_since" not in sig.parameters
   and "waiting_seconds" not in sig.parameters
   and "scored_at" not in sig.parameters,
   "list endpoint accepts NO client-side timestamp parameter")
rr = list_tx(event_id=EID["b15lo"],
             waiting_seconds=999999,
             waiting_since="2020-01-01T00:00:00")
row = rows_by_key(rr)["b15lo"]
ok(rr.status_code == 200
   and row["waiting_since"] == row["scored_at"]
   and row["waiting_seconds"] != 999999,
   "client waiting_seconds/waiting_since are ignored — server decides")

# Parameterized SQL, no interpolation of filter values.
src = (BACKEND / "src" / "front_service" / "main.py").read_text(
    encoding="utf-8").replace("\r\n", "\n")
ok('_AGE_SQL = ("CAST(strftime(\'%s\',\'now\') - strftime(\'%s\', scored_at) "' in src
   or "_AGE_SQL = (" in src,
   "one shared server-side age expression")
ok('where.append(f"{_AGE_SQL} >= ?")' in src
   and 'where.append(f"{_AGE_SQL} < ?")' in src,
   "age bounds travel as ? parameters (never string-interpolated)")
ok("params.append(lo)" in src and "params.append(hi)" in src,
   "age bounds are bound, not f-string values")
ok("age not in _AGE_FILTERS" in src and "queue_sort not in _QUEUE_SORTS" in src,
   "allowlists gate both new parameters")

# No secrets / PANs / token in any response touched by this phase.
for path in ("/admin/api/summary",
             "/admin/api/transactions?review=needs&age=gt24h"
             "&queue_sort=oldest",
             f"/admin/api/transactions/{EID['u1']}",
             f"/admin/api/transactions/{EID['u1']}/review"):
    txt = c.get(path, headers=HDR).text
    ok("passphrase" not in txt and "blob_key" not in txt
       and ADMIN_PASS not in txt and "TOTP" not in txt
       and "otpauth" not in txt,
       f"no secrets in {path.split('?')[0]}")
    ok(not re.search(r"\b4[0-9]{15}\b|\b3[0-9]{14}\b", txt),
       f"no PAN-shaped values in {path.split('?')[0]}")
    ok("token" not in txt.lower() or TOKEN not in txt,
       f"session token never echoed in {path.split('?')[0]}")

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

# Aging is read-only: no new audit event types.
known = {x["event_type"] for x in review_audit_rows()}
ok(known <= {"score_generated", "admin_tx_search", "admin_tx_view",
             "admin_login_success", "admin_review_started",
             "admin_review_completed", "admin_review_reopened",
             "admin_review_note_added"},
   f"aging added no new audit event type ({sorted(known)})")

# ── [10] regression pins: Phase-113/114 behavior ──────────────────────
blocks = re.findall(r"<script>(.*?)</script>", shell, re.S)
ok(len(blocks) == 2, f"exactly two script blocks ({len(blocks)})")
js = "\n".join(blocks)
bare = [m.group(0) for m in re.finditer(r"(?<!\[\.\.\.)\b\w+\.keys\(\)\.length\b", js)
        if not js[max(0, m.start() - 4):m.start()] == "[..."]
ok(not bare, f"URLSearchParams iterator bug stays fixed: {bare}")
ok("const wantOpen = txnOpenExact;" in shell
   and "if (wantOpen && lone && quick0" in shell,
   "auto-open is gated on an explicit search only")
i_pop = js.find("addEventListener('popstate'")
ok(i_pop > 0 and "applyTxnQuery(new URLSearchParams" in js[i_pop:i_pop + 900]
   and "push: false" in js[i_pop:i_pop + 900],
   "popstate re-applies query state WITHOUT pushing (no ping-pong)")
ok("if (!r.detail) applyTxnQuery(new URLSearchParams(window.location.search));"
   in js,
   "popstate only restores LIST entries — no Back-button ping-pong")
ok(js.count("window.location.pathname + window.location.search !== target")
   >= 2, "both push paths are idempotent (identical target never pushes)")
ok("if (livePaused) { setConn('PAUSED'); return; }" in js
   and js.count("if (livePaused) { setConn('PAUSED'); return; }") >= 2,
   "live-monitor pause race stays fixed (guard before and after await)")
rchips = re.findall(r'data-rchip="([a-z]+)"', shell)
ok(rchips == ["all", "needs", "under", "reviewed"],
   f"review chips stay All/Needs/Under/Reviewed ({rchips})")
ok(re.findall(r'data-chip="([a-z]+)"', shell)
   == ["all", "flagged", "approved", "blocked"],
   "original quick-filter chips untouched")
ok("function renderQueuePos()" in shell
   and "Page ${txnPage + 1} · Transaction ${idx + 1} of ${txnCtx.ids.length}"
   in shell,
   "list-side queue position still server-derived")
ok("const REVIEW_QUEUE_LABEL = { needs: 'Needs Review'" in shell
   and "function txnQueueLabel()" in shell,
   "queue labels + label resolver intact")
i_lq = shell.find("function txnQueueLabel")
ok(i_lq > 0 and re.search(
        r"function txnQueueLabel\(\) \{[^}]*return null;",
        shell[i_lq:], re.S),
   "lone exact-ID search yields NO queue label (return null path)")
ok("function updateTxnNav()" in shell
   and "prev.disabled = !(idx > 0);" in shell
   and "(txnCtx.offset + idx + 1) + ' of ' + txnCtx.total" in shell,
   "nav edges + server-derived position intact")
ok("min(int(limit), 200)" in src and "(server-capped)" in shell,
   "bounded history pagination intact (limit <= 200)")
ok("· ${esc(reviewWord(f.review_state))}" in shell and "Investigate" in shell,
   "Live Monitor Investigate + review wording intact")
ok("maxlength=\"2000\"" in shell and "admin_review_note_added" in shell,
   "notes stay Phase-113: bounded, same audit event")
ok("bulk" not in shell.lower(), "no bulk mutation affordance")
ok("Math.random" not in shell, "no random navigation anywhere")
# State machine itself unchanged (Phase 113).
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
                           "gt24h": (86400, None)},
   "filter allowlists are pinned, half-open, lower-inclusive")

# ── [11] canonical constants + governance states ──────────────────────
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

# ── [12] production artifact identity AFTER ───────────────────────────
after = snapshot_artifacts()
after["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
changed = [k for k in WATCHED if before.get(k) != after.get(k)]
ok(not changed,
   f"production artifacts byte-identical across the suite ({changed})")

print()
print(f"PHASE 115 QUEUE AGING: {n_assert} assertions, "
      f"{len(failures)} failures")
if failures:
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PHASE 115 QUEUE AGING: ALL PASS")
