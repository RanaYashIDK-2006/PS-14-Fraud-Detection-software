"""Phase 114 — Flagged transaction history & investigation continuity: test suite.

Sections
--------
 [1] production artifact identity BEFORE the suite runs
 [2] harness: isolated temp DB_DIR, seeded rows spanning every period
     window, admin login
 [3] bounded pagination: server-capped limit, offset boundaries, explicit
     page size, authoritative total, non-overlapping pages
 [4] deterministic sort: scored_at DESC with event_id tiebreak, NULL
     timestamps last — no client or heuristic reordering
 [5] date/period filters: since/until validation, UTC wall-clock format,
     Today / 24h / 7d / 30d / All windows over the backend query
 [6] combined filters: flagged + period, flagged + decision, flagged +
     degraded, flagged + data-quality, exact event/fraud ID lookup
 [7] malformed parameters + SQL-injection strings -> safely rejected, and
     the store survives them
 [8] API authorization: unauthenticated/invalid session -> 401, unknown
     event -> 404, no secrets in responses or shell
 [9] honest null handling: absent timestamps/scores stay null, never
     fabricated
[10] frontend pins: period select, dynamic FLAGGED TRANSACTIONS title,
     shareable URL state, back/prev/next navigation, investigation
     continuity strip, explicit empty/error states, live Investigate →
[11] regression pins: the four Phase-113 bugs must not return
[12] canonical constants + qualified datasets + RWV/promotion states
[13] production artifact identity AFTER (byte-identical)

Everything runs against an ISOLATED temp DB_DIR: no row written here can
reach the shared production DB-3/DB-4, and no production artifact is ever
opened for write. No network, no model fitting, no threshold/release work.

Run from backend/:
  ../.venv/Scripts/python.exe scripts/phase114_transaction_history_test.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DB_TMP = tempfile.mkdtemp(prefix="ps14_p114_")
os.environ["DB_DIR"] = DB_TMP
ADMIN_USER = "admin"
ADMIN_PASS = "p114-suite-pass-5b8e"
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

# ── [2] harness: seed isolated DB-3 (+ DB-4 for decision/DQ) ─────────
import sqlite3  # noqa: E402

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
BASE = datetime.utcnow()

def ts(hours_ago: float) -> str:
    return (BASE - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S.%f")

# event_id, fraud_id, score, band, codes, degraded, scored_at (hours ago /
# None = never recorded). Windows are chosen with wide margins so second-
# level test runtime can never flip an expectation.
E_NEW = "evt-p114-new-001"       # 5 min ago        -> today/24h/7d/30d
E_3H = "evt-p114-3h-002"         # 3 h ago          -> 24h/7d/30d
E_DQB = "evt-p114-dq-003"        # 2 h ago          -> flagged + DQ blocked
E_DEC = "evt-p114-dec-004"       # 2 h ago          -> flagged + decision
E_NODC = "evt-p114-nod-005"      # 2 h ago          -> flagged, decision null
E_1D = "evt-p114-1d-006"         # 30 h ago         -> NOT 24h, yes 7d/30d
E_3D = "evt-p114-3d-007"         # 3 d ago          -> 7d/30d, degraded
E_10D = "evt-p114-10d-008"       # 10 d ago         -> 30d only
E_40D = "evt-p114-40d-009"       # 40 d ago         -> All time only
E_LOW = "evt-p114-low-010"       # 10 min ago       -> never "flagged"
E_TIE_A = "evt-p114-tie-011a"    # exact same ts as B -> event_id tiebreak
E_TIE_B = "evt-p114-tie-011b"
E_NULL = "evt-p114-null-012"     # scored_at never recorded -> honest null
TIE_STAMP = "2020-05-05 05:05:05.000000"

SEEDS = [
    (E_NEW, "F-P114-NEW1", 97, "high", ["VELOCITY_HIGH"], 0, ts(5 / 60)),
    (E_3H, "F-P114-3H", 55, "medium", [], 0, ts(3)),
    (E_DQB, "F-P114-DQB", 91, "high", ["DATA_QUALITY_BLOCKED"], 0, ts(2)),
    (E_DEC, "F-P114-DEC", 88, "high", ["AMOUNT_ANOMALY"], 0, ts(2)),
    (E_NODC, "F-P114-NOD", 77, "high", ["VELOCITY_HIGH"], 0, ts(2)),
    (E_1D, "F-P114-1DAY", 64, "medium", [], 0, ts(30)),
    (E_3D, "F-P114-3DAY", 93, "high", ["AMOUNT_ANOMALY"], 1, ts(72)),
    (E_10D, "F-P114-10DAY", 51, "medium", [], 0, ts(240)),
    (E_40D, "F-P114-40DAY", 82, "high", ["VELOCITY_HIGH"], 0, ts(960)),
    (E_LOW, "F-P114-LOW7", 7, "low", [], 0, ts(10 / 60)),
    (E_TIE_A, "F-P114-TIEA", 70, "high", [], 0, TIE_STAMP),
    (E_TIE_B, "F-P114-TIEB", 71, "high", [], 0, TIE_STAMP),
    (E_NULL, "F-P114-NULL", 86, "high", [], 0, None),
]
for i, (eid, fid, score, band, codes, deg, sat) in enumerate(SEEDS, 1):
    con.execute(
        "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
        "risk_band, reason_codes, model_version, ml_score, rule_score, "
        "degraded, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (f"sc-p114-{i:03d}", fid, eid, score, band, json.dumps(codes),
         MC.CANONICAL_MODEL_VERSION, None if eid == E_NULL else 0.91,
         4.2, deg, sat))
con.commit()
con.close()

# One recorded decision payload (DB-4) so the decision filter and the
# detail view have authoritative data to resolve against.
append_audit_event("F-P114-DEC", "score_generated", {
    "event_id": E_DEC, "fraud_id": "F-P114-DEC", "risk_score": 88,
    "risk_band": "high", "reason_codes": ["AMOUNT_ANOMALY"],
    "model_version": MC.CANONICAL_MODEL_VERSION, "ml_score": 0.91,
    "rule_score": 4.2, "degraded": False, "decision": "step_up",
    "feature_version": MC.CANONICAL_FEATURE_VERSION,
    "runtime_state": "READY",
    "runtime_release_id": MC.CANONICAL_LEGACY_RELEASE_ID,
})
# One data-quality block event for the data_quality=blocked filter.
append_audit_event("F-P114-DQB", "data_quality_blocked", {
    "event_id": E_DQB, "fraud_id": "F-P114-DQB",
    "data_quality_status": "blocked", "decision": "verify",
})
flush_audit_queue(timeout=5)

c = TestClient(app)
r = c.post("/admin/login",
           json={"username": ADMIN_USER, "passphrase": ADMIN_PASS})
ok(r.status_code == 200 and "token" in r.json(),
   f"login -> 200 + token ({r.status_code})")
TOKEN = r.json()["token"]
HDR = {"Authorization": f"Bearer {TOKEN}"}


def tx(params: dict, client: TestClient | None = None, headers=None):
    return (client or c).get("/admin/api/transactions",
                             params=params,
                             headers=HDR if headers is None else headers)


FLAGGED_ALL = {e for e, *_ in SEEDS if e != E_LOW}   # band != 'low'
TOTAL = len(SEEDS)

# ── [3] bounded pagination ────────────────────────────────────────────
r = tx({"limit": 999})
d = r.json()
ok(r.status_code == 200 and d["limit"] == 200,
   f"limit=999 is server-capped at 200 ({d.get('limit')})")
r = tx({"limit": 0})
ok(r.json()["limit"] == 1, "limit=0 floors to 1 (never an unbounded read)")
r = tx({"limit": -5})
ok(r.json()["limit"] == 1, "negative limit floors to 1")
r = tx({"offset": -1})
ok(r.status_code == 400, f"negative offset -> 400 ({r.status_code})")
r = tx({"offset": 10001})
ok(r.status_code == 400, f"offset 10001 -> 400 ({r.status_code})")
r = tx({"offset": 10000})
ok(r.status_code == 200, f"offset 10000 stays legal ({r.status_code})")

r = tx({})
d = r.json()
ok(d["total"] == TOTAL and d["offset"] == 0,
   f"unfiltered total is authoritative ({d['total']}/{TOTAL})")
ok(d["has_more"] is False, "single page -> has_more false, no phantom tail")
ok("generated_at" in d, "response stamps its own generated_at")
rows_all = d["rows"]
ok(len(rows_all) == TOTAL,
   f"one page carries every seeded row ({len(rows_all)})")

# Pages are bounded, disjoint and preserve the server order.
r1 = tx({"limit": 2, "offset": 0}).json()
r2 = tx({"limit": 2, "offset": 2}).json()
ids1 = [x["event_id"] for x in r1["rows"]]
ids2 = [x["event_id"] for x in r2["rows"]]
ok(len(ids1) == 2 and len(ids2) == 2 and not (set(ids1) & set(ids2)),
   "adjacent pages are bounded and disjoint")
ok(ids1 + ids2 == [x["event_id"] for x in rows_all[:4]],
   "pages slice the same authoritative order (no page-local re-sort)")
ok(r1["total"] == r2["total"] == TOTAL,
   "total is stable across pages — computed server-side once per query")

# ── [4] deterministic sort: scored_at DESC, event_id tiebreak ────────
order_ok = True
nulls_seen = False
for a, b in zip(rows_all, rows_all[1:]):
    sa, sb = a["scored_at"], b["scored_at"]
    if sa is None:
        nulls_seen = True
        continue
    if nulls_seen:
        order_ok = False          # a non-null row after a null row
        break
    if sb is None:
        continue
    if (sa, b["event_id"]) < (sb, a["event_id"]) and sa == sb:
        order_ok = False          # tie not broken by event_id ASC
        break
    if sa < sb:
        order_ok = False          # strictly older after newer
        break
ok(order_ok, "rows ordered scored_at DESC with event_id ASC tiebreak")
ok(rows_all[-1]["event_id"] == E_NULL and rows_all[-1]["scored_at"] is None,
   "unknown timestamps sort last, never fabricated into a date")
ties = [x["event_id"] for x in rows_all if x["scored_at"] == TIE_STAMP]
ok(ties == [E_TIE_A, E_TIE_B],
   f"identical timestamps resolve deterministically ({ties})")

# ── [5] date/period filters (backend query, not client filtering) ────
r = tx({"since": "2020-01-01"})
d = r.json()
ok(r.status_code == 200 and d["total"] == TOTAL - 1
   and E_NULL not in {x["event_id"] for x in d["rows"]},
   "since=YYYY-MM-DD accepted; rows with no timestamp stay out (UTC)")
r = tx({"since": "2020-01-01 00:00:00", "until": "2020-01-02 00:00:00"})
ok(r.status_code == 200 and r.json()["total"] == 0,
   "bounded custom window filters server-side (old window -> empty)")
r = tx({"since": (BASE - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")})
d = r.json()
ok(r.status_code == 200 and E_NEW in {x["event_id"] for x in d["rows"]}
   and E_40D not in {x["event_id"] for x in d["rows"]},
   "recent since includes the newest row and excludes old ones")
for label, bad in (
        ("ISO-Z", "2026-01-01T00:00:00Z"),
        ("relative", "yesterday"),
        ("impossible date", "2026-99-99"),
        ("empty-ish", "0000-00-00"),
        ("injection", "2020-01-01' OR '1'='1"),
):
    r = tx({"since": bad})
    ok(r.status_code == 400,
       f"malformed since ({label}) -> 400 ({r.status_code})")
r = tx({"since": "2026-01-02", "until": "2026-01-01"})
ok(r.status_code == 400, f"since > until -> 400 ({r.status_code})")

def period_ids(since_hours: float | None, flagged: bool = False) -> set:
    p = {}
    if since_hours is not None:
        p["since"] = (BASE - timedelta(hours=since_hours)).strftime(
            "%Y-%m-%d %H:%M:%S")
    if flagged:
        p["flagged"] = "true"
    r = tx(p)
    assert r.status_code == 200, (p, r.status_code, r.text)
    return {x["event_id"] for x in r.json()["rows"]}

# Today: the fresh row is in; every older row is out. Rows a few hours
# old may legitimately fall on either side of the UTC midnight boundary,
# so only the deterministic edges are asserted.
r = tx({"since": datetime.utcnow().strftime("%Y-%m-%d")})
today_ids = {x["event_id"] for x in r.json()["rows"]}
ok(E_NEW in today_ids,
   "Today window includes the row scored minutes ago")
ok(not ({E_1D, E_3D, E_10D, E_40D, E_TIE_A, E_TIE_B} & today_ids),
   "Today window excludes every older row")

h24 = period_ids(24, flagged=True)
ok({E_NEW, E_3H, E_DQB, E_DEC, E_NODC} <= h24
   and not ({E_1D, E_3D, E_10D, E_40D, E_TIE_A, E_TIE_B, E_NULL} & h24),
   "Last 24 hours = exactly the fresh flagged rows (server-side)")
h7 = period_ids(24 * 7, flagged=True)
ok({E_NEW, E_3H, E_DQB, E_DEC, E_NODC, E_1D, E_3D} <= h7
   and not ({E_10D, E_40D, E_TIE_A, E_TIE_B, E_NULL} & h7),
   "Last 7 days adds the 1d/3d rows, still excludes 10d+")
h30 = period_ids(24 * 30, flagged=True)
ok({E_NEW, E_3H, E_DQB, E_DEC, E_NODC, E_1D, E_3D, E_10D} <= h30
   and not ({E_40D, E_TIE_A, E_TIE_B, E_NULL} & h30),
   "Last 30 days adds the 10d row, excludes 40d and unknown timestamps")
all_f = period_ids(None, flagged=True)
ok(all_f == FLAGGED_ALL,
   f"All time flagged set is complete and low-band free ({len(all_f)})")
ok(E_NULL not in (h24 | h7 | h30),
   "rows with no recorded timestamp are never guessed into a window")

# ── [6] combined filters + exact lookups ──────────────────────────────
r = tx({"flagged": "true", "since": (BASE - timedelta(hours=24)).strftime(
    "%Y-%m-%d %H:%M:%S"), "degraded": "false"})
ids = {x["event_id"] for x in r.json()["rows"]}
ok(E_3D not in ids and E_NEW in ids,
   "flagged + period + non-degraded combine on the server")
r = tx({"flagged": "true", "degraded": "true"})
ids = {x["event_id"] for x in r.json()["rows"]}
ok(ids == {E_3D}, f"flagged + degraded -> exactly the degraded row ({ids})")
r = tx({"flagged": "true", "decision": "step_up"})
ids = {x["event_id"] for x in r.json()["rows"]}
ok(ids == {E_DEC},
   f"flagged + decision resolves from DB-4 payloads ({ids})")
ok(r.json()["decision_source"] == "audit_payload",
   "decision filter reports its authoritative source")
r = tx({"decision": "allow"})
ok(r.json()["rows"] == [] and r.json()["total"] == 0,
   "decision with no matching payload is an honest empty set, not a guess")
r = tx({"flagged": "true", "data_quality": "blocked"})
ids = {x["event_id"] for x in r.json()["rows"]}
ok(ids == {E_DQB}, f"flagged + data_quality=blocked -> the DQ row ({ids})")
r = tx({"flagged": "true", "band": "high"})
ids = {x["event_id"] for x in r.json()["rows"]}
ok(ids == ({E_NEW, E_DQB, E_DEC, E_NODC, E_3D, E_40D,
            E_TIE_A, E_TIE_B, E_NULL}),
   "flagged + band combine without client-side filtering")
r = tx({"flagged": "true", "min_score": 90})
ids = {x["event_id"] for x in r.json()["rows"]}
ok(ids == {E_NEW, E_DQB, E_3D},
   "flagged + min_score bounded server-side")

r = tx({"event_id": E_3D})
d = r.json()
ok(d["total"] == 1 and d["rows"][0]["event_id"] == E_3D,
   "exact event_id lookup returns exactly that event")
r = tx({"fraud_id": "F-P114-DEC"})
d = r.json()
ok(d["total"] == 1 and d["rows"][0]["event_id"] == E_DEC,
   "exact fraud_id lookup returns exactly that event")
r = tx({"event_id": "evt-p114-none-999"})
ok(r.status_code == 200 and r.json()["total"] == 0,
   "unknown-but-well-formed event_id -> honest empty result (no 500)")

# Detail endpoint: exact 200 / clean 404 / malformed 400.
r = c.get(f"/admin/api/transactions/{E_DEC}", headers=HDR)
ok(r.status_code == 200, f"detail -> 200 ({r.status_code})")
d = r.json()
ok(d["event_id"] == E_DEC and d["decision"] == "step_up",
   "detail carries the recorded decision payload")
ok(d["score"]["scored_at"] is not None, "detail exposes the recorded ts")
r = c.get("/admin/api/transactions/evt-p114-none-999", headers=HDR)
ok(r.status_code == 404, f"unknown event detail -> 404 ({r.status_code})")
r = c.get("/admin/api/transactions/evt!bad", headers=HDR)
ok(r.status_code == 400, f"malformed event detail -> 400 ({r.status_code})")

# ── [7] malformed parameters + SQL injection ──────────────────────────
for label, params in (
        ("band", {"band": "extreme"}),
        ("band injection", {"band": "low' OR 1=1--"}),
        ("decision", {"decision": "reject"}),
        ("decision injection", {"decision": "allow'--"}),
        ("review", {"review": "whatever"}),
        ("data_quality", {"data_quality": "maybe"}),
        ("degraded", {"degraded": "maybe"}),
        ("min_score", {"min_score": 101}),
        ("max_score", {"max_score": -1}),
        ("min>max", {"min_score": 80, "max_score": 20}),
        ("event_id", {"event_id": "evt' OR 1=1--"}),
        ("event_id long", {"event_id": "e" * 129}),
        ("fraud_id", {"fraud_id": "F' UNION SELECT * FROM risk_scores--"}),
        ("fraud_id short", {"fraud_id": "F1"}),
):
    r = tx(params)
    ok(r.status_code in (400, 422),
       f"malformed {label} -> rejected ({r.status_code})")

# Injection strings neither leak nor damage: the store still answers with
# exactly the seeded rows afterwards.
r = tx({"since": "2020-01-01' OR '1'='1",
        "band": "high' UNION SELECT fraud_id, NULL FROM risk_scores--"})
ok(r.status_code == 400, f"chained injection attempt -> 400 ({r.status_code})")
r = tx({})
d = r.json()
ok(d["total"] == TOTAL
   and {x["event_id"] for x in d["rows"]} == {e for e, *_ in SEEDS},
   "store intact after injection attempts (parameterized queries only)")

# ── [8] authorization + secrets ───────────────────────────────────────
fresh = TestClient(app)
for params in ({}, {"flagged": "true"}, {"since": "2020-01-01", "limit": 5},
               {"event_id": E_NEW}, {"q": "evt"}):
    r = fresh.get("/admin/api/transactions", params=params)
    ok(r.status_code == 401,
       f"unauthenticated list {params} -> 401 ({r.status_code})")
r = fresh.get(f"/admin/api/transactions/{E_NEW}")
ok(r.status_code == 401, f"unauthenticated detail -> 401 ({r.status_code})")
r = fresh.get("/admin/api/transactions",
              headers={"Authorization": "Bearer not-a-real-token"})
ok(r.status_code == 401, f"invalid bearer -> 401 ({r.status_code})")
r = fresh.get("/admin/api/transactions",
              cookies={"admin_session": "garbage-session"})
ok(r.status_code == 401, f"invalid session cookie -> 401 ({r.status_code})")

for path in ("/admin/api/transactions?flagged=true&limit=5",
             f"/admin/api/transactions/{E_DEC}",
             f"/admin/api/transactions/{E_DEC}/review"):
    txt = c.get(path, headers=HDR).text
    ok("passphrase" not in txt and "blob_key" not in txt
       and ADMIN_PASS not in txt and "TOTP" not in txt,
       f"no secrets in {path.split('?')[0]}")
    ok(not re.search(r"\b4[0-9]{15}\b|\b3[0-9]{14}\b", txt),
       f"no PAN-shaped values in {path.split('?')[0]}")

shell = c.get("/admin").text
ok(ADMIN_PASS not in shell, "admin passphrase never appears in the shell")
for pat, label in (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)[\"']Bearer\s+[A-Za-z0-9._\-]{40,}[\"']"),
     "hard-coded bearer token"),
    (re.compile(r"[?&](?:access_token|api_key|apikey|secret|totp)=[A-Za-z0-9]"),
     "credential in a URL query string"),
):
    ok(not pat.search(shell), f"shell carries no {label}")

# ── [9] honest null handling ──────────────────────────────────────────
null_row = next(x for x in rows_all if x["event_id"] == E_NULL)
ok(null_row["scored_at"] is None,
   "absent scored_at stays null in the API — never an epoch or 'N/A'")
ok(null_row["ml_score"] is None,
   "absent ml_score stays null — never a fabricated 0.0")
nodc = next(x for x in rows_all if x["event_id"] == E_NODC)
ok(nodc["decision"] is None,
   "no recorded decision stays null (unmatchable, never guessed)")
ok(nodc["review_state"] == "UNREVIEWED" and isinstance(nodc["degraded"], bool),
   "review/degraded fields are always explicit server values")

# ── [10] frontend pins ────────────────────────────────────────────────
# Compact period filter (§3): one select, five choices, backend-mapped.
ok('id="f-range"' in shell and 'aria-label="Date range"' in shell,
   "compact period select present on the primary screen")
for opt in ('<option value="today">Today</option>',
            '<option value="24h">Last 24 hours</option>',
            '<option value="7d">Last 7 days</option>',
            '<option value="30d">Last 30 days</option>',
            '<option value="" selected>All time</option>'):
    ok(opt in shell, f"period choice present: {opt}")
ok("function txRangeSince" in shell
   and "utcStamp(now - 7 * 24 * 3600e3)" in shell
   and "utcStamp(now - 30 * 24 * 3600e3)" in shell
   and "utcStamp(now - 24 * 3600e3)" in shell,
   "period maps to UTC since-params sent to the backend")
ok("new Date($('f-since').value).toISOString()" not in shell,
   "custom since no longer serializes ISO-Z (would 400 on the backend)")
ok("function localToUtcStamp" in shell and "toISOString().slice(0, 19)" in shell,
   "all frontend timestamps normalize to YYYY-MM-DD HH:MM:SS UTC")
ok("set('since', customSince" not in shell and "localToUtcStamp(customSince)" in shell,
   "custom window feeds the validated since/until params")

# Dynamic title + list chrome (§2).
ok('id="tx-title"' in shell, "transactions title has a dynamic id")
ok("'FLAGGED TRANSACTIONS'" in shell
   and "kind === 'flagged' ? 'FLAGGED TRANSACTIONS'" in shell,
   "flagged view titles itself FLAGGED TRANSACTIONS")

# Shareable URL state (§7).
for fn in ("function txnUrlQuery", "function applyTxnQuery",
           "function pushTxnUrl"):
    ok(fn in shell, f"URL-state helper present: {fn}")
ok("sp.get('status') === 'flagged'" in shell,
   "?status=flagged deep link honoured through applyTxnQuery")
ok("p.set('page', String(page))" in shell and "+page - 1" in shell,
   "page round-trips through the URL (1-based outside, 0-based inside)")
ok("p.set('range', range)" in shell
   and "['today', '24h', '7d', '30d'].includes(range)" in shell,
   "period round-trips as range=..., never as raw timestamps")
ok("p.set('status', 'flagged')" in shell
   and "p.delete('flagged')" in shell,
   "flagged filter serializes as status=flagged")
m = re.search(r"const TXN_URL_KEYS = \[[^\]]*\]", shell, re.S)
ok(bool(m), "URL query keys are an explicit allowlist")
if m:
    block = m.group(0)
    for bad in ("token", "passphrase", "totp", "secret", "password", "pan"):
        ok(bad not in block, f"URL allowlist excludes sensitive key: {bad}")
ok("txnLimit = parseInt($('f-limit').value, 10) || 50" in shell,
   "explicit page size still drives limit/offset")

# Detail navigation (§8).
ok('<button id="d-prev" class="table-btn" disabled>← Previous flagged</button>'
   in shell, "Previous flagged control exists, disabled by default")
ok('<button id="d-next" class="table-btn" disabled>Next flagged →</button>'
   in shell, "Next flagged control exists, disabled by default")
ok("'← Back to flagged transactions'" in shell
   and "'← Back to transactions'" in shell,
   "back label reflects the list it returns to")
ok("txnCtx.ids[txnCtx.idx - 1]" in shell
   and "txnCtx.ids[txnCtx.idx + 1]" in shell,
   "prev/next walk the authoritative context ids")
ok("idx < txnCtx.ids.length - 1" in shell and "!(idx > 0)" in shell,
   "edge controls disable instead of wrapping around")
ok("applyTxnQuery(new URLSearchParams(txnCtx.params))" in shell,
   "back restores the exact list state the detail opened from")
ok("ORDER BY scored_at DESC, event_id" in open(
    REPO / "backend" / "src" / "front_service" / "main.py",
    encoding="utf-8").read(),
   "backend keeps the authoritative sort — no client reordering")
ok("syncTxnContext" in shell
   and "async function syncTxnContext(eventId, d)" in shell,
   "detail fetches its navigation context through one bounded query")
ok("p.delete('event_id'); p.delete('fraud_id');" in shell,
   "navigation context is view-scoped — a stale search never narrows it")
ok("ctxQ.set('status', 'flagged')" in shell and "ctxQ.set('range', rng0)" in shell,
   "context back-target uses restore-format URL params (status/range)")

# Investigation continuity (§9): recorded facts only.
for eid in ("d-continuity", "d-cont-status", "d-cont-evidence",
            "d-cont-audit"):
    ok(f'id="{eid}"' in shell, f"continuity element present: {eid}")
ok("'Evidence available'" in shell and "'Audit trail available'" in shell,
   "continuity shows evidence/audit availability when recorded")
ok("'Evidence: N/A'" in shell and "'Audit trail: N/A'" in shell,
   "missing continuity facts render N/A — never invented")
ok(shell.count("contSet('d-cont-") == 3,
   "continuity renders exactly Status/Evidence/Audit — no invented fields")
ok("'Status: '" in shell and "(dqBlocked ? 'BLOCKED'" in shell,
   "status line comes from the recorded band/flag state")

# Empty / loading / error states (§13).
for text in ("No flagged transactions in this period.",
             "No transactions match these filters.",
             "Transaction not found",
             "No matching transactions",
             "No transactions yet"):
    ok(text in shell, f"empty state copy present: {text}")
ok('id="f-retry"' in shell
   and ">Transactions unavailable</b>" in shell,
   "API outage is named explicitly with a retry action")
ok("Unable to load transactions — try again." in shell
   and "Unable to load this transaction — try again." in shell,
   "familiar retry lines kept for the error pin")
ok("retry.style.display = 'none'" in shell
   and "retry2.style.display = 'inline-block'" in shell,
   "retry button hides on success, shows on failure")
ok("0 transactions" not in shell and "0 flagged" not in shell,
   "backend failure is never rendered as a fabricated zero count")

# Live monitor (§11): labelled click-through, behaviour otherwise pinned.
ok("Investigate" in shell and "→</span>" in shell,
   "flagged live rows offer the Investigate click-through label")
ok("reviewWord(f.review_state)" in shell and "clsF = 'warning'" in shell,
   "live feed rows keep their operator words")
ok("w0.requests_per_sec == null ? 'N/A'" in shell
   and "w0.errors == null ? 'N/A'" in shell,
   "live metrics still render N/A instead of fake zeros")

# §14: no duplicate loads on boot — the old one-shot second fetch is gone.
ok(shell.count("goFlagged();") == 1,
   "boot applies URL state once (exactly one goFlagged call site)")

# ── [11] regression pins: the four Phase-113 bugs ─────────────────────
blocks = re.findall(r"<script>(.*?)</script>", shell, re.S)
ok(len(blocks) == 2, f"exactly two script blocks ({len(blocks)})")
js = "\n".join(blocks)
bare = [m.group(0) for m in re.finditer(r"(?<!\[\.\.\.)\b\w+\.keys\(\)\.length\b", js)
        if not js[max(0, m.start() - 4):m.start()] == "[..."]
ok(not bare, f"URLSearchParams iterator bug stays fixed: {bare}")
ok("const wantOpen = txnOpenExact;" in shell
   and "txnOpenExact = false;" in shell
   and "if (wantOpen && lone && quick0" in shell,
   "auto-open is gated on an explicit search only")
i_apply = js.find("function applyTxnQuery")
i_apply_end = js.find("function pushTxnUrl")
ok("txnOpenExact = false;" in js[i_apply:i_apply_end],
   "restoring list state from a URL can never auto-open a detail")
i_pop = js.find("addEventListener('popstate'")
ok(i_pop > 0 and "applyTxnQuery(new URLSearchParams" in js[i_pop:i_pop + 900]
   and "push: false" in js[i_pop:i_pop + 900],
   "popstate re-applies query state WITHOUT pushing (no ping-pong)")
ok("if (!r.detail) applyTxnQuery(new URLSearchParams(window.location.search));"
   in js,
   "popstate only restores LIST entries — passing a detail entry keeps "
   "the context its prev/next and back label run on")
ok(js.count("window.location.pathname + window.location.search !== target")
   >= 2, "both push paths are idempotent (identical target never pushes)")
ok("if (livePaused) { setConn('PAUSED'); return; }" in js
   and js.count("if (livePaused) { setConn('PAUSED'); return; }") >= 2,
   "live-monitor pause race stays fixed (guard before and after await)")
ok("function stopLive" in js and "startDashboardTimer" in js,
   "polling control structures unchanged (no faster polling added)")

# ── [12] canonical constants + governance states ──────────────────────
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

# ── [13] production artifact identity AFTER ───────────────────────────
after = snapshot_artifacts()
after["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
changed = [k for k in WATCHED if before.get(k) != after.get(k)]
ok(not changed,
   f"production artifacts byte-identical across the suite ({changed})")

print()
print(f"PHASE 114 TRANSACTION HISTORY: {n_assert} assertions, "
      f"{len(failures)} failures")
if failures:
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PHASE 114 TRANSACTION HISTORY: ALL PASS")
