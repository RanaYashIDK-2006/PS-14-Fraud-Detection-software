"""Phase 114 — Review queue & operator productivity: test suite.

Sections
--------
 [1] production artifact identity BEFORE the suite runs
 [2] harness: isolated temp DB_DIR, seeded queue rows across all three
     review states, admin login
 [3] queue counts: list `review_counts` is authoritative, scoped like the
     chips (needs excludes low/unknown), independent of the ACTIVE review
     filter, never fabricated — a response without counts means the UI
     renders N/A
 [4] dashboard summary: needs_review + under_review + reviewed scalars
     from the backend (no frontend tally)
 [5] queue ordering: scored_at DESC exactly as the backend stores it,
     bounded pagination (limit clamp, offset bounds)
 [6] search + queue combinations: needs+period, flagged+needs, exact-ID
     precedence, unknown IDs stay empty, malformed filters -> 400
 [7] review transitions move rows between the three buckets and every
     count follows the server (start / replay / complete / reopen /
     invalid -> 409 with the record untouched)
 [8] concurrency: 6-way races serialize to exactly ONE change, replays
     are idempotent, a stale mutation gets 409 and never overwrites,
     exactly one audit event per accepted transition
 [9] security: unauthorized queue API -> 401, invalid session -> 401,
     cookie POST without X-Requested-With -> 403, SQL-injection-shaped
     filter params rejected, forged state/reviewer ignored, no secrets
     in API responses or the served shell
[10] frontend pins: queue banner + N/A, chip counts, dashboard KPI
     sub-line, list/detail queue navigation, URL context preservation,
     server-authoritative start/complete handling, advance-in-queue,
     live-feed terminology, bounded fetching
[11] regression pins: the four Phase-113 bugs stay fixed (URLSearchParams
     iterator, search-only auto-open, Back-button ping-pong, live pause
     race)
[12] canonical constants + qualified_datasets + RWV / promotion states
[13] production artifact identity AFTER (byte-identical)

Everything runs against an ISOLATED temp DB_DIR: no row written here can
reach the shared production DB-3/DB-4, and no production artifact is ever
opened for write. No network, no model fitting, no threshold/release work.

Run from backend/:
  ../.venv/Scripts/python.exe scripts/phase114_review_queue_test.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DB_TMP = tempfile.mkdtemp(prefix="ps14_p114rq_")
os.environ["DB_DIR"] = DB_TMP
ADMIN_USER = "admin"
ADMIN_PASS = "p114rq-suite-pass-9c2b"
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

# ── [2] harness: seed isolated DB-3 queue rows ────────────────────────
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
base = (datetime.utcnow() - timedelta(hours=5)).replace(microsecond=0)


def stamp(minutes: int) -> str:
    return (base + timedelta(minutes=minutes)).strftime(
        "%Y-%m-%d %H:%M:%S")


# (key, fraud_id, event_id, score, band, minutes-offset, initial transition)
# needs queue initially = A, H, B, C, G (five); D under; F reviewed;
# E is a low-band row that must NEVER count toward "needs".
SEEDS = [
    ("a", "F-P114RQ-A", "evt-p114rq-a-0001", 96, "high", 70, None),
    ("h", "F-P114RQ-H", "evt-p114rq-h-0002", 91, "high", 60, None),
    ("b", "F-P114RQ-B", "evt-p114rq-b-0003", 61, "medium", 50, None),
    ("c", "F-P114RQ-C", "evt-p114rq-c-0004", 77, "high", 40, None),
    ("d", "F-P114RQ-D", "evt-p114rq-d-0005", 88, "high", 30, "start"),
    ("e", "F-P114RQ-E", "evt-p114rq-e-0006", 4, "low", 20, None),
    ("f", "F-P114RQ-F", "evt-p114rq-f-0007", 84, "high", 10, "complete"),
    ("g", "F-P114RQ-G", "evt-p114rq-g-0008", 55, "medium", 0, None),
]
for key, fid, eid, score, band, off, _ in SEEDS:
    con.execute(
        "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
        "risk_band, reason_codes, model_version, ml_score, rule_score, "
        "degraded, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (f"sc-p114rq-{key}", fid, eid, score, band, "[]",
         MC.CANONICAL_MODEL_VERSION, 0.5, 4.0, 0, stamp(off)))
con.commit()
con.close()

A = "evt-p114rq-a-0001"
H = "evt-p114rq-h-0002"
B = "evt-p114rq-b-0003"
C = "evt-p114rq-c-0004"
D = "evt-p114rq-d-0005"
E = "evt-p114rq-e-0006"
F = "evt-p114rq-f-0007"
G = "evt-p114rq-g-0008"
NEEDS0 = [A, H, B, C, G]          # scored_at DESC
UNDER0 = [D]
REVIEWED0 = [F]

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
    params = {"limit": 50}
    params.update(kw)
    return c.get("/admin/api/transactions", headers=HDR, params=params)


def eids(r) -> list[str]:
    return [x["event_id"] for x in r.json()["rows"]]


def counts(r) -> dict:
    return r.json().get("review_counts")


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


# Establish the initial states: D under review, F fully reviewed.
r = rv_post(D, "start")
ok(r.status_code == 200 and r.json()["review_state"] == "UNDER_REVIEW",
   "seed: D walked to UNDER_REVIEW")
r = rv_post(F, "start")
r = rv_post(F, "complete")
ok(rv_get(F).json()["review_state"] == "REVIEWED",
   "seed: F walked to REVIEWED")

# ── [3] list review_counts: authoritative + scoped like the chips ─────
r = list_tx()
ok(r.status_code == 200, f"transactions list -> 200 ({r.status_code})")
rc = counts(r)
ok(rc == {"needs": 5, "under": 1, "reviewed": 1},
   f"review_counts: needs 5 / under 1 / reviewed 1, straight from the "
   f"server ({rc})")
ok(all(isinstance(v, int) for v in rc.values()),
   "counts are integers, never nulls dressed up as numbers")

r = list_tx(review="needs")
ok(sorted(eids(r)) == sorted(NEEDS0) and r.json()["total"] == 5,
   "review=needs rows = the five flagged UNREVIEWED rows")
ok(counts(r) == {"needs": 5, "under": 1, "reviewed": 1},
   "counts ignore the ACTIVE review filter (a reviewed view still shows "
   "what switching to needs would return)")

r = list_tx(review="reviewed")
ok(eids(r) == REVIEWED0, "review=reviewed rows = the completed row")
ok(counts(r) == {"needs": 5, "under": 1, "reviewed": 1},
   "chip counts never mirror the active chip (no fabricated 0)")

r = list_tx(flagged="true")
ok(counts(r) == {"needs": 5, "under": 1, "reviewed": 1},
   "counts under flagged scope stay the authoritative three")

r = list_tx(event_id=H)
ok(counts(r) == {"needs": 1, "under": 0, "reviewed": 0},
   "counts are scoped by the OTHER active filters (event_id=H -> "
   "needs 1, rest 0)")

# A response that returns early carries no counts -> the console must
# render N/A, never guess. decision=allow matches nothing in this DB.
r = c.get("/admin/api/transactions", headers=HDR,
          params={"limit": 50, "decision": "allow"})
ok(r.status_code == 200 and "review_counts" not in r.json()
   and r.json()["rows"] == [],
   "count-less response is honest (key absent, not a fabricated zero)")

# ── [4] dashboard summary scalars ─────────────────────────────────────
r = c.get("/admin/api/summary", headers=HDR)
ok(r.status_code == 200, f"summary -> 200 ({r.status_code})")
cnt = (r.json() or {}).get("counts") or {}
ok(cnt.get("needs_review") == 5,
   f"summary needs_review counts flagged UNREVIEWED rows ({cnt.get('needs_review')})")
ok(cnt.get("under_review") == 1,
   f"summary under_review scalar present and authoritative ({cnt.get('under_review')})")
ok(cnt.get("reviewed") == 1,
   f"summary reviewed scalar present and authoritative ({cnt.get('reviewed')})")
for key in ("transactions_24h", "flagged_24h", "blocked_24h"):
    ok(key in cnt, f"pre-existing summary key intact: {key}")

# ── [5] queue ordering + bounded pagination ───────────────────────────
r = list_tx(review="needs", limit=3, offset=0)
d = r.json()
ok(eids(r) == NEEDS0[:3] and d["total"] == 5 and d["has_more"] is True,
   "page 1 of the queue follows scored_at DESC (A, H, B)")
r = list_tx(review="needs", limit=3, offset=3)
d = r.json()
ok(eids(r) == NEEDS0[3:] and d["has_more"] is False,
   "page 2 continues the same authoritative order (C, G)")
r = list_tx(review="needs")
scored = [x["scored_at"] for x in r.json()["rows"]]
ok(scored == sorted(scored, reverse=True),
   "queue rows are strictly newest-first from the backend")
r = list_tx(review="needs", limit=1000)
ok(r.json()["limit"] == 200, "limit is clamped server-side (<= 200)")
r = list_tx(review="needs", limit=0)
ok(r.json()["limit"] == 1 and len(r.json()["rows"]) == 1,
   "limit=0 is clamped up, never unbounded")
r = list_tx(review="needs", offset=10001)
ok(r.status_code == 400, f"offset beyond bound -> 400 ({r.status_code})")
r = list_tx(review="needs", offset=-1)
ok(r.status_code == 400, f"negative offset -> 400 ({r.status_code})")

# ── [6] search + queue combinations ───────────────────────────────────
r = list_tx(review="needs", flagged="true")
ok(sorted(eids(r)) == sorted(NEEDS0),
   "Flagged + Needs Review -> the same five rows")
r = list_tx(review="needs", since=stamp(30))
ok(sorted(eids(r)) == sorted(NEEDS0[:4]),
   "Needs Review + date range narrows the queue rows")
ok(counts(r) == {"needs": 4, "under": 1, "reviewed": 0},
   f"counts follow the period scope too ({counts(r)})")

r = list_tx(event_id=H, review="needs")
ok(eids(r) == [H] and r.json()["total"] == 1,
   "exact ID + Needs Review: the ID takes precedence and the row is "
   "returned for direct navigation")
r = list_tx(event_id=F, review="needs")
ok(eids(r) == [] and r.json()["total"] == 0,
   "exact ID of a REVIEWED row under needs stays empty (no leak across "
   "chips)")
ok(counts(r) == {"needs": 0, "under": 0, "reviewed": 1},
   "counts stay event-scoped in that combination")
r = list_tx(event_id=E, review="needs")
ok(eids(r) == [] and r.json()["total"] == 0,
   "low-band rows are never part of Needs Review")

r = list_tx(event_id="evt-p114rq-none-999")
ok(r.status_code == 200 and eids(r) == [] and r.json()["total"] == 0,
   "unknown ID -> empty result, the UI shows Transaction not found")
r = list_tx(event_id="evt-p114rq-none-999", review="needs")
ok(r.status_code == 200 and eids(r) == [],
   "unknown ID + queue filter stays empty (queue never fabricates a hit)")

r = list_tx(review="whatever")
ok(r.status_code == 400, f"unknown review filter -> 400 ({r.status_code})")
r = list_tx(review="needs' OR '1'='1")
ok(r.status_code == 400,
   f"injection-shaped review filter -> 400 ({r.status_code})")
r = list_tx(review="needs", since=stamp(50), until=stamp(10))
ok(r.status_code == 400, f"since > until -> 400 ({r.status_code})")
r = list_tx(event_id="evt' OR 1=1--")
ok(r.status_code == 400,
   f"injection-shaped event_id -> 400 ({r.status_code})")
r = list_tx(fraud_id="F' OR '1'='1")
ok(r.status_code == 400,
   f"injection-shaped fraud_id -> 400 ({r.status_code})")

# ── [7] transitions move rows between the buckets ─────────────────────
r = rv_post(A, "start")
d = r.json()
ok(r.status_code == 200 and d["review_state"] == "UNDER_REVIEW"
   and d["changed"] is True and d["version"] == 1,
   "start: UNREVIEWED -> UNDER_REVIEW (v1), server response authoritative")
rc = counts(list_tx())
ok(rc == {"needs": 4, "under": 2, "reviewed": 1},
   f"after start: A left needs, joined under ({rc})")
ok(A not in eids(list_tx(review="needs")),
   "the started transaction is already out of the Needs Review queue")

r = rv_post(A, "start")   # replay
d = r.json()
ok(r.status_code == 200 and d["changed"] is False and d["version"] == 1,
   "replayed start is an idempotent success, counts do not drift")
ok(counts(list_tx()) == {"needs": 4, "under": 2, "reviewed": 1},
   "replay leaves every bucket untouched")

r = rv_post(A, "complete")
d = r.json()
ok(r.status_code == 200 and d["review_state"] == "REVIEWED"
   and d["changed"] is True and d["version"] == 2
   and d["reviewed_at"] is not None,
   "complete: UNDER_REVIEW -> REVIEWED with reviewed_at (v2)")
ok(counts(list_tx()) == {"needs": 4, "under": 1, "reviewed": 2},
   "after complete: under decremented, reviewed incremented")
r = c.get("/admin/api/summary", headers=HDR)
cnt = (r.json() or {}).get("counts") or {}
ok(cnt.get("needs_review") == 4 and cnt.get("under_review") == 1
   and cnt.get("reviewed") == 2,
   "dashboard summary scalars move with the same transitions "
   f"({cnt.get('needs_review')}/{cnt.get('under_review')}/"
   f"{cnt.get('reviewed')})")

# Invalid transitions: 409, record untouched, counts unmoved.
r = rv_post(B, "complete")
ok(r.status_code == 409
   and "invalid review transition" in r.json()["detail"],
   f"complete from UNREVIEWED -> 409 ({r.status_code})")
r = rv_post(B, "reopen")
ok(r.status_code == 409, f"reopen from UNREVIEWED -> 409 ({r.status_code})")
d = rv_get(B).json()
ok(d["review_state"] == "UNREVIEWED" and d["version"] == 0,
   "rejected transitions leave the record untouched")
ok(counts(list_tx()) == {"needs": 4, "under": 1, "reviewed": 2},
   "rejected transitions move no counts")

r = rv_post(A, "reopen")
d = r.json()
ok(r.status_code == 200 and d["review_state"] == "UNDER_REVIEW"
   and d["changed"] is True and d["version"] == 3
   and d["reviewed_at"] is None,
   "reopen: REVIEWED -> UNDER_REVIEW clears reviewed_at (v3)")
ok(counts(list_tx()) == {"needs": 4, "under": 2, "reviewed": 1},
   "reopen moves the row back to the under bucket")
r = rv_post(A, "complete")
d = r.json()
ok(r.status_code == 200 and d["version"] == 4
   and d["review_state"] == "REVIEWED",
   "complete after reopen settles the record at REVIEWED (v4)")
ok(counts(list_tx()) == {"needs": 4, "under": 1, "reviewed": 2},
   "counts settle: needs 4 / under 1 / reviewed 2")

# Body-supplied state / reviewer cannot steer the machine.
r = rv_post(B, "start",
            body={"review_state": "REVIEWED", "reviewer_id": "attacker",
                  "admin_identity": "attacker"})
d = r.json()
ok(r.status_code == 200 and d["review_state"] == "UNDER_REVIEW"
   and d["reviewer_id"] == ADMIN_USER,
   "forged state/reviewer in the body are ignored — server decides both")
ok(counts(list_tx()) == {"needs": 3, "under": 2, "reviewed": 2},
   "the honest transition still moved exactly one row between buckets")

# ── [8] concurrency ───────────────────────────────────────────────────
# B is now UNDER_REVIEW; use a fresh needs row (C) for the stale-page case.
d = rv_get(C).json()
ok(d["review_state"] == "UNREVIEWED" and d["version"] == 0,
   "operator B reads the pristine state before A acts")
r = rv_post(C, "complete")     # B's stale mutation: never started
ok(r.status_code == 409,
   f"stale-page mutation is rejected by optimistic concurrency "
   f"({r.status_code})")
d = rv_get(C).json()
ok(d["review_state"] == "UNREVIEWED" and d["version"] == 0,
   "stale page cannot silently overwrite the record")

before_cc = counts(list_tx())


def hammer(eid: str, action: str, n: int = 6) -> list[dict]:
    results: list[dict] = []
    lock = threading.Lock()
    errors: list[str] = []

    def worker():
        try:
            tc = TestClient(app)
            rr = tc.post(
                f"/admin/api/transactions/{eid}/review/{action}",
                json={}, headers={**HDR,
                                  "Content-Type": "application/json"})
            with lock:
                if rr.status_code != 200:
                    errors.append(f"{action}: HTTP {rr.status_code}")
                else:
                    results.append(rr.json())
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


res = hammer(H, "start")
ok(len(res) == 6 and sum(1 for x in res if x["changed"]) == 1,
   "6 concurrent starts -> exactly one change, rest are replays")
ok(rv_get(H).json()["review_state"] == "UNDER_REVIEW"
   and rv_get(H).json()["version"] == 1,
   "raced start settles at UNDER_REVIEW v1 (no lost update)")
ok(counts(list_tx()) == {"needs": before_cc["needs"] - 1,
                         "under": before_cc["under"] + 1,
                         "reviewed": before_cc["reviewed"]},
   "queue counts moved exactly once under the race")

res = hammer(H, "complete")
ok(len(res) == 6 and sum(1 for x in res if x["changed"]) == 1,
   "6 concurrent completes -> exactly one change")
d = rv_get(H).json()
ok(d["review_state"] == "REVIEWED" and d["version"] == 2,
   "raced complete settles at REVIEWED v2")
r = rv_post(H, "start")        # stale page from a colleague -> 409
ok(r.status_code == 409,
   f"start from REVIEWED -> 409, stale page rejected ({r.status_code})")
ok(rv_get(H).json()["version"] == 2,
   "the rejected mutation did not touch the record")
r = rv_post(H, "complete")     # replay
ok(r.status_code == 200 and r.json()["changed"] is False
   and r.json()["version"] == 2,
   "replayed complete stays idempotent (no version drift)")

starts = [json.loads(x["payload_summary"])
          for x in review_audit_rows("admin_review_started")]
comps = [json.loads(x["payload_summary"])
         for x in review_audit_rows("admin_review_completed")]
ok(sum(1 for p in starts if p["event_id"] == H) == 1
   and sum(1 for p in comps if p["event_id"] == H) == 1,
   "races + replays produced exactly ONE audit event per transition")
ok(counts(list_tx()) == {"needs": before_cc["needs"] - 1,
                         "under": before_cc["under"],
                         "reviewed": before_cc["reviewed"] + 1},
   "final counts: exactly the raced row moved buckets")

# ── [9] security ──────────────────────────────────────────────────────
fresh = TestClient(app)
for method, path in (("get", "/admin/api/transactions?review=needs"),
                     ("get", "/admin/api/summary"),
                     ("get", f"/admin/api/transactions/{A}/review")):
    rr = fresh.get(path) if method == "get" else fresh.post(path, json={})
    ok(rr.status_code == 401,
       f"unauthenticated {path.split('?')[0].split('/admin/api')[1]} "
       f"-> 401 ({rr.status_code})")
for method, path in (("post", f"/admin/api/transactions/{C}/review/start"),
                     ("post", f"/admin/api/transactions/{C}/notes")):
    rr = fresh.post(path, json={"note": "x"})
    ok(rr.status_code == 401,
       f"unauthenticated {method.upper()} {path.split('/admin/api')[1]} "
       f"-> 401 ({rr.status_code})")
rr = fresh.get(f"/admin/api/transactions?review=needs",
               headers={"Authorization": "Bearer not-a-real-token"})
ok(rr.status_code == 401, f"invalid bearer token -> 401 ({rr.status_code})")
rr = fresh.get("/admin/api/transactions?review=needs",
               cookies={"admin_session": "garbage-session"})
ok(rr.status_code == 401, f"invalid session cookie -> 401 ({rr.status_code})")

# Cookie path needs X-Requested-With (CSRF) for state-changing POSTs.
nohead = TestClient(app)
nohead.cookies.update(c.cookies)
rr = nohead.post(f"/admin/api/transactions/{D}/review/start")
ok(rr.status_code == 403 and "X-Requested-With" in rr.json()["detail"],
   f"cookie POST without X-Requested-With -> 403 ({rr.status_code})")
rr = nohead.post(f"/admin/api/transactions/{D}/review/start",
                 headers={"X-Requested-With": "XMLHttpRequest"})
ok(rr.status_code == 200 and rr.json()["changed"] is False,
   f"cookie POST with the header succeeds as a harmless replay "
   f"({rr.status_code})")

# No secrets in queue/API responses.
for path in ("/admin/api/summary", "/admin/api/transactions?review=needs",
             f"/admin/api/transactions/{A}",
             f"/admin/api/transactions/{A}/review"):
    txt = c.get(path, headers=HDR).text
    ok("passphrase" not in txt and "blob_key" not in txt
       and ADMIN_PASS not in txt and "TOTP" not in txt
       and "otpauth" not in txt,
       f"no secrets in {path.split('?')[0]}")
    ok(not re.search(r"\b4[0-9]{15}\b|\b3[0-9]{14}\b", txt),
       f"no PAN-shaped values in {path.split('?')[0]}")
    ok("token" not in txt.lower() or TOKEN not in txt,
       f"session token never echoed in {path.split('?')[0]}")

# Served shell: no secrets, no credentials in URLs.
shell = c.get("/admin").text
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

# Queue navigation is read-only: no new audit event types invented.
known = {x["event_type"] for x in review_audit_rows()}
ok(known <= {"score_generated", "admin_tx_search", "admin_tx_view",
             "admin_login_success", "admin_review_started",
             "admin_review_completed", "admin_review_reopened",
             "admin_review_note_added"},
   f"queue navigation added no new audit event type ({sorted(known)})")

# ── [10] frontend pins ────────────────────────────────────────────────
ok('id="f-queue"' in shell and 'id="f-queue-title">NEEDS REVIEW' in shell,
   "queue banner card with NEEDS REVIEW title present")
ok('id="f-queue-sub"' in shell and ">N/A</div>" in shell,
   "queue sub-line defaults to N/A before the server answers")
ok("transactions currently awaiting review" in shell,
   "queue sub-line uses the authoritative awaiting-review wording")
ok("function renderQueueHeader(total)" in shell
   and "const showQ = ($('f-review') && $('f-review').value) === 'needs';"
   in shell
   and "qBox.style.display = showQ ? '' : 'none';" in shell,
   "queue banner shows ONLY for the Needs Review chip")
ok("renderQueueHeader(null);" in shell and "renderQueueHeader(txnTotal);"
   in shell,
   "banner is N/A while loading, then the response's own total")
ok("total != null" in shell
   and "'N/A'" in shell.split("function renderQueueHeader")[1][:600],
   "unavailable count renders N/A, never a fabricated 0")

rchips = re.findall(r'data-rchip="([a-z]+)"', shell)
ok(rchips == ["all", "needs", "under", "reviewed"],
   f"review chips stay All/Needs/Under/Reviewed ({rchips})")
for rid in ("rc-needs", "rc-under", "rc-reviewed"):
    ok(f'id="{rid}"' in shell, f"chip count span present: {rid}")
ok("function renderReviewCounts(rc)" in shell
   and "rc && rc[k] != null" in shell
   and "renderReviewCounts(d.review_counts);" in shell,
   "chip counts render server values or N/A (never guessed)")
ok(re.findall(r'data-chip="([a-z]+)"', shell)
   == ["all", "flagged", "approved", "blocked"],
   "original quick-filter chips untouched")

ok('id="dash-review-sub"' in shell
   and "'Under review: ' + fmt(c.under_review)" in shell
   and "' · Reviewed: ' + fmt(c.reviewed)" in shell,
   "dashboard KPI sub-line: Under review / Reviewed from summary counts")
ok('id="dash-needs-review"' in shell and "goNeedsReview" in shell,
   "dashboard Needs Review KPI still wired to the queue")

ok("function renderQueuePos()" in shell
   and "Page ${txnPage + 1} · Transaction ${idx + 1} of ${txnCtx.ids.length}"
   in shell,
   "list-side position: Page N · Transaction i of len")
ok("function makeTxnCtx(ids, d, p)" in shell
   and "query: p.toString()" in shell
   and "offset: (d && d.offset != null) ? d.offset : 0" in shell
   and "total: (d && d.total != null) ? d.total : null" in shell,
   "one context constructor carries query + server offset/total "
   "(never client-computed)")
ok("makeTxnCtx([], d, p)" in shell
   and "makeTxnCtx(rows.map(x => x.event_id), d, p)" in shell,
   "every list-load path builds the context from the fresh response")

ok('id="d-queue-name"' in shell and 'id="d-queue-pos"' in shell
   and "← Previous flagged" in shell and "Next flagged →" in shell,
   "detail queue bar: queue pill + prev/next + position")
ok("const REVIEW_QUEUE_LABEL = { needs: 'Needs Review'" in shell,
   "queue labels defined for needs/under/reviewed")
ok("function txnQueueLabel()" in shell
   and "if (rv && REVIEW_QUEUE_LABEL[rv]) return REVIEW_QUEUE_LABEL[rv];"
   in shell,
   "queue label resolves from the active chip")
i_lq = shell.find("function txnQueueLabel")
ok(i_lq > 0 and re.search(
        r"function txnQueueLabel\(\) \{[^}]*return null;",
        shell[i_lq:], re.S),
   "lone exact-ID search yields NO queue label (return null path)",)
ok("return null;" in shell[i_lq:shell.find("function updateTxnNav")],
   "txnQueueLabel honestly returns null without a queue context")

ok("function updateTxnNav()" in shell
   and "prev.disabled = !(idx > 0);" in shell
   and "next.disabled = !(txnCtx && idx >= 0 && idx < txnCtx.ids.length - 1);"
   in shell
   and "nameEl.style.display = label ? '' : 'none';" in shell
   and "(txnCtx.offset + idx + 1) + ' of ' + txnCtx.total" in shell,
   "nav disables at the edges; pill hidden and position honest without "
   "a queue; position from SERVER offset + total")

# URL context preservation (§5).
ok("const TXN_URL_KEYS" in shell and "'review'" in
   shell[shell.find("const TXN_URL_KEYS"):shell.find("const TXN_URL_KEYS")
         + 400],
   "review scope is part of the shareable URL allowlist")
ok("if (page > 1) p.set('page', String(page));" in shell
   and "const page = sp.get('page');" in shell
   and "const rev = sp.get('review');" in shell,
   "?review=needs&page=2 round-trips through the URL")
ok("if (txnCtx && txnCtx.params) applyTxnQuery(new URLSearchParams(txnCtx.params));"
   in shell,
   "Back from a detail restores the exact queue/list state it came from")

# Server-authoritative workflow (§9/§10/§16).
ok("const reviewPost = async (path, body) => {" in shell
   and "fail.status = r.status;" in shell,
   "POST helper carries the HTTP status so 409s are distinguishable")
ok("if (d.changed) await afterReviewAction(action);" in shell,
   "queue refresh happens ONLY on a successful server response")
ok("if (e2 && e2.status === 409) await refreshReviewSection();" in shell,
   "409 -> re-fetch the server state (no silent overwrite)")
ok("async function refreshReviewSection()" in shell
   and "'/review'" in shell
   and "if (r.ok) renderReviewSection(d);" in shell,
   "stale detail re-renders the authoritative review record")
ok("async function afterReviewAction(action)" in shell
   and "if (!txnCtx || !txnCtx.query) return;" in shell
   and "const fresh = await fetchTxnCtx(txnCtx.query);" in shell
   and "if (!fresh) return;" in shell,
   "after an action the queue is re-fetched from the backend FIRST "
   "(never a stale client array)")
ok("const wasMember = fresh.ids.indexOf(txnDetailId) !== -1;" in shell
   and "if (action === 'complete' && !wasMember)" in shell,
   "complete-in-queue checks membership against the FRESH queue")
ok("Math.min(anchor, fresh.ids.length - 1)" in shell
   and "activateTab('transactions', { detail: nextId });" in shell,
   "completion advances to the next valid transaction — never random")
ok("if (!fresh.ids.length)" in shell
   and "// refreshed queue, empty state" in shell,
   "empty refreshed queue shows the empty state, not a random pick")
ok("renderQueueHeader(fresh.total);" in shell
   and "renderReviewCounts(fresh.review_counts);" in shell,
   "queue banner + chip counts refresh from the fresh response")
ok("renderReviewSection(d);" in shell,
   "review section re-renders from the server payload after each action")
ok("Math.random" not in shell, "no random navigation anywhere")

# Live feed terminology (§13).
ok("· ${esc(reviewWord(f.review_state))}" in shell,
   "live flagged row shows FLAGGED · <review state>")
ok("reviewWord(f.review_state)" in shell and "Investigate" in shell,
   "live click-through label kept, opening never starts a review")

# Terminology + notes discipline (§11/§12).
for label in ("Needs Review", "Under Review", "Reviewed"):
    ok(label in shell, f"terminology label present: {label}")
ok("maxlength=\"2000\"" in shell
   and "admin_review_note_added" in shell,
   "notes stay Phase-113: bounded, same audit event")
ok("confirm_alert" not in shell and "Confirm fraud" not in shell
   and "Mark as fraud" not in shell,
   "no fraud-confirmation affordance")
ok("bulk" not in shell.lower(), "no bulk mutation affordance")

# Bounded fetching (§15).
ok("min(int(limit), 200)" in
   (REPO / "backend" / "src" / "front_service" / "main.py").read_text(
       encoding="utf-8").replace("\r\n", "\n"),
   "backend list stays clamped (limit <= 200)")
ok("p.set('limit', '200'); p.set('offset', '0');" in shell,
   "deep-link context requery is one bounded page")
ok("(server-capped)" in shell, "UI labels the server-capped page size")

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
   "popstate only restores LIST entries — no Back-button ping-pong")
ok(js.count("window.location.pathname + window.location.search !== target")
   >= 2, "both push paths are idempotent (identical target never pushes)")
ok("if (livePaused) { setConn('PAUSED'); return; }" in js
   and js.count("if (livePaused) { setConn('PAUSED'); return; }") >= 2,
   "live-monitor pause race stays fixed (guard before and after await)")

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
print(f"PHASE 114 REVIEW QUEUE: {n_assert} assertions, "
      f"{len(failures)} failures")
if failures:
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PHASE 114 REVIEW QUEUE: ALL PASS")
