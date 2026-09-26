"""Phase 113 — Flagged transaction review workflow: test suite.

Sections
--------
 [1] production artifact identity BEFORE the suite runs
 [2] harness: isolated temp DB_DIR, seeded flagged rows, admin login
 [3] review state: default, persistence, valid transitions, idempotent
     replay, reopen, invalid transitions -> 409, state injection ignored
 [4] investigation notes: append-only, UTF-8, bounded, server-validated,
     session-authoritative author (forged identity ignored)
 [5] audit: immutable admin_review_* events with previous/new state,
     admin identity and timestamp — replay adds no duplicate event
 [6] integration: transactions list review_state + review filters,
     summary Needs Review KPI count, detail review block, live feed state
 [7] security: unauthorized sweep 401, invalid session 401, malformed
     event_id 400, unknown event 404, cookie POST without X-Requested-With
     403, no secrets in responses or the served shell
 [8] immutability + concurrency: risk_scores untouched (and the front
     service holds no write path to it); concurrent transitions serialize
     to exactly one change with no silent overwrite
 [9] frontend pins: Needs Review KPI, review chips, Review column,
     Investigation section, action buttons, notes timeline, loading /
     empty / error states, responsive stacking, audit human labels
[10] canonical constants + RWV / promotion states unchanged
[11] production artifact identity AFTER (byte-identical)

Everything runs against an ISOLATED temp DB_DIR: no row written here can
reach the shared production DB-3/DB-4, and no production artifact is ever
opened for write. No network, no model fitting, no threshold/release work.

Run from backend/:
  ../.venv/Scripts/python.exe scripts/phase113_review_workflow_test.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DB_TMP = tempfile.mkdtemp(prefix="ps14_p113rw_")
os.environ["DB_DIR"] = DB_TMP
ADMIN_USER = "admin"
ADMIN_PASS = "p113rw-suite-pass-7e4a"
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

# ── [2] harness: seed isolated DB-3 + DB-4 ────────────────────────────
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
now = (datetime.utcnow() - timedelta(minutes=5)).strftime(
    "%Y-%m-%d %H:%M:%S.%f")
EVT_FLAG = "evt-p113rw-flag-001"     # high  -> goes through the full cycle
EVT_NONE = "evt-p113rw-new-002"      # medium -> stays UNREVIEWED (needs review)
EVT_LOW = "evt-p113rw-low-003"       # low   -> never part of "needs review"
SEEDS = [
    ("sc-p113rw-001", "F-P113RW-FLG", EVT_FLAG, 97, "high",
     ["VELOCITY_HIGH", "AMOUNT_ANOMALY"]),
    ("sc-p113rw-002", "F-P113RW-NEW", EVT_NONE, 55, "medium", []),
    ("sc-p113rw-003", "F-P113RW-LOW", EVT_LOW, 7, "low", []),
]
for sid, fid, eid, score, band, codes in SEEDS:
    con.execute(
        "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
        "risk_band, reason_codes, model_version, ml_score, rule_score, "
        "degraded, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (sid, fid, eid, score, band, json.dumps(codes),
         MC.CANONICAL_MODEL_VERSION, 0.91, 8.4, 0, now))
con.commit()
seed_row = con.execute(
    "SELECT * FROM risk_scores WHERE event_id = ?", (EVT_FLAG,)).fetchone()
con.close()

# One recorded score_generated decision for the flagged event, so the
# detail view has a real DB-4 decision payload to reconcile against.
FLAG_PAYLOAD = {
    "event_id": EVT_FLAG,
    "fraud_id": "F-P113RW-FLG",
    "risk_score": 97,
    "risk_band": "high",
    "reason_codes": ["VELOCITY_HIGH", "AMOUNT_ANOMALY"],
    "model_version": MC.CANONICAL_MODEL_VERSION,
    "ml_score": 0.91,
    "rule_score": 8.4,
    "degraded": False,
    "decision": "step_up",
    "feature_version": MC.CANONICAL_FEATURE_VERSION,
    "runtime_state": "READY",
    "runtime_release_id": MC.CANONICAL_LEGACY_RELEASE_ID,
}
append_audit_event("F-P113RW-FLG", "score_generated", FLAG_PAYLOAD)
flush_audit_queue(timeout=5)

c = TestClient(app)
r = c.post("/admin/login",
           json={"username": ADMIN_USER, "passphrase": ADMIN_PASS})
ok(r.status_code == 200 and "token" in r.json(),
   f"login -> 200 + token ({r.status_code})")
TOKEN = r.json()["token"]
HDR = {"Authorization": f"Bearer {TOKEN}"}
CSRF = {"X-Requested-With": "XMLHttpRequest"}


def rv_get(eid: str, client: TestClient | None = None):
    return (client or c).get(
        f"/admin/api/transactions/{eid}/review", headers=HDR)


def rv_post(eid: str, action: str, body: dict | None = None,
            client: TestClient | None = None, headers: dict | None = None):
    h = dict(HDR if headers is None else headers)
    h["Content-Type"] = "application/json"
    return (client or c).post(
        f"/admin/api/transactions/{eid}/review/{action}",
        json={} if body is None else body, headers=h)


def note_post(eid: str, body: dict, client: TestClient | None = None,
              headers: dict | None = None):
    h = dict(HDR if headers is None else headers)
    h["Content-Type"] = "application/json"
    return (client or c).post(f"/admin/api/transactions/{eid}/notes",
                              json=body, headers=h)


def review_audit_rows(event_type: str | None = None) -> list[dict]:
    flush_audit_queue(timeout=5)
    conn = sqlite3.connect(str(fm._DBS["audit"]))
    conn.row_factory = sqlite3.Row
    try:
        sql = ("SELECT seq, fraud_id, event_type, payload_summary, "
               "prev_hash, entry_hash FROM audit_events")
        params: tuple = ()
        if event_type:
            sql += " WHERE event_type = ?"
            params = (event_type,)
        rows = [dict(x) for x in conn.execute(
            sql + " ORDER BY seq", params).fetchall()]
        return rows
    finally:
        conn.close()


# ── [3] default state, transitions, replay, reopen, rejections ────────
r = rv_get(EVT_FLAG)
ok(r.status_code == 200, f"GET review -> 200 ({r.status_code})")
d = r.json()
ok(d["review_state"] == "UNREVIEWED" and d["version"] == 0
   and d["notes"] == [] and d["reviewer_id"] is None,
   "absent record is honest UNREVIEWED defaults (version 0, no notes)")

r = rv_post(EVT_FLAG, "start")
ok(r.status_code == 200, f"start -> 200 ({r.status_code})")
d = r.json()
ok(d["review_state"] == "UNDER_REVIEW" and d["changed"] is True
   and d["version"] == 1, f"UNREVIEWED -> UNDER_REVIEW (v1): {d}")
ok(d["reviewer_id"] == ADMIN_USER,
   f"reviewer comes from the session, not the client ({d['reviewer_id']!r})")
ok(d["audit_logged"] is True, "start produced an audit event")

r = rv_get(EVT_FLAG)
ok(r.json()["review_state"] == "UNDER_REVIEW",
   "review state persists across requests")

r = rv_post(EVT_FLAG, "start")   # replay
d = r.json()
ok(r.status_code == 200 and d["changed"] is False and d["version"] == 1,
   "replayed transition is an idempotent success, no version bump")

r = rv_post(EVT_FLAG, "complete")
d = r.json()
ok(r.status_code == 200 and d["review_state"] == "REVIEWED"
   and d["changed"] is True and d["version"] == 2
   and d["reviewed_at"] is not None,
   "UNDER_REVIEW -> REVIEWED with reviewed_at (v2)")

r = rv_post(EVT_FLAG, "complete")   # replay
d = r.json()
ok(r.status_code == 200 and d["changed"] is False and d["version"] == 2,
   "replayed complete stays idempotent")

r = rv_post(EVT_FLAG, "reopen")
d = r.json()
ok(r.status_code == 200 and d["review_state"] == "UNDER_REVIEW"
   and d["changed"] is True and d["version"] == 3
   and d["reviewed_at"] is None,
   "REVIEWED -> UNDER_REVIEW (reopen) clears reviewed_at (v3)")

# Invalid transitions -> honest 409, no state change, no audit event.
r = rv_post(EVT_NONE, "complete")
ok(r.status_code == 409 and "invalid review transition" in r.json()["detail"],
   f"complete from UNREVIEWED -> 409 ({r.status_code})")
r = rv_post(EVT_NONE, "reopen")
ok(r.status_code == 409, f"reopen from UNREVIEWED -> 409 ({r.status_code})")
r = rv_get(EVT_NONE)
ok(r.json()["review_state"] == "UNREVIEWED" and r.json()["version"] == 0,
   "rejected transitions leave the record untouched")
r = rv_post(EVT_FLAG, "start")     # UNDER_REVIEW -> start = replay, not 409
ok(r.status_code == 200 and r.json()["changed"] is False,
   "start while already under review is a safe replay")

# State injection: the request body cannot choose state or reviewer.
r = rv_post(EVT_LOW, "start",
            body={"review_state": "REVIEWED", "reviewer_id": "attacker",
                  "admin_identity": "attacker"})
d = r.json()
ok(r.status_code == 200 and d["review_state"] == "UNDER_REVIEW"
   and d["reviewer_id"] == ADMIN_USER,
   "body-supplied state/reviewer are ignored — server decides both")
rv_post(EVT_LOW, "complete")       # leave EVT_LOW REVIEWED for later checks

# ── [4] investigation notes ───────────────────────────────────────────
r = note_post(EVT_FLAG, {"note": "Velocity spike shares one device with "
                                 "two earlier flagged events."})
d = r.json()
ok(r.status_code == 200 and d.get("note_added") is True
   and d["audit_logged"] is True, f"note accepted ({r.status_code})")
ok(len(d["notes"]) == 1 and d["notes_total"] == 1
   and d["notes"][0]["reviewer_id"] == ADMIN_USER
   and d["notes"][0]["created_at"].endswith("+00:00"),
   "note recorded with session author and UTC timestamp")

r = note_post(EVT_FLAG, {"note": "观察：同一设备 + Ω≈42 — reviewer ok ✓"})
d = r.json()
ok(r.status_code == 200 and d["notes_total"] == 2
   and "观察" in d["notes"][1]["note"],
   "UTF-8 note round-trips intact")

for label, body in (
        ("empty", {"note": ""}),
        ("whitespace-only", {"note": "   \n  "}),
        ("oversized", {"note": "x" * 2001}),
        ("control chars", {"note": "bad\x07note"}),
):
    r = note_post(EVT_FLAG, body)
    ok(r.status_code == 400, f"{label} note -> 400 ({r.status_code})")

r = note_post(EVT_FLAG,
              {"note": "forged attempt", "reviewer_id": "evil-actor",
               "admin_identity": "evil-actor", "sub": "evil-actor"})
d = r.json()
ok(r.status_code == 200
   and all(n["reviewer_id"] == ADMIN_USER for n in d["notes"]),
   "forged reviewer identity in the body is ignored")
ok(d["notes_total"] == 3, "note appended, never replaced (append-only)")

r = rv_get(EVT_FLAG)
d = r.json()
ok([n["note"] for n in d["notes"]][:2]
   == ["Velocity spike shares one device with two earlier flagged events.",
       "观察：同一设备 + Ω≈42 — reviewer ok ✓"],
   "notes timeline is ordered and immutable")

# ── [5] audit events ──────────────────────────────────────────────────
rows = review_audit_rows()
by_type: dict[str, list[dict]] = {}
for x in rows:
    by_type.setdefault(x["event_type"], []).append(x)
ok(len(by_type.get("admin_review_started", [])) == 2,
   "one admin_review_started per accepted start, replays add none "
   f"({len(by_type.get('admin_review_started', []))})")
ok(len(by_type.get("admin_review_completed", [])) == 2,
   "one admin_review_completed per accepted complete")
ok(len(by_type.get("admin_review_reopened", [])) == 1,
   "one admin_review_reopened for the reopen")
ok(len(by_type.get("admin_review_note_added", [])) == 3,
   "one admin_review_note_added per accepted note")

start_payloads = [json.loads(x["payload_summary"])
                  for x in by_type["admin_review_started"]]
flag_start = next(p for p in start_payloads if p["event_id"] == EVT_FLAG)
ok(flag_start.get("previous_review_state") == "UNREVIEWED"
   and flag_start.get("new_review_state") == "UNDER_REVIEW",
   "audit records previous and new review state")
ok(flag_start.get("admin_identity") == ADMIN_USER
   and isinstance(flag_start.get("timestamp"), str),
   "audit records admin identity and timestamp")
secret_hits = [k for k in ("passphrase", "totp_secret", "token",
                           "recovery_codes", "blob_key")
               if any(k in json.loads(x["payload_summary"])
                      for x in rows if x["event_type"].startswith(
                          "admin_review"))]
ok(not secret_hits, f"review audit payloads carry no secrets ({secret_hits})")
ok(all(x["prev_hash"] and x["entry_hash"] for x in rows
       if x["event_type"].startswith("admin_review")),
   "review audit events are hash-chained like every other entry")

# ── [6] list / filters / summary / detail / live integration ──────────
EVT_DONE = "evt-p113rw-done-004"
con = sqlite3.connect(str(fm._DBS["risk"]))
con.execute(
    "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
    "risk_band, reason_codes, model_version, ml_score, rule_score, "
    "degraded, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    ("sc-p113rw-004", "F-P113RW-DNE", EVT_DONE, 88, "high", "[]",
     MC.CANONICAL_MODEL_VERSION, 0.5, 4.2, 0, now))
con.commit()
con.close()
rv_post(EVT_DONE, "start")
rv_post(EVT_DONE, "complete")
r = rv_get(EVT_DONE)
ok(r.json()["review_state"] == "REVIEWED",
   "second event walks start -> complete for filter coverage")
r = rv_post(EVT_DONE, "start")     # REVIEWED -> start is invalid
ok(r.status_code == 409,
   f"start from REVIEWED -> 409 ({r.status_code})")

r = c.get("/admin/api/transactions", headers=HDR, params={"limit": 50})
ok(r.status_code == 200, f"transactions list -> 200 ({r.status_code})")
rowsd = {x["event_id"]: x for x in r.json()["rows"]}
ok(all("review_state" in rowsd[e]
       for e in (EVT_FLAG, EVT_NONE, EVT_LOW, EVT_DONE)),
   "every list row carries a review_state field")
ok(rowsd[EVT_FLAG]["review_state"] == "UNDER_REVIEW"
   and rowsd[EVT_NONE]["review_state"] == "UNREVIEWED"
   and rowsd[EVT_DONE]["review_state"] == "REVIEWED",
   "list review_state reflects the stored record")


def filter_eids(review: str) -> set[str]:
    r = c.get("/admin/api/transactions", headers=HDR,
              params={"limit": 50, "review": review})
    assert r.status_code == 200, (review, r.status_code, r.text)
    return {x["event_id"] for x in r.json()["rows"]}


ok(filter_eids("needs") == {EVT_NONE},
   "review=needs -> flagged, never-started rows only")
ok(filter_eids("under") == {EVT_FLAG},
   "review=under -> rows an operator is working")
ok(filter_eids("reviewed") == {EVT_LOW, EVT_DONE},
   "review=reviewed -> completed rows")
ok(filter_eids("all") == {EVT_FLAG, EVT_NONE, EVT_LOW, EVT_DONE},
   "review=all -> every row, no constraint")
r = c.get("/admin/api/transactions", headers=HDR,
          params={"review": "whatever"})
ok(r.status_code == 400, f"unknown review filter -> 400 ({r.status_code})")

r = c.get("/admin/api/summary", headers=HDR)
ok(r.status_code == 200, f"summary -> 200 ({r.status_code})")
cnt = (r.json() or {}).get("counts") or {}
ok(cnt.get("needs_review") == 1,
   f"Needs Review KPI counts only flagged UNREVIEWED rows ({cnt.get('needs_review')})")

r = c.get(f"/admin/api/transactions/{EVT_FLAG}", headers=HDR)
ok(r.status_code == 200, f"detail -> 200 ({r.status_code})")
d = r.json()
ok(d["review"]["review_state"] == "UNDER_REVIEW"
   and d["review"]["notes_total"] == 3,
   "detail payload embeds the review record + notes")
ok(d["score"]["risk_score"] == 97 and d["decision"] == "step_up",
   "review workflow left score and decision untouched")

r = c.get("/admin/api/live", headers=HDR)
ok(r.status_code == 200, f"live -> 200 ({r.status_code})")
feed = {x["event_id"]: x.get("review_state")
        for x in (r.json().get("feed") or [])}
ok(feed.get(EVT_FLAG) == "UNDER_REVIEW" and feed.get(EVT_NONE) == "UNREVIEWED",
   "live feed exposes review state for display only")

# ── [7] security ──────────────────────────────────────────────────────
fresh = TestClient(app)
ENDPOINTS = [
    ("get", "/admin/api/transactions/%s/review" % EVT_FLAG),
    ("post", "/admin/api/transactions/%s/review/start" % EVT_FLAG),
    ("post", "/admin/api/transactions/%s/review/complete" % EVT_FLAG),
    ("post", "/admin/api/transactions/%s/review/reopen" % EVT_FLAG),
    ("post", "/admin/api/transactions/%s/notes" % EVT_FLAG),
]
for method, path in ENDPOINTS:
    r = (fresh.get(path) if method == "get"
         else fresh.post(path, json={}))
    ok(r.status_code == 401,
       f"unauthenticated {method.upper()} {path.split('/admin/api')[1]} "
       f"-> 401 ({r.status_code})")

r = fresh.get(ENDPOINTS[0][1],
              headers={"Authorization": "Bearer not-a-real-token"})
ok(r.status_code == 401, f"invalid bearer token -> 401 ({r.status_code})")
r = fresh.get(ENDPOINTS[0][1], cookies={"admin_session": "garbage-session"})
ok(r.status_code == 401, f"invalid session cookie -> 401 ({r.status_code})")

# Cookie path: state-changing POST must carry X-Requested-With (CSRF).
nohead = TestClient(app)
nohead.cookies.update(c.cookies)
r = nohead.post(f"/admin/api/transactions/{EVT_FLAG}/review/start")
ok(r.status_code == 403 and "X-Requested-With" in r.json()["detail"],
   f"cookie POST without X-Requested-With -> 403 ({r.status_code})")
r = nohead.post(f"/admin/api/transactions/{EVT_FLAG}/review/start",
                headers=CSRF)
ok(r.status_code == 200 and r.json()["changed"] is False,
   f"cookie POST with the header succeeds as a replay ({r.status_code})")

# Malformed / unknown event ids.
for bad in ("evt!bad", "e" * 129, "evt+bad"):
    r = c.get(f"/admin/api/transactions/{bad}/review", headers=HDR)
    ok(r.status_code == 400,
       f"malformed event_id {bad[:16]!r}... -> 400 ({r.status_code})")
r = c.get("/admin/api/transactions/evt-p113rw-none-999/review", headers=HDR)
ok(r.status_code == 404, f"unknown event -> 404 ({r.status_code})")

# No secrets in API responses.
for path in (f"/admin/api/transactions/{EVT_FLAG}/review",
             f"/admin/api/transactions/{EVT_FLAG}",
             "/admin/api/summary", "/admin/api/transactions?limit=5"):
    txt = c.get(path, headers=HDR).text
    ok("passphrase" not in txt and "blob_key" not in txt
       and ADMIN_PASS not in txt and "TOTP" not in txt,
       f"no secrets in {path.split('?')[0]}")
    ok(not re.search(r"\b4[0-9]{15}\b|\b3[0-9]{14}\b", txt),
       f"no PAN-shaped values in {path.split('?')[0]}")

# Served shell: no secrets, and no investigation write outside the backend.
shell = c.get("/admin").text
SECRET_PATS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"otpauth://totp/[^\"'\\s]*\\?secret=[A-Z2-7]{16,}"),
     "literal otpauth secret"),
    (re.compile(r"(?i)\\b(?:totp_secret|TOTP_SECRET)\\b\\s*[:=]\\s*[\"']"
                r"[A-Z2-7]{16,}[\"']"), "hard-coded TOTP secret"),
    (re.compile(r"(?i)[\"']Bearer\\s+[A-Za-z0-9._\\-]{40,}[\"']"),
     "hard-coded bearer token"),
    (re.compile(r"[?&](?:access_token|api_key|apikey|secret|totp)=[A-Za-z0-9]"),
     "credential in a URL query string"),
]
hits = [label for pat, label in SECRET_PATS if pat.search(shell)]
ok(not hits, f"no secrets in the served shell ({hits})")
ok(ADMIN_PASS not in shell, "admin passphrase never appears in the shell")

# ── [8] immutability + concurrency ────────────────────────────────────
con = sqlite3.connect(str(fm._DBS["risk"]))
after_row = con.execute(
    "SELECT * FROM risk_scores WHERE event_id = ?",
    (EVT_FLAG,)).fetchone()
con.close()
ok(tuple(seed_row) == tuple(after_row),
   "risk_scores row is byte-identical across the whole workflow")

src_main = (REPO / "backend" / "src" / "front_service" / "main.py").read_text(
    encoding="utf-8")
ok("UPDATE risk_scores" not in src_main
   and "DELETE FROM risk_scores" not in src_main
   and "INSERT INTO risk_scores" not in src_main,
   "front service holds no write path to the transaction store")

EVT_CONC = "evt-p113rw-conc-005"
con = sqlite3.connect(str(fm._DBS["risk"]))
con.execute(
    "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
    "risk_band, reason_codes, model_version, ml_score, rule_score, "
    "degraded, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    ("sc-p113rw-005", "F-P113RW-CC", EVT_CONC, 77, "high", "[]",
     MC.CANONICAL_MODEL_VERSION, 0.4, 3.3, 0, now))
con.commit()
con.close()


def hammer(action: str, n: int = 6) -> list[dict]:
    results: list[dict] = []
    lock = threading.Lock()
    errors: list[str] = []

    def worker():
        try:
            tc = TestClient(app)
            rr = tc.post(
                f"/admin/api/transactions/{EVT_CONC}/review/{action}",
                json={}, headers={**HDR, "Content-Type": "application/json"})
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


res = hammer("start")
ok(len(res) == 6 and sum(1 for x in res if x["changed"]) == 1,
   "6 concurrent starts -> exactly one change, rest are replays")
r = rv_get(EVT_CONC)
d = r.json()
ok(d["review_state"] == "UNDER_REVIEW" and d["version"] == 1,
   "concurrent starts cannot silently overwrite (UNDER_REVIEW, v1)")
res = hammer("complete")
ok(len(res) == 6 and sum(1 for x in res if x["changed"]) == 1,
   "6 concurrent completes -> exactly one change, rest are replays")
r = rv_get(EVT_CONC)
d = r.json()
ok(d["review_state"] == "REVIEWED" and d["version"] == 2,
   "concurrent completes serialize to REVIEWED at v2")
starts = [json.loads(x["payload_summary"]) for x in
          review_audit_rows("admin_review_started")]
ok(sum(1 for p in starts if p["event_id"] == EVT_CONC) == 1,
   "concurrency produced exactly one audit event per transition")

# ── [9] frontend pins ─────────────────────────────────────────────────
ok('id="dash-review-kpi"' in shell and "Needs Review →" in shell
   and "goNeedsReview" in shell and 'id="dash-needs-review"' in shell,
   "dashboard: clickable Needs Review KPI wired to the review queue")
ok("addEventListener('click', goNeedsReview)" in shell,
   "Needs Review KPI has a real click handler")
ok('id="f-rchips"' in shell, "transactions: review filter chip row present")
rchips = re.findall(r'data-rchip="([a-z]+)"', shell)
ok(rchips == ["all", "needs", "under", "reviewed"],
   f"review chips are All/Needs/Under/Reviewed ({rchips})")
ok(re.findall(r'data-chip="([a-z]+)"', shell)
   == ["all", "flagged", "approved", "blocked"],
   "original quick-filter chips untouched")
ok("set('review', " in shell and "reviewWord(row.review_state)" in shell,
   "review filter feeds the API and rows render human words")
ok("<th>Review</th>" in shell and 'data-label="Review"' in shell,
   "results table has the compact Review column (stacked-card labelled)")
for word in ("Needs review", "Under review", "Reviewed"):
    ok(f"{word}:" in shell or f"{word}'" in shell,
       f"human review word defined: {word}")
ok('id="d-investigation"' in shell
   and "operator review workflow — not a fraud decision" in shell,
   "detail: Investigation section, explicitly not a fraud decision")
for label in ("Start review", "Mark reviewed", "Reopen review", "Add note"):
    ok(f"'{label}'" in shell, f"action button present: {label}")
ok("d.review" in shell and "rv.review_state" in shell,
   "review state renders from the backend payload only")
ok("Investigation notes" in shell and "No investigation notes yet" in shell,
   "notes timeline with honest empty state")
ok("n.reviewer_id" in shell and "n.created_at" in shell
   and "white-space: pre-wrap" in shell,
   "notes render author, timestamp and raw text")
ok('id="d-review-error"' in shell
   and "Review update failed" in shell,
   "failed writes surface honest UI feedback")
ok("$('d-review-state').innerHTML = '<span class=\\\"skel\\\"" in shell
   or "$('d-review-state').innerHTML = '<span class=\"skel\"" in shell,
   "loading state uses the skeleton")
ok("maxlength=\"2000\"" in shell and "Notes are limited to 2000 characters."
   in shell, "note length bounded client-side and server-side")
ok("reviewPost('/review/' + action)" in shell
   and "reviewPost('/notes'" in shell,
   "actions POST to the backend endpoints, no frontend-only state")
ok("reviewWord(f.review_state)" in shell,
   "live feed shows review state without changing it")
for etype, label in (("admin_review_started", "Started investigation"),
                     ("admin_review_completed", "Completed investigation"),
                     ("admin_review_reopened", "Reopened investigation"),
                     ("admin_review_note_added", "Added investigation note")):
    ok(f"{etype}: '{label}'" in shell,
       f"audit human label: {etype} -> {label}")
ok("table.stackable thead" in shell and "content: attr(data-label)" in shell,
   "responsive stacked-card layout still in force")
ok("confirm_alert" not in shell and "Confirm fraud" not in shell
   and "Mark as fraud" not in shell,
   "no fraud-confirmation affordance anywhere in the console")
for act in ("Override model", "Change threshold", "Promote model",
            "Modify audit", "Delete transaction", "Change risk score"):
    ok(act not in shell, f"no bypass action offered: {act}")

# ── [10] canonical constants + governance states ──────────────────────
ok(MC.CANONICAL_THRESHOLD == 0.018758, "threshold constant unchanged")
ok(MC.CANONICAL_FEATURE_VERSION == "v1", "feature version unchanged")
ok(MC.CANONICAL_MODEL_ID == "altman_native", "governance model_id unchanged")
ok(MC.CANONICAL_RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904",
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

# ── [11] production artifact identity AFTER ───────────────────────────
after = snapshot_artifacts()
after["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
changed = [k for k in WATCHED if before.get(k) != after.get(k)]
ok(not changed,
   f"production artifacts byte-identical across the suite ({changed})")

print()
print(f"PHASE 113 REVIEW: {n_assert} assertions, {len(failures)} failures")
if failures:
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PHASE 113 REVIEW: ALL PASS")
