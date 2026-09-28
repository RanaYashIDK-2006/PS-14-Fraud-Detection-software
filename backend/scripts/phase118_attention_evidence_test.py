"""Phase 118 — Attention Evidence & Drill-Down: suite.

Sections
--------
 [1] production artifact identity BEFORE the suite runs
 [2] harness: isolated temp DB_DIR, seeds for measurable + unmeasurable
     completed reviews, a reviewable row, admin login, helper functions
     (attention/att_item, cache_guard for the status/chain caches,
     insert_dq, review actions, audit-row readers)
 [3] API contract: 401 (unauthenticated / bad bearer / bad cookie, even
     with injection-shaped queries), response keys exactly ts+items, five
     items in stable order, each item's EXACT key set (Phase-117 keys +
     evidence), evidence shape {available, summary, details} with bounded
     single-line summaries and <= 8 bounded label/value rows, no client
     parameters, two consecutive calls deep-equal (determinism), cold
     sources honestly unavailable (details empty — never a fabricated 0)
 [4] data quality: genuine zero -> evidence available with real 0 row,
     positive count mirrored in evidence, 24h window honored, route row,
     unavailable DB-4 -> evidence unavailable (details empty), the COUNT
     stays a single parameterized query, and the inference path
     (enforce_before_inference) is untouched by this service
 [5] runtime: cached READY / MODEL_NOT_READY / DRIFTED / FAILED preserved
     verbatim in both state and evidence, identity rows come from the
     SAME cached /health body, missing cache -> unavailable evidence, the
     front service never imports or re-runs runtime attestation
 [6] audit: healthy strict verdict, historical quarantined fork (seq 731,
     5 breaks) represented honestly with affected sequences — NEVER
     "clean"/"end-to-end" while strict verification is broken, newly
     detected failure carries the chain's own reason, absent verdict ->
     unavailable evidence, chain evaluation untouched (no second
     verifier)
 [7] review telemetry: measurable completions, telemetry gap mirrored
     exactly (evidence gap == item count), duration source unavailable ->
     completion counts still shown with N/A duration (never zeros),
     workload unavailable -> evidence unavailable/empty, empty period ->
     honest computed zeros, no reviewer identity anywhere in evidence
 [8] readiness: operational readiness / RWV / promotion / qualified
     datasets on four distinct rows with canonical values, no
     failure-words, standing state never presented as a production fault
 [9] security: 401 surfaces, injection-shaped query params ignored,
     viewing evidence writes NO audit events and invents no event type,
     no secrets/PANs/session token in the response
 [10] regression: Phase-113/114/115/116/117 pins (state machine, queue,
      aging, workload/summary contract, URL restoration, navigation,
      concurrency race + exactly one audit event, Attention Center
      behavior), new Phase-118 evidence UI pins (disclosure button,
      aria-expanded/controls, same-response rendering, single fetch),
      script blocks, shared pins
 [11] canonical constants + qualified_datasets + RWV / promotion states
 [12] production artifact identity AFTER (byte-identical)

Everything runs against an ISOLATED temp DB_DIR: no row written here can
reach the shared production DB-3/DB-4, and no production artifact is ever
opened for write. No network, no model fitting, no threshold/release work.

Run from backend/:
  ../.venv/Scripts/python.exe scripts/phase118_attention_evidence_test.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sqlite3
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DB_TMP = tempfile.mkdtemp(prefix="ps14_p118ev_")
os.environ["DB_DIR"] = DB_TMP
ADMIN_USER = "admin"
ADMIN_PASS = "p118ev-suite-pass-7f3d"
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

# ── [2] harness ───────────────────────────────────────────────────────
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

BASE = datetime.now(timezone.utc).replace(microsecond=0)


def ago(sec: int) -> str:
    return (BASE - timedelta(seconds=sec)).isoformat()


def stamp(seconds_ago: int) -> str:
    return (BASE - timedelta(seconds=seconds_ago)).strftime(
        "%Y-%m-%d %H:%M:%S")


EID = {"rev_ok": "evt-p118ev-revok-0001",
       "rev_gap": "evt-p118ev-gap-0001",
       "race": "evt-p118ev-race-0001"}
con.execute(
    "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
    "risk_band, reason_codes, model_version, ml_score, rule_score, "
    "degraded, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    ("sc-p118ev-race", "F-P118EV-0001", EID["race"], 91, "high", "[]",
     MC.CANONICAL_MODEL_VERSION, 0.5, 4.0, 0, stamp(600)))
# rev_ok: fully measurable cycle (start 600s ago, complete 100s ago).
con.execute(
    "INSERT INTO event_reviews (event_id, review_state, reviewer_id, "
    "created_at, updated_at, reviewed_at, version) VALUES (?,?,?,?,?,?,2)",
    (EID["rev_ok"], "REVIEWED", "admin", ago(600), ago(100), ago(100)))
# rev_gap: completed review WITHOUT a start event -> unmeasurable cycle.
con.execute(
    "INSERT INTO event_reviews (event_id, review_state, reviewer_id, "
    "created_at, updated_at, reviewed_at, version) VALUES (?,?,?,?,?,?,2)",
    (EID["rev_gap"], "REVIEWED", "admin", ago(90), ago(20), ago(20)))
con.commit()
con.close()
append_audit_event(
    "F-P118EV-0002", "admin_review_started",
    {"event_id": EID["rev_ok"], "previous_review_state": "UNREVIEWED",
     "new_review_state": "UNDER_REVIEW", "admin_identity": "admin",
     "timestamp": ago(600)})
flush_audit_queue(timeout=10)

c = TestClient(app)
r = c.post("/admin/login",
           json={"username": ADMIN_USER, "passphrase": ADMIN_PASS})
ok(r.status_code == 200 and "token" in r.json(),
   f"login -> 200 + token ({r.status_code})")
TOKEN = r.json()["token"]
HDR = {"Authorization": f"Bearer {TOKEN}"}


def attention(client: TestClient | None = None, params: dict | None = None):
    return (client or c).get("/admin/api/attention", headers=HDR,
                             params=params or {})


def att_item(params: dict | None = None,
             client: TestClient | None = None) -> dict:
    r = attention(client, params)
    if r.status_code != 200:
        return {}
    return {i["type"]: i for i in r.json()["items"]}


def ev_text(item: dict) -> str:
    """Flattened summary + details of one item's evidence, lowercased."""
    ev = item.get("evidence") or {}
    parts = [str(ev.get("summary") or "")]
    parts += [f"{d.get('label')}={d.get('value')}"
              for d in (ev.get("details") or [])]
    return " ".join(parts).lower()


def rv_post(eid: str, action: str):
    h = dict(HDR)
    h["Content-Type"] = "application/json"
    return c.post(f"/admin/api/transactions/{eid}/review/{action}",
                  json={}, headers=h)


def rv_get(eid: str):
    return c.get(f"/admin/api/transactions/{eid}/review", headers=HDR)


def audit_rows(event_type: str | None = None) -> list[dict]:
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
    out: list[str] = []
    for x in audit_rows(etype):
        try:
            p = json.loads(x["payload_summary"])
        except (TypeError, ValueError):
            continue
        ts = p.get("timestamp")
        if p.get("event_id") == eid and isinstance(ts, str):
            out.append(ts)
    return out


class cache_guard:
    """Temporarily replace a module cache dict, restoring it afterwards."""
    def __init__(self, cache: dict, data):
        self.cache, self.data, self.saved = cache, data, None

    def __enter__(self):
        self.saved = self.cache.get("data")
        self.cache.clear()
        self.cache["data"] = self.data
        self.cache["ts"] = datetime.now(timezone.utc)
        return self

    def __exit__(self, *exc):
        self.cache.clear()
        if self.saved is not None:
            self.cache["data"] = self.saved
        return False


EVIDENCE_KEYS = {"available", "summary", "details"}
ITEM_KEYS = {"type", "state", "count", "message", "source", "available",
             "route", "evidence"}
TYPES = ["DATA_QUALITY_BLOCKS", "RUNTIME_ATTESTATION", "AUDIT_INTEGRITY",
         "REVIEW_TELEMETRY", "SYSTEM_READINESS"]
src = (BACKEND / "src" / "front_service" / "main.py").read_text(
    encoding="utf-8").replace("\r\n", "\n")
shell = c.get("/admin").text
shell_lf = shell.replace("\r\n", "\n")

# ── [3] API contract: auth, schema, evidence shape, determinism ──────
fresh = TestClient(app)
for q in ("", "?workload_range=7d", "?type=DATA' OR '1'='1",
          "?evidence=../../etc/passwd"):
    rr = fresh.get("/admin/api/attention" + q)
    ok(rr.status_code == 401,
       f"unauthenticated attention{q.split('&')[0][-24:]} -> 401 "
       f"({rr.status_code})")
rr = fresh.get("/admin/api/attention",
               headers={"Authorization": "Bearer not-a-real-token"})
ok(rr.status_code == 401, f"invalid bearer token -> 401 ({rr.status_code})")
rr = fresh.get("/admin/api/attention",
               cookies={"admin_session": "garbage-session"})
ok(rr.status_code == 401, f"invalid session cookie -> 401 ({rr.status_code})")

sig = inspect.signature(fm.admin_api_attention)
ok(set(sig.parameters) == {"request"},
   f"attention accepts NO client parameters ({list(sig.parameters)})")

r = attention()
ok(r.status_code == 200, f"attention -> 200 with session ({r.status_code})")
d = r.json()
ok(set(d) == {"ts", "items"}, f"response keys exactly ts+items ({sorted(d)})")
ok(datetime.fromisoformat(d["ts"]) is not None, "ts is an ISO timestamp")
items = d["items"]
ok([i["type"] for i in items] == TYPES,
   f"five items in stable order ({[i['type'] for i in items]})")

# The Phase-117 contract plus exactly the bounded evidence block.
for i in items:
    ok(set(i) == ITEM_KEYS,
       f"{i.get('type')} carries exactly the item keys + evidence "
       f"({sorted(i)})")
    ok(i["count"] is None or (isinstance(i["count"], int)
                              and i["count"] >= 0),
       f"{i['type']} count still null or a non-negative int ({i['count']})")
    ev = i.get("evidence")
    ok(isinstance(ev, dict) and set(ev) == EVIDENCE_KEYS,
       f"{i['type']} evidence is exactly available/summary/details "
       f"({sorted(ev) if isinstance(ev, dict) else ev})")
    if not isinstance(ev, dict):
        continue
    ok(isinstance(ev["available"], bool),
       f"{i['type']} evidence available is a boolean")
    ok(isinstance(ev["summary"], str) and 0 < len(ev["summary"]) <= 300
       and "\n" not in ev["summary"],
       f"{i['type']} evidence summary is a bounded single line "
       f"({len(ev.get('summary', ''))})")
    rows = ev["details"]
    ok(isinstance(rows, list) and len(rows) <= 8,
       f"{i['type']} evidence details bounded (<= 8, got "
       f"{len(rows) if isinstance(rows, list) else rows})")
    if isinstance(rows, list):
        if ev["available"]:
            ok(len(rows) >= 1,
               f"{i['type']} available evidence carries at least one row")
        else:
            ok(rows == [],
               f"{i['type']} unavailable evidence has EMPTY details")
    if isinstance(rows, list) and rows:
        ok(all(isinstance(x, dict) and set(x) == {"label", "value"}
               for x in rows),
           f"{i['type']} evidence rows are exactly label/value pairs")
        ok(all(isinstance(x["label"], str) and 0 < len(x["label"]) <= 60
               for x in rows if isinstance(x, dict) and "label" in x),
           f"{i['type']} evidence labels are bounded strings")
        ok(all(isinstance(x["value"], str) and len(x["value"]) <= 200
               for x in rows if isinstance(x, dict) and "value" in x),
           f"{i['type']} evidence values are bounded strings")

# Determinism: two consecutive calls over unchanged sources are identical.
first = attention().json()["items"]
second = attention().json()["items"]
ok(first == second,
   "two consecutive calls return identical items+evidence (deterministic)")

# Cold sources stay honest: unavailable evidence has empty details and
# never a fabricated zero anywhere.
base = att_item()
ok(base["RUNTIME_ATTESTATION"]["evidence"]["available"] is False
   and base["RUNTIME_ATTESTATION"]["evidence"]["details"] == [],
   "cold status cache -> runtime evidence unavailable, details empty")
ok(base["AUDIT_INTEGRITY"]["evidence"]["available"] is False
   and base["AUDIT_INTEGRITY"]["evidence"]["details"] == [],
   "cold chain cache -> audit evidence unavailable, details empty")
ok(base["RUNTIME_ATTESTATION"]["count"] is None
   and base["AUDIT_INTEGRITY"]["count"] is None,
   "unavailable items keep count null (never 0)")
dq0 = base["DATA_QUALITY_BLOCKS"]
ok(dq0["evidence"]["available"] is True
   and any(x == {"label": "Blocked evaluations", "value": "0"}
           for x in dq0["evidence"]["details"]),
   "genuine zero is answered by the store and shown as a real 0")

# Injection-shaped params are ignored — evidence included.
r_inj = attention(params={"type": "DATA' OR '1'='1", "state": "../../etc",
                          "evidence": "1; DROP TABLE event_reviews",
                          "sql": "1; DROP TABLE event_reviews"})
ok(r_inj.status_code == 200
   and r_inj.json()["items"] == items,
   "injection-shaped query params ignored — items AND evidence unchanged")
con = sqlite3.connect(str(fm._DBS["risk"]))
ok(con.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0] == 2,
   "no table dropped by injection-shaped input")
con.close()

# ── [4] data quality ──────────────────────────────────────────────────
def insert_dq(key: str, hours_ago: float) -> None:
    conn = sqlite3.connect(str(fm._DBS["audit"]))
    try:
        created = (BASE - timedelta(hours=hours_ago)).strftime(
            "%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO audit_events (event_id, fraud_id, event_type, "
            "prev_hash, entry_hash, payload_summary, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (f"e118dq-{key}", "F-P118EV-0003", "data_quality_blocked",
             "0" * 64, "1" * 64, "{}", created))
        conn.commit()
    finally:
        conn.close()


dq = att_item()["DATA_QUALITY_BLOCKS"]
ev = dq["evidence"]
ok(ev["available"] is True and dq["count"] == 0
   and "0 blocked evaluations in the last 24h" in ev["summary"],
   f"zero -> evidence answers with a real 0 ({ev['summary']})")
ok({"label": "Window", "value": "last 24h"} in ev["details"]
   and {"label": "Event type", "value": "data_quality_blocked"} in ev["details"]
   and {"label": "Navigation",
        "value": "/admin/transactions?data_quality=blocked"} in ev["details"],
   "evidence carries window, event type, and the existing route")

insert_dq("a", 1)
dq = att_item()["DATA_QUALITY_BLOCKS"]
ev = dq["evidence"]
ok(dq["count"] == 1
   and {"label": "Blocked evaluations", "value": "1"} in ev["details"]
   and ev["summary"] == "1 blocked evaluation in the last 24h",
   f"positive count mirrored exactly ({ev['summary']})")
ok(dq["state"] == "ATTENTION",
   "evidence never changes the item's state")
insert_dq("old", 48)
dq = att_item()["DATA_QUALITY_BLOCKS"]
ok(dq["count"] == 1
   and dq["evidence"]["details"][0] == {"label": "Blocked evaluations",
                                        "value": "1"},
   "48h-old block stays outside the 24h window in evidence too")
insert_dq("b", 2)
dq = att_item()["DATA_QUALITY_BLOCKS"]
ok(dq["count"] == 2
   and dq["evidence"]["summary"] == "2 blocked evaluations in the last 24h",
   "two blocks -> plural evidence summary")

# Unavailable DB-4 -> evidence unavailable, empty details, honest summary.
orig_scalar = fm._ro_scalar


def _broken_scalar(db_name, sql, params=()):
    if db_name == "audit":
        return None
    return orig_scalar(db_name, sql, params)


fm._ro_scalar = _broken_scalar
try:
    dq = att_item()["DATA_QUALITY_BLOCKS"]
finally:
    fm._ro_scalar = orig_scalar
ok(dq["state"] == "UNAVAILABLE" and dq["count"] is None
   and dq["evidence"]["available"] is False
   and dq["evidence"]["details"] == []
   and "unavailable" in dq["evidence"]["summary"].lower(),
   "unavailable DB-4 -> evidence unavailable, no fabricated zero")

# The blocked count stays ONE parameterized query; the front service
# never touches the inference path.
dq_body = src.split("def _attention_dq_item")[1].split(
    "def _attention_runtime_item")[0]
ok("SELECT COUNT(*) FROM audit_events " in dq_body
   and "AND created_at >= ?" in dq_body
   and "(since,)" in dq_body
   and 'f"SELECT' not in dq_body and "f'SELECT" not in dq_body,
   "attention blocked count is a single parameterized query "
   "(no f-string SQL in the attention builder)")
ok("enforce_before_inference" not in src
   and "rules_fallback" not in src,
   "front service never touches the inference/decision path")

# ── [5] runtime attestation ──────────────────────────────────────────
def health_item(health: dict | None):
    data = ({"risk": {"status": "ok", "health": health}}
            if health is not None else {})
    with cache_guard(fm._STATUS_CACHE, data):
        return att_item()["RUNTIME_ATTESTATION"]


for st in ("READY", "MODEL_NOT_READY", "DRIFTED", "FAILED"):
    it = health_item({"runtime_state": st, "model_readiness": "loaded",
                      "release_attested": True})
    ev = it["evidence"]
    ok(it["state"] == st and ev["available"] is True,
       f"canonical state {st} preserved in state AND evidence")
    ok(any(x == {"label": "Runtime state", "value": st}
           for x in ev["details"]),
       f"evidence row carries runtime_state {st} verbatim")
    ok(f"runtime_state={st}" in ev["summary"],
       f"evidence summary carries the raw field for {st}")
    ok(it["count"] is None and it["route"] is None,
       f"runtime item stays uncounted/unrouted for {st}")

full = health_item({"runtime_state": "READY", "model_readiness": "loaded",
                    "release_attested": True, "liveness": "alive",
                    "readiness": "ready",
                    "release_id": "release-altman_native_E_hardneg_cert_x",
                    "model_id": "altman_native", "feature_version": "v1"})
rows = {x["label"]: x["value"] for x in full["evidence"]["details"]}
ok(rows.get("Model readiness") == "loaded"
   and rows.get("Release attestation") == "attested"
   and rows.get("Liveness") == "alive" and rows.get("Readiness") == "ready",
   "evidence exposes cached /health facts (model/attestation/liveness)")
ok(rows.get("Release") == "release-altman_native_E_hardneg_cert_x"
   and rows.get("Model") == "altman_native"
   and rows.get("Feature version") == "v1",
   "evidence carries the identity already safe for admin display")
unatt = health_item({"runtime_state": "DRIFTED", "model_readiness": "missing",
                     "release_attested": False})
ok(any(x == {"label": "Release attestation", "value": "unattested"}
       for x in unatt["evidence"]["details"]),
   "unattested release stated verbatim in evidence")
gone = health_item(None)
ok(gone["state"] == "UNAVAILABLE"
   and gone["evidence"]["available"] is False
   and gone["evidence"]["details"] == []
   and gone["evidence"]["summary"] == "Cached risk /health body absent",
   "missing health -> unavailable evidence with an honest summary")

ok("detect_runtime_drift" not in src and "attest_release" not in src
   and "RuntimeState" not in src,
   "front service never imports or re-runs runtime attestation")
ok(src.count("evaluate_chain") == 2,
   "chain evaluation stays where it was (no second verifier)")
att_body = src.split("def admin_api_attention")[1].split(
    '@app.get("/admin/api/live")')[0]
ok("evaluate_chain" not in att_body and "_refresh_status_cache" not in att_body
   and "_integrity" not in att_body and "append_audit_event" not in att_body,
   "attention endpoint consumes sources read-only in the same response")

# ── [6] audit integrity ──────────────────────────────────────────────
def chain_item(chain):
    with cache_guard(fm._CHAIN_CACHE, chain):
        return att_item()["AUDIT_INTEGRITY"]


clean = chain_item({"ok": True, "strict_ok": True, "n_entries": 42,
                    "first_bad_seq": None, "quarantined_breaks": []})
ev = clean["evidence"]
ok(clean["state"] == "OK" and ev["available"] is True
   and ev["summary"] == "Strict verification passed end-to-end",
   f"healthy strict verdict -> evidence ({ev['summary']})")
ok({"label": "Chain status", "value": "verified end-to-end (strict)"}
   in ev["details"]
   and {"label": "Quarantined forks", "value": "0"} in ev["details"]
   and {"label": "Total entries", "value": "42"} in ev["details"],
   "healthy evidence carries strict status, fork count, entries")

hist = chain_item({"ok": True, "strict_ok": False, "n_entries": 19070,
                   "first_bad_seq": 731,
                   "quarantined_breaks": [731, 735, 740, 745, 750],
                   "finding_ids": ["FQ-1", "FQ-2", "FQ-3", "FQ-4", "FQ-5"]})
ev = hist["evidence"]
text = ev_text(hist)
ok(hist["state"] == "ATTENTION" and hist["count"] == 5
   and ev["available"] is True,
   "historical quarantine keeps ATTENTION count 5 with evidence present")
ok("731" in ev["summary"] and "5 quarantined forks" in ev["summary"],
   f"evidence summary states the fork facts ({ev['summary']})")
rows = {x["label"]: x["value"] for x in ev["details"]}
ok(rows.get("Strict verification") == "broken"
   and rows.get("First broken sequence") == "731"
   and rows.get("Quarantined forks") == "5"
   and rows.get("Affected sequences") == "731, 735, 740, 745, 750"
   and rows.get("Frozen findings") == "5"
   and rows.get("Total entries") == "19070",
   "evidence details expose the authoritative quarantine facts")
ok("clean" not in text and "end-to-end" not in text,
   "strict-broken history is NEVER evidenced as clean (§9)")
ok(hist["state"] != "OK" and "end-to-end" not in hist["message"].lower(),
   "item message stays non-clean as well")

live = chain_item({"ok": False, "strict_ok": False, "n_entries": 19071,
                   "first_bad_seq": 19071, "quarantined_breaks": [],
                   "reason": "hash mismatch at seq 19071"})
ev = live["evidence"]
ok(live["state"] == "ATTENTION"
   and "19071" in ev["summary"]
   and "hash mismatch at seq 19071" in ev["summary"],
   f"live failure evidence carries the chain's own reason ({ev['summary']})")
rows = {x["label"]: x["value"] for x in ev["details"]}
ok(rows.get("Chain status") == "failed"
   and rows.get("Reason") == "hash mismatch at seq 19071"
   and rows.get("First bad sequence") == "19071",
   "live failure evidence rows are factual")
ok("newly detected failure" in ev["summary"].lower()
   and "historical" not in ev["summary"].lower(),
   "live failure is worded distinctly from historical quarantine")

missing = chain_item({"ok": None, "error": "http 503"})
ok(missing["state"] == "UNAVAILABLE"
   and missing["evidence"]["available"] is False
   and missing["evidence"]["details"] == []
   and missing["evidence"]["summary"] == "Cached chain verdict absent",
   "absent verdict -> unavailable evidence (never clean, never 0)")
missing2 = chain_item({})
ok(missing2["evidence"]["available"] is False,
   "empty chain cache -> unavailable evidence")

# ── [7] review telemetry ─────────────────────────────────────────────
tel = att_item()["REVIEW_TELEMETRY"]
ev = tel["evidence"]
ok(tel["state"] == "ATTENTION" and tel["count"] == 1
   and ev["available"] is True,
   "seeded gap -> ATTENTION with evidence present")
rows = {x["label"]: x["value"] for x in ev["details"]}
ok(rows.get("Completed reviews") == "2"
   and rows.get("Measurable cycles") == "1"
   and rows.get("Telemetry gap") == "1"
   and rows.get("Period") == "all time"
   and rows.get("Duration aggregates") == "available",
   "evidence rows mirror the Phase-116 workload numbers exactly")
ok(f"1 of 2 completed reviews have a measurable" in ev["summary"],
   f"evidence summary states measurable/total ({ev['summary']})")
ok(tel["count"] == int(rows["Telemetry gap"]),
   "evidence gap == item count (same source, same response)")
ok("invalid" not in ev_text(tel) and "reviewer" not in ev_text(tel)
   and "admin" not in ev_text(tel),
   "telemetry evidence exposes no reviewer identity and never calls the "
   "reviews invalid")

# Make the second completion measurable -> OK with gap 0 in evidence.
append_audit_event(
    "F-P118EV-0004", "admin_review_started",
    {"event_id": EID["rev_gap"], "previous_review_state": "UNREVIEWED",
     "new_review_state": "UNDER_REVIEW", "admin_identity": "admin",
     "timestamp": ago(60)})
flush_audit_queue(timeout=10)
tel = att_item()["REVIEW_TELEMETRY"]
ev = tel["evidence"]
rows = {x["label"]: x["value"] for x in ev["details"]}
ok(tel["state"] == "OK" and tel["count"] == 0
   and rows.get("Completed reviews") == "2"
   and rows.get("Measurable cycles") == "2"
   and rows.get("Telemetry gap") == "0",
   "fully measurable -> evidence gap 0, state OK")

# Duration source down: completion counts STILL shown, duration N/A.
orig_rows = fm._ro_rows


def _audit_rows_down(db_name, sql, params=()):
    if db_name == "audit":
        return None
    return orig_rows(db_name, sql, params)


fm._ro_rows = _audit_rows_down
try:
    tel = att_item()["REVIEW_TELEMETRY"]
finally:
    fm._ro_rows = orig_rows
ev = tel["evidence"]
rows = {x["label"]: x["value"] for x in ev["details"]}
ok(tel["state"] == "UNAVAILABLE" and tel["count"] is None
   and ev["available"] is True
   and "duration telemetry unavailable" in ev["summary"].lower()
   and rows.get("Completed reviews") == "2"
   and rows.get("Measurable cycles") == "N/A"
   and rows.get("Telemetry gap") == "N/A"
   and rows.get("Duration aggregates") == "unavailable",
   "duration down -> completion counts shown, duration rows N/A (no zeros)")


def _risk_rows_down(db_name, sql, params=()):
    if db_name == "risk":
        return None
    return orig_rows(db_name, sql, params)


fm._ro_rows = _risk_rows_down
try:
    tel = att_item()["REVIEW_TELEMETRY"]
    others = att_item()
finally:
    fm._ro_rows = orig_rows
ok(tel["state"] == "UNAVAILABLE"
   and tel["evidence"]["available"] is False
   and tel["evidence"]["details"] == []
   and tel["evidence"]["summary"] == "Review workload source unavailable",
   "workload down -> evidence unavailable and empty (no fabricated 0)")
ok(others["DATA_QUALITY_BLOCKS"]["count"] == 2,
   "one broken source never spreads to the other items")

# Empty period: honest computed zeros (real store answer, not absence).
orig_workload = fm._review_workload


def _empty_period(now, workload_range, needs, under, reviewed):
    return {"reviewed_in_period": 0,
            "completed_review_duration": {"count": 0}}


fm._review_workload = _empty_period
try:
    tel = att_item()["REVIEW_TELEMETRY"]
finally:
    fm._review_workload = orig_workload
ev = tel["evidence"]
rows = {x["label"]: x["value"] for x in ev["details"]}
ok(tel["state"] == "OK" and tel["count"] == 0
   and ev["available"] is True
   and rows.get("Completed reviews") == "0"
   and rows.get("Measurable cycles") == "0"
   and rows.get("Telemetry gap") == "0",
   "empty period -> honest computed zeros with available evidence")

# ── [8] readiness ────────────────────────────────────────────────────
gov = att_item()["SYSTEM_READINESS"]
ev = gov["evidence"]
rows = {x["label"]: x["value"] for x in ev["details"]}
ok(gov["state"] == SYSTEM_READINESS
   == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET",
   f"readiness state preserved verbatim ({gov['state']})")
ok(ev["available"] is True and gov["count"] is None and gov["route"] is None,
   "readiness evidence always available, never counted or routed")
ok(list(rows) == ["Operational readiness", "Real-world validation",
                  "Promotion state", "Qualified datasets"],
   f"four distinct evidence rows, concepts kept separate ({list(rows)})")
ok(rows["Operational readiness"] == SYSTEM_READINESS
   and rows["Real-world validation"] == REAL_WORLD_VALIDATION
   and rows["Promotion state"] == PROMOTION_STATE
   and rows["Qualified datasets"] == "(none)",
   "each row carries its own canonical value; qualified datasets ()")
for word in ("FAIL", "BROKEN", "CRITICAL", "ERROR", "DOWN", "OUTAGE"):
    ok(word not in ev_text(gov).upper(),
       f"pending dataset is never evidenced as a production failure: {word}")
ok("system_ready_pending_eligible_dataset" in ev["summary"].lower()
   and "blocked_pending_eligible_dataset" in ev["summary"].lower()
   and "promotion_gate_required" in ev["summary"].lower(),
   "evidence summary states the three canonical governance values")

# ── [9] security ─────────────────────────────────────────────────────
seq_before = len(audit_rows())
attention()
flush_audit_queue(timeout=5)
seq_after = len(audit_rows())
ok(seq_before == seq_after,
   f"viewing evidence writes NO audit events ({seq_before} -> {seq_after})")
known = {x["event_type"] for x in audit_rows()}
ok(known <= {"score_generated", "admin_tx_search", "admin_tx_view",
             "admin_login_success", "admin_review_started",
             "admin_review_completed", "admin_review_reopened",
             "admin_review_note_added", "data_quality_blocked"},
   f"no new audit event type ({sorted(known)})")
ok("admin_attention" not in src and "attention_logged" not in src
   and "evidence_logged" not in src,
   "no audit event type invented for evidence viewing")

txt = attention().text
ok(TOKEN not in txt and ADMIN_PASS not in txt
   and "passphrase" not in txt and "blob_key" not in txt
   and "TOTP" not in txt and "otpauth" not in txt,
   "no secrets/credentials in the attention+evidence response")
ok(not re.search(r"\b4[0-9]{15}\b|\b3[0-9]{14}\b", txt),
   "no PAN-shaped values in the response")
ok("traceback" not in txt.lower() and "site-packages" not in txt
   and "C:\\" not in txt,
   "no filesystem/debug output in evidence")
ok(TOKEN not in txt and "session token" not in txt.lower(),
   "session token never echoed in the response")

# ── [10] regression: previous phases + the new evidence UI ───────────
sig = inspect.signature(fm.admin_api_summary)
ok(set(sig.parameters) == {"request", "workload_range"},
   f"summary contract unchanged ({list(sig.parameters)})")
r = c.get("/admin/api/summary", headers=HDR, params={"workload_range": "24h"})
ok(r.status_code == 200, f"summary -> 200 ({r.status_code})")
dsum = r.json()
wl = dsum.get("workload")
ok(isinstance(wl, dict) and {"range", "needs_review", "under_review",
                             "reviewed", "reviewed_last_24h",
                             "reviewed_last_7d", "reviewed_in_period",
                             "oldest_waiting_seconds", "active_reviewers",
                             "completed_review_duration",
                             "duration_buckets"} <= set(wl),
   "Phase-116 workload block intact")
ok(wl.get("needs_review") == dsum["counts"].get("needs_review")
   and wl.get("reviewed") == dsum["counts"].get("reviewed"),
   "workload still mirrors the summary scalars")
_con = sqlite3.connect(str(fm._DBS["risk"]))
_n_rev = _con.execute("SELECT COUNT(*) FROM event_reviews "
                      "WHERE review_state='REVIEWED'").fetchone()[0]
_con.close()
ok(dsum["counts"].get("reviewed") == _n_rev,
   f"reviewed scalar authoritative ({dsum['counts'].get('reviewed')} "
   f"== table count {_n_rev})")

# State machine + allowlists pinned (Phase 113/114/115/116/117).
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
                           "1h_4h": (3600, 14400), "4h_24h": (14400, 86400),
                           "gt24h": (86400, None)}
   and fm._QUEUE_SORTS == ("oldest", "newest"),
   "Phase 114/115 filter allowlists pinned")
ok(fm._WORKLOAD_RANGES == ("24h", "7d", "30d", "all")
   and fm._WORKLOAD_RANGE_DAYS == {"24h": 1, "7d": 7, "30d": 30,
                                   "all": None},
   "Phase-116 workload period allowlist pinned")

# Review concurrency: exactly one transition + one audit event.
r = rv_post(EID["race"], "start")
ok(r.status_code == 200 and r.json()["review_state"] == "UNDER_REVIEW",
   "race row started")
results: list[tuple[int, dict]] = []


def race_complete():
    rr = rv_post(EID["race"], "complete")
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
ok(len(results) == 2 and all(x[0] == 200 for x in results),
   f"concurrent completes both answered 200 ({[x[0] for x in results]})")
ok(sum(1 for x in results if x[1].get("changed") is True) == 1,
   "exactly one transition applied under the race")
ok(len(event_ts_list(EID["race"], "admin_review_completed")) == 1,
   "race produced exactly ONE completion audit event")
ok(rv_get(EID["race"]).json()["review_state"] == "REVIEWED",
   "race row ended REVIEWED")

# Attention Center (Phase 117) still intact.
ok('id="att-section"' in shell and "Attention Center" in shell
   and 'id="att-items"' in shell and 'id="att-count"' in shell,
   "compact Attention Center section on the dashboard")
for eid in ("att-section", "att-items", "att-count"):
    ok(f'id="{eid}"' in shell, f"attention element present: {eid}")
ok("function renderAttention(att)" in shell
   and "function openAttentionItem(route)" in shell
   and "function attentionBadgeCls(state)" in shell,
   "Phase-117 attention renderers present")

# Phase-118 evidence UI pins.
ok("View evidence" in shell and "Hide evidence" in shell,
   "disclosure affordance with explicit collapsed/expanded wording")
ok("function attentionEvidenceHtml(ev)" in shell
   and "function toggleAttentionEvidence(type)" in shell,
   "evidence render + toggle functions present")
ok("const ATTENTION_OPEN = new Set();" in shell
   and "ATTENTION_OPEN.has(type)" in shell
   and "ATTENTION_OPEN.add(type)" in shell
   and "ATTENTION_OPEN.delete(type)" in shell,
   "open state tracked per item so a refresh keeps the panel open")
ok("const opening = panel.hidden;" in shell
   and "panel.hidden = !opening;" in shell,
   "panel visibility toggled through the hidden attribute")
ok("aria-expanded" in shell and "aria-controls" in shell
   and 'data-att-ev="' in shell and "closest('[data-att-ev]')" in shell,
   "native button disclosure with aria state (keyboard accessible)")
ok(".att-ev {" in shell and ".att-ev[hidden] { display: none; }" in shell
   and "overflow-wrap: anywhere" in shell
   and "grid-template-columns: minmax(90px, max-content) 1fr" in shell,
   "evidence panel CSS: collapsed by default, mobile-safe wrapping")
ok("'N/A'" in shell and "Attention status unavailable" in shell,
   "honest N/A rendering kept for missing payloads/evidence")
# One authoritative response per load: evidence is never refetched.
ok(shell_lf.count("'/admin/api/attention'") == 1,
   f"exactly one attention fetch site ({shell_lf.count(chr(39) + '/admin/api/attention' + chr(39))})")
ok("/admin/api/attention/" not in shell_lf
   and "attention/evidence" not in shell_lf,
   "no separate evidence endpoint (same-response consistency, §6)")
ok("const [s, att] = await Promise.all([" in shell,
   "summary + attention still fetched together (no N+1 waterfall)")
ok("data-att-route" in shell and "closest('[data-att-route]')" in shell,
   "attention navigation buttons keep server routes (event delegation)")
ok("applyTxnQuery(new URLSearchParams(route.split('?')[1] || ''));"
   in shell,
   "transaction routes go through the normal filter->tab flow")
ok("activateTab('system')" in shell and "activateTab('audit')" in shell,
   "no-route surfaces map to their existing tabs")
ok("indexOf('SYSTEM_READY') === 0) return '';" in shell,
   "standing governance state never gets the warning style (§11)")
for label in ("DATA QUALITY", "RUNTIME", "AUDIT", "REVIEW TELEMETRY",
              "READINESS"):
    ok(label in shell, f"attention label present: {label}")
ok("'/admin/api/attention'" in shell,
   "dashboard fetches the attention endpoint")

# Phase-113/114/115/116 surfaces intact.
ok('id="dash-review-kpi"' in shell and "Needs Review →" in shell
   and 'id="dash-needs-review"' in shell,
   "Phase-113 dashboard KPI intact")
ok('id="dash-review-sub"' in shell,
   "Phase-114 dashboard KPI sub-line intact")
ok('id="f-queue-oldest"' in shell and 'id="f-queue-aging"' in shell
   and "function renderQueueAging(aging)" in shell,
   "Phase-115 queue aging surface intact")
ok("function fmtWait(sec)" in shell
   and "function renderWorkload(w, c)" in shell
   and "function workloadUrlQuery()" in shell
   and "function applyWorkloadQuery(sp)" in shell,
   "Phase-116 workload surface intact")
ok("const WORKLOAD_RANGES = ['24h', '7d', '30d', 'all'];" in shell
   and "'/admin/api/summary?workload_range='" in shell,
   "Phase-116 period control intact")
ok("if (d.changed) await afterReviewAction(action);" in shell,
   "Phase-114 queue refresh pin intact")
ok("if (!r.detail) applyTxnQuery(new URLSearchParams(window.location.search));"
   in shell
   and "if (!r.detail) applyWorkloadQuery(new URLSearchParams("
       "window.location.search));" in shell,
   "Back/Forward restore list + workload state")
ok(shell.count(
    "window.location.pathname + window.location.search !== target") >= 3,
   "all push paths idempotent (identical target never pushes)")
ok("const wantOpen = txnOpenExact;" in shell and "txnOpenExact = false;"
   in shell, "auto-open gate intact")
ok("if (livePaused) { setConn('PAUSED'); return; }" in shell
   and shell.count("if (livePaused) { setConn('PAUSED'); return; }") >= 2,
   "live-monitor pause race stays fixed")
ok(re.findall(r'data-rchip="([a-z]+)"', shell)
   == ["all", "needs", "under", "reviewed"],
   "review chips stay All/Needs/Under/Reviewed")
ok(re.findall(r'data-chip="([a-z]+)"', shell)
   == ["all", "flagged", "approved", "blocked"],
   "quick-filter chips untouched")
ok("const TXN_URL_KEYS = ['q', 'event_id', 'fraud_id', 'band', 'decision',"
   in shell_lf
   and "'review', 'since', 'until', 'age', 'queue_sort'];"
   in shell_lf,
   "transaction URL allowlist unchanged (URL state untouched by §15)")
ok("function updateTxnNav()" in shell
   and "prev.disabled = !(idx > 0);" in shell
   and "function txnQueueLabel()" in shell
   and "const REVIEW_QUEUE_LABEL = { needs: 'Needs Review'" in shell,
   "queue navigation intact")
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
ids_in_html = set(re.findall(r'id="([^"$]+)"', shell))
refs = set(re.findall(r"\$\('([^']+)'\)", js)) | set(
    re.findall(r"getElementById\('([^']+)'\)", js))
missing_ids = sorted(x for x in refs if x not in ids_in_html)
ok(not missing_ids, f"every JS-referenced id exists ({missing_ids})")
seg = shell[shell.find('id="tab-dashboard"'):shell.find('id="tab-live"')]
for word in ("leaderboard", "productivity", "fastest", "slowest",
             "ranked", "grade:"):
    ok(word not in seg.lower() and word not in js.lower()
       and word not in src.lower(),
       f"no ranking/grade language: {word!r}")
ok("remediate" not in js.lower() and "auto-fix" not in js.lower(),
   "attention/evidence never claims to remediate anything")
ok("severity" not in ev_text(gov) and "priority" not in ev_text(gov),
   "evidence introduces no severity/priority classification")

# ── [11] canonical constants + governance states ─────────────────────
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
mdl = c.get("/admin/api/summary", headers=HDR).json().get(
    "details", {}).get("model") or {}
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

# ── [12] production artifact identity AFTER ───────────────────────────
after = snapshot_artifacts()
after["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
changed = [k for k in WATCHED if before.get(k) != after.get(k)]
ok(not changed,
   f"production artifacts byte-identical across the suite ({changed})")

print()
print(f"PHASE 118 ATTENTION EVIDENCE: {n_assert} assertions, "
      f"{len(failures)} failures")
if failures:
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PHASE 118 ATTENTION EVIDENCE: ALL PASS")
