"""Phase 117 — Operational Exceptions & Attention Center: suite.

Sections
--------
 [1] production artifact identity BEFORE the suite runs
 [2] harness: isolated temp DB_DIR, a measurable completed review (real
     cycle-begin audit event), a flagged row for the concurrency race, and
     admin login — status/chain caches stay cold (no lifespan refresher)
 [3] Attention API: 401 (unauthenticated / bad bearer / bad cookie, even
     with injection-shaped queries), response schema (exact five types in
     order, exact item keys), bounded single-line values, signature with
     no injectable parameter, query injection ignored safely
 [4] data quality: zero -> OK, positive -> ATTENTION (singular/plural
     message), 24h window honored (old row excluded), route to the
     existing blocked filter, source failure -> UNAVAILABLE (never 0)
 [5] runtime: READY / MODEL_NOT_READY / DRIFTED / FAILED preserved
     verbatim from the cached risk /health body, missing health ->
     UNAVAILABLE, no duplicate attestation logic anywhere
 [6] audit: clean strict chain -> OK, historical quarantine -> ATTENTION
     (never reported clean while strict verification is broken), newly
     detected failure -> distinct wording carrying the chain's own
     reason, missing verdict -> UNAVAILABLE; chain evaluation untouched
 [7] review telemetry: measurable -> OK with count 0, partially
     unmeasurable -> ATTENTION with the honest gap, workload/duration
     unavailable -> UNAVAILABLE (count null, never a fabricated zero)
 [8] governance: canonical SYSTEM_READINESS / RWV / promotion states
     verbatim, no false production-failure interpretation
 [9] security: no secrets/PANs/tokens in responses or shell, the
     endpoint writes no audit events, no new audit event type, audit
     chain row count unchanged across a request
 [10] regression: Phase-113/114/115/116 pins (state machine, queue,
      aging, workload/summary contract, URL restoration, navigation,
      concurrency race + exactly one audit event), new Attention Center
      UI pins, script blocks, shared pins
 [11] canonical constants + qualified_datasets + RWV / promotion states
 [12] production artifact identity AFTER (byte-identical)

Everything runs against an ISOLATED temp DB_DIR: no row written here can
reach the shared production DB-3/DB-4, and no production artifact is ever
opened for write. No network, no model fitting, no threshold/release work.

Run from backend/:
  ../.venv/Scripts/python.exe scripts/phase117_attention_center_test.py
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

DB_TMP = tempfile.mkdtemp(prefix="ps14_p117att_")
os.environ["DB_DIR"] = DB_TMP
ADMIN_USER = "admin"
ADMIN_PASS = "p117att-suite-pass-7f3d"
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


EID = {"rev_ok": "evt-p117att-revok-0001",
       "race": "evt-p117att-race-0001"}
con.execute(
    "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
    "risk_band, reason_codes, model_version, ml_score, rule_score, "
    "degraded, scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    ("sc-p117att-race", "F-P117ATT-0001", EID["race"], 91, "high", "[]",
     MC.CANONICAL_MODEL_VERSION, 0.5, 4.0, 0, stamp(600)))
con.commit()
con.close()

# rev_ok: a fully measurable completed review (cycle begin 600s ago,
# completion 100s ago -> exact 500s duration).
con = sqlite3.connect(str(fm._DBS["risk"]))
con.execute(
    "INSERT INTO event_reviews (event_id, review_state, reviewer_id, "
    "created_at, updated_at, reviewed_at, version) VALUES (?,?,?,?,?,?,2)",
    (EID["rev_ok"], "REVIEWED", "admin", ago(600), ago(100), ago(100)))
con.commit()
con.close()
append_audit_event(
    "F-P117ATT-0002", "admin_review_started",
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


def att_item(params: dict | None = None, client: TestClient | None = None
             ) -> dict:
    r = attention(client, params)
    if r.status_code != 200:
        return {}
    return {i["type"]: i for i in r.json()["items"]}


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


# Phase 118 §3 extends the item contract with the bounded `evidence`
# block ({available, summary, details}) — every other key and the five
# item types/order stay exactly as Phase 117 pinned them.
ITEM_KEYS = {"type", "state", "count", "message", "source", "available",
             "route", "evidence"}
TYPES = ["DATA_QUALITY_BLOCKS", "RUNTIME_ATTESTATION", "AUDIT_INTEGRITY",
         "REVIEW_TELEMETRY", "SYSTEM_READINESS"]

# ── [3] Attention API: auth, schema, boundedness ──────────────────────
fresh = TestClient(app)
for q in ("", "?workload_range=7d", "?type=DATA' OR '1'='1",
          "?state=../../etc/passwd"):
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

r = attention()
ok(r.status_code == 200, f"attention -> 200 with session ({r.status_code})")
d = r.json()
ok(set(d) == {"ts", "items"}, f"response keys exactly ts+items ({sorted(d)})")
ok(datetime.fromisoformat(d["ts"]) is not None, "ts is an ISO timestamp")
items = d["items"]
ok([i["type"] for i in items] == TYPES,
   f"exactly five typed items in stable order ({[i['type'] for i in items]})")
for i in items:
    ok(set(i) == ITEM_KEYS,
       f"{i.get('type')} carries exactly the minimal item keys ({sorted(i)})")
    ok(isinstance(i["available"], bool),
       f"{i['type']} available is a boolean")
    ok(isinstance(i["message"], str) and 0 < len(i["message"]) <= 300
       and "\n" not in i["message"],
       f"{i['type']} message is a bounded single line ({len(i.get('message',''))})")
    ok(i["count"] is None or (isinstance(i["count"], int)
                              and i["count"] >= 0),
       f"{i['type']} count is null or a non-negative int ({i['count']})")
    ok(isinstance(i["source"], str) and i["source"],
       f"{i['type']} names its source ({i['source']})")
    ok(i["route"] is None or (isinstance(i["route"], str)
                              and i["route"].startswith("/admin")),
       f"{i['type']} route is null or an existing admin path ({i['route']})")

# Cold caches -> runtime/audit honestly unavailable, DQ zero, review OK.
base = att_item()
ok(base["DATA_QUALITY_BLOCKS"]["state"] == "OK"
   and base["DATA_QUALITY_BLOCKS"]["count"] == 0,
   "cold audit store with no blocks -> DQ OK count 0 (real zero)")
ok(base["RUNTIME_ATTESTATION"]["state"] == "UNAVAILABLE"
   and base["RUNTIME_ATTESTATION"]["available"] is False
   and base["RUNTIME_ATTESTATION"]["count"] is None,
   "cold status cache -> runtime UNAVAILABLE, count null (never 0)")
ok(base["AUDIT_INTEGRITY"]["state"] == "UNAVAILABLE"
   and base["AUDIT_INTEGRITY"]["available"] is False,
   "cold chain cache -> audit UNAVAILABLE (never a false clean)")
ok(base["REVIEW_TELEMETRY"]["state"] == "OK"
   and base["REVIEW_TELEMETRY"]["count"] == 0
   and "all 1 completed reviews" in base["REVIEW_TELEMETRY"]["message"],
   "measurable completion -> telemetry OK count 0 "
   f"({base['REVIEW_TELEMETRY']['message']})")

# No injectable parameter exists; shaped query strings are ignored.
sig = inspect.signature(fm.admin_api_attention)
ok(set(sig.parameters) == {"request"},
   f"attention accepts NO client parameters ({list(sig.parameters)})")
r_inj = attention(params={"type": "DATA' OR '1'='1", "state": "../../etc",
                          "route": "/etc/passwd", "count": "-1",
                          "sql": "1; DROP TABLE event_reviews"})
ok(r_inj.status_code == 200
   and [(i["type"], i["state"]) for i in r_inj.json()["items"]]
   == [(i["type"], i["state"]) for i in items],
   "injection-shaped query params are ignored — items unchanged")
con = sqlite3.connect(str(fm._DBS["risk"]))
ok(con.execute("SELECT COUNT(*) FROM event_reviews").fetchone()[0] == 1,
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
            (f"e117dq-{key}", "F-P117ATT-0003", "data_quality_blocked",
             "0" * 64, "1" * 64, "{}", created))
        conn.commit()
    finally:
        conn.close()


dq = att_item()["DATA_QUALITY_BLOCKS"]
ok(dq["state"] == "OK" and dq["count"] == 0
   and "No evaluations blocked" in dq["message"],
   "zero blocks -> OK with a factual zero message")
ok(dq["route"] == "/admin/transactions?data_quality=blocked",
   f"DQ navigates to the existing blocked filter ({dq['route']})")

insert_dq("a", 1)
dq = att_item()["DATA_QUALITY_BLOCKS"]
ok(dq["state"] == "ATTENTION" and dq["count"] == 1
   and dq["message"].startswith("1 evaluation blocked"),
   f"one recent block -> ATTENTION, singular message ({dq['message']})")
insert_dq("old", 48)
dq = att_item()["DATA_QUALITY_BLOCKS"]
ok(dq["count"] == 1,
   f"48h-old block is outside the 24h window (count={dq['count']})")
insert_dq("b", 2)
dq = att_item()["DATA_QUALITY_BLOCKS"]
ok(dq["count"] == 2 and dq["message"].startswith("2 evaluations blocked"),
   f"two recent blocks -> plural message ({dq['message']})")
ok("last 24h" in dq["message"], "message names its time window")

# Source failure -> UNAVAILABLE with null count (never a fabricated 0).
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
ok(dq["state"] == "UNAVAILABLE" and dq["available"] is False
   and dq["count"] is None,
   "unavailable audit store -> DQ UNAVAILABLE, count null (not 0)")

# ── [5] runtime attestation ──────────────────────────────────────────
def health_item(health: dict | None):
    data = ({"risk": {"status": "ok", "health": health}}
            if health is not None else {})
    with cache_guard(fm._STATUS_CACHE, data):
        return att_item()["RUNTIME_ATTESTATION"]


for st in ("READY", "MODEL_NOT_READY", "DRIFTED", "FAILED"):
    it = health_item({"runtime_state": st, "model_readiness": "loaded",
                      "release_attested": True})
    ok(it["state"] == st and it["available"] is True
       and it["count"] is None and it["route"] is None,
       f"canonical runtime state {st} preserved verbatim ({it['state']})")
    ok(st in it["message"], f"message carries the canonical state ({st})")
it = health_item({"runtime_state": "READY", "model_readiness": "loaded",
                  "release_attested": True})
ok("release attested" in it["message"] and "model loaded" in it["message"],
   f"READY message carries model + attestation facts ({it['message']})")
it = health_item({"runtime_state": "DRIFTED", "model_readiness": "missing",
                  "release_attested": False})
ok("release unattested" in it["message"]
   and "model missing" in it["message"],
   f"unattested/drift facts surface verbatim ({it['message']})")
it = health_item(None)
ok(it["state"] == "UNAVAILABLE" and it["available"] is False
   and it["count"] is None,
   "risk /health missing -> runtime UNAVAILABLE (never OK, never 0)")

# No duplicate attestation/chain implementations in this service.
src = (BACKEND / "src" / "front_service" / "main.py").read_text(
    encoding="utf-8").replace("\r\n", "\n")
ok("detect_runtime_drift" not in src and "attest_release" not in src
   and "RuntimeState" not in src,
   "front service never re-runs runtime attestation")
ok(src.count("evaluate_chain") == 2,
   "chain evaluation stays where it was (detail view only — no second "
   "audit-health implementation)")
att_body = src.split("def admin_api_attention")[1].split(
    '@app.get("/admin/api/live")')[0]
ok("evaluate_chain" not in att_body and "_refresh_status_cache" not in att_body
   and "_integrity" not in att_body,
   "attention consumes caches — it never verifies the chain itself")

# ── [6] audit integrity ───────────────────────────────────────────────
def chain_item(chain):
    with cache_guard(fm._CHAIN_CACHE, chain):
        return att_item()["AUDIT_INTEGRITY"]


clean = chain_item({"ok": True, "strict_ok": True, "n_entries": 42,
                    "first_bad_seq": None, "quarantined_breaks": []})
ok(clean["state"] == "OK" and clean["count"] == 0
   and "end-to-end" in clean["message"],
   f"strictly valid chain -> OK ({clean['message']})")

hist = chain_item({"ok": True, "strict_ok": False, "n_entries": 19070,
                   "first_bad_seq": 731,
                   "quarantined_breaks": [731, 735, 740, 745, 750]})
ok(hist["state"] == "ATTENTION" and hist["count"] == 5,
   f"historical quarantine -> ATTENTION count 5 ({hist['state']})")
ok("Historical quarantined fork present" in hist["message"]
   and "731" in hist["message"] and "frozen findings" in hist["message"],
   f"message surfaces the canonical quarantine facts ({hist['message']})")
ok(hist["state"] != "OK" and "end-to-end" not in hist["message"],
   "strict-broken history is NEVER reported as clean (§9)")

live = chain_item({"ok": False, "strict_ok": False, "n_entries": 19071,
                   "first_bad_seq": 19071,
                   "quarantined_breaks": [],
                   "reason": "hash mismatch at seq 19071"})
ok(live["state"] == "ATTENTION"
   and live["message"].startswith("Chain verification failed:")
   and "hash mismatch at seq 19071" in live["message"]
   and "19071" in live["message"],
   f"live failure carries the chain's own reason ({live['message']})")
ok(live["message"] != hist["message"]
   and "Historical" not in live["message"],
   "live failure is worded distinctly from historical quarantine")

missing = chain_item({"ok": None, "error": "http 503"})
ok(missing["state"] == "UNAVAILABLE" and missing["available"] is False
   and missing["count"] is None,
   "chain verdict unavailable -> UNAVAILABLE (never clean, never 0)")
missing2 = chain_item({})
ok(missing2["state"] == "UNAVAILABLE",
   "absent chain cache -> UNAVAILABLE")

# ── [7] review telemetry ──────────────────────────────────────────────
tel = att_item()["REVIEW_TELEMETRY"]
ok(tel["state"] == "OK" and tel["count"] == 0 and tel["available"] is True
   and "all 1 completed reviews" in tel["message"],
   f"fully measurable -> OK count 0 ({tel['message']})")

con = sqlite3.connect(str(fm._DBS["risk"]))
con.execute(
    "INSERT INTO event_reviews (event_id, review_state, reviewer_id, "
    "created_at, updated_at, reviewed_at, version) VALUES (?,?,?,?,?,?,2)",
    ("evt-p117att-revbad-0001", "REVIEWED", "admin", ago(50), ago(50),
     ago(50)))
con.commit()
con.close()
tel = att_item()["REVIEW_TELEMETRY"]
ok(tel["state"] == "ATTENTION" and tel["count"] == 1
   and "1 of 2 completed reviews lack measurable review-cycle telemetry"
   in tel["message"],
   f"missing start -> honest gap, counted once ({tel['message']})")
ok("invalid" not in tel["message"].lower(),
   "telemetry wording never claims the reviews themselves are invalid")
ok("admin" not in tel["message"].lower(),
   "telemetry exposes no reviewer identity")

orig_rows = fm._ro_rows


def _broken_rows(db_name, sql, params=()):
    if db_name == "risk":
        return None
    return orig_rows(db_name, sql, params)


fm._ro_rows = _broken_rows
try:
    tel = att_item()["REVIEW_TELEMETRY"]
    others = att_item()
finally:
    fm._ro_rows = orig_rows
ok(tel["state"] == "UNAVAILABLE" and tel["available"] is False
   and tel["count"] is None
   and tel["message"] == "Review workload telemetry unavailable",
   "workload unavailable -> telemetry UNAVAILABLE, count null (not 0)")
ok(others["DATA_QUALITY_BLOCKS"]["count"] == 2,
   "one broken source never spreads to the other items")


def _audit_rows_down(db_name, sql, params=()):
    if db_name == "audit":
        return None
    return orig_rows(db_name, sql, params)


fm._ro_rows = _audit_rows_down
try:
    tel = att_item()["REVIEW_TELEMETRY"]
finally:
    fm._ro_rows = orig_rows
ok(tel["state"] == "UNAVAILABLE" and tel["count"] is None
   and tel["message"] == "Review duration telemetry unavailable",
   "duration telemetry unavailable -> distinct honest state")

# ── [8] governance ────────────────────────────────────────────────────
gov = att_item()["SYSTEM_READINESS"]
ok(gov["state"] == SYSTEM_READINESS
   == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET",
   f"readiness state preserved verbatim ({gov['state']})")
ok(SYSTEM_READINESS in gov["message"]
   and REAL_WORLD_VALIDATION in gov["message"]
   and PROMOTION_STATE in gov["message"],
   "message distinguishes readiness / RWV / promotion facts")
ok(gov["available"] is True and gov["count"] is None
   and gov["route"] is None and gov["source"] == "governance",
   "governance item is always available, never counted")
for word in ("FAIL", "BROKEN", "CRITICAL", "ERROR", "DOWN", "OUTAGE"):
    ok(word not in gov["message"].upper() and word != gov["state"],
       f"pending dataset is not presented as a production failure: {word}")
ok("BLOCKED_PENDING_ELIGIBLE_DATASET" == REAL_WORLD_VALIDATION,
   "RWV blocked is the canonical governance value, not a service fault")

# ── [9] security ──────────────────────────────────────────────────────
seq_before = len(audit_rows())
attention()
flush_audit_queue(timeout=5)
seq_after = len(audit_rows())
ok(seq_before == seq_after,
   f"attention writes NO audit events ({seq_before} -> {seq_after})")
ok("admin_attention" not in src and "attention_logged" not in src
   and "append_audit_event" not in att_body,
   "no new audit event type invented for attention")

txt = attention().text
ok("passphrase" not in txt and "blob_key" not in txt
   and ADMIN_PASS not in txt and "TOTP" not in txt
   and "otpauth" not in txt and "internal_token" not in txt,
   "no secrets in the attention response")
ok(not re.search(r"\b4[0-9]{15}\b|\b3[0-9]{14}\b", txt),
   "no PAN-shaped values in the attention response")
ok("token" not in txt.lower() or TOKEN not in txt,
   "session token never echoed in the attention response")
ok("environment" not in txt.lower() and "DATABASE_URL" not in txt,
   "no raw environment values in attention messages")

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

# ── [10] regression ───────────────────────────────────────────────────
# Summary contract unchanged by the Phase-117 extraction.
sig = inspect.signature(fm.admin_api_summary)
ok(set(sig.parameters) == {"request", "workload_range"},
   f"summary contract unchanged ({list(sig.parameters)})")
r = c.get("/admin/api/summary", headers=HDR,
          params={"workload_range": "24h"})
ok(r.status_code == 200, f"summary -> 200 ({r.status_code})")
dsum = r.json()
wl = dsum.get("workload")
ok(isinstance(wl, dict) and {"range", "needs_review", "under_review",
                             "reviewed", "reviewed_last_24h",
                             "reviewed_last_7d", "reviewed_in_period",
                             "oldest_waiting_seconds", "active_reviewers",
                             "completed_review_duration",
                             "duration_buckets"} <= set(wl),
   "Phase-116 workload block intact after extraction")
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

# State machine + allowlists pinned (Phase 113/114/115/116).
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

# Audit event types: attention added none.
known = {x["event_type"] for x in audit_rows()}
ok(known <= {"score_generated", "admin_tx_search", "admin_tx_view",
             "admin_login_success", "admin_review_started",
             "admin_review_completed", "admin_review_reopened",
             "admin_review_note_added", "data_quality_blocked"},
   f"no new audit event type ({sorted(known)})")

# Frontend pins: shared surfaces + the new Attention Center.
ok('id="att-section"' in shell and "Attention Center" in shell
   and 'id="att-items"' in shell and 'id="att-count"' in shell,
   "compact Attention Center section on the dashboard")
for eid in ("att-section", "att-items", "att-count"):
    ok(f'id="{eid}"' in shell, f"attention element present: {eid}")
ok("function renderAttention(att)" in shell
   and "function openAttentionItem(route)" in shell
   and "function attentionBadgeCls(state)" in shell,
   "attention renderers present")
ok("'/admin/api/attention'" in shell,
   "dashboard fetches the attention endpoint")
ok("const [s, att] = await Promise.all([" in shell,
   "summary + attention fetched together (no N+1 dashboard waterfall)")
ok("data-att-route" in shell and "closest('[data-att-route]')" in shell,
   "attention buttons carry server routes (event delegation)")
ok("applyTxnQuery(new URLSearchParams(route.split('?')[1] || ''));"
   in shell,
   "transaction routes go through the normal filter->tab flow "
   "(legitimate history entry)")
ok("activateTab('system')" in shell and "activateTab('audit')" in shell,
   "no-route surfaces map to their existing tabs (no placeholder routes)")
ok("indexOf('SYSTEM_READY') === 0) return '';" in shell,
   "standing governance state never gets the warning style (§11)")
for label in ("DATA QUALITY", "RUNTIME", "AUDIT", "REVIEW TELEMETRY",
              "READINESS"):
    ok(label in shell, f"attention label present: {label}")
ok("Attention status unavailable" in shell
   and "'N/A'" in shell,
   "failed attention payload renders the honest unavailable state")
ok("attention status unavailable" not in shell.lower()
   or "Attention status unavailable" in shell,
   "unavailable wording is factual, not a failure claim")

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
   in shell.replace("\r\n", "\n")
   and "'review', 'since', 'until', 'age', 'queue_sort'];"
   in shell.replace("\r\n", "\n"),
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
missing = sorted(x for x in refs if x not in ids_in_html)
ok(not missing, f"every JS-referenced id exists ({missing})")
# Descriptive only: no rankings/grades anywhere in this feature.
seg = shell[shell.find('id="tab-dashboard"'):shell.find('id="tab-live"')]
for word in ("leaderboard", "productivity", "fastest", "slowest",
             "ranked", "grade:"):
    ok(word not in seg.lower() and word not in js.lower()
       and word not in src.lower(),
       f"no ranking/grade language: {word!r}")
ok("remediate" not in js.lower() and "auto-fix" not in js.lower(),
   "attention never claims to remediate anything")

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
print(f"PHASE 117 ATTENTION CENTER: {n_assert} assertions, "
      f"{len(failures)} failures")
if failures:
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PHASE 117 ATTENTION CENTER: ALL PASS")
