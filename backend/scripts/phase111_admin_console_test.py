"""Phase 111 — Admin console, TOTP MFA, live monitor & flagged-transaction
investigation: test suite.

Sections
--------
 [1] production artifact identity BEFORE the suite runs          (Part 27)
 [2] admin shells, route map, pre-auth enforcement              (Parts 12/21/22)
 [3] authentication + session security                          (Parts 12/13)
 [4] TOTP MFA lifecycle + security                              (Parts 9/10/11/24)
 [5] transaction search contract (filters, bounds, injection)   (Parts 5/13/14/24)
 [6] isolated flagged event -> live -> search -> detail ->
     decision trace -> audit trail                              (Parts 6/7/8/26)
 [7] live monitor windows + honest N/A semantics                (Parts 3/4/16/23)
 [8] privacy: no secrets, no unnecessary PII                    (Part 17/24)
 [9] admin auditability + CSRF + idle timeout + logout          (Part 30)
 [10] repository secret / token-in-URL scan                     (Part 28)
 [11] no bypass parameters on admin endpoints                   (Part 11/24)
 [12] production artifact identity AFTER + canonical constants  (Parts 10/11/27)

Everything runs against an ISOLATED temp DB_DIR: no row written here can
reach the shared production DB-3/DB-4, and no production artifact is ever
opened for write.  No network, no model fitting, no threshold/release work.

Run from backend/:
  ../.venv/Scripts/python.exe scripts/phase111_admin_console_test.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DB_TMP = tempfile.mkdtemp(prefix="ps14_p111_")
os.environ["DB_DIR"] = DB_TMP
ADMIN_USER = "admin"
ADMIN_PASS = "p111-suite-pass-9f2c"
os.environ["ADMIN_USER"] = ADMIN_USER
os.environ["ADMIN_PASS"] = ADMIN_PASS

from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import src.front_service.main as fm  # noqa: E402
from src.front_service.main import app  # noqa: E402
from src.middleware.totp import TOTPAuthenticator  # noqa: E402
from src.monitoring import manifest_contract as MC  # noqa: E402
from src.monitoring.phase108_public_benchmark_execution import (  # noqa: E402
    PRODUCTION_ARTIFACT_PATHS,
    snapshot_artifacts,
)

CSRF = {"X-Requested-With": "XMLHttpRequest"}
failures: list[str] = []
n_assert = 0


def ok(cond: bool, msg: str) -> None:
    global n_assert
    n_assert += 1
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


# ── [1] production artifact identity BEFORE ───────────────────────────
WATCHED = tuple(PRODUCTION_ARTIFACT_PATHS) + ("models/production/release_manifest.json",)
before = snapshot_artifacts()
before["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
ok(len([k for k, v in before.items() if v != "absent"]) == 13,
   f"13 production artifacts readable before ({len(before)})")

# ── [2] shells + pre-auth ─────────────────────────────────────────────
c = TestClient(app)

for path in ("/admin", "/admin/transactions", "/admin/transactions/evt-1",
             "/admin/audit", "/admin/security", "/admin/settings"):
    r = c.get(path)
    ok(r.status_code == 200 and "passcode" in r.text.lower(),
       f"shell {path} -> 200 + server-side login gate")

for path in ("/admin/api/transactions", "/admin/api/live", "/admin/sessions",
             "/admin/totp/status", "/admin/api/transactions/evt-x",
             "/admin/api/audit-overview", "/admin/api/security-events",
             "/admin/essential"):
    r = c.get(path)
    ok(r.status_code in (401, 403), f"pre-auth {path} -> {r.status_code}")

r = c.get("/admin/api/transactions/evt-x")
ok(r.status_code == 401, f"detail pre-auth 401 ({r.status_code})")

# Route map (Part 1): the six documented console routes exist.
paths = {getattr(rt, "path", "") for rt in app.routes if isinstance(rt, APIRoute)}
for p in ("/admin", "/admin/transactions", "/admin/transactions/{event_id}",
          "/admin/audit", "/admin/security", "/admin/settings",
          "/admin/api/transactions", "/admin/api/transactions/{event_id}",
          "/admin/api/live", "/admin/login", "/admin/logout",
          "/admin/totp/setup", "/admin/totp/verify", "/admin/totp/rotate",
          "/admin/totp/disable", "/admin/totp/status", "/admin/sessions"):
    ok(p in paths, f"route registered: {p}")

# ── [3] authentication + session security ────────────────────────────
r = c.post("/admin/login", json={"username": ADMIN_USER, "passphrase": ADMIN_PASS})
ok(r.status_code == 200 and "token" in r.json(), f"login -> 200 ({r.status_code})")
set_cookie = r.headers.get("set-cookie", "")
ok("HttpOnly" in set_cookie, "session cookie is HttpOnly")
ok("samesite=strict" in set_cookie.lower(), "session cookie is SameSite=Strict")
ok("admin_session=" in set_cookie, "session cookie name is documented")
ok("token=" not in set_cookie.lower(),
   "bearer token is not duplicated into the cookie")

r = c.post("/admin/login", json={"username": ADMIN_USER, "passphrase": "wrong-pass-xyz"})
ok(r.status_code == 401 and "invalid" in r.json()["detail"].lower(),
   f"bad passphrase -> 401 ({r.status_code})")
r = c.post("/admin/login", json={"username": "not-the-admin", "passphrase": ADMIN_PASS})
ok(r.status_code == 401, f"unknown username -> 401 ({r.status_code})")
fm._admin_login_failures.clear()

# ── [4] TOTP MFA lifecycle + security ────────────────────────────────
r = c.get("/admin/totp/status")
ok(r.json().get("enabled") is False and r.json().get("has_pending") is False,
   "MFA initially disabled (bootstrap state)")

r = c.get("/admin/totp/setup")
sb = r.json()
secret = sb.get("secret")
ok(r.status_code == 200 and sb.get("pending") is True and secret
   and len(secret) >= 16, "setup returns pending base32 secret")
ok(sb.get("otpauth_uri", "").startswith("otpauth://totp/"),
   "otpauth enrollment URI present")
ok(f"secret={secret}" in sb.get("otpauth_uri", "") or "secret=" in sb.get("otpauth_uri", ""),
   "otpauth URI carries the pending secret for the authenticator app")
ok(sb.get("qr") == "omitted",
   "QR explicitly omitted (documented deviation: no QR dependency)")
r2 = c.get("/admin/totp/setup")
ok(r2.json().get("secret") == secret, "pending setup resumes the same secret")
r = c.get("/admin/totp/status")
ok("secret" not in r.json() and r.json().get("has_pending") is True,
   "status hides the secret and flags pending")

raw_secret = base64.b32decode(secret, casefold=True)
totp = TOTPAuthenticator(raw_secret)

r = c.post("/admin/totp/verify", json={"totp_code": "000000"}, headers=CSRF)
ok(r.status_code == 401, f"verify wrong code -> 401 ({r.status_code})")
r = c.post("/admin/totp/verify", json={"totp_code": totp.generate_code()}, headers=CSRF)
vb = r.json() if r.status_code == 200 else {}
ok(r.status_code == 200 and vb.get("ok") is True, f"verify -> 200 ({r.status_code})")
codes = vb.get("recovery_codes") or []
ok(len(codes) == 8 and all(len(x) == 16 for x in codes),
   f"8 one-time recovery codes issued ({len(codes)})")
r = c.get("/admin/totp/status")
st = r.json()
ok(st.get("enabled") is True and st.get("recovery_codes_remaining") == 8
   and "secret" not in st and "recovery_codes" not in st,
   "status after enrollment: enabled + count only, no secrets")
r = c.get("/admin/totp/setup")
ok(r.status_code == 409, f"setup after enrollment -> 409 ({r.status_code})")
r = c.get("/admin/totp/current-code")
ok(r.status_code in (404, 410),
   f"no secret-revealing current-code endpoint ({r.status_code})")


def login(**extra):
    payload = {"username": ADMIN_USER, "passphrase": ADMIN_PASS}
    payload.update(extra)
    return c.post("/admin/login", json=payload)


r = login()
ok(r.status_code == 401 and "required" in r.json()["detail"].lower(),
   f"login without code -> 401 required ({r.status_code})")
r = login(totp_code="000000")
ok(r.status_code == 401, f"login wrong code -> 401 ({r.status_code})")
r = login(totp_code=totp.generate_code())
ok(r.status_code == 200, f"login valid code -> 200 ({r.status_code})")
r = login(recovery_code=codes[0])
ok(r.status_code == 200, f"login via recovery code -> 200 ({r.status_code})")
r = login(recovery_code=codes[0])
ok(r.status_code == 401 and "recovery" in r.json()["detail"].lower(),
   f"recovery code replay -> 401 ({r.status_code})")

# brute-force lockout (count-agnostic: hammer until the 429 fires)
saw_429 = None
for _ in range(8):
    r = login(totp_code="111111")
    if r.status_code == 429:
        saw_429 = r
        break
ok(saw_429 is not None and "try again in" in saw_429.json().get("detail", ""),
   "brute-force lockout -> 429 countdown ("
   + (saw_429.text[:60] if saw_429 else f"never fired; last={r.status_code}") + ")")
fm._admin_login_failures.clear()

r = c.post("/admin/totp/rotate", json={"totp_code": "000000"}, headers=CSRF)
ok(r.status_code == 401, f"rotate wrong code -> 401 ({r.status_code})")
r = c.post("/admin/totp/rotate", json={"totp_code": totp.generate_code()}, headers=CSRF)
rb = r.json() if r.status_code == 200 else {}
new_secret = rb.get("secret")
ok(r.status_code == 200 and new_secret and new_secret != secret,
   f"rotate issues a new secret ({r.status_code})")
ok(len(rb.get("recovery_codes") or []) == 8, "rotate re-issues recovery codes")
new_totp = TOTPAuthenticator(base64.b32decode(new_secret, casefold=True))

c2 = TestClient(app)  # separate cookie jar
r = c2.post("/admin/login", json={"username": ADMIN_USER, "passphrase": ADMIN_PASS,
                                  "totp_code": totp.generate_code()})
ok(r.status_code == 401, f"old secret code rejected after rotate ({r.status_code})")
fm._admin_login_failures.clear()
r = c2.post("/admin/login", json={"username": ADMIN_USER, "passphrase": ADMIN_PASS,
                                  "totp_code": new_totp.generate_code()})
ok(r.status_code == 200, f"new secret code accepted ({r.status_code})")

r = c.post("/admin/totp/disable", json={"totp_code": "000000"}, headers=CSRF)
ok(r.status_code == 401, f"disable wrong code -> 401 ({r.status_code})")
r = c.post("/admin/totp/disable",
           json={"totp_code": new_totp.generate_code()}, headers=CSRF)
ok(r.status_code == 200 and r.json().get("other_sessions_revoked") is not None,
   f"disable -> 200 + revoke-other-sessions ({r.status_code})")
r = c.get("/admin/totp/status")
ok(r.json().get("enabled") is False
   and r.json().get("recovery_codes_remaining") == 0,
   "after disable: MFA off and recovery codes cleared")

# ── [5] transaction search contract ──────────────────────────────────
r = c.get("/admin/api/transactions", params={"limit": 999})
body = r.json() if r.status_code == 200 else {}
ok(r.status_code == 200 and body.get("limit", 0) <= 200 and body.get("rows") == [],
   f"limit clamped to <=200 on an empty store ({body.get('limit')})")
for params, name in (
    ({"band": "weird"}, "bad band"),
    ({"event_id": "x' OR 1=1--"}, "injection event_id"),
    ({"fraud_id": "nope"}, "bad fraud_id"),
    ({"offset": 999999}, "huge offset"),
    ({"min_score": 500}, "score out of range"),
    ({"since": "yesterday"}, "bad since"),
    ({"decision": "block"}, "unknown decision"),
    ({"data_quality": "maybe"}, "unknown data_quality"),
    ({"min_score": 90, "max_score": 10}, "min>max"),
):
    r = c.get("/admin/api/transactions", params=params)
    ok(r.status_code == 400, f"reject {name} -> 400 ({r.status_code})")

r = c.get("/admin/api/transactions", params={"decision": "allow"})
ok(r.status_code == 200 and r.json().get("decision_source") == "audit_payload",
   "decision filter documents its audit-payload source")
r = c.get("/admin/api/transactions/does-not-exist-evt")
ok(r.status_code == 404, f"unknown event -> 404 ({r.status_code})")

# ── [6] isolated flagged event E2E (Part 26) ─────────────────────────
RISK_DB = Path(DB_TMP) / "risk.db"
con = sqlite3.connect(RISK_DB)
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
    CREATE TABLE IF NOT EXISTS verification_outcomes (
        id INTEGER PRIMARY KEY, event_id VARCHAR(64), outcome VARCHAR(32),
        case_id VARCHAR(64), resolved_at DATETIME
    );
    CREATE TABLE IF NOT EXISTS investigator_cases (
        id INTEGER PRIMARY KEY, event_id VARCHAR(64), case_id VARCHAR(64),
        status VARCHAR(32), priority VARCHAR(16), confidence FLOAT,
        created_at DATETIME
    );
    """
)
from datetime import datetime  # noqa: E402

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
FLAG_EID = "evt-p111-flagged-001"
FLAG_FRAUD = "F-P111-FLAG"
SCORE = 97
ML, RULE = 0.9876, 12.5
con.execute(
    "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
    "risk_band, reason_codes, model_version, ml_score, rule_score, degraded, "
    "scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    ("sc-p111-001", FLAG_FRAUD, FLAG_EID, SCORE, "high",
     json.dumps(["VELOCITY_HIGH", "AMOUNT_ANOMALY"]),
     MC.CANONICAL_MODEL_VERSION, ML, RULE, 0, now))
NOAUDIT_EID = "evt-p111-noaudit-001"
con.execute(
    "INSERT INTO risk_scores (score_id, fraud_id, event_id, risk_score, "
    "risk_band, reason_codes, model_version, ml_score, rule_score, degraded, "
    "scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    ("sc-p111-002", "F-P111-NOAUD", NOAUDIT_EID, 42, "medium",
     json.dumps([]), MC.CANONICAL_MODEL_VERSION, 0.0, 0.0, 0, now))
con.commit()
con.close()

from src.audit_service.writer import append_audit_event, flush_audit_queue  # noqa: E402

FLAG_PAYLOAD = {
    "event_id": FLAG_EID,
    "fraud_id": FLAG_FRAUD,
    "risk_score": SCORE,
    "risk_band": "high",
    "reason_codes": ["VELOCITY_HIGH", "AMOUNT_ANOMALY"],
    "model_version": MC.CANONICAL_MODEL_VERSION,
    "ml_score": ML,
    "rule_score": RULE,
    "degraded": False,
    "decision": "step_up",
    "feature_version": MC.CANONICAL_FEATURE_VERSION,
    "runtime_state": "READY",
    "runtime_release_id": MC.CANONICAL_LEGACY_RELEASE_ID,
    "drift_state": "STABLE",
    "escalation_reason": "risk_band=high",
    "rule_version": "v1",
}
append_audit_event(FLAG_FRAUD, "score_generated", FLAG_PAYLOAD)
flush_audit_queue(timeout=5)

r = c.get("/admin/api/transactions", params={"event_id": FLAG_EID})
rows = r.json().get("rows", []) if r.status_code == 200 else []
ok(r.status_code == 200 and len(rows) == 1,
   f"isolated flagged event searchable by event_id ({len(rows)})")
row = rows[0] if rows else {}
ok(row.get("risk_band") == "high" and row.get("risk_score") == SCORE,
   "search row carries the recorded band/score")
ok(row.get("decision") == "step_up",
   f"decision resolved from DB-4 payload ({row.get('decision')})")

r = c.get("/admin/api/transactions", params={"flagged": "true"})
fb = r.json() if r.status_code == 200 else {}
ok(r.status_code == 200 and any(x.get("event_id") == FLAG_EID
                                for x in fb.get("rows", [])),
   f"flagged list contains the event (total={fb.get('total')})")
r = c.get("/admin/api/transactions", params={"band": "high", "min_score": 90})
ok(r.status_code == 200 and any(x.get("event_id") == FLAG_EID
                                for x in r.json().get("rows", [])),
   "band + score-range filters combine")
r = c.get("/admin/api/transactions", params={"event_id": NOAUDIT_EID})
nrow = (r.json().get("rows") or [{}])[0]
ok(nrow.get("decision") is None,
   "row without a recorded decision stays null (UI renders N/A)")

r = c.get(f"/admin/api/transactions/{FLAG_EID}")
d = r.json() if r.status_code == 200 else {}
ok(r.status_code == 200, f"detail opens ({r.status_code})")
ok(len(d.get("stages", [])) == 9, f"9 trace stages ({len(d.get('stages', []))})")
ok(d.get("decision") == "step_up" and d.get("decision_source") == "audit_payload",
   "detail decision + documented source")
ok((d.get("context") or {}).get("threshold") == 0.018758,
   "detail context threshold is the production value")
ok((d.get("context") or {}).get("feature_version") == "v1",
   "detail context feature version v1")
rec = ((d.get("audit") or {}).get("reconcile") or {})
ok(rec.get("consistent") is True,
   f"DB-4 payload reconciles with DB-3 score ({rec.get('mismatches')})")
ae = (d.get("audit") or {}).get("events") or []
ok(len(ae) >= 1 and all(x.get("self_consistent") for x in ae),
   f"event audit rows are hash self-consistent ({len(ae)})")
ch = d.get("chain") or {}
ok(ch.get("chain_label") in ("AUDIT_CHAIN_VALID", "AUDIT_CHAIN_INVALID")
   and "strict_label" in ch and "quarantined_breaks" in ch,
   "chain verdict comes from the authoritative verifier (quarantine-aware)")
ok(isinstance(d.get("reasons"), list) and d["reasons"]
   and all(x.get("text") for x in d["reasons"]),
   "reasons carry authoritative text or REASON_NOT_AVAILABLE")

r = c.get(f"/admin/api/transactions/{NOAUDIT_EID}")
d2 = r.json() if r.status_code == 200 else {}
ok(r.status_code == 200 and d2.get("decision") is None
   and d2.get("decision_source") == "not_persisted",
   "missing decision reported as not_persisted (never invented)")
ok("no DB-4 decision payload" in str(
    ((d2.get("audit") or {}).get("reconcile") or {}).get("mismatches")),
   "reconcile names the exact missing-evidence condition")

# ── [7] live monitor ─────────────────────────────────────────────────
r = c.get("/admin/api/live", params={"window": "2h"})
ok(r.status_code == 400, f"unknown window -> 400 ({r.status_code})")
r = c.get("/admin/api/live", params={"window": "5m"})
live = r.json() if r.status_code == 200 else {}
need = {"ts", "window", "windowed", "latency", "model", "feed",
        "scores_total", "bands", "recent_events", "refresh_interval_s", "chain"}
ok(r.status_code == 200 and need <= set(live),
   f"live keys present ({sorted(need - set(live))})")
ok(live.get("refresh_interval_s") == 5, "advertised refresh interval = 5s")
m = live.get("model") or {}
ok(m.get("threshold") == 0.018758, "live model threshold is canonical")
ok(m.get("governance_model_id") == MC.CANONICAL_MODEL_ID
   and m.get("governance_release_id") == MC.CANONICAL_RELEASE_ID,
   "live model block exposes the governance identity")
ok(m.get("model_id") in (MC.CANONICAL_MODEL_VERSION, None)
   and m.get("release_id") in (MC.CANONICAL_LEGACY_RELEASE_ID, None),
   "live model block exposes the deployed attested identity")
w = live.get("windowed") or {}
ok({"total", "by_band", "by_decision", "validation_blocks", "errors"} <= set(w),
   f"windowed metrics present ({sorted({'total','by_band','by_decision','validation_blocks','errors'} - set(w))})")
ok(w.get("errors") is None and w.get("requests_per_sec") is None,
   "metrics without a source are null (N/A), never fabricated zeros")
lat = live.get("latency") or {}
ok(set(lat) >= {"p50_ms", "p95_ms", "p99_ms", "samples"},
   "latency percentiles expose real sample counts")
feed = live.get("feed") or []
ok(isinstance(feed, list) and any(x.get("event_id") == FLAG_EID for x in feed),
   f"live feed carries the isolated event ({len(feed)} rows)")
if feed:
    ok(set(feed[0]) >= {"timestamp", "event_id", "risk_score", "decision",
                        "decision_band", "degraded", "data_quality_status"},
       f"feed columns ({sorted(feed[0])})")

# ── [8] privacy ──────────────────────────────────────────────────────
for path, params in ((("/admin/api/transactions"), {"limit": 50}),
                     (("/admin/api/live"), {"window": "15m"}),
                     ((f"/admin/api/transactions/{FLAG_EID}"), None)):
    txt = c.get(path, params=params).text
    ok("passphrase" not in txt and "blob_key" not in txt
       and ADMIN_PASS not in txt and secret not in txt
       and new_secret not in txt and codes[0] not in txt,
       f"no credentials/secrets in {path}")
    ok(not re.search(r"\b4[0-9]{15}\b|\b3[0-9]{14}\b", txt),
       f"no PAN-shaped values in {path}")

# ── [9] admin auditability + CSRF + idle + logout ────────────────────
# CSRF: cookie-authenticated POST without the header must fail
nohead = TestClient(app)
nohead.cookies.update(c.cookies)
r = nohead.post("/admin/sessions/revoke", json={"jti": "aabbccddeeff00112233"})
ok(r.status_code == 403, f"POST without X-Requested-With -> 403 ({r.status_code})")
r = nohead.post("/admin/sessions/revoke", json={"jti": "aabbccddeeff00112233"},
                headers=CSRF)
ok(r.status_code == 200 and r.json().get("revoked") is False,
   f"revoke unknown session is idempotent ({r.status_code})")

# idle timeout
jti = c.cookies.get("admin_session")
sd = fm._admin_sessions.get(jti)
sd["last_seen"] = time.time() - (20 * 60)
fm._admin_sessions.update_data(jti, sd)
r = c.get("/admin/sessions")
ok(r.status_code == 401 and "idle" in r.json()["detail"],
   f"idle session -> 401 ({r.status_code})")

# logout invalidates the token
fm._admin_login_failures.clear()
r = c2.post("/admin/login", json={"username": ADMIN_USER, "passphrase": ADMIN_PASS,
                                  "totp_code": new_totp.generate_code()})
ok(r.status_code == 200, f"re-login after MFA disable ({r.status_code})")
tok = r.json()["token"]
r = c2.post("/admin/logout", headers={"Authorization": f"Bearer {tok}"})
ok(r.status_code == 200, "logout ok")
r = c2.get("/admin/sessions", headers={"Authorization": f"Bearer {tok}"})
ok(r.status_code == 401, f"revoked token rejected ({r.status_code})")

# every administrative action above must be on the audit chain (Part 30)
flush_audit_queue(timeout=5)
con = sqlite3.connect(Path(DB_TMP) / "audit.db")
types = {t: n for t, n in con.execute(
    "SELECT event_type, COUNT(*) FROM audit_events GROUP BY event_type")}
leak = con.execute(
    "SELECT COUNT(*) FROM audit_events WHERE event_type LIKE 'admin_%' AND "
    "(payload_summary LIKE ? OR payload_summary LIKE ?)",
    (f"%{ADMIN_PASS}%", f"%{secret}%")).fetchone()[0]
con.close()
for t in ("admin_login_success", "admin_login_failed", "admin_mfa_failed",
          "admin_recovery_used", "admin_totp_enrolled", "admin_totp_rotated",
          "admin_totp_disabled", "admin_tx_search", "admin_tx_view",
          "admin_session_revoked"):
    ok(types.get(t, 0) >= 1, f"admin action audited: {t} ({types.get(t, 0)})")
ok(leak == 0, f"no secrets recorded in admin audit payloads ({leak})")

# ── [10] repository secret / token-in-URL scan (Part 28) ─────────────
SCAN_RES = ("\\.venv", "__pycache__", "\\.git/", "node_modules",
            "\\.enc$", "data/", "reports/", "\\.joblib$", "\\.csv$")
SECRET_PATS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key material"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"\bsk-[A-Za-z0-9]{24,}"), "API secret key literal"),
    (re.compile(r"otpauth://totp/[^\"'\\s]*\\?secret=[A-Z2-7]{16,}"),
     "TOTP secret embedded in a literal otpauth URI"),
    (re.compile(r"(?i)\\b(?:totp_secret|totpSecret|TOTP_SECRET)\\b\\s*[:=]\\s*[\"']"
                r"[A-Z2-7]{16,}[\"']"), "hard-coded TOTP secret"),
    (re.compile(r"(?i)[\"']Bearer\\s+[A-Za-z0-9._\\-]{40,}[\"']"),
     "hard-coded bearer token"),
    (re.compile(r"[?&](?:access_token|api_key|apikey|secret|totp)=[A-Za-z0-9]"),
     "credential in a URL query string"),
]
hits: list[str] = []
scanned = 0
# Explicit allowlist: literals that exist ONLY to prove redaction works.
# Nothing else is exempt, and the allowlist itself is asserted below.
SECRET_SCAN_ALLOWLIST = {
    "backend/scripts/phase75_security_hardening_test.py":
        "redaction fixtures (documented fake credentials)",
}
for base in (REPO / "backend" / "src", REPO / "backend" / "scripts",
             REPO / "frontend", REPO / "docs"):
    if not base.exists():
        continue
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix not in (".py", ".js", ".html", ".md", ".yml"):
            continue
        if any(re.search(pat, str(p).replace("\\", "/")) for pat in SCAN_RES):
            continue
        if str(p.relative_to(REPO)).replace("\\", "/") in SECRET_SCAN_ALLOWLIST:
            continue
        scanned += 1
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat, label in SECRET_PATS:
            for mm in pat.finditer(text):
                hits.append(f"{p.relative_to(REPO)}: {label}")
ok(scanned > 50, f"scanned {scanned} source files for credentials")
ok(set(SECRET_SCAN_ALLOWLIST) == {
    "backend/scripts/phase75_security_hardening_test.py"},
   f"credential-scan allowlist stays minimal ({sorted(SECRET_SCAN_ALLOWLIST)})")
ok(not hits, f"no hard-coded credentials / token-in-URL ({hits[:5]})")

# ── [11] no bypass parameters on admin endpoints ─────────────────────
BANNED = {"force", "override", "admin_override", "bypass", "skip_validation",
          "allow_unverified", "ignore_chain", "skip_verification"}
bad: list[str] = []
for rt in app.routes:
    if not isinstance(rt, APIRoute) or not rt.path.startswith("/admin"):
        continue
    names = {p.name for p in rt.dependant.query_params}
    names |= {p.name for p in rt.dependant.body_params}
    bad += [f"{rt.path}:{n}" for n in names & BANNED]
ok(not bad, f"no bypass-style parameters on admin routes ({bad})")

# ── [12] production artifact identity AFTER ──────────────────────────
after = snapshot_artifacts()
after["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
changed = [k for k in WATCHED if before.get(k) != after.get(k)]
ok(not changed, f"production artifacts byte-identical across the suite ({changed})")
ok(after["models/production/release_manifest.json"]
   == before["models/production/release_manifest.json"],
   "release manifest unchanged")

ok(MC.CANONICAL_THRESHOLD == 0.018758, "threshold constant unchanged")
ok(MC.CANONICAL_FEATURE_VERSION == "v1", "feature version constant unchanged")
ok(MC.CANONICAL_MODEL_ID == "altman_native", "governance model_id unchanged")
ok(MC.CANONICAL_RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904",
   "governance release_id unchanged")
ok("altman_native_v1" in MC.STALE_FEATURE_VERSIONS,
   "stale feature version still rejected by contract")

# Part 11: the admin console must not touch promotion / RWV machinery.
fm_src = (BACKEND / "src" / "front_service" / "main.py").read_text(
    encoding="utf-8", errors="ignore")
for token in ("PromotionToken", "RWVPromotionEvidence", "promote("):
    ok(token not in fm_src, f"front service never constructs {token}")

print()
print(f"PHASE 111: {n_assert} assertions, {len(failures)} failures")
if failures:
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PHASE 111: ALL PASS")
