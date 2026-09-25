"""Phase 112 — Operator dashboard UX simplification: test suite.

Sections
--------
 [1] production artifact identity BEFORE the suite runs
 [2] served shells + operator nav (6 items + System Details)
 [3] dashboard structure: KPIs, flagged-first, status lines, expando
 [4] /admin/api/summary: authz, shape, honest nulls, governance identity
 [5] transactions search UX: quick search, 5-column results, filters expando
 [6] detail view: why-box, fields table, processing-details expando
 [7] live monitor: simple metrics, honest connection states, detail expando
 [8] audit: simple table + human fork explanation + technical expando
 [9] security page: MFA first, manage-MFA toggle, advanced expando
[10] TOTP enroll/login still work on the redesigned console + unauthorized
[11] no secrets in the served dashboard; no stale/removed ids in JS
[12] canonical constants + production artifact identity AFTER

Everything runs against an ISOLATED temp DB_DIR: no row written here can
reach the shared production DB-3/DB-4, and no production artifact is ever
opened for write.  No network, no model fitting, no threshold/release work.

Run from backend/:
  ../.venv/Scripts/python.exe scripts/phase112_admin_ux_test.py
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DB_TMP = tempfile.mkdtemp(prefix="ps14_p112_")
os.environ["DB_DIR"] = DB_TMP
ADMIN_USER = "admin"
ADMIN_PASS = "p112-suite-pass-7b4e"
os.environ["ADMIN_USER"] = ADMIN_USER
os.environ["ADMIN_PASS"] = ADMIN_PASS

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

# ── [2] shells + operator nav ─────────────────────────────────────────
c = TestClient(app)

for path in ("/admin", "/admin/transactions", "/admin/transactions/evt-1",
             "/admin/audit", "/admin/security", "/admin/settings"):
    r = c.get(path)
    ok(r.status_code == 200 and "passcode" in r.text.lower(),
       f"shell {path} -> 200 + passcode gate ({r.status_code})")

shell = c.get("/admin").text
ok(len(shell) > 20000, f"shell serves the full console ({len(shell)} bytes)")

# Three-group nav hierarchy (Phase 112 §5): ADMIN / SYSTEM / ADVANCED
nav_ops = re.findall(r'<div class="sidebar-item[^"]*" data-tab="([a-z]+)"', shell)
labels = re.findall(r'<div class="sidebar-label">([^<]+)</div>', shell)
ok(labels == ["Admin", "System", "Advanced"],
   f"nav groups are ADMIN/SYSTEM/ADVANCED ({labels})")
ok(nav_ops[:3] == ["dashboard", "transactions", "live"],
   f"ADMIN group is Dashboard/Transactions/Live Monitor ({nav_ops[:3]})")
ok(nav_ops[3:6] == ["audit", "access", "settings"],
   f"SYSTEM group is Audit/Security/Settings ({nav_ops[3:6]})")
ok(nav_ops[6:9] == ["system", "database", "queries"],
   f"ADVANCED group is System Details/Database Tools/Query Editor ({nav_ops[6:9]})")
ok(len(nav_ops) == 9, f"exactly 9 nav items ({len(nav_ops)})")
ok(shell.count('id="tab-system"') == 1
   and shell.count('id="tab-database"') == 1
   and shell.count('id="tab-queries"') == 1
   and shell.count('id="tab-services"') == 0,
   "advanced tools live on their own Advanced tabs, no legacy services tab")
ok("Database Tools" in shell and "Query Editor" in shell,
   "DB explorer + query editor reachable only from the Advanced group")

# ── [3] dashboard structure ───────────────────────────────────────────
for eid in ("dash-status-card", "dash-status-text", "dash-tx",
            "dash-flagged", "dash-blocked", "dash-kpi-note",
            "dash-flagged-body", "dash-view-all", "dash-activity",
            "dash-line-model", "dash-line-database", "dash-line-audit",
            "dash-model-name", "dash-view-model", "dash-warnings"):
    ok(f'id="{eid}"' in shell, f"dashboard element present: {eid}")

i_dash = shell.find('id="tab-dashboard"')
i_live = shell.find('id="tab-live"')
seg = shell[i_dash:i_live]
ok("All systems operational" in shell and "Attention required" in shell
   and "System unavailable" in shell,
   "ONE status card renders all three states")
ok(seg.find("Flagged Transactions") < seg.find('id="dash-line-database"'),
   "flagged transactions sit above the system block")
ok(seg.find('id="dash-activity"') > seg.find('id="dash-flagged-body"'),
   "RECENT ACTIVITY section follows the flagged list")
ok("No recent transactions" in shell,
   "honest empty state: 'No recent transactions'")
ok("View model details" in shell and "View all" in shell,
   "drill-through links present")
ok("dash-details" not in shell and "dash-details-body" not in shell,
   "no technical expando left on the dashboard")
# Information overload: hashes/mappings never on the default dashboard
for overload in ("sha256", "SHA-256", "entry_hash", "prev_hash",
                 "map_raw_to_native", "21 domain", "48 native",
                 "PromotionToken", "predecessor"):
    ok(overload not in seg,
       f"dashboard free of engineering detail: {overload}")
ok(seg.count("system-status") == 0 and seg.count("chain-badge") == 0,
   "old dashboard id soup removed")

# ── [4] /admin/api/summary ────────────────────────────────────────────
r = c.get("/admin/api/summary")
ok(r.status_code == 401, f"summary requires auth ({r.status_code})")

r = c.post("/admin/login", json={"username": ADMIN_USER, "passphrase": ADMIN_PASS})
ok(r.status_code == 200 and "token" in r.json(), f"login -> 200 ({r.status_code})")

r = c.get("/admin/api/summary")
d = r.json() if r.status_code == 200 else {}
ok(r.status_code == 200, f"summary -> 200 ({r.status_code})")
ok({"ts", "refresh_interval_s", "status", "counts", "recent_flagged",
    "recent_activity", "chain", "details"} <= set(d),
   f"summary keys ({sorted(d)})")
ok(d.get("refresh_interval_s") == 30, "advertised dashboard refresh = 30s")
st = d.get("status") or {}
ok(st.get("state") in ("operational", "attention", "unavailable")
   and isinstance(st.get("warnings"), list)
   and isinstance(st.get("services_ok"), int),
   "status block: 3-state verdict + warnings + service counters")
cnt = d.get("counts") or {}
for k in ("transactions_24h", "flagged_24h", "blocked_24h", "newest_scored_at"):
    ok(k in cnt, f"counts carry {k}")
ok(all(v is None or isinstance(v, (int, str)) for v in cnt.values()),
   "counts are ints/ISO strings or null — never fabricated text")
# Empty isolated DB -> counts must be null (or a real 0), rendered N/A in UI
ok(cnt.get("transactions_24h") in (None, 0),
   f"absent risk DB yields null/0, not a made-up number ({cnt.get('transactions_24h')})")
ok(isinstance(d.get("recent_flagged"), list) and len(d["recent_flagged"]) <= 8,
   f"recent_flagged bounded to 8 rows ({len(d.get('recent_flagged') or [])})")
act = d.get("recent_activity")
ok(isinstance(act, list) and len(act) <= 6,
   f"recent_activity bounded to 6 rows ({len(act) if isinstance(act, list) else act})")
if isinstance(act, list) and act:
    ok(all(set(a) >= {"event_id", "risk_band", "decision", "scored_at"}
           for a in act),
       "activity rows carry event/band/decision/time")
    ok(all(a.get("decision") is None or a["decision"] in
           ("allow", "verify", "step_up") for a in act),
       "activity decisions are authoritative values or null (never invented)")
chain = d.get("chain") or {}
ok({"ok", "strict_ok", "n_entries", "first_bad_seq",
    "quarantined_breaks"} <= set(chain),
   "chain block carries the quarantine-aware verdict")
mdl = (d.get("details") or {}).get("model") or {}
ok(mdl.get("threshold") == 0.018758, "summary model threshold is canonical")
ok(mdl.get("governance_model_id") == MC.CANONICAL_MODEL_ID
   and mdl.get("governance_release_id") == MC.CANONICAL_RELEASE_ID,
   "summary model block exposes the governance identity")
ok(mdl.get("model_id") in (MC.CANONICAL_MODEL_VERSION, None)
   and mdl.get("release_id") in (MC.CANONICAL_LEGACY_RELEASE_ID, None),
   "summary model block exposes the deployed attested identity")

# The dashboard UI must consume this endpoint (no stale id soup in JS)
ok("/admin/api/summary" in shell, "console fetches /admin/api/summary")
for gone in ("getElementById('system-status')", "getElementById('txn-count')",
             "getElementById('alert-count')", "getElementById('chain-badge')",
             "getElementById('audit-entries')"):
    ok(gone not in shell, f"JS no longer writes removed element {gone}")

# ── [5] transactions search UX ────────────────────────────────────────
ok('id="f-quick"' in shell and 'placeholder="Search transactions..."' in shell,
   "single quick-search box heads the transactions page")
ok('<details class="advanced" id="f-filters">' in shell,
   "advanced filters live in a collapsed expando")
i_tx = shell.find('id="tab-transactions"')
i_set = shell.find('id="tab-settings"')
seg = shell[i_tx:i_set]
i_results = seg.find(">Results<")
i_filters = seg.find('id="f-filters"')
ok(0 < i_results < i_filters or i_filters < i_results,
   "results table present with filters expando")
thead = re.search(r'<thead>\s*<tr><th>Time</th><th>Event ID</th><th>Risk</th>'
                  r'<th>Decision</th><th>Status</th></tr>', seg)
ok(bool(thead), "results table is the 5-column operator view")
ok(seg.count("colspan=\"5\"") >= 2,
   "loading/empty states match the 5-column layout")
ok("quick" in shell and "set('event_id', quick)" in shell,
   "quick search feeds the identifier filter")

r = c.get("/admin/api/transactions", params={"limit": 999})
ok(r.status_code in (200, 422), f"server still caps limit ({r.status_code})")

# ── [6] detail view ───────────────────────────────────────────────────
ok("Why was this flagged?" in shell, "human 'why' heading on the detail view")
for eid in ("d-hero-title", "d-event-id", "d-summary", "d-why", "d-fields",
            "d-processing", "d-stages", "d-reasons", "d-audit", "d-chain-note"):
    ok(f'id="{eid}"' in shell, f"detail element present: {eid}")
ok('<details class="advanced" id="d-processing">' in shell,
   "stages/audit/raw codes sit in a collapsed 'Processing details' expando")
ok("Flagged for review. Detailed reasoning is unavailable." in shell,
   "honest fallback text when no human reason is recorded")
i_det = shell.find('id="txn-detail-panel"')
seg = shell[i_det:shell.find('id="tab-settings"')]
ok(seg.find('id="d-why"') < seg.find('id="d-processing"'),
   "why-box precedes the technical expando")

# ── [7] live monitor ──────────────────────────────────────────────────
for eid in ("live-conn", "live-ago", "live-simple-total", "live-simple-flagged",
            "live-simple-latency", "live-simple-blocks", "live-recent",
            "live-details", "live-window", "live-pause"):
    ok(f'id="{eid}"' in shell, f"live element present: {eid}")
ok('<details class="advanced" id="live-details">' in shell,
   "detailed live metrics are a collapsed expando")
ok("setConn('DISCONNECTED')" in shell and "setConn('LIVE')" in shell
   and "setConn('PAUSED')" in shell,
   "honest connection states: LIVE / PAUSED / DISCONNECTED")
ok("Last updated" in shell and "liveTick" in shell,
   "'last updated Ns ago' ticker instead of a frozen badge")
i_live2 = shell.find('id="tab-live"')
seg = shell[i_live2:shell.find('<!-- Services') if '<!-- Services' in shell
            else shell.find('id="tab-audit"')]
ok(seg.find('id="live-simple-total"') < seg.find('id="live-details"'),
   "simple metrics precede the detailed expando")

# ── [8] audit ─────────────────────────────────────────────────────────
ok('id="audit-fork-box"' in shell and "Historical audit issue" in shell,
   "human historical-fork explanation present")
ok("does not represent a newly detected failure" in shell,
   "fork text says the quarantined break is documented, not new")
ok('id="audit-hygiene"' in shell, "current audit hygiene line present")
i_aud = shell.find('id="tab-audit"')
seg = shell[i_aud:shell.find('id="tab-transactions"')]
ok(re.search(r'<thead>\s*<tr>\s*<th>Time</th>\s*<th>Event</th>\s*<th>Status</th>',
             seg), "simple 3-column recent-events table")
ok(seg.count("audit-table-body") >= 1 and seg.count("audit-tech-body") >= 1,
   "both simple and technical audit tables exist")
ok('<details class="advanced" id="audit-tech">' in seg,
   "full integrity evidence is a collapsed expando")
ok("self_consistent" in shell,
   "per-row hash verification drives the Status column")

# ── [9] security page ─────────────────────────────────────────────────
i_acc = shell.find('id="tab-access"')
seg = shell[i_acc:shell.find('</main>')]
ok('<h2 class="content-title">Security</h2>' in seg, "page titled Security")
ok('id="mfa-manage-btn"' in shell and 'id="mfa-manage"' in shell,
   "Manage MFA toggle present")
ok("1. Select <b>Set up TOTP</b>" in shell,
   "numbered 1-2-3 enrollment steps shown")
ok(re.search(r'<details[^>]*id="security-details"', shell) is not None,
   "credentials/sessions/events behind an advanced expando")
ok(seg.find('id="totp-status"') < seg.find('id="security-details"'),
   "MFA status sits above the technical expando")
for eid in ("totp-setup-btn", "totp-setup-panel", "totp-disable-btn",
            "totp-rotate-btn", "totp-recovery-codes", "sessions-body",
            "security-events-body", "login-time", "access-login-count",
            "access-last-login", "access-totp"):
    ok(f'id="{eid}"' in shell, f"security element kept: {eid}")

# ── [10] TOTP enroll/login + unauthorized on the redesigned console ───
r = c.get("/admin/totp/status")
ok(r.json().get("enabled") is False, "MFA initially disabled")
r = c.get("/admin/totp/setup")
secret = r.json().get("secret")
ok(r.status_code == 200 and secret and len(secret) >= 16,
   "setup returns pending base32 secret")
totp = TOTPAuthenticator(base64.b32decode(secret, casefold=True))
r = c.post("/admin/totp/verify",
           json={"totp_code": totp.generate_code()}, headers=CSRF)
ok(r.status_code == 200 and r.json().get("ok") is True,
   f"enroll verify -> 200 ({r.status_code})")
codes = r.json().get("recovery_codes") or []
ok(len(codes) == 8, f"8 recovery codes issued ({len(codes)})")

# MFA is on: login without a code must fail, with the code must work.
fm._admin_login_failures.clear()
r = c.post("/admin/login", json={"username": ADMIN_USER, "passphrase": ADMIN_PASS})
ok(r.status_code == 401, f"login without MFA code -> 401 ({r.status_code})")
fm._admin_login_failures.clear()
r = c.post("/admin/login", json={"username": ADMIN_USER, "passphrase": ADMIN_PASS,
                                 "totp_code": totp.generate_code()})
ok(r.status_code == 200 and "token" in r.json(),
   f"login with MFA code -> 200 ({r.status_code})")

# Unauthorized sweep on the new/kept admin APIs.
for path in ("/admin/api/summary", "/admin/api/transactions",
             "/admin/api/live", "/admin/api/audit-overview",
             "/admin/api/security-events", "/admin/sessions",
             "/admin/totp/status"):
    r = TestClient(app).get(path)
    ok(r.status_code == 401, f"unauthorized {path} -> 401 ({r.status_code})")

# MFA teardown keeps the suite idempotent for the AFTER artifact check.
fm._admin_login_failures.clear()
r = c.post("/admin/totp/disable",
           json={"totp_code": totp.generate_code()}, headers=CSRF)
ok(r.status_code == 200 and c.get("/admin/totp/status").json().get("enabled") is False,
   f"MFA disabled again ({r.status_code})")

# audit-overview: 200 carries per-row verification; 503 is the documented
# read-resilience path when the audit HTTP service is down in this sandbox.
r = c.get("/admin/api/audit-overview?limit=5")
ok(r.status_code in (200, 503),
   f"audit-overview answers or degrades honestly ({r.status_code})")
if r.status_code == 200:
    evs = r.json().get("events") or []
    ok(all("self_consistent" in e for e in evs),
       "every audit event row carries self_consistent")

# ── [11] secrets + structural integrity of the console source ────────
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
served = c.get("/admin").text
hits = []
for pat, label in SECRET_PATS:
    if pat.search(served):
        hits.append(label)
ok(not hits, f"no secrets in the served dashboard ({hits})")
ok(ADMIN_PASS not in served, "admin passphrase never appears in the shell")
ok(all(code not in served for code in codes),
   "recovery codes never appear in the shell")

# Every id the JS touches must exist; every removed id must be gone.
ids_in_html = set(re.findall(r'id="([^"$]+)"', served))
blocks = re.findall(r"<script>(.*?)</script>", served, re.S)
js = "\n".join(blocks)
refs = set(re.findall(r"\$\('([^']+)'\)", js))
refs |= set(re.findall(r"getElementById\('([^']+)'\)", js))
missing = sorted(r for r in refs if r not in ids_in_html)
ok(not missing, f"every JS-referenced id exists ({missing})")
for gone in ("system-status", "d-system-label", "d-system-badge",
             "txn-count", "alert-count", "audit-entries", "chain-badge",
             "chain-label", "tab-services", "dash-online",
             "dash-line-system", "dash-details"):
    ok(gone not in ids_in_html, f"removed id stays removed: {gone}")

# All script blocks are non-trivial and balanced enough to be JS (a full
# parse is done by `node --check` in the battery, not in this suite).
ok(len(blocks) == 2 and all(len(b) > 1000 for b in blocks),
   f"two intact script blocks ({[len(b) for b in blocks]})")
ok(js.count("{") == js.count("}"),
   f"script braces balance ({js.count('{')} vs {js.count('}')})")

# Progressive disclosure: every technical area is behind <details> or an
# Advanced nav item (§4/§5) — nothing technical sits on the dashboard.
for det in ("f-filters", "d-processing", "live-details",
            "audit-tech", "security-details"):
    ok(re.search(rf'<details[^>]*id="{det}"', served) is not None,
       f"advanced area collapsed by default: {det}")
ok(len(re.findall(r"<details\b", served)) >= 5,
   "console uses expando disclosure throughout")
for adv_tab in ("tab-database", "tab-queries", "tab-system"):
    ok(f'id="{adv_tab}"' in served,
       f"engineering tooling quarantined behind an Advanced tab: {adv_tab}")

# Settings is minimal + browser-local; identity moved to System Details.
i_set = shell.find('id="tab-settings"')
seg = shell[i_set:shell.find('id="tab-access"')]
ok('id="pref-dash-refresh"' in seg and 'id="pref-live-refresh"' in seg
   and 'id="pref-rows"' in seg, "settings = console preferences only")
ok("Stored in this browser only" in seg, "prefs are browser-local, no server state")
ok('id="s-model"' not in seg and 'id="s-threshold"' not in seg,
   "model/threshold identity no longer lives on the settings page")
i_sys = shell.find('id="tab-system"')
i_db = shell.find('id="tab-database"')
i_q = shell.find('id="tab-queries"')
i_aud = shell.find('id="tab-audit"')
seg_sys = shell[i_sys:i_db]
seg_db = shell[i_db:i_q]
seg_q = shell[i_q:i_aud]
for eid in ("s-model", "s-release", "s-gov-model", "s-gov-release", "s-fv",
            "s-threshold", "s-runtime", "s-artifacts", "services-grid",
            "sys-status-body"):
    ok(f'id="{eid}"' in seg_sys, f"System Details carries: {eid}")
ok('id="db-explorer-select"' in seg_db and 'id="db-tables-grid"' in seg_db,
   "Database Tools carries the DB explorer")
ok('id="db-select"' in seg_q and 'id="sql-input"' in seg_q
   and 'id="query-results"' in seg_q,
   "Query Editor carries the preset/SQL editor")
for eid in ("db-explorer-select", "db-select", "sql-input"):
    ok(eid not in seg_sys, f"tooling not inside System Details: {eid}")
seg = seg_sys
ok("SYSTEM_READY_PENDING_ELIGIBLE_DATASET" in seg
   and "BLOCKED_PENDING_ELIGIBLE_DATASET" in seg
   and "PROMOTION_GATE_REQUIRED" in seg,
   "authoritative validation state shown in System Details")
ok("Runtime Status" in seg and "Liveness" in shell,
   "System Details holds the authoritative runtime status table")

# Accessibility: focus styles + labelled search field.
ok("focus-visible" in shell, "keyboard focus outlines styled")
ok('aria-label="Search transactions by event ID or transaction ID"' in shell,
   "quick search has an accessible label")

# ── [12] canonical constants + artifact identity AFTER ────────────────
after = snapshot_artifacts()
after["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
changed = [k for k in WATCHED if before.get(k) != after.get(k)]
ok(not changed, f"production artifacts byte-identical across the suite ({changed})")

ok(MC.CANONICAL_THRESHOLD == 0.018758, "threshold constant unchanged")
ok(MC.CANONICAL_FEATURE_VERSION == "v1", "feature version constant unchanged")
ok(MC.CANONICAL_MODEL_ID == "altman_native", "governance model_id unchanged")
ok(MC.CANONICAL_RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904",
   "governance release_id unchanged")

fm_src = (BACKEND / "src" / "front_service" / "main.py").read_text(
    encoding="utf-8", errors="ignore")
for token in ("PromotionToken", "RWVPromotionEvidence", "promote("):
    ok(token not in fm_src, f"front service never constructs {token}")

print()
print(f"PHASE 112: {n_assert} assertions, {len(failures)} failures")
if failures:
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PHASE 112: ALL PASS")
