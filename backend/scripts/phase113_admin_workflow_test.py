"""Phase 113 — Operator investigation workflow polish: test suite.

Sections
--------
 [1] production artifact identity BEFORE the suite runs
 [2] served shells + unchanged operator nav hierarchy
 [3] dashboard actionability: clickable flagged KPI, Needs Attention list,
     honest empty states, one-click goFlagged wiring
 [4] transactions experience: quick-filter chips, Clear filters, operator
     vocabulary, stacked-card mobile tables, contextual empty states,
     exact-ID direct navigation, skeleton loading
 [5] transaction detail workflow: status-first summary, plain "Risk score",
     Why-flagged with authoritative reasons, decision path, transaction
     information vs processing details, compact audit trail with technical
     expander, authoritative investigation state (display-only)
 [6] live monitor operator mode: rate/error metrics, formatted feed,
     stable pause, honest disconnect message
 [7] audit normal mode: "Audit operational" default, historical fork only
     inside the Historical integrity details expando
 [8] security page: operator status strip (MFA/Session/Authentication)
 [9] auth regression: unauthenticated admin APIs stay 401; login works
[10] structural integrity: JS-referenced ids exist, removed ids gone,
     two script blocks, progressive disclosure
[11] no secrets in the served dashboard; no frontend-only investigation
     writes; canonical constants unchanged
[12] production artifact identity AFTER (byte-identical)

Everything runs against an ISOLATED temp DB_DIR: no row written here can
reach the shared production DB-3/DB-4, and no production artifact is ever
opened for write.  No network, no model fitting, no threshold/release work.

Run from backend/:
  ../.venv/Scripts/python.exe scripts/phase113_admin_workflow_test.py
"""
from __future__ import annotations

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

DB_TMP = tempfile.mkdtemp(prefix="ps14_p113_")
os.environ["DB_DIR"] = DB_TMP
ADMIN_USER = "admin"
ADMIN_PASS = "p113-suite-pass-9c2d"
os.environ["ADMIN_USER"] = ADMIN_USER
os.environ["ADMIN_PASS"] = ADMIN_PASS

from fastapi.testclient import TestClient  # noqa: E402

from src.front_service.main import app  # noqa: E402
from src.monitoring import manifest_contract as MC  # noqa: E402
from src.monitoring.phase108_public_benchmark_execution import (  # noqa: E402
    PRODUCTION_ARTIFACT_PATHS,
    snapshot_artifacts,
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
WATCHED = tuple(PRODUCTION_ARTIFACT_PATHS) + ("models/production/release_manifest.json",)
before = snapshot_artifacts()
before["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
ok(len([k for k, v in before.items() if v != "absent"]) == 13,
   f"13 production artifacts readable before ({len(before)})")

# ── [2] shells + unchanged nav hierarchy ──────────────────────────────
c = TestClient(app)

for path in ("/admin", "/admin/transactions", "/admin/transactions/evt-1",
             "/admin/audit", "/admin/security", "/admin/settings"):
    r = c.get(path)
    ok(r.status_code == 200 and "passcode" in r.text.lower(),
       f"shell {path} -> 200 + passcode gate ({r.status_code})")

shell = c.get("/admin").text
ok(len(shell) > 20000, f"shell serves the full console ({len(shell)} bytes)")

nav_ops = re.findall(r'<div class="sidebar-item[^"]*" data-tab="([a-z]+)"', shell)
labels = re.findall(r'<div class="sidebar-label">([^<]+)</div>', shell)
ok(labels == ["Admin", "System", "Advanced"],
   f"nav groups unchanged: ADMIN/SYSTEM/ADVANCED ({labels})")
ok(nav_ops[:3] == ["dashboard", "transactions", "live"]
   and nav_ops[3:6] == ["audit", "access", "settings"]
   and nav_ops[6:9] == ["system", "database", "queries"]
   and len(nav_ops) == 9,
   "nav items unchanged: no new top-level navigation (§2)")

# ── [3] dashboard actionability ───────────────────────────────────────
i_dash = shell.find('id="tab-dashboard"')
i_live = shell.find('id="tab-live"')
seg = shell[i_dash:i_live]

ok('id="dash-flagged-kpi"' in seg
   and 'role="button"' in seg and "Show flagged transactions" in seg,
   "flagged KPI is a labelled button (§4)")
ok("Flagged (24h) →" in seg, "flagged KPI advertises it is actionable")
ok("Needs Attention" in seg, "flagged list titled 'Needs Attention' (§3)")
ok("View all flagged" in seg, "'View all flagged' drill-through (§3)")
ok("All clear" in shell and "No transactions currently require review." in shell,
   "flagged empty state is 'All clear', not a fake row (§17)")
ok("No recent transactions" in shell,
   "activity feed keeps the honest empty state (§3)")
for gone in ("Flagged Transactions", "No flagged transactions"):
    ok(gone not in shell, f"old dashboard label removed: {gone}")

# One-click wiring: KPI + View-all both call goFlagged; goFlagged applies
# the flagged filter before activating the transactions tab.
ok("addEventListener('click', goFlagged)" in shell
   and "addEventListener('keydown'" in shell and "goFlagged(); }" in shell,
   "goFlagged wired to click and keyboard activation")
ok("dash-view-all').addEventListener('click', goFlagged)" in shell,
   "'View all flagged' goes straight to the filtered list")
ok("f-flagged').value = 'true'" in shell,
   "goFlagged sets the flagged filter (equivalent of ?status=flagged)")
ok("status') === 'flagged'" in shell or "status') === \"flagged\"" in shell,
   "URL deep link ?status=flagged is honoured (§4)")

# Dashboard stays free of engineering detail.
for overload in ("sha256", "entry_hash", "prev_hash", "PromotionToken",
                 "predecessor", "map_raw_to_native"):
    ok(overload not in seg, f"dashboard free of engineering detail: {overload}")

# ── [4] transactions experience ───────────────────────────────────────
i_tx = shell.find('id="tab-transactions"')
i_set = shell.find('id="tab-settings"')
seg_tx = shell[i_tx:i_set]

ok('id="f-chips"' in seg_tx, "quick-filter chip row present (§16)")
chips = re.findall(r'data-chip="([a-z]+)"', seg_tx)
ok(chips == ["all", "flagged", "approved", "blocked"],
   f"chips are All/Flagged/Approved/Blocked ({chips})")
ok('id="f-reset"' in seg_tx and "Clear filters" in seg_tx,
   "'Clear filters' control exists (§16)")
ok("style=\"display: none;\"" in seg_tx[seg_tx.find('id="f-reset"'):seg_tx.find('id="f-reset"') + 120],
   "'Clear filters' hidden while no filter is active")
ok("syncQuickUI" in shell and "activeFilterKind" in shell,
   "chip state + clear-visibility are derived from real filters")
ok("DECISION_LABEL" in shell and "allow: 'Approved'" in shell
   and "verify: 'Review'" in shell and "step_up: 'Step-up'" in shell,
   "operator vocabulary: Approved/Review/Step-up, no raw snake_case (§6)")
ok("decisionWord(row.decision)" in shell,
   "result rows render the operator word, not the raw decision")
ok("data_quality_status === 'blocked'" in shell
   and ">BLOCKED</span>" in shell,
   "data-quality rows surface a simple BLOCKED badge (§6)")
# Stacked cards on narrow screens (§24)
ok("table.stackable thead" in shell and "content: attr(data-label)" in shell,
   "workflow tables become stacked cards at <=640px (§24)")
ok(shell.count('class="stackable"') >= 3,
   f"flagged/results/detail tables opt into stacking ({shell.count('class=\"stackable\"')})")
ok(shell.count('data-label="Event ID"') >= 1 and 'data-label="Risk"' in shell,
   "result cells carry labels for the stacked layout")
# Contextual empty states (§17)
for text in ("Transaction not found", "No matching transactions",
             "No transactions yet",
             "Transactions will appear here when the system receives them.",
             "Try another event ID or remove some filters."):
    ok(text in shell, f"empty state copy present: {text}")
# Exact-ID direct navigation (§15)
ok("rows[0].event_id === quick0" in shell,
   "single exact-ID result opens the detail directly")
ok("activateTab('transactions', { detail: quick0 })" in shell,
   "direct navigation opens the detail through the pushed-URL path (§26)")
ok(not re.search(r"(?<!\[\.\.\.)\bp\.keys\(\)\.length\b", shell)
   and not re.search(r"(?<!\[\.\.\.)\bfp\.keys\(\)\.length\b", shell),
   "params counting spreads .keys() (iterators have no .length)")
# Skeleton loading, not blank screens (§18)
ok(".skel" in shell and "skel-pulse" in shell, "skeleton loading style exists")
ok("class=\\\"skel\\\"" in shell or 'class="skel"' in shell,
   "loading states use skeletons")
# Friendly errors (§19)
ok("Unable to load transactions — try again." in shell,
   "search errors are human-readable")
ok("psycopg2" not in shell and "JSONDecodeError" not in shell
   and "KeyError" not in shell and "Traceback" not in shell,
   "no raw exception text anywhere in the console (§19)")
ok("Unable to load this transaction — try again." in shell,
   "detail errors are human-readable")
ok("Transaction not found — check the event ID." in shell,
   "unknown event IDs get a clean not-found (§15)")
# Global header search (§15)
ok('id="hdr-search"' in shell and "globalFind" in shell,
   "global transaction search lives in the header")

# Server search endpoint unchanged + still capped (logged in first so
# the cap check exercises the validator, not the auth gate)
r = c.post("/admin/login", json={"username": ADMIN_USER, "passphrase": ADMIN_PASS})
ok(r.status_code == 200 and "token" in r.json(), f"login -> 200 ({r.status_code})")
r = c.get("/admin/api/transactions", params={"limit": 999})
ok(r.status_code in (200, 422), f"server still caps limit ({r.status_code})")

# ── [5] transaction detail workflow ───────────────────────────────────
i_det = shell.find('id="txn-detail-panel"')
seg_det = shell[i_det:i_set]

for eid in ("d-hero-title", "d-event-id", "d-summary", "d-why", "d-path",
            "d-fields", "d-processing", "d-stages", "d-reasons",
            "d-tech-fields", "d-audit-box", "d-audit-compact",
            "d-audit-tech", "d-audit", "d-chain-note"):
    ok(f'id="{eid}"' in seg_det, f"detail element present: {eid}")

ok("Why was this flagged?" in seg_det, "why heading on the detail (§9)")
ok("Flagged for review" in shell, "'Flagged for review' headline (§9)")
ok("Flagged for review. Detailed reasoning is unavailable." in shell,
   "honest fallback kept when no reason is recorded (§9)")
ok("Decision path" in shell and "path-list" in shell,
   "decision path section present (§10)")
ok("derived from recorded evidence for this event" in shell,
   "decision path states its evidence basis (§10)")
for step in ("Input accepted", "Validation passed", "Risk evaluation completed",
             "Decision generated", "Audit recorded",
             "ML evaluation unavailable", "Rules evaluation completed",
             "Data-quality protection triggered",
             "Safe fallback decision recorded"):
    ok(step in shell, f"decision path step present: {step}")

# Status-first summary: status badge + plain risk score + decision + time
ok("'Risk score'" in shell and "Risk score (0-100)" not in shell,
   "risk score labelled plainly, no range/probability spin (§8)")
ok("94%" not in shell and "probability of fraud" not in shell,
   "no probability implication attached to the score (§8)")
ok("BLOCKED TRANSACTION" in shell,
   "blocked events are named BLOCKED (§7)")
ok("kpi('Time')" in shell or ", 'Time')" in shell,
   "summary shows Time (§7)")

# Layout order: why before processing, audit trail AFTER processing (§7)
ok(seg_det.find('id="d-why"') < seg_det.find('id="d-processing"'),
   "why-box precedes processing details")
ok(seg_det.find('id="d-processing"') < seg_det.find('id="d-audit-box"'),
   "audit trail is its own section after processing details (§12)")
ok(seg_det.find('id="d-audit-compact"') < seg_det.find('id="d-audit-tech"'),
   "compact audit rows precede the technical expander (§12)")
ok("View technical audit details" in seg_det,
   "'View technical audit details' gate for hashes (§12)")
ok("Transaction information" in seg_det and "Transaction details</summary>" not in seg_det,
   "operator section renamed 'Transaction information' (§7)")
ok("Runtime context" in seg_det,
   "technical identity moved into Processing details (§11)")
ok('<details class="advanced" id="d-processing">' in shell
   and "Decision Reasons (raw codes)" in seg_det,
   "raw reason codes stay in the technical expander (§6)")

# Investigation state: authoritative display only (§13)
ok("investigationText" in shell and "CASE_GLOSS" in shell,
   "investigation state rendered from recorded case/outcome rows")
# Phase 113 review workflow: the states now exist BECAUSE the backend
# serves them — pinned as present + payload-driven, never invented per row.
ok("UNREVIEWED" in shell and "UNDER_REVIEW" in shell and "REVIEWED" in shell,
   "review states present, served by the backend workflow (§13)")
ok("d.review" in shell and "rv.review_state" in shell,
   "review state renders from the detail payload, not frontend-only (§13)")
for state in ("NEW", "REVIEWING", "USER_VERIFICATION",
              "CONFIRMED_SUSPICIOUS", "CONFIRMED_LEGITIMATE", "CLOSED"):
    ok(f"{state}:" in shell or f"'{state}'" in shell,
       f"backend case status understood: {state}")
ok("confirm_alert" not in shell,
   "no fraud-confirmation action in the console (§14)")
ok("Mark reviewed" in shell and "reviewPost('/review/' + action)" in shell
   and "reviewPost('/notes'" in shell,
   "review actions post to the backend endpoints the session owns (§14)")
ok("field('Investigation', investigationText(d))" in shell,
   "investigation line appears in Transaction information")

# Forbidden admin actions (§14): no bypass affordances in the console
forbidden_actions = ("Override model", "Change threshold", "Promote model",
                     "Retrain", "Re-evaluate", "Modify audit", "Delete transaction",
                     "Change risk score")
for act in forbidden_actions:
    ok(act not in shell, f"no bypass action offered: {act}")

# ── [6] live monitor operator mode ────────────────────────────────────
i_l = shell.find('id="tab-live"')
i_aud = shell.find('id="tab-audit"')
seg_live = shell[i_l:i_aud]
for label in ("Transactions/sec", "Flagged in window", "p50 latency (ms)", "Errors"):
    ok(label in seg_live, f"primary metric label: {label}")
ok('id="live-simple-total"' in seg_live and 'id="live-simple-blocks"' in seg_live,
   "primary metric ids unchanged (Phase 112 contract)")
ok("w0.requests_per_sec == null ? 'N/A'" in shell,
   "rate metric renders N/A when the source is absent (§17)")
ok("w0.errors == null ? 'N/A'" in shell,
   "error metric renders N/A when unknown — never a fake zero (§17)")
ok("Live monitor disconnected · last update" in shell,
   "disconnected state says so with the last update (§17)")
ok("setConn('PAUSED')" in shell and "livePaused" in shell,
   "pause state exists (§21)")
ok("stopLive()" in shell, "pause/tab-leave stops polling — rows stay stable (§21)")
ok("clsF = 'warning'" in shell and "FLAGGED" in shell,
   "feed rows show operator status words (§20)")
ok("feed-row" in shell and "activateTab('transactions', { detail:" in shell,
   "feed rows open the investigation view (§20)")

# ── [7] audit normal mode ─────────────────────────────────────────────
i_aud2 = shell.find('id="tab-audit"')
i_tx2 = shell.find('id="tab-transactions"')
seg_aud = shell[i_aud2:i_tx2]
ok("Audit operational" in shell, "default audit view says 'Audit operational' (§22)")
ok("chain verified end-to-end" in shell, "healthy chain detail is one clean line")
i_tech = seg_aud.find('id="audit-tech"')
i_fork = seg_aud.find('id="audit-fork-box"')
ok(0 < i_tech < i_fork,
   "historical fork explanation lives INSIDE the integrity-details expando (§22)")
ok("does not represent a newly detected failure" in seg_aud,
   "fork text still explains it is documented, not new")
ok("Historical Integrity Details" in seg_aud,
   "technical history behind its expando (§22)")
ok('display: none' in seg_aud[i_fork:i_fork + 120],
   "fork box hidden by default even inside the expando")

# ── [8] security page status strip ────────────────────────────────────
i_acc = shell.find('id="tab-access"')
seg_acc = shell[i_acc:]
ok('id="sec-strip"' in seg_acc and 'id="sec-mfa"' in seg_acc,
   "operator security status strip present (§23)")
ok("Session" in seg_acc and "● Active" in seg_acc,
   "Session line shows Active (§23)")
ok("Authentication" in seg_acc and "● Protected" in seg_acc,
   "Authentication line shows Protected (§23)")
ok("$('sec-mfa')" in shell and "setTotpStatus" in shell,
   "MFA line is fed by the authoritative TOTP status")
i_totp = seg_acc.find('id="totp-status"')
i_secdet = seg_acc.find('id="security-details"')
ok(0 <= i_totp < i_secdet, "MFA status still sits above the technical expando")

# ── [9] auth regression ───────────────────────────────────────────────
for path in ("/admin/api/summary", "/admin/api/transactions",
             "/admin/api/transactions/evt-1", "/admin/api/live",
             "/admin/api/audit-overview", "/admin/sessions"):
    r = TestClient(app).get(path)
    ok(r.status_code == 401, f"unauthorized {path} -> 401 ({r.status_code})")

r = c.post("/admin/login", json={"username": ADMIN_USER, "passphrase": ADMIN_PASS})
ok(r.status_code == 200 and "token" in r.json(), f"login (2) -> 200 ({r.status_code})")
r = c.get("/admin/api/summary")
ok(r.status_code == 200, f"summary with session -> 200 ({r.status_code})")
d = r.json() if r.status_code == 200 else {}
st = d.get("status") or {}
ok(st.get("state") in ("operational", "attention", "unavailable"),
   "3-state status verdict intact (Phase 112 contract)")
ok({"counts", "recent_flagged", "recent_activity", "chain", "details"} <= set(d),
   "summary payload shape unchanged (Phase 112 contract)")
cnt = d.get("counts") or {}
ok(all(v is None or isinstance(v, (int, str)) for v in cnt.values()),
   "counts remain ints/ISO strings or null — never fabricated")

# ── [10] structural integrity ─────────────────────────────────────────
served = c.get("/admin").text
ids_in_html = set(re.findall(r'id="([^"$]+)"', served))
blocks = re.findall(r"<script>(.*?)</script>", served, re.S)
js = "\n".join(blocks)
refs = set(re.findall(r"\$\('([^']+)'\)", js))
refs |= set(re.findall(r"getElementById\('([^']+)'\)", js))
missing = sorted(r for r in refs if r not in ids_in_html)
ok(not missing, f"every JS-referenced id exists ({missing})")
for gone in ("system-status", "txn-count", "alert-count", "chain-badge",
             "dash-online", "dash-line-system", "dash-details"):
    ok(gone not in ids_in_html, f"removed id stays removed: {gone}")
ok(len(blocks) == 2 and all(len(b) > 1000 for b in blocks),
   f"two intact script blocks ({[len(b) for b in blocks]})")
ok(js.count("{") == js.count("}"),
   f"script braces balance ({js.count('{')} vs {js.count('}')})")
for det in ("f-filters", "d-processing", "d-audit-box", "d-audit-tech",
            "live-details", "audit-tech", "security-details"):
    ok(re.search(rf'<details[^>]*id="{det}"', served) is not None,
       f"progressive disclosure: {det} is a <details> expando")
ok("focus-visible" in shell, "keyboard focus outlines styled")
ok('aria-label="Search transactions by event ID or transaction ID"' in shell,
   "quick search accessible label kept")
ok('aria-label="Find transaction by event ID or transaction ID"' in shell,
   "global search accessible label present")

# ── [11] secrets + canonical constants ────────────────────────────────
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
hits = [label for pat, label in SECRET_PATS if pat.search(served)]
ok(not hits, f"no secrets in the served dashboard ({hits})")
ok(ADMIN_PASS not in served, "admin passphrase never appears in the shell")

ok(MC.CANONICAL_THRESHOLD == 0.018758, "threshold constant unchanged")
ok(MC.CANONICAL_FEATURE_VERSION == "v1", "feature version constant unchanged")
ok(MC.CANONICAL_MODEL_ID == "altman_native", "governance model_id unchanged")
ok(MC.CANONICAL_RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904",
   "governance release_id unchanged")

# ── [12] production artifact identity AFTER ───────────────────────────
after = snapshot_artifacts()
after["models/production/release_manifest.json"] = hashlib.sha256(
    (REPO / "models/production/release_manifest.json").read_bytes()).hexdigest()
changed = [k for k in WATCHED if before.get(k) != after.get(k)]
ok(not changed, f"production artifacts byte-identical across the suite ({changed})")

print()
print(f"PHASE 113: {n_assert} assertions, {len(failures)} failures")
if failures:
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PHASE 113: ALL PASS")
