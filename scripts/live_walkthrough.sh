#!/usr/bin/env bash
# Live register-to-audit walkthrough across ALL FIVE PS-14 services
# (the "AI recommends, verification decides" demo, section 1/8/13/14).
#
# Boots Identity, Privacy Layer, Risk Engine, Verification and Audit on a
# dedicated, gitignored DB (`db/walkthrough`), then:
#   1. registers a throwaway user (Identity) and logs in for the JWT,
#   2. seeds six normal transactions into the account's history (Privacy)
#      and AGES the account in DB-2 so time features reflect ~60 days,
#   3. evaluates three scenarios through the Risk Engine and prints the
#      ACTUAL decisions: A normal -> allow, B new device+city -> step_up,
#      C account-takeover attack -> verify,
#   4. runs the verification loop (Verification): lists the high-risk
#      alert, disputes it ("this wasn't me") -> case id + recovery flow,
#   5. prints the HASH-VERIFIED audit trail (Audit): the pseudonymous
#      decision trail for this user, the full-chain compliance export
#      re-verified INDEPENDENTLY by scripts/verify_export.py (HMAC
#      signature + per-entry hash linkage recomputed from genesis), and
#      the Audit Service's own chain-integrity verdict.
#
# The HTTP flow is plain curl; the venv python is used only for JSON
# extraction, the DB-2 aging backdate, and the export re-verification.
# Windows: run from Git Bash.
#
# Default ports are 8001-8005 and must be free (stop the dev stack first);
# set PS14_*_PORT to run alongside the dev stack on other ports - the flow
# is self-contained (direct curl + the shared local DB), so alternate ports
# work without touching the running services.
#
# Usage:
#   bash scripts/live_walkthrough.sh
#
# Env:
#   PS14_INTERNAL_TOKEN   override the dev internal token
#   PS14_IDENTITY_PORT / PS14_PRIVACY_PORT / PS14_RISK_PORT /
#   PS14_VERIFY_PORT / PS14_AUDIT_PORT   service ports (default 8001..8005)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Token the SERVICES accept: they auto-load .env, so prefer its
# INTERNAL_TOKEN; PS14_INTERNAL_TOKEN (explicit override) wins, and the dev
# constant is the last fallback (CI, no .env).
if [ -z "${PS14_INTERNAL_TOKEN:-}" ] && [ -f "$ROOT/.env" ]; then
  _env_tok="$(grep -E '^INTERNAL_TOKEN=' "$ROOT/.env" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
  [ -n "$_env_tok" ] && PS14_INTERNAL_TOKEN="$_env_tok"
fi

TOKEN="${PS14_INTERNAL_TOKEN:-ps14-dev-internal-token-change-me}"
AUTH="X-Internal-Token: $TOKEN"
WT_DIR="db/walkthrough"

P_IDENTITY="${PS14_IDENTITY_PORT:-8001}"
P_PRIVACY="${PS14_PRIVACY_PORT:-8002}"
P_RISK="${PS14_RISK_PORT:-8003}"
P_VERIFY="${PS14_VERIFY_PORT:-8004}"
P_AUDIT="${PS14_AUDIT_PORT:-8005}"
PORTS=("$P_IDENTITY" "$P_PRIVACY" "$P_RISK" "$P_VERIFY" "$P_AUDIT")

if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  PY="$ROOT/.venv/Scripts/python.exe"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

is_windows() { case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) return 0;; *) return 1;; esac; }

# extract a python expression on the parsed JSON from stdin
extract() { "$PY" -c "import sys, json; d = json.load(sys.stdin); print($1)"; }

port_free() {
  if is_windows; then
    netstat -ano 2>/dev/null | grep -E ":$1 " | grep -q LISTENING && return 1 || return 0
  else
    if command -v lsof >/dev/null 2>&1; then
      lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1 && return 1 || return 0
    else
      return 0
    fi
  fi
}

wait_health() {
  local port="$1" tries="$2" name="$3" i
  for i in $(seq 1 "$tries"); do
    if curl -s -m 2 "http://127.0.0.1:$port/health" | grep -q '"ok"'; then
      echo "[$name] up on :$port"
      return 0
    fi
    sleep 1
  done
  echo "[$name] FAILED to become healthy on :$port" >&2
  exit 1
}

STOP_PIDS=()
cleanup() {
  if [ ${#STOP_PIDS[@]} -gt 0 ]; then
    echo
    echo "== stopping services started by this walkthrough =="
    for pid in "${STOP_PIDS[@]}"; do
      if is_windows; then
        taskkill //F //PID "$pid" >/dev/null 2>&1 && echo "  stopped pid $pid" || true
      else
        kill "$pid" >/dev/null 2>&1 && echo "  stopped pid $pid" || true
      fi
    done
  fi
}
trap cleanup EXIT

start_service() {
  local name="$1" port="$2" mod="$3" pid
  if is_windows; then
    # No -RedirectStandardOutput here: the PowerShell 5.1 redirect quirk makes
    # the parent powershell hang. Logs go to the hidden console; health checks
    # are the signal.
    # Git Bash paths (/c/...) are invalid for PowerShell - convert to Windows.
    local py_win root_win
    py_win="$(cygpath -w "$PY" 2>/dev/null || echo "$PY")"
    root_win="$(cygpath -w "$ROOT" 2>/dev/null || echo "$ROOT")"
    pid="$(powershell -NoProfile -Command "(Start-Process -FilePath '$py_win' -ArgumentList '-m','uvicorn','$mod','--port','$port' -WorkingDirectory '$root_win' -WindowStyle Hidden -PassThru).Id" | tr -d '\r')"
  else
    nohup "$PY" -m uvicorn "$mod" --port "$port" >"$WT_DIR/$name.log" 2>&1 &
    pid=$!
  fi
  echo "[$name] started (pid $pid) on :$port"
  STOP_PIDS+=("$pid")
}

# ---------------------------------------------------------------------------
# 0. preflight: ports free, dedicated walkthrough DB
# ---------------------------------------------------------------------------

echo "== PS-14 live register-to-audit walkthrough (all five services) =="
for p in "${PORTS[@]}"; do
  if ! port_free "$p"; then
    echo "port :$p is already in use - stop the dev stack first or set the" >&2
    echo "PS14_*_PORT env vars to free ports (the walkthrough needs its own" >&2
    echo "clean state on db/walkthrough)." >&2
    exit 1
  fi
done

export DB_DIR="$WT_DIR"
# The walkthrough is a dev harness: its children must not run the production
# startup gate (on CI there is no .env, so services would default to
# production mode and sys.exit(1) on the default CORS config before serving).
# Callers who DO want the gate can set PS14_MODE explicitly.
export PS14_MODE="${PS14_MODE:-development}"

# Shared secrets for the children (CI has no .env to coordinate them).
# Random per-process secrets would make every internal call 403 and the
# export signature unverifiable. Missing values are generated once here.
GEN="$(env -u INTERNAL_TOKEN "$PY" -c "import secrets; print(secrets.token_hex(32))")"
export INTERNAL_TOKEN="${PS14_INTERNAL_TOKEN:-$(grep -E '^INTERNAL_TOKEN=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' | tr -d '"' || echo "$GEN")}"
[ -n "${PS14_INTERNAL_TOKEN:-}" ] && export INTERNAL_TOKEN="$PS14_INTERNAL_TOKEN"
export EXPORT_SIGNING_KEY="${EXPORT_SIGNING_KEY:-$(grep -E '^EXPORT_SIGNING_KEY=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' | tr -d '"' || echo "$GEN")}"
export COMPLIANCE_TOKEN="${COMPLIANCE_TOKEN:-$(grep -E '^COMPLIANCE_TOKEN=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' | tr -d '"' || echo "$GEN")}"
export JWT_SECRET="${JWT_SECRET:-$(grep -E '^JWT_SECRET=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' | tr -d '"' || echo "$GEN")}"
export PII_ENCRYPTION_KEY="${PII_ENCRYPTION_KEY:-$(grep -E '^PII_ENCRYPTION_KEY=' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' | tr -d '"' || echo "$GEN")}"

rm -rf "$WT_DIR"
mkdir -p "$WT_DIR"

# ---------------------------------------------------------------------------
# 1. boot identity, privacy, risk, verification, audit
# ---------------------------------------------------------------------------

start_service identity     "$P_IDENTITY" src.identity_service.main:app
start_service privacy      "$P_PRIVACY" src.privacy_layer.main:app
start_service risk         "$P_RISK" src.risk_engine.main:app
start_service verification "$P_VERIFY" src.verification_service.main:app
start_service audit        "$P_AUDIT" src.audit_service.main:app

wait_health "$P_IDENTITY" 15 identity
wait_health "$P_PRIVACY" 15 privacy
wait_health "$P_RISK" 60 risk   # loads the model artifacts at startup
wait_health "$P_VERIFY" 15 verification
wait_health "$P_AUDIT" 15 audit

# ---------------------------------------------------------------------------
# 2. register a throwaway user + login for the verification JWT
# ---------------------------------------------------------------------------

EMAIL="wt-$(date +%s)@example.com"
PHONE="+91$(date +%s)"
echo
echo "== registering $EMAIL =="
REG="$(curl -s -X POST "http://127.0.0.1:$P_IDENTITY/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"full_name\":\"Walkthrough User\",\"email\":\"$EMAIL\",\"phone\":\"$PHONE\",\"password\":\"wt-pass-1234\"}")"
FRAUD_ID="$(echo "$REG" | extract "d['fraud_id']")"
echo "fraud_id=$FRAUD_ID (pseudonymous - no identity in the feature store)"

echo "== logging in for the verification JWT =="
LOGIN="$(curl -s -X POST "http://127.0.0.1:$P_IDENTITY/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"wt-pass-1234\"}")"
JWT="$(echo "$LOGIN" | extract "d['access_token']")"
BEARER="Authorization: Bearer $JWT"
echo "access_token issued (JWT carries only the fraud_id claim)"

# ---------------------------------------------------------------------------
# 3. seed six normal transactions (flat ~$100, noon, own phone)
# ---------------------------------------------------------------------------

echo
echo "== seeding 6 normal transactions (spread over ~7 weeks of 'ts') =="
seed() { # $1=event_id $2=amount $3=ts
  curl -s -X POST "http://127.0.0.1:$P_PRIVACY/internal/ingest-transaction" \
    -H "$AUTH" -H "Content-Type: application/json" \
    -d "{\"event_id\":\"$1\",\"fraud_id\":\"$FRAUD_ID\",\"ts\":\"$3\",\"amount\":$2,\"hour_of_day\":12,\"device_id\":\"phone-android-1\",\"location_id\":\"L-HOME\",\"recipient_id\":\"R-GROCERY\",\"failed_auth_count_24h\":0}" \
    > /dev/null
  # normal history is "allowed": commit it into the behavioral baseline
  curl -s -X POST "http://127.0.0.1:$P_PRIVACY/internal/commit-baseline" \
    -H "$AUTH" -H "Content-Type: application/json" \
    -d "{\"event_id\":\"$1\",\"fraud_id\":\"$FRAUD_ID\"}" > /dev/null
  echo "  seeded $1 (amount \$$2)"
}
seed wt-seed-01 100 2025-06-01T12:00:00
seed wt-seed-02 102 2025-06-10T12:00:00
seed wt-seed-03 105 2025-06-20T12:00:00
seed wt-seed-04 108 2025-07-01T12:00:00
seed wt-seed-05 103 2025-07-10T12:00:00
seed wt-seed-06 107 2025-07-20T12:00:00

# ---------------------------------------------------------------------------
# 4. age the account in DB-2 (demo-harness: time features derive from
#    ingest-time rows, so simulate a real history by backdating created_at)
# ---------------------------------------------------------------------------

echo
echo "== aging the account to ~60 days in DB-2 =="
"$PY" - "$WT_DIR" "$FRAUD_ID" <<'EOF'
import os, sqlite3, sys
from datetime import datetime, timedelta, timezone

wt_dir, fid = sys.argv[1], sys.argv[2]
con = sqlite3.connect(os.path.join(wt_dir, "features.db"))
now = datetime.now(timezone.utc).replace(tzinfo=None)
rows = con.execute(
    "SELECT event_id FROM transaction_features WHERE fraud_id = ? ORDER BY created_at",
    (fid,),
).fetchall()
n = len(rows)
for i, (eid,) in enumerate(rows):
    # spread the seeded history from 8 to 55 days ago (nothing within 24h)
    age = 8 + (55 - 8) * i / max(n - 1, 1)
    con.execute(
        "UPDATE transaction_features SET created_at = ? WHERE event_id = ?",
        ((now - timedelta(days=age)).isoformat(), eid),
    )
con.execute(
    "UPDATE fraud_profiles SET created_at = ? WHERE fraud_id = ?",
    ((now - timedelta(days=60)).isoformat(), fid),
)
con.commit()
con.close()
print(f"  backdated {n} feature rows (8-55 days old) + profile birth 60d ago")
EOF

# ---------------------------------------------------------------------------
# 5. evaluate the three scenarios -> actual allow / step-up / verify
# ---------------------------------------------------------------------------

echo
echo "== evaluating scenarios =="
run_scenario() { # $1=tag $2=event_id $3=amount $4=hour $5=device $6=location $7=recipient $8=auth
  local feat
  feat="$(curl -s -X POST "http://127.0.0.1:$P_PRIVACY/internal/ingest-transaction" \
    -H "$AUTH" -H "Content-Type: application/json" \
    -d "{\"event_id\":\"$2\",\"fraud_id\":\"$FRAUD_ID\",\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"amount\":$3,\"hour_of_day\":$4,\"device_id\":\"$5\",\"location_id\":\"$6\",\"recipient_id\":\"$7\",\"failed_auth_count_24h\":$8}")"
  # keep the section-16 vector as stored (for the derived-vector printout)
  echo "$feat" > "$WT_DIR/feat-$2.json"
  curl -s -X POST "http://127.0.0.1:$P_RISK/internal/evaluate" \
    -H "$AUTH" -H "Content-Type: application/json" \
    -d "{\"event_id\":\"$2\",\"fraud_id\":\"$FRAUD_ID\",\"features\":$feat}"
}

EVAL_A="$(run_scenario "A normal"         wt-a-0001 120 12 phone-android-1 L-HOME   R-GROCERY 0)"
EVAL_B="$(run_scenario "B new device+city" wt-b-0001 120 12 laptop-new-1    L-CITY   R-ONLINE 0)"
EVAL_C="$(run_scenario "C attack"         wt-c-0001 4500 3 attacker-phone-1 L-X      R-X       5)"

echo
echo "== allow / step-up / verify decisions =="
printf "%-22s %-10s %-8s %-7s %s\n" scenario decision band score reasons
for pair in "A:${EVAL_A}" "B:${EVAL_B}" "C:${EVAL_C}"; do
  tag="${pair%%:*}"
  body="${pair#*:}"
  printf "%-22s %-10s %-8s %-7s %s\n" \
    "$tag" \
    "$(echo "$body" | extract "d['decision'].upper()")" \
    "$(echo "$body" | extract "d['risk_band']")" \
    "$(echo "$body" | extract "d['risk_score']")" \
    "$(echo "$body" | extract "','.join(d['reason_codes'])" | sed 's/^$/none/')"
done

echo
echo "derived vector for scenario A as stored by the Privacy Layer (§16 shape,"
echo "no raw amount, no device id, no exact geo - only purpose-limited features):"
for key in amount_ratio txn_freq_last_24h txn_time_unusual new_device_flag \
           known_device_count account_tenure_days hour_of_day; do
  printf "  %-24s %s\n" "$key" "$(cat "$WT_DIR/feat-wt-a-0001.json" | extract "d['$key']")"
done

# ---------------------------------------------------------------------------
# 6. verification loop: the alert is listed, then disputed -> recovery flow
# ---------------------------------------------------------------------------

echo
echo "== verification loop (Verification :$P_VERIFY) =="
ALERTS="$(curl -s "http://127.0.0.1:$P_VERIFY/alerts" -H "$BEARER")"
N_ALERTS="$(echo "$ALERTS" | extract "len(d['alerts'])")"
echo "open high-risk alerts: $N_ALERTS"
echo "$ALERTS" > "$WT_DIR/alerts.json"
"$PY" - "$WT_DIR/alerts.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for a in d["alerts"]:
    print(f"  - {a['event_id']}  score={a['risk_score']} band={a['risk_band']} "
          f"reasons={','.join(a['reason_codes']) or 'none'}")
EOF

ATTACK_EVENT="$(echo "$ALERTS" | extract "[a['event_id'] for a in d['alerts'] if a['event_id'] == 'wt-c-0001'][0]" 2>/dev/null || echo "")"
if [ -z "$ATTACK_EVENT" ]; then
  echo "FAILED: scenario C (wt-c-0001) is not in the alert list" >&2
  exit 1
fi

echo
echo "== disputing the attack alert (this wasn't me) =="
RESOLVE="$(curl -s -X POST "http://127.0.0.1:$P_VERIFY/alerts/$ATTACK_EVENT/confirm" \
  -H "$BEARER" -H "Content-Type: application/json" \
  -d '{"outcome":"this_wasnt_me"}')"
echo "outcome       : $(echo "$RESOLVE" | extract "d['outcome']")"
echo "case_id       : $(echo "$RESOLVE" | extract "d['case_id']")"
echo "message       : $(echo "$RESOLVE" | extract "d['message']")"
echo "baseline kept clean: $(echo "$RESOLVE" | extract "not d['baseline_updated']") (blocked events never enter the profile)"
echo "recovery flow :"
echo "$RESOLVE" > "$WT_DIR/resolve.json"
"$PY" - "$WT_DIR/resolve.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for s in d["recovery_steps"]:
    print(f"  - {s}")
EOF

echo
echo "open alerts after resolution: $(curl -s "http://127.0.0.1:$P_VERIFY/alerts" -H "$BEARER" | extract "len(d['alerts'])") (expected 0)"

# ---------------------------------------------------------------------------
# 7. hash-verified audit trail (Audit :$P_AUDIT)
# ---------------------------------------------------------------------------

echo
echo "== hash-verified audit trail =="

echo "== 7a. pseudonymous decision trail for this user =="
TRAIL="$(curl -s "http://127.0.0.1:$P_AUDIT/audit/events?fraud_id=$FRAUD_ID&limit=100" -H "$AUTH")"
echo "$TRAIL" > "$WT_DIR/trail.json"
"$PY" - "$WT_DIR/trail.json" <<'EOF'
import json, sys
trail = json.load(open(sys.argv[1], encoding="utf-8"))
events = sorted(trail["events"], key=lambda e: e["seq"])
print(f"{'seq':>4}  {'event_type':<24}  {'entry_hash':<14}  payload")
for ev in events:
    p = ev.get("payload", {})
    if ev["event_type"] == "score_generated":
        summary = (f"score={p.get('risk_score')} band={p.get('risk_band')} "
                   f"reasons={','.join(p.get('reason_codes') or []) or 'none'}")
    elif ev["event_type"] == "verification_resolved":
        summary = f"outcome={p.get('outcome')} case={p.get('case_id')} event={p.get('event_id')}"
    elif ev["event_type"] == "fraud_id_resolved":
        summary = f"actor={p.get('actor')} reason={p.get('reason')}"
    else:
        summary = json.dumps(p, sort_keys=True)
    print(f"{ev['seq']:>4}  {ev['event_type']:<24}  {str(ev.get('entry_hash'))[:14]:<14}  {summary}")
by_type = {}
for ev in events:
    by_type[ev["event_type"]] = by_type.get(ev["event_type"], 0) + 1
print("\nper type: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
EOF

echo
echo "== 7b. full-chain compliance export, re-verified INDEPENDENTLY =="
curl -s "http://127.0.0.1:$P_AUDIT/audit/export" -H "$AUTH" > "$WT_DIR/export.json"
echo "  export: $WT_DIR/export.json ($(wc -c < "$WT_DIR/export.json") bytes)"
"$PY" scripts/verify_export.py "$WT_DIR/export.json"

echo
echo "== 7c. Audit Service's own chain-integrity verdict =="
INTEG="$(curl -s "http://127.0.0.1:$P_AUDIT/audit/integrity" -H "$AUTH")"
echo "$INTEG" > "$WT_DIR/integrity.json"
"$PY" - "$WT_DIR/integrity.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
if d["ok"]:
    print(f"  chain: OK - {d['n_entries']} entries, genesis {d['genesis_hash'][:14]}...")
else:
    print(f"  chain: BROKEN at seq {d['first_bad_seq']}")
EOF

echo
echo "walkthrough state kept in $WT_DIR/ (gitignored) for inspection"
echo "DONE"
