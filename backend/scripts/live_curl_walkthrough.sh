#!/usr/bin/env bash
# Live curl walkthrough for PS-14 (the "allow / step-up / verify" demo).
#
# Boots the three core services - Identity (:8001), Privacy Layer (:8002) and
# Risk Engine (:8003) - on a dedicated, gitignored DB (`db/walkthrough`), then:
#   1. registers a throwaway user,
#   2. seeds six normal transactions into the account's history,
#   3. AGES the account: backdates DB-2 rows so the time-derived features
#      (tenure, freq_24h, days_since) reflect a real ~60-day history (a
#      demo-harness convenience - in production with streaming events the
#      timestamps are real), and
#   4. evaluates three scenarios and prints the ACTUAL decisions:
#        A  normal purchase, known device          -> allow
#        B  new device + new city + new merchant  -> step_up
#        C  account-takeover attack                -> verify
#
# The HTTP flow is plain curl; the venv python is used only for JSON
# extraction and the aging backdate. Windows: run from Git Bash.
#
# Ports 8001-8003 must be free (stop the dev stack first). Services started
# by this script are stopped again on exit (trap).
#
# Usage:
#   bash scripts/live_curl_walkthrough.sh
#
# Env:
#   PS14_INTERNAL_TOKEN   override the dev internal token
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

if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  PY="$ROOT/.venv/Scripts/python.exe"
else
  PY="$ROOT/.venv/bin/python"
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

echo "== PS-14 live curl walkthrough =="
for p in 8001 8002 8003; do
  if ! port_free "$p"; then
    echo "port :$p is already in use - stop the dev stack first (the walkthrough"
    echo "needs its own clean state on db/walkthrough)." >&2
    exit 1
  fi
done

export DB_DIR="$WT_DIR"
rm -rf "$WT_DIR"
mkdir -p "$WT_DIR"

# ---------------------------------------------------------------------------
# 1. boot identity, privacy, risk
# ---------------------------------------------------------------------------

start_service identity 8001 src.identity_service.main:app
start_service privacy  8002 src.privacy_layer.main:app
start_service risk     8003 src.risk_engine.main:app

wait_health 8001 15 identity
wait_health 8002 15 privacy
wait_health 8003 60 risk   # loads the model artifacts at startup

# ---------------------------------------------------------------------------
# 2. register a throwaway user
# ---------------------------------------------------------------------------

EMAIL="wt-$(date +%s)@example.com"
PHONE="+91$(date +%s)"
echo
echo "== registering $EMAIL =="
REG="$(curl -s -X POST http://127.0.0.1:8001/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"full_name\":\"Walkthrough User\",\"email\":\"$EMAIL\",\"phone\":\"$PHONE\",\"password\":\"wt-pass-1234\"}")"
FRAUD_ID="$(echo "$REG" | extract "d['fraud_id']")"
echo "fraud_id=$FRAUD_ID (pseudonymous - no identity in the feature store)"

# ---------------------------------------------------------------------------
# 3. seed six normal transactions (flat ~$100, noon, own phone)
# ---------------------------------------------------------------------------

echo
echo "== seeding 6 normal transactions (spread over ~7 weeks of 'ts') =="
seed() { # $1=event_id $2=amount $3=ts
  curl -s -X POST http://127.0.0.1:8002/internal/ingest-transaction \
    -H "$AUTH" -H "Content-Type: application/json" \
    -d "{\"event_id\":\"$1\",\"fraud_id\":\"$FRAUD_ID\",\"ts\":\"$3\",\"amount\":$2,\"hour_of_day\":12,\"device_id\":\"phone-android-1\",\"location_id\":\"L-HOME\",\"recipient_id\":\"R-GROCERY\",\"failed_auth_count_24h\":0}" \
    > /dev/null
  # normal history is "allowed": commit it into the behavioral baseline
  curl -s -X POST http://127.0.0.1:8002/internal/commit-baseline \
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
  feat="$(curl -s -X POST http://127.0.0.1:8002/internal/ingest-transaction \
    -H "$AUTH" -H "Content-Type: application/json" \
    -d "{\"event_id\":\"$2\",\"fraud_id\":\"$FRAUD_ID\",\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"amount\":$3,\"hour_of_day\":$4,\"device_id\":\"$5\",\"location_id\":\"$6\",\"recipient_id\":\"$7\",\"failed_auth_count_24h\":$8}")"
  # keep the section-16 vector as stored (for the derived-vector printout)
  echo "$feat" > "$WT_DIR/feat-$2.json"
  curl -s -X POST http://127.0.0.1:8003/internal/evaluate \
    -H "$AUTH" -H "Content-Type: application/json" \
    -d "{\"event_id\":\"$2\",\"fraud_id\":\"$FRAUD_ID\",\"features\":$feat}"
}

EVAL_A="$(run_scenario "A normal"        wt-a-0001 120 12 phone-android-1 L-HOME   R-GROCERY 0)"
EVAL_B="$(run_scenario "B new device+city" wt-b-0001 120 12 laptop-new-1    L-CITY   R-ONLINE 0)"
EVAL_C="$(run_scenario "C attack"        wt-c-0001 4500 3 attacker-phone-1 L-X      R-X       5)"

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

echo
echo "walkthrough state kept in $WT_DIR/ (gitignored) for inspection"
echo "DONE"
