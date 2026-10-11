# PS-14 Demo Runbook (local synthetic demo)

How to start the demo stack, run the synthetic-transaction demo from the web
page, and shut everything down again. Everything here is a **local synthetic
demo**: it exercises the real software path, not real-world fraud detection,
institutional validation, or production readiness.

- The documented demo path is Privacy Layer (`:8002`) → Risk Engine (`:8003`).
- The inference service (`:8006`) and Redis are **not** part of this demo path
  and are not exercised by the landing-page demo.

## Prerequisites

- Python 3.12 (a `.venv` at the repo root is used when it exists; otherwise the
  system interpreter is used).
- Dependencies from `backend/requirements.txt` installed into that interpreter.
- Model artifacts present under `models/` (the Risk Engine loads the Altman
  native ensemble and the drift baseline at startup; if they are missing the
  service starts degraded and the demo will say so through readiness).
- A `.env` file at the repository root (gitignored). The demo needs these keys
  **set to your own local values** — this runbook never prints or commits them:

  | Key | Used for |
  | --- | --- |
  | `JWT_SECRET` | identity token signing; also derives the PII encryption key |
  | `PII_ENCRYPTION_KEY` | PII encryption at rest |
  | `EXPORT_SIGNING_KEY` | export signatures |
  | `INTERNAL_TOKEN` | service-to-service auth on every `/internal/*` endpoint |
  | `COMPLIANCE_TOKEN` | compliance/read endpoints |
  | `ADMIN_PASS` | bootstrap passphrase for the admin area |

  If a key is unset, `settings` generates an ephemeral dev value for the
  internal token. That is fine for a throwaway run but it means every service
  must be started from the *same* environment — otherwise the Privacy Layer and
  the Risk Engine disagree about the internal token and every call is rejected
  with `401`.

## Starting the stack

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .freebuff/start_stack.ps1
```

The script loads `.env`, strips Postgres/Supabase variables (this launcher is
the LOCAL SQLite preview), sets the per-service URLs, stops any stale listeners
on the demo ports, and starts one uvicorn process per service:

| Service | Port | Module |
| --- | --- | --- |
| Front page (landing + demo UI) | 8000 | `src.front_service.main:app` |
| Identity | 8001 | `src.identity_service.main:app` |
| Privacy Layer | 8002 | `src.privacy_layer.main:app` |
| Risk Engine | 8003 | `src.risk_engine.main:app` |
| Verification | 8004 | `src.verification_service.main:app` |
| Audit | 8005 | `src.audit_service.main:app` |

Equivalent manual start for one service (from the repo root, with `.env`
exported into the environment):

```bash
PYTHONPATH=backend python -m uvicorn src.front_service.main:app --host 127.0.0.1 --port 8000
```

The stack is safe to start twice: the launcher kills existing listeners on
those ports first.

## Checking readiness before you demo

- `http://127.0.0.1:8000/demo/readiness` — per-service **liveness** (its own
  `/health` answered HTTP 200) plus **readiness** / **model_readiness** when
  that service publishes them. `null` means "not published", never "ready".
- `http://127.0.0.1:8000/status` — the same health data that feeds the admin
  monitor, including audit-chain integrity.
- Individual services: `http://127.0.0.1:800X/health`.

The landing page polls readiness every 10 seconds and disables **Run synthetic
transaction** while the demo path is not ready.

## Running the demo

1. Open `http://127.0.0.1:8000/` and jump to **Live Demo**.
2. Pick a scenario (everyday purchase / large amount on a new device / high
   amount with repeated failed logins), or edit the fields.
3. Press **Run synthetic transaction**. The page posts to
   `/demo/synthetic-transaction`, which calls
   `POST /internal/ingest-transaction` on the Privacy Layer and then
   `POST /internal/evaluate` on the Risk Engine — the same documented path the
   CLI demo uses.
4. The result panel shows only fields the backend actually returned: decision,
   risk score/band, ML score, rule score, reason codes, fired rules, limit
   triggers, degraded/calibrated flags, and the derived privacy signals. A
   collapsible section shows the raw backend response verbatim.

### Repeatability

- Every run mints a **unique event id** (`DEMO-<random>`), so replaying a
  scenario never duplicates or de-duplicates a transaction.
- The browser keeps a synthetic demo account id in `localStorage` and reuses
  it, so repeated runs build up Privacy Layer history for that account. Press
  **New account** to start from a clean baseline.
- The demo endpoint is rate-limited per client to a small number of runs per
  minute; wait a minute if you hit the limit.

### Equivalent curl calls

```bash
TOKEN=$(grep -E '^INTERNAL_TOKEN=' .env | head -1 | cut -d= -f2- | tr -d '"')
EVENT="DEMO-$(python -c 'import uuid;print(uuid.uuid4().hex)')"
FRAUD="F$(python -c 'import uuid;print(uuid.uuid4().hex.upper()[:15])' | tr -d 'IO')"

curl -s http://127.0.0.1:8002/internal/ingest-transaction \
  -H "X-Internal-Token: $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"event_id\":\"$EVENT\",\"fraud_id\":\"$FRAUD\",\"amount\":85.0,\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%S)\",\"hour_of_day\":14,\"device_id\":\"demo-device-01\",\"location_id\":\"L-DEMO-A\",\"recipient_id\":\"R-DEMO-A\",\"failed_auth_count_24h\":0}" > /tmp/features.json

curl -s http://127.0.0.1:8003/internal/evaluate \
  -H "X-Internal-Token: $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"event_id\":\"$EVENT\",\"fraud_id\":\"$FRAUD\",\"features\":$(cat /tmp/features.json)}"
```

There is no commit/confirmation step in this path: each HTTP call is
independent, and the event id is what keeps runs distinct.

## Shutting down

Find the listeners and stop those PIDs (Windows):

```bash
netstat -ano | grep -E "LISTENING" | grep -E ":800[0-5]"
taskkill //PID <pid> //F
```

Git Bash users: `kill <pid>` often does not actually stop a Windows process —
use `taskkill //PID <pid> //F` and re-check `netstat` that the port is free.

## Common startup failures

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Port already in use on startup | an older stack is still listening | `netstat -ano \| grep :800X`, `taskkill //PID <pid> //F`, start again |
| Every internal call returns `401` | services disagree about `INTERNAL_TOKEN` (some started with an ephemeral value) | restart the whole stack from one shell so all services share the same environment |
| Risk `/health` reports the model as not loaded | `models/` artifacts missing for this checkout | restore the artifacts, then restart the Risk Engine |
| Privacy or Risk unreachable from the page | that service never started (check its own log) | readiness panel names the unreachable service |
| Demo returns `503` naming a service | that service went down after the page loaded | start it; the readiness list refreshes automatically |
| Demo returns `429` | demo rate limit | wait about a minute |
| Schema drift after a code change | the SQLite stores were created by an older model | demo stores are throwaway; delete `db/features.db` and `db/risk.db` before restarting (this erases demo state) |

## What this demo does and does not show

- It shows the **real software path and response contract**: feature
  derivation, rule evaluation, ML fusion, calibration, velocity limits, and the
  final decision, as returned by the running services.
- It does **not** establish fraud-detection performance, production readiness,
  regulatory compliance, or institutional validation. Scores are environment-,
  artifact-, and state-dependent.
- Real-world validation remains unestablished; institutional dataset validation
  remains blocked pending eligible data.
