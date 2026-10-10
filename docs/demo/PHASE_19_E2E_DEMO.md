# Phase 19 — Privacy → Risk end-to-end demo

This is a minimal, reproducible demonstration of a synthetic transaction moving
through the **actual** PS-14 detection pipeline:

1. **Privacy Layer** — `POST /internal/ingest-transaction` validates the raw
   event, derives the live feature vector, persists derived features only, and
   returns the event-shaped feature dict.
2. **Risk Engine** — `POST /internal/evaluate` validates the feature vector,
   runs rules + ML fusion (or a documented degraded fallback), persists a
   pseudonymous risk score, and returns the decision response.

This is a **research prototype using synthetic/demo data**. It demonstrates
**software behavior**, not real-world fraud-detection performance, institutional
validation, production readiness, or regulatory compliance.

## What this demo is — and is not

- **In scope:** validation, privacy feature derivation, rule evaluation, ML
  fusion or degraded fallback, and the risk-decision response contract.
- **Out of scope here:** the separate inference service on **port 8006** and
  its **Redis** dependency. Those are **not** started by the documented demo
  stack and are not exercised by this demo path.
- **Not claimed:** any fixed fraud outcome, any real-world performance number,
  or any simulated result masquerading as a model prediction. Where the model
  result can vary, this document describes the response **contract** and records
  the **observed** shape rather than inventing a classification.

## Prerequisites

- Repository checkout with `backend/` intact.
- Python environment where `src.*` is importable and the root `db/` is writable
  for the throwaway demo stores. The repo convention is to run from the
  **repository root** with `--app-dir backend` so `src.*` is on the import path
  and the store paths resolve to the root `db/`.
- For the live stack: the six documented backend services, not the inference
  service. The demo launcher is `.freebuff/start_stack.ps1` on Windows.

## Configuration variables (no real secret values)

The services rely on settings from `src/settings.py`. For a local dev/demo run,
the relevant non-secret configuration is the **mode** and the presence of the
internal token/JWT secret used by the internal endpoints.

Do **not** paste real secret values into docs or chat. In a real run, those come
from the environment / `.env` as the project already documents. For the hermetic
TestClient demo and test, the apps still go through their normal settings loader,
so the same env expectations apply.

## Startup order (live stack)

For a live HTTP demo against a running stack, start the services in this order:

```bash
# From the repository root
uvicorn src.front_service.main:app --port 8000 --app-dir backend
uvicorn src.identity_service.main:app --port 8001 --app-dir backend
uvicorn src.privacy_layer.main:app --port 8002 --app-dir backend
uvicorn src.risk_engine.main:app --port 8003 --app-dir backend
uvicorn src.verification_service.main:app --port 8004 --app-dir backend
uvicorn src.audit_service.main:app --port 8005 --app-dir backend
```

Windows launcher:
```powershell
powershell -ExecutionPolicy Bypass -File .freebuff/start_stack.ps1
```

The Privacy Layer (:8002) and Risk Engine (:8003) are the two services this demo
path depends on.

## Health-check procedure

Check that the two services this demo needs are healthy:

```bash
curl -s http://127.0.0.1:8002/health | python -m json.tool
curl -s http://127.0.0.1:8003/health | python -m json.tool
```

Each returns a `status` field. Do not submit demo traffic until both are
returning healthy. Privacy health reflects DB-2 reachability; risk health reflects
model/runtime readiness, which is the more informative signal before demo traffic.

## Submitting a demo transaction (live stack)

The live path uses two internal endpoints. Both require the internal token header.

1. Ingest the raw transaction into the Privacy Layer:
   `POST http://127.0.0.1:8002/internal/ingest-transaction`
   Header: `X-Internal-Token: <internal token>`
2. Evaluate the returned feature vector in the Risk Engine:
   `POST http://127.0.0.1:8003/internal/evaluate`
   Header: `X-Internal-Token: <internal token>`

The Risk Engine request body wraps the Privacy Layer response as a
`features` object:

```json
{
  "event_id": "<same event_id>",
  "fraud_id": "<same fraud_id>",
  "features": { <privacy ingest response> }
}
```

## Input schema (actual)

Privacy Layer `POST /internal/ingest-transaction` accepts something like:

```json
{
  "event_id": "string, 8-64 chars",
  "fraud_id": "F[A-Z2-9]{15}",
  "amount": 120.0,
  "ts": "2026-02-01T14:00:00",
  "hour_of_day": 14,
  "device_id": "demo-device-01",
  "location_id": "L-DEMO-A",
  "recipient_id": "R-DEMO-A",
  "failed_auth_count_24h": 0
}
```

Risk Engine `POST /internal/evaluate` accepts:

```json
{
  "event_id": "string, 8-64 chars",
  "fraud_id": "F[A-Z2-9]{15}",
  "features": { <privacy-layer response> }
}
```

The `features` object is exactly what the Privacy Layer returns; the Risk Engine
validates it through its own `FeatureVector` model and enforcement layer.

## Response schema (actual)

A successful Risk Engine response includes at least:

```json
{
  "event_id": "string",
  "fraud_id": "string",
  "risk_score": 0,
  "risk_band": "low|medium|high",
  "decision": "allow|verify|block",
  "reason_codes": [...],
  "ml_score": 0.0,
  "rule_score": 0.0,
  "fired_rules": [...],
  "degraded": false,
  "calibrated": true,
  "odds": 0.0,
  "limits": { "hard": false, "triggers": [] },
  "uncertainty": { "model_variance": 0.0, "model_disagreement": 0.0, "confidence": "high" },
  "feature_version": "v1",
  "rule_version": "..."
}
```

The exact numeric results depend on model, artifacts, and account state. That is
expected. This demo does **not** assert a hard-coded fraud outcome.

## Worked example (contract shape)

A single demo transaction through the real path looks like this in outline:

1. Privacy ingest request:
   ```json
   {
     "event_id": "demo-evt-0001",
     "fraud_id": "F0000000000000001",
     "amount": 85.0,
     "ts": "2026-02-01T14:00:00",
     "hour_of_day": 14,
     "device_id": "demo-device-01",
     "location_id": "L-DEMO-A",
     "recipient_id": "R-DEMO-A",
     "failed_auth_count_24h": 0
   }
   ```
2. Privacy ingest response includes derived fields such as:
   - `amount_ratio`
   - `txn_freq_last_24h`
   - `txn_time_unusual`
   - `new_device_flag`
   - `unusual_location_flag`
   - `unusual_recipient_flag`
   - `failed_auth_count_24h`
   - `days_since_last_similar_txn`
   - `gradual_escalation_score`
   - `known_device_count`
   - `account_tenure_days`
   - `hour_of_day`
   - `is_weekend`
   - link-analysis fields `shared_device_accounts`, `shared_recipient_accounts`,
     `mule_ring_score`
   - `label: null`

3. Risk evaluate request wraps that as `features`.
4. Risk evaluate response includes the decision payload above.

The demo script `backend/scripts/phase19_e2e_demo.py` prints a live contract
sample when run.

## Running the focused test

```bash
python backend/scripts/phase19_e2e_test.py
```

This is the focused Phase 19 regression test. It uses `TestClient` against the
real app objects and does **not** mock the scorer, rules, or fusion.

## Reproducing the CLI demo

```bash
python backend/scripts/phase19_e2e_demo.py
```

This is the minimal reproducible demo script. It prints a structured summary and
a contract-shape example payload.

## Example invalid-input cases

- Malformed `fraud_id` → Privacy Layer validation error.
- Negative `amount` → Privacy Layer validation error.
- Too-short `event_id` → Privacy Layer validation error.
- Missing `X-Internal-Token` → Risk Engine `401`.
- NaN or infinity in a numeric feature fed to the Risk Engine → Risk Engine
  validation/rejection at the API boundary.

These are intentional demo cases. They confirm that bad input is rejected by the
real validation paths and that no fake risk decision is returned.

## Known limitations

- Results depend on the deployed model, artifacts, rules, and account state.
  Different environments can return different numeric scores for the same
  synthetic input.
- The demo uses synthetic/demo data. It is not evidence of real-world
  effectiveness.
- The inference service on port 8006 and Redis are not part of this demo path.
- Live HTTP verification of this document's walkthrough was not executed in this
  environment turn; the TestClient paths were used instead. Live results should be
  labelled separately when the stack can be started.
- This demo does not touch fraud thresholds, model weights, calibration, feature
  definitions, or fraud-decision policy to make the demo appear more successful.

## Troubleshooting

- If the Privacy Layer returns `422`, inspect the request body against the input
  schema above.
- If the Risk Engine returns `401`, confirm the internal token matches the running
  service config.
- If the Risk Engine returns `degraded: true`, that is a real fail-open/fail-safe
  indicator, not a simulated success. Document it as such.
- If a service is unhealthy, do not proceed with demo traffic. Check health first.
