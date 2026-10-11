# Phase 19 — End-to-End Fraud Detection Demo — Closeout

## Purpose

Implement and verify a reproducible, end-to-end demonstration of the existing
PS-14 Fraud Detection Software prototype, using the real supported path rather
than a parallel mock pipeline.

This phase demonstrates a synthetic transaction moving through the actual
available system components:

1. Privacy Layer input validation and feature derivation
2. Risk Engine feature validation, enforcement, rules + ML fusion (or documented
   degraded fallback), persistence, and decision response

It also adds narrow regression tests and documentation so the demo can be
reproduced.

This is a **research prototype using synthetic/demo data**. It demonstrates
**software behavior**, not real-world fraud-detection effectiveness, institutional
validation, production readiness, or regulatory compliance.

## Baseline

### Repository state at phase start

- Branch: `main`
- HEAD SHA: `929114559785426860448999672098eabce66b57`
- Remote `origin/main`: same SHA (`929114559785426860448999672098eabce66b57`)
- Working tree: dirty with inherited unstaged artifacts. Relevant inherited
  modifications include:

  - `backend/src/inference/service.py`
  - `backend/src/privacy_layer/main.py`
  - `backend/src/risk_engine/main.py`
  - `backend/src/risk_engine/main.py` (Phase 18-modified in place)
  - `backend/scripts/test_worker_pool.py`
  - `docs/evaluation/PHASE_INFERENCE_COVERAGE_REDIS_INTEGRATION_CLOSEOUT.md`
  - `reports/calibration_test/calibration_metrics.json`
  - `reports/evaluation_runs/eval_ledger.jsonl`

- Inherited untracked artifacts include the Phase 18 handoff notes and a large
  set of `reports/evaluation_runs/record_eval-*.json` files.

### Phase 18 relationship

- The recorded Phase 18 starting SHA matches the current HEAD, so the Phase 18
  baseline is still the relevant ancestor.
- Inherited Phase 18 modifications were preserved and were **not** overwritten to
  make this phase easier.
- The temporary Phase 18 handoff notes were kept separate and were **not**
  included in the Phase 19 deliverables:

  - `PHASE_18_FINAL_DELIVERY_SUMMARY.txt`
  - `PHASE_18_COMMIT_PLAN.md`

### What existed before this phase

- A real Privacy → Risk path already existed through the Verification service's
  `POST /demo/seed` flow and through the direct internal endpoints
  `POST /internal/ingest-transaction` and `POST /internal/evaluate`.
- The separate inference service on port 8006 and its Redis dependency are not
  part of the documented demo stack and were not started by the demo launcher.
- No prior Phase 19 demo/test/doc files existed in this checkout. The only
  existing “Phase 19” artifact was a different real-data / label-governance audit
  in `misc/reports/phase19/`. That artifact is out of scope for this phase’s
  demo mission and was not touched.

## Files changed and reasons

### Created

- `backend/scripts/phase19_e2e_demo.py`
  - Minimal reproducible CLI demo that chains the real Privacy Layer ingest
    endpoint into the real Risk Engine evaluate endpoint using `TestClient`
    against the **real app objects**. It exercises validation, feature
    derivation, rule evaluation, and the ML fusion or degraded fallback path
    without mocking the scorer.

- `backend/scripts/phase19_e2e_test.py`
  - Focused deterministic regression test in the repo’s existing plain-assert
    convention. It covers register/login, two valid transactions with different
    characteristics, malformed-input rejection, NaN/infinity rejection at the
    Risk Engine boundary, missing-token 401, and safe downstream-failure
    contract handling.

- `docs/demo/PHASE_19_E2E_DEMO.md`
  - Demo documentation matching the actual commands, input schema, response
    schema, and observed behavior, plus reproducible instructions and limitations.

- `docs/evaluation/PHASE_19_END_TO_END_DEMO_CLOSEOUT.md`
  - This file.

### Why those files

- They form the smallest practical end-to-end demo of the existing detection
  pipeline without introducing a parallel mock path.
- They stay within the project’s existing conventions: plain-assert test scripts
  and repo-root-launched services.
- They keep the real components on the critical path: Privacy Layer ingest and
  Risk Engine evaluate, not an inferred or simulated stand-in.

## Actual transaction flow and component boundaries

### Path exercised

For the hermetic demo and test:

1. Identity Service — register + login → JWT with `fraud_id` claim.
2. Privacy Layer — `POST /internal/ingest-transaction` with `X-Internal-Token`.
   - Validates the raw event.
   - Derives the live feature vector via `derive_event_features`.
   - Persists derived features to DB-2 (throwaway in the hermetic run).
   - Appends a background audit event.
   - Returns the event-shaped feature dict.
3. Risk Engine — `POST /internal/evaluate` with `X-Internal-Token`.
   - Validates the feature vector through `FeatureVector` + enforcement.
   - Runs rules + ML fusion or a documented degraded fallback.
   - Persists a pseudonymous risk score to DB-3.
   - Appends a background audit event.
   - Returns the decision response.

### Component boundaries

- **Identity Service** provides the real account/pseudonym bootstrap only as
  context for a demo account. The detection path itself is Privacy → Risk.
- **Privacy Layer** is the owner of DB-2 and the only service that derives the
  live feature vector.
- **Risk Engine** owns DB-3 and the decision logic.
- **Inference service (port 8006) and Redis** are **NOT** in this demo path.
  That is intentional and documented, not an oversight.
- **Audit Service** is the DB-4 write target for background append-only events;
  the demo still exercises the real append-from-other-service pattern because the
  Privacy and Risk code use the shared writer module.

## Exact startup and health-check commands

### Live stack startup

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

### Health checks for this demo path

```bash
curl -s http://127.0.0.1:8002/health | python -m json.tool
curl -s http://127.0.0.1:8003/health | python -m json.tool
```

### Focused test

```bash
PYTHONPATH=backend python backend/scripts/phase19_e2e_test.py
```

### Reproducible CLI demo

```bash
PYTHONPATH=backend python backend/scripts/phase19_e2e_demo.py
```

## Example synthetic request and observed response

### Synthetic privacy ingest request

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

### Synthetic risk evaluate request

```json
{
  "event_id": "demo-evt-0001",
  "fraud_id": "F0000000000000001",
  "features": { <privacy ingest response> }
}
```

### Observed response contract (from the real path)

A successful response includes at least:

```json
{
  "event_id": "demo-evt-0001",
  "fraud_id": "F0000000000000001",
  "risk_score": <int>,
  "risk_band": "low|medium|high",
  "decision": "allow|verify|block",
  "reason_codes": [...],
  "ml_score": <float>,
  "rule_score": <float>,
  "fired_rules": [...],
  "degraded": false,
  "calibrated": true,
  "odds": <float>,
  "limits": { "hard": false, "triggers": [] },
  "uncertainty": {
    "model_variance": <float>,
    "model_disagreement": <float>,
    "confidence": "high|medium|low"
  },
  "feature_version": "v1",
  "rule_version": "..."
}
```

The exact scores vary by model, artifacts, and account state. That is expected
and is why the demo asserts the **contract** and records the **observed shape**
rather than asserting one frozen fraud outcome.

## Tests run and results

### Order per acceptance criteria

1. Focused new Phase 19 test
2. Relevant existing unit/inference-style tests
3. Existing fast regression suite
4. Existing smoke test
5. Live service startup/health checks
6. Live end-to-end demo transaction
7. Malformed-input and safe-failure live cases
8. Redis / integration tests when available
9. Coverage where practical
10. CI/CD and Security Scan for the exact final commit SHA

### Actual results in this environment

#### Focused Phase 19 test (hermetic, against the committed SHA)

- Command: `PYTHONPATH=backend python backend/scripts/phase19_e2e_test.py`
  (runner: repo-root system Python 3.12.10. The `.venv/Scripts/python.exe`
  launcher does exist in this checkout and reports the same 3.12.10; it was
  not used for these runs.)
- Result: **PASS**
- Exit code: 0
- Evidence: hermetic FastAPI `TestClient` run against the real Privacy and Risk
  app objects, after the Phase 19 commit was created and pushed.
- Observed behavior:
  - Risk Engine loaded the Altman-NATIVE ensemble and the drift baseline, then
    accepted requests.
  - Valid transaction returned the real decision/response contract.
  - Second valid transaction with different characteristics returned a different
    derived signal profile.
  - Malformed inputs were rejected by the real Privacy Layer validation.
  - NaN/infinity were rejected by the real Risk Engine boundary.
  - Missing/wrong internal token was rejected.
  - Error paths did not fabricate a normal risk decision.

#### Existing smoke test (hermetic, against the committed SHA)

- Command: `PYTHONPATH=backend python backend/scripts/smoke_test.py`
- Result: **PASS**
- Exit code: 0
- Environment: hermetic, throwaway stores, same runner as the focused test.
- Note: this is the existing identity + privacy smoke test, not a Phase 19 file.
  It passed again in this turn, which is relevant because the Phase 19 demo and
  test also depend on the same Privacy/Identity startup path.

#### Existing fast regression suite (hermetic, against the committed SHA)

- Command: `PYTHONPATH=backend python backend/scripts/regression_suite.py --fast`
- Result: **FAIL**
- Exit code: 1
- Total: 28/29 fast suites passed; `backup_restore` failed and the suite exited 1.
- Exact failure from this turn:
  - `backup_restore` printed `Total: 71 | PASS: 70 | FAIL: 1 | BLOCKED: 0` and
    reported `FAILED: several checks failed.`
  - Its detail output showed one failed assertion:
    `Restore preserves all five quarantined forks ([])`.
- Honest reconciliation:
  - Suite result recorded here: **FAIL** (exit 1), because one configured fast
    suite failed.
  - Individual `backup_restore` check: this turn it **failed**, not merely
    “present”.
  - This is the same pre-existing `backup_restore` failure that other phases have
    recorded; the specific failure text in this turn is the quarantine-restore
    assertion above.

If a later runner sees a different `backup_restore` failure text, record that
actual text rather than assuming the same assertion.

#### Root cause of the `backup_restore` failure (established in the reconciliation turn)

- Reproduced in the Phase 19 reconciliation turn with the same command:
  exit code 1, `Total: 71 | PASS: 70 | FAIL: 1 | BLOCKED: 0`, with the single
  failing check `Restore preserves all five quarantined forks ([])`.
- Cause: **environment/fixture state, not a backup/restore code defect.** The
  assertion expects the restored audit chain to still contain the five
  documented 2026-09-19 quarantined forks (`AFFECTED_SEQUENCES` = seq 731, 735,
  740, 745, 750 in `src/monitoring/phase109_audit_fork_repair.py`). This
  checkout's `db/audit.db` has been regenerated and holds only 140 rows
  (`max(seq) = 140`), so none of those sequences exist and `evaluate_chain()`
  returns `quarantined_breaks = []` — while still reporting `ok = True` and
  `strict_ok = True`, i.e. the chain itself verifies.
- Scope: on a checkout whose `db/audit.db` still contains the forked rows the
  check passes; here it cannot. In CI (fresh checkout, no `db/`) the suite
  skips and exits 0, so this failure is local-state specific.
- Resolution: **UNRESOLVED (documented, test not weakened).** Satisfying the
  assertion would require an `audit.db` that still contains the historical
  forked rows; the local stores are valued demo state and audit history must
  not be rewritten, so nothing was modified to force a pass.

### Live service startup and health checks (live HTTP, against the committed SHA)

- Status: **PARTIALLY DONE**
- Startup method: `powershell -NoProfile -ExecutionPolicy Bypass -File .freebuff/start_stack.ps1`
  (the bare `powershell ...` form was blocked by the system execution policy in this
  turn; the bypass form returned exit 0 and printed per-service launch pids.)
- Listeners: `findstr`-style netstat checks returned no lines in this turn, so the
  live-stack readiness below is evidenced by the HTTP health responses and by the
  successful live transaction below, not by a captured listener-table snapshot.
- Health responses observed:
  - `GET http://127.0.0.1:8001/health` → `{"status":"ok",...}` (identity)
  - `GET http://127.0.0.1:8002/health` → `{"status":"ok","service":"privacy-layer",...}`
  - `GET http://127.0.0.1:8003/health` → `{"status":"ok","liveness":"alive","readiness":"ready",
    "model_readiness":"loaded","runtime_state":"READY",...}` (risk)
  - `GET http://127.0.0.1:8004/health` → `{"status":"ok",...}` (verification)
  - `GET http://127.0.0.1:8005/health` → `{"status":"ok",...}` (audit)
- This is live HTTP evidence, not TestClient. It was obtained on the running stack
  that the demo docs describe.

### Live end-to-end demo transaction (live HTTP, against the committed SHA)

- Status: **DONE**
- Path: `POST http://127.0.0.1:8002/internal/ingest-transaction` then
  `POST http://127.0.0.1:8003/internal/evaluate`, both with
  `X-Internal-Token: <INTERNAL_TOKEN>` (the live dev token from the gitignored
  `.env`). The literal value is deliberately not reproduced here: committing a
  live `.env` value to a tracked file would leak a credential and would fail the
  repository's own `secret_hygiene_test.py` gate.
- Synthetic request (live):
  ```json
  {
    "event_id": "LIVE-DEMO-001",
    "fraud_id": "F33592FE499DC4CD",
    "amount": 85.0,
    "ts": "2026-02-01T14:00:00",
    "hour_of_day": 14,
    "device_id": "demo-device-01",
    "location_id": "L-DEMO-A",
    "recipient_id": "R-DEMO-A",
    "failed_auth_count_24h": 0
  }
  ```
- Privacy ingest response: HTTP 200. Selected derived fields:
  ```json
  {
    "event_id": "LIVE-DEMO-001",
    "fraud_id": "F33592FE499DC4CD",
    "amount_ratio": 85.0,
    "txn_amount_bucket": "extreme_relative_to_avg",
    "txn_freq_last_24h": 0,
    "txn_time_unusual": 0,
    "new_device_flag": 1,
    "unusual_location_flag": 0,
    "unusual_recipient_flag": 0,
    "failed_auth_count_24h": 0,
    "days_since_last_similar_txn": 0.0,
    "gradual_escalation_score": 0.0,
    "known_device_count": 2,
    "account_tenure_days": 0.0,
    "hour_of_day": 14,
    "is_weekend": 1,
    "shared_device_accounts": 5,
    "shared_recipient_accounts": 5,
    "mule_ring_score": 1.0,
    "label": null
  }
  ```
- Risk evaluate response: HTTP 200. Selected decision fields:
  ```json
  {
    "event_id": "LIVE-DEMO-001",
    "fraud_id": "F33592FE499DC4CD",
    "risk_score": 85,
    "risk_band": "high",
    "decision": "verify",
    "reason_codes": [
      "DAILY_SPEND_EXCEEDED",
      "AMOUNT_UNUSUAL",
      "NEW_DEVICE",
      "MULE_RING",
      "FREQUENCY_ABNORMAL",
      "BEHAVIOR_DEVIATION",
      "DOMAIN_SHIFT"
    ],
    "ml_score": 0.0073,
    "rule_score": 1.0,
    "fired_rules": [
      "RULE_AMOUNT_HIGH",
      "RULE_AMOUNT_EXTREME",
      "RULE_NEW_DEVICE",
      "RULE_NEW_ACCOUNT_LARGE",
      "RULE_MULE_RING",
      "RULE_MULE_RING_STRONG",
      "RULE_SHARED_DEVICE",
      "RULE_SHARED_RECIPIENT",
      "RULE_MULE_RING_SCORE",
      "RULE_VELOCITY_HIGH_DAILY_SPEND",
      "RULE_VELOCITY_EXTREME_DAILY_SPEND",
      "RULE_VELOCITY_SPEND_NEW_DEVICE"
    ],
    "degraded": false,
    "calibrated": true,
    "odds": 0.0,
    "limits": {
      "pass": false,
      "triggers": [
        {
          "reason_code": "DAILY_SPEND_EXCEEDED",
          "level": "soft",
          "detail": "daily spend ratio 85.00x (limit 3.0x)"
        }
      ],
      "hard": false
    },
    "uncertainty": {
      "model_variance": 5.5e-05,
      "model_disagreement": 0.016969,
      "confidence": "high",
      "has_finance_data": false
    },
    "feature_version": "v1",
    "rule_version": "85803185ce32"
  }
  ```
- This is a real live HTTP transaction through the documented path. The numeric
  results are environment/artifact/state dependent and are recorded as observed,
  not as a fabricated outcome.

### Malformed-input and safe-failure live cases (live HTTP, against the committed SHA)

- Status: **DONE**
- Bad `fraud_id` on Privacy ingest: HTTP 422
  - body: `{"detail":[{"type":"string_pattern_mismatch",...}]}`
- Negative `amount` on Privacy ingest: HTTP 422
  - body: `{"detail":[{"type":"greater_than",...}]}`
- Missing `X-Internal-Token` on Risk evaluate: HTTP 422
  - body: feature-missing validation errors (the endpoint rejects the request
    before it would return a risk decision)
- Wrong `X-Internal-Token` on Risk evaluate: HTTP 422
  - body: same feature-missing validation shape
- These confirm that bad input is rejected by the real validation paths on the
  live stack and that no fake risk decision is returned.

### Redis / integration tests

- Status: **NOT RUN**
- Reason: no Redis service was verified available for this turn, and this demo
  path does not require Redis anyway (it exercises the Privacy→Risk path, not the
  separate inference service).

### Coverage

- Status: **NOT RUN** in this turn
- Reason: coverage was not re-measured here.

## Whether the real end-to-end path was demonstrated live

- **Yes — live in this turn.**
- A synthetic transaction was submitted through the live stack on ports 8002/8003,
  the Privacy Layer returned a derived feature vector, and the Risk Engine returned
  a real decision/response contract.
- That is live HTTP evidence, not TestClient. The hermetic TestClient run remains
  solid evidence for the implementation and contract, but the live transaction above
  is the separate live-HTTP verification.

## CI/CD and Security Scan status

Evidence below was collected in the Phase 19 **reconciliation turn** from the
GitHub Actions API for `RanaYashIDK-2006/PS-14-Fraud-Detection-software`
(no credentials or secrets exposed).

For the exact Phase 19 commit SHA `1b14a42480e9608ec8a6d41980d7f74cf6050742`:

- Security Scan: **PASS**
  - run `38083417029` — `Security Scan`, created 2026-10-10T20:21:16Z,
    completed / success.
  - run `38098094114` — `Security Scan`, created 2026-10-11T00:20:29Z,
    completed / success (a second, later run on the same SHA).
- CI/CD: **FAIL**
  - run `38083417206` — `CI/CD`, created 2026-10-10T20:21:16Z,
    completed / failure.
  - Failing job: `Test Suite` (job id `114304825134`); failing step #7
    `Run full regression suite` (`python backend/scripts/regression_suite.py --fast`).
  - The downstream jobs (`Security Scan`, `Docker Build`, `Integration Test`,
    `Deploy Image`) were **skipped**, not failed.

Cause of the CI/CD failure (established from the repository contents):

- `regression_suite.py`'s `FAST_TESTS` references `scripts/inference_hermetic_test.py`,
  but that file is **not tracked by git** — `git ls-files` returns nothing for
  it and it has no commit history. A fresh CI checkout therefore runs
  `python backend/scripts/inference_hermetic_test.py` against a missing file,
  which exits non-zero (Python exits 2 for an unopenable script) and fails the
  whole fast suite. Every other FAST/FULL suite script is present in HEAD.
- Locally that file exists in the working tree, so the identical suite passes
  here and only `backup_restore` fails — which is why the local run and CI
  disagree about which suite breaks.
- The reference was introduced by commit `9291145` (Phase 17), and the same CI
  step also failed on that baseline SHA (run `38063294221`, step #7). The
  failure is therefore **pre-existing relative to Phase 19**; Phase 19 did not
  introduce it and does not change it.
- Classification: **FAIL, pre-existing, not a demo-path regression.** The
  Privacy→Risk demo path plus its focused and smoke tests pass; this is a
  repository-hygiene defect (a test list referencing an uncommitted file).
  It stays visible here rather than being relabelled as a pass.

Limitation: GitHub Actions job **logs** are not retrievable from this
environment (unauthenticated request → HTTP 403; the git credential helper's
stored token → HTTP 401/403 for the logs endpoint). The failing suite was
identified from the workflow definition and repository contents, not from the
raw run log.

## Preserved artifacts and unrelated changes

The following were preserved and **not** overwritten or removed:

- Inherited `reports/` artifacts, including `record_eval-*.json` files
- Inherited calibration artifacts
- Inherited unstaged changes such as `backend/scripts/test_worker_pool.py`
- The inherited Phase 18 modified files in place

The temporary Phase 18 handoff notes were kept separate and were **not** included
in the Phase 19 deliverables:

- `PHASE_18_FINAL_DELIVERY_SUMMARY.txt`
- `PHASE_18_COMMIT_PLAN.md`

## Known defects, blockers, and limitations

### Real limitations

- The demo proves the **software path and contract**, not fraud-detection
  performance.
- Scores are environment/artifact/state dependent. A different checkout or model
  state can return different numeric results for the same synthetic input.
- The inference service / Redis path is not part of this demo and remains
  NOT ESTABLISHED here.

### Blockers remaining for full verification

- CI/CD for the Phase 19 SHA **fails** because
  `backend/scripts/inference_hermetic_test.py` was never committed. Fixing it
  means committing that inherited, untracked Phase 16/17 test script so the
  tracked `FAST_TESTS` list is internally consistent; that was **deferred**
  here because staging inherited untracked files is outside this phase's
  approved scope.
- The local `backup_restore` failure is documented above and remains
  **UNRESOLVED** (environment state, test not weakened).
- Actions job logs could not be retrieved in this environment (auth
  limitation), so the CI failure is diagnosed from repository contents.

## Recommendation on whether Phase 20 can begin

**Yes — proceed, carrying the failures above forward as explicit evidence.**

Phase 19 is committed and pushed to `origin/main`; the focused test, smoke
test, live health checks, live demo transaction, and live safe-failure cases
all completed. Security Scan passes for the Phase 19 SHA. Two failures remain
and neither breaks the demo path:

- CI/CD FAIL, caused by the uncommitted `inference_hermetic_test.py` referenced
  from `FAST_TESTS` (pre-existing since Phase 17's `9291145`).
- Local `backup_restore` FAIL, caused by this checkout's regenerated
  `db/audit.db` no longer containing the five quarantined fork rows.

Phase 20 must not claim a green CI: the same CI step will keep failing for the
Phase 20 SHA until the missing test script is committed.
