# AGENTS.md

Operating notes for future sessions on the PS-14 prototype (architecture in
README.md, preview-stack details in `.freebuff/run.md`). These are facts that
code reading alone does not reveal.

## Environment

- This project is **not a git repository** (OneDrive folder) — `git status` /
  `git log` fail with "not a git repository". There is no commit workflow; all
  changes live on disk.
- The venv `python.exe` is a Windows redirector: launching uvicorn via
  `./.venv/Scripts/python.exe` shows the venv pid, but the actual socket
  listener is a **child process under the winget system Python 3.12**
  (`%LOCALAPPDATA%\Programs\Python\Python312\python.exe`). The venv pid never
  appears in `netstat`, yet killing it does stop the service — use it as the
  handle; don't chase the netstat pid.
- Git Bash's `kill <pid>` **silently fails** on Windows processes: the port
  stays bound and the old service keeps serving stale code while the shell
  reports success (in-process test suites stay green — they never touch the
  live listener). Always terminate with `taskkill //PID <pid> //F`, then
  confirm the port is free with `netstat` before restarting.

## Preview / dev stack

- The parent `powershell` invocation hangs on the PS 5.1 `Start-Process`
  redirect quirk — from Git Bash, pipe its output to a file
  (`powershell -NoProfile -File .freebuff/start_stack.ps1 > out 2>&1`) and it
  returns promptly (exit 0). Otherwise verify via the health loop /
  `netstat`, not by waiting on the shell.
- A service found dead (recurring :8004 "deaths") is usually the Freebuff
  preview system stopping the previously registered server when the preview
  target switches (`register_preview` with `replace: true`) — not a crash.
  Restart with `start_stack.ps1` and re-register the preview.

## Hidden relationships

- `settings.jwt_secret` also derives `fernet_key` and `export_signing_key` —
  do NOT change the 25-byte default "to fix" the `InsecureKeyLengthWarning` in
  the logs; it would break decryption of existing DB-1 PII. Known dev debt.
- The demo seed commits **5/6** warm-ups by design: the first warm-up only
  bootstraps the DB-2 profile (never evaluated - scoring it against zero
  history is a cold-start false positive), the account is then AGED to 60
  days BEFORE the remaining warm-ups are evaluated/committed, and the attack
  is evaluated last. The seed also gives each demo account its OWN device,
  location, and recipient tokens - the link-analysis features count OTHER
  accounts per token globally, so a constant token (e.g. `R-DEMO-01`)
  reused across demos makes every later account look like a mule ring.
- `ml_score` from `/internal/evaluate` is now a CALIBRATED probability
  (Platt scaling — `PlattCalibration` in `src/risk_engine/calibration.py`,
  pickled to `models/artifacts/calibrator.joblib`; isotonic produced a
  step function on the near-separable synthetic data); the response also
  carries `odds` (clamped at 9999 when p == 1 - JSON cannot carry inf, and
  both the audit writer and FastAPI serialize it) and `degraded` (fail-open:
  ML failure or an open circuit breaker falls back to rules-only, tagged in
  DB-3 and the audit chain; a degraded evaluation never 500s).
- The link-analysis features added columns to DB-2 `transaction_features`
  and `risk_scores` (DB-3) gained `degraded`: SQLAlchemy `create_all` does
  NOT add columns to existing tables - delete `db/features.db` and
  `db/risk.db` before restarting the stack (demo state is throwaway;
  `audit.db`/`identity.db` keep their data).
- Front page service (`src/front_service`, port 8000): serves the landing
  page, a server-side `/status` aggregate of all five `/health` endpoints
  (avoids CORS entirely — never fetch services cross-origin from the page),
  and a passphrase-gated ADMIN area. Admin record (`db/admin.json`) stores
  ONLY essential fields: username, scrypt passphrase hash, timestamps,
  login count, and an AES-256-GCM essentials blob (wrap key derived from
  the passphrase). `ADMIN_PASS` is a BOOTSTRAP-only value — after the first
  login only the stored hash is verified; delete `db/admin.json` to re-bootstrap.
- Services do NOT auto-load `.env` (Settings has no env_file) and are
  started detached, so exported vars never reach them. The front service
  merges `.env` itself for `ADMIN_PASS`; other services rely on real env
  vars or dev defaults.
- Dockerized retrain: compose `train` profile service runs
  `training_pipeline.py --full` with `./models` and `./data` BIND-MOUNTED
  (artifacts must persist on the host for the rebuild); the driver
  (`scripts/docker_retrain.sh` / `make retrain`) rebuilds + restarts ONLY
  risk, and only on exit 0. The image has no `db/` — the four stores live
  in the `dbdata` volume; deleting the volume wipes all demo state.
- UIs derive service URLs from `window.location.hostname`, so the same
  stack works on localhost, LAN, and behind a proxy. Docker: one image,
  `docker-compose.yml` picks the module via `MODULE`/`PORT` env; all four
  stores share the `dbdata` volume at `/app/db`.
- Windows: `Start-Process ... -PassThru` prints a transient launcher pid —
  the ACTUAL listener pid comes from `netstat -ano | grep :PORT`; register
  the listener pid with the preview. Also: registering a NEW preview with
  `replace: true` KILLS the previously registered dev server — restart the
  old one afterwards (this killed :8004 while registering :8000).
- Private access info lives encrypted in `docs/ACCESS.md.enc` (Fernet).
  Open: `python scripts/access_doc.py view` - the key is `ACCESS_DOC_KEY`
  in the gitignored `.env`; the plaintext `docs/ACCESS.md` is gitignored.
  NEVER add `JWT_SECRET` (or any other runtime secret) to `.env` with a
  value different from the current one: pydantic-settings reads it, which
  re-derives `fernet_key` and breaks decryption of existing DB-1 PII.
  `scripts/access_doc.py` is NOT in the pipeline test suites (no test
  dependency on it); the suite list is in `training_pipeline.ALL_SUITES`.
- Per-row `FusionEngine.predict` is slow (RF/XGB on 1-row matrices): use
  `predict_many`/`predict_raw_many` (batched) for offline loops - the rules
  simulator and any future sweep must not call `predict` in a loop.
- Group-split `recall_f1` is UNRELIABLE at low fold fraud prevalence: the
  ato fold's validation has ~0.2% fraud (holding out ato removes nearly all
  fraud from the val window), so its F1-optimal threshold lands far above
  held-out ato scores (recall_f1 0.09) while recall@1%FPR stays 1.0. Compare
  generalization with `recall_at_1pct_fpr` (ranking-based), not `recall_f1`.
  The OOD recall gate (`train_compare --ood-recall-floor`, default 0.60 on
  ato,mule) is built on the same reasoning - it FAILS the retrain on
  recall@1%FPR drops, and `--no-ood-gate`/`--ood-recall-floor 0` accepts a
  regressed model deliberately. Gate rows are in `metadata.json` `ood_gate`;
  `scripts/ood_gate_test.py` is in the pipeline's ALL_SUITES.
- Generator noise uses its own rng stream (`seed + 1`) so `--noise 0` and
  `--noise 0.15` share bit-identical row sets; only feature values differ.
  `event_id` is unseeded `uuid.uuid4()`, so compare runs on
  fraud_id+ts+label, never event_id.
- `scripts/train_compare.py` backfills ML features missing from OLD feedback
  snapshots with 0, so snapshots taken before a feature was added stay
  reproducible; fresh exports carry real values.
- `scripts/simulate_rules.py` is the rules "what-if" tool: replays a
  candidate rules.yaml / severity_scale over `data/transactions.csv` with
  the live calibrated fusion and reports recall/FPR/decision deltas. The
  ML pass uses the batched predictor and is computed once per run.
- SQLite trigger aborts (`RAISE(ABORT)`, the append-only enforcement on DB-4's
  `audit_events`) surface as `sqlite3.IntegrityError`, **not**
  `OperationalError` — catch both when exercising the append-only path.
- `src/audit_service/writer.py` is the **single write path** to DB-4: other
  services import `append_audit_event` and write the chain directly (shared
  SQLite file, not the audit service's HTTP API); importing the module also
  creates DB-4's tables + append-only triggers. The hash chain detects
  modified entries but cannot distinguish forged appends — any service with
  writer access can append.
- The audit append is **on the detection critical path**: risk `evaluate` and
  privacy ingest call `append_audit_event` synchronously, un-wrapped, opening
  a second SQLite session to DB-4 per event — a DB-4 lock/outage fails fraud
  decisions. Reads are resilient (`/chain-status` returns 503 when audit is
  down); writes are not.
- The two trail UIs — `src/verification_service/static/index.html` and
  `src/audit_service/static/index.html` — are **independent reimplementations**
  of the same compact trail (different function names, not shared code): any
  trail-UI fix must be applied to both files.
- The verification UI builds its API client from the services' OpenAPI
  schemas at load time (`src/verification_service/static/api.js`,
  `generate_unique_id_function` on the FastAPI apps): **operationId in the
  schema == the generated helper name** (`overview` → `api.verify.overview`,
  `confirm_alert` → `api.verify.confirmAlert`). Adding a backend endpoint
  makes it callable in the UI automatically; renaming a handler function
  renames the helper — don't add hand-written fetch wrappers or pin UI code
  to paths. **Query params are passed by their OpenAPI (snake_case) names**
  (`device_id`, not `deviceId`) — the client forwards them verbatim and
  silently drops unknown keys, so a camelCase key fails as "missing param".
- Velocity limits (`src/risk_engine/limits.py`, configured in
  `rules.yaml` `velocity_limits`): the Risk Engine enforces them from the
  §16 vector the Privacy Layer builds (`txn_freq_last_24h`,
  `account_daily_spend_ratio`, `device_daily_count`), it never queries DB-2.
  `velocity_limits_cfg` is assigned in the lifespan — it MUST be listed in
  the `global` statement there or the assignment silently becomes a local
  and limits never fire (caught as a KeyError on `limits["hard"]`).
- The Privacy Layer's 24h velocity windows are anchored at the event's own
  `ts` (`anchor = min(now, ts)`) and need BOTH bounds
  (`created_at >= anchor-24h AND created_at <= anchor`): with only the
  lower bound, a backdated event (demo seed / replays) counts every row
  ingested later in real time. The demo ages rows AFTER ingest, which is
  exactly why the lower-bound-only window broke it (warm-ups got
  `DAILY_SPEND_EXCEEDED` and dropped to 3/6 commits).
- The model MEMORIZES the synthetic archetypes: the time-split test metrics
  are optimistic because every archetype appears in both train and test
  (same-pattern leakage) - random-split recall ~0.92 vs leave-one-
  archetype-out ~0.50 (ato ~0.51, mule ~0.38). Live/OOD vectors (the five
  pinned scenarios in `data/ood_scenarios.csv`) score ml=0.0, so rules
  carry every live decision. `src/train_compare.py` runs the group-split
  eval by default (`--no-group-eval` skips; ~35s extra per retrain) and
  writes `models/artifacts/generalization_by_archetype.csv`; the tuner
  appends it to its report. Any retrain rewrites artifacts (mtime) and
  therefore invalidates the tuner's per-event score cache (~230s recompute
  on the next tuner/pipeline run - expected, not a bug).
- Device/recipient hashing is inconsistent by design: ingest stores
  `_hash_device(id) = sha256(id + "::" + jwt_secret)[:32]` (keyed,
  truncated), while location/recipient are stored raw. Any lookup endpoint
  (e.g. `/internal/device-graph`) must use `_hash_device` — plain
  `sha256(id)` returns nothing. The compliance graph proxy accepts raw
  tokens and the Privacy Layer hashes on lookup.

## Working style

- User preference: keep changes minimal — cut accumulated extras (telemetry
  added only to confirm a fix, handling for impossible cases, single-use
  constants); when a removal is a judgment call, leave it and flag it.

## Tests

- Each suite is a plain assert script (no pytest): run
  `./.venv/Scripts/python.exe scripts/<name>_test.py` from the project root.
  The ten suites: smoke, risk_engine, verification, audit, drift, feedback,
  k_anonymity, pipeline, tune, federated.
