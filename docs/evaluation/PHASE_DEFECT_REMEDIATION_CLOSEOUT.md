# PS-14 — Phase 15 (Deterministic Test & Defect Remediation) Closeout

**Phase identity:** Phase 15 — Deterministic Test and Defect Remediation (F1–F4)
**Status:** COMPLETE
**Date:** 2026-10-10
**Host:** Windows 11, project venv `.venv/Scripts/python.exe` (Python 3.12); Redis was
listening on `:6379`; the five application services (8000–8005) were **not** running.

---

## 1. Commit / SHA ledger

| Item | Value |
|---|---|
| Branch | `main` |
| Starting SHA (`HEAD` before this phase) | `189b2bfb623f249c4bf3fd39efa47b034b08ad61` |
| Starting point vs. Phase 14 | Phase 14's verified delivery was `c968306`; `189b2bf` is the **one intervening commit** it recorded as discrepancy **D16** (the README "demo accounts" correction). No other commit landed between phases, so this phase's baseline is Phase 14's work plus D16. |
| `origin/main` at start | identical to `HEAD` — `git rev-list --left-right --count HEAD...origin/main` → `0 0` |
| Content commit (this phase) | recorded in the table below after push |
| Results commit | the follow-up commit that records the CI/Security Scan receipts |
| Remote | `https://github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software.git` |

Inherited working-tree artifacts were **preserved and never staged**: `reports/calibration_test/calibration_metrics.json`
(modified), `reports/evaluation_runs/eval_ledger.jsonl` (modified by earlier phases' runs) and the 74 untracked
`reports/evaluation_runs/record_eval-*.json` records. (Running the documented battery is not read-only: the
evaluation ledger grows and the calibration metrics timestamp is refreshed by design.)

---

## 2. Findings: reproduction and root cause

### F1 — `federated_worker.py` local `import sys` kills every worker on full-coverage data

**Reproduction (pre-fix):** run the worker on a 40-row CSV containing **all 21 `ML_FEATURES` columns**, feeding
it `{"type":"shutdown"}` on stdin:

```
python backend/scripts/federated_worker.py --data <all-columns>.csv --name t_0   # exit 1
UnboundLocalError: cannot access local variable 'sys' where it is not associated with a value
  File ".../backend/scripts/federated_worker.py", line 100, in main
    for line in sys.stdin:
```

**Root cause:** `import sys; print(...)` sat *inside* the `if len(available) < len(ML_FEATURES):` branch
(line 79). A function-local import makes `sys` a **local** name for all of `main()`, so when every
`ML_FEATURES` column is present the branch is skipped and `sys.stdin` is unbound — every institution worker
dies at startup and `federated_test.py` fails with `worker t_0 failed to start (rc=1)`. The warm dev tree
masked it because its cached pool CSV lacks a production-only feature, which executes the branch (exactly the
CI-only manifestation Phase 13 recorded).

**Fix:** delete the inline import (the module already imports `sys` at line 27); the informational stderr line
is unchanged. No behavior change on the missing-column path.

### F2 — `audit_test.py` asserts environment-dependent absolute chain lengths

**Reproduction (pre-fix):** with the gitignored `models/production/release_manifest.json` present the suite
passes (`n_entries` 5 → 7); with the manifest temporarily absent — the fresh-checkout equivalent — it fails:

```
python backend/scripts/audit_test.py        # exit 1, 11 CHECK(S) FAILED
  [FAIL] integrity ok                 ({... 'n_entries': 4 ...})
  [FAIL] audit: chain still verifies  ({... 'n_entries': 6 ...})
  ... plus 9 more count/paging/export assertions
```
(The manifest was renamed back immediately; the file was verified intact afterwards.)

**Root cause:** the lifespan appends `runtime_attestation_state` **plus** a `runtime_release_loaded` entry —
one extra startup entry — **only** when a *verified* release manifest is on disk. The suite hard-coded the
dev-tree lengths (`== 5` / `== 7`), the paging page size (`limit=3&offset=4`), and the tamper target
(`seq = 2`) — all of which shift with the startup baseline. On a fresh checkout the chain is 4 → 6 and 11
assertions fail even though every payload reports `ok: True`.

**Fix:** read the real baseline once (`STARTUP_N` from `/audit/integrity`), derive `N3 = STARTUP_N + 3`
(2× `score_generated` + `verification_resolved`) and `N5 = STARTUP_N + 5` (+ `feature_ingested`,
`fraud_id_resolved`), size the final trail page `limit={N5 - 4}`, and derive `SCORE_SEQ` (the first
`score_generated`) as the tamper target. **No assertion was weakened:** hash-chain linkage, payload
pseudonymity, integrity verification, signature verification, independent export re-verification, the
append-only trigger, and tamper detection all remain — the tamper now targets a *real score payload* instead of
whichever entry happens to sit at seq 2, which is strictly more meaningful.

### F3 — `risk_engine/main.py` raises `NameError: model_version` on an untrained checkout

**Reproduction (pre-fix):** isolated equivalent of an untrained checkout — the module constants
`ARTIFACTS_DIR` / `PRODUCTION_DIR` redirected to an empty temp directory (no artifact or manifest on disk is
read, moved, or modified) — then `TestClient(risk_app)` startup:

```
NameError: name 'model_version' is not defined
  File ".../backend/src/risk_engine/main.py", line 414, in lifespan
    _obs_store.model_telemetry.model_id = model_version or "unknown"
```

**Root cause:** `model_version` is a module global assigned only inside `if meta_path.exists():`; the `else`
branch printed `metadata.json MISSING - model_version unknown` but never assigned the name, so the very next
consumer crashed the whole lifespan. Any checkout without trained artifacts — a fresh clone, or CI before the
train step — fails to start.

**Fix:** `model_version = "unknown"` is set *before* the branch (the honest sentinel already used by the
existing `/health` fallbacks — never a fabricated version). The trained path is untouched: a trained tree still
resolves `model_version` from `metadata.json` (verified: `seed42-transactions.csv`).

### F4 — intermittent FIFO assertion failure in `audit_resilience_test.py`

**Reproduction:** a focused harness that replays §7 exactly (DB-4 sessions fail → six `append_audit_event`
calls spool to the pending file → sessions restored → the background retry replays) and reports whether the
recovered chain seq order equals append order:

| DB-4-failure window | iterations | non-ascending | incomplete recoveries |
|---|---|---|---|
| default timing (4 harness variants) | 70 | 0 | 0 |
| 0.05 s between appends | 12 | 0 | 0 |
| **0.12 s between appends** | 12 | **12** | 0 |
| **0.2 s between appends** | 12 | **12** | 0 |

Example observed orders: `[47, 48, 44, 45, 46, 43]`, `[53, 54, 49, 50, 51, 52]` — the head items land last.
Every event was recovered **exactly once** in every iteration (`incomplete_recovery = 0`), so this is an
*ordering* effect, not event loss.

**Root cause:** `_retry_sweep` skips records whose per-item backoff (`next_retry_at`) has not elapsed, so an
item that accrued backoff while the failure window was open is replayed *after* later items. The writer
documents this explicitly (module docstring: "FIFO holds within each stream (queue drain order, pending-file
append order) … durability is prioritized over cross-stream order in that pathological case"). The test's
`seq ascending` assertion therefore encoded an ordering guarantee the writer does not make while the retry
worker is concurrently active — a schedule artifact, which is why Phase 13 saw ~2/12 under coverage
instrumentation and none uninstrumented.

**Fix (test-side, explicit synchronization — no writer change):** suspend `w._retry_sweep` while the pending
file is built and resume it before recovery, so no item can accrue backoff mid-fixture and recovery is a single
pass over the file in append order. The strict FIFO assertion is **retained**, and a new
schedule-independent `exactly-once` assertion was added (6 rows, 6 distinct seqs). Deliberately **not** changed:
the writer's durability-over-order behavior (see §9).

### F5 — `audit_test.py` read-after-async-append race (discovered while remediating F2)

**Reproduction:** one of six comparable env-A runs failed `check("events logged", …)` with
`(['score_generated', 'score_generated'])` — the `verification_resolved` event was missing although its HTTP
call returned 200.

**Root cause:** `append_audit_event` "never performs a DB-4 write on the caller thread" — it queues the event
and returns a stub with empty `seq`/`entry_hash`. Any read issued immediately after an appending call races the
background writer.

**Fix:** the documented barrier `flush_audit_queue(5.0)` after each write burst (the same helper the writer
exposes for shutdown/tests) — explicit synchronization, not a sleep.

### F6 / F7 — deferred, pre-existing, out of F1–F4 scope (reported, **not** fixed)

* **F6 — `backend/scripts/entity_seed_test.py` fails:** `AttributeError: module 'src.risk_engine.models' has no
  attribute 'TransactionFeature'`; 3 checks fail (0 of 729 events seeded). The same missing attribute is logged
  by the risk-engine lifespan (`Entity tracker seeding skipped: …`). `backend/src/risk_engine/models.py` is
  **untouched** by this phase (`git diff` empty) and the table lives in `src/privacy_layer/models.py`
  (`git grep HEAD` shows privacy-layer references only), so the failure is pre-existing and independent of this
  phase. The suite is not registered in `FAST_TESTS`, `FULL_TESTS`, or `training_pipeline.ALL_SUITES`.
* **F7 — `regression_suite.py` parent-console encoding (Windows only):** when a child suite fails and prints
  non-ASCII (`✓`/`✗`/`→`), the runner prints the captured lines to a cp1252 console, raising
  `UnicodeEncodeError`; the runner's `except Exception` path then records a **second** result for that suite
  (0.0 s), producing `32/36 PASSED` and duplicate `FAIL` names. With `PYTHONIOENCODING=utf-8` the accounting is
  correct (`32/34`). CI (Ubuntu, UTF-8) is unaffected; `run_test` already decodes child output with
  `errors="replace"`, only the detail *print* is unguarded. Fix deferred (unrelated to F1–F4).

---

## 3. Files changed and rationale

| File | Change | Why |
|---|---|---|
| `backend/scripts/federated_worker.py` | −1 inline `import sys`, +4 comment lines | F1 fix (root cause). |
| `backend/scripts/federated_worker_test.py` | **new** (hermetic suite, 7 checks) | F1 regression: worker must start and answer the protocol on an all-`ML_FEATURES` CSV *and* on a reduced column set. |
| `backend/scripts/audit_test.py` | baseline-relative lengths (`STARTUP_N`, `N3`, `N5`), derived `SCORE_SEQ` tamper target, `flush_audit_queue` barriers | F2 fix (root cause) + F5 fix (determinism). |
| `backend/src/risk_engine/main.py` | +5 lines: `model_version = "unknown"` default before the metadata branch | F3 fix (root cause). |
| `backend/scripts/risk_engine_test.py` | +3 checks, isolated untrained-checkout boot | F3 regression. |
| `backend/scripts/audit_resilience_test.py` | §7: explicit retry-sweep suspension + `exactly-once` assertion | F4 fix (root cause of the flake), assertion retained. |
| `backend/scripts/regression_suite.py` | `federated_worker` registered in `FAST_TESTS` | Runs the F1 regression in CI's fast job. |
| `backend/scripts/coverage_run.py` | comment only | The Phase 13 exclusion rationale for F1/F2 was made stale by these fixes; now states the findings are fixed and the battery composition is deliberately unchanged. |
| `README.md` | suite counts 33→34 / 26→27 fast, fast-battery time and the CI-table entries, `risk_engine_test.py` 69→72 checks, coverage 23.6→24.0 % (line 26.0→26.4 %, branch 16.3→16.6 %), new suite in both lists | Phase 14's standard: documented numbers must match measured evidence; all of these were invalidated by this phase's changes. |
| `docs/evaluation/PHASE_DEFECT_REMEDIATION_CLOSEOUT.md` | **new** | This closeout. |

No other path was modified; `models/`, `data/`, `db/`, `.github/` and the security controls are untouched.

---

## 4. Regression tests added — pre-fix / post-fix results

| Test | Pre-fix | Post-fix |
|---|---|---|
| `backend/scripts/federated_worker_test.py` (F1) | **2 CHECK(S) FAILED, exit 1** — `rc=1 stderr_tail=UnboundLocalError: cannot access local variable 'sys'` (bug reintroduced verbatim to prove the test fails before the fix) | **7/7 PASS, exit 0** (`rc=0`, `n_train=28`, weight vector of full width 21, `[info] 3 features not in CSV, using 18 available`) |
| `audit_test.py` (F2), manifest present | PASS | **56/56 PASS, exit 0** (×3) |
| `audit_test.py` (F2), manifest absent | **11 CHECK(S) FAILED, exit 1** | **56/56 PASS, exit 0** (×3) |
| `risk_engine_test.py` untrained boot (F3) | `NameError: name 'model_version' is not defined` at `main.py:414` | **72/72 PASS, exit 0** (×2); untrained `/health` 200, `model_readiness=not_loaded`, `runtime_state=FAILED`, `model_version='unknown'`; trained boot still `seed42-transactions.csv` |
| §7 FIFO in `audit_resilience_test.py` (F4) | 12/12 non-ascending with a widened failure window | **0/12** with the fix (both windows); suite **92/92 PASS, exit 0** in 6 consecutive runs |
| F5 barrier | 1 failure in 6 comparable env-A runs | 6/6 clean runs (3 with manifest, 3 without) |

---

## 5. F4 repeated-run evidence

| Run type | Runs | Result |
|---|---|---|
| `audit_resilience_test.py` (plain) | 3 | 92/92 PASS, exit 0, FIFO seqs `[6, 7, 8, 9, 10, 11]` every time |
| `audit_resilience_test.py` under `coverage run --branch` (the Phase 13 flake condition) | 3 | 92/92 PASS, exit 0, FIFO seqs `[6, 7, 8, 9, 10, 11]` every time |
| Focused §7 harness, widened window, fix **absent** | 12 (×2 window sizes) | 12/12 and 12/12 non-ascending |
| Focused §7 harness, widened window, fix **present** | 12 (×2 window sizes) | 0/12 and 0/12 non-ascending; 0 incomplete recoveries |
| Focused §7 harness, default window, fix absent | 70 (4 variants) | 0 non-ascending — the default timing does not expose the race, consistent with Phase 13 |

Total: **6 full-suite runs, 0 failures**; **48 harness iterations** at the reproducing window (24 pre-fix
non-ascending, 24 post-fix ascending).

---

## 6. Full validation results

| Check | Command | Result |
|---|---|---|
| Fast battery | `python backend/scripts/regression_suite.py --fast` | **27/27 PASS, exit 0, 77.8 s** |
| Full battery | `PYTHONIOENCODING=utf-8 python backend/scripts/regression_suite.py --full` | **32/34 PASSED, 217.5 s**, exit 1 — see blocked below |
| Coverage battery | `python backend/scripts/coverage_run.py` | **28/28 PASS, exit 0**; TOTAL **24.0 %** (line 26.4 %, branch 16.6 %; `backend/coverage.xml` `line-rate="0.264"` `branch-rate="0.1658"`) |
| F1 targeted | `python backend/scripts/federated_worker_test.py` | 7/7 PASS, exit 0 |
| F2 targeted | `python backend/scripts/audit_test.py` (with / without manifest) | 56/56 PASS exit 0 in both |
| F3 targeted | `python backend/scripts/risk_engine_test.py` | 72/72 PASS, exit 0 |
| F4 targeted | `python backend/scripts/audit_resilience_test.py` | 92/92 PASS, exit 0 |
| Federated learning | `python backend/scripts/federated_test.py` | **ALL CHECKS PASSED, exit 0** (47 `[PASS]` lines) |
| Entity tracker | `python backend/scripts/entity_seed_test.py` | **exit 1 — F6, pre-existing, deferred** |
| Audit hygiene | `python backend/scripts/audit_hygiene_check.py` | **CLEAN, exit 0** (fixture snapshot exact, forks evidence-matched) |
| Secret hygiene + Bandit medium gate | `python backend/scripts/secret_hygiene_test.py` | **16 passed, 0 failed, exit 0** |
| Custom security scanner | `python backend/scripts/security_scan.py --full --json` | **passed = True**, total/critical/high/medium/low all **0**, exit 0 |
| Penetration test | `PYTHONIOENCODING=utf-8 python backend/scripts/penetration_test.py` | **exit 0** ("All executed attack vectors blocked"); exit 1 *without* the encoding variable (console only) |
| Claim evidence | `python backend/scripts/claim_evidence_check.py` | **PASS (22 claims verified)** |
| Eval-record schema | `python backend/scripts/eval_record_test.py` | **22/22 passed, 0 failed** |
| `py_compile` of every changed/new script | `python -m py_compile …` | PASS |
| README sanity after edits | fence balance / trailing whitespace | 64 fence lines (balanced), 0 trailing-whitespace lines |
| Exploratory (not wired into any suite/CI, **not** part of this phase's gates) | `test_rate_limiter.py` with Redis up | **ALL TESTS PASSED, exit 0** |

**Blocked / environmental (not code failures):**

* `security_test` — **exit 2, `ENVIRONMENTAL: 4 live check(s) not evaluated — identity service unreachable`**;
  the identity service on :8002 is not running in this session.
* `sql_injection` — **cannot run**: "Admin login failed — SQL injection tests cannot run (Start the front service
  on port 8000 first)".
* `test_redis_state.py` / `test_worker_pool.py` / `test_sentinel.py` — additional prerequisites are absent
  (a specific `models/production/altman_lgb_altman_clean_*/model.joblib`, a live worker on :8000, and a Redis
  **Sentinel** on :26379 respectively). These four inference scripts remain unwired into any suite or CI step;
  the `src/inference/*` coverage gap is unchanged.
* Standalone runs of `security_test.py`, `sql_injection_test.py` and `penetration_test.py` without
  `PYTHONIOENCODING=utf-8` fail with `UnicodeEncodeError` on a cp1252 console — a Windows-console artifact
  (CI is UTF-8); the verdicts above are from UTF-8 runs.

---

## 7. Coverage impact and limitations

* Battery composition is **unchanged** (27 fast + the hermetic `verification_test` = 28 scripts; the size grew
  only because `federated_worker_test.py` was added to `FAST_TESTS`).
* TOTAL rose **23.6 % → 24.0 %** (line 26.0 → 26.4 %, branch 16.3 → 16.6 %). The delta is real execution, not
  instrumentation: the new F3 checks boot the risk engine with no artifacts, exercising `main.py`'s
  metadata-missing branch, observability init and the model-load failure path that the warm tree never reached.
* Limitations unchanged and re-confirmed: the measurement covers the **fast** battery only; the live-service
  suites, the frontend (no runner) and `src/inference/*` (~2.7 k statements, needs Redis **and** further
  artifact/service prerequisites) stay outside it. The percentage remains a visibility instrument, not a
  quality gate.

---

## 8. Model and fraud-decision semantics were not changed

`git diff --stat backend/src` is a single file: `backend/src/risk_engine/main.py`, **5 insertions, 0
deletions**, containing only a default value for a version *string* used in telemetry/health. There is no
change to model architecture, weights, training, feature definitions, thresholds, rules, calibration, rate
limits, authentication, authorization or any other security control. `models/`, `data/` and the four SQLite
stores are untouched. Decision behavior is unchanged: the trained boot resolves the same
`model_version` (`seed42-transactions.csv`) and the untrained path only affects the reported identity of a
model that is not loaded (`model_readiness = not_loaded`), which it already reported before the fix. The only
other production-adjacent file touched is `federated_worker.py`, where the change removes a redundant import —
weight payloads, local standardization, feature selection and training are byte-for-byte the same code.

---

## 9. Remaining risks and deferred findings

1. **F6 (deferred)** — `entity_seed_test.py` fails on a pre-existing `TransactionFeature` model mismatch
   (`src.risk_engine.models` vs `src.privacy_layer.models`); 729 events seed 0. Needs a dedicated
   entity-tracker/models phase; **not** fixed here and **not** covered by any suite/CI job.
2. **F7 (deferred)** — `regression_suite.py`'s failure-detail print can raise `UnicodeEncodeError` on a non-UTF-8
   console and double-count a failed suite (`32/36` instead of `32/34`). Windows-only; CI unaffected.
3. **F2's regression home is not in the CI fast job.** `audit_test.py` remains in `FULL_TESTS` (a deliberate
   battery-composition decision, out of this phase's remit — Phase 15 was barred from coverage-methodology
   changes). Its protection today is the full battery run here and `training_pipeline.ALL_SUITES`; re-including
   it in the CI coverage battery is the natural follow-up.
4. **The writer's replay order is deliberately unchanged.** Recovery may append a backed-off head after later
   items; making replay strictly FIFO would introduce head-of-line blocking against the documented
   durability-first design (`MAX_RETRIES_PER_ITEM = 8`, backoff capped at 30 s). This phase made the *test*
   deterministic instead, and asserts the durable invariants (exactly-once recovery, chain verification)
   unconditionally. Any change to that trade-off is an architectural decision for a separate scoped phase.
5. **Live-service verification remains BLOCKED** in this environment (8000–8005 down), so `security_test`,
   `sql_injection` and the live walkthrough are **NOT ESTABLISHED** here.
6. The audit append remains on the detection critical path (unchanged, documented in the Phase 14 closeout).

---

## 10. Evidence labels

| Label | Items |
|---|---|
| **DEMONSTRATED** | F1/F2/F3/F4/F5 root causes and fixes (reproduced before, re-run after); 27/27 fast battery; 28/28 coverage battery; 32/34 full battery; 92/92 audit-resilience across 6 runs; 56/56 `audit_test` in both environments; 72/72 `risk_engine_test`; 7/7 new worker suite; claim-evidence PASS; audit hygiene CLEAN; secret hygiene 16/16; security scan 0 findings |
| **SELF-TESTED** | The regression suites themselves are the evidence for their own findings (plain-assert scripts; exit code authoritative) |
| **SIMULATED** | The F1 fixture CSVs, the F3 empty-artifact directory, and the F4 harness's DB-4 failure window are controlled synthetic fixtures |
| **NOT ESTABLISHED** | Live-service suites (`security_test`, `sql_injection`, walkthrough), the Docker path, `src/inference/*` behavior beyond the one exploratory Redis run |
| **BLOCKED** | Services 8000–8005 not running; no Redis Sentinel on :26379; `test_worker_pool.py` needs a live worker on :8000 |

---

## 11. Final CI/CD and Security Scan results

Recorded in the follow-up results commit (see the two-commit pattern used by Phases 13/14): CI/CD and Security
Scan workflows for the exact final SHA, with the interval status until the runs complete.
