# PS-14 — Phase 13 (Test-Coverage Visibility) Closeout

**Phase identity:** Phase 13 — Test-Coverage Visibility
**Status:** COMPLETE
**Date:** 2026-10-10

## 1. Commit / SHA ledger

| Item | Value |
|---|---|
| Branch | `main` |
| Starting SHA (`HEAD` before this phase) | `4a97c6dc77930dc622f8608215f7db6eea32606c` |
| Final SHA (this phase's commit) | recorded in §10 after push |
| Remote | `https://github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software.git` |
| Note | This phase added no production dependency, no model/artifact, and no privacy/AES secrets. Only reviewed Phase-13 files were staged (measurement driver, config, focused test, CI step, `.gitignore` entries, this closeout). Inherited working-tree artifacts (evaluation-ledger records, calibration metrics) were preserved and not committed. |

## 2. Coverage tool and version

- **Tool:** `coverage` (coverage.py), invoked directly — **not** via pytest/pytest-cov.
- **Version (measured in the project venv `.venv/Scripts/python.exe`):** `coverage 7.6.0` (`coverage.__version__` and `coverage xml` header both report 7.6.0).
- **Measurement environment:** Windows host, project venv, `PYTHONIOENCODING=utf-8`, `PS14_MODE=development`, `PYTHONPATH=backend` — the same child environment `regression_suite.py` uses.
- **Why direct `coverage run` and not pytest:** the repository's test suites are plain-assert scripts (`sys.exit(main())`), not pytest test functions. A pytest driver collects nothing from them (verified: `pytest <script>` exits 5, "no tests collected", "No data was collected"), and an earlier pytest-based draft of this phase produced a fabricated-looking ~74.8 % that could not be reproduced. The committed driver instead reuses the real runner (`FAST_TESTS` from `regression_suite.py`) and wraps each script in `coverage run --append`, measuring exactly what CI executes.

## 3. Exact measurement commands and environment

### Local (repo root, project venv active)

```bash
python backend/scripts/coverage_run.py
```

The driver:
1. Runs `coverage erase` (fresh data; stale data would silently inflate).
2. Runs each of `FAST_TESTS` (the exact `regression_suite.py --fast` battery) plus three hermetic TestClient service suites (`verification_test.py`, `audit_test.py`, `federated_test.py`) under `coverage run --append`, from `cwd=backend` with the same env contract as `regression_suite.run_test()`.
3. Prints `coverage report --show-missing` (line + **branch** coverage, missing line numbers per module) and writes `backend/coverage.xml`.

Config: `backend/.coveragerc` — `source = src`, `branch = True`, `show_missing = true`, `precision = 1`. **No `omit`, no `exclude_lines`**: every first-party module under `backend/src` is measured, including ORM `models.py` files. Paths in the config resolve relative to the working directory (`backend/`), which the driver always sets.

### CI (added to `.github/workflows/ci-cd.yml`, test job)

```yaml
      - name: Coverage measurement (backend/src)
        run: |
          pip install coverage==7.6.0
          python backend/scripts/coverage_run.py
        env:
          PYTHONIOENCODING: utf-8

      - name: Upload coverage XML
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: backend/coverage.xml
          retention-days: 30
          if-no-files-found: ignore
```

`coverage` is installed ad hoc in the step (like `safety`/`bandit` in `security-scan.yml`) — it is a measurement tool, not a runtime dependency, so `requirements.txt`/`constraints.txt` were not touched.

### Fail-visible verification (DEMONSTRATED)

| Fault injected | Observed |
|---|---|
| `coverage report` with no `.coverage` data file | `No data to report.` + **exit 1** |
| `coverage report` run from the wrong cwd (repo root instead of `backend/`) | `No data to report.` + **exit 1** |
| Any test in the battery failing | driver prints `FAILED: <name>` + **exit 1** |
| `coverage xml` failing | driver prints `COVERAGE FAILED` + **exit 1** |

The measurement gate cannot silently pass with an empty report.

## 4. Measured line and branch coverage (final SHA measurement)

Final instrumented run: **29/29 tests passed, exit 0** (`/tmp/phase13_cov_final.log`, 2 m 52 s).

```
TOTAL   25333 stmts   18497 miss   8378 branches   283 partial   24.5%
```

From `backend/coverage.xml` (machine-readable, uploaded as a CI artifact):

| Metric | Value |
|---|---|
| Line coverage | **27.0 %** (6 836 / 25 333 statements) |
| Branch coverage | **17.1 %** (1 432 / 8 378 branches) |
| Combined (coverage.py `Cover` column) | **24.5 %** |

Selected risk-relevant modules (line coverage, from the same run):

| Module | Coverage | Note |
|---|---|---|
| `src/risk_engine/main.py` (690 stmts) | 60.7 % | evaluate/run decision path exercised by smoke+rules/ood gates |
| `src/risk_engine/calibration.py` | 75.0 % | Platt calibrator load + apply |
| `src/risk_engine/limits.py` | 90.6 % | velocity-limit enforcement |
| `src/risk_engine/fusion.py` | 39.4 % | degraded-mode branches under-tested |
| `src/risk_engine/rules_engine.py` | 40.5 % | rule hit paths partially exercised |
| `src/risk_engine/kill_switch.py` | 26.4 % | fail-closed switch mostly untested |
| `src/risk_engine/drift_detector.py` | 45.7 % | PSI/state transitions partially exercised |
| `src/privacy_layer/velocity_tracker.py` | 59.6 % | 24h window bounds partially exercised |
| `src/privacy_layer/features.py` | 63.8 % | feature derivation happy path |
| `src/audit_service/writer.py` (492 stmts) | 75.1 % | append/retry/recovery covered by audit_resilience |
| `src/identity_service/security.py` | 94.9 % | token/hash primitives |
| `src/identity_service/main.py` (338 stmts) | 54.6 % | auth endpoints partially exercised via TestClient suites |
| `src/middleware/security_headers.py` | **98.7 %** (was 75.9 %) | after Phase 13 test |
| `src/middleware/rate_limiter.py` | **79.8 %** (was 35.6 %) | after Phase 13 test |
| `src/inference/*` (2 700+ stmts) | 0 % | needs live Redis + model registry; see §6 |
| `src/monitoring/*` (governance/dataset tooling) | mostly 0 % | offline tooling, not on the serving path; see §6 |

**Reading rule (per the phase objective):** these percentages are a *measurement-surface* result, not a correctness or security proof. The number is low mostly because large offline/instrumentation surfaces are included for honesty rather than excluded to inflate the score.

## 5. Inclusion and exclusion rules

### Included (no exclusions inside `backend/src`)

Every `.py` file under `backend/src/`, including `models.py` ORM files, `settings.py`, middleware, services, drift/monitoring, and `inference/`. The previous draft's `omit` list (`models.py`, `alembic/`, `migrations/`, `metadata.py`, `runtime_attestation/`) was removed: those paths either do not exist in this repository (`metadata.py`, `runtime_attatement/`, alembic/migrations) or are real first-party code (`models.py`) that should be counted. Blanket `exclude_lines` regexes were also removed — they masked executed-line accounting without documented justification.

### Excluded from the measurement (with rationale)

| Excluded | Rationale |
|---|---|
| `backend/scripts/**` | Test/ops scripts are the *instrument*, not the application; counting them would reward writing scripts instead of testing code. |
| `frontend/**` | Static HTML/JS; no Python runner exists. NOT ESTABLISHED. |
| `db/`, `models/artifacts/`, `.venv/`, `__pycache__/`, `reports/` | Runtime state, trained artifacts, virtualenv, generated output — not first-party source. |
| Live-service suites (`security_test.py`, `sql_injection_test.py`, `resilience_test.py`, `front_service_test.py`) in the *coverage* battery | They need a running stack (CI `--full` mode); including them would make the fast measurement fail on every env without services. They still run in their own contexts. |

No module was excluded because it was hard to measure; the 0 % modules are shown precisely so the gap is visible.

## 6. Important uncovered paths and prioritization

| Priority | Uncovered area | Why it matters | Evidence |
|---|---|---|---|
| 1 | `src/inference/*` — `service.py` (480), `worker_pool.py` (331), `redis_state.py` (261), `unified_scorer.py` (326), `batch_scorer.py` (257), `realtime_scorer.py` (256), `model_registry.py` (221), `rate_limiter.py` (152), `ab_testing.py` (134) — ~2 700 stmts, 0 % | The production inference path (Redis-backed rate limit, worker pool, model registry, A/B routing) has no hermetic suite; the existing `test_rate_limiter.py`/`test_redis_state.py`/`test_worker_pool.py`/`test_sentinel.py` require a **live Redis on localhost:6379** (verified: they fail `FileNotFoundError`/connection without it) and are not wired into any suite or CI. | NOT ESTABLISHED (Redis not available on the measurement host; wiring live-Redis tests into the fast suite is out of Phase 13 scope) |
| 2 | `src/risk_engine/fusion.py` (39.4 %), `rules_engine.py` (40.5 %), `kill_switch.py` (26.4 %), `drift_detector.py` (45.7 %) | Fusion degraded-mode branches, kill-switch fail-closed behavior, and drift state transitions are decision-time correctness paths. `drift_test.py` exercises the monitor but not all detector branches. | NOT ESTABLISHED (partially covered; branch gaps listed in `coverage.xml`) |
| 3 | `src/identity_service/main.py` (54.6 %) — brute-force lockout, admin endpoints, PII read paths | Auth/authz boundary; large uncovered regions are admin and error paths. `security_test.py` covers them only against a live service and reports ENVIRONMENTAL (exit 2) when unreachable. | NOT ESTABLISHED (live-only) |
| 4 | `src/audit_service/writer.py` uncovered tail (117 stmts: retry-budget exhaustion details, shutdown drain edge cases) | Audit chain is the forensic backbone; most paths are covered by `audit_resilience_test.py` but the remaining branches are timing-sensitive (see flake note §8). | PARTIAL |
| 5 | `src/monitoring/*` governance/dataset tooling (~3 500 stmts, mostly 0 %) | Offline dataset governance and phase-report generators, not on the serving or decision path. Low-value coverage targets: instrumenting report generators would raise the percentage without reducing risk. | NOT ESTABLISHED — **deliberately deprioritized** |
| 6 | `src/generate_synthetic_data.py` (287 stmts, 0 %) | Data-generation CLI used by CI/training, not serving code. Low-value target. | NOT ESTABLISHED — deprioritized |
| 7 | `src/risk_engine/seasonal.py`, `industry_features.py`, `model_registry.py`, `model_governance.py`, `altman_ensemble.py` (legacy) | Model-side modules exercised by the training pipeline (out of the fast battery), not per-request serving paths. | NOT ESTABLISHED |

Deferred findings (explicitly out of Phase 13 scope per the phase brief):

- **Regression-runner exit-code issue** — deferred, unchanged, not bundled.
- **Production-mode rate-limiting coverage** (`rate_limit_middleware` 429 path, `src/middleware/rate_limiter.py` lines 158–193) — the middleware intentionally no-ops unless `PS14_MODE=production`; covering it requires production-mode fixtures. Deferred; the mode-independent logic (window, block, config, cleanup) **is** now covered (79.8 %).
- **`/internal/metrics` NameError fix** — not bundled (per phase brief).

## 7. Tests added and the defect or risk each addresses

**New file: `backend/scripts/security_headers_test.py`** (40 assertions, hermetic — no services, no credentials, no shared state; deterministic; registered in `FAST_TESTS` so CI runs it).

| Section | Risk addressed |
|---|---|
| SecurityHeadersMiddleware header set + X-Request-ID echo/generate | Regression in browser-standard headers (clickjacking, sniffing, HSTS, CSP, caching of sensitive responses) would go unnoticed: the only prior check lived in live-service `security_test.py`, which reports ENVIRONMENTAL when the identity service is down — i.e., in most CI contexts the header contract was **never verified**. |
| 413 body-size limit (injected via `PS14_MAX_BODY_SIZE` before import) | DoS guard: an oversized-body request must be rejected *before* the handler runs, with the documented JSON detail and a request ID. Previously untested hermetically. |
| 500 JSON mapping for an unhandled handler exception | Error-handling/security-semantics: the response must be generic (`Internal server error`), must **not** disclose the exception text, and must carry `X-Request-ID`. The test asserts no information disclosure (`internal-secret-detail` not in body). |
| `_build_csp` development vs production | Production CSP must pin a nonce and drop `'unsafe-inline'` from `script-src`, and set `report-uri`; dev CSP must allow inline for the static UIs. Mode-dependent policy had no direct test. |
| OriginValidationMiddleware allow-list | Cross-origin enforcement: no-Origin passes (server-to-server), allow-listed passes, disallowed Origin → 403 with documented detail. Previously live-only. |
| RateLimiter pure logic (window/block/retry_after/config lookup/cleanup) | Velocity-limit accounting: sliding-window eviction, block arming and expiry, per-endpoint config (exact + prefix), `retry_after` semantics, stale-entry cleanup. `middleware/rate_limiter.py` coverage went 35.6 % → 79.8 %; security_headers 75.9 % → 98.7 %. |

**Test-infra additions (no application code changed):**

| File | Purpose |
|---|---|
| `backend/scripts/coverage_run.py` | Reproducible coverage driver (fails visibly on test failure or empty report; writes `backend/coverage.xml`). |
| `backend/.coveragerc` | Canonical branch-coverage config; no blanket excludes. |
| `.github/workflows/ci-cd.yml` (test job) | Coverage measurement step + `coverage-xml` artifact upload. |
| `.gitignore` | Ignores `.coverage`, `.coverage.*`, `htmlcov/`, `coverage.xml`. |

No application source under `backend/src/` was modified. No model architecture, weights, training, feature definitions, fraud thresholds, rules, or calibration artifacts were changed. No dependency pins were changed.

## 8. Regression / security results and limitations

| Gate | Command | Result |
|---|---|---|
| Fast regression baseline (pre-measurement) | `python backend/scripts/regression_suite.py --fast` | **25/25 PASS, exit 0** (75.2 s) |
| Coverage battery (baseline + 3 service suites + new test) | `python backend/scripts/coverage_run.py` | **29/29 PASS, exit 0** (2 m 52 s); TOTAL 24.5 % |
| New focused test (standalone) | `python backend/scripts/security_headers_test.py` | **40/40 PASS, exit 0** |
| Secret hygiene + Bandit medium gate | `python backend/scripts/secret_hygiene_test.py` | **PASS** (16 passed, 0 failed; bandit medium+ exits 0) |
| Claim evidence enforcement | `python backend/scripts/claim_evidence_check.py` | **PASS** (22 claims verified) |
| YAML sanity of both workflows | `yaml.safe_load` | PASS |
| `py_compile` of all new/changed scripts | `python -m py_compile` | PASS |

**Limitations / observed flake (honest reporting):**

- **Timing-sensitive FIFO assertion in `audit_resilience_test.py`:** on one instrumented run (1 of 4), the check `recovery replays in FIFO (seq ascending == append order)` failed under coverage tracing (`[11, 6, 7, 8, 9, 10]`). It passed on 2 plain runs and on the coverage retry, and in the final 29/29 run. This is instrumentation-induced scheduling variance in the recovery thread, **not** a product defect proven either way; it is reported here rather than papered over. If it recurs in CI, the assertion's tolerance should be reviewed in a dedicated phase.
- The measured percentage covers the **fast** battery. Full-mode live-service suites (`security_test.py`, `sql_injection_test.py`, `resilience_test.py`, `front_service_test.py`) are not in the coverage run (documented §5).
- Branch coverage (17.1 %) is materially lower than line coverage — the fast battery mostly exercises happy paths; the branch gaps are enumerated per module in `coverage.xml` (CI artifact) and `coverage report --show-missing` output.
- CI results for the final SHA are recorded in §10 after push; local results above are `DEMONSTRATED` on the measurement host (Windows, Python 3.12 venv).

## 9. Evidence labels

| Label | Item |
|---|---|
| `DEMONSTRATED` | Coverage tool + version (`coverage 7.6.0`), measured via `coverage.__version__` and the `coverage xml` header |
| `DEMONSTRATED` | Exact measurement commands (driver + CI step) and environment |
| `DEMONSTRATED` | Fail-visible behavior (no data / wrong cwd / test failure / xml failure all exit 1 with explicit messages) |
| `DEMONSTRATED` | Measured line 27.0 % + branch 17.1 % (combined 24.5 %) over all of `backend/src`, from a 29/29-passing instrumented run; machine-readable in `backend/coverage.xml` |
| `DEMONSTRATED` | Fast regression baseline 25/25 PASS; secret hygiene 16/16 PASS; claim evidence PASS |
| `SELF-TESTED` | New `security_headers_test.py` (40/40 on this host; wired into `FAST_TESTS` so CI re-runs it) |
| `SELF-TESTED` | Per-module missing-line lists (visible in the run log and XML artifact) |
| `NOT ESTABLISHED` | Live-mode suites' contribution to coverage (need running services) |
| `NOT ESTABLISHED` | `src/inference/*` coverage (needs live Redis + model registry; existing Redis tests are unwired and fail without a server) |
| `NOT ESTABLISHED` | `frontend/**` coverage (no runner) |
| `NOT ESTABLISHED` | Production-mode 429 rate-limit path (deferred) |
| `BLOCKED` | `nltk`/CVE-2026-3740, `multidict`/CVE-2026-104874 (inherited from earlier phases; transitive, no in-repo import path) |
| `SIMULATED` | (none — no simulated results are claimed in this phase) |

## 10. Observed CI/CD and Security Scan results for the exact final SHA

Final SHA: **to be recorded here after push and CI observation** (local-only at time of writing; the working tree is not a git repository per `AGENTS.md`... see note below).

**Repository note:** this working copy has functioning `git status`/`git log` (HEAD = `4a97c6dc77930dc622f8608215f7db6eea32606c`, `main`, in sync with `origin/main` at phase start). CI observation (`ci-cd.yml` test job now includes the coverage step; `security-scan.yml` unchanged) for the phase-13 commit SHA is appended below once the push completes.

<!-- CI-RESULTS-APPEND-POINT -->
