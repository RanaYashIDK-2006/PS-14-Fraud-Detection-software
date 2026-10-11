# Phase 21 Closeout — Demo Hardening and End-to-End Verification

**Status: DEMONSTRATED (synthetic, local).** Real-world fraud-detection
validation remains **NOT ESTABLISHED**; institutional dataset validation
remains **BLOCKED_PENDING_ELIGIBLE_DATASET** (README L506; claims registry
C-010/C-013/C-014). This phase does not establish production readiness.

## 1. Git

| | SHA |
| --- | --- |
| Starting HEAD | `ac9f7aec89a3e279c5d3656868ac19063d04fa6c` (= origin/main at phase start) |
| Phase 21 implementation commit | `8d36c52790603f936acb599933dfd8be99b339f3` — `fix(demo): Phase 21 — restore the CI regression gate, fix the stale backup/restore assertion, add batch failure-recovery tests` (7 files, +997/−9) |
| Final HEAD | the `docs(phase21)` commits that record the CI results and these SHAs (this file), direct children of `8d36c52` |
| Remote | `origin/main`, verified via `git ls-remote` after each push |

Branch `main`, tracking `origin/main`, in sync at phase start.

**Inherited dirty files preserved, never staged:** modified
`backend/scripts/test_worker_pool.py`, `backend/src/inference/service.py`,
`backend/src/privacy_layer/main.py`, `backend/src/risk_engine/main.py`,
`docs/evaluation/PHASE_INFERENCE_COVERAGE_REDIS_INTEGRATION_CLOSEOUT.md`,
`reports/calibration_test/calibration_metrics.json`,
`reports/evaluation_runs/eval_ledger.jsonl`; untracked Phase 18 files
(`PHASE_18_COMMIT_PLAN.md`, `PHASE_18_FINAL_DELIVERY_SUMMARY.txt`,
`phase18_demo_error_hygiene_test.py`, `PHASE_18_DEMO_STABILITY_CLOSEOUT.md`)
and all `reports/evaluation_runs/record_eval-*.json` artifacts.

## 2. Regression-gate repair (Part 2)

**Cause (established, not assumed):** `regression_suite.py:82` references
`scripts/inference_hermetic_test.py`. `git log --all -- <file>` is empty — the
file was **never committed**; the reference was added in Phase 17
(`9291145`). The file on disk is the genuine Phase 16 hermetic suite (405
lines: RedisSlidingWindow/AutoScaler/WorkerInfo/AB-splitter/UnifiedScorer
via fakeredis and in-process fakes, no live services), consistent with the
suite entry's own comment. Provenance established → restored by committing it
unchanged (0 lines edited).

- fakeredis is optional by design (same skip pattern as the already-committed
  `test_redis_state.py`); 14 non-fakeredis tests pass without it. fakeredis
  2.37.1 is present in the local venv.
- `PYTHONPATH=backend python backend/scripts/inference_hermetic_test.py` →
  **20/20 PASSED, exit 0**.
- CI invocation confirmed: `ci-cd.yml` step "Run full regression suite" runs
  `python backend/scripts/regression_suite.py --fast` from repo root — the
  missing file was the cause of the red regression job on `1b14a42`/`7ce38a6`.
- No suite invocation was removed or weakened.

## 3. Backup/restore failure (Part 3) — root cause FIXED

Reproduced exactly (28/29 fast suite; the test itself 70/71): only
`Restore preserves all five quarantined forks ([])` failed; every
backup/copy/checksum/integrity/restore/originals-untouched check passed.

**Safety assessment before running:** the test copies databases into a temp
dir (`ps14_brt_*`), corrupts only the copies, restores copies, and removes the
temp dir in a `finally`; live databases receive only a WAL checkpoint and an
audit-queue flush; originals are verified unchanged afterward (historical-
prefix comparison tolerates live appends). Isolated enough to run.

**Root cause (traced, read-only first):** the five documented 2026-09-19 fork
sequences (731/735/740/745/750) still exist in the live chain, but the chain
has since been **legitimately repaired** — `evaluate_chain` over the live
`db/audit.db` returns `strict_ok=True`, `first_bad=None`, hence
`quarantined_breaks=[]`. The check hard-coded the assumption that those forks
must still be live breaks in any faithful backup — a stale environmental
assertion, not a backup/restore defect.

**Fix (evidence-supported, no weakening):** the check now asserts the real
fidelity property — the restored chain's evaluation (ok, strict_ok,
quarantined_breaks) must equal the source copy's evaluation, whatever that
is — plus a check that the five frozen findings remain loadable
(`AFFECTED_SEQUENCES` intact, evidence preserved). A corrupted or modified
restore still fails closed.

- Focused rerun: **72/72 PASS, exit 0** (row count 2249 vs 2245 earlier —
  live audit appends between runs, correctly handled by the prefix check).

## 4. End-to-end user-facing demo, browser-level (Part 4)

Genuine browser session (Chromium tab driven by the agent, real HTTP to the
live stack), plus a screenshot of the rendered report:

| Step | Result |
| --- | --- |
| Landing page `http://127.0.0.1:8000/` | 200, renders; service-readiness block live from each service's `/health` (identity/privacy/risk/verify/audit all "alive"; risk "ready · model loaded"; latencies shown) |
| Navigate to `/demo-page` | 200, tabs Overview/Reports, run configuration (25,000 rows, seed 42, ps14-features-v1), execution-mode selector |
| Readiness vs actual | matches live services (verified against `/demo/readiness`: `demo_path_ready=true`, risk model loaded) |
| Click "Run 25,000-Transaction Demo" | browser-issued POST → 202, run started |
| Progress/status | `/demo/batch/progress` reported the active run then the completed summary |
| Completion | **25,000 processed / 0 failed / 0 skipped**, reconciles exactly |
| Auto-opened report | Reports tab showed the fresh report: decision counts reconcile (24,188 + 640 + 172 = 25,000; table total row 25,000/100%), confusion matrix sums to 25,000 (317+182+495+24,006), metric labels ("synthetic ground truth from this repository's generator", "Flagged means decision != 'allow'"), dataset provenance (SHA-256, seed 42, noise 0.15), model identity (FusionEngine, artifacts fingerprint sha256:b9252ee6…), execution mode label, limitations block, report history table (all 3 prior reports still listed — nothing deleted) |
| CSV export | 112 lines; `field,value` header; `validation.rows_*` and `summary.decision_counts.*` agree with the report |
| JSON export | equals the stored report document |
| Unknown report id | 404 `{"detail":"Report not found"}` |
| Bad export format | 422 `{"detail":"format must be json or csv"}` |
| Failure states | see Part 5 — crashed runs recorded `failed`, never `completed` |

Fresh acceptance run: **`RPT-20261011T071635Z-9aec70`**
(`db/demo_batch/runs/RPT-20261011T071635Z-9aec70/report.json`, gitignored
runtime data). 25,000/25,000/0 · allow 24,188 (96.75%) · verify 640 (2.56%) ·
step_up 172 (0.69%) · 0 degraded · 178 domain-shift · throughput 9,522.8
rows/s · end-to-end 5.28 s · ROC-AUC 0.9504 · PR-AUC 0.6693 · precision 0.390
· recall 0.635 · F1 0.484 · dataset SHA-256
`7538a8ddaac89dd50de74e92f4b4a7023a2d881dc82de4c218dee18d2eb99296` ·
reproducibility git SHA `ac9f7ae` (pre-Phase-21 HEAD; the seed/noise/
generator are unchanged, so a rerun on the new HEAD yields the same
decisions). Execution mode: **offline batch** (in-process FusionEngine +
RulesEngine + live decision policy) — not described as a live service test.

No existing reports or DB-3 rows were deleted; the acceptance run is a fresh
report id alongside the history.

## 5. Failure recovery & repeatability tests (Part 5)

New hermetic suite `backend/scripts/phase21_batch_recovery_test.py`
(**56/56 PASS, exit 0**) — the batch surface had no failure-path coverage
before (Phase 20's suite covers only the single-transaction path). Fake
subprocess boundary, temp runs dir, no live services:

1. invalid launch params (bad mode/rows/seed) → 422, **zero** subprocesses
   spawned;
2. duplicate submission while a run is active → 409 naming the active report
   id, no second spawn;
3. progress retrieval reflects the real active run (processed/total/state);
4. **false-success invariant**: a runner dying with exit 1 and no terminal
   state is rewritten to `state=failed` with the exit code, and gets no
   `report.json`; a run that completed before dying stays `completed`;
5. failed runs appear in the listing as `failed`/partial and their id is an
   honest 404 — no partial report is ever served as complete;
6. retrieval + CSV/JSON export fidelity (flattened fields agree with the
   stored report; download filename set); unknown format 422, unknown id 404,
   traversal/malformed ids rejected without serving content;
7. error hygiene: no internal token, no traceback, no absolute local path in
   any error or export body;
8. batch rate limit: 6/300 s → 429 with an understandable message.

Registered in `regression_suite.py` FAST_TESTS as `phase21_batch_recovery`.

Two initial test-assertion failures were diagnosed as test strictness, not
product defects, and corrected in the test: `csv.writer` emits RFC-4180
CRLF; encoded-slash traversal never matches the Starlette route (safe 404
before the handler). The product behavior was verified in both cases.

## 6. Security & dependency checks (Part 6)

| Check | Command / source | Result |
| --- | --- | --- |
| Bandit medium+ (CI gate) | `bandit -r backend/src --severity-level medium` | **exit 0**, 0 medium/high findings |
| Secret hygiene | `secret_hygiene_test.py` | **16/16 PASS, exit 0** |
| Custom scanner | `security_scan.py --full --json` | **exit 0** |
| pip-audit | `pip_audit` on the local venv | exit 1 — 25 advisories in 6 packages (see below) |
| CI Security Scan (workflow) | run #180 on `8d36c52` (Phase 21 SHA) | **PASS** |
| CI/CD (workflow) | run #175 on `8d36c52` (Phase 21 SHA) | **PASS** — all 5 jobs green (Test Suite, Security Scan, Docker Build, Integration Test, Deploy Image); the regression gate that failed on `1b14a42`/`7ce38a6`/`9d9c598`/`ac9f7ae` is verified fixed in CI |
| (pre-phase context) CI/CD #172 on `7ce38a6` | | FAIL — the regression gate, root-caused and fixed in §2 |
| (pre-phase context) Security Scan #177 on `7ce38a6` | | PASS |

pip-audit finding, investigated: the local venv has **drifted** from the
pinned set (venv PyJWT 2.13.0 vs pinned `PyJWT==2.15.1`; the requirements
file itself documents this drift and pins the patched set). CI installs the
pinned `requirements.txt`+`constraints.txt`, so the advisories above are
**local-environment drift, not the pinned dependency set**; CI's pip-audit
and Security Scan passed on `7ce38a6`. No dependency was upgraded in this
phase (pin changes require re-running the full battery, per requirements.txt
rationale).

**Demo-surface exposure review:** report exports previously carried absolute
filesystem paths including the OS username (`execution.engine.artifacts_dir`,
`dataset.selection.source_csv`, `progress.report_file`). Fixed in the runner:
all three now emit repo-relative POSIX paths (`models/artifacts`,
`db/demo_batch/transactions_causal.csv`). Verified end-to-end on a fresh
500-row run: no backslashes, no username anywhere in the report JSON.
Error responses verified free of credentials, tokens, tracebacks and paths
(Part 5 §7 plus the Phase 20 suite's token-leak checks).

Note: a `dependency-review` workflow does not exist in this repository (no
match in `.github/workflows/` or git history); the earlier-phase mention of a
"dependency-review failure" does not correspond to a workflow in this repo
and is recorded here as NOT ESTABLISHED.

## 7. Test results summary (all executed this phase, repo root, `.venv`
Python 3.12.10, Windows)

| Command | Result |
| --- | --- |
| `regression_suite.py --fast` (before fix) | 28/29 — only backup_restore |
| `regression_suite.py --fast` (after fixes, now 30 suites) | **30/30 PASS, exit 0** (96 s) |
| `inference_hermetic_test.py` | 20/20, exit 0 |
| `backup_restore_test.py` | 72/72, exit 0 |
| `phase21_batch_recovery_test.py` | 56/56, exit 0 |
| `phase20_demo_usability_test.py` | 51/51, exit 0 (regression guard) |
| `phase19_e2e_test.py` | 0 failed, exit 0 (regression guard) |
| `secret_hygiene_test.py` | 16/16, exit 0 |
| bandit medium+ | exit 0 |
| `security_scan.py --full` | exit 0 |
| `node --check` on touched JS | n/a this phase (no JS changed) |

Live-stack checks: all six services healthy (`:8000`–`:8005`); the browser
session in §4 exercised the full user path against them.

## 8. Files changed in the Phase 21 commit

| File | Change |
| --- | --- |
| `backend/scripts/inference_hermetic_test.py` | restored to git (committed byte-identical; never committed before) |
| `backend/scripts/backup_restore_test.py` | replaced the stale five-forks assertion with the source-vs-restored fidelity check + findings-loadable check |
| `backend/scripts/phase21_batch_recovery_test.py` | new hermetic batch failure-recovery suite (56 checks) |
| `backend/scripts/demo_batch_runner.py` | repo-relative paths in report metadata (`_rel` helper; 3 call sites) |
| `backend/scripts/regression_suite.py` | registered `phase21_batch_recovery` in FAST_TESTS |
| `frontend/index.html` | CI status table: Test Suite PASSING (30/30, gate fixed); Docker/Integration ON PUSH |
| `docs/evaluation/PHASE_21_DEMO_HARDENING_CLOSEOUT.md` | this file |

## 9. Limitations & blocked checks

- **Real-world validation: NOT ESTABLISHED.** The acceptance-run metrics are
  on synthetic generator labels from the same pattern family as training
  data — optimistic versus out-of-distribution data (in-report limitations
  state this).
- **Institutional validation: BLOCKED_PENDING_ELIGIBLE_DATASET.**
- pip-audit on the *local* venv fails due to documented venv drift from the
  pinned set; the pinned set is what CI audits and it passes. Re-syncing the
  local venv is deferred (requires full-battery re-run per pin rationale).
- CI/CD #175 and Security Scan #180 on the Phase 21 SHA `8d36c52` both
  PASS (recorded post-push in this evidence-reconciliation edit); the
  historical runs on earlier SHAs remain as they were (CI/CD FAIL on the
  regression gate, Security Scan PASS).
- Docker build/integration jobs were not run locally; they run on push.
- `--full` (live-service) suites were not run this phase; the Phase 19 live
  e2e demo and this phase's browser session cover the live path.
- Screenshot evidence exists for the rendered report (agent browser); no
  video recording was captured.

*Every number above comes from a recorded command output, report artifact,
or rendered page observed during this phase.*
