# Phase 22 Closeout — Final Demo Acceptance and Rehearsal

**Final verdict: `PASS WITH LIMITATIONS`** — the user-facing demo is verified
end-to-end (startup → batch → report → export) in a real browser against the
running stack, with fresh evidence from this phase. The limitations are
recorded in §9; none sit on the demo path.

```
REAL_WORLD_VALIDATION  = NOT ESTABLISHED
INSTITUTIONAL_VALIDATION = BLOCKED_PENDING_ELIGIBLE_DATASET
```

Evidence labels used below: `DEMONSTRATED` (live, observed), `SELF-TESTED`
(hermetic/self-reported suites), `SIMULATED` (synthetic-data results),
`NOT ESTABLISHED`, `BLOCKED`.

## 1. Git and baseline preservation

| | SHA |
| --- | --- |
| Starting HEAD (= origin/main at phase start) | `82a2e9bb35b9130c9b6dbf0aaf97588adf25d998` |
| Final HEAD | this closeout's commit, direct child of `82a2e9b` (see `git log`) |

Branch `main`, tracking `origin/main`, in sync at start; verified via
`git ls-remote` before push.

**Baseline working-tree preservation record:** 133 dirty entries at phase
start (7 modified tracked files: `test_worker_pool.py`,
`inference/service.py`, `privacy_layer/main.py`, `risk_engine/main.py`,
`PHASE_INFERENCE_COVERAGE_REDIS_INTEGRATION_CLOSEOUT.md`,
`calibration_metrics.json`, `eval_ledger.jsonl`; 126 untracked: Phase 18
files, `record_eval-*.json` artifacts, etc.). **All 133 preserved** — none
staged, modified, or deleted; count re-verified after commit. No git
reset/clean/stash/restore was run. Historical reports and DB records were
never modified (the batch history grew only by genuine new runs).

## 2. Startup reproduction (documented runbook) — `DEMONSTRATED`

Environment: Windows, `.venv/Scripts/python.exe` → Python 3.12.10, pip 26.2.1.

Procedure followed exactly from `docs/demo/DEMO_RUNBOOK.md`:

1. **Stop** (runbook shutdown section): `netstat -ano | grep :800[0-5]` → six
   PIDs → `taskkill //PID <pid> //F` → 0 listeners remain. No manual
   intervention needed; the documented Git-Bash `kill` caveat is accurate.
2. **Start**: `powershell -NoProfile -ExecutionPolicy Bypass -File
   .freebuff/start_stack.ps1` → exit 0; six services announced
   (front :8000, identity :8001, privacy :8002, risk :8003, verify :8004,
   audit :8005). Risk listens a few seconds later (model load).
3. **Readiness**: all six `/health` → 200. Risk reports
   `readiness: ready, model_readiness: loaded, model_id:
   altman_native_E_hardneg_cert_20260904`. `GET /demo/readiness` →
   `demo_path_ready: true`, all alive, per-service latencies shown.

Startup failures/manual interventions: **none**. Fresh-environment
reproduction (empty venv/`db/`) is `BLOCKED` by design — demo stores and the
venv are inherited local state; the restart-from-documented-commands test
above is the reproducible-startup evidence.

## 3. Browser acceptance — `DEMONSTRATED` (real Chromium, live HTTP)

Primary URL `http://127.0.0.1:8000/demo-page`; landing demo section
`http://127.0.0.1:8000/#demo` — verified independently:

| Check | Result |
| --- | --- |
| Landing page `http://127.0.0.1:8000/` | renders; "Demo path ready" badge; live readiness block (identity/privacy/risk/verify/audit with latencies); synthetic-data disclaimer (screenshot 1) |
| Demo link navigation | hero "Run the Demo" and nav "Live Demo" both scroll to the rendered `#demo` section; URL updates to `#demo` (screenshot 2) |
| `#demo` single-transaction demo | **live run returned a real backend response**: decision `verify`, band `high`, ML 0.0073, rule 1, `DAILY_SPEND_*` limit trigger (repeat-run account history, as the runbook documents), event id `DEMO-…`, degraded false |
| `/demo-page` opened directly | renders: run configuration (25,000 / seed 42 / ps14-features-v1), execution-mode selector labelled "Offline batch — in-process PS-14 engine + live decision policy" (accurate), service-readiness block matching `/demo/readiness` exactly (screenshot 3) |
| States verified | ready (verified), running + completed (during the §4 run), failure/empty (via §5 live checks + hermetic suites) |
| Console errors | **one real defect found and fixed**: `index.html` referenced the AOS stylesheet from unpkg.com, which this repo's own CSP (`security_headers`) refuses on every load — a permanent console error and a dead resource. Removed the blocked link only (script + `data-aos` attributes untouched; visual behavior unchanged since the CSS never applied). After fix: console clean — only the informational Tailwind-CDN dev warning remains |
| Failed network requests | none — all requests (readiness polling, page assets) 200 |

Screenshots captured (3): landing page, `#demo` section, `/demo-page`
overview; plus the rendered fresh report (screenshot 4).

## 4. Fresh 25,000-transaction rehearsal — `DEMONSTRATED` + `SIMULATED`

Launched from the browser UI ("Run 25,000-Transaction Demo", offline mode)
on the freshly restarted stack.

**Report ID: `RPT-20261011T083933Z-48bd92`**
Artifact: `db/demo_batch/runs/RPT-20261011T083933Z-48bd92/report.json`
(gitignored runtime data, served via `GET /demo/reports/{id}`).

| Field | Value |
| --- | --- |
| State / rows | completed · 25,000 requested / 25,000 processed / **0 failed / 0 skipped** / 0 missing-feature rows |
| Decisions | allow 24,188 (96.75%) · verify 640 (2.56%) · step_up 172 (0.69%) — **sums to 25,000** |
| Runtime / throughput | 5.148 s end-to-end · 9,682.6 rows/s scoring · 0 degraded rows |
| Dataset | synthetic, repository generator (`--causal`); seed **42**, noise 0.15; schema `ps14-features-v1`; selection `pandas.sample(n=25000, random_state=42)` without replacement; **SHA-256 `7538a8ddaac89dd50de74e92f4b4a7023a2d881dc82de4c218dee18d2eb99296`** |
| Model / mode | FusionEngine, artifacts fingerprint `sha256:b9252ee676c38d312b67471f892265e7`; **offline batch** (in-process engine + live decision policy) |
| Code SHA | `82a2e9b` (recorded in the report's reproducibility block) |
| Metrics (synthetic labels) | confusion tp 317 / fp 495 / tn 24,006 / fn 182 · precision 0.390 · recall 0.635 · F1 0.484 · ROC-AUC 0.9504 · PR-AUC 0.6693 |

**Determinism:** dataset SHA-256, decision counts, and confusion matrix are
**identical** to the Phase 21 run on the same seed/artifacts — repeatability
`DEMONSTRATED`. The report labels the labels as "synthetic ground truth from
this repository's generator", carries the circularity note and 3 limitations
items, and states this is not real-world validation.

**Consistency:** UI (screenshot 4) = stored JSON (fetched) = CSV export
(field-by-field: report_id, state, dataset SHA-256, rows_scored, all three
decision counts, confusion cells all agree). JSON export equals the stored
document.

**History intact:** 5 reports total — the 3 Phase 20/21 runs, the fresh
Phase 22 acceptance run, plus a genuine 100-row rehearsal run
(`RPT-20261011T084242Z-766170`, completed, used for the duplicate-launch
check). Nothing deleted; DB-3 untouched by this phase.

## 5. Failure handling and report integrity — `DEMONSTRATED` + `SELF-TESTED`

Reran the existing acceptance suites first (both green post-sync), then
live-checked the same behaviors against the running stack:

| Check | Result |
| --- | --- |
| `phase20_demo_usability_test.py` | **51/51, exit 0** |
| `phase21_batch_recovery_test.py` | **56/56, exit 0** (includes crashed-run false-success invariant, export fidelity, error hygiene) |
| Live invalid batch request (bad mode; rows 999999) | 422 / 422, no run started |
| Live duplicate launch (two rapid POSTs) | first 202, second **409** naming the active report id |
| Live unknown report id | 404 honest not-found |
| Live bad export format (`format=pdf`) | 422 "format must be json or csv" |
| Failed/interrupted run never reported successful | hermetic invariant (exit-code≠0 without terminal state → `failed`, no report.json) |
| Error responses leak nothing | token/traceback/absolute-path checks in both suites pass; live bodies verified |
| Report history available | 5/5 listed after all runs |

No destructive or state-mutating tests were run against valuable data; the
suites use temp dirs and fake subprocess boundaries by construction.

## 6. Dependency and security audit

**Mismatch investigated (§6 requirement):** active interpreter `.venv`
(Python 3.12.10). Authoritative pins: `backend/requirements.txt` +
`backend/constraints.txt`. Established: **environment drift**, not a pin
defect — 31 pinned packages had drifted to older versions than the pins
(PyJWT 2.13.0 vs pinned 2.15.1; fastapi 0.141.1 vs 0.142.2; numpy, pandas,
sqlalchemy, starlette, uvicorn, cryptography …) plus 2 pinned packages
missing entirely (cloudpickle, opentelemetry-api). The pinned set is
documented as a freeze of what CI validates.

**Synchronization (documented project procedure):**
`python -m pip install -r backend/requirements.txt -c backend/constraints.txt`
→ exit 0; all 33 packages now at pinned versions. No pin was altered.

**Re-audit:** `pip_audit` on the synced venv — the 14 PyJWT advisories are
gone. **Residual (11 advisories in 5 packages: multidict, nltk, pytest,
setuptools, urllib3):** none is directly pinned and **none is on the runtime
import path** — they are transitive dependencies of local-only venv tooling
(aiohttp/yarl, safety, pytest-cov, kaggle/requests, torch) that a fresh CI
install never contains. Fix versions recorded (nltk: no fix published).
Remediation needed = remove the unused local extras from the venv or re-pin
after the full battery; **deferred** — uninstalling inherited tooling is
outside this phase's authority. `pip-audit exit 1 (local extras)` is the
honest state; CI's audit of the pinned set passes.

**Post-sync battery (all local, repo root, venv Python 3.12.10):**

| Check | Result |
| --- | --- |
| `regression_suite.py --fast` (30 suites) | **30/30 PASS, exit 0** (88.6 s) |
| `secret_hygiene_test.py` | 16/16, exit 0 |
| `bandit -r backend/src -f json --severity-level medium` (CI command) | **exit 0, 0 results** (JSON-verified; the text-mode Medium/High numbers are by-confidence counts of low-severity issues excluded by the severity filter) |
| `security_scan.py --full` | "No security issues found", exit 0 |

**CI (workflow runs on exact SHAs):** CI/CD **#177** and Security Scan
**#182** on the starting SHA `82a2e9b` — both **PASS** (continuing the green
streak from #175/#180 on the Phase 21 SHA). Local results above are
distinguished from these CI results. Fresh runs on this commit are triggered
by the push; the working tree carries no workflow changes.

## 7. Operator rehearsal checklist (every item tested this phase)

- [x] **Start the application using documented commands** — runbook stop +
      start sequence executed verbatim; exit 0; six listeners.
- [x] **Confirm service and model readiness** — all `/health` 200; risk
      `ready · loaded` (`altman_native_E_hardneg_cert_20260904`);
      `/demo/readiness` `demo_path_ready: true`.
- [x] **Open the correct demo URL** — `/#demo` (landing) and `/demo-page`
      (batch app) both verified independently in the browser.
- [x] **Explain the dataset is synthetic** — disclaimers present on both
      pages and in every report/export (labels, limitations, circularity
      note).
- [x] **Launch the 25,000-transaction batch** — browser click → 202 →
      completed `RPT-20261011T083933Z-48bd92`.
- [x] **Show progress and completed counts** — progress endpoint during the
      run; report shows 25,000/0/0 reconciling to the decision totals.
- [x] **Open the generated report** — auto-opened in the Reports tab
      (screenshot 4).
- [x] **Explain the metrics without presenting them as real-world results**
      — metric cards are labelled synthetic with the circularity caveat and
      "optimistic versus out-of-distribution data" limitation on the page.
- [x] **Demonstrate JSON or CSV export** — CSV verified field-by-field
      against the report; JSON equals the stored document; both carry
      metadata and limitations.
- [x] **Explain known limitations and next validation steps** — limitations
      block + page footer ("real-world validation remains unestablished …
      institutional dataset validation remains blocked pending eligible
      data").
- [x] **Stop the demo safely without deleting persistent reports** —
      runbook taskkill procedure executed at phase start; report history
      survived the full stack restart (5/5 intact), proving persistence.

## 8. Files changed in the Phase 22 commit

| File | Change | Rationale |
| --- | --- | --- |
| `frontend/index.html` | removed one blocked `<link>` (AOS stylesheet from unpkg.com) + explanatory comment | console error on every page load caused by the repo's own CSP; the resource was dead (refused at runtime), so removal changes no visual behavior. AOS script and `data-aos` attributes untouched |
| `docs/evaluation/PHASE_22_FINAL_DEMO_ACCEPTANCE_CLOSEOUT.md` | new | this file |

No model, threshold, rule, policy, feature, or evaluation-methodology change
was made — none was needed. No source/behavior changes beyond the dead-resource
removal. The venv synchronization modified only gitignored environment state.

## 9. Known issues, blockers, and limitations

- **`REAL_WORLD_VALIDATION = NOT ESTABLISHED`.** All reported metrics are
  `SIMULATED` on generator labels from the same pattern family as training
  data — optimistic versus out-of-distribution data. The Phase 23–25 ledger
  and claims registry remain authoritative (external AUC 0.47–0.595).
- **`INSTITUTIONAL_VALIDATION = BLOCKED_PENDING_ELIGIBLE_DATASET`.**
- Local pip-audit residual: 11 advisories in 5 non-pinned, non-runtime,
  local-only transitive packages (see §6); remediation recorded, deferred.
- Fresh-environment reproduction (`BLOCKED`): verified restart-from-documented-
  commands, not an empty-machine install.
- Landing page still uses the Tailwind CDN (informational console warning) —
  known prototype debt, unchanged this phase.
- The landing page's Performance section headlines synthetic-corpus
  "Certified" figures (98.7% ROC-AUC / 99.7% recall, E_hardneg) with the
  honest interpretation text below it — pre-existing content, flagged for a
  future content pass, not a Phase 22 defect.

## 10. Environment and commands (exact)

```
python --version                 → 3.12.10 (.venv/Scripts/python.exe)
pip install -r backend/requirements.txt -c backend/constraints.txt   → exit 0
powershell -NoProfile -ExecutionPolicy Bypass -File .freebuff/start_stack.ps1 → exit 0
python backend/scripts/regression_suite.py --fast          → 30/30, exit 0
python backend/scripts/phase20_demo_usability_test.py      → 51/51, exit 0
python backend/scripts/phase21_batch_recovery_test.py      → 56/56, exit 0
python backend/scripts/secret_hygiene_test.py              → 16/16, exit 0
python -m bandit -r backend/src -f json --severity-level medium → exit 0, 0 results
python backend/scripts/security_scan.py --full             → exit 0
python -m pip_audit                                       → exit 1 (local extras only, §6)
```

*Every claim above is supported by a recorded command output, report
artifact, browser screenshot, or GitHub Actions run observed in this phase.*
