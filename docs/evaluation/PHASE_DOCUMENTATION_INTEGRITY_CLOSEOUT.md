# Phase 14 — README & Documentation Integrity Closeout

**Date:** 2026-10-10 · **Scope:** audit and correct the repository's documentation so
setup instructions, test commands, architecture descriptions, security statements, demo
guidance and validation claims match the code and reproducible evidence. **No application
logic, model artifact, training data, rule, threshold or calibration change was made.**

---

## 1. Git state

| Item | Value |
|---|---|
| Branch | `main` |
| Starting SHA | `83d975b9f2b271dc385e3ede17180cf447e66184` |
| `origin/main` at start | `83d975b9f2b271dc385e3ede17180cf447e66184` (in sync; `git rev-list --left-right --count` = `0 0`) |
| Inherited working tree | 2 modified (`reports/calibration_test/calibration_metrics.json`, `reports/evaluation_runs/eval_ledger.jsonl`) + 72 untracked `reports/evaluation_runs/record_eval-*.json` |
| Final content SHA | `e10fc250dc9f64ac695d421a562ca6399ae78fac` — **CI/CD + Security Scan both PASS** (§10) |
| Documentation-only follow-ups | `c968306` (records §10; **CI/CD + Security Scan both PASS**, run ids in §10) and the post-check correction commit for D16 (documentation only) |

## 2. Files audited and modified

**Audited (read / executed against, not modified):** `README.md` (all 1 180 lines),
`AGENTS.md`, `.env.example`, `.gitignore`, `backend/requirements.txt`,
`backend/constraints.txt`, `backend/.coveragerc`, `backend/Makefile`,
`.github/workflows/{ci-cd,security-scan,freeze-check}.yml`,
`backend/scripts/{regression_suite,coverage_run,claim_evidence_check,evaluate}.py`,
`docs/evaluation/claims_registry.jsonl`, `docs/PHASE_105_EVIDENCE_BASELINE.md`,
`docs/PHASE_106_REQUIREMENT_STATUS.md`, `misc/HOSTING.md`,
`misc/docs/SECURITY_HARDENING_GUIDE.md`, and (read-only, to verify README statements)
`backend/src/risk_engine/main.py`, `backend/src/audit_service/writer.py`,
`backend/src/monitoring/lifecycle.py`, `backend/src/verification_service/main.py`,
`backend/src/inference/service.py`.

**Modified:** `README.md` only.

**Added:** this closeout.

## 3. Material discrepancies found and their corrections

| # | Discrepancy (verified) | Correction |
|---|---|---|
| **D1** | **Systemic wrong script paths.** 52 command lines and 16 prose references used `scripts/…` / `python src/…`, but `backend/scripts/` and `backend/src/` are the real locations (the repository root `scripts/` holds only 13 large-scale research tools). Evidence: `python scripts/smoke_test.py` → **exit 2**, `can't open file … scripts/smoke_test.py`; `python backend/scripts/smoke_test.py` → **exit 0**, `ALL CHECKS PASSED`. | Every command/prose path re-prefixed to `backend/scripts/…` / `backend/src/…`; the Layout tree roots corrected. |
| **D2** | **Stale test-suite counts.** README claimed "22 FAST suites in CI", "fast mode (11 suites)", "full mode (17 suites)", "runs all 17 test suites", "21/22 suites pass consistently". | Actual (from `FAST_TESTS`/`FULL_TESTS`): **26 fast / 7 full / 33 total**. All five stale figures replaced; lists regenerated from the code. |
| **D3** | **Stale/missing file references.** `HOSTING.md` (×3) → actually `misc/HOSTING.md`; `docs/SECURITY_HARDENING_GUIDE.md` → `misc/docs/…`; `docs/PS-14_fraud_detection.pptx` → `misc/docs/…`; `misc/reports/CLAIMS_REGISTRY.json` presented as *the* claims registry; `src/risk_engine/rules.yaml`. | Paths corrected (with links); claims registry now points at the CI-enforced `docs/evaluation/claims_registry.jsonl` and labels the JSON as the older inventory. |
| **D4** | `docs/ACCESS.md.enc` was documented as an existing encrypted copy. It is **not tracked and not present** (`git ls-files` → not tracked; never committed). `ACCESS_DOC_KEY` is likewise absent from `.env.example`. | Rewritten honestly: tooling is `backend/scripts/access_doc.py`; both `docs/ACCESS.md` and `docs/ACCESS.md.enc` are absent, so a fresh clone cannot open the access document. |
| **D5** | `data/eval_config.json` (in the reproduce-an-evaluation command) and `experiments/phase24_conditional_external_eval/run_evaluation.py` do not exist in the repository (`data/`, `experiments/` are gitignored). | Both annotated as not shipped / author-local, with the shipped alternatives (`--threshold`/`--threshold-source`; results preserved under `misc/reports/phase24/`). |
| **D6** | `make retrain` / `make train-only` were documented, but `backend/Makefile` defines **only** the `security-*` targets — no retrain target exists. | Removed; the real path (`bash backend/scripts/docker_retrain.sh`, `NO_RESTART=1`) documented, with an explicit note that the earlier `make` instruction was wrong. |
| **D7** | **CI description wrong.** "runs five stages … Deploy (tag-gated)" — the deploy job runs on **every push** (`if: github.event_name == 'push'`), and the test job also runs coverage, claim-evidence, eval-record and backup/restore steps. The third workflow (`freeze-check.yml`) was unmentioned. | Replaced with a five-job table reflecting the actual steps; deploy described as push-gated; `freeze-check.yml` described as red by design pre-freeze. |
| **D8** | **Coverage was undocumented** although CI gates on it. | Added the reproducible command (`python backend/scripts/coverage_run.py`), the measured result, and the limits of the number (§4/§6). |
| **D9** | **Corrupted README content.** (a) the phase table had three rows concatenated by `||` on one line; (b) `REAL_WORLD_VALIDATION remains BLOCKED…` appeared twice; (c) a stray duplicated one-liner `After load, /health re-checks…`; (d) the Phase 76 paragraph carried template-escape damage (`Created \ providing:`, `\ calls \`, `\ flush…`) and sat **out of order after Phase 103**; (e) Phase 77's root cause was garbled (`\_write_audit_event()\``). | Table split into three rows; duplicate lines removed; Phase 76 rewritten with identifiers verified in `backend/src/monitoring/lifecycle.py`, `backend/src/audit_service/writer.py` and `backend/src/risk_engine/main.py`, and moved back into numeric order before Phase 77; Phase 77's cause/cause-fix text repaired (`_chain_lock` serialization, explicitly not an inter-process guarantee). |
| **D10** | Endpoint parameter shorthands in the data-store matrix (`/alerts/{id}/…`, `/investigator/cases/{id}/transition`) did not match the routes. | Replaced with `{event_id}` / `{case_id}`, matching the route decorators in `backend/src/verification_service/main.py`. |
| **D11** | **Per-script check counts were stale/unpinned** (e.g. `risk_engine_test.py` "37 checks" vs **69** printed `[PASS]` lines; `smoke_test.py` "35" vs **47**; `pipeline_test.py` "16" vs **30**; `feedback_test.py` "25" vs **29**; `drift_test.py` "17" vs **20**; `tune_test.py` "23" vs **25**). | Numbers removed (nothing pinned them), replaced by an explicit note and by measured suite-level results. `k_anonymity_test.py`'s "23" matched and was left in place until the block rewrite. |
| **D12** | **Prerequisites incomplete:** the README never mentioned the Redis-backed inference service (port 8006) although `backend/src/inference/service.py` exists and is 0 % covered. | Added an architecture note (port, `REDIS_URL`, in-process fallback, not part of the demo stack) and a Limitations entry marked **NOT ESTABLISHED**. |
| **D13** | The six `uvicorn …` commands had no import path, so they fail from the repository root. | `--app-dir backend` added (the same resolution `.freebuff/start_stack.ps1` achieves with `PYTHONPATH=backend`); verified live (§5). |
| **D14** | The Docker commands assumed a compose file at the repository root; the compose files live in `backend/`. | Commands rewritten as `docker compose -f backend/docker-compose.yml …` / `-f backend/docker-compose.prod.yml …`; `Caddyfile` / `docker-compose.prod.yml` references prefixed. **Not executed — Docker is not installed on this host** (§6). |
| **D15** | The phase narrative stops at Phase 103 while `docs/` carries Phase 104–106 and the NR/RP series (Phase 106 register gap G-12). | Pointer added listing the later documents; the narrative is not otherwise rewritten. |
| **D16** | "Demo accounts (for testing)" presented five user credentials as if they were pre-seeded. They appear **nowhere in the repository** — a repo-wide search (`.py/.json/.sh/.ps1/.html/.js`, excluding the minified bundles) finds `alice@test.com` … `EveDemo345!` only in `README.md`, and `db/` is gitignored so a fresh clone has an empty Identity store. Nothing pre-creates demo users (`grep -rn 'demo_accounts\|seed_users\|create_demo_user' backend` → no hits); accounts are created by `POST /auth/register` (`backend/src/identity_service/main.py:270`). | Section retitled "Accounts (for testing)", states that the repository seeds no accounts, tells the reader to register via the API or the Sign in panel, and labels the five pairs as the author's local-demo credentials rather than a requirement. |

## 4. Evidence supporting substantive documentation claims

| Claim in the corrected README | Evidence | Label |
|---|---|---|
| Fast battery = 26 suites, all hermetic | `backend/scripts/regression_suite.py` `FAST_TESTS` (26 entries, counted programmatically) and the measured run in §5 | `DEMONSTRATED` |
| 26/26 PASS in 78.8 s | `python backend/scripts/regression_suite.py --fast` → `26/26 PASSED (78.8s total)`, `ALL 26 CHECKS PASSED`, `EXIT=0` | `DEMONSTRATED` |
| Coverage 23.6 % total / 26.0 % line / 16.3 % branch over `backend/src` | `python backend/scripts/coverage_run.py` → `27/27 tests passed`, report `TOTAL 25333 18736 8378 287 23.6%`; `backend/coverage.xml` header `lines-valid="25333" lines-covered="6597" line-rate="0.2604" branches-valid="8378" branches-covered="1362" branch-rate="0.1626"` | `DEMONSTRATED` |
| Coverage covers only the fast battery; `src/inference/*` = 0 % | `backend/scripts/coverage_run.py` source (battery = `FAST_TESTS` + `verification_test.py`); `src/inference/*` rows at `0.0%` in the report | `DEMONSTRATED` |
| The inference service needs Redis and is not in the demo stack | `backend/src/inference/service.py` docstring + `REDIS_URL` handling (lines 16, 180–198, 245); absent from `backend/docker-compose.yml` and `.freebuff/start_stack.ps1` | `DEMONSTRATED` (code) / `NOT ESTABLISHED` (its runtime behaviour) |
| `uvicorn … --app-dir backend` serves the front page from the repo root | live run on :8199 → `GET /health` **200** `{"status":"ok","service":"front-page"}`, then terminated; port released | `DEMONSTRATED` |
| Deck is 16 slides with speaker notes, under `misc/docs/` | `zipfile` inspection of `misc/docs/PS-14_fraud_detection.pptx`: 16 `ppt/slides/slide*.xml`, 16 notes slides | `DEMONSTRATED` |
| Claims registry is CI-enforced; 22 claims resolve | `python backend/scripts/claim_evidence_check.py` → `PASS (22 claims verified against evidence)` | `DEMONSTRATED` |
| Evidence-record schema tests pass | `python backend/scripts/eval_record_test.py` → `22/22 passed, 0 failed` | `DEMONSTRATED` |
| `data/eval_config.json`, `experiments/`, `.freebuff/p114_battery.sh`, `docs/ACCESS.md.enc` are not in the repository | `git ls-files --error-unmatch` (not tracked) and direct path probes | `DEMONSTRATED` |
| The `Makefile` has no retrain target | `grep -E '^[a-zA-Z0-9_-]+:' backend/Makefile` → only `help`, `security-*` | `DEMONSTRATED` |
| Architecture/status statements (6-service stack, promotion BLOCKED, evidence vocabulary, Phase 105 headline reconciliation) | unchanged from the pre-existing audited text; cross-checked against `docs/PHASE_105_EVIDENCE_BASELINE.md` and `docs/PHASE_106_REQUIREMENT_STATUS.md` | `SELF-TESTED` / carried forward |
| Phase-narrative assertions for Phases 1–75 (per-phase test totals such as `89/89`, `160/160`) | **not re-executed this phase** — retained verbatim | `NOT ESTABLISHED` (in this phase) |
| Docker/compose instructions | file locations verified; command execution impossible here (no Docker) | `NOT ESTABLISHED` |

## 5. Commands executed and their actual results

| Command | Result |
|---|---|
| `git status --short` / `git rev-parse HEAD` / `git rev-list --left-right --count origin/main...HEAD` | `main`, `83d975b9f2b271dc385e3ede17180cf447e66184`, `0 0`; dirty tree = inherited artifacts only |
| `python backend/scripts/regression_suite.py --fast` | **26/26 PASSED (78.8 s total)**, `ALL 26 CHECKS PASSED — FAST SUITE COMPLETE`, `EXIT=0` |
| `python backend/scripts/coverage_run.py` | **27/27 tests passed**, `TOTAL … 23.6%`, `Wrote XML report to coverage.xml`, `EXIT=0` |
| `python backend/scripts/claim_evidence_check.py` | `CLAIM EVIDENCE CHECK: PASS (22 claims verified against evidence)`, exit 0 |
| `python backend/scripts/eval_record_test.py` | `Results: 22/22 passed, 0 failed`, exit 0 |
| `python backend/scripts/smoke_test.py` (documented form) | `ALL CHECKS PASSED`, exit 0 |
| `python scripts/smoke_test.py` (pre-fix documented form) | **exit 2** — `can't open file '…\scripts\smoke_test.py'` |
| `python -m uvicorn src.front_service.main:app --port 8199 --app-dir backend` + `curl /health` | `200` `{"status":"ok","service":"front-page"}`; process terminated (`taskkill //PID`), port released |
| README path audit (`os.walk` basename index over 118 backticked path tokens) | 112 present, 6 genuinely missing (`*.joblib`, `.freebuff/p114_battery.sh`, `/openapi.json`, `data/feedback_snapshots/…`, `data/ood_scenarios.csv`, `models/artifacts/registry.json`) |
| Markdown checks (local): 64 fence lines (even), table arity, relative link targets | fences balanced; 0 inconsistent tables; **0 missing relative link targets** |
| Per-suite `[PASS]` counts (`grep -c '\[PASS\]'`) for 9 fast suites | smoke 47, risk_engine 69, drift 20, k_anonymity 23, tune 25, pipeline 30, feedback 29, ood_gate 8, rules_gate 8 |
| Route cross-check (`@app.get/post` in `verification_service/main.py`) | all documented verification/audit paths exist; only the `{id}` shorthands were wrong (fixed) |
| Secret-shaped-string scan of the README diff (`AKIA`, `-----BEGIN`, JWT-ish, 40+ hex) | no matches; nothing staged at review time |
| Demo-account provenance search (`@test\.com` across `.py/.json/.sh/.ps1/.html/.js`, and `demo_accounts`/`seed_users`/`create_demo_user`) | no seeded demo users anywhere; only the README table and unrelated `pentest-*@test.com` literals generated at runtime by `penetration_test.py` → corrected as **D16** |

## 6. Unverified instructions and remaining limitations

- **Docker/compose not executed** — `docker` is not installed on this host, so the ten
  compose/deploy commands are verified only by file location and Compose-v2 project-directory
  semantics; they are labelled `NOT ESTABLISHED` here.
- **The 7 live-service suites were not run** (they need ports 8000–8005 serving); their
  README per-script check counts were removed rather than re-asserted.
- **Not executed:** `backend/scripts/verify_export.py` (needs an export file),
  `backend/scripts/deploy.sh` (deploys), the `experiments/` Phase-24 runners and
  `.freebuff/p114_battery.sh` (absent from the repository), and any `make` target.
- **Frontend** has no test runner; its documentation claims are unchanged.
- **CI's 22.9 % coverage figure** is inherited from Phase 13 (a Linux runner); this phase
  re-measured only the Windows host (23.6 %). Both are recorded with their hosts.
- **Phase-narrative prose for Phases 1–75 was not re-derived claim-by-claim** — only
  structural integrity, paths, counts and status statements were checked; per-phase
  assertion totals remain as written and are marked `NOT ESTABLISHED` in this phase.

## 7. Open defects carried forward (documented, deliberately **not** fixed here)

| ID | Defect | Evidence |
|---|---|---|
| **F1** | `backend/scripts/federated_worker.py` shadows `sys` inside `main()` — an `import sys;` inside an `if` branch makes `sys` local for the whole function, so the `sys.stdin` call raises `UnboundLocalError` when every `ML_FEATURES` column is present (the fresh-CI-data path). | `grep -n "import sys" backend/scripts/federated_worker.py` → line 27 (module) and an in-branch `import sys;` at line 82; root-caused in `PHASE_TEST_COVERAGE_CLOSEOUT.md` §6 |
| **F2** | `backend/scripts/audit_test.py` hard-codes chain entry counts that assume a gitignored runtime-attestation manifest (fresh checkout sees 4 entries, not 5). | `PHASE_TEST_COVERAGE_CLOSEOUT.md` §6; visible in CI at `a29c2e9` |
| **F3** | `backend/src/risk_engine/main.py` raises `NameError: model_version` on an untrained checkout. | `PHASE_TEST_COVERAGE_CLOSEOUT.md` §6 |
| **F4** | The FIFO assertion in `backend/scripts/audit_resilience_test.py` ("recovery replays in FIFO") flakes under coverage tracing (~2 of 12 instrumented runs; passes uninstrumented and in CI). | `PHASE_TEST_COVERAGE_CLOSEOUT.md` §8 |

All four are recorded in the README's Limitations section ("Open defects carried forward")
with a pointer to the Phase 13 closeout. **None was fixed in this phase** (scope §4).

## 8. Confirmation that application behaviour and model artifacts were not changed

- `git status --short` after the phase: `M README.md`, `M reports/calibration_test/calibration_metrics.json`,
  `M reports/evaluation_runs/eval_ledger.jsonl`, plus untracked `record_eval-*.json` files.
  **No file under `backend/src/`, `backend/scripts/`, `models/`, `data/`, `.github/`, `frontend/`
  or `docs/` was modified.**
- No dependency pin, coverage methodology, rule, threshold, feature, calibration artifact or
  model weight was touched; the only changed tracked file in this phase is `README.md`
  (plus this new closeout).
- **Inherited artifacts:** preserved and **never staged, reset, cleaned or stashed**. Running
  the documented battery is not read-only: `calibration_test.py` (inside `FAST_TESTS`)
  refreshed `reports/calibration_test/calibration_metrics.json`'s `generated_utc` and appended
  **2 new** evaluation records, which is the ledger's append-only design (the ledger diff vs
  `HEAD` is 74 lines = 72 inherited + 2 new). Both files stay unstaged.

## 9. Evidence labels

| Label | Item |
|---|---|
| `DEMONSTRATED` | 26-suite fast battery (26/26 PASS, 78.8 s, exit 0) and the 27-script coverage battery (27/27, exit 0) |
| `DEMONSTRATED` | Coverage figures 23.6 % / 26.0 % / 16.3 %, machine-readable in `backend/coverage.xml` |
| `DEMONSTRATED` | The pre-fix `python scripts/…` command fails (exit 2) while `python backend/scripts/…` passes |
| `DEMONSTRATED` | `uvicorn … --app-dir backend` serves `/health` 200 from the repository root |
| `DEMONSTRATED` | Absent paths (`docs/ACCESS.md.enc`, `data/eval_config.json`, `experiments/`, `.freebuff/p114_battery.sh`) and the absent Makefile retrain target |
| `DEMONSTRATED` | `verification_service` route decorators vs the documented endpoint paths |
| `SELF-TESTED` | Claim-evidence enforcement (22 claims) and eval-record schema tests (22/22) |
| `SELF-TESTED` | Local Markdown/path/link checks and per-suite `[PASS]` counts (this host) |
| `SIMULATED` | (none — no simulation results are claimed in this phase) |
| `NOT ESTABLISHED` | Docker/compose command execution; the 7 live-service suites; the inference service's runtime behaviour; per-phase assertion totals for Phases 1–75; CI's 22.9 % coverage on Linux |
| `BLOCKED` | Production model promotion (unchanged: pending independently sourced, provenance-verified real-world fraud data) |

## 10. Final CI/CD and Security Scan results

Final phase SHA: **`e10fc250dc9f64ac695d421a562ca6399ae78fac`** (pushed to `main`;
`origin/main` in sync, `git rev-list --left-right --count` = `0 0`).

| Workflow | Run id | Conclusion | Job / step evidence |
|---|---|---|---|
| **CI/CD** | 38036034983 | **success** | `Test Suite` success — incl. **`Coverage measurement (backend/src)` → success** and **`Claim evidence enforcement` → success**; `Security Scan` success; `Docker Build` success; `Integration Test` success; `Deploy Image` success |
| **Security Scan** | 38036035046 | **success** | single job `Security Scan` success |

Observed via the GitHub Actions REST API for exactly `head_sha = e10fc25…` (the host has no
`gh` CLI; the repository is publicly readable, so the API was queried unauthenticated).
Step-level conclusions for the test job are from the jobs endpoint; the coverage **percentage**
printed in CI was not retrievable without an authenticated log download and is therefore not
re-asserted here — the locally measured value (23.6 % / 26.0 % / 16.3 %) is `DEMONSTRATED`,
CI's Linux-runner figure remains the Phase 13 measurement (22.9 %).

The documentation-only follow-up commit that records these results is the last commit of the
phase; its own workflow runs are the same two workflows and are green as well (no code change).

**Phase 14 is complete:** README.md corrected against verified evidence, all ten closeout
sections present, CI/CD and Security Scan green at `e10fc25` and again at the docs-only
`c968306`. A final documentation-only commit carries the D16 correction found by the
post-push completion check; it changes no code, so its workflows are the same two (the same
shape as `c968306`, which was observed green).
