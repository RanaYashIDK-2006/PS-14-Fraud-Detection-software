# PS-14 — Phase 4 (Strix Security Validation) Closeout

**Phase:** 4 — Strix Security Validation and Silent-Defect Assessment
**Status:** complete (Strix execution itself: `ENVIRONMENT BLOCKED`)
**Assessment report:** [STRIX_SECURITY_ASSESSMENT.md](STRIX_SECURITY_ASSESSMENT.md)
**Machine-readable evidence:** [PHASE4_SECURITY_EVIDENCE.json](PHASE4_SECURITY_EVIDENCE.json)

## Commit / SHA ledger

| Item | SHA |
|---|---|
| Branch | `main` |
| HEAD at assessment start (Phase-3 push tip) | `05d491b950fb4d26b04e7ff3cc1de1c6942c0e42` |
| Phase-3 push commits inspected in CI | `8cfa3ffa…`, `05d491b9…` |
| **Phase-4 fix + evidence commit** (fixes, regression suites, CI wiring, assessment report, evidence bundle) | **`c07523530eed52042415c58f5196a41cb0df6d52`** |
| Phase-4 closeout commit (this document) | the commit immediately following `c075235` on `main` |
| Remote | `https://github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software.git` |

## Strix identity and final Strix result

| Item | Value |
|---|---|
| Tool | `usestrix/strix` v1.7.0 (source HEAD `55bc07991aacb1c26a81b43e7ca53148730d2019`), isolated venv `.freebuff/strix_env` |
| Supplied artifacts | **none** (exhaustive search across repo, sibling dirs, home, archives, git refs) |
| Backend requirement | Docker only (`strix/runtime/backends.py`; `check_docker_installed()` → `sys.exit(1)`) |
| Execution attempts | two this phase, both `rc=1` / `DOCKER NOT INSTALLED` — logs preserved (`.freebuff/strix_run_attempt.log`, `.freebuff/p4_strix_run_attempt.log`) |
| **Final Strix result** | **`ENVIRONMENT BLOCKED` — Strix produced no findings. No Strix finding is reported, and none is invented.** |
| Re-run recipe | `.freebuff/STRIX_PHASE_RESUME.md` (install Docker Desktop → `docker pull ghcr.io/usestrix/strix-sandbox:1.3.0` → point at `http://127.0.0.1:8000` with the local Ollama model) |

## Target environment

Authorized local/staging stack only: `127.0.0.1` ports 8000 (front), 8001 (identity), 8002 (privacy), 8003 (risk),
8004 (verification), 8005 (audit); all six `/health` endpoints 200 before and after testing. No external host,
cloud service or third-party endpoint was contacted; no real credential, banking or personal data was used.

## Findings

### Discovered and confirmed (7)

| ID | Finding | Severity assessment | Status |
|---|---|---|---|
| F-01 | Front-service `/security/scan-history` and `/security/scan-latest` served security-scan reports (scores, vulnerable counts, issue lists) with **no admin session**, while sibling `/security-scan` did require one | high (report data, inconsistent control) | **fixed** — `_require_admin_session` added; regression test added; retest 401/200 |
| F-02 | `POST /run-security-scan` returned 500 on the success path (`NameError`: `env` defined inside the unreachable `if not script.exists():` branch) | medium (functionality broken behind a guard) | **fixed** — assignment moved to the success path; regression test monkeypatches `subprocess.run` and asserts the child env |
| F-03 | NaN / `+Inf` / `-Inf` / `1e309` feature values and NaN amounts answered **500** instead of 422 (framework 422 handler crashed serialising the rejected value; middleware converted it to a generic 500) on risk **and** privacy | medium (wrong status, log noise; rejection itself still occurred before logic) | **fixed** — shared `RequestValidationError` handler + JSON-safe sanitiser in `backend/src/middleware/__init__.py`; regression test added; retest 422 |
| F-04 | Supabase **`service_role` key + project URL hardcoded** in tracked `backend/scripts/smoke_test_supabase.py` (committed 2026-09-12, `6e0f9f8`); value equals the live `.env` credential | high (privileged credential in a public repo) | **source remediated** (env/`.env` + fail-fast); **rotation REQUIRES EXTERNAL ACTION** — value remains in git history |
| F-05 | **JWT signing secret** (live `.env` value, which also derives the PII `fernet_key` and export signing key) plus an internal-token literal hardcoded in tracked `backend/scripts/start_risk_loadtest.py` | high | **source remediated** (env/`.env` + fail-fast); **rotation REQUIRES EXTERNAL ACTION** — rotating re-derives `fernet_key` and would break existing DB-1 PII ciphertext |
| F-06 | CI **Bandit gate red** in all four Phase-3 runs: 15 B608 findings in `front_service/main.py`, **all false positives** on parameterised SQL | medium (gate defect; hid the security steps behind it) | **remediated** — 14 justified per-line `# nosec B608` annotations (identifier-only interpolation, values bound); Bandit exit 0 locally |
| F-07 | CI **secret scanner never executed** (`Check for Secrets` / trufflehog skipped behind the failed Bandit step) | medium (instrumentation gap) | **unblocked** — gate now green; the scanner runs on the next push, and its result is **not yet observed** |

### Unresolved

| Item | State |
|---|---|
| Rotation of the two exposed credentials (F-04, F-05) | **Requires external action by the owner.** Not performed: rotating `jwt_secret` re-derives `fernet_key` and invalidates existing PII ciphertext; rewriting git history is out of scope for this phase |
| CI secret-scanner verdict on the historical key (F-07) | Will execute on this push; `--only-verified` may flag the key until it is rotated |
| Strix execution | `ENVIRONMENT BLOCKED` (no Docker, no admin action taken) |
| `/fraud-report` returns 500 (explicit body) when the report artifact is absent | **Observation**, left unchanged (status-code semantics, no leak, no content on this host) |
| Dev 25-byte `jwt_secret` default and its derived keys | **Pre-existing, intentionally unchanged** documented dev debt; changing it breaks ciphertext |
| Inference/worker module (30 routes) | **NOT TESTED** — no listener in the assessed topology |
| Deployment topology, TLS, proxy, container hardening | **NOT TESTED** — outside local-staging scope |

### False positives / reclassified non-findings

* Suite-B checks **G4, G5, H2, K1, K3** were initially reported as failures; they are **test artifacts** — FastAPI
  validation runs before the token check, so malformed bodies returned 422. Corrected probes (R-G4, R-G5, R-H2,
  R-K1, R-K3) all pass. Reclassified, not silenced.
* Suite-C check **V10** (first run) was a **re-run artifact** of a non-unique event id that the run's own replay
  probe had already scored; unique ids are now used and the check passes.
* **15 Bandit B608 findings**: false positives on parameterised SQL (see F-06). Also recorded: the earlier
  `/fraud-report` 500, unknown-extra-field acceptance, `1e308` amount acceptance, duplicate-key last-wins,
  and unauthenticated HTML shells are **observations**, not defects — each with its rationale in §8/§19 of the report.

## Regression tests

| Suite | Guards | Result |
|---|---|---|
| `backend/scripts/front_service_test.py` | F-01 (unauth 401 / admin 200 on both scan routes, session cookie explicitly cleared) and F-02 (`/run-security-scan` → 200 with `PYTHONIOENCODING` in the child env) | ALL CHECKS PASSED |
| `backend/scripts/risk_engine_test.py` | F-03 (raw NaN / +Inf / −Inf / NaN in an unconstrained field → 422, no traceback or path leak) | ALL CHECKS PASSED |
| `backend/scripts/secret_hygiene_test.py` (**new**) | F-04/F-05/F-06 (no live `.env` value in tracked files; no credential-shaped literal outside self-declared detector scripts; remediated scripts hold no literals and fail fast; Bandit medium+ exits 0) | 15/15 PASS |
| `.github/workflows/security-scan.yml`, `.github/workflows/ci-cd.yml` | the hygiene suite now runs in both CI security jobs | added (YAML validated) |

## Repository verification

| Check | Result |
|---|---|
| Phase-114 regression battery (`bash .freebuff/p114_battery.sh`) | **82/82 PASS, 0 FAIL, 0 MISSING** — ended 2026-10-06T09:19:51+05:30 |
| Security suites inside the battery | `penetration_test`, `security_test`, `security_attack_test`, `sql_injection_test`, `adversarial_test`, `security_ci_gate`, `security_scan`, `audit_hygiene_check` — all PASS |
| `claim_evidence_check` / `eval_record_test` / `review_resolution_check` | PASS / PASS / PASS |
| `review_package_check` | **PASS** (13/13 sections; no licence cleared, no reviewer assigned, no 50M claim, freeze state unchanged) |
| `check_freeze_test.py` | PASS (test of the freeze checker) |
| `scripts/check_freeze.py` | **rc 1 / 78 placeholder failures — expected by design** (pre-freeze RED is the gate working); not touched by this phase |
| Bandit (`-r backend/src/ --severity-level medium`, the exact CI command) | **exit 0** (was exit 1 with 15 findings) |
| Live post-remediation retest (suite C) | **18/18 PASS** against the restarted final-code stack |
| Pre-remediation evidence | preserved before any fix: `phase4_results_a_PREFIX.json`, `phase4_results_b_PREFIX.json`, `phase4_probe_PREFIX.json` |
| CI verification for Phase-3 push | CI/CD *Test Suite* PASS on both commits; both security jobs FAIL at Bandit (F-06) with downstream jobs **BLOCKED**; Freeze Check workflow **NOT RUN (by design, path-filtered)**. Artifact/log download requires auth (HTTP 401), so conclusions are authoritative and the Bandit failure was reproduced locally |

## Production-artifact changes

**None.** No model artifact, calibrator, release manifest, feature contract, threshold, dataset, benchmark,
Research Plan, preregistration or evaluation methodology was modified. `git status` confirms `models/`, `data/`,
`backend/src/risk_engine/rules.yaml`, `docs/RESEARCH_PLAN.md` and
`docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` are untouched by this phase. The only source changes are
the remediations listed above plus comment-only Bandit annotations.

## Remaining security limitations

1. **This is a self-assessment.** The same author designed and interpreted every probe; no independent assessor and
   no third-party tool (Strix) contributed a finding.
2. **Credential rotation is outstanding** for F-04 and F-05; the exposed values remain in git history and, for the
   Supabase key, remain valid until the owner rotates them.
3. **Strix never executed**, so automated agentic testing coverage is zero for this phase.
4. **Scope:** local stack only — no cloud, container, network, TLS or production topology; 30 inference-module
   routes and the deployment surface are untested.
5. **Reproduction is two-client, not independent** (live sockets and in-process ASGI), from the same source tree.
6. **CI's non-Bandit security steps** (pip-audit, custom scanner, penetration suite, Docker build) executed only as
   far as the runner's conclusions show; the secret scanner's verdict arrives with the next push.
7. **No claim of completeness:** absence of findings in 164 probed routes is not proof of absence of defects.

## Final security evidence classification

| Classification | Applies to |
|---|---|
| `CONFIRMED → FIXED → RETEST PASSED` | F-01, F-02, F-03 |
| `CONFIRMED → SOURCE REMEDIATED; ROTATION REQUIRES EXTERNAL ACTION` | F-04, F-05 |
| `CONFIRMED (tooling) → REMEDIATED (gate green locally)` | F-06 |
| `CONFIRMED (process) → UNBLOCKED; RESULT NOT YET OBSERVED` | F-07 |
| `FALSE POSITIVE / TEST ARTIFACT` | suite-B G4, G5, H2, K1, K3; suite-C V10 (first run); 15 Bandit B608 |
| `OBSERVATION` | `/fraud-report` 500 body, extra-field acceptance, `1e308` amount, duplicate-key last-wins, HTML shells, log-line incident from one probe |
| `NOT TESTED` | inference/worker module (30 routes), deployment topology |
| `ENVIRONMENT BLOCKED` | Strix execution |

## Final decision

> ## `SECURITY FINDINGS FIXED — RETEST PASSED`
>
> `SELF-TESTED SECURITY ASSESSMENT — findings observed within the tested local/staging scope. This does not constitute an independent penetration test or guarantee absence of vulnerabilities.`

## Next phase

**Phase 5 — Prototype Optimization**, constrained by the measured Phase-3 bottlenecks and by the findings above.
Carry forward: (a) the open credential-rotation items, (b) the CI security gate must stay green (Bandit + secret
hygiene suite now run in both security jobs), (c) the inference module's untested routes, (d) no optimization may
erase the scientifically important negative results recorded in Phase 3.
