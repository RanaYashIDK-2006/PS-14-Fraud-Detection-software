# PS-14 — Strix Security Validation and Silent-Defect Assessment (Phase 4)

**Document status:** final report for Phase 4.
**Assessment type:** self-directed security assessment of the local/staging stack.
**Bounded conclusion (requested wording, used verbatim):**

> `SELF-TESTED SECURITY ASSESSMENT — findings observed within the tested local/staging scope. This does not constitute an independent penetration test or guarantee absence of vulnerabilities.`

**Final decision (§25 of the phase brief):** `SECURITY FINDINGS FIXED — RETEST PASSED`

Machine-readable companion: [PHASE4_SECURITY_EVIDENCE.json](PHASE4_SECURITY_EVIDENCE.json) (secret-free bundle:
baseline hashes, suite tallies, findings, CI run results, verification results).
Phase closeout: [PHASE_STRIX_SECURITY_CLOSEOUT.md](PHASE_STRIX_SECURITY_CLOSEOUT.md).

---

## 1. Scope, authority, and boundaries

| Item | Value |
|---|---|
| Target | the five-service PS-14 stack running on `127.0.0.1` (8000, 8001, 8002, 8003, 8004, 8005) |
| Authorization basis | owner-directed assessment of the owner's own local prototype |
| External systems touched | **none** — no third-party host, no Supabase call, no public endpoint probed |
| Real credentials used | **none** — synthetic demo accounts created for this phase, or the repo's own dev `.env` values read into headers only |
| Secret values in this document | **none** — findings name keys, files and line numbers only |
| Destructive actions | **none** — no data deletion, no schema change, no volume/branch rewrite, no force-push |
| Test clients used | two independent HTTP clients: `urllib` over the live sockets, and in-process `fastapi.testclient.TestClient` against each service's ASGI app |

Every result below was produced against the local stack, or against a hermetic temporary `DB_DIR` where a suite
requires database isolation. Nothing in this document is a claim about any other environment.

---

## 2. Security baseline (§2 of the phase brief)

Recorded in `.freebuff/phase4_baseline.json` (local scratch) and mirrored into
[PHASE4_SECURITY_EVIDENCE.json](PHASE4_SECURITY_EVIDENCE.json).

| Baseline item | Recorded value |
|---|---|
| Git branch / SHA at phase start | `main` / `05d491b950fb4d26b04e7ff3cc1de1c6942c0e42` |
| Remote | `https://github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software.git` |
| Service versions | shim-free single-service apps; identity/risk report `release_id` / `model_id` = `legacy-altman_native_E_hardneg_cert_20260904`, `feature_version v1` in `/health` |
| Services healthy at baseline | 6/6 `/health` = 200 (8000–8005) |
| Endpoint inventory | **164 routes** across 8 modules (`.freebuff/phase4_routes.json`) |
| Auth configuration (no values) | client JWT (Bearer), `X-Internal-Token`, `X-Compliance-Token`, admin `admin_session` cookie/Bearer with scrypt passphrase (+ optional TOTP), CSRF `X-Requested-With` on cookie-authenticated state changes |
| Model artifacts | sha256 recorded for `random_forest.joblib`, `xgboost.joblib`, `isolation_forest.joblib`, `lr_model.joblib`, `scaler.joblib`, `calibrator.joblib`, `metadata.json` |
| Release manifest | sha256 of `models/production/release_manifest.json` |
| Native feature contract | sha256 of `backend/research/public_feature_contract.json` |
| Config hashes | sha256 of `backend/src/risk_engine/rules.yaml`, `backend/docker-compose.yml`, `.env.example`, the three workflow files |
| Strix identity | `usestrix/strix` v1.7.0, source HEAD `55bc07991aacb1c26a81b43e7ca53148730d2019` |
| Supplied Strix artifacts | **0 files** (exhaustive search; see §3) |

**Expected controls (the claim to be tested, not assumed):** internal-token on `/internal/*`; compliance-token on
`/compliance/*` and the investigator endpoints; schema + finite-value validation on every inference route; admin
session on admin APIs; PII encryption with a `jwt_secret`-derived key; append-only audit with a hash chain; rate
limits on auth/login/admin-login/evaluate; CORS + origin validation and security headers on all five services;
generic 500 bodies with no stack traces; archive-traversal / nested-archive / native-code rejection on ingestion.

**The baseline question this phase asks of every route:** does the guard exist *and* does every path that reaches
the protected functionality actually enforce it?

---

## 3. Strix material: import and inspection (§3 of the phase brief)

| Inspection item | Result |
|---|---|
| Files supplied as "Strix material" | **none found.** Searched the repo, sibling directories, home, Downloads, Documents, git refs, archives and the Freebuff state dir |
| Real Strix implementation | `usestrix/strix` v1.7.0 cloned to `.freebuff/strix-src/strix` (HEAD `55bc079…`), installed into an isolated venv at `.freebuff/strix_env` (`strix.exe --help` exits 0) |
| PyPI package named `strix` | **different project** (GIS/KML tool) — deliberately not installed |
| Backend requirement | Docker is the only execution backend: `strix/runtime/backends.py` registers one backend, `strix/interface/environment.py` → `check_docker_installed()` → `sys.exit(1)` |
| Required dependencies | Docker CLI + `ghcr.io/usestrix/strix-sandbox:1.3.0`; an LLM backend (Ollama `llama3.1:8b` is the only model on this host that emits valid structured tool calls — probe script `.freebuff/strix_toolcall_probe.py`) |
| Target URLs the artifact would use | it takes the target as a CLI argument; it was pointed only at `http://127.0.0.1:8000` |
| Credentials expected by the artifact | none supplied, none invented |
| Destructive-action classes | Strix executes commands inside its own sandbox container; with Docker absent nothing was executed, so no classification was needed beyond "not executed" |
| Classification of available material | **safe read-only**: source inspection only. No vendor payload was executed against the stack |
| Original artifacts | untouched (nothing to preserve beyond the clone and the two attempt logs) |

**Execution status: `ENVIRONMENT BLOCKED`.** Two run attempts (`.freebuff/strix_run_attempt.log`,
`.freebuff/p4_strix_run_attempt.log`) both exit `rc=1` with `DOCKER NOT INSTALLED`; `docker` is not on `PATH`,
WSL2 is active but has **no installed distribution**, and Docker cannot be installed without administrator
action, which this phase did not take. No Strix findings are reported, because Strix produced none. Nothing in
this document is a Strix result.

---

## 4. Attack-surface inventory and unauthenticated reachability

164 routes were inventoried from source, then each was requested **without any credential** by an independent
`urllib` probe (no default headers, no cookies):

| Classification | Count |
|---|---|
| `REJECTS (auth)` — 401/403 | 64 |
| `REACHABLE (no creds)` — 200 | 37 |
| `VALIDATION (pre-handler)` — 422 | 29 |
| `NOT RUNNING` — inference module present in source, no listener in this topology | 30 |
| `NOT FOUND` — inventory entries absent from the live app | 2 |
| `SERVER ERROR` | 1 |
| `HTTP 410` (deliberately retired route) | 1 |

The 37 credential-free 200s were classified by hand rather than trusted:

* **benign by design** — `/health`, `/status`, `/static/*` assets, the HTML shells (`/admin`, `/admin/transactions`,
  `/admin/audit`, `/admin/security`, `/admin/settings`, `/login-page`, `/monitor-page`, `/admin-page`,
  `/fraud-report-page`). Shells contain markup only; every data API behind them was probed separately (§7) and
  rejects without a session.
* **genuine findings** — `/security/scan-history` and `/security/scan-latest` returned security-scan report data
  (scores, vulnerable-count, issue lists) with no credential, while their sibling route `/security-scan` required an
  admin session. See **F-01**.
* **out of scope of this topology** — the inference/worker module (`backend/src/inference/service.py`,
  `rate_limiter.py`) has no listener in the five-service stack; its 30 routes are recorded as **NOT TESTED**.
* `/chameleon-results` and `/admin-only` return 404: the inventory is broader than the live app (paths exist in a
  packaged copy under `misc/dist/`).

---

## 5. Authentication testing

| Probe | Result |
|---|---|
| Missing credentials on protected routes (64) | 401/403 — rejected before the handler, verified by replay probes that returned no stored state |
| Malformed `Authorization` header | 401 |
| Forged / wrong-issuer JWT | 401 |
| Expired-credential handling | covered by the existing suites (`penetration_test`, `security_test` — PASS in §22) |
| Alternate credential locations | query string, `X-Api-Key`, `token=` form field — all rejected; `X-Internal-Token` and `X-Compliance-Token` are the only accepted service credentials |
| Inconsistent authentication between sibling routes | **FOUND (F-01)** — same resource family, one route guarded, two not |
| Default credentials | the dev default `jwt_secret` is a documented pre-existing dev-debt item (AGENTS.md), unchanged by this phase; no default admin passphrase is accepted after first bootstrap |
| Credential leakage into logs | one query-string probe in this phase caused a live internal token to appear in `.freebuff/risk.log` (a socket access line, `422`). The log is gitignored, was truncated by the next stack restart, and no such value appears in any report or committed file. Recorded as an **observation about the test method**, not a product defect |
| Secrets embedded in source | **FOUND (F-04, F-05)** — see §11 |

No authentication bypass was found: every rejection recorded during the probe was re-tested with a
well-formed request to confirm the rejection happened for the *credential* reason and not because a malformed body
short-circuited validation (this distinction produced five false "failures" in the raw suite-B run — see §8 and §21).

---

## 6. Authorization testing

| Area | Probes | Result |
|---|---|---|
| Admin APIs (`/admin/api/*`, `/admin/db-*`) | no session, user JWT, wrong method | 401/403/405 — no state change; guarded by `_require_admin` / `_require_admin_session` |
| Admin row-level read (`/admin/api/transactions/{event_id}`) | admin session vs user JWT | 200 for the admin session, **401** for the user JWT — no IDOR |
| Scan-report routes | no session | **F-01** (fixed, 401 after remediation) |
| Investigator endpoints (`/investigator/cases`) | no token, user JWT, forged `X-Compliance-Token` | 401 (constant-time compare in the handler), 422 pre-handler when the request is malformed |
| Audit/compliance endpoints | no token | 401/403 (`require_compliance`, `require_compliance_token`) |
| Alternate HTTP methods on guarded routes | `POST` on a GET-only guarded route | 405 while still unauthenticated → no bypass |
| Database explorer (`db-schema`, `db-rows`, `db-update`) | unauthenticated, and with an admin session but wrong per-DB passphrase / missing TOTP | rejected (401/403); see §7 for the injection analysis |
| Retired route | `/admin/totp/current-code` | **410 Gone** — an intentional removal, verified as a control (the UI cannot fetch TOTP codes) |

No authorization bypass was found beyond **F-01**, and no guarded route was observed to perform its protected
operation while returning 401/403.

---

## 7. Inference-boundary testing

| Path to model execution | Guard observed | Alternate-path note |
|---|---|---|
| `POST /internal/evaluate` (risk) | `check_access` → 401/403/429, lifecycle 503, idempotency 409 | primary path |
| `POST /internal/evaluate-batch` (risk) | same internal-token check; body is a bare list, accepts `[]` | tested independently with a valid body + wrong token → **401** |
| `POST /internal/attribution` (risk) | internal-token check | tested independently with a full `EvaluateRequest` + wrong token → **401** |
| `POST /internal/ingest-transaction` (privacy) | `verify_internal_token` → 401 | tested with a complete body + wrong token → **401** |
| Front-service proxies (`/evaluate`, attribution UI paths) | the front service forwards with the internal token from the environment; no client-controlled token | no client path to the model without a scoped route |
| Feature validation | `FeatureVector` finite-value validator (`_finite_values`, risk `main.py`), schema/count/type checks | NaN / ±Inf / `1e309` are rejected **before** logic — replay probes confirm no row is stored and no score is produced |

**The primary defect here (F-03)** was not a missing guard but a broken *rejection*: the validator rejected
correctly, then FastAPI's default 422 handler tried to serialise the offending `nan` value into JSON, crashed on
`allow_nan=False`, and the security-header middleware converted it into a generic `500`. The rejection still
happened before logic (proved by the replay probe), but the observable status and body were wrong and a traceback
was logged. Fixed in shared middleware, so all five services inherit it (§15).

Model-selection restrictions and release attestation are enforced outside these routes (`/health` reports
`release_attested: true`); no route in scope accepted a caller-supplied model selector.

---

## 8. Input-fuzzing results

Controlled malformed inputs were sent through raw `content=` bodies (httpx refuses to *encode* NaN, which is itself
a useful signal that non-finite JSON is unusual) and each result was classified with the phase-brief taxonomy:

| Case | Classification |
|---|---|
| Missing required field (feature, fraud_id, amount, ts, device/location/recipient) | `REJECTS SAFELY` (422) |
| Wrong scalar type / nested object where a scalar is expected | `REJECTS SAFELY` (422) |
| `null` for required numeric field | `REJECTS SAFELY` (422) |
| Negative amount, `hour_of_day = -1`, `hour_of_day = 24` | `REJECTS SAFELY` (422) |
| NaN, `+Inf`, `-Inf`, `1e309` in an evaluated feature | **was `CRASHES` (500) → now `REJECTS SAFELY` (422)** — F-03 |
| NaN amount on ingest | **was `CRASHES` (500) → now `REJECTS SAFELY` (422)** — F-03 |
| Extremely large finite amount (`1e308`) | `ACCEPTS` (200) — finite, schema-valid; recorded as an accepted design boundary |
| Empty string / whitespace-only id fields | `REJECTS SAFELY` (422) |
| Extremely long string (id fields) | `REJECTS SAFELY` (422 on length-bounded fields; fixed-width id pattern enforced elsewhere) |
| Malformed JSON body | `REJECTS SAFELY` (422) |
| Duplicate parameters / duplicate JSON keys | last-wins normalisation by the JSON parser, no error — `SILENTLY NORMALIZES`, recorded as an observation (no security impact: values are validated after parsing) |
| Unknown extra field in a feature payload | `ACCEPTS` (200, extra field ignored) — pydantic default; recorded as an observation, not a defect: the extra key cannot reach the model matrix, which is built from the declared schema |
| Oversized payload body | rejected by the framework/size limits in the existing suites (`security_test`, `penetration_test` — PASS) |

**Fuzzing produced exactly one real defect (F-03) and three observations**, all recorded above with their
classification rather than a status code alone.

---

## 9. Data-access and export testing

| Probe | Result |
|---|---|
| Transaction rows via `/admin/api/transactions` without a session | 401 |
| Transaction detail with a *user* JWT | 401 (no cross-role read) |
| Same detail with the admin session | 200 (control) |
| Audit events / chain / overview without a token | rejected (401/403) |
| Audit export without a token | rejected; export signing key remains derived from `jwt_secret` (dev debt, unchanged) |
| Compliance graph with a forged compliance token | 401 |
| Path traversal in path parameters (`..`, encoded variants, backslashes) | rejected / not routed (`security_attack_test` traversal checks PASS) |
| Bulk read via pagination parameters outside limits | clamped by `limit` bounds (`ge`/`le`), verified in the admin suites |
| Internal identifiers in responses | no internal token, DB path or key material observed in any 4xx/5xx body |
| Excessive error detail | none — Error bodies are generic or explicitly constructed (`§13`) |

No evidence was found of unauthorized bulk access or export-authorization bypass. Sensitive row content was not
retained during testing: only shapes, counts and status codes were recorded.

---

## 10. File and parser results

| Check | Evidence |
|---|---|
| Archive traversal (`../`, `..\`, absolute paths) | `backend/src/monitoring/phase107_public_dataset_ingestion.py` classifies `path_traversal:<member>`; suite `phase107_public_dataset_ingestion_test.py` builds traversal/backslash/nested-archive fixtures and asserts rejection — **PASS** |
| Nested archive (`inner.zip` inside a ZIP) | `nested_archive` classification, asserted — **PASS** |
| Symlink / executable members, unexpected mode bits | rejected by the same inspector, asserted — **PASS** |
| Malformed CSV/Parquet, unexpected encoding, invalid schema | ingestion suite asserts explicit rejection paths — **PASS** |
| Oversized file / decompression abuse | ingestion and `security_test` size-bound checks — **PASS** |
| ZIP traversal in the general attack suite | `security_attack_test.py` §7 path-traversal block — **PASS** |
| Archive builders | `build_all_data_zip.py`, `build_audit_zip.py`, `build_security_zip.py` unchanged by this phase |

The existing ZIP-traversal and decompression protections are **still effective** (re-run, not assumed).

---

## 11. Secret and configuration audit

Method: a byte-level containment scan — every value in the local `.env` (11 keys, ≥ 12 chars) was searched
across **all 1 702 git-tracked files**, then across **15 485 files** in the whole worktree (excluding `.git`,
`.venv`, caches). Only key names, paths and counts were printed; values were never emitted.

| Result | Detail |
|---|---|
| Tracked files containing a live `.env` value | **2 secrets in 2 files** — **F-04**, **F-05** below |
| Non-secret allowlist | `CORS_ORIGINS` (a localhost origin list) legitimately appears in `.env.example`, `backend/docker-compose.yml`, `backend/src/settings.py` |
| Real PEM private-key material in tracked files | **none** (11 files *mention* a PEM banner as a detector pattern; none contains a real block) |
| Provider-shaped literals (`eyJ…` JWT, `sk-`, `AKIA…`, `ghp_`, `AIza…`, `xox…`, `*_live_`) | none outside the detector scripts — after F-04/F-05 remediation |
| Generated artifacts | no secret embedded in `reports/`, `models/`, `db/` or `docs/`; `docs/ACCESS.md.enc` is encrypted and `docs/ACCESS.md` is gitignored |
| Ignored files | `.env`, `db/admin.json` (scrypt hash only), `docs/ACCESS.md` are all covered by `.gitignore` (`git check-ignore` verified) |
| Runtime config | services do not auto-load `.env` (documented dev debt); the phase did not change this |

**F-04 — Supabase `service_role` key + project URL hardcoded in a tracked script.**
`backend/scripts/smoke_test_supabase.py` embedded a 219-character `service_role` JWT and the project URL as
literals (committed 2026-09-12 in `6e0f9f8`, i.e. pre-existing). The value equals the live `.env` credential, so
this is a live privileged credential in a public repository, not a sample. **Source remediated**: the script now
reads `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` from the environment/`.env` through the shared loader and exits with
a clear message when they are absent (verified by a stubbed-loader run). **Rotation is still required by the
owner**, because the key remains in git history; the phase did not rewrite history and did not contact the
service.

**F-05 — JWT signing secret + internal-token literal hardcoded in a tracked script.**
`backend/scripts/start_risk_loadtest.py` set `os.environ['JWT_SECRET']` to the live `.env` signing secret and
`INTERNAL_TOKEN` to a literal. Because `settings.jwt_secret` also derives the PII `fernet_key` and the export
signing key, this is the highest-impact secret exposure found in this phase. **Source remediated** the same way
(env/`.env`, fail-fast, no defaults in source). Rotation of the signing secret is an owner action and would
invalidate existing DB-1 PII ciphertext, so it was **not** performed here; the exposure is recorded as an
unresolved item (§19).

Regression guard: `backend/scripts/secret_hygiene_test.py` (new, §16) fails if any live `.env` value or
credential-shaped literal re-enters a tracked file.

---

## 12. Error-handling results

| Observation | Evidence |
|---|---|
| Generic 500 body, no stack trace, no path disclosure | every 500 observed (before remediation) returned `{"detail":"Internal server error"}` from the security-header middleware |
| Traceback capture | service logs contain **only** uvicorn access lines; the traceback for F-02/F-03 was captured through `TestClient(..., raise_server_exceptions=True)` stderr and is paraphrased in the findings, not written to any log file by this phase |
| F-02 | `POST /run-security-scan` on the success path raised `NameError: cannot access local variable 'env' where it is not associated with a value` — the `env` dict was defined *inside* the unreachable `if not script.exists():` branch. Fixed |
| F-03 | `Unhandled exception: Out of range float values are not JSON compliant: nan` — the framework's own 422 handler failed to serialise the rejected input. Fixed |
| `/fraud-report` with no generated data | returns **500** with an explicit body `{"error": "no report data - run scripts/generate_fraud_report.py"}` instead of 404/503. **Observation**: status-code semantics, not a leak; the data file is absent on this host so no report content is exposed. Left unchanged (not a security defect) |
| Operator-visible diagnostics | the explicit bodies above are deliberate and helpful; they name a script, never a key, path or token |
| Rejected-but-internally-executed check | replay probes confirm rejected requests (NaN, wrong token) produce **no** stored row and **no** score |

---

## 13. Strix findings

**None.** Strix v1.7.0 could not be executed against the target: Docker is not installed and cannot be installed
without administrator action in this environment (`ENVIRONMENT BLOCKED`, §3). No vulnerability is attributed to
Strix, and no Strix report is fabricated or paraphrased. The manual assessment in this document is the primary
and only evidence source for Phase 4. The exact re-run recipe is preserved in `.freebuff/STRIX_PHASE_RESUME.md`
and the attempt logs are kept as evidence.

---

## 14. Reproduction, and what "independent" means here

| Reproduction layer | Tool different from the original observation? | Result |
|---|---|---|
| Live socket reproduction | yes — `urllib` over TCP against 8000–8005 | F-01 reproduced unauthenticated (`200` → after fix `401`); F-03 reproduced (`500` → after fix `422`) |
| In-process reproduction | yes — `TestClient` against the ASGI apps, hermetic temp `DB_DIR` | F-02 reproduced with the full traceback; F-03 reproduced on risk **and** privacy |
| Suite reproduction | yes — new regression suites re-derive the same expectation | `front_service_test.py`, `risk_engine_test.py`, `secret_hygiene_test.py` all green |
| CI reproduction | yes — GitHub Actions re-runs Bandit/pip-audit/penetration suites on a clean checkout | see §23 |
| Independent third-party reproduction | **none available** | Strix blocked; no external assessor exists for this phase, and none is claimed |

Each finding therefore has at least two independent clients and one automated regression test behind it. This is
**self-reproduction**, not independent validation; the distinction is kept explicit everywhere in this document.

---

## 15. Remediation

Only changes required to remediate a confirmed finding were made; every one carries a regression test. No
scientific artifact (research plan, preregistration, threshold, dataset, benchmark, evaluation methodology) was
touched.

| ID | File | Change | Behavioural effect |
|---|---|---|---|
| F-01 | `backend/src/front_service/main.py` | `_require_admin_session(request)` added to `security_scan_history` / `security_scan_latest`; no other caller of these paths exists in the repo | unauthenticated callers now get 401; admin session unchanged (200) |
| F-02 | `backend/src/front_service/main.py` | `env = {**os.environ, "PYTHONIOENCODING": "utf-8"}` moved out of the unreachable `if not script.exists():` branch onto the success path | `/run-security-scan` returns 200 with a real scan instead of 500 |
| F-03 | `backend/src/middleware/__init__.py` (shared by all five services) | `RequestValidationError` handler + `_json_safe()` sanitiser → 422 with the offending value rendered as a string | non-finite input now yields 422 and a serialisable body on every service |
| F-04 | `backend/scripts/smoke_test_supabase.py` | literals → `os.environ` via the shared `.env` loader, fail-fast when absent | no credential in source; script exits loudly instead of using a baked-in key |
| F-05 | `backend/scripts/start_risk_loadtest.py` | same pattern for `INTERNAL_TOKEN` / `JWT_SECRET` | no credential in source; `JWT_SECRET` is no longer pinned in code |
| F-06 | `backend/src/front_service/main.py` | 14 per-line `# nosec B608 - <justification>` annotations on provably parameterised SQL (identifier from `sqlite_master` or `^[A-Za-z_][A-Za-z0-9_]*$`-validated, values bound via `?`), mirroring the existing `privacy_layer/main.py` precedent | **comment-only**; query text and parameters unchanged; Bandit medium+ gate green locally |
| F-06 | `.github/workflows/security-scan.yml`, `.github/workflows/ci-cd.yml` | new "Secret hygiene + SAST gate regression" step running `backend/scripts/secret_hygiene_test.py` in both security jobs | the gate and the hygiene invariants are now enforced in CI, not only locally |

Diff size: 8 modified files + 1 new script (plus the two documents and the evidence bundle) — see the closeout for
the file list.

---

## 16. Regression tests

| Suite | New coverage | Result |
|---|---|---|
| `backend/scripts/front_service_test.py` | unauthenticated `scan-history` / `scan-latest` → 401 (with cookies explicitly cleared, since admin login sets a session cookie on the test client); admin session → 200; `/run-security-scan` with `subprocess.run` monkeypatched → 200 and the child env carries `PYTHONIOENCODING=utf-8` | **ALL CHECKS PASSED** |
| `backend/scripts/risk_engine_test.py` | raw-body NaN / `+Inf` / `-Inf` / NaN-in-unconstrained-field → 422, no traceback and no `site-packages` path in the body | **ALL CHECKS PASSED** |
| `backend/scripts/secret_hygiene_test.py` (new) | no live `.env` value in tracked files (with a non-secret allowlist); no credential-shaped literal outside self-declared detector scripts (which must actually contain detector patterns); remediated scripts hold no literal credentials and fail fast; `bandit -r backend/src --severity-level medium` exits 0 | **15/15 PASS** |
| CI wiring | both security jobs now run the hygiene suite after Bandit, with `PYTHONIOENCODING=utf-8` | added in `security-scan.yml` and `ci-cd.yml` |

---

## 17. Rerun results (post-remediation, live)

Suite C was re-run against the restarted stack running the final code:

| Post-remediation check | Result |
|---|---|
| `scan-history` / `scan-latest` unauthenticated | **401** (was 200) |
| `scan-history` with an admin session | 200 |
| `POST /run-security-scan` with an admin session | **200** (was 500) |
| risk `/internal/evaluate` NaN / +Inf | **422** (was 500) |
| privacy `/internal/ingest-transaction` NaN amount | **422** (was 500) |
| replay probe on a rejected NaN event id | no row, no score — rejection never executed |
| corrected token probes: `evaluate-batch`, `attribution`, ingest, investigator GET/POST, forged compliance token | 401 / 422 as expected; forged compliance token reaches the handler and is refused |
| admin transaction detail: admin session vs user JWT | 200 vs 401 |
| **Suite C total** | **18 checks, 18 PASS, 0 FAIL, 0 OBSERVATION** |
| Bandit (CI command) | **exit 0** (was exit 1, 15 findings) |

Pre-remediation evidence was preserved before any fix: `.freebuff/phase4_results_a_PREFIX.json`,
`phase4_results_b_PREFIX.json`, `phase4_probe_PREFIX.json`. Nothing was overwritten.

---

## 18. Silent-control coverage matrix

The matrix answers the phase's central question — *does every path that reaches sensitive functionality enforce
the boundary?* — for the control families that exist in this stack.

| Control | Public route(s) | Alternate path tested | Enforcement point | Evidence |
|---|---|---|---|---|
| Admin session | `/admin/api/*` (data), `/admin/db-*` | user JWT, no session, wrong method, cookie without `X-Requested-With` | `_require_admin` (Depends) / in-body `_require_admin_session` (+CSRF) | suite A; suite C R-J4/R-J4b; guard grep of every call site |
| Scan-report access | `/security/scan-history`, `/security/scan-latest` | no session, `POST` variant | same session guard, added in F-01 | suite C V1/V2/V4/V5 |
| Internal/s2s token | `/internal/evaluate`, `/evaluate-batch`, `/attribution`, `/ingest-transaction`, privacy metadata routes | valid body + wrong token, empty list, malformed body (422 pre-handler) | `check_access` / `verify_internal_token` | suite B + suite C R-G4/R-G5/R-H2 |
| Compliance token | `/investigator/cases`, `/compliance/*`, `/audit/*` | no token, user JWT, forged token, malformed request | handler `hmac.compare_digest`; `require_compliance*` | suite C R-K1/R-K3/R-K7 |
| Inference validation | every route that reaches the model | alternate inference routes, non-finite values, `1e309`, negative/NaN amount, missing fields | pydantic model validators + feature-count checks; F-03 handler makes the failure observable | suite B G6–G9/H6; suite C V7–V10; replay probe |
| PII protection | identity PII read/write | direct DB read, admin DB explorer, compliance graph | fernet-encrypted at rest; PII tables denied in the explorer; masked admin row views | `k_anonymity_test`, `audit_test`, `penetration_test` (PASS) |
| Audit integrity | `/audit/*` reads, DB-4 writes | direct append attempt, trigger check | append-only triggers raise `IntegrityError`; hash chain verified by `/audit/integrity` | `audit_test`, `audit_hygiene_check` (PASS) |
| Rate limiting | login/register/admin-login/evaluate | burst re-sends | per-route limits (3/300s, 5/60s, 500/60s) | suite A auth-burst checks; `security_test` (PASS) |
| Headers / CORS / CSRF | all services | cross-origin preflight, cookie-authenticated state change | `apply_security_middleware` (CORS, OriginValidation, SecurityHeaders) + CSRF header rule | suite A; `phase75_security_hardening_test` (PASS) |
| Error containment | all | malformed input on every fuzz case | middleware generic 500; sanitised 422 | suite B, suite C; §12 |
| Ingestion/parser | dataset ingestion | traversal, nested archive, symlink, executable, oversized | ingestion inspector classifications | `phase107` suite, `security_attack_test` (PASS) |
| Secret hygiene | repository | new/updated source files | `secret_hygiene_test.py` (+ CI step) | 15/15 PASS |

**Coverage gaps stated plainly:** the inference/worker module has no listener in this topology (30 routes NOT
TESTED), Strix contributed nothing, and no external assessor reviewed these results.

---

## 19. Unresolved findings and residual risk

| Item | Status | Why it remains |
|---|---|---|
| F-04 Supabase `service_role` key | **Source remediated; rotation REQUIRES EXTERNAL ACTION** | the credential is still valid and still present in git history; only the owner can rotate it or rewrite history |
| F-05 JWT signing secret | **Source remediated; rotation REQUIRES EXTERNAL ACTION** | rotating it re-derives `fernet_key`, which would break decryption of existing DB-1 PII (documented dev debt); the owner must decide the migration |
| F-07 CI secret scanner | **Unblocked, result NOT yet observed** | the scanner had never run (skipped behind the failed Bandit step); it will execute on the next push, and `--only-verified` may flag the historical key until rotation |
| Dev `jwt_secret` default (25 bytes) and the derived keys | **Pre-existing known dev debt, intentionally unchanged** | changing it breaks existing ciphertext; out of scope for this phase |
| `/fraud-report` returns 500 when the report artifact is absent | **Observation, left unchanged** | explicit body, no leak, no report content present; a status-code cleanup, not a security defect |
| Unauthenticated HTML shells (`/admin*`, `/monitor-page`, `/fraud-report-page`) | **By design** | they serve markup only; the data APIs behind them reject unauthenticated callers (verified) |
| Unknown extra fields accepted; duplicate keys last-win | **Observations** | extra keys never enter the model matrix; validation runs after parsing |
| Inference/service module (30 routes) | **NOT TESTED** | no listener in the assessed topology |
| Deployment topology, TLS, reverse-proxy configuration, container hardening | **NOT TESTED** | outside the local-staging scope of this phase |
| Strix execution | **ENVIRONMENT BLOCKED** | Docker unavailable without administrator action |

---

## 20. Limitations

1. **Self-assessment.** The same author designed, executed and interpreted every probe. This is not an
   independent penetration test and must not be represented as one.
2. **Strix did not run.** The phase's named tool produced no data; findings are manual.
3. **Scope is the local stack.** No cloud, container, network, or production configuration was assessed.
4. **Time-boxed surface.** 164 routes were probed; deep multi-step business-logic abuse (e.g. long-lived review
   workflows, chained race conditions) was sampled via the existing suites, not exhaustively enumerated.
5. **Two-client reproduction is not independence.** Live-socket plus in-process clients share the same source.
6. **State-changing probes were kept reversible.** A handful of synthetic demo events/cases remain in the local
   demo databases — acceptable in a demo store, and none of it touches frozen evidence.
7. **Log coverage.** Service logs capture access lines only; exception text had to be captured through
   `TestClient` stderr.
8. **CI results depend on the host runner.** Local Bandit reproduction is exact for the SAST gate; other CI jobs
   (pip-audit, penetration suite, Docker build) are recorded from the runner, not reproduced here.

---

## 21. Evidence classification

| Item | Classification |
|---|---|
| F-01 unguarded scan-report routes | **CONFIRMED** (live, two clients, pre-remediation evidence preserved) → fixed → **retest PASS** |
| F-02 `/run-security-scan` NameError 500 | **CONFIRMED** (live + in-process traceback) → fixed → **retest PASS** |
| F-03 NaN/Inf → 500 instead of 422 | **CONFIRMED** (4 risk variants + 1 privacy case) → fixed → **retest PASS** |
| F-04 hardcoded Supabase `service_role` credential | **CONFIRMED** (containment scan, tracked file, live value) → source remediated → **rotation REQUIRES EXTERNAL ACTION** |
| F-05 hardcoded JWT signing secret + internal token | **CONFIRMED** → source remediated → **rotation REQUIRES EXTERNAL ACTION** |
| F-06 CI Bandit gate red (15 B608 false positives) | **CONFIRMED** (four CI runs + local reproduction) → remediated → **bandit exit 0** |
| F-07 CI secret scanner skipped | **CONFIRMED** (job conclusions) → unblocked → **result not yet observed** |
| Suite-B checks G4, G5, H2, K1, K3 | **FALSE POSITIVE / test artifact** — the token check is preceded by framework validation; corrected probes (R-G4, R-G5, R-H2, R-K1, R-K3) all pass. Reclassified and excluded from the finding set |
| Suite-C V10 (first run) | **NOT REPRODUCIBLE as a defect** — artifact of a non-unique event id reused by the run's own replay probe; unique ids now used, 18/18 PASS |
| Suite-A B4, suite-B J4, K6, L1, G16, H4, duplicate-key behaviour | **OBSERVATION** — no security impact, documented |
| `/fraud-report` 500 with explicit body | **OBSERVATION** |
| Never-executed path for rejected requests | **CONFIRMED SAFE** (replay probe: no stored row, no score) |
| Strix execution | **ENVIRONMENT BLOCKED** — no findings, none claimed |

---

## 22. Repository invariants (phase brief §22)

* Research Plan, preregistration, statistical/domain review decisions, dataset eligibility, benchmark data,
  production thresholds and scientific evaluation methodology: **unchanged**.
* No test was weakened, skipped or re-pointed to make a result pass; no assertion was deleted; the five
  suite-B "failures" were reclassified as test artifacts and then **re-probed correctly**, not silenced.
* Pre-existing working-tree modifications that predate this phase
  (`reports/calibration_test/calibration_metrics.json`, `reports/evaluation_runs/eval_ledger.jsonl` and the
  `record_eval-*.json` files) are the repository's own hook-driven evidence records — left as they are, not
  authored by this phase, and committed with the phase per the standing pattern.
* The freeze checker stays **RED by design**: `scripts/check_freeze.py` → rc 1 with 78 placeholder failures in
  `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md`. This phase did not attempt to green it.

---

## 23. Verification battery and CI verification

### 23.1 Local repository battery

`bash .freebuff/p114_battery.sh` (Phase-114 regression battery: phase suites 46→118, services/security suites,
security checkers, evidence enforcement, core regression suites):

| Result | Value |
|---|---|
| Entries | 82 |
| PASS | **82** |
| FAIL / MISSING | **0** |
| Completed | 2026-10-06T09:19:51+05:30 — `ALL CHECKS PASSED` |

Security-relevant suites inside it, all PASS: `penetration_test`, `security_test`, `security_attack_test`,
`sql_injection_test`, `adversarial_test`, `security_ci_gate`, `security_scan`, `audit_hygiene_check`,
`front_service_test`, `backup_restore_test`, `rules_gate_test`, `ood_gate_test`.

Checkers run directly:

| Checker | Result |
|---|---|
| `claim_evidence_check` | PASS |
| `eval_record_test` | PASS |
| `review_resolution_check` | PASS |
| `review_package_check` | **PASS** (13/13 sections; no licence cleared, no reviewer assigned, freeze state unchanged) |
| `check_freeze_test` | PASS (this is the *test of* the freeze checker) |
| `scripts/check_freeze.py` | **rc 1 / 78 failures — expected by design** (pre-freeze RED is the gate working) |
| `secret_hygiene_test.py` (new) | 15/15 PASS |
| Bandit (CI command) | exit 0 |

### 23.2 CI verification for the Phase-3 push (phase brief §24)

Inspected through the GitHub REST API (unauthenticated, public repository — `gh` CLI is not installed here).
Artifact/log *download* requires authentication (HTTP 401), so job and step **conclusions** are authoritative and
the Bandit failure was reproduced locally with the workflow's exact command.

| Workflow | Run id | Commit | Job | Steps | Classification |
|---|---|---|---|---|---|
| CI/CD | 37354366368 | `05d491b` | Test Suite | all steps success (full regression suite, claim evidence, eval-record schema, backup/restore) | **PASS** |
| CI/CD | 37354366368 | `05d491b` | Security Scan | **failure at "Run Bandit static analysis"**; pip-audit, custom scanner, penetration test, audit-hygiene skipped | **FAIL** |
| CI/CD | 37354366368 | `05d491b` | Integration Test / Docker Build / Deploy Image | skipped (dependency failed) | **BLOCKED** |
| Security Scan | 37354366248 | `05d491b` | Security Scan | failure at "Run Bandit (SAST)"; **"Check for Secrets" never ran** | **FAIL** |
| CI/CD | 37354126721 | `8cfa3ff` | Test Suite | all steps success | **PASS** |
| CI/CD | 37354126721 | `8cfa3ff` | Security Scan | failure at Bandit; downstream jobs skipped | **FAIL** |
| Security Scan | 37354126833 | `8cfa3ff` | Security Scan | failure at Bandit; secret check skipped | **FAIL** |
| Freeze Check | — | `05d491b` / `8cfa3ff` | — | not triggered (path filter excludes these commits by design) | **NOT RUN (by design)** |

Root cause of the CI failure: **15 Bandit B608 "hardcoded SQL expression" findings in
`backend/src/front_service/main.py`**, i.e. **false positives** on parameterised statements (placeholders built as
`?,?,?`; table names either read from `sqlite_master` or matched against `^[A-Za-z_][A-Za-z0-9_]*$`; column names
validated against the live schema; all values bound). This was a **genuine repository defect of the tooling gate**
(a red build nobody could act on, hiding the security steps behind it), not an application vulnerability. It was
remediated with justified per-line annotations and the gate is now green locally (`exit 0`).

The fix did **not** touch scientific state, thresholds, datasets or workflows beyond adding one verification step
to each security job.

### 23.3 CI verification of the Phase-4 remediation (post-remediation)

The Phase-4 commits were pushed and CI was re-inspected; the previously red jobs are now **green on
commit `941d0841`**:

| Workflow | Run id | Commit | Jobs | Steps | Classification |
|---|---|---|---|---|---|
| Security Scan | 37411380941 | `941d084` | Security Scan | all success — incl. **Run Bandit (SAST)**, the new **Secret hygiene + SAST gate regression**, Run Safety, and **Check for Secrets** (the step that had never executed before) | **PASS** |
| CI/CD | 37411380923 | `941d084` | Test Suite / Security Scan / Docker Build / Integration Test / Deploy Image | **all five jobs success, zero failed steps** — Bandit, secret hygiene, pip-audit, custom scanner, penetration test, audit-chain hygiene, non-root container check, live walkthrough, audit-chain verification, GHCR push | **PASS** |

So F-06 (Bandit gate) and F-07 (secret scanner never executed) are both **resolved and observed in CI**, not
inferred: the secret scanner ran in both workflows and passed. Note the scanner uses `--only-verified` and a
Supabase service key cannot be provider-verified, so this CI pass does **not** retire the credential-rotation item
in §19 — it only proves the instrument now runs.

---

## 24. Final decision

Findings observed: **7** (three application defects, two hardcoded-credential exposures, two CI-gate/instrumentation
defects). All confirmed findings are either fixed with a regression test and a passed retest, or explicitly
recorded as requiring external action (credential rotation) or external verification (Strix). No serious
unresolved application vulnerability remains in the tested scope; the unresolved items are credential rotation and
the unexecuted Strix tool.

**Decision (one of the six permitted options):**

> ## `SECURITY FINDINGS FIXED — RETEST PASSED`

**Mandatory bounded conclusion (verbatim):**

> `SELF-TESTED SECURITY ASSESSMENT — findings observed within the tested local/staging scope. This does not constitute an independent penetration test or guarantee absence of vulnerabilities.`

Stop condition (phase brief §26): Strix execution attempted and recorded as environment-blocked; findings classified
independently of the raw suite output; confirmed defects fixed with regression tests; reruns executed; CI status
recorded; assessment written; repository verification complete. **No optimization was started, no new benchmark was
created, and no institutional validation was performed.** The next phase is Phase 5 — Prototype Optimization,
constrained by the measured Phase-3 bottlenecks and by the findings above (in particular: keep the credential
rotation open item visible, and keep the CI security gate green).
