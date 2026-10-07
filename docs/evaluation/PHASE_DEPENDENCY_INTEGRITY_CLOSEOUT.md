# PS-14 — Phase 6 (Dependency & Fresh-Environment Integrity) Closeout

**Phase identity:** Phase 6 — Dependency & Fresh-Environment Integrity.
Scope: dependency declarations, dependency reproducibility, and proving a fresh
environment can install the declared set and load the native production engine.
No other phase's fixes are included. Started 2026-10-07.

**Final decision:** `DEPENDENCY INTEGRITY PASS WITH LIMITATIONS`
(see §20).

## Commit / SHA ledger

| Item | SHA |
|---|---|
| Branch | `main` |
| Starting SHA (phase start / pre-change tree) | `24a8d3eb52a5897cfb5139984c9a70ec49d37aac` |
| Manifest of the native-engine defect | `69cad0b` / `24a8d3e` (code review source of truth) |
| Phase-6 implementation + evidence + docs commit | `<this commit>` — SHA recorded in `git log` / the CI-record commit below (a commit cannot embed its own SHA) |
| Final phase SHA | the phase-6 delivery commit on `main`; the CI outcome is appended afterwards in a **documentation-only** commit |
| Remote | `https://github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software.git` |

Git history was not rewritten. No model artifacts, databases, logs, caches or
virtual environments were committed (see §L).

---

## Part A — Inspection (pre-change state)

**The single dependency source was `backend/requirements.txt`.** There was **no
lock file, no constraints file, no `pyproject.toml`, no `setup.py`, no
`Pipfile`, and no `environment.yml`** anywhere in the project. (A second,
unrelated `requirements.txt` lives in `misc/dist/ps14-scoring-pipeline/` — a
separate bundled scoring-pipeline distribution with its own README; it is not
the application's dependency source and was left untouched.)

| Aspect | Pre-change state |
|---|---|
| Declared dependencies | 20 entries |
| Pinned (exact `==`) | 2 (`scikit-learn==1.9.0`, `xgboost==3.4.1`) |
| Lock / constraints mechanism | **none** |
| Python version | CI env `PYTHON_VERSION: '3.12'`; `backend/Dockerfile` `python:3.12-slim`; **not stated in any in-repo dependency file** |
| Dependency-drift evidence | same requirements resolved to different patch versions in CI vs the dev venv (e.g. numpy 2.5.3 vs 2.5.2, pandas 3.0.6 vs 3.0.5, fastapi 0.142.2 vs 0.141.1) |

**Install sites that consume the requirements file (pre-change, all
`pip install -r backend/requirements.txt`):**

- `.github/workflows/ci-cd.yml` — lines 39 (test), 88 (security), 153 (build), 215 (integration)
- `.github/workflows/security-scan.yml` — line 39
- `backend/Dockerfile` — line 11 (`RUN pip install --no-cache-dir -r backend/requirements.txt`)

**Native-engine import path (why declarations matter).**
`backend/src/risk_engine/main.py` loads `AltmanNativeEnsembleEngine`
(`altman_native_ensemble.py`) and falls back to `AltmanEnsembleEngine`
(`altman_ensemble.py`). Neither module contains a literal `import lightgbm` /
`import catboost`; they `joblib.load()` **pickled `LGBMClassifier` /
`CatBoostClassifier` members**, so both distributions must be importable at
*unpickle* time. `backend/src/inference/realtime_scorer.py:28` additionally holds
a plain top-level `import lightgbm as lgb`.

**Repository-wide search result.** `lightgbm` and `catboost` are imported by
training scripts under `backend/scripts/` and by `realtime_scorer.py`, and are
required to unpickle the production artifacts — but **neither was declared**.

**Third-party import sweep of `backend/src` (every top-level import examined).**
Two categories emerged:

1. **Native-engine imports missing from the manifest** (the review's findings):
   `lightgbm`, `catboost`.
2. **A third undeclared runtime import found during this sweep: `redis`.** It is
   imported *unconditionally at module top level* by `src/redis_conn.py:27`,
   `src/inference/rate_limiter.py:35`, `src/inference/redis_state.py:28`
   (also `from redis.sentinel import Sentinel`) and
   `src/inference/worker_pool.py:44`, and `src/inference/__init__.py` imports
   `redis_state` — so `import src.inference` (and therefore the deployed
   `inference` service in `backend/docker-compose.prod.yml`) fails without it.
   The documented "graceful in-memory fallback" only applies *after* the import
   succeeds. It was present in the dev venv only as a transitive dependency of
   the test-only `fakeredis`.

No other undeclared runtime import exists: `python-multipart`, `pyotp`,
`jinja2`, `aiofiles`, `prometheus_client` and `requests` are **not** imported by
`backend/src` (TOTP and Prometheus handling are hand-rolled; `scipy` is imported
only by `backend/scripts/` analysis tools and arrives transitively via
scikit-learn/catboost anyway).

---

## Part B — Native-engine declarations

`backend/requirements.txt` now declares every runtime dependency the native
production engine (and the inference service) needs. Each newly declared
package has a direct import / model-loading justification:

| Dependency | Why it is required | Evidence |
|---|---|---|
| `lightgbm==4.7.0` | top-level import in shipped inference code; and the native engine unpickles an `LGBMClassifier` | `backend/src/inference/realtime_scorer.py:28`; `altman_native_ensemble.py` (`self.lgb.predict_proba`); `altman_ensemble.py:419/502` |
| `catboost==1.2.10` | native engine unpickles a `CatBoostClassifier` | `altman_native_ensemble.py` (`self.cb.predict_proba`); `altman_ensemble.py:239/506` (manifest-gated `cb` member) |
| `redis==8.1.0` | unconditional top-level import in shipped modules; `import src.inference` depends on it | `src/redis_conn.py:27`; `src/inference/rate_limiter.py:35`; `src/inference/redis_state.py:28`; `src/inference/worker_pool.py:44` |

Existing correct declarations were preserved; nothing was removed. No package
was added merely because it happened to exist in the developer virtualenv —
`fakeredis`, `bandit`, `pip-audit`, `safety`, `semgrep` and the analysis-only
`scipy` remain intentionally undeclared (test/CI tooling, not runtime deps).

**Versions were not guessed.** `lightgbm 4.7.0` / `catboost 1.2.10` are the
versions the committed artifacts were trained with (and the versions the current
resolver independently selects), and `redis 8.1.0` is the current resolution.

---

## Part C — Reproducible dependency versions

**Strategy: exact-pin every direct dependency, plus a generated full-closure
lock file.**

1. `backend/requirements.txt` — all **23** direct runtime dependencies are
   exact-pinned (`==`).
2. `backend/constraints.txt` — **new, generated** lock pinning the complete
   resolved transitive closure (**61** distributions). Header records the exact
   regeneration command.
3. **Python version is explicit**: the requirements header states CPython 3.12
   (developed on 3.12.10); CI pins `PYTHON_VERSION: '3.12'`; the image is
   `python:3.12-slim`.
4. **One install command everywhere:**
   `pip install -r backend/requirements.txt -c backend/constraints.txt`.
5. **No dependency upgrades.** The pins are the resolution the repository's own
   install command already produces — i.e. a *freeze of the set CI validates*,
   not a version bump. `scikit-learn` (1.9.0) and `xgboost` (3.4.1) keep their
   pre-existing pins because the committed artifacts are pickled against them;
   the header warns against casual bumps of the four ML packages.
6. **Why not freeze the dev venv verbatim?** The developer virtualenv had
   *drifted older* (numpy 2.5.2, pandas 3.0.5, fastapi 0.141.1, …) **and** held
   `PyJWT 2.13.0`, which carries published advisories (PYSEC-2026-4140…4152,
   CVE-2026-102275). Freezing that verbatim would have imported vulnerabilities
   into CI and broken the existing `pip-audit` gate. The lock therefore
   standardises on the resolution that satisfies *every* existing gate, and
   removes the drift that the review recorded.

**No unrelated application code was refactored.** The diff touches dependency
declarations, the CI/Docker install commands that must not bypass them, and two
install instructions.

---

## Part D — Fresh-environment verification

A genuinely clean environment (`python -m venv`, empty site-packages, CPython
3.12.10, pip 25.0.1) was created **outside** the dev venv and installed with the
repository's declared mechanism only.

| Check | Result |
|---|---|
| Python version | **3.12.10** (matches CI's `3.12` line and the image) |
| Install command | `pip install -r backend/requirements.txt -c backend/constraints.txt` |
| Install result | success — **61** distributions, exactly the constrained set |
| `pip check` | **`No broken requirements found.`** |
| `import lightgbm` | **OK** (4.7.0) |
| `import catboost` | **OK** (1.2.10) |
| `import xgboost` | **OK** (3.4.1) |
| Native engine modules | **OK** — `src.risk_engine.altman_native_ensemble`, `src.risk_engine.altman_ensemble` |
| `realtime_scorer` | **OK** — `src.inference.realtime_scorer` (plus `src.inference.service`, `src.redis_conn`, `src.privacy_layer.velocity_tracker`) |
| All six service entrypoints | **OK** — risk / privacy / identity / audit / verification / front `main` |
| Import checks summary | **27 / 27 OK, 0 failures** (recorded in `misc/reports/phase6_dependency_fresh_env.json`) |
| Native model-loading boundary | **reached and loadable locally** — `AltmanNativeEnsembleEngine(models/production/altman_native)` loaded `XGBClassifier` + `LGBMClassifier` + `CatBoostClassifier` (48 features, `altman_native_E_hardneg_cert_20260904`) |

**Dependency installation vs native artifact loading are reported separately.**
`models/production/` is gitignored (`.gitignore:95`), so a clean GitHub checkout
has no native artifacts. The engine loaded here **only because this local
checkout still holds them**; that is not reproducible from a clean clone and is
classified `BLOCKED` in §Part J / §19. No model artifacts were committed to make
this test pass.

---

## Part E — Reproducibility check

A **second** clean environment was created and installed with the *same*
repository-controlled mechanism.

| Item | env #1 | env #2 |
|---|---|---|
| Python | 3.12.10 | 3.12.10 |
| Install command | `-r requirements.txt -c constraints.txt` | identical |
| Distributions installed | 61 | 61 |
| `pip check` | clean | clean |

- **`env1` vs `env2` package set: byte-identical** (`diff` empty).
- **Installed set vs `backend/constraints.txt`: exact match** (nothing resolved
  that the lock did not pin, nothing pinned that the lock did not install).
- Because the lock pins the whole closure, resolution is deterministic rather
  than "works on my machine"; there were **no differences to reconcile**.

---

## Part F — CI / Docker consistency

Only the minimum dependency-installation changes were made so that CI and Docker
cannot bypass the specification. **CI was not redesigned**, and the Phase-7
native-engine CI fixture was **not** implemented.

| File | Change |
|---|---|
| `.github/workflows/ci-cd.yml` | all 4 install steps now `pip install -r backend/requirements.txt -c backend/constraints.txt` (test / security / build / integration) |
| `.github/workflows/security-scan.yml` | install step now uses `-r … -c …` (its `safety check --file backend/requirements.txt` continues to read the same manifest) |
| `backend/Dockerfile` | `COPY backend/requirements.txt backend/constraints.txt ./backend/` and `RUN pip install --no-cache-dir -r backend/requirements.txt -c backend/constraints.txt` |

Result: one clean checkout, CI, and the Docker image all resolve the **same**
dependency set. The CI security job still installs `bandit` / `pip-audit`
explicitly (those are scanner tools, deliberately not constrained).

---

## Part G — Regression verification

| Suite / check | Environment | Result |
|---|---|---|
| `pip check` | clean env #1, #2 | **clean** |
| Import battery (27 modules) | clean env #1 | **27/27 OK** |
| `backend/scripts/regression_suite.py --fast` (23 suites — the CI gate) | **clean locked env #1** | **23/23 PASSED** (41.2 s) |
| `backend/scripts/regression_suite.py --fast` | dev venv (final tree) | **23/23 PASSED** |
| `risk_engine_test.py` | clean env #1 | **ALL CHECKS PASSED** (exercises the native engine batch path) |
| `claim_evidence_check.py` | clean env #1 | **PASS** (22 claims verified) |
| `eval_record_test.py` | clean env #1 | **22/22 passed** |
| `review_resolution_check.py` | clean env #1 | **PASS** (14/14) |
| `review_package_check.py` | clean env #1 | **PASS** (13/13) |
| `secret_hygiene_test.py` | dev venv (has the CI `bandit` tool) | **16 passed, 0 failed** |
| `scripts/check_freeze.py` | clean env #1 | **rc=1** — expected pre-freeze (draft plan placeholders, no FREEZE_RECORD); **not a regression** |
| Full Phase-114 battery (84 suites) | dev venv, six services up | **84/84 PASS, 0 FAIL** (`misc/reports/phase6_dependency_battery_results.tsv`) |

`secret_hygiene_test.py` returns rc=1 in a clean venv because it asserts the
`bandit` scanner is installed, and prints the remediation itself
(`pip install bandit`); CI installs `bandit` in the security job. `bandit` is a
CI tool, **not** a runtime dependency, so it was deliberately not added to the
manifest.

**Battery environmental note.** The battery was run twice. A first run with the
six local services **down** gave 81/84 — the three failures
(`security_test`, `sql_injection_test`, `security_ci_gate`) are live-stack HTTP
suites; each was re-run with the stack up and passed (16/16, 50/50, and rc=0
respectively), and the second full run with the stack up gave **84/84 PASS,
0 FAIL**. The stack-down run is retained for transparency as
`misc/reports/phase6_dependency_battery_stackdown.tsv`. None of the three is
affected by the dependency declarations (they exercise HTTP endpoints and do
not import the added packages).

---

## Part H — Security review of the dependency changes

| Check | Result |
|---|---|
| No secrets / credentials added | **confirmed** — diff contains no secret-shaped literal; `.env` untouched and still gitignored |
| No untrusted package source introduced | **confirmed** — no new index, no `--extra-index-url`, no URL/VCS requirement; PyPI only |
| No dependency replaced with an unrelated package | **confirmed** — every added name is the legitimate upstream package for its import |
| Package names are the real upstream packages | **confirmed** — `lightgbm` (LightGBM), `catboost` (CatBoost), `redis` (redis-py) |
| Existing security controls remain green | `bandit` (medium+) gate unchanged; `secret_hygiene_test` 16/16; the `pip-audit` CI gate is **clean on the new lock** (`No known vulnerabilities found`, exit 0 — recorded in `misc/reports/phase6_dependency_audit.txt`) |

Pinning `PyJWT==2.15.1` (not the dev venv's advisory-laden 2.13.0) is exactly
what keeps the existing `pip-audit` gate green. **CVE coverage is not claimed —
dependency vulnerability scanning is Phase 12.** The audit above only
demonstrates that the *existing* gate does not regress.

---

## Part I — Documentation

Only documentation that became factually incorrect was changed:

- `README.md` — the setup block said `pip install -r requirements.txt` (wrong
  path: there is no root requirements file) and "Requires Python 3.10+". Now
  `pip install -r backend/requirements.txt -c backend/constraints.txt` and
  "Requires Python 3.12".
- `docs/admin_console_staging_deploy.md` — the install lines pointed at the same
  non-existent root file; corrected to the real path plus the lock.

Deliberately **not** touched (broader cleanup deferred): README suite counts
("22 FAST suites"), the stale "79 battery checks" figure, historical evaluation
docs, and the separate `misc/dist` distribution README.

Recorded facts: dependency strategy (pin direct + generated closure lock),
Python 3.12, native ML versions (`lightgbm 4.7.0`, `catboost 1.2.10`,
`xgboost 3.4.1`, `scikit-learn 1.9.0`), the regeneration command, the
verification commands, and the limitations in §19.

---

## Part J and §17 — Before / after dependency state

| | Before | After |
|---|---|---|
| Manifest | `backend/requirements.txt` | same |
| Direct dependencies declared | 20 | **23** |
| Direct deps exact-pinned | 2 | **23** |
| Lock / constraints file | none | **`backend/constraints.txt` (61 pins)** |
| Python version stated in the dependency file | no | **yes (3.12 / 3.12.10)** |
| `lightgbm` | undeclared | **`lightgbm==4.7.0`** |
| `catboost` | undeclared | **`catboost==1.2.10`** |
| `redis` | undeclared | **`redis==8.1.0`** |
| CI installers bypassing the specification | 5 | **0** |
| Fresh-env reproducibility | untested | **2 clean envs, identical** |

### Evidence classification

| Claim | Classification |
|---|---|
| Declared dependency set installs in a clean environment (twice, identically) | **DEMONSTRATED** |
| `pip check` clean on the locked set | **DEMONSTRATED** |
| `lightgbm` / `catboost` / `xgboost` / `redis` import in a clean environment | **DEMONSTRATED** |
| Native engine **modules** import in a clean environment | **DEMONSTRATED** |
| All six service entrypoints import in a clean environment | **DEMONSTRATED** |
| Resolution is deterministic (env1 ≡ env2 ≡ constraints) | **DEMONSTRATED** |
| Fast regression suite (23 suites) green on the locked set | **SELF-TESTED** (project's own suite) |
| Risk-engine, evidence, review-resolution and hygiene suites green | **SELF-TESTED** |
| Full Phase-114 battery (84 suites) green on the final tree | **SELF-TESTED** |
| Native **artifact** loading from a clean checkout | **BLOCKED** — `models/production/` is gitignored; no artifacts exist in a clean clone (it *was* demonstrated locally against the developer's artifacts) |
| CI (Linux) resolves and tests the same set | **NOT ESTABLISHED at writing time** — to be confirmed by the CI run recorded after the push (§CI record) |
| Vulnerability/CVE coverage of the dependency set | **NOT ESTABLISHED** — Phase 12 |

---

## Part K — Required items 1–20

1. **Phase identity** — Phase 6, Dependency & Fresh-Environment Integrity (above).
2. **Starting Git SHA** — `24a8d3eb52a5897cfb5139984c9a70ec49d37aac`.
3. **Ending Git SHA** — the phase-6 delivery commit on `main` (ledger above; the
   CI-record commit follows it).
4. **Files changed** — `backend/requirements.txt`, `backend/constraints.txt`
   (new), `.github/workflows/ci-cd.yml`, `.github/workflows/security-scan.yml`,
   `backend/Dockerfile`, `README.md`,
   `docs/admin_console_staging_deploy.md`, `misc/reports/phase6_dependency_*`
   (evidence: `_state.json`, `_fresh_env.json`, `_audit.txt`,
   `_battery_results.tsv`, `_battery_stackdown.tsv`), this document.
5. **Dependencies added** — `lightgbm`, `catboost`, `redis`.
6. **Dependency versions** — direct pins: lightgbm 4.7.0, catboost 1.2.10,
   redis 8.1.0, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.0, xgboost 3.4.1,
   joblib 1.6.0, matplotlib 3.11.2, fastapi 0.142.2, uvicorn 0.54.0,
   sqlalchemy 2.1.3, psycopg2-binary 2.9.13, pydantic 2.13.5,
   pydantic-settings 2.15.0, email-validator 2.3.0, PyJWT 2.15.1,
   argon2-cffi 25.1.0, cryptography 50.0.2, PyYAML 6.0.3, httpx 0.28.1,
   python-pptx 1.0.2, alembic 1.20.0. Full closure (61 pins) in
   `backend/constraints.txt`.
7. **Dependency-resolution strategy** — exact-pin every direct dependency +
   generated full-closure constraints file; one install command for dev/CI/Docker
   (Part C).
8. **Why each dependency was required** — Part B table (import / unpickle
   evidence for each of the three additions).
9. **Fresh-environment installation result** — Part D: clean CPython 3.12.10
   venv, 61 distributions installed from the declared mechanism, success.
10. **`pip check` result** — `No broken requirements found.` (both clean envs).
11. **Native import results** — 27/27 imports OK, including `lightgbm`,
    `catboost`, `xgboost`, the native engine modules and `realtime_scorer`.
12. **Native artifact-loading result** — loadable against the developer's local
    (gitignored) artifacts; **`BLOCKED` from a clean checkout** (no artifacts).
13. **CI / Docker dependency consistency** — Part F: all four CI jobs, the
    security-scan workflow and the Docker image now consume the pinned manifest
    *and* the lock.
14. **Tests executed** — Part G table (fast suite ×2, risk engine, four
    checkers, secret hygiene, `check_freeze`, Phase-114 battery).
15. **Test results** — fast regression suite **23/23** (both the clean locked env
    and the dev venv), full Phase-114 battery **84/84** on the final tree,
    risk-engine / claim-evidence / eval-record / review-resolution /
    review-package suites green, `secret_hygiene_test` **16/16**, `pip check`
    clean. Two documented non-regressions: `secret_hygiene_test` needs the CI
    `bandit` scanner in a bare venv (prints its own remediation), and
    `check_freeze` is rc=1 pre-freeze **by design**.
16. **Security / hygiene verification** — Part H: no secrets, no new indexes,
    legitimate package names, existing gates green, `pip-audit` clean on the lock.
17. **Before / after dependency state** — Part J table.
18. **Evidence classification** — Part J table (`DEMONSTRATED` / `SELF-TESTED` /
    `NOT ESTABLISHED` / `BLOCKED`).
19. **Limitations** — §19 below.
20. **Final decision** — `DEPENDENCY INTEGRITY PASS WITH LIMITATIONS` (§20).

---

## Part L — Git discipline

- Full diff reviewed: **6 tracked files modified + 1 new** (`backend/constraints.txt`)
  + evidence JSON/TXT + this document. No unrelated files changed.
- **No model artifacts committed** (`models/` stays ignored); **no databases,
  logs, caches, virtualenvs or generated junk** — the two fresh environments and
  the `pip` resolution reports live under the gitignored `.freebuff/p6/` scratch.
- `git diff --check` clean; no line-ending churn (content-only diffs).
- Relevant tests re-run **from the final tree** (Part G) after the last edit.
- Timer-generated calibration records (`reports/evaluation_runs/record_eval-*.json`,
  `eval_ledger.jsonl`, `reports/calibration_test/calibration_metrics.json`) are
  left **uncommitted** and are not part of this phase's changes.

---

## §19 — Limitations

1. **Native artifact loading cannot be proven from a clean clone.**
   `models/production/` is gitignored (`.gitignore:95`), so a fresh GitHub
   checkout has no native ensemble to load; the risk engine loads the fallback
   fused engine there. This phase proves dependency *installation and import*;
   artifact loading is `BLOCKED` from a clean checkout and was only demonstrated
   locally. Committing the large artifacts to make the test pass was explicitly
   rejected.
2. **CI/Linux resolution is asserted, not locally demonstrated.** The lock was
   resolved and installed on Windows/CPython 3.12.10. The Linux CI run after the
   push is the authority for the CI leg; every pinned distribution is a standard
   manylinux/cp312-compatible release.
3. **`redis` is declared but never exercised for a live connection** here — the
   declaration fixes the import failure; the graceful in-memory fallback still
   depends on the import succeeding, which is now possible.
4. **A pre-existing local artifact inconsistency was observed and left alone:**
   `AltmanEnsembleEngine(models/production)` raises a feature-schema mismatch
   against a stale release dir in the developer's checkout. This is artifact
   state, not a dependency defect, and the active `altman_native` engine loads
   correctly. Fixing/validating native artifacts in CI is the Phase-7
   native-engine fixture — out of scope here.
5. **`scikit-learn` is held at 1.9.0 while the current resolver would pick
   1.9.1**, deliberately: the committed artifacts are pickled against 1.9.0, and
   bumping it is an unnecessary upgrade in this phase (header warns against it).
6. **`secret_hygiene_test.py` requires the `bandit` scanner**, which is installed
   by CI's security job rather than declared as a runtime dependency. In a bare
   requirements-only venv it exits 1 with its own remediation message.
7. **No CVE claim.** The `pip-audit` result only shows the existing gate does not
   regress; vulnerability scanning is Phase 12.
8. **Stale documentation elsewhere was intentionally left** (README suite counts,
   historical evaluation docs) — broader documentation cleanup is a later phase.

---

## §20 — Final decision

`DEPENDENCY INTEGRITY PASS WITH LIMITATIONS`

**Why PASS:** every runtime dependency required by the native production engine
(and the shipped inference service) is now explicitly declared — `lightgbm`,
`catboost`, and the additionally-discovered `redis` — with per-package import /
model-loading evidence; the repository now has a deterministic, single-source
dependency mechanism (exact direct pins + a generated 61-package closure lock)
consumed identically by a clean install, CI and Docker; two independent clean
environments installed the declared set with **identical** results and a clean
`pip check`; all native/service modules import; and the existing test and
security gates stay green, with the fast regression suite **23/23** on the
locked set.

**Why "WITH LIMITATIONS":** native **artifact** loading remains `BLOCKED` from a
clean checkout because `models/production/` is gitignored — the dependency layer
is proven, artifact provenance/loading in CI is not (Phase 7). CI/Linux
resolution is asserted from a Windows-authored lock and awaits the post-push CI
run, and no CVE coverage is claimed (Phase 12).
