# PS-14 — Phase 7 (Production-Engine CI Coverage) Closeout

**Phase identity:** Phase 7 — Production-Engine CI Coverage
**Date:** 2026-10-07 (started and completed)
**Starting commit:** `39467add0114299794ca18c9cef4248d047d28fe`
**Final commit:** `ff29a8fec10fb5fc9c60ce1b99c97cc935bebe83` — the substantive
Phase-7 commit (fixture builder + fixture test + suite registration).
Documentation-only closeout commits follow it on `main`; per this repository's
convention a commit cannot embed its own CI result, so the run belonging to the
*last* documentation-only tip is reported in the hand-off (every pushed tip of
this phase was CI-verified — see §CI).
**Final verdict:** **`PASS WITH LIMITATIONS`**

Scope: make a fresh CI checkout execute the **native production-engine code
path** (loader → engine → inference → deterministic assertion) via a small
deterministic test fixture. No production artifacts, no model/ensemble/
calibration/threshold/rule changes, no performance claims, no other phase's
work.

## Commit / SHA ledger

| Item | SHA |
|---|---|
| Branch | `main` |
| Starting SHA | `39467add0114299794ca18c9cef4248d047d28fe` |
| **Phase-7 implementation commit** | **`ff29a8fec10fb5fc9c60ce1b99c97cc935bebe83`** |
| CI verified on that SHA | `CI/CD #152` **success** (5/5 jobs) + `Security Scan #156` **success** |
| Ending Git SHA | this documentation-only closeout commit — the final phase tip on `main` (a commit cannot embed its own SHA) |
| Remote | `https://github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software.git` |

Diff composition (git-reviewed before commit): **3 files, +567 lines** — two
new scripts (`backend/scripts/native_fixture_build.py`,
`backend/scripts/native_fixture_test.py`) and a **+5-line** entry in
`backend/scripts/regression_suite.py`. **Zero files under `backend/src/` were
modified**, no binaries, no production artifacts, no secrets; the timer
records under `reports/` remain uncommitted as required. The `review_package_check.py`
R4 sha-pins (`native_features.py`, `altman_native_ensemble.py`) are therefore
untouched and still verify.

---

## Problem (the Phase 6 limitation this phase fixes)

Phase 6 closed with `PASS WITH LIMITATIONS`, the limitation being:

> Native production artifact loading is `BLOCKED` from a clean checkout
> because `models/production/` is gitignored (`.gitignore:70/95`).

Consequence: a GitHub checkout has no `models/production/manifest.json`, so
`src/risk_engine/main.py`'s manifest-gated selection never reaches the native
branch and the risk engine boots the **fallback** engine
(`AltmanEnsembleEngine`/`FusionEngine`). CI therefore exercised dependency
*installation and imports* but never executed the production native path —
`AltmanNativeEnsembleEngine.__init__` (joblib load of the four native
artifacts), native member inference, the native ensemble composition and the
native-only `predict_combined_many` contract were all **unverified on
CI's Linux/Python 3.12 leg**. The Phase-6 closeout recorded artifact loading
as `BLOCKED` and handed it to this phase, explicitly *not* to be solved by
committing production artifacts.

## Architecture (as inspected, §2 — not assumed)

**Engine selection** (`backend/src/risk_engine/main.py`, lifespan, ~:259–:330):

1. Phase-49 release verification runs first if
   `PRODUCTION_DIR/release_manifest.json` exists (fixture root has none →
   legacy/dev mode, attestation `None`, runtime `READY` after load);
2. `PRODUCTION_DIR/manifest.json` → native is selected **iff**
   `model_type == "xgb_lgb_cb_native"` →
   `fusion = AltmanNativeEnsembleEngine(PRODUCTION_DIR / "altman_native")`;
3. else `xgb_production.joblib` + `lgb_production.joblib` present →
   `AltmanEnsembleEngine(PRODUCTION_DIR)`;
4. else `FusionEngine(ARTIFACTS_DIR)`.

There is **no environment variable or other runtime switch** for engine
selection anywhere in the repository (searched) — selection is manifest-driven
only. `PRODUCTION_DIR` is a module-level global read *at lifespan runtime*,
which makes it the narrowest possible injection point: the test points it at
the fixture root and every production code path (release check, manifest gate,
loader, inference) executes unmodified. **No production function is stubbed,
no prediction is mocked.**

**Native engine** (`backend/src/risk_engine/altman_native_ensemble.py`):
`__init__(model_dir)` does four real `joblib.load`s — `xgb_native.joblib`,
`lgb_native.joblib`, `cb_native.joblib`, `scaler_native.joblib` — plus the
optional calibrator at `model_dir.parent/artifacts/calibrator.joblib` (absent
in the fixture → `calibrator is None`, a supported state) and
`model_dir/manifest.json` (`model_version`, `n_features`, `locked_threshold`).
Inference: `map_raw_to_native` (path 1: raw native columns → the shared
`derive_native_features` derivation used by the retrain; path 3: §16 ML
feature proxy) → **float32 cast before scaling** → `RobustScaler.transform` →
three `predict_proba` → weighted mean 0.34/0.33/0.33 → `clip(0,1)` →
`(prob, dict)` with `individual_outputs {xgboost, lightgbm, catboost}`,
`model_type: "altman_native"`. `predict_combined_many` (:257) is the Phase-5
bit-identical batched contract (float64 widening cast).

**Fallback engines** (contrast for identity): `AltmanEnsembleEngine`
(15/21-feature contract, `_verify_schema` fail-closed) and `FusionEngine`
(ML_FEATURES) — **neither exposes `predict_combined_many`** (verified by
search), so that method, the `model_type` property
(`altman_native_xgb_lgb_cb`), the class name, the `model_version` string and
the loaded directory are five independent identity signals.

**Existing tests:** `risk_engine_test.py`'s Phase-5 block asserts
`predict_combined_many` exactness *only when the native engine is loaded* and
prints a `[SKIP]` in a fresh checkout (fallback loaded);
`feature_parity_test.py` exits `SKIPPED` on a fresh checkout entirely. Neither
was modified.

## Fixture (§3) — Option A, generated at test time

| Aspect | Value |
|---|---|
| Builder (committed) | `backend/scripts/native_fixture_build.py` (`--out DIR [--seed 42]`) |
| Test (committed) | `backend/scripts/native_fixture_test.py` (52 checks) |
| Location at runtime | OS temp dir (`tempfile.mkdtemp(prefix="ps14-native-fixture-")`) — never inside the repository, never under `models/production/`; the builder **refuses** a `production` path or a non-empty output dir |
| Committed binaries | **none** — artifacts (~20 KB total: xgb 7,453 B, lgb 7,304 B, cb ~4,921 B, scaler 858 B) are generated fresh per run and discarded |
| Model types | `XGBClassifier` (8 trees, depth 3, lr 0.5, `tree_method=hist`, `n_jobs=1`), `LGBMClassifier` (8 trees, depth 3, lr 0.5, `num_threads=1`, `deterministic=True`), `CatBoostClassifier` (8 iterations, depth 3, lr 0.5, `thread_count=1`), `RobustScaler` fit on the fixture rows — mirroring production preprocessing |
| Training data | `np.random.default_rng(42)`: 256 × 48 `float32` standard-normal rows + labels from a fixed linear rule plus a noise draw from the same stream; both classes asserted present. **No repository data, no network, no production rows** |
| Feature schema | the production 48-feature contract — `ALTMAN_NATIVE_FEATURES` is *imported* from the engine module (not re-typed) and written into the fixture manifest |
| Determinism controls | one seeded RNG stream; model seeds all = `--seed`; single-threaded training in all three libraries; `sort_keys` JSON manifests; no timestamps/hostnames/paths embedded; fixed joblib compression |
| Manifests | selection manifest: `model_type: "xgb_lgb_cb_native"` + `fixture: true` + `purpose: "TEST FIXTURE - software-path verification only. NOT a production model, NOT production weights, NOT model-performance evidence."`; engine manifest: `model_version: native_fixture_ci_v1`, `n_features: 48`, `locked_threshold: 0.5` |

**Why this is not production evidence (§4):** the fixture exists solely to run
the software path. Its labels come from a synthetic linear rule over random
numbers; it is 256 rows and 8 trees; its outputs are never used for any
accuracy, threshold, calibration or effectiveness statement. What it
demonstrates: loader functionality, dependency compatibility (the Phase-6
pins), native engine execution, schema compatibility, deterministic output
and CI integration — nothing about fraud-detection quality.

## Native execution (§5, §6, §7) — all three members + ensemble + identity

Recorded from the deterministic fixture (seed 42, corpus row 0; values
reproduced by re-running the committed builder — not invented):

| Item | Result |
|---|---|
| Loader | `AltmanNativeEnsembleEngine(fixture/altman_native)` loads all four joblibs; class `XGBClassifier` / `LGBMClassifier` / `CatBoostClassifier` confirmed by type; scaler `RobustScaler(48)` |
| XGBoost member | `predict_proba` shape `(5, 2)`, finite, ∈ [0,1], rows sum to 1; row-0 output **`0.09323325753211975`** (rounded 4 dp `0.0932`) |
| LightGBM member | shape `(5, 2)`, finite, ∈ [0,1], rows sum to 1; row-0 output **`0.16780565912478676`** (`0.1678`) |
| CatBoost member | shape `(5, 2)`, finite, ∈ [0,1], rows sum to 1; row-0 output **`0.0600980054321593`** (`0.0601`) |
| Ensemble (direct engine) | **`0.10690751686471292`** — asserted **exactly equal** to `clip(0.34·xgb + 0.33·lgb + 0.33·cb)` recomputed from `components()` (no tolerance used) |
| Ensemble (via service) | `POST /internal/evaluate` → `200`, `ml_score = 0.1069` == `round(direct_prob, 4)` exactly, `degraded: false`, repeated call identical |
| Runtime | lifespan reached `runtime_state == READY` with the fixture loaded; startup log prints `Loaded Altman-NATIVE ensemble: native_fixture_ci_v1` |

**Engine identity verification (§6)** — five independent signals, all asserted
in-process after the real lifespan ran with `PRODUCTION_DIR` pointed at the
fixture:

1. `type(fusion).__name__ == "AltmanNativeEnsembleEngine"`;
2. `fusion.model_type == "altman_native_xgb_lgb_cb"`;
3. `fusion.model_version == "native_fixture_ci_v1"` — **≠** the production
   version `altman_native_E_hardneg_cert_20260904`, proving the fixture (not
   an accidentally present local `models/production/`) was loaded;
4. `fusion._model_dir` resolves under the temp fixture root — disjoint from
   the repository's `models/production/`;
5. `hasattr(fusion, "predict_combined_many")` — native-only API; both fallback
   engines lack it.

A fallback substitution (or a selection regression) fails checks 1, 2, 5; a
production-artifact read fails checks 3, 4. The test would have failed if the
manifest gate had not fired, if the loader had raised, or if any member could
not predict.

## Batch consistency (§9)

- `predict_combined_many(corpus)` vs `[predict_combined(row) for row in corpus]`:
  scores **bit-identical** (`max_d = 0.0`), uncertainty dicts **exactly
  equal**, batched scores finite — the Phase-5 contract, asserted on the
  fixture, which means it now also executes in CI (it previously skipped
  there because only the native engine has the method).
- The Phase-5 block in `risk_engine_test.py` and its production-artifact
  assertions were **not modified or weakened**; `risk_engine_test` still
  asserts strict equivalence whenever the real native model is present and
  keeps its endpoint-level single-vs-batch checks either way.
- This phase grants no license to alter Phase-5 behaviour; no Phase-5 code
  changed.

## Fresh checkout (§10)

Two independent proofs that CI no longer depends on an accidentally present
`models/production/`:

1. **The test itself is location-disjoint:** it asserts the loaded engine's
   `_model_dir` is under its own temp fixture root (checks 3–4 above), so it
   passes whether or not a local `models/production/` exists — it never reads
   it (the run prints `models/production present in this checkout: True …
   (informational - this test never reads it)` locally, and the assertion set
   is identical when absent).
2. **CI-faithful local reproduction** (`.freebuff/p7/verify.sh`, log
   `.freebuff/p7/verify.log` — gitignored scratch, results quoted here):
   - `git clone --depth 1` of `ff29a8f` → `test ! -d models/production` **OK**
     (checkout has no native artifacts);
   - fresh `python -m venv` + `pip install -r backend/requirements.txt -c
     backend/constraints.txt` (the Phase-6 mechanism) → `pip check` clean;
   - CI's `generate_synthetic_data.py` + `train_compare.py` steps pass;
   - **standalone `native_fixture_test.py` → `ALL CHECKS PASSED`** on the
     bare checkout;
   - `regression_suite.py --fast` → **24/24 PASSED (45.9 s)**;
   - `claim_evidence_check.py` PASS (22), `eval_record_test.py` 22/22,
     `backup_restore_test.py` PASS — all CI test-job steps green locally;
   - script exit: `ALL FRESH-CHECKOUT STEPS PASSED` (`verify_rc=0`).

The production-artifact strategy itself remains untouched: `.gitignore` keeps
`models/production/` ignored, nothing was force-added, and the gitignore rule
was not modified.

## CI (§12)

| Workflow | Run (id) | SHA | Result | Detail |
|---|---|---|---|---|
| **CI/CD** | **#152** (`37627454507`) | `ff29a8f` | **success**, 12 m 33 s (13:18:26–13:30:59 UTC) | Test Suite **2 m 9 s** · Security Scan 1 m 14 s · Docker Build 3 m 53 s · Integration Test 3 m 9 s · Deploy Image 1 m 50 s — **5/5 jobs green** |
| **Security Scan** | **#156** (`37627454575`) | `ff29a8f` | **success**, 1 m 25 s | bandit + pip-audit + trufflehog + hygiene gates |

- **Workflow:** `.github/workflows/ci-cd.yml`; **job:** `Test Suite`
  (step *Run full regression suite* — the existing runner, no new workflow
  step and no second dependency system was introduced);
- **OS / Python:** `ubuntu-latest`, `PYTHON_VERSION: '3.12'`, pip cache,
  install exactly `pip install -r backend/requirements.txt -c
  backend/constraints.txt`;
- **Fixture generation method:** committed seeded builder executed at test
  time inside the suite (no downloads, no committed binaries);
- **Native engines exercised:** XGBoost 3.4.1, LightGBM 4.7.0, CatBoost
  1.2.10 (the Phase-6 pins) + scikit-learn `RobustScaler`;
- **Test result:** `native_fixture` is registered in `FAST_TESTS`
  (23 → 24 suites); `regression_suite.py` exits `1` if any suite fails or
  times out, so the green Test Suite job **proves** the fixture suite ran and
  passed on Linux/CPython 3.12 — including the Phase-5 bit-identical batch
  assertion and cross-build regeneration determinism on that platform.
  Circumstantially consistent: Test Suite 2 m 9 s vs Phase 6's 1 m 25 s–2 m 1 s
  (one extra ~7–15 s suite).

CI job logs remain sign-in gated (403 anonymous); per-job outcomes were read
from the public Actions run page and the public API after its rate-limit
reset — the same access pattern and caveat as Phase 6.

## Regression (§13)

| Check | Environment | Result |
|---|---|---|
| `native_fixture_test.py` standalone | dev venv (local `models/production` present, unused) | **52/52 checks, ALL CHECKS PASSED** |
| `native_fixture_test.py` standalone | fresh clone `ff29a8f`, fresh venv, **no `models/production`** | **ALL CHECKS PASSED** |
| `regression_suite.py --fast` (24 suites, incl. new `native_fixture` 6.8 s) | dev venv | **24/24 PASSED (50.9 s)** |
| `regression_suite.py --fast` | fresh clone + fresh venv (CI reproduction) | **24/24 PASSED (45.9 s)** |
| Full battery `.freebuff/p114_battery.sh` (stack up, ports 8000–8005) | dev machine | **84/84 PASS, 0 FAIL** (18:19:58→18:37:42 +05:30) |
| `risk_engine_test.py` (unchanged; includes Phase-5 block) | both fast runs | ALL CHECKS PASSED |
| `feature_parity_test.py` (deployed artifacts present locally) | dev venv | 11/11 PASSED |
| `claim_evidence_check.py` | local + fresh clone | PASS (22 claims) |
| `eval_record_test.py` | local + fresh clone | 22/22 |
| `review_resolution_check.py` | dev venv | PASS (14/14) |
| `review_package_check.py` (R4 sha-pins) | dev venv | PASS (13/13) — no pinned file edited |
| `check_freeze.py` | repo root | rc=1 (78 placeholders) — **expected pre-freeze state, deliberately not fixed** |

## Security (§15)

- `secret_hygiene_test.py`: **16/16** with the two new files *staged* (they
  contain no credential-shaped literals; the test-process env tokens mirror
  the established `risk_engine_test.py` literals), plus the CI gate
  `bandit -r backend/src --severity-level medium` **exit 0** — and
  `backend/src/` is unchanged this phase.
- CI `Security Scan #156`: **success** (bandit + pip-audit + trufflehog +
  secret hygiene + custom scanner + penetration test + audit-chain hygiene on
  the pushed tree).
- Fixture supply chain: generation uses only already-pinned, already-installed
  dependencies and the standard-library RNG; **no network fetches, no
  downloads of model files, no unpinned installs, no shell-outs to external
  scripts, no secrets, no real/proprietary data** (§15 satisfied by
  construction).
- No CVE scanning performed or claimed (explicitly out of scope — Phase 12).

## Evidence classification (§17)

| Evidence | Classification |
|---|---|
| Native loader executes on a clean checkout (4× `joblib.load`, real estimator classes) | `DEMONSTRATED` |
| XGBoost / LightGBM / CatBoost member inference (shape, finiteness, probability range, row sums) | `DEMONSTRATED` |
| Ensemble composition (exact weighted combination) and service-level `/internal/evaluate` inference through the native engine | `DEMONSTRATED` |
| Engine identity selected by `main.py`'s real manifest gate (5 signals; fallback/artifact substitution detectable) | `DEMONSTRATED` |
| Deterministic output: bit-identical repeat, exact cross-build regeneration, byte-identical artifacts (except the documented cb metadata byte) | `DEMONSTRATED` |
| Phase-5 batch contract (`predict_combined_many` == per-event loop) on the native path in CI | `DEMONSTRATED` |
| Fresh-checkout execution independent of `models/production/` (local clone reproduction + CI run on `ff29a8f`) | `DEMONSTRATED` |
| Pinned dependency set installs and runs the native stack on Linux/CPython 3.12 (CI) | `DEMONSTRATED` |
| The fixture test itself (self-test of the software path) | `SELF-TESTED` |
| Fixture training data / labels (tiny synthetic RNG corpus) | `SIMULATED` |
| Fixture model quality, accuracy, generalization | `NOT ESTABLISHED` (and not attempted — out of scope) |
| Real-world / production fraud-detection effectiveness of the fixture | `NOT ESTABLISHED` |
| Production **artifact bytes** loading in CI (real `models/production/`) | `BLOCKED` by design boundary — `.gitignore` rule preserved; artifact strategy explicitly remains a separate concern (§11). Recorded, not bypassed. (Real-world validation generally remains `BLOCKED` per prior phases; untouched here.) |

The fixture is classified and labelled (`purpose` field in both manifests,
docstrings in both scripts) as **software-path verification only** — never as
model-performance evidence.

## Limitations (§17)

1. **The fixture is not the production model.** CI executes the native code
   path against generated artifacts; the real production bytes
   (`models/production/`, gitignored) are still never loaded by CI. That
   separation is deliberate (§11: committing production artifacts to make a
   test pass was rejected) and remains a separate concern.
2. **Fixture data is synthetic/test-only** (256 RNG rows, linear-rule
   labels). Nothing measured on it says anything about fraud detection.
3. **No real-world effectiveness, generalization or institutional
   validation** is claimed — `NOT ESTABLISHED`/`BLOCKED` exactly as before.
4. **No live-server performance measurement** was taken (out of scope).
5. **No cross-platform pinned golden floats.** Expected values are derived
   *from the fixture itself* by same-host regeneration (exact, zero
   tolerance) rather than hard-coded cross-OS constants: Linux and Windows
   could differ in last-ulp model arithmetic, and pinning numbers a platform
   cannot reproduce would be a fabricated assertion. All equality claims in
   the test are **exact** — no loose tolerance is used anywhere.
6. **CatBoost joblib bytes differ by one byte between two builds**
   (serialization metadata); its member *predictions* are bit-identical and
   that is the asserted contract. xgboost/lightgbm/scaler/manifests are
   asserted byte-identical.
7. **CI job logs are sign-in gated**, so the per-suite stdout line
   (`native_fixture ... PASS`) could not be quoted; the green Test Suite job
   plus `regression_suite`'s exit-1-on-any-failure semantics is the evidence
   used (same access caveat as Phase 6).
8. **`risk_engine_test.py`'s native branch still skips in CI** — it is bound
   to the *real* production artifacts and was deliberately not rewired;
   engine-level coverage in CI comes from the new fixture suite instead. Both
   run in every fast suite.
9. **`feature_parity_test.py` still exits SKIPPED on a fresh checkout** —
   parity against deployed artifacts is a different question from code-path
   execution and was out of scope.
10. **No type checking, because the project has none** (recorded in Phase 6;
    unchanged).
11. **The final documentation-only tip's own CI run is reported in the
    hand-off**, not embedded here — a commit cannot contain its own CI result
    (repository convention; no placeholder is left behind).

## Final verdict (§17)

# `PASS WITH LIMITATIONS`

**Why not `BLOCKED`:** the Phase-6 blocker is closed. A fresh CI checkout now
runs the real native loader, all three native members, the exact ensemble
composition, the Phase-5 batch contract and `main.py`'s manifest-gated native
selection — asserted by 52 checks that fail if any of it does not happen — and
`CI/CD #152` (5/5 jobs, ubuntu-latest/Python 3.12) plus
`Security Scan #156` are green on `ff29a8f`. All 15 success criteria of the
brief are met: no production artifacts committed, no gitignore change, no
mocks around loading/inference, identity explicitly verified, determinism
demonstrated, existing regression/security gates all green (24/24 fast both
environments, 84/84 battery, hygiene/checkers green).

**Why not a plain `PASS`:** non-critical limitations remain and are recorded
rather than smoothed over — the fixture deliberately does not prove loading
of the real production artifact bytes (separate scope by §11), fixture data
and outputs are `SIMULATED` and carry no performance meaning, cross-platform
golden values were not (and could not honestly be) pinned, CI job logs are
sign-in gated, and no live-server or real-world evidence was produced.
