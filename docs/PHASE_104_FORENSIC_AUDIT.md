# Phase 104 — Forensic Audit and Remaining-Work Design

**Date:** 2026-10-03
**Scope:** Read-only audit + planning phase. No model, dataset, or production code was
modified. No contradictions were silently repaired. No metrics were fabricated.
Existing code and executable artifacts take precedence over historical phase reports.

**Evidence rule used throughout:** a claim counts as *measured* only when an artifact
on disk ties the number to a dataset, split, seed, and command. Otherwise it is
classified `NOT ESTABLISHED` regardless of how often it is repeated in documentation.

---

## PART A — REPOSITORY FORENSIC AUDIT

### A.1 Repository identity

| Item | Value |
|---|---|
| Remote | github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software |
| Branch | `main` |
| HEAD | `a0b600b` (repo phase 118) |
| Working tree | **clean** (`git status --porcelain` empty at audit time) |
| Commits | 139; first commit 2026-09-05 |
| Tracked files | 1440 (2.72 MiB) — backend/src 157, backend/scripts 422, misc 719, models 92, docs 3, reports 8, frontend 12, .github 2 |
| Runtime | Python 3.12.10, venv at `.venv/` |
| Naming note | Repo-phase numbering runs 1–118; "Phase 104" here is this audit's own label. Existing repo files `backend/src/monitoring/phase104_external_dataset_report.py` etc. are unrelated (repo-phase-104 = external dataset qualification). No path collision with `docs/PHASE_104_*`. |

### A.2 Scale

| Area | Size |
|---|---|
| `backend/src` | ~63k LOC — monitoring 42.5k, risk_engine 5.3k, front 4.8k, inference 4.7k |
| `backend/scripts` | 404 Python files, 154.7k LOC |
| Test scripts | 115 `*_test.py` files (plain-assert suites, no pytest) |
| Services | 6 FastAPI services (see B.1) |
| Hygiene | 8 **tracked** `.pyc` files under `misc/experiments/phase24.../__pycache__`; 200 `.pyc` on disk overall |

### A.3 Commands (verified; cwd must be `backend/` for suite scripts)

| Purpose | Command |
|---|---|
| Run one suite | `cd backend && PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe scripts/<name>_test.py` (`PYTHONIOENCODING=utf-8` is required — several suites print `✓` and crash on cp1252 otherwise) |
| Full battery (79 checks, phases 46–118 + core suites) | `bash .freebuff/p114_battery.sh` → tally `.freebuff/p114rq_battery_results.tsv`, log `.freebuff/p114rq_battery.log` |
| Training | `backend/scripts/training_pipeline.py` (`--full`), `train_compare.py`, `ibm_train.py` |
| Evaluation | `backend/scripts/compare_ml_systems.py`, `metric_definitions.py`, `scripts/simulate_rules.py` |
| Synthetic data | `backend/scripts/generate_synthetic_data.py` |
| Deploy | `docker-compose.yml` (single image, `MODULE`/`PORT` env), `docker-compose.prod.yml` (Postgres), `scripts/docker_retrain.sh` / `make retrain` |
| Stack (preview) | `.freebuff/start_stack.ps1` — front 8000, identity 8001, privacy 8002, risk 8003, verify 8004, audit 8005 |

Note: README/AGENTS.md paths of the form `python scripts/...` are valid only with
`cwd=backend`; there is no root-level `scripts/` or `src/`.

### A.4 Test battery state

- Last full battery run **2026-09-28: 79/79 PASS**, log tail `ALL CHECKS PASSED`
  (re-verified this session: tally has 79 lines, zero non-PASS).
- Suites re-run this session: `phase86` 77/77, `eval_record_test` 22/22,
  `leakage_structural_test` 123/123, `audit_hygiene` CLEAN (rc=0).
- **Battery does NOT run** `eval_record_test`, claim-audit, or result-matrix suites.
- Suites are self-reported pass/fail scripts; "79/79 PASS" is a **SELF-TESTED** result.

### A.5 CI configuration (`.github/workflows/ci-cd.yml`)

- **test job:** generate_synthetic_data → `train_compare.py` → `regression_suite.py --fast`
  (22 FAST suites of 29) → backup_restore.
- **security job:** bandit, pip-audit, `security_scan`, `penetration_test`, `audit_hygiene`.
- **then:** docker build, integration, deploy to GHCR.
- **Absent:** any phase suite (grep `phase` = 0 matches in workflow), `claim_audit`,
  `eval_record_test`, full (7 non-fast) suites, k-anonymity, calibration.
- **Consequence:** CI **cannot** reject "metric appears in documentation but has no
  evidence/run record." This is gap **G-01** (see gap matrix).

### A.6 Datastores / configuration

- SQLite stores under `db/` (features, risk, audit, identity/verification, sessions + WAL).
- PostgreSQL only in `docker-compose.prod.yml` (`postgres:16`, `DATABASE_URL`) with
  a single alembic migration (`alembic/versions/001`).
- `.gitignore` excludes `data/`, `models/artifacts/`, `models/production/`, `db/`, `.env`
  → **release manifests and model artifacts are not in git**; a fresh clone has no
  trained model (CI re-trains on synthetic data only).
- Services do not auto-load `.env`; only the front service merges it manually.

---

## PART B — ARCHITECTURE AND CODE INVENTORY

### B.1 Runtime services (all six live in the preview stack)

| Service | Port | Runtime? | Notes |
|---|---|---|---|
| `front_service` | 8000 | yes | landing page, server-side `/status` aggregate of the other five `/health` endpoints, passphrase-gated admin |
| `identity_service` | 8001 | yes | identity/PII (DB-1, Fernet) |
| `privacy_layer` | 8002 | yes | §16 feature derivation, velocity windows, 7 bank features |
| `risk_engine` | 8003 | yes | rules + fusion + gating + limits + attribution |
| `verification_service` | 8004 | yes | verification/trail UI |
| `audit_service` | 8005 | yes | DB-4 writer is imported directly (shared file), not called over HTTP |

### B.2 Component classification (selected)

| Component | Runtime/test-only | Active | Doc matches impl? | Notes |
|---|---|---|---|---|
| Rules engine + velocity limits (`rules.yaml`, `risk_engine/limits.py`) | runtime | yes | yes | carried every live decision (ML scores 0.0 on OOD scenarios) |
| Fusion (LR+RF+XGB+IF → Platt) | runtime | yes | partial | per-row `predict` slow; batched `predict_many` required offline |
| Altman native 48-feature model | runtime (deployed) | yes | partial | `feature_schema_version` string in model record says `altman_native_v2`; canonical is **v1** (docs/authoritative_contracts.md §5) |
| `enforce_before_inference` | runtime | yes | yes | applied at `/evaluate` (L853) and `/evaluate-batch` (L1121) |
| `/internal/attribution` | runtime | **gap** | no | calls `fusion.predict` with only an internal-token check — **no `enforce_before_inference`, no Phase-73 `check_access`** (gap G-05) |
| k-anonymity checker (`src/k_anonymity/checker.py`) | **script-only** | no | **no** | not wired into `/audit/export` (main.py:335) — README claims "k-anonymity gate on data exports" (gap G-08) |
| Audit writer (`src/audit_service/writer.py`) | runtime | yes | yes | async queue (maxsize 10000, single writer thread, `BEGIN IMMEDIATE`); Phase 77 fixed prev_hash races; **on the detection critical path** (DB-4 failure fails fraud decisions for writes) |
| Observability (Phase 70/72) | runtime | yes | yes | in-memory, **lost on restart** (gap G-13) |
| Duplicate trail UIs | runtime ×2 | yes | n/a | `verification_service/static/index.html` and `audit_service/static/index.html` are independent reimplementations — any fix must be applied twice |
| `src/inference` (4.7k LOC) | mixed | unclear | partial | consolidation candidate |
| Monitoring phase scripts | mostly test/script | no | n/a | 42.5k LOC, 100+ `phase*` modules; battery runs 79 |
| `misc/` | archive | no | n/a | 719 tracked files: reports for repo-phases 1–25 only; phases 26–103 exist only as README prose (49 `**Phase N**` lines) |

### B.3 Documentation-vs-runtime mismatches found

1. README claims k-anonymity gate on exports — **not reachable at runtime** (G-08).
2. README claims `docs/PS-14_fraud_detection.pptx` — **file missing**.
3. README claims `docs/ACCESS.md.enc` — **file missing**.
4. README claims root-level `scripts/deploy.sh` etc. — they live under `backend/scripts`.
5. README phase table covers only 14–25 and 38–56; phases 57–103 in prose; **0 mentions of repo phases 104–118** — README is stale relative to HEAD.
6. README headline metric provenance errors — see Part F.
7. Malformed markdown table row at README line ~251 (`|| **41** ... || **42**`).

**No deletions performed.** Consolidation candidates: Part Q.

---

## PART C — MODEL AND FEATURE RECONCILIATION

### C.1 Model inventory

| Model | Features | Training data | Split | Calibration | Threshold | Artifact | Active |
|---|---|---|---|---|---|---|---|
| **altman_native** (deployed) | 48 native | IBM v2-derived, hard-negative mined (`b01fa323…`, seed 42) | per model record; **manifest says `training_seed: null`** | Platt (calibrator.joblib) | **locked 0.018758** | `models/production/altman_native/` (xgb+lgb+cb .34/.33/.33, hashes present) | **yes** — release `legacy-altman_native_E_hardneg_cert_20260904`, verdict `LEGACY_ATTESTED` |
| **E_hardneg** | (same lineage) | hard-negative mining | — | — | — | `models/model_records/altman_native_E_hardneg_cert_20260904.json` | attested legacy |
| **P20_45feat** | 45 (`phase20/model_artifacts/feature_list.json`) | phase20/21 | phase21 harness | — | — | `misc/reports/phase20/…` | **inactive**; phase23b degraded 33/45 on Kaggle → ROC 0.47; reconstruction declared *IMPOSSIBLE* for real datasets (phase21) |
| **IBM v2 21-feature fusion** (LR+RF+XGB+IF) | 21 domain | IBM v2 | `ibm_train.py`: **user-disjoint row-range 70/15/15, explicitly NOT chronological** (L246–250) | Platt | 0.43 reported in README | eval artifacts under `misc/reports/` | research/eval only |
| Rules-only fallback | n/a | n/a | n/a | n/a | rules.yaml | runtime degraded path | yes (fail-open) |
| Ensemble components | all inside fusion | | | | | | |

Reproducibility gaps in the deployed release manifest: `source_git_sha: "unknown"`,
`training_config_hash` all-e0e0, empty `python_version`/`dependency_fingerprint`,
`training_seed: null`, signature = **HMAC with a hardcoded key** (Part N).

### C.2 Feature vectors — are they contradictory?

**No. They are three distinct contract layers plus one historical candidate:**

| Vector | Count | Canonical definition | Role |
|---|---|---|---|
| §16 domain | **21** | `ML_FEATURE_ORDER` in `monitoring/feature_contract.py` (feature_version **v1**); derived in `privacy_layer/features.py::derive_section16` | production input contract |
| Altman native | **48** | `ALTMAN_NATIVE_FEATURES` in `risk_engine/altman_native_ensemble.py` + `privacy_layer/native_features.py` (duplicated in `monitoring/model_contract_reconciliation.py`) | deployed model contract |
| P20 | 45 | phase20/21 artifacts | historical candidate, inactive |
| public_v1 | **14** | `backend/research/public_feature_contract.json` (phase65, 13/13 assertions) | research contract for public data — **exists but unused by any executed experiment** |

- **Overlap between the 21 and the 48: zero.** They are not versions of one another;
  they are different layers (privacy-layer output vs model input).
- Transformation is explicit: `map_raw_to_native()` with three paths —
  (a) raw native columns → shared `derive_native_features` (parity path, production);
  (b) already-derived native dict; (c) legacy `_from_ml_features` proxy 21→48 with
  defaults (`amt = ratio*100`, `chip = new_device_flag`, `ref_amt = 100`) — a **lossy,
  documented fallback**.
- Phase 86 test (`model_contract_reconciliation`) re-run this session: **77/77 PASS**,
  verdict `RECONCILED_WITH_TRANSFORMATION`.
- The stale `altman_native_v2` schema string in the model record is a **documentation
  defect**, not a second contract.

### C.3 The seven bank-only features

| Feature | Implementation | Inputs | Public data? |
|---|---|---|---|
| unusual_location_flag | `privacy_layer/features.py` | profile `usual_locations` | **no** |
| unusual_recipient_flag | idem | profile `usual_recipients` | **no** |
| failed_auth_count_24h | idem | auth event log | **no** |
| shared_device_accounts | idem | cross-account device graph (capped `GRAPH_CAP`) | **no** |
| shared_recipient_accounts | idem | cross-account recipient graph | **no** |
| mule_ring_score | idem | link-analysis graph | **no** |
| recipient_novelty | idem | recipient history | **no** |

- Contract specs in `monitoring/feature_contract.py`, freshness in `feature_freshness.py`.
- **Runtime usage:** consumed by *rules* (`rules.yaml` lines 54/57/78/126/128/137/141/148/172/192)
  and by the **21-feature IBM v2 model** (metadata n_features 21) — **not** by the
  deployed 48-native altman model.
- Phase 64 conclusion: **no public dataset provides them** (evidence: phase61–65 and
  the phase88 blocker chain).

### C.4 OPTION A vs OPTION B — evidence, NO silent choice

**OPTION A — reduced public_v1 research contract (14 features).**
- Evidence for: contract already exists and is tested (`public_feature_contract.json`,
  13/13 assertions, phase65); enables immediate public-data benchmarking instead of
  waiting for bank data; aligns evaluation with features that are actually derivable
  outside the bank.
- Evidence against / risk: 14 features exclude all 7 bank features *and* most of the
  48-native space — a public result would validate a **different, weaker model** than
  the deployed one; transfer from 14-feature research to 48-feature production would
  itself need justification.

**OPTION B — defer full-feature validation until bank data.**
- Evidence for: phase64 shows no public dataset carries the 7 features; phase23b shows
  partial reconstruction degrades to ROC 0.47; phase21 declares real-data reconstruction
  impossible; honest claim scope is preserved.
- Evidence against / risk: the project then has **no executable external validation
  path**, and headline claims remain in-domain synthetic-only for an indefinite period;
  bank data acquisition is outside the team's control (may never arrive).

**Decision required by a human reviewer before Gate 2 experiments.** Recorded as gap
**G-03**; the choice is NOT made in this phase.

---

## PART D — DATASET AND PROVENANCE AUDIT

| Dataset | Location | Size/hash | Provenance | Labels/split | Role | Metrics actually generated |
|---|---|---|---|---|---|---|
| ULB creditcard | `data/` (150 MB, sha `76274b69…`) | real, public Kaggle | PCA-anonymized; **0/21 §16 features mappable** | time column; ad-hoc splits in push reports | in-domain benchmark only | `misc/reports/ulb_max_push.json` (time-split xgb AUC **0.966404**, r1 0.815), `ulb_results.json` (0.9758 / 0.8835 / 0.918), `ML_BENCHMARK_REGISTRY.json` (5-fold 0.9849±0.0075) |
| IBM v2 | `data/` (2.35 GB, sha `b01fa323…`) | public, licensed per `data_provenance.json` | user-disjoint 70/15/15 (**not chronological**) | development + deployed-model training | 0.982 (XGB) / 0.978 fused; cross-dataset 0.873 (`reports/cross_dataset/cross_dataset_report.json` = 0.8726) |
| Kaggle fraud (fraudTrain/Test) | `data/kaggle_fraud/` | 1.85M/555K | provenance recorded | external transfer | 0.435 (phase24), →0.595 (phase25) — **backed by** `misc/reports/phase24/kaggle/05_e_hardneg_metrics.json`, `misc/reports/phase25/PHASE25_FINAL_REPORT.md` |
| PaySim | `data/paysim` | public | recorded | external | cross-dataset reports |
| ealtman2019, fraud_data.csv | `data/` | public | recorded | external/aux | misc reports |
| Synthetic PS-14 | `data/transactions.csv` (9,799 rows, sha `9f0f56bf…`) | generated | generator script | **default input of `train_compare.py`** | `metrics_comparison.csv` (fused 0.9838) — synthetic, not ULB |
| Fed shards | `data/fed/` | synthetic federated | recorded | federated tests | federated suite |
| **IEEE-CIS** | not present | — | auth-blocked | — | **BLOCKED** (3/48 features derivable) |
| **NeurIPS 2022 BAF** | `data/external_benchmark/baf/` **does not exist** | — | registry entry exists (`phase106_public_benchmark_registry.py:648`): license **NOT documented**, `semantics_established: false`, `acquisition_status: not_acquired`, class `PUBLIC_SYNTHETIC` | — | **BLOCKED / not acquired** — phase107 expects `base.csv`; do not download during this phase |
| Provenance sweep | `misc/reports/phase23/` (26 artifacts) | — | **114 sources examined, 0 meet provenance standard** | — | — |

Suitability summary: **development** → synthetic + IBM v2; **validation** → IBM v2
(user-disjoint) + ULB time-split; **final testing** → none yet qualifies (no
pre-registered, untouched test set exists — gap G-04); **external replication** →
Kaggle fraud (done, degraded), BAF (blocked), IEEE-CIS (blocked).

---

## PART E — REPRODUCIBILITY AND EVIDENCE

### E.1 Evaluation ledger

- `reports/evaluation_runs/eval_ledger.jsonl` contains **exactly 1 record**:
  `eval-20260914T175546` — a 400-row **smoke fixture**, git `3928144`, seed 42.
- `eval_record_test.py` **22/22 PASS** (re-run this session): the *machinery* works;
  the *coverage* is one smoke record.
- `metric_definitions.py` v1.0 defines ROC, PR, recall@1%FPR, threshold-at-FPR-on-val,
  baselines, bootstrap CI — but has **0 mentions of Brier/ECE** (no reliability-diagram
  metric defined).
- `calibration_and_stress.json` (the artifact that actually computes calibration):
  brier val 0.001101 / test 0.001185; ece val 0.00385 / test 0.003918;
  platt brier_test 0.000469, ece_test 0.000376; isotonic ece_test 6.7e-05.

### E.2 Can CI reject an unevidenced metric?

**No.** Reasons: (1) no phase suites or claim-audit in the workflow; (2) `eval_record_test`
not run in CI; (3) README/docs are not parsed against the ledger; (4) the ledger has
1 record for dozens of documented numbers. → gap **G-01**.

---

## PART F — CLAIM AUDIT

### F.1 Headline numbers

| Claim (README) | Where | Artifact evidence | Classification |
|---|---|---|---|
| ROC-AUC **0.966** on ULB | lines 15/33/114 | nearest: `ulb_max_push.json` time-split **0.966404** (r1 0.815, not 0.918); README line 114 attributes it to `train_compare.py`, but train_compare defaults to **synthetic** `data/transactions.csv` and never references ULB | **NOT ESTABLISHED** as a provenance chain (a near value exists in an ad-hoc push report; the attribution is wrong) |
| Recall@1%FPR **91.8%** | 15/34 | 0.918367 in `ulb_results.json` (pattern_xgb, split unstated); also hardcoded in `compare_ml_systems.py` SYSTEMS dict labeled "MEASURED from live codebase" — it is a **literal** | **NOT ESTABLISHED** as documented |
| PR-AUC **0.877** | 15/35 | 0.8835 in `ulb_results.json`/registry; 0.877 appears as a **hardcoded literal** in `compare_ml_systems.py` and in `phase6b_frontier.json` (dense threshold sweep — not a headline eval record) | **NOT ESTABLISHED** as documented |
| The exact 0.966/91.8/0.877/97.8%@0.43/1.996% combination | 33–37, 86 | **no single artifact on disk contains it**; components co-occur only in the hardcoded dict | **NOT ESTABLISHED** |
| Brier **0.0009**, ECE **0.0013** | 86/124/574 | only literal outside README is a **docstring string** in `phase71_architecture_gap_audit.py`; actual `calibration_and_stress.json` values are 0.001185/0.003918 (raw) and 0.000469/0.000376 (platt) | **NOT ESTABLISHED** as stated |
| External ROC-AUC **0.435 → 0.595** | 140 | `phase24/kaggle/05_e_hardneg_metrics.json`, `phase25/PHASE25_FINAL_REPORT.md` (FPR 44.9→9.9, 47/48 features reconstructed) | **DEMONSTRATED** |
| IBM cross-dataset **0.873** | 59 | `cross_dataset_report.json` (0.8726) | **DEMONSTRATED** (same-generator-family caveat already in README) |
| IBM v2 0.982 / 0.978 | 15 | ibm_train artifacts | **SELF-TESTED** (user-disjoint, not chronological split) |
| "k-anonymity gate on data exports" | README | not wired (B.3) | **FALSE** (doc/runtime mismatch) |
| Leakage audit "123/123" | README | suite re-run this session: 123/123 | **SELF-TESTED** (structural checks only — see Part K) |
| Security "42 attack vectors" / "34 attack vectors" | README / `security_ci_gate.py` header | static count in `penetration_test.py`: 66 `test("` + 43 `test_blocked("` call sites (many inside loops) — neither 42 nor 34 matches; executed count not pinned by any artifact | **NOT ESTABLISHED** (count discrepancy) |
| Battery 79/79 | internal | `.freebuff/p114rq_battery_results.tsv` | **SELF-TESTED** |

### F.2 Claims that CAN remain headline claims today

1. External degradation **0.435–0.595** (DEMONSTRATED, with provenance caveats).
2. IBM cross-dataset 0.873 same-generator-family (DEMONSTRATED + honest caveat).
3. Structural leakage suite and battery pass rates (SELF-TESTED, labeled as such).
4. Rules-carry-live-decisions / OOD ml=0.0 (DEMONSTRATED via pinned scenarios).
5. **0.966 / 91.8 / 0.877 / Brier / ECE must be downgraded to "in progress" or
   re-derived under the ledger** before they can stay headline (gap G-02).

### F.3 Other claim classes

- **Integrity/non-repudiation:** hash chain detects modification but cannot detect
  forged appends by a writer-capable service; HMAC keys are hardcoded (Part N) →
  claims of non-repudiation are **PARTIAL at best**.
- **Institutional/regulatory:** no compliance claim is supported; none should be made
  (Part O).
- `misc/docs/CLAIMS_MATRIX.md` (94 lines, S/P/M/A/C/R items with VERIFIED/…/FALSE)
  exists but repeats the train_compare misattribution (M5) — it must be reconciled
  with this audit in G-02.

---

## PART G/H — DECISION FRAMEWORK AND NEGATIVE-RESULT PLAN

Drafted in `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` — **NOT FINALIZED**;
sections requiring human decisions are marked. Summary: 7 pre-registered criteria
(ensemble helps / not, gate helps / not, features transfer / not, inconclusive), each
with metric, comparison, split, CI method, minimum-difference justification, and
consequence; negative-result conclusions **A / B / C**; no thresholds may be chosen
after viewing final test results.

---

## PART I — EVALUATION CONTENT PLAN

| Capability | Status |
|---|---|
| PR-AUC, recall@1%FPR, bootstrap CI | defined in `metric_definitions.py` v1.0 — **SELF-TESTED** machinery |
| Brier / ECE / reliability diagrams | **computed ad-hoc** in `calibration_and_stress.py`; **not** in metric definitions → add in G-02 |
| Multiple seeds | partial (seed 42 dominant; 5-fold only in registry) |
| Temporal splits | exists for ULB push reports; **not** the default for IBM v2 |
| Entity-disjoint splits | exists (`ibm_train.py` user-disjoint) but **not chronological** |
| Calibration separation / validation-only threshold | documented pattern exists (`threshold-at-FPR-on-validation`) but no enforcement artifact |
| Segment/subgroup analysis | `fairness_review.py` (tenure/device/time-pattern) exists — **test-only**; segments available on public data: amount bands, time periods, merchant/city category (IBM), entity — **geography beyond city not available; no invented segments** |
| Business metrics | **NOT IMPLEMENTED**: no recall-at-fixed-alert-volume, analyst burden, value-caught, or cost-based threshold anywhere |
| Threshold discipline | locked threshold exists for deployed model (0.018758); a frozen-threshold workflow artifact for future evals is **NOT IMPLEMENTED** |

---

## PART J — LABEL DELAY AND POINT-IN-TIME CORRECTNESS

- **Label delay / chargeback timing / investigation outcomes: NOT IMPLEMENTED.**
  No artifact models delayed confirmation; the feedback store ingests outcomes when
  they arrive but training/eval do not simulate label delay.
- **Point-in-time correctness:** feature derivation windows are anchored at event `ts`
  (`anchor = min(now, ts)` in the privacy layer) — *window* correctness is implemented
  and regression-tested (phase of the velocity-window fix). But:
  - `ibm_train.py` computes per-user running stats over each user's history — since the
    frame is sorted by (User, ts), the split is user-disjoint, **but any feature using
    full-history aggregates (e.g. `merchant_user_count`, `city_user_count` at lines
    180–188) is computed over the ENTIRE dataset including future rows** →
    **potential future-information leak, classification: UNKNOWN → must be tested (G-17)**.
  - Graph features (shared device/recipient, mule_ring_score) count over a window in
    production; whether eval-time computation respects the event-time boundary must be
    verified per feature.
- **Per-feature "what was available at scoring moment" inventory: NOT IMPLEMENTED** —
  this is a major future validation area for bank data (R-23).

---

## PART K — LEAKAGE AND STATISTICAL VALIDITY

| Check | Classification | Evidence |
|---|---|---|
| Train/test entity overlap (IBM v2) | **PASS** (for users) | user-disjoint split, `ibm_train.py` L246–250 |
| Chronological split | **FAIL/absent** | split explicitly NOT chronological (acknowledged in code comment) |
| Duplicate / near-duplicate transactions | **UNKNOWN** | no dedup audit artifact found |
| Future information in aggregates | **UNKNOWN** | see Part J (merchant/city user counts over full frame) |
| Target encoding | **NOT APPLICABLE/NOT FOUND** | no target encoding in feature code inspected |
| Time-based aggregates (production) | **PASS** | anchored velocity windows + both-bounds regression tests |
| Calibration leakage | **PARTIAL** | Platt fit on val per `calibration_and_stress.json`; not enforced by CI |
| Threshold leakage | **PARTIAL** | threshold-at-FPR-on-validation defined; README's 0.43/97.8% trace not tied to a validation-only selection record |
| Hyperparameter leakage / **repeated test-set tuning** | **UNKNOWN → risk HIGH** | ~20 `ulb_*` iterative push/optuna reports in `misc/reports/` suggest repeated tuning against the same ULB splits |
| Preprocessing leakage | **UNKNOWN** | no artifact |
| Feature-selection leakage | **UNKNOWN** | P20/45-feature selection history not auditable |
| Structural leakage suite | **PASS (self-tested)** | `leakage_structural_test.py` 123/123 re-run |

---

## PART L — GATING AND ROBUSTNESS

**Trace of `enforce_before_inference`:**

1. Eligibility inputs: runtime release state (READY), model/release identity,
   feature-contract checks.
2. `/evaluate` (main.py:853) and `/evaluate-batch` (main.py:1121) call it → on failure
   `BLOCK_INFERENCE` → **rules-only degraded decision** + `data_quality_blocked` audit
   event. Fail-open by design (a degraded evaluation never 500s).
3. Runtime state not READY → rules-only (`runtime_release_unverified`).
4. **`/internal/attribution` (main.py:1278) bypasses all of it** — direct
   `fusion.predict(features)`, only an internal-token check, no `check_access`
   (gap G-05).

**Experimentally:** grep for `ungated` matches only `model_deploy.py`. **There is no
UNGATED-vs-GATED experiment anywhere.** Implementation demonstrates *prevention of
inference*; it does NOT demonstrate improved outcomes (wrong-confident decisions,
coverage, selective risk). Classification: **NOT IMPLEMENTED** (R-18 plans the
7-condition controlled experiment with confidence intervals).

---

## PART M — SECOND DATASET

| Candidate | Provenance | Terms | Label semantics | Feature compat | Status |
|---|---|---|---|---|---|
| **NeurIPS 2022 BAF** | registry entry exists, `semantics_established: false` | **license NOT documented** | bank-transaction fraud (OK for fraud) | synthetic-ish bank data; entity `md5`/`customer` fields plausible for entity splits | **NOT ACQUIRED** — `data/external_benchmark/baf/` absent; phase107 cannot run; requires download approval + license review (R-20) |
| IEEE-CIS | known public | auth flow required | fraud labels ok | 3/48 native derivable | **BLOCKED** (authentication) |
| Kaggle fraud (fraudTrain) | recorded | public | fraudTrain labels | partially (47/48 reconstructed phase25) | already used; usable for replication |
| ULB | recorded | public | PCA — no entity IDs, no temporal semantics beyond Time | 0/21 mappable | in-domain only; not a "second dataset" for §16 claims |

No dataset is assumed suitable. Alternatives (e.g. FINTRAC-C, bank-internal data)
must pass the same provenance/semantics screen that phase23 applied (0/114 passed).

---

## PART N — ENGINEERING AUDIT

### Persistence
- **SQLite (5 stores + WAL)** is the real path; **PostgreSQL** exists only in
  `docker-compose.prod.yml` + one alembic migration → prod path **unexercised** (G-14).
- Restart persistence: demonstrated for DB stores; **observability is in-memory only**.
- TLS: `Caddyfile` supports auto-HTTPS + login rate-limit zone; **no TLS in dev compose**
  (G-15).

### Security
- Auth: identity service + sessions; admin progressive lockout (3→60s … 15+→3600s,
  `front_service/main.py:1260`); risk-engine rate limiting via Phase 73 `check_access`.
- **Secrets/key management: FAIL-quality.** `_MANIFEST_SECRET = sha256(b"PS14-release-manifest-v1-do-not-change")`
  hardcoded (`release_manifest.py:70`); `_TOKEN_SECRET` hardcoded (`promotion_gate.py:452`);
  export signing derives from `settings.export_signing_key` (env but sha256-derived).
- **HMAC → Ed25519:** docstrings in `settings.py:162`, `audit_service/export.py:6`, and
  `SECURITY_HARDENING_GUIDE.md:128` already recommend asymmetric/KMS signing;
  **no ed25519 code exists anywhere.** Must be evaluated (dependency/risk) before
  adoption — R-29, not assumed.
- CI security job runs bandit/pip-audit/security_scan/penetration/audit_hygiene —
  self-tested only; **no independent pen test has ever happened** (Part P).

### Audit path
- Async queue with single writer thread + `BEGIN IMMEDIATE`: ordering preserved
  within the writer; Phase 77 fixed duplicate-prev_hash races; 64/64 phase77 assertions.
- Residual risks: **writes are synchronous from the caller's perspective** — DB-4
  lock/outage fails fraud decisions; queue-full falls back to sync (backpressure
  OK, availability not); SQLite `IntegrityError` (not `OperationalError`) surfaces
  from trigger aborts; pre-fix duplicate-prev_hash rows remain in historical test DBs.
- Hash chain detects modification, **not forged appends** (any writer-capable service
  can append) → integrity = PASS-with-limits, non-repudiation = PARTIAL.

### Runtime
- Graceful shutdown: audit drain tested (phase77 shutdown-with-pending).
- Backpressure: queue maxsize + sync fallback — implemented.
- Concurrency: thread-safe observability; risk engine under load — **no load-test artifact** (G-16).
- Timeouts/retries: HTTP inter-service — not systematically audited this phase.
- Degradation: rules-only fail-open paths demonstrated by tests.

### Performance plan (PostgreSQL)
Latency (p50/p95/p99 of `/evaluate`), throughput, concurrent clients, CPU/mem,
degradation under DB-4 outage and queue saturation → R-27.

---

## PART O — BANK READINESS (16 deliverables — status today)

| # | Deliverable | Status | Requires |
|---|---|---|---|
| 1 | Field-level data-requirements spec | **MISSING** (fragmented across feature_contract.py) | team + bank data engineer |
| 2 | One-page collaboration proposal | **MISSING** | team |
| 3 | Privacy-preserving in-bank eval architecture | partial (phase88 design notes) → needs standalone doc | team + privacy officer |
| 4 | Exact permitted aggregate outputs | **MISSING** | privacy officer + legal |
| 5 | Publication terms incl. unfavourable results | **MISSING** | legal/counsel + institution |
| 6 | Pre-registered evaluation protocol | **DRAFT CREATED HERE** — not approved | reviewers |
| 7 | Business-relevant metrics | **NOT IMPLEMENTED** (Part I) | team + bank SMEs |
| 8 | Shadow-mode plan | **MISSING** | team + institution |
| 9 | Model-risk documentation | partial (model records/manifests) → needs MRM-style doc | team + risk officer |
| 10 | Analyst-facing explainability review | partial (attribution endpoint) → no analyst review | bank analysts |
| 11 | Human-in-the-loop verification process | partial (verification service) → no documented process | institution |
| 12 | Data retention/deletion requirements | **MISSING** | privacy officer |
| 13 | Access-control requirements | partial (tokens/roles) → undocumented | security reviewer |
| 14 | Pseudonymization requirements | partial (keyed device hashing, Fernet PII) → undocumented | privacy officer |
| 15 | Legal/privacy review | **NOT STARTED** | counsel/privacy officer |
| 16 | Bank compliance review | **NOT STARTED** | compliance team |

**No regulatory compliance is claimed or implied.**

---

## PART P — EXTERNAL ASSURANCE (6 items — NONE have happened)

| # | Item | Scope sketch | Artifacts expected | Status |
|---|---|---|---|---|
| 1 | Independent penetration test | all 6 services, auth/rate-limit/internal-token surfaces incl. `/internal/attribution` | report + severity ratings | **NOT STARTED** |
| 2 | Independent code review | risk_engine, privacy_layer, audit writer | findings report | NOT STARTED |
| 3 | Independent architecture review | service graph, DB-4 critical path, fail-open design | findings report | NOT STARTED |
| 4 | Independent methodology review | splits, tuning discipline, pre-registration adherence | findings report | NOT STARTED |
| 5 | Independent reproduction | fresh clone → ledger records for headline claims | reproduction log | NOT STARTED (blocked by G-01/G-02) |
| 6 | Independent statistical review | CIs, multiple comparisons, calibration claims | findings report | NOT STARTED |

Each needs: defined environment (containerized), read-only repo access, expected report
template, severity classification (critical/high/medium/low), remediation workflow.
No item may be claimed until a third party signs a report (R-40…R-43).

---

## PART Q — REPOSITORY CONSOLIDATION CANDIDATES (no deletion in Phase 104)

| # | Path / item | Reason | Callers | Tests affected | Risk | Migration plan |
|---|---|---|---|---|---|---|
| Q1 | `misc/` (719 tracked files; phase reports 1–25, push reports, backups) | archive sprawl; repeated-tuning artifacts mixed with evidence | none at runtime | none | low | move to `archive/` with an INDEX; keep evidence files referenced by claims |
| Q2 | 8 tracked `.pyc` under `misc/experiments/.../__pycache__` | hygiene | none | `audit_hygiene` currently passes despite them | low | untrack + gitignore |
| Q3 | Duplicate trail UIs (`verification_service/static/index.html` vs `audit_service/static/index.html`) | independent reimplementations of same trail | 2 services | verification/audit suites | medium | extract shared JS/CSS, keep both entry points |
| Q4 | `ALTMAN_NATIVE_FEATURES` defined in 2–3 places | contract duplication | phase86 | phase86 (77) | medium | single source module; re-export elsewhere |
| Q5 | `src/inference` (4.7k LOC) | unclear runtime callers | audit needed | unknown | medium | classify runtime vs test-only before any move |
| Q6 | Test-only monitoring phase scripts (large fraction of 42.5k LOC) | not reachable at runtime | battery (79 suites) | battery | low | relocate under `backend/scripts/phases/` with manifest |
| Q7 | `_from_ml_features` legacy 21→48 proxy | lossy compatibility layer | legacy callers | phase86 | medium | enumerate callers → migrate → deprecate (needs behavior review; NOT in this phase) |
| Q8 | README stale paths/claims + malformed table row | doc drift | humans | none | low | rewrite under R-45 |
| Q9 | `compare_ml_systems.py` hardcoded "MEASURED" SYSTEMS dict | documentation-as-evidence hazard | presentation tooling | none | low | feed from ledger records (depends G-01/G-02) |
| Q10 | Stale `altman_native_v2` schema string in model record | contract mismatch | release verify | phase suites reading records | low | correct in a future model-record regeneration with audit trail |
| Q11 | Root-level path claims in README (`scripts/deploy.sh`) | false paths | humans | none | low | fix in R-45 |
| Q12 | `misc/dist/ps14-scoring-pipeline` on disk, untracked | orphan build output | none | none | low | remove or document |

**This list becomes a genuine engineering phase (R-32), not just a document.**

---

## PART R — OUTPUTS THAT MUST NOT BE FORGOTTEN

1. **README one-screen rewrite** (R-45): what it is / what is demonstrated / what is
   not established / headline measured results (each with dataset+split+caveat) /
   major limitations / links to ledger evidence.
2. **Presentation** (R-46): `backend/scripts/build_presentation.py` exists (16 slides);
   Results slide currently has NO numbers, Limitations slide is qualitative → every
   number must carry dataset, split, caveat; the missing `docs/PS-14_fraud_detection.pptx`
   must be regenerated.
3. **Publication** (R-47): outline — methodology, experimental design, negative results,
   external degradation, gating hypothesis, provenance/evidence methodology, limitations,
   reproducibility package.

---

## PART S — DEPENDENCY-AWARE ROADMAP (GATES 1–6)

Phases are enumerated **R-01…R-48** (independent work is not merged; already-satisfied
work is not listed).

### GATE 1 — FOUNDATION (6 phases)
| ID | Phase | Why first |
|---|---|---|
| R-01 | Evidence-ledger expansion + CI claim-enforcement (CI rejects metrics without run records; battery covers eval_record/claim audits) | makes every later number trustworthy |
| R-02 | Headline-metric provenance reconciliation: trace or withdraw 0.966/91.8/0.877 and Brier 0.0009/ECE 0.0013; add Brier/ECE/reliability to `metric_definitions` v1.1; reconcile `CLAIMS_MATRIX.md` | stops unsupported headline claims |
| R-03 | Feature-contract decision execution (OPTION A vs B — human decision from Part C.4, then implement the chosen contract) | blocks Gate 2 design |
| R-04 | Reproduction of existing headline results into ledger records (ULB, IBM v2; fixed seeds, dataset hashes, commands) | independent reproduction precondition |
| R-05 | Pre-registration review & approval (human sign-off of the protocol doc) | mandatory before experiments |
| R-06 | Internal-endpoint security closure (`/internal/attribution` gating/`check_access` parity) | live security gap |

### GATE 2 — RESEARCH GO/NO-GO (11 phases)
| ID | Phase |
|---|---|
| R-07 | Independent benchmark harness (public-data eval pipeline writing full ledger records) |
| R-08 | Temporal-split evaluation |
| R-09 | Entity-disjoint evaluation |
| R-10 | Baselines & ablations suite |
| R-11 | Ensemble-contribution experiment (criteria 1/2) |
| R-12 | External calibration evaluation (Brier/ECE/reliability, calibration separated from threshold selection) |
| R-13 | Statistical leakage audit (duplicates, repeated test-set tuning, future-information aggregates, feature/preprocessing leakage) |
| R-14 | Subgroup/segment analysis |
| R-15 | Business-relevant metrics (recall@fixed alert volume, burden, cost proxies) |
| R-16 | Threshold-discipline enforcement (train→val→frozen→untouched test tooling) |
| R-17 | **Gate 2 GO/NO-GO report vs pre-registered criteria** (do not proceed as though the model succeeded) |

### GATE 3 — MECHANISM VALIDATION (7 phases)
| ID | Phase |
|---|---|
| R-18 | UNGATED-vs-GATED controlled experiments (7 input conditions; coverage/selective-risk/CIs) |
| R-19 | Drift & feature-corruption robustness validation of monitoring |
| R-20 | BAF acquisition & qualification (download approval + license/semantics review; currently BLOCKED) |
| R-21 | Second-dataset replication experiment (BAF or qualified alternative) |
| R-22 | Feature-shift / feature-availability analysis |
| R-23 | Label-delay & point-in-time correctness analysis (per-feature availability inventory) |
| R-24 | Feedback-loop bias analysis |

### GATE 4 — ENGINEERING (8 phases)
| ID | Phase |
|---|---|
| R-25 | PostgreSQL primary path + migrations beyond `001` |
| R-26 | TLS enforcement across environments |
| R-27 | Performance/concurrency/recovery load testing (PostgreSQL) |
| R-28 | Secrets & key management (remove hardcoded `_MANIFEST_SECRET`/`_TOKEN_SECRET`; env/KMS) |
| R-29 | Ed25519 asymmetric signing **evaluation** → adoption or documented rejection |
| R-30 | Observability persistence across restart |
| R-31 | Audit-path correctness: k-anonymity export wiring + async-audit guarantee re-verification |
| R-32 | Repository consolidation (execute Part Q list) |

### GATE 5 — EXTERNAL / INSTITUTIONAL (11 phases)
| ID | Phase |
|---|---|
| R-33 | Data-requirements spec + collaboration proposal (O1–2) |
| R-34 | In-bank evaluation architecture + permitted aggregate outputs (O3–4) |
| R-35 | Publication terms incl. unfavourable results + external pre-registration (O5–6) |
| R-36 | Business-metric definitions + shadow-mode plan (O7–8) |
| R-37 | Model-risk documentation + explainability & HITL review (O9–11) |
| R-38 | Data retention/access/pseudonymization requirements (O12–14) |
| R-39 | Legal/privacy/compliance review (O15–16 — outside counsel, privacy officer, compliance) |
| R-40 | Independent penetration test |
| R-41 | Independent code & architecture review |
| R-42 | Independent methodology & statistical review |
| R-43 | Independent reproduction |

### GATE 6 — FINAL FREEZE (5 phases)
| ID | Phase |
|---|---|
| R-44 | Final repository audit & freeze |
| R-45 | README one-screen rewrite (evidence-linked) |
| R-46 | Presentation update (measured numbers only) |
| R-47 | Paper/thesis draft |
| R-48 | Reproducibility package (fresh-clone → ledger replay) |

**Total: 6 + 11 + 7 + 8 + 11 + 5 = 48 remaining phases.**
Gate order is mandatory: Gate 2+ experiments must not run before R-05 (approved
pre-registration); Gate 4 work must not displace unresolved Gate 2 go/no-go;
Gate 5 items requiring outside parties can be *initiated* in parallel but are
credited only with signed artifacts.

---

## LIMITATIONS OF THIS AUDIT

- The 79-suite battery was **not re-run** after the environment restart (tally and log
  from 2026-09-28 were re-read and verified; individual key suites were re-run).
- Penetration vector counts are static greps; the executed count is not pinned by any artifact.
- `misc/` was sampled, not read file-by-file (719 files).
- Classification of README numbers relies on artifact search; an artifact containing
  the exact tuple could exist in an untracked location (data/ and models/ are gitignored —
  absence in git is not proof of absence on the author's machine at claim time, but the
  claim is still unevidenced *in the repository*).

---

## SUCCESS-CONDITION CHECK (15 questions)

1. **What exists today?** — Parts A/B/C/D inventories.
2. **What actually works?** — battery 79/79 (self-tested), six live services, deployed model with locked threshold, fail-open gating on `/evaluate`.
3. **What has been measured, 4. on which dataset/split?** — Part F table + Part D metric column.
5. **What is reproducible?** — machinery (eval_record 22/22) yes; headline numbers no (1 smoke ledger record).
6. **Unsupported claims?** — 0.966/91.8/0.877 tuple, Brier/ECE, k-anonymity gate, vector counts (F.1).
7–8. **Scientific/engineering risks?** — repeated-tuning/leakage UNKNOWNs, no untouched test set, no external eval under pre-registration; hardcoded HMAC secrets, attribution bypass, DB-4 on critical path, unexercised Postgres path.
9. **Public-data experiments?** — Gate 2 (R-07…R-17) + R-20/R-21.
10. **Bank data?** — 7 features (C.3), label-delay/point-in-time (R-23), OPTION B branch.
11. **External reviewers?** — R-40…R-43.
12. **Legal/compliance?** — R-39 (O15–16), plus O4/O5 publication terms.
13. **If external performance stays poor?** — conclusions A/B/C in the protocol doc; negative results preserved as evidence; no complexity added to force a positive result.
14. **What experiment decides continuation?** — R-11 (ensemble), R-18 (gate), R-21 (second dataset) judged against pre-registered criteria from R-05.
15. **How many phases remain?** — **48**, per Part S.

---

## PHASE 104 FINAL OUTPUT (Part T)

PHASE 104 STATUS:
PASS WITH LIMITATIONS

REMAINING PHASES:
48

*Limitations: the full 79-suite battery was not re-run after the environment restart
(tally/log of 2026-09-28 re-verified; key suites re-run individually: phase86 77/77,
eval_record 22/22, leakage_structural 123/123, audit_hygiene clean); penetration-test
vector counts are static greps; `misc/` (719 files) was sampled, not read
exhaustively. No model/dataset/code was modified and no phase was auto-started.*
