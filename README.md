# PS-14 — Privacy-First AI Fraud Detection System

## What it is

PS-14 is a privacy-first fraud detection prototype built on a core principle: *detect fraud without unnecessarily knowing who the person is. AI recommends — verification decides.*

The system is a 6-service microservice architecture: an Identity Service (pseudonymous IDs, encrypted PII), a Privacy Layer (feature derivation, PII stripping), a Risk Engine (ML fusion + declarative rules), a Verification Service (human-in-the-loop decisions), an Audit Service (append-only hash-chained trail), and a Front Page (landing + admin). It runs on SQLite (prototype) with a Docker/PostgreSQL/Caddy deployment configuration ready for production infrastructure.

## Current validation status

**Prototype: functionally demonstrated. Production promotion: BLOCKED.**

| Dimension | Status |
|---|---|
| In-domain benchmark performance | **Strong** — ROC-AUC 0.966 on ULB, 0.982 on IBM v2 (synthetic/training distributions) |
| External transfer performance | **Substantially weaker** — ROC-AUC 0.435–0.595 on unprovenanced external data |
| Real-world validation | **Not established** — all evaluations use synthetic or unprovenanced public datasets |
| Deployment infrastructure | **Demonstrated** — Docker, Caddy auto-HTTPS, PostgreSQL config, CI/CD pipeline exist |
| Production model promotion | **BLOCKED** — pending independently sourced, provenance-verified real-world fraud data |
| Independent penetration testing | **Not established** — 42 security scenarios are self-authored regression tests |
| Regulatory approval | **Not established** — no regulatory body has reviewed this system |

## Key Results

### In-domain benchmark

Performance on datasets from the same distribution as training data. These measure discrimination ability within a specific data distribution — not real-world generalization.

**Kaggle ULB Credit Card Fraud** (284,807 transactions, 492 fraud; PCA features, Dal Pozzolo et al.)

| Metric | Value |
|---|---|
| ROC-AUC | 0.966 |
| Recall@1%FPR | 91.8% |
| PR-AUC | 0.877 |
| Recall (threshold 0.43) | 97.8% (4 missed) |
| False Positive Rate | 1.996% |

**IBM v2** (1.2M rows, user-disjoint split)

| Model | ROC-AUC | PR-AUC | Recall@1%FPR |
|---|---|---|---|
| XGB | 0.982 | 0.406 | 0.836 |
| Fused | 0.978 | 0.314 | 0.840 |

> ⚠️ These metrics measure in-domain benchmark performance. They should not be interpreted as evidence of real-world generalization.

### External transfer

Evaluations on datasets not seen during training. Results show substantial degradation.

| Evaluation | Dataset | Model | ROC-AUC | FPR | Notes |
|---|---|---|---|---|---|
| Phase 23B | Kaggle fraudTrain (1.85M) | P20_45feat | 0.47 | — | 33/45 features unavailable |
| Phase 24 | Kaggle test (555K) | E_hardneg | 0.435 | 44.9% | 20/48 features reconstructed |
| Phase 25 | Kaggle test (reconstructed) | E_hardneg | 0.595 | 9.9% | 47/48 features; promotion BLOCKED |
| IBM v2 cross-dataset | IBM holdout (200K) | IBM v2 | 0.873 | — | Same generator family, not independent |

**Key finding:** In-domain ROC-AUC 0.966 drops to 0.435–0.595 on external data. The IBM v2 cross-dataset result (0.873) is from the same generator family — not a truly independent source.

### Security

Self-authored automated security regression suite covering 42 attack scenarios. **These are not independent penetration tests.** A professional pentest is recommended before handling real financial data.

| Test Suite | Checks | Result |
|---|---|---|
| `penetration_test.py` | 42 attack scenarios | Self-authored; not equivalent to independent pentest |
| `security_test.py` | 9 checks (CORS, rate-limiting, headers, key separation) | Hermetic tests pass; connection-dependent tests require live stack |
| `pseudonym_separation_test.py` | 14 checks (DB-2 compromise scenario) | 14/14 pass |
| `leakage_structural_test.py` | 123 checks (target, temporal, entity, scaler, distribution) | 123/123 pass |

### Federated learning

**This is a simulation.** No real financial institution participated. Three simulated institutions on account-disjoint shards of synthetic data, FedAvg over worker processes on a single machine.

| Metric | Value | Caveat |
|---|---|---|
| Federated vs local PR-AUC gain | +0.097 (size-skew) to +0.143 (clean dominant) | Synthetic data; 3 simulated institutions |
| DP cost at ε=8 | ~2/3 PR-AUC loss | Central noise only; no secure aggregation |
| All 43 FL tests | Pass | Algorithm mechanics verified |

## What is demonstrated

- End-to-end fraud scoring pipeline (6 services, register-to-audit flow)
- Privacy-preserving architecture (pseudonymous IDs, PII stripped at ingest, DB isolation)
- Stacked ML fusion (LR + RF + XGB + IF) with Platt-calibrated probabilities (Brier 0.0009, ECE 0.0013)
- Declarative rule engine with PR-reviewed rules and severity tuning
- Append-only, hash-chained audit trail with independent export verification
- Human-in-the-loop verification workflow with case management
- Federated learning algorithm (FedAvg, DP variant, heterogeneity map)
- Deployment infrastructure (Docker, Caddy auto-HTTPS, PostgreSQL config, CI/CD)
- 17-suite regression test suite (21/22 pass; `rules_gate` is a documented fragile test)
- Feedback loop (verification outcomes → labeled data → retraining)
- Drift monitoring (PSI-based, hash-chained alerts)
- k-anonymity gate on data exports

## What is not yet established

- **Production fraud-detection effectiveness** — all performance numbers are from synthetic or unprovenanced public datasets
- **Reliable real-world generalization** — external-transfer evaluations show substantial degradation (ROC-AUC 0.435–0.595)
- **Independent penetration testing** — the 42 security scenarios are self-authored; no professional pentest exists
- **Regulatory approval** — no regulatory body has reviewed or approved this system
- **Institutional deployment** — no financial institution uses or has validated this system
- **Dataset provenance** — 0 of 114 evaluated sources meet all provenance criteria
- **Leakage freedom** — the audit covers 10 specific checks within a defined scope; not a mathematical guarantee

## Claims & Evidence

Every claim in this repository is classified by its evidence type and status.

| # | Claim | Evidence | Evidence Type | Status | Limitation |
|---|---|---|---|---|---|
| 1 | End-to-end fraud scoring pipeline | 6-service architecture, live walkthrough, 50-case batch test | Implementation + automated test | **DEMONSTRATED** | Runs on synthetic data; no real-world transaction flow |
| 2 | ULB benchmark performance (ROC-AUC 0.966) | `train_compare.py`, time-split eval on 284K ULB rows | Experiment | **EXPERIMENTAL** | In-domain only; PCA features do not correspond to §16 production feature space |
| 3 | External dataset generalization | Phases 23B–25, `cross_dataset_eval.py` | Experiment | **NOT ESTABLISHED** | ROC-AUC degrades to 0.435–0.595 on external data; 0.873 on same-family IBM (not truly independent) |
| 4 | No data leakage within audited scope | Phase 25 10-check audit, `leakage_structural_test.py` (123/123) | Self-tested | **SELF-TESTED** | Covers target correlation, temporal ordering, entity contamination, scaler fitting, distribution shift; not a mathematical guarantee against all leakage forms |
| 5 | Privacy/identity separation | `pseudonym_separation_test.py` (14/14), `smoke_test.py` DB-separation checks | Self-tested | **SELF-TESTED** | Tested under specific DB-2 compromise scenarios; not an exhaustive privacy proof |
| 6 | Database isolation (per-store ownership) | `smoke_test.py` (DB-1 has no features; DB-2 has no PII), 5 physically separate SQLite stores | Self-tested | **SELF-TESTED** | Prototype uses SQLite; production would require per-store PostgreSQL with network segments |
| 7 | Security regression testing (42 scenarios) | `penetration_test.py` (42 scenarios), `security_test.py` (9 checks) | Self-tested | **SELF-TESTED** | Self-authored automated tests; not equivalent to independent penetration testing |
| 8 | Independent penetration testing | None | None | **NOT ESTABLISHED** | No independent penetration test exists in the repository; professional pentest recommended |
| 9 | Federated learning | `federated_sim.py`, `federated_test.py` (43/43), 4 reports | Simulation | **SIMULATED** | Simulated institutions on synthetic data, single machine; no real financial institution participated |
| 10 | Production deployment infrastructure | Docker, Caddy auto-HTTPS, PostgreSQL config, CI/CD, `deploy.sh` | Implementation | **DEMONSTRATED** | Infrastructure exists; model promotion to production is BLOCKED |
| 11 | Production model promotion | Phase 22–25 promotion gates, `promotion_guard.py` | Blocked | **BLOCKED** | Pending independently sourced, provenance-verified real-world fraud data |
| 12 | Model calibration (ml_score as calibrated probability) | PlattCalibration, `calibration_test.py` (16/16), Brier 0.0009, ECE 0.0013 | Self-tested | **SELF-TESTED** | Calibrated on synthetic data distribution; calibration on real data not validated |
| 13 | Real-world fraud detection effectiveness | None with provenance-verified data | None | **NOT ESTABLISHED** | All evaluations use synthetic or unprovenanced public datasets |
| 14 | Dataset provenance verification | Phase 23 114-source audit | Experiment | **NOT ESTABLISHED** | 0 of 114 sources meet all provenance criteria; best candidate (Kaggle) has unclear source |
| 15 | Throughput and scalability | `load_test.py` (~40–60 TPS, ~15–25ms p50) | Self-tested | **SELF-TESTED** | Single concurrent request, SQLite backend, no audit chain writes measured |
| 16 | 50-case scenario testing | `batch_cases.py`, invariant checks (determinism, monotonicity, audit chain) | Self-tested | **SELF-TESTED** | Synthetic hand-designed scenarios + seeded perturbations; does not establish detection performance |
| 17 | Cross-domain vs domain-specific training | 4-dataset ROC-AUC matrix, `meta_ensemble.py` | Experiment | **EXPERIMENTAL** | 4 public datasets; diagonal outperformed off-diagonal in this evaluation — not a universal theorem |
| 18 | CI/CD pipeline validation | `.github/workflows/ci-cd.yml`, `security-scan.yml` | Implementation | **DEMONSTRATED** | Pipeline runs on push to main; `rules_gate` is a documented fragile test (21/22 suites pass) |

### What this project does NOT currently claim

The repository does not currently establish:

- **Production fraud-detection effectiveness** — all performance numbers are from synthetic or unprovenanced public datasets under controlled conditions. No evaluation on provenance-verified real-world transaction data has been completed.
- **Institutional deployment** — no financial institution uses or has validated this system. The federated learning component is a local simulation with synthetic data.
- **Independent penetration-test certification** — the 42 security scenarios are self-authored automated regression tests. No professional penetration test or third-party security audit has been conducted.
- **Regulatory approval** — no regulatory body has reviewed or approved this system. The security hardening guide references DPDP Act and RBI guidelines as design considerations, not as achieved compliance.
- **Reliable real-world generalization** — external-transfer evaluations show substantial degradation (ROC-AUC 0.435–0.595 on unprovenanced data vs 0.966 in-domain). The cross-domain matrix demonstrates an observed limitation, not a universal property.
- **Universal superiority of domain-specific training** — the 4-dataset cross-domain experiment shows domain-specific models outperformed cross-domain transfer in this specific evaluation. This is an observed result on these datasets, not a proven theorem about fraud detection.
- **Data leakage freedom** — the leakage audit covers 10 specific checks within a defined scope. It does not guarantee the absence of all possible leakage forms.

## Architecture

### Services

Each service owns a physically separate SQLite store (`db/identity.db` = DB-1, `db/features.db` = DB-2) — no shared credentials, no cross-queries. A production deployment would swap the SQLite engines for per-store PostgreSQL with its own network segment and KMS keys.

```bash
# Front page (landing + status)   → http://127.0.0.1:8000
uvicorn src.front_service.main:app --port 8000

# Identity Service (user-facing)  → http://127.0.0.1:8001
uvicorn src.identity_service.main:app --port 8001

# Privacy Layer (internal only)   → http://127.0.0.1:8002
uvicorn src.privacy_layer.main:app --port 8002

# Risk Engine (internal only)     → http://127.0.0.1:8003
uvicorn src.risk_engine.main:app --port 8003

# Verification Service (user-facing UI) → http://127.0.0.1:8004
uvicorn src.verification_service.main:app --port 8004

# Audit Service (compliance role)  → http://127.0.0.1:8005
uvicorn src.audit_service.main:app --port 8005
```

### Data-store access matrix

Every store is owned by exactly one service; every read passes through that service's API with one of three credentials. A production deployment would replace the shared tokens with per-service mTLS + short-lived RBAC credentials.

| Store | DB file | Owned by | Holds | Passage (exact endpoint) | Credential | Who can use it |
|---|---|---|---|---|---|---|
| DB-1 | `db/identity.db` | Identity Service (:8001) | PII, auth credentials, `pseudonym_mapping` (fraud_id ↔ user) | `POST /auth/register` · `POST /auth/login` | none | anyone creating / owning an account |
| DB-1 | " | " | " | `GET /me/fraud-id` | Bearer JWT (15 min) | the account holder — own pseudonym only |
| DB-1 | " | " | " | `POST /internal/resolve-fraud-id` (break-glass, always logged) | `X-Internal-Token` | Identity Service / authorised ops — the **only** passage that links a pseudonym back to a person (returns masked PII) |
| DB-2 | `db/features.db` | Privacy Layer (:8002) | pseudonymous feature vectors, behavioural profiles, device fingerprints — no PII, no raw amounts | `POST /internal/ingest-transaction` · `POST /internal/commit-baseline` · `POST /internal/demo-age-account` | `X-Internal-Token` | service-to-service (Privacy, Verification) |
| DB-2 | " | " | " | `GET /internal/device-graph` (also proxied as `GET /compliance/device-graph` on :8004) | `X-Internal-Token` / `X-Compliance-Token` (proxy) | compliance role (via the proxy) |
| DB-3 | `db/risk.db` | Risk Engine (:8003); Verification Service shares it in the prototype (§13: `verification_outcomes` sits in DB-3) | `risk_scores` (pseudonymous) + `verification_outcomes` | `POST /internal/evaluate` · `POST /internal/attribution` | `X-Internal-Token` | Risk Engine; analysts (attribution, nothing persisted) |
| DB-3 | " | " | " | `GET /alerts` · `POST /alerts/{id}/confirm` · `GET /alerts/{id}/reason` · `GET /cases` · `GET /history` · `GET /overview` · `GET /chain-status` | Bearer JWT | the account holder — own rows only |
| DB-3 | " | " | " | `POST /demo/seed` | Bearer JWT | DEV ONLY — demo seeding |
| DB-4 | `db/audit.db` | Audit Service (:8005) | append-only, hash-chained `audit_events` (the whole decision trail) | `GET /audit/events` · `GET /audit/integrity` · `GET /audit/overview` · `GET /audit/export` | `X-Internal-Token` (+ `X-Audit-Actor` for reads) | compliance role (service-to-service); `/audit/export` is HMAC-signed so an external regulator can verify independently |
| DB-4 | " | " | " | `GET /compliance/events` · `GET /compliance/integrity` (and the :8004 `GET /compliance/*` proxies) | `X-Compliance-Token` (compliance passphrase) + `X-Analyst-ID` header | compliance-role viewer (browser, pseudonymous trail); each read logged with analyst identity |
| All | `db/*.db` | Front Service (:8000) | read-only SQL access to all five databases | `POST /admin/db-query` · `GET /admin/db-tables` | Admin JWT + admin passphrase (+ TOTP if enabled) | admin only; allowlist queries only (no arbitrary SQL), SAFE_COUNT_TABLES filter, 500-row cap, every query audit-logged |

**Three human passages, one pseudonym↔person passage:** the account holder reaches their own rows via **JWT**; the compliance role reaches the pseudonymous decision trail via the **compliance passphrase** (each analyst identified by `X-Analyst-ID` header, logged to audit chain); and only **break-glass** (`/internal/resolve-fraud-id`, internal token, always logged) can map a fraud_id to a person. Nothing else reads PII or raw amounts anywhere in the system.

### Endpoints

| Service | Endpoint | Notes |
|---|---|---|
| Identity | `POST /auth/register` | creates user + auth credential + `pseudonym_mapping` |
| Identity | `POST /auth/login` | returns JWT with `fraud_id` claim only (15 min) |
| Identity | `GET /me/fraud-id` | Bearer; caller's own ID; access logged |
| Identity | `POST /internal/resolve-fraud-id` | internal token; break-glass, always logged, returns masked PII |
| Privacy | `POST /internal/ingest-transaction` | internal token; returns §16 feature vector; stores derived features only — **does not** update the behavioral baseline; idempotent on event_id (replays return stored vector without re-auditing) |
| Privacy | `POST /internal/commit-baseline` | internal token; applies a stored vector to the profile (median, typical hours, device) — called ONLY for allowed/confirmed events |
| Risk | `POST /internal/evaluate` | internal token; fuses models + rules → score, band, decision, reason codes, `ml_score` (calibrated probability), `odds`, `degraded` (fail-open flag), `uncertainty` (model variance, disagreement, confidence level), `feature_version`, `rule_version`; enforces `velocity_limits` from the §16 vector first; idempotent on event_id |
| Risk | `POST /internal/attribution` | internal token; SHAP-style attribution — per-model + per-feature contributions + uncertainty (analyst only, nothing persisted) |
| Verify | `GET /alerts` | Bearer; high-risk alerts for the caller (unresolved) |
| Verify | `POST /alerts/{event_id}/confirm` | Bearer; `this_was_me` \| `this_wasnt_me` → outcome + case ID |
| Verify | `GET /alerts/{event_id}/reason` | Bearer; §11 category-level explanation |
| Verify | `POST /demo/seed` | Bearer; DEV ONLY — chains Privacy→Risk for demo alerts; commits allowed events to the baseline |
| Verify | `GET /cases` | Bearer; resolved verification cases, newest first, joined with risk score + reason codes/texts |
| Verify | `GET /feedback-status` | Bearer; feedback-pool counts + the retrain trigger threshold (§6) |
| Verify | `GET /overview` | Bearer; **BFF aggregate** — alerts + cases + feedback + chain status in one round trip; the UI renders everything from this. Served from a short-lived in-process cache (10s TTL), invalidated on every confirm/demo-seed write |
| Verify | `GET /compliance/events` | compliance token; pseudonymous decision trail (scores, reason codes, outcomes, case IDs), proxied from Audit |
| Verify | `GET /compliance/integrity` | compliance token; hash-chain integrity (ok, n_entries, first_bad_seq) |
| Verify | `GET /compliance/overview` | compliance token; **BFF aggregate** — integrity + latest case (joined with its score) + paged events + chain-wide aggregates in one call; the compliance view loads from this |
| Verify | `GET /investigator/cases` | compliance token; list investigator cases filtered by status/priority, sorted by priority descending |
| Verify | `POST /investigator/cases` | compliance token; create investigator case with auto-computed priority (score × confidence_weight) |
| Verify | `POST /investigator/cases/{case_id}/transition` | compliance token; transition case state (NEW→REVIEWING→USER_VERIFICATION→CONFIRMED_*/CLOSED), audit-logged |
| Audit | `GET /audit/events` | internal token (compliance role); pseudonymous trail, reads logged |
| Audit | `GET /audit/integrity` | internal token; recomputes the whole chain, reports first bad link |
| Audit | `GET /compliance/events` · `/compliance/integrity` · `/audit/overview` | compliance passphrase / internal token; same trail/integrity plus the BFF aggregate for the browser viewers |
| Audit | `GET /` | static compliance viewer UI (per-event hash linkage + integrity banner) |
| Front | `POST /admin/login` | public; admin passphrase (+ optional TOTP code) → JWT + essentials |
| Front | `POST /admin/db-query` | admin JWT + passphrase; read-only SQL against any of the five databases |
| Front | `GET /admin/db-tables` | admin JWT + passphrase; list tables + row counts for a given database |
| Front | `GET /admin/totp/setup` | admin JWT; generate a pending TOTP secret (Base32) |
| Front | `POST /admin/totp/verify` | admin JWT; verify TOTP code and enable 2FA |
| Front | `POST /admin/totp/disable` | admin JWT; disable 2FA (requires valid code) |
| Front | `GET /admin/totp/status` | admin JWT; check whether TOTP is enabled |
| Front | `POST /admin/rotate` | admin JWT; change the admin passphrase (revokes all sessions) |

## Validation phases

A rigorous, adversarial audit of the system's readiness. Every conclusion is evidence-classified; firewalls are verified.

> **Leakage audit limitation:** The leakage audit (Phase 25, 10 checks) establishes that the tested leakage classes were not observed under the audited pipeline. It is not a guarantee against all possible future leakage.

| Phase | Title | Classification | Key Finding |
|---|---|---|---|
| **14** | Chip supervision transfer | `NO_ROBUST_WIN` | Chip supervision **hurt**: C2 chip recall 32.3% vs C0 57.7% (−25.4pp) |
| **15** | Distribution-shift forensics | `CASE_B_PARTIALLY_REPRESENTED` | Channel composition shift PROVEN; 95% of 2017 patterns have weak/no historical analogue |
| **16** | Data-inventory boundary | `OUTCOME_C — NONE` | 51 datasets inventoried, 1 development-eligible (IBM, synthetic); boundary is protocol-constrained |
| **17** | IBM generator audit | `SYNTHETIC_REGIME_ARTIFACT` | Abrupt step change at 2017 affecting the whole population; real-world validity UNVERIFIED |
| **18** | System readiness gate | `CONDITIONALLY_DEPLOYABLE` | Security/monitoring/rollback PASS; 3 fraud-rate features with unverified label availability |
| **19** | Real-world data gate | `DATA_ACQUISITION_REQUIRED` | Every dataset in the environment is synthetic; no ground-truth labels; production prevalence unknown |
| **20** | Decision-time remediation | `CLEAN_PROD_COMPAT_CANDIDATE` | P20_45feat: 45 features, all decision-time valid, parity + causality PASS (leakage checked by Phase 25 audit) |
| **21** | Real-world validation gate | `INSUFFICIENT_REAL_WORLD_EVIDENCE` | No legitimate real-world transaction dataset available; promotion blocked pending data acquisition |
| **22** | Executable validation harness | `DATA_ACQUISITION_BLOCKED` | Complete CLI runner built (`phase22.run_real_world_validation`); READY but blocked by data absence |
| **23** | Data acquisition gate | `CONDITIONALLY_AVAILABLE` | 114 sources evaluated; best candidate Kaggle fraudTrain (1.85M rows); provenance unclear, no channel info, no label timing |
| **23B** | Frozen Kaggle external eval | `CONDITIONAL_EXTERNAL_EVALUATION_COMPLETE` | P20_45feat on Kaggle (1.85M rows, 0.58% fraud): ROC-AUC 0.47 (degraded features); E_hardneg unlabeled; production promotion BLOCKED |
| **24** | Conditional external eval | `CONDITIONAL_EXTERNAL_EVALUATION_COMPLETE` | E_hardneg on Kaggle (555K test rows): ROC-AUC 0.435, recall 26.8%, FPR 44.9%; 5 of 12 promotion gates failed; production promotion BLOCKED |
| **25** | Leakage audit & corrections | `NO_LEAKAGE_FOUND — within audit scope` | Fixed Phase 24's import-path + tz-comparison + index-order bugs; maximal causal reconstruction (47/48 features); no target/temporal/feature leakage detected by the 10-check audit suite; promotion BLOCKED |
| **38** | External validation readiness | `IMPLEMENTED` | Dataset intake contract, 9 eligibility gates, causality audit, frozen-model evaluation, 8-gate promotion check; READY, blocked pending an eligible dataset |
| **39** | Reproducible evaluation & statistical confidence | `IMPLEMENTED` | Immutable evaluation records (model/data hashes, git SHA, seeds); validation-only threshold discipline; bootstrap CIs withheld on small cells; single-source metric definitions; append-only ledger |
| **40** | Model & data drift monitoring + post-deployment safeguards | `IMPLEMENTED` | Schema drift, feature-availability drift, prediction-score drift, alert hysteresis with dedup/cooldown, label-awareness (no fabricate outcomes), baseline governance, promotion/retraining safeguards; 57/57 tests pass || **41** | Data quality, feature reliability & decision-time integrity | `IMPLEMENTED` | Feature contract (21 features, classification, validation rules), numerical robustness (NaN/Inf/clamp), missing-data policy (MISSING != ZERO != NOT_APPLICABLE), feature freshness, temporal leakage safeguards, data quality gates (OK/WARNING/BLOCKED); 59/59 tests pass || **42** | End-to-end decision-time enforcement audit | `PASS` | Runtime enforcement layer (`enforce_before_inference()`), adversarial regression tests (67/67 pass), invalid input BLOCKED, feature ordering enforced, NaN/Inf caught, missing != zero preserved, no mutation, <0.001ms overhead |
| **43** | Live inference enforcement integration | `PASS` | `enforce_before_inference()` wired into `/internal/evaluate` and `/internal/evaluate-batch`; BLOCK_INFERENCE skips fusion.predict() and uses rules-only; audit event `data_quality_blocked` appended; 35/35 integration tests pass |
| **44** | Decision integrity, version traceability & audit consistency | `PASS` | Canonical decision trace with deterministic hash, feature-vector hashing (ordering-agnostic), response/DB/audit consistency verification, version traceability (model/feature/rule), 37/37 tests pass |
| **45** | Transactional integrity, concurrency & failure-recovery | `PASS` | Audit chain tamper detection, DB-3/DB-4 reconciliation, concurrent idempotency (10 threads), trace-hash stability, 31/31 tests pass |
| **46** | Promotion gate & production readiness enforcement | `PASS` | Centralized promotion gate (8 gates), REAL_WORLD_VALIDATION hard block, artifact/feature binding, governance/security gates, ModelRegistry.promote() gated, cmd_deploy() gated; 58/58 adversarial tests pass |
| **47** | Bypass-proof activation & mandatory gate enforcement | `PASS` | promote(None) removed, PromotionToken (HMAC-signed receipt) required, artifact/feature binding verified, gate freshness (1h max), thread-safe promotion (lock), 52/52 adversarial tests pass |
| **48** | Signed release manifest & supply-chain integrity | `PASS` | ReleaseManifest (HMAC-signed, canonical JSON SHA-256), binds model+preprocessing+features+rules+evaluation, deterministic hashing, artifact/feature/binding verification, 79/79 adversarial tests pass |
| **49** | Runtime release attestation, deployment integrity & active-model drift | `PASS` | Pre-load release verification in the risk-engine lifespan (manifest hash/signature/artifact-set/component bindings/gate verdict), runtime attestation endpoint (`/internal/release-attestation`), fail-closed runtime states (STARTING/READY/MODEL_NOT_READY/INCONSISTENT/DRIFTED/FAILED), post-load drift surveillance in /health, inference gated on READY, DecisionTrace/audit bound to runtime release identity, blocked decisions carry the rules-only decision in the audit trail; 63/63 tests pass + live HTTP evidence |
| **50** | Legacy model attestation & zero-legacy runtime | `PASS` | Reconstructed ReleaseManifest from existing altman_native artifacts/model record; LEGACY_ATTESTED gate verdict (attestation != promotion); legacy model now runtime-attested (`release_attested: true`, `runtime_state: READY`); promotion remains BLOCKED; 65/65 adversarial tests pass + live HTTP evidence |
| **51** | Release lifecycle integrity | `PASS` | State machine (DISCOVERED → … → RUNTIME_ATTESTED), revocation, rollback verification, concurrency lock, drift detection; 55/55 tests pass |
| **52** | Forensic release history & operational governance | `PASS` | Hash-chained lifecycle transitions, incident records, forensic snapshots, gap detection, safe export; 59/59 tests pass |
| **53** | Real-world validation admission & evaluation framework | `PASS` | 7-gate dataset admission, source classification, label semantics, temporal validation, leakage detection, evaluation protocol binding, threshold policy, evaluation record integrity; 89/89 adversarial tests pass |
| **54** | Dataset discovery & provenance verification | `PASS` | Candidate registry, provenance verification (SOURCE_REPORTED/DOCUMENTED/INDEPENDENTLY_VERIFIED), source chain tracking, fraud label verification, independence evidence, admission packages, external discovery; 84/84 adversarial tests pass |
| **55** | Dataset acquisition & controlled evaluation | `PASS` | Deterministic acquisition records, SHA-256 artifact identity, checksum verification (VERIFIED/NOT_PROVIDED/MISMATCH), source evidence bundles, admission-gated evaluation packages, threshold policy enforcement; 70/70 adversarial tests pass |
| **56** | Dataset certification & eligibility evidence execution | `PASS` | 9 candidates investigated with evidence matrices, SOURCE_REPORTED blocks at source classification, all synthetic/derived/unknown blocked, label timing unknown blocks, evaluation NOT executed (no eligible dataset); 160/160 adversarial tests pass |


### Runtime Enforcement (Phases 42-43)

Phase 41's safeguards are enforced at the runtime inference boundary via `enforce_before_inference()` — integrated into the actual `/internal/evaluate` and `/internal/evaluate-batch` production endpoints (Phase 43).

**Enforcement verdicts:**
- `PROCEED` — all checks passed
- `PROCEED_WITH_WARNINGS` — non-critical issues (stale features, extra features)
- `BLOCK_INFERENCE` — critical issue (missing required, invalid values, wrong ordering)
- `FALLBACK_RULES` — ML unavailable, use rules-only

**What is enforced at runtime:**
- Feature ordering matches `ML_FEATURES` contract
- Required features present (missing with `reject` policy → BLOCK)
- Invalid values BLOCKED (out-of-range, wrong datatype, NaN/Inf)
- Numerical robustness applied (safe_float, clamping)
- Feature freshness checked (stale = warning)
- Original feature dict never mutated
- Enforcement overhead <0.001ms (sub-microsecond)

**BLOCK behavior (Phase 43):** When enforcement returns `BLOCK_INFERENCE`, the `/internal/evaluate` endpoint skips `fusion.predict()` entirely, uses rules-only scoring, tags the result `degraded=True`, appends a `data_quality_blocked` audit event to DB-4, and returns `DATA_QUALITY_BLOCKED` in reason codes.

**Idempotency:** The duplicate `event_id` path returns stored results without re-running enforcement. This is safe because stored results were computed with full validation at original ingest time.

**Adversarial test coverage (67 + 35 = 102 tests):**
- Valid transactions pass normally
- Missing/invalid/NaN/Inf inputs are caught
- Feature ordering violations detected
- Future timestamps produce warnings
- Rolling-feature temporal causality verified
- Label/outcome features cannot reach inference
- Fraud-rate features not in standard ML_FEATURES
- Fallback decisions are explicit and auditable
- Sensitive data not exposed in error messages
- Monitoring receives quality signals
- Performance overhead negligible (<0.001ms)

```
python backend/scripts/phase42_enforcement_test.py   # 67 tests
```

### Data Quality & Feature Integrity (Phase 41)

Decision-time feature validity is enforced through a multi-layer pipeline that runs before inference:

**Feature contract:** Each of the 21 ML features has a defined datatype, valid range, missing policy, and category (transaction-derived, historical/behavioral, device/channel, aggregated). Label-derived and future/post-event features are explicitly blocked from decision-time use.

**Numerical robustness:** NaN, +Infinity, -Infinity are caught and handled per feature policy (reject, use_neutral, use_imputed, fail_closed). Values outside valid ranges are clamped. `MISSING` is never silently conflated with `ZERO` — each feature documents its own missing-data behavior.

**Feature freshness:** Time-sensitive features (velocity counts, rolling aggregates) have explicit maximum-age requirements. Stale features are flagged before they can silently appear current.

**Temporal leakage safeguards:** Rolling/aggregate features are verified to use only events at or before the transaction time. Future events cannot contaminate historical feature values. Naive/aware datetime mismatches are caught.

**Data quality gates:** A unified `assess_data_quality()` function combines contract validation, numerical robustness, freshness, and temporal checks into a single OK/WARNING/BLOCKED status. Blocked vectors are rejected; warning vectors are flagged but allowed.

```
python backend/scripts/phase41_quality_test.py   # 59 tests
```

### Drift Monitoring & Post-Deployment Safeguards (Phase 40)

The system implements a comprehensive drift monitoring layer that detects feature-distribution drift, schema drift, feature-availability issues, and prediction-score drift. **Drift detection does NOT automatically authorize retraining or promotion.**

**Monitoring types:**
- **Feature drift** (PSI) — existing runtime detector (in-process per evaluate call) + offline windowed check
- **Schema drift** — missing/unexpected features, datatype mismatches
- **Feature-availability drift** — null rates, stale features, degradation detection
- **Prediction-score drift** — risk-score distribution shift, decision-band proportion changes
- **Label-awareness** — outcome monitoring only when verified labels exist; no fabricated metrics
- **Data-quality signals** — missingness, invalid values, range violations

**Alert lifecycle:**
- Consecutive-window hysteresis (single noisy window does NOT trigger alerts)
- Deduplication with cooldown (identical alerts suppressed for 5 minutes)
- State machine: NORMAL → WATCH → WARNING → CRITICAL → RECOVERING → NORMAL
- Storm protection: 40+ simultaneously drifting features aggregated into one systemic alert

**Safeguards:**
- `DRIFT DETECTED ≠ AUTOMATIC RETRAINING`
- `DRIFT DETECTED ≠ AUTOMATIC PROMOTION`
- Promotion requires all existing gates (leakage, data quality, security, etc.) to pass
- Critical drift blocks promotion even when all other gates pass
- Baseline governance: explicit metadata, source classification, approval tracking
- External/test data cannot silently become the production drift baseline

**Running Phase 40 tests:**
```
.venv/Scripts/python.exe backend/scripts/phase40_monitor_test.py   # 57 tests
.venv/Scripts/python.exe backend/scripts/drift_detector_test.py    # 31 tests (existing)
.venv/Scripts/python.exe backend/scripts/drift_test.py             # 22 tests (existing)
```

### Current Model Status

| Model | Features | Decision-Time Valid | Real-World Evidence | Status |
|---|---|---|---|---|
| **E_hardneg** (incumbent) | 48 | 3 fraud-rate features UNVERIFIED | None (synthetic only) | ACTIVE IN PROTOTYPE, UNTOUCHED |
| **P20_45feat** (candidate) | 45 | All features valid | None (synthetic only) | ELIGIBLE for real-world validation |

### Real-World Validation Status

```text
PHASE22_RUNNER = READY
PHASE23_ACQUISITION = CONDITIONALLY_AVAILABLE_WITH_CAVEATS
PHASE23B_EVAL = CONDITIONAL_EXTERNAL_EVALUATION_COMPLETE
PHASE24_EVAL = CONDITIONAL_EXTERNAL_EVALUATION_COMPLETE (E_hardneg: ROC-AUC 0.435)
PHASE25_AUDIT = NO_LEAKAGE_FOUND (10-check audit scope; post-reconstruction re-eval: ROC-AUC 0.595, recall 18.5%, FPR 9.9%)
REAL_WORLD_VALIDATION = BLOCKED (provenance unclear)
PROMOTION = BLOCKED
NEXT_ACTION = Continue authorized real-world data acquisition; wire Phase 25's
              reconstruction-diagnostics + leakage-audit outputs into promotion_guard.py's
              FEATURE_AVAILABILITY/FEATURE_PARITY gates (still Phase 24's hardcoded FAIL stubs)
```

**Honest interpretation:** E_hardneg is validated on the synthetic IBM corpus (recall 99.67%, FPR 10.99% on the untouched 2018–2020 window) and is the active model within the prototype. The 2017 regime shift is strongly supported as a generator artifact, so these numbers must not be read as real-world performance. Model promotion of P20_45feat is blocked pending real-world data — the next milestone is data acquisition, not more modeling.

Phase 23 evaluated **114 data sources** classified as:

| Category | Count | Description |
|---|---|---|
| Synthetic | 5 | IBM v2, PaySim, PS-14 derived, federated shards |
| Real (legacy) | 3 | ULB creditcard, Kaggle fraudTrain/Test |
| Unknown | 2 | Elliptic Bitcoin (not credit card) |
| Derived | 100+ | PS-14 feature vectors, scores, validation sets |
| **Real-world eligible** | **0** | None meet all criteria |

> **Dataset classification key:** A "public dataset" is freely available. A "real-world-origin dataset" may contain actual transactions but lacks provenance verification. A "provenance-verified dataset" has documented source, authorization, and label governance. An "independently sourced evaluation dataset" is obtained outside the training pipeline with verified chain of custody. "Production data" is live transaction data from a financial institution. The Kaggle fraudTrain is a public dataset with unclear provenance — it is NOT provenance-verified.

The best candidate (Kaggle fraudTrain) has documented limitations:
- **Provenance unclear** — may be real or simulated; source undisclosed
- **No channel information** — chip/swipe/online not available
- **No label timing** — when was fraud confirmed relative to transaction?
- **Label definition undisclosed** — what exactly does `is_fraud` mean?
- **PII present** — names, addresses, dates of birth in the dataset

Phase 24 evaluation is fully executable:
```bash
python experiments/phase24_conditional_external_eval/run_evaluation.py
python experiments/phase24_conditional_external_eval/run_evaluation.py --dry-run
```

Phase 25 corrected Phase 24's defects and closed the remaining audit gap: fixed the zip import-path bug (the runner now works from an extracted zip), rebuilt feature reconstruction maximally causally (47/48 features available; only `err` is unreconstructable — no error/decline code in Kaggle), and ran the leakage audit (10 checks covering target correlation, temporal ordering, entity contamination, scaler fitting, and distribution shift): no leakage detected within the audited scope; temporal check (now bug-fixed) finds 0 future timestamps. The cold-start audit verified the rate features are healthy in the shipped train-aggregate design (user 24.8% / merchant 1.3% / city 21.6% degenerate; median 1,054 tx/card history) — the widely-cited "76% degenerate" figure applies only to a running-rate design that was NOT evaluated. Metrics were unchanged (ROC-AUC 0.595, recall 18.5%, FPR 9.9%); promotion remains BLOCKED and real-world data acquisition remains the next milestone.

## Deployment infrastructure

> ⚠️ These are prototype deployment configurations, not production-approved deployments.

Three deployment options (see `HOSTING.md` for full guide):

```bash
# Option 1: Dev (SQLite, localhost)
docker compose up --build -d

# Option 2: Prototype production config (PostgreSQL, Caddy auto-HTTPS)
cp .env.example .env   # generate ALL secrets
docker compose -f docker-compose.prod.yml --env-file .env up --build -d

# Option 3: One-command deploy (validates + tests + builds + deploys)
bash scripts/deploy.sh              # full deploy
bash scripts/deploy.sh --dry-run    # validate only
bash scripts/deploy.sh --compose-prod  # use production compose
```

Prototype production infrastructure (not production-approved):
- **Auto-HTTPS** via Caddy with Let's Encrypt (`Caddyfile`)
- **Per-store PostgreSQL** with separate credentials (`docker-compose.prod.yml`)
- **Internal-only network** (services can't reach the internet)
- **Fail-closed startup** when `PS14_MODE=production`
- **Automated data retention** (`scripts/retention-cron.sh` + systemd timers)
- **Hourly audit chain verification** (`scripts/ps14-audit-chain.timer`)
- **CI/CD gates** (`.github/workflows/ci-cd.yml`) block deployment on failures

### CI/CD pipeline

Both workflows (`.github/workflows/ci-cd.yml`, `.github/workflows/security-scan.yml`) run on every push to `main`. The CI/CD pipeline runs five stages: Test Suite (regression checks on a clean checkout), Security Scan (bandit medium+ clean, dependency audit, penetration test), Docker Build (image builds, runs as non-root, `/health` answers), Integration Test (the live register-to-audit walkthrough boots all five services over real HTTP, then the compose stack health-checks and the audit chain verifies), and Deploy (tag-gated).

> **Note:** The `rules_gate` test is a documented fragile test — ML alone catches ~97% of synthetic fraud, so rule-removal cannot move recall. 21/22 suites pass consistently.

### Promotion gate (Phases 46-47)

A centralized promotion gate (`src/monitoring/promotion_gate.py`) prevents ineligible models from becoming active.  Every activation path — `ModelRegistry.promote()`, `model_deploy.py`, and manual promotion — MUST pass through `evaluate_promotion()` before the model becomes active.

**Phase 47 hardening:** `promote(None)` no longer works — gate_decision and a signed PromotionToken are both mandatory.  The token proves `evaluate_promotion()` was called and returned ELIGIBLE; the registry verifies the token's HMAC signature, model binding, artifact binding, feature-version binding, and freshness (max 1 hour).

**Phase 48 hardening:** A signed ReleaseManifest is now also mandatory.  The manifest is a canonical JSON record (SHA-256 hashed, HMAC-signed) that binds the exact artifact set, feature schema, preprocessing, rules, and evaluation evidence.  Activation rejects mismatched/tampered artifacts, wrong feature versions, or stale evaluation evidence.

**Phase 49 hardening (runtime attestation):** The invariant now extends to the running service:

    APPROVED RELEASE == DEPLOYED RELEASE == LOADED RELEASE == INFERENCE RELEASE

The risk engine's lifespan **verifies the release before loading it**: it discovers the ReleaseManifest (`models/production/release_manifest.json`), verifies the canonical manifest hash, the HMAC signature, the aggregate artifact-set hash against the exact bytes on disk, the component bindings (feature/schema/preprocessing/rule/evaluation), and the gate verdict — all BEFORE the model is deserialized.  If any check fails, the model is NOT loaded, no runtime attestation is published, health reports `model: error` / `runtime_state: FAILED`, and a `runtime_release_verification_failed` audit event is appended.  Normal ML inference is gated on `runtime_state == READY`; any other state (MODEL_NOT_READY / INCONSISTENT / DRIFTED / FAILED) fail-closes to rules-only degraded decisions tagged `RUNTIME_RELEASE_UNVERIFIED`.

**Phase 50 hardening:** The previously unattested legacy altman_native model is now runtime-attested via a reconstructed ReleaseManifest (`LEGACY_ATTESTED` gate verdict), built from the actual artifact bytes, altman_native/manifest.json, model record, and rules.yaml.  The live service now reports `release_attested: true` while promotion remains `BLOCKED_PENDING_ELIGIBLE_DATASET` — attestation and promotion are explicitly separate states.  Environment variables cannot attest a release; the manifest is the sole cryptographic identity source.

**Phase 51 hardening — release lifecycle integrity:** A full release lifecycle state machine (`src/monitoring/release_lifecycle.py`) enforces explicit states (DISCOVERED → MANIFEST_VERIFIED → GATE_VERIFIED → TOKEN_VERIFIED → REGISTERED → ACTIVATED → RUNTIME_ATTESTED) with forbidden transitions rejected.  Release revocation/quarantine is implemented: a REVOKED release cannot be activated or rolled back to.  Rollback requires the same verification guarantees as forward activation — a rollback target must have a valid manifest, verified artifact-set hash, matching component bindings, and a non-revoked state.  Concurrent activation is protected by a registry-level lock preventing inconsistent runtime identity.  Runtime drift detection re-checks artifact bytes on every health call, and post-load corruption is flagged within one request cycle.  Every lifecycle transition is audited with release_id, manifest hash, artifact hash, and reason.

**Phase 52 hardening — forensic release history and operational governance:** A forensic release history (`src/monitoring/forensic_release_history.py`) records every lifecycle transition with a cryptographic hash chain binding release_id, manifest_hash, artifact_set_hash, feature/schema versions, timestamps, and reason to the chain.  The chain detects any deletion, reordering, or field tampering.  An active-release timeline reconstructs exactly which release was active when, with deactivation timestamps and reasons.  Structured incident records (ARTIFACT_TAMPERING, MANIFEST_TAMPERING, SIGNATURE_FAILURE, HASH_MISMATCH, RUNTIME_DRIFT, etc.) persist through restarts with an explicit state machine (DETECTED → CONTAINED → INVESTIGATING → RESOLVED).  Gap detection identifies inconsistencies between registry, runtime, and history (e.g., registry says ACTIVE but no activation event exists).  Forensic snapshots preserve point-in-time state for security-critical failures.  Safe export provides auditor-reconstructable history without exposing HMAC keys, signing secrets, or credentials.

**Phase 53 — real-world validation admission & evaluation framework:** A dataset evidence model (`src/monitoring/dataset_evidence.py`) defines structured evidence records with 7-gate admission pipeline: source classification (SYNTHETIC/DERIVED/UNKNOWN → BLOCKED), provenance verification, label semantics (timing, generation method), temporal semantics (order, timestamps), schema integrity (dataset hash, feature schema), feature compatibility (missing/incompatible features), and minimum size.  Leakage detection covers target leakage, future information, duplicate contamination, and train/test contamination.  Temporal validation detects timestamp disorder.  Evaluation protocol binds dataset_hash, model_hash, feature_version, and prevents threshold-on-test leakage via a frozen threshold policy.  Evaluation record integrity uses SHA-256 tamper-evident records.  Current dataset audit classifies all 6 known candidates as ineligible (IBM Altman SDV = SYNTHETIC, ULB = UNKNOWN provenance, Kaggle DV = unclear source, PaySim = SYNTHETIC, Elliptic = different label semantics, PS-14 derived = NOT_INDEPENDENT).  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET.  89/89 adversarial tests pass.

**Phase 54 — dataset discovery & provenance verification workflow:** A candidate dataset registry (`src/monitoring/dataset_discovery.py`) tracks candidates through DISCOVERED → CANDIDATE → EVIDENCE_COLLECTION → PROVENANCE_VERIFIED → SEMANTICALLY_VERIFIED → LEAKAGE_CHECKED → FEATURE_COMPATIBLE → ELIGIBLE.  Provenance is classified as SOURCE_REPORTED, DOCUMENTED, or INDEPENDENTLY_VERIFIED — only the latter two pass the admission gate.  Source chain tracking records original source → release → distribution → local artifact with SHA-256 integrity.  Fraud label evidence requires: definition, granularity, author, timing, positive/negative class definitions, and verification status.  Independence evidence verifies: source independence, record independence, no shared identifiers, no synthetic generation.  Admission packages are cryptographically bound to the exact dataset with SHA-256 hashes of all evidence components.  External candidate discovery identifies 3 candidates (IEEE-CIS, PaySim, UCI Credit Card) — all have evidence gaps preventing eligibility.  Re-evaluation of all 6 known candidates confirms: 0 eligible, all blocked/ineligible.  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET.  84/84 adversarial tests pass.

**Phase 55 — dataset acquisition & controlled evaluation:** A deterministic acquisition record (`src/monitoring/dataset_acquisition.py`) captures artifact identity (SHA-256 hash, file size, filename), checksum verification (VERIFIED/NOT_PROVIDED/MISMATCH/UNVERIFIED), source evidence bundles (structured evidence items with content hashes and verification status), and acquisition state machine (NOT_ACQUIRED → ACQUIRED → IDENTITY_VERIFIED → CHECKSUM states → EXTRACTION_COMPLETE).  Controlled evaluation packages are created ONLY after successful admission — cryptographically bound to dataset hash, artifact-set hash, feature version, model identity, and admission package hash.  Threshold policy enforcement prevents test-set leakage (fit_on_test blocks).  Evaluation cannot occur before admission; no promotion bypass exists.  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET.  70/70 adversarial tests pass.

**Phase 56 — dataset certification & eligibility evidence execution:** A certification workflow (`src/monitoring/dataset_certification.py`) investigated 9 candidate datasets (IBM Altman SDV, ULB/MLG Credit Card, Kaggle Fraud Detection, PaySim, Elliptic, PS-14 derived, IEEE-CIS, Synthetic Financial SD, UCI Credit Card) with structured evidence matrices.  Source classification gate blocks: SYNTHETIC (3 candidates), DERIVED (1), SOURCE_REPORTED (4), UNKNOWN (1).  Label timing unknown blocks: ULB, Kaggle DV, IEEE-CIS, Elliptic, UCI.  No candidate satisfies all mandatory admission requirements.  External evaluation was NOT executed (NO_ELIGIBLE_DATASET).  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET.  160/160 adversarial tests pass.

**Phase 57 — evaluation harness & reproducible external benchmark:** A deterministic evaluation harness (`src/monitoring/external_evaluation.py`) provides the complete execution path for external validation.  Hard admission boundary enforces 16 mandatory checks (candidate, acquisition, artifact identity, dataset hash, certification, eligibility, certification hash, admission package hash, model identity, artifact-set hash, feature version, release attestation, protocol frozen, threshold frozen, no force flag, test-fixture isolation) — all must pass, no overrides exist.  Pre-evaluation snapshot freezes all identities (dataset, model, release, features, preprocessing, threshold) and any post-snapshot change invalidates the evaluation.  Frozen threshold policy blocks fit-on-test and unknown sources.  Deterministic exclusion policy (5 rules, versioned, hashed) is defined before evaluation.  Result integrity is tamper-evident (SHA-256 excludes timestamps for reproducibility).  TEST_FIXTURE path exercises complete evaluation mechanics without affecting REAL_WORLD_VALIDATION.  No eligible dataset exists — external evaluation NOT executed.  100/100 adversarial tests pass.

**Phase 58 — real-world dataset acquisition & eligibility execution:** A controlled acquisition-to-eligibility workflow (`src/monitoring/real_world_dataset_execution.py`) provides 11-state lifecycle (DISCOVERED → ELIGIBILITY_CERTIFIED) with 9 mandatory gates: provenance verification (SOURCE_REPORTED blocks, only DOCUMENTED/INDEPENDENTLY_VERIFIED pass), artifact identity (SHA-256 hash match), label semantics (timing known, generation method, granularity required), temporal integrity (timestamp field, collection period, ordering verified), independence verification (7 fields: PS-14 derived, synthetic generation, shared IDs, duplicated rows, feature overlap, source lineage, augmented copy — all must be VERIFIED_INDEPENDENT), feature compatibility (no MISSING/UNSUPPORTED features), and leakage checks (target, future info, duplicates, train/eval contamination).  Eligibility certificate is tamper-evident (SHA-256, binds dataset/acquisition/provenance/label/temporal/independence/feature/leakage hashes to model identity).  Forensic event chain with hash-linking records every transition.  9 known candidates investigated — no candidate passes all gates.  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET.  TEST_FIXTURE path exercises complete mechanics.  130/130 adversarial tests pass.

**Phase 59 -- eligible dataset evaluation & real-world validation execution:** The final controlled path (`src/monitoring/real_world_validation_execution.py`) ties Phases 53-58 into a single deterministic chain: dataset selection -> certificate verification -> evaluation snapshot -> frozen preprocessing -> inference -> metrics -> tamper-evident result -> forensic record.  22 mandatory selection conditions (candidate exists, acquisition exists, artifact identity verified, dataset hash verified, provenance verified, label semantics verified, temporal semantics verified, independence verified, feature compatibility verified, leakage checks pass, eligibility certificate exists and valid, not TEST_FIXTURE, not synthetic, not derived, not UNKNOWN, not SOURCE_REPORTED-only, release attestation valid, model identity matches, artifact_set_hash matches, feature_version matches, evaluation protocol frozen, threshold frozen) -- all must pass, no override, no force flag.  Certificate verification recomputes and validates all evidence hashes (dataset, acquisition, provenance, label, temporal, independence, feature mapping, leakage) against current state and model identity.  Pre-evaluation snapshot freezes all identities and any post-snapshot drift blocks evaluation.  Frozen preprocessing blocks fit-on-test and records exclusion reasons.  Deterministic metrics (TP, TN, FP, FN, precision, recall, FPR, specificity, accuracy, F1, prevalence, ROC-AUC, PR-AUC) computed from confusion matrix.  Result hash is tamper-evident (SHA-256 excludes volatile timestamps for reproducibility).  Complete forensic event chain with hash-linking records every lifecycle event.  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET -- no genuinely eligible dataset exists.  TEST_FIXTURE path exercises complete mechanics without affecting real-world status.  Promotion safety: evaluation completion does NOT automatically grant promotion eligibility.  272/272 adversarial tests pass.

**Phase 60 -- reproducible dataset evidence ingestion & audit package:** An evidence-ingestion layer (`src/monitoring/dataset_evidence_ingestion.py`) provides deterministic evidence packaging, manifest generation, export/import, change detection, and certification binding.  8-state workflow (EVIDENCE_PACKAGE_CREATED -> READY_FOR_CERTIFICATION) with failure states BLOCKED, INVALID, INCOMPLETE.  Evidence package captures source identity, artifact identity, and 9 evidence categories (PROVENANCE, ARTIFACT, LABEL_SEMANTICS, TEMPORAL, INDEPENDENCE, FEATURE_COMPATIBILITY, LEAKAGE, LICENSE, SOURCE_IDENTITY).  Each category is independently assessed as PASS, FAIL, MISSING, or NOT_APPLICABLE -- MISSING never silently becomes PASS.  Evidence manifest hash is deterministic (SHA-256 over canonical sorted manifest).  Export/import round-trip preserves integrity: import captures a content hash, validate recomputes it, tamper detection catches post-import modifications.  Evidence diff detects changes across source, publisher, license, artifact hash, evidence types, category gates, and manifest hash.  Certification binding verifies manifest hash matches at validation time.  Forensic event chain records every lifecycle transition with hash-linking.  TEST_FIXTURE exercises complete mechanics without affecting real-world eligibility.  Evidence ingestion does NOT itself establish real-world eligibility.  228/228 adversarial tests pass.

**Phase 61 -- actual real-world fraud dataset acquisition & validation:** Investigated the ULB/MLG Credit Card Fraud dataset (284,807 rows, 31 columns, SHA-256: 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89, already present in `data/creditcard.csv`).  Ran through Phase 53 admission gates: BLOCKED because source classification is UNKNOWN (cannot independently verify ULB/Kaggle provenance without direct access to original source documentation).  Provenance evidence: SOURCE_REPORTED only.  Independence evidence: SOURCE_REPORTED only.  Source identity evidence: SOURCE_REPORTED only.  Label semantics: DOCUMENTED (binary fraud label, post-investigation timing).  Temporal semantics: DOCUMENTED (Time column verified, 2-day collection period).  Feature compatibility: DOCUMENTED (PCA mapping exists, but V1-V28 are anonymized).  Leakage: DOCUMENTED (no target-derived columns).  License: DOCUMENTED (CC BY 4.0).  Evidence package created with 9 items, 3 blocking categories missing independent verification.  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET -- the system correctly refuses to admit a dataset whose provenance cannot be independently verified.  This is the expected and correct outcome.

**Phase 62 -- real-world fraud dataset acquisition & evaluation execution:** Actually ran the ULB Credit Card Fraud dataset through the complete Phase 53-60 pipeline with honest evidence.  Dataset: 284,807 rows, 31 columns (V1-V28 PCA + Time + Amount + Class), 492 fraud (0.17%), SHA-256: 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89.  Source evidence: Zenodo DOI: 10.5281/zenodo.7395559, OpenML ID: 1597, IEEE CIDM 2015 paper (Dal Pozzolo et al.), ULB MLG + Worldline.  Phase 53 admission: 6/7 gates PASS, INELIGIBLE because feature_compatibility requires a feature_mapping_version.  Phase 58 execution: BLOCKED (no valid feature mappings from ULB PCA components to PS-14 domain features).  Phase 59 evaluation: BLOCKED by selection condition C09 (feature compatibility).  Phase 60 evidence package: CREATED with 7 evidence items documenting the incompatibility.  KEY FINDING: The ULB dataset's V1-V28 PCA-transformed components have no semantic mapping to PS-14's 21 domain features (amount_ratio, txn_freq_last_24h, device/location/recipient features, etc.).  The original feature semantics are destroyed by PCA transformation.  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET -- correctly blocked by feature incompatibility.  This demonstrates the system honestly blocks datasets that are real-world in origin but incompatible with the model's feature contract.

**Phase 63 -- complete real-world fraud validation attempt:** Investigated additional dataset candidates beyond the ULB dataset.  Candidates examined: (1) ULB Credit Card Fraud — already acquired, BLOCKED at feature compatibility (V1-V28 PCA components vs PS-14 domain features); (2) Zenodo production banking dataset (DOI: 10.5281/zenodo.20030064, 56,962 rows) — BLOCKED because README explicitly states "These are not real customer banking transactions from a licensed financial institution" (public test API submissions, not genuine banking data); (3) AIZBank (Zenodo DOI: 10.5281/zenodo.14636312) — BLOCKED as DERIVED (curated from multiple existing datasets); (4) Kaggle bank transaction dataset (valakhorasani) — BLOCKED because fraud labels are derived from Isolation Forest anomaly detection, not genuine investigation labels; (5) IEEE-CIS Fraud Detection (Vesta Corporation) — BLOCKED because download requires Kaggle authentication (no credentials available in this environment).  Conclusion: No genuinely eligible real-world fraud dataset exists that satisfies ALL of PS-14's mandatory gates simultaneously — the model's 21 domain features (amount_ratio, txn_freq_last_24h, device/location/recipient/account metadata) require transaction-level metadata that no publicly available dataset provides in a compatible format.  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET.  All existing Phase 53-60 tests pass.  This is the correct and expected outcome.

**Phase 64 -- real-world dataset feature compatibility:** Definitive analysis of the PS-14 21-feature contract against all investigated real-world fraud datasets.  The 21 features require 18 unique raw data sources including: transaction_amount, account_payment_history, transaction_history, timestamps, device_id, account_device_history, geolocation, account_location_history, recipient_id, account_recipient_history, auth_event_log, transaction_categories, amounts, account_first_transaction_ts, device_account_graph, recipient_account_graph, account_graph.  Feature categories: 5 TRANSACTION_DERIVED, 9 HISTORICAL_BEHAVIORAL, 3 DEVICE_CHANNEL, 4 AGGREGATED.  5 candidates investigated through actual Phase 53 admission gates: (1) Kaggle fraud (kartik2112) — BLOCKED as SYNTHETIC (Amazon Science FDB documents Sparkov data generator; 1,000 simulated customers, 800 merchants, 6 months); would be compatible for 11/21 features if real; (2) ULB/MLG Credit Card — INELIGIBLE at feature compatibility (0/21 features mappable; PCA transformation destroys all original feature semantics); (3) IEEE-CIS (Vesta Corporation) — ACQUISITION_BLOCKED (requires Kaggle authentication; estimated 13/21 features, graph features still missing); (4) IBM Altman v2 — BLOCKED as SYNTHETIC; (5) PaySim — BLOCKED as SYNTHETIC.  7 of 21 features cannot be provided by ANY investigated dataset: unusual_location_flag, unusual_recipient_flag, failed_auth_count_24h, shared_device_accounts, shared_recipient_accounts, mule_ring_score, recipient_novelty.  These features require device-location-recipient graph cross-referencing that no publicly accessible fraud dataset provides.  CONCLUSION: The blocker is ARCHITECTURAL/DATA-CONTRACT — the 21-feature PS-14 model requires institutional-grade transaction metadata (device IDs, location baselines, recipient graphs, auth logs) that publicly available fraud datasets do not provide.  The ULB dataset's PCA transformation destroys all semantic information.  The only potentially compatible candidate (IEEE-CIS) requires authentication.  The validation infrastructure (Phases 53-63) is working correctly by blocking ineligible datasets.  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET.  5/5 compatibility assertions pass.

**Phase 65 -- external validation surrogate model (RESEARCH ONLY):** Research-track evaluation answering "Can PS-14 concepts be externally benchmarked?"  Uses a separate `research/` namespace (never mixes with production).  Defines a public feature contract (`public_v1`, 14 features: amount, hour, weekend, amount_zscore, amount_ratio, cumulative count, 4 PCA components, category rate, merchant rate, distance, auth proxy) and documents 7 production features impossible from public data (device/location/recipient graph features).  Trains threshold and linear models on ULB (284K rows, 8 features: amount stats + PCA) and Kaggle/Sparkov (1.85M rows, 9 features: amount + time + category + distance).  All results cryptographically bound (SHA-256, timestamps excluded for determinism).  Production model/release/gates/REAL_WORLD_VALIDATION untouched.  Research models cannot be promoted without full governance.  Key findings: (1) external benchmarking IS possible with reduced features; (2) ULB provides 8/14 public features, Kaggle provides 9/14; (3) 7 features (device_id, geolocation, recipient graphs, auth logs) are impossible from public data; (4) production external validation requires institutional-grade telemetry not available publicly.  13/13 assertions pass.

**Phase 66 -- production reproducibility and disaster recovery audit:** Comprehensive audit of the production model's reconstruction, verification, restoration, and operational capabilities.  CRITICAL FINDING: `models/artifacts/` directory is EMPTY — no model files, no manifest, no metadata exist on disk.  Production risk engine requires retraining to recover.  Audit verified: (1) all 6 core code modules exist and load correctly; (2) manifest HMAC sign/verify works — tampered manifests are detected (forged signature rejected); (3) artifact_set_hash works correctly on empty and populated directories; (4) risk engine handles missing artifacts gracefully (FAILED state, rules-only fallback); (5) release lifecycle state machine works — valid transitions succeed, invalid transitions blocked, INVALID->DISCOVERED recovery path exists; (6) registry persists releases across restarts; (7) duplicate release IDs are rejected; (8) forensic transition chain integrity verified; (9) forensic incident creation works; (10) ModelRegistry rollback tracking exists; (11) 10 failure scenarios documented in failure matrix.  Recovery procedure: retrain (scripts/ibm_train.py), verify artifacts, start risk engine, verify attestation.  Known limitations: model artifacts NOT in repo, database files NOT in repo, forensic history lost on db/ deletion, no automated disaster recovery.  REAL_WORLD_VALIDATION unchanged (BLOCKED_PENDING_ELIGIBLE_DATASET).  41/41 assertions pass.

**Phase 67 -- rebuild, register, and re-attest production model:** Production model artifacts already existed at `models/production/altman_native/` (xgb_native 1MB, lgb_native 637KB, cb_native 278KB, scaler_native 683B).  Model version: `altman_native_E_hardneg_cert_20260904` (48 features, XGB+LGB+CB ensemble, locked threshold 0.018758).  All 5 artifact hashes verified against manifest.  AltmanNativeEnsembleEngine loads and produces valid inference scores.  Release manifest created with HMAC signature (verified).  Full lifecycle completed: DISCOVERED -> MANIFEST_VERIFIED -> GATE_VERIFIED -> TOKEN_VERIFIED -> REGISTERED -> ACTIVATED -> RUNTIME_ATTESTED.  Tamper detection works: modified artifact hashes detected.  Registry persists across restarts (state survives reload).  Forensic transition chain integrity verified.  Reproducibility record created.  The model is REBUILT and REGISTERED but NOT PROMOTED — LEGACY_ATTESTED gate verdict reflects pre-existing model, not new promotion.  Synthetic training does NOT constitute real-world validation.  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET.  31/31 assertions pass.

**Phase 68 -- end-to-end production risk evaluation:** Verified the complete transaction flow through the actual PS-14 production architecture.  Request flow: X-Internal-Token authentication -> Pydantic schema validation -> idempotency check -> drift detection -> runtime state check -> feature enforcement (contract, robustness, freshness) -> AltmanNativeEnsemble model inference -> RulesEngine evaluation -> band-of decision -> DB-3 RiskScore persistence -> DB-4 audit event.  All 43 integration assertions pass: valid transaction produces ML score 0.000642, rules score 0.2, band low, decision allow; invalid inputs correctly rejected (missing field, short event_id, invalid fraud_id, negative amount_ratio); determinism verified (identical inputs produce identical score); DecisionTrace structure verified (canonical features, feature hash); velocity limits evaluated (normal passes, high triggers); privacy verified (no PII in FeatureVector); model/release binding confirmed (altman_native, altman_native_v1, release-altman_native_E_hardneg_cert_20260904); failure behavior correct (missing feature blocks, out-of-range detected, empty features blocked); feature contract binding verified (21 core features all in ML_FEATURE_CONTRACT).  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET.  Test transactions are controlled fixtures, NOT real-world evidence.  43/43 assertions pass.

**Phase 69 -- production load, concurrency and reliability test:** Stress-tested the PS-14 production risk path using controlled test fixtures (1,820+ evaluations across all tests).  Results: baseline ML latency 1.4ms avg, p50=1.3ms, p95=1.6ms, p99=1.9ms; throughput 649 req/s sequential, 590 req/s at concurrency 25; zero errors across all concurrency levels (1/5/10/25 threads); deterministic: ML score, risk score, and decision identical across 50 concurrent threads on same input; idempotency: exactly 1 creator in 20-thread race on same event_id, 19 replays; 500-request large workload: zero errors, all scores in [0,1], all decisions valid; resource stability: 1,000 requests with zero degradation (1.03x ratio); velocity limits: normal passes, high-freq and high-spend correctly trigger hard limits; feature enforcement: valid features proceed, missing features blocked, out-of-range blocked/warned; band boundaries: monotonic across 13 test points; audit chain: 100-event hash chain integrity verified, tamper detection confirmed; security: schema rejects invalid types, missing fields, invalid fraud_id/event_id; concurrent mixed risk: all 60 mixed-level requests complete with scores in range.  REAL_WORLD_VALIDATION remains BLOCKED_PENDING_ELIGIBLE_DATASET.  82/82 assertions pass.


After load, `/health` re-checks the artifact set on every call (post-load drift surveillance): a file replaced after startup flips the state to `DRIFTED`, drops the model, and reports degraded — a file modification never becomes a new active model.  The read-only `/internal/release-attestation` endpoint (internal-token protected) exposes the verified runtime identity (release_id, model_version, artifact_hash, manifest_hash, feature/schema/preprocessing/rule hashes, loaded_at) with no secrets.  Decisions carry `runtime_state` / `runtime_release_id` / `runtime_manifest_hash` / `runtime_attestation_hash` in the audit payload, and blocked/degraded decisions (data-quality or runtime-integrity failures) record the resulting rules-only decision in the audit trail so every case stays traceable.

**Limitations:** runtime identity rests on verified artifact bytes + controlled loading (joblib objects have no stable post-load cryptographic identity); single-process per store is assumed (multi-worker deployments must share the same release directory and would each attest independently); environment variables can NEVER attest a release — `MODEL_VERSION=x` in the environment is not evidence.

**Required gates (all must PASS):**

| Gate | Description | Current status |
|---|---|---|
| REAL_WORLD_VALIDATION | Independently sourced, provenance-verified real-world fraud data | **BLOCKED** — no eligible dataset acquired |
| MODEL_ARTIFACT_BINDING | Candidate artifact matches evaluated artifact | Verified per-deployment |
| FEATURE_SCHEMA_BINDING | Feature/schema version matches between eval and promotion | Verified per-deployment |
| MODEL_GOVERNANCE | Leakage, data quality, performance, robustness, security, approval gates | Verified per-model record |
| SECURITY_CI | Security scan + penetration test + regression suite | Self-authored; not independent audit |
| DRIFT_STATUS | No critical drift detected | Non-blocking warning |
| EXTERNAL_DATASET_ELIGIBILITY | External dataset passed eligibility gates | Checked when applicable |
| ROLLBACK_AVAILABILITY | Rollback source exists | Non-blocking |

**Promotion remains BLOCKED** because REAL_WORLD_VALIDATION has not been satisfied.  The gate is machine-readable and cannot be bypassed by documentation alone.

## Limitations

- **Synthetic data limitations** — all training and in-domain evaluation uses synthetic data (IBM generator, PaySim, PS-14 derived). The 2017 regime shift is classified as a synthetic artifact. Absolute performance numbers (ROC-AUC 0.966, recall 99.67%) are measured on synthetic distributions and must not be extrapolated to real-world transaction streams.
- **External distribution shift** — models trained on synthetic data show substantial degradation on external datasets (ROC-AUC 0.435–0.595 on unprovenanced data). This is expected when training and test distributions differ, but it means the system has not demonstrated generalization.
- **Provenance limitations** — 0 of 114 evaluated data sources meet all provenance criteria. The best candidate (Kaggle fraudTrain) has unclear source, no channel information, no label timing, and undisclosed label definition. Any evaluation on this data carries fundamental credibility limitations.
- **Lack of institutional validation** — no financial institution has used, tested, or validated this system. The federated learning component is a local simulation with synthetic data.
- **Lack of independent penetration testing** — the 42 security scenarios are self-authored automated regression tests. No professional penetration test or third-party security audit has been conducted. A pentest is recommended before handling real financial data.
- **Prototype status** — the system runs on SQLite with shared internal tokens. Production would require per-store PostgreSQL, per-service mTLS, KMS envelope encryption, and network segmentation. The deployment infrastructure exists but has not been exercised in a production environment.
- **Leakage audit scope** — the 10-check leakage audit covers target correlation, temporal ordering, entity contamination, scaler fitting, and distribution shift within the audited pipeline. It does not guarantee the absence of all possible leakage forms.
- **Model calibration scope** — Platt calibration (Brier 0.0009, ECE 0.0013) was fit and validated on synthetic data distribution. Calibration on real-world data has not been attempted.

## Reproducible Evaluation

Every ML evaluation run produces an **immutable evaluation record** that binds the reported metrics to exactly what produced them:

- **Model identity** — SHA-256 over each artifact file (`*.joblib`) plus an aggregate `model_hash`. Any byte change in any artifact changes the evaluation identity.
- **Dataset identity** — SHA-256 fingerprint over the dataset file's raw bytes (what is hashed: exactly the bytes on disk — no header reinterpretation or normalization).
- **Configuration** — threshold, `threshold_source` (`validation` | `fixed`), seed, split policy, metric-definitions version.
- **Environment** — git commit SHA, Python/NumPy/pandas/sklearn/xgboost versions.
- **Timestamp and evaluation ID** — deterministic ID derived from timestamp + model hash + dataset hash + record content.

Records are stored **append-only** in `reports/evaluation_runs/eval_ledger.jsonl` (one JSON line per run). Re-running an evaluation never overwrites the previous record — failed or unfavorable evaluations are preserved exactly like favorable ones.

**Test/external-data protection:** the operating threshold is selected on validation only, or fixed a priori in the config. `backend/scripts/evaluate.py` refuses runs whose `threshold_source` would tune on the evaluation data itself (`test` / `external` / `holdout` are rejected by design).

### Reproducing an evaluation

```bash
# Frozen-model evaluation with full provenance record (Phase 39):
python backend/scripts/evaluate.py \
    --model-dir models/artifacts \
    --dataset data/transactions.csv \
    --config data/eval_config.json \
    --model-identifier ps14_fused_ensemble \
    --training-data data/transactions.csv

# → reports/evaluation_runs/eval_ledger.jsonl (append-only)
# → reports/evaluation_runs/record_<eval-id>.json (machine-readable record)
# → reports/evaluation_runs/report_<eval-id>.md (human-readable summary)
```

`train_compare.py` also appends a record automatically on every training run (threshold source recorded as `validation` — thresholds are selected on the validation split, never on test).

### Metric definitions

All metrics are computed by the single-source calculators in `backend/scripts/metric_definitions.py`, with authoritative definitions in [`docs/metric_definitions.md`](docs/metric_definitions.md): ROC-AUC, PR-AUC, recall, precision, FPR, Recall@1%FPR (validation-selected threshold, applied unchanged to test), confusion matrix, prevalence. One implementation per metric — no script may silently reimplement a metric differently.

### Seeds and determinism

All stochastic components record their seed (`--seed 42` default in `train_compare.py` and `evaluate.py`; bootstrap CI seed recorded per report). Model training uses seeded sklearn/XGBoost estimators. **Not claimed:** bit-for-bit reproducibility across different hardware/BLAS builds — floating-point reductions in tree ensembles and linear algebra libraries can differ across platforms; records pin the environment so a reviewer can match it.

### Cross-dataset result matrix

All major evaluations are reported in the standardized matrix (`backend/scripts/result_matrix.py`): Dataset | Domain | Training relationship | Samples | Fraud rate | ROC-AUC | PR-AUC | Recall | FPR | Status. Rows are classified in a closed set (`IN_DOMAIN`, `SAME_GENERATOR_FAMILY`, `CROSS_DOMAIN`, `EXTERNAL`, `UNKNOWN` — unknown is never upgraded) and are **never aggregated into a single score**. Poor external results (ROC-AUC 0.435–0.595) remain visible alongside in-domain results.

## Statistical Interpretation

Every reported metric is an **estimate from a finite dataset**, not a universal population value:

- **In-domain benchmark numbers (e.g. ULB ROC-AUC 0.966) carry sampling uncertainty.** Bootstrap confidence intervals (percentile method, stratified by class, seed recorded) are available via `--bootstrap`; the point estimates in this README are the full-sample values.
- **Small or imbalanced splits produce unstable estimates.** Confidence intervals are withheld when a conditioning cell has fewer than 30 observations; reports carry explicit `SMALL_POSITIVE_CLASS` / `HEAVY_IMBALANCE` warnings rather than spuriously precise numbers.
- **Prevalence context is mandatory.** At fraud prevalence *p*, an always-legitimate classifier achieves accuracy `1 − p` — accuracy is never the headline metric here, and PR-AUC is always read against its prevalence baseline.
- **No statistical-significance claim** is made for any model-vs-baseline or model-vs-model comparison unless a documented test supports it; comparisons share the same split and metric implementation, which makes them fair but not automatically significant.
- **Baselines** (majority-class, random-score) are evaluated on the same eligible split as the model so "better than baseline" is checkable, not asserted.
- **Distribution-shift findings are descriptive**: where poor external performance coincides with measured shift, reports say the result is *consistent with* distribution shift — not that shift provably caused it.

## Reproducibility

### Training and evaluation

```bash
python src/generate_synthetic_data.py          # → data/transactions.csv
python src/train_compare.py                    # → models/artifacts/ (metrics, EVALUATION.md, model files)
python backend/scripts/ibm_train.py --max-rows 1200000  # → IBM v2 model (1.2M rows)
python backend/scripts/cross_dataset_eval.py   # → reports/cross_dataset/cross_dataset_report.json
```

### External evaluation

```bash
python experiments/phase24_conditional_external_eval/run_evaluation.py           # E_hardneg on Kaggle
python experiments/phase24_conditional_external_eval/run_evaluation.py --dry-run  # dry run
```

### Regression tests

```bash
python scripts/regression_suite.py          # fast mode (11 suites, ~30s)
python scripts/regression_suite.py --full   # full mode (17 suites, ~95s)
```

### Live walkthrough

```bash
bash scripts/live_curl_walkthrough.sh       # 3-service allow/step-up/verify (fastest)
bash scripts/live_walkthrough.sh            # full 5-service register-to-audit chain
```

### Key artifacts

| Artifact | Location | Description |
|---|---|---|
| Model metrics | `models/artifacts/metrics_comparison.csv` | Per-model ROC-AUC, PR-AUC, recall@1%FPR |
| Generalization | `models/artifacts/generalization_by_archetype.csv` | Leave-one-archetype-out evaluation |
| Calibration | `models/artifacts/calibrator.joblib` | Platt calibration model |
| Rules config | `src/risk_engine/rules.yaml` | PR-reviewed rules + severity_scale |
| Metadata | `models/artifacts/metadata.json` | Training provenance, feedback source, OOD gate |
| Eval ledger | `reports/evaluation_runs/eval_ledger.jsonl` | Append-only evaluation records (model/data hashes, git SHA, thresholds, seeds) |
| Metric definitions | `docs/metric_definitions.md` | Authoritative metric semantics (v1.0) |
| Cross-dataset | `reports/cross_dataset/cross_dataset_report.json` | IBM v2 vs synthetic model on 3 datasets |
| Leakage audit | `misc/reports/phase25/PHASE25_FINAL_REPORT.md` | 10-check audit results |
| Claims registry | `misc/reports/CLAIMS_REGISTRY.json` | Evidence-classified claim inventory |

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Demo accounts (for testing)

| Account | Email | Password | Role |
|---|---|---|---|
| Alice | `alice@test.com` | `AlicePass123!` | user |
| Bob | `bob@test.com` | `BobSecure456!` | user |
| Carol | `carol@test.com` | `CarolSafe789!` | user |
| Dave | `dave@test.com` | `DaveTest012!` | user |
| Eve | `eve@test.com` | `EveDemo345!` | user |

Admin access:
- **Front page**: http://127.0.0.1:8000 → click 🔒 Admin
- **Admin passphrase**: from `.env` `ADMIN_PASS`
- **Compliance passphrase**: from `.env` `COMPLIANCE_TOKEN`

## Run

### ML pipeline

```bash
python src/generate_synthetic_data.py          # → data/transactions.csv
python src/train_compare.py                    # → models/artifacts/ (metrics, EVALUATION.md, model files)
```

### Feedback loop (§7: verification outcome → labeled data → retraining)

Confirmed/disputed outcomes in DB-3 are exported into a labeled training pool (joined to the stored feature vectors in DB-2, `confirmed` → 0, `disputed` → 1) and merged into the synthetic set for retraining. The pool is **accumulative and versioned**: every export rebuilds it from all DB-3 outcomes (idempotent, no double counting) and writes a dated snapshot `data/feedback_snapshots/feedback_YYYYMMDD_HHMMSS.csv`, so any point in the pool's history is reproducible:

```bash
python scripts/export_feedback.py                    # → data/feedback_labeled.csv + dated snapshot
python src/train_compare.py --feedback-date latest  # consume the newest snapshot
python src/train_compare.py --feedback-date 20260816   # reproduce training from any date
python src/train_compare.py --feedback data/feedback_labeled.csv  # explicit file (unchanged)
```

The trainer records the consumed snapshot's path and a content hash in `metadata.json` (`feedback_source`, `feedback_sha256`), so trained artifacts are attributable to an exact pool state. `scripts/training_pipeline.py` consumes the latest snapshot by default (falling back to the working pool file only when no snapshots exist).

**Section-6 gate + retrain-queue trigger.** An export is **gated** (exit code 2, nothing written) while the pool has fewer than `--min-outcomes` REAL verified outcomes (default 10; simulated rows never satisfy it; `--force` bypasses) — the pool must be big enough to train on. Once the pool's real **disputed** events reach `--disputed-threshold` (default 5), the export appends a hash-chained `retrain_trigger` event to the DB-4 audit trail (mirroring the drift monitor's `drift_alert`) — the signal that enough fraud-positive feedback has accumulated to justify a retraining round. A successful export that leaves the pool unchanged writes no new snapshot. A gated export leaves the latest snapshot intact, so the pipeline keeps consuming the last valid pool.

`--simulate N` appends a mechanics-demo wave sampled from an existing labeled CSV (archetype `feedback_sim`) — circular by construction, for validating the pipeline at volume, not as evidence of real-world gain. Retrained artifacts carry the feedback count in `model_version` (e.g. `seed42-transactions.csv+2fb`). See `models/feedback_loop_report.md` for the comparison.

### Training pipeline (§6/§7, one command)

Retrain → retune → apply → test as a single reproducible command:

```bash
python scripts/training_pipeline.py                       # retrain (with the feedback pool),
                                                          # sweep severity_scale, write rules.yaml,
                                                          # re-run the risk-engine tests
python scripts/training_pipeline.py --skip-train          # retune + apply + test on current artifacts
python scripts/training_pipeline.py --full                # also run the other six test suites
python scripts/training_pipeline.py --rules <staging.yaml> --skip-test   # dry-run apply elsewhere
```

Deterministic by construction (fixed seed; the sweep is pure numpy), so the same inputs reproduce the same artifacts, the same recommended scale, and the same green test run. The tuned `severity_scale` is written into `src/risk_engine/rules.yaml` atomically with the previous file kept as `rules.yaml.bak`; a scale-only change does NOT invalidate the tuner's per-event score cache (the key hashes the per-rule severities/conditions, not the scale), so re-runs stay ~0.1s. `--rules` targets a staging copy for dry-runs without touching the live config.

### Federated learning simulation (§19, stretch)

> **⚠️ This is a simulation.** No real financial institution participated in this experiment. The federation consists of simulated clients trained on partitions of synthetic data. The results demonstrate the algorithm's behavior under controlled conditions, not real-world institutional deployment.

Three simulated institutions on account-disjoint shards of one synthetic population. Each institution is a separate worker process that trains a balanced logistic regression (or, with `--mlp`, a one-hidden-layer ReLU MLP) locally and exchanges ONLY weight vectors with the coordinator, which averages them with FedAvg (weighted by local size). Raw data, features, and labels never cross the boundary. This is a local simulation — all worker processes run on the same machine. Compares local-only vs federated vs a centralized oracle:

```bash
python scripts/federated_test.py   # smoke test, 43 checks
python scripts/federated_sim.py    # full run -> models/federated_report.md
python scripts/federated_sim.py --mlp   # + LR vs MLP on the same shards -> models/federated_mlp_report.md
python scripts/federated_sim.py --dp-sweep 0.5,1,2,4,8   # DP privacy-utility trade-off
python scripts/federated_sim.py --hetero-sweep   # size x fraud-skew map -> models/federated_heterogeneity_report.md
```

**MLP comparison** (`--mlp`, `--hidden 16`, `--mlp-lr 0.1`): the same account-disjoint shards, same rounds and local epochs, run once with LR and once with the MLP (`src/federated/mlp.py` — hand-rolled for the same params/epochs contract as `LocalLR`; aggregation is per-layer FedAvg via `fedavg_params`, and the DP path uses a flat-vector variant, `dp_aggregate_flat`). The report compares FedAvg convergence and the **separability benefit** (fed MLP − fed LR PR-AUC). The honest finding on this dataset: the 6 engineered per-event features already linearize the fraud patterns (AND-of-flags interactions like CNP testing / ATO become linear flag combinations), so the MLP's nonlinear capacity buys nothing — it converges and tracks its own centralized bound, but the gap is ~0 (±0.01). The benefit needs raw inputs with genuinely nonlinear structure; the point of the comparison is the FL mechanics, not SOTA.

**Heterogeneity map** (`--hetero-sweep`): the same population re-sharded into the 3 institutions with different **size weights × fraud-rate skews** (`split_institutions_skewed` — each account is drawn into institution i with weight `w_i·(1 + (skew_i−1)·u)`, where u is the account's own fraud rate normalized to [0,1], so skew>1 pulls fraud-heavy accounts in and skew<1 repels them; account-disjointness always holds). Each config reports the FL gain (fed − local macro PR-AUC) and the per-institution **drag** (local_i − fed_i on i's own test — positive = the global model is worse for that institution than its own local model). The map's findings: FL helps most when data is concentrated (size-skew +0.097) or a clean dominant exists (clean dominant +0.143) — small institutions jump from local PR-AUC ~0.57–0.72 to ~1.00 under FedAvg; the only positive drag appears at the clean dominant's own test (+0.008) where its clean patterns are marginally diluted; and when the dominant already holds the fraud (fraud dominant) FL adds ~nothing (+0.002). Institutions whose test split has no fraud are marked n/a and excluded from the macro (a 0.0000 there would be a false 'disaster').

**Differential privacy** (`--dp-sweep` / `--dp-epsilon`): client-level DP FedAvg (McMahan-style) — the coordinator clips each per-client update delta to a norm bound S and adds Gaussian noise calibrated by **RDP composition** over all rounds (`src/federated/dp.py`). S is auto-calibrated to the clean run's median update norm; each epsilon is averaged over `--dp-seeds` noise draws (± std), and the trade-off vs the clean baseline lands in `models/federated_dp_report.md`.

Honest caveats (also in the reports): naive FedAvg without differential privacy leaks information via weight updates; the DP variant is CENTRAL (server-side noise — a real deployment would pair it with secure aggregation across actual network boundaries); with only 3 simulated institutions the DP budget is shared across a few contributors, so even ε = 8 costs ~2/3 of the clean PR-AUC and ε ≤ 4 leaves the model indistinguishable from random (that is the quantified price); LR and the small MLP are the weight-averageable architectures (RF/XGB/ISO are not); per-institution local standardization; the oracle is the theoretical bound computed out-of-band; synthetic data is cleanly separable so absolute scores are inflated. These results are from a local simulation, not a distributed deployment across real organizations.

### Drift monitoring (§6: population stability index + Phase 40 comprehensive monitoring)

Week-over-week feature-distribution checks against the distribution the models were trained on, extended by Phase 40 with schema drift, feature-availability drift, prediction-score drift, alert hysteresis, and promotion/retraining safeguards. The baseline is stored once as bin edges + expected proportions (never the raw training rows); each check compares only the current window — drift triggers a retraining review, it does not require retaining raw data:

```bash
python scripts/drift_monitor.py build-baseline                # → data/drift_baseline.json
python scripts/drift_monitor.py check                         # last 7 days from DB-2 (live path)
python scripts/drift_monitor.py check --window-csv window.csv # explicit window (offline/CI)
```

Exit codes: 0 = ok, 1 = warn (PSI ≥ 0.10), 2 = alert (PSI ≥ 0.25), 3 = insufficient data. An alert-level breach automatically appends a hash-chained `drift_alert` event to the DB-4 audit trail (feature PSI only, pseudonymous) — the compliance viewer shows it like any other event.

### k-anonymity gate on exports (threat scenario D)

Re-identification defense for *released* feature-store datasets: an attacker with auxiliary knowledge of a person's transaction behavior (amount, time, device, payee pattern) can match it against a released pseudonymous export and defeat pseudonymization. The gate rejects any release where a profile is uniquely identifiable — every combination of quasi-identifiers must occur in ≥ k rows drawn from ≥ k *distinct accounts* (k rows all from one account still reveal that account):

```bash
python scripts/k_anonymity_check.py --csv data/transactions.csv --k 5
python scripts/k_anonymity_check.py --db --days 30                 # live DB-2 export
python scripts/k_anonymity_check.py --csv release.csv --k 5 \
    --report kanon_report.json --suppress safe_release.csv
```

Quasi-identifiers default to the store's coarse, purpose-limited behavioral attributes (`txn_amount_bucket` + the boolean flags; §16 — no new raw data is collected to protect the release). Exit codes: 0 = k-anonymous, 1 = REJECT (uniquely identifiable profiles), 2 = input error. `--suppress` removes the violating rows into a k-anonymous release (publish less, do not collect more); the exit code still reflects the original dataset, so re-check the release. The current training set fails k=5 (103 long-tail rows in `extreme_relative_to_avg` flag combinations) and passes only after suppression — a real finding about what is safe to release.

### Services details

**Front page:** `src/front_service` (port 8000) serves the project landing page with live per-service status tiles (health, latency, uptime since restart, plus the audit chain's integrity as a footer badge), quick access to the alerts and compliance UIs, and the architecture/access overview — `GET /status` aggregates all five health checks server-side (no CORS needed), with the O(chain) integrity recompute cached on a 30s TTL. The page polls every 8s, backing off to 30s after three consecutive network failures (a dead host isn't hammered) and snapping back to 8s on the first success — a healthy all-down report keeps the fast cadence so recovery is spotted promptly. The offline path is covered twice: `scripts/front_service_test.py` forces it hermetically (unreachable URLs), and `scripts/front_offline_check.py run` stops the real five backends, asserts the all-down shape + badge, and restores the stack (opt-in — never part of the pipeline). Two corner entry points sit top-right: **👤 Sign in** logs in an account holder against the Identity Service's JWT (`/auth/login`, password never stored here — only the pseudonymous token is kept in the tab's sessionStorage, with an "Open the alerts UI" link for the next step) and **🔒 Admin** is the passphrase-gated admin area: login with `ADMIN_USER`/`ADMIN_PASS` (bootstrap passphrase from `.env`; only a scrypt hash is stored afterwards) to view the essential access payload — service map, credentials (masked by default), break-glass — which is persisted AES-256-GCM encrypted under a key derived from the passphrase (`db/admin.json`, minimal fields only; rotate via the panel). The admin area supports **TOTP 2FA** (RFC 6238, 30-second codes): once enabled, login requires both the passphrase and a 6-digit authenticator code. TOTP setup is inline — generate a Base32 secret, scan it with any authenticator app, verify the code to enable. Disable requires a valid code.

A hidden **Admin Database Explorer** (double-click the footer to reveal) provides read-only SQL access to all five databases (identity, features, risk, verify, audit). It requires admin JWT + passphrase, blocks all write operations (INSERT/UPDATE/DELETE/DROP), caps results at 500 rows, and audit-logs every query to the hash chain. Click a table name to auto-fill a `SELECT *` query.

The UIs build their service links from the browser's hostname, so the same stack works on localhost, LAN, or a VPS behind a proxy (see `HOSTING.md`).

Windows one-command stack: `powershell -ExecutionPolicy Bypass -File .freebuff/start_stack.ps1` — idempotent (stops python/uvicorn listeners on ports 8000–8005 and restarts all six); also the recovery path when a service has died. The demo button surfaces its seed summary ("Seeded 7 events · baseline committed 5/6 · 1 high-risk alert") — 5/6 is the typical count on a fresh account: the birth event steps up on its own brand-new device and is excluded from the baseline; re-seeds can exclude more (new demo device stays unknown until an allowed event), and the summary always shows the real count. After a decision, the verification UI's **History** toggle lists the caller's resolved cases (`GET /cases`, joined with the risk score) and the **feedback-pool** status (`GET /feedback-status`: confirmed/disputed counts + the retrain trigger threshold), auto-refreshing when a new decision lands while open. Each case card expands to show why it was flagged (reason chips) — `/cases` joins the reason codes/texts through to history — and a summary chip row (`N cases · confirmed · disputed`) matches the compliance trail's compact treatment. The whole screen is one fetch: the UI loads `GET /overview` (alerts + cases + feedback + chain status in a single BFF payload) instead of chaining `/alerts` + `/cases` + `/feedback-status` + `/chain-status`; opening History re-renders from that cached payload with zero extra calls. The alerts screen also **auto-refreshes every 30s while it's open** (silent — no spinner, errors keep the last good state; a generation token makes the newest load win), so new alerts appear without clicking Refresh.

**API client is generated from the OpenAPI schema** (`src/verification_service/static/api.js`): index.html fetches each service's `/openapi.json` at load time and builds a callable per endpoint (`api.identity.login(...)`, `api.verify.confirmAlert(...)`, `api.verify.complianceEvents({limit, offset})`, ...) — a new backend endpoint is automatically callable with no hand-written fetch wrapper. The FastAPI apps set `generate_unique_id_function` so `operationId == handler name` (stable, readable helper names); auth is per-operation (bearer default, compliance for the trail view, none for register/login/chain-status).

Dev secrets live in `backend/src/settings.py` (env-overridable; see `.env.example`).

**Presentation deck:** `docs/PS-14_fraud_detection.pptx` (16 slides, dark theme, speaker notes) — regenerate with `python scripts/build_presentation.py`.

**Private access document (encrypted at rest):** `docs/ACCESS.md.enc` is a Fernet-encrypted copy of all private access info — credentials, service map, the data-store access matrix, and quick procedures. Open it with `python scripts/access_doc.py view` (the passphrase lives in the gitignored `.env` as `ACCESS_DOC_KEY`, so no prompt is needed locally); rotate with `python scripts/access_doc.py rekey`. The plaintext master `docs/ACCESS.md` is gitignored — never commit it.

### Smoke test

Exercises the full flow and asserts the DB-separation guarantee from section 2 (DB-1 has no feature tables; DB-2 has no PII, raw amounts, or device IDs):

```bash
python scripts/smoke_test.py        # identity + privacy, 35 checks
python scripts/risk_engine_test.py  # risk engine, 37 checks
python scripts/verification_test.py # verification flow, 30 checks (incl. compliance viewer)
python scripts/audit_test.py        # audit chain, 33 checks (incl. tamper)
python scripts/verify_export.py export.json   # regulator-side export verifier
python scripts/drift_test.py        # PSI drift monitor, 17 checks (incl. hash-chained alert)
python scripts/k_anonymity_test.py  # threat-D k-anonymity gate, 23 checks
python scripts/federated_test.py    # federated learning + DP + MLP + hetero map, 43 checks
python scripts/front_service_test.py # front page + admin + offline status, 33 checks
python scripts/security_test.py     # CORS + rate-limiting + headers + key separation, 9 checks
python scripts/resilience_test.py    # resilience beyond ML: DB/audit/privacy failure paths
python scripts/ood_gate_test.py      # OOD recall gate validation
python scripts/rules_gate_test.py    # rules backtesting gate validation
python scripts/pseudonym_separation_test.py  # tests that DB-2 compromise scenario doesn't directly reveal identity
python scripts/feature_compatibility_test.py # training/serving feature skew prevention
python scripts/production_gate_test.py       # fail-closed startup in production mode
```

Full test suite (17 suites, all passing):

```bash
python scripts/regression_suite.py          # fast mode (11 suites, ~30s)
python scripts/regression_suite.py --full   # full mode (17 suites, ~95s)
#   risk_engine_test.py     # 35 checks: scoring, bands, rules, attribution, DB-3
#   smoke_test.py           # full pipeline: register → ingest → evaluate → audit
#   k_anonymity_test.py     # k-anonymity gate on exports
#   pseudonym_separation_test.py  # identity/fraud DB isolation
#   ood_gate_test.py        # out-of-distribution recall gate
#   rules_gate_test.py      # rules.yaml validation gate
#   pipeline_test.py        # training pipeline mechanics
#   security_test.py        # CORS, rate-limiting, headers, key separation
#   verification_test.py    # verification flow, compliance UI, external scripts
#   audit_test.py           # audit chain, hash integrity, compliance UI
#   front_service_test.py   # front page, admin, offline status
#   resilience_test.py      # DB/audit/privacy failure paths
#   feedback_test.py        # feedback loop, snapshots, gate trigger
#   tuner_test.py           # operating-point tuning, cache mechanics
#   drift_test.py           # PSI drift monitor, alerts
#   federated_test.py       # federated learning, DP, heterogeneity
#   feature_compatibility_test.py  # training/serving feature skew
```

Additional tooling:

```bash
python scripts/load_test.py -n 500 -c 20   # performance/load testing
python scripts/data_retention.py --dry-run  # data retention enforcement (preview)
python scripts/fairness_review.py           # bias analysis across account segments
python scripts/security_scan.py --full     # security scan (custom SAST + dependency check)
python scripts/quality_scorecard.py        # consolidated metrics dashboard
python scripts/penetration_test.py         # self-authored security regression suite: 42 attack scenarios (results not equivalent to independent pentest)
python scripts/benchmark_comparison.py     # 1.2M dataset benchmark vs industry
python scripts/compare_ml_systems.py       # PS-14 vs 8 industry systems (rankings, privacy, explainability)
python scripts/ps14_client.py              # easy API client (CLI + Python)
```

## Hosting

One command, anywhere: `docker compose up --build -d` runs all six services (front :8000 + the five) with a shared data volume, health checks, and internal service-name URLs — the landing page is at http://localhost:8000. Full guide: **`HOSTING.md`** (LAN, VPS + Caddy reverse proxy, secrets, backups, hardening checklist). Without Docker, `.freebuff/start_stack.ps1` boots the same six locally.

### Security Hardening

For prototype hardening reference, see **`docs/SECURITY_HARDENING_GUIDE.md`** which covers:
- Pre-deployment checklist
- Secrets management and rotation
- Network security (TLS, mTLS, CORS)
- Authentication and authorization
- Data protection (encryption, backups)
- Container security
- Monitoring and logging
- Incident response
- Compliance (DPDP Act, RBI guidelines)

Run security scans before deployment:
```bash
python scripts/security_scan.py --full  # Security scan (custom SAST — not a comprehensive audit)
python scripts/security_test.py         # Runtime security tests (9 checks)
python scripts/penetration_test.py      # Self-authored security regression (42 attack scenarios; not a professional pentest)
python scripts/security_ci_gate.py      # CI gate: fails on ANY finding
```

**⚠️ Security Note**: The security tests (`security_scan.py`, `penetration_test.py`, `security_ci_gate.py`) are self-authored automated security regression tests, not independent penetration testing or a third-party security audit. They verify specific known attack patterns against the running prototype. A professional penetration test by qualified security personnel is recommended before handling real financial data.

**Dockerized retrain:** `bash scripts/docker_retrain.sh` (or `make retrain`) runs `training_pipeline.py --full` inside a compose `train` profile container (`./models` + `./data` bind-mounted), then rebuilds and restarts **only the risk service** — and only when every step passes, including the OOD recall gate. `make train-only` / `NO_RESTART=1` skips the rebuild.

### Cross-domain evaluation (4 public datasets)

PS-14 was evaluated against 4 public datasets mapped to the §16 feature space. This experiment demonstrates that domain-specific training substantially outperformed cross-domain transfer in this evaluation — it does not constitute a universal proof about fraud detection.

> **⚠️ These datasets are public research benchmarks, not real-world fraud data with verified provenance and label governance. The cross-domain matrix shows an observed limitation of this specific model family on these specific datasets.**

| Dataset | Source | Rows | Positive | Rate |
|---|---|---|---|---|
| Kaggle ULB Credit Card Fraud | Dal Pozzolo et al. | 284,807 | 492 fraud | 0.17% |
| UCI Credit Card Default | Yeh & Lien, 2009 | 30,000 | 6,636 default | 22.1% |
| UCI Bank Marketing | UCI #222 | 41,188 | 4,640 subscribe | 11.3% |
| UCI Diabetes Readmit | UCI #296 | 101,766 | 11,357 readmit | 11.2% |

**Cross-domain ROC-AUC matrix** (train → test):

| Train \ Test | Kaggle | Default | Bank | Diabetes |
|---|---|---|---|---|
| Kaggle | **0.94** | 0.36 | 0.44 | 0.60 |
| Default | 0.71 | **0.74** | 0.56 | 0.54 |
| Bank | 0.53 | 0.53 | **0.94** | 0.56 |
| Diabetes | 0.35 | 0.36 | 0.38 | **0.61** |

**Key finding:** In this evaluation, the diagonal (same-domain train/test) consistently outperformed off-diagonal (cross-domain) results. Cross-domain transfer dropped 17–58%, suggesting that PCA features and payment-behavior features encode substantially different signals across these datasets. This is an observed limitation of this model family on these datasets, not a general theorem about fraud detection.

> **Note:** The cross-domain data files are not included in the repository. Results require downloading datasets — see `backend/scripts/real_datasets_expanded.py`.

```bash
python backend/scripts/real_datasets_expanded.py   # 4×4 cross-domain evaluation
python backend/scripts/meta_ensemble.py            # 6-strategy meta-ensemble comparison
python backend/scripts/train_compare_uci.py        # UCI vs Kaggle head-to-head
```

### Meta-ensemble strategies

Six strategies were compared for combining domain-specific models on the same 4-dataset benchmark. These results are specific to this evaluation setup and should not be generalized to other fraud detection contexts:

| Strategy | Avg ROC-AUC | vs Single-Domain |
|---|---|---|
| Single-domain (best) | **0.825** | — |
| Universal meta-learner | 0.798 | -2.7% |
| Weighted average | 0.777 | -5.8% |
| Simple average | 0.773 | -6.3% |
| Best-of (max) | 0.771 | -6.5% |
| Meta-learner (LOO) | 0.396 | -52.0% |

**Verdict:** In this evaluation, if you know the domain → the domain-specific model performed best. If the domain is unknown or mixed → the universal meta-learner showed the smallest drop (-2.7%). These findings are specific to these 4 datasets and this model family.

### Model uncertainty (§2.1)

`/internal/evaluate` and `/internal/attribution` now return an `uncertainty` dict with `model_variance` (variance of the four base learner outputs), `model_disagreement` (max − min spread), and a derived `confidence` level (`"high"` if variance < 0.01, `"medium"` < 0.05, `"low"` otherwise). Low-confidence + high-score cases naturally surface for human review via the investigator case priority calculation. Raw per-model outputs are never exposed to end users — the public surface keeps §11 category-level reason codes. The uncertainty fields are audit-logged on every `score_generated` event so analysts can retroactively see which decisions the model was uncertain about.

### Idempotency / replay protection (§2.2)

Both `POST /internal/ingest-transaction` and `POST /internal/evaluate` are idempotent on `event_id`: a duplicate request returns the stored result with `"idempotent_replay": true` without re-auditing, so the hash chain never double-counts a replayed event. Expired idempotency records are bounded by the natural retention window (feature vectors after 90 days, risk scores after 180 days — see Data retention below).

### Staged model rollout (§2.3)

`src/risk_engine/model_registry.py` provides a file-based registry (`models/artifacts/registry.json`) for staged rollout: **shadow** mode runs the candidate alongside production, logging predictions separately without affecting decisions; **canary** mode routes a configurable % of traffic to the candidate; **auto-rollback** triggers on latency p95 > 500ms, error rate > 2%, or FPR spike. The registry keeps the last-known-good model for instant revert. Direct deploy (`mode: "direct"`) remains the default — shadow/canary is opt-in config, not a required detour.

### Model & feature versioning per decision (§2.4)

Every `score_generated` audit event now carries `feature_version` (currently `"v1"` — bumped when `ML_FEATURES` changes) and `rule_version` (SHA-256 hash of `rules.yaml`, first 12 chars). An investigator can answer "which model, which feature schema, which rule set produced this decision" from the audit trail alone, without cross-referencing `models/artifacts/`.

### Investigator case workflow (§2.5)

A case-state machine for medium/high-risk events: `NEW → REVIEWING → USER_VERIFICATION → CONFIRMED_SUSPICIOUS / CONFIRMED_LEGITIMATE → CLOSED`. Each transition is audited to the hash chain. Investigators see pseudonymous ID + score + confidence + reason cases + prior related alerts — never real identity outside break-glass. Priority is auto-computed as `risk_score × confidence_weight` (low-confidence gets 1.5× boost for human review). Endpoints: `GET /investigator/cases` (filter by status/priority), `POST /investigator/cases` (create), `POST /investigator/cases/{id}/transition` (state change).

### Data retention policy (§2.9)

`scripts/data_retention.py` enforces per-store retention periods:
- **DB-2 (features.db):** 90-day retention for feature vectors; behavioral profiles kept indefinitely (they ARE the account baseline).
- **DB-3 (risk.db):** 180-day retention for risk scores without outcomes; verification outcomes kept indefinitely (training data).
- **DB-4 (audit.db):** 365-day retention via redaction-in-place (append-only triggers temporarily disabled, payloads replaced with redacted stubs, triggers re-created — the hash chain stays unbroken).

Run with `--dry-run` for safe preview. Schedule as a cron/systemd timer in a deployment environment.

### Fairness / bias review (§2.10)

`scripts/fairness_review.py` segments risk scores by account tenure, device count, and time pattern to detect disparities. It also analyzes the §16 feature set for plausible proxy variables (hour_of_day → work schedule, unusual_location_flag → geography, account_tenure_days → age) and recommends whether each is necessary. No new sensitive attributes are collected — the review works entirely from features the system already has.

### Performance / load testing (§2.7)

`scripts/load_test.py` measures avg/p50/p95/p99/max latency and throughput for `/internal/evaluate` and `/internal/ingest-transaction` with configurable request count and concurrency. Run against a live stack:

```bash
python scripts/load_test.py -n 500 -c 20    # 500 requests, 20 workers
python scripts/load_test.py -n 100 --json    # JSON output for CI
```

### Resilience beyond ML failure (§2.11)

`scripts/resilience_test.py` verifies safe failure for DB unavailability, audit-service unavailability, and Privacy Layer unavailability — each should fail safely (no silent security downgrade, no ambiguous state), matching the existing ML fail-open pattern. Also confirms security middleware (CORS, rate-limiting, headers) is present on all 5 services, idempotency checks exist at ingest and evaluate, model uncertainty is computed, feature/rule versions are stamped per decision, and the investigator case workflow is functional.

## Layout

```
src/
  settings.py                   # shared config (DB paths, secrets) - env-overridable
  generate_synthetic_data.py    # synthetic transaction generator (§16 shape)
  train_compare.py              # four-model comparison + artifact export
  identity_service/             # DB-1: PII, auth, pseudonym_mapping (FastAPI)
  privacy_layer/
    features.py                 # shared derivation: buckets, escalation, ML_FEATURES
    main.py                     # DB-2: ingest-transaction (FastAPI)
  risk_engine/
    main.py                     # DB-3: /internal/evaluate (FastAPI)
    fusion.py                   # stacked meta-learner over the 4 models
    model_registry.py           # staged rollout: shadow/canary/rollback
    rules_engine.py             # declarative rule evaluator
    rules.yaml                  # PR-reviewed rules config (§12)
    reason_codes.py             # §11 category-level explanation texts
  verification_service/
    main.py                     # alerts / confirm / reason + demo seed
                                # + /compliance/* proxy + investigator case workflow
    models.py                   # VerificationOutcome + InvestigatorCase (DB-3, §13)
    static/index.html           # self-contained verification UI (no build)
                                # incl. compliance trail view + integrity
  audit_service/
    main.py                     # compliance viewer + integrity checks
    writer.py                   # hash-chaining append path (shared)
    models.py                   # audit_events + audit_access_log (DB-4, §13)
  drift_monitor/
    psi.py                      # PSI math, baseline binning, thresholds (§6)
  k_anonymity/
    checker.py                  # threat-D k-anonymity + suppression (§16)
data/
  transactions.csv              # numeric feature matrix + label (ML input)
  events_sample.jsonl           # §16 "live event" shape, label=null
models/artifacts/
  metrics_comparison.csv / recall_by_archetype.csv / EVALUATION.md / *.joblib / metadata.json
db/                             # SQLite stores (gitignored): identity.db, features.db, risk.db
scripts/
  smoke_test.py                 # identity + privacy end-to-end + separation
  risk_engine_test.py           # bands, reason codes, DB-3 separation
  verification_test.py          # four-service verification flow
  audit_test.py                 # hash chain, tamper detection, append-only
  verify_export.py              # regulator-side verifier for the signed export
  drift_monitor.py              # PSI baseline build + window check + alerts (§6)
  drift_test.py                 # PSI math, drift detection, alert/audit
  k_anonymity_check.py          # k-anonymity gate on exports + suppression
  k_anonymity_test.py           # threat-D checks (unique profile, accounts)
  tune_operating_point.py       # cost-weighted tuning: global severity-scale sweep
                                #   and constrained per-rule rebalancing (§6)
  tune_test.py                  # tuner score-cache + per-rule search (23 checks)
  training_pipeline.py          # one-command retrain -> retune -> apply -> test (§6/§7)
  pipeline_test.py              # pipeline apply/backup + train-argv wiring (16 checks)
  feedback_test.py              # versioned feedback pool: snapshots, resolution,
                                #   gate + retrain-queue trigger (25 checks)
  live_walkthrough.py           # live HTTP walkthrough (servers must be up)
  live_curl_walkthrough.sh      # self-contained curl walkthrough (boots 3 services
                                #   on a clean db/walkthrough, registers + seeds +
                                #   ages, prints the allow/step-up/verify decisions)
  live_walkthrough.sh           # register-to-audit walkthrough (boots ALL FIVE
                                #   services, adds the verification loop, prints the
                                #   hash-verified audit trail + independent export check)
  load_test.py                  # performance/load testing: latency, throughput, error rates
  data_retention.py             # per-store retention enforcement (90d/180d/365d)
  fairness_review.py            # bias analysis across account segments + proxy variable review
  resilience_test.py            # resilience beyond ML: DB/audit/privacy failure paths
  security_test.py              # CORS, rate-limiting, headers, key separation tests
  front_service_test.py         # front page + admin + offline status tests
  penetration_test.py           # self-authored security regression: auth bypass, SQLi, JWT forging, IDOR, allowlist tests
  pseudonym_separation_test.py  # verifies DB-2 isolation under tested compromise scenario
  feature_compatibility_test.py # prevents training/serving feature skew
  production_gate_test.py       # fail-closed startup validation in production mode
  regression_suite.py           # unified test runner for CI/CD
  quality_scorecard.py          # consolidated metrics dashboard
  real_datasets_expanded.py     # 4-dataset cross-domain evaluation
  meta_ensemble.py              # 6-strategy meta-ensemble comparison
  train_compare_uci.py          # UCI vs Kaggle head-to-head training
  chameleon_rules_test.py       # chameleon fraud rule effectiveness
  fp_reduction_test.py          # false positive reduction analysis
  analyze_fp_rules.py           # FP rule analysis tool
  tune_parameters.py            # severity_scale sweep tool
  tune_rules.py                 # rule threshold tuning tool
  import_uci_default.py         # UCI dataset → PS-16 feature mapper
  deploy.sh                     # one-command prototype deployment
  retention-cron.sh             # cron wrapper for data retention
```

The model registry (`src/risk_engine/model_registry.py`) supports staged rollout (shadow → canary → rollback) with auto-rollback on breach. The investigator case workflow endpoints are in `src/verification_service/main.py` with the `InvestigatorCase` model in `src/verification_service/models.py`.

### Automated regression gates

Unified test runner for CI/CD:
```bash
python scripts/regression_suite.py   # runs all 17 test suites
```

Separate suites:
- **Fast** (11 suites, ~30s): pseudonym_separation, production_gate, smoke_test, risk_engine, drift_monitor, k_anonymity, ood_gate, rules_gate, tuner, pipeline, feedback_loop
- **Full** (+6, ~95s): security_test, verification_flow, audit_chain, front_service, resilience, federated_learning

## Live curl walkthrough (allow / step-up / verify)

The checked-in script `scripts/live_curl_walkthrough.sh` is the fastest way to see the whole flow over real HTTP — no pre-started servers needed. It boots Identity (:8001), Privacy (:8002) and Risk (:8003) on a dedicated, gitignored `db/walkthrough/` store, registers a throwaway pseudonymous user, seeds six normal transactions, **ages** the account to ~60 days (backdates DB-2 rows so the time-derived features reflect a real history — a demo harness; streaming events would carry real timestamps), and evaluates three scenarios via plain `curl`:

```
bash scripts/live_curl_walkthrough.sh
```

```
scenario               decision   band     score   reasons
A                      ALLOW      low      2       none
B                      STEP_UP    medium   64      NEW_DEVICE,LOCATION_UNUSUAL,RECIPIENT_UNUSUAL
C                      VERIFY     high     100     AMOUNT_UNUSUAL,NEW_DEVICE,UNUSUAL_TIME,
                                                   LOCATION_UNUSUAL,RECIPIENT_UNUSUAL,
                                                   AUTH_ANOMALY,BEHAVIOR_DEVIATION
```

It also prints the §16 derived vector the Privacy Layer actually stored for scenario A (no raw amount, no device id, no exact geo). Ports 8001–8003 must be free — stop the dev stack first; the services it starts are stopped again on exit (trap).

## Live register-to-audit walkthrough (all five services)

`scripts/live_walkthrough.sh` is the full pipeline version: it boots **all five** services — Identity (:8001), Privacy (:8002), Risk (:8003), Verification (:8004), Audit (:8005) — on the same clean `db/walkthrough/` store, and runs the whole register-to-audit chain over plain `curl`:

```
bash scripts/live_walkthrough.sh
```

1. register a throwaway user → pseudonymous `fraud_id`, log in → JWT,
2. seed six normal transactions + age the account to ~60 days (as above),
3. evaluate the three scenarios → the allow / step-up / verify table,
4. **verification loop** — list the high-risk alert (`/alerts` with the JWT), dispute it (`this_wasnt_me`) → case id + recovery flow; the blocked event never enters the behavioral baseline,
5. **hash-verified audit trail** — the pseudonymous decision trail for the user (`feature_ingested` × 9, `score_generated` × 3, `verification_resolved` × 1, each with its `entry_hash`), the full-chain compliance export re-verified **independently** by `scripts/verify_export.py` (HMAC signature + per-entry hash linkage recomputed from genesis), and the Audit Service's own integrity verdict.

Ports 8001–8005 must be free by default; set `PS14_*_PORT` (`PS14_IDENTITY_PORT` … `PS14_AUDIT_PORT`) to run alongside the dev stack on other ports — the flow is self-contained (direct curl + the shared local DB), so the running services are untouched. Services started by the script are stopped again on exit (trap).

## 50-case scenario/invariant test (live scoring)

> **⚠️ This is not real-world validation.** The 50 cases are hand-designed transaction scenarios and seeded perturbations, not verified real-world transactions. They test scoring behavior and system invariants, not detection performance on actual fraud.

`backend/scripts/batch_cases.py` runs 50 synthetic scoring scenarios through the LIVE Risk Engine (`/internal/evaluate`) and checks invariants that must hold regardless of model weights:

```
python backend/scripts/batch_cases.py [--recheck N] [--seed 7]
```

**Structure:**
* **5 hand-designed scenarios inspired by common fraud patterns** (new device at the usual location; old device 250 miles away; new device + $1,500; usual device + $1,200; different location, small amount) — these are named and printed in their own table for visual inspection;
* **45 seeded perturbations** across the §16 feature space (amount ratio, velocity, tenure, mule-ring, spend caps, auth anomalies) generated from a base vector with random feature values.

**What it measures:**
* Scoring behavior: valid score (0–100), band, and reason codes for all 50
* Determinism: re-evaluating a vector yields the same decision
* Monotonicity: adding a red flag never lowers the score
* Audit chain integrity: hash chain grows by exactly the number of events and still verifies from genesis

**Limitations:** This test does not establish detection performance on real-world transactions. The scenarios are synthetic and the perturbations are randomly generated. Results reflect scoring consistency, not fraud detection accuracy.

Requires the live stack (:8003 + :8005) — it is a live check, not part of the pipeline. Exit 0 = all invariants held; the decision table and the scenario bands are printed for reading. A clean run writes a last-run summary to `DB_DIR/batch_report.json` (gitignored) which the front page serves at `GET /batch` and renders as the **"Last 50-case batch run"** panel — run time/seed, the allow / step-up / verify distribution with bars, chain growth and integrity, and the five named scenarios — refreshed on load and every 60s, hidden until a report exists.

## Notes

- **Risk Engine composition** (§5): the four model outputs are fused by a class-balanced logistic stacker fit on validation (`stacker.joblib`), and the final score is `100 * max(ml_fusion, rule_score)` — the strongest signal wins, so hard red flags can't be diluted by a confident ML pass. Critical rules (`level: critical`) floor the score at 80. The fusion beats every solo model (current test split: PR-AUC 0.9375 vs best solo 0.9340). These metrics are on synthetic data.
- **Cold-start false positives** (fixed): the generator and the live Privacy Layer both clamped time-derived features to a 1-day floor, so training never saw young accounts and every brand-new account's normal transaction scored ~0.7. The floor is removed (real sub-day tenure, minutes-to-hours first swipes, and zero-known-device accounts are all represented in training) and the live clamp is gone, so a fresh account's normal first event scores ~0.41 (step-up, not verify) and a known-device event ~0.02.
- **Operating point** (§6): `severity_scale: 0.45` in `rules.yaml` is tuned by `scripts/tune_operating_point.py` with a cost-weighted objective (C_FN/C_FP = 20; step-up friction 0.15–0.25). Per-event scores are scale-invariant, so the tuner computes them ONCE and caches them on disk (`models/cache/`, gitignored), keyed on the dataset content + model artifacts + the per-rule severity/condition state: a re-run (any cost ratio) hits the cache and finishes in ~0.1s instead of ~3 min; retraining, regenerating the data, or editing a rule's severity/condition invalidates it, while a `severity_scale`-only write (e.g. by `scripts/training_pipeline.py`) deliberately does not. The post-cold-start-fix retune cuts legit challenges from 1.77% (25 events) to 0.14% (2) on the test split while holding fraud recall at 92.3% (48/52) and lowering total cost 118.2 → 70.6. The residual 2 legit challenges are ML-driven (above any rule scaling). See `models/tuning_report.md`. `--per-rule`
- **Link analysis / device graphs**: three cross-account features (`shared_device_accounts`, `shared_recipient_accounts`, `mule_ring_score`) are derived at ingest from DB-2 (counts of OTHER accounts on the device / recipient) and synthesized in training by a new mule-ring archetype (2–4 accounts sharing one device and a converging sink recipient, moderate amounts so amount features see nothing). A declarative `RULE_MULE_RING` (reason code `MULE_RING`) fires on shared device + shared/new recipient or a heavily shared device alone. Trained, tuned, and verified live: a three-account ring trips score 100 → VERIFY while a family phone shared by two legit accounts stays low (device-only sharing contributes 1/4 to the composite).
- **Calibration + odds**: the stacker output is mapped to the true fraud probability by **Platt scaling** (`PlattCalibration` in `src/risk_engine/calibration.py`) fit on an out-of-archetype pool blended with the prototype model's own validation predictions — the isotonic fit on near-separable data produced a step function, so it was replaced (`models/artifacts/calibrator.joblib`); `ml_score` is therefore a calibrated probability and the response carries `odds = p/(1-p)` (clamped at 9999 when p == 1) so thresholds carry business meaning across model versions. The score remains `100 × max(ml, rule)`. Calibration metrics (validated by `backend/scripts/calibration_test.py`): Brier score 0.0009, ECE 0.0013, 16/16 tests pass.
- **Fail-safe degraded mode**: if ML fusion raises or the circuit breaker (3 consecutive failures → open 30s, then half-open probe) is open, the evaluation falls back to the declarative rules with a fail-safe floor (score ≥ 31 = STEP_UP), tags the decision `degraded: true` with reason code `ML_UNAVAILABLE`, and records it in DB-3 + the audit chain — ML failure never silently produces ALLOW. A degraded ML never 500s the request.
- **Rules backtesting**: `scripts/simulate_rules.py` replays a candidate rule config (or a `--severity-scale`) over the labeled training history with the live calibrated fusion, reporting recall / FPR / decision-mix deltas and the exact events whose allow/step-up/verify decision would change — the "what-if" step before editing `rules.yaml`.
- **Velocity / spend limits** (pre-scoring enforcement, section 7): the Risk Engine evaluates per-account (max 10 txns / 24h hard cap, 3× median daily spend soft cap) and per-device (max 20 txns / 24h across accounts) caps configured in `rules.yaml` `velocity_limits`, derived entirely from the §16 vector the Privacy Layer builds (`account_daily_spend_ratio`, `device_daily_count` — raw amounts never reach the Risk Engine). Hard caps force score 100 → VERIFY regardless of ML+rules; soft caps floor at step-up. The 24h windows are anchored at the event's own timestamp, so backdated history (demo seed, replays) counts correctly.
- **SHAP-style attribution** (internal analyst tool): `POST /internal/attribution` explains the fused probability with exact per-model contributions (logistic stacker, logit space) and per-feature perturbation contributions against the training median — no `shap` dependency, nothing persisted, and never exposed to end users (the public surface keeps §11 category-level reason codes).
- **Device-graph viewer** (compliance role): `GET /compliance/device-graph` on the Verification Service proxies the Privacy Layer's `/internal/device-graph` behind the compliance passphrase. Given a hashed device or recipient token it returns the pseudonymous accounts sharing it (event counts, first/last seen) — the mule-ring signature. The compliance UI includes a lookup card with a one-click "use demo device" quick-fill after a demo seed.
- **Baseline discipline** (blocked events never pollute): ingest stores the §16 vector but **does not** touch the account's behavioral baseline. `POST /internal/commit-baseline` applies a stored vector to the profile (median EMA in ratio space — `median × (0.9 + 0.1 × ratio)` — so the raw amount is never stored; plus typical hours and device registration), and is called ONLY for **allowed** events (by the flow owner) or **confirmed** events (by the Verification Service on `this_was_me`). Blocked (step-up / verify) or **disputed** events never enter the baseline: a 4500 attack no longer inflates the median (later normal events keep ratio ~1.0, not 0.32) and an attacker's device never becomes "known". Feature inputs (freq, recent-window escalation) still count all attempts — only the stored baseline is gated. The walkthrough re-verifies this live.
- **Demo seed warm-up** (`/demo/seed`, dev-only — the UI's "Simulate demo activity"): before evaluating the attack, the seed first ingests six normal transactions spread over the past ~7 weeks and AGES the account in DB-2 via the Privacy Layer's dev-only `POST /internal/demo-age-account` (backdates `created_at` across [8, 55] days and the profile birth to 60 days — the service-side version of the curl walkthrough's aging step). The preview therefore shows exactly ONE alert (the attack, scored against a mature history) instead of cold-start false positives from a brand-new account. Each demo account gets its own device id: the fingerprint store is global (one row per hashed device), so a constant demo device shared across accounts would be "known" only to its first owner and every warm-up event would look like a brand-new device.
- **Audit trail** (§4/§13): `entry_hash = sha256(prev_entry_hash + canonical(payload))`, first entry linked to a documented genesis hash. Append-only is enforced at the storage layer (SQLite triggers abort UPDATE/DELETE), every compliance read is logged to `audit_access_log`, and `/audit/integrity` recomputes the chain and reports the first bad link — including payloads corrupted into non-JSON. Every privacy-sensitive action appends to the same chain: `score_generated` (Risk Engine), `verification_resolved` (Verification Service), `feature_ingested` (Privacy Layer — event id only, never the §16 features), `fraud_id_resolved` (Identity Service break-glass — actor + reason only; the PII stays in DB-1 and never touches the chain), and `drift_alert` (§6 drift monitor). The verification UI footer and the alerts screen show live chain-integrity status (polling `GET /chain-status` every 60s plus an immediate refresh after a demo seed or confirm; each poll is itself logged to `audit_access_log` as `verification-ui-footer`), and the compliance view auto-refreshes its integrity banner + trail every 20s while open. `GET /audit/export` dumps the full chain as a single signed document for an external regulator: HMAC-SHA256 over the canonical body (dev key from `export_signing_key`; production would use KMS/HSM asymmetric signing with the public half published), plus the integrity report. `scripts/verify_export.py` validates a copy independently - signature (authenticity) AND chain recomputation from genesis (integrity, not trusting the service's own report) - so a doctored export is rejected even if the service's `integrity` field still claims ok.
- Four fraud archetypes in the synthetic data: boiling-frog escalation, one-off impulse fraud, account-takeover bursts, and card-not-present testing (small, rapid txns to new merchants - deliberately invisible to amount-based signals), plus the mule-ring pattern. Each event is tagged with its `archetype`, and the trainer reports recall per pattern (`models/artifacts/recall_by_archetype.csv`, `EVALUATION.md`).
- **Chameleon fraud detection** (§12 rules): 6 deterministic rules in `rules.yaml` catch "chameleon frauds" — transactions that spread weak signals across many features instead of having one strong anomalous spike. These rules combine multiple moderate §16 signals (new device + unusual recipient + unusual time, etc.) and fire at severity 0.45–0.65. They complement the ML model, which misses ~14 out of 492 frauds (2.8%) on the Kaggle dataset because no single feature exceeds 3σ.
- **False positive reduction**: a micro-transaction discount in the Risk Engine reduces FPs by 23% (13→10) with only 0.2% fraud loss. The discount targets Amount < 0.1 with ML score 0.50–0.70 and no critical rules — the exact FP pattern where 10/13 FPs are micro-transactions with genuinely anomalous feature profiles.
- **Rules optimization**: `RULE_AMOUNT_HIGH` threshold raised from 1.3× to 1.8× average (cuts 1,713 FPs); `RULE_UNUSUAL_LOCATION` and `RULE_UNUSUAL_RECIPIENT` severity reduced from 0.30 to 0.20. Net result: same fraud recall, 29% fewer false positives, 23% fewer challenges to legitimate users.
- **Label-preserving feature noise** (`--noise`, default 0.15): raw events are perturbed before feature derivation (flag flips, cross-profile hour redraws, amount drift, auth jitter) so archetypes are not crisply separable - the models must learn structure instead of memorizing signatures. `--noise 0` restores the crisp generator. Noise draws from its OWN rng stream (`seed + 1`), so the row set and archetype mix are bit-identical across noise levels and comparisons stay apples-to-apples. Cost: cold-start FPR on held-out legit accounts rises with noise (2.7% at noise 0 -> 4.5% at 0.10 -> 20.3% at 0.15, F1-threshold basis), while the pinned live OOD vectors gain real discrimination (s3 $1,500 ml 0.013 -> 0.079) - pick the level for your FPR budget.
- **Honest generalization metrics**: the time-split test is optimistic - every fraud archetype also appears in training, so same-pattern near-duplicates leak across the split. `src/train_compare.py` therefore also runs a **leave-one-archetype-out evaluation** (the full ensemble is RETRAINED with each fraud archetype excluded and recall is measured on the never-seen pattern) plus a **cold-start probe** (whole legit accounts held out entirely, FPR on their rows). Results: `models/artifacts/generalization_by_archetype.csv` and the "Generalization" section of `EVALUATION.md`; the operating-point tuner appends the same table (plus per-archetype recall at the chosen scale) to `models/tuning_report.md`. The **pinned OOD set** (`data/ood_scenarios.csv` - the five live walkthrough vectors) is scored with the shipped artifacts every retrain as a regression check. `--no-group-eval` skips all of it.
- **OOD recall gate**: the retrain FAILS (exit 1, pipeline stops before retune/apply) when a gated held-out archetype's `recall_at_1pct_fpr` drops below a floor - default floor 0.60 on `ato,mule` (the weak archetypes), configurable via `--ood-recall-floor` / `--ood-gate-archetypes`, overridable with `--no-ood-gate`. The metric is the ranking-based recall@1%FPR, NOT `recall_f1` (unreliable at low fold fraud prevalence - the ato fold's validation is ~0.2% fraud). Gate results ride in `metadata.json` `ood_gate`.
- ~1/3 of rare-archetype accounts start late in the timeline so every pattern is represented in the out-of-time test split (raises the test fraud rate above the train rate - expected, by design).
- Feature derivation logic is shared (`privacy_layer/features.py`) between the generator, the trainer, and the live ingest path, so training features and production features stay consistent.
- Identifiers are CSPRNG-random 16-char base32-style IDs — not hashes of any PII — fitting the doc's `VARCHAR(16)` pseudonym column (80 bits; a true 128-bit ID would need ~26 chars, a schema change flagged in review).
- PII at rest is encrypted (Fernet/AES-256; a production deployment would use KMS envelope encryption); login uses a blind index on the email so plaintext is never queried.
- The fraud-id->identity bridge is internal-only and every resolution is logged with actor + reason (break-glass).
