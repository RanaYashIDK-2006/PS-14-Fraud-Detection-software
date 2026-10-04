# Phase 104 — Gap Matrix

**Date:** 2026-10-03
**Companion:** `docs/PHASE_104_FORENSIC_AUDIT.md` (evidence for every row),
`docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` (decision rules).

**Classification key:** DEMONSTRATED / SELF-TESTED / SIMULATED / NOT ESTABLISHED / BLOCKED.
**Requires key:** PUB = public-data experiment, BANK = bank data, EXT = external party,
LEGAL = legal/privacy/compliance, INT = internal engineering.

Phase IDs (R-01…R-48) are the dependency-aware roadmap IDs from the audit (Part S).
Nothing in this matrix is fixed in Phase 104; each row is work for a future phase.

## 1. Evidence and claims gaps

| ID | Gap | Evidence | Class | Requires | Phase |
|---|---|---|---|---|---|
| G-01 | CI cannot reject "metric in docs without evidence/run record" | `.github/workflows/ci-cd.yml` has no phase/claim/eval_record suites; ledger = 1 smoke record | NOT ESTABLISHED | INT | R-01 |
| G-02 | README headline tuple 0.966/91.8/0.877/97.8%@0.43 has no single provenance artifact; attributed to train_compare which runs synthetic data | `misc/reports/ulb_max_push.json` 0.966404 (r1 0.815); `ulb_results.json` 0.9758/0.8835/0.918; hardcoded dict in `compare_ml_systems.py` | NOT ESTABLISHED | INT/PUB | R-02, R-04 |
| G-03 | Brier 0.0009 / ECE 0.0013 not found in any measurement artifact (actual: 0.001185/0.003918 raw, 0.000469/0.000376 platt) | `misc/reports/calibration_and_stress.json`; only literal outside README is a phase71 docstring | NOT ESTABLISHED | INT/PUB | R-02, R-12 |
| G-04 | No untouched, pre-registered final test set exists | repeated `ulb_*` push/optuna reports (~20) indicate iterative tuning on same splits | NOT ESTABLISHED | PUB | R-16, R-13 |
| G-05 | `/internal/attribution` bypasses `enforce_before_inference` and `check_access` | `risk_engine/main.py:1278` vs :736/:853/:1121 | NOT ESTABLISHED (gap at runtime) | INT | R-06 |
| G-06 | Metric definitions lack Brier/ECE/reliability diagrams | `metric_definitions.py` v1.0: 0 mentions | NOT IMPLEMENTED | INT | R-02 |
| G-07 | `CLAIMS_MATRIX.md` M5 repeats train_compare misattribution | `misc/docs/CLAIMS_MATRIX.md` | NOT ESTABLISHED | INT | R-02 |
| G-08 | README claims "k-anonymity gate on data exports" — not wired into `/audit/export` | `audit_service/main.py:335`; grep in audit/verify/front empty | FALSE (doc/runtime mismatch) | INT | R-31, R-45 |
| G-09 | README claims missing files (`docs/PS-14_fraud_detection.pptx`, `docs/ACCESS.md.enc`) and root `scripts/deploy.sh` | paths absent / under `backend/scripts` | FALSE | INT | R-45, R-46 |
| G-10 | Penetration vector count discrepancy: README 42, `security_ci_gate.py` 34, static grep 66+43 call sites | `penetration_test.py` | NOT ESTABLISHED | INT/EXT | R-40 (with pen test), interim in R-01 |
| G-11 | Battery omits eval_record/claim audits; battery itself is self-reported | `.freebuff/p114_battery.sh` | SELF-TESTED | INT | R-01 |
| G-12 | README stale vs HEAD (no mention of repo phases 104–118; malformed table row L251) | README.md | NOT ESTABLISHED | INT | R-45 |
| G-13 | Observability lost on restart (in-memory) | Phase 71/72 notes in README L503 | PARTIAL | INT | R-30 |
| G-14 | PostgreSQL path unexercised (only compose.prod + migration 001) | `docker-compose.prod.yml`, `alembic/versions/001` | NOT ESTABLISHED | INT | R-25, R-27 |
| G-15 | No TLS in dev/preview compose | Caddyfile exists; compose lacks tls | PARTIAL | INT | R-26 |
| G-16 | No load/concurrency/recovery test artifacts | no perf suite among 115 tests | NOT IMPLEMENTED | INT | R-27 |

## 2. Scientific gaps

| ID | Gap | Evidence | Class | Requires | Phase |
|---|---|---|---|---|---|
| G-17 | Possible future-information in full-frame aggregates (`merchant_user_count`, `city_user_count`) and unknown dedup status | `ibm_train.py:180–188`; no dedup artifact | UNKNOWN (risk HIGH) | PUB | R-13 |
| G-18 | IBM v2 split user-disjoint but NOT chronological | `ibm_train.py:246–250` code comment | PARTIAL | PUB | R-08 |
| G-19 | OPTION A (public_v1, 14 features) vs OPTION B (defer to bank) undecided | `backend/research/public_feature_contract.json` exists; phase64: no public dataset has the 7 features | DECISION REQUIRED | PUB/BANK | R-03 |
| G-20 | 7 bank-only features unusable on public data; deployed 48-native model never evaluated externally on them | `privacy_layer/features.py`; phase61–65/88 blocker chain | BLOCKED | BANK | R-23, OPTION B branch |
| G-21 | No UNGATED-vs-GATED experiment (gating effectiveness unproven) | grep `ungated` → only `model_deploy.py` | NOT IMPLEMENTED | PUB | R-18 |
| G-22 | Ensemble superiority never tested against single best model under pre-registration | no ablation suite artifact | NOT ESTABLISHED | PUB | R-10, R-11 |
| G-23 | Calibration never validated externally | `calibration_and_stress.json` is in-domain only | SELF-TESTED | PUB | R-12 |
| G-24 | No subgroup/segment analysis on evaluated models (fairness_review is test-only) | `scripts/fairness_review.py` | NOT IMPLEMENTED | PUB | R-14 |
| G-25 | No business metrics (alert volume, burden, cost) | no artifact | NOT IMPLEMENTED | PUB/BANK | R-15 |
| G-26 | Label delay / point-in-time per-feature inventory not built | Part J audit | NOT IMPLEMENTED | BANK/PUB | R-23 |
| G-27 | BAF not acquired; license/semantics undocumented | `phase106_public_benchmark_registry.py:648`; `data/external_benchmark/baf/` absent | BLOCKED | PUB (download approval) | R-20, R-21 |
| G-28 | IEEE-CIS auth-blocked | Part M | BLOCKED | EXT | (only if pursued) |
| G-29 | Only 1 external-transfer dataset family exercised (Kaggle fraud + same-generator IBM) | phase24/25, cross_dataset_report | PARTIAL | PUB | R-21 |
| G-30 | Feedback-loop bias not analyzed | no artifact | NOT IMPLEMENTED | PUB/BANK | R-24 |
| G-31 | Feature-shift/availability analysis not done | no artifact | NOT IMPLEMENTED | PUB/BANK | R-22 |

## 3. Engineering gaps

| ID | Gap | Evidence | Class | Requires | Phase |
|---|---|---|---|---|---|
| G-32 | Hardcoded HMAC secrets (`_MANIFEST_SECRET`, `_TOKEN_SECRET`) | `release_manifest.py:70`, `promotion_gate.py:452` | FAIL-quality secrets hygiene | INT | R-28 |
| G-33 | No asymmetric signing (Ed25519) despite docstring recommendations | `settings.py:162`, `audit_service/export.py:6`, `SECURITY_HARDENING_GUIDE.md:128`; no ed25519 code | NOT IMPLEMENTED | INT | R-29 |
| G-34 | Audit writes synchronous on critical path; DB-4 outage fails fraud decisions | `writer.py` usage in risk/privacy | PARTIAL | INT | R-31 (re-verify) / R-27 |
| G-35 | Hash chain cannot detect forged appends by writer-capable services | Part N audit path | PARTIAL | INT/EXT | R-41 |
| G-36 | Deployed release manifest: `source_git_sha: "unknown"`, null seed, empty fingerprints | `release_manifest.json` | NOT ESTABLISHED | INT | R-04 |
| G-37 | Stale `altman_native_v2` schema string vs canonical v1 | `model_records/altman_native_E_hardneg_cert_20260904.json` | DOCUMENTATION DEFECT | INT | R-02/R-44 |
| G-38 | Repository sprawl (misc 719 files, 8 tracked .pyc, duplicate trail UIs, duplicated feature lists, test-only 42.5k LOC) | Part Q | CONSOLIDATION REQUIRED | INT | R-32 |
| G-39 | 45/P20 feature path declared impossible to reconstruct; degraded 33/45 → ROC 0.47 | phase21/23b artifacts | HISTORICAL / INACTIVE | — | documented only (no phase needed) |
| G-40 | Model artifacts gitignored — fresh clone cannot reproduce deployed model without retraining | `.gitignore` | PARTIAL | INT | R-48 (repro package decision) |

## 4. Institutional / external gaps

| ID | Gap | Evidence | Class | Requires | Phase |
|---|---|---|---|---|---|
| G-41 | 16 bank-readiness deliverables: 10 MISSING, 4 partial, protocol draft only | Part O | NOT STARTED | BANK/LEGAL/EXT | R-33…R-39 |
| G-42 | Zero external assurance items completed (6 planned) | Part P | NOT STARTED | EXT | R-40…R-43 |
| G-43 | No legal/privacy/compliance review | Part O15–16 | NOT STARTED | LEGAL | R-39 |
| G-44 | Pre-registration not approved | protocol doc marked DRAFT | PENDING REVIEW | EXT/INT | R-05 |
| G-45 | No paper/thesis, no reproducibility package, presentation lacks numbers | Part R | NOT STARTED | INT | R-46…R-48 |

## 5. Gate assignment summary

| Gate | Phases | Gap IDs addressed |
|---|---|---|
| 1 Foundation | R-01…R-06 | G-01…G-12 (evidence/claims), G-05, G-19 |
| 2 Research go/no-go | R-07…R-17 | G-02/03/04/17/18/22/23/24/25, G-44 |
| 3 Mechanism | R-18…R-24 | G-20/21/26/27/29/30/31 |
| 4 Engineering | R-25…R-32 | G-13/14/15/16/32/33/34/36/38/40, G-08 |
| 5 External/institutional | R-33…R-43 | G-28, G-35, G-41/42/43 |
| 6 Final freeze | R-44…R-48 | G-09/12/37, G-45 |

**REMAINING PHASES: 48** (full rationale in the audit, Part S).
