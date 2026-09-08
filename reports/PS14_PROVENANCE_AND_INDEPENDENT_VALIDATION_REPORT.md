# PS-14 — CHECKS #23 & #24: DATA PROVENANCE / REPRODUCIBILITY + INDEPENDENT FINAL VALIDATION

**Date:** 2026-09-03
**Runs:** `scripts/data_provenance.py` (190 s) + `scripts/independent_validation.py` (993 s, full 24.4M-row pipeline re-run); `scripts/_fix_contamination_diffs.py` corrected a units bug in two presentation stats.
**Evidence:** `reports/data_provenance.json`, `reports/independent_validation.json`, `provenance_validation.log`.

---

## Check #23 — Data Provenance & Reproducibility: PASS with one CRITICAL structural finding

- **Dataset integrity reproduced exactly**: 24,386,900 rows / 29,757 fraud / 24,357,143 legit (0.122%); split counts 14,632,140 / 4,877,380 / 4,877,380 and fraud counts 17,736 / 6,260 / 5,761 all match the recorded report. Raw CSV SHA-256 recorded (`b01fa323…`); 13/13 deployed-artifact hashes match the governance record; `features_all.parquet` hash equals the deployed model's recorded `training_dataset_sha256` (`4ea90501…`) — the deployed model's training input is verified.
- **Label audit**: label = `Is Fraud? == "Yes"`, pre-labeled source file, 0 missing labels, no update mechanism. Delay/mutability/label-quality: **UNVERIFIED** (no confirmation timestamps exist) — carried from earlier checks.
- **Reproducibility**: double-run of a fixed seeded experiment (2M rows, XGB seed 42) → bit-identical (AUC 0.9937177614589153 both runs). The recorded forensic metrics were reproduced **exactly** by the fresh full-pipeline run (below) — run #1 vs run #2 differ nowhere.
- **Verbatim feature copy verified**: AST-extracted source of `expanding_features_fixed` from `forensic_revalidate.py` is byte-identical (CRLF-normalized) to the copy in `scripts/_forensic_common.py`.

## Check #24 — Independent Final Validation: **FAIL**

Independence delivered by: numpy-only metrics (rank-based ROC-AUC with average-rank ties; sklearn-equivalent AP-style PR-AUC; bincount confusion matrix), fresh pandas dataset recounts, independently written leakage tests, and independent val-only threshold re-derivation.

**Verified claims (fresh run reproduced every recorded number exactly):**

| Claim | Recorded | Independently verified | Status |
|---|---:|---:|---|
| ROC-AUC | 0.9948 | 0.9948 | PASS |
| PR-AUC | 0.7965 | 0.7965 | PASS |
| Recall | 0.9181 | 0.9181 | PASS |
| FPR | 0.009489 | 0.009489 | PASS |
| Precision | 0.1027 | 0.1027 | PASS |
| TP/FP/FN/TN | 5289/46226/472/4825393 | identical | PASS |
| Alerts | 51,515 (105.6/10K) | identical | PASS |
| Locked threshold | 0.094 (val-selected) | 0.094 (independent numpy sweep, val-only) | PASS |
| Bootstrap CI (ROC) | [0.9926, 0.9954] | [0.9921, 0.9959] (200-iter independent) | PASS (overlap) |

**CRITICAL failures — both are evaluation-design flaws the audit's own checks could not see:**

1. **The "chronological 60/20/20 split" is FALSE.** The file is ordered by **user** (2,000 contiguous user blocks), not by time. Each split block spans the full date range **1991–2020**; the row-count boundaries slice through user blocks (user 1199 straddles train/val, user 1585 straddles **val/test** — one test user's transactions informed validation threshold selection *and* final evaluation). The audit's sampled monotonicity check (every 100K rows) found 0 regressions only because date dips are localized at user transitions — the **full** check finds 5,789 regressions. The report's metadata (`train_end 2020-02`, `val_end 2013-07`…) are boundary-*row* dates mislabeled as period boundaries.
2. **Test features are contaminated with future-dated (relative to the row's own timestamp) merchant/city fraud.** Because the expanding-window state is file-order (user-major), an early-dated test row's `merch_fraud_rate`/`city_fraud_rate` include *other users'* transactions at the same merchant/city dated up to 2020 — after that row's own date. Quantified on a 500K-row test window (612 fraud): raw feature comparison shows **66.8% of merchant rows and 61.5% of city rows change** under strict chronology, with max |Δ| = 1.0 (audit rate ≈100% collapsing to ≈0% for early-dated rows whose merchant's fraud arrives later); re-scoring the same window with chronologically-correct rates (same model, same scaler) drops **AUC 0.9952 → 0.9229** and the locked-threshold operating point collapses (FPR 0.85% → 50.1%, recall 92.3% → 95.6%). The audit's "future-row perturbation PASS (max diff 0.0)" proves only *file-order* causality.

**Final accounting:** CRITICAL 2, HIGH 2 (shared val/test user; "temporal stability" windows are user-block segments spanning 1991–2020, not time windows), MEDIUM 1, MINOR 2. Verified claims 22/23 in the metrics table; the chronological-split claim is the one FAIL. **FINAL VALIDATION = FAIL** — the recorded numbers are real and bit-reproducible, but they describe a file-order (entity-holdout) experiment, not the chronological one the report claims.

**Files:** `scripts/_forensic_common.py`, `scripts/data_provenance.py`, `scripts/independent_validation.py`, `scripts/_fix_contamination_diffs.py`, `reports/data_provenance.json`, `reports/independent_validation.json`.
