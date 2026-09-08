# PS-14 — Threshold Governance & Continuous Validation Report (Checks 19 + 20)

**Date:** 2026-09-03
**Companion data:** `reports/threshold_registry.json`, `reports/continuous_validation.json`, `reports/PS14_CONTINUOUS_VALIDATION_DASHBOARD.md`, `models/model_records/audit_25feat_xgb_inline_threshold.json`
**Code:** `scripts/threshold_governance.py`, `scripts/continuous_validation.py`, one fix in `src/risk_engine/main.py`

---

## Check 19 — Formal Threshold-Selection & Governance Process

### Documented procedure (validation-only, constraint-driven)

PS-14 operational constraints — **not** "industry-standard" claims; alert capacity was never specified, so no value is invented:

| Constraint | Value |
|---|---|
| FPR hard cap | 1.0% |
| Validation selection target | FPR < 0.9% (buffer for drift) |
| Minimum recall floor | 85% (Check-12 retraining rule) |
| Alert capacity | **UNVERIFIED** (never specified) |

Selection method (already enforced in the forensic pipeline, now codified in `scripts/threshold_governance.py`): sweep thresholds on **validation only** → pick the boundary operating point (max recall under FPR cap) → **LOCK** → evaluate once on the untouched test → verify stability on later chronological windows. Any threshold change requires a new model version + full revalidation.

### Operating points (audit model, val-selected, evaluated on untouched test)

| Threshold | FPR | Recall | Precision | TP | FP | FN | Alerts | Alerts/10K |
|----------:|----:|-------:|----------:|----:|----:|----:|-------:|-----------:|
| **0.094 (locked, 1%)** | 0.949% | 91.8% | 10.3% | 5,289 | 46,226 | 472 | 51,515 | 105.6 |
| 0.167 (0.5% target) | 0.489% | 89.4% | 17.8% | — | — | — | 28,987 | 59.4 |
| 0.491 (0.1% target) | 0.098% | 82.1% | 49.7% | — | — | — | 9,515 | 19.5 |

Val-selected thresholds: 0.094 / 0.167 / 0.491 for the 1% / 0.5% / 0.1% FPR targets (validation recall at 1% target: 93.5%). Test set never touched selection.

### Stability across chronological windows (locked threshold)

The Check-12 rolling protocol locked threshold **0.111** on window 1, then evaluated the **same** threshold on windows 2–5 (4 forward windows, ~16.3M rows):

| Window | FPR | Recall | Alerts/10K | Fraud rate |
|-------:|----:|-------:|-----------:|-----------:|
| 1 (selection) | 0.900% | 88.3% | 101.0 | 0.126% |
| 2 (forward) | 0.841% | 90.7% | 95.0 | 0.121% |
| 3 (forward) | 0.609% | 91.4% | 72.0 | 0.123% |
| 4 (forward) | 0.784% | 92.7% | 90.3 | 0.130% |
| 5 (forward) | 0.797% | 91.6% | 90.5 | 0.119% |

**Forward FPR range 0.61–0.84%, recall range 90.7–92.7% → threshold STABLE.** FPR stayed under the 1% cap and recall above the 85% floor in every window. The old per-window threshold reselection (which pinned FPR at 0.9% by construction) is gone — these are honest forward numbers.

### Threshold locking

- **Audit model (25-feat XGB):** locked thresholds stored in `models/model_records/audit_25feat_xgb_inline_threshold.json` (0.094, val-selected, test FPR 0.9489%, recall 91.81%, stability verdict, lock date).
- **Deployed lean model (`altman_lean_15feat_20260830_200346`):** **NOT LOCKED.** Governance record has `selected_threshold: None`; manifest records `temporal_test_r1: 0.0` (no recall@1%FPR was ever measured on this artifact). **FAIL — production runs a decision threshold with no validated provenance on the deployed artifact.**

### Production code threshold audit — 3 findings

1. **`band_of()` in `src/risk_engine/main.py`** — score ≥ 85 band (docstring claims ml_prob ≥ 0.85 → FPR ~0.8%) is a hardcoded comment with **no registry linkage** and is on a different scale than the audit model's locked thresholds. Verdict: **UNVERIFIED / NOT TRACEABLE**.
2. **Recall-gate defaults were inconsistent** — module constant `_RECALL_GATE_THRESHOLD = 0.85` ("FPR < 1% default") contradicted `set_recall_gate()` and the `/internal/recall-gate` endpoint default of `0.007` ("98.5% recall at ~8% FPR"). Enabling the gate without an explicit threshold silently flipped the operating point. **FIXED**: module default aligned to 0.007 (the documented high-recall point); `band_of()` (score ≥ 85) remains the FPR<1% decision; comment corrected. `smoke` and `risk_engine` suites still pass.
3. **`models/artifacts/threshold_config.json`** — orphaned thresholds (~0.86–0.94, different scale) with no model-version linkage and no loader in `src/`. Verdict: **ORPHANED**.

---

## Check 20 — Continuous Model Validation & Automatic Fail-Safes

`scripts/continuous_validation.py` is an always-runnable health gate (≈3 s, no retraining) that re-verifies the live engine every run and aggregates all prior evidence into GREEN / YELLOW / RED states. Bounds are PS-14 operational thresholds from Check 12.

### Dashboard state (current)

| Dimension | Status | Detail |
|---|---|---|
| Model performance | **GREEN** | recall 91.6% ≥ 85%, FPR 0.80% ≤ 1%, PR-AUC 0.801 (window 5) |
| Feature drift | **YELLOW** | 2 WARNING (PSI 0.114–0.246), 0 CRITICAL |
| Data quality | **RED** | monitor CRITICAL: 0.51% invalid timestamps (seed artifact — fix source before retrain) |
| Model integrity | **GREEN** | live: engine loads, 14/14 artifact-hash + schema checks pass |
| Production parity | **RED** | city_fraud_rate cache bug + audited model ≠ deployed model (Check 16) |
| Probability calibration | **GREEN** | test ECE 0.39% ≤ 1% |
| Alert volume | **GREEN** | 90.5/10K (capacity UNVERIFIED) |
| Threshold governance | **RED** | deployed model has no registry-locked threshold (Check 19) |
| Fail-safe probes | **GREEN** | 3/3 live (below) |
| Promotion gate | **RED** | 3 gates FAIL |

**Overall: RED — deployment not allowed. Promotion: BLOCKED.** This is the correct, honest state: the model is not blocked by its own performance (that dimension is GREEN) but by governance gaps (parity + threshold) and a data-quality flag.

### Live fail-safe probes (prove it fails visibly, not silently)

| Probe | Result | Detail |
|---|---|---|
| Tampered artifact (byte-flip in xgb) | PASS | engine refused to load |
| Feature-schema mismatch (feature_list altered) | PASS | engine refused to load |
| Missing artifact (scaler deleted) | PASS | engine refused to load |

Every failure mode in the Check-20 list maps to a documented response (full 7/7 matrix with corrupt-load/missing/invalid/schema/tamper/NaN/Inf is in `reports/deployment_test.json` from Check 18; 7/7 failed safely).

### Automatic promotion gate

| Gate | Result |
|---|---|
| Leakage validation | PASS (causality PASS) |
| Data-quality validation | **FAIL** (DQ CRITICAL) |
| Feature-parity validation | **FAIL** (city_fraud_rate cache bug + model mismatch) |
| Threshold validation | **FAIL** (deployed threshold not registry-locked) |
| Performance validation | PASS |
| Robustness validation | PASS (stress 24/24) |
| Security validation | PASS |
| Rollback validation | PASS (real rollback, identical V1 predictions) |
| Governance integrity | PASS |

**MODEL PROMOTION = BLOCKED** — a better AUC cannot bypass these gates; the gates themselves are the evidence.

### Label honesty

Performance metrics (recall/FPR/PR-AUC) are only computed on windows **with ground truth** (the rolling protocol). Unlabeled recent production rows contribute only to drift, alert-volume, and score-distribution checks — no fabricated recall/FPR is reported for label-free data.

---

## Files

- `scripts/threshold_governance.py`, `scripts/continuous_validation.py`
- `src/risk_engine/main.py` (recall-gate default fix)
- `reports/threshold_registry.json`, `models/model_records/audit_25feat_xgb_inline_threshold.json`
- `reports/continuous_validation.json`, `reports/PS14_CONTINUOUS_VALIDATION_DASHBOARD.md`

## Verdict

**The system is now continuously auditable**: any run of `continuous_validation.py` re-proves the live model's integrity, re-checks the promotion gates against evidence, and reports an honest overall state. What keeps PS-14 from being deployable is no longer hidden — it is a small, well-understood list: fix the `city_state` row-group reset, run the untouched-test protocol against the deployed artifact (locking its threshold into the registry), and clean up the DQ timestamp source.
