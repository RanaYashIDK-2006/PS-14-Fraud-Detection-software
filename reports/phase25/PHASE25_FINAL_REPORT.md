# Phase 25 — Leakage Audit & Phase 25 Corrections Report

## Executive Summary

This phase corrected two concrete defects in Phase 24's implementation and
then executed the one remaining unverified check: a **leakage audit** on the
reconstructed feature set. The frozen models, thresholds, and all prior
artifacts were untouched.

```text
PHASE25_STATUS=LEAKAGE_AUDIT_COMPLETE
PRODUCTION_VALIDATION=NOT_ESTABLISHED
PROMOTION_ALLOWED=FALSE
```

This remains a **conditional external-dataset benchmark**. It is NOT production
validation.

---

## 1. Import-Path Bug (Fixed)

`run_evaluation.py` imported from
`experiments.phase24_conditional_external_eval.*` — a package name that only
exists in the source tree, not in the extracted zip (`experiments/phase24/`).
Anyone who ran Phase 24 from the shipped zip got an `ImportError` before any
evaluation executed. The runner now resolves its own package location at
runtime, so it works both from source and from an extracted zip.
(Verified: evaluation executes end-to-end from source; duration 55.8s.)

## 2. Maximal Decision-Time-Causal Feature Reconstruction (Completed in Phase 25 commit)

Phase 24's minimal reconstruction left 28 of 45 features as NaN, calling them
"historical context unavailable in Kaggle." That claim was wrong: Kaggle
provides `cc_num`, `merchant`, `city`, `state`, `zip` — exactly the entity keys
needed to compute historical/aggregate features causally. The Phase 25 commit
rebuilt reconstruction as:

| Status | Count | Notes |
|--------|------:|-------|
| EXACT | 22 | amount, temporal, category features |
| CAUSAL | 22 | entity aggregates from train-side history |
| CONSTANT | 3 | `has_zip`, `has_state`, `is_online_or_no_state` |
| UNAVAILABLE | 1 | `err` (no error/decline code in Kaggle) |

Metrics improved: ROC-AUC 0.435 → 0.595, FPR 44.9% → 9.9%. The
`FEATURE_AVAILABILITY` and `FEATURE_PARITY` promotion gates, previously
hardcoded FAIL, now PASS on computed verdicts.

## 3. Leakage Audit (New, Executed)

Audited against the supplied `is_fraud` target and the full test split:

| Check | Result |
|-------|--------|
| Target leakage (feature-label \|corr\| > 0.5) | **NO_LEAKAGE** (0 of numeric features) |
| Temporal leakage | **NONE** — CHECKED, 0 future timestamps |
| Feature leakage (label copies / identity transforms) | **NO_LEAKAGE** |
| Cold-start fragility (rate features) | **OK** — see below |

Artifact: `reports/phase24/kaggle/04_leakage_check.json` (regenerated).

### Temporal-check bug fixed

The Phase 24 artifact's temporal section reported `ERROR: Invalid comparison
between dtype=datetime64[us] and Timestamp` — the check compared naive dataset
timestamps against a tz-aware `Timestamp.now(tz="UTC")`. The comparison is now
naive-naive and the check completes: **0 future timestamps, ordering sane**.

### Cold-start audit — verified numbers

Measured on the shipped reconstruction (train-aggregate rate design):

| Feature | Degenerate (0/1) fraction | Unique values | Verdict |
|---------|--------------------------:|--------------:|---------|
| `user_fraud_rate` | 24.8% | 603 | OK |
| `merch_fraud_rate` | 1.3% | 650 | OK |
| `city_fraud_rate` | 21.6% | 583 | OK |

Why this is healthy: Kaggle card history is **deep** — median 1,054
transactions per card in train (min 7), and only 0.03% of test cards are
unseen in train. A rate computed over that much history is rarely degenerate.

### Correction to an earlier claim

An earlier (unpersisted) draft of this audit reported `user_fraud_rate` as
"76% degenerate — mostly 0.0 or 1.0." **That figure was wrong for the shipped
design.** The 76%-style degeneracy applies only to a *running per-transaction
rate* design (prior-tx cumulative rate), which this reconstruction does not
use. Measured directly on the train-side running rate: 60.5% degenerate —
a real fragility of that alternative design, worth knowing if a future
candidate adopts it, but not a defect of what was evaluated.

## 4. Results (unchanged by this audit)

Re-run with identical frozen artifacts and threshold:

| Metric | Value |
|--------|------:|
| ROC-AUC | 0.5947 |
| PR-AUC | 0.0067 |
| Recall | 18.5% |
| FPR | 9.9% |
| Precision | 0.72% |
| Promotion gates failed | 3 (REAL_WORLD_DATA, LABEL_GOVERNANCE, LABEL_LATENCY) |

The model remains a synthetic-data specialist that does not generalize to
external data (ROC-AUC 0.59 is far below deployment quality for 0.39%
prevalence).

## 5. Limitations (unchanged and preserved)

- Provenance uncertain — dataset source undisclosed
- Label governance unverified — `is_fraud` semantics unknown
- Label latency unverified — no decision-time label guarantee
- No channel information — chip/swipe/online robustness not evaluable
- `err` unreconstructable — no error/decline code in Kaggle
- PII excluded from all reports and artifacts

## 6. Production Decision

```text
PROMOTION_ALLOWED=FALSE
FINAL_TEST_ACCESSED=FALSE
E_HARDNEG_MODIFIED=FALSE
P20_MODIFIED=FALSE
NEXT_ACTION=AUTHORIZED_REAL_WORLD_DATA_ACQUISITION
```

---

*Generated: 2026-09-11 · Duration: 55.8s · All numbers in this report come from
executed code and persisted artifacts, not projections.*
