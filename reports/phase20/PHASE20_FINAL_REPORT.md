# PHASE 20 — DECISION-TIME FEATURE REMEDIATION & REAL-WORLD VALIDATION PREP

## Executive Summary

Built a 45-feature production-native candidate (P20_45feat) that does NOT
depend on the 3 unavailable fraud-rate features. All 45 features are
genuinely available at transaction decision time.

Classification: **CLEAN_PRODUCTION_COMPATIBLE_CANDIDATE**

## Candidate Identity

- ID: P20_45feat
- Features: 45 (removed user_fraud_rate, merch_fraud_rate, city_fraud_rate)
- Model: XGBoost + LightGBM + CatBoost ensemble
- Training: IBM dataset, time-based split (< 2016 train, 2016 validate)
- Seed: 42

## Feature Remediation

Removed exactly:
- `user_fraud_rate` — requires confirmed labels not available at scoring time
- `merch_fraud_rate` — same issue
- `city_fraud_rate` — same issue

No replacements with model scores, pseudo-labels, or proxy labels.

## Validation Results

| Metric | Value |
|--------|-------|
| Threshold | 0.067597 |
| Validation Recall | 0.9953 |
| Validation FPR | 0.2369 |
| Validation Precision | 0.4719 |
| Validation Alerts/1K | 369.89 |
| ROC-AUC | 0.9953 |

Forward validation (2017):
| Metric | Value |
|--------|-------|
| Recall | 0.9333 |
| FPR | 0.2322 |
| Precision | 0.0565 |
| Alerts/1K | 242.49 |

Note: 2017 is a synthetic regime artifact (Phase 17). Results reflect
IBM generator behavior, not real-world performance.

## Production Compatibility

- All 45 features genuinely available at decision time
- No label-dependent features
- No future-data dependencies
- Cold-start behavior: historical aggregates default to 0
- Feature parity with E_hardneg's 45 non-fraud-rate features: PASS

## What Changed vs E_hardneg

| Aspect | E_hardneg | P20_45feat |
|--------|-----------|------------|
| Features | 48 | 45 |
| Fraud-rate features | 3 (UNAVAILABLE) | 0 (removed) |
| Decision-time validity | CONDITIONAL | ALL_VALID |
| Production compatibility | CONDITIONAL | PASS |

## What Has NOT Changed

- E_hardneg: UNTOUCHED
- Final test: LOCKED
- Production: UNMODIFIED
- P11: UNCHANGED

## Real-World Validation Readiness

The candidate is ELIGIBLE for real-world validation but the validation
cannot be executed because no real-world dataset exists.

STATUS: BLOCKED_BY_DATA_ACQUISITION

## Recommendation

1. P20_45feat is the cleanest production-compatible candidate
2. It should be evaluated on real-world data when available
3. It should NOT replace E_hardneg without real-world validation
4. The 3 fraud-rate features should be permanently removed from the
   production pipeline

## Firewall Status

- FINAL_TEST_ACCESSED: FALSE
- FINAL_TEST_AUTHORIZED: FALSE
- PRODUCTION_MODEL_STATUS: UNTOUCHED
- E_HARDNEG_STATUS: UNCHANGED

## Classification: CLEAN_PRODUCTION_COMPATIBLE_CANDIDATE
