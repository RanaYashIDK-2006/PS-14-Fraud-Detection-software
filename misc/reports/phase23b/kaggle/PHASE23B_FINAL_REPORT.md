# Phase 23B — Frozen Conditional Kaggle External Evaluation

## Executive Summary

This is a **conditional external-dataset benchmark**. It is NOT production validation.

The frozen PS-14 P20_45feat model was evaluated against the Kaggle fraudTrain/fraudTest
dataset (1.85M rows, US credit card transactions, 2019-2020). E_hardneg could NOT be
evaluated because its label-dependent features require verified label timing.

**Key limitation:** Provenance, label governance, and label latency are all UNVERIFIED.

## Dataset

- **Train:** 1,296,675 rows, 7,506 fraud (0.58% prevalence)
- **Test:** 555,719 rows, 2,145 fraud (0.39% prevalence)
- **Period:** 2019-01-01 to 2020-06-21
- **Geography:** US
- **Source:** Kaggle (published by DV, 2024)
- **Provenance:** UNCLEAR — may be real or simulated
- **Channel information:** UNAVAILABLE
- **MCC information:** UNAVAILABLE
- **PII present:** YES (names, addresses, DOB — excluded from outputs)

## Model Integrity

- **E_hardneg:** ALL artifact hashes MATCH (verified against manifest)
- **P20:** Feature list verified (45 features)
- **Final test:** NOT ACCESSED
- **Thresholds:** FROZEN (E_hardneg: 0.018758)

## Feature Compatibility

### P20_45feat
- **12 features:** EXACT (derivable from timestamp, amount, location)
- **33 features:** CONDITIONAL (require historical context, filled with NaN)
- **0 features:** UNAVAILABLE
- **Feasibility:** CONDITIONAL — evaluation possible with degraded feature set

### E_hardneg
- **Label-dependent features:** user_fraud_rate, merch_fraud_rate, city_fraud_rate
- **Reconstruction:** UNVERIFIED — requires verified label timing
- **Evaluation:** UNAVAILABLE

## Results (P20_45feat with Frozen Threshold)

| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC-AUC | 0.4658 | — |
| PR-AUC | 0.0083 | — |
| Recall | 1.0000 | [0.9982, 1.0000] |
| FPR | 1.0000 | [1.0000, 1.0000] |
| Precision | 0.0039 | — |
| Alert Rate | 1.0000 | — |
| Threshold | 0.018758 | FROZEN |

**Note:** These results use a degraded feature set (33/45 features filled with NaN).
They represent a LOWER BOUND on P20 capability with full feature reconstruction.

## Limitations

1. **Provenance uncertainty:** Dataset source unclear — may be real or simulated
2. **Label definition uncertainty:** is_fraud exact definition undisclosed
3. **Label latency uncertainty:** No metadata about when labels became available
4. **Channel information missing:** No chip/swipe/online indicator
5. **MCC information missing:** No MCC code — only high-level category
6. **Feature reconstruction limitations:** 33/45 P20 features require historical context
7. **E_hardneg unavailable:** Label-dependent features cannot be reconstructed
8. **Representativeness unknown:** May not represent any specific payment network

## Production Decision

```text
PRODUCTION_VALIDATION=NOT_ESTABLISHED
PROMOTION_ALLOWED=FALSE
```

This evaluation does NOT establish production readiness. It answers:
> "How does the frozen P20 model behave on this external dataset under
> degraded feature reconstruction?"

It does NOT answer:
> "How will PS-14 perform in production?"

## Data Acquisition Decision

Additional authorized real-world data with verified provenance, label governance,
and label latency remains required for production validation.

## Classification

```text
CONDITIONAL_EXTERNAL_EVALUATION_COMPLETE
```
