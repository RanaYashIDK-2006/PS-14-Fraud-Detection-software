# PHASE 19 — REAL-WORLD DATA & LABEL-GOVERNANCE GATE

## Executive Summary

PS-19 classification: **DATA_ACQUISITION_REQUIRED**.

No suitable real-world validation dataset exists in the repository.
All data is synthetic. The project cannot validate production performance.

## Key Finding

**The project has NO real-world labeled transaction dataset.**

| Dataset | Real-World? | Labels | Volume | Suitable? |
|---------|:-----------:|--------|--------|:---------:|
| IBM credit_card_transactions | No (synthetic) | Generated | 24.4M | No |
| Kaggle fraud | No (synthetic) | Unknown origin | 1.85M | No |
| PaySim | No (simulator) | Generated | 636K | No |
| UCI creditcard | No (PCA) | Unknown | 284K | No |
| Feedback snapshot | Yes (production) | Verification outcomes | 5 | No (insufficient) |

## Fraud-Rate Features

The 3 fraud-rate features (user_fraud_rate, merch_fraud_rate, city_fraud_rate)
are **UNAVAILABLE_AT_DECISION_TIME**.

- Training uses ground-truth expanding means (real fraud labels)
- Production uses proxy labels (score >= 70) — model outputs, not ground truth
- This creates a systematic distribution shift between training and production

**Recommendation: REMOVE_AND_RETRAIN**

## Label Governance

| Dataset | Label Definition | Latency | Decision-Time Usable |
|---------|-----------------|---------|:--------------------:|
| IBM | Generated (instant) | N/A (synthetic) | No |
| Feedback | VerificationOutcome | Retrospective | No |
| Production | UNKNOWN | UNKNOWN | UNKNOWN |

## Readiness Matrix

| Requirement | Status | Blocker |
|-------------|--------|---------|
| Real transaction data | UNAVAILABLE | No real-world dataset |
| Fraud labels | UNAVAILABLE | No ground-truth labels |
| Label provenance | UNAVAILABLE | No production pipeline |
| Label latency | UNVERIFIED | No timing data |
| Decision-time features | CONDITIONAL | 3/48 fail |
| Operational capacity | UNVERIFIED | No documented capacity |

## IBM Disposition

IBM remains valid for:
- Software testing, CI/CD, regression testing
- Feature pipeline testing, leakage testing
- Methodology development

IBM is NOT sufficient for:
- Claiming real-world performance
- Claiming production prevalence/precision/recall
- Validating production deployment

## Recommendations

1. **Obtain real-world validation data** (CRITICAL)
2. **Remove 3 fraud-rate features and retrain** (HIGH)
3. **Document production label-latency requirements** (HIGH)
4. **Establish operational alert-handling capacity** (HIGH)
5. **Keep E_hardneg deployed** as best available model
6. **Final test stays locked** until new candidate passes certification

## Firewall Status

- FINAL_TEST_ACCESSED: FALSE
- FINAL_TEST_AUTHORIZED: FALSE
- PRODUCTION_MODEL_STATUS: UNTOUCHED

## Classification: DATA_ACQUISITION_REQUIRED

The project needs real-world data before it can make production claims.
