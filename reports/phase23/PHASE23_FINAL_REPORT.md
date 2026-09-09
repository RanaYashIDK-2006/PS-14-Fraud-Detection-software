# Phase 23 — Legitimate Real-World Data Acquisition Gate

## Executive Summary

Phase 23 evaluated **114 data sources** in the repository for real-world validation eligibility.

**No source has VERIFIED real-world provenance with complete metadata for PS-14 production validation.**

The best candidate is the **Kaggle fraudTrain/fraudTest dataset** (1.85M rows, US credit card transactions, 2019-2020), but it has critical gaps:

- **Provenance unclear** — may be real or simulated; source transactions undisclosed
- **No channel information** — chip/swipe/online not available
- **No label timing** — when was fraud confirmed relative to transaction?
- **Label definition undisclosed** — what exactly does `is_fraud` mean?
- **PII present** — names, addresses, dates of birth in the dataset

**Classification: CONDITIONALLY_AVAILABLE_WITH_CAVEATS**

The project must now decide whether to:
1. Proceed with conditional Kaggle evaluation (acknowledging all caveats)
2. Continue external data acquisition for production-grade evidence

---

## Sources Evaluated

| Category | Count | Examples |
|----------|-------|----------|
| SYNTHETIC | 5 | IBM v2, PaySim, PS-14 derived, federated shards |
| REAL (LEGACY) | 3 | ULB creditcard, Kaggle fraudTrain/Test |
| UNKNOWN | 2 | Elliptic Bitcoin (not credit card) |
| DERIVED | 100+ | PS-14 feature vectors, scores, validation sets |
| **REAL_WORLD_ELIGIBLE** | **0** | None meet all criteria |

---

## Best Candidate: Kaggle fraudTrain

| Criterion | Status | Notes |
|-----------|--------|-------|
| Real transactions | UNCLEAR | May be real or simulated |
| Provenance | PARTIAL | Kaggle published; source undisclosed |
| Authorization | PASS | CC0 public domain |
| Label definition | UNCLEAR | `is_fraud` column, exact meaning unknown |
| Label timing | UNVERIFIED | No metadata about when labels were created |
| Timestamps | CONDITIONAL | Present, parseable, timezone unknown |
| Schema compatibility | PARTIAL | 6/8 required fields; missing channel, MCC |
| P20 features | PARTIAL | ~30/45 derivable with rolling windows |
| E_hardneg features | CONDITIONAL | Requires label history |
| Statistical power | SUFFICIENT | 1.85M rows, ~8700 fraud cases |
| Temporal coverage | LIMITED | 2 years only (2019-2020) |
| Channel coverage | NOT_AVAILABLE | No channel column |
| Entity coverage | DERIVABLE | Can compute from cc_num/merchant |
| PII present | YES | Names, addresses, DOB — must minimize in reports |

---

## The Honest Assessment

This dataset is **real-world-looking** but not **real-world-verified**. The key uncertainty is whether these are actual bank transactions or sophisticated simulations. Without that provenance verification, any evaluation results carry a fundamental credibility limitation.

The correct classification is:

```
REAL_WORLD_DATA = CONDITIONALLY_AVAILABLE
PROVENANCE = UNCERTAIN
LABEL_GOVERNANCE = UNVERIFIED
LABEL_LATENCY = UNVERIFIED
PROMOTION_ALLOWED = FALSE
```

---

## Machine Summary

```
PHASE23_STATUS=COMPLETE

SOURCES_EVALUATED=114
REAL_WORLD_SOURCES_FOUND=3
REAL_WORLD_SOURCES_VERIFIED=0
REAL_WORLD_SOURCES_ACCEPTED=0

DATASET_NAME=Kaggle fraudTrain (BEST CANDIDATE)
DATASET_HASH=N/A (not yet hashed for evaluation)
DATASET_ROWS=1296676 (train) + 555720 (test)
DATASET_DATE_RANGE=2019-01-01 to 2020-12-31
DATASET_PREVALENCE=0.47%

PROVENANCE=UNCERTAIN
AUTHORIZATION=AUTHORIZED
LABEL_DEFINITION=UNVERIFIED
LABEL_GOVERNANCE=UNVERIFIED
LABEL_LATENCY=UNVERIFIED
TIMESTAMP_QUALITY=CONDITIONAL

SCHEMA_COMPATIBILITY=PARTIAL
P20_COMPATIBILITY=PARTIAL
E_HARDNEG_COMPATIBILITY=CONDITIONAL

STATISTICAL_POWER=SUFFICIENT_FOR_AGGREGATE
TEMPORAL_COVERAGE=LIMITED
CHANNEL_COVERAGE=NOT_AVAILABLE
ENTITY_COVERAGE=DERIVABLE
REPRESENTATIVENESS=UNCERTAIN

PRIVACY=REQUIRES_PII_MINIMIZATION
SECURITY=STANDARD
DATA_QUALITY=ACCEPTABLE_ON_SAMPLE

PHASE22_HANDOFF_READY=FALSE

IBM_STATUS=REJECTED_FOR_REAL_WORLD_VALIDATION

FINAL_TEST_ACCESSED=FALSE
FINAL_TEST_AUTHORIZED=FALSE

PRODUCTION_MODIFIED=FALSE
E_HARDNEG_MODIFIED=FALSE
P20_MODIFIED=FALSE

PROMOTION_ALLOWED=FALSE

NEXT_ACTION=DECIDE_KAGGLE_CONDITIONAL_EVALUATION_OR_CONTINUE_ACQUISITION

FINAL_DECISION=CONDITIONALLY_AVAILABLE_WITH_CAVEATS
```
