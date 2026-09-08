# PHASE 17 REPORT -- IBM Generator / Data-Generating-Process Audit

## Executive Summary

Phase 17 investigated whether the dramatic 2017 chip-fraud regime transition
(7.8% chip fraud in 2016 -> 97.3% in 2017) is a generator artifact or a
meaningful temporal phenomenon.

**Verdict: SYNTHETIC_REGIME_ARTIFACT (STRONGLY_SUPPORTED)**

The 2017 transition is MOST LIKELY caused by an explicit generator-level
regime change, not organic concept drift.

## Key Evidence

### 1. Generator Source Not in Repository

The IBM `credit_card_transactions-ibm_v2.csv` dataset is a publicly available
synthetic dataset from IBM Research (SDV framework). The generator source code
is NOT in this repository. Analysis is purely statistical.

### 2. Abrupt Step Change at 2017

| Year | Chip % (all txns) | Chip % (fraud) | Fraud Rate |
|------|-------------------|----------------|------------|
| 2013 | ~5% | 0.0% | 0.12% |
| 2014 | ~8% | 0.0% | 0.06% |
| 2015 | ~12% | 9.2% | 0.19% |
| 2016 | ~17% | 7.8% | 0.21% |
| **2017** | **~80%+** | **97.3%** | **0.015%** |
| 2018 | ~80%+ | 84.8% | 0.14% |
| 2019 | ~80%+ | 90.8% | 0.12% |

The transition is ABRUPT (step change, not gradual) and AFFECTS ALL
TRANSACTIONS (not just fraud).

### 3. Persistent Regime

The chip-dominant regime persists from 2017 through 2019. This is NOT
a one-year anomaly.

### 4. Multivariate Shift

The transition affects channel composition, MCC distribution, merchant
diversity, and amount distribution simultaneously.

### 5. Statistical Dependence

Chi-squared tests confirm fraud and channel are dependent in all tested
years (p < 0.001). The dependence is strongest in 2017.

## Interpretation

The 2017 chip-fraud failure is a **SYNTHETIC ARTIFACT** of the IBM
data-generating process. It should NOT be interpreted as evidence of:

- Real-world concept drift
- Production temporal risk
- Model incapacity
- Data distribution limitation in the traditional sense

The most plausible real-world analogue is EMV chip adoption (US Liability
Shift, October 2015), but this is SPECULATION, not verified generator
documentation.

## Implications

1. **The 2017 failure is valid at the dataset level** but its production
   significance is UNVERIFIED.

2. **Further modeling against this dataset** to "fix" the 2017 failure
   is NOT justified -- it would be optimizing against a generator artifact.

3. **Protocol restructuring** would not help -- the underlying data is
   a generator artifact regardless of how you split it.

4. **Real-world validation data** is needed to determine whether the
   2017-like transition is a genuine production risk.

5. **E_hardneg remains the best production model.** The 2017 failure
   is a known limitation that should be documented and accepted.

## Firewall Status

- FINAL_TEST_ACCESSED: FALSE
- FINAL_TEST_AUTHORIZED: FALSE
- PRODUCTION_MODEL_STATUS: UNTOUCHED
- E_HARDNEG_STATUS: UNCHANGED

## Classification: SYNTHETIC_REGIME_ARTIFACT

The 2017 chip-fraud regime is MOST LIKELY a generator-level artifact.
Real-world validity is UNVERIFIED.
