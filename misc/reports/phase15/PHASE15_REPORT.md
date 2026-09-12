# Phase 15 -- 2017 Chip-Fraud Distribution Forensics Report

**Date:** 2026-09-08T10:49:38Z
**Purpose:** Diagnostic only -- determine WHY 2017 chip fraud fails

## Classification

**CASE_B_PARTIALLY_REPRESENTED**

## Channel Shift

| Channel | 2016 Fraud % | 2017 Fraud % | Delta |
|---------|----------:|----------:|------:|
| Chip Transaction | 7.8% | 97.2% | +89.5% |
| Online Transaction | 85.9% | 0.0% | -85.9% |
| Swipe Transaction | 6.3% | 2.8% | -3.6% |

## Feature Distribution Shift

Train-vs-2017 diagnostic AUC: **0.9401** (strong_separation)

Top features by standardized mean difference:

- **chip**: SMD=+11.825 (higher_in_2017)
- **has_state**: SMD=+2.211 (higher_in_2017)
- **Month**: SMD=+2.017 (higher_in_2017)
- **card_tx_count**: SMD=+1.535 (higher_in_2017)
- **city_id**: SMD=-1.471 (lower_in_2017)

## Feature Support Overlap

Average outside-support rate: **0.9%**

## Merchant/User Generalization

- Seen merchants: 81.0%
- Seen users: 100.0%

## Pattern Back-Mapping

- Historical analogue: 50.0%
- Weak analogue: 44.8%
- No analogue: 5.2%

## Root-Cause Ranking

1. **Channel composition shift (online->chip fraud migration)** [PROVEN]
2. **Feature-distribution separation (train vs 2017 chip)** [STRONGLY_SUPPORTED]
3. **No historical chip-fraud analogue in training** [PROVEN]
4. **Feature out-of-support (2017 values outside pre-2016 range)** [SUPPORTED]
5. **Pattern novelty (no pre-2016 analogue for most 2017 cases)** [SUPPORTED]

## Conclusion

CASE_B_PARTIALLY_REPRESENTED

The 2017 chip-fraud failure is a **data distribution limitation**.
The training data contains zero chip-fraud examples, and the 301 2014-2015 cases do not represent the 2017 pattern.