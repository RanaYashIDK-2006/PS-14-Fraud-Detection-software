# Phase 24 — Conditional External Evaluation Report

## Executive Summary

This is a **conditional external-dataset benchmark**. It is NOT production validation.

The frozen PS-14 E_hardneg ensemble (XGB+LGB+CatBoost, 48 features) was evaluated
against the Kaggle fraudTrain/fraudTest dataset using a minimal feature reconstruction.

## Dataset

- **Train:** 1,296,675 rows, 7,506 fraud (0.5789%)
- **Test:** 555,719 rows, 2,145 fraud (0.3860%)
- **Period:** 2019-01-01 00:00:18 to 2020-12-31 23:59:34
- **Provenance:** UNVERIFIED — may be real or simulated

## Model Integrity

- **E_hardneg:** ALL artifact hashes MATCH
- **Thresholds:** FROZEN (0.018758)
- **Final test:** NOT ACCESSED

## Feature Reconstruction

- **20 features:** EXACT (derivable from timestamp, amount, category)
- **28 features:** NaN-filled (require historical context unavailable in Kaggle)

## Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC-AUC | 0.4348 | — |
| PR-AUC | 0.0102 | — |
| Recall | 0.2681 | [0.2497, 0.2872] |
| FPR | 0.4486 | [0.4473, 0.4499] |
| Precision | 0.0023 | [0.0021, 0.0025] |
| Alert Rate | 0.4479 | — |

## Promotion Guard

- **Promotion allowed:** False
- **Failed gates:** REAL_WORLD_DATA, LABEL_GOVERNANCE, LABEL_LATENCY, FEATURE_AVAILABILITY, FEATURE_PARITY
- **Reason:** Conditional external evaluation — NOT production validation. Failed gates: REAL_WORLD_DATA, LABEL_GOVERNANCE, LABEL_LATENCY, FEATURE_AVAILABILITY, FEATURE_PARITY

## Limitations

1. **Provenance uncertain** — Kaggle dataset source is undisclosed
2. **Label governance unverified** — what constitutes fraud is unknown
3. **Label latency unverified** — cannot confirm labels at decision time
4. **Feature reconstruction partial** — 28 of 45 features are NaN
5. **No channel information** — cannot evaluate chip/swipe/online robustness
6. **No entity history** — cannot evaluate seen vs unseen merchants/users

## Production Decision

- **PROMOTION_ALLOWED:** FALSE
- **NEXT_ACTION:** Continue authorized real-world data acquisition

---

*Generated: 2026-09-10T14:22:05.836164+00:00*
*Duration: 17.2s*
