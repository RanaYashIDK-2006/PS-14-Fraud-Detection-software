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

- **22 features:** EXACT (directly derivable)
- **22 features:** CAUSAL (computed from training entity history)
- **3 features:** CONSTANT (always available)
- **1 features:** UNAVAILABLE (1/48)

## Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC-AUC | 0.5947 | — |
| PR-AUC | 0.0067 | — |
| Recall | 0.1851 | [0.1692, 0.2021] |
| FPR | 0.0993 | [0.0985, 0.1001] |
| Precision | 0.0072 | [0.0065, 0.0079] |
| Alert Rate | 0.0996 | — |

## Promotion Guard

- **Promotion allowed:** False
- **Failed gates:** REAL_WORLD_DATA, LABEL_GOVERNANCE, LABEL_LATENCY
- **Reason:** Conditional external evaluation — NOT production validation. Failed gates: REAL_WORLD_DATA, LABEL_GOVERNANCE, LABEL_LATENCY

## Limitations

1. **Provenance uncertain** — Kaggle dataset source is undisclosed
2. **Label governance unverified** — what constitutes fraud is unknown
3. **Label latency unverified** — cannot confirm labels at decision time
4. **Feature reconstruction maximal** - 47/48 features available (only err unavailable)
5. **No channel information** — cannot evaluate chip/swipe/online robustness
6. **No entity history** — cannot evaluate seen vs unseen merchants/users

## Production Decision

- **PROMOTION_ALLOWED:** FALSE
- **NEXT_ACTION:** Continue authorized real-world data acquisition

---

*Generated: 2026-09-11T04:42:30.740776+00:00*
*Duration: 55.8s*
