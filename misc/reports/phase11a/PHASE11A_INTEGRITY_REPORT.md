# Phase 11A — Integrity Reconciliation Report

**Date:** 2026-09-07T15:18:10Z

## Contradiction Resolution

### Contradiction A: 0.1175 vs 0.8127 recall

The machine-readable summary in Phase 11 reported **test** recall (0.1175) and test FPR (0.01101) as the "RECALL_AT_SELECTED_THRESHOLD" and "FPR_AT_SELECTED_THRESHOLD". These are TEST SET metrics (year >= 2018), not validation metrics. The validation metrics are correct:

- Validation recall @ locked threshold: **0.8127**
- Validation FPR @ locked threshold: **0.00998**

This is a **reporting bug** in the machine-readable summary section, not a computation bug.

### Contradiction B: 2017 recall = 0.2667

The 2017 forward validation shows **recall = 0.2667** with 255 fraud cases. This is significantly lower than 2016 (recall = 0.8516, 3579 fraud cases).

**Investigation findings:**

- 2017 has only 255 fraud cases (vs 3579 in 2016)
- Mean fraud score in 2017: 0.54583
- Mean fraud score in 2016: 0.90381
- 2017 fraud above threshold: 68 / 255

**Verdict:** CASE_2_VALID_DEGRADATION
**Explanation:** GENUINE DEGRADATION — 2017 fraud scores shifted down by -0.3580 on average; model does not generalize to 2017 fraud patterns

## Final Status

- METRIC_INTEGRITY: PASS
- FORWARD_VALIDATION_EXECUTION: PASS
- TEMPORAL_ROBUSTNESS: FAIL
- CERTIFICATION_STATUS: BLOCKED
- FINAL_TEST_AUTHORIZED: FALSE