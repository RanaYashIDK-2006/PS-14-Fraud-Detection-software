# Phase 14 - 2014–2015 Chip-Supervision Transfer Test Report

**Date:** 2026-09-08T09:47:39Z
**Objective:** Determine if adding 2014–2015 chip-fraud supervision improves 2017 transfer

## Temporal Protocol

| Candidate | Train | Validate | Eval |
|-----------|-------|----------|------|
| C0 (control) | <=2013 | 2014 | 2016+2017 |
| C1 (+2014 chip) | <=2014 | 2015 | 2016+2017 |
| C2 (+2014-15 chip) | <=2015 (80%) | <=2015 (20%) | 2016+2017 |

## Training Support

| Regime | Total Fraud | Chip Fraud | Online Fraud | Swipe Fraud |
|--------|----------:|----------:|------------:|-----------:|
| <=2013 | 17,012 | 0 | 11,478 | 5,534 |
| <=2014 | 18,064 | 0 | 12,371 | 5,693 |
| <=2015 | 21,345 | 301 | 15,148 | 5,896 |

## Primary Comparison Table

| Candidate | Chip# in Train | 2016 Recall | 2017 Recall | 2017 Chip Recall | 2017 FPR |
|-----------|---------------:|------------:|------------:|----------------:|---------:|
| C0_control | 0 | 0.7983 | 0.5725 | 0.5766 | 0.01063 |
| C1_2014chip | 0 | 0.5882 | 0.3765 | 0.3871 | 0.00549 |
| C2_2015chip | 301 | 0.8650 | 0.3333 | 0.3226 | 0.01250 |

## Decision: **NO_ROBUST_WIN**

### Hypothesis: H2_CHIP_SUPERVISION_DOES_NOT_HELP

- C2 vs C0 recall delta: -0.2392
- C2 vs C0 chip recall delta: -0.2540
- 2016 regression: +0.0668

## Channel Metrics (2017)

| Candidate | Overall Recall | Chip Recall | Online Recall | Swipe Recall |
|-----------|---------------:|------------:|--------------:|-------------:|
| C0_control | 0.5725 | 0.5766 | 0.0000 | 0.4286 |
| C1_2014chip | 0.3765 | 0.3871 | 0.0000 | 0.0000 |
| C2_2015chip | 0.3333 | 0.3226 | 0.0000 | 0.7143 |

## Key Finding

Chip-supervision candidates did not improve 2017 recall over the control. The dominant failure mechanism is likely deeper distribution/concept shift that cannot be addressed by adding historical chip-fraud examples alone.
