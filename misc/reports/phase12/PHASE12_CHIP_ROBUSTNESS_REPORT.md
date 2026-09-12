# Phase 12 — Chip-Robustness Experiment Report

**Date:** 2026-09-07T15:47:14Z

## Comparison Table

| Model | 2016 Rec | 2016 FPR | 2017 Rec | 2017 FPR | Chip 2017 | Online 2017 | Status |
|-------|----------|----------|----------|----------|-----------|-------------|--------|
| A_p11_baseline | 0.7972 | 0.00760 | 0.3373 | 0.00882 | 0.3387 | 0.0000 | NOT_WINNER |
| B_channel_balanced | 0.7972 | 0.00802 | 0.1961 | 0.00899 | 0.2016 | 0.0000 | NOT_WINNER |
| C_chip_hardneg | 0.7972 | 0.00760 | 0.3373 | 0.00882 | 0.3387 | 0.0000 | NOT_WINNER |
| D_no_online_features | 0.8064 | 0.00850 | 0.5098 | 0.01104 | 0.5121 | 0.0000 | POTENTIAL_WINNER |
| E_balanced_combo | 0.7972 | 0.00802 | 0.1961 | 0.00899 | 0.2016 | 0.0000 | NOT_WINNER |

## Decision: **ROBUST_WIN**
Best candidate: D_no_online_features

## Key Finding

The 2016→2017 recall collapse is caused by channel composition shift (online→chip). Pre-2016 training data contains some chip fraud signal but the model cannot fully compensate for the dramatic channel transition.