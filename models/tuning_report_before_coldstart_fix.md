# Risk Engine operating-point tuning (section 6)

Cost-weighted objective with C_FN / C_FP = 20.0
(step-up: fraud 0.25x, legit 0.15x). Scanned `severity_scale` on the
**validation** split. Selection rule: eliminate legit challenges
(legit_high == 0), then pick the HIGHEST scale among those (closest to the
original calibration, so fraud coverage is maximally preserved). Metrics
below are on the held-out **test** split
(44 fraud / 1456 legit).

| metric | scale 1.0 (before) | scale 0.55 (chosen) |
|---|---|---|
| legit events challenged (verify) | 21 (1.44%) | 0 (0.00%) |
| fraud events caught (verify) | 39 (88.6%) | 38 (86.4%) |
| fraud allowed outright (low) | 1 (2.3%) | 2 (4.5%) |
| total cost | 87.40 | 75.90 |

**Recommended `severity_scale`: 0.55** (set in
`src/risk_engine/rules.yaml`; critical rules still floor at
`severity_floor`, so account-takeover signals are unaffected).

Validation cost table (test counts out of 44 fraud / 1456 legit):

| scale | val cost | test cost | legit high | fraud high (recall) | fraud low (missed) |
|---|---|---|---|---|---|
| 0.45 | 7.10 | 93.15 | 0 | 38 (86.4%) | 4 |
| 0.50 | 13.10 | 67.35 | 0 | 38 (86.4%) | 2 |
| 0.55 | 22.45 | 75.90 | 0 | 38 (86.4%) | 2 |
| 0.60 | 26.05 | 78.75 | 3 | 38 (86.4%) | 2 |
| 0.65 | 28.45 | 81.80 | 5 | 38 (86.4%) | 2 |
| 0.70 | 28.90 | 81.95 | 5 | 38 (86.4%) | 2 |
| 0.75 | 36.55 | 83.75 | 13 | 39 (88.6%) | 2 |
| 0.80 | 37.40 | 85.45 | 15 | 39 (88.6%) | 2 |
| 0.85 | 48.45 | 88.85 | 19 | 39 (88.6%) | 2 |
| 0.90 | 64.60 | 86.55 | 20 | 39 (88.6%) | 1 |
| 0.95 | 65.45 | 87.40 | 21 | 39 (88.6%) | 1 |
| 1.00 | 65.45 | 87.40 | 21 | 39 (88.6%) | 1 |
