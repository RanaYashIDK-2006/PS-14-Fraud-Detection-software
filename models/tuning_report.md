# Risk Engine operating-point tuning (section 6)

Cost-weighted objective with C_FN / C_FP = 20.0
(step-up: fraud 0.25x, legit 0.15x). Scanned `severity_scale` on the
**validation** split. Selection rule: eliminate legit challenges
(legit_high == 0), then pick the HIGHEST scale among those (closest to the
original calibration, so fraud coverage is maximally preserved). Metrics
below are on the held-out **test** split
(147 fraud / 29853 legit).

| metric | scale 1.0 (before) | scale 1.00 (chosen) |
|---|---|---|
| legit events challenged (verify) | 29853 (100.00%) | 29853 (100.00%) |
| fraud events caught (verify) | 147 (100.0%) | 147 (100.0%) |
| fraud allowed outright (low) | 0 (0.0%) | 0 (0.0%) |
| total cost | 29853.00 | 29853.00 |

**Recommended `severity_scale`: 1.00** (set in
`src/risk_engine/rules.yaml`; critical rules still floor at
`severity_floor`, so account-takeover signals are unaffected).

Validation cost table (test counts out of 147 fraud / 29853 legit):

| scale | val cost | test cost | legit high | fraud high (recall) | fraud low (missed) |
|---|---|---|---|---|---|
| 0.45 | 29832.00 | 29858.00 | 29853 | 146 (99.3%) | 0 |
| 0.50 | 29832.00 | 29858.00 | 29853 | 146 (99.3%) | 0 |
| 0.55 | 29832.00 | 29858.00 | 29853 | 146 (99.3%) | 0 |
| 0.60 | 29832.00 | 29858.00 | 29853 | 146 (99.3%) | 0 |
| 0.65 | 29812.00 | 29853.00 | 29853 | 147 (100.0%) | 0 |
| 0.70 | 29812.00 | 29853.00 | 29853 | 147 (100.0%) | 0 |
| 0.75 | 29812.00 | 29853.00 | 29853 | 147 (100.0%) | 0 |
| 0.80 | 29812.00 | 29853.00 | 29853 | 147 (100.0%) | 0 |
| 0.85 | 29812.00 | 29853.00 | 29853 | 147 (100.0%) | 0 |
| 0.90 | 29812.00 | 29853.00 | 29853 | 147 (100.0%) | 0 |
| 0.95 | 29812.00 | 29853.00 | 29853 | 147 (100.0%) | 0 |
| 1.00 | 29812.00 | 29853.00 | 29853 | 147 (100.0%) | 0 |

## Out-of-distribution generalization (group-split eval)

From `src/train_compare.py` leave-one-archetype-out retraining: the ensemble is retrained with each fraud archetype EXCLUDED, so the recall below is on fraud patterns the model has never seen - the honest counterpart to the test-split table above. `novel legit` rows are the cold-start FPR on accounts held out entirely.

| held-out group | n fraud | recall (F1) | recall@1%FPR | OOD legit FPR | avg ml (fraud) |
|---|---|---|---|---|---|
| ato | 67 | 0.09 | 1.0 | nan | 0.0567 |
| cnp | 30 | 0.933 | 1.0 | nan | 0.8308 |
| escalation | 15 | 0.467 | 1.0 | nan | 0.4857 |
| impulse | 35 | 0.514 | 0.914 | nan | 0.3311 |
| mule | 34 | 0.559 | 1.0 | nan | 0.3366 |
| 40 novel legit accounts | 0 | nan | nan | 0.203 | nan |
