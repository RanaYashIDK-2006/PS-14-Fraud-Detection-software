# Per-rule severity tuning (section 6)

Constrained search over individual rule severities, objective: validation
cost with C_FN / C_FP = 20.0. Greedy coordinate descent
over per-rule multipliers bounded to [0.5, 1.5]
(step 0.1); ties keep the higher multiplier (preserves fraud
coverage), then multipliers that do not strictly reduce the validation
cost are reverted (6 neutral change(s) pruned) so only
cost-proven rebalances are recommended. `severity_scale` stays at
0.45. Metrics below are on the held-out **test** split
(52 fraud / 1412 legit).

| rule | base severity | multiplier | new severity |
|---|---|---|---|
| RULE_AMOUNT_HIGH | 0.300 | 0.50 | 0.150 |
| RULE_AMOUNT_EXTREME | 0.550 | 0.90 | 0.495 |
| RULE_NEW_DEVICE | 0.350 | 0.60 | 0.210 |
| RULE_UNUSUAL_TIME | 0.200 | 1.00 | 0.200 |
| RULE_UNUSUAL_LOCATION | 0.300 | 0.50 | 0.150 |
| RULE_UNUSUAL_RECIPIENT | 0.300 | 1.00 | 0.300 |
| RULE_AUTH_FAILURES | 0.450 | 1.00 | 0.450 |
| RULE_HIGH_FREQUENCY | 0.350 | 1.00 | 0.350 |
| RULE_ESCALATION | 0.700 | 0.70 | 0.490 |
| RULE_NEW_ACCOUNT_LARGE | 0.600 | 1.00 | 0.600 |
| RULE_CNP_SIGNATURE | 0.550 | 1.00 | 0.550 |
| RULE_ATO_SIGNATURE | 0.900 | 1.00 | 0.900 |

| metric | before (mult = 1) | after (rebalanced) |
|---|---|---|
| legit events challenged (verify) | 2 (0.14%) | 1 (0.07%) |
| fraud events caught (verify) | 48 (92.3%) | 48 (92.3%) |
| fraud allowed outright (low) | 3 (5.8%) | 3 (5.8%) |
| total cost | 71.35 | 67.50 |

Validation cost trace (greedy passes):

| pass | val cost |
|---|---|
| 0 | 6.20 |
| 1 | 3.35 |
| 2 | 2.90 |
| 3 | 2.90 |

**Recommended per-rule severities** (multiplier x base; set directly in
`src/risk_engine/rules.yaml`, `severity_scale` unchanged at 0.45):
RULE_AMOUNT_HIGH 0.150, RULE_AMOUNT_EXTREME 0.495, RULE_NEW_DEVICE 0.210, RULE_UNUSUAL_TIME 0.200, RULE_UNUSUAL_LOCATION 0.150, RULE_UNUSUAL_RECIPIENT 0.300, RULE_AUTH_FAILURES 0.450, RULE_HIGH_FREQUENCY 0.350, RULE_ESCALATION 0.490, RULE_NEW_ACCOUNT_LARGE 0.600, RULE_CNP_SIGNATURE 0.550, RULE_ATO_SIGNATURE 0.900
