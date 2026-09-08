# Federated MLP comparison (section 19)

Same account-disjoint shards, same rounds (20) x local epochs (5); LR lr=0.5, MLP lr=0.1, hidden=16. Per-institution PR-AUC:

| institution | LR local | LR fed | LR centr | MLP local | MLP fed | MLP centr |
|---|---|---|---|---|---|---|
| inst_0 | 0.9481 | 0.9600 | 0.9551 | 0.8899 | 0.9476 | 0.9141 |
| inst_1 | 0.9567 | 0.9548 | 0.9669 | 0.9403 | 0.9639 | 0.9337 |
| inst_2 | 0.9428 | 0.9723 | 0.9536 | 0.9138 | 0.9491 | 0.9458 |

| model (macro PR-AUC) | local-only | federated | centralized |
|---|---|---|---|
| logistic regression | 0.9492 | 0.9624 | 0.9585 |
| MLP (hidden=16) | 0.9147 | 0.9536 | 0.9312 |

FedAvg convergence (global param norm per round, sampled):

| round | LR | MLP |
|---|---|---|
| 0 | 1.9213 | 3.7299 |
| 2 | 2.7985 | 3.8600 |
| 4 | 3.4297 | 3.9960 |
| 6 | 3.9049 | 4.1139 |
| 8 | 4.2848 | 4.2147 |
| 10 | 4.6025 | 4.3017 |
| 12 | 4.8765 | 4.3775 |
| 14 | 5.1182 | 4.4447 |
| 16 | 5.3350 | 4.5053 |
| 18 | 5.5318 | 4.5605 |
| final | 5.6240 | 4.5865 |

## Reading the table
- The **separability benefit** is fed MLP minus fed LR PR-AUC (-0.0088 here). A positive gap means the nonlinear boundary separates fraud a linear model cannot; a gap near zero (or negative) means the features are already (nearly) linearly separable.
- The honest finding on THIS dataset: the 6 engineered per-event features linearize the fraud patterns (the AND-of-flags interactions like CNP testing / ATO become linear flag combinations in feature space), so the MLP's nonlinear capacity buys nothing and only adds parameters for FedAvg to average. The mechanism works — the MLP converges and tracks its own centralized bound — but the separability benefit needs raw inputs with genuinely nonlinear structure (e.g. high-cardinality categoricals or raw sequences) that the engineered features already absorb.
- The MLP's weight norm lives on a different scale than LR's (per-layer matrices vs a single weight vector), so the convergence rows are a trend, not an absolute comparison.
- The MLP is deliberately tiny (one hidden layer) and trained with plain batch GD — the point is the FL mechanics + the comparison, not SOTA.