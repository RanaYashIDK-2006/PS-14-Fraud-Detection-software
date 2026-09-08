# Federated learning simulation (section 19)

Simulated institutions: 3 disjoint account shards of one synthetic population (15000 events, 1.5% fraud, seed 7). FedAvg: 20 rounds x 5 local epochs, lr=0.5; baseline/oracle: 50 epochs. Local split 70%/30% time-based.

| institution | n_test (fraud) | model | PR-AUC | ROC-AUC | F1 | Recall |
|---|---|---|---|---|---|---|
| inst_0 | 2241 (49) | local-only | 0.9481 | 0.9979 | 0.786 | 0.939 |
| inst_0 | 2241 (49) | federated | 0.9600 | 0.9985 | 0.811 | 0.918 |
| inst_0 | 2241 (49) | centralized | 0.9551 | 0.9982 | 0.793 | 0.939 |
| inst_1 | 1279 (39) | local-only | 0.9567 | 0.9952 | 0.822 | 0.949 |
| inst_1 | 1279 (39) | federated | 0.9548 | 0.9969 | 0.864 | 0.897 |
| inst_1 | 1279 (39) | centralized | 0.9669 | 0.9971 | 0.822 | 0.949 |
| inst_2 | 920 (21) | local-only | 0.9428 | 0.9966 | 0.775 | 0.905 |
| inst_2 | 920 (21) | federated | 0.9723 | 0.9991 | 0.905 | 0.905 |
| inst_2 | 920 (21) | centralized | 0.9536 | 0.9978 | 0.844 | 0.905 |

| model (macro-avg) | PR-AUC | ROC-AUC | F1 | Recall |
|---|---|---|---|---|
| local-only | 0.9492 | 0.9966 | 0.795 | 0.931 |
| federated | 0.9624 | 0.9982 | 0.860 | 0.907 |
| centralized (bound) | 0.9585 | 0.9977 | 0.820 | 0.931 |

Convergence (global weight norm per round): 1.921, 2.799, 3.430, 3.905, 4.285, 4.603, 4.877, 5.118, 5.335, 5.532

## Honest caveats
- Naive FedAvg without differential privacy leaks information via the weight updates (gradient-inversion attacks); a real deployment adds DP noise + secure aggregation.
- Logistic regression only — RF/XGB/ISO are not weight-averageable the same way.
- Each institution standardizes features with its own statistics; the shared model lives in per-institution standardized space (no statistics are shared).
- The centralized oracle is a theoretical bound computed out-of-band — the very relaxation federated learning exists to avoid.
- Federated can match or slightly edge the centralized bound here: each institution runs R x local_epochs = 100 local epochs vs the oracle's 50, evaluation uses each institution's own scaler, and per-institution class balancing adapts to local prevalence. The margins are small and synthetic — treat 'federated >= oracle' as an implementation detail of this prototype, not a general claim.
- Synthetic data is cleanly separable; absolute scores are inflated. The comparison is about the FL mechanics, not expected production performance.