# Federated DP privacy-utility trade-off (section 19)

Client-level DP FedAvg (clip norm S = 0.2728, delta = 1e-05, rounds = 20) with Gaussian noise calibrated by RDP composition. Macro-averaged across 3 institutions; the same account-disjoint shards are used for every epsilon, and each DP row is the mean over 4 noise draw(s) (+/- std on PR-AUC).

| epsilon | sigma | noise std | PR-AUC (+/- std) | ROC-AUC | F1 | Recall | PR-AUC vs clean |
|---|---|---|---|---|---|---|---|
| 0.5 | 34.290 | 9.356 | 0.0498 +/- 0.0298 | 0.5440 | 0.076 | 0.525 | -0.9126 |
| 1.0 | 18.092 | 4.936 | 0.0284 +/- 0.0052 | 0.4407 | 0.041 | 0.521 | -0.9339 |
| 2.0 | 9.614 | 2.623 | 0.1677 +/- 0.1783 | 0.6541 | 0.156 | 0.609 | -0.7947 |
| 4.0 | 5.178 | 1.413 | 0.1028 +/- 0.1259 | 0.4645 | 0.060 | 0.499 | -0.8595 |
| 8.0 | 2.854 | 0.779 | 0.3213 +/- 0.1673 | 0.6856 | 0.057 | 0.691 | -0.6411 |
| clean FedAvg | 0.000 | 0.000 | 0.9624 | 0.9982 | 0.860 | 0.907 | +0.0000 |

## Reading the table
- Smaller epsilon = stronger privacy = larger sigma = more noise = lower utility. The gap to the clean row is the price of the guarantee.
- sigma is the RDP-composed noise scale over all rounds (normalized to sensitivity 1); the applied per-round noise std is sigma x S.
- The headline finding: with only 3 institutions, the DP budget is shared across just a few contributors, so even a weak guarantee (epsilon = 8) costs about 2/3 of the clean PR-AUC (0.32 vs 0.96), and any epsilon below that leaves the model statistically indistinguishable from random (PR-AUC ~ prevalence, huge run-to-run std). This is the honest price of client-level DP at this federation size - more institutions (or a bigger per-institution budget) would shift the curve left.

## Honest caveats
- Central DP: the coordinator adds the noise; individual clipped updates still cross the boundary, so production pairs this with secure aggregation. Per-client DP-SGD at the workers (per-sample clipping) is the stronger local-DP variant and would need substantially more noise for the same epsilon.
- delta = 1e-5 with only 3 clients is weak in an absolute sense; production would use delta ~ 1/N_clients or smaller and a battle-tested accountant.
- The RDP conversion assumes the per-round aggregate mechanism has sensitivity S; the exact client-count factor is absorbed into the reported sigma (standard FedAvg-DP convention).
- Synthetic data is cleanly separable; absolute scores are inflated. The trade-off curve is about the DP mechanics, not expected production performance.