# Federated heterogeneity map (section 19)

Same population (15000 events, 1.5% fraud, seed 7), same rounds (20) x local epochs (5), LR. Each config RE-ASSIGNS accounts to the 3 institutions with different size weights and fraud-rate skews (`split_institutions_skewed`); all metrics are macro PR-AUC on each institution's own held-out test. FL gain = fed - local (positive: FL helps overall); max drag = largest local_i - fed_i (positive: the global model is worse for that institution than its own local model).

| config | realized sizes | realized fraud rates | local | fed | oracle | FL gain | max drag (at) |
|---|---|---|---|---|---|---|---|
| balanced | 45%/34%/20% | 1.76%/1.22%/2.34% | 0.9643 | 0.9684 | 0.9649 | +0.0041 | -0.0011 (h0_0) |
| size-skew only | 87%/8%/5% | 1.66%/1.96%/1.85% | 0.8907 | 0.9875 | 0.9866 | +0.0968 | +0.0000 (h1_2) |
| fraud-skew only | 47%/31%/22% | 0.78%/1.08%/4.49% | 0.9276 | 0.9665 | 0.9576 | +0.0389 | -0.0015 (h2_2) |
| clean dominant | 83%/10%/7% | 0.83%/1.17%/12.41% | 0.8519 | 0.9947 | 0.9896 | +0.1428 | +0.0076 (h3_0) |
| fraud dominant | 90%/8%/3% | 1.78%/1.06%/0.51% | 0.9802 | 0.9822 | 0.9808 | +0.0020 (2/3) | +0.0000 (h4_1) |

## Per-config detail (PR-AUC on each institution's own test)

### balanced
| institution | local | fed | drag (local - fed) |
|---|---|---|---|
| h0_0 | 0.9402 | 0.9413 | -0.0011 |
| h0_1 | 0.9763 | 0.9810 | -0.0047 |
| h0_2 | 0.9763 | 0.9829 | -0.0065 |

### size-skew only
| institution | local | fed | drag (local - fed) |
|---|---|---|---|
| h1_0 | 0.9554 | 0.9624 | -0.0070 |
| h1_1 | 0.7167 | 1.0000 | -0.2833 |
| h1_2 | 1.0000 | 1.0000 | +0.0000 |

### fraud-skew only
| institution | local | fed | drag (local - fed) |
|---|---|---|---|
| h2_0 | 0.9619 | 0.9889 | -0.0270 |
| h2_1 | 0.8511 | 0.9396 | -0.0885 |
| h2_2 | 0.9697 | 0.9712 | -0.0015 |

### clean dominant
| institution | local | fed | drag (local - fed) |
|---|---|---|---|
| h3_0 | 0.9928 | 0.9852 | +0.0076 |
| h3_1 | 0.5727 | 1.0000 | -0.4273 |
| h3_2 | 0.9901 | 0.9988 | -0.0087 |

### fraud dominant
| institution | local | fed | drag (local - fed) |
|---|---|---|---|
| h4_0 | 0.9604 | 0.9645 | -0.0040 |
| h4_1 | 1.0000 | 1.0000 | +0.0000 |
| h4_2 | n/a (no fraud in test) | n/a | n/a |

## Reading the map
- **FL helps most at clean dominant** (FL gain +0.1428): small/noisy institutions whose local models are poor borrow the population signal through the average - the size-skew and clean-dominant rows both show small institutions jumping from local PR-AUC ~0.57-0.72 to ~1.00 under FedAvg.
- **Weakest case: fraud dominant** (FL gain +0.0020): when the data-dominant institution already holds the fraud signal, FL has nothing to add - the dominant's local model is already strong, and the small institutions' tests are too small to measure (one had no fraud in its test window at all).
- A positive max drag is the 'dominant institution drags the global model down' signature: size-weighted averaging dilutes an institution's own signal below what its local model alone achieves on its own test data. It appears exactly where expected - at the clean dominant's own test (+0.0076) its locally-learned clean patterns are marginally diluted by the fraud-heavy institutions' weights - and it is small here because the synthetic patterns are shared and cleanly separable (a real deployment with distribution shift between institutions would show a larger drag).
- Size-weighted FedAvg means a data-dominant institution's optimum dominates the average: when it is signal-poor (clean dominant), the fraud signal is diluted for everyone; when it holds the fraud, the small institutions inherit a strong model for free. The asymmetry is the map: FL is a redistribution of signal from where data is abundant to where it is scarce - most valuable when the abundant institution is NOT the one holding the fraud.