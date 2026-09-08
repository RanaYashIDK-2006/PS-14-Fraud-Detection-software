# PS-14 ML VALIDITY REPORT - GROUND-UP REBUILD (2026-09-03)

Leakage-free chronological protocol. Historical metrics from the legacy `transactions.csv` era are **INVALID** for the amount-history features (account-wide median included future transactions - proven by the perturbation audit) and are preserved only as prior evidence, not compared here.


## A. Dataset Integrity

| Property | Value |
| --- | --- |
| File | `data/transactions_causal.csv` |
| SHA-256 | `c81c26cff1b3f39d31b1ca08dd1047ee...` |
| Rows | 197,645 |
| Fraud / Legit | 2,354 / 195,291 (1.19%) |
| Timestamp range | 2025-06-01 02:01:29.297060013+00:00 .. 2025-12-20 07:05:35.691618204+00:00 |
| Duplicate rows | 0 |
| Chronological ordering | monotonic_non_decreasing (gate: PASS) |

## B. Split Definition (chronological, reproducible)

| Split | Rows | Fraud | Start (UTC) | End (UTC) |
| --- | ---: | ---: | --- | --- |
| train | 138,352 | 1,158 | 2025-06-01 02:01:29.297060013+00:00 | 2025-08-24 02:05:52.123246670+00:00 |
| validation | 29,646 | 219 | 2025-08-24 02:06:55.755968809+00:00 | 2025-09-11 08:17:07.315063477+00:00 |
| final_test | 29,647 | 977 | 2025-09-11 08:17:30.248820782+00:00 | 2025-12-20 07:05:35.691618204+00:00 |

Future-contamination gate: PASS (no later-split row predates an earlier split). Split reproducibility key: `bd87ea08759e84cc`.


## C. Feature Causality (perturbation audit, 10 scenarios x 4 perturbations)

Method: recompute the target row's features from history up to t; append strictly-future perturbed events (fraud burst / amount shock / label flip / extra volume); recompute. A causal feature must be bit-identical. The legacy derive is run on the same rows for disclosure.

| Feature | Historical? | Label-derived? | Future-sensitive (causal) | Causal? |
| --- | --- | --- | --- | --- |
| amount_ratio | yes (prior-only) | no | max|d|=0.0e+00 (legacy 0.9178) | PASS |
| account_tenure_days | yes (prior-only) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| amount_zscore | yes (prior-only) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| days_since_last_similar_txn | yes (prior-only) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| failed_auth_count_24h | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| gradual_escalation_score | yes (prior-only) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| hour_deviation | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| hour_of_day | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| is_weekend | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| known_device_count | yes (prior-only) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| mule_ring_score | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| new_device_flag | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| recipient_novelty | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| shared_device_accounts | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| shared_recipient_accounts | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| txn_freq_last_24h | yes (prior-only) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| txn_regularity | yes (prior-only) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| txn_time_unusual | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| unusual_location_flag | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| unusual_recipient_flag | no (instant) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |
| velocity_deviation | yes (prior-only) | no | max|d|=0.0e+00 (legacy 0.0000) | PASS |

Causality gate: **PASS**. No model feature is label-derived (0 suspect lines); label-latency cannot leak into features by construction and the label is used only as the outcome.


## D. Final Metrics (untouched final temporal test, locked threshold)

Locked threshold: **0.427603** (selected on VALIDATION only, highest score with validation FPR <= 1%). Test-side disclosure: a threshold fit to the test set would be 0.428522 - reported but **not used**.

| Metric | Final Temporal Test |
| --- | ---: |
| ROC-AUC | 0.998608 |
| PR-AUC | 0.974199 |
| FPR (exact, unrounded) | 0.010010464 (>1%) |
| Recall | 0.979529 |
| Precision | 0.769293 |
| Specificity | 0.989990 |
| TP | 957 |
| TN | 28383 |
| FP | 287 |
| FN | 20 |
| Alerts | 1244 (419.6 per 10k) |
| AUC 95% CI (paired bootstrap, 2000 iters, seed 42) | 0.998007 .. 0.999057 |

Bootstrap method: paired (y, score) row resampling; 2.5/97.5 percentiles; fixed seed. Resample prevalence 0.032936 +/- 0.001054 (original 0.032954).


## D2. Ablation (identical chronological protocol, each group re-trains with its own validation-only threshold)

| Group | Features | Val AUC | Test AUC | Test PR-AUC | Test FPR | Test Recall | Test Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_causal_21 | 21 | 0.9981 | 0.9986 | 0.9742 | 0.0100 | 0.9795 | 0.7693 |
| no_amount_history | 18 | 0.9994 | 0.9991 | 0.9772 | 0.0120 | 0.9908 | 0.7372 |
| instant_only | 6 | 0.9513 | 0.9340 | 0.4981 | 0.0102 | 0.4248 | 0.5870 |
| flags_only | 5 | 0.9467 | 0.9318 | 0.4592 | 0.0095 | 0.4197 | 0.6003 |

## E. Temporal Stability (4 chronological windows of the final test)

| Window | Rows | Fraud | Start | End | AUC | PR-AUC | FPR@lock | Recall@lock |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| window_1 | 7,411 | 93 | 2025-09-11 08:17:30.248820782+00:00 | 2025-09-17 15:08:17.507044791+00:00 | 0.9999 | 0.9933 | 0.0094 | 1.0000 |
| window_2 | 7,411 | 101 | 2025-09-17 15:09:47.922792673+00:00 | 2025-09-25 15:50:00.966405869+00:00 | 0.9999 | 0.9940 | 0.0105 | 1.0000 |
| window_3 | 7,411 | 99 | 2025-09-25 15:50:24.088515520+00:00 | 2025-10-08 01:50:24.668965816+00:00 | 0.9996 | 0.9827 | 0.0093 | 0.9899 |
| window_4 | 7,414 | 684 | 2025-10-08 01:51:41.349792957+00:00 | 2025-12-20 07:05:35.691618204+00:00 | 0.9980 | 0.9838 | 0.0108 | 0.9722 |

No 'no drift' claim is made without a formal test; variation is reported as measured above.


## F. Independent Validation

| Check | Result |
| --- | --- |
| Independently reproduced (separate code path) | YES |
| Independent rank-sum AUC | 0.9986083121 (recorded 0.9986083121) |
| Independent manual PR-AUC | 0.9741992578 (recorded 0.9741992578) |
| Confusion-matrix identity checks | 178 checks, 0 failed |
| Causality gate | PASS |
| Final-test contamination gates | scaler fit on TRAIN / early-stop on VALIDATION / threshold on VALIDATION - source-asserted |
| Reproducibility (same data/seed/params) | byte-identical protocol content + predictions npz across repeated executions (only the embedded built_at timestamp differs) |

## G. Verdict

**ML-VALIDITY VERDICT: PASS**

All critical validation requirements passed: dataset ordering proven monotonic; every historical feature is future-immune under perturbation (bit-identical); no feature is label-derived; final test is untouched (scaler/early stopping/threshold/calibration decisions use train+validation only); all confusion-matrix identities hold on every published table; threshold is locked from validation and reproduced independently; bootstrap CIs are consistent.

**Final-test integrity note:** the protocol was executed multiple times AFTER the model/threshold were locked (same data, seed, parameters) purely to prove determinism - the runs produced bit-identical content and NO decision was ever changed after seeing final-test results, so the final test remained untouched.

**Documented limitations (not gate failures):** the synthetic dataset reuses the same behavioural archetypes across the time split, so within-dataset temporal metrics overstate cross-population generalization (leave-one-archetype-out performance is materially lower, per prior audits); scores are ranking-optimised and were NOT probability-calibrated in this rebuild (calibration is a separate decision-quality task); threshold is FPR<=1% on validation, exact test FPR is 0.010010.

---
Generated by `scripts/independent_validator.py` on 2026-09-04T17:26:18.085912+00:00.
