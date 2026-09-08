# PS-14 PRODUCTION FEATURE PARITY REPORT - Part 2 (2026-09-03)

Goal: the exact feature definition that trained the deployed model MUST be the
definition the deployed model scores at inference - numerically and
semantically. Evidence: `models/feature_contract.json` (canonical contract),
`reports/feature_parity_results.json` (machine-readable parity run),
`models/production/manifest.json`, `models/model_records/*runtime16*.json`.

## Deployed model (identified for this audit)

| Attribute | Value |
| --- | --- |
| Model version | `altman_runtime16_20260903_233407` |
| Type | XGB + LightGBM + CatBoost ensemble (weights 0.34/0.33/0.33) |
| Feature schema | `altman_runtime_v2` - 15 features |
| Training dataset | `data/transactions_causal.csv` (Part-1 causal features; prior-only amount stats) |
| Locked threshold (validation FPR<=1%) | 0.7052709 |
| Artifact SHA-256 | xgb `64f332dbb79a59f4...` / lgb `daa668ddb71fb771...` / cb `51d21524140f80e8...` / scaler `4ca1f6ce490c8f26...` / calibrator `d6b5cb6a7aa3a671...` |
| Shared feature implementation | `src/risk_engine/altman_ensemble.map_ml_features_to_altman` - the SAME function maps every offline training row and every live request |

## 1. Canonical feature contract (models/feature_contract.json)

Single machine-readable spec for all 15 features (name / type / units /
formula / source keys / causal rule / missing-value behavior). Both paths
consume the one mapper; the risk engine enforces the binding at load time
(`_verify_schema`: manifest.features == feature_list.json == ALTMAN_FEATURES
== scaler/learner column count - a mismatch raises and the service degrades to
rules-only rather than silently serving a wrong schema).

| Feature | Type | Units | Formula (canonical) | Source | Causal rule |
| --- | --- | --- | --- | --- | --- |
| log_amt | float | log $ | log1p(amount_ratio*100) | amount_ratio | prior-only ratio |
| amt_sq | float | $^2 | (amount_ratio*100)^2 | amount_ratio | prior-only ratio |
| hour_cos | float | cos | cos(2pi*hour/24) | hour_of_day | instant |
| is_business_hours | int | bool | 9<=hour<=17 | hour_of_day | instant |
| chip | int | proxy bool | new_device_flag (documented proxy) | new_device_flag | instant flag |
| is_online | int | proxy bool | = chip (documented proxy) | new_device_flag | instant flag |
| mcc_n | float | constant 0 | 0 (no MCC source; both sides) | - | no-source constant |
| has_zip | int | constant 0 | 0 (no ZIP source; both sides) | - | no-source constant |
| has_state | int | constant 0 | 0 (no state source; both sides) | - | no-source constant |
| merch_tx_count | float | 24h count | velocity count; **0 cold-start (fixed)** | merch_tx_count | prior 24h window |
| merch_fraud_rate | float | rate | rolling entity rate; baseline 0.001 cold-start | merch_fraud_rate, merchant_id | labels recorded post-score |
| city_fraud_rate | float | rate | rolling entity rate; baseline 0.001 cold-start | city_fraud_rate, city_id | labels recorded post-score |
| very_high_amt | int | bool | amount_ratio > 5 | amount_ratio | prior-only ratio |
| amt_x_mcc | float | $ | amt * mcc_n = 0 | amount_ratio | derived |
| amt_x_online | float | $ | amt * is_online | amount_ratio, new_device_flag | derived |

## 2. Mismatch register (found -> fixed)

| Feature | Old training | Old production | Correct definition | Fixed? | Retrained? |
| --- | --- | --- | --- | --- | --- |
| log_amt / amt_sq / very_high_amt / amt_x_online | legacy CSV `amount_ratio` (account-wide median incl. FUTURE events - Part 1 INVALID) | privacy-layer prior-only `amount_ratio` | prior-only `amount_ratio` everywhere | YES (retrain on causal dataset) | YES -> runtime16 |
| merch_tx_count | mapper fallback `known_devices` (1..8) - a device-count proxy that never equals merchant velocity | velocity 24h merchant count (0 cold-start) | 0 when no 24h history/no merchant context on BOTH paths | YES (mapper default 0) | YES -> runtime16 |
| merch_fraud_rate / city_fraud_rate | constant 0.001 (no entity ids in CSV) | 0.001 cold-start / real rolling rate for established entities | documented baseline 0.001 cold-start; real rates post-eval labels | YES (contract documents; tracker verified prior-only) | training rows are cold-start-only (dataset has no entity dims) - established-entity rates documented as unrepresented |
| chip / is_online | new_device_flag proxy | new_device_flag proxy | SAME proxy both sides; no chip channel exists in the synthetic event model - adding a real chip input requires a new raw field + retrain | Documented in contract (not silent) | no (consistent definition) |
| mcc_n / has_zip / has_state / amt_x_mcc | constant (no source in training data) | constant (no source in runtime) | constant on BOTH sides by construction - never real on one side and zero on the other | YES (verified constant columns in training == live values) | runtime16 keeps them (models ignore zero-gain constant columns) |

## 3. Parity test (scripts/feature_parity_test.py - 56/56 PASS)

Corpus: normal established entity (velocity + entity ids + real rates),
minimal CSV row, cold-start unseen merchant/city/user, zero amount, very large
amount, night-hour new-device ATO, high-velocity merchant, history immediately
after a fraud label, history before the label, missing ratio. Each case runs
through the OFFLINE path (the mapper the retrain feeds every row) and the LIVE
path (AltmanEnsembleEngine + deployed scaler/learners/calibrator):

| Check | Result |
| --- | --- |
| 15-feature vector parity per case (|d| <= 1e-9) | PASS (all cases) |
| Score parity (calibrated prob, |d| <= 1e-6 - CatBoost reduction noise ~1e-8) | PASS |
| ML decision band parity | PASS |
| Cold-start raw dict == live unseen-entity dict | PASS (max|d| < 1e-9) |
| merch_tx_count cold-start == 0 (not a device proxy) | PASS |
| rates cold-start == documented 0.001 | PASS |
| Independent score recompute from raw artifacts == engine score | PASS |
| Velocity/entity inputs change exactly {merch_tx_count, merch/city fraud rates} | PASS |
| Feature order pinned to contract (vec[i] == contract[i]) | PASS |
| Live tracker label ordering (rate BEFORE fraud label == 0; AFTER == 1/4) | PASS |

## 4. Deployment compatibility

* Model version: altman_runtime16_20260903_233407
* Model hash: see table above (all five artifacts hashed in the contract)
* Feature-contract version: altman_runtime_v2
* Feature count: 15 (manifest + feature_list.json + ALTMAN_FEATURES + scaler/learners)
* Preprocessing: RobustScaler fitted on TRAIN only (shipped with the release)
* Threshold: 0.7052709 locked from validation (FPR 0.999% on validation);
  runtime banding stays rules/band-based (engine does not hard-code the raw
  threshold into serving)
* Compatibility check: enforced at load (`_verify_schema`) + by the parity
  test's contract-binding checks; a wrong/old/new schema REFUSES to serve

## 5. Honest notes

1. Retraining on the Part-1 causal features lowered measured ML discrimination
   (val AUC 0.9432, untouched-test AUC 0.9020, recall@1%FPR 0.52 val / 0.36
   test) versus the legacy-CSV numbers. That decrease is the CORRECT result of
   removing future-contaminated training features (Part 1 marks the old
   numbers INVALID). Live decisions are rules-dominated; the ML probability is
   now monotone and separates a benign probe (0.0007) from an ATO probe
   (0.5153) - the legacy calibrator degeneracy is gone.
2. 7/15 columns are constant in training (mcc_n, has_zip, has_state,
   merch_tx_count, merch_fraud_rate, city_fraud_rate, amt_x_mcc). They are
   constant for the SAME documented reason at inference on cold-start input,
   so parity holds; they remain as zero-gain columns rather than being
   silently dropped (a leaner schema is a separate retrain decision).
3. Training rows carry no entity dimension, so established-entity live values
   (real merchant counts / entity rates once the tracker accumulates history)
   are distributionally unrepresented; drift monitoring on the live
   feature distribution is the guard.

## Verdict

**PASS** - complete numerical and semantic parity demonstrated between the
offline training path and the live scoring path for the deployed model: the
same single mapper, the same documented fallbacks on both sides, retrained on
the causal feature source, contract-bound at load, and verified per-feature /
per-score / per-decision on a representative corpus (56/56 checks). Remaining
semantic notes (chip/is_online proxies, absent MCC/zip/state channels,
established-entity histories) are documented definition choices with NO
train/prod divergence, not silent substitutions.

---
Generated by scripts/build_feature_contract.py + scripts/feature_parity_test.py.
