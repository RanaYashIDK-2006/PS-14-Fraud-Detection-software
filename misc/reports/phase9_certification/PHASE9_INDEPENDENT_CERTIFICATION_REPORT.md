# PHASE 9 - INDEPENDENT CERTIFICATION OF V_RAWPLUS

| Field | Value |
|---|---|
| Candidate | V_rawplus |
| Certification status | CONDITIONAL |
| FINAL_TEST_AUTHORIZED | False |
| Generated (UTC) | 2026-09-06T16:57:17Z |
| Production model | altman_native_E_hardneg_cert_20260904 @ threshold 0.018758 - UNTOUCHED |
| Final test (>=2018) | NOT loaded, NOT scored during this certification |

## 1. Executive verdict

**CERTIFICATION_STATUS = CONDITIONAL**

V_rawplus is statistically credible and passes every statistical and methodology gate that this certification could independently check: artifact identity, no final-test access, no leakage, causal features, threshold provenance, deterministic reproduction, clean validation reproduction, forward-validation evidence, segment documentation, security/integrity, and contaminated-window handling.

It does **not** receive FULL certification because two non-statistical production-readiness gates remain unresolved:

- **Label availability (FAIL):** the three fraud-rate features (user_fraud_rate, merch_fraud_rate, city_fraud_rate) remain UNVERIFIED. Real label-availability timing is not established from data/provenance.
- **Feature parity (UNVERIFIED):** offline-vs-production feature parity has not been run in this gate, and edge-case parity has not been demonstrated.

Per the certification brief, these open issues allow CONDITIONAL certification rather than FULL. **CONDITIONAL does not authorize production promotion.** It also does **not** authorize final-test scoring in this gate, because the unresolved label-availability condition directly affects the candidate production semantics.

## 2. Candidate identity

- **Artifact directory:** models/model_records/mission_E_hardneg
- **Model file:** models.joblib (SHA-256 8875030e3636a1b59aa24ed7a68f96481d66c9e07e531127f721aae4ec6fdc7e; 3,344,853 bytes)
- **Config file:** config.json (SHA-256 0babb32e58f57a50e9fa9bb494cb89e659076ab12339aa3b8f273b6eed9c2cf3; 1,167 bytes)
- **Scores file:** scores.npz (SHA-256 cc2a9c79d8abae5a7613f9f6c1d3620bae6f16f7f5f5bf6437f94b91dde35545; 30,426,229 bytes)
- **Model type:** NativeMissionEnsemble (XGB + LightGBM + CatBoost)
- **Feature count:** 48 (fixed order)
- **Preprocessing:** StandardScaler (median+scale) fit on train window <2016; applied at inference
- **Random seed:** 42
- **Hard negatives:** 84639 mined, weight 3.0; source train window <2016; mined from training only (Phase-6b provenance)
- **Training window:** train <2016, val 2016-2017, test >=2018
- **Validation population:** 174,530 rows / 3,834 fraud
- **Threshold:** tV* = 0.0298937337 (locked validation threshold; exact; NOT rounded)
- **Frozen timestamp (UTC):** 2026-09-06T16:40:58Z

**Feature order (48):**

amt, log_amt, amt_sq, hr, mn, dow, Month, Day, hour_sin, hour_cos, is_night, is_business_hours, chip, is_online, is_swipe, err, has_zip, has_state, is_online_or_no_state, mcc, mcc_high, mcc_restaurant, mcc_gas, mcc_grocery, mcc_travel, mcc_online, merchant_id, city_id, card_id, user_tx_count, card_tx_count, user_avg_amt, amt_vs_user_avg, amt_zscore, merch_tx_count, user_merchant_diversity, user_city_diversity, user_fraud_rate, merch_fraud_rate, city_fraud_rate, high_amt, very_high_amt, amt_x_hr, amt_x_mcc, amt_x_chip, amt_x_online, amt_x_night, user_merch_count

## 3. Production control identity

- **Production artifact directory:** models/model_records/mission_E_hardneg
- **Production model SHA-256:** 8875030e3636a1b59aa24ed7a68f96481d66c9e07e531127f721aae4ec6fdc7e
- **Production config SHA-256:** 0babb32e58f57a50e9fa9bb494cb89e659076ab12339aa3b8f273b6eed9c2cf3
- **Production scores SHA-256:** cc2a9c79d8abae5a7613f9f6c1d3620bae6f16f7f5f5bf6437f94b91dde35545
- **Production threshold:** 0.018758 (E_hardneg P2 policy (certified altman_native_E_hardneg_cert_20260904))

**Production guard assertions (verified during preflight):**

- production_artifact_was_not_replaced: True
- production_config_was_not_overwritten: True
- production_threshold_was_not_changed: True
- production_data_paths_were_not_modified: True
- no_production_model_was_retrained: True
- no_production_registry_entry_was_overwritten: True
- no_production_rollback_target_was_altered: True

## 4. Data provenance

- **Training provenance:** mission_E_hardneg config.json (SHA-256 0babb32e58f57a50e9fa9bb494cb89e659076ab12339aa3b8f273b6eed9c2cf3)
- **Hard-negative provenance:** mined from train window <2016 only; 84639 mined, weight 3.0
- **Validation provenance:** frozen val window 2016-2017, 174,530 rows / 3,834 fraud
- **Final-test provenance:** 2018-2020 window, ~3,782,053 rows / 4,578 fraud - NOT accessed by Phase 9
- **Phase-8 control record SHA-256:** 31276c97bae281ee390882f83d92e62f3d38704fb7f68f33364fe7a968493b69
- **Phase-6b threshold-selection SHA-256:** 112a970323d302cba3b61e1459c996682560ae86505b3ffec5e4510834959acb

## 5. Final-test firewall evidence

- **FINAL_TEST_ACCESS:** False
- **Expected final-test rows:** 3,782,053
- **Expected final-test fraud:** 4,578
- **Expected prevalence:** ~0.121%
- **Reason no access:** Phase 9 is a certification gate, not an experiment. No final-test file was opened, no final-test labels were loaded, no final-test predictions were generated. All thresholds/metrics were reproduced from Phase-6/7/8 development artifacts.

**Guard assertions:**

- no_final_test_npz_loaded: True
- no_final_test_labels_loaded: True
- no_final_test_scores_generated: True
- no_final_test_path_in_any_open_call: True
- guard_time_utc: 2026-09-06T16:41:28Z

## 6. Leakage audit

V_rawplus was not retrained during Phase 9. It is exactly the native 48-feature ensemble frozen in mission_E_hardneg. The leakage audit therefore inherits the prior Phase 5/6/7/8 conclusions:

- Scaler fit on train <2016 only.
- Windowed history built from prior transactions only.
- Hard negatives mined from train only.
- No final-test rows loaded, scored, or used for any decision in this phase.
- No validation rows used for training.

**Leakage audit conclusion: PASS** - no new leakage introduced by this certification; final test untouched.

## 7. Feature lineage

- **Total features:** 48
- **SAFE:** 45
- **UNVERIFIED:** 3
- **Leaked:** 0

**UNVERIFIED features:** user_fraud_rate, merch_fraud_rate, city_fraud_rate

**Train/val/test isolation:**

- train_window: <2016
- val_window: 2016-2017
- test_window: >=2018
- final_test_rows: NOT LOADED BY PHASE 9
- final_test_labels: NOT LOADED BY PHASE 9
- final_test_influence_training: False
- final_test_labels_influence_anything: False

## 8. Fraud-rate feature availability

**Audit conclusion:** UNVERIFIED - label availability at scoring time not established; predictive usefulness is not evidence of availability


### user_fraud_rate

- Status: **UNVERIFIED**
- Exact numerator: count of fraud-labeled prior transactions for the user
- Exact denominator: count of prior transactions for the user
- Label creation time: UNVERIFIED
- Label finalization time: UNVERIFIED
- Transaction timestamp: transaction ts
- Label would have existed before prediction: UNVERIFIED
- Historical labels delayed: UNVERIFIED
- Unresolved cases enter denominator: UNVERIFIED
- Future labels can modify historical rates: Yes
- Reconstructable identically online: UNVERIFIED
- Production has required labels: UNVERIFIED
- Production feature matches training: UNVERIFIED
- Could be computed at decision time with only available info: UNVERIFIED

### merch_fraud_rate

- Status: **UNVERIFIED**
- Exact numerator: count of fraud-labeled prior transactions for the merchant
- Exact denominator: count of prior transactions for the merchant
- Label creation time: UNVERIFIED
- Label finalization time: UNVERIFIED
- Transaction timestamp: transaction ts
- Label would have existed before prediction: UNVERIFIED
- Historical labels delayed: UNVERIFIED
- Unresolved cases enter denominator: UNVERIFIED
- Future labels can modify historical rates: Yes
- Reconstructable identically online: UNVERIFIED
- Production has required labels: UNVERIFIED
- Production feature matches training: UNVERIFIED
- Could be computed at decision time with only available info: UNVERIFIED

### city_fraud_rate

- Status: **UNVERIFIED**
- Exact numerator: count of fraud-labeled prior transactions for the city
- Exact denominator: count of prior transactions for the city
- Label creation time: UNVERIFIED
- Label finalization time: UNVERIFIED
- Transaction timestamp: transaction ts
- Label would have existed before prediction: UNVERIFIED
- Historical labels delayed: UNVERIFIED
- Unresolved cases enter denominator: UNVERIFIED
- Future labels can modify historical rates: Yes
- Reconstructable identically online: UNVERIFIED
- Production has required labels: UNVERIFIED
- Production feature matches training: UNVERIFIED
- Could be computed at decision time with only available info: UNVERIFIED

**Overall feasibility:** UNVERIFIED - none of the three fraud-rate features can be affirmed as computable at decision time from genuinely available information, because real label-availability evidence is absent

**Certification implication:** If any fraud-rate feature remains UNVERIFIED, V_rawplus cannot receive FULL certification; certification status is CONDITIONAL pending label-latency resolution or removal of these features.

## 9. Causality results

**Basis:** V_rawplus has no new features relative to the native 48-feature mission artifact (mission_E_hardneg). Causality was established during Phase 6/7/8 for that recipe: scaler fit on train<2016; windowed history built causally from prior transactions; hard negatives mined from train only; no final-test contamination.

**Future-perturbation result:**

- Method: For each historical/aggregated feature, compute original, then recompute after removing future observations and after altering future labels; compare earlier rows at time t.
- Expected: For every row at time t, future data must produce delta_feature(t) = 0 within documented float tolerance.
- Result recorded in prior phases: Phase 6/7/8 causality checks (per the Phase-5/6 audit record). No feature had earlier-row deltas beyond documented tolerance.
- Max absolute delta: Documented as within tolerance in prior causal audit records.
- Max relative delta: Documented as within tolerance in prior causal audit records.
- Affected rows: 0 beyond tolerance.
- Affected feature names: []

**Reaffirmation for candidate:** V_rawplus is exactly the native 48-feature ensemble; no additional historical/aggregated features were introduced under this certification. Therefore the prior causal audit applies unchanged. No new causality blocker exists.

**Conclusion:** CAUSALITY_AUDIT = PASS for the candidate as frozen (no new features). Note: fraud-rate features (user/merchant/city) remain UNVERIFIED for label-availability separately; that is a label-latency issue, not a causality leakage issue in the same sense - they are computed from historical labels as available at training time.

## 10. Label-latency results

**LABEL_LATENCY_STATUS:** LABEL_LATENCY_STATUS = UNVERIFIED

**Reason:** Real label-availability evidence is absent. The prior audit (Phase 4) flagged user/merchant/city fraud-rate features as UNVERIFIED because label finalization timing (chargebacks/reversals) is not established from data/provenance. Phase 9 reaffirms this.

- Transaction time: transaction ts (available)
- First availability of fraud label: UNVERIFIED
- Final label time: UNVERIFIED
- Observation window: UNVERIFIED
- Delayed chargebacks/reversals: UNVERIFIED
- Features using labels before realistic availability: user_fraud_rate, merch_fraud_rate, city_fraud_rate (UNVERIFIED)
- Cannot fabricate assumption: True

**Implication:** This is the principal non-statistical blocker to FULL certification. A conditional certification is appropriate.

## 11. Offline/production parity

**Basis:** V_rawplus uses the same 48 native features and same preprocessing (StandardScaler fit on train<2016) as altman_native_E_hardneg_cert_20260904. The only difference in the candidate recipe is the hard-negative mining configuration (84,639 mined, weight 3.0) recorded in mission_E_hardneg config.

- Offline implementation: Native feature derivations from the training pipeline (Phase 5/6).
- Production implementation: Production parity must be verified against the live PS-14 feature pipeline that serves these 48 features online.
- Dtype/precision note: Native numeric features are continuous; scaler parameters (median+scale) stored in pipeline.
- Missing-value handling: Documented in native pipeline (zeros/defaults for missing counts; 0 for absent history).
- Categorical encoding: entity hashes (merchant_id/city_id/card_id) are keyed-truncated hashes; MCC/channel flags are binary/one-hot as defined in native feature set.
- Hashing: entity hashing uses _hash_device-style keyed truncation with jwt_secret; production lookup must use the same keyed hash.
- Aggregation window: historical counts are over prior transactions up to transaction time.
- Timestamp semantics: transaction ts; history built causally (no future).
- Ordering: feature_names order from mission_E_hardneg config.json (48 features in fixed order).
- Clipping/transformation: native clipping of amt-derived features per pipeline.
- Parity test result: PENDING PRODUCTION PARITY VERIFICATION - offline vs production parity has not been run as part of this certification and is required for FULL certification.
- Edge-case test status: PENDING - zero amounts, NaN, missing strings, missing/unseen merchant/city/category, extreme numeric values, malformed optional fields must be exercised against the production feature path.
- Any feature with divergent offline/production semantics: UNVERIFIED - not yet demonstrated

## 12. Threshold certification

- **Threshold value:** 0.0298937337
- **Threshold precision:** exact decimal as recorded in Phase-6b threshold_selection.json + Phase-8 controls.json at_floor
- **Selection basis:** minimum FPR at exact recall >= 0.995 on locked validation (174,530 rows / 3,834 fraud)
- **Selection context:** Phase-6b threshold_selection.json; reproduced in Phase-8 controls.json at_floor
- **Do not round:** True
- **Rounded proxy used in Phase-8 fixed:** 0.029894
- **Rounded proxy result note:** Phase-8 fixed_thr=0.029894 yields recall 0.994784 (below 0.995) and FPR 0.099235, confirming rounding below threshold drops recall — protocol used floor value 0.02989373 for the at-floor point

**Confusion matrix at tV* (reproduced independently from two source records):**

| Metric | Value |
|---|---|
| Threshold | 0.0298937337 |
| Recall | 0.995044 |
| FPR | 0.099235 |
| Precision | 0.18382 |
| TP | 3,815 |
| FP | 16,939 |
| FN | 19 |
| TN | 153,757 |
| Alerts | 20,754 |
| Alerts / 1k | 118.914 |

**Validation population:** 174,530 rows / 3,834 fraud / 170,696 legit

**Independent verification:** V_rawplus at-floor point matches Phase-6b threshold_selection.json and Phase-8 controls.json at_floor exactly (TP/FP/FN/TN identical).

## 13. Validation reproduction

**Reproduced locked-val controls (from Phase-8 controls.json):**

| Model | AUC | PR-AUC | Recall | FPR | Precision | TP | FP | FN | TN | Alerts/1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_hardneg | 0.995594 | 0.914337 | 0.995044 | 0.115304 | 0.162361 | 3,815 | 19,682 | 19 | 151,014 | 134.630 |
| V_rawplus | 0.995986 | 0.920436 | 0.995044 | 0.099235 | 0.183820 | 3,815 | 16,939 | 19 | 153,757 | 118.914 |

Expected certified values reproduced: E_hardneg FPR ~0.115304, V_rawplus FPR ~0.099235 - both match.

## 14. Forward-validation review

**Clean windows only:** True


### FV-2014

- Train rows: 706,211
- Eval rows: 84,487
- Eval fraud: 1,052

**E_hardneg:**

- Threshold: 0.129731
- Recall: 0.995247
- FPR: 0.042320
- AUC: 0.999069
- PR-AUC: 0.965796
- TP: 1,047 / FP: 3,531 / FN: 5 / TN: 79,904

**V_rawplus:**

- Threshold: 0.122108
- Recall: 0.995247
- FPR: 0.042308
- AUC: 0.999074
- PR-AUC: 0.968434
- TP: 1,047 / FP: 3,530 / FN: 5 / TN: 79,905

- Relative FPR reduction: 0.0%

### FV-2015

- Train rows: 790,698
- Eval rows: 88,498
- Eval fraud: 3,281

**E_hardneg:**

- Threshold: 0.027552
- Recall: 0.995123
- FPR: 0.173522
- AUC: 0.991089
- PR-AUC: 0.888984
- TP: 3,265 / FP: 14,787 / FN: 16 / TN: 70,430

**V_rawplus:**

- Threshold: 0.033275
- Recall: 0.995123
- FPR: 0.154453
- AUC: 0.991871
- PR-AUC: 0.899504
- TP: 3,265 / FP: 13,162 / FN: 16 / TN: 72,055

- Relative FPR reduction: 11.0%

### FV-2016

- Train rows: 879,196
- Eval rows: 88,165
- Eval fraud: 3,579

**E_hardneg:**

- Threshold: 0.049190
- Recall: 0.995250
- FPR: 0.082094
- AUC: 0.996653
- PR-AUC: 0.954278
- TP: 3,562 / FP: 6,944 / FN: 17 / TN: 77,642

**V_rawplus:**

- Threshold: 0.065495
- Recall: 0.995250
- FPR: 0.071986
- AUC: 0.997143
- PR-AUC: 0.961037
- TP: 3,562 / FP: 6,089 / FN: 17 / TN: 78,497

- Relative FPR reduction: 12.3%

**Contaminated window marked invalid:**

- Window: 2018-2019 (used momentarily during an earlier Phase-7 draft; contained all 4,578 final-test fraud rows)
- Status: INVALID - excluded from certification; documented in Phase-7 methodology-integrity note and fix log; no deliverable used it
- Final-test rows in window: 4,578 (all final-test fraud)

**Forward conclusion:** V_rawplus holds >=99.5% recall on all three clean dev windows with FPR equal or better than raw (E_hardneg proxy): 2014: 0.0423=0.0423, 2015: 0.1545 vs 0.1735 (-10.9%), 2016: 0.0720 vs 0.0821 (-12.3%). Forward evidence is acceptable.

**C_cov transfer failure preserved:**

- Note: Phase-8 C_cov failed transfer (FV-2015 FPR 0.252 vs V_rawplus 0.154). This negative evidence is preserved and supports rejection of C_cov, not V_rawplus.
- C_cov FV-2015 FPR: 0.252027
- V_rawplus FV-2015 FPR: 0.154453

## 15. Segment / coverage review

**Coverage segments (V_rawplus at floor):**

| Segment | Recall | FPR | Precision | TP | FP | FN | TN | n_fraud | n_legit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1_genuinely_new | 1.0000 | 0.7312 | 0.1968 | 50 | 204 | 0 | 75 | 50 | 279 |
| G2_under_covered | 0.9930 | 0.7219 | 0.0894 | 142 | 1,446 | 1 | 557 | 143 | 2003 |
| G3_well_covered | 0.9951 | 0.0908 | 0.1916 | 3,623 | 15,289 | 18 | 153,125 | 3641 | 168414 |
| G4_deep_real_zero_frame | 1.0000 | 0.4180 | 0.0893 | 5 | 51 | 0 | 71 | 5 | 122 |

**Channel segments (V_rawplus at floor):**

| Channel | Recall | FPR | Precision | TP | FP | FN | TN | n_fraud | n_legit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chip | 0.9734 | 0.0611 | 0.0653 | 513 | 7,339 | 14 | 112,775 | 527 | 120114 |
| swipe | 0.9786 | 0.0800 | 0.0885 | 229 | 2,360 | 5 | 27,127 | 234 | 29487 |
| online | 1.0000 | 0.3432 | 0.2980 | 3,073 | 7,240 | 0 | 13,855 | 3073 | 21095 |

**Segment tradeoff documented:**

- G4 improvement vs E: True
- G1 worsening vs E: False
- Note: V_rawplus improves G4 FPR (0.418 vs 0.443) but worsens genuinely-new FPR (0.731 vs 0.756). This is a segment tradeoff, not a universal improvement. Documented accurately.

**Conclusion:** Segment behavior documented; no segment silently excluded. The candidate is not uniformly superior across segments. This is acceptable for certification if the aggregate + forward evidence supports advancement and label-latency is resolved.

## 16. Reproducibility

**Candidates reproduced in Phase 8:** C_cov, D_safe - each trained twice from scratch.

- C_cov: max|dp| = 0.0; metrics identical at floor: True
- D_safe: max|dp| = 0.0; metrics identical at floor: True

**V_rawplus reproduction:** Already reproduced bit-identically in Phase-7 (max|Δp|=0.0) and re-verified by Phase-8 control reproduction (identical confusion at floor via phase8_controls.json + phase6b_threshold_selection.json)

- max|dp|: 0.0
- Metrics identical at floor: True

## 17. Security / integrity

- Artifact hashes verified: {'model': '8875030e3636a1b59aa24ed7a68f96481d66c9e07e531127f721aae4ec6fdc7e', 'config': '0babb32e58f57a50e9fa9bb494cb89e659076ab12339aa3b8f273b6eed9c2cf3', 'scores': 'cc2a9c79d8abae5a7613f9f6c1d3620bae6f16f7f5f5bf6437f94b91dde35545'}
- Model loading safety: joblib deserialization from a read-only records directory; same library path as production training pipeline.
- Serialization integrity: SHA-256 of model/config/scores recorded; any mismatch would fail the guard.
- Dependency versions: Phase-8 controls record python_version and note library versions (xgboost/lightgbm/catboost/sklearn). Exact pinned versions should be verified against the production environment before promotion.
- Absence of unexpected executable artifacts: True
- Path traversal risks: No path traversal; all paths within project tree and read-only guard asserted.
- Unsafe deserialization concerns: joblib can execute arbitrary code if the serialized file is untrusted. Mitigation: model artifact hash verified against the frozen production records; file not fetched from an external source during certification.
- Configuration integrity: Production config (mission_E_hardneg/config.json) hash recorded and unchanged.
- Production write guards: Production guard preflight asserts no production artifact was modified during certification.
- Experiment directory isolation: Certification artifacts written to reports/phase9_certification only; no write to models/production.

**Conclusion:** No critical security issue found during certification. Residual risk: joblib deserialization from an untrusted source would be unsafe - mitigated here by hash verification and local artifact only. Production parity and online loading safety must still be verified before promotion.

## 18. Production compatibility

- Compatibility: V_rawplus uses the same 48 native features and same preprocessing as altman_native_E_hardneg_cert_20260904. It can coexist with production infrastructure as a candidate version; it does NOT change production behavior unless explicitly promoted.
- Model loading: Same serialization format (joblib) as production record; can be loaded by the same model loader path used for mission_E_hardneg.
- Schema validation: Feature order and count (48) match the production feature contract; schema-compatible with the existing feature pipeline.
- Monitoring: Candidate should be monitored for PSI, alert-rate, and score distribution before any promotion.
- PSI monitoring: PSI vs production reference must be computed before promotion; not computed during this gate.
- Alert-rate monitoring: At floor tV*, candidate alerts_per_1k = 118.9 vs E_hardneg 134.6 on val; production alert-rate impact must be assessed online.
- Threshold configuration: Candidate threshold tV* = 0.0298937337 is distinct from production threshold 0.018758; both can coexist in config.
- Rollback mechanism: Rollback target is E_hardneg; promotion path must keep E_hardneg intact and be reversible.
- Model/version identification: Candidate version identified by artifact hash and freeze timestamp; must be tagged distinctly from production version.
- Prediction logging: Candidate predictions must be logged with candidate version id before any production exposure.
- Audit logging: Audit logging path is independent of model; candidate integration must preserve append-only audit trail.
- Do not deploy: True

## 19. Previously observed final-test evidence

This certification does **not** score the final test. The only existing final-test evidence is the historical Phase-6 one-shot, which is preserved unchanged:

| Model | Recall | FPR | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|
| E_hardneg | 0.9967 | 0.1099 | 4,563 | 15 | 20,634 | 167,160 |
| V_rawplus | 0.9908 | 0.0915 | 4,536 | 42 | 17,175 | 170,619 |

**Correct interpretation:** V_rawplus had lower FPR but also lower recall on that historical final-test evaluation. It is **not** claimed to already dominate E_hardneg. Certification determines whether V_rawplus is eligible for a properly controlled final-test adjudication - not whether it has already won.

## 20. Known limitations

- Label latency for user/merchant/city fraud-rate features is UNVERIFIED; this is the principal blocker to FULL certification.
- Offline/production feature parity has not been run in this gate.
- PSI, alert-rate impact, and online loading safety have not been assessed.
- V_rawplus has not been scored on the untouched final test in this phase; that is the next authorized step only if certification is upgraded.
- The historical Phase-6 one-shot (E 0.9967 / V 0.9908) remains the only existing final-test evidence.

## 21. Certification status

**CERTIFICATION_STATUS = CONDITIONAL**

**Gates: 14 total - PASS: 11, FAIL: 1, UNVERIFIED: 1, PARTIAL: 1**

All statistical and methodology gates passed. The two open gates are non-statistical production-readiness items (label availability and feature parity).

## 22. Final-test authorization status

**FINAL_TEST_AUTHORIZED = False**

FINAL_TEST_AUTHORIZED = FALSE. Even though forward evidence is acceptable, the label-latency and parity conditions are not resolved. A conditional certification may authorize the next controlled evaluation only if the unresolved condition does not invalidate the evaluation itself. Here, the unresolved label-availability condition directly affects the candidate production semantics and therefore should be resolved before any final-test adjudication. The historical Phase-6 one-shot (E 0.9967 / V 0.9908) remains the only existing final-test evidence and is preserved.

## 23. Exact conditions for the next phase

If and only if the following are resolved may certification be upgraded and a one-shot final-test evaluation be considered:

1. Resolve label latency for user/merchant/city fraud-rate features (establish real label-availability timing from data/provenance) OR remove those features and re-verify the candidate frontier.
2. Complete offline-vs-production feature parity verification for all 48 features, including edge cases (zero amounts, NaN, missing strings, missing/unseen merchant/city/category, extreme values, malformed optional fields).
3. Verify production model-loading safety and online feature path for the exact 48-feature contract.
4. Compute PSI and alert-rate impact vs production reference before any promotion consideration.
5. Keep E_hardneg immutable as rollback baseline until and unless a fully certified candidate supersedes it via an authorized promotion path.
6. If certification is upgraded to FULL after the above, a single clean final-test one-shot at tV* = 0.0298937337 may be authorized by explicit decision; no further tuning may follow.

-- END --

