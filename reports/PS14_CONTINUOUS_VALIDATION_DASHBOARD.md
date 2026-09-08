# PS-14 Continuous Validation Dashboard

**Generated:** 2026-09-03 06:51:51 UTC  |  
**Overall status:** RED  |  
**Deployment allowed:** False  |  
**Promotion:** BLOCKED

Status is computed from PS-14 validated evidence (Checks 11-19). Trigger bounds are 
PS-14 operational thresholds (recall >= 85%, FPR <= 1%, PR-AUC >= 0.60, PSI <= 0.25), 
not industry-standard claims.

| Dimension | Status | Detail |
|---|---|---|
| model_performance | GREEN | recall 91.6% >= 85%, FPR 0.80% <= 1%, PR-AUC 0.801 |
| feature_drift | YELLOW | 2 feature(s) WARNING, 0 CRITICAL |
| data_quality | RED | monitor says CRITICAL |
| model_integrity | GREEN | engine loaded + all artifact hashes and schema verified |
| production_parity | RED | Feature-generation parity is near-perfect (7/8 expanding features bit-exact vs the training cache); one REAL production bug: city_fraud_rate resets per parquet row group in the cache builder. The model mismatch remains: the forensic audit evaluated an inline 25-feature XGB, not the deployed 15-feature lean artifact. PRODUCTION PARITY = FAIL (model mismatch + city-rate cache bug). |
| probability_calibration | GREEN | test ECE = 0.3918% (within 1% PS-14 bound) |
| alert_volume | GREEN | latest window 90.5 alerts/10K |
| threshold_governance | RED | deployed model has no validated, registry-locked threshold |
| promotion_gate | RED | A critical gate failed: Data-quality validation; Feature-parity validation; Threshold validation |
| fail_safe_probes | GREEN | 3/3 live fail-safe probes passed (tamper/schema/missing all refused to load) |

### Automatic promotion gate

| Gate | Result | Evidence |
|---|---|---|
| Leakage validation | PASS | causality PASS on audit pipeline |
| Data-quality validation | FAIL | DQ=CRITICAL |
| Feature-parity validation | FAIL | parity bottom line (city_fraud_rate cache bug + model mismatch on record) |
| Threshold validation | FAIL | threshold locked in registry |
| Performance validation | PASS | latest window within bounds |
| Robustness validation | PASS | stress matrix 24/24 |
| Security validation | PASS | deployment-gate security PASS |
| Rollback validation | PASS | real rollback -> identical V1 predictions |
| Governance integrity | PASS | artifact hashes verified |

**Promotion verdict: BLOCKED**

**Reason:** A critical gate failed: Data-quality validation; Feature-parity validation; Threshold validation

### Fail-safe probes (live)

| Probe | Result | Detail |
|---|---|---|
| tampered_artifact | PASS | refused to load: XGBoostError |
| feature_schema_mismatch | PASS | refused to load: XGBoostError |
| missing_artifact | PASS | refused to load: XGBoostError |

### Open blocking items

1. **Production parity FAIL** - city_fraud_rate cache bug in 
   `train_altman_fullscale.py` + audited model != deployed model (Check 16).
2. **Threshold governance FAIL** - deployed model has no registry-locked threshold (Check 19).
3. **Data quality CRITICAL** - 0.51% invalid timestamps in recent window 
   (synthetic seed artifact; source must be fixed before retraining).
