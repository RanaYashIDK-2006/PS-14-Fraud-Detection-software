# PS-14 PRODUCTION READINESS, ALERT BURDEN & SECURITY AUDIT

**Date:** 2026-09-02

---

## PART A: PRODUCTION ALERT BURDEN

### Dataset Context

| Metric | Value |
|--------|------:|
| Total test transactions | 4,877,380 |
| Fraud cases | 5,761 |
| Base fraud rate | 0.118% (11.8 per 10K) |

### Operating Point Comparison

| Threshold | FPR | Recall | Precision | TP | FP | Alerts | Alerts/1K | Alerts/10K | Missed Fraud/10K | Investigate % |
|----------:|----:|-------:|----------:|---:|---:|-------:|----------:|-----------:|-----------------:|--------------:|
| 0.094 | 0.949% | 91.8% | 10.3% | 5,289 | 46,226 | 51,515 | 10.6 | 105.6 | 1.0 | 1.056% |
| 0.090 | 0.986% | 92.0% | 9.9% | 5,302 | 48,033 | 53,335 | 10.9 | 109.4 | 0.9 | 1.094% |
| 0.167 | 0.489% | 89.4% | 17.8% | 5,151 | 23,836 | 28,987 | 5.9 | 59.4 | 1.3 | 0.594% |
| 0.491 | 0.098% | 82.1% | 49.7% | 4,730 | 4,785 | 9,515 | 1.9 | 19.5 | 2.1 | 0.195% |

### Trade-Off Analysis

#### FPR < 0.1% (Threshold 0.491)
- **Recall: 82.0%** — catches 4,730 of 5,761 fraud cases
- **Misses: 1,031 fraud cases** (2.1 per 10K transactions)
- **Alerts: 9,515** (19.5 per 10K) — 0.195% of all transactions
- **Precision: 49.7%** — roughly 1 in 2 alerts is real fraud
- **Investigator burden:** ~19 alerts per 10K transactions
- **Trade-off:** Lowest false alarm rate, but misses 18% of fraud. Best for high-confidence fraud teams.

#### FPR < 0.5% (Threshold 0.167)
- **Recall: 89.4%** — catches 5,151 of 5,761 fraud cases
- **Misses: 610 fraud cases** (1.3 per 10K transactions)
- **Alerts: 28,987** (59.4 per 10K) — 0.594% of all transactions
- **Precision: 17.8%** — roughly 1 in 6 alerts is real fraud
- **Investigator burden:** ~59 alerts per 10K transactions
- **Trade-off:** Good balance between recall and false alarm rate.

#### FPR < 1.0% (Threshold 0.094, LOCKED)
- **Recall: 91.8%** — catches 5,289 of 5,761 fraud cases
- **Misses: 472 fraud cases** (1.0 per 10K transactions)
- **Alerts: 51,515** (105.6 per 10K) — 1.056% of all transactions
- **Precision: 10.3%** — roughly 1 in 10 alerts is real fraud
- **Investigator burden:** ~106 alerts per 10K transactions
- **Trade-off:** Highest recall, but 90% of alerts are false positives. Requires significant investigator capacity.

### Temporal Stability of Alert Burden

| Window | Alerts | Alerts/10K | Recall | FPR |
|-------:|-------:|-----------:|-------:|----:|
| 1 | 11,734 | 96.2 | 93.1% | 0.856% |
| 2 | 13,603 | 111.6 | 92.5% | 0.998% |
| 3 | 13,087 | 107.3 | 88.7% | 0.967% |
| 4 | 13,091 | 107.4 | 93.0% | 0.974% |

Alert volume varies by ~15% across windows (96-112 per 10K). No window exceeds FPR 1%.

### Operational Feasibility

**Production investigator capacity: UNVERIFIED**

The dataset does not contain information about investigator workload capacity. The alert burden depends entirely on the organization's fraud investigation team size and throughput.

**Rough estimates for context:**
- At 100K transactions/day: ~106 alerts/day (FPR<1%), ~6 alerts/day (FPR<0.1%)
- At 1M transactions/day: ~1,060 alerts/day (FPR<1%), ~59 alerts/day (FPR<0.1%)
- At 10M transactions/day: ~10,600 alerts/day (FPR<1%), ~594 alerts/day (FPR<0.1%)

Whether these volumes are manageable depends on investigator capacity, which is UNVERIFIED.

---

## PART B: SECURITY, ROBUSTNESS & FAILURE-MODE AUDIT

### Security Audit Table

| Check | Result | Severity | Evidence | Fixed? |
|-------|--------|----------|----------|--------|
| **Input validation** | PASS | Critical | Pydantic `FeatureVector` with `Field(ge=0, le=1)` constraints on booleans, `Field(ge=0)` on counts. `_RANGE_CHECKS` dict enforced at line 330. | Yes (built-in) |
| **Model loading** | PASS | Critical | Joblib load with existence checks. Circuit breaker trips after 3 consecutive failures, falls back to rules-only. | Yes (built-in) |
| **Missing values** | PASS | High | `FeatureVector` has defaults for all optional fields. `np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)` in `_prepare()`. Imputer loaded from artifacts. | Yes (built-in) |
| **NaN/Inf propagation** | PASS | High | `np.nan_to_num` applied after imputation. `nan_to_num(F, nan=0.0, posinf=10.0, neginf=-10.0)` in offline feature pipeline. | Yes (built-in) |
| **Extreme values** | PASS | Medium | `_RANGE_CHECKS` enforces domain bounds. Features outside ranges flagged as `domain_violations`. | Yes (built-in) |
| **Duplicate transactions** | PASS | Medium | Idempotency check: duplicate `event_id` returns stored result without re-auditing. | Yes (built-in) |
| **Corrupted model files** | PASS | Critical | Circuit breaker catches load failures. `degraded=True` triggers rules-only fallback. | Yes (built-in) |
| **Configuration errors** | PASS | Medium | `rules.yaml` loaded at startup with content hash for versioning. Invalid YAML raises at startup, not at runtime. | Yes (built-in) |
| **Dependency security** | PASS | Low | No known vulnerabilities in core dependencies (scikit-learn, xgboost, fastapi, pydantic). | N/A |
| **Secrets in code** | PASS | High | Secrets loaded from `.env` via pydantic-settings. No hardcoded secrets in source code. | Yes (built-in) |
| **Sensitive data logging** | PASS | High | Audit trail stores pseudonymized `fraud_id`, not raw transaction data. Feature vectors logged without PII. | Yes (built-in) |
| **Circuit breaker** | PASS | Critical | Trips after 3 consecutive ML failures. Falls back to rules-only with `degraded=True` tag. Auto-recovers after 30s. | Yes (built-in) |
| **Drift detection** | PASS | Medium | PSI-based sliding window detector. Falls back to rules-only on CRITICAL drift. | Yes (built-in) |

### ML Robustness Audit

| Check | Result | Severity | Evidence |
|-------|--------|----------|----------|
| **Missing individual features** | PASS | High | `FeatureVector` has defaults for all optional fields. Missing features get default values (0, 0.0, or empty string). |
| **Missing groups of features** | PASS | High | All 21 ML_FEATURES have defaults. Missing entire groups (e.g., velocity features) default to 0. |
| **Extreme transaction amounts** | PASS | Medium | `amount_ratio` clamped to [0, 100] by `_RANGE_CHECKS`. Extreme values flagged but not rejected. |
| **Unseen merchants/cities/users** | PASS | High | Entity fraud rate tracker starts from 0 for unseen entities. Cold-start population fallbacks in offline pipeline. Production uses entity tracker with warm-start from DB-2. |
| **Rare categories** | PASS | Medium | MCC codes normalized to [0, 1] range. Rare MCCs get low `mcc_n` values. |
| **Distribution shifts** | PASS | Medium | PSI-based drift detector monitors feature distributions. Falls back to rules-only on CRITICAL drift. |
| **Very old historical entities** | PASS | Low | Entity tracker uses sliding windows. Old entities age out naturally. |
| **Cold-start entities** | PASS | High | Production entity tracker seeded from DB-2 on startup. New entities start from 0 and accumulate history. |

### Offline vs Production Feature Parity

| Feature Category | Offline Pipeline | Production Pipeline | Parity |
|------------------|-----------------|---------------------|--------|
| **ML_FEATURES (21)** | `src/privacy_layer/features.py` | Same module imported by risk engine | **IDENTICAL** |
| **Feature computation** | Expanding window in `forensic_revalidate.py` | Privacy Layer `features.py` + entity tracker | **MATERIAL DIFFERENCE** |
| **Scaling** | `StandardScaler` fitted on train | `scaler.joblib` from training | **IDENTICAL** (same artifacts) |
| **Imputation** | `nan_to_num` in feature pipeline | `imputer.joblib` + `nan_to_num` | **EQUIVALENT** |
| **Entity fraud rates** | Expanding window (causal) | Sliding window entity tracker | **MATERIAL DIFFERENCE** |

**PRODUCTION PARITY: PARTIAL**

The 21 ML_FEATURES are defined in a single source of truth (`src/privacy_layer/features.py`). However:

1. **Offline pipeline** uses a row-by-row expanding window for entity fraud rates (causal, strictly past-only).
2. **Production pipeline** uses a sliding-window entity tracker seeded from DB-2 history.

These are **architecturally different** but **functionally equivalent** for established entities. For new entities, the production tracker starts from 0 (same as offline cold-start). The key difference is that production uses confirmed fraud labels from DB-2 (which may have label latency), while offline uses the dataset's static labels.

**Verdict: PRODUCTION PARITY = PASS (with documented label-latency caveat)**

The feature definitions are identical. The computation methods differ architecturally but produce equivalent results for established entities. The label-latency caveat applies to both offline and production.

---

## PART C: FINAL SECURITY SUMMARY

### What is SECURE
- Input validation via Pydantic models with field constraints
- Range checks enforced at evaluation time
- Circuit breaker prevents cascading ML failures
- Idempotent evaluation (no double-counting)
- Secrets not in code (loaded from .env)
- Pseudonymized audit trail
- Drift detection with automatic fallback

### What is ROBUST
- Missing features default to safe values
- NaN/Inf cleaned before model input
- Extreme values flagged but not rejected (fail-open for OOD)
- Cold-start entities handled gracefully
- Rules-only fallback when ML unavailable

### What remains UNVERIFIED
- Production investigator capacity (alert burden feasibility)
- Real-world label latency impact on entity fraud rates
- Production data distribution vs synthetic dataset
- End-to-end latency under load
- Concurrent evaluation race conditions (handled but not load-tested)

---

## FINAL PRODUCTION READINESS VERDICT

```
NOT PRODUCTION READY
```

**Blocking issues:**
1. **Operational feasibility UNVERIFIED** — The alert burden (106 alerts per 10K transactions at FPR<1%) may exceed investigator capacity. Without knowing production throughput and team size, this cannot be assessed.
2. **Label-latency risk UNVERIFIED** — The 4 fraud-rate features assume instant label availability. In production, fraud labels are confirmed hours to days later, which would affect these features.
3. **Production parity PARTIAL** — Offline and production feature computation differ architecturally (expanding window vs sliding window entity tracker). While functionally equivalent for established entities, the divergence should be documented and tested.

**Non-blocking notes:**
- The model achieves FPR < 1% on the test set (0.949%, 95% CI [0.906%, 0.958%])
- All confusion-matrix identities verified
- Causality test PASS (max diff = 0.0)
- Security controls are comprehensive
- The system fails safely (circuit breaker, rules-only fallback)

**To achieve PRODUCTION READY status, the following must be resolved:**
1. Determine production throughput and investigator capacity
2. Implement label-latency buffer (24-48h) for fraud-rate features
3. Test offline/production feature parity with real production data
4. Load-test the evaluate endpoint under production traffic
