# PHASE 10B — PARITY AND PRODUCTION-IMPACT EXECUTION REPORT

**Candidate:** `V_rawplus` (artifact = `models/model_records/mission_E_hardneg/models.joblib`)
**Candidate threshold:** `tV* = 0.0298937337`
**Production model:** `altman_native_E_hardneg_cert_20260904` @ `0.018758` — **UNTOUCHED**
**Final test (2018–2020):** **NOT accessed, NOT scored** — `FINAL_TEST_ACCESS = FALSE` (asserted in both harnesses)
**Date:** 2026-09-07T14:10Z

---

## 0. What was actually executed

Two harnesses ran real code against the locked train-window shadow input
(`data/_raw_parity_cache_tr.npz`, 879,196 rows, 1991–2015, SHA-256
`9c866f11…`, **max(year) < 2016 asserted programmatically before any feature
work**):

- `phase10b_gateB.py` — 48-feature parity matrix, 20-case edge suite,
  cache/reset/reorder regressions, temporal parity, fraud-rate parity split,
  score parity, precision audit, zero-value audit.
- `phase10b_gateC.py` — shadow alert-rate analysis at the two frozen
  thresholds, PSI compatibility with a controlled shift, online loading,
  rollback compatibility, resource performance.

The prior placeholder `alert_rate_analysis.json` (which carried identical
score distributions for both models — it had read one score array twice) was
replaced by real shadow scoring.

---

## 1. Input manifest (firewall evidence)

| Field | Value |
|---|---|
| Source path | `data/_raw_parity_cache_tr.npz` |
| SHA-256 | `9c866f118ac2c7065a7b668daa46ba7d94ec801d30f92ca63c6ad090fd5aa882` |
| Row count | 879,196 |
| Fraud / legit | 21,345 / 857,851 |
| Year range | 1991–2015 (max year 2015 < 2016) |
| Entities | users/merchants/cities/cards in `input_manifest_tr.json` |
| Final-test overlap | 0 rows ≥2016; 0 rows ≥2018; 0 fraud rows 2018–2020 |
| Filter | `year < 2016` (train window only) |
| Reason permitted | development sample strictly outside the final-test window |

---

## 2. Gate B — 48-feature parity (executed)

**Paths compared**

- **Offline:** the training-time expanding context
  (`retrain_native_consistent.build_context_and_features`) → shared
  `derive_native_features`; the cached `X` IS this vector.
- **Production-as-wired:** the live wiring — privacy layer forwards raw native
  columns + 4 velocity keys from `UserVelocityTracker`; the risk engine
  `FeatureVector` fills every other native field with its declared default
  (`0.0` / `""`); `AltmanNativeEnsembleEngine.predict()` is called **without an
  entity tracker**. Both paths were replayed chronologically over the identical
  879,196-row stream.

**48-row matrix result (300,000-row locked sample, tolerance 1e-4)**

| Feature | offline mean | production mean | frac rows > tol | status |
|---|--:|--:|--:|:--:|
| merchant_id | 54,740.84 | **0.0000** | 1.0000 | **FAIL** |
| user_merchant_diversity | 67.15 | **1.0000** | 0.9964 | **FAIL** |
| user_city_diversity | 32.82 | **1.0000** | 0.9932 | **FAIL** |
| user_merch_count | 39.88 | **0.0000** | 0.8474 | **FAIL** |
| user_fraud_rate | 0.0212 | **0.0010** | 0.5835 | **FAIL** |
| merch_fraud_rate | 0.0201 | **0.0010** | 0.5724 | **FAIL** |
| city_fraud_rate | 0.0228 | **0.0010** | 0.4772 | **FAIL** |
| user_avg_amt | 46.05 | 46.05 | 0.0007 | FAIL* |
| amt_vs_user_avg | 1.26 | 1.26 | 0.0002 | FAIL* |
| amt_zscore | 0.26 | 0.26 | 0.0002 | FAIL* |
| other 38 features | — | — | 0.0 | PASS |

\* precision-boundary: max delta = 1e-4 exactly (float32 cache storage vs
float64 compute) on <0.08% of rows; not semantic. The 7 semantic failures
affect 48–100% of rows.

**Root causes (all PROVEN by code + execution):**

1. **`merchant_id` = constant 0.0** — the privacy layer `IngestTransactionRequest`
   schema has **no merchant-name/merchant-id field**, so `merchant_id` defaults
   to `""`, and `_from_raw_native` maps it to `_code("") = 0.0` for every row.
   The model was trained on real merchant hash codes.
2. **`user_merchant_diversity` / `user_city_diversity` = constant 1.0** — the
   `FeatureVector` default is `0.0` and `derive_native_features` clamps
   `max(0, 1.0) = 1.0`; the privacy layer never sends the real distinct counts.
3. **`user_merch_count` = constant 0.0** — never tracked in production.
4. **fraud rates = constant 0.001** — the wired path feeds `0.0` defaults →
   `COLD_START_FRAUD_RATE 0.001`; even the `EntityFraudRateTracker` variant
   (100-event sliding window, min 5 events) differs numerically from the
   offline **unbounded expanding mean** (48–58% of rows exceed tolerance).
5. **No timestamp cutoff in production trackers** — `UserVelocityTracker` /
   `EntityFraudRateTracker` accumulate in **ingestion order**; a future-dated
   row ingested first (backdate/replay) changes earlier rows' features
   (measured max delta 9,999 on 6 velocity/amount features). Offline sorts by
   `ts` before building expanding context (strict-before-ts by construction).

**Edge cases (20/20): PASS** — when context inputs are identical, the shared
derivation produces identical vectors on all 20 cases (zero/missing amount,
NaN/malformed fields, unseen entities, online/swipe/chip, extreme amounts).

**Cache/reset regression: PASS** — cold == warm == reset == restart; reordered
batches show expected causal dependence (first-observed row always sees empty
context); amount=0 and count=0 are preserved (not missing).

**Fraud-rate parity split:**

```
CAUSALITY_STATUS                 = PASS      (prior-only aggregation, verified)
NUMERICAL_PARITY_STATUS (wired)  = FAIL      (constant 0.001 vs expanding means)
NUMERICAL_PARITY_STATUS (tracker)= FAIL      (100-window/min-5 vs unbounded mean)
PRODUCTION_LABEL_AVAILABILITY    = UNVERIFIED (no confirmed label at scoring time;
                                              risk engine feeds score>=70 proxy)
```

**Score parity at `tV* = 0.0298937337`:** max |Δ| = 0.995;
**67,576 / 300,000 decision disagreements (22.5%)** between offline scoring and
production-path scoring. `DECISION_DISAGREEMENTS != 0`.

**Zero-value audit:** amount=0 preserved; count=0 preserved; 40 candidate
falsy-default regex hits reviewed — the wired `0.0 → 0.001` fraud-rate
conversion is the documented `COLD_START_FRAUD_RATE` behavior, not a
zero→missing bug.

**Precision audit:** both paths cast to float32 before scaling (matches
training); residual float32/float64 deltas recorded in `precision_audit.json`.

> ### `GATE_B_STATUS = FAIL`
> Seven of 48 features are semantically divergent in production-as-wired
> (48–100% of rows), temporal semantics diverge for out-of-order ingestion, and
> 22.5% of frozen-threshold decisions flip between offline and production
> scoring. This is a material, execution-proven mismatch.

---

## 3. Gate C — Production impact (executed)

**Shadow alert-rate analysis** (879,196 train-window rows, frozen thresholds,
no tuning):

| Model | Threshold | Rows | Alerts | Alert Rate | Alerts / 1k |
|---|---:|---:|---:|---:|---:|
| E_hardneg | 0.018758 | 879,196 | 204,668 | 0.2328 | 232.79 |
| V_rawplus | 0.0298937337 | 879,196 | 163,198 | 0.1856 | 185.62 |

`OPERATIONAL_ALERT_LIMIT = UNDEFINED` (no formal ceiling exists; none invented).

**Important structural finding:** the candidate ensemble is **weight-identical
to the certified production model** (max |Δ| = 0.0 on random vectors). The
alert-rate reduction is **threshold-only** — it is not a model improvement.
`E@0.018758` vs `V@0.0298937337` differ only by the frozen threshold.

**Channel alert rates (per 1,000):** online E=496 / V=446 · swipe E=208 / V=158
· chip E=71 / V=55.

**PSI compatibility (existing production implementation):**

| Window | amt PSI | log_amt PSI | mcc PSI | Level |
|---|---:|---:|---:|---|
| Normal shadow | 0.0030 | 0.0030 | 0.0009 | ok |
| Controlled shift (amt×1.6, mcc+2000) | 0.1525 | 0.1526 | 11.7237 | **warn / alert** |

Controlled shift **detected**; no NaN, no division-by-zero, no empty-bin
crash, no false "no drift" claim. `PSI_COMPATIBILITY = PASS`.

**Online loading:** artifact loads (104 ms), hash
`8875030e…`, 48-feature contract matches, inference OK, malformed/missing
input fails safe (degraded rules-only, never 500s).

**Rollback compatibility:** reload of `altman_native_E_hardneg_cert_20260904`
→ identity OK, manifest threshold 0.018758, scores reproduced on reload
(`allclose`), production untouched throughout.

**Resource performance:** load 104 ms · batch 10k rows 0.0082 ms/row · single-row
scaler 0.12 ms · failure rate 0.

> ### `GATE_C_STATUS = PASS`
> The technical shadow harness runs cleanly. Gate C is not the blocker.

---

## 4. Required final gate table

| Gate | Status | Evidence | Blocking? |
|---|---:|---|:---:|
| A — Label availability | **FAIL / UNVERIFIED** | Phase-10A evidence + `label_availability_evidence.json` (no confirmed label at scoring time; risk engine feeds score≥70 proxy) | **YES** |
| B — Feature parity | **FAIL** | actual harness: 7/48 features semantically divergent (48–100% rows), temporal cutoff divergence, 22.5% decision disagreement at tV* | **YES** |
| C — Production impact | **PASS** | actual harness: alert-rate table, PSI shift detected, loading/rollback/perf OK | NO |
| Final-test firewall | **PASS** | runtime assertions: max(year)<2016, 0 final-test rows loaded, 0 2018–2020 fraud rows | **YES** |
| Production integrity | **PASS** | hash/checkpoint: production model+threshold+config untouched; candidate hashes match Phase 9 | **YES** |

```
CERTIFICATION_STATUS   = FAIL
FINAL_TEST_AUTHORIZED  = FALSE
```

`FINAL_TEST_AUTHORIZED` **must remain FALSE** — Gate A is unresolved AND Gate B
now FAILs by execution. Under the Phase-10B interpretation rules
(Gate A FAIL + Gate B FAIL ⇒ `FAIL`), `V_rawplus` is **not** `CONDITIONAL` and
does **not** advance to a final-test evaluation.

---

## 5. What this means (no invented PASSes)

- **Gate B failed for real, structural reasons.** The deployed wiring cannot
  reproduce 7 of the 48 features the model was trained on. This is not a
  report-writing issue; it is a production/offline feature divergence that
  flips 22.5% of decisions at the frozen threshold.
- **The "V_rawplus improvement" is threshold-only.** Candidate and production
  ensembles are the same weights (verified bit-identical); the FPR/alert-rate
  advantage comes from raising the operating threshold from 0.018758 to
  0.0298937337, not from a better model.
- **Gate C passes technically** but changes nothing: PSI works, loading and
  rollback work, resources are fine. Those were never the blocker.
- **Recommended next step:** do NOT patch the candidate. Fix the production
  feature wiring (forward merchant identity, diversity, merch_count, and
  confirmed-label fraud rates through the live path — a new version requires
  new validation and certification), or drop/retrain without the seven
  divergent features as a genuinely new candidate. Re-certify from scratch.

---

## 6. Evidence classification

| Conclusion | Class |
|---|---|
| merchant_id is constant 0.0 in production-as-wired (100% rows) | **PROVEN** |
| diversity/merch_count constants in production-as-wired | **PROVEN** |
| fraud-rate features constant 0.001 wired; tracker variant ≠ offline | **PROVEN** |
| production trackers lack a ts cutoff (out-of-order pollution) | **PROVEN** |
| score parity: 22.5% decision disagreements at tV* | **PROVEN** |
| candidate == production weights (threshold-only difference) | **PROVEN** |
| edge-case derivation parity (20/20) | **PROVEN** |
| cache/reset safety, zero-value safety | **PROVEN** |
| PSI compatibility with controlled-shift detection | **PROVEN** |
| label availability at scoring time | **UNVERIFIED** |

Artifacts: `reports/phase10_remediation/` — `feature_parity_results.json`,
`edge_case_parity.json`, `cache_reset_regression.json`, `temporal_parity.json`,
`fraud_rate_parity.json`, `score_parity.json`, `precision_audit.json`,
`zero_value_audit.json`, `input_manifest_tr.json`, `alert_rate_analysis.json`,
`psi_compatibility.json`, `online_loading_test.json`,
`rollback_compatibility.json`, `resource_performance.json`,
`gate_b_decision.json`, `gate_c_decision.json`, `phase10_gate_summary.json`.