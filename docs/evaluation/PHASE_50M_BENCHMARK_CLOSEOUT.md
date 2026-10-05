# Phase Closeout — 50M Synthetic Scale Benchmark Construction

**Final status: `PASS WITH LIMITATIONS`**

**Date:** 2026-10-05
**Repository:** PS-14-Fraud-Detection-software
**Git SHA (unchanged, nothing committed):** `5a5ff55cc03df73318fdf31a16e3665f159a4720`
**Specification:** [`SYNTHETIC_50M_BENCHMARK_SPEC.md`](SYNTHETIC_50M_BENCHMARK_SPEC.md) v1.0 — written *before* generation, per §4

> This is **not** a bare `PASS`. Generation completed and every structural,
> integrity, leakage and reproducibility check passed, but there are material
> statistical deviations from the real foundation, and the fraud labels are
> assumption-driven by construction. Both are itemised below and in the spec.

---

## 1. Did 50M rows get generated?

**Yes — exactly 50,000,000.**

| field | value |
|---|---|
| target rows | 50,000,000 |
| **actual rows** | **50,000,000** (exact) |
| fraud rows | 61,109 |
| fraud prevalence | 0.00122218 (target 0.00122020; +0.16% relative) |
| partitions | 166 monthly Parquet partitions |
| storage | 3.7 GB (zstd, 1,000,000-row row groups) |
| generation wall time | 265.3 s |
| feature contract | 48 features, manifest order verified |

## 2. Source dataset

| field | value |
|---|---|
| path | `data/credit_card_transactions-ibm_v2.csv` |
| SHA-256 | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |
| observed rows | 24,386,900 |
| why this source | the only acquired dataset satisfying the native 48-feature contract (48/48) |

The source hash equals `training_dataset_sha256` in
`models/production/manifest.json` — independent cross-check that the
foundation is the dataset the production model was trained on.

**The real dataset was not modified.** Verified byte-identical after generation
(§6).

## 3. Seed, generator version, hashes

| field | value |
|---|---|
| seed | `20261005` |
| generator version | `1.0.0` |
| generator | [`scripts/generate_synthetic_50m.py`](../../scripts/generate_synthetic_50m.py) |
| validator | [`scripts/validate_synthetic_50m.py`](../../scripts/validate_synthetic_50m.py) |
| profile builder | [`scripts/profile_ibm_v2.py`](../../scripts/profile_ibm_v2.py) |
| manifest | `data/synthetic_50m/manifest.json` |
| schema hash | `fb1d8e6a19ffa5837a28747bd928a6c40d4111a6c36eb6d89049bde620e09d96` |
| validation report | `misc/reports/synthetic_50m_validation.json` |
| empirical profile | `misc/reports/ibm_v2_empirical_profile.json` |

Per-partition SHA-256 for all 166 partitions is recorded in the manifest.

## 4. Entity scaling

| entity | observed | benchmark |
|---|---|---|
| users | 2,000 | 4,101 |
| merchants | 100,343 | 205,731 |
| cities | 13,429 | 27,533 |
| cards | — | 8,202 |

Per-entity intensity held at the observed means; new entities use disjoint
`synthetic ID ranges so no synthetic row can be confused with an observed one.

## 5. Validation results

**`VALIDATION PASS — failures=0`.** Full check list:

| check | result |
|---|---|
| partitions_present (166) | PASS |
| contract_features_present | PASS |
| no_unexpected_columns | PASS |
| feature_order_matches_manifest | PASS |
| no_index_column | PASS |
| row_count_exact (50,000,000) | PASS |
| label_domain (int8 0/1) | PASS |
| label_not_reconstructable_from_single_feature | PASS — max exact AUC **0.7205** (`amt_sq`) vs 0.95 threshold |
| train_test_entity_disjoint | PASS — 4,101 user_ids, **0** in both splits |
| metadata_not_in_features | PASS |
| past_only_historical_aggregate | PASS — **0 mismatches over 301,205 rows** |
| reproducible_from_seed | PASS — partition `2002-09` hash `5db67b31a506c7e3…` reproduced exactly |

### 5.1 Statistical comparison (deviations reported, not required to be zero)

| quantity | IBM v2 (observed) | benchmark | note |
|---|---|---|---|
| fraud prevalence | 0.0012202 | 0.0012222 | matched by construction |
| channel swipe/chip/online | .6310/.2578/.1112 | .6308/.2578/.1113 | excellent |
| amount p25 / p50 / p75 | 9.47 / 32.19 / 66.83 | 9.70 / 31.79 / 70.16 | close |
| amount p95 | 147.41 | 181.44 | **+23% — heavier tail** |
| amount p99 | 313.91 | 422.34 | **+35% — heavier tail** |
| amount p1 | −96.00 | 0.00 | **see §7.1** |
| error rate | 0.01536 | 0.02413 | **+57% — by design** |

## 6. Invariants (§15) — all verified

| invariant | result |
|---|---|
| real dataset byte-identical | **UNCHANGED** `b01fa323…` |
| Research Plan unchanged | **UNCHANGED** `eab5a061…` |
| preregistration unchanged | **UNCHANGED** `fdde71cb…` |
| production manifest unchanged | **UNCHANGED** `e6da1a43…` |
| `native_features.py` unchanged | **UNCHANGED** `5760a203…` |
| `altman_native_ensemble.py` unchanged | **UNCHANGED** `55f4ad52…` |
| model threshold unchanged | `0.7847116291110687` |
| reviewer assignments unchanged | `review_resolution_check.py` rc=0, 0/14 resolved |
| 50M represented as real data | **No** — every row `provenance_class = SYNTHETIC` |

### Repository verification battery

**84 PASS / 0 FAIL.** Named checkers, all rc=0:

`review_package_check.py` · `review_resolution_check.py` ·
`claim_evidence_check.py` · `eval_record_test.py` · `check_freeze_test.py`

Freeze checker: **rc=1 / 78 problems** — expected pre-freeze state preserved.
(`check_freeze_test.py` requires `PYTHONIOENCODING=utf-8`; a pre-existing
cp1252 quirk, not a regression.)

## 7. Known limitations

**7.1 Negative amounts are clamped.** IBM v2 p1 is −96.00 (refunds/reversals);
the benchmark p1 is 0.00. This is **not** a generation error: the runtime
contract does `amt = max(_f2(raw.get("amount")), 0.0)`, so the deployed model
never sees a negative amount. The benchmark reproduces what the model actually
consumes. The underlying negative tail exists in the generator and is erased
only at feature level.

**7.2 Error rate is ~57% higher** (0.02413 vs 0.01536). The spec inflates
card-not-present error probability by 2.4× as a declared assumption. This is a
deliberate, documented deviation, not a fit failure.

**7.3 Right tail is heavier** (p99 +35%). The quantile mapping is anchored at 10
empirical points and then scaled by a per-user lognormal tendency, which fattens
the upper tail relative to the source.

**7.4 All fraud labels are assumption-driven.** A declared logistic rule over
generator-internal latents (amount magnitude, online, night, error, new-merchant,
velocity, distance-from-home). Prevalence is matched by solving the intercept
by bisection — it is **not** derived from a real fraud process, and this
benchmark will be **easier** to learn than real fraud.

**7.5 Per-entity intensity is matched in the mean only.** The full degree
distribution is approximated, not replicated.

**7.6 `has_zip` / `has_state` are constant 1** because IBM v2 has 0.00% blanks
in both. Those two features carry no signal — as in the real data.

**7.7 The model has a pre-existing train/serve skew.** The benchmark implements
the **runtime** contract (`native_features.py`) causally, deliberately *not* the
training script's method, because the latter computes `user_fraud_rate`,
`merch_fraud_rate` and `city_fraud_rate` from `groupby(label).agg(["mean","count"])`
over the whole dataset with no shift. That is target leakage, and it makes
`target_leakage: false` in the production manifest inaccurate. **Not fixed here**
— it is a research-integrity decision for the human reviewers, and this phase is
forbidden from touching models, thresholds or the manifest.

**7.8 `merchant_id` is lossy.** The runtime `_code` is `sha256(...) % 100000`,
so 205,731 merchants collapse onto 100,000 codes. Entity identity is therefore
not recoverable from that column; use `user_id`.

## 8. Is the benchmark suitable for the next phase?

**Yes, with the limitations above carried forward.** It is fit for
*engineering scale testing* of the existing system: throughput, memory profile,
partition pruning, feature-pipeline correctness at 50M rows.

It is **not** fit for any claim about real-world fraud-detection effectiveness,
generalisation, or institutional validation. Any metric computed on it is a
**synthetic-data metric** and must be labelled as such.

## 9. What this phase did not do

- No model tuning, architecture change, or threshold change.
- No change to the Research Plan, preregistration, or reviewer assignments.
- No institutional validation and no generalization claim.
- No confirmatory Track M and no large-scale model testing — per the stop
  condition, the next phase consumes this benchmark.
- No large-scale model testing was run; only the benchmark was built and
  validated.
- The 50M dataset is **not** committed to git (`data/` is gitignored); it is
  recreated by re-running the generator with the recorded seed and source hash.

## 10. Reproduction

```bash
./.venv/Scripts/python.exe scripts/profile_ibm_v2.py                  # empirical profile
./.venv/Scripts/python.exe scripts/generate_synthetic_50m.py --rows 50000000
./.venv/Scripts/python.exe scripts/validate_synthetic_50m.py
```

The reproducibility check regenerates all 166 partitions from the seed and
compares hashes; partition `2002-09` reproduced byte-identically
(`5db67b31a506c7e3…`).
