# Synthetic 50M Benchmark — Independent Validation Report

**Decision: `VALIDATED WITH LIMITATIONS`**

**Date:** 2026-10-05
**Repository:** PS-14-Fraud-Detection-software
**Report scope:** independent validation of the benchmark built in the previous phase. No model was tuned, trained, or evaluated against this benchmark.

---

## 1. Benchmark identity

Exactly the benchmark validated is identified below. If any of these change, this validation does **not** apply.

| field | value |
|---|---|
| Git SHA at generation | `5a5ff55cc03df73318fdf31a16e3665f159a4720` |
| Git SHA at validation | `5a5ff55cc03df73318fdf31a16e3665f159a4720` (unchanged) |
| benchmark path | `data/synthetic_50m/data/year_month=YYYY-MM/part-0.parquet` |
| partitions | 166 monthly Parquet partitions, zstd |
| combined partition hash | `c2db7142e24dd8d9c2806ef8a1ec0c23e62cbf2d8fcc4b4b614396922b47edf1` |
| row count | 50,000,000 |
| schema hash | `fb1d8e6a19ffa5837a28747bd928a6c40d4111a6c36eb6d89049bde620e09d96` |
| generator version | `1.0.0` |
| generation seed | `20261005` |
| creation timestamp | 2026-10-05T18:47:52 (manifest mtime) |
| manifest | `data/synthetic_50m/manifest.json` |
| validation outputs | `misc/reports/phase2_validation.json`, `phase2_comparison.json`, `phase2_separation_attribution.json` |

## 2. Source identity

| field | value |
|---|---|
| source | `data/credit_card_transactions-ibm_v2.csv` |
| SHA-256 | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |
| observed rows / fraud | 24,386,900 / 29,757 (0.0012202) |
| re-verified byte-identical after this phase | **YES** |

## 3. Generator identity

| field | value |
|---|---|
| generator | `scripts/generate_synthetic_50m.py` |
| validator (phase 2) | `scripts/validate_50m_phase2.py` |
| comparator | `scripts/compare_real_synthetic.py` |
| real reference builder | `scripts/build_real_reference_features.py` |
| feature semantics | runtime contract `backend/src/privacy_layer/native_features.py` |
| historical aggregates | strictly past-only |
| label mechanism | declared logistic over generator-internal latents; **no model consulted** |

## 4. Structural validation

Full streaming pass over all 166 partitions, 0 failures.

| check | result |
|---|---|
| exactly 50,000,000 data rows | **PASS** |
| duplicate columns | **PASS** — none |
| unexpected columns | **PASS** — none |
| schema identical across all 166 partitions | **PASS** |
| truncated / empty partitions | **PASS** — none |
| accidental index column | **PASS** — none (53 columns: 48 features + `label`, `user_id`, `year_month`, `split`, `provenance_class`) |
| nulls across 48 features | **0** |
| non-finite values (NaN/Inf) | **0** |
| duplicate rows | not observed; see §11 |
| label domain | `{0, 1}` only |
| data types | uniform across partitions (single schema verified) |

## 5. Feature-contract validation

All 48 native features are present, in manifest order, with the exact names in `models/production/manifest.json`. Acceptance is on **semantic** grounds, not name matching: each is produced by the runtime contract's own definition, verified by reading `native_features.py`.

Four features are **constant** across all 50M rows:

| feature | value | cause |
|---|---|---|
| `has_zip` | 1.0 | IBM v2 has 0.00% blank `Zip` — matches the real foundation |
| `has_state` | 1.0 | IBM v2 has 0.00% blank `Merchant State` — matches the real foundation |
| `mcc_travel` | 0.0 | **generator artefact** — the MCC sampler draws from the 20 most frequent observed codes, none of which fall in the 3000–3350 travel band |
| `mcc_online` | 0.0 | **generator artefact** — same cause; no drawn code falls in 5967–5969 |

`mcc_travel` and `mcc_online` being constant is a **real dependency defect** (see §8): the benchmark cannot exercise two of the 48 native features at all. This does not invalidate the benchmark for scale testing but is a documented generation limitation.

## 6. Label validation

| measure | value |
|---|---|
| total rows | 50,000,000 |
| fraud rows | 61,109 |
| non-fraud rows | 49,938,891 |
| prevalence | 0.00122218 (target 0.0012202, **+0.16%**) |
| prevalence across 166 monthly segments | min 0.001089, max 0.001374, spread 0.00029 |
| labels derived from model predictions | **NO** — the generator imports no model, weights, or threshold |
| label trivially encoded in one feature | **NO** — max single-feature exact AUC 0.7205 (`amt_sq`), and that is the declared amount signal, not an encoding |

Prevalence is stable over time (6× tighter than the overall rate), so there is no accidental temporal fraud artifact.

## 7. Distribution comparison

Compared against a real reference built from IBM v2 using the **same** 48-feature runtime contract and the same past-only discipline (`scripts/build_real_reference_features.py`, 1,200,000 rows).

Fourteen **history-depth features** are **excluded** from the head-to-head comparison: the real reference is a *strided* sample (see §13 limitation), so cumulative velocities on the real side are not comparable. Comparing them would manufacture a difference.

Largest standardised mean differences among the 34 comparable features:

| feature | SMD | synthetic mean | real mean |
|---|---|---|---|
| `Month` | 1.106 | 9.247 | 6.508 |
| `Day` | 0.302 | 13.054 | 15.705 |
| `merchant_id` | 0.196 | 50048.9 | 55704.0 |
| `city_id` | 0.179 | 50089.1 | 54776.9 |
| `mcc_high` | 0.170 | 0.812 | 0.742 |
| `is_online` | 0.139 | 0.111 | 0.159 |

Entity-id means differ only because they are hash codes over a different id space. `Month`/`Day` differences reflect the real reference's sampling, not necessarily a generator defect.

## 8. Dependency comparison

Spearman correlations over the 34 comparable features: **mean |Δρ| = 0.0534, max = 0.5868, 63 pairs above 0.10.**

### Relationships DESTROYED (scientifically important)

| pair | synthetic ρ | real ρ | Δ |
|---|---|---|---|
| `mcc` × `is_online` | −0.001 | **−0.443** | 0.441 |
| `mcc_high` × `is_online` | 0.000 | **−0.587** | 0.587 |
| `mcc` × `amt_x_online` | −0.001 | −0.443 | 0.442 |

In the real data, online transactions carry low MCC codes (online MCCs 5967–5969 are below the 5000 "high" threshold), producing a strong negative relationship. **The generator samples channel and MCC independently, so this relationship is entirely absent.** A model trained on this benchmark learns no MCC–channel interaction that exists in reality.

### Relationships ABSENT from the synthetic fraud process

Fraud-predictor AUC (single feature vs label):

| feature | synthetic | real | gap |
|---|---|---|---|
| `city_fraud_rate` | 0.5195 | **0.9155** | 0.3960 |
| `merch_fraud_rate` | 0.5066 | **0.7847** | 0.2781 |
| `user_fraud_rate` | 0.5093 | **0.7297** | 0.2205 |
| `merch_tx_count` | 0.5356 | 0.7530 | 0.2174 |
| `is_online` | 0.5411 | 0.7066 | 0.1655 |

This is the most consequential finding. In the real data, the historical entity fraud rates are the **strongest** fraud predictors (city fraud rate alone reaches 0.92 AUC). In the benchmark they are **uninformative** (≈0.51), because the declared label mechanism deliberately does not consume them — by design, to avoid circularity. Consequence: **a model trained on this benchmark would learn to ignore the feature family that dominates real fraud prediction.** This is a structural limitation of the benchmark, not a bug.

## 9. Entity validation

| measure | value |
|---|---|
| distinct users | 4,101 (synthetic IDs; explicit, not observed) |
| transactions per user | mean 12,192.1, min 11,842, max 12,623, p95 12,376 |
| top-1% users' share of transactions | 0.010 |
| user-level fraud rate | min 0.00008, max 0.00382, mean 0.00122 |
| merchants / cities | 205,731 / 27,533 |
| entity identifiers | **synthetic** — `SYN_MERCH_*`, `SYN_CITY_*`, `SYN_CARD_*`, and integer `user_id` |

Per-user volume is **deliberately near-uniform** (min 11,842 vs max 12,623 — a 6.6% spread). Real fraud data has a heavy-tailed activity distribution. Concentration is therefore **not** preserved: `top-1% users hold 0.010` of rows means essentially no user is dominant. A benchmark task that depends on realistic entity skew (e.g. card-testing bursts from one account) will not exercise that behaviour.

## 10. Temporal validation

| check | result |
|---|---|
| temporal ordering | **PASS** — rows emitted in non-decreasing time within each partition; historical features never use a later row |
| timestamp validity | **PASS** — all `hr`/`mn`/`Month`/`Day`/`dow` within valid ranges; 166 valid month partitions |
| past-only invariant | **PASS** — 0 mismatches over 301,205 rows when `user_fraud_rate` is independently reconstructed from prior labels |
| future information in historical features | **NONE** — verified, not assumed |
| fraud prevalence artifact over time | **NONE** — monthly spread 0.00029, 6× tighter than the base rate |
| intended partitions created | 166 monthly partitions, plus a `split` column (train/test) and a `year_month` key |

## 11. Train/test contamination

| check | result |
|---|---|
| users appearing in both splits | **0 of 4,101** — entity-disjointness verified mechanically, not by seed assumption |
| train fraction of users | 0.499 (0.5 expected) |
| can `split` be predicted from the 48 features? | **AUC 0.5053 — no.** Split membership carries no generator artefact |
| duplicate transactions | none detected (no duplicate `event_id`; rows are generated, not copied) |
| source records copied into the benchmark | **NONE** — the benchmark shares no row identity with IBM v2; all rows are `SYNTHETIC` |
| generator metadata leaking into features | **NONE** — `provenance_class`, `split`, `year_month`, `user_id` are outside the 48-feature list |

## 12. Leakage attack tests

| attack | AUC | verdict |
|---|---|---|
| label itself (sanity check) | 1.0000 | control |
| row position / generation order | 0.5113 | **no leakage** |
| `year_month` partition key | 0.5155 | **no leakage** |
| `user_id` | 0.5057 | **no leakage** |
| `split` membership from features | 0.5053 | **no leakage** |

No metadata field, ordering artefact, or identifier encodes the label. Source-row identifiers are absent by construction.

## 13. Real-vs-synthetic distinguishability

**Observed: a classifier separates synthetic from real at CV ROC-AUC = 1.0000 (±0.0000).**

Single-feature separation:

| feature | AUC |
|---|---|
| `card_tx_count` | 0.9982 |
| `merch_tx_count` | 0.9767 |
| `user_fraud_rate` | 0.7725 |
| `Day` | 0.7538 |
| `Month` | 0.7056 |

Permutation importance is dominated by `user_city_diversity` (AUC drop 0.2498). **Leave-one-feature-out leaves AUC at 1.0000 for every single feature removed**, so separation is redundant across many features rather than caused by one.

### Interpretation — and an important caveat

The separation axis is **history-depth features**. Those are precisely the features whose real-side values this validation could **not** reconstruct faithfully: IBM v2 is sorted by `User`, so computing full-history velocities requires reading all 24,386,900 rows, and the reference used here is a strided sample. On a strided stream the real velocities are understated by construction.

**Therefore the magnitude of this result is CONFOUNDED.** This test cannot presently distinguish "the benchmark is unrealistic" from "the reference's velocity features are wrong". The qualitative direction is real — the benchmark's velocity profile is not what a full-history real reference would produce — but the AUC of 1.0 must **not** be cited as proof of unrealisticity without a full-history reference.

**This is a limitation of the validation, not a clean pass and not a clean fail.**

## 14. Reproducibility

| check | result |
|---|---|
| 8 randomly sampled partition hashes vs manifest | **8/8 MATCH, 0 mismatch** |
| combined partition hash | `c2db7142e24dd8d9c2806ef8a1ec0c23e62cbf2d8fcc4b4b614396922b47edf1` — unchanged from the identity recorded in §1 |
| full regeneration from seed | verified in the construction phase: partition `2002-09` regenerated byte-identically (`5db67b31a506c7e3…`) |

The benchmark on disk is the benchmark the manifest describes. The generator is not committed to git but is recorded by version and is deterministic given the seed and source hash.

## 15. Scale and storage measurements

| measure | value |
|---|---|
| total on disk | 3.65 GiB |
| parquet bytes | 3.65 GiB (all content is Parquet) |
| bytes per row | 78.33 |
| partitions | 166 (mean ~22.5 MiB each) |
| full sequential scan | 3.65 GiB in ~90 s |
| aggregate read throughput | ~42 MiB/s |
| validation memory | flat; only aggregates retained per 250k-row batch |

These are the baseline figures for the next phase. No application optimisation was performed.

## 16. Scientific suitability

| purpose | status |
|---|---|
| Large-scale storage testing | **ESTABLISHED** |
| Data-pipeline testing | **ESTABLISHED** |
| Throughput testing | **ESTABLISHED** |
| Training scalability testing | **ESTABLISHED** |
| Model-performance comparison | **LIMITED** — usable only for scaling/throughput; the fraud structure is materially simpler than reality (§8) and the historical fraud-rate features carry no signal that they carry in real data |
| Real-world fraud effectiveness | **NOT ESTABLISHED** |
| Institutional validation | **NOT ESTABLISHED** |

The benchmark's existence implies nothing about real-world validity.

## 17. Failures

No validation check returned FAIL. Three defects were found and fixed **in the validation tooling itself** during this phase, and are recorded for honesty:

1. `dict.update()` **replaces** rather than accumulates, so a month spanning two 250k batches recorded only the last batch's row count. This inflated reported per-month prevalence ~6× and produced a false PASS on the temporal check. Fixed; per-month prevalence is now internally consistent with the overall rate.
2. The real-reference builder marked `err = 1` for every row because `NaN` became the literal string `"nan"`. The runtime contract guards exactly this (`not in ("", "nan", "None")`). Fixed; real error rate is now 0.019 vs IBM's raw 0.0154.
3. The real reference initially read a **prefix** of a user-sorted file, covering ~100 of 2,000 users. Fixed to strided sampling spanning all users.

## 18. Limitations

1. **All 50M rows and all fraud labels are synthetic.** Nothing here is evidence about real fraud.
2. **Fraud structure is materially simpler than reality.** Historical entity fraud rates — the strongest real predictors — are uninformative in the benchmark (§8). A model trained on it would under-weight them.
3. **MCC–channel interaction is destroyed** (§8), and `mcc_travel` / `mcc_online` are constant (§5), so 2 of 48 features are untestable.
4. **Entity activity is near-uniform** (6.6% spread) rather than heavy-tailed (§9); card-testing and burst behaviour cannot be exercised.
5. **The distinguishability result is confounded** (§13) and must not be cited as proof of unrealisticity.
6. **Fourteen history-depth features were excluded** from the head-to-head comparison; their real-side values are not faithfully reconstructible from a strided sample.
7. Amount right tail is heavier than IBM (p99 422 vs 314) and the error rate is ~57% higher by declared CNP assumption.
8. `has_zip` / `has_state` are constant 1 because IBM v2 has no blanks.

## 19. Final recommendation

### `VALIDATED WITH LIMITATIONS`

The benchmark is **suitable for the next large-scale engineering/model-testing phase** for storage, pipeline, throughput and training-scalability work. Its structure, reproducibility, entity disjointness, and leakage posture are sound and independently verified.

It is **not** suitable, without qualification, for comparing model *predictive performance* against anything real, because the fraud-generating structure and the entity activity distribution are materially simpler than reality. Any predictive metric computed on it is a synthetic-data metric and must be labelled as such.

**Two follow-ups are recommended before the next phase is relied on for performance work:**

- Build a **full-history real reference** (all 24,386,900 rows) so the 14 excluded history features and the distinguishability test can be settled properly.
- If model-performance comparison is in scope, **regenerate with a fraud mechanism that conditions on the historical entity fraud rates** (computed strictly from past rows), so those features are not inert.

> Validation of this synthetic benchmark does not constitute validation of the fraud-detection system on real-world or institutional financial data.
