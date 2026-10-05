# Synthetic 50M Scale Benchmark — Specification

**Status:** DRAFT SPECIFICATION — written *before* generation, per §4.
**Created:** 2026-10-05
**Repository:** PS-14-Fraud-Detection-software @ `5a5ff55`
**Specification version:** 1.0

> This document defines the benchmark *before* any rows are produced. The
> generation rules below are fixed now and are not chosen to make any model
> score well. No model is executed during generation.

---

## 1. Purpose

To provide a **50,000,000-row** tabular benchmark on PS-14's native 48-feature
contract, so that the existing fraud-detection prototype can be stress-tested
and optimised at large scale in a later phase.

The benchmark exists to measure **engineering scalability and behaviour at
volume**. It is a scale harness, not evidence of anything scientific about
real banking fraud.

## 2. Non-purpose (explicit)

This benchmark is **NOT**:

- 50 million real financial transactions — **every row is synthetic**;
- evidence that the model generalises to real banking data;
- institutional validation of fraud-detection effectiveness;
- a substitute for the independent human review tracks;
- a licence to tune the production model, change thresholds, or change the
  Research Plan.

The 50,000,000 figure counts **generated** rows. It must never be reported as
observed data volume.

## 3. Source dataset (empirical foundation)

| field | value |
|---|---|
| path | `data/credit_card_transactions-ibm_v2.csv` |
| SHA-256 | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |
| columns | 15 (`User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,Zip,MCC,Errors?,Is Fraud?`) |
| data rows | 24,386,900 |
| fraud rows | 29,757 (prevalence 0.0012202) |
| entities | 2,000 users · 100,343 merchants · 13,429 cities · 223 states |
| amount | mean 43.634, σ 82.022, p1 −96.00, p50 32.19, p75 66.83, p99 313.91, p99.9 950.96 |
| channel | Swipe 63.10% · Chip 25.78% · Online 11.12% |
| errors | 374,543 rows (≈1.54%), dominated by `Insufficient Balance` (242,783) |
| missingness | `Zip` blank 0.00% · `Merchant State` blank 0.00% |

The source SHA-256 equals `training_dataset_sha256` in
`models/production/manifest.json` — an independent cross-check that the
foundation is the dataset the production model was trained on.

**Why IBM v2:** the repository's own evidence closure establishes it as the
**only** acquired dataset satisfying the native 48-feature contract (48/48). No
other acquired source reaches the contract without synthetic feature
fabrication, so IBM v2 is the strongest defensible foundation available.

**Measured profile** (reproducible via `scripts/profile_ibm_v2.py`):
`misc/reports/ibm_v2_empirical_profile.json`.

## 4. Observed / derived / synthetic classification

The benchmark ships a per-row `provenance_class` column. Values:

| class | meaning | count in benchmark |
|---|---|---|
| `OBSERVED` | value copied from a real IBM v2 row (entity id, MCC, state, channel label, error label, amount/timestamp) | only in `--smoke` mode; **0 rows in the 50M run** |
| `DERIVED_FROM_OBSERVED` | value computed deterministically from observed statistics (quantile transform, fitted frequency table, fitted entity prior) | 0 as a standalone class; these are *inputs* to synthetic generation |
| `SYNTHETIC` | value produced by the generator's stochastic process | 50,000,000 rows |

In the production 50M run every row is `SYNTHETIC`. The `OBSERVED` class is
implemented and exercised only by the smoke mode, so the distinction is
testable rather than aspirational. The **real dataset is never modified,
appended to, or merged into the benchmark.**

## 5. Scale and entity design

| quantity | observed (IBM v2) | benchmark (50M) | basis |
|---|---|---|---|
| rows | 24,386,900 | 50,000,000 | §4 target |
| users | 2,000 | 4,101 | 50M / 12,193 tx-per-user (observed) |
| merchants | 100,343 | 205,761 | 50M / 243 tx-per-merchant (observed) |
| cities | 13,429 | 27,537 | scaled with merchants |
| states | 223 | 223 | observed support, not scaled |
| MCC codes | 109 | 109 | observed support, not scaled |

Per-entity transaction intensities are held at the observed means, so entity
*degree distribution* is preserved while the population grows. Existing observed
IDs are retained for the first 2,000 users; the remainder are **new synthetic
IDs in a disjoint range** (`SYN_USER_*`, `SYN_MERCH_*`) so that no synthetic row
can be confused with an observed one.

## 6. Feature-generation strategy

The benchmark is emitted on the **explicit 48-feature contract** from
`models/production/manifest.json`, in manifest order. Raw columns are generated
first, then features, so the structure mirrors the real data flow.

Features are produced by the **runtime semantics in
`backend/src/privacy_layer/native_features.py`**, not by the training script —
see §11 for why, which is a material decision.

### 6.1 Channel → chip / online / swipe

Drawn from the observed channel frequencies (Swipe .6310, Chip .2578,
Online .1112), with card-not-present transactions (Online + Chip) more likely
to carry errors — a relationship observed in the source.

### 6.2 Amount

Fitted to the observed amount distribution by **quantile mapping**: synthetic
u(0,1) draws are mapped through the observed empirical quantile function
(built from a 2,000,000-value reservoir, pinned by hash). This preserves
skew, the negative tail (refunds/reversals, p1 = −96.00) and the heavy right
tail (p99.9 = 950.96) far better than a fitted lognormal. Amount is then
scaled per-user by that user's latent spend tendency (lognormal, mean 1.0) and
multiplied by a channel factor (online 1.15, chip 1.0, swipe 0.95) — a
documented, assumption-driven joint structure, **not** an independently
sampled marginal.

### 6.3 Temporal

Start 2002-09-01, 5,000 days (~13.7 y) to 2016-06-25, matching the observed
span. Hour-of-day drawn from the observed 24-bin histogram; day-of-week from
the observed 7-bin histogram. Weekday/weekend and night-hour effects are
inherited from those empirical histograms rather than assumed uniform.

### 6.4 Entities

Users, cards, merchants and cities are assigned so that each user's merchant
and city sets are **nested and persistent** (a user has a home city and a
recurring merchant set), reproducing the observed user-merchant and user-city
diversity rather than sampling every cell independently.

### 6.5 Missingness and errors

`Zip` and `Merchant State` are blank in **0.00%** of observed rows, so
`has_zip = has_state = 1` throughout; `is_online_or_no_state` therefore reduces
to `is_online` and this is recorded as a known degeneracy, not hidden.
Error labels are drawn from the observed 21-value frequency table
(`Insufficient Balance` dominant), with card-not-present transactions
error-inflated.

## 7. Label-generation mechanism (assumption-driven)

**The production model is never executed during generation. No model
prediction, score, or decision is used as a ground-truth label. The generator
contains no reference to the model, its weights, or its threshold.**

Fraud is assigned by an explicit behavioural rule over generator-side latent
factors — deliberately *not* over the 48 emitted features, so the label is not
a trivially invertible function of any single feature:

```
logit = b0
      + b_amt  * z(log|amount|)        # unusually large
      + b_onl  * is_online
      + b_night* is_night
      + b_err  * has_error
      + b_newm * is_new_merchant_for_user
      + b_vel  * z(user_tx_velocity)
      + b_far  * z(distance_from_home_city_profile)
P(fraud) = sigmoid(logit)
```

`b0` is solved by bisection so that expected prevalence matches the observed
**0.0012202**. Latent factors (`newm`, `far`, `vel`) are generator-internal and
are **not** emitted as features; their contribution is only representable
through the legitimate historical aggregates, which is what makes the task
non-trivial in the same way the real task is.

Because the mechanism is assumption-driven, **all fraud labels in this
benchmark are `SYNTHETIC` and are not a reproduction of IBM's fraud process.**

## 8. Leakage controls (normative)

1. **Historical aggregates are strictly past-only.** For every row, the
   running entity state used to build `user_tx_count`, `user_avg_amt`,
   `user_fraud_rate`, `merch_fraud_rate`, `city_fraud_rate`, `merch_tx_count`,
   `card_tx_count`, `user_merchant_diversity`, `user_city_diversity`,
   `user_merch_count`, `amt_vs_user_avg`, `amt_zscore` is the state **strictly
   before that row is folded in**. Within-chunk ordering uses `cumcount`/
   `cumsum`; across-chunk state is carried forward. The row's own label and
   amount are never in its own features.
2. **Global statistics** (global mean/σ used by `amt_zscore` under the trainer
   definition) are computed from a **frozen prior chunk** or the generator
   constants, never from the chunk being emitted.
3. **Label non-reconstructability** is tested, not assumed (§13).
4. **Entity-disjoint evaluation split** is provided as a separate column
   (`split`) assigning users to `train`/`test` disjointly by user, so a
   downstream phase can evaluate on users never seen in training.
5. **Temporal ordering** is monotonic in generation order; no feature for a row
   at time *t* uses any row at time > *t*.
6. **No generator metadata leaks into features.** `provenance_class`, `split`,
   `chunk_index`, and generation timestamps are stored in a **separate
   metadata sidecar**, never as model features.

## 9. Partition design and format

**Format: Parquet** (`pyarrow` 25.0.1), partitioned by month.

- Rationale: columnar, typed, compressed, and supports **predicate/partition
  pruning** so a 50M-row benchmark can be scanned a month at a time without
  loading it all. CSV would be ~2 orders of magnitude larger and require full
  scans; the machine has 7.1 GB free RAM, so in-memory or single-file options
  are not viable.
- Compression: `zstd` level 3.
- Row groups: 1,000,000.
- Partition key: `year_month` (e.g. `2015-03`) — 166 monthly partitions.
- Output root: `data/synthetic_50m/` (gitignored; `data/` is in `.gitignore`).

Schema is explicitly declared (no pandas index column is written). All 48
feature columns are `float32`; `label` is `int8`; `split` and `provenance_class`
are `string`.

## 10. Reproducibility

Fixed `seed = 20261005`. The generator uses **independent per-month child
streams** (`numpy.random.SeedSequence(seed).spawn(n_months)`, 166 months) so
that any single partition can be regenerated in isolation and still be
bit-identical — partition boundaries do not shift the random stream. (Entity
state is initialised from a separate stream at `seed + 1`.)

**Generator version: `1.0.0`** — `scripts/generate_synthetic_50m.py`. Bump this
string on any change to the generation rules; it is recorded in the benchmark
manifest alongside the seed and the source hash.

Recorded in `data/synthetic_50m/manifest.json`:

| field | value |
|---|---|
| generator_version | `1.0.0` |
| git_sha | `5a5ff55…` (recorded at run time) |
| seed | `20261005` |
| source_dataset_sha256 | `b01fa323…` |
| target_rows | `50000000` |
| actual_rows | (recorded) |
| schema_hash | sha256 of the canonical schema string |
| file hashes | per-partition sha256 + a manifest-level hash |

Re-running with the same seed and source hash reproduces byte-identical
partitions (verified in §13).

## 11. Decision: which feature semantics to implement

The repository contains **two different definitions** of several of the same
48 feature names:

- `backend/src/privacy_layer/native_features.py` — documents *shifted,
  past-only* inputs, "the event being scored is never counted in its own
  velocity / fraud rate / average"; `high_amt = amt > user_avg_amt*2`;
  `amt_zscore` is a per-user ratio.
- `backend/scripts/altman_production_train.py` — computes `user_fraud_rate`,
  `merch_fraud_rate`, `city_fraud_rate` by `groupby(label).agg(["mean","count"])`
  over the **whole dataset** (no shift), `user_avg_amt` by
  `groupby("User")["amt"].transform("mean")` (includes the row's own amount),
  `high_amt = amt > 500` (absolute), `amt_zscore` global. It also selects
  features by exclusion and therefore does **not** emit the manifest's 48
  (it yields `amt_x_user_rate`/`amt_x_merch_rate`, which are not in the
  contract, and omits `merchant_id`, `city_id`, `card_id`, the `mcc_*` flags,
  `user_merchant_diversity`, `user_city_diversity`, `user_merch_count`).

**Decision:** the benchmark implements the **runtime contract**
(`native_features.py`), causally. Reasons: (a) it is what the deployed model
actually consumes, so the benchmark tests the real inference path; (b) the
trainer method would embed target leakage into a 50M-row artifact and produce
optimistic metrics for the wrong reason, which §7 declares invalid.

**Recorded, not fixed:** the trainer's leakage and the `target_leakage: false`
claim in `models/production/manifest.json` are *pre-existing repository
findings*. This phase does not alter models, artifacts, thresholds, or the
manifest (§12, §15). They are reported to the human reviewers as a
research-integrity item.

## 12. Out of scope for this phase

No model tuning, no architecture change, no threshold change, no Research Plan
change, no institutional validation, no generalization claim, no confirmatory
Track M, no large-scale model testing.

## 13. Validation plan (executed before acceptance)

**Structural** — exactly 50,000,000 rows; 48 features in manifest order and
name; no index column; no unexpected columns; dtypes as declared; label ∈ {0,1}.

**Statistical** (benchmark vs IBM v2, deviations *reported*, not required to be
zero) — amount quantiles, channel mix, hour and day-of-week histograms, error
rate, fraud prevalence, per-entity intensity, amount–channel joint means,
fraud rate by MCC.

**Integrity** — duplicate `event_id` = 0; no label reconstructable from any
single feature (AUC of a per-feature decision stump vs label must be far from
1.0); train/test user sets disjoint; past-only invariant asserted directly on a
sample; metadata columns absent from the feature list; re-generation from the
recorded seed reproduces partition hashes.

## 14. Limitations (declared up front)

1. All rows and all labels are synthetic; prevalence is *matched by
   construction*, not derived from a real fraud process.
2. The fraud mechanism is a declared logistic rule over latent factors. It is
   assumption-driven and will be easier to learn than real fraud.
3. Per-entity intensity is matched in the mean; the full degree distribution is
   approximated, not replicated.
4. `has_zip`/`has_state` are constant 1 because the source has no blanks; those
   two features carry no signal, as in the real data.
5. The benchmark inherits the repository's train/serve skew (§11) if consumed
   by the current model — the skew is a property of the model, not the data.
6. Negative amounts (refunds) are reproduced only as a distributional tail, not
   as linked refund events.

## 15. Scientific interpretation boundary

Valid statements: *"the system processed 50,000,000 synthetic rows under the
native 48-feature contract"*, *"throughput/memory behaviour at 50M synthetic
rows"*, *"the pipeline's arithmetic and memory profile at scale"*.

Invalid statements: *"the model detects fraud in 50M transactions"*, *"accuracy
on 50M real transactions"*, *"validated at scale"*, *"generalises"*. Any metric
computed on this benchmark is a **synthetic-data metric** and must be labelled
as such wherever reported.
