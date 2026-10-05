# Phase Closeout — 50M Synthetic Benchmark Validation

**Validation status: `VALIDATED WITH LIMITATIONS`**

**Date:** 2026-10-05
**Repository:** PS-14-Fraud-Detection-software
**Git SHA:** `5a5ff55cc03df73318fdf31a16e3665f159a4720` (unchanged; nothing committed)
**Validation report:** [`SYNTHETIC_50M_VALIDATION_REPORT.md`](SYNTHETIC_50M_VALIDATION_REPORT.md)

> Validation of this synthetic benchmark does not constitute validation of the
> fraud-detection system on real-world or institutional financial data.

---

## 1. Benchmark identity validated

| field | value |
|---|---|
| benchmark | `data/synthetic_50m/` (166 monthly Parquet partitions) |
| combined partition hash | `c2db7142e24dd8d9c2806ef8a1ec0c23e62cbf2d8fcc4b4b614396922b47edf1` |
| schema hash | `fb1d8e6a19ffa5837a28747bd928a6c40d4111a6c36eb6d89049bde620e09d96` |
| generator version | `1.0.0` |
| seed | `20261005` |
| source dataset | IBM v2, `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |

## 2. Exact row count

**50,000,000 data rows**, confirmed by a full streaming pass over every partition.

## 3. Validation status

`VALIDATED WITH LIMITATIONS` — suitable for storage, pipeline, throughput and
training-scalability testing. **Not** suitable for unqualified model-performance
comparison against reality.

## 4. Major validation findings

**Sound:**
- Exactly 50,000,000 rows; no duplicate/unexpected columns; identical schema across all 166 partitions; 0 nulls and 0 non-finite values across 48 features; label domain `{0,1}`.
- Prevalence 0.00122218 vs target 0.0012202 (**+0.16%**), stable across all 166 monthly segments (spread 0.00029).
- Past-only historical aggregates **independently reconstructed with 0 mismatches over 301,205 rows**.

**Material limitations found:**
1. **Historical entity fraud rates are inert in the benchmark.** `city_fraud_rate` scores 0.5195 AUC against the label here, versus **0.9155** in the real data; `merch_fraud_rate` 0.5066 vs 0.7847; `user_fraud_rate` 0.5093 vs 0.7297. These are the strongest real fraud predictors, and the declared label mechanism deliberately does not consume them. A model trained on this benchmark would learn to ignore them.
2. **The MCC–channel relationship is destroyed.** Real data has `mcc` × `is_online` ρ = −0.443 (online transactions carry low MCC codes); the benchmark has −0.001. The generator samples channel and MCC independently.
3. **Two of 48 features are constant.** `mcc_travel` and `mcc_online` are always 0 because the MCC sampler draws only from the 20 most frequent observed codes, none in those bands. (`has_zip`/`has_state` are constant 1 for a legitimate reason — IBM v2 has no blanks.)
4. **Entity activity is near-uniform, not heavy-tailed.** Per-user transaction counts span 11,842–12,623 (6.6%); the top 1% of users hold only 1.0% of rows. Card-testing and burst behaviour cannot be exercised.

## 5. Leakage status

**CLEAN.** No metadata field, identifier, or ordering artefact encodes the label.

| attack | AUC |
|---|---|
| row position / generation order | 0.5113 |
| `year_month` partition key | 0.5155 |
| `user_id` | 0.5057 |
| `split` membership predicted from features | 0.5053 |
| max single-feature AUC vs label | 0.7205 (`amt_sq` — the declared amount signal) |

Labels are not derived from any model prediction; the generator imports no model, weights, or threshold.

## 6. Contamination status

**CLEAN.** 0 of 4,101 users appear in both splits — entity-disjointness verified mechanically rather than by seed assumption. `split` membership is unpredictable from the 48 features (AUC 0.5053), so no deterministic generator artefact makes split membership predictable. No source record is copied into the benchmark; all rows are `SYNTHETIC`, and metadata columns sit outside the feature list.

## 7. Reproducibility status

**VERIFIED.** 8 randomly sampled partitions re-hashed against the manifest: **8/8 match, 0 mismatch**. Combined partition hash unchanged from the identity recorded at the start of validation. Full regeneration from the seed was verified in the construction phase (partition `2002-09` reproduced byte-identically).

## 8. Scientific suitability

| purpose | status |
|---|---|
| Large-scale storage testing | **ESTABLISHED** |
| Data-pipeline testing | **ESTABLISHED** |
| Throughput testing | **ESTABLISHED** |
| Training scalability testing | **ESTABLISHED** |
| Model-performance comparison | **LIMITED** |
| Real-world fraud effectiveness | **NOT ESTABLISHED** |
| Institutional validation | **NOT ESTABLISHED** |

## 9. Limitations

1. All 50M rows and all fraud labels are synthetic; nothing here is evidence about real fraud.
2. Fraud structure is materially simpler than reality (finding 1 above).
3. MCC–channel interaction absent; `mcc_travel`/`mcc_online` untestable.
4. Entity activity near-uniform rather than heavy-tailed.
5. **Real-vs-synthetic distinguishability measured at CV AUC 1.0000, but that result is CONFOUNDED** and must not be cited as proof of unrealisticity. IBM v2 is sorted by `User`, so full-history velocity features require reading all 24,386,900 rows; the reference used here is a strided sample whose real-side velocities are understated by construction. Leave-one-feature-out leaves AUC at 1.0000 for every feature removed, so the separation is redundant rather than attributable to one axis.
6. Fourteen history-depth features were excluded from the head-to-head comparison for the same reason.
7. Amount right tail heavier than IBM (p99 422 vs 314); error rate ~57% higher by declared CNP assumption.

## 10. Defects found in validation tooling (fixed)

Recorded for honesty; all three were bugs in this phase's own tooling, not in the benchmark:

1. `dict.update()` replaces rather than accumulates, so a month spanning two 250k batches recorded only the last batch's row count — inflating reported per-month prevalence ~6× and producing a **false PASS** on the temporal check. Fixed; per-month prevalence is now internally consistent.
2. The real-reference builder marked `err = 1` for every row because `NaN` became the literal string `"nan"` (the runtime contract guards exactly this). Fixed; real error rate now 0.019 vs IBM raw 0.0154.
3. The real reference initially read a prefix of a user-sorted file, covering ~100 of 2,000 users. Fixed to strided sampling.

## 11. Repository verification

| check | result |
|---|---|
| full test battery | **84 PASS / 0 FAIL** |
| `review_package_check.py` | rc=0 |
| `review_resolution_check.py` | rc=0 |
| `claim_evidence_check.py` | rc=0 |
| `eval_record_test.py` | rc=0 |
| `check_freeze_test.py` | rc=0 |
| `scripts/check_freeze.py` | **rc=1 / 78 problems** — expected pre-freeze state preserved |
| IBM v2 dataset | `b01fa323…` **UNCHANGED** |
| Research Plan | `eab5a061…` **UNCHANGED** |
| Preregistration | `fdde71cb…` **UNCHANGED** |
| production manifest | `e6da1a43…` **UNCHANGED** (4 model artifacts present) |
| `native_features.py` | `5760a203…` **UNCHANGED** |

No model was tuned, no threshold changed, no architecture altered, no
preregistration or Research Plan edit, no reviewer approval manufactured.

## 12. May Phase 3 begin?

**YES — for scale, storage, pipeline, throughput and training-scalability work.**

Phase 3 (Large-Scale Model Testing) may begin, subject to:

- Predictive metrics computed on this benchmark are **synthetic-data metrics** and must be labelled as such wherever reported.
- Model-performance comparisons against real-world effectiveness are **out of scope** and must not be drawn from this benchmark.

**Two follow-ups are recommended before relying on the benchmark for performance work:**

- Build a **full-history real reference** (all 24,386,900 rows) to settle the 14 excluded history features and the confounded distinguishability result.
- If model-performance comparison is in scope, **regenerate with a fraud mechanism that conditions on the historical entity fraud rates** (strictly from past rows), so those features are not inert.
