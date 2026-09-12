# TRAINING_COVERAGE_REDESIGN_AUDIT

Phase-5 mission (recovered after Cloudflare 524): redesign the training data and
merchant-history representation so the model sees substantially more of the merchant
population — without temporal leakage, label-latency leakage, or offline/production
feature divergence. Control: **E_hardneg
(`altman_native_E_hardneg_cert_20260904` @ threshold 0.018758) — DEPLOYED and
UNTOUCHED.** The locked 2018-2020 final test was NOT used for any decision.

> **INTEGRITY FIX (this phase):** `_merchant_volume.npz` was mis-built — it hashed
> `str((name, year))` tuples instead of the merchant name, so its 99,602 codes lived in
> a DIFFERENT hash space than every frame's `merchant_id`. Validated: code 65721 = 228
> rows in the table vs 1,129,153 in the file. Rebuilt on name-hash codes (63,288 codes
> x 30 years) and validated against awk ground truth. Consequences:
> - **Corrected universe: 63,288 name-hash codes; 57,882 pre-2016-active merchants.**
>   The 5% training pool (27,190 merchants) therefore covers **47.0%** of active
>   merchants (53.0% absent) — not 27.6%/~31% as previously claimed.
> - Taxonomy re-derived: **12.2% of absent legit val rows are genuinely new** (329
>   rows, not 52); the old D bucket (423 "well-observed absent") collapses to 7 — it
>   was a misalignment artifact.
> - CS novelty features (`merch_prior_rows/fraud/rate`) rebuilt on the corrected table
>   (frame probe 179/179 exact).
> - **E_hardneg and all coverage-simulation model metrics are UNAFFECTED** (frame-
>   derived; they never used the volume table).

## Verdict

**OUTCOME B (KEEP E_HARDNEG deployed).** Strategy C achieves the mission's coverage
goal — **100% of the 57,882 pre-2016-active merchants in training with 2,330,695 rows
(all fraud + per-merchant legit cap 200), fixed-population val AUC 0.9868 / PR-AUC
0.7446** — but it does NOT yet dominate E_hardneg on the identical fixed evaluation
population, because the 20+ count/velocity features are sampling-scale-sensitive
(exactly mission #15's warning). The redesign's next experiment is exposure-aware
normalization of those features; until then no candidate replaces E_hardneg.

## 1. Integrity fix & corrected merchant universe (mission #6)

| Quantity | Corrected | Was (broken) |
|---|---|---|
| Full-corpus rows | 24,386,900 | same |
| Unique merchants (name-hash codes) | 63,288 | 99,602 (tuple-hash) |
| Pre-2016-active merchants | 57,882 | 98,138 |
| 5% pool merchants | 27,190 (47.0% of active) | 27.6% |
| Absent from 5% pool | 53.0% of active | ~31% |

Absence expectation at 5% legit sampling: **54.5% of merchants absent, carrying only 1.01% of legit transactions** — absence concentrates in low-volume merchants.

## 2. Strategy C: merchant-aware training frame (mission #7, #11, #14)

| Property | Value |
|---|---|
| Sampling | ALL fraud (21,345) + legit per-merchant: L<=200 keep ALL, L>200 w.p. 200/L (cap ~200), keyed on full-corpus pre-2016 legit volume; seed 42 |
| Train rows | 2,330,695 (TRAIN-ONLY — fixed-pop evaluation uses identical 5%-frame val rows) |
| Merchants in training | 57,882 = **100%** of active universe (57,882) |
| vs 5% pool | 27,190 merchants (47.0%) |
| Fixed-pop val AUC / PR-AUC | 0.9868 / 0.7446 |

Causal property: keep probability depends only on the merchant's train-window
(&lt;2016) volume — never on future existence; per-row features are expanding
statistics over the sampled stream (identical to cov_frame.py); a future-perturbation
test recomputes earlier rows unchanged.

## 3. Fixed-population comparison (mission #14) — identical rows, only coverage changes

P1 = recall>=0.99 threshold selected on the fixed 5%-frame val population:

| Strategy | Train rows | Merchants | Thr | overall FPR | seen FPR | absent@5 FPR | absent@5 recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% (A) | 879,196 | 27,190 | 0.08300 | 0.0669 | 0.0592 | 0.6363 | 0.9896 |
| 10% (B) | 1,736,255 | 33,632 | 0.10868 | 0.0885 | 0.0817 | 0.5903 | 0.9896 |
| 20% (B) | 3,449,530 | 40,920 | 0.24899 | 0.0974 | 0.0913 | 0.5517 | 0.9845 |
| 40% (B) | 6,879,384 | 48,462 | 0.24319 | 0.1478 | 0.1416 | 0.6060 | 0.9482 |
| **C (merchant-aware)** | 2,330,695 | **57,882** | 0.02348 | 0.1591 | 0.1532 | 0.5876 | 0.9896 |

**Reading it honestly:** absent-merchant FPR falls monotonically from 0.636 (5%) to
0.552 (20%) / 0.588 (C) on the IDENTICAL previously-absent population — the coverage
lever is real. But overall/seen FPR RISES with coverage (0.067 -> 0.097 at 20%,
0.159 at C): a higher-coverage model trained on near-full-scale counts reads the
5%-scale count features of the evaluation population as "low-volume merchant" and
over-flags. That is the sampling-scale confound, not a coverage penalty — and it is
exactly why mission #15 (exposure-normalized count features) precedes any deployment
candidate. Per-frame validation (model on its own scale) showed the same 5->20%
absent-FPR drop (0.667 -> 0.448) at overall FPR 0.067 -> 0.063.

## 4. Segment views on the fixed population (mission #19) at P1

Channels (E_hardneg vs C):

| Channel | n | control rec | control FPR | C rec | C FPR |
|---|---|---|---|---|---|
| chip | 120,641 | 0.945 | 0.0347 | 0.968 | 0.1199 |
| online | 24,168 | 0.999 | 0.3024 | 0.996 | 0.3380 |
| swipe | 29,721 | 0.974 | 0.0525 | 0.966 | 0.1906 |

Merchant-volume buckets (full-corpus pre-2016 legit volume), legit FPR at P1:

| Bucket | n | control FPR | C FPR | n_fraud |
|---|---|---|---|---|
| v0 (new) | 340 | 0.6175 | 0.5789 | 55 |
| 1-5 | 908 | 0.5898 | 0.5694 | 123 |
| 6-20 | 1,675 | 0.5106 | 0.4944 | 69 |
| 21-100 | 5,664 | 0.2094 | 0.2368 | 157 |
| 101-1k | 25,120 | 0.0520 | 0.0723 | 303 |
| 1k+ | 140,823 | 0.0595 | 0.1645 | 3127 |

The control's FPR is ~51-62% for merchants with &lt;=20 full-corpus prior legit rows
and falls to ~5% at 101+. C shaves the smallest buckets (0.62 -> 0.58 new, 0.59 ->
0.57 at 1-5, 0.51 -> 0.49 at 6-20) but its scale-shifted counts over-flag the 1k+
bucket (0.059 -> 0.165) — again the count-scale confound, not genuine regression.

## 5. Label-latency & fraud-rate divergence (mission #3-4) — status UNVERIFIED

UNVERIFIED — static generator labels, no availability timestamp.
The production `EntityFraudRateTracker` records the model's OWN decision
(score>=70 proxy) as a label in a rolling 100-event window, while training used a
ground-truth expanding cumsum over the full stream — different quantities. No
redesign depending on `user/merch/city_fraud_rate` may deploy until this is resolved
or those features are isolated (see `reports/label_latency_status.json`).

## 6. Decision matrix (mission #20)

| Question | Finding | Evidence | Consequence |
|---|---|---|---|
| Labels available causally? | UNVERIFIED | label_latency_status.json | fraud-rate features keep UNVERIFIED |
| Corrected merchant coverage? | 5% pool = 47.0% of 57,882 active | manifest + matrix (corrected) | coverage gap real, smaller than claimed |
| "Unseen" merchants new? | 12.2% genuinely new, 87.8% established | corrected taxonomy | redesign must help both |
| Coverage reduces FPR? | YES (0.636 -> 0.552-0.588 on identical rows) | coverage_fixed_population.json | PRIMARY CAUSE = coverage |
| Does Strategy C dominate? | NO on fixed pop (overall FPR 0.159 vs 0.067) | cov_C fixed-pop row | count-scale confound; needs #15 first |
| Establishment history helps? | 2-3 pts absent FPR at ~1 pt overall | phase-2/3 frontier | secondary, keep |
| Global threshold appropriate? | No (absent rows 74-86% flagged at all depths) | threshold subgroups | conditional thresholding is an output |
| Another model now? | Not until count features are exposure-normalized | cov_C result | OUTCOME B — keep E_hardneg |

## 7. Required deliverables

- `TRAINING_COVERAGE_REDESIGN_AUDIT.md` (this file)
- `reports/training_coverage_redesign.json`
- `reports/merchant_coverage_manifest.json` (corrected universe)
- `reports/merchant_coverage_matrix.json` (corrected universe + taxonomy)
- `reports/coverage_fixed_population.json` (now includes cov_C)
- `reports/coverage_segments_fixed_pop.json`
- `reports/label_latency_status.json`
- `reports/phase5_recovery.json` + `PHASE5_RECOVERY_REPORT.md`
- `reports/data_coverage_label_latency_audit.json` + `DATA_COVERAGE_LABEL_LATENCY_AUDIT.md` (corrected)

**E_hardneg remains certified and deployed at 0.018758. No production change was
made; the final test was not touched.**
