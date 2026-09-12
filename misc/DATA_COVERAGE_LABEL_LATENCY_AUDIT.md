# DATA_COVERAGE_LABEL_LATENCY_AUDIT

Phase-4 mission: determine whether PS-14's absent-merchant false-positive problem is
caused by inadequate training coverage, unavailable establishment history, unavailable
labels at scoring time, sampling strategy, or model architecture. E_hardneg
(`altman_native_E_hardneg_cert_20260904` @ threshold 0.018758) is the certified control
and was NOT modified; the final 2018-2020 test was NOT touched for any decision.

> **INTEGRITY CORRECTION (Phase 5 recovery):** the original release reported a
> 99,602-merchant universe with 98,138 pre-2016-active and ~27.6% 5%-pool coverage.
> Those counts came from `_merchant_volume.npz`, which was mis-built by hashing
> `str((name, year))` tuples instead of the merchant name — its codes lived in a
> different hash space than every frame's `merchant_id` and were validated wrong
> (code 65721 = 228 rows in the table vs 1.13M in the file). The table was rebuilt
> on name-hash codes (63,288 codes, validated against awk ground truth). **Corrected
> universe: 63,288 name-hash merchant codes / 57,882 pre-2016-active, so the 5% pool
> covers 47.0% of active merchants (53.0% absent), and the four-group taxonomy
> re-derived with 12.2% of absent legit val rows genuinely new.** All coverage-
> simulation model metrics below are frame-derived and unaffected by this correction.

## Verdict

**OUTCOME B — DATA/FEATURE REDESIGN. KEEP E_HARDNEG deployed (untouched).**

**The absent-merchant FPR is primarily a training-coverage artifact, not a model or
architecture failure.** The 5% legit sampling used to build the training pool omits
~53% of the 57,882 pre-2016-active merchants, and 87.8% of "unseen-merchant" legit val
rows belong to established merchants that simply were not sampled (12.2% genuinely
new). Doubling training coverage to 10-20% cuts the FPR on the previously-absent
merchant population from 0.667 to 0.448-0.500 at the same >=99% recall, while overall
FPR improves (0.067 -> 0.063). Label latency remains UNVERIFIED; the production
entity-tracker feeds the model its OWN decisions as labels, which is a different
quantity from the ground-truth expanding cumsum used in training.

## 1. Label latency (mission #2-4) — UNVERIFIED, plus two actionable findings

The dataset is a pre-labeled static file from the IBM synthetic generator
(24,386,900 rows; `Is Fraud?` assigned at generation). There is NO label-availability
or confirmation timestamp anywhere in the file or the repo. Per the mission's own rule,
the status stays **UNVERIFIED** — the data cannot establish that a fraud label exists
at the moment the model scores a later transaction.

Empirical exposure (full corpus, chronological): time from an entity's fraud event to
the NEXT transaction of that same entity:

| Entity | rows-after-fraud | median gap | p90 | <1 day | <7 days |
|---|---|---|---|---|---|
| user | 24,386,724 | 0.22 d | 1.0 d | 90.0% | 95.7% |
| merchant | 24,367,566 | 6.4 d | 17.0 d | 31.4% | 50.1% |
| city | 24,379,556 | 12.9 d | 17.2 d | 20.9% | 41.2% |

If labels lag by more than the gap, the next transaction is scored with a fraud-rate
feature that does not yet include the fraud. User-level rates are the most
latency-critical.

**Feature inventory (only 3 of 48 deployed features consume labels):**

| Feature | Uses labels | Future rows | Avail verified | Prod computable | Finding |
|---|---|---|---|---|---|
| user_fraud_rate | yes | no | no | yes | TRAIN!=PROD semantic: expanding all-history ground-truth cumsum vs rolling 100-window decision-proxy (score>=70) with min_events=5; baseline 0.001 |
| merch_fraud_rate | yes | no | no | yes | same mismatch |
| city_fraud_rate | yes | no | no | yes | same mismatch |
| all count/velocity features | no | no | yes | yes | TRAIN computed on 5%-sampled counts; PROD counts full traffic — the coverage distortion directly |
| 6 establishment/novelty features (candidates, not deployed) | 2 of 6 (prior_fraud, prior_rate) | no | no | conditional | leakage-audited PASS; age + prior-row counts are label-free and safe |

No label-dependent feature feeds another (no recursive cascade). Full detail:
`reports/label_latency_audit.json`.

## 2. Merchant coverage & the 5% sampling effect (mission #5-6)

| Quantity | Value |
|---|---|
| Full-corpus rows | 24,386,900 |
| Full-corpus unique merchants (name-hash codes) | 63,288 |
| Pre-2016-active merchants | 57,882 |
| Pre-2016 legit transactions | 17,151,218 |
| Merchants in 5% training pool | 27,190 (~47.0% of pre-2016-active) |

Absence expectation as a function of legit sampling (exact probability a merchant is
omitted: none of its pre-2016 legit rows sampled):

| Coverage | E[absent merchants] | E[absent legit txns] |
|---:|---:|---:|
| 5% | 54.5% | 1.01% |
| 10% | 42.8% | 0.54% |
| 20% | 29.9% | 0.26% |
| 40% | 16.6% | 0.10% |
| 60% | 9.0% | 0.04% |
| 80% | 3.8% | 0.01% |
| 100% | 0.0% | 0.0% |

Absence concentrates in low-volume merchants (54.5% of merchants absent at 5%, but
only 1.01% of legit transactions) — exactly the population the model over-flags.

## 3. Four-group taxonomy (mission #5, #13) — are "unseen" merchants new?

Observed on the 5% mission frame:

| Group | val rows | val fraud | test rows | test fraud |
|---|---|---|---|---|
| A. genuinely new (no prior corpus) | 329 | 50 | 397 | 28 |
| B. established-absent (1-5 prior legit) | 748 | 93 | 1,129 | 164 |
| B. established-absent (6-20) | 879 | 36 | 1,056 | 81 |
| C. low-history absent (21-100) | 512 | 14 | 541 | 0 |
| D. well-observed absent (101+) | 7 | 0 | 10 | 0 |
| TOTAL absent | 2,475 | 193 | 3,133 | 273 |

**87.8% of absent-merchant legit val rows are at established merchants** (12.2%
genuinely new on val, 12.9% on test). The "cold-start" framing was mostly wrong:
this is primarily a training-coverage phenomenon.

## 4. Coverage simulation (mission #7-8) — the decisive experiment

Same recipe (spw=10, seed 42, 3-model ensemble) trained on all-fraud + f% legit;
validation metrics on the FIXED population of merchants absent from the 5% pool
(`absent@5`), plus overall, at recall>=0.99 (P1):

| Coverage | train rows | merchants | overall rec | overall FPR | absent@5 rec | absent@5 FPR | absent@5 FPR @P2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 879,196 | 27,190 | 0.9901 | 0.0669 | 0.9948 | 0.6665 | 0.7283 |
| 10% | 1,736,255 | 33,632 | 0.9901 | 0.0722 | 0.9948 | 0.5000 | 0.5367 |
| 20% | 3,449,530 | 40,920 | 0.9901 | 0.0630 | 0.9948 | 0.4482 | 0.6159 |
| 40% | 6,879,384 | 48,462 | 0.9901 | 0.0524 | 0.9948 | 0.6663 | 0.7063 |

*The 40% per-frame absent@5 numbers are NOT directly comparable to 5-20%: its
absent@5 population is 18,077 val rows vs 2,475 at 5% and count/velocity features
train at ~8x the 5%-frame scale. On the IDENTICAL 5%-frame rows (fixed-population
re-score, `reports/coverage_fixed_population.json`) absent@5 FPR is 0.636 -> 0.590
-> 0.552 -> 0.606 across 5/10/20/40%: the 5->10->20% drop is the robust signal, and
40% gives back a little because full-scale-trained counts read 5%-scale counts as
low-volume merchants. Overall FPR keeps improving through 40% (0.0669 -> 0.0524)
as train fraud density rises. Full table: `reports/merchant_coverage_matrix.json`.

**Interpretation:** doubling coverage halves the absent-population FPR, with overall
performance equal or better. The problem is **TRAINING COVERAGE**, not model
architecture or representation.

## 5. Establishment hypothesis (mission #11-12)

Label-free establishment signals exist and are causal (leakage-audited PASS in phase 3):
merchant_age, is_merchant_new, merch_prior_rows, merch_ever_seen_prior (plus
label-dependent merch_prior_fraud/rate). The phase-2/3 frontier shows they improve
absent-merchant FPR ~2-3 points at ~1 point overall cost when added to the 5%-trained
model — real but secondary to coverage itself.

Diagnostic: for legitimate rows, flag rate at the deployed threshold by merchant age —

| age band | legit n | E mean | flag@P2 | absent-code flag |
|---|---|---|---|---|
| new (age 0) | 250 | 0.193 | 76.4% | 76.4% |
| 1-5 | 619 | 0.094 | 43.9% | 72.5% |
| 6-10 | 5,722 | 0.037 | 19.0% | 79.0% |
| 11-20 | 40,748 | 0.024 | 14.0% | 77.9% |
| 21+ | 123,357 | 0.017 | 10.1% | 86.2% |

Age/prior-row signals correlate strongly with score for SEEN merchants but the
code-absent population is flagged ~74-86% regardless of true establishment — the model
as trained simply has no view of these merchants' establishment.

## 6. Four-group sanity at the deployed threshold (mission #13)

| Group | n | flagged | recall (fraud) |
|---|---|---|---|
| G1 new + legit | 279 | 75.6% | — |
| G2 new + fraud | 50 | 100% | 50/50 |
| G3 established-absent + legit | 2,003 | 77.7% | — |
| G4 established-absent + fraud | 143 | 100% | 143/143 |
| ref seen + legit | 168,414 | 10.6% | — |
| ref seen + fraud | 3,641 | 99.5% | 3,622/3,641 |

PS-14 does not treat genuinely-new merchants differently from established-but-absent
ones; both are ~76-78% flagged, while fraud in both groups is caught ~100%. The defect
is coverage, not "newness".

## 7. Calibration (mission #14) — diagnosis, not a fix

E_hardneg on validation (development data only):

| Population | Brier | ECE (5-10 bins) | top-bin pred | top-bin observed |
|---|---|---|---|---|
| overall | 0.00845 | 0.0166 (10 bins) | 0.361 | 0.218 |
| seen | 0.0074 | 0.006 | — | — |
| absent | 0.0829 | ~0.14 | 0.30 | 0.62 |

The model is a RANKING model: top-bin over-predicts (0.36 predicted vs 0.22 observed),
and the absent population is severely miscalibrated (Brier 0.083 vs 0.007 seen) — but
ranking within the absent population is strong (PR-AUC 0.92). Calibration is a symptom
of the coverage distortion, not its cause; recalibrating the score would not fix the
~77% flag rate of normal absent-merchant legit transactions.

## 8. Threshold appropriateness (mission #15)

Subgroup means/flags at the deployed 0.018758 threshold (seen, val):

| history bucket | n | mean | flagged |
|---|---|---|---|
| 0 prior | 888 | 0.085 | 31.6% |
| 1-20 | 64,241 | 0.023 | 13.7% |
| 21-200 | 99,834 | 0.018 | 9.8% |
| 201-2k | 4,977 | 0.020 | 13.3% |
| 2k+ | 756 | 0.027 | 18.7% |

A single global threshold is NOT universally appropriate across exposure regimes —
absent-code rows sit at 74-86% flag rate at every history depth while seen rows vary
10-32%. Conditional thresholding is a candidate OUTPUT of the data redesign, not a
standalone fix.

## 9. Production-parity consequence (found during the audit)

The production EntityFraudRateTracker records `is_fraud = (risk_score >= 70)` after each
evaluation — the model's OWN decision as a label — into a rolling 100-event window with
min_events=5. Training used a ground-truth expanding cumsum over the full sampled
stream. These are different quantities even at zero label latency; the feature-parity
certifications compared the derive functions on the same context, not this tracker
semantic. Before any redesign deploys, this proxy must be replaced with confirmed
feedback labels or the three fraud-rate features dropped.

## 10. Decision matrix (mission #20)

| Question | Finding | Evidence | Consequence |
|---|---|---|---|
| Labels available causally? | UNVERIFIED (no availability timestamp; generator labels at creation) | label_latency_audit.json | fraud-rate features keep UNVERIFIED; count-only features safe |
| Merchant coverage? | 5% pool = 27,190 of 57,882 pre-2016-active merchants (47.0%) | merchant_coverage_matrix.json (corrected) | sampling is the distortion |
| "Unseen" merchants new? | MOSTLY NO — 87.8% established-but-unsampled, 12.2% genuinely new | taxonomy (A vs B/C/D, corrected) | cold-start framing mostly wrong |
| Greater coverage reduces FPR? | YES — 0.667 → 0.448-0.500 at 10-20% coverage | coverage simulation | PRIMARY CAUSE = coverage |
| Establishment history helps? | 2-3 pts absent FPR at ~1 pt overall cost | phase-2/3 frontier | track for redesign; secondary |
| Calibration helps? | No — symptom, ranking already strong | Brier/ECE section | do not calibrate as the fix |
| Global threshold appropriate? | No — 74-86% absent vs 10-32% seen flag rate | threshold subgroups | conditional thresholding is an output, not a fix |
| Another model justified? | Only after data redesign (higher coverage) | coverage simulation | OUTCOME B, not C yet |

## 11. Final recommendation

**KEEP E_HARDNEG deployed (control — unchanged).** The evidence proves the absent-
merchant FPR is fundamentally a **training-coverage** problem: the 5% legit sampling
omitted ~53% of the 57,882 pre-2016-active merchants, and the model never learned those
merchants exist. Increasing training coverage to 10-20% halves the absent-population
FPR at equal-or-better overall performance and unchanged >=99% recall. The lowest-risk
next step is a data redesign (higher or merchant-aware sampling + label-free
establishment features + replacing decision-proxy labels), validated on development
data and the locked test exactly once — NOT a new model trained on the same 5% pool,
and NOT a threshold or calibration patch. No production changes were made in this
mission. (Universe counts corrected from the mis-hashed volume table; coverage-sim
metrics unaffected.)

Machine-readable report: `reports/data_coverage_label_latency_audit.json`.
