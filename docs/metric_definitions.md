# Metric Definitions (Authoritative — v1.0)

`METRIC_DEFINITIONS_VERSION = "1.0"` in `backend/scripts/metric_definitions.py`
is the single source of truth for every metric reported in this repository.
The human description below mirrors the code; where they disagree, the code is
authoritative and this file must be updated.

All metrics assume binary classification with **positive class = fraud = 1**.
Positive prediction at threshold `t`: `score >= t`.

## Ranking metrics (threshold-free)

| Metric | Definition | Notes |
|---|---|---|
| ROC-AUC | Probability a uniformly chosen positive is ranked above a uniformly chosen negative; ties earn 0.5. | Computed with `sklearn.metrics.roc_auc_score` (trapezoidal). Undefined (NaN) if the split has a single class — reported as NaN, never as 0. |
| PR-AUC | Area under the precision–recall curve via `average_precision_score` (step-preserving summary, not trapezoidal interpolation). | Random-classifier expectation equals prevalence. Never compare PR-AUC across datasets with different prevalence without stating the baselines. |

## Operating-point metrics (threshold-dependent)

At threshold `t`, with TP/FP/TN/FN counted under `score >= t`:

| Metric | Numerator / denominator |
|---|---|
| Recall (TPR) | TP / (TP + FN); 0.0 when the split has no positives |
| Precision | TP / (TP + FP); 0.0 when nothing is predicted positive |
| FPR | FP / (FP + TN); 0.0 when the split has no negatives |
| FNR | FN / (TP + FN) |
| F1 | Harmonic mean of precision and recall at `t` |

Ties: `>=` means scores exactly at the threshold are predicted positive.
Missing predictions: NaN labels or scores are **rejected** — the calculators
raise rather than silently dropping or imputing.

## Recall@1%FPR — threshold discipline

1. The threshold is selected on the **validation split only**
   (`threshold_at_fpr_on_validation`) as the grid point whose validation FPR is
   closest to the 1% target.
2. The selected threshold is applied **unchanged** to the test/evaluation split.
3. Selecting the threshold against evaluation labels and then reporting the
   same split's recall as unbiased performance is **test-set threshold tuning**
   and is forbidden. `backend/scripts/evaluate.py` refuses configurations whose
   `threshold_source` is `test`/`external`/`holdout`, and every evaluation
   record states `threshold_source` (`validation`, `fixed`, or `none`) so
   reviewers can verify the discipline.

## Class-imbalance reporting (mandatory in every report)

Every evaluation report must include: total observations, positive count,
negative count, and prevalence. Accuracy is at most a **secondary** metric: at
prevalence *p*, a trivial always-legitimate classifier achieves accuracy
`1 − p`, so accuracy must never be presented as the headline number for fraud
data.

## Confidence intervals

- Method: **percentile bootstrap**, resampling unit = individual observation,
  stratified by class (each replicate keeps the original positive/negative
  counts).
- Defaults: 1,000 replicates, 95% level, seed 42 — all recorded in the report.
- Applied to: ROC-AUC, PR-AUC, recall at the stated operating point.
- **Withholding rule:** if a conditioning cell (positives for recall/PR-AUC,
  negatives for FPR/ROC-AUC) has fewer than 30 observations, or a majority of
  replicates are undefined, the CI is withheld (`ci_low = ci_high = null`) with
  an explicit `withheld_reason`. No fabricated precision on small samples.

## Baselines

- `majority_class`: always predict the majority class (recall 0, accuracy
  `1 − p`).
- `random_scores`: uniform random scores with a 0.5 threshold; expected
  ROC-AUC 0.5, PR-AUC ≈ prevalence.
- Baselines are evaluated on the **same eligible split** with the same metric
  implementations. A model-vs-baseline comparison does not by itself assert
  statistical significance.

## Minimum sample conditions

- Confidence intervals: ≥ 30 observations in the limiting class (see above).
- Reports on data with < 50 positives or < 50 negatives must carry the
  generated `SMALL_POSITIVE_CLASS` / `SMALL_NEGATIVE_CLASS` warnings.
- Prevalence < 1% triggers a `HEAVY_IMBALANCE` warning and the report must
  foreground PR-AUC and the stated operating point, not accuracy.

## Consistency guarantee

`backend/scripts/eval_record_test.py` pins: tie handling, `>=` threshold
semantics, random PR-AUC ≈ prevalence, validation-only threshold selection,
NaN rejection, and CI withholding. A metric refactor that changes any of these
semantics must fail these tests — historical interpretation may not drift
silently. Metric semantics are versioned: records carry
`metric_definitions_version`, and comparisons across versions are not
automatically meaningful.
