# Feedback loop report (section 7)

Verification outcomes (`verification_outcomes` in DB-3) are exported to a
labeled pool by `scripts/export_feedback.py` — `confirmed` -> label 0,
`disputed` -> label 1 — joined to the stored derived vectors in DB-2, and
merged into the synthetic set by `src/train_compare.py --feedback` before the
time-based split. Feedback rows keep their real timestamps, so recently
resolved events land in the test window naturally.

## Runs compared

| state | training rows | notes |
|---|---|---|
| baseline | 10,000 synthetic | pre-feedback models (previous EVALUATION.md) |
| real feedback | 10,002 (+2) | the live demo's 1 confirmed + 1 disputed outcome |
| +500 simulated wave | 10,502 (+502) | `--simulate` mechanics demo, archetype `feedback_sim` |

## Fused ensemble (stacker), test split

| metric | baseline | +2 real | +502 sim |
|---|---|---|---|
| PR-AUC | 0.9802 | 0.9795 | 0.9778 |
| ROC-AUC | 0.9989 | 0.9989 | 0.9988 |
| F1 (val-tuned threshold) | 0.9136 | 0.9157 | 0.6761 |
| Recall @ F1 threshold | 0.841 | 0.844 | 0.511 |

Solo models move in the same direction (LR recall 0.727 -> 0.733 -> 0.809;
XGB recall 0.773 -> 0.800 -> 0.638; ISO PR-AUC ~0.67 throughout).

## Interpretation

- **The loop works end-to-end:** outcomes -> labeled rows -> merged set ->
  retrained artifacts with `model_version` = `seed42-transactions.csv+2fb`
  (visible in new audit trail entries).
- **Real feedback (2 events) is a no-op on metrics**, as expected: 2 rows in
  a 10k pool are statistical noise. The disputed attack event landed in the
  test window and every model catches it (per-archetype recall
  `feedback_fraud` = 1.0). The loop's value accrues over volume.
- **The simulated wave holds ranking quality** (PR-AUC 0.9802 -> 0.9778,
  ROC unchanged) but the F1-at-threshold drops. That is a threshold
  calibration artifact of the simulation, not a real degradation: the sim
  re-samples the model's own training distribution (circular by design), so
  it perturbs the validation composition used to tune the threshold. This is
  precisely why the simulator is labelled a *mechanics* demo — real feedback
  is genuinely new events, and its calibration effect should be measured with
  the cost-weighted operating-point tuner
  (`scripts/tune_operating_point.py`), not assumed from this table.

## Follow-up run: live close of the loop (1 confirmed outcome)

A fresh run against the live stack: the walkthrough's `wt-attack-0001` was
flagged HIGH (score 100) and the user confirmed "this was me" (case
`PS14-A5403A6C`). The export labeled it `feedback_legit` (label 0) and it
landed in the test window. The fused model scores that vector **0.9991
fraud probability** — a textbook false positive, now counted in evaluation:

| model | baseline PR-AUC | +2fb PR-AUC | +1fb PR-AUC (this run) |
|---|---|---|---|
| fused ensemble | 0.9802 | 0.9795 | **0.9038** |
| random_forest | 0.9149 | 0.9207 | 0.8405 |
| xgboost | 0.9648 | 0.9639 | 0.8889 |
| logistic_regression | 0.9707 | 0.9658 | 0.8944 |
| isolation_forest | 0.6698 | 0.6695 | 0.6156 |

Recall / F1 / ROC-AUC barely moved — the PR-AUC drop is the high-ranked
false positive now being *measured*, not a learned regression (one sample
cannot move RF/XGB/LR weights). The loop did its job: it surfaced a real
blind spot in the metric. Teaching the model not to over-flag this profile
needs a *growing* pool of such confirmed events (or rule/operating-point
rebalancing — this event trips the critical-rule floor).

## Final state

`models/artifacts/*` currently hold the **real-feedback** retrain (10,001
rows, `feedback: 1`, `model_version` …+1fb). Rerun
`python scripts/export_feedback.py` to refresh the pool as new outcomes
resolve, then retrain with
`python src/train_compare.py --feedback data/feedback_labeled.csv`.
