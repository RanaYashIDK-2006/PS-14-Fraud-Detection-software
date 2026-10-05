# Phase NR-05 — Exploratory Method and Evaluation Diagnostics

**Status:** PASS WITH LIMITATIONS (exploratory diagnostics executed successfully;
scientific conclusions remain EXPLORATORY / NON-INDEPENDENT)
**Research Plan:** `docs/RESEARCH_PLAN.md` — **DRAFT / NOT FROZEN / NOT APPROVED**
**Driver:** `backend/scripts/nr05_diagnostics.py` (seed 42, N_BOOT 300)
**Results:** `reports/nr05/*.json`, artifacts `reports/nr05/artifacts/<experiment>/`
**Evidence:** 12 records in `reports/evaluation_runs/eval_ledger.jsonl` (7 final +
5 superseded harness iterations, preserved append-only), git `5a5ff55cc03df73318fdf31a16e3665f159a4720`
**Classification of every result:** EXPLORATORY — previously exposed datasets;
NOT confirmatory Track M evidence; NOT independent replication.

---

## 1. Objective

Exploratory method-development and diagnostic work on the previously exposed
datasets (ULB, Kaggle fraudTrain/fraudTest, IBM v2) to determine, before the
eventual confirmatory Track M evaluation:

- whether the intended methodology (baseline vs ensemble, calibration, gating,
  temporal/entity-aware evaluation) reproduces cleanly on independently
  structured public datasets;
- how much performance comes from the ensemble versus simpler baselines;
- whether the gating mechanism provides measurable value rather than merely
  rejecting difficult inputs — with Class B (subtle/plausible shifts) as the
  **primary** investigation;
- where dataset-specific feature/label/split differences cause failures;
- what must be frozen before untouched confirmatory execution;
- which results are positive, negative, failed, or inconclusive.

## 2. Scope

Executed (all by `python backend/scripts/nr05_diagnostics.py`, seed 42):

| # | Experiment | Datasets / split |
|---|---|---|
| E1 | `nr05_ulb_discrimination` | ULB temporal 60/20/20 (train 170,884/360 pos; cal 56,961/57; test 56,962/75) |
| E2 | `nr05_kaggle_discrimination` | fraudTrain by unix_time 70/30 train/cal (907,672/5,121; 389,003/2,385), fraudTest = final test (555,719/2,145) |
| E3 | `nr05_kaggle_entity_disjoint` | fraudTrain with 20% of `cc_num` held out entirely; test = held-out-user rows (240,831/1,515) |
| E4 | `nr05_ibm_discrimination` | IBM v2 stride-1/37 deterministic sample (659,105 rows), temporal 60/20/20 (395,463/491; 131,821/185; 131,821/134) |
| E5 | `nr05_ulb_gating` | Class A + Class B perturbations on E1's test window, same model |
| E6 | `nr05_kaggle_gating` | same, E2's test window |
| E7 | `nr05_ibm_gating` | same, E4's test window |

Not executed: any confirmatory Track M experiment; any IEEE-CIS/BAF
acquisition or use; any unrecorded external dataset; any threshold tuning
against test labels; any hyperparameter search.

## 3. Dataset exposure status

Authoritative source: `docs/evaluation/DATASET_EXPOSURE_LEDGER.md` (§3/§14).

| Dataset | Exposure | NR-05 role |
|---|---|---|
| ULB `data/creditcard.csv` (`76274b69…`) | Used during development — EXPLORATORY | exploratory diagnostics only |
| Kaggle fraudTrain/fraudTest (`fd713920…` / `12d553ab…`) | Used during development — EXPLORATORY | exploratory diagnostics only |
| IBM v2 (`b01fa323…`) | Used during development — EXPLORATORY | exploratory diagnostics only |
| IEEE-CIS | CANDIDATE CONFIRMATORY, not acquired | **not used** (BLOCKED) |
| BAF | CANDIDATE CONFIRMATORY, not acquired | **not used** (BLOCKED) |

Nothing in NR-05 makes an exposed dataset "untouched". No dataset file was
modified; all five repository dataset SHA-256 values verified unchanged after
the runs.

## 4. Dataset-specific feature contracts (Track M, dataset-appropriate)

No dataset was forced into the native 21/48-feature contract; no proxy mapping
to the native model was invented. Each representation is NEW-model,
dataset-appropriate, documented in the driver and frozen in the emitted JSON
(`feature_schema_version = nr05_<dataset>_features_v1`):

| Dataset | Features | Label | Temporal field | Entity | Unavailable native features |
|---|---|---|---|---|---|
| ULB | V1–V28, Amount, log₁₊Amount (30) | `Class` | `Time` (seconds from first txn; chronological order only, no calendar semantics) | **none** | all native graph/velocity features — entity-disjoint **NOT APPLICABLE** (no entity id) |
| Kaggle | amt, log_amt, hour, dow, is_weekend, category_code (fit on TRAIN), log_city_pop, haversine distance to merchant, age (9) | `is_fraud` | `unix_time` | `cc_num` | native contract not used by design |
| IBM v2 | amount, log_amount, hour, dow, is_weekend, day, month, chip_code, errors_flag, tenure_days, amount_ratio (11) | `Is Fraud?` | parsed Year/Month/Day+Time | `User` | same |

Preprocessing (shared by every system, gated and ungated): coerce to numeric →
replace ±Inf with TRAIN median → fill NaN with TRAIN median; LR additionally
scaled inside its own pipeline (StandardScaler fit on train). Category codes
fit on train only (unseen → −1). Labels were never modified anywhere.

## 5. Historical evidence boundary

Preserved unchanged (plan §14/§15; exposure ledger §5):

- ULB in-domain → **DEMONSTRATED / SELF-TESTED** (existing records);
- IBM v2 in-domain → **DEMONSTRATED / SELF-TESTED** (existing records);
- IBM cross-dataset ~0.873 → **EXPLORATORY / NOT INDEPENDENT**;
- Kaggle transfer ~0.435–0.595 → **FAILED / NON-CONFORMING EVALUATION**
  (native contract unavailable; reconstructed subset).

NR-05 produced no evidence that reclassifies any of these. NR-05 results are
new, separately identified exploratory rows (`nr05_*` identifiers) and are
never blended into the historical classifications.

## 6. Experimental matrix (results summary)

All numbers are test-window values from the FINAL runs; full precision in
`reports/nr05/*.json`.

**Discrimination — ensemble (mean LR+RF+XGB+IF → Platt on cal) vs members:**

| Exp | ROC-AUC | PR-AUC | Recall@1%FPR | Brier | ECE | test pos |
|---|---|---|---|---|---|---|
| E1 ULB | 0.9588 | 0.7111 | 0.813 | 0.00050 | 0.00033 | 75 |
| E2 Kaggle | 0.9632 | 0.8029 | 0.854 | 0.00126 | 0.00046 | 2,145 |
| E3 Kaggle entity-disjoint | 0.9697 | 0.8559 | 0.876 | 0.00154 | 0.00033 | 1,515 |
| E4 IBM | **0.3732** | 0.00075 | 0.000 | 0.00102 | 0.00045 | 134 |

**Members (ROC-AUC / PR-AUC):**

| Exp | LR | RF | XGB | IF | ensemble (raw) | ensemble (Platt) |
|---|---|---|---|---|---|---|
| E1 ULB | **0.9739**/0.7002 | 0.9396/**0.7843** | 0.9734/0.7710 | 0.9444/0.0354 | 0.9588/0.7111 | 0.9588/0.7111 |
| E2 Kaggle | 0.8347/0.1583 | 0.9901/**0.8636** | **0.9970**/0.8517 | 0.8860/0.1439 | 0.9632/0.8029 | 0.9632/0.8029 |
| E3 entity | 0.8316/0.2479 | 0.9919/**0.9065** | **0.9973**/0.9057 | 0.8930/0.2209 | 0.9697/0.8559 | 0.9697/0.8559 |
| E4 IBM | 0.5773/0.0016 | 0.6056/**0.0033** | **0.6396**/0.0026 | 0.6071/0.0022 | 0.6268/0.0025 | **0.3732**/0.00075 |

**Paired ensemble-vs-baseline deltas (dependence-aware bootstrap, EXPLORATORY —
METHOD NOT YET FROZEN):**

| Exp | Δ vs XGB (95% CI) | Δ vs LR (95% CI) |
|---|---|---|
| E1 ULB (block) | **−0.0145 [−0.0308, −0.0036]** (ensemble worse; CI excludes 0) | −0.0150 [−0.0408, +0.0041] (inconclusive) |
| E2 Kaggle (entity) | **−0.0338 [−0.0407, −0.0267]** (ensemble worse) | +0.1285 [+0.1175, +0.1405] (ensemble better) |
| E3 entity (entity) | **−0.0276 [−0.0344, −0.0210]** (ensemble worse) | +0.1381 [+0.1236, +0.1528] (ensemble better) |
| E4 IBM (entity) | −0.2663 [−0.3483, −0.1830] (both failed) | −0.2041 [−0.2915, −0.1131] |

**Gating (clean window and Class B primary):** see §§10–12.

## 7. Baseline definitions

Per §6 of the task: same dataset, split, feature representation, preprocessing,
evaluation population and tuning budget (a fixed a-priori config — **no
hyperparameter search** of any kind):

- **Simple baseline:** LogisticRegression (scaled, C=1.0, max_iter=1000).
- **Representative single model:** XGBoost (200 trees, depth 6, lr 0.1,
  subsample 0.8, colsample 0.8) — the repo's headline model family.
- Also reported (diagnostic): RandomForest (100 trees, min_leaf 5) and
  IsolationForest (100 trees; fraud score = −decision_function, normalized on
  train — matching `fusion.py`).

The confirmatory **baseline identity is `[TO BE FROZEN]` (plan §3)**; NR-05
reports BOTH LR and XGB and resolves nothing. The strongest single model
differs by dataset (LR on ULB by ROC-AUC, XGB on Kaggle) — recorded as an issue.

## 8. Ensemble definitions

Intended ensemble = the repo's own fusion recipe (`FusionEngine`: LR + RF + XGB
+ IF, mean of member probabilities → 2-parameter `PlattCalibration` fit on the
CAL window — the same construction as `retrain_v3.py`: `platt.fit(fused_val,
y_val)`). All members trained on TRAIN only; Platt on CAL only; test never
influences fitting or threshold selection.

The **selected ensemble identity is `[TO BE FROZEN]` (plan §3 RQ-M1)** — NR-05
evaluates the mean-fusion candidate and does not select anything. Result: the
mean-fusion candidate **loses to the best single model on every dataset**
(§6 deltas), driven by the weaker RF/IF members diluting LR/XGB.

## 9. Gating definitions

Unit under test (plan §5.0 — two components measured **separately**):

1. **Row-level enforcement** (`row_gated`): Phase-41 enforcement *semantics*
   (BLOCK on missing / non-finite / out-of-range / wrong-type) rebuilt from the
   TRAINING window only: bounds = q01/q99 ± 2·IQR per feature. No native
   contract assumptions.
2. **Window-level drift monitor** (`window_gated`): the repo's
   `src/drift_monitor/psi.py` unchanged — baseline bins from TRAIN; the whole
   window falls back to the rule when any feature PSI ≥ 0.25 (psi.py's standard
   ALERT threshold, chosen a priori from the module's documented constants).

**Ungated comparator:** same model, same preprocessing/imputation, same inputs,
same evaluation population, both gates bypassed; invalid inputs are scored via
the frozen median imputation (it never crashes — plan §5.1).

**Fallback rule:** `amount > TRAIN q99.5 → FRAUD else LEGIT` (train-only,
fixed a priori; rules count as always-confident). **Confident decision:**
ML `p ≥ 0.8 or p ≤ 0.2` (NR-05 exploratory definition; plan §5.3 remains
`[TO BE FROZEN]`).

**Class A (secondary):** NaN / +Inf / clearly out-of-range / wrong-type —
4 disjoint quarters totalling 10% of test rows, one random feature per row.
**Class B (primary):** on 30% of rows — Gaussian noise σ ∈ {0.25, 0.5, 1.0}×
train-std on the top-3-variance features; amount × {1.10, 1.25, 1.50}; cyclic
hour +3 (Kaggle/IBM; ULB hour NOT APPLICABLE). Magnitudes fixed a priori; the
full investigated range is reported, never a selected best point. Labels are
never altered; wrong decisions are judged against original labels; the same
perturbed data is given to gated and ungated systems.

---

## 10. Class A results (designed-for detection — secondary evidence)

Row-gate block rates on the test window (clean baseline in parentheses):

| Condition | ULB | Kaggle | IBM |
|---|---|---|---|
| clean | 4.76% | 0.61% | 71.97% |
| A_nan | 7.14% | 3.09% | 72.69% |
| A_inf | 7.17% | 3.09% | 72.68% |
| A_oor | 7.10% | 3.09% | 72.65% |
| A_type | 7.13% | 3.09% | 72.64% |

Injected rows = 2.5% of test per kind. On ULB and Kaggle the gate blocks
essentially **100% of injected rows** (rate rises from the clean baseline by
≈ the full injected share) — designed-for detection works as expected at the
row level. On IBM the ~0.7pp increment sits atop a 72% clean block rate that
is itself a contract defect (§17/§18: `chip_code`), so IBM Class A rates are
uninformative about injection detection.

Window-level PSI: never fires on Kaggle/IBM Class A (row gate absorbs the
perturbation before distributional mass moves); fires on all ULB windows —
including clean (§11).

## 11. Class B results (not-designed-for — PRIMARY exploratory investigation)

Per-condition detail (gated vs matched ungated on identical perturbed inputs):

**Kaggle (coverage / recall / wrong-confident):**

| Condition | row cov | row recall | ungated recall | row WC | ungated WC | window blocked (PSI) |
|---|---|---|---|---|---|---|
| clean | 0.994 | 0.729 | 0.854 | 5,403 | 5,606 | no (0.083) |
| B_noise σ0.25 | 0.994 | 0.722 | 0.845 | 7,004 | 7,222 | no (0.083) |
| B_noise σ0.50 | 0.990 | 0.721 | 0.843 | 9,176 | 9,488 | no (0.083) |
| B_noise σ1.00 | 0.959 | 0.717 | 0.838 | 13,060 | 14,391 | no (0.083) |
| B_amount ×1.10 | 0.994 | 0.723 | 0.853 | 5,586 | 5,913 | no (0.083) |
| B_amount ×1.25 | 0.993 | 0.720 | 0.851 | 6,083 | 6,668 | no (0.083) |
| B_amount ×1.50 | 0.992 | 0.714 | 0.853 | 7,160 | 8,382 | no (0.083) |
| B_hour +3h | 0.994 | 0.690 | 0.815 | 9,418 | 9,584 | no (0.083) |

- Row gate: wrong-confident **lower than ungated in 7/7** Class B conditions
  (e.g. σ1.0: 13,060 vs 14,391), min coverage 0.959, but recall ≈ 12–14pp
  below ungated in every condition (fallback quality, §12).
- Window gate: **never fires** on any Class B condition (max PSI 0.083 < 0.10)
  → window-level monitoring provides **no Class B detection** at these
  magnitudes (component result: NOT DETECTED).
- Ungated wrong-confident itself grows sharply under shift (5,606 clean →
  14,391 at σ1.0): subtle shifts do produce confident errors — the phenomenon
  the primary gating question targets.

**ULB:** row-gate wrong-confident is lower than ungated in **6/6** Class B
conditions — but the margin is flat (row WC 320–321 vs ungated 404–414 across
all magnitudes, essentially the clean-data difference of §12), while recall is
0.040 vs ungated 0.813 in every condition (0.027 at σ1.0). Coverage
0.888–0.952. The apparent Class B "benefit" is therefore the clean-window
blocking effect, not drift-specific protection. Ungated WC itself barely moves
on ULB under Class B (404 → 414 max): the shift does not destabilize ULB
scores the way it does Kaggle's. Window gate: fires on **all** windows (clean
PSI 1.46 — real temporal drift, 7/30 features ≥ 0.25) → coverage 0 everywhere:
it "wins" wrong-confident only by rejecting 100% of rows, which the task's §9
forbids counting as success.

**IBM:** ungated recall 0.000 at the ensemble threshold (inverted calibrator,
§13) and row coverage 0.280 clean → Class B comparisons are degenerate:
row gate worse than ungated in 7/7, window blocked in 7/7 (coverage 0).
Recorded as INCONCLUSIVE (§18), not as evidence for or against gating.

## 12. Coverage / recall / fallback analysis

Clean-window tradeoff (the gate must not win by rejecting everything):

| System | ULB | Kaggle | IBM |
|---|---|---|---|
| ungated coverage / recall / WC | 1.000 / 0.813 / 404 | 1.000 / 0.854 / 5,606 | 1.000 / 0.000 / 474 |
| row_gated coverage / recall / WC | 0.952 / **0.040** / 320 | 0.994 / 0.729 / 5,403 | 0.280 / 0.030 / 1,067 |
| row_gated false-reject legit | 2,652 | 2,411 | 94,748 |
| row_gated fallback errors | 296 | 2,025 | 712 |
| window_gated coverage | 0.000 (blocked, PSI 1.46) | 0.994 (not blocked) | 0.000 (blocked) |

Findings:

1. **ULB recall collapse (0.813 → 0.040):** fraud lives in feature tails; a
   statistical range contract (q01/q99 ± 2·IQR) blocks precisely the rows the
   model detects (72 of 75 test frauds blocked), and the amount-only fallback
   recovers almost none. The gate reduces wrong-confident (320 vs 404) only by
   discarding the detection task.
2. **Kaggle is the favorable case:** 0.61% clean block, WC −203, recall −12.5pp
   — a real but bounded tradeoff whose acceptability depends on the
   `[TO BE FROZEN]` §5.5 bounds.
3. **Window gate pathology (both directions):** over-triggers on ULB's real
   temporal drift (coverage 0 even clean) and never triggers on Kaggle Class B
   (PSI 0.083). Its baseline/window/block rule needs a freeze decision (§20).
4. Fallback quality is the binding constraint everywhere: a train-only
   amount rule catches almost no blocked frauds (fallback errors ≈ all blocked
   positives). §5.5's "fallback quality floor" cannot be met by this fallback.

## 13. Calibration results

Sequence used everywhere: **training → calibration/validation → test**, Platt
fit on the CAL window only, test never touched (§11 of the task). All values
executed in this run — no historical calibration number was copied.

| Exp | cal positives | raw Brier / ECE | Platt Brier / ECE |
|---|---|---|---|
| E1 ULB | 57 | 0.00266 / 0.03912 | **0.00050 / 0.00033** |
| E2 Kaggle | 2,385 | 0.00688 / 0.06781 | **0.00126 / 0.00046** |
| E3 entity | 1,867 | 0.00779 / 0.07071 | **0.00154 / 0.00033** |
| E4 IBM | 185 | 0.02599 / 0.14882 | 0.00102 / 0.00045 **(but inverted)** |

- ULB/Kaggle/entity: Platt improves both metrics by an order of magnitude —
  calibration **works** where the signal is strong.
- **IBM inversion (negative result):** the 2-parameter fit on the IBM cal
  window learned **coef −0.7029, intercept −6.4186** (reproduced bit-identically
  by an independent refit on the same data), mapping everything to ≈ prior and
  **flipping the ranking** (cal raw AUC 0.652 → Platt 0.348; test raw 0.627 →
  calibrated 0.373, i.e. below chance). On cal, E[raw|pos] = 0.1675 >
  E[raw|neg] = 0.1294 and raw AUC = 0.652, yet the fitted linear-logit slope is
  negative — consistent with an optimizer/scale pathology under 185-positive
  extreme imbalance on a near-flat loss, but NR-05 does NOT resolve which
  (frozen method = `PlattCalibration` defaults, same construction the repo
  uses in `retrain_v3.py`). Recorded as a §6/§18 issue: calibration-population
  adequacy and a calibrator-sanity guard must be decided at freeze, not here.
- Brier/ECE can look excellent while ranking is broken (IBM ECE 0.00045 ≈
  predicting the prior) — an honest-reporting caveat for §4A.3.

## 14. Temporal / entity analysis

**Temporal (per test-time quartile ROC-AUC):**

- ULB: 0.925 / 0.811 / 0.927 / **0.670** (quartile positives 23/30/11/11 —
  last window degrades; small-n, INCONCLUSIVE-leaning).
- Kaggle: 0.961 / 0.963 / 0.962 / 0.971 — stable across the 193.5-day test
  span; prevalence varies (618/603/670/254 positives).
- IBM: **temporal collapse** — train-period AUCs 0.83–1.00 (RF memorizes,
  1.000) fall to 0.58–0.64 on the final20% window (1,020-day span): the
  method does not transport across time on this representation (negative
  result, first-class).

**Entity-disjoint (Kaggle, 20% of `cc_num` held out entirely):** 0.9697
(ROC-AUC) / 0.8559 (PR-AUC) on 240,831 held-out-user rows — **no collapse**;
slightly above the chronological test but on a different population (higher
prevalence 0.63%), so not a like-for-like comparison. Member pattern unchanged
(XGB 0.9973, ensemble 0.9697).

**Entity-disjoint NOT APPLICABLE** on ULB (no entity identifier — reason
recorded per §12 of the task). IBM entity-disjoint not executed (stride
sampling makes per-user histories partial; recorded as scope limitation).
Segments: Kaggle amount-quartile AUCs 0.905/0.862/0.926/0.989 — fraud is
concentrated in the top amount quartile (1,668/2,145), and the second quartile
is weakest; ULB segment q2 = 0.441 with n=9 positives → **INCONCLUSIVE
(small-n)**, withheld rather than reported as performance.

## 15. Business / alert-volume metrics

No operational cost model was invented; alert-volume measures only:

| Exp | top-1% alerts | alerts/day | recall@top1% | top-5% recall | test window |
|---|---|---|---|---|---|
| E1 ULB | 569 | 1,785* | 0.787 | 0.787 | 0.32 days* |
| E2 Kaggle | 5,557 | 28.7 | 0.844 | 0.890 | 193.5 days |
| E3 entity | 2,408 | 4.5 | 0.862 | 0.907 | 537.5 days |
| E4 IBM | 1,318 | 1.29 | **0.000** | 0.030 | 1,020.6 days |

*ULB's `Time` axis spans ≈48 hours total, so per-day rates are dataset
artifacts, not operational forecasts — labelled illustrative, never
institutional (plan §4A.2). IBM alert-volume recall ≈ 0 is the inverted-
calibrator effect (§13), not an operational statement.

## 16. Statistical methodology used

- Point metrics via `metric_definitions.py` v1.0 (ROC-AUC, PR-AUC = average
  precision, `threshold_at_fpr_on_validation` on CAL only, `recall_at_fpr`,
  confusion counts, prevalence report, small-sample warnings).
- CIs: **IID stratified percentile bootstrap** (existing `bootstrap_ci`,
  n=300, seed 42) reported only as contrast, PLUS dependence-aware resampling:
  **entity-clustered** (resample `cc_num`/`User` clusters with replacement) for
  Kaggle/IBM, **temporal block** (20 contiguous time blocks) for ULB — for the
  main metric and paired deltas vs both baselines (same indices for both
  models). All dependence-aware intervals are labelled
  **EXPLORATORY — METHOD NOT YET FROZEN**; none is presented as confirmatory.
  IID is explicitly NOT the default interpretation (task §14).
- Above-chance reference (plan §20, diagnostic only): dependence-aware lower
  CI bounds for the ensemble are 0.920 (ULB), 0.956 (Kaggle), 0.962 (entity),
  all > 0.50; IBM's interval [0.319, 0.422] lies entirely **below** 0.50.
- n_bootstrap = 300 is an NR-05 exploratory choice; the preregistered
  replicate count remains PENDING (protocol marker). Single seed (42);
  §18 minimum-seed rule `[TO BE FROZEN]`.

## 17. Negative results (first-class)

1. **Ensemble loses to the best single model on every dataset** (§6): mean
   fusion is dragged down by weaker members — ULB −0.0145 vs XGB (CI excludes
   0), Kaggle −0.0338, entity −0.0276, IBM −0.2663.
2. **IBM: methodology fails end-to-end** — temporal collapse (0.63 best
   member), Platt inversion → calibrated AUC 0.373 (below chance),
   recall@1%FPR 0.000, top-1% alert recall 0.000.
3. **Gating reduces recall catastrophically on ULB** (0.813 → 0.040 clean):
   range contracts reject tail-dwelling fraud; fallback recovers almost none.
4. **Window-level PSI gate unusable as configured:** over-triggers on real
   ULB drift (coverage 0 even clean) and never fires on Kaggle Class B
   (PSI 0.083 at all magnitudes).
5. **IF member PR-AUC is degenerate** (0.035/0.144/0.002): its normalized
   score has coarse resolution — ROC-useful, precision-useless.
6. ULB last time-quartile AUC 0.670 and segment q2 0.441 (n=9) — recorded,
   small-n, INCONCLUSIVE-leaning; not tuned away.
7. Gating never produced a net win at acceptable coverage+recall under the
   §5.5-style reading — benefit **NOT ESTABLISHED** (criterion itself
   `[TO BE FROZEN]`).

No result above was tuned, re-run with different magnitudes, or dropped.

## 18. Non-conforming / inconclusive experiments

- **`nr05_ibm_gating` = INCONCLUSIVE.** Its row contract marks a *valid*
  categorical code out-of-range: `chip_code` bounds [1.0, 2.0] while **70.6%
  of test rows carry code 0** (train code-0 share ≲1% — q01 ≥ 1). A temporal
  shift in the "Use Chip" mix (train span vs test span over 1,020 days) makes
  the train-window statistical contract misclassify legitimate later-period
  input as invalid (72% clean block; gate reasons: 95,986 out-of-range
  occurrences + 6,352 "missing" — the latter are constructed-feature NaNs from
  negative refund amounts fed to log₁₊). The comparison cannot support gating
  conclusions in either direction.
- **ULB `window_gated` rows** are structurally coverage-0 (over-triggering on
  real drift) — reported, not interpreted as gating success.
- Historical **Kaggle transfer ~0.435–0.595 stays FAILED / NON-CONFORMING**;
  NR-05's Kaggle runs are a different (dataset-appropriate) representation and
  never reclassify it.
- No experiment used a representation non-conforming to ITS OWN declared
  design; no silent dataset substitution occurred.

## 19. Reproducibility records

Evidence via the existing `eval_record.py` v1.1 infrastructure (append-only
ledger + `record_<id>.json` sidecars) — no competing schema. Each record
carries dataset SHA-256, model artifact hashes (joblib members +
`pipeline_state` + `platt` per experiment), seed, threshold +
`threshold_source=validation`, git SHA `5a5ff55`, command, config, metrics,
and warnings flagging EXPLORATORY/NON-INDEPENDENT status.

| Experiment | FINAL record (last per identifier) | status |
|---|---|---|
| nr05_ulb_discrimination | `eval-20261004T105308+0000-2fc57be9b303` | COMPLETED |
| nr05_ulb_gating | `eval-20261004T105317+0000-6b3effcad402` | COMPLETED |
| nr05_kaggle_discrimination | `eval-20261004T110124+0000-cb7205186560` | COMPLETED |
| nr05_kaggle_gating | `eval-20261004T110224+0000-b25acd4a530e` | COMPLETED |
| nr05_kaggle_entity_disjoint | `eval-20261004T110810+0000-b534aa20bc92` | COMPLETED |
| nr05_ibm_discrimination | `eval-20261004T110430+0000-72423ef6dfc8` | COMPLETED |
| nr05_ibm_gating | `eval-20261004T110449+0000-2af15d92a8bc` | COMPLETED |

**Superseded harness iterations (preserved, never deleted):** 5 earlier
records — 2 ULB discrimination + 3 ULB gating — produced while two harness
defects were being found and fixed *before* any result was reported:
(a) Class-A injection cross-contamination (each A-condition applied all four
injection kinds, and the wrong-type slice overflowed to 92% of rows),
(b) isolation-forest score sign (used `+decision_function` instead of the
repo's `−decision_function` fusion direction, inverting that member).
Both defects were fixed, unit-checked, and the ENTIRE matrix re-run; only the
FINAL records back this report. The intermediate records remain in the ledger
because append-only discipline forbids deletion; their metrics are NOT cited.

Ledger after NR-05: **81 records at verification time** (append-only; the
calibration hook appends further records periodically), 12 with `nr05_`
identifiers (7 final + 5 superseded), 0 `prereg_*`/harness records (no
confirmatory or preregistered experiment ran).

---

## 20. Issues requiring later freeze/reviewer decisions

Recorded, NOT resolved (plan §17: document, never silently edit):

1. **Baseline identity** (§3): LR vs XGB — strongest single differs per
   dataset; both reported; selection is reviewer-owned.
2. **Ensemble identity** (§3 RQ-M1): mean-fusion loses to best single on all
   four runs — whether confirmatory uses mean/weighted/stacked fusion must be
   frozen BEFORE any confirmatory run, and NR-05's result must not be used to
   pick a winner post hoc on untouched data.
3. **§5.3 confident-decision definition** — 0.8/0.2 was an NR-05 choice.
4. **§5.5 gating bounds** — min reduction / max coverage loss / max recall
   loss / fallback-quality floor: unmeasurable until frozen; the ULB recall
   collapse shows the fallback floor is decisive.
5. **§5.0 contract construction** — statistical q01/q99±2·IQR bounds fail on
   (a) tail-dwelling fraud (ULB: blocks 72/75 test frauds), (b) categorical
   codes with temporal mix-shift (IBM chip_code), (c) constructed-feature
   NaNs (log₁₊ of negative amounts). Semantic vs statistical bounds, padding,
   and per-feature-type rules need reviewer design.
6. **Window-gate definition** — baseline window, window size, any-feature vs
   fraction-of-features rule, and applicability when natural drift exists:
   over-triggers on ULB (real PSI 1.46), never fires on Kaggle Class B (0.083).
7. **§6 calibration** — Platt inversion under weak signal/185 cal positives:
   calibrator-sanity guard (sign/monotonicity check) and calibration-population
   floor must be decided at freeze; Brier/ECE alone masked the inversion.
8. **§18 power floors** — ULB test 75 positives / cal 57; IBM test 134 / cal
   185: all operating-point and calibration conclusions here are
   small-n-exposed; floors will likely mark several of these INCONCLUSIVE.
9. **§19 method** — entity-clustered/block bootstrap choices, replicate count
   (300 here), and IID's non-default status: METHOD NOT YET FROZEN.
10. **Single seed** — §18 minimum-seed rule `[TO BE FROZEN]`.
11. **Member operating points** — per-member `recall_at_1pct_fpr` above uses
    the shared ensemble-selected threshold (not per-member thresholds); the
    confirmatory reporting rule for member operating points must be frozen.
12. **Alert-volume semantics** — top-k% is computed on the test score
    distribution (labels unused); confirmatory rule (cal-anchored vs
    test-ranked) must be frozen.
13. **Large-dataset sampling** — IBM stride-1/37 is an NR-05 exploratory
    choice; any confirmatory sampling rule for 24M-row datasets must be frozen
    before inspection.
14. **Segment/time-window small-n** — withholding rules for quartiles with
    <50 positives (metric_definitions `min_cell`) vs the stricter §18 floors.

## 21. Recommendations for confirmatory Track M

1. Freeze baseline + ensemble identities FIRST (issues 1–2) — the central
   RQ-M1 comparison is otherwise underdetermined, and NR-05 shows the choice
   materially changes conclusions.
2. Redesign the row contract semantically (issue 5) before any gating claim:
   statistical range contracts are not fit for tail-dwelling fraud or
   evolving categoricals.
3. Decide the window-gate applicability rule (issue 6) — as configured it is
   unusable in both observed directions.
4. Require a calibration guard + power floors (issues 7–8) before trusting
   calibrated operating points.
5. Keep dependence-aware CIs as the confirmatory default once §19 freezes
   them; ULB/Kaggle/IBM showed IID vs dependence-aware intervals differ
   (e.g. ULB block [0.920, 0.981] vs IID [0.929, 0.983]).
6. Do NOT promote any NR-05 number into preregistered claims; these datasets
   stay exploratory (§14 of the plan).
7. The methodology reproduces cleanly on Kaggle (end-to-end: discrimination,
   calibration, gating harness, entity-disjoint) — the operational machinery
   is sound; the open questions are methodological freezes, not engineering.

## 22. Explicit non-claims

NR-05 does NOT claim: independent replication; confirmatory Track M
validation; native 48-feature generalization (never tested — native contract
not used); value of native-only device/location/recipient features; any
untouched-dataset result; frozen or preregistered results; institutional
effectiveness; production readiness; eligibility of IEEE-CIS/BAF; a cost
model or institutional alert burden; that any gate "works" (benefit is NOT
ESTABLISHED); that the ensemble is superior (it is inferior to the best
single model in every run performed); that historical classifications changed
(they did not).

## 23. Final status

**PASS WITH LIMITATIONS** — all seven experiments executed reproducibly with
complete evidence records; scientific conclusions are EXPLORATORY /
NON-INDEPENDENT by construction. Limitations: single seed; small-n on ULB/IBM;
IBM gating INCONCLUSIVE (contract defect); window-gate component unusable as
configured; dependence-aware CIs not frozen; 14 open freeze issues (§20).

**Per-experiment status (§19 taxonomy, not collapsed to PASS/FAIL):**

| Experiment | Status | Note |
|---|---|---|
| nr05_ulb_discrimination | EXPLORATORY | executed; ensemble-inferiority result |
| nr05_kaggle_discrimination | EXPLORATORY | executed; cleanest end-to-end run |
| nr05_kaggle_entity_disjoint | EXPLORATORY | executed; no entity collapse |
| nr05_ibm_discrimination | EXPLORATORY (pipeline FAILED) | calibrated system below chance |
| nr05_ulb_gating | EXPLORATORY | benefit NOT ESTABLISHED; recall collapse |
| nr05_kaggle_gating | EXPLORATORY | WC↓ 7/7 at ~12pp recall cost; window NOT DETECTED |
| nr05_ibm_gating | INCONCLUSIVE | contract defect (chip_code) |

---

### What NR-05 established

- The methodology (train→cal→test, members, fusion, gating harness,
  dependence-aware resampling, evidence records) **reproduces cleanly on
  independently structured exposed datasets** — Kaggle end-to-end, ULB with
  caveats, with every number backed by an append-only evidence record.
- The mean-fusion ensemble **loses to the best single model** on all four
  discrimination runs (paired CIs exclude 0 vs XGB everywhere except where
  noted), while beating the simple LR baseline on Kaggle/entity.
- Platt calibration improves Brier/ECE by an order of magnitude where signal
  is strong (ULB/Kaggle), and can **invert** a ranking where it is weak (IBM,
  reproducible coef −0.7029).
- Class A (designed-for) detection works at the row gate (~100% of injected
  rows blocked on ULB/Kaggle).
- Under Class B, the row gate reduces wrong-confident decisions in 13/13
  conditions across ULB+Kaggle — but with recall costs from −12pp (Kaggle) to
  catastrophic (ULB), and the window component detected **no** Class B shift
  on Kaggle while blocking everything on ULB.
- Entity-disjoint Kaggle performance does not collapse (0.9697 ROC-AUC);
  IBM temporal performance does (0.93 train → 0.63 test).

### What NR-05 failed to establish

- That gating provides acceptable net benefit under any §5.5-style criterion
  (bounds not frozen; ULB recall collapse; IBM INCONCLUSIVE).
- That the ensemble is superior to the best single model (observed opposite).
- Usable calibrated discrimination on IBM (0.373, below chance).
- Window-level drift detection of Class B shifts (not detected at tested
  magnitudes).
- Any per-member operating-point conclusion (shared-threshold caveat).

### What remains blocked

- **Native 48-feature validation (Track N)** — no eligible Group-A dataset
  (IEEE-CIS/BAF not acquired; BLOCKED).
- **Untouched confirmatory Track M** — blocked until the complete freeze
  procedure (statistical review → placeholders → `FREEZE_RECORD.json` →
  freeze check → tagged commit). NR-05 does not and cannot unblock it.
- Reviewer sign-off on the 14 issues in §20.

### What must be frozen later

Baseline identity; ensemble identity; confident-decision definition; §5.5
coverage/recall/fallback bounds; contract construction rules (semantic vs
statistical, categorical/time handling); window-gate baseline/size/block
rule; calibration guard + population floor; §18 power floors; §19
resampling method + replicate count + seed count; member operating-point
reporting rule; alert-volume rule; large-dataset sampling rule; small-n
withholding rules; multiplicity/aggregation (not exercised here).

### Recommended next step

The exploratory diagnostics are sufficient to proceed to
**statistical review → freeze preparation**: the machinery is proven, the
failure modes are enumerated, and every unresolved item is a methodological
freeze decision (§20) that only reviewers can make — no further exploratory
diagnostic is genuinely required on these exposed datasets before that stage.
NR-05 does NOT begin confirmatory Track M; the next roadmap phase should feed
the §20 issues into the review/freeze track, not run more exposed-data
experiments for their own sake.

