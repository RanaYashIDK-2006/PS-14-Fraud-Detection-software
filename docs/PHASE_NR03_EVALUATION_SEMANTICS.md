# NR-03 — Evaluation Semantics & Frozen Final-Test Design

**Phase:** NR-03 (Gate 1) · **Date:** 2026-10-03
**Scope:** evaluation-semantics design only. No final scientific experiment was
executed; no model, dataset, threshold, or methodology was changed to improve any
expected result; no reviewer approval is claimed. Protocol amended to **0.1.2-draft**
(freeze rules + clarifications; markers preserved at 11).

---

## A. Objective

Convert the `REVIEW REQUIRED` preregistered protocol into a precise final-test design:
explicit dataset/split semantics, an operational final-test lock, seed/repetition and
failed-run rules, frozen preprocessing and metric semantics, CI/MMD semantics,
full-matrix reporting, and an evidence/provenance freeze — resolving only what the
existing project record determines, and preserving everything that genuinely requires
statistical or domain review.

---

## B. Source Documents and Implementation Reviewed

Documents: the protocol (0.1.2-draft), `PHASE_NR02_PROTOCOL_REVIEW.md`,
`PHASE_NR01_EVIDENCE_ENFORCEMENT.md`, `PHASE_105_EVIDENCE_BASELINE.md`,
`EVIDENCE_RECORD_SCHEMA.md`, `BASELINE_REPRODUCTION.md`, `PHASE_106_ROADMAP_RECONCILIATION.md`,
`PHASE_106_REQUIREMENT_STATUS.md`, `RESEARCH_GATE_READINESS.md`.

Implementations inspected (code, not prose): `metric_definitions.py` (v1.0: `roc_auc`,
`pr_auc`, `confusion_counts`, `operating_point`, `threshold_at_fpr_on_validation`,
`recall_at_fpr` (threshold variant), `bootstrap_ci`, baselines, `small_sample_warning`);
`eval_ulb.py` (split, scalers, `recall_at_fpr` (ranking variant), local `bootstrap_ci`);
`ibm_train.py` (user-disjoint split comment :247–250, fused = mean(lr,rf,xgb) :293,
Platt :329); `train_compare.py` (time_70_15_15, `StandardScaler().fit(X_train)` :221);
`cross_dataset_eval.py` (proxy warning); `calibration_test.py` (`compute_brier_score` :41,
`compute_ece` :46, equal-width bins); `evaluate.py` (`refuse_test_tuning` :65);
`eval_record.py` (schema v1.1 fields); plus the four record blocks. Dataset files were
re-hashed this phase (verification command only).

---

## C. Evaluation Semantics Matrix

Status key: **F** = FROZEN · **P** = PENDING REVIEW · **B** = BLOCKED · **N** = NOT
APPLICABLE. Experiments: E1 ensemble contribution (criteria 1/2) · E2 gating (3/4) ·
E3 external transfer (5/6) · E4 external calibration · E5 subgroup analysis (§6) ·
E6 business metrics (§7) · E7 baselines & ablations (supporting C1/C2).

| Field (required) | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|---|---|---|---|---|---|---|---|
| Experiment ID | F | F | F | F | F | F | F |
| Research question | F | F | F | F | F | F | F |
| Dataset | P | P | F | F | P | P | P |
| Dataset version/hash | P | P | F | F | P | P | P |
| Population | P | P | F | F | P | P | P |
| Split | P | P | P | P | P | P | P |
| Train population | P | P | P | N | N | N | P |
| Validation population | P | P | N | N | N | N | P |
| Final-test population | P | P | F | F | P | P | P |
| Seed policy | P | P | P | P | P | P | P |
| Model | F | F | F | F | F | F | F |
| Feature contract | P | P | P | P | P | P | P |
| Preprocessing | F | F | F | F | F | F | F |
| Calibration | F | F | F | F | N | N | N |
| Threshold policy | F | F | F | F | F | F | F |
| Primary metric | F | F | F | P | F | F | F |
| Secondary metrics | F | F | F | P | F | F | F |
| Uncertainty method | P | P | P | P | P | P | P |
| Statistical test (if applicable) | P | P | P | P | N | N | P |
| Selection data | F | F | F | F | F | F | F |
| Locked data | F | F | F | F | F | F | F |
| Evidence artifact | F | F | F | F | F | F | F |
| Failure rule | F | F | F | F | F | F | F |

**161 cells: FROZEN 87 · PENDING REVIEW 62 · BLOCKED 0 · NOT APPLICABLE 12**
(machine-counted from the table rows above).
(Not-applicable cells are analyses that fit/select nothing; BLOCKED lives at the
experiment level for the 21+7 contract — see §N; PENDING cells concentrate in
dataset/split/seed/contract/CI fields tied to the §0 scope marker, the §1 split marker,
the §5 seed marker, and the §2 CI/binning markers.)

Key rules frozen even where an identity is pending: no external-set training or
selection ever (E3/E4); fit-train-only preprocessing everywhere; validation-only
selection; ranking primaries are threshold-free; every run must produce a v1.1 ledger
record or the cell cannot be MEASURED.

---

## D. Dataset/Split Freeze

Re-hashed this phase (verification only; no data modified):

| Dataset | sha256 | Role |
|---|---|---|
| `data/creditcard.csv` (ULB) | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` | in-domain candidate (matches ledger `…e58d68c2bf88`) |
| `data/credit_card_transactions-ibm_v2.csv` | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` | entity-disjoint candidate |
| `data/kaggle_fraud/fraudTest.csv` | `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0` | final external (locked in protocol §9) |
| `data/kaggle_fraud/fraudTrain.csv` | `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d` | external-train portion (only ever usable for development, never for selection against E3 claims) |
| BAF | — | BLOCKED (absent) |

Per-criterion:

- **Criteria 1–2 (D-1):** dataset **PENDING REVIEW**. Resolved as far as facts allow:
  ULB cannot host the deployed-ensemble comparison (production model is 5/48-usable
  there — fail-close, phase108); BAF is blocked; Kaggle fraudTest is reserved as the
  final external set (using it as the in-domain test would double-role the external
  set); that leaves **IBM v2 (hash above) for the public_v1/21 track** (it hosts an
  ensemble: fused = mean(lr,rf,xgb) + Platt, `ibm_train.py:293`) and, if the §0 scope
  selects S1 (48-native), **no untainted in-domain dataset exists** — Kaggle would have
  to be dual-roleed, which is a scope decision, not a repository inference. Final
  selection = the §0/§1 markers. NOT chosen here.
- **Criterion 3–4 (D-1b, new):** the 7 gating conditions have **no named dataset**
  either — same class of gap as D-1; recorded for sign-off.
- **Criterion 5–6:** dataset **FROZEN** = Kaggle `fraudTest.csv` (protocol §1 assigns
  it the external-transfer role; it is the only non-same-generator family available).
  Protocol already fixes "temporal + entity-safe split"; exact boundaries **PENDING**
  (not specified anywhere). Full-claim replication second dataset = BAF → BLOCKED.
- **Split structures (as implemented today):** `eval_ulb` = random stratified 80/20,
  `random_state=42`; `ibm_train` = row-range split of a (User,ts)-sorted frame →
  **user-disjoint, explicitly NOT chronological** (code comment :247–250); `train_compare`
  = `time_70_15_15`. Unit of independence: transaction row; entity = user/card per
  dataset. Entities must not cross partitions under entity-disjoint splits; the
  chronological split admits entity recurrence (documented, reviewable).
- **Preprocessing-fit audit:** `eval_ulb` scalers fit on train only (:82/:85);
  `train_compare` scaler fit on `X_train` only (:221); no test-influenced fitting found
  anywhere. **Final-test selection audit:** no current code computes outcomes on any
  designated final test — no freeze exists yet to violate (freeze defined in §S).

Historical exposure (recorded, not reset): ULB was repeatedly tuned (~20 push/optuna
reports, G-04); Kaggle results influenced historical promotion decisions (phases 24/25);
IBM v2 was used in development. No public dataset is pristine — see §L.

---

## E. Final-Test Lock

Defined operationally in protocol **§9** (added this phase; full 11-question spec in
§S below): the lock engages the instant the frozen-test artifact + freeze record exist
in `eval_ledger.jsonl`, before any final-test execution. After lock: no model/
hyperparameter/variant/seed/threshold/preprocessing/calibration/endpoint/segment/MMD
changes informed by test or external outcomes; no selective repetition; no post-hoc
exclusions; external results never influence selection. Pre-lock observation is
structural only (rows, schema, class balance may be counted for power checks; **no
outcome/metric may be computed**). Violation ⇒ confirmatory status void; results
reported as exploratory; recorded in `evaluation_config.deviations` and disclosed.

---

## F. Seed and Repetition Policy

| Question | Determination |
|---|---|
| Number of seeds | **PENDING REVIEW** (protocol proposal: 5 fixed seeds — retained as the working proposal) |
| Seed values | **PENDING REVIEW** (proposed for review: 42, 43, 44, 45, 46 — pre-declared, value-independent; not frozen here) |
| Scope | All stochastic operations: training initialization, any random split/shuffling, and CI resampling (CI seed is a separate declared parameter) |
| Replicate identity | Each seed = one independent replicate with its own ledger record |
| Reported statistic | Per-seed results MUST all appear (matrix, §M); the across-seed **primary aggregation is PENDING REVIEW** (NR-02 D-5: mean vs median vs pooled — no method chosen because it looks better) |
| Uncertainty across seeds | PENDING REVIEW (paired with the aggregation choice) |
| Seed exclusion | Never (protocol §9.9) |
| Seed failure | Recorded FAILED; identical-seed rerun only for implementation failures (§G) |

---

## G. Failed-Run Policy

Classification (all outcomes append-only; a failed run is never silently replaced):

| Failure | Class | Counts? | Rerun? | Excludable? | Determined by |
|---|---|---|---|---|---|
| Crash | implementation | as FAILED cell | yes, identical frozen config, both recorded | never post-hoc | mechanical (harness) |
| Invalid output | implementation/data | FAILED | same | pre-registered validity rules only | mechanical |
| Feature-contract violation | invalid experiment | cell = NOT ESTABLISHED (or BLOCKED if contract unavailable) | only after contract fixed, new experiment ID | invalidity is mechanical, not a choice | contract check (§N) |
| Cannot produce evidence | evidence failure | cell = NOT ESTABLISHED after one evidence-only rerun | yes, identical config, both attempts recorded | — | §9.7 |
| Missing data | data quality | FAILED/BLOCKED | no (dataset unchanged) | — | hash check |
| NaN/Inf in scores or metrics | implementation/data | FAILED | after fix, identical config | — | metric domain check (§8) |
| Resource limit | implementation/infra | FAILED | yes, identical config, note resources | — | harness |
| Reproducibility check fails | evidence failure | NOT ESTABLISHED | yes, both records kept | — | §9.7 |
| Metric outside mathematical domain (e.g. ROC-AUC on single-class partition) | data quality → experiment invalid for that dataset/split | FAILED | no — split design issue → back to reviewer | — | metric domain check |

Bootstrap-level invalid resamples (not whole runs): existing rule in
`metric_definitions.py` — if fewer than 50% of resamples are valid, the CI is
**withheld** rather than reported (tested by `eval_record_test`). Frozen as-is.

---

## H. Preprocessing Semantics

Frozen rules (per experiment): each fitted step's population is the **train partition
only**; transformations apply unmodified to validation/final-test; no test statistic
(including scaling means/vars, imputation, category levels, feature selection) may be
derived from validation or test. Fit-once per split; refit only as part of a new
pre-registered run (new record), never after lock.

Implemented sequences (verified): `eval_ulb` — per-model `StandardScaler`
fit-transform train / transform test (:82, :85); `train_compare` — `StandardScaler`
fit on `X_train` (:221); `ibm_train` — no scaling (tree/linear with
`class_weight='balanced'`); `cross_dataset` — preprocessing embedded in the frozen
model artifacts (scaler shipped inside); `calibration_test` — none (scores are fixed
inputs).

Recorded values (the vocabulary frozen this phase, populated via the minimal
implementation correction — §R):

| Pipeline | `preprocessing_version` | `feature_schema_version` |
|---|---|---|
| `eval_ulb` | `standardscaler_fit_on_train` | `ulb_pca30_research` |
| `train_compare` | `standardscaler_fit_on_train` | `ps14_synthetic_48_native` |
| `cross_dataset_eval` | `embedded_in_artifacts` | `production_48_native` |
| `calibration_test` | `none_fixed_artifacts` | `none_fixed_scores` |
| future harness runs (NR-04) | per-experiment step list | `public_v1` / `native48` (contract file name + hash) |

Legacy records keep their `"1.0"` defaults (back-compat verified); old records are not
rewritten.

---

## I. Metric Semantics

| Metric | Definition (exact) | Positive class | Weighting | Undefined / missing behavior | Implementation | Matches protocol? |
|---|---|---|---|---|---|---|
| ROC-AUC | `metric_definitions.roc_auc` (rank-based; ties handled — tested) | label = 1 (fraud) | none | single-class input → invalid (data-quality failure, §G) | `metric_definitions.py:122` | MATCH |
| PR-AUC | average precision (`pr_auc`; random scores → prevalence — tested) | 1 | none | single-class → invalid | `metric_definitions.py:130` | MATCH (semantic pin: **average precision**, not trapezoidal) |
| Brier | `mean((p − y)²)` over evaluated rows | 1 | none | NaN propagates → FAILED (§G) | `calibration_test.py:41` | MATCH — unambiguous |
| ECE | `Σ_b \|acc_b − conf_b\| · n_b/n` over **10 equal-width bins**, `np.linspace(0,1,11)`, bounds `[lo,hi)`, empty bins skipped | 1 | row weights via n_b | empty bins contribute 0 | `calibration_test.py:46` | **PARTIAL** — protocol proposes 10–15 *equal-mass*; binning is PENDING REVIEW (marker retained). See §B-option below |
| recall / TPR, precision, FPR | `confusion_counts` → `operating_point` at a given threshold | 1 | none | threshold outside [0,1] invalid | `metric_definitions.py:101/115` | MATCH |
| recall@1%FPR | **ranking variant:** prepend (0,0) to `roc_curve`, take TPR at the first point whose FPR ≥ 0.01 (`searchsorted`, `eval_ulb.py:24–30`) | 1 | none | single-class invalid | `eval_ulb.py:24` (train_compare uses the same ranking semantics) | MATCH by reproduction (0.918367 was computed exactly this way) — **semantic pin recorded**: on a discrete curve the operating point may sit slightly above 1% FPR; the threshold-based variant at `metric_definitions.py:161` is a *different function* for operating-point reporting and must not be substituted |
| Selective risk (gating) | errors among non-abstained ÷ non-abstained | 1 | none | zero-coverage → undefined → FAILED cell | not yet implemented (NR-04) | PARTIAL — matches C3 text |
| Wrong-confident | C3 text: score ≥ "threshold-band top decile" ∧ wrong — **the population defining the decile is unspecified** | 1 | — | — | not implemented | **PENDING REVIEW** (new item T-7) |
| Alert rate / coverage / burden | counts over evaluated rows | — | none | — | to be built NR-04 | MATCH (§6/§7 text) |

MMDs are decision thresholds, not metric functions (§K).

**Brier/ECE status — Option B chosen:** no metric was upgraded. `metric_definitions`
stays v1.0 without Brier/ECE; the governing implementations for preregistered runs are
exactly the functions above (`calibration_test.py`), results remain labeled
**provisional** per protocol §2, and the equal-mass/equal-width choice remains
`PENDING REVIEW`. Rationale: Option A would bake a review-pending binning convention
into an "established" definitions file, silently resolving the marker.

---

## J. Confidence-Interval Semantics

Inventory of the **three** existing bootstrap implementations (discrepancy widened by
inspection, not changed):

| Where | Default replicates | Structure | Seed |
|---|---|---|---|
| `metric_definitions.bootstrap_ci` | **1000** (`BOOTSTRAP_DEFAULT_SAMPLES`) | percentile, class-stratified, single-metric | 42 |
| `eval_ulb.bootstrap_ci` | **200** | single-metric | 42 |
| Protocol §2 proposal | **2000** | stratified | fixed (value TBD) |

- 1000-vs-2000 (and 200) is **substantive, not factual**: the protocol itself marks
  replicates `[REQUIRES DECISION]`, so nothing was silently changed. **PENDING REVIEW**;
  the harness must pass an explicit `n_bootstrap` once confirmed (NR-04 requirement,
  recorded).
- Construction method FROZEN: percentile bootstrap, class-stratified resampling,
  95% level (per criteria text), invalid-resample rule = withhold if <50% valid
  (existing, tested).
- **Paired-difference CI: NOT IMPLEMENTED and REQUIRED before E1/E2/E3 may run**
  (criteria 1/3 demand CI of the difference). Must be built in NR-04/07; recorded as
  an implementation prerequisite, not built here (it would be dead code without the
  harness).
- Per-CI semantics table: estimand = the named metric or metric difference; unit =
  transaction row; paired structure for Δ-metrics over identical rows; seed declared
  per run in `evaluation_config`.

---

## K. MMD Semantics (thresholds unchanged)

| MMD | Quantity | Reference pop. | Comparison pop. | Abs/rel | Direction | Equality | Rule type |
|---|---|---|---|---|---|---|---|
| C1 ≥2% PR-AUC | (ens − single)/single | best single **chosen on validation** | pre-registered ensemble | relative | higher better | `≥ 0.02` — exactly 2% = MET | hard gate for the claim; needs BOTH relative ≥2% AND 95% CI of Δ excluding 0 (`0 ∉ [lo,hi]`; an interval touching 0 does **not** exclude it → not met → inconclusive branch) |
| C3 ≥20% / ≤2pp | wrong-confident reduction; coverage loss | ungated (same rows, same model) | gated | reduction relative ≥ 0.20 (exactly 20% = MET); coverage loss absolute ≤ 0.02 (exactly 2pp = MET) | reduction higher better; loss lower better | as stated | hard gate; all three conditions together (reduction, loss, CI excluding 0) |
| C5 ≥0.70 / ≥80% | external ROC-AUC; retention = external ÷ in-domain | chance 0.5; the E1 in-domain value (**identity PENDING — tracks D-1**) | frozen candidate on `fraudTest` | absolute AUC; relative retention | higher better | `≥ 0.70` / `≥ 0.80` — equality = MET; CI must exclude 0.5 strictly | hard gate |

None altered. C2/C4/C6 are the complementary "not demonstrated" rules (CI includes 0 /
MMD not reached). Descriptive-statistic readings (e.g., reporting the point estimate
beside the gate) are allowed but may not substitute for the gate.

---

## L. External-Set Selection Rules

Four tiers frozen (protocol §9): **development** (train; model/hyperparameter decisions)
→ **validation** (calibration, threshold, constituent selection only) → **final test**
(locked, single use) → **final external** (`fraudTest.csv`: evaluation only; results
must never influence model, threshold, protocol, or report selection).

Honest audit of current language: the protocol never authorizes external-set selection,
and §9 now says it explicitly. The repository history, however, **did** use Kaggle
results in promotion decisions (phases 24/25 — E_hardneg promotion is BLOCKED on those
results). That exposure cannot be undone; therefore:

- From 0.1.2-draft onward, external results are evaluation-only (frozen).
- Whether `fraudTest` can still carry **confirmatory** status despite prior exposure —
  or whether that requires the pristine BAF — is **PENDING REVIEW** (T-8), recorded
  honestly rather than papered over.

---

## M. Full-Matrix Reporting

Frozen reporting rule (protocol §9.11): the reported object is the complete tensor
**experiment × dataset × seed × feature contract × condition × transfer direction**.
Every cell carries exactly one status:

- `MEASURED` — a v1.1 ledger record exists and resolves;
- `NOT ESTABLISHED` — attempted but evidence incomplete / invalid experiment (e.g.,
  contract violation);
- `BLOCKED` — external dependency absent (e.g., BAF, 21+7);
- `FAILED` — run failed and is preserved as FAILED;
- `NOT APPLICABLE` — cell is meaningless for that experiment (e.g., seeds × descriptive
  analysis).

No cell may be omitted, zero-filled, or silently merged; unavailable ≠ 0. Selective
presentation of favorable cells is a protocol violation (§9.5).

---

## N. Feature-Contract Semantics

| Experiment / track | Required contract | Available | Unavailable / derived | Prohibited proxies | Valid? |
|---|---|---|---|---|---|
| E1/E2/E7 on public_v1 track | 21 domain + public_v1 (14-declared contract file — count inconsistency N-02 → NR-04 fix) | full on IBM v2 | — | substituting 48-native outputs while claiming public_v1 (contract downgrade = violation) | VALID on IBM v2 |
| E1/E2/E7 on S1 track | 48-native | 47/48 on Kaggle fraud (phase25 reconstruction = derived features, must be labeled derived) | 1 feature unavailable (phase25 artifact) | PCA↔§16 pseudo-mappings (`kaggle_ulb_pca_proxy` is explicitly rank-preserving only) for any feature-semantic claim | valid only with derived-labeled cells; in-domain ULB run **invalid** (5/48 → fail-close) |
| E3/E4 | candidate's own contract (tied to §0) | Kaggle supports 48-native (47/48) | 21+7 track: **all 7 absent everywhere public** | any invented mapping of available features onto the 7 | **`BLOCKED — FEATURE CONTRACT` for the 21+7 full-contract claim**, by design |
| Subgroup × availability group (E5) | 14-public_v1 vs 48-native groups (§6) | both groups measurable | — | — | VALID |

Silent downgrade (running fewer features while naming the full contract) is a protocol
violation (§9.4/§9.5). No mapping was invented this phase.

---

## O. Evidence/Provenance Freeze

Every future final-test result must yield an NR-01-compatible record with:

| Item | Field | Population status |
|---|---|---|
| Experiment ID | `evaluation_config.experiment_id` (free-form map; harness, NR-04) | planned |
| Dataset identity/hash | `dataset.sha256` | FROZEN (enforced for DEMONSTRATED) |
| Split | `evaluation_config.split` (free-form today; harness will use a fixed vocabulary — NR-04) | planned |
| Seed | `seed` / `seed_not_applicable` | FROZEN (enforced) |
| Command | `command` (v1.1) | FROZEN (enforced) |
| Model/config identity | `model_hash` + `artifact_file_hashes` | FROZEN |
| Feature-contract identity | `feature_schema_version` | **populated this phase** for the four canonical pipelines (vocabulary §H); harness extends |
| Preprocessing identity | `preprocessing_version` | **populated this phase** (vocabulary §H) |
| Threshold state | `threshold` + `threshold_source` | FROZEN |
| Metric definitions | `metric_definitions_version` (+ provisional labels for Brier/ECE) | FROZEN |
| Protocol linkage | `evaluation_config.protocol_version = "0.1.2-draft"` | planned (harness sets at run) |
| Deviations | `evaluation_config.deviations` | planned (§9.5/§9.10) |
| Artifact identity | hashes | FROZEN |
| Git SHA | `git_commit` | FROZEN (enforced) |

No new evidence infrastructure created; no duplicate ledger; existing fields only.

---

## P. Outcome Hierarchy (per experiment)

| Experiment | Primary outcome | Secondary outcomes | Descriptive | Failure outcomes |
|---|---|---|---|---|
| E1 | ΔPR-AUC vs validation-chosen best single (C1 gate) | recall@1%FPR Δ; per-seed values | constituent identities, training curves | contract violation; no ledger record; split invalid |
| E2 | wrong-confident reduction at matched coverage (C3 gate) | selective risk curves, FP/FN, coverage | per-condition breakdown | zero-coverage cells; condition not executable |
| E3 | external ROC-AUC + retention (C5 gate) | PR-AUC, Brier/ECE (provisional), per-direction | degradation narrative | single-class partition; contract invalid |
| E4 | Brier (frozen); ECE **provisional/pending** | reliability diagram bins | calibration table | scores unavailable |
| E5 | none (descriptive) | — | per-segment PR-AUC/r@1%/alert-rate + CIs | small-n segments flagged, never over-read |
| E6 | none (descriptive) | — | recall@top-k curves, burden, cost-ratio parameterized curves | capacity numbers missing → illustrative labels required |
| E7 | supports E1 primary (reference lines) | baseline tables | majority/random references | — |

**Anti-promotion rule (frozen):** a secondary or descriptive outcome may never replace
a primary outcome after results are observed; any change requires a dated protocol
amendment BEFORE test access, else confirmatory status is void (§9.5).

---

## Q. Acceptance-Criterion Semantics (numbers unchanged)

| Criterion | Input | Metric | Dataset | Split | Threshold | Direction | Equality | Evidence required | If unavailable |
|---|---|---|---|---|---|---|---|---|---|
| C1 | frozen candidates' scores on final test | PR-AUC (+ r@1%FPR sec.) | **PENDING (D-1)** | **PENDING (§1 marker)** | ranking: none; threshold secondaries use frozen threshold | higher | ≥2% relative MET at equality; CI excludes 0 strictly | paired-Δ CI + ledger records | cell FAILED/NOT ESTABLISHED → inconclusive branch |
| C2 | same as C1 | same | PENDING | PENDING | same | CI includes 0 ⇒ "not helped" | include-0 at equality ⇒ C2 direction | same | same |
| C3 | gated vs ungated, same rows, 7 pre-declared conditions | wrong-confident reduction + coverage | **PENDING (D-1b)** | PENDING | wrong-confident cut **PENDING (T-7)** | reduction up, loss down | ≥20% / ≤2pp MET at equality; CI excludes 0 | paired CI + records | inconclusive → curves, no claim |
| C4 | same | same | PENDING | PENDING | same | as C3 complement | include-0 ⇒ C4 | same | same |
| C5 | frozen candidate on external | ROC-AUC (+PR-AUC) | **FROZEN: fraudTest** | temporal+entity-safe, **boundaries PENDING** | none (ranking) | higher | ≥0.70 / ≥80% MET at equality; CI excludes 0.5 strictly | ledger record + paired-vs-chance CI | BAF blocked ⇒ one-dataset = provisional only |
| C6 | same | same | FROZEN | same | none | complement | CI excludes MMD on both attempted datasets | same | blocked replication ⇒ C7 |
| C7 | — | — | — | — | — | — | triggered by overlap/admission-failure/blocked replication | documented next experiment | terminal state valid (Conclusion C) |

Deterministic given the pending identities; no criterion value changed.

---

## R. Protocol Change Classification

| Change | Class | Where |
|---|---|---|
| §9 final-test freeze (11 rules + data tiers) | SEMANTIC CLARIFICATION | protocol §9 |
| Version note 0.1.2-draft + Change history rows | FACTUAL/DOCUMENTARY | protocol header/history |
| Brier/ECE governing-implementation documentation (Option B — no code/metric change) | FACTUAL/DOCUMENTARY | this report (protocol §2 already says "provisional") |
| `preprocessing_version` / `feature_schema_version` population in the 4 canonical pipelines + `record_from_run` optional kwargs | IMPLEMENTATION CORRECTION (fields already existed; legacy defaults unchanged; kwargs-only) | 5 files, §H vocabulary |
| Everything else (dataset choice, seeds, replicates, binning, MMDs, wrong-confident cut, external confirmatory status) | recorded, **NOT changed** | §T |

**SUBSTANTIVE METHODOLOGICAL CHANGE: none.** Markers removed: **0** (11 → 11).

Verification of the implementation correction: `py_compile` OK ×5; in-memory
construct-direct/forwarded/legacy-default field test PASS; `eval_record_test` 22/22;
`claim_evidence_check` PASS (22 claims). No experiment ran; no ledger record was
appended by this phase (the verification constructed records in memory only).

---

## S. Final-Test Freeze Specification

(Protocol §9, verbatim; the 11 questions answered.)

1. **What is locked:** designated final-test partitions (by file sha256 + split
   definition + boundaries), `fraudTest.csv` (hash in §9), all weights, preprocessing,
   calibration, thresholds, seeds, metric definitions, MMD rules — recorded in the
   frozen-test artifact + freeze record.
2. **When:** the instant those artifacts exist — before any final-test execution.
3. **Allowed before lock:** development on train; calibration/threshold/constituent/
   hyperparameter selection on validation; protocol edits via change history.
4. **Forbidden after lock:** any outcome-informed change (model, hyperparameters,
   seeds, thresholds, preprocessing, calibration, metrics, endpoints, segments, MMDs);
   selective repetition; post-hoc exclusions; external-set-driven selection.
5. **Violation:** any item in 4, or running before the freeze exists ⇒ confirmatory
   status void, results exploratory, recorded in `evaluation_config.deviations` and the
   Gate-2 report.
6. **Failed run:** recorded FAILED, preserved, reported FAILED in the matrix.
7. **Evidence failure:** one evidence-only rerun under identical frozen config, both
   recorded; persistent failure ⇒ cell NOT ESTABLISHED.
8. **Rerun:** only implementation/evidence failures, identical config, never to change
   an outcome, every attempt recorded.
9. **Seed exclusion:** never post-hoc; identical-seed rerun only for implementation
   failures; every declared seed appears in the matrix.
10. **Deviation authority:** only a dated Change-history amendment made BEFORE the
    affected decision; after outcomes exist nothing converts it back to confirmatory.
11. **Reporting:** full matrix with MEASURED / NOT ESTABLISHED / BLOCKED / FAILED /
    NOT APPLICABLE; no omitted, zero-filled, or selective cells.

---

## T. Remaining Reviewer Decisions

**11 protocol markers (unchanged):** §0 scope/candidate · §1 split axis per dataset ·
§2 ECE binning · §2 CI replicates+seed · §2 multiplicity · §3 all MMDs · §4 A/B/C
wording + publication terms · §5 seed count/values · §5 operating point · §5 S1
threshold carry · §7 analyst capacity.

**11 open sign-off items recorded in reports (5 new this phase, †):**
T-1 D-1 dataset for criteria 1–2 (tied to §0) · **T-2† D-1b dataset for criteria 3–4** ·
T-3 seed values (proposal 42–46) · **T-4† across-seed aggregation method** ·
T-5 which CI implementation/replicate count governs (3 exist: 200/1000/2000) ·
T-6 ECE equal-mass vs equal-width (marker-bound) · **T-7† wrong-confident decile
population** · **T-8† external set confirmatory status given prior promotion exposure** ·
**T-9† external split boundaries** · T-10 in-domain reference identity for C5 retention ·
T-11 preprocessing-freeze/external-set-selection confirmation of §E/§L rules (reviewer
acknowledgment).

Nothing in this list was resolved by assumption.

---

## U. Gate 1 Impact

NR-03 delivers the third Gate-1 prerequisite's substance: semantics are explicit, the
freeze exists, provenance fields are populated and verified. Gate 1 **remains PARTIAL**
because its completion criteria in the roadmap also include NR-01's blocked GitHub-runner
verification and NR-02's four-role sign-off (both external). Research gate: **PARTIAL**.
What changed: the protocol is now executable-as-written *pending* the §T decisions —
the Gate-2 harness (NR-04) has a precise contract to implement.

---

## V. Next Requirement

**NR-04 — Preregistered benchmark harness (+ IBM v2 ledger upgrade, N-02 contract-count
check)**: one ledger-writing pipeline that implements this design — frozen-test
artifact, experiment IDs, paired-Δ CI, explicit `n_bootstrap`, contract-count check,
`protocol_version` linkage, full-matrix cell emission. Identified only — **not started.**

---

## W. Scope Verification

- Final scientific experiments executed: **none** (dataset hashing + in-memory record
  construction + compile checks only; no model training, no metric computation on any
  dataset, no ledger append by this phase).
- Model architecture / training data / feature definitions / production thresholds /
  research question: **unchanged** (`models/`, `data/`, `risk_engine/`,
  `privacy_layer/` untouched; threshold 0.018758 untouched).
- Thresholds optimized: none. MMD numbers: unchanged. Criteria: unchanged.
- Markers: 11 → 11. Reviewer approval: none claimed.
- Files changed this phase: protocol (semantic §9 + history, version note), 5
  pipeline/record files (kwargs-only field population), this report.
- Checks rerun after edits: `py_compile` ×5 PASS; in-memory field test PASS;
  `eval_record_test` 22/22 PASS; `claim_evidence_check` PASS (22 claims);
  marker count 11 PASS.

---

## X. Final Status

```
PASS WITH LIMITATIONS
```

Limitations: 11 markers + 11 sign-off items (5 newly identified) remain genuinely
pending external/owner review; D-1/D-1b dataset identities are deliberately unresolved;
the external set's confirmatory status is honestly questionable given historical
exposure. Every determinable semantic was frozen or explicitly recorded; no experiment
ran; no favorable-result optimization exists anywhere in this design; NR-04 is the
next requirement.
