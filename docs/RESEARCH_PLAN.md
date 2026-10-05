# Research Plan

**Status:** DRAFT — not frozen  
**Document:** `docs/RESEARCH_PLAN.md`  
**Purpose:** Define the research questions, dataset eligibility, exposure history, statistical rules, success/failure criteria, and decision procedure before substantive confirmatory experiments.

> **Important:** This document is a planning artifact. A value shown as a bracketed TO-BE-FROZEN marker must be resolved before the confirmatory freeze. The freeze process must fail if any such placeholder remains.

---

# Executive Summary

## Research tracks

| Track | Purpose | Current status |
|---|---|---|
| **N — Native 48-feature** | Test the actual native 21→48 feature contract | **BLOCKED** |
| **M — Method-level replication** | Test the detection/gating methodology independently on dataset-specific features | **PROPOSED PRIMARY PUBLIC TRACK** |
| **P — Reduced public contract** | Test a shared reduced feature contract if enough datasets support it | **CONDITIONAL** |
| **I — Institutional** | Eventually evaluate the native contract with authorized institution-controlled data | **BLOCKED / NOT ESTABLISHED** |

## Five decision rules

1. **No dataset exposure is ignored.** Previously used datasets are not treated as untouched confirmatory evidence for the already-developed model.
2. **The primary gating claim uses subtle/plausible drift**, not only invalid-input rejection. Designed-for detection and not-designed-for drift are reported separately.
3. **A gate cannot win by rejecting everything.** Wrong confident decisions include confident fallback/rules decisions, and success requires bounded coverage/recall loss.
4. **Statistical inference respects dependence.** Entity-clustered or temporal-block bootstrap is used where applicable; ordinary IID bootstrap is not the default for dependent transaction data.
5. **Every track has independent outcomes:** `GO`, `NO-GO`, `REVISE`, `INCONCLUSIVE`, `BLOCKED`, `NOT ESTABLISHED`, or `NOT APPLICABLE`, according to frozen rules.

## Freeze requirement

The plan, preregistration, dataset eligibility manifest, metric definitions, and exposure record must be frozen at a tagged Git commit with recorded SHA-256 hashes and ledger entry.

A freeze check must fail if any bracketed placeholder remains (see Section 30).

A statistical reviewer must sign off on the statistical decision sections before the freeze.

---

# 1. Purpose

This document defines four research tracks:

1. **Track N — Native 48-feature validation**
2. **Track M — Method-level replication**
3. **Track P — Optional reduced public contract**
4. **Track I — Institutional evaluation**

The tracks must not be conflated.

In particular:

- `BLOCKED` is not a failed experiment;
- `FAILED` means an experiment actually ran and failed its stated criterion;
- `INCONCLUSIVE` means the experiment cannot support a GO or NO-GO decision under the frozen evidence/power rules;
- `NOT ESTABLISHED` means the evidence required for a claim does not yet exist;
- method-level results do not establish native 48-feature generalization;
- reduced-contract results do not establish the value of native-only features.

---

# 2. Research Questions

## Track N — Native 48-Feature Model

### RQ-N1

Can the native `21-domain-feature → 48-native-feature` model be evaluated on an independently obtained dataset satisfying the complete feature contract?

### RQ-N2

If an eligible independent dataset becomes available, does the native model satisfy the preregistered discrimination, calibration, robustness, and reproducibility criteria?

### RQ-N3

Do the native device/location/recipient-graph features provide measurable value relative to preregistered ablations?

### Current status

**BLOCKED_PENDING_ELIGIBLE_DATASET**

The public benchmark registry currently provides no confirmed Group-A dataset for the complete native contract.

This is a data-eligibility block, not a failed native-model experiment.

---

# 3. Track M — Method-Level Replication

Track M trains and evaluates **new dataset-specific models** rather than reusing the existing native model.

The purpose is to test whether the methodological approach generalizes across independently available datasets.

This track does **not** claim that dataset-specific models reproduce the native 48-feature model.

## Prior design exposure

Although the models in Track M are newly trained, prior project work has already informed:

- architecture choices;
- feature ideas;
- gating concepts;
- evaluation design;
- hyperparameter search practices.

Therefore Track M is not treated as if the research team had no prior information.

Before touching an untouched confirmatory dataset:

- architecture families must be frozen;
- feature-construction rules must be frozen;
- hyperparameter ranges must be frozen;
- tuning budget must be frozen;
- preprocessing rules must be frozen;
- selection criteria must be frozen.

No search over IEEE-CIS or BAF may expand these choices after inspecting their results.

---

## RQ-M1 — Baseline comparison

Does the frozen ensemble outperform the frozen single-model baseline on eligible datasets?

### Primary metric

Chosen from Section 4A. `[TO BE FROZEN]`

### Ensemble identity

The exact ensemble implementation must be frozen by repository artifact/configuration.

Candidate implementations known from prior work include:

- XGBoost + LightGBM + CatBoost;
- stacked/fusion approaches involving LR + RF + XGB + IF.

These are not interchangeable.

**Selected ensemble:** `[TO BE FROZEN]`

### Baseline identity

**Primary baseline:** `[TO BE FROZEN]`

The baseline and ensemble must use:

- the same dataset;
- the same permitted feature representation;
- the same train/validation/test boundaries;
- the same preprocessing policy;
- the same tuning budget;
- the same information-availability rules.

A baseline may not be deliberately under-tuned.

### Dataset-level success

A dataset-level success requires:

- ensemble improvement on the primary metric;
- preregistered paired confidence interval excluding zero;
- adequate test population;
- no frozen validity violation.

### Track-level aggregation

Because the number of untouched confirmatory public datasets may be as low as one or two, the aggregation rule must explicitly handle both cases.

**If two eligible confirmatory datasets exist:**

`[TO BE FROZEN — e.g. predefined rule based on both dataset-level outcomes]`

**If exactly one eligible confirmatory dataset exists:**

`[TO BE FROZEN — e.g. result is dataset-specific evidence and cannot by itself establish broad cross-dataset replication]`

**If zero eligible confirmatory datasets exist:**

`BLOCKED`

No majority/all-dataset rule may be chosen after results are observed.

### Multiplicity

**Multiplicity method:** `[TO BE FROZEN]`

The method must specify how multiple datasets, metrics, and secondary hypotheses are handled.

---

# 4. RQ-M2 — Ensemble Contribution and Ablation

The ensemble must be compared with its relevant constituent models and preregistered ablations.

Required candidates:

- exact selected ensemble;
- selected single-model baseline;
- relevant constituent models;
- preregistered feature/method ablations.

The ablation set must be frozen before confirmatory results are examined.

Ablations may not be added solely because they produce a favourable explanation.

---

# 4A. Performance Measures

This section defines which performance measures are used, how they are tiered, and how they are reported. Metric formulas live in one place, `docs/metric_definitions.md`, at a version recorded in the freeze record. This plan does not redefine them.

## 4A.1 Tiers

| Tier | Role | Rule |
|---|---|---|
| **Primary** | Decides Track M signal (RQ-M1) | Exactly one metric, frozen before results |
| **Secondary** | Reported with confidence intervals; may support but never override the primary | Frozen list |
| **Diagnostic** | Explains behaviour; no decision weight | Frozen list; additions after freeze are exploratory |

**Primary metric:** `[TO BE FROZEN — one metric from the table below]`

Fraud data is heavily imbalanced, so accuracy is never reported as a performance measure, and ROC-AUC is never the only headline number.

## 4A.2 Measures

| Group | Measure | Question it answers | Notes that must be fixed in advance |
|---|---|---|---|
| Ranking | **ROC-AUC** | How well are frauds ranked above non-frauds? | Chance is 0.50. Can look strong while precision is poor under heavy imbalance. |
| Ranking | **PR-AUC** | How well is precision held as recall rises? | Chance level equals prevalence, so report prevalence beside it. Metric definitions v1.0 specifies average precision (`average_precision_score`), not trapezoidal interpolation. Adopt it unchanged. |
| Operating point | **Recall at fixed FPR** | How much fraud is caught at an acceptable false-alarm rate? | FPR levels: `[TO BE FROZEN]`. Report the FPR actually achieved on test, not only the target. |
| Operating point | **Precision at fixed recall** | How clean are the alerts at a required catch rate? | Recall levels: `[TO BE FROZEN]`. |
| Alert volume | **Recall and precision at fixed alert volume** (top-k or top x%) | What happens at the volume a team can review? | Volume levels and unit: `[TO BE FROZEN]`. |
| Threshold-based | **Confusion matrix, precision, recall, F-beta** at the frozen threshold | What does the deployed rule do? | Threshold chosen on the validation window only. Beta: `[TO BE FROZEN]`. Always report TP, FP, TN, FN counts. |
| Cost | **Expected cost or value captured** | What is the monetary-style trade-off? | Cost matrix: `[TO BE FROZEN]`. Without real cost data, label it illustrative, never institutional. |
| Calibration | **Brier score, log loss** | Are probabilities accurate? | Computed on the test window with the calibrator fitted earlier. |
| Calibration | **ECE and reliability diagram** | Do stated risks match observed rates? | Binning scheme and bin count: `[TO BE FROZEN]`. ECE is sensitive to binning, so report the diagram too. |
| Calibration | **Calibration slope and intercept** | Is risk over- or under-stated? | Method: `[TO BE FROZEN]`. |
| Gating | **Coverage, abstention rate, wrong-confident rate, risk-coverage curve, fallback quality, recall loss** | Does the gate cut confident errors at an acceptable cost? | Definitions follow Section 5. |
| Stability | **Seed-to-seed spread; per-window metric over time** | Is the result stable? | Report SD and range across seeds, and the metric per temporal window. |
| Segments | Any measure above, per frozen segment | Where does it work or fail? | Segments frozen in advance (RQ-M9). |

## 4A.3 Reporting rules

For every reported measure:

- the point estimate and the dependence-aware confidence interval (Section 19);
- the number of positives and negatives in the evaluation set;
- dataset, split, window, seed set and model identity;
- where a threshold is used, the window it was selected on;
- for model comparisons, the **paired difference** with its interval, not two separate intervals.

Further rules:

- Operating-point measures are reported with the achieved rate on test. A large gap between the validation-selected and test-achieved FPR is itself a result.
- Ties in scores are handled by one frozen rule.
- A measure whose interval cannot be estimated, or whose test window fails the power rule in Section 18, is reported as `INCONCLUSIVE`, not as a number alone.
- Every metric implementation is tested against a reference on a fixed fixture, and the metric-definition version is recorded.
- No measure may be added, dropped or redefined after the freeze, except as a clearly labelled exploratory addition.

## 4A.4 Alignment with `docs/metric_definitions.md` (v1.0)

Checked against the public repository copy. This table must be re-verified against the version recorded at freeze.

| Measure group | Covered by v1.0? | Action before freeze |
|---|---|---|
| ROC-AUC, PR-AUC, recall at stated FPR, validation-only threshold discipline | Yes | Adopt unchanged |
| Class-imbalance reporting, baselines, minimum sample conditions, confidence intervals | Yes | Reconcile with Sections 18 and 19 (see below) |
| Precision/recall at fixed alert volume | No | Add definitions and tests |
| F-beta other than F1 | No (F1 only) | Add only if a different beta is frozen |
| Brier, log loss, ECE, reliability, calibration slope and intercept | No | Add definitions, binning and tests |
| Expected cost / value captured | No | Add only if a cost matrix is frozen |
| Coverage, abstention, wrong-confident rate, risk-coverage, fallback quality | No | Add definitions that match Section 5 |
| Seed and temporal-window stability | No | Add reporting definitions |

**Reconciliation required.** v1.0 withholds a confidence interval when a conditioning cell has fewer than 30 observations. That is a minimum for reporting an interval, not a power rule. The confirmatory minimum fraud and sample counts in Section 18 are separate, are expected to be much stricter, and take precedence for GO / NO-GO. The interval method in v1.0 must also be made consistent with the dependence-aware method in Section 19. A bump to the definitions version is required, and the new version is what the freeze record hashes.

---

## 4A.5 Operational performance (separate from detection performance)

Latency, throughput and failure behaviour are engineering measures, tracked under Section 27. They are not part of any scientific success criterion and are never combined with detection metrics.

When reported they must state the hardware, the database backend (SQLite or PostgreSQL), concurrency level, audit-write mode, and p50, p95 and p99 latency, throughput and error rate.

---

# 5. RQ-M3 — Gating Effectiveness

The safety gate must be evaluated against a matched ungated system.

## 5.0 Unit under test

The gate is not one mechanism. Two components must be evaluated separately and named in every result:

| Component | Unit | Designed to detect |
|---|---|---|
| **Row-level enforcement** (contract, range, NaN/Inf, schema checks before inference) | One transaction | Invalid or out-of-contract inputs |
| **Window-level drift monitoring** (e.g. PSI over a window, with its alert/block rule) | A window of transactions | Distribution shift |

Class A (Section 5.2) primarily tests row-level enforcement. Class B primarily tests window-level drift monitoring and whatever blocking rule it triggers. Row-level enforcement is not expected to detect subtle shift, and a Class B result must not be credited to it.

For Track M the native 48-feature contract and drift baselines do not apply. For each dataset the following must be defined and frozen **before** inspecting that dataset's confirmatory data, using training-window data only:

- the dataset-specific feature contract (types, ranges, missing policy);
- the drift baseline distribution and the window size;
- the alert/block thresholds;
- the rule by which a window-level block affects row decisions.

No contract, baseline or threshold may be derived from validation or test windows. If a component cannot detect a perturbation class by design, the result is reported as `NOT DESIGNED TO DETECT`, not omitted.

**Frozen gate-component definitions:** `[TO BE FROZEN]`

---

## 5.1 Ungated definition

The **ungated system** is:

> The same model, preprocessing, imputation, calibration, threshold, and evaluation pipeline, with the safety/enforcement gate bypassed.

The ungated system must not be allowed to crash merely because an injected input is invalid.

The same preregistered imputation behaviour must be applied where imputation is part of the frozen pipeline.

Any difference between gated and ungated systems must therefore be attributable to the gating decision rather than inconsistent preprocessing.

---

## 5.2 Two perturbation classes

Perturbations are divided before results are examined.

### Class A — Designed-for detection

Examples may include:

- NaN/invalid values;
- explicit out-of-range values;
- schema violations;
- known invalid/provenance conditions.

Exact injection definitions:

`[TO BE FROZEN]`

These results test whether the implemented gate catches conditions it was designed to catch.

They are reported separately and are **not sufficient by themselves** to support the primary gating-effectiveness claim.

### Class B — Not-designed-for detection

The primary gating claim uses plausible but shifted conditions such as:

- subtle distribution drift;
- plausible but shifted feature values;
- other preregistered drift conditions not explicitly encoded as deterministic validation rules.

Exact perturbation families:

`[TO BE FROZEN]`

Exact magnitudes/severity levels:

`[TO BE FROZEN]`

These conditions are the primary test of whether gating provides protection beyond deterministic input validation.

### Perturbation rules (Class A and Class B)

- Perturbations modify **feature values only**. Labels are never altered, re-derived, or resampled.
- Wrong decisions are judged against the **original, unperturbed labels**.
- Perturbation families, magnitudes, affected-feature proportions, windows of application and random seeds are frozen before any confirmatory result is inspected.
- Perturbations are applied to the evaluation window only. Training and calibration data stay unperturbed unless a frozen experiment states otherwise.
- The same perturbed data is given to the gated and ungated systems.

---

## 5.3 Confident decisions

A confident decision is defined by:

`[TO BE FROZEN]`

The definition must be applied identically to the model and fallback/rules pathway.

---

## 5.4 Wrong confident decisions

A wrong confident decision includes:

1. a confident ML prediction that is wrong; **or**
2. a confident fallback/rules-engine decision that is wrong.

A gate may not reduce measured errors by simply relabelling rejected cases as "fallback."

Fallback errors must therefore be counted in the same primary error accounting.

---

## 5.5 Primary gating criterion

The primary gating claim requires, for Class B perturbations:

- fewer wrong confident decisions than the matched ungated system;
- minimum reduction:

`[TO BE FROZEN]`

while also satisfying:

- minimum coverage:

`[TO BE FROZEN]`

- maximum coverage loss:

`[TO BE FROZEN]`

- maximum recall loss:

`[TO BE FROZEN]`

- fallback/rules quality floor:

`[TO BE FROZEN]`

A gate that rejects all or nearly all cases cannot pass because its coverage constraint will fail.

Class A and Class B results must be reported separately.

---

# 6. RQ-M4 — Calibration

Specify exactly where calibration occurs.

For temporal evaluation:

**Training window → calibration/validation window → test window**

The calibration window must be later than training and earlier than the test window.

The test window must not influence calibrator fitting.

**Calibration method:** `[TO BE FROZEN]`

**Primary calibration metric:** `[TO BE FROZEN]`

**Acceptance bound:** `[TO BE FROZEN]`

---

# 7. RQ-M5 — Drift and Robustness

Evaluate preregistered distribution shifts and degraded-data conditions.

Report:

- absolute performance;
- degradation from the unperturbed condition;
- confidence intervals;
- gating behaviour where applicable;
- coverage changes.

Negative robustness results remain part of the research record.

---

# 8. RQ-M6 — Temporal Generalization

Where meaningful timestamps exist:

- training precedes calibration/validation;
- calibration/validation precedes test;
- feature availability is restricted to the prediction timestamp;
- no future information influences preprocessing or feature construction.

If temporal evaluation is not meaningful:

**NOT APPLICABLE**, with justification.

---

# 9. RQ-M7 — Entity-Disjoint Generalization

Where meaningful entity identifiers exist, prevent the same relevant entity from crossing the required evaluation boundaries.

If entity-disjoint evaluation is impossible or semantically meaningless:

**NOT APPLICABLE**, with justification.

---

# 10. RQ-M8 — Business and Alert-Volume Metrics

Where dataset semantics permit:

- recall at fixed alert volume;
- recall at fixed alert-rate/FPR;
- precision at fixed alert volume;
- alert count;
- analyst-review burden or a clearly labelled proxy.

**Primary business metric:** `[TO BE FROZEN]`

**Proxy definition, if needed:** `[TO BE FROZEN]`

A proxy must not be described as measured institutional analyst burden.

---

# 11. RQ-M9 — Segment Analysis

Evaluate only preregistered, semantically justified segments.

Candidate dimensions:

- transaction amount bands;
- temporal periods;
- dataset-defined categories;
- other frozen domain-relevant partitions.

**Segment definitions:** `[TO BE FROZEN]`

The project must not search arbitrary segments after seeing results.

---

# 12. RQ-M10 — Point-in-Time Correctness and Label Delay

Where timestamps and/or delayed labels exist, verify:

- feature availability at prediction time;
- no future-information leakage;
- label observation timing;
- delayed-label handling;
- training/evaluation eligibility under the frozen observation window.

If the dataset lacks the necessary information:

**NOT APPLICABLE**, with justification.

This requirement applies to Track M wherever relevant dataset information exists.

---

# 13. Dataset Eligibility for Track M

Each candidate dataset must have:

- provenance;
- licence/access status;
- acquisition status;
- dataset hash/version;
- feature semantics;
- label semantics;
- temporal information;
- entity identifiers;
- leakage assessment;
- preprocessing requirements;
- real/synthetic classification;
- prior project exposure;
- applicable research track;
- eligibility decision.

### Synthetic data

Synthetic data may be used only when explicitly classified `SIMULATED/SYNTHETIC`.

Synthetic results cannot establish real-world performance.

### Provenance-unclear data

Unresolved provenance/licensing prevents confirmatory use.

---

# 14. Dataset Exposure History

Prior exposure is part of dataset identity.

| Dataset | Prior project exposure | Existing-model status | New Track-M confirmatory status |
|---|---|---|---|
| ULB | Used during development/evaluation | **EXPLORATORY** | May be used only under explicitly frozen new-model rules |
| Kaggle fraudTrain/fraudTest | Used during development/evaluation | **EXPLORATORY** | May be used only under explicitly frozen new-model rules |
| IBM v2 | Used during development/evaluation | **EXPLORATORY** | May be used only under explicitly frozen new-model rules |
| IEEE-CIS | No established prior model exposure | Candidate | **CANDIDATE CONFIRMATORY** pending eligibility |
| BAF | No established prior model exposure | Candidate | **CANDIDATE CONFIRMATORY** pending eligibility |

The exposure ledger must be independently audited before the confirmatory freeze.

Prior exposure of model design choices means that "new model" does not mean "no prior information."

For untouched datasets, architecture families, feature-construction rules, hyperparameter ranges, tuning budgets, preprocessing, and selection rules must be frozen before data inspection that could influence them.

---

# 15. Existing External-Transfer Evidence

The prior transfer results must be separated by experiment.

| Evidence | Status | Interpretation |
|---|---|---|
| ULB in-domain | **DEMONSTRATED / SELF-TESTED according to existing evidence record** | In-domain performance; not independent external validation |
| IBM v2 in-domain | **DEMONSTRATED / SELF-TESTED according to existing evidence record** | In-domain performance |
| IBM cross-dataset (~0.873) | **EXPLORATORY / NOT INDEPENDENT** | Cross-dataset result but dataset/model exposure prevents independent confirmatory interpretation |
| Kaggle transfer (~0.435–0.595) | **FAILED / NON-CONFORMING** | Actual evaluation under degraded/non-conforming feature representation |

The exact status labels must follow the repository's evidence ledger where already established.

The Kaggle result is a real failed evaluation under a non-conforming contract.

It does not establish failure of a conforming native 48-feature evaluation.

---

# 16. Track P — Optional Reduced Public Contract

Track P is permitted only if at least:

`[N — TO BE FROZEN]`

independently eligible datasets share a sufficiently large and semantically defensible feature set.

The threshold must be frozen before results are compared.

A shared contract containing only superficially similar fields is insufficient.

Track P can test:

- ensemble vs baseline;
- gating;
- calibration;
- drift;
- temporal behaviour;
- entity-disjoint behaviour where supported.

Track P cannot establish:

- native 48-feature generalization;
- value of native-only features;
- equivalence between reduced and native contracts.

---

# 17. Track I — Institutional Evaluation

Institutional evaluation requires authorized access to suitable data and evaluation conditions.

Potential requirements include:

- native feature availability;
- feature semantics;
- fraud labels;
- timestamps;
- entity information;
- point-in-time availability;
- authorized research use;
- reproducible evaluation environment.

Current status:

**BLOCKED / NOT ESTABLISHED**

No institutional effectiveness claim may be made before these dependencies exist.

---

# 18. Statistical Power

Fraud labels may be rare and operating-point estimates may be unstable.

Every confirmatory test window must satisfy:

**Minimum fraud count:** `[TO BE FROZEN]`

**Minimum total evaluation count:** `[TO BE FROZEN]`

**Minimum valid seeds/runs:** `[TO BE FROZEN]`

If these requirements are not satisfied:

**INCONCLUSIVE / UNDERPOWERED**

The result does not count as GO or NO-GO for the affected criterion.

---

# 19. Dependence-Aware Confidence Intervals

Transaction observations may be correlated by card, account, recipient, device, household, or other entities. Temporal data may also contain serial dependence.

Ordinary IID bootstrap is therefore not the default confirmatory method.

### Where meaningful entity identifiers exist

Use:

**Entity-clustered bootstrap:** `[TO BE FROZEN]`

Clusters must be sampled at the preregistered entity level.

### Where temporal dependence is material

Use:

**Temporal/block bootstrap:** `[TO BE FROZEN]`

Block construction:

`[TO BE FROZEN]`

### Where neither meaningful clustering nor temporal dependence exists

Use:

**Independent-observation bootstrap:** `[TO BE FROZEN]`

The applicability rule must be frozen before results are examined.

The same dependence policy must be used consistently for paired model comparisons.

---

# 20. Above-Chance Definition

For ROC-AUC:

**Above chance requires the lower confidence bound to exceed 0.50.**

Where baseline comparison is required:

**The lower confidence bound of the paired difference must exceed zero.**

For metrics without a 0.50 chance reference, the appropriate preregistered reference value must be defined separately.

A point estimate above 0.50 alone is insufficient.

---

# 21. Primary Success Criteria

### Signal

The selected ensemble must satisfy:

- improvement over the frozen baseline;
- statistical criterion;
- adequate evaluation population.

Primary metric:

`[TO BE FROZEN]`

Minimum effect:

`[TO BE FROZEN]`

### Transfer

An independent confirmatory dataset must:

- satisfy eligibility;
- clear the frozen performance floor;
- satisfy the statistical above-chance rule.

Performance floor:

`[TO BE FROZEN]`

### Gating

Class B perturbations must show the preregistered reduction in wrong confident decisions while satisfying coverage and recall constraints.

### Calibration

Calibration error must remain within:

`[TO BE FROZEN]`

### Reproducibility

A clean reproduction must match within:

`[TO BE FROZEN]`

---

# 22. Decision Outcomes

Each research track can produce the following outcomes:

| Outcome | Meaning |
|---|---|
| **GO** | All required primary criteria satisfied |
| **NO-GO** | Adequately powered eligible evaluation ran and a required criterion failed |
| **REVISE** | A frozen revision condition explicitly permits one targeted revision |
| **INCONCLUSIVE** | Evidence is insufficient to establish success or failure, including underpowered evaluation |
| **BLOCKED** | Required data/access/dependency is unavailable |
| **NOT ESTABLISHED** | Evidence required for the claim has not been demonstrated |
| **NOT APPLICABLE** | Criterion genuinely does not apply to the dataset/track |

These outcomes must not be collapsed.

---

# 23. Track-Level Go/No-Go

## Track N

No conforming eligible dataset:

**BLOCKED**

Conforming dataset + adequately powered failed criterion:

**NO-GO**

Conforming dataset + all primary criteria satisfied:

**GO**

## Track M

The two-dataset and one-dataset aggregation rules must be frozen separately.

### Two eligible confirmatory datasets

`[TO BE FROZEN]`

### One eligible confirmatory dataset

`[TO BE FROZEN]`

A single eligible dataset may provide dataset-specific evidence but must not automatically be described as broad replication.

### Zero eligible confirmatory datasets

**BLOCKED**

### Underpowered dataset

**INCONCLUSIVE**, not NO-GO.

## Track M roll-up

Outcomes are produced at three levels in this order: **per-RQ per-dataset → dataset → track.**

**Primary RQs** (these determine the dataset outcome): `[TO BE FROZEN — candidates: RQ-M1, RQ-M3 Class B, RQ-M4]`

**Supporting RQs** (always reported, never used to override a primary RQ): all others.

Proposed dataset-level rule, subject to statistical reviewer approval (`[TO BE FROZEN — reviewer to confirm or replace]`):

1. Any primary RQ `NO-GO` with adequate power → dataset `NO-GO`.
2. Otherwise any primary RQ `INCONCLUSIVE` → dataset `INCONCLUSIVE`.
3. Otherwise all primary RQs `GO` → dataset `GO`.
4. A primary RQ may be `NOT APPLICABLE` only if the justification was frozen in advance.

The full **per-RQ outcome profile** must be published alongside the dataset outcome, so a strong signal result is not hidden by a failure elsewhere, and the reverse.

## Two-dataset outcome matrix

Dataset outcomes are `GO`, `NO-GO`, `INCONCLUSIVE` or `BLOCKED`. Frozen track outcome for each combination:

| Dataset 1 | Dataset 2 | Track M outcome |
|---|---|---|
| GO | GO | GO (replication on two datasets) |
| GO | NO-GO | `[TO BE FROZEN]` |
| GO | INCONCLUSIVE | `[TO BE FROZEN]` |
| GO | BLOCKED | Dataset-specific GO only; not broad replication (proposed) |
| NO-GO | NO-GO | NO-GO |
| NO-GO | INCONCLUSIVE | `[TO BE FROZEN]` |
| NO-GO | BLOCKED | Dataset-specific NO-GO (proposed) |
| INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| INCONCLUSIVE | BLOCKED | INCONCLUSIVE |
| BLOCKED | BLOCKED | BLOCKED |

Rows marked "proposed" and all frozen outcomes require statistical reviewer approval. The one-dataset rule must also be frozen and be consistent with this matrix.

---

# 24. REVISE Rules

A REVISE outcome is permitted only where the preregistered failure condition explicitly allows it.

At most **one revision cycle** is permitted.

The revision must:

1. preserve the original result;
2. identify the failure;
3. define the narrow corrective action;
4. obtain required approval;
5. run as a separately identified experiment;
6. be classified as exploratory if outside the original confirmatory specification.

Repeated iteration until a favourable result appears is prohibited.

## 24.1 Eligible REVISE triggers

A revision may be triggered only by a **documented defect in the experiment or pipeline**, established independently of whether the result was favourable. Eligible triggers:

- a confirmed data-handling or preprocessing bug;
- a confirmed leakage finding (target, temporal, entity or preprocessing);
- a confirmed deviation from the frozen protocol;
- corrupted or incomplete input data confirmed by checksum or source comparison.

**Not eligible:** a null or negative primary result, a confidence interval containing zero, a failed threshold, an unfavourable ablation, or a wish to try another model or feature set.

Any further trigger types: `[TO BE FROZEN]`

Each revision record must state the defect, the evidence that it exists, and the narrow corrective action, and must be approved by someone other than the experimenter.

## 24.2 INCONCLUSIVE consequences

An `INCONCLUSIVE` outcome (including UNDERPOWERED) is not NO-GO and not GO. Permitted next steps:

1. Report as `INCONCLUSIVE` and stop. This is the default.
2. Run one additional evaluation on a larger or later window or additional eligible data, only if its trigger and design were frozen in advance (`[TO BE FROZEN]`) and it is recorded as a separate experiment with the original retained.

Not permitted: re-splitting, changing the primary metric, lowering a power requirement, or pooling in other data to reach significance.

---

# 25. Failed-Result Preservation

Original failed experiments must never be deleted, overwritten, silently rerun under changed conditions, or relabelled as successful.

The evidence record must preserve:

- original configuration;
- dataset;
- split;
- seed;
- result;
- status;
- revision;
- revised result;
- reason for revision.

---

# 26. NO-GO Consequences

A NO-GO does not erase scientific value.

### Generalization diagnosis

Investigate where and why performance failed.

### Robustness diagnosis

Characterize relevant failure modes.

### Reproducibility

A failed result remains subject to the full reproducibility requirement.

### Collaboration package

The final research package may document:

- methodology;
- datasets;
- preregistration;
- results;
- failures;
- limitations;
- reproducibility evidence;
- unresolved dependencies;
- future research questions.

---

# 27. Engineering Scope

The following are **out of scope for this research plan's scientific success criteria**:

- admin console expansion;
- review queue expansion;
- attention center work;
- unrelated UI/product modules.

Engineering hardening is tracked separately:

- PostgreSQL evaluation;
- TLS;
- secret/key management;
- persistent observability;
- asymmetric signing;
- repository consolidation.

Only engineering changes necessary to execute the frozen research protocol may be introduced before confirmatory execution.

After scientific freeze, confirmatory conditions must not be changed through unrelated engineering work.

Engineering improvements cannot be used as evidence that a scientific NO-GO became a GO.

---

# 28. Manual Dependencies

| Dependency | Owner | Action | Deadline | Status |
|---|---|---|---|---|
| IEEE-CIS | `[OWNER]` | Read terms/use restrictions before download; verify labelled-data availability; obtain/download if permitted | `[DATE]` | **PENDING** |
| BAF | `[OWNER]` | Read licence/terms; verify provenance and generation process; classify real/synthetic; obtain/download if permitted | `[DATE]` | **PENDING** |
| Exposure audit | `[OWNER]` | Audit repository evidence for prior dataset exposure | `[DATE]` | **PENDING** |
| Statistical review | `[STATISTICAL REVIEWER]` | Sign off every section listed in Section 32 before freeze | `[DATE]` | **PENDING** |
| Domain review | `[DOMAIN REVIEWER]` | Review outcome/metric semantics | `[DATE]` | **PENDING** |

### IEEE-CIS checks

Before treating IEEE-CIS as confirmatory:

- verify competition/data-use restrictions;
- read the applicable terms before downloading;
- verify which labelled portion is actually available;
- verify that the available labelled data can satisfy the preregistered temporal split;
- verify that the required fraud count is achievable in the test window.

Any unavailable competition test labels must be recorded accurately if confirmed; this plan must not assume their availability without verification.

### BAF checks

Before treating BAF as confirmatory:

- verify the licence and access terms;
- read the dataset documentation/paper;
- establish how the data were generated;
- determine whether it should be classified as real, synthetic, privacy-preserving synthetic, or another explicit category;
- verify feature and label semantics;
- verify whether it satisfies the required temporal/entity/power rules.

---

# 29. Protocol Freeze Mechanism

The plan and preregistration are frozen at a concrete repository state.

A document cannot contain its own hash. The freeze metadata therefore lives in a **separate file**, `docs/FREEZE_RECORD.json`, not in this plan. It must contain: git tag, git SHA, SHA-256 of this plan, of the preregistration, of the metric definitions, of the dataset eligibility manifest, of the exposure ledger and of the freeze-check script itself, the freeze timestamp, owner, statistical reviewer and approval timestamp.

The freeze-check script is part of the frozen artifact set: changing it after the freeze invalidates the freeze.

---

# 30. Automated Freeze Check

`scripts/check_freeze.py` must:

1. scan this plan and the preregistration for **any** unresolved placeholder: a bracketed marker beginning TO BE FROZEN, OWNER, DATE, HASH, SHA, VERSION, TAG, NAME, TIMESTAMP, STATISTICAL, DOMAIN or N, or any bracketed token written entirely in capitals;
2. require every field in `docs/FREEZE_RECORD.json` to be present and non-placeholder;
3. recompute SHA-256 for every listed artifact, including the script itself, and compare it with the record;
4. confirm statistical reviewer approval fields are present;
5. exit non-zero on any failure.

It runs in CI on every commit that touches the plan, the preregistration or the record, and again before the freeze tag is created. A freeze is not valid because a person declared it, only because the check passed on the tagged commit.

---

# 31. Evidence Requirements

Every quantitative result must identify:

- experiment ID;
- dataset identity/hash/version;
- dataset exposure status;
- dataset role;
- split;
- population;
- seed;
- model identity;
- feature-contract identity;
- preprocessing identity;
- calibration configuration;
- threshold;
- metric-definition version;
- statistical configuration;
- protocol version/hash;
- research-plan hash;
- Git SHA;
- execution command;
- evidence artifact.

A result missing required provenance cannot be labelled `MEASURED`.

---

# 32. Review and Approval

The statistical reviewer must explicitly approve the statistical decision framework before freeze, including:

- minimum-power rules;
- dependence-aware bootstrap;
- confidence-interval construction;
- multiplicity;
- above-chance definition;
- dataset aggregation;
- GO/NO-GO/REVISE/INCONCLUSIVE rules.

**Sign-off scope (identical to the Section 28 reviewer task):** Section 3 (primary metric, aggregation, multiplicity), Section 4A (performance measures and tiers), Section 5.0 and 5.5 (gate definitions and thresholds), Section 6 (calibration), Sections 18 to 24 (power, dependence-aware intervals, above-chance rule, success criteria, decision outcomes, go/no-go and roll-up, REVISE and INCONCLUSIVE rules), and Section 32 itself.

The approval must be recorded in the evidence ledger.

A single developer's self-approval is not sufficient for the statistical sections.

Domain review must separately verify the semantic appropriateness of the primary outcomes where required.

---

# 33. Explicit Non-Claims

## Track N

Does not claim:

- public-data validation when the native contract is unavailable;
- native feature generalization from reduced/non-conforming datasets;
- institutional effectiveness;
- production readiness.

## Track M

Does not claim:

- native 48-feature generalization;
- value of native-only device/location/recipient features;
- feature equivalence between datasets;
- institutional effectiveness.

## Track P

Does not claim:

- native-model validation;
- native-only feature value;
- equivalence between reduced and native contracts.

## Track I

Does not claim validation until eligible authorized institutional data actually exist.

---

# 34. Current Status

| Evidence / track | Status | Interpretation |
|---|---|---|
| Native 48-feature public validation | **BLOCKED** | No confirmed public Group-A dataset |
| ULB in-domain | **DEMONSTRATED / SELF-TESTED** according to existing evidence record | In-domain evidence; not independent external validation |
| IBM v2 in-domain | **DEMONSTRATED / SELF-TESTED** according to existing evidence record | In-domain evidence |
| IBM cross-dataset (~0.873) | **EXPLORATORY / NOT INDEPENDENT** | Prior exposure prevents independent confirmatory interpretation |
| Kaggle transfer (~0.435–0.595) | **FAILED / NON-CONFORMING** | Actual degraded-contract transfer evaluation |
| Track M method replication | **PROPOSED** | Primary public-data research route |
| Track P reduced public contract | **CONDITIONAL** | Requires preregistered shared-feature eligibility |
| Institutional evaluation | **BLOCKED / NOT ESTABLISHED** | Requires authorized eligible data |

Exact classifications for historical results must follow the repository evidence ledger rather than this summary table overriding recorded evidence.

---

# 35. Pre-Freeze Checklist

- [ ] research questions frozen;
- [ ] exact ensemble identity frozen;
- [ ] baseline identity frozen;
- [ ] tuning budget frozen;
- [ ] dataset exposure history audited;
- [ ] IEEE-CIS terms checked;
- [ ] BAF terms/provenance checked;
- [ ] confirmatory datasets identified;
- [ ] licence/access verified;
- [ ] feature semantics verified;
- [ ] split rules frozen;
- [ ] temporal/entity rules frozen;
- [ ] point-in-time/label-delay rules frozen;
- [ ] Class A gating injections frozen;
- [ ] Class B gating injections frozen;
- [ ] gating magnitudes frozen;
- [ ] confident-decision definition frozen;
- [ ] fallback error accounting frozen;
- [ ] coverage/recall bounds frozen;
- [ ] business metrics frozen;
- [ ] segment definitions frozen;
- [ ] minimum fraud-count rule frozen;
- [ ] dependence-aware CI method frozen;
- [ ] multiplicity method frozen;
- [ ] above-chance rule frozen;
- [ ] one-dataset aggregation rule frozen;
- [ ] two-dataset aggregation rule frozen;
- [ ] GO/NO-GO/REVISE/INCONCLUSIVE rules frozen;
- [ ] one-revision rule frozen;
- [ ] statistical reviewer signed off;
- [ ] domain review completed where required;
- [ ] automated placeholder check passes;
- [ ] research-plan hash recorded;
- [ ] preregistration hash recorded;
- [ ] Git freeze tag created;
- [ ] freeze recorded in evidence ledger.

If any mandatory item remains unresolved, the affected confirmatory experiment is **NOT READY**.

---

# 36. Decision Principle

The project follows:

**Repository evidence → exposure/eligibility audit → research questions → preregistration → independent review → freeze → experiment → evidence record → track-specific decision → diagnosis/reproducibility/collaboration package.**

Results must never determine the rules used to judge those same results.

---

# 34. Referenced Artifacts

Every artifact the plan relies on must exist in the repository and be linked here before freeze.

| Artifact | Path | Status |
|---|---|---|
| Metric definitions | `docs/metric_definitions.md` | Present in public repo at last check; must cover every measure in Section 4A, with version recorded |
| Preregistration | `[TO BE FROZEN — path]` | Not located in public repo at last check |
| Dataset exposure ledger | `[TO BE FROZEN — path]` | Not located in public repo at last check |
| Dataset eligibility manifest | `[TO BE FROZEN — path]` | Not located in public repo at last check |
| Freeze record | `docs/FREEZE_RECORD.json` | To be created at freeze |
| Freeze-check script | `scripts/check_freeze.py` | Draft provided |

If an artifact cannot be produced, the claims that depend on it are `NOT ESTABLISHED`.
