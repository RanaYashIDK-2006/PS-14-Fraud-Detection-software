# Statistical + Domain Review Resolution Package

> **This document prepares and records reviewer decisions. It does not manufacture
> statistical or domain approval.**

**Status:** REVIEW-SUPPORT ARTIFACT — DRAFT. **Not a freeze record. Not an approval
certificate. Not a substitute for a qualified reviewer.**

| | |
|---|---|
| Research Plan | **DRAFT / NOT FROZEN / NOT APPROVED** |
| Preregistration | **DRAFT / NOT APPROVED** (`0.1.2-draft`, "Finalized: NO") |
| Statistical reviewer | **`NOT ASSIGNED`** |
| Domain reviewer | **`NOT ASSIGNED`** |
| Confirmatory Track M | **BLOCKED** |
| NR-05 | COMPLETE / EXPLORATORY |

**Decisions resolved by this package: zero.** No reviewer identity, approval,
timestamp, statistical decision, threshold, metric, dataset eligibility or
experimental outcome is created here. Where the repository already establishes a
fact mechanically, it is recorded as `REPOSITORY-MECHANICAL VERIFICATION` and
still marked for reviewer confirmation — resolving a placeholder is a separate,
later, explicitly-approved act.

**Authored at git state:** `5a5ff55cc03df73318fdf31a16e3665f159a4720` (2026-10-04).
No commit is implied.

---

## 1. Authoritative sources

Read before filling anything. Repository state, not recollection, governs.

| Artifact | Path | Role here |
|---|---|---|
| Research Plan | `docs/RESEARCH_PLAN.md` | normative; owner of every `[TO BE FROZEN]` cell |
| Preregistration | `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` | 11 `[REQUIRES DECISION/APPROVAL]` markers |
| Statistical review memo | `docs/evaluation/STATISTICAL_REVIEW_DECISION_MEMO.md` | **authoritative list of the 14 decisions** (`/01`–`/14`) |
| NR-05 diagnostics | `docs/PHASE_NR05_EXPLORATORY_DIAGNOSTICS.md` | all exploratory evidence, including the negatives (§5 here) |
| NR-05 driver / results | `backend/scripts/nr05_diagnostics.py`, `reports/nr05/` | reproducible code and machine-readable outputs |
| Evidence ledger | `reports/evaluation_runs/eval_ledger.jsonl` (`eval_record.py` v1.1) | append-only; 93 records at authoring, 12 `nr05_*` |
| Exposure ledger | `docs/evaluation/DATASET_EXPOSURE_LEDGER.md` (`1.0-draft`) | prior exposure per dataset |
| Eligibility manifest | `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` (`1.0-draft`) | §13 identity fields and blocking gates |
| Metric definitions | `docs/metric_definitions.md` (v1.0) | metric implementations, CI withholding, small-sample rules |
| Authoritative contracts | `docs/authoritative_contracts.md` | canonical shapes |
| Freeze check | `scripts/check_freeze.py`, `backend/scripts/check_freeze_test.py` | plan §30 gate |
| Claim enforcement | `backend/scripts/claim_evidence_check.py` | claim → evidence integrity |
| Phase records | `docs/PHASE_RP01_RESEARCH_PLAN_ADOPTION.md`, `docs/PHASE_106_ROADMAP_RECONCILIATION.md` | adoption, freeze ordering, roadmap |

**Identifier convention:** `/01`…`/14` are the memo's existing identifiers. No new
phase-numbering scheme is introduced anywhere in this package.

---

## 2. The fourteen decisions

Statuses are carried over **unchanged** from the memo. None was altered to make the
table look complete.

| ID | Decision | Reviewer Type | Current Status | Decision Required | Allowed Options | Evidence | Freeze Impact | Reviewer | Approval |
|---|---|---|---|---|---|---|---|---|---|
| `/01` | Baseline identity | STATISTICAL REVIEW | `REQUIRES DECISION` | Name the single-model primary baseline the ensemble must beat | A fixed family a-priori · per-dataset on validation only · both with one primary · reject/defer | NR-05 §6, §7 (LR vs XGB; strongest single differs per dataset) | Blocks freeze; blocks primary Track M (RQ-M1) | `NOT ASSIGNED` | `NOT APPROVED` |
| `/02` | Ensemble identity | STATISTICAL REVIEW | `REQUIRES DECISION` | Freeze the exact ensemble implementation despite mean fusion losing to the best single model | fixed ensemble · strongest single-model as primary arm · pre-specified architecture comparison · another named family · reject/defer | NR-05 §6, §8 (mean fusion loses on all 4 runs, CIs exclude 0) | Blocks freeze; blocks primary Track M (RQ-M1) | `NOT ASSIGNED` | `NOT APPROVED` |
| `/03` | Confident-decision definition | STATISTICAL REVIEW + DOMAIN REVIEW | `REQUIRES DECISION` | Freeze the "confident decision" band replacing NR-05's `p≥0.8 or p≤0.2` | fixed numeric band · quantile band on cal window · two frozen bands · reject/defer | NR-05 §9, §11–§12 | Blocks freeze; blocks Class B primary only | `NOT ASSIGNED` | `NOT APPROVED` |
| `/04` | Gating bounds (§5.5) | BOTH (statistical bounds + domain cost posture) | `NOT ESTABLISHED` + `REQUIRES DECISION` | Freeze the five §5.5 bounds: min reduction, min coverage, max coverage loss, max recall loss, fallback quality floor | protocol ≥20% / ≤2pp proposal · other a-priori bounds · coverage–risk curve criterion · reject/defer | NR-05 §11–§12 (ULB recall 0.813→0.040); preregistration Criterion 3 MMD (unapproved assumption) | Blocks freeze; blocks Class B primary only | `NOT ASSIGNED` | `NOT APPROVED` |
| `/05` | Feature-contract construction (§5.0) | STATISTICAL REVIEW + DOMAIN REVIEW (semantics) | `REQUIRES DECISION` | Freeze how each dataset's contract (types, ranges, missing policy) is built | semantic bounds · statistical quantiles with per-type padding · contract on frozen train subsample · reject/defer | NR-05 §9, §20 (72/75 ULB frauds blocked; IBM `chip_code` 70.6% OOV; 33,685 NaN amounts) | Blocks freeze; blocks primary Track M | `NOT ASSIGNED` | `NOT APPROVED` |
| `/06` | Window-gate definition | STATISTICAL REVIEW + DOMAIN REVIEW (material shift) | `NOT ESTABLISHED` + `REQUIRES DECISION` | Freeze baseline window, window size, any-feature vs fraction rule, block rule | PSI-magnitude-calibrated · fraction rule + minimum-drift guard · narrow claim to `NOT DESIGNED TO DETECT` · reject/defer | NR-05 §9, §11 (ULB clean PSI 1.46 → coverage 0; Kaggle Class B max PSI 0.083) | Blocks freeze; blocks Class B primary only | `NOT ASSIGNED` | `NOT APPROVED` |
| `/07` | Calibration (§6) | STATISTICAL REVIEW + DOMAIN REVIEW (acceptance bound meaning) | `NOT ESTABLISHED` + `REQUIRES DECISION` | Freeze calibration method, primary metric, acceptance bound, sanity guard, calibration-positive floor | Platt + guard + floor mandatory · other monotone calibrator + guard · calibration-free discrimination primary · reject/defer | NR-05 §13 (IBM inversion coef −0.7029, 185 cal positives, test AUC 0.373 below chance) | Blocks freeze; blocks primary Track M (RQ-M4) | `NOT ASSIGNED` | `NOT APPROVED` |
| `/08` | Power floors (§18) | STATISTICAL REVIEW | `REQUIRES DECISION` | Freeze minimum fraud count, minimum total evaluation count, minimum valid seeds/runs | numeric minimums · precision-based rule · most conservative pre-registered proposal · reject/defer | NR-05 §20 item 8 (ULB test 75 / cal 57; IBM test 134 / cal 185) | Blocks freeze; blocks primary Track M | `NOT ASSIGNED` | `NOT APPROVED` |
| `/09` | Dependence-aware method (§19) | STATISTICAL REVIEW | `REQUIRES DECISION` | Name the authoritative CI method, applicability rule and replicate count | §19 branch structure with frozen counts · single conservative method · all methods with one declared primary · reject/defer | NR-05 §16 (IID, entity-clustered, temporal-block; all METHOD NOT YET FROZEN; n=300) | Blocks freeze; blocks primary Track M | `NOT ASSIGNED` | `NOT APPROVED` |
| `/10` | Minimum seed / run rule | STATISTICAL REVIEW | `REQUIRES DECISION` | Freeze seed count, seed values, and the no-exclusion reporting rule | protocol's 5 fixed seeds · other pre-declared count · single seed labelled as such · reject/defer | NR-05 §16, §23 (single seed 42); preregistration §5.1 | Blocks freeze; blocks primary Track M | `NOT ASSIGNED` | `NOT APPROVED` |
| `/11` | Member operating points | STATISTICAL REVIEW | `REQUIRES DECISION` | Freeze the per-member operating-point reporting convention | shared ensemble threshold · per-member threshold on validation only · both, one primary · threshold-free only · reject/defer | NR-05 §20 item 11 | Secondary analysis only (RQ-M2) | `NOT ASSIGNED` | `NOT APPROVED` |
| `/12` | Alert-volume semantics | STATISTICAL REVIEW + DOMAIN REVIEW | `REQUIRES DECISION` | Freeze the fixed-alert-volume anchor (cal-anchored vs test-ranked) | cal-anchored · test-ranked labelled non-operational · both, one primary · reject/defer | NR-05 §15 (test-ranked top-k%; ULB per-day labelled artefact) | Blocks freeze (§35 item); does not block primary Track M | `NOT ASSIGNED` | `NOT APPROVED` |
| `/13` | Large-dataset sampling | STATISTICAL REVIEW | `REQUIRES DECISION` | Freeze the sampling rule for datasets too large to score in full | full population + streaming · frozen deterministic stride · frozen random subsample with seed · reject/defer | NR-05 (IBM 24,386,899 rows / 2.35 GB at stride-1/37, exploratory) | Blocks freeze; blocks primary Track M where it applies | `NOT ASSIGNED` | `NOT APPROVED` |
| `/14` | Segment / small-n withholding | STATISTICAL REVIEW (+ metric-definition owner if changed) | `REQUIRES DECISION` | Reconcile `metric_definitions.md` v1.0 small-n rules with the §18 floors | §18 governs primaries, v1.0 governs CI withholding · §18 supersedes, re-version metrics · set §18 equal to v1.0 · leave both unset | `metric_definitions.md` v1.0 (`<30` withhold, `<50` warn); NR-05 §14 (ULB q4 = 11 positives, segment q2 = 9 positives withheld) | Secondary analysis only (RQ-M9) | `NOT ASSIGNED` | `NOT APPROVED` |

**Counts:** 14/14 mapped · 11 block the freeze · 8 block primary Track M · 3 are
secondary-analysis-only · **0 resolved**.

---

## 3. Ownership classification

Ownership is by **reviewer type**, never by person. A role is not an assignment.

| ID | STATISTICAL REVIEW | DOMAIN REVIEW | BOTH | REPOSITORY-MECHANICAL VERIFICATION | Assigned individual |
|---|---|---|---|---|---|
| `/01` | ✔ | — | — | Freeze the baseline config as a repository artifact once chosen | `NOT ASSIGNED` |
| `/02` | ✔ | — | — | Freeze the ensemble implementation + hash once chosen | `NOT ASSIGNED` |
| `/03` | ✔ | ✔ (is the band operationally meaningful?) | ✔ | — | `NOT ASSIGNED` |
| `/04` | ✔ (bounds) | ✔ (cost-of-abstention posture behind coverage loss) | ✔ | — | `NOT ASSIGNED` |
| `/05` | ✔ (bound construction) | ✔ (which fields carry real semantics) | ✔ | Contract must be provably train-window-only | `NOT ASSIGNED` |
| `/06` | ✔ (rule) | ✔ (what shift magnitude is material) | ✔ | PSI thresholds sourced from `psi.py` constants | `NOT ASSIGNED` |
| `/07` | ✔ | ✔ (acceptance bound meaning) | ✔ | Sanity guard implemented + unit-tested | `NOT ASSIGNED` |
| `/08` | ✔ | — | — | Arithmetic re-derivation of cell counts per dataset | `NOT ASSIGNED` |
| `/09` | ✔ | — | — | Paired-resample index identity verifiable in code | `NOT ASSIGNED` |
| `/10` | ✔ | — | — | Seed list recorded in every evidence record | `NOT ASSIGNED` |
| `/11` | ✔ | — | — | Convention computable identically across arms | `NOT ASSIGNED` |
| `/12` | ✔ | ✔ (operational reading) | ✔ | Anchor reproducible from the frozen split | `NOT ASSIGNED` |
| `/13` | ✔ | — | — | Population definition + hash per plan §31 | `NOT ASSIGNED` |
| `/14` | ✔ | — | — | v1.0 rule implemented in `metric_definitions.py` | `NOT ASSIGNED` |

**No domain-owned decision may be closed by a statistical ruling alone**, and
`/04` in particular encodes a domain cost posture inside what looks like a
statistical bound. Plan §32: *"A single developer's self-approval is not
sufficient for the statistical sections."*
---

## 4. Decision sheets

Each sheet is fill-in. The six-line block below is the fill format (state values in
§8). Nothing in a fill block is pre-filled with anything other than an unresolved
state.

**Reviewer components.** Every one of the 14 decisions requires the **statistical**
component, recorded in the sheet's original reviewer block. The six decisions
classified `BOTH` in §3 (`/03` `/04` `/05` `/06` `/07` `/12`) additionally carry a
**domain** component block, whose fields are prefixed `Domain Reviewer …`. Each
sheet declares its own requirement on a `Required reviewer components:` line, and
that declaration must agree with §3 — a sheet cannot opt itself out of a role, and
a `STATISTICAL REVIEW` sheet may not widen its ownership by carrying a domain block.

> **Decisions classified as `BOTH` require independent statistical and domain review
> before final approval. One reviewer cannot satisfy both roles.**

```text
Reviewer Decision:
PENDING REVIEW

Reviewer:
NOT ASSIGNED

Decision Date:
NOT ESTABLISHED

Rationale:
PENDING REVIEW

Evidence Reviewed:
PENDING REVIEW

Approval:
NOT APPROVED

Approval Timestamp:
NOT ESTABLISHED
```

### `/01` — Baseline identity

**Decision.** Name the single-model primary baseline that the ensemble must beat
(plan §3, "Primary baseline").

**Why it matters.** RQ-M1 is the primary signal criterion. Plan §3 requires the
baseline and ensemble to share dataset, feature representation, split,
preprocessing, tuning budget and information-availability rules, and forbids
deliberate under-tuning. Without a frozen baseline there is no reference arm for
the plan §20 paired-difference rule.

**Existing evidence.** NR-05 §7 ran LogisticRegression and XGBoost under one fixed
a-priori config with no hyperparameter search, and reported both without selecting.
NR-05 §6: the strongest single model differs by dataset — LR on ULB by ROC-AUC
0.9739, XGB on Kaggle 0.9970.

**Negative evidence.** The ensemble lost to XGB on every run (§6) but **beat** LR
on Kaggle (+0.1285 [+0.1175, +0.1405]) and entity-disjoint (+0.1381 [+0.1236,
+0.1528]), while being inconclusive against LR on ULB (−0.0150 [−0.0408, +0.0041]).
Choosing LR as the baseline because it is the arm the ensemble beats would be
selection on an exposed dataset — not available.

**Allowed decision options** (from plan §3 and memo `/01`; no others added).
A fixed family a-priori · per-dataset selection on the validation window only ·
both reported with a pre-declared primary and secondary · reject/defer, making
RQ-M1 `NOT ESTABLISHED` for Track M.

**Recommended reviewer question.** *Which single-model baseline is scientifically
defensible as the frozen reference arm, fixed before any untouched confirmatory
dataset is opened?*

Required reviewer components: STATISTICAL REVIEW

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** and **primary Track M**
(RQ-M1).

---

### `/02` — Ensemble identity

**Decision.** Freeze the exact ensemble implementation for confirmatory Track M
(plan §3, "Selected ensemble").

**Why it matters.** RQ-M1 and RQ-M2 both depend on it; plan §14 forbids search over
IEEE-CIS or BAF after inspecting results; plan §35 lists "exact ensemble identity
frozen". This is the single most consequential open choice in the package.

**Existing evidence.** NR-05 §8 evaluated the repository's own fusion recipe
(LR + RF + XGB + IF, mean of member probabilities → 2-parameter Platt fit on the
calibration window — the construction used by `retrain_v3.py`) and selected nothing.

**Negative evidence.** Mean fusion **lost to the best single model on all four
discrimination runs**, with paired CIs excluding zero: ULB −0.0145 [−0.0308,
−0.0036]; Kaggle −0.0338 [−0.0407, −0.0267]; entity-disjoint −0.0276 [−0.0344,
−0.0210]; IBM −0.2663 [−0.3483, −0.1830]. NR-05 attributes this to weak RF/IF
members diluting LR/XGB. This is exploratory, small-n, exposed-dataset evidence —
it informs the choice but cannot select the arm.

**Allowed decision options.** Fixed ensemble as-is · strongest single-model family
as the primary arm with the ensemble demoted to ablation · pre-specified
architecture comparison (multiplicity-adjusted) · another named fusion family
defined independently of NR-05 · reject/defer (`NOT ESTABLISHED`).

**Recommended reviewer question.** *Which ensemble arm can be frozen a priori as a
genuine test of RQ-M1, given that the NR-05 candidate was inferior — and how is the
NR-05 result handled without post-hoc selection on confirmatory data?*

Required reviewer components: STATISTICAL REVIEW

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** and **primary Track M**.

---

### `/03` — Confident-decision definition

**Decision.** Freeze the "confident decision" band used by the wrong-confident
metric (plan §5.3).

**Why it matters.** §5.3 defines the numerator of the Class B primary gating
criterion (§5.5) and of the wrong-confident rate. A different band changes every
gating number. Plan §35 lists "confident-decision definition frozen".

**Existing evidence.** NR-05 §9 used ML `p ≥ 0.8 or p ≤ 0.2` and recorded it as an
NR-05 exploratory choice; NR-05 §11–§12 show the metric is band-sensitive in
practice.

**Negative evidence.** The ULB row gate's apparent Class B "benefit" (row WC 320–321
vs ungated 404–414) is essentially the clean-window difference, while clean recall
fell 0.813 → 0.040 — the number a wider band would make worse, a narrower band may
make look different. The band materially changes the Class B verdict.

**Allowed decision options.** Fixed probability band, named numerically, frozen ·
quantile-based band over the calibration window · two frozen bands reported
separately · reject/defer, making the wrong-confident criterion `NOT ESTABLISHED`
and re-specifying §5.5 on a different primary.

**Recommended reviewer question.** *What band defines a confident decision, on which
score (calibrated probability, raw member score, or fallback output), and is that
computable identically in the gated and ungated systems?*

Required reviewer components: BOTH (STATISTICAL REVIEW + DOMAIN REVIEW)

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Domain reviewer component** — required because this decision is classified `BOTH`
(§3). It cannot be satisfied by the statistical component above.

```text
Domain Reviewer Decision: PENDING REVIEW
Domain Reviewer: NOT ASSIGNED
Domain Reviewer Decision Date: NOT ESTABLISHED
Domain Reviewer Rationale: PENDING REVIEW
Domain Reviewer Evidence Reviewed: PENDING REVIEW
Domain Reviewer Approval: NOT APPROVED
Domain Reviewer Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** and the **Class B primary**
criterion only (RQ-M3 Class B); does not block discrimination RQs.

---

### `/04` — Gating bounds (§5.5)

**Decision.** Freeze the five §5.5 bounds: minimum reduction in wrong confident
decisions, minimum coverage, maximum coverage loss, maximum recall loss, and the
fallback/rules quality floor. All five are `[TO BE FROZEN]`.

**Why it matters.** §5.5 is the primary gating criterion. The plan states a gate
that rejects all or nearly all cases cannot pass because its coverage constraint
will fail. Plan §35 lists "coverage/recall bounds frozen" and "fallback error
accounting frozen".

**Existing evidence.** The **only** numeric proposal in the repository is
preregistration Criterion 3: ≥20% relative reduction in wrong-confident decisions at
≤2 percentage-point coverage loss, CI excluding 0 — annotated there as an assumption
requiring a domain assumption about the cost of confident errors vs abstentions.
It is unapproved.

**Negative evidence.** NR-05 §11–§12: the bounds are decisive, not cosmetic. ULB
row gating reduced wrong-confident 6/6 Class B conditions while clean recall fell
**0.813 → 0.040**. The window gate reached coverage **0** on ULB. On Kaggle the row
gate bought ~1,300 fewer wrong-confident decisions at σ1.0 for a **12–14 pp** recall
cost. Class B gating benefit is `NOT ESTABLISHED`.

**Allowed decision options.** Adopt the preregistration's ≥20% / ≤2pp proposal plus
frozen recall-loss and fallback-quality floors · different a-priori bounds justified
independently of NR-05 · replace the scalar bound with a coverage–risk curve
criterion · reject/defer, leaving gating as a safety mechanism with no effectiveness
claim. *Any specific bound adopted here is `CANDIDATE — REQUIRES REVIEW` until the
reviewer rules.*

**Recommended reviewer question.** *What minimum reduction, coverage floor, coverage-
loss ceiling, recall-loss ceiling and fallback-quality floor make a gating claim
scientifically defensible, and what domain cost posture justifies the coverage-loss
tolerance?*

Required reviewer components: BOTH (STATISTICAL REVIEW + DOMAIN REVIEW)

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Domain reviewer component** — required because this decision is classified `BOTH`
(§3). It cannot be satisfied by the statistical component above.

```text
Domain Reviewer Decision: PENDING REVIEW
Domain Reviewer: NOT ASSIGNED
Domain Reviewer Decision Date: NOT ESTABLISHED
Domain Reviewer Rationale: PENDING REVIEW
Domain Reviewer Evidence Reviewed: PENDING REVIEW
Domain Reviewer Approval: NOT APPROVED
Domain Reviewer Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** and the **Class B primary**
criterion only.

---

### `/05` — Feature-contract construction (§5.0)

**Decision.** Freeze how each dataset's Track-M feature contract (types, ranges,
missing policy) is constructed — statistical bounds, semantic bounds, or a hybrid
with per-feature-type rules and padding.

**Why it matters.** Plan §5.0 requires the contract, drift baseline, window size and
thresholds to be frozen **before** inspecting that dataset's confirmatory data,
using training-window data only, and forbids deriving any contract from validation
or test windows. The contract *is* the row gate, so it gates Class A and Class B
alike.

**Existing evidence.** NR-05 §9 used training-only bounds `q01/q99 ± 2·IQR`, which
failed three ways (NR-05 §20 item 5).

**Negative evidence.** (a) **Tail-dwelling fraud** — the ULB contract blocked
**72/75** test frauds, collapsing clean recall to 0.040. (b) **Categorical codes
with temporal mix-shift** — IBM `chip_code` train bounds [1.0, 2.0] while **70.6%**
of test rows carry code 0. (c) **Constructed-feature NaNs** — **33,685** IBM
negative-amount rows make `log1p` NaN, so they are treated as missing and blocked.
This is why IBM gating is `INCONCLUSIVE` rather than evidence about gating.

**Allowed decision options.** Semantic bounds where the dataset documents them,
statistical quantiles elsewhere · statistical quantiles with frozen per-type padding
and an explicit tail-overlap allowance · contract on a documented train-window
subsample with a frozen rule · reject/defer, leaving no Track-M gating experiment
specified.

**Recommended reviewer question.** *How must each feature type — categorical codes,
tail-heavy monetary features, engineered transforms — be bounded so the contract
does not block legitimate fraud, and what padding policy applies?*

Required reviewer components: BOTH (STATISTICAL REVIEW + DOMAIN REVIEW)

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Domain reviewer component** — required because this decision is classified `BOTH`
(§3). It cannot be satisfied by the statistical component above.

```text
Domain Reviewer Decision: PENDING REVIEW
Domain Reviewer: NOT ASSIGNED
Domain Reviewer Decision Date: NOT ESTABLISHED
Domain Reviewer Rationale: PENDING REVIEW
Domain Reviewer Evidence Reviewed: PENDING REVIEW
Domain Reviewer Approval: NOT APPROVED
Domain Reviewer Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** and **primary Track M**.

---

### `/06` — Window-gate definition

**Decision.** Freeze the window-level drift monitor's definition: baseline window,
window size, any-feature vs fraction-of-features rule, the block rule, and its
applicability where natural drift already exists.

**Why it matters.** Plan §5.0 names window-level drift monitoring as the component
**designed to detect** Class B shift, and requires it frozen per dataset before
confirmatory inspection.

**Existing evidence.** NR-05 §9 used `backend/src/drift_monitor/psi.py` unchanged —
train-window bins, block when **any** feature PSI ≥ 0.25 (the module's documented
ALERT constant).

**Negative evidence.** The monitor was **unusable as configured on both endpoints**.
On ULB it fired on all windows because real temporal drift gives clean PSI **1.46**
(7/30 features ≥ 0.25), producing coverage **0** — it "won" on wrong-confident only
by rejecting 100% of rows. On Kaggle it **never fired** on any Class B condition
(max PSI **0.083**, below the 0.10 warn level) → `NOT DETECTED`. On IBM coverage was
0 in 7/7 due to the `/05` contract defect.

**Allowed decision options.** PSI-magnitude-calibrated gate with threshold set from
the frozen training-window drift distribution · fraction-of-features rule plus a
minimum-drift guard · narrow the claim to `NOT DESIGNED TO DETECT` for Class B at
frozen magnitudes · reject/defer, dropping the window component. *Choosing among
these on the basis of NR-05 outcomes would be post-hoc selection on exposed data and
is not permitted.*

**Recommended reviewer question.** *Is a window gate part of the confirmatory
specification at all, and if so what minimum detectable shift magnitude must it
target?*

Required reviewer components: BOTH (STATISTICAL REVIEW + DOMAIN REVIEW)

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Domain reviewer component** — required because this decision is classified `BOTH`
(§3). It cannot be satisfied by the statistical component above.

```text
Domain Reviewer Decision: PENDING REVIEW
Domain Reviewer: NOT ASSIGNED
Domain Reviewer Decision Date: NOT ESTABLISHED
Domain Reviewer Rationale: PENDING REVIEW
Domain Reviewer Evidence Reviewed: PENDING REVIEW
Domain Reviewer Approval: NOT APPROVED
Domain Reviewer Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** and the **Class B primary**
criterion only.

---

### `/07` — Calibration procedure (§6)

**Decision.** Freeze the calibration method, the primary calibration metric, the
acceptance bound, a calibrator-sanity guard, and a calibration-positive floor.

**Why it matters.** Plan §6 fixes the training → calibration/validation → test
sequence, requires the test window not to influence fitting, and leaves all three
calibration cells `[TO BE FROZEN]`. Calibration sits on the detection critical path:
the IBM run produced recall 0.000 from an inverted calibrator.

**Existing evidence.** NR-05 §13 — Platt fitted on the calibration window only,
calibration-positive counts ULB **57**, Kaggle 2,385, entity 1,867, IBM **185**.

**Negative evidence.** The IBM fit learned **coef −0.7029, intercept −6.4186**,
flipping the ranking (cal raw AUC 0.652 → 0.348; test raw 0.627 → **0.373, below
chance**) while **Brier 0.00102 and ECE 0.00045 still looked excellent**. NR-05
reproduced the inversion bit-identically on an independent refit and did not
resolve the cause. Brier/ECE can look excellent while ranking is broken.

**Allowed decision options.** Platt with a frozen sanity guard **and** a
calibration-positive floor, both mandatory · another monotone calibrator with the
same guard · report calibration-free discrimination as primary and treat calibrated
probabilities as secondary until a guard exists · reject/defer (`NOT ESTABLISHED`).
*Any guard threshold or floor adopted here is `CANDIDATE — REQUIRES REVIEW`.*

**Recommended reviewer question.** *Must a sign/monotonicity sanity guard and a
minimum calibration-positive count be mandatory, and may Brier/ECE ever be reported
as evidence of calibration without the ranking check?*

Required reviewer components: BOTH (STATISTICAL REVIEW + DOMAIN REVIEW)

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Domain reviewer component** — required because this decision is classified `BOTH`
(§3). It cannot be satisfied by the statistical component above.

```text
Domain Reviewer Decision: PENDING REVIEW
Domain Reviewer: NOT ASSIGNED
Domain Reviewer Decision Date: NOT ESTABLISHED
Domain Reviewer Rationale: PENDING REVIEW
Domain Reviewer Evidence Reviewed: PENDING REVIEW
Domain Reviewer Approval: NOT APPROVED
Domain Reviewer Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** and **primary Track M**
(RQ-M4).

---

### `/08` — Power floors (§18)

**Decision.** Freeze the three §18 minimums — minimum fraud count, minimum total
evaluation count, minimum valid seeds/runs — and resolve the `min_cell`
discrepancy with `metric_definitions.md` v1.0.

**Why it matters.** §18 is explicit: if the requirements are not satisfied the
result is `INCONCLUSIVE / UNDERPOWERED` and "does not count as GO or NO-GO for the
affected criterion". A floor chosen after seeing operating-point estimates is the
post-hoc selection plan §36 forbids.

**Existing evidence.** `docs/metric_definitions.md` v1.0 already mandates CI
withholding below **30** observations in the limiting class and
`SMALL_POSITIVE_CLASS` / `SMALL_NEGATIVE_CLASS` warnings below **50**. Plan §18
proposes a different, unset, minimum fraud count. Plan §35 lists "minimum
fraud-count rule frozen".

**Negative evidence.** NR-05 §20 item 8: **ULB test 75 positives / cal 57; IBM test
134 / cal 185**. Every operating-point and calibration conclusion on those two
datasets is small-n-exposed. NR-05 §14: ULB quartile 4 = 11 positives (AUC 0.670);
ULB amount-quartile segment q2 = **9 positives** (AUC 0.441, withheld). NR-05
states the floors "will likely mark several of these INCONCLUSIVE".

**Allowed decision options.** Numeric minimums justified from the estimand's
precision requirement · a precision-based rule (minimum CI half-width) instead of
raw counts · the most conservative of several pre-registered proposals, accepting
more `INCONCLUSIVE` outcomes · reject/defer, leaving Track M `BLOCKED`.
*No number is proposed here; choosing one is the reviewer's act.*

**Recommended reviewer question.** *What minimum fraud count, minimum total
evaluation count and minimum valid seeds/runs must a confirmatory window satisfy,
and does that floor supersede, complement, or sit alongside the v1.0 `min_cell`
rule?*

Required reviewer components: STATISTICAL REVIEW

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** and **primary Track M**.

---

### `/09` — Dependence-aware uncertainty method (§19)

**Decision.** Declare the authoritative uncertainty method for confirmatory Track
M, its applicability rule, replicate count and seed.

**Why it matters.** §19 states "Ordinary IID bootstrap is therefore not the default
confirmatory method" and requires the applicability rule frozen **before results are
examined**, applied consistently for paired comparisons. Plan §35 lists
"dependence-aware CI method frozen".

**Existing evidence.** NR-05 §16 exercised **IID stratified percentile bootstrap**
(n=300, seed 42), **entity-clustered** (`cc_num` / `User` resampled with
replacement; Kaggle, IBM) and **temporal block** (20 contiguous blocks; ULB), and
labelled every dependence-aware interval **"EXPLORATORY — METHOD NOT YET FROZEN"**,
with IID explicitly not the default interpretation. Preregistration §2 separately
proposes 2,000 replicates behind a `[REQUIRES DECISION/APPROVAL]` seed marker.

**Negative evidence.** These methods are not interchangeable and no NR-05 interval
is authoritative. NR-05 §6's ULB paired interval used the block method while the
Kaggle and entity intervals used entity-clustering — the width and therefore the
"CI excludes 0" verdict are method-dependent, and the plan §20 above-chance rule is
applied to whichever method is frozen.

**Allowed decision options.** §19's own branch structure — entity-clustered where
meaningful entity identifiers exist, temporal/block where serial dependence is
material, independent-observation otherwise — with counts frozen · a single
conservative method for all datasets · report all applicable methods with one
declared primary · reject/defer (`BLOCKED`).

**Recommended reviewer question.** *Which method is authoritative on each dataset,
with what replicate count, and how is pairing preserved across both arms of every
model comparison?*

Required reviewer components: STATISTICAL REVIEW

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** and **primary Track M**.

---

### `/10` — Minimum seed / run rule

**Decision.** Freeze the minimum number of valid seeds/runs per experiment, the seed
values, and the no-exclusion reporting rule.

**Why it matters.** Single-seed results are not estimable evidence. Plan §18's
"minimum valid seeds/runs" cell is `[TO BE FROZEN]`; preregistration §5.1 proposes
5 fixed seeds behind a `[REQUIRES DECISION/APPROVAL]` marker, and §9.9 prohibits
post-hoc seed exclusion.

**Existing evidence.** NR-05 ran **single seed 42** for every experiment and listed
"single seed" first in its §23 limitations.

**Negative evidence.** Every NR-05 interval therefore rests on one realisation of the
training procedure. This is recorded in NR-05 §16 and §23 and is not repaired by
any number of bootstrap replicates — resampling the test set does not resample
training stochasticity.

**Allowed decision options.** Adopt the preregistration's 5 fixed seeds · a
different pre-declared count with a stated precision rationale · single seed with
results labelled as such, defensible only where the window will be `INCONCLUSIVE`
anyway · reject/defer.

**Recommended reviewer question.** *How many seeds must every pre-declared seed
appear for, and what rule forbids excluding a seed after its outcome is seen?*

Required reviewer components: STATISTICAL REVIEW

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** and **primary Track M**.

---

### `/11` — Member operating points

**Decision.** Freeze how per-member operating points are computed and reported.

**Why it matters.** RQ-M2 requires the ensemble to be compared with its relevant
constituent models. If members are scored at a threshold selected for the ensemble,
member recall is not comparable to the ensemble's and the ablation table invites a
wrong reading.

**Existing evidence.** NR-05 §20 item 11 — per-member `recall_at_1pct_fpr` used the
**shared ensemble-selected threshold**, recorded as an exploratory choice, not
defended.

**Negative evidence.** NR-05 §6's member table is the direct consequence: members
reported at the ensemble's operating point cannot be read as each member's own best.
Any ablation conclusion drawn from that table without acknowledging the convention
is unsupported.

**Allowed decision options.** Shared ensemble-selected threshold for every member,
with the caveat that it is not each member's own optimum · per-member threshold
chosen on the **validation window only** · both, with one declared primary ·
threshold-free metrics for members only · reject/defer.

**Recommended reviewer question.** *Which table is the reportable member
ablation — shared ensemble threshold, or per-member validation-selected threshold?*

Required reviewer components: STATISTICAL REVIEW

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** **Neither** Research Plan freeze nor primary Track M. Affects
**secondary Track M analysis** (RQ-M2 ablation) only; must be frozen before any
confirmatory ablation is reported.

---

### `/12` — Alert-volume semantics

**Decision.** Freeze whether the fixed-alert-volume operating point is
**cal-anchored** (fixed top-k or alert rate chosen on the calibration window,
applied unchanged to test) or **test-ranked** (top-k% recomputed on the test score
distribution).

**Why it matters.** Plan §10 (RQ-M8) and §35 "business metrics frozen". A
cal-anchored volume measures operational stability over time; a test-ranked volume
measures the best case attainable on that window. Only the first supports an
operational reading.

**Existing evidence.** NR-05 §15 used **test-ranked top-k%**, computed on the test
score distribution with labels unused, and reported no cost model at all.

**Negative evidence.** NR-05 §15 also recorded that ULB's per-day rates are dataset
artefacts (the `Time` axis spans ≈48 hours total) and that IBM alert-volume recall
≈ 0 is the **inverted-calibrator effect**, not an operational statement. Reading
either as an operational forecast would be a misuse of the exploratory result.

**Allowed decision options.** Cal-anchored alert volume and rate · test-ranked,
explicitly labelled non-operational · both, one declared primary · reject/defer
(RQ-M8 `NOT APPLICABLE` / `NOT ESTABLISHED` for Track M).

**Recommended reviewer question.** *Which fixed-alert-volume anchor can be frozen
as the primary business metric, and which analyst-burden figures may be reported as
clearly labelled proxies only?*

Required reviewer components: BOTH (STATISTICAL REVIEW + DOMAIN REVIEW)

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Domain reviewer component** — required because this decision is classified `BOTH`
(§3). It cannot be satisfied by the statistical component above.

```text
Domain Reviewer Decision: PENDING REVIEW
Domain Reviewer: NOT ASSIGNED
Domain Reviewer Decision Date: NOT ESTABLISHED
Domain Reviewer Rationale: PENDING REVIEW
Domain Reviewer Evidence Reviewed: PENDING REVIEW
Domain Reviewer Approval: NOT APPROVED
Domain Reviewer Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze** (§35 checklist item).
Does **not** block primary Track M — RQ-M8 is a supporting RQ (plan §23 primary
candidates are RQ-M1, RQ-M3 Class B, RQ-M4).

---

### `/13` — Large-dataset sampling

**Decision.** Freeze the sampling rule (if any) for datasets too large to load and
score in full, and confirm it is fixed before any inspection that could influence
method choices.

**Why it matters.** Plan §14 forbids choices informed by untouched confirmatory data;
plan §31 requires dataset identity, population and split provenance for every
quantitative result. A subsampled population is a different population.

**Existing evidence.** NR-05 loaded IBM v2 (**24,386,899 data rows, 2.35 GB**) with
a **stride-1/37 subsample**, recorded as an exploratory choice. IBM is exposed, so
this carried no confirmatory risk there; it becomes live for any large candidate.

**Negative evidence.** Because the IBM stride was chosen during exploration, any
IBM-derived number is population-limited in a way the manifest does not yet record
as a distinct population. That does not invalidate NR-05; it does mean IBM results
must not be read as full-population results.

**Allowed decision options.** Full population wherever the environment allows, with
streaming score computation · frozen deterministic stride with the stride factor
recorded in the evidence record · frozen random subsample with seed and rate
recorded · reject/defer, making any dataset beyond the compute budget `BLOCKED`.

**Recommended reviewer question.** *What sampling rule applies to datasets too large
to score in full, and is it fixed before acquisition or first inspection of any
confirmatory dataset?*

Required reviewer components: STATISTICAL REVIEW

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** Blocks **Research Plan freeze**; blocks **primary Track M**
for any dataset where it applies, otherwise `NOT APPLICABLE`.

---

### `/14` — Segment / small-n withholding

**Decision.** Reconcile the repository's existing small-sample rule with the §18
power floors — which rule governs a cell with fewer than N positives, and whether
such a cell is withheld, `INCONCLUSIVE`, or `NOT APPLICABLE`.

**Why it matters.** `docs/metric_definitions.md` v1.0 (the authoritative metric
implementation) already mandates CI withholding below 30 observations in the
limiting class and small-class warnings below 50. Plan §18 proposes a different,
unset, minimum fraud count. Two documents can require different things from the
same cell.

**Existing evidence.** `METRIC_DEFINITIONS_VERSION = "1.0"` in
`backend/scripts/metric_definitions.py`; `docs/metric_definitions.md` §"Confidence
intervals" and §"Minimum sample conditions".

**Negative evidence.** NR-05 §14 withheld a ULB segment cell with **9 positives**
(AUC 0.441) rather than report it as performance. Under the v1.0 rule that CI is
withheld for two independent reasons; under an unset §18 floor its status is
undefined. The same ambiguity would apply to ULB quartile 4 (**11 positives**,
AUC 0.670).

**Allowed decision options.** §18 governs **primary criteria** while v1.0 continues
to govern CI withholding and warning flags · §18 supersedes entirely, with
`metric_definitions.md` re-versioned · v1.0 as the sole rule, with §18's minimum
fraud count set equal to it · leave both unset and treat every sub-floor cell as
`INCONCLUSIVE`.

**Recommended reviewer question.** *Which rule takes precedence for a sub-threshold
cell, and does the choice require a version bump of `metric_definitions.md`?*

Required reviewer components: STATISTICAL REVIEW

```text
Reviewer Decision: PENDING REVIEW
Reviewer: NOT ASSIGNED
Decision Date: NOT ESTABLISHED
Rationale: PENDING REVIEW
Evidence Reviewed: PENDING REVIEW
Approval: NOT APPROVED
Approval Timestamp: NOT ESTABLISHED
```

**Freeze consequence.** **Neither** Research Plan freeze nor primary Track M.
Affects **secondary Track M analysis** (RQ-M9 segment and time-window reporting).

---

## 5. Negative Evidence That Must Survive the Freeze

These are recorded outcomes, not open questions. They must appear unchanged in any
future protocol, decision record or report. **None may be dropped because it
complicates the preferred narrative.** Every one is first-class under plan §25 and
the preregistration's negative-result plan (§4).

| # | Finding | Source | Status |
|---|---|---|---|
| N1 | Mean-fusion ensemble **underperformed XGB across all four** NR-05 evaluations (paired CIs exclude 0) | NR-05 §6, §8 | `EXPLORATORY` — negative result, first-class |
| N2 | IBM Platt calibration produced the **reproducible inversion** (coef −0.7029, intercept −6.4186, 185 cal positives; reproduced bit-identically on independent refit) | NR-05 §13 | `EXPLORATORY` — cause unresolved, not explained away |
| N3 | IBM test AUC fell **below chance** (0.627 → **0.373**) **despite apparently strong Brier/ECE** (0.00102 / 0.00045) | NR-05 §13 | `EXPLORATORY` — honest-reporting caveat for §4A.3 |
| N4 | ULB clean recall **collapsed** under the tested row gate: **0.813 → 0.040**, with 72/75 test frauds blocked | NR-05 §11, §12 | `EXPLORATORY` — contract defect, not a gating verdict |
| N5 | Window PSI **over-fired on ULB** (clean PSI 1.46, coverage 0) because real temporal drift triggers the fixed 0.25 threshold | NR-05 §11 | `EXPLORATORY` — component unusable as configured |
| N6 | Window PSI **did not detect** the tested Kaggle Class-B condition (max PSI **0.083**, below the 0.10 warn level) | NR-05 §11 | `NOT DETECTED` |
| N7 | **Class-B gating benefit remained `NOT ESTABLISHED`** | NR-05 §11–§12, §23 | `NOT ESTABLISHED` |
| N8 | The **Kaggle entity-disjoint result was materially stronger** than the IBM temporal result (0.9697 held-out ROC-AUC on a higher-prevalence population vs IBM falling to 0.58–0.64) | NR-05 §14 | `EXPLORATORY` — not a like-for-like comparison |
| N9 | **IBM temporal performance degraded substantially** (train-period AUC 0.83–1.00 → final-20% window 0.58–0.64 over a 1,020-day span) | NR-05 §14 | `EXPLORATORY` — negative result, first-class |
| N10 | IBM gating is **`INCONCLUSIVE`** because of feature/value-bound defects (`chip_code` train bounds [1.0, 2.0] vs 70.6% of test rows at code 0; 33,685 negative-amount rows making `log1p` NaN) | NR-05 §9, §11, §20 | `INCONCLUSIVE` |
| N11 | ULB window-gate "wins" were an artefact of rejecting **100% of rows** (coverage 0), which the plan's §9/§5.5 forbids counting as success | NR-05 §11 | `EXPLORATORY` |
| N12 | Two harness defects were found and fixed **before** any result was reported (Class-A injection cross-contamination; IsolationForest score sign), and the superseded records are preserved append-only with their metrics **not cited** | NR-05 §19 | Documented, not deleted |
| N13 | ULB quartile 4 (11 positives) and segment q2 (9 positives) are **small-n and withheld**, not performance | NR-05 §14 | `INCONCLUSIVE` |

**Consequence for the freeze.** If any of N1–N13 is later found to be softened,
removed, or re-labelled favourable, that is a protocol violation under
preregistration §9.5 and a plan §25 failure. The reviewer may not resolve N1–N13 in
favour of a positive claim; they may only decide what the confirmatory protocol
*asks* going forward.

---

## 6. Placeholder Resolution Map

**Exact current freeze-check composition** (`scripts/check_freeze.py`, exit 1,
78 findings) — verified, not estimated:

| Source | Findings | Detail |
|---|---:|---|
| `missing docs/FREEZE_RECORD.json` | 1 | plan §29/§30.2 — expected pre-freeze |
| Unresolved placeholder, `docs/RESEARCH_PLAN.md` | **65** | across **60 unique lines**; plan §28 rows (lines 1008–1012) each carry **two** markers (`[OWNER]`/`[DATE]`, `[STATISTICAL REVIEWER]`/`[DATE]`, `[DOMAIN REVIEWER]`/`[DATE]`) |
| Unresolved placeholder, preregistration | **12** | across 12 unique lines (11 marker occurrences; line 211's marker is wrapped across a line break, which is why an occurrence count returns 11) |
| **Total** | **78** | unchanged from before this task |

**No placeholder was resolved by this task.** The map records which could
legitimately be resolved, from what source, and whether a reviewer is still
required.

Resolution types used below: **existing repository decision** · **explicitly
approved reviewer decision** · **purely mechanical repository fact**.

| Placeholder (plan §, representative lines) | Count | Source | Resolution Type | Requires Reviewer? | Current State |
|---|---:|---|---|---|---|
| §3 primary metric (L221 + §4A L231–238) | 9 | — | — | **Yes** (statistical) | `REQUIRES DECISION` — open. Metric choice is the reviewer's; no existing artifact selects one |
| §3 selected ensemble (L140) | 1 | — | — | **Yes** (statistical) | `REQUIRES DECISION` (`/02`) — open |
| §3 primary baseline (L144) | 1 | — | — | **Yes** (statistical) | `REQUIRES DECISION` (`/01`) — open |
| §3 two-dataset / one-dataset aggregation (L172, L176) | 2 | plan §23 | — | **Yes** (statistical) | `REQUIRES DECISION` — open |
| §3 multiplicity method (L186) | 1 | — | — | **Yes** (statistical) | `REQUIRES DECISION` — open |
| §5.0 frozen gate-component definitions (L312) | 1 | — | — | **Yes** (statistical + domain) | `REQUIRES DECISION` (`/05`, `/06`) — open |
| §5.2 Class A/B perturbation magnitudes (L345, L361, L365) | 3 | — | — | **Yes** (statistical) | `REQUIRES DECISION` — open |
| §5.3 confident-decision definition (L383) | 1 | — | — | **Yes** (statistical + domain) | `REQUIRES DECISION` (`/03`) — open |
| §5.5 gating bounds (L409, L415, L419, L423, L427) | 5 | preregistration Criterion 3 (≥20% / ≤2pp) | existing repository decision — **but unapproved** | **Yes** (statistical + domain) | `NOT ESTABLISHED` + `REQUIRES DECISION` (`/04`). The proposal is `CANDIDATE — REQUIRES REVIEW`; it is explicitly annotated in the protocol as an assumption |
| §6 calibration method / metric / bound (L447, L449, L451) | 3 | — | — | **Yes** (statistical + domain) | `REQUIRES DECISION` (`/07`) — open |
| §10 primary business metric + proxy (L506, L508) | 2 | — | — | **Yes** (statistical + domain) | `REQUIRES DECISION` (`/12`) — open |
| §11 segment definitions (L525) | 1 | — | — | **Yes** (statistical) | `REQUIRES DECISION` — open |
| §16 Track P new-model rules (L623) | 1 | — | — | **Yes** | `[N — TO BE FROZEN]` — open; outside the 14 decisions |
| §18 power floors (L677, L679, L681) | 3 | `metric_definitions.md` v1.0 (`<30`/`<50`) | mechanical fact for the **v1.0** rule only | **Yes** (statistical) | `REQUIRES DECISION` (`/08`, `/10`) — v1.0 values exist but §18 requires its own floors |
| §19 dependence-aware methods (L701, L709, L713, L719) | 4 | `psi.py` constants, `nr05_diagnostics.py` resamplers | mechanical fact that implementations exist | **Yes** (statistical) | `REQUIRES DECISION` (`/09`) — NR-05 used all three; none is frozen |
| §21 primary success criteria (L755, L759, L771, L781, L787) | 5 | — | — | **Yes** (statistical) | `REQUIRES DECISION` — open |
| §23 dataset-level + roll-up + two-dataset matrix (L831, L835, L851, L855, L871, L872, L875) | 7 | plan §23 itself marks several "proposed" | — | **Yes** (statistical) | `REQUIRES DECISION` — open; includes the primary-RQ candidate list |
| §24 REVISE triggers + INCONCLUSIVE next step (L913, L922) | 2 | — | — | **Yes** (statistical) | `REQUIRES DECISION` — open |
| §28 owners/dates/reviewers (L1008–L1012) | 10 findings / 5 lines | — | — | **Yes** — requires real people | `PENDING` in plan §28; **all five owners unassigned**. No identity may be invented |
| §34 preregistration path (L1224) | 1 | `scripts/check_freeze.py` `PREREG = "docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md"`; the file exists at that path | **purely mechanical repository fact** | Confirmation only | `CANDIDATE — REQUIRES REVIEW`. Mechanically determinable, but the plan cell is a freeze-time decision; **not written by this task** |
| §34 exposure-ledger path (L1225) | 1 | `docs/evaluation/DATASET_EXPOSURE_LEDGER.md` exists | mechanical fact of existence | Confirmation only | The ledger **itself** states: "This file is the *proposed* repository location; the path decision itself remains an unresolved freeze-time decision and has **not** been written into the plan." Not resolved |
| §34 eligibility-manifest path (L1226) | 1 | `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` exists | mechanical fact of existence | Confirmation only | Same: the manifest explicitly defers the path decision. Not resolved |
| Preregistration §0 candidate S1/S2 (L20) | 1 | — | — | **Yes** (domain + statistical) | `[REQUIRES DECISION/APPROVAL]` — open; the protocol states "The option choice is a human decision; this protocol does not choose it" |
| Preregistration §1 split axis per dataset (L48) | 1 | — | — | **Yes** (statistical) | open — bears on §6.6/§6.7 |
| Preregistration §2 ECE bins / replicates+seed / multiplicity (L61, L72, L73) | 3 | `calibration_test.py` uses 10 equal-width bins | mechanical fact of current code | **Yes** (statistical) | open; bins are a documented implementation note, not an approval |
| Preregistration §3 MMD set (L83) | 1 | protocol §3 proposals | existing repository decision — **unapproved** | **Yes** (statistical + domain) | "Numerical MMDs below are proposals with justification, not settled facts" |
| Preregistration §4 conclusion wording A/B/C (L175) | 1 | — | — | **Yes** (domain) | open |
| Preregistration §5 seeds / fixed FPR / S1 threshold carry-over (L182, L185, L190) | 3 | protocol proposes 5 seeds; S1 threshold 0.018758 is locked by phase 93 | mixed: locked-threshold carry-over is an existing fact; the rest are proposals | **Yes** | open; **no threshold was changed by this task** |
| Preregistration §7 analyst capacity (L211) | 1 | — | — | **Yes** (domain) | open; protocol §7 already forbids describing it as measured burden |

**Net:** every one of the 78 findings requires either a reviewer decision, a real
person's assignment, or a freeze-time act. **None is resolvable from an existing
authoritative decision alone.** The three §34 path cells are the only candidates
for a purely mechanical resolution, and both companion artifacts explicitly decline
to resolve them, so they stay open.

---

## 7. Dataset Decisions Still Requiring Evidence

Source of record: `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` (`1.0-draft`,
**NOT frozen, NOT approved; no dataset is declared eligible by that document**) and
`docs/evaluation/DATASET_EXPOSURE_LEDGER.md` (`1.0-draft`, independent audit
`NOT ESTABLISHED`). **No eligibility is granted or implied here.**

### 7.1 Established — facts directly evidenced in the repository

| Fact | Evidence |
|---|---|
| Five dataset files exist with stable SHA-256, all re-verified unchanged after NR-05 | manifest §1–§4; NR-05 §3 |
| ULB has **no entity identifier** column | manifest §1 (`ESTABLISHED`) — makes entity-disjoint `NOT APPLICABLE`, not a gap |
| Kaggle `fraudTrain` has `cc_num` → entity-disjoint evaluation *possible* | manifest §2 (`ESTABLISHED`) |
| IBM v2 has `cc_num` | manifest §4 (`ESTABLISHED`) |
| IBM `hour_diff` is derived during feature extraction and is **not point-in-time safe as evaluated** (NR-01 §A) | manifest §4 (`ESTABLISHED` — a recorded defect); mitigation `NOT ESTABLISHED` |
| PS-14 synthetic data is `SIMULATED/SYNTHETIC` and cannot establish real-world performance | manifest §7; plan §13 |
| ULB, Kaggle and IBM are all **EXPLORATORY** under plan §14; their new-model-use rules are `[TO BE FROZEN]` | manifest §1–§4 eligibility rows |
| The exposure ledger exists and records prior exposure; its **independent audit is `NOT ESTABLISHED`** | exposure ledger header |

### 7.2 Pending review

| Item | Current state | Reviewer |
|---|---|---|
| ULB licence/access | `PENDING REVIEW` — **conflicting repo references, not merged**: plan §35 "licence/access verified" unchecked; `docs/metric_definitions.md` credits Kaggle CC BY-SA 4.0; `backend/scripts/dataset_card.py` writes an "AGPL-3.0" string; NR-02 §G says "Open licence" | Domain / data-governance |
| Kaggle provenance + licence | `PENDING REVIEW` — no Kaggle licence citation located for `fraudTrain.csv` | Domain / data-governance |
| IBM provenance + licence | `PENDING REVIEW` — "Open licence" claimed at protocol era; exact licence string unresolved | Domain / data-governance |
| Kaggle exposure interpretation | `EXPLORATORY`; whether `fraudTrain` may serve as a confirmatory *test* under frozen new-model rules | Statistical |
| `fraudTest.csv` confirmatory-vs-exploratory interpretation | `EXPLORATORY USE ONLY pending reviewer resolution`; not confirmatory-eligible | Statistical |
| Leakage audits (ULB, Kaggle) | `NOT ESTABLISHED` — no repo-archived audit | Statistical |
| IBM `hour_diff` point-in-time mitigation | `NOT ESTABLISHED` | Statistical + domain |
| Field-level metadata / feature semantics for unverified rows | `NOT ESTABLISHED` for several fields | Domain |
| Preprocessing requirements (all datasets) | `[TO BE FROZEN]` / `NOT ESTABLISHED` — plan §3/§14 new-model rules | Statistical |
| Independent audit of the exposure ledger | `NOT ESTABLISHED` (plan §14 requires it before the confirmatory freeze) | Statistical |

### 7.3 Blocked — external acquisition or external confirmation required

| Item | State | Blocker |
|---|---|---|
| IEEE-CIS terms verification | `BLOCKED` | plan §35 "IEEE-CIS terms checked" **unchecked**; manual verification required, not performed |
| IEEE-CIS acquisition | `BLOCKED — not acquired** | `data/external/` absent (verified) |
| IEEE-CIS labelled-portion availability, temporal suitability, fraud-count sufficiency | `NOT ESTABLISHED` | plan §28 IEEE-CIS checks cannot run before acquisition |
| BAF licence/terms, provenance, generation process, real/synthetic classification | `BLOCKED` | plan §35 unchecked; manifest §6 |
| BAF acquisition | `BLOCKED — not acquired** | `data/external_benchmark/` absent (verified); preregistered E1 precondition "dataset is acquired and validated" |
| Institutional (Track I) data | `BLOCKED / NOT ESTABLISHED` | plan §33: no validation until authorized eligible institutional data exist |
| Any external credentials or third-party access | `BLOCKED` | not held by this task |

**Consequence.** With both candidate confirmatory datasets `BLOCKED` and all three
exposed datasets `EXPLORATORY`, plan §23's "Zero eligible confirmatory datasets"
applies: Track M is **`BLOCKED`** regardless of how the 14 decisions are ruled.
Ruling on `/01`–`/14` is necessary for a freeze but **not sufficient** for a
confirmatory experiment.

---

## 8. Reviewer input format and completion states

### 8.1 Fill format

Each decision sheet carries this block verbatim. Fill it in place; do not delete
the block and do not replace an unresolved state with a blank.

```text
Reviewer Decision:
PENDING REVIEW

Reviewer:
NOT ASSIGNED

Decision Date:
NOT ESTABLISHED

Rationale:
PENDING REVIEW

Evidence Reviewed:
PENDING REVIEW

Approval:
NOT APPROVED

Approval Timestamp:
NOT ESTABLISHED
```

### 8.2 Permitted states

All are already compatible with the project's decision vocabulary (plan §22, §24;
memo §4). **Unresolved questions must not be collapsed into `APPROVED`.**

| State | Meaning here |
|---|---|
| `PENDING REVIEW` | no ruling yet (the current state of all 14) |
| `APPROVED` | ruled and accepted — requires a named reviewer, a decision date, a rationale, and `Approval: APPROVED` with a timestamp |
| `REJECTED` | ruled against; the criterion becomes `NOT ESTABLISHED` for Track M |
| `REVISE` | see §9; requires its own approval before it becomes authoritative |
| `NOT ESTABLISHED` | evidence required for the decision has not been demonstrated |
| `BLOCKED` | required data/access/dependency unavailable |
| `INCONCLUSIVE` | evidence insufficient to establish success or failure, including underpowered evaluation |

**Consistency rule (mechanically enforced by §11's checker):** `Reviewer Decision:
APPROVED` may not coexist with `Reviewer: NOT ASSIGNED`, or with
`Approval: NOT APPROVED`, or with `Approval Timestamp: NOT ESTABLISHED`.

**Two-sided rule for `BOTH` decisions.** A `BOTH` decision cannot be recorded as
`APPROVED` unless **both** components are complete: statistical reviewer named,
statistical decision resolved, statistical rationale present, statistical approval
recorded, **and** the same four for the domain reviewer. Assignment is insufficient;
a reviewer name alone is insufficient; a single approval — statistical *or* domain —
is insufficient. The two components stand or fall together: approving one while the
other is pending is rejected, exactly as is approving both without updating the
sheet's own final decision.

### 8.3 What a ruling does *not* do

A ruling on `/01`–`/14` does not freeze the plan, does not create
`docs/FREEZE_RECORD.json`, does not create a tag, and does not resolve a plan
placeholder. After a ruling is **approved**, the corresponding plan or preregistration
cell is edited as a separate, recorded act (plan §17: document, never silently
edit).

---

## 9. `REVISE` handling

Plan §24 permits `REVISE` **only** where a frozen revision condition explicitly
allows it, and permits **at most one revision cycle**. A `REVISE` ruling here is a
*proposal*; it is not applied automatically.

If a reviewer selects `REVISE`, the following seven fields are required before the
proposal can be considered:

```text
REVISE — decision being revised:
Reviewer:            NOT ASSIGNED
Reason:
Exact proposed change:
Evidence motivating the change:
Affected protocol section:
Affected metrics / criteria:
Reviewer approval of the revision: NOT APPROVED
```

| # | Required field | Rule |
|---|---|---|
| 1 | decision being revised | one of `/01`–`/14`, named singly |
| 2 | reason | a **documented defect** in the experiment or pipeline, established independently of whether the result was favourable |
| 3 | exact proposed change | one named artefact, cell or procedure — not "the approach" |
| 4 | evidence motivating the change | must exist in the repository before the change is proposed |
| 5 | affected protocol section | which preregistration section changes |
| 6 | affected metrics / criteria | which RQ or success criterion changes; if the primary endpoint changes, the revision is exploratory (plan §24.6) |
| 7 | reviewer approval | someone **other than the experimenter** (plan §24.1) |

**Not eligible triggers (plan §24.1), restated so they cannot be used:** a null or
negative primary result; a confidence interval containing zero; a failed threshold;
an unfavourable ablation; a wish to try another model or feature set. In
particular **none of N1–N13 in §5 is an eligible `REVISE` trigger** — they are
results, not defects.

**A revision does not modify the protocol automatically.** It becomes authoritative
only after its own review, and the original result is preserved (plan §25: original
records are never deleted, overwritten, silently rerun under changed conditions, or
relabelled). The one-revision rule means a second `REVISE` on the same decision is
**not available**.

---

## 10. Freeze readiness matrix

Initially every row is unresolved. This matrix makes the current freeze legality
unambiguous.

| Decision | Resolved? | Reviewer Assigned? | Approval Recorded? | Blocks Freeze? | Blocks Track M? |
|---|---|---|---|---|---|
| `/01` Baseline identity | **No** | **No** (`NOT ASSIGNED`) | **No** (`NOT APPROVED`) | **Yes** | **Yes** |
| `/02` Ensemble identity | **No** | **No** | **No** | **Yes** | **Yes** |
| `/03` Confident-decision definition | **No** | **No** | **No** | **Yes** | Class B primary |
| `/04` Gating bounds | **No** | **No** | **No** | **Yes** | Class B primary |
| `/05` Feature-contract construction | **No** | **No** | **No** | **Yes** | **Yes** |
| `/06` Window-gate definition | **No** | **No** | **No** | **Yes** | Class B primary |
| `/07` Calibration | **No** | **No** | **No** | **Yes** | **Yes** |
| `/08` Power floors | **No** | **No** | **No** | **Yes** | **Yes** |
| `/09` Dependence-aware method | **No** | **No** | **No** | **Yes** | **Yes** |
| `/10` Minimum seed / run rule | **No** | **No** | **No** | **Yes** | **Yes** |
| `/11` Member operating points | **No** | **No** | **No** | No | No (secondary) |
| `/12` Alert-volume semantics | **No** | **No** | **No** | **Yes** | No (supporting) |
| `/13` Large-dataset sampling | **No** | **No** | **No** | **Yes** | **Yes** where it applies |
| `/14` Segment / small-n withholding | **No** | **No** | **No** | No | No (secondary) |

**Freeze legality: NOT LEGAL.** 0/14 resolved · 0/14 reviewer assigned · 0/14
approved · 11 rows block the freeze · 8 rows block primary Track M.

This matrix is consistent with plan §35: while any mandatory checklist item is
unresolved, the affected confirmatory experiment is **NOT READY**, and
`scripts/check_freeze.py` exits non-zero by design.

---

## 11. Mechanical validation

A lightweight checker exists: **`backend/scripts/review_resolution_check.py`**
(`python backend/scripts/review_resolution_check.py`, `--self-test` for fixtures).
It follows the existing repository pattern of a document-consistency checker with a
built-in self-test, as used by `backend/scripts/claim_evidence_check.py`.

It verifies, and exits non-zero on any failure:

1. all **14** identifiers `/01`…`/14` exist, each **exactly once** as a sheet
   heading (a duplicated, fully-formed sheet is rejected);
2. **no decision is omitted** and **no extra decision ID exists** (so the list
   cannot silently grow or shrink);
3. every decision sheet contains all required fields (Decision, Why it matters,
   Existing evidence, Negative evidence, Allowed decision options, Recommended
   reviewer question, the seven-line reviewer block, Freeze consequence);
4. unresolved reviewer fields are **explicitly marked** — an empty
   `Reviewer:`/`Rationale:`/`Decision Date:` fails;
5. **`APPROVED` cannot coexist with `Reviewer: NOT ASSIGNED`**;
6. **`Approval: APPROVED` requires the decision itself to be `APPROVED`** —
   approval cannot be recorded against a `PENDING REVIEW` or `REJECTED` decision;
7. **`APPROVED` cannot coexist with `Approval Timestamp: NOT ESTABLISHED`**;
8. **no fabricated identity or date** — while a decision is still
   `PENDING REVIEW`, neither a reviewer name nor a decision date may be filled in;
   a `REJECTED` / `REVISE` / `APPROVED` ruling legitimately carries both;
9. the 13 negative-evidence rows N1–N13 are **all present** (none removed);
10. the artifact **creates no freeze record** — `docs/FREEZE_RECORD.json` must still
    be absent, and no git freeze tag may exist;
11. the memo and plan must still list the same 14 identifiers (no drift);
12. **`BOTH` decisions cannot be `APPROVED` without both reviewer components** —
    each sheet's `Required reviewer components:` line must match the §3 ownership
    table; a `BOTH` sheet must carry a domain component block; one component may
    never be approved while its counterpart is not; an approved component needs a
    rationale; a `STATISTICAL REVIEW` sheet may not carry a domain block.

Its `--self-test` runs **36 fixture checks**, and it was validated by
tampering the real artifact: duplicating a decision sheet, renaming `/14`→`/15`,
deleting a sheet, flipping `Approval` to `APPROVED` while the decision was pending,
setting `APPROVED` beside `Reviewer: NOT ASSIGNED`, naming a reviewer or a date
while pending, blanking a field, removing negative row `N7`, approving only the
statistical side of a `BOTH` decision, approving only its domain side, leaving a
counterpart ruling pending, omitting one of the two approvals, and having both
sides approved without updating the sheet's own final decision — each is detected
with a non-zero exit.

**Integration.** The checker is registered in the repository battery
(`.freebuff/p114_battery.sh`, evidence-enforcement block, next to
`claim_evidence_check.py`, `eval_record_test.py`, `prereg_harness_test.py` and
`check_freeze_test.py`). This is the established pattern for exactly this class of
artifact checker, and it **does not touch the scientific experiment pipeline** — no
model, threshold, dataset, metric or evaluation path is imported or executed.

**Why a checker was warranted rather than skipped.** The whole risk of this
artifact is that an unfilled decision later looks filled, or that `APPROVED` appears
next to `NOT ASSIGNED`. A prose review cannot reliably catch that across 14 sheets;
a 60-line deterministic check can, and it runs in CI.

---

## 12. Hard invariants — attestation

This task changed **no** scientific state. Verified after authoring:

| Invariant | State |
|---|---|
| `docs/FREEZE_RECORD.json` | **absent** — not created |
| Freeze tag | **none** — 0 git tags |
| Confirmatory Track M execution | **none** — no experiment run |
| Dataset acquisition | **none** — `data/external*` absent |
| New scientific experiment | **none** |
| New scientific metric | **none** — no metric defined, changed, or version-bumped |
| Threshold change | **none** — S1's locked 0.018758 untouched |
| Model change | **none** |
| Production model artifact change | **none** — `git status models/` clean |
| Reviewer identity fabricated | **none** — both reviewers `NOT ASSIGNED` |
| Approval fabricated | **none** — all 14 `NOT APPROVED` |
| Statistical decision silently resolved | **none** — 0/14 resolved |
| Negative NR-05 result removed or softened | **none** — N1–N13 preserved verbatim in §5 |
| Research Plan frozen | **no** — DRAFT / NOT FROZEN / NOT APPROVED |
| Preregistration frozen | **no** — `0.1.2-draft`, "Finalized: NO" |
| Evidence ledger append-only | **preserved** — no record deleted, rewritten, or reordered |
| Exposure ledger / eligibility manifest rewritten | **no** — byte-identical |
| New phase-numbering scheme | **none** — `/01`–`/14` are the memo's existing identifiers |

## 13. What this package does not claim

- that any decision has been made, or that any placeholder has been resolved;
- that a statistical or domain reviewer exists, has been appointed, or has signed;
- that IEEE-CIS or BAF is eligible, acquired, or verified (`BLOCKED`);
- that NR-05 is confirmatory or constitutes independent replication;
- that gating, calibration, ensemble superiority or cross-dataset generalisation is
  established — §5 records the opposite where the evidence says so;
- that the repository is ready to freeze. It is ready for **reviewer rulings**,
  which is a different thing.

---

## 14. Next legitimate action

The repository is now ready for actual statistical and domain reviewer rulings on
`/01`–`/14`. After a ruling is approved and recorded, and only then, may the
corresponding Research Plan or preregistration cell be edited as a separate,
recorded act. Freeze itself additionally requires real reviewer assignment (plan
§28), resolution of the 12 preregistration markers, the plan §35 checklist, the
exposure-ledger audit, and at least one eligible confirmatory dataset — none of
which this task performs.
