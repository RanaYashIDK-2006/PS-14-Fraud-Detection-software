# Statistical Review Decision Memo — Freeze Preparation

**Status:** PREPARATION DOCUMENT — DRAFT. **This memo does not freeze anything.**
**Research Plan state:** DRAFT / NOT FROZEN / NOT APPROVED.
**Confirmatory Track M:** BLOCKED.
**NR-05:** COMPLETE / EXPLORATORY.

**Purpose:** this is the single reviewer-facing entry point for the unresolved
statistical and freeze decisions catalogued in `docs/PHASE_NR05_EXPLORATORY_DIAGNOSTICS.md`
§20 (14 issues). It exists so a statistical reviewer can open one document,
systematically answer every reviewer-owned decision, record the decision and its
rationale, and see exactly what remains before the Research Plan can be frozen.

**What this memo is not.** It records no decision. It resolves no placeholder. It
introduces no new experimental evidence and no new scientific metric. Every number
quoted here already exists in a cited repository artifact. Where the Reviewer has
not yet ruled, the status is `REQUIRES DECISION` / `REQUIRES APPROVAL` /
`PENDING REVIEW` — never `RESOLVED`.

**Authored at git state:** `5a5ff55cc03df73318fdf31a16e3665f159a4720` (2026-10-04).
**No commit is implied.** This file is uncommitted working-tree content.

**Plan rule honoured:** plan §17 — document, never silently edit. Nothing in the
Research Plan, the preregistration, NR-05, the exposure ledger or the eligibility
manifest was modified by the production of this memo (hashes in §1.2).

---

## 0. How to use this memo

### 0.1 Reviewer workflow

1. Read §3 (exposure and independence boundary). This constrains every decision
   that follows: no decision may be justified by NR-05 performance alone.
2. Work §5 in order. Each decision record is self-contained: decision, why it
   matters, existing evidence, status, owner, required action, allowed options,
   freeze impact, evidence required, and an unfilled approval record.
3. Cross-check each record against §6, which frames the nine methodological review
   areas the plan §32 sign-off must cover.
4. Record every ruling in §7 (freeze readiness matrix) and in the per-decision
   approval record. Use the plan §22 outcome vocabulary only (§4).
5. Sign §9 only when §7 shows no remaining blocker for the scope you approved.

### 0.2 Constraints the memo places on the reviewer

- **No favourable-result selection (§9 of the review task).** The question on
  every record is *"which methodological choice is scientifically justified and
  can be frozen before untouched confirmatory evaluation?"* — never *"which option
  performed best in NR-05?"*
- **NR-05 is a prior-exposure set, not a tuning set.** NR-05 ran on ULB, Kaggle
  and IBM, all of which plan §14 classifies as **EXPLORATORY** (exposed). Their
  prior exposure is permanent; it is recorded, not reset.
- **No reviewer decision is pre-made here.** Where an option looks attractive in
  NR-05 that is stated as a *fact about NR-05*, never as an endorsement.

---

## 1. Sources

### 1.1 Authoritative documents (do not edit for this review)

| Artifact | Path | Role |
|---|---|---|
| Research Plan | `docs/RESEARCH_PLAN.md` | normative: all statistical rules; source of every `[TO BE FROZEN]` cell |
| Preregistration | `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` | DRAFT, 11 `[REQUIRES DECISION/APPROVAL]` markers |
| NR-05 diagnostics | `docs/PHASE_NR05_EXPLORATORY_DIAGNOSTICS.md` | the 14-issue catalogue (§20) and all exploratory evidence cited here |
| NR-05 driver | `backend/scripts/nr05_diagnostics.py` | the code that produced every cited exploratory number |
| NR-05 results | `reports/nr05/*.json`, `reports/nr05/artifacts/` | machine-readable exploratory outputs |
| Evidence ledger | `reports/evaluation_runs/eval_ledger.jsonl` + `record_<id>.json` | append-only evidence (`eval_record.py` v1.1) |
| Metric definitions | `docs/metric_definitions.md` (v1.0) | metric implementations, CI withholding rule, small-sample warnings |
| Authoritative contracts | `docs/authoritative_contracts.md` | canonical response/artifact shapes |
| Exposure ledger | `docs/evaluation/DATASET_EXPOSURE_LEDGER.md` (`exposure-ledger 1.0-draft`) | prior-exposure record |
| Eligibility manifest | `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` (`eligibility-manifest 1.0-draft`) | §13 identity fields and blocking gates per dataset |
| Freeze check | `scripts/check_freeze.py` + `backend/scripts/check_freeze_test.py` | plan §30 automated gate |
| RP-01 adoption record | `docs/PHASE_RP01_RESEARCH_PLAN_ADOPTION.md` | adoption, freeze ordering (§H), limitations (§L) |
| Roadmap reconciliation | `docs/PHASE_106_ROADMAP_RECONCILIATION.md` | authoritative roadmap (§J–§L) |

### 1.2 Invariant hashes at memo authoring

| Artifact | SHA-256 |
|---|---|
| `docs/RESEARCH_PLAN.md` | `eab5a0615801db7e50ab4d1c0ec73f93905e461b5bd8eacc617e0418d656b9ce` |
| `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` | `fdde71cbd00e8e34ce439f3cc582e31e73452a0405ea3ff51c11136adc9065a6` |
| `docs/PHASE_NR05_EXPLORATORY_DIAGNOSTICS.md` | `9895fe8215606b31caf82cca6322a6cbead383650bf705f9ff47b805f0f6c457` |
| `docs/evaluation/DATASET_EXPOSURE_LEDGER.md` | `3b1b280040379407b49aff68c79672ea30fae1e10d5233dd9a92f6e4667788e4` |
| `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` | `92b766bea42642b84e76e911e2ac0427bfb151123cc33ae7474382eee117efc5` |
| `docs/metric_definitions.md` | `8665549b3d62fef36402d737910cd8aa1acfbde025dfee4aa1395a9e3d99ed07` |

A document cannot carry its own hash (plan §29); the memo's hash belongs in
`docs/FREEZE_RECORD.json` **at freeze**, which does not exist.

---

## 2. Current freeze state (recorded, not changed)

`./.venv/Scripts/python.exe scripts/check_freeze.py` → **exit 1**, **78 problems**:

| Composition | Count | Nature |
|---|---:|---|
| `missing docs/FREEZE_RECORD.json` | 1 | expected pre-freeze (plan §29/§30.2) |
| unresolved placeholder in `docs/RESEARCH_PLAN.md` | 65 | the §35 checklist cells and §3/§5/§6/§10/§18/§19/§23/§24/§28/§34 decisions |
| `[REQUIRES DECISION/APPROVAL]` marker in the preregistration | 12 | 11 documented markers + one marker line the scanner counts |

Also true at authoring: 0 git tags; no `docs/FREEZE_RECORD.json`; 56
`TO BE FROZEN` occurrences in the plan; `backend/scripts/check_freeze_test.py`
22/22 PASS. **The freeze check must continue to fail.** No placeholder was
resolved to make it pass.

---

## 3. Exposure and independence boundary

### 3.1 Previously exposed (NR-05 used these)

| Dataset | Plan §14 status | SHA-256 (verified unchanged) | NR-05 role |
|---|---|---|---|
| ULB `data/creditcard.csv` | EXPLORATORY | `76274b69…` | exploratory diagnostics only |
| Kaggle `fraudTrain` / `fraudTest` | EXPLORATORY | `fd713920…` / `12d553ab…` | exploratory diagnostics only |
| IBM v2 | EXPLORATORY | `b01fa323…` | exploratory diagnostics only |
| PS-14 synthetic `data/transactions.csv` | SIMULATED/SYNTHETIC | `9f0f56bf…` | not used in NR-05 |

NR-05 §3 states it directly: *"Nothing in NR-05 makes an exposed dataset
'unTouched'."* All five repository dataset hashes were re-verified unchanged after
the runs.

### 3.2 Potential future untouched confirmatory datasets

| Dataset | Plan §14 status | Manifest state | Gate |
|---|---|---|---|
| IEEE-CIS | CANDIDATE CONFIRMATORY | **BLOCKED — not acquired** | `data/external/` absent; licence/terms unread; labelled portion unverified |
| BAF | CANDIDATE CONFIRMATORY | **BLOCKED — not acquired** | `data/external_benchmark/` absent; licence/provenance/generation process unverified |

Both remain **dataset-eligibility decisions, not statistical conclusions**. This
memo does not declare either dataset eligible, and acquiring them is out of scope
(plan §28 manual dependencies, status `PENDING`, owners `[OWNER]`).

### 3.3 What NR-05 cannot be used for

NR-05 results **cannot** support a claim of independent replication (plan §14,
§33 Track M non-claims). They may inform *methodological* choices about what to
freeze. Any confirmatory Track M result on ULB/Kaggle/IBM is dataset-specific
evidence under explicitly frozen new-model rules — never broad cross-dataset
replication.

---

## 4. Decision framework (plan §22 vocabulary only)

The Reviewer must select one of these outcomes per decision. No competing
framework is introduced by this memo.

| Outcome | Meaning (plan §22) |
|---|---|
| `GO` | all required primary criteria satisfied |
| `NO-GO` | adequately powered eligible evaluation ran and a required criterion failed |
| `REVISE` | a frozen revision condition explicitly permits one targeted revision |
| `INCONCLUSIVE` | evidence insufficient to establish success or failure, including underpowered evaluation |
| `BLOCKED` | required data/access/dependency unavailable |
| `NOT ESTABLISHED` | evidence required for the claim has not been demonstrated |
| `NOT APPLICABLE` | criterion genuinely does not apply to the dataset/track |

"These outcomes must not be collapsed" (§22). `NOT ESTABLISHED` and `BLOCKED` are
the correct states for most records below — they are **not** pending-analysis
states and must not be upgraded by re-running anything.

**REVISE is narrow (plan §24).** Permitted only for a documented *defect in the
experiment or pipeline*, established independently of whether the result was
favourable. **Not eligible:** a null or negative primary result, a CI containing
zero, a failed threshold, an unfavourable ablation, or a wish to try another model
or feature set. At most **one** revision cycle. See §8 of this memo.

---

## 5. Decision records — the fourteen §20 issues

Identifiers are citations of NR-05 §20 item numbers (`NR05-§20/01` … `/14`). This
is a citation label, **not** a new project-wide phase numbering system.

Each record carries: Decision · Why it matters · Existing evidence · Current
status · Decision owner · Required reviewer action · Allowed options · Freeze
impact · Evidence required · Approval record.

---

### NR05-§20/01 — Baseline identity

- **Decision:** freeze the single-model primary baseline that the ensemble must
  beat (plan §3 RQ-M1 "Baseline identity": **Primary baseline: `[TO BE FROZEN]`**).
- **Why it matters:** RQ-M1 is the primary signal criterion. The baseline, tuning
  budget, split, feature representation and preprocessing must be identical to the
  ensemble's (plan §3), and "a baseline may not be deliberately under-tuned". The
  plan §35 pre-freeze checklist carries "baseline identity frozen". Without it,
  §20 (above-chance / paired difference) has no reference arm.
- **Existing evidence:** NR-05 §7 reports **both** LR and XGB under one fixed
  a-priori config (no hyperparameter search) and resolves nothing; the strongest
  single model **differs by dataset** (LR on ULB by ROC-AUC 0.9739, XGB on Kaggle
  0.9970 — NR-05 §6). NR-05 §6 paired deltas: ensemble vs XGB is negative
  everywhere; ensemble vs LR is +0.1285 [+0.1175, +0.1405] on Kaggle and
  +0.1381 [+0.1236, +0.1528] on entity-disjoint, but inconclusive on ULB
  (−0.0150 [−0.0408, +0.0041]).
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer (plan §32, §28 `[STATISTICAL REVIEWER]` — not yet named). Protocol §5 sign-off block lists "Lead researcher" and "Statistical reviewer (external, R-42)", both `_pending_`. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** name the single-model baseline family **before** any confirmatory dataset is opened, and state whether selection is fixed a-priori or made per dataset on the **validation window only**.
- **Allowed options:** (A) one fixed baseline family for all datasets, named a-priori; (B) per-dataset selection on validation only, with the rule frozen; (C) both reported with a pre-declared primary and secondary; (D) reject/defer — declare RQ-M1 `NOT ESTABLISHED` for Track M and drop the comparison.
- **Freeze impact:** **Blocks freeze** (plan §35 item). **Blocks Track M** — RQ-M1 is a candidate primary RQ (plan §23).
- **Evidence required:** a frozen, version-identified baseline configuration in the repository (the plan requires the exact implementation be frozen by repository artifact/configuration, plan §3), plus the demonstration that its tuning budget equals the ensemble's.
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/02 — Ensemble identity (mean fusion loses to the best single model)

- **Decision:** freeze the exact ensemble implementation for confirmatory Track M
  (plan §3 "Selected ensemble: `[TO BE FROZEN]`"), given that the mean-fusion
  candidate **lost to the best single model on all four NR-05 discrimination runs**.
- **Why it matters:** RQ-M1 and RQ-M2 both depend on it; plan §14 forbids search
  over IEEE-CIS/BAF after inspecting results; plan §35 carries "exact ensemble
  identity frozen". This is the single most consequential open choice in the memo.
- **Existing evidence:** NR-05 §6/§8 — mean fusion (LR+RF+XGB+IF → Platt on cal)
  paired ROC-AUC deltas vs XGB: ULB −0.0145 [−0.0308, −0.0036] (block bootstrap);
  Kaggle −0.0338 [−0.0407, −0.0267]; entity-disjoint −0.0276 [−0.0344, −0.0210];
  IBM −0.2663 [−0.3483, −0.1830]. **All CIs exclude zero; the ensemble is worse.**
  NR-05 attributes this to weak RF/IF members diluting LR/XGB. This is an
  **exploratory, small-n, exposed-dataset** finding (ULB test n=75 positives), not
  a confirmatory result, and must not be used to select a winner on untouched data.
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer, with Lead researcher supplying the frozen implementation artifact. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** decide what is frozen as the "ensemble" arm — and state explicitly how the NR-05 negative result is handled without post-hoc selection on confirmatory data.
- **Allowed options:** (A) freeze the mean-fusion ensemble as-is and accept a likely NO-GO on RQ-M1; (B) freeze the strongest single-model family **as the primary arm** and demote the ensemble to a secondary/ablation arm; (C) freeze a **pre-specified architecture comparison** (mean vs weighted vs stacked, decided a-priori, multiplicity-adjusted per plan §3) so the confirmatory run *tests* the question instead of assuming it; (D) freeze a different named fusion family defined independently of NR-05; (E) reject/defer — RQ-M1 becomes `NOT ESTABLISHED`.
- **Freeze impact:** **Blocks freeze** (plan §35). **Blocks Track M.**
- **Evidence required:** a repository-frozen ensemble specification with an implementation hash; the plan §3 "no search after inspection" rule written into that specification; for option (C), a frozen multiplicity method (plan §3 "Multiplicity method").
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/03 — Confident-decision definition (plan §5.3)

- **Decision:** freeze the "confident decision" band used by the wrong-confident
  metric (plan §5.3). NR-05 used ML `p ≥ 0.8 or p ≤ 0.2`; that was an **NR-05
  choice**, not a plan value.
- **Why it matters:** §5.3 defines the numerator of the Class B primary gating
  criterion (§5.5) and of the "wrong confident decisions" rate used throughout
  NR-05 §11–§12. A different band changes every gating number. Plan §35 carries
  "confident-decision definition frozen".
- **Existing evidence:** NR-05 §9 records the band as exploratory; NR-05 §11/§12
  show the metric is band-sensitive in practice (ULB row-gate WC 320–321 vs
  ungated 404–414; Kaggle σ1.0 13,060 vs 14,391). No plan or protocol value exists.
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer (definition); Domain reviewer (whether the band is operationally meaningful). Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** fix the band, and whether "confident" is defined on the calibrated probability, the raw member score, or the fallback rules output (NR-05 treated fallback rules as always-confident).
- **Allowed options:** (A) fixed probability band, named numerically, frozen; (B) quantile-based band over the calibration window; (C) two frozen bands reported separately (high-confidence and low-confidence); (D) reject/defer — the wrong-confident criterion becomes `NOT ESTABLISHED` and §5.5 must be re-specified on a different primary.
- **Freeze impact:** **Blocks freeze.** **Blocks the Track M Class B primary criterion only** (RQ-M3 Class B); not discrimination.
- **Evidence required:** a frozen definition in the plan/protocol, plus confirmation it is computable identically in the gated and ungated systems (plan §5.1 requires differences be attributable to the gating decision only).
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/04 — Gating bounds (plan §5.5)

- **Decision:** freeze the five §5.5 bounds: minimum reduction in wrong confident
  decisions, minimum coverage, maximum coverage loss, maximum recall loss, and the
  fallback/rules quality floor. **All five are `[TO BE FROZEN]`.**
- **Why it matters:** §5.5 is the primary gating criterion. The plan states
  explicitly: *"A gate that rejects all or nearly all cases cannot pass because its
  coverage constraint will fail."* Plan §35 carries "coverage/recall bounds
  frozen" and "fallback error accounting frozen".
- **Existing evidence:** NR-05 §11–§12 show the bounds are **decisive**, not
  cosmetic. ULB row gating reduced wrong-confident 6/6 Class B conditions but at
  clean recall **0.813 → 0.040** — the statistical contract blocked 72/75 test
  frauds (NR-05 §5.0 issue `/05`). The window gate reached coverage **0** on ULB by
  rejecting everything (clean PSI 1.46). On Kaggle the row gate bought a 12–14pp
  recall loss for a ~1,300 wrong-confident reduction at σ1.0. The **only** numeric
  proposal anywhere in the repository is the preregistration Criterion 3 MMD:
  ≥20% relative reduction at ≤2 percentage-point coverage loss — explicitly
  annotated there as *"an assumption to justify: reflects an operational risk
  posture — requires a domain assumption about the cost of confident errors vs
  abstentions."*
- **Current status:** `NOT ESTABLISHED` (no bound is justified by existing evidence) **and** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer (bounds) **jointly with** Domain reviewer (the cost-of-abstention posture behind them). Protocol §7 makes the same point for business metrics. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** set all five bounds a-priori, or record that the criterion cannot be specified and mark RQ-M3 Class B `NOT ESTABLISHED`.
- **Allowed options:** (A) adopt the preregistration's ≥20% / ≤2pp proposal plus frozen recall-loss and fallback-quality floors; (B) set different a-priori bounds justified independently of NR-05; (C) replace the scalar bound with a **coverage–risk curve** criterion (allowed by the plan, which requires the curve to fail a degenerate all-reject gate); (D) reject/defer — RQ-M3 Class B = `NOT ESTABLISHED`, gating remains a **safety mechanism only** with no effectiveness claim.
- **Freeze impact:** **Blocks freeze.** **Blocks the Track M Class B primary criterion only.** Does not block the discrimination RQs.
- **Evidence required:** a frozen numeric or curve-based rule; a frozen fallback-quality floor (NR-05's fallback was `amount > TRAIN q99.5 → FRAUD else LEGIT`, an a-priori exploratory choice); the domain assumption behind any coverage-loss tolerance, recorded as such.
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/05 — Feature-contract construction (plan §5.0)

- **Decision:** freeze how each dataset's Track-M feature contract (types, ranges,
  missing policy) is constructed — statistical quantile bounds, semantic bounds,
  or a hybrid with per-feature-type rules and padding.
- **Why it matters:** plan §5.0 requires the dataset-specific contract, drift
  baseline, window size and thresholds to be frozen **before** inspecting that
  dataset's confirmatory data, using training-window data only, and forbids
  deriving any contract from validation or test windows. Plan §35 carries "feature
  semantics verified" and gating-injection freezes.
- **Existing evidence:** NR-05 §9 used training-only bounds `q01/q99 ± 2·IQR`,
  which failed three ways (NR-05 §20 item 5): (a) **tail-dwelling fraud** — ULB
  blocked 72/75 test frauds, collapsing clean recall to 0.040; (b) **categorical
  codes with temporal mix-shift** — IBM `chip_code` train bounds [1.0, 2.0] while
  70.6% of test rows carry code 0; (c) **constructed-feature NaNs** — 33,685 IBM
  negative-amount rows make `log1p` NaN, so they are treated as "missing" and
  blocked.
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer (bound construction rule) with Domain reviewer (which fields carry real semantics). Dataset-eligibility interaction: manifest §3/§4 feature-semantics rows. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** specify the contract-construction rule by feature type, including categorical codes, tail-heavy monetary features and engineered transforms, plus the padding policy.
- **Allowed options:** (A) semantic bounds where the dataset documents them, statistical quantiles elsewhere; (B) statistical quantiles with frozen per-type padding and an explicit tail-overlap allowance; (C) contract on a documented **train-window subsample** with a frozen rule; (D) reject/defer — no Track-M gating experiment is specified.
- **Freeze impact:** **Blocks freeze.** **Blocks Track M** (both Class A and Class B, since the contract *is* the row gate).
- **Evidence required:** a per-dataset contract specification derived only from the training window and **verified not to be derived from validation/test**; the manifest feature-semantics row for each dataset (`NOT ESTABLISHED` for several today).
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/06 — Window-gate definition (plan §5.0 second component)

- **Decision:** freeze the window-level drift monitor's definition: baseline
  window, window size, the any-feature vs fraction-of-features rule, the block
  rule, and its applicability where natural drift already exists.
- **Why it matters:** plan §5.0 names window-level drift monitoring as the
  component **designed to detect** Class B shift. It must be frozen per dataset
  before confirmatory inspection, and "Class B results must not be credited" to
  row-level enforcement.
- **Existing evidence:** NR-05 §9 used `src/drift_monitor/psi.py` unchanged —
  train-window bins, block when **any** feature PSI ≥ 0.25 (the module's documented
  ALERT constant). Results (§11): on Kaggle it **never fired** on any Class B
  condition (max PSI **0.083** vs the 0.10 warn level) → **NOT DETECTED**; on ULB it
  fired on **all** windows because real temporal drift gives clean PSI **1.46**
  (7/30 features ≥ 0.25) → coverage **0**; on IBM coverage 0 in 7/7 (contract
  defect, see `/05`). The monitor was therefore **unusable as configured on both**
  endpoints — over-triggering on real drift, silent on subtle shift.
- **Current status:** `NOT ESTABLISHED` (no demonstrated Class B detection) **and** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer (rule), Domain reviewer (what shift magnitude is operationally material). Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** decide whether a window gate is in the confirmatory specification at all, and if so freeze its detection target, window size, feature-aggregation rule and block behaviour.
- **Allowed options:** (A) freeze an **PSI-magnitude–calibrated** window gate whose threshold is set from the frozen training-window drift distribution rather than a fixed constant; (B) freeze a **fraction-of-features** rule plus a minimum-drift guard so real baseline drift cannot alone trigger a block; (C) **narrow the claim**: window monitoring is reported as a monitoring component with `NOT DESIGNED TO DETECT` for Class B at frozen magnitudes, and Class B detection is not claimed; (D) reject/defer — drop the window component from Track M.
- **Freeze impact:** **Blocks freeze.** **Blocks the Track M Class B primary criterion only.**
- **Evidence required:** a frozen per-dataset window definition (baseline window, size, aggregation, thresholds) built from training data only; a stated minimum detectable shift magnitude. **No new experiment is required to choose — but note that choosing between (A)/(B)/(C) on the basis of NR-05 outcomes would be post-hoc selection on exposed data and is disallowed.**
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/07 — Calibration procedure (plan §6), including the IBM inversion

- **Decision:** freeze the calibration method, the primary calibration metric, the
  acceptance bound, and — new, surfaced by NR-05 — a **calibrator-sanity guard**
  and a **calibration-population floor**.
- **Why it matters:** plan §6 fixes the sequence training → calibration/validation
  → test and states the test window must not influence calibrator fitting; all
  three calibration cells are `[TO BE FROZEN]`. Calibration sits on the detection
  critical path: an inverted calibrator produced recall 0.000 in NR-05's IBM run.
- **Existing evidence:** NR-05 §13 — Platt on the cal window only improved ULB
  ECE **0.03912 → 0.00033**. On IBM the same frozen `PlattCalibration` learned
  **coef −0.7029, intercept −6.4186** under **185 cal positives**, flipping the
  ranking (cal raw AUC 0.652 → 0.348; test 0.627 → **0.373, below chance**), while
  **Brier 0.00102 and ECE 0.00045 looked excellent**. NR-05 reproduced the
  inversion bit-identically on an independent refit and did **not** resolve the
  cause. The inversion is preserved here verbatim; it is not explained away.
  Calibration-positive counts: ULB **57**, Kaggle 2,385, entity 1,867, IBM **185**.
- **Current status:** `NOT ESTABLISHED` (the IBM case demonstrates Brier/ECE alone are insufficient) **and** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer; Domain reviewer for the acceptance bound's operational meaning. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** freeze (i) the calibration method, (ii) the primary metric, (iii) the acceptance bound, (iv) a sanity guard (e.g. refuse a calibrator with a sign-flipped or near-flat slope, or fall back to identity), and (v) a minimum calibration-positive count below which calibration is `INCONCLUSIVE` and the threshold-free metrics only are reported.
- **Allowed options:** (A) Platt with a frozen sanity guard **and** a calibration-positive floor, both mandatory; (B) isotonic or another monotone calibrator with the same guard; (C) report calibration-free discrimination as primary and treat calibrated probabilities as secondary until a guard exists; (D) reject/defer — RQ-M4 = `NOT ESTABLISHED`.
- **Freeze impact:** **Blocks freeze.** **Blocks Track M** (RQ-M4 is a candidate primary RQ).
- **Evidence required:** a frozen calibrator specification including the guard and floor; a written rule that Brier/ECE may never be reported without the ranking check (NR-05 §13 caveat, plan §4A.3 reporting rules). The guard must be justified as a property of the procedure, **not** fitted to make IBM pass.
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.
---

### NR05-§20/08 — Power floors (plan §18)

- **Decision:** freeze the three §18 minimums — **minimum fraud count**, **minimum
  total evaluation count**, **minimum valid seeds/runs** — all currently
  `[TO BE FROZEN]`. Resolve the `min_cell` discrepancy (§6.1 below).
- **Why it matters:** §18 is explicit: *"If these requirements are not satisfied:
  INCONCLUSIVE / UNDERPOWERED … The result does not count as GO or NO-GO for the
  affected criterion."* A floor chosen after seeing operating-point estimates is
  exactly the post-hoc selection plan §36 forbids ("Results must never determine
  the rules used to judge those same results").
- **Existing evidence:** NR-05 §20 item 8 — **ULB test 75 positives / cal 57;
  IBM test 134 / cal 185**. Every operating-point and calibration conclusion in
  NR-05 on those two datasets is small-n-exposed. NR-05 §14 records ULB quartile 4
  at 11 positives and a segment cell at **n=9 positives** marked INCONCLUSIVE.
  NR-05 states these floors *"will likely mark several of these INCONCLUSIVE."*
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** set all three minimums a-priori, and state whether the floor applies per test window, per dataset, or per cell.
- **Allowed options:** (A) set numeric minimums justified from the estimand's precision requirement; (B) set a precision-based rule (e.g. minimum half-width of the primary CI) instead of raw counts; (C) adopt the most conservative of several pre-registered proposals, accepting more INCONCLUSIVE outcomes; (D) reject/defer — no adequately powered confirmatory experiment is possible and Track M is `BLOCKED`.
- **Freeze impact:** **Blocks freeze** (plan §35 "minimum fraud-count rule frozen"). **Blocks Track M.**
- **Evidence required:** the frozen rule; a per-candidate-dataset feasibility check showing which of the §13-listed datasets could satisfy it (currently `BLOCKED` for IEEE-CIS/BAF — no data).
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/09 — Dependence-aware uncertainty method (plan §19)

- **Decision:** declare which uncertainty method is **authoritative** for
  confirmatory Track M, and freeze the applicability rule and replicate count.
- **Why it matters:** §19 states *"Ordinary IID bootstrap is therefore not the
  default confirmatory method"* and requires the applicability rule to be frozen
  **before results are examined**, applied consistently for paired comparisons.
  All three §19 cells are `[TO BE FROZEN]`. Plan §35 carries "dependence-aware CI
  method frozen".
- **Existing evidence:** NR-05 §16 explored **three** methods on the same data —
  IID stratified percentile bootstrap (n=300, seed 42), **entity-clustered**
  (`cc_num` / `User` resampled with replacement; Kaggle, IBM) and **temporal block**
  (20 contiguous time blocks; ULB) — and labelled every dependence-aware interval
  **"EXPLORATORY — METHOD NOT YET FROZEN"**, with IID explicitly not the default
  interpretation. n=300 was an NR-05 exploratory choice; the preregistration §2
  separately proposes 2,000 replicates with a `[REQUIRES DECISION/APPROVAL]` seed.
  These are **not** interchangeable and none is promoted here.
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** name the authoritative method per §19's three applicability branches, the replicate count, the interval construction, and how the pairing is preserved for model-vs-model deltas.
- **Allowed options:** (A) entity-clustered where meaningful entity ids exist, temporal-block where serial dependence is material, iid otherwise — exactly §19's structure, with counts frozen; (B) a single conservative method for all datasets; (C) report all applicable methods with one declared primary (note: NR-05's own §3 shows multiple methods can disagree in width — this is a reporting-width cost, not a correctness cost); (D) reject/defer — no confirmatory interval method, Track M = `BLOCKED`.
- **Freeze impact:** **Blocks freeze.** **Blocks Track M.**
- **Evidence required:** the frozen method per branch, replicate count, seed, and CI level; and confirmation the paired comparison uses identical resample indices for both arms (NR-05 did this).
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/10 — Minimum seed / run rule

- **Decision:** freeze the minimum number of valid seeds/runs per experiment
  (plan §18 "Minimum valid seeds/runs: `[TO BE FROZEN]`") and the seed set.
- **Why it matters:** single-seed results are not estimable evidence. Plan §35
  carries "minimum fraud-count rule frozen" (same §18 block); the preregistration
  §5.1 proposes 5 fixed seeds behind a `[REQUIRES DECISION/APPROVAL]` marker and
  §9.9 prohibits post-hoc seed exclusion.
- **Existing evidence:** NR-05 ran **single seed 42** for every experiment
  (§16, §23 limitation list). The protocol's 5-seed proposal exists but is
  unapproved.
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** fix the seed count, the seed values, and the reporting rule (all pre-declared seeds appear in the matrix; a crashed seed may be rerun only with the identical seed).
- **Allowed options:** (A) adopt the preregistration's 5 fixed seeds; (B) a different pre-declared count with a stated precision rationale; (C) single seed with results labelled as such — defensible only for underpowered datasets that will be `INCONCLUSIVE` anyway; (D) reject/defer.
- **Freeze impact:** **Blocks freeze.** **Blocks Track M.**
- **Evidence required:** the frozen seed list and the §9.9-style no-exclusion rule written into the protocol.
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/11 — Member operating-point reporting rule

- **Decision:** freeze how per-member operating points are computed and reported
  in confirmatory Track M (per-member thresholds vs the shared ensemble-selected
  threshold).
- **Why it matters:** RQ-M2 requires the ensemble to be compared with its relevant
  constituent models. If members are scored at a threshold selected for the
  ensemble, member recall numbers are not comparable to the ensemble's and the
  ablation table invites a wrong reading.
- **Existing evidence:** NR-05 §20 item 11 — the per-member
  `recall_at_1pct_fpr` figures use the **shared ensemble-selected threshold**, not
  per-member thresholds. This was an exploratory reporting choice, recorded, not
  defended.
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** fix the operating-point convention for constituent models and state which table is the reportable one.
- **Allowed options:** (A) shared ensemble-selected threshold for every member, with the caveat that it is not each member's own optimum (NR-05's approach); (B) per-member threshold chosen on the **validation window only**, matching each member's own 1%FPR operating point; (C) report both, with one declared primary; (D) reject/defer — report threshold-free metrics only for members.
- **Freeze impact:** **Affects only a secondary analysis** (RQ-M2 ablation, a supporting RQ). Does not block the primary discrimination RQs. Not an explicit §35 checklist cell, so **does not block freeze on its own**, but must be frozen before any confirmatory ablation is reported.
- **Evidence required:** the frozen convention; evidence that any per-member selection used validation data only.
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.


---

### NR05-§20/12 — Alert-volume semantics

- **Decision:** freeze whether the fixed-alert-volume operating point is
  **cal-anchored** (a fixed top-k or alert rate chosen on the calibration window
  and applied unchanged to test) or **test-ranked** (top-k% recomputed on the test
  score distribution).
- **Why it matters:** plan §10 (RQ-M8) and §35 "business metrics frozen". The two
  conventions answer different questions: a cal-anchored volume measures
  operational stability over time; a test-ranked volume measures the best case
  attainable on that window. Only the first supports an operational reading.
- **Existing evidence:** NR-05 §15 used **test-ranked top-k%**, computed on the
  test score distribution with labels unused, and labelled ULB per-day rates as
  dataset artefacts rather than forecasts (the ULB `Time` axis spans ≈48 hours in
  total). IBM alert-volume recall ≈ 0 is the inverted-calibrator effect (§13), not
  an operational statement.
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer, with Domain reviewer for any operational interpretation. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** fix the anchor and name the frozen primary business metric.
- **Allowed options:** (A) cal-anchored alert volume and rate; (B) test-ranked, explicitly labelled non-operational; (C) both, with one declared primary; (D) reject/defer — RQ-M8 becomes `NOT APPLICABLE` / `NOT ESTABLISHED` for Track M.
- **Freeze impact:** **Blocks freeze** (§35 checklist item). **Does not block the primary Track M RQs** — RQ-M8 is a supporting RQ (plan §23 primary candidates are RQ-M1, RQ-M3 Class B, RQ-M4).
- **Evidence required:** the frozen anchor and metric. Per plan §10 and preregistration §7, any analyst-burden figure must be a **clearly labelled proxy**, never measured institutional analyst burden.
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/13 — Large-dataset sampling rule

- **Decision:** freeze the sampling rule (if any) for datasets too large to load
  and score in full, and confirm it is fixed **before** any inspection that could
  influence method choices.
- **Why it matters:** plan §14 forbids choices informed by untouched confirmatory
  data. Plan §31 requires dataset identity, population and split provenance for
  every quantitative result, and a subsampled population is a different population.
- **Existing evidence:** NR-05 loaded IBM v2 (**24,386,899 data rows, 2.35 GB**) with a **stride-1/37 subsample**, recorded as an exploratory choice. IBM is exposed, so this carried no confirmatory risk there; the rule becomes live the moment a candidate confirmatory dataset is large.
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer, with dataset-eligibility input from the manifest owner (§28 `[OWNER]`). Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** state the rule — full population, frozen deterministic stride, or frozen random sample with recorded seed — and confirm it is fixed before acquisition or inspection of any confirmatory dataset.
- **Allowed options:** (A) full population wherever the environment allows, with streaming score computation; (B) frozen deterministic stride with the stride factor recorded in the evidence record; (C) frozen random subsample with seed and rate recorded; (D) reject/defer — any dataset exceeding the compute budget is `BLOCKED`.
- **Freeze impact:** **Blocks freeze** (precondition for §14 compliance). **Blocks Track M** for any dataset where it applies; `NOT APPLICABLE` for small datasets.
- **Evidence required:** the frozen rule; for option (B)/(C), the population definition and hash recorded per plan §31.
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

### NR05-§20/14 — Segment / time-window small-n withholding

- **Decision:** reconcile the repository's **existing** small-sample rule with the
  §18 power floors — which rule governs a cell with fewer than N positives, and
  whether such a cell is reported as withheld, `INCONCLUSIVE`, or `NOT APPLICABLE`.
- **Why it matters:** `docs/metric_definitions.md` (v1.0, the authoritative metric
  implementation) already mandates a **CI withheld** result when a conditioning
  cell has **< 30** observations, and `SMALL_POSITIVE_CLASS` /
  `SMALL_NEGATIVE_CLASS` warnings below **50** positives or negatives. Plan §18
  proposes a different, currently unset, minimum fraud count. Two documents can
  therefore require different things from the same cell.
- **Existing evidence:** NR-05 §14 — ULB quartile 4 has 11 positives (AUC 0.670,
  marked INCONCLUSIVE-leaning) and the ULB amount-quartile segment q2 has **9
  positives** at AUC 0.441, which NR-05 **withheld** rather than report as
  performance. Under the v1.0 rule that cell's CI is withheld for two independent
  reasons; under an unset §18 floor its status is undefined.
- **Current status:** `REQUIRES DECISION`
- **Decision owner:** Statistical reviewer (reconciliation rule); the metric-definition owner for any change to `metric_definitions.md` — currently `METRIC_DEFINITIONS_VERSION = "1.0"`, so a change requires a version bump. Individual assignee: `NOT ASSIGNED` (role only; plan §28 leaves every owner unnamed).
- **Required reviewer action:** state the governing rule and its precedence over `metric_definitions.md`.
- **Allowed options:** (A) §18 floor applies to **primary criteria** while `metric_definitions.md` v1.0 continues to govern **CI withholding and warning flags** — complementary, not competing; (B) §18 floor supersedes entirely and `metric_definitions.md` is re-versioned; (C) leave v1.0 as the sole rule and set §18's minimum fraud count equal to it; (D) reject/defer — leave both unset and treat every sub-floor cell as `INCONCLUSIVE`.
- **Freeze impact:** **Affects only a secondary analysis** (segment and time-window reporting, RQ-M9). Does not block the primary RQs and **does not block freeze on its own**.
- **Evidence required:** the written precedence rule; if option (B), a re-versioned `metric_definitions.md` with a changelog entry.
- **Approval record:** Reviewer `PENDING REVIEW` · Decision `PENDING REVIEW` · Date `PENDING REVIEW` · Rationale `PENDING REVIEW` · Approval `PENDING REVIEW`.

---

## 6. Cross-cutting methodological review areas (plan §32 sign-off scope)

These nine areas are what plan §32 requires the statistical reviewer to approve.
Each is a framing of the decisions above, not a new decision.

### 6.1 Power and sample-size rules — decision `/08`, `/14`

The §18 minimum fraud count, minimum total evaluation count and minimum valid
seeds/runs are all unset, and the existing `min_cell` rule in
`metric_definitions.md` (withhold below 30 in the limiting class; warn below 50)
does not currently reconcile with §18. **The Reviewer does not choose a number
here.** The memo records the discrepancy, the affected populations (ULB test 75
positives / cal 57; IBM test 134 / cal 185; ULB quartile 4 = 11 positives; ULB
segment q2 = 9 positives) and the permitted outcomes: set a floor a-priori, or
accept that several criteria will land on `INCONCLUSIVE / UNDERPOWERED`, which
plan §18 says counts as neither GO nor NO-GO.

### 6.2 Dependence-aware uncertainty — decision `/09`

Plan §19 makes IID non-default and requires the applicability rule frozen before
results are examined. NR-05 exercised **IID, entity-clustered and temporal-block**
resampling and labelled all dependence-aware intervals *METHOD NOT YET FROZEN*.
**No exploratory bootstrap choice is promoted to the frozen method by this memo.**
The Reviewer names the authoritative method, its applicability branch, the
replicate count (NR-05 used 300; the preregistration proposes 2,000 behind an open
marker) and the seed.

### 6.3 Gating — decisions `/03`, `/04`, `/05`, `/06`

The four NR-05 gating findings, stated without softening:

- **Class A (designed-for invalidity):** row-level enforcement blocked ~100% of
  injected rows on ULB and Kaggle (NR-05 §10). Secondary evidence.
- **Class B benefit: `NOT ESTABLISHED`.** This is the primary exploratory
  question (plan §5.2, §5.5) and it did not produce a positive result.
- **Row gating caused severe clean-recall loss:** ULB clean recall **0.813 →
  0.040**, because the statistical contract blocked 72/75 test frauds
  (`/05`). On Kaggle the row gate did lower wrong-confident decisions in **7/7**
  Class B conditions, but at a **12–14 pp** recall cost.
- **Window PSI gating over-fired on ULB** (clean PSI **1.46** → coverage **0**)
  **and failed to detect the Kaggle Class B shift** (max PSI **0.083** below the
  0.10 warn level) → `NOT DETECTED` for Class B at the tested magnitudes.
- **IBM gating is `INCONCLUSIVE`** — `chip_code` train bounds [1.0, 2.0] against
  70.6% of test rows carrying code 0, plus 33,685 negative-amount rows whose
  `log1p` is NaN. This is a contract defect (`/05`), **not** evidence for or
  against gating.

The memo does not convert this into a positive claim, and the Reviewer is not
asked which gating option "worked best".

### 6.4 Ensemble architecture — decision `/02`, `/01`

Mean fusion **lost to the best single model on all four NR-05 discrimination
runs**, with paired CIs excluding zero (ULB −0.0145 [−0.0308, −0.0036]; Kaggle
−0.0338; entity-disjoint −0.0276; IBM −0.2663). It beat LR on Kaggle and
entity-disjoint. The Reviewer decides what to freeze: a fixed ensemble, the
strongest single-model baseline, both, a pre-specified architecture comparison, or
another named alternative. **Selecting the NR-05 winner and calling it
preregistered is not available as an option.** Prior NR-05 exposure on ULB,
Kaggle and IBM remains documented in `docs/evaluation/DATASET_EXPOSURE_LEDGER.md`
and is not reset by any decision taken here.

### 6.5 Calibration — decision `/07`

NR-05 used training → calibration/validation → test with Platt fit on the
calibration window only. It improved ULB ECE 0.03912 → 0.00033, and produced a
**reproducible inversion on IBM** (coef −0.7029, intercept −6.4186 under **185**
calibration positives; test ROC-AUC 0.627 → **0.373, below chance**) while Brier
(0.00102) and ECE (0.00045) **still looked excellent**. The inversion is stated
here in full and is not suppressed, explained away, or fitted away. The
Reviewer decides the authoritative calibration procedure, including whether a
calibrator-sanity guard and a calibration-positive floor are mandatory. The
standing rule: Brier/ECE may never be reported as evidence of calibration without
the ranking check (NR-05 §13 caveat, plan §4A.3 reporting rules).

### 6.6 Temporal validation — no standalone §20 issue; surfaced here

NR-05 §14 observed an **IBM temporal collapse**: train-period AUCs 0.83–1.00
falling to **0.58–0.64** on the final-20% window across a 1,020-day span. Kaggle
was stable across its 193.5-day test span; ULB's last quartile fell to 0.670 with
11 positives. **This is one dataset's exploratory finding and is not generalised
to all datasets.** The question put to the Reviewer is whether confirmatory
Track M requires a **mandatory temporal evaluation** and how per-window results
aggregate — not whether temporal collapse is expected.

### 6.7 Entity-disjoint evaluation — no standalone §20 issue; surfaced here

NR-05 §14 ran Kaggle entity-disjoint (20% of `cc_num` held out entirely): ROC-AUC
**0.9697**, PR-AUC 0.8559 on 240,831 held-out-user rows — **no collapse**. It is
`NOT APPLICABLE` on ULB (**no entity identifier exists** — recorded, not forced),
and IBM entity-disjoint was **not executed** because stride sampling makes
per-user histories partial (a scope limitation, not a null result). The Reviewer
decides whether entity-disjoint evaluation is primary, secondary, dataset-dependent,
or `NOT APPLICABLE` where no meaningful entity exists. **No entity definition is
forced onto a dataset that does not support one.**

### 6.8 Business / alert-volume metrics — decision `/12`

NR-05 §15 invented **no operational cost model**; it reported alert-volume measures
only (top-1% alerts, recall@top1%, top-5% recall) and explicitly labelled ULB
per-day rates as dataset artefacts. Defensible candidates the Reviewer may freeze
without an unsupported cost model: **recall at fixed alert volume**, **precision at
fixed alert volume**, **alert count**, and **analyst burden as a clearly labelled
proxy**. Value-caught or cost-based thresholds are permitted only as parameterized
curves until defensible loss figures exist (preregistration §7), and never as
realized financial benefit.

### 6.9 Dataset eligibility — see §3.2 and `/13`; not a statistical conclusion

IEEE-CIS and BAF remain **BLOCKED — not acquired**. Outstanding gates: licence and
terms verification (read before download), provenance, labelled-data availability,
temporal suitability against the frozen split, fraud-count sufficiency against the
§18 floor, feature semantics, and leakage assessment. All are recorded per dataset
in `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` §1–§8, with the cross-dataset
gates in manifest §8 — **none satisfied yet**. The exposure ledger's own independent
audit (plan §14) is itself `NOT ESTABLISHED`. **This memo does not declare any
dataset eligible.**

---

## 7. Freeze readiness matrix

All fourteen NR-05 §20 issues appear. No issue is omitted. "Blocks Freeze?" and
"Blocks Track M?" use the categories required by the review task: **blocks freeze**,
**blocks Track M**, **affects only a secondary analysis**, **informational**.

| Decision | Status | Blocks Freeze? | Blocks Track M? | Evidence Available | Reviewer Decision | Approval |
|---|---|---|---|---|---|---|
| `/01` Baseline identity | REQUIRES DECISION | **Blocks freeze** | **Blocks Track M** (RQ-M1 primary) | NR-05 §6, §7 — LR and XGB both reported; strongest single differs per dataset | — | PENDING REVIEW |
| `/02` Ensemble identity | REQUIRES DECISION | **Blocks freeze** | **Blocks Track M** (RQ-M1 primary) | NR-05 §6, §8 — mean fusion loses to best single on all 4 runs, CIs exclude 0 | — | PENDING REVIEW |
| `/03` Confident-decision definition | REQUIRES DECISION | **Blocks freeze** | Blocks Class B primary only | NR-05 §9 — `p≥0.8 or p≤0.2` was an NR-05 choice; §11–12 band-sensitive | — | PENDING REVIEW |
| `/04` Gating bounds (§5.5) | NOT ESTABLISHED + REQUIRES DECISION | **Blocks freeze** | Blocks Class B primary only | NR-05 §11–§12 — ULB recall 0.813→0.040; only numeric proposal is preregistration Criterion 3 (≥20% / ≤2pp, an unapproved assumption) | — | PENDING REVIEW |
| `/05` Feature-contract construction | REQUIRES DECISION | **Blocks freeze** | **Blocks Track M** (row gate = Class A+B) | NR-05 §9, §20 — q01/q99±2·IQR blocked 72/75 ULB frauds; IBM `chip_code` 70.6% OOV; 33,685 NaN amounts | — | PENDING REVIEW |
| `/06` Window-gate definition | NOT ESTABLISHED + REQUIRES DECISION | **Blocks freeze** | Blocks Class B primary only | NR-05 §9, §11 — over-fires on ULB (PSI 1.46, coverage 0); never fires on Kaggle Class B (0.083) | — | PENDING REVIEW |
| `/07` Calibration procedure (§6) | NOT ESTABLISHED + REQUIRES DECISION | **Blocks freeze** | **Blocks Track M** (RQ-M4 primary) | NR-05 §13 — IBM reproducible Platt inversion (coef −0.7029, 185 cal positives, test AUC 0.373 below chance); Brier/ECE looked fine | — | PENDING REVIEW |
| `/08` Power floors (§18) | REQUIRES DECISION | **Blocks freeze** | **Blocks Track M** | NR-05 §20 item 8 — ULB test 75/cal 57; IBM test 134/cal 185 | — | PENDING REVIEW |
| `/09` Dependence-aware method (§19) | REQUIRES DECISION | **Blocks freeze** | **Blocks Track M** | NR-05 §16 — IID, entity-clustered, temporal-block all exercised, all labelled METHOD NOT YET FROZEN; n=300 | — | PENDING REVIEW |
| `/10` Minimum seed / run rule | REQUIRES DECISION | **Blocks freeze** | **Blocks Track M** | NR-05 §16, §23 — single seed 42; preregistration §5.1 proposes 5 seeds (open marker) | — | PENDING REVIEW |
| `/11` Member operating points | REQUIRES DECISION | No (not an §35 cell) | No — secondary analysis (RQ-M2 ablation) | NR-05 §20 item 11 — shared ensemble-selected threshold used for members | — | PENDING REVIEW |
| `/12` Alert-volume semantics | REQUIRES DECISION | **Blocks freeze** (§35 item) | No — supporting RQ-M8 | NR-05 §15 — test-ranked top-k%; ULB per-day labelled dataset artefact | — | PENDING REVIEW |
| `/13` Large-dataset sampling | REQUIRES DECISION | **Blocks freeze** | **Blocks Track M** where it applies | NR-05 — IBM 24,386,899 rows / 2.35 GB loaded at stride-1/37 (exploratory) | — | PENDING REVIEW |
| `/14` Segment / small-n withholding | REQUIRES DECISION | No | No — secondary analysis (RQ-M9) | `metric_definitions.md` v1.0 (`<30` withhold, `<50` warn) vs unset §18 floor; NR-05 §14 — ULB q4 = 11 positives, segment q2 = 9 positives withheld | — | PENDING REVIEW |

**Summary of the current matrix:** 11 of 14 block the freeze; 8 of 14 block the
primary Track M criteria; 3 are secondary-analysis-only. Zero decisions are
resolved. The freeze cannot proceed until the eleven blocking rows carry a
recorded decision and approval.

---

## 8. If the Reviewer selects `REVISE`

Plan §24 permits `REVISE` **only** where a frozen revision condition explicitly
allows it, and only once. If the Reviewer selects `REVISE` for any decision in §5,
the following seven fields must be recorded. An incomplete `REVISE` is not a
`REVISE`.

1. **Exactly what needs revision** — the single named artefact, cell or procedure
   (e.g. "§5.5 minimum-reduction bound", not "the gating approach").
2. **Why** — the documented **defect** in the experiment or pipeline, established
   *independently of whether the result was favourable*.
3. **Whether it changes the scientific question** — if yes, the revision is
   classified **exploratory**, not confirmatory (plan §24.6).
4. **Whether it changes the primary endpoint** — if yes, the same consequence
   applies; the original result is preserved.
5. **Whether it affects prior exposure** — revisions never reset the exposure
   status of ULB, Kaggle or IBM (`docs/evaluation/DATASET_EXPOSURE_LEDGER.md`).
6. **Whether a new exploratory analysis is required** — state it and its scope.
7. **Whether the one-revision rule is triggered** — plan §24: **at most one
   revision cycle**; "repeated iteration until a favourable result appears is
   prohibited". A second `REVISE` is not available.

**Not eligible triggers (plan §24.1), stated here so they cannot be used:** a
null or negative primary result; a confidence interval containing zero; a failed
threshold; an unfavourable ablation; a wish to try another model or feature set.
Notably, **the NR-05 ensemble-inferiority result, the Class B `NOT ESTABLISHED`
result, and the IBM calibration inversion are not eligible REVISE triggers.**

Each revision record must be approved by someone **other than the experimenter**
(plan §24.1), and must preserve the original result (plan §25: original failed
experiments are never deleted, overwritten, silently rerun under changed
conditions, or relabelled).

---

## 9. Statistical reviewer sign-off (plan §32)

Plan §32 requires the statistical reviewer to explicitly approve the statistical
decision framework before freeze, and records: *"A single developer's
self-approval is not sufficient for the statistical sections."*

- **Statistical reviewer:** `NOT ASSIGNED`
  *(plan §28 `[STATISTICAL REVIEWER]`; preregistration §5 sign-off block lists
  "Statistical reviewer (external, R-42)" — `_pending_`. No identity is invented
  here.)*
- **Review date:** `NOT ESTABLISHED`
- **Decision:** `PENDING REVIEW`
- **Approval:** `NOT APPROVED`
- **Rationale:** `PENDING REVIEW`

**Sections reviewed (plan §32 sign-off scope, identical to the §28 reviewer
task):**

| # | Section | Subject | Decision record |
|---|---|---|---|
| 1 | Plan §3 | primary metric, dataset aggregation, multiplicity, ensemble and baseline identity | `/01`, `/02` |
| 2 | Plan §4A | performance measures and tiers; §4A.3 reporting rules | `/11` |
| 3 | Plan §5.0 | unit under test, dataset-specific contract construction | `/05`, `/06` |
| 4 | Plan §5.3 | confident-decision definition | `/03` |
| 5 | Plan §5.5 | gate thresholds and bounds | `/04` |
| 6 | Plan §6 | calibration | `/07` |
| 7 | Plan §18 | minimum-power rules | `/08` |
| 8 | Plan §19 | dependence-aware bootstrap | `/09` |
| 9 | Plan §20 | above-chance definition | (no §20 issue; applies to `/02`, `/09`) |
| 10 | Plan §21 | success criteria | `/02`, `/04`, `/07` |
| 11 | Plan §22 | decision outcomes | §4 of this memo |
| 12 | Plan §23 | go/no-go and roll-up, one- and two-dataset aggregation | `/08`, `/13` |
| 13 | Plan §24 | REVISE and INCONCLUSIVE rules | §8 of this memo |
| 14 | Plan §10 | business / alert-volume metrics | `/12` |
| 15 | Plan §11, §12 | segment definitions, point-in-time correctness | `/14` |
| 16 | Plan §28 | manual dependencies and reviewer roles | §3.2, §6.9 |
| 17 | Plan §32 | this sign-off itself | — |

**Domain review** (plan §32: *"Domain review must separately verify the semantic
appropriateness of the primary outcomes where required"*) is **NOT ESTABLISHED**
— §28 `[DOMAIN REVIEWER]` is unassigned. It is required for `/04` (the
coverage-loss tolerance encodes a cost-of-abstention assumption) and `/12` (any
operational reading of alert volume).

**Approval is recorded in the evidence ledger** (plan §32); it is not recorded in
this document alone.

---

## 10. What remains before the Research Plan can be frozen

This memo covers the **fourteen §20 issues**. They are **not** the whole of what
blocks a freeze. Plan §35 (Pre-Freeze Checklist) is the authoritative list and is
**not** restated here; the items below are only those outside §20 that a reviewer
working from this memo would otherwise have to discover elsewhere.

| Remaining item | Plan reference | Current state |
|---|---|---|
| The 11 `[REQUIRES DECISION/APPROVAL]` markers in the preregistration | preregistration §0–§5 | open; the protocol remains `0.1.2-draft`, "Finalized: NO" |
| Plan §23 one-dataset and two-dataset aggregation rules | plan §23 | `[TO BE FROZEN]` |
| Plan §23 primary-RQ candidate list (RQ-M1, RQ-M3 Class B, RQ-M4) | plan §23 | `[TO BE FROZEN — candidates: …]`, reviewer to confirm or replace |
| Plan §3 primary metric selection | plan §3, §4A.1–§4A.2 | `[TO BE FROZEN]` |
| Plan §3 multiplicity method | plan §3 | `[TO BE FROZEN]` |
| Plan §7 drift baselines and thresholds; §8/§9/§11/§12 split, segment, point-in-time and label-delay rules | plan §7–§12 | `[TO BE FROZEN]` |
| Plan §28 manual dependencies: IEEE-CIS terms, BAF terms/provenance, exposure audit, statistical review, domain review | plan §28 | all `PENDING`, owners unassigned |
| Plan §29 freeze record and plan §30 freeze-check pass | plan §29–§30 | `docs/FREEZE_RECORD.json` does not exist; check exits 1 by design |
| Independent audit of the exposure ledger (plan §14) | plan §14 | `NOT ESTABLISHED` |
| §34 artifact paths (preregistration, exposure ledger, eligibility manifest) | plan §34 | all three rows still read `[TO BE FROZEN — path]` even though the files now exist — the path decision itself remains the reviewer's |
| Domain reviewer sign-off | plan §32 | `NOT ESTABLISHED` |

**Nothing in this memo changes any of these states.**

---

## 11. Explicit non-claims of this memo

This document does not claim, and must not be cited for:

- that the Research Plan is frozen, approved, or ready to freeze;
- that any §20 decision has been made, or that any placeholder has been resolved;
- that IEEE-CIS or BAF is eligible, acquired, or verified (`BLOCKED`);
- that any NR-05 result is confirmatory, or constitutes independent replication;
- that Class B gating benefit is established — it is `NOT ESTABLISHED`;
- that the ensemble is superior — NR-05 found the opposite on exposed data;
- that calibration is validated — Brier/ECE masked a reproducible inversion;
- that a statistical reviewer or domain reviewer has been appointed, or has signed;
- that a machine-readable decision record exists (none was created — see below).

**No machine-readable companion was created.** The repository's established
schema (`eval_record.py` v1.1 append-only ledger) records *executed experiments*,
not pending reviewer decisions; writing placeholder decisions into that ledger
would corrupt an append-only evidence trail. The Markdown memo is the single
reviewer-facing record.

---

## 12. Change discipline for this memo

- This memo records decisions; it never makes them. Amendment happens by adding a
  dated note to the relevant decision record, never by deleting an earlier entry
  (plan §25: original records are never deleted, overwritten or relabelled).
- After a reviewer decision is recorded, the corresponding plan/protocol cell is
  resolved **in the plan or protocol** — not here. This memo points at those
  cells; it does not replace them.
- Invariant: the Research Plan, the preregistration, NR-05, the exposure ledger
  and the eligibility manifest are unchanged by this memo (hashes in §1.2).

| Version | Change | Type |
|---|---|---|
| 1.0-draft | Initial memo. 14 decision records from NR-05 §20, freeze readiness matrix, plan §32 sign-off block. No decisions recorded. | PREPARATION DOCUMENT |
