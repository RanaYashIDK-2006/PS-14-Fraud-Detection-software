# Pre-Registered Evaluation Protocol — DRAFT, NOT APPROVED

**Status:** DRAFT for review. This document is **NOT finalized**. Sections marked
`[REQUIRES DECISION/APPROVAL]` must be resolved by reviewers before any experiment
under this protocol is executed. No experiment may claim pre-registration benefits
until this document is versioned, dated, and signed off (NR-02 sign-off; superseded
roadmap phase R-05).

**Created:** Phase 104 (2026-10-03), `docs/evaluation/`.
**Companion:** `docs/PHASE_104_FORENSIC_AUDIT.md`, `docs/PHASE_104_GAP_MATRIX.md`.

**Hard rule:** no threshold, hyperparameter, or model variant may be selected after
viewing final-test results. Final test data is touched exactly once per experiment,
after the threshold is frozen on validation data.

---

## 0. Scope and model identity

`[REQUIRES DECISION/APPROVAL]` — confirm which system is under evaluation:

- **Candidate S1:** deployed altman_native 48-feature ensemble (xgb+lgb+cb,
  Platt-calibrated, locked threshold 0.018758, release
  `legacy-altman_native_E_hardneg_cert_20260904`).
- **Candidate S2:** whichever contract OPTION (A public_v1-14 / B deferred) is chosen
  at sign-off (see audit Part C.4; note: the binary A/B framing was superseded by the
  evidence-based three-contract determination in `docs/PHASE_106_ROADMAP_RECONCILIATION.md`
  §E — 48-native evaluable on Kaggle, 21/public_v1 on IBM/public, 21+7 BLOCKED on bank
  data — the scope selection remains THIS marker's human decision). **The option choice
  is a human decision; this protocol does not choose it.**

Everything below applies to the selected candidate(s); recording which is part of
the sign-off.

---

## 1. Evaluation data and splits

| Split | Definition | Rule |
|---|---|---|
| Train | model fitting | — |
| Validation | threshold selection, calibration fitting, hyperparameters | may be inspected freely |
| **Final test** | the numbers reported in claims | **frozen threshold, single use, never for selection** |

- Candidate datasets: IBM v2 (user-disjoint), ULB (time-ordered), Kaggle fraud
  (external transfer), BAF (blocked until NR-16 acquisition + license review;
  superseded roadmap phase R-20).
- `[REQUIRES DECISION/APPROVAL]` — choose per dataset: temporal split only, or
  entity-disjoint only, or both axes reported separately. **Assumption to justify:**
  that the chosen split matches the deployment claim being made (production is a
  stream over time across many entities; audit note: `ibm_train.py` today is
  user-disjoint but NOT chronological).
- Every run must record: dataset hash, git SHA, seed, split definition, command,
  and write an `eval_ledger.jsonl` record (schema per `eval_record_test.py`, 22/22).

---

## 2. Metrics

Primary: **PR-AUC**, **recall@1%FPR**, **ROC-AUC**.
Secondary: Brier, ECE (10–15 equal-mass bins `[REQUIRES DECISION/APPROVAL]` — bin
count fixed before running), reliability diagram, precision/recall/F1 at the frozen
threshold, alert rate, coverage (for gated systems).

- Brier/ECE must first be added to `metric_definitions.py` (NR-03; superseded roadmap
  R-02 — still not done: file remains v1.0 with no Brier/ECE definitions); until then
  results using them are labeled *provisional*.
- *Implementation note (factual, not a decision):* the only existing ECE code
  (`backend/scripts/calibration_test.py`) computes **10 equal-width bins** today; the
  binning choice above remains an open decision marker (retained, not resolved).
- Confidence intervals: stratified bootstrap, 2000 replicates, fixed seed
  `[REQUIRES DECISION/APPROVAL]` — confirm replicates and seed.
- Multiple comparisons: `[REQUIRES DECISION/APPROVAL]` — if ≥3 model variants are
  compared on one test set, confirm whether Holm–Bonferroni or false-discovery-rate
  correction applies.

---

## 3. Pre-registered decision criteria

Each criterion states: metric, comparison, dataset/split, CI procedure, minimum
meaningful difference (MMD), replication, and consequence. **Numerical MMDs below are
proposals with justification, not settled facts** — all are `[REQUIRES DECISION/APPROVAL]`.

### Criterion 1 — "The ensemble helps"
- **Metric:** PR-AUC (primary) and recall@1%FPR (secondary) on final test.
- **Comparison:** pre-registered ensemble vs its **best single constituent** chosen on
  validation only (expected: LightGBM or XGB — the choice must be made on validation
  before test is opened).
- **MMD (proposal):** ensemble must exceed best single model by a relative ≥2% in
  PR-AUC **and** the 95% paired-bootstrap CI of the difference must exclude 0.
  *Justification:* absolute differences below ~1–2 points on a 0.8-scale metric are
  within seed/split noise for tabular fraud models — **this magnitude is an assumption
  that must be justified or revised before testing.**
- **Replication:** same comparison reproduced on a second dataset/split (NR-17;
  superseded R-21) to count
  as *demonstrated*; one dataset = *provisional*.
- **Met →** claim "ensemble advantage under temporal/independent evaluation"
  (Conclusion A candidate). **Not met →** no ensemble-superiority claim; project moves
  toward Conclusion B. **CI straddles 0 / below MMD →** Inconclusive for this criterion.

### Criterion 2 — "The ensemble does not help"
- Statistically: the 95% CI of (ensemble − best single) **includes 0** or favors the
  single model, on both primary metrics.
- **Met →** ensemble-superiority claims are withdrawn; methodology/gating/provenance
  framed as the contribution (Conclusion B candidate). Reported regardless of outcome.

### Criterion 3 — "The gate helps"
- **Metric:** selective risk (error rate among non-abstained decisions) at matched
  **coverage**, plus rate of *wrong confident decisions* (score ≥ threshold-band top
  decile while wrong), false positives, false negatives.
- **Comparison:** GATED vs UNGATED under identical models across the 7 pre-defined
  conditions (clean; missing features; invalid features; drift; corruption;
  adversarially unusual-valid; combinations) — NR-14 (superseded R-18).
- **CI:** paired bootstrap over test events.
- **MMD (proposal):** gated reduces wrong-confident decisions by ≥20% relative at no
  more than 2-percentage-point coverage loss, CI excluding 0.
  *Assumption to justify: the 20%/2pp trade-off reflects an operational risk posture
  — requires a domain assumption about the cost of confident errors vs abstentions.*
- **Met →** gating-effectiveness claim permitted (implementation-only claims stay
  forbidden). **Not met →** gating described purely as a safety mechanism, no
  effectiveness claim. **Inconclusive →** report coverage–risk curves without a claim.

### Criterion 4 — "The gate does not help"
- CI of the wrong-confident-decision reduction includes 0, or coverage loss exceeds
  the MMD while risk does not improve, across conditions.
- **Met →** gating effectiveness is *not demonstrated*; keep gating only if its
  fail-open availability cost is justified independently; preserve the negative result.

### Criterion 5 — "The features carry transferable signal"
- **Metric:** ROC-AUC and PR-AUC of the pre-registered feature set on an
  **independent dataset** (not same-generator as training), temporal + entity-safe split.
- **MMD (proposal):** external ROC-AUC ≥0.70 **and** ≥80% of the in-domain
  performance retained, CI excluding chance (0.5). *Justification pending review —
  0.70 is a conventional "usable discrimination" floor, not a domain-derived figure.*
- **Replication:** two independent datasets required for the full claim; one gives
  *provisional* transfer evidence.
- **Met →** transfer claim permitted. **Not met (observed external evidence to date:
  artifact-backed 0.595; the 0.435 intermediate is NOT ESTABLISHED — prose-only per
  Phase 105, registry C-011…C-014) →** transfer claim
  prohibited; Conclusion B or C candidates. **Inconclusive →** more data, no claim.

### Criterion 6 — "The features do not demonstrate transferable signal"
- External CI excludes the MMD on both attempted datasets, with admission criteria
  (provenance, semantics) satisfied — i.e. the datasets *could* have shown signal.
- **Met →** negative result recorded as evidence; original model-claim language is
  retired; project proceeds under Conclusion B or C. **No additional complexity may
  be added solely to overturn this result.**

### Criterion 7 — "Results are inconclusive"
- Triggered when: CIs overlap both success and failure regions; admission criteria
  failed (dataset cannot test the feature space — see Conclusion C); or required
  replication could not be run (blocked data).
- **Consequence:** say so explicitly; do not round toward success; define the single
  next experiment that would resolve it, or conclude C.

---

## 4. Negative-result plan (Part H)

Negative results are **preserved as evidence** (ledger records + reports), not
discarded or re-run until positive. The project must NOT keep adding complexity to
obtain a positive result.

Final conclusions (exactly one at freeze):

- **A. Demonstrated advantage** under appropriate temporal/independent evaluation
  (criteria 1, 3, 5 met with replication).
- **B. No demonstrated ensemble advantage**; methodology/gating/provenance is the
  stronger contribution (criteria 2 and/or 4, 6 — with honest engineering evidence).
- **C. Inconclusive because available public data cannot test the intended feature
  space** (criterion 7 with failed admission — e.g. the 7 bank features absent from
  every public source, audit Part C.3/D).

`[REQUIRES DECISION/APPROVAL]` — confirm A/B/C wording and that publication terms
(bank deliverable O5) commit to publishing unfavourable outcomes (B/C).

---

## 5. Threshold discipline (training → validation → frozen → test)

1. Train on train split (seeds pre-declared: `[REQUIRES DECISION/APPROVAL]` —
   how many seeds, which values; proposal: 5 fixed seeds).
2. Fit calibration and choose threshold **on validation only** (operating point:
   fixed FPR `[REQUIRES DECISION/APPROVAL]` — 1%? — or fixed alert volume).
3. Freeze: write threshold + model hash + git SHA to the eval record.
4. Evaluate final test **once**; no re-run with different thresholds; any later
   re-run is labeled exploratory, not confirmatory.
5. Deployed threshold 0.018758 is already LOCKED for S1 (phase 93 protocol); this
   protocol must not silently re-tune it — `[REQUIRES DECISION/APPROVAL]` whether
   S1's locked threshold is carried into external evaluation untouched (recommended).

---

## 6. Segment / subgroup analysis plan (Part I)

Segments available on candidate public data (do not invent others):
amount bands (pre-deciled on validation), time periods (hour/day), merchant/city
category where present (IBM), entity tenure proxies where present, feature-availability
groups (14-feature public_v1 vs 48-native), dataset of origin for transfer analyses.
Geography beyond city and fraud-type taxonomies are **not available** — excluded.

Report per segment: PR-AUC, recall@1%FPR, alert rate, with bootstrap CIs; flag
segments where CI is uninformative (small n) rather than over-reading them.

---

## 7. Business metrics plan

- Recall at fixed alert volume (e.g. top-k% of scores) — the primary operational view.
- Alerts per analyst per day **given an assumed analyst capacity** `[REQUIRES DECISION/
  APPROVAL] — capacity number is a domain assumption; without bank input, use
  clearly-labeled illustrative capacities only.
- False-positive burden (non-fraud alerts / total alerts).
- Value-caught and cost-based thresholds: **only** with defensible loss/cost figures;
  until bank data exists these are reported as parameterized curves
  (cost ratio on x-axis), **never as realized financial benefit**.

---

## 8. What this protocol does NOT claim

- No regulatory or institutional compliance.
- No external validation has occurred.
- No numerical threshold in §3 is approved until the sign-off below exists.

---

## 9. Final-test freeze (added NR-03 — semantic clarification; no methodology changed)

1. **What is locked:** the designated final-test partition of every dataset named in
   an experiment (identified by file sha256 + split definition + boundaries), the
   external-transfer set (`data/kaggle_fraud/fraudTest.csv`, sha256
   `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0`), all
   model/constituent weights, preprocessing parameters, calibration parameters,
   thresholds, seeds, metric definitions, and MMD decision rules — as recorded in the
   frozen-test artifact and the freeze record appended to `eval_ledger.jsonl`.
2. **When:** the instant the frozen-test artifact and freeze record exist — before any
   final-test execution. Structure may be inspected beforehand; no outcome may be
   computed before the freeze.
3. **Allowed before locking:** all development on train; calibration, threshold,
   constituent and hyperparameter selection on validation only; protocol edits via the
   Change history.
4. **Forbidden after locking:** model/hyperparameter/variant/seed/threshold/
   preprocessing/calibration changes informed by final-test or external-set outcomes;
   selective repetition to improve a reported outcome; post-hoc seed or run exclusion;
   metric, endpoint, segment, or MMD changes; using external-transfer results for any
   model or protocol selection.
5. **Protocol violation:** any action in (4), or executing a final test before the
   freeze exists. Consequence: the affected results lose confirmatory status and are
   reported as exploratory; the violation is recorded in `evaluation_config.deviations`
   and disclosed in the Gate-2 report.
6. **Failed runs:** recorded as `status: FAILED` records (append-only, preserved) and
   reported as FAILED in the full matrix.
7. **Evidence-generation failure:** the run cannot support claims. An evidence-only
   rerun is permitted with the identical frozen configuration (same seed/command/
   config); both attempts are recorded. If evidence still cannot be produced, the cell
   is NOT ESTABLISHED.
8. **Reruns:** permitted only for implementation/evidence failures under the identical
   frozen configuration — never to change an outcome; every attempt is recorded.
9. **Seed exclusion:** never permitted post-hoc. Every pre-declared seed appears in the
   matrix (MEASURED / FAILED / …); a crashed seed may be rerun only with the identical
   seed and only for an implementation failure.
10. **Deviations:** only a dated amendment in the Change history, made BEFORE the
    affected decision, can authorize a deviation; after outcomes exist, no
    authorization converts the result to confirmatory status.
11. **Reporting:** the full matrix (experiment × dataset × seed × contract × condition
    × transfer direction) is reported with cell statuses MEASURED / NOT ESTABLISHED /
    BLOCKED / FAILED / NOT APPLICABLE — no omitted, zero-filled, or selectively
    presented cells.

**Data tiers:** development (train) → validation (calibration / threshold / constituent
selection only) → final test (locked, single use) → final external
(`fraudTest.csv`: evaluation only; must never influence model, threshold, or protocol
selection). Historical prior exposure of any dataset (§G of the NR-02/NR-03 reports)
is a recorded limitation, not a reset.

---

## Sign-off block (to be completed before experiments)

| Role | Name | Date | Scope of approval |
|---|---|---|---|
| Lead researcher | _pending_ | — | criteria 1–7, MMDs |
| Statistical reviewer (external, R-42) | _pending_ | — | CI/procedure/multiplicity |
| Domain reviewer (bank SME or equivalent) | _pending_ | — | business metrics, assumptions |
| Data-governance/privacy (if bank data) | _pending_ | — | dataset admission |

**Version:** 0.1-draft — Phase 104. **Amended:** 0.1.1-draft (NR-02, factual
corrections) → **0.1.2-draft** (NR-03, semantic clarifications + §9 final-test freeze;
see Change history); no substantive methodology changed; all 11 markers preserved.
**Finalized:** NO.

---

## Change history

| Version | Phase | Change | Type |
|---|---|---|---|
| 0.1.1-draft | NR-02 | Sign-off cross-reference R-05 → NR-02 (roadmap rebuilt in Phase 106) | factual/documentary |
| 0.1.1-draft | NR-02 | §0 OPTION decision reference R-03 → Phase 106 §E three-contract determination (decision marker preserved) | factual/documentary |
| 0.1.1-draft | NR-02 | BAF cross-reference R-20 → NR-16 | factual/documentary |
| 0.1.1-draft | NR-02 | `metric_definitions` cross-reference R-02 → NR-03 (+ factual note that it is still v1.0) | factual/documentary |
| 0.1.1-draft | NR-02 | Cross-references R-21 → NR-17, R-18 → NR-14 | factual/documentary |
| 0.1.1-draft | NR-02 | Criterion 5 failure branch: replaced the prose-only 0.435 with the artifact-backed 0.595 evidence basis (Phase 105: C-011/C-012 NOT ESTABLISHED, C-013/C-014 DEMONSTRATED); MMD ≥0.70 and consequences unchanged | factual (evidence citation), rule unchanged |
| 0.1.1-draft | NR-02 | §2 added factual implementation note: current ECE code is 10 equal-width bins; binning decision marker retained | factual (implementation state), decision retained |
| 0.1.2-draft | NR-03 | Added §9 final-test freeze: lock definition, violation/rerun/deviation rules, four data tiers, full-matrix reporting statuses | SEMANTIC CLARIFICATION |
| 0.1.2-draft | NR-03 | Brier/ECE remain **provisional**; the exact governing implementations are documented in `docs/PHASE_NR03_EVALUATION_SEMANTICS.md` (Option B — no metric upgraded, no definition changed) | FACTUAL/DOCUMENTARY |
| 0.1.2-draft | NR-03 | Canonical pipelines populate the already-required `preprocessing_version` / `feature_schema_version` evidence fields with the vocabulary frozen in the NR-03 report (fields and defaults unchanged for legacy records) | IMPLEMENTATION CORRECTION |

No substantive methodology, MMD, criterion, or decision marker was added, removed, or
changed by NR-02.
