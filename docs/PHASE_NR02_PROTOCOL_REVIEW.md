# NR-02 — Preregistered Evaluation Protocol Review & Sign-Off Readiness

**Phase:** NR-02 (Gate 1) · **Date:** 2026-10-03
**Scope:** protocol review only. No experiment was executed; no model, dataset,
feature, threshold, or methodology was changed. Factual/documentary corrections were
applied to the protocol with a change history; all 11 decision/approval markers were
preserved; no reviewer approval is claimed anywhere in this report.

---

## A. Objective

Determine whether `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` is precise,
internally consistent, reproducible, and reviewable for the preregistered comparative
experiments — and package everything a qualified external reviewer needs. Approve
nothing; record what must be reviewed.

---

## B. Protocol Baseline

| Item | Value (after this phase) |
|---|---|
| Status | **DRAFT, NOT APPROVED** (unchanged) |
| Version | 0.1-draft (Phase 104) → **0.1.1-draft** (NR-02, factual amendments only) |
| Unresolved markers | **11** `[REQUIRES DECISION/APPROVAL]` (verified by count before and after NR-02 — none added, none removed) |
| Sign-off | 4-role block, all `_pending_` — **no reviewer has reviewed or approved** |
| Datasets named | IBM v2 (user-disjoint), ULB (time-ordered), Kaggle fraud (external transfer), BAF (blocked) |
| Splits | Train / Validation (inspectable) / Final test (frozen, single-use); per-dataset axis choice (temporal vs entity-disjoint vs both) = open marker |
| Metrics | Primary: PR-AUC, recall@1%FPR, ROC-AUC. Secondary: Brier, ECE (10–15 equal-mass bins — open marker), reliability diagram, P/R/F1 at frozen threshold, alert rate, coverage |
| Statistical procedures | Stratified bootstrap CI (proposal: 2000 replicates, fixed seed — open marker); multiplicity Holm–Bonferroni vs FDR (open marker); paired-bootstrap CI of differences for criteria 1/3 |
| Decision criteria | 7 criteria (ensemble helps / does not; gate helps / does not; transfer / does not; inconclusive) with proposed MMDs, each marked |
| Exclusion criteria | Segments unavailable on public data excluded (§6); best-single-constituent chosen on validation only; no post-test selection (hard rule) |
| Stopping rules | None needed (single confirmatory run per experiment); later re-runs labeled exploratory (§5.4) |
| Negative-result handling | §4 explicit: preserved as evidence, no complexity to overturn, conclusions A/B/C |
| Reproducibility requirements | §1: dataset hash, git SHA, seed, split definition, command, ledger record (schema `eval_record_test.py` 22/22 — still accurate) |
| Sign-off requirements | Versioned, dated, signed by 4 roles before any experiment claims preregistration benefits |

### Classification of protocol content

- **Already specified:** hard rule (no post-test threshold/hyperparameter/variant
  selection; test touched once); split roles; metric hierarchy; criteria 1–7 with
  consequence mapping; negative-result plan; threshold-discipline order (§5);
  segment list; business-metric constraints (never realized financial benefit);
  required record fields; sign-off structure.
- **Underspecified:** preprocessing order; missing/invalid-row/outlier/duplicate
  treatment; class-imbalance handling; hyperparameter-change policy beyond the hard
  rule; retraining policy; statistical treatment of failed runs / failed experiments;
  how 5-seed results are aggregated; **which dataset criteria 1–2 run on** (§D item D-1).
- **Contradictory:** **none found** — no two sections prescribe incompatible behavior
  (this is why the status is not NOT READY).
- **Implementation-dependent:** ECE binning (see §C); bootstrap replicate count;
  ULB split description vs current implementation; metric-definitions v1.0 gap
  (protocol already anticipates it — "provisional" label).
- **External-review dependent:** all 11 markers — MMD magnitudes, 0.70 floor,
  20%/2pp gating trade-off, seed values, ECE binning, split axes, multiplicity,
  A/B/C wording + publication terms, S1 threshold carry, analyst capacity.

---

## C. Repository Reconciliation

| Protocol Item | Current Implementation | Match | Issue |
|---|---|---|---|
| Dataset loading | Per-script fixed paths (`data/creditcard.csv`, `credit_card_transactions-ibm_v2.csv`, `data/kaggle_fraud/*.csv`) — no shared loader | PARTIAL | No single canonical loader; harness (NR-04) must standardize |
| Dataset identity/versioning | `eval_record.dataset_fingerprint()` = SHA-256 over raw bytes; required for DEMONSTRATED claims (NR-01 enforcement) | MATCH | — |
| Feature preparation | `eval_ulb` (PCA research features), `train_compare` (48-native), `public_feature_contract.json` (public_v1) | PARTIAL | Contract file declares 14 features, lists 13 (N-02 → NR-04); no per-run contract field |
| Feature availability | 7 bank features absent from every public dataset (phase64); phase108: 5/48 usable on ULB | BLOCKED | 21+7 contract experiments cannot run; three-track scope decision (§0 marker) |
| Train/validation/test splitting | `train_compare` time_70_15_15 (temporal); `eval_ulb` random stratified 80/20 seed 42 (historical reproduction — no separate val); `ibm_train` user-disjoint NOT chronological (protocol's audit note is accurate) | PARTIAL | Per-dataset axis choice is the §1 marker; current ULB reproduction is not the protocol experiment |
| Preprocessing | Scaler fitted per script; `preprocessing_version`/`feature_schema_version` record fields exist but pipelines leave defaults ("1.0") | PARTIAL | Protocol silent on order/freeze; record fields unpopulated — see §J |
| Calibration | Platt fit on validation only (`calibration_test`, `train_compare`); artifact-bound | MATCH | Protocol §5.2 satisfied by design |
| Model training | Defaults per script; protocol forbids post-test hyperparameter/variant selection | PARTIAL | No pre-registered hyperparameter freeze artifact exists (harness scope, NR-04) |
| Prediction | Batched `predict_many` / per-script scoring | MATCH | — |
| Threshold selection | Validation-only + `refuse_test_tuning` guard (`evaluate.py:65`, tested by `eval_record_test` 22/22); `threshold_source` recorded; S1 threshold 0.018758 locked | PARTIAL | Frozen-test design artifact (machine-checkable single-use) not built → NR-03 |
| Metric calculation | `metric_definitions.py` **v1.0, zero Brier/ECE definitions** (re-verified) | NOT IMPLEMENTED | Protocol already mandates "provisional" labels until NR-03 delivers v1.1 |
| Cross-dataset evaluation | `cross_dataset_eval.py` — ledgered, warns "rank-preserving proxy mapping, not feature-semantic evaluation"; IBM same-generator caveat recorded | PARTIAL | Not a preregistered run; supports (does not satisfy) criteria 5/6 |
| External-transfer evaluation | phase24/25 artifacts exist (0.595 artifact-backed; 0.435 prose-only); not a preregistered execution | NOT IMPLEMENTED | Criterion 5/6 experiments pending NR-05/06/17 |
| Confidence intervals / tests | `bootstrap_ci`: percentile, class-stratified, seed 42, **default 1000 replicates**; **no paired CI-of-difference implementation** | PARTIAL / NOT IMPLEMENTED | Protocol proposes 2000 (open marker — implementation default differs); paired-difference CI needed by criteria 1/3 must be built in NR-04/07 |
| Random seeds | `seed` field or `seed_not_applicable`; both required for DEMONSTRATED | PARTIAL | Which/how many seeds = §5 marker (proposal: 5 fixed) |
| Artifact recording | `record_from_run`/`create_evaluation_record` → model hashes, artifact hashes, v1.1 `command` | MATCH | — |
| Evidence recording | Append-only ledger + `record_*.json` + claims registry + CI-enforced check (31-case self-test, live PASS) | MATCH | NR-01 verified |
| Threshold freeze write (§5.3) | Record carries `threshold`, `threshold_source`, `model_hash`, `git_commit` | MATCH | — |

**Implementation discrepancies surfaced (not altered):** (a) ECE = 10 equal-width bins
in code vs "10–15 equal-mass" proposal — factual note added to protocol §2, decision
marker retained; (b) bootstrap default 1000 vs proposal 2000 — reviewer confirms and
the harness must pass the confirmed value; (c) no paired-difference CI exists yet.

---

## D. Ambiguity Audit

| # | Ambiguity | Status |
|---|---|---|
| D-1 | **Criteria 1–2 do not name their dataset** (protocol lists candidate datasets but the ensemble comparison says only "final test") | Marked for reviewer/owner decision at sign-off — choosing it now would be a substantive scope decision |
| D-2 | Exact dataset version/hash per experiment | Resolvable at run time by the mandated hash record; dataset *choice* per experiment = §1 marker |
| D-3 | Split axis (temporal vs entity vs both) | Existing §1 marker — reviewer decision |
| D-4 | Random seed values / count / repeated-seed policy | Existing §5 marker (proposal: 5 seeds) — reviewer decision |
| D-5 | Seed aggregation method (mean±range? per-seed reporting?) | Reviewer decision (not currently marked — recorded here as a sign-off item, not silently added to the protocol) |
| D-6 | Preprocessing order and freeze | Reviewer decision; minimum rule suggested for review: preprocessing frozen with the model freeze record |
| D-7 | Missing values / invalid rows / outliers / duplicate records | Reviewer decision; note: dedup/leakage covered operationally by NR-10 audit, not by a protocol rule |
| D-8 | Treatment of unavailable production features | Substantially specified: three-track determination (Phase 106 §E) + §0 marker selects scope; per-run feature-availability reporting = §6 segment list includes feature-availability groups |
| D-9 | Calibration procedure | Specified: Platt on validation (matches code); binning = §2 marker |
| D-10 | Threshold-selection procedure | Specified: validation → freeze → single-use test; operating point (1% FPR vs alert volume) = §5 marker |
| D-11 | Primary/secondary metrics | Specified (§2); ECE semantics = marker |
| D-12 | CI method / significance / multiplicity | Partially specified (stratified bootstrap, proposals); Holm vs FDR = marker; paired-difference CI unimplemented |
| D-13 | Failed runs / failed experiments | **Not specified in protocol** (implementation preserves FAILED records append-only). Statistical treatment (exclude? count as failure?) = reviewer decision — recorded for sign-off |
| D-14 | Paired vs unpaired comparison | Specified for criteria 1/3 (paired bootstrap over test events); silent elsewhere — reviewer confirms |
| D-15 | Hyperparameter changes / retraining | Hard rule covers post-test selection; pre-test retraining policy not stated — reviewer confirms |
| D-16 | Class imbalance handling | Not stated in protocol; implementation uses defaults (e.g. `scale_pos_weight` noted in reports) — reviewer confirms whether this must be frozen pre-run |

No ambiguity was resolved by assumption. None was chosen because it would look better.

---

## E. Result-Dependent Decision Audit

| Flexibility | Prevented? | Mechanism |
|---|---|---|
| Change threshold after seeing results | **YES** | Hard rule + `refuse_test_tuning` guard + §5 freeze; frozen-test artifact pending NR-03 (residual) |
| Select favorable split | **PARTIALLY** | §1 marker must be resolved *before* runs; nothing yet enforces pre-declaration timing — sign-off ordering must fix it |
| Select favorable seeds | **PARTIALLY** | §5 marker pre-declares seeds; enforcement = harness (NR-04) |
| Exclude inconvenient runs | **YES** | Append-only ledger preserves FAILED/unfavorable records (NR-01 §I); §4 forbids re-running until positive |
| Change metrics after outcomes | **YES** | §2 fixed hierarchy; ECE binning must be fixed pre-run (marker wording already says "bin count fixed before running") |
| Change primary endpoint | **YES** | §2 primary set fixed |
| Remove negative results | **YES** | §4 + publication-terms marker commits to unfavourable outcomes |
| Selectively report successful models | **PARTIALLY** | Criterion 1 chooses best-single on *validation only*; full-matrix reporting not explicitly required — reviewer item |
| Alter preprocessing after transfer results | **NOT EXPLICIT** | Minimum clarification proposed for review (D-6): preprocessing frozen at model freeze |
| Tune against external evaluation set | **PARTIALLY** | §5.4 labels later re-runs exploratory; explicit "external set is never used for selection" sentence absent — proposed for sign-off |
| Choose statistical test after seeing significance | **PARTIALLY** | Multiplicity method is marked, but test *choice* isn't pre-fixed beyond the CI procedure — reviewer item |
| Redefine failure after results | **YES** | MMDs are pre-registered proposals; all marked, none settled |
| Favorable segment selection | **YES** | §6 fixes the segment list ("do not invent others") |

Net: the protocol already blocks the largest post-hoc channels; the residual gaps
(preprocessing freeze, external-set-as-selection, failed-run treatment, full-matrix
reporting) are recorded as minimum clarifications **for reviewer sign-off** — none was
silently inserted.

---

## F. Negative-Result Policy

Checked against the ten required scenarios:

| Scenario | Protocol handling |
|---|---|
| Model below baseline | Criterion 2/6 → Conclusion B; explicitly reported regardless of outcome |
| ROC-AUC < 0.5 | Covered by criteria 5/6 "not met" + §4 preservation (below-chance is itself informative) |
| PR-AUC poor | Same (PR-AUC is primary in criterion 1; "not met" → B) |
| Transfer degrades | Criteria 5/6 — 0.595 artifact-backed reference corrected this phase |
| Calibration worsens | **Not addressed by any criterion** — residual gap documented (secondary metric; no claim depends on it alone). No new criterion invented |
| Feature contract unsatisfiable | Criterion 7 + Conclusion C (7 bank features — explicitly named) |
| Dataset cannot be obtained | Criterion 7 ("required replication could not be run — blocked data") |
| Pipeline fails | Not stated in protocol; implementation preserves FAILED records (schema status) — recorded as sign-off item |
| Statistical test inconclusive | Criterion 7 — "say so explicitly; do not round toward success" |
| CIs overlap | Criterion 7 |
| Fails a preregistered acceptance criterion | Criteria 2/4/6 consequences + frozen MMDs |

**Verdict:** the protocol never requires a positive finding; Conclusions A/B/C are all
valid terminal states and publication terms must cover B/C (marker). Negative results
remain valid results. The two residual gaps (calibration-worsening, pipeline-failure
wording) are documented, not papered over.

---

## G. Dataset Eligibility

| Dataset | Identity / location | Role | Status |
|---|---|---|---|
| ULB credit card fraud | `data/creditcard.csv` (150,828,752 B; sha256 `76274b69…` pinned in ledger `…e58d68c2bf88`) | In-domain benchmark | **AVAILABLE** — executed, hash-verified. License terms: NOT ESTABLISHED (no license file in repo) |
| IBM v2 | `data/credit_card_transactions-ibm_v2.csv` (2.35 GB) | Entity-disjoint / same-generator transfer | **AVAILABLE** — executed. License: NOT ESTABLISHED. Caveat: same generator family as synthetic training data (not independent) |
| Kaggle fraud | `data/kaggle_fraud/fraudTrain.csv` + `fraudTest.csv` (both present) | External transfer (the only non-same-generator family used) | **AVAILABLE** — used by phase24/25 and `cross_dataset_eval`. License: NOT ESTABLISHED (Kaggle terms not recorded) |
| BAF (NeurIPS 2022) | `data/external_benchmark/` **does not exist** | Second independent dataset (NR-17) | **BLOCKED** — download approval + license/semantics review (NR-16) |
| IEEE-CIS | not present | Conditional alternate | **BLOCKED** — credentials (only if pursued, G-28) |
| Bank institution data | not present | Full 21+7 contract evaluation | **BLOCKED** — institution-controlled environment (7 features absent everywhere public) |
| — cross-domain 4-UCI files referenced by README tables | not all verified present | Not named by the protocol | UNSUITABLE FOR THIS REVIEW (out of protocol scope; README FN-4, NR-01) |

No dataset was downloaded, substituted, or fabricated this phase. Any substitution
would be a protocol change requiring review.

---

## H. Statistical Review — `STATISTICAL REVIEW REQUIRED`

Per-comparison design (as far as the protocol specifies it):

| Comparison | Estimand | Unit | Population | Primary metric | Uncertainty | Test/H₀ | Status |
|---|---|---|---|---|---|---|---|
| C1/C2 Ensemble vs best single | ΔPR-AUC (ensemble − best-single) | transaction/event | named dataset's frozen test window (**dataset unnamed — D-1**) | PR-AUC (recall@1%FPR secondary) | 95% paired bootstrap of Δ | H₀: Δ ≤ MMD (relative ≥2%) | STATISTICAL REVIEW REQUIRED (MMD, dataset, seed aggregation) |
| C3/C4 Gated vs ungated | Δ wrong-confident rate at matched coverage | event | 7 fixed input conditions | wrong-confident reduction (selective risk) | 95% paired bootstrap | H₀: Δ ≤ 20% rel. at ≤2pp coverage loss | STATISTICAL REVIEW REQUIRED (domain-derived trade-off) |
| C5/C6 Transfer | External ROC-AUC (and retention ratio) | transaction | independent dataset, temporal + entity-safe split | ROC-AUC | CI excluding 0.5; ≥0.70 floor; ≥80% retention | H₀: AUC ≤ 0.70-or-chance per MMD text | STATISTICAL REVIEW REQUIRED (0.70 conventional, not domain-derived — protocol says so) |
| C7 Inconclusive | — | — | — | — | CI straddles both regions | — | Specified |

Additional statistical-review items: bootstrap replicates 2000 vs implemented default
1000; equal-mass vs equal-width ECE; Holm–Bonferroni vs FDR; 5-seed policy + how seeds
aggregate; paired-difference CI construction (unimplemented); failed-run statistical
treatment (D-13); full-matrix reporting (E-row). Whether stratified-by-class
resampling matches each estimand (it matches i.i.d. event-level sampling; clustered
by-entity dependence is a question for the reviewer given entity-disjoint splits).

---

## I. Domain Review — reviewer questions (not answered here)

1. Do the fraud labels on each candidate dataset represent the intended target
   (instant fraud vs chargeback-confirmed)?
2. Is recall@1%FPR the operationally meaningful primary for alert-budget settings,
   or is PR-AUC/alert-volume more faithful to the research question?
3. Does same-generator IBM family shift make C5-style comparisons uninterpretable?
4. Does feature availability (5/48 on ULB, 47/48 on Kaggle, 7 bank-only) make any
   proposed comparison invalid rather than merely degraded?
5. Is the evaluation population (public synthetic/semi-synthetic transactions)
   appropriate for any real-world claim at all?
6. Does the locked threshold 0.018758 retain a legitimate domain interpretation when
   carried into external evaluation (§5.5 marker)?
7. Analyst capacity number for alert-volume metrics (§7 marker) — needs bank SME.
8. Is a 20% wrong-confident reduction at ≤2pp coverage loss (C3) a defensible risk
   posture, and is the 0.70 external floor (C5) defensible for fraud?
9. Are cost-ratio curves (never realized benefit) the right substitute until bank
   cost figures exist?

---

## J. Reproducibility Contract

Required provenance vs the NR-01 evidence system (schema v1.1):

| Requirement | Captured? | Where / gap |
|---|---|---|
| Git SHA | YES | `git_commit`; mandatory for DEMONSTRATED |
| Dataset identity/version/hash | YES | `dataset.sha256` + `size_bytes` |
| Split | PARTIAL | `evaluation_config.split` — free-form string, no fixed vocabulary (harness, NR-04) |
| Seed | YES | `seed` or `evaluation_config.seed_not_applicable` |
| Command | YES | v1.1 `command` (legacy records without it cannot back DEMONSTRATED) |
| Model/config identity | YES | `model_hash` + `artifact_file_hashes` |
| Feature contract | **GAP** | `feature_schema_version` field exists but pipelines leave it "1.0"; contract file itself inconsistent (14 vs 13, NR-01/Phase 106 N-02) |
| Preprocessing configuration | **GAP** | `preprocessing_version` field exists but unpopulated distinctly |
| Threshold-selection procedure | YES/PARTIAL | `threshold_source` enum + `refuse_test_tuning`; machine-checkable frozen-test artifact pending NR-03 |
| Evaluation procedure | YES | `command` + `evaluation_config` |
| Artifact identity | YES | hashes per file + aggregate |
| Evidence record | YES | ledger append + sidecar + registry + CI-enforced check |

**Verdict:** two schema *fields* already exist for the two gaps — no new evidence
infrastructure is needed; populating them (plus a `protocol_version` linkage, which
does not exist yet) is a harness requirement recorded for NR-03/NR-04, not built here.

---

## K. Canonical Pipeline Mapping

| Pipeline | Part of preregistered experiment? | Question | Dataset/split | Output & evidence | Fully specified? | Executable now? | External data needed? |
|---|---|---|---|---|---|---|---|
| `calibration_test` | Supporting (secondary metrics on validation) — not itself a preregistered run | Is the deployed calibrator well-calibrated in-domain? | ULB-derived synthetic validation split (n=1470) | artifact + ledger + sidecar ✓ | PARTIAL — protocol's ECE must appear on *final test* per §2; current tool measures validation only (external calibration = NR-09) | YES | No |
| `eval_ulb` | Baseline reproduction (feeds NR-07 baselines), not the confirmatory experiment | What does the historical ULB benchmark actually reproduce? | ULB random stratified 80/20, seed 42 | ledger `…e58d68c2bf88` + `ulb_results.json` ✓ | YES as reproduction; the protocol's ULB experiment split is still the §1 marker | YES | No |
| `cross_dataset_eval` | Supporting for C5/C6 (provisional) | How do trained models degrade across datasets? | `transactions.csv` + IBM rows (proxy mapping warned) | ledger + report ✓ | PARTIAL — proxy mapping is explicitly NOT feature-semantic evaluation; criterion 5 requires an independent dataset (Kaggle path) | YES | No |
| `train_compare` | Closest executable analogue of C1/C2 (ensemble vs single) + gating evidence, but on **synthetic** data — the protocol's §1 dataset list does NOT name it | Does the fused ensemble beat its best single constituent? | synthetic time_70_15_15, seed 42 | ledger + metadata + OOD gate rows ✓ | NO — dataset unnamed in protocol (D-1); paired-Δ CI not computed | YES | No |

None was executed this phase. Note (carried from NR-01): `train_compare --outdir`
defaults to `models/artifacts` and overwrites production artifacts — the NR-04 harness
must pin output locations.

---

## L. Reviewer Packet

Prepared in this report; evidence pointers let a reviewer work without undocumented
project knowledge:

**Statistical review questions:** all items in §H (MMD justifications, binning,
replicates, multiplicity, seed policy/aggregation, paired-Δ construction, failed-run
treatment, clustered-dependence under entity splits, full-matrix reporting).

**Domain review questions:** all nine in §I.

**Repository-verifiable answers already in-repo (no reviewer needed):**
split mechanics (`train_compare` time_70_15_15; `ibm_train` user-disjoint code
comment); calibration procedure (Platt on validation, code + ledger); threshold guard
(`refuse_test_tuning`, tested 22/22); evidence pipeline (NR-01: 31-case matrix,
81/81 battery, live PASS 22 claims); dataset hashes (ledger records); negative-result
machinery (append-only ledger, FAILED preservation); current ECE implementation
(10 equal-width, `calibration_test.py`); metric-definitions gap (v1.0, zero
Brier/ECE — verified this phase).

**External blockers:** BAF (download approval + license), IEEE-CIS (credentials),
bank data (institution), bank SME/capacity inputs, counsel for publication terms
(O5–6 → NR-31), the reviewers themselves (statistical + domain + governance roles
in the sign-off block).

The packet is **not approved** — it awaits real reviewers.

---

## M. Corrections

Applied to `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` (all recorded in
the protocol's new Change history table; version 0.1-draft → 0.1.1-draft):

1. Sign-off cross-ref `R-05` → NR-02 (Phase 106 rebuilt the roadmap).
2. §0 OPTION decision cross-ref `R-03` → Phase 106 §E three-contract determination
   (decision marker preserved; framing note added, no scope chosen).
3. BAF cross-ref `R-20` → NR-16.
4. `metric_definitions` cross-ref `R-02` → NR-03, plus factual note it is still v1.0.
5. Cross-refs `R-21` → NR-17, `R-18` → NR-14.
6. Criterion 5 failure branch: the prose-only 0.435 was replaced as the evidence
   citation by artifact-backed 0.595 (0.435/44.9% = NOT ESTABLISHED, Phase 105,
   C-011/C-012; 0.595/9.9% = DEMONSTRATED, C-013/C-014). **MMD ≥0.70 and all
   consequences unchanged** — evidence-basis correction only.
7. §2 factual implementation note added: existing ECE code = 10 equal-width bins;
   binning decision marker retained.

**Not done (by rule):** no marker removed (count verified 11 → 11); no MMD, criterion,
or substantive methodology altered; no unresolved state hidden. Residual markers in a
*meta* sentence were reworded so the marker literal count stays exactly 11.

---

## N. Preregistration Status

```
REVIEW REQUIRED
```

The protocol is internally coherent (no contradictions found) and now factually
current, but contains substantive statistical and domain decisions — the 11 existing
markers plus the sign-off items identified in §D/§E — that require qualified review
before any experiment begins. It is **not** READY FOR REVIEW only in the sense that
those decisions are expected to be *made during* review; it is **not** NOT READY (no
internal contradictions or unreviewable ambiguity); it is **not** BLOCKED (review can
proceed on available material). It is **not APPROVED** — no reviewer has reviewed it.
Practical state: **PENDING EXTERNAL REVIEW** (statistical reviewer, domain reviewer,
lead-researcher scope decision on §0, governance role for bank data).

---

## O. Gate 1 Impact

NR-02 does **not** close Gate 1. What changed vs Phase 106:

- Protocol is now factually current (7 corrections, change-history preserved) —
  the Phase 106 §H correction list is discharged.
- The reviewer packet exists (§L of this report): statistical, domain,
  repo-verifiable, and blocker questions are isolated.
- Preregistration status is classified from evidence (REVIEW REQUIRED), not assumed.

What still blocks Gate 1 completion: actual sign-off (needs the four roles — external,
not available in-repo) and **NR-03** (evaluation semantics v1.1 + frozen-test design).
Current research gate: **PARTIAL** (unchanged) — NR-02 moved the *readiness* of Gate 1
up, not the gate itself.

---

## P. Blockers

1. **Qualified reviewers unavailable in-repo** — statistical reviewer, domain reviewer
   (bank SME or equivalent), governance role; lead-researcher scope decision on §0 is
   a human decision, not a repository inference.
2. Publication-terms/counsel input (O5–6) defers to NR-31 (Gate 5) — not required for
   *review*, only for the sign-off row marked "(if bank data)".
3. Dataset access blockers (BAF, IEEE-CIS, bank data) block *experiments*, not this
   review (§G).

---

## Q. Next Requirement

**NR-03 — Evaluation semantics & frozen final-test design**: `metric_definitions`
v1.1 (Brier/ECE with the reviewer-chosen bin semantics, reliability-diagram spec),
plus the machine-checkable frozen-test artifact (per-dataset split definitions +
hashes, single-use rule, test-access detection dry run). Identified only — **not
started.**

---

## R. Scope Verification

- **Experiments executed:** none (only file reads, greps, and `ls`; no evaluation
  script was run this phase).
- **Model/dataset/feature/threshold/methodology changes:** none. `models/`, `data/`,
  `backend/src/risk_engine/`, `backend/src/privacy_layer/` untouched.
- **Protocol edits:** factual/documentary only, itemized in §M and in the protocol's
  own Change history; marker count 11 → 11; MMDs/criteria/sign-off block byte-identical
  in substance.
- **Enforcement safety:** `claim_evidence_check` re-run after protocol edits —
  PASS (22 claims), rc=0 (protocol is not a managed surface; registry/ledger untouched).
- **Files changed this phase:** exactly two — the protocol (factual corrections) and
  this report.
- **No reviewer approval fabricated anywhere.**

---

## S. Final Status

```
PASS WITH LIMITATIONS
```

Every acceptance criterion is met except those that inherently require unavailable
people: sign-off remains PENDING EXTERNAL REVIEW (recorded honestly, not fabricated),
and the substantive decisions stay explicitly marked. No experiment was executed; no
scientific change was introduced; NR-03 is identified as the next requirement.
