# Phase 106 — Post-Evidence Roadmap Reconciliation & Research Gate Reset

**Date:** 2026-10-03 · **Scope:** read-only reconciliation. No model, dataset, threshold,
methodology, or production code was changed in this phase. The preregistered protocol
was **not** edited; factual corrections are recorded here as findings (§H).
The three files created by this phase are the only writes.

---

## A. Phase status

**PASS**

All 45 Phase 104 gaps received a current status (§E), all 48 Phase 104 roadmap
requirements received a disposition (§F), Phase 105 corrections are incorporated (§C),
the evidence baseline is separated from historical claims (§D), the protocol's factual
dependencies were reviewed without modifying it (§H), Phase-105-introduced gaps were
considered (§G), the next research gate has explicit prerequisites (§I, Doc 3), and the
remaining roadmap is dependency-aware with a derived phase count (§J, §K).

---

## B. Starting state

| Item | Value |
|---|---|
| Git SHA | `a0b600ba2edd917498d3a9551c19e10225930e4b` (`a0b600b`) — unchanged through Phases 105–106 (nothing committed) |
| Phase 105 status | **PASS WITH LIMITATIONS** |
| Working tree | Phase 105 completion state: 13 modified tracked files (CI workflow, README, 9 backend scripts, CLAIMS_MATRIX, ledger + cross-dataset report) + untracked Phase 104/105 deliverables, `backend/scripts/claim_evidence_check.py`, evidence outputs (`reports/ulb_results.json`, `reports/calibration_test/`, `reports/phase105/`, 10 new `record_*.json`) |
| Evidence ledger | **11 records** at Phase 106 read (1 legacy smoke + Phase 105 executions + hook-triggered identical calibration re-runs; append-only, 0 rejected) |
| Evidence artifacts | `docs/evaluation/claims_registry.jsonl` (22 entries), `reports/calibration_test/calibration_metrics.json`, `reports/ulb_results.json`, `reports/cross_dataset/cross_dataset_report.json`, `reports/phase105/*` |
| CI changes | 2 steps added to the test job (`claim_evidence_check.py`, `eval_record_test.py`) — locally verified, **not yet run on the GitHub runner** |
| README changes | benchmark region rewritten with claim annotations; historical tuple marked NOT ESTABLISHED; k-anonymity and pentest-count wording corrected; claims-table rows 2/3/4/7/12/18 updated |
| Confirmed | `backend/src/privacy_layer` and `backend/src/risk_engine` untouched (`git status` empty for both) — no feature/model/behavior change |

The state being reconciled **is** the Phase 105 completion state.

---

## C. What Phase 105 changed (explicit corrections to Phase 104)

| # | Phase 104 position | Phase 105 verified fact | Effect |
|---|---|---|---|
| 1 | Recall@1%FPR **91.8%** = NOT ESTABLISHED as documented | Reproduced **bit-identically** (0.918367, `eval_ulb.py`, ledger `…e58d68c2bf88`); the historical tuple's *attribution* was wrong, not this component | component upgraded to **DEMONSTRATED** |
| 2 | Brier 0.0009 / ECE 0.0013 = NOT ESTABLISHED (compared against `calibration_and_stress.json`) | Reproduced **exactly**: 0.0009280 / 0.0012790, n=1470 validation split, ledger `…72d714bd81b3`. Phase 104 tested the *wrong artifact* and never ran `calibration_test.py` | **DEMONSTRATED (validation-split scope)**; Phase 104 finding overturned |
| 3 | External "0.435–0.595 DEMONSTRATED (phase24 `05_e_hardneg_metrics.json` backs both)" | That file contains **0.594662 / FPR 0.099313**; **0.435 / 44.9%** exists only in prose (`PHASE25_FINAL_REPORT.md` L46), repo-wide search found no artifact | **0.595 DEMONSTRATED; 0.435 NOT ESTABLISHED** |
| 4 | (not known) | `train_compare.py`'s ledger append **never worked since Phase 39** — `UnboundLocalError: gate_rows` swallowed by a bare `except` (explains the 1-record ledger despite CI runs); fixed in place | evidence-generation defect closed |
| 5 | Ledger = 1 record; no `command` field | Ledger = **11**; schema **v1.1** adds `command`; DEMONSTRATED rules machine-enforced | G-01 substantively implemented |
| 6 | Open limitation: battery not re-run post-restart | Battery executed 14:06:46→14:21:56: **76/79 as-run (services down), 3/3 re-run PASS with stack up → 79/79 effective**; both outcomes preserved | Phase 104 limitation resolved |
| 7 | CI cannot reject unevidenced claims | `claim_evidence_check.py` + `eval_record_test.py` added to CI; valid PASS / invalid FAIL / 7-fixture self-test demonstrated **locally** | G-01 PARTIAL (runner unverified) |
| 8 | README/`compare_ml_systems.py`/`fraud_report.py` presented hardcoded values as MEASURED | values now loaded from executed artifacts or refused (`N/A` / `BLOCKED`); `CLAIMS_MATRIX` M5 corrected from VERIFIED → UNSUPPORTED with the true evidence path | G-02/G-07 closed |
| 9 | (context) | phase108 artifact shows the **production model on ULB = rules-only fail-close, ROC-AUC 0.5** (5/48 features usable) — independent confirmation that 0.966 was never the production system | strengthens §D |
| 10 | README claimed a k-anonymity export gate (FALSE) | wording corrected to match runtime (checker is test/CLI-only); wiring itself remains future work | G-08 closed as stated; wiring → NR-27 |

---

## D. Scientific evidence baseline (Phase 105 evidence only; no rankings)

### Demonstrated (artifact + provenance/ledger)
- ULB research pattern-XGB: ROC-AUC 0.975807, PR-AUC 0.883465, **Recall@1%FPR 0.918367** — ledger `…e58d68c2bf88`, seed 42, stratified 80/20, dataset sha `76274b69…`.
- **Brier 0.0009280 / ECE 0.0012790** on the synthetic validation split (n=1470) — ledger `…72d714bd81b3`.
- IBM v2 cross-dataset ROC-AUC **0.8726** (README 0.873) — ledger `…e2a14ae94bef`, `--ibm-rows 200000`, bit-identical reproduction; same-generator-family caveat retained.
- External Kaggle fraudTest **ROC-AUC 0.594662 / FPR 0.099313** (locked threshold 0.018758) — artifact + registry provenance block.
- Synthetic fused baseline: ROC 0.983797 / PR 0.950814 / R@1% 0.955556 — ledger `…359039aac6cf` (**synthetic only**).
- Production-model-on-ULB fail-close (ROC 0.5, recall 0) — phase108 manifest+result hash.
- Battery 79/79 effective; leakage suite 123/123 (as suite evidence).

### Self-tested (internal validation; no external-effectiveness claim)
- IBM v2 table (0.9819 / 0.4063 / 0.8356; fused 0.9779 / 0.3138 / 0.84) — artifact exists, no seed/git; split user-disjoint but not chronological.
- Phase 23B external 0.4658; security/pentest suites; historical load_test TPS/latency; eval-record machinery (22/22); claim-enforcement self-test (7/7).

### Simulated (synthetic/simulated evidence only)
- All training and in-domain evaluation (IBM generator, PaySim, PS-14 derived); federated results (3 simulated institutions); rules behavior on synthetic streams.

### Not established
- Historical README tuple **0.966 / 0.877 / 97.8%@0.43 / 1.996%** (registry C-101…C-104).
- External intermediate **0.435 / FPR 44.9%** (C-011/C-012).
- Real-world effectiveness; **ensemble superiority** (no preregistered comparison has run); **gating effectiveness** (implementation exists; no UNGATED-vs-GATED experiment exists); external calibration; subgroup/business-metric claims; regulatory or institutional validation; production promotion; executed pentest-scenario count.

### Blocked
- Full 21+7-feature contract evaluation (bank data — 7 features absent from every public dataset); NeurIPS 2022 BAF (not acquired; license undocumented; download approval required); IEEE-CIS (auth-blocked); independent assurance items (external parties); legal/privacy review.

---

## E. Gap reconciliation — every G-01…G-45

Classes: CLOSED · PARTIALLY CLOSED · OPEN · BLOCKED · SUPERSEDED · DUPLICATE.

| ID | Status | Evidence / reason | Action |
|---|---|---|---|
| G-01 | **PARTIALLY CLOSED** | `claim_evidence_check.py` + `eval_record_test.py` in CI; local valid-PASS/invalid-FAIL/self-test proven; **remaining:** never run on GitHub runner, evidence-commit coupling unproven, local battery still omits both suites | carry → **NR-01** |
| G-02 | **CLOSED** | tuple classified NOT ESTABLISHED (C-101…C-104), README corrected, replaced by reproduced 0.976/0.918/0.883 with ledger ids; `compare_ml_systems.py` de-hardcoded | retain closure |
| G-03 | **CLOSED** | Brier/ECE reproduced exactly, ledger-bound, scoped to validation split (Phase 104's contrary finding overturned — see §C.2) | retain closure |
| G-04 | **OPEN** | no untouched pre-registered final-test set exists; repeated historical tuning on same ULB splits unchanged | → **NR-03** (frozen-test design) |
| G-05 | **OPEN** | `/internal/attribution` still bypasses `enforce_before_inference`/`check_access` (untouched this phase) | → **NR-24** (moved to Gate 4, see §J ordering note) |
| G-06 | **OPEN** | `metric_definitions.py` still v1.0 with 0 Brier/ECE mentions (Phase 105 deliberately scoped it out) | → **NR-03** |
| G-07 | **CLOSED** | `CLAIMS_MATRIX.md` M5 → UNSUPPORTED with corrected evidence; M6 downgraded to PARTIAL | retain closure |
| G-08 | **CLOSED** (as stated) | README wording now matches runtime; the *missing wiring* remains and is tracked separately | wiring → **NR-27** |
| G-09 | **OPEN** | README still claims `docs/PS-14_fraud_detection.pptx`, `docs/ACCESS.md.enc`, root `scripts/deploy.sh` paths | → **NR-41** |
| G-10 | **PARTIALLY CLOSED** | README now documents the 42/34/66+43 discrepancy instead of asserting 42; executed count still unpinned by any artifact | final resolution with external pen test → **NR-36** |
| G-11 | **PARTIALLY CLOSED** | CI now runs both suites; the local battery script (`.freebuff/p114_battery.sh`) still does not | → **NR-01** |
| G-12 | **OPEN** | README phase table still stops at 56 + prose; malformed row ~L251 untouched; no repo-phase 104–118 coverage | → **NR-41** |
| G-13 | **OPEN** | observability still in-memory | → **NR-26** |
| G-14 | **OPEN** | PostgreSQL path still only compose.prod + migration 001 | → **NR-21** |
| G-15 | **OPEN** | no TLS in dev/preview compose | → **NR-22** |
| G-16 | **OPEN** | no load/concurrency/recovery artifacts | → **NR-23** |
| G-17 | **OPEN** | full-frame aggregates (`ibm_train.py:180–188`) and dedup status untested | → **NR-10** |
| G-18 | **OPEN** | IBM split still user-disjoint, not chronological | → **NR-05** |
| G-19 | **SUPERSEDED** | binary OPTION A/B framing dissolved by evidence-based contract decomposition (§ Step 9 determination, below): different contracts are evaluable on *different* datasets; the 7-feature remainder is BLOCKED regardless | replacement: scope decision inside **NR-02** + design spread across NR-04/05/06 |
| G-20 | **BLOCKED** | 7 bank features absent from every public dataset (phase64); Phase 105 changed nothing (privacy_layer untouched) | remains BLOCKED → **NR-19** + bank branch |
| G-21 | **OPEN** | no UNGATED-vs-GATED experiment exists | → **NR-14** |
| G-22 | **OPEN** | no preregistered ensemble-vs-single comparison | → **NR-07**, **NR-08** |
| G-23 | **OPEN** | in-domain calibration now DEMONSTRATED; external calibration still never evaluated | → **NR-09** |
| G-24 | **OPEN** | no subgroup analysis on evaluated models | → **NR-11** |
| G-25 | **OPEN** | no business metrics | → **NR-12** |
| G-26 | **OPEN** | no label-delay/point-in-time inventory | → **NR-19** |
| G-27 | **BLOCKED** | BAF not acquired; license undocumented; download approval required | → **NR-16** (blocked) |
| G-28 | **BLOCKED** | IEEE-CIS auth-blocked | conditional: only if pursued |
| G-29 | **OPEN** | still one external family (Kaggle) + same-generator IBM | → **NR-17** |
| G-30 | **OPEN** | no feedback-loop analysis | → **NR-20** |
| G-31 | **OPEN** | no feature-shift/availability analysis | → **NR-18** |
| G-32 | **OPEN** | `_MANIFEST_SECRET`/`_TOKEN_SECRET` still hardcoded | → **NR-24** |
| G-33 | **OPEN** | no Ed25519 code anywhere (docstring recommendations only) | → **NR-25** |
| G-34 | **OPEN** | audit writes still synchronous on the detection path | → **NR-27** |
| G-35 | **OPEN** | hash chain still cannot detect forged appends | → **NR-37** (independent review scope) |
| G-36 | **OPEN** | deployed release manifest still `source_git_sha: unknown`, null seed, empty fingerprints; Phase 105 completed provenance only for *new* evaluation records | → **NR-44** (repro package must include complete release provenance, regenerated without touching weights) |
| G-37 | **OPEN** | stale `altman_native_v2` schema string in model record | → **NR-28** |
| G-38 | **OPEN** | repository sprawl unchanged (misc 719, tracked .pyc, duplicate trail UIs, duplicated feature lists) | → **NR-28** |
| G-39 | **CLOSED** | P20/45 path documented as historical/inactive — required no work by design | retain closure |
| G-40 | **PARTIALLY CLOSED** | fresh-clone data/model-artifact blocker now documented in `BASELINE_REPRODUCTION.md` §7; the distribution/artifact policy decision remains | → **NR-44** |
| G-41 | **OPEN** | 16 bank deliverables unchanged (protocol is still a draft) | → **NR-29…NR-35** |
| G-42 | **OPEN** | zero external assurance items completed; Phase 105 changed nothing here | → **NR-36…NR-39** |
| G-43 | **OPEN** | no legal/privacy/compliance review | → **NR-35** |
| G-44 | **OPEN** | protocol still DRAFT/NOT APPROVED (11 approval markers outstanding) | → **NR-02** |
| G-45 | **OPEN** | no paper, repro package, or numeric presentation | → **NR-40…NR-44** |

**Totals: CLOSED 5 (G-08 as stated) · PARTIALLY CLOSED 4 · OPEN 32 · BLOCKED 3 · SUPERSEDED 1 · DUPLICATE 0 = 45.** (G-08 counted as CLOSED because the README wording now matches runtime; its unwired-remainder is tracked in NR-27.)

### Step-9 determination — the seven unavailable features (evidence-based)

Status unchanged by Phase 105 (`privacy_layer`/`risk_engine` untouched; no public dataset
provides them — phase64 blocker chain; phase108 shows even the 48-native production
vector is only 5/48 usable on ULB, while phase25 reconstructed 47/48 on Kaggle fraud).

Roadmap determination — **evaluate the contracts separately, and treat the 7-feature
full contract as its own blocked research question**:

1. **48-native production contract** — externally evaluable on Kaggle fraud (phase24/25
   prove reconstruction is possible; 0.595 artifact exists) → evaluated under
   preregistration in Gate 2 (NR-04…NR-09), not deferred.
2. **21-domain / public_v1 research contract** — evaluable on IBM v2 (21 features) and,
   at 14 declared features (`backend/research/public_feature_contract.json`,
   `research_feature_version: public_v1`), on public data → NR-04/NR-05/NR-06.
3. **21+7 full production-features contract** — **BLOCKED on institution data**; becomes
   its own research question answered only with bank data (NR-19, NR-29+).

Why not the other options: "wait for institution data" would stall all external
validation indefinitely (bank data is outside the team's control); "reduced public
contract only" would silently substitute a weaker model for the deployed one (the
original Phase 104 objection); "feature gap as the only question" ignores that two of
three contracts are already externally evaluable today. This determination dissolves
the binary OPTION A/B choice (G-19 → SUPERSEDED) and **feeds protocol §0 at sign-off —
the protocol itself is not edited here.**

---

## F. Roadmap reconciliation — every R-01…R-48

| ID | Disposition | Evidence | Replacement | Notes |
|---|---|---|---|---|
| R-01 | **PARTIALLY CLOSED** | ledger expansion ✓, CI enforcement ✓ (local), battery coverage ✗ | NR-01 | remainder only |
| R-02 | **PARTIALLY CLOSED** | headline reconciliation ✓, CLAIMS_MATRIX ✓; `metric_definitions` v1.1 ✗ | NR-03 | remainder folded with R-16 |
| R-03 | **SUPERSEDED** | OPTION A/B framing dissolved by contract decomposition (§E G-19) | NR-02 (scope decision §0) | human decision still required — now three-track |
| R-04 | **PARTIALLY CLOSED** | ULB, calibration, cross-dataset, train_compare reproduced into ledger; **ibm_train not re-executed** (trace-only, no seed/git in artifact) | NR-04 | IBM upgrade folded into harness build |
| R-05 | **OPEN** | protocol still DRAFT, 11 approval markers | NR-02 | merged with R-03's decision |
| R-06 | **OPEN** | attribution bypass unchanged | NR-24 | **moved Gate 1 → Gate 4**: it does not affect research validity; "research validity before production hardening" |
| R-07 | **OPEN** | no preregistered harness exists | NR-04 | absorbs R-04 remainder |
| R-08 | **OPEN** | no temporal evaluation under preregistration | NR-05 | |
| R-09 | **OPEN** | entity-disjoint eval exists only ad hoc (`ibm_train`) | NR-06 | |
| R-10 | **OPEN** | no baselines/ablations suite | NR-07 | |
| R-11 | **OPEN** | ensemble contribution never tested | NR-08 | |
| R-12 | **OPEN** | external calibration never evaluated | NR-09 | |
| R-13 | **OPEN** | statistical leakage audit not done (structural suite ≠ statistical) | NR-10 | |
| R-14 | **OPEN** | no subgroup analysis | NR-11 | |
| R-15 | **OPEN** | no business metrics | NR-12 | |
| R-16 | **SUPERSEDED** | threshold-discipline *tooling* must exist **before** any experiment, not as a mid-Gate-2 phase | NR-03 | merged with R-02 remainder |
| R-17 | **OPEN** | Gate 2 decision not run | NR-13 | |
| R-18 | **OPEN** | no gating experiment | NR-14 | |
| R-19 | **OPEN** | drift/corruption robustness unvalidated | NR-15 | |
| R-20 | **BLOCKED** | BAF not acquired; license undocumented | NR-16 | external: download approval |
| R-21 | **OPEN** (dependent) | needs R-20 or a qualified alternative | NR-17 | |
| R-22 | **OPEN** | feature-shift analysis not done | NR-18 | |
| R-23 | **OPEN** | label-delay/point-in-time not built | NR-19 | bank-heavy |
| R-24 | **OPEN** | feedback-loop bias not analyzed | NR-20 | |
| R-25 | **OPEN** | PostgreSQL path unexercised | NR-21 | |
| R-26 | **OPEN** | no TLS in compose | NR-22 | |
| R-27 | **OPEN** | no load tests | NR-23 | |
| R-28 | **OPEN** | hardcoded HMAC secrets | NR-24 | now also absorbs R-06 |
| R-29 | **OPEN** | no Ed25519 evaluation | NR-25 | |
| R-30 | **OPEN** | observability lost on restart | NR-26 | |
| R-31 | **OPEN** | k-anon wiring absent; async-audit re-verification pending | NR-27 | |
| R-32 | **OPEN** | sprawl unchanged | NR-28 | now also dispositions `fraud_report` legacy artifacts + G-37 |
| R-33 | **OPEN** | O1–2 missing | NR-29 | |
| R-34 | **OPEN** | O3–4 missing | NR-30 | |
| R-35 | **OPEN** | O5–6 missing | NR-31 | |
| R-36 | **OPEN** | O7–8 missing | NR-32 | |
| R-37 | **OPEN** | O9–11 missing/partial | NR-33 | |
| R-38 | **OPEN** | O12–14 missing | NR-34 | |
| R-39 | **OPEN** | O15–16 not started | NR-35 | BLOCKED on counsel/compliance |
| R-40 | **OPEN** | no independent pentest | NR-36 | BLOCKED on external party |
| R-41 | **OPEN** | no independent code/architecture review | NR-37 | BLOCKED on external party |
| R-42 | **OPEN** | no methodology/statistical review | NR-38 | BLOCKED on external party |
| R-43 | **OPEN** | no independent reproduction | NR-39 | BLOCKED on external party (partly enabled by Phase 105 manual) |
| R-44 | **OPEN** | no final freeze | NR-40 | |
| R-45 | **OPEN** | README rewrite outstanding (Phase 105 made targeted corrections only) | NR-41 | also carries G-09/G-12 |
| R-46 | **OPEN** | presentation lacks measured numbers | NR-42 | |
| R-47 | **OPEN** | no paper/thesis | NR-43 | |
| R-48 | **OPEN** | no reproducibility package; G-36/G-40 policy pending | NR-44 | |

**Totals: PARTIALLY CLOSED 3 · SUPERSEDED 2 · OPEN 42 · BLOCKED 1 · CLOSED 0 · DUPLICATE 0 = 48.**
No requirement is fully CLOSED: Phase 105 completed the *substance* of R-01/R-02/R-04
with precisely defined remainders (this is recorded rather than glossed over).

---

## G. New gaps introduced or exposed by Phase 105

| ID | Gap | Necessity | Disposition |
|---|---|---|---|
| **N-01** | CI evidence steps exist only locally: never executed on the GitHub runner, and CI fails if evidence files are not committed together with the README (commit coupling) | **genuinely necessary** — G-01's remaining acceptance criterion | folded into **NR-01** (no new phase) |
| **N-02** | `backend/research/public_feature_contract.json` declares `public_features_count: 14` but its `features` array holds 13 entries — internal inconsistency in the contract that NR-04's harness will consume | **genuinely necessary** — would otherwise corrupt a preregistered run or silently drop a feature | folded into **NR-04** pre-run contract check |
| — (considered, no requirement) | Hook-triggered duplicate calibration records (ledger growth) | Records are deterministic, identical, append-only, and honestly preserved — deduplication machinery would be governance without benefit | accepted condition; document in NR-01 notes only |
| — (considered, no requirement) | Expanding claim-enforcement surfaces beyond the README benchmark region | Covered as a stated Phase 105 limitation; headline risk is covered; expansion is discretionary and can ride along with NR-41's README rewrite if desired | deferred, not scheduled |

---

## H. Protocol impact — preregistration review required (protocol NOT edited here)

The protocol remains **DRAFT/NOT APPROVED** with 11 `[REQUIRES DECISION/APPROVAL]`
markers. Phase 105 changed no decision criterion, but four factual corrections must be
incorporated at sign-off (NR-02):

1.
   `OLD FACTUAL ASSUMPTION` — criterion 5's failure branch presupposes external ROC-AUC
   "stays 0.435–0.595" with 0.435 treated as evidence (Phase 104 classified the pair
   DEMONSTRATED).
   `NEW VERIFIED FACT` — only **0.595** is artifact-backed; **0.435 is NOT ESTABLISHED**
   (prose-only).
   `WHY IT CHANGED` — Phase 105 repo-wide artifact search.
   `PROTOCOL REVIEW REQUIRED` — **YES** (editorial factual update; the criterion's
   statistical rule is unchanged).

2.
   `OLD` — §0 points to "roadmap phase R-03" for the OPTION decision; §1 points to
   "R-20" for BAF; §2 points to "R-02" for metric definitions.
   `NEW` — R-03 is SUPERSEDED (three-track contract scope, §E Step-9 determination);
   BAF = NR-16; metric definitions = NR-03.
   `WHY` — Phase 106 roadmap rebuild.
   `REVIEW` — **YES** (cross-reference updates at sign-off).

3.
   `OLD` — §2 proposes "ECE (10–15 **equal-mass** bins) [REQUIRES DECISION]".
   `NEW` — the only existing implementation (`calibration_test.py`) uses **10
   equal-width bins**, and the reproduced ECE 0.0012790 was measured that way.
   `WHY` — Phase 105 executed the calibration path and revealed the concrete semantics.
   `REVIEW` — **YES**: sign-off must choose to adopt the implemented definition or
   change the implementation (a methodology change that would then require re-running
   the Phase 105 calibration record — *not* done in this phase).

4.
   `OLD` — §0 scope = S1 vs S2 (single OPTION choice).
   `NEW` — evidence supports three separately evaluable contract tracks (§E Step-9).
   `REVIEW` — **YES**: scope section must state which candidate(s) each experiment
   covers.

Unchanged and therefore requiring no correction: MMD proposals (still pending approval),
CI/replication rules, negative-result conclusions A/B/C, threshold-selection procedure
(train→validation→frozen→single-use test; `refuse_test_tuning` guard verified), locked
S1 threshold 0.018758, dataset candidate list (BAF still correctly marked blocked).

---

## I. Gate readiness

| Gate | Status | Basis |
|---|---|---|
| **Gate 1 — Evidence Foundation** | **PARTIAL** | substance done by Phase 105; NR-01 (runner verification + battery coverage), NR-02 (protocol approval + scope), NR-03 (metric definitions v1.1 + frozen final-test design) remain |
| **Gate 2 — Research Go/No-Go** | **NOT READY** | prerequisites not met: protocol unapproved, semantics/frozen-test not fixed, harness does not exist; per-step assessment in `RESEARCH_GATE_READINESS.md` |
| **Gate 3 — Mechanism Validation** | **NOT READY** | strictly post-Gate-2; no gating/drift experiments exist |
| **Gate 4 — Engineering** | **NOT READY (and deliberately deferred)** | research validity precedes production hardening; requirements remain valid with unchanged prerequisites (§10 below) |
| **Gate 5 — External/Institutional** | **BLOCKED / DEPENDENT ON RESEARCH RESULT** | every item needs either signed external parties, legal review, or a completed Gate 2 result to review |
| **Gate 6 — Final Freeze** | **NOT STARTED** | depends on all prior gates; G-45 unchanged |

**Current research gate: PARTIAL** — the evidence foundation is sufficiently trustworthy
to *finish Gate 1*, but not yet to *begin* preregistered comparisons (three explicit
prerequisites, none of them external — see Doc 3).

---

## J. New authoritative roadmap

Column key for all tables: **P** = Prerequisites · **In** = Inputs · **Out** = Expected
output · **Acc** = Acceptance criteria · **Evd** = Evidence required · **Dep** =
Dependency · **Ext** = blocking external dependency (— if none).

**Freeze ordering — RESOLVED (RP-01 §H).** Exploratory work may occur before
the Research Plan (`docs/RESEARCH_PLAN.md`) freeze; confirmatory Track M
execution may begin only after freeze. **NR-05 may therefore proceed before
the freeze only as exploratory work on previously exposed datasets**
(ULB/Kaggle/IBM remain EXPLORATORY per plan §14): it cannot produce
confirmatory Track M evidence, make an exposed dataset "untouched", be
presented as independent replication, or be used to retrospectively select
or alter the confirmatory decision rules. Confirmatory Track M on untouched
eligible datasets stays blocked until all five freeze conditions hold:
statistical review complete · all freeze placeholders resolved ·
`docs/FREEZE_RECORD.json` exists and is complete · the freeze checker
passes · frozen artifacts committed at the tagged Git state. The Research
Plan's freeze-before-confirmatory-experiment rule remains authoritative;
NR-05 is still the next roadmap requirement.

### GATE 1 — EVIDENCE FOUNDATION (3 phases)

| ID | Title | Purpose | P | In | Out | Acc | Evd | Dep | Gate | Ext |
|---|---|---|---|---|---|---|---|---|---|---|
| NR-01 | Evidence enforcement verification & battery coverage | close G-01/G-11/N-01: prove CI enforcement on the real runner and cover the suites locally | authorized commit+push | Phase 105 tree, CI steps, battery script | green GitHub CI run showing both new steps executed; battery script runs `eval_record_test` + `claim_evidence_check`; evidence files committed with README (commit-coupling honored) | CI log URL/record shows PASS on runner; local battery tally includes both suites; `claim_evidence_check` PASS post-commit | CI run record, battery tally | Phase 105 state | 1 | **commit/push authorization** |
| NR-02 | Pre-registration factual correction, contract-scope decision & sign-off | make the protocol approvable and choose the evaluation scope | NR-01 (enforcement trustworthy) | protocol draft, §H findings, §E Step-9 determination | versioned, dated, signed protocol (v1.0) with the 4 factual corrections applied, S1/scope selected per contract track, 11 approval markers resolved | protocol leaves DRAFT; every `[REQUIRES DECISION/APPROVAL]` resolved or explicitly deferred with owner; §0 names the evaluated candidate(s) | signed protocol file + decision log | NR-01 (parallelizable if push is withheld) | 1 | reviewers (statistical/domain) |
| NR-03 | Evaluation semantics & frozen final-test design | fix metric semantics and the untouched-test discipline before any experiment | NR-02 (approved criteria) | `metric_definitions.py` v1.0, protocol §1/§2, `refuse_test_tuning` guard | `metric_definitions` v1.1 (Brier, ECE with chosen bin semantics, reliability-diagram spec); a machine-checkable frozen-test design artifact (per-dataset split definitions + hashes, single-use rule); ledger records carry `metric_definitions_version` 1.1 | Brier/ECE defined authoritatively; frozen-test artifact exists and is referenced by the protocol; a dry-run validation shows test-access detection works | definitions file, frozen-test artifact, validation output | NR-02 | 1 | — |

### GATE 2 — RESEARCH GO/NO-GO (10 phases)

| ID | Title | Purpose | P | In | Out | Acc | Evd | Dep | Gate | Ext |
|---|---|---|---|---|---|---|---|---|---|---|
| NR-04 | Preregistered benchmark harness (+ IBM v2 ledger upgrade) | one ledgered pipeline for all Gate-2 evaluations | NR-02, NR-03 | eval_record v1.1, claims registry, contract files (**incl. N-02 fix**), datasets with verified hashes | harness writing full provenance records for every run; `ibm_train` results re-produced as ledger records (closes R-04 remainder); public_v1/21/48 contract selection per NR-02 | every harness run yields a DEMONSTRATED-grade record; contract-count check passes; chain command→artifact→record→registry demonstrated | ledger records, harness docs | NR-03 | 2 | — |
| NR-05 | Temporal-split evaluation | answer the time-generalization question | NR-04 | datasets with time fields (IBM v2, Kaggle, ULB where applicable) | temporal results as ledger records per candidate contract | splits pre-declared in frozen-test design; no test-set inspection before threshold freeze | records + split artifacts | NR-04 | 2 | — |
| NR-06 | Entity-disjoint evaluation | answer the new-entity generalization question | NR-04 | same datasets, entity keys | entity-disjoint results as ledger records | identical discipline as NR-05; both axes reported separately | records | NR-04 | 2 | — |
| NR-07 | Baselines & ablations suite | establish comparison baselines | NR-04 | harness | baseline/ablation matrix (single models, feature-set ablations) | baselines chosen on validation only; every cell ledgered | records, matrix artifact | NR-04 | 2 | — |
| NR-08 | Ensemble-contribution experiment | adjudicate protocol criteria 1/2 | NR-07, NR-02 (criteria signed) | baselines, ensemble candidate | paired-bootstrap comparison vs best single constituent on final test | executed exactly per approved criteria; result reported whether positive or negative | records + criterion report | NR-07 | 2 | — |
| NR-09 | External calibration evaluation | test calibration beyond the validation split | NR-04, NR-03 (ECE semantics) | calibrated candidates | Brier/ECE/reliability on external splits as ledger records | calibration separated from threshold selection; CIs per protocol | records + reliability artifacts | NR-03 | 2 | — |
| NR-10 | Statistical leakage audit | close G-17 and related UNKNOWNs | NR-04 | feature code, datasets | audit covering duplicates/near-duplicates, full-frame aggregates (`merchant_user_count`, `city_user_count`), preprocessing/feature-selection leakage, historical repeated-tuning exposure | every check PASS/FAIL/PARTIAL with evidence; FAILs become tracked gaps | audit report + suite output | NR-04 | 2 | — |
| NR-11 | Subgroup/segment analysis | segment-level performance honesty | NR-05/06 results | available segments only (amount bands, time, merchant/city, entity, feature-availability groups) | per-segment metrics with CIs | no invented segments; small-n segments flagged | records + segment report | NR-05 | 2 | — |
| NR-12 | Business-relevant metrics (public-data portion) | recall@fixed alert volume, burden, cost-proxy curves | NR-05 | results + protocol §7 | metric curves parameterized by cost ratio | no realized financial claim; capacities labeled illustrative | records + curves | NR-05 | 2 | bank SME input for real numbers |
| NR-13 | **Gate 2 GO/NO-GO report** | adjudicate the project against approved criteria | NR-08…NR-12 | all Gate 2 evidence | signed decision: Conclusion A / B / C candidate, or inconclusive-with-next-experiment | judged strictly against signed criteria; negative results preserved; no complexity added to force a positive result | report + criteria mapping | NR-08…NR-12 | 2 | — |

### GATE 3 — MECHANISM VALIDATION (7 phases)

| ID | Title | Purpose | P | In | Out | Acc | Evd | Dep | Gate | Ext |
|---|---|---|---|---|---|---|---|---|---|---|
| NR-14 | UNGATED-vs-GATED experiments | adjudicate protocol criteria 3/4 | Gate 2 decision (NR-13), harness | 7 pre-defined input conditions | coverage/selective-risk/wrong-confident comparisons with paired CIs | executed per approved criteria; implementation-only claims stay forbidden until then | records + curves | NR-13 | 3 | — |
| NR-15 | Drift & corruption robustness | validate monitoring under injected drift/corruption | NR-14 | drift monitor, harness | detection+response evidence | synthetic drift injections with known ground truth | records | NR-14 | 3 | — |
| NR-16 | BAF acquisition & qualification | enable the second dataset | approval to download | registry entry (license undocumented) | acquired `base.csv` + license/semantics review + admission gates passed, **or** a documented rejection | no file downloaded before approval; admission result recorded either way | acquisition record | NR-13 | 3 | **download approval; license review** |
| NR-17 | Second-dataset replication | replicate criteria-1/5 results externally | NR-16 or a qualified alternative | qualified dataset | replication ledger records | same protocol, no re-tuning | records | NR-16 | 3 | NR-16 blocker |
| NR-18 | Feature-shift / availability analysis | quantify feature-distribution shift + availability across datasets | NR-04 | feature pipelines | shift/availability report | descriptive language only (no causal claims) | report + records | NR-04 | 3 | — |
| NR-19 | Label-delay & point-in-time correctness analysis | per-feature "available at scoring time" inventory | NR-05 | feature code, (later) bank data | inventory + delayed-label simulations on public data; bank-side plan | every feature gets an availability verdict; post-event information flagged | inventory artifact | NR-05 | 3 | bank data for full coverage |
| NR-20 | Feedback-loop bias analysis | assess outcome-feedback bias | NR-19 | verification outcome simulation | analysis report | simulations clearly labeled SIMULATED | report | NR-19 | 3 | bank data for real outcomes |

### GATE 4 — ENGINEERING (8 phases)

| ID | Title | Purpose | P | In | Out | Acc | Evd | Dep | Gate | Ext |
|---|---|---|---|---|---|---|---|---|---|---|
| NR-21 | PostgreSQL primary path | exercise production persistence | Gate 2 direction established (NR-13) | compose.prod, alembic 001 | Postgres as primary store behind config; migrations green | store data survives restart; SQLite dev path intact | migration+test evidence | NR-13 | 4 | — |
| NR-22 | TLS enforcement | HTTPS across environments | NR-21 (or parallel) | Caddyfile | TLS in compose/deploy configs | no plaintext production path documented | config + test evidence | NR-13 | 4 | certs/domains |
| NR-23 | Performance/concurrency/recovery testing | latency/throughput/degradation evidence on PostgreSQL | NR-21 | load harness | benchmark artifacts | p50/p95/p99, saturation behavior, DB-4-outage degradation measured | benchmark reports | NR-21 | 4 | — |
| NR-24 | Secrets, key management & internal-endpoint enforcement | close G-32 + G-05 | NR-13 | hardcoded secrets, attribution endpoint | env/KMS-based secrets; `/internal/attribution` gated with `enforce_before_inference` + `check_access` parity | no hardcoded runtime secrets; endpoint parity test passes; existing suites stay green | tests + config | NR-13 | 4 | — |
| NR-25 | Asymmetric signing evaluation (Ed25519) | adopt or formally reject | NR-24 | docstring recommendations, dependency risk | decision record + implementation if adopted | evaluation covers dependency/risk/migration; outcome documented either way | decision record + tests | NR-24 | 4 | — |
| NR-26 | Observability persistence | survive restarts | NR-13 | Phase 70/72 store | persisted metrics/alerts/security events | restart test demonstrates retention | test evidence | NR-13 | 4 | — |
| NR-27 | Audit-path correctness | k-anonymity export wiring + async-audit guarantee re-verification (+ G-34 assessment) | NR-13 | writer, checker | wired export gate; re-verified ordering/integrity under load | suites pass; write-path availability risk documented or mitigated | tests + report | NR-13 | 4 | — |
| NR-28 | Repository consolidation | execute Part Q list (incl. `fraud_report` legacy-artifact disposition, G-37 schema string, tracked .pyc, duplicate trail UIs) | NR-13 | Part Q candidates | consolidated tree with migration notes | no deletion without caller/test review; battery stays green | change log + battery tally | NR-13 | 4 | — |

### GATE 5 — EXTERNAL / INSTITUTIONAL (11 phases)

| ID | Title | Purpose | P | In | Out | Acc | Evd | Dep | Gate | Ext |
|---|---|---|---|---|---|---|---|---|---|---|
| NR-29 | Data-requirements spec + collaboration proposal (O1–2) | make the bank ask concrete | NR-13 result (know what to ask for) | feature contracts, 7-feature inventory | field-level spec + one-pager | every §16/7-feature field has source/owner/volume needs | documents | NR-13 | 5 | institution |
| NR-30 | In-bank evaluation architecture + permitted outputs (O3–4) | privacy-preserving in-bank design | NR-29 | architecture audit | design + exact aggregate list allowed to leave | approved by privacy officer eventually | design docs | NR-29 | 5 | privacy officer |
| NR-31 | Publication terms + external pre-registration (O5–6) | commit to publishing unfavourable results | NR-02 (internal protocol) | protocol v1.0 | externally shareable protocol + terms draft | explicitly covers Conclusions B and C | documents | NR-02 | 5 | counsel |
| NR-32 | Business-metric definitions + shadow-mode plan (O7–8) | operationalize evaluation | NR-12, NR-13 | Gate 2 metrics | definitions + shadow plan | capacity/cost inputs labeled by source | documents | NR-12 | 5 | bank SMEs |
| NR-33 | Model-risk documentation + explainability & HITL review (O9–11) | MRM-style package | NR-13 | model records, attribution endpoint | docs + analyst review record | reviewed by actual reviewers, not self-attested | reports | NR-13 | 5 | analysts |
| NR-34 | Retention/access/pseudonymization requirements (O12–14) | data governance spec | NR-29 | current privacy design | requirements doc | maps to actual implemented controls | document | NR-29 | 5 | privacy officer |
| NR-35 | Legal/privacy/compliance review (O15–16) | institutional sign-off | NR-29…NR-34 | full package | review outcomes | no compliance claim until signed | signed reviews | NR-34 | 5 | **counsel, privacy officer, compliance** |
| NR-36 | Independent penetration test | external security assurance | NR-24 (self-fixes first) | scoped system | third-party report + severity ratings | counters G-10 (pins executed scope); remediation workflow defined | external report | NR-24 | 5 | **external vendor** |
| NR-37 | Independent code & architecture review | external technical assurance | NR-13 (stable package) | repo snapshot | findings report incl. G-35 assessment | findings triaged | external report | NR-13 | 5 | **external reviewers** |
| NR-38 | Independent methodology & statistical review | validate the research design | NR-02 + Gate 2 results | protocol + records | findings report | reviewer sees raw records, not summaries | external report | NR-13 | 5 | **statistical reviewer** |
| NR-39 | Independent reproduction | third party replays baseline | NR-44 manual stable | `BASELINE_REPRODUCTION.md` | reproduction log by outsider | outsider reaches same metrics from commands | external log | NR-44 (manual can draft earlier) | 5 | **external reproducer** |

### GATE 6 — FINAL FREEZE (5 phases)

| ID | Title | Purpose | P | In | Out | Acc | Evd | Dep | Gate | Ext |
|---|---|---|---|---|---|---|---|---|---|---|
| NR-40 | Final repository audit & freeze | last forensic pass | Gates 2–5 outcomes | tree + evidence | final audit with claim classifications | every headline number still registry-backed at freeze | audit doc | NR-13+ | 6 | — |
| NR-41 | README one-screen rewrite | honest front page (also closes G-09/G-12) | NR-40 | registry + audit | one-screen structure: what/demonstrated/not-established/headline/limitations/evidence links | `claim_evidence_check` passes with expanded surfaces; no stale paths | README + check PASS | NR-40 | 6 | — |
| NR-42 | Presentation update | measured numbers only | NR-41 | `build_presentation.py` | regenerated deck; every number carries dataset+split+caveat | Results slide contains only ledger-backed values | deck + registry refs | NR-41 | 6 | — |
| NR-43 | Paper/thesis draft | publication package | NR-13 + Gate 3 results | all evidence | outline→draft incl. negative results, external degradation, limitations | no claim beyond classifications in §D | manuscript | NR-13 | 6 | — |
| NR-44 | Reproducibility package & release state | fresh-clone replay + release provenance (closes G-36/G-40 remainders) | NR-40 | manual, ledger, release manifest | package where a fresh clone + data distribution reproduces records; release manifest regenerated with git sha/seed (no weight changes) | outsider/CI replay succeeds; manifest fields non-null | replay log + manifest | NR-40 | 6 | data-distribution policy decision |

---

## K. Remaining phase count (derived, not inherited)

| Gate | Phases | IDs |
|---|---|---|
| 1 — Evidence foundation | **3** | NR-01…NR-03 |
| 2 — Research go/no-go | **10** | NR-04…NR-13 |
| 3 — Mechanism validation | **7** | NR-14…NR-20 |
| 4 — Engineering | **8** | NR-21…NR-28 |
| 5 — External/institutional | **11** | NR-29…NR-39 |
| 6 — Final freeze | **5** | NR-40…NR-44 |
| **Total** | **44** | |

Derivation from the old 48: four consolidations (R-05+R-03 → NR-02; R-02-remainder+R-16 →
NR-03; R-07+R-04 → NR-04; R-28+R-06 → NR-24) remove 4 net items; no genuinely new
standalone requirement was added (N-01/N-02 are folded into existing phases); one
reordering (R-06 Gate 1 → Gate 4) with stated rationale; one supersession chain
(G-19/R-03 → three-track scope decision). 48 − 4 = **44**. No requirement was split
(the smallest unit, NR-03, is already the minimum coherent bundle of
semantics+threshold discipline).

---

## L. Limitations (everything unresolved)

1. CI enforcement is still **locally verified only** — the GitHub runner has executed
   nothing from Phase 105 (blocked on commit/push authorization, NR-01).
2. The protocol remains **DRAFT**; all experiments stay forbidden until NR-02 sign-off;
   §H corrections are recorded but not applied to the file.
3. Phase-era external results (IBM v2, 23B, 24/25) remain **traced, not re-executed**;
   their SELF-TESTED/DEMONSTRATED classifications rest on artifact inspection.
4. Three gaps stay **BLOCKED** by externals (7-feature contract bank data, BAF
   acquisition, IEEE-CIS credentials); Gate 5 entirely so.
5. `fraud_report.py` success path remains untestable (no producer artifacts) —
   disposition deferred to NR-28.
6. Ledger count is live (hook-triggered identical re-runs append records; 11 at read
   time) — reports must cite it as a timestamped observation.
7. This phase changed no model, dataset, threshold, methodology, or evidence; no claim
   was upgraded beyond Phase 105's classifications; the protocol file was not edited.
