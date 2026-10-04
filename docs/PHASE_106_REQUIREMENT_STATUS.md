# Phase 106 — Requirement Status Register

**Date:** 2026-10-03 · **Companion to:** `docs/PHASE_106_ROADMAP_RECONCILIATION.md`
(Document 1). Every Phase 104 gap and roadmap requirement receives an explicit
disposition; the new authoritative requirements follow.

Status classes (exact, per Phase 106 Step 5): **CLOSED** · **PARTIALLY CLOSED** ·
**OPEN** · **BLOCKED** · **SUPERSEDED** · **DUPLICATE**.
Evidence cites Phase 105 artifacts only (see Document 1 §C–§D).

---

## 1. Phase 104 gaps — G-01…G-45

| Old ID | Description | Status | Evidence | Replacement ID | Notes |
|---|---|---|---|---|---|
| G-01 | CI cannot reject "metric in docs without evidence/run record" | PARTIALLY CLOSED | `claim_evidence_check.py` + `eval_record_test.py` added to `.github/workflows/ci-cd.yml`; local valid-PASS / invalid-FAIL / 7-fixture self-test proven | NR-01 | Remaining: never run on GitHub runner; evidence-commit coupling unproven; local battery omits both suites |
| G-02 | README headline tuple 0.966/91.8/0.877/97.8%@0.43 has no provenance artifact | CLOSED | Tuple classified NOT ESTABLISHED (C-101…C-104), README corrected to reproduced 0.976/0.918/0.883 with ledger ids; `compare_ml_systems.py` de-hardcoded | — | Retain closure |
| G-03 | Brier 0.0009 / ECE 0.0013 not found in any measurement artifact | CLOSED | Reproduced exactly: 0.0009280 / 0.0012790, n=1470 validation split, ledger `…72d714bd81b3` | — | Phase 104's NOT ESTABLISHED verdict overturned (Doc 1 §C.2) |
| G-04 | No untouched, pre-registered final test set exists | OPEN | Repeated historical tuning on same ULB splits unchanged | NR-03 | Frozen-test design required before any experiment |
| G-05 | `/internal/attribution` bypasses `enforce_before_inference` and `check_access` | OPEN | Endpoint untouched this phase | NR-24 | Moved Gate 1 → Gate 4: does not affect research validity |
| G-06 | Metric definitions lack Brier/ECE/reliability diagrams | OPEN | `metric_definitions.py` still v1.0 (Phase 105 deliberately scoped out) | NR-03 | Folded with threshold-discipline tooling |
| G-07 | `CLAIMS_MATRIX.md` M5 repeats train_compare misattribution | CLOSED | M5 → UNSUPPORTED with corrected evidence path; M6 downgraded to PARTIAL | — | Retain closure |
| G-08 | README claims "k-anonymity gate on data exports" — not wired | CLOSED (as stated) | README wording corrected to match runtime (checker is test/CLI-only) | NR-27 | The *missing wiring* itself remains and is tracked separately |
| G-09 | README claims missing files (pptx, ACCESS.md.enc, deploy.sh paths) | OPEN | Paths still claimed in README | NR-41 | Bundled into README one-screen rewrite |
| G-10 | Penetration vector count discrepancy (42 vs 34 vs 66+43) | PARTIALLY CLOSED | README now documents the discrepancy instead of asserting 42 | NR-36 | Executed count still unpinned; final resolution with external pen test |
| G-11 | Battery omits eval_record/claim audits; battery self-reported | PARTIALLY CLOSED | CI runs both suites; `.freebuff/p114_battery.sh` still does not | NR-01 | Local battery coverage is the remainder |
| G-12 | README stale vs HEAD (no repo phases 104–118; malformed row ~L251) | OPEN | Phase table still stops at 56; malformed row untouched | NR-41 | |
| G-13 | Observability lost on restart (in-memory) | OPEN | Unchanged this phase | NR-26 | |
| G-14 | PostgreSQL path unexercised (compose.prod + migration 001 only) | OPEN | Unchanged this phase | NR-21 | |
| G-15 | No TLS in dev/preview compose | OPEN | Unchanged this phase | NR-22 | |
| G-16 | No load/concurrency/recovery test artifacts | OPEN | Unchanged this phase | NR-23 | |
| G-17 | Possible future-information in full-frame aggregates; unknown dedup status | OPEN | `ibm_train.py:180–188` aggregates untested | NR-10 | Statistical leakage audit scope |
| G-18 | IBM v2 split user-disjoint but NOT chronological | OPEN | Split code unchanged | NR-05 | |
| G-19 | OPTION A (public_v1) vs OPTION B (defer to bank) undecided | SUPERSEDED | Binary framing dissolved by evidence-based three-contract decomposition (Doc 1 §E Step-9) | NR-02 (+ NR-04/05/06 design) | Human scope decision still required — now three-track, not two-option |
| G-20 | 7 bank-only features unusable on public data | BLOCKED | Absent from every public dataset (phase64); `privacy_layer` untouched by Phase 105 | NR-19 (+ NR-29+ bank branch) | External: institution data |
| G-21 | No UNGATED-vs-GATED experiment | OPEN | No artifact exists | NR-14 | |
| G-22 | Ensemble superiority never tested under pre-registration | OPEN | No ablation-suite artifact | NR-07, NR-08 | |
| G-23 | Calibration never validated externally | OPEN | In-domain calibration now DEMONSTRATED; external calibration still never evaluated | NR-09 | Upgraded evidence on the in-domain half only |
| G-24 | No subgroup/segment analysis on evaluated models | OPEN | Unchanged this phase | NR-11 | |
| G-25 | No business metrics (alert volume, burden, cost) | OPEN | Unchanged this phase | NR-12 | |
| G-26 | Label delay / point-in-time inventory not built | OPEN | Unchanged this phase | NR-19 | |
| G-27 | BAF not acquired; license/semantics undocumented | BLOCKED | `data/external_benchmark/baf/` absent | NR-16 | External: download approval + license review |
| G-28 | IEEE-CIS auth-blocked | BLOCKED | Credential wall unchanged | (conditional) | Only if pursued; no NR scheduled unless chosen |
| G-29 | Only 1 external-transfer dataset family exercised | OPEN | Still Kaggle fraud + same-generator IBM | NR-17 | |
| G-30 | Feedback-loop bias not analyzed | OPEN | Unchanged this phase | NR-20 | |
| G-31 | Feature-shift/availability analysis not done | OPEN | Unchanged this phase | NR-18 | |
| G-32 | Hardcoded HMAC secrets (`_MANIFEST_SECRET`, `__TOKEN_SECRET`) | OPEN | Still hardcoded | NR-24 | |
| G-33 | No asymmetric signing (Ed25519) despite docstring recommendations | OPEN | No ed25519 code anywhere | NR-25 | |
| G-34 | Audit writes synchronous on critical path | OPEN | Writer usage unchanged | NR-27 | |
| G-35 | Hash chain cannot detect forged appends | OPEN | Part N analysis unchanged | NR-37 | Assessment folded into independent review scope |
| G-36 | Release manifest: `source_git_sha: "unknown"`, null seed, empty fingerprints | OPEN | Phase 105 completed provenance only for *new* evaluation records | NR-44 | Regenerate manifest without touching weights |
| G-37 | Stale `altman_native_v2` schema string in model record | OPEN | Unchanged this phase | NR-28 | |
| G-38 | Repository sprawl (misc 719 files, tracked .pyc, duplicate trail UIs) | OPEN | Unchanged this phase | NR-28 | |
| G-39 | 45/P20 feature path historical/inactive | CLOSED | Documented as historical; required no work by design | — | Retain closure |
| G-40 | Model artifacts gitignored — fresh clone cannot reproduce deployed model | PARTIALLY CLOSED | Fresh-clone blocker now documented in `BASELINE_REPRODUCTION.md` §7 | NR-44 | Distribution/artifact policy decision remains |
| G-41 | 16 bank-readiness deliverables missing/partial | OPEN | Protocol still a draft | NR-29…NR-35 | |
| G-42 | Zero external assurance items completed | OPEN | Phase 105 changed nothing here | NR-36…NR-39 | |
| G-43 | No legal/privacy/compliance review | OPEN | Unchanged this phase | NR-35 | |
| G-44 | Pre-registration not approved | OPEN | Protocol still DRAFT/NOT APPROVED (11 markers) | NR-02 | |
| G-45 | No paper/thesis, no reproducibility package, presentation lacks numbers | OPEN | Unchanged this phase | NR-40…NR-44 | |

**Totals: CLOSED 5 (G-02, G-03, G-07, G-08, G-39) · PARTIALLY CLOSED 4 (G-01, G-10, G-11, G-40) · OPEN 32 · BLOCKED 3 (G-20, G-27, G-28) · SUPERSEDED 1 (G-19) · DUPLICATE 0 = 45.**
*(G-08 counted as CLOSED as-stated; its wiring remainder lives in NR-27 — identical accounting to Document 1 §E.)*

---

## 2. Phase 104 roadmap requirements — R-01…R-48

| Old ID | Description | Status | Evidence | Replacement ID | Notes |
|---|---|---|---|---|---|
| R-01 | Evidence-ledger expansion + CI claim-enforcement | PARTIALLY CLOSED | Ledger 1→11 records ✓; CI enforcement ✓ (local); battery coverage ✗ | NR-01 | Remainder only |
| R-02 | Headline-metric provenance reconciliation + `metric_definitions` v1.1 + CLAIMS_MATRIX | PARTIALLY CLOSED | Headline reconciliation ✓ (ledger-bound); CLAIMS_MATRIX ✓; `metric_definitions` v1.1 ✗ | NR-03 | Remainder folded with R-16 |
| R-03 | Feature-contract decision (OPTION A vs B) | SUPERSEDED | Binary framing dissolved by three-track contract decomposition (Doc 1 §E Step-9) | NR-02 | Human decision still required — now a scope decision across contract tracks |
| R-04 | Reproduction of existing headline results into ledger records | PARTIALLY CLOSED | ULB, calibration, cross-dataset, synthetic train_compare reproduced into ledger; ibm_train not re-executed (trace-only, no seed/git in artifact) | NR-04 | IBM upgrade folded into harness build |
| R-05 | Pre-registration review & approval | OPEN | Protocol still DRAFT, 11 approval markers | NR-02 | Merged with R-03's scope decision |
| R-06 | Internal-endpoint security closure (`/internal/attribution` gating) | OPEN | Bypass unchanged | NR-24 | **Moved Gate 1 → Gate 4:** research validity before production hardening |
| R-07 | Independent benchmark harness | OPEN | No preregistered harness exists | NR-04 | Absorbs R-04 remainder |
| R-08 | Temporal-split evaluation | OPEN | No temporal evaluation under preregistration | NR-05 | |
| R-09 | Entity-disjoint evaluation | OPEN | Exists only ad hoc in `ibm_train` | NR-06 | |
| R-10 | Baselines & ablations suite | OPEN | No suite artifact | NR-07 | |
| R-11 | Ensemble-contribution experiment (criteria 1/2) | OPEN | Never tested | NR-08 | |
| R-12 | External calibration evaluation | OPEN | Never evaluated | NR-09 | |
| R-13 | Statistical leakage audit | OPEN | Structural suite ≠ statistical audit | NR-10 | |
| R-14 | Subgroup/segment analysis | OPEN | Not done | NR-11 | |
| R-15 | Business-relevant metrics | OPEN | Not done | NR-12 | |
| R-16 | Threshold-discipline enforcement tooling | SUPERSEDED | Tooling must exist **before** any experiment, not as a mid-Gate-2 phase | NR-03 | Merged with R-02 remainder |
| R-17 | Gate 2 GO/NO-GO report vs pre-registered criteria | OPEN | Not run | NR-13 | |
| R-18 | UNGATED-vs-GATED controlled experiments | OPEN | No experiment exists | NR-14 | |
| R-19 | Drift & feature-corruption robustness validation | OPEN | Unvalidated | NR-15 | |
| R-20 | BAF acquisition & qualification | BLOCKED | Not acquired; license undocumented | NR-16 | External: download approval |
| R-21 | Second-dataset replication experiment | OPEN (dependent) | Needs R-20/NR-16 or a qualified alternative | NR-17 | |
| R-22 | Feature-shift / feature-availability analysis | OPEN | Not done | NR-18 | |
| R-23 | Label-delay & point-in-time correctness analysis | OPEN | Not built | NR-19 | Bank-heavy; public-data portion schedulable |
| R-24 | Feedback-loop bias analysis | OPEN | Not analyzed | NR-20 | |
| R-25 | PostgreSQL primary path + migrations | OPEN | Unexercised | NR-21 | |
| R-26 | TLS enforcement across environments | OPEN | Not in compose | NR-22 | |
| R-27 | Performance/concurrency/recovery load testing | OPEN | No load tests | NR-23 | |
| R-28 | Secrets & key management | OPEN | Hardcoded HMAC secrets remain | NR-24 | Now also absorbs R-06 |
| R-29 | Ed25519 asymmetric signing evaluation | OPEN | No evaluation exists | NR-25 | |
| R-30 | Observability persistence across restart | OPEN | In-memory store remains | NR-26 | |
| R-31 | Audit-path correctness (k-anon wiring + async-audit re-verification) | OPEN | Wiring absent; re-verification pending | NR-27 | |
| R-32 | Repository consolidation (Part Q) | OPEN | Sprawl unchanged | NR-28 | Also dispositions `fraud_report` legacy artifacts + G-37 |
| R-33 | Data-requirements spec + collaboration proposal (O1–2) | OPEN | Missing | NR-29 | |
| R-34 | In-bank evaluation architecture + permitted outputs (O3–4) | OPEN | Missing | NR-30 | |
| R-35 | Publication terms + external pre-registration (O5–6) | OPEN | Missing | NR-31 | |
| R-36 | Business-metric definitions + shadow-mode plan (O7–8) | OPEN | Missing | NR-32 | |
| R-37 | Model-risk documentation + explainability & HITL review (O9–11) | OPEN | Missing/partial | NR-33 | |
| R-38 | Retention/access/pseudonymization requirements (O12–14) | OPEN | Missing | NR-34 | |
| R-39 | Legal/privacy/compliance review (O15–16) | OPEN | Not started | NR-35 | Blocked on counsel/compliance at execution |
| R-40 | Independent penetration test | OPEN | Not performed | NR-36 | Blocked on external party at execution |
| R-41 | Independent code & architecture review | OPEN | Not performed | NR-37 | Blocked on external party at execution |
| R-42 | Independent methodology & statistical review | OPEN | Not performed | NR-38 | Blocked on external party at execution |
| R-43 | Independent reproduction | OPEN | Not performed | NR-39 | Partly enabled by Phase 105 manual; blocked on external reproducer |
| R-44 | Final repository audit & freeze | OPEN | Not done | NR-40 | |
| R-45 | README one-screen rewrite | OPEN | Phase 105 made targeted corrections only | NR-41 | Also carries G-09/G-12 |
| R-46 | Presentation update (measured numbers only) | OPEN | Lacks measured numbers | NR-42 | |
| R-47 | Paper/thesis draft | OPEN | Not started | NR-43 | |
| R-48 | Reproducibility package (fresh-clone → ledger replay) | OPEN | Not built; G-36/G-40 policy pending | NR-44 | |

**Totals: PARTIALLY CLOSED 3 (R-01, R-02, R-04) · SUPERSEDED 2 (R-03, R-16) · OPEN 42 · BLOCKED 1 (R-20) · CLOSED 0 · DUPLICATE 0 = 48.**
No requirement is fully CLOSED: Phase 105 completed the *substance* of R-01/R-02/R-04
with precisely defined remainders.

---

## 3. New requirements — NR-01…NR-44

Full 11-field rows (incl. inputs/output/evidence/dependency/external) live in
Document 1 §J; this table is the machine-readable summary.

| New ID | Requirement | Gate | Prerequisite | Acceptance Criterion |
|---|---|---|---|---|
| NR-01 | Evidence enforcement verification & battery coverage (folds N-01) | 1 | commit/push authorization | Green GitHub CI run shows both new steps executed; battery script runs `eval_record_test` + `claim_evidence_check`; evidence files committed with README |
| NR-02 | Pre-registration factual correction, contract-scope decision & sign-off (folds R-03, R-05, G-19, G-44; carries §H corrections) | 1 | NR-01 (parallelizable if push withheld); reviewers | Protocol leaves DRAFT; all 11 approval markers resolved or explicitly deferred with owner; §0 names the evaluated contract track(s) |
| NR-03 | Evaluation semantics & frozen final-test design (folds R-02-remainder, R-16; closes G-04, G-06) | 1 | NR-02 | `metric_definitions` v1.1 defines Brier/ECE (chosen bin semantics); machine-checkable frozen-test artifact exists, is protocol-referenced, and dry-run shows test-access detection |
| NR-04 | Preregistered benchmark harness + IBM v2 ledger upgrade (folds R-04-remainder, R-07; fixes N-02 contract count) | 2 | NR-02, NR-03 | Every harness run yields a DEMONSTRATED-grade ledger record; contract-count check passes; chain command→artifact→record→registry demonstrated |
| NR-05 | Temporal-split evaluation | 2 | NR-04 | Splits pre-declared in frozen-test design; no test-set inspection before threshold freeze; results ledgered |
| NR-06 | Entity-disjoint evaluation | 2 | NR-04 | Same discipline as NR-05; both generalization axes reported separately |
| NR-07 | Baselines & ablations suite | 2 | NR-04 | Baselines chosen on validation only; every matrix cell ledgered |
| NR-08 | Ensemble-contribution experiment (criteria 1/2) | 2 | NR-07, NR-02 criteria | Executed exactly per approved criteria; result reported whether positive or negative |
| NR-09 | External calibration evaluation | 2 | NR-04, NR-03 | Calibration separated from threshold selection; Brier/ECE on external splits with CIs, ledgered |
| NR-10 | Statistical leakage audit (closes G-17) | 2 | NR-04 | Every check PASS/FAIL/PARTIAL with evidence; FAILs become tracked gaps |
| NR-11 | Subgroup/segment analysis | 2 | NR-05/06 | Per-segment metrics with CIs; no invented segments; small-n flagged |
| NR-12 | Business-relevant metrics (public-data portion) | 2 | NR-05 | Metric curves parameterized by cost ratio; no realized financial claim |
| NR-13 | Gate 2 GO/NO-GO report | 2 | NR-08…NR-12 | Judged strictly against signed criteria; Conclusion A/B/C or inconclusive-with-next-experiment; negative results preserved |
| NR-14 | UNGATED-vs-GATED experiments (criteria 3/4) | 3 | NR-13 | Executed per approved criteria; implementation-only claims stay forbidden until then |
| NR-15 | Drift & corruption robustness | 3 | NR-14 | Synthetic drift injections with known ground truth produce detection/response evidence |
| NR-16 | BAF acquisition & qualification | 3 | download approval | No file downloaded before approval; acquisition record or documented rejection, either way recorded |
| NR-17 | Second-dataset replication | 3 | NR-16 or qualified alternative | Same protocol, no re-tuning; replication records ledgered |
| NR-18 | Feature-shift / availability analysis | 3 | NR-04 | Shift/availability report in descriptive language only (no causal claims) |
| NR-19 | Label-delay & point-in-time correctness analysis | 3 | NR-05 | Every feature gets an availability verdict; post-event information flagged; bank-side plan defined |
| NR-20 | Feedback-loop bias analysis | 3 | NR-19 | Simulations clearly labeled SIMULATED; report produced |
| NR-21 | PostgreSQL primary path | 4 | NR-13 | Store data survives restart; SQLite dev path intact; migrations green |
| NR-22 | TLS enforcement | 4 | NR-21 (or parallel); certs/domains | No plaintext production path documented; config + test evidence |
| NR-23 | Performance/concurrency/recovery testing | 4 | NR-21 | p50/p95/p99, saturation behavior, DB-4-outage degradation measured |
| NR-24 | Secrets, key management & internal-endpoint enforcement (folds R-06, R-28; closes G-05, G-32) | 4 | NR-13 | No hardcoded runtime secrets; `/internal/attribution` gated with parity test passing; suites stay green |
| NR-25 | Asymmetric signing evaluation (Ed25519) | 4 | NR-24 | Decision record covers dependency/risk/migration; adoption or documented rejection |
| NR-26 | Observability persistence | 4 | NR-13 | Restart test demonstrates retention of metrics/alerts/security events |
| NR-27 | Audit-path correctness (k-anon export wiring + async-audit re-verification) | 4 | NR-13 | Suites pass; write-path availability risk documented or mitigated |
| NR-28 | Repository consolidation (folds G-37, fraud_report disposition) | 4 | NR-13 | No deletion without caller/test review; battery stays green; change log produced |
| NR-29 | Data-requirements spec + collaboration proposal (O1–2) | 5 | NR-13 | Every §16/7-feature field has source/owner/volume needs |
| NR-30 | In-bank evaluation architecture + permitted outputs (O3–4) | 5 | NR-29 | Design + exact aggregate list allowed to leave |
| NR-31 | Publication terms + external pre-registration (O5–6) | 5 | NR-02 | Explicitly covers Conclusions B and C |
| NR-32 | Business-metric definitions + shadow-mode plan (O7–8) | 5 | NR-12, NR-13 | Capacity/cost inputs labeled by source |
| NR-33 | Model-risk documentation + explainability & HITL review (O9–11) | 5 | NR-13 | Reviewed by actual reviewers, not self-attested |
| NR-34 | Retention/access/pseudonymization requirements (O12–14) | 5 | NR-29 | Maps to actual implemented controls |
| NR-35 | Legal/privacy/compliance review (O15–16) | 5 | NR-29…NR-34 | No compliance claim until signed |
| NR-36 | Independent penetration test | 5 | NR-24 | Third-party report with severity ratings; pins executed scope (G-10 remainder) |
| NR-37 | Independent code & architecture review (incl. G-35 assessment) | 5 | NR-13 | Findings report triaged |
| NR-38 | Independent methodology & statistical review | 5 | NR-02 + Gate 2 results | Reviewer sees raw records, not summaries |
| NR-39 | Independent reproduction | 5 | NR-44 manual (draftable earlier) | Outsider reaches same metrics from commands |
| NR-40 | Final repository audit & freeze | 6 | NR-13+ | Every headline number still registry-backed at freeze |
| NR-41 | README one-screen rewrite (closes G-09, G-12) | 6 | NR-40 | `claim_evidence_check` passes with expanded surfaces; no stale paths |
| NR-42 | Presentation update (measured numbers only) | 6 | NR-41 | Results slide contains only ledger-backed values |
| NR-43 | Paper/thesis draft | 6 | NR-13 + Gate 3 | No claim beyond Document 1 §D classifications |
| NR-44 | Reproducibility package & release state (closes G-36, G-40 remainders) | 6 | NR-40 | Fresh-clone replay succeeds; release manifest fields non-null (git sha/seed) regenerated without weight changes |

**Count: 44** (48 old − 4 consolidations; 2 new gaps N-01/N-02 folded into existing
phases; derivation in Document 1 §K).
