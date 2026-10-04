# Research Gate Readiness — Can the preregistered comparative experiments begin?

**Date:** 2026-10-03 · **Phase:** 106 (read-only reconciliation) ·
**Companion to:** `docs/PHASE_106_ROADMAP_RECONCILIATION.md` (Doc 1 §I).

This document answers exactly one question:

> **Is the repository's evidence foundation sufficiently trustworthy to begin the
> preregistered comparative research experiments (Gate 2)?**

It makes **no performance judgment** about any model, ensemble, or approach, and
assigns no ranking. Every status below concerns *evidence infrastructure*, not
model quality.

Status classes: **READY** · **PARTIAL** · **NOT READY** · **BLOCKED**.

---

## Readiness table

| Requirement | Status | Evidence | Blocking Issue |
|---|---|---|---|
| Reproducibility | **PARTIAL** | Bit-identical replays achieved for ULB (roc 0.975807 / pr 0.883465 / r@1% 0.918367, ledger `…e58d68c2bf88`), calibration (Brier 0.0009280 / ECE 0.0012790, ledger `…72d714bd81b3`), IBM cross-dataset 0.8726 (ledger `…e2a14ae94bef`), synthetic fused train_compare (ledger `…359039aac6cf`); `BASELINE_REPRODUCTION.md` documents commands | Historical phase-era results (IBM v2 table, phase 23B/24/25) are traced, not re-executed; fresh-clone artifact distribution policy undecided (G-40) — NR-44 |
| Provenance | **PARTIAL** | Schema v1.1 records carry command, seed, dataset hash, git SHA; chain command→artifact→record demonstrated for Phase 105 runs; release-manifest regeneration still pending (G-36) | Deployed release manifest still `source_git_sha: unknown`, null seed, empty fingerprints — NR-44 (Gate 6; does not block Gate 2) |
| Evidence ledger | **READY** | `reports/evaluation_runs/eval_ledger.jsonl`: 1→11 records this phase, append-only, 0 rejected, 0 fabricated; `train_compare.py` `gate_rows` append bug fixed (never worked since Phase 39); `eval_record_test.py` 22/22 | None for beginning experiments; live growth from hook re-runs is documented, deterministic, and benign |
| CI enforcement | **PARTIAL** | `.github/workflows/ci-cd.yml` test job runs `claim_evidence_check.py` + `eval_record_test.py`; local valid-PASS / invalid-FAIL(rc=1) / 7-fixture self-test all demonstrated | Never executed on the GitHub runner; evidence-commit coupling unproven (N-01) — NR-01 |
| Dataset availability | **PARTIAL** | ULB (hash-verified), IBM v2, Kaggle fraudTest all present and previously executed; external 0.595 artifact-backed | NeurIPS 2022 BAF not acquired (license undocumented, download approval required → BLOCKED, NR-16); IEEE-CIS auth-blocked. **Partial is sufficient:** the preregistered comparisons can run on available datasets; BAF blocks only the second-dataset replication track (NR-17) |
| Feature availability | **PARTIAL** | Three-track contract decomposition (Doc 1 §E Step-9): 48-native production contract evaluable on Kaggle fraud (phase24/25 reconstruction proven); 21-domain/public_v1 contract evaluable on IBM v2 and public data | 21+7 full contract BLOCKED on institution data (7 features absent from every public dataset, G-20) — does not block the two evaluable tracks; contract file declares 14 features but lists 13 (N-02) — fixed as NR-04 pre-run check |
| Protocol state | **NOT READY** | `PRE_REGISTERED_EVALUATION_PROTOCOL.md` exists and its decision rules are intact (MMD proposals, CI/replication rules, conclusions A/B/C, threshold procedure, `refuse_test_tuning` guard verified, locked S1 threshold 0.018758) | Still DRAFT/NOT APPROVED: 11 `[REQUIRES DECISION/APPROVAL]` markers + 4 factual corrections from Doc 1 §H (0.435→0.595; R-cross-refs → NR-IDs; equal-mass vs equal-width ECE bins; three-track §0 scope) — NR-02 |
| Baseline availability | **READY** | Reproduced, ledger-bound baselines exist: ULB pattern-XGB row, synthetic fused train_compare row, calibration row, IBM cross-dataset row; de-hardcoded `compare_ml_systems.py` reads `reports/ulb_results.json` (N/A if absent) | The historical README tuple (0.966/0.877/97.8%@0.43/1.996%) stays NOT ESTABLISHED and is excluded from use as a baseline (C-101…C-104) |
| Calibration | **PARTIAL** | In-domain/validation-split calibration DEMONSTRATED (Brier 0.0009280 / ECE 0.0012790, n=1470, ledger-bound); README 0.0009/0.0013 confirmed as rounded | External calibration never evaluated (→ NR-09); ECE bin semantics unresolved — protocol says equal-mass [DECISION], implementation uses equal-width (Doc 1 §H.3) — resolved by NR-02/NR-03 before experiments |
| Leakage controls | **PARTIAL** | Structural leakage suite 123/123 in battery evidence; performance-measurement isolation (`refuse_test_tuning`) verified; frozen-test discipline partially enforced by schema | Statistical leakage audit not run (duplicates, full-frame aggregates `merchant_user_count`/`city_user_count`, repeated historical tuning exposure — G-17); no untouched pre-registered final-test set exists yet (G-04) — NR-03 + NR-10 |
| Threshold discipline | **PARTIAL** | Train→validation→frozen→single-use-test procedure specified in protocol; `refuse_test_tuning` guard verified; locked S1 threshold 0.018758 artifact-backed; reproduced runs used validation-only thresholds | Tooling does not yet *enforce* the frozen-test boundary (no frozen-test artifact, no metric-definitions v1.1) — R-16 must precede experiments, merged into NR-03 |
| Negative-result handling | **READY** | Protocol conclusions A/B/C explicitly cover negative outcomes; CI/replication rules and "no complexity added to force a positive result" are pre-specified; Phase 105 demonstrated the practice (README tuple honestly classified NOT ESTABLISHED rather than repaired) | None. Final wording locks at NR-02 sign-off (conclusions themselves unchanged) |

---

### READY

What can legitimately begin now:

- **Gate 1 completion work** — the evidence foundation is substantially built and the
  remaining steps are internal:
  - **NR-01**: verify CI enforcement on the actual GitHub runner and extend the local
    battery to cover both new suites (needs commit/push authorization).
  - **NR-02**: apply the four recorded factual corrections, decide the evaluated
    contract-track scope, resolve the 11 approval markers, sign the protocol
    (needs reviewers).
  - **NR-03**: `metric_definitions` v1.1 (Brier, chosen ECE bin semantics) and the
    machine-checkable frozen final-test design.
- **Assembling Gate 2 inputs that do not depend on the protocol** — harness scaffolding
  around the existing `eval_record` v1.1 machinery, dataset hash registration, and the
  N-02 contract-count check may be built and tested against the ledger before sign-off,
  as long as **no comparative experiment is executed** before NR-02/NR-03 land.

### NOT READY

What must be fixed first (all internal — none external):

1. **Protocol approval (NR-02)** — DRAFT with 11 approval markers and 4 factual
   corrections (0.435 prose-only; superseded R-cross-references; ECE equal-mass vs
   equal-width; three-track §0 scope). Until signed, any experiment would not be
   preregistered.
2. **Metric semantics + frozen-test design (NR-03)** — `metric_definitions.py` is still
   v1.0 with no Brier/ECE definitions, and no untouched pre-registered final-test set
   exists (G-04, G-06). Threshold discipline is specified but not tooling-enforced.
3. **Preregistered harness (NR-04)** — no single ledger-writing pipeline exists for
   Gate-2 runs; IBM v2 results were never re-executed with seed/git (R-04 remainder).
   The N-02 contract-count inconsistency must be fixed as a pre-run check.
4. **CI runner verification (NR-01)** — enforcement is locally proven only; a green
   GitHub runner run (with evidence committed alongside README) is required before the
   enforcement claim is more than self-tested.
5. **Statistical leakage audit scope (NR-10)** — can begin at Gate 2 but its known
   items (full-frame aggregates, repeated-tuning exposure) must be on the run checklist.

### BLOCKED

What requires an external dependency (none of which blocks starting Gate 1 or the
available-data portion of Gate 2):

| Item | Requirement | External dependency |
|---|---|---|
| NeurIPS 2022 BAF acquisition (second-dataset replication) | NR-16 → NR-17 | Download approval + license/semantics review |
| IEEE-CIS (optional alternate external dataset) | conditional (G-28) | Credentials; only if pursued |
| 21+7 full-contract evaluation (7 bank-only features) | NR-19, NR-29+ | Institution-provided data/environment |
| Realized business-metric numbers (beyond illustrative curves) | NR-12 partial, NR-32 | Bank SME input |
| All Gate 5 assurance items (pentest, code review, methodology review, reproduction, legal) | NR-35…NR-39 | External parties / counsel |

---

## Verdict

**Current research gate: PARTIAL.**

The evidence foundation is sufficiently trustworthy to **finish Gate 1** — provenance,
the ledger, baseline availability, and negative-result handling are READY; the ledger
and reproducibility machinery have demonstrably worked end-to-end this phase. It is
**not yet sufficient to begin** the preregistered comparative experiments: the protocol
is unapproved, metric semantics and the frozen-test boundary are unfixed, and the
harness does not exist.

**Exact prerequisites to open Gate 2 (all internal):** NR-01 (runner-verified
enforcement) → NR-02 (signed protocol with the four corrections and a contract-track
scope) → NR-03 (metric definitions v1.1 + frozen-test artifact) → NR-04 (harness).
The first blocked external item (BAF) affects only the second-dataset track and does
not gate this decision.

No model performance claim is made or implied anywhere in this document.
