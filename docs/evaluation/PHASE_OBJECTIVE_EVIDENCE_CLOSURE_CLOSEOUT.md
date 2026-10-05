# Phase Closeout — Objective Evidence Closure & Freeze-Readiness Preparation

**Artifact path:** `docs/evaluation/PHASE_OBJECTIVE_EVIDENCE_CLOSURE_CLOSEOUT.md`
**Version:** `phase-objective-evidence-closure-closeout 1.0-draft`
**Phase status:** **`OBJECTIVE EVIDENCE CLOSED — AWAITING HUMAN REVIEW`**
**Date:** 2026-10-05

> **The repository's objective evidence work is closed.** Every currently unresolved
> item is classified; every finding the repository can resolve from authoritative
> evidence alone has been measured and recorded; nothing further can move without a
> qualified reviewer, authoritative external evidence, institutional access, or a
> deliberate freeze act. **Review has not commenced:** statistical reviewer
> `NOT ASSIGNED`, domain reviewer `NOT ASSIGNED`, independent auditor not assigned,
> licence review not completed, **0/14** decisions resolved.

---

## 1. Files created / modified

| File | Action | Purpose |
|---|---|---|
| `docs/evaluation/OBJECTIVE_EVIDENCE_CLOSURE.md` | **created** (290 lines) | The audit: U-1…U-10 and O-1…O-8 classified; R-1…R-7 measured/prepared; 50M determination; 65 plan + 12 prereg markers classified; §35 (36) grouped; reviewer handoff matrix |
| `docs/evaluation/PHASE_OBJECTIVE_EVIDENCE_CLOSURE_CLOSEOUT.md` | **created** (this file) | Phase closeout (cannot carry its own hash — plan §29 convention) |

No other file was changed: no checker, dataset, model, threshold, plan, preregistration,
metric definition, review record, ledger record (written by this phase), or freeze
state. Battery side effects are enumerated in §11.

---

## 2. Exact hashes

| Artifact | SHA-256 |
|---|---|
| `docs/evaluation/OBJECTIVE_EVIDENCE_CLOSURE.md` | `5b5dfa9ae4fb075750d4d10ce4fa7e85821b3a242aa21768f317dcc32dfcb4c3` |

**Invariants re-measured after all edits and after the battery — all unchanged:**

| Invariant | SHA-256 |
|---|---|
| `docs/RESEARCH_PLAN.md` | `eab5a0615801db7e50ab4d1c0ec73f93905e461b5bd8eacc617e0418d656b9ce` |
| `docs/metric_definitions.md` | `8665549b3d62fef36402d737910cd8aa1acfbde025dfee4aa1395a9e3d99ed07` |
| `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` | `fdde71cbd00e8e34ce439f3cc582e31e73452a0405ea3ff51c11136adc9065a6` |
| `docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` | `17a95e2afc18b1dd2b189aa307f90b79ed1e03701e51f37727e2754ca2d42d24` |
| `docs/evaluation/STATISTICAL_REVIEW_DECISION_MEMO.md` | `b99c32f49561a7ed2f764ab3cb3a48843e13968fb103bc89be6dcec74bd0131d` |
| `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` | `13f46ed218bb9c4c9b164ee81a72b33bb13c31b1476bd747cfa45c2e97e513ad` |
| `docs/evaluation/DATASET_EXPOSURE_LEDGER.md` | `31af86fd7c089c6242ab27b28b3ecdf03e3bbe616944522f0e3074caeec6a501` |
| `docs/evaluation/DATASET_PROVENANCE_RECONCILIATION.md` | `da73e40b6bb566381dd01bed9aafde89bb0f90549b07e5131625aba016876be3` |
| `docs/evaluation/DATASET_48_FEATURE_COVERAGE.md` | `83397f937055349ce74d67d0dbefa13da549aea9537ea2ed79d03e3fc1a3ad92` |
| `docs/evaluation/DATASET_LICENCE_EVIDENCE_CHECKLIST.md` | `bfb6dd3d977e426f66bd74f84d9d6fdbc8036456fe55e2d6d533647b7c8acd6a` |
| `docs/evaluation/INDEPENDENT_EVIDENCE_REVIEW_PACKAGE.md` | `0b249652fc78fa02321295790d2ebfbfd79466a2ed06b8402daaa22d70280627` |
| `docs/evaluation/REVIEWER_ENGAGEMENT_BRIEF.md` | `bc1568fa55c6ff3ac5b8cc5d0f3b226894abdb5a7a06c6d1c37183829527e5d3` |
| `docs/evaluation/REVIEWER_ASSIGNMENT_RECORD.md` | `b62e4f054523257cbc8ab1b752b65e21fed2a1da92605674f4da4e4a31a00d3e` |
| `docs/evaluation/PHASE_REVIEWER_ENGAGEMENT_CLOSEOUT.md` | `3b537c5d3a53e9b169de4bc98e7379a480bb960bbb88d9848d4ae190c7902fc5` |
| `models/production/manifest.json` (threshold `0.7847116291110687`) | `e6da1a433ec618576aa12208190d35df3b53e9349b1bb6e54c16465e324b2485` |
| `models/model_records/altman_native_v2_20260904_115703.json` | `45849237c96986b4e5aaffd94b3912c4358195ccea74b21b139fcda3b4d42091` |
| `backend/src/privacy_layer/native_features.py` (48-feature contract) | `5760a20376306598312cfaf32d8d12b5e5f3ff268fcc0b4de4baff64e90f57d1` |
| `backend/src/risk_engine/altman_native_ensemble.py` | `55f4ad5284ec6f45481660dfe2f9b256e90ad6ee2ff74abf68e876b8ebec2d24` |
| `backend/scripts/nr05_diagnostics.py` | `0e4b69d07ea2e97a0219d644d63635e9f32866323d4d5c4329c31d47722e12fc` |
| `misc/benchmarks/datasets.json` | `6d8e22d37f6eefb7f7e312d5494b8941021637735ec514d55e8a0ee86860620a` |
| `backend/scripts/review_resolution_check.py` | `13a04fb3151eec2a6e7f41e7ac5253f4bea6ae5f0e338c636f829942103f0880` |
| `backend/scripts/review_package_check.py` | `9b3db5be94a4d59944130de077738eec106f991c86b675e380085233aa0b6719` |
| `scripts/check_freeze.py` | `6a8b9719cc0fdfc419648103e13b0cc0c5969bdd524228ee788509d9055fd601` |

---

## 3. Objective findings resolved (R-1 … R-7 in the closure audit)

| # | Finding | Result |
|---|---|---|
| R-1 | **U-2** authoritative IBM v2 row count | Re-measured: `wc -l` = 24,386,901 → **24,386,900 data rows**. Distinctions preserved: authoritative observed count 24,386,900 · historical NR-05/memo/resolution constant 24,386,899 (0.00015% artefact) · scientific consequence none at a decision-relevant level · reviewer disposition required for the three cells |
| R-2 | **O-7** paysim column count | Both files measured: **8 columns**; coverage §3.2's “9” is superseded; scientific effect none (no native field; not in any eligibility row) |
| R-3 | §34 referenced-artifact paths (plan lines 1224–1226) | Located: preregistration (tracked), exposure ledger + eligibility manifest (untracked); fill content prepared for freeze-time plan edit; **plan not edited** |
| R-4 | **U-9** freeze-finding composition | **78 = 1 missing record + 65 plan markers (60 lines) + 12 prereg markers**; §35 = **36 unchecked**; reconciliation §13’s “56 + 11” text is superseded by measurement (and does not sum to its own stated 78); **record not edited** |
| R-5 | **O-1** superseded-cell presence | `cc_num` / `hour_diff` still present in resolution §7.1; `24,386,899` still present in memo §7; all still labelled reviewer-owned; **not corrected** |
| R-6 | Git/tracking facts (O-2/O-4/O-8) | HEAD `5a5ff55…`, 140 commits, 0 tags; plan + ledger + manifest untracked, preregistration tracked; tree dirty (135 entries) |
| R-7 | Reviewer-state counts (U-10) | `0/14 decisions resolved`; component states `NOT ASSIGNED=20`; both roles `NOT ASSIGNED` |

**None of R-1…R-7 changes any scientific conclusion**, any classification, any
eligibility, any licence state, or any review state.

---

## 4. Findings that remain reviewer-owned

- **O-1 disposition** — the superseded IBM cells (`cc_num`, `hour_diff`, `24,386,899`) stay in place until a reviewer rules (U-2 class).
- **U-2** — the three stale cells (NR-05, memo §7, resolution §7.1): statistical reviewer.
- **U-5** — `fraudTest` confirmatory-vs-exposed tension: statistical reviewer.
- **U-8** — in-window benchmark metrics: statistical reviewer; the held-out re-run is specified (reconciliation Appendix B) and **was not executed**.
- **The 14 decisions / D-1 (statistical) / D-2 (domain) / D-3 (auditor)** — all `NOT ASSIGNED`; plan §32 forbids self-approval.
- Review package §7 items A–O (15/15 `REQUIRES REVIEW`) and §8 items 1–11 — the reviewer work queue.
- **Deferred record corrections** (deliberate, not silent): O-3 duplicate `§34` numbering (freeze-time editorial fix); O-7 coverage §3.2 “9 → 8”; U-9 §13 count text — each belongs to the record owner / reviewer disposition, not to this phase.

---

## 5. Findings requiring external evidence or institutional access

| Item | Requirement |
|---|---|
| **U-1** licence/terms | Authoritative terms evidence for ULB, Kaggle ×2, IBM v2 (no terms archived; prior claims retracted C-5/C-7; all four outcomes remain `U`); requires a licence-qualified reviewer |
| **U-4** IBM publisher provenance | An archived publisher record; in-repo material holds only an unarchived origin claim (phase16 `label_source`; phase17 `known_origin`); manifest records `NOT ESTABLISHED` |
| **U-6** IEEE-CIS / BAF | Terms review **and** acquisition; acquisition not authorised (`BLOCKED — not acquired`) |
| **U-7** exposure audit | An independent auditor (external by definition; amending a record is not certifying it) |

---

## 6. Findings intentionally left unresolved

- **U-3** historical cause of the D1 mismatch — `HISTORICAL CAUSE NOT ESTABLISHED` preserved; the repository **is** git, but the affected records are untracked, so no history establishes their entry point; git does not prove who introduced the erroneous values. Reason restated (O-2); record not rewritten.
- **50M corpus** — `NOT AUTHORIZED / NOT YET SCIENTIFICALLY JUSTIFIED`; see §10.
- **Freeze / Track M / Track N** — gated on the above.
- The deferred record corrections in §4 (no silent rewrites; pinned artifacts).

---

## 7. Current freeze-check result

`scripts/check_freeze.py` → **rc=1, 78 problem(s)** — expected pre-freeze:
**1** missing `docs/FREEZE_RECORD.json` + **65** plan placeholders on 60 lines +
**12** preregistration markers. Plan §35 has **36 unchecked** items. The checker was
not changed, weakened, or bypassed; it must keep failing until freeze is legitimate.

---

## 8. Current reviewer-assignment state

| Role | State |
|---|---|
| Statistical reviewer (D-1) | `NOT ASSIGNED` — 0/8 statistical-only decisions + 0/6 statistical halves |
| Domain reviewer (D-2) | `NOT ASSIGNED` — 0/6 domain components |
| Independent dataset/provenance auditor (D-3) | not assigned — U-7 open |
| Licence/terms reviewer | not completed — all four datasets outcome `U` |
| Decisions | **0/14** resolved; readiness `0/14 decisions resolved` (component states `NOT ASSIGNED=20`) |

No identity, affiliation, qualification, date, conflict declaration or approval was
invented, prefilled, or implied anywhere in this phase.

---

## 9. Current Track M state

**`BLOCKED`.** Zero confirmatory-eligible datasets: ULB, Kaggle ×2 and IBM v2 remain
`EXPLORATORY`; IEEE-CIS / BAF `BLOCKED — not acquired`; PS-14 synthetic is
simulation-only. Plan §23 applies unchanged; no confirmatory experiment was executed.

---

## 10. Current 50M state

**`NOT AUTHORIZED / NOT YET SCIENTIFICALLY JUSTIFIED`.** No 50M dataset exists or was
generated. Largest acquired real corpus: IBM v2, **24,386,900** rows — the only
dataset covering the native 48 contract in full (48/48); total acquired real rows
across all four acquired datasets = **26,524,101** (≈53% of 50M), duplicates excluded
by rule.

> **No 50M native corpus should be created unless its provenance, feature semantics,
> independence, and scientific purpose are established first.**

---

## 11. Tests — all re-run this phase

| Check | Result |
|---|---|
| `python scripts/check_freeze.py` | **rc=1 — 78 problem(s)** (expected pre-freeze; unchanged) |
| `python backend/scripts/review_resolution_check.py` | **rc=0** — PASS; `--readiness` = `0/14`, `NOT ASSIGNED=20` |
| `python backend/scripts/review_package_check.py` | **rc=0** — PASS (13/13 sections, A1–A9 assignment rules, A10 brief rules; no licence cleared, no reviewer assigned/approved, no 50M claim, 15/15 audit items at `REQUIRES REVIEW`) |
| `python backend/scripts/review_package_check.py --self-test` | **rc=0** — 25/25 tamper and clean-fixture checks passed |
| `python backend/scripts/claim_evidence_check.py` | **rc=0** — PASS (22 claims verified against evidence) |
| `python backend/scripts/eval_record_test.py` | **rc=0** — 22/22 passed |
| `python backend/scripts/check_freeze_test.py` | **rc=0 — 22/22 passed** (with `PYTHONIOENCODING=utf-8`, which the battery exports; a direct run without it hits a pre-existing cp1252 console-decode quirk — environmental, not a regression) |
| Full battery (`.freebuff/p114_battery.sh`, 84 suites) | **84 PASS / 0 FAIL**, rc=0 — START 2026-10-05T12:02:19 → END 2026-10-05T12:15:42, six services healthy (6/6 `/health`=200) before the run |

Evidence: `.freebuff/oec_freezechk_base.out`, `.freebuff/oec_freezechk_final.out`,
`.freebuff/oec_battery_driver.out`, `.freebuff/p114rq_battery_results.tsv` (84 rows).
No test was skipped, weakened, or suppressed.

**Expected battery side effects (unchanged in kind):** `data/` suite outputs rewritten
(O-5 — not evidence records); `models/` mtime touches with hash-identical content
(O-6 — verification basis is hash identity); the append-only eval ledger grew
**142 → 146**, all four new records byte-identical deterministic `calibration_test.py`
re-runs (brier `0.01073368217015549`, ece `0.014450218995775772`, n 1470) — O-5 class,
not evidence change.

---

## 12. Preserved negative findings

All NR-05 / resolution negatives are intact under their pinned hashes (resolution
`17a95e2a…`, NR-05 diagnostics `0e4b69d0…`): **N1** ensemble loses to XGB; **N2** IBM
Platt inversion; **N3** IBM test AUC 0.627 → 0.373 with good Brier/ECE; **N4** ULB
recall 0.813 → 0.040; **N5** ULB PSI 1.46 vs 0.25; **N6** Kaggle PSI max 0.083;
**N7** Class-B benefit NOT ESTABLISHED; **N8** Kaggle entity-disjoint 0.9697;
**N9** IBM temporal 0.83–1.00 → 0.58–0.64; **N10** gating INCONCLUSIVE; **N11** ULB
native coverage 0; **N12** harness defects fixed pre-report; **N13** ULB q4=11 /
q2=9 withheld. Also preserved: the `24,386,899` historical constant (U-2), the
superseded reviewer-owned cells (O-1), and the U-8 in-window-metric limitation (not
erased by any re-run).

---

## 13. Invariant verification

| Invariant | Pinned | Re-measured | Match |
|---|---|---|---|
| Research Plan | `eab5a061…` | `eab5a061…` | ✅ |
| Preregistration | `fdde71cb…` | `fdde71cb…` | ✅ |
| Metric definitions | `8665549b…` | `8665549b…` | ✅ |
| Native 48-feature contract | `5760a203…` | `5760a203…` | ✅ |
| Production threshold + manifest | `e6da1a43…` (`0.7847116291110687`) | `e6da1a43…` | ✅ |
| Model record | `45849237…` | `45849237…` | ✅ |
| Altman pins (cb/lgb/scaler/xgb/feature_list) | `22b8377b…`/`d59aebcb…`/`b98fadf3…`/`a1cdebdf…`/`657459ac…` | identical | ✅ |
| NR-05 outputs | `0e4b69d0…` | `0e4b69d0…` | ✅ |
| Dataset bytes (ULB / Kaggle train / Kaggle test / IBM v2 / User0) | 150,828,752 / 351,238,196 / 150,354,339 / 2,350,744,057 / 1,899,258 | identical | ✅ |
| Dataset row counts | 284,807 / 1,296,675 / 555,719 / 24,386,900 / 19,963 | identical | ✅ |
| Exposure classifications | ledger `31af86fd…` | `31af86fd…` | ✅ (no status upgraded) |
| Reviewer decisions / identities / approvals | 0/14, none | 0/14, none | ✅ |
| Freeze state | absent | absent | ✅ |
| Track M status | `BLOCKED` | `BLOCKED` | ✅ |

---

## 14. Intentionally unresolved blockers and exact next action

**Blockers (all human/external):** reviewer assignment (D-1/D-2/D-3); licence
determination (U-1); IBM publisher provenance (U-4); IEEE-CIS/BAF terms + acquisition
(U-6); independent audit (U-7); reviewer dispositions (O-1, U-2, U-5, U-8); the
deferred record corrections (§4).

**Exact next action:** hand `docs/evaluation/REVIEWER_ENGAGEMENT_BRIEF.md` to
prospective reviewers/auditors and record **at least one** of — (a) an assigned
statistical reviewer (D-1), (b) an assigned domain reviewer (D-2), (c) an assigned
independent dataset/provenance auditor (D-3), (d) authoritative licence evidence
(U-1) — in the existing machinery (assignment record §1/§3; resolution §4; licence
checklist §3; review package §7/§8).

**Stop condition honoured:** no further internal preparation phase is warranted. The
intended sequence is objective evidence closure → qualified human review → Research
Plan freeze → confirmatory Track M → scientific decision → 50M-scale evaluation only
if justified. The human-review gate must not be skipped merely because the repository
is technically capable of running the experiments.

> **The repository's objective evidence is closed, and review has not commenced because
> qualified human reviewers have not yet been assigned. The project now waits at
> `OBJECTIVE EVIDENCE CLOSED — AWAITING HUMAN REVIEW`.**

---

## What this phase did not do

It did not assign or invent a reviewer (fictional, AI, or author); did not approve,
reject, or revise any decision; did not freeze the plan or create
`FREEZE_RECORD.json`; did not run Track M or any confirmatory experiment; did not
change the preregistered methodology, thresholds, model weights or the native
48-feature contract; did not retrain the production model; did not acquire, generate,
or pad any dataset (including the 50M); did not change dataset eligibility or exposure
classifications; did not silently rewrite any reviewer-owned finding or dated record;
did not convert exploratory evidence into confirmatory evidence; and did not create
another governance, monitoring, security, or registry subsystem.

| Version | Change | Type |
|---|---|---|
| 1.0-draft | Initial objective evidence closure: closure audit created (U-1…U-10, O-1…O-8, freeze inventory, R-1…R-7, 50M determination, 65 + 12 marker classifications, handoff matrix); all checks re-run (battery 84/84; freeze rc=1 / 78 expected); all invariants re-verified unchanged; no review, licence, eligibility, exposure, freeze, or Track M state changed. | OBJECTIVE EVIDENCE CLOSURE |
