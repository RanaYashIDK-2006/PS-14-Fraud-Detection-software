# Phase RP-01 — Research Plan Adoption & Freeze Mechanism

**Status:** PASS WITH LIMITATIONS
**Date:** 2026-10-04
**Input:** `RESEARCH_PLAN (2).md` (DRAFT — not frozen), supplied by the owner with the instruction "work according to this".
**Prerequisite:** NR-04 (preregistered harness) completed and verified — see `docs/PHASE_NR04_PREREGISTERED_HARNESS.md`.

---

## A. Scope of this phase

Interpretation of "work according to this": adopt the DRAFT research plan as the
governing artifact for subsequent work, implement the one mechanism the plan
mandates that does not require reviewer decisions (§30 automated freeze check),
and reconcile the plan's requirements against what the repository actually
contains — **without** resolving any `[TO BE FROZEN]` value, performing any
freeze, running any confirmatory experiment, or starting NR-05.

Explicitly out of scope (per plan §36 and NR-04 discipline):

- resolving any of the 65 plan placeholders or 12 preregistration markers
  (reviewer/owner-owned);
- creating `docs/FREEZE_RECORD.json` (plan §29: "To be created at freeze");
- creating a freeze tag, signing off statistical sections, or claiming any
  track outcome (`GO`/`NO-GO`/etc.);
- downloading IEEE-CIS/BAF or substituting blocked datasets.

---

## B. What was adopted

| Item | Value |
|---|---|
| Plan installed at | `docs/RESEARCH_PLAN.md` (plan's own declared path) |
| Lines | 1230 |
| SHA-256 | `eab5a0615801db7e50ab4d1c0ec73f93905e461b5bd8eacc617e0418d656b9ce` |
| Integrity | byte-identical copy of the supplied input |
| Status line in document | `DRAFT — not frozen` (unchanged) |

From this phase forward the plan governs: four tracks (N/M/P/I), five decision
rules, seven decision outcomes (never collapsed), and the freeze requirement
(§29/§30) apply to all confirmatory work.

---

## C. Freeze mechanism — `scripts/check_freeze.py` (plan §30)

The plan's artifact table lists the freeze-check script as "Draft provided",
but no draft exists anywhere in the supplied file or the repository. The
script was therefore implemented directly from the §30 specification, at the
exact path the plan requires (`scripts/check_freeze.py`).

The five §30 duties, and how each is implemented:

| §30 duty | Implementation |
|---|---|
| 1. Scan plan **and** preregistration for unresolved placeholders — marker words `TO BE FROZEN, OWNER, DATE, HASH, SHA, VERSION, TAG, NAME, TIMESTAMP, STATISTICAL, DOMAIN, N` **or any bracketed token written entirely in capitals** | `placeholder_tokens()`: bracket scan with whitespace-normalised tokens (handles the marker that wraps across lines at protocol L211), marker-prefix match **case-insensitive** (a lowercase spelling cannot slip through), plus all-caps token rule. Markdown checkboxes `[ ]` are not tokens, so the plan's own §35 checklist cannot self-trigger. |
| 2. Every field in `docs/FREEZE_RECORD.json` present and non-placeholder | `REQUIRED_FIELDS` (git tag, git SHA, freeze timestamp, owner, statistical reviewer, approval timestamp, artifacts); empty/None/marker-valued fields fail as `record field empty/placeholder` |
| 3. Recompute SHA-256 for **every** listed artifact, **including the script itself** | every `artifacts[]` entry re-hashed against `root/path`; `freeze_check_script` covers self-hash — editing the check after freeze breaks the freeze, as §29 requires |
| 4. Statistical reviewer approval fields present | `statistical_reviewer` + `statistical_reviewer_approval_timestamp` in `REQUIRED_FIELDS`; both presence and non-placeholder value checked |
| 5. Exit non-zero on any failure | `rc=1` on any failure, `rc=0` only when the full freeze state is valid |

**Record schema** (documented in the script header; created at freeze, not
before): git tag/SHA, freeze timestamp, owner, statistical reviewer +
approval timestamp, and an `artifacts[]` list of `{name, path, sha256}` for
`research_plan`, `preregistration`, `metric_definitions`,
`dataset_eligibility_manifest`, `exposure_ledger`, `freeze_check_script`.

**Observed behavior on the real repository (pre-freeze, by design):**

```
FREEZE CHECK FAILED — 78 problem(s):
  FAIL missing docs/FREEZE_RECORD.json — freeze not recorded (expected pre-freeze)
  FAIL placeholder docs/RESEARCH_PLAN.md:127: [TO BE FROZEN]
  … 65 plan placeholders …
  FAIL placeholder docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md:211: […]
  … 12 preregistration markers …
rc=1
```

Breakdown of the 78: 1 missing record + 65 plan placeholders (54 `TO BE
FROZEN`, 5 `DATE`, 3 `OWNER`, 1 `N`, 1 `STATISTICAL`, 1 `DOMAIN`) + 12
preregistration all-caps tokens (11 `[REQUIRES DECISION/APPROVAL]` markers +
1 wrap-split duplicate at line 211). **The freeze check failing is the gate
working**, not a defect.

---

## D. Test suite — `backend/scripts/check_freeze_test.py`

Plain assert script (house style, no pytest). **22/22 PASS.**

- real-repo verdict is **freeze-state-aware**: asserts `rc!=0` + missing
  record + placeholder counts while pre-freeze, and flips to asserting
  `rc==0` once `docs/FREEZE_RECORD.json` exists — the suite stays valid
  across the freeze transition instead of breaking at the milestone it
  guards;
- a fully-frozen miniature repo (temp fixture: clean plan, clean
  preregistration, artifacts, complete record with correct hashes) passes
  `rc=0`;
- each §30 duty is proven to **independently** force failure: tampered
  artifact hash; tampered `check_freeze.py` (self-hash); missing reviewer
  approval field; placeholder *value* in a reviewer field; missing required
  field; missing artifact entry; reintroduced plan placeholder; all-caps
  preregistration token; absent record; malformed record JSON;
- marker semantics: 6/6 marker+all-caps forms detected, checkboxes and
  ordinary bracketed prose not flagged, wrapped markers normalised,
  lowercase marker spelling detected, field placeholder logic correct.

---

## E. CI wiring (§30: "runs in CI on every commit that touches the plan, the
preregistration or the record")

Added `.github/workflows/freeze-check.yml`: path-filtered to
`docs/RESEARCH_PLAN.md`, the preregistration, `docs/FREEZE_RECORD.json`,
`scripts/check_freeze.py`, and itself; plus `freeze-*` tags (the "again
before the freeze tag is created" clause) and `workflow_dispatch`.

Deliberate design choice, flagged for the owner: the workflow is a
**separate** file, so its pre-freeze red X (documented in the header comment)
signals "freeze not valid" without blocking the main test/security pipeline
on unrelated commits. GitHub-runner execution remains a **carried blocker**
(no `gh` CLI, nothing pushed — same blocker as NR-01/NR-04).

---

## F. Battery registration

`.freebuff/p114_battery.sh` line 88: `run check_freeze_test.py`, inserted
after the NR-04 suite (edited via terminal — `.freebuff/` is blocked to file
tools; `bash -n` syntax check passes). Battery row count: 82 → 83.

---

## G. Reconciliation — plan requirements vs. repository reality

| Plan requirement | Current repository state | Verdict |
|---|---|---|
| §4A.4 metric definitions must cover every 4A measure, versioned | `docs/metric_definitions.md` + `METRIC_DEFINITIONS_VERSION = "1.0"` (`backend/scripts/metric_definitions.py:50`). Covers ROC/PR-AUC, recall@FPR, imbalance baselines. **Missing**: precision/recall at fixed alert volume, F-beta≠1, Brier/log loss/ECE/reliability/calibration slope+intercept, expected cost, gating measures (coverage/abstention/wrong-confident/risk-coverage/fallback), seed/window stability. Plan also demands the v1.0 CI method be reconciled with §19 → **version bump required before freeze** | GAP — pre-freeze work, blocked on §18/§19 decisions for the CI part |
| §18 power rule (min fraud / min total / min seeds) is separate from and stricter than v1.0 `min_cell=30` | v1.0 withholds an interval when a cell has <30 observations — an *interval-reporting* floor, not a power rule; the §18 numbers are `[TO BE FROZEN]` | PENDING (reviewer-owned); no code change possible until frozen |
| §19 dependence-aware bootstrap; **IID is not the default** | Three IID percentile implementations exist (1000 reps seed-42 in metric_definitions, 200 in `eval_ulb`, 2000 proposed in protocol); entity-clustered and temporal-block bootstrap **not implemented**; applicability rule `[TO BE FROZEN]` | GAP — implementation deferred until the method is frozen (implementing a speculative clustered bootstrap would pre-empt §32 sign-off) |
| paired-difference CI (§4A.3, §20) | NR-04 `paired_difference_ci` exists **boundary-only** and raises `PendingReview` — refuses rather than invents a method | CONSISTENT with plan (refusal is correct until §19 method frozen) |
| §34 artifact: `docs/metric_definitions.md` | EXISTS | OK |
| §34 artifact: preregistration path `[TO BE FROZEN — path]` | Candidate in-repo: `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` (DRAFT, 12 markers). Path decision left to owner/checkpoint | PROPOSED, not frozen |
| §34 artifact: dataset exposure ledger `[TO BE FROZEN — path]` | **PRODUCED (draft `exposure-ledger 1.0-draft`)** at `docs/evaluation/DATASET_EXPOSURE_LEDGER.md`: §14 table preserved, dataset hashes recomputed and verified against `prereg_harness.py::DATASETS`, §15/claims-registry classifications cross-referenced, open items listed (independent audit, licence conflicts, fraudTest confirmatory ambiguity) | ARTIFACT EXISTS (draft); independent §14 audit + path decision still NOT ESTABLISHED |
| §34 artifact: dataset eligibility manifest `[TO BE FROZEN — path]` | **PRODUCED (draft `eligibility-manifest 1.0-draft`)** at `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md`: all §13 fields per candidate (ULB/Kaggle×2/IBM v2/IEEE-CIS/BAF/synthetic) with explicit ESTABLISHED / NOT ESTABLISHED / BLOCKED / PENDING REVIEW states; no dataset declared eligible; deterministic classifier remains `backend/scripts/external_validation/eligibility_gates.py` (referenced, not duplicated) | ARTIFACT EXISTS (draft); no eligibility granted; IEEE-CIS/BAF remain BLOCKED; path decision NOT ESTABLISHED |
| §34 artifact: `docs/FREEZE_RECORD.json` | Absent (by design — created at freeze) | OK |
| §34 artifact: `scripts/check_freeze.py` | IMPLEMENTED this phase | OK |
| §14 exposure history: ULB/Kaggle/IBM = EXPLORATORY; IEEE-CIS/BAF = CANDIDATE CONFIRMATORY | Matches repo evidence (NR-03 dataset registry; BAF absent → BLOCKED in harness `DATASETS`) | CONSISTENT |
| §15 external-transfer evidence labels | Match the claims registry (IBM cross ~0.873 EXPLORATORY/NOT INDEPENDENT; Kaggle ~0.435–0.595 FAILED/NON-CONFORMING) | CONSISTENT |
| §27 engineering scope freeze on confirmatory conditions | No engineering changes made this phase beyond the freeze mechanism itself (which §30 mandates) | OK |
| §30 CI run | Workflow added (§E above); execution pending push | PARTIAL — carried blocker |
| §32 statistical reviewer sign-off | Not obtained (single-developer sign-off explicitly insufficient) | BLOCKED |
| **Freeze ordering vs roadmap** | Roadmap places final freeze at **NR-40 (Gate 6, end)**; plan §36 requires **freeze → experiment** for confirmatory work | **RESOLVED** (see §H) |

---

## H. Freeze ordering — RESOLVED

**Status: RESOLVED**

> NR-05 may proceed before the Research Plan freeze only as exploratory work
> on previously exposed datasets. It cannot generate confirmatory Track M
> evidence or alter the frozen decision framework.
>
> All confirmatory Track M execution on untouched eligible datasets begins
> only after statistical review, resolution of all freeze placeholders,
> creation of `docs/FREEZE_RECORD.json`, and a successful freeze check on
> the tagged commit.
>
> The Research Plan's freeze-before-confirmatory-experiment rule therefore
> remains authoritative.
>
> This resolves the apparent conflict between the roadmap ordering and the
> Research Plan. NR-05 remains the next roadmap requirement, but its
> pre-freeze work is explicitly exploratory. The freeze remains the
> mandatory gate before confirmatory Track M execution.

Supporting context (unchanged from the original flag): the plan's decision
principle is `… preregistration → review → freeze → experiment`, while the
roadmap runs `NR-05 … NR-39 (experiments) → NR-40 (final freeze)`. Under the
resolution above these no longer conflict:

- per plan §14, ULB/Kaggle/IBM are **EXPLORATORY** — NR-05's temporal-split
  work on them is pre-freeze **exploratory** work and produces no
  confirmatory Track M evidence; no exploratory result may make an exposed
  dataset "untouched", be presented as independent replication, or be used
  to retrospectively select or alter the confirmatory decision rules;
- the confirmatory route (Track M on untouched eligible datasets, e.g.
  IEEE-CIS/BAF once acquired) stays **BLOCKED** until all five freeze
  conditions hold: (1) statistical review complete, (2) all freeze
  placeholders resolved, (3) `docs/FREEZE_RECORD.json` exists and is
  complete, (4) the freeze checker passes, (5) frozen artifacts committed
  at the tagged Git state;
- plan §35 prerequisites (exposure history audit, eligibility manifest)
  remain pre-freeze requirements — they map to the §G gaps and to existing
  phases, not to new work created here.

The resolution is also recorded in the authoritative roadmap:
`docs/PHASE_106_ROADMAP_RECONCILIATION.md` §J (freeze-ordering note above
the gate tables).

---

## I. What was NOT done (deliberately)

- no `[TO BE FROZEN]` value resolved (65 plan + 12 protocol markers remain);
- no `docs/FREEZE_RECORD.json`, no freeze tag, no ledger freeze entry;
- no statistical/domain sign-off claimed;
- no confirmatory experiment, no threshold/seed/bootstrap choice;
- no IEEE-CIS/BAF acquisition;
- NR-04 harness untouched (its 22-entry `PENDING_DECISIONS` registry stays
  authoritative for execution guards; the plan's freeze gate is complementary
  — document-level vs experiment-level);
- NR-05 **not started**.

---

## J. Verification

| Check | Result |
|---|---|
| `scripts/check_freeze.py` on real repo | rc=1, 78 problems — **expected pre-freeze** |
| `backend/scripts/check_freeze_test.py` | **22/22 PASS**, rc=0 |
| `bash -n .freebuff/p114_battery.sh` | syntax OK; `check_freeze_test.py` at line 88 |
| Battery after registration | see §K |
| Plan copy hash | `eab5a061…656b9ce`, 1230 lines, byte-identical |
| Preregistration protocol | untouched this phase; 11 markers (`grep -o` literal) |
| NR-04 suite re-run after all edits | 31/31 PASS (carried from NR-04 closeout verification) |
| Freeze-ordering resolution (§H) — plan untouched | `docs/RESEARCH_PLAN.md` hash unchanged (`eab5a061…656b9ce`); placeholder counts unchanged (65 plan / 11 protocol markers); `scripts/check_freeze.py` still rc=1 with the same 78 problems |
| Freeze-ordering resolution — no evidence changed | `git status` shows only documentation files touched; ledger record count unchanged (0 harness records); `claim_evidence_check` + `eval_record_test` re-run after the edits |

---

## K. Battery result

Full battery `bash .freebuff/p114_battery.sh` after registration:

```
===== BATTERY END 2026-10-04T14:02:21+05:30 =====
PASS 83
FAIL 0
```

83 rows (82 previous + `check_freeze_test`), all PASS. Key rows:

| Suite | Result | Time |
|---|---|---|
| `check_freeze_test` | PASS | 1s |
| `prereg_harness_test` | PASS | 5s |
| `eval_record_test` | PASS | 6s |
| `claim_evidence_check` | PASS | 1s |

### Re-run after the freeze-ordering resolution (§H update)

Full battery re-run: **82 PASS / 1 FAIL** — evidence rows all PASS
(`claim_evidence_check`, `eval_record_test`, `prereg_harness_test`,
`check_freeze_test`). The single failure,
`phase109_audit_fork_repair_test`, is an **environmental flake, not a
regression from this documentation-only change**: its
`models/artifacts` tree-hash check observed a transient entry inside the
suite's 23-second window (14:18:01–14:18:24) that no longer exists on
disk (all 25 files under `models/artifacts` still carry Oct-3 mtimes; the
same suite hashed the identical current state `9b103daea2e2` before and
after in the previous battery). The window coincides with the client-side
hook's calibration run completing at 14:18:06 (documented environmental
actor); source inspection shows no battery suite, subprocess, or the hook
script itself writes or deletes under `models/artifacts`. Standalone
re-run: **96/96 PASS, rc=0** (not reproducible).

### Re-run after producing the two §34 artifacts (exposure ledger + eligibility manifest)

Full battery re-run: **83 PASS / 0 FAIL** (BATTERY END
2026-10-04T15:35:18+05:30) — `phase109_audit_fork_repair_test` PASSED
this time (the §K flake did not recur), and `claim_evidence_check`,
`eval_record_test`, `prereg_harness_test`, `check_freeze_test` all PASS.
Verified in the same window: plan SHA unchanged (`eab5a061…`), plan +
protocol diff EMPTY, 11 markers, `FREEZE_RECORD.json` absent, 0 tags,
evidence ledger 62 records with 0 harness/prereg entries (no experiment
ran), no file under `models/` written after battery start, and all five
dataset SHA-256 values unchanged post-battery.

---

## L. Limitations

1. **Freeze check cannot pass pre-freeze** — by design; the 78 failures are
   the gate, and will only clear when reviewers resolve the placeholders and
   the record is created at freeze.
2. **Two §34 artifacts now exist as drafts** (dataset exposure ledger,
   dataset eligibility manifest — both `1.0-draft`, this document's §G),
   but their §34 path cells remain `[TO BE FROZEN]`, the §14 independent
   exposure audit is not performed, and no dataset has been declared
   eligible — so claims depending on their *approval* remain
   `NOT ESTABLISHED` until the freeze procedure adopts them.
3. **CI execution unverified** — workflow written but GitHub runner remains
   the carried NR-01 blocker (no push performed).
4. **§4A metric coverage and §19 dependence-aware CI are gaps** requiring a
   metric-definitions version bump and reviewer-frozen methods first.

---

## M. Remaining blockers / next requirements

- Statistical reviewer sign-off (plan §32 scope) — blocks freeze and hence
  any confirmatory execution.
- Owner decisions: preregistration path, exposure-ledger path, eligibility
  manifest path. (Freeze ordering: **RESOLVED** — §H.)
- Dataset acquisition (IEEE-CIS terms, BAF licence/provenance) — blocks
  Track M confirmatory route.
- **NR-05 (temporal-split evaluation)** remains the next roadmap requirement.
  Its pre-freeze work is **explicitly exploratory** (previously exposed
  datasets only); confirmatory Track M execution cannot begin before the
  freeze (§H).

**Final Status: PASS WITH LIMITATIONS.**
