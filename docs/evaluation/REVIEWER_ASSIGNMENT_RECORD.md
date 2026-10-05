# Reviewer Assignment Record

> This document records reviewer responsibility. It does not manufacture reviewer identity, statistical approval, domain approval, or Research Plan approval.

**Status:** ADMINISTRATIVE READINESS ARTIFACT — DRAFT.

| | |
|---|---|
| Research Plan | **DRAFT / NOT FROZEN / NOT APPROVED** |
| Preregistration | **DRAFT / NOT APPROVED** (`0.1.2-draft`) |
| Statistical reviewer | **`NOT ASSIGNED`** |
| Domain reviewer | **`NOT ASSIGNED`** |
| Decisions resolved | **0 of 14** |
| Confirmatory Track M | **BLOCKED** |
| Freeze findings | **78 — unchanged by this document** |

**Current blocker:** no actual statistical reviewer and no actual domain reviewer
has been assigned. This is a human, external dependency. It cannot be resolved from
inside the repository: no fictional person, generic organization, AI reviewer or
project author may be substituted for a qualified reviewer unless the repository
already explicitly establishes that person as the qualified reviewer.

**Roles recorded here are pre-existing repository roles**, taken from
`docs/RESEARCH_PLAN.md` §28 and §32 and from the preregistration §5 sign-off block.
**No new reviewer category is introduced.**

**Authored at git state:** `5a5ff55cc03df73318fdf31a16e3665f159a4720` (2026-10-04).
No commit is implied.

---

## 1. Reviewer roles

### 1.1 Statistical Reviewer

Required by plan §28 (`[STATISTICAL REVIEWER]`, action "Sign off every section
listed in Section 32 before freeze", status `PENDING`) and by the preregistration §5
sign-off row "Statistical reviewer (external, R-42)".

```text
Role:
STATISTICAL REVIEWER

Assigned individual:
NOT ASSIGNED

Affiliation:
NOT ESTABLISHED

Assignment date:
NOT ESTABLISHED

Conflict-of-interest declaration:
NOT ESTABLISHED

Scope:
Plan §32 — "Sign off scope (identical to the Section 28 reviewer task): Section 3
(primary metric, aggregation, multiplicity), Section 4A (performance measures and
tiers), Section 5.0 and 5.5 (gate definitions and thresholds), Section 6
(calibration), Sections 18 to 24 (power, dependence-aware intervals, above-chance
rule, success criteria, decision outcomes, go/no-go and roll-up, REVISE and
INCONCLUSIVE rules), and Section 32 itself." Within the 14 open decisions this
covers baseline identity (/01), ensemble identity (/02), confident-decision
definition (/03), feature-contract methodology (/05), calibration (/07), power
(/08), dependence-aware uncertainty (/09), seed/run rule (/10), member operating
points (/11), sampling (/13), segment/small-n withholding (/14), and the other
statistical decisions identified by the resolution matrix. Preregistration §5
restates the scope as "CI/procedure/multiplicity".

Assignment status:
NOT ASSIGNED

Approval authority:
NOT ESTABLISHED
```

**Constraint already recorded in plan §32:** *"A single developer's self-approval is
not sufficient for the statistical sections."* This record does not assert that any
particular person satisfies that constraint; it is recorded as `NOT ESTABLISHED`
pending an actual, explicit assignment.

### 1.2 Domain Reviewer

Required by plan §28 (`[DOMAIN REVIEWER]`, action "Review outcome/metric semantics",
status `PENDING`) and by the preregistration §5 sign-off row "Domain reviewer (bank
SME or equivalent)".

```text
Role:
DOMAIN REVIEWER

Assigned individual:
NOT ASSIGNED

Affiliation:
NOT ESTABLISHED

Assignment date:
NOT ESTABLISHED

Conflict-of-interest declaration:
NOT ESTABLISHED

Scope:
Plan §32 — "Domain review must separately verify the semantic appropriateness of
the primary outcomes where required." Plan §28 — "Review outcome/metric semantics."
Preregistration §5 states the scope as "business metrics, assumptions". Within the
14 open decisions this covers cost/abstention interpretation (/04), alert-volume
semantics (/12), operational meaning, segment interpretation (/14 where semantics
apply), domain assumptions affecting evaluation criteria, and the domain half of
every decision classified BOTH (/03, /04, /05, /06, /07, /12). Preregistration §7
forbids describing an assumed analyst capacity as measured analyst burden.

Assignment status:
NOT ASSIGNED

Approval authority:
NOT ESTABLISHED
```

### 1.3 Other existing unassigned dependencies — recorded, not filled

These are pre-existing plan §28 rows and preregistration §5 roles. They are **not**
reviewer categories for `/01`–`/14` and are listed only so this record does not read
as a complete assignment register.

| Dependency | Plan §28 owner cell | Required action | Status |
|---|---|---|---|
| IEEE-CIS terms/use restrictions | `[OWNER]` | read terms before download; verify labelled-data availability | **PENDING** |
| BAF licence/terms/provenance | `[OWNER]` | read licence; verify provenance and generation process | **PENDING** |
| Exposure audit | `[OWNER]` | audit repository evidence for prior dataset exposure | **PENDING** |
| Lead researcher | preregistration §5 | criteria 1–7, MMDs | `_pending_` |
| Data-governance/privacy (if bank data) | preregistration §5 | dataset admission | `_pending_` |

**None of these is assigned here.**

---

## 2. Decision-to-reviewer matrix

Roles are carried over **unchanged** from
`docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` §3. Where that artifact
requires both roles, this matrix says `BOTH` — no decision is silently collapsed to
one role.

| ID | Decision | Required Reviewer Role | Reviewer Identity | Assignment Status | Approval Status |
|---|---|---|---|---|---|
| `/01` | Baseline identity | STATISTICAL REVIEW | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/02` | Ensemble identity | STATISTICAL REVIEW | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/03` | Confident-decision definition | **BOTH** | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/04` | Gating bounds | **BOTH** | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/05` | Feature-contract construction | **BOTH** | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/06` | Window-gate definition | **BOTH** | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/07` | Calibration | **BOTH** | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/08` | Power floors | STATISTICAL REVIEW | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/09` | Dependence-aware method | STATISTICAL REVIEW | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/10` | Minimum seed / run rule | STATISTICAL REVIEW | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/11` | Member operating points | STATISTICAL REVIEW | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/12` | Alert-volume semantics | **BOTH** | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/13` | Large-dataset sampling | STATISTICAL REVIEW | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |
| `/14` | Segment / small-n withholding | STATISTICAL REVIEW | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT APPROVED` |

**Totals:** 14/14 covered · 8 STATISTICAL REVIEW · 6 BOTH · 0 DOMAIN-only ·
0 assigned · 0 approved. For the 6 `BOTH` rows, a statistical ruling alone is
**not** sufficient; the domain half remains open until a domain reviewer rules.

**Two independent components, mechanically enforced.** A `BOTH` row is decided by
two separate reviewer inputs, recorded in
`docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` §4 as a statistical
component (the sheet's own reviewer block) and a domain component (the
`Domain Reviewer …` block). `backend/scripts/review_resolution_check.py` rule R12
rejects any `BOTH` decision that is `APPROVED` unless both components are complete,
and rejects a sheet whose `Required reviewer components:` line disagrees with the
resolution artifact's §3 ownership table. **One reviewer cannot satisfy both roles,
and assignment alone moves nothing.**

---

## 3. Reviewer qualification record

All fields below remain `NOT ESTABLISHED` until actually supplied by a real person.
No qualification, credential, affiliation, date or conflict declaration is invented
here.

### 3.1 Statistical Reviewer — qualification

| Field | Value |
|---|---|
| Identity | `NOT ESTABLISHED` |
| Relevant expertise (statistical method, dependence-aware inference, calibration, sample-size/power) | `NOT ESTABLISHED` |
| Independence / conflict-of-interest declaration | `NOT ESTABLISHED` |
| Date assigned | `NOT ESTABLISHED` |
| Scope accepted | `NOT ESTABLISHED` |
| Evidence reviewed (plan §32 sections, preregistration, NR-05, exposure/eligibility artifacts) | `NOT ESTABLISHED` |
| Signature / approval reference | `NOT ESTABLISHED` |

### 3.2 Domain Reviewer — qualification

| Field | Value |
|---|---|
| Identity | `NOT ESTABLISHED` |
| Relevant expertise (fraud-operations semantics, cost/abstention posture, alert-volume interpretation) | `NOT ESTABLISHED` |
| Independence / conflict-of-interest declaration | `NOT ESTABLISHED` |
| Date assigned | `NOT ESTABLISHED` |
| Scope accepted | `NOT ESTABLISHED` |
| Evidence reviewed | `NOT ESTABLISHED` |
| Signature / approval reference | `NOT ESTABLISHED` |

---

## 4. Review input contract

What a real reviewer must supply. **This reuses the existing resolution
representation** — `docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` §4
already contains the fields; nothing new is defined here, and no schema is added.

### 4.1 Statistical reviewer — supplies once per role

| Required input | Where it is recorded | Current value |
|---|---|---|
| Identity | §1.1 `Assigned individual:` and §3.1 `Identity` | `NOT ASSIGNED` / `NOT ESTABLISHED` |
| Qualification / affiliation | §1.1 `Affiliation:`, §3.1 `Relevant expertise` | `NOT ESTABLISHED` |
| Conflict-of-interest declaration | §1.1 `Conflict-of-interest declaration:`, §3.1 | `NOT ESTABLISHED` |
| Assignment date | §1.1 `Assignment date:`, §3.1 `Date assigned` | `NOT ESTABLISHED` |
| Approval authority | §1.1 `Approval authority:` | `NOT ESTABLISHED` |
| Scope accepted | §3.1 `Scope accepted` | `NOT ESTABLISHED` |
| Signature / approval reference | §3.1 | `NOT ESTABLISHED` |

Plan §32 states: *"A single developer's self-approval is not sufficient for the
statistical sections."* That constraint is recorded, not enforced by this document.

### 4.2 Statistical reviewer — supplies per assigned decision

For each of `/01 /02 /03 /04 /05 /06 /07 /08 /09 /10 /11 /12 /13 /14`, in that
sheet's statistical reviewer block:

| Field | Meaning |
|---|---|
| `Reviewer Decision` | `APPROVED` / `REJECTED` / `REVISE` / `NOT ESTABLISHED` / `BLOCKED` / `INCONCLUSIVE` / `PENDING REVIEW` |
| `Reviewer` | the statistical reviewer's identity |
| `Decision Date` | ruling date |
| `Rationale` | statistical rationale (mandatory before `APPROVED` — checker R12) |
| `Evidence Reviewed` | plan/protocol/NR-05 sections consulted |
| `Approval` | `APPROVED` / `NOT APPROVED` |
| `Approval Timestamp` | recorded approval timestamp |

### 4.3 Domain reviewer — supplies once per role

Same role-level fields as §4.1, in §1.2 and §3.2. All currently `NOT ASSIGNED` /
`NOT ESTABLISHED`.

### 4.4 Domain reviewer — supplies per `BOTH` decision only

For each of `/03 /04 /05 /06 /07 /12`, in that sheet's prefixed domain block:

| Field | Meaning |
|---|---|
| `Domain Reviewer Decision` | as §4.2 |
| `Domain Reviewer` | the domain reviewer's identity |
| `Domain Reviewer Decision Date` | ruling date |
| `Domain Reviewer Rationale` | domain rationale (mandatory before `APPROVED`) |
| `Domain Reviewer Evidence Reviewed` | evidence consulted |
| `Domain Reviewer Approval` | `APPROVED` / `NOT APPROVED` |
| `Domain Reviewer Approval Timestamp` | recorded approval timestamp |

The domain reviewer has **no** input on the eight `STATISTICAL REVIEW` decisions, and
those sheets carry no domain block by design (checker R12 rejects one if added).

### 4.5 Mechanical readiness reporting

`python backend/scripts/review_resolution_check.py --readiness` prints, per decision,
the derived state of each reviewer component — `NOT ASSIGNED`,
`ASSIGNED / PENDING REVIEW`, `REVIEWED / PENDING APPROVAL`, `APPROVED`,
`REJECTED`, `REVISE` — plus a resolved count. It is **read-only**: it never writes
to the artifact, never advances a decision, and always exits 0. Current output is
`0/14 decisions resolved | component states: NOT ASSIGNED=20`.

---

## 5. Decision ownership rule

> **Assignment establishes responsibility to review. Assignment does not constitute
> a decision, approval, or freeze authorization.**

> **A decision remains `PENDING REVIEW` until the assigned reviewer records a ruling
> and rationale.**

> **An approval cannot be recorded while the decision remains `PENDING REVIEW`.**

These three statements agree with the mechanical rules already enforced by
`backend/scripts/review_resolution_check.py`:

- R5 — `APPROVED` cannot coexist with `Reviewer: NOT ASSIGNED`;
- R6 — `Approval: APPROVED` requires the decision itself to be `APPROVED`;
- R7 — `APPROVED` cannot coexist with `Approval Timestamp: NOT ESTABLISHED`;
- R8 — while a decision is `PENDING REVIEW`, neither a reviewer name nor a decision
  date may be filled in;
- R12 — a `BOTH` decision cannot be `APPROVED` unless **both** independent reviewer
  components are complete.

Assigning a person therefore **cannot** move any decision to `APPROVED`, and this
record does not attempt to.

### 5.1 Ownership boundary — what each class requires

Taken from the authoritative ownership table in
`docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` §3.

**`STATISTICAL REVIEW`** (8 decisions: `/01 /02 /08 /09 /10 /11 /13 /14`) requires:

1. statistical reviewer
2. statistical ruling
3. statistical rationale
4. statistical approval

**`BOTH`** (6 decisions: `/03 /04 /05 /06 /07 /12`) requires:

1. statistical reviewer
2. statistical ruling
3. statistical rationale
4. statistical approval
5. domain reviewer
6. domain ruling
7. domain rationale
8. domain approval

> **No reviewer may satisfy both roles.** The two components are recorded under
> different field names and must be completed independently; one named person
> cannot appear in both.

**No domain-only category exists.** It is not introduced here, and a
`STATISTICAL REVIEW` sheet that acquires a domain component block is rejected by
checker R12.

---

## 6. Review sequence

The intended sequence. **None of it has occurred.** Steps 1 and 2 are blocked.

1. Assign statistical reviewer.
2. Assign domain reviewer where required (the 6 `BOTH` rows).
3. Reviewer reads the authoritative Research Plan, preregistration, NR-05
   diagnostics, exposure ledger, eligibility manifest, the statistical-review memo,
   and the review-resolution artifact.
4. Reviewer records a decision for each assigned item, using the completion states
   `PENDING REVIEW` / `APPROVED` / `REJECTED` / `REVISE` / `NOT ESTABLISHED` /
   `BLOCKED` / `INCONCLUSIVE`.
5. Reviewer records rationale and evidence reviewed.
6. If `REVISE`, follow the existing one-revision rule (plan §24: at most one revision
   cycle, approved by someone other than the experimenter, original result
   preserved). A revision does not modify the protocol automatically.
7. Approved decisions may then be applied to the Research Plan / preregistration as a
   **separately recorded change** (plan §17: document, never silently edit).
8. Re-run all structural and evidence checks (`check_freeze.py`,
   `claim_evidence_check.py`, `eval_record_test.py`, `check_freeze_test.py`,
   `review_resolution_check.py`, battery).
9. Only after every freeze requirement is satisfied may freeze preparation proceed.

---

## 7. Freeze effect

> **Reviewer assignment alone does not reduce the 78 current freeze findings.**

Current composition, from `scripts/check_freeze.py` (exit 1): 65 unresolved
placeholder findings in the Research Plan (60 unique lines) + 12 in the
preregistration + 1 missing `docs/FREEZE_RECORD.json` = **78**. None of these is
removed by naming a reviewer, because each requires a decision, a written value, or
a freeze-time act.

`scripts/check_freeze.py` is **not modified** to recognise assignment, and is not
part of the frozen artifact set yet (plan §29 freezes it at freeze time). The
Research Plan remains **`DRAFT / NOT FROZEN / NOT APPROVED`** until its actual
requirements are satisfied.

Even with all 14 decisions ruled and approved, Track M would remain `BLOCKED`:
both candidate confirmatory datasets are `BLOCKED`/unacquired and all three exposed
datasets are `EXPLORATORY`, so plan §23's "zero eligible confirmatory datasets"
applies.

---

## 8. No automatic assignment

The repository agent is **not** permitted to infer that:

- the project author is the statistical reviewer;
- the project author is the domain reviewer;
- an AI agent is a reviewer;
- a GitHub contributor is a qualified reviewer;
- a university contact is a reviewer;
- a previous conversation participant is a reviewer.

**Only an explicitly provided reviewer identity may change `NOT ASSIGNED`.** Until
then, every identity, affiliation, date, credential and conflict field stays
`NOT ESTABLISHED`.

---

## 9. No new schema

This record deliberately creates **no**: JSON reviewer registry, database table,
evidence-ledger schema, phase number, experiment record, scientific metric, or
governance system. It is Markdown, and it adds nothing to
`reports/evaluation_runs/eval_ledger.jsonl`. `/01`–`/14` numbering is unchanged from
the statistical-review memo.

---

## 10. What this record does not claim

- that a reviewer exists, has been approached, or has been assigned;
- that any of `/01`–`/14` has been reviewed, decided, or approved;
- that the Research Plan is frozen or ready to freeze;
- that IEEE-CIS or BAF is eligible, acquired, or verified;
- that NR-05 is confirmatory or constitutes independent replication;
- that the NR-05 negative findings N1–N13 have changed — they are preserved verbatim
  in `docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` §5.

---

## 11. External dependency statement

Added 2026-10-05 by the reviewer-engagement phase. **No field in this record was
populated, cleared, or renamed by that phase.** Pre-edit SHA-256 of this file:
`52a32bdee9f78b413e9dcf3a6c44404ee2733c91cea55247fa7965e390259f42`.

> **Reviewer assignment remains an external dependency and cannot be completed from
> repository evidence alone.**

What that means, concretely:

- No reviewer identity, affiliation, expertise statement, conflict-of-interest
  declaration, assignment date, scope acceptance, approval authority or signature
  exists for either role. Every such field above remains `NOT ASSIGNED` /
  `NOT ESTABLISHED`.
- The reviewer-engagement phase added no name, no example, no placeholder person and
  no fabricated credential to this record — see §8 (no automatic assignment).
- The externally usable entry point for a prospective reviewer is
  `docs/evaluation/REVIEWER_ENGAGEMENT_BRIEF.md`, whose status is
  **`READY FOR REVIEWER RECRUITMENT — REVIEW NOT YET COMMENCED`**.
- The §4 input contract is the complete list of what a real reviewer must supply;
  §3 is where qualification evidence is recorded; §1.1/§1.2 are where identity is
  recorded. Nothing else is required and no new form exists.
- The A-rules in `backend/scripts/review_package_check.py` mechanically reject a
  named assignment that lacks the required qualification fields, a person holding
  both decision roles, an approval recorded without an identity, and any
  non-eligible identity marker (AI/assistant, project author). Those rules are
  dormant while both roles remain `NOT ASSIGNED`.
- Assignment alone still moves nothing: see §5's ownership rule and
  `backend/scripts/review_resolution_check.py` rules R5–R8 and R12.

**Current state after this statement: statistical reviewer `NOT ASSIGNED`, domain
reviewer `NOT ASSIGNED`, decisions resolved 0/14, freeze findings 78 — all unchanged.**

---

| Version | Change | Type |
|---|---|---|
| 1.0-draft | Initial assignment record. Roles taken from plan §28/§32 and preregistration §5; 14-row matrix mirrors the resolution artifact's ownership classification. No identity assigned. | ADMINISTRATIVE READINESS |
| 1.1-draft | Added §11 external-dependency statement (reviewer-engagement phase). No field populated; both roles still `NOT ASSIGNED`; no approval, identity, date or credential introduced. | ADMINISTRATIVE READINESS |