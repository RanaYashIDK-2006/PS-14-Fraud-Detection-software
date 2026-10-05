# Reviewer Engagement Brief

**Artifact path:** `docs/evaluation/REVIEWER_ENGAGEMENT_BRIEF.md`
**Version:** `reviewer-engagement-brief 1.0-draft`
**Status:** **`READY FOR REVIEWER RECRUITMENT — REVIEW NOT YET COMMENCED`**
**Date:** 2026-10-05.
**Authored at git state:** HEAD `5a5ff55cc03df73318fdf31a16e3665f159a4720` (140 commits, 0 tags).
**Audience:** a prospective qualified statistical reviewer, domain reviewer,
independent dataset/provenance auditor, or licence/terms reviewer.

**This brief does not assign anyone, does not approve anything, and does not claim
that any review has occurred.** No reviewer identity, affiliation, credential, date,
approval or conflict declaration exists in this repository; every such field is
`NOT ASSIGNED` / `NOT ESTABLISHED`. If you are reading this and are considering the
role, the repository is inviting you to make an **independent judgement on evidence
that is already public in the repository** — nothing here asks you to endorse the
project's own conclusions.

**Where to start (existing machinery, no parallel system):**

| Need | Document |
|---|---|
| The 14 decisions to be ruled on | `docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` §4 (sheets), §8 (fill format) |
| Reviewer-independent evidence summary | `docs/evaluation/STATISTICAL_REVIEW_DECISION_MEMO.md` §5–§7 |
| Role-level identity/qualification fields | `docs/evaluation/REVIEWER_ASSIGNMENT_RECORD.md` §1, §3, §4 |
| Dataset/provenance verification checklist | `docs/evaluation/INDEPENDENT_EVIDENCE_REVIEW_PACKAGE.md` §7 (A–O), §8 (11 IBM items) |
| Licence evidence capture | `docs/evaluation/DATASET_LICENCE_EVIDENCE_CHECKLIST.md` §3, §6 |
| Exposure/eligibility records | `docs/evaluation/DATASET_EXPOSURE_LEDGER.md`, `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` |
| Mechanical enforcement | `backend/scripts/review_resolution_check.py`, `backend/scripts/review_package_check.py`, `scripts/check_freeze.py` |

---

## 1. Project and review purpose

PS-14 is a **privacy-first fraud-detection research prototype seeking independent
evaluation**. It is a research codebase with a documented evaluation plan, an
executed exploratory diagnostic experiment (NR-05), and a pre-registration draft.

**PS-14 is not being presented as production-ready, institutionally validated,
regulator-approved, or deployment-ready.** It has no institutional data, no bank
partner, no regulatory submission, and no confirmatory result. Its public datasets
are previously exposed to its own model development, so it cannot presently claim
independent validation of anything.

The purpose of this engagement is to obtain the **external human judgement the
project cannot supply for itself**:

1. statistical rulings on the 14 open methodological decisions (`/01`–`/14`);
2. domain rulings on the 6 decisions classified `BOTH`;
3. an independent audit of dataset identity, provenance and exposure;
4. authoritative licence/terms determinations for the datasets it holds.

Until those four inputs exist, the project's Research Plan stays
`DRAFT / NOT FROZEN / NOT APPROVED`, confirmatory Track M stays `BLOCKED`, and every
result in the repository keeps its existing classification — including the
unfavourable ones.

---

## 2. Current evidence position

Stated as the repository records it. **No classification is upgraded here and no new
metric is introduced.** Full detail: `docs/PHASE_NR05_EXPLORATORY_DIAGNOSTICS.md`,
the decision memo §3/§5/§6, the resolution package §5 (N1–N13), and
`docs/evaluation/claims_registry.jsonl`.

### 2.1 Strong in-domain results (already present, `SELF-TESTED`/`DEMONSTRATED`)

| Evidence | Recorded classification |
|---|---|
| ULB in-domain: ROC-AUC 0.976 (C-001), recall@1%FPR 91.8% (C-002), PR-AUC 0.883 (C-003) | `DEMONSTRATED` — in-domain, **not independent external validation** (plan §15) |
| IBM v2 in-domain: XGB ROC-AUC 0.982 (C-004) | `SELF-TESTED` — artifact records no seed/git provenance (claims registry) |
| Deployed native 48-feature ensemble (`altman_native_v2_20260904_115703`): test AUC 0.972237, PR-AUC 0.78749, recall@1%FPR 0.412626; threshold locked `0.7847116291110687` from the **validation** window only | production record; in-domain, single-split, previously consumed |

### 2.2 Failed / non-conforming external transfer (already present)

| Evidence | Recorded classification |
|---|---|
| Kaggle transfer (~0.435–0.595 ROC-AUC); fraudTrain 0.47 (C-010), fraudTest 0.595 (C-013), FPR 9.9% (C-014) | **`FAILED / NON-CONFORMING`** — degraded/contract-mismatched transfer, a genuine negative result (plan §15; ledger §5) |
| IBM cross-dataset ROC-AUC ~0.873 (C-015) | `EXPLORATORY / NOT INDEPENDENT` — same generator family |
| Two benchmark scripts reported metrics on rows **inside** the training window (`honest_benchmark.py`, `improved_paysim_ealtman.py`, reading `data/ealtman2019/User0_…csv`) | recorded limitation **U-8**; the held-out re-run is specified and **was not executed** |

### 2.3 NR-05 exploratory findings on the current ensemble (N1–N13, first-class)

NR-05 is an **exploratory diagnostic run on previously exposed datasets**. It makes
no dataset "untouched" and cannot support a claim of independent replication.

| # | Finding |
|---|---|
| N1 | Mean-fusion ensemble **underperformed XGB across all four** NR-05 evaluations (paired CIs exclude 0) |
| N2 | IBM Platt calibration produced a **reproducible inversion** (coef −0.7029, intercept −6.4186, 185 calibration positives; reproduced bit-identically on independent refit) |
| N3 | IBM test AUC fell **below chance** (0.627 → **0.373**) **despite apparently strong Brier/ECE** (0.00102 / 0.00045) |
| N4 | ULB clean recall **collapsed** under the tested row gate: **0.813 → 0.040** (72/75 test frauds blocked) |
| N5 | Window PSI **over-fired on ULB** (clean PSI 1.46, coverage 0) against the fixed 0.25 threshold |
| N6 | Window PSI **did not detect** the tested Kaggle Class-B condition (max PSI 0.083, below the 0.10 warn level) |
| N7 | Class-B gating benefit remained **`NOT ESTABLISHED`** |
| N8 | Kaggle entity-disjoint result (0.9697 held-out ROC-AUC) was materially stronger than the IBM temporal result — **not a like-for-like comparison** |
| N9 | **IBM temporal performance degraded substantially**: train-period AUC 0.83–1.00 → final-20% window **0.58–0.64** over a 1,020-day span |
| N10 | IBM gating **`INCONCLUSIVE`** from feature/value-bound defects (`chip_code` train bounds [1.0, 2.0] vs 70.6% of test rows at code 0; 33,685 negative-amount rows making `log1p` NaN) |
| N11 | ULB window-gate "wins" were an artefact of rejecting **100% of rows** (coverage 0) — not a success |
| N12 | Two harness defects were found and fixed **before** any result was reported; superseded records preserved append-only with their metrics **not cited** |
| N13 | ULB quartile 4 (11 positives) and segment q2 (9 positives) are **small-n and withheld**, not performance |

**Calibration caveat for `/07`:** the IBM inversion above is the reason calibration
is a decision, not a settled procedure. Brier/ECE alone did not reveal it.

**Power/dependence caveats for `/08`,`/09`,`/10`:** ULB test 75 / cal 57 positives;
IBM test 134 / cal 185; dependence-aware methods were all exercised but are labelled
**METHOD NOT YET FROZEN**; NR-05 ran a single seed (42) while the preregistration
proposes 5.

### 2.4 Native 48-feature availability

| Dataset | Native-48 coverage | Note |
|---|---|---|
| **IBM v2** (acquired, 24,386,900 rows) | **48/48** — 4 `OBSERVED` + 44 `DERIVABLE` | the only dataset that satisfies the contract in full |
| Kaggle fraudTrain/Test (acquired) | 1 `OBSERVED` / 18 `DERIVABLE` / 10 `PARTIALLY_DERIVABLE` / 19 `UNAVAILABLE` | no MCC, no chip/online channel, no merchant-state semantics |
| ULB (acquired) | 1 / 2 / 45 | PCA-anonymised; no entity identifier |
| IEEE-CIS (not acquired, `BLOCKED`) | 1 / 2 / 3 / 41 + 1 `UNKNOWN` | relative timestamps; no user/merchant/MCC/zip/channel |
| BAF (not acquired, `BLOCKED`) | 48 `REJECTED` | unit of record is account-**opening**, not a transaction |
| Zenodo 2026 (not acquired) | 1 / 13 / 34 | proof-of-concept demo, verification-biased labels |
| PS-14 synthetic | 48 `SYNTHETIC_ONLY` | cannot establish real-world performance |

Separately, **9 of the 21 causal features** the runtime scores have **no public
source** (`new_device_flag`, `unusual_location_flag`, `unusual_recipient_flag`,
`failed_auth_count_24h`, `known_device_count`, `shared_device_accounts`,
`shared_recipient_accounts`, `mule_ring_score`, `recipient_novelty`); the public
contract (`public_v1`, 14 features) records `failed_auth_proxy.available_in = []`.
**No public failed-authentication/ATO event dataset was found.**

### 2.5 Current confirmatory status

| Fact | State |
|---|---|
| Confirmatory-eligible datasets | **none** — exposed sets are `EXPLORATORY`; IEEE-CIS/BAF `BLOCKED` (not acquired) |
| Track M | **`BLOCKED`** (plan §23: zero eligible confirmatory datasets) |
| Research Plan / preregistration | `DRAFT / NOT FROZEN / NOT APPROVED`; `0.1.2-draft` |
| Reviewer decisions resolved | **0 / 14** |
| Reviewers assigned | **none** (`NOT ASSIGNED` ×2) |
| Independent audit of the exposure ledger (plan §14) | **`NOT ESTABLISHED`** (**U-7**) |
| Licences cleared | **none** (plan §35 checkbox unchecked for every dataset; **U-1**) |
| Freeze findings | **78** (`scripts/check_freeze.py` exit 1 — expected pre-freeze) |
| 50M dataset | **not authorized / not yet scientifically justified**; ~24.39M real native-complete rows exist and were not padded |

---

## 3. What the reviewer is being asked to do

Four separable roles. **A person may hold one role; the two `/01`–`/14` roles
cannot be held by the same person for the same decision** (see §8).

### 3.1 Statistical reviewer

Independently assess the applicable decisions:

**`STATISTICAL REVIEW` decisions (8):** `/01` Baseline identity · `/02` Ensemble
identity · `/08` Power floors · `/09` Dependence-aware inference · `/10` Minimum
seed/run rule · `/11` Member operating points · `/13` Large-dataset sampling ·
`/14` Segment/small-n withholding.

**Statistical side of the `BOTH` decisions (6):** `/03` Confident-decision
definition · `/04` Gating bounds · `/05` Feature-contract construction · `/06`
Window-gate definition · `/07` Calibration · `/12` Alert-volume semantics.

**Do not assume every decision must be approved.** `APPROVED`, `REJECTED`,
`REVISE`, `PENDING REVIEW`, `NOT ESTABLISHED`, `BLOCKED` and `INCONCLUSIVE` are all
legitimate outcomes where the existing framework supports them
(`STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` §8.2; plan §22/§24). The repository
already contains negative results (N1–N13) that no ruling may soften.

### 3.2 Domain reviewer

Independently assess the **domain-dependent portion** of the six `BOTH` decisions —
`/03`, `/04`, `/05`, `/06`, `/07`, `/12` — and specifically determine whether the
proposed **fraud-detection semantics are defensible from a domain perspective**:
what a confident decision should mean operationally; whether the gate bounds and
their magnitudes are defensible; whether the feature contract's semantics (and its
9 unsourced causal features) are usable in a real fraud function; what calibration
means in fraud operations; and what alert volume vs analyst burden may honestly be
claimed (preregistration §7 forbids presenting an assumed capacity as measured
burden).

The domain reviewer has **no** input on the eight `STATISTICAL REVIEW` decisions;
those sheets carry no domain block by design, and the checker rejects one if added.

### 3.3 Independent dataset/provenance auditor

Verify the dataset identity and provenance claims from the repository's own evidence
— `INDEPENDENT_EVIDENCE_REVIEW_PACKAGE.md` §7 (A–O) and §8 (11 IBM items) — including:

- IBM v2 identity, row count, byte size, schema, feature derivations;
- duplicate relationships (`data/ealtman2019/` copy, `User0_…csv` extract) and
  whether anything counts them as independent evidence;
- exposure classification and the independence claims (including **U-8**);
- the preserved historical discrepancies — D1 (the absent 168-column dataset),
  D2 (the `24,386,899` vs `24,386,900` off-by-one), the previously incorrect
  `cc_num` / `hour_diff` / acquisition-size cells;
- provenance evidence, and whether the available evidence actually supports each
  claimed dataset status.

**The auditor must not be asked to certify anything beyond the evidence available to
them.** Where no publisher/retrieval record exists, `NOT ESTABLISHED` is the correct
finding (publisher provenance is currently unarchived — **U-4**).

### 3.4 Licence/terms reviewer

Determine, **from actual authoritative evidence**, for each acquired dataset:

- applicable licence; permitted use; redistribution restrictions; research-use
  restrictions; derivative-data restrictions; attribution requirements;
  commercial-use restrictions if stated; and whether the repository's intended use
  is compatible with them.

Admissible evidence bar (`DATASET_LICENCE_EVIDENCE_CHECKLIST.md` §6): the **exact
terms URL** as retrieved with date, the **verbatim operative clause**, the **named
licensor**, the **access route actually used**, and the **reviewer's identity**.
A summary, a blog post, an aggregator page, or another project's assertion does not
clear a cell. **Unknown must remain unknown if evidence is insufficient** — the
outcome range (F-P / F-R / R / U) is deliberately open, and the current state of
every acquired dataset is **`U`**.

---

## 4. Reviewer qualification requirements

Minimum evidence per role. **No credential, name, date or affiliation is invented by
this document**, and none may be filled in by the repository on a reviewer's behalf.

| Field | Statistical reviewer | Domain reviewer | Dataset/provenance auditor | Licence/terms reviewer |
|---|---|---|---|---|
| Reviewer name | required | required | required | required |
| Affiliation / organisation | required | required | required | required |
| Relevant expertise | demonstrable **statistical / ML evaluation** expertise (dependence-aware inference, calibration, power, multiplicity) | demonstrable **fraud-risk / financial-transaction / banking** expertise, or an equivalent qualification **explicitly accepted by the project review framework** | demonstrable **data provenance / reproducibility / audit** expertise | **legal or licensing competence**, or another explicitly documented basis for the determination |
| Role accepted | one of: `STATISTICAL REVIEW` | one of: domain half of `BOTH` | audit role (not a `/01`–`/14` role) | licence role (not a `/01`–`/14` role) |
| Conflict-of-interest declaration | required | required | required | required |
| Assignment date | required | required | required | required |
| Authority to provide the review | required | required | required | required |
| Contact / verification information | where appropriate | where appropriate | where appropriate | where appropriate |
| Independence acknowledgement | the reviewer confirms independence from the project **to the degree the review requires** | same | same | same |

**Structural constraints already in force** (mechanically enforced, not advisory):
plan §32 — *"A single developer's self-approval is not sufficient for the statistical
sections"*; the project author is not an eligible independent reviewer of their own
work; an AI agent is not a reviewer; assignment alone establishes no approval; and
one person cannot satisfy both the statistical and the domain side of a `BOTH`
decision (`review_resolution_check.py` R12).

---

## 5. Reviewer input packet — the eight questions

Each reviewer can answer the same eight questions from the same evidence set. This
is the whole input contract; nothing else is required of you, and no new form or
registry is introduced.

| # | Question | Where the answer is recorded (existing fields only) |
|---|---|---|
| 1 | **What evidence am I reviewing?** | statistical: resolution §4 sheets + memo §5–§7; domain: the `BOTH` sheets; auditor: review package §7/§8 + ledger/manifest; licence: licence checklist §3 + review package §6 |
| 2 | **What am I not being asked to review?** | statistical: domain semantics; domain: the eight statistical-only decisions; auditor: nothing outside the evidence; licence: nothing beyond the terms, and not eligibility itself |
| 3 | **What exact decision am I making?** | one recorded ruling per assigned item (`Reviewer Decision:`), or a role-level identity/qualification entry — nothing else |
| 4 | **What evidence supports the decision?** | `Evidence Reviewed:` field (resolution §8.1) / licence checklist §3 / review package §7 item outcomes |
| 5 | **What uncertainty remains?** | `Rationale:` (mandatory before `APPROVED`) — state it, including `INCONCLUSIVE` where the evidence is insufficient |
| 6 | **What changes, if any, are required?** | `REVISE` — plan §24 permits **at most one** revision cycle, approved by someone other than the experimenter, original result preserved (memo §8 lists the seven fields a complete `REVISE` needs) |
| 7 | **Approve, reject, revise, or withhold?** | `APPROVED` / `REJECTED` / `REVISE` / `NOT ESTABLISHED` / `BLOCKED` / `INCONCLUSIVE` / `PENDING REVIEW` (resolution §8.2) |
| 8 | **What is my rationale?** | `Rationale:` — and, for identity fields, the qualification block in the assignment record §3 |

**What a ruling does not do:** it does not freeze the plan, create
`docs/FREEZE_RECORD.json`, create a git tag, or resolve a plan placeholder. After a
ruling is approved, the corresponding plan or preregistration cell is edited as a
**separate, recorded act** (plan §17: document, never silently edit).

---

## 6. No-silent-correction rule — items awaiting reviewer disposition

The repository deliberately **has not** edited the following. They are recorded, not
resolved, and each requires a reviewer disposition rather than a quiet fix.

| Item | What it is | Where it survives |
|---|---|---|
| **O-1** | Reviewer-owned documents still carry **superseded IBM cells**: resolution §7.1 lists "IBM v2 has `cc_num`" and "IBM `hour_diff` … `ESTABLISHED` — a recorded defect"; the memo §7 `/13` row cites "24,386,899 rows". The acquired file has neither column and has **24,386,900** rows | resolution §7.1; memo §7 — **checker-enforced, deliberately unedited** |
| **D1** | The 168-column `credit_card_transactions.csv` description was a **real repository/data mismatch** with broken citations; corrected additively in the eligibility manifest §4.1 (C-2/C-3/C-6), old text preserved verbatim | manifest §4.1; reconciliation §5 |
| **D2** | The `24,386,899` constant in `nr05_diagnostics.py:370` is **preserved as history** (0.00015% effect, consumed by no computation), cited by 4 sites in 2 reviewer-owned documents | reconciliation §6; open item **U-2** |
| **U-3** | The *cause* of D1 is not recoverable from these files. Note: the project **is** a git repository (HEAD `5a5ff55`, 140 commits), but the plan, NR-0x artifacts, preregistration and every `docs/evaluation/*` review artifact are **untracked**, so they have no history. The reconciliation's stated reason ("not a git repository") is superseded; its conclusion stands | review package O-2 |
| **U-4** | IBM v2 **publisher provenance** is unarchived | manifest §4; ledger §7 |
| **U-7** | The exposure ledger and eligibility manifest are **not independently audited** — plan §14 requires this **before the confirmatory freeze** | ledger §7 |
| Licence states | All acquired datasets: `PENDING REVIEW` / `NOT ESTABLISHED`; the protocol-era "Open licence" claims were found unsupported and retracted (C-5, C-7) | manifest §1/§4; licence checklist §3 |
| Other open items | U-1 (licence), U-5 (`fraudTest` confirmatory-vs-exposed), U-6 (IEEE-CIS/BAF), U-8 (in-window benchmark metrics), U-9 (78 freeze findings), U-10 (0/14) | reconciliation §13 |

**No reviewer-owned document was edited to make it look cleaner.** If you decide
these items are wrong, your ruling is the mechanism that changes them — in a
separate, recorded act.

---

## 7. Recommended review order

Recorded as the recommended sequence. **None of these stages has occurred.**

| Stage | What happens | Gate to next stage |
|---|---|---|
| **A — reviewer qualification and assignment** | Actual qualified people are identified and their identity/qualification/conflict fields are completed in `REVIEWER_ASSIGNMENT_RECORD.md` §1 and §3 | At least a statistical reviewer, a domain reviewer, a dataset/provenance auditor and a licence reviewer identified |
| **B — independent dataset/provenance review** | Review package §7 (A–O) and §8 items are verified; the factual dataset boundary is settled | Findings recorded; `ESTABLISHED` / `NOT ESTABLISHED` / `FAILED` / `BLOCKED` / `INCONCLUSIVE` stated per item |
| **C — licence/terms determination** | Authoritative terms evidence is captured per dataset | Outcome recorded per dataset (F-P / F-R / R / U) with evidence; plan §35 cell actionable |
| **D — statistical review** | `/01 /02 /08 /09 /10 /11 /13 /14` ruled; statistical side of `/03 /04 /05 /06 /07 /12` ruled | Each ruling carries identity + date + rationale + approval fields as applicable |
| **E — domain review** | Domain side of `/03 /04 /05 /06 /07 /12` ruled | Both components complete where `BOTH` |
| **F — resolution** | `backend/scripts/review_resolution_check.py` re-run | Checker passes: `STATISTICAL REVIEW` has the statistical side complete; `BOTH` has **both** sides; one person cannot satisfy both; no approval inferred from assignment; unresolved decisions stay unresolved; N1–N13 intact |
| **G — freeze readiness** | Only after all applicable decisions and plan §35 requirements are legitimately resolved | `scripts/check_freeze.py` exits 0, `docs/FREEZE_RECORD.json` complete, freeze tag on the tagged commit — **only then** may Track M be considered |

**A ruling is not a freeze, and an assignment is not a ruling.** No decision moves to
`APPROVED` by being assigned.

---

## 8. How decisions are recorded, and what enforces them

| Layer | Recording location | Enforcement |
|---|---|---|
| Decision ruling | `STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` §4 sheet block (§8.1 fill format) | `review_resolution_check.py` R1–R12: all 14 sheets present exactly once; no blank unresolved fields; `APPROVED` cannot coexist with `NOT ASSIGNED`, `NOT APPROVED`, or a missing timestamp; no fabricated name/date while pending; N1–N13 intact; no freeze record created; `BOTH` requires **both** components |
| Reviewer identity / qualification | `REVIEWER_ASSIGNMENT_RECORD.md` §1.1/§1.2 (role blocks), §3.1/§3.2 (qualification), §4 (input contract) | `review_package_check.py` reviewer-qualification rules (A-rules): required fields present, no named assignment with `NOT ESTABLISHED` qualification fields, one person cannot hold both decision roles, no approval without identity, no non-eligible identity (AI/author markers) |
| Licence determination | `DATASET_LICENCE_EVIDENCE_CHECKLIST.md` §3 | `review_package_check.py` licence rules: no licence cell may read cleared/approved/verified/open |
| Dataset/provenance findings | `INDEPENDENT_EVIDENCE_REVIEW_PACKAGE.md` §6 (decisions), §7 (A–O outcomes), §8 (IBM items) | `review_package_check.py`: sections present, all datasets covered, all 15 items still at `REQUIRES REVIEW` until a reviewer actually rules |
| Exposure-ledger audit | `DATASET_EXPOSURE_LEDGER.md` §7 row + §8 change discipline | no status may be upgraded without a cited, reviewable source |
| Freeze | plan §29/§30; `docs/FREEZE_RECORD.json` | `scripts/check_freeze.py` — stays **exit 1 / 78 findings** until a legitimate freeze; this brief does not change it |

The two review-side checkers are **read-only document checkers**: they never write,
never advance a decision, never import a model or dataset, and always exit 0 on a
consistent artifact.

---

## 9. What remains blocked if no qualified reviewer is obtained

The repository remains scientifically honest and fully functional without reviewers,
but the following stay exactly where they are:

| Blocked item | Consequence |
|---|---|
| `/01`–`/14` rulings | plan placeholder cells stay unresolved; 65 plan + 12 preregistration findings remain |
| Domain side of the 6 `BOTH` decisions | those decisions can never be `APPROVED`; the checker rejects a one-sided approval |
| Independent audit (U-7) | plan §14's pre-freeze requirement stays unmet; expected scope of the amended dataset records stays unverified |
| Licence determination (U-1) | plan §35's "licence/access verified" stays unchecked; no dataset can be confirmatory-used |
| Publisher provenance (U-4) | IBM v2's provenance field stays `NOT ESTABLISHED` |
| **Freeze** | `DRAFT / NOT FROZEN / NOT APPROVED`; `FREEZE_RECORD.json` absent; no tag |
| **Confirmatory Track M** | `BLOCKED` — zero eligible confirmatory datasets remains true even after rulings |
| **50M work** | `NOT AUTHORIZED / NOT YET SCIENTIFICALLY JUSTIFIED`; downstream of rulings, freeze, confirmatory evidence and a documented construction methodology |
| Track I / Track N | `BLOCKED / NOT ESTABLISHED` — requires authorised institutional data |

**The repository cannot manufacture any of these.** A checker can verify
consistency; it cannot rule, licence, audit, or freeze.

---

## 10. Engagement status

> # READY FOR REVIEWER RECRUITMENT — REVIEW NOT YET COMMENCED

| | |
|---|---|
| Review-ready artifacts | ✔ present (resolution package, memo, review package, licence worksheet, exposure/eligibility records, two read-only checkers) |
| Statistical reviewer | **`NOT ASSIGNED`** |
| Domain reviewer | **`NOT ASSIGNED`** |
| Independent dataset/provenance auditor | **not assigned** |
| Licence/terms determination | **not completed** |
| Review performed | **no** |
| Anything approved | **no** |
| Research Plan frozen | **no** |
| Track M | `BLOCKED` |
| Next meaningful event | a real qualified reviewer (or auditor, or authoritative licence evidence) enters the process — or the project explicitly records that external review is unavailable |

*This brief creates no reviewer, no approval, no licence outcome, no eligibility, no
freeze, and no new governance system. It is an invitation and a map.*

*End of `reviewer-engagement-brief 1.0-draft`.*
