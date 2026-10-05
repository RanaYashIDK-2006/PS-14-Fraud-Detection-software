# Objective Evidence Closure — Pre-Review Evidence Audit

**Artifact path:** `docs/evaluation/OBJECTIVE_EVIDENCE_CLOSURE.md`
**Version:** `objective-evidence-closure 1.0-draft`
**Status:** DRAFT audit — **grants nothing, approves nothing, freezes nothing, acquires nothing.**
**Date:** 2026-10-05
**Phase:** OBJECTIVE EVIDENCE CLOSURE & FREEZE-READINESS PREPARATION
**Final phase status:** **`OBJECTIVE EVIDENCE CLOSED — AWAITING HUMAN REVIEW`**

---

## 0. What this document is, and is not

| It IS | It is NOT |
|---|---|
| A classification of every currently unresolved issue | An approval, a review, or a reviewer |
| A record of objective facts re-measured from the repository | A rewrite of any dated, pinned, or reviewer-owned record |
| A preparation of freeze-time inputs (marker inventory, exact paths, measured counts) | A freeze, a `FREEZE_RECORD.json`, or a Research Plan edit |
| A 50M-status determination from existing evidence | A 50M dataset, an acquisition, or an eligibility grant |

Boundaries honoured: no reviewer invented; no decision `APPROVED`/`REJECTED`/`REVISE`ed;
plan not frozen; no Track M run; no threshold, model, feature-contract or methodology
change; no dataset acquired or generated; no reviewer-owned finding silently rewritten;
no exploratory evidence promoted to confirmatory.

Classification codes used below: **ORN** objectively resolvable now · **RQR** requires a
qualified reviewer · **RAEE** requires authoritative external evidence · **RIA**
requires institutional access · **IU** intentionally unresolved · **AE**
already established (re-verified here).

---

## 1. Baseline re-measured at phase start (before any change)

| Item | Measured state | Basis |
|---|---|---|
| `scripts/check_freeze.py` | **rc=1 — 78 findings** (1 missing record + 65 plan markers on 60 lines + 12 prereg markers) | `.freebuff/oec_freezechk_base.out` |
| Reviewers | statistical `NOT ASSIGNED`, domain `NOT ASSIGNED`, auditor not assigned; licence review not completed | assignment record §1; resolution §4 |
| Decisions | **0/14** resolved; readiness `0/14 decisions resolved` (component states `NOT ASSIGNED=20`) | `review_resolution_check.py` |
| Research Plan | `DRAFT / NOT FROZEN / NOT APPROVED` | plan §34 `Current Status` |
| `FREEZE_RECORD.json` | absent | checker |
| Track M / Track N | `BLOCKED` — zero confirmatory-eligible datasets | review package §12; plan §23 |
| 50M-scale work | `NOT AUTHORIZED / NOT YET SCIENTIFICALLY JUSTIFIED`; no 50M dataset | reconciliation §14; review package §12 |
| Plan §35 pre-freeze checklist | **36 unchecked items** | `grep -c "^- \[ \]" docs/RESEARCH_PLAN.md` = 36 |
| Git | HEAD `5a5ff55…`, **140 commits**, 0 tags; review artifacts **untracked** (O-2/O-4) | `git log`; `git ls-files` |
| Dataset files | IBM v2, ULB, Kaggle ×2 byte-identical to their recorded hashes (dataset bytes untouched) | prior hash sweep; unchanged |

---

## 2. Classification of every unresolved issue

### 2.1 Open items U-1 … U-10

| ID | Item | Class | Why it cannot move internally | Who resolves |
|---|---|---|---|---|
| **U-1** | Licence/terms unverified for every acquired dataset | **RAEE** | No terms document is archived for any dataset; the only in-repo claims were unsupported and retracted (manifest §4.1 C-5, C-7); all four acquired datasets remain outcome `U`; the worksheet (§3) is ready but unfilled | Licence-qualified reviewer with authoritative sources |
| **U-2** | `24,386,899` survives in NR-05 + memo + resolution as *what those records said* | **RQR** — disposition of the three stale cells (the authoritative-count component is already established: re-measured 24,386,900 data rows, R-1) | The stale cells are reviewer-owned/historical and deliberately preserved; only the reviewer may dispose of them | Statistical reviewer (disposition) |
| **U-3** | Root cause of the D1 mismatch unknown | **IU** | The repository **is** git (O-2), but the affected records (plan, NR-0x, prereg, `docs/evaluation/*`) are **untracked**, so no blame-bearing history exists. `HISTORICAL CAUSE NOT ESTABLISHED` is preserved; git does not prove who introduced the erroneous values | Nobody — documented, not resolvable |
| **U-4** | IBM v2 publisher provenance unarchived | **RAEE** | In-repo material contains an *unarchived origin claim* (phase16 `label_source` = “IBM synthetic credit card generator (2019)”; phase17 `known_origin` naming IBM Research / SDV / Altman 2019 / `github.com/sdv-dev/SDV`) and a Kaggle association (`Base-FraudDetection`, coverage §5 Provenance). These are internal pointers, **not** an archived publisher record; manifest §4 records publisher provenance `NOT ESTABLISHED` | Publisher/source evidence, or reviewer adjudication |
| **U-5** | `fraudTest` confirmatory-vs-exposed tension | **RQR** | It is a role-assignment question (can this file serve a confirmatory role given its exposure history?) | Statistical reviewer |
| **U-6** | IEEE-CIS / BAF: terms unverified, not acquired | **RAEE** (acquisition would additionally require institutional authorisation) | Terms cannot be reviewed without the data and the licensor's terms; acquisition is not authorised here | Data-governance + acquisition authorisation |
| **U-7** | Exposure ledger / eligibility manifest not independently audited | **RQR** | An audit is by definition performed by someone other than the project; amending a record is not certifying it | Independent dataset/provenance auditor (none assigned) |
| **U-8** | `honest_benchmark.py` / `improved_paysim_ealtman.py` report in-window metrics | **RQR** | The held-out re-run is fully specified (reconciliation Appendix B) but **was not executed**; re-running now to erase the limitation is out of scope — that is a reviewer-directed decision | Statistical reviewer |
| **U-9** | Freeze findings inventory | **AE** (counts measured — see §6) | Composition re-measured: **78 = 1 + 65 + 12**; §35 = 36 unchecked. Note: reconciliation §13’s U-9 row text (“56 placeholders + 11 prereg markers”) is **superseded by measurement** (65 + 12) and internally inconsistent with its own total (56+11+1 = 68 ≠ 78). The dated record was **not** edited | Reviewers + freeze procedure |
| **U-10** | 14 reviewer decisions unresolved | **RQR** | Plan §32 forbids self-approval; 8 statistical-only + 6 `BOTH` decisions | Statistical reviewer + domain reviewer (D-1/D-2) |

### 2.2 Observations O-1 … O-8 (re-verified this phase)

| ID | Item | Class | Re-verified state |
|---|---|---|---|
| **O-1** | Superseded reviewer-owned IBM cells (`cc_num`, `hour_diff`, `24,386,899`) | **RQR** — reviewer disposition required (the presence-and-flagging component is already established, R-5) | `cc_num` / `hour_diff` still present in resolution §7.1; `24,386,899` still present in memo §7 `/13`; both documented as reviewer-owned in the brief §6 no-silent-correction table and packet §2.4 O-1. **Not edited** |
| **O-2** | U-3’s stated rationale superseded (repo IS git) | **AE** | Re-verified: HEAD `5a5ff55…`, 140 commits, `git log` works; `git ls-files` shows the plan, exposure ledger and eligibility manifest **untracked** (the preregistration **is** tracked). U-3’s *conclusion* stands; its *reason* is restated in §2.1 U-3 — no record edited |
| **O-3** | Plan contains two `# 34.` sections | **ORN — prepared, not applied** | Confirmed: line 1147 `# 34. Current Status`, line 1217 `# 34. Referenced Artifacts` (after §36). Freeze-time editorial fix (renumber one section); the plan is left untouched pre-freeze |
| **O-4** | Review artifacts untracked; tree dirty | **AE** (freeze-time owner action follows) | A `freeze-*` tag cannot cover uncommitted artifacts; committing/packaging is a freeze-procedure act, not done here |
| **O-5** | `data/` carries battery-suite outputs, not evidence | **AE** | Unchanged; must not be cited as dataset evidence (packet §2.4 O-5) |
| **O-6** | Battery side effects in `models/production/` | **AE** | Unchanged; verification basis is hash identity (`xgb_native.joblib` still `a1cdebdf…`); `release_manifest.json` is not the authoritative record |
| **O-7** | Coverage records paysim ×2 as 9 columns | **ORN — measured now** | Both files measured **8 columns** (`type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`). Coverage §3.2’s “9” is superseded; **scientific effect none** (no native field; not in any eligibility row). Coverage doc left unedited (pinned phase artifact) |
| **O-8** | phase16 inventory lists one SHA (`61c4ed49…`) twice | **AE** | Unchanged; must not be read as two independent datasets |

### 2.3 Reviewer checklist surface and roles

- Review package §7 items **A–O (15/15)** and §8 items **1–11** remain at `REQUIRES REVIEW` → **RQR** (this is the reviewer work queue, not an internal gap).
- D-1 (statistical reviewer), D-2 (domain reviewer), D-3 (independent auditor) → **RQR**, all unassigned.
- D-4 / licence outcomes (F-P / F-R / R / U): all four acquired datasets remain **`U`** → **RAEE**.

---

## 3. Objectively resolvable findings — resolved or prepared now

| # | Finding | Source / file | Evidence measured | Verification command | Expected | Resulting artifact | Changes any scientific conclusion? |
|---|---|---|---|---|---|---|---|
| **R-1** | **U-2** authoritative IBM v2 row count | `data/credit_card_transactions-ibm_v2.csv` | **24,386,901 lines incl. header → 24,386,900 data rows** | `wc -l data/credit_card_transactions-ibm_v2.csv` | 24,386,901 | Recorded here; manifest/ledger state stands (`ESTABLISHED`); NR-05/memo/resolution left with `24,386,899` as historical record | **No** — confirms existing records; distinguishes: *authoritative observed count* = 24,386,900 · *historical experiment constant* = 24,386,899 (0.00015% artefact) · *scientific consequence* = none quantified at a decision-relevant level · *reviewer disposition* required for the three cells |
| **R-2** | **O-7** paysim column count | `data/paysim.csv`, `data/paysim_1m.csv` | **8 columns each** | `head -1 data/paysim.csv \| awk -F, '{print NF}'` (and `_1m`) | 8 | Recorded here; coverage §3.2 “9” superseded; doc correction deferred to the record owner (pinned artifact, change-history convention) | **No** — no native-field, no eligibility row |
| **R-3** | §34 referenced-artifact paths (§34 markers, plan lines 1224–1226) | `docs/RESEARCH_PLAN.md` §34 `Referenced Artifacts` | Located in the working repository: preregistration `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` (**tracked**); exposure ledger `docs/evaluation/DATASET_EXPOSURE_LEDGER.md` (**untracked**); eligibility manifest `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` (**untracked**) | `git ls-files <path>`; `ls <path>` | Files exist at those repo-relative paths | **Prepared fill content** for freeze-time plan edit; plan not edited | **No** — path bookkeeping; note the two untracked files must be committed for a public-repo path to be truthful |
| **R-4** | **U-9** freeze-finding composition | `scripts/check_freeze.py` output; plan §35 | **78 = 1 (missing record) + 65 plan markers (60 lines) + 12 prereg markers**; §35 = **36 unchecked** | `python scripts/check_freeze.py`; `grep -c` / `awk` on output | 78 problems | Recorded here; reconciliation §13 U-9 text superseded (56+11), **not edited** | **No** — inventory count only; freeze stays blocked |
| **R-5** | **O-1** superseded-cell presence | resolution §7.1; memo §7; brief §6; packet §2.4 | Still present; still labelled reviewer-owned; **not corrected** | `grep -n "cc_num\|hour_diff\|24,386,899" docs/evaluation/*.md` | Presence + labels confirmed | Recorded here | **No** — nothing edits reviewer-owned docs |
| **R-6** | Git/tracking facts (O-2/O-4/O-8) | repo metadata | HEAD `5a5ff55…`; 140 commits; 0 tags; listed review artifacts untracked | `git log --oneline -1`; `git rev-list --count HEAD`; `git ls-files` | As recorded | Recorded here | **No** |
| **R-7** | Reviewer-state counts (U-10) | resolution §4; assignment record | `0/14 decisions resolved` and component states `NOT ASSIGNED=20`; both roles `NOT ASSIGNED` | `python backend/scripts/review_resolution_check.py --readiness` | 0/14, NOT ASSIGNED=20 | Recorded here | **No** |

**No other item is objectively resolvable now.** Everything else requires a qualified
reviewer, authoritative external evidence, institutional access, or is intentionally
unresolved (§2).

---

## 4. Items that cannot be resolved internally, and why

1. **Any reviewer decision (0/14)** — plan §32 requires independent sign-off; self-approval is forbidden. Nothing in the repository can substitute for it.
2. **Licence determination (U-1)** — requires quoting operative terms from authoritative sources; no such document exists in-repo, and inferring from popularity, platform, paper or filename is prohibited.
3. **IBM publisher provenance (U-4)** — requires an archived publisher record; the in-repo origin claim is itself unarchived.
4. **IEEE-CIS / BAF (U-6)** — requires acquisition (not authorised) plus terms review.
5. **Independent exposure audit (U-7)** — by definition external.
6. **Historical cause of D1 (U-3)** — no history exists for the affected untracked records; unknowable.
7. **U-5 / U-8 disposition** — reviewer judgment about the confirmatory role of `fraudTest` and about historical in-window metrics.
8. **Freeze (78 findings, §35 × 36)** — gated on the above; the checker must keep failing until then.
9. **Track M / Track N** — gated on freeze and on at least one eligible confirmatory dataset (none established).
10. **50M corpus** — no eligible source path exists today (§5).

---

## 5. 50M dataset status — definitive, evidence-based

1. **Largest currently acquired real dataset:** IBM v2 — `data/credit_card_transactions-ibm_v2.csv`.
2. **Rows:** **24,386,900 data rows** (re-measured `wc -l` = 24,386,901 incl. header); 15 columns; 2,350,744,057 bytes; SHA `b01fa323…`; 29,757 positives (0.122%).
3. **Native 48-feature contract:** **yes — 48/48 from IBM v2 alone** (4 `OBSERVED` + 44 `DERIVABLE`, 0 gaps; coverage §8.2). **No other acquired dataset can carry the native contract**: per-dataset partial coverage (e.g. Kaggle 1/18/10/19) never reaches 48, and native-48 coverage drops to 0/48 for every other dataset (coverage §8.3). The 9 unsourced causal/link-analysis features (device, recipient, auth) have no public source and are **not** part of the native 48.
4. **Sources investigated as expansion candidates:** ULB `creditcard.csv` (284,807 rows, V1–V28 PCA unit-of-record); Kaggle `fraudTrain` 1,296,675 + `fraudTest` 555,719 rows (23 cols, `cc_num`-level, not native); `User0_credit_card_transactions.csv` (19,963 rows — **duplicate extract of IBM v2**, positional identity verified); `ealtman2019/credit_card_transactions-ibm_v2.csv` (**byte-identical copy**, 0 new rows); `ealtman2019/sd254_cards.csv` / `sd254_users.csv` (auxiliary tables); PaySim `paysim.csv` 100,000 + `paysim_1m.csv` 1,200,000 (simulator output, no native fields); PS-14 synthetic (simulation-only); IEEE-CIS (590,540 train + 506,284 test — **`BLOCKED — not acquired`**); BAF (**`BLOCKED — not acquired`**); Zenodo 2026 / FreeFraudDetection50M / Dal Pozzolo / Worldline / Novatti / Elliptic (**not eligible**).
5. **Nature of each:** real/acquired = IBM v2, ULB, Kaggle ×2; real/not acquired = IEEE-CIS, BAF; synthetic = PaySim ×2, PS-14 synthetic; duplicates adding **0** independent rows = User0, `ealtman2019` copy; wrong-unit/incompatible/not eligible = the §4 list remainder; institution-controlled (Track I) = not located.
6. **Can any identified source defensibly increase the native-complete corpus to ≈50M?** **No.** Total acquired real rows across all four acquired datasets = 24,386,900 + 284,807 + 1,296,675 + 555,719 = **26,524,101** (≈53% of 50M) — and only IBM v2 satisfies the native contract. No identified external source supplies the native 48 schema. Duplicates are excluded by rule (§14: “distinct real volume”).
7. **Evidence required before such a dataset could be constructed:** established provenance, established feature semantics (native 48), established independence, and an established scientific purpose — none of which exists for any expansion path today. Construction is also downstream of reviewer decisions and freeze (plan §23).
8. **Would construction require synthetic feature generation?** For every non-IBM source, yes — either synthetic/bridged features, or padding by duplication. Both convert the corpus into **non-native or simulated volume**; duplication additionally violates the no-inflation rule.
9. **Would that change the scientific question?** **Yes.** A synthetic/padded 50M corpus is a different estimand (feature and label semantics are not native), and labelling it “native-complete” would be a misrepresentation. This phase produced no dataset and changed no plan text.

> **No 50M native corpus should be created unless its provenance, feature semantics,
> independence, and scientific purpose are established first.**

Status remains: **`NOT AUTHORIZED / NOT YET SCIENTIFICALLY JUSTIFIED`** — unchanged by this phase.

---

## 6. Freeze placeholder reconciliation

**Inventory (re-measured):** 78 findings = 1 missing `FREEZE_RECORD.json` + **65 plan
markers on 60 lines** + **12 prereg markers**. Marker kinds in the plan: 46 ×
`[TO BE FROZEN]`, 5 × `[DATE]`, 3 × `[OWNER]`, 3 × `[TO BE FROZEN — path]`, 2 ×
`[STATISTICAL REVIEWER]`/`[DOMAIN REVIEWER]`, 6 specials with inline hints.
**No placeholder was deleted or filled.** Class codes: RV-S statistical reviewer
decision · RV-D domain reviewer decision · MC methodological choice (reviewer) · SP
statistical parameter · MD metric definition · DE dataset-eligibility decision · AS
assignment / external dependency · OS obsolete/stale marker (prepared).

### 6.1 Plan markers (65 on 60 lines)

| Line | Section | Item | Class |
|---|---|---|---|
| 127 | §4A Primary metric | Chosen primary metric | MD / RV-S |
| 140 | §4 Ensemble identity | Selected ensemble | RV-S (`/02`) |
| 144 | §4 Baseline identity | Primary baseline | RV-S (`/01`) |
| 172 | §4 Track-level aggregation | Aggregation rule | MC / RV-S |
| 176 | §4 Track-level aggregation | Non-replication wording rule | MC / RV-S |
| 186 | §4 Multiplicity | Multiplicity method | SP / RV-S |
| 221 | §4A.1 Tiers | Primary metric (one from table) | MD / RV-S |
| 231 | §4A.2 Measures | FPR levels | SP |
| 232 | §4A.2 Measures | Recall levels | SP |
| 233 | §4A.2 Measures | Alert-volume levels & unit | SP (alert-volume semantics `/12`, RV-D side) |
| 234 | §4A.2 Measures | F-beta choice | SP |
| 235 | §4A.2 Measures | Cost matrix | SP (cost semantics `/12`, RV-D side) |
| 237 | §4A.2 Measures | ECE binning scheme & count | MD (`/07`) |
| 238 | §4A.2 Measures | Calibration slope/intercept method | MD (`/07`) |
| 312 | §5.0 Unit under test | Frozen gate-component definitions | MC (`/04`) |
| 345 | §5.2 Class A | Exact injection definitions | MC (`/04`) |
| 361 | §5.2 Class B | Perturbation families | MC (`/04`) |
| 365 | §5.2 Class B | Magnitudes / severity levels | MC / SP (`/04`) |
| 383 | §5.3 Confident decisions | Confident-decision definition | MC (`/03`, `BOTH`) |
| 409 | §5.5 Primary gating criterion | Criterion (1) | MC / SP (`/04`) |
| 415 | §5.5 Primary gating criterion | Criterion (2) | MC / SP (`/04`) |
| 419 | §5.5 Primary gating criterion | Criterion (3) | MC / SP (`/04`) |
| 423 | §5.5 Primary gating criterion | Criterion (4) | MC / SP (`/04`) |
| 427 | §5.5 Primary gating criterion | Criterion (5) | MC / SP (`/04`) |
| 447 | §6 RQ-M4 Calibration | Calibration method | MC (`/07`) |
| 449 | §6 RQ-M4 Calibration | Primary calibration metric | MD (`/07`) |
| 451 | §6 RQ-M4 Calibration | Acceptance bound | SP (`/07`) |
| 506 | §10 RQ-M8 Business | Primary business metric | MD (`/12`, `BOTH`) |
| 508 | §10 RQ-M8 Business | Proxy definition | MC (`/12`) |
| 525 | §11 RQ-M9 Segments | Segment definitions | MC (`/14`) |
| 623 | §16 Track P | Public-contract size `N` | MC |
| 677 | §18 Statistical power | Minimum fraud count | SP (`/08`) |
| 679 | §18 Statistical power | Minimum total evaluation count | SP (`/08`) |
| 681 | §18 Statistical power | Minimum valid seeds/runs | SP (`/10`) |
| 701 | §19 Dependence-aware CI | Entity-clustered bootstrap | SP (`/09`) |
| 709 | §19 Dependence-aware CI | Temporal/block bootstrap | SP (`/09`) |
| 713 | §19 Dependence-aware CI | Block design detail | SP (`/09`) |
| 719 | §19 Dependence-aware CI | Independent-observation bootstrap | SP (`/09`) |
| 755 | §21 Primary success criteria — Signal | Primary metric | MD (`/02`) |
| 759 | §21 Primary success criteria — Signal | Minimum effect | SP (`/02`) |
| 771 | §21 Primary success criteria — Transfer | Performance floor | SP |
| 781 | §21 Primary success criteria — Calibration | Acceptance bound | SP |
| 787 | §21 Primary success criteria — Reproducibility | Match tolerance | SP / MC |
| 831 | §23 Track-level Go/No-Go | Two-dataset aggregation rule | MC / RV-S |
| 835 | §23 Track-level Go/No-Go | One-dataset aggregation rule | MC / RV-S |
| 851 | §23 Track M roll-up | Primary RQs | RV-S |
| 855 | §23 Track M roll-up | Dataset-level rule | RV-S |
| 871 | §23 Outcome matrix | GO / NO-GO cell | RV-S |
| 872 | §23 Outcome matrix | GO / INCONCLUSIVE cell | RV-S |
| 875 | §23 Outcome matrix | NO-GO / INCONCLUSIVE cell | RV-S |
| 913 | §24.1 REVISE triggers | Further trigger types | RV-S |
| 922 | §24.2 INCONCLUSIVE consequences | Pre-frozen extra-evaluation design | RV-S |
| 1008 | §28 Manual dependencies | IEEE-CIS owner + date | AS / RAEE |
| 1009 | §28 Manual dependencies | BAF owner + date | AS / RAEE |
| 1010 | §28 Manual dependencies | Exposure-audit owner + date | AS / RQR (U-7 auditor) |
| 1011 | §28 Manual dependencies | Statistical-review identity + date | AS / RQR (D-1) |
| 1012 | §28 Manual dependencies | Domain-review identity + date | AS / RQR (D-2) |
| 1224 | §34 Referenced Artifacts | Preregistration path | **OS** — path prepared (R-3) |
| 1225 | §34 Referenced Artifacts | Exposure-ledger path | **OS** — path prepared (R-3) |
| 1226 | §34 Referenced Artifacts | Eligibility-manifest path | **OS** — path prepared (R-3) |

None of the 65 markers is removable by this phase; **3 lines (1224–1226)** are
objectively resolvable and their fill content is prepared in R-3; all others require a
reviewer decision, a statistical/domain parameter choice, an assignment, or the freeze
procedure itself.

### 6.2 Preregistration markers (12)

| Line | Section | Item | Class |
|---|---|---|---|
| 4 | Header | Scope resolution by reviewers before any experiment | RV-S |
| 20 | §0 Scope & model identity | Confirm system under evaluation | RV-S (`/01`,`/02`) |
| 48 | §1 Data & splits | Per-dataset split choice | MC / RV-S |
| 61 | §2 Metrics | ECE bin count (10–15 equal-mass) | MD / SP (`/07`) |
| 72 | §2 Metrics | Replicates and seed | SP (`/10`) |
| 73 | §2 Metrics | Multiple-comparisons treatment | SP |
| 83 | §3 Decision criteria | Confirm/replace proposed criteria | RV-S |
| 175 | §4 Negative-result plan | A/B/C wording; publication terms | RV-S |
| 182 | §5 Threshold discipline | Pre-declared training seeds | SP |
| 185 | §5 Threshold discipline | Operating point (fixed FPR vs volume) | SP |
| 190 | §5 Threshold discipline | Re-tune policy | MC |
| 211 | §7 Business metrics | Assumed analyst capacity | RV-D (`/12`, `BOTH`) |

### 6.3 Plan §35 pre-freeze checklist — 36 unchecked

| Group | Items | Resolved by |
|---|---|---|
| Freeze-the-parameters items | RQ freeze, ensemble/baseline identity, tuning budget (4) | Reviewer decisions in §6.1/§6.2 |
| External-dependency items | exposure-history audit (U-7), IEEE-CIS terms (U-6), BAF terms/provenance (U-6), confirmatory datasets identified, licence/access verified (U-1), feature semantics verified (audit D-3) (6) | Auditors + licence review + acquisitions |
| Statistical/domain parameter items | split rules, temporal/entity rules, PIT/label-delay, Class A/B injections, gating magnitudes, confident-decision definition, fallback accounting, coverage/recall bounds, business metrics, segment definitions, minimum fraud count, CI method, multiplicity, above-chance rule, one- and two-dataset aggregation, GO/NO-GO rules, one-revision rule (19) | Reviewers (§6.1/§6.2) |
| Sign-offs | statistical reviewer signed off (D-1), domain review completed (D-2) (2) | Reviewers |
| Mechanical / procedural | automated placeholder check passes, plan hash recorded, prereg hash recorded, freeze tag created, freeze recorded in ledger (5) | Freeze procedure (after the above) |

### 6.4 The missing `FREEZE_RECORD.json`

Created only when freeze is legitimate. `scripts/check_freeze.py` **must keep
exiting non-zero** until then; it was not changed, weakened, or bypassed (rc=1, 78
findings — §1).

---

## 7. Reviewer handoff matrix

| Item | Can repository resolve? | Required external actor | Evidence required | Current status |
|---|---|---|---|---|
| Statistical decisions (`/01 /02 /08 /09 /10 /11 /13 /14` + stat halves of `/03–/07 /12`) | No | Qualified statistical reviewer (D-1) | Independent statistical ruling per decision | `NOT ASSIGNED` |
| Domain decisions (domain halves of `/03 /04 /05 /06 /07 /12`) | No | Qualified domain reviewer (D-2) | Domain ruling on fraud-semantics, cost, alert-volume, segments | `NOT ASSIGNED` |
| Exposure audit (U-7) | No | Independent dataset/provenance auditor (D-3) | Exposure/provenance verification of manifest + ledger | `NOT ASSIGNED` |
| Licence determination (U-1) | Usually no | Licence-qualified reviewer / authoritative source | Exact terms URL, verbatim clause, licensor, access route, retrieval date | `OPEN` (`U` for all four datasets) |
| IBM provenance (U-4) | Maybe (internal pointers exist; archived record does not) | Source evidence / reviewer | Archived publisher record or equivalent authoritative source | `OPEN` (`NOT ESTABLISHED`) |
| Historical U-3 cause | No | — | No sufficient evidence exists | `HISTORICAL CAUSE NOT ESTABLISHED` |
| Research Plan freeze | No | Reviewers + freeze process | Resolved decisions; §35 complete; hashes recorded; tag created | `BLOCKED` |
| Track M | No | — | Frozen plan + ≥1 confirmatory-eligible dataset | `BLOCKED` |
| 50M experiment | No | — | Scientific justification + eligible corpus + freeze | `NOT AUTHORIZED` |

Terminology follows the existing records (assignment record, resolution §4, exposure
ledger §7, eligibility manifest §9, review package §12/§13).

---

## 8. Exact next action

Obtain real external input: hand `docs/evaluation/REVIEWER_ENGAGEMENT_BRIEF.md` to
prospective reviewers/auditors and record **at least one** of — (a) an assigned
statistical reviewer (D-1), (b) an assigned domain reviewer (D-2), (c) an assigned
independent dataset/provenance auditor (D-3), (d) authoritative licence evidence
(U-1) — in the existing machinery. No further internal preparation phase is warranted;
the repository now waits at **`AWAITING HUMAN REVIEW`**.

---

## Change history

| Version | Change | Type |
|---|---|---|
| 1.0-draft | Initial objective evidence closure: U-1…U-10, O-1…O-8 and the freeze findings classified; R-1…R-7 objectively resolvable findings measured/prepared (IBM row count 24,386,900; paysim 8 columns; §34 paths; counts 78 = 1+65+12; §35 × 36); 50M status determined; 65 plan + 12 prereg markers classified. No review state, eligibility, licence, exposure, freeze or Track M state changed. | OBJECTIVE EVIDENCE CLOSURE |
