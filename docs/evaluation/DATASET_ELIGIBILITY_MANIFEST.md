# Dataset Eligibility Manifest

**Required by:** `docs/RESEARCH_PLAN.md` §34 (Referenced Artifacts) and §13 (Dataset
Eligibility for Track M).
**Artifact path:** `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md`
**Version:** `eligibility-manifest 1.0-draft`
**Status:** DRAFT — pre-freeze preparation artifact. **NOT frozen, NOT approved; no
dataset is declared eligible by this document.**
**Authored at git state:** `5a5ff55cc03df73318fdf31a16e3665f159a4720` (2026-10-04).
**§34 path cell:** the plan's §34 row reads `[TO BE FROZEN — path]`; this file is the
proposed location, and that path decision remains an unresolved freeze-time decision.
**Self-hash:** recorded at freeze in `docs/FREEZE_RECORD.json` (plan §29), not here.
**Companion artifact:** `docs/evaluation/DATASET_EXPOSURE_LEDGER.md` (exposure history
and prior-result classifications; this manifest does not restate them).

## Purpose

For each candidate dataset, record the §13 identity fields, the eligibility decision
state, and the exact gate that blocks it — without converting missing information into
eligibility. `ESTABLISHED` means repository evidence exists and is cited; it never means
"eligible". Eligibility additionally requires the unmet gates in §6 below.

Eligibility decision vocabulary follows the plan: `CANDIDATE CONFIRMATORY` (§14),
`BLOCKED`, `NOT ESTABLISHED`, `PENDING REVIEW`. No `FAILED`/`INCONCLUSIVE` rows exist
yet — no confirmatory experiment has run (a `FAILED`/`INCONCLUSIVE` row would require an
actual experiment).

---

## 1. ULB (`data/creditcard.csv`)

| §13 field | Value | State |
|---|---|---|
| provenance | Kaggle "Credit Card Fraud Detection" dataset; file present in repo | **ESTABLISHED** (file identity); provenance *documentation* page not archived in repo → provenance record `NOT ESTABLISHED` |
| licence/access status | **CORRECTED 2026-10-04 (§4.1 C-7).** Conflicting repo references, still unresolved: plan-era §35 checklist has unchecked "licence/access verified"; `docs/metric_definitions.md` credits Kaggle CC BY-SA 4.0; `backend/scripts/dataset_card.py` writes an "AGPL-3.0" licence string. The prior cell also cited `docs/PHASE_NR02_PROTOCOL_REVIEW.md` "Open licence" (§G) — **NR-02 §G actually records "License terms: NOT ESTABLISHED (no license file in repo)"**, so that citation did not support the claim and is retracted. | **PENDING REVIEW** — conflicting, not merged, not chosen between |
| acquisition status | **CORRECTED 2026-10-04 (§4.1 C-9).** file present, **150,828,752 bytes** (measured; prior cell stated 171,712,196, which does not match the file — NR-02 §G's "150,828,752 B" was correct) | **ESTABLISHED** |
| dataset hash/version | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` (recomputed 2026-10-04) | **ESTABLISHED** |
| feature semantics | 30 PCA components + `Time`/`Amount`/`Class` | **ESTABLISHED** |
| label semantics | `Class` = fraud indicator (quote-wrapped header `"Class"`) | **ESTABLISHED** |
| temporal information | `Time` = seconds from first transaction; no calendar dates | **ESTABLISHED** |
| entity identifiers | **none** (no card/account/device id column) | **ESTABLISHED** |
| leakage assessment | no repo-archived pre-ULB leakage audit | **NOT ESTABLISHED** |
| preprocessing requirements | `[TO BE FROZEN]` (plan §3/§14 new-model rules) | **NOT ESTABLISHED** |
| real/synthetic classification | real-world (anonymized card transactions) | **ESTABLISHED** |
| prior project exposure | used during development/evaluation — **EXPLORATORY** (plan §14) | **ESTABLISHED** |
| applicable research track | Track M (only under explicitly frozen new-model rules) | per plan §14 |
| eligibility decision | **EXPLORATORY USE ONLY — not confirmatory-eligible as an untouched dataset**; new-model-use rules `[TO BE FROZEN]` | per plan §14 |

| row count | 284,807 (measured 2026-10-04: full CSV parse; `Class` positives = 492) | **ESTABLISHED** |

## 2. Kaggle fraudTrain (`data/kaggle_fraud/fraudTrain.csv`)

| §13 field | Value | State |
|---|---|---|
| provenance | Kaggle "Fraud Detection"; file present | **ESTABLISHED** (file); archived provenance record `NOT ESTABLISHED` |
| licence/access status | unresolved (see ULB conflict; no Kaggle licence citation located for this file) | **PENDING REVIEW** |
| acquisition status | **CORRECTED 2026-10-04 (§4.1 C-9).** present, **351,238,196 bytes** (measured; prior cell stated 117,543,644, which does not match the file) | **ESTABLISHED** |
| dataset hash/version | `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d` | **ESTABLISHED** |
| feature semantics | `cc_num` merchant/category/amt/job/geo etc. | **ESTABLISHED** (column semantics partial → detailed field semantics `NOT ESTABLISHED` archived) |
| label semantics | `is_fraud` | **ESTABLISHED** |
| temporal information | `trans_date_trans_time` datetimes | **ESTABLISHED** |
| entity identifiers | `cc_num` present → entity-disjoint evaluation *possible* | **ESTABLISHED** |
| leakage assessment | no repo-archived audit | **NOT ESTABLISHED** |
| preprocessing requirements | `[TO BE FROZEN]` | **NOT ESTABLISHED** |
| real/synthetic classification | real-world synthetic-fraud-labelled generation per Kaggle; repo classification record `NOT ESTABLISHED` explicitly | — record explicitly absent |
| prior project exposure | used during development/evaluation — **EXPLORATORY** (plan §14); historical transfer results influenced promotion decisions | **ESTABLISHED** |
| applicable research track | Track M (only under frozen new-model rules) | per plan §14 |
| eligibility decision | **EXPLORATORY USE ONLY — not confirmatory-eligible as an untouched dataset** | per plan §14 |

## 3. Kaggle fraudTest (`data/kaggle_fraud/fraudTest.csv`)

Same §13 fields as fraudTrain, with differences:

| §13 field | Value | State |
|---|---|---|
| dataset hash/version | `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0` | **ESTABLISHED** |
| acquisition status | **CORRECTED 2026-10-04 (§4.1 C-9).** present, **150,354,339 bytes** (measured; prior cell stated 49,313,965, which does not match the file) | **ESTABLISHED** |
| prior project exposure | used during development/evaluation — **EXPLORATORY** (plan §14) | **ESTABLISHED** |
| prior-exposure → confirmatory tension | preregistration assigns `kaggle_test` to confirmatory E2/E3/E5/E6, while §14 lists Kaggle as exploratory (fraudTrain); the ambiguity *fraudTest confirmatory or not* is an open reviewer question | **PENDING REVIEW** (recorded in NR-03; no resolution invented here) |
| eligibility decision | **EXPLORATORY USE ONLY pending reviewer resolution**; not confirmatory-eligible | per plan §14 default |

## 4. IBM v2 (`data/credit_card_transactions-ibm_v2.csv`)

| §13 field | Value | State |
|---|---|---|
| provenance | File present and hashed. **CORRECTED 2026-10-04 (§4.1 C-4):** the file's own schema is the 15-column card-transaction layout distributed as `User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name, Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?`; it is **not** the 168-column `credit_card_transactions.csv` layout. Archived publisher/provenance record still not in repo | **ESTABLISHED** (file identity + measured schema); publisher provenance `NOT ESTABLISHED` |
| licence/access status | **CORRECTED 2026-10-04 (§4.1 C-5).** The prior cell said `"Open licence" claimed at protocol-era`, citing NR-02 §G. NR-02 §G actually records **"License: NOT ESTABLISHED"** for IBM v2 — the cited source contradicts the claim. Plan §35's licence/access checkbox remains unchecked. | **PENDING REVIEW** (claim retracted to `NOT ESTABLISHED`; reviewer must verify terms) |
| acquisition status | **CORRECTED 2026-10-04 (§4.1 C-9).** present, **2,350,744,057 bytes** (measured; prior cell stated 2,466,191,524, which does not match the file — NR-02 §G's "2.35 GB" was correct). A byte-identical copy lives at `data/ealtman2019/credit_card_transactions-ibm_v2.csv` (same size, same sha256) — see §4.2. | **ESTABLISHED** (measured) |
| dataset hash/version | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` (recomputed 2026-10-04; matches `prereg_harness.py::DATASETS["ibm_v2"]`) | **ESTABLISHED** |
| feature semantics | **CORRECTED 2026-10-04 (§4.1 C-2/C-3).** **15 raw columns** (measured): `User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name, Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?`. There is **no `hour_diff` column** and no 168-column IBM file anywhere in the repository. | **ESTABLISHED** (measured, 2026-10-04) |
| label semantics | **CORRECTED 2026-10-04 (§4.1 C-6).** Label column is **`Is Fraud?`** with values `Yes`/`No` (29,757 `Yes`). The prior cell said `is_fraud`, which is a column of the *absent* 168-column file. | **ESTABLISHED** (measured) |
| temporal information | **CORRECTED 2026-10-04 (§4.1 C-6).** Calendar fields are **`Year`, `Month`, `Day`** plus clock field **`Time`** (`HH:MM`). Observed span 1995–2020 (from `models/model_records/…` split boundaries and the `Year` column). The prior cell cited `trans_date`/`trans_time`/`unix_time`, none of which exist in this file. | **ESTABLISHED** (measured) |
| entity identifiers | **CORRECTED 2026-10-04 (§4.1 C-6).** Identifiers present: **`User`**, **`Card`**, **`Merchant Name`**, **`Merchant City`**. The prior cell cited `cc_num`, which does not exist in this file. | **ESTABLISHED** (measured) |
| leakage assessment | **CORRECTED 2026-10-04 (§4.1 C-3).** The prior cell recorded a `hour_diff` point-in-time defect "as evaluated" (NR-01 §A). The acquired file contains **no `hour_diff` column**, and NR-01 §A ("Objective") contains **no such claim** — the citation did not support the assertion. `hour_diff` as a *dataset column* is therefore **not applicable to this file**. This file ships no pre-extracted delta/history column at all; every history feature used by the native 48 is derived by this project with a prior-only shift (`expanding().mean().shift(1)`), so there is no imported-delta leakage channel. **No repo-archived leakage audit exists for this dataset**, so the assessment itself stays `NOT ESTABLISHED` — the correction removes a defect that does not apply; it does not certify the dataset. | **`NOT ESTABLISHED`** (C-3) |
| preprocessing requirements | `[TO BE FROZEN]` | **NOT ESTABLISHED** |
| real/synthetic classification | synthetic transaction generator (IBM); classification record in repo `NOT ESTABLISHED` explicitly — synthetic results cannot establish real-world performance (plan §13) | record absent |
| prior project exposure | used during development/evaluation — **EXPLORATORY** (plan §14) | **ESTABLISHED** |
| applicable research track | Track M (only under frozen new-model rules) | per plan §14 |
| eligibility decision | **EXPLORATORY USE ONLY — not confirmatory-eligible as an untouched dataset** | per plan §14 |
| row count | **ESTABLISHED = 24,386,900** data rows (§4.1 C-1). Measured 2026-10-04: full CSV parse counting rows carrying all 15 fields = **24,386,900**; rows with any other field count = **0**; `wc -l` = 24,386,901 including the header. Positives `Is Fraud? == Yes` = **29,757** (0.122%). | **ESTABLISHED** |
| duplicate extracts | **DUPLICATE — NOT independent evidence.** `data/User0_credit_card_transactions.csv` (sha256 `61c4ed49…`) is the first **19,963** data rows of this file, position-for-position identical (19,963/19,963 verified); `data/ealtman2019/` holds byte-identical copies of both. Contributes **0** independent rows. See §4.2. | **ESTABLISHED** |

### 4.1 Change history (append-only; prior text preserved verbatim)

Corrections applied **2026-10-04** under
`docs/evaluation/DATASET_PROVENANCE_RECONCILIATION.md`. This table covers
**manifest-wide** corrections (C-1…C-6 and C-9 concern §4 IBM v2; C-7 and C-8
concern §1 ULB). Each is a *factual* correction from measured
repository evidence. **None upgrades eligibility, exposure, or any
`FAILED`/`NON-CONFORMING` classification.** Pre-edit SHA-256 of this file:
`92b766bea42642b84e76e911e2ac0427bfb151123cc33ae7474382eee117efc5`.

| ID | Field | OLD claim (verbatim, preserved here) | NEW claim | Evidence | Scientific effect |
|---|---|---|---|---|---|
| **C-1** | row count | `NOT ESTABLISHED (not recorded in repo registries)` | `ESTABLISHED = 24,386,900` (+29,757 positives) | full CSV parse; `wc -l` 24,386,901 incl. header | None — records a previously-unknown factual count |
| **C-2** | feature semantics | `168 raw features; hour_diff = derived (deltas), preprocessing fidelity concerns flagged in NR-01` \| `ESTABLISHED (concern)` | `15 raw columns` (enumerated) \| `ESTABLISHED` (measured) | header read; `_forensic_common.py::USECOLS` (14 of the 15); `prereg_harness.py::DATASETS["ibm_v2"].label_column = "Is Fraud?"`; `misc/reports/phase16/dataset_inventory.json` records `n_cols: 15` for the sibling extract | **Material** — removes a defect record that never applied to this file |
| **C-3** | leakage assessment | `hour_diff derived during feature extraction (deltas), not point-in-time safe as evaluated (NR-01 §A)` \| `ESTABLISHED (defect)` | `NOT ESTABLISHED` — no `hour_diff` column exists; NR-01 §A makes no such claim | no `hour_diff` anywhere except a local variable in `unified_detection_test.py`; NR-01 §A is "Objective" and contains no such statement | **Material** — retracts an unsupported defect finding |
| **C-4** | provenance | `IBM Base Fraud Detection dataset` | file identity + 15-column schema `ESTABLISHED`; publisher provenance still `NOT ESTABLISHED` | measured schema | Clarifies which dataset this record describes |
| **C-5** | licence/access status | `"Open licence" claimed at protocol-era` \| `PENDING REVIEW` | claim **retracted**: NR-02 §G records `License: NOT ESTABLISHED` \| `PENDING REVIEW` | `docs/PHASE_NR02_PROTOCOL_REVIEW.md` §G | **Material** — removes an unsupported licence assertion |
| **C-6** | label semantics / temporal / entity ids | `is_fraud` / `trans_date`+`trans_time`+`unix_time` / `cc_num` \| `ESTABLISHED` | `Is Fraud?` (Yes/No) / `Year`+`Month`+`Day`+`Time` / `User`,`Card`,`Merchant Name`,`Merchant City` \| `ESTABLISHED` | measured header — none of the three old field names exists in the file | **Material** — the prior row described a *different, absent* dataset |
| **C-7** | ULB licence/access status | `… docs/PHASE_NR02_PROTOCOL_REVIEW.md "Open licence" (§G)` | that citation **retracted** (NR-02 §G records `License terms: NOT ESTABLISHED`); the metric_definitions / dataset_card conflicts are retained verbatim | `docs/PHASE_NR02_PROTOCOL_REVIEW.md` §G read directly | **Material** — removes an unsupported licence assertion; `PENDING REVIEW` unchanged |
| **C-8** | ULB row count | free-text: `Row count 284,807: datasets.json says 284,808 (header-inclusive)… conflict recorded, not resolved` | measured `ESTABLISHED = 284,807` data rows (492 `Class` positives); the 284,808 figure is **header-inclusive** counting | full CSV parse of `data/creditcard.csv` | None — reconciles a counting convention, not a value |
| **C-9** | acquisition status (byte size) — **all four acquired datasets** | ULB `171,712,196`; fraudTrain `117,543,644`; fraudTest `49,313,965`; IBM v2 `2,466,191,524` \| all `ESTABLISHED` | measured: ULB **150,828,752**; fraudTrain **351,238,196**; fraudTest **150,354,339**; IBM v2 **2,350,744,057** — **none of the four prior figures matched its file** | `os.path.getsize`; NR-02 §G corroborates ULB (150,828,752 B) and IBM (2.35 GB); IBM corroborated by its byte-identical copy's matching size | None — corrects a systematically mis-stated field; hash pins unaffected |

**Unchanged by this correction:** `licence/access status` remains `PENDING REVIEW`;
`eligibility decision` remains `EXPLORATORY USE ONLY`; `prior project exposure`
remains `EXPLORATORY`; no dataset is declared eligible.

### 4.2 Duplicate extracts of this dataset (recorded, not deleted)

| File | SHA-256 | Relationship | Independent rows |
|---|---|---|---|
| `data/credit_card_transactions-ibm_v2.csv` | `b01fa323…` | the canonical acquired file | 24,386,900 |
| `data/ealtman2019/credit_card_transactions-ibm_v2.csv` | `b01fa323…` | **byte-identical copy** of the above | **0** |
| `data/User0_credit_card_transactions.csv` | `61c4ed49…` | first **19,963** data rows of the above, position-identical | **0** |
| `data/ealtman2019/User0_credit_card_transactions.csv` | `61c4ed49…` | **byte-identical copy** of the User0 extract | **0** |

None of these appears in `prereg_harness.py::DATASETS` or
`misc/benchmarks/datasets.json`, so no registry currently counts them as datasets.
`misc/reports/phase16/dataset_inventory.json` does list **two** separate entries with
identical hashes (`61c4ed49…`) — that historical report must not be read as two
independent datasets. See
`docs/evaluation/DATASET_PROVENANCE_RECONCILIATION.md` §7.

## 5. IEEE-CIS (expected path `data/external/IEEE_CIS/`)

| §13 field | Value | State |
|---|---|---|
| provenance | IEEE-CIS / Vesta anti-fraud transaction dataset (Kaggle IEEE-CIS); source page not archived in repo | **NOT ESTABLISHED** |
| licence/access status | terms checked? plan §35 checklist item "IEEE-CIS terms checked" is **unchecked** | **BLOCKED** (manual verification required, not performed) |
| acquisition status | directory `data/external/` does not exist (verified 2026-10-04) | **BLOCKED — not acquired** |
| dataset hash/version | none — cannot hash an unacquired dataset | **NOT ESTABLISHED** |
| feature semantics | known broadly (transaction + identity tables); repo-archived record `NOT ESTABLISHED` | **NOT ESTABLISHED** |
| label semantics | `is_fraud` (per Kaggle) — not verified in repo | **NOT ESTABLISHED** |
| temporal information | timestamps known to exist; not verified in repo | **NOT ESTABLISHED** |
| entity identifiers | identity/card links known to exist; not verified in repo | **NOT ESTABLISHED** |
| leakage assessment | cannot assess before acquisition | **NOT ESTABLISHED** |
| preprocessing requirements | `[TO BE FROZEN]` | **NOT ESTABLISHED** |
| real/synthetic classification | real-world (Vesta company transactions) — per public description, not repo-verified | **NOT ESTABLISHED** in repo |
| prior project exposure | plan §14: "No established prior model exposure" | per plan §14 (not independently audited → see ledger §7) |
| label sufficiency / fraud-count sufficiency / temporal suitability / test-label availability | cannot be assessed pre-acquisition | **NOT ESTABLISHED** (cannot evaluate power rules §18 either) |
| applicable research track | Track M/Track P candidate | per plan |
| eligibility decision | **CANDIDATE CONFIRMATORY pending eligibility (§14)** — concretely **BLOCKED on acquisition + terms/provenance verification**; *not eligible* | — |

## 6. BAF (Bank Account Fraud, expected path `data/external_benchmark/`)

| §13 field | Value | State |
|---|---|---|
| provenance | Bank Account Fraud (NeurIPS 2022); source page not archived in repo | **NOT ESTABLISHED** |
| licence/access status | "BAF terms/provenance checked" plan §35 checklist **unchecked** | **BLOCKED** (manual verification required) |
| acquisition status | `data/external_benchmark/` absent (verified 2026-10-04); preregistered E1 status `BLOCKED: DATASET UNAVAILABLE`; harness `validate-dataset baf` → rc=4, hash mismatch, reason `DATASET UNAVAILABLE` | **BLOCKED — not acquired** (double-sourced) |
| dataset hash/version | none | **NOT ESTABLISHED** |
| feature semantics | `fraud_bool`, income, expense, etc. (public description); not repo-verified | **NOT ESTABLISHED** |
| label semantics | `fraud_bool` — not repo-verified | **NOT ESTABLISHED** |
| temporal information | `month` field (public description); not repo-verified | **NOT ESTABLISHED** |
| entity identifiers | none known publicly; not repo-verified | **NOT ESTABLISHED** |
| leakage assessment | cannot assess before acquisition | **NOT ESTABLISHED** |
| preprocessing requirements | `[TO BE FROZEN]` | **NOT ESTABLISHED** |
| real/synthetic classification | synthetic/semi-synthetic (public description); not repo-verified — synthetic results cannot establish real-world performance (plan §13) | **NOT ESTABLISHED** in repo |
| prior project exposure | plan §14: "No established prior model exposure" | per plan §14 (not independently audited) |
| label/fraud-count/temporal sufficiency | preconditions unknown; preregistration E1 precondition = "dataset is acquired and validated" | **NOT ESTABLISHED** |
| applicable research track | Track M candidate (confirmatory E1); Track P conditional | per plan |
| eligibility decision | **CANDIDATE CONFIRMATORY pending eligibility (§14)** — concretely **BLOCKED on acquisition + terms/provenance verification**; *not eligible* | — |

## 7. PS-14 synthetic (`data/transactions.csv`)

| §13 field | Value | State |
|---|---|---|
| provenance | generated by in-repo generator; hash `9f0f56bf0549fccf0c6335ece1514d1ae515a57f809a72658ce5fcaf48525d63` | **ESTABLISHED** |
| real/synthetic classification | **SIMULATED/SYNTHETIC** (plan §13 requirement: must be explicitly classified so) | **ESTABLISHED** |
| prior project exposure | used throughout development | **ESTABLISHED** |
| eligibility decision | exploratory/simulation use only — **cannot establish real-world performance** (plan §13) | per plan §13 |

---

## 8. Dataset class register (what this manifest currently describes)

Added 2026-10-04 per the provenance-reconciliation phase. **This is a classification
index, not an eligibility grant** — eligibility decisions stay in §1–§7 and remain
governed by the plan.

| Class | Datasets | Basis |
|---|---|---|
| **ACQUIRED + hashed** | ULB, Kaggle fraudTrain, Kaggle fraudTest, IBM v2, PS-14 synthetic | file present, SHA-256 pinned in `prereg_harness.py::DATASETS` and/or recomputed |
| **ACQUIRED + hashed, DUPLICATE** | `data/User0_credit_card_transactions.csv`, `data/ealtman2019/*` | byte-identical to, or an extract of, IBM v2 (§4.2) — **0 independent rows** |
| **DESCRIBED PUBLICLY, NOT ACQUIRED** | IEEE-CIS, BAF | public schema documented; `data/external/` and `data/external_benchmark/` absent |
| **NOT OBTAINABLE** | Dal Pozzolo ~49.86M (confidential), Worldline, Novatti, FreeFraudDetection50M (could not be located) | no public release / no access path |
| **REJECTED for the native contract** | BAF (account-**opening** records, not transactions) | assessed and refused; see `DATASET_48_FEATURE_COVERAGE.md` §4.4 |
| **REQUIRES REVIEWER DECISION** | fraudTrain/fraudTest confirmatory-vs-exposed tension (`PENDING REVIEW`); all licence/access cells | plan §32 reviewers `NOT ASSIGNED`; 0/14 decisions resolved |

**Row-count basis note.** Counts in this manifest are **measured data rows**
(header excluded). `misc/benchmarks/datasets.json` uses header-inclusive counting for
ULB (284,808 vs 284,807); the measured convention is stated here so the two are
comparable rather than in conflict.

---

## 9. Cross-dataset gates (apply before any confirmatory use; none satisfied yet)

| Gate | State |
|---|---|
| Licence/access verified per dataset (plan §35) | **NOT ESTABLISHED** for all (conflicts recorded above) |
| Exposure ledger independently audited (§14) | **NOT ESTABLISHED** |
| Eligibility manifest itself approved at freeze | pending — `[TO BE FROZEN]` path in §34 |
| Statistical/power gates (§18) evaluable | **BLOCKED** for IEEE-CIS/BAF (no data); N/A for exposed sets pre-freeze |
| New-model rules for exposed datasets (§14) | `[TO BE FROZEN]` — **NOT ESTABLISHED** |

## 10. Change discipline

No row may be upgraded to eligible without (a) the dataset actually being acquired and
hashed, (b) terms/licence/provenance manually verified, (c) the plan's freeze procedure
completed for confirmatory use. Nothing here overrides the exposure ledger or the plan;
where they differ, the plan governs.

*End of `eligibility-manifest 1.0-draft`.*
