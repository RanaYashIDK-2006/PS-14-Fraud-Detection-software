# Dataset Licence Evidence Checklist (reviewer worksheet)

**Artifact path:** `docs/evaluation/DATASET_LICENCE_EVIDENCE_CHECKLIST.md`
**Version:** `licence-evidence-checklist 1.0-draft`
**Status:** DRAFT worksheet — **grants nothing, resolves nothing, asserts no licence.**
**Date:** 2026-10-04
**Resolves open item:** **U-1** in `docs/evaluation/DATASET_PROVENANCE_RECONCILIATION.md` §13
  *("licence/terms unverified for every acquired dataset — plan §35 checkbox unchecked")*.

---

## 0. What this document is, and is not

| It IS | It is NOT |
|---|---|
| A list of what a reviewer must **capture** to clear a dataset's licence | A statement of any dataset's licence |
| A record of *where the terms are published* | A legal opinion or interpretation |
| A worksheet that keeps the outcome **range** open | A clearance, an approval, or an eligibility grant |
| An input to plan §35's unchecked checkbox | A substitute for that checkbox being checked |

**Nothing here changes any eligibility, exposure, or `FAILED`/`NON-CONFORMING`
classification.** The eligibility manifest and exposure ledger were deliberately
**not** written back to from this worksheet; their `PENDING REVIEW` /
`NOT ESTABLISHED` states stand unchanged.

**This worksheet asserts no licence for any dataset.** Where a prior repository claim
existed, it was found to be unsupported and has already been retracted
(manifest §4.1 C-5, C-7). No replacement claim is offered here.

---

## 1. Why this is blocking

Plan §35 carries a pre-freeze checklist item **"licence/access verified"**. It is
**unchecked for every dataset**, including the four acquired ones. Separately, the
eligibility manifest §9 records the gate as `NOT ESTABLISHED` for all datasets.

Consequence: **even if every other blocker were removed today, no dataset is
licence-cleared for confirmatory use.**

---

## 2. The outcome range (all four remain open)

A reviewer must select **one** outcome per dataset. The range is deliberately kept
wide; nothing is pre-selected or narrowed here.

| Code | Outcome | Meaning | Effect on plan §35 |
|---|---|---|---|
| **F-P** | Free — permissive | Redistribution/derivation permitted with attribution; no use restriction | may permit checkbox |
| **F-R** | Free — restricted | Free to obtain, but use/redistribution restricted (e.g. non-commercial, research-only, share-alike, no-derivatives) | reviewer must state the restriction's effect on PS-14 |
| **R** | Restricted | Access conditional on agreement/approval/credentials; not freely redistributable | does **not** permit checkbox without a signed agreement |
| **U** | Unknown / not published | No terms located or terms ambiguous; **this is the current state of all four datasets** | cannot permit checkbox |

> **Current state: `U` for all four.** Any movement off `U` requires reviewer
> evidence, not this document.

---

## 3. Per-dataset worksheet (unfilled — reviewer completes)

Complete one block per dataset. **Leave a cell blank rather than guessing.**

### 3.1 ULB — `data/creditcard.csv`

| Field | Value |
|---|---|
| SHA-256 (fixed) | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| Byte size (measured) | 150,828,752 |
| Outcome selected | ☐ F-P ☐ F-R ☐ R ☐ U |
| Terms page URL (exact) | ______________________ |
| Terms text quoted verbatim | ______________________ |
| Publisher named in terms | ______________________ |
| Access route used | ☐ direct download ☐ Kaggle API ☐ other: __________ |
| Attribution required? | ☐ yes → required form: ____________ ☐ no |
| Use restriction | ☐ none ☐ non-commercial ☐ research-only ☐ share-alike ☐ other: ______ |
| Redistribution of derived artifacts permitted? | ☐ yes ☐ no ☐ conditional: __________ |
| Reviewer name / date / evidence location | ______________________ |

**Known conflicting in-repo claims — reviewer must reconcile, not merge silently:**
`docs/metric_definitions.md` states CC BY-SA 4.0; `backend/scripts/dataset_card.py`
emits an "AGPL-3.0" string. (A third citation — NR-02 §G "Open licence" — was
already found unsupported and retracted.)

### 3.2 Kaggle fraudTrain — `data/kaggle_fraud/fraudTrain.csv`

| Field | Value |
|---|---|
| SHA-256 (fixed) | `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d` |
| Byte size (measured) | 351,238,196 |
| Outcome selected | ☐ F-P ☐ F-R ☐ R ☐ U |
| Terms page URL (exact) | ______________________ |
| Terms text quoted verbatim | ______________________ |
| Does Kaggle's competition/data terms cover this file specifically? | ______________________ |
| Use restriction | ______________________ |
| Reviewer name / date / evidence location | ______________________ |

### 3.3 Kaggle fraudTest — `data/kaggle_fraud/fraudTest.csv`

| Field | Value |
|---|---|
| SHA-256 (fixed) | `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0` |
| Byte size (measured) | 150,354,339 |
| Outcome selected | ☐ F-P ☐ F-R ☐ R ☐ U |
| Terms page URL (exact) | ______________________ |
| Reviewer name / date / evidence location | ______________________ |

> fraudTest additionally carries open item **U-5** (confirmatory-vs-exposed tension,
> `PENDING REVIEW`). A licence outcome does **not** resolve U-5.

### 3.4 IBM v2 — `data/credit_card_transactions-ibm_v2.csv`

| Field | Value |
|---|---|
| SHA-256 (fixed) | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |
| Byte size (measured) | 2,350,744,057 |
| Rows / columns (measured) | 24,386,900 / 15 |
| Outcome selected | ☐ F-P ☐ F-R ☐ R ☐ U |
| Terms page URL (exact) | ______________________ |
| Terms text quoted verbatim | ______________________ |
| Publisher named in terms | ______________________ |
| Use restriction | ______________________ |
| Reviewer name / date / evidence location | ______________________ |

> NR-02 §G records, verbatim, **"License: NOT ESTABLISHED"**. Any later claim must
> cite the terms page, not NR-02.

### 3.5 Byte-identical copies — same licence, no separate review

| File | SHA-256 | Handling |
|---|---|---|
| `data/ealtman2019/credit_card_transactions-ibm_v2.csv` | `b01fa323…` | **same licence as §3.4** — a byte-identical copy cannot have different terms |
| `data/User0_credit_card_transactions.csv` | `61c4ed49…` | derived/extract of §3.4 — **inherits §3.4's outcome**; contributes 0 independent rows |
| `data/ealtman2019/User0_credit_card_transactions.csv` | `61c4ed49…` | identical copy — inherits §3.4's outcome |

---

## 4. Datasets with **no** licence question to answer yet

Licence cannot be assessed for a dataset that has not been acquired. Listed so the
omission is deliberate and visible.

| Dataset | State | Consequence |
|---|---|---|
| IEEE-CIS | `BLOCKED — not acquired` (`data/external/` absent) | no terms reviewed; also **U-6** |
| BAF | `BLOCKED — not acquired` (`data/external_benchmark/` absent) | no terms reviewed; also **U-6** |
| Dal Pozzolo ~49.86M | confidential, never publicly released | no public terms exist to verify |
| Worldline, Novatti | confidential, institutional access required | requires an institutional agreement, not a terms-page check |
| FreeFraudDetection50M | **could not be located** | no terms page exists to verify |
| PS-14 synthetic (`data/transactions.csv`) | generated in-repo | not third-party data; no external licence |

---

## 5. What completing this worksheet does and does not unblock

**Unblocks:** plan §35's "licence/access verified" checkbox for the datasets whose
outcome is **F-P** or **F-R** *and* whose restriction the reviewer records as
compatible with PS-14's use.

**Does NOT unblock:** reviewer assignment (§32), the 14 unresolved decisions, the
78 freeze findings, `FREEZE_RECORD.json`, independent audit of the exposure ledger
(**U-7**), IBM v2 publisher provenance (**U-4**), or Track M.

**F-P and F-R are not equivalent.** A `F-R` outcome that forbids redistribution of
derived artifacts may still block this project even though the checkbox could be
ticked. The reviewer must state the restriction's effect explicitly (§3, per dataset).

---

## 6. Evidence-quality bar

A reviewer entry is admissible only if it carries **all** of:

1. the **exact terms URL** as retrieved, with retrieval date;
2. **verbatim quoted text** for the operative clause — not a paraphrase;
3. the **named licensor/publisher** as stated in the terms;
4. the **access route** actually used to obtain the file;
5. the reviewer's **identity and date**.

A summary, a blog post, an aggregator page, or another project's assertion does
**not** clear a cell. This bar mirrors why the existing manifest claims were
retracted: each cited a source that did not say what was attributed to it.

---

*End of `licence-evidence-checklist 1.0-draft`. This worksheet grants no licence,
resolves no eligibility item, and is not written back into any record.*