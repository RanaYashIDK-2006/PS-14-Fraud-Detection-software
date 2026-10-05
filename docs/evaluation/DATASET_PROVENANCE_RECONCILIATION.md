# Dataset Provenance Reconciliation

**Artifact path:** `docs/evaluation/DATASET_PROVENANCE_RECONCILIATION.md`
**Version:** `provenance-reconciliation 1.0-draft`
**Status:** DRAFT — provenance reconciliation artifact.
**NOT frozen, NOT independently audited, NOT approved.**
**Date of work:** 2026-10-04.
**Upstream phase:** `docs/evaluation/DATASET_48_FEATURE_COVERAGE.md` ("Phase 1",
SHA `83397f937055349ce74d67d0dbefa13da549aea9537ea2ed79d03e3fc1a3ad92`).

**Records amended by this artifact (additive, with change history):**
`docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` (pre-edit `92b766be…`),
`docs/evaluation/DATASET_EXPOSURE_LEDGER.md` (pre-edit `3b1b2800…`).

**Deliberately NOT amended:** `docs/RESEARCH_PLAN.md`, the pre-registered protocol,
`docs/metric_definitions.md`, model artifacts, thresholds, feature definitions,
the decision memo, the review-resolution package, the reviewer assignment record,
and `backend/scripts/nr05_diagnostics.py`. Reasons are given per item in §12.

**What this phase did not do:** no experiment was run, no model was trained, no
dataset was acquired, no 50M dataset was generated, no reviewer was assigned, no
approval was created, and no eligibility was granted.

---

## 1. Objective

Convert the flow from

```
DOCUMENT CLAIM  →  ASSUMED DATA
```

into

```
ACTUAL DATA  →  AUTHORITATIVE REPOSITORY RECORDS  →  FUTURE REVIEW / FREEZE
```

Every correction below is traceable to a measurement reproduced in this document,
and every prior value is preserved verbatim so the change history is auditable.

---

## 2. Source hierarchy

Which file is authoritative for each kind of fact. Where two sources disagreed,
the higher rank won and the lower-rank claim was corrected, not deleted.

| Rank | Source | Authoritative for |
|---|---|---|
| 1 | **The data files themselves** (`data/*.csv`), read directly | row counts, column sets, label columns, entity identifiers, byte sizes, label distributions |
| 2 | **Executable pins**: `backend/scripts/prereg_harness.py::DATASETS` | dataset identity: path, label column, SHA-256, role, blocked status |
| 3 | **Executable derivations**: `native_features.py`, `altman_native_ensemble.py`, `train_altman_native.py`, `_forensic_common.py` | the native 48 contract, which raw fields the model consumes, how history features are derived |
| 4 | **Deployed manifest**: `models/production/manifest.json`, `models/model_records/*.json` | what the production model actually trained on, its split periods and locked threshold |
| 5 | **Records**: `DATASET_ELIGIBILITY_MANIFEST.md`, `DATASET_EXPOSURE_LEDGER.md`, `DATASET_48_FEATURE_COVERAGE.md` | eligibility decisions, exposure classifications, coverage claims |
| 6 | **Narrative/protocol documents**: `RESEARCH_PLAN.md`, `PHASE_NR0*.md` | rules and protocol intent — **not** a substitute for measurement |

**Key consequence.** Phase 1's report was *treated as a claim to verify*, not as an
authority. Every Phase 1 measurement was independently re-derived here, and the
corrections below go **beyond** what Phase 1 found.

---

## 3. IBM v2 actual file identity

All measured 2026-10-04 from the working tree.

| Property | Value |
|---|---|
| Path | `data/credit_card_transactions-ibm_v2.csv` |
| SHA-256 | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |
| Byte size | **2,350,744,057** bytes |
| Data rows | **24,386,900** |
| Columns | **15** |
| Label column | `Is Fraud?` (values `Yes` / `No`) |
| Positives | **29,757** (`Yes`) — 0.122% |
| Malformed rows | **0** (every data row carried all 15 fields) |
| `wc -l` | 24,386,901 (header + data rows) |
| Matches pin? | **Yes** — `prereg_harness.py::DATASETS["ibm_v2"]` gives the same path, `label_column = "Is Fraud?"`, and the same SHA-256 |

**Authoritative row count: 24,386,900.** Derived by a full streaming CSV parse that
counted rows carrying all 15 fields, cross-checked against `wc -l`.

---

## 4. IBM v2 actual schema

The complete header, exactly as read:

```
User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,Zip,MCC,Errors?,Is Fraud?
```

Mapping to the native 48's raw-field contract (§2.4 of `native_features.py`):

| Native raw field | Column present? | Native raw field | Column present? |
|---|---|---|---|
| `amount` | ✅ `Amount` (`$134.09` style strings) | `merchant_id` | ✅ `Merchant Name` |
| `ts` | ✅ `Year`+`Month`+`Day`+`Time` | `city_id` | ✅ `Merchant City` |
| `use_chip` | ✅ `Use Chip` (Chip/Online/Swipe) | `card_id` | ✅ `Card` |
| `mcc` | ✅ `MCC` | `user_id` | ✅ `User` |
| `merchant_state` | ✅ `Merchant State` | `zip` | ✅ `Zip` |
| `errors` | ✅ `Errors?` | | |

**All 12 raw-field groups the native 48 needs are present.** The 15-column schema is
the schema the native contract was written against.

### 4.1 Independent corroboration (no 168-column file anywhere)

| Evidence | Result |
|---|---|
| `_forensic_common.py::USECOLS` (the forensic pipeline's own declaration) | **14 columns**, all from the 15-column schema; no `hour_diff` |
| `prereg_harness.py::DATASETS["ibm_v2"].label_column` | `"Is Fraud?"` — the 15-column file's label, **not** `isFraud` |
| `misc/reports/phase16/dataset_inventory.json` | records `n_cols: 15` for the sibling extract, with the same 15 column names |
| Repo-wide search for `168` in `docs/` | **one** hit: the eligibility manifest claim itself |
| Repo-wide search for `hour_diff` | only a **local variable** in `unified_detection_test.py:89-91` (an hour-deviation computation); **no dataset column anywhere** |

---

## 5. D1 discrepancy investigation

### 5.1 What the record claimed

`DATASET_ELIGIBILITY_MANIFEST.md` §4 (pre-edit) asserted, all marked **ESTABLISHED**:

| Field | Old claim |
|---|---|
| feature semantics | `168 raw features; hour_diff = derived (deltas), preprocessing fidelity concerns flagged in NR-01` |
| label semantics | `is_fraud` |
| temporal information | `trans_date` + `trans_time`, `unix_time` |
| entity identifiers | `cc_num present` |
| leakage assessment | `hour_diff` derived during feature extraction (deltas), **not point-in-time safe as evaluated** (NR-01 §A) |
| acquisition status | `present, 2,466,191,524 bytes` |

### 5.2 Finding: the whole row described a different, absent dataset

**None** of `is_fraud`, `cc_num`, `trans_date`, `trans_time`, `unix_time`, or
`hour_diff` exists in the acquired file. All six are fields of the **168-column
`credit_card_transactions.csv`** layout — a different artifact that is **not present
anywhere in this repository**.

This is larger than Phase 1 recorded. Phase 1 found the *feature-count* and
*`hour_diff`* cells wrong; this investigation shows the **label, temporal, entity-id
and byte-size cells were wrong too**. The §4 row was written against a dataset this
project does not have.

### 5.3 Citation audit — the supporting sources contradict the claims

| Claim | Cited source | What the source actually says | Verdict |
|---|---|---|---|
| "flagged in NR-01", "`hour_diff` … not point-in-time safe (NR-01 §A)" | `docs/PHASE_NR01_EVIDENCE_ENFORCEMENT.md` §A | §A is titled **"Objective"**. It contains **no** `hour_diff` statement, **no** `168`, and no dataset-schema claim. The only `IBM` mention is claim ID C-004 (XGB ROC-AUC). | **Broken citation** |
| "Open licence" claimed at protocol-era | `docs/PHASE_NR02_PROTOCOL_REVIEW.md` §G | §G records, verbatim: **"License: NOT ESTABLISHED"** for IBM v2, and **"License terms: NOT ESTABLISHED (no license file in repo)"** for ULB | **Contradicted by the cited source** |

So the `hour_diff` claim had **no support anywhere**, and the "Open licence" claim was
contradicted by the very document it cited.

### 5.4 Classification (the brief's A/B/C/D)

**D — an actual repository/data mismatch**, with a broken-citation paper trail.

- Not **A** (a stale description of another absent dataset): if it were, an earlier
  record would have substantiated it. **No record ever did** — the string `168 raw
  features` appears in exactly one place in the entire repository, with nothing
  behind it.
- Not **B** (a historical artifact of a real file): no 168-column file ever existed in
  this tree; there is no hash, no registry entry, no code path.
- Not **C** (a copied external claim): the *field names* are consistent with the
  public 168-column layout, so the content was plausibly copied from the external
  description — but **no provenance for that copy is asserted here**, because none is
  knowable from the repository.
- It **is** a real mismatch: a documentary record asserting measured facts about a
  file it does not describe.

### 5.5 What is *not* concluded

The repository does **not** establish *why* the error occurred or *who* introduced it
— the project is not a git repository, so there is no blame-bearing history to read.
No provenance is invented for the 168-column claim.

---

## 6. D2/D3 row-count reconciliation

### 6.1 The discrepancy

`backend/scripts/nr05_diagnostics.py:370` hard-codes `n_file_rows=24_386_899`.
The file's true count is **24,386,900**.

### 6.2 What the constant is

It is a **diagnostic dict field**, passed once into a metadata dict. Repo-wide search
shows `n_file_rows` appears **exactly once** in all code — at its definition. **No
computation consumes it.** NR-05's subsample is formed by
`skiprows=lambda i: i > 0 and i % stride != 0` over the real file, so the constant
cannot have influenced any NR-05 number.

### 6.3 Materiality: zero

| Quantity | With 24,386,899 | With 24,386,900 |
|---|---|---|
| stride-1/37 subsample size | 659,104 | 659,105 |
| difference | — | **1 row (0.00015%)** |

**No NR-05 result, metric, or conclusion changes.**

### 6.4 Decision: preserved as history, not rewritten

`nr05_diagnostics.py` was **not modified**, and neither was the decision memo (2 sites)
nor the review-resolution package (2 sites), all of which cite `24,386,899` as *what
NR-05 recorded*.

Rationale, applying the brief's own distinction — *"a historical experiment record
saying that NR-05 used a particular constant must not be rewritten merely to make
history look cleaner"*:

1. NR-05 is an executed, recorded experiment; the constant is part of what it recorded.
2. The memo and resolution are **reviewer-owned**, `PENDING REVIEW`, at 0/14 resolved,
   and are enforced by `review_resolution_check.py` (R1–R12). Editing an evidence
   citation inside a reviewer decision row is not this phase's authority.
3. The correction is permitted by the brief (*"may be corrected"*) but is not required,
   it has zero scientific effect, and applying it would break the documented
   correspondence between those documents and NR-05's recorded constant.

**Therefore the authoritative count is recorded in the records that govern dataset
facts** — the eligibility manifest §4 and the exposure ledger §4 (both now
`ESTABLISHED = 24,386,900`) — while the historical `24,386,899` is preserved in place
and flagged here as a **reviewer-disposition item** (§13, item U-2).

---

## 7. Duplicate extract finding

### 7.1 Finding: `User0_credit_card_transactions.csv` is a 100% duplicate extract

| File | SHA-256 | Rows | Independent rows |
|---|---|---|---|
| `data/credit_card_transactions-ibm_v2.csv` | `b01fa323…` | 24,386,900 | 24,386,900 |
| `data/ealtman2019/credit_card_transactions-ibm_v2.csv` | `b01fa323…` | 24,386,900 | **0** (byte-identical copy) |
| `data/User0_credit_card_transactions.csv` | `61c4ed49…` | 19,963 | **0** (extract) |
| `data/ealtman2019/User0_credit_card_transactions.csv` | `61c4ed49…` | 19,963 | **0** (byte-identical copy of the extract) |

**Verification method.** The IBM v2 file is grouped by `User`, so User 0's block is
contiguous at the start. A positional comparison was run: reading the first 19,968
data rows of IBM v2 and comparing them element-wise against all 19,963 rows of the
User0 extract gave **19,963 / 19,963 exact matches**, with a clean boundary — IBM row
index 19,963 is `User = 1`, while the extract's last row is `User = 0`.

**The two User0 files are byte-identical to each other** (same SHA-256), and
`data/ealtman2019/` is a **full duplicate copy of the whole corpus** (~2.35 GB).

### 7.2 Is the repository treating it as independent?

| Question | Answer | Evidence |
|---|---|---|
| Is it a registered dataset? | **No** | absent from `prereg_harness.py::DATASETS`, `misc/benchmarks/datasets.json`, `models/production/manifest.json`, `_forensic_common.py`, `data_provenance.py` |
| Does any training pipeline use it? | **No** | the native trainer (`train_altman_native.py`) reads only `data/credit_card_transactions-ibm_v2.csv`; the production manifest's `training_dataset` is that file |
| Does any evaluation pipeline use it? | **Yes — as a benchmark input** | `backend/scripts/honest_benchmark.py:272` and `improved_paysim_ealtman.py:222` read `data/ealtman2019/User0_credit_card_transactions.csv` |
| Do the exposure records count it as independent evidence? | **The historical phase-16 report does** | `misc/reports/phase16/dataset_inventory.json` lists the two identical files as **two separate entries** (both sha `61c4ed49…`) |
| Does any current authoritative record count it? | **No** | neither `prereg_harness.py` nor `datasets.json` registers it |

### 7.3 Consequences recorded (files **not** deleted)

1. Those two benchmark scripts evaluate on **rows drawn from the same corpus the
   production model trained on** — the User0 block lies inside the training window
   (User 0's rows span 2002–2009; training period is 1995-06-01…2015-12-31). Any
   metric they report is therefore **not independent evidence**. Recorded, not re-run.
2. **Effective distinct real transaction volume remains 24,386,900**, not
   24,406,863.
3. The relationship is now recorded in the eligibility manifest §4.2 and the exposure
   ledger §4.2 so no future phase can mistake the extract for independent data.

---

## 8. Native 48 contract verification

**The contract was not modified, and does not need to be.** Verified:

| Check | Result |
|---|---|
| `ALTMAN_NATIVE_FEATURES` (privacy_layer) | 48 entries, 48 unique |
| `ALTMAN_NATIVE_FEATURES` (risk_engine) | 48 entries, element-wise **equal** to the above |
| Order preserved | `amt, log_amt, amt_sq … amt_x_online, amt_x_night, user_merch_count` |
| Renamed / redefined during this phase? | **No** — `native_features.py` SHA `5760a203…` and `altman_native_ensemble.py` SHA `55f4ad52…` are unchanged |
| `feature_list.json` vs deployed manifest | both declare 48, `altman_native_v2` |
| IBM v2 raw fields map to the construction? | **Yes** — all 12 raw-field groups present (§4) |

The Phase 1 coverage matrix's premise (native 48 = 48/48 derivable from IBM v2) is
re-confirmed against the file rather than assumed.

---

## 9. Exposure-ledger corrections

Applied to `docs/evaluation/DATASET_EXPOSURE_LEDGER.md`. **Additive only.**

| Field | OLD | NEW | Evidence | Exposure effect |
|---|---|---|---|---|
| IBM v2 row count (§4 table) | `NOT ESTABLISHED (size 2.35 GB, NR-02 §G)` | **ESTABLISHED = 24,386,900** (29,757 positives) | full CSV parse + `wc -l` | **none** — IBM stays `EXPLORATORY` |
| IBM v2 raw schema | *(absent from ledger)* | **ESTABLISHED = 15 columns** (enumerated, new §4.1) | header read + 3 independent corroborations | **none** |
| IBM row count in §7 open items | `NOT ESTABLISHED (not recorded in repository registries)` | **ESTABLISHED = 24,386,900** | same | **none** |
| Duplicate extracts | *(absent)* | new §4.2: 4 files, **0 independent rows** for 3 of them | SHA-256 + positional comparison | **none** — prevents future double-counting |
| Licence row in §7 | `NOT ESTABLISHED` | unchanged, but the "Open licence" cross-reference is now annotated as retracted | NR-02 §G | **none** |

### 9.1 Explicitly **not** changed in the ledger

| Item | Kept as-is | Why |
|---|---|---|
| ULB / Kaggle / IBM exposure rows | `EXPLORATORY` | nothing in this phase changes exposure |
| Kaggle transfer result | `FAILED / NON-CONFORMING` | unaffected |
| IEEE-CIS / BAF exposure | `CANDIDATE CONFIRMATORY`, not acquired | unaffected |
| §5 claim classifications | unchanged (C-001…C-018) | no claim re-measured |
| §6 exposure rules | unchanged | plan-governed |
| §7 "Independent audit of this ledger" | `NOT ESTABLISHED` | cannot be self-certified |
| §34 path cell | `[TO BE FROZEN]` | freeze-time decision, not ours |

**No status was upgraded.** The only `NOT ESTABLISHED → ESTABLISHED` transitions are
two *measured* facts (row count, column count), neither of which touches eligibility,
exposure, or any `FAILED`/`NON-CONFORMING` classification.

---

## 10. Eligibility-manifest corrections

Applied to `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md`, recorded in a new
**§4.1 Change history** with every prior value preserved verbatim.

| ID | Field | OLD (preserved in §4.1) | NEW | Material? |
|---|---|---|---|---|
| **C-1** | IBM row count | `NOT ESTABLISHED` | `ESTABLISHED = 24,386,900` | no |
| **C-2** | IBM feature semantics | `168 raw features; hour_diff = derived (deltas)…` `ESTABLISHED (concern)` | `15 raw columns` (enumerated) `ESTABLISHED` (measured) | **yes** |
| **C-3** | IBM leakage assessment | `hour_diff … not point-in-time safe (NR-01 §A)` `ESTABLISHED (defect)` | `NOT ESTABLISHED` — no such column; citation unsupported | **yes** |
| **C-4** | IBM provenance | `IBM Base Fraud Detection dataset` | file identity + 15-col schema `ESTABLISHED`; publisher provenance still `NOT ESTABLISHED` | clarifying |
| **C-5** | IBM licence | `"Open licence" claimed at protocol-era` `PENDING REVIEW` | claim **retracted** (NR-02 §G says `License: NOT ESTABLISHED`); state stays `PENDING REVIEW` | **yes** |
| **C-6** | IBM label / temporal / entity ids | `is_fraud` / `trans_date`+`trans_time`+`unix_time` / `cc_num` | `Is Fraud?` (Yes/No) / `Year`+`Month`+`Day`+`Time` / `User`,`Card`,`Merchant Name`,`Merchant City` | **yes** |
| **C-7** | ULB licence | cited NR-02 §G `"Open licence"` | citation **retracted** (NR-02 §G: `NOT ESTABLISHED`); other conflicts retained verbatim | **yes** |
| **C-8** | ULB row count | free-text conflict note (284,807 vs 284,808) | `ESTABLISHED = 284,807`; 284,808 is **header-inclusive** | no |
| **C-9** | acquisition size, **all four acquired datasets** | ULB `171,712,196`; fraudTrain `117,543,644`; fraudTest `49,313,965`; IBM `2,466,191,524` | ULB `150,828,752`; fraudTrain `351,238,196`; fraudTest `150,354,339`; IBM `2,350,744,057` — **none of the four matched its file** | no |

Also added: **§8 Dataset class register**, separating *acquired + hashed* /
*acquired + hashed, duplicate* / *described publicly, not acquired* /
*not obtainable* / *rejected for the native contract* / *requires reviewer decision*.

### 10.1 What the manifest deliberately still refuses to do

- **No dataset is declared eligible.** Eligibility remains `PENDING REVIEW` /
  `BLOCKED` / `EXPLORATORY USE ONLY` exactly as before.
- **Row count is not an eligibility criterion.** The 50M target does **not** appear
  in the manifest and was not introduced. Plan §35 and §13 criteria are untouched.
- **Licences remain uncleared** for all datasets.
- **IEEE-CIS and BAF remain `BLOCKED — not acquired`.**

---

## 11. Phase 1 artifact impact

**No factual error was found in the Phase 1 coverage matrix, so it was not
rewritten.** Re-verified mechanically:

| Check | Result |
|---|---|
| Matrix shape | 48 rows × 11 columns (index + feature + 9 datasets) |
| Total classification cells | 432 = 48 × 9 |
| Vocabulary | only the 7 permitted classes |
| §7.3 totals vs matrix cells | all 9 columns consistent |
| Native 48 names referenced | all 48 present |

**One extension.** Phase 1's D1 was scoped to the feature-count and `hour_diff` cells.
This phase found three further wrong cells in the same manifest row (label, temporal,
entity ids — C-6) plus the byte-size class (C-9). Phase 1's *finding* stands; the
*scope* is now wider. Phase 1 §14.1 (D1) is therefore **narrower than the current
evidence**, and this artifact is the authoritative record of the discrepancy.

**Not changed in Phase 1:** the coverage matrix, all classification totals, the
derivability analysis, and the 50M conclusion.

---

## 12. Historical evidence preservation

| Historical item | Disposition | Reason |
|---|---|---|
| Manifest §4 old cells (168/`hour_diff`/`is_fraud`/`cc_num`/`unix_time`/2,466,191,524) | **Preserved verbatim** in manifest §4.1 change history (C-1…C-6, C-9) | required: the fact that the record existed must survive |
| ULB old licence text | **Preserved verbatim** in manifest §4.1 C-7 and as an inline note in §1 | conflicts retained, not merged or deleted |
| `nr05_diagnostics.py` `n_file_rows=24_386_899` | **Left in place, unmodified** | historical constant of an executed experiment; zero scientific effect; cited by 4 sites in 2 reviewer-owned documents (§6.4) |
| Decision memo `24,386,899` (2 sites) | **Left unmodified** | reviewer-owned, `PENDING REVIEW`, 0/14, checker-enforced |
| Review-resolution package `24,386,899` (2 sites) + `hour_diff` rows | **Left unmodified** | same; also N1–N13 negatives are checker-enforced |
| `misc/reports/phase16/dataset_inventory.json` (two identical entries) | **Left unmodified** | historical report artifact; annotated here and in ledger §4.2 |
| Model artifacts, thresholds, scalers, manifests | **Untouched** | verified by hash below |
| `docs/RESEARCH_PLAN.md`, `docs/metric_definitions.md` | **Untouched** | verified by hash below |

### 12.1 Verified-unchanged (SHA-256)

| File | SHA-256 | Status |
|---|---|---|
| `docs/RESEARCH_PLAN.md` | `eab5a061…` | unchanged |
| `docs/metric_definitions.md` | `8665549b…` | unchanged |
| `models/production/manifest.json` | `e6da1a43…` | unchanged |
| `backend/src/privacy_layer/native_features.py` | `5760a203…` | unchanged |
| `backend/src/risk_engine/altman_native_ensemble.py` | `55f4ad52…` | unchanged |
| `backend/scripts/nr05_diagnostics.py` | `0e4b69d0…` | unchanged |
| `docs/evaluation/STATISTICAL_REVIEW_DECISION_MEMO.md` | `b99c32f4…` | unchanged |
| `docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` | `17a95e2a…` | unchanged |
| `docs/evaluation/DATASET_48_FEATURE_COVERAGE.md` | `83397f93…` | unchanged |

**Changed:** `DATASET_ELIGIBILITY_MANIFEST.md` (was `92b766be…`) and
`DATASET_EXPOSURE_LEDGER.md` (was `3b1b2800…`), both with in-document change history.

---

## 13. Remaining unresolved discrepancies

Open items that this phase could **not** close. Each names what is missing and who
must resolve it. **None is invented, guessed, or worked around.**

| ID | Item | State | Blocks | Who resolves |
|---|---|---|---|---|
| **U-1** | **Licence/terms unverified for every acquired dataset** (ULB, Kaggle ×2, IBM v2). Plan §35's "licence/access verified" is unchecked | `PENDING REVIEW` / `NOT ESTABLISHED` | freeze; any confirmatory use | Data-governance/privacy + reviewers |
| **U-2** | `24,386,899` remains in NR-05 + memo + resolution as *what NR-05 recorded* | historical, documented | nothing scientifically (0.00015% effect) | Reviewer disposition — confirm the authoritative count is 24,386,900 |
| **U-3** | **Root cause of the D1 mismatch is unknown** — the project is not a git repository, so no blame-bearing history exists | `NOT ESTABLISHED` | understanding *how* a record came to describe an absent file | Not resolvable from the repository |
| **U-4** | IBM v2 **publisher provenance** is still not archived (the file's origin record does not exist in-repo) | `NOT ESTABLISHED` | §13 provenance field | Data-governance |
| **U-5** | `fraudTest` confirmatory-vs-exposed tension | `PENDING REVIEW` | Track M role assignment | Statistical reviewer |
| **U-6** | IEEE-CIS / BAF: terms unverified, not acquired | `BLOCKED` | both as confirmatory datasets | Requires acquisition + manual verification (not authorised here) |
| **U-7** | Exposure ledger and eligibility manifest are **not independently audited** | `NOT ESTABLISHED` | confirmatory freeze (plan §14) | An independent auditor — **none assigned** |
| **U-8** | `honest_benchmark.py` / `improved_paysim_ealtman.py` report metrics on rows inside the training window | documented, not re-run | interpretation of those historical metrics | Statistical reviewer |
| **U-9** | 56 placeholders + 11 prereg markers + missing `FREEZE_RECORD.json` | 78 freeze findings | freeze | Reviewers + freeze procedure |
| **U-10** | 14 reviewer decisions unresolved; both reviewers `NOT ASSIGNED` | 0/14 | review, freeze, Track M | Plan §32 sign-off |

---

## 14. 50M dataset status

**UNCHANGED. Phase 1's conclusion stands verbatim: a defensible ~50,000,000-row
native-complete corpus is not established, and no such dataset should be generated.**

| Check | Result |
|---|---|
| Real native-complete rows available | **24,386,900** (IBM v2, unchanged by this phase) |
| Distinct real volume after removing duplicates | **24,386,900** — the User0 extract and `ealtman2019` copy add **0** |
| Duplicate rows used to inflate the count | **0** — this phase removed, rather than added, an inflation risk |
| 50M target present in the Research Plan? | **No** — and it was not introduced |
| 50M recorded as an eligibility criterion? | **No** |
| Any 50M dataset generated? | **No** |

The brief's prohibitions were all observed: nothing was synthesized, no rows were
duplicated to reach a target, no datasets were concatenated, no native feature was
fabricated, and no synthetic transaction is labelled real.

---

## 15. Reviewer / freeze implications

**This phase does not make the project ready for statistical/domain review, and does
not attempt to.** It removes provenance defects; it does not supply the things review
actually requires.

| Precondition for review | State | Changed here? |
|---|---|---|
| Statistical reviewer assigned | `NOT ASSIGNED` | no |
| Domain reviewer assigned | `NOT ASSIGNED` | no |
| Reviewer decisions resolved | **0 / 14** | no |
| Research Plan frozen | `DRAFT / NOT FROZEN / NOT APPROVED` | no |
| Freeze findings | **78** | no |
| `FREEZE_RECORD.json` | absent | no |
| Exposure ledger independently audited | `NOT ESTABLISHED` | no |
| Licence/access verified (plan §35) | unchecked for all datasets | no |
| Dataset record factual accuracy | **was materially wrong in 6 cells** | **yes — corrected (§9, §10)** |
| Track M | `BLOCKED` | no |

**Net effect:** the *inputs* to review are now factually sound, but the *preconditions*
to start review are unchanged. Nothing here should be read as reviewer readiness.

---

## 16. Final status

> # PROVENANCE RECONCILED / PASS WITH LIMITATIONS

### What was verified

| Item | Verdict |
|---|---|
| IBM v2 row count | **24,386,900** — measured, cross-checked twice |
| IBM v2 raw schema | **15 columns**, enumerated; every one independently corroborated |
| D1 (168 features / `hour_diff`) | **Resolved.** Real repository/data mismatch with broken citations; corrected in the eligibility manifest (§4.1 C-2/C-3) |
| Additional defects found | 4 more (label, temporal, entity ids, all four byte sizes) → C-6, C-9 |
| Duplicate extract | **Resolved.** 19,963/19,963 rows positionally identical; recorded in both records; files not deleted |
| Native 48 contract | **Unchanged and verified** |
| Exposure ledger | Changed — 2 measured facts established, 1 open item closed, duplicate-extract section added. **No status upgraded.** |
| Eligibility manifest | Changed — 9 corrections logged (C-1…C-9) + dataset-class register. **No eligibility granted.** |
| Phase 1 artifact | **Unchanged**; no factual error found |
| 50M conclusion | **Unchanged** |
| Model / thresholds / features / metrics / plan | **Unchanged**, verified by SHA-256 |

### Limitations making this a PASS **with** limitations, not a clean PASS

1. **U-1** — no dataset is licence-cleared; plan §35 unchecked for all.
2. **U-3** — the *cause* of the D1 mismatch is unknowable from the repository.
3. **U-4** — IBM v2 publisher provenance still unarchived.
4. **U-7** — the amended records are **not independently audited**, which plan §14
   requires before confirmatory freeze. Amending a record is not certifying it.
5. **U-2** — a stale off-by-one survives deliberately in 3 documents as historical
   record, pending reviewer disposition.

**No genuine unresolved contradiction remains** between the repository's dataset
records and the measured data: every corrected cell now matches the files, and the
surviving known-wrong values are confined to explicitly-labelled historical text.

**This phase establishes no real-world fraud-detection effectiveness, no native
validation, and no eligibility. It establishes only that the repository's dataset
records now describe the data that are actually on disk.**

---

## Appendix A — The twelve required answers

**1. What is the authoritative IBM v2 row count?**
**24,386,900** data rows. Measured by full streaming CSV parse (rows carrying all 15
fields = 24,386,900; malformed = 0), cross-checked by `wc -l` = 24,386,901 including
the header. Positives: 29,757 (`Is Fraud? == Yes`, 0.122%). Recorded as
`ESTABLISHED` in eligibility manifest §4 and exposure ledger §4.

**2. What is the authoritative IBM v2 raw schema?**
**15 columns:** `User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?`. Corroborated by
`_forensic_common.py::USECOLS` (14), `prereg_harness.py::DATASETS` (`label_column =
"Is Fraud?"`), and `misc/reports/phase16/dataset_inventory.json` (`n_cols: 15`).

**3. What caused the 168-feature/`hour_diff` discrepancy?**
It is a **real repository/data mismatch** (category D), and its citations were
**broken**. The `hour_diff` claim cited NR-01 §A, which is titled "Objective" and
contains no such statement; the "Open licence" claim cited NR-02 §G, which records
`License: NOT ESTABLISHED`. The string `168 raw features` exists in exactly one place
in the repository with nothing behind it, and the whole §4 row described the
**absent** 168-column dataset (`is_fraud`, `cc_num`, `unix_time`, `trans_date`).
The *reason* it was introduced is **not recoverable** — the project is not a git
repository — and no provenance was invented for it.

**4. Was the discrepancy corrected, and where?**
**Yes** — in `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md`, in place (§4), with a
verbatim-preserving change history at §4.1 (entries **C-2** and **C-3**), and
duplicates recorded at §4.2. Supporting size/count facts were also added to
`docs/evaluation/DATASET_EXPOSURE_LEDGER.md` §4.1–4.2. No competing manifest was
created.

**5. Was the NR-05 row-count constant corrected or merely documented as historical?**
**Documented as historical — deliberately not modified.**
`backend/scripts/nr05_diagnostics.py:370` still reads `n_file_rows=24_386_899`, and
the 4 citation sites in the decision memo and review-resolution package are unchanged.
The constant is a diagnostic annotation consumed by no computation; the discrepancy is
1 row (0.00015% of the stride-1/37 subsample); and those citing documents are
reviewer-owned, `PENDING REVIEW`, at 0/14, and checker-enforced. The authoritative
count is recorded instead in the records that govern dataset facts. Logged as
reviewer-disposition item **U-2**.

**6. Is `User0_...csv` independent data or a duplicate extract?**
A **duplicate extract**: the first **19,963** data rows of IBM v2, verified
**19,963/19,963 exact positional matches**, with a clean boundary (IBM row 19,963 is
`User = 1`). Its SHA-256 `61c4ed49…` also matches the byte-identical copy at
`data/ealtman2019/…`. It contributes **0 independent rows** and is now recorded as
such in both the manifest (§4.2) and the ledger (§4.2). It is registered in no current
dataset registry. Two benchmark scripts do read the `ealtman2019` copy, so their
metrics are not independent evidence (**U-8**). **The file was not deleted.**

**7. Is the native 48-feature contract unchanged?**
**Yes — verified, not modified.** 48 entries, 48 unique, and the privacy-layer and
risk-engine lists are element-wise equal. Both source files are byte-unchanged
(`5760a203…`, `55f4ad52…`). No feature was renamed or redefined. All 12 raw-field
groups the contract needs are present in IBM v2, re-confirming Phase 1's mapping.

**8. Did the exposure ledger change?**
**Yes, additively.** IBM v2 row count `NOT ESTABLISHED → ESTABLISHED (24,386,900)`;
IBM raw schema added as `ESTABLISHED (15 columns)`; §7's open item closed; new §4.2
duplicate-extract section. **No exposure status was upgraded** — ULB/Kaggle/IBM remain
`EXPLORATORY`, Kaggle transfer remains `FAILED / NON-CONFORMING`, IEEE-CIS and BAF
remain unacquired, and the ledger remains unaudited (`NOT ESTABLISHED`).

**9. Did the eligibility manifest change?**
**Yes.** Nine corrections (C-1…C-9), each with the prior text preserved verbatim in
§4.1: row count, feature semantics, leakage assessment, provenance, IBM licence
(claim retracted), label/temporal/entity identifiers, ULB licence citation, ULB row
count, and byte sizes for all four acquired datasets. A dataset-class register was
added at §8. **No dataset became eligible**; all licences remain uncleared; IEEE-CIS
and BAF remain `BLOCKED`; the 50M target was **not** introduced as a criterion.

**10. Did Phase 1's 50M conclusion change?**
**No.** A defensible ~50M native-complete corpus is still not established and should
not be generated. Real native-complete volume remains **24,386,900**; this phase
*removed* an inflation risk (3 files that could have been double-counted) rather than
adding one. No 50M dataset was generated and no target was adopted.

**11. Is the project now ready for statistical/domain review?**
**No.** Provenance is reconciled, but readiness is unchanged: both reviewers remain
`NOT ASSIGNED`, **0 / 14** decisions resolved, the Research Plan is
`DRAFT / NOT FROZEN / NOT APPROVED`, `check_freeze.py` still reports **78** findings,
and `FREEZE_RECORD.json` is absent. This phase repaired the *inputs* to review, not
the *preconditions* to begin it.

**12. What still blocks the next review phase?**
In order: **(a)** reviewer assignment — nothing can start without it (plan §32);
**(b)** the 14 unresolved decisions; **(c)** 78 freeze findings and the missing
`FREEZE_RECORD.json`; **(d)** **U-1** licence/terms unverified for every dataset
(plan §35 unchecked); **(e)** **U-7** neither the exposure ledger nor this manifest is
independently audited, which plan §14 requires before confirmatory freeze; **(f)**
**U-4** IBM v2 publisher provenance unarchived. Remaining items U-2, U-3, U-5, U-6,
U-8, U-9, U-10 are enumerated in §13.

---

## Appendix B — U-8 held-out re-run specification (**prepared, NOT executed**)

`honest_benchmark.py:272` and `improved_paysim_ealtman.py:222` read
`data/ealtman2019/User0_credit_card_transactions.csv`, whose rows (User 0, spanning
2002–2009) fall **inside** the production training window (1995-06-01…2015-12-31,
`models/model_records/altman_native_v2_20260904_115703.json`). Their metrics are
therefore in-window and not independent evidence.

### B.1 Why this was not executed

1. **This phase forbids it.** Its brief states: *"Do not run a new fraud-detection
   experiment. Do not train a model."* Re-scoring a dataset with a deployed model is
   a new experiment.
2. **It would deepen the blocking exposure.** Both scripts score IBM v2 rows. IBM v2
   is already `EXPLORATORY` (plan §14); producing more IBM v2 results increases
   exposure without improving eligibility.
3. **No result could be recorded.** Directive: do not write back.

### B.2 Specification, ready for a phase that permits experiments

| Item | Value |
|---|---|
| Objective | Re-measure the two benchmarks on rows **outside** the production training window |
| Model | `altman_native_v2_20260904_115703`, loaded read-only; **no retraining, no refit, no threshold change** |
| Locked threshold | `0.7847116291110687` — used as-is |
| Data selection rule | IBM v2 rows with **derived timestamp > 2015-12-31** (i.e. after the training window closes) |
| Entity control | Report both (a) time-filtered all-entity and (b) entity-disjoint variants; do **not** silently reuse an in-window entity split |
| Contrast | Same metric set as the original runs, so the delta is attributable to the window change alone |
| Rows consumed | Must be **counted and recorded** — this is additional IBM v2 exposure and must be logged as such |
| Preconditions | Independent audit of the exposure ledger (**U-7**) should precede it, so the new exposure is recorded against an audited ledger |
| Prohibited | Writing the result into any `ESTABLISHED`/`DEMONSTRATED` claim; presenting it as independent evidence; altering the historical values already recorded |

### B.3 Status

**NOT EXECUTED.** Open item **U-8** stays open. The specification above changes
nothing about the current state and carries no result.

---

*End of `provenance-reconciliation 1.0-draft`.*