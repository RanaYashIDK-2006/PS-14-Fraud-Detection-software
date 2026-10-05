# Dataset Exposure Ledger

**Required by:** `docs/RESEARCH_PLAN.md` §34 (Referenced Artifacts) and §14 (Dataset Exposure History).
**Artifact path:** `docs/evaluation/DATASET_EXPOSURE_LEDGER.md`
**Version:** `exposure-ledger 1.0-draft`
**Status:** DRAFT — pre-freeze preparation artifact. **NOT frozen, NOT independently audited, NOT approved.**
**Authored at git state:** `5a5ff55cc03df73318fdf31a16e3665f159a4720` (2026-10-04).
**§34 path cell:** the plan's §34 row reads `[TO BE FROZEN — path]`. This file is the
*proposed* repository location; the path decision itself remains an unresolved freeze-time
decision and has **not** been written into the plan.
**Self-hash:** a document cannot carry its own hash (plan §29). This artifact's SHA-256 is
to be recorded in `docs/FREEZE_RECORD.json` **at freeze**, not here.
**Independent audit (§14: "must be independently audited before the confirmatory freeze"):**
`NOT ESTABLISHED` — no independent auditor has reviewed this ledger.

---

## 1. Purpose

Record, in one append-only place, which datasets the project has been exposed to, what
hashes identify them, and how prior results on them must be classified — so that no
exploratory result can later be presented as untouched confirmatory evidence.

This ledger **records** exposure facts; it does not grant eligibility. Eligibility
state lives in the companion `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md`.

## 2. Authoritative sources (SHA-256 of the source files as read on 2026-10-04)

| Source | SHA-256 | Role here |
|---|---|---|
| `docs/RESEARCH_PLAN.md` (§13/§14/§15) | `eab5a0615801db7e50ab4d1c0ec73f93905e461b5bd8eacc617e0418d656b9ce` | normative exposure rules |
| `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` | `fdde71cbd00e8e34ce439f3cc582e31e73452a0405ea3ff51c11136adc9065a6` | preregistered dataset roles (DRAFT, 11 markers) |
| `docs/evaluation/claims_registry.jsonl` | `336ba4a0478a1bf0466fd37a80f10522f06c16c913927ccb550d9640c49f9100` | claim-level classifications (22 claims) |
| `docs/PHASE_NR03_EVALUATION_SEMANTICS.md` | `451f663c879c6309472bc1c3315696ac353be6f6b6f998df3f91b642665dbac2` | pinned dataset identities/roles |
| `docs/PHASE_NR02_PROTOCOL_REVIEW.md` (§G) | `913579cd817ae7c0508f5c99d52a420ddbd14fe9fe4c811d81a37bef8b91abac` | prior dataset-availability table |
| `misc/benchmarks/datasets.json` | `6d8e22d37f6eefb7f7e312d5494b8941021637735ec514d55e8a0ee86860620a` | row counts / sizes registry |
| `backend/scripts/prereg_harness.py` (`DATASETS`) | `7a75ab5ab97d31aa9582c32d0d31a3ac2cd2174cbe9c51aff7d0528c3bc86e9f` | executable dataset identity pins |

Where another document is authoritative, it is referenced rather than restated.

## 3. Exposure history (plan §14 — reproduced, not reinterpreted)

| Dataset | Prior project exposure | Existing-model status | New Track-M confirmatory status |
|---|---|---|---|
| ULB | Used during development/evaluation | **EXPLORATORY** | May be used only under explicitly frozen new-model rules |
| Kaggle fraudTrain/fraudTest | Used during development/evaluation | **EXPLORATORY** | May be used only under explicitly frozen new-model rules |
| IBM v2 | Used during development/evaluation | **EXPLORATORY** | May be used only under explicitly frozen new-model rules |
| IEEE-CIS | No established prior model exposure | Candidate | **CANDIDATE CONFIRMATORY** pending eligibility |
| BAF | No established prior model exposure | Candidate | **CANDIDATE CONFIRMATORY** pending eligibility |

Additional recorded exposure fact (plan-consistent): prior Kaggle results **influenced
historical promotion decisions** (phases 24/25 — `docs/PHASE_NR03_EVALUATION_SEMANTICS.md`),
so exposure includes influence on model-selection history, not only reported numbers.

## 4. Dataset identity (hashes recomputed from the working tree, 2026-10-04)

Repository-file hashes (dataset identity), verified to match the pins in
`prereg_harness.py::DATASETS` and the NR-03 registry exactly:

| Dataset | Path | SHA-256 (recomputed) | Rows (registry) | Acquired |
|---|---|---|---|---|
| ULB | `data/creditcard.csv` | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` | 284,807 | yes |
| IBM v2 | `data/credit_card_transactions-ibm_v2.csv` | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` | **24,386,900** (was `NOT ESTABLISHED`; measured 2026-10-04 — see §4.1) | yes |
| Kaggle fraudTest | `data/kaggle_fraud/fraudTest.csv` | `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0` | 555,719 | yes |
| Kaggle fraudTrain | `data/kaggle_fraud/fraudTrain.csv` | `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d` | 1,296,675 | yes |
| PS-14 synthetic | `data/transactions.csv` | `9f0f56bf0549fccf0c6335ece1514d1ae515a57f809a72658ce5fcaf48525d63` | NOT ESTABLISHED | yes (generated) |
| IEEE-CIS | `data/external/IEEE_CIS/` (expected) | **no dataset hash — NOT ACQUIRED** | — | no |
| BAF | `data/external_benchmark/` | **no dataset hash — NOT ACQUIRED** (directory absent, verified 2026-10-04) | — | no |

No dataset hash is asserted for any dataset that has not been acquired.

### 4.1 Additive reconciliation entry (2026-10-04)

Appended per `docs/evaluation/DATASET_PROVENANCE_RECONCILIATION.md`. Pre-edit SHA-256
of this file: `3b1b280040379407b49aff68c79672ea30fae1e10d5233dd9a92f6e4667788e4`.
**No status is upgraded; no exposure classification changes.**

| Field | OLD value | NEW value | Evidence | Exposure effect |
|---|---|---|---|---|
| IBM v2 row count | `NOT ESTABLISHED (size 2.35 GB, NR-02 §G)` | **ESTABLISHED = 24,386,900** data rows; 29,757 positives (0.122%) | full CSV parse (rows with all 15 fields = 24,386,900; malformed = 0); `wc -l` = 24,386,901 incl. header | **none** — IBM remains `EXPLORATORY` |
| IBM v2 raw schema | *(not recorded here; see eligibility manifest §4)* | **ESTABLISHED = 15 columns**: `User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name, Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?` | header read; `prereg_harness.py::DATASETS["ibm_v2"].label_column = "Is Fraud?"`; `_forensic_common.py::USECOLS` (14 of 15) | **none** |

### 4.2 Duplicate extracts — recorded so they are never counted as independent evidence

| Path | SHA-256 | Relationship | Independent rows |
|---|---|---|---|
| `data/ealtman2019/credit_card_transactions-ibm_v2.csv` | `b01fa323…` | byte-identical copy of IBM v2 | **0** |
| `data/User0_credit_card_transactions.csv` | `61c4ed49…` | first 19,963 data rows of IBM v2, position-identical | **0** |
| `data/ealtman2019/User0_credit_card_transactions.csv` | `61c4ed49…` | byte-identical copy of the User0 extract | **0** |

Neither `prereg_harness.py::DATASETS` nor `misc/benchmarks/datasets.json` registers
any of these, so no current registry counts them as datasets.
`misc/reports/phase16/dataset_inventory.json` lists the two identical files as **two
separate entries**; that historical report must not be read as independent evidence.
**Effective distinct real transaction volume remains 24,386,900 — not 24,406,863.**

## 5. Prior-result classifications (plan §15 — preserved verbatim, with claim-level detail)

Plan-level (§15) classifications are **not rewritten here**; claim-level registry
classifications are shown alongside so the two layers stay distinguishable:

| Evidence (§15) | §15 classification | Claim-level classification (`claims_registry.jsonl`) |
|---|---|---|
| ULB in-domain | **DEMONSTRATED / SELF-TESTED** according to existing evidence record | C-001 ROC-AUC 0.976 `DEMONSTRATED`; C-002 Recall@1%FPR 91.8% `DEMONSTRATED`; C-003 PR-AUC 0.883 `DEMONSTRATED`; historical headlines C-101…C-104 registered |
| IBM v2 in-domain | **DEMONSTRATED / SELF-TESTED** according to existing evidence record | C-004 XGB ROC-AUC 0.982 `SELF-TESTED` (artifact records no seed/git → stays SELF-TESTED) |
| IBM cross-dataset (~0.873) | **EXPLORATORY / NOT INDEPENDENT** | C-015 ROC-AUC 0.873 `DEMONSTRATED` as a *measurement*, notes: "Same generator family — not independent" |
| Kaggle transfer (~0.435–0.595) | **FAILED / NON-CONFORMING** (degraded/non-conforming feature representation) | C-010 0.47 fraudTrain `SELF-TESTED`; C-011 0.435 intermediate `NOT ESTABLISHED`; C-012 FPR 44.9% intermediate `NOT ESTABLISHED`; C-013 0.595 fraudTest `DEMONSTRATED` as a *measurement*; C-014 FPR 9.9% `DEMONSTRATED` |

An experiment-level `FAILED / NON-CONFORMING` classification and a claim-level
`DEMONSTRATED` measurement are different statements and both are preserved as recorded.

## 6. Exposure rules carried forward

1. **Previously exposed datasets (ULB, Kaggle, IBM v2) are exploratory evidence only.**
   They cannot become "untouched" confirmatory Track M datasets by any amount of
   re-analysis (§14).
2. **Pre-freeze exploratory work is permitted; confirmatory Track M is not**
   (freeze-ordering resolution, `docs/PHASE_RP01_RESEARCH_PLAN_ADOPTION.md` §H):
   NR-05 may run pre-freeze *only as exploratory work on previously exposed datasets*,
   cannot generate confirmatory Track M evidence, and cannot alter the frozen decision
   framework. Confirmatory Track M on untouched eligible datasets requires the complete
   freeze procedure (statistical review · all placeholders resolved ·
   `docs/FREEZE_RECORD.json` complete · freeze checker passes · tagged commit).
3. **Use of exposed datasets for new models** requires "explicitly frozen new-model
   rules" (§14) — those rules are `[TO BE FROZEN]` and are not established here.
4. **Untouched datasets** (IEEE-CIS, BAF per §14, subject to eligibility): architecture
   families, feature rules, hyperparameter ranges, tuning budgets, preprocessing and
   selection rules must be frozen *before* any inspection that could influence them (§14).
5. **"New model" ≠ "no prior information"** — design choices were informed by prior
   exposure (§14, plan §3 Track M preamble).

## 7. Not established / open items (explicitly not resolved by this artifact)

| Item | Status |
|---|---|
| Independent audit of this exposure ledger (§14) | **NOT ESTABLISHED** — required before confirmatory freeze; owner/auditor not assigned (no owner invented) |
| §34 path cell for this ledger | **`[TO BE FROZEN]`** — remains unresolved in the plan by design |
| Licence/access status of ULB, IBM v2, Kaggle files | **NOT ESTABLISHED** (see eligibility manifest §4.1 C-5/C-7 — the earlier "Open licence" assertion is retracted; NR-02 §G records `License: NOT ESTABLISHED` for both ULB and IBM v2) |
| IEEE-CIS prior-exposure verification beyond §14's table | **NOT ESTABLISHED** — §14 records "no established prior model exposure"; no independent audit has confirmed it |
| BAF prior-exposure verification beyond §14's table | **NOT ESTABLISHED** — dataset not acquired |
| Row count of IBM v2 file | **ESTABLISHED = 24,386,900** (measured 2026-10-04; §4.1) — *was `NOT ESTABLISHED`* |
| Whether `fraudTest` retains confirmatory status despite prior exposure | **PENDING REVIEW** (open question recorded in NR-03; reviewer-owned) |

## 8. Change discipline

Append-only: every future entry must cite its source file/hash. No entry may upgrade a
status (`NOT ESTABLISHED` → `ESTABLISHED`, `BLOCKED` → eligible, `EXPLORATORY` →
untouched) without a cited, reviewable source. Nothing in this ledger may be used to
retrospectively alter confirmatory decision rules (plan §36).

*End of `exposure-ledger 1.0-draft`.*
