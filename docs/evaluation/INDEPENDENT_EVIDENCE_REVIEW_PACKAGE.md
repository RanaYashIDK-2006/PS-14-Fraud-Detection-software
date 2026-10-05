# Independent Evidence Review Package

**Artifact path:** `docs/evaluation/INDEPENDENT_EVIDENCE_REVIEW_PACKAGE.md`
**Version:** `independent-evidence-review-package 1.0-draft`
**Status:** DRAFT review packet. **This is NOT an independent audit, NOT a licence
clearance, NOT a statistical approval, NOT a domain approval, and NOT a freeze.**
**Date of work:** 2026-10-05.
**Authored at git state:** HEAD `5a5ff55cc03df73318fdf31a16e3665f159a4720`
(140 commits, **0 tags**; this file is **untracked**, like every other review
artifact — see §2.4 O-4). No commit is implied.

**Upstream artifacts carried forward (read, not reinterpreted):**

| Artifact | SHA-256 as measured 2026-10-05 | Role |
|---|---|---|
| `docs/RESEARCH_PLAN.md` | `eab5a0615801db7e50ab4d1c0ec73f93905e461b5bd8eacc617e0418d656b9ce` | normative plan (DRAFT) |
| `docs/metric_definitions.md` | `8665549b3d62fef36402d737910cd8aa1acfbde025dfee4aa1395a9e3d99ed07` | metric definitions (v1.0) |
| `docs/evaluation/DATASET_48_FEATURE_COVERAGE.md` | `83397f937055349ce74d67d0dbefa13da549aea9537ea2ed79d03e3fc1a3ad92` | Feature coverage + provenance (§3, §5, §7) |
| `docs/evaluation/DATASET_PROVENANCE_RECONCILIATION.md` | `da73e40b6bb566381dd01bed9aafde89bb0f90549b07e5131625aba016876be3` | Provenance reconciliation, U-1…U-10 |
| `docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` | `13f46ed218bb9c4c9b164ee81a72b33bb13c31b1476bd747cfa45c2e97e513ad` | Eligibility (§13 fields), C-1…C-9 |
| `docs/evaluation/DATASET_EXPOSURE_LEDGER.md` | `31af86fd7c089c6242ab27b28b3ecdf03e3bbe616944522f0e3074caeec6a501` | Exposure history (§14) |
| `docs/evaluation/DATASET_LICENCE_EVIDENCE_CHECKLIST.md` | `bfb6dd3d977e426f66bd74f84d9d6fdbc8036456fe55e2d6d533647b7c8acd6a` | Licence worksheet (unfilled) |
| `docs/evaluation/STATISTICAL_REVIEW_DECISION_MEMO.md` | `b99c32f49561a7ed2f764ab3cb3a48843e13968fb103bc89be6dcec74bd0131d` | 14 decisions, §5 |
| `docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` | `17a95e2afc18b1dd2b189aa307f90b79ed1e03701e51f37727e2754ca2d42d24` | Decision sheets + ownership |
| `docs/evaluation/REVIEWER_ASSIGNMENT_RECORD.md` | `b62e4f054523257cbc8ab1b752b65e21fed2a1da92605674f4da4e4a31a00d3e` | Roles + input contract (pre-engagement SHA `52a32bde…`; §11 dependency statement added 2026-10-05, no field populated) |
| `docs/evaluation/REVIEWER_ENGAGEMENT_BRIEF.md` | `bc1568fa55c6ff3ac5b8cc5d0f3b226894abdb5a7a06c6d1c37183829527e5d3` | **External-facing entry point** for a prospective reviewer (`READY FOR REVIEWER RECRUITMENT — REVIEW NOT YET COMMENCED`) |
| `docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md` | `fdde71cbd00e8e34ce439f3cc582e31e73452a0405ea3ff51c11136adc9065a6` | Preregistration (`0.1.2-draft`) |
| `docs/evaluation/claims_registry.jsonl` | `336ba4a0478a1bf0466fd37a80f10522f06c16c913927ccb550d9640c49f9100` | 22 claim classifications |
| `models/production/manifest.json` | `e6da1a433ec618576aa12208190d35df3b53e9349b1bb6e54c16465e324b2485` | production model record |
| `models/model_records/altman_native_v2_20260904_115703.json` | `45849237c96986b4e5aaffd94b3912c4358195ccea74b21b139fcda3b4d42091` | training governance record |
| `backend/scripts/prereg_harness.py` | `7a75ab5ab97d31aa9582c32d0d31a3ac2cd2174cbe9c51aff7d0528c3bc86e9f` | executable dataset identity pins |
| `backend/scripts/review_resolution_check.py` | `13a04fb3151eec2a6e7f41e7ac5253f4bea6ae5f0e338c636f829942103f0880` | decision-state checker |
| `scripts/check_freeze.py` | `6a8b9719cc0fdfc419648103e13b0cc0c5969bdd524228ee788509d9055fd601` | freeze checker (§30) |

**External entry point:** a prospective reviewer should start with
`docs/evaluation/REVIEWER_ENGAGEMENT_BRIEF.md` (added 2026-10-05), which carries
qualification requirements, the per-role review scope, and the recommended review
order; this packet remains the dataset/licence/audit evidence layer behind it.

**What this packet does not do:** it asserts no reviewer, no reviewer identity, no
approval, no licence outcome, no eligibility, no eligibility change, no exposure
change, no freeze, and no acquisition. It runs no experiment, trains nothing,
changes nothing, and imports no model or evaluation path. It is a *packet* a real
reviewer can execute — not a substitute for one.

---

## 1. Purpose and review boundary

### 1.1 Purpose

Make the evidence package **reviewable**: one place where a qualified statistical
reviewer, a qualified domain reviewer, and an independent dataset/provenance auditor
can see (a) what data exist and what is established about them, (b) which questions
are already answered by repository evidence, (c) which questions only they can
answer, and (d) exactly what they must verify before freeze can be considered.

This packet **collects and points**. It does not decide.

### 1.2 What this packet is not

| It is NOT | Because |
|---|---|
| An independent audit | No independent auditor exists. **No independent audit has occurred** (`DATASET_EXPOSURE_LEDGER.md` §7; reconciliation U-7). |
| A licence clearance | No dataset licence is cleared. Plan §35's "licence/access verified" is **unchecked for every dataset**; all outcomes remain open (licence checklist §2). |
| A statistical approval | 0 of 14 decisions resolved; both reviewers `NOT ASSIGNED`. |
| A domain approval | Same. |
| A freeze | Research Plan `DRAFT / NOT FROZEN / NOT APPROVED`; `docs/FREEZE_RECORD.json` **absent**; 0 tags. |
| A new governance framework or registry | No JSON registry, database table, ledger schema, phase number, metric, or approval state is created. §6–§8 reuse the project's existing vocabularies. |

### 1.3 Hard boundary observed by this phase

Not performed, and asserted nowhere in this packet: claiming an independent audit
occurred · claiming any licence is cleared · assigning a reviewer · inventing a
reviewer identity · approving a statistical or domain decision · freezing the
Research Plan · running confirmatory Track M · training a model · generating a 50M
dataset · acquiring a restricted dataset · changing model architecture · changing
thresholds · changing metric definitions · altering plan decision criteria.

### 1.4 Outcome vocabulary used in this packet

§7's audit-item outcomes use exactly the six words the review task permits —
`ESTABLISHED`, `NOT ESTABLISHED`, `FAILED`, `BLOCKED`, `INCONCLUSIVE`,
`REQUIRES REVIEW` — and **every audit item starts at `REQUIRES REVIEW`**: the packet
pre-fills no item as `ESTABLISHED` and certifies nothing.

These six words are **audit-item findings inside this packet only**. They are *not* a
new approval state and they do **not** replace the decision vocabulary that already
exists for `/01`–`/14` (`PENDING REVIEW` / `APPROVED` / `REJECTED` / `REVISE` /
`NOT ESTABLISHED` / `BLOCKED` / `INCONCLUSIVE`, resolution §8.2). No decision sheet is
edited by this packet.

### 1.5 Mechanical companion

`backend/scripts/review_package_check.py` verifies this packet's own structure and
non-claims: required sections present, required datasets present, no licence marked
cleared, no reviewer marked assigned/approved, no 50M dataset claimed, and the
freeze state unchanged. It is a document-consistency checker in the same class as
`review_resolution_check.py` and `claim_evidence_check.py`; it is **not** registered
in the battery (rationale in the checker docstring — it must not duplicate
`review_resolution_check.py`'s anti-fabrication coverage, and this packet is a draft
navigation artifact, not a frozen one).

---

## 2. Current project state

### 2.1 State table (recorded, not changed)

| Fact | State | Source |
|---|---|---|
| Research Plan | **`DRAFT / NOT FROZEN / NOT APPROVED`** | plan §34; memo §2 |
| Preregistration | **`0.1.2-draft`, finalized: NO**, 11 markers | protocol header |
| Statistical reviewer | **`NOT ASSIGNED`** | assignment record §1.1 |
| Domain reviewer | **`NOT ASSIGNED`** | assignment record §1.2 |
| Decisions resolved | **0 of 14**; component states `NOT ASSIGNED=20` | resolution §10; `review_resolution_check.py --readiness` |
| Freeze findings | **78** (65 plan placeholders / 12 prereg markers / 1 missing `FREEZE_RECORD.json`) | `scripts/check_freeze.py` → exit 1 |
| `docs/FREEZE_RECORD.json` | **absent** | filesystem |
| Git freeze tag | **none** — `git tag` = 0 | git |
| Track M | **BLOCKED** — zero eligible confirmatory datasets | plan §23/§14; ledger §6 |
| Dataset eligible for confirmatory use | **none** | eligibility manifest §1–§9 |
| Licences cleared | **none** — plan §35 checkbox unchecked for all | licence checklist §1 |
| Exposure ledger independently audited | **`NOT ESTABLISHED`** (U-7) | ledger §7 |
| `data/external/`, `data/external_benchmark/` | **absent** (verified 2026-10-05) | filesystem |
| 50M dataset generated | **no** | §12 of this packet |
| Last recorded battery (reviewer-engagement phase) | **84 PASS / 0 FAIL** — 84 suites, run 2026-10-05T11:1x → 11:26:11, with the six services live (6/6 `/health` = 200). The pre-engagement run of the same battery recorded **81 PASS / 3 FAIL(1)**: the three live-service security suites (`security_test`, `sql_injection_test`, `security_ci_gate`) failed only because no listener was up on :8000–:8005 after a restart; each passed once the stack was restored (isolated re-runs: `sql_injection_test` 50/50 rc=0, `security_ci_gate` ALL CLEAR rc=0, `security_test` 15/16 — its one assertion is state-dependent and passes in the full battery) | `.freebuff/p114rq_battery_results.tsv`; `.freebuff/re_battery_driver.out`; previous-phase outputs `.freebuff/rp_*` |
| Invariant hashes | **unchanged** (§2.2) | measured 2026-10-05 |
| Git working tree | **dirty, 119 entries; not a single commit this session** | §2.4 O-4 |

### 2.2 Invariant hashes (measured 2026-10-05, unchanged)

| Artifact | SHA-256 | Status |
|---|---|---|
| `docs/RESEARCH_PLAN.md` | `eab5a061…` | unchanged |
| `docs/metric_definitions.md` | `8665549b…` | unchanged |
| `models/production/manifest.json` | `e6da1a43…` | unchanged |
| `models/production/altman_native/*` artifact pins | `22b8377b…` (cb), `d59aebcb…` (lgb), `b98fadf3…` (scaler), `a1cdebdf…` (xgb), `657459ac…` (feature_list) | **all five match the model record's pins** |
| `backend/src/privacy_layer/native_features.py` | `5760a203…` | unchanged (48-feature contract) |
| `backend/src/risk_engine/altman_native_ensemble.py` | `55f4ad52…` | unchanged (48-feature contract) |
| `backend/scripts/nr05_diagnostics.py` | `0e4b69d0…` | unchanged (`n_file_rows=24_386_899` deliberately preserved) |
| Locked production threshold | `0.7847116291110687` | unchanged (`manifest.json`) |
| Test metrics in the model record | AUC `0.972237`, PR-AUC `0.78749`, recall@1%FPR `0.412626` | unchanged |

No model, threshold, metric definition, feature definition, split rule or plan
decision criterion was altered by this phase.

### 2.3 What the reviewer receives that is *new* since the reconciliation

Only this packet. The reconciliation phase (`da73e40b…`) corrected six stale IBM
cells (C-2/C-3/C-6) plus the byte-size class (C-9) and recorded the duplicate
extracts; nothing else changed. Both records were amended *additively* with change
history and **no eligibility, licence, or exposure status was upgraded**.

### 2.4 Observations recorded by this packet (none resolved here)

These are verified repository observations that a reviewer must disposition. They
are **not** corrections, and this packet writes none of them back into the artifacts
they describe.

| ID | Observation | Evidence (measured 2026-10-05) | Owner |
|---|---|---|---|
| **O-1** | Reviewer-owned documents still carry the **superseded IBM cells**: resolution §7.1 lists "IBM v2 has `cc_num`" and "IBM `hour_diff` … `ESTABLISHED` — a recorded defect"; memo §7 `/13` cites "IBM 24,386,899 rows". The acquired file has **neither `cc_num` nor `hour_diff`** and has **24,386,900** rows | §4.3 below; measured header; full CSV parse | Reviewer disposition (U-2 class). Those files are checker-enforced and were deliberately not edited |
| **O-2** | **U-3's stated rationale is superseded.** The reconciliation says the root cause of D1 is unknowable "because the project is not a git repository". This **is** a git repository: HEAD `5a5ff55…`, **140 commits**, `git log` works. What is true instead: the plan, the NR-0x artifacts, the preregistration and every `docs/evaluation/*` review artifact are **untracked**, so they have no history | `git rev-parse HEAD`, `git rev-list --count HEAD` = 140, `git status --porcelain` shows `??` for all of them | Reviewer / data-governance. U-3's *conclusion* may still hold for these files; its *reason* must be restated |
| **O-3** | The Research Plan contains **two sections numbered 34**: `# 34. Current Status` (line 1147) and `# 34. Referenced Artifacts` (line 1217, placed after §36). Other artifacts cite "§34 (Referenced Artifacts)" by name | `grep -n "^# 3[0-6]." docs/RESEARCH_PLAN.md` | Freeze-time plan correction; **the plan is not edited by this packet** |
| **O-4** | The authoritative review artifacts are **untracked** and the tree is dirty (119 entries). Nothing in this session was committed; the freeze mechanism (plan §29/§30, tag `freeze-*`) cannot cover artifacts that are not committed | `git status --porcelain` | Freeze procedure / owner |
| **O-5** | `data/` contains **regression-suite outputs that are not evidence records**: `cross_dataset_report.txt`, `cross_dataset_results.json`, `extended_dataset_results.json`, `real_dataset_results.json`, `security_attack_report.json` (written 2026-10-05 00:01–00:04 IST by battery suites `multi_dataset_test.py`, `extended_dataset_test.py`, `real_dataset_test.py`, `security_attack_test.py`). The eval ledger (135 records) has **no record** for those commands | battery log lines "Report saved to data/…"; `.freebuff/p114_battery.sh` lines 65, 76–78; ledger scan | Reviewer: must not be cited as dataset evidence |
| **O-6** | `models/production/` carries **battery side effects**: `altman_native/xgb_native.joblib` was rewritten 2026-10-04 23:55:49 with **hash-identical content** (still `a1cdebdf…`, verified against the model record), and `release_manifest.json` (23:56) describes a *different* `model_id` (`altman_native_E_hardneg_cert_20260904`) with placeholder-style values (`training_config_hash` = `e0e0…`, `source_git_sha` = `unknown`, `training_seed` = null). **Hash identity, not mtime, is the verification basis**; `release_manifest.json` is **not** the authoritative production record (the manifest points to `models/model_records/altman_native_v2_20260904_115703.json`) | `ls --time-style=full-iso`, `sha256sum`, manifest contents | Reviewer |
| **O-7** | `DATASET_48_FEATURE_COVERAGE.md` §3.2 records `paysim.csv` and `paysim_1m.csv` as **9 columns**; measured **8** (`type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`) | header read 2026-10-05 | Reviewer; **scientific effect none** — neither file carries a native-contract field and neither appears in any eligibility row |
| **O-8** | `misc/reports/phase16/dataset_inventory.json` lists **two** entries with the identical SHA `61c4ed49…`; it must not be read as two independent datasets | already recorded: ledger §4.2, manifest §4.2 | Restated here for the reviewer |

**None of O-1…O-8 is resolved by this packet**, and none changes any eligibility,
licence, exposure, or decision state.
---

## 3. Authoritative dataset inventory

Everything below is **measured or already recorded** in the cited artifact. Where a
fact is not established, it says so. Nothing here is acquired, generated, padded, or
re-classified by this packet. Full SHA-256 pins for the five registered datasets live
in `backend/scripts/prereg_harness.py::DATASETS` (four of them; PS-14 synthetic
pinned in `misc/benchmarks/datasets.json` and the ledger).

### 3.1 Datasets relevant to the plan — acquired

| Dataset | Path | Acquisition state | Row count (measured) | File size (bytes, measured) | Schema status | Native-48 contract status | Provenance status | Licence status | Exposure classification | Independent-evidence status |
|---|---|---|---|---|---|---|---|---|---|---|
| **ULB** Credit Card Fraud Detection | `data/creditcard.csv` | **ACQUIRED**, SHA `76274b691b16a6c4…` | **284,807** (492 `Class`=1, 0.173%) | **150,828,752** | **ESTABLISHED** (31 cols: `Time`, V1–V28, `Amount`, `"Class"`; `Time` = offset seconds; **no entity id**) | 1 `OBSERVED` / 2 `DERIVABLE` / 45 `UNAVAILABLE` | file identity `ESTABLISHED`; provenance *documentation* not archived → `NOT ESTABLISHED` | **`PENDING REVIEW`** — three conflicting in-repo references, not merged | **`EXPLORATORY`** (plan §14) | **`NOT ESTABLISHED`** — exploratory use only |
| **Kaggle fraudTrain** | `data/kaggle_fraud/fraudTrain.csv` | **ACQUIRED**, SHA `fd7139200dbfcbed…` | **1,296,675** (7,506 `is_fraud`, 0.579%) | **351,238,196** | **ESTABLISHED** (23 cols; `cc_num` entity; absolute `trans_date_trans_time`/`unix_time`; cardholder-level geo) | 1 `OBSERVED` / 18 `DERIVABLE` / 10 `PARTIALLY_DERIVABLE` / 19 `UNAVAILABLE` | file identity `ESTABLISHED`; archived provenance record `NOT ESTABLISHED` | **`PENDING REVIEW`** — no Kaggle licence citation located for this file | **`EXPLORATORY`** (plan §14) | **`NOT ESTABLISHED`** |
| **Kaggle fraudTest** | `data/kaggle_fraud/fraudTest.csv` | **ACQUIRED**, SHA `12d553ab19440c75…` | **555,719** (2,145, 0.386%) | **150,354,339** | **ESTABLISHED** (same 23-col schema) | 1 / 18 / 10 / 19 (same as fraudTrain) | file identity `ESTABLISHED`; archived provenance record `NOT ESTABLISHED` | **`PENDING REVIEW`** + open tension **U-5** (confirmatory-vs-exposed) | **`EXPLORATORY`**, `PENDING REVIEW` on the tension | **`NOT ESTABLISHED`** |
| **IBM v2** (15-column card-transaction corpus) | `data/credit_card_transactions-ibm_v2.csv` | **ACQUIRED**, SHA `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` | **24,386,900** (29,757 `Is Fraud?`=Yes, **0.122%**) | **2,350,744,057** | **ESTABLISHED — 15 columns** (enumerated in §4) | **48 / 48** — 4 `OBSERVED` + 44 `DERIVABLE`, 0 gaps | file identity + measured schema `ESTABLISHED`; **publisher provenance `NOT ESTABLISHED` (U-4)** | **`PENDING REVIEW`** — prior "Open licence" claim **retracted** (C-5) | **`EXPLORATORY`** (production training set) | **`NOT ESTABLISHED`** — production-trained, therefore *not* independent |
| **PS-14 synthetic** | `data/transactions.csv` | **ACQUIRED (generated in-repo)**, SHA `9f0f56bf0549fccf…` | **`NOT ESTABLISHED`** (generator output; not counted here) | **1,749,115** | in-repo generator output, 27 cols incl. the 21-feature causal vector | 48 `SYNTHETIC_ONLY` | **`ESTABLISHED`** (in-repo generator) | not third-party data; no external licence | used throughout development | **synthetic — cannot establish real-world performance** (plan §13) |

**Only IBM v2 satisfies the native-48 contract in full (48/48).** That is a coverage
fact, not an eligibility fact: every acquired dataset remains `EXPLORATORY` or
synthetic-only, and **no dataset is confirmatory-eligible**.

### 3.2 Duplicate extracts and auxiliary files (0 independent rows)

| Path | SHA-256 / measurement | Relationship | Independent rows | Registered anywhere? |
|---|---|---|---|---|
| `data/ealtman2019/credit_card_transactions-ibm_v2.csv` | `b01fa323…`, 2,350,744,057 bytes | **byte-identical copy** of IBM v2 | **0** | no — absent from `prereg_harness.py::DATASETS` and `datasets.json` |
| `data/User0_credit_card_transactions.csv` | `61c4ed49a98bf924c5df0a740e1c415b91624c83821f8415bd8db34c2d6a0d3c`, 19,963 rows, 15 cols | **extract**: first 19,963 data rows of IBM v2, position-identical (19,963/19,963) | **0** | no |
| `data/ealtman2019/User0_credit_card_transactions.csv` | `61c4ed49…`, 1,899,258 bytes | byte-identical copy of the extract | **0** | no |
| `data/ealtman2019/sd254_cards.csv` | `85af46a5789cf5ac9674cee4184c828688fcf2c00566549f752393112ed14a7b`, 6,146 rows × 13 cols, 487,120 bytes | auxiliary card table present in the duplicate corpus directory; **not analysed by any phase** | n/a | no — no record classifies it |
| `data/ealtman2019/sd254_users.csv` | `99eff71679d5dcf3f49e537fd65ba848271faa4e8ba8e4d9ab368c26034a3d03`, 2,000 rows × 18 cols, 224,394 bytes | auxiliary user table, same directory | n/a | no |

`misc/reports/phase16/dataset_inventory.json` lists the two `61c4ed49…` files as **two
separate entries** — see O-8. **Effective distinct real volume remains 24,386,900**
(not 24,406,863).

### 3.3 Working files present in `data/` that no eligibility record classifies

Measured 2026-10-05 (this packet); recorded so the inventory is complete and so no
reviewer mistakes them for registered datasets. None appears in
`prereg_harness.py::DATASETS`; `paysim_1m` and `fraud_data` appear in
`misc/benchmarks/datasets.json` (row counts only).

| Path | Rows × cols (measured) | Bytes | SHA-256 | Status |
|---|---|---|---|---|
| `data/validation_kaggle.csv` | 284,807 × 23 | 46,192,650 | `0ffb090e07a62a5aa545cb0397820bc7aec298c4f410f99ec70e51ccd7434d0a` | PS-14 21-feature-contract table with ULB's row count; **no eligibility record; provenance `NOT ESTABLISHED`** |
| `data/validation_realistic.csv` | 1,514 × 20 | 211,989 | `ac213f3f41c93273d41de13807f51eefd759abc8b6c7e78799911a407c52c9fe` | same contract family; **no record** |
| `data/fraud_data.csv` | 21,693 × 30 | 7,699,716 | `3702c41af6f706f95006f0f64d3b971dec7e472d818d4b144b6c30d3517b5030` | ULB-schema sample; no native fields |
| `data/paysim.csv` | 100,000 × **8** | 10,543,137 | `71c42b3b1c961ee25c37049c78fdac7d1f3fd90c225c716c3b7b6c882cdcd647` | PaySim output; no native fields (see **O-7**: coverage artifact records 9 columns) |
| `data/paysim_1m.csv` | 1,200,000 × **8** | 63,358,199 | `fa1df0f1b65905cc8ed217e62d1d577f82c0029eac8ac369e914683d72f551f6` | PaySim output; no native fields (O-7) |
| `data/batch_scores.csv` | 200,000 × 3 | 6,337,083 | `d23db00d51e07f9902719218a761d8f86dc630939dc8c83bdc48c073cdd2c6f5` | score dump (`ml_score`, `risk_score`, `decision`); not a dataset |
| `data/feedback_labeled.csv` | 5 × 19 | 986 | `2dd28cabd123ccc85949c2e453cf5b67ea628c4f33482b5bcdd78c9530cfcf55` | feedback sample |
| `data/cross_dataset_report.txt`, `data/cross_dataset_results.json`, `data/extended_dataset_results.json`, `data/real_dataset_results.json`, `data/security_attack_report.json` | — | — | — | **regression-suite outputs**, written 2026-10-05 00:01–00:04 IST; **not in the evidence ledger** (O-5) |

**None of these is eligible for, or usable as, confirmatory evidence.**

### 3.4 Candidate datasets **not** acquired (no hash is asserted)

| Dataset | Source / reference (as recorded) | Acquisition state | Rows (as recorded) | Native-48 contract status | Provenance | Licence status | Exposure classification | Independent-evidence status |
|---|---|---|---|---|---|---|---|---|
| **IEEE-CIS** (Vesta) | `kaggle.com/c/ieee-fraud-detection`; IEEE DataPort | **`NOT ACQUIRED`** — `data/external/` absent | 590,540 train (+506,284 test) | 1 `OBSERVED` / 2 `DERIVABLE` / 3 `PARTIALLY_DERIVABLE` / 41 `UNAVAILABLE` / **1 `UNKNOWN`** | `NOT ESTABLISHED` | **`BLOCKED`** — terms unchecked (plan §35) | plan §14: "no established prior model exposure" (not independently confirmed) | **`NOT ESTABLISHED`** |
| **BAF** (Bank Account Fraud) | `kaggle.com/datasets/sgpjesus/bank-account-fraud-neurips-2022` | **`NOT ACQUIRED`** — `data/external_benchmark/` absent | 6 synthetic variants (as recorded) | 48 `REJECTED` — unit of record is account-**opening**, not a transaction | `NOT ESTABLISHED` | **`BLOCKED`** | plan §14 candidate | **`NOT ESTABLISHED`** |
| **Zenodo 2026 deployment-derived** | `zenodo.org/records/20359708`; DOI `10.5281/zenodo.20030064` | **`NOT ACQUIRED`** | 56,962 × 38 (30 input), 98 labelled fraud (0.172%) | 1 `OBSERVED` / 13 `DERIVABLE` / 34 `UNAVAILABLE` | recorded limitations: proof-of-concept demo (not a licensed bank); verification bias; model-output/label circularity; 12 known FNs | **CC BY 4.0 recorded from the record page** — not yet reviewer-admissible (§6 evidence bar) | none | **`NOT ESTABLISHED`** — recorded as not eligible as unbiased ground truth |
| **FreeFraudDetection50M** | *could not be located* (HF API searches failed) | **`NOT ACQUIRED`** | 50,000,000 (claimed) | 48 `SYNTHETIC_ONLY` | **`NOT ESTABLISHED`** | "CC BY-NC 4.0" **claimed**, not established | none | **`NOT ESTABLISHED`** |
| **Dal Pozzolo ~49.86M** | ESPA 41(10):4915–4928, 2014 | **`NOT OBTAINABLE`** — confidential, never released | ~49,858,600 | 48 `UNAVAILABLE` | `NOT ESTABLISHED` | no public terms exist | none | **`NOT ESTABLISHED`** |
| **Worldline** `INST-WORLDLINE-001` | no public URL | **`NOT ACQUIRED`** — institutional access | ~60,000,000 (as recorded) | not assessed | `NOT ESTABLISHED` | confidential | none | **`NOT ESTABLISHED`** |
| **Novatti** `INST-NOVATTI-001` | no public URL | **`NOT ACQUIRED`** — confidentiality | 126,184 | not assessed | `NOT ESTABLISHED` | confidential | none | **`NOT ESTABLISHED`** |
| **Elliptic / Elliptic++** | `kaggle.com/ellipticco/elliptic-data-set` | **`NOT ACQUIRED`** | 203,769 nodes | not assessed (wrong domain: crypto graph) | `NOT ESTABLISHED` | public research release (as recorded) | none | **`NOT ESTABLISHED`** |

**No acquisition was performed by this phase and none is requested by it.**

---

## 4. IBM v2 provenance evidence

### 4.1 What is established (repository evidence)

| Property | Value | How established |
|---|---|---|
| Path | `data/credit_card_transactions-ibm_v2.csv` | filesystem |
| SHA-256 | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` | recomputed 2026-10-04; matches `prereg_harness.py::DATASETS["ibm_v2"]` and `models/production/manifest.json` |
| Byte size | **2,350,744,057** | `os.path.getsize`; corrected under C-9 (prior figure 2,466,191,524 matched no file) |
| Data rows | **24,386,900** | full streaming CSV parse (rows carrying all 15 fields = 24,386,900; malformed = 0); `wc -l` = 24,386,901 incl. header |
| Positives | **29,757** (`Is Fraud? == Yes`) = 0.122% | full parse |
| Raw columns | **15** | measured header (below) |
| Label | `Is Fraud?` (`Yes` / `No`) | measured; matches `prereg_harness` pin |
| Temporal fields | `Year`, `Month`, `Day`, `Time` (`HH:MM`) | measured |
| Entity identifiers | `User`, `Card`, `Merchant Name`, `Merchant City` (+ `Zip`, `Merchant State`) | measured |

```
User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,Zip,MCC,Errors?,Is Fraud?
```

**Relation to the native 48.** All 12 raw-field groups the contract consumes are
present, so the native 48 contract maps to this file (48/48 coverage: 4 `OBSERVED` +
44 `DERIVABLE`). The production model's `feature_source` records
`shared src/privacy_layer/native_features.py — train == production by construction`,
and `training_dataset_sha256` is the same `b01fa323…` pin.

### 4.2 Duplicate-copy relationships (0 independent rows)

| File | Relationship | Independent rows |
|---|---|---|
| `data/ealtman2019/credit_card_transactions-ibm_v2.csv` | byte-identical **copy of the whole corpus** | **0** |
| `data/User0_credit_card_transactions.csv` | **extract**: first 19,963 data rows, position-identical; clean boundary (IBM row index 19,963 is `User = 1`) | **0** |
| `data/ealtman2019/User0_credit_card_transactions.csv` | byte-identical copy of the extract | **0** |

Two benchmark scripts read the `ealtman2019` copy
(`backend/scripts/honest_benchmark.py:272`, `improved_paysim_ealtman.py:222`), whose
rows fall **inside** the production training window (User 0 spans 2002–2009; training
window 1995-06-01…2015-12-31). Their metrics are **not independent evidence**
(open item **U-8**; the held-out re-run is specified in the reconciliation Appendix B
and was **not executed**).

### 4.3 Historical conflicting records (preserved, not rewritten)

| Conflicting record | Where it survives | Status |
|---|---|---|
| "168 raw features; `hour_diff = derived (deltas)`"; `is_fraud`; `trans_date`+`trans_time`+`unix_time`; `cc_num`; size `2,466,191,524` | eligibility manifest **§4.1 change history C-2/C-3/C-6/C-9** (verbatim) | **Corrected in place** with the old text preserved |
| "Open licence" for IBM v2 | manifest §4.1 **C-5**; NR-02 §G actually records `License: NOT ESTABLISHED` | **Retracted**; state stays `PENDING REVIEW` |
| `n_file_rows=24_386_899` | `backend/scripts/nr05_diagnostics.py:370` (unmodified, deliberately) | **Preserved as history** (U-2); zero scientific effect (1 row = 0.00015% of the stride-1/37 subsample); consumed by no computation |
| `24,386,899` cited as what NR-05 recorded | decision memo (2 sites), resolution package (2 sites) | **Preserved**; reviewer disposition (U-2) |
| `cc_num` still listed as an IBM `ESTABLISHED` fact; `hour_diff` still listed as an IBM defect | resolution §7.1 (reviewer-owned); coverage doc §3.4 warns those rows inherit the caveat | **Still present (O-1)** — not edited by this packet |

**Current authoritative record:** eligibility manifest §4 (`ESTABLISHED` schema, row
count, label/temporal/entity fields) + exposure ledger §4/§4.1/§4.2 (row count,
schema, duplicate extracts). **D1** was classified as a *real repository/data
mismatch* with broken citations (NR-01 §A = "Objective", no such claim; NR-02 §G
contradicts the licence claim). The *cause* of the mismatch is **not recoverable**
(U-3 — reason restated under O-2).

### 4.4 Explicit statement required by the review task

> **The IBM v2 provenance reconciliation is repository evidence, not independent
> certification.**

No independent auditor has examined this file, its publisher provenance (U-4), its
licence, or the reconciliation's conclusions. The measurements above are reproducible
from the working tree by the reviewer; they are not an audit, and they do not make
the dataset eligible.

### 4.5 What the reviewer must establish about IBM v2

See §8 — eleven items, each **"Reviewer verification required."** Nothing in §4 is
self-certifying, and §8 explicitly forbids treating this packet as the verification.

---

## 5. Dataset exposure / independence boundary

Preserved from plan §14, the exposure ledger §3/§6 and the memo §3. **No independence
is upgraded by this packet.**

| Category | Datasets | Established basis |
|---|---|---|
| **Training data** | **IBM v2 only** — `train_rows` 193,027; training window 1995-06-01…2015-12-31; `train_seed` 42; `target_leakage: false`. Historical development runs also used PS-14 synthetic | `models/production/manifest.json`; model record |
| **Validation data** | **IBM v2** validation window 2016-01-01…2017-12-31 — used for calibration/threshold selection (`threshold_policy: "validation-only: max recall within val FPR<=1%"`; locked threshold `0.7847116291110687`) | model record |
| **Data already consumed as in-domain test** | **IBM v2** final-test window 2018-01-01…2020-02-29 (AUC 0.972237 / PR-AUC 0.78749 / recall@1%FPR 0.412626 recorded) | model record |
| **Previously exposed evaluation data** | **ULB, Kaggle fraudTrain, Kaggle fraudTest, IBM v2** — all `EXPLORATORY` under plan §14; prior Kaggle results also influenced historical promotion decisions | ledger §3, §5 |
| **Exploratory evaluations** | NR-05 diagnostics ran on the exposed sets (stride-1/37 IBM subsample; ULB; Kaggle); all labelled METHOD NOT YET FROZEN where applicable. **NR-05 cannot support independent replication** | memo §3.3; resolution §5 |
| **Duplicate extracts** | `User0_…csv` (×2) and `data/ealtman2019/*` — **0 independent rows**; must never be concatenated or counted separately | ledger §4.2; manifest §4.2 |
| **Prior transfer evaluations** | IBM cross-dataset (~0.873) `EXPLORATORY / NOT INDEPENDENT`; Kaggle transfer (~0.435–0.595) `FAILED / NON-CONFORMING`; the two in-window benchmark scripts (**U-8**) | ledger §5; reconciliation Appendix B |
| **Confirmatory-eligible data** | **NONE.** IEEE-CIS and BAF are `BLOCKED — not acquired`; every acquired dataset is `EXPLORATORY` or synthetic-only; Zenodo 2026 is not eligible as unbiased ground truth; the confidential corpora are unobtainable | plan §14/§23; manifest §9 |
| **Genuinely independent data** | **`NOT ESTABLISHED` — none exists in this project.** No dataset is untouched by exposure *and* acquired *and* licence-verified | ledger §7; manifest §9 |

**Consequence preserved:** plan §23's "zero eligible confirmatory datasets" applies,
so Track M stays `BLOCKED` regardless of any reviewer ruling.

---

## 6. Licence / provenance review worksheet

A reviewer-completed worksheet. **This packet asserts no licence.** Unknown
information is written `NOT ESTABLISHED`; datasets not acquired are written
`NOT ACQUIRED / REVIEW BLOCKED`. The words *cleared*, *approved*, *open* and
*verified* are **not** used as licence outcomes here (see §6.3). The detailed
evidence-capture form is `docs/evaluation/DATASET_LICENCE_EVIDENCE_CHECKLIST.md` §3;
this section carries the same required field set so a reviewer can work from the
packet alone.

### 6.1 Required fields — acquired datasets

| Field | ULB | Kaggle fraudTrain | Kaggle fraudTest | IBM v2 | PS-14 synthetic |
|---|---|---|---|---|---|
| Dataset | `data/creditcard.csv` | `data/kaggle_fraud/fraudTrain.csv` | `data/kaggle_fraud/fraudTest.csv` | `data/credit_card_transactions-ibm_v2.csv` | `data/transactions.csv` |
| Publisher / source | Kaggle "Credit Card Fraud Detection" dataset, ULB ML Group (as recorded) | Kaggle "Fraud Detection" (as recorded) | Kaggle "Fraud Detection" (as recorded) | Kaggle `credit_card_transactions` / IBM synthetic corpus family (as recorded) | in-repo generator (PS-14) |
| Source URL / repository reference **where already known** | `kaggle.com/datasets/mlg-ulb/creditcardfraud` | Kaggle dataset page (URL not archived) | Kaggle dataset page (URL not archived) | Kaggle page (URL not archived) | n/a — in-repo |
| Licence statement | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` (claim retracted, C-5) | n/a |
| Licence evidence location | **none archived**; three conflicting in-repo *references* (`metric_definitions.md` CC BY-SA 4.0; `dataset_card.py` "AGPL-3.0"; NR-02 §G `NOT ESTABLISHED`) | none archived | none archived | none archived; NR-02 §G records `License: NOT ESTABLISHED` | in-repo |
| Data-use restrictions | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | none (project-generated) |
| Redistribution restrictions | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | none |
| Research-use restrictions | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | none |
| Commercial-use restrictions (if stated) | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | none |
| Derivative-data restrictions (if stated) | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | none |
| Attribution requirements | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | n/a |
| Acquisition terms | `NOT ESTABLISHED` (public download; terms not read) | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | n/a |
| Provenance evidence | file identity + hash `ESTABLISHED`; provenance *record* `NOT ESTABLISHED` | same | same | file identity + schema `ESTABLISHED`; **publisher provenance `NOT ESTABLISHED` (U-4)** | in-repo generator |
| Archived provenance available? | **No** | **No** | **No** | **No** | Yes (generator source) |
| Independent verification required? | **Yes** | **Yes** | **Yes** | **Yes** | not for licence; yes for exposure role |
| Reviewer determination | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` |
| Reviewer identity | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT ASSIGNED` | `NOT ASSIGNED` |
| Reviewer date | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` |
| Reviewer rationale | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` | `NOT ESTABLISHED` |

### 6.2 Reviewer decision lines (machine-checkable; the only permitted values)

**ULB** — `Reviewer decision: NOT ESTABLISHED`

**Kaggle fraudTrain** — `Reviewer decision: NOT ESTABLISHED`

**Kaggle fraudTest** — `Reviewer decision: NOT ESTABLISHED` (a licence outcome does
**not** resolve **U-5**)

**IBM v2** — `Reviewer decision: NOT ESTABLISHED`

**PS-14 synthetic** — `Reviewer decision: NOT ACQUIRED / REVIEW BLOCKED`
(no third-party licence to clear; exposure role for evaluation still requires review)

**Duplicate extracts — `data/ealtman2019/credit_card_transactions-ibm_v2.csv`,
`data/User0_credit_card_transactions.csv`,
`data/ealtman2019/User0_credit_card_transactions.csv`** — `Reviewer decision:
NOT ESTABLISHED` (they inherit IBM v2's outcome and contribute 0 independent rows;
a byte-identical copy cannot carry different terms)

**Auxiliary files — `data/ealtman2019/sd254_cards.csv`,
`data/ealtman2019/sd254_users.csv`** — `Reviewer decision: NOT ACQUIRED / REVIEW
BLOCKED` (identity/schema newly recorded in §3.2; no record classifies them)

**IEEE-CIS** — `Reviewer decision: NOT ACQUIRED / REVIEW BLOCKED`

**BAF** — `Reviewer decision: NOT ACQUIRED / REVIEW BLOCKED`

**Zenodo 2026 deployment-derived** — `Reviewer decision: NOT ACQUIRED / REVIEW BLOCKED`

**FreeFraudDetection50M** — `Reviewer decision: NOT ACQUIRED / REVIEW BLOCKED`

**Dal Pozzolo ~49.86M** — `Reviewer decision: NOT ACQUIRED / REVIEW BLOCKED`

**Worldline** — `Reviewer decision: NOT ACQUIRED / REVIEW BLOCKED`

**Novatti** — `Reviewer decision: NOT ACQUIRED / REVIEW BLOCKED`

**Elliptic / Elliptic++** — `Reviewer decision: NOT ACQUIRED / REVIEW BLOCKED`

### 6.3 Explicit non-claims

- No dataset's licence is cleared. Plan §35's "licence/access verified" remains
  **unchecked for every dataset**.
- The four-outcome range of the licence checklist (**F-P / F-R / R / U**) is **not
  narrowed** by this packet. The current state for all acquired datasets is the
  checklist's **`U`**.
- `CC BY 4.0` for Zenodo 2026 and `CC BY-NC 4.0` for FreeFraudDetection50M are
  **recorded claims**, not reviewer-admissible determinations: the checklist §6
  evidence bar (exact URL + verbatim clause + named licensor + access route +
  reviewer identity) is **not met for any dataset**.
- The outdated repository licence assertions for ULB and IBM v2 were already found
  unsupported and retracted (C-5, C-7); no replacement is offered here.
---

## 7. Independent audit checklist (executable by an actual independent reviewer)

Fifteen items (A–O). **Every item is currently `REQUIRES REVIEW`** — this packet
pre-fills nothing as `ESTABLISHED` and certifies nothing. Permitted outcomes are
exactly: `ESTABLISHED` · `NOT ESTABLISHED` · `FAILED` · `BLOCKED` · `INCONCLUSIVE` ·
`REQUIRES REVIEW`. No new approval state is created: an item outcome is an audit
finding, not eligibility, not a licence outcome, and not a decision on `/01`–`/14`.

**A. Dataset identity**
*Evidence required:* for each dataset, path + SHA-256 + byte size, and a statement of which registry pins it. *Repository artifact:* `prereg_harness.py::DATASETS`, `misc/benchmarks/datasets.json`, eligibility manifest §1–§4, this packet §3. *Reviewer action:* recompute hashes and sizes from the working tree; confirm each matches the pin and the record; confirm no dataset is described by a document that does not match the file. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**B. File hash verification**
*Evidence required:* recomputed SHA-256 for every acquired file, including duplicates. *Repository artifact:* ledger §4; manifest §4.2; this packet §3. *Reviewer action:* recompute independently; record the command and the tool version; treat a mismatch as `FAILED`. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**C. Row-count verification**
*Evidence required:* data-row count with the parsing convention stated (header excluded; embedded newlines handled). *Repository artifact:* manifest §4 / ledger §4.1 (IBM = 24,386,900); coverage §3.2; `misc/reports/phase16/dataset_inventory.json`. *Reviewer action:* recount with an independent parser; confirm the counting convention; for IBM also confirm `wc -l` = 24,386,901 incl. header. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**D. Raw schema verification**
*Evidence required:* the exact header of each acquired file and the field-level mapping to the native contract. *Repository artifact:* reconciliation §4; coverage §3.3/§7; `_forensic_common.py::USECOLS`. *Reviewer action:* read headers; confirm IBM v2 = 15 columns with **no `hour_diff`** and **no 168-column file anywhere**; confirm `prereg_harness.py::DATASETS["ibm_v2"].label_column == "Is Fraud?"`. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**E. Feature-derivation verification**
*Evidence required:* proof that each of the 48 native features is either a direct read or a deterministic, prior-only function of source fields. *Repository artifact:* `native_features.py`, `altman_native_ensemble.py`, `train_altman_native.py`, coverage §7, `backend/research/public_feature_contract.json`. *Reviewer action:* inspect the derivations; confirm the two 48-name lists are element-wise equal; confirm history features use prior-only shifts (`expanding().mean().shift(1)`, `cumcount`) with no future information; confirm the 21-feature *public* contract's 9 unsourced features are not silently treated as observed. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**F. Label semantics**
*Evidence required:* the label column, its values, positive count and rate, and the label's provenance/generation method. *Repository artifact:* manifest §1–§4; coverage §3.2; model record. *Reviewer action:* confirm IBM `Is Fraud?` ∈ {Yes,No} with 29,757 positives (0.122%); confirm ULB `Class` = 492; confirm Kaggle `is_fraud` = 7,506 / 2,145; for every candidate dataset, confirm whether labels are human, processor-post-hoc, model-derived, or generated. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**G. Entity identifiers**
*Evidence required:* which entity keys exist per dataset, and whether entity-disjoint evaluation is possible. *Repository artifact:* manifest §1–§4; coverage §5. *Reviewer action:* confirm ULB has **no** entity identifier (entity-disjoint = not applicable, not a gap); confirm Kaggle `cc_num`; confirm IBM `User`/`Card`/`Merchant Name`/`Merchant City`; confirm **`cc_num` is not present in IBM v2** (O-1). *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**H. Temporal semantics**
*Evidence required:* timestamps, their granularity, and whether they are absolute or offsets. *Repository artifact:* manifest §1–§4; coverage §5; `_forensic_common.py`. *Reviewer action:* confirm IBM = `Year`/`Month`/`Day` + `Time` (`HH:MM`); ULB = offset seconds only; Kaggle = absolute datetimes; confirm the production split boundaries (1995-06-01…2015-12-31 / 2016-01-01…2017-12-31 / 2018-01-01…2020-02-29) are consistent with the data. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**I. Leakage assessment**
*Evidence required:* a point-in-time/leakage analysis per dataset, covering imported delta/history columns, target leakage, and label delay. *Repository artifact:* manifest §4 (C-3 retraction); `manifest.json` `target_leakage: false`; reconciliation §4. *Reviewer action:* verify that no acquired file ships a pre-extracted history/delta column; verify feature derivations are prior-only; verify the training/validation/test split does not overlap; assess label-delay risk for datasets whose labels are post-hoc. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**J. Train/test exposure**
*Evidence required:* an explicit statement of which rows/datasets were used for training, calibration, validation and testing, and which remain untouched. *Repository artifact:* model record; ledger §3/§5/§6; memo §3; §5 of this packet. *Reviewer action:* reconcile the production window against the record; confirm every exposed dataset's classification; confirm **no** dataset is claimed untouched. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**K. Duplicate / overlap assessment**
*Evidence required:* hash-level and row-level duplicate detection across all files. *Repository artifact:* ledger §4.2; manifest §4.2; coverage §3.2. *Reviewer action:* re-run the positional comparison for the User0 extract (19,963/19,963); confirm the `ealtman2019` copy is byte-identical; confirm no registry counts duplicates as datasets; check `misc/reports/phase16/dataset_inventory.json` (O-8) is not treated as evidence. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**L. Provenance**
*Evidence required:* for each dataset, a publisher/retrieval record (who distributed it, from where, when, under what terms, with what published schema). *Repository artifact:* coverage §5; manifest §1–§6; reconciliation §3–§5, U-4. *Reviewer action:* attempt to locate the publisher record for each acquired file; for IBM v2 confirm whether the on-disk file corresponds to a publicly described IBM synthetic corpus release and record the mapping; where no record exists, record `NOT ESTABLISHED`. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**M. Licence / terms**
*Evidence required:* exact terms URL, verbatim operative clause, named licensor, access route, retrieval date, reviewer identity (checklist §6 bar). *Repository artifact:* `DATASET_LICENCE_EVIDENCE_CHECKLIST.md`; §6 of this packet. *Reviewer action:* complete the worksheet per dataset; reconcile the retained conflicting ULB references; state each restriction's effect on this project (F-P/F-R/R/U); a summary or another project's assertion does **not** clear a cell. *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**N. Reproducibility**
*Evidence required:* evidence that the production artifacts and the reported metrics are reproducible from the pinned inputs. *Repository artifact:* model record; `train_altman_native.py`; `models/production/altman_native/*` pins; `backend/scripts/check_freeze_test.py`, `eval_record_test.py`. *Reviewer action:* confirm the five artifact pins match the files (this packet verified hash identity under O-6); confirm the threshold and split policy are recorded; confirm no artifact was silently regenerated (hash identity, not mtime). *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

**O. Evidence-artifact integrity**
*Evidence required:* that every claimed result resolves to a recorded evaluation, and that no unrecorded artifact is presented as evidence. *Repository artifact:* `reports/evaluation_runs/eval_ledger.jsonl` (135 at authoring → **142** at the closeout's final sweep; append-only; every extra record is a deterministic `calibration_test.py` re-run whose on-disk metrics, dataset hash and artifact hashes are byte-identical to the registered record), `docs/evaluation/claims_registry.jsonl` (22 claims), `claim_evidence_check.py`, `eval_record_test.py`. *Reviewer action:* sample claims and resolve them to ledger records; confirm the battery-suite outputs in `data/` (O-5) and `models/production/release_manifest.json` (O-6) are **not** used as evidence; confirm the append-only guarantee and the absence of fabricated approvals/badges (all six suites recorded in §15 of the review task). *Possible outcomes:* all six. **Current packet state: `REQUIRES REVIEW`.**

---

## 8. IBM-specific audit items (reviewer verification required)

**This checklist does not certify any of the following facts.** Each item states what
the independent reviewer must verify; each is currently **`REQUIRES REVIEW`**.

| # | Item to verify | Evidence the reviewer must produce | Packet's pointer |
|---|---|---|---|
| 1 | **24,386,900 rows** | independent parse + `wc -l` = 24,386,901 incl. header; parse convention stated | §4.1; manifest §4; ledger §4.1 |
| 2 | **15 actual raw columns** | the header as read, compared with the manifest's enumerated 15 | §4.1; reconciliation §4 |
| 3 | **No `hour_diff` field** | a repo-wide search for `hour_diff` proving no dataset column exists (only a local variable in `unified_detection_test.py:89-91`) | reconciliation §4.1/§5; manifest C-3 |
| 4 | **No 168-column file is being used** | proof that no 168-column file exists in the tree and that no pipeline reads one; the string `168 raw features` occurs in exactly one place (the retracted manifest cell) | reconciliation §4.1/§5.2 |
| 5 | **The six previously incorrect manifest claims were corrected** | the §4.1 change history C-1…C-9, each old claim preserved verbatim, each new claim matched to a measurement | manifest §4.1; §4.3 here |
| 6 | **`User0_…csv` is a duplicate extract** | positional comparison re-run (19,963/19,963) and the boundary case (`User = 1` at IBM row index 19,963) | §4.2; reconciliation §7 |
| 7 | **`data/ealtman2019/` is a duplicate corpus** | SHA-256 of both copies plus byte-size equality | §3.2, §4.2 |
| 8 | **Duplicate copies are not counted as independent evidence** | registry scan showing none of the duplicates is registered, and confirmation that no metric cites them as independent — including **U-8** (two benchmark scripts read the copy) | §3.2; ledger §4.2; reconciliation §7.2 |
| 9 | **Production training dataset identity is consistent with `models/production/manifest.json`** | `training_dataset_sha256` = `b01fa323…` matched to the file's recomputed hash; `train_rows` 193,027 vs the recorded window | §4.1; manifest; model record |
| 10 | **Native 48-feature derivation is consistent with the authoritative feature contract** | the two `ALTMAN_NATIVE_FEATURES` lists (48 unique, element-wise equal) and the prior-only derivations in the trainer | reconciliation §8; coverage §2.4/§7 |
| 11 | **Historical NR-05 row count 24,386,899 remains preserved as historical evidence** rather than silently rewritten | the constant still present at `nr05_diagnostics.py:370`, plus the four citation sites in the memo/resolution, all unmodified; zero scientific effect (1 row, 0.00015%) | §4.3; reconciliation §6, U-2 |

**Each item above: Reviewer verification required.** None is certified by this packet,
and none may be closed on the strength of this packet's own text.

---

## 9. Exposure / independence audit

Built only from repository records — **no experiment was run to establish this**. `—`
means "not applicable / not established". "Duplicate" entries are not datasets.

| Dataset | Previously exposed | Training | Calibration | Validation | Exploratory evaluation | Prior transfer evaluation | Duplicate / overlapping | Eligible for future confirmatory evaluation | Independent status |
|---|---|---|---|---|---|---|---|---|---|
| ULB | **Yes** | No (production) | No | No | **Yes** (NR-05 exploratory) | **Yes** (transfer, `FAILED / NON-CONFORMING`) | No | **No** | **`NOT ESTABLISHED`** |
| Kaggle fraudTrain | **Yes** | No | No | No | **Yes** | **Yes** (transfer) | No | **No** | **`NOT ESTABLISHED`** |
| Kaggle fraudTest | **Yes** | No | No | No | **Yes** | **Yes** (transfer) | No (distinct file; **U-5** open) | **No** (U-5 `PENDING REVIEW`) | **`NOT ESTABLISHED`** |
| IBM v2 | **Yes** | **Yes** (193,027 rows) | **Yes** (validation window, Platt/threshold) | **Yes** (2016–2017) | **Yes** (NR-05; in-domain test) | **Yes** (cross-dataset ~0.873) | **Yes** — 3 duplicate files (§3.2) | **No** | **`NOT ESTABLISHED`** |
| PS-14 synthetic | Used throughout development | Historical dev runs | — | — | **Yes** | — | No | **No** (synthetic; cannot establish real-world performance) | **`NOT ESTABLISHED`** |
| `User0_…csv` (×2), `ealtman2019/*` | via IBM v2 | No | No | No | read by two benchmark scripts (**U-8**) | — | **Duplicates of IBM v2 — 0 independent rows** | **No** | **`NOT ESTABLISHED`** |
| IEEE-CIS | **No established prior exposure** (not independently audited) | No | No | No | No | No | No | **No — `BLOCKED` (not acquired; terms unchecked)** | **`NOT ESTABLISHED`** |
| BAF | **No established prior exposure** (not independently audited) | No | No | No | No | No | No | **No — `BLOCKED` (not acquired; rejected for the unit of record)** | **`NOT ESTABLISHED`** |
| Zenodo 2026 | No | No | No | No | No | No | No | **No** — recorded as not eligible as unbiased ground truth | **`NOT ESTABLISHED`** |
| FreeFraudDetection50M | No | No | No | No | No | No | No | **No** — could not be located | **`NOT ESTABLISHED`** |
| Dal Pozzolo ~49.86M | No | No | No | No | No | No | No | **No** — confidential, never released | **`NOT ESTABLISHED`** |
| Worldline | No | No | No | No | No | No | No | **No** — institutional access required | **`NOT ESTABLISHED`** |
| Novatti | No | No | No | No | No | No | No | **No** — confidential | **`NOT ESTABLISHED`** |
| Elliptic / Elliptic++ | No | No | No | No | No | No | No | **No** — wrong domain for the native contract | **`NOT ESTABLISHED`** |

**No independence is upgraded. No experiment was run to establish this table.**

---

## 10. Reviewer input contract (mapped to existing fields)

**No new reviewer schema is created.** Every required input maps to a field that
already exists in `docs/evaluation/REVIEWER_ASSIGNMENT_RECORD.md` §4 and
`docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` §8.

### 10.1 Role vocabulary — and why one person cannot satisfy both

| Term | Where it is authoritative | Meaning here | Current state |
|---|---|---|---|
| `STATISTICAL REVIEW` | resolution §3 ownership table | 8 decisions (`/01 /02 /08 /09 /10 /11 /13 /14`) requiring the statistical component only | all `NOT APPROVED` |
| **`BOTH`** | resolution §3 ownership table | 6 decisions (`/03 /04 /05 /06 /07 /12`) requiring **two independent** components: statistical **and** domain | all `NOT APPROVED` |
| `NOT ASSIGNED` | assignment record §1.1/§1.2 | no reviewer identity exists for either role | both roles |
| `PENDING REVIEW` | resolution §8.2 | no ruling yet | all 14 |
| `APPROVED` / `REJECTED` / `REVISE` | resolution §8.2; plan §22/§24 | a ruled decision; `APPROVED` requires name + date + rationale + timestamp | 0 of 14 |
| `NOT APPROVED` | resolution §8.1 fill block | the approval field for an unresolved decision | all 14 |

**Mechanically enforced boundaries** (`review_resolution_check.py` R5–R8, R12):
a statistical reviewer **cannot** satisfy domain review and a domain reviewer
**cannot** satisfy statistical review — the two components are recorded under
different field names and must be completed independently; a `BOTH` decision cannot
be `APPROVED` unless both components are complete; a `STATISTICAL REVIEW` sheet that
acquires a domain block is rejected. **No reviewer identity is invented anywhere in
this packet** (assignment record §8).

### 10.2 Required inputs and where they go (nothing new is defined)

| Required reviewer input | Existing field | Current value | Supplied by |
|---|---|---|---|
| Reviewer identity (per role) | assignment record §1.1/§1.2 `Assigned individual:`; §3.1/§3.2 `Identity` | `NOT ASSIGNED` / `NOT ESTABLISHED` | the real person |
| Qualification / affiliation | §3.1/§3.2 `Relevant expertise`, §1.1 `Affiliation:` | `NOT ESTABLISHED` | the real person |
| Conflict-of-interest declaration | §1.1/§1.2 `Conflict-of-interest declaration:` | `NOT ESTABLISHED` | the real person |
| Assignment date | §1.1 `Assignment date:` | `NOT ESTABLISHED` | the real person |
| Scope accepted | §3.1/§3.2 `Scope accepted` | `NOT ESTABLISHED` | the real person |
| Per-decision ruling | resolution §8.1 block: `Reviewer Decision` | `PENDING REVIEW` (14/14) | statistical reviewer |
| Per-decision rationale | resolution §8.1 `Rationale` | `PENDING REVIEW` | statistical reviewer (mandatory before `APPROVED`) |
| Per-decision evidence reviewed | resolution §8.1 `Evidence Reviewed` | `PENDING REVIEW` | statistical reviewer |
| Per-decision approval + timestamp | resolution §8.1 `Approval`, `Approval Timestamp` | `NOT APPROVED` / `NOT ESTABLISHED` | statistical reviewer |
| Domain ruling, rationale, evidence, approval (6 `BOTH` decisions only) | resolution §8.1 prefixed `Domain Reviewer …` block | `PENDING REVIEW` / `NOT APPROVED` | domain reviewer |
| Licence/terms determination | `DATASET_LICENCE_EVIDENCE_CHECKLIST.md` §3 blocks; §6 of this packet | `NOT ESTABLISHED` per dataset | licence/terms reviewer (data-governance) |
| Dataset/provenance audit findings | §7 checklist A–O of this packet; ledger §7 (U-7) | `REQUIRES REVIEW` | independent dataset/provenance auditor |
| Independent audit of the exposure ledger | ledger §7 row "Independent audit of this exposure ledger" | `NOT ESTABLISHED` | independent auditor |

**Read-only readiness reporting** (never writes, never advances a decision):
`python backend/scripts/review_resolution_check.py --readiness` → currently
`0/14 decisions resolved | component states: NOT ASSIGNED=20`.

---

## 11. Freeze readiness matrix

The Research Plan is **not** changed by this packet. Requirement rows are the freeze
requirements relevant to this packet's scope (the plan's own §35 carries **36**
checklist items, all unchecked; the assignment record §7 records how they compose
into the 78 findings).

| Requirement | Current evidence | Reviewer required? | Current state | Blocks freeze? |
|---|---|---|---|---|
| Dataset provenance | coverage §5; reconciliation §3–§5; this packet §4 | **Yes** (independent audit) | file identity `ESTABLISHED`; publisher provenance for IBM v2 `NOT ESTABLISHED` (**U-4**) | **Yes** |
| Dataset licence | licence checklist; this packet §6 | **Yes** | `PENDING REVIEW` / `NOT ESTABLISHED` for all; §35 checkbox unchecked | **Yes** |
| Dataset eligibility | manifest §1–§9 | **Yes** | no dataset eligible; IEEE-CIS/BAF `BLOCKED`; exposed sets `EXPLORATORY` | **Yes** |
| Exposure ledger | ledger (amended, `31af86fd…`) | **Yes** (independent audit, plan §14) | present; audit `NOT ESTABLISHED` (**U-7**) | **Yes** |
| Independent audit | — | **Yes** | **has not occurred** | **Yes** |
| Statistical decisions `/01`–`/14` | resolution §10 | **Yes** | **0/14** | **Yes** (11 of 14 rows block) |
| Domain decisions (`BOTH` rows) | resolution §3/§10 | **Yes** | **0/6** domain components | **Yes** |
| Plan placeholders | `check_freeze.py` | **Yes** | 36 §35 checklist items unchecked; 65 unresolved placeholders; 56 `TO BE FROZEN` occurrences | **Yes** |
| Preregistration placeholders | `check_freeze.py`; protocol | **Yes** | 12 markers (`0.1.2-draft`) | **Yes** |
| Freeze record | filesystem; plan §29/§34 | freeze-time act | `docs/FREEZE_RECORD.json` **absent** | **Yes** |
| Reviewer assignments | assignment record §1 | **Yes** | both `NOT ASSIGNED` | **Yes** |
| Statistical reviewer approval | assignment record §1.1/§3.1 | **Yes** | `NOT ESTABLISHED`; plan §32 forbids self-approval | **Yes** |
| Domain reviewer approval | assignment record §1.2/§3.2 | **Yes** | `NOT ESTABLISHED` | **Yes** |
| Automated freeze check (§30) | `scripts/check_freeze.py`; `.github/workflows/freeze-check.yml` | no (mechanical) | **exit 1 / 78 findings** — red **by design** pre-freeze | **Yes** |
| Git freeze tag (§29) | git | freeze-time act | **0 tags**; artifacts to freeze are **untracked** (O-4) | **Yes** |

**Freeze legality: NOT LEGAL.** 15/15 rows unresolved; no row can be closed by this
packet, by the project author, or by an automated checker.

---

## 12. 50M status

> **50M dataset generation = NOT AUTHORIZED / NOT YET SCIENTIFICALLY JUSTIFIED.**

| Fact | Value |
|---|---|
| Current real native-complete corpus | **24,386,900** rows (IBM v2) — approximately **24.39M** |
| Distinct real volume after removing duplicates | **24,386,900** — the User0 extract and the `ealtman2019` copies add **0** |
| Rows padded, duplicated, or synthesized to reach a target | **0** |
| Is the existing corpus called "50M" anywhere in this packet? | **No** |
| Has any 50M dataset been generated? | **No** |
| Would forcing 50M be real transactions? | **No** — it would be 24.4M real + ≥25.6M generated, and would inherit IBM v2's `EXPLORATORY` exposure | coverage §11.3 |

**The construction question remains downstream of, in order:** (1) reviewer
decisions on `/01`–`/14`; (2) freeze; (3) confirmatory evidence; (4) an approved and
documented construction methodology. Until all four exist, 50M work is **not
authorized and not scientifically justified**. No target is adopted by this packet,
and the 50M target does not appear in the Research Plan or in any eligibility
criterion.

---

## 13. External / Independent Dependencies

Each row is a **human or institutional dependency the repository cannot manufacture**.
None may be satisfied by the project author, by an AI agent, or by an automated
checker (plan §32; assignment record §8).

| # | Dependency | What must be supplied | Why the repository cannot manufacture it | Current state | What can proceed once supplied |
|---|---|---|---|---|---|
| **D-1** | **Qualified statistical reviewer** | identity, affiliation, qualification, conflict-of-interest declaration, assignment date, scope accepted, signature/approval reference — plus a ruling + rationale + approval per assigned decision (`/01–/14`, 8 statistical-only + 6 `BOTH` statistical halves) | A checker can verify consistency but cannot rule on baseline identity, ensemble identity, power, dependence-aware inference, calibration, sampling, seeds, or gating; plan §32 forbids self-approval | **`NOT ASSIGNED`** (0/14) | statistical rulings on the assigned decisions → plan/preregistration cells may be edited as separate recorded changes (plan §17) |
| **D-2** | **Qualified domain reviewer** | same role-level fields; plus domain rulings on the 6 `BOTH` decisions (`/03 /04 /05 /06 /07 /12`): cost/abstention semantics, alert-volume interpretation, segment semantics, domain assumptions | Fraud-operations semantics and analyst-capacity assumptions are not derivable from code; preregistration §7 forbids presenting an assumed capacity as measured analyst burden | **`NOT ASSIGNED`** (0/6 domain components) | domain half of each `BOTH` decision; only then can those rows be `APPROVED` |
| **D-3** | **Independent dataset / provenance auditor** | independent verification of dataset identity, hashes, row counts, schemas, label semantics, temporal semantics, leakage, exposure, duplicates, provenance and the amended records (§7 A–O; §8) | An audit is by definition performed by someone other than the project; the ledger's own §7 records its audit as `NOT ESTABLISHED`, and plan §14 requires it *before* the confirmatory freeze | **not assigned; audit has not occurred** (U-7) | the amended manifest/ledger may be treated as audited inputs; U-7 closes; U-4 (publisher provenance) can be adjudicated |
| **D-4** | **Licence / terms determination** (where repository evidence is insufficient) | for each acquired dataset: exact terms URL, verbatim operative clause, named licensor, access route, retrieval date, reviewer identity, and the restriction's effect on this project (outcome F-P / F-R / R / U) | No terms document is archived for any dataset; the only in-repo claims were unsupported and retracted (C-5, C-7); a legal/terms reading requires a person empowered to make it | **`PENDING REVIEW` / `NOT ESTABLISHED`** for all datasets (U-1) | plan §35's "licence/access verified" cell for datasets whose outcome is F-P/F-R and whose restrictions the reviewer records as compatible |
| **D-5** | **Institution-controlled data / evaluation environment** (Track I / Track N) | authorised, eligible institutional data with an access agreement; the matching evaluation environment | The repository holds no institutional access; Worldline/Novatti-style corpora are confidential and require an agreement; plan §33 forbids claiming validation before such data exist | **`BLOCKED` / `NOT ESTABLISHED`** | Track I/N work only; **not** a precondition for Track M review |

**No substitution is permitted:** the project author is not a reviewer, an AI agent is
not a reviewer, and a passing checker is not an independent audit. **This packet
therefore ends at the point where human input begins.**

---

## 14. What this packet does not claim

- that an independent audit has occurred, or that any item in §7/§8 has been verified;
- that any dataset licence is cleared, or that plan §35's licence cell is checked;
- that any dataset is eligible, or that any exposure/independence status changed;
- that a statistical or domain reviewer exists, has been approached, or has signed;
- that any of `/01`–`/14` has been reviewed, ruled, or approved;
- that the Research Plan, the preregistration, or any artifact is frozen;
- that IEEE-CIS or BAF is acquired, eligible, or verified (`BLOCKED`);
- that NR-05 is confirmatory, or that any metric here is independent evidence;
- that a 50M dataset exists, is authorized, or is justified;
- that this packet is a governance framework, a registry, or a new approval state.

## 15. Change history

| Version | Change | Type |
|---|---|---|
| `1.0-draft` | Initial review package. §1–§13 prepared from existing records; eight observations (O-1…O-8) recorded as reviewer-owned; §7 A–O and §8 items 1–11 left at `REQUIRES REVIEW`; mechanical companion `backend/scripts/review_package_check.py` added. No decision, licence, eligibility, exposure, or freeze state changed. | PRE-REVIEW PREPARATION |
| `1.1-draft` | Post-engagement refresh: cited `REVIEWER_ASSIGNMENT_RECORD.md` hash updated (pre-edit `52a32bde…` → `b62e4f05…`, §11 dependency statement added; no field populated), the engagement brief added to the source table and named as the external entry point, the final battery row (§2.1) and ledger count (§7 item O) updated with measured results. No claim above changed; no state changed. | REVIEWER ENGAGEMENT (documents only) |

**Limitations of this draft:** every §7/§8 item is unverified by definition; the
licence worksheet is unfilled; the reviewer roles are unassigned; `U-1`…`U-10` remain
open; and the observations O-1…O-8 are recorded, not resolved.

> # REVIEW PACKAGE READY / PASS WITH LIMITATIONS

*End of `independent-evidence-review-package 1.0-draft`. This packet grants no
eligibility, asserts no licence, assigns no reviewer, approves no decision, and
freezes nothing.*
