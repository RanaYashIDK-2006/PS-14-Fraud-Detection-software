# Dataset Discovery, Provenance & 48-Feature Coverage

**Artifact path:** `docs/evaluation/DATASET_48_FEATURE_COVERAGE.md`
**Version:** `dataset-48-coverage 1.0-draft`
**Status:** DRAFT — dataset-discovery and feature-contract analysis artifact.
**NOT frozen, NOT independently audited, NOT approved.**
**Authored at git state:** `5a5ff55cc03df73318fdf31a16e3665f159a4720` (2026-10-04).
**Companion artifacts (referenced, NOT modified):**
`docs/evaluation/DATASET_EXPOSURE_LEDGER.md`,
`docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md`,
`docs/evaluation/REVIEWER_ASSIGNMENT_RECORD.md`,
`docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md`.

**Supporting artifact:** `docs/evaluation/DATASET_DISCOVERY_REGISTER.md` was
**NOT created**. Every candidate that would have gone in it is recorded with full
provenance in §5 of this document. A second registry would duplicate §5 and add a
governance surface the project does not need (see §15.4).

**Self-hash:** a document cannot carry its own hash (plan §29). This artifact's
SHA-256 is to be recorded in `docs/FREEZE_RECORD.json` **at freeze**, not here.

**No model, plan, or governance state is changed by this document.** No training,
no threshold change, no feature redefinition, no reviewer assignment, no decision
resolution, no Track M, no freeze.

---

## 1. Objective

Establish, from repository evidence first and external sources second:

1. which datasets PS-14 actually **trained** on, **evaluated** on, used only
   **exploratorily**, was **exposed** to, and has never touched;
2. the authoritative **native 48-feature contract** as it exists in executable code,
   feature by feature, with type, semantics, source, transformation, temporal and
   entity requirements, leakage risk, and derivability;
3. which **additional public datasets** could legitimately contribute missing
   native information, with provenance and licence/access status;
4. a **per-feature × per-dataset coverage matrix**, every cell backed by evidence;
5. whether a defensible **~50,000,000-row** native-contract dataset is achievable.

**Out of scope for this phase (explicitly not done):** generating any dataset,
retraining, changing thresholds or feature definitions, editing the Research Plan,
assigning reviewers, resolving reviewer decisions, or claiming native validation or
real-world effectiveness.

**Method note.** Where a README, a manifest cell, or a narrative document and an
executable artifact disagree, the executable artifact wins and the disagreement is
recorded as a numbered discrepancy in §14.1 rather than silently reconciled. Five
such discrepancies were found; two of them (D1, D5) change how the coverage matrix
must be read.

---

## 2. Repository-Derived Native 48-Feature Contract

### 2.1 Where the contract lives (executable, not narrative)

The 48-feature contract is defined in **two independent Python modules** with an
identical, order-sensitive list:

| Source | Symbol | Count |
|---|---|---|
| `backend/src/privacy_layer/native_features.py` | `ALTMAN_NATIVE_FEATURES` | 48 |
| `backend/src/risk_engine/altman_native_ensemble.py` | `ALTMAN_NATIVE_FEATURES` | 48 |

Verified by execution: both lists are **length 48**, contain **no duplicates**, and
compare **element-wise equal** (`A == B` → `True`).

The single shared derivation function is
`derive_native_features(raw, vel, rates)` in
`backend/src/privacy_layer/native_features.py`; the ordered-vector function is
`native_vector(features)`. Its module docstring states the design invariant:

> "Leakage-safe by construction: every historical input is shifted (the event being
> scored is never counted in its own velocity / fraud rate / average)."

The canonical runtime transformation between the 21-feature public contract and the
48-feature native vector is **`map_raw_to_native`** (`CANONICAL_TRANSFORMATION =
"map_raw_to_native"`, `backend/src/monitoring/external_dataset_contract.py:54`).

The machine-readable prose contract (`_native_entries()`,
`backend/scripts/build_feature_contract.py`) documents name, type, units, formula,
source keys, causal rule and missing-value behaviour for each of the same 48
features. That file and the two modules are the authority for this document.

### 2.2 Raw source fields the contract consumes

From `derive_native_features`, the 48 features are a pure function of these raw
event columns plus shifted history:

| Raw field | Used by |
|---|---|
| `amount` | `amt`, `log_amt`, `amt_sq`, `amt_vs_user_avg`, `amt_zscore`, `high_amt`, `very_high_amt`, `amt_x_*` |
| `ts` (datetime) | `hr`, `mn`, `dow`, `Month`, `Day`, `hour_sin`, `hour_cos`, `is_night`, `is_business_hours`, `amt_x_hr`, `amt_x_night` |
| `use_chip` | `chip`, `is_online`, `is_swipe`, `is_online_or_no_state`, `amt_x_chip`, `amt_x_online` |
| `mcc` | `mcc`, `mcc_high`, `mcc_restaurant`, `mcc_gas`, `mcc_grocery`, `mcc_travel`, `mcc_online`, `amt_x_mcc` |
| `merchant_state` | `has_state`, `is_online_or_no_state` |
| `zip` | `has_zip` |
| `errors` | `err` |
| `merchant_id` (name) | `merchant_id` |
| `city_id` (merchant city) | `city_id` |
| `card_id` (card) | `card_id` |
| `user_id` | all `user_*` history features |
| `vel` (shifted history) | `user_tx_count`, `card_tx_count`, `user_avg_amt`, `merch_tx_count`, `user_merchant_diversity`, `user_city_diversity`, `user_merch_count` |
| `rates` (shifted label history) | `user_fraud_rate`, `merch_fraud_rate`, `city_fraud_rate` |

**Consequence that governs the whole matrix:** a dataset that does not carry
*amount + absolute timestamp + channel + MCC + merchant identity + merchant city +
card identity + user identity + prior history + labels* cannot populate the native
48. This is a schema requirement, not a modelling preference.

### 2.3 Feature tiers

The 48 features fall into ten tiers by what source information they consume. Tier
membership is used throughout §7 and §9.

| Tier | Features | n | Source requirement |
|---|---|---|---|
| A — amount | `amt`, `log_amt`, `amt_sq` | 3 | raw amount only |
| B — calendar/clock | `hr`, `mn`, `dow`, `Month`, `Day`, `hour_sin`, `hour_cos`, `is_night`, `is_business_hours` | 9 | **absolute** timestamp |
| C — channel & state | `chip`, `is_online`, `is_swipe`, `err`, `has_zip`, `has_state`, `is_online_or_no_state` | 7 | channel + merchant state + zip + errors |
| D — merchant category | `mcc` + 6 MCC band flags | 7 | MCC code |
| E — entity codes | `merchant_id`, `city_id`, `card_id` | 3 | merchant / merchant-city / card identity |
| F — prior velocity & amount history | `user_tx_count`, `card_tx_count`, `user_avg_amt`, `amt_vs_user_avg`, `amt_zscore`, `merch_tx_count`, `user_merchant_diversity`, `user_city_diversity` | 8 | ordered history per entity |
| G — prior label history | `user_fraud_rate`, `merch_fraud_rate`, `city_fraud_rate` | 3 | ordered history **+ labels**, strictly prior-only |
| H — amount thresholds | `high_amt`, `very_high_amt` | 2 | amount + prior user mean |
| I — interactions | `amt_x_hr`, `amt_x_mcc`, `amt_x_chip`, `amt_x_online`, `amt_x_night` | 5 | products of A/B/C/D |
| J — user–merchant interaction count | `user_merch_count` | 1 | user × merchant identity + history |

### 2.4 Authoritative 48-feature inventory

Names, types, semantics, source keys, transformation, temporal rule and leakage
risk are taken from `build_feature_contract.py::_native_entries()` (units/formula
column) and `native_features.py::derive_native_features` (actual implementation).
"Entity req." is the identity the feature needs; "Calc. from history?" is whether it
can be computed from a prior-only history of real transactions.

| # | Feature | Type | Semantics (units / formula) | Source keys | Temporal rule | Entity req. | Leakage risk | Calc. from history? |
|---|---|---|---|---|---|---|---|---|
| 1 | `amt` | float | dollars; raw amount clamped ≥ 0 | `amount` | instant | — | none | n/a (instant) |
| 2 | `log_amt` | float | log dollars; `log1p(amt)` | `amount` | instant | — | none | n/a (instant) |
| 3 | `amt_sq` | float | dollars²; `amt ** 2` | `amount` | instant | — | none | n/a (instant) |
| 4 | `hr` | float | hour [0,24); `ts.hour` | `ts` | instant | — | none | n/a (instant) |
| 5 | `mn` | float | minute [0,60); `ts.minute` | `ts` | instant | — | none | n/a (instant) |
| 6 | `dow` | float | weekday [0,6); `ts.weekday()` | `ts` | instant | — | none | n/a (instant) |
| 7 | `Month` | float | month [1,12] | `ts` | instant | — | none | n/a (instant) |
| 8 | `Day` | float | day-of-month [1,31] | `ts` | instant | — | none | n/a (instant) |
| 9 | `hour_sin` | float | `sin(2π·hr/24)` | `ts` | instant | — | none | n/a (instant) |
| 10 | `hour_cos` | float | `cos(2π·hr/24)` | `ts` | instant | — | none | n/a (instant) |
| 11 | `is_night` | int {0,1} | `1 if hr < 6 or hr > 22` | `ts` | instant | — | none | n/a (instant) |
| 12 | `is_business_hours` | int {0,1} | `1 if 9 ≤ hr ≤ 17` | `ts` | instant | — | none | n/a (instant) |
| 13 | `chip` | int {0,1} | `use_chip == "Chip Transaction"` | `use_chip` | instant | — | none | n/a (instant) |
| 14 | `is_online` | int {0,1} | `use_chip == "Online Transaction"` | `use_chip` | instant | — | none | n/a (instant) |
| 15 | `is_swipe` | int {0,1} | `use_chip == "Swipe Transaction"` | `use_chip` | instant | — | none | n/a (instant) |
| 16 | `err` | int {0,1} | `errors` field non-empty | `errors` | instant | — | none | n/a (instant) |
| 17 | `has_zip` | int {0,1} | `zip` field non-empty | `zip` | instant | — | none | n/a (instant) |
| 18 | `has_state` | int {0,1} | `merchant_state` non-empty | `merchant_state` | instant | — | none | n/a (instant) |
| 19 | `is_online_or_no_state` | int {0,1} | `is_online OR has_state == 0` | `use_chip`, `merchant_state` | instant | — | none | n/a (instant) |
| 20 | `mcc` | float | raw MCC code | `mcc` | instant | — | none | n/a (instant) |
| 21 | `mcc_high` | int {0,1} | `mcc ≥ 5000` | `mcc` | instant | — | none | n/a (instant) |
| 22 | `mcc_restaurant` | int {0,1} | `5812 ≤ mcc ≤ 5814` | `mcc` | instant | — | none | n/a (instant) |
| 23 | `mcc_gas` | int {0,1} | `5541 ≤ mcc ≤ 5542` | `mcc` | instant | — | none | n/a (instant) |
| 24 | `mcc_grocery` | int {0,1} | `5411 ≤ mcc ≤ 5422` | `mcc` | instant | — | none | n/a (instant) |
| 25 | `mcc_travel` | int {0,1} | `3000 ≤ mcc ≤ 3350` | `mcc` | instant | — | none | n/a (instant) |
| 26 | `mcc_online` | int {0,1} | `5967 ≤ mcc ≤ 5969` | `mcc` | instant | — | none | n/a (instant) |
| 27 | `merchant_id` | float | stable code [0,100000); `sha256(merchant name) mod 100000` | `merchant_id` | instant | merchant | none (stable across split) | n/a (instant) |
| 28 | `city_id` | float | stable code; `sha256(merchant_city) mod 100000` | `city_id` | instant | merchant city | none | n/a (instant) |
| 29 | `card_id` | float | stable code; `sha256(card) mod 100000` | `card_id` | instant | card | none | n/a (instant) |
| 30 | `user_tx_count` | float | count; prior events for `user_id` | shifted velocity | prior-only; event excluded | user | none if shifted | **yes** |
| 31 | `card_tx_count` | float | count; prior events for card | shifted velocity | prior-only | card | none if shifted | **yes** |
| 32 | `user_avg_amt` | float | dollars; prior mean amount for user | shifted velocity | prior-only | user | none if shifted | **yes** |
| 33 | `amt_vs_user_avg` | float | `amt / user_avg_amt` | `amount`, `user_avg_amt` | prior-only mean | user | none if shifted | **yes** |
| 34 | `amt_zscore` | float | `(amt − prior mean)/(prior mean + 1e-6)` | `amount`, `user_avg_amt` | prior-only stats | user | none if shifted | **yes** |
| 35 | `merch_tx_count` | float | count; prior events for merchant | shifted velocity | prior-only | merchant | none if shifted | **yes** |
| 36 | `user_merchant_diversity` | float | count; distinct prior merchants for user | shifted velocity | prior-only, clamped ≥ 1 | user × merchant | none if shifted | **yes** |
| 37 | `user_city_diversity` | float | count; distinct prior cities for user | shifted velocity | prior-only, clamped ≥ 1 | user × city | none if shifted | **yes** |
| 38 | `user_fraud_rate` | float | rate [0,1]; prior fraud rate for user | shifted entity tracker | prior-only; label post-event | user | **label-latency sensitive** | **yes, with labels** |
| 39 | `merch_fraud_rate` | float | rate [0,1]; prior fraud rate for merchant | shifted entity tracker | prior-only; label post-event | merchant | **label-latency sensitive** | **yes, with labels** |
| 40 | `city_fraud_rate` | float | rate [0,1]; prior fraud rate for city | shifted entity tracker | prior-only; label post-event | city | **label-latency sensitive** | **yes, with labels** |
| 41 | `high_amt` | int {0,1} | `amt > 2 × user_avg_amt` | `amount`, `user_avg_amt` | prior-only | user | none if shifted | **yes** |
| 42 | `very_high_amt` | int {0,1} | `amt > 5 × user_avg_amt` | `amount`, `user_avg_amt` | prior-only | user | none if shifted | **yes** |
| 43 | `amt_x_hr` | float | `amt × hr` | `amount`, `ts` | instant | — | none | n/a (instant) |
| 44 | `amt_x_mcc` | float | `amt × mcc` | `amount`, `mcc` | instant | — | none | n/a (instant) |
| 45 | `amt_x_chip` | float | `amt × chip` | `amount`, `use_chip` | instant | — | none | n/a (instant) |
| 46 | `amt_x_online` | float | `amt × is_online` | `amount`, `use_chip` | instant | — | none | n/a (instant) |
| 47 | `amt_x_night` | float | `amt × is_night` | `amount`, `ts` | instant | — | none | n/a (instant) |
| 48 | `user_merch_count` | float | count; prior merchant count for user | shifted velocity | prior-only | user × merchant | none if shifted | **yes** |

**Required vs optional.** The contract exposes no per-feature optionality flag: the
deployed manifest asserts all 48 (`models/production/manifest.json` →
`"n_features": 48`, `feature_schema_version: "altman_native_v2"`), and
`build_feature_contract.py` refuses to build a contract if
`feature_list.json` disagrees with the resolved schema. All 48 are therefore
**required** for any native-contract dataset. Missing-value behaviour exists
(cold-start defaults: `0.0`, `12.0`, `0.001` for fraud rates), but a cold-start
default is a **substitution**, not a reason to treat a feature as optional — the
trainer and runtime must agree, and `feature_parity_test.py` enforces that.

**Cold-start constants (from code, unchanged):** `COLD_START_FRAUD_RATE = 0.001`;
`user_merchant_diversity` and `user_city_diversity` clamp to ≥ 1;
`amt_vs_user_avg` → `1.0`; `amt_zscore` → `0.0`; `amt` clamps negatives to `0.0`.

### 2.5 What the native 48 does **not** contain (and why §6 matters)

The native 48 has **no device, recipient, authentication, or graph features**. The
five features named in this phase's brief — `recipient_novelty`,
`shared_device_accounts`, `shared_recipient_accounts`, `failed_auth_count_24h`,
`mule_ring_score` — are **not in the native 48**. Verified by execution against
`ALTMAN_NATIVE_FEATURES`.

They belong to the **21-feature causal contract**, confirmed by execution:

```
src.privacy_layer.features.ML_FEATURES  -> 21 entries
```

containing, among others, `failed_auth_count_24h`, `shared_device_accounts`,
`shared_recipient_accounts`, `mule_ring_score`, `recipient_novelty`,
`new_device_flag`, `unusual_location_flag`, `unusual_recipient_flag`.

They are additionally listed in
`backend/research/public_feature_contract.json` under `missing_from_public`
(count 7: `new_device_flag`, `unusual_location_flag`, `unusual_recipient_flag`,
`shared_device_accounts`, `shared_recipient_accounts`, `mule_ring_score`,
`recipient_novelty`).

**This is discrepancy D5 (§14.1).** The coverage matrix in §7 is the native 48 as
specified. The brief's five named features are analysed separately in §8.5 against
the 21-feature causal contract, because analysing them "within the native 48"
would be analysing features that are not in it.

### 2.6 The three contracts in the repository (do not conflate)

| Contract | Size | Authority | Location | Role |
|---|---|---|---|---|
| Native Altman | **48** | `ALTMAN_NATIVE_FEATURES` (×2) + `derive_native_features` | `src/privacy_layer/native_features.py`, `src/risk_engine/altman_native_ensemble.py` | **the contract this document maps** |
| Causal §16 | 21 | `CAUSAL_FEATURES` / `ML_FEATURES` | `src/privacy_layer/features.py`, `build_feature_contract.py::_causal_entries()` | live ingest vector; device/recipient/auth/graph live here |
| Public `public_v1` | 14 | `backend/research/public_feature_contract.json` | JSON | research surrogate; `failed_auth_proxy` has `available_in: []` by design |
| P20 / phase-23B | 45 | `claims_registry.jsonl` C-010 | `misc/reports/phase23b/kaggle/13_p20_metrics.json` | historical, **degraded**: 33/45 features reconstructed, NaN-filled |

Mapping between them is `map_raw_to_native` (`CANONICAL_TRANSFORMATION`). Note the
documented proxy substitutions on the mapped (non-native) runtime path: `chip` and
`is_online` are both fed by `new_device_flag`; `mcc_n`, `has_zip`, `has_state` are
**constant 0.0** there. Those proxies apply to the mapped runtime only and are
**not** a licence to synthesise the native columns — see §12.
---

## 3. Previously Exposed Datasets

### 3.1 What PS-14 actually trained on

Determined from executable artifacts, not README claims.

`models/production/manifest.json` (model `altman_native_v2_20260904_115703`) records:

| Field | Value |
|---|---|
| `model_type` | `xgb_lgb_cb_native` |
| `n_features` | `48` |
| `feature_schema_version` | `altman_native_v2` |
| `feature_source` | `shared src/privacy_layer/native_features.py — train == production by construction` |
| **`training_dataset`** | **`data/credit_card_transactions-ibm_v2.csv`** |
| `training_dataset_sha256` | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |
| `train_rows` | `193027` |
| `train_seed` | `42` |
| `locked_threshold` | `0.7847116291110687` |
| `target_leakage` | `false` |

Corroborated by the governance record
`models/model_records/altman_native_v2_20260904_115703.json`:

- `training_period` `1995-06-01..2015-12-31`
- `validation_period` `2016-01-01..2017-12-31`
- `final_test_period` `2018-01-01..2020-02-29`
- `threshold_policy` `validation-only: max recall within val FPR<=1%`
- test metrics: AUC `0.972237`, PR-AUC `0.78749`, recall@1%FPR `0.412626`

**Answer: the deployed production model trained on exactly one dataset — the
repository's IBM v2 file — using the native 48 contract.** The trainer is
`backend/scripts/train_altman_native.py`, which parses `Amount`, `Year/Month/Day`,
`Time`, `Use Chip`, `MCC`, `Errors?`, `Zip`, `Merchant State`, `Merchant Name`,
`Merchant City`, `Card`, `User` and builds the 48 features with
`groupby(...).cumcount()` / `expanding().mean().shift(1)` — i.e. the same prior-only
derivations the runtime uses.

### 3.2 Measured identity of the acquired datasets

All figures below were **measured from the files**, not copied from a registry.

| Dataset | Path | Columns | Data rows | Positives | Positive rate | SHA-256 (pinned) |
|---|---|---|---|---|---|---|
| ULB | `data/creditcard.csv` | 31 | **284,807** | 492 | 0.173% | `76274b69…` |
| Kaggle fraudTrain | `data/kaggle_fraud/fraudTrain.csv` | 23 | **1,296,675** | 7,506 | 0.579% | `fd713920…` |
| Kaggle fraudTest | `data/kaggle_fraud/fraudTest.csv` | 23 | **555,719** | 2,145 | 0.386% | `12d553ab…` |
| IBM v2 | `data/credit_card_transactions-ibm_v2.csv` | **15** | **24,386,900** | **29,757** | **0.122%** | `b01fa323…` |
| PS-14 synthetic | `data/transactions.csv` | 27 | generator output | — | — | `9f0f56bf…` |

Additional files in `data/` and their disposition:

| File | Columns | Rows | Disposition |
|---|---|---|---|
| `data/User0_credit_card_transactions.csv` | 15 (identical IBM schema) | 19,963 | **Duplicate extract of IBM v2 — must not be concatenated** (see below) |
| `data/fraud_data.csv` | 30 (`V1..V28, Amount, Class`) | 21,693 | ULB-schema sample; no native fields |
| `data/paysim.csv` | 9 | 100,000 | PaySim simulator output; no native fields |
| `data/paysim_1m.csv` | 9 | 1,200,000 | PaySim simulator output; no native fields |

**Duplicate-extract finding (verified, not assumed).** The first two data rows of
`data/User0_credit_card_transactions.csv` are byte-identical to the first two data
rows of `data/credit_card_transactions-ibm_v2.csv` (`0,0,2002,9,1,06:21,$134.09,
Swipe Transaction,3527213246127876953,La Verne,CA,91750.0,5300,,No` and the 06:42
row). It is a per-user slice of the same source file, so concatenating it would
**double-count real transactions**. It contributes **0** additional rows.

### 3.3 Measured columns — the decisive schema evidence

```
ULB    : "Time","V1".."V28","Amount","Class"
Kaggle : ,trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,
         street,city,state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,
         merch_lat,merch_long,is_fraud
IBM v2 : User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,
         Merchant City,Merchant State,Zip,MCC,Errors?,Is Fraud?
```

**The IBM v2 column set maps onto the native 48 raw-field contract almost
one-for-one.** That is not a coincidence of naming: the 48-feature native contract
was built for exactly this schema. This is the single most important fact in this
document and it drives §7 and §9.

### 3.4 Discrepancy D1 — the eligibility manifest describes a different IBM file

`docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md` §4 records, for IBM v2:

> `feature semantics | 168 raw features; hour_diff = derived (deltas), preprocessing fidelity concerns flagged in NR-01 | ESTABLISHED (concern)`
> `leakage assessment | hour_diff derived during feature extraction (deltas), not point-in-time safe as evaluated (NR-01 §A) | ESTABLISHED (defect)`

**Measured against the acquired file: 15 columns, no `hour_diff` column.** No file
anywhere in the repository has 168 columns. The "168 raw features + `hour_diff`"
description matches a *different* artifact — the Kaggle Base-FraudDetection
`credit_card_transactions.csv` — which is **not present in this repository**.

Consequences, all recorded rather than resolved:

1. The manifest's IBM "feature semantics" and "leakage assessment" cells describe
   an artifact that is **not** the one PS-14 trained on. Against the file actually
   used, those cells are **`NOT ESTABLISHED`**, and the `hour_diff` point-in-time
   defect **cannot apply** to this file (no such column exists).
2. `STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md` §N and §O carry the same `hour_diff`
   item forward; those rows inherit this caveat.
3. The exposure-ledger row for IBM v2 `Rows` = `NOT ESTABLISHED` is now
   **established by measurement**: 24,386,900.

The manifest has **not** been edited. Changing it is not this phase's authority
(§11), and downgrading an `ESTABLISHED` cell is a reviewer decision.

### 3.5 Discrepancy D2 — NR-05's row-count constant

`backend/scripts/nr05_diagnostics.py::load_ibm` hard-codes
`n_file_rows=24_386_899`. Measured: **24,386,900** data rows (a full CSV parse
counting rows with all 15 fields: 24,386,900; rows with a different field count: 0;
`wc -l` = 24,386,901 including the header). The constant is **off by one**. It is a
display/annotation constant only — it does not enter any feature computation — but
it is recorded here so no later document cites 24,386,899 as measured.

### 3.6 What was exploratory vs exposed vs untouched

| Dataset | Trained on | Evaluated on | Exposure state | Source |
|---|---|---|---|---|
| IBM v2 | **yes** (production, 48 native) | yes (in-domain, chronological split) | **EXPLORATORY** | manifest §4; ledger §3 |
| ULB | no | yes | **EXPLORATORY** | ledger §3 |
| Kaggle fraudTrain / fraudTest | no | yes (transfer; `FAILED / NON-CONFORMING`) | **EXPLORATORY** | ledger §3, §5 |
| PS-14 synthetic (`data/transactions.csv`) | yes (historical development runs) | yes | used throughout development | manifest §7 |
| IEEE-CIS | **no** | **no** | `CANDIDATE CONFIRMATORY` pending eligibility; **BLOCKED — not acquired** | manifest §5 |
| BAF | **no** | **no** | `CANDIDATE CONFIRMATORY` pending eligibility; **BLOCKED — not acquired** | manifest §6 |
| `User0_…csv`, `fraud_data.csv`, `paysim*.csv` | no | no | incidental working files; no eligibility record | measured |

**Untouched by this project:** IEEE-CIS, BAF, and every candidate in §4. The plan's
§14 records "no established prior model exposure" for IEEE-CIS and BAF; no
independent audit has confirmed that (ledger §7), and this document does not upgrade
it.

---

## 4. Newly Discovered Datasets

Searches covered transaction, behavioural, device, security/authentication,
geographic, relationship/network and large-scale categories. Findings are grouped
by what they could contribute. **No dataset was acquired; no dataset was
downloaded; nothing was merged.**

### 4.1 The historical ~49,858,600-transaction dataset

| Question | Answer | Evidence |
|---|---|---|
| Exact source | **Dal Pozzolo, Caelen, Le Borgne, Waterschoot, Bontempi — "Learned lessons in credit card fraud detection from a practitioner perspective", *Expert Systems with Applications* 41(10):4915–4928, 2014** | Paper abstract/record; the 13-month window (January 2006 – January 2007) and ≈50M ("49,858,600") transactions on ≈1M cards match the description exactly |
| Raw data obtainable? | **No.** The paper's own framing is the scarcity of public transaction datasets; it is a practitioner study on a bank-processor stream, not a released corpus | Paper text ("The scarcity of public available dataset in credit card transactions gives little chance to the community to test and assess…") |
| Licence / terms | **Confidential — proprietary institutional data.** Not published under any open licence | Paper; access is via the authors' institution, not a download |
| Schema | Partially described (transaction amount, card, timestamp, post-hoc label), **not** published in full | Paper only |
| Identifiers | Card identity reported; user/recipient/device identity **not** reported | Paper only |
| Timestamps | Present in the study (it uses temporal splits) | Paper |
| Fraud labels | Post-hoc fraud determination by the processor | Paper |
| Feature compatibility | **Unknown and unverifiable without access**; no `mcc`, `use_chip`, merchant-name/city, zip or errors fields are documented | — |
| Reproducibility | **Not reproducible by this project** | — |
| Permissible here? | **No.** Confidential data, no licence, no acquisition path | — |

**Conclusion: the ~49.86M / ~1.17M-card / 13-month dataset is a *published
description*, not an obtainable dataset.** Per the brief's instruction, it is
**not** treated as available. All 48 native features are classified `UNAVAILABLE`
from it (§7).

**Note on the 1.17M figure.** The public paper describes ≈1M cards; the "1.17
million" figure is not confirmed by any located primary source. Both the 49,858,600
count and the ~1M card count are from the paper's abstract text; the 1.17M variant
is recorded here as **not independently confirmed**.

**Distinguish this from the 24.4M file PS-14 already holds.** The repository's IBM
v2 file is a *different artifact* (15 columns, 24,386,900 rows, 1995–2020). The
Dal Pozzolo dataset is 13 months in 2006–2007 and confidential. They are not the
same data and must never be conflated.

### 4.2 FreeFraudDetection50M

| Question | Answer | Evidence |
|---|---|---|
| Claimed size / fields | 50,000,000 rows, 19 fields, fields including transaction id, timestamp, user id, amount, currency, merchant id, merchant category, card type, IP, device id, country, shipping/billing ZIP, international flag, account age, previous transaction amount, fraud label/type | As described in the phase brief |
| Claimed licence | CC BY-NC 4.0 | As described in the phase brief |
| **Could the dataset be located?** | **No.** Targeted search of the HuggingFace dataset API (`search=fraud` full listing, `search=FraudDetection50M`, `search=50m fraud`, `search=50M`, `search=transaction fraud 50 million`), web search on the exact name, and the announced publisher's HuggingFace organisation returned **no such dataset** | HF API queries executed; `Zia-Data-Labs` organisation enumerated (8 datasets, all security/telemetry, **none** a 50M fraud corpus); the Kaggle announcement exists but the dataset it advertises is not retrievable |
| Classification | **`SYNTHETIC`** (as instructed) **and** existence **`NOT ESTABLISHED`** | — |

**Conclusion: treated as `SYNTHETIC_ONLY` for every native feature, and its
availability is `NOT ESTABLISHED`.** If it were acquired it would remain
**synthetic** and could never serve as independent real-world evidence (§10). Two
independent reasons to exclude it from real coverage: it is synthetic, and it could
not be verified to exist.

### 4.3 IEEE-CIS (Kaggle / Vesta)

Handled **preserving the existing PS-14 finding.** The repository already contains a
complete, executable 48-feature forensic audit:
`backend/src/monitoring/ieee_cis_forensic_audit.py::_build_derivability_matrix()`,
returning **48 entries whose names match `ALTMAN_NATIVE_FEATURES` exactly**
(asserted at generation time). Its classification counts, extracted by execution:

| Repository class | Count | Mapped to this document |
|---|---|---|
| `direct` | **1** | `OBSERVED` |
| `deterministic_derivation` | **2** | `DERIVABLE` |
| `not_derivable` | **41** | `UNAVAILABLE` |
| `unknown` | **1** | `UNKNOWN` |
| `leakage_risk` | **3** | `PARTIALLY_DERIVABLE` |

**This preserves — and quantifies — the existing PS-14 finding that IEEE-CIS does
not satisfy the complete native 48 contract: only 3 of 48 are obtainable, and 41 of
48 are not derivable at all.**

Root causes recorded in the repo's own audit:

- `TransactionDT` is a **timedelta from an undisclosed reference**, not an absolute
  timestamp → tier B (9 features) is unattainable; every feature built on `hr` fails
  with it.
- **No `user_id`** → `user_tx_count`, `user_avg_amt` and every `user_*` feature fail.
- **No `merchant_id`, no merchant city** → `merchant_id`, `city_id`, `merch_tx_count`,
  and the diversity features fail.
- **No `mcc`, no `use_chip`, no `zip`, no `errors`, no merchant state** → tiers C
  and D fail entirely.
- `isFraud` **is** present, so the three fraud-rate features are obtainable only by
  consuming label history → `leakage_risk`, mapped here to `PARTIALLY_DERIVABLE`.
- `card1..card6` are anonymised with a different schema → `card_id` is `UNKNOWN`.

Independently corroborated: IEEE DataPort's own description states "The
TransactionDT feature is a timedelta from a given reference datetime (not an actual
timestamp)."

**Status unchanged:** `BLOCKED — not acquired`. `data/external/` does not exist.
This document does not acquire it and does not change its eligibility.

### 4.4 BAF (Bank Account Fraud, NeurIPS 2022)

Assessed **separately**, and **not** merged into a transaction table.

| Property | Value | Evidence |
|---|---|---|
| Source | Feedzai / "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation", NeurIPS 2022 Datasets & Benchmarks | `github.com/feedzai/bank-account-fraud`; paper record |
| Publisher | Feedzai (NeurIPS 2022) | same |
| Composition | **6 synthetic variants**, generated by state-of-the-art tabular generation applied to an anonymised **real-world bank account *opening* fraud** dataset | repo README + paper abstract |
| Unit of record | **Bank account *opening* application — not a transaction** | paper; `fraud_bool`, income/expense fields |
| Rows | Order 10⁴–10⁵ per variant (the suite is sampled from a larger base) | paper |
| Amount field | Income/expense attributes of an application, **not a transaction amount** | paper |
| Timestamp | `month` granularity only | public description recorded in manifest §6 |
| Entity identifiers | **None** documented publicly | manifest §6 (`NOT ESTABLISHED`) |
| Licence | Public download via Kaggle | repo README |
| Real/synthetic | **Synthetic** (generation applied to an anonymised real base) | paper abstract |
| Status | `BLOCKED — not acquired` (`data/external_benchmark/` absent; preregistered E1 `BLOCKED: DATASET UNAVAILABLE`; harness rc=4) | manifest §6 |

**Conclusion: `REJECTED` for the native 48 across the board.** Not because the data
are bad, but because **the unit of record is wrong**: there is no transaction, no
transaction amount, no transaction timestamp, no merchant, and no channel. Appending
BAF rows to a transaction table to raise a row count would be exactly the fake
row-level merge §13 forbids. Row count gained this way would be fictitious scale.

### 4.5 The 2026 Zenodo deployment-derived online-banking dataset

**Investigated, with the publication's own limitations treated as first-class
evidence.**

| §9 field | Value | Source |
|---|---|---|
| Name | A deployment-derived online banking fraud detection inference-log dataset from a live cloud-based deep learning system | Data in Brief, 2026;68:113153 |
| Version | Zenodo v5 (published 2026-05-04); concept DOI `10.5281/zenodo.20030064` | Zenodo record |
| Source URL | `https://zenodo.org/records/20359708` | Zenodo record |
| Publisher | Zenodo; authors Sulaimani Polytechnic University (Fatah, H.H.; Rashid, Z.N.) | Zenodo record |
| Licence | **CC BY 4.0**, open access, no embargo | Zenodo record |
| Access / acquisition | Open access — **but NOT acquired by this project** | — |
| SHA-256 | **NOT ESTABLISHED** (not acquired). Zenodo publishes MD5 only: `856540b34eb427697c824d19b255236d` for the 21.2 MB CSV | Zenodo record |
| Rows × columns | **56,962 × 38** — 30 input features + 8 output/metadata fields | Specifications Table |
| Input features | `time_value` (seconds from first txn), `V1`–`V28` (PCA), `amount` (USD) | Table 1 |
| Additional fields | `transaction_id`, `is_fraud`, `fraud_probability`, `risk_level`, `confidence`, `recommendation`, `response_time_ms`, `timestamp` (UTC), `ip_address` (last octet zeroed) | Table 1 |
| Time range | **31 days, 01–31 January 2026** (absolute UTC `timestamp` present) | abstract + §4.2 |
| Entity identifiers | **None** for card/account/user. `transaction_id` is a record key only ("not a modelling feature") | Table 1 |
| Labels | `is_fraud`; **98 confirmed fraud cases, 0.172%** | abstract; Table 2a |
| Label generation | **Only `BLOCK`-flagged rows (`fraud_probability > 0.43`) were manually reviewed; all `ALLOW` rows were auto-assigned `is_fraud = 0` without review** | §4.2 |
| Provenance | **Proof-of-concept public demonstration API, not a licensed bank**; rows are public API submissions during testing | §4.2, "Note on data source location" |

**First-class limitations carried from the publication (quoted, not paraphrased
away):**

1. *"The deployment was a proof-of-concept system operating as a public
   demonstration interface and was not a licensed banking institution."*
2. *"This procedure introduces verification bias: any fraud case missed by the model
   threshold is permanently recorded as legitimate."*
3. *"fraud_probability is the same model output that determined BLOCK/ALLOW labels,
   creating a circular dependency between model outputs and is_fraud labels for
   BLOCK-flagged records."*
4. The authors state the dataset *"is primarily intended for benchmarking,
   simulation, and evaluation of fraud detection systems rather than unbiased
   supervised model training"*, and recommend excluding the model-output columns to
   avoid leakage.
5. The authors note the 12 known false negatives sit at `fraud_probability` 0.30–0.42
   and that *"the complete population of ALLOW transactions below the MEDIUM-risk
   threshold (fraud_probability below 0.30) was not audited; additional false
   negatives in that range cannot be excluded."*

**Assessment for this project's purposes.** Its one genuine advantage over ULB is an
**absolute UTC `timestamp`**, which makes tier B (9 features) derivable where ULB's
offset-only `Time` makes them impossible. Everything else is ULB-shaped: `V1–V28`
are PCA components whose original attributes are *"not disclosed"* and, per the
authors, **cannot be independently recovered**. There are no entity identifiers, no
MCC, no channel, no merchant.

**Conclusion:** `1 OBSERVED + 13 DERIVABLE = 14 of 48`, and `34 UNAVAILABLE`. As
independent evidence it is **not usable** — the label procedure is model-dependent
and biased toward the very model that generated the scores, so it cannot serve as
unbiased ground truth for evaluating a fraud model. Its realistic role is a
**recent-window behavioural/logistics benchmark**, not a confirmatory dataset, and
not a source of native-contract coverage.

### 4.6 Other candidates examined (recorded; none changes the picture)

| Candidate | What it is | Why it does not close a native gap |
|---|---|---|
| **Elliptic / Elliptic++** (Bitcoin) | 203,769 node transactions + 234,355 edges, 2% illicit | Only source with a real **entity graph** — but crypto; no card/merchant/MCC/channel schema. Relevant to §9C as the only public precedent that graph features require a graph |
| **TSAI-MetaFraud** (arXiv 2607.09528) | Multimodal metaverse benchmark: behaviour + transactions + graph + bot labels, released on GitHub | Synthetic **metaverse** simulator; virtual economies, not banking; no native fields |
| **CiferAI fraud datasets** (HF) | Fully synthetic, privacy-oriented | Synthetic; `SYNTHETIC_ONLY` by definition |
| **Mendeley "Profile-Driven" 15M credit-card dataset** | Large-scale **synthetic**, 15M transactions | Synthetic; no verified native schema |
| **Kaggle `aryan208` 5M financial transactions** | 5M **synthetically generated** | Synthetic |
| **Worldline Belgium** (`INST-WORLDLINE-001`) | ~60M real e-commerce rows, 2017 | `AccessStatus.CONFIDENTIAL`, `AcquisitionStatus.ACCESS_REQUEST_REQUIRED` — repo already records it as blocked |
| **Novatti** (`INST-NOVATTI-001`) | 126,184 real merchant transactions, 2023–2025 | Confidential; customer **and card identifiers removed**; far too small for power |
| **FiFAR** (Springer/figshare, 2025) | 30K bank-account-opening applications + 50 synthetic judges | Same wrong unit of record as BAF |
| **Authentication / ATO event logs** | — | **No public dataset of failed-authentication events exists.** Searched explicitly; every result was vendor marketing. `failed_auth_count_24h` therefore has **no public source at all** (§8.5) |

**Discovery-method note.** Discovery was executed against the HuggingFace datasets
API (full `search=fraud` listing, plus four targeted queries) and against web search
across all six requested categories. The API queries are reproducible; the absence
of any auth-event or device-telemetry dataset in a ~120-entry fraud listing is
itself evidence, not merely a search failure.

---

## 5. Provenance

Every candidate, using the project's existing exposure/eligibility vocabulary.
**No existing exposure or eligibility artifact is edited by this document** (§11).

| # | Dataset | Version | Source URL | Publisher | Licence | Access status | Acquisition status | SHA-256 | Rows | Feat. | Time range | Entity IDs | Label definition | Known limitations | PS-14 exposure | Eligibility | Evidence class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ULB Credit Card Fraud Detection | Kaggle mirror | `kaggle.com/datasets/mlg-ulb/creditcardfraud` | ULB ML Group | **`PENDING REVIEW`** — conflicting refs (Kaggle CC BY-SA 4.0 in `metric_definitions.md`; "AGPL-3.0" string in `dataset_card.py`) | open on Kaggle | **ACQUIRED** | `76274b69…` | 284,807 | 31 cols (30 PCA + Time/Amount/Class) | 2 days, Sept 2013, offset seconds only | **none** | `Class` = fraud indicator | PCA-transformed (original semantics lost); no calendar date; no channel; no entity | **EXPLORATORY** | `EXPLORATORY USE ONLY` | Real (anonymised) |
| 2 | Kaggle fraudTrain | Kaggle | Kaggle | Kaggle | `PENDING REVIEW` — no licence citation located | Kaggle terms | **ACQUIRED** | `fd713920…` | 1,296,675 | 23 cols | `trans_date_trans_time` / `unix_time` (absolute) | `cc_num` (card); customer via `cc_num` | `is_fraud` | no MCC/channel; `zip`/`state`/`city` are **cardholder**, not merchant | **EXPLORATORY** | `EXPLORATORY USE ONLY` | Real-synthetic-labelled |
| 3 | Kaggle fraudTest | Kaggle | Kaggle | Kaggle | `PENDING REVIEW` | Kaggle terms | **ACQUIRED** | `12d553ab…` | 555,719 | 23 cols | absolute | `cc_num` | `is_fraud` | confirmatory-vs-exposed tension recorded in NR-03 → `PENDING REVIEW` | **EXPLORATORY** | `PENDING REVIEW` | Real-synthetic-labelled |
| 4 | IBM v2 (Kaggle Base-FraudDetection 15-col form) | v2 | Kaggle | Kaggle | `PENDING REVIEW` | Kaggle terms | **ACQUIRED** | `b01fa323…` | **24,386,900** | **15 cols** | `Year/Month/Day` + `Time`, 1995–2020 | `User`, `Card`, `Merchant Name`, `Merchant City` | `Is Fraud?` = Yes/No | **15 cols — manifest's "168 features"/`hour_diff` describes a different, absent file (§3.4)** | **EXPLORATORY** (production training set) | `EXPLORATORY USE ONLY` | Real |
| 5 | PS-14 synthetic | generator | in-repo | PS-14 | in-repo | n/a | **ACQUIRED** (generated) | `9f0f56bf…` | n/e | 27 cols (21-feature causal vector) | generator | generator-only ids | `label` | **`SIMULATED/SYNTHETIC`** — cannot establish real-world performance | used throughout development | simulation use only | **Synthetic** |
| 6 | IEEE-CIS Fraud Detection | Kaggle competition 2019 | `kaggle.com/c/ieee-fraud-detection`; IEEE DataPort | Vesta / IEEE | `BLOCKED` — terms unchecked | competition rules | **NOT ACQUIRED** — `data/external/` absent | none (unacquired) | 590,540 train (+506,284 test) | ~434 cols | `TransactionDT` **relative timedelta** | card1–card6 (anonymised), identity table ~24% join | `isFraud` | relative timestamp; no user/merchant/city id; label method undocumented | **none established** (plan §14) | `CANDIDATE CONFIRMATORY` **BLOCKED** | Real (Vesta) |
| 7 | BAF | NeurIPS 2022 | `kaggle.com/datasets/sgpjesus/bank-account-fraud-neurips-2022` | Feedzai | public download via Kaggle | open | **NOT ACQUIRED** — `data/external_benchmark/` absent | none (unacquired) | 6 synthetic variants | ~30 cols | `month` only | **none documented** | `fraud_bool` | **unit of record is account-opening, not a transaction** | **none established** (plan §14) | `CANDIDATE CONFIRMATORY` **BLOCKED** | **Synthetic** (generated on anonymised real base) |
| 8 | Dal Pozzolo ~49.86M | 2014 paper | none published | author institution | **confidential / proprietary** | **NOT publicly obtainable** | **NOT ACQUIRED — no acquisition path** | none (unobtainable) | ~49,858,600 | undisclosed | 13 months, Jan 2006–Jan 2007 | card only; no user/device | processor post-hoc fraud | **confidential; not released; not reproducible** | none | **not eligible — no access** | Real (unobtainable) |
| 9 | FreeFraudDetection50M | as claimed | **not located** | not established | CC BY-NC 4.0 (claimed) | **NOT ESTABLISHED** | **NOT ACQUIRED — could not be located** | none | 50,000,000 (claimed) | 19 (claimed) | claimed | user id, device id (claimed) | fraud label/type (claimed) | **could not be verified to exist** | none | not eligible | **`SYNTHETIC`** |
| 10 | Zenodo deployment-derived online banking | v5, 2026-05-04 | `zenodo.org/records/20359708`; DOI `10.5281/zenodo.20030064` | Sulaimani Polytechnic Univ. | **CC BY 4.0** | **OPEN** | **NOT ACQUIRED** | none (MD5 only: `856540b3…`) | 56,962 | 38 (30 input) | 31 days, Jan 2026 | **none** (transaction_id = record key only) | `is_fraud`, 98 confirmed | **proof-of-concept demo, not a bank; verification bias; model-dependent labels; 0.172%** | none | **not eligible as unbiased ground truth** | Real (public API submissions) |
| 11 | Elliptic / Elliptic++ | 2019 / 2022 | `kaggle.com/ellipticco/elliptic-data-set` | Elliptic Ltd | public research release | open | **NOT ACQUIRED** | none | 203,769 nodes | graph + 166 features | timestamps present | **entity graph** | illicit / licit | crypto domain; no card/MCC/channel | none | not eligible (wrong domain) | Real |
| 12 | Worldline Belgium | 2017 | no public URL | Worldline | **confidential** | **institutional access required** | **NOT ACQUIRED** | none | ~60,000,000 | undisclosed | Jan–Jul 2017 | customer + terminal | human-investigator | schema/label method undocumented | none | `ACCESS_REQUEST_REQUIRED` | Real (unobtainable) |
| 13 | Novatti | 2023–2025 | no public URL | Novatti Group Ltd | **confidential** | institutional | **NOT ACQUIRED** | none | 126,184 | undisclosed | absolute | merchant only (**customer + card removed**) | chargeback | identifiers removed | none | confidential | Real (unobtainable) |

---

## 6. Licensing and Access Status

### 6.1 Summary

| Access tier | Datasets | Consequence for this project |
|---|---|---|
| **Acquired and usable** | IBM v2, ULB, Kaggle fraudTrain/fraudTest, PS-14 synthetic | IBM v2 is the only one that satisfies the native 48 schema |
| **Public but not acquired** | IEEE-CIS, BAF, Zenodo 2026, Elliptic | Acquisition is a separate, unauthorised step for this phase |
| **Claimed but unlocatable** | FreeFraudDetection50M | Cannot be planned against |
| **Confidential / institutional** | Dal Pozzolo 49.86M, Worldline, Novatti | Permanently unavailable without an institutional agreement |

### 6.2 Licence defects that remain open

These are carried forward **unchanged** from
`docs/evaluation/DATASET_ELIGIBILITY_MANIFEST.md`; this phase resolves none of them.

| Item | State | Note |
|---|---|---|
| ULB licence | **`PENDING REVIEW`** | Three conflicting in-repo references (Kaggle CC BY-SA 4.0 / "AGPL-3.0" / plan §35 unchecked). Not merged, not chosen |
| Kaggle fraudTrain / fraudTest licence | **`PENDING REVIEW`** | No licence citation located |
| IBM v2 licence | **`PENDING REVIEW`** | "Open licence" claimed at protocol-era; exact string unresolved |
| IEEE-CIS terms | **`BLOCKED`** | Manual verification required, not performed |
| BAF terms | **`BLOCKED`** | Manual verification required, not performed |
| Zenodo 2026 | **CC BY 4.0, open** | The only newly identified candidate with an unambiguous, permissive, machine-checkable licence |
| FreeFraudDetection50M | **claimed CC BY-NC 4.0** | Claimed, and `NOT ESTABLISHED`; NC terms would also constrain any use |

**Consequence:** even if every other blocker were removed, plan §35's "licence/access
verified" checkbox is **unchecked for every dataset**, so no dataset is
licence-cleared for confirmatory use today.---

## 7. Feature Coverage Matrix

### 7.1 Classification vocabulary (used verbatim, as specified)

| Class | Meaning used here |
|---|---|
| `OBSERVED` | The value exists as a source field in the dataset (or is a direct, lossless read of one) |
| `DERIVABLE` | A deterministic function of source fields that exist in that dataset |
| `PARTIALLY_DERIVABLE` | Derivable only through a proxy or under a semantic/label constraint that is documented but not equivalent to the contract |
| `UNAVAILABLE` | The dataset does not carry the information |
| `SYNTHETIC_ONLY` | A value could be produced only by generation; no real observation exists in that source |
| `UNKNOWN` | Not enough verified information to classify |
| `REJECTED` | Assessed and refused for this contract (unit-of-record / domain mismatch) |

The repository's own vocabularies (`FeatureAvailability`: `directly_available` /
`deterministically_derivable` / `unavailable` / `leakage_risk` / `ambiguous`;
`DerivabilityClassification`: `direct` / `deterministic_derivation` /
`historical_derivation` / `not_derivable` / `leakage_risk` / `unknown`) are mapped,
not renamed — see §7.3.

### 7.2 The matrix — 48 native features × 9 datasets

Column keys: **IBM v2** = acquired 15-col file (real); **Kaggle** = fraudTrain/Test
(real-synthetic-labelled); **ULB** = acquired PCA benchmark; **PS-14 synth** =
in-repo generator; **IEEE-CIS** = public schema, not acquired; **BAF** = public
schema, not acquired; **Zenodo26** = public, not acquired; **DalPozzolo** =
confidential; **FFD50M** = claimed synthetic, not located.

| # | native feature | IBM v2 | Kaggle | ULB | PS-14 synth | IEEE-CIS | BAF | Zenodo26 | DalPozzolo | FFD50M |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `amt` | OBSERVED | OBSERVED | OBSERVED | SYNTHETIC_ONLY | OBSERVED | REJECTED | OBSERVED | UNAVAILABLE | SYNTHETIC_ONLY |
| 2 | `log_amt` | DERIVABLE | DERIVABLE | DERIVABLE | SYNTHETIC_ONLY | DERIVABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 3 | `amt_sq` | DERIVABLE | DERIVABLE | DERIVABLE | SYNTHETIC_ONLY | DERIVABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 4 | `hr` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 5 | `mn` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 6 | `dow` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 7 | `Month` | OBSERVED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 8 | `Day` | OBSERVED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 9 | `hour_sin` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 10 | `hour_cos` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 11 | `is_night` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 12 | `is_business_hours` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 13 | `chip` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 14 | `is_online` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 15 | `is_swipe` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 16 | `err` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 17 | `has_zip` | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 18 | `has_state` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 19 | `is_online_or_no_state` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 20 | `mcc` | OBSERVED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 21 | `mcc_high` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 22 | `mcc_restaurant` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 23 | `mcc_gas` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 24 | `mcc_grocery` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 25 | `mcc_travel` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 26 | `mcc_online` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 27 | `merchant_id` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 28 | `city_id` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 29 | `card_id` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNKNOWN | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 30 | `user_tx_count` | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 31 | `card_tx_count` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 32 | `user_avg_amt` | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 33 | `amt_vs_user_avg` | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 34 | `amt_zscore` | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 35 | `merch_tx_count` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 36 | `user_merchant_diversity` | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 37 | `user_city_diversity` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 38 | `user_fraud_rate` | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | PARTIALLY_DERIVABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 39 | `merch_fraud_rate` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | PARTIALLY_DERIVABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 40 | `city_fraud_rate` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | PARTIALLY_DERIVABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 41 | `high_amt` | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 42 | `very_high_amt` | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 43 | `amt_x_hr` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 44 | `amt_x_mcc` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 45 | `amt_x_chip` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 46 | `amt_x_online` | DERIVABLE | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 47 | `amt_x_night` | DERIVABLE | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY |
| 48 | `user_merch_count` | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNAVAILABLE | REJECTED | UNAVAILABLE | UNAVAILABLE | SYNTHETIC_ONLY |

### 7.3 Column totals

| Dataset | n | OBSERVED | DERIVABLE | PARTIALLY_DERIVABLE | UNAVAILABLE | SYNTHETIC_ONLY | UNKNOWN | REJECTED |
|---|---|---|---|---|---|---|---|---|
| **IBM v2** (acquired, real) | 48 | **4** | **44** | 0 | 0 | 0 | 0 | 0 |
| Kaggle (acquired) | 48 | 1 | 18 | 10 | 19 | 0 | 0 | 0 |
| ULB (acquired) | 48 | 1 | 2 | 0 | 45 | 0 | 0 | 0 |
| PS-14 synth | 48 | 0 | 0 | 0 | 0 | 48 | 0 | 0 |
| IEEE-CIS | 48 | 1 | 2 | 3 | 41 | 0 | 1 | 0 |
| BAF | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 48 |
| Zenodo26 | 48 | 1 | 13 | 0 | 34 | 0 | 0 | 0 |
| DalPozzolo | 48 | 0 | 0 | 0 | 48 | 0 | 0 | 0 |
| FFD50M | 48 | 0 | 0 | 0 | 0 | 48 | 0 | 0 |

### 7.4 Evidence for each column (no cell filled from intuition)

**IBM v2 — 4 `OBSERVED`, 44 `DERIVABLE`, 0 gaps.** Source columns measured in §3.3.
The four `OBSERVED` are `amt` (column `Amount`), `Month` (column `Month`), `Day`
(column `Day`), `mcc` (column `MCC`). Every other cell is `DERIVABLE` because
`derive_native_features` computes it from fields this file carries, and
`train_altman_native.py` already implements every one of those derivations on this
exact file: `chip/is_online/is_swipe` from `Use Chip`; `err` from `Errors?`;
`has_zip` from `Zip`; `has_state` from `Merchant State`; `mcc_*` from `MCC`;
`hr/mn/dow` from `Time` + `Year/Month/Day`; `merchant_id/city_id/card_id` from
`sha256` of `Merchant Name` / `Merchant City` / `Card`; the eight velocity/diversity
features from `groupby(...).cumcount()`; the three fraud rates from
`expanding().mean().shift(1)`. No cell depends on a field this file lacks.

**Kaggle — `PARTIALLY_DERIVABLE` is used exactly where a proxy stands in for a
missing identity.** Measured columns (§3.3) contain `cc_num` (card) and `merchant`,
but **no `user_id`, no MCC, no channel field, and no merchant city or merchant
state**. Therefore: `card_tx_count`, `merch_tx_count`, `merch_fraud_rate`,
`merchant_id`, `card_id` are `DERIVABLE`; but `user_tx_count`, `user_avg_amt`,
`amt_vs_user_avg`, `amt_zscore`, `user_merchant_diversity`, `user_fraud_rate`,
`high_amt`, `very_high_amt`, `user_merch_count` are `PARTIALLY_DERIVABLE` because
they are computed **per card, not per user** — a different estimand, not the
contract's. `has_zip` is `PARTIALLY_DERIVABLE` because Kaggle's `zip` is the
**cardholder's** postal code while the contract's `has_zip` tests the **merchant**
`Zip` field; a non-empty test on a different column is not the same predicate.
`has_state` and `user_city_diversity` are `UNAVAILABLE` because the contract's
`merchant_state` / merchant city do not exist (Kaggle's `state`/`city` are the
cardholder's). Tier B is `DERIVABLE` because `trans_date_trans_time` / `unix_time`
are absolute.

**ULB — 45 of 48 `UNAVAILABLE`.** Measured columns are `"Time"`, `V1..V28`,
`"Amount"`, `"Class"`. `"Amount"` gives tier A. **`"Time"` is seconds from an
undisclosed first transaction, with no calendar date** — so no hour, minute,
weekday, month or day can be recovered. The repository says so itself in
`backend/research/public_feature_contract.json`: `"notes": "Cannot compute from ULB
Time (seconds offset)"` for both `hour_of_day` and `is_weekend`. `V1..V28` are PCA
components with **no disclosed source attribute**, so they cannot stand in for
channel, MCC, merchant, city or card. There is **no entity identifier of any kind**,
so every velocity, diversity and rate feature is unobtainable. Only tier A survives.

**PS-14 synth — `SYNTHETIC_ONLY` for all 48, by construction.** The file is the
project's own generator output (`manifest §7`: "SIMULATED/SYNTHETIC"). No cell is
`OBSERVED` or `DERIVABLE` because no cell is a real observation. Its *structural*
coverage is separately recorded in §7.5 — noting in particular that it stores
`amount_ratio`, **not** a raw `amount`, and has no `mcc`, `merchant`, `city`, `card`,
`user`, `zip`, `state` or `errors` column, which is exactly why the mapped runtime
path substitutes constants for `mcc_n`, `has_zip` and `has_state`.

**IEEE-CIS — taken verbatim from the repository's own executable audit.** The column
is not re-derived here: `ieee_cis_forensic_audit._build_derivability_matrix()`
already returns one entry per native feature with its own class and source field, and
its `feature_name` set was asserted equal to `ALTMAN_NATIVE_FEATURES`. Mapping:
`direct→OBSERVED`, `deterministic_derivation→DERIVABLE`,
`historical_derivation→DERIVABLE`, `not_derivable→UNAVAILABLE`,
`leakage_risk→PARTIALLY_DERIVABLE` (derivable only by consuming `isFraud` history,
so it is constrained rather than free), `unknown→UNKNOWN`. Counts in §7.3 are
extracted by execution, not typed by hand. Root causes in §4.3.

**BAF — `REJECTED` × 48.** Unit of record is a bank account *opening application*,
not a transaction (§4.4). There is no transaction amount, no transaction timestamp,
no merchant, no channel, no card. This is an assessed refusal, not an absence of
investigation.

**Zenodo26 — 1 `OBSERVED` + 13 `DERIVABLE`.** `amount` is a direct column; the nine
tier-B features come from the **absolute UTC `timestamp`**; `log_amt` and `amt_sq`
from `amount`. Everything else is `UNAVAILABLE`: `V1–V28` are PCA components the
authors state cannot be independently recovered, and the dataset has **no entity
identifiers, no MCC, no channel, no merchant** (§4.5). Its labels are not used to
grant coverage, because they are model-dependent and verification-biased.

**DalPozzolo — `UNAVAILABLE` × 48.** Not obtainable; no access path; schema not
published (§4.1). `UNAVAILABLE` (a factual state) rather than `REJECTED`, because no
assessment was possible.

**FFD50M — `SYNTHETIC_ONLY` × 48**, per instruction, and the dataset could not be
located at all (§4.2).

### 7.5 PS-14 synthetic generator — structural coverage (informational)

Not a claim of coverage; recorded so that "derivable" is never read as "observed"
for this source.

| Generator field | Native-48 equivalent |
|---|---|
| `ts` | supports tier B derivation (but generated) |
| `amount_ratio` | **not** `amt`; the mapped runtime reconstructs `amt := amount_ratio × 100` (reference $100) — a proxy, not the raw amount |
| `hour_of_day`, `is_weekend` | partial tier B |
| `txn_freq_last_24h`, `days_since_last_similar_txn`, `gradual_escalation_score`, `amount_zscore`, `txn_regularity` | causal-contract features, **not** native-48 features |
| `new_device_flag`, `unusual_location_flag`, `unusual_recipient_flag`, `failed_auth_count_24h`, `known_device_count`, `account_tenure_days`, `shared_device_accounts`, `shared_recipient_accounts`, `mule_ring_score`, `recipient_novelty`, `hour_deviation`, `velocity_deviation`, `txn_time_unusual` | **21-feature causal contract**, outside the native 48 |
| absent entirely | `mcc`, `use_chip`, `errors`, `zip`, `merchant_state`, `merchant_name`, `merchant_city`, `card`, `user_id` |

---

## 8. Derivability Analysis

### 8.1 Rule applied

A feature is `DERIVABLE` from dataset D only if **all** of the following hold:

1. every source field the contract's formula reads exists in D;
2. the computation is deterministic and documented in
   `derive_native_features` / `build_feature_contract.py`;
3. it can be computed **prior-only** (no future rows, no future labels);
4. it needs no field from a *different* dataset.

Rule 4 is what forbids fake row-level merges (§13). Where a proxy is substituted for
a missing identity, the cell is `PARTIALLY_DERIVABLE`, never `DERIVABLE`.

### 8.2 Features that are `UNAVAILABLE` from **every** public source

**None within the native 48.** All 48 are covered by IBM v2 alone
(4 `OBSERVED` + 44 `DERIVABLE`). The `UNAVAILABLE` cells are dataset-specific, not
contract-wide.

This is the central empirical result of the phase, and it is the opposite of what
the brief anticipated. The native 48 was built around one schema, and the project
already holds a 24.4M-row file in that schema.

### 8.3 Where the irreducible risk sits instead

The risk is not coverage — it is **concentration and label dependency**:

1. **Single-source dependency.** 100% of native-48 coverage comes from one file. If
   IBM v2's provenance or licence is not cleared (it is `PENDING REVIEW`), native-48
   coverage drops from 48/48 to **0/48** for every other dataset in this study.
   There is no second source to fall back on. This is a *provenance* risk, not a
   *coverage* risk, and cannot be fixed by searching harder — the alternatives are
   confidential, synthetic, or wrong-unit-of-record.
2. **Label-latency dependency.** Three features (`user_fraud_rate`,
   `merch_fraud_rate`, `city_fraud_rate`) require **labels**. They are
   `PARTIALLY_DERIVABLE` on any labelled dataset and `UNAVAILABLE` on an unlabelled
   one. Their correctness rests entirely on the shift discipline
   (`expanding().mean().shift(1)`), which is a code property, not a data property.
   A dataset whose labels arrive with latency or verification bias (§4.5) silently
   corrupts these three. This is why the Zenodo 2026 dataset is refused for coverage
   despite its open licence.
3. **Cold-start distribution.** The contract's cold-start defaults (`0.001` fraud
   rate, `1.0` ratio, `12.0` hour, `0.0`) mean the *training* prevalence of
   cold-start rows determines what the model actually learns. IBM v2 covers
   1995–2020 with a 0.122% positive rate; a different corpus would have different
   cold-start mass and a different learned prior.

### 8.4 `failed_auth_count_24h` — explicitly not derived from transaction activity

This feature belongs to the 21-feature causal contract (§2.5), not the native 48.
The brief's constraint is correct and is honoured:

- It requires **actual authentication/security events**. Transaction rows contain no
  authentication outcomes.
- **No public dataset of failed-authentication events exists** (§4.6 — searched
  explicitly).
- Therefore its only honest status is: **no real public source; any value must be
  generated under an explicitly documented augmentation procedure, or it must remain
  a cold-start `0.0`.**
- Deriving it from transaction frequency or amount anomalies would be a
  **redefinition**, which §11 forbids and which the contract's own `public_v1` file
  already refuses: `failed_auth_proxy` carries `"available_in": []` with the note
  that "Semantics beyond the name were never specified or implemented ... so no run
  can silently use it".

### 8.5 The five features named in the brief (analysed against the 21-feature contract)

Because they are **not in the native 48** (D5), each is classified against the
datasets that could ever supply it:

| Feature | Would require | Any public source? | Status | Why it is not "derivable" here |
|---|---|---|---|---|
| `recipient_novelty` | **recipient identity** distinct from the payer, plus prior ordering | **No.** IBM v2, Kaggle, ULB, IEEE-CIS, BAF and Zenodo26 all lack a recipient/payee identifier (Kaggle's `R_emaildomain`-style fields do not exist; IEEE-CIS has `R_emaildomain` ≈76% missing and it is a domain, not an account) | **UNAVAILABLE (no public source)** | A card-merchant pairing is not a payer-recipient pairing. Substituting one for the other changes the estimand |
| `shared_device_accounts` | **device id** + account id, joined | **No.** No acquired dataset has a device identifier. IEEE-CIS has `DeviceInfo` but joined to only ~24% of rows and it is a device *string*, not a stable cross-account key | **UNAVAILABLE (no public source)** | Requires a device–account graph that no public corpus provides |
| `shared_recipient_accounts` | recipient identity + sender identity | **No** (same reason as `recipient_novelty`) | **UNAVAILABLE (no public source)** | Needs a recipient graph |
| `failed_auth_count_24h` | authentication events | **No** — searched, none exists | **UNAVAILABLE (no public source)** | §8.4. Explicitly **not** derived from transaction activity |
| `mule_ring_score` | a defensible **account/recipient graph** from which the contract's composite can be computed | **No** for a payments graph. Elliptic provides a real entity graph but in the Bitcoin domain with no payer-recipient semantics matching the contract | **UNAVAILABLE (no payments-domain graph)** | The brief's constraint is honoured: not treated as an observed fact without a defensible graph. IBM v2's merchant network is a **merchant** graph, not a **recipient** graph — using it would silently redefine the feature |

**Additional note on the repository's own misuse of `mule_ring_score`.** Several
offline experiment scripts assign it from unrelated quantities — e.g.
`combined_eval.py:54` sets it to `df.groupby('cc_num')['city'].transform('nunique')`
(unique cities per card) and `ealtman_eval.py:156` sets it from a user-city percentile
flag. These are **experiment-local surrogates for offline sweeps**, not the contract
definition, and they must not be read as evidence that the feature is derivable from
transaction data. They are noted here because their existence could otherwise be
mistaken for a derivation.

### 8.6 Summary: irreducible gaps

| Scope | Irreducible gaps |
|---|---|
| **Native 48** | **0** — fully covered by IBM v2 (48/48) |
| **21-feature causal contract**, device / recipient / auth / graph features | **5 of the 5 named in the brief have no public source**; the remaining device/recipient flags (`new_device_flag`, `unusual_location_flag`, `unusual_recipient_flag`, `known_device_count`) share the same gap |
| **Across all candidates combined** | No dataset supplies device telemetry, recipient identity, authentication events, or a payments-domain relationship graph |---

## 9. Missing-Feature Analysis

### 9.1 Missing-feature analysis **within the native 48**

| Metric | IBM v2 (only full source) | Best non-IBM real dataset | Best overall public candidate |
|---|---|---|---|
| OBSERVED | **4** | Kaggle 1 | IBM v2 4 |
| DERIVABLE | **44** | Kaggle 18 | IBM v2 44 |
| PARTIALLY_DERIVABLE | 0 | Kaggle 10 | Kaggle 10 |
| UNAVAILABLE | **0** | ULB 45 / Zenodo26 34 | ULB 45 (as an *independent* source) |
| **Coverage total** | **48/48** | 29/48 | **48/48** |

The native 48 has **no missing features** from IBM v2. The "missing-feature"
question therefore has a different answer than the brief anticipated, and stating it
plainly matters more than manufacturing a gap list.

### 9.2 Where features *are* missing — the 21-feature causal contract

| Contract | Missing from all public data | Nature of the gap |
|---|---|---|
| Native 48 | **none** | — |
| 21-feature causal (`ML_FEATURES`) | `new_device_flag`, `unusual_location_flag`, `unusual_recipient_flag`, `failed_auth_count_24h`, `known_device_count`, `shared_device_accounts`, `shared_recipient_accounts`, `mule_ring_score`, `recipient_novelty` — **9 of 21** | institutional-grade **device telemetry**, **recipient identity**, **authentication events**, **account graph** |

The repository already records this exact boundary in
`backend/research/public_feature_contract.json`:

> `"missing_from_public": { "count": 7, "features": [...], "note": "These require institutional-grade device/location/recipient telemetry" }`

§8.5 extends that count from 7 to **9 of 21** for the causal contract by including
`failed_auth_count_24h` and `known_device_count`, which share the same telemetry
dependency. **This is an analysis in this document, not a change to that file.**

### 9.3 Max real observed coverage

**A. Maximum real OBSERVED coverage — `4 of 48`.**
Directly observed source columns are only `amt`, `Month`, `Day`, `mcc` (IBM v2).
Across every acquired real dataset the **union** of directly-observed fields is also
4: the other datasets' only observed field is `amount`/`amt`, already counted.
No public dataset exposes a raw MCC, a channel string, a merchant name, or a
merchant postal code **together with** the rest of the contract.

**B. Maximum defensible derived coverage — `48 of 48`.**
Reached by IBM v2 alone, using the derivations the project already implements in
`train_altman_native.py`. No feature would need to be invented.

**C. Remaining irreducible gaps.**
- *Native 48:* **0 features**.
- *Realism/robustness of that coverage:* 1 structural gap — everything rests on one
  file whose licence is `PENDING REVIEW` (§6.2). If IBM v2 fails licence or
  provenance verification, native-48 coverage becomes **0/48** and no alternative in
  this study exceeds **29/48** (Kaggle) or **14/48** (Zenodo26).

**D. Augmentation requirement.**

| Scope | Features requiring generation |
|---|---|
| Native 48, using IBM v2 | **0** |
| Native 48, IBM v2 unavailable (Kaggle-only) | **19** (`UNAVAILABLE` in the Kaggle column) |
| Native 48, no real source at all | 48 |
| 21-feature causal contract | **9** (`new_device_flag`, `unusual_location_flag`, `unusual_recipient_flag`, `failed_auth_count_24h`, `known_device_count`, `shared_device_accounts`, `shared_recipient_accounts`, `mule_ring_score`, `recipient_novelty`) |

**E. Row-scale feasibility — see §11.**

### 9.4 Distinguishing what the existing pipeline already does

Several repository scripts *fill* unavailable native features with substitutes. These
are documented experiment-local devices and must not be mistaken for coverage:

| Script | Substitute | Nature |
|---|---|---|
| `build_feature_contract.py` (mapped runtime) | `chip` ← `new_device_flag`; `is_online` ← `new_device_flag`; `mcc_n`, `has_zip`, `has_state` ← **constant 0.0** | documented proxy, constant on both sides by construction |
| `combined_eval.py`, `comprehensive_multi_eval.py`, `cross_domain_*`, `ealtman_*` | `mule_ring_score` ← unique cities per card, or a user-city percentile flag | offline sweep surrogate; **redefines** the feature |
| `cross_dataset_eval.py` | `PCA-PROXY mapping onto the 21-feature contract (V1..V21 slot-filling)` | proxy; the plan records these results as `FAILED / NON-CONFORMING` |
| `phase23b_evaluation.py` | P20 45-feature run with **33/45 features reconstructed**, NaN-filled | degraded; recorded in claim C-010 |

None of these changes a cell in §7.2. They are listed so that a future reader cannot
mistake a proxy for an observation.

---

## 10. Scientific Boundary — Real / Derived / Augmented / Synthetic

These four labels are used strictly, and every dataset in §5 carries one.

| Label | Definition | Datasets in this study |
|---|---|---|
| **REAL** | Observed directly in source data | IBM v2, ULB, IEEE-CIS (schema), Zenodo26, Elliptic, Dal Pozzolo (unobtainable), Worldline / Novatti (unobtainable) |
| **DERIVED** | Deterministically calculated from real source information | All 44 non-`OBSERVED` native features on IBM v2; tier B on Kaggle and Zenodo26; 10 `PARTIALLY_DERIVABLE` proxies on Kaggle |
| **AUGMENTED** | Generated from real source information under a documented transformation | **None in this phase.** The repository has SMOTE-based expansion only inside the Zenodo2026 authors' own CNN-LSTM training (their described `~6.3M` from `284,807`), which is that paper's method, not a PS-14 dataset |
| **SYNTHETIC** | Generated independently | PS-14 `data/transactions.csv`; Kaggle fraudTrain/Test (per Kaggle's generation description); BAF (generated on an anonymised real base); FreeFraudDetection50M (claimed); PaySim files; CiferAI; Mendeley 15M; TSAI-MetaFraud |

### 10.1 The mandatory statement

**A 50,000,000-row dataset containing synthetic fields must never be described as
50M real transactions.** Concretely, for any future augmentation:

- the row count of **real** transactions and the count of **generated** rows must be
  reported **separately and always**;
- a dataset must never be labelled "50M transactions" if any fraction is generated —
  the phrasing would be "50M rows, of which N are real transactions";
- `dataset_id` / `source_row_id` / `feature_provenance` (§13) must make the split
  machine-checkable per row, not merely documented in prose.

### 10.2 Labelled-fraction summary for the real corpus

| Source | Rows | Real | Derived columns | Synthetic columns |
|---|---|---|---|---|
| IBM v2 native-48 projection | 24,386,900 | **24,386,900** | 44 of 48 columns | **0** |
| ULB | 284,807 | 284,807 | — | 45 of 48 columns unobtainable |
| Kaggle fraudTrain | 1,296,675 | real-labelled | — | generation-labelled per Kaggle |
| Kaggle fraudTest | 555,719 | real-labelled | — | generation-labelled per Kaggle |
| **Total acquired** | **26,524,101** | — | — | — |

---

## 11. Large-Scale Dataset Candidates and 50M Feasibility

### 11.1 Candidate ladder by real row count

| Candidate | Real rows | Native-48 coverage | Obtainable? | Verdict |
|---|---|---|---|---|
| IBM v2 (acquired) | **24,386,900** | **48/48** | yes (in repo) | **the only viable real corpus** |
| Kaggle fraudTrain + fraudTest | 1,852,394 | 29/48 combined-by-proxy, 19 unusable | yes | cannot extend the native corpus (§11.2) |
| Zenodo26 | 56,962 | 14/48 | yes (open) | too small; labels verification-biased |
| ULB | 284,807 | 3/48 | yes | too small; 45/48 unobtainable |
| Elliptic | 203,769 | 0/48 | yes | wrong domain |
| IEEE-CIS | 590,540 (+506,284 test) | 3/48 | **BLOCKED** (not acquired) | 41/48 not derivable even after acquisition |
| BAF | ~10⁴–10⁵ per variant | 0/48 (REJECTED) | **BLOCKED** | wrong unit of record |
| Worldline | ~60,000,000 | unknown | **no** (confidential) | unreachable |
| Novatti | 126,184 | unknown (customer + card ids removed) | **no** (confidential) | unreachable |
| Dal Pozzolo | ~49,858,600 | unknown | **no** (confidential, never released) | unreachable |
| FFD50M | 50,000,000 (claimed) | unknown | **could not be located** | unreachable |

### 11.2 Why the other real corpora cannot extend the native dataset

Each fails for a **schema** reason, not a size reason:

- **ULB** — no entity id, no absolute date, PCA-only columns. Appending ULB rows would
  require fabricating `use_chip`, `mcc`, `merchant_id`, `city_id`, `card_id`,
  `user_id`, `zip`, `merchant_state`, `errors` — **31 invented columns per row**.
- **Kaggle** — no `mcc`, no channel, no merchant state, no merchant city, no `user_id`.
  Appending would require inventing `mcc` and all six MCC flags, all three channel
  flags, `err`, `has_state`, `is_online_or_no_state`, `city_id`, `city_fraud_rate`,
  `user_city_diversity` — **13 invented columns per row**.
- **Zenodo26** — same ULB-shaped problem, plus unusable labels.
- **User0 extract** — **duplicate** of IBM v2 (§3.2): concatenating would double-count.

Adding Kaggle's 1,852,394 rows would raise the count from 24.4M to 26.2M while
**fabricating 13 columns per appended row**. That is precisely the fake row-level
merge §13 forbids, and it would make the result *less* valid, not more.

### 11.3 50M feasibility verdict

| Question | Answer |
|---|---|
| Maximum real, non-duplicative, native-48-complete rows available to this project | **24,386,900** (IBM v2) |
| Shortfall against 50,000,000 | **25,613,100 rows (51.2%)** |
| Can that shortfall be filled with real, licensed, native-schema data? | **No.** The only corpora of that size (Dal Pozzolo ~49.86M, Worldline ~60M) are confidential and were never released. IEEE-CIS is ~1.1M and yields 3/48 features |
| Can it be filled without violating §13? | **No.** Every available alternative requires inventing contract columns per row |
| **Is a defensible ~50,000,000-row native-contract dataset technically feasible?** | **NO** |

**Forcing exactly 50M would require generating ≥25.6M rows.** That is permitted in
principle (§12) but it would produce a dataset that is, by construction, **not 50M
real transactions** — it would be 24.4M real + ≥25.6M generated. It would also be
exposed anyway: IBM v2 is `EXPLORATORY` (§3.6), so a corpus built from it inherits
exposure and cannot serve as untouched confirmatory evidence under plan §14.

**The brief explicitly asks not to force exactly 50M if the evidence supports a
better design. The evidence supports keeping the 24,386,900 real rows as the
native-contract corpus and labelling them as such.**

---

## 12. Augmentation Boundary

### 12.1 What augmentation is permitted, and what it may never be called

| Rule | Statement |
|---|---|
| **ALLOWED** | Generating additional rows/columns **from IBM v2's own real fields** under a documented, deterministic transformation, each row carrying `dataset_id`, `source_row_id`, `feature_provenance`, and a REAL/GENERATED label |
| **ALLOWED** | Generating a dataset that is **explicitly named as synthetic** and used only for engineering, load/throughput and robustness testing |
| **NEVER** | Presenting an augmented corpus as "50M real transactions" (§10.1) |
| **NEVER** | Using generated rows as confirmatory Track M evidence (plan §13/§14) |
| **NEVER** | Generating the 9 device/recipient/auth/graph features and recording them as observed or derived (§8.4, §8.5) |
| **NEVER** | Merging rows across datasets without a proven entity/time linkage (§13) |
| **NEVER** | Filling `mcc`, `use_chip`, `zip`, `state`, `merchant`, `city`, `card`, `user` columns from a generator and then reporting coverage as `OBSERVED`/`DERIVABLE` |

### 12.2 The 9 causally irreducible features

Any future augmentation must label these as generated **with an explicit
`feature_provenance = "synthetic"`**, and no experiment may use them to support a
real-world claim:

`new_device_flag`, `unusual_location_flag`, `unusual_recipient_flag`,
`failed_auth_count_24h`, `known_device_count`, `shared_device_accounts`,
`shared_recipient_accounts`, `mule_ring_score`, `recipient_novelty`

These are outside the native 48 (§2.5), so they do **not** block a native-48 corpus.
They block any claim about the causal §16 vector.

### 12.3 Correct treatment of cold-start defaults

The contract's cold-start constants are **substitutions with documented meaning**,
not fabrications, and they are already applied identically in training and runtime:

| Feature | Cold-start value |
|---|---|
| `user_fraud_rate`, `merch_fraud_rate`, `city_fraud_rate` | `0.001` (`COLD_START_FRAUD_RATE`) |
| `amt_vs_user_avg` | `1.0` |
| `amt_zscore`, `log_amt`, `amt_sq`, `hr` | `0.0` / `12.0` as per `_native_entries()` |
| `user_merchant_diversity`, `user_city_diversity` | clamp ≥ `1.0` |

Any dataset that adopts the native 48 must use **these same constants**, or
train/runtime parity breaks — which is exactly what
`backend/scripts/feature_parity_test.py` and
`map_raw_to_native`'s shared-derivation path exist to prevent.

### 12.4 Standing constraints

Augmentation is **not authorised in this phase**. It would additionally require,
per plan §14/§35: frozen new-model rules (currently `[TO BE FROZEN]`), cleared
licences (§6.2), a passed freeze checker (currently 78 findings), and reviewer
sign-off (plan §32, currently `NOT ASSIGNED`). **None of these hold.**---

## 13. Dataset-Combination Rules

### 13.1 Prohibited combinations (the fake row-level merge)

The following is **forbidden** and no part of this document performs it:

```
Dataset A transaction  +  Dataset B device  +  Dataset C authentication event
   -> presented as one real transaction row
```

A row may combine observations from multiple datasets **only** if all three hold:

1. a **legitimate entity linkage** (a shared, verified identifier across sources);
2. a **time linkage** proving the observations belong to the same event or episode;
3. the linkage is **recorded per row**, not inferred.

**No candidate pair in this study satisfies condition 1.** There is no shared
identifier between IBM v2, ULB, Kaggle, IEEE-CIS, BAF or Zenodo26, and none is
obtainable. Consequently **no cross-dataset row-level merge is possible today.**

### 13.2 Required per-row provenance keys

Any future dataset must carry, per row (not per file):

| Key | Purpose |
|---|---|
| `dataset_id` | which source corpus the row came from |
| `source_row_id` | the row's identifier within that corpus (never re-generated) |
| `feature_provenance` | per feature: `observed` / `derived` / `augmented` / `synthetic` |

`feature_provenance` must be **per feature**, not per row, because a single row can
legitimately hold observed amount, derived velocity and synthetic device fields at
once. A row-level label would be lossy and would invite exactly the
"50M real transactions" error §10.1 forbids.

### 13.3 Permitted combinations

| Combination | Permitted? | Conditions |
|---|---|---|
| Concatenating two **disjoint slices of the same source** | yes | Only if non-overlapping; `User0_…csv` is a **slice**, not disjoint — verified duplicate (§3.2) |
| Union across datasets at the **dataset level** (separate tables) | yes | Dataset identity preserved; used for coverage reporting, never for row inflation |
| Vertical append requiring invented contract columns | **no** | Fabricates the contract (§11.2) |
| Horizontal column join without shared entity ids | **no** | Fabricates co-occurrence |
| Merging BAF rows into a transaction table for row count | **no** | Wrong unit of record (§4.4) |
| Merging Zenodo26 rows into a real corpus for row count | **no** | 34/48 features would be fabricated; labels verification-biased (§4.5) |

### 13.4 Dataset identity must survive every operation

Each dataset keeps its own identity, hash and exposure state through any
transformation. A derived artifact that mixes sources must carry a **multi-dataset
provenance manifest**, never a single `dataset_id`.

---

## 14. Scientific Limitations

### 14.1 Discrepancies found (recorded, not silently repaired)

| # | Discrepancy | Evidence | Action taken |
|---|---|---|---|
| **D1** | `DATASET_ELIGIBILITY_MANIFEST.md` §4 describes IBM v2 as "**168 raw features**" with a derived `hour_diff` leakage defect, `ESTABLISHED`. The acquired file has **15 columns** and no `hour_diff`. No 168-column file exists anywhere in the repository | measured header + full-repo search | **Recorded here; manifest NOT edited.** Downgrading an `ESTABLISHED` cell is a reviewer decision (§11). The `hour_diff` defect cannot apply to this file |
| **D2** | `nr05_diagnostics.py` hard-codes `n_file_rows=24_386_899`; measured **24,386,900** | full CSV parse (24,386,900 rows × 15 fields; 0 malformed) + `wc -l` 24,386,901 | **Recorded; NR-05 NOT edited** (out of scope). Annotation-only constant |
| **D3** | Exposure ledger §4 records IBM v2 `Rows` as `NOT ESTABLISHED` | now measured | **Recorded here; ledger NOT edited** (additive update is not this phase's authority) |
| **D4** | The **50M target appears nowhere in the repository** — no `50M`, `50,000,000`, `49,858,600`, `FreeFraud` reference exists in `docs/`, `backend/`, `scripts/` or `README.md` | repo-wide search | **Recorded.** The target is external to the codebase; the plan has no row-count requirement to satisfy |
| **D5** | The five features named in the brief (`recipient_novelty`, `shared_device_accounts`, `shared_recipient_accounts`, `failed_auth_count_24h`, `mule_ring_score`) are **not in the native 48**; they are in the 21-feature causal contract and in `public_v1.missing_from_public` | verified against `ALTMAN_NATIVE_FEATURES` and `ML_FEATURES` | **Both analyses given**: native-48 matrix (§7) and a separate causal-contract analysis (§8.5) |

### 14.2 Methodological limitations

1. **Single-source coverage.** 48/48 coverage rests on one file. This is a
   **concentration** risk, not a coverage risk (§8.3.1).
2. **No acquisition was performed.** IEEE-CIS and BAF classifications come from
   their public schemas and the repository's existing audit, not from inspecting
   acquired files. Their cells could change if the real files differ from the public
   description.
3. **"Real" is not "verified real."** IBM v2 is classified real per its provenance
   description; its provenance record is `NOT ESTABLISHED` in the manifest and its
   licence is `PENDING REVIEW`. Nothing here upgrades that.
4. **External candidate details are source-reported.** For non-acquired candidates,
   provenance comes from the publisher's own description. The brief's warning is
   honoured: no candidate is treated as available merely because a paper or a listing
   describes it. FFD50M could not be located at all; Dal Pozzolo / Worldline /
   Novatti are confidential.
5. **The 21-feature causal contract is out of the matrix's scope.** §8.5 covers it
   qualitatively. It has no per-feature × per-dataset matrix here, because the brief
   scoped the matrix to the native 48.
6. **Power is not evaluated.** Plan §18 gates cannot be evaluated for any candidate
   this phase did not acquire. The 98-positive Zenodo26 dataset would fail a
   §18-style power gate on count alone, before any other consideration.

### 14.3 What this document does **not** claim

- ❌ No native validation. No model was run on any candidate.
- ❌ No real-world effectiveness claim.
- ❌ No Track M execution, no freeze, no reviewer assignment, no decision resolution.
- ❌ No eligibility upgrade for any dataset.
- ❌ No claim that 48/48 coverage makes IBM v2 confirmatory — it is `EXPLORATORY`.
- ❌ No claim that any dataset's licence is cleared (§6.2: none are).
- ❌ No row count other than the measured ones in §3.2.

---

## 15. Recommendation for the Next Phase

### 15.1 Verdict on generation readiness

> **Is the next phase ready to generate the augmentation dataset?**
> **NO.**

Three independent reasons, each sufficient on its own:

1. **Scientific (§11.3).** A defensible ~50M-row native-contract dataset is **not
   feasible**: only 24,386,900 real native-complete rows exist and every alternative
   requires fabricating contract columns. Reaching 50M means generating ≥25.6M rows.
2. **Governance (§12.4).** Augmentation requires frozen new-model rules
   (`[TO BE FROZEN]`), cleared licences (**none cleared**), a passing freeze checker
   (**78 findings**), and reviewer sign-off (**NOT ASSIGNED**, plan §32). None hold.
3. **Provenance (§3.4/D1).** Before any generation built on IBM v2, the eligibility
   manifest's demonstrably incorrect description of that file (D1) must be corrected
   by a reviewer — otherwise a generated dataset inherits an unverified provenance
   record.

**Per the brief's instruction, the augmentation dataset was therefore NOT
generated.** No augmentation specification is issued, because none is currently
warranted.

### 15.2 Recommended next actions (in order)

| # | Action | Owner | Precondition |
|---|---|---|---|
| 1 | Reviewers resolve **D1** (manifest's "168 features"/`hour_diff` for IBM v2) and **D2** (NR-05 off-by-one), then correct the affected cells | Statistical + domain reviewers | §32 sign-off exists — **it does not**; this is the prerequisite for everything below |
| 2 | Manually verify the licence/terms of IBM v2, ULB and Kaggle (plan §35 unchecked boxes) | Data-governance/privacy | — |
| 3 | Add an **additive** row to the exposure ledger recording the measured IBM v2 row count (24,386,900) and this document's provenance | Data-governance | action 1 |
| 4 | Decide whether to **acquire IEEE-CIS** (only ~3/48 features — needs a justification beyond "it's untouched") | Statistical reviewer | action 1 |
| 5 | Archive an independent provenance record for IBM v2 (what institution, what population, what label procedure) | Data-governance | action 1 |
| 6 | **Do not** design a 50M corpus. If scale is later required, specify it as a labelled real+synthetic corpus per §10.1 | — | actions 1–2, 5 |

### 15.3 If a corpus is later required — specification constraints (not an authorisation)

Should a future authorised phase want more rows, the specification must:

- report **real vs generated counts separately and always**;
- use `dataset_id` / `source_row_id` / per-feature `feature_provenance` on every row;
- never invent `mcc`, `use_chip`, `zip`, `state`, `merchant`, `city`, `card` or
  `user` columns for appended real rows;
- label the 9 causally irreducible features (§12.2) `synthetic`;
- keep the 24,386,900 real rows addressable as their own `dataset_id`;
- never be described as "50M real transactions".

### 15.4 Why no supporting register was created

`DATASET_DISCOVERY_REGISTER.md` was **not** created. §5 already holds the complete
provenance record for all 13 candidates with every field the brief specified, in one
table that is readable against §7. A second registry would duplicate that content,
add a fourth governance artifact to keep in sync with the exposure ledger and
eligibility manifest, and satisfy the brief's own instruction not to "create
unnecessary registries or governance systems".

### 15.5 State at end of this phase (unchanged)

| Item | State |
|---|---|
| Research Plan | **DRAFT / NOT FROZEN / NOT APPROVED** |
| Track M | **BLOCKED** |
| Statistical reviewer | **NOT ASSIGNED** |
| Domain reviewer | **NOT ASSIGNED** |
| Reviewer decisions | **0 / 14 resolved** |
| Freeze findings | **78** |
| Production model | unchanged (`altman_native_v2_20260904_115703`) |
| Dataset eligibility | unchanged — **no dataset is eligible** |
| Datasets acquired | **none** (no new download performed) |

---

## Appendix A — Answers to the ten phase questions

**1. What datasets did PS-14 actually train on?**
`data/credit_card_transactions-ibm_v2.csv` — and nothing else for the deployed model.
`models/production/manifest.json` records `training_dataset` = that path,
`training_dataset_sha256` = `b01fa323…`, `n_features` = 48,
`feature_schema_version` = `altman_native_v2`, `train_rows` = 193,027, seed 42;
`models/model_records/altman_native_v2_20260904_115703.json` corroborates with the
1995–2015 train / 2016–2017 validation / 2018–2020 test split. The in-repo
`data/transactions.csv` generator was used throughout historical development, but not
by the deployed native model.

**2. What datasets did it actually evaluate on?**
IBM v2 (in-domain, chronological split — the model's own final test window),
Kaggle fraudTrain/fraudTest (transfer, recorded `FAILED / NON-CONFORMING`),
ULB (exploratory), and the PS-14 synthetic generator. IEEE-CIS and BAF were **never
evaluated on** — not acquired.

**3. What additional datasets can legitimately contribute missing features?**
For the **native 48: none that are both obtainable and additive.** Kaggle would add
29/48 but requires proxying 10 features and inventing 13 columns; Zenodo26 adds
14/48 with verification-biased labels; IEEE-CIS would add only 3/48 even after
acquisition; BAF is `REJECTED`; Dal Pozzolo / Worldline / Novatti are confidential;
FFD50M could not be located. **The native 48 has no missing features** to fill.

**4. Which of the 48 features are already covered?**
**All 48**, from IBM v2: **4 `OBSERVED`** (`amt`, `Month`, `Day`, `mcc`) and
**44 `DERIVABLE`** — using derivations the project already implements.

**5. Which can be defensibly derived?**
The same 44, plus the 4 observed ones, i.e. **48/48** — every one deterministically,
prior-only, from fields the file carries. On other datasets: Kaggle 18 clean +
10 proxy-only; ULB 2; Zenodo26 13; IEEE-CIS 2.

**6. Which remain impossible to obtain from public data?**
**None of the native 48.** But five features named in the brief are impossible, and
they are **not native-48 features**: `recipient_novelty`,
`shared_device_accounts`, `shared_recipient_accounts`, `failed_auth_count_24h`,
`mule_ring_score` — all in the 21-feature causal contract. No public dataset
provides recipient identity, device-account linkage, authentication events, or a
payments-domain graph. Nine of the 21 causal features are in this position.

**7. Is a ~50,000,000-row construction technically and scientifically defensible?**
**No.** Maximum real, non-duplicative, native-48-complete rows available:
**24,386,900** (51.2% short of 50M). The only larger corpora are confidential and
unreleased. Every obtainable alternative requires fabricating contract columns per
row, which §13 forbids. The brief's allowance not to force exactly 50M applies, and
the evidence supports keeping 24.4M real rows as the corpus.

**8. What exactly must be generated?**
**For the native 48 using IBM v2: nothing — zero features.** If a larger corpus is
ever authorised: at minimum ≥25,613,100 additional rows (labelled generated, not
real), plus the 9 causally irreducible features (§12.2) for any use of the causal
§16 vector — with `failed_auth_count_24h` generated only under an explicitly
documented procedure, never inferred from transaction activity.

**9. What should remain separate rather than merged?**
`data/User0_credit_card_transactions.csv` (a verified duplicate slice of IBM v2);
ULB, Kaggle, Zenodo26 and IEEE-CIS (incompatible schemas — horizontal joins would
fabricate co-occurrence); BAF (wrong unit of record — account-opening applications,
not transactions); and every confidential corpus. **No cross-dataset row-level merge
is possible today**, because no shared identifier exists between any pair.

**10. Is the next phase ready to generate the augmentation dataset?**
**NO — do not generate.** Not scientifically feasible at 50M without fabricating
contract columns; not governance-authorised (plan §14 rules `[TO BE FROZEN]`, no
licence cleared, freeze checker at 78 findings, reviewers `NOT ASSIGNED`, 0/14
decisions); and not provenance-safe while D1 leaves IBM v2's recorded feature
semantics contradicted by the file itself. No augmentation specification is issued.
Prerequisites are listed in §15.2.

---

*End of `dataset-48-coverage 1.0-draft`.*