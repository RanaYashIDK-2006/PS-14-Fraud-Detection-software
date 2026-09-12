# COLD_START_GENERALIZATION_AUDIT

Phase-2 mission: determine why unseen merchants have ~64.5% FPR and whether PS-14 can
retain ≥99% fraud recall while cutting false positives for BOTH known and genuinely
unseen merchants. Final chronological test (2018-2020, 4,578 fraud) was NEVER touched
for any development decision in this mission; all experiments below were selected on the
2016-2017 validation window only.

## Executive Summary

Certified deployed baseline (unchanged through this mission):

**E_hardneg (deployed, certified) — 99.67% recall / 10.99% FPR on the untouched test.**

Mission outcome after 7 candidates + 5 diagnosis experiments:

> **No tested configuration achieved the mission's target of ≥99% recall with a
> materially lower FPR that also improved unseen merchants. The unseen-merchant FPR is
> a Pareto trade-off with aggregate FPR, not a fixable defect at the global-threshold
> operating point.**

**Recommendation: KEEP E_HARDNEG** (deployed model unchanged). The causal novelty
features (merchant age + prior volume) are validated to help the unseen population but
cost ~1 point of overall FPR at the same recall; they are documented as the highest-value
next iteration rather than promoted.

## 1. Reproduction of the unseen-merchant claim (mission #2)

Definition (identical to the certified claim): *seen merchant* = the merchant hash-code
feature value appears among the rows the model ACTUALLY trained on (sampled train,
all fraud + 5% legit, <2016). Diagnosis only — never used to retune.

| Window | Split | Rows | Fraud | Recall @0.0188 | FPR @0.0188 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| val 2016-17 | seen | 172,055 | 3,641 | 0.9948 | 0.1064 | 0.9963 | 0.9261 |
| val 2016-17 | **unseen** | **2,475** | **193** | **1.0000** | **0.7743** | 0.9051 | 0.5707 |
| test 2018-20 | seen | 189,239 | 4,305 | 0.9965 | 0.1016 | 0.9878 | 0.6249 |
| test 2018-20 | **unseen** | **3,133** | **273** | **1.0000** | **0.6451** | 0.9912 | 0.9234 |

The 64.5% test figure is reproduced exactly. Crucially, the validation window shows the
SAME defect (77.4% unseen FPR), so this is a model property present out-of-sample in
development — not test-driven drift.

## 2. Root-cause analysis (why unseen merchants over-flag)

Full-corpus merchant table (one streamed pass over all 24,386,900 rows) shows the
"unseen" population is **not predominantly new merchants**:

* unseen-merchant TEST legit rows: mean merchant age **11.0 years**; only **9.8%** are
  age ≤ 1; **63.7%** are age ≥ 10. The same holds on validation (mean age 9.0).
* **99.6%** of unseen test rows are at merchants that demonstrably had prior corpus
  activity before the transaction.
* unseen-merchant fraud is concentrated at merchants that turned fraudulent AFTER 2016
  (born earlier, legit-only pre-2016, compromised later) — their codes are absent from
  the sampled train precisely because their pre-2016 history was legit (5%-sampled).

Mechanism table (unseen-merchant TEST rows, mean feature values):

| feature | unseen fraud | unseen legit | seen legit |
|---|---|---|---|
| is_online | 0.000 | 0.000 | 0.127 |
| chip | 0.894 | 0.820 | 0.704 |
| amt_vs_user_avg | 3.555 | 1.729 | 1.082 |
| amt_zscore | 2.555 | 0.729 | 0.082 |
| merch_tx_count | 36.9 | 0.68 | 14,570 |
| merch_fraud_rate | 0.978 | 0.004 | 0.018 |
| user_merch_count | 0.11 | 0.58 | 90.8 |

Interpretation: the model learns no split for a code-unseen merchant and falls back to
its "novel merchant + amount deviation → risk" archetype. Within the unseen population
ranking is already strong (PR-AUC 0.923) — the defect is that the absolute score level
of unseen-merchant LEGIT rows sits above the low global threshold demanded by ≥99%
recall. City/card codes remain known for 96-100% of these rows, so the missing
information is specifically **merchant-level establishment**, which no trained feature
encoded.

## 3. Feature findings (mission #3, #8)

See the feature-status table below. Dead-feature investigation: `err` and `has_state`
were already repaired by the state-parse fix in the certification round and stay fixed;
`mcc_online` is constant-0 because no corpus row carries MCC 5967-5969 — it is a
defensive production feature, and dropping it (CS3) changed nothing on validation;
hash-code removal (F_nohash) is *worse* on both axes.

| Feature | Status | Root cause | Training | Production | Decision |
|---|---|---|---|---|---|
| err | REPAIRED | NaN-truthiness parse bug | 1.7% rate | matches | fixed |
| has_state | REPAIRED | NaN→"nan" string | varies | matches | fixed |
| mcc_online | DEAD const-0 | MCC range absent from corpus | const 0 | const 0 | retain defensively; schema-cleanup next release |
| merchant/city/card codes | RETAINED | identity memorization | learned splits | unseen→default branch | no-hash removal worse on both axes |
| merchant_age + prior volume | VALIDATED (new) | establishment signal missing | varies | merchant table shipped | improves unseen 2-3 pts at ~1 pt overall cost; not deployed |

## 4. Experiment matrix (mission #11; all numbers VALIDATION, recall-locked per candidate)

| Candidate | Description | OP | Thr | Overall rec | Overall FPR | Seen FPR | Unseen FPR | Unseen rec |
|---|---|---|---|---|---|---|---|---|
| **E_hardneg (deployed)** | certified baseline | P1 | 0.053558 | 0.9901 | **0.0708** | 0.0635 | 0.6152 | 0.9948 |
| | | P2 | 0.018758 | 0.9950 | **0.1153** | 0.1064 | 0.7743 | 1.0000 |
| F_nohash | minus merchant/city/card codes | P1 | 0.051335 | 0.9901 | 0.0768 | 0.0694 | 0.6310 | 0.9948 |
| CS1 | E recipe + 6 novelty feats | P1 | 0.046758 | 0.9901 | 0.0825 | 0.0759 | 0.5723 | 0.9948 |
| CS2 | CS1 w/ cold-start-only negatives | P1 | 0.044369 | 0.9901 | 0.0943 | 0.0876 | 0.6266 | 0.9948 |
| CS3 | CS1 minus dead mcc_online | P1 | 0.047120 | 0.9901 | 0.0816 | 0.0746 | 0.5947 | 0.9948 |
| | | P2 | 0.014238 | 0.9950 | 0.1230 | 0.1144 | 0.7564 | 1.0000 |
| CS4 | CS1 + cold negatives extra weight 6 | P1 | 0.047206 | 0.9901 | 0.0811 | 0.0737 | 0.6424 | 0.9948 |
| CS5 | CS1 at spw=15 | P1 | 0.057211 | 0.9901 | 0.0797 | 0.0729 | 0.5798 | 0.9948 |
| | | P2 | 0.014699 | 0.9950 | 0.1256 | 0.1170 | 0.7581 | 1.0000 |
| CS6 | CS1 w/ log1p-scaled novelty counts | P1 | 0.047417 | 0.9901 | 0.0822 | — | 0.5951 | 0.9948 |
| | | P2 | 0.014451 | 0.9950 | 0.1229 | — | 0.7542 | 1.0000 |

(CS1 P2 numbers were superseded by the CS5 rerun; CS6 P2 is shown. Machine-readable
full matrix: `reports/coldstart_val_matrix.json`.)

**Verdict on the frontier**: every candidate that lowers unseen-merchant FPR (best:
CS1/CS5 ~57-58% vs E's 61.5% at P1; CS3/CS6 ~75.4-75.8% vs E's 77.4% at P2) raises
overall FPR by ~1 point at the same recall. No candidate dominates the certified model.
Removing hash codes does NOT help unseen merchants (card/city codes are the known
anchor). Targeted cold-start negative weighting (CS2/CS4) does not help.

## 5. Channel generalization (mission #9)

Per-channel validation FPR at each candidate's own P2 threshold:

| Candidate | Thr P2 | online FPR | chip FPR | swipe FPR |
|---|---|---|---|---|
| E_hardneg | 0.018758 | 0.4174 | 0.0653 | 0.1030 |
| F_nohash | 0.020889 | 0.3794 | 0.0727 | 0.0978 |
| CS3 | 0.014238 | 0.5106 | 0.0645 | 0.0838 |
| CS5 | 0.014699 | 0.5271 | 0.0654 | 0.0837 |

Channel recall at P2 is 1.0 (online), 0.9715 (chip), 0.9829 (swipe) for every
candidate. The channel shift documented in the prior mission (2016 online-fraud wave →
2018-19 chip/swipe) is unchanged; CS3/CS5 trade online FPR for swipe FPR, another
symptom of the same Pareto frontier.

## 6. Production impact at real prevalence (mission #14)

The certified E_hardneg figures stand (offline certification): ~0.121% prevalence →
**110.9 alerts / 1,000** (1,204 fraud caught per 100k, 10,974 legit alerts per 100k,
precision ~1.09%). Unseen-merchant rows are 1.6% of test traffic; their FPR is a
documented limitation that at production prevalence adds a small number of alerts
relative to the whole (all unseen fraud — 100% recall — is still caught).

## 7. Cold-start safety gate (mission #13) — applied on validation

| Gate | Requirement | Best candidates | Result |
|---|---|---|---|
| Overall recall | ≥ 99% | all | PASS |
| Unseen-merchant recall | no material regression | all ≥ 0.9948 (E: 0.9948) | PASS |
| Unseen-merchant FPR | improve or justified bound | CS1/3/5/6 improve 2-3 pts | PASS (candidates) |
| Aggregate FPR | must not regress (mission #16) | all improvers regress ~1 pt | **FAIL (no candidate)** |

The gate that matters is #16's: a candidate may replace E_hardneg only with genuine
improvement, and every candidate that improves unseen merchants regresses the aggregate.

## 8. Remaining risks

* **Label latency — UNVERIFIED** (unchanged): the entity fraud-rate and now the
  merchant prior-fraud features assume past labels are available at scoring time; the
  raw dataset cannot establish the real delay. The new merchant-volume table inherits
  this assumption and is documented accordingly.
* **Unseen-merchant FPR ~61-77%** (validation/test at the locked operating point):
  documented limitation; ~64% of it is old rare merchants whose establishment signal
  the model lacks, ~10% is the irreducible cold-start cost of genuinely new merchants.
* **Channel drift** persists (online→chip/swipe migration quantified in the prior
  mission, PSI≈2); all candidates recall every channel at ≥0.97.
* A production deployment of the novelty features would require shipping the
  full-corpus merchant table (birth year + cumulative per-year counts) with the model
  and a new offline/live parity gate; not done because no candidate earned promotion.

## 9. Recommendation

**KEEP E_HARDNEG**

The certified deployed model remains the best aggregate model on locked validation at
both ≥99% and ≥99.5% recall. The unseen-merchant FPR is real, reproduced on
validation, and mechanistically explained; the causal merchant-establishment features
(age, prior volume) are the validated highest-value next step and should be revisited
inside a model change that improves aggregate discrimination first (e.g., better
fraud-side features) so the trade-off becomes a win on both axes. The final test was
not used for any decision in this mission and remains locked.
