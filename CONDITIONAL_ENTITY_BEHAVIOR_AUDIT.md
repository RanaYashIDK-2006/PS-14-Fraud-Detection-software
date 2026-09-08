# CONDITIONAL_ENTITY_BEHAVIOR_AUDIT

Phase-3 mission: test whether a conditional entity+behavior architecture can use
entity-specific information when trustworthy and fall back to causal behavioral
evidence when a merchant is absent - without treating identity absence as fraud
evidence. The 2018-2020 final test was NEVER touched; every decision below is
validation (2016-2017) only. E_hardneg stays the certified control.

## Executive summary

**Control (certified, deployed, unchanged):** `altman_native_E_hardneg_cert_20260904`,
threshold 0.018758, test recall 99.67% / FPR 10.99% (TP 4,563 / FN 15 / FP 20,634 / TN 167,160).

**Verdict: KEEP E_HARDNEG.**

A dead-code bug invalidated the earlier Phase-3 "PB1/PB2 code-free model" results:
in `cs_train.py` the PB code-drops were appended to the `drop` list AFTER `keep` and
`X` were computed, so PB1/PB2 silently trained on the FULL 54-feature frame and were
bit-identical to CS1 (verified: CS1==PB1==PB2 max|Δ| = 0.0). The bug is fixed and
both models retrained code-free (PB1 drops merchant+city+card, PB2 drops merchant
only; 51/53 features). The corrected experiments show:

1. **Genuine code-free models do NOT help absent merchants.** PB1/PB2 overall FPR
   worsens (0.092/0.086 vs E 0.071 at P1) and absent-merchant FPR is NOT improved at
   E's operating point (0.632/0.637 vs 0.615 at P1-fixed). The identity codes carry
   real signal for seen merchants and removing them only hurts.
2. **What the earlier session actually measured was CS1** (novelty features, codes
   retained). Gating CS1's score onto absent rows only - present rows keep E - is the
   only configuration that improves the absent population: absent FPR 0.581 (P1) /
   0.723 (P2) vs E 0.615/0.774. But it costs: at P2 it misses 1/193 absent fraud
   (recall 0.9948 vs E 1.0000) and the seen-side FPR edge is unchanged. This is a
   *trade*, not a dominance.
3. **Cold-start sanity FAILS for every candidate.** Behaviorally-normal legit
   transactions at absent merchants are flagged at 74-80% regardless of model. The
   model genuinely cannot tell "established merchant absent from the 5% training
   sample" from "risky novel merchant" at the score level demanded by ≥99% recall.
4. **The novelty/establishment features pass the leakage audit** (future-row and
   label perturbations change nothing - max|Δ| ≈ 2×10⁻⁸, float32 rounding only).

## 1. Bug found and fixed (invalidates earlier PB1/PB2 results)

`cs_train.py` lines 129-139 (original): the PB branch appended merchant_id (and for
PB1 city/card) to `drop` AFTER `keep`/`X`/`feats_used` were already built from the
`drop` list. Consequence: PB1 == PB2 == CS1 bit-identical. The earlier session's
claim "city/card codes add zero standalone signal once the merchant code is gone"
was an artifact of this bug; the reported gate numbers used a code-FULL CS1 model.
Fixed by assembling the full drop list before `keep` is computed, then retrained both
models (PB1: 51 features no merchant/city/card; PB2: 53 features no merchant code).
PB1/PB2 now genuinely differ from CS1 (max|Δ| 0.42/0.46) and from each other (0.29).

## 2. Frontier (validation-locked; thresholds at recall ≥99% / ≥99.5%)

| Model | OP | Threshold | Overall rec | Overall FPR | Seen FPR | Absent rec | Absent FPR |
|---|---|---|---|---|---|---|---|
| E_hardneg | P1 | 0.053558 | 0.9901 | 0.0708 | 0.0635 | 0.9948 | 0.6152 |
| PB1_nocodes | P1 | 0.050169 | 0.9901 | 0.0919 | 0.0844 | 1.0000 | 0.6433 |
| PB2_nomerchant | P1 | 0.063723 | 0.9901 | 0.0863 | 0.0792 | 1.0000 | 0.6096 |
| CS1_entity+behav | P1 | 0.046758 | 0.9901 | 0.0825 | 0.0756 | 0.9948 | 0.5955 |
| GATE_PB1_absent | P1 | 0.053558 | 0.9901 | 0.0711 | 0.0635 | 0.9948 | 0.6323 |
| GATE_PB2_absent | P1 | 0.057817 | 0.9901 | 0.0686 | 0.0610 | 1.0000 | 0.6240 |
| GATE_CS1_absent | P1 | 0.053558 | 0.9901 | 0.0704 | 0.0635 | 0.9948 | 0.5806 |

| E_hardneg | P2 | 0.018758 | 0.9950 | 0.1153 | 0.1064 | 1.0000 | 0.7743 |
| PB1_nocodes | P2 | 0.018408 | 0.9950 | 0.1377 | 0.1288 | 1.0000 | 0.7932 |
| PB2_nomerchant | P2 | 0.018757 | 0.9950 | 0.1405 | 0.1317 | 1.0000 | 0.7940 |
| CS1_entity+behav | P2 | 0.014461 | 0.9950 | 0.1229 | 0.1144 | 1.0000 | 0.7537 |
| GATE_PB1_absent | P2 | 0.018758 | 0.9950 | 0.1155 | 0.1064 | 1.0000 | 0.7901 |
| GATE_PB2_absent | P2 | 0.018758 | 0.9950 | 0.1156 | 0.1064 | 1.0000 | 0.7940 |
| GATE_CS1_absent | P2 | 0.018525 | 0.9950 | 0.1153 | 0.1070 | 0.9948 | 0.7252 |


At E's FIXED thresholds (not re-locked), absent-only FPR at P1 (0.053558):

| Model (gate present=E) | Absent rec | Absent FPR | Overall rec | Overall FPR |
|---|---|---|---|---|
| E (no gate) | 0.9948 | 0.6152 | 0.9901 | 0.0708 |
| CS1 on absent | 0.9948 | **0.5806** | 0.9901 | **0.0704** |
| PB1 (code-free) on absent | 0.9948 | 0.6323 | 0.9901 | 0.0711 |
| PB2 on absent | 1.0000 | 0.6372 | 0.9903 | 0.0711 |

At E's FIXED P2 threshold (0.018758, the deployed one):

| Model (gate present=E) | Absent rec | Absent FPR | Overall rec | Overall FPR |
|---|---|---|---|---|
| E (no gate) | **1.0000** | 0.7743 | 0.9950 | 0.1153 |
| CS1 on absent | 0.9948 | **0.7230** | 0.9948 | **0.1146** |
| PB1 (code-free) on absent | 1.0000 | 0.7901 | 0.9950 | 0.1155 |
| PB2 on absent | 1.0000 | 0.7940 | 0.9950 | 0.1156 |

The CS1-gate cuts ~5 FPR points on the absent population but sacrifices 1 of 193
absent frauds at the deployed threshold - a trade along the same frontier, not
dominance. True code-free models are worse on both axes.

## 3. History-depth buckets (absent merchants only, E vs CS1-gate, deployed P2 thr)

| Bucket (prior corpus rows) | Rows | Fraud | E rec | Gate rec | E FPR | Gate FPR |
|---|---|---|---|---|---|---|
| 0-0 | 28 | 3 | 1.0 | 1.0 | 0.88 | 0.8 |
| 1-5 | 255 | 16 | 1.0 | 1.0 | 0.8033 | 0.7448 |
| 6-20 | 647 | 55 | 1.0 | 1.0 | 0.8007 | 0.75 |
| 21-100 | 1107 | 90 | 1.0 | 0.9889 | 0.7443 | 0.6971 |
| 101-1000 | 381 | 26 | 1.0 | 1.0 | 0.7775 | 0.7239 |
| 1001-+ | 57 | 3 | 1.0 | 1.0 | 0.8519 | 0.7778 |


## 4. Channel table (validation, P2 threshold)

| Channel | Rows | Fraud | E rec | E FPR | Gate-CS1 rec | Gate-CS1 FPR |
|---|---|---|---|---|---|---|
| online | 24,168 | 3,073 | 1.0000 | 0.4174 | 1.0000 | 0.4174 |
| chip | 120,641 | 527 | 0.9715 | 0.0653 | 0.9715 | 0.0645 |
| swipe | 29,721 | 234 | 0.9829 | 0.1030 | 0.9786 | 0.1022 |

The CS1-gate shaves chip FPR slightly and costs 1 swipe fraud (recall 0.9786 vs
0.9829) - consistent with the absent-fraud loss at P2.

## 5. Score continuity (mission #6)

Mean / median score by history bucket shows E, PB2 and CS1 all concentrate risk on
low-history merchants (bucket 0-0 mean ~0.15-0.19 vs 21-100 ~0.037). Absent-code rows
sit at p50 0.128 (E) / 0.143 (PB2) / 0.103 (CS1) vs 0.0006-0.0007 for seen rows - a
massive score-level discontinuity driven by the *absence of merchant-history
features*, not by the code column itself (the earlier code-substitution probe: zeroing
or randomizing the merchant code of absent rows changed nothing, 77.4%→77.8%).

## 6. Cold-start sanity (mission #10) - FAIL for all candidates

| Model | Normal-absent legit flag rate | All-absent legit flag rate | Absent fraud caught |
|---|---|---|---|
| E | 0.7570 | 0.7743 | 193/193 |
| PB2 | 0.7968 | 0.7940 | 193/193 |
| CS1 (gated) | 0.7390 | 0.7230 | 192/193 |

Behaviorally-normal absent-merchant legit rows (|zscore|<0.5, amt<1.5× user avg,
8-22h, chip) are flagged at 74-80% by every model at the deployed threshold. The
system does NOT satisfy "missing entity identity ≠ high fraud probability".

## 7. Leakage audit (mission #14) - PASS

The six merchant-establishment features (merchant_age, is_merchant_new,
merch_prior_rows/fraud/rate, merch_ever_seen_prior) were recomputed from perturbed
full-corpus merchant tables:
- Future-removal (zero all years ≥ 2018): validation rows unchanged, max|Δ| 2×10⁻⁸.
- Label-delay (zero fraud ≥ 2017): rows 2015-2017 unchanged, max|Δ| 2×10⁻⁸.
- Own-year fraud zeroed (2016/2017/2018): unchanged, max|Δ| 2×10⁻⁸.
All residuals are float32 rounding; the features are causal by construction (cumulative
volume strictly in years < the row's year).

## 8. Label latency (mission #15) - UNVERIFIED (unchanged)

The entity fraud-rate features and the merchant prior-fraud feature assume past labels
are available at scoring time. The raw dataset cannot establish the real delay;
status stays UNVERIFIED and is carried on every candidate including the gated ones.

## 9. Reproducibility

The candidate that matters - E_hardneg - was already certified bit-identical
(mission rerun max|Δ| ≈ 3×10⁻⁸, replay). PB1/PB2 retrains here share the identical
seeded training path (seed 42) and are diagnostic only - they are NOT deployment
candidates. No candidate is promoted, so no new reproduction artifact is required.

## 10. Real-prevalence operational estimate (0.121% fraud)

| OP | Alerts/1K | Alerts/10K | Alerts/100K | Fraud caught | Legit alerts | Precision |
|---|---|---|---|---|---|---|
| E P2 (deployed) | 116.4 | 1163.7 | 11637 | 120 | 11516 | 1.03% |

(E-fixed-threshold gate-CS1 cuts ~117 alerts/100k but at 99.48% recall of the absent
population - documented above as a trade.)

## 11. Feature findings (mission #8)

| Feature | Status | Root cause / finding |
|---|---|---|
| merchant_id/city_id/card_id codes | RETAINED | codes carry genuine signal; PB removal worsens both overall and absent FPR |
| merchant_age + prior volume (novelty) | VALIDATED, NOT DEPLOYED | causal; the only lever that improves absent FPR (~5 pts gated) at ~1 pt overall FPR cost ungated, or 1/193 absent-fraud cost gated at P2 |
| err / has_state | REPAIRED (pre-mission) | state-parse fix; no longer constant |
| mcc_online | DEAD const-0 | MCC 5967-5969 absent from corpus; defensive only |

## 12. Remaining risks

- **Absent-merchant over-flagging is structural** at the ≥99%-recall operating point:
  no model configuration separates "established merchant not in the 5% train sample"
  from risky novelty at the score level required. ~64% of the population is old rare
  merchants (age ≥ 10y), only ~10% genuinely new.
- **Label latency UNVERIFIED** (unchanged, carried on all candidates).
- **Channel drift** persists (online→chip/swipe migration); every candidate recalls
  each channel ≥ 0.97 but online FPR is high (0.42) for all.

## 13. Recommendation

**KEEP E_HARDNEG** (deployed model unchanged; `altman_native_E_hardneg_cert_20260904`
@ 0.018758).

The conditional entity+behavior architecture does not produce a dominating model.
Identity codes are informative, not the memorization culprit; removing them (true
Model B) is strictly worse. The only absent-improving configuration - gating a
novelty-feature model onto absent rows - is a Pareto trade (better absent FPR, worse
absent recall at the deployed threshold, extra serving surface: second artifact +
merchant table + parity gate) and does not meet the mission's dominance bar. The
honest answer to the Phase-3 question is: at ≥99% recall, PS-14 cannot yet separate
"unknown merchant identity" from "fraud-like behavior" - the missing information is
merchant establishment history, which exists in the full corpus but was not in the
trained feature set, and adding it does not resolve the frontier trade.
