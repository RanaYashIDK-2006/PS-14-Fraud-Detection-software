# PHASE 8 — ROBUST MODEL IMPROVEMENT & FORWARD-GENERALIZATION REPORT

* Generated: 2026-09-06T12:15:34Z UTC
* Production: **altman_native_E_hardneg_cert_20260904** @ threshold **0.018758** — **untouched** (guard verified)
* Final test (>=2018): **not loaded, not scored** (all thresholds on validation only)

## 1. What Phase 7 proved

* V_rawplus dev-robust at >=99.5% (val FPR 0.0992 vs E 0.1153).
* Forward windows 2014/2015/2016: rawplus >=99.5% with equal/better FPR.
* Exposure ablation: raw 0.1051, norm 0.1048, raw+norm 0.0992 -> raw counts matter.
* Val FN parity E=19/V=19 at exact >=99.5% points.
* Channel: V cuts chip/swipe/online FPR with no asymmetric recall loss.

## 2. What Phase 7 disproved

* Training-coverage scaling is the absent-merchant FPR lever (0.68-0.79 at all levels).
* Swipe is the asymmetric failure channel (no asymmetric loss at the floor).
* A conditional raw-for-under-covered / rates-elsewhere architecture is needed.

## 3. Hypotheses tested (register)

* **H1**: Full-corpus merchant depth (label-free, causal) reduces G4/absent FPR at >=99.5%.
* **H2**: Dropping the 3 UNVERIFIED label-dependent fraud-rate cols costs little on dev.
* **H3**: Any fixed-val win transfers to the 2014/2015/2016 forward windows.

## 4. Candidate families

| Family | Feature set | Purpose |
|---|---|---|
| **A — E_hardneg control** | 48 native (certified artifacts) | unchanged production reference |
| **B — V_rawplus** | 48 native + 6 exposure-normalized rates (frozen Phase 6) | Phase 6/7 dev-robust baseline |
| **C_cov** | V_rawplus (54) + 4 label-free real-corpus merchant-depth cols | test the coverage hypothesis (H1) |
| **D_safe** | C_cov minus the 3 UNVERIFIED label-dependent fraud-rate cols (37–39) | label-latency-safe variant (H2) |

Real-depth columns (all causal: years strictly < row year, from the full-corpus merchant volume table, no labels): `real_prior_tx_log`, `real_active_years`, `real_age_years`, `under_covered`.

## 5. Validation results (fixed val 2016–2017, 174,530 rows / 3,834 fraud; all points at exact ≥99.5% recall)

| Model | floor thr | Recall | FPR | Precision | TP | FP | FN | TN | AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_hardneg | 0.01875816 | 0.995044 | 0.115304 | 0.162361 | 3815 | 19682 | 19 | 151014 | 0.995594 | 0.914337 |
| V_rawplus | 0.02989373 | 0.995044 | 0.099235 | 0.18382 | 3815 | 16939 | 19 | 153757 | 0.995986 | 0.920436 |
| C_cov | 0.03107652 | 0.995044 | 0.112112 | 0.166216 | 3815 | 19137 | 19 | 151559 | 0.991742 | 0.851371 |
| D_safe | 0.0941377 | 0.995044 | 0.134883 | 0.142144 | 3815 | 23024 | 19 | 147672 | 0.986851 | 0.806129 |

Controls reproduced exactly against certified records: E floor fpr 0.115304 (cert 0.1153), V floor fpr 0.099235 (cert 0.0992). AUC/PR-AUC/Brier/ECE: E 0.9956/0.9143/brier 0.0084; V 0.9960/0.9204/0.0113; C_cov 0.9917/0.8514/0.0296; D_safe 0.9869/0.8061/0.0408. Both new families rank worse than the control on the validation AUC/PR-AUC — the added real-depth columns and the removal of the fraud-rate columns both cost discrimination.

## 6. Forward-validation results (dev-only expanding windows, strictly pre-2018)

E_hardneg (raw_only) and V_rawplus (rawplus) rows are the accepted Phase-7 measurements on the identical windows/frames; C_cov and D_safe were retrained per window in Phase 8 with the identical recipe. Each row is at that window's exact ≥99.5% recall point.

| Window | Model | Recall | FPR | AUC |
|---|---:|---:|---:|---:|
| FV-2014 | E_hardneg | 0.9952471482889734 | 0.042320369149637445 | 0.999069 |
| FV-2014 | V_rawplus | 0.9952471482889734 | 0.04230838377179841 | 0.999074 |
| FV-2014 | C_cov | 0.995247 | 0.060107 | 0.99838 |
| FV-2014 | D_safe | 0.995247 | 0.115419 | 0.994058 |
| FV-2015 | E_hardneg | 0.9951234379762267 | 0.17352171515073284 | 0.991089 |
| FV-2015 | V_rawplus | 0.9951234379762267 | 0.15445275003813794 | 0.991871 |
| FV-2015 | C_cov | 0.995123 | 0.252027 | 0.991221 |
| FV-2015 | D_safe | 0.995123 | 0.157668 | 0.989395 |
| FV-2016 | E_hardneg | 0.9952500698519139 | 0.08209396354006573 | 0.996653 |
| FV-2016 | V_rawplus | 0.9952500698519139 | 0.07198590783344762 | 0.997143 |
| FV-2016 | C_cov | 0.99525 | 0.077046 | 0.994504 |
| FV-2016 | D_safe | 0.99525 | 0.112702 | 0.992424 |

## 7. Coverage-segment results (fixed val, at each model's ≥99.5% floor)

| Model | Segment | Recall | FPR | Precision | n_fraud | n_legit |
|---|---:|---:|---:|---:|---:|---:|
| E_hardneg | G1_genuinely_new | 1.0 | 0.756272 | 0.191571 | 50 | 279 |
| E_hardneg | G2_under_covered | 1.0 | 0.776835 | 0.084167 | 143 | 2,003 |
| E_hardneg | G3_well_covered | 0.994507 | 0.106375 | 0.168137 | 3,641 | 168,414 |
| E_hardneg | G4_deep_real_zero_frame | 1.0 | 0.442623 | 0.084746 | 5 | 122 |
| V_rawplus | G1_genuinely_new | 1.0 | 0.731183 | 0.19685 | 50 | 279 |
| V_rawplus | G2_under_covered | 0.993007 | 0.721917 | 0.089421 | 143 | 2,003 |
| V_rawplus | G3_well_covered | 0.995056 | 0.090782 | 0.191571 | 3,641 | 168,414 |
| V_rawplus | G4_deep_real_zero_frame | 1.0 | 0.418033 | 0.089286 | 5 | 122 |
| C_cov | G1_genuinely_new | 1.0 | 0.817204 | 0.179856 | 50 | 279 |
| C_cov | G2_under_covered | 0.993007 | 0.711433 | 0.090619 | 143 | 2,003 |
| C_cov | G3_well_covered | 0.995056 | 0.103816 | 0.171649 | 3,641 | 168,414 |
| C_cov | G4_deep_real_zero_frame | 1.0 | 0.344262 | 0.106383 | 5 | 122 |
| D_safe | G1_genuinely_new | 1.0 | 0.856631 | 0.17301 | 50 | 279 |
| D_safe | G2_under_covered | 0.993007 | 0.771343 | 0.084173 | 143 | 2,003 |
| D_safe | G3_well_covered | 0.995056 | 0.126118 | 0.145719 | 3,641 | 168,414 |
| D_safe | G4_deep_real_zero_frame | 1.0 | 0.311475 | 0.116279 | 5 | 122 |

## 8. Channel-segment results (fixed val, at each model's ≥99.5% floor)

| Model | Channel | Recall | FPR | Precision | n_fraud | n_legit |
|---|---:|---:|---:|---:|---:|---:|
| E_hardneg | chip | 0.969639 | 0.065271 | 0.06119 | 527 | 120,114 |
| E_hardneg | swipe | 0.982906 | 0.103028 | 0.070379 | 234 | 29,487 |
| E_hardneg | online | 1.0 | 0.41735 | 0.258735 | 3,073 | 21,095 |
| E_hardneg | other | 0.0 | 0.0 | 0.0 | 0 | 0 |
| V_rawplus | chip | 0.973435 | 0.0611 | 0.065334 | 527 | 120,114 |
| V_rawplus | swipe | 0.978632 | 0.080035 | 0.088451 | 234 | 29,487 |
| V_rawplus | online | 1.0 | 0.343209 | 0.297973 | 3,073 | 21,095 |
| V_rawplus | other | 0.0 | 0.0 | 0.0 | 0 | 0 |
| C_cov | chip | 0.973435 | 0.055481 | 0.071478 | 527 | 120,114 |
| C_cov | swipe | 0.978632 | 0.059721 | 0.115075 | 234 | 29,487 |
| C_cov | online | 1.0 | 0.507798 | 0.222923 | 3,073 | 21,095 |
| C_cov | other | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D_safe | chip | 0.971537 | 0.044941 | 0.086633 | 527 | 120,114 |
| D_safe | swipe | 0.982906 | 0.043952 | 0.150721 | 234 | 29,487 |
| D_safe | online | 1.0 | 0.774117 | 0.158378 | 3,073 | 21,095 |
| D_safe | other | 0.0 | 0.0 | 0.0 | 0 | 0 |

## 9. Threshold stability (validation, around each ≥99.5% floor)

| Model | floor thr | gap to next fraud score (rel %) |
|---|---:|---:|
| E_hardneg | 0.01875816 | 3.22% |
| V_rawplus | 0.02989373 | 12.35% |
| C_cov | 0.03107652 | 8.36% |
| D_safe | 0.0941377 | 0.44% |

Detailed recall/FPR/FN curves around each floor are in `reports/phase8_threshold_analysis.json`.

## 10. Label-latency status

The three label-dependent native features `user_fraud_rate`, `merch_fraud_rate`, `city_fraud_rate` remain **UNVERIFIED** (no real-world label-availability timing established). **C_cov** retains them (research only). **D_safe** drops them, so it is the only family that is label-latency-safe for production. No Phase-8 candidate depends on a new label-dependent feature.

## 11. Feature parity status

Real-depth columns are built from `data/_merchant_volume.npz` (full-corpus tx counts by merchant-code × year, label-free, name-hash space identical to frame codes; alignment asserted). Causality: only volume years strictly < the scored row's year are used. Cold start (no real history): `real_prior_tx_log=0`, `real_active_years=0`, `real_age_years=0`, `under_covered=0`. Feature order is fixed and identical for train/val/forward windows. Production parity requires the same volume table at inference time (documented requirement).

## 12. Reproducibility (train twice from scratch)

| Model | max |dp| | metrics identical at floor | pass |
|---|---:|---|---|
| C_cov | 0.00e+00 | True | True |
| D_safe | 0.00e+00 | True | True |

## 13. Does any candidate genuinely beat E_hardneg?

* **C_cov**: val recall 0.995044 at floor thr 0.03107652; FPR 0.112112 vs V_rawplus 0.099235 (+13.0%) and vs E_hardneg 0.115304 (-2.8%).
* **D_safe**: val recall 0.995044 at floor thr 0.0941377; FPR 0.134883 vs V_rawplus 0.099235 (+35.9%) and vs E_hardneg 0.115304 (+17.0%).

#### Segment evidence (fixed val, at each ≥99.5% floor)

| Segment FPR | E_hardneg | V_rawplus | C_cov | D_safe |
|---|---:|---:|---:|---:|
| G1_genuinely_new | 0.756 | 0.731 | 0.817 | 0.857 |
| G2_under_covered | 0.777 | 0.722 | 0.711 | 0.771 |
| G3_well_covered | 0.106 | 0.091 | 0.104 | 0.126 |
| G4_deep_real_zero_frame | 0.443 | 0.418 | 0.344 | 0.311 |

The coverage hypothesis (H1) is **partially true on G4 but not overall**: the real-depth columns do cut the deep-real-history/zero-frame FPR (E 0.443 → C_cov 0.344 → D_safe 0.312 at recall 1.0 on that segment) — but they simultaneously **worsen the genuinely-new-merchant FPR** (V 0.731 → C_cov 0.817 → D_safe 0.857) and the aggregate FPR never beats V_rawplus. The label-latency hypothesis (H2) is **rejected on dev**: dropping the fraud-rate columns costs AUC 0.9960 → 0.9869 and raises overall FPR 0.099 → 0.135. The transfer hypothesis (H3) is **rejected**: C_cov's small fixed-val gain does not hold on forward windows (FV-2015 FPR 0.252 vs V 0.154).

* No candidate materially beat V_rawplus at ≥99.5% recall on validation, so none advanced to the freeze step.

## 14. Is independent certification authorized?

* **NO** — outcome C (NO WIN (no candidate materially beats V_rawplus at >=99.5%)). No candidate qualifies for certification in this phase.

## 15. Is final-test scoring authorized?

* **NO.** Phase 8 itself does not authorize final-test scoring. A future one-shot requires: candidate frozen + threshold frozen + feature contract frozen + independent certification + explicit authorization + one clean evaluation.

## 16. Exact next action

* Keep E_hardneg. V_rawplus remains dev-robust candidate for certification review.

## Appendix — outcome decision record

```json
{
  "outcome": "C",
  "outcome_label": "NO WIN (no candidate materially beats V_rawplus at >=99.5%)",
  "best_candidate": null,
  "evidence": {
    "val_fpr_comparison": {
      "E_hardneg": 0.115304,
      "V_rawplus": 0.099235,
      "C_cov": 0.112112,
      "D_safe": 0.134883
    },
    "val_recall_comparison": {
      "E_hardneg": 0.995044,
      "V_rawplus": 0.995044,
      "C_cov": 0.995044,
      "D_safe": 0.995044
    },
    "forward_pass": true,
    "reproducibility_pass": true,
    "label_safety": "D_safe drops UNVERIFIED cols 37-39; C_cov keeps them (research only)"
  },
  "production_untouched": true,
  "final_test_untouched": true,
  "next_action": "Keep E_hardneg. V_rawplus remains dev-robust candidate for certification review."
}
```

---
*Integrity: `production_modified=False`, `final_test_accessed=False`. All thresholds selected on the locked validation population only (exact ≥99.5% constraint, no pre-rounding). Forward windows strictly pre-2018.*
