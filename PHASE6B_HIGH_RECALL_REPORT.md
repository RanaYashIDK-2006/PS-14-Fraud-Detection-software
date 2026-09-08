# PHASE 6B — High-Recall Validation Frontier & False-Negative Forensics

## 1. Preflight (reports/phase6b_preflight.json)
- Production `altman_native_E_hardneg_cert_20260904` @ 0.018758 — artifact hashes match manifest: **True**
- Frozen candidate dir hashed: 4 files; frame sha256 recorded
- Final-test labels never loaded by selection code: **True**

## 2. Reproduction on validation (174,530 rows / 3,834 fraud)
| Model | Point | Recall | FPR | Precision | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E_hardneg | 0.018758 | 0.995044 | 0.115304 | 0.1624 | 3815 | 19682 | 19 | 151014 |
| V_rawplus | 0.029894 | 0.994784 | 0.099235 | 0.1838 | 3814 | 16939 | 20 | 153757 |

Rescore vs frozen scores max|Δ| = 2.98e-08 — **deterministic**.

## 3. Dominance at matched validation recall floors (exact constraints)

| Floor | E thr | E rec | E FPR | V thr | V rec | V FPR | ΔFPR abs | ΔFPR rel | Δprec | Δalerts/1k |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.990 | 0.053558 | 0.990089 | 0.0708 | 0.070777 | 0.990089 | 0.0683 | 0.0026 | +3.6% | +0.0068 | +2.50 |
| 0.995 | 0.018758 | 0.995044 | 0.1153 | 0.029894 | 0.995044 | 0.0992 | 0.0161 | +13.9% | +0.0215 | +15.72 |
| 0.997 | 0.012503 | 0.997131 | 0.1363 | 0.014667 | 0.997131 | 0.1378 | -0.0015 | -1.1% | -0.0013 | -1.45 |
| 0.998 | 0.008030 | 0.998174 | 0.1647 | 0.010307 | 0.998174 | 0.1622 | 0.0025 | +1.5% | +0.0016 | +2.41 |
| 0.999 | 0.001255 | 0.999218 | 0.3748 | 0.003990 | 0.999218 | 0.2443 | 0.1304 | +34.8% | +0.0276 | +127.58 |
## 3b. Step-4 rounding trap — frozen point vs exact floor
- Exact ≥99.5% floor point: **tV* = 0.0298937337** (recall 0.995044340, FN 19)
- Phase-6 frozen point 0.029894 sits 2.66e-07 above it — the gap holds **1 val fraud row(s) and 0 legit alerts** → rounding artifact only.

## 4. False-negative forensics (validation, ≥99.5% points)

- E_hardneg FN: **19** | V_rawplus FN: **19** | both missed: 16 | E-caught/V-missed: **3** | V-caught/E-missed: 3

E-caught/V-missed set vs all-val-fraud baseline:
- under-covered 33.3% vs 3.7%; genuinely-new 0.0% vs 1.4%
- median full history depth 61520 vs 20802; zero-frame-history 33.3% vs 6.2%
- median amount 22.40 vs 83.19
- median score drop (V−E) -0.04161 (mean -0.03680)

## 5. Segment analysis (reports/phase6b_segment_analysis.json)
| Segment | n | fraud | E rec | E FPR | V rec | V FPR |
|---|---:|---:|---:|---:|---:|---:|
| seen_merchants | 172,055 | 3,641 | 0.9948 | 0.1064 | 0.9951 | 0.0908 |
| under_covered_established | 2,146 | 143 | 1.0000 | 0.7768 | 0.9930 | 0.7219 |
| genuinely_new | 340 | 55 | 1.0000 | 0.7544 | 1.0000 | 0.7298 |
| zero_frame_history | 2,916 | 237 | 1.0000 | 0.8458 | 0.9958 | 0.8070 |
| low_frame_history_1_5 | 7,049 | 150 | 0.9867 | 0.2928 | 0.9867 | 0.2835 |
| well_observed_frame_gt100 | 132,341 | 3,096 | 0.9968 | 0.1004 | 0.9964 | 0.0774 |
| full_hist_0_new | 340 | 55 | 1.0000 | 0.7544 | 1.0000 | 0.7298 |
| full_hist_1_5 | 908 | 123 | 1.0000 | 0.7503 | 1.0000 | 0.7185 |
| full_hist_6_50 | 4,066 | 148 | 1.0000 | 0.5362 | 0.9932 | 0.5026 |
| full_hist_gt50 | 169,216 | 3,508 | 0.9946 | 0.1012 | 0.9949 | 0.0857 |
| channel_chip | 120,641 | 527 | 0.9715 | 0.0653 | 0.9734 | 0.0611 |
| channel_swipe | 29,721 | 234 | 0.9829 | 0.1030 | 0.9786 | 0.0800 |
| channel_online | 24,168 | 3,073 | 1.0000 | 0.4173 | 1.0000 | 0.3432 |

## 6. Verdict
- Step-13 (validation only): **CANDIDATE QUALIFIED FOR FINAL CERTIFICATION**
- Step-9 hypothesis: **PARTIALLY SUPPORTED**
- Promotion: **OUTCOME B — E_HARDNEG REMAINS DEPLOYED**
- Reason: Validation frontier at the exact >=99.5% floor favors V_rawplus (FPR 0.1153 -> 0.0992, rel -13.9%; worst segment recall delta -0.0070). The frozen point itself misses the exact floor by 2.7e-7 (one val fraud row, zero alerts — rounding artifact, see knife_edge). However the Phase-6 one-shot untouched evaluation of this operating point recorded recall 0.9908 (< 0.995) with FN 42 against E_hardneg's 0.9967 / FN 15: V_rawplus's val->test recall transfer is -0.0043 while E_hardneg's is +0.0016, and validation offers NO FN-count analogue (val FN 20 vs 19) — the gap is distribution shift, not a val-visible weakness. The >=99.5% recall requirement is NOT maintained on untouched data. Promotion rejected (Step 12): E_hardneg stays deployed.

## 7. Next action
No final-test rescoring (consumed once in Phase 6; rule 2). Re-freezing at tV* would be cosmetic (1 val row, 0 alerts). Real blocker = val->test recall transfer of the candidate; next model target = the segments where the transfer degrades (under-covered established merchants, swipe) plus a drift-robust threshold margin.