# PHASE 6 — Exposure-Normalized Features Campaign (E_hardneg_exposure_norm_v1)

## 0. Production lock (unchanged through the entire campaign)

| Item | Value |
|---|---|
| Deployed model | `altman_native_E_hardneg_cert_20260904` |
| Locked threshold | **0.018758** |
| Status | **DEPLOYED — untouched** (guard verified before every job) |
| Model SHA-256 | see `reports/phase6_production_snapshot.json` (verified vs manifest) |
| Snapshot | `reports/phase6_production_snapshot.json` |
| Backup (hash-verified) | `BACKUPS/E_hardneg_cert_20260904/` |

## 1. Steps executed

1. **Snapshot + backup (verified).** Every production artifact re-hashed and
   compared to the manifest record — all matched; byte-identical backup created.
2. **Experiment tree + safety guard.** `experiments/phase6_exposure_normalization/`
   with `phase6_guard` aborting on production drift or any write under
   `models/production`.
3. **Control gate (Step 6).** Certified E_hardneg artifacts rescored on the
   locked mission-frame validation — reproduced exactly (AUC 0.995594,
   PR-AUC 0.914337, P2 thr 0.018758 → recall 0.9950 / FPR 0.115304).
4. **Frame.** `data/_p6_frame.npz` (1,246,098 rows; all fraud + 5% legit, rng 42)
   = **row-identical to the mission frame**, with six exposure-rate features
   appended (columns 48-53). Formula audit in `experiments/.../features/p6_derive.py`.
5. **Causality (Step 8).** Future-perturbation test over 461,131 pre-2011 rows:
   removing future rows and flipping future labels both leave every earlier
   feature **bit-identical** (max |Δ| = 0). **PASS.**
6. **Candidates (Step 10)** trained with the certified recipe on identical
   validation rows:

| Candidate | Features | Val P2 (rec≥0.995) |
|---|---:|---:|
| control_E_hardneg | 48 native | thr 0.018758 · FPR 0.1153 |
| V_norm (counts→rates) | 42 native + 6 rates | thr 0.038742 · FPR 0.1048 |
| **V_rawplus (counts+rates)** | **48 native + 6 rates** | **thr 0.029894 · FPR 0.0992** |
| V_Cnorm (coverage+rates) | 42 native + 6 rates | FPR 0.2800 (cross-density probe) |

   Selection rule (locked pre-test): minimize overall val FPR at the val P2
   recall floor with no subgroup regression >1pp → **V_rawplus frozen at 0.029894**.
7. **Reproducibility (Step 14).** Winner retrained twice from scratch; both runs
   match the frozen predictions (max|Δp| = 3.0e-8 ≤ 1e-6 documented OpenMP
   tolerance) with identical metrics at the locked threshold (Δ = 0). **PASS.**
8. **Final test ONCE (Step 15).** Untouched 2018-2020 window, one shot:

| Model | Threshold | Recall | FPR | Precision | TP | FP | FN | TN |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E_hardneg (certified) | 0.018758 | **0.9967** | 0.1099 | 0.1811 | 4563 | 20634 | 15 | 167160 |
| **V_rawplus (frozen)** | 0.029894 | 0.9908 | **0.0915** | **0.2089** | 4536 | 17175 | 42 | 170619 |

| Segment (test) | E_hardneg rec / FPR | V_rawplus rec / FPR |
|---|---:|---:|
| seen merchant | 0.9965 / 0.1016 | 0.9902 / 0.0838 |
| absent@5 | 1.0000 / 0.6451 | 1.0000 / **0.5881** |
| genuinely new | 1.0000 / 0.6341 | 1.0000 / **0.5827** |

## 2. Honest interpretation

- **What improved (the Phase-5/6 target).** On the absent/genuinely-new merchant
  segments — the entire reason this campaign exists — the frozen candidate
  holds **100% recall while cutting FPR ≈ 9%** (0.6451→0.5881; 0.6341→0.5827).
  Overall FPR falls 16.7% and precision rises 15%.
- **The cost.** At its locked val-P2 operating point the candidate's overall
  test recall is 99.08% vs the certified model's 99.67% (FN 15 → 42). The
  val→test recall gap of the candidate (0.9950 → 0.9908) is real, documented
  temporal drift on ~27 additional hard rows.
- **Alert-cap policy (Step 12).** Neither model reaches recall ≥ 99% at val
  FPR ≤ 2% (≈95.4% @ FPR2; ≈98.3% @ FPR5) — a genuine model-separation limit,
  not a threshold artifact. Reported, not papered over.
- **V_Cnorm probe (Step 11).** Exposure normalization does **not** repair the
  Strategy-C cross-density count-scale mismatch (val FPR 0.28). Normalization
  helps only when train and scoring densities match — which is the production
  deployment shape (matched density, matched semantics).

## 3. Verdict

**E_hardneg remains the deployed production model.** The frozen candidate is
**CERTIFICATION_PENDING** (not promoted): the FPR/precision gains are real and
broad, but the recall trade at the locked point (FN 15→42) requires the
step-18 independent certification and an explicit high-recall operating-point
decision before any promotion. Rule 3 of the brief is honored — no candidate
replaced E_hardneg.

Deliverables:
`experiments/phase6_exposure_normalization/reports/phase6_exposure_normalization.json`,
`frontier_summary.json`, `final_test_one_shot.json`, `val_comparison.json`,
`causal_test.json`, `control_reproducibility.json`, `reports/phase6_production_snapshot.json`,
`BACKUPS/E_hardneg_cert_20260904/`.
