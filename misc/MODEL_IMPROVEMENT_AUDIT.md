# PS-14 MODEL IMPROVEMENT AUDIT

*Date: 2026-09-04 · Discipline: chronological, validation-only development, single final evaluation of the untouched 2018–2020 test window.*

## Executive Summary

The objective was to **move the ROC/PR curve left**: reach ≥99% fraud recall with materially fewer false positives than the deployed model — without test-set tuning.

| System | Locked threshold | Test recall | Test FPR | Prod alerts/1K* | Status |
|---|---|---|---|---|---|
| v2 (deployed until 9/4) | 0.78471 | 44.6% | 1.3% | 13.4 | superseded |
| v3 (deployed) | 0.027328 | 99.54% | 15.87% | 159.8 | baseline |
| Improved candidate (pre-campaign) | 0.065153 | 97.8% | 7.1% | 72.4 | not deployed |
| **E_hardneg (campaign winner)** | **0.026346** | **99.43%** | **9.48%** | **95.9** | **candidate — HOLD** |

\* reweighted to true production prevalence (0.121%), not the fraud-enriched sample.

**The curve moved left.** At matched ~99.5% test recall the required FPR fell from **15.9% (v3) → ~10.3% (E_hardneg)**; at ≥99% recall from ~13%+ → **8.0%**. The milestone (≥99% recall at <7% FPR) was **not** reached: the best honest point is **99% recall at 8.0% FPR / 99.4% recall at 9.5% FPR**. Improvement came from **hard-negative mining on validation-internal data**, not from threshold manipulation: the locked thresholds were chosen on validation by a pre-registered rule (`data/test_lock.json`) and the untouched test was scored exactly once at the end.

All numbers in this document are independently recomputed from stored artifacts (`reports/mission_results.json`, `models/model_records/mission_*/`); the winning configuration reran **bit-identical** (run-1 == run-2).

---

## 1. Locked test (never touched during development)

Machine-readable lock: `data/test_lock.json`. The 2018-01-01..2020-02-29 window (3,782,053 full-corpus rows, 4,578 fraud, true prevalence 0.121%) was only ever *scored* — by shipped artifacts at the start and by candidates **once, at the end**, at thresholds pre-locked on validation. No test label fed feature engineering, tuning, early stopping, calibration, class weighting, hard-negative mining, segment definition, or the choice of winner. Choosing a variant by test metrics would have voided it; the winner was selected by a validation-only rule before its test numbers were printed.

## 2. Baseline freeze (before any experiment)

`reports/mission_baseline.json` records, from the independently verified `reports/v_verification_final.json`:

* v3 deployed: thr 0.027328 → TP 4,557 / FN 21 / FP 5,940 / TN 31,478 → recall 0.99541, FPR 0.15875, precision 0.4341 (enriched), 159.8 alerts/1K at production prevalence.
* Improved candidate (pre-campaign): thr 0.065153 → test recall 0.97750, FPR 0.07133.

The campaign's **A_baseline rerun reproduces the improved candidate exactly** (max |Δ| 3×10⁻⁸ on validation probabilities).

## 3. Feature forensic audit (every feature, all windows)

Full scan in `reports/mission_feature_audit.json` (48 features × train/val/test means, std, zero-fraction, PSI, flags). Findings:

| Finding | Severity | Detail / action |
|---|---|---|
| `err` was constant-1 in training vs 0 in production (the earlier bug) | FIXED (HIGH) | NaN-safe parse in both the shared derivation and the retrain raw build; now 1.7% rate, non-constant, train==prod semantics. |
| `mcc_online` is **constant 0** in the whole corpus (MCC 5967–5969 absent) | dead feature | harmless to the trees (never splits); drop at the next feature-schema bump. |
| `has_state` is **constant 1** (every merchant has a state) | dead feature | makes `is_online_or_no_state ≡ is_online`; redundant. |
| `chip` / `is_swipe` shift massively train→test (PSI ≈ 2) | drift (flagged) | swipe 0.81→0.17, chip 0.07→0.71 — the payment-channel migration; quantified in §7. |
| `user_tx_count`, `card_tx_count`, diversity counters PSI > 1 | scale caveat | cumulative counts are functions of history length AND of the 5%-legit training sampling; trees are partly rank-robust but see §12 for the live-data implication. |
| `hour_sin/cos`, `is_night`, flags, one-hot MCC groups "near-constant" | benign | low-cardinality by design (hours, weekday, booleans) — expected, not bugs. |
| 1,057 rows with negative `user_avg_amt` | minor | small fraction; not investigated as a driver. |

No other constant/NaN/string-conversion bug of the `err` class was found: every feature is derived by the shared production module (`src/privacy_layer/native_features.py`), computed from shifted pre-event context, so the derivation cannot silently diverge between training and production.

## 4. Canonical feature contract

The 48-vector contract lives in one place (`ALTMAN_NATIVE_FEATURES` in `src/privacy_layer/native_features.py`), consumed by training (`retrain_native_consistent.py`), the verification cache, and the runtime engine. A promoted model's feature schema is verified against the contract before deployment; parity tests (`scripts/feature_parity_test.py`) re-ran **11/11 PASS** on the deployed path. A full machine-readable `feature_contract.json` per feature (definition, source columns, missing policy, causality requirement, train vs prod implementation) remains a **recommended next step** for the schema-bump that drops the dead features — the current shared-module design already guarantees one implementation.

## 5. Experiments (all validation-only decisions, single final test eval)

Shared frame: full corpus streamed once (keep all fraud + 5% legit, rng 42) → 879,196 train / 174,530 val / 192,372 test rows. Every model: XGB+LGB+CatBoost ensemble (seed 42), RobustScaler fit on train only, windows train<2016 | val 2016–17 | test ≥2018 untouched. Thresholds locked from **validation**: P1 = min val FPR at val recall ≥0.99; P2 = min val FPR at val recall ≥0.995.

| Model | Features | Train window | Weighting | Test ROC-AUC | Test PR-AUC | Val-locked op. point (P2) test recall / FPR | Status |
|---|---|---|---|---|---|---|---|
| v3 (reference) | 48 native | <2016 | spw capped 20 | 0.97224 | 0.78749* | 0.9954 / 0.1587 | deployed |
| A_baseline (=pre-campaign improved) | 48 native | <2016 | spw 10 | 0.98611 | 0.66524 | 0.9965 / 0.1200 | reference |
| B_recent6 | 48 native | 2010–2015 | spw 10 | 0.96875 | 0.43257 | 0.9747 / 0.1614 | REJECTED (drops recall) |
| C_recw | 48 native | <2016 | recency weight 1×→4× | 0.98338 | 0.61938 | 0.9902 / 0.1078 | worse than A |
| D_channel | 48 native | <2016 | per-channel sub-ensembles | 0.97799 | 0.59011 | 0.9937 / 0.1717 | REJECTED (no gain) |
| **E_hardneg** | **48 native** | **<2016** | **hard negatives (OOS train) ×3** | **0.98573** | **0.59931** | **0.9943 / 0.0948** | **WINNER** |
| F_nohash | 45 (no entity codes) | <2016 | spw 10 | 0.97993 | 0.52828 | 0.9932 / 0.1137 | generic probe — see §12 |

\* v3 PR-AUC is on the 1%-legit enriched test; the campaign rows use 5%-legit sampling, so PR-AUC magnitudes are not comparable across the two sampling regimes — recall/FPR denominators are comparable (same 4,578 test fraud).

**Winner by the pre-registered rule** (min val FPR at val recall ≥0.995; tie-break P1 then AUC): **E_hardneg**, val P2 thr 0.026346 (val recall 0.99504, val FPR 0.09945).

## 6. Recall/FPR frontier on the untouched test (capability, not selection)

Minimum test FPR needed to reach each recall (full frontier in `reports/mission_results.json`):

| Recall | v3 | A_baseline | C_recw | E_hardneg | F_nohash |
|---|---|---|---|---|---|
| 95% | — | 5.2% | 6.2% | **4.8%** | 6.4% |
| 97% | — | 6.5% | 7.6% | **5.9%** | 7.6% |
| 98% | — | 7.4% | 8.8% | **6.6%** | 8.4% |
| 99% | ~13% | 8.9% | 10.7% | **8.0%** | 10.2% |
| 99.5% | **15.9%** | 11.1% | 13.5% | **10.3%** | 12.1% |

Hard-negative mining shifts the frontier left across the whole high-recall region; the win is largest where the business operates (≥98–99.5% recall). No tested configuration reaches ≥99% recall below 7% FPR — **we did not prove impossibility, only that none of the six tested configurations did it** (see §17 below for the honest framing).

## 7. Temporal / channel drift — quantified, not asserted

By-year at the deployed model's point and the winner's locked point (full per-year table in `reports/mission_results.json`):

* 2018: 2,491 fraud — E_hardneg recall 0.9936 @ FPR 9.9% (v3: recall 0.9932 @ FPR 16.1%).
* 2019: 2,087 fraud — E_hardneg recall 0.9952 @ FPR 9.5% (v3: recall 0.9981 @ FPR 16.0%).
* 2020 Jan–Feb: **0 fraud** in the corpus (row noise only) — no recall to report.

Drift driver: **channel shift** — 2016 validation fraud was ~86% online; 2018–19 test fraud is almost entirely chip/swipe in-person, and the legit channel mix itself migrated (swipe 81%→17%, chip 7%→71%, §3). This is why thresholds locked on the 2016-dominated validation window needed a recall buffer (pre-registered P2 = 99.5%) to land ≥99% on test.

## 8. Leakage audit

| Check | Result | Evidence |
|---|---|---|
| Target leakage (label used to build a feature) | PASS | all historical features use shifted (pre-event) context; `err`-style label-feeders absent after the fix |
| Temporal leakage (future rows in features) | PASS | expanding context is shifted; windows strictly chronological; val_max < test_min |
| Test contamination (test labels seen during dev) | PASS | training/selection code paths read val only; test labels loaded only by the final-eval step |
| Threshold leakage | PASS | P1/P2 locked on validation, pre-registered; test never modified a threshold |
| Early-stopping contamination | PASS | XGB/CB early stopping would use val; here iterations are fixed (no ES on test) |
| Label-availability timing (when fraud labels exist) | **UNVERIFIED** | the raw dataset records the outcome but not its confirmation delay; entity fraud-rate features assume past labels are known at scoring time (previously documented caveat) |

## 9. Production parity

| Component | Train | Production | Exact match | Evidence |
|---|---|---|---|---|
| Feature derivation | shared `native_features.py` | same module (live ingest) | YES | one implementation by construction |
| Feature order / schema | 48-vector contract | engine contract same list | YES | identical `ALTMAN_NATIVE_FEATURES` |
| Preprocessing | RobustScaler fit on train only | same scaler artifact at serve | YES (schema unchanged) | replay max|Δ| 3×10⁻⁸ |
| Model replay | stored members | — | YES | recomputed probs == stored probs |
| Live-path parity drill | — | — | NOT RE-RUN this session | previously 11/11 on the deployed path; E_hardneg is a new artifact set that still needs the deploy-cut parity rerun (§16) |

The winner uses the **same 48-feature schema as the deployed v3** — the model changes (weights + hard-negative mining), not the feature contract.

## 10. Robustness (best candidates)

Probed the winner with hostile inputs: NaN rows score deterministically (no crash); **inf raises at the model boundary** (defense-in-depth gap — the production engine sanitizes via `_f2` before scoring, so this cannot silently approve/reject live); unseen merchant/city hash codes fall to a benign cold bucket (~mean score, no false spike); entity fraud-rate = 0.9 correctly raises the score. Missing/unseen-category handling at the service level was previously exercised by the 10 canonical suites + 22/22 regression (green) and the drift-detector degraded/fail-open path.

## 11. Reproducibility

Every run records: shared-frame metadata (sampling + windows), seed 42, hyperparameters, weights, thresholds, metrics, artifact hashes. **Run-2 of the winner is bit-identical to run-1** (val AUC, P1, P2 all exactly equal); replay of the shipped members reproduces stored probabilities to 3×10⁻⁸. The full corpus stream + cache build is deterministic for a given file and chunk size.

## 12. Generic live-data learning (the "not just this dataset" requirement)

* **Behavioral, entity-relative features generalize.** Velocity, amount-vs-own-average, entity fraud rates are causal aggregates over *that entity's* history — they transfer to any card/merchant/city stream and are exactly what a live deployment computes.
* **The hash-code features are the memorization risk.** `merchant_id/city_id/card_id` are stable hash codes; trees learn fixed effects of *this dataset's* entities. Probe **F_nohash** (codes removed) loses ~1.2–2.2 pts of FPR on the 2018–20 test — because those merchants recur from training. For genuinely unseen live entities the codes are useless noise; a general live model should drop or cap them (schema bump), accepting F_nohash-class performance.
* **Cold-start discipline.** Unseen entities hash into cold buckets, never silently into a trained entity; entity fraud rates default to 0.001 with no history; velocity starts at 0. The engine degrades to rules-only (tagged, never silent) when features/models fail — audited previously.
* **Scale caveat for live velocity.** Cumulative count features were trained on a 5%-legit sampled universe; a live deployment counting *all* ingested history produces larger absolute counts. Trees tolerate monotone scale shifts only partly — the recommended deploy-time check is a production-distribution comparison (the monitoring stack's PSI path) and, for a fully generic model, log/cap or full-corpus-context features at the next schema bump.

## 13. Recommendation

**HOLD FOR FURTHER VALIDATION** — one decision, stated plainly.

The offline evidence is promotion-grade (feature parity by construction, causal audit clean, chronology valid, leakage audit PASS with one documented UNVERIFIED caveat, reproducibility bit-identical, frontier improved). What stands between this candidate and production is *operational*, not ML: (1) package `mission_E_hardneg` as a governance-recorded model version with locked threshold **0.026346** and rollback archive; (2) re-run the live-path parity drill and the alert-workflow/audit/monitoring suites against the new artifact set; (3) record a promotion reason and obtain approval. The threshold change policy (Part 19) requires a new version + revalidation, which this audit constitutes.

**Why not DEPLOY right now:** no automatic deployment without the operational cut; the candidate's PR-AUC is slightly below A_baseline's even though its high-recall operating point is better, so the choice is a documented business-objective trade-off (we optimized "min FPR at recall ≥99.5%", per the locked rule).

## 14. What was NOT achieved

No tested configuration reached **≥99% recall at <7% FPR** on the untouched test. The honest frontier is: 99.4% recall at 9.5% FPR (winner's locked point) and 99% recall at 8.0% FPR (curve minimum). The claim that this is impossible has **not** been proven — it bounds what six configurations did, not the theoretical limit.

## 15. Records produced

`data/test_lock.json` · `reports/mission_baseline.json` · `reports/mission_feature_audit.json` · `reports/mission_results.json` · `models/model_records/mission_{A_baseline,B_recent6,C_recw,D_channel,E_hardneg,F_nohash}/` (models.joblib, scores.npz, result_val.json, config.json) · `data/_mission_frame.npz` + `_mission_frame_meta.json`. Public website / audit report / fix log / certification **not modified** — per instruction, they await these independently reproduced results.
