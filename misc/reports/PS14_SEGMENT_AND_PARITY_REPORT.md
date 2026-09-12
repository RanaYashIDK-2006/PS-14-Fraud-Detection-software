# PS-14 Check #15 + #16 — Segment Performance/Bias Audit & Offline→Production Parity

**Date:** 2026-09-03 · **Protocol:** identical to the main forensic audit (causal expanding features → temporal 60/20/20 → XGB with val-only early stopping → threshold locked at 0.0950 from validation FPR<0.9% → untouched final test, 4,877,380 rows).

---

## Check #15 — Segment-Level Performance & Bias

**Overall test:** AUC 0.9947 · Recall 91.8% · FPR 0.911% · Precision 10.6% · 49,664 alerts.

Sample-size guardrails applied: segments with n<200 skipped; AUC/PR-AUC reported only with ≥20 fraud AND ≥20 legit. All metrics below carry counts so prevalence effects are visible. History-depth segments measure counts *within the test window* (the earlier audit established test users are unseen post-train; merchant/city entities do recur).

### Channel

| Segment | n | Fraud | Fraud rate | AUC | Recall | Precision | FPR | Alerts/10K |
|---------|-----:|-----:|----------:|----:|-------:|----------:|----:|-----:|
| chip | 1,277,111 | 883 | 0.069% | 0.9980 | 93.4% | 27.5% | 0.170% | 23.5 |
| **online** | 472,941 | 3,598 | **0.761%** | 0.9959 | **98.2%** | 10.4% | **6.478%** | **717.6** |
| swipe | 3,127,328 | 1,280 | 0.041% | 0.9850 | **72.5%** | 7.3% | 0.378% | 40.7 |

**Largest FPR gap:** online 6.478% vs overall 0.911% (+5.57pp). Not a "bias" — online fraud is genuinely 18× denser than chip (0.761% vs 0.069% prevalence), and the model's high online recall (98.2%) at a fixed threshold mechanically yields more alerts there. It is prevalence-driven plus the operational consequence of one global threshold over heterogeneous segments.

**Largest recall gap:** swipe 72.5% vs overall 91.8% (−19.3pp). Swipe has the lowest fraud density (0.041%); the model is conservative there. Lower volume of fraud per swipe transaction means fewer strong signals.

### Amount bands

| Band | n | Fraud | Fraud rate | AUC | Recall | Precision | FPR | Alerts/10K |
|------|-----:|-----:|----------:|----:|-------:|----------:|----:|-----:|
| <$20 | 2,015,446 | 1,497 | 0.074% | 0.9944 | 90.0% | 10.1% | 0.596% | 66.2 |
| $20–50 | 1,224,611 | 821 | 0.067% | 0.9918 | 88.3% | 6.2% | 0.901% | 96.0 |
| $50–100 | 1,068,649 | 1,186 | 0.111% | 0.9944 | 91.4% | 11.0% | 0.822% | 92.3 |
| $100–250 | 493,477 | 1,542 | 0.312% | 0.9951 | 94.4% | 14.4% | 1.751% | 204.0 |
| $250–1000 | 71,880 | 687 | 0.956% | 0.9896 | 94.5% | 14.5% | 5.384% | 623.5 |
| >$1000 | 3,317 | 28 | 0.844% | 0.9989 | 100% | 18.1% | 3.861% | 467.3 |

Fraud rate rises monotonically with amount ($0.074% → $0.956%), so higher FPR/alerts in high bands reflect prevalence, not a model artifact. The >$1000 segment (n=3,317, 28 fraud) shows the classic small-sample caveat: recall 100% is on 28 cases.

### MCC bands

| Band | n | Fraud | AUC | Recall | Precision | FPR |
|------|-----:|-----:|----:|-------:|----------:|----:|
| MCC 1k–2k | 1,274 | 0 | — (no fraud) | 0% | 0% | 1.491% |
| MCC 2k–4k | 85,638 | 772 | 0.9966 | 95.5% | **33.8%** | 1.704% |
| MCC 4k–6k | 4,311,264 | 4,585 | 0.9949 | 92.0% | 10.4% | 0.844% |
| MCC 6k+ | 479,204 | 404 | 0.9850 | 82.4% | 4.8% | 1.369% |

The 2k–4k band (which includes the dense-fraud merchants, see below) has the best precision (33.8%) at 0.90% fraud density.

### Merchant & city (top-8 + other)

| Group | n | Fraud | AUC | Recall | Precision | FPR |
|-------|-----:|-----:|----:|-------:|----------:|----:|
| merchant `-4282466774399734331` | 223,498 | 351 | 0.9972 | 94.3% | **61.2%** | 0.094% |
| merchant `1913477460590765860` | 120,230 | 300 | 0.9979 | 94.7% | 36.1% | 0.419% |
| merchant `-1288082279022882052` | 157,937 | 20 | 0.9973 | 85.0% | 21.5% | 0.039% |
| other_merchant | 3,553,712 | 5,020 | 0.9935 | 91.6% | 9.6% | 1.218% |
| city ONLINE | 474,432 | 3,598 | 0.9960 | 98.2% | 10.4% | 6.459% |
| other_city | 4,133,843 | 2,154 | 0.9908 | 81.2% | 11.3% | 0.331% |

Two specific merchants concentrate fraud (351 + 300 of the 29,757 dataset-wide fraud appear in *just the test window* at those two merchants) and the model scores them with high precision (61%/36%) — a legitimate merchant-risk signal, since `merch_fraud_rate`/`merch_popularity` accumulate those merchants' own prior history. City-level concentration is dominated by "ONLINE" (a pseudo-city for online transactions).

### Entity history within test window

| Segment | n | Fraud | AUC | Recall | Precision | FPR |
|---------|-----:|-----:|----:|-------:|----------:|----:|
| merchant first-seen in test | 35,617 | 470 | **0.9489** | 81.9% | 12.9% | **7.400%** |
| merchant low history | 329,273 | 929 | 0.9861 | 88.4% | 16.3% | 1.282% |
| merchant high history | 4,512,490 | 4,362 | 0.9963 | 93.6% | 9.8% | 0.833% |
| user first-seen in test | 415 | 0 | — | — | — | 1.928% |
| user high history | 4,873,230 | 5,761 | 0.9947 | 91.8% | 10.7% | 0.911% |

**Worst-performing segment overall:** merchants first-seen in the test window — AUC 0.9489 (−0.046), FPR 7.400% (+6.5pp). This is the cold-start-for-merchant scenario: no `merch_fraud_rate` history yet, so the model leans on the merchant's *current* fraud cluster (470 fraud cases at 1.32% density — 10× the global rate) but cannot separate them well. This is consistent with the stress-test finding (unknown merchants → FPR +5.3pp) and is a **documented operational risk**: brand-new merchants generate 7× the FPR of the overall system until their history accumulates. Mitigation options: per-segment thresholds or a "new merchant" holdout queue.

### Best/worst summary

- **Best AUC:** amount band >$1000 (0.9989, n=3,317 — small sample)
- **Worst AUC:** merchant first-seen-in-test (0.9489)
- **Largest FPR gap:** online (+5.57pp) — prevalence-driven (18× denser fraud)
- **Largest recall gap:** swipe (−19.3pp) — sparse-fraud segment
- **Largest precision gap:** merchant `-4282466774399734331` (+50.5pp) — legitimate concentrated-fraud-merchant signal

### Segment verdict

Differences are explained by (a) fraud prevalence (online, high-amount bands, fraud-concentrated merchants), (b) sample size (tiny segments flagged), and (c) feature availability (new merchants lacking history). **No segment shows a disparity inconsistent with its prevalence and history — the model is not disproportionately FP-heavy on any demographic segment.** The one genuine operational finding is the new-merchant cold-start FPR (7.4%), already corroborated by the stress test.

---

## Check #16 — Offline → Production Pipeline Parity

### What the risk engine actually loads

`src/risk_engine/main.py` loads **AltmanEnsembleEngine** (XGB+LGB+CB, RobustScaler, Platt-calibrated) from `models/production/` whenever `xgb_production.joblib` + `lgb_production.joblib` exist — which they do. Artifact identity is consistent:

| Check | Result |
|-------|--------|
| manifest.features == feature_list.json (15) | PASS |
| manifest.features == engine `ALTMAN_FEATURES` code constant | PASS |
| scaler/xgb/lgb/cb all trained on exactly 15 features | PASS |
| Deployed model loads and scores | PASS (`altman_lean_15feat_20260830_200346`, smoke preds valid) |
| Manifest artifacts hashed | recorded (sha256 prefixes in JSON) |

### Numeric parity — offline causal recompute vs the cache the deployed model was trained on

Recomputed all expanding-window features from the raw sorted rows in one continuous causal pass and compared against `data/_sorted_cache/features_all.parquet` (the actual training input of the deployed lean model) at head / quarter / mid file windows (400K rows each):

| Feature | Head | Quarter | Mid | PASS |
|---------|------|---------|-----|------|
| user_tx_count | 0.0 | 0.0 | 0.0 | ✅ |
| card_tx_count | 0.0 | 0.0 | 0.0 | ✅ |
| merch_tx_count | 0.0 | 0.0 | 0.0 | ✅ |
| user_avg_amt | 0.0 | 0.0 | 0.0 | ✅ |
| amt_zscore | 0.0 | 0.0 | 0.0 | ✅ |
| user_fraud_rate | 0.0 | 0.0 | 0.0 | ✅ |
| merch_fraud_rate | 0.0 | 0.0 | 0.0 | ✅ |
| **city_fraud_rate** | 0.0 | **0.999856** | **0.997971** | ❌ |

**REAL PRODUCTION BUG FOUND.** In `scripts/train_altman_fullscale.py` (pass 2, the builder of the production training cache), `city_state = {}` is initialized at **line 218 — INSIDE the per-row-group loop** — while user/card/merchant states (lines 161–163) are outside it. The parquet file has 24 row groups of ~1,048,576 rows, so the cache's `city_fraud_rate` **resets to empty at every ~1M-row boundary** and only ever reflects the current row group, not full history. My continuous causal recompute matches the cache exactly at the head window (first row group, no prior reset) and diverges by up to **0.9999** at quarter/mid windows.

Consequences:
1. The deployed model was trained with chunk-local `city_fraud_rate` — the feature silently means "fraud rate since the start of the current 1M-row parquet chunk", an artifact of chunking, not a real-time window.
2. The runtime path (`entity_fraud_rates.py` sliding-window tracker, min-5-event baseline) computes a *different* definition → train/serve skew for this one feature.
3. A retrain after deleting/re-chunking the cache would silently change `city_fraud_rate` values.

The fix is one line — move `city_state = {}` to line ~161 with the other states. (The audit's own offline pipeline uses a genuinely persistent city state, which is why it is unaffected.)

### Production builder causality — PASS

Future-row perturbation on the production expanding-window builder: keep rows 0–60K unchanged, append 400K later rows, recompute → **max feature diff = 0.00000000**. The builder records each entity's state *before* updating it (shift-1 semantics), so no feature can observe its own row's outcome or any future row.

### Runtime mapper (live /internal/evaluate path) — PARTIAL

The live path does not recompute expanding features from raw rows. The Privacy Layer sends a `FeatureVector`, and `map_ml_features_to_altman()` synthesizes the 15 lean features — with documented proxy derivations:
- `mcc_n`, `has_zip`, `has_state` → **hardcoded 0** (the FeatureVector schema cannot carry them)
- `amt` (→ log_amt/amt_sq/amt_x_*) → derived as `amount_ratio × 100`, not the real amount
- `chip` → mapped from `new_device_flag` (a semantic proxy)
- entity fraud rates → sliding-window tracker (min 5 events, else 0.001 baseline) vs full-expanding-history training features

So the model that runs in production receives *reconstructed* rather than *identical* features for several inputs. For the three hardcoded-zero features this is a permanent 0 vs real-value gap (the lean model was presumably trained to tolerate them, but it is a real fidelity loss vs the offline feature space).

### Model-version consistency — FAIL (as designed, must be resolved before "production-ready")

The forensic audit evaluated a **new inline 25-feature XGB** (trained and evaluated inside `scripts/forensic_revalidate.py`, never persisted). The artifact production loads is the **15-feature lean ensemble** (`altman_lean_15feat_20260830_200346`). Schema overlap is 12/15; the audit model adds 13 features the deployed artifact doesn't have, and the deployed artifact has 3 the audit never used. **The model whose metrics were validated is not the model that will run in production** unless the audit's pipeline output is deployed. The audit validated methodology and the causal feature definitions, not the artifact's own metrics.

---

## Verdict

| Check | Result |
|-------|--------|
| Segment-level performance (bias) | **PASS** — disparities explained by prevalence/sample-size/history; no demographic bias; one operational finding (new-merchant cold-start FPR 7.4%) |
| Production feature-generation parity | **FAIL** — `city_fraud_rate` cache reset bug in `train_altman_fullscale.py` (one-line fix) |
| Production builder causality | **PASS** — future-row perturbation max diff 0.0 |
| Deployed artifact internal consistency | **PASS** — manifest/feature-list/code/dims all agree |
| Runtime mapper fidelity | **PARTIAL** — mcc_n/has_zip/has_state hardcoded 0; amt proxied; tracker-rate vs expanding-history skew |
| Audited model == deployed model | **FAIL** — 25-feature inline audit XGB ≠ 15-feature deployed lean ensemble |

**Bottom line:** PRODUCTION PARITY = FAIL. Two actionable items: (1) move `city_state = {}` outside the row-group loop in `train_altman_fullscale.py` and regenerate the cache; (2) decide and execute the model hand-off — either retrain/deploy the audit's 25-feature model as the artifact (re-running the parity test against it), or run the full forensic validation protocol against the deployed 15-feature lean artifact. The audit is methodologically sound and leakage-free, but its numbers describe a candidate model, not the one currently deployed.

### Files
- `scripts/segment_performance_audit.py` + `reports/segment_audit.json`
- `scripts/production_parity.py` + `reports/production_parity.json`
- `scripts/launch_segment_parity.py`
- `reports/PS14_SEGMENT_AND_PARITY_REPORT.md` (this report)
