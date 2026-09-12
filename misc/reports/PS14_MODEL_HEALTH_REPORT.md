# PS-14 Model Health Report — Rolling Validation, Monitoring & Retraining Safety

**Date:** 2026-09-03 · **Dataset:** `credit_card_transactions-ibm_v2.csv` (24,386,900 rows, 29,757 fraud / 0.122%) · **Report:** `reports/rolling_validation_and_monitoring.json`

---

## 1. Rolling Forward Validation — Honest Protocol

Five expanding temporal windows. Each window:

- **Train (fit):** rows `0 → 75%` of the window's training period
- **Internal holdout (early stopping):** last 25% of the training period — the forward validation period **never** influences training or early stopping
- **Forward validation period (eval):** the next ~4.06M chronologically-later rows, completely untouched
- **Threshold:** selected ONCE on window 1's validation period (FPR < 0.9%, max recall) and **locked at 0.1110** for all subsequent windows — it is never reselected per window

> **Protocol fix applied in this run:** the previous version early-stopped on each window's own validation period and reselected the threshold per window, which pinned FPR at the 0.9% target by construction (all windows showed ≈0.89–0.90% FPR). With the locked threshold, forward FPR is an honest out-of-sample measure.

### Results

| Window | Train rows | Eval period rows | ROC-AUC | PR-AUC | Recall | Precision | FPR | Alerts | Alerts/10K | Fraud rate (eval) |
|-------:|-----------:|-----------------:|--------:|-------:|-------:|----------:|----:|-------:|-----------:|------------------:|
| 1 (calibration) | 4,064,483 | 4,064,483 | 0.9917 | 0.7018 | 88.2% | 11.0% | 0.900% | 41,033 | 101.0 | 0.126% |
| 2 | 8,128,966 | 4,064,483 | 0.9948 | 0.8021 | 90.7% | 11.6% | 0.841% | 38,604 | 95.0 | 0.121% |
| 3 | 12,193,449 | 4,064,483 | 0.9950 | 0.8260 | 91.3% | 15.6% | 0.609% | 29,270 | 72.0 | 0.123% |
| 4 | 16,257,932 | 4,064,483 | 0.9956 | 0.8194 | 92.7% | 13.4% | 0.784% | 36,720 | 90.3 | 0.130% |
| 5 | 20,322,415 | 4,064,483 | 0.9948 | 0.8008 | 91.6% | 12.1% | 0.797% | 36,792 | 90.5 | 0.119% |

**All confusion-matrix identities verified for every window** (TP+FN = fraud, TN+FP = legitimate, Alerts = TP+FP, FPR/Recall/Precision recomputed from the matrix within rounding tolerance).

### Stability verdict

- **ROC-AUC:** 0.9917–0.9956 (spread 0.004) — stable
- **PR-AUC:** 0.7018–0.8260 — window 1 (smallest training set) is the weakest; windows 2–5 are stable in the 0.80–0.83 band
- **Recall:** 88.2–92.7% — all windows above the 85% operational floor
- **FPR (locked threshold):** 0.609–0.900% — every window below the 1.5% operational ceiling and below the 1% target
- **Fraud rate:** 0.119–0.130% — no material shift in base rate
- **Alert volume:** 72.0–101.0 per 10K transactions

**Honest wording:** performance is *stable across the four out-of-sample forward periods* with a fixed operating threshold. This is a descriptive stability statement based on the five-window experiment; it is **not** a claim of "no drift" as a statistical conclusion — a formal drift test was not run (see §5).

---

## 2. Post-Deployment Monitoring System (deliverable)

### Model performance monitoring

Track per day/week/month and segmented by merchant, city, MCC, online/chip, and customer segment:

- ROC-AUC, PR-AUC, recall, precision, FPR
- Fraud rate, false-positive rate, alert volume

Implementation note: these are computed on rolling windows with **labels** — in production, labels arrive with confirmation latency, so performance metrics lag real time by the label-latency period (see §6).

### Feature monitoring (drift)

PS-14 operational thresholds (documented as such — not claimed to be industry standards):

| Level | PSI range | Action |
|-------|-----------|--------|
| HEALTHY | PSI ≤ 0.02 | None |
| INFO | 0.02 < PSI ≤ 0.10 | Log |
| WARNING | 0.10 < PSI ≤ 0.25 | Alert, monitor, prepare retrain |
| CRITICAL | PSI > 0.25 | Investigate pipeline; retrain if drift is genuine |

Per feature, monitor: missing-value rate, mean/median, min/max, distribution (PSI vs reference), new/unseen categories, spikes, stale values.

### Data-quality alerts

Automatically flag: missing-data spikes, invalid timestamps, unexpected categories, feature-calculation failure, sudden fraud-rate change, sudden alert-volume change, model-score distribution change, production/offline feature mismatch. Overall status is a single **HEALTHY / WARNING / CRITICAL** roll-up; CRITICAL blocks silent operation.

### Reference windows

- Reference distribution: first 20% of rows (chronological)
- Production window: last 20% of rows
- Comparisons are recomputed each monitoring cycle, preserving history so a degradation can be traced back to when/which feature drifted.

---

## 3. Drift Detection Results (first 20% vs last 20% of dataset)

| Summary | Count |
|---------|------:|
| HEALTHY | 21 |
| WARNING | 2 |
| CRITICAL | 0 |

| Feature | PSI | Level | Ref mean | Prod mean | Note |
|---------|----:|-------|---------:|----------:|------|
| `city_popularity` | 0.114 | WARNING | 4.874 | 4.981 | distribution shift toward denser cities |
| `user_merch_diversity` | 0.246 | WARNING | 3.730 | 2.526 | near the CRITICAL threshold; user behavior changed |

Caveat: z-scores are 100s–1000s because n ≈ 4.9M per group — statistical significance is uninformative at this sample size. PSI magnitude is the operational signal; both WARNINGs reflect genuine distribution shifts, not data errors.

No formal statistical drift test was performed; PSI thresholds above are PS-14 operational thresholds.

---

## 4. Data-Quality Monitoring Results (last 100K rows)

| Status | Issue | Detail |
|--------|-------|--------|
| WARNING | Missing Zip | 7.8% |
| **CRITICAL** | Invalid timestamps | 0.51% (Year/Month/Day out of valid range) |

Fraud rate: reference 0.1126% vs recent 0.1181% (ratio ≈ 1.05× — no base-rate shift).

---

## 5. Retraining Rules (PS-14 operational thresholds)

| Rule | Condition (PS-14 threshold) | Current | Triggered |
|------|-----------------------------|--------:|:---------:|
| Recall degradation | Recall < 85% | 91.6% | No |
| FPR exceedance | FPR > 1.5% | 0.797% | No |
| PR-AUC drop | PR-AUC < 0.60 | 0.8008 | No |
| Feature drift CRITICAL | any PSI > 0.25 | 0 features | No |
| Feature drift WARNING | > 5 features PSI > 0.10 | 2 features | No |
| Fraud distribution shift | rate ratio > 2× or < 0.5× | 1.05× | No |
| **Data quality degradation** | overall health = CRITICAL | CRITICAL | **Yes** |
| Scheduled retrain | 90 days / 10M new transactions | Scheduled | No |

**1 of 8 rules triggered: Data quality degradation.** All performance rules are within limits; the model-health status is driven entirely by the data-quality flag.

---

## 6. Model Health Summary

| Field | Value |
|-------|-------|
| Current model status | **RETRAINING RECOMMENDED** (precautionary — data quality, not performance) |
| Last validation date | 2026-09-02 (this report) |
| Latest performance (window 5) | AUC 0.9948 · PR-AUC 0.8008 · Recall 91.6% · FPR 0.797% · 90.5 alerts/10K |
| Drift status | 0 CRITICAL / 2 WARNING / 21 HEALTHY |
| Data-quality status | **CRITICAL** (0.51% invalid timestamps in recent window) |
| Production-parity status | PARTIAL (offline expanding window vs production sliding-window tracker — requires real-data validation) |
| Retraining recommendation | **RETRAIN** — but only after the data-quality issue is diagnosed |
| Reason | Rule "Data quality degradation" triggered. The correct sequence is: (1) fix the data pipeline / investigate the invalid-timestamp source, (2) re-run DQ checks to HEALTHY/WARNING, (3) then decide whether retraining is needed. Do not retrain on a degraded data stream — that bakes the artifact in. |

### Retraining safety protocol (never auto-deploy)

1. Current model → Candidate model (retrained on fresh, quality-checked data)
2. Same untouched evaluation protocol (causal features, internal early-stop holdout, locked threshold from calibration period)
3. Compare performance + stability across the 5 rolling windows (not just one test)
4. Security/data-quality checks (same suite as production audit)
5. Leakage verification (future-row perturbation test, permutation test)
6. **Human approval required before deployment**
7. Canary deployment (10% traffic) → full rollout, with rollback

A candidate model is never deployed solely because its training/validation score is higher; it must pass the identical leakage, temporal, robustness, and parity gates as the incumbent.

---

## 7. Limitations (honest)

1. **No formal statistical drift test** — PSI-based PS-14 operational thresholds only; "no drift" is not claimed as a statistical conclusion.
2. **Label latency** — rolling performance metrics assume fraud labels are available at eval time; real-time label confirmation latency is not modeled in this dataset (same UNVERIFIED status as the main forensic audit).
3. **Production parity** — the offline expanding-window feature computation is architecturally different from the production sliding-window entity tracker; parity is PARTIAL until verified on identical historical transactions.
4. **Window 1 is weaker** (PR-AUC 0.70) — the smallest training set; this is a cold-start-of-training effect, not a deployment signal.
5. **Entity space is small** — ~1,200 users, 9 cards in the dataset; merchant/city-level generalization is the primary signal.
6. The DQ CRITICAL (invalid timestamps) shows the monitor works, but the *source* of the bad timestamps is not diagnosed in this dataset — that is the immediate operational follow-up.

---

## 8. Verdict

The model is **performance-stable across all four out-of-sample forward windows** under an honest, locked-threshold protocol (AUC ≥ 0.9917, FPR ≤ 0.900% everywhere, recall ≥ 88.2%). The monitoring system detects degradation drivers (feature drift and data quality) and blocks silent operation.

However, the **data-quality CRITICAL** in the most recent window means the system is currently in **RETRAINING RECOMMENDED / WARNING** status operationally: fix the invalid-timestamp source before retraining, and re-verify DQ to HEALTHY/WARNING before any model replacement.