# PS-14 — Checks #21 & #22: Explainability / Audit Trails + Incident Response

**Date:** 2026-09-03 · **Scope:** deployed model `altman_lean_15feat_20260830_200346`
(XGB+LGB+CB ensemble, 15 lean features, RobustScaler, Platt calibrator) — the model
production actually loads, not the audit's inline 25-feature candidate.

---

## Check #21 — Model Explainability & Decision Audit Trails: **PASS (with 1 HIGH finding)**

### Methodology (documented, not assumed)

Native **TreeSHAP** — the SHAP algorithm as implemented *inside* each booster:
`XGBoost Booster.predict(pred_contribs=True)`, `LightGBM predict(pred_contrib=True)`,
`CatBoost get_feature_importance(type='ShapValues')`. Same algorithm as
`shap.TreeExplainer`, no extra dependency, exact per-model decomposition.
Ensemble attribution = weighted average of per-model SHAP vectors (documented
approximation: contributions live in logit space, the ensemble raw is a weighted
*mean of probabilities*). SHAP is an attribution method, not a causal claim — the
report never calls importance causal.

### Explanation validation — all checks PASS on 122 transactions

| Check | Result | Evidence |
|---|---|---|
| Explanations match the actual prediction | PASS | Per-model additivity: `sum(contribs)+base → sigmoid == predict_proba`. Max abs error: XGB 4.9e-07, LGB 0.0, CatBoost 1.7e-15 (float32 precision floor). |
| Same feature values as the model | PASS | Hand-replicated map→scale→predict→calibrate reproduces the engine's final score to **2.2e-16** max diff. SHAP is computed from the identical scaled matrix X. |
| No future/post-event info | PASS | All 15 features derive from the incoming pre-transaction dict + historical entity rates; the mapper is stateless (no time dimension, no label, no future rows). Full provenance table recorded per feature. |
| Tested on known examples | PASS | Crafted high-risk txn → top drivers `mcc_n`(+42.8), `has_zip`(+16.7), `merch_tx_count`(+6.3); low-risk txn → same three at ~1/6 scale. See finding below. |
| Reproducibility | PASS | Fresh engine instance, same 120 transactions: score diff **0.0**, SHAP diff **0.0** (tree models are deterministic; no nondeterminism in the deployed path). Same txn + model + feature version + config ⇒ same decision. |

### Decision audit trail — immutable and pseudonymous

122 decision records appended to the **hash-chained append-only audit DB** (DB-4,
`append_audit_event`, event_type=`model_decision`), each containing: hashed txn id,
prediction timestamp, model version, feature-schema version, calibrated score,
decision threshold (production `band_of` 0.85), decision, band, and top
contributing/suppressing features with values. Chain re-verification over all
1,344 entries: **OK**. Immutability proven for real: a direct SQL `UPDATE` on a
written row is blocked by the append-only `RAISE(ABORT)` trigger
(`sqlite3.IntegrityError`). Payloads store only hashed ids and numbers — no raw
transaction data.

### HIGH finding: the deployed decision path is inert

Measured on the actual deployed pipeline (any input batch): the lean model's raw
ensemble scores **saturate** (batch p50 raw = 0.945), and the loaded
`models/artifacts/calibrator.joblib` — a `PlattCalibration` fit by
`src/train_compare.py` on the **PS-14 synthetic model family's** out-of-archetype
scores, *not* on this model — compresses the entire output into **[0.027, 0.254]**.
Production's `band_of()` flags `score*100 >= 85`; the deployed calibrated score
**never reaches 0.85**, so:

- The crafted high-risk transaction (5× amount, online, elevated merchant/city
  fraud rates, failed auths) scores **0.254 → decision "allow"**.
- At the audit-locked threshold 0.111, the deployed scores would flag **53% of
  transactions** — the opposite failure (see FPR incident).

Two coupled defects: the calibrator was fit for a different model family, and the
raw ensemble is so saturated that no single monotone map can be well-behaved.
Root cause is the same as check #16's FAIL — the artifact set was never validated
end-to-end. **Recommendation:** fit a calibrator on the deployed model's own OOF
scores (or drop calibration and lock a threshold on raw), then re-validate.

Second finding: the **top contributors of every decision are constant features**
(`mcc_n`, `has_zip` — hardcoded 0 in the runtime mapper). Their SHAP values are
large because the training model weights the zero-value state, so production's
"signal" is dominated by features that never vary in production — concrete
evidence of the runtime-mapper train/serve skew recorded in check #16.

---

## Check #22 — Fraud-Model Incident Response System: **PASS (two open actions)**

### Kill switch (implemented for real)

`src/risk_engine/kill_switch.py`, wired into `AltmanEnsembleEngine.predict*`:

- **Authorization:** `mallory` cannot arm (PermissionError) and the attempt is
  logged to the audit chain; `ops-oncall` can. Same for disarm.
- **Quarantine:** when armed, the engine raises `KillSwitchActiveError` — the
  caller degrades to the documented rules-only safe state; no score is served.
- **Restore:** disarm returns service; post-disarm predictions are **byte-identical
  to baseline** (parity proof). Activation/deactivation events (`arm`,
  `arm_denied`, `disarm`, `disarm_denied`) are appended to the audit chain
  (4+ events verified).

### Incident simulations (stage-by-stage, verified)

| Incident | Severity | Detection | Safe response | Rollback/containment | Recovery | Stages |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Corrupted model artifact | Critical | PASS (governance hash / XGBoostError) | PASS (quarantine) | PASS (restore from known-good + verify) | PASS (predictions identical) | **6/6** |
| Broken feature pipeline | High | **FAIL** (engine scores all-default input silently) | PASS (input-schema guard added) | PASS | PASS | 4/6 |
| FPR exceeds limit | Critical | PASS (53.3% alert rate at threshold 0.111 — measured) | PASS (quarantine) | PASS (RCA = calibrator mismatch) | PASS | **6/6** |
| Offline/online feature mismatch | Critical | PASS (parity gate FAIL) | PASS (promotion blocked) | PASS (V1 stays) | **OPEN** (city_state fix) | 5/6 |
| Critical data-quality failure | Critical | PASS (DQ monitor RED, 0.51% invalid timestamps) | PASS (no retrain on degraded stream) | PASS | **OPEN** (timestamp-source fix) | 5/6 |

Two genuine system properties were proven:

1. **Corrupted-artifact lifecycle** (6/6): tamper → detected at load → armed →
   engine refuses → restored from known-good → predictions byte-identical.
2. **FPR incident is real, not synthetic** (6/6): the alert rate was *measured* on
   the actual deployed score distribution at the audit-locked threshold — 53% —
   far above the 1% operational limit. RCA ties it to the check-#21 calibrator
   mismatch. This is the strongest possible illustration that the deployed
   decision path is not trustworthy as-is.

### Incident Response Matrix

| Incident | Detection | Severity | Automatic Action | Manual Action | Recovery | Tested? |
|---|---|---|---|---|---|---|
| Data leakage discovered | governance + audit | Critical | quarantine | RCA, fix pipeline | revalidate | covered by prior checks |
| Feature pipeline corruption | **GAP (fixed by guard)** | High | no score on malformed input | wire guard into main.py | revalidate | 4/6 |
| Production/offline mismatch | parity gate FAIL | Critical | block promotion | fix `city_state` row-group bug | revalidate | 5/6 (open) |
| Model artifact corruption | hash verify FAIL | Critical | quarantine (kill switch) | restore known-good | verify parity | 6/6 |
| Sudden FPR spike | monitor RED | Critical | quarantine | RCA (calibrator) | re-fit + revalidate | 6/6 |
| Sudden recall degradation | monitor RED (labels) | Critical | quarantine | RCA | revalidate | covered by monitoring |
| Unexpected alert explosion | alert-rate monitor | Critical | quarantine | RCA | revalidate | 6/6 (FPR case) |
| Major feature drift | PSI monitor | Warning | investigate | RCA | revalidate | covered by drift monitor |
| Incorrect fraud labels | DQ monitor | High | no retrain on bad stream | fix label source | revalidate | covered by monitoring |
| Model serving failure | health checks | Critical | restart / quarantine | RCA | verify parity | covered by #18 |
| Security compromise | access logs | Critical | quarantine + revoke | forensics | re-audit | documented |

### Open actions (same three blockers as before, plus one new)

1. **Fix `city_state` row-group reset** in `train_altman_fullscale.py` (line ~218) — parity incident recovery.
2. **Fix invalid-timestamp source** in the recent production window — DQ incident recovery.
3. **Wire the input-schema guard into `main.py`** feature ingestion (featpipe mitigation; currently demonstrated in the IR harness).
4. **Re-fit the calibrator on the deployed model's own OOF scores** (or drop it) — FPR incident root cause; also unblocks the threshold question from check #19.

---

## Verdict

- **Check #21:** explanations are validated and reproducible; the decision audit
  trail is immutable, hash-chained, and pseudonymous. The audit itself exposed
  that **the deployed model's decisions are not trustworthy** — the cross-model
  calibrator compresses every score below the production decision band, and the
  top drivers are constant features. Explainability here didn't just describe the
  model; it demonstrated why the deployed path needs the parity/calibration fixes.
- **Check #22:** the system now **detects, contains, explains, and recovers** —
  the kill switch is authorization-controlled, logged, tested for real, and
  restores known-good behavior bit-for-bit. Two recoveries remain open because
  the underlying bugs (`city_state`, timestamp source) are real production
  defects, not simulation artifacts.

**PS-14 remains not production-ready** for the same three documented reasons
(calibrator/decision-path mismatch, `city_state` cache bug, DQ timestamp source) —
but every one of them is now a concrete, tracked incident with a defined response.

---

**Files:** `scripts/explainability_audit.py`, `scripts/incident_response.py`,
`src/risk_engine/kill_switch.py` (new), `src/risk_engine/altman_ensemble.py`
(kill-switch wiring), `reports/explainability_audit.json`,
`reports/incident_response.json`. Regression: `smoke` and `risk_engine` suites
**PASS** with the engine changes live. ~1,390 events now in the audit chain, all
hash-verified.