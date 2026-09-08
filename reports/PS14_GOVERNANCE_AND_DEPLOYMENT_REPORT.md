# PS-14 — Governance & Deployment Report (Checks 17 + 18)

**Date:** 2026-09-03
**Scope:** Model versioning/governance for the deployed artifact, and a real deployment/rollback/DR test.
**Companion data:** `reports/model_governance.json`, `reports/deployment_test.json`
**Code:** `src/risk_engine/model_governance.py`, `src/risk_engine/altman_ensemble.py` (wired integrity check), `scripts/model_governance.py`, `scripts/deployment_test.py`

---

## Check 17 — Model Versioning & Governance

### What was built

1. **`src/risk_engine/model_governance.py`** — permanent model-record schema plus artifact integrity verification:
   - SHA-256 hash of every artifact file in the model directory, compared against the recorded manifest.
   - Verification covers model binaries, scaler, feature list, manifest, drift reference, and the engine's supporting files (13 checks in total).
   - Promotion checklist with documented gates (leakage → data quality → validation → untouched test → robustness → parity → security → approval).

2. **Engine wiring** — `AltmanEnsembleEngine` now runs an integrity verification at load time *when a governance record exists for the model version*. Mismatch raises `ModelIntegrityError`, which the risk engine's existing load path turns into a visible, safe failure (model unavailable rather than silently serving a corrupted/wrong artifact).

3. **`scripts/model_governance.py`** — generates the permanent record for the deployed artifact and re-verifies it.

### Governance record — deployed model `altman_lean_15feat_20260830_200346`

| Field | Value |
|---|---|
| Feature schema version | `altman_lean_v1` (15 features) |
| Training dataset SHA-256 | `4ea90501…3562396ac` (`features_all.parquet`, 24,386,900 rows) |
| Algorithm | xgb_lgb_ensemble_lean (+ scaler + calibrator) |
| Training seed | 42 |
| Calibration | `calibrator.joblib` (Platt, `models/artifacts`) |
| Artifact verification | **PASS — 13/13 checks, 0 failures** |
| Record created | 2026-09-03 06:38 UTC |

The artifact files are cryptographically hashed in the record; the engine verifies the recorded hashes against the bytes it actually loads.

### Promotion gate — NOT APPROVED

| Gate | Status |
|---|---|
| Leakage checks | PASS (future-row perturbation max diff 0.0 on the production builder; audit pipeline causality PASS) |
| Data quality | PASS (DQ flags investigated as dataset artifact) |
| Validation performance | PASS (val AUC ≈ 0.9959) |
| Untouched test | PASS **for the audit 25-feature model** — the deployed 15-feature artifact's own untouched-test eval was NOT rerun → its status is **UNVERIFIED** |
| Robustness | PASS (stress 24/24, calibration, segment audits) |
| Production parity | **PARTIAL → FAIL** (`city_fraud_rate` cache reset in `train_altman_fullscale.py`; audit-vs-deployed model mismatch — see `PS14_SEGMENT_AND_PARITY_REPORT.md`) |
| Security | PASS (13/13) |
| Human approval | **UNVERIFIED** (no approval record exists) |
| **Overall** | **approved: False** |

**Reason recorded:** *The deployed lean artifact predates the forensic revalidation. The audit validated an inline 25-feature XGB, NOT this artifact. Promotion to "fully validated" status requires running the untouched-test protocol against THIS artifact.*

This is the correct governance outcome: the model was **not** promoted merely because validation AUC is high — the untouched-test evidence gap and the parity failure block it.

### Accident-prevention behavior (verified live)

| Condition | Detection | Behavior |
|---|---|---|
| Wrong/missing artifact | FileNotFoundError | Safe failure |
| Corrupted artifact (truncated pickle) | EOFError | Safe failure |
| Hash mismatch / tampered file | `ModelIntegrityError` (governance) | Safe failure — model not served |
| Feature-schema mismatch | `ModelIntegrityError` (governance) | Safe failure |

---

## Check 18 — Deployment, Rollback & Disaster Recovery

### Failure simulations — 7/7 failed safely

Run against the engine with staged production artifacts:

| Scenario | Detected | Safe | Mechanism |
|---|---|---|---|
| Candidate corrupt load | YES | YES | EOFError → not served |
| Missing artifact | YES | YES | FileNotFoundError → not served |
| Invalid artifact | YES | YES | IndexError → not served |
| Wrong feature schema | YES | YES | `ModelIntegrityError` (governance) |
| Hash mismatch (tamper) | YES | YES | `ModelIntegrityError` (governance) |
| NaN propagation | no | YES | engine output check — scores stay finite |
| Inf propagation | no | YES | output check — scores stay finite |

NaN/Inf are *not "detected"* as events because they are neutralized by XGB's built-in handling before they can propagate; the output check confirms no invalid scores are emitted. Safe by construction, not by explicit guard.

### Rollback test — REAL artifact swap, PASS

- **V1 (known good):** `altman_lean_15feat_20260830_200346`, prediction fingerprint `240e272e…` (200 rows).
- **V2 candidate:** `altman_clean_20260830_123703` (32-feature schema) — **blocked at load** by a scaler schema mismatch (15 vs 32 features). The candidate was never allowed to serve, which is itself the safe failure under test.
- **Rollback:** artifacts swapped back to V1 → predictions **identical to the baseline fingerprint (max diff = 0.0)**.

The rollback path was exercised with real file swaps, not documented-only.

### Performance (measured in-process; no capacity requirement exists, so none invented)

| Metric | Value |
|---|---|
| Single-row latency | p50 1.30 ms, p95 1.63 ms, p99 2.65 ms, mean 1.40 ms |
| Batch 1000 | mean 17.05 ms → **58,650 rows/s** |
| Batch 5000 | mean 64.7 ms → **77,283 rows/s** |
| Failure rate | 0% across all trials |

> **UNVERIFIED:** wall-clock serving latency under HTTP, concurrent-load throughput, and memory/CPU footprint under real traffic. Capacity requirements for a live investigation queue were never specified.

### Deployment gate

```
Security             = PASS
Data validation      = PASS
Feature parity       = PARTIAL -> FAIL   (city_fraud_rate cache bug; audit model != deployed model)
Model integrity      = PASS
Performance          = PASS (measured in-process)
Health checks        = PASS
Rollback test        = PASS

PRODUCTION DEPLOYMENT = BLOCKED
```

**Reason:** Rollback + integrity verified, but feature-parity FAIL blocks deployment of the unvalidated candidate. The current V1 remains operational and safe.

---

## Verdict

| Area | Result |
|---|---|
| Model versioning & artifact integrity | **PASS** — hashed records + load-time verification + safe failure on mismatch |
| Promotion governance | **PASS (correctly conservative)** — `approved: False`; no promotion without untouched-test evidence on the deployed artifact |
| Failure handling | **PASS** — 7/7 scenarios fail safely and visibly |
| Rollback | **PASS** — real artifact swap, bit-identical V1 predictions |
| Deployment gate | **BLOCKED** — as it must be until the check-16 parity failures are fixed |

**Bottom line:** the governance and rollback machinery works and is honest — it refuses to bless the deployed artifact until the parity issues from check 16 are resolved and the untouched-test protocol is run against that artifact. The blocking items are the same two fixes already documented: the one-line `city_state` placement fix in `train_altman_fullscale.py`, and either deploying the audited model or re-running validation against the deployed 15-feature artifact.
