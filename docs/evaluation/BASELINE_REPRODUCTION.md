# Baseline Reproduction Manual (Phase 105)

Everything needed for another engineer to reproduce the verified baseline from a fresh
checkout. Commands are run from the **repository root** unless noted.

---

## 1. Environment

| Item | Value |
|---|---|
| OS | Windows 11 (records: `Windows-11-10.0.26200-SP0`); commands via Git Bash |
| Python | 3.12.10 — `./.venv/Scripts/python.exe` (create with `uv pip install -r backend/requirements.txt` into `.venv/`) |
| Packages | numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, xgboost 3.4.1, scipy 1.18.0 (recorded by every ledger record) |
| Encoding | **always** `PYTHONIOENCODING=utf-8` (suites print `✓`; cp1252 crashes otherwise) |
| Git SHA of this baseline | `a0b600ba2edd917498d3a9551c19e10225930e4b` (`a0b600b`) |
| Services | not required for the evaluations below; **required** for `security_test`, `sql_injection_test`, `security_ci_gate` — start with `powershell -NoProfile -ExecutionPolicy Bypass -File .freebuff/start_stack.ps1` (ports 8000–8005) |

## 2. Datasets (paths + hashes — verify before evaluating)

| Dataset | Path | sha256 |
|---|---|---|
| ULB credit card | `data/creditcard.csv` | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| Synthetic PS-14 | `data/transactions.csv` | `9f0f56bf0549fccf0c6335ece1514d1ae515a57f809a72658ce5fcaf48525d63` |
| IBM v2 | `data/credit_card_transactions-ibm_v2.csv` | `b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15` |
| Kaggle fraudTest | `data/kaggle_fraud/fraudTest.csv` | `12d553ab19440c75…` (truncated in `misc/benchmarks/datasets.json`) |
| Calibration eval set | `models/artifacts/validation_labels.joblib` | `d1d44f03455cb0f2…` (in ledger record) |

`data/` and `models/` are gitignored: a fresh clone must obtain them from the project's
data distribution (never regenerate hashes silently — a hash mismatch means a different
dataset and a different experiment).

## 3. Reproduce the verified baseline (in order)

### 3.1 Calibration (README "Brier 0.0009, ECE 0.0013")

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe backend/scripts/calibration_test.py
```
- Requires: `models/artifacts/{calibrator,validation_labels,fused_val_scores}.*.joblib`
- Expected: `RESULTS: 16/16 passed`; console `Brier score (0.0009)`, `ECE (0.0013)`
- Produces: `reports/calibration_test/calibration_metrics.json`
  (exact: brier `0.0009280049005246071`, ece `0.001279013244298246`, n=1470)
  + a new append-only ledger record (`command: python backend/scripts/calibration_test.py`)
- Seed: none (deterministic over fixed artifacts)

### 3.2 ULB research benchmark (README ULB table: 0.976 / 91.8% / 0.883)

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe backend/scripts/eval_ulb.py
```
- Requires: `data/creditcard.csv`
- Seed 42; split random stratified 80/20 (`random_state=42`); trains 7 models + 5-fold CV
  (~3–5 min)
- Expected `reports/ulb_results.json` → `results.pattern_xgb`:
  `roc_auc 0.975807`, `pr_auc 0.883465`, `r1 0.918367` (bit-identical on this machine,
  2026-10-03); `cv_5fold_xgb.roc_auc_mean 0.984873 ± 0.007484`
- Produces: ledger record `eval-20261003T084934+0000-e58d68c2bf88` (a re-run appends a
  new id with the same metrics) + `record_<id>.json`

### 3.3 Cross-dataset comparison (README external table: 0.873)

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe backend/scripts/cross_dataset_eval.py --ibm-rows 200000
```
- Requires: `models/artifacts/{logistic_regression,random_forest,xgboost,lr_model,rf_model,xgb_model}.joblib`,
  `data/transactions.csv`, IBM v2, `data/creditcard.csv`
- **The flag matters:** default `300000` yields a different window (IBM model 0.9192);
  `200000` reproduces the documented `ibm_v2_holdout` result exactly
  (`synthetic_model 0.8512`, `ibm_v2_model 0.8726`)
- Expected: `reports/cross_dataset/cross_dataset_report.json` differs from the tracked
  version only in `generated` timestamp and `command`
- Produces: ledger record `eval-20261003T085632+0000-e2a14ae94bef`

### 3.4 Baseline model train+eval (synthetic fused: ROC 0.9838 / PR 0.9508 / R@1% 0.9556)

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe backend/src/train_compare.py
```
- Requires: `data/transactions.csv`; writes `models/artifacts/*` (regenerated
  deterministically, seed 42)
- Expected: `models/artifacts/metrics_comparison.csv` row
  `fused_ensemble (stacker)` → `roc_auc 0.983797…`, `pr_auc 0.950814…`,
  `recall_at_1pct_fpr 0.955556…`; OOD gate PASS (ato/mule recall@1%FPR 1.000)
- Produces: ledger record with `command: python src/train_compare.py …` and
  `threshold_source: validation`
- Note: before Phase 105 this command **silently failed to append its record**
  (`UnboundLocalError: gate_rows`); fixed — if you see
  `(evaluation record skipped: …)` the record was NOT written and the run is incomplete
  evidence.

### 3.5 Leakage structural suite (README "123/123")

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe backend/scripts/leakage_structural_test.py
```
- Expected: `RESULTS: 123/123 passed, 0 failed`, rc=0
- Phase 105 capture: `reports/phase105/leakage_structural_test.out`

### 3.6 Evidence enforcement (the CI gate itself)

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe backend/scripts/claim_evidence_check.py   # expect PASS
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe backend/scripts/claim_evidence_check.py --self-test  # 7/7
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe backend/scripts/eval_record_test.py       # 22/22
```

### 3.7 Full battery

```bash
bash .freebuff/p114_battery.sh     # 79 checks; start the stack first (§1)
```
- Phase 105 result: 76/79 as-run with services down; 3/3 re-run PASS with stack up →
  effective **79/79**; tally `reports/phase105/battery_results.tsv`, re-runs
  `reports/phase105/battery_reruns.txt`
- Phase 104's prior tally is preserved at `.freebuff/p104_phase_battery_results.tsv`

## 4. Expected evidence after reproduction

| Command | Artifact | Ledger record (this phase's run) |
|---|---|---|
| calibration_test.py | `reports/calibration_test/calibration_metrics.json` | `eval-20261003T084644+0000-72d714bd81b3` |
| eval_ulb.py | `reports/ulb_results.json` | `eval-20261003T084934+0000-e58d68c2bf88` |
| cross_dataset_eval.py --ibm-rows 200000 | `reports/cross_dataset/cross_dataset_report.json` | `eval-20261003T085632+0000-e2a14ae94bef` |
| train_compare.py | `models/artifacts/metrics_comparison.csv` + artifacts | `eval-20261003T090047+0000-359039aac6cf` |
| leakage_structural_test.py | `reports/phase105/leakage_structural_test.out` | n/a (suite evidence, registry C-018) |

Re-runs append **new** records with identical metrics — the ledger never overwrites.
Compare scientifically meaningful fields (metrics, dataset hash, split, seed), never
timestamps or generated ids.

## 5. Known non-reproducible historical results (do not chase)

| Historical result | Why it cannot be reproduced | Status |
|---|---|---|
| ULB tuple 0.966 / 0.877 / 97.8%@0.43 / 1.996% | no single artifact; no producing command; component values come from different experiments (§6 of the Phase 105 report) | NOT ESTABLISHED (registry C-101…C-104) |
| Phase 24 external 0.435 / FPR 44.9% | prose-only; the phase24 metrics artifact was regenerated to the Phase 25 result (0.594662) | NOT ESTABLISHED (C-011/C-012) |
| fraud_report.py performance header | `xgb_kaggle.joblib` / `scaler_kaggle.joblib` absent and **no producer script exists** | BLOCKED (script now refuses to print unmeasured numbers) |
| Penetration "42 scenarios" | executed count is pinned by no artifact (42 vs 34 vs 66+43 call sites) | NOT ESTABLISHED (wording corrected) |

## 6. Trace-only results (artifacts exist; pipelines NOT re-executed in Phase 105)

| Result | Artifact | Re-execution cost / blocker |
|---|---|---|
| IBM v2 XGB 0.9819 / fused 0.9779 | `reports/ibm_train/training_report.json` | `ibm_train.py` retrains on 1.2M rows (~tens of minutes); artifact lacks seed/git |
| Phase 23B 0.4658 | `misc/reports/phase23b/kaggle/13_p20_metrics.json` | phase-era pipeline; no ledger mechanism at the time |
| Phase 25 0.594662 / FPR 0.099313 | `misc/reports/phase24/kaggle/05_e_hardneg_metrics.json` | phase-era pipeline; provenance captured in registry `provenance` block |
| Phase 108 production-on-ULB (rules-only fail-close, ROC 0.5) | `reports/external_benchmark/phase108/ulb/evaluation_results.json` (+ manifest, integrity hashes) | heavy protocol execution; artifact hash-verified |

## 7. Blockers

1. **Data acquisition:** `data/*` and `models/*` are gitignored — a fresh clone cannot
   reproduce anything without the dataset distribution (hashes above are the contract).
2. **Services:** three battery suites fail without the six-service stack (§1).
3. **Windows encoding:** without `PYTHONIOENCODING=utf-8`, several suites crash on cp1252.
4. **Commit coupling:** CI's evidence steps require the Phase 105 evidence files
   (`reports/ulb_results.json`, `reports/phase105/*`, `reports/calibration_test/*`,
   updated ledger, registry, this file) to be committed **together with** the README —
   a partial commit will (correctly) fail CI on missing evidence.
5. **BAF / IEEE-CIS** remain not-acquired / auth-blocked (Phase 104 Parts D/M) — no
   second-dataset replication is possible until that changes.
