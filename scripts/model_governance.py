#!/usr/bin/env python3
"""
PS-14 CHECK #17: MODEL VERSIONING & GOVERNANCE

1. Generate a permanent, immutable ModelRecord for the deployed production
   artifact (models/production): dataset hash, artifact SHA-256s, feature
   schema, algorithm/hyperparameters, thresholds, metrics.
2. Verify the on-disk artifacts match the record.
3. Evaluate the promotion gate.
4. Emit a governance report.
"""
import json, os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:5.0f}s] {msg}", flush=True)

from src.risk_engine.model_governance import (
    ModelRecord, make_record, verify, verify_artifacts, sha256_file, promote_gate, ModelIntegrityError,
)

P = ROOT / "models" / "production"
RECORDS = ROOT / "models" / "model_records"
FEATS = ROOT / "data" / "_sorted_cache" / "features_all.parquet"

manifest = json.loads((P / "manifest.json").read_text(encoding="utf-8"))
feature_list = json.loads((P / "feature_list.json").read_text(encoding="utf-8"))
model_id = manifest["model_version"]

# Hyperparameters from the actual fitted models
try:
    import joblib
    xgb = joblib.load(P / "xgb_production.joblib")
    lgb = joblib.load(P / "lgb_production.joblib")
    cb = joblib.load(P / "cb_production.joblib") if (P / "cb_production.joblib").exists() else None
    hp = {
        "xgb": {"n_estimators": int(getattr(xgb, "n_estimators", 0)),
                "max_depth": int(getattr(xgb, "max_depth", 0)),
                "learning_rate": float(getattr(xgb, "learning_rate", 0))},
        "lgb": {"n_estimators": int(getattr(lgb, "n_estimators", 0)),
                "num_leaves": int(getattr(getattr(lgb, "booster_", None), "num_trees", 0) or
                                  getattr(lgb, "n_estimators", 0))},
    }
    if cb is not None:
        hp["cb"] = {"iterations": int(getattr(cb, "tree_count_", 0)),
                    "depth": int(getattr(cb, "depth", 0))}
except Exception as e:
    hp = {"note": f"could not introspect: {e}"}

log(f"Hashing training dataset {FEATS.name} ({FEATS.stat().st_size/1e9:.1f} GB)...")
ds_hash = sha256_file(FEATS)

log("Building model record...")
record = make_record(
    model_id=model_id,
    artifact_dir=P,
    training_dataset_path=str(FEATS),
    dataset_sha256=ds_hash,
    dataset_rows=24_386_900,
    code_version="models/production + src/risk_engine/altman_ensemble.py",
    feature_schema_version=manifest.get("feature_version", "altman_lean_v1"),
    features=feature_list,
    algorithm=manifest.get("model_type", "xgb_lgb_cb_ensemble"),
    hyperparameters=hp,
    training_seed=42,
    selected_threshold=manifest.get("thresholds", {}).get("1pct_fpr") if isinstance(manifest.get("thresholds"), dict) else None,
    performance_metrics={
        "temporal_test_auc": manifest.get("temporal_test_auc"),
        "temporal_test_r1": manifest.get("temporal_test_r1"),
        "cv_auc_mean": manifest.get("cv_auc_mean"),
    },
    calibration_version="calibrator.joblib (models/artifacts)",
)
# Promotion gate: document what this model passed. Gaps are recorded as
# UNVERIFIED by promote_gate(); this record documents the CURRENT state.
record.promotion = {
    "gates": {
        "leakage_checks": "PASS (future-row perturbation max diff 0.0 on production builder; "
                          "audit pipeline causality PASS)",
        "data_quality": "PASS (data-quality monitor HEALTHY/WARNING; DQ CRITICAL on timestamps "
                        "investigated as dataset artifact)",
        "validation_performance": "PASS (val AUC ~0.9959)",
        "untouched_test": "PASS for audit model (25-feat); DEPLOYED 15-feat artifact's own "
                          "untouched-test eval NOT rerun -> UNVERIFIED below",
        "robustness": "PASS (stress matrix 24/24, calibration/segment audits)",
        "production_parity": "PARTIAL -> FAIL documented: city_fraud_rate cache reset + "
                             "audit-vs-deployed model mismatch (see production_parity.json)",
        "security": "PASS (security audit 13/13)",
        "approval": "UNVERIFIED (no human approval record)",
    },
    "reason": "Deployed lean artifact predates the forensic revalidation. Audit validated an "
              "inline 25-feature XGB, NOT this artifact. Promotion to 'fully validated' status "
              "requires running the untouched-test protocol against THIS artifact.",
}
rec_path = record.save(RECORDS)
log(f"Record saved: {rec_path}")

log("Verifying on-disk artifacts against record...")
checks = verify_artifacts(record, P)
fail = [c for c in checks if not c["pass"]]
for c in checks:
    log(f"  {c['name']}: {'PASS' if c['pass'] else 'FAIL'} ({c['detail']})")
log(f"Artifact verification: {'PASS' if not fail else f'{len(fail)} FAILURES'}")

log("Evaluating promotion gate...")
gate = promote_gate(record)
for g in gate["gate"]:
    log(f"  [{g['status']}] {g['gate']}")
log(f"Promotion approved: {gate['approved']}")

report = {
    "record_path": str(rec_path),
    "model_id": model_id,
    "artifact_verification": {"pass": not fail, "n_checks": len(checks), "failures": fail},
    "promotion_gate": gate,
    "record": json.loads(rec_path.read_text(encoding="utf-8")),
}
(ROOT / "reports" / "model_governance.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
log(f"Report saved: reports/model_governance.json ({time.time()-T0:.0f}s)")
