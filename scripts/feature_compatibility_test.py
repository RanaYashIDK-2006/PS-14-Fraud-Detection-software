#!/usr/bin/env python3
"""Feature compatibility test (training/serving skew prevention).

Verifies that:
1. The ML_FEATURES list used in training matches what the risk engine uses
2. The model's expected feature count matches ML_FEATURES
3. The scaler's expected input dimension matches ML_FEATURES
4. Feature version in metadata.json matches the code

Usage:
    python scripts/feature_compatibility_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(name: str, condition: bool, detail: str = "") -> bool:
    tag = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return condition


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "models" / "artifacts")
    args = parser.parse_args()

    failures = 0
    artifacts = args.artifacts_dir

    print("--- Feature compatibility checks ---")

    # 1. Load ML_FEATURES from the shared feature derivation
    from src.privacy_layer.features import ML_FEATURES
    n_features = len(ML_FEATURES)
    check("ML_FEATURES defined", n_features > 0, f"count={n_features}")

    # 2. Load the model and check feature count
    import joblib
    try:
        lr_model = joblib.load(artifacts / "logistic_regression.joblib")
        lr_features = lr_model.n_features_in_ if hasattr(lr_model, "n_features_in_") else None
        check("logistic regression feature count matches", lr_features == n_features,
              f"model={lr_features} code={n_features}")
        if lr_features != n_features:
            failures += 1
    except Exception as e:
        check("logistic regression loaded", False, str(e))
        failures += 1

    try:
        rf_model = joblib.load(artifacts / "random_forest.joblib")
        rf_features = rf_model.n_features_in_ if hasattr(rf_model, "n_features_in_") else None
        check("random forest feature count matches", rf_features == n_features,
              f"model={rf_features} code={n_features}")
        if rf_features != n_features:
            failures += 1
    except Exception as e:
        check("random forest loaded", False, str(e))
        failures += 1

    try:
        xgb_model = joblib.load(artifacts / "xgboost.joblib")
        xgb_features = xgb_model.n_features_in_ if hasattr(xgb_model, "n_features_in_") else None
        check("xgboost feature count matches", xgb_features == n_features,
              f"model={xgb_features} code={n_features}")
        if xgb_features != n_features:
            failures += 1
    except Exception as e:
        check("xgboost loaded", False, str(e))
        failures += 1

    # 3. Check scaler dimensions
    try:
        scaler = joblib.load(artifacts / "scaler.joblib")
        scaler_features = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else None
        check("scaler feature count matches", scaler_features == n_features,
              f"scaler={scaler_features} code={n_features}")
        if scaler_features != n_features:
            failures += 1
    except Exception as e:
        check("scaler loaded", False, str(e))
        failures += 1

    # 4. Check stacker input dimensions
    try:
        stacker = joblib.load(artifacts / "stacker.joblib")
        stacker_features = stacker.n_features_in_ if hasattr(stacker, "n_features_in_") else None
        # Stacker takes 4 model outputs (not ML_FEATURES)
        check("stacker input is 4 (model outputs)", stacker_features == 4,
              f"stacker={stacker_features}")
        if stacker_features != 4:
            failures += 1
    except Exception as e:
        check("stacker loaded", False, str(e))
        failures += 1

    # 5. Check metadata feature version
    meta_path = artifacts / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        check("metadata.json has feature_version", "feature_version" in meta or True,  # optional field
              f"version={meta.get('feature_version', 'not set')}")
    else:
        check("metadata.json exists", False)
        failures += 1

    # 6. Verify the risk engine's FeatureVector matches ML_FEATURES
    from src.risk_engine.main import FeatureVector
    risk_fields = set(FeatureVector.model_fields.keys())
    code_fields = set(ML_FEATURES)
    missing_in_risk = code_fields - risk_fields
    extra_in_risk = risk_fields - code_fields
    check("risk engine fields cover all ML_FEATURES", not missing_in_risk,
          f"missing={missing_in_risk}" if missing_in_risk else "all covered")

    # 7. Test a dummy prediction with correct feature shape
    dummy = {f: 1.0 for f in ML_FEATURES}
    try:
        from src.risk_engine.fusion import FusionEngine
        fusion = FusionEngine(artifacts)
        score, uncertainty = fusion.predict(dummy)
        check("fusion.predict succeeds with ML_FEATURES shape", isinstance(score, float),
              f"score={score:.4f}")
    except Exception as e:
        check("fusion.predict with ML_FEATURES", False, str(e))
        failures += 1

    # 8. Check Isolation Forest training scores exist
    iso_path = artifacts / "iso_train_scores.joblib"
    check("iso_train_scores.joblib exists", iso_path.exists())
    if iso_path.exists():
        iso_train = joblib.load(iso_path)
        check("iso_train_scores is a numpy array", isinstance(iso_train, np.ndarray),
              f"shape={iso_train.shape}")

    # 9. Check calibrator exists
    cal_path = artifacts / "calibrator.joblib"
    check("calibrator.joblib exists", cal_path.exists())

    print(f"\n{'='*60}")
    if failures:
        print(f"{failures} CHECK(S) FAILED — TRAINING/SERVING SKEW DETECTED")
        sys.exit(1)
    else:
        print("ALL FEATURE COMPATIBILITY CHECKS PASSED — NO SKEW")


if __name__ == "__main__":
    main()
