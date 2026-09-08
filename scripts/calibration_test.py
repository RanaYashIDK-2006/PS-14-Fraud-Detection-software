#!/usr/bin/env python3
"""Calibration validation tests.

REQUIRES real artifacts:
  - models/artifacts/calibrator.joblib
  - models/artifacts/validation_labels.joblib
  - models/artifacts/fused_val_scores.joblib

If any required artifact is missing, the test FAILS with clear message.
No synthetic/fake calibration data is generated.

Usage:
    python scripts/calibration_test.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        errors.append(name)


def compute_brier_score(y_true, y_prob):
    """Brier score: mean((y_prob - y_true)^2). Lower is better. Perfect = 0."""
    return float(np.mean((np.array(y_prob) - np.array(y_true)) ** 2))


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(y_true, y_prob, n_bins=10):
    """Compute reliability diagram data."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bins.append({
            "bin_lower": float(bin_edges[i]),
            "bin_upper": float(bin_edges[i + 1]),
            "mean_predicted": float(y_prob[mask].mean()),
            "fraction_positive": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return bins


def run_tests():
    global passed, failed, errors
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("CALIBRATION VALIDATION TESTS")
    print("=" * 70)

    artifacts_dir = Path(__file__).resolve().parent.parent / "models" / "artifacts"

    # ---- Step 1: Require calibrator artifact ----
    print("\n[1] Required calibration artifacts")
    calibrator_path = artifacts_dir / "calibrator.joblib"
    test("Calibrator artifact exists",
         calibrator_path.exists(),
         f"MISSING: {calibrator_path}")

    labels_path = artifacts_dir / "validation_labels.joblib"
    scores_path = artifacts_dir / "fused_val_scores.joblib"
    test("Validation labels artifact exists",
         labels_path.exists(),
         f"MISSING: {labels_path}")
    test("Validation scores artifact exists",
         scores_path.exists(),
         f"MISSING: {scores_path}")

    # If required artifacts missing, fail immediately — no fallback
    if not calibrator_path.exists() or not labels_path.exists() or not scores_path.exists():
        print(f"\n  REQUIRED ARTIFACTS MISSING — cannot validate calibration")
        print(f"  Run: python scripts/build_model_artifacts.py")
        print(f"\n{'=' * 70}")
        total = passed + failed
        print(f"RESULTS: {passed}/{total} passed, {failed} failed")
        print("=" * 70)
        return 1

    # ---- Step 2: Load and validate calibrator ----
    print("\n[2] Load calibrator")
    import joblib
    calibrator = joblib.load(calibrator_path)
    test("Calibrator is PlattCalibration",
         hasattr(calibrator, 'lr') and hasattr(calibrator, 'predict'),
         f"Type: {type(calibrator)}")
    test("Calibrator has been fitted",
         hasattr(calibrator.lr, 'coef_') and calibrator.lr.coef_ is not None)

    # ---- Step 3: Load real validation data ----
    print("\n[3] Load real validation data")
    y_true = joblib.load(labels_path)
    y_prob_raw = joblib.load(scores_path)  # raw uncalibrated scores
    test(f"Real validation data loaded ({len(y_true)} samples)",
         len(y_true) > 100 and len(y_true) == len(y_prob_raw),
         f"Labels: {len(y_true)}, Scores: {len(y_prob_raw)}")

    # Apply calibrator to raw scores to get calibrated probabilities
    y_prob = np.clip(calibrator.predict(y_prob_raw), 0.0, 1.0)
    test("Calibrator applied to raw scores",
         len(y_prob) == len(y_true),
         f"Calibrated: {len(y_prob)}")

    # ---- Step 4: Calibration metrics ----
    print("\n[4] Calibration metrics")

    brier = compute_brier_score(y_true, y_prob)
    test(f"Brier score ({brier:.4f})",
         0 <= brier <= 1,
         f"Value: {brier}")

    ece = compute_ece(y_true, y_prob, n_bins=10)
    test(f"ECE ({ece:.4f})",
         0 <= ece <= 1,
         f"Value: {ece}")

    curve = reliability_curve(y_true, y_prob, n_bins=10)
    test("Reliability curve has bins",
         len(curve) > 0,
         f"Bins: {len(curve)}")

    # Check calibration quality: bins should show increasing fraud rate
    if len(curve) >= 2:
        fraud_rates = [b["fraction_positive"] for b in curve]
        # Fraud rate should generally increase with predicted probability
        monotonic_pairs = sum(1 for i in range(len(fraud_rates)-1) 
                             if fraud_rates[i] <= fraud_rates[i+1] + 0.05)
        test("Calibration shows increasing fraud rate with probability",
             monotonic_pairs >= len(fraud_rates) - 2,
             f"Pairs: {monotonic_pairs}/{len(fraud_rates)-1}")

    # ---- Step 5: Platt calibration properties ----
    print("\n[5] Platt calibration properties")

    # Monotonicity
    test_points = np.linspace(0.01, 0.99, 20)
    calibrated = calibrator.predict(test_points)
    diffs = np.diff(calibrated)
    test("Calibration is monotonically increasing",
         np.all(diffs >= -0.001),
         f"Min diff: {diffs.min():.6f}")

    # Range
    test("Calibration output in [0, 1]",
         all(0 <= c <= 1 for c in calibrated),
         f"Range: [{calibrated.min():.4f}, {calibrated.max():.4f}]")

    # Low raw → low calibrated
    low_raw = calibrator.predict(np.array([0.05]))[0]
    test("Low raw (0.05) → low calibrated",
         low_raw < 0.3,
         f"Raw 0.05 → {low_raw:.4f}")

    # High raw → reasonable calibrated (not compressed to near-zero)
    high_raw = calibrator.predict(np.array([0.95]))[0]
    test("High raw (0.95) → non-trivial calibrated",
         high_raw > 0.05,
         f"Raw 0.95 → {high_raw:.4f}")

    # ---- Step 6: Metadata check ----
    print("\n[6] Calibration metadata")
    import json
    metadata_path = artifacts_dir / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        test("metadata.json has calibration info",
             "calibrator" in str(meta).lower() or "calibration" in str(meta).lower())
    else:
        test("metadata.json exists", False)

    # ---- Summary ----
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e}")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
