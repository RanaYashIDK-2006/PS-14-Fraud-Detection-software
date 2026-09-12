#!/usr/bin/env python3
"""Structural feature leakage tests.

Verifies that the feature registry prevents label-derived, future-data,
and post-event-outcome features from entering online scoring.  Also tests
that the ML feature list matches the registry and that training would
reject illegal features.
"""

import sys
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


def run_tests():
    global passed, failed, errors
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("FEATURE LEAKAGE STRUCTURAL TESTS")
    print("=" * 70)

    try:
        from src.privacy_layer.feature_registry import (
            FEATURE_REGISTRY, ILLEGAL_FEATURES,
            validate_feature_list, get_scoring_features,
            FeatureMeta, PrivacyClass,
        )
        from src.privacy_layer.features import ML_FEATURES
    except ImportError as e:
        print(f"\n  FATAL: Cannot import modules: {e}")
        return 1

    # ---- Registry integrity ----
    print("\n[1] Registry integrity")
    registry_names = {f.name for f in FEATURE_REGISTRY}
    test("Registry is non-empty",
         len(FEATURE_REGISTRY) > 0,
         f"got {len(FEATURE_REGISTRY)} features")
    test("All registry features have unique names",
         len(registry_names) == len(FEATURE_REGISTRY),
         f"duplicates found")

    # ---- ML_FEATURES must match registry ----
    print("\n[2] ML_FEATURES ↔ Registry alignment")
    ml_set = set(ML_FEATURES)
    missing_from_registry = ml_set - registry_names
    test("All ML_FEATURES are in the registry",
         len(missing_from_registry) == 0,
         f"Unregistered: {missing_from_registry}" if missing_from_registry else "")

    extra_in_registry = registry_names - ml_set
    test("All registry features are in ML_FEATURES",
         len(extra_in_registry) == 0,
         f"Extra in registry: {extra_in_registry}" if extra_in_registry else "")

    # ---- No feature uses labels ----
    print("\n[3] No feature uses labels")
    for feat in FEATURE_REGISTRY:
        test(f"'{feat.name}' does not use labels",
             not feat.uses_label,
             "LEAKAGE: uses_label=True" if feat.uses_label else "")

    # ---- No feature uses future data ----
    print("\n[4] No feature uses future data")
    for feat in FEATURE_REGISTRY:
        test(f"'{feat.name}' does not use future data",
             not feat.uses_future_data,
             "LEAKAGE: uses_future_data=True" if feat.uses_future_data else "")

    # ---- No feature uses post-event outcomes ----
    print("\n[5] No feature uses post-event outcomes")
    for feat in FEATURE_REGISTRY:
        test(f"'{feat.name}' does not use post-event outcomes",
             not feat.uses_post_event_outcome,
             "LEAKAGE: uses_post_event_outcome=True" if feat.uses_post_event_outcome else "")

    # ---- All features are allowed at scoring time ----
    print("\n[6] All features allowed at scoring time")
    for feat in FEATURE_REGISTRY:
        test(f"'{feat.name}' is allowed at scoring time",
             feat.allowed_at_scoring_time,
             "Not allowed for online scoring" if not feat.allowed_at_scoring_time else "")

    # ---- Validation catches illegal features ----
    print("\n[7] Validation catches illegal features")
    illegal_tests = [
        ["amount_ratio", "label"],
        ["amount_ratio", "is_fraud"],
        ["amount_ratio", "chargeback"],
        ["amount_ratio", "email"],
        ["amount_ratio", "raw_amount"],
        ["amount_ratio", "transaction_amount"],
    ]
    for feature_list in illegal_tests:
        errs = validate_feature_list(feature_list)
        test(f"Validation catches illegal features: {feature_list[1]}",
             len(errs) > 0,
             f"No error raised for {feature_list[1]}" if not errs else "")

    # ---- Validation accepts clean feature list ----
    print("\n[8] Validation accepts clean feature list")
    errs = validate_feature_list(ML_FEATURES)
    test("Clean ML_FEATURES passes validation",
         len(errs) == 0,
         f"Errors: {errs}" if errs else "")

    # ---- Scoring features are a safe subset ----
    print("\n[9] Scoring features are safe")
    scoring = get_scoring_features()
    scoring_set = set(scoring)
    test("Scoring features is non-empty",
         len(scoring) > 0)
    test("Scoring features ⊆ ML_FEATURES",
         scoring_set.issubset(ml_set))
    test("All scoring features have no labels/future/outcome",
         all(
             not f.uses_label and not f.uses_future_data and not f.uses_post_event_outcome
             for f in FEATURE_REGISTRY if f.name in scoring_set
         ))

    # ---- No amount features are raw ----
    print("\n[10] No amount features are raw")
    amount_features = [f for f in FEATURE_REGISTRY if "amount" in f.name.lower()]
    for feat in amount_features:
        test(f"'{feat.name}' has safe source (not raw amount)",
             "raw" not in feat.source.lower() and "actual" not in feat.source.lower(),
             f"Source: {feat.source}")

    # ---- Privacy class assignment ----
    print("\n[11] Privacy class assignments")
    for feat in FEATURE_REGISTRY:
        test(f"'{feat.name}' has a privacy class",
             feat.privacy_class in (PrivacyClass.SAFE, PrivacyClass.DERIVED, PrivacyClass.AGGREGATE))

    # ---- Simulate training rejection ----
    print("\n[12] Training would reject label-derived features")
    # Simulate: what if someone tried to add a label feature?
    fake_features = ML_FEATURES + ["was_fraud"]
    errs = validate_feature_list(fake_features)
    test("Training rejects feature list containing 'was_fraud'",
         any("was_fraud" in e for e in errs))

    fake_features2 = ML_FEATURES + ["chargeback_result"]
    errs2 = validate_feature_list(fake_features2)
    test("Training rejects feature list containing 'chargeback_result'",
         any("chargeback_result" in e for e in errs2))

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
