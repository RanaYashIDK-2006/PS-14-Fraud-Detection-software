#!/usr/bin/env python3
"""PS-14 Label Leakage Regression Test.

Checks that no synthetic dataset generator produces features that are
trivially separable by label BEFORE any model is fit. This catches the
class of bug where labels are generated first, then features are modified
based on the label (data leakage).

Usage:
    python scripts/leakage_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    icon = "✓" if passed else "✗"
    print(f"  [{icon}] {name}")
    if detail:
        print(f"         → {detail}")


def detect_leakage(features: np.ndarray, labels: np.ndarray,
                   feature_names: list[str]) -> list[str]:
    """Check each feature for trivial separability by label.

    Returns list of feature names that show leakage.
    """
    leaked = []
    fraud_mask = labels == 1
    legit_mask = labels == 0

    if fraud_mask.sum() < 10 or legit_mask.sum() < 10:
        return []  # not enough data to test

    for i, name in enumerate(feature_names):
        fraud_vals = features[fraud_mask, i]
        legit_vals = features[legit_mask, i]

        fraud_mean = fraud_vals.mean()
        legit_mean = legit_vals.mean()
        fraud_std = fraud_vals.std()
        legit_std = legit_vals.std()

        # Check 1: Deterministic separation
        # If all fraud values are identical and all legit values are different
        if fraud_std < 1e-10 and abs(fraud_mean - legit_mean) > 0.01:
            leaked.append(f"{name}: deterministic (fraud={fraud_mean:.4f}, legit={legit_mean:.4f})")
            continue

        # Check 2: Near-deterministic separation
        # If fraud values have very low variance AND are far from legit
        if fraud_std < 0.01 and abs(fraud_mean - legit_mean) > 2 * max(fraud_std, legit_std, 0.01):
            leaked.append(f"{name}: near-deterministic (fraud={fraud_mean:.4f}±{fraud_std:.4f}, legit={legit_mean:.4f}±{legit_std:.4f})")
            continue

        # Check 3: Trivial overlap
        # If fraud values are all in a range that legit values never reach
        fraud_min, fraud_max = fraud_vals.min(), fraud_vals.max()
        legit_min, legit_max = legit_vals.min(), legit_vals.max()

        if fraud_min > legit_max + 0.01 or fraud_max < legit_min - 0.01:
            leaked.append(f"{name}: no overlap (fraud=[{fraud_min:.4f},{fraud_max:.4f}], legit=[{legit_min:.4f},{legit_max:.4f}])")
            continue

        # Check 4: Extreme mean ratio (>10x) with low variance
        if abs(fraud_mean) > 1e-6 and abs(legit_mean) > 1e-6:
            ratio = fraud_mean / legit_mean
            if ratio > 10 or ratio < 0.1:
                if fraud_std / max(abs(fraud_mean), 1e-6) < 0.3:
                    leaked.append(f"{name}: extreme mean ratio ({ratio:.1f}x, CV={fraud_std/max(abs(fraud_mean),1e-6):.2f})")

    return leaked


def main():
    print("=" * 70)
    print("PS-14 LABEL LEAKAGE REGRESSION TEST")
    print("=" * 70)

    ML_FEATURES = [
        "hour_of_day", "is_weekend", "amount_ratio", "txn_amount_bucket",
        "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
        "unusual_location_flag", "unusual_recipient_flag",
        "failed_auth_count_24h", "days_since_last_similar_txn",
        "gradual_escalation_score", "known_device_count",
        "account_tenure_days", "shared_device_accounts",
        "shared_recipient_accounts",
    ]

    # Test 1: Sparkov synthetic fallback
    print("\n--- Sparkov synthetic fallback ---")
    from scripts.import_real_datasets import _generate_sparkov_synthetic
    features, labels = _generate_sparkov_synthetic()
    print(f"  Generated {len(labels)} rows, {int(labels.sum())} fraud ({labels.mean():.1%})")
    leaked = detect_leakage(features, labels, ML_FEATURES)
    check("Sparkov synthetic: no label leakage", len(leaked) == 0,
          f"Leaked features: {leaked}" if leaked else "Clean — label derived from features")

    # Test 2: Elliptic synthetic fallback
    print("\n--- Elliptic synthetic fallback ---")
    from scripts.import_real_datasets import _generate_elliptic_synthetic
    features, labels = _generate_elliptic_synthetic()
    print(f"  Generated {len(labels)} rows, {int(labels.sum())} fraud ({labels.mean():.1%})")
    leaked = detect_leakage(features, labels, ML_FEATURES)
    check("Elliptic synthetic: no label leakage", len(leaked) == 0,
          f"Leaked features: {leaked}" if leaked else "Clean — label derived from features")

    # Test 3: Sparkov real-data mapping (dist column must NOT be a fraud proxy)
    print("\n--- Sparkov real-data mapping (dist proxy test) ---")
    from scripts.import_real_datasets import map_sparkov_to_features
    import pandas as pd
    rng_test = np.random.RandomState(99)
    n_mock = 10000
    mock_dist = rng_test.lognormal(5, 2, n_mock)
    # Mark top 5% as fraud (Sparkov's construction)
    mock_labels = (mock_dist > np.percentile(mock_dist, 95)).astype(int)
    mock_df = pd.DataFrame({
        "amt": rng_test.lognormal(4, 1.5, n_mock),
        "dist": mock_dist,
        "is_fraud": mock_labels,
    })
    features_mock, labels_mock = map_sparkov_to_features(mock_df)
    print(f"  Mock: {n_mock} rows, {int(labels_mock.sum())} fraud ({labels_mock.mean():.1%})")
    leaked_mock = detect_leakage(features_mock, labels_mock, ML_FEATURES)
    check("Sparkov mapping: dist not used as fraud proxy",
          "unusual_location_flag" not in [l.split(":")[0] for l in leaked_mock],
          f"Leaked: {leaked_mock}" if leaked_mock else "Clean — dist column ignored")

    # Test 4: ULB (real data — should have some signal but not leakage)
    print("\n--- ULB Credit Card Fraud (real data) ---")
    data_path = ROOT / "data" / "creditcard.csv"
    if data_path.exists():
        import pandas as pd
        df = pd.read_csv(data_path)
        # Map to features
        n = min(30000, len(df))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), n, replace=False)
        sub = df.iloc[idx]

        features = np.zeros((n, 16))
        features[:, 0] = (sub["Time"].values % 86400) / 3600
        features[:, 1] = ((sub["Time"].values % 604800) / 86400 > 5).astype(float)
        features[:, 2] = sub["Amount"].values / max(sub["Amount"].mean(), 1)
        features[:, 3] = np.digitize(sub["Amount"].values, [0, 10, 50, 100, 500, 1000])
        features[:, 4] = rng.poisson(2, n)
        features[:, 5] = features[:, 0] / 24
        features[:, 6] = rng.binomial(1, 0.1, n)
        features[:, 7] = rng.binomial(1, 0.05, n)
        features[:, 8] = rng.binomial(1, 0.08, n)
        features[:, 9] = rng.poisson(0.1, n)
        features[:, 10] = rng.exponential(30, n)
        features[:, 11] = rng.beta(2, 5, n)
        features[:, 12] = rng.poisson(1, n)
        features[:, 13] = rng.exponential(200, n)
        features[:, 14] = rng.poisson(0.2, n)
        features[:, 15] = rng.poisson(0.3, n)

        labels = sub["Class"].values
        print(f"  Loaded {n} rows, {int(labels.sum())} fraud ({labels.mean():.1%})")
        leaked = detect_leakage(features, labels, ML_FEATURES)
        check("ULB real data: no trivial leakage", len(leaked) <= 2,
              f"Features with signal: {leaked}" if leaked else "Clean")
    else:
        # data/creditcard.csv (ULB, ~150 MB) is a local-only, gitignored
        # dataset - skip (not fail) on fresh checkouts / CI. The check
        # still runs locally whenever the file is present.
        check("ULB real data: SKIPPED (dataset not present)", True,
              f"local-only dataset absent: {data_path}")

    # Summary
    print()
    print("=" * 70)
    ok = sum(1 for _, p, _ in RESULTS if p)
    fail = sum(1 for _, p, _ in RESULTS if not p)
    print(f"Results: {ok} passed, {fail} failed out of {len(RESULTS)}")
    if fail:
        print("\nFAILED CHECKS:")
        for name, p, detail in RESULTS:
            if not p:
                print(f"  ✗ {name}")
                if detail:
                    print(f"    → {detail}")
    else:
        print("\n✅ ALL CHECKS PASSED — no label leakage detected")
    print("=" * 70)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
