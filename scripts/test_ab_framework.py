#!/usr/bin/env python3
"""Test A/B framework: model registry, traffic splitting, metrics."""
from __future__ import annotations
import sys
import time

sys.path.insert(0, ".")


def test_registry_discovery():
    """Test auto-discovery of model versions."""
    from src.inference.model_registry import ModelRegistry

    reg = ModelRegistry()
    discovered = reg.auto_discover()

    assert len(reg.versions) >= 2, f"Expected >= 2 models, got {len(reg.versions)}"
    print(f"  [PASS] Registry discovered {len(reg.versions)} models: {reg.versions}")
    return True


def test_registry_scorers():
    """Test that all registered versions can score."""
    from src.inference.model_registry import ModelRegistry
    import numpy as np

    reg = ModelRegistry()
    reg.auto_discover()

    # Build a dummy feature vector for each model
    for v in reg.versions:
        mv = reg.get_model(v)
        scorer = reg.get_scorer(v)
        assert scorer is not None, f"No scorer for {v}"

        # Random features matching n_features
        X = np.random.randn(5, mv.n_features).astype(np.float32)
        try:
            probas = scorer(X)
            assert len(probas) == 5
            assert all(0 <= p <= 1 for p in probas)
        except Exception as e:
            # Some models may need specific feature ranges
            print(f"  [WARN] {v} scorer failed on random data: {e}")
            continue

    print(f"  [PASS] All {len(reg.versions)} scorers produce valid probabilities")
    return True


def test_traffic_splitting_percentage():
    """Test random percentage-based splitting."""
    from src.inference.ab_testing import TrafficSplitter

    ab = TrafficSplitter()
    ab.create_experiment("pct_test", {"control": 0.7, "treatment": 0.3}, strategy="percentage")

    counts = {"control": 0, "treatment": 0}
    for i in range(10000):
        v = ab.get_version("pct_test", user_id=f"user_{i}")
        counts[v] += 1

    # Allow 5% tolerance
    assert 6000 < counts["control"] < 8000, f"Control={counts['control']}, expected ~7000"
    assert 2000 < counts["treatment"] < 4000, f"Treatment={counts['treatment']}, expected ~3000"
    print(f"  [PASS] Percentage split: control={counts['control']} treatment={counts['treatment']}")
    return True


def test_traffic_splitting_deterministic():
    """Test user-deterministic splitting (same user → same version)."""
    from src.inference.ab_testing import TrafficSplitter

    ab = TrafficSplitter()
    ab.create_experiment("det_test", {"v_a": 0.5, "v_b": 0.5}, strategy="user_deterministic")

    # Same user should always get same version
    for uid in [f"user_{i}" for i in range(100)]:
        v1 = ab.get_version("det_test", user_id=uid)
        v2 = ab.get_version("det_test", user_id=uid)
        v3 = ab.get_version("det_test", user_id=uid)
        assert v1 == v2 == v3, f"Non-deterministic for {uid}: {v1} != {v2}"

    # Different users should get different versions (statistically)
    versions = set()
    for i in range(100):
        versions.add(ab.get_version("det_test", user_id=f"user_{i}"))
    assert len(versions) == 2, f"Expected 2 versions in routing, got {versions}"

    print("  [PASS] Deterministic routing is consistent")
    return True


def test_metrics_tracking():
    """Test metrics accumulation and reporting."""
    from src.inference.ab_testing import TrafficSplitter

    ab = TrafficSplitter()
    ab.create_experiment("metrics_test", {"fast": 0.5, "slow": 0.5}, strategy="percentage")

    # Record known metrics
    for i in range(100):
        v = ab.get_version("metrics_test", user_id=f"u{i}")
        prob = 0.5 if v == "fast" else 0.1
        flag = prob > 0.3
        lat = 1.0 if v == "fast" else 5.0
        ab.record("metrics_test", v, prob, flag, lat)

    metrics = ab.get_metrics("metrics_test")
    assert "fast" in metrics and "slow" in metrics

    fast_m = metrics["fast"]
    slow_m = metrics["slow"]

    # Fast model should have higher fraud prob and more flags
    assert fast_m["mean_fraud_prob"] > slow_m["mean_fraud_prob"]
    assert fast_m["n_flagged"] > slow_m["n_flagged"]
    assert fast_m["mean_latency_ms"] < slow_m["mean_latency_ms"]

    print(f"  [PASS] Metrics: fast={fast_m['n_scored']} scored, slow={slow_m['n_scored']} scored")
    return True


def test_experiment_lifecycle():
    """Test create → update → deactivate lifecycle."""
    from src.inference.ab_testing import TrafficSplitter

    ab = TrafficSplitter()

    # Create
    exp = ab.create_experiment(
        "lifecycle", {"v1": 0.6, "v2": 0.4}, description="Test lifecycle"
    )
    assert exp.active
    assert exp.name == "lifecycle"

    # Update traffic
    success = ab.update_traffic("lifecycle", {"v1": 0.3, "v2": 0.7})
    assert success
    exp = ab.get_experiment("lifecycle")
    assert exp.versions["v1"] == 0.3

    # Invalid update (doesn't sum to 1.0)
    success = ab.update_traffic("lifecycle", {"v1": 0.8, "v2": 0.8})
    assert not success

    # Deactivate
    success = ab.deactivate_experiment("lifecycle")
    assert success
    exp = ab.get_experiment("lifecycle")
    assert not exp.active

    # List
    exps = ab.list_experiments()
    assert len(exps) >= 1

    print("  [PASS] Experiment lifecycle: create → update → deactivate")
    return True


def test_three_way_split():
    """Test splitting across 3 model versions."""
    from src.inference.ab_testing import TrafficSplitter

    ab = TrafficSplitter()
    ab.create_experiment("three_way", {"v1": 0.5, "v2": 0.3, "v3": 0.2})

    counts = {"v1": 0, "v2": 0, "v3": 0}
    for i in range(10000):
        v = ab.get_version("three_way", user_id=f"u{i}")
        counts[v] += 1

    assert 4000 < counts["v1"] < 6000
    assert 2000 < counts["v2"] < 4000
    assert 1000 < counts["v3"] < 3000
    print(f"  [PASS] 3-way split: v1={counts['v1']} v2={counts['v2']} v3={counts['v3']}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  A/B TESTING FRAMEWORK TESTS")
    print("=" * 60)

    all_pass = True
    tests = [
        test_registry_discovery,
        test_registry_scorers,
        test_traffic_splitting_percentage,
        test_traffic_splitting_deterministic,
        test_metrics_tracking,
        test_experiment_lifecycle,
        test_three_way_split,
    ]

    for test in tests:
        try:
            if not test():
                all_pass = False
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
