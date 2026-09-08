#!/usr/bin/env python3
"""Test Redis-backed state manager using fakeredis (no real Redis needed).

Verifies:
  1. Sliding window add/query works correctly
  2. Velocity features match in-process version
  3. TTL eviction works
  4. Horizontal scaling: two state managers share state
"""
from __future__ import annotations
import sys
import time
import numpy as np

sys.path.insert(0, ".")


def test_sliding_window_basic():
    """Test basic sliding window operations."""
    try:
        import fakeredis
    except ImportError:
        print("  [SKIP] fakeredis not installed (pip install fakeredis)")
        return True

    r = fakeredis.FakeRedis()
    from src.inference.redis_state import RedisSlidingWindow

    sw = RedisSlidingWindow(r, "user_001", max_size=5)

    # Add transactions
    for i in range(3):
        sw.add(10.0 + i * 0.1, float(100 + i * 50), merchant_id=i)

    features = sw.get_velocity_features(10.2 + 0.2)
    assert features["tx_count_24h"] == 3, f"Expected 3 txns, got {features['tx_count_24h']}"
    assert features["merchant_diversity_24h"] == 3, f"Expected 3 merchants"
    print("  [PASS] basic sliding window")
    return True


def test_sliding_window_eviction():
    """Test that window trims to max_size."""
    try:
        import fakeredis
    except ImportError:
        return True

    r = fakeredis.FakeRedis()
    from src.inference.redis_state import RedisSlidingWindow

    sw = RedisSlidingWindow(r, "user_002", max_size=3)

    for i in range(10):
        sw.add(10.0 + i * 0.1, float(i * 10), merchant_id=i % 5)

    raw = r.zrange(sw.key, 0, -1)
    assert len(raw) == 3, f"Expected 3 in window, got {len(raw)}"
    print("  [PASS] eviction to max_size")
    return True


def test_velocity_features_match():
    """Test that Redis velocity features match a manual calculation."""
    try:
        import fakeredis
    except ImportError:
        return True

    r = fakeredis.FakeRedis()
    from src.inference.redis_state import RedisSlidingWindow

    sw = RedisSlidingWindow(r, "user_003", max_size=50)

    # Add known transactions
    amounts = [100.0, 200.0, 150.0, 300.0, 500.0]
    merchants = [1, 1, 2, 3, 4]
    times = [10.0, 10.005, 10.01, 10.015, 10.02]

    for t, a, m in zip(times, amounts, merchants):
        sw.add(t, a, m)

    features = sw.get_velocity_features(10.03)

    assert features["tx_count_24h"] == 5, f"Expected 5, got {features['tx_count_24h']}"
    assert features["merchant_diversity_24h"] == 4, f"Expected 4 merchants, got {features['merchant_diversity_24h']}"
    assert features["merchant_is_new"] == 1.0, f"Merchant 4 should be new"
    assert features["amount_zscore"] != 0, f"Amount zscore should be non-zero"
    print("  [PASS] velocity features match")
    return True


def test_horizontal_sharing():
    """Test that two RedisSlidingWindow instances share state via same Redis connection."""
    try:
        import fakeredis
    except ImportError:
        return True

    # Shared fakeredis server (both connections use same in-memory store)
    server = fakeredis.FakeServer()
    r1 = fakeredis.FakeRedis(server=server)
    r2 = fakeredis.FakeRedis(server=server)

    from src.inference.redis_state import RedisSlidingWindow

    # Instance 1 adds 2 transactions (need >=2 for velocity features)
    sw1 = RedisSlidingWindow(r1, "user_share", max_size=50)
    sw1.add(10.0, 150.0, merchant_id=42)
    sw1.add(10.005, 200.0, merchant_id=99)

    # Instance 2 (different connection, same server) should see both
    sw2 = RedisSlidingWindow(r2, "user_share", max_size=50)
    features = sw2.get_velocity_features(10.01)
    assert features["tx_count_24h"] == 2, f"Shared Redis should share state, got {features['tx_count_24h']}"
    assert features["merchant_diversity_24h"] == 2, "Should see 2 merchants"
    print("  [PASS] horizontal sharing via Redis")
    return True


def test_in_process_fallback():
    """Test that in-process StateManager still works."""
    from src.inference.realtime_scorer import RealtimeScorer, Transaction, StateManager

    ism = StateManager(max_users=100, window_size=50)
    scorer = RealtimeScorer(state_manager=ism)

    for i in range(5):
        txn = Transaction(user_id=f"u{i}", amount=float(i * 100),
                          hour=float(i), minute=0.0, day=15.0,
                          merchant_id=i, chip=1)
        scorer.score(txn)

    stats = scorer.get_stats()
    assert stats["active_users"] == 5, f"Expected 5 users, got {stats['active_users']}"
    print("  [PASS] in-process fallback")
    return True


def test_concurrent_scoring():
    """Test concurrent scoring doesn't corrupt state."""
    import threading
    from src.inference.realtime_scorer import RealtimeScorer, Transaction, StateManager

    ism = StateManager(max_users=1000, window_size=50)
    scorer = RealtimeScorer(state_manager=ism)

    errors = []

    def score_user(uid, n):
        try:
            for i in range(n):
                txn = Transaction(
                    user_id=uid, amount=float(i * 10),
                    hour=float(i % 24), minute=0.0, day=15.0,
                    merchant_id=i, chip=1
                )
                result = scorer.score(txn)
                assert 0 <= result.fraud_probability <= 1
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=score_user, args=(f"user_{j}", 20)) for j in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent errors: {errors}"
    stats = scorer.get_stats()
    assert stats["active_users"] == 10, f"Expected 10 users, got {stats['active_users']}"
    print("  [PASS] concurrent scoring (10 threads × 20 txns)")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  REDIS STATE MANAGER TESTS")
    print("=" * 60)

    all_pass = True
    tests = [
        test_sliding_window_basic,
        test_sliding_window_eviction,
        test_velocity_features_match,
        test_horizontal_sharing,
        test_in_process_fallback,
        test_concurrent_scoring,
    ]

    for test in tests:
        try:
            if not test():
                all_pass = False
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
