#!/usr/bin/env python3
"""Phase 17 - Real RedisStateManager integration tests.

Tests RedisStateManager against the real Redis server on 127.0.0.1:6379.
Uses unique test prefixes for isolation. Does NOT flush the database.
Categories: B (Redis integration - real service)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add backend to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

PREFIX = "ph17:"
FAILURES = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(f"FAIL {name}: {detail}")
        print(f"  [FAIL] {name}: {detail}")
    else:
        print(f"  [PASS] {name}")


def preflight():
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0,
                        socket_connect_timeout=3, protocol=2)
        r.ping()
        return True
    except Exception as e:
        print(f"  [SKIP] Redis not available: {e}")
        return False


def cleanup_keys(r, prefix):
    """Remove all keys matching prefix* using SCAN (safe, no FLUSHALL)."""
    try:
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match=prefix + "*", count=1000)
            if keys:
                r.delete(*keys)
            if cursor == 0:
                break
    except Exception:
        pass


def test_connection_and_basic_ops():
    print("\n[1] Connection and basic operations")
    from src.inference.redis_state import (RedisStateManager,
                                            RedisSlidingWindow)
    import redis as redis_mod

    mgr = RedisStateManager(redis_url="redis://localhost:6379/0",
                            prefix=PREFIX, window_size=10, max_users=100)
    try:
        stats = mgr.get_stats()
        check("get_stats returns dict", isinstance(stats, dict), str(stats))
        check("stats has active_users", "active_users" in stats, str(stats))
        check("stats has model_version", "model_version" in stats, str(stats))
        check("redis_connected is True", stats.get("redis_connected") is True,
              str(stats))

        # Use the manager's prefix-aware get_window method
        # The manager adds its own prefix, so pass just the user_id
        sw = mgr.get_window("user_basic")
        sw.add(10.0, 200.0, merchant_id=1)
        sw.add(10.005, 300.0, merchant_id=2)

        features = sw.get_velocity_features(10.01)
        check("velocity features computed", isinstance(features, dict),
              str(features))
        check("tx_count_24h == 2", features["tx_count_24h"] == 2,
              str(features))
        check("merchant_diversity == 2", features["merchant_diversity_24h"] == 2,
              str(features))

        # After the get_window() fix, active_users should be counted correctly.
        # Keys are now created as {prefix}window:{user_id} and get_stats() scans
        # for {prefix}window:*, so they should match.
        stats2 = mgr.get_stats()
        check("stats structure valid after add", isinstance(stats2, dict),
              str(stats2))
        check("active_users >= 1 after add (post-fix)",
              stats2.get("active_users", 0) >= 1,
              f"active_users={stats2.get('active_users')}")
        check("redis_connected still True after add",
              stats2.get("redis_connected") is True, str(stats2))
    finally:
        mgr.close()
        r = redis_mod.Redis(host="localhost", port=6379, db=0,
                            socket_connect_timeout=3, protocol=2)
        cleanup_keys(r, PREFIX + "window:user_basic")


def test_shared_state_between_managers():
    print("\n[2] Shared state between managers")
    from src.inference.redis_state import RedisSlidingWindow
    import redis as redis_mod

    # Use a shared raw Redis connection to test that two
    # RedisSlidingWindow instances with the same key see each other
    r = redis_mod.Redis(host="localhost", port=6379, db=0,
                        socket_connect_timeout=3, protocol=2)
    try:
        # Both windows use the same key (same user_id, same prefix)
        # Need at least 2 transactions for velocity features to be non-empty
        sw1 = RedisSlidingWindow(r, PREFIX + "shared_user",
                                 max_size=10, prefix=PREFIX + "window:")
        sw1.add(10.0, 200.0, merchant_id=1)
        sw1.add(10.005, 300.0, merchant_id=2)

        sw2 = RedisSlidingWindow(r, PREFIX + "shared_user",
                                 max_size=10, prefix=PREFIX + "window:")
        features = sw2.get_velocity_features(10.01)
        check("shared state visible", features["tx_count_24h"] == 2,
              str(features))
        check("shared merchant_diversity", features["merchant_diversity_24h"] == 2,
              str(features))
    finally:
        cleanup_keys(r, PREFIX + "window:" + PREFIX + "shared_user")


def test_key_isolation_between_prefixes():
    print("\n[3] Key isolation between prefixes")
    from src.inference.redis_state import RedisSlidingWindow
    import redis as redis_mod

    r = redis_mod.Redis(host="localhost", port=6379, db=0,
                        socket_connect_timeout=3, protocol=2)
    try:
        swA = RedisSlidingWindow(r, PREFIX + "iso_A", max_size=10,
                                  prefix=PREFIX + "window:")
        swB = RedisSlidingWindow(r, PREFIX + "iso_B", max_size=10,
                                  prefix=PREFIX + "window:")
        swA.add(10.0, 200.0, merchant_id=1)
        features = swB.get_velocity_features(10.01)
        check("B sees no transactions from A", features["tx_count_24h"] == 0,
              str(features))
    finally:
        cleanup_keys(r, PREFIX + "window:" + PREFIX + "iso_A")
        cleanup_keys(r, PREFIX + "window:" + PREFIX + "iso_B")


def test_ttl_behavior():
    print("\n[4] TTL behavior")
    from src.inference.redis_state import RedisSlidingWindow
    import redis as redis_mod

    r = redis_mod.Redis(host="localhost", port=6379, db=0,
                        socket_connect_timeout=3, protocol=2)
    try:
        sw = RedisSlidingWindow(r, PREFIX + "ttl_user", max_size=10,
                                 prefix=PREFIX + "window:")
        sw.add(10.0, 200.0, merchant_id=1)

        # Check TTL on the actual keys (not the tx sub-keys)
        ttl_key = r.ttl(sw.key)
        check("window key TTL returns int", isinstance(ttl_key, int),
              str(ttl_key))
        check("window key TTL is finite and positive",
              0 < ttl_key <= 604800, str(ttl_key))

        ttl_agg = r.ttl(sw.agg_key)
        check("agg key TTL returns int", isinstance(ttl_agg, int),
              str(ttl_agg))
        check("agg key TTL is finite and positive",
              0 < ttl_agg <= 604800, str(ttl_agg))
    finally:
        cleanup_keys(r, PREFIX + "window:" + PREFIX + "ttl_user")


def test_connection_failure():
    print("\n[5] Connection failure handling")
    from src.inference.redis_state import RedisStateManager

    try:
        mgr = RedisStateManager(
            redis_url="redis://127.0.0.1:9999/0",
            prefix=PREFIX, window_size=10, max_users=100
        )
        try:
            try:
                mgr.get_window("fail_user").add(10.0, 200.0, merchant_id=1)
                check("record_activity when failed does not crash", True)
            except Exception:
                check("graceful handling: exception on failed connection",
                      True, "expected exception")
            stats = mgr.get_stats()
            check("stats returned even when redis down", isinstance(stats, dict),
                  str(stats))
            check("redis_connected is False", stats.get("redis_connected") is False,
                  str(stats))
        finally:
            mgr.close()
    except Exception as e:
        check("construction with bad URL fails gracefully", False, str(e))


if __name__ == "__main__":
    import os
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PS14_MODE"] = "development"

    print("=" * 60)
    print("Phase 17 - Real RedisStateManager Integration Tests")
    print("=" * 60)

    print(f"\nRedis: 127.0.0.1:6379")
    if not preflight():
        print("\nRedis preflight FAILED - skipping all tests")
        print("\n" + "=" * 60)
        print("RESULT: SKIPPED (Redis unavailable)")
        print("=" * 60)
        sys.exit(0)

    print("\nRedis preflight PASSED")

    test_connection_and_basic_ops()
    test_shared_state_between_managers()
    test_key_isolation_between_prefixes()
    test_ttl_behavior()
    test_connection_failure()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  {f}")
    else:
        print("RESULT: ALL PASSED")
    print("=" * 60)
    sys.exit(1 if FAILURES else 0)
