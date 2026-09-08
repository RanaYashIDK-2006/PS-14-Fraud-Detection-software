#!/usr/bin/env python3
"""Test Redis-backed per-user and global rate limiting."""
import sys
import time
sys.path.insert(0, ".")

import redis
from src.inference.rate_limiter import RedisRateLimiter


def test_per_user_limit():
    """Test per-user rate limiting."""
    print("\n[1] Per-user rate limiting")
    r = redis.Redis(host="localhost", port=6379, db=0, protocol=2)
    limiter = RedisRateLimiter(r, per_user_limit=5, global_limit=1000, window_seconds=10)

    # Should allow first 5
    for i in range(5):
        result = limiter.check("test_user")
        assert result.allowed, f"Request {i+1} should be allowed"
    print(f"  PASS: first 5 requests allowed (user_count={result.user_count})")

    # 6th should be blocked
    result = limiter.check("test_user")
    assert not result.allowed, "Request 6 should be blocked"
    assert result.retry_after_seconds > 0
    print(f"  PASS: request 6 blocked (retry_after={result.retry_after_seconds:.1f}s)")

    # Different user should still work
    result = limiter.check("other_user")
    assert result.allowed, "Different user should be allowed"
    print(f"  PASS: different user allowed (user_count={result.user_count})")

    # Check headers
    headers = result.to_headers()
    assert "X-RateLimit-Limit" in headers
    assert "X-RateLimit-Remaining" in headers
    print(f"  PASS: headers present: {list(headers.keys())}")

    limiter.reset_user("test_user")
    limiter.reset_user("other_user")
    return True


def test_global_limit():
    """Test global rate limiting."""
    print("\n[2] Global rate limiting")
    r = redis.Redis(host="localhost", port=6379, db=0, protocol=2)
    r.delete("rl:global")  # Reset global state from previous tests
    limiter = RedisRateLimiter(r, per_user_limit=1000, global_limit=10, window_seconds=10)

    # Exhaust global limit
    for i in range(10):
        result = limiter.check(f"user_{i}")
        assert result.allowed, f"Request {i+1} should be allowed"
    print(f"  PASS: first 10 requests allowed (global_count={result.global_count})")

    # 11th should be blocked (global limit)
    result = limiter.check("user_new")
    assert not result.allowed, "Request 11 should be blocked by global limit"
    assert result.global_count >= 10
    print(f"  PASS: request 11 blocked by global limit (global_count={result.global_count})")

    limiter.reset_global()
    return True


def test_check_only():
    """Test dry-run check (no recording)."""
    print("\n[3] Dry-run check (check_only)")
    r = redis.Redis(host="localhost", port=6379, db=0, protocol=2)
    limiter = RedisRateLimiter(r, per_user_limit=3, global_limit=1000, window_seconds=10)

    # Check-only shouldn't consume quota
    for _ in range(5):
        result = limiter.check_only("dry_user")
        assert result.allowed, "check_only should always allow"
    print(f"  PASS: 5 check_only calls all allowed")

    # Real check should still work (quota not consumed)
    result = limiter.check("dry_user")
    assert result.allowed, "First real check should be allowed"
    assert result.user_count == 1, f"Expected user_count=1, got {result.user_count}"
    print(f"  PASS: first real check allowed (user_count={result.user_count})")

    limiter.reset_user("dry_user")
    return True


def test_window_expiry():
    """Test that old entries expire."""
    print("\n[4] Window expiry")
    r = redis.Redis(host="localhost", port=6379, db=0, protocol=2)
    # Very short window for testing
    limiter = RedisRateLimiter(r, per_user_limit=2, global_limit=1000, window_seconds=1)

    # Use up limit
    limiter.check("expiry_user")
    limiter.check("expiry_user")
    result = limiter.check("expiry_user")
    assert not result.allowed, "Should be blocked"
    print(f"  PASS: blocked after using limit")

    # Wait for window to expire
    time.sleep(1.5)

    # Should be allowed again
    result = limiter.check("expiry_user")
    assert result.allowed, f"Should be allowed after window expiry, got allowed={result.allowed}"
    print(f"  PASS: allowed after window expiry")

    limiter.reset_user("expiry_user")
    return True


def test_stats():
    """Test rate limiter statistics."""
    print("\n[5] Statistics")
    r = redis.Redis(host="localhost", port=6379, db=0, protocol=2)
    limiter = RedisRateLimiter(r, per_user_limit=5, global_limit=100, window_seconds=60)

    # Generate some traffic
    for i in range(3):
        limiter.check(f"stats_user_{i}")

    stats = limiter.get_stats()
    assert stats["total_checked"] >= 3
    assert stats["per_user_limit"] == 5
    assert stats["global_limit"] == 100
    print(f"  PASS: stats={stats}")

    return True


def test_to_dict():
    """Test result serialization."""
    print("\n[6] Result serialization")
    r = redis.Redis(host="localhost", port=6379, db=0, protocol=2)
    limiter = RedisRateLimiter(r, per_user_limit=10, global_limit=50, window_seconds=30)

    result = limiter.check("dict_user")
    d = result.to_dict()
    assert "allowed" in d
    assert "user_count" in d
    assert "global_count" in d
    assert "retry_after_seconds" in d
    assert "user_remaining" in d
    print(f"  PASS: dict keys={list(d.keys())}")

    limiter.reset_user("dict_user")
    return True


def main():
    print("=" * 60)
    print("REDIS RATE LIMITER TEST")
    print("=" * 60)

    tests = [
        test_per_user_limit,
        test_global_limit,
        test_check_only,
        test_window_expiry,
        test_stats,
        test_to_dict,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{passed + failed} passed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failed} TEST(S) FAILED")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
