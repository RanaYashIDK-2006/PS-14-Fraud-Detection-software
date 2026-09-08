#!/usr/bin/env python3
"""Test worker pool manager: metrics collection, auto-scaling, worker lifecycle."""
import sys
import time
sys.path.insert(0, ".")

import redis
from src.inference.worker_pool import (
    RedisMetricsCollector, AutoScaler, WorkerPoolManager, WorkerInfo
)


def test_metrics_collection():
    """Test Redis metrics collector."""
    print("\n[1] Redis metrics collection")
    r = redis.Redis(host="localhost", port=6379, db=0, protocol=2)
    collector = RedisMetricsCollector(r)

    metrics = collector.collect()
    assert "timestamp" in metrics
    assert "active_users" in metrics
    assert "redis_memory_mb" in metrics
    assert "connected_clients" in metrics
    print(f"  PASS: metrics={ {k: v for k, v in metrics.items() if k != 'timestamp'} }")

    # Collect again and check history
    time.sleep(0.1)
    collector.collect()
    assert len(collector.history) == 2
    print(f"  PASS: history length={len(collector.history)}")

    # Trend
    trend = collector.get_trend("redis_memory_mb")
    assert "trend" in trend
    assert "current" in trend
    print(f"  PASS: trend={trend}")

    return True


def test_auto_scaler():
    """Test auto-scaling decisions."""
    print("\n[2] Auto-scaler decisions")
    scaler = AutoScaler(
        min_workers=2, max_workers=8,
        target_latency_ms=50.0,
        scale_up_consecutive=2,
        scale_down_consecutive=3,
        cooldown_seconds=0,  # no cooldown for testing
    )

    # Normal load → hold
    decision = scaler.evaluate(
        metrics={"active_users": 100, "redis_memory_mb": 1.0},
        current_workers=4,
        worker_latencies=[30.0, 35.0, 40.0, 32.0],
        worker_errors=[0, 0, 0, 0],
        worker_requests=[100, 100, 100, 100],
    )
    assert decision["action"] == "hold"
    print(f"  PASS: normal load → hold (latency={decision['avg_latency_ms']}ms)")

    # High latency → scale up (after consecutive checks)
    for _ in range(2):
        decision = scaler.evaluate(
            metrics={"active_users": 500, "redis_memory_mb": 5.0},
            current_workers=4,
            worker_latencies=[80.0, 90.0, 85.0, 88.0],
            worker_errors=[0, 0, 0, 0],
            worker_requests=[500, 500, 500, 500],
        )
    assert decision["action"] == "scale_up"
    assert decision["target_workers"] == 5
    print(f"  PASS: high latency → scale up to {decision['target_workers']}")

    # Low latency → scale down (after consecutive checks)
    for _ in range(3):
        decision = scaler.evaluate(
            metrics={"active_users": 10, "redis_memory_mb": 0.5},
            current_workers=5,
            worker_latencies=[10.0, 12.0, 11.0, 9.0, 13.0],
            worker_errors=[0, 0, 0, 0, 0],
            worker_requests=[10, 10, 10, 10, 10],
        )
    assert decision["action"] == "scale_down"
    assert decision["target_workers"] == 4
    print(f"  PASS: low latency → scale down to {decision['target_workers']}")

    # At min workers → hold
    for _ in range(3):
        decision = scaler.evaluate(
            metrics={"active_users": 5, "redis_memory_mb": 0.1},
            current_workers=2,
            worker_latencies=[5.0, 8.0],
            worker_errors=[0, 0],
            worker_requests=[5, 5],
        )
    assert decision["action"] == "hold"
    assert decision["current_workers"] == 2
    print(f"  PASS: at min workers → hold")

    # At max workers → hold
    scaler2 = AutoScaler(min_workers=2, max_workers=4, target_latency_ms=50.0,
                         scale_up_consecutive=1, cooldown_seconds=0)
    for _ in range(2):
        decision = scaler2.evaluate(
            metrics={"active_users": 1000, "redis_memory_mb": 50.0},
            current_workers=4,
            worker_latencies=[100.0, 110.0, 95.0, 105.0],
            worker_errors=[0, 0, 0, 0],
            worker_requests=[1000, 1000, 1000, 1000],
        )
    assert decision["action"] == "hold"
    assert decision["current_workers"] == 4
    print(f"  PASS: at max workers → hold")

    return True


def test_worker_info():
    """Test WorkerInfo dataclass."""
    print("\n[3] WorkerInfo dataclass")
    w = WorkerInfo(port=8006, pid=1234, started_at=time.time() - 60)
    assert w.uptime > 59
    assert w.url == "http://127.0.0.1:8006"

    d = w.to_dict()
    assert "port" in d
    assert "pid" in d
    assert "uptime_s" in d
    print(f"  PASS: WorkerInfo={d}")

    return True


def test_pool_manager_status():
    """Test pool manager status without actually starting workers."""
    print("\n[4] Pool manager status")
    pool = WorkerPoolManager(
        base_port=8099,  # high port to avoid conflicts
        min_workers=0,
        max_workers=2,
        redis_url="redis://localhost:6379/0",
    )

    status = pool.get_status()
    assert "running" in status
    assert "workers" in status
    assert "n_workers" in status
    assert status["running"] is False
    assert status["n_workers"] == 0
    print(f"  PASS: status={ {k: v for k, v in status.items() if k not in ('workers', 'metrics', 'request_rate_trend', 'recent_scaling')} }")

    return True


def test_port_detection():
    """Test port-in-use detection."""
    print("\n[5] Port detection")
    pool = WorkerPoolManager(base_port=8099)

    # Port 8000 should be in use (front service)
    assert pool._is_port_in_use(8000), "Port 8000 should be in use"
    print("  PASS: port 8000 detected as in-use")

    # Random high port should be free
    assert not pool._is_port_in_use(19999), "Port 19999 should be free"
    print("  PASS: port 19999 detected as free")

    return True


def main():
    print("=" * 60)
    print("WORKER POOL MANAGER TEST")
    print("=" * 60)

    tests = [
        test_metrics_collection,
        test_auto_scaler,
        test_worker_info,
        test_pool_manager_status,
        test_port_detection,
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
