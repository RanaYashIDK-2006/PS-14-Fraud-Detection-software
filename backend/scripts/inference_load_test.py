#!/usr/bin/env python3
"""Load test for the Real-Time Inference Service (port 8006).

Tests /score, /score-batch, and /ab/score endpoints under concurrent load.
Measures latency percentiles, throughput, error rates, and Redis state consistency.

Usage:
    python scripts/inference_load_test.py [--requests 500] [--concurrency 20]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

BASE = "http://127.0.0.1:8006"


def random_transaction(user_id: str = None) -> dict:
    """Generate a random transaction matching the ScoreRequest schema."""
    return {
        "user_id": user_id or f"load_user_{random.randint(0, 9999)}",
        "amount": round(random.expovariate(0.01), 2),
        "hour": float(random.randint(0, 23)),
        "minute": float(random.randint(0, 59)),
        "day": float(random.randint(1, 28)),
        "merchant_id": random.randint(0, 5000),
        "city_id": random.randint(0, 200),
        "chip": random.choice([0, 1, 2]),
        "mcc": round(random.uniform(0, 1), 4),
        "is_online": random.choice([0, 1]),
        "is_error": random.choice([0, 0, 0, 1]),
        "card": float(random.randint(0, 5)),
        "year": 2024.0,
        "month": float(random.randint(1, 12)),
    }


def test_health(client: httpx.Client) -> dict:
    """Verify service is healthy."""
    r = client.get(f"{BASE}/health", timeout=5)
    r.raise_for_status()
    return r.json()


def test_single_score(client: httpx.Client, txn: dict) -> dict:
    """Score a single transaction."""
    t0 = time.perf_counter()
    r = client.post(f"{BASE}/score", json=txn, timeout=5)
    latency_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    data["_latency_ms"] = round(latency_ms, 3)
    return data


def test_batch_score(client: httpx.Client, txns: list[dict]) -> dict:
    """Score a batch of transactions."""
    t0 = time.perf_counter()
    r = client.post(f"{BASE}/score-batch", json={"transactions": txns}, timeout=30)
    latency_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    data["_total_latency_ms"] = round(latency_ms, 3)
    return data


def test_ab_score(client: httpx.Client, txn: dict, experiment: str = "default") -> dict:
    """Score through A/B routing."""
    payload = {**txn, "experiment": experiment}
    t0 = time.perf_counter()
    r = client.post(f"{BASE}/ab/score", json=payload, timeout=5)
    latency_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    data["_latency_ms"] = round(latency_ms, 3)
    return data


def test_concurrent_single(client: httpx.Client, n: int, concurrency: int) -> list[dict]:
    """Run n single-score requests with given concurrency."""
    results = []
    errors = []

    def _do_one(i: int):
        txn = random_transaction()
        try:
            return test_single_score(client, txn)
        except Exception as e:
            return {"error": str(e), "_latency_ms": 0}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_do_one, i): i for i in range(n)}
        for f in as_completed(futures):
            results.append(f.result())

    return results


def test_concurrent_batch(client: httpx.Client, n_batches: int, batch_size: int,
                          concurrency: int) -> list[dict]:
    """Run n batch-score requests with given concurrency."""
    results = []

    def _do_one(i: int):
        txns = [random_transaction() for _ in range(batch_size)]
        try:
            return test_batch_score(client, txns)
        except Exception as e:
            return {"error": str(e), "_total_latency_ms": 0}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_do_one, i): i for i in range(n_batches)}
        for f in as_completed(futures):
            results.append(f.result())

    return results


def test_concurrent_ab(client: httpx.Client, n: int, concurrency: int) -> list[dict]:
    """Run n A/B-score requests with given concurrency."""
    results = []

    def _do_one(i: int):
        txn = random_transaction()
        try:
            return test_ab_score(client, txn)
        except Exception as e:
            return {"error": str(e), "_latency_ms": 0}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_do_one, i): i for i in range(n)}
        for f in as_completed(futures):
            results.append(f.result())

    return results


def compute_stats(latencies: list[float], label: str) -> dict:
    """Compute latency statistics."""
    if not latencies:
        return {"label": label, "n": 0}
    valid = [l for l in latencies if l > 0]
    sorted_l = sorted(valid)
    return {
        "label": label,
        "n": len(valid),
        "errors": len(latencies) - len(valid),
        "mean_ms": round(statistics.mean(valid), 3),
        "median_ms": round(statistics.median(valid), 3),
        "p95_ms": round(sorted_l[int(len(sorted_l) * 0.95)], 3) if len(sorted_l) > 1 else valid[0],
        "p99_ms": round(sorted_l[int(len(sorted_l) * 0.99)], 3) if len(sorted_l) > 1 else valid[0],
        "min_ms": round(min(valid), 3),
        "max_ms": round(max(valid), 3),
        "throughput_tps": round(1000 / statistics.mean(valid), 0) if valid else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference service load test")
    parser.add_argument("--requests", type=int, default=500, help="Number of requests per test")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent workers")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for /score-batch")
    parser.add_argument("--batch-count", type=int, default=20, help="Number of batch requests")
    args = parser.parse_args()

    print("=" * 70)
    print("  INFERENCE SERVICE LOAD TEST")
    print(f"  Target: {BASE} | Requests: {args.requests} | Concurrency: {args.concurrency}")
    print("=" * 70)

    client = httpx.Client(base_url=BASE, timeout=10)

    # ── Health check ──
    print("\n[1] Health check...")
    health = test_health(client)
    print(f"  Status: {health['status']} | Model: {health.get('model_version', '?')}")
    print(f"  Active users: {health.get('active_users', 0)} | State: {health.get('state_backend', '?')}")
    print(f"  Registered models: {health.get('registered_models', 0)} | A/B experiments: {health.get('ab_experiments', 0)}")

    if health["status"] != "ok":
        print("  ERROR: Service not healthy, aborting")
        sys.exit(1)

    # ── Warmup ──
    print("\n[2] Warmup (10 requests)...")
    for _ in range(10):
        test_single_score(client, random_transaction())
    print("  Done")

    # ── Test 1: Single-score throughput ──
    print(f"\n[3] Single-score load test ({args.requests} requests, {args.concurrency} workers)...")
    t0 = time.time()
    results = test_concurrent_single(client, args.requests, args.concurrency)
    wall_time = time.time() - t0

    latencies = [r.get("_latency_ms", 0) for r in results]
    errors = sum(1 for r in results if "error" in r)
    stats = compute_stats(latencies, "single-score")
    actual_tps = args.requests / wall_time

    print(f"  Completed: {args.requests - errors}/{args.requests} ({errors} errors)")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Latency: mean={stats['mean_ms']}ms  p50={stats['median_ms']}ms  "
          f"p95={stats['p95_ms']}ms  p99={stats['p99_ms']}ms")
    print(f"  Throughput: {actual_tps:.0f} req/s (actual) | {stats['throughput_tps']:.0f} req/s (from latency)")

    # ── Test 2: Batch-score throughput ──
    print(f"\n[4] Batch-score load test ({args.batch_count} batches × {args.batch_size}, {args.concurrency} workers)...")
    t0 = time.time()
    batch_results = test_concurrent_batch(client, args.batch_count, args.batch_size, args.concurrency)
    wall_time = time.time() - t0

    batch_latencies = [r.get("_total_latency_ms", 0) for r in batch_results]
    batch_errors = sum(1 for r in batch_results if "error" in r)
    total_scored = sum(r.get("batch_size", 0) for r in batch_results if "error" not in r)
    batch_stats = compute_stats(batch_latencies, "batch-score")
    actual_tps = total_scored / wall_time if wall_time > 0 else 0

    print(f"  Completed: {args.batch_count - batch_errors}/{args.batch_count} batches ({batch_errors} errors)")
    print(f"  Total scored: {total_scored} transactions in {wall_time:.1f}s")
    print(f"  Batch latency: mean={batch_stats['mean_ms']}ms  p50={batch_stats['median_ms']}ms  "
          f"p95={batch_stats['p95_ms']}ms")
    print(f"  Throughput: {actual_tps:.0f} txn/s (actual)")

    # ── Test 3: A/B-score throughput ──
    print(f"\n[5] A/B-score load test ({args.requests} requests, {args.concurrency} workers)...")
    t0 = time.time()
    ab_results = test_concurrent_ab(client, args.requests, args.concurrency)
    wall_time = time.time() - t0

    ab_latencies = [r.get("_latency_ms", 0) for r in ab_results]
    ab_errors = sum(1 for r in ab_results if "error" in r)
    ab_stats = compute_stats(ab_latencies, "ab-score")
    actual_tps = args.requests / wall_time

    # Count version distribution
    version_counts = {}
    for r in ab_results:
        v = r.get("model_version", "unknown")
        version_counts[v] = version_counts.get(v, 0) + 1

    print(f"  Completed: {args.requests - ab_errors}/{args.requests} ({ab_errors} errors)")
    print(f"  Latency: mean={ab_stats['mean_ms']}ms  p50={ab_stats['median_ms']}ms  "
          f"p95={ab_stats['p95_ms']}ms  p99={ab_stats['p99_ms']}ms")
    print(f"  Throughput: {actual_tps:.0f} req/s")
    print(f"  Version distribution: {version_counts}")

    # ── Test 4: State consistency ──
    print(f"\n[6] State consistency check...")
    uid = f"consistency_test_{int(time.time())}"

    # Reset state for this user first
    try:
        client.delete(f"/state/{uid}")
    except Exception:
        pass

    # Send 10 sequential transactions, verify velocity features accumulate
    for i in range(10):
        txn = random_transaction(user_id=uid)
        txn["amount"] = 100.0 + i * 50
        txn["hour"] = 14.0
        txn["minute"] = 30.0
        txn["day"] = 15.0
        test_single_score(client, txn)

    # Final check
    final_txn = random_transaction(user_id=uid)
    final_txn["amount"] = 999.0
    final_txn["hour"] = 14.0
    final_txn["minute"] = 30.0
    final_txn["day"] = 15.0
    result = test_single_score(client, final_txn)
    velocity = result.get("velocity_features", {})
    tx_count = velocity.get("tx_count_24h", 0)

    print(f"  User {uid}: tx_count_24h={tx_count} (expected=11)")
    state_ok = tx_count >= 10
    print(f"  State accumulation: {'PASS' if state_ok else 'FAIL'}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  LOAD TEST RESULTS")
    print("=" * 70)
    print(f"  Single-score:  {stats['mean_ms']}ms mean, {actual_tps:.0f} txn/s")
    print(f"  Batch-score:   {batch_stats['mean_ms']}ms mean, {actual_tps:.0f} txn/s (from batch test)")
    print(f"  A/B-score:     {ab_stats['mean_ms']}ms mean, {actual_tps:.0f} txn/s")
    print(f"  State check:   {'PASS' if state_ok else 'FAIL'}")
    print(f"  Error rate:    {(errors + batch_errors + ab_errors) / (args.requests * 2 + args.batch_count) * 100:.1f}%")
    print("=" * 70)

    # Save results
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"requests": args.requests, "concurrency": args.concurrency,
                    "batch_size": args.batch_size, "batch_count": args.batch_count},
        "health": health,
        "single_score": stats,
        "batch_score": {**batch_stats, "total_scored": total_scored},
        "ab_score": {**ab_stats, "version_distribution": version_counts},
        "state_consistency": {"tx_count": tx_count, "pass": state_ok},
        "total_errors": errors + batch_errors + ab_errors,
    }
    out = ROOT / "reports" / "inference_load_test.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {out}")

    client.close()


if __name__ == "__main__":
    main()
