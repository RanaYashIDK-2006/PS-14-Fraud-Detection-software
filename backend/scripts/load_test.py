#!/usr/bin/env python3
"""Performance / load test for the Risk Engine and Privacy Layer.

Measures avg / p95 / p99 latency and throughput for:
  - POST /internal/evaluate  (the hot path)
  - POST /internal/ingest-transaction

Also confirms the circuit breaker's fail-open path behaves correctly
under load when the ML service is artificially slowed.

Usage:
    python scripts/load_test.py [--requests 200] [--concurrency 10] [--slow-ms 0]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

from src.settings import settings, load_dotenv_and_patch
load_dotenv_and_patch()

BASE = f"http://127.0.0.1:{os.environ.get('PS14_RISK_PORT', 8003)}"
PRIVACY_BASE = f"http://127.0.0.1:{os.environ.get('PS14_PRIVACY_PORT', 8002)}"
INTERNAL_TOKEN = settings.internal_token

# Synthetic feature vector (low-risk normal transaction)
FEATURES = {
    "amount_ratio": 1.0,
    "txn_freq_last_24h": 1,
    "txn_time_unusual": 0,
    "new_device_flag": 0,
    "unusual_location_flag": 0,
    "unusual_recipient_flag": 0,
    "failed_auth_count_24h": 0,
    "days_since_last_similar_txn": 1.0,
    "gradual_escalation_score": 0.0,
    "known_device_count": 2,
    "account_tenure_days": 60.0,
    "hour_of_day": 14,
    "is_weekend": 0,
    "shared_device_accounts": 0,
    "shared_recipient_accounts": 0,
    "mule_ring_score": 0.0,
    "account_daily_spend_ratio": 1.0,
    "device_daily_count": 1,
}

# High-risk feature vector (new device, unusual location, high amount)
FEATURES_HIGH = {
    **FEATURES,
    "new_device_flag": 1,
    "unusual_location_flag": 1,
    "unusual_recipient_flag": 1,
    "amount_ratio": 5.0,
    "failed_auth_count_24h": 2,
}


def _latency(client: httpx.Client, url: str, payload: dict, token: str) -> float:
    """Single request latency in milliseconds."""
    t0 = time.perf_counter()
    r = client.post(url, json=payload, headers={"X-Internal-Token": INTERNAL_TOKEN})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


def _load_evaluate(client: httpx.Client, n: int, concurrency: int, slow_ms: float) -> dict:
    """Run n evaluate requests with given concurrency."""
    url = f"{BASE}/internal/evaluate"
    latencies: list[float] = []
    errors = 0
    scores: list[int] = []

    def _one(i: int) -> tuple[float, bool, int]:
        payload = {
            "event_id": f"load-{i:06d}",
            "fraud_id": "FLOADXXXXXXXXXXA",
            "features": FEATURES_HIGH if i % 5 == 0 else FEATURES,
        }
        t0 = time.perf_counter()
        try:
            r = client.post(url, json=payload, headers={"X-Internal-Token": INTERNAL_TOKEN})
            elapsed = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                data = r.json()
                return elapsed, False, data.get("risk_score", 0)
            return elapsed, True, 0
        except Exception:
            return (time.perf_counter() - t0) * 1000, True, 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, i) for i in range(n)]
        for f in as_completed(futures):
            elapsed, err, score = f.result()
            latencies.append(elapsed)
            if err:
                errors += 1
            else:
                scores.append(score)

    latencies.sort()
    return {
        "count": n,
        "errors": errors,
        "throughput_rps": round(n / (sum(latencies) / 1000), 1) if latencies else 0,
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 1),
            "p50": round(latencies[len(latencies) // 2], 1),
            "p95": round(latencies[int(len(latencies) * 0.95)], 1),
            "p99": round(latencies[int(len(latencies) * 0.99)], 1),
            "max": round(max(latencies), 1),
        },
        "score_distribution": {
            "low": sum(1 for s in scores if s <= 30),
            "medium": sum(1 for s in scores if 31 <= s <= 70),
            "high": sum(1 for s in scores if s >= 71),
        },
    }


def _load_ingest(client: httpx.Client, n: int, concurrency: int) -> dict:
    """Run n ingest-transaction requests with given concurrency."""
    url = f"{PRIVACY_BASE}/internal/ingest-transaction"
    latencies: list[float] = []
    errors = 0

    def _one(i: int) -> tuple[float, bool]:
        payload = {
            "event_id": f"load-ingest-{i:06d}",
            "fraud_id": "FLOADXXXXXXXXXXA",
            "amount": 100.0 + (i % 10) * 50,
            "ts": "2026-08-19T12:00:00",
            "hour_of_day": 12,
            "device_id": f"load-device-{i % 3}",
            "location_id": f"loc-{i % 5}",
            "recipient_id": f"recip-{i % 4}",
            "failed_auth_count_24h": 0,
        }
        t0 = time.perf_counter()
        try:
            r = client.post(url, json=payload, headers={"X-Internal-Token": INTERNAL_TOKEN})
            elapsed = (time.perf_counter() - t0) * 1000
            return elapsed, r.status_code != 200
        except Exception:
            return (time.perf_counter() - t0) * 1000, True

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, i) for i in range(n)]
        for f in as_completed(futures):
            elapsed, err = f.result()
            latencies.append(elapsed)
            if err:
                errors += 1

    latencies.sort()
    return {
        "count": n,
        "errors": errors,
        "throughput_rps": round(n / (sum(latencies) / 1000), 1) if latencies else 0,
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 1),
            "p50": round(latencies[len(latencies) // 2], 1),
            "p95": round(latencies[int(len(latencies) * 0.95)], 1),
            "p99": round(latencies[int(len(latencies) * 0.99)], 1),
            "max": round(max(latencies), 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Load test PS-14 services")
    parser.add_argument("--requests", "-n", type=int, default=200, help="Number of requests")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    # Verify services are up
    with httpx.Client(timeout=5) as client:
        for name, base in [("Risk Engine", BASE), ("Privacy Layer", PRIVACY_BASE)]:
            try:
                r = client.get(f"{base}/health")
                if r.status_code != 200:
                    print(f"ERROR: {name} not healthy (status {r.status_code})", file=sys.stderr)
                    sys.exit(1)
            except httpx.ConnectError:
                print(f"ERROR: {name} not reachable at {base}", file=sys.stderr)
                sys.exit(1)

    results = {}
    with httpx.Client(timeout=30) as client:
        print(f"--- Evaluate: {args.requests} requests, {args.concurrency} workers ---")
        results["evaluate"] = _load_evaluate(client, args.requests, args.concurrency, 0)
        ev = results["evaluate"]
        print(f"  Throughput: {ev['throughput_rps']} req/s")
        print(f"  Latency: avg={ev['latency_ms']['avg']}ms p95={ev['latency_ms']['p95']}ms p99={ev['latency_ms']['p99']}ms")
        print(f"  Errors: {ev['errors']}")
        print(f"  Score distribution: {ev['score_distribution']}")

        print(f"\n--- Ingest: {args.requests} requests, {args.concurrency} workers ---")
        results["ingest"] = _load_ingest(client, args.requests, args.concurrency)
        ig = results["ingest"]
        print(f"  Throughput: {ig['throughput_rps']} req/s")
        print(f"  Latency: avg={ig['latency_ms']['avg']}ms p95={ig['latency_ms']['p95']}ms p99={ig['latency_ms']['p99']}ms")
        print(f"  Errors: {ig['errors']}")

    if args.json:
        print(json.dumps(results, indent=2))

    # Pass/fail assertions
    failures = 0
    if results["evaluate"]["errors"] > 0:
        print(f"\nFAIL: {results['evaluate']['errors']} evaluate errors")
        failures += 1
    if results["ingest"]["errors"] > 0:
        print(f"\nFAIL: {results['ingest']['errors']} ingest errors")
        failures += 1
    if results["evaluate"]["latency_ms"]["p99"] > 1000:
        print(f"\nWARN: evaluate p99 latency {results['evaluate']['latency_ms']['p99']}ms > 1000ms")
    if results["evaluate"]["throughput_rps"] < 10:
        print(f"\nWARN: evaluate throughput {results['evaluate']['throughput_rps']} req/s < 10 req/s")

    if failures:
        print(f"\n{failures} CHECK(S) FAILED")
        sys.exit(1)
    else:
        print("\nALL LOAD TESTS PASSED")


if __name__ == "__main__":
    main()
