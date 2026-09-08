#!/usr/bin/env python3
"""Stress test the unified scorer at progressive concurrency levels.

Tests: /unified-score, /score, /unified-score-batch
Measures: throughput, latency percentiles, error rate, memory
Find the breaking point where errors spike or latency degrades beyond SLOs.
"""
import asyncio
import time
import json
import statistics
import sys
from pathlib import Path
from dataclasses import dataclass, field

import httpx

URL = "http://127.0.0.1:8006"
CONCURRENCY_LEVELS = [10, 50, 100, 250, 500, 750, 1000]
REQUESTS_PER_LEVEL = 500
TIMEOUT_S = 15
REPORT_PATH = Path("reports/stress_test.json")

TX_TEMPLATE = {
    "amount_ratio": 1.5, "hour_of_day": 14, "is_weekend": 0,
    "new_device_flag": 0, "txn_freq_last_24h": 5, "known_device_count": 3,
    "account_tenure_days": 180, "gradual_escalation_score": 0.3,
    "unusual_location_flag": 0, "unusual_recipient_flag": 0,
    "failed_auth_count_24h": 0, "hour_deviation": 0.2,
    "amount_zscore": 0.5, "velocity_deviation": 0.3,
    "shared_device_accounts": 0, "shared_recipient_accounts": 0,
    "mule_ring_score": 0.1, "txn_amount_bucket": 2,
    "txn_time_unusual": 0, "days_since_last_similar_txn": 7,
    "amount_ratio_sq": 2.25,
}

import random

def make_tx(i):
    tx = TX_TEMPLATE.copy()
    tx["amount_ratio"] = round(random.uniform(0.5, 5.0), 2)
    tx["hour_of_day"] = i % 24
    tx["txn_freq_last_24h"] = (i % 20) + 1
    return tx


@dataclass
class Result:
    endpoint: str
    concurrency: int
    total_requests: int
    success: int = 0
    errors: int = 0
    timeouts: int = 0
    latencies: list = field(default_factory=list)
    status_codes: dict = field(default_factory=dict)

    @property
    def error_rate(self):
        return round(self.errors / max(self.total_requests, 1) * 100, 2)

    @property
    def throughput(self):
        wt = getattr(self, '_wall_time', None)
        if wt:
            return round(self.success / max(wt, 0.01), 1)
        total_time = sum(self.latencies) / 1000 if self.latencies else 1
        return round(self.success / max(total_time, 0.01), 1)

    def percentiles(self):
        if not self.latencies:
            return {}
        s = sorted(self.latencies)
        n = len(s)
        return {
            "p50_ms": round(s[n // 2], 2),
            "p90_ms": round(s[int(n * 0.90)], 2),
            "p95_ms": round(s[int(n * 0.95)], 2),
            "p99_ms": round(s[int(n * 0.99)], 2),
            "min_ms": round(s[0], 2),
            "max_ms": round(s[-1], 2),
            "mean_ms": round(statistics.mean(s), 2),
            "stdev_ms": round(statistics.stdev(s), 2) if len(s) > 1 else 0,
        }


async def single_score(client, i, result):
    """Send a single /unified-score request."""
    tx = make_tx(i)
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{URL}/unified-score", json=tx, timeout=TIMEOUT_S)
        lat = (time.perf_counter() - t0) * 1000
        result.latencies.append(lat)
        result.status_codes[r.status_code] = result.status_codes.get(r.status_code, 0) + 1
        if r.status_code == 200:
            result.success += 1
        else:
            result.errors += 1
    except httpx.TimeoutException:
        result.latencies.append((time.perf_counter() - t0) * 1000)
        result.timeouts += 1
        result.errors += 1
    except Exception as e:
        result.latencies.append((time.perf_counter() - t0) * 1000)
        result.errors += 1


async def batch_score(client, i, result):
    """Send a batch /unified-score-batch request (10 txns each)."""
    batch = [make_tx(i * 10 + j) for j in range(10)]
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{URL}/unified-score-batch", json=batch, timeout=TIMEOUT_S)
        lat = (time.perf_counter() - t0) * 1000
        result.latencies.append(lat)
        result.status_codes[r.status_code] = result.status_codes.get(r.status_code, 0) + 1
        if r.status_code == 200:
            result.success += 1
        else:
            result.errors += 1
    except httpx.TimeoutException:
        result.latencies.append((time.perf_counter() - t0) * 1000)
        result.timeouts += 1
        result.errors += 1
    except Exception as e:
        result.latencies.append((time.perf_counter() - t0) * 1000)
        result.errors += 1


async def run_level(concurrency, endpoint_type="single"):
    """Run one concurrency level."""
    n_requests = REQUESTS_PER_LEVEL
    result = Result(endpoint=endpoint_type, concurrency=concurrency, total_requests=n_requests)

    limits = httpx.Limits(
        max_connections=concurrency + 10,
        max_keepalive_connections=concurrency,
    )
    async with httpx.AsyncClient(limits=limits, http2=False) as client:
        # Warmup
        warmup_n = min(20, concurrency)
        warmup = [single_score(client, -1, Result(endpoint="warmup", concurrency=0, total_requests=1)) for _ in range(warmup_n)]
        await asyncio.gather(*warmup)

        # Actual test
        t0 = time.perf_counter()
        if endpoint_type == "single":
            tasks = [single_score(client, i, result) for i in range(n_requests)]
        else:
            tasks = [batch_score(client, i, result) for i in range(n_requests)]

        # Throttle: launch in waves to avoid overwhelming
        wave_size = min(concurrency, n_requests)
        for wave_start in range(0, n_requests, wave_size):
            wave = tasks[wave_start:wave_start + wave_size]
            await asyncio.gather(*wave)
        wall_time = time.perf_counter() - t0

    result._wall_time = wall_time
    return result


async def main():
    print("=" * 70)
    print("UNIFIED SCORER STRESS TEST")
    print("=" * 70)
    print(f"  Endpoint: {URL}/unified-score")
    print(f"  Requests per level: {REQUESTS_PER_LEVEL}")
    print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"  Timeout: {TIMEOUT_S}s")
    print()

    all_results = []
    breaking_point = None

    for level in CONCURRENCY_LEVELS:
        print(f"--- Concurrency: {level} ---")
        t0 = time.time()

        # Run single scoring
        r = await run_level(level, "single")
        elapsed = time.time() - t0
        pcts = r.percentiles()

        status = "✅" if r.error_rate < 1 else ("⚠️" if r.error_rate < 5 else "❌")
        print(f"  {status} Single:  {r.success}/{r.total_requests} OK | "
              f"Errors: {r.errors} ({r.error_rate}%) | "
              f"Throughput: {r.throughput} txn/s | "
              f"P50: {pcts.get('p50_ms', 0):.0f}ms | "
              f"P99: {pcts.get('p99_ms', 0):.0f}ms | "
              f"Wall: {elapsed:.1f}s")
        if r.errors > 0:
            print(f"       Status codes: {r.status_codes}")
        all_results.append({"level": level, "type": "single", "result": {
            "success": r.success, "errors": r.errors, "timeouts": r.timeouts,
            "error_rate": r.error_rate, "throughput": r.throughput,
            "percentiles": pcts, "status_codes": r.status_codes,
        }})

        if r.error_rate >= 5 and breaking_point is None:
            breaking_point = level

        # Run batch scoring
        r2 = await run_level(level, "batch")
        elapsed2 = time.time() - t0 - elapsed
        pcts2 = r2.percentiles()
        batch_throughput_txn = round(r2.success * 10 / max(elapsed2, 0.01), 1)

        status2 = "✅" if r2.error_rate < 1 else ("⚠️" if r2.error_rate < 5 else "❌")
        print(f"  {status2} Batch:   {r2.success}/{r2.total_requests} OK | "
              f"Errors: {r2.errors} ({r2.error_rate}%) | "
              f"Throughput: {batch_throughput_txn} txn/s ({r2.success} batch/s) | "
              f"P50: {pcts2.get('p50_ms', 0):.0f}ms | "
              f"P99: {pcts2.get('p99_ms', 0):.0f}ms")
        all_results.append({"level": level, "type": "batch", "result": {
            "success": r2.success, "errors": r2.errors, "timeouts": r2.timeouts,
            "error_rate": r2.error_rate, "throughput": r2.throughput,
            "throughput_txn": batch_throughput_txn,
            "percentiles": pcts2, "status_codes": r2.status_codes,
        }})
        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Level':<10} {'Type':<8} {'OK':<8} {'Err%':<8} {'Txn/s':<10} {'P50':<10} {'P99':<10}")
    print(f"  {'-'*64}")
    for r in all_results:
        res = r["result"]
        pcts = res["percentiles"]
        tput = res.get("throughput_txn", res["throughput"])
        print(f"  {r['level']:<10} {r['type']:<8} {res['success']:<8} {res['error_rate']:<8} "
              f"{tput:<10} {pcts.get('p50_ms', 0):<10.0f} {pcts.get('p99_ms', 0):<10.0f}")

    if breaking_point:
        print(f"\n  ⚠️  BREAKING POINT: {breaking_point} concurrent workers (error rate >= 5%)")
    else:
        max_level = max(r["level"] for r in all_results)
        print(f"\n  ✅ NO BREAKING POINT found up to {max_level} concurrent workers")

    # Find peak throughput
    single_results = [r for r in all_results if r["type"] == "single"]
    peak = max(single_results, key=lambda r: r["result"]["throughput"])
    print(f"  📈 Peak single-scorer throughput: {peak['result']['throughput']} txn/s at {peak['level']} workers")

    batch_results = [r for r in all_results if r["type"] == "batch"]
    peak_batch = max(batch_results, key=lambda r: r["result"].get("throughput_txn", 0))
    print(f"  📈 Peak batch throughput: {peak_batch['result'].get('throughput_txn', 0)} txn/s at {peak_batch['level']} workers")

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "url": URL,
            "requests_per_level": REQUESTS_PER_LEVEL,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "timeout_s": TIMEOUT_S,
        },
        "breaking_point": breaking_point,
        "peak_single_throughput": peak["result"]["throughput"],
        "peak_batch_throughput": peak_batch["result"].get("throughput_txn", 0),
        "results": all_results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
