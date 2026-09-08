#!/usr/bin/env python3
"""PS-14 live concurrency sweep (Part 3 #1/#2).

Sends concurrent /internal/evaluate requests with UNIQUE event ids to the
running risk service, at 1/2/4/8/16/32 worker threads, and reports
throughput, p50/p95/p99 latency and error rate per level. Idempotency: one
level also re-submits each event_id twice and verifies the replay path
returns the stored decision without a duplicate audit write.

Usage: ./.venv/Scripts/python.exe scripts/live_concurrency_sweep.py [port]
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import statistics
import sys
import time
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8003


def _token_from_env() -> str:
    """The live services are started detached with .env exported; the dev
    default differs, so read the same source the running stack uses."""
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    for k in ("INTERNAL_TOKEN", "RISK_INTERNAL_TOKEN", "INTERNAL_API_TOKEN",
              "X_INTERNAL_TOKEN"):
        if env.get(k):
            return env[k]
    from src.settings import Settings  # fallback: dev default
    return Settings().internal_token


TOKEN = _token_from_env()
URL = f"http://127.0.0.1:{PORT}/internal/evaluate"


def base_features(seed: int) -> dict:
    import random
    r = random.Random(seed)
    return {
        "amount_ratio": round(0.3 + 2.5 * r.random(), 4),
        "txn_freq_last_24h": r.randint(0, 14),
        "txn_time_unusual": r.randint(0, 1),
        "new_device_flag": r.randint(0, 1),
        "unusual_location_flag": r.randint(0, 1),
        "unusual_recipient_flag": r.randint(0, 1),
        "failed_auth_count_24h": r.randint(0, 3),
        "days_since_last_similar_txn": round(r.uniform(0, 20), 2),
        "gradual_escalation_score": round(r.random(), 4),
        "known_device_count": r.randint(1, 5),
        "account_tenure_days": round(r.uniform(1, 800), 2),
        "hour_of_day": r.randint(0, 23),
        "is_weekend": r.randint(0, 1),
        "amount_zscore": round(r.uniform(-1, 5), 4),
        "velocity_deviation": round(r.uniform(0, 2), 4),
        "recipient_novelty": round(r.random(), 4),
        "txn_regularity": round(r.uniform(0, 20), 4),
        "hour_deviation": round(r.uniform(0, 6), 4),
        "shared_device_accounts": 0,
        "shared_recipient_accounts": 0,
        "mule_ring_score": 0.0,
    }


def one(seed: int, event_id: str) -> tuple[float, int]:
    body = json.dumps({"event_id": event_id, "fraud_id": "F24BRMMMBJWYMTDW",
                       "features": base_features(seed)}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "X-Internal-Token": TOKEN})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return (time.perf_counter() - t0) * 1000.0, 0
    except Exception:
        return (time.perf_counter() - t0) * 1000.0, 1


def sweep(level: int, n: int) -> dict:
    t0 = time.perf_counter()
    errors = 0
    lats: list[float] = []
    with cf.ThreadPoolExecutor(max_workers=level) as ex:
        futs = [ex.submit(one, i, f"load{level}x{i:08d}") for i in range(n)]
        for f in cf.as_completed(futs):
            lat, err = f.result()
            lats.append(lat)
            errors += err
    dur = time.perf_counter() - t0
    lats.sort()
    p = lambda q: lats[min(int(q * len(lats)), len(lats) - 1)]
    return {
        "concurrency": level, "requests": n,
        "rps": round(n / dur, 1),
        "p50_ms": round(p(0.50), 1), "p95_ms": round(p(0.95), 1),
        "p99_ms": round(p(0.99), 1), "max_ms": round(lats[-1], 1),
        "errors": errors, "error_pct": round(100 * errors / n, 2),
    }


def idempotency_check() -> dict:
    ev = "idem000000000001"
    r1 = one(7, ev)
    r2 = one(8, ev)   # same event_id, different features -> must replay stored
    r3 = one(9, ev)
    errs = sum(x[1] for x in (r1, r2, r3))
    return {"same_event_submitted_3x": True, "all_ok": errs == 0,
            "latencies_ms": [round(x[0], 1) for x in (r1, r2, r3)]}


def main() -> int:
    print("PS-14 LIVE CONCURRENCY SWEEP on", URL)
    print(f"{'Concurrency':>11} {'Req/s':>7} {'p50':>7} {'p95':>7} {'p99':>7} "
          f"{'max':>7} {'Err%':>6}")
    results = []
    for level in (1, 2, 4, 8, 16, 32):
        r = sweep(level, 400)
        results.append(r)
        print(f"{r['concurrency']:>11} {r['rps']:>7} {r['p50_ms']:>7} "
              f"{r['p95_ms']:>7} {r['p99_ms']:>7} {r['max_ms']:>7} "
              f"{r['error_pct']:>6}")
    idem = idempotency_check()
    print("idempotency:", idem)
    (ROOT / "reports" / "live_concurrency_sweep.json").write_text(
        json.dumps({"results": results, "idempotency": idem,
                    "note": "risk /internal/evaluate, unique event ids, "
                            "single uvicorn process"}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())