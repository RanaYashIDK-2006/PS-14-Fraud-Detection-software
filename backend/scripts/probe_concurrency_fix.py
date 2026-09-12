#!/usr/bin/env python3
"""Concurrency probe for the check-#26 SQLite pool fix.

Boots the isolated perf stack (identity:8001 privacy:8002 risk:8003, fresh
db/perf_audit stores) and hammers POST /internal/evaluate with 8 concurrent
workers. Before the fix (StaticPool - one shared SQLite connection across
request threads) this produced sqlite3.InterfaceError 500s on ~24-34% of
requests; the fix must bring the error rate to ~0.

Usage:
  python scripts/launch_perf_stack.py   (boot first, wait for STACK READY)
  python scripts/probe_concurrency_fix.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
LOG_DIR = ROOT / "db" / "perf_audit"

RISK = "http://127.0.0.1:8003"


def _token() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("INTERNAL_TOKEN="):
            return line.split("=", 1)[1]
    raise SystemExit("INTERNAL_TOKEN missing from .env")


TOKEN = _token()

WORKERS = 8
PER_WORKER = 150


def base_vector(i: int) -> dict:
    # Benign, in-distribution vectors (ratio ~1, normal hours, no attack flags).
    return {
        "amount_ratio": round(0.6 + (i % 5) * 0.2, 4),
        "txn_freq_last_24h": 1 + (i % 3),
        "txn_time_unusual": 0,
        "new_device_flag": 0,
        "unusual_location_flag": 0,
        "unusual_recipient_flag": 0,
        "failed_auth_count_24h": 0,
        "days_since_last_similar_txn": round(1.0 + (i % 10) * 1.5, 2),
        "gradual_escalation_score": 0.0,
        "known_device_count": 2 + (i % 5),
        "account_tenure_days": 30.0 + (i % 300),
        "hour_of_day": 9 + (i % 10),
        "is_weekend": 1 if i % 7 < 2 else 0,
        "shared_device_accounts": 0,
        "shared_recipient_accounts": 0,
        "mule_ring_score": 0.0,
        "hour_deviation": round(0.05 + (i % 7) * 0.03, 4),
        "amount_zscore": round(-0.5 + (i % 9) * 0.2, 4),
        "velocity_deviation": round(0.1 + (i % 4) * 0.1, 4),
        "recipient_novelty": 0.0,
        "txn_regularity": round(0.4 + (i % 5) * 0.1, 4),
    }


def post(payload: dict) -> tuple[int, float]:
    import httpx
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{RISK}/internal/evaluate", json=payload,
                       headers={"X-Internal-Token": TOKEN}, timeout=30)
        return r.status_code, (time.perf_counter() - t0) * 1000
    except Exception as e:  # noqa: BLE001
        return -1, (time.perf_counter() - t0) * 1000


def main() -> int:
    # wait for health
    ok = False
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"{RISK}/health", timeout=2) as r:
                if r.status == 200:
                    ok = True
                    break
        except Exception:
            time.sleep(2)
    if not ok:
        print("risk service not healthy - boot with scripts/launch_perf_stack.py")
        return 2

    payloads = []
    for w in range(WORKERS):
        for n in range(PER_WORKER):
            i = w * 10000 + n
            payloads.append({
                "event_id": f"fixprobe-{i:07d}",
                "fraud_id": "F24BRMMMBJWYMTDW",
                "features": base_vector(i),
            })

    t0 = time.perf_counter()
    statuses = []
    latencies = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for st, ms in ex.map(post, payloads):
            statuses.append(st)
            if ms >= 0:
                latencies.append(ms)
    elapsed = time.perf_counter() - t0

    total = len(statuses)
    errs = [s for s in statuses if s >= 400]
    s5 = [s for s in statuses if s >= 500]
    err_rate = 100.0 * len(errs) / total
    lat = sorted(latencies)
    p50 = lat[int(len(lat) * 0.5)]
    p95 = lat[int(len(lat) * 0.95)]
    p99 = lat[int(len(lat) * 0.99)]
    print(f"requests={total} workers={WORKERS} elapsed={elapsed:.1f}s "
          f"throughput={total / elapsed:.1f}/s")
    print(f"errors={len(errs)} (4xx/5xx) 5xx={len(s5)} error_rate={err_rate:.2f}%")
    print(f"latency p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms max={lat[-1]:.1f}ms")

    # scan the risk log for the pre-fix failure signature
    logf = LOG_DIR / "risk.log"
    txt = logf.read_text(encoding="utf-8", errors="replace") if logf.exists() else ""
    interface_errs = txt.count("InterfaceError") + txt.lower().count("bad parameter or other api misuse")
    print(f"risk.log sqlite InterfaceError signatures: {interface_errs}")

    # drift detector state after the load
    try:
        import httpx
        ds = httpx.get(f"{RISK}/internal/drift-status",
                       headers={"X-Internal-Token": TOKEN}, timeout=10).json()
        print("drift-status:", json.dumps({k: ds.get(k) for k in
              ("state", "max_psi", "total_events", "baseline_loaded", "window_size")}))
    except Exception as e:  # noqa: BLE001
        print("drift-status unavailable:", e)

    ok = (len(s5) == 0) and (err_rate < 1.0)
    print("\nRESULT:", "PASS (concurrency fix effective)" if ok
          else f"FAIL (error rate {err_rate:.1f}%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
