#!/usr/bin/env python3
"""End-to-end latency benchmark: Privacy Layer ingest → Risk Engine evaluate."""
import os, sys, time, statistics, secrets, json, random

os.environ.setdefault("INTERNAL_TOKEN", "ps14-dev-internal-token-change-me")
os.environ.setdefault("PS14_MODE", "development")
os.environ.setdefault("JWT_SECRET", "test-secret-for-benchmark-only-do-not-use-in-prod")

from fastapi.testclient import TestClient

# --- Privacy Layer ---
from src.privacy_layer.main import app as privacy_app
from src.privacy_layer.db import engine as priv_engine
import src.privacy_layer.models as pm

# --- Risk Engine ---
from src.risk_engine.main import app as risk_app
from src.risk_engine.db import engine as risk_engine

INTERNAL = os.environ["INTERNAL_TOKEN"]


def setup():
    with risk_engine.begin() as conn:
        from src.risk_engine.models import Base as RB
        RB.metadata.create_all(conn)
    with priv_engine.begin() as conn:
        pm.Base.metadata.create_all(conn)


def stats(times):
    s = sorted(times)
    n = len(s)
    return f"p50={s[n//2]:.1f}ms  p95={s[int(n*0.95)]:.1f}ms  mean={statistics.mean(s):.1f}ms"


def bench_privacy_ingest(client, fraud_id, N=50):
    times = []
    for i in range(N):
        payload = {
            "event_id": f"bench-{secrets.token_hex(8)}",
            "fraud_id": fraud_id,
            "amount": 50.0 + (i * 10),
            "ts": "2026-08-25T12:00:00Z",
            "hour_of_day": 12,
            "device_id": "bench-device-001",
            "location_id": "US",
            "recipient_id": "bench-recv-001",
        }
        t0 = time.perf_counter()
        resp = client.post("/internal/ingest-transaction", json=payload,
                           headers={"X-Internal-Token": INTERNAL})
        t1 = time.perf_counter()
        if resp.status_code != 200:
            print(f"  Ingest FAILED on i={i}: {resp.status_code} {resp.text[:200]}")
            continue
        times.append((t1 - t0) * 1000)
    return times


def bench_risk_evaluate(client, N=50):
    times = []
    features = {
        "txn_amount": 150.0, "hour_of_day": 12, "day_of_week": 0,
        "txn_time_unusual": 0, "amount_zscore": 0.5, "merchant_risk": 0.1,
        "new_device_flag": 0, "known_device_count": 1, "device_daily_count": 1,
        "txn_freq_last_24h": 1, "account_daily_spend_ratio": 0.01,
        "location_risk": 0.05, "unusual_location_flag": 0, "recipient_risk": 0.1,
        "shared_recipient_accounts": 0, "shared_device_accounts": 0,
    }
    for i in range(N):
        payload = {"event_id": f"bench-risk-{secrets.token_hex(8)}", "features": features}
        t0 = time.perf_counter()
        resp = client.post("/internal/evaluate", json=payload,
                           headers={"X-Internal-Token": INTERNAL})
        t1 = time.perf_counter()
        if resp.status_code != 200:
            print(f"  Risk FAILED on i={i}: {resp.status_code} {resp.text[:200]}")
            continue
        times.append((t1 - t0) * 1000)
    return times


def bench_batch_risk(client, batch_size=50, N_batches=10):
    features = {
        "txn_amount": 150.0, "hour_of_day": 12, "day_of_week": 0,
        "txn_time_unusual": 0, "amount_zscore": 0.5, "merchant_risk": 0.1,
        "new_device_flag": 0, "known_device_count": 1, "device_daily_count": 1,
        "txn_freq_last_24h": 1, "account_daily_spend_ratio": 0.01,
        "location_risk": 0.05, "unusual_location_flag": 0, "recipient_risk": 0.1,
        "shared_recipient_accounts": 0, "shared_device_accounts": 0,
    }
    times = []
    for b in range(N_batches):
        events = [
            {"event_id": f"bench-batch-{b}-{j}", "features": features}
            for j in range(batch_size)
        ]
        t0 = time.perf_counter()
        resp = client.post("/internal/evaluate-batch", json={"events": events},
                           headers={"X-Internal-Token": INTERNAL})
        t1 = time.perf_counter()
        if resp.status_code != 200:
            print(f"  Batch FAILED on b={b}: {resp.status_code} {resp.text[:200]}")
            continue
        times.append((t1 - t0) * 1000)
    return times


def bench_e2e_pipeline(priv_client, risk_client, fraud_id, N=20):
    features = {
        "txn_amount": 150.0, "hour_of_day": 12, "day_of_week": 0,
        "txn_time_unusual": 0, "amount_zscore": 0.5, "merchant_risk": 0.1,
        "new_device_flag": 0, "known_device_count": 1, "device_daily_count": 1,
        "txn_freq_last_24h": 1, "account_daily_spend_ratio": 0.01,
        "location_risk": 0.05, "unusual_location_flag": 0, "recipient_risk": 0.1,
        "shared_recipient_accounts": 0, "shared_device_accounts": 0,
    }
    times = []
    for i in range(N):
        evt_id = f"bench-e2e-{secrets.token_hex(8)}"
        t0 = time.perf_counter()
        priv_client.post("/internal/ingest-transaction", json={
            "event_id": evt_id, "fraud_id": fraud_id,
            "amount": 50.0 + i, "ts": "2026-08-25T12:00:00Z",
            "hour_of_day": 12, "device_id": "bench-device-001",
            "location_id": "US", "recipient_id": "bench-recv-001",
        }, headers={"X-Internal-Token": INTERNAL})
        t1 = time.perf_counter()
        risk_client.post("/internal/evaluate", json={
            "event_id": evt_id, "features": features,
        }, headers={"X-Internal-Token": INTERNAL})
        t2 = time.perf_counter()
        times.append({"priv": (t1 - t0) * 1000, "risk": (t2 - t1) * 1000, "total": (t2 - t0) * 1000})
    return times


def main():
    setup()
    print("=" * 70)
    print("PS-14 END-TO-END LATENCY BENCHMARK")
    print("=" * 70)

    fraud_id = "FBENCH0000000001"

    with TestClient(privacy_app) as priv, TestClient(risk_app) as risk:
        print("\n--- 1. Privacy Layer Ingest ---")
        priv_times = bench_privacy_ingest(priv, fraud_id, N=50)
        if priv_times:
            print(f"   {stats(priv_times)}")
            print(f"   Throughput: {1000/statistics.median(priv_times):.0f} ingest/s")
        else:
            print("   NO DATA")

        print("\n--- 2. Risk Engine Evaluate (single event) ---")
        risk_times = bench_risk_evaluate(risk, N=50)
        if risk_times:
            print(f"   {stats(risk_times)}")
            print(f"   Throughput: {1000/statistics.median(risk_times):.0f} eval/s")
        else:
            print("   NO DATA")

        print("\n--- 3. Risk Engine Evaluate-Batch ---")
        batch_times = bench_batch_risk(risk, batch_size=50, N_batches=10)
        if batch_times:
            per_event = [t / 50 for t in batch_times]
            print(f"   Batch (50 events): {stats(batch_times)}")
            print(f"   Per-event: {stats(per_event)}")
            print(f"   Throughput: {50*1000/statistics.median(batch_times):.0f} eval/s")
        else:
            print("   NO DATA")

        print("\n--- 4. Full Pipeline (Privacy -> Risk) ---")
        e2e = bench_e2e_pipeline(priv, risk, fraud_id, N=20)
        if e2e:
            print(f"   Privacy: {stats([d['priv'] for d in e2e])}")
            print(f"   Risk:    {stats([d['risk'] for d in e2e])}")
            print(f"   Total:   {stats([d['total'] for d in e2e])}")
            print(f"   Throughput: {1000/statistics.median([d['total'] for d in e2e]):.0f} events/s")

        import concurrent.futures
        print("\n--- 5. Concurrent Single Requests (10 parallel) ---")
        features = {
            "txn_amount": 150.0, "hour_of_day": 12, "day_of_week": 0,
            "txn_time_unusual": 0, "amount_zscore": 0.5, "merchant_risk": 0.1,
            "new_device_flag": 0, "known_device_count": 1, "device_daily_count": 1,
            "txn_freq_last_24h": 1, "account_daily_spend_ratio": 0.01,
            "location_risk": 0.05, "unusual_location_flag": 0, "recipient_risk": 0.1,
            "shared_recipient_accounts": 0, "shared_device_accounts": 0,
        }

        def single_eval(i):
            t0 = time.perf_counter()
            risk.post("/internal/evaluate", json={
                "event_id": f"bench-conc-{i}-{secrets.token_hex(4)}",
                "features": features,
            }, headers={"X-Internal-Token": INTERNAL})
            return (time.perf_counter() - t0) * 1000

        conc_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            conc_times = list(ex.map(single_eval, range(10)))
        conc_total = (time.perf_counter() - conc_start) * 1000
        print(f"   Individual: {stats(conc_times)}")
        print(f"   Wall clock (10 parallel): {conc_total:.1f}ms")
        print(f"   Effective throughput: {10*1000/conc_total:.0f} eval/s")

    print("\n" + "=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
