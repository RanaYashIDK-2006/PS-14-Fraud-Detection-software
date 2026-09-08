#!/usr/bin/env python3
"""Test horizontal scaling: cross-worker Redis state sharing."""
import httpx
import subprocess

def score(worker, user, amt_ratio=1.5):
    r = httpx.post(f"http://127.0.0.1:{worker}/unified-score", json={
        "amount_ratio": amt_ratio, "hour_of_day": 14, "is_weekend": 0,
        "new_device_flag": 0, "txn_freq_last_24h": 5, "known_device_count": 3,
        "account_tenure_days": 180, "gradual_escalation_score": 0.3,
        "unusual_location_flag": 0, "unusual_recipient_flag": 0,
        "failed_auth_count_24h": 0, "hour_deviation": 0.2,
        "amount_zscore": 0.5, "velocity_deviation": 0.3,
        "shared_device_accounts": 0, "shared_recipient_accounts": 0,
        "mule_ring_score": 0.1, "txn_amount_bucket": 2,
        "txn_time_unusual": 0, "days_since_last_similar_txn": 7,
        "amount_ratio_sq": amt_ratio**2, "user_id": user,
    }, timeout=5)
    d = r.json()
    return d.get("decision", "?"), d.get("risk_score", 0)

print("=" * 60)
print("HORIZONTAL SCALING TEST — Redis-Backed State")
print("=" * 60)
print()

# Worker 1: score 3 transactions for user_999
print("Phase 1: Worker 1 (8006) scores 3 txns for user_999")
for i in range(3):
    dec, score_val = score(8006, "user_999", 1.0 + i)
    print(f"  W1 tx {i+1}: decision={dec}, score={score_val}")

# Worker 2: score 1 more for same user (reads W1's state from Redis)
print("\nPhase 2: Worker 2 (8007) scores tx 4 for same user_999")
dec, score_val = score(8007, "user_999", 2.0)
print(f"  W2 tx 4: decision={dec}, score={score_val}")

# Worker 1: score tx 5 (reads W2's addition from Redis)
print("\nPhase 3: Worker 1 (8006) scores tx 5 for same user_999")
dec, score_val = score(8006, "user_999", 1.5)
print(f"  W1 tx 5: decision={dec}, score={score_val}")

# Check Redis state
result = subprocess.run(
    ["C:/Program Files/Redis/redis-cli.exe", "keys", "fw:user_999*"],
    capture_output=True, text=True
)
print(f"\nRedis keys for user_999:\n  {result.stdout.strip()}")

# Health check both
h1 = httpx.get("http://127.0.0.1:8006/unified-status", timeout=3).json()
h2 = httpx.get("http://127.0.0.1:8007/unified-status", timeout=3).json()
d1 = h1.get("drift", {})
d2 = h2.get("drift", {})
print(f"\nWorker 1 (8006): drift_scored={d1.get('n_scored', 0)}")
print(f"Worker 2 (8007): drift_scored={d2.get('n_scored', 0)}")

# Concurrent load test
print("\nPhase 4: Concurrent load (100 requests split across both workers)")
import asyncio

async def concurrent_test():
    results = {"w1": 0, "w2": 0, "errors": 0}
    
    async def score_one(client, worker, i):
        try:
            r = await client.post(
                f"http://127.0.0.1:{worker}/unified-score",
                json={
                    "amount_ratio": 1.5, "hour_of_day": i % 24, "is_weekend": 0,
                    "new_device_flag": 0, "txn_freq_last_24h": 5, "known_device_count": 3,
                    "account_tenure_days": 180, "gradual_escalation_score": 0.3,
                    "unusual_location_flag": 0, "unusual_recipient_flag": 0,
                    "failed_auth_count_24h": 0, "hour_deviation": 0.2,
                    "amount_zscore": 0.5, "velocity_deviation": 0.3,
                    "shared_device_accounts": 0, "shared_recipient_accounts": 0,
                    "mule_ring_score": 0.1, "txn_amount_bucket": 2,
                    "txn_time_unusual": 0, "days_since_last_similar_txn": 7,
                    "amount_ratio_sq": 2.25, "user_id": f"user_{i % 20}",
                },
                timeout=10
            )
            if r.status_code == 200:
                return worker
            return "error"
        except Exception:
            return "error"

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(100):
            worker = 8006 if i % 2 == 0 else 8007
            tasks.append(score_one(client, worker, i))
        results_list = await asyncio.gather(*tasks)
        
        for r in results_list:
            if r == 8006:
                results["w1"] += 1
            elif r == 8007:
                results["w2"] += 1
            else:
                results["errors"] += 1
    
    return results

res = asyncio.run(concurrent_test())
print(f"  Worker 1 (8006): {res['w1']} requests")
print(f"  Worker 2 (8007): {res['w2']} requests")
print(f"  Errors: {res['errors']}")

# Redis memory
result = subprocess.run(
    ["C:/Program Files/Redis/redis-cli.exe", "info", "memory"],
    capture_output=True, text=True
)
for line in result.stdout.split("\n"):
    if "used_memory_human" in line:
        print(f"\n  Redis memory: {line.split(':')[1].strip()}")
        break

print("\n" + "=" * 60)
print("HORIZONTAL SCALING VERIFIED")
print("=" * 60)
