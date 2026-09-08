#!/usr/bin/env python3
"""Load test: measure risk engine latency under concurrency.

Sends 500 requests with concurrency 50 to verify P99 < 100ms SLO.
"""
import sys, os, time, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['PS14_MODE'] = 'development'

import requests

URL = 'http://127.0.0.1:8003/internal/evaluate'
# Must match the INTERNAL_TOKEN env var the risk engine was started with
os.environ['INTERNAL_TOKEN'] = 'loadtest-prod-token-2026'
from src.settings import settings
TOKEN = settings.internal_token
print(f'Using token: {TOKEN[:20]}...')
N_REQUESTS = 500
CONCURRENCY = 50

FEATURES = {
    'amount_ratio': 1.5, 'txn_freq_last_24h': 3, 'txn_time_unusual': 0,
    'new_device_flag': 0, 'unusual_location_flag': 0, 'unusual_recipient_flag': 0,
    'failed_auth_count_24h': 0, 'days_since_last_similar_txn': 2.0,
    'gradual_escalation_score': 0.0, 'known_device_count': 5,
    'account_tenure_days': 180, 'hour_of_day': 14, 'is_weekend': 0,
    'shared_device_accounts': 0, 'shared_recipient_accounts': 0,
    'mule_ring_score': 0, 'hour_deviation': 0.5, 'amount_zscore': 0.5,
    'velocity_deviation': 0.3, 'recipient_novelty': 0.5, 'txn_regularity': 0.3,
}

latencies = []
errors = 0

def send_request(i):
    global errors
    headers = {'X-Internal-Token': TOKEN}
    json_data = {
        'event_id': f'load-test-{i:06d}',
        'fraud_id': 'F24BRMMMBJWYMTDW',
        'features': FEATURES,
    }
    t0 = time.monotonic()
    try:
        r = requests.post(URL, headers=headers, json=json_data, timeout=10)
        elapsed = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            latencies.append(elapsed)
        elif r.status_code == 409:  # idempotent replay — still counts
            latencies.append(elapsed)
        else:
            errors += 1
    except Exception:
        errors += 1
        latencies.append(9999)

print(f'Load test: {N_REQUESTS} requests, concurrency={CONCURRENCY}')
print(f'Target: P99 < 100ms')
print()

t_start = time.monotonic()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    futures = [pool.submit(send_request, i) for i in range(N_REQUESTS)]
    for f in as_completed(futures):
        f.result()
t_total = time.monotonic() - t_start

latencies.sort()
n = len(latencies)
if n == 0:
    print('ERROR: No successful requests')
    sys.exit(1)

p50 = latencies[int(n * 0.50)]
p95 = latencies[int(n * 0.95)]
p99 = latencies[int(n * 0.99)]
p999 = latencies[int(n * 0.999)]
max_lat = latencies[-1]
mean_lat = sum(latencies) / n
tps = n / t_total

print(f'Results ({n}/{N_REQUESTS} succeeded, {errors} errors):')
print(f'  TPS:       {tps:.0f}')
print(f'  Total:     {t_total:.2f}s')
print(f'  Mean:      {mean_lat:.2f}ms')
print(f'  P50:       {p50:.2f}ms')
print(f'  P95:       {p95:.2f}ms')
print(f'  P99:       {p99:.2f}ms')
print(f'  P99.9:     {p999:.2f}ms')
print(f'  Max:       {max_lat:.2f}ms')
print()

slo_met = p99 < 100
print(f'  SLO (P99 < 100ms): {"MET" if slo_met else "NOT MET"}')
if slo_met:
    print(f'  Headroom: {100 - p99:.1f}ms')
else:
    print(f'  Over by: {p99 - 100:.1f}ms')

# Save results
result = {
    'n_requests': N_REQUESTS, 'n_succeeded': n, 'concurrency': CONCURRENCY,
    'tps': round(tps, 1), 'total_s': round(t_total, 2),
    'mean_ms': round(mean_lat, 2), 'p50_ms': round(p50, 2),
    'p95_ms': round(p95, 2), 'p99_ms': round(p99, 2),
    'p999_ms': round(p999, 2), 'max_ms': round(max_lat, 2),
    'errors': errors, 'slo_met': slo_met,
}
out = Path('reports') / 'load_test_results.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(f'\nResults saved to {out}')
