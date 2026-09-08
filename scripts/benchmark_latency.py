#!/usr/bin/env python3
"""Benchmark per-transaction inference latency for PS-14 XGBoost model.

Measures:
  - Cold-start latency (first prediction after model load)
  - Warm single-transaction latency
  - Batched latency (various batch sizes)
  - Throughput (transactions/second)

Output: benchmarks/latency_results.json
"""

import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42


def benchmark_latency():
    print("=" * 70)
    print("PS-14 INFERENCE LATENCY BENCHMARK")
    print("=" * 70)

    # ── Load model artifacts ──────────────────────────────────────────────
    print("\nLoading model artifacts...")
    t0 = time.time()
    import joblib
    from src.privacy_layer.features import ML_FEATURES

    model_path = ROOT / "models" / "artifacts"
    xgb = joblib.load(model_path / "xgboost.joblib")
    scaler = joblib.load(model_path / "scaler.joblib")
    print(f"  Loaded in {time.time()-t0:.3f}s")

    # ── Generate synthetic feature vectors ────────────────────────────────
    rng = np.random.default_rng(SEED)
    n_features = len(ML_FEATURES)

    # Create realistic feature vectors (simulating different fraud scenarios)
    scenarios = {
        "normal": {
            "amount_ratio": 1.0, "txn_freq_last_24h": 5, "txn_time_unusual": 0.5,
            "new_device_flag": 0, "unusual_location_flag": 0,
            "unusual_recipient_flag": 0, "failed_auth_count_24h": 0,
            "days_since_last_similar_txn": 1.0, "gradual_escalation_score": 1.0,
            "known_device_count": 10, "account_tenure_days": 365,
            "hour_of_day": 14, "is_weekend": 0, "shared_device_accounts": 1,
            "shared_recipient_accounts": 1, "mule_ring_score": 1,
            "hour_deviation": 2, "amount_zscore": 0.5,
            "velocity_deviation": 0.2, "recipient_novelty": 0.5,
            "txn_regularity": 1.0,
        },
        "suspicious": {
            "amount_ratio": 5.0, "txn_freq_last_24h": 20, "txn_time_unusual": 0.9,
            "new_device_flag": 1, "unusual_location_flag": 1,
            "unusual_recipient_flag": 1, "failed_auth_count_24h": 3,
            "days_since_last_similar_txn": 0.01, "gradual_escalation_score": 3.0,
            "known_device_count": 2, "account_tenure_days": 30,
            "hour_of_day": 2, "is_weekend": 1, "shared_device_accounts": 5,
            "shared_recipient_accounts": 5, "mule_ring_score": 5,
            "hour_deviation": 10, "amount_zscore": 3.0,
            "velocity_deviation": 2.0, "recipient_novelty": 0.1,
            "txn_regularity": 20.0,
        },
    }

    # Create test batches
    n_single = 1
    batch_sizes = [1, 10, 50, 100, 500, 1000]

    # Generate random features (scaled to realistic ranges)
    feature_vectors = {}
    for name, base in scenarios.items():
        vec = np.zeros(n_features, dtype=np.float32)
        for i, feat in enumerate(ML_FEATURES):
            if feat in base:
                vec[i] = base[feat]
            else:
                vec[i] = rng.uniform(0, 1)
        feature_vectors[name] = vec

    # ── Benchmark 1: Cold start ───────────────────────────────────────────
    print("\n--- Cold Start Latency ---")
    # Simulate cold start by doing the first prediction
    t_cold_start = time.time()
    X = feature_vectors["normal"].reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = xgb.predict_proba(X_scaled)[0, 1]
    cold_start_ms = (time.time() - t_cold_start) * 1000
    print(f"  Cold start (model already loaded): {cold_start_ms:.3f}ms")
    print(f"  Prediction: {prob:.6f}")

    # ── Benchmark 2: Warm single-transaction latency ──────────────────────
    print("\n--- Warm Single-Transaction Latency ---")
    n_warmup = 100
    n_measure = 10000

    # Warmup
    for _ in range(n_warmup):
        X_scaled = scaler.transform(feature_vectors["normal"].reshape(1, -1))
        xgb.predict_proba(X_scaled)

    # Measure
    times_normal = []
    times_suspicious = []
    for _ in range(n_measure):
        X = feature_vectors["normal"].reshape(1, -1)
        X_scaled = scaler.transform(X)
        t1 = time.time()
        xgb.predict_proba(X_scaled)
        times_normal.append((time.time() - t1) * 1000)

    for _ in range(n_measure):
        X = feature_vectors["suspicious"].reshape(1, -1)
        X_scaled = scaler.transform(X)
        t1 = time.time()
        xgb.predict_proba(X_scaled)
        times_suspicious.append((time.time() - t1) * 1000)

    times_normal = np.array(times_normal)
    times_suspicious = np.array(times_suspicious)

    print(f"  Normal scenario ({n_measure} runs):")
    print(f"    Mean:   {np.mean(times_normal):.3f}ms")
    print(f"    Median: {np.median(times_normal):.3f}ms")
    print(f"    P95:    {np.percentile(times_normal, 95):.3f}ms")
    print(f"    P99:    {np.percentile(times_normal, 99):.3f}ms")
    print(f"    Min:    {np.min(times_normal):.3f}ms")
    print(f"    Max:    {np.max(times_normal):.3f}ms")

    print(f"\n  Suspicious scenario ({n_measure} runs):")
    print(f"    Mean:   {np.mean(times_suspicious):.3f}ms")
    print(f"    Median: {np.median(times_suspicious):.3f}ms")
    print(f"    P95:    {np.percentile(times_suspicious, 95):.3f}ms")
    print(f"    P99:    {np.percentile(times_suspicious, 99):.3f}ms")

    # ── Benchmark 3: Batched inference ────────────────────────────────────
    print("\n--- Batched Inference Latency ---")
    batch_results = []
    for bs in batch_sizes:
        X_batch = np.tile(feature_vectors["normal"], (bs, 1))
        X_scaled_batch = scaler.transform(X_batch)

        # Warmup
        xgb.predict_proba(X_scaled_batch)

        # Measure
        n_runs = 20
        times_batch = []
        for _ in range(n_runs):
            t1 = time.time()
            xgb.predict_proba(X_scaled_batch)
            times_batch.append((time.time() - t1) * 1000)

        mean_ms = np.mean(times_batch)
        per_txn = mean_ms / bs
        throughput = 1000 / per_txn if per_txn > 0 else 0

        print(f"  Batch {bs:>5}: {mean_ms:>8.2f}ms total, {per_txn:>6.3f}ms/txn, {throughput:>8.0f} txn/s")
        batch_results.append({
            "batch_size": bs,
            "total_ms": round(mean_ms, 3),
            "per_txn_ms": round(per_txn, 4),
            "throughput_tps": round(throughput, 1),
        })

    # ── Benchmark 4: Full scoring pipeline (with feature engineering) ─────
    print("\n--- Full Pipeline Latency (feature engineering + inference) ---")
    # Simulate a minimal feature vector construction + inference
    n_pipeline = 1000
    times_pipeline = []
    for _ in range(n_pipeline):
        t1 = time.time()
        # Feature engineering (vectorized, minimal)
        vec = np.zeros(n_features, dtype=np.float32)
        vec[0] = rng.uniform(0.5, 2.0)   # amount_ratio
        vec[1] = rng.integers(1, 20)     # txn_freq
        vec[2] = rng.uniform(0, 1)       # time_unusual
        vec[6] = rng.integers(0, 3)      # failed_auth
        vec[7] = rng.uniform(0, 30)      # days_since
        vec[10] = rng.uniform(10, 1000)  # tenure
        vec[11] = rng.integers(0, 24)    # hour
        # Scale + predict
        X_scaled = scaler.transform(vec.reshape(1, -1))
        prob = xgb.predict_proba(X_scaled)[0, 1]
        times_pipeline.append((time.time() - t1) * 1000)

    times_pipeline = np.array(times_pipeline)
    print(f"  Full pipeline ({n_pipeline} runs):")
    print(f"    Mean:   {np.mean(times_pipeline):.3f}ms")
    print(f"    Median: {np.median(times_pipeline):.3f}ms")
    print(f"    P95:    {np.percentile(times_pipeline, 95):.3f}ms")
    print(f"    P99:    {np.percentile(times_pipeline, 99):.3f}ms")
    print(f"    Throughput: {1000/np.mean(times_pipeline):.0f} txn/s")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "model": "XGBoost (600 trees, depth 10)",
        "n_features": n_features,
        "cold_start_ms": round(cold_start_ms, 3),
        "warm_single_normal": {
            "mean_ms": round(float(np.mean(times_normal)), 3),
            "median_ms": round(float(np.median(times_normal)), 3),
            "p95_ms": round(float(np.percentile(times_normal, 95)), 3),
            "p99_ms": round(float(np.percentile(times_normal, 99)), 3),
        },
        "warm_single_suspicious": {
            "mean_ms": round(float(np.mean(times_suspicious)), 3),
            "median_ms": round(float(np.median(times_suspicious)), 3),
            "p95_ms": round(float(np.percentile(times_suspicious, 95)), 3),
            "p99_ms": round(float(np.percentile(times_suspicious, 99)), 3),
        },
        "batched": batch_results,
        "full_pipeline": {
            "mean_ms": round(float(np.mean(times_pipeline)), 3),
            "median_ms": round(float(np.median(times_pipeline)), 3),
            "p95_ms": round(float(np.percentile(times_pipeline, 95)), 3),
            "throughput_tps": round(1000 / np.mean(times_pipeline), 0),
        },
    }

    # Print final summary
    print("\n" + "=" * 70)
    print("LATENCY SUMMARY")
    print("=" * 70)
    print(f"  Model:                  {summary['model']}")
    print(f"  Features:               {summary['n_features']}")
    print(f"  Cold start:             {summary['cold_start_ms']:.3f}ms")
    print(f"  Warm (single, normal):  {summary['warm_single_normal']['median_ms']:.3f}ms median")
    print(f"  Warm (single, P99):     {summary['warm_single_normal']['p99_ms']:.3f}ms")
    print(f"  Full pipeline:          {summary['full_pipeline']['median_ms']:.3f}ms median")
    print(f"  Throughput:             {summary['full_pipeline']['throughput_tps']:.0f} txn/s")
    print(f"  Batch 100:              {batch_results[4]['throughput_tps']:.0f} txn/s")

    # Save
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    (out / "latency_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {out / 'latency_results.json'}")

    return summary


if __name__ == "__main__":
    benchmark_latency()
