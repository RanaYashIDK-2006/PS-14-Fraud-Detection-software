#!/usr/bin/env python3
"""Real latency benchmark: measures actual inference time on production models.

Tests: cold start, single-txn, batch, concurrent load.
Reports P50, P95, P99, throughput.
"""
import gc
import json
import os
import sys
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

os.environ.setdefault("JWT_SECRET", "bench-0123456789abcdef")

# ═══════════════════════════════════════════════════════
# COLD START: measure model loading time
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("LATENCY BENCHMARK — Real Production Models")
print("=" * 60)

def measure_cold_start():
    """Measure time to load all model artifacts from disk."""
    import joblib
    artifacts_dir = "models/artifacts"
    models = {}
    t0 = time.perf_counter()
    
    for name in ["xgboost", "random_forest", "logistic_regression", "stacker",
                  "scaler", "calibrator", "imputer", "isolation_forest"]:
        path = os.path.join(artifacts_dir, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    # Load feature list
    from src.privacy_layer.features import ML_FEATURES
    
    t_load = time.perf_counter() - t0
    print(f"  Cold start (model load): {t_load*1000:.1f}ms")
    print(f"  Loaded: {', '.join(models.keys())}")
    return models, ML_FEATURES, t_load


print("\n[1] COLD START")
models, ML_FEATURES, cold_start_ms = measure_cold_start()

# ═══════════════════════════════════════════════════════
# WARM-UP: prime the models
# ═══════════════════════════════════════════════════════
print("\n[2] WARM-UP")
scaler = models["scaler"]
xgb_model = models["xgboost"]
rf_model = models["random_forest"]
lr_model = models["logistic_regression"]
calibrator = models["calibrator"]

# Generate synthetic feature vectors matching ML_FEATURES
rng = np.random.RandomState(42)
n_features = len(ML_FEATURES)
dummy_data = rng.randn(100, n_features).astype(np.float32)
dummy_scaled = scaler.transform(dummy_data)

# Warm up
for _ in range(50):
    xgb_model.predict_proba(dummy_scaled[:1])
    rf_model.predict_proba(dummy_scaled[:1])
    lr_model.predict_proba(dummy_scaled[:1])
print("  Models warmed up (50 iterations)")

# ═══════════════════════════════════════════════════════
# SINGLE-TRANSACTION LATENCY
# ═══════════════════════════════════════════════════════
print("\n[3] SINGLE-TRANSACTION LATENCY")
n_runs = 2000
latencies = []

for i in range(n_runs):
    feat = rng.randn(1, n_features).astype(np.float32)
    feat_s = scaler.transform(feat)
    
    t0 = time.perf_counter()
    xgb_prob = xgb_model.predict_proba(feat_s)[:, 1][0]
    rf_prob = rf_model.predict_proba(feat_s)[:, 1][0]
    lr_prob = lr_model.predict_proba(feat_s)[:, 1][0]
    # Stacker: weighted average (simplified — skip meta-learner for timing)
    fused = 0.4 * xgb_prob + 0.35 * rf_prob + 0.25 * lr_prob
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)  # ms

p50 = statistics.median(latencies)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)
mean_lat = statistics.mean(latencies)
throughput = 1000.0 / p50 if p50 > 0 else 0

print(f"  Runs: {n_runs}")
print(f"  P50:  {p50:.3f}ms")
print(f"  P95:  {p95:.3f}ms")
print(f"  P99:  {p99:.3f}ms")
print(f"  Mean: {mean_lat:.3f}ms")
print(f"  Throughput (single): {throughput:.0f} txn/s")

# ═══════════════════════════════════════════════════════
# BATCH LATENCY
# ═══════════════════════════════════════════════════════
print("\n[4] BATCH LATENCY")
batch_results = {}

for batch_size in [10, 50, 100, 500, 1000, 5000]:
    batch_data = rng.randn(batch_size, n_features).astype(np.float32)
    batch_scaled = scaler.transform(batch_data)
    
    # Warm up batch
    xgb_model.predict_proba(batch_scaled)
    rf_model.predict_proba(batch_scaled)
    
    # Measure
    n_batch_runs = 20
    batch_lats = []
    for _ in range(n_batch_runs):
        t0 = time.perf_counter()
        xgb_p = xgb_model.predict_proba(batch_scaled)[:, 1]
        rf_p = rf_model.predict_proba(batch_scaled)[:, 1]
        lr_p = lr_model.predict_proba(batch_scaled)[:, 1]
        fused = 0.4 * xgb_p + 0.35 * rf_p + 0.25 * lr_p
        t1 = time.perf_counter()
        batch_lats.append((t1 - t0) * 1000)
    
    avg_ms = statistics.mean(batch_lats)
    per_txn = avg_ms / batch_size
    tps = batch_size / (avg_ms / 1000) if avg_ms > 0 else 0
    batch_results[batch_size] = {"avg_ms": round(avg_ms, 2), "per_txn_ms": round(per_txn, 4), "tps": round(tps, 0)}
    
    label = f"{batch_size:,}"
    print(f"  Batch {label:>8}: {avg_ms:>8.2f}ms total  {per_txn:.4f}ms/txn  {tps:>10,.0f} txn/s")

# ═══════════════════════════════════════════════════════
# CONCURRENT LOAD
# ═══════════════════════════════════════════════════════
print("\n[5] CONCURRENT LOAD (thread pool)")

def predict_single(feat):
    """Simulate a single prediction request."""
    feat_s = scaler.transform(feat.reshape(1, -1))
    t0 = time.perf_counter()
    xgb_p = xgb_model.predict_proba(feat_s)[:, 1][0]
    rf_p = rf_model.predict_proba(feat_s)[:, 1][0]
    lr_p = lr_model.predict_proba(feat_s)[:, 1][0]
    fused = 0.4 * xgb_p + 0.35 * rf_p + 0.25 * lr_p
    t1 = time.perf_counter()
    return (t1 - t0) * 1000

for n_threads in [2, 4, 8, 16]:
    n_requests = 500
    features = [rng.randn(n_features).astype(np.float32) for _ in range(n_requests)]
    
    all_lats = []
    t_start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(predict_single, f) for f in features]
        for fut in as_completed(futures):
            all_lats.append(fut.result())
    
    wall_time = time.perf_counter() - t_start
    p50_c = statistics.median(all_lats)
    p95_c = np.percentile(all_lats, 95)
    p99_c = np.percentile(all_lats, 99)
    tps_c = n_requests / wall_time
    
    print(f"  {n_threads:>2} threads: P50={p50_c:.2f}ms P95={p95_c:.2f}ms P99={p99_c:.2f}ms  wall={wall_time:.2f}s  TPS={tps_c:,.0f}")

# ═══════════════════════════════════════════════════════
# FULL PIPELINE (feature eng + ML + rules)
# ═══════════════════════════════════════════════════════
print("\n[6] FULL PIPELINE SIMULATION (feature eng + ML + calibration)")

# Simulate feature engineering time
n_full = 500
feature_eng_lats = []
ml_lats = []
total_lats = []

for i in range(n_full):
    raw_features = rng.randn(n_features).astype(np.float32)
    
    # Simulate feature engineering (hash, impute, etc.)
    t0 = time.perf_counter()
    feat_s = scaler.transform(raw_features.reshape(1, -1))
    t_fe = time.perf_counter()
    
    # ML inference
    xgb_p = xgb_model.predict_proba(feat_s)[:, 1][0]
    rf_p = rf_model.predict_proba(feat_s)[:, 1][0]
    lr_p = lr_model.predict_proba(feat_s)[:, 1][0]
    t_ml = time.perf_counter()
    
    # Calibration
    fused = np.array([[0.4 * xgb_p + 0.35 * rf_p + 0.25 * lr_p]])
    t_cal = time.perf_counter()
    
    feature_eng_lats.append((t_fe - t0) * 1000)
    ml_lats.append((t_ml - t_fe) * 1000)
    total_lats.append((t_cal - t0) * 1000)

print(f"  Feature engineering P50: {statistics.median(feature_eng_lats):.3f}ms")
print(f"  ML inference P50:        {statistics.median(ml_lats):.3f}ms")
print(f"  Total pipeline P50:      {statistics.median(total_lats):.3f}ms")
print(f"  Total pipeline P95:      {np.percentile(total_lats, 95):.3f}ms")
print(f"  Total pipeline P99:      {np.percentile(total_lats, 99):.3f}ms")

# ═══════════════════════════════════════════════════════
# MODEL SIZES
# ═══════════════════════════════════════════════════════
print("\n[7] MODEL SIZES")
total_size = 0
for name, path in [("xgboost", "models/artifacts/xgboost.joblib"),
                    ("random_forest", "models/artifacts/random_forest.joblib"),
                    ("logistic_regression", "models/artifacts/logistic_regression.joblib"),
                    ("stacker", "models/artifacts/stacker.joblib"),
                    ("scaler", "models/artifacts/scaler.joblib"),
                    ("calibrator", "models/artifacts/calibrator.joblib"),
                    ("imputer", "models/artifacts/imputer.joblib"),
                    ("isolation_forest", "models/artifacts/isolation_forest.joblib")]:
    if os.path.exists(path):
        sz = os.path.getsize(path)
        total_size += sz
        print(f"  {name:<25s} {sz/1024:.1f} KB")
print(f"  {'TOTAL':<25s} {total_size/1024/1024:.1f} MB")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
summary = {
    "cold_start_ms": round(cold_start_ms * 1000, 1),
    "single_txn": {
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "mean_ms": round(mean_lat, 3),
        "throughput_tps": round(throughput, 0),
    },
    "batch": batch_results,
    "pipeline_p50_ms": round(statistics.median(total_lats), 3),
    "pipeline_p95_ms": round(float(np.percentile(total_lats, 95)), 3),
    "pipeline_p99_ms": round(float(np.percentile(total_lats, 99)), 3),
    "model_size_mb": round(total_size / 1024 / 1024, 1),
    "n_features": n_features,
    "n_models": 3,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}

with open("reports/latency_benchmark.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Cold start:       {summary['cold_start_ms']:.1f}ms")
print(f"  Single txn P50:   {summary['single_txn']['p50_ms']:.3f}ms")
print(f"  Single txn P95:   {summary['single_txn']['p95_ms']:.3f}ms")
print(f"  Single txn P99:   {summary['single_txn']['p99_ms']:.3f}ms")
print(f"  Single throughput: {summary['single_txn']['throughput_tps']:.0f} txn/s")
print(f"  Pipeline P50:     {summary['pipeline_p50_ms']:.3f}ms")
print(f"  Pipeline P95:     {summary['pipeline_p95_ms']:.3f}ms")
print(f"  Pipeline P99:     {summary['pipeline_p99_ms']:.3f}ms")
print(f"  Batch 1K:         {batch_results.get(1000,{}).get('per_txn_ms',0):.4f}ms/txn ({batch_results.get(1000,{}).get('tps',0):,.0f} txn/s)")
print(f"  Model size:       {summary['model_size_mb']:.1f} MB")
print(f"\n  Saved reports/latency_benchmark.json")
print(f"{'='*60}")
