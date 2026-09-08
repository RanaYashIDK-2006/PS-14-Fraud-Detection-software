#!/usr/bin/env python3
"""Real-time batch inference pipeline with sliding-window velocity features.

Architecture:
  TransactionQueue → VelocityComputer → ModelScorer → ScoredTransaction

Key design:
  1. SlidingWindow per user: keeps last N transactions (default 50)
  2. Incremental velocity: O(1) update per new transaction (no full-history scan)
  3. Batch scoring: accumulate N transactions, score together for efficiency
  4. Thread-safe: state manager uses locks for concurrent access

Latency target: <100ms per transaction (including velocity computation)
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import lightgbm as lgb


# ── Data Structures ──

@dataclass
class Transaction:
    """Input transaction."""
    user_id: str
    amount: float
    hour: float          # 0-23
    minute: float        # 0-59
    day: float           # day of month
    merchant_id: int = 0
    city_id: int = 0
    chip: int = 0
    mcc: float = 0.0
    is_online: int = 0
    is_error: int = 0
    card: float = 0.0
    year: float = 2024.0
    month: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScoredTransaction:
    """Output scored transaction."""
    user_id: str
    fraud_probability: float
    fraud_flag: bool
    velocity_features: dict
    latency_ms: float
    model_version: str


# ── Sliding Window State ──

class SlidingWindow:
    """Per-user sliding window of recent transactions.

    Maintains a fixed-size deque and pre-computed running statistics
    for O(1) velocity feature computation.
    """
    __slots__ = ("max_size", "_times", "_amounts", "_merchants",
                 "_sum_amount", "_sum_amount_sq", "_count")

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._times: deque = deque(maxlen=max_size)
        self._amounts: deque = deque(maxlen=max_size)
        self._merchants: deque = deque(maxlen=max_size)
        self._sum_amount: float = 0.0
        self._sum_amount_sq: float = 0.0
        self._count: int = 0

    def add(self, time_decimal: float, amount: float, merchant_id: int) -> None:
        """Add a transaction. O(1) amortized."""
        if len(self._times) == self.max_size:
            # Evict oldest
            old_amt = self._amounts[0]
            self._sum_amount -= old_amt
            self._sum_amount_sq -= old_amt * old_amt
            self._count -= 1
        self._times.append(time_decimal)
        self._amounts.append(amount)
        self._merchants.append(merchant_id)
        self._sum_amount += amount
        self._sum_amount_sq += amount * amount
        self._count += 1

    def get_velocity_features(self, current_time: float) -> dict:
        """Compute velocity features from the sliding window. O(k) where k = window size."""
        if self._count < 2:
            return self._empty_features()

        times = np.array(self._times)
        amounts = np.array(self._amounts)
        merchants = np.array(self._merchants)

        # Time deltas
        deltas_h = (current_time - times) * 24  # hours

        # tx_count_1h: transactions in last 1 hour
        tx_count_1h = int((deltas_h <= 1.0).sum())

        # tx_count_24h: transactions in last 24 hours
        tx_count_24h = int((deltas_h <= 24.0).sum())

        # amount_accel: (mean last 5) - (mean prev 5)
        n = len(amounts)
        if n >= 10:
            amount_accel = float(amounts[-5:].mean() - amounts[-10:-5].mean())
        elif n >= 5:
            amount_accel = float(amounts[-5:].mean() - amounts[:-5].mean()) if n > 5 else 0.0
        else:
            amount_accel = 0.0

        # merchant_diversity_24h: unique merchants in last 24h
        last24 = deltas_h <= 24.0
        merchant_diversity_24h = int(np.unique(merchants[last24]).shape[0])

        # avg_amount_24h
        avg_amount_24h = float(amounts[last24].mean()) if last24.sum() > 0 else 0.0

        # tx_gap_mean: mean hours between last 5 transactions
        if n >= 2:
            last5_times = times[-min(5, n):]
            gaps = np.diff(last5_times) * 24
            tx_gap_mean = float(gaps.mean())
        else:
            tx_gap_mean = 0.0

        # tx_regularity: coefficient of variation of gaps
        if n >= 3:
            last5_times = times[-min(5, n):]
            gaps = np.diff(last5_times) * 24
            tx_regularity = float(gaps.std() / (gaps.mean() + 1e-6))
        else:
            tx_regularity = 0.0

        # amount_zscore: current amount vs recent mean
        if n >= 5:
            recent_mean = amounts[-5:].mean()
            recent_std = amounts[-5:].std() + 1e-6
            amount_zscore = float((amounts[-1] - recent_mean) / recent_std)
        else:
            amount_zscore = 0.0

        # tx_freq_accel: difference in frequency between recent and older
        if n >= 6:
            f_rec = 1.0 / (gaps[-min(3, len(gaps)):].mean() + 0.01)
            f_prev = 1.0 / (np.diff(times[-6:-3]) * 24).mean() + 0.01
            tx_freq_accel = f_rec - f_prev
        else:
            tx_freq_accel = 0.0

        # merchant_is_new: 1 if this merchant wasn't in previous 10 tx
        if n >= 2:
            recent_merchants = merchants[:-1][-min(10, n - 1):]
            merchant_is_new = 1.0 if merchants[-1] not in recent_merchants else 0.0
        else:
            merchant_is_new = 0.0

        return {
            "tx_count_1h": tx_count_1h,
            "tx_count_24h": tx_count_24h,
            "amount_accel": amount_accel,
            "merchant_diversity_24h": merchant_diversity_24h,
            "avg_amount_24h": avg_amount_24h,
            "tx_gap_mean": tx_gap_mean,
            "tx_regularity": tx_regularity,
            "amount_zscore": amount_zscore,
            "tx_freq_accel": tx_freq_accel,
            "merchant_is_new": merchant_is_new,
        }

    def _empty_features(self) -> dict:
        return {
            "tx_count_1h": 0, "tx_count_24h": 0, "amount_accel": 0,
            "merchant_diversity_24h": 0, "avg_amount_24h": 0,
            "tx_gap_mean": 0, "tx_regularity": 0, "amount_zscore": 0,
            "tx_freq_accel": 0, "merchant_is_new": 0,
        }

    def __len__(self) -> int:
        return self._count


# ── State Manager ──

class StateManager:
    """Thread-safe per-user sliding window state management.

    Maintains a dict of SlidingWindow per user_id.
    Supports eviction of inactive users (LRU-style).
    """
    def __init__(self, max_users: int = 100_000, window_size: int = 50):
        self.max_users = max_users
        self.window_size = window_size
        self._windows: dict[str, SlidingWindow] = {}
        self._access_order: deque = deque()
        self._lock = threading.Lock()

    def get_window(self, user_id: str) -> SlidingWindow:
        """Get or create sliding window for a user. O(1) amortized."""
        with self._lock:
            if user_id not in self._windows:
                # Evict LRU if at capacity
                if len(self._windows) >= self.max_users:
                    oldest = self._access_order.popleft()
                    self._windows.pop(oldest, None)
                self._windows[user_id] = SlidingWindow(self.window_size)
            else:
                # Move to end of access order
                try:
                    self._access_order.remove(user_id)
                except ValueError:
                    pass
            self._access_order.append(user_id)
            return self._windows[user_id]

    def add_transaction(self, txn: Transaction) -> dict:
        """Add transaction and return velocity features. O(1) amortized."""
        window = self.get_window(txn.user_id)
        time_decimal = txn.day + txn.hour / 24.0 + txn.minute / 1440.0
        window.add(time_decimal, txn.amount, txn.merchant_id)
        return window.get_velocity_features(time_decimal)

    def get_stats(self) -> dict:
        """Get state manager statistics."""
        with self._lock:
            return {
                "active_users": len(self._windows),
                "total_windows": len(self._windows),
                "window_size": self.window_size,
                "max_users": self.max_users,
            }


# ── Feature Builder ──

def build_feature_vector(txn: Transaction, velocity: dict) -> np.ndarray:
    """Build the exact 52-feature vector matching the training schema.

    Feature order matches models/production/altman_lgb_*/feature_schema.json:
      0-26: base features (Month, Day, amt, chip, mcc_n, err, hr, mn, merchant_id,
             city_id, state_id, card_n, year_n, is_online, month_n, day_n, hr_bin,
             amt_log, day_decimal, log_amt, amt_x_hr, amt_x_mcc, amt_x_chip,
             amt_sq, high_amt, night_tx, weekend)
      27-29: target encoding (user_te, merchant_te, city_te)
      30-38: interactions (amt_x_ute, amt_x_mte, amt_x_online, hr_x_chip,
              ute_x_mte, chip_x_online, err_x_amt, hr_x_ute, mcc_x_ute)
      39-48: velocity (tx_count_1h, tx_count_24h, amount_accel, merchant_diversity_24h,
              avg_amount_24h, tx_gap_mean, tx_regularity, amount_zscore,
              tx_freq_accel, merchant_is_new)
      49-51: velocity interactions (amt_x_tx5, amt_x_tx24, ute_x_tx24)
    """
    hr = txn.hour
    mn = txn.minute
    amt = txn.amount
    dd = txn.day + hr / 24.0 + mn / 1440.0

    return np.array([
        # 0-26: base features
        txn.month,               # 0: Month
        txn.day,                 # 1: Day
        amt,                     # 2: amt
        txn.chip,                # 3: chip
        txn.mcc / 10000,         # 4: mcc_n
        txn.is_error,            # 5: err
        hr,                      # 6: hr
        mn,                      # 7: mn
        txn.merchant_id,         # 8: merchant_id
        txn.city_id,             # 9: city_id
        0,                       # 10: state_id (external)
        txn.card,                # 11: card_n
        txn.year - 2010,         # 12: year_n
        txn.is_online,           # 13: is_online
        txn.month,               # 14: month_n
        txn.day,                 # 15: day_n
        min(int(hr // 6), 3),    # 16: hr_bin
        np.log1p(min(amt, 1e9)), # 17: amt_log
        dd,                      # 18: day_decimal
        np.log1p(amt),           # 19: log_amt
        amt * hr,                # 20: amt_x_hr
        amt * (txn.mcc / 10000), # 21: amt_x_mcc
        amt * txn.chip,          # 22: amt_x_chip
        amt ** 2,                # 23: amt_sq
        float(amt > 1000),       # 24: high_amt
        float(hr >= 22 or hr <= 6),  # 25: night_tx
        float((txn.day % 7) >= 5),   # 26: weekend
        # 27-29: target encoding (placeholders)
        0.0,                     # 27: user_te
        0.0,                     # 28: merchant_te
        0.0,                     # 29: city_te
        # 30-38: interactions
        0.0,                     # 30: amt_x_ute
        0.0,                     # 31: amt_x_mte
        amt * txn.is_online,     # 32: amt_x_online
        hr * txn.chip,           # 33: hr_x_chip
        0.0,                     # 34: ute_x_mte
        txn.is_online,           # 35: chip_x_online (approx)
        txn.is_error * amt,      # 36: err_x_amt
        0.0,                     # 37: hr_x_ute
        0.0,                     # 38: mcc_x_ute
        # 39-48: velocity features
        velocity["tx_count_1h"],
        velocity["tx_count_24h"],
        velocity["amount_accel"],
        velocity["merchant_diversity_24h"],
        velocity["avg_amount_24h"],
        velocity["tx_gap_mean"],
        velocity["tx_regularity"],
        velocity["amount_zscore"],
        velocity["tx_freq_accel"],
        velocity["merchant_is_new"],
        # 49-51: velocity interactions
        amt * velocity["tx_count_1h"],
        amt * velocity["tx_count_24h"],
        0.0,                     # ute_x_tx24 (no user_te in real-time)
    ], dtype=np.float32)


# ── Model Scorer ──

class ModelScorer:
    """Load and run the trained LGB model for batch scoring."""

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            # Auto-discover latest model
            artifacts_dir = Path("models/production")
            latest_file = artifacts_dir / "latest"
            if latest_file.exists():
                version = latest_file.read_text().strip()
                model_path = artifacts_dir / f"altman_lgb_{version}" / "model.joblib"
            else:
                raise FileNotFoundError("No model found. Run production_pipeline.py first.")

        self.model = joblib.load(model_path)
        self.model_version = model_path.parent.name
        self._n_features = self.model.n_features_

    def score_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Score a batch of transactions. Returns fraud probabilities."""
        return self.model.predict_proba(feature_matrix)[:, 1]

    def score_single(self, feature_vector: np.ndarray) -> float:
        """Score a single transaction."""
        return self.model.predict_proba(feature_vector.reshape(1, -1))[0, 1]


# ── Real-Time Inference Pipeline ──

class RealtimeScorer:
    """Complete real-time inference pipeline.

    Usage:
        scorer = RealtimeScorer()
        result = scorer.score(transaction)
        results = scorer.score_batch([txn1, txn2, txn3])
    """

    def __init__(self, model_path: Optional[Path] = None,
                 max_users: int = 100_000, window_size: int = 50,
                 state_manager=None):
        if state_manager is not None:
            self.state = state_manager
        else:
            self.state = StateManager(max_users=max_users, window_size=window_size)
        self.scorer = ModelScorer(model_path)
        self._feature_names = None

    def score(self, txn: Transaction) -> ScoredTransaction:
        """Score a single transaction. Target: <100ms."""
        t0 = time.perf_counter()

        # 1. Update state and get velocity features
        velocity = self.state.add_transaction(txn)

        # 2. Build feature vector
        features = build_feature_vector(txn, velocity)

        # 3. Score
        proba = self.scorer.score_single(features)

        latency_ms = (time.perf_counter() - t0) * 1000

        return ScoredTransaction(
            user_id=txn.user_id,
            fraud_probability=round(float(proba), 6),
            fraud_flag=proba > 0.5,
            velocity_features=velocity,
            latency_ms=round(latency_ms, 3),
            model_version=self.scorer.model_version,
        )

    def score_batch(self, transactions: list[Transaction]) -> list[ScoredTransaction]:
        """Score a batch of transactions efficiently. Target: <100ms total for 100 txns."""
        t0 = time.perf_counter()
        results = []

        # 1. Update state for all transactions
        feature_matrix = []
        for txn in transactions:
            velocity = self.state.add_transaction(txn)
            features = build_feature_vector(txn, velocity)
            feature_matrix.append(features)
            results.append((txn, velocity))

        # 2. Score all at once (batched prediction)
        X = np.array(feature_matrix)
        probas = self.scorer.score_batch(X)

        # 3. Build results
        latency_ms = (time.perf_counter() - t0) * 1000
        per_tx_ms = latency_ms / len(transactions)

        scored = []
        for i, (txn, velocity) in enumerate(results):
            scored.append(ScoredTransaction(
                user_id=txn.user_id,
                fraud_probability=round(float(probas[i]), 6),
                fraud_flag=probas[i] > 0.5,
                velocity_features=velocity,
                latency_ms=round(per_tx_ms, 3),
                model_version=self.scorer.model_version,
            ))
        return scored

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        state_stats = self.state.get_stats()
        return {
            **state_stats,
            "model_version": self.scorer.model_version,
            "model_type": type(self.scorer.model).__name__,
        }


# ── Latency Benchmark ──

def benchmark_latency(scorer: RealtimeScorer, n_warmup: int = 100,
                      batch_sizes: list[int] = None) -> dict:
    """Comprehensive latency benchmark."""
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 500, 1000]

    rng = np.random.RandomState(42)
    results = []

    print(f"\n  {'Batch':>8s}  {'Median (ms)':>12s}  {'P95 (ms)':>10s}  {'P99 (ms)':>10s}  {'Per-tx (µs)':>12s}  {'Throughput':>12s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}")

    for bs in batch_sizes:
        # Generate synthetic transactions
        txns = []
        for _ in range(bs):
            uid = f"user_{rng.randint(0, 5000)}"
            txns.append(Transaction(
                user_id=uid,
                amount=round(rng.exponential(100), 2),
                hour=float(rng.randint(0, 24)),
                minute=float(rng.randint(0, 60)),
                day=float(rng.randint(1, 28)),
                merchant_id=rng.randint(0, 1000),
                city_id=rng.randint(0, 100),
                chip=rng.randint(0, 3),
                mcc=round(rng.uniform(0, 1), 4),
                is_online=rng.randint(0, 2),
                is_error=rng.randint(0, 2),
            ))

        # Warmup
        for _ in range(n_warmup):
            scorer.score_batch(txns[:min(10, bs)])

        # Benchmark (20 iterations)
        times = []
        for _ in range(20):
            t = time.perf_counter()
            scorer.score_batch(txns)
            times.append((time.perf_counter() - t) * 1000)

        times = np.array(times)
        median_ms = np.median(times)
        p95_ms = np.percentile(times, 95)
        p99_ms = np.percentile(times, 99)
        per_tx_us = (median_ms / bs) * 1000
        throughput = bs / (median_ms / 1000)

        results.append({
            "batch_size": bs,
            "median_ms": round(median_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "p99_ms": round(p99_ms, 2),
            "per_tx_us": round(per_tx_us, 1),
            "throughput_tps": round(throughput, 0),
        })
        print(f"  {bs:>8d}  {median_ms:>12.2f}  {p95_ms:>10.2f}  {p99_ms:>10.2f}  {per_tx_us:>12.1f}  {throughput:>12.0f}")

    return {"results": results, "stats": scorer.get_stats()}


# ── CLI ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time fraud scoring pipeline")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--score", type=str, help="Score a JSON transaction file")
    parser.add_argument("--output", type=str, help="Output file for scored results")
    args = parser.parse_args()

    print("=" * 60)
    print("  REAL-TIME FRAUD SCORING PIPELINE")
    print("=" * 60)

    print("\n  Initializing pipeline...")
    t0 = time.time()
    scorer = RealtimeScorer()
    print(f"  Model loaded: {scorer.get_stats()['model_version']} ({time.time()-t0:.1f}s)")

    if args.benchmark:
        bench = benchmark_latency(scorer)
        # Save
        out = Path("reports") / "realtime_benchmark.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(bench, indent=2))
        print(f"\n  Saved: {out}")

    if args.score:
        print(f"\n  Scoring: {args.score}")
        with open(args.score) as f:
            txns_data = json.load(f)
        txns = [Transaction(**t) for t in txns_data]
        results = scorer.score_batch(txns)
        output = [{
            "user_id": r.user_id,
            "fraud_probability": r.fraud_probability,
            "fraud_flag": r.fraud_flag,
            "latency_ms": r.latency_ms,
        } for r in results]
        if args.output:
            Path(args.output).write_text(json.dumps(output, indent=2))
            print(f"  Output: {args.output}")
        else:
            for r in results[:5]:
                print(f"  {r.user_id}: p={r.fraud_probability:.4f} flag={r.fraud_flag} ({r.latency_ms:.2f}ms)")

    if not args.benchmark and not args.score:
        # Default: run benchmark
        bench = benchmark_latency(scorer)
        out = Path("reports") / "realtime_benchmark.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(bench, indent=2))
        print(f"\n  Saved: {out}")
