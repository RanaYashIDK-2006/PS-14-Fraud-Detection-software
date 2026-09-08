"""Batch Scorer — optimized XGB+LGB ensemble with latency benchmarking.

Wraps the Optuna-tuned production model for high-throughput batch scoring.
Includes:
- Vectorized feature mapping (no per-row Python loops)
- Pre-fitted scaler cached in memory
- Latency benchmarking with percentile breakdown
- Model versioning with artifact metadata
"""
from __future__ import annotations

import gc
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import joblib


ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models" / "production"


@dataclass
class ModelVersion:
    """Metadata for a loaded model version."""
    version: str
    model_type: str
    n_features: int
    features: list[str]
    artifacts: list[str]
    loaded_at: float
    ensemble_weights: dict
    temporal_test_auc: float | None = None
    temporal_test_r1: float | None = None
    dataset_rows: int = 0
    training_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "n_features": self.n_features,
            "n_artifacts": len(self.artifacts),
            "ensemble_weights": self.ensemble_weights,
            "temporal_test_auc": self.temporal_test_auc,
            "temporal_test_r1": self.temporal_test_r1,
            "dataset_rows": self.dataset_rows,
            "training_rows": self.training_rows,
            "loaded_at": self.loaded_at,
        }


@dataclass
class LatencyBenchmark:
    """Latency benchmark results."""
    batch_size: int
    total_ms: float
    per_txn_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_tps: float
    n_iterations: int
    model_version: str

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "total_ms": round(self.total_ms, 2),
            "per_txn_ms": round(self.per_txn_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "throughput_tps": round(self.throughput_tps, 1),
            "n_iterations": self.n_iterations,
            "model_version": self.model_version,
        }


class BatchScorer:
    """High-throughput batch scorer wrapping the Optuna-tuned XGB+LGB ensemble.

    Thread-safe: model inference is GIL-released by XGBoost/LightGBM C extensions.
    Feature mapping is vectorized (numpy) — no per-row Python loops.
    """

    def __init__(self, prod_dir: Path | None = None):
        self._prod_dir = prod_dir or MODELS_DIR
        self._lock = threading.Lock()
        self._xgb = None
        self._lgb = None
        self._scaler = None
        self._version: ModelVersion | None = None
        self._feature_map: dict[str, int] | None = None
        self._total_scored = 0
        self._total_batch_ms = 0.0

    def load(self) -> ModelVersion:
        """Load model artifacts and metadata."""
        import json

        manifest_path = self._prod_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest at {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        features = manifest.get("features", [])

        # Load models
        xgb_path = self._prod_dir / "xgb_production.joblib"
        lgb_path = self._prod_dir / "lgb_production.joblib"
        scaler_path = self._prod_dir / "scaler_production.joblib"

        if not xgb_path.exists() or not lgb_path.exists():
            raise FileNotFoundError("Missing model artifacts")

        self._xgb = joblib.load(xgb_path)
        self._lgb = joblib.load(lgb_path)
        self._scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        # Build feature name → index mapping
        self._feature_map = {name: i for i, name in enumerate(features)}

        # Parse ensemble weights from manifest
        weights = manifest.get("ensemble_weights", {"xgb": 0.5, "lgb": 0.5})

        self._version = ModelVersion(
            version=manifest.get("model_version", "unknown"),
            model_type=manifest.get("model_type", "unknown"),
            n_features=len(features),
            features=features,
            artifacts=[f.name for f in self._prod_dir.glob("*.joblib")],
            loaded_at=time.time(),
            ensemble_weights=weights,
            temporal_test_auc=manifest.get("temporal_test_auc"),
            temporal_test_r1=manifest.get("temporal_test_r1"),
            dataset_rows=manifest.get("dataset_rows_scanned", 0),
            training_rows=manifest.get("training_rows", 0),
        )
        return self._version

    @property
    def version(self) -> ModelVersion | None:
        return self._version

    @property
    def is_loaded(self) -> bool:
        return self._xgb is not None and self._lgb is not None

    def _map_features_vectorized(self, transactions: list[dict]) -> np.ndarray:
        """Map ML_FEATURES dicts to the model's feature matrix (vectorized).

        Uses numpy arrays instead of per-row dict lookups for speed.
        """
        n = len(transactions)
        n_feat = len(self._version.features)
        X = np.zeros((n, n_feat), dtype=np.float32)

        # Batch-extract common fields
        amount_ratios = np.array([t.get("amount_ratio", 1.0) for t in transactions], dtype=np.float32)
        hours = np.array([t.get("hour_of_day", 12) for t in transactions], dtype=np.float32)
        is_weekends = np.array([t.get("is_weekend", 0) for t in transactions], dtype=np.float32)
        new_devices = np.array([t.get("new_device_flag", 0) for t in transactions], dtype=np.float32)
        txn_freqs = np.array([t.get("txn_freq_last_24h", 5) for t in transactions], dtype=np.float32)
        known_devices = np.array([t.get("known_device_count", 5) for t in transactions], dtype=np.float32)
        tenures = np.array([t.get("account_tenure_days", 180) for t in transactions], dtype=np.float32)
        failed_auths = np.array([t.get("failed_auth_count_24h", 0) for t in transactions], dtype=np.float32)
        amt_zscores = np.array([t.get("amount_zscore", 0.0) for t in transactions], dtype=np.float32)
        vel_devs = np.array([t.get("velocity_deviation", 0.3) for t in transactions], dtype=np.float32)

        # Velocity from tracker (real values, not proxies)
        user_tx = np.array([t.get("user_tx_count", 0) for t in transactions], dtype=np.float32)
        user_avg = np.array([t.get("user_avg_amt", 100.0) for t in transactions], dtype=np.float32)
        card_tx = np.array([t.get("card_tx_count", 0) for t in transactions], dtype=np.float32)
        merch_tx = np.array([t.get("merch_tx_count", 0) for t in transactions], dtype=np.float32)

        # Fraud rates (from tracker or defaults)
        user_fr = np.array([t.get("user_fraud_rate", 0.001) for t in transactions], dtype=np.float32)
        merch_fr = np.array([t.get("merch_fraud_rate", 0.001) for t in transactions], dtype=np.float32)
        city_fr = np.array([t.get("city_fraud_rate", 0.001) for t in transactions], dtype=np.float32)

        # Derived amounts
        ref_amt = 100.0
        amts = amount_ratios * ref_amt
        log_amts = np.log1p(amts)
        amt_sqs = amts ** 2

        # MCC (not in ML_FEATURES — default 0)
        mccs = np.zeros(n, dtype=np.float32)

        # Map to feature columns by name
        fmap = self._feature_map
        for i, name in enumerate(self._version.features):
            if name == "amt":
                X[:, i] = amts
            elif name == "log_amt":
                X[:, i] = log_amts
            elif name == "amt_sq":
                X[:, i] = amt_sqs
            elif name == "hr":
                X[:, i] = hours
            elif name == "mn":
                X[:, i] = 0.0
            elif name == "dow":
                X[:, i] = is_weekends
            elif name == "Month":
                X[:, i] = 8.0  # default
            elif name == "Day":
                X[:, i] = 15.0  # default
            elif name == "hour_sin":
                X[:, i] = np.sin(2 * np.pi * hours / 24)
            elif name == "hour_cos":
                X[:, i] = np.cos(2 * np.pi * hours / 24)
            elif name == "is_night":
                X[:, i] = ((hours < 6) | (hours > 22)).astype(np.float32)
            elif name == "is_business_hours":
                X[:, i] = ((hours >= 9) & (hours <= 17)).astype(np.float32)
            elif name == "chip":
                X[:, i] = new_devices  # proxy: online = new device
            elif name == "is_online":
                X[:, i] = new_devices
            elif name == "err":
                X[:, i] = (failed_auths > 0).astype(np.float32)
            elif name == "mcc_n":
                X[:, i] = mccs
            elif name == "user_tx_count":
                X[:, i] = user_tx
            elif name == "card_tx_count":
                X[:, i] = card_tx
            elif name == "user_avg_amt":
                X[:, i] = user_avg
            elif name == "amt_vs_user_avg":
                X[:, i] = amts / (user_avg + 1e-6)
            elif name == "amt_zscore":
                X[:, i] = amt_zscores
            elif name == "merch_tx_count":
                X[:, i] = merch_tx
            elif name == "user_fraud_rate":
                X[:, i] = user_fr
            elif name == "merch_fraud_rate":
                X[:, i] = merch_fr
            elif name == "city_fraud_rate":
                X[:, i] = city_fr
            elif name == "has_zip":
                X[:, i] = 0.0
            elif name == "has_state":
                X[:, i] = 0.0
            elif name == "is_online_or_no_state":
                X[:, i] = new_devices
            elif name == "high_amt":
                X[:, i] = (amts > user_avg * 2).astype(np.float32)
            elif name == "very_high_amt":
                X[:, i] = (amts > user_avg * 5).astype(np.float32)
            elif name == "amt_x_hr":
                X[:, i] = amts * hours
            elif name == "amt_x_mcc":
                X[:, i] = amts * mccs
            elif name == "amt_x_chip":
                X[:, i] = amts * new_devices
            elif name == "amt_x_online":
                X[:, i] = amts * new_devices
            elif name == "amt_x_night":
                X[:, i] = amts * ((hours < 6) | (hours > 22)).astype(np.float32)
            elif name == "account_tenure_days":
                X[:, i] = tenures
            elif name == "gradual_escalation_score":
                X[:, i] = 0.0
            elif name == "known_device_count":
                X[:, i] = known_devices
            elif name == "txn_freq_last_24h":
                X[:, i] = txn_freqs
            elif name == "txn_time_unusual":
                X[:, i] = 0.0
            elif name == "unusual_location_flag":
                X[:, i] = 0.0
            elif name == "unusual_recipient_flag":
                X[:, i] = 0.0
            elif name == "days_since_last_similar_txn":
                X[:, i] = 3.0
            elif name == "shared_device_accounts":
                X[:, i] = 0.0
            elif name == "shared_recipient_accounts":
                X[:, i] = 0.0
            elif name == "mule_ring_score":
                X[:, i] = 0.0
            elif name == "hour_deviation":
                X[:, i] = 0.0
            elif name == "velocity_deviation":
                X[:, i] = vel_devs
            elif name == "recipient_novelty":
                X[:, i] = 0.0
            elif name == "txn_regularity":
                X[:, i] = 0.5

        # Replace inf/nan
        np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        return X

    def score_batch(self, transactions: list[dict]) -> list[dict]:
        """Score a batch of transactions. Returns list of result dicts.

        Each result: {score, latency_ms, model_version, features_count}
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded — call load() first")

        t0 = time.perf_counter()

        # Vectorized feature mapping
        X = self._map_features_vectorized(transactions)

        # Scale
        if self._scaler is not None:
            X_s = self._scaler.transform(X)
        else:
            X_s = X

        # Ensemble prediction
        w = self._version.ensemble_weights
        p_xgb = self._xgb.predict_proba(X_s)[:, 1]
        p_lgb = self._lgb.predict_proba(X_s)[:, 1]
        scores = w.get("xgb", 0.5) * p_xgb + w.get("lgb", 0.5) * p_lgb

        total_ms = (time.perf_counter() - t0) * 1000

        # Update stats
        with self._lock:
            self._total_scored += len(transactions)
            self._total_batch_ms += total_ms

        # Build results
        results = []
        for i in range(len(transactions)):
            results.append({
                "score": round(float(scores[i]), 6),
                "xgb_score": round(float(p_xgb[i]), 6),
                "lgb_score": round(float(p_lgb[i]), 6),
                "model_version": self._version.version,
                "features_count": self._version.n_features,
            })

        return results

    def benchmark(self, n_transactions: int = 1000, n_iterations: int = 10,
                  batch_sizes: list[int] | None = None) -> list[LatencyBenchmark]:
        """Run latency benchmark across multiple batch sizes.

        Returns list of LatencyBenchmark for each batch size tested.
        """
        if batch_sizes is None:
            batch_sizes = [1, 10, 50, 100, 500, 1000]

        # Generate synthetic transactions for benchmarking
        rng = np.random.RandomState(42)
        synthetic = []
        for _ in range(n_transactions):
            synthetic.append({
                "amount_ratio": round(rng.lognormal(0, 0.5), 4),
                "hour_of_day": int(rng.randint(0, 24)),
                "is_weekend": int(rng.random() < 0.14),
                "new_device_flag": int(rng.random() < 0.05),
                "txn_freq_last_24h": int(rng.poisson(5)),
                "known_device_count": int(rng.poisson(3)),
                "account_tenure_days": round(rng.uniform(1, 1000), 1),
                "failed_auth_count_24h": int(rng.random() < 0.02),
                "amount_zscore": round(rng.normal(0, 1), 4),
                "velocity_deviation": round(rng.beta(2, 5), 4),
                "user_tx_count": int(rng.poisson(50)),
                "user_avg_amt": round(rng.lognormal(4, 0.8), 2),
                "card_tx_count": int(rng.poisson(30)),
                "merch_tx_count": int(rng.poisson(20)),
                "user_fraud_rate": round(rng.beta(1, 100), 6),
                "merch_fraud_rate": round(rng.beta(1, 50), 6),
                "city_fraud_rate": round(rng.beta(1, 50), 6),
            })

        benchmarks = []
        for bs in batch_sizes:
            bs = min(bs, n_transactions)
            per_iter_latencies = []

            for _ in range(n_iterations):
                batch = synthetic[:bs]
                t0 = time.perf_counter()
                self.score_batch(batch)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                per_iter_latencies.append(elapsed_ms)

            arr = np.array(per_iter_latencies)
            total_ms = float(arr.mean())
            per_txn = total_ms / bs
            throughput = bs / (total_ms / 1000) if total_ms > 0 else 0

            benchmarks.append(LatencyBenchmark(
                batch_size=bs,
                total_ms=total_ms,
                per_txn_ms=per_txn,
                p50_ms=float(np.percentile(arr, 50)),
                p95_ms=float(np.percentile(arr, 95)),
                p99_ms=float(np.percentile(arr, 99)),
                throughput_tps=throughput,
                n_iterations=n_iterations,
                model_version=self._version.version if self._version else "unknown",
            ))

        return benchmarks

    @property
    def stats(self) -> dict:
        """Scorer statistics."""
        avg_batch_ms = (self._total_batch_ms / max(self._total_scored, 1))
        return {
            "total_scored": self._total_scored,
            "total_batches_ms": round(self._total_batch_ms, 2),
            "avg_per_txn_ms": round(avg_batch_ms, 3),
            "model_loaded": self.is_loaded,
            "model_version": self._version.version if self._version else None,
            "n_features": self._version.n_features if self._version else 0,
        }


# Module-level singleton
_scorer: BatchScorer | None = None


def get_batch_scorer() -> BatchScorer:
    """Get or create the singleton batch scorer."""
    global _scorer
    if _scorer is None:
        _scorer = BatchScorer()
    return _scorer
