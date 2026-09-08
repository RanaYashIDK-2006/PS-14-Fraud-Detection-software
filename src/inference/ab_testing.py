"""A/B testing framework for fraud detection models.

Manages traffic splitting between model versions, tracks per-version
metrics, and provides experiment lifecycle management.

Traffic splitting strategies:
  1. Percentage-based: random split by configurable percentages
  2. User-deterministic: same user always goes to same model (consistent hashing)
  3. Header-based: explicit version override via request header

Metrics tracked per version:
  - Score count, mean fraud probability, flag rate
  - Latency (mean, p50, p95, p99)
  - Feature distribution stats (for drift detection)
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VersionMetrics:
    """Accumulated metrics for a single model version."""
    version: str
    n_scored: int = 0
    n_flagged: int = 0
    sum_latency_ms: float = 0.0
    sum_latency_sq: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    sum_fraud_prob: float = 0.0
    sum_fraud_prob_sq: float = 0.0
    latencies: list = field(default_factory=list)  # kept bounded
    started_at: float = field(default_factory=time.time)
    last_scored_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, fraud_prob: float, fraud_flag: bool, latency_ms: float) -> None:
        with self._lock:
            self.n_scored += 1
            self.n_flagged += int(fraud_flag)
            self.sum_latency_ms += latency_ms
            self.sum_latency_sq += latency_ms * latency_ms
            self.min_latency_ms = min(self.min_latency_ms, latency_ms)
            self.max_latency_ms = max(self.max_latency_ms, latency_ms)
            self.sum_fraud_prob += fraud_prob
            self.sum_fraud_prob_sq += fraud_prob * fraud_prob
            self.last_scored_at = time.time()
            # Keep last 10000 latencies for percentile computation
            if len(self.latencies) < 10_000:
                self.latencies.append(latency_ms)

    def to_dict(self) -> dict:
        with self._lock:
            n = self.n_scored
            if n == 0:
                return {
                    "version": self.version, "n_scored": 0,
                    "flag_rate": 0, "mean_latency_ms": 0,
                    "uptime_s": round(time.time() - self.started_at, 1),
                }

            mean_lat = self.sum_latency_ms / n
            mean_prob = self.sum_fraud_prob / n

            # Percentiles from bounded list
            lats = sorted(self.latencies) if self.latencies else [0]
            p50 = lats[len(lats) // 2] if lats else 0
            p95 = lats[int(len(lats) * 0.95)] if lats else 0
            p99 = lats[int(len(lats) * 0.99)] if lats else 0

            return {
                "version": self.version,
                "n_scored": n,
                "n_flagged": self.n_flagged,
                "flag_rate": round(self.n_flagged / n, 4),
                "mean_fraud_prob": round(mean_prob, 6),
                "mean_latency_ms": round(mean_lat, 3),
                "median_latency_ms": round(p50, 3),
                "p95_latency_ms": round(p95, 3),
                "p99_latency_ms": round(p99, 3),
                "min_latency_ms": round(self.min_latency_ms, 3),
                "max_latency_ms": round(self.max_latency_ms, 3),
                "uptime_s": round(time.time() - self.started_at, 1),
                "last_scored_s_ago": round(time.time() - self.last_scored_at, 1) if self.last_scored_at else None,
            }


@dataclass
class ABExperiment:
    """A single A/B experiment configuration."""
    name: str
    versions: dict[str, float]  # version → traffic percentage (0.0–1.0)
    strategy: str = "percentage"  # "percentage" or "user_deterministic"
    created_at: float = field(default_factory=time.time)
    active: bool = True
    description: str = ""

    def __post_init__(self):
        total = sum(self.versions.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 1.0, got {total}")


class TrafficSplitter:
    """Routes transactions to model versions based on experiment config.

    Strategies:
      - percentage: random split using hash(transaction_id) % 1000
      - user_deterministic: same user always goes to same version (consistent)
    """

    def __init__(self):
        self._experiments: dict[str, ABExperiment] = {}
        self._metrics: dict[str, dict[str, VersionMetrics]] = defaultdict(dict)
        self._lock = threading.Lock()

    def create_experiment(self, name: str, versions: dict[str, float],
                          strategy: str = "percentage",
                          description: str = "") -> ABExperiment:
        """Create a new A/B experiment."""
        exp = ABExperiment(name=name, versions=versions, strategy=strategy,
                           description=description)
        with self._lock:
            self._experiments[name] = exp
            # Initialize metrics for each version
            for v in versions:
                if v not in self._metrics[name]:
                    self._metrics[name][v] = VersionMetrics(version=v)
        return exp

    def get_version(self, experiment: str, user_id: str = None,
                    txn_id: str = None) -> str:
        """Determine which model version to use for this request.

        Returns the version string to use for scoring.
        """
        exp = self._experiments.get(experiment)
        if not exp or not exp.active:
            return ""

        if exp.strategy == "user_deterministic" and user_id:
            return self._user_hash_version(exp, user_id)
        else:
            return self._percentage_version(exp, txn_id or user_id or str(time.time()))

    def _percentage_version(self, exp: ABExperiment, key: str) -> str:
        """Random percentage-based routing using consistent hash."""
        h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10000
        bucket = h / 10000.0  # 0.0–0.9999

        cumulative = 0.0
        for version, pct in exp.versions.items():
            cumulative += pct
            if bucket < cumulative:
                return version

        # Fallback to last version
        return list(exp.versions.keys())[-1]

    def _user_hash_version(self, exp: ABExperiment, user_id: str) -> str:
        """Deterministic: same user always gets same version."""
        h = int(hashlib.md5(f"ab_{user_id}_{exp.name}".encode()).hexdigest()[:8], 16) % 10000
        bucket = h / 10000.0

        cumulative = 0.0
        for version, pct in exp.versions.items():
            cumulative += pct
            if bucket < cumulative:
                return version

        return list(exp.versions.keys())[-1]

    def record(self, experiment: str, version: str, fraud_prob: float,
               fraud_flag: bool, latency_ms: float) -> None:
        """Record a scoring result for metrics tracking."""
        if experiment in self._metrics and version in self._metrics[experiment]:
            self._metrics[experiment][version].record(fraud_prob, fraud_flag, latency_ms)

    def get_metrics(self, experiment: str) -> dict:
        """Get metrics for all versions in an experiment."""
        with self._lock:
            if experiment not in self._metrics:
                return {}
            return {
                v: m.to_dict()
                for v, m in self._metrics[experiment].items()
            }

    def get_experiment(self, name: str) -> Optional[ABExperiment]:
        """Get experiment configuration."""
        return self._experiments.get(name)

    def list_experiments(self) -> list[dict]:
        """List all experiments."""
        with self._lock:
            result = []
            for name, exp in self._experiments.items():
                # Inline metrics to avoid re-acquiring lock
                metrics = {}
                if name in self._metrics:
                    metrics = {v: m.to_dict() for v, m in self._metrics[name].items()}
                result.append({
                    "name": exp.name,
                    "versions": exp.versions,
                    "strategy": exp.strategy,
                    "active": exp.active,
                    "description": exp.description,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(exp.created_at)),
                    "metrics": metrics,
                })
            return result

    def deactivate_experiment(self, name: str) -> bool:
        """Deactivate an experiment (all traffic goes to default)."""
        with self._lock:
            if name in self._experiments:
                self._experiments[name].active = False
                return True
            return False

    def update_traffic(self, name: str, versions: dict[str, float]) -> bool:
        """Update traffic percentages for an experiment."""
        with self._lock:
            if name not in self._experiments:
                return False
            total = sum(versions.values())
            if abs(total - 1.0) > 0.01:
                return False
            self._experiments[name].versions = versions
            # Ensure metrics exist for new versions
            for v in versions:
                if v not in self._metrics[name]:
                    self._metrics[name][v] = VersionMetrics(version=v)
            return True
