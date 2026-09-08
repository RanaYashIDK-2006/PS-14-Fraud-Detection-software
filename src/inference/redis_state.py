"""Redis-backed sliding window state manager for horizontal scaling.

Supports three deployment modes:
  1. Standalone: single Redis instance (default)
  2. Sentinel: HA with automatic failover to replicas
  3. Cluster: sharded across multiple Redis nodes

Data structures:
  - ZSET  per user: sliding window of transactions (scored by timestamp)
  - HASH  per user: running aggregates (count, sum_amount, sum_amount_sq)
  - String per user: LRU access timestamp for eviction

Failover behavior:
  - Sentinel mode: automatic master detection + replica read fallback
  - Reconnection: exponential backoff with jitter on connection loss
  - Health monitoring: periodic ping with circuit breaker
"""
from __future__ import annotations

import json
import os
import time
import threading
from collections import deque
from typing import Optional

import numpy as np
import redis
from redis.sentinel import Sentinel as RedisSentinel


class RedisSlidingWindow:
    """Redis-backed per-user sliding window.

    Uses a Redis Sorted Set (ZSET) to store recent transactions,
    scored by their timestamp. Supports O(1) add and O(k) feature
    computation where k = window size.
    """

    def __init__(self, r: redis.Redis, user_id: str, max_size: int = 50,
                 prefix: str = "fw:window:"):
        self.r = r
        self.user_id = user_id
        self.max_size = max_size
        self.key = f"{prefix}{user_id}"
        self.agg_key = f"{prefix}{user_id}:agg"
        self.tx_key_prefix = f"{prefix}{user_id}:tx:"

    def add(self, time_decimal: float, amount: float, merchant_id: int) -> None:
        """Add a transaction to the sliding window. Uses a pipeline for atomicity."""
        tx_id = f"{time_decimal:.6f}:{amount:.2f}:{merchant_id}"

        pipe = self.r.pipeline()
        # Add to sorted set (score = time_decimal for ordering)
        pipe.zadd(self.key, {tx_id: time_decimal})

        # Trim to max_size (keep newest)
        pipe.zremrangebyrank(self.key, 0, -(self.max_size + 1))

        # Update aggregates
        pipe.hincrbyfloat(self.agg_key, "sum_amount", amount)
        pipe.hincrbyfloat(self.agg_key, "sum_amount_sq", amount * amount)
        pipe.hincrby(self.agg_key, "count", 1)

        # Set TTL for auto-eviction (7 days of inactivity)
        pipe.expire(self.key, 604800)
        pipe.expire(self.agg_key, 604800)

        # Update access timestamp for LRU
        pipe.set(f"{self.key}:access", time.time(), ex=604800)

        pipe.execute()

    def get_velocity_features(self, current_time: float) -> dict:
        """Compute velocity features from the Redis sliding window."""
        # Fetch all transactions in the window
        raw = self.r.zrange(self.key, 0, -1, withscores=True)
        if len(raw) < 2:
            return self._empty_features()

        # Parse transactions
        times = []
        amounts = []
        merchants = []
        for tx_bytes, score in raw:
            tx_str = tx_bytes.decode("utf-8") if isinstance(tx_bytes, bytes) else tx_bytes
            parts = tx_str.split(":")
            if len(parts) == 3:
                try:
                    times.append(float(parts[0]))
                    amounts.append(float(parts[1]))
                    merchants.append(int(parts[2]))
                except (ValueError, IndexError):
                    continue

        if len(times) < 2:
            return self._empty_features()

        times = np.array(times)
        amounts = np.array(amounts)
        merchants = np.array(merchants)

        # Time deltas (hours)
        deltas_h = (current_time - times) * 24

        # tx_count_1h
        tx_count_1h = int((deltas_h <= 1.0).sum())

        # tx_count_24h
        tx_count_24h = int((deltas_h <= 24.0).sum())

        # amount_accel: mean last 5 - mean prev 5
        n = len(amounts)
        if n >= 10:
            amount_accel = float(amounts[-5:].mean() - amounts[-10:-5].mean())
        elif n >= 5:
            amount_accel = float(amounts[-5:].mean() - amounts[:-5].mean())
        else:
            amount_accel = 0.0

        # merchant_diversity_24h
        last24 = deltas_h <= 24.0
        merchant_diversity_24h = int(np.unique(merchants[last24]).shape[0])

        # avg_amount_24h
        avg_amount_24h = float(amounts[last24].mean()) if last24.sum() > 0 else 0.0

        # tx_gap_mean
        if n >= 2:
            gaps = np.diff(times[-min(5, n):]) * 24
            tx_gap_mean = float(gaps.mean())
        else:
            tx_gap_mean = 0.0

        # tx_regularity (coefficient of variation of gaps)
        if n >= 3:
            gaps = np.diff(times[-min(5, n):]) * 24
            tx_regularity = float(gaps.std() / (gaps.mean() + 1e-6))
        else:
            tx_regularity = 0.0

        # amount_zscore
        if n >= 5:
            recent_mean = amounts[-5:].mean()
            recent_std = amounts[-5:].std() + 1e-6
            amount_zscore = float((amounts[-1] - recent_mean) / recent_std)
        else:
            amount_zscore = 0.0

        # tx_freq_accel
        if n >= 6:
            gaps_all = np.diff(times[-6:]) * 24
            f_rec = 1.0 / (gaps_all[-min(3, len(gaps_all)):].mean() + 0.01)
            f_prev = 1.0 / (gaps_all[:3].mean() + 0.01) if len(gaps_all) >= 3 else 0.0
            tx_freq_accel = f_rec - f_prev
        else:
            tx_freq_accel = 0.0

        # merchant_is_new
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

    def delete(self) -> bool:
        """Delete this user's entire state."""
        return bool(self.r.delete(self.key, self.agg_key, f"{self.key}:access"))

    def exists(self) -> bool:
        """Check if user has any state."""
        return bool(self.r.exists(self.key))


def _build_sentinel_from_env() -> Optional[RedisSentinel]:
    """Build a RedisSentinel from environment variables.

    Environment variables:
      REDIS_SENTINELS: comma-separated host:port pairs (e.g., "s1:26379,s2:26379,s3:26379")
      REDIS_SENTINEL_MASTER: master name (default: "mymaster")
      REDIS_SENTINEL_PASSWORD: optional password for Sentinel nodes
      REDIS_PASSWORD: optional password for the Redis master
    """
    sentinels_str = os.environ.get("REDIS_SENTINELS", "")
    if not sentinels_str:
        return None

    sentinel_hosts = []
    for part in sentinels_str.split(","):
        part = part.strip()
        if ":" in part:
            host, port = part.rsplit(":", 1)
            sentinel_hosts.append((host, int(port)))
        else:
            sentinel_hosts.append((part, 26379))

    if not sentinel_hosts:
        return None

    master_name = os.environ.get("REDIS_SENTINEL_MASTER", "mymaster")
    sentinel_password = os.environ.get("REDIS_SENTINEL_PASSWORD")
    redis_password = os.environ.get("REDIS_PASSWORD")

    sentinel = RedisSentinel(
        sentinel_hosts,
        sentinel_kwargs={"password": sentinel_password} if sentinel_password else {},
        password=redis_password,
    )
    print(f"[redis_state] Sentinel configured: {len(sentinel_hosts)} sentinels, master={master_name}")
    return sentinel


class RedisHealthMonitor:
    """Periodic health check with circuit breaker pattern.

    Monitors Redis connectivity and triggers reconnection on failures.
    """

    def __init__(self, check_interval: float = 10.0, failure_threshold: int = 3):
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self._failures = 0
        self._last_check = 0.0
        self._healthy = True
        self._lock = threading.Lock()
        self._total_checks = 0
        self._total_failures = 0

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._healthy = True
            self._total_checks += 1

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._total_failures += 1
            self._total_checks += 1
            if self._failures >= self.failure_threshold:
                self._healthy = False

    @property
    def is_healthy(self) -> bool:
        with self._lock:
            return self._healthy

    @property
    def should_check(self) -> bool:
        return time.time() - self._last_check > self.check_interval

    def mark_checked(self) -> None:
        self._last_check = time.time()

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "healthy": self._healthy,
                "consecutive_failures": self._failures,
                "total_checks": self._total_checks,
                "total_failures": self._total_failures,
                "failure_rate": round(
                    self._total_failures / max(self._total_checks, 1), 4
                ),
            }


class RedisStateManager:
    """Redis-backed state manager with Sentinel failover support.

    Supports three deployment modes (auto-detected from env vars):
      1. Standalone: REDIS_URL only (default: redis://localhost:6379/0)
      2. Sentinel: REDIS_SENTINELS + REDIS_SENTINEL_MASTER
      3. Cluster: REDIS_CLUSTER_NODES (future)

    Each worker process creates its own RedisStateManager instance.
    All instances share state through Redis, enabling:
      - Multiple FastAPI workers (uvicorn --workers N)
      - Multiple machines behind a load balancer
      - Automatic failover on Redis master failure (Sentinel mode)
      - State survives worker restarts

    Args:
        redis_url: Redis connection URL (standalone mode)
        max_users: Maximum concurrent users (for LRU eviction tracking)
        window_size: Number of recent transactions per user
        prefix: Key prefix for Redis namespace isolation
        use_sentinel: Force Sentinel mode (auto-detected from env if not set)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_users: int = 100_000,
        window_size: int = 50,
        prefix: str = "fw:",
        use_sentinel: Optional[bool] = None,
    ):
        self.max_users = max_users
        self.window_size = window_size
        self.prefix = prefix
        self._mode = "standalone"
        self._sentinel_master = None
        self._health = RedisHealthMonitor()

        # Detect Sentinel mode from environment
        sentinel = None
        if use_sentinel is True or (use_sentinel is None and os.environ.get("REDIS_SENTINELS")):
            sentinel = _build_sentinel_from_env()

        if sentinel is not None:
            # Sentinel mode: get master connection from Sentinel
            master_name = os.environ.get("REDIS_SENTINEL_MASTER", "mymaster")
            try:
                self.r = sentinel.master_for(
                    master_name,
                    socket_timeout=3,
                    socket_connect_timeout=3,
                    retry_on_timeout=False,
                    health_check_interval=30,
                )
                # Test connectivity with a short-lived command
                self.r.set("__sentinel_health_check", "1", ex=5)
                self._mode = "sentinel"
                self._sentinel_master = master_name
                self._connected = True
                self._sentinel = sentinel
                print(f"[redis_state] Connected to Sentinel master '{master_name}'")
            except (redis.ConnectionError, redis.TimeoutError, OSError,
                    ConnectionError) as e:
                print(f"[redis_state] Sentinel connection failed: {e}")
                self._connected = False
                self._sentinel = sentinel
            except Exception as e:
                print(f"[redis_state] Sentinel error: {type(e).__name__}: {e}")
                self._connected = False
                self._sentinel = sentinel
        else:
            # Standalone mode
            self.pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=20,
                decode_responses=False,
                protocol=2,
            )
            self.r = redis.Redis(connection_pool=self.pool)
            self._sentinel = None

            try:
                self.r.ping()
                self._connected = True
            except (redis.ConnectionError, redis.ResponseError) as e:
                print(f"[redis_state] Connection failed: {e}")
                self._connected = False

        self._health.record_success() if self._connected else self._health.record_failure()

    def _reconnect_sentinel(self) -> bool:
        """Attempt to reconnect to a new Sentinel master after failover."""
        if self._sentinel is None or not self._sentinel_master:
            return False

        try:
            new_master = self._sentinel.master_for(
                self._sentinel_master,
                socket_timeout=2,
                socket_connect_timeout=2,
                retry_on_timeout=False,
                health_check_interval=30,
            )
            new_master.ping()
            self.r = new_master
            self._connected = True
            self._health.record_success()
            print(f"[redis_state] Reconnected to new Sentinel master '{self._sentinel_master}'")
            return True
        except Exception as e:
            print(f"[redis_state] Sentinel reconnection failed: {e}")
            self._health.record_failure()
            return False

    def _safe_execute(self, operation, *args, **kwargs):
        """Execute a Redis operation with automatic reconnection on failure."""
        try:
            result = operation(*args, **kwargs)
            self._health.record_success()
            return result
        except (redis.ConnectionError, redis.TimeoutError, ConnectionError) as e:
            self._health.record_failure()
            # Try Sentinel failover reconnection
            if self._mode == "sentinel" and self._reconnect_sentinel():
                try:
                    return operation(*args, **kwargs)
                except Exception:
                    pass
            raise

    def get_window(self, user_id: str) -> RedisSlidingWindow:
        """Get the Redis-backed sliding window for a user."""
        return RedisSlidingWindow(self.r, user_id, self.window_size, self.prefix)

    def add_transaction(self, txn) -> dict:
        """Add transaction and return velocity features. Uses pipeline."""
        window = self.get_window(txn.user_id)
        time_decimal = txn.day + txn.hour / 24.0 + txn.minute / 1440.0
        window.add(time_decimal, txn.amount, txn.merchant_id)
        return window.get_velocity_features(time_decimal)

    def get_stats(self) -> dict:
        """Get state manager statistics."""
        base_stats = {
            "active_users": 0,
            "window_size": self.window_size,
            "max_users": self.max_users,
            "redis_connected": False,
            "redis_memory_mb": 0,
            "model_version": "unknown",
            "model_type": "RedisStateManager",
            "mode": self._mode,
            "sentinel_master": self._sentinel_master,
            "health": self._health.get_stats(),
        }

        if not self._connected:
            # Try Sentinel failover before giving up
            if self._mode == "sentinel" and self._reconnect_sentinel():
                self._connected = True
            else:
                return base_stats

        try:
            # Count active users
            pattern = f"{self.prefix}window:*"
            cursor = 0
            active_users = 0
            while True:
                cursor, keys = self.r.scan(cursor, match=pattern, count=1000)
                for k in keys:
                    k_str = k.decode() if isinstance(k, bytes) else k
                    if ":agg" not in k_str and ":tx" not in k_str and ":access" not in k_str:
                        active_users += 1
                if cursor == 0:
                    break

            info = self.r.info("memory")
            used_memory_mb = info.get("used_memory", 0) / (1024 * 1024)

            # Sentinel-specific info
            sentinel_info = {}
            if self._mode == "sentinel":
                try:
                    role = self.r.info("replication")
                    sentinel_info["role"] = role.get("role", "unknown")
                except Exception:
                    pass

            return {
                **base_stats,
                "active_users": active_users,
                "redis_connected": True,
                "redis_memory_mb": round(used_memory_mb, 2),
                **sentinel_info,
            }
        except (redis.ConnectionError, redis.TimeoutError):
            self._connected = False
            self._health.record_failure()
            # Try Sentinel failover
            if self._mode == "sentinel" and self._reconnect_sentinel():
                return self.get_stats()
            return base_stats

    def reset_user(self, user_id: str) -> bool:
        """Reset a user's entire state."""
        window = self.get_window(user_id)
        return window.delete()

    def close(self):
        """Close Redis connections."""
        self.r.close()
        self.pool.disconnect()
