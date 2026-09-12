"""Centralized Redis connection manager with fallback.

Provides a single Redis connection shared across all PS-14 services:
  - Session store (admin login sessions)
  - Velocity tracker (per-user/card/merchant tx counts)
  - Entity fraud rates (rolling fraud rates per entity)
  - Rate limiter (per-IP sliding window)
  - Sliding window state (per-user transaction history)

Connection modes:
  - Standalone: REDIS_URL env var (default: redis://localhost:6379/0)
  - Sentinel: REDIS_SENTINEL_URL + REDIS_SENTINEL_MASTER
  - Unavailable: graceful fallback to in-memory (dict-based)

All callers get a connection from the pool, use it, and let the pool
manage reconnection. No caller creates its own Redis client.

Thread-safe: Redis connection pool handles concurrent access.
"""
from __future__ import annotations

import os
import time
import threading
from typing import Optional

import redis


_pool: Optional[redis.ConnectionPool] = None
_client: Optional[redis.Redis] = None
_lock = threading.Lock()


def get_redis() -> Optional[redis.Redis]:
    """Get the shared Redis client. Returns None if Redis is unavailable.

    Reuses a single connection pool across all callers in the process.
    The pool handles reconnection automatically.
    """
    global _pool, _client
    if _client is not None:
        return _client

    with _lock:
        if _client is not None:
            return _client

        # Try Sentinel first
        sentinel_url = os.environ.get("REDIS_SENTINEL_URL", "")
        sentinel_master = os.environ.get("REDIS_SENTINEL_MASTER", "")
        redis_url = os.environ.get("REDIS_URL", "")

        if sentinel_url and sentinel_master:
            try:
                from redis.sentinel import Sentinel as RedisSentinel
                hosts = []
                for part in sentinel_url.split(","):
                    host_port = part.strip().split(":")
                    if len(host_port) == 2:
                        hosts.append((host_port[0], int(host_port[1])))
                sentinel = RedisSentinel(hosts, socket_timeout=2)
                _client = sentinel.master_for(
                    sentinel_master, socket_timeout=2, socket_connect_timeout=2
                )
                _client.ping()
                return _client
            except Exception:
                pass

        if redis_url:
            try:
                _pool = redis.ConnectionPool.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_timeout=2,
                    socket_connect_timeout=2,
                    max_connections=20,
                )
                _client = redis.Redis(connection_pool=_pool)
                _client.ping()
                return _client
            except Exception:
                _pool = None
                _client = None

    return None


def redis_available() -> bool:
    """Quick check if Redis is available (cached)."""
    return get_redis() is not None


def flush_all() -> int:
    """Flush all PS-14 keys (development/testing only)."""
    client = get_redis()
    if client is None:
        return 0
    try:
        keys = client.keys("ps14:*")
        if keys:
            return client.delete(*keys)
    except Exception:
        pass
    return 0
