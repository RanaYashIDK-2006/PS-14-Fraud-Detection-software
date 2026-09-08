"""Entity-level rolling fraud rate tracker.

Computes user_fraud_rate, merch_fraud_rate, city_fraud_rate using ONLY
historical data (no target leakage). Each entity maintains a sliding
window of recent events with their labels, and the fraud rate is the
fraction of confirmed fraud events in that window.

Architecture:
  EntityFraudRateTracker (in-memory, thread-safe)
    → called by risk engine after each evaluation
    → stores (label, timestamp) per entity in a deque
    → computes rolling fraud rate over last N events

Thresholds:
  - window_size: 100 events per entity (configurable)
  - min_events: 5 before computing a rate (otherwise return baseline)
  - baseline: global fraud rate estimate (~0.001 for Altman)

Usage:
    tracker = EntityFraudRateTracker()
    # After scoring:
    tracker.record("user_123", "merchant_456", "city_789", is_fraud=False)
    # Before scoring:
    rates = tracker.get_rates("user_123", "merchant_456", "city_789")
    # → {"user_fraud_rate": 0.02, "merch_fraud_rate": 0.01, "city_fraud_rate": 0.005}
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EntityWindow:
    """Sliding window of (label, timestamp) for one entity."""
    events: deque = field(default_factory=lambda: deque(maxlen=100))

    def add(self, is_fraud: bool, ts: float = None) -> None:
        self.events.append((1 if is_fraud else 0, ts or time.time()))

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def fraud_rate(self) -> float:
        if len(self.events) == 0:
            return 0.0
        return sum(label for label, _ in self.events) / len(self.events)

    @property
    def fraud_count(self) -> int:
        return sum(label for label, _ in self.events)


class EntityFraudRateTracker:
    """Thread-safe entity-level rolling fraud rate tracker.

    Maintains three separate entity windows:
    - user_id → user_fraud_rate
    - merchant_id → merch_fraud_rate
    - city_id → city_fraud_rate

    The fraud rate is the fraction of confirmed fraud events in the
    sliding window. This is LEAKAGE-SAFE because:
    1. The rate is computed BEFORE the current event is scored
    2. Only HISTORICAL events (already labeled) contribute
    3. The window is per-entity, not global
    """

    def __init__(self, window_size: int = 100, min_events: int = 5,
                 baseline_rate: float = 0.001):
        self._window_size = window_size
        self._min_events = min_events
        self._baseline = baseline_rate

        # Entity windows (thread-safe)
        self._lock = threading.Lock()
        self._users: dict[str, EntityWindow] = defaultdict(
            lambda: EntityWindow(events=deque(maxlen=window_size)))
        self._merchants: dict[str, EntityWindow] = defaultdict(
            lambda: EntityWindow(events=deque(maxlen=window_size)))
        self._cities: dict[str, EntityWindow] = defaultdict(
            lambda: EntityWindow(events=deque(maxlen=window_size)))

        # Global stats
        self._total_events = 0
        self._total_fraud = 0

        # Redis for multi-worker state sharing (lazy-init via centralized connection)
        self._redis = None
        self._redis_tried = False
        self._redis_prefix = 'ps14:efr:'

    def _ensure_redis(self) -> None:
        """Lazy-init Redis from centralized connection manager."""
        if self._redis_tried:
            return
        self._redis_tried = True
        try:
            from src.redis_conn import get_redis
            self._redis = get_redis()
        except Exception:
            pass

    def record(self, user_id: str, merchant_id: str, city_id: str,
               is_fraud: bool, ts: float = None) -> None:
        """Record a labeled event for an entity.

        Call this AFTER the event has been evaluated and the true label
        is known (e.g., from user feedback or investigation).
        """
        if not user_id and not merchant_id and not city_id:
            return

        t = ts or time.time()

        self._ensure_redis()
        # Write to Redis if available
        if self._redis:
            try:
                p = self._redis.pipeline()
                for eid, prefix in [(user_id, 'u'), (merchant_id, 'm'), (city_id, 'c')]:
                    if eid:
                        key = f'{self._redis_prefix}{prefix}:{eid}'
                        p.hincrby(key, 'total', 1)
                        if is_fraud:
                            p.hincrby(key, 'fraud', 1)
                        p.expire(key, 86400)  # 24h TTL
                p.execute()
            except Exception:
                pass

        # Always update in-memory
        with self._lock:
            if user_id:
                self._users[user_id].add(is_fraud, t)
            if merchant_id:
                self._merchants[merchant_id].add(is_fraud, t)
            if city_id:
                self._cities[city_id].add(is_fraud, t)
            self._total_events += 1
            if is_fraud:
                self._total_fraud += 1

    def get_rates(self, user_id: str = "", merchant_id: str = "",
                  city_id: str = "") -> dict[str, float]:
        """Get entity-level fraud rates. Checks Redis first if available."""
        self._ensure_redis()
        # Try Redis first
        if self._redis:
            try:
                user_rate = self._redis.hget(f'{self._redis_prefix}u:{user_id}', 'rate')
                merch_rate = self._redis.hget(f'{self._redis_prefix}m:{merchant_id}', 'rate')
                city_rate = self._redis.hget(f'{self._redis_prefix}c:{city_id}', 'rate')
                return {
                    'user_fraud_rate': float(user_rate) if user_rate else self._baseline,
                    'merch_fraud_rate': float(merch_rate) if merch_rate else self._baseline,
                    'city_fraud_rate': float(city_rate) if city_rate else self._baseline,
                }
            except Exception:
                pass

        # In-memory fallback
        with self._lock:
            user_rate = self._entity_rate(user_id, self._users)
            merch_rate = self._entity_rate(merchant_id, self._merchants)
            city_rate = self._entity_rate(city_id, self._cities)

        return {
            "user_fraud_rate": round(user_rate, 6),
            "merch_fraud_rate": round(merch_rate, 6),
            "city_fraud_rate": round(city_rate, 6),
        }

    def _entity_rate(self, entity_id: str, store: dict) -> float:
        """Compute fraud rate for one entity, falling back to baseline."""
        if not entity_id:
            return self._baseline
        window = store.get(entity_id)
        if window is None or window.n_events < self._min_events:
            return self._baseline
        return window.fraud_rate

    def get_entity_stats(self, entity_type: str = "user",
                         entity_id: str = "") -> dict:
        """Get detailed stats for one entity."""
        with self._lock:
            if entity_type == "user":
                window = self._users.get(entity_id)
            elif entity_type == "merchant":
                window = self._merchants.get(entity_id)
            elif entity_type == "city":
                window = self._cities.get(entity_id)
            else:
                return {}

            if window is None:
                return {"n_events": 0, "fraud_rate": self._baseline,
                        "fraud_count": 0}
            return {
                "n_events": window.n_events,
                "fraud_rate": round(window.fraud_rate, 6),
                "fraud_count": window.fraud_count,
            }

    def get_global_stats(self) -> dict:
        """Get global tracker statistics."""
        with self._lock:
            return {
                "total_events": self._total_events,
                "total_fraud": self._total_fraud,
                "global_fraud_rate": round(
                    self._total_fraud / max(self._total_events, 1), 6),
                "n_users": len(self._users),
                "n_merchants": len(self._merchants),
                "n_cities": len(self._cities),
                "window_size": self._window_size,
                "min_events": self._min_events,
            }

    def seed_from_training(self, user_id: str, merchant_id: str,
                           city_id: str, is_fraud: bool,
                           weight: float = 1.0) -> None:
        """Seed the tracker with training data for warm-start.

        weight controls how many historical events to simulate per entity.
        This is used to avoid cold-start at deployment.
        """
        t = time.time() - 86400  # backdate by 1 day
        with self._lock:
            for _ in range(max(1, int(weight))):
                if user_id:
                    self._users[user_id].add(is_fraud, t)
                if merchant_id:
                    self._merchants[merchant_id].add(is_fraud, t)
                if city_id:
                    self._cities[city_id].add(is_fraud, t)
                t -= 60  # space events 1 minute apart
                self._total_events += 1
                if is_fraud:
                    self._total_fraud += 1

    def clear(self) -> None:
        """Reset all state."""
        with self._lock:
            self._users.clear()
            self._merchants.clear()
            self._cities.clear()
            self._total_events = 0
            self._total_fraud = 0


# Singleton instance (shared across risk engine requests)
_tracker: Optional[EntityFraudRateTracker] = None


def get_tracker() -> EntityFraudRateTracker:
    """Get the singleton entity fraud rate tracker."""
    global _tracker
    if _tracker is None:
        _tracker = EntityFraudRateTracker()
    return _tracker
