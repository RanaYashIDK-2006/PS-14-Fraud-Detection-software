"""Per-user velocity tracker — maintains running transaction stats at ingest time.

Computes user_tx_count, user_avg_amt, card_tx_count, and merch_tx_count
so the Altman ensemble receives real velocity values instead of proxy approximations.

Thread-safe: uses threading.Lock for concurrent request safety.
Leakage-safe: all stats are computed from historical events only (shifted by 1).
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class UserStats:
    """Running statistics for a single user."""
    tx_count: int = 0
    total_amount: float = 0.0
    recent_amounts: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class CardStats:
    """Running statistics for a single card."""
    tx_count: int = 0


@dataclass
class MerchantStats:
    """Running statistics for a single merchant."""
    tx_count: int = 0


class UserVelocityTracker:
    """Per-user velocity tracker with Redis backing.

    Uses Redis when available (multi-worker safe), falls back to in-memory.
    All values represent HISTORICAL state (before the current event).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._users: dict[str, UserStats] = defaultdict(UserStats)
        self._cards: dict[str, CardStats] = defaultdict(CardStats)
        self._merchants: dict[str, MerchantStats] = defaultdict(MerchantStats)
        self._redis_prefix = 'ps14:vel:'
        # Lazy-init Redis via centralized connection manager
        self._redis = None
        self._redis_tried = False

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

    def get_velocity(self, user_id: str, card_id: str = "",
                     merchant_id: str = "") -> dict:
        """Get historical velocity stats BEFORE the current event."""
        self._ensure_redis()
        # Try Redis first (multi-worker safe)
        if self._redis:
            try:
                p = self._redis.pipeline()
                p.hgetall(f'{self._redis_prefix}u:{user_id}')
                if card_id:
                    p.hgetall(f'{self._redis_prefix}c:{card_id}')
                else:
                    p = None
                results = p.execute() if p else [{}]
                udata = results[0] if results else {}
                cdata = results[1] if len(results) > 1 else {}
                user_tx_count = int(udata.get('tx_count', 0))
                user_total = float(udata.get('total_amount', 0))
                user_avg_amt = (user_total / user_tx_count) if user_tx_count > 0 else 0.0
                card_tx_count = int(cdata.get('tx_count', 0))
                # Merchant count is global, stored separately
                merch_tx_count = int(self._redis.hget(f'{self._redis_prefix}m:{merchant_id}', 'tx_count') or 0) if merchant_id else 0
                return {
                    'user_tx_count': user_tx_count,
                    'user_avg_amt': round(user_avg_amt, 4),
                    'card_tx_count': card_tx_count,
                    'merch_tx_count': merch_tx_count,
                }
            except Exception:
                pass  # Fall back to in-memory

        # In-memory fallback
        with self._lock:
            user = self._users.get(user_id)
            card = self._cards.get(card_id) if card_id else None
            merch = self._merchants.get(merchant_id) if merchant_id else None
            user_tx_count = user.tx_count if user else 0
            user_avg_amt = (
                (user.total_amount / user.tx_count) if user and user.tx_count > 0
                else 0.0
            )
            card_tx_count = card.tx_count if card else 0
            merch_tx_count = merch.tx_count if merch else 0

        return {
            'user_tx_count': user_tx_count,
            'user_avg_amt': round(user_avg_amt, 4),
            'card_tx_count': card_tx_count,
            'merch_tx_count': merch_tx_count,
        }

    def record_event(self, user_id: str, amount: float, card_id: str = "",
                     merchant_id: str = "") -> None:
        """Record a scored event to update running stats."""
        self._ensure_redis()
        # Write to Redis if available (multi-worker safe)
        if self._redis:
            try:
                p = self._redis.pipeline()
                p.hincrby(f'{self._redis_prefix}u:{user_id}', 'tx_count', 1)
                p.hincrbyfloat(f'{self._redis_prefix}u:{user_id}', 'total_amount', amount)
                p.expire(f'{self._redis_prefix}u:{user_id}', 86400)  # 24h TTL
                if card_id:
                    p.hincrby(f'{self._redis_prefix}c:{card_id}', 'tx_count', 1)
                    p.expire(f'{self._redis_prefix}c:{card_id}', 86400)
                if merchant_id:
                    p.hincrby(f'{self._redis_prefix}m:{merchant_id}', 'tx_count', 1)
                    p.expire(f'{self._redis_prefix}m:{merchant_id}', 86400)
                p.execute()
            except Exception:
                pass  # In-memory still updated below

        # Always update in-memory (works as fallback and for same-process reads)
        with self._lock:
            user = self._users[user_id]
            user.tx_count += 1
            user.total_amount += amount
            user.recent_amounts.append(amount)
            if card_id:
                self._cards[card_id].tx_count += 1
            if merchant_id:
                self._merchants[merchant_id].tx_count += 1

    def seed_from_db(self, events: list[dict]) -> int:
        """Seed the tracker from historical events (called on startup).

        Args:
            events: list of dicts with keys: fraud_id, user_id, amount,
                    card_id, merchant_id, is_fraud

        Returns:
            Number of events seeded.
        """
        count = 0
        for ev in events:
            uid = ev.get("user_id", "")
            if not uid:
                continue
            self.record_event(
                user_id=uid,
                amount=ev.get("amount", 0.0),
                card_id=ev.get("card_id", ""),
                merchant_id=ev.get("merchant_id", ""),
            )
            count += 1
        return count

    @property
    def stats(self) -> dict:
        """Tracker statistics for monitoring."""
        with self._lock:
            return {
                "users": len(self._users),
                "cards": len(self._cards),
                "merchants": len(self._merchants),
            }


# Module-level singleton
_tracker: Optional[UserVelocityTracker] = None


def get_tracker() -> UserVelocityTracker:
    """Get or create the singleton velocity tracker."""
    global _tracker
    if _tracker is None:
        _tracker = UserVelocityTracker()
    return _tracker
