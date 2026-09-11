"""Session store with Redis-first, SQLite-fallback architecture.

Provides multi-worker-safe session storage:
  - Primary: Redis (instant cross-worker sync, TTL-based expiry)
  - Fallback: SQLite WAL (when Redis is unavailable)

Thread-safe: Redis connection pool handles concurrent access.
SQLite fallback uses per-thread connections with WAL mode.

Usage:
    from src.session_store import SessionStore
    store = SessionStore(db_path, session_type="admin")
    store.put(jti, data, expires_at)
    data = store.get(jti)
    store.revoke(jti)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_local = threading.local()


def _get_conn(db_path: Path) -> sqlite3.Connection:
    """Per-thread connection with WAL mode."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            jti TEXT PRIMARY KEY,
            session_type TEXT NOT NULL,
            data TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            revoked INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_type
        ON sessions(session_type, revoked)
    """)
    conn.commit()
    return conn


class SessionStore:
    """Thread-safe, multi-worker-safe session store.

    Uses Redis when available (instant cross-worker sync, automatic TTL expiry).
    Falls back to SQLite WAL when Redis is unavailable.
    """

    def __init__(self, db_path: Path, session_type: str = "admin"):
        self.db_path = Path(db_path)
        # sqlite3.connect cannot create the file if its directory is missing
        # (e.g. a fresh DB_DIR volume in Docker) — make it first.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_type = session_type
        self._ensure_table()
        # Lazy-init Redis via centralized connection manager
        self._redis = None
        self._redis_tried = False
        self._redis_prefix = f"ps14:session:{session_type}:"

    def _ensure_table(self) -> None:
        conn = _get_conn(self.db_path)
        try:
            conn.close()
        except Exception:
            pass

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

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(_local, "connections"):
            _local.connections = {}
        key = str(self.db_path)
        if key not in _local.connections:
            _local.connections[key] = _get_conn(self.db_path)
        return _local.connections[key]

    def put(self, jti: str, data: dict, expires_at: datetime) -> None:
        """Store a session."""
        self._ensure_redis()

        # Write to Redis if available
        if self._redis:
            try:
                key = f"{self._redis_prefix}{jti}"
                ttl = max(1, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
                pipe = self._redis.pipeline()
                pipe.set(key, json.dumps(data, default=str), ex=ttl)
                pipe.hset(f"{self._redis_prefix}idx", jti, expires_at.isoformat())
                pipe.execute()
                return  # Redis-only write
            except Exception:
                pass  # Fall through to SQLite

        # SQLite fallback
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO sessions
               (jti, session_type, data, expires_at, created_at, revoked)
               VALUES (?, ?, ?, ?, ?, 0)""",
            (
                jti,
                self.session_type,
                json.dumps(data, default=str),
                expires_at.isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    def get(self, jti: str) -> dict | None:
        """Retrieve a non-revoked, non-expired session."""
        self._ensure_redis()

        # Try Redis first
        if self._redis:
            try:
                key = f"{self._redis_prefix}{jti}"
                raw = self._redis.get(key)
                if raw is None:
                    return None
                return json.loads(raw)
            except Exception:
                pass  # Fall through to SQLite

        # SQLite fallback
        conn = self._conn()
        row = conn.execute(
            "SELECT data, expires_at, revoked FROM sessions WHERE jti = ? AND session_type = ?",
            (jti, self.session_type),
        ).fetchone()
        if row is None:
            return None
        data_str, expires_str, revoked = row
        if revoked:
            return None
        expires_at = datetime.fromisoformat(expires_str)
        if datetime.now(timezone.utc) > expires_at:
            self.revoke(jti)
            return None
        return json.loads(data_str)

    def revoke(self, jti: str) -> None:
        """Mark a session as revoked."""
        self._ensure_redis()

        # Revoke in Redis
        if self._redis:
            try:
                key = f"{self._redis_prefix}{jti}"
                self._redis.delete(key)
                self._redis.hdel(f"{self._redis_prefix}idx", jti)
                return  # Redis-only revoke
            except Exception:
                pass

        # SQLite fallback
        conn = self._conn()
        conn.execute(
            "UPDATE sessions SET revoked = 1 WHERE jti = ? AND session_type = ?",
            (jti, self.session_type),
        )
        conn.commit()

    def revoke_all(self) -> int:
        """Revoke all sessions of this type. Returns count revoked."""
        self._ensure_redis()

        # Revoke all in Redis
        if self._redis:
            try:
                idx_key = f"{self._redis_prefix}idx"
                all_jtis = self._redis.hgetall(idx_key)
                count = len(all_jtis)
                if count > 0:
                    pipe = self._redis.pipeline()
                    for jti in all_jtis:
                        pipe.delete(f"{self._redis_prefix}{jti}")
                    pipe.delete(idx_key)
                    pipe.execute()
                return count
            except Exception:
                pass

        # SQLite fallback
        conn = self._conn()
        cursor = conn.execute(
            "UPDATE sessions SET revoked = 1 WHERE session_type = ? AND revoked = 0",
            (self.session_type,),
        )
        conn.commit()
        return cursor.rowcount

    def cleanup_expired(self) -> int:
        """Delete expired sessions. Returns count deleted."""
        self._ensure_redis()

        # Redis: TTL handles expiry automatically; clean up index
        if self._redis:
            try:
                idx_key = f"{self._redis_prefix}idx"
                all_jtis = self._redis.hgetall(idx_key)
                count = 0
                now = datetime.now(timezone.utc)
                for jti, expires_str in all_jtis.items():
                    try:
                        expires_at = datetime.fromisoformat(expires_str)
                        if now > expires_at:
                            self._redis.delete(f"{self._redis_prefix}{jti}")
                            self._redis.hdel(idx_key, jti)
                            count += 1
                    except Exception:
                        pass
                return count
            except Exception:
                pass

        # SQLite fallback
        conn = self._conn()
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            "DELETE FROM sessions WHERE expires_at < ? AND session_type = ?",
            (now, self.session_type),
        )
        conn.commit()
        return cursor.rowcount

    def count_active(self) -> int:
        """Count non-revoked, non-expired sessions."""
        self._ensure_redis()

        # Try Redis first
        if self._redis:
            try:
                idx_key = f"{self._redis_prefix}idx"
                all_jtis = self._redis.hgetall(idx_key)
                now = datetime.now(timezone.utc)
                count = 0
                for jti, expires_str in all_jtis.items():
                    try:
                        expires_at = datetime.fromisoformat(expires_str)
                        if now < expires_at:
                            count += 1
                    except Exception:
                        pass
                return count
            except Exception:
                pass

        # SQLite fallback
        conn = self._conn()
        now = datetime.now(timezone.utc).isoformat()
        row = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_type = ? AND revoked = 0 AND expires_at > ?",
            (self.session_type, now),
        ).fetchone()
        return row[0] if row else 0

    @property
    def backend(self) -> str:
        """Return which backend is active."""
        self._ensure_redis()
        return "redis" if self._redis else "sqlite"
