"""Redis-backed rate limiting using ZSET-based sliding window counters.

Provides two levels of rate limiting:
  1. Per-user: limits requests per user per time window
  2. Global: limits total requests across all users per time window

Algorithm: Sliding Window Log (ZSET-based)
  - Each request is logged as a ZSET member with timestamp as score
  - On check: count members within the window, evict old ones
  - O(log N) per check where N = requests in window

Configuration:
  - per_user_limit: max requests per user per window (default: 100)
  - global_limit: max total requests per window (default: 10000)
  - window_seconds: sliding window size (default: 60s)

Usage:
    limiter = RedisRateLimiter(redis_client)
    allowed, info = limiter.check("user_123")
    if not allowed:
        return 429, {"retry_after": info["retry_after_seconds"]}

    # Decorator for FastAPI
    @app.post("/score")
    @rate_limit(per_user=100, global_limit=10000, window=60)
    async def score(req: ScoreRequest): ...
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable

import redis


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    user_count: int       # requests by this user in window
    global_count: int     # total requests in window
    user_limit: int       # user limit
    global_limit: int     # global limit
    retry_after_seconds: float  # seconds until oldest request expires
    window_seconds: int   # window size
    user_remaining: int   # remaining requests for user
    global_remaining: int # remaining requests globally

    def to_headers(self) -> dict[str, str]:
        """HTTP headers for 429 response (RFC 6585 compliant)."""
        return {
            "X-RateLimit-Limit": str(self.user_limit),
            "X-RateLimit-Remaining": str(max(0, self.user_remaining)),
            "X-RateLimit-Reset": str(int(time.time() + self.retry_after_seconds)),
            "Retry-After": str(max(1, int(self.retry_after_seconds))),
        }

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "user_count": self.user_count,
            "global_count": self.global_count,
            "user_limit": self.user_limit,
            "global_limit": self.global_limit,
            "retry_after_seconds": round(self.retry_after_seconds, 1),
            "user_remaining": self.user_remaining,
            "global_remaining": self.global_remaining,
        }


class RedisRateLimiter:
    """Redis-backed sliding window rate limiter.

    Uses Redis Sorted Sets (ZSETs) for efficient sliding window counting:
      - Key pattern: rl:user:{user_id} (per-user window)
      - Key pattern: rl:global (global window)
      - Score: request timestamp (float)
      - Member: unique request ID (timestamp + random)

    Eviction: old entries are removed on each check (lazy cleanup).
    TTL: keys expire after 2x window_seconds (safety net).

    Thread-safe: all operations use Redis atomic commands.
    """

    def __init__(
        self,
        r: redis.Redis,
        per_user_limit: int = 100,
        global_limit: int = 10000,
        window_seconds: int = 60,
        prefix: str = "rl:",
    ):
        self.r = r
        self.per_user_limit = per_user_limit
        self.global_limit = global_limit
        self.window_seconds = window_seconds
        self.prefix = prefix

        # Stats
        self._total_checked = 0
        self._total_rejected = 0
        self._lock = threading.Lock()

    def _user_key(self, user_id: str) -> str:
        return f"{self.prefix}user:{user_id}"

    def _global_key(self) -> str:
        return f"{self.prefix}global"

    def check(self, user_id: str = "", record: bool = True) -> RateLimitResult:
        """Check if a request is allowed under rate limits.

        Args:
            user_id: User identifier (empty = global-only check)
            record: If True, also record this request (consume quota)

        Returns:
            RateLimitResult with allowed flag and counters
        """
        import uuid
        now = time.time()
        window_start = now - self.window_seconds
        request_id = f"{now:.6f}:{uuid.uuid4().hex[:8]}"

        # Use pipeline for atomicity
        pipe = self.r.pipeline()

        # --- Per-user check ---
        user_key = self._user_key(user_id) if user_id else None
        if user_key:
            # Evict old entries
            pipe.zremrangebyscore(user_key, 0, window_start)
            # Count current entries
            pipe.zcard(user_key)
            # Set TTL (2x window as safety net)
            pipe.expire(user_key, self.window_seconds * 2)

        # --- Global check ---
        global_key = self._global_key()
        pipe.zremrangebyscore(global_key, 0, window_start)
        pipe.zcard(global_key)
        pipe.expire(global_key, self.window_seconds * 2)

        results = pipe.execute()

        # Parse results
        idx = 0
        user_count = 0
        if user_key:
            user_count = results[idx + 1]  # zcard
            idx += 3

        global_count = results[idx + 1]  # zcard
        idx += 3

        # Determine if allowed
        user_ok = not user_key or user_count < self.per_user_limit
        global_ok = global_count < self.global_limit
        allowed = user_ok and global_ok

        # Calculate retry_after
        retry_after = 0.0
        if not allowed and user_key:
            oldest = self.r.zrange(user_key, 0, 0, withscores=True)
            if oldest:
                retry_after = max(0, (oldest[0][1] + self.window_seconds) - now)
        elif not global_ok:
            oldest = self.r.zrange(global_key, 0, 0, withscores=True)
            if oldest:
                retry_after = max(0, (oldest[0][1] + self.window_seconds) - now)

        # Record this request if allowed
        if allowed and record:
            pipe2 = self.r.pipeline()
            if user_key:
                pipe2.zadd(user_key, {request_id: now})
                pipe2.expire(user_key, self.window_seconds * 2)
            pipe2.zadd(global_key, {request_id: now})
            pipe2.expire(global_key, self.window_seconds * 2)
            pipe2.execute()

        # Update stats
        with self._lock:
            self._total_checked += 1
            if not allowed:
                self._total_rejected += 1

        return RateLimitResult(
            allowed=allowed,
            user_count=user_count + (1 if allowed and record else 0),
            global_count=global_count + (1 if allowed and record else 0),
            user_limit=self.per_user_limit,
            global_limit=self.global_limit,
            retry_after_seconds=retry_after,
            window_seconds=self.window_seconds,
            user_remaining=max(0, self.per_user_limit - user_count - (1 if allowed else 0)),
            global_remaining=max(0, self.global_limit - global_count - (1 if allowed else 0)),
        )

    def check_only(self, user_id: str = "") -> RateLimitResult:
        """Check rate limit without recording (dry run)."""
        return self.check(user_id=user_id, record=False)

    def reset_user(self, user_id: str) -> bool:
        """Reset a user's rate limit window."""
        return bool(self.r.delete(self._user_key(user_id)))

    def reset_global(self) -> bool:
        """Reset the global rate limit window."""
        return bool(self.r.delete(self._global_key()))

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            total = self._total_checked
            rejected = self._total_rejected

        # Count active user windows
        try:
            pattern = f"{self.prefix}user:*"
            cursor = 0
            active_users = 0
            while True:
                cursor, keys = self.r.scan(cursor, match=pattern, count=1000)
                active_users += len(keys)
                if cursor == 0:
                    break
        except Exception:
            active_users = -1

        # Global window count
        global_key = self._global_key()
        global_count = 0
        try:
            window_start = time.time() - self.window_seconds
            self.r.zremrangebyscore(global_key, 0, window_start)
            global_count = self.r.zcard(global_key)
        except Exception:
            pass

        return {
            "per_user_limit": self.per_user_limit,
            "global_limit": self.global_limit,
            "window_seconds": self.window_seconds,
            "total_checked": total,
            "total_rejected": rejected,
            "rejection_rate": round(rejected / max(total, 1), 4),
            "active_user_windows": active_users,
            "global_window_count": global_count,
        }


# --- FastAPI Decorator ---

def rate_limit(
    per_user: int = 100,
    global_limit: int = 10000,
    window: int = 60,
    get_user_id: Callable = None,
):
    """FastAPI dependency/decorator for rate limiting.

    Usage:
        @app.post("/score")
        @rate_limit(per_user=100, global_limit=10000, window=60)
        async def score(req: ScoreRequest): ...

        # With custom user ID extraction:
        @app.post("/score")
        @rate_limit(per_user=50, get_user_id=lambda req: req.user_id)
        async def score(req: ScoreRequest): ...
    """
    def decorator(func):
        from functools import wraps
        from fastapi import Request
        from starlette.responses import JSONResponse

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the request object
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if request is None:
                for v in kwargs.values():
                    if isinstance(v, Request):
                        request = v
                        break

            if request is None:
                return await func(*args, **kwargs)

            # Get rate limiter from app state
            limiter: Optional[RedisRateLimiter] = getattr(
                request.app.state, "rate_limiter", None
            )
            if limiter is None:
                return await func(*args, **kwargs)

            # Extract user ID
            user_id = ""
            if get_user_id:
                # Try to get from the first non-Request arg (likely the body)
                for arg in args:
                    if not isinstance(arg, Request):
                        try:
                            user_id = str(get_user_id(arg))
                        except Exception:
                            pass
                        break
            if not user_id:
                user_id = request.client.host if request.client else "anonymous"

            result = limiter.check(user_id)
            if not result.allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": "rate_limit_exceeded", **result.to_dict()},
                    headers=result.to_headers(),
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator
