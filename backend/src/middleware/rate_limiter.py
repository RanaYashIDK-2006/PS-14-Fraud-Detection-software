"""Rate limiting middleware for PS14 services — DEVELOPMENT ONLY.

Provides in-memory sliding window rate limiting to prevent brute-force
attacks on authentication endpoints.

LIMITATIONS (do NOT use in production without a shared backend):
  - State is process-local (not shared across workers/replicas)
  - Different workers have independent rate-limit counters
  - Horizontal scaling defeats rate limiting entirely
  - No persistence across restarts

For production, use Redis-backed rate limiting or a reverse proxy
(Caddy/nginx) rate limiter.

Usage:
    from src.middleware.rate_limiter import RateLimiter, rate_limit_middleware
    
    app = FastAPI()
    rate_limiter = RateLimiter()
    app.add_middleware(rate_limit_middleware, limiter=rate_limiter)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests: int = 5  # Maximum requests per window
    window_seconds: int = 60  # Time window in seconds
    block_duration_seconds: int = 300  # Block duration after exceeding limit (5 minutes)


@dataclass
class ClientInfo:
    """Tracks request history for a client."""
    requests: list[float] = field(default_factory=list)
    blocked_until: float = 0.0
    
    def is_blocked(self, now: float) -> bool:
        """Check if client is currently blocked."""
        return now < self.blocked_until
    
    def add_request(self, now: float, config: RateLimitConfig) -> bool:
        """Add a request and check if rate limit is exceeded.
        
        Returns True if request should be allowed, False if blocked.
        """
        # Check if blocked
        if self.is_blocked(now):
            return False
        
        # Remove old requests outside the window
        cutoff = now - config.window_seconds
        self.requests = [t for t in self.requests if t > cutoff]
        
        # Check if limit exceeded
        if len(self.requests) >= config.max_requests:
            self.blocked_until = now + config.block_duration_seconds
            return False
        
        # Allow and record
        self.requests.append(now)
        return True


class RateLimiter:
    """In-memory rate limiter with sliding window."""
    
    def __init__(self):
        self.clients: dict[str, ClientInfo] = defaultdict(ClientInfo)
        self._config = RateLimitConfig()
        
        # Per-endpoint configuration
        self.endpoint_configs: dict[str, RateLimitConfig] = {
            "/auth/login": RateLimitConfig(max_requests=5, window_seconds=60, block_duration_seconds=300),
            "/auth/register": RateLimitConfig(max_requests=3, window_seconds=300, block_duration_seconds=600),
            "/admin/login": RateLimitConfig(max_requests=5, window_seconds=60, block_duration_seconds=300),
            "/internal/evaluate": RateLimitConfig(max_requests=500, window_seconds=60, block_duration_seconds=60),
            "/internal/attribution": RateLimitConfig(max_requests=30, window_seconds=60, block_duration_seconds=300),
        }
    
    def get_config(self, path: str) -> RateLimitConfig:
        """Get rate limit config for a specific endpoint."""
        # Check for exact match first
        if path in self.endpoint_configs:
            return self.endpoint_configs[path]
        
        # Check for prefix match
        for pattern, config in self.endpoint_configs.items():
            if path.startswith(pattern):
                return config
        
        # Default config
        return self._config
    
    def check_rate_limit(self, client_id: str, path: str) -> tuple[bool, dict]:
        """Check if a request should be allowed.
        
        Returns:
            (allowed, info_dict)
        """
        now = time.time()
        config = self.get_config(path)
        client = self.clients[client_id]
        
        allowed = client.add_request(now, config)
        
        remaining = max(0, config.max_requests - len(client.requests))
        reset_at = int(now + config.window_seconds)
        
        info = {
            "allowed": allowed,
            "limit": config.max_requests,
            "remaining": remaining,
            "reset_at": reset_at,
            "retry_after": int(client.blocked_until - now) if not allowed else 0,
        }
        
        return allowed, info
    
    def cleanup_old_entries(self, max_age_seconds: int = 3600):
        """Remove entries older than max_age_seconds to prevent memory leaks."""
        now = time.time()
        cutoff = now - max_age_seconds
        
        to_remove = []
        for client_id, client in self.clients.items():
            if not client.requests or max(client.requests) < cutoff:
                to_remove.append(client_id)
        
        for client_id in to_remove:
            del self.clients[client_id]


async def rate_limit_middleware(
    request: Request,
    call_next: RequestResponseEndpoint,
    limiter: RateLimiter,
) -> Response:
    """Middleware that applies rate limiting to specified endpoints."""
    
    # Skip rate limiting in development mode (tests need rapid requests)
    import os
    if os.environ.get('PS14_MODE', 'production').lower() != 'production':
        return await call_next(request)

    # Apply to auth + admin + internal endpoints, exclude OPTIONS (CORS preflight)
    path = request.url.path
    if request.method == "OPTIONS" or not any(path.startswith(p) for p in ["/auth/", "/admin/", "/internal/"]):
        return await call_next(request)
    
    # Get client identifier: IP-only to prevent User-Agent rotation bypass
    # Using user_agent in the key allows attackers to rotate headers and
    # bypass rate limits entirely. IP is the only stable client identifier.
    client_id = request.client.host if request.client else "unknown"
    
    # Check rate limit
    allowed, info = limiter.check_rate_limit(client_id, path)
    
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded. Please try again later.",
                "retry_after": info["retry_after"],
            },
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info["reset_at"]),
                "Retry-After": str(info["retry_after"]),
            },
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(info["reset_at"])
    
    return response


def create_rate_limiter() -> RateLimiter:
    """Factory function to create a rate limiter instance."""
    return RateLimiter()