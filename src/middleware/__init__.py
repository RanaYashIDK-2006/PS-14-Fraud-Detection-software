"""Shared security middleware for all PS-14 services.

Import ``apply_security_middleware(app)`` in each service's main.py to
uniformly add CORS, rate limiting, security headers, and origin validation.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.middleware.rate_limiter import RateLimiter, rate_limit_middleware
from src.middleware.security_headers import SecurityHeadersMiddleware, OriginValidationMiddleware
from src.settings import settings


def apply_security_middleware(app: FastAPI) -> None:
    """Wire CORS, rate limiting, security headers, and origin validation onto a FastAPI app."""
    # 1. CORS — restrict to known origins (no wildcards)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Internal-Token",
                        "X-Compliance-Token", "X-Temp-Key", "X-Request-ID"],
        allow_credentials=True,
    )

    # 2. Origin validation — reject cross-origin requests not in allowed list
    # (CORSMiddleware handles preflight; this validates the Origin on actual requests)
    app.add_middleware(OriginValidationMiddleware, allowed_origins=settings.allowed_cors_origins)

    # 3. Rate limiting — brute-force protection on every service
    _rl = RateLimiter()

    class _RLMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            return await rate_limit_middleware(request, call_next, _rl)

    app.add_middleware(_RLMiddleware)

    # 4. Security headers — X-Content-Type-Options, X-Frame-Options, X-Request-ID, etc.
    app.add_middleware(SecurityHeadersMiddleware)
