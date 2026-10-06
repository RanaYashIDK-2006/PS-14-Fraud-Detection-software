"""Shared security middleware for all PS-14 services.

Import ``apply_security_middleware(app)`` in each service's main.py to
uniformly add CORS, rate limiting, security headers, and origin validation.
"""

from __future__ import annotations

import math

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.middleware.rate_limiter import RateLimiter, rate_limit_middleware
from src.middleware.security_headers import SecurityHeadersMiddleware, OriginValidationMiddleware
from src.settings import settings


def _json_safe(obj):
    """Make a pydantic error payload JSON-serializable (phase 4 finding F3).

    FastAPI's default RequestValidationError handler embeds the raw input in
    the 422 detail; a NaN/±Inf float (or a ValueError in ``ctx``) then crashes
    starlette's JSONResponse (allow_nan=False) and the request surfaces as a
    generic 500 instead of the intended 422. Sanitize non-finite floats to
    their repr strings and unknown objects to repr so the rejection survives.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return repr(obj)


async def _validation_error_handler(request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})


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

    # 5. Validation errors must stay 422 even when the offending input is a
    #    non-finite float (F3: NaN in a feature payload used to 500).
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
