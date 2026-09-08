"""Security headers middleware for PS14 services.

Browser-standard security headers:
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- Strict-Transport-Security (HSTS) with preload
- Content-Security-Policy with frame-ancestors, upgrade-insecure-requests
- Referrer-Policy: strict-origin-when-cross-origin
- Permissions-Policy: disable unnecessary browser features
- Cache-Control for sensitive endpoints
- Request ID tracking (X-Request-ID)
- Request body size limits

CSP is conditional:
  - Development: relaxed to allow inline scripts/styles (required by self-contained HTML)
  - Production (PS14_MODE=production): strict policy with report-uri
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

_MAX_BODY_SIZE = int(os.environ.get("PS14_MAX_BODY_SIZE", "10485760"))  # 10MB default


def _get_ps14_mode() -> str:
    """Re-read PS14_MODE at call time so .env values are picked up."""
    return os.environ.get("PS14_MODE", "production").lower()

logger = logging.getLogger("ps14.security")


def _build_csp(nonce: str = "") -> str:
    """Build Content-Security-Policy header.

    Development mode: 'unsafe-inline' for convenience.
    Production mode: nonce-based CSP for inline scripts.
    """
    directives = [
        "default-src 'self'",
    ]
    if _get_ps14_mode() == "production" and nonce:
        directives.append(f"script-src 'self' 'nonce-{nonce}'")
        directives.append("style-src 'self' 'unsafe-inline' https://fonts.googleapis.com")
    else:
        # Development: 'self' covers external .js files; no inline scripts needed
        directives.append("script-src 'self'")
        directives.append("style-src 'self' 'unsafe-inline' https://fonts.googleapis.com")

    # Common directives for both modes
    directives.extend([
        # Images: self-hosted + data: for inline SVGs/icons
        "img-src 'self' data: blob:",
        # Fonts: self-hosted only
        "font-src 'self' https://fonts.gstatic.com",
        # API calls: same origin + identity service proxy
        "connect-src 'self' http://127.0.0.1:* http://localhost:*",
        # Prevent framing entirely (replaces X-Frame-Options for CSP-aware browsers)
        "frame-ancestors 'none'",
        # Block object/embed/applet
        "object-src 'none'",
        # Block base tag hijacking
        "base-uri 'self'",
        # Block form action to external origins
        "form-action 'self'",
        # Upgrade mixed content (HTTP resources on HTTPS page)
        "upgrade-insecure-requests",
    ])

    # Production: add report-uri for CSP violation reporting
    if _get_ps14_mode() == "production":
        directives.append("report-uri /csp-report")

    return "; ".join(directives)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds browser-standard security headers, request ID tracking,
    and request body size limits to all responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate or use provided request ID for distributed tracing
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # Enforce request body size limit (DoS prevention)
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > _MAX_BODY_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"},
                    headers={"X-Request-ID": request_id},
                )
        
        # Track request timing for latency metrics
        start_time = time.time()
        
        try:
            response = await call_next(request)
        except Exception as e:
            # Log unhandled exceptions with request ID
            logger.error(f"Unhandled exception: {e} [request_id={request_id}]")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
                headers={"X-Request-ID": request_id},
            )
        
        # Calculate request duration
        duration_ms = round((time.time() - start_time) * 1000, 2)
        
        # Generate nonce for CSP (unique per request)
        import base64 as _b64
        nonce = _b64.b64encode(os.urandom(16)).decode()
        request.state.csp_nonce = nonce

        # Core security headers
        security_headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",

            # Prevent clickjacking (X-Frame-Options for older browsers;
            # frame-ancestors in CSP handles modern browsers)
            "X-Frame-Options": "DENY",

            # HSTS — force HTTPS, include subdomains, enable preload list
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",

            # Content Security Policy (nonce-based in production)
            "Content-Security-Policy": _build_csp(nonce),

            # Referrer — send origin only on cross-origin, full URL on same-origin
            "Referrer-Policy": "strict-origin-when-cross-origin",

            # Permissions — disable unnecessary browser features
            "Permissions-Policy": (
                "camera=(), "
                "microphone=(), "
                "geolocation=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "accelerometer=(), "
                "autoplay=(), "
                "encrypted-media=(), "
                "picture-in-picture=()"
            ),

            # Cache control — sensitive endpoints must not be cached
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            
            # Request tracking for distributed tracing
            "X-Request-ID": request_id,
            
            # Performance metrics
            "X-Response-Time": f"{duration_ms}ms",
        }

        for header, value in security_headers.items():
            response.headers[header] = value

        # Log slow requests (>500ms) for performance monitoring
        if duration_ms > 500:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {duration_ms}ms [request_id={request_id}]"
            )

        return response


class OriginValidationMiddleware(BaseHTTPMiddleware):
    """Validate Origin header on cross-origin requests.
    
    This supplements FastAPI's CORSMiddleware by:
    - Rejecting requests with Origin headers not in the allowed list
    - Allowing requests without Origin (server-to-server, curl, etc.)
    - Logging blocked origins for security monitoring
    
    CORSMiddleware handles browser preflight (OPTIONS); this middleware
    validates the Origin on the actual request.
    """
    
    def __init__(self, app, allowed_origins: list[str]):
        super().__init__(app)
        # Build a set of allowed origins for O(1) lookup
        self._allowed_origins = set(allowed_origins)
        # Also allow same-origin requests (no Origin header)
        self._logger = logging.getLogger("ps14.origin")
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        origin = request.headers.get("origin")
        
        # No Origin header = same-origin request, server-to-server, or non-browser client
        # These are safe to allow
        if not origin:
            return await call_next(request)
        
        # Check if the origin is in the allowed list
        if origin in self._allowed_origins:
            return await call_next(request)
        
        # Log the blocked origin for security monitoring
        self._logger.warning(
            f"Blocked request from unauthorized origin: {origin} "
            f"path={request.url.path} method={request.method}"
        )
        
        # Reject with 403 Forbidden
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        return JSONResponse(
            status_code=403,
            content={"detail": "Origin not allowed"},
            headers={"X-Request-ID": request_id},
        )


def get_security_headers_middleware() -> SecurityHeadersMiddleware:
    """Factory function to create security headers middleware."""
    return SecurityHeadersMiddleware()
