#!/usr/bin/env python3
"""Phase 13: hermetic tests for the security-headers middleware and the
in-memory rate limiter.

Covers the high-risk middleware paths that previously had NO hermetic tests
(they were only checked against a live service by security_test.py, which
reports ENVIRONMENTAL when the service is unreachable):

  1. SecurityHeadersMiddleware — full browser header set, X-Request-ID echo
     vs generation, 413 body-size limit (request never reaches the app), and
     the 500 JSON mapping for an unhandled handler exception (no traceback
     or internal detail disclosed).
  2. _build_csp — development (unsafe-inline) vs production (nonce +
     report-uri) policy construction.
  3. OriginValidationMiddleware — pass without Origin, pass for allow-listed
     Origin, 403 for a disallowed Origin.
  4. RateLimiter pure logic — sliding-window accounting, block + retry_after,
     per-endpoint config (exact and prefix), and stale-entry cleanup.

DELIBERATELY NOT COVERED (deferred, see PHASE_TEST_COVERAGE_CLOSEOUT.md):
the production-mode 429 middleware path in rate_limit_middleware — it only
fires with PS14_MODE=production and is tracked as a deferred finding.

Hermetic: no services, no credentials, no shared state. The body-size limit
is injected via PS14_MAX_BODY_SIZE before import (it is read at import time).

Usage:
    python scripts/security_headers_test.py

Exit codes:
    0 - every assertion passed
    1 - at least one assertion failed
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must be set BEFORE importing the middleware: _MAX_BODY_SIZE is read once at
# import time. A 1024-byte limit keeps the 413 probe tiny and deterministic.
os.environ["PS14_MAX_BODY_SIZE"] = "1024"

passed = 0
failed = 0
errors: list[str] = []


def test(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  \u2713 {name}")
    else:
        failed += 1
        msg = f"  \u2717 {name}"
        if detail:
            msg += f" \u2014 {detail}"
        print(msg)
        errors.append(name)


def run_tests() -> int:
    global passed, failed, errors
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("SECURITY HEADERS + RATE LIMITER MIDDLEWARE TESTS (hermetic)")
    print("=" * 70)

    try:
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from src.middleware.rate_limiter import ClientInfo, RateLimiter, RateLimitConfig
        from src.middleware.security_headers import (
            OriginValidationMiddleware,
            SecurityHeadersMiddleware,
            _build_csp,
        )
    except ImportError as e:
        print(f"\n  FATAL: cannot import middleware: {e}")
        return 1

    # ---- 1. SecurityHeadersMiddleware: header set + request-ID handling ----
    print("\n[1] SecurityHeadersMiddleware \u2014 headers and request IDs")
    app = FastAPI()

    @app.get("/ok")
    async def ok():  # noqa: ANN202
        return {"ok": True}

    @app.post("/echo")
    async def echo():  # noqa: ANN202
        return {"ok": True}

    @app.get("/boom")
    async def boom():  # noqa: ANN202
        raise RuntimeError("internal-secret-detail")

    app.add_middleware(SecurityHeadersMiddleware)
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/ok")
    h = r.headers
    test("200 on healthy route", r.status_code == 200, str(r.status_code))
    test("X-Content-Type-Options: nosniff", h.get("X-Content-Type-Options") == "nosniff")
    test("X-Frame-Options: DENY", h.get("X-Frame-Options") == "DENY")
    test("HSTS with includeSubDomains+preload",
         "max-age=31536000" in h.get("Strict-Transport-Security", "")
         and "includeSubDomains" in h.get("Strict-Transport-Security", "")
         and "preload" in h.get("Strict-Transport-Security", ""))
    csp = h.get("Content-Security-Policy", "")
    test("CSP default-src 'self'", "default-src 'self'" in csp, csp)
    test("CSP frame-ancestors 'none'", "frame-ancestors 'none'" in csp, csp)
    test("Referrer-Policy strict-origin-when-cross-origin",
         h.get("Referrer-Policy") == "strict-origin-when-cross-origin")
    pp = h.get("Permissions-Policy", "")
    test("Permissions-Policy disables camera+geolocation",
         "camera=()" in pp and "geolocation=()" in pp, pp)
    test("Cache-Control no-store for sensitive responses",
         "no-store" in h.get("Cache-Control", ""), h.get("Cache-Control", ""))
    generated_id = h.get("X-Request-ID", "")
    test("X-Request-ID generated when absent", len(generated_id) > 10, generated_id)

    r2 = client.get("/ok", headers={"X-Request-ID": "trace-abc-123"})
    test("X-Request-ID echoed when supplied",
         r2.headers.get("X-Request-ID") == "trace-abc-123",
         r2.headers.get("X-Request-ID", ""))
    test("X-Response-Time present", "X-Response-Time" in r2.headers)

    # ---- 2. 413 body-size limit (DoS guard) ----
    print("\n[2] Body-size limit \u2014 413 before the handler runs")
    big = b"x" * 2048  # > the 1024-byte limit set above
    r3 = client.post("/echo", content=big)
    test("413 when content-length exceeds limit", r3.status_code == 413, str(r3.status_code))
    test("413 body is the documented JSON detail",
         r3.json().get("detail") == "Request body too large", r3.text)
    test("413 carries X-Request-ID", bool(r3.headers.get("X-Request-ID")))
    r4 = client.post("/echo", content=b"small")
    test("small body still accepted", r4.status_code == 200, str(r4.status_code))

    # ---- 3. 500 mapping \u2014 no information disclosure ----
    print("\n[3] Unhandled exception \u2014 500 mapping, no detail leak")
    r5 = client.get("/boom")
    test("500 on unhandled exception", r5.status_code == 500, str(r5.status_code))
    test("500 body is generic JSON detail",
         r5.json().get("detail") == "Internal server error", r5.text)
    test("exception text not disclosed", "internal-secret-detail" not in r5.text, r5.text)
    test("500 carries X-Request-ID", bool(r5.headers.get("X-Request-ID")))

    # ---- 4. CSP: development vs production construction ----
    print("\n[4] _build_csp \u2014 development vs production")
    saved_mode = os.environ.get("PS14_MODE")
    try:
        os.environ["PS14_MODE"] = "development"
        dev_csp = _build_csp("anonce")
        test("dev CSP allows inline scripts",
             "'unsafe-inline'" in dev_csp and "nonce-anonce" not in dev_csp, dev_csp)
        test("dev CSP has no report-uri", "report-uri" not in dev_csp, dev_csp)

        os.environ["PS14_MODE"] = "production"
        prod_csp = _build_csp("anonce")
        test("prod CSP pins nonce", "nonce-anonce" in prod_csp, prod_csp)
        test("prod CSP has no unsafe-inline in script-src",
             "script-src 'self' 'nonce-anonce'" in prod_csp, prod_csp)
        test("prod CSP sets report-uri", "report-uri /csp-report" in prod_csp, prod_csp)
    finally:
        if saved_mode is None:
            os.environ.pop("PS14_MODE", None)
        else:
            os.environ["PS14_MODE"] = saved_mode

    # ---- 5. OriginValidationMiddleware ----
    print("\n[5] OriginValidationMiddleware \u2014 allow-list enforcement")
    oapp = FastAPI()

    @oapp.get("/data")
    async def data():  # noqa: ANN202
        return {"data": 1}

    oapp.add_middleware(
        OriginValidationMiddleware, allowed_origins=["https://app.example"]
    )
    oclient = TestClient(oapp, raise_server_exceptions=False)

    r6 = oclient.get("/data")
    test("no Origin header passes (server-to-server)", r6.status_code == 200, str(r6.status_code))
    r7 = oclient.get("/data", headers={"Origin": "https://app.example"})
    test("allow-listed Origin passes", r7.status_code == 200, str(r7.status_code))
    r8 = oclient.get("/data", headers={"Origin": "https://evil.example"})
    test("disallowed Origin rejected with 403", r8.status_code == 403, str(r8.status_code))
    test("403 body is the documented JSON detail",
         r8.json().get("detail") == "Origin not allowed", r8.text)
    test("403 carries X-Request-ID", bool(r8.headers.get("X-Request-ID")))

    # ---- 6. RateLimiter pure logic (mode-independent) ----
    print("\n[6] RateLimiter \u2014 window, block, config lookup, cleanup")
    limiter = RateLimiter()

    # Config lookup: exact, prefix, default
    login_cfg = limiter.get_config("/auth/login")
    test("exact endpoint config (auth/login 5/60/300)",
         (login_cfg.max_requests, login_cfg.window_seconds, login_cfg.block_duration_seconds)
         == (5, 60, 300))
    batch_cfg = limiter.get_config("/internal/evaluate/batch")
    test("prefix-matched internal config (500/min)", batch_cfg.max_requests == 500,
         str(batch_cfg.max_requests))
    default_cfg = limiter.get_config("/unmapped")
    test("default config for unmapped path",
         (default_cfg.max_requests, default_cfg.window_seconds) == (5, 60))

    # Sliding window + block via explicit clocks (deterministic)
    cfg = RateLimitConfig(max_requests=3, window_seconds=60, block_duration_seconds=300)
    info = ClientInfo()
    t0 = time.time()
    outcomes = [info.add_request(t0 + i, cfg) for i in range(4)]
    test("first max_requests allowed, next denied",
         outcomes == [True, True, True, False], str(outcomes))
    # The 4th (denied) request at t0+3 arms the block until t0+3+300.
    test("block armed after exceeding", info.is_blocked(t0 + 4))
    test("block honours block_duration_seconds",
         info.is_blocked(t0 + 302) and not info.is_blocked(t0 + 304))

    # Window expiry: all requests age out -> allowed again (not blocked now)
    expired = ClientInfo(requests=[t0 - 61, t0 - 62], blocked_until=0.0)
    test("stale requests evicted at check time", expired.add_request(t0, cfg))

    # check_rate_limit end-to-end: allow, then deny with retry_after
    allowed, info1 = limiter.check_rate_limit("phase13-client", "/auth/register")
    test("first request allowed with remaining budget",
         allowed and info1["limit"] == 3 and info1["remaining"] == 2, str(info1))
    limiter.check_rate_limit("phase13-client", "/auth/register")
    limiter.check_rate_limit("phase13-client", "/auth/register")
    denied, info2 = limiter.check_rate_limit("phase13-client", "/auth/register")
    test("4th request denied with retry_after > 0",
         not denied and info2["retry_after"] > 0 and info2["remaining"] == 0, str(info2))

    # cleanup_old_entries: stale removed, fresh and empty kept
    now = time.time()
    limiter.clients["stale"] = ClientInfo(requests=[now - 7200])
    limiter.clients["fresh"] = ClientInfo(requests=[now - 10])
    limiter.clients["empty"] = ClientInfo()
    limiter.cleanup_old_entries(max_age_seconds=3600)
    # cleanup_old_entries evicts clients whose requests are all older than
    # max_age AND clients with no requests at all (documented behaviour).
    test("cleanup evicts stale and empty, keeps fresh",
         "stale" not in limiter.clients
         and "empty" not in limiter.clients
         and "fresh" in limiter.clients
         and "phase13-client" in limiter.clients,
         str(sorted(limiter.clients.keys())))

    # ---- summary ----
    print("\n" + "=" * 70)
    print(f"{passed}/{passed + failed} passed")
    if failed:
        print("FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("ALL SECURITY MIDDLEWARE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
