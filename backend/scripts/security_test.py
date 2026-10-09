#!/usr/bin/env python3
"""Security Tests for PS14 Fraud Detection System.

These tests verify that security controls are properly implemented
and functioning as expected.

Usage:
    python scripts/security_test.py [--url http://host:port]

Exit codes:
    0 - every security assertion evaluated and passed
    1 - at least one security assertion FAILED
    2 - no assertion failed, but the identity service was unreachable, so
        live checks are reported as ENVIRONMENTAL (security NOT verified)
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)


class SecurityTest:
    """Base class for security tests."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url.rstrip("/")
        # Disable SSL verification for self-signed certs (dev/self-signed)
        self.client = httpx.Client(timeout=5.0, verify=False)
        self.results: list[dict] = []
    
    def log(self, test: str, passed: bool, details: str = ""):
        """Log a test result."""
        status = "PASS" if passed else "FAIL"
        self.results.append({"test": test, "passed": passed, "details": details})
        print(f"  [{status}] {test}")
        if details and not passed:
            print(f"         {details}")

    def log_env(self, test: str, details: str):
        """Record a check that could NOT be evaluated (service unreachable).

        Kept separate from log(): an unreachable service is an
        environmental condition, never a passed or failed security
        assertion.
        """
        self.results.append({"test": test, "passed": None, "details": details, "env": True})
        print(f"  [ENVIRONMENTAL] {test}")
        print(f"         {details}")
    
    def run_all(self) -> bool:
        """Run all tests and return True if all passed."""
        raise NotImplementedError


def _is_unreachable(exc: Exception) -> bool:
    """Transport-level failure = service unavailable, not a failed assertion."""
    return isinstance(exc, httpx.TransportError)


def _rejection_budget() -> int:
    """Attempts needed to observe a 429 from ANY starting lockout state.

    The identity service checks the failed-login lockout BEFORE recording
    the failure, so a cold service needs threshold+1 attempts before the
    429 fires; already-warmed or actively-blocked state trips on attempt 1.
    The middleware limiter (production mode only) rejects at its own
    /auth/login max_requests. Both limits are read from source so the
    budget stays correct if they drift; a formatting change fails this
    test loudly rather than silently shrinking the budget.
    """
    ident = (ROOT / "backend" / "src" / "identity_service" / "main.py").read_text(encoding="utf-8")
    lockout = int(re.search(r"_LOGIN_LOCKOUT_THRESHOLD\s*=\s*(\d+)", ident).group(1))
    rl = (ROOT / "backend" / "src" / "middleware" / "rate_limiter.py").read_text(encoding="utf-8")
    login = int(re.search(r'"/auth/login":\s*RateLimitConfig\(max_requests=(\d+)', rl).group(1))
    return max(lockout, login) + 1


class CORSRegistrationTest(SecurityTest):
    """Test CORS configuration."""
    
    def run_all(self) -> bool:
        print("\n[CORS Tests]")
        
        # Test 1: Wildcard should be rejected
        try:
            resp = self.client.options(
                f"{self.base_url}/auth/login",
                headers={
                    "Origin": "https://evil-website.com",
                    "Access-Control-Request-Method": "POST",
                }
            )
            allowed = resp.headers.get("access-control-allow-origin", "")
            self.log(
                "Reject wildcard origin",
                allowed not in ["*", "https://evil-website.com"],
                f"Got: {allowed!r}"
            )
        except Exception as e:
            if _is_unreachable(e):
                self.log_env("Reject wildcard origin", f"identity service unreachable: {e}")
            else:
                self.log("Reject wildcard origin", False, str(e))
        
        # Test 2: Trusted origin should be allowed
        try:
            resp = self.client.options(
                f"{self.base_url}/auth/login",
                headers={
                    "Origin": "http://127.0.0.1:8000",
                    "Access-Control-Request-Method": "POST",
                }
            )
            allowed = resp.headers.get("access-control-allow-origin", "")
            self.log(
                "Allow trusted origin",
                allowed == "http://127.0.0.1:8000",
                f"Got: {allowed!r}"
            )
        except Exception as e:
            if _is_unreachable(e):
                self.log_env("Allow trusted origin", f"identity service unreachable: {e}")
            else:
                self.log("Allow trusted origin", False, str(e))
        
        return all(r["passed"] for r in self.results)


class RateLimitTest(SecurityTest):
    """Test rate limiting."""
    
    def run_all(self) -> bool:
        print("\n[Rate Limit Tests]")
        
        # The service must reject a brute-force client with 429 within this
        # budget, regardless of lockout state left by previous runs: cold
        # state needs threshold+1 attempts (the check runs before the
        # failure is recorded), warm/blocked state rejects on attempt 1.
        budget = _rejection_budget()
        blocked = False
        unreachable = False
        for i in range(budget):
            try:
                resp = self.client.post(
                    f"{self.base_url}/auth/login",
                    json={"email": "test@test.com", "password": f"wrong{i}"}
                )
                if resp.status_code == 429:
                    blocked = True
                    self.log(
                        "Block after rate limit exceeded",
                        True,
                        f"Blocked on attempt {i+1} of {budget}"
                    )
                    # Visible on PASS too: records which starting state the
                    # run exercised (attempt 1 = already warm/blocked,
                    # attempt {lockout+1} = cold service).
                    print(f"         (429 after {i + 1} of {budget} attempts)")
                    break
            except Exception as e:
                if _is_unreachable(e):
                    self.log_env("Block after rate limit exceeded",
                                 f"identity service unreachable: {e}")
                    unreachable = True
                else:
                    self.log("Block after rate limit exceeded", False, str(e))
                break
        
        if not blocked and not unreachable:
            self.log("Block after rate limit exceeded", False,
                     f"Never blocked within {budget} attempts")
        
        return all(r["passed"] for r in self.results)


class SecurityHeadersTest(SecurityTest):
    """Test security headers (browser-standard)."""
    
    def run_all(self) -> bool:
        print("\n[Security Headers Tests]")
        
        try:
            resp = self.client.get(f"{self.base_url}/health")
            
            # Core headers
            required_headers = {
                "x-content-type-options": "nosniff",
                "x-frame-options": "DENY",
                "referrer-policy": "strict-origin-when-cross-origin",
            }
            
            for header, expected in required_headers.items():
                actual = resp.headers.get(header, "MISSING")
                self.log(
                    f"Header {header}",
                    actual == expected,
                    f"Expected: {expected}, Got: {actual}"
                )
            
            # CSP must exist and contain key directives
            csp = resp.headers.get("content-security-policy", "")
            self.log(
                "CSP header present",
                bool(csp),
                f"Got: {csp[:80]}..." if csp else "MISSING"
            )
            self.log(
                "CSP frame-ancestors",
                "frame-ancestors" in csp,
                "frame-ancestors not in CSP"
            )
            self.log(
                "CSP upgrade-insecure-requests",
                "upgrade-insecure-requests" in csp,
                "upgrade-insecure-requests not in CSP"
            )
            self.log(
                "CSP object-src none",
                "object-src 'none'" in csp,
                "object-src not restricted"
            )
            
            # HSTS
            hsts = resp.headers.get("strict-transport-security", "")
            self.log(
                "HSTS header",
                "max-age=" in hsts,
                f"Got: {hsts}" if hsts else "MISSING"
            )
            
            # Permissions policy
            pp = resp.headers.get("permissions-policy", "")
            self.log(
                "Permissions-Policy",
                bool(pp) and "camera=()" in pp,
                f"Got: {pp[:60]}..." if pp else "MISSING"
            )
            
            # X-XSS-Protection is DEPRECATED — should NOT be present
            # (it can cause vulnerabilities in some browsers)
            xssp = resp.headers.get("x-xss-protection", "")
            self.log(
                "X-XSS-Protection absent (deprecated)",
                not xssp,
                f"Should be absent, got: {xssp}" if xssp else "Correctly absent"
            )
        except Exception as e:
            if _is_unreachable(e):
                self.log_env("Security headers (10 checks)",
                             f"identity service unreachable: {e} — checks not evaluated")
            else:
                self.log("Security headers check", False, str(e))
        
        return all(r["passed"] for r in self.results)


class KeySeparationTest(SecurityTest):
    """Test that security keys are properly separated."""
    
    def run_all(self) -> bool:
        print("\n[Key Separation Tests]")
        
        try:
            from src.settings import settings, load_dotenv_and_patch
            load_dotenv_and_patch()
            
            jwt = settings.jwt_secret
            pii = settings.pii_encryption_key
            exp = settings.export_signing_key_str
            
            self.log(
                "JWT and PII keys different",
                jwt != pii,
                f"JWT: {jwt[:20]}..., PII: {pii[:20]}..."
            )
            
            self.log(
                "JWT and Export keys different",
                jwt != exp,
                f"JWT: {jwt[:20]}..., Export: {exp[:20]}..."
            )
            
            self.log(
                "PII and Export keys different",
                pii != exp,
                f"PII: {pii[:20]}..., Export: {exp[:20]}..."
            )
        except Exception as e:
            self.log("Key separation check", False, str(e))
        
        return all(r["passed"] for r in self.results)


def run_all_tests(base_url: str = "http://127.0.0.1:8001") -> int:
    """Run all security tests. Returns the process exit code:

    0 = all assertions evaluated and passed;
    1 = at least one security assertion failed;
    2 = nothing failed but the identity service was unreachable (live
        checks reported as ENVIRONMENTAL, security NOT verified).
    """
    base_url = base_url.rstrip("/")
    print("=" * 60)
    print("PS14 SECURITY TESTS")
    print("=" * 60)
    
    # Readiness probe: establish the precondition for the live checks and
    # report unavailability as an environmental condition up front.
    try:
        httpx.get(f"{base_url}/health", timeout=5.0)
    except httpx.TransportError as e:
        print(f"\nWARNING: identity service unreachable at {base_url} ({e})")
        print("         Live checks will be reported as ENVIRONMENTAL, "
              "not as security failures.")
    
    all_results = []
    
    # Run test suites
    test_suites = [
        CORSRegistrationTest(base_url),
        RateLimitTest(base_url),
        SecurityHeadersTest(base_url),
        KeySeparationTest(base_url),
    ]
    
    for suite in test_suites:
        suite.run_all()
        all_results.extend(suite.results)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in all_results if r["passed"] is True)
    failed = sum(1 for r in all_results if r["passed"] is False)
    env = sum(1 for r in all_results if r.get("env"))
    
    print(f"Total: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if env:
        print(f"Environmental (not evaluated): {env}")
    
    if failed:
        print(f"\n❌ {failed} SECURITY TEST(S) FAILED")
    elif env:
        print(f"\n⚠️ ENVIRONMENTAL: {env} live check(s) not evaluated — "
              "identity service unreachable, security NOT verified")
    else:
        print("\n✅ ALL SECURITY TESTS PASSED")
    
    print("=" * 60)
    
    if failed:
        return 1
    return 2 if env else 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PS14 Security Tests")
    parser.add_argument("--url", default="http://127.0.0.1:8001",
                        help="Identity service origin (default: http://127.0.0.1:8001)")
    args = parser.parse_args()
    
    sys.exit(run_all_tests(args.url))


if __name__ == "__main__":
    main()