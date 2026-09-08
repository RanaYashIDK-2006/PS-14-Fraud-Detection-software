#!/usr/bin/env python3
"""Security Tests for PS14 Fraud Detection System.

These tests verify that security controls are properly implemented
and functioning as expected.

Usage:
    python scripts/security_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)


class SecurityTest:
    """Base class for security tests."""
    
    def __init__(self, base_url: str = "http://127.0.0.1"):
        self.base_url = base_url
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
    
    def run_all(self) -> bool:
        """Run all tests and return True if all passed."""
        raise NotImplementedError


class CORSRegistrationTest(SecurityTest):
    """Test CORS configuration."""
    
    def run_all(self) -> bool:
        print("\n[CORS Tests]")
        
        # Test 1: Wildcard should be rejected
        try:
            resp = self.client.options(
                f"{self.base_url}:8001/auth/login",
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
            self.log("Reject wildcard origin", False, str(e))
        
        # Test 2: Trusted origin should be allowed
        try:
            resp = self.client.options(
                f"{self.base_url}:8001/auth/login",
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
            self.log("Allow trusted origin", False, str(e))
        
        return all(r["passed"] for r in self.results)


class RateLimitTest(SecurityTest):
    """Test rate limiting."""
    
    def run_all(self) -> bool:
        print("\n[Rate Limit Tests]")
        
        # Test: Should block after multiple attempts
        blocked = False
        for i in range(10):
            try:
                resp = self.client.post(
                    f"{self.base_url}:8001/auth/login",
                    json={"email": "test@test.com", "password": f"wrong{i}"}
                )
                if resp.status_code == 429:
                    blocked = True
                    self.log(
                        "Block after rate limit exceeded",
                        True,
                        f"Blocked on attempt {i+1}"
                    )
                    break
            except Exception as e:
                self.log("Rate limit check", False, str(e))
                break
        
        if not blocked:
            self.log("Block after rate limit exceeded", False, "Never blocked")
        
        return all(r["passed"] for r in self.results)


class SecurityHeadersTest(SecurityTest):
    """Test security headers (browser-standard)."""
    
    def run_all(self) -> bool:
        print("\n[Security Headers Tests]")
        
        try:
            resp = self.client.get(f"{self.base_url}:8001/health")
            
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


def run_all_tests(base_url: str = "http://127.0.0.1") -> bool:
    """Run all security tests."""
    print("=" * 60)
    print("PS14 SECURITY TESTS")
    print("=" * 60)
    
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
    
    passed = sum(1 for r in all_results if r["passed"])
    failed = sum(1 for r in all_results if not r["passed"])
    
    print(f"Total: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✅ ALL SECURITY TESTS PASSED")
    else:
        print(f"\n❌ {failed} SECURITY TEST(S) FAILED")
    
    print("=" * 60)
    
    return failed == 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PS14 Security Tests")
    parser.add_argument("--url", default="http://127.0.0.1", help="Base URL")
    args = parser.parse_args()
    
    success = run_all_tests(args.url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()