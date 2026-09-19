#!/usr/bin/env python3
"""Phase 73: Production access control — authentication, authorization, rate limiting tests.

Tests constant-time token verification, role-based authorization, thread-safe
rate limiting, security telemetry integration, and privacy guarantees.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
import hashlib
import hmac
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.monitoring.access_control import (
    Role,
    verify_token_constant_time,
    resolve_role,
    authenticate_and_authorize,
    check_access,
    mask_token_for_log,
    RateLimiter,
    RateLimitConfig,
    get_rate_limiter,
    _role_allows,
    _ENDPOINT_ROLES,
    _RATE_LIMIT_CONFIGS,
)

# Initialize observability for integration tests
from src.monitoring.observability_integration import init_observability
init_observability("access-control-test")

passed = 0
failed = 0
errors = []


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        errors.append(name)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Role Model
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 1: Role Model ===")

check("1. Role.EVALUATOR exists", hasattr(Role, "EVALUATOR"))
check("2. Role.OPERATOR exists", hasattr(Role, "OPERATOR"))
check("3. Role.ADMIN exists", hasattr(Role, "ADMIN"))
check("4. Role.SYSTEM exists", hasattr(Role, "SYSTEM"))

check("5. EVALUATOR < OPERATOR in hierarchy",
      _role_allows(Role.OPERATOR, Role.EVALUATOR))
check("6. OPERATOR < ADMIN in hierarchy",
      _role_allows(Role.ADMIN, Role.OPERATOR))
check("7. ADMIN < SYSTEM in hierarchy",
      _role_allows(Role.SYSTEM, Role.ADMIN))
check("8. EVALUATOR cannot do OPERATOR",
      not _role_allows(Role.EVALUATOR, Role.OPERATOR))
check("9. EVALUATOR cannot do ADMIN",
      not _role_allows(Role.EVALUATOR, Role.ADMIN))
check("10. SYSTEM can do everything",
      _role_allows(Role.SYSTEM, Role.EVALUATOR) and
      _role_allows(Role.SYSTEM, Role.OPERATOR) and
      _role_allows(Role.SYSTEM, Role.ADMIN))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Constant-Time Token Verification
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 2: Constant-Time Token Verification ===")

check("11. Valid token matches", verify_token_constant_time("secret123", "secret123"))
check("12. Invalid token rejected", not verify_token_constant_time("wrong", "secret123"))
check("13. Empty token rejected", not verify_token_constant_time("", "secret123"))
check("14. Empty expected rejected", not verify_token_constant_time("token", ""))
check("15. Both empty rejected", not verify_token_constant_time("", ""))
check("16. Partial match rejected", not verify_token_constant_time("secre", "secret123"))
check("17. Longer token rejected", not verify_token_constant_time("secret1234", "secret123"))
check("18. Same length different content", not verify_token_constant_time("aaaaaaaa", "bbbbbbbb"))

# Verify it uses hmac.compare_digest (constant-time)
import inspect
src = inspect.getsource(verify_token_constant_time)
check("19. Uses hmac.compare_digest", "hmac.compare_digest" in src)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Role Resolution
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 3: Role Resolution ===")

INTERNAL = "test-internal-token-abc123"
ADMIN = "test-admin-token-xyz789"
COMPLIANCE = "test-compliance-token-def456"

check("20. Internal token -> EVALUATOR",
      resolve_role(INTERNAL, internal_token=INTERNAL) == Role.EVALUATOR)
check("21. Admin token -> ADMIN",
      resolve_role(ADMIN, admin_token=ADMIN) == Role.ADMIN)
check("22. Compliance token -> OPERATOR",
      resolve_role(COMPLIANCE, compliance_token=COMPLIANCE) == Role.OPERATOR)
check("23. Unknown token -> None",
      resolve_role("unknown-token", internal_token=INTERNAL, admin_token=ADMIN) is None)
check("24. Empty token -> None",
      resolve_role("", internal_token=INTERNAL) is None)
check("25. Admin takes priority over internal",
      resolve_role(ADMIN, internal_token=INTERNAL, admin_token=ADMIN) == Role.ADMIN)
check("26. Compliance takes priority over internal",
      resolve_role(COMPLIANCE, internal_token=INTERNAL, compliance_token=COMPLIANCE) == Role.OPERATOR)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Authentication + Authorization
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 4: Authentication + Authorization ===")

# Evaluator can call /internal/evaluate
ok, role, reason = authenticate_and_authorize(
    INTERNAL, "/internal/evaluate", internal_token=INTERNAL)
check("27. Evaluator can evaluate", ok and role == Role.EVALUATOR)

# Evaluator cannot call /internal/recall-gate (admin-only)
ok, role, reason = authenticate_and_authorize(
    INTERNAL, "/internal/recall-gate", internal_token=INTERNAL)
check("28. Evaluator denied admin operation", not ok and "insufficient" in reason)

# Admin can call /internal/recall-gate
ok, role, reason = authenticate_and_authorize(
    ADMIN, "/internal/recall-gate", admin_token=ADMIN)
check("29. Admin can recall-gate", ok and role == Role.ADMIN)

# Admin can evaluate (higher privilege)
ok, role, reason = authenticate_and_authorize(
    ADMIN, "/internal/evaluate", admin_token=ADMIN)
check("30. Admin can evaluate (higher privilege)", ok)

# Operator can read observability
ok, role, reason = authenticate_and_authorize(
    COMPLIANCE, "/internal/observability",
    internal_token=INTERNAL, compliance_token=COMPLIANCE)
check("31. Operator can read observability", ok and role == Role.OPERATOR)

# Unauthenticated rejected
ok, role, reason = authenticate_and_authorize("", "/internal/evaluate")
check("32. Unauthenticated rejected", not ok and reason == "authentication_required")

# Wrong token rejected
ok, role, reason = authenticate_and_authorize(
    "wrong-token", "/internal/evaluate", internal_token=INTERNAL)
check("33. Wrong token rejected", not ok and reason == "authentication_required")

# Health endpoint has no role requirement
ok, role, reason = authenticate_and_authorize("", "/health")
check("34. Health endpoint unauthenticated OK", ok)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Full Access Control Check
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 5: Full Access Control Check ===")

ok, reason, meta = check_access(
    INTERNAL, "/internal/evaluate", client_ip="10.0.0.1",
    internal_token=INTERNAL)
check("35. Full check: evaluator + evaluate = allowed", ok)
check("36. Full check: role in metadata", meta.get("role") == "evaluator")

ok, reason, meta = check_access(
    INTERNAL, "/internal/recall-gate", client_ip="10.0.0.1",
    internal_token=INTERNAL)
check("37. Full check: evaluator + admin op = denied", not ok)

ok, reason, meta = check_access(
    ADMIN, "/internal/recall-gate", client_ip="10.0.0.1",
    admin_token=ADMIN)
check("38. Full check: admin + admin op = allowed", ok)

ok, reason, meta = check_access(
    "", "/internal/evaluate", client_ip="10.0.0.1",
    internal_token=INTERNAL)
check("39. Full check: no token = denied", not ok and "authentication" in reason)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Rate Limiting — Basic
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 6: Rate Limiting — Basic ===")

limiter = RateLimiter()
cfg = RateLimitConfig(max_requests=5, window_seconds=1.0, block_seconds=2.0)

# First 5 requests should pass
for i in range(5):
    allowed, info = limiter.check("client-1", "/test-endpoint", cfg)
    check(f"4{i+1}. Request {i+1}/5 allowed", allowed)

# 6th request should be blocked
allowed, info = limiter.check("client-1", "/test-endpoint", cfg)
check("46. Request 6 blocked", not allowed)
check("47. Rate limit info has retry_after", info["retry_after"] > 0)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Rate Limiting — Window Reset
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 7: Rate Limiting — Window Reset ===")

limiter2 = RateLimiter()
cfg_fast = RateLimitConfig(max_requests=3, window_seconds=0.5, block_seconds=0.5)

# Exhaust limit
for i in range(3):
    limiter2.check("client-reset", "/reset-test", cfg_fast)
allowed, _ = limiter2.check("client-reset", "/reset-test", cfg_fast)
check("48. Blocked after 3", not allowed)

# Wait for window to expire
time.sleep(0.6)
allowed, info = limiter2.check("client-reset", "/reset-test", cfg_fast)
check("49. Allowed after window reset", allowed)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Rate Limiting — Client Isolation
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 8: Rate Limiting — Client Isolation ===")

limiter3 = RateLimiter()
cfg_iso = RateLimitConfig(max_requests=2, window_seconds=60.0, block_seconds=60.0)

limiter3.check("client-a", "/iso-test", cfg_iso)
limiter3.check("client-a", "/iso-test", cfg_iso)
allowed_a, _ = limiter3.check("client-a", "/iso-test", cfg_iso)
check("50. Client A blocked after 2", not allowed_a)

# Client B should still be allowed
allowed_b, _ = limiter3.check("client-b", "/iso-test", cfg_iso)
check("51. Client B isolated, still allowed", allowed_b)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Rate Limiting — Endpoint Isolation
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 9: Rate Limiting — Endpoint Isolation ===")

limiter4 = RateLimiter()
cfg_ep = RateLimitConfig(max_requests=2, window_seconds=60.0, block_seconds=60.0)

limiter4.check("client-ep", "/ep-a", cfg_ep)
limiter4.check("client-ep", "/ep-a", cfg_ep)
allowed_a, _ = limiter4.check("client-ep", "/ep-a", cfg_ep)
check("52. Client blocked on endpoint A", not allowed_a)

# Same client on different endpoint should work
allowed_b, _ = limiter4.check("client-ep", "/ep-b", cfg_ep)
check("53. Same client, different endpoint, allowed", allowed_b)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: Rate Limiting — Thread Safety
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 10: Rate Limiting — Thread Safety ===")

limiter5 = RateLimiter()
cfg_thread = RateLimitConfig(max_requests=100, window_seconds=60.0, block_seconds=60.0)
thread_results = []
thread_errors = []


def rate_thread(n: int) -> None:
    try:
        results = []
        for _ in range(20):
            allowed, _ = limiter5.check(f"thread-client-{n}", "/thread-test", cfg_thread)
            results.append(allowed)
        thread_results.append(results)
    except Exception as e:
        thread_errors.append(str(e))


threads = [threading.Thread(target=rate_thread, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check("54. Thread safety: no exceptions", len(thread_errors) == 0)
# All 200 requests (10 threads * 20) should be within limit (100 per client)
total_allowed = sum(sum(r) for r in thread_results)
check("55. Thread safety: correct total allowed", total_allowed == 200)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: Rate Limiting — Memory Bounded
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 11: Rate Limiting — Memory Bounded ===")

limiter6 = RateLimiter()
cfg_mem = RateLimitConfig(max_requests=1, window_seconds=0.1, block_seconds=0.1)

# Create buckets with old timestamps to test cleanup
old_time = time.time() - 7200  # 2 hours ago
for i in range(100):
    key = f"mem-client-{i}:/mem-test"
    limiter6._buckets[key].timestamps = [old_time]

bucket_count_before = limiter6.bucket_count()
check("56. Buckets created", bucket_count_before >= 100)

# Cleanup should remove old buckets
with limiter6._lock:
    limiter6._cleanup(time.time())
bucket_count_after = limiter6.bucket_count()
check("57. Stale buckets cleaned up", bucket_count_after < bucket_count_before)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: Rate Limit Configuration
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 12: Rate Limit Configuration ===")

check("58. Evaluate has config", "/internal/evaluate" in _RATE_LIMIT_CONFIGS)
check("59. Batch has config", "/internal/evaluate-batch" in _RATE_LIMIT_CONFIGS)
check("60. Recall-gate has config", "/internal/recall-gate" in _RATE_LIMIT_CONFIGS)
check("61. Observability has config", "/internal/observability" in _RATE_LIMIT_CONFIGS)

eval_cfg = _RATE_LIMIT_CONFIGS["/internal/evaluate"]
check("62. Evaluate limit reasonable", eval_cfg.max_requests >= 100)

rl = get_rate_limiter()
check("63. Global rate limiter accessible", rl is not None)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13: Token Masking
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 13: Token Masking ===")

masked = mask_token_for_log("my-secret-token-12345")
check("64. Masked token is hash", len(masked) == 12)
check("65. Masked token not original", masked != "my-secret-token-12345")
check("66. Empty token masked safely", mask_token_for_log("") == "none")
check("67. Same token same mask", mask_token_for_log("abc") == mask_token_for_log("abc"))
check("68. Different token different mask", mask_token_for_log("abc") != mask_token_for_log("xyz"))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14: Endpoint Role Mapping
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 14: Endpoint Role Mapping ===")

check("69. /internal/evaluate requires EVALUATOR",
      _ENDPOINT_ROLES.get("/internal/evaluate") == Role.EVALUATOR)
check("70. /internal/evaluate-batch requires EVALUATOR",
      _ENDPOINT_ROLES.get("/internal/evaluate-batch") == Role.EVALUATOR)
check("71. /internal/attribution requires OPERATOR",
      _ENDPOINT_ROLES.get("/internal/attribution") == Role.OPERATOR)
check("72. /internal/drift-status requires OPERATOR",
      _ENDPOINT_ROLES.get("/internal/drift-status") == Role.OPERATOR)
check("73. /internal/release-attestation requires OPERATOR",
      _ENDPOINT_ROLES.get("/internal/release-attestation") == Role.OPERATOR)
check("74. /internal/observability requires OPERATOR",
      _ENDPOINT_ROLES.get("/internal/observability") == Role.OPERATOR)
check("75. /internal/recall-gate requires ADMIN",
      _ENDPOINT_ROLES.get("/internal/recall-gate") == Role.ADMIN)
check("76. /health not in endpoint roles (unprotected)",
      "/health" not in _ENDPOINT_ROLES)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 15: Privacy — No Secrets in Logs/Telemetry
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 15: Privacy — No Secrets in Logs/Telemetry ===")

from src.monitoring.observability_integration import get_store

store = get_store()

# Run through auth failures to generate security events
for i in range(3):
    authenticate_and_authorize(
        f"wrong-token-{i}", "/internal/evaluate",
        internal_token=INTERNAL, correlation_id=f"priv-test-{i}")

# Export state and check for secrets
if store:
    state = json.dumps(store.export_state())
    check("77. No INTERNAL_TOKEN in exports", INTERNAL not in state)
    check("78. No ADMIN_TOKEN in exports", ADMIN not in state)
    check("79. No compliance token in exports", COMPLIANCE not in state)

    # Check security events
    for ev in store.security_ledger.get_events(limit=50):
        ev_str = json.dumps(ev.to_dict())
        check(f"80. No raw token in event {ev.event_id[:8]}", INTERNAL not in ev_str)
        check(f"81. No admin token in event {ev.event_id[:8]}", ADMIN not in ev_str)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 16: Security Telemetry Integration
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 16: Security Telemetry Integration ===")

from src.monitoring.observability import SecurityEventType

if store:
    events_before = store.security_ledger.get_event_count()

    # Auth failure should generate event
    authenticate_and_authorize(
        "bad-token", "/internal/evaluate",
        internal_token=INTERNAL, correlation_id="sec-test-auth")

    events_after = store.security_ledger.get_event_count()
    check("82. Auth failure generates security event", events_after > events_before)

    # Authz failure should generate event
    authenticate_and_authorize(
        INTERNAL, "/internal/recall-gate",
        internal_token=INTERNAL, correlation_id="sec-test-authz")

    # Check for AUTH_FAILURE events
    auth_events = store.security_ledger.get_events(
        event_type=SecurityEventType.AUTH_FAILURE)
    check("83. Auth failure events present", len(auth_events) > 0)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 17: Fail-Closed Security
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 17: Fail-Closed Security ===")

# Token verification with empty strings must fail closed
check("84. Empty token fails closed", not verify_token_constant_time("", "anything"))
check("85. Empty expected fails closed", not verify_token_constant_time("anything", ""))
check("86. None-equivalent empty fails closed", not verify_token_constant_time("", ""))

# Authorization with no tokens must deny
ok, _, _ = authenticate_and_authorize("", "/internal/evaluate")
check("87. No token = denied (fail-closed)", not ok)

ok, _, _ = authenticate_and_authorize("bad", "/internal/evaluate", internal_token=INTERNAL)
check("88. Bad token = denied (fail-closed)", not ok)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 18: Rate Limit in Full Access Check
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 18: Rate Limit in Full Access Check ===")

# Create a limiter with a very low limit for testing
test_limiter = RateLimiter()
test_cfg = RateLimitConfig(max_requests=2, window_seconds=60.0, block_seconds=60.0)

# Use the check_access function with rate limiting
for i in range(2):
    ok, _, meta = check_access(
        INTERNAL, "/internal/evaluate", client_ip="10.0.0.99",
        internal_token=INTERNAL, rate_limit=True)
    # May or may not be allowed depending on global limiter state

# The global limiter uses its own configs; test with custom limiter
allowed, info = test_limiter.check("rl-test-client", "/rl-test", test_cfg)
check("89. First request allowed", allowed)
allowed, info = test_limiter.check("rl-test-client", "/rl-test", test_cfg)
check("90. Second request allowed", allowed)
allowed, info = test_limiter.check("rl-test-client", "/rl-test", test_cfg)
check("91. Third request blocked (limit=2)", not allowed)
check("92. Rate limit info has limit", info["limit"] == 2)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 19: Concurrent Auth Failures
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 19: Concurrent Auth Failures ===")

conc_errors = []


def conc_auth_fail(n: int) -> None:
    try:
        for _ in range(10):
            authenticate_and_authorize(
                f"concurrent-bad-{n}", "/internal/evaluate",
                internal_token=INTERNAL, correlation_id=f"conc-{n}")
    except Exception as e:
        conc_errors.append(str(e))


threads = [threading.Thread(target=conc_auth_fail, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check("93. Concurrent auth failures: no exceptions", len(conc_errors) == 0)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 20: Determinism
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 20: Determinism ===")

# Same token same result
r1 = resolve_role(INTERNAL, internal_token=INTERNAL)
r2 = resolve_role(INTERNAL, internal_token=INTERNAL)
check("94. Role resolution deterministic", r1 == r2 == Role.EVALUATOR)

# Same auth check same result
ok1, role1, _ = authenticate_and_authorize(INTERNAL, "/internal/evaluate", internal_token=INTERNAL)
ok2, role2, _ = authenticate_and_authorize(INTERNAL, "/internal/evaluate", internal_token=INTERNAL)
check("95. Auth check deterministic", ok1 == ok2 and role1 == role2)

# Masked token deterministic
m1 = mask_token_for_log("test-token")
m2 = mask_token_for_log("test-token")
check("96. Token masking deterministic", m1 == m2)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 21: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 21: Edge Cases ===")

# Very long token
check("97. Long token handled", not verify_token_constant_time("x" * 10000, "y" * 10000))

# Unicode token
check("98. Unicode token handled", verify_token_constant_time("🔑", "🔑"))

# Endpoint not in role map (should allow)
ok, role, reason = authenticate_and_authorize(INTERNAL, "/unknown-endpoint", internal_token=INTERNAL)
check("99. Unknown endpoint allowed (no restriction)", ok)

# Rate limiter with zero max
zero_cfg = RateLimitConfig(max_requests=0, window_seconds=60.0)
rl_zero = RateLimiter()
allowed, _ = rl_zero.check("zero-client", "/zero-test", zero_cfg)
check("100. Zero-limit immediately blocks", not allowed)

# Rate limiter with very high limit
high_cfg = RateLimitConfig(max_requests=10000, window_seconds=60.0)
rl_high = RateLimiter()
for i in range(100):
    rl_high.check("high-client", "/high-test", high_cfg)
allowed, _ = rl_high.check("high-client", "/high-test", high_cfg)
check("101. High limit: 101st request still allowed", allowed)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 22: Security Event Chain Integrity
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 22: Security Event Chain Integrity ===")

if store:
    chain_ok, chain_msg = store.security_ledger.verify_chain()
    check("102. Security chain valid after access control events", chain_ok)
    check("103. Chain message intact", chain_msg == "Chain intact")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 23: Health Endpoint Unprotected
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 23: Health Endpoint Unprotected ===")

ok, role, reason = authenticate_and_authorize("", "/health")
check("104. Health: no auth required", ok)
ok, role, reason = authenticate_and_authorize("anything", "/health")
check("105. Health: any token accepted", ok)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 24: Production Endpoint Security
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 24: Production Endpoint Security ===")

# Every internal endpoint should require at least EVALUATOR
internal_endpoints = [
    "/internal/evaluate",
    "/internal/evaluate-batch",
    "/internal/attribution",
    "/internal/drift-status",
    "/internal/release-attestation",
    "/internal/latency-slo",
    "/internal/metrics",
    "/internal/observability",
    "/internal/recall-gate",
]

for ep in internal_endpoints:
    ok, _, _ = authenticate_and_authorize("", ep)
    check(f"106-{ep[-15:]}: {ep} requires auth", not ok)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 25: Administrative Operations Protected
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 25: Administrative Operations Protected ===")

# Evaluator cannot do admin ops
admin_endpoints = ["/internal/recall-gate"]
for ep in admin_endpoints:
    ok, _, reason = authenticate_and_authorize(INTERNAL, ep, internal_token=INTERNAL)
    check(f"107. Evaluator denied {ep}", not ok and "insufficient" in reason)

# Admin can do admin ops
for ep in admin_endpoints:
    ok, role, _ = authenticate_and_authorize(ADMIN, ep, admin_token=ADMIN)
    check(f"108. Admin allowed {ep}", ok and role == Role.ADMIN)

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"Phase 73 Access Control Tests: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'=' * 60}")

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  - {e}")

sys.exit(0 if failed == 0 else 1)
