"""Phase 73: Production access control — authentication, authorization, rate limiting.

Provides a single integrated access control layer for PS-14 services:

- Constant-time token verification (hmac.compare_digest)
- Role-based authorization (EVALUATOR, OPERATOR, ADMIN)
- Thread-safe sliding-window rate limiting per client+role
- Privacy-safe security telemetry integration with Phase 70/72
- Fail-closed for security controls, fail-open for observability

Scope: single-process, in-memory. Not distributed. Documented limitation.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.monitoring.observability_integration import (
    get_store,
    on_auth_failure,
    on_validation_failure,
)
from src.monitoring.observability import (
    SecurityEventType,
    SecuritySeverity,
)


# ── Roles ────────────────────────────────────────────────────────────────

class Role(str, Enum):
    """Authorization roles with least-privilege escalation."""
    EVALUATOR = "evaluator"    # Can call /internal/evaluate
    OPERATOR = "operator"      # Can read metrics, drift status, health
    ADMIN = "admin"            # Can modify runtime state (recall-gate, etc.)
    SYSTEM = "system"          # Internal service-to-service (full access)


# ── Permission Model ─────────────────────────────────────────────────────

# Maps endpoint prefixes to minimum required role
_ENDPOINT_ROLES: dict[str, Role] = {
    "/internal/evaluate": Role.EVALUATOR,
    "/internal/evaluate-batch": Role.EVALUATOR,
    "/internal/attribution": Role.OPERATOR,
    "/internal/drift-status": Role.OPERATOR,
    "/internal/release-attestation": Role.OPERATOR,
    "/internal/latency-slo": Role.OPERATOR,
    "/internal/metrics": Role.OPERATOR,
    "/internal/observability": Role.OPERATOR,
    "/internal/recall-gate": Role.ADMIN,
    "/internal/recall-gate": Role.ADMIN,
    # Health is intentionally unprotected — required for liveness probes
}

_ROLE_HIERARCHY: dict[Role, int] = {
    Role.EVALUATOR: 1,
    Role.OPERATOR: 2,
    Role.ADMIN: 3,
    Role.SYSTEM: 99,
}

# Token-to-role mapping: HMAC(token) -> role
# In production, multiple tokens with different roles would be configured.
# For the prototype, we derive a single evaluator token from INTERNAL_TOKEN
# and an admin token from ADMIN_TOKEN (if set).


def _role_allows(user_role: Role, required_role: Role) -> bool:
    """Check if user_role satisfies required_role via hierarchy."""
    return _ROLE_HIERARCHY.get(user_role, 0) >= _ROLE_HIERARCHY.get(required_role, 0)


# ── Token Verification ──────────────────────────────────────────────────

def verify_token_constant_time(token: str, expected: str) -> bool:
    """Constant-time token comparison. Never leaks timing information."""
    if not token or not expected:
        return False
    # hmac.compare_digest requires matching types; encode to bytes for safety
    try:
        return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))
    except (TypeError, UnicodeDecodeError):
        return False


def resolve_role(
    token: str,
    internal_token: str = "",
    admin_token: str = "",
    compliance_token: str = "",
) -> Role | None:
    """Resolve a bearer token to a role. Returns None if unauthenticated.

    Resolution order (constant-time comparisons):
    1. ADMIN_TOKEN -> ADMIN
    2. COMPLIANCE_TOKEN -> OPERATOR
    3. INTERNAL_TOKEN -> EVALUATOR
    4. None -> unauthenticated
    """
    if not token:
        return None

    # Admin token (highest priority)
    if admin_token and verify_token_constant_time(token, admin_token):
        return Role.ADMIN

    # Compliance token (operator-level)
    if compliance_token and verify_token_constant_time(token, compliance_token):
        return Role.OPERATOR

    # Internal token (evaluator-level)
    if internal_token and verify_token_constant_time(token, internal_token):
        return Role.EVALUATOR

    return None


def authenticate_and_authorize(
    token: str,
    endpoint: str,
    internal_token: str = "",
    admin_token: str = "",
    compliance_token: str = "",
    correlation_id: str = "",
) -> tuple[bool, Role | None, str]:
    """Authenticate a token and check authorization for the endpoint.

    Returns:
        (allowed, role, denial_reason)
    """
    # Check if endpoint requires any role at all
    required = _ENDPOINT_ROLES.get(endpoint)
    if required is None:
        # No role required for this endpoint (e.g. /health) — skip auth
        role = resolve_role(token, internal_token, admin_token, compliance_token)
        return True, role, ""

    # 1. Authenticate
    role = resolve_role(token, internal_token, admin_token, compliance_token)

    if role is None:
        # Authentication failure
        on_auth_failure(correlation_id=correlation_id)
        return False, None, "authentication_required"

    # 2. Authorize
    if not _role_allows(role, required):
        # Authorization failure
        _record_authz_failure(correlation_id, endpoint, role, required)
        return False, role, f"insufficient_permissions: need {required.value}, have {role.value}"

    return True, role, ""


def _record_authz_failure(
    correlation_id: str,
    endpoint: str,
    user_role: Role,
    required_role: Role,
) -> None:
    """Record an authorization failure in observability. Fail-open."""
    try:
        store = get_store()
        if store is None:
            return
        store.metrics.record_auth_failure()
        store.security_ledger.record_event(
            SecurityEventType.AUTH_FAILURE,
            SecuritySeverity.MEDIUM,
            "access_control",
            correlation_id=correlation_id,
            attempted_operation=endpoint,
            outcome="AUTHORIZATION_DENIED",
            details={
                "user_role": user_role.value,
                "required_role": required_role.value,
            },
        )
    except Exception:
        pass  # Fail-open observability


# ── Rate Limiting ────────────────────────────────────────────────────────

@dataclass
class RateLimitConfig:
    """Configuration for a rate limit bucket."""
    max_requests: int = 100
    window_seconds: float = 60.0
    block_seconds: float = 60.0


# Default rate limit configurations per endpoint category
_RATE_LIMIT_CONFIGS: dict[str, RateLimitConfig] = {
    "/internal/evaluate": RateLimitConfig(max_requests=500, window_seconds=60.0, block_seconds=30.0),
    "/internal/evaluate-batch": RateLimitConfig(max_requests=100, window_seconds=60.0, block_seconds=60.0),
    "/internal/attribution": RateLimitConfig(max_requests=30, window_seconds=60.0, block_seconds=60.0),
    "/internal/recall-gate": RateLimitConfig(max_requests=10, window_seconds=60.0, block_seconds=120.0),
    "/internal/observability": RateLimitConfig(max_requests=60, window_seconds=60.0, block_seconds=30.0),
    # Health is intentionally not rate-limited
}


@dataclass
class _ClientBucket:
    """Sliding window counter for a single client+endpoint."""
    timestamps: list[float] = field(default_factory=list)
    blocked_until: float = 0.0

    def is_blocked(self, now: float) -> bool:
        return now < self.blocked_until

    def add_and_check(self, now: float, config: RateLimitConfig) -> tuple[bool, int, float]:
        """Add a request timestamp and check against limit.

        Returns (allowed, remaining, retry_after).
        """
        if self.is_blocked(now):
            retry = max(0.0, self.blocked_until - now)
            return False, 0, retry

        # Evict old entries outside window
        cutoff = now - config.window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]

        if len(self.timestamps) >= config.max_requests:
            self.blocked_until = now + config.block_seconds
            return False, 0, config.block_seconds

        self.timestamps.append(now)
        remaining = config.max_requests - len(self.timestamps)
        return True, remaining, 0.0


class RateLimiter:
    """Thread-safe in-memory sliding-window rate limiter.

    Scoped to single process. Documented limitation.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, _ClientBucket] = defaultdict(_ClientBucket)
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def check(
        self,
        client_id: str,
        endpoint: str,
        config: RateLimitConfig | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Check rate limit for a client+endpoint.

        Returns (allowed, info_dict).
        """
        now = time.time()
        cfg = config or _RATE_LIMIT_CONFIGS.get(endpoint, RateLimitConfig())
        key = f"{client_id}:{endpoint}"

        with self._lock:
            bucket = self._buckets[key]
            allowed, remaining, retry_after = bucket.add_and_check(now, cfg)

            # Periodic cleanup to prevent unbounded memory
            if now - self._last_cleanup > 300:
                self._cleanup(now)
                self._last_cleanup = now

        info = {
            "allowed": allowed,
            "limit": cfg.max_requests,
            "remaining": remaining,
            "window_seconds": cfg.window_seconds,
            "retry_after": round(retry_after, 1),
        }

        if not allowed:
            # Record rate-limit security event
            try:
                store = get_store()
                if store is not None:
                    store.metrics.record_malformed()  # counts as rejection
                    store.security_ledger.record_event(
                        SecurityEventType.REQUEST_BURST,
                        SecuritySeverity.MEDIUM,
                        "rate_limiter",
                        attempted_operation=endpoint,
                        outcome="RATE_LIMITED",
                        details={
                            "client_id_hash": hashlib.sha256(client_id.encode()).hexdigest()[:16],
                            "limit": cfg.max_requests,
                            "window": cfg.window_seconds,
                        },
                    )
            except Exception:
                pass  # Fail-open observability

        return allowed, info

    def _cleanup(self, now: float) -> None:
        """Remove stale buckets to bound memory. Must be called under lock."""
        stale_keys = [
            key for key, bucket in self._buckets.items()
            if (not bucket.timestamps or all(t < now - 3600 for t in bucket.timestamps))
            and bucket.blocked_until < now
        ]
        for key in stale_keys:
            del self._buckets[key]

    def get_config(self, endpoint: str) -> RateLimitConfig:
        """Get the rate limit config for an endpoint."""
        return _RATE_LIMIT_CONFIGS.get(endpoint, RateLimitConfig())

    def bucket_count(self) -> int:
        """Number of active client buckets (for testing)."""
        with self._lock:
            return len(self._buckets)


# ── Access Control Coordinator ───────────────────────────────────────────

# Global rate limiter instance (single-process scope)
_global_limiter = RateLimiter()
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter."""
    return _global_limiter


def check_access(
    token: str,
    endpoint: str,
    client_ip: str = "unknown",
    internal_token: str = "",
    admin_token: str = "",
    compliance_token: str = "",
    correlation_id: str = "",
    rate_limit: bool = True,
) -> tuple[bool, str, dict[str, Any]]:
    """Full access control check: authenticate + authorize + rate limit.

    Returns:
        (allowed, denial_reason, metadata)
        metadata includes role, rate_limit_info, etc.
    """
    metadata: dict[str, Any] = {"endpoint": endpoint}

    # 1. Authentication + Authorization
    allowed, role, denial = authenticate_and_authorize(
        token=token,
        endpoint=endpoint,
        internal_token=internal_token,
        admin_token=admin_token,
        compliance_token=compliance_token,
        correlation_id=correlation_id,
    )
    metadata["role"] = role.value if role else None
    metadata["authz"] = denial if not allowed else "ok"

    if not allowed:
        return False, denial, metadata

    # 2. Rate limiting (skip for /health and SYSTEM role)
    if rate_limit and endpoint != "/health" and role != Role.SYSTEM:
        limiter = get_rate_limiter()
        client_key = f"{client_ip}:{role.value}"
        rl_allowed, rl_info = limiter.check(client_key, endpoint)
        metadata["rate_limit"] = rl_info

        if not rl_allowed:
            return False, "rate_limit_exceeded", metadata

    return True, "", metadata


def mask_token_for_log(token: str) -> str:
    """Create a safe pseudonymous identifier for logging. Never exposes token."""
    if not token:
        return "none"
    return hashlib.sha256(token.encode()).hexdigest()[:12]
