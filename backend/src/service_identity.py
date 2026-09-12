"""Per-service JWT credentials for service-to-service authentication.

Replaces the shared INTERNAL_TOKEN with short-lived, scoped JWTs.
Each service gets its own signing key and identity. In production,
these would be backed by a secrets manager or service mesh.

Usage:
    # Service A calls Service B:
    token = issue_service_token("privacy", "risk", ["evaluate"])
    # Service B verifies:
    claims = verify_service_token(token, expected_service="risk")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

# Service identities and their allowed scopes
SERVICE_IDENTITIES = {
    "front":     {"port": 8000, "scopes": ["read", "admin"]},
    "identity":  {"port": 8001, "scopes": ["identity", "read"]},
    "privacy":   {"port": 8002, "scopes": ["privacy", "ingest"]},
    "risk":      {"port": 8003, "scopes": ["risk", "evaluate"]},
    "verify":    {"port": 8004, "scopes": ["verify", "alerts"]},
    "audit":     {"port": 8005, "scopes": ["audit", "compliance"]},
}

# Default scope grants: which services can call which endpoints
# In production this would be an RBAC policy file
SCOPE_GRANTS = {
    # "caller_service": ["allowed scopes on the target"]
    "identity":  ["identity"],
    "privacy":   ["privacy", "ingest"],
    "risk":      ["risk", "evaluate"],
    "verify":    ["verify", "alerts", "read"],
    "audit":     ["audit", "compliance", "read"],
    "front":     ["read", "admin"],
}


class ServiceTokenManager:
    """Manages per-service JWT tokens with HMAC signing."""

    def __init__(self, master_key: str | None = None):
        """Initialize with a master key for signing service tokens.

        In production, each service would have its own key pair.
        For the prototype, a single master key derives per-service keys.
        """
        if master_key is None:
            master_key = os.environ.get("SERVICE_TOKEN_KEY", "")
        if not master_key:
            master_key = secrets.token_hex(32)

        self._master_key = master_key.encode()
        # Derive per-service signing keys from master
        self._service_keys: dict[str, bytes] = {}
        for service_name in SERVICE_IDENTITIES:
            self._service_keys[service_name] = hashlib.sha256(
                f"service-key:{service_name}:{master_key}".encode()
            ).digest()

    def issue_token(
        self,
        caller_service: str,
        target_service: str,
        scopes: list[str] | None = None,
        ttl_seconds: int = 300,  # 5 minutes default
    ) -> str:
        """Issue a short-lived service token.

        Args:
            caller_service: The service making the call
            target_service: The service being called
            scopes: Specific scopes requested (defaults to caller's identity scopes)
            ttl_seconds: Token lifetime (default 5 minutes)

        Returns:
            Signed JWT-like token string
        """
        if caller_service not in SERVICE_IDENTITIES:
            raise ValueError(f"Unknown service: {caller_service}")
        if target_service not in SERVICE_IDENTITIES:
            raise ValueError(f"Unknown target: {target_service}")

        now = int(time.time())
        if scopes is None:
            scopes = SERVICE_IDENTITIES[caller_service]["scopes"]

        payload = {
            "iss": caller_service,      # issuer (who called)
            "aud": target_service,      # audience (who is called)
            "sub": caller_service,      # subject
            "scopes": scopes,
            "iat": now,
            "exp": now + ttl_seconds,
            "jti": secrets.token_hex(16),
        }

        # Sign with caller's derived key
        body = json.dumps(payload, separators=(",", ":"))
        key = self._service_keys[caller_service]
        signature = hmac.new(key, body.encode(), hashlib.sha256).hexdigest()

        return f"{_b64encode(body)}.{signature}"

    def verify_token(
        self,
        token: str,
        expected_service: str,
        required_scopes: list[str] | None = None,
    ) -> dict:
        """Verify a service token and return claims.

        Args:
            token: The token to verify
            expected_service: The service that should receive this token
            required_scopes: Scopes that must be present

        Returns:
            Verified claims dict

        Raises:
            ValueError: If token is invalid, expired, or lacks required scopes
        """
        parts = token.split(".")
        if len(parts) != 2:
            raise ValueError("Invalid token format")

        body_b64, signature = parts

        # Decode body
        try:
            body = json.loads(_b64decode(body_b64))
        except Exception:
            raise ValueError("Invalid token payload")

        # Check expiry
        now = int(time.time())
        if body.get("exp", 0) < now:
            raise ValueError("Token expired")

        # Check audience
        if body.get("aud") != expected_service:
            raise ValueError(f"Token not intended for {expected_service}")

        # Verify signature with caller's key
        caller = body.get("iss", "")
        if caller not in self._service_keys:
            raise ValueError(f"Unknown caller: {caller}")

        key = self._service_keys[caller]
        expected_sig = hmac.new(key, _b64decode(body_b64), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            raise ValueError("Invalid signature")

        # Check scopes
        if required_scopes:
            token_scopes = set(body.get("scopes", []))
            for scope in required_scopes:
                if scope not in token_scopes:
                    raise ValueError(f"Missing required scope: {scope}")

        return body

    def get_service_key(self, service_name: str) -> bytes:
        """Get the signing key for a service (for direct HMAC verification)."""
        return self._service_keys.get(service_name, b"")


def _b64encode(data: str | bytes) -> str:
    """URL-safe base64 encode without padding."""
    if isinstance(data, str):
        data = data.encode()
    return secrets.token_urlsafe(len(data)).replace("-", "_").replace("+", "/")[:0] + \
        __import__("base64").urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(data: str) -> bytes:
    """URL-safe base64 decode with padding."""
    import base64
    padding = 4 - len(data) % 4
    return base64.urlsafe_b64decode(data + "=" * padding)


# Singleton instance
_token_manager: ServiceTokenManager | None = None


def get_token_manager() -> ServiceTokenManager:
    """Get or create the singleton token manager."""
    global _token_manager
    if _token_manager is None:
        from src.settings import settings
        _token_manager = ServiceTokenManager(settings.internal_token)
    return _token_manager
