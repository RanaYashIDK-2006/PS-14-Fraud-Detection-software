"""Phase 75: Transport security and secret/key lifecycle hardening.

Provides:
- Centralized secret redaction for logs/errors/responses
- PostgreSQL TLS configuration
- Configuration validation for production security
- Key lifecycle documentation and validation
- Health-safe key status reporting

This module does NOT implement:
- Enterprise IAM
- Cloud KMS/HSM
- Offensive security
- Regulatory compliance

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ── Secret Redaction ────────────────────────────────────────────────────

# Patterns that look like secrets
_SECRET_PATTERNS = [
    # Connection strings
    re.compile(r'postgresql://[^:]+:[^@]+@', re.IGNORECASE),
    re.compile(r'postgres://[^:]+:[^@]+@', re.IGNORECASE),
    re.compile(r'mysql://[^:]+:[^@]+@', re.IGNORECASE),
    re.compile(r'redis://[^:]*:[^@]+@', re.IGNORECASE),
    # Bearer tokens
    re.compile(r'Bearer\s+[A-Za-z0-9\-._~+/]+=*', re.IGNORECASE),
    # Hex tokens (32+ chars)
    re.compile(r'\b[A-Fa-f0-9]{32,}\b'),
    # token=value patterns
    re.compile(r'(token|key|secret|password|auth)=\S+', re.IGNORECASE),
    # JWT-like
    re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'),
    # AWS-style keys
    re.compile(r'AKIA[0-9A-Z]{16}'),
    # Private key headers
    re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----'),
]

# Fields that should never appear in logs
_SENSITIVE_FIELDS = frozenset({
    'password', 'passwd', 'secret', 'token', 'api_key', 'apikey',
    'private_key', 'privatekey', 'access_key', 'secret_key',
    'authorization', 'credential', 'jwt_secret', 'fernet_key',
    'internal_token', 'compliance_token', 'database_url',
    'DATABASE_URL', 'INTERNAL_TOKEN', 'COMPLIANCE_TOKEN',
    'JWT_SECRET', 'PII_ENCRYPTION_KEY', 'EXPORT_SIGNING_KEY',
    'ADMIN_PASS', 'ADMIN_TOKEN',
})


def redact_string(value: str, max_length: int = 200) -> str:
    """Redact potential secrets from a string.

    Preserves structure but replaces sensitive content with [REDACTED].
    Truncates to max_length to prevent log flooding.
    """
    if not value or not isinstance(value, str):
        return str(value)[:max_length]

    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub('[REDACTED]', result)

    return result[:max_length]


def redact_dict(d: dict[str, Any], max_depth: int = 3) -> dict[str, Any]:
    """Recursively redact potential secrets from a dictionary.

    Preserves structure but replaces sensitive values with [REDACTED].
    """
    if max_depth <= 0:
        return {'...': '[TRUNCATED]'}

    result = {}
    for key, value in d.items():
        key_lower = str(key).lower()

        if key_lower in {f.lower() for f in _SENSITIVE_FIELDS}:
            result[key] = '[REDACTED]'
        elif isinstance(value, dict):
            result[key] = redact_dict(value, max_depth - 1)
        elif isinstance(value, str):
            result[key] = redact_string(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [
                redact_dict(item, max_depth - 1) if isinstance(item, dict)
                else redact_string(str(item)) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def safe_error_message(error: Exception) -> str:
    """Create a safe error message that doesn't leak secrets.

    Truncates and redacts the error message.
    """
    msg = str(error)
    return redact_string(msg, max_length=150)


def mask_token(token: str) -> str:
    """Create a safe pseudonymous identifier for logging.

    Never exposes the actual token value.
    """
    if not token:
        return 'none'
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def mask_database_url(url: str) -> str:
    """Mask a database URL to hide credentials.

    PostgreSQL://user:***@host:port/db
    """
    if not url:
        return 'not-configured'

    try:
        # Parse and mask
        prefix = ''
        if url.startswith('postgresql://'):
            prefix = 'postgresql://'
            rest = url[len(prefix):]
        elif url.startswith('postgres://'):
            prefix = 'postgres://'
            rest = url[len(prefix):]
        else:
            return redact_string(url, max_length=30)

        # Find user:pass@host — use rfind('@') to handle @ in passwords
        at_idx = rest.rfind('@')
        if at_idx > 0:
            user_part = rest[:at_idx]
            host_part = rest[at_idx + 1:]

            # Mask password — use first colon for user:pass split
            colon_idx = user_part.find(':')
            if colon_idx > 0:
                user = user_part[:colon_idx]
                return f'{prefix}{user}:***@{host_part}'
            else:
                return f'{prefix}{user_part}@{host_part}'

        return redact_string(url, max_length=30)
    except Exception:
        return '[REDACTED]'


# ── PostgreSQL TLS Configuration ────────────────────────────────────────

class TLSMode(str, Enum):
    """PostgreSQL TLS modes."""
    DISABLE = 'disable'           # No TLS (development only)
    ALLOW = 'allow'               # Try TLS, fall back to no TLS
    PREFER = 'prefer'             # Prefer TLS, accept no TLS
    REQUIRE = 'require'           # Require TLS
    VERIFY_CA = 'verify-ca'       # Require TLS + verify CA
    VERIFY_FULL = 'verify-full'   # Require TLS + verify CA + hostname


@dataclass
class PostgreSQLTLSConfig:
    """PostgreSQL TLS configuration."""
    sslmode: TLSMode = TLSMode.REQUIRE
    sslrootcert: str = ''         # CA certificate path
    sslcert: str = ''             # Client certificate path
    sslkey: str = ''              # Client private key path
    connect_timeout: int = 10
    options: str = ''             # Additional connection options

    def to_connect_args(self) -> dict[str, str]:
        """Convert to SQLAlchemy connect_args for PostgreSQL."""
        args = {
            'sslmode': self.sslmode.value,
        }
        if self.sslrootcert:
            args['sslrootcert'] = self.sslrootcert
        if self.sslcert:
            args['sslcert'] = self.sslcert
        if self.sslkey:
            args['sslkey'] = self.sslkey
        if self.connect_timeout:
            args['connect_timeout'] = str(self.connect_timeout)
        return args

    def to_url_params(self) -> str:
        """Convert to URL query parameters."""
        params = [f'sslmode={self.sslmode.value}']
        if self.sslrootcert:
            params.append(f'sslrootcert={self.sslrootcert}')
        if self.sslcert:
            params.append(f'sslcert={self.sslcert}')
        if self.sslkey:
            params.append(f'sslkey={self.sslkey}')
        return '&'.join(params)

    def validate(self) -> tuple[bool, str]:
        """Validate TLS configuration for production use."""
        if self.sslmode in (TLSMode.DISABLE, TLSMode.ALLOW):
            return False, f'TLS mode {self.sslmode.value} is not allowed in production'

        if self.sslmode in (TLSMode.REQUIRE, TLSMode.VERIFY_CA, TLSMode.VERIFY_FULL):
            if self.sslrootcert and not Path(self.sslrootcert).exists():
                return False, f'CA certificate not found: {self.sslrootcert}'

        if self.sslcert and not Path(self.sslcert).exists():
            return False, f'Client certificate not found: {self.sslcert}'

        if self.sslkey and not Path(self.sslkey).exists():
            return False, f'Client key not found: {self.sslkey}'

        return True, 'TLS configuration valid'


def get_postgresql_tls_config() -> PostgreSQLTLSConfig:
    """Get PostgreSQL TLS configuration from environment."""
    sslmode = os.environ.get('PGSSLMODE', 'require')
    sslrootcert = os.environ.get('PGSSLROOTCERT', '')
    sslcert = os.environ.get('PGSSLCERT', '')
    sslkey = os.environ.get('PGSSLKEY', '')
    timeout = int(os.environ.get('PGCONNECT_TIMEOUT', '10'))

    return PostgreSQLTLSConfig(
        sslmode=TLSMode(sslmode),
        sslrootcert=sslrootcert,
        sslcert=sslcert,
        sslkey=sslkey,
        connect_timeout=timeout,
    )


# ── Key Lifecycle ───────────────────────────────────────────────────────

class KeyStatus(str, Enum):
    """Key lifecycle status."""
    ACTIVE = 'active'
    PREVIOUS = 'previous'     # Accept for verification, not for signing
    REVOKED = 'revoked'
    EXPIRED = 'expired'


@dataclass
class KeyRecord:
    """Record of a cryptographic key with lifecycle metadata."""
    key_id: str
    purpose: str
    algorithm: str
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    rotated_at: float | None = None
    expires_at: float | None = None
    fingerprint: str = ''  # Safe hash of key, never the key itself

    def is_usable(self) -> bool:
        """Whether this key can be used for its purpose."""
        if self.status in (KeyStatus.REVOKED, KeyStatus.EXPIRED):
            return False
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True

    def to_safe_dict(self) -> dict[str, Any]:
        """Export key metadata without key material."""
        return {
            'key_id': self.key_id,
            'purpose': self.purpose,
            'algorithm': self.algorithm,
            'status': self.status.value,
            'created_at': self.created_at,
            'rotated_at': self.rotated_at,
            'expires_at': self.expires_at,
            'fingerprint': self.fingerprint,
        }


class KeyManager:
    """Manages cryptographic key lifecycle.

    Does NOT store actual key material — only metadata and fingerprints.
    Key values are loaded from environment/configuration at runtime.
    """

    def __init__(self) -> None:
        self._keys: dict[str, KeyRecord] = {}
        self._lock = threading.Lock()

    def register_key(
        self,
        key_id: str,
        purpose: str,
        algorithm: str,
        fingerprint: str = '',
    ) -> KeyRecord:
        """Register a key with lifecycle metadata."""
        with self._lock:
            record = KeyRecord(
                key_id=key_id,
                purpose=purpose,
                algorithm=algorithm,
                fingerprint=fingerprint,
            )
            self._keys[key_id] = record
            return record

    def get_key(self, key_id: str) -> KeyRecord | None:
        """Get a key record by ID."""
        return self._keys.get(key_id)

    def get_active_key(self, purpose: str) -> KeyRecord | None:
        """Get the active key for a purpose."""
        for record in self._keys.values():
            if record.purpose == purpose and record.status == KeyStatus.ACTIVE:
                return record
        return None

    def rotate_key(self, old_key_id: str, new_key_id: str) -> tuple[bool, str]:
        """Rotate a key: old becomes PREVIOUS, new becomes ACTIVE."""
        with self._lock:
            old = self._keys.get(old_key_id)
            new = self._keys.get(new_key_id)

            if old is None:
                return False, f'Old key {old_key_id} not found'
            if new is None:
                return False, f'New key {new_key_id} not found'
            if old.purpose != new.purpose:
                return False, f'Purpose mismatch: {old.purpose} vs {new.purpose}'

            old.status = KeyStatus.PREVIOUS
            old.rotated_at = time.time()
            new.status = KeyStatus.ACTIVE

            return True, f'Rotated {old_key_id} -> {new_key_id}'

    def revoke_key(self, key_id: str) -> tuple[bool, str]:
        """Revoke a key."""
        with self._lock:
            record = self._keys.get(key_id)
            if record is None:
                return False, f'Key {key_id} not found'

            record.status = KeyStatus.REVOKED
            record.rotated_at = time.time()
            return True, f'Revoked {key_id}'

    def get_all_keys(self) -> list[dict[str, Any]]:
        """Get all key records (safe export, no key material)."""
        return [r.to_safe_dict() for r in self._keys.values()]

    def verify_key_usable(self, key_id: str) -> tuple[bool, str]:
        """Verify a key is usable."""
        record = self._keys.get(key_id)
        if record is None:
            return False, f'Key {key_id} not found'
        if not record.is_usable():
            return False, f'Key {key_id} status is {record.status.value}'
        return True, 'Key is usable'


# Global key manager
_key_manager = KeyManager()


def get_key_manager() -> KeyManager:
    """Get the global key manager."""
    return _key_manager


# ── Configuration Validation ────────────────────────────────────────────

class ConfigSecurityLevel(str, Enum):
    """Security configuration level."""
    DEVELOPMENT = 'development'
    STAGING = 'staging'
    PRODUCTION = 'production'


@dataclass
class SecurityConfigCheck:
    """Result of a security configuration check."""
    check_name: str
    passed: bool
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    message: str
    remediation: str = ''


def validate_production_config(
    ps14_mode: str = '',
    database_url: str = '',
    internal_token: str = '',
    jwt_secret: str = '',
    admin_token: str = '',
) -> list[SecurityConfigCheck]:
    """Validate security configuration for the given deployment mode.

    Returns a list of configuration checks with pass/fail status.
    """
    checks = []
    mode = ps14_mode.lower() if ps14_mode else os.environ.get('PS14_MODE', 'production').lower()

    # 1. Database URL
    if not database_url:
        checks.append(SecurityConfigCheck(
            check_name='database_url_configured',
            passed=False,
            severity='HIGH',
            message='DATABASE_URL not configured — using SQLite fallback',
            remediation='Set DATABASE_URL for PostgreSQL in production',
        ))
    elif database_url.startswith('postgresql'):
        # Check TLS in URL
        if 'sslmode=disable' in database_url.lower():
            checks.append(SecurityConfigCheck(
                check_name='database_tls_enabled',
                passed=False,
                severity='CRITICAL',
                message='PostgreSQL connection has sslmode=disable',
                remediation='Remove sslmode=disable or set sslmode=require',
            ))
        else:
            checks.append(SecurityConfigCheck(
                check_name='database_tls_enabled',
                passed=True,
                severity='INFO',
                message='PostgreSQL TLS configuration present',
            ))

    # 2. Secrets not empty in production
    if mode == 'production':
        if not internal_token:
            checks.append(SecurityConfigCheck(
                check_name='internal_token_set',
                passed=False,
                severity='CRITICAL',
                message='INTERNAL_TOKEN not set in production',
                remediation='Set INTERNAL_TOKEN environment variable',
            ))
        else:
            checks.append(SecurityConfigCheck(
                check_name='internal_token_set',
                passed=True,
                severity='INFO',
                message='INTERNAL_TOKEN configured',
            ))

        if not jwt_secret:
            checks.append(SecurityConfigCheck(
                check_name='jwt_secret_set',
                passed=False,
                severity='CRITICAL',
                message='JWT_SECRET not set in production',
                remediation='Set JWT_SECRET environment variable',
            ))
        else:
            checks.append(SecurityConfigCheck(
                check_name='jwt_secret_set',
                passed=True,
                severity='INFO',
                message='JWT_SECRET configured',
            ))

        # Check for dev-only defaults
        if internal_token and len(internal_token) < 32:
            checks.append(SecurityConfigCheck(
                check_name='internal_token_strength',
                passed=False,
                severity='HIGH',
                message='INTERNAL_TOKEN appears too short for production',
                remediation='Use a cryptographically random token of at least 32 bytes',
            ))

    # 3. TLS mode
    pg_tls = get_postgresql_tls_config()
    if mode == 'production' and database_url:
        tls_valid, tls_msg = pg_tls.validate()
        checks.append(SecurityConfigCheck(
            check_name='postgresql_tls_valid',
            passed=tls_valid,
            severity='CRITICAL' if not tls_valid else 'INFO',
            message=tls_msg,
            remediation='Configure appropriate PostgreSQL TLS settings',
        ))

    return checks


# ── Health-Safe Status ──────────────────────────────────────────────────

def get_security_health_status(
    ps14_mode: str = '',
    database_url: str = '',
) -> dict[str, Any]:
    """Get security health status safe for /health endpoints.

    Returns only safe metadata — never secrets, keys, or credentials.
    """
    mode = ps14_mode.lower() if ps14_mode else os.environ.get('PS14_MODE', 'production').lower()

    # TLS status
    pg_tls = get_postgresql_tls_config()
    tls_valid, tls_msg = pg_tls.validate()

    # Config validation
    checks = validate_production_config(
        ps14_mode=mode,
        database_url=database_url,
    )

    failed_critical = [c for c in checks if not c.passed and c.severity == 'CRITICAL']
    failed_high = [c for c in checks if not c.passed and c.severity == 'HIGH']

    # Key manager status — try to use init if available, otherwise direct
    try:
        keys = get_key_manager().get_all_keys()
    except Exception:
        keys = []

    return {
        'security_mode': mode,
        'tls_configured': pg_tls.sslmode != TLSMode.DISABLE,
        'tls_mode': pg_tls.sslmode.value,
        'tls_valid': tls_valid,
        'config_checks_total': len(checks),
        'config_checks_passed': sum(1 for c in checks if c.passed),
        'config_checks_failed_critical': len(failed_critical),
        'config_checks_failed_high': len(failed_high),
        'keys_registered': len(keys),
        'keys_active': sum(1 for k in keys if k.get('status') == 'active'),
        'database_masked': mask_database_url(database_url),
    }


# ── Initialization ──────────────────────────────────────────────────────

def init_security_hardening() -> None:
    """Initialize security hardening components.

    Called once at startup to register known keys and validate config.
    """
    from src.settings import settings

    # Register known keys (metadata only, not key values)
    get_key_manager().register_key(
        key_id='jwt-secret',
        purpose='jwt-signing',
        algorithm='HS256',
        fingerprint=hashlib.sha256(settings.jwt_secret.encode()).hexdigest()[:16] if settings.jwt_secret else '',
    )

    get_key_manager().register_key(
        key_id='internal-token',
        purpose='internal-auth',
        algorithm='hmac-sha256',
        fingerprint=hashlib.sha256(settings.internal_token.encode()).hexdigest()[:16] if settings.internal_token else '',
    )

    get_key_manager().register_key(
        key_id='compliance-token',
        purpose='compliance-auth',
        algorithm='hmac-sha256',
        fingerprint=hashlib.sha256(settings.compliance_token.encode()).hexdigest()[:16] if settings.compliance_token else '',
    )

    get_key_manager().register_key(
        key_id='fernet-key',
        purpose='pii-encryption',
        algorithm='AES-256-CBC',
        fingerprint=hashlib.sha256(settings.fernet_key).hexdigest()[:16] if settings.fernet_key else '',
    )

    get_key_manager().register_key(
        key_id='export-signing',
        purpose='export-signing',
        algorithm='hmac-sha256',
        fingerprint=hashlib.sha256(settings.export_signing_key).hexdigest()[:16] if settings.export_signing_key else '',
    )
