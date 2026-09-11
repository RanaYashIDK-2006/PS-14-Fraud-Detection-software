"""Shared configuration for the PS-14 prototype services (dev defaults).

Everything here is overridable via environment variables (see .env.example).
Production would replace the SQLite paths with per-store PostgreSQL DSNs and
the dev secrets with KMS/HSM-backed material (architecture section 4).

Set PS14_MODE=production to enable production security gates.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# Project root: parent of src/ directory. Used to anchor relative paths.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- Production mode gate ---------------------------------------------------
# PS14_MODE=production enables fail-closed startup validation.
# Default is PRODUCTION (fail-closed) — explicitly set PS14_MODE=development
# in your environment to relax security gates during local development.
# This prevents accidental deployment with dev holes open.
PS14_MODE: str = os.environ.get("PS14_MODE", "production").lower()


class Settings(BaseSettings):
    # Physically separate stores: DB-1 (Identity) and DB-2 (Feature Store).
    # Anchored to project root so services work regardless of CWD.
    db_dir: str = str(_PROJECT_ROOT / "db")

    # PostgreSQL connection URL (e.g. Supabase, Neon, RDS).
    # When set, all services use this single PostgreSQL instance instead of
    # per-service SQLite files.  Falls back to SQLite when unset.
    # Also checks per-service env vars (IDENTITY_DB_URL, RISK_DB_URL, etc.)
    # used by docker-compose.prod.yml's separate-Postgres-per-service model.
    database_url: str = os.environ.get("DATABASE_URL", "")

    # Per-service PostgreSQL schema.  Each service writes to its own schema
    # to preserve the isolation the SQLite-per-file design gave for free.
    # Set DB_SCHEMA=<name> in each service's own environment (e.g. DB_SCHEMA=identity).
    # Each service is a separate process — the same env var name, different values.
    db_schema: str = os.environ.get("DB_SCHEMA", "public")

    @property
    def use_postgres(self) -> bool:
        return bool(self.database_url and self.database_url.startswith("postgresql"))

    @property
    def use_pg_pooler(self) -> bool:
        """True when connecting via Supabase/Neon pooled port (6543).

        Transaction-mode poolers don't support prepared statements, so we
        must use NullPool to avoid "prepared statement does not exist" errors.
        """
        return self.use_postgres and (":6543/" in self.database_url
                                     or "pooler.supabase.com" in self.database_url)

    # SECURITY: Separate keys for different purposes.
    # Each key serves a specific function and can be rotated independently.
    # In production these MUST come from .env or a secrets manager.
    # Dev defaults are randomly generated per-process (no hardcoded secrets).
    jwt_secret: str = ""
    pii_encryption_key: str = ""
    export_signing_key_str: str = os.environ.get("EXPORT_SIGNING_KEY", "")
    jwt_expiry_minutes: int = 15
    internal_token: str = ""

    # Compliance-role gate for the audit viewer (dev default; production:
    # real RBAC + mTLS per section 3 - never a shared passphrase).
    compliance_token: str = ""

    # SECURITY: CORS allowed origins (production: restrict to specific domains)
    cors_origins: str = "http://127.0.0.1:8000,http://127.0.0.1:8004,http://localhost:8000,http://localhost:8004"

    # Service URLs for server-to-server chaining (Verification demo seed,
    # compliance viewer).
    identity_url: str = "http://127.0.0.1:8001"
    privacy_url: str = "http://127.0.0.1:8002"
    risk_url: str = "http://127.0.0.1:8003"
    audit_url: str = "http://127.0.0.1:8005"

    def model_post_init(self, __context: object) -> None:
        """Generate random secrets for any empty fields (no env var set).

        This ensures dev mode works without a .env file while never using
        hardcoded secrets that could leak into production.

        Also resolves per-service database URLs from docker-compose.prod.yml
        (IDENTITY_DB_URL, RISK_DB_URL, etc.) by matching SERVICE_NAME.
        """
        if not self.jwt_secret:
            object.__setattr__(self, 'jwt_secret', secrets.token_hex(32))
        if not self.pii_encryption_key:
            object.__setattr__(self, 'pii_encryption_key', secrets.token_hex(32))
        if not self.export_signing_key_str:
            object.__setattr__(self, 'export_signing_key_str', secrets.token_hex(32))
        if not self.internal_token:
            object.__setattr__(self, 'internal_token', secrets.token_hex(32))
        if not self.compliance_token:
            object.__setattr__(self, 'compliance_token', secrets.token_hex(32))
        # Resolve per-service database URL (docker-compose.prod.yml sets
        # IDENTITY_DB_URL, RISK_DB_URL, etc. — not DATABASE_URL).
        if not self.database_url:
            svc = os.environ.get("SERVICE_NAME", "").upper()
            per_svc_var = f"{svc}_DB_URL" if svc else ""
            per_svc_url = os.environ.get(per_svc_var, "") if per_svc_var else ""
            if per_svc_url:
                object.__setattr__(self, 'database_url', per_svc_url)
                # Also derive schema from SERVICE_NAME (identity/privacy/risk/audit)
                if self.db_schema == "public":
                    svc_lower = os.environ.get("SERVICE_NAME", "").lower()
                    if svc_lower in ("identity", "privacy", "risk", "audit"):
                        object.__setattr__(self, 'db_schema', svc_lower)

    @property
    def allowed_cors_origins(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def identity_db_path(self) -> Path:
        return Path(self.db_dir) / "identity.db"

    @property
    def features_db_path(self) -> Path:
        return Path(self.db_dir) / "features.db"

    @property
    def risk_db_path(self) -> Path:
        return Path(self.db_dir) / "risk.db"

    @property
    def audit_db_path(self) -> Path:
        return Path(self.db_dir) / "audit.db"

    @property
    def verify_db_path(self) -> Path:
        return Path(self.db_dir) / "verify.db"

    @property
    def fernet_key(self) -> bytes:
        """PII encryption key - SEPARATE from JWT secret.

        Production: envelope encryption with per-tenant keys backed by
        KMS/HSM (architecture section 3/4).
        """
        return base64.urlsafe_b64encode(hashlib.sha256(self.pii_encryption_key.encode()).digest())

    @property
    def export_signing_key(self) -> bytes:
        """Export signing key - SEPARATE from JWT secret.

        Production: sign with a KMS/HSM asymmetric key (Ed25519) and
        publish the public half so regulators can verify without any
        shared secret.
        """
        return hashlib.sha256(self.export_signing_key_str.encode()).digest()


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def load_dotenv_and_patch() -> None:
    """Load .env into os.environ and patch settings if env has values Settings missed.

    Settings has no env_file, so services started detached never see .env vars.
    Call this once at module level, BEFORE any other code reads settings.*.
    """
    import os as _os
    from pathlib import Path as _Path
    env_path = _Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in _os.environ:
            _os.environ[k] = v
    # Patch settings for secrets that Settings defaulted to empty/random.
    # Always override with .env value when present — Settings may have
    # already generated a random default, but .env is the source of truth
    # for multi-service coordination (e.g. INTERNAL_TOKEN shared across
    # identity, audit, front, etc.).
    # Env-var names, not field names: os.environ is case-sensitive on Linux,
    # and export_signing_key_str's env name (EXPORT_SIGNING_KEY) differs from
    # the field name anyway.
    for field, env_name in (
        ("internal_token", "INTERNAL_TOKEN"),
        ("compliance_token", "COMPLIANCE_TOKEN"),
        ("jwt_secret", "JWT_SECRET"),
        ("pii_encryption_key", "PII_ENCRYPTION_KEY"),
        ("export_signing_key_str", "EXPORT_SIGNING_KEY"),
    ):
        env_val = _os.environ.get(env_name)
        if env_val:
            object.__setattr__(settings, field, env_val)


# --- Production security gate ----------------------------------------------
# When PS14_MODE=production, validate that no dev defaults are active.
# If any critical secret still carries a dev prefix, abort immediately.
_DEV_SECRETS = {
    "jwt_secret": settings.jwt_secret,
    "pii_encryption_key": settings.pii_encryption_key,
    "export_signing_key_str": settings.export_signing_key_str,
    "internal_token": settings.internal_token,
    "compliance_token": settings.compliance_token,
}


def validate_production_config() -> list[str]:
    """Validate that production mode has no insecure defaults.

    Returns a list of validation errors. Empty list = valid.
    Called once at startup; in production mode, errors trigger sys.exit(1).
    """
    errors: list[str] = []
    # Re-read from environment (may have been set by .env loader after import)
    current_mode = os.environ.get("PS14_MODE", "production").lower()
    if current_mode != "production":
        return errors

    # In production, every secret MUST come from an env var.
    # model_post_init generates random values for empty fields, which is
    # secure but not reproducible — production requires explicit config.
    for name, value in _DEV_SECRETS.items():
        env_key = name.upper()
        if env_key == "EXPORT_SIGNING_KEY_STR":
            env_key = "EXPORT_SIGNING_KEY"
        if not os.environ.get(env_key):
            errors.append(
                f"PRODUCTION MODE: {name} not set via environment variable. "
                f"Set {env_key} in .env or environment."
            )

    # CORS must not be the default dev set
    if "localhost" in settings.cors_origins and "127.0.0.1" in settings.cors_origins:
        errors.append(
            "PRODUCTION MODE: CORS origins still set to localhost/127.0.0.1. "
            "Set CORS_ORIGINS to your production domains."
        )

    # Database: production Docker deployments MUST use PostgreSQL.
    # Only enforced when SERVICE_NAME is set (indicating containerized deployment).
    # Without SERVICE_NAME, the service is likely running outside Docker
    # (local dev, CI, testing) and SQLite fallback is acceptable.
    has_postgres = bool(settings.database_url and settings.database_url.startswith("postgresql"))
    svc = os.environ.get("SERVICE_NAME", "").upper()
    if svc:  # Only enforce when running as a named service (Docker)
        if not has_postgres:
            per_svc_var = f"{svc}_DB_URL"
            per_svc_url = os.environ.get(per_svc_var, "")
            has_postgres = bool(per_svc_url and per_svc_url.startswith("postgresql"))
        if not has_postgres:
            errors.append(
                f"PRODUCTION MODE ({svc}): No PostgreSQL URL configured. "
                f"Set DATABASE_URL or {svc}_DB_URL. "
                f"Production must NOT silently fall back to SQLite."
            )

    # Schema isolation: in production with Postgres, each service MUST use its own schema.
    # "public" means all services share one schema, defeating per-service isolation.
    if has_postgres and settings.db_schema == "public":
        errors.append(
            "PRODUCTION MODE: DB_SCHEMA is 'public' with Postgres. "
            "Set DB_SCHEMA=<service_schema> (identity/privacy/risk/audit) "
            "in each service's environment for proper isolation."
        )

    return errors


def enforce_production_gate() -> None:
    """Call at the start of every service's lifespan.

    Re-reads PS14_MODE from os.environ at call time (not at import time)
    because .env may have been loaded between module import and lifespan start.
    In production mode, if validation fails, abort with a clear message.
    In development mode, just warn.
    """
    # Re-read PS14_MODE from the environment (may have been set by .env loader)
    current_mode = os.environ.get("PS14_MODE", "production").lower()
    errors = validate_production_config()
    if not errors:
        return

    msg = "\n".join("  " + e for e in errors)
    if current_mode == "production":
        print(f"\nFATAL: Production configuration errors:\n{msg}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"WARNING: Development mode config issues:\n{msg}", file=sys.stderr)
