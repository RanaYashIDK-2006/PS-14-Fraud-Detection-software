"""Alembic environment for PS-14 multi-schema PostgreSQL.

Supports:
- Per-service schema isolation (identity, privacy, risk, audit, verification)
- PostgreSQL with connection pooling
- SQLite fallback for local development
- Configurable via DATABASE_URL environment variable
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.settings import settings

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models for autogenerate support
from src.risk_engine.models import Base as RiskBase
from src.audit_service.models import Base as AuditBase
from src.identity_service.models import Base as IdentityBase
from src.privacy_layer.models import Base as PrivacyBase

# Combine all metadata for autogenerate
target_metadata = RiskBase.metadata

# Add other service metadata if they exist
try:
    for table in AuditBase.metadata.tables.values():
        if table.name not in target_metadata.tables:
            target_metadata._add_table(table)
except Exception:
    pass

# Determine database URL
def get_url():
    """Get database URL from environment or settings."""
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url

    if settings.use_postgres:
        return settings.database_url

    # Fallback to SQLite for local development
    return f"sqlite:///{settings.risk_db_path}"


# Per-service schemas for PostgreSQL
SCHEMAS = {
    "identity": ["users", "pseudonym_mapping"],
    "privacy": ["transaction_features", "velocity_windows"],
    "risk": ["risk_scores"],
    "audit": ["audit_events", "audit_access_log"],
    "verification": ["verification_outcomes"],
}


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL
    and not an Engine.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table="alembic_version",
        version_table_schema="public",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an Engine and associates a connection
    with the context.
    """
    url = get_url()

    # Configure engine based on database type
    if url.startswith("postgresql"):
        connectable = engine_from_config(
            {"sqlalchemy.url": url},
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
            pool_pre_ping=True,
        )
    else:
        # SQLite fallback
        connectable = engine_from_config(
            {"sqlalchemy.url": url},
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        # For PostgreSQL, set search_path to include all schemas
        is_pg = url.startswith("postgresql")
        if is_pg:
            connection.execute(
                "SET search_path TO identity, privacy, risk, audit, verification, public"
            )

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=is_pg,
            version_table="alembic_version",
            version_table_schema="public" if is_pg else None,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
