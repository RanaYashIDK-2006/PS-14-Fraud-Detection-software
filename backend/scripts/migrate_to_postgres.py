#!/usr/bin/env python3
"""PostgreSQL Schema Migration for PS-14

Creates per-service schemas and all tables in a single PostgreSQL instance.
Each service gets its own schema for isolation (matching the SQLite-per-file
design).

Usage:
    # Set DATABASE_URL first
    export DATABASE_URL=postgresql://user:pass@localhost:5432/ps14

    # Create all schemas and tables
    python scripts/migrate_to_postgres.py

    # Or create a specific schema only
    python scripts/migrate_to_postgres.py --schema identity
    python scripts/migrate_to_postgres.py --schema privacy
    python scripts/migrate_to_postgres.py --schema risk
    python scripts/migrate_to_postgres.py --schema audit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker


# Service schemas
SCHEMAS = {
    "identity": {
        "description": "DB-1: Identity Store (PII, auth, pseudonyms)",
        "env_var": "IDENTITY_DB_URL",
    },
    "privacy": {
        "description": "DB-2: Feature Store (pseudonymous behavioral data)",
        "env_var": "PRIVACY_DB_URL",
    },
    "risk": {
        "description": "DB-3: Risk & Model Store (scores, decisions)",
        "env_var": "RISK_DB_URL",
    },
    "audit": {
        "description": "DB-4: Audit Store (hash-chained events)",
        "env_var": "AUDIT_DB_URL",
    },
    "verification": {
        "description": "DB-5: Verification Store (outcomes, cases)",
        "env_var": "VERIFY_DB_URL",
    },
}


def get_engine(database_url: str | None = None) -> tuple:
    """Create engine from DATABASE_URL or per-service env vars."""
    url = database_url or os.environ.get("DATABASE_URL", "")
    if not url:
        # Try per-service URLs
        for svc, info in SCHEMAS.items():
            svc_url = os.environ.get(info["env_var"], "")
            if svc_url:
                url = svc_url
                break
    if not url or not url.startswith("postgresql"):
        print("ERROR: No PostgreSQL URL found.")
        print("Set DATABASE_URL or one of: " + ", ".join(info["env_var"] for info in SCHEMAS.values()))
        sys.exit(1)

    engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    return engine, url


def create_schemas(engine, schemas: list[str] | None = None):
    """Create PostgreSQL schemas for each service."""
    target_schemas = schemas or list(SCHEMAS.keys())
    with engine.connect() as conn:
        for schema in target_schemas:
            if schema not in SCHEMAS:
                print(f"  WARNING: Unknown schema '{schema}', skipping")
                continue
            print(f"  Creating schema: {schema}")
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            # Grant usage to default role (for connection pooling)
            conn.execute(text(f"GRANT USAGE ON SCHEMA {schema} TO PUBLIC"))
            conn.execute(text(f"GRANT ALL ON SCHEMA {schema} TO PUBLIC"))
        conn.commit()
    print("  Schemas created successfully")


def create_tables(engine, schema: str):
    """Create tables for a specific schema using SQLAlchemy models."""
    print(f"\n  Creating tables for schema: {schema}")

    if schema == "identity":
        from src.identity_service.models import Base
    elif schema == "privacy":
        from src.privacy_layer.models import Base
    elif schema == "risk":
        from src.risk_engine.models import Base
    elif schema == "audit":
        from src.audit_service.models import Base
    elif schema == "verification":
        from src.verification_service.models import Base
    else:
        print(f"  ERROR: Unknown schema '{schema}'")
        return

    # Create a new engine bound to this schema
    schema_engine = engine.execution_options(
        schema_translate_map={None: schema}
    )

    # Create all tables in the schema
    Base.metadata.create_all(bind=schema_engine)
    print(f"  Tables created in schema '{schema}'")


def create_audit_triggers(engine, schema: str):
    """Create append-only triggers for audit_events table (PostgreSQL version)."""
    print(f"\n  Creating audit triggers for schema: {schema}")

    trigger_sql = """
    -- Prevent UPDATE on audit_events
    CREATE OR REPLACE FUNCTION prevent_audit_update()
    RETURNS TRIGGER AS $$
    BEGIN
        RAISE EXCEPTION 'audit_events is append-only: UPDATE blocked';
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql;

    -- Prevent DELETE on audit_events
    CREATE OR REPLACE FUNCTION prevent_audit_delete()
    RETURNS TRIGGER AS $$
    BEGIN
        RAISE EXCEPTION 'audit_events is append-only: DELETE blocked';
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql;

    -- Drop existing triggers if they exist
    DROP TRIGGER IF EXISTS audit_events_no_update ON audit_events;
    DROP TRIGGER IF EXISTS audit_events_no_delete ON audit_events;

    -- Create triggers
    CREATE TRIGGER audit_events_no_update
        BEFORE UPDATE ON audit_events
        FOR EACH ROW
        EXECUTE FUNCTION prevent_audit_update();

    CREATE TRIGGER audit_events_no_delete
        BEFORE DELETE ON audit_events
        FOR EACH ROW
        EXECUTE FUNCTION prevent_audit_delete();
    """

    # Create triggers in the specified schema
    schema_engine = engine.execution_options(
        schema_translate_map={None: schema}
    )

    with schema_engine.connect() as conn:
        # Split and execute each statement
        statements = [s.strip() for s in trigger_sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                conn.execute(text(stmt))
            except Exception as e:
                # Some statements may fail if objects don't exist yet
                print(f"    Note: {str(e)[:80]}...")
        conn.commit()
    print(f"  Audit triggers created in schema '{schema}'")


def verify_schema(engine, schema: str):
    """Verify schema was created correctly."""
    print(f"\n  Verifying schema: {schema}")

    with engine.connect() as conn:
        # Check schema exists
        result = conn.execute(text(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name = :schema"
        ), {"schema": schema})
        if not result.fetchone():
            print(f"    ERROR: Schema '{schema}' not found")
            return False

        # List tables in schema
        result = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = :schema ORDER BY table_name"
        ), {"schema": schema})
        tables = [row[0] for row in result.fetchall()]
        print(f"    Tables: {', '.join(tables) if tables else 'none'}")

        # Check indexes
        result = conn.execute(text(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname = :schema ORDER BY indexname"
        ), {"schema": schema})
        indexes = [row[0] for row in result.fetchall()]
        print(f"    Indexes: {len(indexes)}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Create PostgreSQL schemas and tables for PS-14")
    parser.add_argument("--schema", choices=list(SCHEMAS.keys()),
                       help="Create only this schema (default: all)")
    parser.add_argument("--database-url", help="PostgreSQL connection URL")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing schemas")
    args = parser.parse_args()

    print("=" * 70)
    print("PS-14 PostgreSQL Schema Migration")
    print("=" * 70)

    engine, url = get_engine(args.database_url)
    print(f"\nConnected to: {url.split('@')[-1] if '@' in url else url}")

    if args.verify_only:
        schemas = [args.schema] if args.schema else list(SCHEMAS.keys())
        for schema in schemas:
            verify_schema(engine, schema)
        return

    # Create schemas
    schemas = [args.schema] if args.schema else list(SCHEMAS.keys())
    create_schemas(engine, schemas)

    # Create tables for each schema
    for schema in schemas:
        create_tables(engine, schema)

    # Create audit triggers for audit schema
    if args.schema is None or args.schema == "audit":
        create_audit_triggers(engine, "audit")

    # Verify
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)
    for schema in schemas:
        verify_schema(engine, schema)

    print("\n" + "=" * 70)
    print("Migration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
