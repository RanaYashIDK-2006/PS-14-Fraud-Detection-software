#!/usr/bin/env python3
"""Complete PostgreSQL Migration for PS-14

This script handles the full migration from SQLite to PostgreSQL:
1. Creates schemas for each service
2. Creates all tables with proper indexes
3. Creates append-only triggers for audit table
4. Migrates existing data from SQLite databases
5. Verifies the migration

Usage:
    # Set DATABASE_URL
    export DATABASE_URL=postgresql://user:pass@localhost:5432/ps14

    # Full migration
    python scripts/pg_migration.py

    # Schema only (no data)
    python scripts/pg_migration.py --schema-only

    # Verify only
    python scripts/pg_migration.py --verify-only
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker


# Database mappings
DB_MAPPING = {
    "identity": {
        "sqlite_file": "identity.db",
        "schema": "identity",
        "models_module": "src.identity_service.models",
    },
    "privacy": {
        "sqlite_file": "features.db",
        "schema": "privacy",
        "models_module": "src.privacy_layer.models",
    },
    "risk": {
        "sqlite_file": "risk.db",
        "schema": "risk",
        "models_module": "src.risk_engine.models",
    },
    "audit": {
        "sqlite_file": "audit.db",
        "schema": "audit",
        "models_module": "src.audit_service.models",
    },
    "verification": {
        "sqlite_file": "verify.db",
        "schema": "verification",
        "models_module": "src.verification_service.models",
    },
}


def get_engine(database_url: str | None = None):
    """Create PostgreSQL engine."""
    url = database_url or os.environ.get("DATABASE_URL", "")
    if not url or not url.startswith("postgresql"):
        print("ERROR: Set DATABASE_URL to a PostgreSQL connection string")
        sys.exit(1)
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)


def create_schemas(engine):
    """Create all service schemas."""
    print("\n[1] Creating schemas...")
    schemas = ["identity", "privacy", "risk", "audit", "verification"]
    with engine.connect() as conn:
        for schema in schemas:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            conn.execute(text(f"GRANT USAGE ON SCHEMA {schema} TO PUBLIC"))
            conn.execute(text(f"GRANT ALL ON SCHEMA {schema} TO PUBLIC"))
            print(f"  ✓ Schema: {schema}")
        conn.commit()


def create_tables(engine, schema: str, models_module: str):
    """Create tables for a specific schema."""
    print(f"\n[2] Creating tables for schema: {schema}")

    # Import models
    module = __import__(models_module, fromlist=["Base"])
    Base = module.Base

    # Create schema-aware engine
    schema_engine = engine.execution_options(
        schema_translate_map={None: schema}
    )

    # Create all tables
    Base.metadata.create_all(bind=schema_engine)

    # List created tables
    inspector = inspect(schema_engine)
    tables = inspector.get_table_names(schema=schema)
    print(f"  ✓ Tables created: {', '.join(tables)}")


def create_audit_triggers(engine):
    """Create append-only triggers for audit_events table."""
    print("\n[3] Creating audit triggers...")

    # Create audit schema engine
    schema_engine = engine.execution_options(
        schema_translate_map={None: "audit"}
    )

    with schema_engine.connect() as conn:
        # Create trigger functions
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION prevent_audit_update()
            RETURNS TRIGGER AS $$
            BEGIN
                RAISE EXCEPTION 'audit_events is append-only: UPDATE blocked';
                RETURN NULL;
            END;
            $$ LANGUAGE plpgsql;
        """))

        conn.execute(text("""
            CREATE OR REPLACE FUNCTION prevent_audit_delete()
            RETURNS TRIGGER AS $$
            BEGIN
                RAISE EXCEPTION 'audit_events is append-only: DELETE blocked';
                RETURN NULL;
            END;
            $$ LANGUAGE plpgsql;
        """))

        # Create triggers
        conn.execute(text("""
            DROP TRIGGER IF EXISTS audit_events_no_update ON audit_events;
            CREATE TRIGGER audit_events_no_update
                BEFORE UPDATE ON audit_events
                FOR EACH ROW
                EXECUTE FUNCTION prevent_audit_update();
        """))

        conn.execute(text("""
            DROP TRIGGER IF EXISTS audit_events_no_delete ON audit_events;
            CREATE TRIGGER audit_events_no_delete
                BEFORE DELETE ON audit_events
                FOR EACH ROW
                EXECUTE FUNCTION prevent_audit_delete();
        """))

        conn.commit()
        print("  ✓ Audit append-only triggers created")


def migrate_data(engine, db_name: str, sqlite_path: Path, schema: str, batch_size: int = 1000):
    """Migrate data from SQLite to PostgreSQL."""
    if not sqlite_path.exists():
        print(f"  ⚠ SQLite file not found: {sqlite_path}")
        return 0

    conn = sqlite3.connect(str(sqlite_path))
    cursor = conn.cursor()

    # Get tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    total_migrated = 0

    for table in tables:
        # Count rows
        cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
        count = cursor.fetchone()[0]
        if count == 0:
            continue

        print(f"  Migrating {table} ({count} rows)...")

        # Get columns
        cursor.execute(f"PRAGMA table_info([{table}])")
        columns = [row[1] for row in cursor.fetchall()]

        # Build insert statement
        placeholders = ", ".join([f":{col}" for col in columns])
        insert_sql = f"""
            INSERT INTO {schema}.{table} ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
        """

        # Migrate in batches
        cursor.execute(f"SELECT * FROM [{table}]")
        pg_conn = engine.connect()

        try:
            migrated = 0
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                batch = [dict(row) for row in rows]
                pg_conn.execute(text(insert_sql), batch)
                migrated += len(batch)

                if migrated % 10000 == 0 or migrated == count:
                    print(f"    {migrated}/{count} rows ({100*migrated//count}%)")

            pg_conn.commit()
            total_migrated += migrated
        except Exception as e:
            pg_conn.rollback()
            print(f"    ERROR: {e}")
        finally:
            pg_conn.close()

    conn.close()
    return total_migrated


def verify_migration(engine):
    """Verify migration was successful."""
    print("\n[5] Verifying migration...")

    schemas = ["identity", "privacy", "risk", "audit", "verification"]
    for schema in schemas:
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema=schema)
        print(f"\n  Schema: {schema}")
        for table in tables:
            result = engine.execute(text(f"SELECT COUNT(*) FROM {schema}.{table}"))
            count = result.fetchone()[0]
            print(f"    {table}: {count} rows")


def main():
    parser = argparse.ArgumentParser(description="PS-14 PostgreSQL Migration")
    parser.add_argument("--database-url", help="PostgreSQL connection URL")
    parser.add_argument("--schema-only", action="store_true",
                       help="Create schemas and tables only (no data migration)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing migration")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for data migration (default: 1000)")
    args = parser.parse_args()

    print("=" * 70)
    print("PS-14 PostgreSQL Migration")
    print("=" * 70)

    engine = get_engine(args.database_url)
    print(f"\nConnected to PostgreSQL")

    if args.verify_only:
        verify_migration(engine)
        return

    # Create schemas
    create_schemas(engine)

    # Create tables for each schema
    for db_name, info in DB_MAPPING.items():
        create_tables(engine, info["schema"], info["models_module"])

    # Create audit triggers
    create_audit_triggers(engine)

    if args.schema_only:
        print("\n✓ Schema-only migration complete")
        return

    # Migrate data
    print("\n[4] Migrating data...")
    start_time = time.time()
    total_migrated = 0

    for db_name, info in DB_MAPPING.items():
        sqlite_path = ROOT / "db" / info["sqlite_file"]
        migrated = migrate_data(engine, db_name, sqlite_path, info["schema"], args.batch_size)
        total_migrated += migrated

    elapsed = time.time() - start_time
    print(f"\n  Total rows migrated: {total_migrated}")
    print(f"  Time: {elapsed:.1f}s")

    # Verify
    verify_migration(engine)

    print("\n" + "=" * 70)
    print("Migration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
