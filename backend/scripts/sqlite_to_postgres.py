#!/usr/bin/env python3
"""SQLite to PostgreSQL Data Migration for PS-14

Migrates data from SQLite databases to PostgreSQL with per-service schema
isolation. Handles all 5 databases and their tables.

Usage:
    # Set DATABASE_URL
    export DATABASE_URL=postgresql://user:pass@localhost:5432/ps14

    # Migrate all databases
    python scripts/sqlite_to_postgres.py

    # Migrate specific database
    python scripts/sqlite_to_postgres.py --db identity
    python scripts/sqlite_to_postgres.py --db privacy
    python scripts/sqlite_to_postgres.py --db risk
    python scripts/sqlite_to_postgres.py --db audit
    python scripts/sqlite_to_postgres.py --db verification

    # Dry run (show what would be migrated)
    python scripts/sqlite_to_postgres.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker


# Database mappings: SQLite file -> PostgreSQL schema
DB_MAPPING = {
    "identity": {
        "sqlite_file": "identity.db",
        "schema": "identity",
        "description": "DB-1: Identity Store (PII, auth, pseudonyms)",
    },
    "privacy": {
        "sqlite_file": "features.db",
        "schema": "privacy",
        "description": "DB-2: Feature Store (pseudonymous behavioral data)",
    },
    "risk": {
        "sqlite_file": "risk.db",
        "schema": "risk",
        "description": "DB-3: Risk & Model Store (scores, decisions)",
    },
    "audit": {
        "sqlite_file": "audit.db",
        "schema": "audit",
        "description": "DB-4: Audit Store (hash-chained events)",
    },
    "verification": {
        "sqlite_file": "verify.db",
        "schema": "verification",
        "description": "DB-5: Verification Store (outcomes, cases)",
    },
}


def get_pg_engine(database_url: str | None = None):
    """Create PostgreSQL engine."""
    url = database_url or os.environ.get("DATABASE_URL", "")
    if not url or not url.startswith("postgresql"):
        print("ERROR: Set DATABASE_URL to a PostgreSQL connection string")
        sys.exit(1)
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)


def get_sqlite_path(db_name: str) -> Path:
    """Get SQLite database path."""
    db_dir = ROOT / "db"
    sqlite_file = DB_MAPPING[db_name]["sqlite_file"]
    return db_dir / sqlite_file


def count_sqlite_rows(sqlite_path: Path) -> dict[str, int]:
    """Count rows in all tables of a SQLite database."""
    if not sqlite_path.exists():
        return {}

    conn = sqlite3.connect(str(sqlite_path))
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    counts = {}
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
            counts[table] = cursor.fetchone()[0]
        except Exception:
            counts[table] = 0

    conn.close()
    return counts


def migrate_table_data(
    sqlite_path: Path,
    pg_engine,
    schema: str,
    table_name: str,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> int:
    """Migrate data from a SQLite table to PostgreSQL."""
    if not sqlite_path.exists():
        print(f"    SQLite file not found: {sqlite_path}")
        return 0

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    if not cursor.fetchone():
        conn.close()
        return 0

    # Get column names
    cursor.execute(f"PRAGMA table_info([{table_name}])")
    columns = [row[1] for row in cursor.fetchall()]

    # Count rows
    cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
    total_rows = cursor.fetchone()[0]

    if total_rows == 0:
        conn.close()
        return 0

    if dry_run:
        print(f"    Would migrate {total_rows} rows from {table_name}")
        conn.close()
        return total_rows

    print(f"    Migrating {total_rows} rows from {table_name}...")

    # Build INSERT statement
    placeholders = ", ".join([f":{col}" for col in columns])
    insert_sql = f"""
        INSERT INTO {schema}.{table_name} ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT DO NOTHING
    """

    # Migrate in batches
    migrated = 0
    cursor.execute(f"SELECT * FROM [{table_name}]")
    pg_conn = pg_engine.connect()

    try:
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            # Convert Row objects to dicts
            batch = [dict(row) for row in rows]

            # Execute batch insert
            pg_conn.execute(text(insert_sql), batch)
            migrated += len(batch)

            # Progress indicator
            if migrated % 10000 == 0 or migrated == total_rows:
                print(f"      {migrated}/{total_rows} rows ({100*migrated//total_rows}%)")

        pg_conn.commit()
    except Exception as e:
        pg_conn.rollback()
        print(f"    ERROR migrating {table_name}: {e}")
        raise
    finally:
        pg_conn.close()

    conn.close()
    return migrated


def migrate_database(
    db_name: str,
    pg_engine,
    dry_run: bool = False,
) -> dict[str, int]:
    """Migrate a single SQLite database to PostgreSQL."""
    info = DB_MAPPING[db_name]
    sqlite_path = get_sqlite_path(db_name)
    schema = info["schema"]

    print(f"\n{'='*60}")
    print(f"Migrating: {info['description']}")
    print(f"  SQLite: {sqlite_path}")
    print(f"  PostgreSQL schema: {schema}")
    print(f"{'='*60}")

    if not sqlite_path.exists():
        print(f"  SQLite file not found: {sqlite_path}")
        return {}

    # Count source rows
    source_counts = count_sqlite_rows(sqlite_path)
    print(f"\n  Source tables:")
    for table, count in source_counts.items():
        print(f"    {table}: {count} rows")

    if dry_run:
        print("\n  DRY RUN - no data will be migrated")
        return source_counts

    # Get tables from PostgreSQL schema
    metadata = MetaData(schema=schema)
    metadata.reflect(bind=pg_engine)
    pg_tables = list(metadata.tables.keys())

    print(f"\n  PostgreSQL tables: {', '.join(pg_tables)}")

    # Migrate each table
    migration_counts = {}
    for table_name in pg_tables:
        if table_name in source_counts:
            count = migrate_table_data(
                sqlite_path, pg_engine, schema, table_name, dry_run=dry_run
            )
            migration_counts[table_name] = count

    # Verify migration
    print(f"\n  Verification:")
    for table_name, expected_count in source_counts.items():
        if table_name in migration_counts:
            actual_count = migration_counts[table_name]
            status = "✓" if actual_count == expected_count else "✗"
            print(f"    {status} {table_name}: expected {expected_count}, got {actual_count}")

    return migration_counts


def verify_migration(pg_engine, schema: str):
    """Verify migration was successful."""
    print(f"\n  Verifying schema: {schema}")

    with pg_engine.connect() as conn:
        # List tables
        result = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = :schema ORDER BY table_name"
        ), {"schema": schema})
        tables = [row[0] for row in result.fetchall()]

        print(f"    Tables: {', '.join(tables)}")

        # Count rows in each table
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{table}"))
                count = result.fetchone()[0]
                print(f"      {table}: {count} rows")
            except Exception as e:
                print(f"      {table}: ERROR - {e}")


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite databases to PostgreSQL")
    parser.add_argument("--db", choices=list(DB_MAPPING.keys()),
                       help="Migrate only this database (default: all)")
    parser.add_argument("--database-url", help="PostgreSQL connection URL")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be migrated without doing it")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for inserts (default: 1000)")
    args = parser.parse_args()

    print("=" * 70)
    print("PS-14 SQLite → PostgreSQL Migration")
    print("=" * 70)

    # Get PostgreSQL engine
    pg_engine = get_pg_engine(args.database_url)
    print(f"\nConnected to PostgreSQL")

    # Determine which databases to migrate
    databases = [args.db] if args.db else list(DB_MAPPING.keys())

    # Migrate each database
    start_time = time.time()
    total_migrated = 0

    for db_name in databases:
        counts = migrate_database(db_name, pg_engine, dry_run=args.dry_run)
        total_migrated += sum(counts.values())

    elapsed = time.time() - start_time

    # Verify
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    for db_name in databases:
        schema = DB_MAPPING[db_name]["schema"]
        verify_migration(pg_engine, schema)

    # Summary
    print("\n" + "=" * 70)
    print("Migration Summary")
    print("=" * 70)
    print(f"  Databases: {len(databases)}")
    print(f"  Total rows: {total_migrated}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'COMPLETED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
