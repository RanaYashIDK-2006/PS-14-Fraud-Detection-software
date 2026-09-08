"""Shared database factory — eliminates duplicated db.py across services.

Each service calls `make_engine(db_path)` or `make_pg_engine(schema)`
instead of copying the same SQLAlchemy boilerplate.

Usage in a service's db.py:

    from src.shared_db import make_engine, make_session_local
    engine = make_engine(settings.identity_db_path)
    SessionLocal = make_session_local(engine)

Production PostgreSQL usage:

    from src.shared_db import make_pg_engine, make_session_local
    engine = make_pg_engine(settings.database_url, settings.db_schema)
    SessionLocal = make_session_local(engine)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, StaticPool


def make_engine(db_path: str | Path) -> Engine:
    """Create a SQLite engine with WAL mode for a local dev database.

    Pooling: file-backed databases use NullPool (a fresh connection per
    checkout). StaticPool hands the SAME underlying SQLite connection to
    every concurrent session, so parallel FastAPI request threads mutate one
    cursor -> sqlite3.InterfaceError ("bad parameter or other API misuse")
    on INSERT...RETURNING and drop ~1/3 of requests at >4 workers
    (perf/reliability audit, check #26). With NullPool each request thread
    owns its connection; WAL + busy_timeout (set below) keep readers and the
    single writer safe. StaticPool is retained only for in-memory databases,
    where one connection must stay alive for tables to persist across
    sessions.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    is_memory = str(db_path) == ":memory:"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool if is_memory else NullPool,
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn: Any, _connection_record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine


def make_pg_engine(
    database_url: str,
    schema: str,
    use_pooler: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    pool_recycle: int = 1800,
    pool_pre_ping: bool = True,
) -> Engine:
    """Create a PostgreSQL engine with schema isolation and connection pooling.

    Args:
        database_url: PostgreSQL connection string
        schema: Schema name for this service (identity, privacy, risk, audit)
        use_pooler: True for Supabase/Neon pooled connections (NullPool)
        pool_size: Persistent connections in the pool
        max_overflow: Additional connections beyond pool_size
        pool_timeout: Seconds to wait for a connection from the pool
        pool_recycle: Seconds before recycling a connection (prevents stale connections)
        pool_pre_ping: Test connections before using them

    Returns:
        SQLAlchemy Engine configured for PostgreSQL
    """
    # Set schema via search_path
    connect_args = {"options": f"-c search_path={schema},public"}

    if use_pooler:
        # Transaction-mode poolers (Supabase/Neon port 6543) don't support
        # prepared statements, so we use NullPool
        from sqlalchemy.pool import NullPool
        return create_engine(
            database_url,
            poolclass=NullPool,
            connect_args=connect_args,
            pool_pre_ping=pool_pre_ping,
        )

    return create_engine(
        database_url,
        pool_pre_ping=pool_pre_ping,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        connect_args=connect_args,
    )


def make_session_local(engine: Engine) -> sessionmaker:
    """Create a SessionLocal factory bound to the given engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def make_get_db(SessionLocal: sessionmaker):
    """Return a FastAPI dependency that yields a DB session."""
    def _get_db():
        db: Session = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    return _get_db


def ensure_schema(engine: Engine, schema: str) -> None:
    """Create schema if it doesn't exist (idempotent)."""
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        conn.commit()


def create_all_tables(engine: Engine, metadata) -> None:
    """Create all tables defined in the metadata."""
    metadata.create_all(bind=engine)


def get_pool_status(engine: Engine) -> dict[str, Any]:
    """Get connection pool status for monitoring."""
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total_connections": pool.checkedin() + pool.checkedout(),
    }


def test_connection(engine: Engine) -> bool:
    """Test database connection health."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
