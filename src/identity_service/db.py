"""DB-1: Identity Store. Uses shared database factory."""

from __future__ import annotations

from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from src.settings import settings
from src.shared_db import (
    make_engine,
    make_pg_engine,
    make_session_local,
    ensure_schema,
    create_all_tables,
)

if settings.use_postgres:
    engine = make_pg_engine(
        settings.database_url, settings.db_schema,
        use_pooler=settings.use_pg_pooler,
        pool_size=10,
        max_overflow=20,
    )
    ensure_schema(engine, settings.db_schema)
    from src.identity_service.models import Base
    create_all_tables(engine, Base.metadata)
else:
    engine = make_engine(settings.identity_db_path)

SessionLocal: sessionmaker = make_session_local(engine)
