"""DB-4 schema (architecture section 13): hash-chained, tamper-evident audit
events. Each entry stores `prev_hash` (the previous entry's `entry_hash`) and
`entry_hash = sha256(prev_hash + canonical(payload))`, so any modification
breaks the chain and is detectable by `GET /audit/integrity`.

Append-only is enforced at the storage layer with SQLite triggers that abort
any UPDATE or DELETE on `audit_events`.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, event, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AuditEvent(Base):
    __tablename__ = "audit_events"

    # `seq` is the chain order (autoincrement), separate from the event id.
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    fraud_id: Mapped[str] = mapped_column(String(16), index=True)  # pseudonymous only
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    entry_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    payload_summary: Mapped[str] = mapped_column(Text, nullable=False)  # JSON, no PII
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class AuditAccessLog(Base):
    """Who read the audit trail and when (compliance access is itself logged)."""

    __tablename__ = "audit_access_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    query_summary: Mapped[str] = mapped_column(Text, default="")
    accessed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


@event.listens_for(Base.metadata, "after_create")
def _append_only_triggers(target, connection, **kw):  # pragma: no cover - DDL
    # Detect database dialect for appropriate trigger syntax
    dialect = connection.dialect.name

    if dialect == "sqlite":
        # SQLite triggers
        for op in ("UPDATE", "DELETE"):
            name = "no_update" if op == "UPDATE" else "no_delete"
            connection.exec_driver_sql(
                f"""
                CREATE TRIGGER IF NOT EXISTS audit_events_{name}
                BEFORE {op} ON audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'audit_events is append-only: {op.lower()} blocked');
                END
                """
            )
    elif dialect == "postgresql":
        # PostgreSQL triggers
        connection.exec_driver_sql("""
            CREATE OR REPLACE FUNCTION prevent_audit_update()
            RETURNS TRIGGER AS $
            BEGIN
                RAISE EXCEPTION 'audit_events is append-only: UPDATE blocked';
                RETURN NULL;
            END;
            $$ LANGUAGE plpgsql;

            CREATE OR REPLACE FUNCTION prevent_audit_delete()
            RETURNS TRIGGER AS $
            BEGIN
                RAISE EXCEPTION 'audit_events is append-only: DELETE blocked';
                RETURN NULL;
            END;
            $$ LANGUAGE plpgsql;
        """)
        for op in ("UPDATE", "DELETE"):
            name = "no_update" if op == "UPDATE" else "no_delete"
            connection.exec_driver_sql(f"""
                DROP TRIGGER IF EXISTS audit_events_{name} ON audit_events;
                CREATE TRIGGER audit_events_{name}
                    BEFORE {op} ON audit_events
                    FOR EACH ROW
                    EXECUTE FUNCTION prevent_audit_{name.replace('no_', '')}();
            """)
