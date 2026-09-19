"""Database helpers for PS-14 PostgreSQL persistence.

Provides:
- Database-level idempotency via ON CONFLICT
- Connection pool health monitoring
- Safe persistence patterns
- Transaction boundary helpers

Preserves existing SQLite compatibility while enabling PostgreSQL features.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import time
import threading
from typing import Any

from sqlalchemy import text, inspect as sa_inspect
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool


# ── Idempotency ─────────────────────────────────────────────────────────

def insert_risk_score_idempotent(
    db: Session,
    *,
    fraud_id: str,
    event_id: str,
    risk_score: int,
    risk_band: str,
    reason_codes: list[str],
    model_version: str,
    ml_score: float,
    rule_score: float,
    degraded: bool,
) -> tuple[bool, dict[str, Any]]:
    """Insert a risk score with database-level idempotency.

    Uses PostgreSQL ON CONFLICT to guarantee exactly-once semantics.
    Falls back to application-level check for SQLite.

    Returns:
        (inserted, existing_or_new_record_dict)
        inserted=True if this call created the record
        inserted=False if the event_id already existed (idempotent replay)
    """
    from src.settings import settings
    from src.risk_engine.models import RiskScore

    if settings.use_postgres:
        # PostgreSQL: use INSERT ... ON CONFLICT for atomic idempotency
        result = db.execute(
            text("""
                INSERT INTO risk_scores (
                    score_id, fraud_id, event_id, risk_score, risk_band,
                    reason_codes, model_version, ml_score, rule_score, degraded
                ) VALUES (
                    gen_random_uuid()::text, :fraud_id, :event_id, :risk_score,
                    :risk_band, :reason_codes, :model_version, :ml_score,
                    :rule_score, :degraded
                )
                ON CONFLICT (event_id) DO NOTHING
                RETURNING score_id, event_id
            """),
            {
                "fraud_id": fraud_id,
                "event_id": event_id,
                "risk_score": risk_score,
                "risk_band": risk_band,
                "reason_codes": json.dumps(reason_codes),
                "model_version": model_version,
                "ml_score": ml_score,
                "rule_score": rule_score,
                "degraded": degraded,
            },
        )
        row = result.fetchone()
        db.commit()

        if row is not None:
            # Record was inserted
            return True, {
                "score_id": row[0],
                "event_id": row[1],
                "risk_score": risk_score,
                "risk_band": risk_band,
                "inserted": True,
            }
        else:
            # Record already existed (ON CONFLICT DO NOTHING)
            existing = db.query(RiskScore).filter(
                RiskScore.event_id == event_id
            ).first()
            return False, {
                "score_id": existing.score_id if existing else None,
                "event_id": event_id,
                "risk_score": existing.risk_score if existing else 0,
                "risk_band": existing.risk_band if existing else "unknown",
                "inserted": False,
            }
    else:
        # SQLite: application-level idempotency
        existing = db.query(RiskScore).filter(
            RiskScore.event_id == event_id
        ).first()

        if existing is not None:
            return False, {
                "score_id": existing.score_id,
                "event_id": event_id,
                "risk_score": existing.risk_score,
                "risk_band": existing.risk_band,
                "inserted": False,
            }

        try:
            score_record = RiskScore(
                fraud_id=fraud_id,
                event_id=event_id,
                risk_score=risk_score,
                risk_band=risk_band,
                reason_codes=json.dumps(reason_codes),
                model_version=model_version,
                ml_score=ml_score,
                rule_score=rule_score,
                degraded=degraded,
            )
            db.add(score_record)
            db.commit()
            return True, {
                "score_id": score_record.score_id,
                "event_id": event_id,
                "risk_score": risk_score,
                "risk_band": risk_band,
                "inserted": True,
            }
        except Exception:
            db.rollback()
            # Race condition: another thread inserted first
            existing = db.query(RiskScore).filter(
                RiskScore.event_id == event_id
            ).first()
            return False, {
                "score_id": existing.score_id if existing else None,
                "event_id": event_id,
                "risk_score": existing.risk_score if existing else 0,
                "risk_band": existing.risk_band if existing else "unknown",
                "inserted": False,
            }


# ── Connection Pool Health ──────────────────────────────────────────────

def get_pool_health(engine) -> dict[str, Any]:
    """Get connection pool health status.

    Returns a dictionary suitable for health/readiness endpoints.
    Handles StaticPool (in-memory SQLite) and NullPool (on-disk SQLite)
    as well as QueuePool (PostgreSQL).
    """
    try:
        pool = engine.pool
        pool_class = pool.__class__.__name__
        status = {
            "status": "healthy",
            "pool_class": pool_class,
        }

        # Add pool stats for pool types that support them
        if hasattr(pool, 'size'):
            status.update({
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total_connections": pool.checkedin() + pool.checkedout(),
            })

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            status["connectivity"] = "ok"

        return status
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)[:100],  # Truncate error to avoid leaking secrets
            "connectivity": "failed",
        }


def test_database_connection(engine) -> tuple[bool, str]:
    """Test database connection health.

    Returns:
        (is_healthy, status_message)
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "connection_ok"
    except Exception as e:
        error_msg = str(e)[:100]  # Truncate to avoid leaking secrets
        return False, f"connection_failed: {error_msg}"


# ── Audit Chain Helpers ─────────────────────────────────────────────────

def verify_audit_chain_integrity(db: Session) -> dict[str, Any]:
    """Verify the audit chain integrity.

    Returns a dictionary with chain status and statistics.
    """
    from src.audit_service.models import AuditEvent
    from src.audit_service.writer import canonical, GENESIS_HASH

    try:
        # Get all audit events in order
        events = db.query(AuditEvent).order_by(AuditEvent.seq.asc()).all()

        if not events:
            return {
                "status": "empty",
                "entries": 0,
                "chain_valid": True,
            }

        prev_hash = GENESIS_HASH
        bad_seq = None

        for event in events:
            try:
                body = canonical(json.loads(event.payload_summary))
                expected_hash = hashlib.sha256(
                    (prev_hash + body).encode("utf-8")
                ).hexdigest()

                if event.prev_hash != prev_hash or event.entry_hash != expected_hash:
                    bad_seq = event.seq
                    break

                prev_hash = event.entry_hash
            except (json.JSONDecodeError, TypeError):
                bad_seq = event.seq
                break

        return {
            "status": "valid" if bad_seq is None else "broken",
            "entries": len(events),
            "chain_valid": bad_seq is None,
            "first_bad_seq": bad_seq,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)[:100],
            "chain_valid": False,
        }


# ── Schema Version Tracking ─────────────────────────────────────────────

_SCHEMA_VERSION = "001"

def get_schema_version(db: Session) -> dict[str, Any]:
    """Get the current schema version."""
    try:
        # Try to read from alembic_version table
        result = db.execute(
            text("SELECT version_num FROM alembic_version LIMIT 1")
        )
        row = result.fetchone()
        if row:
            return {
                "version": row[0],
                "source": "alembic",
                "status": "ok",
            }
    except Exception:
        pass

    # Fallback: check if tables exist
    try:
        inspector = sa_inspect(db.get_bind())
        tables = inspector.get_table_names()
        return {
            "version": _SCHEMA_VERSION,
            "source": "create_all",
            "tables": len(tables),
            "status": "ok",
        }
    except Exception as e:
        return {
            "version": "unknown",
            "source": "error",
            "status": str(e)[:100],
        }
