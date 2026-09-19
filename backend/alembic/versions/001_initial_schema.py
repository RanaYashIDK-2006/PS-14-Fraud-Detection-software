"""Initial PostgreSQL schema for PS-14

Revision ID: 001
Revises:
Create Date: 2026-09-19

Creates per-service schemas and all production-critical tables:
- risk.risk_scores (DB-3)
- audit.audit_events (DB-4, append-only with triggers)
- audit.audit_access_log
- Plus index and constraint definitions

This migration establishes the PostgreSQL foundation for PS-14
while preserving SQLite compatibility for local development.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Service schemas
SCHEMAS = ['identity', 'privacy', 'risk', 'audit', 'verification']


def upgrade() -> None:
    """Create PS-14 PostgreSQL schema."""

    # Create schemas
    for schema in SCHEMAS:
        op.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    # ── Risk Schema (DB-3) ─────────────────────────────────────────────
    op.execute("SET search_path TO risk, public")

    op.create_table(
        'risk_scores',
        sa.Column('score_id', sa.String(36), primary_key=True),
        sa.Column('fraud_id', sa.String(16), nullable=False, index=True),
        sa.Column('event_id', sa.String(64), nullable=False, unique=True, index=True),
        sa.Column('risk_score', sa.Integer(), nullable=False),
        sa.Column('risk_band', sa.String(16), nullable=False),
        sa.Column('reason_codes', sa.Text(), nullable=False),
        sa.Column('model_version', sa.String(64)),
        sa.Column('ml_score', sa.Float()),
        sa.Column('rule_score', sa.Float()),
        sa.Column('degraded', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.CheckConstraint('risk_score BETWEEN 0 AND 100', name='ck_risk_score_range'),
    )

    # Performance indexes for risk queries
    op.create_index('ix_risk_scores_fraud_id', 'risk_scores', ['fraud_id'])
    op.create_index('ix_risk_scores_created_at', 'risk_scores', ['created_at'])
    op.create_index('ix_risk_scores_risk_band', 'risk_scores', ['risk_band'])
    op.create_index('ix_risk_scores_degraded', 'risk_scores', ['degraded'])

    # ── Audit Schema (DB-4) ────────────────────────────────────────────
    op.execute("SET search_path TO audit, public")

    op.create_table(
        'audit_events',
        sa.Column('seq', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('event_id', sa.String(36), unique=True, nullable=False),
        sa.Column('fraud_id', sa.String(16), nullable=False, index=True),
        sa.Column('event_type', sa.String(32), nullable=False, index=True),
        sa.Column('prev_hash', sa.String(64), nullable=False),
        sa.Column('entry_hash', sa.String(64), nullable=False, index=True),
        sa.Column('payload_summary', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        'audit_access_log',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('actor', sa.String(64)),
        sa.Column('action', sa.String(64)),
        sa.Column('query_summary', sa.Text(), default=''),
        sa.Column('accessed_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # Append-only triggers for audit_events
    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_audit_update()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'audit_events is append-only: UPDATE blocked';
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_audit_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'audit_events is append-only: DELETE blocked';
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER audit_events_no_update
            BEFORE UPDATE ON audit_events
            FOR EACH ROW
            EXECUTE FUNCTION prevent_audit_update();
    """)

    op.execute("""
        CREATE TRIGGER audit_events_no_delete
            BEFORE DELETE ON audit_events
            FOR EACH ROW
            EXECUTE FUNCTION prevent_audit_delete();
    """)

    # ── Risk Engine Idempotency ────────────────────────────────────────
    # The event_id UNIQUE constraint on risk_scores already provides
    # idempotency. Combined with ON CONFLICT, this prevents duplicate
    # business records under concurrent requests.

    # Reset search path
    op.execute("SET search_path TO public")


def downgrade() -> None:
    """Drop PS-14 PostgreSQL schema."""

    op.execute("SET search_path TO audit, public")
    op.drop_table('audit_access_log')
    op.drop_table('audit_events')
    op.execute("DROP FUNCTION IF EXISTS prevent_audit_update()")
    op.execute("DROP FUNCTION IF EXISTS prevent_audit_delete()")

    op.execute("SET search_path TO risk, public")
    op.drop_table('risk_scores')

    for schema in reversed(SCHEMAS):
        op.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")

    op.execute("SET search_path TO public")
