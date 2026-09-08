"""DB-3 schema addition: verification outcomes + investigator cases (section 13).

    verification_id   UUID PRIMARY KEY
    score_id          UUID REFERENCES risk_scores(score_id)
    outcome           'confirmed' | 'disputed'
    resolved_at       TIMESTAMPTZ

    case_id (investigator)  UUID PRIMARY KEY
    status                  NEW | REVIEWING | USER_VERIFICATION | CONFIRMED_SUSPICIOUS | CONFIRMED_LEGITIMATE | CLOSED
    investigator_id         pseudonymous analyst ID (never real identity)
    confidence              model uncertainty at time of case creation
    priority                computed score * confidence * amount tier

Reuses the Risk Engine's Base so both services create/read the same tables
in the shared store.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from src.risk_engine.models import Base as RiskBase

# Valid case states for the investigator workflow
CASE_STATES = (
    "NEW",           # just created, awaiting triage
    "REVIEWING",     # investigator actively reviewing
    "USER_VERIFICATION",  # user verification requested
    "CONFIRMED_SUSPICIOUS",
    "CONFIRMED_LEGITIMATE",
    "CLOSED",
)

# Valid state transitions
CASE_TRANSITIONS = {
    "NEW": ("REVIEWING", "CONFIRMED_LEGITIMATE", "CLOSED"),
    "REVIEWING": ("USER_VERIFICATION", "CONFIRMED_SUSPICIOUS", "CONFIRMED_LEGITIMATE", "CLOSED"),
    "USER_VERIFICATION": ("CONFIRMED_SUSPICIOUS", "CONFIRMED_LEGITIMATE", "CLOSED"),
    "CONFIRMED_SUSPICIOUS": ("CLOSED",),
    "CONFIRMED_LEGITIMATE": ("CLOSED",),
    "CLOSED": (),
}


class VerificationOutcome(RiskBase):
    __tablename__ = "verification_outcomes"

    verification_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    score_id: Mapped[str] = mapped_column(ForeignKey("risk_scores.score_id"), index=True)
    fraud_id: Mapped[str] = mapped_column(String(16), index=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)  # confirmed | disputed
    case_id: Mapped[str] = mapped_column(String(32))
    resolved_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class InvestigatorCase(RiskBase):
    """Investigator case workflow: NEW → REVIEWING → USER_VERIFICATION →
    CONFIRMED_SUSPICIOUS / CONFIRMED_LEGITIMATE → CLOSED.

    Each transition is audited. Investigators see pseudonymous ID + score +
    confidence + reason codes + prior related alerts — never real identity
    outside break-glass.

    One case per event: ``event_id`` is UNIQUE so retries/replays can never
    create duplicate cases (alert-quality audit #30). Created in the shared
    risk store (DB-3) next to the risk_scores it references.
    """
    __tablename__ = "investigator_cases"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_investigator_cases_event_id"),
    )

    case_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    fraud_id: Mapped[str] = mapped_column(String(16), index=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    score_id: Mapped[str] = mapped_column(ForeignKey("risk_scores.score_id"), index=True)
    status: Mapped[str] = mapped_column(String(24), default="NEW")
    investigator_id: Mapped[str | None] = mapped_column(String(16), nullable=True)  # pseudonymous
    confidence: Mapped[str] = mapped_column(String(8), default="medium")  # high/medium/low from uncertainty
    priority: Mapped[float] = mapped_column(default=0.0)  # score * confidence_weight * amount_tier
    risk_score: Mapped[int] = mapped_column(default=0)
    reason_codes: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
