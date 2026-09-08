"""DB-3 schema (architecture section 13): risk scores keyed by fraud_id and
event_id only. No PII, no raw features - just the decision trail.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RiskScore(Base):
    __tablename__ = "risk_scores"
    __table_args__ = (
        CheckConstraint("risk_score BETWEEN 0 AND 100", name="ck_risk_score_range"),
    )

    score_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    fraud_id: Mapped[str] = mapped_column(String(16), index=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True, unique=True)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_band: Mapped[str] = mapped_column(String(16), nullable=False)  # low | medium | high
    reason_codes: Mapped[str] = mapped_column(Text, nullable=False)  # JSON list, category-level only
    model_version: Mapped[str] = mapped_column(String(64))
    ml_score: Mapped[float] = mapped_column(Float)
    rule_score: Mapped[float] = mapped_column(Float)
    # True when ML fusion was unavailable and the decision came from the
    # declarative rules alone (fail-open degraded mode).
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    @property
    def codes(self) -> list[str]:
        return json.loads(self.reason_codes)
