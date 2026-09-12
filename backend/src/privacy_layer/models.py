"""DB-2 schema (architecture section 13): pseudonymous behavioral data only.

No name, phone, email, account number, raw amount, exact geo, or device ID
ever appears in this store. Device IDs arrive pre-hashed; locations and
recipients are coarse, opaque tokens assigned outside the store.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FraudProfile(Base):
    """Statistical summary of an account's behavior — not raw events.

    Lists below are JSON-encoded (SQLite lacks native arrays; PostgreSQL uses
    INT[] / JSONB per section 13).
    """

    __tablename__ = "fraud_profiles"

    fraud_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    avg_txn_amount_90d: Mapped[float] = mapped_column(Float, nullable=False)  # EMA of amount
    txn_freq_7d: Mapped[int] = mapped_column(Integer, default=0)
    typical_txn_hours: Mapped[str] = mapped_column(Text, nullable=False)  # JSON int[]
    known_device_count: Mapped[int] = mapped_column(Integer, default=0)
    usual_locations: Mapped[str] = mapped_column(Text, nullable=False)  # JSON list
    usual_recipients: Mapped[str] = mapped_column(Text, nullable=False)  # JSON list
    behavioral_baseline_vector: Mapped[str] = mapped_column(Text, nullable=False)  # JSON summary
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_updated: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    @property
    def median_amount(self) -> float:
        return float(self.avg_txn_amount_90d)

    @property
    def typical_hours(self) -> list[int]:
        return json.loads(self.typical_txn_hours)

    @property
    def locations(self) -> list[str]:
        return json.loads(self.usual_locations)

    @property
    def recipients(self) -> list[str]:
        return json.loads(self.usual_recipients)


class TransactionFeature(Base):
    """One derived feature vector per event. Keyed by fraud_id only."""

    __tablename__ = "transaction_features"
    __table_args__ = (
        # Composite index for 24h velocity queries (freq, spend, device count)
        Index("ix_tf_fraud_ts", "fraud_id", "created_at"),
        # Index for device_daily_count (device_hash + timestamp range)
        Index("ix_tf_device_ts", "device_hash", "created_at"),
        # Index for recipient_id link-analysis query
        Index("ix_tf_recipient", "recipient_id"),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    fraud_id: Mapped[str] = mapped_column(ForeignKey("fraud_profiles.fraud_id"), index=True)
    amount_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    txn_amount_bucket: Mapped[str] = mapped_column(String(32), nullable=False)
    txn_freq_last_24h: Mapped[int] = mapped_column(Integer, default=0)
    txn_time_unusual: Mapped[bool] = mapped_column(Boolean, default=False)
    new_device_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    unusual_location_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    unusual_recipient_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    failed_auth_count_24h: Mapped[int] = mapped_column(Integer, default=0)
    days_since_last_similar_txn: Mapped[float] = mapped_column(Float, nullable=False)
    gradual_escalation_score: Mapped[float] = mapped_column(Float, default=0.0)
    known_device_count: Mapped[int] = mapped_column(Integer, default=0)
    account_tenure_days: Mapped[float] = mapped_column(Float, default=1.0)
    hour_of_day: Mapped[int] = mapped_column(Integer)
    is_weekend: Mapped[bool] = mapped_column(Boolean, default=False)
    # Cross-account link-analysis signals (device graphs / mule rings): counts
    # of OTHER accounts seen on this device / recipient at ingest time.
    shared_device_accounts: Mapped[int] = mapped_column(Integer, default=0)
    shared_recipient_accounts: Mapped[int] = mapped_column(Integer, default=0)
    mule_ring_score: Mapped[float] = mapped_column(Float, default=0.0)
    # Extended features (v3 retrain on kartik2112)
    hour_deviation: Mapped[float] = mapped_column(Float, default=0.0)
    amount_zscore: Mapped[float] = mapped_column(Float, default=0.0)
    velocity_deviation: Mapped[float] = mapped_column(Float, default=0.0)
    recipient_novelty: Mapped[float] = mapped_column(Float, default=0.0)
    txn_regularity: Mapped[float] = mapped_column(Float, default=0.0)

    # Deferred-baseline bookkeeping: the statistical profile (median,
    # typical hours) is updated ONLY from events committed as allowed or
    # confirmed, so a blocked/disputed event can never pollute the account's
    # baseline. The coarse identifiers are stored so the commit can register
    # the device/location/recipient at decision time (hashed device, opaque
    # coarse tokens - never raw IDs or PII, per section 16).
    baseline_committed: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    device_hash: Mapped[str] = mapped_column(String(32), nullable=True)
    location_id: Mapped[str] = mapped_column(String(64), nullable=True)
    recipient_id: Mapped[str] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class DeviceFingerprint(Base):
    """Hashed device identifiers only (SHA-256 of raw device id) — section 2.

    Composite PK (device_hash, fraud_id): multiple accounts can share
    one device.  The sole-PK-on-device_hash schema was a bug that caused
    cross-account bleed (only the first account's row was stored).
    """

    __tablename__ = "device_fingerprints"
    __table_args__ = (
        # Index for cross-account device lookup (other accounts sharing this device)
        Index("ix_df_fraud", "fraud_id"),
    )

    device_hash: Mapped[str] = mapped_column(String(32), primary_key=True)
    fraud_id: Mapped[str] = mapped_column(String(64), ForeignKey("fraud_profiles.fraud_id"), primary_key=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
