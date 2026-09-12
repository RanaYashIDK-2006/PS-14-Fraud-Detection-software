"""DB-1 schema (architecture section 13): the ONLY place PII lives and the
ONLY place fraud_id -> user_id can be resolved. `pseudonym_mapping` is
accessible only to a small, logged, break-glass role - never to the fraud
pipeline, analysts, or ML jobs.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # PII is column-encrypted (BYTEA/BLOB at rest); never plaintext.
    # full_name / account_number / account_type were plaintext until the
    # privacy audit (#25) - migrated by scripts/migrate_identity_pii.py.
    full_name: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    phone_encrypted: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    email_encrypted: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Blind index (HMAC-style) so login can look up by email without ever
    # storing or querying the plaintext. Standard pattern for searchable
    # encryption.
    email_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    kyc_doc_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Account details (user-editable with proper authentication)
    account_number: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    account_type: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)  # savings, current, fixed_deposit
    address_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    # RBAC: role determines which endpoints the user can access.
    # "user" (default) → own profile only; "analyst" → pseudonymous
    # risk data (no PII); "admin" → user management + everything.
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class AuthCredential(Base):
    __tablename__ = "auth_credentials"

    credential_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    credential_type: Mapped[str] = mapped_column(String(32), default="password")
    secret_hash: Mapped[str] = mapped_column(Text, nullable=False)  # Argon2id
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class PseudonymMapping(Base):
    """The single real<->pseudonym bridge. CSPRNG fraud_id, never a hash of
    PII (a hash of an email is reversible via rainbow tables; a random ID is
    not - architecture section 3)."""

    __tablename__ = "pseudonym_mapping"

    fraud_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    rotation_group: Mapped[int] = mapped_column(Integer, default=1)
    access_log_required: Mapped[bool] = mapped_column(Boolean, default=True)


class PseudonymAccessLog(Base):
    """Break-glass logging: every resolution of fraud_id -> user_id is
    recorded with actor and reason (section 3 RBAC table)."""

    __tablename__ = "pseudonym_access_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fraud_id: Mapped[str] = mapped_column(String(16), index=True)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text, default="")
    accessed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
