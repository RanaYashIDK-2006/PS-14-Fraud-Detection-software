"""Phase 81: Outcome trust governance.

Makes production-trust outcome recording subject to explicit authorization
and provenance controls.  Not every caller that can record an outcome is
automatically trusted to assert a production-grade label.

The trust policy is declarative, deterministic, and versioned.  Historical
outcomes remain interpretable after future policy changes because the policy
version is recorded with the outcome.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.risk_engine.models import Base
from src.monitoring.outcome_pipeline import (
    OutcomeRecord,
    OutcomeSource,
    SOURCE_TRUST,
    PRODUCTION_SOURCES,
)


# ── Trust classification ──────────────────────────────────────────────

class TrustClass(str, Enum):
    """Derived trust classification for outcome sources.

    This is COMPUTED from caller authorization + source — never
    self-asserted by the caller.
    """
    PRODUCTION_TRUSTED = "production_trusted"
    PRODUCTION_UNVERIFIED = "production_unverified"
    TEST_ONLY = "test_only"
    RESEARCH_ONLY = "research_only"


class OutcomeTrustPolicyVersion(str, Enum):
    V1 = "outcome_trust_policy_v1"


# ── Source policy ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class SourcePolicy:
    """Declarative policy for a single outcome source.

    Attributes:
        source: The outcome source identifier.
        trust_class: Derived trust classification.
        required_role: Minimum RBAC role to submit this source.
        eligible_for_training: Whether this source is eligible for
            future supervised-learning datasets.
        eligible_for_rvw: Whether this source counts toward
            REAL_WORLD_VALIDATION (independent of global RV status).
        required_provenance: Whether a provenance reference is required.
        required_provenance_min_length: Minimum length of provenance ref.
        allowed_caller_roles: Set of role values that may submit.
    """
    source: str
    trust_class: TrustClass
    required_role: str  # minimum role name ("evaluator", "operator", "admin", "system")
    eligible_for_training: bool
    eligible_for_rvw: bool
    required_provenance: bool = False
    required_provenance_min_length: int = 0
    allowed_caller_roles: tuple[str, ...] = ()


# ── Policy definition ─────────────────────────────────────────────────

_ROLE_ORDER = {"evaluator": 1, "operator": 2, "admin": 3, "system": 99}

SOURCE_POLICIES: dict[str, SourcePolicy] = {
    # ── Production sources ────────────────────────────────────────────
    OutcomeSource.HUMAN_VERIFICATION.value: SourcePolicy(
        source=OutcomeSource.HUMAN_VERIFICATION.value,
        trust_class=TrustClass.PRODUCTION_TRUSTED,
        required_role="operator",
        eligible_for_training=True,
        eligible_for_rvw=True,
        required_provenance=True,
        required_provenance_min_length=6,
        allowed_caller_roles=("operator", "admin", "system"),
    ),
    OutcomeSource.CHARGEBACK.value: SourcePolicy(
        source=OutcomeSource.CHARGEBACK.value,
        trust_class=TrustClass.PRODUCTION_TRUSTED,
        required_role="operator",
        eligible_for_training=True,
        eligible_for_rvw=True,
        required_provenance=True,
        required_provenance_min_length=6,
        allowed_caller_roles=("operator", "admin", "system"),
    ),
    OutcomeSource.CONFIRMED_INVESTIGATION.value: SourcePolicy(
        source=OutcomeSource.CONFIRMED_INVESTIGATION.value,
        trust_class=TrustClass.PRODUCTION_TRUSTED,
        required_role="admin",
        eligible_for_training=True,
        eligible_for_rvw=True,
        required_provenance=True,
        required_provenance_min_length=6,
        allowed_caller_roles=("admin", "system"),
    ),
    OutcomeSource.EXTERNAL_ADJUDICATION.value: SourcePolicy(
        source=OutcomeSource.EXTERNAL_ADJUDICATION.value,
        trust_class=TrustClass.PRODUCTION_TRUSTED,
        required_role="admin",
        eligible_for_training=True,
        eligible_for_rvw=True,
        required_provenance=True,
        required_provenance_min_length=6,
        allowed_caller_roles=("admin", "system"),
    ),

    # ── Non-production sources ────────────────────────────────────────
    OutcomeSource.SYSTEM_TEST.value: SourcePolicy(
        source=OutcomeSource.SYSTEM_TEST.value,
        trust_class=TrustClass.TEST_ONLY,
        required_role="evaluator",
        eligible_for_training=False,
        eligible_for_rvw=False,
        required_provenance=False,
        allowed_caller_roles=("evaluator", "operator", "admin", "system"),
    ),
    OutcomeSource.SYNTHETIC.value: SourcePolicy(
        source=OutcomeSource.SYNTHETIC.value,
        trust_class=TrustClass.RESEARCH_ONLY,
        required_role="evaluator",
        eligible_for_training=False,
        eligible_for_rvw=False,
        required_provenance=False,
        allowed_caller_roles=("evaluator", "operator", "admin", "system"),
    ),
}

POLICY_VERSION = OutcomeTrustPolicyVersion.V1.value


def get_source_policy(source: str) -> SourcePolicy | None:
    """Look up the trust policy for an outcome source."""
    return SOURCE_POLICIES.get(source)


# ── PII patterns for provenance ──────────────────────────────────────

_PII_PATTERNS = (
    "@", "phone", "ssn", "credit card", "passport",
    "password", "secret", "api_key", "token",
)


# ── Trust evaluation ──────────────────────────────────────────────────

class OutcomeTrustError(Exception):
    """Raised when an outcome fails trust governance checks."""
    pass


@dataclass(frozen=True)
class TrustDecision:
    """Immutable, reproducible trust decision for a recorded outcome.

    Stored with the outcome for historical immutability.
    """
    trust_class: str
    eligible_for_training: bool
    eligible_for_real_world_validation: bool
    policy_version: str
    caller_role: str
    authorization_basis: str  # "role_hierarchy_match" or similar
    provenance_valid: bool
    provenance_basis: str  # why provenance passed/failed


def evaluate_outcome_trust(
    caller_role: str,
    source: str,
    provenance_ref: str = "",
) -> TrustDecision:
    """Evaluate trust governance for an outcome submission.

    This is the core authorization + provenance gate.

    Args:
        caller_role: The authenticated role of the caller ("evaluator",
            "operator", "admin", "system").  Must come from server-side
            authentication, NOT from a user-supplied field.
        source: The claimed outcome source.
        provenance_ref: The provenance reference string.

    Returns:
        TrustDecision with derived trust classification and eligibility.

    Raises:
        OutcomeTrustError on authorization failure, source escalation,
            or provenance violation.
    """
    policy = get_source_policy(source)
    if policy is None:
        raise OutcomeTrustError(f"unknown source: {source}")

    # 1. Authorization: caller role must satisfy required role
    caller_level = _ROLE_ORDER.get(caller_role, 0)
    required_level = _ROLE_ORDER.get(policy.required_role, 99)

    if caller_level < required_level:
        raise OutcomeTrustError(
            f"authorization_denied: {caller_role} cannot submit {source} "
            f"(requires >= {policy.required_role})"
        )

    # Verify caller is in the allowed set
    if policy.allowed_caller_roles and caller_role not in policy.allowed_caller_roles:
        raise OutcomeTrustError(
            f"authorization_denied: {caller_role} is not in allowed roles "
            f"for {source}"
        )

    # 2. Provenance validation
    provenance_valid = True
    provenance_basis = "accepted"

    if policy.required_provenance:
        if not provenance_ref or len(provenance_ref.strip()) == 0:
            provenance_valid = False
            provenance_basis = "missing_required_provenance"
        elif len(provenance_ref) < policy.required_provenance_min_length:
            provenance_valid = False
            provenance_basis = (
                f"provenance_too_short: {len(provenance_ref)} < "
                f"{policy.required_provenance_min_length}"
            )
        else:
            # PII check
            lower_prov = provenance_ref.lower()
            for pat in _PII_PATTERNS:
                if pat in lower_prov:
                    provenance_valid = False
                    provenance_basis = f"pii_detected: {pat}"
                    break

    if policy.required_provenance and not provenance_valid:
        raise OutcomeTrustError(
            f"provenance_denied: {provenance_basis}"
        )

    # 3. Derive trust class (always from policy, never caller-asserted)
    trust_class = policy.trust_class.value

    # 4. Eligibility determination
    eligible_training = policy.eligible_for_training
    eligible_rvw = policy.eligible_for_rvw and provenance_valid

    return TrustDecision(
        trust_class=trust_class,
        eligible_for_training=eligible_training,
        eligible_for_real_world_validation=eligible_rvw,
        policy_version=POLICY_VERSION,
        caller_role=caller_role,
        authorization_basis="role_hierarchy_match",
        provenance_valid=provenance_valid,
        provenance_basis=provenance_basis,
    )


# ── Trust-augmented outcome record ───────────────────────────────────

class OutcomeTrustRecord(Base):
    """Trust governance metadata attached to an outcome.

    Stored separately from OutcomeRecord to keep the pipeline clean.
    Joined on outcome_id for queries.
    """
    __tablename__ = "outcome_trust_records"

    outcome_id: Mapped[str] = mapped_column(
        String(36), primary_key=True,
    )
    trust_class: Mapped[str] = mapped_column(String(32), nullable=False)
    eligible_for_training: Mapped[bool] = mapped_column(Boolean, nullable=False)
    eligible_for_real_world_validation: Mapped[bool] = mapped_column(Boolean, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    caller_role: Mapped[str] = mapped_column(String(16), nullable=False)
    authorization_basis: Mapped[str] = mapped_column(String(64), nullable=False)
    provenance_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    provenance_basis: Mapped[str] = mapped_column(String(128), nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "trust_class": self.trust_class,
            "eligible_for_training": self.eligible_for_training,
            "eligible_for_real_world_validation": self.eligible_for_real_world_validation,
            "policy_version": self.policy_version,
            "caller_role": self.caller_role,
            "authorization_basis": self.authorization_basis,
            "provenance_valid": self.provenance_valid,
            "provenance_basis": self.provenance_basis,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
        }


# ── Trust-gated pipeline wrapper ─────────────────────────────────────

class OutcomeTrustGovernor:
    """Wraps OutcomePipeline to enforce trust governance.

    Every outcome recorded through this governor is subject to:
    - caller authorization
    - source trust policy
    - provenance validation
    - trust classification
    - audit/security integration

    The governor does NOT bypass the underlying pipeline's validation.
    It adds an authorization + trust layer on top.
    """

    def __init__(self, pipeline, session_factory):
        self._pipeline = pipeline
        self._Session = session_factory

    def record_outcome(
        self,
        event_id: str,
        fraud_id: str,
        label: str,
        source: str,
        effective_at: datetime,
        observed_at: datetime,
        caller_role: str,
        created_by: str,
        provenance_ref: str = "",
        supersedes: str | None = None,
    ) -> dict[str, Any]:
        """Record an outcome with full trust governance.

        caller_role MUST come from authenticated server-side context,
        NOT from a user-supplied request field.

        Returns dict with outcome + trust decision.
        Raises OutcomeTrustError on trust governance failure.
        Raises OutcomeValidationError on input validation failure.
        """
        # 1. Trust governance (authorization + provenance)
        trust_decision = evaluate_outcome_trust(
            caller_role=caller_role,
            source=source,
            provenance_ref=provenance_ref,
        )

        # 2. Record through the standard pipeline
        result = self._pipeline.record_outcome(
            event_id=event_id,
            fraud_id=fraud_id,
            label=label,
            source=source,
            effective_at=effective_at,
            observed_at=observed_at,
            created_by=created_by,
            provenance_ref=provenance_ref,
            supersedes=supersedes,
        )

        # 3. Persist trust governance metadata
        trust_record = OutcomeTrustRecord(
            outcome_id=result["outcome_id"],
            trust_class=trust_decision.trust_class,
            eligible_for_training=trust_decision.eligible_for_training,
            eligible_for_real_world_validation=trust_decision.eligible_for_real_world_validation,
            policy_version=trust_decision.policy_version,
            caller_role=trust_decision.caller_role,
            authorization_basis=trust_decision.authorization_basis,
            provenance_valid=trust_decision.provenance_valid,
            provenance_basis=trust_decision.provenance_basis,
        )

        db = self._Session()
        try:
            db.add(trust_record)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        # 4. Audit the trust decision
        self._audit_trust(
            fraud_id=fraud_id,
            event_type="OUTCOME_TRUST_DECIDED",
            outcome_id=result["outcome_id"],
            trust_decision=trust_decision,
        )

        # 5. Merge into result
        result["trust"] = trust_decision.__dict__
        return result

    def get_trust_record(self, outcome_id: str) -> dict[str, Any] | None:
        """Retrieve trust governance metadata for an outcome."""
        db = self._Session()
        try:
            record = db.query(OutcomeTrustRecord).filter(
                OutcomeTrustRecord.outcome_id == outcome_id,
            ).first()
            return record.to_dict() if record else None
        finally:
            db.close()

    def count_by_trust_class(self) -> dict[str, int]:
        """Count outcomes by trust class for observability."""
        db = self._Session()
        try:
            from sqlalchemy import func
            results = db.query(
                OutcomeTrustRecord.trust_class,
                func.count(OutcomeTrustRecord.outcome_id),
            ).group_by(OutcomeTrustRecord.trust_class).all()
            return {tc: count for tc, count in results}
        finally:
            db.close()

    def _audit_trust(
        self,
        fraud_id: str,
        event_type: str,
        outcome_id: str,
        trust_decision: TrustDecision,
    ) -> None:
        """Emit audit event for trust decision."""
        try:
            from src.audit_service.writer import append_audit_event
            append_audit_event(fraud_id, event_type, {
                "outcome_id": outcome_id,
                "trust_class": trust_decision.trust_class,
                "eligible_for_training": trust_decision.eligible_for_training,
                "eligible_for_rvw": trust_decision.eligible_for_real_world_validation,
                "policy_version": trust_decision.policy_version,
                "caller_role": trust_decision.caller_role,
                "provenance_valid": trust_decision.provenance_valid,
            })
        except Exception:
            pass  # audit must not block outcome recording
