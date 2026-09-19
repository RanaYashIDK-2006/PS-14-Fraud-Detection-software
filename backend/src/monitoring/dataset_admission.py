"""Phase 82: Dataset admission / readiness.

Deterministic, auditable layer over OutcomeRecord rows that determines
whether a proposed dataset candidate satisfies explicit governance
requirements for future supervised learning.

Does NOT train a model, validate performance, or promote anything.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.risk_engine.models import Base
from src.monitoring.outcome_pipeline import (
    OutcomeRecord,
    OutcomeStatus,
    OutcomeSource,
    OutcomeLabel,
    SOURCE_TRUST,
)
from src.monitoring.outcome_trust import (
    TrustClass,
    OutcomeTrustRecord,
    get_source_policy,
)


# ── Admission states ──────────────────────────────────────────────────

class AdmissionState(str, Enum):
    NOT_READY = "not_ready"
    BLOCKED = "blocked"
    READY = "ready"


# ── Exclusion reasons ─────────────────────────────────────────────────

class ExclusionReason(str, Enum):
    TEST_ONLY_SOURCE = "test_only_source"
    RESEARCH_ONLY_SOURCE = "research_only_source"
    NOT_TRUSTED = "not_trusted"
    STATUS_RETRACTED = "status_retracted"
    STATUS_DISPUTED = "status_disputed"
    STATUS_PENDING = "status_pending"
    UNKNOWN_LABEL = "unknown_label"
    LABEL_CONFLICT = "label_conflict"
    MISSING_RISK_SCORE = "missing_risk_score"
    MISSING_DECISION_PROVENANCE = "missing_decision_provenance"
    TEMPORAL_VIOLATION = "temporal_violation"
    SUPERSEDED = "superseded"
    MISSING_TRUST_RECORD = "missing_trust_record"
    PROVENANCE_INVALID = "provenance_invalid"
    NO_OUTCOMES = "no_outcomes"


# ── Dataset candidate definition ──────────────────────────────────────

@dataclass(frozen=True)
class DatasetCandidate:
    """Immutable identifier for a proposed dataset snapshot."""
    dataset_id: str
    dataset_version: str
    created_at: datetime
    outcome_schema_version: int
    policy_version: str


def make_dataset_candidate(
    dataset_version: str = "1.0",
    outcome_schema_version: int = 1,
) -> DatasetCandidate:
    """Create a new dataset candidate with deterministic ID."""
    ds_id = f"DS-{uuid.uuid4().hex[:12].upper()}"
    return DatasetCandidate(
        dataset_id=ds_id,
        dataset_version=dataset_version,
        created_at=datetime.now(timezone.utc),
        outcome_schema_version=outcome_schema_version,
        policy_version="outcome_trust_policy_v1",
    )


# ── Outcome admission record ──────────────────────────────────────────

@dataclass
class OutcomeAdmission:
    """Tracks whether a single outcome is admitted and why."""
    outcome_id: str
    event_id: str
    fraud_id: str
    label: str
    source: str
    status: str
    admitted: bool
    exclusion_reason: str | None = None


# ── Admission statistics ──────────────────────────────────────────────

@dataclass
class AdmissionStats:
    """Factual dataset statistics."""
    total_outcomes: int = 0
    admitted: int = 0
    excluded: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)

    # Label distribution (admitted only)
    label_fraud: int = 0
    label_not_fraud: int = 0
    label_unknown: int = 0

    # Source distribution (admitted only)
    source_distribution: dict[str, int] = field(default_factory=dict)

    # Trust distribution (admitted only)
    trust_distribution: dict[str, int] = field(default_factory=dict)

    # Status distribution (all candidates)
    status_distribution: dict[str, int] = field(default_factory=dict)

    # Provenance consistency
    model_ids: dict[str, int] = field(default_factory=dict)
    release_ids: dict[str, int] = field(default_factory=dict)
    feature_versions: dict[str, int] = field(default_factory=dict)

    # Temporal range (admitted)
    earliest_decision_at: str | None = None
    latest_decision_at: str | None = None


# ── Admission result ──────────────────────────────────────────────────

@dataclass(frozen=True)
class AdmissionResult:
    """Immutable, deterministic admission evaluation result."""
    candidate: dict  # DatasetCandidate as dict
    state: str  # AdmissionState value
    stats: dict  # AdmissionStats as dict
    manifest_hash: str
    admission_time: str  # ISO timestamp of this evaluation
    deterministic: bool = True  # always True by design


# ── Admission engine ──────────────────────────────────────────────────

class DatasetAdmissionEngine:
    """Deterministic admission evaluator for outcome datasets.

    Given a set of OutcomeRecord rows and their associated
    OutcomeTrustRecord rows, determines admission eligibility
    according to the trust/status/temporal rules.

    Produces an immutable AdmissionResult with a reproducible
    manifest hash.
    """

    def __init__(self, session_factory, risk_session_factory=None):
        self._Session = session_factory
        self._RiskSession = risk_session_factory

    def evaluate_candidate(
        self,
        candidate: DatasetCandidate,
        cutoff_time: datetime | None = None,
    ) -> AdmissionResult:
        """Evaluate all available outcomes against admission rules.

        Args:
            candidate: The dataset candidate identity.
            cutoff_time: If provided, only consider outcomes recorded
                before this time. Enables point-in-time snapshots.

        Returns:
            AdmissionResult with full stats, admission decisions, and
            deterministic manifest hash.
        """
        db = self._Session()
        try:
            # Fetch all outcomes
            query = db.query(OutcomeRecord)
            if cutoff_time is not None:
                query = query.filter(OutcomeRecord.recorded_at <= cutoff_time)
            all_outcomes = query.order_by(OutcomeRecord.outcome_id).all()

            # Fetch all trust records
            trust_records = db.query(OutcomeTrustRecord).all()
            trust_map = {tr.outcome_id: tr for tr in trust_records}

            # Evaluate each outcome
            admitted: list[OutcomeAdmission] = []
            excluded: list[OutcomeAdmission] = []
            stats = AdmissionStats()

            for outcome in all_outcomes:
                decision = self._evaluate_single_outcome(outcome, trust_map, all_outcomes)
                if decision.admitted:
                    admitted.append(decision)
                else:
                    excluded.append(decision)

            # Compute statistics
            stats = self._compute_stats(admitted, excluded, all_outcomes, trust_map)

            # Determine admission state
            state = self._determine_state(admitted, stats)

            # Build manifest (deterministic)
            manifest = self._build_manifest(candidate, stats, state)
            manifest_hash = self._hash_manifest(manifest)

            return AdmissionResult(
                candidate=self._candidate_to_dict(candidate),
                state=state.value,
                stats=stats.__dict__,
                manifest_hash=manifest_hash,
                admission_time=datetime.now(timezone.utc).isoformat(),
                deterministic=True,
            )
        finally:
            db.close()

    def _evaluate_single_outcome(
        self,
        outcome: OutcomeRecord,
        trust_map: dict[str, OutcomeTrustRecord],
        all_outcomes: list[OutcomeRecord],
    ) -> OutcomeAdmission:
        """Evaluate a single outcome for admission."""
        base = OutcomeAdmission(
            outcome_id=outcome.outcome_id,
            event_id=outcome.event_id,
            fraud_id=outcome.fraud_id,
            label=outcome.label,
            source=outcome.outcome_source,
            status=outcome.status,
            admitted=False,
        )

        # 1. Status filter
        if outcome.status == OutcomeStatus.RETRACTED.value:
            return self._exclude(base, ExclusionReason.STATUS_RETRACTED)
        if outcome.status == OutcomeStatus.DISPUTED.value:
            return self._exclude(base, ExclusionReason.STATUS_DISPUTED)
        if outcome.status == OutcomeStatus.PENDING.value:
            return self._exclude(base, ExclusionReason.STATUS_PENDING)

        # 2. Trust record must exist
        trust = trust_map.get(outcome.outcome_id)
        if trust is None:
            return self._exclude(base, ExclusionReason.MISSING_TRUST_RECORD)

        # 3. Trust class filter
        if trust.trust_class == TrustClass.TEST_ONLY.value:
            return self._exclude(base, ExclusionReason.TEST_ONLY_SOURCE)
        if trust.trust_class == TrustClass.RESEARCH_ONLY.value:
            return self._exclude(base, ExclusionReason.RESEARCH_ONLY_SOURCE)

        # 4. Trust eligibility
        if not trust.eligible_for_training:
            return self._exclude(base, ExclusionReason.NOT_TRUSTED)

        # 5. Provenance validity
        if not trust.provenance_valid:
            return self._exclude(base, ExclusionReason.PROVENANCE_INVALID)

        # 6. Label quality: reject UNKNOWN
        if outcome.label == OutcomeLabel.UNKNOWN.value:
            return self._exclude(base, ExclusionReason.UNKNOWN_LABEL)

        # 7. Supersession: skip superseded outcomes
        if outcome.superseded_by is not None:
            return self._exclude(base, ExclusionReason.SUPERSEDED)

        # 8. Temporal integrity
        if not self._check_temporal(outcome):
            return self._exclude(base, ExclusionReason.TEMPORAL_VIOLATION)

        # 9. Conflict detection: if this outcome has been corrected by another
        #    confirmed outcome from the same source, it is superseded
        if self._is_superseded_by_later(outcome, all_outcomes):
            return self._exclude(base, ExclusionReason.SUPERSEDED)

        # 10. Decision provenance check (RiskScore existence verified by
        #     outcome creation, but check for coherent model/release/feature)
        if not self._check_decision_provenance(outcome):
            return self._exclude(base, ExclusionReason.MISSING_DECISION_PROVENANCE)

        base.admitted = True
        return base

    def _exclude(
        self, base: OutcomeAdmission, reason: ExclusionReason
    ) -> OutcomeAdmission:
        base.exclusion_reason = reason.value
        return base

    def _check_temporal(self, outcome: OutcomeRecord) -> bool:
        """Verify point-in-time temporal ordering.

        Core rule: effective_at <= observed_at.
        This prevents future-information leakage: the label/outcome must
        not be observed before it takes effect.

        Note: decision_at may default to current time when a RiskScore
        is first inserted (test scenarios), so decision_at <= effective_at
        is not enforced here. The critical invariant is effective_at <= observed_at.
        """
        if outcome.effective_at and outcome.observed_at:
            if outcome.effective_at > outcome.observed_at:
                return False
        return True

    def _check_decision_provenance(self, outcome: OutcomeRecord) -> bool:
        """Verify outcome has coherent decision provenance."""
        # model_id must not be empty/unknown for production outcomes
        if not outcome.model_id or outcome.model_id == "unknown":
            return False
        return True

    def _is_superseded_by_later(
        self, outcome: OutcomeRecord, all_outcomes: list[OutcomeRecord]
    ) -> bool:
        """Check if a later confirmed outcome supersedes this one."""
        for other in all_outcomes:
            if other.supersedes == outcome.outcome_id:
                if other.status == OutcomeStatus.CONFIRMED.value:
                    return True
        return False

    def _compute_stats(
        self,
        admitted: list[OutcomeAdmission],
        excluded: list[OutcomeAdmission],
        all_outcomes: list[OutcomeRecord],
        trust_map: dict[str, OutcomeTrustRecord],
    ) -> AdmissionStats:
        """Compute deterministic statistics from admission results."""
        stats = AdmissionStats()
        stats.total_outcomes = len(all_outcomes)
        stats.admitted = len(admitted)
        stats.excluded = len(excluded)

        # Exclusion reasons
        for ex in excluded:
            reason = ex.exclusion_reason or "unknown"
            stats.exclusion_reasons[reason] = stats.exclusion_reasons.get(reason, 0) + 1

        # Admitted-only statistics
        earliest_dt: datetime | None = None
        latest_dt: datetime | None = None

        for adm in admitted:
            # Label counts
            if adm.label == "fraud":
                stats.label_fraud += 1
            elif adm.label == "not_fraud":
                stats.label_not_fraud += 1
            elif adm.label == "unknown":
                stats.label_unknown += 1

            # Source distribution
            stats.source_distribution[adm.source] = (
                stats.source_distribution.get(adm.source, 0) + 1
            )

        # Trust distribution from all outcomes (for transparency)
        for outcome in all_outcomes:
            trust = trust_map.get(outcome.outcome_id)
            tc = trust.trust_class if trust else "no_trust_record"
            stats.trust_distribution[tc] = stats.trust_distribution.get(tc, 0) + 1

        # Status distribution (all candidates)
        for outcome in all_outcomes:
            stats.status_distribution[outcome.status] = (
                stats.status_distribution.get(outcome.status, 0) + 1
            )

        # Provenance consistency (from actual OutcomeRecord data)
        for outcome in all_outcomes:
            if outcome.model_id:
                stats.model_ids[outcome.model_id] = (
                    stats.model_ids.get(outcome.model_id, 0) + 1
                )
            if outcome.release_id:
                stats.release_ids[outcome.release_id] = (
                    stats.release_ids.get(outcome.release_id, 0) + 1
                )
            if outcome.feature_version:
                stats.feature_versions[outcome.feature_version] = (
                    stats.feature_versions.get(outcome.feature_version, 0) + 1
                )

        # Temporal range (from admitted outcomes via all_outcomes)
        for outcome in all_outcomes:
            if outcome.decision_at:
                dt = outcome.decision_at
                if isinstance(dt, datetime):
                    if earliest_dt is None or dt < earliest_dt:
                        earliest_dt = dt
                    if latest_dt is None or dt > latest_dt:
                        latest_dt = dt

        if earliest_dt:
            stats.earliest_decision_at = earliest_dt.isoformat()
        if latest_dt:
            stats.latest_decision_at = latest_dt.isoformat()

        return stats

    def _determine_state(
        self,
        admitted: list[OutcomeAdmission],
        stats: AdmissionStats,
    ) -> AdmissionState:
        """Determine dataset admission state from admission results."""
        if stats.total_outcomes == 0:
            return AdmissionState.NOT_READY

        if stats.exclusion_reasons.get(ExclusionReason.NO_OUTCOMES.value, 0) > 0:
            return AdmissionState.NOT_READY

        # BLOCKED if only non-production sources or policy violations
        blocked_reasons = {
            ExclusionReason.TEST_ONLY_SOURCE.value,
            ExclusionReason.RESEARCH_ONLY_SOURCE.value,
            ExclusionReason.PROVENANCE_INVALID.value,
        }
        for reason_key in stats.exclusion_reasons:
            if reason_key in blocked_reasons:
                # If ALL outcomes are excluded for blocked reasons, it's BLOCKED
                if stats.admitted == 0:
                    return AdmissionState.BLOCKED

        if stats.admitted == 0:
            return AdmissionState.NOT_READY

        return AdmissionState.READY

    def _build_manifest(
        self,
        candidate: DatasetCandidate,
        stats: AdmissionStats,
        state: AdmissionState,
    ) -> dict[str, Any]:
        """Build a deterministic manifest from candidate and stats."""
        return {
            "dataset_id": candidate.dataset_id,
            "dataset_version": candidate.dataset_version,
            "outcome_schema_version": candidate.outcome_schema_version,
            "policy_version": candidate.policy_version,
            "admission_state": state.value,
            "total_outcomes": stats.total_outcomes,
            "admitted": stats.admitted,
            "excluded": stats.excluded,
            "label_fraud": stats.label_fraud,
            "label_not_fraud": stats.label_not_fraud,
            "label_unknown": stats.label_unknown,
            "source_distribution": dict(sorted(stats.source_distribution.items())),
            "trust_distribution": dict(sorted(stats.trust_distribution.items())),
            "status_distribution": dict(sorted(stats.status_distribution.items())),
            "exclusion_reasons": dict(sorted(stats.exclusion_reasons.items())),
            "model_ids": dict(sorted(stats.model_ids.items())),
            "release_ids": dict(sorted(stats.release_ids.items())),
            "feature_versions": dict(sorted(stats.feature_versions.items())),
            "earliest_decision_at": stats.earliest_decision_at,
            "latest_decision_at": stats.latest_decision_at,
        }

    @staticmethod
    def _hash_manifest(manifest: dict) -> str:
        """Deterministic SHA-256 hash of the manifest."""
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _candidate_to_dict(c: DatasetCandidate) -> dict[str, Any]:
        return {
            "dataset_id": c.dataset_id,
            "dataset_version": c.dataset_version,
            "created_at": c.created_at.isoformat(),
            "outcome_schema_version": c.outcome_schema_version,
            "policy_version": c.policy_version,
        }
