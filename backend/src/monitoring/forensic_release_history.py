"""Phase 52: Forensic release history and operational governance.

Tracks every release lifecycle transition with cryptographic binding,
detects evidence gaps between registry/runtime/history, persists
structured incident records with state machine, and provides safe
export for auditor reconstruction.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from src.monitoring.release_lifecycle import (
    ReleaseRecord,
    ReleaseRegistry,
    ReleaseState,
)


# ── Incident State Machine ──────────────────────────────────────────────

class IncidentState(str, Enum):
    DETECTED = "DETECTED"
    CONTAINED = "CONTAINED"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"


class IncidentCategory(str, Enum):
    ARTIFACT_TAMPERING = "ARTIFACT_TAMPERING"
    MANIFEST_TAMPERING = "MANIFEST_TAMPERING"
    SIGNATURE_FAILURE = "SIGNATURE_FAILURE"
    HASH_MISMATCH = "HASH_MISMATCH"
    RUNTIME_DRIFT = "RUNTIME_DRIFT"
    REGISTRY_MISMATCH = "REGISTRY_MISMATCH"
    INVALID_RELEASE = "INVALID_RELEASE"
    REVOKED_RELEASE = "REVOKED_RELEASE"
    FAILED_STARTUP = "FAILED_STARTUP"
    FAILED_ROLLBACK = "FAILED_ROLLBACK"
    PARTIAL_DEPLOYMENT = "PARTIAL_DEPLOYMENT"
    OTHER = "OTHER"


_VALID_INCIDENT_TRANSITIONS: dict[IncidentState, set[IncidentState]] = {
    IncidentState.DETECTED: {IncidentState.CONTAINED},  # must go through CONTAINED
    IncidentState.CONTAINED: {IncidentState.INVESTIGATING, IncidentState.RESOLVED},
    IncidentState.INVESTIGATING: {IncidentState.RESOLVED},
    IncidentState.RESOLVED: set(),  # terminal
}


# ── Release Transition Record ───────────────────────────────────────────

@dataclass
class ReleaseTransitionRecord:
    """A single auditable lifecycle transition for a release."""
    transition_id: str
    release_id: str
    model_id: str
    manifest_hash: str
    artifact_set_hash: str
    feature_version: str
    schema_version: str
    previous_state: str
    new_state: str
    timestamp: float
    reason: str
    actor_type: str  # SYSTEM, OPERATOR, AUTOMATION
    audit_event_ref: str | None = None
    transition_hash: str = ""

    def compute_hash(self, previous_hash: str = "") -> str:
        """Compute hash binding this transition to the chain."""
        payload = json.dumps({
            "transition_id": self.transition_id,
            "release_id": self.release_id,
            "manifest_hash": self.manifest_hash,
            "artifact_set_hash": self.artifact_set_hash,
            "feature_version": self.feature_version,
            "schema_version": self.schema_version,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "actor_type": self.actor_type,
            "previous_hash": previous_hash,
        }, sort_keys=True)
        self.transition_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.transition_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "release_id": self.release_id,
            "model_id": self.model_id,
            "manifest_hash": self.manifest_hash,
            "artifact_set_hash": self.artifact_set_hash,
            "feature_version": self.feature_version,
            "schema_version": self.schema_version,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "actor_type": self.actor_type,
            "audit_event_ref": self.audit_event_ref,
            "transition_hash": self.transition_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReleaseTransitionRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Incident Record ─────────────────────────────────────────────────────

@dataclass
class IncidentRecord:
    """Structured incident record for security/release failures."""
    incident_id: str
    category: str  # IncidentCategory value
    timestamp: float
    affected_release_id: str | None = None
    manifest_hash: str | None = None
    artifact_hash: str | None = None
    runtime_state: str | None = None
    detection_source: str = ""
    response_state: str = IncidentState.DETECTED.value
    audit_reference: str | None = None
    resolution_status: str = "OPEN"
    state_history: list[dict[str, Any]] = field(default_factory=list)

    def transition_incident(self, new_state: IncidentState, reason: str = "") -> bool:
        """Attempt an incident state transition."""
        current = IncidentState(self.response_state)
        allowed = _VALID_INCIDENT_TRANSITIONS.get(current, set())
        if new_state not in allowed:
            return False
        self.state_history.append({
            "from": self.response_state,
            "to": new_state.value,
            "reason": reason,
            "timestamp": time.time(),
        })
        self.response_state = new_state.value
        if new_state == IncidentState.RESOLVED:
            self.resolution_status = "RESOLVED"
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IncidentRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Forensic Snapshot ───────────────────────────────────────────────────

@dataclass
class ForensicSnapshot:
    """Point-in-time forensic snapshot for security-critical failures."""
    snapshot_id: str
    timestamp: float
    release_id: str | None = None
    manifest_hash: str | None = None
    artifact_set_hash: str | None = None
    runtime_state: str | None = None
    feature_version: str | None = None
    schema_version: str | None = None
    preprocessing_hash: str | None = None
    rule_hash: str | None = None
    audit_chain_reference: str | None = None
    snapshot_hash: str = ""

    def compute_hash(self) -> str:
        payload = json.dumps({
            k: v for k, v in asdict(self).items() if k != "snapshot_hash"
        }, sort_keys=True)
        self.snapshot_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.snapshot_hash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ForensicSnapshot:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Evidence Gap ────────────────────────────────────────────────────────

@dataclass
class EvidenceGap:
    """A detected inconsistency between evidence sources."""
    gap_id: str
    category: str  # REGISTRY_HIST_MISMATCH, RUNTIME_HIST_MISMATCH, etc.
    description: str
    timestamp: float
    severity: str = "WARNING"  # WARNING, CRITICAL
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Forensic Release History ────────────────────────────────────────────

class ForensicReleaseHistory:
    """Tracks release lifecycle transitions with cryptographic binding.

    Persisted to models/releases/forensic_history.json alongside the registry.
    """

    def __init__(self, history_path: Path):
        self.history_path = history_path
        self._transitions: list[ReleaseTransitionRecord] = []
        self._incidents: dict[str, IncidentRecord] = {}
        self._snapshots: list[ForensicSnapshot] = []
        self._chain_head: str = ""  # last transition_hash
        self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self.history_path.exists():
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
            self._transitions = [
                ReleaseTransitionRecord.from_dict(t)
                for t in data.get("transitions", [])
            ]
            self._incidents = {
                iid: IncidentRecord.from_dict(inc)
                for iid, inc in data.get("incidents", {}).items()
            }
            self._snapshots = [
                ForensicSnapshot.from_dict(s)
                for s in data.get("snapshots", [])
            ]
            self._chain_head = data.get("chain_head", "")

    def _save(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "transitions": [t.to_dict() for t in self._transitions],
            "incidents": {iid: inc.to_dict() for iid, inc in self._incidents.items()},
            "snapshots": [s.to_dict() for s in self._snapshots],
            "chain_head": self._chain_head,
        }
        self.history_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── Transition Recording ─────────────────────────────────────────

    def record_transition(
        self,
        release: ReleaseRecord,
        previous_state: str,
        new_state: str,
        reason: str = "",
        actor_type: str = "SYSTEM",
        audit_event_ref: str | None = None,
    ) -> ReleaseTransitionRecord:
        """Record a lifecycle transition with cryptographic chain binding."""
        rec = ReleaseTransitionRecord(
            transition_id=f"tr-{uuid.uuid4().hex[:12]}",
            release_id=release.release_id,
            model_id=release.model_id,
            manifest_hash=release.manifest_hash,
            artifact_set_hash=release.artifact_hash,
            feature_version=release.feature_version,
            schema_version=release.schema_version,
            previous_state=previous_state,
            new_state=new_state,
            timestamp=time.time(),
            reason=reason,
            actor_type=actor_type,
            audit_event_ref=audit_event_ref,
        )
        rec.compute_hash(self._chain_head)
        self._chain_head = rec.transition_hash
        self._transitions.append(rec)
        self._save()
        return rec

    def get_transitions(
        self, release_id: str | None = None
    ) -> list[ReleaseTransitionRecord]:
        """Get transitions, optionally filtered by release_id."""
        if release_id is None:
            return list(self._transitions)
        return [t for t in self._transitions if t.release_id == release_id]

    def get_transition_chain(self) -> list[ReleaseTransitionRecord]:
        """Get the full transition chain in order."""
        return list(self._transitions)

    def verify_transition_chain(self) -> tuple[bool, str]:
        """Verify the cryptographic integrity of the transition chain."""
        prev_hash = ""
        for i, t in enumerate(self._transitions):
            expected_hash = hashlib.sha256(json.dumps({
                "transition_id": t.transition_id,
                "release_id": t.release_id,
                "manifest_hash": t.manifest_hash,
                "artifact_set_hash": t.artifact_set_hash,
                "feature_version": t.feature_version,
                "schema_version": t.schema_version,
                "previous_state": t.previous_state,
                "new_state": t.new_state,
                "timestamp": t.timestamp,
                "reason": t.reason,
                "actor_type": t.actor_type,
                "previous_hash": prev_hash,
            }, sort_keys=True).encode()).hexdigest()
            if t.transition_hash != expected_hash:
                return False, f"Chain broken at transition {i} ({t.transition_id}): expected {expected_hash[:16]}, got {t.transition_hash[:16]}"
            prev_hash = t.transition_hash
        return True, "Chain intact"

    # ── Active-Release Timeline ──────────────────────────────────────

    def get_active_timeline(self) -> list[dict[str, Any]]:
        """Reconstruct the active-release timeline.

        Returns a list of (release_id, activated_at, deactivated_at) tuples
        representing when each release was the active release.
        """
        timeline: list[dict[str, Any]] = []
        current_active: str | None = None
        activated_at: float | None = None

        for t in self._transitions:
            if t.new_state in ("ACTIVATED", "RUNTIME_ATTESTED") and current_active != t.release_id:
                if current_active is not None:
                    timeline.append({
                        "release_id": current_active,
                        "activated_at": activated_at,
                        "deactivated_at": t.timestamp,
                        "deactivation_reason": t.reason,
                    })
                current_active = t.release_id
                activated_at = t.timestamp
            elif t.new_state in ("REVOKED", "DRIFTED", "FAILED") and current_active == t.release_id:
                timeline.append({
                    "release_id": current_active,
                    "activated_at": activated_at,
                    "deactivated_at": t.timestamp,
                    "deactivation_reason": t.reason,
                })
                current_active = None
                activated_at = None

        if current_active is not None:
            timeline.append({
                "release_id": current_active,
                "activated_at": activated_at,
                "deactivated_at": None,
                "deactivation_reason": None,
            })

        return timeline

    # ── Incident Management ──────────────────────────────────────────

    def create_incident(
        self,
        category: IncidentCategory,
        affected_release_id: str | None = None,
        manifest_hash: str | None = None,
        artifact_hash: str | None = None,
        runtime_state: str | None = None,
        detection_source: str = "",
        audit_reference: str | None = None,
    ) -> IncidentRecord:
        """Create a new incident record."""
        inc = IncidentRecord(
            incident_id=f"inc-{uuid.uuid4().hex[:12]}",
            category=category.value,
            timestamp=time.time(),
            affected_release_id=affected_release_id,
            manifest_hash=manifest_hash,
            artifact_hash=artifact_hash,
            runtime_state=runtime_state,
            detection_source=detection_source,
            audit_reference=audit_reference,
        )
        self._incidents[inc.incident_id] = inc
        self._save()
        return inc

    def transition_incident(
        self, incident_id: str, new_state: IncidentState, reason: str = ""
    ) -> bool:
        """Transition an incident to a new state."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        ok = inc.transition_incident(new_state, reason)
        if ok:
            self._save()
        return ok

    def get_incident(self, incident_id: str) -> IncidentRecord | None:
        return self._incidents.get(incident_id)

    def get_open_incidents(self) -> list[IncidentRecord]:
        return [i for i in self._incidents.values() if i.resolution_status != "RESOLVED"]

    def get_incidents_for_release(self, release_id: str) -> list[IncidentRecord]:
        return [i for i in self._incidents.values() if i.affected_release_id == release_id]

    # ── Forensic Snapshots ───────────────────────────────────────────

    def take_forensic_snapshot(
        self,
        release_id: str | None = None,
        manifest_hash: str | None = None,
        artifact_set_hash: str | None = None,
        runtime_state: str | None = None,
        feature_version: str | None = None,
        schema_version: str | None = None,
        preprocessing_hash: str | None = None,
        rule_hash: str | None = None,
        audit_chain_reference: str | None = None,
    ) -> ForensicSnapshot:
        """Take a point-in-time forensic snapshot."""
        snap = ForensicSnapshot(
            snapshot_id=f"snap-{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            release_id=release_id,
            manifest_hash=manifest_hash,
            artifact_set_hash=artifact_set_hash,
            runtime_state=runtime_state,
            feature_version=feature_version,
            schema_version=schema_version,
            preprocessing_hash=preprocessing_hash,
            rule_hash=rule_hash,
            audit_chain_reference=audit_chain_reference,
        )
        snap.compute_hash()
        self._snapshots.append(snap)
        self._save()
        return snap

    def get_snapshots(self, release_id: str | None = None) -> list[ForensicSnapshot]:
        if release_id is None:
            return list(self._snapshots)
        return [s for s in self._snapshots if s.release_id == release_id]

    # ── Gap Detection ────────────────────────────────────────────────

    def detect_gaps(
        self,
        registry: ReleaseRegistry,
        runtime_release_id: str | None = None,
        runtime_manifest_hash: str | None = None,
    ) -> list[EvidenceGap]:
        """Detect inconsistencies between registry, runtime, and history.

        Args:
            registry: The active release registry.
            runtime_release_id: The release_id reported by the running service.
            runtime_manifest_hash: The manifest hash reported by the running service.

        Returns:
            List of detected evidence gaps.
        """
        gaps: list[EvidenceGap] = []
        now = time.time()

        active_record = registry.active_record()
        active_id = registry.active_release_id

        # 1. Registry says ACTIVE but no activation transition exists
        if active_id:
            activation_transitions = [
                t for t in self._transitions
                if t.release_id == active_id and t.new_state == "ACTIVATED"
            ]
            if not activation_transitions:
                gaps.append(EvidenceGap(
                    gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                    category="REGISTRY_HIST_MISMATCH",
                    description=f"Registry active={active_id} but no ACTIVATED transition in history",
                    timestamp=now,
                    severity="CRITICAL",
                    details={"release_id": active_id},
                ))

        # 2. Runtime says release A but history says release B
        if runtime_release_id and active_id and runtime_release_id != active_id:
            gaps.append(EvidenceGap(
                gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                category="RUNTIME_HIST_MISMATCH",
                description=f"Runtime={runtime_release_id} but registry active={active_id}",
                timestamp=now,
                severity="CRITICAL",
                details={"runtime_release_id": runtime_release_id, "registry_active_id": active_id},
            ))

        # Get historical transitions for the active release
        hist_transitions = [
            t for t in self._transitions
            if t.release_id == active_id and t.new_state in ("ACTIVATED", "RUNTIME_ATTESTED")
        ] if active_id else []

        # 3. Manifest hash mismatch between runtime and history
        if runtime_manifest_hash and active_record:
            if hist_transitions:
                last = hist_transitions[-1]
                if last.manifest_hash != runtime_manifest_hash:
                    gaps.append(EvidenceGap(
                        gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                        category="MANIFEST_HASH_MISMATCH",
                        description=f"Runtime manifest hash differs from historical record",
                        timestamp=now,
                        severity="CRITICAL",
                        details={
                            "runtime_hash": runtime_manifest_hash[:16],
                            "history_hash": last.manifest_hash[:16],
                        },
                    ))

        # 4. Artifact hash mismatch between runtime and history
        if active_record and hist_transitions:
            last = hist_transitions[-1]
            if last.artifact_set_hash != active_record.artifact_hash:
                gaps.append(EvidenceGap(
                    gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                    category="ARTIFACT_HASH_MISMATCH",
                    description=f"Registry artifact hash differs from historical record",
                    timestamp=now,
                    severity="WARNING",
                    details={
                        "registry_hash": active_record.artifact_hash[:16],
                        "history_hash": last.artifact_set_hash[:16],
                    },
                ))

        # 5. Transition chain integrity
        chain_ok, chain_msg = self.verify_transition_chain()
        if not chain_ok:
            gaps.append(EvidenceGap(
                gap_id=f"gap-{uuid.uuid4().hex[:8]}",
                category="CHAIN_INTEGRITY_FAILURE",
                description=f"Transition chain broken: {chain_msg}",
                timestamp=now,
                severity="CRITICAL",
                details={"chain_message": chain_msg},
            ))

        return gaps

    # ── Export / Reconstruction ──────────────────────────────────────

    def export_history(self, include_secrets: bool = False) -> dict[str, Any]:
        """Export release history for auditor reconstruction.

        Safe export that never includes HMAC keys, signing secrets,
        or credentials regardless of the include_secrets flag.
        """
        return {
            "format": "ps14-forensic-history-v1",
            "exported_at": time.time(),
            "transitions": [t.to_dict() for t in self._transitions],
            "chain_head": self._chain_head,
            "chain_integrity": self.verify_transition_chain(),
            "incidents": {iid: inc.to_dict() for iid, inc in self._incidents.items()},
            "snapshots": [s.to_dict() for s in self._snapshots],
            "active_timeline": self.get_active_timeline(),
            "open_incidents": [i.to_dict() for i in self.get_open_incidents()],
        }
