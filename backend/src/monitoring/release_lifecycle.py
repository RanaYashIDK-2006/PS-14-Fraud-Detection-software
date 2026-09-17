"""Phase 51: Release lifecycle state machine, revocation, and operational integrity.

Proves that releases can be created, activated, rolled back, rotated,
revoked, and recovered safely without allowing an unverified artifact
to become active.

The lifecycle states:

    DISCOVERED -> MANIFEST_VERIFIED -> GATE_VERIFIED -> TOKEN_VERIFIED
    -> REGISTERED -> ACTIVATED -> RUNTIME_ATTESTED

Failure/revocation states:

    INVALID, REVOKED, DRIFTED, ROLLBACK_PENDING, FAILED

Every state transition has explicit validation requirements. Invalid
transitions are rejected.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.monitoring.release_manifest import (
    ReleaseManifest,
    artifact_set_hash,
    sha256_file,
)
from src.monitoring.runtime_attestation import (
    RuntimeState,
    verify_release_for_load,
    build_attestation,
    RuntimeAttestation,
)


# ── Lifecycle States ─────────────────────────────────────────────────────


class ReleaseState(str, Enum):
    """Release lifecycle states with explicit transition rules."""

    DISCOVERED = "DISCOVERED"           # artifact files found on disk
    MANIFEST_VERIFIED = "MANIFEST_VERIFIED"  # ReleaseManifest loaded & verified
    GATE_VERIFIED = "GATE_VERIFIED"     # gate verdict accepted
    TOKEN_VERIFIED = "TOKEN_VERIFIED"   # PromotionToken verified (if applicable)
    REGISTERED = "REGISTERED"           # registered in release registry
    ACTIVATED = "ACTIVATED"             # loaded into runtime
    RUNTIME_ATTESTED = "RUNTIME_ATTESTED"  # attestation published, inference allowed

    # Failure/revocation states
    INVALID = "INVALID"                 # verification failed
    REVOKED = "REVOKED"                 # explicitly quarantined/revoked
    DRIFTED = "DRIFTED"                 # artifacts changed after activation
    FAILED = "FAILED"                   # startup/load failure


# Valid state transitions (source -> set of allowed targets)
_VALID_TRANSITIONS: dict[ReleaseState, set[ReleaseState]] = {
    ReleaseState.DISCOVERED: {ReleaseState.MANIFEST_VERIFIED, ReleaseState.INVALID},
    ReleaseState.MANIFEST_VERIFIED: {ReleaseState.GATE_VERIFIED, ReleaseState.INVALID},
    ReleaseState.GATE_VERIFIED: {ReleaseState.TOKEN_VERIFIED, ReleaseState.REGISTERED, ReleaseState.INVALID},
    ReleaseState.TOKEN_VERIFIED: {ReleaseState.REGISTERED, ReleaseState.INVALID},
    ReleaseState.REGISTERED: {ReleaseState.ACTIVATED, ReleaseState.INVALID, ReleaseState.REVOKED},
    ReleaseState.ACTIVATED: {ReleaseState.RUNTIME_ATTESTED, ReleaseState.DRIFTED, ReleaseState.FAILED, ReleaseState.REVOKED},
    ReleaseState.RUNTIME_ATTESTED: {ReleaseState.DRIFTED, ReleaseState.FAILED, ReleaseState.REVOKED},
    # Recovery from failure states requires re-discovery
    ReleaseState.INVALID: {ReleaseState.DISCOVERED},
    ReleaseState.DRIFTED: {ReleaseState.DISCOVERED},
    ReleaseState.FAILED: {ReleaseState.DISCOVERED},
    ReleaseState.REVOKED: set(),  # cannot un-revoke
}

# States that allow inference
_INFERENCE_READY = frozenset({ReleaseState.ACTIVATED, ReleaseState.RUNTIME_ATTESTED})

# States that block inference
_INFERENCE_BLOCKED = frozenset({
    ReleaseState.INVALID, ReleaseState.REVOKED,
    ReleaseState.DRIFTED, ReleaseState.FAILED,
})


# ── Release Record ───────────────────────────────────────────────────────


@dataclass
class ReleaseRecord:
    """Tracks the full lifecycle of a single release."""

    release_id: str
    model_id: str
    model_version: str
    artifact_hash: str
    manifest_hash: str
    feature_version: str
    schema_version: str
    rule_hash: str
    preprocessing_hash: str
    gate_verdict: str  # PROMOTION_ELIGIBLE or LEGACY_ATTESTED

    state: ReleaseState = ReleaseState.DISCOVERED
    state_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    activated_at: float | None = None
    revoked_at: float | None = None
    revocation_reason: str = ""

    def transition(self, new_state: ReleaseState, reason: str = "") -> bool:
        """Attempt a state transition. Returns True if valid."""
        allowed = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            return False
        self.state_history.append({
            "from": self.state.value,
            "to": new_state.value,
            "reason": reason,
            "timestamp": time.time(),
        })
        self.state = new_state
        if new_state == ReleaseState.ACTIVATED:
            self.activated_at = time.time()
        if new_state == ReleaseState.REVOKED:
            self.revoked_at = time.time()
            self.revocation_reason = reason
        return True

    def can_activate(self) -> tuple[bool, str]:
        """Check whether this release can currently be activated.

        Allows activation from REGISTERED, TOKEN_VERIFIED, GATE_VERIFIED, or
        re-activation from ACTIVATED/RUNTIME_ATTESTED (rollback scenario).
        """
        _ALLOWED = {
            ReleaseState.REGISTERED, ReleaseState.TOKEN_VERIFIED,
            ReleaseState.GATE_VERIFIED, ReleaseState.ACTIVATED,
            ReleaseState.RUNTIME_ATTESTED,
        }
        if self.state not in _ALLOWED:
            return False, f"Current state {self.state.value} does not allow activation"
        if self.state == ReleaseState.REVOKED:
            return False, "Release is revoked"
        return True, "OK"

    def can_infer(self) -> bool:
        """Whether inference is allowed for this release."""
        return self.state in _INFERENCE_READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "artifact_hash": self.artifact_hash,
            "manifest_hash": self.manifest_hash,
            "feature_version": self.feature_version,
            "schema_version": self.schema_version,
            "rule_hash": self.rule_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "gate_verdict": self.gate_verdict,
            "state": self.state.value,
            "state_history": self.state_history[-20:],  # last 20 transitions
            "created_at": self.created_at,
            "activated_at": self.activated_at,
            "revoked_at": self.revoked_at,
            "revocation_reason": self.revocation_reason,
        }


# ── Release Registry ─────────────────────────────────────────────────────


class ReleaseRegistry:
    """File-based registry tracking release lifecycle states.

    Persisted to models/releases/registry.json.
    """

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self._releases: dict[str, ReleaseRecord] = {}
        self._active_release_id: str | None = None
        self._lock = __import__('threading').Lock()
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            for rid, rd in data.get("releases", {}).items():
                rd["state"] = ReleaseState(rd["state"])
                self._releases[rid] = ReleaseRecord(**{
                    k: v for k, v in rd.items()
                    if k in ReleaseRecord.__dataclass_fields__
                })
            self._active_release_id = data.get("active_release_id")

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "releases": {rid: r.to_dict() for rid, r in self._releases.items()},
            "active_release_id": self._active_release_id,
        }
        self.registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def register(self, record: ReleaseRecord) -> bool:
        """Register a new release. Returns True if registered."""
        with self._lock:
            if record.release_id in self._releases:
                return False
            self._releases[record.release_id] = record
            self._save()
            return True

    def get(self, release_id: str) -> ReleaseRecord | None:
        return self._releases.get(release_id)

    def activate(self, release_id: str) -> tuple[bool, str]:
        """Activate a release. Returns (success, reason)."""
        with self._lock:
            record = self._releases.get(release_id)
            if record is None:
                return False, f"Release {release_id} not found"
            # Already the active release — idempotent
            if self._active_release_id == release_id and record.state in _INFERENCE_READY:
                return True, "already active"
            ok, reason = record.can_activate()
            if not ok:
                return False, reason
            ok = record.transition(ReleaseState.ACTIVATED, "activated")
            if not ok:
                return False, f"State transition to ACTIVATED failed"
            self._active_release_id = release_id
            self._save()
            return True, "OK"

    def attest(self, release_id: str) -> tuple[bool, str]:
        """Mark a release as runtime-attested."""
        with self._lock:
            record = self._releases.get(release_id)
            if record is None:
                return False, f"Release {release_id} not found"
            ok = record.transition(ReleaseState.RUNTIME_ATTESTED, "attested")
            if not ok:
                return False, f"State transition to RUNTIME_ATTESTED failed from {record.state.value}"
            self._save()
            return True, "OK"

    def revoke(self, release_id: str, reason: str = "") -> tuple[bool, str]:
        """Revoke/quarantine a release. Returns (success, reason)."""
        with self._lock:
            record = self._releases.get(release_id)
            if record is None:
                return False, f"Release {release_id} not found"
            if record.state == ReleaseState.REVOKED:
                return False, "Already revoked"
            # Revocation is allowed from any non-REVOKED state
            old_state = record.state
            record.state_history.append({
                "from": record.state.value,
                "to": ReleaseState.REVOKED.value,
                "reason": reason,
                "timestamp": time.time(),
            })
            record.state = ReleaseState.REVOKED
            record.revoked_at = time.time()
            record.revocation_reason = reason
            if self._active_release_id == release_id:
                self._active_release_id = None
            self._save()
            return True, f"Revoked from {old_state.value}"

    def mark_drifted(self, release_id: str, reason: str = "") -> tuple[bool, str]:
        """Mark a release as drifted."""
        with self._lock:
            record = self._releases.get(release_id)
            if record is None:
                return False, f"Release {release_id} not found"
            ok = record.transition(ReleaseState.DRIFTED, reason)
            if not ok:
                return False, f"State transition to DRIFTED failed from {record.state.value}"
            if self._active_release_id == release_id:
                self._active_release_id = None
            self._save()
            return True, "OK"

    @property
    def active_release_id(self) -> str | None:
        return self._active_release_id

    def active_record(self) -> ReleaseRecord | None:
        if self._active_release_id:
            return self._releases.get(self._active_release_id)
        return None

    def all_releases(self) -> dict[str, ReleaseRecord]:
        return dict(self._releases)


# ── Lifecycle Verifier ───────────────────────────────────────────────────


def verify_release_lifecycle(
    manifest: ReleaseManifest,
    artifact_dir: Path,
    registry: ReleaseRegistry | None = None,
    *,
    expected_rule_hash: str = "",
) -> tuple[ReleaseRecord, list[str]]:
    """Run the full release lifecycle verification.

    Returns (record, failures). The record's state reflects the result.
    """
    failures: list[str] = []

    # 1. DISCOVERED — artifact files exist
    if not artifact_dir.is_dir():
        failures.append(f"Artifact directory missing: {artifact_dir}")
        record = ReleaseRecord(
            release_id=manifest.release_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            artifact_hash=manifest.artifact_hash,
            manifest_hash=manifest.compute_manifest_hash(),
            feature_version=manifest.feature_version,
            schema_version=manifest.schema_version,
            rule_hash=manifest.rule_hash,
            preprocessing_hash=manifest.preprocessing_hash,
            gate_verdict=manifest.gate_verdict,
        )
        record.state = ReleaseState.INVALID
        return record, failures

    # 2. MANIFEST_VERIFIED — signature + hash integrity
    sig_ok, sig_reason = manifest.verify_signature()
    if not sig_ok:
        failures.append(f"manifest_signature: {sig_reason}")

    manifest_hash = manifest.compute_manifest_hash()

    # 3. GATE_VERIFIED — accepted gate verdict
    _ACCEPTED_VERDICTS = ("PROMOTION_ELIGIBLE", "LEGACY_ATTESTED")
    if manifest.gate_verdict not in _ACCEPTED_VERDICTS:
        failures.append(f"gate_verdict: {manifest.gate_verdict!r} not in {_ACCEPTED_VERDICTS}")

    # 4. Artifact-set hash
    art_ok, art_reason = manifest.verify_artifacts(artifact_dir)
    if not art_ok:
        failures.append(f"artifacts: {art_reason}")

    # 5. Component bindings
    bind_ok, bind_reason = manifest.verify_binding(
        expected_feature_version=manifest.feature_version,
    )
    if not bind_ok:
        failures.append(f"binding: {bind_reason}")
    if expected_rule_hash and manifest.rule_hash != expected_rule_hash:
        failures.append(f"rule_hash: manifest={manifest.rule_hash[:16]} != live={expected_rule_hash[:16]}")

    # Build record
    record = ReleaseRecord(
        release_id=manifest.release_id,
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        artifact_hash=manifest.artifact_hash,
        manifest_hash=manifest_hash,
        feature_version=manifest.feature_version,
        schema_version=manifest.schema_version,
        rule_hash=manifest.rule_hash,
        preprocessing_hash=manifest.preprocessing_hash,
        gate_verdict=manifest.gate_verdict,
    )

    if failures:
        record.state = ReleaseState.INVALID
        record.transition(ReleaseState.INVALID, "; ".join(failures[:3]))
    else:
        record.transition(ReleaseState.DISCOVERED, "artifact dir discovered")
        record.transition(ReleaseState.MANIFEST_VERIFIED, "manifest signature OK")
        record.transition(ReleaseState.GATE_VERIFIED, f"gate verdict = {manifest.gate_verdict}")
        record.transition(ReleaseState.REGISTERED, "all checks passed")

    return record, failures


def create_revoked_manifest(manifest: ReleaseManifest, reason: str = "") -> ReleaseManifest:
    """Create a revoked version of a manifest by setting gate_verdict to REVOKED."""
    rm = ReleaseManifest(
        release_id=manifest.release_id,
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        artifact_hash=manifest.artifact_hash,
        artifact_files=dict(manifest.artifact_files),
        feature_version=manifest.feature_version,
        schema_version=manifest.schema_version,
        preprocessing_hash=manifest.preprocessing_hash,
        rule_hash=manifest.rule_hash,
        evaluation_record_hash=manifest.evaluation_record_hash,
        training_config_hash=manifest.training_config_hash,
        source_git_sha=manifest.source_git_sha,
        python_version=manifest.python_version,
        dependency_fingerprint=manifest.dependency_fingerprint,
        training_seed=manifest.training_seed,
        created_at=manifest.created_at,
        gate_verdict="REVOKED",
        gate_timestamp=time.time(),
    )
    rm.sign()
    return rm
