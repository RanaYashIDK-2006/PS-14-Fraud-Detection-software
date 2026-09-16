"""Phase 49: Runtime release attestation.

The Phase 48 activation chain ends at ModelRegistry.promote() — a signed
ReleaseManifest authorizes promotion, but NOTHING verified that the model
the risk engine actually loads at startup IS the approved release.

Phase 49 closes that gap.  The invariant:

    APPROVED RELEASE == DEPLOYED RELEASE == LOADED RELEASE == INFERENCE RELEASE

The runtime identity is derived from the VERIFIED ReleaseManifest + artifact
bytes on disk — never from an environment variable.  MODEL_VERSION=approved
in the environment is not evidence; the cryptographic manifest identity is.

Startup order (enforced by attest_release()):
    discover release
      -> verify manifest integrity (canonical hash)
      -> verify manifest signature (HMAC)
      -> verify artifact-set hash (exact bytes on disk)
      -> verify component bindings (feature/schema/rule/evaluation)
      -> verify release authorization (gate verdict)
      -> [caller loads the model]
      -> publish runtime attestation
      -> inference ready

If verification fails: the model is NOT exposed as active.  The service
reports MODEL_NOT_READY (health shows model NOT ready) — the failure is
auditable and fail-closed.  A corrupted artifact can never reach healthy
inference.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.monitoring.release_manifest import (
    ReleaseManifest,
    artifact_set_hash,
)

# ── Runtime states ───────────────────────────────────────────────────────


class RuntimeState(str, Enum):
    """Lifecycle of the loaded release (Phase 49 §12 health semantics)."""

    STARTING = "STARTING"          # process up, verification not finished
    READY = "READY"                # manifest verified, model loaded, inference allowed
    MODEL_NOT_READY = "MODEL_NOT_READY"  # service alive but model unavailable
    INCONSISTENT = "INCONSISTENT"  # registry state != runtime release (fail-closed)
    DRIFTED = "DRIFTED"            # artifacts changed after load (fail-closed)
    FAILED = "FAILED"              # verification failed at load time


# Operations that are legitimate controlled transitions (allowed to change
# registry state / artifacts under the promotion architecture).  Anything
# outside this set that mutates a verified release is DRIFT.
CONTROLLED_TRANSITIONS = frozenset({
    "promote", "rollback", "deploy", "retrain", "load_release",
})

# File basename of the ReleaseManifest as saved by Phase 48.
RELEASE_MANIFEST_FILENAME = "release_manifest.json"


# ── Attestation ──────────────────────────────────────────────────────────


@dataclass
class RuntimeAttestation:
    """Immutable identity of the release actually loaded into this process.

    Every field is derived from the verified ReleaseManifest / artifact
    bytes — the caller cannot inject an arbitrary identity (unlike an
    env var).  loaded_at is recorded once at load; the attestation hash
    binds it for tamper detection.
    """

    release_id: str
    model_id: str
    model_version: str
    artifact_hash: str
    manifest_hash: str
    feature_version: str
    schema_version: str
    preprocessing_hash: str
    rule_hash: str
    evaluation_record_hash: str
    source_git_sha: str = ""
    loaded_at: float = field(default_factory=time.time)
    verified: bool = False
    verification_failures: list[str] = field(default_factory=list)
    artifact_dir: str = ""

    def attestation_hash(self) -> str:
        """Deterministic SHA-256 over the attestation identity fields."""
        payload = {
            "release_id": self.release_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "artifact_hash": self.artifact_hash,
            "manifest_hash": self.manifest_hash,
            "feature_version": self.feature_version,
            "schema_version": self.schema_version,
            "preprocessing_hash": self.preprocessing_hash,
            "rule_hash": self.rule_hash,
            "evaluation_record_hash": self.evaluation_record_hash,
            "loaded_at": self.loaded_at,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    def to_safe_dict(self) -> dict[str, Any]:
        """Operator-facing view — NO secrets, NO HMAC keys, NO model internals."""
        return {
            "release_id": self.release_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "artifact_hash": self.artifact_hash,
            "manifest_hash": self.manifest_hash,
            "feature_version": self.feature_version,
            "schema_version": self.schema_version,
            "preprocessing_hash": self.preprocessing_hash,
            "rule_hash": self.rule_hash,
            "evaluation_record_hash": self.evaluation_record_hash,
            "source_git_sha": self.source_git_sha,
            "artifact_dir": self.artifact_dir,
            "loaded_at": self.loaded_at,
            "verified": self.verified,
            "attestation_hash": self.attestation_hash(),
        }


# ── Verification pipeline ────────────────────────────────────────────────


def verify_release_for_load(
    manifest: ReleaseManifest,
    artifact_dir: Path,
    *,
    expected_feature_version: str = "",
    expected_rule_hash: str = "",
) -> tuple[bool, list[str], dict[str, str]]:
    """Pre-load verification pipeline (Phase 49 §3).

    Order is mandatory:
      1. manifest signature  (is this the manifest we signed?)
      2. manifest hash integrity (canonical hash recomputation)
      3. gate verdict        (was this release actually approved?)
      4. artifact-set hash   (are the exact bytes on disk what we approved?)
      5. component bindings  (feature schema / rules still what we approved?)

    Returns (ok, failures, identity) where identity carries the verified
    fields used to build the RuntimeAttestation.  On any failure the model
    must NOT be loaded.
    """
    failures: list[str] = []

    # 1. Manifest signature
    sig_ok, sig_reason = manifest.verify_signature()
    if not sig_ok:
        failures.append(f"manifest_signature: {sig_reason}")

    # 2. Manifest hash integrity — recompute; if the record was mutated the
    #    signature check above already fails, but an unsigned-but-parsed
    #    manifest still needs a canonical identity to attest.
    try:
        manifest_hash = manifest.compute_manifest_hash()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"manifest_hash: {exc}")
        manifest_hash = ""

    # 3. Gate verdict — only an approved or attested release may load.
    #    PROMOTION_ELIGIBLE = fully promoted; LEGACY_ATTESTED = legacy
    #    artifact reconstructed from existing evidence (attestation ≠ promotion).
    _ACCEPTED_VERDICTS = ("PROMOTION_ELIGIBLE", "LEGACY_ATTESTED")
    if manifest.gate_verdict not in _ACCEPTED_VERDICTS:
        failures.append(
            f"gate_verdict: {manifest.gate_verdict!r} not in {_ACCEPTED_VERDICTS}"
        )

    # 4. Artifact-set verification (exact bytes on disk)
    art_ok, art_reason = manifest.verify_artifacts(artifact_dir)
    if not art_ok:
        failures.append(f"artifacts: {art_reason}")

    # 5. Component bindings requested by the caller (runtime knows the live
    #    feature version / rules.yaml hash and can demand they match).
    bind_ok, bind_reason = manifest.verify_binding(
        expected_feature_version=expected_feature_version,
    )
    if not bind_ok:
        failures.append(f"binding: {bind_reason}")
    if expected_rule_hash and manifest.rule_hash != expected_rule_hash:
        failures.append(
            f"rule_hash: manifest={manifest.rule_hash[:16] if manifest.rule_hash else '(empty)'} "
            f"!= live rules={expected_rule_hash[:16] if expected_rule_hash else '(empty)'}"
        )

    identity = {
        "release_id": manifest.release_id,
        "model_id": manifest.model_id,
        "model_version": manifest.model_version,
        "artifact_hash": manifest.artifact_hash,
        "manifest_hash": manifest_hash,
        "feature_version": manifest.feature_version,
        "schema_version": manifest.schema_version,
        "preprocessing_hash": manifest.preprocessing_hash,
        "rule_hash": manifest.rule_hash,
        "evaluation_record_hash": manifest.evaluation_record_hash,
        "source_git_sha": manifest.source_git_sha,
    }
    return (len(failures) == 0), failures, identity


def build_attestation(
    identity: dict[str, str],
    artifact_dir: Path,
) -> RuntimeAttestation:
    """Build the runtime attestation from verified identity fields."""
    return RuntimeAttestation(
        release_id=identity.get("release_id", ""),
        model_id=identity.get("model_id", ""),
        model_version=identity.get("model_version", ""),
        artifact_hash=identity.get("artifact_hash", ""),
        manifest_hash=identity.get("manifest_hash", ""),
        feature_version=identity.get("feature_version", ""),
        schema_version=identity.get("schema_version", ""),
        preprocessing_hash=identity.get("preprocessing_hash", ""),
        rule_hash=identity.get("rule_hash", ""),
        evaluation_record_hash=identity.get("evaluation_record_hash", ""),
        source_git_sha=identity.get("source_git_sha", ""),
        verified=True,
        artifact_dir=str(artifact_dir),
    )


# ── Runtime drift detection ──────────────────────────────────────────────


def detect_runtime_drift(
    attestation: RuntimeAttestation,
    artifact_dir: Path,
) -> tuple[RuntimeState, list[str]]:
    """Detect drift between the attested release and the current disk state.

    Called on health checks / periodically.  Distinguishes:
      - READY:      artifacts still match the attested release
      - DRIFTED:    any artifact file changed after load (fail-closed:
                    a file modification must NOT silently become a new
                    active model — the old identity no longer matches)

    Does NOT reload files automatically.
    """
    drift: list[str] = []
    if not artifact_dir.is_dir():
        return RuntimeState.DRIFTED, [f"artifact directory missing: {artifact_dir}"]
    actual = artifact_set_hash(artifact_dir)
    if actual["model_hash"] != attestation.artifact_hash:
        drift.append(
            f"artifact_hash drift: current={str(actual['model_hash'])[:16] if actual['model_hash'] else 'None'}… "
            f"attested={attestation.artifact_hash[:16] if attestation.artifact_hash else '(empty)'}…"
        )
    if drift:
        return RuntimeState.DRIFTED, drift
    return RuntimeState.READY, []


# ── Registry ↔ runtime consistency ──────────────────────────────────────


def check_registry_runtime_consistency(
    registry_active: dict[str, Any] | None,
    attestation: RuntimeAttestation,
) -> tuple[RuntimeState, list[str]]:
    """Compare the ModelRegistry ACTIVE state with the runtime-loaded release.

    The following must match when the registry records an active release:
      release_id, model_id, artifact_hash, manifest_hash,
      feature_version, schema_version, preprocessing_hash, rule_hash,
      model_version.

    registry_active=None means the registry has no release-tracking state
    (pre-Phase-49 registries / legacy deployments) — no inconsistency is
    claimed, but the attestation alone governs readiness.

    Mismatch → INCONSISTENT: the system must NOT silently continue as
    though the state were valid (fail-closed; inference readiness blocked).
    """
    if not registry_active:
        return RuntimeState.READY, []
    mismatches: list[str] = []
    field_map = {
        "release_id": attestation.release_id,
        "model_id": attestation.model_id,
        "model_version": attestation.model_version,
        "artifact_hash": attestation.artifact_hash,
        "manifest_hash": attestation.manifest_hash,
        "feature_version": attestation.feature_version,
        "schema_version": attestation.schema_version,
        "preprocessing_hash": attestation.preprocessing_hash,
        "rule_hash": attestation.rule_hash,
    }
    for key, runtime_val in field_map.items():
        reg_val = registry_active.get(key)
        if reg_val is None:
            continue  # registry does not track this field — skip
        if str(reg_val) != str(runtime_val):
            mismatches.append(f"{key}: registry={reg_val} != runtime={runtime_val}")
    if mismatches:
        return RuntimeState.INCONSISTENT, mismatches
    return RuntimeState.READY, []


# ── Audit payload (Phase 49 §14) ─────────────────────────────────────────


def attestation_audit_payload(
    attestation: RuntimeAttestation | None,
    state: RuntimeState,
    failures: list[str] | None = None,
) -> dict[str, Any]:
    """Audit-safe payload for release load / failure events.  No secrets."""
    payload: dict[str, Any] = {
        "runtime_state": state.value,
        "attestation_hash": attestation.attestation_hash() if attestation else None,
    }
    if attestation is not None:
        payload.update({
            "release_id": attestation.release_id,
            "model_id": attestation.model_id,
            "model_version": attestation.model_version,
            "artifact_hash": attestation.artifact_hash,
            "manifest_hash": attestation.manifest_hash,
            "feature_version": attestation.feature_version,
        })
    if failures:
        payload["failures"] = list(failures)[:20]  # bound size
    return payload
