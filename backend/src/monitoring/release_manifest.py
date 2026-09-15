"""Phase 48: Signed model release manifest.

A ReleaseManifest is a tamper-evident, deterministic record that binds
exactly which model artifact, preprocessing, feature schema, rule config,
and evaluation evidence constitute an approved release.  The manifest hash
is the single identity of the release; any mutation changes the hash.

Activation chain:
    evaluate_promotion() → PromotionDecision
        → PromotionToken (HMAC-signed)
            → ReleaseManifest (canonical JSON → SHA-256)
                → ExactArtifactSet verification
                    → ModelRegistry.promote()

The registry verifies the manifest hash before activation, preventing:
- token for release A used with release B
- modified artifact after approval
- modified preprocessing/schema/rules after approval
- stale evaluation evidence

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Hashing ──────────────────────────────────────────────────────────────

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def artifact_set_hash(artifact_dir: Path, include: list[str] | None = None) -> dict[str, Any]:
    """Deterministic hash of all artifacts in a directory.

    Returns {files: {name: sha256}, model_hash: aggregate_sha256}.
    If include is given, only those files are hashed.
    """
    files: dict[str, str] = {}
    if artifact_dir.is_dir():
        for p in sorted(artifact_dir.iterdir()):
            if p.is_file() and not p.name.endswith((".bak", ".tmp", ".json")):
                if include is None or p.name in include:
                    files[p.name] = sha256_file(p)
    digest_src = "\n".join(f"{k}:{v}" for k, v in sorted(files.items()))
    return {
        "files": files,
        "model_hash": hashlib.sha256(digest_src.encode()).hexdigest() if files else None,
    }


# ── Manifest signing ─────────────────────────────────────────────────────

_MANIFEST_SECRET = hashlib.sha256(
    b"PS14-release-manifest-v1-do-not-change"
).digest()


def _sign_manifest(payload: str) -> str:
    return hmac.new(_MANIFEST_SECRET, payload.encode(), hashlib.sha256).hexdigest()


# ── Release Manifest ─────────────────────────────────────────────────────

@dataclass
class ReleaseManifest:
    """Immutable, signed release record binding all components of a model release."""

    # Identity
    release_id: str = ""
    model_id: str = ""
    model_version: str = ""

    # Artifact binding
    artifact_hash: str = ""          # aggregate hash of model artifacts
    artifact_files: dict[str, str] = field(default_factory=dict)  # name -> sha256

    # Component binding
    feature_version: str = ""
    schema_version: str = ""
    preprocessing_hash: str = ""     # hash of preprocessing artifacts/files
    rule_hash: str = ""              # hash of rule config (rules.yaml etc.)
    evaluation_record_hash: str = "" # hash of the evaluation record used for promotion
    training_config_hash: str = ""   # hash of training config/hyperparameters

    # Provenance
    source_git_sha: str = ""
    python_version: str = ""
    dependency_fingerprint: str = "" # hash of requirements/lock file
    training_seed: int | None = None
    created_at: float = field(default_factory=time.time)

    # Gate evidence
    gate_verdict: str = ""           # PROMOTION_ELIGIBLE etc.
    gate_timestamp: float = 0.0

    # Signature
    _nonce: str = ""
    _signature: str = ""

    # ── Canonical serialization ──────────────────────────────────────────

    def _canonical_dict(self) -> dict:
        """Deterministic dict for hashing — sorted keys, no _private fields."""
        d = {
            "release_id": self.release_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "artifact_hash": self.artifact_hash,
            "artifact_files": self.artifact_files,
            "feature_version": self.feature_version,
            "schema_version": self.schema_version,
            "preprocessing_hash": self.preprocessing_hash,
            "rule_hash": self.rule_hash,
            "evaluation_record_hash": self.evaluation_record_hash,
            "training_config_hash": self.training_config_hash,
            "source_git_sha": self.source_git_sha,
            "python_version": self.python_version,
            "dependency_fingerprint": self.dependency_fingerprint,
            "training_seed": self.training_seed,
            "created_at": self.created_at,
            "gate_verdict": self.gate_verdict,
            "gate_timestamp": self.gate_timestamp,
        }
        return d

    def _canonical_json(self) -> str:
        """Canonical JSON: sorted keys, no extra whitespace, stable floats."""
        return json.dumps(self._canonical_dict(), sort_keys=True, separators=(",", ":"))

    def compute_manifest_hash(self) -> str:
        """SHA-256 of the canonical manifest JSON."""
        return hashlib.sha256(self._canonical_json().encode()).hexdigest()

    def sign(self) -> None:
        """Sign the manifest.  Must be called before saving."""
        self._nonce = secrets.token_hex(8)
        payload = self._canonical_json() + "|" + self._nonce
        self._signature = _sign_manifest(payload)

    def verify_signature(self) -> tuple[bool, str]:
        """Verify the manifest signature.  Returns (ok, reason)."""
        if not self._signature or not self._nonce:
            return False, "Manifest not signed"
        payload = self._canonical_json() + "|" + self._nonce
        expected = _sign_manifest(payload)
        if not hmac.compare_digest(self._signature, expected):
            return False, "Manifest signature invalid — forged or tampered"
        return True, "Signature verified"

    def verify_artifacts(self, artifact_dir: Path) -> tuple[bool, str]:
        """Verify that on-disk artifacts match the manifest."""
        if not artifact_dir.is_dir():
            return False, f"Artifact directory missing: {artifact_dir}"
        actual = artifact_set_hash(artifact_dir)
        if actual["model_hash"] != self.artifact_hash:
            return False, (
                f"Artifact hash mismatch: "
                f"actual={actual['model_hash'][:16]}… "
                f"manifest={self.artifact_hash[:16]}…"
            )
        # Check individual files
        missing = [n for n in self.artifact_files if n not in actual["files"]]
        extra = [n for n in actual["files"] if n not in self.artifact_files]
        if missing:
            return False, f"Missing artifacts: {missing}"
        for name in self.artifact_files:
            if actual["files"].get(name) != self.artifact_files[name]:
                return False, f"Artifact file changed: {name}"
        return True, "Artifacts verified"

    def verify_binding(
        self,
        expected_model_id: str = "",
        expected_feature_version: str = "",
        expected_artifact_hash: str = "",
    ) -> tuple[bool, str]:
        """Verify the manifest is bound to the expected release identity."""
        if expected_model_id and self.model_id != expected_model_id:
            return False, (
                f"Model ID mismatch: manifest={self.model_id} "
                f"expected={expected_model_id}"
            )
        if expected_feature_version and self.feature_version != expected_feature_version:
            return False, (
                f"Feature version mismatch: manifest={self.feature_version} "
                f"expected={expected_feature_version}"
            )
        if expected_artifact_hash and self.artifact_hash != expected_artifact_hash:
            return False, (
                f"Artifact hash mismatch: manifest={self.artifact_hash[:16]}… "
                f"expected={expected_artifact_hash[:16]}…"
            )
        return True, "Binding verified"

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = self._canonical_dict()
        d["_nonce"] = self._nonce
        d["_signature"] = self._signature
        d["_manifest_hash"] = self.compute_manifest_hash()
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> ReleaseManifest:
        # Extract private fields before stripping
        nonce = data.get("_nonce", "")
        sig = data.get("_signature", "")
        d = {k: v for k, v in data.items() if not k.startswith("_")}
        m = cls(**d)
        m._nonce = nonce
        m._signature = sig
        return m

    @classmethod
    def load(cls, path: Path) -> ReleaseManifest:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ── Factory ──────────────────────────────────────────────────────────────

def create_release_manifest(
    *,
    release_id: str,
    model_id: str,
    model_version: str,
    artifact_dir: Path,
    feature_version: str = "",
    schema_version: str = "",
    preprocessing_hash: str = "",
    rule_hash: str = "",
    evaluation_record_hash: str = "",
    training_config_hash: str = "",
    source_git_sha: str = "",
    python_version: str = "",
    dependency_fingerprint: str = "",
    training_seed: int | None = None,
    gate_verdict: str = "",
    gate_timestamp: float = 0.0,
) -> ReleaseManifest:
    """Create and sign a release manifest from components."""
    art = artifact_set_hash(artifact_dir)
    manifest = ReleaseManifest(
        release_id=release_id,
        model_id=model_id,
        model_version=model_version,
        artifact_hash=art["model_hash"] or "",
        artifact_files=art["files"],
        feature_version=feature_version,
        schema_version=schema_version or feature_version,
        preprocessing_hash=preprocessing_hash,
        rule_hash=rule_hash,
        evaluation_record_hash=evaluation_record_hash,
        training_config_hash=training_config_hash,
        source_git_sha=source_git_sha,
        python_version=python_version,
        dependency_fingerprint=dependency_fingerprint,
        training_seed=training_seed,
        created_at=time.time(),
        gate_verdict=gate_verdict,
        gate_timestamp=gate_timestamp,
    )
    manifest.sign()
    return manifest
