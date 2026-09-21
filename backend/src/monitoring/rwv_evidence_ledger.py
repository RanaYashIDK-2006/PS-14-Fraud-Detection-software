"""Phase 99: RWV Evidence Ledger & Reproducibility Certification.

Append-only, hash-chained evidence ledger for the RWV pipeline.
Every RWV artifact is cryptographically bound to its predecessors.

Does NOT perform RWV, acquire data, modify models, retrain, or promote.
STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.monitoring.rwv_readiness_audit import (
    MODEL_ID,
    RELEASE_ID,
    FEATURE_VERSION,
)


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

LEDGER_SCHEMA_VERSION = "phase99_v1"
NATIVE_FEATURE_VERSION = "v1"
EVAL_PROTOCOL_VERSION = "phase93_v1"
ACCEPTANCE_SPEC_VERSION = "phase91_v1"
PREPROCESSING_HASH = "stable_preprocessing_v1"
RULE_HASH = "rules_v1"
BUNDLE_VERSION = "phase99_bundle_v1"


# ══════════════════════════════════════════════════════════════════════
# EVIDENCE TYPES
# ══════════════════════════════════════════════════════════════════════

class EvidenceType(str, Enum):
    PROVIDER_EVIDENCE = "provider_evidence"
    DATASET_QUALIFICATION = "dataset_qualification"
    RWV_SESSION = "rwv_session"
    RWV_EVALUATION = "rwv_evaluation"
    RWV_ADJUDICATION = "rwv_adjudication"
    RWV_PROMOTION_EVIDENCE = "rwv_promotion_evidence"


class EvidenceStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETAINED = "retained"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class LineageStatus(str, Enum):
    LINEAGE_COMPLETE = "lineage_complete"
    LINEAGE_INCOMPLETE = "lineage_incomplete"
    LINEAGE_BROKEN = "lineage_broken"
    LINEAGE_TAMPERED = "lineage_tampered"
    LINEAGE_STALE = "lineage_stale"


class LedgerVerificationResult(str, Enum):
    VERIFIED = "verified"
    TAMPERED = "tampered"
    BROKEN_CHAIN = "broken_chain"
    MISSING_ENTRY = "missing_entry"
    REORDERED = "reordered"
    DUPLICATE_CONFLICT = "duplicate_conflict"


# ══════════════════════════════════════════════════════════════════════
# HASHING UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, compact, ASCII, stable."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, default=str,
    )


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_dict(d: dict[str, Any]) -> str:
    return _hash_bytes(_canonical_json(d).encode())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_git_sha() -> str:
    """Best-effort source git SHA; 'unknown' if unavailable."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _get_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_platform_info() -> str:
    return f"{platform.system()}-{platform.machine()}"


# ══════════════════════════════════════════════════════════════════════
# ENVIRONMENT FINGERPRINT
# ══════════════════════════════════════════════════════════════════════

def compute_environment_fingerprint() -> str:
    """Deterministic fingerprint of the reproducibility-relevant environment.

    Never includes secrets, tokens, credentials, or private keys.
    """
    components = {
        "python_version": _get_python_version(),
        "platform": _get_platform_info(),
        "source_git_sha": _get_git_sha(),
    }
    return _hash_dict(components)


# ══════════════════════════════════════════════════════════════════════
# LEDGER ENTRY (frozen)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RWVEvidenceLedgerEntry:
    """Single immutable ledger entry."""
    ledger_entry_id: int
    evidence_type: str
    evidence_id: str
    evidence_hash: str
    parent_evidence_hash: str
    model_id: str
    release_id: str
    dataset_id: str
    dataset_version: str
    protocol_version: str
    acceptance_spec_version: str
    created_at: str
    source_git_sha: str
    environment_fingerprint: str
    status: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def to_canonical(self) -> dict[str, Any]:
        """Fields used for hashing — excludes entry_hash itself."""
        return {k: v for k, v in self.__dict__.items() if k != "entry_hash"}


# ══════════════════════════════════════════════════════════════════════
# AUDIT EVENT
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LedgerAuditEvent:
    event_type: str
    ledger_entry_id: int
    timestamp: str
    details: str
    event_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def create_ledger_audit_event(
    event_type: str, entry_id: int, details: str,
) -> LedgerAuditEvent:
    ts = _now_iso()
    raw = f"ledger_audit|{event_type}|{entry_id}|{ts}|{details}"
    return LedgerAuditEvent(
        event_type=event_type, ledger_entry_id=entry_id,
        timestamp=ts, details=details,
        event_hash=_hash_bytes(raw.encode()),
    )


# ══════════════════════════════════════════════════════════════════════
# LEDGER
# ══════════════════════════════════════════════════════════════════════

class RWVEvidenceLedger:
    """Append-only, hash-chained evidence ledger."""

    def __init__(self) -> None:
        self._entries: list[RWVEvidenceLedgerEntry] = []
        self._evidence_ids: set[str] = set()
        self._env_fingerprint = compute_environment_fingerprint()
        self._git_sha = _get_git_sha()

    # ── public API ──────────────────────────────────────────────────

    def append_entry(
        self,
        evidence_type: str,
        evidence_id: str,
        evidence_hash: str,
        *,
        model_id: str = MODEL_ID,
        release_id: str = RELEASE_ID,
        dataset_id: str = "",
        dataset_version: str = "",
        protocol_version: str = EVAL_PROTOCOL_VERSION,
        acceptance_spec_version: str = ACCEPTANCE_SPEC_VERSION,
    ) -> RWVEvidenceLedgerEntry:
        """Append a new entry. Rejects duplicate evidence_id (non-idempotent)."""
        if evidence_id in self._evidence_ids:
            raise ValueError(f"Duplicate evidence_id: {evidence_id}")

        parent_hash = self._entries[-1].entry_hash if self._entries else "genesis"
        entry_id = len(self._entries)
        ts = _now_iso()

        entry = RWVEvidenceLedgerEntry(
            ledger_entry_id=entry_id,
            evidence_type=evidence_type,
            evidence_id=evidence_id,
            evidence_hash=evidence_hash,
            parent_evidence_hash=parent_hash,
            model_id=model_id,
            release_id=release_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            protocol_version=protocol_version,
            acceptance_spec_version=acceptance_spec_version,
            created_at=ts,
            source_git_sha=self._git_sha,
            environment_fingerprint=self._env_fingerprint,
            status=EvidenceStatus.ACTIVE.value,
            entry_hash="",  # placeholder
        )

        # Compute entry_hash from canonical fields + parent
        canonical = entry.to_canonical()
        canonical["_parent_evidence_hash_input"] = parent_hash
        entry_hash = _hash_dict(canonical)

        entry = RWVEvidenceLedgerEntry(**{**entry.__dict__, "entry_hash": entry_hash})

        self._entries.append(entry)
        self._evidence_ids.add(evidence_id)
        return entry

    @property
    def head(self) -> RWVEvidenceLedgerEntry | None:
        return self._entries[-1] if self._entries else None

    @property
    def head_hash(self) -> str:
        return self.head.entry_hash if self.head else "empty"

    @property
    def length(self) -> int:
        return len(self._entries)

    def get_entry(self, ledger_entry_id: int) -> RWVEvidenceLedgerEntry | None:
        if 0 <= ledger_entry_id < len(self._entries):
            return self._entries[ledger_entry_id]
        return None

    def get_by_evidence_id(self, evidence_id: str) -> RWVEvidenceLedgerEntry | None:
        for e in self._entries:
            if e.evidence_id == evidence_id:
                return e
        return None

    def lineage(self, evidence_id: str) -> list[RWVEvidenceLedgerEntry]:
        """Return the ancestry chain for a given evidence_id, oldest-first."""
        entry = self.get_by_evidence_id(evidence_id)
        if entry is None:
            return []
        chain = [entry]
        while chain[-1].parent_evidence_hash != "genesis":
            parent_hash = chain[-1].parent_evidence_hash
            found = False
            for e in reversed(self._entries):
                if e.entry_hash == parent_hash:
                    chain.append(e)
                    found = True
                    break
            if not found:
                break
        chain.reverse()
        return chain

    def lineage_status(self, evidence_id: str) -> str:
        """Determine lineage completeness for a given evidence_id."""
        entry = self.get_by_evidence_id(evidence_id)
        if entry is None:
            return LineageStatus.LINEAGE_BROKEN.value

        chain = self.lineage(evidence_id)
        if len(chain) < 2:
            return LineageStatus.LINEAGE_INCOMPLETE.value

        # Verify chain integrity: each entry's parent must point to the previous entry's hash
        for i in range(1, len(chain)):
            if chain[i].parent_evidence_hash != chain[i - 1].entry_hash:
                return LineageStatus.LINEAGE_BROKEN.value

        return LineageStatus.LINEAGE_COMPLETE.value

    def verify_chain(self) -> LedgerVerificationResult:
        """Verify the entire ledger chain integrity."""
        if not self._entries:
            return LedgerVerificationResult.VERIFIED

        prev_hash = "genesis"
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()

        for i, entry in enumerate(self._entries):
            # Check sequence
            if entry.ledger_entry_id != i:
                return LedgerVerificationResult.REORDERED

            # Check parent chain
            if entry.parent_evidence_hash != prev_hash:
                return LedgerVerificationResult.BROKEN_CHAIN

            # Check duplicate evidence_id
            if entry.evidence_id in seen_ids:
                return LedgerVerificationResult.DUPLICATE_CONFLICT
            seen_ids.add(entry.evidence_id)

            # Recompute entry_hash
            canonical = entry.to_canonical()
            canonical["_parent_evidence_hash_input"] = entry.parent_evidence_hash
            expected_hash = _hash_dict(canonical)
            if expected_hash != entry.entry_hash:
                return LedgerVerificationResult.TAMPERED

            prev_hash = entry.entry_hash
            seen_hashes.add(entry.entry_hash)

        return LedgerVerificationResult.VERIFIED

    # ── export / import ─────────────────────────────────────────────

    def export_bundle(self) -> dict[str, Any]:
        """Export the ledger as a deterministic evidence bundle."""
        entries_data = [e.to_dict() for e in self._entries]
        ledger_hash = _hash_dict({
            "entries": entries_data,
            "schema_version": LEDGER_SCHEMA_VERSION,
            "length": len(self._entries),
        })

        bundle = {
            "bundle_version": BUNDLE_VERSION,
            "schema_version": LEDGER_SCHEMA_VERSION,
            "root_evidence_hash": self._entries[0].entry_hash if self._entries else "",
            "ledger_hash": ledger_hash,
            "ledger_length": len(self._entries),
            "entries": entries_data,
            "environment_fingerprint": self._env_fingerprint,
            "source_git_sha": self._git_sha,
            "exported_at": _now_iso(),
        }

        bundle_content = {k: v for k, v in bundle.items() if k != "bundle_hash"}
        bundle["bundle_hash"] = _hash_dict(bundle_content)
        return bundle

    @staticmethod
    def verify_bundle(bundle: dict[str, Any]) -> tuple[bool, str]:
        """Verify an exported bundle's integrity. Returns (valid, reason)."""
        if "bundle_version" not in bundle:
            return False, "Missing bundle_version"
        if "entries" not in bundle:
            return False, "Missing entries"
        if "bundle_hash" not in bundle:
            return False, "Missing bundle_hash"

        stored_hash = bundle["bundle_hash"]
        bundle_content = {k: v for k, v in bundle.items() if k != "bundle_hash"}
        recomputed = _hash_dict(bundle_content)
        if stored_hash != recomputed:
            return False, "Bundle hash mismatch — tampered"

        # Verify chain within bundle
        entries = bundle["entries"]
        prev_hash = "genesis"
        seen_ids: set[str] = set()
        for e_data in entries:
            if e_data.get("parent_evidence_hash") != prev_hash:
                return False, "Broken chain in bundle"
            if e_data.get("evidence_id") in seen_ids:
                return False, "Duplicate evidence_id in bundle"
            seen_ids.add(e_data["evidence_id"])

            # Recompute entry_hash
            canonical = {k: v for k, v in e_data.items() if k != "entry_hash"}
            canonical["_parent_evidence_hash_input"] = e_data["parent_evidence_hash"]
            expected = _hash_dict(canonical)
            if expected != e_data.get("entry_hash"):
                return False, f"Entry hash mismatch at index {e_data.get('ledger_entry_id')}"

            prev_hash = e_data["entry_hash"]

        return True, "Bundle verified"

    @classmethod
    def import_from_bundle(cls, bundle: dict[str, Any]) -> RWVEvidenceLedger | None:
        """Import a ledger from a verified bundle. Returns None if invalid."""
        valid, reason = cls.verify_bundle(bundle)
        if not valid:
            return None

        ledger = cls.__new__(cls)
        ledger._entries = []
        ledger._evidence_ids = set()
        ledger._env_fingerprint = bundle.get("environment_fingerprint", "")
        ledger._git_sha = bundle.get("source_git_sha", "")

        for e_data in bundle["entries"]:
            entry = RWVEvidenceLedgerEntry(**{
                k: v for k, v in e_data.items()
                if k in RWVEvidenceLedgerEntry.__dataclass_fields__
            })
            ledger._entries.append(entry)
            ledger._evidence_ids.add(entry.evidence_id)

        return ledger


# ══════════════════════════════════════════════════════════════════════
# DETECTION UTILITIES
# ══════════════════════════════════════════════════════════════════════

def detect_entry_tampering(
    entry: RWVEvidenceLedgerEntry,
) -> tuple[bool, str]:
    """Recompute entry_hash and compare. Returns (is_valid, reason)."""
    canonical = entry.to_canonical()
    canonical["_parent_evidence_hash_input"] = entry.parent_evidence_hash
    expected = _hash_dict(canonical)
    if expected == entry.entry_hash:
        return True, "Entry hash verified"
    return False, "Entry hash mismatch — tampered"


def detect_chain_break(ledger: RWVEvidenceLedger) -> tuple[bool, str]:
    """Check parent_hash linkage. Returns (is_valid, reason)."""
    result = ledger.verify_chain()
    if result == LedgerVerificationResult.VERIFIED:
        return True, "Chain verified"
    return False, f"Chain broken: {result.value}"
