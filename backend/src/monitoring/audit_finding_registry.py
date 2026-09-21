"""Phase 101: Audit Finding Registry.

Immutable finding records for tracking audit findings through
disposition lifecycle. Prevents silent remediation claims.

Does NOT perform RWV, acquire data, modify models, or promote.
STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ══════════════════════════════════════════════════════════════════════
# FINDING STATES
# ══════════════════════════════════════════════════════════════════════

class FindingStatus(str, Enum):
    OPEN = "open"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    REMEDIATED = "remediated"
    ACCEPTED_RISK = "accepted_risk"
    DEFERRED = "deferred"
    REJECTED_WITH_EVIDENCE = "rejected_with_evidence"


class FindingSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


# ══════════════════════════════════════════════════════════════════════
# HASHING
# ══════════════════════════════════════════════════════════════════════

def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def _hash_dict(d: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(d).encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════
# FINDING RECORD (frozen)
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AuditFinding:
    """Immutable audit finding record."""
    finding_id: str
    source_phase: int
    invariant_id: str
    title: str
    severity: str
    status: str
    description: str
    evidence: str
    root_cause: str
    affected_components: tuple[str, ...]
    remediation: str
    regression_test: str
    created_at: str
    resolved_at: str
    resolution_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["affected_components"] = list(d["affected_components"])
        return d

    def to_canonical(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ══════════════════════════════════════════════════════════════════════
# FINDING REGISTRY
# ══════════════════════════════════════════════════════════════════════

class AuditFindingRegistry:
    """Append-only finding registry."""

    def __init__(self) -> None:
        self._findings: dict[str, AuditFinding] = {}

    def register(self, finding: AuditFinding) -> None:
        """Register a new finding. Rejects duplicate finding_id."""
        if finding.finding_id in self._findings:
            raise ValueError(f"Duplicate finding_id: {finding.finding_id}")
        self._findings[finding.finding_id] = finding

    def get(self, finding_id: str) -> AuditFinding | None:
        return self._findings.get(finding_id)

    def all_findings(self) -> list[AuditFinding]:
        return list(self._findings.values())

    def open_findings(self) -> list[AuditFinding]:
        return [f for f in self._findings.values() if f.status == FindingStatus.OPEN.value]

    def remediated_findings(self) -> list[AuditFinding]:
        return [f for f in self._findings.values() if f.status == FindingStatus.REMEDIATED.value]


# ══════════════════════════════════════════════════════════════════════
# FACTORY: create finding
# ══════════════════════════════════════════════════════════════════════

def create_finding(
    finding_id: str,
    source_phase: int,
    invariant_id: str,
    title: str,
    severity: str,
    description: str,
    evidence: str,
    root_cause: str,
    affected_components: tuple[str, ...],
    remediation: str,
    regression_test: str,
    status: str = FindingStatus.OPEN.value,
) -> AuditFinding:
    """Create a deterministic finding record."""
    ts = _now_iso()
    finding = AuditFinding(
        finding_id=finding_id,
        source_phase=source_phase,
        invariant_id=invariant_id,
        title=title,
        severity=severity,
        status=status,
        description=description,
        evidence=evidence,
        root_cause=root_cause,
        affected_components=affected_components,
        remediation=remediation,
        regression_test=regression_test,
        created_at=ts,
        resolved_at="",
        resolution_hash="",
    )
    resolution_hash = _hash_dict(finding.to_canonical())
    return AuditFinding(**{**finding.__dict__, "resolution_hash": resolution_hash})
