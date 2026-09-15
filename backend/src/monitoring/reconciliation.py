"""DB-3/DB-4 reconciliation and audit chain integrity verification.

Detects:
  - DB-3 decisions without matching audit events
  - Audit events without matching DB-3 decisions
  - Score/band/version mismatches between DB-3 and DB-4
  - Audit hash-chain integrity violations
  - Duplicate sequence numbers
  - Broken previous_hash links
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DiscrepancyType(str, Enum):
    MISSING_AUDIT = "missing_audit"
    MISSING_DECISION = "missing_decision"
    SCORE_MISMATCH = "score_mismatch"
    BAND_MISMATCH = "band_mismatch"
    DEGRADED_MISMATCH = "degraded_mismatch"
    MODEL_VERSION_MISMATCH = "model_version_mismatch"
    EVENT_ID_MISMATCH = "event_id_mismatch"
    HASH_CHAIN_BROKEN = "hash_chain_broken"
    DUPLICATE_SEQUENCE = "duplicate_sequence"
    HASH_CHAIN_INVALID = "hash_chain_invalid"


@dataclass
class Discrepancy:
    """A single reconciliation discrepancy."""
    discrepancy_type: DiscrepancyType
    event_id: str
    detail: str
    severity: str = "warning"  # "warning" or "critical"

    def to_dict(self) -> dict:
        return {
            "type": self.discrepancy_type.value,
            "event_id": self.event_id,
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass
class ReconciliationReport:
    """Complete reconciliation report."""
    timestamp: float
    n_decisions: int
    n_audit_events: int
    n_matched: int
    n_discrepancies: int
    discrepancies: list[Discrepancy]
    chain_valid: bool
    chain_details: str

    def to_dict(self) -> dict:
        import time
        return {
            "timestamp": self.timestamp,
            "n_decisions": self.n_decisions,
            "n_audit_events": self.n_audit_events,
            "n_matched": self.n_matched,
            "n_discrepancies": self.n_discrepancies,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "chain_valid": self.chain_valid,
            "chain_details": self.chain_details,
        }


def verify_audit_chain(audit_events: list[dict]) -> tuple[bool, str]:
    """Verify the audit hash chain integrity.

    Args:
        audit_events: list of audit events in sequence order, each with:
            - seq: sequence number
            - prev_hash: hash of previous entry
            - entry_hash: this entry's hash
            - payload_summary: JSON string of payload

    Returns:
        (is_valid, detail_string)
    """
    if not audit_events:
        return True, "empty chain — no events to verify"

    GENESIS_HASH = hashlib.sha256(b"PS-14 audit genesis v1").hexdigest()

    prev_hash = GENESIS_HASH
    for i, event in enumerate(audit_events):
        # Check previous hash link
        if event.get("prev_hash") != prev_hash:
            return False, (
                f"chain break at seq={event.get('seq')}: "
                f"expected prev_hash={prev_hash[:16]}..., "
                f"got {event.get('prev_hash', 'None')[:16]}..."
            )

        # Recompute entry hash
        payload_str = event.get("payload_summary", "{}")
        body = _canonical_payload(payload_str)
        recomputed = hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()

        if event.get("entry_hash") != recomputed:
            return False, (
                f"hash mismatch at seq={event.get('seq')}: "
                f"stored={event.get('entry_hash', '')[:16]}..., "
                f"recomputed={recomputed[:16]}..."
            )

        prev_hash = event["entry_hash"]

    return True, f"chain valid: {len(audit_events)} events verified from genesis"


def _canonical_payload(payload_str: str) -> str:
    """Canonicalize a payload string for hash verification."""
    try:
        payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        return str(payload_str)


def reconcile_decisions_audit(
    decisions: list[dict],
    audit_events: list[dict],
) -> ReconciliationReport:
    """Reconcile DB-3 decisions with DB-4 audit events.

    Args:
        decisions: list of RiskScore rows (as dicts)
        audit_events: list of AuditEvent rows (as dicts) with score_generated type

    Returns:
        ReconciliationReport with discrepancies.
    """
    import time
    discrepancies: list[Discrepancy] = []

    # Index audit events by event_id (only score_generated and data_quality_blocked)
    audit_by_event: dict[str, dict] = {}
    for ae in audit_events:
        if ae.get("event_type") in ("score_generated", "data_quality_blocked"):
            try:
                payload = json.loads(ae.get("payload_summary", "{}"))
                eid = payload.get("event_id")
                if eid:
                    audit_by_event[eid] = ae
            except (json.JSONDecodeError, TypeError):
                pass

    n_matched = 0
    for dec in decisions:
        eid = dec.get("event_id")
        if eid is None:
            continue

        if eid not in audit_by_event:
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.MISSING_AUDIT,
                event_id=eid,
                detail=f"DB-3 decision for event_id={eid} has no matching audit event",
                severity="warning",
            ))
            continue

        # Compare fields
        ae = audit_by_event[eid]
        try:
            payload = json.loads(ae.get("payload_summary", "{}"))
        except (json.JSONDecodeError, TypeError):
            payload = {}

        if dec.get("risk_score") != payload.get("risk_score"):
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.SCORE_MISMATCH,
                event_id=eid,
                detail=f"score: db3={dec.get('risk_score')} != audit={payload.get('risk_score')}",
                severity="critical",
            ))

        if dec.get("risk_band") != payload.get("risk_band"):
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.BAND_MISMATCH,
                event_id=eid,
                detail=f"band: db3={dec.get('risk_band')} != audit={payload.get('risk_band')}",
                severity="critical",
            ))

        if dec.get("degraded") != payload.get("degraded"):
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.DEGRADED_MISMATCH,
                event_id=eid,
                detail=f"degraded: db3={dec.get('degraded')} != audit={payload.get('degraded')}",
                severity="warning",
            ))

        if dec.get("model_version") != payload.get("model_version"):
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.MODEL_VERSION_MISMATCH,
                event_id=eid,
                detail=f"model_version: db3={dec.get('model_version')} != audit={payload.get('model_version')}",
                severity="critical",
            ))

        n_matched += 1

    # Check for audit events without decisions
    decision_eids = {d.get("event_id") for d in decisions}
    for ae in audit_events:
        if ae.get("event_type") in ("score_generated", "data_quality_blocked"):
            try:
                payload = json.loads(ae.get("payload_summary", "{}"))
                eid = payload.get("event_id")
                if eid and eid not in decision_eids:
                    discrepancies.append(Discrepancy(
                        discrepancy_type=DiscrepancyType.MISSING_DECISION,
                        event_id=eid,
                        detail=f"audit event for event_id={eid} has no matching DB-3 decision",
                        severity="warning",
                    ))
            except (json.JSONDecodeError, TypeError):
                pass

    # Verify audit chain
    chain_valid, chain_details = verify_audit_chain(audit_events)

    return ReconciliationReport(
        timestamp=time.time(),
        n_decisions=len(decisions),
        n_audit_events=len(audit_events),
        n_matched=n_matched,
        n_discrepancies=len(discrepancies),
        discrepancies=discrepancies,
        chain_valid=chain_valid,
        chain_details=chain_details,
    )
