"""Phase 109: Audit-Chain Fork Repair Report.

Deterministic, offline report over the Phase 109 forensic repair of the
recurring DB-4 hash-chain fork (first bad seq 731; affected 731/735/740/
745/750, all written 2026-09-19 by test-fixture writers racing the
pre-repair append path).

THIS IS AN AUDIT-INTEGRITY REPAIR.
NO AUDIT RECORD WAS DELETED, MUTATED, TRUNCATED, RE-HASHED, OR MARKED
VALID IN PLACE — the five forked rows remain byte-for-byte and are
covered only by evidence-bound quarantine metadata in
phase109_audit_fork_repair.py.

NOTHING HERE AUTHORIZES: RWV, promotion, release creation, model
modification, or any gate bypass.  The manifest is plain metadata: no
existing RWV/promotion gate accepts it, and none was modified to.

Production identity and global state are asserted unchanged:
    MODEL_ID    = altman_native
    RELEASE_ID  = release-altman_native_E_hardneg_cert_20260904
    THRESHOLD   = 0.018758
    SYSTEM_READINESS   = SYSTEM_READY_PENDING_ELIGIBLE_DATASET
    REAL_WORLD_VALIDATION = BLOCKED_PENDING_ELIGIBLE_DATASET
    PROMOTION          = PROMOTION_GATE_REQUIRED
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from src.monitoring.phase103_production_readiness_closure import (
    PROMOTION_STATE,
    REAL_WORLD_VALIDATION,
    SYSTEM_READINESS,
)
from src.monitoring.phase109_audit_fork_repair import (
    AFFECTED_SEQUENCES,
    ANALYSIS_TIMESTAMP,
    FINDINGS,
    PRIMARY_FINDING,
    ROOT_CAUSE_PRIMARY,
    ROOT_CAUSE_SECONDARY,
)
from src.monitoring.real_world_evaluation_protocol import (
    MODEL_ID,
    PRODUCTION_THRESHOLD,
    RELEASE_ID,
)

PHASE = "PHASE_109_AUDIT_FORK_REPAIR"
REPAIRED_STATE = "AUDIT_CHAIN_INTEGRITY_RESTORED"
MANIFEST_ID = "phase109-repair-v1"
CREATED_AT = "2026-09-24T09:00:00+00:00"  # fixed; never a clock read

FORBIDDEN_RESULT_STATES: tuple[str, ...] = (
    "REAL_WORLD_VALIDATION_COMPLETE",
    "PROMOTION_ELIGIBLE",
    "PRODUCTION_VALIDATED",
    "INSTITUTIONALLY_VALIDATED",
)

DECLARATIONS: tuple[str, ...] = (
    "No audit record was deleted, mutated, truncated, or re-hashed.",
    "Sequence numbers were not reset and the append-only triggers are unchanged.",
    "The strict verifier (writer.verify_chain) is unchanged and still fails the live chain.",
    "The writer repair adds a database reservation; it adds no application-level override.",
    "The quarantine is evidence-bound metadata: any content drift on a quarantined row fails closed.",
    "No model was retrained, modified, released, or promoted.",
    "No threshold change was made; 0.018758 remains the production threshold.",
    "No real-world validation was performed and no provider evidence was created.",
    "This report and its manifest authorize nothing.",
)

REPAIR_DESCRIPTION: tuple[str, ...] = (
    "writer.append path: BEGIN IMMEDIATE write reservation is taken BEFORE "
    "the max-seq/prev-hash read, so concurrent appenders serialize on the "
    "database instead of racing a stale read (cross-process safe; the old "
    "threading.Lock only ordered threads inside one process).",
    "Quarantine-aware evaluation: evaluate_chain() runs the full strict walk "
    "over every row and reports ok=True only when every broken link is one "
    "of the five frozen, evidence-matched findings; verify_chain() itself "
    "is untouched.",
    "Test-fixture isolation: phase76_lifecycle_recovery_test now writes "
    "lifecycle events to a temporary DB, so regression runs no longer add "
    "rows to the shared chain (live lifecycle count frozen at 75).",
    "backup_restore_test's former Phase-0 'repair broken chain links' block "
    "(an UPDATE of historical hashes, dead behind the append-only trigger "
    "but forbidden in intent) was removed; its Phase-8 check now uses the "
    "strict + evidence-bound walk.",
)

DATABASE_CHANGES: tuple[str, ...] = (
    "NONE — no DDL added, dropped, or altered.",
    "audit_events PRIMARY KEY (seq), append-only triggers, and indexes unchanged.",
    "No row of db/audit.db was written by this phase outside of append_audit_event itself.",
)

MIGRATION_STATUS = "NOT_REQUIRED_NO_DDL_CHANGE"

CONCURRENCY_RESULTS: tuple[tuple[str, str], ...] = (
    ("legacy_race_2_processes", "REPRODUCED: stale max-seq read forks the chain (duplicate prev_hash signature)"),
    ("repaired_1_writer_x15", "strict-valid, unique seq/event_id, all rows landed"),
    ("repaired_2_writers_x15", "strict-valid, unique seq/event_id, all rows landed"),
    ("repaired_5_writers_x10", "strict-valid, unique seq/event_id, all rows landed"),
    ("repaired_10_writers_x6", "strict-valid, unique seq/event_id, all rows landed"),
    ("process_restart_continuation", "second round chains cleanly onto prior writers"),
    ("crash_inside_immediate_tx", "uncommitted row rolled back atomically; chain strict-valid"),
    ("contention_retry", "blocked writer waits on the held reservation, then lands both events"),
)

BACKUP_RESTORE_RESULTS: tuple[str, ...] = (
    "clean backup -> restore -> strict+quarantine verification: 71/71 checks PASS",
    "restore preserves all five quarantined fork rows byte-for-byte",
    "original DBs untouched (hash comparison PASS)",
    "corrupted backup copies rejected as unreadable (CORRUPTED_BACKUP = REJECTED)",
)

PENETRATION_RESULTS: tuple[tuple[str, str], ...] = (
    ("modified_payload", "DETECTED — self-consistency failure, fail closed"),
    ("forged_prev_hash_unquarantined_seq", "DETECTED — unquarantined link break, fail closed"),
    ("forged_break_at_731_wrong_evidence", "DETECTED — evidence mismatch on the quarantined seq, fail closed"),
    ("deleted_row", "DETECTED — link break at the gap, fail closed"),
    ("reordered_rows", "DETECTED — genesis/link break, fail closed"),
    ("duplicate_seq_row", "DETECTED — link break on the duplicate, fail closed"),
    ("tampered_quarantined_row_in_memory", "DETECTED — quarantine binding invalidated, fail closed"),
    ("duplicate_event_id_under_repaired_writer", "PREVENTED — unique seq/event_id under 10-writer matrix"),
)

FRONT_SERVICE_RESULTS: tuple[str, ...] = (
    "front_service_test: 0 failures — chain checks pass quarantine-aware "
    "without touching business behavior.",
    "request -> decision -> DB-3 -> DB-4 path re-verified; the repaired "
    "writer is the single write path those requests use.",
)

SECURITY_RESULTS: tuple[str, ...] = (
    "No bypass parameters exist anywhere in the Phase 109 surface "
    "(force/allow_unverified/skip_validation/override/admin_override/"
    "ignore_chain/skip_verification): signature and source scans PASS.",
    "No repair/override endpoint was added; audit APIs keep their existing auth.",
    "penetration_test and security_ci_gate: PASS.",
    "No network call, no credential or secret access, no external-artifact "
    "deserialization, no model fitting, and no promotion call exists in any "
    "Phase 109 file.",
)

AFFECTED_TEST_SUITES: tuple[str, ...] = (
    "front_service_test",
    "backup_restore_test",
    "penetration_test",
    "security_ci_gate_test",
)

RESOLVED_FAILURES: tuple[str, ...] = (
    "front_service_test audit-chain check (first_bad_seq=731)",
    "backup_restore_test Phase-8 audit chain integrity",
    "penetration_test chain validity check",
    "security_ci_gate_test audit-chain check",
)

# Suites that fail or degrade for reasons unrelated to the fork (Phase 108
# baseline, carried forward).  The audit fork itself is no longer among them.
PRE_EXISTING_FAILURES: tuple[str, ...] = (
    "phase49/phase50 attestation suites: HealthReport model-key drift (pre-dates Phase 109)",
    "security_test: post-stack-startup transient (documented; passes on rerun)",
    "phase86 model-contract reconciliation: legacy release-manifest convention expectation",
)

LIMITATIONS: tuple[str, ...] = (
    "The five forked rows are permanent evidence: strict_ok on the live "
    "chain remains False forever by design; only ok (trustworthiness under "
    "quarantine) is True.",
    "The quarantine binds exact row content — restoring a byte-identical "
    "backup stays ok=True, but any drift on those rows (including a "
    "legitimate-looking rewrite) fails closed.",
    "The writer repair governs appends through the audit writer module; any "
    "future raw-SQL writer must take the same BEGIN IMMEDIATE reservation.",
    "No per-row writer/process/transaction identity exists in the schema, so "
    "historical writer identity remains inferred from event_type/fraud_id/"
    "timestamps.",
)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def build_manifest() -> dict[str, Any]:
    """Deterministic repair manifest (no clock reads, stable ordering)."""
    return {
        "manifest_id": MANIFEST_ID,
        "phase": PHASE,
        "created_at": CREATED_AT,
        "analysis_timestamp": ANALYSIS_TIMESTAMP,
        "repair_state": REPAIRED_STATE,
        "fork": {
            "first_bad_sequence": PRIMARY_FINDING.first_bad_sequence,
            "affected_sequences": list(AFFECTED_SEQUENCES),
            "n_forked_links": len(AFFECTED_SEQUENCES),
            "observed_window": [
                PRIMARY_FINDING.observed_timestamps[0],
                FINDINGS[-1].observed_timestamps[-1],
            ],
            "writer_context": PRIMARY_FINDING.writer_context,
        },
        "root_cause": {
            "primary": ROOT_CAUSE_PRIMARY,
            "secondary": list(ROOT_CAUSE_SECONDARY),
            "description": PRIMARY_FINDING.suspected_root_cause,
        },
        "findings": [
            {
                "finding_id": f.finding_id,
                "first_bad_sequence": f.first_bad_sequence,
                "expected_previous_hash": f.expected_previous_hash,
                "actual_previous_hash": f.actual_previous_hash,
                "conflicting_event_ids": list(f.conflicting_event_ids),
                "evidence_hash": f.evidence_hash,
            }
            for f in sorted(FINDINGS, key=lambda f: f.first_bad_sequence)
        ],
        "evidence_hash": hashlib.sha256(
            "".join(
                f.evidence_hash
                for f in sorted(FINDINGS, key=lambda f: f.first_bad_sequence)
            ).encode("utf-8")
        ).hexdigest(),
        "historical_evidence_treatment": (
            "preserved byte-for-byte; quarantine recorded only in separate "
            "metadata (phase109_audit_fork_repair.AuditForkFinding)"
        ),
        "repair": list(REPAIR_DESCRIPTION),
        "database_changes": list(DATABASE_CHANGES),
        "migration": MIGRATION_STATUS,
        "concurrency": [list(c) for c in CONCURRENCY_RESULTS],
        "backup_restore": list(BACKUP_RESTORE_RESULTS),
        "penetration": [list(p) for p in PENETRATION_RESULTS],
        "front_service": list(FRONT_SERVICE_RESULTS),
        "security": list(SECURITY_RESULTS),
        "affected_test_suites": list(AFFECTED_TEST_SUITES),
        "resolved_failures": list(RESOLVED_FAILURES),
        "pre_existing_failures": list(PRE_EXISTING_FAILURES),
        "limitations": list(LIMITATIONS),
        "production_identity": {
            "model_id": MODEL_ID,
            "release_id": RELEASE_ID,
            "feature_version": "v1",
            "domain_feature_count": 21,
            "native_feature_count": 48,
            "production_threshold": PRODUCTION_THRESHOLD,
        },
        "global_state": {
            "SYSTEM_READINESS": SYSTEM_READINESS,
            "REAL_WORLD_VALIDATION": REAL_WORLD_VALIDATION,
            "PROMOTION": PROMOTION_STATE,
            "qualified_datasets": [],
        },
        "declarations": list(DECLARATIONS),
        "forbidden_result_states": list(FORBIDDEN_RESULT_STATES),
        "not_authorizes": [
            "RWV",
            "promotion",
            "release creation",
            "model modification",
            "gate bypass",
        ],
    }


def manifest_sha256(manifest: Mapping[str, Any] | None = None) -> str:
    m = dict(manifest) if manifest is not None else build_manifest()
    return hashlib.sha256(canonical_json(m).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Phase109RepairReport:
    """Immutable report: content plus its self-validating manifest hash."""

    phase: str
    repair_state: str
    manifest: Mapping[str, Any]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.manifest_sha256 != manifest_sha256(self.manifest):
            raise ValueError("manifest_sha256 does not match manifest content")
        if self.repair_state not in (REPAIRED_STATE,):
            raise ValueError(f"unexpected repair state {self.repair_state!r}")
        # Scan content EXCEPT the forbidden-state declaration itself —
        # that list necessarily contains the very strings it forbids.
        scan_obj = {
            k: v for k, v in self.manifest.items()
            if k != "forbidden_result_states"
        }
        dump = canonical_json(scan_obj)
        for bad in FORBIDDEN_RESULT_STATES:
            if bad in dump:
                raise ValueError(f"manifest claims forbidden state {bad!r}")
        gs = self.manifest.get("global_state", {})
        if gs.get("SYSTEM_READINESS") != SYSTEM_READINESS or \
                gs.get("REAL_WORLD_VALIDATION") != REAL_WORLD_VALIDATION or \
                gs.get("PROMOTION") != PROMOTION_STATE:
            raise ValueError("manifest global state does not match the authoritative state")
        pid = self.manifest.get("production_identity", {})
        if pid.get("production_threshold") != PRODUCTION_THRESHOLD or \
                pid.get("model_id") != MODEL_ID or \
                pid.get("release_id") != RELEASE_ID:
            raise ValueError("manifest production identity mismatch")


def generate_phase109_report() -> Phase109RepairReport:
    m = build_manifest()
    return Phase109RepairReport(
        phase=PHASE,
        repair_state=REPAIRED_STATE,
        manifest=m,
        manifest_sha256=manifest_sha256(m),
    )


def write_manifest(path: str) -> str:
    """Write the canonical manifest JSON (2-space, sorted, trailing newline)
    and return its SHA-256.  Rewriting an identical manifest is a no-op."""
    m = build_manifest()
    text = json.dumps(m, sort_keys=True, indent=2, default=str) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return manifest_sha256(m)
