"""Phase 109: Audit-chain fork forensic repair & evidence-bound quarantine.

FORENSIC FINDINGS (immutable once written)
-------------------------------------------
Between 2026-09-19 07:56:32 and 08:02:53 UTC, five hash-chain links in
DB-4 (db/audit.db) were written with a stale predecessor hash: two writer
domains (two overlapping runs of scripts/phase76_lifecycle_recovery_test.py)
each read the same last row — the pre-Phase-109 writer took no database
reservation before its max-seq read, and its ``threading.Lock`` only
ordered threads inside one process — then both chained onto that same
predecessor.  The loser's insert landed at the next autoincrement seq, so
seq N carries prev_hash = entry_hash(N-2) instead of entry_hash(N-1).
Affected sequences: 731, 735, 740, 745, 750.  Every affected row is
*internally* consistent (entry_hash == sha256(own prev + canonical
payload)); only the link to its predecessor is wrong, and everything
downstream chains self-consistently onto the stored hashes.

Classification
--------------
primary_cause   = CONCURRENT_WRITER_ORDERING
secondary_causes = (TEST_FIXTURE_POLLUTION,)

Historical evidence (Part 3)
----------------------------
The forked rows are preserved byte-for-byte — no delete, no UPDATE, no
re-hash.  They were test-generated events (event_type lifecycle_test_*,
fraud_id CHAIN-*), i.e. fixture data that leaked into the shared chain;
they are distinguished here as a QUARANTINE recorded in this separate
metadata structure, never as "silently valid" history.  Each quarantine
entry binds the offending row's full content (event id, type, fraud id,
payload digest, entry hash, timestamp) into an evidence hash: tampering
with a quarantined row invalidates its finding and the chain fails
closed again.

Verification semantics
----------------------
``writer.verify_chain`` (the strict verifier) is UNCHANGED.
``evaluate_chain(rows)`` runs the same strict checks over every row
(genesis link, per-row link, per-row hash recomputation, JSON payload
validity) and only then applies the quarantine: the set of broken links
must be a subset of the five evidence-bound sequences, and each broken
row must match its frozen finding field-for-field.  Any extra break, any
self-inconsistency, any evidence mismatch => ok=False.  No bypass
parameters exist.

This module performs no I/O beyond an optional read-only forensic
re-derivation helper; it never mutates audit data.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.audit_service.writer import GENESIS_HASH, canonical

# ── Classification (Part 2) ───────────────────────────────────────────
ROOT_CAUSE_PRIMARY = "CONCURRENT_WRITER_ORDERING"
ROOT_CAUSE_SECONDARY: tuple[str, ...] = ("TEST_FIXTURE_POLLUTION",)

# Fixed forensic-analysis timestamp (deterministic; not a runtime clock read)
ANALYSIS_TIMESTAMP = "2026-09-24T07:30:00+00:00"

WRITER_CONTEXT = (
    "scripts/phase76_lifecycle_recovery_test.py SECTION 7 called "
    "append_audit_event(...) three times per run (queue + background writer "
    "thread) from two overlapping test processes against the shared "
    "db/audit.db on 2026-09-19 07:56-08:03; the pre-Phase-109 writer held "
    "only a per-process threading.Lock and took no SQLite write reservation "
    "before its 'SELECT max(seq)' read, so both processes observed the same "
    "last row; no per-row writer/process/transaction identity column exists "
    "in the schema — identity is inferred from event_type/fraud_id/timestamps; "
    "SQLite autocommit statements expose no transaction ids"
)

SUSPECTED_ROOT_CAUSE = (
    "cross-process stale max-seq read: two writer domains both chained onto "
    "the same predecessor entry hash; the later insert kept the stale "
    "prev_hash while SQLite autoincrement gave it the next seq, producing a "
    "link whose prev_hash equals the predecessor-of-predecessor's entry hash"
)


def _evidence_digest(fields: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
        .encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AuditForkFinding:
    """Immutable forensic record for one broken chain link (Part 1)."""

    finding_id: str
    first_bad_sequence: int
    expected_previous_hash: str
    actual_previous_hash: str
    conflicting_event_ids: tuple[str, ...]
    observed_timestamps: tuple[str, ...]
    writer_context: str
    suspected_root_cause: str
    evidence_hash: str
    analysis_timestamp: str
    # Evidence binding for the quarantined row itself (Parts 3/9)
    offender_event_type: str
    offender_fraud_id: str
    offender_entry_hash: str
    offender_payload_sha256: str
    offender_created_at: str

    def __post_init__(self) -> None:
        # evidence_hash is derived from every other field: the record is
        # self-validating and immutable once constructed.
        digest = _evidence_digest(self.binding_fields())
        if self.evidence_hash and self.evidence_hash != digest:
            raise ValueError(
                f"{self.finding_id}: evidence_hash does not match its fields"
            )
        object.__setattr__(self, "evidence_hash", digest)

    def binding_fields(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "first_bad_sequence": self.first_bad_sequence,
            "expected_previous_hash": self.expected_previous_hash,
            "actual_previous_hash": self.actual_previous_hash,
            "conflicting_event_ids": list(self.conflicting_event_ids),
            "observed_timestamps": list(self.observed_timestamps),
            "offender_event_type": self.offender_event_type,
            "offender_fraud_id": self.offender_fraud_id,
            "offender_entry_hash": self.offender_entry_hash,
            "offender_payload_sha256": self.offender_payload_sha256,
            "offender_created_at": self.offender_created_at,
        }

    def matches_row(self, row: Any) -> bool:
        """Evidence binding: does this live row (dict or ORM) still carry
        exactly the content this finding quarantined?  Any mismatch means
        the quarantine no longer applies and verification must fail."""
        get = _get
        try:
            return (
                str(get(row, "event_id")) == self.conflicting_event_ids[1]
                and str(get(row, "event_type")) == self.offender_event_type
                and str(get(row, "fraud_id")) == self.offender_fraud_id
                and str(get(row, "prev_hash")) == self.actual_previous_hash
                and str(get(row, "entry_hash")) == self.offender_entry_hash
                and hashlib.sha256(
                    str(get(row, "payload_summary")).encode("utf-8")
                ).hexdigest()
                == self.offender_payload_sha256
                and str(get(row, "created_at")) == self.offender_created_at
            )
        except Exception:
            return False


def _get(row: Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    return getattr(row, key)


# ── The five frozen findings (computed once, at import) ──────────────
# Row facts read forensically on 2026-09-24 from db/audit.db (read-only);
# historical rows are append-only-protected and were never modified.
_RAW_FINDINGS: tuple[dict[str, Any], ...] = (
    {
        "seq": 731,
        "expected": "6ddb2c14e87631d7ce117621d392978044750c179da9eef264111831829ea3e5",
        "actual": "bfa463fa6b4d5d63b543573dc419e9308555fa38681d11268795f7c57eaa6faa",
        "pred_event_id": "5f3b3441-28f6-4b6e-89dd-e5724a041f01",
        "offender_event_id": "4153fa4a-57c1-4d1c-ad32-078661ef5866",
        "pred_ts": "2026-09-19 07:56:32",
        "offender_ts": "2026-09-19 07:56:32",
        "event_type": "lifecycle_test_2",
        "fraud_id": "CHAIN-TEST",
        "entry": "140df258c7deb7a8cc29281bd4c9b3739f734b51edef917a13dcd1f945388623",
        "payload_sha": "a7801198f3e90b2b2fa82239fc6fc265972292c2c2b967f7684b007c4ed9d8ab",
    },
    {
        "seq": 735,
        "expected": "378f0277bab7c1d7843380ea8721fe8e55098e2cb8c7fd69af66b79906ee391f",
        "actual": "57d96a33f50b5b724c625241662eec3c8fe49c9ce0abad7517e33318c8070117",
        "pred_event_id": "d2747086-9f77-43e2-af9d-b8764ee3ea65",
        "offender_event_id": "b366f62f-9746-4e7e-b9c9-0ed634cd591c",
        "pred_ts": "2026-09-19 07:58:29",
        "offender_ts": "2026-09-19 07:58:29",
        "event_type": "lifecycle_test_2",
        "fraud_id": "CHAIN-TEST",
        "entry": "c95c42faae6210a65a4ebe88599973e10355bceabe8d7880645b79cc2908b71f",
        "payload_sha": "e53680c7abb4d7ecfa4b6abdaa8041245f97b179d8f04d13f017512ee3bc141c",
    },
    {
        "seq": 740,
        "expected": "7114d8231a726ae1988cdd081e58e55cddd15e21b5a91b516bb9749862aa48b9",
        "actual": "543d8821ca9635fe97f11786193d8c5e2a9f15da02be65f7f8d2d7ab8179b530",
        "pred_event_id": "2ec15f82-dbec-4ebc-92d7-83e3e32426c4",
        "offender_event_id": "bb79a795-4085-4f2f-ab6d-480559baee91",
        "pred_ts": "2026-09-19 08:00:18",
        "offender_ts": "2026-09-19 08:00:18",
        "event_type": "lifecycle_test_1",
        "fraud_id": "CHAIN-92912C67",
        "entry": "44d05d394b8e91deccf9db54a67c91e24cb8a3a4897b551b231db78df5a1b0ab",
        "payload_sha": "01244772e26a2fdf026bbf629e0a71ffb92fd717b8db3172b62bdcfd1cf9a197",
    },
    {
        "seq": 745,
        "expected": "1c211f26f270dbe93e0a1965aa6c7a08d69e942339342115ea38275c01a8a1f0",
        "actual": "09e952a5f0c942bafb10b40bd299b3f0797462c8ae652ee5fcba3d187b472db1",
        "pred_event_id": "f1fa7789-d98c-4b87-ae74-c9d713b14f86",
        "offender_event_id": "5f2c4ddb-7d00-4789-82ad-889ae6fc470f",
        "pred_ts": "2026-09-19 08:01:45",
        "offender_ts": "2026-09-19 08:01:45",
        "event_type": "lifecycle_test_2",
        "fraud_id": "CHAIN-08CAFC2D",
        "entry": "deff7e88038143711a5911c9caa9cafccea1da15c3954e728003fa3dbcf8925d",
        "payload_sha": "7b34bccaae55c4400e00d0b723f32f9af40249c466b8c3ec66b594f99758c6a3",
    },
    {
        "seq": 750,
        "expected": "eb952f94ad998d7d3e85d02cb28f3a236780c6f715874b5ff3944d93c4141464",
        "actual": "eb5f6190d163e502e5442d58a56523a857b7d26809b8ab63b85f063b1cbc780f",
        "pred_event_id": "cf4d381d-7e58-4e0d-94db-5ab830915d14",
        "offender_event_id": "e9f98114-1c5f-4a86-9059-ddfe13f172de",
        "pred_ts": "2026-09-19 08:02:53",
        "offender_ts": "2026-09-19 08:02:53",
        "event_type": "lifecycle_test_2",
        "fraud_id": "CHAIN-60DA2D5B",
        "entry": "da7322f9034b8d96a42e0621eec042998189031202fff4c7a8cbaf949dfb9b71",
        "payload_sha": "3b6b18db9cfb8d5ed0b3c759a8afbc3c3cb558322deb7be54c7d1d66f6ee6097",
    },
)

FINDINGS: tuple[AuditForkFinding, ...] = tuple(
    AuditForkFinding(
        finding_id=f"P109-FORK-{r['seq']}",
        first_bad_sequence=r["seq"],
        expected_previous_hash=r["expected"],
        actual_previous_hash=r["actual"],
        conflicting_event_ids=(r["pred_event_id"], r["offender_event_id"]),
        observed_timestamps=(r["pred_ts"], r["offender_ts"]),
        writer_context=WRITER_CONTEXT,
        suspected_root_cause=SUSPECTED_ROOT_CAUSE,
        evidence_hash="",  # derived in __post_init__
        analysis_timestamp=ANALYSIS_TIMESTAMP,
        offender_event_type=r["event_type"],
        offender_fraud_id=r["fraud_id"],
        offender_entry_hash=r["entry"],
        offender_payload_sha256=r["payload_sha"],
        offender_created_at=r["offender_ts"],
    )
    for r in _RAW_FINDINGS
)

FINDING_BY_SEQ: dict[int, AuditForkFinding] = {
    f.first_bad_sequence: f for f in FINDINGS
}
AFFECTED_SEQUENCES: tuple[int, ...] = tuple(
    sorted(FINDING_BY_SEQ)
)
PRIMARY_FINDING: AuditForkFinding = FINDINGS[0]  # seq 731 (first divergent)


# ── Quarantine-aware chain evaluation ─────────────────────────────────
def evaluate_chain(rows: Iterable[Any]) -> dict:
    """Strict verification of every row, then evidence-bound quarantine.

    ``rows`` must be ordered by seq ascending (dicts or ORM objects with
    seq/event_id/fraud_id/event_type/prev_hash/entry_hash/payload_summary/
    created_at).  Returns::

        ok                chain is trustworthy: either strictly valid, or
                          every broken link is one of the five frozen,
                          evidence-matched findings
        strict_ok         raw verdict of the strict walk (no quarantine)
        n_entries         number of rows evaluated
        first_bad_seq     first broken link (seq), None when no link breaks
        quarantined_breaks  sorted list of breaks covered by findings
        finding_ids       finding ids covering those breaks
        genesis_hash      documented chain root
        reason            human-readable failure reason when ok is False

    Fails closed on: genesis mismatch, any link break outside the five
        documented sequences, any row whose entry_hash does not match
        sha256(own prev + canonical payload), any non-JSON payload, or any
        evidence mismatch on a quarantined row.  There are no bypass
        parameters.
    """
    row_list = list(rows)
    n = len(row_list)

    link_breaks: list[int] = []
    self_bad: list[int] = []
    evidence_fail: list[int] = []
    quarantined: list[int] = []

    prev_entry = GENESIS_HASH
    for row in row_list:
        seq = int(_get(row, "seq"))
        payload_text = str(_get(row, "payload_summary"))
        entry_hash = str(_get(row, "entry_hash"))
        prev_hash = str(_get(row, "prev_hash"))

        # Per-row self-consistency (tamper check), for EVERY row
        try:
            body = canonical(json.loads(payload_text))
        except Exception:
            self_bad.append(seq)
        else:
            recomputed = hashlib.sha256(
                (prev_hash + body).encode("utf-8")
            ).hexdigest()
            if recomputed != entry_hash:
                self_bad.append(seq)

        # Link check against the walk head (genesis for the first row)
        if prev_hash != prev_entry:
            link_breaks.append(seq)
            finding = FINDING_BY_SEQ.get(seq)
            if finding is not None and finding.matches_row(row):
                quarantined.append(seq)
            else:
                evidence_fail.append(seq)

        prev_entry = entry_hash

    strict_ok = not link_breaks and not self_bad

    if self_bad:
        return {
            "ok": False,
            "strict_ok": False,
            "n_entries": n,
            "first_bad_seq": link_breaks[0] if link_breaks else self_bad[0],
            "quarantined_breaks": [],
            "finding_ids": [],
            "genesis_hash": GENESIS_HASH,
            "reason": f"self-consistency failure at seq {self_bad[0]}",
        }
    if evidence_fail:
        return {
            "ok": False,
            "strict_ok": False,
            "n_entries": n,
            "first_bad_seq": link_breaks[0] if link_breaks else evidence_fail[0],
            "quarantined_breaks": [],
            "finding_ids": [],
            "genesis_hash": GENESIS_HASH,
            "reason": (
                "unquarantined link break at seq "
                f"{[s for s in link_breaks if s in evidence_fail][0]}"
            ),
        }
    if link_breaks:
        # All breaks matched frozen findings -> documented quarantine
        return {
            "ok": True,
            "strict_ok": False,
            "n_entries": n,
            "first_bad_seq": link_breaks[0],
            "quarantined_breaks": sorted(quarantined),
            "finding_ids": sorted(
                FINDING_BY_SEQ[s].finding_id for s in quarantined
            ),
            "genesis_hash": GENESIS_HASH,
            "reason": None,
        }
    return {
        "ok": True,
        "strict_ok": True,
        "n_entries": n,
        "first_bad_seq": None,
        "quarantined_breaks": [],
        "finding_ids": [],
        "genesis_hash": GENESIS_HASH,
        "reason": None,
    }


# ── Read-only forensic re-derivation (for tests / reports) ────────────
def derive_findings_from_db(db_path: str) -> list[dict]:
    """Walk an audit.db READ-ONLY and re-derive every link break with the
    full offender context — used to prove the frozen findings still match
    the on-disk evidence (and only that; it never writes)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute("SELECT * FROM audit_events ORDER BY seq").fetchall()
    finally:
        con.close()

    derived: list[dict] = []
    prev_entry = GENESIS_HASH
    prev_row = None
    for row in rows:
        if str(row["prev_hash"]) != prev_entry:
            derived.append(
                {
                    "seq": row["seq"],
                    "expected_previous_hash": prev_entry,
                    "actual_previous_hash": row["prev_hash"],
                    "pred_event_id": prev_row["event_id"] if prev_row else None,
                    "offender_event_id": row["event_id"],
                    "pred_ts": str(prev_row["created_at"]) if prev_row else None,
                    "offender_ts": str(row["created_at"]),
                    "event_type": row["event_type"],
                    "fraud_id": row["fraud_id"],
                    "entry": row["entry_hash"],
                    "payload_sha": hashlib.sha256(
                        str(row["payload_summary"]).encode("utf-8")
                    ).hexdigest(),
                }
            )
        prev_entry = row["entry_hash"]
        prev_row = row
    return derived
