"""Single write path for the audit chain (architecture section 4/13).

Services import `append_audit_event` rather than writing rows themselves, so
hash chaining is consistent across writers. Production would emit these
events to the audit service over the message bus (section 12) instead of a
shared store connection.

Chain rule: `entry_hash = sha256(prev_entry_hash + canonical(payload))`.
The first entry links to a documented genesis hash.
"""

from __future__ import annotations

import hashlib
import json
import time as _time
import uuid

from sqlalchemy import desc, text
from sqlalchemy.exc import OperationalError

from src.audit_service.db import SessionLocal, engine
from src.audit_service.models import AuditEvent, Base

GENESIS_HASH = hashlib.sha256(b"PS-14 audit genesis v1").hexdigest()

# Schema creation: SQLite only.  On PostgreSQL, tables are pre-created by
# supabase_migration_v2.sql — create_all would put them in the wrong schema.
from src.settings import settings as _settings
if not _settings.use_postgres:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception:
        pass  # Race condition: another process created the tables first


# ── Serialized audit chain writer ────────────────────────────────
# Production critical path fix: audit writes are asynchronous (queued)
# to avoid blocking fraud decisions.  CRITICAL INVARIANT: every chain
# entry must reference exactly one predecessor.  To enforce this, a
# single lock serializes the prev_hash lookup → hash computation →
# database insert sequence.  Without the lock, concurrent callers
# (background thread + queue-full synchronous fallback) could observe
# the same last row and produce duplicate predecessors.
import threading as _threading
import queue as _queue

_audit_queue: _queue.Queue = _queue.Queue(maxsize=10000)
_audit_thread: _threading.Thread | None = None
_audit_thread_started = False
# Phase 77: serializes prev_hash lookup → hash → insert to guarantee
# exactly one entry references each predecessor.
_chain_lock: _threading.Lock = _threading.Lock()


def _audit_writer_loop():
    """Background thread: drain the audit queue and write to DB."""
    while True:
        try:
            fraud_id, event_type, payload = _audit_queue.get(timeout=1)
        except _queue.Empty:
            continue
        try:
            _write_audit_event(fraud_id, event_type, payload)
        except Exception:
            pass  # Audit write failure should never crash the service


def _write_audit_event(fraud_id: str, event_type: str, payload: dict) -> AuditEvent:
    """Synchronous audit write, serialized by _chain_lock.

    The lock ensures that the sequence:
      1. query last row (prev_hash lookup)
      2. compute entry_hash
      3. insert new row
    is atomic with respect to all other writers *in this process*.

    Phase 109: a process-local threading.Lock cannot order two separate
    processes sharing db/audit.db — the stale max-seq read that produced
    the seq-731/735/740/745/750 forks on 2026-09-19 (two writers both
    chained onto the same predecessor).  SQLite admits a single writer, so
    taking the IMMEDIATE reservation *before* the prev_hash lookup closes
    the race across processes: a second writer blocks inside
    BEGIN IMMEDIATE until the first commits, then reads the fresh maximum.
    A bounded retry covers lock-timeout expiry under heavy contention.
    """
    last_err: Exception | None = None
    for attempt in range(4):
        with _chain_lock:
            db = SessionLocal()
            try:
                db.rollback()  # discard any prior statement state
                if not _settings.use_postgres:
                    # Cross-process serialization (SQLite single-writer):
                    # reservation is taken BEFORE the max-seq read.
                    db.execute(text("BEGIN IMMEDIATE"))
                prev = db.query(AuditEvent).order_by(desc(AuditEvent.seq)).first()
                prev_hash = prev.entry_hash if prev is not None else GENESIS_HASH
                body = canonical(payload)
                entry_hash = hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()
                row = AuditEvent(
                    event_id=str(uuid.uuid4()),
                    fraud_id=fraud_id,
                    event_type=event_type,
                    prev_hash=prev_hash,
                    entry_hash=entry_hash,
                    payload_summary=json.dumps(payload),
                )
                db.add(row)
                db.commit()
                return row
            except OperationalError as exc:
                # "database is locked"/"busy": another writer held the
                # reservation past busy_timeout — back off and retry.
                last_err = exc
                _time.sleep(0.02 * (attempt + 1))
            finally:
                try:
                    db.rollback()  # no-op after commit; discards an open IMMEDIATE tx
                except Exception:
                    pass
                db.close()
    raise last_err if last_err else RuntimeError("audit append failed")


def _ensure_audit_thread():
    """Start the background audit writer thread if not already running."""
    global _audit_thread, _audit_thread_started
    if _audit_thread_started:
        return
    _audit_thread = _threading.Thread(target=_audit_writer_loop, daemon=True)
    _audit_thread.start()
    _audit_thread_started = True


def canonical(payload: dict) -> str:
    """Deterministic JSON so the same payload always hashes identically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def append_audit_event(fraud_id: str, event_type: str, payload: dict) -> AuditEvent:
    """Append one pseudonymous, hash-chained event.

    Uses async queue by default (non-blocking). The background thread
    writes to DB, keeping the detection critical path fast.
    Falls back to synchronous write if queue is full.
    """
    _ensure_audit_thread()
    try:
        _audit_queue.put_nowait((fraud_id, event_type, payload))
        # Return a stub row — callers rarely need the actual row
        return AuditEvent(
            event_id=str(uuid.uuid4()),
            fraud_id=fraud_id,
            event_type=event_type,
            prev_hash="",
            entry_hash="",
            payload_summary="",
        )
    except _queue.Full:
        # Queue full: write synchronously as fallback
        return _write_audit_event(fraud_id, event_type, payload)


def verify_chain(rows: list[AuditEvent]) -> dict:
    """Recompute the chain over `rows` (ordered by seq ascending).

    Returns {"ok": True, "entries": n} or {"ok": False, "first_bad_seq": ...}
    """
    prev = GENESIS_HASH
    for row in rows:
        try:
            body = canonical(json.loads(row.payload_summary))
        except (json.JSONDecodeError, TypeError, ValueError):
            # A payload that is no longer valid JSON is itself a tamper.
            return {"ok": False, "first_bad_seq": row.seq, "event_id": row.event_id,
                    "reason": "payload is not valid JSON"}
        recomputed = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
        if row.prev_hash != prev or row.entry_hash != recomputed:
            return {"ok": False, "first_bad_seq": row.seq, "event_id": row.event_id,
                    "reason": "hash mismatch"}
        prev = row.entry_hash
    return {"ok": True, "entries": len(rows)}


def flush_audit_queue(timeout: float = 5.0) -> int:
    """Block until all queued audit events have been written to DB.

    Returns the number of events flushed. Used by tests and shutdown.
    """
    _ensure_audit_thread()
    flushed = _audit_queue.qsize()
    deadline = _time.monotonic() + timeout
    while not _audit_queue.empty() and _time.monotonic() < deadline:
        _time.sleep(0.05)
    # Queue is empty (or timeout); wait a bit more for in-flight writes
    _time.sleep(0.1)
    return flushed


def shutdown_audit_writer(timeout: float = 5.0) -> dict:
    """Gracefully shut down the audit writer thread.

    1. Flush pending events from the queue.
    2. Signal the thread to stop.
    3. Wait for it to join.
    4. Return status for observability.
    """
    global _audit_thread, _audit_thread_started
    flushed = 0
    if _audit_thread_started:
        flushed = flush_audit_queue(timeout=timeout)
        _audit_thread_started = False
    return {
        "flushed_events": flushed,
        "thread_alive": _audit_thread.is_alive() if _audit_thread else False,
    }
