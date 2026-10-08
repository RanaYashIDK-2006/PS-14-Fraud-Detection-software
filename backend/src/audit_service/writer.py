"""Single write path for the audit chain (architecture section 4/13).

Services import `append_audit_event` rather than writing rows themselves, so
hash chaining is consistent across writers. Production would emit these
events to the audit service over the message bus (section 12) instead of a
shared store connection.

Chain rule: `entry_hash = sha256(prev_entry_hash + canonical(payload))`.
The first entry links to a documented genesis hash.

Phase 8 -- audit decision-path resilience:

The pre-Phase-8 queue drained into a background thread that swallowed write
failures (`except Exception: pass`) with no retry and no pending store, so a
transient DB-4 failure silently lost audit events; the queue-full fallback
performed a synchronous DB write (up to busy_timeout) on the CALLER thread
and could raise OperationalError into the decision path.

Design invariants now enforced by this file:

1. Decision path never blocks on DB-4 and never raises. `append_audit_event`
   accepts the event (queue put or durable spool), returns a stub row with a
   stable `event_id`, and propagates no exception into callers.
2. No silent loss. Every accepted event ends in exactly one observable
   disposition: PERSISTED (row in the chained table), RETRYING/PENDING (a
   payload-bearing line in the append-only `audit_pending.jsonl`), or FAILED
   (an explicit durable line + counter). Failure is recorded, never dropped.
3. Replay uses the ORIGINAL payload. Pending records carry the full event
   payload (the same content DB-4 itself stores in `payload_summary`, under
   the same `db/` protections), so a replay after process restart writes the
   original content -- it never reconstructs `{}` or a placeholder. A legacy
   line without a payload is marked FAILED (`replay_payload_missing`) instead
   of being replayed as empty content.
4. Idempotency: `event_id` is assigned once at enqueue; before inserting,
   `_persist_item` checks for an existing row with the same `event_id`
   (payload must match, else append-only mismatch -> FAILED). Replaying the
   same pending record any number of times yields one logical audit record.
5. Bounded everything: queue maxsize=10000; pending file capped at
   MAX_PENDING_LINES (oldest UNREPLIED line is converted to FAILED, oldest
   terminal line recycled beyond that); per-item retry budget
   MAX_RETRIES_PER_ITEM with exponential backoff (RETRY_BACKOFF_BASE * 2^n,
   capped at RETRY_BACKOFF_CAP); per-sweep attempt budget
   MAX_RETRIES_PER_SWEEP; status counters are bounded integers (no unbounded
   id sets).
6. Thread survival: the writer and retry loops wrap each item/sweep in
   try/except; session creation is guarded; `_ensure_*_thread` is
   aliveness-based (a dead thread is replaced, a stopped retry loop restarts
   on the next append). One malformed event or DB-4 outage cannot kill the
   background workers.
7. Shutdown/atexit: `shutdown_audit_writer` flushes the queue and drains
   still-in-memory items into the durable pending store (recoverable by the
   next process start); an atexit hook does the same on normal exit.

Structural properties preserved from the previous design:
- Single process-local serialization via `_chain_lock` (BEGIN IMMEDIATE
  before the max-seq read) so two in-process writers never observe the same
  predecessor. This is NOT an inter-process ordering guarantee -- cross-
  process ordering remains phase109's concern and its test
  (`phase109_audit_fork_repair_test`) exercises that invariant independently.
- `verify_chain`, `canonical`, `GENESIS_HASH` are unchanged.
- The append-only triggers on `audit_events` are unchanged -- they abort
  UPDATE/DELETE with RAISE(ABORT) -> sqlite3.IntegrityError (NOT
  OperationalError), which existing tests and the charm tests exercise.

Ordering note: FIFO holds within each stream (queue drain order, pending-file
append order). On queue overflow the event is spooled directly to the pending
file rather than written synchronously on the caller thread; durability is
prioritized over cross-stream order in that pathological case.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import queue as _queue
import threading
import time as _time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import desc, text
from sqlalchemy.exc import OperationalError

from src.audit_service.db import SessionLocal, engine
from src.audit_service.models import AuditEvent, Base
from src.settings import settings as _settings

GENESIS_HASH = hashlib.sha256(b"PS-14 audit genesis v1").hexdigest()

# Schema creation: SQLite only.  On PostgreSQL, tables are pre-created by
# supabase_migration_v2.sql -- create_all would put them in the wrong schema.
if not _settings.use_postgres:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception:  # noqa: BLE001 - race: another process created the tables first
        pass

# -- Named failure dispositions (public so tests and the closeout can cite them) --
PENDING = "pending"          # accepted by append_audit_event, not yet written
RETRYING = "retrying"        # re-attempts in progress (bounded backoff)
PERSISTED = "persisted"      # written to the chained audit_events table
FAILED = "failed"            # retry exhausted or queue/store bound reached

# Bounded resources (documented in the closeout; tests may lower them locally
# to exercise eviction/exhaustion quickly -- defaults are what runs here).
MAX_PENDING_LINES = 50000    # hard cap on audit_pending.jsonl lines per process
MAX_RETRIES_PER_ITEM = 8     # retry budget per event
MAX_RETRIES_PER_SWEEP = 50   # retry attempts per sweep across all events
RETRY_BACKOFF_BASE = 0.1     # seconds; attempt n waits base * 2**(n-1)
RETRY_BACKOFF_CAP = 30.0     # seconds
_RETRY_IDLE_SLEEP = 0.5      # seconds between sweeps when nothing is pending
_RETRY_MAX_DEFER = 1.0       # cap on sleep-until-next-due per sweep

# -- Serialized audit chain writer --
# CRITICAL INVARIANT: every chain entry must reference exactly one
# predecessor. A single lock serializes prev_hash lookup -> hash computation
# -> insert so concurrent in-process writers cannot observe the same last
# row and produce duplicate predecessors.
_chain_lock = threading.Lock()

_audit_queue: _queue.Queue = _queue.Queue(maxsize=10000)
_audit_thread: threading.Thread | None = None
_audit_thread_started = False
_thread_ctl_lock = threading.Lock()  # serializes thread (re)starts


def _audit_writer_loop() -> None:
    """Background thread: drain the audit queue and write to DB.

    Never exits on its own: each item is guarded so a single malformed event
    or DB-4 outage cannot kill the consumer (phase-8 thread-survival rule).
    """
    while True:
        try:
            item = _audit_queue.get(timeout=1)
        except _queue.Empty:
            continue
        try:
            _drain_one(item)
        except Exception:  # noqa: BLE001 - last-resort guard for the loop
            try:
                _spool_pending(item, reason="drain_exception")
            except BaseException:
                _force_failed(item, reason="drain_exception_spool_failed")


def _drain_one(item: AuditItem) -> None:
    """Persist a single queued item; on failure spool it to the pending store."""
    if not _persist_item(item):
        _spool_pending(item, reason="write_failed")


def _persist_item(item: AuditItem) -> bool:
    """Try to write one item to the chained table.

    Returns True on success (row written OR an identical row already exists),
    False on any failure -- session creation, lock timeout, hash/encode
    errors. Never raises: all exceptions are contained so the writer/retry
    threads survive.
    """
    with _chain_lock:
        try:
            db = SessionLocal()
        except Exception:  # noqa: BLE001 - guarded session creation (DB-4 gone)
            return False
        try:
            try:
                db.rollback()
                if not _settings.use_postgres:
                    db.execute(text("BEGIN IMMEDIATE"))
                prev = db.query(AuditEvent).order_by(desc(AuditEvent.seq)).first()
                prev_hash = prev.entry_hash if prev is not None else GENESIS_HASH

                # Idempotency: an existing row with this event_id is the same
                # logical event. Payload must match; a mismatch is an anomaly
                # we never overwrite (append-only) -- surface it as failure.
                existing = (
                    db.query(AuditEvent)
                    .filter(AuditEvent.event_id == item.event_id)
                    .first()
                )
                if existing is not None:
                    if (
                        existing.fraud_id == item.fraud_id
                        and existing.event_type == item.event_type
                        and existing.payload_summary == canonical(item.payload)
                    ):
                        _mark_persisted(item)
                        return True
                    return False

                body = canonical(item.payload)
                entry_hash = hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()
                row = AuditEvent(
                    event_id=item.event_id,
                    fraud_id=item.fraud_id,
                    event_type=item.event_type,
                    prev_hash=prev_hash,
                    entry_hash=entry_hash,
                    payload_summary=body,
                )
                db.add(row)
                db.commit()
                _mark_persisted(item)
                return True
            except Exception:  # noqa: BLE001 - OperationalError and everything else
                return False
        finally:
            try:
                db.rollback()
            except Exception:  # noqa: BLE001
                pass
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass


# ── Pending durable store (append-only per process; lines are rewritten only
# for status transitions of their own event) ────────────────────────────────
# A JSONL file under the audit db directory: same locality/protection class
# as the SQLite stores rather than inventing another DB. Each record carries
# the FULL event payload so replay reproduces the original audit content.
_PENDING_DIR: Path | None = None
_PENDING_PATH: Path | None = None
_PENDING_LINE_COUNT = 0
# Reentrant: internal helpers (_evict, _mark_failed) nest under the same lock.
_pending_lock = threading.RLock()


def _pending_path() -> Path:
    global _PENDING_DIR, _PENDING_PATH
    if _PENDING_PATH is None:
        db_dir = Path(_settings.audit_db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        _PENDING_DIR = db_dir
        _PENDING_PATH = db_dir / "audit_pending.jsonl"
    return _PENDING_PATH


def _dump(rec: dict[str, Any]) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)


def _pending_record(item: AuditItem, *, status: str, reason: str) -> dict[str, Any]:
    """Full durable representation of an accepted-but-unresolved event."""
    return {
        "event_id": item.event_id,
        "fraud_id": item.fraud_id,
        "event_type": item.event_type,
        "payload": item.payload,          # replay source of truth
        "status": status,
        "reason": reason,
        "retry_count": item.retry_count,
        "first_seen": item.first_seen,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "next_retry_at": 0.0,             # epoch seconds; 0 = due now
    }


def _spool_pending(item: AuditItem, *, reason: str) -> None:
    """Append a pending record so the event becomes durable + replayable.

    Bounded: at MAX_PENDING_LINES the oldest UNREPLIED line is converted to
    FAILED (evicted), keeping the file hard-capped. Spooling is idempotent
    per item (`item.spooled`), so a queued item that was flushed and later
    drained cannot produce duplicate lines.
    """
    global _PENDING_LINE_COUNT
    rec = _pending_record(item, status=PENDING, reason=reason)
    with _pending_lock:
        if item.spooled:
            return  # already durable; line will be replayed from the file
        if _PENDING_LINE_COUNT >= MAX_PENDING_LINES:
            _evict_oldest_unreplied()
        try:
            with open(_pending_path(), "a", encoding="utf-8") as fh:
                fh.write(_dump(rec) + "\n")
            _PENDING_LINE_COUNT += 1
            item.spooled = True
            with _status_lock:
                _pending_events.pop(item.event_id, None)  # durable now
        except (OSError, TypeError, ValueError):
            # File itself unwritable: record explicit FAILED (never silent).
            _mark_failed(item, reason=f"spool_write_failed:{reason}")


def _evict_oldest_unreplied() -> None:
    """Hard-cap the pending file so one spool can always make room.

    Called at capacity, BEFORE the new line is appended, and always nets
    -1 line so `count + 1 <= MAX_PENDING_LINES` afterwards:

    1. Oldest UNREPLIED line is converted IN PLACE to a FAILED record
       (reason `pending_store_evicted`, `permanently_failed` counter +1) --
       no new line is created by the conversion.
    2. One further line is then dropped: preferably an already-terminal one
       (`evicted_lines` counter -- permanent-failure evidence is itself
       bounded, oldest recycled); otherwise another unreplied event's line
       (`permanently_failed` +1 without in-file marker).
    3. If every line is terminal/corrupt, the oldest line is recycled
       directly.

    Counters in `audit_writer_status` remain the authoritative totals.
    """
    global _PENDING_LINE_COUNT
    path = _pending_path()
    if not path.exists():
        return
    try:
        with _pending_lock:
            entries: list[tuple[str, dict[str, Any] | None]] = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        entries.append((line, json.loads(line)))
                    except json.JSONDecodeError:
                        entries.append((line, None))
            if not entries:
                return

            def _is_unreplied(rec: dict[str, Any] | None) -> bool:
                return rec is not None and rec.get("status") in (PENDING, RETRYING)

            v1 = next(
                (i for i, (_ln, r) in enumerate(entries) if _is_unreplied(r)),
                0,
            )
            if _is_unreplied(entries[v1][1]):
                # Keep this event's failure as durable evidence, in place.
                marker = dict(entries[v1][1])
                marker["status"] = FAILED
                marker["reason"] = f"pending_store_evicted:max_lines={MAX_PENDING_LINES}"
                marker["updated_at"] = _now_iso()
                entries[v1] = (_dump(marker), marker)
                _bump_counter("failed")
                # Now drop one different line to net -1: prefer terminal.
                j = next(
                    (i for i, (_ln, r) in enumerate(entries)
                     if i != v1 and not _is_unreplied(r)),
                    None,
                )
                if j is None:
                    j = next((i for i in range(len(entries)) if i != v1), None)
                if j is not None:
                    if _is_unreplied(entries[j][1]):
                        _bump_counter("failed")
                    else:
                        _bump_counter("evicted")
                    entries.pop(j)
            else:
                # All terminal/corrupt: recycle the oldest line.
                entries.pop(v1)
                _bump_counter("evicted")

            keep = [ln for ln, _r in entries]
            with open(path, "w", encoding="utf-8") as fh:
                for ln in keep:
                    fh.write(ln + "\n")
            _PENDING_LINE_COUNT = len(keep)
    except OSError:
        # Cannot cap the file right now; per-item retry bounds still hold and
        # the next spool retries the eviction (cap may overshoot by the
        # number of failed evictions until one succeeds).
        pass


def _remove_pending_record(event_id: str) -> None:
    """Drop an event's line after successful persistence (natural compaction)."""
    global _PENDING_LINE_COUNT
    with _pending_lock:
        path = _pending_path()
        if not path.exists():
            return
        try:
            keep: list[str] = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        keep.append(line)
                        continue
                    if rec.get("event_id") != event_id:
                        keep.append(line)
            with open(path, "w", encoding="utf-8") as fh:
                for ln in keep:
                    fh.write(ln + "\n")
            _PENDING_LINE_COUNT = len(keep)
        except OSError:
            # Line stays; the next sweep will re-read it, see the row exists
            # (idempotent match), and retry the removal. No duplicate row.
            pass


def _update_pending_record(
    item: AuditItem, *, status: str, reason: str, next_retry_at: float | None = None
) -> None:
    """Rewrite this event's line with new retry state (append if absent).

    The rewrite touches ONLY lines for `item.event_id`, under `_pending_lock`,
    so concurrent spools/updates cannot interleave or lose lines.
    """
    global _PENDING_LINE_COUNT
    with _pending_lock:
        path = _pending_path()
        try:
            replaced = False
            new_lines: list[str] = []
            if path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            new_lines.append(line)
                            continue
                        if rec.get("event_id") == item.event_id:
                            rec["status"] = status
                            rec["reason"] = reason
                            rec["retry_count"] = item.retry_count
                            rec["updated_at"] = _now_iso()
                            # payload untouched: the record already carries the
                            # original content (and a payload-less legacy record
                            # must stay visibly payload-less, not gain "{}").
                            if next_retry_at is not None:
                                rec["next_retry_at"] = next_retry_at
                            new_lines.append(_dump(rec))
                            replaced = True
                        else:
                            new_lines.append(line)
            if not replaced:
                rec = _pending_record(item, status=status, reason=reason)
                if next_retry_at is not None:
                    rec["next_retry_at"] = next_retry_at
                new_lines.append(_dump(rec))
                item.spooled = True
            with open(path, "w", encoding="utf-8") as fh:
                for ln in new_lines:
                    fh.write(ln + "\n")
            _PENDING_LINE_COUNT = len(new_lines)
        except (OSError, TypeError, ValueError):
            # Best-effort durable bookkeeping; status counters still recorded
            # by the caller.
            pass


def _read_pending_lines() -> list[dict[str, Any]]:
    """Current pending records from disk (retry loop, diagnostics)."""
    path = _pending_path()
    with _pending_lock:
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # corrupt line: skipped by readers, evicted at cap
        except OSError:
            pass
        return out


def _pending_records_pending_or_retrying() -> list[dict[str, Any]]:
    """FIFO: file order preserved."""
    return [
        r for r in _read_pending_lines()
        if r.get("status") in (PENDING, RETRYING)
    ]


def _pending_payload(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Replay payload from a pending record -- or None if genuinely absent.

    Returning None (instead of `{}`) is deliberate: a record without its
    payload must be marked FAILED, never replayed as empty content.
    """
    payload = rec.get("payload")
    return payload if isinstance(payload, dict) else None


# -- Status accounting: bounded integer counters (no unbounded id sets) --
_persisted_count = 0
_failed_count = 0
_evicted_count = 0
_pending_events: dict[str, AuditItem | None] = {}  # accepted, not yet durable/terminal
_status_lock = threading.Lock()


def _bump_counter(name: str) -> None:
    """Bump a status counter under `_status_lock`.

    Safe to call while holding `_pending_lock` (lock order is always
    pending -> status; `_status_lock` holders never take `_pending_lock`).
    """
    global _failed_count, _evicted_count
    with _status_lock:
        if name == "failed":
            _failed_count += 1
        elif name == "evicted":
            _evicted_count += 1


def _note_event_status(event_id: str, status: str) -> None:
    with _status_lock:
        if status == PERSISTED:
            global _persisted_count
            _persisted_count += 1
            _pending_events.pop(event_id, None)
        elif status == FAILED:
            global _failed_count
            _failed_count += 1
            _pending_events.pop(event_id, None)
        # PENDING/RETRYING: already durable in the pending file; nothing to
        # track in memory (prevents the dict growing on every retry tick).


def audit_writer_status() -> dict[str, Any]:
    """Lightweight operational state (no PII, bounded).

    `persisted` / `permanently_failed` count observed transitions (a rare
    restart-replay of an already-written row can re-count one event; the DB
    row count is authoritative for accounting).
    """
    with _status_lock:
        pending = len(_pending_events)
        persisted = _persisted_count
        failed = _failed_count
        evicted = _evicted_count
    return {
        "in_memory_pending": pending,
        "persisted": persisted,
        "permanently_failed": failed,
        "evicted_lines": evicted,
        "queue_size": _audit_queue.qsize(),
        "pending_file_lines": _PENDING_LINE_COUNT,
        "max_pending_lines": MAX_PENDING_LINES,
        "max_retries_per_item": MAX_RETRIES_PER_ITEM,
        "max_retries_per_sweep": MAX_RETRIES_PER_SWEEP,
        "retry_backoff_cap": RETRY_BACKOFF_CAP,
    }


def audit_writer_logs() -> list[dict[str, Any]]:
    """Latest pending-file records as plain dicts (payload redacted -- this
    is a diagnostic view; the file itself is the durable store)."""
    out: list[dict[str, Any]] = []
    for rec in _read_pending_lines()[-200:]:
        out.append({
            "event_id": rec.get("event_id"),
            "fraud_id": rec.get("fraud_id"),
            "event_type": rec.get("event_type"),
            "status": rec.get("status"),
            "reason": rec.get("reason"),
            "retry_count": rec.get("retry_count", 0),
            "first_seen": rec.get("first_seen"),
        })
    return out


# -- Status transitions on items --
def _mark_persisted(item: AuditItem) -> None:
    item.status = PERSISTED
    _note_event_status(item.event_id, PERSISTED)


def _mark_failed(item: AuditItem, *, reason: str) -> None:
    """Terminal failure: counter + durable FAILED line (replacing this
    event's pending line if it exists, so no duplicate lines accumulate)."""
    item.status = FAILED
    _note_event_status(item.event_id, FAILED)
    _update_pending_record(item, status=FAILED, reason=reason)


def _force_failed(item: AuditItem, *, reason: str) -> None:
    """Last-resort failure recording: increments the counter even when the
    pending file itself is unwritable, so failure is never invisible."""
    try:
        _mark_failed(item, reason=reason)
    except BaseException:  # noqa: BLE001
        with _status_lock:
            global _failed_count
            _failed_count += 1
            _pending_events.pop(item.event_id, None)


# -- Retry loop --
_retry_thread: threading.Thread | None = None
_retry_started = False
_retry_stop = threading.Event()


def _ensure_retry_thread() -> None:
    """(Re)start the retry worker if it is not running.

    Aliveness-based: after `shutdown_audit_writer` (which stops it) the next
    append restarts it; a crashed thread is replaced instead of leaving
    `_retry_started=True` pointing at a dead worker.
    """
    global _retry_thread, _retry_started
    with _thread_ctl_lock:
        if _retry_thread is not None and _retry_thread.is_alive():
            _retry_started = True
            return
        _retry_stop.clear()
        _retry_thread = threading.Thread(target=_retry_loop, name="audit-retry", daemon=True)
        _retry_thread.start()
        _retry_started = True


def _retry_sweep() -> float:
    """One bounded pass over the pending store; returns seconds to sleep.

    Budget: at most MAX_RETRIES_PER_SWEEP persist attempts, skipping records
    that are not yet due (`next_retry_at`), so backoff never blocks the sweep
    and a large pending backlog cannot starve the loop (per-sweep budget).
    """
    now = _time.time()
    attempted = 0
    min_due: float | None = None
    for rec in _pending_records_pending_or_retrying():
        if _retry_stop.is_set():
            break
        if attempted >= MAX_RETRIES_PER_SWEEP:
            break
        event_id = rec.get("event_id")
        if not event_id:
            continue
        due = float(rec.get("next_retry_at") or 0.0)
        if due > now:
            min_due = due if min_due is None else min(min_due, due)
            continue
        payload = _pending_payload(rec)
        if payload is None:
            # A record without its payload cannot be replayed faithfully.
            # Explicit FAILED beats writing `{}` (append-only, auditable).
            stub = AuditItem(
                event_id=event_id,
                fraud_id=rec.get("fraud_id") or "",
                event_type=rec.get("event_type") or "unknown",
                payload={},
                retry_count=int(rec.get("retry_count") or 0),
                first_seen=rec.get("first_seen"),
            )
            _mark_failed(stub, reason="replay_payload_missing")
            continue
        try:
            item = AuditItem(
                event_id=event_id,
                fraud_id=rec.get("fraud_id") or "",
                event_type=rec.get("event_type") or "unknown",
                payload=payload,
                retry_count=int(rec.get("retry_count") or 0),
                first_seen=rec.get("first_seen"),
                status=rec.get("status", PENDING),
            )
        except Exception:  # noqa: BLE001 - malformed record: keep sweeping
            continue

        if item.retry_count >= MAX_RETRIES_PER_ITEM:
            _mark_failed(item, reason=f"retry_exhausted:{item.retry_count}_attempts")
            continue

        attempted += 1
        item.retry_count += 1
        backoff = min(RETRY_BACKOFF_CAP, RETRY_BACKOFF_BASE * (2 ** (item.retry_count - 1)))
        if _persist_item(item):
            _remove_pending_record(item.event_id)
        else:
            item.status = RETRYING
            _update_pending_record(
                item, status=RETRYING, reason="retry_in_progress",
                next_retry_at=now + backoff,
            )

    if min_due is not None:
        return max(0.05, min(_RETRY_MAX_DEFER, min_due - _time.time()))
    if attempted:
        return 0.2
    return _RETRY_IDLE_SLEEP


def _retry_loop() -> None:
    """Replay pending events FIFO with per-item backoff and per-sweep budget.

    Survives anything: an exception inside a sweep costs one idle pause, not
    the worker.
    """
    while not _retry_stop.is_set():
        try:
            delay = _retry_sweep()
        except Exception:  # noqa: BLE001 - never let the worker die
            delay = _RETRY_IDLE_SLEEP
        _retry_stop.wait(delay)


# -- AuditItem: the unit of work moving through queue -> pending -> persisted/failed --
class AuditItem:
    """A single audit event accepted for persistence.

    `event_id` is assigned once at enqueue so retries are idempotent.
    `payload` travels with the item and IS written to the durable pending
    store, so replay after restart reproduces the original content.
    `spooled` marks that the durable line already exists (no double-spool).
    """

    __slots__ = ("event_id", "fraud_id", "event_type", "payload", "retry_count",
                 "first_seen", "status", "spooled")

    def __init__(
        self,
        *,
        event_id: str,
        fraud_id: str,
        event_type: str,
        payload: dict,
        retry_count: int = 0,
        first_seen: str | None = None,
        status: str = PENDING,
        spooled: bool = False,
    ) -> None:
        self.event_id = event_id
        self.fraud_id = fraud_id
        self.event_type = event_type
        self.payload = payload
        self.retry_count = retry_count
        self.first_seen = first_seen or _now_iso()
        self.status = status
        self.spooled = spooled

    def __repr__(self) -> str:
        return (f"AuditItem(event_id={self.event_id!r}, fraud_id={self.fraud_id!r}, "
                f"event_type={self.event_type!r}, status={self.status!r})")


# Helpers
def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ── Public API ──────────────────────────────────────────────────────────────────
def append_audit_event(fraud_id: str, event_type: str, payload: dict) -> AuditEvent:
    """Append one pseudonymous, hash-chained event.

    Decision-path contract: ACCEPTS every event and returns a non-None
    AuditEvent; never raises and never performs a DB-4 write on the caller
    thread. The event is queued (common case) or spooled to the durable
    pending store (queue full / thread-start failure); either way it is
    recoverable and observable via audit_writer_status / audit_writer_logs.

    The returned stub carries the SAME event_id that will eventually be
    written to the chain; seq/entry_hash are empty until persistence.
    """
    item = AuditItem(
        event_id=str(uuid.uuid4()),
        fraud_id=fraud_id,
        event_type=event_type,
        payload=payload,
    )
    try:
        _ensure_audit_thread()
        _ensure_retry_thread()
        with _status_lock:
            _pending_events[item.event_id] = item
        try:
            _audit_queue.put_nowait(item)
        except _queue.Full:
            # Overflow: durable spool, not a synchronous caller-thread write.
            _spool_pending(item, reason="queue_full")
    except BaseException:  # noqa: BLE001 - never leak into decision callers
        # Fall back straight to the durable store; if even that fails, the
        # explicit-failed counter keeps the loss observable (never silent).
        try:
            _spool_pending(item, reason="append_path_failed")
        except BaseException:  # noqa: BLE001
            _force_failed(item, reason="append_and_spool_failed")
    return _stub_for(item, persisted=False)


def _stub_for(item: AuditItem, *, persisted: bool) -> AuditEvent:
    """Build the AuditEvent returned to callers (real ORM instance so
    `.event_id` / `.fraud_id` / `.event_type` access keeps working)."""
    return AuditEvent(
        event_id=item.event_id,
        fraud_id=item.fraud_id,
        event_type=item.event_type,
        prev_hash="",
        entry_hash="",
        payload_summary="",
    )


def _write_audit_event(fraud_id: str, event_type: str, payload: dict) -> AuditEvent:
    """Synchronous write, serialized by _chain_lock. Raises on failure.

    Kept for compatibility with in-process callers and tests that call this
    directly. It bypasses the queue and pending store, so transient failures
    surface as exceptions -- NOT the decision-path contract (that is
    `append_audit_event`). Session creation is inside the retry guard so a
    DB-4 outage raises a clean error instead of a stray exception.
    """
    last_err: BaseException | None = None
    for attempt in range(4):
        with _chain_lock:
            db = None
            try:
                db = SessionLocal()
                db.rollback()
                if not _settings.use_postgres:
                    db.execute(text("BEGIN IMMEDIATE"))
                prev = db.query(AuditEvent).order_by(desc(AuditEvent.seq)).first()
                prev_hash = prev.entry_hash if prev is not None else GENESIS_HASH
                body = canonical(payload)
                entry_hash = hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()
                eid = str(uuid.uuid4())
                existing = db.query(AuditEvent).filter(AuditEvent.event_id == eid).first()
                if existing is None:
                    row = AuditEvent(
                        event_id=eid,
                        fraud_id=fraud_id,
                        event_type=event_type,
                        prev_hash=prev_hash,
                        entry_hash=entry_hash,
                        payload_summary=body,
                    )
                    db.add(row)
                    db.commit()
                    return row
                continue  # astronomically unlikely uuid4 collision
            except OperationalError as exc:
                last_err = exc
                _time.sleep(0.02 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001 - session creation etc.
                last_err = exc
                _time.sleep(0.02 * (attempt + 1))
            finally:
                if db is not None:
                    try:
                        db.rollback()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        db.close()
                    except Exception:  # noqa: BLE001
                        pass
    raise last_err if last_err else RuntimeError("audit append failed")


def canonical(payload: dict) -> str:
    """Deterministic JSON so the same payload always hashes identically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def verify_chain(rows: list[AuditEvent]) -> dict:
    """Recompute the chain over `rows` (ordered by seq ascending).

    Returns {"ok": True, "entries": n} or {"ok": False, "first_bad_seq": ...}
    """
    prev = GENESIS_HASH
    for row in rows:
        try:
            body = canonical(json.loads(row.payload_summary))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"ok": False, "first_bad_seq": row.seq, "event_id": row.event_id,
                    "reason": "payload is not valid JSON"}
        recomputed = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
        if row.prev_hash != prev or row.entry_hash != recomputed:
            return {"ok": False, "first_bad_seq": row.seq, "event_id": row.event_id,
                    "reason": "hash mismatch"}
        prev = row.entry_hash
    return {"ok": True, "entries": len(rows)}


def flush_audit_queue(timeout: float = 5.0) -> int:
    """Block until queued audit events have been drained (written or spooled).

    Returns the number of events that were queued at call time. Used by
    tests and shutdown; does not raise.
    """
    try:
        _ensure_audit_thread()
    except Exception:  # noqa: BLE001
        pass
    flushed = _audit_queue.qsize()
    deadline = _time.monotonic() + timeout
    while not _audit_queue.empty() and _time.monotonic() < deadline:
        _time.sleep(0.05)
    _time.sleep(0.1)
    return flushed


def shutdown_audit_writer(timeout: float = 5.0) -> dict:
    """Gracefully wind down the audit writers.

    1. Flush the queue (each item is written or spooled -- never dropped).
    2. Drain any still-in-memory accepted items into the durable pending
       store so queued-but-unwritten events survive process exit.
    3. Stop the retry loop (it restarts on the next append).

    Returns observability state; never raises.
    """
    flushed = 0
    if _audit_thread_started:
        flushed = flush_audit_queue(timeout=timeout)
    _flush_pending_to_durable()
    _retry_stop.set()
    return {
        "flushed_events": flushed,
        "thread_alive": _audit_thread.is_alive() if _audit_thread else False,
        "status": audit_writer_status(),
    }


def _flush_pending_to_durable() -> int:
    """Safety net: spool every accepted-but-not-yet-durable in-memory item.

    Called at shutdown and atexit so queued-but-unwritten events become
    recoverable pending records instead of vanishing with the process.
    """
    count = 0
    with _status_lock:
        items = [
            i for i in _pending_events.values()
            if i is not None and i.status not in (PERSISTED, FAILED)
        ]
        _pending_events.clear()
    for item in items:
        if item.spooled:
            count += 1
            continue  # already durable in the pending file
        _spool_pending(item, reason="shutdown_flush")
        count += 1
    return count


# -- Start/stop helpers for the background writer thread --
def _ensure_audit_thread() -> None:
    """(Re)start the queue-drain thread if it is not running (aliveness-based)."""
    global _audit_thread, _audit_thread_started
    with _thread_ctl_lock:
        if _audit_thread is not None and _audit_thread.is_alive():
            _audit_thread_started = True
            return
        _audit_thread = threading.Thread(target=_audit_writer_loop, name="audit-writer", daemon=True)
        _audit_thread.start()
        _audit_thread_started = True


# Safety net: a process that exits without calling shutdown_audit_writer
# (short-lived scripts, unnamed service paths) still drains its queue and
# in-memory items to the durable store on normal exit.
def _atexit_flush() -> None:
    try:
        flush_audit_queue(timeout=1.0)
    except Exception:  # noqa: BLE001 - atexit must never prevent exit
        pass
    try:
        _flush_pending_to_durable()
    except Exception:  # noqa: BLE001
        pass


atexit.register(_atexit_flush)
