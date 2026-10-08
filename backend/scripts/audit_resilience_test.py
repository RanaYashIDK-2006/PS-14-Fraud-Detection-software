#!/usr/bin/env python3
"""Phase 8 — Audit Decision-Path Resilience test suite.

Verifies, against the CURRENT writer.py, that a failing DB-4 (audit store)
can no longer silently lose events or raise into the fraud decision path:

  * failure modes A-D: session-creation failure, pending/replay failure,
    retry-worker failure, chain/content integrity;
  * normal / unavailable / locked / delayed DB-4, repeated failures,
    recovery, duplicate replay, FIFO, shutdown, concurrency, endpoints,
    batch pairing;
  * decision-path invariant: /internal/evaluate is byte-equivalent with
    healthy vs failing DB-4 (same deterministic request);
  * bounds: retry per-item / per-sweep, queue, pending file, backoff,
    overflow behavior, explicit permanent-failure state;
  * controlled local performance: audit delay never becomes decision or
    append latency (not a throughput benchmark — that is Phase 9).

Run from the project root:
  python backend/scripts/audit_resilience_test.py
"""

from __future__ import annotations

import json
import os
import queue as _queue
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

TMP = tempfile.mkdtemp(prefix="ps14-p8-")
os.environ["DB_DIR"] = TMP
os.environ["PS14_MODE"] = "development"
os.environ["JWT_SECRET"] = "smoke-test-secret-0123456789abcdef"
os.environ["INTERNAL_TOKEN"] = "smoke-internal-token"
os.environ["COMPLIANCE_TOKEN"] = "smoke-compliance-token"

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

from src.audit_service import writer as w  # noqa: E402
from src.audit_service.db import SessionLocal  # noqa: E402  (real sessions for assertions)
from src.audit_service.models import AuditEvent  # noqa: E402
from src.risk_engine.main import app as risk_app  # noqa: E402

FRAUD_ID = "F24BRMMMBJWYMTDW"
TOKEN = "smoke-internal-token"
AUDIT_DB = Path(TMP) / "audit.db"
PENDING_FILE = Path(TMP) / "audit_pending.jsonl"
REAL_SESSION = w.SessionLocal
_SESSION_MODE = {"mode": "real"}

# Zero-overhead failure tracing: every writer operation is recorded in memory
# (no I/O on the hot path) and dumped only when a check fails.
TRACE: list[str] = []
_traced = {
    "spool": w._spool_pending,
    "update": w._update_pending_record,
    "mark": w._mark_failed,
    "remove": w._remove_pending_record,
    "persist": w._persist_item,
}


def _tr_spool(item, *, reason):
    TRACE.append(f"SPOOL {item.event_id[:8]} reason={reason} spooled={item.spooled}")
    return _traced["spool"](item, reason=reason)


def _tr_update(item, *, status, reason, next_retry_at=None):
    TRACE.append(f"UPDATE {item.event_id[:8]} -> {status}/{reason} n={item.retry_count}")
    return _traced["update"](item, status=status, reason=reason, next_retry_at=next_retry_at)


def _tr_mark(item, *, reason):
    TRACE.append(f"MARK {item.event_id[:8]} reason={reason} n={item.retry_count}")
    return _traced["mark"](item, reason=reason)


def _tr_remove(event_id: str):
    TRACE.append(f"REMOVE {event_id[:8]}")
    return _traced["remove"](event_id)


def _tr_persist(item):
    ok = _traced["persist"](item)
    TRACE.append(f"PERSIST {item.event_id[:8]} -> {ok}")
    return ok


w._spool_pending = _tr_spool
w._update_pending_record = _tr_update
w._mark_failed = _tr_mark
w._remove_pending_record = _tr_remove
w._persist_item = _tr_persist

failures: list[str] = []
# Registry: stub event_id (DB column) -> exact payload that must appear in DB-4.
EXPECT_ROW: dict[str, dict] = {}


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)
        # Self-diagnosing dump: raw pending file state at failure time.
        try:
            raw = PENDING_FILE.read_text(encoding="utf-8") if PENDING_FILE.exists() else "<absent>"
            for ln in raw.strip().split("\n")[:40]:
                try:
                    r = json.loads(ln)
                    print(f"      raw: {str(r.get('event_id',''))[:8]} {r.get('status')} "
                          f"{r.get('reason')} n={r.get('retry_count')} "
                          f"m={r.get('payload', {}).get('marker') if isinstance(r.get('payload'), dict) else None}")
                except json.JSONDecodeError:
                    print(f"      raw: <unparseable> {ln[:60]}")
            print(f"      status: {w.audit_writer_status()}")
        except Exception as exc:  # noqa: BLE001
            print(f"      (dump failed: {exc!r})")


def section(title: str) -> None:
    print(f"\n-- {title} --")


def wait_until(pred, timeout: float = 8.0, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if pred():
                return True
        except Exception:  # noqa: BLE001 - transient DB states during failure injection
            pass
        time.sleep(interval)
    return False


def fail_sessions() -> None:
    """Failure mode A: session creation itself raises (DB-4 gone)."""
    def _boom(*_a, **_k):
        raise OperationalError("unable to open database file", {}, Exception("db-gone"))
    w.SessionLocal = _boom
    _SESSION_MODE["mode"] = "BOOM"


def restore_sessions() -> None:
    w.SessionLocal = REAL_SESSION
    _SESSION_MODE["mode"] = "real"


def all_rows() -> list[AuditEvent]:
    s = SessionLocal()
    try:
        return s.query(AuditEvent).order_by(AuditEvent.seq).all()
    finally:
        s.close()


def row_by_event_id(event_id: str) -> AuditEvent | None:
    s = SessionLocal()
    try:
        return (
            s.query(AuditEvent).filter(AuditEvent.event_id == event_id).one_or_none()
        )
    finally:
        s.close()


def pending_records() -> list[dict]:
    return w._read_pending_lines()


def pending_with_marker(marker: str) -> list[dict]:
    return [
        r for r in pending_records()
        if isinstance(r.get("payload"), dict) and r["payload"].get("marker") == marker
    ]


def first_pending(marker: str) -> dict:
    recs = pending_with_marker(marker)
    return recs[0] if recs else {}


def chain_ok() -> tuple[bool, str]:
    rows = all_rows()
    res = w.verify_chain(rows)
    return bool(res.get("ok")), json.dumps(res)


def rows_for_payload_event(payload_event_id: str) -> list[AuditEvent]:
    return [
        r for r in all_rows()
        if isinstance(r.payload_summary, str)
        and f'"event_id":"{payload_event_id}"' in r.payload_summary
    ]


def append_tracked(payload: dict, event_type: str = "score_generated"):
    ev = w.append_audit_event(FRAUD_ID, event_type, payload)
    EXPECT_ROW[ev.event_id] = payload
    return ev


def vector(**overrides) -> dict:
    base = {
        "amount_ratio": 0.95,
        "txn_freq_last_24h": 1,
        "txn_time_unusual": 0,
        "new_device_flag": 0,
        "unusual_location_flag": 0,
        "unusual_recipient_flag": 0,
        "failed_auth_count_24h": 0,
        "days_since_last_similar_txn": 3.0,
        "gradual_escalation_score": 0.0,
        "known_device_count": 3,
        "account_tenure_days": 90.0,
        "hour_of_day": 12,
        "is_weekend": 0,
        "shared_device_accounts": 0,
        "shared_recipient_accounts": 0,
        "mule_ring_score": 0,
        "hour_deviation": 0.5,
        "amount_zscore": 0,
        "velocity_deviation": 0,
        "recipient_novelty": 0,
        "txn_regularity": 0,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    t_suite = time.perf_counter()
    perf: dict[str, float] = {}

    # ══ 1. Documented bounds (§10: determine the actual limits) ══════════════
    section("documented limits (§10)")
    check("queue maxsize is 10000", w._audit_queue.maxsize == 10000, str(w._audit_queue.maxsize))
    check("pending file cap default 50000", w.MAX_PENDING_LINES == 50000, str(w.MAX_PENDING_LINES))
    check("per-item retry budget default 8", w.MAX_RETRIES_PER_ITEM == 8, str(w.MAX_RETRIES_PER_ITEM))
    check("per-sweep retry budget exists", w.MAX_RETRIES_PER_SWEEP == 50, str(w.MAX_RETRIES_PER_SWEEP))
    check("backoff base 0.1s cap 30s",
          w.RETRY_BACKOFF_BASE == 0.1 and w.RETRY_BACKOFF_CAP == 30.0,
          f"base={w.RETRY_BACKOFF_BASE} cap={w.RETRY_BACKOFF_CAP}")
    st = w.audit_writer_status()
    check("status counters are ints (bounded)", all(
        isinstance(st[k], int) for k in ("in_memory_pending", "persisted",
                                         "permanently_failed", "evicted_lines")))

    # ══ 2. Normal path ═══════════════════════════════════════════════════════
    section("normal path")
    n0 = len(all_rows())
    ev1 = append_tracked({"marker": "norm-1", "event_id": "ev-p8-norm-1", "risk_score": 7})
    ev2 = append_tracked({"marker": "norm-2", "event_id": "ev-p8-norm-2", "risk_score": 12})
    check("append returns a stub with stable event_id",
          isinstance(ev1.event_id, str) and len(ev1.event_id) == 36)
    w.flush_audit_queue(5.0)
    ok_rows = wait_until(lambda: len(all_rows()) == n0 + 2, timeout=6.0)
    check("audit events persist on healthy path", ok_rows,
          f"rows={len(all_rows())} expected={n0 + 2}")
    row1 = row_by_event_id(ev1.event_id)
    check("stored payload == submitted payload",
          row1 is not None and json.loads(row1.payload_summary) ==
          {"marker": "norm-1", "event_id": "ev-p8-norm-1", "risk_score": 7})
    check("pending store empty after success", len(pending_records()) == 0,
          f"lines={len(pending_records())}")
    okc, detail = chain_ok()
    check("chain valid after normal path", okc, detail)

    # ══ 3. Failure mode A+B: session creation failure, payload-bearing replay ═
    section("failure mode A/B: session-creation failure + payload replay")
    fail_sessions()
    lat0 = time.perf_counter()
    evA = append_tracked({"marker": "outage-1", "event_id": "ev-p8-outage-1",
                          "risk_score": 91, "special": "payload-must-survive"})
    append_lat = (time.perf_counter() - lat0) * 1000
    check("append does not raise while DB-4 is gone", evA.event_id is not None)
    check("append latency under DB-4 failure is non-blocking",
          append_lat < 100, f"{append_lat:.1f}ms")

    check("writer thread survives session-creation failure",
          w._audit_thread is not None and w._audit_thread.is_alive())
    check("retry thread started and alive",
          w._retry_thread is not None and w._retry_thread.is_alive())

    spooled = wait_until(lambda: len(pending_with_marker("outage-1")) == 1, timeout=6.0)
    check("event becomes recoverable (spooled to pending store)", spooled)
    rec = (pending_with_marker("outage-1") or [{}])[0]
    check("pending record carries the FULL payload",
          rec.get("payload", {}).get("special") == "payload-must-survive",
          str(rec.get("payload"))[:80])
    check("pending record has identity/status/retry fields", all(
        k in rec for k in ("event_id", "fraud_id", "event_type", "status",
                           "retry_count", "first_seen", "next_retry_at")))
    transitioned = wait_until(
        lambda: first_pending("outage-1").get("status") == "retrying" and
        first_pending("outage-1").get("retry_count", 0) >= 1,
        timeout=6.0)
    check("retry attempts happen (state -> retrying, retry_count grows)", transitioned,
          str(first_pending("outage-1").get("retry_count")))
    st = w.audit_writer_status()
    check("no in-memory leak while pending (durable spool)",
          st["in_memory_pending"] == 0, str(st["in_memory_pending"]))
    check("retry thread still alive after failed attempts",
          w._retry_thread is not None and w._retry_thread.is_alive())

    restore_sessions()
    recovered = wait_until(lambda: row_by_event_id(evA.event_id) is not None, timeout=10.0)
    check("recovery succeeds when DB-4 returns", recovered)
    rowA = row_by_event_id(evA.event_id)
    check("recovered row payload is EXACT (not {})",
          rowA is not None and json.loads(rowA.payload_summary).get("special")
          == "payload-must-survive")
    check("exactly one row for the event (no duplicate)",
          rowA is not None and
          len([r for r in all_rows() if r.event_id == evA.event_id]) == 1)
    check("pending line cleared after recovery",
          wait_until(lambda: len(pending_with_marker("outage-1")) == 0, timeout=6.0))
    okc, detail = chain_ok()
    check("chain valid after outage + recovery", okc, detail)

    # ══ 4. Failure mode C: retry-worker exception does not kill the worker ═══
    section("failure mode C: retry-worker exception survival")
    real_persist = w._persist_item

    def _exploding(_item):
        raise RuntimeError("injected retry-worker crash")

    w._persist_item = _exploding
    time.sleep(1.3)  # at least one sweep hits the injected exception
    alive = w._retry_thread is not None and w._retry_thread.is_alive()
    w._persist_item = real_persist
    check("retry worker survives an unexpected sweep exception", alive)
    evC = append_tracked({"marker": "post-crash", "event_id": "ev-p8-post-crash"})
    w.flush_audit_queue(5.0)
    check("retry/queue machinery works after worker exception",
          wait_until(lambda: row_by_event_id(evC.event_id) is not None, timeout=8.0))

    # ══ 5. Payload-less / malformed pending records (§7: never replay {}) ════
    section("malformed / payload-less pending records")
    legacy_id = f"legacy-{os.getpid()}-{int(time.time() * 1000)}"
    legacy_line = json.dumps({
        "event_id": legacy_id, "fraud_id": FRAUD_ID, "event_type": "legacy",
        "status": "pending", "reason": "legacy_no_payload", "retry_count": 0,
        "first_seen": w._now_iso(), "created_at": w._now_iso(),
        "updated_at": w._now_iso(), "next_retry_at": 0.0,
    }, sort_keys=True, separators=(",", ":"))
    with open(PENDING_FILE, "a", encoding="utf-8") as fh:
        fh.write(legacy_line + "\n")
    with open(PENDING_FILE, "a", encoding="utf-8") as fh:
        fh.write("this-is-not-json{\n")
    marked = wait_until(
        lambda: any(r.get("event_id") == legacy_id and r.get("status") == "failed"
                    for r in pending_records()), timeout=6.0)
    check("payload-less record marked FAILED (never replayed as {})", marked)
    check("no DB row written for payload-less record",
          row_by_event_id(legacy_id) is None)
    check("malformed line does not kill the retry worker",
          w._retry_thread is not None and w._retry_thread.is_alive())

    # ══ 6. Idempotency / duplicate replay (§9) ═══════════════════════════════
    section("idempotency: duplicate replay")
    payloadD = {"marker": "dup", "event_id": "ev-p8-dup", "risk_score": 3}
    evD = append_tracked(payloadD)
    w.flush_audit_queue(5.0)
    check("seed event persisted",
          wait_until(lambda: row_by_event_id(evD.event_id) is not None, timeout=6.0))
    itemD = w.AuditItem(event_id=evD.event_id, fraud_id=FRAUD_ID,
                        event_type="score_generated", payload=payloadD)
    r1 = w._persist_item(itemD)
    r2 = w._persist_item(itemD)
    r3 = w._persist_item(itemD)
    n_dup = len([r for r in all_rows() if r.event_id == evD.event_id])
    check("direct re-persist of same event_id is idempotent",
          r1 and r2 and r3 and n_dup == 1, f"rows={n_dup}")
    # File-level replay: a pending line for an already-written event must be
    # consumed without creating a second row.
    dup_line = json.dumps({
        "event_id": evD.event_id, "fraud_id": FRAUD_ID,
        "event_type": "score_generated", "payload": payloadD,
        "status": "pending", "reason": "duplicate_replay_probe", "retry_count": 0,
        "first_seen": w._now_iso(), "created_at": w._now_iso(),
        "updated_at": w._now_iso(), "next_retry_at": 0.0,
    }, sort_keys=True, separators=(",", ":"))
    with open(PENDING_FILE, "a", encoding="utf-8") as fh:
        fh.write(dup_line + "\n")
    consumed = wait_until(
        lambda: not any(r.get("event_id") == evD.event_id
                        for r in pending_records()), timeout=8.0)
    n_dup2 = len([r for r in all_rows() if r.event_id == evD.event_id])
    check("file-level duplicate consumed without duplicate row",
          consumed and n_dup2 == 1, f"rows={n_dup2}")

    # ══ 7. FIFO order under failure (§5) ═════════════════════════════════════
    section("FIFO order under failure")
    fail_sessions()
    fifo_ids = []
    for i in range(6):
        ev = append_tracked({"marker": f"fifo-{i}", "event_id": f"ev-p8-fifo-{i}"})
        fifo_ids.append(ev.event_id)
    got = wait_until(lambda: len([r for r in pending_records()
                                  if isinstance(r.get("payload"), dict)
                                  and r["payload"].get("marker", "").startswith("fifo-")]) == 6,
                     timeout=8.0)
    fifo_recs = [r for r in pending_records() if isinstance(r.get("payload"), dict)
                 and r["payload"].get("marker", "").startswith("fifo-")]
    order_ok = [r["payload"]["marker"] for r in fifo_recs] == [f"fifo-{i}" for i in range(6)]
    check("pending file preserves append order (FIFO)", got and order_ok,
          str([r["payload"].get("marker") for r in fifo_recs]))
    restore_sessions()
    rec_ok = wait_until(lambda: all(row_by_event_id(e) is not None for e in fifo_ids),
                        timeout=12.0)
    check("all FIFO events recovered", rec_ok)
    rows_f = [row_by_event_id(e) for e in fifo_ids]
    seqs = [r.seq for r in rows_f if r is not None]
    check("recovery replays in FIFO (seq ascending == append order)",
          seqs == sorted(seqs) and len(seqs) == 6, str(seqs))

    # ══ 8. Delayed audit persistence does not block the decision path (§5) ══
    section("delayed persistence (§5 delayed DB-4)")
    def slow_session():
        time.sleep(0.3)  # DB-4 slow, not failing
        return REAL_SESSION()
    w.SessionLocal = slow_session
    lats = []
    for i in range(5):
        t0 = time.perf_counter()
        append_tracked({"marker": f"slow-{i}", "event_id": f"ev-p8-slow-{i}"})
        lats.append((time.perf_counter() - t0) * 1000)
    perf["delayed_append_median_ms"] = sorted(lats)[len(lats) // 2]
    w.flush_audit_queue(6.0)
    restored_ok = wait_until(
        lambda: len([r for r in all_rows() if '"marker":"slow-' in r.payload_summary]) == 5,
        timeout=8.0)
    check("delayed writes do not become append/decision latency",
          max(lats) < 100, f"max={max(lats):.1f}ms median={perf['delayed_append_median_ms']:.1f}ms")
    check("delayed audit still completes later", restored_ok)
    restore_sessions()

    # ══ 9. Repeated failures: bounded retry + explicit FAILED (§10) ══════════
    section("repeated failures: bounded retry, explicit failed state")
    saved = (w.MAX_RETRIES_PER_ITEM, w.RETRY_BACKOFF_BASE, w.RETRY_BACKOFF_CAP)
    w.MAX_RETRIES_PER_ITEM = 3
    w.RETRY_BACKOFF_BASE = 0.02
    w.RETRY_BACKOFF_CAP = 0.08
    fail_sessions()
    failed_before = w.audit_writer_status()["permanently_failed"]
    evE1 = append_tracked({"marker": "exh-1", "event_id": "ev-p8-exh-1"})
    evE2 = append_tracked({"marker": "exh-2", "event_id": "ev-p8-exh-2"})
    # Explicit poll with state-change logging (the mystery lives here).
    deadline = time.monotonic() + 8.0
    exhausted = False
    _last_state = None
    while time.monotonic() < deadline:
        a = pending_with_marker("exh-1")
        b = pending_with_marker("exh-2")
        state = tuple((r.get("status"), r.get("reason"), r.get("retry_count")) for r in a + b)
        if state != _last_state:
            print(f"    [poll] t={time.time():.3f} state={state} "
                  f"session={_SESSION_MODE['mode']} "
                  f"alive={w._retry_thread.is_alive()}")
            _last_state = state
        if (all(r.get("status") == "failed" and str(r.get("reason", "")).startswith("retry_exhausted")
                for r in a + b) and len(a) == 1 and len(b) == 1):
            exhausted = True
            break
        time.sleep(0.05)
    check("retry budget enforced -> explicit FAILED state", exhausted)
    _recs_now = pending_with_marker("exh-1") + pending_with_marker("exh-2")
    _state_failed = (len(_recs_now) == 2
                     and all(r.get("status") == "failed" for r in _recs_now))
    check("exhaustion flag consistent with immediate file re-read",
          exhausted == _state_failed,
          f"exhausted={exhausted!r} state_failed={_state_failed} "
          f"recs={[(r.get('status'), r.get('reason')) for r in _recs_now]}")
    recs_exh = pending_with_marker("exh-1") + pending_with_marker("exh-2")
    check("failed records carry retry_count at budget",
          all(r.get("retry_count") == 3 for r in recs_exh),
          str([(r.get("status"), r.get("reason"), r.get("retry_count"))
               for r in recs_exh]))
    st = w.audit_writer_status()
    check("permanently_failed counter incremented",
          st["permanently_failed"] >= failed_before + 2,
          f"{failed_before} -> {st['permanently_failed']} "
          f"(max={w.MAX_RETRIES_PER_ITEM})")
    check("retry worker alive after exhaustion",
          w._retry_thread is not None and w._retry_thread.is_alive())
    check("exhausted events never reached the chain",
          row_by_event_id(evE1.event_id) is None and row_by_event_id(evE2.event_id) is None)
    w.MAX_RETRIES_PER_ITEM, w.RETRY_BACKOFF_BASE, w.RETRY_BACKOFF_CAP = saved
    restore_sessions()
    # FAILED records stay FAILED after DB-4 returns (no zombie retries).
    time.sleep(1.0)
    check("FAILED state is terminal (not retried after recovery)",
          all(r.get("status") == "failed" for r in pending_with_marker("exh-1")
              + pending_with_marker("exh-2")))

    # ══ 10. Pending-store bound / overflow (§10) ══════════════════════════════
    section("pending store bound (overflow behavior)")
    saved_cap = w.MAX_PENDING_LINES
    w.MAX_PENDING_LINES = 6
    fail_sessions()
    max_seen = 0
    for i in range(10):
        append_tracked({"marker": f"bound-{i}", "event_id": f"ev-p8-bound-{i}"})
        time.sleep(0.05)
        n_lines = w.audit_writer_status()["pending_file_lines"]
        max_seen = max(max_seen, n_lines)
    time.sleep(0.8)
    final_lines = w.audit_writer_status()["pending_file_lines"]
    max_seen = max(max_seen, final_lines)
    check("pending file never exceeds cap", final_lines <= 6 and max_seen <= 7,
          f"max_seen={max_seen} final={final_lines}")
    st = w.audit_writer_status()
    check("overflow recorded as explicit failures/evictions (not silent)",
          st["permanently_failed"] + st["evicted_lines"] > 0,
          f"failed={st['permanently_failed']} evicted={st['evicted_lines']}")
    check("writer/retry threads alive under overflow",
          w._audit_thread.is_alive() and w._retry_thread.is_alive())
    w.MAX_PENDING_LINES = saved_cap
    restore_sessions()
    wait_until(lambda: w.audit_writer_status()["pending_file_lines"] <=
               len([r for r in pending_records() if r.get("status") == "failed"]) + 1,
               timeout=10.0)

    # ══ 11. Queue overflow (queue_full -> durable spool, never caller DB) ════
    section("queue overflow (queue_full path)")
    big_q = w._audit_queue
    small_q: _queue.Queue = _queue.Queue(maxsize=2)
    w._chain_lock.acquire()  # stall the drain thread inside _persist_item
    try:
        w._audit_queue = small_q
        fail_sessions()
        q1 = append_tracked({"marker": "qfull-1", "event_id": "ev-p8-q-1"})
        q2 = append_tracked({"marker": "qfull-2", "event_id": "ev-p8-q-2"})
        # The drain thread may have stolen one slot (it is now parked inside
        # _persist_item on the chain lock); top up until the queue is full.
        fillers = []
        n_fill = 0
        while not small_q.full() and n_fill < 5:
            fillers.append(append_tracked({"marker": f"qfull-fill-{n_fill}",
                                           "event_id": f"ev-p8-q-fill-{n_fill}"}))
            n_fill += 1
        check("small queue saturated for overflow test", small_q.full(),
              f"size={small_q.qsize()} fillers={n_fill}")
        t0 = time.perf_counter()
        q3 = append_tracked({"marker": "qfull-3", "event_id": "ev-p8-q-3"})
        qfull_lat = (time.perf_counter() - t0) * 1000
        qspool = wait_until(lambda: len(pending_with_marker("qfull-3")) == 1, timeout=4.0)
        check("queue-full append does not do caller-thread DB write",
              qfull_lat < 100, f"{qfull_lat:.1f}ms")
        check("queue-full event spooled durably (reason queue_full)",
              qspool and first_pending("qfull-3").get("reason") == "queue_full",
              str(first_pending("qfull-3").get("reason")))
        # Restore: move queued items back to the big queue, restore everything,
        # then release the drain thread.
        restore_sessions()
        while True:
            try:
                item = small_q.get_nowait()
            except _queue.Empty:
                break
            big_q.put_nowait(item)
        w._audit_queue = big_q
    finally:
        w._chain_lock.release()
    recovery_ids = [q1.event_id, q2.event_id, q3.event_id] + \
        [e.event_id for e in fillers]
    check("queue-full events all recover after DB-4 returns",
          wait_until(lambda: all(row_by_event_id(e) is not None
                                 for e in recovery_ids),
                     timeout=10.0))

    # ══ 12. Lock contention: DB-4 locked (§5 locked) ═════════════════════════
    section("DB-4 locked (write-lock contention)")
    lock_con = sqlite3.connect(AUDIT_DB, timeout=30)
    lock_con.isolation_level = None
    lock_con.execute("BEGIN IMMEDIATE")
    try:
        t0 = time.perf_counter()
        l1 = append_tracked({"marker": "lock-1", "event_id": "ev-p8-lock-1",
                             "seq_note": "first"})
        lat_l1 = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        l2 = append_tracked({"marker": "lock-2", "event_id": "ev-p8-lock-2",
                             "seq_note": "second"})
        lat_l2 = (time.perf_counter() - t0) * 1000
        check("decisions (appends) not blocked while DB-4 locked",
              lat_l1 < 200 and lat_l2 < 200, f"{lat_l1:.1f}ms / {lat_l2:.1f}ms")
        spool_lock = wait_until(
            lambda: len(pending_with_marker("lock-1")) >= 1, timeout=14.0)
        check("locked-DB failure lands in pending store (recoverable)", spool_lock)
    finally:
        lock_con.execute("ROLLBACK")
        lock_con.close()
    rec_lock = wait_until(
        lambda: row_by_event_id(l1.event_id) is not None
        and row_by_event_id(l2.event_id) is not None, timeout=15.0)
    check("locked events recover after lock release", rec_lock)
    rl1 = row_by_event_id(l1.event_id)
    check("locked-event payload intact after replay",
          rl1 is not None and json.loads(rl1.payload_summary).get("seq_note") == "first")
    okc, detail = chain_ok()
    check("chain valid after lock scenario", okc, detail)

    # ══ 13. Shutdown safety (§5 shutdown, §11 lifecycle) ══════════════════════
    section("shutdown + repeated startup")
    fail_sessions()
    sh_ids = [append_tracked({"marker": f"shut-{i}", "event_id": f"ev-p8-shut-{i}"}).event_id
              for i in range(3)]
    wait_until(lambda: sum(len(pending_with_marker(f"shut-{i}")) for i in range(3)) == 3,
               timeout=6.0)
    res1 = w.shutdown_audit_writer(3.0)
    check("shutdown returns flushed_events int + thread_alive + status",
          isinstance(res1.get("flushed_events"), int)
          and isinstance(res1.get("thread_alive"), bool)
          and isinstance(res1.get("status"), dict),
          str({k: res1.get(k) for k in ("flushed_events", "thread_alive")}))
    check("pending events remain durable after shutdown",
          sum(len(pending_with_marker(f"shut-{i}")) for i in range(3)) == 3)
    check("retry worker stopped after shutdown",
          wait_until(lambda: not w._retry_thread.is_alive(), timeout=4.0))
    restore_sessions()
    res2 = w.shutdown_audit_writer(2.0)  # repeated shutdown must be safe
    check("repeated shutdown is safe", isinstance(res2, dict))
    # Restart path: next append re-ensures both workers (aliveness-based).
    evR = append_tracked({"marker": "restart", "event_id": "ev-p8-restart"})
    check("writer/retry restart after shutdown",
          wait_until(lambda: w._retry_thread is not None and w._retry_thread.is_alive(),
                     timeout=4.0))
    check("pre-shutdown events replay after restart",
          wait_until(lambda: all(row_by_event_id(e) is not None for e in sh_ids),
                     timeout=10.0))
    check("post-restart append persists", wait_until(
        lambda: row_by_event_id(evR.event_id) is not None, timeout=8.0))

    # ══ 14. Concurrency (§5 concurrency) ═════════════════════════════════════
    section("concurrent appends")
    n_before = len(all_rows())
    threads_n, per_thread = 8, 10
    errors: list[str] = []

    def worker(tid: int) -> None:
        try:
            for j in range(per_thread):
                ev = w.append_audit_event(FRAUD_ID, "score_generated",
                                          {"marker": f"conc-{tid}-{j}",
                                           "event_id": f"ev-p8-c-{tid}-{j}"})
                if ev is None or not ev.event_id:
                    errors.append(f"tid={tid} j={j} bad stub")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"tid={tid}: {exc!r}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    w.flush_audit_queue(8.0)
    expected = threads_n * per_thread
    got_all = wait_until(lambda: len(all_rows()) == n_before + expected, timeout=15.0)
    rows_now = all_rows()
    check("no exceptions across concurrent appends", not errors, str(errors[:3]))
    check("exact logical event count (no loss, no duplicates)",
          len(rows_now) == n_before + expected,
          f"rows={len(rows_now)} expected={n_before + expected}")
    ids = [r.event_id for r in rows_now]
    check("no duplicate event_ids", len(set(ids)) == len(ids))
    prevs = [r.prev_hash for r in rows_now]
    check("no chain forks (unique prev_hash per row)",
          len(set(prevs)) == len(prevs) and got_all)
    okc, detail = chain_ok()
    check("chain valid after concurrency", okc, detail)

    # ══ 15. Endpoints: decision-path invariant (§5 endpoint, §6) ═════════════
    section("decision-path invariant: healthy vs DB-4 failure (§6)")
    feats = vector(amount_ratio=1.1, failed_auth_count_24h=4, new_device_flag=1,
                   unusual_location_flag=1, unusual_recipient_flag=1, hour_of_day=2)
    with TestClient(risk_app) as c:
        # Healthy DB-4
        t0 = time.perf_counter()
        rh = c.post("/internal/evaluate", headers={"X-Internal-Token": TOKEN},
                    json={"event_id": "ev-p8-healthy-01", "fraud_id": FRAUD_ID,
                          "features": feats})
        lat_healthy = (time.perf_counter() - t0) * 1000
        body_h = rh.json()
        wait_until(lambda: len(rows_for_payload_event("ev-p8-healthy-01")) == 1, timeout=6.0)

        # DB-4 unavailable
        fail_sessions()
        t0 = time.perf_counter()
        rf = c.post("/internal/evaluate", headers={"X-Internal-Token": TOKEN},
                    json={"event_id": "ev-p8-failed-01", "fraud_id": FRAUD_ID,
                          "features": feats})
        lat_failing = (time.perf_counter() - t0) * 1000
        body_f = rf.json()
        perf["decision_healthy_ms"] = lat_healthy
        perf["decision_failing_ms"] = lat_failing

        check("HTTP status identical (200) with failing DB-4",
              rh.status_code == 200 and rf.status_code == 200,
              f"{rh.status_code} vs {rf.status_code}")
        h = {k: v for k, v in body_h.items() if k != "event_id"}
        f = {k: v for k, v in body_f.items() if k != "event_id"}
        diff_keys = sorted(k for k in set(h) | set(f) if h.get(k) != f.get(k))
        check("response bodies identical (minus event_id)",
              h == f, f"diff={diff_keys}")
        core = ("decision", "risk_score", "risk_band", "ml_score", "rule_score",
                "reason_codes", "degraded")
        check("fraud decision / score / metadata identical",
              all(body_h.get(k) == body_f.get(k) for k in core),
              str({k: (body_h.get(k), body_f.get(k)) for k in core
                   if body_h.get(k) != body_f.get(k)}))
        check("decision response not delayed by audit failure",
              lat_failing < max(3000.0, lat_healthy + 1500),
              f"healthy={lat_healthy:.0f}ms failing={lat_failing:.0f}ms")

        # Batch during audit failure (§13)
        b_ids = ["ev-p8-batch-0001", "ev-p8-batch-0002", "ev-p8-batch-0003"]
        t0 = time.perf_counter()
        rb = c.post("/internal/evaluate-batch", headers={"X-Internal-Token": TOKEN},
                    json=[{"event_id": b, "fraud_id": FRAUD_ID, "features": feats}
                          for b in b_ids])
        lat_batch = (time.perf_counter() - t0) * 1000
        perf["batch_failing_ms"] = lat_batch
        check("batch endpoint answers during audit failure",
              rb.status_code == 200, f"status={rb.status_code}")
        bres = rb.json().get("results", [])
        check("batch returns one result per request",
              rb.json().get("count") == 3 and len(bres) == 3,
              str(rb.json().get("count")))
        check("batch results carry decisions",
              all(r is not None and "decision" in r for r in bres))

        restore_sessions()

        # Recovery + pairing for the failed-mode events
        got_eval = wait_until(
            lambda: len(rows_for_payload_event("ev-p8-failed-01")) == 1, timeout=10.0)
        check("evaluate audit event recoverable after DB-4 returns", got_eval)
        got_batch = wait_until(
            lambda: all(len(rows_for_payload_event(b)) == 1 for b in b_ids), timeout=10.0)
        check("batch audit events all recovered", got_batch)
        pair_ok = True
        pair_detail = []
        for i, b in enumerate(b_ids):
            rows_b = rows_for_payload_event(b)
            if not rows_b:
                pair_ok = False
                continue
            p = json.loads(rows_b[0].payload_summary)
            same = p.get("risk_score") == bres[i].get("risk_score")
            pair_detail.append((b, same))
            pair_ok = pair_ok and same
        check("batch score<->event pairing intact (Phase 5 invariant)",
              pair_ok, str(pair_detail))
        check("one logical audit record per batch event",
              all(len(rows_for_payload_event(b)) == 1 for b in b_ids))

    # ══ 16. Final integrity sweep: chain + stored payload content (§12) ══════
    section("final integrity: chain + payload content")
    okc, detail = chain_ok()
    check("full-chain verification ok", okc, detail)
    rows_all = all_rows()
    content_bad = []
    for eid, payload in EXPECT_ROW.items():
        row = row_by_event_id(eid)
        if row is None:
            # Only acceptable if the event was deliberately terminal-failed.
            continue
        stored = json.loads(row.payload_summary)
        if stored != payload:
            content_bad.append((eid, stored))
    check("every stored payload matches what was submitted (no {} substitution)",
          not content_bad, str(content_bad[:2]))
    check("no empty payload rows written",
          all(r.payload_summary not in ("{}", '""', "")
              for r in rows_all),
          str([r.event_id for r in rows_all if r.payload_summary in ("{}", "")][:3]))
    ids = [r.event_id for r in rows_all]
    check("no duplicate logical records across the whole run",
          len(set(ids)) == len(ids), f"rows={len(rows_all)}")
    st = w.audit_writer_status()
    check("status views bounded and consistent",
          st["in_memory_pending"] == 0 and st["queue_size"] == 0,
          str({k: st[k] for k in ("in_memory_pending", "queue_size")}))

    # ══ 17. Controlled performance measurements (§17) ════════════════════════
    section("controlled performance (local, not Phase 9)")
    lats = []
    for i in range(50):
        t0 = time.perf_counter()
        append_tracked({"marker": f"perf-h-{i}", "event_id": f"ev-p8-perf-h-{i}"})
        lats.append((time.perf_counter() - t0) * 1000)
    perf["append_healthy_median_ms"] = sorted(lats)[len(lats) // 2]
    perf["append_healthy_pmax_ms"] = max(lats)
    fail_sessions()
    lats_f = []
    for i in range(50):
        t0 = time.perf_counter()
        append_tracked({"marker": f"perf-f-{i}", "event_id": f"ev-p8-perf-f-{i}"})
        lats_f.append((time.perf_counter() - t0) * 1000)
    perf["append_failing_median_ms"] = sorted(lats_f)[len(lats_f) // 2]
    perf["append_failing_pmax_ms"] = max(lats_f)
    restore_sessions()
    check("healthy append latency (median < 50ms)",
          perf["append_healthy_median_ms"] < 50,
          f"{perf['append_healthy_median_ms']:.2f}ms")
    check("DB-4-failure append latency non-blocking (median < 50ms)",
          perf["append_failing_median_ms"] < 50,
          f"{perf['append_failing_median_ms']:.2f}ms")
    t_rec = time.monotonic()
    recovered_all = wait_until(
        lambda: len([r for r in all_rows()
                     if '"marker":"perf-f-' in r.payload_summary]) == 50, timeout=20.0)
    perf["recovery_replay_seconds"] = time.monotonic() - t_rec
    check("all 50 failure-mode perf events replay after recovery", recovered_all)
    print("  measurements:", json.dumps({k: round(v, 2) for k, v in perf.items()}))

    # ══ 18. Final state ══════════════════════════════════════════════════════
    section("final state")
    okc, detail = chain_ok()
    check("final chain valid", okc, detail)
    shutdown = w.shutdown_audit_writer(3.0)
    check("final shutdown clean", isinstance(shutdown.get("flushed_events"), int))

    print(f"\n== Phase 8 audit-resilience: "
          f"{len(failures)} failed check(s), "
          f"{time.perf_counter() - t_suite:.1f}s ==")
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        print("writer trace (non-routine ops + last 20):")
        interesting = [t for t in TRACE
                       if not (t.startswith("PERSIST") and t.endswith("-> True"))]
        for t in (interesting + TRACE[-20:])[-120:]:
            print(f"  {t}")
        try:
            trace_path = Path(__file__).resolve().parents[2] / ".freebuff" / "p8" / "trace_last.txt"
            trace_path.write_text("\n".join(TRACE) + "\n", encoding="utf-8")
            print(f"full trace: {trace_path}")
        except Exception:  # noqa: BLE001
            pass
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
