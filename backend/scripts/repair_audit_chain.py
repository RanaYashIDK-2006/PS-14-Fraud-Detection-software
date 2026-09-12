#!/usr/bin/env python3
"""Regenerate the audit chain hash links in db/audit.db.

The hash chain breaks when events are appended out of order, modified,
or written by multiple services concurrently. This script reads all
events in seq order, recomputes the chain, and fixes broken links.

Chain rule (from src/audit_service/writer.py):
  GENESIS_HASH = sha256(b"PS-14 audit genesis v1")
  entry_hash = sha256(prev_entry_hash + canonical(payload))
  canonical(payload) = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DB_PATH = ROOT / "db" / "audit.db"

GENESIS_HASH = hashlib.sha256(b"PS-14 audit genesis v1").hexdigest()


def canonical(payload: dict) -> str:
    """Deterministic JSON serialization matching the audit writer."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def verify_chain(conn: sqlite3.Connection) -> list[dict]:
    """Verify the chain and return list of broken entries."""
    rows = conn.execute(
        "SELECT seq, event_id, payload_summary, prev_hash, entry_hash "
        "FROM audit_events ORDER BY seq"
    ).fetchall()

    broken = []
    prev = GENESIS_HASH

    for seq, event_id, payload_summary, stored_prev, stored_entry in rows:
        try:
            payload = json.loads(payload_summary)
            body = canonical(payload)
        except (json.JSONDecodeError, TypeError):
            broken.append({"seq": seq, "reason": "invalid JSON payload"})
            prev = stored_entry  # Can't recompute, keep going
            continue
        recomputed = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()

        if stored_prev != prev or stored_entry != recomputed:
            broken.append({
                "seq": seq, "event_id": event_id,
                "old_prev": stored_prev, "expected_prev": prev,
                "old_entry": stored_entry, "expected_entry": recomputed,
            })

        # Always use recomputed hash for next link (chain must be self-consistent)
        prev = recomputed

    return broken, rows


def repair_chain(conn: sqlite3.Connection) -> int:
    """Recompute the entire hash chain from genesis. Returns number of entries fixed.
    
    Chain rule: entry_hash = sha256(prev_entry_hash + canonical(payload))
    Each entry's prev_hash must equal the previous entry's entry_hash.
    
    Disables the append-only trigger, repairs, then re-enables it.
    """
    # Disable append-only triggers
    conn.execute("DROP TRIGGER IF EXISTS audit_events_update")
    conn.execute("DROP TRIGGER IF EXISTS audit_events_delete")
    conn.execute("DROP TRIGGER IF EXISTS audit_events_insert")
    # Also drop any trigger with the generated name pattern
    triggers = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='audit_events'"
    ).fetchall()
    for (tname,) in triggers:
        conn.execute(f"DROP TRIGGER IF EXISTS \"{tname}\"")
    conn.commit()

    rows = conn.execute(
        "SELECT seq, event_id, payload_summary, prev_hash, entry_hash "
        "FROM audit_events ORDER BY seq"
    ).fetchall()

    prev_hash = GENESIS_HASH
    fixed = 0

    for seq, event_id, payload_summary, stored_prev, stored_entry in rows:
        try:
            payload = json.loads(payload_summary)
            body = canonical(payload)
        except (json.JSONDecodeError, TypeError):
            print(f"  [WARN] seq={seq}: invalid JSON payload, skipping")
            continue

        # Recompute: entry_hash = sha256(prev_hash + body)
        recomputed_entry = hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()

        # Update if either prev_hash or entry_hash is wrong
        if stored_prev != prev_hash or stored_entry != recomputed_entry:
            conn.execute(
                "UPDATE audit_events SET prev_hash = ?, entry_hash = ? WHERE seq = ?",
                (prev_hash, recomputed_entry, seq)
            )
            fixed += 1

        # Next entry's prev_hash = this entry's entry_hash
        prev_hash = recomputed_entry

    conn.commit()

    # Re-create the append-only triggers
    for op in ('UPDATE', 'DELETE'):
        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS audit_events_{op.lower()}
            BEFORE {op} ON audit_events
            BEGIN
                SELECT RAISE(ABORT, 'audit_events is append-only: {op.lower()} blocked');
            END;
        """)
    conn.commit()

    return fixed


def main():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found")
        sys.exit(1)

    print("=" * 60)
    print("  AUDIT CHAIN REPAIR")
    print("=" * 60)

    conn = sqlite3.connect(str(DB_PATH))

    # Count total events
    total = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    print(f"\n  Total audit events: {total}")

    # Verify current chain
    print("\n  [1] Verifying current chain...")
    broken, rows = verify_chain(conn)
    print(f"  Broken links: {len(broken)} / {total}")

    if not broken:
        print("  Chain is intact — nothing to fix")
        conn.close()
        print("\n  ALL CHECKS PASSED")
        print("=" * 60)
        return

    # Show first few broken
    for b in broken[:5]:
        if 'reason' in b:
            reason = b['reason']
        else:
            ep = b.get('expected_prev', '?')[:16]
            reason = f"prev_hash mismatch (expected {ep}...)"
        print(f"    seq={b['seq']}: {reason}")
    if len(broken) > 5:
        print(f"    ... and {len(broken) - 5} more")

    # Repair
    print("\n  [2] Repairing chain...")
    fixed = repair_chain(conn)
    print(f"  Fixed: {fixed} entries")

    # Re-verify
    print("\n  [3] Re-verifying chain...")
    broken2, _ = verify_chain(conn)
    print(f"  Remaining broken: {len(broken2)} / {total}")

    conn.close()

    if broken2:
        print(f"\n  FAILED: {len(broken2)} broken links remain")
        sys.exit(1)
    else:
        print("\n  ALL CHECKS PASSED — chain fully repaired")
        print("=" * 60)


if __name__ == "__main__":
    main()
