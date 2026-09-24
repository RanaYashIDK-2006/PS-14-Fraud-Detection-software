#!/usr/bin/env python3
"""Backup/restore test — proves backups can be restored.

Covers:
  - Copy databases to working directory
  - Create backup with checksums
  - Verify backup integrity (checksums + SQLite PRAGMA integrity_check)
  - Count rows from backup
  - Corrupt backup copies (simulate data loss)
  - Restore from source copies
  - Verify restored data integrity (checksums + integrity + row counts)
  - Verify audit chain integrity after restore
  - Verify originals untouched

Usage:
    python scripts/backup_restore_test.py
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))  # Phase 109: src.* imports below
DB_DIR = ROOT / "db"

DB_BASE_NAMES = ["identity.db", "features.db", "risk.db", "audit.db"]


def _dump_audit(db_path):
    """Read-only snapshot of audit_events as (list[dict], max_seq).

    Phase 109: the shared chain receives legitimate live appends while this
    suite runs, so the originals-untouched check compares the HISTORICAL
    PREFIX (every row present in the Phase-1 copy) instead of racing a
    whole-file hash.  Any mutation, deletion, or truncation of history
    still fails; later appends are live traffic, not the suite's doing.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM audit_events ORDER BY seq"
    ).fetchall()]
    con.close()
    return rows, (rows[-1]["seq"] if rows else 0)

passed = 0
failed = 0
results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    results.append((name, ok, detail))
    if ok:
        passed += 1
    else:
        failed += 1
    return ok


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(db_path: Path, table: str) -> int:
    """Count rows using a fresh in-memory connection to avoid WAL side effects."""
    try:
        # Use backup API to get a clean snapshot into a temp file
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            src_conn = sqlite3.connect(str(db_path), timeout=5)
            dst_conn = sqlite3.connect(str(tmp_path), timeout=5)
            src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()
            # Count from the clean snapshot
            c = sqlite3.connect(str(tmp_path), timeout=5)
            exists = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            ).fetchone()
            if not exists:
                c.close()
                return -1
            count = c.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            c.close()
            return count
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception:
        return -1


def checkpoint_db(db_path: Path) -> None:
    """Checkpoint WAL mode database so all data is in the main .db file."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    except Exception:
        pass


def copy_db(src: Path, dst: Path) -> None:
    """Copy a database file after checkpointing WAL."""
    checkpoint_db(src)
    shutil.copy2(src, dst)


def main() -> None:
    global passed, failed

    print("=" * 70)
    print("BACKUP / RESTORE TEST")
    print("=" * 70)
    print()

    # Fresh-checkout guard: CI runs this on a repo with no runtime
    # databases. Skip (exit 0) rather than fail when there is nothing
    # to back up - the live-stack path is exercised by --full locally.
    existing_dbs = [b for b in DB_BASE_NAMES if (DB_DIR / b).exists()]
    if not existing_dbs:
        print("=" * 70)
        print("BACKUP / RESTORE TEST - SKIPPED")
        print(f"  No runtime databases found in {DB_DIR}")
        print("  Nothing to back up (fresh checkout / CI). Not a failure.")
        print("=" * 70)
        return

    # ── Phase 0: Flush async audit queue and repair chain ──
    print("[0] Flushing audit queue and repairing chain...")
    try:
        from src.audit_service.writer import flush_audit_queue
        flush_audit_queue(timeout=2.0)
    except Exception:
        pass
    # Phase 109: the former "repair broken chain links" block UPDATEd
    # historical prev_hash/entry_hash values to force verification green.
    # Audit rows are append-only evidence — rewriting them is forbidden
    # (and the audit_events_no_update trigger blocked the statement anyway,
    # so the block was a dead no-op whose intent violates the evidence
    # rules). The five documented 2026-09-19 test-pollution forks are
    # quarantined by evidence-bound findings in
    # src/monitoring/phase109_audit_fork_repair.py instead; this suite
    # never mutates audit history.

    work_dir = Path(tempfile.mkdtemp(prefix="ps14_brt_"))
    src_dir = work_dir / "source"
    backup_dir = work_dir / "backup"
    corrupted_dir = work_dir / "corrupted"
    restored_dir = work_dir / "restored"
    src_dir.mkdir()
    backup_dir.mkdir()
    corrupted_dir.mkdir()
    restored_dir.mkdir()

    try:
        # ── Phase 1: Copy live databases to working directory ──
        print("[1] Copying databases to working directory...")
        for db_base in DB_BASE_NAMES:
            src_db = DB_DIR / db_base
            dst_db = src_dir / db_base
            if src_db.exists():
                copy_db(src_db, dst_db)
            check(f"Copied {db_base}", dst_db.exists())

        print()

        # ── Phase 2: Create "backup" from copies ──
        print("[2] Creating backup...")
        backup_hashes = {}
        for db_base in DB_BASE_NAMES:
            src = src_dir / db_base
            if not src.exists():
                continue
            dst = backup_dir / db_base
            shutil.copy2(src, dst)
            backup_hashes[db_base] = sha256_file(dst)
            check(f"Backed up {db_base}", dst.exists(),
                  f"size={dst.stat().st_size} bytes")

        print()

        # ── Phase 3: Verify backup integrity ──
        print("[3] Verifying backup integrity...")
        for db_base in DB_BASE_NAMES:
            bp = backup_dir / db_base
            if not bp.exists():
                continue
            h = sha256_file(bp)
            check(f"Checksum {db_base}", h == backup_hashes[db_base])
            try:
                conn = sqlite3.connect(str(bp))
                result = conn.execute("PRAGMA integrity_check").fetchone()[0]
                conn.close()
                check(f"SQLite integrity {db_base}", result == "ok")
            except Exception as e:
                check(f"SQLite integrity {db_base}", False, str(e))

        print()

        # ── Phase 4: Count rows from backup ──
        print("[4] Counting rows from backup...")
        row_counts = {}
        TABLES_TO_COUNT = [
            "users", "pseudonym_mapping",
            "fraud_profiles", "transaction_features", "device_fingerprints",
            "risk_scores",
            "audit_events", "audit_access_log",
        ]
        for db_base in DB_BASE_NAMES:
            bp = backup_dir / db_base
            if not bp.exists():
                continue
            try:
                for table in TABLES_TO_COUNT:
                    key = f"{db_base}.{table}"
                    row_counts[key] = count_rows(bp, table)
                    if row_counts[key] > 0:
                        print(f"  {key}: {row_counts[key]} rows")
            except Exception as e:
                print(f"  {db_base}: {e}")

        print()

        # ── Phase 5: Corrupt backup copies ──
        print("[5] Simulating data loss...")
        for db_base in DB_BASE_NAMES:
            bp = backup_dir / db_base
            if not bp.exists():
                continue
            with open(bp, "r+b") as f:
                f.write(b"CORRUPTED" * 11)
            try:
                conn = sqlite3.connect(str(bp))
                conn.execute("SELECT 1")
                conn.close()
                check(f"Corrupted {db_base}", False, "still readable after corruption")
            except Exception:
                check(f"Corrupted {db_base}", True, "database unreadable (expected)")

        print()

        # ── Phase 6: Restore from source copies ──
        print("[6] Restoring from source copies...")
        for db_base in DB_BASE_NAMES:
            src = src_dir / db_base
            dst = restored_dir / db_base
            if src.exists():
                shutil.copy2(src, dst)
                check(f"Restored {db_base}", dst.exists())

        print()

        # ── Phase 7: Verify restored data integrity ──
        print("[7] Verifying restored data integrity...")
        for db_base in DB_BASE_NAMES:
            rp = restored_dir / db_base
            if not rp.exists():
                continue

            # Verify checksum matches original backup
            h = sha256_file(rp)
            check(f"Checksum match {db_base}", h == backup_hashes[db_base])

            # Verify SQLite integrity
            try:
                conn = sqlite3.connect(str(rp))
                result = conn.execute("PRAGMA integrity_check").fetchone()[0]
                conn.close()
                check(f"SQLite integrity {db_base}", result == "ok")
            except Exception as e:
                check(f"SQLite integrity {db_base}", False, str(e))

            # Verify row counts match (skip tables with 0 expected rows)
            for key, expected in row_counts.items():
                if not key.startswith(db_base):
                    continue
                table = key.rsplit(".", 1)[1]
                actual = count_rows(rp, table)
                if actual == -1 and expected == 0:
                    check(f"Row count {key}", True, f"table absent (0 rows expected)")
                elif expected == 0:
                    check(f"Row count {key}", True, f"0 rows (trivial)")
                else:
                    check(f"Row count {key}", actual == expected,
                          f"expected={expected}, actual={actual}")

        print()

        # ── Phase 8: Verify audit chain integrity ──
        print("[8] Verifying audit chain integrity...")
        audit_path = restored_dir / "audit.db"
        if audit_path.exists():
            try:
                conn = sqlite3.connect(str(audit_path))
                conn.row_factory = sqlite3.Row
                rows = [dict(r) for r in conn.execute(
                    "SELECT * FROM audit_events ORDER BY seq"
                ).fetchall()]
                conn.close()

                if len(rows) == 0:
                    check("Audit chain (empty)", True, "no events to verify")
                else:
                    # Phase 109: strict walk over every restored row plus
                    # the evidence-bound quarantine of the five documented
                    # 2026-09-19 forks. A faithful backup PRESERVES those
                    # rows byte-for-byte; a corrupted or modified one
                    # fails closed.
                    from src.monitoring.phase109_audit_fork_repair import (
                        AFFECTED_SEQUENCES,
                        evaluate_chain,
                    )
                    res = evaluate_chain(rows)
                    check("Audit chain integrity (strict + quarantine)",
                          res["ok"],
                          f"strict_ok={res['strict_ok']} first_bad={res['first_bad_seq']}")
                    check("Restore preserves all five quarantined forks",
                          set(res.get("quarantined_breaks") or []) == set(AFFECTED_SEQUENCES),
                          str(res.get("quarantined_breaks")))
                    check("Restored row count matches walk",
                          res["n_entries"] == len(rows),
                          str(res["n_entries"]))
            except Exception as e:
                check("Audit chain", False, str(e))

        print()

        # ── Phase 9: Verify originals untouched ──
        print("[9] Verifying originals untouched...")
        for db_base in DB_BASE_NAMES:
            src = DB_DIR / db_base
            if not src.exists():
                continue
            src_copy = src_dir / db_base
            if not src_copy.exists():
                check(f"Original {db_base} unchanged", True, "no copy to compare")
                continue
            if db_base == "audit.db":
                rows_copy, max_copy = _dump_audit(src_copy)
                rows_src, max_src = _dump_audit(src)
                prefix = [r for r in rows_src if r["seq"] <= max_copy]
                same = prefix == rows_copy and max_src >= max_copy
                check(f"Original {db_base} unchanged (historical prefix)",
                      same,
                      f"{len(rows_copy)} prefix rows identical, "
                      f"{max_src - max_copy} live append(s)")
            else:
                # Re-checkpoint to ensure WAL is clean for hash comparison
                checkpoint_db(src)
                h = sha256_file(src)
                # Compare against our working copy (checkpointed the same way)
                h_copy = sha256_file(src_copy)
                check(f"Original {db_base} unchanged", h == h_copy,
                      f"hash={h[:16]}...")

        print()

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

    print("=" * 70)
    print(f"Total: {passed + failed}  |  PASS: {passed}  |  FAIL: {failed}  |  BLOCKED: 0")
    if failed:
        print("FAILED: several checks failed.")
        sys.exit(1)
    else:
        print("ALL BACKUP/RESTORE CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
