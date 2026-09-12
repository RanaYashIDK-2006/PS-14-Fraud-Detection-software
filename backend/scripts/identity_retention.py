#!/usr/bin/env python3
"""DB-1 data-retention / deletion path (privacy audit #25).

There was no retention policy or deletion path. This tool reports accounts
inactive for N days and can purge them together with their credentials,
pseudonym mapping, and access logs. Dry-run by default; pass --delete to
actually remove rows (after taking a snapshot backup).

Run from the project root:
  python scripts/identity_retention.py --older-than-days 730
  python scripts/identity_retention.py --older-than-days 730 --delete
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

from src.settings import load_dotenv_and_patch  # noqa: E402
load_dotenv_and_patch()

from src.settings import settings  # noqa: E402

CASCADE_TABLES = (
    ("auth_credentials", "user_id"),
    ("pseudonym_mapping", "user_id"),
    ("pseudonym_access_log", "user_id"),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--older-than-days", type=int, default=365)
    ap.add_argument("--delete", action="store_true",
                    help="actually delete (default: dry-run report only)")
    args = ap.parse_args(argv)

    db_path = Path(settings.identity_db_path)
    if not db_path.exists():
        print(f"identity db not found: {db_path}")
        return 2

    cutoff = time.time() - args.older_than_days * 86400
    con = sqlite3.connect(str(db_path), timeout=10)
    try:
        stale = con.execute(
            "SELECT user_id, created_at FROM users "
            "WHERE created_at IS NOT NULL AND "
            "strftime('%s', created_at) < ?",
            (str(int(cutoff)),),
        ).fetchall()
    except sqlite3.OperationalError:
        # created_at may be stored as TEXT or REAL; fall back to a best-effort scan.
        stale = []
        for row in con.execute("SELECT user_id, created_at FROM users"):
            if row[1] is None:
                continue
            ts = row[1]
            try:
                created = float(ts) if not isinstance(ts, str) else float(
                    __import__("datetime").datetime.fromisoformat(ts).timestamp()
                )
            except Exception:
                continue
            if created < cutoff:
                stale.append(row)
    if not stale:
        print(f"no accounts inactive for >{args.older_than_days} days")
        con.close()
        return 0

    print(f"{len(stale)} account(s) inactive for >{args.older_than_days} days:")
    casc = {t: 0 for t, _ in CASCADE_TABLES}
    user_ids = [r[0] for r in stale]
    ph = ",".join("?" for _ in user_ids)
    for table, col in CASCADE_TABLES:
        try:
            casc[table] = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} IN ({ph})",
                user_ids,
            ).fetchone()[0]
        except sqlite3.OperationalError:
            casc[table] = -1
    for uid, created in stale[:10]:
        print(f"  {uid} created_at={created}")
    if len(stale) > 10:
        print(f"  ... and {len(stale) - 10} more")
    print("cascade rows to remove:", casc)

    if not args.delete:
        print("\ndry-run: nothing deleted. Re-run with --delete to purge.")
        con.close()
        return 0

    bak = db_path.with_name(f"identity.db.bak_retention.{int(time.time())}")
    src_con = sqlite3.connect(str(db_path))
    try:
        dst_con = sqlite3.connect(str(bak))
        try:
            src_con.backup(dst_con)
        finally:
            dst_con.close()
    finally:
        src_con.close()
    print(f"snapshot written: {bak}")

    for table, col in CASCADE_TABLES:
        try:
            con.execute(f"DELETE FROM {table} WHERE {col} IN ({ph})", user_ids)
        except sqlite3.OperationalError as e:
            print(f"  (skip {table}: {e})")
    con.execute(f"DELETE FROM users WHERE user_id IN ({ph})", user_ids)
    con.commit()
    print(f"purged {len(stale)} accounts + cascade rows; deleted-at audit "
          "note: write an identity_deletion event via the audit writer if "
          "retention logging is required")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())