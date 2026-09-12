#!/usr/bin/env python3
"""Encrypt the DB-1 PII columns left plaintext (privacy audit #25).

full_name / account_number / account_type -> Fernet-encrypted BLOB, using
the SAME key the Identity Service uses at runtime (.env pii_encryption_key
via load_dotenv_and_patch). Idempotent: rows already stored as bytes are
skipped, so it is safe to re-run. A consistent snapshot is taken first
(db/identity.db.bak_preencrypt.<timestamp>).

Run from the project root:
  python scripts/migrate_identity_pii.py
"""
from __future__ import annotations

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
from src.identity_service.security import decrypt_pii, encrypt_pii  # noqa: E402

FIELDS = ("full_name", "account_number", "account_type")


def main() -> int:
    db_path = Path(settings.identity_db_path)
    if not db_path.exists():
        print(f"identity db not found: {db_path}")
        return 2

    # Consistent snapshot before touching anything.
    bak = db_path.with_name(f"identity.db.bak_preencrypt.{int(time.time())}")
    src_con = sqlite3.connect(str(db_path))
    try:
        dst_con = sqlite3.connect(str(bak))
        try:
            src_con.backup(dst_con)
        finally:
            dst_con.close()
    finally:
        src_con.close()
    print(f"backup written: {bak}")

    con = sqlite3.connect(str(db_path), timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    try:
        rows = con.execute(
            "SELECT user_id, full_name, account_number, account_type FROM users"
        ).fetchall()
    except sqlite3.OperationalError as e:
        con.close()
        print(f"could not read users table: {e}")
        return 2

    updated = {f: 0 for f in FIELDS}
    already = {f: 0 for f in FIELDS}
    for user_id, full_name, account_number, account_type in rows:
        values = {
            "full_name": full_name,
            "account_number": account_number,
            "account_type": account_type,
        }
        new_values = {}
        for f in FIELDS:
            v = values[f]
            if v is None:
                continue
            if isinstance(v, (bytes, bytearray)):
                already[f] += 1
                continue
            new_values[f] = encrypt_pii(str(v))
        if new_values:
            sets = ", ".join(f"{f}=?" for f in new_values)
            con.execute(
                f"UPDATE users SET {sets} WHERE user_id=?",
                (*new_values.values(), user_id),
            )
            for f in new_values:
                updated[f] += 1
    con.commit()

    print(f"users scanned: {len(rows)}")
    for f in FIELDS:
        print(f"  {f}: {updated[f]} encrypted, {already[f]} already bytes")
    con.close()

    # Round-trip check on one migrated row.
    con = sqlite3.connect(str(db_path))
    sample = con.execute(
        "SELECT full_name FROM users WHERE typeof(full_name)='blob' LIMIT 1"
    ).fetchone()
    con.close()
    if sample:
        name = decrypt_pii(bytes(sample[0]))
        print(f"round-trip OK (decrypted sample full_name: {name!r})")
    elif rows:
        print("WARNING: no BLOB rows after migration - check field values")
    else:
        print("no users to migrate")
    return 0


if __name__ == "__main__":
    sys.exit(main())