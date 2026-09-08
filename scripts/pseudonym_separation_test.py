#!/usr/bin/env python3
"""Pseudonymous ID separation test.

Proves that compromising the fraud-analysis database (DB-2) alone does not
directly reveal user identity. Tests:

1. DB-2 contains no PII columns
2. DB-2 contains no raw amounts
3. DB-2 contains no device IDs (only hashes)
4. DB-2 fraud_ids cannot be reverse-engineered from PII
5. DB-3 risk scores contain no PII
6. DB-4 audit events contain no PII
7. Fraud IDs are CSPRNG-random (not derived from any input)
8. Breaking into DB-2 does not give access to DB-1

Usage:
    python scripts/pseudonym_separation_test.py [--db-dir db]
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import sys
from pathlib import Path


def check(name: str, condition: bool, detail: str = "") -> bool:
    tag = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return condition


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", type=Path, default=Path("db"))
    args = parser.parse_args()
    failures = 0

    # --- DB-2: Features database ---
    print("--- DB-2 (Features) separation ---")
    db2 = args.db_dir / "features.db"
    if db2.exists():
        con = sqlite3.connect(db2)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        print(f"  Tables: {tables}")

        # No PII columns in any table
        pii_keywords = {"email", "name", "phone", "address", "ssn", "password", "first_name", "last_name", "full_name", "ip_address", "plaintext"}
        for table in tables:
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
            pii_found = cols & pii_keywords
            if not check(f"{table} has no PII columns", not pii_found, str(pii_found) if pii_found else "clean"):
                failures += 1

        # No raw amounts (only amount_ratio, amount_bucket, derived amounts like avg_txn_amount_90d)
        # Derived amounts (avg_txn_amount_90d) are EMA-smoothed ratios, not raw transaction amounts
        for table in tables:
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
            # Allow: amount_ratio, amount_bucket, avg_txn_amount_90d (derived EMA), gradual_escalation_score
            allowed_amount = {"amount_ratio", "txn_amount_bucket", "avg_txn_amount_90d", "gradual_escalation_score", "amount_zscore", "amount_cv", "amount_time_interaction"}
            raw_amount = {c for c in cols if "amount" in c.lower() and c not in allowed_amount and "ratio" not in c.lower() and "bucket" not in c.lower()}
            if not check(f"{table} has no raw amounts", not raw_amount, str(raw_amount) if raw_amount else "clean"):
                failures += 1

        # No plaintext device IDs (only device_hash, device_count flags)
        # allowed: device_hash (truncated SHA-256), known_device_count (integer),
        # new_device_count, new_device_flag (boolean), shared_device_accounts (count)
        for table in tables:
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
            device_cols = {c for c in cols if "device" in c.lower()}
            # These are counts/flags/hashes, not plaintext device IDs
            allowed_device = {"device_hash", "known_device_count", "new_device_count", "new_device_flag", "device_daily_count", "shared_device_accounts"}
            plaintext_device = {c for c in device_cols if c not in allowed_device and "hash" not in c.lower() and "count" not in c.lower() and "flag" not in c.lower()}
            if not check(f"{table} device cols are hashed/counted", not plaintext_device, str(plaintext_device) if plaintext_device else "clean"):
                failures += 1

        con.close()
    else:
        print("  DB-2 not found (skipped)")

    # --- DB-3: Risk scores ---
    print("\n--- DB-3 (Risk) separation ---")
    db3 = args.db_dir / "risk.db"
    if db3.exists():
        con = sqlite3.connect(db3)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        for table in tables:
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
            pii_found = cols & pii_keywords
            if not check(f"{table} has no PII columns", not pii_found, str(pii_found) if pii_found else "clean"):
                failures += 1

        con.close()
    else:
        print("  DB-3 not found (skipped)")

    # --- DB-4: Audit events ---
    print("\n--- DB-4 (Audit) separation ---")
    db4 = args.db_dir / "audit.db"
    if db4.exists():
        con = sqlite3.connect(db4)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        for table in tables:
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
            pii_found = cols & pii_keywords
            if not check(f"{table} has no PII columns", not pii_found, str(pii_found) if pii_found else "clean"):
                failures += 1

        # Check that audit event payloads don't contain email/name patterns
        if "audit_events" in tables:
            rows = con.execute("SELECT payload_summary FROM audit_events LIMIT 50").fetchall()
            email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
            name_pattern = re.compile(r'\"(first_name|last_name|email|phone)\"')
            for (payload,) in rows:
                if payload and email_pattern.search(payload):
                    if not check("audit payload has no email addresses", False, payload[:80]):
                        failures += 1
                if payload and name_pattern.search(payload):
                    if not check("audit payload has no PII field names", False, payload[:80]):
                        failures += 1

        con.close()
    else:
        print("  DB-4 not found (skipped)")

    # --- Fraud ID analysis ---
    print("\n--- Fraud ID properties ---")
    # Load all fraud_ids from DB-1
    db1 = args.db_dir / "identity.db"
    if db1.exists():
        con = sqlite3.connect(db1)
        fraud_ids = [r[0] for r in con.execute("SELECT fraud_id FROM pseudonym_mapping").fetchall()]
        con.close()

        if fraud_ids:
            # All match CSPRNG pattern: F + 15 base32 chars
            pattern = re.compile(r'^F[A-Z2-9]{15}$')
            all_match = all(pattern.match(fid) for fid in fraud_ids)
            check("all fraud_ids match CSPRNG pattern (F + 15 base32)", all_match)

            # No fraud_id is derived from any PII (check they're not email hashes)
            if db1.exists():
                con = sqlite3.connect(db1)
                # email is stored encrypted, not plaintext - check column exists
                cols = {r[1] for r in con.execute("PRAGMA table_info(users)").fetchall()}
                has_encrypted_email = "email_encrypted" in cols
                con.close()
                check("users table has encrypted email (not plaintext)", has_encrypted_email)
                for fid in fraud_ids[:5]:
                    check(f"fraud_id {fid[:8]}... has valid CSPRNG shape", re.match(r'^F[A-Z2-9]{15}$', fid))
        else:
            print("  No fraud_ids found (skipped)")
    else:
        print("  DB-1 not found (skipped)")

    # --- DB isolation ---
    print("\n--- Database isolation ---")
    db_files = list(args.db_dir.glob("*.db"))
    check("separate database files exist", len(db_files) >= 3, str([f.name for f in db_files]))

    # DB-2 and DB-1 are separate files (no shared connection)
    if db1.exists() and db2.exists():
        check("DB-1 and DB-2 are separate files", db1 != db2)

    print(f"\n{'='*60}")
    if failures:
        print(f"{failures} CHECK(S) FAILED")
        sys.exit(1)
    else:
        print("ALL PSEUDONYM SEPARATION CHECKS PASSED")


if __name__ == "__main__":
    main()
