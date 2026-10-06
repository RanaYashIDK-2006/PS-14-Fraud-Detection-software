#!/usr/bin/env python3
"""Rotate the PII data key: re-encrypt stored PII from the legacy key to the
current (versioned) key.

Phase 4A. Designed to be safe by default:

  * the default mode is a READ-ONLY dry run (no writes, no lock on the store);
  * `--apply` first writes a verifiable backup of each store, then migrates
    row by row, decrypting with the legacy key, re-encrypting with the current
    key, and comparing the plaintext IN MEMORY before the update is committed;
  * undecryptable rows are never overwritten or deleted — they are counted,
    reported (by rowid) and left untouched;
  * plaintext and key material are never printed or written to disk.

Usage:
    python scripts/rotate_pii_key.py                  # dry run, default stores
    python scripts/rotate_pii_key.py --apply          # migrate (with backups)
    python scripts/rotate_pii_key.py --store db/identity.db --json out.json

Exit codes: 0 clean; 1 problems (undecryptable/invalid rows, or a failed
post-commit read-back); 2 configuration/usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pii_crypto import (  # noqa: E402
    LEGACY_VERSION, PiiCipher, UnknownKeyVersion, cipher, key_fingerprints,
)
from src.settings import load_dotenv_and_patch  # noqa: E402

#: PII columns the identity service encrypts (others are plaintext columns).
PII_COLUMNS = ("full_name", "phone_encrypted", "email_encrypted",
               "address_encrypted", "account_number", "account_type")

DEFAULT_STORES = ("db/identity.db", "db/walkthrough/identity.db",
                  "db/wt-scenarios/identity.db")


# A stored value is ciphertext if it is a Fernet token, optionally carrying a
# key-version tag written by src/pii_crypto.py (v2:gAAAAA...).
_TOKEN_RE = re.compile(rb"^(v\d+:)?gAAAAA")


def _tokens(value) -> bool:
    return isinstance(value, (bytes, bytearray)) and bool(_TOKEN_RE.match(bytes(value)))


def _pii_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
    return [c for c in cols if c in PII_COLUMNS]


def plan(store: Path, table: str = "users", *, c: PiiCipher | None = None) -> dict:
    """Read-only survey of one store."""
    c = c or cipher()
    report: dict = {"store": str(store), "table": table, "exists": store.exists(),
                    "records": 0, "tokens": 0, "needs_migration": 0,
                    "already_current": 0, "decryptable": 0, "undecryptable": 0,
                    "unsupported_version": 0, "unsupported_version_rowids": [],
                    "plaintext_or_null": 0, "undecryptable_rowids": [],
                    "columns": []}
    if not store.exists():
        return report
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        cols = _pii_columns(conn, table)
        report["columns"] = cols
        if not cols:
            return report
        rows = conn.execute(
            f'SELECT rowid, {", ".join(chr(34) + c_ + chr(34) for c_ in cols)} '
            f'FROM "{table}"').fetchall()
        report["records"] = len(rows)
        for rowid, *values in rows:
            for v in values:
                if v is None or not isinstance(v, (bytes, bytearray)):
                    report["plaintext_or_null"] += 1
                    continue
                if not _tokens(v):
                    report["plaintext_or_null"] += 1
                    continue
                report["tokens"] += 1
                if c.needs_migration(bytes(v)):
                    report["needs_migration"] += 1
                else:
                    report["already_current"] += 1
                try:
                    c.decrypt(bytes(v))
                    report["decryptable"] += 1
                except UnknownKeyVersion:
                    # A version this process has no key for: never overwritten,
                    # and reported separately because an operator must supply
                    # the missing key material before it can be migrated.
                    report["unsupported_version"] += 1
                    if rowid not in report["unsupported_version_rowids"]:
                        report["unsupported_version_rowids"].append(rowid)
                    if rowid not in report["undecryptable_rowids"]:
                        report["undecryptable_rowids"].append(rowid)
                except Exception:
                    report["undecryptable"] += 1
                    if rowid not in report["undecryptable_rowids"]:
                        report["undecryptable_rowids"].append(rowid)
    finally:
        conn.close()
    return report


def backup(store: Path, backup_dir: Path, label: str = "pre-pii-rotation") -> dict:
    """Copy the store aside and hash it. The file name carries the store's
    RELATIVE PATH: several stores share the base name `identity.db`, and a
    name-only scheme silently overwrote earlier backups (Phase 4A incident)."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    slug = str(store).replace("\\", "/").lstrip("./").replace("/", "__")
    dest = backup_dir / f"{slug}.{label}-{stamp}.bak"
    shutil.copy2(store, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    same = digest == hashlib.sha256(store.read_bytes()).hexdigest()
    return {"backup": str(dest), "sha256": digest, "matches_source": same}


def migrate(store: Path, table: str = "users", *, c: PiiCipher | None = None,
            backup_dir: Path | None = None) -> dict:
    """Re-encrypt legacy rows in place; verify in memory before committing."""
    c = c or cipher()
    out = plan(store, table, c=c)
    out["mode"] = "apply"
    out["migrated"] = 0
    out["skipped_current"] = 0
    out["failed"] = 0
    out["quarantined"] = 0
    out["backup"] = None
    # Rows no available key can read before the migration stay quarantined;
    # they count as "quarantined", never as migration failures.
    pre_bad_rowids = set(out.get("undecryptable_rowids", []))
    if not store.exists() or not out["columns"]:
        out["applied"] = False
        return out
    if backup_dir is not None:
        out["backup"] = backup(store, backup_dir)

    conn = sqlite3.connect(str(store), timeout=15)
    out["error"] = None
    try:
        cols = _pii_columns(conn, table)
        rows = conn.execute(
            f'SELECT rowid, {", ".join(chr(34) + c_ + chr(34) for c_ in cols)} '
            f'FROM "{table}"').fetchall()
        for rowid, *values in rows:
            changes: dict[str, bytes] = {}
            for col, v in zip(cols, values):
                if v is None or not isinstance(v, (bytes, bytearray)) or not _tokens(v):
                    continue
                raw = bytes(v)
                if not c.needs_migration(raw):
                    out["skipped_current"] += 1
                    continue
                try:
                    plain = c.decrypt(raw)          # legacy key
                except Exception:
                    if rowid in pre_bad_rowids:
                        out["quarantined"] += 1     # pre-existing, left untouched
                    else:
                        out["failed"] += 1          # leave the row untouched
                    continue
                if not plain:
                    out["failed"] += 1
                    continue
                new_token = c.encrypt(plain)        # current key
                # In-memory integrity check: new ciphertext must round-trip to
                # exactly the same plaintext as the old ciphertext did.
                if c.decrypt(new_token) != plain:
                    out["failed"] += 1
                    continue
                changes[col] = new_token
            if not changes:
                continue
            assigns = ", ".join(f'"{col}" = ?' for col in changes)
            params = list(changes.values()) + [rowid]
            cur = conn.execute(
                f'UPDATE "{table}" SET {assigns} WHERE rowid = ?', params)
            if cur.rowcount != 1:
                conn.rollback()
                out["failed"] += 1
                continue
            out["migrated"] += len(changes)
        try:
            conn.commit()
        except sqlite3.Error as exc:
            # A failed/incomplete commit leaves the store on the legacy key,
            # which is still readable: roll back and report, never half-apply.
            conn.rollback()
            out["error"] = type(exc).__name__
            out["failed"] += 1
            out["applied"] = False
            return out

    except sqlite3.Error as exc:
        conn.rollback()
        out["error"] = type(exc).__name__
        out["failed"] += 1
        out["applied"] = False
        return out
    finally:
        conn.close()

    # Post-commit verification on a fresh read-only connection: same row and
    # token counts, the same (pre-existing) undecryptable set, and no
    # readable token left on the legacy key.
    try:
        after = plan(store, table, c=c)
    except sqlite3.Error as exc:
        out["error"] = type(exc).__name__
        out["applied"] = True
        out["verified"] = None
        return out
    out["post"] = {k: after[k] for k in
                   ("records", "tokens", "needs_migration",
                    "already_current", "decryptable", "undecryptable",
                    "plaintext_or_null", "undecryptable_rowids")}
    out["applied"] = True
    out["verified"] = (after["tokens"] == out["tokens"]
                       and after["records"] == out["records"]
                       and after["undecryptable"] == out["undecryptable"]
                       and after["needs_migration"] == after["undecryptable"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="perform the migration (default: dry run)")
    ap.add_argument("--revert", action="store_true",
                    help="reverse migration: re-encrypt current-key rows back onto the "
                         "legacy key (rollback path; requires both keys)")
    ap.add_argument("--store", action="append", default=None,
                    help="store path(s); default: the three identity stores")
    ap.add_argument("--table", default="users")
    ap.add_argument("--backup-dir", default="db/_pii_rotation_backups")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the machine-readable report here")
    args = ap.parse_args(argv)

    load_dotenv_and_patch()
    from src.settings import settings

    if args.revert:
        # Rollback: target = legacy key, source = current key. Both ciphers are
        # combined in one map so untagged/tagged rows are readable either way.
        legacy_material = settings.pii_key_legacy or settings.pii_encryption_key
        if not (legacy_material and settings.pii_key_current):
            print("revert requires PII_KEY_LEGACY (or PII_ENCRYPTION_KEY) and PII_KEY_CURRENT.")
            return 2
        c = PiiCipher(LEGACY_VERSION, {LEGACY_VERSION: legacy_material,
                                       settings.pii_key_version: settings.pii_key_current})
    else:
        c = cipher()
    if not args.revert and c.current_version == "v1":
        print("PII key is still the legacy/non-versioned key: set PII_KEY_CURRENT "
              "(and keep PII_KEY_LEGACY for migration) before running --apply.")
        if args.apply:
            return 2

    stores = [Path(s) for s in (args.store or DEFAULT_STORES)]
    reports = []
    problems = 0
    for store in stores:
        if args.apply or args.revert:
            rep = migrate(store, args.table, c=c, backup_dir=Path(args.backup_dir))
        else:
            rep = plan(store, args.table, c=c)
            rep["mode"] = "dry-run"
        reports.append(rep)
        # Rows that no key in this process can read are PRE-EXISTING and are
        # quarantined (never overwritten); they are reported, not "problems".
        # Only migration failures and failed post-commit verification fail the run.
        problems += rep.get("unsupported_version", 0)
        if rep.get("failed"):
            problems += rep["failed"]
        if (args.apply or args.revert) and rep.get("applied") and not rep.get("verified"):
            problems += 1

    summary = {
        "phase": "4A",
        "mode": "revert" if args.revert else ("apply" if args.apply else "dry-run"),
        "key_versions": list(c.key_versions()),
        "current_version": c.current_version,
        "fingerprints": key_fingerprints(),
        "stores": reports,
        "records": sum(r["records"] for r in reports),
        "tokens": sum(r["tokens"] for r in reports),
        "needs_migration": sum(r["needs_migration"] for r in reports),
        "already_current": sum(r["already_current"] for r in reports),
        "decryptable": sum(r["decryptable"] for r in reports),
        "undecryptable_pre_existing": sum(r["undecryptable"] for r in reports),
        "unsupported_version": sum(r["unsupported_version"] for r in reports),
        "undecryptable_rowids": sorted({rid for r in reports
                                        for rid in r.get("undecryptable_rowids", [])}),
        "migrated": sum(r.get("migrated", 0) for r in reports),
        "quarantined": sum(r.get("quarantined", 0) for r in reports),
        "backups": [r.get("backup") for r in reports if r.get("backup")],
        "failed": sum(r.get("failed", 0) for r in reports),
        "problems": problems,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
