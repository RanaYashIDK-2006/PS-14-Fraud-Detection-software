#!/usr/bin/env python3
"""Phase 4A: versioned PII key rotation — safety and failure tests.

Hermetic: the suite builds its own key material, temp SQLite stores and calls
the migration tool's functions directly. No service, no .env, no live PII.

Covered (phase brief §18/§19):
  1  encrypt with the current key            10 JWT rotation does not break PII
  2  decrypt with the current key            11 PII rotation does not break JWT
  3  migrate legacy -> current               12 missing current key fails closed
  4  decrypt the migrated value              13 missing legacy key fails closed
  5  reject invalid ciphertext               14 unknown key version is refused
  6  reject unknown key version              15 corrupted row is never overwritten
  7  plaintext never logged                  16 write failure rolls back safely
  8  key material never in errors            17 duplicate migration is idempotent
  9  quarantined rows stay untouched         18 no plaintext/keys in the report
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

import jwt  # noqa: E402

from rotate_pii_key import migrate  # noqa: E402
from src.pii_crypto import (  # noqa: E402
    LEGACY_VERSION, PiiCipher, UnknownKeyVersion, derive_fernet_key,
)

LEGACY_MAT = "phase4a-legacy-key-material-0001"
NEW_MAT = "phase4a-current-key-material-0002"
MARKER = "PHASE4A-PLAINTEXT-MARKER-7f31c9"

legacy_cipher = PiiCipher(LEGACY_VERSION, {LEGACY_VERSION: LEGACY_MAT})
rotating_cipher = PiiCipher("v2", {LEGACY_VERSION: LEGACY_MAT, "v2": NEW_MAT})

passed = failed = 0
errors: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        errors.append(f"{name}{' — ' + detail if detail else ''}")
        print(f"  [FAIL] {name}{' — ' + detail if detail else ''}")


def make_store(rows: int = 3, *, trigger: bool = False) -> Path:
    """Temp identity-style store with legacy-key ciphertext."""
    d = Path(tempfile.mkdtemp(prefix="ps14_p4a_"))
    store = d / "identity.db"
    conn = sqlite3.connect(store)
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            full_name BLOB, phone_encrypted BLOB, email_encrypted BLOB,
            address_encrypted BLOB, account_number BLOB, account_type BLOB
        )""")
    for i in range(1, rows + 1):
        conn.execute(
            "INSERT INTO users (full_name, phone_encrypted, email_encrypted) "
            "VALUES (?, ?, ?)",
            (legacy_cipher.encrypt(f"{MARKER}-name-{i}"),
             legacy_cipher.encrypt(f"{MARKER}-phone-{i}"),
             legacy_cipher.encrypt(f"{MARKER}-email-{i}")))
    if trigger:
        conn.execute("CREATE TRIGGER block_updates BEFORE UPDATE ON users "
                     "BEGIN SELECT RAISE(ABORT, 'write blocked'); END")
    conn.commit()
    conn.close()
    return store


def _rejects(fn) -> bool:
    """True when the call raises (fail-closed behavior)."""
    try:
        fn()
        return False
    except Exception:  # noqa: BLE001 - any refusal counts
        return True


def rows_of(store: Path) -> list[tuple]:
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT rowid, full_name, phone_encrypted, email_encrypted FROM users"
        ).fetchall()
    finally:
        conn.close()


print("=== 1-4: round trip, migration, read-back ===")
tok = rotating_cipher.encrypt("x")
check("encrypt with current key round-trips",
      rotating_cipher.decrypt(tok) == b"x")
legacy_tok = legacy_cipher.encrypt("legacy-value")
check("legacy (untagged) token readable under rotation",
      rotating_cipher.decrypt(legacy_tok) == b"legacy-value")
check("legacy token flagged for migration",
      rotating_cipher.needs_migration(legacy_tok) and not rotating_cipher.needs_migration(tok))

store = make_store(rows=3)
pre = rows_of(store)
pre_plain = [[rotating_cipher.decrypt(v) for v in row[1:]] for row in pre]
rep = migrate(store, c=rotating_cipher)
post = rows_of(store)
check("migration reports every token migrated",
      rep["migrated"] == 9 and rep["failed"] == 0 and rep["verified"] is True,
      json.dumps({k: rep.get(k) for k in ("migrated", "failed", "verified")}))
check("no row dropped or duplicated", [r[0] for r in post] == [r[0] for r in pre])
check("migrated tokens are tagged with the new version",
      all(bytes(v).startswith(b"v2:") for row in post for v in row[1:]))
check("migrated tokens decrypt to the same plaintext",
      [[rotating_cipher.decrypt(v) for v in row[1:]] for row in post] == pre_plain)
check("post-migration store needs no further migration",
      rep["post"]["needs_migration"] == 0 and rep["post"]["decryptable"] == rep["post"]["tokens"])

print("=== 5-6: invalid ciphertext and unknown versions ===")
corrupt = bytes(legacy_tok[:10] + b"AAAA" + legacy_tok[14:])
try:
    rotating_cipher.decrypt(corrupt)
    check("corrupted ciphertext rejected", False, "no exception")
except Exception as exc:
    check("corrupted ciphertext rejected",
          isinstance(exc, Exception) and not isinstance(exc, SystemExit))
try:
    rotating_cipher.decrypt(b"v9:gAAAAABunknown")
    check("unknown key version refused", False, "no exception")
except UnknownKeyVersion:
    check("unknown key version refused", True)
except Exception as exc:  # noqa: BLE001
    check("unknown key version refused", False, type(exc).__name__)

print("=== 7-8, 18: no plaintext and no key material in logs, errors or report ===")
records: list[str] = []


class Capture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        records.append(record.getMessage())


root = logging.getLogger()
handler = Capture()
root.addHandler(handler)
try:
    store_log = make_store(rows=2)
    migrate(store_log, c=rotating_cipher)
    rotating_cipher.decrypt(rotating_cipher.encrypt(MARKER))
finally:
    root.removeHandler(handler)
log_text = "\n".join(records)
check("plaintext never appears in logs", MARKER not in log_text)
check("key material never appears in logs",
      NEW_MAT not in log_text and LEGACY_MAT not in log_text)
err_texts: list[str] = []
for fn in (lambda: rotating_cipher.decrypt(b"gAAAAAbogus"),
           lambda: rotating_cipher.decrypt(b"v9:gAAAAAbogus"),
           lambda: derive_fernet_key("")):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        err_texts.append(str(exc))
check("key material never appears in error messages",
      all(NEW_MAT not in t and LEGACY_MAT not in t for t in err_texts))
report_text = json.dumps(rep)
check("no plaintext or key material in the migration report",
      MARKER not in report_text and NEW_MAT not in report_text
      and LEGACY_MAT not in report_text)

print("=== 9, 15: quarantined and corrupted rows are never overwritten ===")
store_bad = make_store(rows=2)
conn = sqlite3.connect(store_bad)
conn.execute("INSERT INTO users (full_name, phone_encrypted, email_encrypted) "
             "VALUES (?, ?, ?)", (b"gAAAAAcorrupt-row", b"gAAAAAalsobad", b"gAAAAAalsobad2"))
conn.commit()
conn.close()
before = rows_of(store_bad)
rep_bad = migrate(store_bad, c=rotating_cipher)
after = rows_of(store_bad)
bad_before = [r for r in before if r[0] == 3][0]
bad_after = [r for r in after if r[0] == 3][0]
check("unreadable row left byte-identical",
      bad_before[1:] == bad_after[1:])
check("good rows still migrated around the quarantined row",
      rep_bad["migrated"] == 6)
check("unreadable row accounted (quarantined or failed)",
      (rep_bad.get("quarantined", 0) + rep_bad.get("failed", 0)) > 0,
      json.dumps({"quarantined": rep_bad.get("quarantined"),
                  "failed": rep_bad.get("failed")}))

print("=== 16: write failure rolls back, store stays readable on the old key ===")
store_locked = make_store(rows=2, trigger=True)
rep_locked = migrate(store_locked, c=rotating_cipher)
after_locked = rows_of(store_locked)
check("write failure reported, not raised",
      rep_locked["applied"] is False and rep_locked["error"] is not None
      and rep_locked["failed"] > 0,
      json.dumps({"error": rep_locked.get("error"), "failed": rep_locked.get("failed")}))
check("no row was changed by the failed migration",
      all(bytes(v).startswith(b"gAAAAA") for row in after_locked for v in row[1:]))
check("rows still readable with the legacy key after rollback",
      all(rotating_cipher.decrypt(v).startswith(MARKER.encode())
          for row in after_locked for v in row[1:]))

print("=== 17: duplicate migration is idempotent ===")
store_idem = make_store(rows=2)
first = migrate(store_idem, c=rotating_cipher)
first_rows = rows_of(store_idem)
second = migrate(store_idem, c=rotating_cipher)
second_rows = rows_of(store_idem)
check("second migration migrates nothing",
      second["migrated"] == 0 and second["skipped_current"] == 6
      and second["verified"] is True)
check("second migration leaves ciphertext byte-identical",
      first_rows == second_rows and first["migrated"] == 6)

print("=== 10-13: rotation independence and fail-closed key handling ===")
jwt_old, jwt_new = "phase4a-jwt-old", "phase4a-jwt-new"
token = rotating_cipher.encrypt(MARKER)
signed = jwt.encode({"sub": "F000000000000001"}, jwt_old, algorithm="HS256")
assert jwt.decode(signed, jwt_old, algorithms=["HS256"])["sub"] == "F000000000000001"
jwt_rotated = jwt.encode({"sub": "F000000000000001"}, jwt_new, algorithm="HS256")
check("JWT rotation does not break PII decryption",
      rotating_cipher.decrypt(token) == MARKER.encode()
      and jwt.decode(jwt_rotated, jwt_new, algorithms=["HS256"])["sub"]
      == "F000000000000001")
migrated_token = rotating_cipher.encrypt(MARKER)
check("PII rotation does not break JWT authentication",
      jwt.decode(jwt_rotated, jwt_new, algorithms=["HS256"])["sub"]
      == "F000000000000001"
      and rotating_cipher.decrypt(migrated_token) == MARKER.encode())
check("old JWT secret no longer validates the new token",
      _rejects(lambda: jwt.decode(jwt_rotated, jwt_old, algorithms=["HS256"])))
check("missing current key material fails closed",
      _rejects(lambda: PiiCipher("v2", {})))
untagged = legacy_cipher.encrypt(MARKER)
check("legacy token unreadable without the legacy key",
      _rejects(lambda: PiiCipher("v2", {"v2": NEW_MAT}).decrypt(untagged)))
check("old key cannot read ciphertext written under the new key",
      _rejects(lambda: legacy_cipher.decrypt(rotating_cipher.encrypt(MARKER))))

print("=== 19: reverse migration (documented rollback path) ===")
store_rev = make_store(rows=2)
forward = migrate(store_rev, c=rotating_cipher)
rev_cipher = PiiCipher(LEGACY_VERSION, {LEGACY_VERSION: LEGACY_MAT, "v2": NEW_MAT})
back = migrate(store_rev, c=rev_cipher)
rows_back = rows_of(store_rev)
check("reverse migration moves rows back onto the legacy key",
      back["migrated"] == forward["migrated"] and back["verified"] is True
      and back["post"]["needs_migration"] == 0,
      json.dumps({"migrated": back["migrated"], "verified": back["verified"]}))
legacy_only = PiiCipher(LEGACY_VERSION, {LEGACY_VERSION: LEGACY_MAT})
check("reverted rows are readable by a legacy-only configuration",
      all(legacy_only.decrypt(v).startswith(MARKER.encode())
          for row in rows_back for v in row[1:]))
check("reverted rows are not readable by the new key alone",
      all(_rejects(lambda v=v: PiiCipher("v2", {"v2": NEW_MAT}).decrypt(v))
          for row in rows_back for v in row[1:]))

print("=" * 60)
print(f"PII KEY ROTATION TESTS: {passed} passed, {failed} failed")
print("=" * 60)
if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  - {e}")
sys.exit(0 if failed == 0 else 1)
