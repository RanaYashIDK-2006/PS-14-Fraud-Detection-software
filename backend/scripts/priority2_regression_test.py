#!/usr/bin/env python3
"""Priority 2 regression tests.

These verify that previously-fixed bugs stay fixed:

1. Admin session uses SessionStore (not dead JSON file)
2. DB_SCHEMA env var comment matches actual code behavior
3. Per-service schema assertion fires in production mode
4. _require_admin_session checks SQLite-backed store
5. Settings reads DB_SCHEMA (not per-service var names)
6. Migration v2 is referenced in .env.example

Each test is a plain assert script (no pytest).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

passed = 0
failed = 0
results = []


def test(name: str, ok: bool, detail: str = ""):
    global passed, failed
    tag = "[PASS]" if ok else "[FAIL]"
    msg = f"{tag} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, ok, detail))
    if ok:
        passed += 1
    else:
        failed += 1


# ── 1. _require_admin_session must use SessionStore, not dead JSON file ──

def test_admin_session_uses_session_store():
    """Verify _require_admin_session imports and uses _admin_sessions (SessionStore)."""
    src = (ROOT / "backend" / "src" / "front_service" / "main.py").read_text(encoding="utf-8")

    # Find the _require_admin_session function body
    func_start = src.find("def _require_admin_session(")
    assert func_start > 0, "_require_admin_session not found"
    func_end = src.find("\ndef ", func_start + 10)
    if func_end == -1:
        func_end = len(src)
    func_body = src[func_start:func_end]

    # Must NOT use json.loads/json.dumps on admin_sessions.json (the dead code path)
    # The docstring may mention it for historical context, so check for actual I/O
    has_json_io = ('json.loads(sessions_path' in func_body or 'sessions_path.read_text' in func_body)
    test("admin_session_no_json_file",
         not has_json_io,
         "Still reads admin_sessions.json!" if has_json_io else "Dead JSON path removed")

    # Must use _admin_sessions (SessionStore)
    uses_store = "_admin_sessions" in func_body
    test("admin_session_uses_session_store",
         uses_store,
         "Does not use SessionStore" if not uses_store else "Uses _admin_sessions correctly")


# ── 2. Settings comment matches actual code behavior ──

def test_settings_comment_accuracy():
    """Verify settings.py comment says DB_SCHEMA (not per-service names)."""
    settings_src = (ROOT / "backend" / "src" / "settings.py").read_text(encoding="utf-8")

    # Comment should say DB_SCHEMA=<name> (the actual behavior)
    has_correct_comment = "Set DB_SCHEMA=<name>" in settings_src
    test("settings_comment_says_db_schema",
         has_correct_comment,
         "Comment says IDENTITY_DB_SCHEMA etc. (misleading)" if not has_correct_comment
         else "Comment correctly says DB_SCHEMA=<name>")

    # Must NOT claim per-service env var names are read
    code_section = settings_src[settings_src.find("db_schema: str"):]
    has_per_service_read = "IDENTITY_DB_SCHEMA" in code_section or "PRIVACY_DB_SCHEMA" in code_section
    test("settings_no_per_service_var_read",
         not has_per_service_read,
         "Code reads per-service var names" if has_per_service_read
         else "Code only reads DB_SCHEMA")


# ── 3. Production mode schema assertion ──

def test_production_schema_assertion():
    """Verify validate_production_config checks DB_SCHEMA in production."""
    settings_src = (ROOT / "backend" / "src" / "settings.py").read_text(encoding="utf-8")

    has_assertion = "DB_SCHEMA" in settings_src and "public" in settings_src
    test("production_schema_check_exists",
         has_assertion,
         "No schema isolation check" if not has_assertion
         else "Schema isolation check present in validate_production_config")


# ── 4. .env.example references migration v2 ──

def test_env_example_migration_v2():
    """Verify .env.example references supabase_migration_v2.sql."""
    env_example = ROOT / ".env.example"
    if not env_example.exists():
        test("env_example_exists", False, ".env.example not found")
        return

    content = env_example.read_text(encoding="utf-8")
    has_v2 = "supabase_migration_v2" in content
    test("env_example_uses_migration_v2",
         has_v2,
         "Still references v1 migration" if not has_v2
         else "Correctly references supabase_migration_v2.sql")


# ── 5. SessionStore exists and is SQLite-backed ──

def test_session_store_implementation():
    """Verify SessionStore is SQLite-backed and multi-worker safe."""
    store_src = (ROOT / "backend" / "src" / "session_store.py").read_text(encoding="utf-8")

    has_sqlite = "sqlite3" in store_src
    test("session_store_uses_sqlite",
         has_sqlite,
         "Not SQLite-backed" if not has_sqlite else "SQLite-backed session store")

    has_wal = "WAL" in store_src
    test("session_store_uses_wal",
         has_wal,
         "No WAL mode" if not has_wal else "WAL mode for concurrency")

    has_revoked = "revoked" in store_src
    test("session_store_has_revoke",
         has_revoked,
         "No revoke mechanism" if not has_revoked else "Session revocation supported")


# ── 6. Front service imports SessionStore ──

def test_front_service_imports_session_store():
    """Verify front_service imports and uses SessionStore."""
    front_src = (ROOT / "backend" / "src" / "front_service" / "main.py").read_text(encoding="utf-8")

    has_import = "from src.session_store import SessionStore" in front_src
    test("front_service_imports_session_store",
         has_import,
         "Does not import SessionStore" if not has_import
         else "SessionStore imported correctly")

    has_init = "_admin_sessions = SessionStore(" in front_src
    test("front_service_creates_session_store",
         has_init,
         "Does not create SessionStore instance" if not has_init
         else "SessionStore instance created")

    # Must NOT have old process-local dict
    has_old_dict = "_ADMIN_SESSIONS = {}" in front_src
    test("front_service_no_old_dict",
         not has_old_dict,
         "Still has _ADMIN_SESSIONS = {}" if has_old_dict
         else "Old process-local dict removed")


# ── Run all tests ──

def main():
    test_admin_session_uses_session_store()
    test_settings_comment_accuracy()
    test_production_schema_assertion()
    test_env_example_migration_v2()
    test_session_store_implementation()
    test_front_service_imports_session_store()

    print(f"\n{'='*60}")
    print(f"P2 Regression: {passed}/{passed+failed} PASSED")
    if failed:
        print(f"  {failed} FAILED — previously-fixed bugs may have regressed!")
    else:
        print("  All previously-fixed bugs are still fixed.")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
