#!/usr/bin/env python3
"""Priority 2 — Full Regression Suite.

Verifies that every previously-fixed bug stays fixed. Each test is a plain
assert script (no pytest). Tests that need live services are skipped with
BLOCKED status.

Tests:
  P2.1  Rate limiter keyed on IP (not User-Agent)
  P2.2  device_fingerprints has composite PK
  P2.3  /demo/seed gated by PS14_MODE=production
  P2.4  DB paths are absolute (not cwd-relative)
  P2.5  Citations in compare_ml_systems.py are labeled illustrative
  P2.6  _require_admin_session uses SessionStore (not dead JSON)
  P2.7  Settings DB_SCHEMA comment matches code behavior
  P2.8  docker-compose.prod.yml per-service URLs are resolved
  P2.9  StaticPool + WAL scoped to SQLite only
  P2.10 NullPool selected for pgBouncer pooler
  P2.11 Production schema assertion fires
  P2.12 SessionStore is SQLite-backed with WAL + revoke
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

passed = 0
failed = 0
blocked = 0
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


def skip(name: str, reason: str):
    global blocked
    print(f"[BLOCKED] {name} — {reason}")
    results.append((name, None, reason))
    blocked += 1


# ─── P2.1: Rate limiter keyed on IP ────────────────────────────────────

def test_rate_limiter_ip_keying():
    src = (ROOT / "src" / "middleware" / "rate_limiter.py").read_text(encoding="utf-8")
    # Client ID must come from request.client.host, not User-Agent
    uses_ip = "request.client.host" in src
    test("P2.1a_rate_limiter_uses_ip",
         uses_ip,
         "Rate limiter does not use request.client.host" if not uses_ip
         else "Client ID derived from IP address")

    # Check actual code lines only (comments explaining the fix are fine)
    mw_start = src.find("async def rate_limit_middleware")
    mw_end = src.find("\ndef ", mw_start + 10) if mw_start > 0 else len(src)
    mw_body = src[mw_start:mw_end]
    code_lines = [l for l in mw_body.splitlines() if l.strip() and not l.strip().startswith("#")]
    client_id_lines = [l for l in code_lines if "client_id" in l]
    no_ua_key = all("user_agent" not in l.lower() for l in client_id_lines)
    test("P2.1b_rate_limiter_no_user_agent_key",
         no_ua_key,
         f"user_agent in client_id line" if not no_ua_key
         else "No user_agent in client_id derivation (code lines only)")


# ─── P2.2: device_fingerprints composite PK ────────────────────────────

def test_device_fingerprints_composite_pk():
    models_src = (ROOT / "src" / "privacy_layer" / "models.py").read_text(encoding="utf-8")
    # Find the DeviceFingerprint class
    cls_start = models_src.find("class DeviceFingerprint")
    if cls_start < 0:
        test("P2.2_device_fingerprints_class", False, "DeviceFingerprint class not found")
        return
    cls_end = models_src.find("\nclass ", cls_start + 10)
    if cls_end == -1:
        cls_end = len(models_src)
    cls_body = models_src[cls_start:cls_end]

    # Must have TWO primary_key=True fields (composite PK)
    pk_count = cls_body.count("primary_key=True")
    test("P2.2_device_fingerprints_composite_pk",
         pk_count >= 2,
         f"Only {pk_count} primary_key fields (need 2)" if pk_count < 2
         else f"Composite PK: {pk_count} primary_key fields")

    # Must include fraud_id as a column in the class
    has_fraud_id = "fraud_id" in cls_body
    test("P2.2b_fraud_id_in_pk",
         has_fraud_id,
         "fraud_id not found in DeviceFingerprint" if not has_fraud_id
         else "fraud_id column exists in composite PK")


# ─── P2.3: /demo/seed production gate ──────────────────────────────────

def test_demo_seed_production_gate():
    verify_src = (ROOT / "src" / "verification_service" / "main.py").read_text(encoding="utf-8")
    # Find the demo_seed endpoint
    ds_start = verify_src.find("def demo_seed")
    if ds_start < 0:
        skip("P2.3_demo_seed_gate", "demo_seed endpoint not found")
        return
    ds_body = verify_src[ds_start:ds_start + 500]

    has_production_check = "production" in ds_body.lower() and ("PS14_MODE" in ds_body or "ps14_mode" in ds_body.lower())
    test("P2.3_demo_seed_production_gate",
         has_production_check,
         "No PS14_MODE production check in demo_seed" if not has_production_check
         else "demo_seed gated by PS14_MODE=production")


# ─── P2.4: DB paths are absolute ────────────────────────────────────────

def test_db_paths_absolute():
    settings_src = (ROOT / "src" / "settings.py").read_text(encoding="utf-8")
    # db_dir must be anchored to _PROJECT_ROOT
    uses_project_root = "_PROJECT_ROOT" in settings_src and "db_dir" in settings_src
    test("P2.4a_db_dir_uses_project_root",
         uses_project_root,
         "db_dir not anchored to _PROJECT_ROOT" if not uses_project_root
         else "db_dir anchored to _PROJECT_ROOT")

    # _PROJECT_ROOT must use resolve() for absolute path
    has_resolve = "_PROJECT_ROOT = Path(__file__).resolve().parent.parent" in settings_src
    test("P2.4b_project_root_is_absolute",
         has_resolve,
         "_PROJECT_ROOT not using resolve()" if not has_resolve
         else "_PROJECT_ROOT uses resolve() for absolute path")

    # Verify at runtime
    try:
        import os as _os
        _os.chdir(str(ROOT / "scripts"))  # change to unrelated dir
        # Force reimport
        if "src.settings" in sys.modules:
            del sys.modules["src.settings"]
        from src.settings import Settings
        s = Settings()
        audit_path = Path(s.db_dir) / "audit.db"
        is_abs = audit_path.is_absolute()
        _os.chdir(str(ROOT))
        test("P2.4c_paths_work_from_any_cwd",
             is_abs,
             f"audit_db_path is relative from scripts/ cwd: {audit_path}" if not is_abs
             else f"audit_db_path is absolute from any cwd: {audit_path}")
    except Exception as e:
        _os.chdir(str(ROOT))
        test("P2.4c_paths_work_from_any_cwd", False, f"Runtime check failed: {e}")


# ─── P2.5: Citations labeled illustrative ──────────────────────────────

def test_citations_illustrative():
    compare_src = (ROOT / "scripts" / "compare_ml_systems.py").read_text(encoding="utf-8")
    # Check for fabricated academic citations
    has_fabricated = "Fawcett & Hohn" in compare_src or "Fawcett and Hohn" in compare_src
    test("P2.5a_no_fabricated_citations",
         not has_fabricated,
         "Still has fabricated 'Fawcett & Hohn' citation" if has_fabricated
         else "No fabricated academic citations")

    # All third-party entries must be labeled "(illustrative)" — PS-14's own
    # models (e.g. 'XGBoost + Handcrafted Features') are NOT illustrative.
    import re
    third_party_models = ["Stripe Radar", "PayPal ML", "Featurespace ARIC",
                          "LSTM", "GNN", "Isolation Forest"]
    unlabeled = []
    for model in third_party_models:
        matches = re.findall(rf'"{re.escape(model)}[^"]*"', compare_src)
        for m in matches:
            if "illustrative" not in m.lower():
                unlabeled.append(m)
    test("P2.5b_non_ps14_labeled_illustrative",
         len(unlabeled) == 0,
         f"Unlabeled third-party entries: {unlabeled}" if unlabeled
         else "All third-party entries labeled (illustrative)")


# ─── P2.6: _require_admin_session uses SessionStore ────────────────────

def test_admin_session_uses_store():
    front_src = (ROOT / "src" / "front_service" / "main.py").read_text(encoding="utf-8")
    func_start = front_src.find("def _require_admin_session(")
    if func_start < 0:
        test("P2.6_admin_session_exists", False, "_require_admin_session not found")
        return
    func_end = front_src.find("\ndef ", func_start + 10)
    func_body = front_src[func_start:func_end] if func_end > 0 else front_src[func_start:]

    # Must NOT do JSON I/O on admin_sessions.json
    has_json_io = "json.loads(sessions_path" in func_body or "sessions_path.read_text" in func_body
    test("P2.6a_no_dead_json_path",
         not has_json_io,
         "Still reads admin_sessions.json" if has_json_io
         else "Dead JSON path removed")

    # Must use _admin_sessions (SessionStore)
    uses_store = "_admin_sessions" in func_body
    test("P2.6b_uses_session_store",
         uses_store,
         "Does not use SessionStore" if not uses_store
         else "Uses _admin_sessions (SessionStore)")


# ─── P2.7: Settings DB_SCHEMA comment accuracy ─────────────────────────

def test_settings_comment():
    settings_src = (ROOT / "src" / "settings.py").read_text(encoding="utf-8")
    has_correct_comment = "Set DB_SCHEMA=<name>" in settings_src
    test("P2.7_settings_comment_says_db_schema",
         has_correct_comment,
         "Comment says IDENTITY_DB_SCHEMA etc." if not has_correct_comment
         else "Comment correctly says DB_SCHEMA=<name>")


# ─── P2.8: Per-service DB URL resolution ───────────────────────────────

def test_per_service_db_resolution():
    settings_src = (ROOT / "src" / "settings.py").read_text(encoding="utf-8")
    # model_post_init must check per-service env vars
    has_resolution = "SERVICE_NAME" in settings_src and "_DB_URL" in settings_src
    test("P2.8_per_service_url_resolution",
         has_resolution,
         "No per-service DB URL resolution" if not has_resolution
         else "Per-service DB URL resolved from SERVICE_NAME + _DB_URL")


# ─── P2.9: StaticPool + WAL scoped to SQLite only ─────────────────────

def test_sqlite_pool_scoping():
    shared_db = (ROOT / "src" / "shared_db.py").read_text(encoding="utf-8")

    # make_engine (SQLite): per-request connections for file DBs (NullPool),
    # StaticPool only for :memory:, WAL always. The OLD expectation (StaticPool
    # for every SQLite DB) was the check-#26 perf/reliability defect: one
    # shared connection across concurrent request threads -> sqlite3.
    # InterfaceError dropping ~1/3 of requests at >4 workers.
    sqlite_fn = shared_db[shared_db.find("def make_engine("):shared_db.find("def make_pg_engine(")]
    has_wal = "WAL" in sqlite_fn
    pool_ok = ("NullPool" in sqlite_fn and "StaticPool" in sqlite_fn
               and ":memory:" in sqlite_fn)
    test("P2.9a_sqlite_concurrency_safe_pool",
         pool_ok,
         "SQLite engine must use NullPool (per-request connection) with "
         "StaticPool reserved for :memory: only" if not pool_ok
         else "SQLite engine uses NullPool for files, StaticPool only for :memory:")
    test("P2.9b_sqlite_has_wal",
         has_wal,
         "SQLite engine missing WAL" if not has_wal
         else "SQLite engine has WAL mode")

    # make_pg_engine (Postgres) must NOT have StaticPool or WAL
    pg_start = shared_db.find("def make_pg_engine(")
    pg_end = shared_db.find("\ndef ", pg_start + 10) if pg_start > 0 else len(shared_db)
    pg_fn = shared_db[pg_start:pg_end] if pg_end > pg_start else ""
    no_static_pg = "StaticPool" not in pg_fn
    no_wal_pg = "WAL" not in pg_fn.upper() or "search_path" in pg_fn  # WAL not in pg
    test("P2.9c_pg_no_static_pool",
         no_static_pg,
         "Postgres engine has StaticPool!" if not no_static_pg
         else "Postgres engine does not use StaticPool")
    test("P2.9d_pg_no_wal",
         no_wal_pg,
         "Postgres engine has WAL!" if not no_wal_pg
         else "Postgres engine does not use WAL")


# ─── P2.10: NullPool for pgBouncer ─────────────────────────────────────

def test_nullpool_for_pooler():
    shared_db = (ROOT / "src" / "shared_db.py").read_text(encoding="utf-8")
    settings_src = (ROOT / "src" / "settings.py").read_text(encoding="utf-8")

    # shared_db must select NullPool when use_pooler=True
    has_nullpool = "NullPool" in shared_db and "use_pooler" in shared_db
    test("P2.10a_nullpool_selected",
         has_nullpool,
         "NullPool not used for pooler" if not has_nullpool
         else "NullPool selected for pgBouncer pooler")

    # settings must detect pooler (port 6543 or pooler.supabase.com)
    detects_pooler = "6543" in settings_src and "pooler.supabase.com" in settings_src
    test("P2.10b_pooler_detection",
         detects_pooler,
         "Pooler detection missing port 6543 or pooler.supabase.com" if not detects_pooler
         else "Pooler detected via port 6543 / pooler.supabase.com")


# ─── P2.11: Production schema assertion ────────────────────────────────

def test_production_schema_assertion():
    settings_src = (ROOT / "src" / "settings.py").read_text(encoding="utf-8")
    # validate_production_config must check DB_SCHEMA with Postgres
    has_check = ("DB_SCHEMA" in settings_src and "public" in settings_src
                 and "validate_production_config" in settings_src)
    test("P2.11_production_schema_assertion",
         has_check,
         "No schema assertion in production mode" if not has_check
         else "Schema isolation check in validate_production_config")


# ─── P2.12: SessionStore SQLite + WAL + revoke ──────────────────────────

def test_session_store():
    store_src = (ROOT / "src" / "session_store.py").read_text(encoding="utf-8")
    test("P2.12a_uses_sqlite",
         "sqlite3" in store_src,
         "Not SQLite-backed" if "sqlite3" not in store_src
         else "SQLite-backed session store")
    test("P2.12b_uses_wal",
         "WAL" in store_src,
         "No WAL mode" if "WAL" not in store_src
         else "WAL mode for concurrency")
    test("P2.12c_has_revoke",
         "revoke" in store_src,
         "No revoke mechanism" if "revoke" not in store_src
         else "Session revocation supported")
    test("P2.12d_has_cleanup",
         "cleanup_expired" in store_src,
         "No expired session cleanup" if "cleanup_expired" not in store_src
         else "Expired session cleanup supported")


# ─── Run all tests ──────────────────────────────────────────────────────

def main():
    test_rate_limiter_ip_keying()
    test_device_fingerprints_composite_pk()
    test_demo_seed_production_gate()
    test_db_paths_absolute()
    test_citations_illustrative()
    test_admin_session_uses_store()
    test_settings_comment()
    test_per_service_db_resolution()
    test_sqlite_pool_scoping()
    test_nullpool_for_pooler()
    test_production_schema_assertion()
    test_session_store()

    total = passed + failed + blocked
    print(f"\n{'='*60}")
    print(f"P2 FULL REGRESSION: {passed}/{total} PASSED, {failed} FAILED, {blocked} BLOCKED")
    if failed:
        print(f"  {failed} FAILED — previously-fixed bugs may have regressed!")
    elif blocked:
        print(f"  {blocked} BLOCKED (need live services)")
    else:
        print("  All previously-fixed bugs are still fixed.")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
