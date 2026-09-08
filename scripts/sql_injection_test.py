#!/usr/bin/env python3
"""SQL injection regression tests for the admin DB query endpoint.

Tests that the allowlist-only query system properly rejects all forms of
SQL injection.  The correct result for every unauthorized query is rejection.
"""

import sys
import os
import json
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("SKIP: requests not installed")
    sys.exit(0)

BASE = "http://127.0.0.1:8000"
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")

# Read ADMIN_PASS from .env if not in environment
def _load_admin_pass() -> str:
    p = os.environ.get("ADMIN_PASS")
    if p:
        return p
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ADMIN_PASS="):
                return line.split("=", 1)[1].strip()
    return "admin12345"

ADMIN_PASS = _load_admin_pass()

passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        errors.append(name)


def get_admin_token():
    """Login as admin and get a JWT token (front service admin login)."""
    try:
        r = requests.post(f"{BASE}/admin/login", json={
            "username": ADMIN_USER,
            "passphrase": ADMIN_PASS,
        }, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("token") or data.get("session_token") or data.get("access_token")
    except Exception:
        pass
    return None


def admin_db_query(token, db_name, query, passphrase=None, limit=50):
    if passphrase is None:
        passphrase = ADMIN_PASS
    """Execute a query through the admin DB query endpoint."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "db_name": db_name,
        "passphrase": passphrase,
        "query": query,
        "limit": limit,
    }
    try:
        r = requests.post(f"{BASE}/admin/db-query", json=payload,
                          headers=headers, timeout=10)
        return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as e:
        return 0, {"error": str(e)}


def run_tests():
    global passed, failed, errors
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("SQL INJECTION REGRESSION TESTS")
    print("=" * 70)

    token = get_admin_token()
    if not token:
        print("\n  ✗ Admin login failed — SQL injection tests cannot run")
        print("  (Start the front service on port 8000 first)")
        return 1  # Non-zero: service required but unavailable

    # ---- UNION injection ----
    print("\n[1] UNION injection")
    payloads = [
        "recent_audit UNION SELECT 1,2,3,4,5",
        "recent_audit UNION SELECT * FROM users",
        "recent_audit UNION ALL SELECT user_id, email, phone, 1, 1 FROM users",
        "recent_scores UNION SELECT 1,2,3,4,5,6,7",
    ]
    for p in payloads:
        code, resp = admin_db_query(token, "risk", p)
        test(f"UNION injection rejected: {p[:50]}...",
             code == 400 or code == 401,
             f"got {code}: {resp.get('detail', '')[:80]}")

    # ---- Comment injection ----
    print("\n[2] Comment injection")
    payloads = [
        "recent_audit; --",
        "recent_audit /* comment */",
        "recent_audit -- SELECT * FROM users",
        "recent_audit /**/ UNION SELECT 1,2,3,4,5",
    ]
    for p in payloads:
        code, resp = admin_db_query(token, "risk", p)
        test(f"Comment injection rejected: {p[:50]}",
             code == 400,
             f"got {code}: {resp.get('detail', '')[:80]}")

    # ---- Quoted identifiers ----
    print("\n[3] Quoted identifier injection")
    payloads = [
        'recent_audit" UNION SELECT * FROM users --',
        "recent_audit' UNION SELECT * FROM users --",
        'row_count:users" OR "1"="1',
        "row_count:users' OR '1'='1",
    ]
    for p in payloads:
        code, resp = admin_db_query(token, "features", p)
        test(f"Quoted identifier rejected: {p[:50]}",
             code == 400,
             f"got {code}: {resp.get('detail', '')[:80]}")

    # ---- Nested SELECT ----
    print("\n[4] Nested SELECT injection")
    payloads = [
        "recent_audit WHERE 1=(SELECT 1)",
        "recent_audit WHERE fraud_id IN (SELECT fraud_id FROM users)",
        "recent_audit HAVING 1=1",
        "recent_audit GROUP BY 1 HAVING 1=1",
    ]
    for p in payloads:
        code, resp = admin_db_query(token, "risk", p)
        test(f"Nested SELECT rejected: {p[:50]}",
             code == 400,
             f"got {code}: {resp.get('detail', '')[:80]}")

    # ---- PRAGMA attempts ----
    print("\n[5] PRAGMA attempts")
    payloads = [
        "PRAGMA table_info(users)",
        "PRAGMA database_list",
        "PRAGMA compile_options",
        "PRAGMA integrity_check",
    ]
    for p in payloads:
        code, resp = admin_db_query(token, "identity", p)
        test(f"PRAGMA rejected: {p}",
             code == 400,
             f"got {code}: {resp.get('detail', '')[:80]}")

    # ---- ATTACH attempts ----
    print("\n[6] ATTACH/DETACH attempts")
    payloads = [
        "ATTACH DATABASE 'identity.db' AS idb",
        "DETACH DATABASE idb",
        "ATTACH 'db/identity.db'",
    ]
    for p in payloads:
        code, resp = admin_db_query(token, "features", p)
        test(f"ATTACH rejected: {p[:50]}",
             code == 400,
             f"got {code}: {resp.get('detail', '')[:80]}")

    # ---- Subquery attempts ----
    print("\n[7] Subquery attempts")
    payloads = [
        "(SELECT * FROM users)",
        "recent_audit UNION SELECT * FROM (SELECT * FROM users)",
        "recent_audit WHERE fraud_id = (SELECT fraud_id FROM pseudonym_mapping LIMIT 1)",
    ]
    for p in payloads:
        code, resp = admin_db_query(token, "risk", p)
        test(f"Subquery rejected: {p[:60]}",
             code == 400,
             f"got {code}: {resp.get('detail', '')[:80]}")

    # ---- Boolean manipulation ----
    print("\n[8] Boolean manipulation")
    payloads = [
        "recent_audit OR 1=1",
        "recent_audit AND 1=1",
        "recent_audit WHERE 1=1",
        "1=1 OR recent_audit",
    ]
    for p in payloads:
        code, resp = admin_db_query(token, "risk", p)
        test(f"Boolean manipulation rejected: {p[:50]}",
             code == 400,
             f"got {code}: {resp.get('detail', '')[:80]}")

    # ---- Malformed parameters ----
    print("\n[9] Malformed parameters")
    code, resp = admin_db_query(token, "", "recent_audit")
    test("Empty db_name rejected",
         code in (400, 422),
         f"got {code}")

    code, resp = admin_db_query(token, "nonexistent_db", "recent_audit")
    test("Unknown db_name rejected",
         code == 400,
         f"got {code}")

    code, resp = admin_db_query(token, "risk", "")
    test("Empty query rejected",
         code in (400, 422),
         f"got {code}")

    code, resp = admin_db_query(token, "risk", "DROP TABLE users")
    test("DROP statement rejected",
         code == 400,
         f"got {code}")

    code, resp = admin_db_query(token, "risk", "DELETE FROM risk_scores")
    test("DELETE statement rejected",
         code == 400,
         f"got {code}")

    code, resp = admin_db_query(token, "risk", "INSERT INTO risk_scores VALUES (1)")
    test("INSERT statement rejected",
         code == 400,
         f"got {code}")

    code, resp = admin_db_query(token, "risk", "UPDATE risk_scores SET risk_score = 0")
    test("UPDATE statement rejected",
         code == 400,
         f"got {code}")

    code, resp = admin_db_query(token, "risk", "VACUUM")
    test("VACUUM rejected",
         code == 400,
         f"got {code}")

    code, resp = admin_db_query(token, "risk", "EXPLAIN SELECT * FROM risk_scores")
    test("EXPLAIN rejected",
         code == 400,
         f"got {code}")

    # ---- row_count injection ----
    print("\n[10] row_count table name injection")
    table_payloads = [
        "row_count:users",
        "row_count:pseudonym_mapping",
        "row_count:auth_credentials",
        "row_count:sqlite_master",
        "row_count:users; DROP TABLE users",
        "row_count:' OR '1'='1",
    ]
    for p in table_payloads:
        code, resp = admin_db_query(token, "identity", p)
        table = p.split(":")[1] if ":" in p else p
        test(f"row_count:{table} rejected (PII/sensitive table)",
             code == 400,
             f"got {code}: {resp.get('detail', '')[:80]}")

    # ---- row_count allowed tables ----
    print("\n[11] row_count allowed tables (should work)")
    allowed = [
        ("risk", "row_count:risk_scores"),
        ("features", "row_count:transaction_features"),
        ("audit", "row_count:audit_events"),
    ]
    for db, p in allowed:
        table = p.split(":")[1]
        code, resp = admin_db_query(token, db, p)
        test(f"row_count:{table} allowed",
             code == 200,
             f"got {code}")

    # ---- Wrong passphrase ----
    print("\n[12] Authentication tests")
    code, resp = admin_db_query(token, "risk", "recent_audit", passphrase="wrong_password")
    test("Wrong passphrase rejected",
         code == 401,
         f"got {code}")

    # ---- No auth ----
    headers = {}
    try:
        r = requests.post(f"{BASE}/admin/db-query", json={
            "db_name": "risk",
            "passphrase": "admin",
            "query": "recent_audit",
        }, headers=headers, timeout=5)
        test("No auth token rejected",
             r.status_code == 401 or r.status_code == 403)
    except Exception as e:
        test("No auth token rejected", False, str(e))

    # ---- Summary ----
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e}")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
