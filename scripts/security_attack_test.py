#!/usr/bin/env python3
"""Security Attack Test - Attempt to break into PS-14 databases and find vulnerabilities."""
import json
import sys
import sqlite3
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "db"

failures = []
findings = []

def check(name, passed, detail=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}" + (f" ({detail})" if detail else ""))
    if not passed:
        failures.append(name)
    return passed

def finding(severity, name, detail, fix=""):
    findings.append({"severity": severity, "name": name, "detail": detail, "fix": fix})
    print(f"\n  🔴 {severity}: {name}")
    print(f"     {detail}")
    if fix:
        print(f"     FIX: {fix}")

print("=" * 70)
print("PS-14 SECURITY ATTACK TEST")
print("=" * 70)

# 1. Direct database file access
print("\n[1] DIRECT DATABASE FILE ACCESS")
print("-" * 40)

for db_name in ["identity.db", "features.db", "risk.db", "audit.db"]:
    db_path = DB_DIR / db_name
    if db_path.exists():
        size = db_path.stat().st_size
        print(f"  Found: {db_name} ({size:,} bytes)")
        
        # Check file permissions (should not be world-readable on production)
        # On Windows, we check if the file is accessible
        
        # Try to read the database directly
        try:
            conn = sqlite3.connect(str(db_path))
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            print(f"    Tables: {tables}")
            
            # Check for PII in each table
            pii_keywords = {"email", "name", "phone", "address", "ssn", "password", "ip_address"}
            for table in tables:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
                pii_found = set(cols) & pii_keywords
                if pii_found:
                    finding("HIGH", f"PII in {db_name}.{table}", f"Columns with PII: {pii_found}")
            
            # Check for raw amounts
            for table in tables:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
                raw_amounts = [c for c in cols if "amount" in c.lower() and "ratio" not in c.lower() and "bucket" not in c.lower()]
                if raw_amounts:
                    finding("MEDIUM", f"Raw amounts in {db_name}.{table}", f"Columns: {raw_amounts}")
            
            conn.close()
        except Exception as e:
            print(f"    Error reading: {e}")

# 2. Check if identity.db has plaintext PII
print("\n[2] IDENTITY DATABASE PII CHECK")
print("-" * 40)

identity_path = DB_DIR / "identity.db"
if identity_path.exists():
    conn = sqlite3.connect(str(identity_path))
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    
    for table in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        print(f"  {table}: {cols}")
        
        # Check if email/phone are encrypted
        if "email_encrypted" in cols:
            # Sample a few rows to check if they're actually encrypted
            try:
                rows = conn.execute(f"SELECT email_encrypted FROM {table} LIMIT 3").fetchall()
                for (val,) in rows:
                    if val and isinstance(val, bytes) and len(val) > 50:
                        print(f"    ✓ email_encrypted appears encrypted (bytes, len={len(val)})")
                    elif val and isinstance(val, str) and "@" in val:
                        finding("CRITICAL", f"Plaintext email in {table}", f"Value: {val[:50]}...")
                    else:
                        print(f"    ? email_encrypted value type: {type(val)}")
            except (sqlite3.OperationalError, sqlite3.InterfaceError):
                pass

        if "phone_encrypted" in cols:
            try:
                rows = conn.execute(f"SELECT phone_encrypted FROM {table} LIMIT 3").fetchall()
                for (val,) in rows:
                    if val and isinstance(val, bytes) and len(val) > 20:
                        print(f"    ✓ phone_encrypted appears encrypted (bytes, len={len(val)})")
                    elif val and isinstance(val, str) and val.isdigit():
                        finding("CRITICAL", f"Plaintext phone in {table}", f"Value: {val}")
            except (sqlite3.OperationalError, sqlite3.InterfaceError):
                pass
    
    conn.close()

# 3. Check audit.db for PII leakage
print("\n[3] AUDIT DATABASE PII LEAKAGE")
print("-" * 40)

audit_path = DB_DIR / "audit.db"
if audit_path.exists():
    conn = sqlite3.connect(str(audit_path))
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    
    for table in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        print(f"  {table}: {cols}")
        
        # Check payload_summary for PII
        if "payload_summary" in cols:
            try:
                rows = conn.execute(f"SELECT payload_summary FROM {table} LIMIT 20").fetchall()
                import re
                email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
                phone_pattern = re.compile(r'\b\d{10}\b')
                name_pattern = re.compile(r'"(first_name|last_name|full_name)"')
                
                for (payload,) in rows:
                    if payload:
                        if email_pattern.search(payload):
                            finding("CRITICAL", f"Email in audit payload", f"Payload: {payload[:100]}")
                        if phone_pattern.search(payload):
                            finding("HIGH", f"Phone pattern in audit payload", f"Payload: {payload[:100]}")
                        if name_pattern.search(payload):
                            finding("HIGH", f"Name field in audit payload", f"Payload: {payload[:100]}")
            except Exception as e:
                print(f"    Error checking payloads: {e}")
    
    conn.close()

# 4. Check admin.json for hardcoded credentials
print("\n[4] ADMIN CREDENTIALS CHECK")
print("-" * 40)

admin_path = DB_DIR / "admin.json"
if admin_path.exists():
    with open(admin_path) as f:
        admin_data = json.load(f)
    
    print(f"  Admin fields: {list(admin_data.keys())}")
    
    # Check for weak password hash
    if "pass_hash" in admin_data:
        ph = admin_data["pass_hash"]
        if isinstance(ph, str):
            # Argon2 hashes start with $argon2id$ and are ~90+ chars
            # scrypt hashes are base64-encoded ~44 chars (32 bytes)
            import base64
            try:
                decoded = base64.b64decode(ph)
                if len(decoded) >= 32:  # scrypt output is 32 bytes
                    print(f"  ✓ Password hash appears secure (scrypt, decoded={len(decoded)} bytes)")
                else:
                    finding("HIGH", "Weak password hash", f"Decoded length: {len(decoded)} (should be 32+)")
            except Exception:
                if ph.startswith("$") or len(ph) > 50:
                    print(f"  ✓ Password hash appears secure (Argon2/scrypt, len={len(ph)})")
                else:
                    finding("HIGH", "Weak password hash", f"Hash length: {len(ph)} (should be scrypt/argon2)")
    
    # Check for hardcoded admin password in .env
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            env_content = f.read()
        if "ADMIN_PASS=" in env_content:
            admin_pass_line = [l for l in env_content.splitlines() if l.startswith("ADMIN_PASS=")][0]
            admin_pass = admin_pass_line.split("=", 1)[1]
            if len(admin_pass) < 12:
                finding("HIGH", "Weak admin password in .env", f"Password length: {len(admin_pass)} (should be 16+)")
            print(f"  ADMIN_PASS length: {len(admin_pass)} chars")

# 5. Check .env for exposed secrets
print("\n[5] SECRETS EXPOSURE CHECK")
print("-" * 40)

env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        env_lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
    
    sensitive_keys = {"JWT_SECRET", "PII_ENCRYPTION_KEY", "EXPORT_SIGNING_KEY", "INTERNAL_TOKEN", "COMPLIANCE_TOKEN", "ADMIN_PASS"}
    
    for line in env_lines:
        if "=" in line:
            key = line.split("=", 1)[0]
            value = line.split("=", 1)[1]
            
            if key in sensitive_keys:
                if len(value) < 16:
                    finding("HIGH", f"Weak secret: {key}", f"Length: {len(value)} (should be 32+)")
                else:
                    print(f"  ✓ {key}: {len(value)} chars (OK)")
                
                # Check if it's a known weak value
                weak_values = {"dev-jwt-secret-replace-in-production", "secret", "password", "changeme"}
                if value.lower() in weak_values:
                    finding("CRITICAL", f"Known weak secret: {key}", f"Value: {value}")

# 6. Check for SQL injection in admin DB query
print("\n[6] SQL INJECTION VULNERABILITY CHECK")
print("-" * 40)

# Check the admin DB query endpoint
admin_query_path = ROOT / "src" / "front_service" / "main.py"
if admin_query_path.exists():
    with open(admin_query_path) as f:
        content = f.read()
    
    # Check for raw SQL execution
    if "conn.execute(req.query" in content:
        # Check if there are protections
        has_admin_auth = "_require_admin" in content
        has_totp = "totp" in content.lower()
        has_select_only = "only SELECT" in content
        has_blocked_keywords = "_BLOCKED" in content
        
        protections = []
        if has_admin_auth: protections.append("admin auth")
        if has_totp: protections.append("TOTP 2FA")
        if has_select_only: protections.append("SELECT only")
        if has_blocked_keywords: protections.append("keyword filter")
        
        if len(protections) >= 3:
            print(f"  ⚠ SQL endpoint has protections: {', '.join(protections)}")
            finding("MEDIUM", "Raw SQL execution (with protections)", 
                    f"Endpoint requires: {', '.join(protections)}",
                    "Consider implementing a query whitelist for additional defense in depth")
        else:
            finding("CRITICAL", "Raw SQL execution in admin endpoint", 
                    "Endpoint /admin/db-query executes user-supplied SQL directly",
                    "Implement parameterized queries or whitelist allowed queries only")
    
    # Check for SQL injection in table listing
    if 'SELECT COUNT(*) FROM' in content:
        finding("MEDIUM", "String concatenation in SQL", 
                "Table name is concatenated into SQL query",
                "Use parameterized queries for all SQL")

# 7. Check for path traversal
print("\n[7] PATH TRAVERSAL CHECK")
print("-" * 40)

# Check if any endpoint serves files based on user input
if admin_query_path.exists():
    with open(admin_query_path) as f:
        content = f.read()
    
    if "FileResponse" in content:
        # Check if FileResponse uses user input
        import re
        file_response_pattern = re.compile(r'FileResponse\([^)]*\)')
        matches = file_response_pattern.findall(content)
        for match in matches:
            if "req." in match or "param" in match:
                finding("HIGH", "FileResponse with user input", f"Match: {match}")

# 8. Check for authentication bypass
print("\n[8] AUTHENTICATION BYPASS CHECK")
print("-" * 40)

# Check if internal endpoints require auth
internal_endpoints = ["/internal/evaluate", "/internal/attribution"]
for endpoint in internal_endpoints:
    if endpoint in content:
        # Check if it requires X-Internal-Token
        if "X-Internal-Token" in content:
            print(f"  ✓ {endpoint} requires X-Internal-Token")
        else:
            finding("CRITICAL", f"No auth on {endpoint}", "Internal endpoint does not require authentication")

# 9. Check for data exposure in API responses
print("\n[9] API RESPONSE DATA EXPOSURE")
print("-" * 40)

# Check if PII is returned in any API response
if admin_query_path.exists():
    with open(admin_query_path) as f:
        content = f.read()
    
    # Check for PII masking
    if "_PII_MASK_COLUMNS" in content:
        print("  ✓ PII masking implemented for admin queries")
    else:
        finding("HIGH", "No PII masking in admin queries", "Sensitive columns may be exposed in API responses")

# 10. Check for weak cryptography
print("\n[10] CRYPTOGRAPHY CHECK")
print("-" * 40)

# Check encryption keys
if env_path.exists():
    with open(env_path) as f:
        env_content = f.read()
    
    if "PII_ENCRYPTION_KEY" in env_content:
        key_line = [l for l in env_content.splitlines() if l.startswith("PII_ENCRYPTION_KEY=")][0]
        key = key_line.split("=", 1)[1]
        if len(key) < 32:
            finding("MEDIUM", "Short encryption key", f"Key length: {len(key)} (should be 32+)")
        else:
            print(f"  ✓ PII_ENCRYPTION_KEY: {len(key)} chars (OK)")

# Summary
print("\n" + "=" * 70)
print("SECURITY ATTACK TEST SUMMARY")
print("=" * 70)

print(f"\nFindings: {len(findings)}")
for f in findings:
    print(f"  [{f['severity']}] {f['name']}")

print(f"\nFailures: {len(failures)}")
for f in failures:
    print(f"  - {f}")

# Save report
report = {
    "findings": findings,
    "failures": failures,
    "total_findings": len(findings),
    "critical": len([f for f in findings if f["severity"] == "CRITICAL"]),
    "high": len([f for f in findings if f["severity"] == "HIGH"]),
    "medium": len([f for f in findings if f["severity"] == "MEDIUM"]),
}

with open(ROOT / "data" / "security_attack_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to data/security_attack_report.json")

sys.exit(1 if failures else 0)
