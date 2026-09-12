#!/usr/bin/env python3
"""Full pipeline smoke test against Supabase PostgreSQL."""
from supabase import create_client
import json, hashlib, time

import requests as _req

url = "https://aihweclhmdgryjahmamy.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFpaHdlY2xobWRncnlqYWhtYW15Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4NzY2MTQyMywiZXhwIjoyMTAzMjM3NDIzfQ.s7thpwjK7HO1JBLf6XENiqpmMBzZOjremzIJ-UN-_qI"

# Schema-aware helper using raw requests with Accept-Profile header
class SchemaClient:
    def __init__(self, schema):
        self.schema = schema
        self._base = f"{url}/rest/v1"
        self._headers = {
            "apikey": key, "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept-Profile": schema, "Content-Profile": schema,
        }
    def table(self, name):
        return _TableProxy(self._base, self._headers, name)

class _Result:
    def __init__(self, data=None, count=0):
        self.data = data or []
        self.count = count
    def execute(self):
        return self

class _TableProxy:
    def __init__(self, base, headers, name):
        self._url = f"{base}/{name}"
        self._headers = headers
    def insert(self, data):
        r = _req.post(self._url, headers={**self._headers, "Prefer": "return=representation"}, json=data)
        if r.status_code >= 400: raise Exception(r.text[:200])
        return _Result(r.json() if r.text else [])
    def upsert(self, data):
        h = {**self._headers, "Prefer": "return=representation,resolution=merge-duplicates"}
        r = _req.post(self._url, headers=h, json=data)
        if r.status_code >= 400: raise Exception(r.text[:200])
        return _Result(r.json() if r.text else [])
    def select(self, cols="*"):
        return _Query(self._url, self._headers, cols)
    def update(self, data):
        return _MutateQuery(self._url, self._headers, "PATCH", data)
    def delete(self):
        return _MutateQuery(self._url, self._headers, "DELETE")

class _Query:
    def __init__(self, url, headers, cols):
        self._url, self._headers, self._cols = url, headers, cols
    def eq(self, col, val):
        self._url += f"?{col}=eq.{val}"
        return self
    def order(self, col):
        self._url += f"&order={col}" if "?" in self._url else f"?order={col}"
        return self
    def limit(self, n):
        self._url += f"&limit={n}" if "?" in self._url else f"?limit={n}"
        return self
    def execute(self):
        r = _req.get(self._url, headers={**self._headers, "Prefer": "count=exact"})
        if r.status_code >= 400: raise Exception(r.text[:200])
        ct = r.headers.get("content-range", "")
        count = int(ct.split("/")[1]) if "/" in ct else len(r.json()) if r.text else 0
        return type("R", (), {"data": r.json() if r.text else [], "count": count})()

class _MutateQuery:
    def __init__(self, url, headers, method, data=None):
        self._url, self._headers, self._method, self._data = url, headers, method, data
    def eq(self, col, val):
        self._url += f"?{col}=eq.{val}"
        return self
    def execute(self):
        if self._method == "DELETE":
            r = _req.delete(self._url, headers=self._headers)
        else:
            r = _req.patch(self._url, headers=self._headers, json=self._data)
        if r.status_code >= 400: raise Exception(r.text[:200])
        return type("R", (), {"data": []})()

identity_c = SchemaClient("identity")
privacy_c = SchemaClient("privacy")
risk_c = SchemaClient("risk")
audit_c = SchemaClient("audit")

GENESIS = hashlib.sha256(b"PS-14 audit genesis v1").hexdigest()
RUN_ID = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
results = []

def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((name, ok))
    suffix = f" - {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")

print("=" * 70)
print("PS-14 FULL PIPELINE SMOKE TEST (Supabase PostgreSQL)")
print("=" * 70)

# === 1. IDENTITY SERVICE ===
print("\n--- 1. Identity Service ---")

user_id = f"smoke-{RUN_ID}"
fraud_id = f"FSMOKE{RUN_ID[:6].upper()}22"

try:
    identity_c.table("users").upsert({
        "user_id": user_id, "full_name": "Smoke Test User",
        "phone_encrypted": "enc_phone_b64", "email_encrypted": "enc_email_b64",
        "email_hash": hashlib.sha256(b"smoke@test.com").hexdigest(), "role": "user",
    }).execute()
    check("users: insert", True)
except Exception as e:
    check("users: insert", False, str(e)[:80])

try:
    identity_c.table("auth_credentials").upsert({
        "credential_id": f"cred-{RUN_ID}", "user_id": user_id,
        "credential_type": "password", "secret_hash": "hash123",
    }).execute()
    check("auth_credentials: insert", True)
except Exception as e:
    check("auth_credentials: insert", False, str(e)[:80])

try:
    identity_c.table("pseudonym_mapping").upsert({
        "fraud_id": fraud_id, "user_id": user_id,
        "rotation_group": 1, "access_log_required": True,
    }).execute()
    check("pseudonym_mapping: insert", True)
except Exception as e:
    check("pseudonym_mapping: insert", False, str(e)[:80])

try:
    identity_c.table("auth_credentials").upsert({
        "credential_id": f"fk-{RUN_ID}", "user_id": "NONEXISTENT",
        "credential_type": "password", "secret_hash": "x",
    }).execute()
    check("auth_credentials: FK enforced", False, "Should have failed")
except Exception:
    check("auth_credentials: FK enforced", True, "FK correctly rejected bad user_id")

# === 2. PRIVACY SERVICE ===
print("\n--- 2. Privacy Service ---")

try:
    privacy_c.table("fraud_profiles").upsert({
        "fraud_id": fraud_id, "avg_txn_amount_90d": 150.0, "txn_freq_7d": 5,
        "typical_txn_hours": json.dumps(list(range(8, 22))),
        "known_device_count": 2, "usual_locations": json.dumps(["US"]),
        "usual_recipients": json.dumps(["recv-001"]),
        "behavioral_baseline_vector": json.dumps({"txn_count": 10}),
    }).execute()
    check("fraud_profiles: insert", True)
except Exception as e:
    check("fraud_profiles: insert", False, str(e)[:80])

try:
    privacy_c.table("device_fingerprints").upsert({"device_hash": "smoke_d_a", "fraud_id": fraud_id}).execute()
    privacy_c.table("device_fingerprints").upsert({"device_hash": "smoke_d_b", "fraud_id": fraud_id}).execute()
    r = privacy_c.table("device_fingerprints").select("*").eq("fraud_id", fraud_id).execute()
    check("device_fingerprints: composite PK", len(r.data) == 2, f"{len(r.data)} rows")
except Exception as e:
    check("device_fingerprints: composite PK", False, str(e)[:80])

event_id = f"smoke-{RUN_ID}"
try:
    privacy_c.table("transaction_features").insert({
        "event_id": event_id, "fraud_id": fraud_id, "amount_ratio": 1.5,
        "txn_amount_bucket": "typical", "txn_freq_last_24h": 3,
        "txn_time_unusual": False, "new_device_flag": False,
        "unusual_location_flag": False, "unusual_recipient_flag": False,
        "failed_auth_count_24h": 0, "days_since_last_similar_txn": 2.5,
        "gradual_escalation_score": 0.1, "known_device_count": 2,
        "account_tenure_days": 60.0, "hour_of_day": 12, "is_weekend": False,
        "shared_device_accounts": 0, "shared_recipient_accounts": 0,
        "mule_ring_score": 0.0, "baseline_committed": False,
        "device_hash": "smoke_d_a", "location_id": "US", "recipient_id": "recv-001",
    }).execute()
    check("transaction_features: insert", True)
except Exception as e:
    check("transaction_features: insert", False, str(e)[:80])

try:
    r = privacy_c.table("transaction_features").select("*").eq("fraud_id", fraud_id).execute()
    check("transaction_features: indexed query", len(r.data) >= 1, f"{len(r.data)} rows")
except Exception as e:
    check("transaction_features: indexed query", False, str(e)[:80])

# === 3. RISK SERVICE ===
print("\n--- 3. Risk Service ---")

score_id = f"score-{RUN_ID}"
try:
    risk_c.table("risk_scores").upsert({
        "score_id": score_id, "fraud_id": fraud_id, "event_id": event_id,
        "risk_score": 75, "risk_band": "high",
        "reason_codes": json.dumps(["NEW_DEVICE", "BEHAVIOR_DEVIATION"]),
        "model_version": "v1.0", "ml_score": 0.82, "rule_score": 0.3, "degraded": False,
    }).execute()
    check("risk_scores: insert", True)
except Exception as e:
    check("risk_scores: insert", False, str(e)[:80])

try:
    risk_c.table("risk_scores").upsert({
        "score_id": f"fail-{RUN_ID}", "fraud_id": fraud_id, "event_id": f"fail-{RUN_ID}",
        "risk_score": 150, "risk_band": "high", "reason_codes": "[]",
        "model_version": "v1.0", "ml_score": 0.9, "rule_score": 0.5, "degraded": False,
    }).execute()
    check("risk_scores: CHECK constraint", False, "Should have rejected score=150")
except Exception:
    check("risk_scores: CHECK constraint", True, "Correctly rejected score > 100")

try:
    risk_c.table("verification_outcomes").upsert({
        "verification_id": f"verify-{RUN_ID}", "score_id": score_id,
        "fraud_id": fraud_id, "event_id": event_id,
        "outcome": "confirmed_fraud", "case_id": f"case-{RUN_ID}",
    }).execute()
    check("verification_outcomes: insert", True)
except Exception as e:
    check("verification_outcomes: insert", False, str(e)[:80])

try:
    risk_c.table("investigator_cases").upsert({
        "case_id": f"case-{RUN_ID}", "fraud_id": fraud_id, "event_id": event_id,
        "score_id": score_id, "status": "open", "confidence": "high",
        "priority": 0.85, "risk_score": 75,
        "reason_codes": json.dumps(["NEW_DEVICE"]), "notes": "Smoke test",
    }).execute()
    check("investigator_cases: insert", True)
except Exception as e:
    check("investigator_cases: insert", False, str(e)[:80])

# === 4. AUDIT SERVICE ===
print("\n--- 4. Audit Service ---")

prev_hash = GENESIS
audit_chain = [
    ("score_generated", {"event_id": event_id, "risk_score": 75}),
    ("feature_ingested", {"event_id": event_id}),
    ("verification_resolved", {"event_id": event_id, "outcome": "confirmed_fraud"}),
]

for i, (etype, payload) in enumerate(audit_chain):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    entry_hash = hashlib.sha256((prev_hash + body).encode()).hexdigest()
    try:
        audit_c.table("audit_events").insert({
            "event_id": f"smoke-{RUN_ID}-{i:03d}", "fraud_id": fraud_id,
            "event_type": etype, "prev_hash": prev_hash,
            "entry_hash": entry_hash, "payload_summary": json.dumps(payload),
        }).execute()
        check(f"audit_events: insert ({etype})", True)
        prev_hash = entry_hash
    except Exception as e:
        check(f"audit_events: insert ({etype})", False, str(e)[:80])

# Chain integrity
try:
    r = audit_c.table("audit_events").select("*").eq("fraud_id", fraud_id).order("seq").execute()
    rows = r.data
    chain_ok = True
    for row in rows:
        body = json.dumps(json.loads(row["payload_summary"]), sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256((row["prev_hash"] + body).encode()).hexdigest()
        if row["entry_hash"] != expected:
            chain_ok = False
            break
    check("audit_events: chain integrity", chain_ok, f"{len(rows)} entries")
except Exception as e:
    check("audit_events: chain integrity", False, str(e)[:80])

try:
    audit_c.table("audit_events").update({"payload_summary": "x"}).eq("event_id", f"smoke-{RUN_ID}-000").execute()
    check("audit_events: append-only (UPDATE)", False, "Not blocked")
except Exception:
    check("audit_events: append-only (UPDATE)", True, "UPDATE correctly blocked")

try:
    audit_c.table("audit_events").delete().eq("event_id", f"smoke-{RUN_ID}-000").execute()
    check("audit_events: append-only (DELETE)", False, "Not blocked")
except Exception:
    check("audit_events: append-only (DELETE)", True, "DELETE correctly blocked")

try:
    audit_c.table("audit_access_log").insert({
        "actor": "smoke-test", "action": "chain_verify",
        "query_summary": "Verified chain",
    }).execute()
    check("audit_access_log: insert", True)
except Exception as e:
    check("audit_access_log: insert", False, str(e)[:80])

# === 5. CROSS-SERVICE FK ===
print("\n--- 5. Cross-Service FK Enforcement ---")

try:
    risk_c.table("verification_outcomes").upsert({
        "verification_id": f"fk-fail-{RUN_ID}", "score_id": "NONEXISTENT",
        "fraud_id": fraud_id, "event_id": event_id,
        "outcome": "test", "case_id": "test",
    }).execute()
    check("FK: verification -> risk_scores", False, "Should have failed")
except Exception:
    check("FK: verification -> risk_scores", True, "FK correctly enforced")

try:
    privacy_c.table("transaction_features").insert({
        "event_id": f"fk-fail-{RUN_ID}", "fraud_id": "FNONEXISTENT000022",
        "amount_ratio": 1.0, "txn_amount_bucket": "typical",
        "txn_freq_last_24h": 0, "txn_time_unusual": False,
        "new_device_flag": False, "unusual_location_flag": False,
        "unusual_recipient_flag": False, "failed_auth_count_24h": 0,
        "days_since_last_similar_txn": 0, "gradual_escalation_score": 0,
        "known_device_count": 0, "account_tenure_days": 0,
        "hour_of_day": 12, "is_weekend": False,
        "shared_device_accounts": 0, "shared_recipient_accounts": 0,
        "mule_ring_score": 0,
    }).execute()
    check("FK: transaction_features -> fraud_profiles", False, "Should have failed")
except Exception:
    check("FK: transaction_features -> fraud_profiles", True, "FK correctly enforced")

# === 6. POSTGRES TYPING ===
print("\n--- 6. Postgres Strict Typing ---")

try:
    privacy_c.table("transaction_features").insert({
        "event_id": f"type-{RUN_ID}", "fraud_id": fraud_id,
        "amount_ratio": 1.0, "txn_amount_bucket": "typical",
        "txn_freq_last_24h": 0, "txn_time_unusual": 0,
        "new_device_flag": 0, "unusual_location_flag": 0,
        "unusual_recipient_flag": 0, "failed_auth_count_24h": 0,
        "days_since_last_similar_txn": 0, "gradual_escalation_score": 0,
        "known_device_count": 0, "account_tenure_days": 0,
        "hour_of_day": 12, "is_weekend": 0,
        "shared_device_accounts": 0, "shared_recipient_accounts": 0,
        "mule_ring_score": 0, "baseline_committed": 0,
    }).execute()
    check("BOOLEAN coercion (int->bool)", True, "Postgres accepted")
except Exception as e:
    check("BOOLEAN coercion (int->bool)", False, str(e)[:80])

# === SUMMARY ===
print("\n" + "=" * 70)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"RESULTS: {passed}/{len(results)} PASSED, {failed} FAILED")
if failed:
    print("\nFAILURES:")
    for name, ok in results:
        if not ok:
            print(f"  - {name}")
print("=" * 70)
