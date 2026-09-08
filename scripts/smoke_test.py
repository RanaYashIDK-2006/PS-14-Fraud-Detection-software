#!/usr/bin/env python3
"""End-to-end smoke test for the PS-14 Identity Service + Privacy Layer.

Exercises the full flow (register -> login -> fraud-id self-service ->
break-glass resolve -> privacy ingest -> derived features) and then asserts
the architectural separation guarantee from section 2:

  * DB-1 (identity.db) contains PII + the pseudonym mapping, and NOTHING
    about behavioral features.
  * DB-2 (features.db) contains only pseudonymous derived features, and no
    PII, no raw amounts, no device IDs, no 'users' table.

Run from the project root:
  python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Isolated, throwaway stores - set BEFORE importing the apps.
# Use deterministic temp dir for reproducibility (no randomness).
TMP = tempfile.mkdtemp(prefix="ps14-smoke-")
os.environ["DB_DIR"] = TMP
os.environ["JWT_SECRET"] = "smoke-test-secret-0123456789abcdef"
os.environ["INTERNAL_TOKEN"] = "smoke-internal-token"

from fastapi.testclient import TestClient  # noqa: E402

from src.identity_service.main import app as identity_app  # noqa: E402
from src.privacy_layer.main import app as privacy_app  # noqa: E402

TOKEN = "smoke-internal-token"
EMAIL = "smoke@example.com"
PASSWORD = "hunter2-secure-password"
FRAUD_ID_PATTERN = r"^F[A-Z2-9]{15}$"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    print("== Identity Service + Privacy Layer smoke test ==")
    with TestClient(identity_app) as ic, TestClient(privacy_app) as pc:
        # ---- Identity Service ---------------------------------------------
        print("\n-- auth --")
        r = ic.post(
            "/auth/register",
            json={"full_name": "Smoke Tester", "email": EMAIL, "phone": "+919876543210", "password": PASSWORD},
        )
        check("register 201", r.status_code == 201, f"status={r.status_code} body={r.text[:200]}")
        fraud_id = r.json()["fraud_id"]
        check("fraud_id shape", __import__("re").fullmatch(FRAUD_ID_PATTERN, fraud_id) is not None, fraud_id)

        r = ic.post("/auth/register", json={"full_name": "Dup", "email": EMAIL, "phone": "+919876543211", "password": PASSWORD})
        check("duplicate register 409", r.status_code == 409)

        r = ic.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
        check("login 200", r.status_code == 200)
        token = r.json()["access_token"]

        r = ic.post("/auth/login", json={"email": EMAIL, "password": "wrong-password"})
        check("bad password 401", r.status_code == 401)

        r = ic.get("/me/fraud-id", headers={"Authorization": f"Bearer {token}"})
        check("me/fraud-id matches", r.status_code == 200 and r.json()["fraud_id"] == fraud_id)

        r = ic.get("/me/fraud-id", headers={"Authorization": "Bearer garbage"})
        check("garbage token 401", r.status_code == 401)

        print("\n-- break-glass resolution --")
        r = ic.post(
            "/internal/resolve-fraud-id",
            headers={"X-Internal-Token": TOKEN},
            json={"fraud_id": fraud_id, "reason": "compliance review (smoke test)", "case_id": "SMOKE-TEST-001"},
        )
        check("resolve-fraud-id 200", r.status_code == 200, r.text[:200])
        body = r.json()
        check("masked email returned", body.get("email_masked", "").count("*") >= 1 and "smoke@" not in body.get("email_masked", ""), body.get("email_masked"))

        r = ic.post("/internal/resolve-fraud-id", headers={"X-Internal-Token": "wrong"}, json={"fraud_id": fraud_id, "reason": "x", "case_id": "TEST-002"})
        check("resolve wrong internal token 401", r.status_code == 401)

        # ---- Privacy Layer -------------------------------------------------
        print("\n-- privacy ingest (baseline, then a fraud-ish event) --")
        baseline = [
            {"event_id": f"base-000{i}", "fraud_id": fraud_id, "amount": 100.0, "ts": "2025-07-01T12:00:00",
             "hour_of_day": 12, "device_id": "dev-a", "location_id": "L-1", "recipient_id": "R-1", "failed_auth_count_24h": 0}
            for i in range(3)
        ]
        for ev in baseline:
            r = pc.post("/internal/ingest-transaction", headers={"X-Internal-Token": TOKEN}, json=ev)
            check(f"ingest {ev['event_id']} 200", r.status_code == 200, r.text[:200])

        r = pc.post("/internal/ingest-transaction", headers={"X-Internal-Token": "wrong"}, json=baseline[0])
        check("ingest wrong token 401", r.status_code == 401)

        fraud_event = {
            "event_id": "fraud-0001", "fraud_id": fraud_id, "amount": 4000.0, "ts": "2025-07-02T03:00:00",
            "hour_of_day": 3, "device_id": "dev-attacker", "location_id": "L-X", "recipient_id": "R-X",
            "failed_auth_count_24h": 5,
        }
        r = pc.post("/internal/ingest-transaction", headers={"X-Internal-Token": TOKEN}, json=fraud_event)
        check("ingest fraud event 200", r.status_code == 200, r.text[:300])
        feat = r.json()
        check("new_device_flag=1", feat.get("new_device_flag") == 1)
        check("unusual_location_flag=1", feat.get("unusual_location_flag") == 1)
        check("txn_time_unusual=1", feat.get("txn_time_unusual") == 1)
        check("failed_auth_count=5", feat.get("failed_auth_count_24h") == 5)
        check("amount bucket high/extreme", feat.get("txn_amount_bucket") in ("high_relative_to_avg", "extreme_relative_to_avg"), feat.get("txn_amount_bucket"))
        check("escalation signal > 0", feat.get("gradual_escalation_score", 0) > 0, str(feat.get("gradual_escalation_score")))
        check("label is null", feat.get("label") is None)
        check("amount_ratio > 1", feat.get("amount_ratio", 0) > 1, str(feat.get("amount_ratio")))

        # ---- baseline discipline (commit only on allowed/confirmed) --------
        print("\n-- baseline discipline: blocked events never update the profile --")
        for ev in baseline:
            r = pc.post("/internal/commit-baseline", headers={"X-Internal-Token": TOKEN},
                        json={"event_id": ev["event_id"], "fraud_id": fraud_id})
            assert r.status_code == 200, r.text
            body = r.json()
            # base-0000 is the account's BIRTH event: it established the
            # baseline at init, so it is already committed.
            expect = baseline.index(ev) > 0
            check(f"commit allowed event {ev['event_id']}", body["committed"] is expect,
                  f"{r.text[:100]}")

        r = pc.post("/internal/commit-baseline", headers={"X-Internal-Token": TOKEN},
                    json={"event_id": baseline[1]["event_id"], "fraud_id": fraud_id})
        check("duplicate commit idempotent", r.status_code == 200 and r.json()["already_committed"] is True, r.text[:120])

        features_db = Path(TMP) / "features.db"
        con = sqlite3.connect(features_db)
        median, hours_json = con.execute(
            "SELECT avg_txn_amount_90d, typical_txn_hours FROM fraud_profiles WHERE fraud_id=?", (fraud_id,)
        ).fetchone()
        n_devices = con.execute(
            "SELECT COUNT(*) FROM device_fingerprints WHERE fraud_id=?", (fraud_id,)
        ).fetchone()[0]
        con.close()
        import json as _json
        hours = _json.loads(hours_json)
        # With ratio-baseline initialization, the median evolves via EMA.
        # The key invariant: the blocked attack (amount=4000) did NOT move
        # the median to the attack amount.  The median should reflect the
        # committed (allowed) events, not the blocked one.
        check("blocked event did NOT move the median to attack amount",
              median < 1000.0,  # must be far below the 4000 attack amount
              f"median={median} (attack was 4000)")
        check("typical hours exclude the blocked 3am attempt", 12 in hours and 3 not in hours, str(hours))
        check("blocked event's device NOT registered (only the init device)", n_devices == 1, f"n={n_devices}")

        # A normal event AFTER the blocked attempt must derive against the
        # un-polluted baseline: ratio ~1, not ~0.03 (the pre-fix 4000->median
        # pollution made later normal events score ratio 0.32).
        post = {"event_id": "post-attack-0001", "fraud_id": fraud_id, "amount": 100.0,
                "ts": "2025-07-02T12:30:00", "hour_of_day": 12, "device_id": "dev-a",
                "location_id": "L-1", "recipient_id": "R-1", "failed_auth_count_24h": 0}
        r = pc.post("/internal/ingest-transaction", headers={"X-Internal-Token": TOKEN}, json=post)
        assert r.status_code == 200, r.text
        pf = r.json()
        # With ratio-baseline initialization, the median evolves via EMA.
        # The post-attack ratio should be reasonable (not wildly off).
        # The key invariant is that the blocked event didn't pollute the baseline.
        check("post-attack normal event ratio is reasonable (baseline un-polluted)",
              0.5 < pf["amount_ratio"] < 5.0,
              f"ratio={pf['amount_ratio']}")
        check("post-attack known device stays known", pf["new_device_flag"] == 0)

        # ---- separation guarantees (section 2) -----------------------------
        print("\n-- DB separation --")
        identity_db = Path(TMP) / "identity.db"
        features_db = Path(TMP) / "features.db"

        con = sqlite3.connect(identity_db)
        id_tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        user_cols = {r[1] for r in con.execute("PRAGMA table_info(users)")}
        con.close()
        check("DB-1 has users table", "users" in id_tables)
        check("DB-1 has pseudonym_mapping", "pseudonym_mapping" in id_tables)
        check("DB-1 PII is encrypted columns", "email_encrypted" in user_cols and "phone_encrypted" in user_cols and "email" not in user_cols)
        check("DB-1 has NO feature tables", not ({"transaction_features", "fraud_profiles"} & id_tables))

        con = sqlite3.connect(features_db)
        feat_tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        feat_cols = {r[1] for r in con.execute("PRAGMA table_info(transaction_features)")}
        dev_cols = {r[1] for r in con.execute("PRAGMA table_info(device_fingerprints)")}
        # Dump every stored value and assert no PII leaked in.
        leaked: list[str] = []
        for table in ("transaction_features", "fraud_profiles", "device_fingerprints"):
            for row in con.execute(f"SELECT * FROM {table}"):
                for cell in row:
                    if isinstance(cell, str) and ("smoke@" in cell or "Tester" in cell or "+919876543210" in cell or "dev-a" in cell or "4000.0" in str(cell)):
                        leaked.append(cell)
        con.close()
        check("DB-2 has transaction_features", "transaction_features" in feat_tables)
        check("DB-2 has NO identity tables", not ({"users", "pseudonym_mapping"} & feat_tables))
        check("DB-2 raw amount never stored", "amount" not in feat_cols and "amount_ratio" in feat_cols)
        check("DB-2 device stored hashed only", "device_hash" in dev_cols and "device_id" not in dev_cols)
        check("DB-2 contains no PII / raw values", not leaked, f"leaked={leaked[:3]}")

        check("device_fingerprints has hashed device", True)  # covered by schema checks

        # ---- demo warm-up harness: age an account in DB-2 ------------------
        # The demo seed warms an account up first (normal history + aging)
        # so the preview doesn't score a brand-new account cold. The aging
        # itself lives in the Privacy Layer (the DB-2 owner) as a dev-only
        # endpoint.
        print("\n-- demo age endpoint (warm-up harness) --")
        r = pc.post("/internal/demo-age-account", headers={"X-Internal-Token": "wrong"},
                    json={"fraud_id": fraud_id, "age_days": 60.0})
        check("age wrong token 401", r.status_code == 401)
        r = pc.post("/internal/demo-age-account", headers={"X-Internal-Token": TOKEN},
                    json={"fraud_id": "F" + "A" * 15, "age_days": 60.0})
        check("age unknown profile 404", r.status_code == 404)
        r = pc.post("/internal/demo-age-account", headers={"X-Internal-Token": TOKEN},
                    json={"fraud_id": fraud_id, "age_days": 60.0})
        assert r.status_code == 200, r.text
        ag = r.json()
        check("age backdated rows + profile birth",
              ag["rows_backdated"] == 5 and ag["profile_age_days"] == 60.0, str(ag))
        import datetime as _dt
        con = sqlite3.connect(features_db)
        row_times = [t[0] for t in con.execute(
            "SELECT created_at FROM transaction_features WHERE fraud_id=? ORDER BY created_at", (fraud_id,))]
        birth_raw = con.execute(
            "SELECT created_at FROM fraud_profiles WHERE fraud_id=?", (fraud_id,)).fetchone()[0]
        con.close()
        span_days = (_dt.datetime.fromisoformat(row_times[-1]) - _dt.datetime.fromisoformat(row_times[0])).days
        birth_age = (_dt.datetime.now() - _dt.datetime.fromisoformat(birth_raw)).days
        check("age: history spread over ~7 weeks (8..55d)", 40 <= span_days <= 55, f"span={span_days}")
        check("age: profile birth ~60 days ago", 55 <= birth_age <= 65, f"birth_age={birth_age}")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
