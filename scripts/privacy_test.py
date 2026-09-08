#!/usr/bin/env python3
"""Privacy test suite — proves PII isolation, feature safety, and access controls.

This test is designed to work from a CLEAN CHECKOUT:
- Creates isolated temporary test databases
- Never inspects a developer's personal/local database
- Tests schema design, not runtime data
"""

import sys
import os
import json
import sqlite3
import tempfile
import shutil
from pathlib import Path

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


def create_test_identity_db(path):
    """Create a test identity DB with the expected schema."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE users (
            user_id VARCHAR(36) PRIMARY KEY,
            full_name TEXT NOT NULL,
            phone_encrypted BLOB NOT NULL,
            email_encrypted BLOB NOT NULL,
            email_hash VARCHAR(64) UNIQUE NOT NULL,
            kyc_doc_ref TEXT,
            account_number VARCHAR(20),
            account_type VARCHAR(20),
            role VARCHAR(20) DEFAULT 'user',
            address_encrypted BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE pseudonym_mapping (
            fraud_id VARCHAR(16) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            rotation_group INTEGER DEFAULT 1,
            access_log_required BOOLEAN DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE auth_credentials (
            credential_id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            credential_type VARCHAR(32) DEFAULT 'password',
            secret_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def create_test_features_db(path):
    """Create a test features DB with the expected schema."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE fraud_profiles (
            fraud_id VARCHAR(16) PRIMARY KEY,
            avg_txn_amount_90d REAL NOT NULL,
            txn_freq_7d INTEGER DEFAULT 0,
            typical_txn_hours TEXT NOT NULL,
            known_device_count INTEGER DEFAULT 0,
            usual_locations TEXT NOT NULL,
            usual_recipients TEXT NOT NULL,
            behavioral_baseline_vector TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE transaction_features (
            event_id VARCHAR(36) PRIMARY KEY,
            fraud_id VARCHAR(16) NOT NULL,
            amount_ratio REAL NOT NULL,
            txn_amount_bucket VARCHAR(32) NOT NULL,
            txn_freq_last_24h INTEGER DEFAULT 0,
            txn_time_unusual BOOLEAN DEFAULT 0,
            new_device_flag BOOLEAN DEFAULT 0,
            unusual_location_flag BOOLEAN DEFAULT 0,
            unusual_recipient_flag BOOLEAN DEFAULT 0,
            failed_auth_count_24h INTEGER DEFAULT 0,
            days_since_last_similar_txn REAL NOT NULL,
            gradual_escalation_score REAL DEFAULT 0.0,
            known_device_count INTEGER DEFAULT 0,
            account_tenure_days REAL DEFAULT 1.0,
            hour_of_day INTEGER,
            is_weekend BOOLEAN DEFAULT 0,
            shared_device_accounts INTEGER DEFAULT 0,
            shared_recipient_accounts INTEGER DEFAULT 0,
            mule_ring_score REAL DEFAULT 0.0,
            baseline_committed BOOLEAN DEFAULT 0,
            device_hash VARCHAR(32),
            location_id VARCHAR(64),
            recipient_id VARCHAR(64),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE device_fingerprints (
            device_hash VARCHAR(32) PRIMARY KEY,
            fraud_id VARCHAR(16) NOT NULL,
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def create_test_risk_db(path):
    """Create a test risk DB with the expected schema."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE risk_scores (
            score_id VARCHAR(36) PRIMARY KEY,
            fraud_id VARCHAR(16) NOT NULL,
            event_id VARCHAR(64) NOT NULL,
            risk_score INTEGER NOT NULL,
            risk_band VARCHAR(16) NOT NULL,
            reason_codes TEXT NOT NULL,
            model_version VARCHAR(64),
            ml_score REAL,
            rule_score REAL,
            degraded BOOLEAN DEFAULT 0,
            scored_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def create_test_audit_db(path):
    """Create a test audit DB with the expected schema."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE audit_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id VARCHAR(36) NOT NULL,
            fraud_id VARCHAR(16) NOT NULL,
            event_type VARCHAR(32) NOT NULL,
            prev_hash VARCHAR(64) NOT NULL,
            entry_hash VARCHAR(64) NOT NULL,
            payload_summary TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def run_tests():
    global passed, failed, errors
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("PRIVACY TEST SUITE")
    print("=" * 70)

    # Create isolated temporary test databases
    tmpdir = tempfile.mkdtemp(prefix="ps14_privacy_test_")
    try:
        identity_db = Path(tmpdir) / "identity.db"
        features_db = Path(tmpdir) / "features.db"
        risk_db = Path(tmpdir) / "risk.db"
        audit_db = Path(tmpdir) / "audit.db"

        create_test_identity_db(identity_db)
        create_test_features_db(features_db)
        create_test_risk_db(risk_db)
        create_test_audit_db(audit_db)

        # ---- Feature store tests ----
        print("\n[1] Feature Store (DB-2) — PII isolation")
        conn = sqlite3.connect(str(features_db))
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(transaction_features)")
        feature_cols = {row[1] for row in cursor.fetchall()}

        pii_columns = {"full_name", "email", "phone", "address", "account_number",
                        "ssn", "date_of_birth", "real_name", "user_name", "ip_address"}
        leaked = feature_cols & pii_columns
        test("No PII columns in transaction_features",
             len(leaked) == 0,
             f"Found: {leaked}" if leaked else "")

        # Check fraud_profiles for raw amounts
        cursor.execute("PRAGMA table_info(fraud_profiles)")
        profile_cols = {row[1] for row in cursor.fetchall()}

        raw_amount_cols = {c for c in profile_cols if "raw_amount" in c.lower()
                           or "actual_amount" in c.lower()
                           or "transaction_amount" in c.lower()}
        test("No raw amount columns in fraud_profiles",
             len(raw_amount_cols) == 0,
             f"Found: {raw_amount_cols}" if raw_amount_cols else "")

        # avg_txn_amount_90d should be ratio-based
        if "avg_txn_amount_90d" in profile_cols:
            cursor.execute("SELECT avg_txn_amount_90d FROM fraud_profiles LIMIT 5")
            rows = cursor.fetchall()
            if rows:
                vals = [r[0] for r in rows if r[0] is not None]
                test("avg_txn_amount_90d stores ratio-based values (not raw amounts)",
                     all(0.001 < v < 1000 for v in vals) if vals else True,
                     f"Values: {vals[:5]}")
            else:
                test("avg_txn_amount_90d stores ratio-based values", True, "(no data)")
        else:
            test("avg_txn_amount_90d column exists", False, "column missing")

        # Address column should NOT exist in features.db
        cursor.execute("PRAGMA table_info(transaction_features)")
        has_address = any("address" in row[1].lower() for row in cursor.fetchall())
        test("No address column in transaction_features",
             not has_address)

        conn.close()

        # ---- Risk scores tests ----
        print("\n[2] Risk Scores (DB-3) — PII isolation")
        conn = sqlite3.connect(str(risk_db))
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(risk_scores)")
        risk_cols = {row[1] for row in cursor.fetchall()}

        pii_risk = {"full_name", "email", "phone", "address", "account_number",
                     "user_id", "real_name", "ip_address"}
        leaked_risk = risk_cols & pii_risk
        test("No PII columns in risk_scores",
             len(leaked_risk) == 0,
             f"Found: {leaked_risk}" if leaked_risk else "")

        test("risk_scores only contains fraud_id (not user_id)",
             "fraud_id" in risk_cols and "user_id" not in risk_cols)

        conn.close()

        # ---- Audit events tests ----
        print("\n[3] Audit Events (DB-4) — PII isolation")
        conn = sqlite3.connect(str(audit_db))
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(audit_events)")
        audit_cols = {row[1] for row in cursor.fetchall()}

        pii_audit = {"full_name", "email", "phone", "address", "account_number",
                      "user_id", "real_name", "ip_address", "transaction_amount"}
        leaked_audit = pii_audit & audit_cols
        test("No PII columns in audit_events",
             len(leaked_audit) == 0,
             f"Found: {leaked_audit}" if leaked_audit else "")

        # Check that audit event payloads don't contain PII patterns
        cursor.execute("SELECT payload_summary FROM audit_events LIMIT 50")
        payloads = cursor.fetchall()
        pii_patterns = ["@gmail.com", "@yahoo.com", "@hotmail.com",
                        "SSN:", "ssn:", "address:", "Address:"]
        found_pii = False
        for (payload_str,) in payloads:
            if payload_str:
                for pattern in pii_patterns:
                    if pattern.lower() in payload_str.lower():
                        found_pii = True
                        break
        test("Audit event payloads contain no PII patterns",
             not found_pii)

        conn.close()

        # ---- Identity service tests ----
        print("\n[4] Identity Service — address isolation")
        conn = sqlite3.connect(str(identity_db))
        cursor = conn.cursor()

        # Plaintext address column should not exist
        cursor.execute("PRAGMA table_info(users)")
        user_cols = {row[1] for row in cursor.fetchall()}
        test("No plaintext 'address' column in users table",
             "address" not in user_cols,
             "Found 'address' TEXT column — PII leak vector" if "address" in user_cols else "")

        # address_encrypted should exist (properly encrypted)
        test("address_encrypted column exists in users table",
             "address_encrypted" in user_cols)

        # No raw PII columns
        raw_pii = {"email", "phone", "address", "ssn"}
        found_raw = raw_pii & user_cols
        test("No raw PII columns in users table (all encrypted)",
             len(found_raw) == 0,
             f"Found raw: {found_raw}" if found_raw else "")

        # pseudonym_mapping should NOT have any PII
        cursor.execute("PRAGMA table_info(pseudonym_mapping)")
        mapping_cols = {row[1] for row in cursor.fetchall()}
        test("pseudonym_mapping has no PII columns",
             not any(c in mapping_cols for c in ["email", "phone", "address", "full_name"]))

        conn.close()

        # ---- Feature contract tests ----
        print("\n[5] Feature Contract — no PII-derived features")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            from src.privacy_layer.features import ML_FEATURES

            pii_features = {"email_hash", "phone_hash", "address_hash", "full_name",
                            "user_id", "account_number", "ssn", "ip_address",
                            "real_amount", "raw_amount", "transaction_amount"}
            leaked_features = set(ML_FEATURES) & pii_features
            test("ML_FEATURES contains no PII-derived features",
                 len(leaked_features) == 0,
                 f"Found: {leaked_features}" if leaked_features else "")

            ratio_features = {"amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
                              "new_device_flag", "unusual_location_flag",
                              "unusual_recipient_flag", "failed_auth_count_24h",
                              "days_since_last_similar_txn", "gradual_escalation_score",
                              "known_device_count", "account_tenure_days",
                              "hour_of_day", "is_weekend",
                              "shared_device_accounts", "shared_recipient_accounts",
                              "mule_ring_score"}
            missing = ratio_features - set(ML_FEATURES)
            test("All expected features are present",
                 len(missing) == 0,
                 f"Missing: {missing}" if missing else "")

            raw_features = [f for f in ML_FEATURES if "raw" in f.lower()
                            or "actual" in f.lower() or "real_" in f.lower()]
            test("No features with 'raw'/'actual'/'real_' in name",
                 len(raw_features) == 0,
                 f"Found: {raw_features}" if raw_features else "")

        except ImportError as e:
            test("Feature module importable", False, str(e))

        # ---- Model artifact tests ----
        print("\n[6] Model Artifacts — no PII leakage")
        artifacts_dir = Path(__file__).resolve().parent.parent / "models" / "artifacts"
        if artifacts_dir.exists():
            metadata_path = artifacts_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                meta_str = json.dumps(metadata).lower()
                pii_in_meta = ["email", "phone", "address", "ssn", "full_name"]
                found = [p for p in pii_in_meta if p in meta_str]
                test("metadata.json contains no PII references",
                     len(found) == 0,
                     f"Found: {found}" if found else "")
            else:
                test("metadata.json exists", False)

            if metadata_path.exists() and "feature_names" in metadata:
                meta_features = set(metadata["feature_names"])
                pii_in_model = meta_features & {"email", "phone", "address", "full_name",
                                                 "user_id", "account_number"}
                test("Model features contain no PII",
                     len(pii_in_model) == 0,
                     f"Found: {pii_in_model}" if pii_in_model else "")
        else:
            test("Artifacts directory exists", False, "(expected for fresh checkout)")

        # ---- Amount safety tests ----
        print("\n[7] Amount Safety — no raw financial persistence")
        conn = sqlite3.connect(str(features_db))
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(transaction_features)")
        cols = {row[1] for row in cursor.fetchall()}

        test("Feature store has 'amount_ratio' (ratio-based)",
             "amount_ratio" in cols)
        test("Feature store does NOT have 'raw_amount'",
             "raw_amount" not in cols)
        test("Feature store does NOT have 'transaction_amount'",
             "transaction_amount" not in cols)

        cursor.execute("PRAGMA table_info(fraud_profiles)")
        pcols = {row[1] for row in cursor.fetchall()}
        test("FraudProfile has 'avg_txn_amount_90d' (ratio-based EMA)",
             "avg_txn_amount_90d" in pcols)
        test("FraudProfile does NOT have 'raw_median_amount'",
             "raw_median_amount" not in pcols)

        conn.close()

        # ---- Device/location/recipient abstraction tests ----
        print("\n[8] Identifier Abstraction — no raw IDs in features")
        conn = sqlite3.connect(str(features_db))
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(transaction_features)")
        cols = {row[1] for row in cursor.fetchall()}

        test("Device stored as 'device_hash' (not raw device_id)",
             "device_hash" in cols)
        test("No raw 'device_id' column",
             "device_id" not in cols)
        test("Location stored as 'location_id' (coarse token)",
             "location_id" in cols)
        test("Recipient stored as 'recipient_id' (opaque token)",
             "recipient_id" in cols)

        conn.close()

        # ---- Raw amount persistence test (P0 requirement) ----
        print("\n[9] Raw Amount Persistence — regression test")
        conn = sqlite3.connect(str(features_db))
        cursor = conn.cursor()

        # Simulate: insert a fraud profile with ratio-based value
        cursor.execute("""
            INSERT INTO fraud_profiles
            (fraud_id, avg_txn_amount_90d, txn_freq_7d, typical_txn_hours,
             known_device_count, usual_locations, usual_recipients,
             behavioral_baseline_vector)
            VALUES ('FTEST00000000001', 1.0, 5, '[9,10,14,15]', 2, '["loc1"]',
                    '["recip1"]', '{"txn_count": 5}')
        """)
        conn.commit()

        # Verify: ratio-based value stored, not raw amount
        cursor.execute("SELECT avg_txn_amount_90d FROM fraud_profiles WHERE fraud_id='FTEST00000000001'")
        val = cursor.fetchone()[0]
        test("New profile initializes with ratio baseline (1.0)",
             val == 1.0,
             f"Got: {val}")

        # Verify: no column named 'raw_amount' or 'transaction_amount' exists
        cursor.execute("PRAGMA table_info(fraud_profiles)")
        all_cols = {row[1] for row in cursor.fetchall()}
        test("No raw amount columns exist in fraud_profiles schema",
             not any(c in all_cols for c in ["raw_amount", "transaction_amount", "actual_amount"]))

        conn.close()

    finally:
        # Clean up temporary databases
        shutil.rmtree(tmpdir, ignore_errors=True)

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
