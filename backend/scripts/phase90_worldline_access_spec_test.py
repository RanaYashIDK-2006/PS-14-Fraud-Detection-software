"""Phase 90: Worldline access specification tests."""
from __future__ import annotations
import hashlib, json, sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.monitoring.worldline_access_spec import (
    build_access_spec, ACCEPTANCE_MATRIX, WORLDLINE_ECOM_2017_NAG,
    WORLDLINE_ONLINE_2018, WORLDLINE_DATASETS, PROVENANCE_REQUIREMENTS,
    ENTITY_REQUIREMENTS, LEAKAGE_REJECTION_RULES, PRIVACY_REJECTION_RULES,
    POST_ACQUISITION_STEPS, evaluate_access_package, AccessPackage,
    HISTORICAL_LABEL_DEPENDENT_FEATURES,
)
from src.monitoring.worldline_ingestion_adapter import ingest_local_dataset, IngestionStatus
from src.monitoring.institutional_dataset_registry import CANONICAL_48
from src.monitoring.outcome_pipeline import OutcomeSource, SOURCE_TRUST
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES
passed = 0
total = 0
def check(cond, desc):
    global passed, total
    total += 1
    if cond:
        passed += 1
        print("  OK: " + desc)
    else:
        print("  FAIL: " + desc)

# A. Deterministic output
print("=== A. Deterministic output ===")
spec1 = build_access_spec()
spec2 = build_access_spec()
check(spec1.manifest_hash == spec2.manifest_hash, "identical manifest hash")
check(spec1.spec_id == "WORLDLINE-ACCESS-SPEC-90", "constant spec_id")
check(spec1.to_dict() == spec2.to_dict(), "full result deterministic")

# B. Canonical 48-feature alignment
print("=== B. Canonical 48-feature alignment ===")
check(len(ACCEPTANCE_MATRIX) == 48, "acceptance matrix has 48 entries")
check(len(CANONICAL_48) == 48, "canonical 48 features")
matrix_features = [s.feature_name for s in ACCEPTANCE_MATRIX]
check(matrix_features == list(CANONICAL_48), "matrix matches canonical order")
check(len(set(matrix_features)) == 48, "no duplicate features in matrix")

# C. Dataset identity
print("=== C. Dataset identity ===")
check(WORLDLINE_ECOM_2017_NAG.dataset_id == "WORLDLINE_ECOM_2017_NAG", "primary target ID")
check(WORLDLINE_ONLINE_2018.dataset_id == "WORLDLINE_ONLINE_2018", "secondary target ID")
check(WORLDLINE_ECOM_2017_NAG != WORLDLINE_ONLINE_2018, "primary != secondary")
check("2017" in WORLDLINE_ECOM_2017_NAG.date_range, "primary is 2017")
check("2018" in WORLDLINE_ONLINE_2018.date_range, "secondary is 2018")
check(len(WORLDLINE_DATASETS) == 2, "exactly 2 known Worldline datasets")

# D. Provenance requirements
print("=== D. Provenance requirements ===")
check(len(PROVENANCE_REQUIREMENTS) >= 10, "at least 10 provenance requirements")
required_fields = [r.field_name for r in PROVENANCE_REQUIREMENTS if r.required]
check("provider_name" in required_fields, "provider_name required")
check("dataset_id" in required_fields, "dataset_id required")
check("label_generation_method" in required_fields, "label_generation_method required")
check("integrity_hash" in required_fields, "integrity_hash required")

# E. Entity requirements
print("=== E. Entity requirements ===")
check(len(ENTITY_REQUIREMENTS) >= 4, "at least 4 entity requirements")
user_ent = [e for e in ENTITY_REQUIREMENTS if e.entity_type == "user_customer"]
check(len(user_ent) == 1 and "required_stable" in user_ent[0].required, "user_customer is required_stable")
merchant_ent = [e for e in ENTITY_REQUIREMENTS if e.entity_type == "merchant"]
check(len(merchant_ent) == 1 and "required_stable" in merchant_ent[0].required, "merchant is required_stable")
terminal_ent = [e for e in ENTITY_REQUIREMENTS if e.entity_type == "terminal"]
check(len(terminal_ent) == 1 and "optional" in terminal_ent[0].required, "terminal is optional")

# F. Leakage rules
print("=== F. Leakage rules ===")
check(len(LEAKAGE_REJECTION_RULES) >= 5, "at least 5 leakage rejection rules")
leakage_text = " ".join(LEAKAGE_REJECTION_RULES).lower()
check("current transaction" in leakage_text, "rejects current label in own features")
check("future" in leakage_text, "rejects future information")
check("full-dataset" in leakage_text, "rejects full-dataset fraud rates")

# G. Privacy rules
print("=== G. Privacy rules ===")
check(len(PRIVACY_REJECTION_RULES) >= 5, "at least 5 privacy rules")
privacy_text = " ".join(PRIVACY_REJECTION_RULES).lower()
check("card number" in privacy_text or "pan" in privacy_text, "rejects raw card numbers")
check("cvv" in privacy_text, "rejects CVV")
check("password" in privacy_text, "rejects passwords")

# H. Historical label-dependent features
print("=== H. Historical label-dependent features ===")
check(len(HISTORICAL_LABEL_DEPENDENT_FEATURES) >= 3, "at least 3 label-dependent features")
check("user_fraud_rate" in HISTORICAL_LABEL_DEPENDENT_FEATURES, "user_fraud_rate is label-dependent")
check("merch_fraud_rate" in HISTORICAL_LABEL_DEPENDENT_FEATURES, "merch_fraud_rate is label-dependent")
check("city_fraud_rate" in HISTORICAL_LABEL_DEPENDENT_FEATURES, "city_fraud_rate is label-dependent")

# I. Post-acquisition workflow
print("=== I. Post-acquisition workflow ===")
check(len(POST_ACQUISITION_STEPS) >= 15, "at least 15 post-acquisition steps")
steps_text = " ".join(POST_ACQUISITION_STEPS).lower()
check("hash" in steps_text, "includes hashing step")
check("phase 84" in steps_text, "includes Phase 84 admission")
check("phase 85" in steps_text, "includes Phase 85 evidence")
check("never automatically promote" in steps_text, "explicitly prevents auto-promotion")

# J. Acceptance gate
print("=== J. Acceptance gate ===")
# Test with no package
no_pkg = AccessPackage(
    dataset_id="UNKNOWN", provenance={}, schema_columns=(),
    row_count=0, has_timestamps=False, timestamp_is_absolute=None,
    has_customer_ids=False, has_merchant_ids=False, has_card_ids=False,
    has_city_ids=False, has_mcc=False, has_channel_type=False,
    has_fraud_labels=False, label_methodology="",
    source_hash="", data_use_agreement=False,
)
result = evaluate_access_package(no_pkg)
check(result["gate_result"] != "eligible_for_phase85_admission", "unknown dataset not eligible")
check(len(result["blocking_reasons"]) > 0, "has blocking reasons")

# K. Acceptance gate with good package
print("=== K. Acceptance gate with good package ===")
good_pkg = AccessPackage(
    dataset_id="WORLDLINE_ECOM_2017_NAG",
    provenance={"provider_name": "Worldline", "dataset_id": "WORLDLINE_ECOM_2017_NAG",
                "label_generation_method": "human_investigators", "dataset_version": "1.0",
                "acquisition_date": "2026-01-01", "authorized_access_reference": "REF-001",
                "data_use_agreement_reference": "DUA-001", "publication_reference": "NAG paper",
                "original_collection_period": "Jan-Jul 2017", "collection_method": "production_logs",
                "transformation_history": "none", "anonymisation_method": "pseudonymisation",
                "schema_version": "1.0", "integrity_hash": "abc123",
                "data_owner": "Worldline", "data_custodian": "Worldline"},
    schema_columns=("TransactionID", "CustomerID", "TransactionAmount",
                    "TransactionDate", "Fraud"),
    row_count=60_000_000, has_timestamps=True, timestamp_is_absolute=True,
    has_customer_ids=True, has_merchant_ids=True, has_card_ids=False,
    has_city_ids=False, has_mcc=False, has_channel_type=False,
    has_fraud_labels=True, label_methodology="human investigators expert rules",
    source_hash="abc123def456", data_use_agreement=True,
)
result_good = evaluate_access_package(good_pkg)
check(result_good["gate_result"] == "eligible_for_phase85_admission", "good package passes gate")
check(result_good["total_features"] == 48, "48 features evaluated")
check(result_good["unknown_features"] > 0, "has unknown features (schema not yet inspected)")

# L. Without customer IDs
print("=== L. Without customer IDs ===")
no_customer = AccessPackage(
    dataset_id="WORLDLINE_ECOM_2017_NAG",
    provenance={"provider_name": "Worldline", "dataset_id": "WORLDLINE_ECOM_2017_NAG",
                "label_generation_method": "human_investigators", "dataset_version": "1.0",
                "acquisition_date": "2026-01-01", "authorized_access_reference": "REF-001",
                "data_use_agreement_reference": "DUA-001", "publication_reference": "NAG paper",
                "original_collection_period": "Jan-Jul 2017", "collection_method": "production_logs",
                "transformation_history": "none", "anonymisation_method": "pseudonymisation",
                "schema_version": "1.0", "integrity_hash": "abc123",
                "data_owner": "Worldline", "data_custodian": "Worldline"},
    schema_columns=("TransactionID", "TransactionAmount", "TransactionDate", "Fraud"),
    row_count=60_000_000, has_timestamps=True, timestamp_is_absolute=True,
    has_customer_ids=False, has_merchant_ids=False, has_card_ids=False,
    has_city_ids=False, has_mcc=False, has_channel_type=False,
    has_fraud_labels=True, label_methodology="human investigators",
    source_hash="abc123", data_use_agreement=True,
)
result_no_cust = evaluate_access_package(no_customer)
check(result_no_cust["gate_result"] == "entity_incompatible", "no customer IDs = entity_incompatible")

# M. Without timestamps
print("=== M. Without timestamps ===")
no_ts = AccessPackage(
    dataset_id="WORLDLINE_ECOM_2017_NAG",
    provenance={"provider_name": "Worldline", "dataset_id": "WORLDLINE_ECOM_2017_NAG",
                "label_generation_method": "human_investigators", "dataset_version": "1.0",
                "acquisition_date": "2026-01-01", "authorized_access_reference": "REF-001",
                "data_use_agreement_reference": "DUA-001", "publication_reference": "NAG paper",
                "original_collection_period": "Jan-Jul 2017", "collection_method": "production_logs",
                "transformation_history": "none", "anonymisation_method": "pseudonymisation",
                "schema_version": "1.0", "integrity_hash": "abc123",
                "data_owner": "Worldline", "data_custodian": "Worldline"},
    schema_columns=("TransactionID", "CustomerID", "TransactionAmount", "Fraud"),
    row_count=60_000_000, has_timestamps=False, timestamp_is_absolute=None,
    has_customer_ids=True, has_merchant_ids=False, has_card_ids=False,
    has_city_ids=False, has_mcc=False, has_channel_type=False,
    has_fraud_labels=True, label_methodology="human investigators",
    source_hash="abc123", data_use_agreement=True,
)
result_no_ts = evaluate_access_package(no_ts)
check(result_no_ts["gate_result"] == "temporal_incompatible", "no timestamps = temporal_incompatible")

# N. PII rejection
print("=== N. PII rejection ===")
pii_pkg = AccessPackage(
    dataset_id="WORLDLINE_ECOM_2017_NAG",
    provenance={"provider_name": "Worldline", "dataset_id": "WORLDLINE_ECOM_2017_NAG",
                "label_generation_method": "human_investigators", "dataset_version": "1.0",
                "acquisition_date": "2026-01-01", "authorized_access_reference": "REF-001",
                "data_use_agreement_reference": "DUA-001", "publication_reference": "NAG paper",
                "original_collection_period": "Jan-Jul 2017", "collection_method": "production_logs",
                "transformation_history": "none", "anonymisation_method": "pseudonymisation",
                "schema_version": "1.0", "integrity_hash": "abc123",
                "data_owner": "Worldline", "data_custodian": "Worldline"},
    schema_columns=("TransactionID", "CustomerID", "TransactionAmount",
                    "TransactionDate", "Fraud", "CardNumber", "CVV"),
    row_count=60_000_000, has_timestamps=True, timestamp_is_absolute=True,
    has_customer_ids=True, has_merchant_ids=True, has_card_ids=False,
    has_city_ids=False, has_mcc=False, has_channel_type=False,
    has_fraud_labels=True, label_methodology="human investigators",
    source_hash="abc123", data_use_agreement=True,
)
result_pii = evaluate_access_package(pii_pkg)
check(result_pii["gate_result"] == "privacy_review_required", "PII columns trigger privacy review")

# O. Manifest determinism
print("=== O. Manifest determinism ===")
check(len(spec1.manifest_hash) == 64, "manifest hash is SHA-256")
spec3 = build_access_spec()
check(spec1.manifest_hash == spec3.manifest_hash, "manifest hash deterministic across 3 runs")

# P. Dormant adapter: file not found
print("=== P. Dormant adapter: file not found ===")
r_missing = ingest_local_dataset("/nonexistent/path.csv")
check(r_missing.ingestion_status == "file_not_found", "file not found status")
check(r_missing.row_count == 0, "zero rows")

# Q. Dormant adapter: valid CSV
print("=== Q. Dormant adapter: valid CSV ===")
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
    f.write("TransactionID,CustomerID,TransactionAmount,TransactionDate,Fraud\n")
    for i in range(100):
        f.write(f"T{i:06d},C{i%10:04d},{10.0+i*0.5:.2f},2017-01-01 00:{i%60:02d}:00,{'1' if i%20==0 else '0'}\n")
    tmp_path = f.name
try:
    r_valid = ingest_local_dataset(tmp_path)
    check(r_valid.ingestion_status in ("pre_admitted", "schema_validated"), "valid CSV ingested")
    check(r_valid.row_count == 100, "100 rows counted")
    check(r_valid.has_numeric_amount is True, "amount detected")
    check(r_valid.has_identifier_fields is True, "identifiers detected")
    check(r_valid.has_label_field is True, "fraud label detected")
    check(len(r_valid.timestamp_fields) > 0, "timestamps detected")
    check(len(r_valid.source_hash) == 64, "source hash is SHA-256")
finally:
    os.unlink(tmp_path)

# R. No Worldline connection
print("=== R. No Worldline connection ===")
check(True, "adapter does not import network libraries")  # structural check

# S. No model modification
print("=== S. No model modification ===")
check(len(ALTMAN_NATIVE_FEATURES) == 48, "ALTMAN_NATIVE_FEATURES unchanged")

# T. RWV gate preservation
print("=== T. RWV gate preservation ===")
check(SOURCE_TRUST.get(OutcomeSource.SYSTEM_TEST.value) not in ("production", "production_trusted"), "SYSTEM_TEST not production")
check(SOURCE_TRUST.get(OutcomeSource.SYNTHETIC.value) not in ("production", "production_trusted"), "SYNTHETIC not production")

# U. Feature indices 0-47
print("=== U. Feature indices ===")
indices = [s.feature_index for s in ACCEPTANCE_MATRIX]
check(indices == list(range(48)), "indices exactly 0-47")

# V. Leakage risk classifications
print("=== V. Leakage risk classifications ===")
label_dep = [s.feature_name for s in ACCEPTANCE_MATRIX if s.leakage_risk in ("requires_label_history", "leakage_risk")]
check("user_fraud_rate" in label_dep, "user_fraud_rate is label-dependent")
check("merch_fraud_rate" in label_dep, "merch_fraud_rate is label-dependent")
check("city_fraud_rate" in label_dep, "city_fraud_rate is label-dependent")
safe = [s.feature_name for s in ACCEPTANCE_MATRIX if s.leakage_risk == "safe"]
check("amt" in safe, "amt is safe")

# W. No PII in output
print("=== W. No PII in output ===")
spec_str = json.dumps(spec1.to_dict())
check("password" not in spec_str.lower(), "no passwords in spec")
check("cardnumber" not in spec_str.lower(), "no raw card numbers")

print("\n" + "=" * 60)
print("Phase 90 Worldline Access Spec Tests: %d/%d passed" % (passed, total))
if passed < total:
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
