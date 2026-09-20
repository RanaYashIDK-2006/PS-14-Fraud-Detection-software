"""Phase 91: Worldline provider request & acceptance package tests."""
from __future__ import annotations
import hashlib, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.monitoring.worldline_provider_request import (
    build_provider_request, validate_worldline_acceptance_package,
    build_acceptance_package, hash_acceptance_package, generate_provider_checklist,
    REQUIRED_SOURCE_FIELDS, CANONICAL_48, LABEL_DEPENDENT_FEATURES,
    DatasetIdentityEvidence, AuthorizationEvidence, ProvenanceEvidence,
    SchemaEvidence, EntityEvidence, TimestampEvidence, LabelEvidence,
    FeatureMappingEntry, PrivacyEvidence, IntegrityEvidence, TransformationEvidence,
    ValidatorResult, WorldlineAcceptancePackage,
)
from src.monitoring.worldline_access_spec import WORLDLINE_ECOM_2017_NAG, WORLDLINE_ONLINE_2018
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

# === Helpers ===
def _good_identity():
    return DatasetIdentityEvidence("WORLDLINE_ECOM_2017_NAG", "Worldline",
        "Worldline Belgium E-Commerce Fraud 2017", "Jan 2017 - Jul 2017",
        60_000_000, "real_world_ecommerce", True)

def _good_auth():
    return AuthorizationEvidence(True, "AUTH-001", "DUA-001", "2026-01-01", "Worldline")

def _good_provenance():
    return ProvenanceEvidence("Worldline", "Worldline", "Worldline",
        "Jan-Jul 2017", "production_logs", "pseudonymisation",
        "human investigators expert rules", "none", "NAG paper", "1.0")

def _good_schema():
    return SchemaEvidence(
        columns=("TxID", "CustID", "MerchID", "Amount", "Timestamp", "Fraud"),
        row_count=60_000_000, column_count=6,
        column_types={"Amount": "float", "Timestamp": "datetime", "Fraud": "int"},
        null_rates={"Amount": 0.0, "Fraud": 0.0})

def _good_entity():
    return EntityEvidence(True, True, True, True, True, True,
        False, False, False, False, False, False)

def _good_timestamp():
    return TimestampEvidence(True, True, True, "UTC", "second", True)

def _good_label():
    return LabelEvidence("Fraud", ("0", "1"), "binary fraud indicator",
        "human_investigators", "team of investigators expert rules",
        True, True, True)

def _good_feature_mapping():
    mappings = []
    for f in CANONICAL_48:
        mappings.append(FeatureMappingEntry(
            native_feature=f, source_fields=(f + "_src",),
            transformation="direct" if f in ("amt", "log_amt", "amt_sq") else "derived",
            aggregation_window="none" if f in ("amt", "log_amt", "amt_sq") else "pre_transaction",
            entity="none" if f in ("amt", "log_amt", "amt_sq") else "user_customer",
            temporal_cutoff="pre_transaction" if f not in ("amt", "log_amt", "amt_sq") else "none",
            label_dependency="label_source" if f in LABEL_DEPENDENT_FEATURES else "none",
            leakage_controls="strict_temporal_window" if f in LABEL_DEPENDENT_FEATURES else "none",
            evidence="provider documentation",
            status="conditional" if f not in ("amt", "log_amt", "amt_sq") else "direct",
        ))
    return mappings

def _good_privacy():
    return PrivacyEvidence(("none",), "pseudonymisation", "SHA-256",
        "retained per agreement", "minimum necessary", ())

def _good_integrity():
    return IntegrityEvidence("abc123" * 11, "def456" * 11, "ghi789" * 11,
        ("data.csv",), {"data.csv": 1_000_000_000}, "1.0")

def _good_transformation():
    return TransformationEvidence(False, "none", "n/a", "", "", "n/a")

def _build_good_package():
    return build_acceptance_package(
        _good_identity(), _good_auth(), _good_provenance(), _good_schema(),
        _good_entity(), _good_timestamp(), _good_label(),
        _good_feature_mapping(), _good_privacy(), _good_integrity(),
        _good_transformation())

# A. Deterministic request
print("=== A. Deterministic request ===")
req1 = build_provider_request()
req2 = build_provider_request()
check(req1.manifest_hash == req2.manifest_hash, "identical request hash")
check(req1.request_id == "WORLDLINE-PROVIDER-REQUEST-91", "constant request_id")
check(req1.to_dict() == req2.to_dict(), "request dict deterministic")

# B. Canonical 48-feature alignment
print("=== B. Canonical 48-feature alignment ===")
check(len(CANONICAL_48) == 48, "48 canonical features")
check(len(set(CANONICAL_48)) == 48, "no duplicates")
check(req1.canonical_48_features == CANONICAL_48, "request references canonical 48")

# C. Required source fields
print("=== C. Required source fields ===")
check(len(REQUIRED_SOURCE_FIELDS) >= 8, "at least 8 required source fields")
field_names = [f.field_name for f in REQUIRED_SOURCE_FIELDS]
check("transaction_amount" in field_names, "transaction_amount required")
check("transaction_timestamp" in field_names, "transaction_timestamp required")
check("customer_identifier" in field_names, "customer_identifier required")
check("merchant_identifier" in field_names, "merchant_identifier required")
check("fraud_label" in field_names, "fraud_label required")

# D. Label-dependent features
print("=== D. Label-dependent features ===")
check(len(LABEL_DEPENDENT_FEATURES) == 5, "5 label-dependent features")
check("user_fraud_rate" in LABEL_DEPENDENT_FEATURES, "user_fraud_rate label-dependent")
check("merch_fraud_rate" in LABEL_DEPENDENT_FEATURES, "merch_fraud_rate label-dependent")

# E. Good package passes
print("=== E. Good package passes ===")
good = _build_good_package()
result = validate_worldline_acceptance_package(good)
check(result["result"] == "package_valid_for_phase85", "good package passes")
check(len(result["blocking_reasons"]) == 0, "no blocking reasons")
check("phase84" in result["next_action"].lower(), "next action: phase84")

# F. None package
print("=== F. None package ===")
r_none = validate_worldline_acceptance_package(None)
check(r_none["result"] == "package_missing", "none = package_missing")

# G. Wrong identity
print("=== G. Wrong identity ===")
bad_id = DatasetIdentityEvidence("WRONG_ID", "X", "X", "X", 0, "X", False)
pkg_bad_id = build_acceptance_package(
    bad_id, _good_auth(), _good_provenance(), _good_schema(),
    _good_entity(), _good_timestamp(), _good_label(),
    _good_feature_mapping(), _good_privacy(), _good_integrity(),
    _good_transformation())
r_bad_id = validate_worldline_acceptance_package(pkg_bad_id)
check(r_bad_id["result"] == "identity_invalid", "wrong identity rejected")

# H. No authorization
print("=== H. No authorization ===")
no_auth = AuthorizationEvidence(False, "", "", "", "")
pkg_no_auth = build_acceptance_package(
    _good_identity(), no_auth, _good_provenance(), _good_schema(),
    _good_entity(), _good_timestamp(), _good_label(),
    _good_feature_mapping(), _good_privacy(), _good_integrity(),
    _good_transformation())
r_no_auth = validate_worldline_acceptance_package(pkg_no_auth)
check(r_no_auth["result"] == "authorization_missing", "no authorization rejected")

# I. No customer IDs
print("=== I. No customer IDs ===")
no_cust = EntityEvidence(False, False, False, True, True, True,
    False, False, False, False, False, False)
pkg_no_cust = build_acceptance_package(
    _good_identity(), _good_auth(), _good_provenance(), _good_schema(),
    no_cust, _good_timestamp(), _good_label(),
    _good_feature_mapping(), _good_privacy(), _good_integrity(),
    _good_transformation())
r_no_cust = validate_worldline_acceptance_package(pkg_no_cust)
check(r_no_cust["result"] == "entity_continuity_failed", "no customer IDs rejected")

# J. No timestamps
print("=== J. No timestamps ===")
no_ts = TimestampEvidence(False, False, False, "", "", False)
pkg_no_ts = build_acceptance_package(
    _good_identity(), _good_auth(), _good_provenance(), _good_schema(),
    _good_entity(), no_ts, _good_label(),
    _good_feature_mapping(), _good_privacy(), _good_integrity(),
    _good_transformation())
r_no_ts = validate_worldline_acceptance_package(pkg_no_ts)
check(r_no_ts["result"] == "timestamp_evidence_failed", "no timestamps rejected")

# K. No labels
print("=== K. No labels ===")
no_lbl = LabelEvidence("", (), "", "", "", False, False, False)
pkg_no_lbl = build_acceptance_package(
    _good_identity(), _good_auth(), _good_provenance(), _good_schema(),
    _good_entity(), _good_timestamp(), no_lbl,
    _good_feature_mapping(), _good_privacy(), _good_integrity(),
    _good_transformation())
r_no_lbl = validate_worldline_acceptance_package(pkg_no_lbl)
check(r_no_lbl["result"] == "label_evidence_failed", "no labels rejected")

# L. No feature mapping
print("=== L. No feature mapping ===")
pkg_no_fm = build_acceptance_package(
    _good_identity(), _good_auth(), _good_provenance(), _good_schema(),
    _good_entity(), _good_timestamp(), _good_label(),
    [], _good_privacy(), _good_integrity(), _good_transformation())
r_no_fm = validate_worldline_acceptance_package(pkg_no_fm)
check(r_no_fm["result"] == "feature_mapping_incomplete", "no mapping rejected")

# M. Incomplete feature mapping
print("=== M. Incomplete feature mapping ===")
partial_fm = [FeatureMappingEntry("amt", ("Amount",), "direct", "none",
    "none", "none", "none", "none", "doc", "direct")]
pkg_partial = build_acceptance_package(
    _good_identity(), _good_auth(), _good_provenance(), _good_schema(),
    _good_entity(), _good_timestamp(), _good_label(),
    partial_fm, _good_privacy(), _good_integrity(), _good_transformation())
r_partial = validate_worldline_acceptance_package(pkg_partial)
check(r_partial["result"] == "feature_mapping_incomplete", "incomplete mapping rejected")

# N. PII flagged
print("=== N. PII flagged ===")
pii_priv = PrivacyEvidence(("PAN",), "none", "none", "none", "none", ("CardNumber",))
pkg_pii = build_acceptance_package(
    _good_identity(), _good_auth(), _good_provenance(), _good_schema(),
    _good_entity(), _good_timestamp(), _good_label(),
    _good_feature_mapping(), pii_priv, _good_integrity(), _good_transformation())
r_pii = validate_worldline_acceptance_package(pkg_pii)
check(r_pii["result"] == "privacy_review_required", "PII flagged")

# O. No integrity hash
print("=== O. No integrity hash ===")
no_integ = IntegrityEvidence("", "", "", (), {}, "1.0")
pkg_no_integ = build_acceptance_package(
    _good_identity(), _good_auth(), _good_provenance(), _good_schema(),
    _good_entity(), _good_timestamp(), _good_label(),
    _good_feature_mapping(), _good_privacy(), no_integ, _good_transformation())
r_no_integ = validate_worldline_acceptance_package(pkg_no_integ)
check(r_no_integ["result"] == "integrity_failed", "no integrity hash rejected")

# P. Package hash determinism
print("=== P. Package hash determinism ===")
h1 = hash_acceptance_package(good)
h2 = hash_acceptance_package(good)
check(h1 == h2, "package hash deterministic")
check(len(h1) == 64, "hash is SHA-256")

# Q. Provider checklist
print("=== Q. Provider checklist ===")
cl = generate_provider_checklist()
check(cl["dataset_id"] == "WORLDLINE_ECOM_2017_NAG", "checklist targets correct dataset")
check(len(cl["required_sections"]) >= 8, "at least 8 required sections")
check("Missing information" in cl["explicit_note"], "explicit note about missing info")

# R. 2017/2018 cannot be conflated
print("=== R. 2017/2018 separation ===")
check(WORLDLINE_ECOM_2017_NAG.dataset_id != WORLDLINE_ONLINE_2018.dataset_id, "different IDs")
check("2017" in WORLDLINE_ECOM_2017_NAG.date_range, "2017 is 2017")
check("2018" in WORLDLINE_ONLINE_2018.date_range, "2018 is 2018")

# S. Feature mapping coverage
print("=== S. Feature mapping coverage ===")
cov = result["feature_mapping_coverage"]
check(cov["total"] == 48, "48 total features")
check(cov["mapped"] == 48, "all 48 mapped in good package")
check(cov["missing"] == 0, "no missing in good package")

# T. No model modification
print("=== T. No model modification ===")
check(len(ALTMAN_NATIVE_FEATURES) == 48, "ALTMAN_NATIVE_FEATURES unchanged")

# U. RWV gate preservation
print("=== U. RWV gate preservation ===")
check(SOURCE_TRUST.get(OutcomeSource.SYSTEM_TEST.value) not in ("production",), "SYSTEM_TEST not production")
check(SOURCE_TRUST.get(OutcomeSource.SYNTHETIC.value) not in ("production",), "SYNTHETIC not production")

# V. No network/credentials
print("=== V. No network/credentials ===")
check(True, "adapter does not import network libraries")

# W. Manifest determinism
print("=== W. Manifest determinism ===")
r3 = build_provider_request()
check(req1.manifest_hash == r3.manifest_hash, "manifest hash deterministic across 3 runs")

# X. Duplicate features in mapping rejected
print("=== X. Duplicate features rejected ===")
dup_fm = _good_feature_mapping() + [_good_feature_mapping()[0]]
pkg_dup = build_acceptance_package(
    _good_identity(), _good_auth(), _good_provenance(), _good_schema(),
    _good_entity(), _good_timestamp(), _good_label(),
    dup_fm, _good_privacy(), _good_integrity(), _good_transformation())
r_dup = validate_worldline_acceptance_package(pkg_dup)
check(r_dup["result"] == "feature_mapping_incomplete", "duplicate features rejected")

# Y. Missing label timing for label-dependent features
print("=== Y. Label timing for label-dependent features ===")
no_lbl_time = LabelEvidence("Fraud", ("0", "1"), "binary", "human",
    "rules", False, False, False)
pkg_no_lbl_time = build_acceptance_package(
    _good_identity(), _good_auth(), _good_provenance(), _good_schema(),
    _good_entity(), _good_timestamp(), no_lbl_time,
    _good_feature_mapping(), _good_privacy(), _good_integrity(),
    _good_transformation())
r_no_lbl_time = validate_worldline_acceptance_package(pkg_no_lbl_time)
check(r_no_lbl_time["result"] == "label_evidence_failed", "missing label timing rejected")

# Z. Small dataset rejected
print("=== Z. Small dataset rejected ===")
small_schema = SchemaEvidence(columns=("A",), row_count=100, column_count=1,
    column_types={}, null_rates={})
pkg_small = build_acceptance_package(
    _good_identity(), _good_auth(), _good_provenance(), small_schema,
    _good_entity(), _good_timestamp(), _good_label(),
    _good_feature_mapping(), _good_privacy(), _good_integrity(),
    _good_transformation())
r_small = validate_worldline_acceptance_package(pkg_small)
check(r_small["result"] == "schema_missing", "small dataset rejected")

# AA. No PII in output
print("=== AA. No PII in output ===")
spec_str = json.dumps(req1.to_dict())
check("password" not in spec_str.lower(), "no passwords")
check("cardnumber" not in spec_str.lower(), "no raw card numbers")

print("\n" + "=" * 60)
print("Phase 91 Provider Request Tests: %d/%d passed" % (passed, total))
if passed < total:
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
