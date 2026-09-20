"""Phase 88: IEEE-CIS forensic eligibility audit tests.

Deterministic, adversarial tests verifying the forensic audit module
correctly identifies IEEE-CIS as feature-incompatible and BLOCKED
for PS-14 real-world validation.
"""
from __future__ import annotations

import hashlib
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES as NATIVE48
from src.monitoring.ieee_cis_forensic_audit import (
    DERIVABILITY_MATRIX,
    ENTITY_CONTINUITY,
    IEEE_CIS_PROVENANCE,
    LABEL_PROVENANCE,
    DerivabilityClassification,
    TemporalLeakageClassification,
    OverallReadiness,
    FeatureDerivability,
    get_derivability_summary,
    get_leakage_summary,
    assess_phase85_readiness,
    run_forensic_audit,
)

passed = 0
total = 0


def check(condition: bool, desc: str):
    global passed, total
    total += 1
    if condition:
        passed += 1
        print(f"  OK: {desc}")
    else:
        print(f"  FAIL: {desc}")


# ══════════════════════════════════════════════════════════════════════
# A. Deterministic output
# ══════════════════════════════════════════════════════════════════════
print("\n=== A. Deterministic output ===")
r1 = run_forensic_audit()
r2 = run_forensic_audit()
check(r1.manifest_hash == r2.manifest_hash, "identical manifest hash on repeated runs")
check(r1.audit_id == r2.audit_id == "IEEE-CIS-FORENSIC-AUDIT-88", "audit_id is constant")
check(r1.audit_version == r2.audit_version == "phase88_v1", "audit_version is constant")
check(len(r1.blockers) == len(r2.blockers), "same blocker count on repeated runs")
check(r1.to_dict() == r2.to_dict(), "full result dict is deterministic")

# ══════════════════════════════════════════════════════════════════════
# B. Canonical 48-feature list alignment
# ══════════════════════════════════════════════════════════════════════
print("\n=== B. Canonical 48-feature list alignment ===")
matrix_features = [e.feature_name for e in DERIVABILITY_MATRIX]
check(len(DERIVABILITY_MATRIX) == 48, f"matrix has 48 entries (got {len(DERIVABILITY_MATRIX)})")
check(len(NATIVE48) == 48, f"ALTMAN_NATIVE_FEATURES has 48 entries (got {len(NATIVE48)})")
check(matrix_features == list(NATIVE48), "matrix features exactly match ALTMAN_NATIVE_FEATURES order")

# ══════════════════════════════════════════════════════════════════════
# C. All 48 features represented exactly once
# ══════════════════════════════════════════════════════════════════════
print("\n=== C. All 48 features represented exactly once ===")
check(len(matrix_features) == len(set(matrix_features)), "no duplicate feature names")
check(set(matrix_features) == set(NATIVE48), "feature name sets are identical")

# ══════════════════════════════════════════════════════════════════════
# D. Feature indices are 0-47
# ══════════════════════════════════════════════════════════════════════
print("\n=== D. Feature indices ===")
indices = [e.feature_index for e in DERIVABILITY_MATRIX]
check(indices == list(range(48)), "indices are exactly 0-47 in order")

# ══════════════════════════════════════════════════════════════════════
# E. Derivability summary
# ══════════════════════════════════════════════════════════════════════
print("\n=== E. Derivability summary ===")
s = get_derivability_summary()
check(s.get("direct", 0) == 1, f"exactly 1 DIRECT feature (amt), got {s.get('direct', 0)}")
check(s.get("deterministic_derivation", 0) == 2, f"exactly 2 DETERMINISTIC_DERIVATION (log_amt, amt_sq), got {s.get('deterministic_derivation', 0)}")
check(s.get("not_derivable", 0) == 41, f"41 NOT_DERIVABLE, got {s.get('not_derivable', 0)}")
check(s.get("leakage_risk", 0) == 3, f"3 LEAKAGE_RISK (fraud rates), got {s.get('leakage_risk', 0)}")
check(s.get("unknown", 0) == 1, f"1 UNKNOWN (card_id), got {s.get('unknown', 0)}")
total_counted = sum(s.values())
check(total_counted == 48, f"all 48 features classified (sum={total_counted})")

# ══════════════════════════════════════════════════════════════════════
# F. Leakage classification
# ══════════════════════════════════════════════════════════════════════
print("\n=== F. Leakage classification ===")
l = get_leakage_summary()
check(l.get("safe_pre_transaction", 0) == 3, f"3 SAFE_PRE_TRANSACTION (amt/log_amt/amt_sq), got {l.get('safe_pre_transaction', 0)}")
check(l.get("not_derivable", 0) == 41, f"41 NOT_DERIVABLE, got {l.get('not_derivable', 0)}")
check(l.get("leakage_risk", 0) == 3, f"3 LEAKAGE_RISK, got {l.get('leakage_risk', 0)}")
check(l.get("unknown", 0) == 1, f"1 UNKNOWN, got {l.get('unknown', 0)}")

# ══════════════════════════════════════════════════════════════════════
# G. Fraud-rate features are LEAKAGE_RISK, not safe
# ══════════════════════════════════════════════════════════════════════
print("\n=== G. Fraud-rate features are LEAKAGE_RISK ===")
fraud_rate_features = {"user_fraud_rate", "merch_fraud_rate", "city_fraud_rate"}
for entry in DERIVABILITY_MATRIX:
    if entry.feature_name in fraud_rate_features:
        check(
            entry.classification == DerivabilityClassification.LEAKAGE_RISK.value,
            f"{entry.feature_name} classified as LEAKAGE_RISK",
        )
        check(
            entry.leakage_classification == TemporalLeakageClassification.LEAKAGE_RISK.value,
            f"{entry.feature_name} leakage classified as LEAKAGE_RISK",
        )

# ══════════════════════════════════════════════════════════════════════
# H. Time features NOT DERIVABLE (TransactionDT is relative)
# ══════════════════════════════════════════════════════════════════════
print("\n=== H. Time features NOT DERIVABLE ===")
time_features = {"hr", "mn", "dow", "Month", "Day", "hour_sin", "hour_cos", "is_night", "is_business_hours"}
for entry in DERIVABILITY_MATRIX:
    if entry.feature_name in time_features:
        check(
            entry.classification == DerivabilityClassification.NOT_DERIVABLE.value,
            f"{entry.feature_name} is NOT_DERIVABLE (TransactionDT is relative)",
        )

# ══════════════════════════════════════════════════════════════════════
# I. MCC features NOT DERIVABLE (no MCC in IEEE-CIS)
# ══════════════════════════════════════════════════════════════════════
print("\n=== I. MCC features NOT DERIVABLE ===")
mcc_features = {"mcc", "mcc_high", "mcc_restaurant", "mcc_gas", "mcc_grocery", "mcc_travel", "mcc_online"}
for entry in DERIVABILITY_MATRIX:
    if entry.feature_name in mcc_features:
        check(
            entry.classification == DerivabilityClassification.NOT_DERIVABLE.value,
            f"{entry.feature_name} is NOT_DERIVABLE (no MCC in IEEE-CIS)",
        )

# ══════════════════════════════════════════════════════════════════════
# J. Transaction type features NOT DERIVABLE
# ══════════════════════════════════════════════════════════════════════
print("\n=== J. Transaction type features NOT DERIVABLE ===")
txn_type_features = {"chip", "is_online", "is_swipe"}
for entry in DERIVABILITY_MATRIX:
    if entry.feature_name in txn_type_features:
        check(
            entry.classification == DerivabilityClassification.NOT_DERIVABLE.value,
            f"{entry.feature_name} is NOT_DERIVABLE (no use_chip in IEEE-CIS)",
        )

# ══════════════════════════════════════════════════════════════════════
# K. Amount features are derivable
# ══════════════════════════════════════════════════════════════════════
print("\n=== K. Amount features are derivable ===")
amt_features = {"amt": "direct", "log_amt": "deterministic_derivation", "amt_sq": "deterministic_derivation"}
for entry in DERIVABILITY_MATRIX:
    if entry.feature_name in amt_features:
        expected = amt_features[entry.feature_name]
        check(
            entry.classification == expected,
            f"{entry.feature_name} classified as {expected}",
        )

# ══════════════════════════════════════════════════════════════════════
# L. Entity continuity
# ══════════════════════════════════════════════════════════════════════
print("\n=== L. Entity continuity ===")
check(len(ENTITY_CONTINUITY) >= 4, f"at least 4 entity assessments (got {len(ENTITY_CONTINUITY)})")
defensible = [e for e in ENTITY_CONTINUITY if e.continuity_defensible]
check(len(defensible) == 0, f"no IEEE-CIS field has defensible continuity (got {len(defensible)})")

# ══════════════════════════════════════════════════════════════════════
# M. Provenance evidence
# ══════════════════════════════════════════════════════════════════════
print("\n=== M. Provenance evidence ===")
check(IEEE_CIS_PROVENANCE.dataset_type == "real_transactions", "dataset_type is real_transactions")
check(IEEE_CIS_PROVENANCE.independent_from_ps14 is True, "independent_from_ps14 is True")
check(IEEE_CIS_PROVENANCE.timestamp_is_relative is True, "timestamp_is_relative is True")
check(IEEE_CIS_PROVENANCE.has_fraud_label is True, "has_fraud_label is True")
check("Vesta" in IEEE_CIS_PROVENANCE.original_provider, "provider is Vesta Corporation")
check(IEEE_CIS_PROVENANCE.row_count_train > 500000, "train row count > 500K")

# ══════════════════════════════════════════════════════════════════════
# N. Label provenance evidence
# ══════════════════════════════════════════════════════════════════════
print("\n=== N. Label provenance evidence ===")
check(LABEL_PROVENANCE.label_field == "isFraud", "label field is isFraud")
check(
    LABEL_PROVENANCE.label_type not in ("independent_human", "independent_institutional"),
    f"label type ({LABEL_PROVENANCE.label_type}) is NOT independent_human/independent_institutional",
)
check(LABEL_PROVENANCE.independent_of_ps14 is True, "labels are independent of PS-14")
check(len(LABEL_PROVENANCE.blockers) > 0, "label provenance has blockers")

# ══════════════════════════════════════════════════════════════════════
# O. Phase 85 readiness assessment
# ══════════════════════════════════════════════════════════════════════
print("\n=== O. Phase 85 readiness assessment ===")
readiness = assess_phase85_readiness()
check(readiness.readiness == OverallReadiness.BLOCKED.value, f"readiness is BLOCKED (got {readiness.readiness})")
check(readiness.feature_compatibility_ok is False, "feature_compatibility_ok is False")
check(readiness.label_provenance_ok is False, "label_provenance_ok is False")
check(readiness.temporal_validity_ok is False, "temporal_validity_ok is False")
check(len(readiness.blocking_reasons) >= 3, f"at least 3 blocking reasons (got {len(readiness.blocking_reasons)})")

# ══════════════════════════════════════════════════════════════════════
# P. Full audit result
# ══════════════════════════════════════════════════════════════════════
print("\n=== P. Full audit result ===")
result = run_forensic_audit()
check(result.feature_compatibility["compatibility_result"] == "INCOMPATIBLE", "feature compatibility is INCOMPATIBLE")
check(result.phase85_readiness["readiness"] == "blocked", "readiness is blocked")
check(len(result.blockers) >= 5, f"at least 5 blockers (got {len(result.blockers)})")
check(len(result.unresolved_questions) >= 1, "has unresolved questions")
check(len(result.manifest_hash) == 64, "manifest hash is SHA-256 (64 hex chars)")

# ══════════════════════════════════════════════════════════════════════
# Q. Manifest determinism
# ══════════════════════════════════════════════════════════════════════
print("\n=== Q. Manifest determinism ===")
r3 = run_forensic_audit()
check(result.manifest_hash == r3.manifest_hash, "manifest hash is deterministic across runs")

# Verify hash is actually computed from content
manifest_content = {
    "audit_id": "IEEE-CIS-FORENSIC-AUDIT-88",
    "audit_version": "phase88_v1",
    "provenance_status": "verified_independent",
    "label_provenance_type": LABEL_PROVENANCE.label_type,
    "feature_compatibility": "INCOMPATIBLE",
    "total_features": 48,
    "directly_derivable": 3,
    "not_derivable": 41,
    "leakage_risk": 3,
    "readiness": "blocked",
    "blocker_count": len(result.blockers),
    "independent_from_ps14": True,
}
canonical = json.dumps(manifest_content, sort_keys=True, separators=(",", ":"))
expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
check(result.manifest_hash == expected_hash, "manifest hash matches manual SHA-256 computation")

# ══════════════════════════════════════════════════════════════════════
# R. No fabricated mappings
# ══════════════════════════════════════════════════════════════════════
print("\n=== R. No fabricated mappings ===")
for entry in DERIVABILITY_MATRIX:
    if entry.classification in (
        DerivabilityClassification.DIRECT.value,
        DerivabilityClassification.DETERMINISTIC_DERIVATION.value,
    ):
        check(
            "IEEE-CIS" in entry.notes or "TransactionAmt" in entry.source_field_or_history,
            f"{entry.feature_name} derivation references IEEE-CIS source (not fabricated)",
        )

# ══════════════════════════════════════════════════════════════════════
# S. Derivability classifications are valid enum values
# ══════════════════════════════════════════════════════════════════════
print("\n=== S. Valid enum values ===")
valid_deriv = {c.value for c in DerivabilityClassification}
valid_leak = {c.value for c in TemporalLeakageClassification}
for entry in DERIVABILITY_MATRIX:
    check(
        entry.classification in valid_deriv,
        f"{entry.feature_name} classification '{entry.classification}' is valid",
    )
    check(
        entry.leakage_classification in valid_leak,
        f"{entry.feature_name} leakage '{entry.leakage_classification}' is valid",
    )

# ══════════════════════════════════════════════════════════════════════
# T. Temporal structure
# ══════════════════════════════════════════════════════════════════════
print("\n=== T. Temporal structure ===")
ts = result.temporal_structure
check(ts["is_absolute"] is False, "TransactionDT is not absolute")
check(ts["reference_datetime_published"] is False, "reference datetime not published")
check(ts["can_compute_clock_features"] is False, "cannot compute clock features")
check(ts["chronological_ordering"] is True, "chronological ordering possible")

# ══════════════════════════════════════════════════════════════════════
# U. Independence assessment
# ══════════════════════════════════════════════════════════════════════
print("\n=== U. Independence assessment ===")
ia = result.independence_assessment
check(ia["independent_from_ps14"] is True, "IEEE-CIS is independent from PS-14")
check(ia["training_data_used"] is False, "not used in PS-14 training")
check(ia["threshold_selection_used"] is False, "not used in threshold selection")
check(ia["calibration_used"] is False, "not used in calibration")
check(ia["feature_engineering_used"] is False, "not used in feature engineering")

# ══════════════════════════════════════════════════════════════════════
# V. REAL_WORLD_VALIDATION gate preservation
# ══════════════════════════════════════════════════════════════════════
print("\n=== V. REAL_WORLD_VALIDATION gate preservation ===")
from src.monitoring.outcome_pipeline import OutcomeSource, SOURCE_TRUST
check(
    SOURCE_TRUST.get(OutcomeSource.SYSTEM_TEST.value) not in ("production", "production_trusted"),
    "SYSTEM_TEST trust level is not production",
)
check(
    SOURCE_TRUST.get(OutcomeSource.SYNTHETIC.value) not in ("production", "production_trusted"),
    "SYNTHETIC trust level is not production",
)
check(
    result.phase85_readiness["readiness"] == "blocked",
    "IEEE-CIS readiness is blocked (RWV gate preserved)",
)

# ══════════════════════════════════════════════════════════════════════
# W. Blockers contain key incompatibilities
# ══════════════════════════════════════════════════════════════════════
print("\n=== W. Key blockers documented ===")
blocker_text = " ".join(result.blockers).lower()
check("not derivable" in blocker_text, "blockers mention not-derivable features")
check("transactiondt" in blocker_text or "relative" in blocker_text, "blockers mention TransactionDT limitation")
check("mcc" in blocker_text, "blockers mention missing MCC")
check("label" in blocker_text, "blockers mention label provenance")

# ══════════════════════════════════════════════════════════════════════
# X. No PII in audit output
# ══════════════════════════════════════════════════════════════════════
print("\n=== X. No PII in audit output ===")
result_str = json.dumps(result.to_dict())
check("@" not in result_str or "@" not in result_str.split("email")[0] if "email" in result_str else True, "no email addresses in audit output")
check("ssn" not in result_str.lower(), "no SSN in audit output")
check("credit card" not in result_str.lower(), "no credit card numbers in audit output")
check("password" not in result_str.lower(), "no passwords in audit output")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Phase 88 IEEE-CIS Forensic Audit Tests: {passed}/{total} passed")
if passed < total:
    print("FAILURES DETECTED")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
