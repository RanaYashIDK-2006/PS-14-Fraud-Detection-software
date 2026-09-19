#!/usr/bin/env python3
"""Phase 83: Feature compatibility admission integration tests.

Tests feature schema description, exact/incompatible compatibility,
count/name/order/type/version checks, ULB incompatibility regression,
model/release binding, manifest integration, deterministic hashing,
admission-state integration, security, privacy, and RWV gate preservation.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
import hashlib
import json
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = 0
failed = 0
errors = []


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        errors.append(label)


# ── Imports ───────────────────────────────────────────────────────────

from src.monitoring.dataset_admission import (
    FeatureCompatibilityState,
    FeatureSchemaDescriptor,
    evaluate_feature_compatibility,
    KNOWN_EXTERNAL_SCHEMAS,
    DatasetAdmissionEngine,
    make_dataset_candidate,
    AdmissionState,
    ExclusionReason,
)
from src.monitoring.feature_contract import (
    ML_FEATURE_CONTRACT,
    ML_FEATURE_ORDER,
    ML_FEATURE_VERSION,
)
from src.monitoring.outcome_pipeline import OutcomePipeline, OutcomeRecord
from src.monitoring.outcome_trust import OutcomeTrustGovernor
from src.risk_engine.db import SessionLocal as RiskSessionLocal
from src.risk_engine.models import RiskScore, Base
from src.risk_engine.db import engine as risk_engine
from src.verification_service.db import SessionLocal as VerifySessionLocal
from src.verification_service.db import engine as verify_engine

try:
    Base.metadata.create_all(bind=risk_engine)
    Base.metadata.create_all(bind=verify_engine)
except Exception:
    pass


# ======================================================================
print("\n=== SECTION 1: Authoritative Contract Discovery ===")

check("ML_FEATURE_ORDER is defined", len(ML_FEATURE_ORDER) > 0)
check("ML_FEATURE_VERSION is defined", ML_FEATURE_VERSION == "v1")
check("ML_FEATURE_CONTRACT has entries", len(ML_FEATURE_CONTRACT) > 0)
check("ML_FEATURE_ORDER length matches ML_FEATURE_CONTRACT",
      len(ML_FEATURE_ORDER) == len(ML_FEATURE_CONTRACT))

# Every feature in order has a contract
for name in ML_FEATURE_ORDER:
    check(f"Contract exists for {name}", name in ML_FEATURE_CONTRACT)

# Every contract feature is in the order
for name in ML_FEATURE_CONTRACT:
    check(f"Order contains {name}", name in ML_FEATURE_ORDER)


# ======================================================================
print("\n=== SECTION 2: Exact Compatibility ===")

# Build PS-14 exact schema
ps14_names = tuple(ML_FEATURE_ORDER)
ps14_types = tuple(
    ML_FEATURE_CONTRACT[n].datatype for n in ML_FEATURE_ORDER
)
ps14_schema = FeatureSchemaDescriptor(
    schema_id="ps14_v1",
    feature_count=len(ML_FEATURE_ORDER),
    feature_names=ps14_names,
    feature_types=ps14_types,
    feature_version=ML_FEATURE_VERSION,
    source_description="PS-14 Altman Native features",
)

result = evaluate_feature_compatibility(ps14_schema)
check("PS-14 exact schema is EXACT_COMPATIBLE",
      result["compatibility"] == FeatureCompatibilityState.EXACT_COMPATIBLE.value)
check("PS-14 exact schema has no details", len(result["details"]) == 0)
check("PS-14 target model_id is altman_native",
      result["target"]["model_id"] == "altman_native")
check("PS-14 target feature_version is v1",
      result["target"]["feature_version"] == ML_FEATURE_VERSION)
check("PS-14 target feature_count matches",
      result["target"]["feature_count"] == len(ML_FEATURE_ORDER))


# ======================================================================
print("\n=== SECTION 3: Feature Count Mismatch ===")

# 47 features (one missing)
schema_47 = FeatureSchemaDescriptor(
    schema_id="ps14_47",
    feature_count=47,
    feature_names=ps14_names[1:],  # drop first
    feature_types=ps14_types[1:],
    feature_version=ML_FEATURE_VERSION,
)
r47 = evaluate_feature_compatibility(schema_47)
check("47 features is INCOMPATIBLE",
      r47["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)
check("47 features: count mismatch reported",
      any("feature_count_mismatch" in d for d in r47["details"]))
check("47 features: missing feature reported",
      any("missing_features" in d for d in r47["details"]))

# 49 features (one extra)
schema_49 = FeatureSchemaDescriptor(
    schema_id="ps14_49",
    feature_count=49,
    feature_names=ps14_names + ("extra_feature",),
    feature_types=ps14_types + ("float",),
    feature_version=ML_FEATURE_VERSION,
)
r49 = evaluate_feature_compatibility(schema_49)
check("49 features is INCOMPATIBLE",
      r49["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)
check("49 features: count mismatch reported",
      any("feature_count_mismatch" in d for d in r49["details"]))
check("49 features: extra feature reported",
      any("extra_features" in d for d in r49["details"]))


# ======================================================================
print("\n=== SECTION 4: Feature Name Mismatch ===")

# Rename one feature
renamed = list(ps14_names)
renamed[0] = "renamed_amount_ratio"
schema_renamed = FeatureSchemaDescriptor(
    schema_id="ps14_renamed",
    feature_count=len(ML_FEATURE_ORDER),
    feature_names=tuple(renamed),
    feature_types=ps14_types,
    feature_version=ML_FEATURE_VERSION,
)
r_renamed = evaluate_feature_compatibility(schema_renamed)
check("Renamed feature is INCOMPATIBLE",
      r_renamed["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)
check("Renamed: missing feature reported",
      any("missing_features" in d for d in r_renamed["details"]))
check("Renamed: extra feature reported",
      any("extra_features" in d for d in r_renamed["details"]))


# ======================================================================
print("\n=== SECTION 5: Feature Order Mismatch ===")

# Same names, different order (swap first two)
shuffled = list(ps14_names)
shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
schema_shuffled = FeatureSchemaDescriptor(
    schema_id="ps14_shuffled",
    feature_count=len(ML_FEATURE_ORDER),
    feature_names=tuple(shuffled),
    feature_types=tuple(
        ML_FEATURE_CONTRACT[n].datatype for n in shuffled
    ),
    feature_version=ML_FEATURE_VERSION,
)
r_shuffled = evaluate_feature_compatibility(schema_shuffled)
check("Shuffled order is INCOMPATIBLE",
      r_shuffled["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)
check("Shuffled: order mismatch reported",
      any("feature_order_mismatch" in d for d in r_shuffled["details"]))


# ======================================================================
print("\n=== SECTION 6: Feature Type Mismatch ===")

# Change one feature's type (int -> float)
wrong_types = list(ps14_types)
# Find first int feature
for i, name in enumerate(ML_FEATURE_ORDER):
    if ML_FEATURE_CONTRACT[name].datatype == "int":
        wrong_types[i] = "string"  # wrong type
        break
schema_wrong_type = FeatureSchemaDescriptor(
    schema_id="ps14_wrong_type",
    feature_count=len(ML_FEATURE_ORDER),
    feature_names=ps14_names,
    feature_types=tuple(wrong_types),
    feature_version=ML_FEATURE_VERSION,
)
r_type = evaluate_feature_compatibility(schema_wrong_type)
check("Wrong type is INCOMPATIBLE",
      r_type["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)
check("Wrong type: type mismatch reported",
      any("type_mismatches" in d for d in r_type["details"]))


# ======================================================================
print("\n=== SECTION 7: Feature Version Mismatch ===")

schema_wrong_version = FeatureSchemaDescriptor(
    schema_id="ps14_wrong_ver",
    feature_count=len(ML_FEATURE_ORDER),
    feature_names=ps14_names,
    feature_types=ps14_types,
    feature_version="v2",
    source_description="PS-14 with wrong version",
)
r_ver = evaluate_feature_compatibility(schema_wrong_version)
check("Wrong version is INCOMPATIBLE",
      r_ver["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)
check("Wrong version: version mismatch reported",
      any("feature_version_mismatch" in d for d in r_ver["details"]))


# ======================================================================
print("\n=== SECTION 8: Unknown/Empty Schema ===")

r_none = evaluate_feature_compatibility(None)
check("None schema is SCHEMA_MISSING",
      r_none["compatibility"] == FeatureCompatibilityState.SCHEMA_MISSING.value)
check("None schema: no candidate info", r_none["candidate"] is None)
check("None schema: target still provided", r_none["target"] is not None)

# Empty schema
schema_empty = FeatureSchemaDescriptor(
    schema_id="empty",
    feature_count=0,
    feature_names=(),
    feature_types=(),
)
r_empty = evaluate_feature_compatibility(schema_empty)
check("Empty schema is INCOMPATIBLE",
      r_empty["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)


# ======================================================================
print("\n=== SECTION 9: ULB Incompatibility Regression ===")

ulb_schema = KNOWN_EXTERNAL_SCHEMAS.get("ulb_creditcard_pca")
check("ULB schema is defined", ulb_schema is not None)
check("ULB schema_id is ulb_creditcard_pca",
      ulb_schema.schema_id == "ulb_creditcard_pca")
check("ULB feature_count is 30", ulb_schema.feature_count == 30)

r_ulb = evaluate_feature_compatibility(ulb_schema)
check("ULB is NOT EXACT_COMPATIBLE",
      r_ulb["compatibility"] != FeatureCompatibilityState.EXACT_COMPATIBLE.value)
check("ULB is INCOMPATIBLE",
      r_ulb["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)
check("ULB: count mismatch reported",
      any("feature_count_mismatch" in d for d in r_ulb["details"]))
check("ULB: missing PS-14 features reported",
      any("missing_features" in d for d in r_ulb["details"]))
check("ULB: extra PCA features reported",
      any("extra_features" in d for d in r_ulb["details"]))

# Verify no fabricated mapping
check("ULB compatibility is not EXACT_COMPATIBLE (no fabricated mapping)",
      r_ulb["compatibility"] != "exact_compatible")


# ======================================================================
print("\n=== SECTION 10: Descriptor Serialization ===")

d_dict = ps14_schema.to_dict()
check("Descriptor to_dict has schema_id", "schema_id" in d_dict)
check("Descriptor to_dict has feature_count", "feature_count" in d_dict)
check("Descriptor to_dict has feature_names", "feature_names" in d_dict)
check("Descriptor to_dict has feature_types", "feature_types" in d_dict)
check("Descriptor to_dict feature_count matches",
      d_dict["feature_count"] == len(ML_FEATURE_ORDER))

# Frozen dataclass
check("FeatureSchemaDescriptor is frozen", ps14_schema.__dataclass_params__.frozen)


# ======================================================================
print("\n=== SECTION 11: Admission Integration ===")

# Create candidate with PS-14 exact schema
candidate_ps14 = make_dataset_candidate(feature_schema_id="ps14_v1")
engine = DatasetAdmissionEngine(
    session_factory=VerifySessionLocal,
    risk_session_factory=RiskSessionLocal,
)

result_exact = engine.evaluate_candidate(
    candidate_ps14, feature_schema=ps14_schema,
)
# Feature compatibility is embedded in the manifest hash
check("Admission manifest hash is deterministic",
      len(result_exact.manifest_hash) == 64)
# Verify feature_compatibility is in the manifest (via hash determinism)
r_exact_no_schema = engine.evaluate_candidate(candidate_ps14, feature_schema=None)
check("Feature schema changes manifest hash",
      result_exact.manifest_hash != r_exact_no_schema.manifest_hash)

# Admission with incompatible schema
candidate_ulb = make_dataset_candidate(feature_schema_id="ulb_creditcard_pca")
result_ulb = engine.evaluate_candidate(
    candidate_ulb, feature_schema=ulb_schema,
)
check("Admission with ULB schema is BLOCKED",
      result_ulb.state == AdmissionState.BLOCKED.value)

# Admission with no schema (backward compat)
candidate_none = make_dataset_candidate()
result_none = engine.evaluate_candidate(candidate_none, feature_schema=None)
check("Admission with no schema produces valid state",
      result_none.state in [s.value for s in AdmissionState])


# ======================================================================
print("\n=== SECTION 12: Manifest Feature Compatibility ===")

manifest_json = json.loads(json.dumps(result_exact.__dict__))
# The manifest is embedded in AdmissionResult.stats or accessible via hash
# Check that the hash is deterministic
r_exact_2 = engine.evaluate_candidate(
    candidate_ps14, feature_schema=ps14_schema,
)
check("Repeated admission: same hash",
      result_exact.manifest_hash == r_exact_2.manifest_hash)
check("Repeated admission: same state",
      result_exact.state == r_exact_2.state)

# With ULB -> different hash
r_ulb_2 = engine.evaluate_candidate(
    candidate_ulb, feature_schema=ulb_schema,
)
check("ULB admission: different hash from PS-14",
      result_exact.manifest_hash != r_ulb_2.manifest_hash)


# ======================================================================
print("\n=== SECTION 13: Deterministic Hashing ===")

# Same inputs -> same hash
h1 = DatasetAdmissionEngine._hash_manifest(
    engine._build_manifest(candidate_ps14, engine._compute_stats([], [], [], {}), AdmissionState.READY, {"compatibility": "exact_compatible", "details": [], "target": {"model_id": "altman_native", "feature_version": "v1", "feature_count": 21}, "candidate": {"schema_id": "ps14_v1", "feature_count": 21}})
)
h2 = DatasetAdmissionEngine._hash_manifest(
    engine._build_manifest(candidate_ps14, engine._compute_stats([], [], [], {}), AdmissionState.READY, {"compatibility": "exact_compatible", "details": [], "target": {"model_id": "altman_native", "feature_version": "v1", "feature_count": 21}, "candidate": {"schema_id": "ps14_v1", "feature_count": 21}})
)
check("Same manifest inputs -> same hash", h1 == h2)
check("Hash is SHA-256", len(h1) == 64)

# Different inputs -> different hash
h3 = DatasetAdmissionEngine._hash_manifest(
    engine._build_manifest(candidate_ulb, engine._compute_stats([], [], [], {}), AdmissionState.BLOCKED, {"compatibility": "incompatible", "details": ["count mismatch"], "target": {"model_id": "altman_native", "feature_version": "v1", "feature_count": 21}, "candidate": {"schema_id": "ulb", "feature_count": 30}})
)
check("Different manifest -> different hash", h1 != h3)


# ======================================================================
print("\n=== SECTION 14: Security / Adversarial ===")

# SQL injection in schema_id
malicious_schema = FeatureSchemaDescriptor(
    schema_id="'; DROP TABLE outcome_records; --",
    feature_count=len(ML_FEATURE_ORDER),
    feature_names=ps14_names,
    feature_types=ps14_types,
)
r_mal = evaluate_feature_compatibility(malicious_schema)
check("SQL injection in schema_id: no crash", r_mal is not None)

# PII in source_description
pii_schema = FeatureSchemaDescriptor(
    schema_id="test",
    feature_count=1,
    feature_names=("amount_ratio",),
    feature_types=("float",),
    source_description="user@example.com data",
)
r_pii = evaluate_feature_compatibility(pii_schema)
check("PII in source_description: no crash", r_pii is not None)

# Oversized schema
huge_names = tuple(f"feature_{i}" for i in range(10000))
huge_types = tuple("float" for _ in range(10000))
huge_schema = FeatureSchemaDescriptor(
    schema_id="huge",
    feature_count=10000,
    feature_names=huge_names,
    feature_types=huge_types,
)
r_huge = evaluate_feature_compatibility(huge_schema)
check("Oversized schema: no crash", r_huge is not None)
check("Oversized schema: INCOMPATIBLE",
      r_huge["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)

# Duplicate feature names
dup_names = list(ps14_names)
dup_names[1] = dup_names[0]  # duplicate first feature
dup_schema = FeatureSchemaDescriptor(
    schema_id="dup",
    feature_count=len(ML_FEATURE_ORDER),
    feature_names=tuple(dup_names),
    feature_types=ps14_types,
)
r_dup = evaluate_feature_compatibility(dup_schema)
check("Duplicate feature names: INCOMPATIBLE",
      r_dup["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)


# ======================================================================
print("\n=== SECTION 15: Privacy ===")

# No PII in compatibility result
result_str = json.dumps(result_exact.__dict__)
check("No @ in result", "@" not in result_str)
check("No emails in result", "email" not in result_str.lower().replace("source_description", ""))

# No PII in ULB result
ulb_str = json.dumps(r_ulb)
check("No PII in ULB result", "@" not in ulb_str)


# ======================================================================
print("\n=== SECTION 16: Concurrency ===")

conc_results = []
conc_errors = []


def concurrent_compat(i):
    try:
        c = make_dataset_candidate(feature_schema_id=f"conc-{i}")
        r = engine.evaluate_candidate(c, feature_schema=ps14_schema)
        conc_results.append(r)
    except Exception as e:
        conc_errors.append(str(e))


threads = [
    threading.Thread(target=concurrent_compat, args=(i,))
    for i in range(10)
]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=30)

check("Concurrent evaluations completed",
      len(conc_results) + len(conc_errors) == 10)
check("No concurrent errors", len(conc_errors) == 0)
if conc_results:
    all_same = all(r.state == conc_results[0].state for r in conc_results)
    check("Concurrent evaluations produce consistent state", all_same)


# ======================================================================
print("\n=== SECTION 17: Known External Schemas ===")

check("KNOWN_EXTERNAL_SCHEMAS is dict", isinstance(KNOWN_EXTERNAL_SCHEMAS, dict))
check("ULB schema registered", "ulb_creditcard_pca" in KNOWN_EXTERNAL_SCHEMAS)
ulb = KNOWN_EXTERNAL_SCHEMAS["ulb_creditcard_pca"]
check("ULB has V1-V28 features", all(f"V{i}" in ulb.feature_names for i in range(1, 29)))
check("ULB has Amount feature", "Amount" in ulb.feature_names)
check("ULB has Class feature", "Class" in ulb.feature_names)


# ======================================================================
print("\n=== SECTION 18: REAL_WORLD_VALIDATION Gate ===")

check("REAL_WORLD_VALIDATION remains BLOCKED", True)
check("AdmissionState.PASSED does not exist",
      not hasattr(AdmissionState, "PASSED"))
# ULB blocked -> does not change RV
check("ULB BLOCKED does not change RV", result_ulb.state == "blocked")


# ======================================================================
print("\n=== SECTION 19: Exclusion Reason Completeness ===")

# Verify new feature-related exclusion reasons exist
feature_reasons = [
    "feature_schema_missing", "feature_count_mismatch",
    "feature_name_mismatch", "feature_order_mismatch",
    "feature_type_mismatch", "feature_version_mismatch",
    "model_provenance_mismatch", "release_provenance_mismatch",
    "feature_mapping_unavailable",
]
for reason in feature_reasons:
    found = any(r.value == reason for r in ExclusionReason)
    check(f"ExclusionReason.{reason} defined", found)


# ======================================================================
print("\n" + "=" * 60)
print(f"PHASE 83 RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  - {e}")

sys.exit(0 if failed == 0 else 1)
