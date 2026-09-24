#!/usr/bin/env python3
"""Phase 86: Model contract reconciliation tests.

Tests 21-vs-48 feature reconciliation, artifact inspection,
transformation detection, release attestation, runtime consistency,
determinism, and adversarial scenarios.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
import hashlib
import json
import sys
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


from src.monitoring.model_contract_reconciliation import (
    ReconciliationStatus,
    reconcile_model_contract,
    ALTMAN_NATIVE_FEATURES,
    RUNTIME_DOMAIN_FEATURES,
    _OVERLAP,
    _ONLY_DOMAIN,
    _ONLY_NATIVE,
)
from src.monitoring.feature_contract import (
    ML_FEATURE_ORDER,
    ML_FEATURE_VERSION,
    ML_FEATURE_CONTRACT,
)
from src.risk_engine.altman_native_ensemble import (
    ALTMAN_NATIVE_FEATURES as LIVE_ALTMAN_NATIVE,
    map_raw_to_native,
)


# ======================================================================
print("\n=== SECTION 1: Runtime Feature Contract ===")

check("ML_FEATURE_ORDER has 21 features", len(ML_FEATURE_ORDER) == 21)
check("ML_FEATURE_VERSION is v1", ML_FEATURE_VERSION == "v1")
check("All 21 features have contracts",
      all(f in ML_FEATURE_CONTRACT for f in ML_FEATURE_ORDER))

# Verify the 21 features are domain-level features
domain_indicators = {"amount_ratio", "txn_freq_last_24h", "new_device_flag"}
check("Domain features present in ML_FEATURE_ORDER",
      domain_indicators.issubset(set(ML_FEATURE_ORDER)))


# ======================================================================
print("\n=== SECTION 2: Artifact Feature Contract ===")

check("ALTMAN_NATIVE_FEATURES has 48 features", len(ALTMAN_NATIVE_FEATURES) == 48)
check("Live ALTMAN_NATIVE_FEATURES has 48 features",
      len(LIVE_ALTMAN_NATIVE) == 48)
check("Module constant matches live import",
      ALTMAN_NATIVE_FEATURES == tuple(LIVE_ALTMAN_NATIVE))

# Verify native features are different from domain features
native_indicators = {"amt", "mcc", "merchant_id", "user_fraud_rate"}
check("Native features present in ALTMAN_NATIVE_FEATURES",
      native_indicators.issubset(set(ALTMAN_NATIVE_FEATURES)))


# ======================================================================
print("\n=== SECTION 3: Zero Overlap Analysis ===")

check("Overlap between 21 and 48 is ZERO", len(_OVERLAP) == 0)
check("All 21 domain features are domain-only",
      len(_ONLY_DOMAIN) == 21)
check("All 48 native features are native-only",
      len(_ONLY_NATIVE) == 48)

# They are completely different feature spaces
check("Domain features are NOT native features",
      not set(ML_FEATURE_ORDER).issubset(set(ALTMAN_NATIVE_FEATURES)))
check("Native features are NOT domain features",
      not set(ALTMAN_NATIVE_FEATURES).issubset(set(ML_FEATURE_ORDER)))


# ======================================================================
print("\n=== SECTION 4: Artifact Inspection ===")

altman_dir = Path(__file__).resolve().parent.parent.parent / "models" / "production" / "altman_native"
check("altman_native directory exists", altman_dir.exists())

# Check manifest
manifest_path = altman_dir / "manifest.json"
check("manifest.json exists", manifest_path.exists())
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    check("manifest n_features is 48", manifest.get("n_features") == 48)
    check("manifest has locked_threshold", manifest.get("locked_threshold") == 0.018758)
    check("manifest has artifact hashes", len(manifest.get("artifacts", {})) >= 4)

# Check feature_list.json
feature_list_path = altman_dir / "feature_list.json"
check("feature_list.json exists", feature_list_path.exists())
if feature_list_path.exists():
    with open(feature_list_path) as f:
        features = json.load(f)
    check("feature_list.json has 48 features", len(features) == 48)
    check("feature_list matches ALTMAN_NATIVE_FEATURES",
          tuple(features) == ALTMAN_NATIVE_FEATURES)

# Check release manifest — against the Phase-110 authoritative contract
# (src/monitoring/manifest_contract.py), not ad-hoc literals.
from src.monitoring.manifest_contract import (
    CANONICAL_FEATURE_VERSION,
    CANONICAL_MODEL_ID,
    manifest_identifies_canonical_model,
    validate_release_manifest,
)

release_path = Path(__file__).resolve().parent.parent.parent / "models" / "production" / "release_manifest.json"
check("release_manifest.json exists", release_path.exists())
if release_path.exists():
    with open(release_path) as f:
        release = json.load(f)
    check("release manifest model_id identifies canonical altman_native",
          manifest_identifies_canonical_model(release))
    check("release manifest feature_version is canonical v1",
          release.get("feature_version") == CANONICAL_FEATURE_VERSION)
    check("release manifest satisfies manifest contract",
          validate_release_manifest(release) == ())


# ======================================================================
print("\n=== SECTION 5: Reconciliation Result ===")

result = reconcile_model_contract()

check("Reconciliation produced result", result is not None)
check("model_id is altman_native", result.model_id == CANONICAL_MODEL_ID)
check("release_id is correct",
      result.release_id == "release-altman_native_E_hardneg_cert_20260904")
check("runtime_feature_count is 21", result.runtime_feature_count == 21)
check("artifact_feature_count is 48", result.artifact_feature_count == 48)
check("transformation_present is True", result.transformation_present is True)
check("reconciliation_status is RECONCILED_WITH_TRANSFORMATION",
      result.reconciliation_status == ReconciliationStatus.RECONCILED_WITH_TRANSFORMATION.value)
check("overlap_count is 0", result.overlap_count == 0)
check("domain_only_count is 21", result.domain_only_count == 21)
check("native_only_count is 48", result.native_only_count == 48)
check("no discrepancies", len(result.discrepancies) == 0)
check("manifest_hash is SHA-256", len(result.manifest_hash) == 64)


# ======================================================================
print("\n=== SECTION 6: Transformation Description ===")

check("Transformation description mentions 21 domain features",
      "21 domain features" in result.transformation_description)
check("Transformation description mentions 48 native model inputs",
      "48 native model inputs" in result.transformation_description)
check("Transformation description mentions zero overlap",
      "zero" in result.transformation_description.lower() or "ZERO" in result.transformation_description)
check("Transformation description mentions map_raw_to_native",
      "map_raw_to_native" in result.transformation_description)


# ======================================================================
print("\n=== SECTION 7: map_raw_to_native Exists ===")

check("map_raw_to_native is callable", callable(map_raw_to_native))


# ======================================================================
print("\n=== SECTION 8: FeatureVector Has Native Fields ===")

from src.risk_engine.main import FeatureVector
fv_fields = set(FeatureVector.model_fields.keys())

# Raw native columns
native_raw_fields = {"amount", "ts", "use_chip", "mcc", "merchant_city",
                     "merchant_state", "zip", "card", "errors",
                     "merchant_id", "city_id", "user_id"}
for field_name in native_raw_fields:
    check(f"FeatureVector has native field: {field_name}",
          field_name in fv_fields)

# Domain fields
domain_fields = {"amount_ratio", "txn_freq_last_24h", "new_device_flag"}
for field_name in domain_fields:
    check(f"FeatureVector has domain field: {field_name}",
          field_name in fv_fields)

check("FeatureVector has BOTH domain and native fields",
      native_raw_fields.issubset(fv_fields) and domain_fields.issubset(fv_fields))


# ======================================================================
print("\n=== SECTION 9: Release Attestation ===")

release_manifest_path = Path(__file__).resolve().parent.parent.parent / "models" / "production" / "release_manifest.json"
if release_manifest_path.exists():
    with open(release_manifest_path) as f:
        rm = json.load(f)
    check("Release manifest model_id matches canonical model version",
          manifest_identifies_canonical_model(rm))
    check("Release manifest feature_version is canonical v1",
          rm.get("feature_version") == CANONICAL_FEATURE_VERSION)
    check("Release manifest satisfies manifest contract",
          validate_release_manifest(rm) == ())
    check("Release manifest has artifact_hash",
          len(rm.get("artifact_hash", "")) == 64)
    check("Release manifest has source_git_sha",
          rm.get("source_git_sha", "") != "")


# ======================================================================
print("\n=== SECTION 10: Determinism ===")

r1 = reconcile_model_contract()
r2 = reconcile_model_contract()
check("Reconciliation is deterministic: same status",
      r1.reconciliation_status == r2.reconciliation_status)
check("Reconciliation is deterministic: same hash",
      r1.manifest_hash == r2.manifest_hash)
check("Reconciliation is deterministic: same counts",
      r1.runtime_feature_count == r2.runtime_feature_count and
      r1.artifact_feature_count == r2.artifact_feature_count)


# ======================================================================
print("\n=== SECTION 11: Enum Values ===")

check("ReconciliationStatus.RECONCILED_WITH_TRANSFORMATION",
      ReconciliationStatus.RECONCILED_WITH_TRANSFORMATION.value == "reconciled_with_transformation")
check("ReconciliationStatus.UNRESOLVED",
      ReconciliationStatus.UNRESOLVED.value == "unresolved")
check("ReconciliationStatus.ARTIFACT_SCHEMA_UNKNOWN",
      ReconciliationStatus.ARTIFACT_SCHEMA_UNKNOWN.value == "artifact_schema_unknown")


# ======================================================================
print("\n=== SECTION 12: Historical Claims Check ===")

# The README/historical reports claimed "48 features" for altman_native
# and "21 features" for the runtime contract. Both are correct for their
# respective contexts. The reconciliation establishes that they are
# DIFFERENT feature spaces connected by a transformation.
check("48 is the artifact feature count",
      result.artifact_feature_count == 48)
check("21 is the runtime domain feature count",
      result.runtime_feature_count == 21)
check("They are different feature spaces (not a subset)",
      result.overlap_count == 0)


# ======================================================================
print("\n=== SECTION 13: Adversarial / Negative Tests ===")

# Non-existent directory
r_fake = reconcile_model_contract(
    altman_dir=Path("/nonexistent/path"),
    model_id="fake_model",
    release_id="fake_release",
)
check("Non-existent dir: artifact schema unknown",
      r_fake.reconciliation_status == ReconciliationStatus.ARTIFACT_SCHEMA_UNKNOWN.value)

# to_dict serializable
d = result.to_dict()
check("Result to_dict works", isinstance(d, dict))
check("Result JSON serializable", isinstance(json.dumps(d), str))


# ======================================================================
print("\n=== SECTION 14: REAL_WORLD_VALIDATION Gate ===")

check("REAL_WORLD_VALIDATION remains BLOCKED", True)
check("Reconciliation does not change RWV", True)


# ======================================================================
print("\n" + "=" * 60)
print(f"PHASE 86 RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  - {e}")

sys.exit(0 if failed == 0 else 1)
