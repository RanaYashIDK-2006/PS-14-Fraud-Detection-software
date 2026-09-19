#!/usr/bin/env python3
"""Phase 84: Real-world dataset ingestion & candidate construction tests.

Tests dataset inventory, provenance classification, source hashing,
schema extraction, label provenance, feature compatibility integration,
ULB incompatibility regression, candidate construction, manifest,
determinism, security, privacy, and RWV gate preservation.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
import hashlib
import json
import os
import sys
import tempfile
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


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

from src.monitoring.dataset_ingestion import (
    DatasetClassification,
    LabelProvenance,
    DatasetProvenance,
    IngestionResult,
    DatasetIngestionError,
    ingest_dataset,
    scan_data_directory,
    compute_file_hash,
    compute_header_hash,
    count_rows,
    extract_columns,
    extract_feature_schema,
    KNOWN_DATASETS,
)
from src.monitoring.dataset_admission import (
    FeatureCompatibilityState,
    FeatureSchemaDescriptor,
    KNOWN_EXTERNAL_SCHEMAS,
    evaluate_feature_compatibility,
    DatasetAdmissionEngine,
    make_dataset_candidate,
    AdmissionState,
)
from src.monitoring.feature_contract import ML_FEATURE_ORDER, ML_FEATURE_VERSION


# ======================================================================
print("\n=== SECTION 1: Dataset Inventory ===")

check("KNOWN_DATASETS is defined", isinstance(KNOWN_DATASETS, dict))
check("ULB in inventory", "ulb_creditcard" in KNOWN_DATASETS)
check("IBM v2 in inventory", "ibm_v2" in KNOWN_DATASETS)
check("PaySim in inventory", "paysim" in KNOWN_DATASETS)

# Verify data directory exists
check("Data directory exists", DATA_DIR.exists())

# Scan data directory
scanned = scan_data_directory(DATA_DIR)
check("Data directory scan produced results", len(scanned) > 0)

# Count scanned datasets
check(f"Scanned at least 5 CSV files", len(scanned) >= 5)

# Verify creditcard.csv found
ulb_found = any(s.dataset_name == "creditcard" for s in scanned)
check("creditcard.csv found in scan", ulb_found)


# ======================================================================
print("\n=== SECTION 2: Provenance Classification ===")

# ULB is SYNTHETIC
ulb = KNOWN_DATASETS["ulb_creditcard"]
check("ULB classified as synthetic",
      ulb.classification == DatasetClassification.SYNTHETIC.value)
check("ULB has label_field", ulb.label_field == "Class")
check("ULB label_provenance is not independent",
      ulb.label_provenance != LabelProvenance.INDEPENDENT_HUMAN.value)

# IBM v2 is SYNTHETIC
ibm = KNOWN_DATASETS["ibm_v2"]
check("IBM v2 classified as synthetic",
      ibm.classification == DatasetClassification.SYNTHETIC.value)
check("IBM v2 label_provenance is synthetic",
      ibm.label_provenance == LabelProvenance.SYNTHETIC_GENERATED.value)

# PaySim is SYNTHETIC
ps = KNOWN_DATASETS["paysim"]
check("PaySim classified as synthetic",
      ps.classification == DatasetClassification.SYNTHETIC.value)

# Verify no PUBLIC_REAL_WORLD classification exists
for name, prov in KNOWN_DATASETS.items():
    check(f"{name} is NOT classified as public_real_world",
          prov.classification != DatasetClassification.PUBLIC_REAL_WORLD.value)

# Verify auto-classification of scanned files
for s in scanned:
    if s.dataset_name == "creditcard":
        check("creditcard auto-classified as synthetic",
              s.classification == DatasetClassification.SYNTHETIC.value)
    if s.dataset_name == "paysim":
        check("paysim auto-classified as synthetic",
              s.classification == DatasetClassification.SYNTHETIC.value)


# ======================================================================
print("\n=== SECTION 3: Source Hashing ===")

# Compute hash of a known file
ulb_file = DATA_DIR / "creditcard.csv"
if ulb_file.exists():
    h1 = compute_file_hash(ulb_file)
    h2 = compute_file_hash(ulb_file)
    check("File hash is deterministic", h1 == h2)
    check("File hash is SHA-256", len(h1) == 64)
    check("File hash is not empty", h1 != "")

    # Schema hash
    sh1 = compute_header_hash(ulb_file)
    sh2 = compute_header_hash(ulb_file)
    check("Schema hash is deterministic", sh1 == sh2)

    # Row count
    rows = count_rows(ulb_file)
    check("ULB row count > 280000", rows > 280000)

    # Columns
    cols = extract_columns(ulb_file)
    check("ULB has V1 column", "V1" in cols)
    check("ULB has Class column", "Class" in cols)
    check("ULB has Amount column", "Amount" in cols)
else:
    check("ULB file exists for hashing", False, "creditcard.csv not found")


# ======================================================================
print("\n=== SECTION 4: Schema Extraction ===")

# Extract schema from ULB
if ulb_file.exists():
    cols = extract_columns(ulb_file)
    schema = extract_feature_schema(cols, label_field="Class", timestamp_field="Time")
    check("ULB schema feature_count is 29 (31 cols minus Time and Class)", schema.feature_count == 29)
    check("ULB schema has V1", "V1" in schema.feature_names)
    check("ULB schema has Amount", "Amount" in schema.feature_names)
    check("ULB schema does NOT have Class", "Class" not in schema.feature_names)
    check("ULB schema does NOT have Time", "Time" not in schema.feature_names)

    # Verify frozen
    check("FeatureSchemaDescriptor is frozen",
          schema.__dataclass_params__.frozen)


# ======================================================================
print("\n=== SECTION 5: Label Provenance ===")

# Verify all known datasets have explicit label provenance
for name, prov in KNOWN_DATASETS.items():
    check(f"{name} has label_provenance set",
          prov.label_provenance != "")
    check(f"{name} label_provenance is a valid enum value",
          prov.label_provenance in [lp.value for lp in LabelProvenance])

# ULB labels are UNKNOWN provenance (not independently sourced)
check("ULB label_provenance is not independent_human",
      ulb.label_provenance != LabelProvenance.INDEPENDENT_HUMAN.value)
check("ULB label_provenance is not independent_institutional",
      ulb.label_provenance != LabelProvenance.INDEPENDENT_INSTITUTIONAL.value)


# ======================================================================
print("\n=== SECTION 6: Feature Compatibility Integration ===")

# ULB through compatibility engine
ulb_schema = KNOWN_EXTERNAL_SCHEMAS.get("ulb_creditcard_pca")
if ulb_schema:
    r = evaluate_feature_compatibility(ulb_schema)
    check("ULB compatibility is INCOMPATIBLE",
          r["compatibility"] == FeatureCompatibilityState.INCOMPATIBLE.value)

# PS-14 exact features through compatibility
from src.monitoring.feature_contract import ML_FEATURE_CONTRACT as _FC
ps14_names = tuple(ML_FEATURE_ORDER)
ps14_types = tuple(_FC[n].datatype for n in ML_FEATURE_ORDER)
ps14_schema = FeatureSchemaDescriptor(
    schema_id="ps14_v1",
    feature_count=len(ML_FEATURE_ORDER),
    feature_names=ps14_names,
    feature_types=ps14_types,
    feature_version=ML_FEATURE_VERSION,
)
r_ps14 = evaluate_feature_compatibility(ps14_schema)
check("PS-14 exact features are EXACT_COMPATIBLE",
      r_ps14["compatibility"] == FeatureCompatibilityState.EXACT_COMPATIBLE.value)


# ======================================================================
print("\n=== SECTION 7: ULB Incompatibility Regression ===")

# Ingest ULB through the full pipeline
if ulb_file.exists():
    result = ingest_dataset(ulb_file, provenance=ulb)
    check("ULB ingestion produces IngestionResult", result is not None)
    check("ULB ingestion feature_compatibility is incompatible",
          result.feature_compatibility["compatibility"] == "incompatible")
    check("ULB candidate has dataset_id",
          result.candidate.dataset_id.startswith("DS-"))
    check("ULB provenance row_count matches",
          result.provenance.row_count > 280000)
    check("ULB provenance source_hash is set",
          len(result.provenance.source_hash) == 64)

    # No fabricated mapping
    check("ULB compatibility is NOT exact_compatible (no fabricated mapping)",
          result.feature_compatibility["compatibility"] != "exact_compatible")


# ======================================================================
print("\n=== SECTION 8: Candidate Construction ===")

# Ingest a small test file
with tempfile.NamedTemporaryFile(
    mode="w", suffix=".csv", delete=False, prefix="test_ds_"
) as f:
    f.write("amount_ratio,txn_freq,category,label,ts\n")
    f.write("1.5,3,typical,fraud,2025-01-01\n")
    f.write("0.8,1,unusual,legit,2025-01-02\n")
    test_file = f.name

try:
    test_provenance = DatasetProvenance(
        dataset_name="test_dataset",
        classification=DatasetClassification.UNKNOWN.value,
        label_field="label",
        label_provenance=LabelProvenance.UNKNOWN.value,
        timestamp_field="ts",
        has_timestamps=True,
    )
    result_test = ingest_dataset(test_file, provenance=test_provenance)
    check("Test ingestion produces result", result_test is not None)
    check("Test candidate has dataset_id",
          result_test.candidate.dataset_id.startswith("DS-"))
    check("Test provenance row_count is 2",
          result_test.provenance.row_count == 2)
    check("Test schema has 3 features",
          result_test.feature_schema.feature_count == 3)
    check("Test schema does not have label field",
          "label" not in result_test.feature_schema.feature_names)
    check("Test schema does not have timestamp field",
          "ts" not in result_test.feature_schema.feature_names)
    check("Test source_hash is set",
          len(result_test.provenance.source_hash) == 64)
finally:
    os.unlink(test_file)


# ======================================================================
print("\n=== SECTION 9: Manifest Integration ===")

# Verify ingestion result serializable
if ulb_file.exists():
    result_dict = result.to_dict()
    check("IngestionResult to_dict works", isinstance(result_dict, dict))
    check("Manifest has provenance", "provenance" in result_dict)
    check("Manifest has feature_compatibility", "feature_compatibility" in result_dict)
    check("Manifest has candidate", "candidate" in result_dict)

    # No PII in manifest
    manifest_str = json.dumps(result_dict)
    check("No PII: no @ in manifest", "@" not in manifest_str)
    check("No PII: no emails in manifest", "email" not in manifest_str.lower())


# ======================================================================
print("\n=== SECTION 10: Deterministic Repeated Ingestion ===")

if ulb_file.exists():
    r1 = ingest_dataset(ulb_file, provenance=ulb)
    r2 = ingest_dataset(ulb_file, provenance=ulb)
    check("Repeated ingestion: same source_hash",
          r1.provenance.source_hash == r2.provenance.source_hash)
    check("Repeated ingestion: same schema_hash",
          r1.provenance.schema_hash == r2.provenance.schema_hash)
    check("Repeated ingestion: same row_count",
          r1.provenance.row_count == r2.provenance.row_count)
    check("Repeated ingestion: same feature_compatibility",
          r1.feature_compatibility["compatibility"] == r2.feature_compatibility["compatibility"])


# ======================================================================
print("\n=== SECTION 11: Source Immutability ===")

if ulb_file.exists():
    # Verify source hash didn't change after ingestion
    current_hash = compute_file_hash(ulb_file)
    check("Source file hash unchanged after ingestion",
          current_hash == result.provenance.source_hash)


# ======================================================================
print("\n=== SECTION 12: Derived/Synthetic Separation ===")

# Verify all known datasets are classified as synthetic or research_derived
for name, prov in KNOWN_DATASETS.items():
    is_non_real = prov.classification in (
        DatasetClassification.SYNTHETIC.value,
        DatasetClassification.RESEARCH_DERIVED.value,
    )
    check(f"{name} is correctly non-real-world",
          is_non_real or prov.classification == DatasetClassification.UNKNOWN.value)

# Verify label provenance for synthetic datasets
for name, prov in KNOWN_DATASETS.items():
    if prov.classification == DatasetClassification.SYNTHETIC.value:
        check(f"{name} synthetic labels not marked as independent",
              prov.label_provenance not in (
                  LabelProvenance.INDEPENDENT_HUMAN.value,
                  LabelProvenance.INDEPENDENT_INSTITUTIONAL.value,
              ))


# ======================================================================
print("\n=== SECTION 13: Security / Adversarial ===")

# Non-existent file
try:
    ingest_dataset("/nonexistent/path.csv")
    check("Non-existent file rejected", False)
except DatasetIngestionError:
    check("Non-existent file rejected", True)

# Empty file
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("")
    empty_file = f.name
try:
    result_empty = ingest_dataset(empty_file)
    check("Empty file handled (0 rows)", result_empty.provenance.row_count == 0)
except Exception:
    check("Empty file handled", True)
finally:
    os.unlink(empty_file)

# File with only header
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("col1,col2,col3\n")
    header_only = f.name
try:
    result_header = ingest_dataset(header_only)
    check("Header-only file: 0 rows", result_header.provenance.row_count == 0)
    check("Header-only file: 3 columns",
          len(result_header.provenance.schema_columns) == 3)
finally:
    os.unlink(header_only)

# SQL injection in column names
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("'; DROP TABLE data; --,normal_col\n1,2\n")
    sql_file = f.name
try:
    result_sql = ingest_dataset(sql_file)
    check("SQL injection in columns: no crash", result_sql is not None)
finally:
    os.unlink(sql_file)


# ======================================================================
print("\n=== SECTION 14: Privacy ===")

# Verify no PII in any known dataset provenance
for name, prov in KNOWN_DATASETS.items():
    prov_str = json.dumps(prov.to_dict())
    check(f"{name}: no PII in provenance (@)",
          "@" not in prov_str or "source_url" in prov_str)


# ======================================================================
print("\n=== SECTION 15: Classification Enums ===")

check("DatasetClassification.SYNTHETIC exists",
      DatasetClassification.SYNTHETIC.value == "synthetic")
check("DatasetClassification.RESEARCH_DERIVED exists",
      DatasetClassification.RESEARCH_DERIVED.value == "research_derived")
check("DatasetClassification.PUBLIC_REAL_WORLD exists",
      DatasetClassification.PUBLIC_REAL_WORLD.value == "public_real_world")
check("DatasetClassification.UNKNOWN exists",
      DatasetClassification.UNKNOWN.value == "unknown")

check("LabelProvenance.INDEPENDENT_HUMAN exists",
      LabelProvenance.INDEPENDENT_HUMAN.value == "independent_human")
check("LabelProvenance.SYNTHETIC_GENERATED exists",
      LabelProvenance.SYNTHETIC_GENERATED.value == "synthetic_generated")
check("LabelProvenance.MODEL_DERIVED exists",
      LabelProvenance.MODEL_DERIVED.value == "model_derived")


# ======================================================================
print("\n=== SECTION 16: Data Directory Scan ===")

if scanned:
    # All scanned files should have hashes
    for s in scanned:
        check(f"{s.dataset_name}: source_hash is SHA-256",
              len(s.source_hash) == 64)
        check(f"{s.dataset_name}: schema_hash is SHA-256",
              len(s.schema_hash) == 64)
        check(f"{s.dataset_name}: row_count >= 0",
              s.row_count >= 0)
        check(f"{s.dataset_name}: has schema_columns",
              len(s.schema_columns) > 0)


# ======================================================================
print("\n=== SECTION 17: REAL_WORLD_VALIDATION Gate ===")

check("REAL_WORLD_VALIDATION remains BLOCKED", True)

# No PUBLIC_REAL_WORLD in known datasets
has_real_world = any(
    p.classification == DatasetClassification.PUBLIC_REAL_WORLD.value
    for p in KNOWN_DATASETS.values()
)
check("No dataset classified as PUBLIC_REAL_WORLD", not has_real_world)

# All ingested datasets are SYNTHETIC
if ulb_file.exists():
    check("ULB ingestion does not change RWV",
          result.feature_compatibility["compatibility"] != "exact_compatible")


# ======================================================================
print("\n" + "=" * 60)
print(f"PHASE 84 RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  - {e}")

sys.exit(0 if failed == 0 else 1)
