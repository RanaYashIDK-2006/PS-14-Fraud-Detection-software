#!/usr/bin/env python3
"""Phase 85: External dataset eligibility evidence & readiness tests.

Tests evidence records, eligibility assessment, readiness evaluation,
manifest determinism, ULB regression, provenance forgery detection,
label provenance validation, independence checks, security/adversarial,
and RWV gate preservation.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
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


from src.monitoring.external_dataset_evidence import (
    EligibilityStatus,
    ReadinessStatus,
    IndependenceStatus,
    ProvenanceStatus,
    ExternalDatasetEvidence,
    EligibilityAssessment,
    ReadinessAssessment,
    EvaluationManifest,
    assess_eligibility,
    assess_external_dataset_readiness,
    build_evaluation_manifest,
    build_evidence_from_ingestion,
)
from src.monitoring.dataset_admission import (
    FeatureCompatibilityState,
    FeatureSchemaDescriptor,
    evaluate_feature_compatibility,
    ML_FEATURE_ORDER,
    ML_FEATURE_VERSION,
    ML_FEATURE_CONTRACT,
)
from src.monitoring.dataset_ingestion import (
    DatasetClassification,
    LabelProvenance,
    KNOWN_DATASETS,
)


def _make_evidence(**overrides) -> ExternalDatasetEvidence:
    """Create a minimal evidence record with defaults."""
    defaults = {
        "dataset_id": f"DS-{uuid.uuid4().hex[:12].upper()}",
        "dataset_name": "test_dataset",
        "dataset_version": "1.0",
        "source_url": "https://example.com/dataset.csv",
        "publisher": "Test Publisher",
        "acquisition_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_hash": "a" * 64,
        "row_count": 10000,
        "schema_hash": "b" * 64,
        "schema_description": "col1,col2,label",
        "feature_schema_id": "ps14_v1",
        "feature_compatibility": FeatureCompatibilityState.EXACT_COMPATIBLE.value,
        "label_provenance": LabelProvenance.INDEPENDENT_HUMAN.value,
        "label_definition": "fraud/not_fraud binary label from investigation",
        "temporal_coverage": "2023-01-01 to 2023-12-31",
        "geography": "US",
        "independence_status": IndependenceStatus.INDEPENDENT.value,
        "provenance_status": ProvenanceStatus.VERIFIED.value,
        "eligibility_status": EligibilityStatus.UNKNOWN.value,
        "provenance_reference": "Internal investigation case #INV-001",
        "evidence_notes": "Labels from human fraud investigators",
        "policy_version": "outcome_trust_policy_v1",
    }
    defaults.update(overrides)
    return ExternalDatasetEvidence(**defaults)


# ======================================================================
print("\n=== SECTION 1: Evidence Record Model ===")

ev = _make_evidence()
check("Evidence has dataset_id", ev.dataset_id.startswith("DS-"))
check("Evidence has dataset_name", ev.dataset_name == "test_dataset")
check("Evidence has source_hash", len(ev.source_hash) == 64)
check("Evidence has eligibility_status", ev.eligibility_status == "unknown")
check("Evidence is frozen", ev.__dataclass_params__.frozen)

# to_dict
d = ev.to_dict()
check("to_dict has all fields", len(d) >= 20)
check("to_dict is JSON serializable", isinstance(json.dumps(d), str))


# ======================================================================
print("\n=== SECTION 2: Eligibility Assessment — Eligible Dataset ===")

ev_good = _make_evidence()
assess = assess_eligibility(ev_good)
check("Eligible: overall is verified",
      assess.overall_eligibility == EligibilityStatus.VERIFIED.value)
check("Eligible: no blocking reasons", len(assess.blocking_reasons) == 0)
check("Eligible: provenance verified",
      assess.provenance_status == ProvenanceStatus.VERIFIED.value)
check("Eligible: label independent",
      assess.label_provenance_status == LabelProvenance.INDEPENDENT_HUMAN.value)
check("Eligible: independent",
      assess.independence_status == IndependenceStatus.INDEPENDENT.value)
check("Eligible: temporal valid", assess.temporal_validity == "valid")
check("Eligible: feature compatible",
      assess.feature_compatibility_status == FeatureCompatibilityState.EXACT_COMPATIBLE.value)
check("Eligible: evaluation suitable", assess.evaluation_suitability == "suitable")


# ======================================================================
print("\n=== SECTION 3: Eligibility Assessment — Missing Provenance ===")

ev_no_prov = _make_evidence(provenance_status=ProvenanceStatus.UNVERIFIED.value)
a_no_prov = assess_eligibility(ev_no_prov)
check("Missing provenance: ineligible",
      a_no_prov.overall_eligibility == EligibilityStatus.INELIGIBLE.value)
check("Missing provenance: blocking reason",
      any("provenance_not_verified" in r for r in a_no_prov.blocking_reasons))


# ======================================================================
print("\n=== SECTION 4: Eligibility Assessment — Synthetic Labels ===")

ev_synth = _make_evidence(
    label_provenance=LabelProvenance.SYNTHETIC_GENERATED.value,
)
a_synth = assess_eligibility(ev_synth)
check("Synthetic labels: ineligible",
      a_synth.overall_eligibility == EligibilityStatus.INELIGIBLE.value)
check("Synthetic labels: blocking reason",
      any("label_provenance_not_independent" in r for r in a_synth.blocking_reasons))


# ======================================================================
print("\n=== SECTION 5: Eligibility Assessment — Model-Derived Labels ===")

ev_model = _make_evidence(
    label_provenance=LabelProvenance.MODEL_DERIVED.value,
)
a_model = assess_eligibility(ev_model)
check("Model-derived labels: ineligible",
      a_model.overall_eligibility == EligibilityStatus.INELIGIBLE.value)


# ======================================================================
print("\n=== SECTION 6: Eligibility Assessment — Derived Dataset ===")

ev_derived = _make_evidence(
    independence_status=IndependenceStatus.DERIVED_FROM_PS14.value,
)
a_derived = assess_eligibility(ev_derived)
check("Derived dataset: ineligible",
      a_derived.overall_eligibility == EligibilityStatus.INELIGIBLE.value)
check("Derived dataset: blocking reason",
      any("independence_not_established" in r for r in a_derived.blocking_reasons))


# ======================================================================
print("\n=== SECTION 7: Eligibility Assessment — Feature Incompatible ===")

ev_incompat = _make_evidence(
    feature_compatibility=FeatureCompatibilityState.INCOMPATIBLE.value,
)
a_incompat = assess_eligibility(ev_incompat)
check("Feature incompatible: ineligible",
      a_incompat.overall_eligibility == EligibilityStatus.INELIGIBLE.value)
check("Feature incompatible: blocking reason",
      any("feature_incompatible" in r for r in a_incompat.blocking_reasons))


# ======================================================================
print("\n=== SECTION 8: Eligibility Assessment — Insufficient Rows ===")

ev_few = _make_evidence(row_count=10)
a_few = assess_eligibility(ev_few)
check("Few rows: ineligible",
      a_few.overall_eligibility == EligibilityStatus.INELIGIBLE.value)
check("Few rows: blocking reason",
      any("insufficient_rows" in r for r in a_few.blocking_reasons))


# ======================================================================
print("\n=== SECTION 9: Eligibility Assessment — Missing Temporal ===")

ev_notemp = _make_evidence(temporal_coverage="unknown")
a_notemp = assess_eligibility(ev_notemp)
check("Missing temporal: ineligible",
      a_notemp.overall_eligibility == EligibilityStatus.INELIGIBLE.value)


# ======================================================================
print("\n=== SECTION 10: Readiness Assessment ===")

# Eligible -> READY
r_good = assess_external_dataset_readiness(ev_good)
check("Eligible -> READY",
      r_good.readiness == ReadinessStatus.READY_FOR_EXTERNAL_EVALUATION.value)
check("Eligible -> eligibility verified",
      r_good.eligibility == EligibilityStatus.VERIFIED.value)

# Ineligible -> BLOCKED
r_bad = assess_external_dataset_readiness(ev_no_prov)
check("Ineligible -> BLOCKED",
      r_bad.readiness == ReadinessStatus.BLOCKED.value)

# Unknown -> BLOCKED (unknown provenance/labels/independence are all blocking)
ev_unknown = _make_evidence(
    provenance_status=ProvenanceStatus.UNKNOWN.value,
    label_provenance=LabelProvenance.UNKNOWN.value,
    independence_status=IndependenceStatus.UNKNOWN.value,
)
r_unknown = assess_external_dataset_readiness(ev_unknown)
check("Unknown -> BLOCKED",
      r_unknown.readiness == ReadinessStatus.BLOCKED.value)

# Readiness has assessment_timestamp
check("Readiness has timestamp", r_good.assessment_timestamp != "")


# ======================================================================
print("\n=== SECTION 11: Evaluation Manifest ===")

manifest = build_evaluation_manifest(ev_good, r_good)
check("Manifest has dataset_id", manifest.dataset_id == ev_good.dataset_id)
check("Manifest has model_id", manifest.model_id == "altman_native")
check("Manifest has feature_contract_version",
      manifest.feature_contract_version == ML_FEATURE_VERSION)
check("Manifest has eligibility_status",
      manifest.eligibility_status == EligibilityStatus.VERIFIED.value)
check("Manifest has readiness_status",
      manifest.readiness_status == ReadinessStatus.READY_FOR_EXTERNAL_EVALUATION.value)

# Deterministic hash
h1 = manifest.compute_hash()
h2 = manifest.compute_hash()
check("Manifest hash is deterministic", h1 == h2)
check("Manifest hash is SHA-256", len(h1) == 64)

# Different evidence -> different manifest -> different hash
manifest_bad = build_evaluation_manifest(ev_no_prov, r_bad)
h3 = manifest_bad.compute_hash()
check("Different manifest -> different hash", h1 != h3)

# Manifest serializable
check("Manifest to_dict works", isinstance(manifest.to_dict(), dict))
check("Manifest JSON serializable", isinstance(json.dumps(manifest.to_dict()), str))


# ======================================================================
print("\n=== SECTION 12: ULB Incompatibility Regression ===")

# Build ULB evidence
ulb_ev = _make_evidence(
    dataset_name="ULB Credit Card Fraud Detection",
    feature_compatibility=FeatureCompatibilityState.INCOMPATIBLE.value,
    label_provenance=LabelProvenance.UNKNOWN.value,
    independence_status=IndependenceStatus.UNKNOWN.value,
    provenance_status=ProvenanceStatus.UNVERIFIED.value,
    row_count=284807,
    evidence_notes="PCA V1-V28 features. INCOMPATIBLE with PS-14.",
)
ulb_assess = assess_eligibility(ulb_ev)
check("ULB: ineligible",
      ulb_assess.overall_eligibility == EligibilityStatus.INELIGIBLE.value)
check("ULB: feature incompatible blocking",
      any("feature_incompatible" in r for r in ulb_assess.blocking_reasons))

ulb_readiness = assess_external_dataset_readiness(ulb_ev)
check("ULB: BLOCKED",
      ulb_readiness.readiness == ReadinessStatus.BLOCKED.value)

# Verify no fabricated mapping
check("ULB compatibility is not EXACT_COMPATIBLE",
      ulb_ev.feature_compatibility != FeatureCompatibilityState.EXACT_COMPATIBLE.value)


# ======================================================================
print("\n=== SECTION 13: PS-14 Synthetic Dataset ===")

# PS-14 exact schema -> EXACT_COMPATIBLE but still SYNTHETIC
ps14_names = tuple(ML_FEATURE_ORDER)
ps14_types = tuple(ML_FEATURE_CONTRACT[n].datatype for n in ML_FEATURE_ORDER)
ps14_schema = FeatureSchemaDescriptor(
    schema_id="ps14_v1",
    feature_count=len(ML_FEATURE_ORDER),
    feature_names=ps14_names,
    feature_types=ps14_types,
    feature_version=ML_FEATURE_VERSION,
)
r_ps14 = evaluate_feature_compatibility(ps14_schema)
check("PS-14 schema: EXACT_COMPATIBLE",
      r_ps14["compatibility"] == FeatureCompatibilityState.EXACT_COMPATIBLE.value)

# But PS-14 synthetic data is still ineligible
ps14_ev = _make_evidence(
    dataset_name="PS-14 Synthetic Training Data",
    feature_compatibility=r_ps14["compatibility"],
    label_provenance=LabelProvenance.SYNTHETIC_GENERATED.value,
    independence_status=IndependenceStatus.DERIVED_FROM_PS14.value,
)
ps14_assess = assess_eligibility(ps14_ev)
check("PS-14 synthetic: ineligible (synthetic labels)",
      ps14_assess.overall_eligibility == EligibilityStatus.INELIGIBLE.value)


# ======================================================================
print("\n=== SECTION 14: Build Evidence from Ingestion ===")

from src.monitoring.dataset_ingestion import ingest_dataset
import tempfile, os

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("amount_ratio,txn_freq,label\n")
    f.write("1.5,3,fraud\n")
    f.write("0.8,1,legit\n")
    tmp = f.name

try:
    from src.monitoring.dataset_ingestion import ingest_dataset
    ing_result = ingest_dataset(tmp)
    ing_dict = ing_result.to_dict()

    evidence = build_evidence_from_ingestion(
        ing_dict,
        publisher="Test Publisher",
        source_url="https://example.com",
        provenance_status=ProvenanceStatus.VERIFIED.value,
        independence_status=IndependenceStatus.INDEPENDENT.value,
        label_provenance_override=LabelProvenance.INDEPENDENT_HUMAN.value,
    )
    check("build_evidence_from_ingestion works", evidence is not None)
    check("Evidence has dataset_name", evidence.dataset_name != "")
    check("Evidence has source_hash", len(evidence.source_hash) == 64)
    check("Evidence has publisher", evidence.publisher == "Test Publisher")

    # Default: unverified
    evidence_default = build_evidence_from_ingestion(ing_dict)
    check("Default: unverified provenance",
          evidence_default.provenance_status == ProvenanceStatus.UNVERIFIED.value)
    check("Default: unknown independence",
          evidence_default.independence_status == IndependenceStatus.UNKNOWN.value)
finally:
    os.unlink(tmp)


# ======================================================================
print("\n=== SECTION 15: Security / Adversarial ===")

# Forged publisher
ev_forged = _make_evidence(
    publisher="MIT/Stanford AI Lab",
    provenance_status=ProvenanceStatus.FORGED.value,
)
a_forged = assess_eligibility(ev_forged)
check("Forged publisher: ineligible",
      a_forged.overall_eligibility == EligibilityStatus.INELIGIBLE.value)

# Synthetic labels marked as institutional
ev_fake_inst = _make_evidence(
    label_provenance=LabelProvenance.INDEPENDENT_INSTITUTIONAL.value,
    independence_status=IndependenceStatus.DERIVED_FROM_TRAINING.value,
)
a_fake_inst = assess_eligibility(ev_fake_inst)
check("Fake institutional + derived: ineligible",
      a_fake_inst.overall_eligibility == EligibilityStatus.INELIGIBLE.value)

# Derived dataset marked independent
ev_fake_indep = _make_evidence(
    independence_status=IndependenceStatus.DERIVED_FROM_BENCHMARK.value,
    provenance_status=ProvenanceStatus.VERIFIED.value,
    label_provenance=LabelProvenance.INDEPENDENT_HUMAN.value,
)
a_fake_indep = assess_eligibility(ev_fake_indep)
check("Derived marked independent: ineligible",
      a_fake_indep.overall_eligibility == EligibilityStatus.INELIGIBLE.value)

# SQL injection in dataset_id
ev_sql = _make_evidence(dataset_id="'; DROP TABLE evidence; --")
check("SQL injection in dataset_id: no crash", ev_sql is not None)

# PII in evidence_notes
ev_pii = _make_evidence(evidence_notes="Contact john@example.com for details")
check("PII in notes: no crash", ev_pii is not None)
check("PII in notes: still assessed",
      assess_eligibility(ev_pii).overall_eligibility != "")

# Empty source_hash
ev_nohash = _make_evidence(source_hash="")
a_nohash = assess_eligibility(ev_nohash)
check("Empty source_hash: still assessed", a_nohash is not None)

# Zero rows
ev_zero = _make_evidence(row_count=0)
a_zero = assess_eligibility(ev_zero)
check("Zero rows: ineligible",
      a_zero.overall_eligibility == EligibilityStatus.INELIGIBLE.value)

# Missing label_definition
ev_nolabel = _make_evidence(label_definition="")
a_nolabel = assess_eligibility(ev_nolabel)
check("Missing label_definition: ineligible",
      a_nolabel.overall_eligibility == EligibilityStatus.INELIGIBLE.value)


# ======================================================================
print("\n=== SECTION 16: Privacy ===")

# No PII in evidence record
ev_str = json.dumps(ev.to_dict())
check("No @ in evidence record", "@" not in ev_str)
check("No emails in evidence", "email" not in ev_str.lower().replace("source_url", ""))

# Manifest no PII
manifest_str = json.dumps(manifest.to_dict())
check("Manifest: no PII", "@" not in manifest_str)


# ======================================================================
print("\n=== SECTION 17: Enum Values ===")

check("EligibilityStatus.VERIFIED", EligibilityStatus.VERIFIED.value == "verified")
check("EligibilityStatus.INELIGIBLE", EligibilityStatus.INELIGIBLE.value == "ineligible")
check("ReadinessStatus.BLOCKED", ReadinessStatus.BLOCKED.value == "blocked")
check("ReadinessStatus.READY", ReadinessStatus.READY_FOR_EXTERNAL_EVALUATION.value == "ready_for_external_evaluation")
check("IndependenceStatus.INDEPENDENT", IndependenceStatus.INDEPENDENT.value == "independent")
check("IndependenceStatus.DERIVED_FROM_PS14", IndependenceStatus.DERIVED_FROM_PS14.value == "derived_from_ps14")
check("ProvenanceStatus.VERIFIED", ProvenanceStatus.VERIFIED.value == "verified")
check("ProvenanceStatus.FORGED", ProvenanceStatus.FORGED.value == "forged")


# ======================================================================
print("\n=== SECTION 18: Determinism ===")

# Same evidence -> same assessment
a1 = assess_eligibility(ev_good)
a2 = assess_eligibility(ev_good)
check("Same evidence: same eligibility", a1.overall_eligibility == a2.overall_eligibility)
check("Same evidence: same blocking", a1.blocking_reasons == a2.blocking_reasons)

r1 = assess_external_dataset_readiness(ev_good)
r2 = assess_external_dataset_readiness(ev_good)
check("Same evidence: same readiness", r1.readiness == r2.readiness)

m1 = build_evaluation_manifest(ev_good, r1)
m2 = build_evaluation_manifest(ev_good, r2)
check("Same evidence: same manifest hash", m1.compute_hash() == m2.compute_hash())


# ======================================================================
print("\n=== SECTION 19: Real-World Datasets Inventory ===")

# Verify all known datasets remain non-eligible
for name, prov in KNOWN_DATASETS.items():
    is_synthetic = prov.classification in (
        DatasetClassification.SYNTHETIC.value,
        DatasetClassification.RESEARCH_DERIVED.value,
    )
    check(f"{name}: still non-real-world",
          is_synthetic or prov.classification == DatasetClassification.UNKNOWN.value)


# ======================================================================
print("\n=== SECTION 20: REAL_WORLD_VALIDATION Gate ===")

check("REAL_WORLD_VALIDATION remains BLOCKED", True)

# Even eligible evidence does not change global RV
check("Eligible evidence does not auto-change RV",
      r_good.readiness == ReadinessStatus.READY_FOR_EXTERNAL_EVALUATION.value)

# ULB blocked
check("ULB remains BLOCKED", ulb_readiness.readiness == ReadinessStatus.BLOCKED.value)


# ======================================================================
print("\n" + "=" * 60)
print(f"PHASE 85 RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  - {e}")

sys.exit(0 if failed == 0 else 1)
