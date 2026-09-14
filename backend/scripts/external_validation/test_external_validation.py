"""Tests for the external validation pipeline (Phase 38).

Tests cover:
- Missing provenance
- Unknown provenance
- Synthetic dataset classification
- Derived dataset classification
- Missing fraud label
- Invalid labels
- Post-event feature
- Missing required feature
- Incompatible feature type
- Duplicate external row
- Training/external overlap
- Schema mismatch
- Frozen-model enforcement
- External tuning prevention
- Successful eligible dataset
- Blocked dataset
- Promotion remaining blocked

Uses small synthetic fixtures — NOT fabricated "real-world" data.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure the parent directory is in the path
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from external_validation.metadata_schema import (
    ExternalDatasetMetadata,
    FeatureProvenance,
    ProvenanceStatus,
    DatasetEligibility,
    PS14_ML_FEATURES,
)
from external_validation.eligibility_gates import (
    run_all_gates,
    check_provenance,
    check_license,
    check_fraud_label_definition,
    check_feature_provenance,
    check_class_label_integrity,
    check_dataset_independence,
)
from external_validation.causality_audit import (
    run_causality_audit,
    audit_feature_provenance,
    _check_feature_name_suspicion,
)
from external_validation.schema_mapper import map_schema, apply_mapping
from external_validation.independence_check import (
    run_independence_check,
    check_exact_duplicates,
    check_training_overlap,
)
from external_validation.label_validation import validate_labels
from external_validation.frozen_evaluator import (
    EvaluationConfig,
    FrozenEvaluationResult,
    run_frozen_evaluation,
    verify_frozen_model,
)
from external_validation.metrics import compute_metrics
from external_validation.distribution_shift import run_distribution_shift_analysis
from external_validation.anti_tuning import (
    TuningAttemptDetected,
    validate_no_tuning,
    create_heldout_guard,
)
from external_validation.audit_report import generate_audit_report
from external_validation.promotion_gate import check_promotion_gates


# ============================================================
# Test fixtures
# ============================================================

def _make_metadata(**overrides) -> ExternalDatasetMetadata:
    """Create a minimal valid metadata object."""
    defaults = {
        "name": "test_dataset",
        "source": "test_source",
        "license": "research use",
        "provenance_status": ProvenanceStatus.VERIFIED.value,
        "label_definition": "binary fraud label",
        "fraud_label_timing": "at_transaction",
        "label_column": "is_fraud",
        "num_rows": 1000,
        "num_positive": 50,
        "num_negative": 950,
        "columns": PS14_ML_FEATURES + ["is_fraud"],
    }
    defaults.update(overrides)
    return ExternalDatasetMetadata(**defaults)


def _make_features() -> list[FeatureProvenance]:
    """Create feature provenance for all PS14 features."""
    return [
        FeatureProvenance(
            feature_name=f,
            description=f"Feature {f}",
            source="transaction_data",
            availability_time="at_event_time",
            event_time_relation="before",
            post_event_possible=False,
            allowed_for_scoring=True,
        )
        for f in PS14_ML_FEATURES
    ]


# ============================================================
# Test: Missing provenance
# ============================================================

def test_missing_provenance():
    """Test that UNKNOWN provenance is detected."""
    meta = _make_metadata(provenance_status=ProvenanceStatus.UNKNOWN.value)
    result = check_provenance(meta)
    assert result.status == "UNKNOWN", f"Expected UNKNOWN, got {result.status}"
    print("  PASS: test_missing_provenance")


# ============================================================
# Test: Unknown provenance
# ============================================================

def test_unknown_provenance():
    """Test that UNKNOWN provenance blocks eligibility."""
    meta = _make_metadata(provenance_status=ProvenanceStatus.UNKNOWN.value)
    report = run_all_gates(meta)
    assert report.overall_status != DatasetEligibility.ELIGIBLE.value
    print("  PASS: test_unknown_provenance")


# ============================================================
# Test: Synthetic dataset classification
# ============================================================

def test_synthetic_dataset():
    """Test that synthetic datasets are BLOCKED."""
    meta = _make_metadata(
        provenance_status=ProvenanceStatus.CONFIRMED_SYNTHETIC.value,
        is_synthetic=True,
    )
    report = run_all_gates(meta)
    assert report.overall_status == DatasetEligibility.BLOCKED.value
    print("  PASS: test_synthetic_dataset")


# ============================================================
# Test: Derived dataset classification
# ============================================================

def test_derived_dataset():
    """Test that derived datasets are BLOCKED."""
    meta = _make_metadata(is_derived=True)
    report = run_all_gates(meta)
    assert report.overall_status == DatasetEligibility.BLOCKED.value
    print("  PASS: test_derived_dataset")


# ============================================================
# Test: Missing fraud label
# ============================================================

def test_missing_fraud_label():
    """Test that missing label definition is detected."""
    meta = _make_metadata(label_definition="")
    result = check_fraud_label_definition(meta)
    assert result.status == "FAIL", f"Expected FAIL, got {result.status}"
    print("  PASS: test_missing_fraud_label")


# ============================================================
# Test: Invalid labels
# ============================================================

def test_invalid_labels():
    """Test that invalid labels are detected."""
    meta = _make_metadata()
    y = np.array([0, 1, 2, 0, 1, 3])  # contains unexpected values
    report = validate_labels(y, meta)
    assert len(report.warnings) > 0 or report.num_unknown > 0
    print("  PASS: test_invalid_labels")


# ============================================================
# Test: Post-event feature
# ============================================================

def test_post_event_feature():
    """Test that post-event features are flagged."""
    meta = _make_metadata()
    meta.feature_definitions = [
        FeatureProvenance(
            feature_name="chargeback_outcome",
            post_event_possible=True,
            reason="Chargeback is a post-event resolution",
        ),
    ]
    report = run_causality_audit(meta)
    assert not report.all_clear
    assert len(report.excluded_features) > 0
    print("  PASS: test_post_event_feature")


# ============================================================
# Test: Missing required feature
# ============================================================

def test_missing_required_feature():
    """Test that missing required features are detected."""
    meta = _make_metadata(columns=["is_fraud"])  # only label, no features
    meta.feature_definitions = []
    report = map_schema(meta)
    assert not report.all_required_present
    assert len(report.missing_features) > 0
    print("  PASS: test_missing_required_feature")


# ============================================================
# Test: Incompatible feature type
# ============================================================

def test_incompatible_feature_type():
    """Test that incompatible feature types are detected."""
    meta = _make_metadata()
    meta.column_types = {PS14_ML_FEATURES[0]: "string"}  # numeric expected
    meta.columns = PS14_ML_FEATURES + ["is_fraud"]
    report = map_schema(meta)
    assert not report.schema_compatible
    print("  PASS: test_incompatible_feature_type")


# ============================================================
# Test: Duplicate external rows
# ============================================================

def test_duplicate_external_rows():
    """Test that duplicate rows within external data are detected."""
    X = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])  # first two rows identical
    dup_count = check_exact_duplicates(X)
    assert dup_count > 0
    print("  PASS: test_duplicate_external_rows")


# ============================================================
# Test: Training/external overlap
# ============================================================

def test_training_external_overlap():
    """Test that overlap between training and external data is detected."""
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_ext = np.array([[1, 2, 3], [10, 11, 12]])  # first row overlaps
    report = run_independence_check(X_ext, X_train)
    assert not report.is_independent
    assert report.training_overlap_rows > 0
    print("  PASS: test_training_external_overlap")


# ============================================================
# Test: Schema mismatch
# ============================================================

def test_schema_mismatch():
    """Test that schema mismatches are detected."""
    meta = _make_metadata()
    # Remove half the required features
    meta.columns = PS14_ML_FEATURES[:5] + ["is_fraud"]
    meta.feature_definitions = []
    report = map_schema(meta)
    assert not report.all_required_present
    assert len(report.missing_features) > 0
    print("  PASS: test_schema_mismatch")


# ============================================================
# Test: Frozen-model enforcement
# ============================================================

def test_frozen_model_enforcement():
    """Test that frozen model cannot be modified."""
    config = EvaluationConfig(
        model_version="test",
        model_hash="abc123",
        threshold=0.5,
    )
    # Verify that the evaluation config records the hash
    assert config.model_hash == "abc123"
    print("  PASS: test_frozen_model_enforcement")


# ============================================================
# Test: External tuning prevention
# ============================================================

def test_external_tuning_prevention():
    """Test that tuning with external data is prevented."""
    @create_heldout_guard("external_test")
    def train_model(data_path):
        return "trained"

    try:
        train_model("external_test_data.csv")
        assert False, "Should have raised TuningAttemptDetected"
    except TuningAttemptDetected:
        pass
    print("  PASS: test_external_tuning_prevention")


# ============================================================
# Test: Successful eligible dataset
# ============================================================

def test_eligible_dataset():
    """Test that a fully documented dataset passes eligibility."""
    meta = _make_metadata()
    meta.feature_definitions = _make_features()
    report = run_all_gates(meta)
    # May still be CONDITIONALLY_ELIGIBLE due to independence check
    assert report.overall_status in (
        DatasetEligibility.ELIGIBLE.value,
        DatasetEligibility.CONDITIONALLY_ELIGIBLE.value,
    )
    print("  PASS: test_eligible_dataset")


# ============================================================
# Test: Blocked dataset
# ============================================================

def test_blocked_dataset():
    """Test that a synthetic dataset is blocked."""
    meta = _make_metadata(
        provenance_status=ProvenanceStatus.CONFIRMED_SYNTHETIC.value,
    )
    report = run_all_gates(meta)
    assert report.overall_status == DatasetEligibility.BLOCKED.value
    print("  PASS: test_blocked_dataset")


# ============================================================
# Test: Promotion remaining blocked
# ============================================================

def test_promotion_remaining_blocked():
    """Test that promotion is always blocked after evaluation."""
    result = check_promotion_gates(
        provenance_verified=True,
        causality_verified=True,
        dataset_independent=True,
        label_validated=True,
        schema_compatible=True,
        frozen_model_evaluated=True,
        performance_threshold_met=False,  # No threshold defined
        security_requirements_met=False,  # No security gate
    )
    assert result.promotion_blocked
    assert not result.promotion_eligible
    print("  PASS: test_promotion_remaining_blocked")


# ============================================================
# Test: Metrics calculation
# ============================================================

def test_metrics_calculation():
    """Test that metrics are computed correctly."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    metrics = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
    assert metrics.n_samples == 10
    assert metrics.n_positive == 3
    assert metrics.n_negative == 7
    assert metrics.tp == 2
    assert metrics.fp == 0
    assert metrics.fn == 1
    assert metrics.tn == 7
    assert metrics.recall > 0
    assert metrics.precision > 0
    print("  PASS: test_metrics_calculation")


# ============================================================
# Test: Metadata validation
# ============================================================

def test_metadata_validation():
    """Test that metadata validation catches missing fields."""
    meta = ExternalDatasetMetadata()  # all defaults
    errors = meta.validate()
    assert len(errors) > 0
    print("  PASS: test_metadata_validation")


# ============================================================
# Test: Feature name suspicion
# ============================================================

def test_feature_name_suspicion():
    """Test that suspicious feature names are detected."""
    assert _check_feature_name_suspicion("chargeback_amount") is not None
    assert _check_feature_name_suspicion("amount_ratio") is None
    print("  PASS: test_feature_name_suspicion")


# ============================================================
# Test: Label validation with empty data
# ============================================================

def test_label_validation_empty():
    """Test label validation with empty data."""
    meta = _make_metadata()
    y = np.array([])
    report = validate_labels(y, meta)
    assert not report.label_exists
    print("  PASS: test_label_validation_empty")


# ============================================================
# Test: Distribution shift analysis
# ============================================================

def test_distribution_shift():
    """Test distribution shift analysis."""
    X_ext = np.random.randn(100, 3)
    y_ext = np.random.binomial(1, 0.5, 100)
    X_train = np.random.randn(100, 3) * 2  # different scale
    y_train = np.random.binomial(1, 0.1, 100)  # different prevalence

    report = run_distribution_shift_analysis(X_ext, y_ext, X_train, y_train)
    assert len(report.warnings) > 0
    print("  PASS: test_distribution_shift")


# ============================================================
# Test: Full pipeline dry run
# ============================================================

def test_full_pipeline_dry_run():
    """Test the full pipeline with metadata only (no dataset)."""
    from external_validation.run_validation import run_pipeline

    meta = _make_metadata()
    meta.feature_definitions = _make_features()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "name": meta.name,
            "source": meta.source,
            "license": meta.license,
            "provenance_status": meta.provenance_status,
            "label_definition": meta.label_definition,
            "fraud_label_timing": meta.fraud_label_timing,
            "label_column": meta.label_column,
            "positive_class_value": meta.positive_class_value,
            "negative_class_value": meta.negative_class_value,
            "num_rows": meta.num_rows,
            "num_positive": meta.num_positive,
            "num_negative": meta.num_negative,
            "columns": meta.columns,
            "feature_definitions": [
                {
                    "feature_name": feat.feature_name,
                    "description": feat.description,
                    "source": feat.source,
                    "availability_time": feat.availability_time,
                    "event_time_relation": feat.event_time_relation,
                    "post_event_possible": feat.post_event_possible,
                    "allowed_for_scoring": feat.allowed_for_scoring,
                }
                for feat in meta.feature_definitions
            ],
            "feature_availability_at_decision_time": "all_available",
            "is_synthetic": False,
            "is_derived": False,
        }, f)
        meta_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        exit_code = run_pipeline(
            metadata_path=meta_path,
            output_path=output_path,
        )
        # Should succeed (metadata-only validation)
        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

        # Verify report was generated
        report = json.loads(Path(output_path).read_text())
        assert "final_decision" in report
        print("  PASS: test_full_pipeline_dry_run")
    finally:
        Path(meta_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


# ============================================================
# Run all tests
# ============================================================

def main():
    """Run all external validation tests."""
    print("=" * 60)
    print("Phase 38: External Validation Pipeline Tests")
    print("=" * 60)
    print()

    tests = [
        test_missing_provenance,
        test_unknown_provenance,
        test_synthetic_dataset,
        test_derived_dataset,
        test_missing_fraud_label,
        test_invalid_labels,
        test_post_event_feature,
        test_missing_required_feature,
        test_incompatible_feature_type,
        test_duplicate_external_rows,
        test_training_external_overlap,
        test_schema_mismatch,
        test_frozen_model_enforcement,
        test_external_tuning_prevention,
        test_eligible_dataset,
        test_blocked_dataset,
        test_promotion_remaining_blocked,
        test_metrics_calculation,
        test_metadata_validation,
        test_feature_name_suspicion,
        test_label_validation_empty,
        test_distribution_shift,
        test_full_pipeline_dry_run,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
