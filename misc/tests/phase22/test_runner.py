"""Unit tests for Phase 22 validation runner.

These tests use SYNTHETIC data ONLY to verify software correctness.
They are NOT real-world model validation.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase22.contracts import (
    MetadataContract, compute_file_hash, compute_string_hash,
    P20_REQUIRED_FEATURES, E_HARDNEG_EXTRA_FEATURES,
)
from phase22.gates import (
    ValidationGates, gate_firewall, gate_real_world_data,
    gate_authorization, gate_label_definition, gate_label_latency,
    gate_schema_compatibility, gate_feature_reconstruction,
    gate_temporal_ordering, gate_threshold_frozen, gate_final_holdout,
    gate_promotion,
)
from phase22.feature_engine import (
    reconstruct_p20_features, check_e_hardneg_label_features,
)
from phase22.evaluator import (
    EvaluationMetrics, compute_uncertainty, wilson_score_ci,
    evaluate_temporal_windows,
)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestContracts:
    def test_compute_file_hash(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = compute_file_hash(f)
        assert len(h) == 64  # SHA-256 hex
        # Same content → same hash
        assert compute_file_hash(f) == h

    def test_compute_string_hash(self):
        h = compute_string_hash("test")
        assert len(h) == 64

    def test_metadata_contract_valid(self, tmp_path):
        meta = {
            "dataset_name": "test_real_world",
            "source": "test",
            "owner": "test",
            "real_world_status": "REAL_WORLD",
            "authorization_status": "AUTHORIZED",
            "label_definition": "chargeback",
            "label_source": "issuer",
            "transaction_time_field": "ts",
        }
        f = tmp_path / "meta.json"
        f.write_text(json.dumps(meta))
        contract = MetadataContract.from_json(f)
        errors = contract.validate()
        assert errors == []

    def test_metadata_contract_synthetic_rejected(self, tmp_path):
        meta = {
            "dataset_name": "test",
            "source": "test",
            "owner": "test",
            "real_world_status": "SYNTHETIC",
            "authorization_status": "AUTHORIZED",
            "label_definition": "flag",
            "label_source": "generator",
            "transaction_time_field": "ts",
        }
        f = tmp_path / "meta.json"
        f.write_text(json.dumps(meta))
        contract = MetadataContract.from_json(f)
        errors = contract.validate()
        assert len(errors) > 0
        assert any("real_world_status" in e for e in errors)


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------

class TestGates:
    def test_firewall_pass(self):
        g = gate_firewall(False, False, False, False)
        assert g.passed

    def test_firewall_fail(self):
        g = gate_firewall(True, False, False, False)
        assert not g.passed
        assert "METHODOLOGY_FAILURE" in g.reason

    def test_real_world_data_pass(self):
        g = gate_real_world_data("REAL_WORLD")
        assert g.passed

    def test_real_world_data_fail(self):
        g = gate_real_world_data("SYNTHETIC")
        assert not g.passed

    def test_authorization_pass(self):
        g = gate_authorization("AUTHORIZED")
        assert g.passed

    def test_authorization_fail(self):
        g = gate_authorization("UNAUTHORIZED")
        assert not g.passed

    def test_label_definition_pass(self):
        g = gate_label_definition("chargeback confirmed by issuer")
        assert g.passed

    def test_label_definition_fail(self):
        g = gate_label_definition("UNVERIFIED")
        assert not g.passed

    def test_label_latency_pass(self):
        g = gate_label_latency(True)
        assert g.passed

    def test_label_latency_fail(self):
        g = gate_label_latency(False)
        assert not g.passed

    def test_schema_compatibility_pass(self):
        g = gate_schema_compatibility(45, 45)
        assert g.passed

    def test_schema_compatibility_fail(self):
        g = gate_schema_compatibility(20, 45)
        assert not g.passed

    def test_feature_reconstruction_pass(self):
        g = gate_feature_reconstruction(True)
        assert g.passed

    def test_feature_reconstruction_fail(self):
        g = gate_feature_reconstruction(False)
        assert not g.passed

    def test_temporal_ordering_pass(self):
        g = gate_temporal_ordering(True)
        assert g.passed

    def test_temporal_ordering_fail(self):
        g = gate_temporal_ordering(False)
        assert not g.passed

    def test_threshold_frozen_pass(self):
        g = gate_threshold_frozen(True, True)
        assert g.passed

    def test_threshold_frozen_fail(self):
        g = gate_threshold_frozen(True, False)
        assert not g.passed

    def test_final_holdout_pass(self):
        g = gate_final_holdout(True, True, True, True)
        assert g.passed

    def test_final_holdout_fail(self):
        g = gate_final_holdout(True, False, True, True)
        assert not g.passed

    def test_validation_gates_all_pass(self):
        vg = ValidationGates()
        vg.record("a", True, "ok")
        vg.record("b", True, "ok")
        assert vg.all_passed
        assert len(vg.failures) == 0

    def test_validation_gates_has_failure(self):
        vg = ValidationGates()
        vg.record("a", True, "ok")
        vg.record("b", False, "fail")
        assert not vg.all_passed
        assert len(vg.failures) == 1

    def test_gate_promotion_blocks_on_failure(self):
        vg = ValidationGates()
        vg.record("a", True)
        vg.record("b", False)
        g = gate_promotion(vg)
        assert not g.passed


# ---------------------------------------------------------------------------
# Feature engine tests
# ---------------------------------------------------------------------------

class TestFeatureEngine:
    def test_reconstruct_p20_with_matching_columns(self):
        # Synthetic data with PS-14 columns
        df = pd.DataFrame({
            "ts": ["2024-01-01 10:00:00"],
            "amount": [100.0],
            "fraud": [0],
        })
        result = reconstruct_p20_features(df, {}, "ts")
        assert result.n_features > 0
        assert result.status in ("PASS", "PARTIAL", "FAIL")

    def test_check_e_hardneg_with_labels(self):
        df = pd.DataFrame({"amount": [100]})
        result = check_e_hardneg_label_features(df, has_historical_labels=True)
        assert result["status"] == "RECONSTRUCTABLE"

    def test_check_e_hardneg_without_labels(self):
        df = pd.DataFrame({"amount": [100]})
        result = check_e_hardneg_label_features(df, has_historical_labels=False)
        assert result["status"] == "NOT_RECONSTRUCTABLE"


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_metrics_from_predictions(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0])
        y_score = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.1, 0.05, 0.6, 0.15])
        m = EvaluationMetrics.from_predictions(y_true, y_score, threshold=0.5)
        assert m.n_total == 10
        assert m.n_fraud == 4
        assert m.tp == 4
        assert m.recall == 1.0
        assert m.roc_auc > 0.9

    def test_metrics_zero_prevalence(self):
        y_true = np.array([0, 0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        m = EvaluationMetrics.from_predictions(y_true, y_score, threshold=0.5)
        assert m.n_fraud == 0
        assert m.prevalence == 0.0

    def test_wilson_score_ci(self):
        lower, upper = wilson_score_ci(50, 1000, 0.95)
        assert 0.0 < lower < upper < 1.0
        assert abs((lower + upper) / 2 - 0.05) < 0.02  # ~5% centered

    def test_uncertainty_computation(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0])
        y_score = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.1, 0.05, 0.6, 0.15])
        u = compute_uncertainty(y_true, y_score, threshold=0.5)
        assert u.recall_ci_lower <= u.recall_ci_upper
        assert u.fpr_ci_lower <= u.fpr_ci_upper

    def test_temporal_windows(self):
        n = 100
        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "label": [1 if i % 20 == 0 else 0 for i in range(n)],
        })
        scores = np.random.RandomState(42).rand(n)
        results = evaluate_temporal_windows(df, scores, df["label"].values, "ts", 0.5, n_windows=4)
        assert len(results) == 4
        assert all(r.n_total > 0 for r in results)


# ---------------------------------------------------------------------------
# Integration: dry-run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_prepare_mode_runs(self, tmp_path):
        """Verify the prepare mode runs without errors."""
        from phase22.run_real_world_validation import run_prepare_mode
        result = run_prepare_mode(tmp_path)
        assert result["runner_status"] == "READY"
        assert result["real_world_data_available"] is False
        assert result["promotion_blocked"] is True
        # Verify output files
        assert (tmp_path / "01_preflight.json").exists()


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_prepare_mode_cli(self, tmp_path, monkeypatch):
        """Test CLI in prepare mode."""
        from phase22 import run_real_world_validation
        monkeypatch.setattr(sys, "argv", [
            "run_real_world_validation",
            "--mode", "prepare",
            "--output", str(tmp_path),
        ])
        exit_code = run_real_world_validation.main()
        assert exit_code == 0
        assert (tmp_path / "machine_summary.json").exists()

    def test_validate_mode_no_data_fails(self, tmp_path, monkeypatch):
        """Test CLI in validate mode without data fails."""
        from phase22 import run_real_world_validation
        monkeypatch.setattr(sys, "argv", [
            "run_real_world_validation",
            "--mode", "validate",
            "--output", str(tmp_path),
        ])
        with pytest.raises(SystemExit):
            run_real_world_validation.main()
