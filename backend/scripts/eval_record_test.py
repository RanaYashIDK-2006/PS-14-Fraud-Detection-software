#!/usr/bin/env python3
"""Phase 39 regression tests — reproducible evaluation infrastructure.

Plain assert script (repository convention, no pytest). Run:
    python backend/scripts/eval_record_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import eval_record as er
import metric_definitions as md
import result_matrix as rm
import evaluate as ev


def test_model_hash_changes_when_artifact_changes():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        f = d / "model.joblib"
        f.write_bytes(b"weights-v1")
        h1 = er.artifact_hashes([f])["model_hash"]
        f.write_bytes(b"weights-v2")
        h2 = er.artifact_hashes([f])["model_hash"]
        assert h1 != h2, "model hash must change when artifact bytes change"


def test_artifact_hash_order_independent():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "a.joblib").write_bytes(b"aaa")
        (d / "b.joblib").write_bytes(b"bbb")
        h1 = er.artifact_hashes([d / "a.joblib", d / "b.joblib"])["model_hash"]
        h2 = er.artifact_hashes([d / "b.joblib", d / "a.joblib"])["model_hash"]
        assert h1 == h2, "aggregate hash must be order-independent"


def test_dataset_fingerprint_detects_any_change():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        f = d / "data.csv"
        f.write_bytes(b"a,b,label\n1,2,0\n")
        s1 = er.dataset_fingerprint(f)["sha256"]
        f.write_bytes(b"a,b,label\n1,2,0\n3,4,1\n")
        s2 = er.dataset_fingerprint(f)["sha256"]
        assert s1 != s2
        f.write_bytes(b"a,b,label\n1,2,0\n")
        assert er.dataset_fingerprint(f)["sha256"] == s1, "fingerprint must be deterministic"


def test_missing_dataset_fingerprint_is_not_fabricated():
    fp = er.dataset_fingerprint(Path("does/not/exist.csv"))
    assert fp["sha256"] is None and fp["exists"] is False


def test_evaluation_record_binds_model_and_dataset():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "m.joblib").write_bytes(b"model-bytes")
        ds = d / "data.csv"
        ds.write_bytes(b"label\n0\n1\n")
        rec = er.create_evaluation_record(
            model_identifier="test-model", artifact_paths=[d / "m.joblib"],
            dataset_path=ds, seed=42, threshold=0.5, threshold_source="validation",
            metrics={"roc_auc": 0.9})
        dd = rec.to_dict()
        assert dd["model_hash"] and dd["dataset"]["sha256"]
        assert dd["seed"] == 42 and dd["threshold"] == 0.5
        assert dd["threshold_source"] == "validation"
        assert dd["metrics"]["roc_auc"] == 0.9
        assert dd["status"] == "COMPLETED"


def test_evaluation_id_changes_when_model_or_data_changes():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        f = d / "m.joblib"
        f.write_bytes(b"v1")
        ds = d / "data.csv"
        ds.write_bytes(b"1")
        r1 = er.new_evaluation_id("2026-01-01T00:00:00+00:00",
                                  er.artifact_hashes([f])["model_hash"], "ds1")
        f.write_bytes(b"v2")
        r2 = er.new_evaluation_id("2026-01-01T00:00:00+00:00",
                                  er.artifact_hashes([f])["model_hash"], "ds1")
        r3 = er.new_evaluation_id("2026-01-01T00:00:00+00:00",
                                  er.artifact_hashes([f])["model_hash"], "ds2")
        assert r1 != r2 and r1 != r3


def test_ledger_is_append_only_and_preserves_failures():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        ds = d / "data.csv"
        ds.write_bytes(b"1")
        led = er.EvaluationLedger(d / "ledger.jsonl")
        rec1 = er.create_evaluation_record(
            model_identifier="m", artifact_paths=[], dataset_path=ds,
            metrics={"roc_auc": 0.42}, status="FAILED",
            warnings=["ROC-AUC below random — preserved, not hidden"])
        rec2 = er.create_evaluation_record(
            model_identifier="m", artifact_paths=[], dataset_path=ds,
            metrics={"roc_auc": 0.96}, status="COMPLETED")
        led.append(rec1)
        led.append(rec2)
        rows = led.records()
        assert len(rows) == 2
        assert rows[0]["status"] == "FAILED" and rows[0]["metrics"]["roc_auc"] == 0.42
        assert rows[1]["status"] == "COMPLETED"
        assert rows[0]["evaluation_id"] != rows[1]["evaluation_id"]


def test_metric_definitions_roc_auc_ties_and_random():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])  # all ties -> AUC 0.5
    assert abs(md.roc_auc(y, s) - 0.5) < 1e-9
    s2 = np.array([0.1, 0.2, 0.8, 0.9])  # perfect ranking
    assert abs(md.roc_auc(y, s2) - 1.0) < 1e-9


def test_metric_definitions_pr_auc_random_equals_prevalence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    s = rng.random(2000)
    prev = y.mean()
    assert abs(md.pr_auc(y, s) - prev) < 0.05, "random PR-AUC should ≈ prevalence"


def test_metric_definitions_rejects_nan_predictions():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, np.nan, 0.2, 0.3])
    try:
        md.confusion_counts(y, s, 0.5)
        raise AssertionError("NaN scores must be rejected, not silently handled")
    except ValueError:
        pass


def test_confusion_counts_threshold_semantics():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.6, 0.6, 0.4, 0.4])
    # threshold semantics: score >= threshold is positive
    tp, fp, tn, fn = md.confusion_counts(y, s, 0.6)
    assert (tp, fp, tn, fn) == (1, 1, 1, 1)


def test_threshold_at_fpr_uses_validation_only():
    rng = np.random.default_rng(1)
    y_val = rng.integers(0, 2, 4000)
    s_val = np.where(y_val == 1, rng.random(4000) * 0.5 + 0.5, rng.random(4000) * 0.5)
    t = md.threshold_at_fpr_on_validation(y_val, s_val, target_fpr=0.01)
    _, fp, tn, _ = md.confusion_counts(y_val, s_val, t)
    fpr = fp / (fp + tn)
    assert abs(fpr - 0.01) < 0.01, f"validation FPR {fpr:.4f} should hit the 1% target"


def test_operating_point_counts_and_rates():
    op = md.operating_point(
        np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        0.5, "fixed")
    assert (op.tp, op.fp, op.tn, op.fn) == (2, 1, 7, 0)
    assert abs(op.recall - 1.0) < 1e-9
    assert abs(op.fpr - 1 / 8) < 1e-9
    assert op.n_alerts == 3


def test_bootstrap_ci_covers_truth_and_withholds_on_small_cells():
    rng = np.random.default_rng(7)
    y = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    s = np.where(y == 1, rng.random(1000) * 0.4 + 0.6, rng.random(1000) * 0.4)
    ci = md.bootstrap_ci(md.roc_auc, y, s, n_bootstrap=200, seed=42)
    assert ci.ci_low is not None and ci.ci_high is not None
    assert ci.ci_low <= ci.point_estimate + 1e-9 and ci.point_estimate <= ci.ci_high + 1e-9
    assert ci.method == "percentile bootstrap" and ci.resampling_unit.startswith("observation")
    # small-sample withholding: 5 positives only
    y_small = np.array([1] * 5 + [0] * 100)
    s_small = rng.random(105)
    ci_small = md.bootstrap_ci(md.roc_auc, y_small, s_small, seed=42)
    assert ci_small.ci_low is None and ci_small.withheld_reason is not None


def test_prevalence_report_explains_accuracy_secondary():
    rep = md.prevalence_report(np.array([1] * 20 + [0] * 1980))
    assert rep["positive_count"] == 20 and rep["negative_count"] == 1980
    assert abs(rep["prevalence"] - 0.01) < 1e-9
    assert "SECONDARY" in rep["note"]


def test_baselines_are_evaluated_honestly():
    y = np.array([1] * 50 + [0] * 950)
    maj = md.majority_baseline(y)
    assert maj["recall"] == 0.0 and maj["roc_auc"] == 0.5
    rnd = md.random_baseline(y, seed=3)
    assert rnd["baseline"] == "random_scores"
    assert abs(rnd["expected_roc_auc"] - 0.5) < 1e-9
    cmp_ = md.compare_to_baseline({"roc_auc": 0.966, "pr_auc": 0.877}, rnd)
    assert "No statistical-significance claim" in cmp_["model_vs_baseline"]["note"]


def test_small_sample_warnings_fire():
    warns = md.small_sample_warning(np.array([1] * 10 + [0] * 90))
    assert any("SMALL_POSITIVE_CLASS" in w for w in warns)
    y_imb = np.array([1] * 5 + [0] * 2000)
    warns2 = md.small_sample_warning(y_imb, min_per_class=3)
    assert any("HEAVY_IMBALANCE" in w for w in warns2)


def test_full_evaluation_metrics_structure():
    rng = np.random.default_rng(2)
    y = np.concatenate([np.zeros(400, dtype=int), np.ones(400, dtype=int)])
    s = np.where(y == 1, rng.random(800) * 0.5 + 0.5, rng.random(800) * 0.5)
    m = md.FullEvaluationMetrics(y_true=y, scores=s, threshold=0.5,
                                 threshold_source="fixed", bootstrap=True, seed=42)
    d = m.to_dict()
    assert d["metric_definitions_version"] == md.METRIC_DEFINITIONS_VERSION
    assert "operating_point" in d and "class_imbalance" in d
    assert "confidence_intervals" in d and "baselines" in d
    assert set(d["confidence_intervals"].keys()) == {"roc_auc", "pr_auc", "recall_at_threshold"}


def test_result_matrix_classification_closed_set():
    row = rm.build_matrix([{"dataset": "X", "training_relationship": "in_domain",
                            "domain": "synthetic", "roc_auc": 0.966}])["rows"][0]
    assert row["training_relationship"] == "IN_DOMAIN"
    assert row["domain"] == "SYNTHETIC"
    # unknown stays unknown — never upgraded
    row2 = rm.build_matrix([{"dataset": "Y", "training_relationship": "weird",
                             "domain": "mystery"}])["rows"][0]
    assert row2["training_relationship"] == "UNKNOWN"
    assert row2["domain"] == "UNKNOWN"


def test_result_matrix_does_not_aggregate():
    m = rm.build_matrix([
        {"dataset": "A", "roc_auc": 0.966, "training_relationship": "IN_DOMAIN"},
        {"dataset": "B", "roc_auc": 0.435, "training_relationship": "EXTERNAL"},
    ])
    assert "aggregate" not in json.dumps(m).lower() or m["note"]
    assert len(m["rows"]) == 2
    assert m["rows"][1]["roc_auc"] == 0.435  # poor result preserved, not hidden
    md_txt = rm.matrix_to_markdown(m)
    assert "| B |" in md_txt and "0.435" in md_txt


def test_refuse_test_tuning_guard():
    for bad in ("test", "external", "holdout", "TEST"):
        try:
            ev.refuse_test_tuning(bad)
            raise AssertionError(f"threshold_source={bad} must be refused")
        except SystemExit as e:
            assert "REFUSED" in str(e)
    ev.refuse_test_tuning("validation")
    ev.refuse_test_tuning("fixed")


def test_git_commit_none_outside_repo():
    # Should not raise whether or not we're in a repo; returns str or None
    val = er.git_commit()
    assert val is None or (isinstance(val, str) and len(val) >= 7)


def main() -> int:
    tests = [
        test_model_hash_changes_when_artifact_changes,
        test_artifact_hash_order_independent,
        test_dataset_fingerprint_detects_any_change,
        test_missing_dataset_fingerprint_is_not_fabricated,
        test_evaluation_record_binds_model_and_dataset,
        test_evaluation_id_changes_when_model_or_data_changes,
        test_ledger_is_append_only_and_preserves_failures,
        test_metric_definitions_roc_auc_ties_and_random,
        test_metric_definitions_pr_auc_random_equals_prevalence,
        test_metric_definitions_rejects_nan_predictions,
        test_confusion_counts_threshold_semantics,
        test_threshold_at_fpr_uses_validation_only,
        test_operating_point_counts_and_rates,
        test_bootstrap_ci_covers_truth_and_withholds_on_small_cells,
        test_prevalence_report_explains_accuracy_secondary,
        test_baselines_are_evaluated_honestly,
        test_small_sample_warnings_fire,
        test_full_evaluation_metrics_structure,
        test_result_matrix_classification_closed_set,
        test_result_matrix_does_not_aggregate,
        test_refuse_test_tuning_guard,
        test_git_commit_none_outside_repo,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 60}\nResults: {passed}/{passed + failed} passed, {failed} failed\n{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
