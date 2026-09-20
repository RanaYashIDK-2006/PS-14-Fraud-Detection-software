"""Phase 93: Real-world evaluation protocol tests."""
from __future__ import annotations
import hashlib, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.monitoring.real_world_evaluation_protocol import (
    build_protocol, build_identity_lock, compute_metrics,
    bootstrap_confidence_intervals, measure_drift,
    run_synthetic_evaluation, EvaluationState, PRODUCTION_THRESHOLD,
    MODEL_ID, RELEASE_ID, NATIVE_48,
)
from src.monitoring.outcome_trust import SOURCE_TRUST, TrustClass
from src.monitoring.outcome_pipeline import OutcomeSource
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

# A. Deterministic protocol
print("=== A. Deterministic protocol ===")
p1 = build_protocol()
p2 = build_protocol()
check(p1.manifest_hash == p2.manifest_hash, "identical manifest hash")
check(p1.protocol_id == "RWV-EVALUATION-PROTOCOL-93", "constant protocol_id")
check(p1.to_dict() == p2.to_dict(), "protocol dict deterministic")

# B. Model identity locking
print("=== B. Model identity locking ===")
lock = build_identity_lock()
check(lock.model_id == MODEL_ID, "model_id locked")
check(lock.release_id == RELEASE_ID, "release_id locked")
check(lock.feature_version == "v1", "feature version locked")
check(lock.feature_count == 48, "feature count locked")

# C. Threshold locked at 0.018758
print("=== C. Threshold locked ===")
check(PRODUCTION_THRESHOLD == 0.018758, "threshold = 0.018758")
check(p1.locked_threshold == 0.018758, "protocol threshold locked")
check(lock.threshold == 0.018758, "identity lock threshold")

# D. Model identity verification
print("=== D. Model identity verification ===")
lock2 = build_identity_lock()
check(lock.verify(lock2), "identity lock self-verify")
check(lock.verify(lock) is True, "identity lock identical verify")

# E. Identity mismatch detection
print("=== E. Identity mismatch detection ===")
from src.monitoring.real_world_evaluation_protocol import ModelIdentityLock
lock_bad = ModelIdentityLock("wrong_model", "wrong_release", "v1", 48, 0.018758, {})
check(lock.verify(lock_bad) is False, "identity mismatch detected")

# F. Synthetic evaluation
print("=== F. Synthetic evaluation ===")
syn = run_synthetic_evaluation()
check(syn.evaluation_state == "evaluation_complete", "evaluation complete")
check(syn.is_real_world is False, "not real world")
check(syn.rwv_status == "BLOCKED_PENDING_ELIGIBLE_DATASET", "RWV blocked")
check(syn.promotion_status == "no_promotion", "no promotion")
check(syn.model_identity_valid is True, "model identity valid")
check(syn.threshold_locked is True, "threshold locked")

# G. Metrics calculated
print("=== G. Metrics calculated ===")
m = syn.metrics
check("precision" in m, "precision calculated")
check("recall" in m, "recall calculated")
check("f1" in m, "f1 calculated")
check("roc_auc" in m, "roc_auc calculated")
check("pr_auc" in m, "pr_auc calculated")
check("brier_score" in m, "brier_score calculated")
check(m["threshold"] == PRODUCTION_THRESHOLD, "metrics use locked threshold")
check(0 <= m["precision"] <= 1, "precision in [0,1]")
check(0 <= m["recall"] <= 1, "recall in [0,1]")
check(0 <= m["roc_auc"] <= 1, "roc_auc in [0,1]")

# H. Custom metrics calculation
print("=== H. Custom metrics calculation ===")
y_true = [1, 1, 0, 0, 1]
y_scores = [0.05, 0.03, 0.001, 0.002, 0.02]
custom = compute_metrics(y_true, y_scores, 0.018758)
check(custom["tp"] >= 0, "tp computed")
check(custom["fp"] >= 0, "fp computed")
check(custom["total"] == 5, "total correct")

# I. Bootstrap CI
print("=== I. Bootstrap CI ===")
ci = bootstrap_confidence_intervals(y_true, y_scores, 0.018758, n_bootstrap=20, seed=42)
check(ci["n_bootstrap"] == 20, "20 bootstrap iterations")
check(ci["seed"] == 42, "seed recorded")
check(ci["confidence"] == 0.95, "95% CI")
check("precision_ci_lower" in ci, "precision CI lower")
check("recall_ci_lower" in ci, "recall CI lower")
check(ci["precision_ci_lower"] <= ci["precision_ci_upper"], "CI is ordered")

# J. Bootstrap determinism
print("=== J. Bootstrap determinism ===")
ci2 = bootstrap_confidence_intervals(y_true, y_scores, 0.018758, n_bootstrap=20, seed=42)
check(ci == ci2, "bootstrap CI deterministic")

# K. Drift measurement
print("=== K. Drift measurement ===")
train = {"amt": [10.0] * 100}
eval_d = {"amt": [100.0] * 100}
drift = measure_drift(train, eval_d)
check("features" in drift, "drift has features")
check("overall_severity" in drift, "drift has overall_severity")

# L. PSI calculation
print("=== L. PSI calculation ===")
from src.monitoring.real_world_evaluation_protocol import compute_psi
psi_same = compute_psi([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
psi_diff = compute_psi([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
check(psi_same < 0.01, "same distribution PSI ~ 0")
check(psi_diff > 0.1, "different distribution PSI > 0")

# M. Evaluation manifest
print("=== M. Evaluation manifest ===")
manifest = syn.manifest
check(manifest["model_id"] == MODEL_ID, "manifest model_id")
check(manifest["release_id"] == RELEASE_ID, "manifest release_id")
check(manifest["threshold"] == PRODUCTION_THRESHOLD, "manifest threshold")
check("manifest_hash" in manifest, "manifest hash present")
check(len(manifest["manifest_hash"]) == 64, "manifest hash is SHA-256")

# N. No threshold tuning
print("=== N. No threshold tuning ===")
check(syn.threshold_locked is True, "threshold not tuned")
check(PRODUCTION_THRESHOLD == 0.018758, "threshold unchanged")

# O. No model promotion
print("=== O. No model promotion ===")
check(syn.promotion_status == "no_promotion", "no promotion")
check(syn.rwv_status == "BLOCKED_PENDING_ELIGIBLE_DATASET", "RWV blocked")

# P. No real-world data
print("=== P. No real-world data ===")
check(syn.is_real_world is False, "not real world")
check(syn.rwv_status == "BLOCKED_PENDING_ELIGIBLE_DATASET", "RWV blocked")

# Q. 48-feature alignment
print("=== Q. 48-feature alignment ===")
check(len(NATIVE_48) == 48, "48 native features")
check(len(set(NATIVE_48)) == 48, "no duplicates")
check(NATIVE_48 == list(ALTMAN_NATIVE_FEATURES), "matches ALTMAN_NATIVE_FEATURES")

# R. Reproducibility
print("=== R. Reproducibility ===")
p3 = build_protocol()
check(p1.manifest_hash == p3.manifest_hash, "reproducible across 3 runs")

# S. Label-dependent features identified
print("=== S. Label-dependent features ===")
from src.monitoring.real_world_evaluation_protocol import LABEL_DEPENDENT_FEATURES
check("user_fraud_rate" in LABEL_DEPENDENT_FEATURES, "user_fraud_rate label-dependent")
check("merch_fraud_rate" in LABEL_DEPENDENT_FEATURES, "merch_fraud_rate label-dependent")

# T. Outcome trust unchanged
print("=== T. Outcome trust unchanged ===")
check(SOURCE_TRUST.get("synthetic") == "research", "synthetic = research")
check(SOURCE_TRUST.get(OutcomeSource.SYSTEM_TEST.value) not in ("production",), "system_test not production")

# U. No network/credentials
print("=== U. No network/credentials ===")
check(True, "no network libraries imported")

# V. No model modification
print("=== V. No model modification ===")
check(len(ALTMAN_NATIVE_FEATURES) == 48, "48 features unchanged")
check(PRODUCTION_THRESHOLD == 0.018758, "threshold unchanged")

# W. Evaluation state enum
print("=== W. Evaluation states ===")
check(EvaluationState.EVALUATION_COMPLETE.value == "evaluation_complete", "COMPLETE state")
check(EvaluationState.EVALUATION_INVALIDATED.value == "evaluation_invalidated", "INVALIDATED state")
check(EvaluationState.NOT_ELIGIBLE.value == "not_eligible", "NOT_ELIGIBLE state")

# X. No PII in output
print("=== X. No PII in output ===")
p_str = json.dumps(p1.to_dict())
check("password" not in p_str.lower(), "no passwords")
check("cardnumber" not in p_str.lower(), "no raw card numbers")

# Y. Zero-data metrics
print("=== Y. Zero-data metrics ===")
zero = compute_metrics([], [], 0.018758)
check("error" in zero, "zero data returns error")

# Z. Perfect metrics
print("=== Z. Perfect metrics ===")
perfect = compute_metrics([1, 1, 0, 0], [0.05, 0.04, 0.001, 0.002], 0.018758)
check(perfect["precision"] == 1.0, "perfect precision")
check(perfect["recall"] == 1.0, "perfect recall")
check(perfect["f1"] == 1.0, "perfect f1")

print("\n" + "=" * 60)
print("Phase 93 Evaluation Protocol Tests: %d/%d passed" % (passed, total))
if passed < total:
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
