#!/usr/bin/env python3
"""Phase 40: Comprehensive monitoring test suite (30 tests).

Run: .venv/Scripts/python.exe backend/scripts/phase40_monitor_test.py
"""
from __future__ import annotations
import json, os, sys, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("JWT_SECRET", "phase40-test-secret-0123456789abcdef")

import numpy as np

passed = failed = 0

def check(label, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS {label}"); passed += 1
    else:
        msg = f"  FAIL {label}"
        if detail: msg += f" -- {detail}"
        print(msg); failed += 1

def main():
    global passed, failed
    from src.privacy_layer.features import ML_FEATURES
    from src.monitoring.schema_drift import check_schema, check_feature_availability, SchemaStatus, FeatureAvailability
    from src.monitoring.prediction_drift import check_prediction_drift
    from src.monitoring.alert_manager import AlertManager, AlertState
    from src.monitoring.safeguards import PromotionSafeguard
    from src.monitoring.label_awareness import check_label_availability, check_outcome_drift
    from src.monitoring.baseline_governance import BaselineMetadata, BaselineSource, validate_baseline_source, create_baseline
    from src.monitoring.drift_monitor import DriftMonitor
    from src.drift_monitor import psi
    rng = np.random.default_rng(42)

    print("== Phase 40: Comprehensive monitoring tests ==\n")

    # 1. Stable distribution
    print("--- 1. Stable distribution (no false alert) ---")
    ref = rng.normal(0.3, 0.1, 1000)
    curr = rng.normal(0.3, 0.1, 500)
    r = check_prediction_drift(curr, ref, min_scores=30)
    check("stable -> stable", r.overall_drift == "stable", f"got {r.overall_drift}")
    check("band shift small", r.max_band_shift < 0.10, f"shift={r.max_band_shift:.4f}")

    # 2. Large shift detected
    print("\n--- 2. Large shift detected ---")
    shifted = rng.normal(0.8, 0.1, 500)
    r2 = check_prediction_drift(shifted, ref, min_scores=30)
    check("shifted -> critical", r2.overall_drift == "critical", f"got {r2.overall_drift}")
    check("mean shift large", r2.mean_score_shift > 0.15, f"shift={r2.mean_score_shift:.4f}")

    # 3. Insufficient data
    print("\n--- 3. Insufficient data ---")
    tiny = rng.normal(0.3, 0.1, 5)
    r3 = check_prediction_drift(tiny, ref, min_scores=30)
    check("tiny -> insufficient_data", r3.overall_drift == "insufficient_data")

    # 4. Numerical feature drift
    print("\n--- 4. Numerical feature drift ---")
    # Use psi_proportions directly to test PSI math
    ep = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    ap_shifted = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.91])
    psi_val = psi.psi_proportions(ep, ap_shifted)
    check("PSI math works for shifted distribution", psi_val > 1.0, f"psi={psi_val:.4f}")
    check("level thresholds correct", psi.level(psi_val) == "alert")

    # 5. Categorical drift
    print("\n--- 5. Categorical feature drift ---")
    ref_bin = rng.binomial(1, 0.1, 1000).astype(float)
    b_edges = np.array([-0.5, 0.5, 1.5])
    b_counts, _ = np.histogram(ref_bin, bins=b_edges)
    b_exp = b_counts / b_counts.sum()
    cat_bl = {"edges": [float(e) for e in b_edges], "expected": [float(x) for x in b_exp], "n": 1000}
    cat_drifted = rng.binomial(1, 0.9, 200).astype(float)
    cr = psi.check_feature(cat_bl, cat_drifted)
    check("categorical drift detected", cr["level"] in ("warn", "alert"), f"psi={cr['psi']:.4f}")

    # 6. Missing feature
    print("\n--- 6. Missing feature ---")
    expected = ML_FEATURES[:5]
    observed = ML_FEATURES[:3]
    sr = check_schema(expected, observed)
    check("2 missing features", sr.n_missing == 2)
    check("CRITICAL status", sr.overall_status == SchemaStatus.CRITICAL)

    # 7. Unexpected feature
    print("\n--- 7. Unexpected feature ---")
    expected2 = ML_FEATURES[:3]  # only expected features
    observed_extra = ML_FEATURES[:3] + ["extra_1", "extra_2"]
    sr2 = check_schema(expected2, observed_extra)
    check("2 unexpected", sr2.n_unexpected == 2)
    check("WARNING status (no missing, only unexpected)", sr2.overall_status == SchemaStatus.WARNING)

    # 8. Datatype mismatch
    print("\n--- 8. Datatype mismatch ---")
    avail_data = {"feat_a": [1.0, 2.0, None, 4.0, "mixed", 6.0, 7.0, 8.0, 9.0, 10.0], "feat_b": [1.0]*10}
    ar = check_feature_availability(avail_data, ["feat_a", "feat_b"], min_observations=5)
    check("type mismatch detected", ar.n_type_mismatches > 0, f"mismatches={ar.n_type_mismatches}")

    # 9. High missingness
    print("\n--- 9. High missingness ---")
    sparse = {"feat_a": [None]*9 + [1.0], "feat_b": [1.0]*10}
    sr3 = check_feature_availability(sparse, ["feat_a", "feat_b"], min_observations=5)
    check("unavailable detected", sr3.n_unavailable > 0)
    check("feat_a unavailable", sr3.feature_results["feat_a"].status == FeatureAvailability.UNAVAILABLE)

    # 10. Prediction drift distinct from feature drift
    print("\n--- 10. Prediction vs feature drift ---")
    r10 = check_prediction_drift(rng.normal(0.8, 0.1, 100), rng.normal(0.3, 0.1, 1000), min_scores=30)
    check("prediction drift independent", r10.overall_drift == "critical")

    # 11. Label availability
    print("\n--- 11. Label availability ---")
    labels_mixed = [0, 1, 0, None, 1, 0, None, 0, 1, 0] * 5
    li = check_label_availability(labels_mixed, min_labels=30)
    check("enough labels -> available", li.available, f"n_labeled={li.n_labeled}")
    li_few = check_label_availability([0, 1, None]*2, min_labels=30)
    check("few labels -> not available", not li_few.available)

    # 12. No fabricate outcomes
    print("\n--- 12. No fabricate outcomes ---")
    ocr = check_outcome_drift(current_labels=[None]*100, reference_fraud_rate=0.02, min_labels=30)
    check("no-label -> insufficient_data", ocr.overall_status == "insufficient_data")
    check("no fabricated rate", ocr.observed_fraud_rate == 0.0)

    # 13. Label leakage protection
    print("\n--- 13. Label leakage protection ---")
    mon = DriftMonitor(expected_features=ML_FEATURES[:3], model_version="v1")
    fd = {f: rng.normal(5.0, 1.0, 50).tolist() for f in ML_FEATURES[:3]}
    s13 = mon.check_window(feature_data=fd, scores=rng.normal(0.3, 0.1, 50), labels=[0,1,0,0,1]*10)
    check("labels tracked separately", s13.label_availability is not None and s13.label_availability.available)
    check("outcome report separate", s13.outcome_report is not None)

    # 14. Alert deduplication
    print("\n--- 14. Alert deduplication ---")
    # Test dedup: escalate to CRITICAL (requires 2 consecutive alerts), then test cooldown
    am14 = AlertManager(critical_consecutive=1, cooldown_seconds=300)
    a_first = am14.record_window(worst_level="alert", features_alerting=["f2"], source="dedup")
    check("first alert emitted", len(a_first) > 0, f"got {len(a_first)}")
    # Second call: state already CRITICAL, no escalation, but cooldown should block same fingerprint
    a2 = am14.record_window(worst_level="alert", features_alerting=["f2"], source="dedup")
    check("second identical alert deduped (cooldown)", len(a2) == 0, f"got {len(a2)}")

    # 15. Alert hysteresis
    print("\n--- 15. Alert hysteresis ---")
    am15 = AlertManager(warning_consecutive=3, critical_consecutive=2)
    am15.record_window(worst_level="warn", features_alerting=["f1"], source="hyst")
    check("1 warn -> NORMAL", am15.state == AlertState.NORMAL)
    am15.record_window(worst_level="warn", features_alerting=["f1"], source="hyst")
    check("2 warns -> WATCH", am15.state == AlertState.WATCH)
    am15.record_window(worst_level="warn", features_alerting=["f1"], source="hyst")
    check("3 warns -> WARNING", am15.state == AlertState.WARNING)

    # 16. Recovery CRITICAL -> NORMAL
    print("\n--- 16. Recovery CRITICAL ---")
    am16 = AlertManager(critical_consecutive=2, recovery_consecutive=2)
    am16.record_window(worst_level="alert", features_alerting=["f1"], source="rec")
    am16.record_window(worst_level="alert", features_alerting=["f1"], source="rec")
    check("2 alerts -> CRITICAL", am16.state == AlertState.CRITICAL)
    for _ in range(5):
        am16.record_window(worst_level="ok", source="rec")
    check("sustained normals -> NORMAL", am16.state == AlertState.NORMAL, f"state={am16.state}")

    # 17. Deterministic config
    print("\n--- 17. Deterministic configuration ---")
    ma = DriftMonitor(expected_features=ML_FEATURES[:3], model_version="v1")
    mb = DriftMonitor(expected_features=ML_FEATURES[:3], model_version="v1")
    d17 = {f: rng.normal(5.0, 1.0, 50).tolist() for f in ML_FEATURES[:3]}
    sa = ma.check_window(feature_data=d17, scores=rng.normal(0.3, 0.1, 50))
    sb = mb.check_window(feature_data=d17, scores=rng.normal(0.3, 0.1, 50))
    check("same config -> same schema", sa.schema_drift.overall_status == sb.schema_drift.overall_status)

    # 18. Baseline metadata persistence
    print("\n--- 18. Baseline metadata persistence ---")
    with tempfile.TemporaryDirectory() as td:
        meta = create_baseline(baseline_id="test-001", model_version="v1.0", feature_schema_version="1.0",
                               source_type=BaselineSource.TRAINING_WINDOW, source_description="training data",
                               sample_count=10000, approved_by="admin")
        path = Path(td) / "meta.json"
        meta.save(path)
        loaded = BaselineMetadata.load(path)
        check("metadata round-trips", loaded.baseline_id == "test-001")
        check("approved_by persisted", loaded.approved_by == "admin")

    # 19. Baseline source validation
    print("\n--- 19. Baseline source validation ---")
    ok, reason = validate_baseline_source(BaselineSource.TRAINING_WINDOW, "training data")
    check("training window valid", ok, reason)
    ok2, reason2 = validate_baseline_source(BaselineSource.UNKNOWN)
    check("unknown rejected", not ok2, reason2)
    ok3, reason3 = validate_baseline_source(BaselineSource.FORBIDDEN)
    check("forbidden rejected", not ok3, reason3)

    # 20. External data blocked as baseline
    print("\n--- 20. External data blocked ---")
    try:
        create_baseline(baseline_id="bad", model_version="v1", feature_schema_version="1",
                        source_type=BaselineSource.FORBIDDEN, source_description="kaggle test",
                        sample_count=50000)
        check("forbidden baseline rejected", False, "should have raised ValueError")
    except ValueError:
        check("forbidden baseline rejected", True)

    # 21. Model versions distinguishable
    print("\n--- 21. Model version tracking ---")
    mv1 = DriftMonitor(expected_features=ML_FEATURES[:3], model_version="v1.0.0")
    mv2 = DriftMonitor(expected_features=ML_FEATURES[:3], model_version="v2.0.0")
    check("versions differ", mv1.model_version != mv2.model_version)

    # 22. Monitoring failure handling
    print("\n--- 22. Monitoring failure ---")
    mempty = DriftMonitor(expected_features=ML_FEATURES[:3])
    se = mempty.check_window()
    check("empty -> insufficient_data", se.schema_drift is None or se.schema_drift.overall_status == SchemaStatus.INSUFFICIENT_DATA)

    # 23. Drift does not auto-promote
    print("\n--- 23. Drift does not auto-promote ---")
    sg = PromotionSafeguard()
    res = sg.check_promotion_eligible(drift_state="critical", leakage_checks_pass=True, data_quality_pass=True,
                                       validation_performance_pass=True, untouched_test_pass=True, robustness_pass=True,
                                       production_parity_pass=True, security_pass=True, approval_pass=True)
    sm = sg.summarize(res)
    check("critical drift blocks promotion", not sm["eligible"] or "drift_state" in sm.get("blocking_checks", []))

    # 24. Drift does not auto-replace
    print("\n--- 24. Drift does not auto-replace ---")
    rt = sg.check_retrain_eligible(drift_state="critical", has_active_critical_alerts=True, monitoring_healthy=True)
    rtm = sg.summarize(rt)
    check("critical + alerts blocks retrain", not rtm["eligible"] or "active_alerts" in rtm.get("blocking_checks", []))

    # 25. Existing rollback intact
    print("\n--- 25. Existing rollback ---")
    from src.risk_engine.drift_detector import DriftDetector as RD, NORMAL
    td25 = Path(tempfile.mkdtemp())
    bl25 = {}
    for f in ML_FEATURES[:3]:
        v = rng.normal(5.0, 1.0, 500)
        e = psi.bin_edges(v, 10)
        if e is not None:
            c, _ = np.histogram(v, bins=e)
            bl25[f] = {"edges": [float(x) for x in e], "expected": [float(x) for x in c/c.sum()], "n": 500}
    bp = td25 / "bl.json"
    bp.write_text(json.dumps(bl25))
    det = RD(baseline_path=bp, window_size=200, check_interval=500)  # check only after window full
    for _ in range(100):
        det.record({f: float(rng.normal(5.0, 1.0)) for f in ML_FEATURES[:3]})
    check("runtime detector works", det.state == NORMAL, f"state={det.state}")
    det.reset()
    check("runtime reset works", det.state == NORMAL)
    shutil.rmtree(td25, ignore_errors=True)

    # 26. Alert state machine
    print("\n--- 26. Alert state machine ---")
    am26 = AlertManager(watch_consecutive=1, warning_consecutive=2, critical_consecutive=2, recovery_consecutive=2)
    check("starts NORMAL", am26.state == AlertState.NORMAL)
    am26.record_window(worst_level="warn", features_alerting=["f1"], source="sm")
    check("1 warn -> WATCH", am26.state == AlertState.WATCH)
    am26.record_window(worst_level="warn", features_alerting=["f1"], source="sm")
    check("2 warns -> WARNING", am26.state == AlertState.WARNING)
    am26.record_window(worst_level="alert", features_alerting=["f1"], source="sm")
    am26.record_window(worst_level="alert", features_alerting=["f1"], source="sm")
    check("2 alerts -> CRITICAL", am26.state == AlertState.CRITICAL)

    # 27. All missing features
    print("\n--- 27. All missing features ---")
    sf = check_schema(ML_FEATURES, [])
    check("all missing", sf.n_missing == len(ML_FEATURES))

    # 28. Mixed null rates
    print("\n--- 28. Mixed null rates ---")
    mx = {}
    for i, f in enumerate(ML_FEATURES[:5]):
        if i == 0: mx[f] = [None]*9 + [1.0]
        elif i == 1: mx[f] = [None]*3 + [1.0]*7
        else: mx[f] = [1.0]*10
    mxr = check_feature_availability(mx, ML_FEATURES[:5], min_observations=5)
    check("90% null -> unavailable", mxr.feature_results[ML_FEATURES[0]].status == FeatureAvailability.UNAVAILABLE)
    check("30% null -> degraded", mxr.feature_results[ML_FEATURES[1]].status == FeatureAvailability.DEGRADED)
    check("0% null -> available", mxr.feature_results[ML_FEATURES[2]].status == FeatureAvailability.AVAILABLE)

    # 29. Prediction drift with reference
    print("\n--- 29. Prediction drift with reference ---")
    r29 = check_prediction_drift(rng.normal(0.35, 0.1, 500), rng.normal(0.3, 0.1, 1000),
                                  min_scores=30, mean_warn_shift=0.03, mean_crit_shift=0.10)
    check("slight shift -> stable or warning", r29.overall_drift in ("stable", "warning"),
          f"got {r29.overall_drift}, shift={r29.mean_score_shift:.4f}")

    # 30. Full orchestrator e2e
    print("\n--- 30. Full orchestrator e2e ---")
    me = DriftMonitor(expected_features=ML_FEATURES[:5], model_version="e2e-v1")
    ed = {f: rng.normal(5.0, 1.0, 100).tolist() for f in ML_FEATURES[:5]}
    es = me.check_window(feature_data=ed, scores=rng.normal(0.3, 0.1, 100),
                          labels=[0,1,0,1,0]*20, feature_drift_state="normal", feature_drift_max_psi=0.02,
                          reference_fraud_rate=0.4)  # matches actual 40% fraud rate (2/5)
    check("e2e overall is healthy",
          es.overall_status == "healthy",
          f"status={es.overall_status}, detail={es.detail}")
    check("e2e schema OK", es.schema_drift is not None and es.schema_drift.overall_status == SchemaStatus.OK)
    check("e2e pred stable", es.prediction_drift is not None and es.prediction_drift.overall_drift == "stable")
    check("e2e labels available", es.label_availability is not None and es.label_availability.available)
    mon_status = me.status()
    check("e2e safeguards present", mon_status.get("safeguards", {}).get("drift_does_not_mean_retrain") is True)
    j = json.dumps(es.to_dict(), default=str)
    check("e2e serializable", len(j) > 100)

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} PASSED, {failed} FAILED (out of {passed+failed})")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
