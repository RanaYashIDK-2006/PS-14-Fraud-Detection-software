#!/usr/bin/env python3
"""Test runtime drift detector — PSI computation, state transitions, fallback.

Run: .venv/Scripts/python.exe scripts/drift_detector_test.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

os.environ.setdefault("JWT_SECRET", "drift-detector-test-0123456789abcdef")

passed = failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  PASS {label}")
        passed += 1
    else:
        msg = f"  FAIL {label}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        failed += 1


def make_baseline(tmp_dir: Path) -> Path:
    """Create a deterministic baseline from 5000 samples."""
    from src.drift_monitor import psi as psi_mod
    from src.privacy_layer.features import ML_FEATURES

    rng = np.random.default_rng(42)
    baseline = {}
    for f in ML_FEATURES:
        vals = rng.normal(5.0, 1.0, 5000)
        edges = psi_mod.bin_edges(vals, 10)
        if edges is None:
            continue
        counts, _ = np.histogram(vals, bins=edges)
        baseline[f] = {
            "edges": [float(e) for e in edges],
            "expected": [float(x) for x in counts / counts.sum()],
            "n": 5000,
        }
    path = tmp_dir / "baseline.json"
    path.write_text(json.dumps(baseline))
    return path


def main():
    global passed, failed

    from src.privacy_layer.features import ML_FEATURES
    from src.risk_engine.drift_detector import DriftDetector, NORMAL, WARNING, CRITICAL

    print("== Runtime drift detector tests ==\n")

    tmp_dir = Path(tempfile.mkdtemp())
    baseline_path = make_baseline(tmp_dir)

    # --- 1. No baseline (passthrough) ---
    print("--- 1. No baseline (passthrough) ---")
    det = DriftDetector(baseline_path=None, window_size=100, check_interval=10)
    check("state starts NORMAL", det.state == NORMAL)
    check("should_fallback is False", det.should_fallback is False)
    for i in range(50):
        det.record({f: float(i % 10) for f in ML_FEATURES})
    check("still NORMAL without baseline", det.state == NORMAL)
    check("total_events counted", det.total_events == 50)

    # --- 2. Baseline loads correctly ---
    print("\n--- 2. Baseline loading ---")
    det2 = DriftDetector(baseline_path=baseline_path, window_size=200, check_interval=100)
    check("baseline loaded", det2._baseline is not None)
    check(f"baseline has {len(det2._baseline)} features", len(det2._baseline) >= 10)

    # --- 3. Normal data stays NORMAL ---
    print("\n--- 3. Normal data (same distribution) ---")
    rng = np.random.default_rng(99)
    det3 = DriftDetector(baseline_path=baseline_path, window_size=500, check_interval=500)
    for i in range(500):
        row = {f: float(rng.normal(5.0, 1.0)) for f in ML_FEATURES}
        det3.record(row)
    # With check_interval=500, only 1 check ran at event 500
    check("state is NORMAL for matching data", det3.state == NORMAL, f"got {det3.state}")
    check("should_fallback is False", det3.should_fallback is False)
    if det3.max_psi > 0:
        check("max_psi below warn threshold", det3.max_psi < 0.10, f"max_psi={det3.max_psi:.4f}")

    # --- 4. Drifted data transitions to WARNING/CRITICAL ---
    print("\n--- 4. Drifted data (shifted distribution) ---")
    det4 = DriftDetector(baseline_path=baseline_path, window_size=300, check_interval=100)
    # Fill window with normal data (need > check_interval before drift)
    rng2 = np.random.default_rng(100)
    for i in range(150):
        row = {f: float(rng2.normal(5.0, 1.0)) for f in ML_FEATURES}
        det4.record(row)
    # Inject heavily shifted data (mean=20 instead of 5)
    for i in range(300):
        row = {f: float(rng2.normal(20.0, 5.0)) for f in ML_FEATURES}
        det4.record(row)
    check(
        "state transitions to WARNING or CRITICAL",
        det4.state in (WARNING, CRITICAL),
        f"got {det4.state}",
    )
    check("max_psi > 0", det4.max_psi > 0.0, f"max_psi={det4.max_psi}")
    if det4.state == CRITICAL:
        check("should_fallback is True", det4.should_fallback is True)
        check("is_critical is True", det4.is_critical is True)

    # --- 5. status() snapshot ---
    print("\n--- 5. status() snapshot ---")
    status = det4.status()
    for key in ("state", "max_psi", "total_events", "baseline_loaded", "thresholds", "window_size"):
        check(f"status has '{key}'", key in status)
    check("baseline_loaded is True", status["baseline_loaded"] is True)

    # --- 6. Reset ---
    print("\n--- 6. Reset ---")
    det4.reset()
    check("state resets to NORMAL", det4.state == NORMAL)
    check("should_fallback resets to False", det4.should_fallback is False)
    check("max_psi resets to 0", det4.max_psi == 0.0)
    check("window cleared", len(det4._window) == 0)
    check("total_events resets to 0", det4.total_events == 0)

    # --- 7. Reload baseline ---
    print("\n--- 7. Reload baseline ---")
    det5 = DriftDetector(baseline_path=None, window_size=100, check_interval=10)
    check("no baseline initially", det5._baseline is None)
    det5.reload_baseline(baseline_path)
    check("baseline reloaded", det5._baseline is not None)
    check("state reset after reload", det5.state == NORMAL)

    # --- 8. Insufficient data ---
    print("\n--- 8. Insufficient data (< 30 events) ---")
    det6 = DriftDetector(baseline_path=baseline_path, window_size=200, check_interval=10)
    rng3 = np.random.default_rng(101)
    for i in range(25):
        row = {f: float(rng3.normal(20.0, 5.0)) for f in ML_FEATURES}
        det6.record(row)
    check("state stays NORMAL with < 30 events", det6.state == NORMAL, f"got {det6.state}")

    # --- 9. PSI accuracy for identical data ---
    print("\n--- 9. PSI accuracy ---")
    det7 = DriftDetector(baseline_path=baseline_path, window_size=500, check_interval=500)
    rng4 = np.random.default_rng(42)
    for i in range(500):
        row = {f: float(rng4.normal(5.0, 1.0)) for f in ML_FEATURES}
        det7.record(row)
    per = det7.per_feature_psi
    if per:
        max_feat_psi = max(per.values())
        check(
            "identical distribution PSI < 0.10",
            max_feat_psi < 0.10,
            f"max={max_feat_psi:.4f}",
        )
    else:
        check("PSI computed (non-empty)", False, "per_feature_psi is empty")

    # --- 10. State recovery ---
    print("\n--- 10. State recovery ---")
    det8 = DriftDetector(baseline_path=baseline_path, window_size=200, check_interval=50)
    # Push to CRITICAL
    rng5 = np.random.default_rng(102)
    for i in range(200):
        row = {f: float(rng5.normal(30.0, 10.0)) for f in ML_FEATURES}
        det8.record(row)
    was_critical = det8.state in (WARNING, CRITICAL)
    # Now push many normal events to fully fill the window and recover
    rng6 = np.random.default_rng(103)
    for i in range(300):  # 3x window_size to guarantee full turnover
        row = {f: float(rng6.normal(5.0, 1.0)) for f in ML_FEATURES}
        det8.record(row)
    check(
        "recovers to NORMAL after normal data",
        det8.state == NORMAL,
        f"was_critical={was_critical}, now={det8.state}",
    )

    # Cleanup
    baseline_path.unlink(missing_ok=True)
    tmp_dir.rmdir()

    # Summary
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
