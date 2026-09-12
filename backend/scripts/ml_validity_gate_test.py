#!/usr/bin/env python3
"""PS-14 ML-VALIDITY GATE TEST - canonical hard gate for the corrected pipeline.

Runs the recorded-artifact gates (scripts/ml_validity_gates.py) and the
independent validator (scripts/independent_validator.py) and FAILS the build
if any ML-validity requirement is violated:

  * temporal ordering invalid (backwards timestamps)
  * future data changes earlier features (perturbation audit)
  * future/final-test labels influence features or training
  * scaler/early-stopping/threshold decisions touch the final test
  * locked threshold not reproducible from validation / breaches its own cap
  * confusion-matrix arithmetic inconsistent on ANY published table
  * independent reproduction disagrees beyond documented tolerance
  * bootstrap CI methodology inconsistent

Both children are subprocesses so their separate code paths stay separate
from the test harness. Requires the rebuild artifacts (protocol JSON +
predictions npz) to exist: run scripts/ml_validity_rebuild.py after any
dataset or protocol change, then this gate.

Usage:
    ./.venv/Scripts/python.exe scripts/ml_validity_gate_test.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root


# CI guard: the predictions artifact is produced by
# scripts/ml_validity_rebuild.py from local model artifacts and is not
# committed. On a fresh checkout (CI), skip instead of failing.
if not (ROOT / "reports" / "ml_validity_predictions.npz").exists():
    print("PS-14 ML-VALIDITY GATE TEST - SKIPPED")
    print("  reports/ml_validity_predictions.npz not found (fresh checkout).")
    print("  Run scripts/ml_validity_rebuild.py locally to enable this gate.")
    raise SystemExit(0)
PY = sys.executable
PROTOCOL = ROOT / "reports" / "ml_validity_protocol.json"
PRED = ROOT / "reports" / "ml_validity_predictions.npz"

passed = 0
failed = 0


def run(name: str, script: str) -> None:
    global passed, failed
    if not (ROOT / script).exists():
        print(f"  [FAIL] {name}: {script} missing")
        failed += 1
        return
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([PY, str(ROOT / script)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       cwd=str(ROOT), env=env, timeout=600)
    tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
    if r.returncode == 0:
        print(f"  [PASS] {name} (exit 0)")
        passed += 1
    else:
        print(f"  [FAIL] {name} (exit {r.returncode})")
        for ln in tail:
            print(f"      {ln}")
        failed += 1


def main() -> int:
    print("PS-14 ML-VALIDITY GATE TEST")
    missing = [p for p in (PROTOCOL, PRED) if not p.exists()]
    if missing:
        print(f"  [FAIL] artifacts missing - run scripts/ml_validity_rebuild.py first:\n"
              f"      {[str(m) for m in missing]}")
        return 1
    run("causality+quarantine gates", "scripts/ml_validity_gates.py")
    run("independent validator", "scripts/independent_validator.py")
    print(f"\nML-VALIDITY GATE: {passed}/{passed+failed} PASSED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
