#!/usr/bin/env python3
"""Launch calibration_and_stress.py with proper stdout/stderr capture on Windows."""
import subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

script = os.path.join(ROOT, "scripts", "calibration_and_stress.py")
log_out = os.path.join(ROOT, "calibration_stress.log")
log_err = os.path.join(ROOT, "calibration_stress_err.log")

with open(log_out, "w") as f_out, open(log_err, "w") as f_err:
    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=f_out,
        stderr=f_err,
        cwd=ROOT,
    )
print(f"Launched calibration_and_stress.py with PID {proc.pid}")
