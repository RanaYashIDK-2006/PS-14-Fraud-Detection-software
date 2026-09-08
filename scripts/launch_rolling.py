#!/usr/bin/env python3
"""Launch rolling_validation_and_monitoring.py with proper stdout/stderr capture on Windows."""
import subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

script = os.path.join(ROOT, "scripts", "rolling_validation_and_monitoring.py")
log_out = os.path.join(ROOT, "rolling_validation.log")
log_err = os.path.join(ROOT, "rolling_validation_err.log")

with open(log_out, "w") as f_out, open(log_err, "w") as f_err:
    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=f_out,
        stderr=f_err,
        cwd=ROOT,
    )
print(f"Launched rolling_validation_and_monitoring.py with PID {proc.pid}")