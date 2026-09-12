#!/usr/bin/env python3
"""Launch forensic_revalidate.py with proper stdout/stderr capture on Windows."""
import subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

script = os.path.join(ROOT, "scripts", "forensic_revalidate.py")
log_out = os.path.join(ROOT, "forensic_reval.log")
log_err = os.path.join(ROOT, "forensic_reval_err.log")

# Use subprocess.Popen with file handles for reliable Windows redirect
with open(log_out, "w") as f_out, open(log_err, "w") as f_err:
    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=f_out,
        stderr=f_err,
        cwd=ROOT,
    )
print(f"Launched forensic_revalidate.py with PID {proc.pid}")
