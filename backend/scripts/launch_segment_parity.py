#!/usr/bin/env python3
"""Launch the segment audit + production parity scripts detached on Windows."""
import subprocess, sys, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

jobs = [
    ("scripts/segment_performance_audit.py", "segment_audit.log", "segment_audit_err.log"),
    ("scripts/production_parity.py", "parity.log", "parity_err.log"),
]
pids = {}
for script, out, err in jobs:
    with open(out, "w") as f_out, open(err, "w") as f_err:
        p = subprocess.Popen([sys.executable, script], stdout=f_out, stderr=f_err, cwd=ROOT)
        pids[script] = p.pid
        print(f"Launched {script} with PID {p.pid}")
    time.sleep(2)
print("pids:", pids)