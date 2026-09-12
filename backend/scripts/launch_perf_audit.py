# -*- coding: utf-8 -*-
"""Boots the perf stack (launch_perf_stack.py) then runs the check-#26 audit
(perf_reliability_audit.py), streaming both to perf_audit.log."""
import subprocess
import sys
import time

PY = sys.executable
with open("perf_audit.log", "w", encoding="utf-8", buffering=1) as f:
    f.write(f"launcher start {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
    f.flush()
    r0 = subprocess.run([PY, "scripts/launch_perf_stack.py"], stdout=f, stderr=subprocess.STDOUT)
    print(f"[boot exit code: {r0.returncode}]", file=f, flush=True)
    if r0.returncode == 0:
        r1 = subprocess.run([PY, "scripts/perf_reliability_audit.py"], stdout=f, stderr=subprocess.STDOUT)
        print(f"[audit exit code: {r1.returncode}]", file=f, flush=True)
print("launcher done")