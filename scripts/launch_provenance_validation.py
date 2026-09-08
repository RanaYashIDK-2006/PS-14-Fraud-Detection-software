# -*- coding: utf-8 -*-
"""Launcher: runs checks #23 (data_provenance) then #24 (independent_validation)
sequentially, streaming output live to provenance_validation.log."""
import subprocess
import sys
import time

PY = sys.executable
with open("provenance_validation.log", "w", encoding="utf-8", buffering=1) as f:
    f.write(f"launcher start {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
    f.flush()

    print("=" * 70, file=f, flush=True)
    print("CHECK #23: DATA PROVENANCE & REPRODUCIBILITY", file=f, flush=True)
    print("=" * 70, file=f, flush=True)
    r1 = subprocess.run([PY, "scripts/data_provenance.py"], stdout=f, stderr=subprocess.STDOUT)
    print(f"\n[check23 exit code: {r1.returncode}]", file=f, flush=True)

    print("=" * 70, file=f, flush=True)
    print("CHECK #24: INDEPENDENT FINAL VALIDATION", file=f, flush=True)
    print("=" * 70, file=f, flush=True)
    r2 = subprocess.run([PY, "scripts/independent_validation.py"], stdout=f, stderr=subprocess.STDOUT)
    print(f"\n[check24 exit code: {r2.returncode}]", file=f, flush=True)

print("launcher done")