#!/usr/bin/env python3
"""Launch forensic_audit.py as a detached subprocess."""
import subprocess, sys, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

log_path = os.path.join(ROOT, "forensic_audit.log")
err_path = os.path.join(ROOT, "forensic_audit_err.log")

# Clear old logs
for p in [log_path, err_path]:
    if os.path.exists(p):
        os.remove(p)

print(f"Launching forensic_audit.py...")
print(f"  stdout -> {log_path}")
print(f"  stderr -> {err_path}")

# Launch detached
with open(log_path, "w") as log_f, open(err_path, "w") as err_f:
    proc = subprocess.Popen(
        [sys.executable, "scripts/forensic_audit.py"],
        stdout=log_f,
        stderr=err_f,
        cwd=ROOT,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

print(f"  PID: {proc.pid}")
print(f"  Waiting 5s to verify...")
time.sleep(5)

# Check if still running
poll = proc.poll()
if poll is None:
    print(f"  [OK] Process {proc.pid} is RUNNING")
else:
    print(f"  [FAIL] Process exited with code {poll}")
    # Show any error output
    if os.path.exists(err_path) and os.path.getsize(err_path) > 0:
        with open(err_path) as f:
            print(f"  stderr: {f.read()[:500]}")
