"""Launcher that starts max_capacity_test.py and writes output to log."""
import subprocess, sys, os, time

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)

log_path = os.path.join(root, "freebuff_mc.log")
err_path = os.path.join(root, "freebuff_mc_err.log")

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

print(f"Launching max_capacity_test.py at {time.strftime('%H:%M:%S')}", flush=True)
print(f"Log: {log_path}", flush=True)

proc = subprocess.Popen(
    [sys.executable, "-u", "scripts/max_capacity_test.py"],
    stdout=open(log_path, "w", encoding="utf-8"),
    stderr=open(err_path, "w", encoding="utf-8"),
    env=env,
    cwd=root,
)

print(f"PID: {proc.pid}", flush=True)
print(f"Waiting for completion...", flush=True)

proc.wait()

print(f"Exit code: {proc.returncode}", flush=True)
print(f"Finished at {time.strftime('%H:%M:%S')}", flush=True)
