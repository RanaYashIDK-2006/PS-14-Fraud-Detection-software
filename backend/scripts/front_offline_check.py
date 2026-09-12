#!/usr/bin/env python3
"""LIVE offline-path check for the front page /status (opt-in, disruptive).

Stops the five backend services (identity/privacy/risk/verify/audit), verifies
the running front page answers /status with every service "down" and no audit
chain (the badge renders "Audit chain: unavailable"), then ALWAYS restores the
stack — try/finally, so a failed assertion still brings the services back.

Unlike `scripts/front_service_test.py` (hermetic: monkeypatched URLs, safe to
run in the pipeline), this script touches the real stack on ports 8001-8005.
Do NOT add it to ALL_SUITES. The front page itself (8000) is left running.

Usage (from the project root, stack up):
  python scripts/front_offline_check.py run      # stop -> assert -> restore
  python scripts/front_offline_check.py stop     # stop the five backends only
  python scripts/front_offline_check.py check    # assert /status shows all-down
  python scripts/front_offline_check.py restore  # bring the five backends back

Exit codes: 0 ok, 1 an assertion failed (stack still restored), 2 usage/health
error (nothing was stopped).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
FRONT_URL = "http://127.0.0.1:8000"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

# (name, port, uvicorn module) — the five backend services, never the front.
BACKENDS = [
    ("identity", 8001, "src.identity_service.main"),
    ("privacy", 8002, "src.privacy_layer.main"),
    ("risk", 8003, "src.risk_engine.main"),
    ("verify", 8004, "src.verification_service.main"),
    ("audit", 8005, "src.audit_service.main"),
]

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def _pid_on_port(port: int) -> int | None:
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                             timeout=15).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    for line in out.splitlines():
        if f":{port} " in line and "LISTENING" in line:
            pid = line.split()[-1]
            if pid.isdigit() and pid != "0":
                return int(pid)
    return None


def _front_up() -> bool:
    try:
        with urllib.request.urlopen(f"{FRONT_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _fetch_status() -> dict:
    with urllib.request.urlopen(f"{FRONT_URL}/status", timeout=10) as r:
        return json.load(r)


def stop() -> None:
    print("== stopping the five backends ==")
    for name, port, _mod in BACKENDS:
        pid = _pid_on_port(port)
        if pid is None:
            print(f"  {name}:{port} — nothing listening, nothing to stop")
            continue
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True, text=True)
        print(f"  {name}:{port} stopped (pid {pid})")
    time.sleep(1.5)  # let the listeners fully release the ports


def check_offline() -> None:
    print("== asserting /status shows all backends down ==")
    if not _front_up():
        check("front page still up", False, "front service on :8000 is not answering")
        return
    body = _fetch_status()
    check("offline /status still 200", True)
    check("all five keys present",
          set(body) == {n for n, _p, _m in BACKENDS},
          str(sorted(body)))
    check("every backend reports down",
          all(v.get("status") == "down" for v in body.values()),
          str({k: v.get("status") for k, v in body.items()}))
    check("audit entry carries no chain (badge -> unavailable)",
          "chain" not in body.get("audit", {}),
          str(body.get("audit")))


def restore() -> None:
    print("== restoring the five backends ==")
    ps = (
        f"$py = '{PY.as_posix()}'; $wd = '{ROOT.as_posix()}'; "
        + "; ".join(
            f"Start-Process -FilePath $py -ArgumentList '-m','uvicorn','{mod}:app',"
            f"'--port','{port}' -WorkingDirectory $wd -WindowStyle Hidden"
            for _n, port, mod in BACKENDS
        )
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   capture_output=True, text=True)
    # The Risk Engine takes a few seconds to load model artifacts before it
    # answers; poll all five until healthy (or give up after ~40s).
    deadline = time.monotonic() + 40
    healthy = {n: False for n, _p, _m in BACKENDS}
    while time.monotonic() < deadline and not all(healthy.values()):
        for name, port, _mod in BACKENDS:
            if healthy[name]:
                continue
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    healthy[name] = r.status == 200
            except Exception:
                pass
        if not all(healthy.values()):
            time.sleep(2)
    for name, port, _mod in BACKENDS:
        print(f"  {name}:{port} {'healthy' if healthy[name] else 'NOT UP'}")
    check("all five backends restored healthy", all(healthy.values()),
          str(healthy))


def run() -> int:
    if not _front_up():
        print("ERROR: the front page (:8000) is not answering — start it first.", file=sys.stderr)
        return 2
    stop()
    try:
        check_offline()
    finally:
        restore()  # never leave the stack down, even if an assertion failed
    if failures:
        print(f"\n{len(failures)} CHECK(S) FAILED (stack restored)")
        return 1
    print("\nALL LIVE OFFLINE CHECKS PASSED — stack restored")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "stop":
        stop()
        return 0
    if cmd == "check":
        check_offline()
        return 1 if failures else 0
    if cmd == "restore":
        restore()
        return 1 if failures else 0
    if cmd == "run":
        return run()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
