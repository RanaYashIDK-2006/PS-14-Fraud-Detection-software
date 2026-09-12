# -*- coding: utf-8 -*-
"""Boot identity(:8001), privacy(:8002), risk(:8003) on an isolated DB_DIR
(db/perf_audit) for the check-#26 perf/reliability audit. Writes PIDs and logs,
waits for /health on all three. Services are left running (teardown is done by
the audit harness or kill_perf_stack())."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DB_DIR = ROOT / "db" / "perf_audit"
PIDFILE = ROOT / "db" / "perf_stack_pids.json"
LOG_DIR = ROOT / "db" / "perf_audit"

SERVICES = [
    {"name": "identity", "port": 8001, "module": "src.identity_service.main:app"},
    {"name": "privacy", "port": 8002, "module": "src.privacy_layer.main:app"},
    {"name": "risk", "port": 8003, "module": "src.risk_engine.main:app"},
]

PY = os.environ.get("PERF_PY", str(ROOT / ".venv" / "Scripts" / "python.exe"))


def env_with():
    e = dict(os.environ)
    e["DB_DIR"] = str(DB_DIR)
    e["PS14_MODE"] = "development"
    e.pop("SERVICE_NAME", None)  # avoid the docker-only production gate
    return e


def port_free(port):
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "LISTENING" in line and f":{port}" in line:
            return False
    return True


def main():
    if not port_free(8001) or not port_free(8002) or not port_free(8003):
        print("ERROR: ports 8001-8003 busy - stop the dev stack first", flush=True)
        sys.exit(1)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # clean state
    for f in LOG_DIR.glob("*.db*"):
        f.unlink()
    pids = {}
    for svc in SERVICES:
        logf = open(LOG_DIR / f"{svc['name']}.log", "w", encoding="utf-8", buffering=1)
        p = subprocess.Popen(
            [PY, "-m", "uvicorn", svc["module"], "--port", str(svc["port"])],
            cwd=str(ROOT), env=env_with(), stdout=logf, stderr=subprocess.STDOUT,
        )
        pids[svc["name"]] = {"pid": p.pid, "port": svc["port"]}
        print(f"  {svc['name']} started pid={p.pid} port={svc['port']}", flush=True)
    PIDFILE.write_text(json.dumps(pids, indent=1))

    t0 = time.time()
    for svc in SERVICES:
        ok = False
        while time.time() - t0 < 120:
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://127.0.0.1:{svc['port']}/health", timeout=2) as r:
                    if r.status == 200:
                        ok = True
                        break
            except Exception:
                time.sleep(2)
        print(f"  {svc['name']} healthy in {time.time()-t0:.0f}s (ok={ok})", flush=True)
        if not ok:
            print("  --- tail of log ---")
            print((LOG_DIR / f"{svc['name']}.log").read_text()[-1500:])
            sys.exit(2)
    print("STACK READY", flush=True)


if __name__ == "__main__":
    main()