#!/usr/bin/env python3
"""Detached test-sweep driver: runs every scripts/*_test.py sequentially with
a per-suite timeout, records exit codes/durations, and never dies with the
launcher shell. Usage:
    python scripts/_sweep_runner.py
Writes scripts/_sweep_out.log and scripts/_sweep_summary.json in ROOT.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
TIMEOUT_PER_SUITE = 1200  # seconds; heavy training suites need up to ~15 min

LOG = ROOT / "backend" / "scripts" / "_sweep_out.log"
SUMMARY = ROOT / "backend" / "scripts" / "_sweep_summary.json"


def main() -> int:
    suites = sorted(Path(ROOT / "backend" / "scripts").glob("*_test.py"))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    results = []
    with open(LOG, "a", encoding="utf-8") as logf:
        for idx, s in enumerate(suites, 1):
            name = s.stem
            line = f"\n===== {name} ({idx}/{len(suites)}) ====="
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()
            t0 = time.time()
            try:
                r = subprocess.run(
                    [str(VENV_PY), str(s)], cwd=str(ROOT), env=env,
                    stdout=logf, stderr=subprocess.STDOUT, timeout=TIMEOUT_PER_SUITE,
                )
                code, timed_out = r.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = -9, True
                logf.write(f"\n[TIMEOUT after {TIMEOUT_PER_SUITE}s - killed]\n")
            dur = time.time() - t0
            tail = f"exit={code} ({dur:.0f}s{' TIMEOUT' if timed_out else ''})\n"
            print(tail.strip(), flush=True)
            logf.write(tail)
            logf.flush()
            results.append({"suite": name, "exit": code, "seconds": round(dur, 1),
                            "timed_out": timed_out})
    SUMMARY.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(LOG, "a", encoding="utf-8") as logf:
        logf.write("SWEEP_COMPLETE\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())