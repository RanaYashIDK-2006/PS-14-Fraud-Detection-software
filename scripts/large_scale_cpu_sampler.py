"""Phase 3 §6 — sample CPU utilisation and RSS while the runner executes.

The runner itself reports no CPU figure, so this launches it as a subprocess
and samples its process tree with psutil. Recorded numbers are MEASURED, not
assumed, and the sample covers only the run it wraps.

Usage:
    ./.venv/Scripts/python.exe scripts/large_scale_cpu_sampler.py \
        --rows 5000000 --out misc/reports/phase3_cpu.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

PY = str(Path(".venv/Scripts/python.exe"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument("--out", default="misc/reports/phase3_cpu.json")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    out_json = Path("misc/reports/phase3_cpu_run.json")
    p = subprocess.Popen([PY, "scripts/large_scale_runner.py", "--rows", str(args.rows),
                          "--out", str(out_json), "--label", f"cpu-sample {args.rows:,}"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    proc = psutil.Process(p.pid)
    # Process objects MUST be kept alive across samples: psutil's
    # cpu_percent(interval=None) caches its baseline on the instance, so a
    # fresh object every iteration would return a meaningless 0.0 each time.
    procs: dict[int, psutil.Process] = {p.pid: proc}
    cpu, rss, ncore = [], [], psutil.cpu_count()
    t0 = time.time()
    while p.poll() is None:
        try:
            # sum the process tree (the venv python.exe is a redirector; the
            # real worker is its child)
            for pr in proc.children(recursive=True):
                procs.setdefault(pr.pid, pr)
            c = 0.0
            r = 0
            dead: list[int] = []
            for pid, pr in procs.items():
                try:
                    if not pr.is_running():
                        dead.append(pid)
                        continue
                    c += pr.cpu_percent(interval=None)
                    r += pr.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    dead.append(pid)
            for pid in dead:
                procs.pop(pid, None)
            cpu.append(c)
            rss.append(r / 2**20)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(args.interval)
    wall = time.time() - t0
    stdout = p.stdout.read() if p.stdout else ""

    run = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    res = {
        "label": f"CPU-utilisation sample over a {args.rows:,}-row runner pass",
        "evidence_class": "SELF-TESTED — SYNTHETIC 50M — SCALE EXPERIMENT",
        "rows": args.rows,
        "wall_seconds_sampled": wall,
        "logical_cores": ncore,
        "samples": len(cpu),
        "sample_interval_s": args.interval,
        "cpu_percent_mean": (sum(cpu) / len(cpu)) if cpu else None,
        "cpu_percent_p50": sorted(cpu)[len(cpu) // 2] if cpu else None,
        "cpu_percent_p95": sorted(cpu)[int(len(cpu) * 0.95)] if cpu else None,
        "cpu_percent_max": max(cpu) if cpu else None,
        "cpu_cores_busy_mean": (sum(cpu) / len(cpu) / ncore) if cpu and ncore else None,
        "rss_MB_mean": (sum(rss) / len(rss)) if rss else None,
        "rss_MB_max": max(rss) if rss else None,
        "runner_exit_code": p.returncode,
        "runner": {k: run.get(k) for k in
                   ("rows_processed", "wall_seconds", "rows_per_second", "roc_auc",
                    "latency_ms_per_batch", "stage_seconds")},
        "runner_stdout_tail": (stdout or "").strip().splitlines()[-4:],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "runner"}, indent=2, default=str))
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())