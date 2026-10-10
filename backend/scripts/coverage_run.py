#!/usr/bin/env python3
"""Phase 13 — measure backend/src coverage over the fast regression battery.

Reproducible coverage entry point, for local runs and CI. It reuses the exact
FAST_TESTS battery and child environment of regression_suite.py --fast (same
scripts, same cwd=backend, same PYTHONPATH/PS14_MODE), plus the three
hermetic TestClient service suites (verification/audit/federated), and runs
each script under `coverage run --append`, then prints a line+branch report
with missing line numbers for every first-party module under backend/src.

Exit codes:
    0 - every fast test passed AND a non-empty coverage report was produced
    1 - at least one fast test failed, or coverage produced no data/report

Usage (from the repo root, project venv active):
    python backend/scripts/coverage_run.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent  # backend/
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Same battery and env contract as regression_suite.run_test(): cwd=backend,
# PYTHONPATH=backend so child scripts import src.*, UTF-8 stdout, dev mode.
from regression_suite import FAST_TESTS  # noqa: E402

# Hermetic service suites (TestClient, no live stack) that exercise the four
# FastAPI mains beyond the FAST_TESTS battery. Verified green standalone in
# the Phase 13 baseline run.
EXTRA_TESTS = [
    ("verification_flow", "scripts/verification_test.py", {}),
    ("audit_chain", "scripts/audit_test.py", {}),
    ("federated_learning", "scripts/federated_test.py", {}),
]

BATTERY = FAST_TESTS + EXTRA_TESTS

PY = sys.executable


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PS14_MODE"] = "development"
    env["PYTHONPATH"] = str(BACKEND)
    return env


def _coverage(args: list[str], env: dict[str, str]) -> int:
    return subprocess.run(
        [PY, "-m", "coverage", *args],
        cwd=str(BACKEND), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    ).returncode


def main() -> int:
    env = _child_env()

    # Fresh data file (backend/.coverage) — stale data would silently inflate.
    if _coverage(["erase"], env) != 0:
        print("COVERAGE FAILED: `coverage erase` returned non-zero", file=sys.stderr)
        return 1

    results: list[tuple[str, bool, float]] = []
    for name, script, env_overrides in BATTERY:
        child_env = dict(env)
        child_env.update(env_overrides)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [PY, "-m", "coverage", "run", "--append", str(BACKEND / script)],
            cwd=str(BACKEND), env=child_env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        dt = time.perf_counter() - t0
        passed = proc.returncode == 0
        results.append((name, passed, dt))
        print(f"  {name}: {'PASS' if passed else 'FAIL'} ({dt:.1f}s)")

    failed = [n for n, p, _ in results if not p]
    print(f"\n{len(results) - len(failed)}/{len(results)} tests passed")

    # Report: fails visibly (non-zero, no percentage) when data is empty or
    # the measurement path is broken. Uses the canonical backend/.coveragerc.
    report = subprocess.run(
        [PY, "-m", "coverage", "report", "--show-missing"],
        cwd=str(BACKEND), env=env, text=True,
        encoding="utf-8", errors="replace",
    )
    if report.returncode != 0:
        print("COVERAGE FAILED: `coverage report` returned non-zero "
              "(no data collected or broken measurement path)", file=sys.stderr)
        return 1

    # Machine-readable artifact for CI upload (backend/coverage.xml).
    xml = subprocess.run(
        [PY, "-m", "coverage", "xml", "-o", "coverage.xml"],
        cwd=str(BACKEND), env=env,
    )
    if xml.returncode != 0:
        print("COVERAGE FAILED: `coverage xml` returned non-zero", file=sys.stderr)
        return 1

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
