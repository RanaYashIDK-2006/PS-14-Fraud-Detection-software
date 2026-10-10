#!/usr/bin/env python3
"""Phase 17 - Real Redis integration test runner.

Runs the real-Redis integration tests (requires Redis on 127.0.0.1:6379).
This is separate from the fast battery (which is hermetic) and the full
regression suite (which tests live services).

Usage:
    python backend/scripts/redis_integration_runner.py

Exit codes:
    0 - Redis preflight passed AND all integration tests passed
    0 - Redis preflight failed (tests skipped, not a failure)
    1 - Redis preflight passed BUT integration tests failed
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

INTEGRATION_TESTS = [
    ("redis_state_integration", "scripts/redis_state_integration_test.py"),
    # Note: test_worker_pool.py also uses real Redis but is not included here
    # because it tests worker pool management, not RedisStateManager specifically.
    # It can be run separately when worker testing is needed.
]


def preflight() -> bool:
    """Check if Redis is available on 127.0.0.1:6379."""
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0,
                        socket_connect_timeout=3, protocol=2)
        r.ping()
        return True
    except Exception as e:
        print(f"Redis preflight failed: {e}")
        return False


def run_test(name: str, script: str) -> tuple[bool, float, str]:
    """Run a single integration test. Returns (passed, duration, output_tail)."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PS14_MODE"] = "development"
    env["PYTHONPATH"] = str(ROOT)

    t0 = time.perf_counter()
    result = subprocess.run(
        [PY, str(ROOT / script)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=env, timeout=60,
    )
    duration = time.perf_counter() - t0
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    lines = [line for line in output.strip().split("\n") if line.strip()]
    return passed, duration, "\n".join(lines[-5:]), lines


def main() -> int:
    print("=" * 60)
    print("Phase 17 - Real Redis Integration Tests")
    print("=" * 60)
    print()

    # Preflight check
    print("Redis: 127.0.0.1:6379")
    if not preflight():
        print()
        print("Redis preflight FAILED - skipping all integration tests")
        print()
        print("=" * 60)
        print("RESULT: SKIPPED (Redis unavailable)")
        print("=" * 60)
        print()
        print("Note: Integration tests require a running Redis server.")
        print("Start Redis with: redis-server")
        print()
        return 0  # Not a failure - just skipped

    print("Redis preflight PASSED")
    print()

    results = []
    for name, script in INTEGRATION_TESTS:
        print(f"  Running {name}... ", end="", flush=True)
        try:
            passed, duration, tail, lines = run_test(name, script)
            status = "PASS" if passed else "FAIL"
            print(f"{status} ({duration:.1f}s)")
            results.append((name, passed, duration, tail))
            if not passed:
                print(f"    --- failure detail ---")
                for ln in lines[-20:]:
                    print(f"      {ln}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT (60s)")
            results.append((name, False, 60.0, "TIMEOUT"))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, False, 0.0, str(e)))
        print()

    # Summary
    print("=" * 60)
    passed_count = sum(1 for _, p, _, _ in results if p)
    total = len(results)
    total_time = sum(d for _, _, d, _ in results)

    for name, passed, duration, _ in results:
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name} ({duration:.1f}s)")

    print(f"\n{passed_count}/{total} PASSED ({total_time:.1f}s total)")

    if passed_count == total:
        print()
        print("=" * 60)
        print(f"ALL {total} INTEGRATION TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        failed = [name for name, p, _, _ in results if not p]
        print()
        print(f"FAILED: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
