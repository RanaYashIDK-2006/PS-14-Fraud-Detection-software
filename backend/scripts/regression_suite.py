#!/usr/bin/env python3
"""PS-14 Automated Regression Suite.

Runs all critical checks in two modes:
  --fast    : unit + security + privacy separation (no live services needed)
  --full    : all tests including live service checks (requires running stack)

Usage:
    python scripts/regression_suite.py --fast     # CI gate (no services)
    python scripts/regression_suite.py --full     # pre-deploy (all services up)
    python scripts/regression_suite.py            # defaults to --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

FAST_TESTS = [
    # Security / privacy (no services needed)
    ("pseudonym_separation", "scripts/pseudonym_separation_test.py", {}),
    ("production_gate", "scripts/production_gate_test.py", {}),
    # Phase 4A: versioned PII key rotation (hermetic; temp stores + keys)
    ("pii_key_rotation", "scripts/pii_key_rotation_test.py", {}),
    # ML integrity (offline)
    ("smoke_test", "scripts/smoke_test.py", {}),
    ("risk_engine", "scripts/risk_engine_test.py", {}),
    # Phase 7: native production-engine path on a clean checkout — tiny
    # deterministic TEST FIXTURE drives the real AltmanNativeEnsembleEngine
    # loader, all three native members, the Phase 5 batch contract and
    # main.py's native selection (models/production/ stays gitignored).
    ("native_fixture", "scripts/native_fixture_test.py", {}),
    # Phase 8: audit decision-path resilience — DB-4 failure injection,
    # payload-bearing pending store, bounded retry/eviction, recovery,
    # idempotency, FIFO, shutdown, concurrency, decision-path invariant.
    ("audit_resilience", "scripts/audit_resilience_test.py", {}),
    ("drift_monitor", "scripts/drift_test.py", {}),
    ("k_anonymity", "scripts/k_anonymity_test.py", {}),
    ("ood_gate", "scripts/ood_gate_test.py", {}),
    ("rules_gate", "scripts/rules_gate_test.py", {}),
    ("tuner", "scripts/tune_test.py", {}),
    ("pipeline", "scripts/pipeline_test.py", {}),
    ("ml_validity_gate", "scripts/ml_validity_gate_test.py", {}),  # ground-up ML validity
    ("feature_parity", "scripts/feature_parity_test.py", {}),  # offline==online features (#P2)
    ("feedback_loop", "scripts/feedback_test.py", {}),
    ("leakage", "scripts/leakage_test.py", {}),  # label leakage regression
    # P0: Privacy / leakage structural / temporal correctness
    ("privacy", "scripts/privacy_test.py", {}),
    ("leakage_structural", "scripts/leakage_structural_test.py", {}),
    ("temporal", "scripts/temporal_test.py", {}),
    # P1: Calibration + adversarial
    ("calibration", "scripts/calibration_test.py", {}),
    ("adversarial", "scripts/adversarial_test.py", {}),
    # Negative validation
    ("negative_validation", "scripts/negative_test.py", {}),
    # Backup/restore
    ("backup_restore", "scripts/backup_restore_test.py", {}),
    # P2 regression: previously-fixed bugs must stay fixed
    ("p2_regression", "scripts/priority2_full_regression_test.py", {}),
    # Phase 13: hermetic middleware security (413 body limit, 500 mapping,
    # CSP dev/prod, Origin allow-list, rate-limiter window/block/cleanup) —
    # previously only checked against a live service by security_test.py
    ("security_headers", "scripts/security_headers_test.py", {}),
    # Phase 15: the federated institution worker must START and answer the
    # protocol on a CSV that carries EVERY ML_FEATURES column (the
    # fresh-checkout/CI data shape). Regression for F1: a function-local
    # `import sys` inside the missing-column branch made `sys` local for all
    # of main(), so the all-columns path died with UnboundLocalError at
    # startup and federated_test.py failed with `worker t_0 failed to start`.
    ("federated_worker", "scripts/federated_worker_test.py", {}),

    # Phase 16: hermetic inference tests - RedisSlidingWindow (fakeredis),
    # AutoScaler, WorkerInfo, RedisMetricsCollector, WorkerPoolManager status,
    # AB traffic splitter, UnifiedScorer, failure-path tests. No Redis needed.
    ("inference_hermetic", "scripts/inference_hermetic_test.py", {}),
    # Phase 17: hermetic Redis state tests using fakeredis (no real Redis needed)
    ("redis_state_hermetic", "scripts/test_redis_state.py", {}),
]

FULL_TESTS = [
    # Live service tests
    ("security_test", "scripts/security_test.py", {"PYTHONIOENCODING": "utf-8"}),
    ("sql_injection", "scripts/sql_injection_test.py", {}),
    ("verification_flow", "scripts/verification_test.py", {}),
    ("audit_chain", "scripts/audit_test.py", {}),
    ("front_service", "scripts/front_service_test.py", {}),
    ("resilience", "scripts/resilience_test.py", {}),
    ("federated_learning", "scripts/federated_test.py", {}),
]


def run_test(name: str, script: str, env_overrides: dict) -> tuple[bool, float, str]:
    """Run a single test suite. Returns (passed, duration_seconds, output_tail)."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PS14_MODE"] = "development"  # tests use TestClient, not live services
    env["PYTHONPATH"] = str(ROOT)  # child suites import src.*
    env.update(env_overrides)
    t0 = time.perf_counter()
    # Child suites print non-ASCII (✓/—) and write UTF-8; on Windows the
    # parent's locale is cp1252, which would crash the reader thread, so
    # decode explicitly as UTF-8 with replacement.
    result = subprocess.run(
        [PY, str(ROOT / script)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=env, timeout=120,
    )
    duration = time.perf_counter() - t0
    output = result.stdout + result.stderr
    # Authoritative: exit code 0 = PASS, non-zero = FAIL
    # Output strings are informational only.
    passed = result.returncode == 0
    lines = [line for line in output.strip().split("\n") if line.strip()]
    return passed, duration, "\n".join(lines[-3:]), lines


def main():
    parser = argparse.ArgumentParser(description="PS-14 Regression Suite")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true", help="Fast suite (no services needed)")
    group.add_argument("--full", action="store_true", help="Full suite (all services up)")
    args = parser.parse_args()

    if args.full:
        tests = FAST_TESTS + FULL_TESTS
        mode = "FULL"
    else:
        tests = FAST_TESTS
        mode = "FAST"

    print(f"{'='*60}")
    print(f"PS-14 REGRESSION SUITE — {mode} MODE")
    print(f"{'='*60}")
    print()

    results = []
    for name, script, env in tests:
        print(f"  Running {name}...", end=" ", flush=True)
        try:
            passed, duration, tail, lines = run_test(name, script, env)
            status = "PASS" if passed else "FAIL"
            print(f"{status} ({duration:.1f}s)")
            results.append((name, passed, duration, tail))
            if not passed:
                print(f"    Last lines: {tail}")
                # Show more of a failed suite's output so the specific
                # failing check is identifiable from the runner log.
                print("    --- failure detail ---")
                for ln in lines[-25:]:
                    print(f"      {ln}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT (120s)")
            results.append((name, False, 120.0, "TIMEOUT"))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, False, 0.0, str(e)))

    # Summary
    print(f"\n{'='*60}")
    passed_count = sum(1 for _, p, _, _ in results if p)
    total = len(results)
    total_time = sum(d for _, _, d, _ in results)

    for name, passed, duration, _ in results:
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name} ({duration:.1f}s)")

    print(f"\n{passed_count}/{total} PASSED ({total_time:.1f}s total)")

    if passed_count == total:
        print(f"\n{'='*60}")
        print(f"ALL {total} CHECKS PASSED — {mode} SUITE COMPLETE")
        print(f"{'='*60}")
    else:
        failed = [name for name, p, _, _ in results if not p]
        print(f"\nFAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
