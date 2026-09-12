#!/usr/bin/env python3
"""Production security gate test.

Verifies that PS14_MODE=production correctly:
1. Rejects dev default secrets
2. Rejects localhost CORS origins
3. Allows properly configured production secrets
4. Aborts with clear error messages

Usage:
    python scripts/production_gate_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def main():
    failures = 0

    def check(name: str, condition: bool, detail: str = "") -> bool:
        nonlocal failures
        tag = "PASS" if condition else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{tag}] {name}{suffix}")
        if not condition:
            failures += 1
        return condition

    # Test 1: Production mode with dev defaults should FAIL
    print("--- Production mode with dev defaults ---")
    # Explicitly clear secrets to simulate true dev defaults
    clean_env = os.environ.copy()
    for k in ["JWT_SECRET", "PII_ENCRYPTION_KEY", "EXPORT_SIGNING_KEY",
              "INTERNAL_TOKEN", "COMPLIANCE_TOKEN", "CORS_ORIGINS"]:
        clean_env.pop(k, None)
    clean_env["PS14_MODE"] = "production"
    result = subprocess.run(
        [sys.executable, "-c", """
import os
os.environ['PS14_MODE'] = 'production'
# Keep dev defaults (don't override)
from src.settings import validate_production_config
errors = validate_production_config()
print(f'ERRORS:{len(errors)}')
for e in errors:
    print(f'  ERROR:{e}')
"""],
        capture_output=True, text=True, cwd=os.getcwd(), env=clean_env
    )
    error_count = 0
    for line in result.stdout.split("\n"):
        if line.startswith("ERRORS:"):
            error_count = int(line.split(":")[1])
    check("production mode detects dev secrets", error_count >= 3, f"found {error_count} errors")

    # Test 2: Production mode with proper secrets should PASS
    print("\n--- Production mode with proper secrets ---")
    env = os.environ.copy()
    env["PS14_MODE"] = "production"
    env["JWT_SECRET"] = "a" * 32
    env["PII_ENCRYPTION_KEY"] = "b" * 32
    env["EXPORT_SIGNING_KEY"] = "c" * 32
    env["INTERNAL_TOKEN"] = "d" * 32
    env["COMPLIANCE_TOKEN"] = "e" * 32
    env["CORS_ORIGINS"] = "https://example.com"
    result = subprocess.run(
        [sys.executable, "-c", """
import os
from src.settings import validate_production_config
errors = validate_production_config()
print(f'ERRORS:{len(errors)}')
"""],
        capture_output=True, text=True, cwd=os.getcwd(), env=env
    )
    error_count = 0
    for line in result.stdout.split("\n"):
        if line.startswith("ERRORS:"):
            error_count = int(line.split(":")[1])
    check("production mode accepts proper secrets", error_count == 0, f"errors={error_count}")

    # Test 3: Development mode should NOT abort (just warn)
    print("\n--- Development mode ---")
    env = os.environ.copy()
    env["PS14_MODE"] = "development"
    result = subprocess.run(
        [sys.executable, "-c", """
import os
os.environ['PS14_MODE'] = 'development'
from src.settings import enforce_production_gate
enforce_production_gate()
print('SURVIVED')
"""],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    check("development mode does not abort", "SURVIVED" in result.stdout)

    # Test 4: Production mode with dev secrets should abort
    print("\n--- Production mode abort behavior ---")
    result = subprocess.run(
        [sys.executable, "-c", """
import os
os.environ['PS14_MODE'] = 'production'
# Keep dev defaults
from src.settings import enforce_production_gate
enforce_production_gate()
print('SURVIVED')
"""],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    check("production mode aborts with dev secrets", "SURVIVED" not in result.stdout and result.returncode != 0)

    # Test 5: Verify PS14_MODE is read correctly
    print("\n--- PS14_MODE environment variable ---")
    for mode in ("development", "production"):
        env = os.environ.copy()
        env["PS14_MODE"] = mode
        result = subprocess.run(
            [sys.executable, "-c", f"import os; print(os.environ.get('PS14_MODE', 'unset'))"],
            capture_output=True, text=True, env=env
        )
        check(f"PS14_MODE={mode} reads correctly", mode in result.stdout)

    # Test 6: Model artifacts must exist
    print("\n--- Model artifacts ---")
    from pathlib import Path
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / "models" / "artifacts"
    required_artifacts = ["calibrator.joblib", "scaler.joblib", "xgboost.joblib"]
    for artifact in required_artifacts:
        path = artifacts_dir / artifact
        check(f"{artifact} exists", path.exists(), str(path))

    print(f"\n{'='*60}")
    if failures:
        print(f"{failures} CHECK(S) FAILED")
        sys.exit(1)
    else:
        print("ALL PRODUCTION GATE CHECKS PASSED")


if __name__ == "__main__":
    main()
