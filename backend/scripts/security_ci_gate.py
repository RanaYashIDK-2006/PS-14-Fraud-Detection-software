"""CI Security Gate — runs security tests and fails if any finding exists.

This script runs:
1. security_scan.py — code-level security scan
2. penetration_test.py — 34 attack vectors
3. risk_engine_test.py — core regression tests

Exit code:
  0 = all clear (0 security findings, all tests pass)
  1 = security findings OR test failures

Unlike the individual scripts which print summaries, this script:
- Surfaces RAW findings (not just "0 issues")
- Fails hard on ANY security issue
- Can be used as a CI gate (blocks merge on failure)
"""

import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
PYTHON = sys.executable

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def run_test(name: str, script: str, args: list[str] | None = None) -> tuple[bool, str]:
    """Run a test script and return (passed, output)."""
    cmd = [PYTHON, str(ROOT / "backend" / "scripts" / script)] + (args or [])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(ROOT), encoding="utf-8", errors="replace"
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0
        return passed, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after 120s"
    except Exception as e:
        return False, f"ERROR: {e}"


def main():
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}PS-14 SECURITY CI GATE{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print()
    print("This runs security tests and FAILS if any finding exists.")
    print("Raw findings are surfaced — not just summary counts.")
    print()

    findings = []
    all_passed = True

    # 1. Security scan
    print(f"{BOLD}[1/3] Security Code Scan{RESET}")
    passed, output = run_test("security_scan", "security_scan.py")
    if passed:
        print(f"  {GREEN}✓ PASSED{RESET}")
        # Check for any findings in output
        if "No security issues found" in output:
            print(f"    0 findings")
        else:
            # Extract findings
            for line in output.split("\n"):
                if "HIGH" in line or "MEDIUM" in line or "LOW" in line:
                    findings.append(("SECURITY_SCAN", line.strip()))
                    print(f"    {RED}FINDING: {line.strip()}{RESET}")
    else:
        print(f"  {RED}✗ FAILED{RESET}")
        all_passed = False
        findings.append(("SECURITY_SCAN", "Script failed"))
    print()

    # 2. Penetration test
    print(f"{BOLD}[2/3] Penetration Test (34 vectors){RESET}")
    passed, output = run_test("penetration", "penetration_test.py")
    if passed:
        print(f"  {GREEN}✓ PASSED{RESET}")
        # Parse the summary line
        for line in output.split("\n"):
            if "Blocked:" in line and "Vulnerable:" in line:
                print(f"    {line.strip()}")
    else:
        print(f"  {RED}✗ FAILED{RESET}")
        all_passed = False
        # Extract vulnerable vectors
        in_vulns = False
        for line in output.split("\n"):
            if "VULNERABILITIES:" in line:
                in_vulns = True
                continue
            if in_vulns and line.strip().startswith("✗"):
                findings.append(("PENTEST", line.strip()))
                print(f"    {RED}VULNERABLE: {line.strip()}{RESET}")
            if in_vulns and not line.strip().startswith("✗") and line.strip():
                in_vulns = False
    print()

    # 3. Core regression tests
    print(f"{BOLD}[3/3] Core Regression Tests{RESET}")
    passed, output = run_test("risk_engine", "risk_engine_test.py")
    if passed:
        print(f"  {GREEN}✓ risk_engine_test.py PASSED{RESET}")
    else:
        print(f"  {RED}✗ risk_engine_test.py FAILED{RESET}")
        all_passed = False
        findings.append(("REGRESSION", "risk_engine_test.py failed"))

    passed2, output2 = run_test("smoke", "smoke_test.py")
    if passed2:
        print(f"  {GREEN}✓ smoke_test.py PASSED{RESET}")
    else:
        print(f"  {RED}✗ smoke_test.py FAILED{RESET}")
        all_passed = False
        findings.append(("REGRESSION", "smoke_test.py failed"))
    print()

    # Summary
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print()

    if all_passed and not findings:
        print(f"  {GREEN}{BOLD}✅ ALL CLEAR — 0 security findings, all tests pass{RESET}")
        print()
        print(f"  NOTE: These are self-authored test suites, not a third-party audit.")
        print(f"  A professional penetration test is recommended before handling")
        print(f"  real financial data.")
        print()
        sys.exit(0)
    else:
        print(f"  {RED}{BOLD}❌ BLOCKED — {len(findings)} finding(s) detected{RESET}")
        print()
        print(f"  Raw findings:")
        for category, detail in findings:
            print(f"    [{category}] {detail}")
        print()
        print(f"  Fix all findings before merging.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
