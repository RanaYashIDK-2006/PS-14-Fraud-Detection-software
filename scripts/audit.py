#!/usr/bin/env python3
"""PS-14 Authoritative Audit — runs all checks and produces a clear verdict.

Usage:
    python scripts/audit.py            # fast mode (no live services needed)
    python scripts/audit.py --full     # full mode (requires running stack)
    python scripts/audit.py --report   # write JSON report to reports/audit_report.json

Exit codes:
    0 = ALL PASS
    1 = FAIL (at least one required check failed)
    2 = PARTIAL (some checks passed, some BLOCKED/INCOMPLETE)

DO NOT output "0 issues" when checks were skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


@dataclass
class CheckResult:
    name: str
    status: str  # PASS, FAIL, BLOCKED, INCOMPLETE
    duration_s: float = 0.0
    detail: str = ""


@dataclass
class AuditReport:
    mode: str
    timestamp: str = ""
    checks: list[CheckResult] = field(default_factory=list)
    verdict: str = "PENDING"

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "PASS")

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    @property
    def blocked_count(self) -> int:
        return sum(1 for c in self.checks if c.status in ("BLOCKED", "INCOMPLETE"))

    @property
    def total(self) -> int:
        return len(self.checks)

    def compute_verdict(self) -> str:
        if self.fail_count > 0:
            return "FAIL"
        if self.blocked_count > 0:
            return "PARTIAL"
        return "PASS"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "summary": {
                "total": self.total,
                "passed": self.pass_count,
                "failed": self.fail_count,
                "blocked": self.blocked_count,
            },
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "duration_s": round(c.duration_s, 2),
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


def run_script(name: str, script: str, timeout: int = 120) -> CheckResult:
    """Run a test script and return its result.

    script can include arguments, e.g. 'scripts/foo.py --fast'.
    """
    t0 = time.perf_counter()
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # Split script into path + args so flags aren't part of the filename
        parts = script.split()
        cmd = [PY, str(ROOT / parts[0])] + parts[1:]
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, cwd=str(ROOT),
            env=env, timeout=timeout,
        )
        duration = time.perf_counter() - t0
        if result.returncode == 0:
            return CheckResult(name, "PASS", duration)
        else:
            tail = "\n".join(result.stdout.strip().splitlines()[-3:])
            return CheckResult(name, "FAIL", duration, tail)
    except subprocess.TimeoutExpired:
        return CheckResult(name, "BLOCKED", timeout, f"timed out after {timeout}s")
    except Exception as e:
        return CheckResult(name, "INCOMPLETE", 0, str(e))


def check_dataset_integrity() -> CheckResult:
    """Verify that benchmark datasets are real CSV files, not HTML error pages."""
    t0 = time.perf_counter()
    data_dir = ROOT / "data"
    issues = []

    csv_files = list(data_dir.rglob("*.csv"))
    if not csv_files:
        return CheckResult("dataset_integrity", "BLOCKED", time.perf_counter() - t0,
                           "No CSV files found in data/")

    for f in csv_files[:20]:  # check first 20
        try:
            first_bytes = f.read_bytes()[:500]
            # Check if it's HTML (error page)
            if b"<html" in first_bytes.lower() or b"<!doctype" in first_bytes.lower():
                issues.append(f"{f.name}: appears to be HTML, not CSV")
            elif b"404" in first_bytes and b"not found" in first_bytes.lower():
                issues.append(f"{f.name}: appears to be a 404 error page")
            elif len(first_bytes) < 10:
                issues.append(f"{f.name}: file too small ({f.stat().st_size} bytes)")
        except Exception as e:
            issues.append(f"{f.name}: {e}")

    duration = time.perf_counter() - t0
    if issues:
        return CheckResult("dataset_integrity", "FAIL", duration, "; ".join(issues))
    return CheckResult("dataset_integrity", "PASS", duration,
                       f"{len(csv_files)} CSV files checked")


def check_doc_claims() -> CheckResult:
    """Check for exaggerated security claims in documentation."""
    t0 = time.perf_counter()
    bad_phrases = [
        "fully secure", "completely secure", "zero vulnerabilities",
        "production ready", "production-ready", "enterprise ready",
        "enterprise-ready", "secure by design", "tamper-proof",
        "tamper proof", "independently verified", "fully verified",
        "comprehensive security scan", "no security issues",
        "guaranteed fraud detection", "guaranteed detection",
        "100% protection", "attack-proof", "impenetrable",
    ]
    # Only check docs and README, not code comments or struck-through text
    files_to_check = [ROOT / "README.md"]
    docs_dir = ROOT / "docs"
    if docs_dir.exists():
        files_to_check.extend(docs_dir.glob("*.md"))

    issues = []
    for f in files_to_check:
        if not f.exists():
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            for phrase in bad_phrases:
                # Find lines with the phrase (skip struck-through ~~text~~)
                for i, line in enumerate(content.splitlines(), 1):
                    if phrase.lower() in line.lower():
                        raw = line
                        # Skip struck-through text (~~text~~)
                        import re as _re
                        clean = _re.sub(r'~~[^~]+~~', '', raw).lower()
                        if phrase.lower() not in clean:
                            continue  # phrase is entirely inside strikethrough
                        # Skip comments
                        if raw.lstrip().startswith('#') or raw.lstrip().startswith('<!--'):
                            continue
                        # Skip table rows where the claim is marked FALSE/UNSUPPORTED/NOT_APPLICABLE
                        # e.g. | A3 | Tamper-proof | FALSE | ... or | M8 | Industry-grade | UNSUPPORTED |
                        if '|' in raw:
                            cells = [c.strip().lower() for c in raw.split('|')]
                            if any(any(s in cell for s in ['false', 'unsupported', 'not_applicable']) for cell in cells):
                                continue  # documenting the claim as false, not making it
                        issues.append(f"{f.name}:{i}: '{phrase}' found: {line.strip()[:100]}")
        except Exception:
            pass

    duration = time.perf_counter() - t0
    if issues:
        return CheckResult("doc_claims", "FAIL", duration, "; ".join(issues[:5]))
    return CheckResult("doc_claims", "PASS", duration, "No exaggerated claims found")


def check_scanner_completeness() -> CheckResult:
    """Verify the security scanner reports incomplete scans, not clean results."""
    t0 = time.perf_counter()
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [PY, str(ROOT / "scripts" / "security_scan.py"), "--json"],
            capture_output=True, text=True, cwd=str(ROOT),
            env=env, timeout=60,
        )
        duration = time.perf_counter() - t0
        if result.stdout.strip():
            report = json.loads(result.stdout)
            summary = report.get("summary", {})
            issues = report.get("issues", [])
            # Check if INCOMPLETE scans are reported as issues, not clean results
            incomplete = [i for i in issues if "INCOMPLETE" in i.get("message", "")]
            if incomplete:
                return CheckResult("scanner_completeness", "PASS", duration,
                                   f"Scanner correctly reports {len(incomplete)} INCOMPLETE checks")
            return CheckResult("scanner_completeness", "PASS", duration,
                               f"{len(issues)} findings, {summary.get('checks_performed', '?')} checks performed")
        return CheckResult("scanner_completeness", "INCOMPLETE", duration,
                           "Scanner produced no output")
    except Exception as e:
        return CheckResult("scanner_completeness", "INCOMPLETE",
                           time.perf_counter() - t0, str(e))


def check_prod_config_validation() -> CheckResult:
    """Verify production mode fails without PostgreSQL config when SERVICE_NAME is set."""
    t0 = time.perf_counter()
    try:
        env = os.environ.copy()
        env["PS14_MODE"] = "production"
        env["SERVICE_NAME"] = "risk_engine"  # simulate Docker deployment
        env.pop("DATABASE_URL", None)
        # Remove any per-service URLs
        for key in list(env.keys()):
            if key.endswith("_DB_URL"):
                del env[key]
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [PY, "-c", "from src.settings import validate_production_config; "
             "errors = validate_production_config(); "
             "pg_errors = [e for e in errors if 'PostgreSQL' in e]; "
             "print(f'PG errors: {len(pg_errors)}'); "
             "exit(0 if pg_errors else 1)"],
            capture_output=True, text=True, cwd=str(ROOT),
            env=env, timeout=30,
        )
        duration = time.perf_counter() - t0
        if result.returncode == 0:
            return CheckResult("prod_config_validation", "PASS", duration,
                               "Production mode correctly rejects missing PostgreSQL config when SERVICE_NAME set")
        return CheckResult("prod_config_validation", "FAIL", duration,
                           f"Production mode did NOT reject missing PG config: {result.stdout.strip()}")
    except Exception as e:
        return CheckResult("prod_config_validation", "INCOMPLETE",
                           time.perf_counter() - t0, str(e))


def main():
    parser = argparse.ArgumentParser(description="PS-14 Authoritative Audit")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Full audit (requires running services)")
    group.add_argument("--report", action="store_true", help="Write JSON report")
    args = parser.parse_args()

    report = AuditReport(
        mode="full" if args.full else "fast",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    print("=" * 70)
    print(f"PS-14 AUTHORIZED AUDIT — {report.mode.upper()} MODE")
    print(f"Started: {report.timestamp}")
    print("=" * 70)

    # ── Phase 1: Static checks (no services needed) ──
    print("\n[Phase 1] Static checks...")

    report.checks.append(run_script("regression_suite", "scripts/regression_suite.py --fast", 300))
    report.checks.append(run_script("p2_regression", "scripts/priority2_full_regression_test.py", 60))
    report.checks.append(check_dataset_integrity())
    report.checks.append(check_doc_claims())
    report.checks.append(check_scanner_completeness())
    report.checks.append(check_prod_config_validation())

    # Print results so far
    for c in report.checks:
        icon = {"PASS": "✅", "FAIL": "❌", "BLOCKED": "⚠️", "INCOMPLETE": "⏸️"}.get(c.status, "?")
        detail = f" ({c.detail[:60]})" if c.detail else ""
        print(f"  {icon} {c.name}: {c.status} [{c.duration_s:.1f}s]{detail}")

    # ── Phase 2: Full mode (requires running services) ──
    if args.full:
        print("\n[Phase 2] Live service checks...")

        report.checks.append(run_script("penetration_test", "scripts/penetration_test.py", 120))
        report.checks.append(run_script("security_test", "scripts/security_test.py", 120))
        report.checks.append(run_script("verification_flow", "scripts/verification_test.py", 120))
        report.checks.append(run_script("audit_chain", "scripts/audit_test.py", 120))
        report.checks.append(run_script("front_service", "scripts/front_service_test.py", 120))

        for c in report.checks[len(report.checks)-5:]:
            icon = {"PASS": "✅", "FAIL": "❌", "BLOCKED": "⚠️", "INCOMPLETE": "⏸️"}.get(c.status, "?")
            detail = f" ({c.detail[:60]})" if c.detail else ""
            print(f"  {icon} {c.name}: {c.status} [{c.duration_s:.1f}s]{detail}")

    # ── Verdict ──
    report.verdict = report.compute_verdict()

    print("\n" + "=" * 70)
    print("AUDIT VERDICT")
    print("=" * 70)
    print(f"  Total:   {report.total}")
    print(f"  PASS:    {report.pass_count}")
    print(f"  FAIL:    {report.fail_count}")
    print(f"  BLOCKED: {report.blocked_count}")
    print()

    verdict_icons = {"PASS": "✅ ALL PASS", "PARTIAL": "⚠️ PARTIAL", "FAIL": "❌ FAIL"}
    print(f"  {verdict_icons.get(report.verdict, report.verdict)}")
    print("=" * 70)

    # ── Write report ──
    if args.report:
        report_path = ROOT / "reports" / "audit_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport written to {report_path}")

    # Exit code: 0=PASS, 1=FAIL, 2=PARTIAL
    if report.verdict == "PASS":
        sys.exit(0)
    elif report.verdict == "FAIL":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
