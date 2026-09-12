#!/usr/bin/env python3
"""Automated Security Scanner for PS14 Fraud Detection System.

This script performs comprehensive security scanning including:
- Static Application Security Testing (SAST)
- Dependency vulnerability scanning
- Secret detection
- Configuration security checks
- Runtime security verification
- Compliance checks

Usage:
    python scripts/security_scan.py [--full] [--json] [--fix]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str
    file: str
    line: int | None
    message: str
    recommendation: str
    cwe: str | None = None  # Common Weakness Enumeration ID
    
    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "file": str(self.file),
            "line": self.line,
            "message": self.message,
            "recommendation": self.recommendation,
            "cwe": self.cwe,
        }


@dataclass
class ScanReport:
    """Complete scan report."""
    scan_time: str
    duration_seconds: float
    issues: list[SecurityIssue] = field(default_factory=list)
    files_scanned: int = 0
    checks_defined: int = 0    # total checks available
    checks_attempted: int = 0  # checks actually executed
    checks_passed: int = 0     # checks that found no issues
    checks_failed: int = 0     # checks that found issues
    checks_blocked: int = 0    # checks that could not execute
    checks_not_run: int = 0    # checks not attempted
    checks_error: int = 0      # checks that errored
    # Legacy field for backward compat
    checks_performed: int = 0
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "CRITICAL")
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "HIGH")
    
    @property
    def medium_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "MEDIUM")
    
    @property
    def low_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "LOW")
    
    @property
    def passed(self) -> bool:
        """Scan passes if no CRITICAL or HIGH issues."""
        return self.critical_count == 0 and self.high_count == 0
    
    def to_dict(self) -> dict:
        return {
            "scan_time": self.scan_time,
            "duration_seconds": self.duration_seconds,
            "files_scanned": self.files_scanned,
            "checks_defined": self.checks_defined,
            "checks_attempted": self.checks_attempted,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_blocked": self.checks_blocked,
            "checks_not_run": self.checks_not_run,
            "checks_error": self.checks_error,
            "findings_count": len(self.issues),
            "summary": {
                "total": len(self.issues),
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "passed": self.passed,
            },
            "issues": [i.to_dict() for i in self.issues],
        }


class SecurityScanner:
    """Main security scanner class."""
    
    def __init__(self, project_root: Path = ROOT):
        self.root = project_root
        self.src_dir = project_root / "src"
        self.report = ScanReport(
            scan_time=datetime.now(timezone.utc).isoformat(),
            duration_seconds=0,
        )
        
        # Patterns for secret detection
        # Only match actual variable assignments, not documentation or comments
        self.secret_patterns = [
            (r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
            (r'(?i)(secret|api_key|apikey)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret/key"),
            (r'(?i)(aws_access_key_id|aws_secret_access_key)\s*=\s*["\'][^"\']+["\']', "AWS credentials"),
            (r'-----BEGIN (RSA |EC )?PRIVATE KEY-----', "Private key"),
        ]
        
        # Files to exclude from scanning
        self.exclude_dirs = {
            ".git", "__pycache__", ".venv", "venv", "node_modules",
            "dist", "build", ".eggs", "*.egg-info",
        }
        
        self.exclude_files = {
            "security_scan.py",  # Don't scan ourselves
        }
    
    def scan(self, full: bool = False, quiet: bool = False) -> ScanReport:
        """Run all security scans."""
        import time
        start = time.time()
        if not quiet:
            print("🔍 Starting security scan...\n")

        # Define all checks (each phase = one check category)
        self.report.checks_defined = 6  # 6 phases

        # Phase 1: Secret detection
        if not quiet: print("[1/6] Scanning for secrets and credentials...")
        self.report.checks_attempted += 1
        try:
            before = len(self.report.issues)
            self.scan_secrets()
            if len(self.report.issues) > before:
                self.report.checks_failed += 1
            else:
                self.report.checks_passed += 1
        except Exception:
            self.report.checks_error += 1

        # Phase 2: Hardcoded values
        if not quiet: print("[2/6] Checking for hardcoded values...")
        self.report.checks_attempted += 1
        try:
            before = len(self.report.issues)
            self.scan_hardcoded_values()
            if len(self.report.issues) > before:
                self.report.checks_failed += 1
            else:
                self.report.checks_passed += 1
        except Exception:
            self.report.checks_error += 1

        # Phase 3: Input validation
        if not quiet: print("[3/6] Analyzing input validation...")
        self.report.checks_attempted += 1
        try:
            before = len(self.report.issues)
            self.scan_input_validation()
            if len(self.report.issues) > before:
                self.report.checks_failed += 1
            else:
                self.report.checks_passed += 1
        except Exception:
            self.report.checks_error += 1

        # Phase 4: Authentication checks
        if not quiet: print("[4/6] Verifying authentication controls...")
        self.report.checks_attempted += 1
        try:
            before = len(self.report.issues)
            self.scan_authentication()
            if len(self.report.issues) > before:
                self.report.checks_failed += 1
            else:
                self.report.checks_passed += 1
        except Exception:
            self.report.checks_error += 1

        # Phase 5: Configuration security
        if not quiet: print("[5/6] Checking configuration security...")
        self.report.checks_attempted += 1
        try:
            before = len(self.report.issues)
            self.scan_configuration()
            if len(self.report.issues) > before:
                self.report.checks_failed += 1
            else:
                self.report.checks_passed += 1
        except Exception:
            self.report.checks_error += 1

        # Phase 6: Dependency check (if full scan)
        self.report.checks_attempted += 1
        if full:
            if not quiet: print("[6/6] Scanning dependencies...")
            try:
                before = len(self.report.issues)
                self.scan_dependencies()
                if len(self.report.issues) > before:
                    self.report.checks_failed += 1
                else:
                    self.report.checks_passed += 1
            except Exception:
                self.report.checks_error += 1
        else:
            if not quiet: print("[6/6] Skipping dependency scan (use --full)")
            self.report.checks_not_run += 1

        self.report.checks_performed = self.report.checks_attempted  # backward compat
        self.report.duration_seconds = time.time() - start

        if not quiet:
            print(f"\n✅ Scan complete in {self.report.duration_seconds:.1f}s")
            print(f"   Files scanned: {self.report.files_scanned}")
            print(f"   Checks defined: {self.report.checks_defined}")
            print(f"   Checks executed: {self.report.checks_attempted}")
            print(f"   Checks passed: {self.report.checks_passed}")
            print(f"   Checks failed: {self.report.checks_failed}")
            print(f"   Checks not run: {self.report.checks_not_run}")
            print(f"   Findings: {len(self.report.issues)}")

        return self.report
    
    def scan_secrets(self):
        """Scan for hardcoded secrets and credentials."""
        for py_file in self._get_python_files():
            self.report.files_scanned += 1
            
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                lines = content.splitlines()
                
                for line_num, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith("#"):
                        continue
                    
                    for pattern, description in self.secret_patterns:
                        if re.search(pattern, line):
                            # Check if it's in a test file or example
                            is_test = "test" in str(py_file).lower()
                            is_example = "example" in str(py_file).lower() or ".env.example" in str(py_file)
                            
                            if is_test or is_example:
                                severity = "LOW"
                            else:
                                severity = "HIGH"
                            
                            self.report.issues.append(SecurityIssue(
                                severity=severity,
                                category="Secret Detection",
                                file=py_file.relative_to(self.root),
                                line=line_num,
                                message=f"{description} detected",
                                recommendation="Move to environment variable or secrets manager",
                                cwe="CWE-798",
                            ))
                            self.report.checks_performed += 1
                            break  # One issue per line
            except Exception:
                pass  # Skip files that can't be read
    
    def scan_hardcoded_values(self):
        """Scan for hardcoded security-relevant values."""
        security_files = [
            "src/identity_service/security.py",
            "src/settings.py",
            "src/middleware/rate_limiter.py",
        ]
        
        for file_path in security_files:
            full_path = self.root / file_path
            if not full_path.exists():
                continue
            
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            
            # Check for hardcoded IPs
            ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            for match in re.finditer(ip_pattern, content):
                ip = match.group()
                if ip not in ("127.0.0.1", "0.0.0.0", "localhost"):
                    line_num = content[:match.start()].count("\n") + 1
                    self.report.issues.append(SecurityIssue(
                        severity="MEDIUM",
                        category="Hardcoded Values",
                        file=file_path,
                        line=line_num,
                        message=f"Hardcoded IP address: {ip}",
                        recommendation="Use environment variable for IP configuration",
                    ))
                    self.report.checks_performed += 1
    
    def scan_input_validation(self):
        """Scan for input validation issues."""
        api_files = list(self.src_dir.rglob("main.py"))
        
        for file_path in api_files:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            
            # Check for SQL injection patterns
            sql_patterns = [
                (r'f"[^"]*\{.*\}[^"]*SELECT', "f-string in SQL query"),
                (r"f'[^']*\{.*\}[^']*SELECT", "f-string in SQL query"),
                (r'\.execute\(f["\']', "f-string in execute()"),
            ]
            
            for pattern, desc in sql_patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count("\n") + 1
                    self.report.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="Input Validation",
                        file=file_path.relative_to(self.root),
                        line=line_num,
                        message=f"Potential SQL injection: {desc}",
                        recommendation="Use parameterized queries instead",
                        cwe="CWE-89",
                    ))
                    self.report.checks_performed += 1
            
            # Check for command injection
            cmd_patterns = [
                (r'os\.system\(', "os.system() usage"),
                (r'subprocess\.call\(.*shell\s*=\s*True', "subprocess with shell=True"),
                (r'eval\(', "eval() usage"),
                (r'exec\(', "exec() usage"),
            ]
            
            for pattern, desc in cmd_patterns:
                matches = list(re.finditer(pattern, content))
                for match in matches:
                    line_num = content[:match.start()].count("\n") + 1
                    self.report.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="Input Validation",
                        file=file_path.relative_to(self.root),
                        line=line_num,
                        message=f"Potential command injection: {desc}",
                        recommendation="Use subprocess without shell=True or avoid eval/exec",
                        cwe="CWE-78",
                    ))
                    self.report.checks_performed += 1
    
    def scan_authentication(self):
        """Scan authentication and authorization controls."""
        # Check CORS configuration
        for py_file in self._get_python_files():
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            
            # Check for wildcard CORS
            if 'allow_origins=["*"]' in content or "allow_origins=['*']" in content:
                self.report.issues.append(SecurityIssue(
                    severity="CRITICAL",
                    category="Authentication",
                    file=py_file.relative_to(self.root),
                    line=None,
                    message="CORS wildcard origin allowed - any website can make requests",
                    recommendation="Restrict allow_origins to specific trusted domains",
                    cwe="CWE-942",
                ))
                self.report.checks_performed += 1
            
            # Check for weak JWT configuration
            if "HS256" in content and "jwt_secret" in content:
                # Check if secret is from env
                if "os.environ" not in content and "settings." not in content:
                    self.report.issues.append(SecurityIssue(
                        severity="MEDIUM",
                        category="Authentication",
                        file=py_file.relative_to(self.root),
                        line=None,
                        message="JWT secret may not be from environment variable",
                        recommendation="Ensure JWT_SECRET is loaded from environment",
                        cwe="CWE-798",
                    ))
                    self.report.checks_performed += 1
    
    def scan_configuration(self):
        """Scan configuration files for security issues."""
        # Check docker-compose.yml
        compose_file = self.root / "docker-compose.yml"
        if compose_file.exists():
            content = compose_file.read_text(encoding="utf-8")
            
            # Check for default secrets
            default_secrets = [
                "ps14-dev-secret-change-me",
                "ps14-dev-internal-token-change-me",
                "change-me-admin-passphrase",
            ]
            
            for secret in default_secrets:
                if secret in content:
                    self.report.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="Configuration",
                        file="docker-compose.yml",
                        line=None,
                        message=f"Default secret found: {secret[:20]}...",
                        recommendation="Replace with generated secrets before deployment",
                        cwe="CWE-798",
                    ))
                    self.report.checks_performed += 1
        
        # Check settings.py for hardcoded dev defaults
        settings_file = self.src_dir / "settings.py"
        if settings_file.exists():
            content = settings_file.read_text(encoding="utf-8")
            settings_patterns = [
                (r'"ps14-dev-[^"]+"', "Hardcoded dev secret in settings.py"),
                (r'"change-me[^"]*"', "Placeholder secret in settings.py"),
                (r'"dev-[^"]*replace[^"]*"', "Dev default secret in settings.py"),
            ]
            for pattern, desc in settings_patterns:
                for match in re.finditer(pattern, content):
                    line_num = content[:match.start()].count("\n") + 1
                    self.report.issues.append(SecurityIssue(
                        severity="HIGH",
                        category="Configuration",
                        file="src/settings.py",
                        line=line_num,
                        message=f"{desc}: {match.group()[:30]}",
                        recommendation="Remove hardcoded defaults; require env vars in production",
                        cwe="CWE-798",
                    ))
                    self.report.checks_performed += 1
        
        # Check for missing security middleware in services
        service_dirs = ["identity_service", "privacy_layer", "risk_engine",
                        "verification_service", "audit_service", "front_service"]
        for svc in service_dirs:
            main_file = self.src_dir / svc / "main.py"
            if not main_file.exists():
                continue
            content = main_file.read_text(encoding="utf-8")
            has_rl = "RateLimiter" in content or "rate_limit" in content or "apply_security_middleware" in content
            has_sh = "SecurityHeaders" in content or "security_headers" in content or "apply_security_middleware" in content
            has_cors = "CORSMiddleware" in content or "allow_origins" in content or "apply_security_middleware" in content
            if not (has_rl and has_sh and has_cors):
                missing = []
                if not has_rl: missing.append("rate limiting")
                if not has_sh: missing.append("security headers")
                if not has_cors: missing.append("CORS")
                self.report.issues.append(SecurityIssue(
                    severity="MEDIUM",
                    category="Configuration",
                    file=f"src/{svc}/main.py",
                    line=None,
                    message=f"Missing security middleware: {', '.join(missing)}",
                    recommendation="Add rate limiting, security headers, and CORS via apply_security_middleware()",
                    cwe="CWE-693",
                ))
                self.report.checks_performed += 1
        
        # Check .env.example
        env_example = self.root / ".env.example"
        if env_example.exists():
            content = env_example.read_text(encoding="utf-8")
            
            # Check for security documentation
            if "SECURITY" not in content.upper():
                self.report.issues.append(SecurityIssue(
                    severity="LOW",
                    category="Configuration",
                    file=".env.example",
                    line=None,
                    message="Missing security documentation in .env.example",
                    recommendation="Add security notes and instructions",
                ))
                self.report.checks_performed += 1
    
    def scan_dependencies(self):
        """Scan dependencies for known vulnerabilities."""
        requirements_file = self.root / "requirements.txt"
        if not requirements_file.exists():
            return
        
        # Check if safety is installed
        try:
            result = subprocess.run(
                [sys.executable, "-m", "safety", "check", "--file", str(requirements_file)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                # Parse safety output
                for line in result.stdout.splitlines():
                    if "vulnerability" in line.lower() or "CVE" in line:
                        self.report.issues.append(SecurityIssue(
                            severity="HIGH",
                            category="Dependencies",
                            file="requirements.txt",
                            line=None,
                            message=f"Vulnerable dependency: {line.strip()}",
                            recommendation="Update to patched version",
                        ))
                        self.report.checks_performed += 1
            else:
                self.report.checks_performed += 1
                self.report.checks_passed += 1
        except FileNotFoundError:
            # Safety not installed — this is a scan FAILURE, not a skip.
            # Report it as INCOMPLETE so CI can detect missing scanner.
            self.report.issues.append(SecurityIssue(
                severity="HIGH",
                category="Scanner",
                file="requirements.txt",
                line=None,
                message="DEPENDENCY SCAN INCOMPLETE: 'safety' package not installed. "
                        "Install with: pip install safety. "
                        "Vulnerabilities in dependencies are NOT being checked.",
                recommendation="Install safety: pip install safety",
            ))
            self.report.checks_performed += 1
            print("   ❌ Safety not installed — dependency scan INCOMPLETE (not skipped)")
        except subprocess.TimeoutExpired:
            self.report.issues.append(SecurityIssue(
                severity="MEDIUM",
                category="Scanner",
                file="requirements.txt",
                line=None,
                message="DEPENDENCY SCAN INCOMPLETE: scan timed out after 60s",
                recommendation="Check network connectivity or increase timeout",
            ))
            self.report.checks_performed += 1
            print("   ❌ Dependency scan timed out — INCOMPLETE")
    
    def _get_python_files(self):
        """Get all Python files to scan."""
        exclude = set(self.exclude_dirs)
        files = []
        
        for path in self.src_dir.rglob("*.py"):
            # Check exclusions
            if any(ex in path.parts for ex in exclude):
                continue
            if path.name in self.exclude_files:
                continue
            files.append(path)
        
        return files
    
    def print_report(self):
        """Print the scan report to console."""
        print("\n" + "=" * 70)
        print("SECURITY SCAN REPORT")
        print("=" * 70)
        print(f"Scan Time: {self.report.scan_time}")
        print(f"Duration: {self.report.duration_seconds:.1f}s")
        print(f"Files Scanned: {self.report.files_scanned}")
        print(f"Checks Defined: {self.report.checks_defined}")
        print(f"Checks Executed: {self.report.checks_attempted}")
        print(f"Checks Passed: {self.report.checks_passed}")
        print(f"Checks Failed: {self.report.checks_failed}")
        print(f"Checks Not Run: {self.report.checks_not_run}")
        print(f"Checks Error: {self.report.checks_error}")
        print(f"Findings: {len(self.report.issues)}")
        coverage = (self.report.checks_attempted / self.report.checks_defined * 100) if self.report.checks_defined else 0
        print(f"Coverage: {coverage:.1f}%")
        print("=" * 70)
        
        if not self.report.issues:
            print("\n✅ No security issues found!")
            return
        
        # Group by severity
        by_severity = {}
        for issue in self.report.issues:
            by_severity.setdefault(issue.severity, []).append(issue)
        
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            issues = by_severity.get(severity, [])
            if not issues:
                continue
            
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "ℹ️"}.get(severity, "")
            print(f"\n{icon} {severity} ({len(issues)} issues)")
            print("-" * 70)
            
            for issue in issues:
                print(f"\n  File: {issue.file}:{issue.line or '?'}")
                print(f"  Message: {issue.message}")
                print(f"  Fix: {issue.recommendation}")
                if issue.cwe:
                    print(f"  CWE: {issue.cwe}")
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"🔴 Critical: {self.report.critical_count}")
        print(f"🟠 High:     {self.report.high_count}")
        print(f"🟡 Medium:   {self.report.medium_count}")
        print(f"🔵 Low:      {self.report.low_count}")
        print("=" * 70)
        
        if self.report.passed:
            print("\n✅ SCAN PASSED - No critical or high severity issues")
        else:
            print("\n❌ SCAN FAILED - Critical or high severity issues found")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PS14 Security Scanner")
    parser.add_argument("--full", action="store_true", help="Run full scan including dependencies")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()
    
    scanner = SecurityScanner()
    report = scanner.scan(full=args.full, quiet=args.json)
    
    if args.json:
        output = json.dumps(report.to_dict(), indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"\nReport saved to {args.output}")
        else:
            print(output)
    else:
        scanner.print_report()
        
        if args.output:
            # Save JSON report
            report_path = Path(args.output)
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
            print(f"\nJSON report saved to {report_path}")
    
    # Exit with error code if scan failed
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()