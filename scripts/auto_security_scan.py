#!/usr/bin/env python3
"""Automated Security Scanner for PS-14.

Runs security_scan.py and penetration_test.py, stores results with timestamps,
and maintains a rolling history (last 90 scans).

Usage:
    python scripts/auto_security_scan.py [--output-dir db/security_scans]
    python scripts/auto_security_scan.py --schedule daily  # Create scheduled task
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "db" / "security_scans"
HISTORY_FILE = "scan_history.json"
MAX_HISTORY = 90  # Keep last 90 scans (3 months of daily scans)


def run_security_scan(output_dir: Path) -> dict:
    """Run the security scanner and return results."""
    script = ROOT / "scripts" / "security_scan.py"
    if not script.exists():
        return {"error": "security_scan.py not found", "exit_code": -1}

    try:
        result = subprocess.run(
            [sys.executable, str(script), "--full", "--json"],
            capture_output=True, text=True, timeout=120,
            cwd=str(ROOT),
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return {
            "error": result.stderr[:500] if result.stderr else "scan failed",
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "security scan timed out (120s)", "exit_code": -1}
    except json.JSONDecodeError as e:
        return {"error": f"invalid JSON output: {e}", "exit_code": -1}
    except Exception as e:
        return {"error": str(e), "exit_code": -1}


def run_penetration_test(output_dir: Path) -> dict:
    """Run the penetration test and return results."""
    script = ROOT / "scripts" / "penetration_test.py"
    if not script.exists():
        return {"error": "penetration_test.py not found", "exit_code": -1}

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=180,
            cwd=str(ROOT),
        )
        # Parse the text output into structured data
        output = result.stdout + result.stderr
        return parse_penetration_output(output, result.returncode)
    except subprocess.TimeoutExpired:
        return {"error": "penetration test timed out (180s)", "exit_code": -1}
    except Exception as e:
        return {"error": str(e), "exit_code": -1}


def parse_penetration_output(output: str, exit_code: int) -> dict:
    """Parse penetration test text output into structured data."""
    lines = output.splitlines()
    results = {
        "total": 0,
        "blocked": 0,
        "vulnerable": 0,
        "tests": [],
        "vulnerabilities": [],
    }

    current_section = ""
    for line in lines:
        line = line.strip()
        if line.startswith("---"):
            current_section = line.strip("- ").strip()
        elif line.startswith("[✓]") or line.startswith("[✗]"):
            passed = line.startswith("[✓]")
            test_name = line[4:].strip()
            # Extract the arrow response
            if "→" in test_name:
                parts = test_name.split("→", 1)
                test_name = parts[0].strip()
                response = parts[1].strip() if len(parts) > 1 else ""
            else:
                response = ""
            results["tests"].append({
                "section": current_section,
                "name": test_name,
                "passed": passed,
                "response": response,
            })
            results["total"] += 1
            if passed:
                results["blocked"] += 1
            else:
                results["vulnerable"] += 1
        elif line.startswith("Total:"):
            # Parse summary line
            parts = line.split("|")
            for part in parts:
                part = part.strip()
                if "Total:" in part:
                    try:
                        results["total"] = int(part.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif "Blocked:" in part:
                    try:
                        results["blocked"] = int(part.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif "Vulnerable:" in part:
                    try:
                        results["vulnerable"] = int(part.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
        elif line.startswith("✗"):
            results["vulnerabilities"].append(line[1:].strip())

    results["exit_code"] = exit_code
    return results


def calculate_score(scan_result: dict, pentest_result: dict) -> dict:
    """Calculate an overall security score (0-100)."""
    score = 100
    issues = []

    # Security scan findings
    if "summary" in scan_result:
        summary = scan_result["summary"]
        score -= summary.get("critical", 0) * 25
        score -= summary.get("high", 0) * 10
        score -= summary.get("medium", 0) * 3
        score -= summary.get("low", 0) * 1
        if summary.get("critical", 0) > 0:
            issues.append(f"{summary['critical']} critical issues")
        if summary.get("high", 0) > 0:
            issues.append(f"{summary['high']} high issues")

    # Penetration test findings
    if "vulnerable" in pentest_result and "total" in pentest_result:
        total = pentest_result["total"]
        vulnerable = pentest_result["vulnerable"]
        if total > 0:
            vuln_rate = vulnerable / total
            score -= int(vuln_rate * 30)  # Up to 30 points for vulns
            if vulnerable > 0:
                issues.append(f"{vulnerable}/{total} penetration vulnerabilities")

    score = max(0, min(100, score))

    # Grade
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": score,
        "grade": grade,
        "issues": issues,
        "assessment": "Excellent" if score >= 90 else "Good" if score >= 80 else "Needs Improvement" if score >= 60 else "Critical",
    }


def store_results(output_dir: Path, scan_result: dict, pentest_result: dict, score: dict) -> Path:
    """Store scan results and update history."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    scan_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Individual scan result
    scan_data = {
        "scan_id": scan_id,
        "timestamp": timestamp,
        "security_scan": scan_result,
        "penetration_test": pentest_result,
        "score": score,
    }

    # Save individual scan
    scan_file = output_dir / f"scan_{scan_id}.json"
    scan_file.write_text(json.dumps(scan_data, indent=2), encoding="utf-8")

    # Update history (rolling window)
    history_file = output_dir / HISTORY_FILE
    history = []
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            history = []

    # Add summary to history (don't store full results in history)
    history.append({
        "scan_id": scan_id,
        "timestamp": timestamp,
        "score": score["score"],
        "grade": score["grade"],
        "security_issues": scan_result.get("summary", {}).get("total", 0),
        "pentest_vulnerable": pentest_result.get("vulnerable", 0),
        "pentest_total": pentest_result.get("total", 0),
    })

    # Keep only last MAX_HISTORY entries
    history = history[-MAX_HISTORY:]
    history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return scan_file


def print_report(scan_result: dict, pentest_result: dict, score: dict, scan_file: Path):
    """Print a human-readable report."""
    print("\n" + "=" * 70)
    print("PS-14 AUTOMATED SECURITY SCAN")
    print("=" * 70)
    print(f"Scan Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Results: {scan_file}")
    print()

    # Security Score
    print(f"{'═' * 70}")
    print(f"  SECURITY SCORE: {score['score']}/100 (Grade: {score['grade']})")
    print(f"  Assessment: {score['assessment']}")
    if score["issues"]:
        print(f"  Issues: {', '.join(score['issues'])}")
    print(f"{'═' * 70}")

    # Security Scan Results
    print("\n📋 Security Scan:")
    if "summary" in scan_result:
        summary = scan_result["summary"]
        print(f"  Files scanned: {scan_result.get('files_scanned', '?')}")
        print(f"  Critical: {summary.get('critical', 0)}")
        print(f"  High: {summary.get('high', 0)}")
        print(f"  Medium: {summary.get('medium', 0)}")
        print(f"  Low: {summary.get('low', 0)}")
        print(f"  Status: {'✅ PASSED' if summary.get('passed') else '❌ FAILED'}")
    elif "error" in scan_result:
        print(f"  Error: {scan_result['error']}")

    # Penetration Test Results
    print("\n🔍 Penetration Test:")
    if "total" in pentest_result:
        print(f"  Total tests: {pentest_result['total']}")
        print(f"  Blocked: {pentest_result['blocked']}")
        print(f"  Vulnerable: {pentest_result['vulnerable']}")
        if pentest_result.get("vulnerabilities"):
            print("  Vulnerabilities:")
            for vuln in pentest_result["vulnerabilities"]:
                print(f"    ✗ {vuln}")
    elif "error" in pentest_result:
        print(f"  Error: {pentest_result['error']}")

    print(f"\n{'═' * 70}")


def create_scheduled_task():
    """Create a Windows Task Scheduler task for daily scans."""
    script = r"C:\Users\Rana Yash Singh\OneDrive\Documents\Project\.venv\Scripts\python.exe"
    args = r"C:\Users\Rana Yash Singh\OneDrive\Documents\Project\scripts\auto_security_scan.py"
    task_name = "PS14-DailySecurityScan"

    # Create the task via PowerShell
    ps_cmd = f"""
$Action = New-ScheduledTaskAction -FilePath '{script}' -Argument '{args}' -WorkingDirectory 'C:\\Users\\Rana Yash Singh\\OneDrive\\Documents\\Project'
$Trigger = New-ScheduledTaskTrigger -Daily -At '2:00AM'
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName '{task_name}' -Action $Action -Trigger $Trigger -Settings $Settings -Description 'PS-14 Daily Security Scan' -Force
"""
    print(f"\nTo create the scheduled task, run this in PowerShell (as Administrator):")
    print(f"{'-' * 70}")
    print(ps_cmd)
    print(f"{'-' * 70}")
    print(f"\nThe task will run daily at 2:00 AM.")


def main():
    parser = argparse.ArgumentParser(description="PS-14 Automated Security Scanner")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to store scan results")
    parser.add_argument("--schedule", choices=["daily"], help="Create scheduled task")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if args.schedule == "daily":
        create_scheduled_task()
        return

    output_dir = Path(args.output_dir)

    if not args.quiet:
        print("🔍 Starting automated security scan...")

    # Run security scan
    if not args.quiet:
        print("\n[1/2] Running security scan...")
    scan_result = run_security_scan(output_dir)

    # Run penetration test
    if not args.quiet:
        print("[2/2] Running penetration test...")
    pentest_result = run_penetration_test(output_dir)

    # Calculate score
    score = calculate_score(scan_result, pentest_result)

    # Store results
    scan_file = store_results(output_dir, scan_result, pentest_result, score)

    if not args.quiet:
        print_report(scan_result, pentest_result, score, scan_file)
    else:
        # Quiet mode: just output JSON
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score": score,
            "scan_file": str(scan_file),
        }
        print(json.dumps(result))

    # Exit with error if critical issues found
    sys.exit(0 if score["score"] >= 70 else 1)


if __name__ == "__main__":
    main()
