#!/usr/bin/env python3
"""Auto-generate security/ML reports from ACTUAL test results.

Every claim in the generated reports comes from real test execution.
No hard-coded pass/fail claims.

Usage:
    python scripts/generate_reports.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
ARTIFACTS = ROOT / "models" / "artifacts"


def run_test(script: str, env_overrides: dict = None, timeout: int = 120) -> tuple[bool, int, str]:
    """Run a test script. Returns (passed, exit_code, full_output)."""
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if env_overrides:
        env.update(env_overrides)
    try:
        result = subprocess.run(
            [PY, str(ROOT / script)],
            capture_output=True, text=True, cwd=str(ROOT),
            env=env, timeout=timeout,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, result.returncode, output
    except subprocess.TimeoutExpired:
        return False, -1, "TIMEOUT"
    except Exception as e:
        return False, -2, str(e)


def count_results(output: str) -> tuple[int, int]:
    """Extract passed/failed counts from test output."""
    import re
    # Look for "RESULTS: X/Y passed, Z failed" pattern
    m = re.search(r"RESULTS:\s*(\d+)/(\d+)\s*passed,\s*(\d+)\s*failed", output)
    if m:
        return int(m.group(1)), int(m.group(3))
    # Look for "X/Y PASSED" pattern
    m = re.search(r"(\d+)/(\d+)\s*PASSED", output)
    if m:
        return int(m.group(1)), 0
    return 0, 0


def generate_security_report(results: dict[str, dict]):
    """Generate SECURITY_REMEDIATION_REPORT.md from actual test outcomes."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r["passed"])

    lines = [
        "# Security Remediation Report",
        "",
        f"**Generated:** {now}",
        f"**Test Results:** {passed_tests}/{total_tests} tests passed",
        "",
        "## Test Suite Results",
        "",
        "| Test Suite | Exit Code | Status | Details |",
        "|------------|-----------|--------|---------|",
    ]

    for name, r in results.items():
        output = r["output"].lower()
        if r["exit_code"] == 0:
            status = "PASS"
        elif r["exit_code"] == -1:
            status = "TIMEOUT"
        elif r["exit_code"] == -2:
            status = "ERROR"
        elif "admin login failed" in output or "service unavailable" in output or "cannot run" in output:
            status = "BLOCKED"
        elif "missing" in output and "required" in output:
            status = "BLOCKED"
        else:
            status = "FAIL"

        passed_count, failed_count = count_results(r["output"])
        if status == "BLOCKED":
            details = "service unavailable or required artifact missing"
        elif passed_count or failed_count:
            details = f"{passed_count} passed, {failed_count} failed"
        else:
            details = f"exit {r['exit_code']}"
        lines.append(f"| {name} | {r['exit_code']} | {status} | {details} |")

    # Security findings — only claim what tests actually verified
    lines.extend([
        "",
        "## Security Findings",
        "",
    ])

    # Check each finding against actual test results
    privacy_passed = results.get("privacy", {}).get("passed", False)
    sql_passed = results.get("sql_injection", {}).get("passed", False)

    lines.extend([
        "| Finding | Severity | Status | Evidence |",
        "|---------|----------|--------|----------|",
        f"| Plaintext address column | HIGH | {'FIXED' if privacy_passed else 'NOT VERIFIED'} | privacy_test: {'address isolation verified' if privacy_passed else 'NOT RUN or FAILED'} |",
        f"| Raw SQL execution | MEDIUM | {'FIXED' if sql_passed else 'NOT VERIFIED'} | sql_injection_test: {'injection tests pass' if sql_passed else 'NOT RUN or FAILED'} |",
        f"| SQL identifier concatenation | MEDIUM | {'FIXED' if sql_passed else 'NOT VERIFIED'} | sql_injection_test: {'fixed TABLE_COUNT_QUERIES' if sql_passed else 'NOT RUN or FAILED'} |",
        f"| Raw amounts in feature store | MEDIUM | {'FIXED' if privacy_passed else 'NOT VERIFIED'} | privacy_test: {'no raw amounts in schema' if privacy_passed else 'NOT RUN or FAILED'} |",
    ])

    lines.extend([
        "",
        "## Acceptance Criteria",
        "",
        f"- [{'x' if privacy_passed else ' '}] No raw transaction amount is persisted in feature/risk storage",
        f"- [{'x' if privacy_passed else ' '}] Address cannot leak outside the identity boundary",
        f"- [{'x' if sql_passed else ' '}] Arbitrary SQL execution is impossible",
        f"- [{'x' if sql_passed else ' '}] SQL identifiers use fixed internal mappings",
        f"- [{'x' if privacy_passed else ' '}] Privacy tests pass from a clean environment",
        f"- [{'x' if sql_passed else ' '}] SQL injection tests pass",
    ])

    report_path = ROOT / "SECURITY_REMEDIATION_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Generated {report_path.name} ({passed_tests}/{total_tests} tests passed)")


def generate_ml_report(results: dict[str, dict]):
    """Generate ML_VALIDATION_REPORT.md from actual test outcomes."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Load metadata
    meta = {}
    meta_path = ARTIFACTS / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    metrics = meta.get("metrics", [])
    split_counts = meta.get("split_counts", {})
    test_prevalence = meta.get("test_prevalence", 0)

    # Check which ML tests actually passed
    calibration_passed = results.get("calibration", {}).get("passed", False)
    adversarial_passed = results.get("adversarial", {}).get("passed", False)
    leakage_passed = results.get("leakage_structural", {}).get("passed", False)
    temporal_passed = results.get("temporal", {}).get("passed", False)

    lines = [
        "# ML Validation Report",
        "",
        f"**Generated:** {now}",
        f"**Seed:** {meta.get('seed', 'N/A')}",
        f"**Training Data:** {meta.get('data', 'N/A')}",
        "",
        "## Test Results",
        "",
        f"- Calibration validation: {'PASS' if calibration_passed else 'FAIL/NOT RUN'}",
        f"- Adversarial testing: {'PASS' if adversarial_passed else 'FAIL/NOT RUN'}",
        f"- Leakage structural: {'PASS' if leakage_passed else 'FAIL/NOT RUN'}",
        f"- Temporal correctness: {'PASS' if temporal_passed else 'FAIL/NOT RUN'}",
        "",
        "## Dataset",
        "",
        f"- **Train:** {split_counts.get('train', 'N/A'):,} samples",
        f"- **Validation:** {split_counts.get('val', 'N/A'):,} samples",
        f"- **Test:** {split_counts.get('test', 'N/A'):,} samples",
        f"- **Test Fraud Rate:** {test_prevalence:.4f}",
        "",
        "## In-Domain Metrics",
        "",
        "| Model | PR-AUC | ROC-AUC | Recall@1%FPR | Precision | F1 |",
        "|-------|--------|---------|--------------|-----------|-----|",
    ]

    for m in metrics:
        name = m.get("model", "unknown")
        lines.append(
            f"| {name} | {m.get('pr_auc', 0):.4f} | {m.get('roc_auc', 0):.4f} | "
            f"{m.get('recall_at_1pct_fpr', 0):.4f} | {m.get('precision', 0):.4f} | "
            f"{m.get('f1', 0):.4f} |"
        )

    lines.extend([
        "",
        "## Calibration",
        "",
        f"- {'[x]' if calibration_passed else '[ ]'} Calibrator validated on real validation data",
        "",
        "## Adversarial Results",
        "",
        f"- {'[x]' if adversarial_passed else '[ ]'} Adversarial cases tested against real model",
        "",
        "## Limitations",
        "",
        "- Model memorizes synthetic archetypes",
        "- Cross-domain generalization limited",
        "- Adversarial cases may score low (honest limitation)",
    ])

    report_path = ROOT / "ML_VALIDATION_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Generated {report_path.name}")


def generate_model_card():
    """Generate MODEL_CARD.md from actual model metadata."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    meta = {}
    meta_path = ARTIFACTS / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    features = meta.get("features", [])
    thresholds = meta.get("thresholds", {})

    lines = [
        "# Model Card",
        "",
        f"**Generated:** {now}",
        f"**Model Version:** seed{meta.get('seed', 'N/A')}-{Path(meta.get('data', 'unknown')).stem}",
        f"**Feature Version:** v1",
        "",
        "## Intended Use",
        "",
        "- Fraud detection for financial transactions",
        "- Risk scoring with calibrated probabilities",
        "- Human-in-the-loop verification for medium/high risk",
        "",
        "## Prohibited Use",
        "",
        "- Autonomous decision-making without human oversight",
        "- PII-based discrimination",
        "",
        "## Training Data",
        "",
        f"- **Source:** {meta.get('data', 'N/A')}",
        f"- **Seed:** {meta.get('seed', 'N/A')}",
        "",
        "## Features ({len(features)})",
        "",
    ]

    for f in features:
        lines.append(f"- `{f}`")

    lines.extend([
        "",
        "## Thresholds",
        "",
        "| Model | F1 Threshold | 1% FPR Threshold |",
        "|-------|--------------|------------------|",
    ])

    for model_name, t in thresholds.items():
        lines.append(
            f"| {model_name} | {t.get('f1', 'N/A'):.4f} | {t.get('1pct_fpr', 'N/A'):.4f} |"
        )

    lines.extend([
        "",
        "## Known Failure Modes",
        "",
        "- OOD fraud archetypes not in training data",
        "- Cross-domain generalization limited",
        "- Subtle multi-feature fraud may evade detection",
    ])

    report_path = ROOT / "MODEL_CARD.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Generated {report_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate reports from actual test results")
    parser.add_argument("--fast", action="store_true", help="Fast mode (no services)")
    args = parser.parse_args()

    print("=" * 60)
    print("PS-14 REPORT GENERATOR (evidence-driven)")
    print("=" * 60)
    print()

    # Run test suite
    print("Running tests...")
    results = {}

    fast_tests = [
        ("privacy", "scripts/privacy_test.py"),
        ("leakage_structural", "scripts/leakage_structural_test.py"),
        ("temporal", "scripts/temporal_test.py"),
        ("calibration", "scripts/calibration_test.py"),
        ("adversarial", "scripts/adversarial_test.py"),
        ("production_gate", "scripts/production_gate_test.py"),
        ("smoke", "scripts/smoke_test.py"),
        ("risk_engine", "scripts/risk_engine_test.py"),
    ]

    for name, script in fast_tests:
        passed, exit_code, output = run_test(script)
        results[name] = {"passed": passed, "exit_code": exit_code, "output": output}
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} (exit {exit_code})")

    # Generate reports
    print()
    print("Generating reports from actual results...")
    generate_security_report(results)
    generate_ml_report(results)
    generate_model_card()

    # Summary
    total = len(results)
    passed = sum(1 for r in results.values() if r["passed"])
    print()
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
