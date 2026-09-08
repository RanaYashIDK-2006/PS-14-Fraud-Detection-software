#!/usr/bin/env python3
"""Documentation claim auditor — searches for unsupported high-risk claims.

Scans documentation for absolute/security-sensitive terms and reports
each occurrence with its evidence status. Does NOT auto-fix; requires
human review.

Run: python scripts/claim_audit.py
Exit: 0 = all claims have evidence, 1 = unsupported claims found
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# High-risk phrases to audit
CLAIM_PATTERNS = [
    # Absolute security claims
    (r"\bproduction[- ]ready\b", "production-ready", "VERIFIED", "System is labeled as research prototype"),
    (r"\bindustry[- ]grade\b", "industry-grade", "UNSUPPORTED", "No independent industry validation"),
    (r"\benterprise[- ]ready\b", "enterprise-ready", "UNSUPPORTED", "Prototype system"),
    (r"\b100%\b.*(?:blocked|secure|safe|pass)", "100% claim", "PARTIALLY_VERIFIED", "Requires live services; pentest shows BLOCKED when services unavailable"),
    (r"\btamper[- ]proof\b", "tamper-proof", "FALSE", "Hash chains provide tamper-EVIDENT not tamper-PREVENTION"),
    (r"\bimmutable\b", "immutable", "FALSE", "SQLite can be modified; triggers prevent app-level deletes only"),
    (r"\battack[- ]proof\b", "attack-proof", "FALSE", "No system is attack-proof"),
    (r"\bcannot be bypassed\b", "cannot be bypassed", "FALSE", "Too absolute; controls can be defeated given sufficient access"),
    (r"\bguaranteed\b", "guaranteed", "UNSUPPORTED", "Security/privacy cannot be guaranteed in absolute terms"),
    (r"\b100%\b.*(?:block|detect|prevent|secure)", "100% security claim", "UNSUPPORTED", "No security system is 100%"),
    (r"\bfully secure\b", "fully secure", "UNSUPPORTED", "No system is fully secure"),
    (r"\battack[- ]proof\b", "attack-proof", "UNSUPPORTED", "No system is attack-proof"),
    # Privacy claims
    (r"\bno raw amount.*(?:anywhere|never|any)", "no raw amounts anywhere", "PARTIALLY_VERIFIED", "Raw amounts exist transiently during processing"),
    (r"\bprivacy[- ]preserving\b", "privacy-preserving", "PARTIALLY_VERIFIED", "Privacy features exist but not fully proven"),
    (r"\bprivacy guaranteed\b", "privacy guaranteed", "UNSUPPORTED", "Privacy cannot be guaranteed absolutely"),
    # ML claims
    (r"\bfail[- ]safe\b", "fail-safe ML", "VERIFIED", "ML_UNAVAILABLE forces score>=31 (step_up)"),
    (r"\bindustry[- ]grade fraud detection\b", "industry-grade fraud detection", "UNSUPPORTED", "Only synthetic/cross-domain evaluation"),
    (r"\bgeneralizes?\b.*(?:fraud|domain|real)", "generalization claim", "PARTIALLY_VERIFIED", "Cross-domain eval exists but limited"),
    # Audit trail
    (r"\btamper[- ]proof\b", "tamper-proof audit", "FALSE", "Hash chain is tamper-EVIDENT not tamper-PROOF"),
    (r"\bimmutable\b", "immutable audit", "FALSE", "SQLite is not immutable"),
    (r"\btamper[- ]evident\b", "tamper-evident audit", "VERIFIED", "Hash chain with append-only triggers"),
    # Test claims
    (r"\ball.*(?:test|check).*(?:pass|block)", "all tests pass", "PARTIALLY_VERIFIED", "Some tests may be BLOCKED"),
    (r"\b\d+/\d+.*pass", "specific test count", "PARTIALLY_VERIFIED", "Count depends on run context"),
]

# Files to scan
DOC_EXTENSIONS = {".md", ".txt"}
SCAN_DIRS = [ROOT / "README.md", ROOT / "docs", ROOT / "reports"]

# Files to exclude
EXCLUDE_PATTERNS = [
    "node_modules", ".git", "__pycache__", ".venv",
    "ps14_security_audit.zip", "ps14_full_audit.zip",
]


def should_exclude(path: Path) -> bool:
    s = str(path)
    return any(ex in s for ex in EXCLUDE_PATTERNS)


def find_doc_files() -> list[Path]:
    files = []
    for item in SCAN_DIRS:
        if item.is_file() and item.suffix in DOC_EXTENSIONS:
            files.append(item)
        elif item.is_dir():
            files.extend(f for f in item.rglob("*") if f.suffix in DOC_EXTENSIONS)
    # Also scan source for docstrings/comments with claims
    for f in (ROOT / "src").rglob("*.py"):
        if not should_exclude(f):
            files.append(f)
    return [f for f in files if not should_exclude(f)]


def scan_file(path: Path) -> list[dict]:
    findings = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings

    lines = text.split("\n")
    for line_no, line in enumerate(lines, 1):
        for pattern, claim_name, status, note in CLAIM_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                findings.append({
                    "file": str(path.relative_to(ROOT)),
                    "line": line_no,
                    "claim": claim_name,
                    "status": status,
                    "note": note,
                    "context": line.strip()[:120],
                })
    return findings


def main():
    files = find_doc_files()
    all_findings = []
    for f in files:
        all_findings.extend(scan_file(f))

    # Group by claim
    by_claim: dict[str, list[dict]] = {}
    for f in all_findings:
        by_claim.setdefault(f["claim"], []).append(f)

    # Print report
    print("=" * 70)
    print("PS-14 CLAIM AUDIT REPORT")
    print("=" * 70)
    print(f"\nScanned {len(files)} files, found {len(all_findings)} occurrences\n")

    # Summary by status
    status_counts: dict[str, int] = {}
    for f in all_findings:
        status_counts[f["status"]] = status_counts.get(f["status"], 0) + 1

    print("STATUS SUMMARY:")
    for status in ["VERIFIED", "PARTIALLY_VERIFIED", "UNSUPPORTED", "FALSE", "NOT_TESTED"]:
        count = status_counts.get(status, 0)
        icon = {"VERIFIED": "✅", "PARTIALLY_VERIFIED": "⚠️", "UNSUPPORTED": "❌", "FALSE": "🚫", "NOT_TESTED": "❓"}.get(status, "?")
        print(f"  {icon} {status}: {count}")
    print()

    # Detailed findings
    unsupported = [f for f in all_findings if f["status"] in ("UNSUPPORTED", "FALSE")]
    partial = [f for f in all_findings if f["status"] == "PARTIALLY_VERIFIED"]
    verified = [f for f in all_findings if f["status"] == "VERIFIED"]

    if verified:
        print("--- VERIFIED CLAIMS ---")
        for f in verified:
            print(f"  ✅ [{f['status']}] {f['claim']}")
            print(f"     {f['file']}:{f['line']} — {f['note']}")
        print()

    if partial:
        print("--- PARTIALLY VERIFIED CLAIMS ---")
        for f in partial:
            print(f"  ⚠️  [{f['status']}] {f['claim']}")
            print(f"     {f['file']}:{f['line']} — {f['note']}")
        print()

    if unsupported:
        print("--- UNSUPPORTED / FALSE CLAIMS ---")
        for f in unsupported:
            icon = "🚫" if f["status"] == "FALSE" else "❌"
            print(f"  {icon} [{f['status']}] {f['claim']}")
            print(f"     {f['file']}:{f['line']} — {f['note']}")
            print(f"     Context: {f['context']}")
        print()

    # Save results
    report_path = ROOT / "reports"
    report_path.mkdir(exist_ok=True)

    results = {
        "total_files_scanned": len(files),
        "total_occurrences": len(all_findings),
        "status_counts": status_counts,
        "findings": all_findings,
    }
    (report_path / "claim_audit_results.json").write_text(json.dumps(results, indent=2))

    n_unsupported = len(unsupported)
    n_false = sum(1 for f in all_findings if f["status"] == "FALSE")

    print("=" * 70)
    if n_unsupported or n_false:
        print(f"RESULT: {n_unsupported} unsupported + {n_false} false claims need attention")
        print("Exit code: 1 (review required)")
    else:
        print("RESULT: All claims have evidence or accurate disclaimers")
        print("Exit code: 0")
    print("=" * 70)

    return 1 if (n_unsupported or n_false) else 0


if __name__ == "__main__":
    sys.exit(main())
