#!/usr/bin/env python3
"""Automated freeze check — docs/RESEARCH_PLAN.md §30.

Verifies, against a repository root:

1. No unresolved bracketed placeholder remains in the plan or the
   preregistration: a marker beginning TO BE FROZEN / OWNER / DATE / HASH /
   SHA / VERSION / TAG / NAME / TIMESTAMP / STATISTICAL / DOMAIN / N, or
   any bracketed token written entirely in capitals (case-insensitive
   marker matching, so lowercase spellings cannot slip through).
2. ``docs/FREEZE_RECORD.json`` exists and every required field is present
   and non-placeholder (§29: git tag, git SHA, freeze timestamp, owner,
   statistical reviewer, reviewer approval timestamp, artifacts).
3. Every artifact listed in the record re-hashes to its recorded SHA-256,
   including this script itself — changing the check after the freeze
   invalidates the freeze.
4. Statistical-reviewer approval fields are present (§30.4).

Exit 0 only when the freeze state is valid; ANY failure exits non-zero.
Pre-freeze (no record / placeholders present) this check FAILS BY DESIGN —
that is the gate, not a bug.

Expected record schema (created at freeze, not before):

    {
      "git_tag": "...", "git_sha": "...",
      "freeze_timestamp": "...", "owner": "...",
      "statistical_reviewer": "...",
      "statistical_reviewer_approval_timestamp": "...",
      "artifacts": [
        {"name": "research_plan", "path": "docs/RESEARCH_PLAN.md", "sha256": "..."},
        {"name": "preregistration", "path": "...", "sha256": "..."},
        {"name": "metric_definitions", "path": "...", "sha256": "..."},
        {"name": "dataset_eligibility_manifest", "path": "...", "sha256": "..."},
        {"name": "exposure_ledger", "path": "...", "sha256": "..."},
        {"name": "freeze_check_script", "path": "scripts/check_freeze.py", "sha256": "..."}
      ]
    }

Usage: python scripts/check_freeze.py [--root DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MARKER_RE = re.compile(
    r"^(?:TO BE FROZEN|OWNER|DATE|HASH|SHA|VERSION|TAG|NAME|"
    r"TIMESTAMP|STATISTICAL|DOMAIN|N)\b",
    re.IGNORECASE,
)
BRACKET_RE = re.compile(r"\[([^[\]]+)\]")

PLAN = "docs/RESEARCH_PLAN.md"
PREREG = "docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md"
RECORD = "docs/FREEZE_RECORD.json"

REQUIRED_FIELDS = (
    "git_tag",
    "git_sha",
    "freeze_timestamp",
    "owner",
    "statistical_reviewer",
    "statistical_reviewer_approval_timestamp",
    "artifacts",
)
REQUIRED_ARTIFACTS = (
    "research_plan",
    "preregistration",
    "metric_definitions",
    "dataset_eligibility_manifest",
    "exposure_ledger",
    "freeze_check_script",
)


def placeholder_tokens(text: str) -> list[tuple[int, str]]:
    """Bracketed placeholders: marker-word prefix or entirely capitals."""
    hits: list[tuple[int, str]] = []
    for m in BRACKET_RE.finditer(text):
        tok = " ".join(m.group(1).split())  # normalise wrapped markers
        if not tok:
            continue  # markdown checkbox "[ ]"
        letters = [c for c in tok if c.isalpha()]
        all_caps = bool(letters) and all(c.isupper() for c in letters)
        if MARKER_RE.match(tok) or all_caps:
            line = text.count("\n", 0, m.start()) + 1
            hits.append((line, tok))
    return hits


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def field_is_placeholder(value: object) -> bool:
    """§30.2: field absent, empty, or carrying a placeholder marker."""
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return True
        if MARKER_RE.match(s):
            return True
        if placeholder_tokens(s):
            return True
    return False


def check(root: Path) -> int:
    failures: list[str] = []

    # ── read the freeze record (or note its absence — expected pre-freeze) ──
    record: object = None
    record_path = root / RECORD
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            failures.append(f"record unreadable: {RECORD} ({exc})")
    else:
        failures.append(f"missing {RECORD} — freeze not recorded (expected pre-freeze)")

    # ── 1. placeholder scan: plan + preregistration (+ record alternates) ──
    scan_paths: list[str] = [PLAN, PREREG]
    if isinstance(record, dict):
        arts = record.get("artifacts")
        if isinstance(arts, list):
            for a in arts:
                if isinstance(a, dict) and a.get("name") in ("research_plan", "preregistration"):
                    p = a.get("path")
                    if isinstance(p, str) and p and p not in scan_paths:
                        scan_paths.append(p)
    for rel in scan_paths:
        p = root / rel
        if not p.is_file():
            failures.append(f"missing document: {rel}")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            failures.append(f"unreadable document: {rel} ({exc})")
            continue
        for line, tok in placeholder_tokens(text):
            failures.append(f"placeholder {rel}:{line}: [{tok}]")

    # ── 2/4. required fields + statistical-reviewer approval ──
    if isinstance(record, dict):
        for f in REQUIRED_FIELDS:
            if f not in record:
                failures.append(f"record field missing: {f}")
            elif field_is_placeholder(record[f]):
                failures.append(f"record field empty/placeholder: {f}")

        # ── 3. recompute SHA-256 for every listed artifact, incl. the script ──
        arts = record.get("artifacts")
        if isinstance(arts, list):
            # every listed artifact re-hashes (§30.3), required ones must exist
            seen: set[str] = set()
            for i, a in enumerate(arts):
                if not isinstance(a, dict):
                    failures.append(f"artifacts[{i}] not an object")
                    continue
                name = a.get("name")
                if not isinstance(name, str) or not name.strip():
                    failures.append(f"artifacts[{i}] missing name")
                    continue
                seen.add(name)
                path, digest = a.get("path"), a.get("sha256")
                if not isinstance(path, str) or not path.strip():
                    failures.append(f"artifact path missing: {name}")
                    continue
                if not isinstance(digest, str) or not digest.strip() or MARKER_RE.match(digest.strip()):
                    failures.append(f"artifact sha256 missing/placeholder: {name}")
                    continue
                fp = root / path
                if not fp.is_file():
                    failures.append(f"artifact file missing: {name} -> {path}")
                    continue
                actual = sha256_file(fp)
                if actual != digest:
                    failures.append(
                        f"artifact hash mismatch: {name} ({path}) "
                        f"recorded {digest} actual {actual}"
                    )
            for name in REQUIRED_ARTIFACTS:
                if name not in seen:
                    failures.append(f"record artifact missing: {name}")

    # ── report ──
    if failures:
        print(f"FREEZE CHECK FAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print(
        "FREEZE CHECK PASSED — record valid, reviewer approval present, "
        "all artifact hashes match, no unresolved placeholders."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-plan freeze check (§30)")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)
    return check(args.root)


if __name__ == "__main__":
    sys.exit(main())
