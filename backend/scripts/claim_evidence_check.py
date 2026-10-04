#!/usr/bin/env python3
"""Phase 105: claim-to-evidence enforcement.

One canonical evidence path:

    execute experiment -> artifact -> evidence record (eval ledger)
            -> claims registry entry -> documentation references the claim id

This check enforces two things:

1. REGISTRY INTEGRITY — every entry in docs/evaluation/claims_registry.jsonl
   resolves against real evidence: a ledger record or a tracked artifact must
   actually contain the metric value, and classification rules must hold
   (DEMONSTRATED requires full provenance; NOT ESTABLISHED/BLOCKED require an
   explicit reason/blocker instead of evidence).

2. DOCUMENT COVERAGE — inside README claim surfaces marked with
   <!-- claims:managed:start --> ... <!-- claims:managed:end --> every
   quantitative table row must carry a <!-- claim:C-NNN --> annotation that is
   registered. A newly introduced quantitative claim inside a managed surface
   without a registry entry FAILS the check.

This is deliberately NOT a bare regex sweep of all prose: formulas, examples,
version numbers, dates, code and historical audit tables outside the managed
surfaces are not scanned (documented limitation — surfaces can grow over time).

Usage:
    python backend/scripts/claim_evidence_check.py            # enforce
    python backend/scripts/claim_evidence_check.py --self-test  # fixture tests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "docs" / "evaluation" / "claims_registry.jsonl"
DEFAULT_LEDGER = ROOT / "reports" / "evaluation_runs" / "eval_ledger.jsonl"
DEFAULT_SURFACES = [
    ROOT / "README.md",
    ROOT / "docs" / "PHASE_105_EVIDENCE_BASELINE.md",
]

CLASSIFICATIONS = {
    "DEMONSTRATED",
    "SELF-TESTED",
    "SIMULATED",
    "NOT ESTABLISHED",
    "BLOCKED",
}

CLAIM_ID_RE = re.compile(r"^C-\d{3,4}$")
# Annotation forms:  <!-- claim:C-001 -->  or  <!-- claims:C-001 C-002 -->
ANNOTATION_RE = re.compile(r"<!--\s*claims?:\s*((?:C-\d{3,4}[\s,]*)+)\s*-->")
ANNOTATION_IDS_RE = re.compile(r"C-\d{3,4}")
# Prefix match so markers may carry an inline explanation after the keyword.
MANAGED_START = "<!-- claims:managed:start"
MANAGED_END = "<!-- claims:managed:end"
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------- helpers

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"REGISTRY_PARSE_ERROR {path}:{i}: {e}")
    return out


def resolve_json_path(obj: Any, dotted: str) -> Any:
    """Resolve a dotted path of dict keys (no list indices). Missing -> KeyError."""
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def extract_numbers(text: str) -> list[tuple[float, bool]]:
    """All numbers in a row, with a percent flag for 'N%' occurrences."""
    out: list[tuple[float, bool]] = []
    for m in NUMBER_RE.finditer(text):
        raw = m.group(0)
        end = m.end()
        is_pct = end < len(text) and text[end] == "%"
        out.append((float(raw), is_pct))
    return out


def _strip_inline(text: str) -> str:
    text = ANNOTATION_RE.sub("", text)
    return re.sub(r"<!--.*?-->", "", text).strip()


def row_is_quantitative(row: str) -> bool:
    """True when at least one cell holds a bare numeric value (metric cell).

    Header cells like 'Recall@1%FPR' or 'Phase 23B' are embedded digits, not
    values, and do not make a row quantitative.
    """
    for cell in row.strip().strip("|").split("|"):
        cell = _strip_inline(cell).strip().strip("*").strip()
        if re.fullmatch(r"-?\d+(\.\d+)?%?", cell):
            return True
    return False


def value_matches_documented(documented: float, tolerance: float, row_text: str) -> bool:
    """True if some number in the documented row agrees with the registry value."""
    for n, is_pct in extract_numbers(row_text):
        if abs(n - documented) <= tolerance:
            return True
        if is_pct and abs(n / 100.0 - documented) <= tolerance:
            return True
    return False


# ------------------------------------------------- evidence verification

def verify_evidence(claim: dict[str, Any], ledger: dict[str, dict[str, Any]],
                    base_dir: Path) -> list[str]:
    """Return a list of problems (empty = evidence OK for its classification)."""
    errs: list[str] = []
    cid = claim.get("claim_id", "?")
    cls = claim.get("classification")
    ev = claim.get("evidence")
    if not isinstance(ev, dict):
        return [f"{cid}: evidence must be an object"]
    etype = ev.get("type")

    if etype == "none":
        if cls != "NOT ESTABLISHED":
            errs.append(f"{cid}: evidence.type=none is only valid for NOT ESTABLISHED (got {cls})")
        if not ev.get("reason"):
            errs.append(f"{cid}: evidence.type=none requires a reason")
        return errs

    if etype == "blocked":
        if cls != "BLOCKED":
            errs.append(f"{cid}: evidence.type=blocked is only valid for BLOCKED (got {cls})")
        if not ev.get("blocker"):
            errs.append(f"{cid}: evidence.type=blocked requires a blocker")
        return errs

    # value-bearing evidence
    documented = claim.get("documented_value")
    tolerance = claim.get("value_tolerance")
    if not isinstance(documented, (int, float)) or not isinstance(tolerance, (int, float)):
        errs.append(f"{cid}: documented_value and value_tolerance are required numbers")
        return errs

    evidence_value: Any = None
    if etype == "ledger":
        rec = ledger.get(ev.get("evaluation_id", ""))
        if rec is None:
            errs.append(f"{cid}: ledger record {ev.get('evaluation_id')} not found")
            return errs
        try:
            evidence_value = resolve_json_path(rec, ev["json_path"])
        except KeyError:
            errs.append(f"{cid}: json_path {ev.get('json_path')} not in ledger record")
            return errs
        # full-provenance requirements for DEMONSTRATED records
        if cls == "DEMONSTRATED":
            if not (rec.get("dataset") or {}).get("sha256"):
                errs.append(f"{cid}: DEMONSTRATED record lacks dataset.sha256")
            if not rec.get("git_commit"):
                errs.append(f"{cid}: DEMONSTRATED record lacks git_commit")
            if rec.get("status") != "COMPLETED":
                errs.append(f"{cid}: DEMONSTRATED record status is {rec.get('status')}")
            if not rec.get("command"):
                errs.append(f"{cid}: DEMONSTRATED record lacks command (run a v1.1 record-generating execution)")
            seed_ok = rec.get("seed") is not None or \
                bool((rec.get("evaluation_config") or {}).get("seed_not_applicable"))
            if not seed_ok:
                errs.append(f"{cid}: DEMONSTRATED record lacks seed (or seed_not_applicable)")

    elif etype == "artifact":
        apath = base_dir / ev.get("path", "")
        if not apath.exists():
            errs.append(f"{cid}: artifact {ev.get('path')} does not exist")
            return errs
        try:
            artifact = json.loads(apath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            errs.append(f"{cid}: artifact {ev.get('path')} unreadable: {e}")
            return errs
        try:
            evidence_value = resolve_json_path(artifact, ev["json_path"])
        except KeyError:
            errs.append(f"{cid}: json_path {ev.get('json_path')} not in artifact {ev.get('path')}")
            return errs
        if cls == "DEMONSTRATED":
            prov = claim.get("provenance") or {}
            for key in ("dataset_sha256", "split", "producer"):
                if not prov.get(key):
                    errs.append(f"{cid}: DEMONSTRATED artifact evidence requires provenance.{key}")
            if prov.get("seed") is None and not prov.get("seed_not_applicable"):
                errs.append(f"{cid}: DEMONSTRATED artifact evidence requires provenance.seed "
                            "(or provenance.seed_not_applicable)")

    elif etype == "suite":
        spath = base_dir / ev.get("path", "")
        if not spath.exists():
            errs.append(f"{cid}: suite log {ev.get('path')} does not exist")
            return errs
        text = spath.read_text(encoding="utf-8", errors="replace")
        if ev.get("expect") not in text:
            errs.append(f"{cid}: suite log does not contain expected marker {ev.get('expect')!r}")
            return errs
        if cls == "DEMONSTRATED":
            errs.append(f"{cid}: suite evidence can support SELF-TESTED at best, not DEMONSTRATED")
            return errs
        # suite evidence carries no numeric value comparison
        return errs

    else:
        errs.append(f"{cid}: unknown evidence type {etype!r}")
        return errs

    if evidence_value is not None and etype in ("ledger", "artifact"):
        try:
            ev_num = float(evidence_value)
        except (TypeError, ValueError):
            errs.append(f"{cid}: evidence value {evidence_value!r} is not numeric")
            return errs
        if abs(ev_num - float(documented)) > float(tolerance):
            errs.append(f"{cid}: documented {documented} != evidence {ev_num} "
                        f"(tolerance {tolerance})")

    # DEMONSTRATED must rest on executed evidence, not absence of it
    if cls == "DEMONSTRATED" and etype not in ("ledger", "artifact"):
        errs.append(f"{cid}: DEMONSTRATED requires ledger or artifact evidence")
    if cls in ("SELF-TESTED", "SIMULATED") and etype not in ("ledger", "artifact", "suite"):
        errs.append(f"{cid}: {cls} requires ledger/artifact/suite evidence")
    return errs


# ------------------------------------------------------- surface scanning

def scan_surfaces(surfaces: list[Path], registry_by_id: dict[str, dict[str, Any]],
                  base_dir: Path) -> list[str]:
    """Every quantitative table row inside a managed region must carry a
    registered claim annotation whose value agrees with the row."""
    errs: list[str] = []
    for sf in surfaces:
        if not sf.exists():
            errs.append(f"surface missing: {sf}")
            continue
        rel = (sf.relative_to(base_dir).as_posix()
               if sf.is_relative_to(base_dir) else str(sf))
        lines = sf.read_text(encoding="utf-8").splitlines()
        in_region = False
        region_start = 0
        for i, line in enumerate(lines, 1):
            if MANAGED_START in line:
                if in_region:
                    errs.append(f"{rel}:{i}: nested managed-region start")
                in_region = True
                region_start = i
                continue
            if MANAGED_END in line:
                if not in_region:
                    errs.append(f"{rel}:{i}: managed-region end without start")
                in_region = False
                continue
            if not in_region:
                continue
            stripped = line.strip()
            if not stripped or not stripped.startswith("|"):
                # non-table content is not claim surface; ignore
                continue
            if not row_is_quantitative(stripped):
                # header / separator rows carry no quantitative claim
                continue
            m = ANNOTATION_RE.search(line)
            if not m:
                errs.append(f"{rel}:{i}: quantitative row inside managed region "
                            f"(region starts line {region_start}) lacks a <!-- claim:C-NNN --> "
                            f"annotation: {stripped[:90]}")
                continue
            for cid in ANNOTATION_IDS_RE.findall(m.group(1)):
                claim = registry_by_id.get(cid)
                if claim is None:
                    errs.append(f"{rel}:{i}: annotation claim:{cid} is not in the registry")
                    continue
                if claim.get("documented_in") != rel:
                    errs.append(f"{cid}: documented_in={claim.get('documented_in')!r} "
                                f"but annotated in {rel}")
                documented = claim.get("documented_value")
                if isinstance(documented, (int, float)) and \
                        not value_matches_documented(float(documented),
                                                     float(claim.get("value_tolerance", 0.0)),
                                                     stripped):
                    errs.append(f"{rel}:{i}: row for {cid} does not contain the registered "
                                f"value {documented} (stale documentation?)")
    if in_region:
        errs.append(f"managed region starting line {region_start} is never closed")
    return errs


# ------------------------------------------------------------ main check

def run_check(registry_path: Path, ledger_path: Path, surfaces: list[Path],
              base_dir: Path) -> list[str]:
    errs: list[str] = []
    claims = load_jsonl(registry_path)
    if not claims:
        errs.append(f"registry empty or missing: {registry_path}")

    ledger_records: list[dict[str, Any]] = []
    if ledger_path.exists():
        for i, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                ledger_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errs.append(f"ledger parse error line {i}: {e}")
    else:
        errs.append(f"ledger missing: {ledger_path}")
    ledger_by_id = {r.get("evaluation_id"): r for r in ledger_records}

    registry_by_id: dict[str, dict[str, Any]] = {}
    for claim in claims:
        cid = claim.get("claim_id")
        if not cid or not CLAIM_ID_RE.match(str(cid)):
            errs.append(f"invalid/missing claim_id: {cid!r}")
            continue
        if cid in registry_by_id:
            errs.append(f"duplicate claim_id: {cid}")
            continue
        registry_by_id[cid] = claim
        cls = claim.get("classification")
        if cls not in CLASSIFICATIONS:
            errs.append(f"{cid}: invalid classification {cls!r}")
            continue
        if not claim.get("claim"):
            errs.append(f"{cid}: missing claim text")
        di = claim.get("documented_in")
        if not di:
            errs.append(f"{cid}: missing documented_in")
        else:
            dpath = base_dir / di
            if not dpath.exists():
                errs.append(f"{cid}: documented_in file missing: {di}")
            else:
                text = dpath.read_text(encoding="utf-8")
                if not re.search(rf"\b{cid}\b", text):
                    errs.append(f"{cid}: {di} contains no <!-- claim:{cid} --> annotation")
        errs.extend(verify_evidence(claim, ledger_by_id, base_dir))

    errs.extend(scan_surfaces(surfaces, registry_by_id, base_dir))
    return errs


# --------------------------------------------------------------- self-test
# Fixtures prove BOTH directions: a valid evidenced claim passes, and
# unregistered/mismatched/unevidenced claims actually FAIL before any
# fixture is removed.

def _write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def self_test() -> int:
    passed = failed = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name} {detail}")

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        led = base / "ledger.jsonl"
        reg = base / "claims.jsonl"
        surface = base / "README.md"
        art = base / "artifact.json"

        # --- fixture evidence: a real execution artifact + ledger record ---
        _write(art, json.dumps({"metrics": {"roc_auc": 0.5123}}))
        rec = {
            "evaluation_id": "eval-fixture-0001",
            "status": "COMPLETED",
            "command": "python scripts/fixture_eval.py",
            "seed": 42,
            "git_commit": "deadbeef" * 5,
            "dataset": {"sha256": "0" * 64},
            "evaluation_config": {},
            "metrics": {"roc_auc": 0.5123},
        }
        _write(led, json.dumps(rec) + "\n")

        def registry(entries: list[dict]) -> None:
            _write(reg, "\n".join(json.dumps(e) for e in entries) + "\n")

        def surface_with(rows: list[str]) -> None:
            _write(surface, "# fixture\n"
                   + MANAGED_START + "\n"
                   + "| Metric | Value |\n|---|---|\n"
                   + "\n".join(rows) + "\n"
                   + MANAGED_END + "\n")

        valid_claim = {
            "claim_id": "C-001",
            "claim": "Fixture ROC-AUC 0.5123",
            "documented_in": "README.md",
            "metric": "roc_auc",
            "documented_value": 0.5123,
            "value_tolerance": 0.0001,
            "classification": "DEMONSTRATED",
            "evidence": {"type": "ledger", "evaluation_id": "eval-fixture-0001",
                         "json_path": "metrics.roc_auc"},
            "notes": "fixture",
        }
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->"])
        registry([valid_claim])
        errs = run_check(reg, led, [surface], base)
        check("valid evidenced claim PASSES", not errs, f"errors={errs}")

        # INVALID 1: newly introduced quantitative row without a claim annotation
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->",
                      "| ROC-AUC | 0.999 |"])
        errs = run_check(reg, led, [surface], base)
        check("invalid: unregistered quantitative row FAILS",
              any("lacks a <!-- claim" in e for e in errs), f"errors={errs}")

        # INVALID 2: annotated row whose value disagrees with the registry
        surface_with(["| Fixture ROC-AUC | 0.999 | <!-- claim:C-001 -->"])
        errs = run_check(reg, led, [surface], base)
        check("invalid: documented value mismatch FAILS",
              any("does not contain the registered value" in e for e in errs),
              f"errors={errs}")

        # INVALID 3: DEMONSTRATED without any evidence
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->"])
        registry([dict(valid_claim,
                       evidence={"type": "none", "reason": "no evidence yet"})])
        errs = run_check(reg, led, [surface], base)
        check("invalid: DEMONSTRATED with no evidence FAILS",
              any("only valid for NOT ESTABLISHED" in e for e in errs), f"errors={errs}")

        # INVALID 4: registered claim pointing at a missing ledger record
        registry([dict(valid_claim,
                       evidence={"type": "ledger", "evaluation_id": "eval-missing",
                                 "json_path": "metrics.roc_auc"})])
        errs = run_check(reg, led, [surface], base)
        check("invalid: missing ledger record FAILS",
              any("not found" in e for e in errs), f"errors={errs}")

        # VALID 2: historical claim explicitly marked NOT ESTABLISHED passes
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->",
                      "| Historical ROC-AUC (not established) | 0.966 | <!-- claim:C-002 -->"])
        registry([
            dict(valid_claim,
                 evidence={"type": "ledger", "evaluation_id": "eval-fixture-0001",
                           "json_path": "metrics.roc_auc"}),
            {"claim_id": "C-002",
             "claim": "Historical headline 0.966 — provenance chain not established",
             "documented_in": "README.md",
             "metric": "roc_auc",
             "documented_value": 0.966,
             "value_tolerance": 0.0005,
             "classification": "NOT ESTABLISHED",
             "evidence": {"type": "none",
                          "reason": "no single artifact contains the historical tuple"},
             "notes": "retained for traceability"},
        ])
        errs = run_check(reg, led, [surface], base)
        check("valid: NOT ESTABLISHED historical claim PASSES", not errs, f"errors={errs}")

        # INVALID 5: artifact evidence with mismatched value
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->"])
        registry([dict(valid_claim,
                       documented_value=0.9,
                       evidence={"type": "artifact", "path": "artifact.json",
                                 "json_path": "metrics.roc_auc"})])
        errs = run_check(reg, led, [surface], base)
        check("invalid: artifact value mismatch FAILS",
              any("!= evidence" in e for e in errs), f"errors={errs}")

        # --------------------------------------------------------------
        # NR-01 matrix extension (Phase: NR-01 evidence-enforcement
        # verification). Every row of the required claim-enforcement test
        # matrix gets an explicit expected-result case here.
        # --------------------------------------------------------------
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->"])

        # VALID: artifact evidence with complete DEMONSTRATED provenance
        registry([dict(valid_claim,
                       evidence={"type": "artifact", "path": "artifact.json",
                                 "json_path": "metrics.roc_auc"},
                       provenance={"dataset_sha256": "0" * 64,
                                   "split": "validation",
                                   "producer": "scripts/fixture_eval.py",
                                   "seed": 42})])
        errs = run_check(reg, led, [surface], base)
        check("valid: artifact evidence with full provenance PASSES",
              not errs, f"errors={errs}")

        # INCOMPLETE PROVENANCE: four DEMONSTRATED ledger requirements
        ledger_recs = [rec]
        for field_name, missing, needle in [
                ("dataset", "dataset.sha256", "lacks dataset.sha256"),
                ("git_commit", "git_commit", "lacks git_commit"),
                ("command", "command", "lacks command"),
                ("seed", "seed", "lacks seed")]:
            broken = dict(rec, evaluation_id=f"eval-fixture-{field_name}")
            if field_name == "dataset":
                broken["dataset"] = {"sha256": None}
            else:
                broken[field_name] = None
            ledger_recs.append(broken)
            _write(led, "\n".join(json.dumps(r) for r in ledger_recs) + "\n")
            registry([dict(valid_claim,
                           evidence={"type": "ledger",
                                     "evaluation_id": broken["evaluation_id"],
                                     "json_path": "metrics.roc_auc"})])
            errs = run_check(reg, led, [surface], base)
            check(f"invalid: DEMONSTRATED missing {missing} FAILS",
                  any(needle in e for e in errs), f"errors={errs}")

        # WRONG PROVENANCE: evidence value belongs to another experiment
        registry([dict(valid_claim, documented_value=0.9,
                       evidence={"type": "ledger",
                                 "evaluation_id": "eval-fixture-0001",
                                 "json_path": "metrics.roc_auc"})])
        errs = run_check(reg, led, [surface], base)
        check("invalid: wrong-provenance value (other experiment) FAILS",
              any("!= evidence" in e for e in errs), f"errors={errs}")

        # WRONG PROVENANCE: json_path does not exist in the record
        registry([dict(valid_claim,
                       evidence={"type": "ledger",
                                 "evaluation_id": "eval-fixture-0001",
                                 "json_path": "metrics.nonexistent"})])
        errs = run_check(reg, led, [surface], base)
        check("invalid: json_path absent from ledger record FAILS",
              any("json_path" in e and "not in ledger" in e for e in errs),
              f"errors={errs}")

        # MALFORMED EVIDENCE: unknown evidence type
        registry([dict(valid_claim,
                       evidence={"type": "clairvoyant", "path": "x"})])
        errs = run_check(reg, led, [surface], base)
        check("invalid: unknown evidence type FAILS",
              any("unknown evidence type" in e for e in errs), f"errors={errs}")

        # EVIDENCE ARTIFACT: missing file
        registry([dict(valid_claim,
                       evidence={"type": "artifact", "path": "absent.json",
                                 "json_path": "metrics.roc_auc"})])
        errs = run_check(reg, led, [surface], base)
        check("invalid: evidence artifact missing on disk FAILS",
              any("does not exist" in e for e in errs), f"errors={errs}")

        # EVIDENCE ARTIFACT: present but invalid schema (json_path absent)
        _write(base / "bad_schema.json", json.dumps({"unexpected": 1}))
        registry([dict(valid_claim,
                       evidence={"type": "artifact", "path": "bad_schema.json",
                                 "json_path": "metrics.roc_auc"})])
        errs = run_check(reg, led, [surface], base)
        check("invalid: evidence artifact with invalid schema FAILS",
              any("not in artifact" in e for e in errs), f"errors={errs}")

        # SUITE evidence: sufficient for SELF-TESTED, forbidden for DEMONSTRATED
        _write(base / "suite.log", "leakage_structural_test: 123/123 PASS\n")
        surface_with(["| Leakage suite checks | 123 | <!-- claim:C-001 -->"])
        registry([dict(valid_claim, documented_value=123, value_tolerance=0,
                       evidence={"type": "suite", "path": "suite.log",
                                 "expect": "123/123"})])
        errs = run_check(reg, led, [surface], base)
        check("invalid: suite evidence cannot support DEMONSTRATED FAILS",
              any("SELF-TESTED at best" in e for e in errs), f"errors={errs}")
        registry([dict(valid_claim, classification="SELF-TESTED",
                       documented_value=123, value_tolerance=0,
                       evidence={"type": "suite", "path": "suite.log",
                                 "expect": "123/123"})])
        errs = run_check(reg, led, [surface], base)
        check("valid: suite evidence with matching marker PASSES as SELF-TESTED",
              not errs, f"errors={errs}")
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->"])

        # BLOCKED classification with an explicit blocker
        registry([dict(valid_claim, classification="BLOCKED",
                       evidence={"type": "blocked",
                                 "blocker": "dataset download approval required"})])
        errs = run_check(reg, led, [surface], base)
        check("valid: BLOCKED claim with explicit blocker PASSES",
              not errs, f"errors={errs}")
        registry([dict(valid_claim, classification="NOT ESTABLISHED",
                       evidence={"type": "blocked", "blocker": "x"})])
        errs = run_check(reg, led, [surface], base)
        check("invalid: NOT ESTABLISHED with type=blocked FAILS",
              any("only valid for BLOCKED" in e for e in errs), f"errors={errs}")

        # MALFORMED REGISTRY: unparseable line aborts with a parse error
        _write(reg, json.dumps(valid_claim) + "\n{not json\n")
        try:
            run_check(reg, led, [surface], base)
            parse_error_raised = False
        except SystemExit as e:
            parse_error_raised = "REGISTRY_PARSE_ERROR" in str(e)
        check("invalid: malformed registry JSON aborts with REGISTRY_PARSE_ERROR",
              parse_error_raised, "no SystemExit raised")
        registry([valid_claim])

        # Registered claim whose documented file lacks the annotation
        _write(base / "README.md",
               "# fixture\n" + MANAGED_START + "\n"
               "| Metric | Value |\n|---|---|\n"
               "| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->\n"
               + MANAGED_END + "\n")
        _write(base / "orphan.md", "# no annotation here\n")
        registry([dict(valid_claim, documented_in="orphan.md")])
        errs = run_check(reg, led, [surface], base)
        check("invalid: registered claim with no annotation in documented_in FAILS",
              any("no <!-- claim" in e and "annotation" in e for e in errs),
              f"errors={errs}")

        # Annotation inside the managed region pointing at an unregistered id
        _write(base / "orphan.md", "x")
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->",
                      "| Rogue row | 0.42 | <!-- claim:C-999 -->"])
        registry([dict(valid_claim, documented_in="README.md")])
        errs = run_check(reg, led, [surface], base)
        check("invalid: annotation referencing unregistered claim id FAILS",
              any("not in the registry" in e for e in errs), f"errors={errs}")

        # Region structure violations
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->"])
        txt = surface.read_text(encoding="utf-8")
        _write(surface, txt.replace(MANAGED_START,
                                    MANAGED_START + "\n" + MANAGED_START))
        errs = run_check(reg, led, [surface], base)
        check("invalid: nested managed region FAILS",
              any("nested" in e for e in errs), f"errors={errs}")
        _write(surface, txt.rsplit(MANAGED_END, 1)[0])
        errs = run_check(reg, led, [surface], base)
        check("invalid: unclosed managed region FAILS",
              any("never closed" in e for e in errs), f"errors={errs}")
        _write(surface, txt)

        # FALSE-POSITIVE probes: legitimate numbers must not be blocked.
        # Metadata/historical/example rows OUTSIDE the managed region are
        # exempt by design (region-free lines are not scanned at all).
        _write(base / "outside.md",
               "| Date | 2026-10-03 |\n"
               "| Schema version | 1.1 |\n"
               "| Dataset rows | 284,807 |\n"
               "| Config limit | 100 |\n"
               "| Example | 0.5 |\n"
               "| Fixture value | 0.5123 |\n"
               "| Historical result | 0.966 |\n")
        errs = run_check(reg, led, [Path(base / "outside.md")], base)
        check("fp: dates/versions/sizes/limits/examples/fixtures/historical "
              "outside a managed region PASS", not errs, f"errors={errs}")
        # Inside the region: headers, separators, prose with numbers and
        # non-bare-numeric cells are not quantitative claims.
        _write(base / "inside.md",
               "# f\n" + MANAGED_START + "\n"
               "| Metric | Value |\n|---|---|\n"
               "Prose with numbers 284807 and seed 42 ignored.\n"
               "| Rows | 1.2M |\n"
               "| Phase 23B | note |\n"
               + MANAGED_END + "\n")
        errs = run_check(reg, led, [Path(base / "inside.md")], base)
        check("fp: headers/separators/prose/non-bare cells inside a region PASS",
              not errs, f"errors={errs}")

        # COMMIT-COUPLING scenario C: a ledger record no registry claim
        # references is allowed by design (evidence may exist without a
        # displayed claim; the converse is enforced).
        _write(led, "\n".join(json.dumps(r) for r in [rec, dict(
            rec, evaluation_id="eval-fixture-unclaimed")]) + "\n")
        registry([dict(valid_claim, documented_in="README.md")])
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->"])
        errs = run_check(reg, led, [surface], base)
        check("scenario C: evidence without a claim PASSES (defined behavior)",
              not errs, f"errors={errs}")

        # COMMIT-COUPLING scenario E: historical value modified in the doc
        # but not in the registry -> FAIL; modified consistently in both ->
        # PASS (checker enforces consistency, not authorship — the registry
        # is a tracked file, so a consistent edit is visible in review).
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->",
                      "| Historical ROC-AUC (not established) | 0.966 | "
                      "<!-- claim:C-002 -->"])
        registry([
            dict(valid_claim, documented_in="README.md"),
            {"claim_id": "C-002",
             "claim": "Historical headline 0.966 — provenance not established",
             "documented_in": "README.md", "metric": "roc_auc",
             "documented_value": 0.966, "value_tolerance": 0.0005,
             "classification": "NOT ESTABLISHED",
             "evidence": {"type": "none", "reason": "no artifact"},
             "notes": "fixture"},
        ])
        surface_with(["| Fixture ROC-AUC | 0.5123 | <!-- claim:C-001 -->",
                      "| Historical ROC-AUC (not established) | 0.900 | "
                      "<!-- claim:C-002 -->"])
        errs = run_check(reg, led, [surface], base)
        check("scenario E: historical value edited without registry FAILS",
              any("does not contain the registered value" in e for e in errs),
              f"errors={errs}")
        registry([
            dict(valid_claim, documented_in="README.md"),
            {"claim_id": "C-002",
             "claim": "Historical headline 0.900 — provenance not established",
             "documented_in": "README.md", "metric": "roc_auc",
             "documented_value": 0.900, "value_tolerance": 0.0005,
             "classification": "NOT ESTABLISHED",
             "evidence": {"type": "none", "reason": "no artifact"},
             "notes": "fixture"},
        ])
        errs = run_check(reg, led, [surface], base)
        check("scenario E: consistent doc+registry edit PASSES "
              "(defined behavior — consistency, not authorship)",
              not errs, f"errors={errs}")

    print(f"\nself-test: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--surface", type=Path, action="append", default=None,
                    help="documentation file with managed claim regions (repeatable)")
    ap.add_argument("--base", type=Path, default=ROOT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    surfaces = args.surface if args.surface else DEFAULT_SURFACES
    errs = run_check(args.registry, args.ledger, surfaces, args.base)
    claims = load_jsonl(args.registry)
    if errs:
        print(f"CLAIM EVIDENCE CHECK: FAIL ({len(errs)} problem(s))")
        for e in errs:
            print(f"  - {e}")
        return 1
    print(f"CLAIM EVIDENCE CHECK: PASS ({len(claims)} claims verified against evidence)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
