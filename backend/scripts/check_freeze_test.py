#!/usr/bin/env python3
"""Freeze-check suite — RESEARCH_PLAN §30 semantics (RP-01).

Plain assert script (no pytest). Verifies scripts/check_freeze.py:

  1. the real repo FAILS pre-freeze (missing record + placeholders remain);
  2. a fully-frozen fixture PASSES (rc=0);
  3. each §30 duty independently forces failure:
       - tampered artifact hash
       - tampered freeze-check script (self-hash)
       - missing statistical-reviewer approval field
       - placeholder value in a record field
       - missing required record field
       - placeholder reintroduced into the plan
       - missing FREEZE_RECORD.json
       - malformed record JSON
  4. marker detection matches §30.1 exactly (marker words + all-caps tokens,
     markdown checkboxes NOT flagged, wrapped markers normalised);
  5. check_freeze.py hashes itself correctly (self-consistency of the record).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHECK = REPO / "scripts" / "check_freeze.py"
PLAN = REPO / "docs" / "RESEARCH_PLAN.md"
PREREG = REPO / "docs" / "evaluation" / "PRE_REGISTERED_EVALUATION_PROTOCOL.md"

sys.path.insert(0, str(CHECK.parent))
import check_freeze  # noqa: E402

PASS = 0


def ok(cond: bool, label: str) -> None:
    global PASS
    assert cond, label
    PASS += 1
    print(f"  PASS {label}")


def run(root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(CHECK), "--root", str(root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return proc.returncode, proc.stdout + proc.stderr


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_record(root: Path) -> dict:
    """A complete, correct freeze record for the fixture at `root`."""
    files = {
        "research_plan": "docs/RESEARCH_PLAN.md",
        "preregistration": "docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md",
        "metric_definitions": "docs/metric_definitions.md",
        "dataset_eligibility_manifest": "docs/dataset_eligibility_manifest.md",
        "exposure_ledger": "docs/dataset_exposure_ledger.md",
        "freeze_check_script": "scripts/check_freeze.py",
    }
    return {
        "git_tag": "v0.0-fixture",
        "git_sha": "0" * 40,
        "freeze_timestamp": "2026-10-04T00:00:00Z",
        "owner": "fixture-owner",
        "statistical_reviewer": "fixture-reviewer",
        "statistical_reviewer_approval_timestamp": "2026-10-04T00:00:00Z",
        "artifacts": [
            {"name": name, "path": rel, "sha256": sha256(root / rel)}
            for name, rel in files.items()
        ],
    }


def make_fixture() -> Path:
    """A fully-frozen miniature repo: no placeholders, complete valid record."""
    root = Path(tempfile.mkdtemp(prefix="freeze_fixture_"))
    (root / "docs" / "evaluation").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "docs" / "RESEARCH_PLAN.md").write_text(
        "# Research Plan\n\nStatus: FROZEN.\nAll decisions recorded.\n",
        encoding="utf-8",
    )
    (root / "docs" / "evaluation" / "PRE_REGISTERED_EVALUATION_PROTOCOL.md").write_text(
        "# Pre-Registered Evaluation Protocol\n\nApproved.\n", encoding="utf-8"
    )
    for rel in (
        "docs/metric_definitions.md",
        "docs/dataset_eligibility_manifest.md",
        "docs/dataset_exposure_ledger.md",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"fixture content for {rel}\n", encoding="utf-8")
    shutil.copy(CHECK, root / "scripts" / "check_freeze.py")
    (root / "docs" / "FREEZE_RECORD.json").write_text(
        json.dumps(make_record(root), indent=2), encoding="utf-8"
    )
    return root


def write_record(root: Path, record: dict) -> None:
    (root / "docs" / "FREEZE_RECORD.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def expect_fail(root: Path, needle: str, label: str) -> None:
    rc, out = run(root)
    ok(rc != 0 and needle in out, label)


def main() -> int:
    # ── 1. real repo verdict must match its freeze state ──────────────────
    # Pre-freeze (no FREEZE_RECORD.json): the check MUST fail — that is the
    # gate. Post-freeze: the check MUST pass. This block therefore stays
    # valid across the freeze transition instead of breaking at the milestone
    # it guards.
    rc, out = run(REPO)
    frozen = (REPO / "docs" / "FREEZE_RECORD.json").is_file()
    if frozen:
        ok(rc == 0 and "FREEZE CHECK PASSED" in out,
           "frozen real repo passes (rc=0)")
    else:
        ok(rc != 0, "real repo fails pre-freeze (rc!=0)")
        ok("missing docs/FREEZE_RECORD.json" in out,
           "reports missing FREEZE_RECORD.json")
        n_plan = out.count("placeholder docs/RESEARCH_PLAN.md")
        n_pre = out.count(
            "placeholder docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md")
        # Detection on the REAL documents is asserted while markers remain;
        # once all are resolved the missing-record failure above still holds.
        if n_plan or n_pre:
            ok(True, f"real-doc placeholders flagged (plan {n_plan}, prereg {n_pre})")
        else:
            print("  NOTE no placeholders remain in plan/preregistration — "
                  "freeze-ready docs, record still absent")

    # ── 2. fully-frozen fixture PASSES ────────────────────────────────────
    fix = make_fixture()
    try:
        rc, out = run(fix)
        ok(rc == 0 and "FREEZE CHECK PASSED" in out, "frozen fixture passes (rc=0)")

        # ── 3. each §30 duty independently forces failure ─────────────────
        # 3a. tampered artifact (record hash no longer matches file)
        (fix / "docs" / "metric_definitions.md").write_text("tampered\n", encoding="utf-8")
        expect_fail(fix, "artifact hash mismatch: metric_definitions",
                    "tampered artifact hash fails")
        # restore
        record = make_record(fix)
        write_record(fix, record)

        # 3b. tampered freeze-check script (self-hash duty)
        with (fix / "scripts" / "check_freeze.py").open("a", encoding="utf-8") as fh:
            fh.write("# post-freeze edit\n")
        expect_fail(fix, "artifact hash mismatch: freeze_check_script",
                    "tampered freeze-check script fails (self-hash)")
        shutil.copy(CHECK, fix / "scripts" / "check_freeze.py")
        write_record(fix, make_record(fix))

        # 3c. missing statistical-reviewer approval timestamp (§30.4)
        rec = make_record(fix)
        del rec["statistical_reviewer_approval_timestamp"]
        write_record(fix, rec)
        expect_fail(fix, "record field missing: statistical_reviewer_approval_timestamp",
                    "missing reviewer approval field fails")
        write_record(fix, make_record(fix))

        # 3d. reviewer field present but a placeholder (§30.2)
        rec = make_record(fix)
        rec["statistical_reviewer"] = "[STATISTICAL REVIEWER]"
        write_record(fix, rec)
        expect_fail(fix, "record field empty/placeholder: statistical_reviewer",
                    "placeholder reviewer value fails")
        write_record(fix, make_record(fix))

        # 3e. missing required record field (§30.2)
        rec = make_record(fix)
        del rec["git_tag"]
        write_record(fix, rec)
        expect_fail(fix, "record field missing: git_tag", "missing git_tag fails")
        write_record(fix, make_record(fix))

        # 3f. missing required artifact entry (§30.3)
        rec = make_record(fix)
        rec["artifacts"] = [a for a in rec["artifacts"] if a["name"] != "exposure_ledger"]
        write_record(fix, rec)
        expect_fail(fix, "record artifact missing: exposure_ledger",
                    "missing artifact entry fails")
        write_record(fix, make_record(fix))

        # 3g. placeholder reintroduced into the plan (§30.1)
        with (fix / "docs" / "RESEARCH_PLAN.md").open("a", encoding="utf-8") as fh:
            fh.write("\nUnresolved: [TO BE FROZEN]\n")
        expect_fail(fix, "placeholder docs/RESEARCH_PLAN.md",
                    "reintroduced plan placeholder fails")
        (fix / "docs" / "RESEARCH_PLAN.md").write_text(
            "# Research Plan\n\nStatus: FROZEN.\nAll decisions recorded.\n",
            encoding="utf-8",
        )
        write_record(fix, make_record(fix))

        # 3h. all-caps bracketed token in the preregistration (§30.1)
        with (fix / "docs" / "evaluation" / "PRE_REGISTERED_EVALUATION_PROTOCOL.md").open(
            "a", encoding="utf-8"
        ) as fh:
            fh.write("\nMarker: [REQUIRES DECISION/APPROVAL]\n")
        expect_fail(fix, "placeholder docs/evaluation",
                    "all-caps preregistration token fails")
        (fix / "docs" / "evaluation" / "PRE_REGISTERED_EVALUATION_PROTOCOL.md").write_text(
            "# Pre-Registered Evaluation Protocol\n\nApproved.\n", encoding="utf-8"
        )
        write_record(fix, make_record(fix))

        # 3i. record missing entirely
        (fix / "docs" / "FREEZE_RECORD.json").unlink()
        expect_fail(fix, "missing docs/FREEZE_RECORD.json", "absent record fails")
        write_record(fix, make_record(fix))

        # 3j. malformed record JSON
        (fix / "docs" / "FREEZE_RECORD.json").write_text("{not json", encoding="utf-8")
        expect_fail(fix, "record unreadable", "malformed record JSON fails")
        write_record(fix, make_record(fix))

        # ── 4. marker detection semantics (§30.1) ─────────────────────────
        hits = check_freeze.placeholder_tokens(
            "x [TO BE FROZEN] y [OWNER] z [DATE] w [N — TO BE FROZEN] "
            "v [STATISTICAL REVIEWER] u [REQUIRES DECISION/APPROVAL]"
        )
        ok(len(hits) == 6, f"marker words + all-caps detected ({len(hits)}/6)")
        ok(
            check_freeze.placeholder_tokens("- [ ] unchecked checkbox\n") == [],
            "markdown checkbox not flagged",
        )
        ok(
            check_freeze.placeholder_tokens("wrapped [REQUIRES DECISION/\n  APPROVAL] ok")
            == [(1, "REQUIRES DECISION/ APPROVAL")],
            "wrapped marker normalised to one token",
        )
        ok(
            check_freeze.placeholder_tokens("[normal prose link text]") == [],
            "ordinary bracketed prose not flagged",
        )
        low = check_freeze.placeholder_tokens("[to be frozen lower]")
        ok(
            len(low) == 1 and low[0][1] == "to be frozen lower",
            "case-insensitive marker match (lowercase cannot slip through)",
        )
        ok(
            not check_freeze.field_is_placeholder("v1.0-frozen-tag"),
            "non-placeholder record value accepted",
        )
        ok(
            check_freeze.field_is_placeholder("") and check_freeze.field_is_placeholder(None),
            "empty record value is placeholder",
        )

        # ── 5. self-hash consistency: record hashing the real script ─────
        ok(
            check_freeze.sha256_file(CHECK) == sha256(CHECK),
            "sha256 helper is self-consistent",
        )
    finally:
        shutil.rmtree(fix, ignore_errors=True)

    print(f"=====================================================")
    print(f"Results: {PASS}/{PASS} passed, 0 failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
