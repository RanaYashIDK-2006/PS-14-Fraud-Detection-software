#!/usr/bin/env python3
"""Tests for scripts/training_pipeline.py mechanics.

Covers the pieces that are cheap to test without a full retrain:
  1. apply_scale writes the new value, preserves every other line
     byte-for-byte (comments included), and keeps a .bak of the old file,
  2. apply_scale is idempotent (same scale -> no change, no second backup),
  3. apply_scale fails loudly when severity_scale is missing,
  4. the default train_compare invocation carries the seed, data, artifact
     dir, and the feedback pool (preserving the current artifacts'
     provenance), and disables feedback when the flag is cleared.

Run from the project root:
  python scripts/pipeline_test.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

import training_pipeline as pipe  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


YAML = """\
severity_floor: 80      # critical rules push the score to at least this
severity_scale: 0.45    # tuned by scripts/tune_operating_point.py (C_FN/C_FP=20)
                        # post-cold-start-fix retune: 1.77% -> 0.14% legit
                        # challenges, fraud recall held at 92.3%

rules:
  - id: RULE_AMOUNT_HIGH
    description: Transaction amount is unusual for this account
    reason_code: AMOUNT_UNUSUAL
    severity: 0.30
    condition:
      feature: amount_ratio
      op: ">="
      value: 1.3
"""


def main() -> int:
    print("== training-pipeline mechanics tests ==")
    tmp = Path(tempfile.mkdtemp(prefix="ps14-pipe-"))
    rules = tmp / "rules.yaml"
    rules.write_text(YAML, encoding="utf-8")

    # ---- 1. apply writes the value, preserves everything else ------------
    changed = pipe.apply_scale(rules, 0.60)
    check("apply reports a change", changed is True)
    check("new value written", rules.read_text(encoding="utf-8").startswith(
        "severity_floor: 80      # critical rules"), rules.read_text(encoding="utf-8").splitlines()[1])
    check("severity_scale updated to 0.60",
          rules.read_text(encoding="utf-8").splitlines()[1].startswith("severity_scale: 0.60"),
          rules.read_text(encoding="utf-8").splitlines()[1])
    preserved = rules.read_text(encoding="utf-8").splitlines()
    orig = YAML.splitlines()
    # every line except the severity_scale value line must be unchanged
    for i, line in enumerate(orig):
        if i == 1:
            continue  # the tuned line itself
        check(f"line {i} preserved: {line[:40]!r}", line == preserved[i], preserved[i])

    bak = rules.with_suffix(".yaml.bak")
    check("backup holds the previous content",
          bak.exists() and "severity_scale: 0.45" in bak.read_text(encoding="utf-8"))
    check("no temp file left behind", not rules.with_suffix(".yaml.new").exists())

    # ---- 2. idempotent: same scale -> no change, no second backup ---------
    bak_mtime = bak.stat().st_mtime_ns
    changed2 = pipe.apply_scale(rules, 0.60)
    check("same scale -> no change", changed2 is False)
    check("no second backup", bak.stat().st_mtime_ns == bak_mtime)

    # ---- 3. missing severity_scale -> loud failure ------------------------
    bad = tmp / "bad.yaml"
    bad.write_text("severity_floor: 80\nrules: []\n", encoding="utf-8")
    try:
        pipe.apply_scale(bad, 0.60)
        check("missing severity_scale raises", False)
    except SystemExit as e:
        check("missing severity_scale raises", "severity_scale" in str(e), str(e))

    # ---- 4. train_compare argv wiring -------------------------------------
    ns = argparse.Namespace(
        data=tmp / "data" / "transactions.csv", artifacts_dir=tmp / "models" / "artifacts",
        seed=42, feedback=tmp / "data" / "feedback_labeled.csv", feedback_date=None,
    )
    argv = pipe.build_train_argv(ns)
    joined = " ".join(argv)
    check("argv carries the seed", "--seed 42" in joined)
    check("argv carries the data path", "--data" in joined and "transactions.csv" in joined)
    check("argv carries the artifact dir", "--outdir" in joined and "artifacts" in joined)
    check("argv carries the feedback pool", "--feedback" in joined and "feedback_labeled.csv" in joined)
    ns.feedback = None
    argv_nf = pipe.build_train_argv(ns)
    check("feedback can be disabled", "--feedback" not in " ".join(argv_nf))
    ns.feedback_date = "latest"
    argv_fd = pipe.build_train_argv(ns)
    check("feedback-date passes through", "--feedback-date latest" in " ".join(argv_fd))

    # ---- 5. pipeline default pool resolution -------------------------------
    snaps = tmp / "snaps"
    snaps.mkdir()
    (snaps / "feedback_20260815_100000.csv").write_text("a\n", encoding="utf-8")
    (snaps / "feedback_20260816_090000.csv").write_text("b\n", encoding="utf-8")
    ns = argparse.Namespace(feedback=None, feedback_date=None, snapshot_dir=snaps)
    pipe.resolve_default_feedback(ns, tmp / "pool.csv")
    check("pipeline default resolves the latest snapshot",
          ns.feedback is not None and ns.feedback.name == "feedback_20260816_090000.csv",
          str(ns.feedback))

    ns = argparse.Namespace(feedback=None, feedback_date=None, snapshot_dir=tmp / "empty")
    fallback = tmp / "pool.csv"
    fallback.write_text("x\n", encoding="utf-8")
    pipe.resolve_default_feedback(ns, fallback)
    check("pipeline falls back to the pool file when no snapshots",
          ns.feedback is not None and ns.feedback.resolve() == fallback.resolve(),
          str(ns.feedback))

    ns = argparse.Namespace(feedback=None, feedback_date=None, snapshot_dir=tmp / "empty2")
    pipe.resolve_default_feedback(ns, tmp / "missing.csv")
    check("pipeline leaves feedback off when nothing exists", ns.feedback is None)

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
