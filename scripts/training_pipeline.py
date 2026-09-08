#!/usr/bin/env python3
"""End-to-end reproducible training pipeline (sections 6/7):

    retrain -> retune severity_scale -> update rules.yaml -> re-run risk-engine tests

One command:

    python scripts/training_pipeline.py [--cost-ratio 20] [--data data/transactions.csv]
                                        [--feedback data/feedback_labeled.csv] [--seed 42]
                                        [--skip-train] [--skip-test] [--full] [--no-cache]

Deterministic by construction - seeds are fixed (default 42), the tuning
sweep is pure numpy, and the tests are deterministic - so the same inputs
reproduce the same artifacts, the same recommended `severity_scale`, and
the same green test run. Each step fails fast with a non-zero exit.

Steps:
  1. RETRAIN   python src/train_compare.py --outdir models/artifacts --seed <seed>
              [--feedback ...]. The tuner's cache key includes the artifact
              file states, so fresh artifacts automatically invalidate the
              old per-event score cache.
  2. RETUNE    global `severity_scale` sweep on the fresh artifacts
              (cache-aware: the first post-retrain run recomputes the
              per-event signals, ~3-6 min; re-runs hit the cache). The
              recommended scale is scale-agnostic - it does not depend on
              the scale currently in rules.yaml.
  3. APPLY     recommended scale written into src/risk_engine/rules.yaml
              atomically (temp file + os.replace); the previous file is
              preserved as rules.yaml.bak. A severity_scale-only change
              does NOT invalidate the tuner cache.
  4. TEST      scripts/risk_engine_test.py + scripts/tune_test.py
              (--full adds the other six suites).

Flags:
  --rules PATH   target a different rules.yaml (staging/dry-run; combined
                 with --skip-test this never touches the live config)
  --skip-train   reuse the current artifacts (retune + apply + test only)
  --skip-test    skip the test step (e.g. staging apply)
  --full         run all eight test suites in step 4
  --no-cache     force recompute of the per-event scores
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import tune_operating_point as tune  # noqa: E402
from src.train_compare import resolve_feedback  # noqa: E402

ARTIFACTS_DIR = ROOT / "models" / "artifacts"
RULES_PATH = ROOT / "src" / "risk_engine" / "rules.yaml"
DATA_PATH = ROOT / "data" / "transactions.csv"
REPORT_PATH = ROOT / "models" / "tuning_report.md"
FEEDBACK_PATH = ROOT / "data" / "feedback_labeled.csv"

ALL_SUITES = [
    "smoke_test", "risk_engine_test", "verification_test", "audit_test",
    "drift_test", "k_anonymity_test", "federated_test", "tune_test",
    "rules_gate_test", "ood_gate_test", "front_service_test",
]


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

def run_step(name: str, argv: list[str]) -> None:
    """Run a subprocess step; exit the pipeline on failure."""
    print(f"\n=== {name} ===")
    t0 = time.time()
    p = subprocess.run([sys.executable, *argv], cwd=ROOT)
    elapsed = time.time() - t0
    ok = p.returncode == 0
    print(f"--- {name}: {'OK' if ok else f'FAILED (exit {p.returncode})'} in {elapsed:.1f}s")
    if not ok:
        sys.exit(p.returncode)


def build_train_argv(args: argparse.Namespace) -> list[str]:
    """The train_compare invocation for this pipeline run (testable)."""
    argv = [
        str(ROOT / "src" / "train_compare.py"),
        "--data", os.path.relpath(args.data, ROOT),
        "--outdir", os.path.relpath(args.artifacts_dir, ROOT),
        "--seed", str(args.seed),
    ]
    if args.feedback:
        argv += ["--feedback", os.path.relpath(args.feedback, ROOT)]
    elif args.feedback_date:
        argv += ["--feedback-date", args.feedback_date]
    return argv


def resolve_default_feedback(args: argparse.Namespace, fallback_pool: Path) -> None:
    """Default pool resolution: consume the LATEST dated snapshot when the
    user did not pin --feedback / --feedback-date, falling back to the
    working pool file only if no snapshots exist yet (backward compat)."""
    if args.feedback is not None or args.feedback_date is not None:
        return
    try:
        latest, tag = resolve_feedback(None, "latest", args.snapshot_dir)
        args.feedback = latest
        print(f"feedback pool: latest snapshot {tag} ({len(pd.read_csv(latest))} rows)")
    except SystemExit:
        if fallback_pool.exists():
            args.feedback = fallback_pool
            print(f"feedback pool: no snapshots yet - using {fallback_pool.name}")
        else:
            print("feedback pool: none (no snapshots, no pool file)")


def retune(cost_ratio: float, use_cache: bool, data_path: Path,
           rules_path: Path, report_path: Path, artifacts_dir: Path) -> float:
    """Global severity_scale sweep on the (fresh) artifacts -> best scale."""
    print("\n=== RETUNE (global severity_scale sweep) ===")
    t0 = time.time()
    df = pd.read_csv(data_path, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    n = len(df)
    vi, ti = int(n * 0.70), int(n * 0.85)
    val = df.iloc[vi:ti].reset_index(drop=True)
    test = df.iloc[ti:].reset_index(drop=True)

    tune.ARTIFACTS_DIR = artifacts_dir
    tune.RULES_PATH = rules_path
    fusion = tune.FusionEngine(artifacts_dir)
    rules = tune.RulesEngine.from_yaml(rules_path)
    _, sev, _ = tune._rules_config()

    t1 = time.time()
    ml_tail, fired_tail, crit_tail, sev_tail, key, hit = tune.load_or_compute(
        df, vi, ti, fusion, rules, use_cache=use_cache
    )
    print(f"per-event scores: {'cache HIT' if hit else 'computed'} ({time.time() - t1:.1f}s)  key={key}")

    n_val = ti - vi
    best_scale, _ = tune.sweep_global_scale(
        ml_tail[:n_val], fired_tail[:n_val], crit_tail[:n_val], sev_tail,
        val["label"].to_numpy(),
        ml_tail[n_val:], fired_tail[n_val:], crit_tail[n_val:],
        test["label"].to_numpy(),
        cost_ratio, report_path,
    )
    print(f"\nRecommended severity_scale: {best_scale:.2f}")
    return float(best_scale)


def apply_scale(rules_path: Path, best_scale: float) -> bool:
    """Write `best_scale` into rules.yaml atomically; keep a .bak of the
    previous file. Returns True if the file changed. Only the value on the
    `severity_scale:` line is touched - comments and everything else are
    preserved byte-for-byte."""
    text = rules_path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"^(severity_scale:\s*)[\d.]+", rf"\g<1>{best_scale:.2f}",
        text, count=1, flags=re.M,
    )
    if n != 1:
        raise SystemExit(f"could not locate a `severity_scale:` line in {rules_path}")
    if new_text == text:
        print(f"rules.yaml already at severity_scale {best_scale:.2f} - no change")
        return False
    bak = rules_path.with_suffix(rules_path.suffix + ".bak")
    shutil.copy2(rules_path, bak)
    tmp = rules_path.with_suffix(rules_path.suffix + ".new")
    tmp.write_text(new_text, encoding="utf-8")
    os.replace(tmp, rules_path)
    cfg = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    assert abs(float(cfg["severity_scale"]) - best_scale) < 1e-9, cfg["severity_scale"]
    print(f"severity_scale: {best_scale:.2f} written to {rules_path} (previous kept in {bak})")
    return True


def refresh_baseline(rules_path: Path, baseline_path: Path) -> None:
    """The tuned rules become the last-accepted baseline snapshot the CI
    rules gate compares future edits against."""
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rules_path, baseline_path)
    print(f"baseline snapshot refreshed: {baseline_path.name} <- {rules_path.name}")


def run_tests(full: bool, skip_test: bool) -> None:
    if skip_test:
        return
    suites = ALL_SUITES if full else ["risk_engine_test", "tune_test"]
    for s in suites:
        run_step(f"TEST {s}", [str(ROOT / "scripts" / f"{s}.py")])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-ratio", type=float, default=20.0)
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--feedback", default=None,
                    help="explicit labeled feedback CSV to merge (default: latest snapshot)")
    ap.add_argument("--feedback-date", default=None,
                    help="YYYYMMDD[HHMMSS] or 'latest': consume the feedback snapshot at or before that time")
    ap.add_argument("--snapshot-dir", default=str(ROOT / "data" / "feedback_snapshots"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rules", default=str(RULES_PATH),
                    help="rules.yaml to retune and write (staging/dry-run)")
    ap.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    ap.add_argument("--report", default=str(REPORT_PATH))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="also run the other six suites in step 4")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    args.data = Path(args.data)
    args.feedback = Path(args.feedback) if args.feedback else None
    args.rules = Path(args.rules)
    args.artifacts_dir = Path(args.artifacts)
    args.report = Path(args.report)
    args.snapshot_dir = Path(args.snapshot_dir)
    resolve_default_feedback(args, ROOT / "data" / "feedback_labeled.csv")

    print("=" * 72)
    print("PS-14 training pipeline (retrain -> retune -> apply -> test)")
    print(f"  seed={args.seed}  data={args.data.name}  feedback={'on' if args.feedback else 'off'}"
          f"  cost_ratio C_FN/C_FP={args.cost_ratio:.1f}")
    print(f"  artifacts={args.artifacts_dir}  rules={args.rules}  report={args.report}")
    print("=" * 72)

    if not args.skip_train:
        run_step("RETRAIN", build_train_argv(args))

    best_scale = retune(args.cost_ratio, not args.no_cache, args.data,
                        args.rules, args.report, args.artifacts_dir)
    apply_scale(args.rules, best_scale)
    # Only a production config apply refreshes the accepted baseline; a
    # --rules staging dry-run must not move it.
    if args.rules.resolve() == RULES_PATH.resolve():
        refresh_baseline(args.rules, ROOT / "models" / "rules_baseline.yaml")
    run_tests(args.full, args.skip_test)

    print("\n=== PIPELINE COMPLETE (exit 0) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
