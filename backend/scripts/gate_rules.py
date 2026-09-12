#!/usr/bin/env python3
"""Rules-simulation CI gate: block a rules.yaml edit that worsens the
allow-vs-challenge trade-off.

Every rules.yaml edit is replayed over the labeled history with the live
calibrated ML fusion (shared cached scores), and the candidate's metrics are
compared with the LAST-ACCEPTED baseline snapshot
(`models/rules_baseline.yaml`, refreshed by `--accept` and by the training
pipeline after each retune). The gate fails the edit if:

  * FPR increases by more than `--max-fpr-delta` (default +0.02, 2pp), or
  * recall drops by more than `--max-recall-delta` (default -0.02, 2pp),
  * and either absolute floor is breached (`--max-fpr`, `--min-recall`).

Usage (CI / pre-commit style):

  python scripts/gate_rules.py                 # gate current rules.yaml
  python scripts/gate_rules.py --rules staging.yaml --max-fpr-delta 0.01
  python scripts/gate_rules.py --accept        # approve current rules.yaml
  python scripts/gate_rules.py --json gate.json

Exit codes: 0 = pass (safe to ship), 1 = GATE BREACHED (block the edit),
2 = configuration error (missing baseline/data/artifacts).

Reuses the simulator's evaluation machinery (`simulate_rules.evaluate_config`)
and its disk-cached ML scores, so an unchanged dataset + artifacts makes
each gate run finish in seconds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

import yaml  # noqa: E402

import simulate_rules as sim  # noqa: E402
from src.risk_engine.rules_engine import RulesEngine  # noqa: E402

RULES_PATH = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
BASELINE_PATH = ROOT / "models" / "rules_baseline.yaml"
DATA_PATH = ROOT / "data" / "transactions.csv"


def load_rules(path: Path) -> RulesEngine:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RulesEngine(cfg["rules"], severity_scale=float(cfg.get("severity_scale", 1.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", default=str(RULES_PATH), help="candidate rules.yaml to gate (default: current)")
    ap.add_argument("--baseline", default=str(BASELINE_PATH), help="last-accepted baseline snapshot")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--max-fpr-delta", type=float, default=0.02, help="block FPR increase beyond this (abs pp)")
    ap.add_argument("--max-recall-delta", type=float, default=0.02, help="block recall drop beyond this (abs pp)")
    ap.add_argument("--max-fpr", type=float, default=None, help="optional absolute FPR ceiling")
    ap.add_argument("--min-recall", type=float, default=None, help="optional absolute recall floor")
    ap.add_argument("--json", default=None, help="write the machine-readable verdict to this path")
    ap.add_argument("--no-cache", action="store_true", help="recompute ML scores, ignore the disk cache")
    ap.add_argument("--accept", action="store_true",
                    help="approve the candidate: copy it over the baseline snapshot and exit 0")
    args = ap.parse_args()

    cand_path = ROOT / args.rules
    base_path = ROOT / args.baseline
    data_path = ROOT / args.data
    if not cand_path.exists():
        sys.exit(f"candidate rules not found: {cand_path}")
    if not data_path.exists():
        sys.exit(f"data not found: {data_path}")

    cand_text = cand_path.read_text(encoding="utf-8")

    # --accept: the reviewed/tuned config becomes the new accepted baseline.
    if args.accept:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cand_path, base_path)
        print(f"accepted: {cand_path.name} copied to {base_path}")
        return 0

    if not base_path.exists():
        print(f"no baseline snapshot at {BASELINE_PATH} - run "
              f"`python scripts/gate_rules.py --accept` (or the training pipeline) first")
        return 2

    base_text = base_path.read_text(encoding="utf-8")
    if base_text == cand_text:
        print("candidate is byte-identical to the accepted baseline - no gate needed (PASS)")
        return 0

    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(data_path)
    if not {"label"}.issubset(df.columns):
        sys.exit(f"{data_path.name} has no `label` column - expected a labeled training CSV")

    fusion = sim.FusionEngine(sim.ARTIFACTS)
    ml = sim.compute_ml(df, fusion, use_cache=not args.no_cache)
    print(f"rules gate over {data_path.name} ({len(df):,} events, "
          f"{int(df['label'].sum())} fraud), ML scores shared from cache")

    base_cfg = yaml.safe_load(base_text)
    cand_cfg = yaml.safe_load(cand_text)
    base_floor = int(base_cfg.get("severity_floor", 80))
    cand_floor = int(cand_cfg.get("severity_floor", 80))

    base = sim.evaluate_config(df, ml, load_rules(base_path), base_floor)
    cand = sim.evaluate_config(df, ml, load_rules(cand_path), cand_floor)

    print()
    sim.summarize(f"baseline (accepted: {base_path.name})", base)
    sim.summarize(f"candidate ({cand_path.name})", cand)

    fpr_delta = cand["fpr"] - base["fpr"]
    recall_delta = cand["recall"] - base["recall"]
    print(f"\n  FPR    {base['fpr']:.4f} -> {cand['fpr']:.4f}  (delta {fpr_delta:+.4f})")
    print(f"  recall {base['recall']:.4f} -> {cand['recall']:.4f}  (delta {recall_delta:+.4f})")

    breaches: list[str] = []
    if fpr_delta > args.max_fpr_delta:
        breaches.append(f"FPR increased {fpr_delta:+.4f} (cap +{args.max_fpr_delta:.2f})")
    if recall_delta < -args.max_recall_delta:
        breaches.append(f"recall dropped {recall_delta:+.4f} (cap -{args.max_recall_delta:.2f})")
    if args.max_fpr is not None and cand["fpr"] > args.max_fpr:
        breaches.append(f"FPR {cand['fpr']:.4f} above absolute ceiling {args.max_fpr:.2f}")
    if args.min_recall is not None and cand["recall"] < args.min_recall:
        breaches.append(f"recall {cand['recall']:.4f} below absolute floor {args.min_recall:.2f}")

    changed = int((base["decisions"] != cand["decisions"]).sum())
    verdict = "PASS" if not breaches else "GATE BREACHED"
    print(f"\n  decision changes: {changed} of {len(df):,}")
    print(f"  verdict: {verdict}" + ("" if not breaches else " -> " + "; ".join(breaches)))

    if args.json:
        report = {
            "candidate": str(cand_path),
            "baseline": str(base_path),
            "n_events": int(len(df)),
            "baseline_metrics": {k: v for k, v in base.items() if k not in ("scores", "decisions")},
            "candidate_metrics": {k: v for k, v in cand.items() if k not in ("scores", "decisions")},
            "fpr_delta": round(fpr_delta, 4),
            "recall_delta": round(recall_delta, 4),
            "n_decision_changes": changed,
            "breaches": breaches,
            "verdict": "pass" if not breaches else "block",
        }
        (ROOT / args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  report written to {args.json}")

    return 0 if not breaches else 1


if __name__ == "__main__":
    sys.exit(main())
