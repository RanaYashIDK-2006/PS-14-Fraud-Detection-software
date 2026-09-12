#!/usr/bin/env python3
"""Rule backtesting simulator - the "what-if" step of a rules studio.

Replays a rule configuration over labeled history (data/transactions.csv,
the same set the models were trained on) and reports how the final
allow / step-up / verify decisions would change compared with the rules
currently in src/risk_engine/rules.yaml - so a rule edit can be evaluated
against the past before it ships.

  python scripts/simulate_rules.py                      # baseline only
  python scripts/simulate_rules.py --rules candidate.yaml
  python scripts/simulate_rules.py --severity-scale 0.60
  python scripts/simulate_rules.py --rules r.yaml --json out.json

The ML fusion score is computed ONCE with the live FusionEngine (calibrated)
and shared across configurations, so a sweep of candidate rule sets only
re-evaluates the rules (fast). Metrics at the allow-vs-challenge boundary
(any decision other than ALLOW counts as challenged):

  recall        fraction of fraud events challenged
  fpr           fraction of legit events challenged (false-positive rate)
  challenge     overall fraction of events challenged
  verify/step   decision mix

`--live-db PATH` adds a label-free pass over a stored DB-2 feature store
(scripts/drift_monitor-style), reporting the decision mix over real ingested
events when no labels exist.

Exit code 0 on success; non-zero on missing inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.privacy_layer.features import ML_FEATURES  # noqa: E402
from src.risk_engine.fusion import FusionEngine  # noqa: E402
from src.risk_engine.rules_engine import RulesEngine  # noqa: E402

ARTIFACTS = ROOT / "models" / "artifacts"
RULES_PATH = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
DATA_PATH = ROOT / "data" / "transactions.csv"


def band_of(score: int) -> str:
    """Mirror of src/risk_engine/main.py::band_of — FPR < 1%."""
    if score < 85:
        return "allow"
    return "verify"


def final_decision(ml: float, rule: dict, severity_floor: int) -> tuple[int, str]:
    score = round(100 * min(1.0, max(ml, rule["score"])))
    if rule["critical"] and score < severity_floor:
        score = severity_floor
    return score, band_of(score)


def evaluate_config(features: pd.DataFrame, ml: np.ndarray, rules: RulesEngine,
                    severity_floor: int) -> dict:
    """Decisions for one rules config over the full dataset."""
    labels = features["label"].to_numpy(dtype=int)
    scores = np.empty(len(features), dtype=int)
    decisions: list[str] = []
    fired = 0
    for i, row in enumerate(features.itertuples(index=False)):
        fdict = {name: getattr(row, name) for name in ML_FEATURES}
        rule = rules.evaluate(fdict)
        s, d = final_decision(ml[i], rule, severity_floor)
        scores[i] = s
        decisions.append(d)
        fired += bool(rule["fired_rules"])
    decisions = np.asarray(decisions)
    challenged = decisions != "allow"
    return {
        "n": int(len(features)),
        "n_fraud": int(labels.sum()),
        "recall": float(challenged[labels == 1].mean()),
        "fpr": float(challenged[labels == 0].mean()),
        "challenge_rate": float(challenged.mean()),
        "verify": int((decisions == "verify").sum()),
        "step_up": int((decisions == "step_up").sum()),
        "allow": int((decisions == "allow").sum()),
        "avg_score_fraud": float(scores[labels == 1].mean()),
        "avg_score_legit": float(scores[labels == 0].mean()),
        "n_events_with_rules": fired,
        "scores": scores,
        "decisions": decisions,
    }


def summarize(label: str, r: dict) -> None:
    print(f"\n{label}:")
    print(f"  events            : {r['n']:,}")
    print(f"  decision mix      : {r['allow']:,} allow / {r['step_up']:,} step-up / {r['verify']:,} verify")
    print(f"  recall (fraud)    : {r['recall']:.4f}   ({int(round(r['recall'] * r['n_fraud']))} of {r['n_fraud']} fraud challenged)")
    print(f"  fpr (legit)       : {r['fpr']:.4f}")
    print(f"  challenge rate    : {r['challenge_rate']:.4f}")
    print(f"  avg score         : fraud {r['avg_score_fraud']:.1f} / legit {r['avg_score_legit']:.1f}")


def _content_hash(df: pd.DataFrame) -> str:
    """Hash of the labeled feature matrix - the gate must be deterministic
    for a given dataset + model artifacts (independent of file paths)."""
    cols = ML_FEATURES + ["label"]
    payload = df[cols].to_numpy().tobytes()
    return hashlib.sha1(payload).hexdigest()[:16]


def _artifacts_hash() -> str:
    """Hash of the model-artifact file states (content + mtime)."""
    h = hashlib.sha1()
    for p in sorted(ARTIFACTS.iterdir()):
        if p.suffix in (".joblib", ".json"):
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def compute_ml(df: pd.DataFrame, fusion: FusionEngine, use_cache: bool = True) -> np.ndarray:
    """Calibrated ML scores for every row, computed once and cached on disk.

    The cache key covers the dataset content + the model artifacts (both
    rules-INDEPENDENT), so a gate/sweep that only changes rules.yaml reuses
    the same scores and runs in seconds instead of minutes. `--no-cache`
    forces a recompute (e.g. after a manual artifacts edit).
    """
    key = _content_hash(df) + _artifacts_hash()
    cache_dir = ROOT / "models" / "cache"
    cache_path = cache_dir / f"rules_ml_{key}.npy"
    if use_cache and cache_path.exists():
        return np.load(cache_path)
    ml = fusion.predict_many([
        {f: float(getattr(row, f)) for f in ML_FEATURES} for row in df.itertuples(index=False)
    ])
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, ml)
        print(f"ML scores cached to {cache_path.name} (key {key})")
    return ml


def live_distribution(db_path: Path, rules: RulesEngine, severity_floor: int) -> dict:
    """Label-free pass over a stored DB-2 feature store: decision mix only."""
    con = sqlite3.connect(db_path)
    cols = ", ".join(f'"{c}"' for c in ML_FEATURES)
    rows = con.execute(f"SELECT {cols} FROM transaction_features").fetchall()
    con.close()
    if not rows:
        print(f"  live DB {db_path.name}: no stored events")
        return {}
    ml = np.array([fusion.predict(dict(zip(ML_FEATURES, r)))[0] for r in rows])
    fdicts = [dict(zip(ML_FEATURES, r)) for r in rows]
    counts = {"allow": 0, "step_up": 0, "verify": 0}
    for f, p in zip(fdicts, ml):
        _, d = final_decision(p, rules.evaluate(f), severity_floor)
        counts[d] += 1
    print(f"  live DB {db_path.name}: {len(rows)} stored events -> "
          f"{counts['allow']} allow / {counts['step_up']} step-up / {counts['verify']} verify")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", default=None, help="candidate rules.yaml (default: none, baseline only)")
    ap.add_argument("--severity-scale", type=float, default=None,
                    help="candidate severity_scale override (applied to the baseline rules)")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--live-db", default=None, help="optional DB-2 features.db for a label-free pass")
    ap.add_argument("--json", default=None, help="write the machine-readable report to this path")
    ap.add_argument("--no-cache", action="store_true", help="recompute ML scores, ignore the disk cache")
    args = ap.parse_args()

    if not args.rules and args.severity_scale is None:
        ap.error("nothing to simulate: pass --rules or --severity-scale")
    if args.rules and args.severity_scale is not None:
        ap.error("--rules and --severity-scale are mutually exclusive")

    data_path = ROOT / args.data
    if not data_path.exists():
        sys.exit(f"data not found: {data_path}")

    global fusion
    fusion = FusionEngine(ARTIFACTS)
    base_cfg = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    floor = int(base_cfg.get("severity_floor", 80))
    base_rules = RulesEngine(base_cfg["rules"], severity_scale=float(base_cfg.get("severity_scale", 1.0)))

    df = pd.read_csv(data_path)
    if not {"label"}.issubset(df.columns):
        sys.exit(f"{data_path.name} has no `label` column - expected a labeled training CSV")
    print(f"rules simulator over {data_path.name} "
          f"({len(df):,} events, {int(df['label'].sum())} fraud)\n")

    ml = compute_ml(df, fusion, use_cache=not args.no_cache)
    print(f"ML fusion ready ({len(ml):,} events, calibrated) - shared by all configs")

    base = evaluate_config(df, ml, base_rules, floor)
    summarize("baseline (rules.yaml)", base)

    if args.severity_scale is not None:
        cand_rules = RulesEngine(base_cfg["rules"], severity_scale=args.severity_scale)
        cand_label = f"candidate (severity_scale {args.severity_scale})"
    else:
        cand_path = ROOT / args.rules
        cand_cfg = yaml.safe_load(cand_path.read_text(encoding="utf-8"))
        cand_rules = RulesEngine(cand_cfg["rules"], severity_scale=float(cand_cfg.get("severity_scale", 1.0)))
        cand_label = f"candidate ({cand_path.name})"
        floor = int(cand_cfg.get("severity_floor", floor))
    cand = evaluate_config(df, ml, cand_rules, floor)
    summarize(cand_label, cand)

    # ---- delta -------------------------------------------------------------
    changed = np.where(base["decisions"] != cand["decisions"])[0]
    print(f"\ndecision changes: {len(changed):,} of {len(df):,} events")
    for i in changed[:10]:
        ev = df.iloc[i]
        print(f"  {ev['event_id']}  {base['decisions'][i]:>8} -> {cand['decisions'][i]:<8} "
              f"(score {base['scores'][i]} -> {cand['scores'][i]}, label {int(ev['label'])})")
    if len(changed) > 10:
        print(f"  ... and {len(changed) - 10} more")

    if args.live_db:
        live_distribution(ROOT / args.live_db, cand_rules, floor)

    if args.json:
        report = {
            "data": str(data_path),
            "baseline": {k: v for k, v in base.items() if k not in ("scores", "decisions")},
            "candidate": {k: v for k, v in cand.items() if k not in ("scores", "decisions")},
            "n_decision_changes": int(len(changed)),
        }
        (ROOT / args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
