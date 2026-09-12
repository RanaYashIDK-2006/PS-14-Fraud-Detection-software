#!/usr/bin/env python3
"""Fast severity_scale sweep: find optimal scale for max fraud detection with FPR < 5%."""
import json, sys, yaml
from pathlib import Path
import pandas as pd, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk_engine.fusion import FusionEngine
from src.risk_engine.rules_engine import RulesEngine
from src.privacy_layer.features import ML_FEATURES

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
ARTIFACTS = ROOT / "models" / "artifacts"
RULES_PATH = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
DATA_PATH = ROOT / "data" / "transactions.csv"


def main():
    # Load data and models
    df = pd.read_csv(DATA_PATH)
    labels = df["label"].values
    fusion = FusionEngine(ARTIFACTS)
    cfg = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    rule_defs = cfg["rules"]

    n = len(df)
    n_fraud = int(labels.sum())
    n_legit = n - n_fraud
    fraud_idx = np.where(labels == 1)[0]
    legit_idx = np.where(labels == 0)[0]

    # Build feature dicts
    feature_dicts = [{k: float(df.iloc[i][k]) for k in ML_FEATURES if k in df.columns} for i in range(n)]

    # Batched ML prediction
    print("Computing ML scores...")
    ml_scores = fusion.predict_many(feature_dicts)

    # Precompute per-rule firing for all severities
    print("Computing rule firings...")
    rule_ids = [r["id"] for r in rule_defs]
    rule_sevs = np.array([r["severity"] for r in rule_defs])
    n_rules = len(rule_defs)

    # Evaluate which rules fire for each row
    fired = np.zeros((n, n_rules), dtype=bool)
    critical = np.zeros(n, dtype=bool)

    for j, rule in enumerate(rule_defs):
        cond = rule.get("condition", {})
        level = rule.get("level", "")
        if level == "critical":
            # Mark critical rows
            for i in range(n):
                if _check_condition(cond, feature_dicts[i]):
                    critical[i] = True
                    fired[i, j] = True
        else:
            for i in range(n):
                if _check_condition(cond, feature_dicts[i]):
                    fired[i, j] = True

    # Sweep severity_scale from 0.10 to 1.00 in steps of 0.05
    scales = np.round(np.arange(0.10, 1.05, 0.05), 2)
    threshold = 80  # severity_floor

    results = []
    print(f"\nSweeping severity_scale ({len(scales)} values) on {n} events...")
    print(f"{'scale':>6} {'fraud_caught':>13} {'fraud_%':>8} {'legit_flag':>11} {'fpr_%':>7} {'precision':>10} {'f1':>6}")
    print("-" * 72)

    for scale in scales:
        # Rule score = min(1, sum(sev_of_fired_rules) * scale)
        rule_scores = np.minimum(1.0, (fired @ rule_sevs) * scale)
        combined = np.maximum(ml_scores, rule_scores)
        final = np.round(100 * np.minimum(1.0, combined)).astype(int)
        # Critical rules floor at severity_floor
        final = np.where(critical, np.maximum(final, threshold), final)

        # Threshold: step-up = 30, challenge = 50
        flagged = final >= 30
        tp = int(flagged[fraud_idx].sum())
        fp = int(flagged[legit_idx].sum())
        fraud_pct = tp / n_fraud * 100 if n_fraud else 0
        fpr_pct = fp / n_legit * 100 if n_legit else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) else 0
        recall = tp / n_fraud if n_fraud else 0
        f1 = 2 * precision/100 * recall / (precision/100 + recall) if (precision/100 + recall) else 0

        results.append({
            "scale": scale,
            "fraud_caught": tp,
            "fraud_pct": round(fraud_pct, 1),
            "legit_flagged": fp,
            "fpr_pct": round(fpr_pct, 4),
            "precision_pct": round(precision, 1),
            "f1": round(f1, 4),
        })

        print(f"{scale:>6.2f} {tp:>5}/{n_fraud:<5} {fraud_pct:>6.1f}% {fp:>5}/{n_legit:<5} {fpr_pct:>6.3f}% {precision:>8.1f}% {f1:>5.3f}")

    # Find optimal: maximize fraud detection while FPR < 5%
    eligible = [r for r in results if r["fpr_pct"] < 5.0]
    if eligible:
        # Among eligible, pick the one with highest fraud detection
        best = max(eligible, key=lambda r: r["fraud_caught"])
    else:
        # If no scale achieves FPR < 5%, pick the lowest FPR
        best = min(results, key=lambda r: r["fpr_pct"])

    print(f"\n{'='*72}")
    print(f"OPTIMAL: severity_scale = {best['scale']:.2f}")
    print(f"  Fraud caught: {best['fraud_caught']}/{n_fraud} ({best['fraud_pct']}%)")
    print(f"  False positives: {best['legit_flagged']}/{n_legit} ({best['fpr_pct']}%)")
    print(f"  Precision: {best['precision_pct']}%")
    print(f"  F1: {best['f1']}")
    print(f"{'='*72}")

    # Save results
    out = {
        "optimal_scale": best["scale"],
        "results": results,
        "best": best,
        "n_fraud": n_fraud,
        "n_legit": n_legit,
    }
    out_path = ROOT / "data" / "severity_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


def _check_condition(cond, feat):
    """Check if a rule condition fires on a feature dict."""
    if "feature" in cond:
        val = feat.get(cond["feature"], 0)
        op = cond["op"]
        thr = cond["value"]
        if op == ">=": return val >= thr
        if op == ">": return val > thr
        if op == "<=": return val <= thr
        if op == "<": return val < thr
        if op == "==": return val == thr
        if op == "!=": return val != thr
        return False
    if "all_of" in cond:
        return all(_check_condition(c, feat) for c in cond["all_of"])
    if "any_of" in cond:
        return any(_check_condition(c, feat) for c in cond["any_of"])
    return False


if __name__ == "__main__":
    main()
