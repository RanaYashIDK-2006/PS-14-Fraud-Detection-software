#!/usr/bin/env python3
"""Tune decision threshold and rule severities to maximize precision while keeping detection ≥ 95%."""
import json, sys, yaml
from pathlib import Path
import pandas as pd, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk_engine.fusion import FusionEngine
from src.risk_engine.rules_engine import RulesEngine
from src.privacy_layer.features import ML_FEATURES

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "models" / "artifacts"
RULES_PATH = ROOT / "src" / "risk_engine" / "rules.yaml"


def main():
    df = pd.read_csv(ROOT / "data" / "transactions.csv")
    labels = df["label"].values
    fusion = FusionEngine(ARTIFACTS)
    cfg = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    rule_defs = cfg["rules"]
    scale = cfg.get("severity_scale", 0.25)

    n = len(df)
    n_fraud = int(labels.sum())
    n_legit = n - n_fraud
    fraud_idx = np.where(labels == 1)[0]
    legit_idx = np.where(labels == 0)[0]

    feature_dicts = [{k: float(df.iloc[i][k]) for k in ML_FEATURES if k in df.columns} for i in range(n)]

    print("Computing ML scores...")
    ml_scores = fusion.predict_many(feature_dicts)

    print("Computing rule firings...")
    rule_sevs = np.array([r["severity"] for r in rule_defs])
    fired = np.zeros((n, len(rule_defs)), dtype=bool)
    critical = np.zeros(n, dtype=bool)

    for j, rule in enumerate(rule_defs):
        cond = rule.get("condition", {})
        if rule.get("level") == "critical":
            for i in range(n):
                if _check(cond, feature_dicts[i]):
                    critical[i] = True
                    fired[i, j] = True
        else:
            for i in range(n):
                if _check(cond, feature_dicts[i]):
                    fired[i, j] = True

    rule_scores = np.minimum(1.0, (fired @ rule_sevs) * scale)
    combined = np.maximum(ml_scores, rule_scores)

    # Sweep threshold from 20 to 80
    print(f"\n{'threshold':>10} {'fraud':>10} {'det%':>6} {'fp':>6} {'fpr%':>6} {'prec%':>7} {'f1':>6} {'f1/prec':>8}")
    print("-" * 72)

    best_f1_at_95 = None
    best_prec_at_95 = None

    for thresh in range(20, 81, 2):
        final = np.round(100 * np.minimum(1.0, combined)).astype(int)
        final = np.where(critical, np.maximum(final, 80), final)

        flagged = final >= thresh
        tp = int(flagged[fraud_idx].sum())
        fp = int(flagged[legit_idx].sum())
        det_pct = tp / n_fraud * 100
        fpr_pct = fp / n_legit * 100
        prec = tp / (tp + fp) * 100 if (tp + fp) else 0
        recall = tp / n_fraud
        f1 = 2 * (prec/100) * recall / ((prec/100) + recall) if ((prec/100) + recall) else 0
        eff = f1 / (prec/100) if prec > 0 else 0  # efficiency ratio

        marker = ""
        if det_pct >= 95.0:
            if best_f1_at_95 is None or f1 > best_f1_at_95["f1"]:
                best_f1_at_95 = {"thresh": thresh, "det": det_pct, "fp": fp, "fpr": fpr_pct, "prec": prec, "f1": f1}
            if best_prec_at_95 is None or prec > best_prec_at_95["prec"]:
                best_prec_at_95 = {"thresh": thresh, "det": det_pct, "fp": fp, "fpr": fpr_pct, "prec": prec, "f1": f1}

        if thresh in [30, 40, 50, 60]:
            marker = " <-- current" if thresh == 30 else ""

        print(f"{thresh:>10} {tp:>5}/{n_fraud:<4} {det_pct:>5.1f}% {fp:>6} {fpr_pct:>5.2f}% {prec:>6.1f}% {f1:>5.3f} {eff:>7.3f}{marker}")

    print(f"\n{'='*72}")
    print(f"OPTIMAL (max F1, det≥95%): threshold={best_f1_at_95['thresh']}")
    print(f"  Fraud: {best_f1_at_95['det']:.1f}% | FP: {best_f1_at_95['fp']} ({best_f1_at_95['fpr']:.2f}%)")
    print(f"  Precision: {best_f1_at_95['prec']:.1f}% | F1: {best_f1_at_95['f1']:.3f}")
    print(f"\nOPTIMAL (max Precision, det≥95%): threshold={best_prec_at_95['thresh']}")
    print(f"  Fraud: {best_prec_at_95['det']:.1f}% | FP: {best_prec_at_95['fp']} ({best_prec_at_95['fpr']:.2f}%)")
    print(f"  Precision: {best_prec_at_95['prec']:.1f}% | F1: {best_prec_at_95['f1']:.3f}")

    # Also show what happens at common thresholds
    print(f"\n{'='*72}")
    print("RECOMMENDED THRESHOLDS:")
    for t in [30, 40, 50, 60]:
        final = np.round(100 * np.minimum(1.0, combined)).astype(int)
        final = np.where(critical, np.maximum(final, 80), final)
        flagged = final >= t
        tp = int(flagged[fraud_idx].sum())
        fp = int(flagged[legit_idx].sum())
        det = tp/n_fraud*100
        fpr = fp/n_legit*100
        prec = tp/(tp+fp)*100 if (tp+fp) else 0
        print(f"  threshold={t}: det={det:.1f}% FPR={fpr:.2f}% prec={prec:.1f}%")

    # Save
    out = {"best_f1": best_f1_at_95, "best_prec": best_prec_at_95, "scale": scale}
    with open(ROOT / "data" / "precision_tuning_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to data/precision_tuning_results.json")


def _check(cond, feat):
    if "feature" in cond:
        val = feat.get(cond["feature"], 0)
        op, thr = cond["op"], cond["value"]
        if op == ">=": return val >= thr
        if op == ">": return val > thr
        if op == "<=": return val <= thr
        if op == "<": return val < thr
        if op == "==": return val == thr
        return False
    if "all_of" in cond:
        return all(_check(c, feat) for c in cond["all_of"])
    if "any_of" in cond:
        return any(_check(c, feat) for c in cond["any_of"])
    return False


if __name__ == "__main__":
    main()
