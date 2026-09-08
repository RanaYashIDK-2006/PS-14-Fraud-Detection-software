#!/usr/bin/env python3
"""Analyze each rule's FP contribution to find optimization opportunities."""
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk_engine.fusion import FusionEngine
from src.privacy_layer.features import ML_FEATURES

ROOT = Path(__file__).resolve().parent.parent


def check(cond, feat):
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
        return all(check(c, feat) for c in cond["all_of"])
    if "any_of" in cond:
        return any(check(c, feat) for c in cond["any_of"])
    return False


def main():
    df = pd.read_csv(ROOT / "data" / "transactions.csv")
    labels = df["label"].values
    fusion = FusionEngine(ROOT / "models" / "artifacts")
    cfg = yaml.safe_load((ROOT / "src" / "risk_engine" / "rules.yaml").read_text())
    rule_defs = cfg["rules"]
    scale = cfg["severity_scale"]

    n = len(df)
    n_fraud = int(labels.sum())
    n_legit = n - n_fraud

    feature_dicts = [{k: float(df.iloc[i][k]) for k in ML_FEATURES if k in df.columns} for i in range(n)]

    print(f"Dataset: {n} events ({n_fraud} fraud, {n_legit} legit)")
    print(f"severity_scale: {scale}\n")

    print(f"{'rule_id':<42} {'fraud':>8} {'legit':>8} {'prec%':>7} {'score':>7}")
    print("-" * 80)

    low_prec_rules = []
    for rule in rule_defs:
        if rule.get("level") == "critical":
            continue
        cond = rule.get("condition", {})
        fires_f = sum(1 for i in range(n) if labels[i] == 1 and check(cond, feature_dicts[i]))
        fires_l = sum(1 for i in range(n) if labels[i] == 0 and check(cond, feature_dicts[i]))
        total = fires_f + fires_l
        prec = fires_f / total * 100 if total else 0
        sev = rule["severity"]
        score = sev * scale
        print(f"{rule['id']:<42} {fires_f:>5}/{n_fraud:<3} {fires_l:>5}/{n_legit:<3} {prec:>6.1f}% {score:>6.3f}")
        if prec < 20 and fires_l > 50:
            low_prec_rules.append((rule["id"], prec, fires_l, sev))

    print(f"\n{'='*80}")
    print("LOW-PRECISION RULES (prec<20% AND fires on 50+ legit):")
    print(f"{'='*80}")
    for rid, prec, fp, sev in sorted(low_prec_rules, key=lambda x: x[2], reverse=True):
        print(f"  {rid}: precision={prec:.1f}%, FP={fp}, severity={sev}")
        print(f"    → Consider: reduce severity or tighten condition")


if __name__ == "__main__":
    main()
