#!/usr/bin/env python3
"""Analyze which rules cause the most false positives."""

import csv
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def eval_condition(cond: dict, features: dict) -> bool:
    if "all_of" in cond:
        return all(eval_condition(c, features) for c in cond["all_of"])
    if "any_of" in cond:
        return any(eval_condition(c, features) for c in cond["any_of"])
    op = cond["op"]
    value = features.get(cond["feature"])
    if value is None:
        return False
    ops = {">=": lambda a,b: a>=b, ">": lambda a,b: a>b, "<=": lambda a,b: a<=b,
           "<": lambda a,b: a<b, "==": lambda a,b: a==b, "!=": lambda a,b: a!=b}
    return ops[op](value, cond["value"])

def main():
    with open(ROOT / "src/risk_engine/rules.yaml") as f:
        cfg = yaml.safe_load(f)
    
    rules = cfg["rules"]
    
    # Count per-rule FP and fraud
    fp_by_rule = {}
    fraud_by_rule = {}
    total_fp = 0
    total_fraud = 0
    
    with open(ROOT / "data/transactions.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse features
            features = {}
            for k, v in row.items():
                if k == "event_id":
                    continue
                try:
                    features[k] = float(v)
                except ValueError:
                    features[k] = v
            
            label = int(features.pop("label", 0))
            
            for rule in rules:
                rid = rule["id"]
                try:
                    fired = eval_condition(rule["condition"], features)
                except Exception:
                    fired = False
                
                if fired:
                    if label == 0:
                        fp_by_rule[rid] = fp_by_rule.get(rid, 0) + 1
                    else:
                        fraud_by_rule[rid] = fraud_by_rule.get(rid, 0) + 1
            
            if label == 0:
                total_fp += 1
            else:
                total_fraud += 1
    
    # Print analysis
    print("=" * 75)
    print("RULE FALSE POSITIVE ANALYSIS")
    print("=" * 75)
    print(f"Total: {total_fp} legit, {total_fraud} fraud")
    print()
    print(f"{'Rule':<35} {'FP':>5} {'Fraud':>6} {'FP/Fraud':>9} {'Severity':>9}")
    print("-" * 75)
    
    # Sort by FP count descending
    all_rules = set(list(fp_by_rule.keys()) + list(fraud_by_rule.keys()))
    data = []
    for rid in all_rules:
        fp = fp_by_rule.get(rid, 0)
        fraud = fraud_by_rule.get(rid, 0)
        severity = next((r["severity"] for r in rules if r["id"] == rid), 0)
        ratio = fp / max(fraud, 1)
        data.append((rid, fp, fraud, ratio, severity))
    
    data.sort(key=lambda x: -x[1])
    
    for rid, fp, fraud, ratio, severity in data:
        marker = " *** FP HEAVY" if ratio > 20 and fp > 50 else ""
        print(f"{rid:<35} {fp:>5} {fraud:>6} {ratio:>9.1f} {severity:>9.2f}{marker}")
    
    print()
    print("RECOMMENDATIONS:")
    print("-" * 75)
    for rid, fp, fraud, ratio, severity in data[:5]:
        if ratio > 20 and fp > 50:
            print(f"  {rid}: {fp} FPs, {fraud} fraud catches")
            print(f"    -> Consider raising threshold or lowering severity")
        elif ratio > 10 and fp > 30:
            print(f"  {rid}: {fp} FPs, {fraud} fraud catches")
            print(f"    -> Consider raising threshold slightly")

if __name__ == "__main__":
    main()
