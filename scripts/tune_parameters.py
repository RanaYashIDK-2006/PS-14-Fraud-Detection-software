#!/usr/bin/env python3
"""Sweep severity_scale values to find optimal FPR/recall tradeoff."""

import subprocess
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_sim(scale: float) -> dict:
    """Run rules simulator with given severity_scale and parse output."""
    result = subprocess.run(
        [sys.executable, "scripts/simulate_rules.py",
         "--severity-scale", str(scale)],
        cwd=str(ROOT),
        capture_output=True, text=True, timeout=120
    )
    output = result.stdout
    
    # Parse baseline output
    data = {}
    for line in output.splitlines():
        line = line.strip()
        if "recall (fraud)" in line:
            m = re.search(r":\s+([\d.]+)\s+\((\d+) of (\d+)", line)
            if m:
                data["recall"] = float(m.group(1))
                data["fraud_caught"] = int(m.group(2))
                data["fraud_total"] = int(m.group(3))
        elif "fpr (legit)" in line:
            m = re.search(r":\s+([\d.]+)", line)
            if m:
                data["fpr"] = float(m.group(1))
        elif "challenge rate" in line:
            m = re.search(r":\s+([\d.]+)", line)
            if m:
                data["challenge_rate"] = float(m.group(1))
        elif "avg score" in line:
            m = re.search(r"fraud\s+([\d.]+)\s*/\s*legit\s+([\d.]+)", line)
            if m:
                data["avg_fraud_score"] = float(m.group(1))
                data["avg_legit_score"] = float(m.group(2))
        elif "decision mix" in line:
            m = re.search(r"(\d+)\s+allow\s*/\s*(\d+)\s+step-up\s*/\s*(\d+)\s+verify", line)
            if m:
                data["allow"] = int(m.group(1))
                data["step_up"] = int(m.group(2))
                data["verify"] = int(m.group(3))
    
    return data

def main():
    print("=" * 70)
    print("SEVERITY_SCALE SWEEP — Optimal FPR/Recall Tradeoff")
    print("=" * 70)
    print()
    
    # Sweep range: 0.25 to 0.65 in 0.05 steps
    scales = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    
    results = []
    for scale in scales:
        print(f"Testing severity_scale={scale:.2f}...", end=" ", flush=True)
        data = run_sim(scale)
        if data:
            data["scale"] = scale
            results.append(data)
            print(f"recall={data.get('recall',0):.3f}  FPR={data.get('fpr',0):.4f}  "
                  f"challenge={data.get('challenge_rate',0):.3f}  "
                  f"fraud_avg={data.get('avg_fraud_score',0):.1f}  "
                  f"legit_avg={data.get('avg_legit_score',0):.1f}")
        else:
            print("FAILED")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scale':>6} {'Recall':>8} {'FPR':>8} {'Challenge':>10} {'F/L Score':>12} {'FP':>6}")
    print("-" * 70)
    
    total_events = 9799
    total_fraud = 181
    
    for r in results:
        recall = r.get("recall", 0)
        fpr = r.get("fpr", 0)
        challenge = r.get("challenge_rate", 0)
        fraud_avg = r.get("avg_fraud_score", 0)
        legit_avg = r.get("avg_legit_score", 0)
        # Estimate FPs: FPR * total_legit
        total_legit = total_events - total_fraud
        fp_est = int(fpr * total_legit)
        
        marker = ""
        if r["scale"] == 0.45:
            marker = " <-- CURRENT"
        
        print(f"{r['scale']:>6.2f} {recall:>8.3f} {fpr:>8.4f} {challenge:>10.3f} "
              f"{fraud_avg:>5.1f}/{legit_avg:<5.1f} {fp_est:>6}{marker}")
    
    print()
    
    # Find best balance point
    if results:
        # Optimize for: maximize recall while minimizing FPR
        # Score = recall - 0.5 * FPR (weighted by cost ratio)
        best = None
        best_score = -1
        for r in results:
            recall = r.get("recall", 0)
            fpr = r.get("fpr", 0)
            # Cost-weighted: FN is 20x more costly than FP
            score = recall * 20 - fpr * 100  # normalize
            if score > best_score:
                best_score = score
                best = r
        
        if best:
            print(f"OPTIMAL: severity_scale={best['scale']:.2f}")
            print(f"  Recall: {best.get('recall',0):.3f} ({best.get('fraud_caught',0)}/{best.get('fraud_total',0)} fraud caught)")
            print(f"  FPR: {best.get('fpr',0):.4f}")
            print(f"  Challenge rate: {best.get('challenge_rate',0):.3f}")
            print(f"  Avg scores: fraud={best.get('avg_fraud_score',0):.1f} / legit={best.get('avg_legit_score',0):.1f}")
            print(f"  Decision mix: {best.get('allow',0)} allow / {best.get('step_up',0)} step-up / {best.get('verify',0)} verify")

if __name__ == "__main__":
    main()
