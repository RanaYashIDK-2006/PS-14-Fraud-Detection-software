#!/usr/bin/env python3
"""Test individual rule threshold adjustments to optimize FPR/recall."""

import subprocess
import sys
import re
import yaml
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_sim(rules_path: str) -> dict:
    """Run rules simulator and parse output."""
    result = subprocess.run(
        [sys.executable, "scripts/simulate_rules.py", "--rules", rules_path],
        cwd=str(ROOT),
        capture_output=True, text=True, timeout=120
    )
    output = result.stdout
    
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

def modify_rules(base_path: Path, modifications: dict) -> Path:
    """Create a modified rules.yaml and return path."""
    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    for rule_id, changes in modifications.items():
        for rule in cfg["rules"]:
            if rule["id"] == rule_id:
                for key, value in changes.items():
                    if key == "condition.feature":
                        rule["condition"]["feature"] = value
                    elif key == "condition.op":
                        rule["condition"]["op"] = value
                    elif key == "condition.value":
                        rule["condition"]["value"] = value
                    elif key == "severity":
                        rule["severity"] = value
                    else:
                        rule[key] = value
                break
    
    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=str(ROOT / "src" / "risk_engine"))
    yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return Path(tmp.name)

def main():
    base_rules = ROOT / "src" / "risk_engine" / "rules.yaml"
    
    print("=" * 70)
    print("RULE THRESHOLD TUNING")
    print("=" * 70)
    print()
    
    # Baseline
    print("BASELINE (current rules.yaml):")
    baseline = run_sim(str(base_rules))
    print(f"  Recall: {baseline.get('recall',0):.3f}  FPR: {baseline.get('fpr',0):.4f}  "
          f"Challenge: {baseline.get('challenge_rate',0):.3f}")
    print()
    
    # Test 1: Raise severity_scale to 0.30 (still same recall, lower FPR)
    print("TEST 1: severity_scale=0.30 (reduced from 0.45)")
    with open(base_rules, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["severity_scale"] = 0.30
    tmp1 = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=str(ROOT / "src" / "risk_engine"))
    yaml.dump(cfg, tmp1, default_flow_style=False, sort_keys=False)
    tmp1.close()
    r1 = run_sim(str(tmp1.name))
    print(f"  Recall: {r1.get('recall',0):.3f}  FPR: {r1.get('fpr',0):.4f}  "
          f"Challenge: {r1.get('challenge_rate',0):.3f}")
    Path(tmp1.name).unlink()
    print()
    
    # Test 2: Raise new_device rule severity (catch more fraud at lower scale)
    print("TEST 2: severity_scale=0.30 + new_device severity=0.45 (was 0.35)")
    mods2 = {"severity_scale": 0.30}
    tmp2 = modify_rules(base_rules, {"RULE_NEW_DEVICE": {"severity": 0.45}})
    # Also set severity_scale
    with open(tmp2, "r") as f:
        cfg2 = yaml.safe_load(f)
    cfg2["severity_scale"] = 0.30
    with open(tmp2, "w") as f:
        yaml.dump(cfg2, f, default_flow_style=False, sort_keys=False)
    r2 = run_sim(str(tmp2))
    print(f"  Recall: {r2.get('recall',0):.3f}  FPR: {r2.get('fpr',0):.4f}  "
          f"Challenge: {r2.get('challenge_rate',0):.3f}")
    tmp2.unlink()
    print()
    
    # Test 3: Raise chameleon rules to catch more missed fraud
    print("TEST 3: severity_scale=0.30 + chameleon rules boosted")
    tmp3 = modify_rules(base_rules, {
        "RULE_CHAMELEON_MULTI_FLAG": {"severity": 0.65},
        "RULE_CHAMELEON_NIGHT_STEALTH": {"severity": 0.60},
        "RULE_CHAMELEON_MICRO_TEST": {"severity": 0.55},
    })
    with open(tmp3, "r") as f:
        cfg3 = yaml.safe_load(f)
    cfg3["severity_scale"] = 0.30
    with open(tmp3, "w") as f:
        yaml.dump(cfg3, f, default_flow_style=False, sort_keys=False)
    r3 = run_sim(str(tmp3))
    print(f"  Recall: {r3.get('recall',0):.3f}  FPR: {r3.get('fpr',0):.4f}  "
          f"Challenge: {r3.get('challenge_rate',0):.3f}")
    tmp3.unlink()
    print()
    
    # Test 4: Raise ATO_SIGNATURE severity (account takeover is high-value)
    print("TEST 4: severity_scale=0.30 + ATO severity=1.0 (critical, was 0.90)")
    tmp4 = modify_rules(base_rules, {"RULE_ATO_SIGNATURE": {"severity": 1.0}})
    with open(tmp4, "r") as f:
        cfg4 = yaml.safe_load(f)
    cfg4["severity_scale"] = 0.30
    with open(tmp4, "w") as f:
        yaml.dump(cfg4, f, default_flow_style=False, sort_keys=False)
    r4 = run_sim(str(tmp4))
    print(f"  Recall: {r4.get('recall',0):.3f}  FPR: {r4.get('fpr',0):.4f}  "
          f"Challenge: {r4.get('challenge_rate',0):.3f}")
    tmp4.unlink()
    print()
    
    # Test 5: Combined best
    print("TEST 5: COMBINED — scale=0.30 + boosted chameleons + new_device=0.45")
    tmp5 = modify_rules(base_rules, {
        "RULE_NEW_DEVICE": {"severity": 0.45},
        "RULE_CHAMELEON_MULTI_FLAG": {"severity": 0.65},
        "RULE_CHAMELEON_NIGHT_STEALTH": {"severity": 0.60},
        "RULE_CHAMELEON_MICRO_TEST": {"severity": 0.55},
    })
    with open(tmp5, "r") as f:
        cfg5 = yaml.safe_load(f)
    cfg5["severity_scale"] = 0.30
    with open(tmp5, "w") as f:
        yaml.dump(cfg5, f, default_flow_style=False, sort_keys=False)
    r5 = run_sim(str(tmp5))
    print(f"  Recall: {r5.get('recall',0):.3f}  FPR: {r5.get('fpr',0):.4f}  "
          f"Challenge: {r5.get('challenge_rate',0):.3f}")
    tmp5.unlink()
    print()
    
    # Summary
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    total_legit = 9799 - 181
    tests = [
        ("Baseline (current)", baseline),
        ("scale=0.30", r1),
        ("scale=0.30 + new_device=0.45", r2),
        ("scale=0.30 + chameleon boost", r3),
        ("scale=0.30 + ATO=1.0", r4),
        ("COMBINED best", r5),
    ]
    
    print(f"{'Config':<30} {'Recall':>8} {'FPR':>8} {'FPs':>6} {'Challenge':>10}")
    print("-" * 70)
    for name, r in tests:
        recall = r.get("recall", 0)
        fpr = r.get("fpr", 0)
        fp = int(fpr * total_legit)
        challenge = r.get("challenge_rate", 0)
        print(f"{name:<30} {recall:>8.3f} {fpr:>8.4f} {fp:>6} {challenge:>10.3f}")
    
    print()
    # Find the best combined
    best_name, best_r = max(tests, key=lambda x: x[1].get("recall", 0) * 20 - x[1].get("fpr", 0) * 100)
    print(f"RECOMMENDED: {best_name}")
    print(f"  Recall: {best_r.get('recall',0):.3f} ({best_r.get('fraud_caught',0)}/{best_r.get('fraud_total',0)})")
    print(f"  FPR: {best_r.get('fpr',0):.4f} ({int(best_r.get('fpr',0) * total_legit)} false positives)")
    print(f"  Challenge rate: {best_r.get('challenge_rate',0):.3f}")

if __name__ == "__main__":
    main()
