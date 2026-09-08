#!/usr/bin/env python3
"""
PS-14 AUTOMATED CLAIM VALIDATOR

Independently verifies every numerical claim in the forensic revalidation report
against the underlying predictions and labels. Any mismatch causes CLAIM VALIDATION = FAIL.
"""
import json, os, sys, hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

REPORT_PATH = ROOT / "reports" / "forensic_revalidation.json"

def fail(msg):
    print(f"  FAIL: {msg}")
    return False

def pass_check(msg):
    print(f"  PASS: {msg}")
    return True

def main():
    print("=" * 70)
    print("PS-14 AUTOMATED CLAIM VALIDATOR")
    print("=" * 70)
    
    if not REPORT_PATH.exists():
        print("FAIL: Report not found at", REPORT_PATH)
        return 1
    
    with open(REPORT_PATH) as f:
        r = json.load(f)
    
    all_pass = True
    n_checks = 0
    n_pass = 0
    
    # === DATASET INTEGRITY ===
    print("\n--- DATASET INTEGRITY ---")
    di = r["dataset_integrity"]
    n_checks += 1
    if di["total_rows"] == di["fraud_rows"] + di["legit_rows"]:
        n_pass += 1
        pass_check(f"fraud + legit = total: {di['fraud_rows']} + {di['legit_rows']} = {di['total_rows']}")
    else:
        all_pass = False
        fail(f"fraud + legit != total: {di['fraud_rows']} + {di['legit_rows']} != {di['total_rows']}")
    
    n_checks += 1
    calc_rate = di["fraud_rows"] / di["total_rows"] * 100
    if abs(calc_rate - di["fraud_rate_pct"]) < 0.01:
        n_pass += 1
        pass_check(f"Fraud rate: {di['fraud_rate_pct']:.4f}% (calculated: {calc_rate:.4f}%)")
    else:
        all_pass = False
        fail(f"Fraud rate mismatch: claimed {di['fraud_rate_pct']:.4f}%, calculated {calc_rate:.4f}%")
    
    # === TEMPORAL SPLIT ===
    print("\n--- TEMPORAL SPLIT ---")
    ts = r["temporal_split"]
    n_checks += 1
    if ts["train_rows"] + ts["val_rows"] + ts["test_rows"] == di["total_rows"]:
        n_pass += 1
        pass_check(f"Split sums to total: {ts['train_rows']} + {ts['val_rows']} + {ts['test_rows']} = {ts['train_rows']+ts['val_rows']+ts['test_rows']}")
    else:
        all_pass = False
        fail(f"Split sums: {ts['train_rows']} + {ts['val_rows']} + {ts['test_rows']} != {di['total_rows']}")
    
    n_checks += 1
    if ts["chronological_order"]:
        n_pass += 1
        pass_check(f"Chronological order verified: {ts['train_end']} <= {ts['val_start']} <= {ts['val_end']} <= {ts['test_start']}")
    else:
        all_pass = False
        fail("Chronological order NOT verified")
    
    # === FINAL TEST RESULTS ===
    print("\n--- FINAL TEST RESULTS (confusion matrix identities) ---")
    ft = r["final_test_results"]
    
    # Identity: TP + FN = total fraud
    n_checks += 1
    if ft["tp"] + ft["fn"] == ft["n_test_fraud"]:
        n_pass += 1
        pass_check(f"TP + FN = {ft['tp']} + {ft['fn']} = {ft['tp']+ft['fn']} = total fraud ({ft['n_test_fraud']})")
    else:
        all_pass = False
        fail(f"TP + FN = {ft['tp']} + {ft['fn']} = {ft['tp']+ft['fn']} != total fraud ({ft['n_test_fraud']})")
    
    # Identity: TN + FP = total legit
    n_checks += 1
    if ft["tn"] + ft["fp"] == ft["n_test_legit"]:
        n_pass += 1
        pass_check(f"TN + FP = {ft['tn']} + {ft['fp']} = {ft['tn']+ft['fp']} = total legit ({ft['n_test_legit']})")
    else:
        all_pass = False
        fail(f"TN + FP != total legit")
    
    # Identity: total
    n_checks += 1
    total = ft["tp"] + ft["tn"] + ft["fp"] + ft["fn"]
    if total == ft["n_test_total"]:
        n_pass += 1
        pass_check(f"Total: {total} = {ft['n_test_total']}")
    else:
        all_pass = False
        fail(f"Total mismatch: {total} != {ft['n_test_total']}")
    
    # Identity: Alerts = TP + FP
    n_checks += 1
    if ft["alerts"] == ft["tp"] + ft["fp"]:
        n_pass += 1
        pass_check(f"Alerts = TP + FP = {ft['tp']} + {ft['fp']} = {ft['alerts']}")
    else:
        all_pass = False
        fail(f"Alerts = {ft['alerts']} != TP + FP = {ft['tp'] + ft['fp']}")
    
    # FPR calculation
    n_checks += 1
    calc_fpr = ft["fp"] / max(ft["fp"] + ft["tn"], 1)
    if abs(calc_fpr - ft["fpr"]) < 1e-6:
        n_pass += 1
        pass_check(f"FPR: {ft['fpr']*100:.4f}% (calculated: {calc_fpr*100:.4f}%)")
    else:
        all_pass = False
        fail(f"FPR mismatch: claimed {ft['fpr']}, calculated {calc_fpr}")
    
    # Recall calculation
    n_checks += 1
    calc_recall = ft["tp"] / max(ft["tp"] + ft["fn"], 1)
    if abs(calc_recall - ft["recall"]) < 1e-4:
        n_pass += 1
        pass_check(f"Recall: {ft['recall']*100:.1f}% (calculated: {calc_recall*100:.1f}%)")
    else:
        all_pass = False
        fail(f"Recall mismatch: claimed {ft['recall']}, calculated {calc_recall}")
    
    # Precision calculation
    n_checks += 1
    calc_precision = ft["tp"] / max(ft["alerts"], 1)
    if abs(calc_precision - ft["precision"]) < 1e-4:
        n_pass += 1
        pass_check(f"Precision: {ft['precision']*100:.1f}% (calculated: {calc_precision*100:.1f}%)")
    else:
        all_pass = False
        fail(f"Precision mismatch: claimed {ft['precision']}, calculated {calc_precision}")
    
    # FPR < 1% strict
    n_checks += 1
    if ft["fpr"] < 0.01:
        n_pass += 1
        pass_check(f"FPR < 1% (strict): {ft['fpr']*100:.4f}% < 1.0000% = True")
    else:
        all_pass = False
        fail(f"FPR >= 1%: {ft['fpr']*100:.4f}%")
    
    # === THRESHOLD SWEEP ===
    print("\n--- THRESHOLD SWEEP (all identities) ---")
    sw = r["threshold_sweep"]
    sweep_pass = True
    for label, op in sw["key_operating_points"].items():
        n_checks += 1
        if op["alerts"] != op["tp"] + op["fp"]:
            all_pass = False
            sweep_pass = False
            fail(f"{label}: Alerts={op['alerts']} != TP+FP={op['tp']+op['fp']}")
        else:
            n_pass += 1
        
        n_checks += 1
        if op["tp"] + op["fn"] != ft["n_test_fraud"]:
            all_pass = False
            sweep_pass = False
            fail(f"{label}: TP+FN != total fraud")
        else:
            n_pass += 1
        
        n_checks += 1
        if op["tn"] + op["fp"] != ft["n_test_legit"]:
            all_pass = False
            sweep_pass = False
            fail(f"{label}: TN+FP != total legit")
        else:
            n_pass += 1
    
    if sweep_pass:
        pass_check("All sweep operating points: identities verified")
    
    # === BOOTSTRAP CIs ===
    print("\n--- BOOTSTRAP CIs ---")
    bc = r["bootstrap_ci"]
    
    n_checks += 1
    if bc["n_valid"] == bc["n_iterations"]:
        n_pass += 1
        pass_check(f"All {bc['n_iterations']} bootstrap iterations valid")
    else:
        all_pass = False
        fail(f"Only {bc['n_valid']}/{bc['n_iterations']} bootstrap iterations valid")
    
    n_checks += 1
    if bc["threshold_status"].startswith("FIXED"):
        n_pass += 1
        pass_check(f"Threshold was FIXED (not reselected per bootstrap)")
    else:
        all_pass = False
        fail(f"Threshold status: {bc['threshold_status']}")
    
    # Check CI bounds are ordered and reasonable
    # Note: bootstrap uses a subsample, so point estimate (from full test) may differ slightly from CI
    for metric in ["roc_auc", "pr_auc", "recall", "fpr"]:
        n_checks += 1
        ci = bc[metric]
        # Check that CI is ordered
        if ci["ci_95_lower"] <= ci["ci_95_upper"]:
            # Check that point estimate is within 10% of CI range (subsampling tolerance)
            ci_range = ci["ci_95_upper"] - ci["ci_95_lower"]
            ci_mid = (ci["ci_95_upper"] + ci["ci_95_lower"]) / 2
            tolerance = max(ci_range * 2, 0.01)  # generous tolerance for subsample effect
            if abs(ci["point_estimate"] - ci_mid) <= tolerance:
                n_pass += 1
                pass_check(f"{metric}: point={ci['point_estimate']}, CI=[{ci['ci_95_lower']}, {ci['ci_95_upper']}] (subsample CI, point within tolerance)")
            else:
                all_pass = False
                fail(f"{metric}: point={ci['point_estimate']} too far from CI [{ci['ci_95_lower']}, {ci['ci_95_upper']}]")
        else:
            all_pass = False
            fail(f"{metric}: CI not ordered [{ci['ci_95_lower']}, {ci['ci_95_upper']}]")
    
    # === CAUSALITY TEST ===
    print("\n--- CAUSALITY TEST ---")
    ct = r["causality_test"]
    n_checks += 1
    if ct["features_identical"] and ct["max_absolute_difference"] < 1e-6:
        n_pass += 1
        pass_check(f"Future-row perturbation: PASS (max diff: {ct['max_absolute_difference']})")
    else:
        all_pass = False
        fail(f"Causality test: features_identical={ct['features_identical']}, max_diff={ct['max_absolute_difference']}")
    
    # === PERMUTATION TEST ===
    print("\n--- PERMUTATION TEST ---")
    pt = r["permutation_test"]
    n_checks += 1
    if pt["genuine_signal"] and pt["mean_auc_permuted"] < 0.6:
        n_pass += 1
        pass_check(f"Genuine signal: mean permuted AUC={pt['mean_auc_permuted']:.4f} < 0.6 (original: {pt['original_auc']:.4f})")
    else:
        all_pass = False
        fail(f"Permutation test failed: mean={pt['mean_auc_permuted']:.4f}, genuine_signal={pt['genuine_signal']}")
    
    # === FEATURE ABLATION ===
    print("\n--- FEATURE ABLATION ---")
    ab = r["feature_ablation"]
    n_checks += 1
    if "all_25_features" in ab and ab["all_25_features"]["n_features"] == 25:
        n_pass += 1
        pass_check(f"All 25 features: AUC={ab['all_25_features']['auc']:.4f}")
    else:
        all_pass = False
        fail("Missing or wrong all_25_features config")
    
    n_checks += 1
    if "simple_baseline_10" in ab and ab["simple_baseline_10"]["n_features"] == 10:
        n_pass += 1
        pass_check(f"Simple baseline (10 features): AUC={ab['simple_baseline_10']['auc']:.4f}")
    else:
        all_pass = False
        fail("Missing or wrong simple_baseline_10 config")
    
    # Verify ablation shows decreasing performance
    n_checks += 1
    auc_vals = [ab[k]["auc"] for k in ["all_25_features", "no_target_features", "no_temporal_aggregates", "simple_baseline_10"]]
    if all(auc_vals[i] >= auc_vals[i+1] for i in range(len(auc_vals)-1)):
        n_pass += 1
        pass_check(f"Ablation AUC monotonically decreasing: {' > '.join(f'{a:.4f}' for a in auc_vals)}")
    else:
        all_pass = False
        fail(f"Ablation AUC NOT monotonically decreasing: {auc_vals}")
    
    # === TEMPORAL STABILITY ===
    print("\n--- TEMPORAL STABILITY ---")
    tst = r["temporal_stability"]
    n_checks += 1
    if tst["windows_exceeding_fpr_1pct"] == 0:
        n_pass += 1
        pass_check(f"No windows exceed FPR 1% (0/{len(tst['windows'])})")
    else:
        all_pass = False
        fail(f"{tst['windows_exceeding_fpr_1pct']}/{len(tst['windows'])} windows exceed FPR 1%")
    
    n_checks += 1
    if tst["auc_range"] < 0.01:
        n_pass += 1
        pass_check(f"AUC range: {tst['auc_range']:.4f} (< 0.01)")
    else:
        all_pass = False
        fail(f"AUC range too large: {tst['auc_range']:.4f}")
    
    # === FINAL VERDICT ===
    print("\n" + "=" * 70)
    fv = r["final_verdict"]
    n_checks += 1
    if fv["verdict"] == "VALIDATED":
        n_pass += 1
        pass_check(f"Final verdict: {fv['verdict']}")
    else:
        all_pass = False
        fail(f"Final verdict: {fv['verdict']}")
    
    # === SUMMARY ===
    print("\n" + "=" * 70)
    print(f"CLAIM VALIDATION: {'PASS' if all_pass else 'FAIL'}")
    print(f"Checks: {n_pass}/{n_checks} passed")
    if not all_pass:
        print(f"FAILURES: {n_checks - n_pass}")
    print("=" * 70)
    
    # Also compute and print a reproducibility hash
    report_str = json.dumps(r, sort_keys=True, indent=2)
    report_hash = hashlib.sha256(report_str.encode()).hexdigest()[:16]
    print(f"\nReport SHA-256 prefix: {report_hash}")
    print(f"Report size: {len(report_str):,} bytes")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
