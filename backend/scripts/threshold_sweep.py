#!/usr/bin/env python3
"""
Threshold Sweep — Evaluate FPR/Recall/Precision at thresholds 0.1, 0.25, 0.5, 0.75, 1.0
for ULB (creditcard.csv) and Altman (credit_card_transactions-ibm_v2.csv).

Measures actual FPR at each decision point and finds the sweet spot.
"""
import sys, time, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import RobustScaler

# ─── Thresholds to evaluate ───
THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 1.0]

# ─── Helpers ───
def metrics_at_threshold(y_true, y_prob, thr):
    """Compute all metrics at a given probability threshold."""
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total_neg = int((y_true == 0).sum())
    total_pos = int((y_true == 1).sum())
    
    fpr = fp / max(total_neg, 1)
    fnr = fn / max(total_pos, 1)
    tpr = tp / max(total_pos, 1)  # recall
    tnr = tn / max(total_neg, 1)  # specificity
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-12)
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    
    return {
        "threshold": thr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "recall": round(tpr, 6),
        "specificity": round(tnr, 6),
        "precision": round(prec, 6),
        "f1": round(f1, 6),
        "accuracy": round(acc, 6),
        "total_pos": total_pos,
        "total_neg": total_neg,
    }


# ══════════════════════════════════════════════════════════════════════
# ULB Dataset
# ══════════════════════════════════════════════════════════════════════
def eval_ulb():
    """Evaluate ULB creditcard dataset with production models."""
    print("=" * 80)
    print("ULB DATASET — Threshold Sweep")
    print("=" * 80)
    
    t0 = time.time()
    df = pd.read_csv("data/creditcard.csv")
    print(f"  Loaded: {len(df):,} rows, {df['Class'].sum():,} fraud ({df['Class'].mean()*100:.3f}%)")
    
    y = df["Class"].values
    feat_cols = [c for c in df.columns if c != "Class"]
    X = df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Scale
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    
    # Split 80/20 stratified
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"  Train: {len(Xtr):,} | Test: {len(Xte):,}")
    print(f"  Test fraud: {yte.sum():,}")
    
    # Train XGBoost
    import xgboost as xgb
    spw = (len(ytr) - int(ytr.sum())) / max(int(ytr.sum()), 1)
    
    print("\n  Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, gamma=2,
        min_child_weight=5, scale_pos_weight=min(spw, 100),
        tree_method="hist", eval_metric="auc",
        random_state=42, n_jobs=4
    )
    xgb_model.fit(Xtr, ytr, verbose=False)
    
    # Train LightGBM
    import lightgbm as lgb
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
        scale_pos_weight=min(spw, 100),
        random_state=42, n_jobs=4, verbose=-1
    )
    lgb_model.fit(Xtr, ytr)
    
    # Ensemble (50/50)
    p_xgb = xgb_model.predict_proba(Xte)[:, 1]
    p_lgb = lgb_model.predict_proba(Xte)[:, 1]
    y_prob = 0.5 * p_xgb + 0.5 * p_lgb
    
    # Overall AUC
    auc = roc_auc_score(yte, y_prob)
    
    # Threshold sweep
    print(f"\n  Overall ROC-AUC: {auc:.4f}")
    print(f"\n  {'Threshold':>10} | {'FPR':>8} | {'Recall':>8} | {'Prec':>8} | {'F1':>8} | {'Accuracy':>8} | {'TP':>6} {'FP':>6} {'TN':>8} {'FN':>4}")
    print("  " + "-" * 100)
    
    results = []
    for thr in THRESHOLDS:
        m = metrics_at_threshold(yte, y_prob, thr)
        results.append(m)
        print(f"  {thr:>10.2f} | {m['fpr']:>7.4f} | {m['recall']:>7.4f} | {m['precision']:>7.4f} | {m['f1']:>7.4f} | {m['accuracy']:>7.4f} | {m['tp']:>6} {m['fp']:>6} {m['tn']:>8} {m['fn']:>4}")
    
    # Find best threshold for FPR < 0.01 with max recall
    best_fpr1 = max((r for r in results if r["fpr"] <= 0.01), key=lambda r: r["recall"], default=None)
    # Best threshold for FPR < 0.001
    best_fpr01 = max((r for r in results if r["fpr"] <= 0.001), key=lambda r: r["recall"], default=None)
    # Best F1
    best_f1 = max(results, key=lambda r: r["f1"])
    # Best recall where FPR <= 0.005
    best_balanced = max((r for r in results if r["fpr"] <= 0.005), key=lambda r: r["recall"], default=None)
    
    print(f"\n  === Summary ===")
    if best_fpr1:
        print(f"  Best @ FPR ≤ 1%:    thr={best_fpr1['threshold']:.2f}  recall={best_fpr1['recall']:.4f}  prec={best_fpr1['precision']:.4f}")
    if best_fpr01:
        print(f"  Best @ FPR ≤ 0.1%:  thr={best_fpr01['threshold']:.2f}  recall={best_fpr01['recall']:.4f}  prec={best_fpr01['precision']:.4f}")
    if best_balanced:
        print(f"  Best @ FPR ≤ 0.5%:  thr={best_balanced['threshold']:.2f}  recall={best_balanced['recall']:.4f}  prec={best_balanced['precision']:.4f}")
    print(f"  Best F1:            thr={best_f1['threshold']:.2f}  f1={best_f1['f1']:.4f}  prec={best_f1['precision']:.4f}")
    
    # Additional fine-grained sweep around optimal
    print(f"\n  Fine-grained sweep (0.05 steps):")
    print(f"  {'Threshold':>10} | {'FPR':>8} | {'Recall':>8} | {'Prec':>8} | {'F1':>8}")
    print("  " + "-" * 60)
    fine_thr = np.arange(0.05, 1.0, 0.05)
    fine_results = []
    for thr in fine_thr:
        m = metrics_at_threshold(yte, y_prob, thr)
        fine_results.append(m)
        print(f"  {thr:>10.2f} | {m['fpr']:>7.4f} | {m['recall']:>7.4f} | {m['precision']:>7.4f} | {m['f1']:>7.4f}")
    
    elapsed = time.time() - t0
    
    # Store results
    out = {
        "dataset": "ulb_creditcard",
        "n_rows": len(df),
        "n_test": len(yte),
        "fraud_test": int(yte.sum()),
        "roc_auc": round(auc, 6),
        "sweep_results": results,
        "fine_sweep": fine_results,
        "elapsed_seconds": round(elapsed, 1),
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/threshold_sweep_ulb.json", "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"\n  Saved: reports/threshold_sweep_ulb.json ({elapsed:.1f}s)")
    return out


# ══════════════════════════════════════════════════════════════════════
# Altman Dataset (sampled for speed)
# ══════════════════════════════════════════════════════════════════════
def eval_altman():
    """Evaluate Altman dataset with merchant/city/card features."""
    print("\n" + "=" * 80)
    print("ALTMAN DATASET — Threshold Sweep")
    print("=" * 80)
    
    t0 = time.time()
    
    # Read in chunks, keep all fraud, sample legit
    print("  Loading Altman (streaming, ~200K sample)...")
    rng = np.random.RandomState(42)
    rows_all = []
    n_fraud = 0
    n_legit = 0
    n_legit_target = 150_000  # target legit rows
    
    for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
                              low_memory=False, chunksize=200_000):
        # Parse label
        if "Is Fraud?" in chunk.columns:
            chunk["label"] = (chunk["Is Fraud?"].astype(str).str.strip() == "Yes").astype(int)
        elif "label" not in chunk.columns:
            chunk["label"] = 0
        
        fraud_rows = chunk[chunk["label"] == 1]
        legit_rows = chunk[chunk["label"] == 0]
        
        n_fraud += len(fraud_rows)
        rows_all.append(fraud_rows)
        
        if n_legit < n_legit_target:
            n_sample = min(len(legit_rows), n_legit_target - n_legit)
            if n_sample > 0:
                rows_all.append(legit_rows.sample(n=n_sample, random_state=rng))
                n_legit += n_sample
        
        if n_legit >= n_legit_target:
            # skip remaining legit but still grab fraud
            pass
    
    df = pd.concat(rows_all, ignore_index=True)
    print(f"  Sampled: {len(df):,} rows ({n_fraud:,} fraud, {n_legit:,} legit)")
    
    # Feature engineering
    print("  Engineering features...")
    gm = df["label"].mean()
    
    # Basic time features
    if "hr" not in df.columns and "Time" in df.columns:
        # Time is a string like '12:34 PM' — parse it
        def _parse_hr(t):
            try:
                s = str(t).strip()
                parts = s.replace('.', ':').split(':')
                h = int(parts[0])
                if 'pm' in s.lower() and h != 12:
                    h += 12
                elif 'am' in s.lower() and h == 12:
                    h = 0
                return h
            except:
                return 12
        df["hr"] = df["Time"].apply(_parse_hr)
    elif "hr" not in df.columns:
        # Use Month/Day for basic time proxy
        df["hr"] = 12  # default noon
    
    # Amount features — Altman stores Amount as string with $ sign
    amt_col = "Amount" if "Amount" in df.columns else "amt"
    if amt_col in df.columns:
        df["amt"] = pd.to_numeric(df[amt_col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False), errors='coerce').fillna(0)
    else:
        amt_col = df.select_dtypes(include=[np.number]).columns[0]
        df["amt"] = df[amt_col].astype(float)
    df["log_amt"] = np.log1p(df["amt"])
    df["amt_sq"] = df["amt"] ** 2
    df["very_high_amt"] = (df["amt"] > 1000).astype(int)
    df["high_amt"] = (df["amt"] > 500).astype(int)
    
    # Merchant features
    if "Merchant Name" in df.columns:
        merch_fraud = df.groupby("Merchant Name")["label"].agg(["mean", "count"])
        merch_fraud["merch_fr"] = (merch_fraud["mean"] * merch_fraud["count"] + gm * 50) / (merch_fraud["count"] + 50)
        df["merch_fraud_rate"] = df["Merchant Name"].map(merch_fraud["merch_fr"]).fillna(gm)
        df["merch_tx_count"] = df.groupby("Merchant Name").cumcount() + 1
    else:
        df["merch_fraud_rate"] = gm
        df["merch_tx_count"] = 1
    
    # City features
    if "Merchant City" in df.columns:
        city_fraud = df.groupby("Merchant City")["label"].agg(["mean", "count"])
        city_fraud["city_fr"] = (city_fraud["mean"] * city_fraud["count"] + gm * 50) / (city_fraud["count"] + 50)
        df["city_fraud_rate"] = df["Merchant City"].map(city_fraud["city_fr"]).fillna(gm)
    else:
        df["city_fraud_rate"] = gm
    
    # MCC
    if "MCC" in df.columns:
        df["mcc_n"] = pd.to_numeric(df["MCC"], errors="coerce").fillna(0)
    else:
        df["mcc_n"] = 0
    
    # Chip/Online
    if "Use Chip" in df.columns:
        df["chip"] = (df["Use Chip"].astype(str).str.contains("Swipe|Chip", case=False, na=False)).astype(int)
        df["is_online"] = (df["Use Chip"].astype(str).str.contains("Online", case=False, na=False)).astype(int)
    else:
        df["chip"] = 0
        df["is_online"] = 0
    
    # Zip/State
    if "Zip" in df.columns:
        df["has_zip"] = df["Zip"].notna().astype(int)
    else:
        df["has_zip"] = 0
    if "Merchant State" in df.columns:
        df["has_state"] = df["Merchant State"].notna().astype(int)
    else:
        df["has_state"] = 0
    
    # Time features
    if "hr" in df.columns:
        hr = df["hr"].fillna(12).astype(float)
        df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
        df["is_business_hours"] = ((hr >= 9) & (hr <= 17)).astype(int)
    else:
        df["hour_cos"] = 0.0
        df["is_business_hours"] = 1
    
    # Interactions
    df["amt_x_mcc"] = df["amt"] * df["mcc_n"]
    df["amt_x_online"] = df["amt"] * df["is_online"]
    
    # Card-level
    if "Card" in df.columns and "User" in df.columns:
        df["card_key"] = df["User"].astype(str) + "_" + df["Card"].astype(str)
        df["card_tx_count"] = df.groupby("card_key").cumcount() + 1
    else:
        df["card_tx_count"] = 1
    
    # Final feature set (15 lean features)
    lean_features = [
        "log_amt", "amt_sq", "hour_cos", "is_business_hours",
        "chip", "is_online", "mcc_n", "has_zip", "has_state",
        "merch_tx_count", "merch_fraud_rate", "city_fraud_rate",
        "very_high_amt", "amt_x_mcc", "amt_x_online"
    ]
    
    # Add card_tx_count if available
    if "card_tx_count" in df.columns:
        lean_features.append("card_tx_count")
    
    # Ensure all features exist
    for f in lean_features:
        if f not in df.columns:
            df[f] = 0
    
    X = np.nan_to_num(df[lean_features].values.astype(np.float32), nan=0, posinf=100, neginf=-100)
    y = df["label"].values.astype(int)
    
    print(f"  Features: {len(lean_features)} | X shape: {X.shape}")
    print(f"  Fraud rate: {y.mean()*100:.4f}%")
    
    # Scale
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    
    # Split 80/20 stratified
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"  Train: {len(Xtr):,} | Test: {len(Xte):,}")
    
    # Train XGBoost
    import xgboost as xgb
    spw = (len(ytr) - int(ytr.sum())) / max(int(ytr.sum()), 1)
    
    print("\n  Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, gamma=2,
        min_child_weight=5, scale_pos_weight=min(spw, 100),
        tree_method="hist", eval_metric="auc",
        random_state=42, n_jobs=4
    )
    xgb_model.fit(Xtr, ytr, verbose=False)
    
    # Train LightGBM
    import lightgbm as lgb
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
        scale_pos_weight=min(spw, 100),
        random_state=42, n_jobs=4, verbose=-1
    )
    lgb_model.fit(Xtr, ytr)
    
    # Ensemble
    p_xgb = xgb_model.predict_proba(Xte)[:, 1]
    p_lgb = lgb_model.predict_proba(Xte)[:, 1]
    y_prob = 0.5 * p_xgb + 0.5 * p_lgb
    
    auc = roc_auc_score(yte, y_prob)
    
    # Threshold sweep
    print(f"\n  Overall ROC-AUC: {auc:.4f}")
    print(f"\n  {'Threshold':>10} | {'FPR':>8} | {'Recall':>8} | {'Prec':>8} | {'F1':>8} | {'Accuracy':>8} | {'TP':>6} {'FP':>6} {'TN':>8} {'FN':>4}")
    print("  " + "-" * 100)
    
    results = []
    for thr in THRESHOLDS:
        m = metrics_at_threshold(yte, y_prob, thr)
        results.append(m)
        print(f"  {thr:>10.2f} | {m['fpr']:>7.4f} | {m['recall']:>7.4f} | {m['precision']:>7.4f} | {m['f1']:>7.4f} | {m['accuracy']:>7.4f} | {m['tp']:>6} {m['fp']:>6} {m['tn']:>8} {m['fn']:>4}")
    
    # Best thresholds
    best_fpr1 = max((r for r in results if r["fpr"] <= 0.01), key=lambda r: r["recall"], default=None)
    best_fpr01 = max((r for r in results if r["fpr"] <= 0.001), key=lambda r: r["recall"], default=None)
    best_f1 = max(results, key=lambda r: r["f1"])
    best_balanced = max((r for r in results if r["fpr"] <= 0.005), key=lambda r: r["recall"], default=None)
    
    print(f"\n  === Summary ===")
    if best_fpr1:
        print(f"  Best @ FPR ≤ 1%:    thr={best_fpr1['threshold']:.2f}  recall={best_fpr1['recall']:.4f}  prec={best_fpr1['precision']:.4f}")
    if best_fpr01:
        print(f"  Best @ FPR ≤ 0.1%:  thr={best_fpr01['threshold']:.2f}  recall={best_fpr01['recall']:.4f}  prec={best_fpr01['precision']:.4f}")
    if best_balanced:
        print(f"  Best @ FPR ≤ 0.5%:  thr={best_balanced['threshold']:.2f}  recall={best_balanced['recall']:.4f}  prec={best_balanced['precision']:.4f}")
    print(f"  Best F1:            thr={best_f1['threshold']:.2f}  f1={best_f1['f1']:.4f}  prec={best_f1['precision']:.4f}")
    
    # Fine-grained sweep
    print(f"\n  Fine-grained sweep (0.05 steps):")
    print(f"  {'Threshold':>10} | {'FPR':>8} | {'Recall':>8} | {'Prec':>8} | {'F1':>8}")
    print("  " + "-" * 60)
    fine_thr = np.arange(0.05, 1.0, 0.05)
    fine_results = []
    for thr in fine_thr:
        m = metrics_at_threshold(yte, y_prob, thr)
        fine_results.append(m)
        print(f"  {thr:>10.2f} | {m['fpr']:>7.4f} | {m['recall']:>7.4f} | {m['precision']:>7.4f} | {m['f1']:>7.4f}")
    
    elapsed = time.time() - t0
    
    out = {
        "dataset": "ibm_altman",
        "n_rows": len(df),
        "n_test": len(yte),
        "fraud_test": int(yte.sum()),
        "roc_auc": round(auc, 6),
        "features": lean_features,
        "sweep_results": results,
        "fine_sweep": fine_results,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open("reports/threshold_sweep_altman.json", "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"\n  Saved: reports/threshold_sweep_altman.json ({elapsed:.1f}s)")
    return out


# ══════════════════════════════════════════════════════════════════════
# Cross-dataset comparison
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("THRESHOLD SWEEP — FPR Analysis at 0.1, 0.25, 0.5, 0.75, 1.0")
    print("=" * 80)
    
    ulb = eval_ulb()
    alt = eval_altman()
    
    # Cross-dataset comparison table
    print("\n" + "=" * 80)
    print("CROSS-DATASET COMPARISON — FPR vs Recall at Each Threshold")
    print("=" * 80)
    print(f"  {'Threshold':>10} | {'ULB FPR':>10} {'ULB Recall':>11} {'ULB F1':>8} | {'Alt FPR':>10} {'Alt Recall':>11} {'Alt F1':>8}")
    print("  " + "-" * 90)
    
    ulb_map = {r["threshold"]: r for r in ulb["sweep_results"]}
    alt_map = {r["threshold"]: r for r in alt["sweep_results"]}
    
    for thr in THRESHOLDS:
        u = ulb_map.get(thr, {})
        a = alt_map.get(thr, {})
        print(f"  {thr:>10.2f} | {u.get('fpr',0):>9.4f} {u.get('recall',0):>10.4f} {u.get('f1',0):>7.4f} | {a.get('fpr',0):>9.4f} {a.get('recall',0):>10.4f} {a.get('f1',0):>7.4f}")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    for name, res in [("ULB", ulb), ("Altman", alt)]:
        sweep = res["sweep_results"]
        # Find threshold with FPR < 0.5% and best recall
        safe = [r for r in sweep if r["fpr"] <= 0.005]
        if safe:
            best = max(safe, key=lambda r: r["recall"])
            print(f"\n  {name}: Use threshold = {best['threshold']:.2f}")
            print(f"    FPR: {best['fpr']*100:.3f}% | Recall: {best['recall']*100:.1f}% | Precision: {best['precision']*100:.1f}% | F1: {best['f1']:.4f}")
            print(f"    False positives per 10K legit: {int(best['fp'] / max(best['total_neg'],1) * 10000)}")
            print(f"    Missed fraud: {best['fn']} out of {best['total_pos']}")
        else:
            print(f"\n  {name}: No threshold achieves FPR ≤ 0.5% — consider raising threshold further")
    
    # Save comparison
    comparison = {
        "ulb": ulb,
        "altman": alt,
    }
    with open("reports/threshold_sweep_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n  Saved: reports/threshold_sweep_comparison.json")
    print("  DONE")


if __name__ == "__main__":
    main()
