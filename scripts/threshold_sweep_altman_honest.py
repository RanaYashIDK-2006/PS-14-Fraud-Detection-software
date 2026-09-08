#!/usr/bin/env python3
"""
Honest Altman threshold sweep with temporal split and target-leakage-free features.
The merchant_fraud_rate and city_fraud_rate features use label information from
the training set that leaks into the test set via random split. We use a proper
temporal split and compute fraud rates only from training data.
"""
import sys, time, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 1.0]
FINE_THR = np.arange(0.05, 1.0, 0.05)


def metrics_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total_neg = int((y_true == 0).sum())
    total_pos = int((y_true == 1).sum())
    fpr = fp / max(total_neg, 1)
    fnr = fn / max(total_pos, 1)
    tpr = tp / max(total_pos, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-12)
    return {
        "threshold": thr, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fpr": round(fpr, 6), "fnr": round(fnr, 6), "recall": round(tpr, 6),
        "precision": round(prec, 6), "f1": round(f1, 6),
        "total_pos": total_pos, "total_neg": total_neg,
    }


def main():
    print("=" * 80)
    print("ALTMAN HONEST — Temporal Split, No Target Leakage")
    print("=" * 80)

    t0 = time.time()
    
    # Load ALL data to get proper temporal ordering
    print("  Loading full Altman dataset (streaming)...")
    chunks = []
    for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
                              low_memory=False, chunksize=500_000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Total rows: {len(df):,}")
    
    # Parse label
    df["label"] = (df["Is Fraud?"].astype(str).str.strip() == "Yes").astype(int)
    print(f"  Fraud: {df['label'].sum():,} ({df['label'].mean()*100:.4f}%)")
    
    # Parse time features
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
    
    # Parse amount
    df["amt"] = pd.to_numeric(
        df["Amount"].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False),
        errors='coerce'
    ).fillna(0)
    
    # Parse MCC
    df["mcc_n"] = pd.to_numeric(df["MCC"], errors='coerce').fillna(0)
    
    # Use temporal ordering: first 80% of rows = train, last 20% = test
    # (maintains temporal separation)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"  Train: {len(train_df):,} ({train_df['label'].sum():,} fraud)")
    print(f"  Test:  {len(test_df):,} ({test_df['label'].sum():,} fraud)")
    
    # ── Feature engineering on TRAIN ONLY (no leakage) ──
    print("\n  Engineering features (no target leakage)...")
    
    gm_train = train_df["label"].mean()
    
    # Merchant fraud rate (from train only)
    merch_fraud = train_df.groupby("Merchant Name")["label"].agg(["mean", "count"])
    merch_fraud["merch_fr"] = (merch_fraud["mean"] * merch_fraud["count"] + gm_train * 50) / (merch_fraud["count"] + 50)
    merch_map = merch_fraud["merch_fr"].to_dict()
    
    # City fraud rate (from train only)
    city_fraud = train_df.groupby("Merchant City")["label"].agg(["mean", "count"])
    city_fraud["city_fr"] = (city_fraud["mean"] * city_fraud["count"] + gm_train * 50) / (city_fraud["count"] + 50)
    city_map = city_fraud["city_fr"].to_dict()
    
    # Merchant tx count (from train)
    merch_counts = train_df.groupby("Merchant Name").size().to_dict()
    
    def engineer(df_part):
        """Build features without target leakage."""
        out = pd.DataFrame()
        
        out["log_amt"] = np.log1p(df_part["amt"])
        out["amt_sq"] = df_part["amt"] ** 2
        out["very_high_amt"] = (df_part["amt"] > 1000).astype(int)
        
        hr = df_part["hr"].fillna(12).astype(float)
        out["hour_cos"] = np.cos(2 * np.pi * hr / 24)
        out["is_business_hours"] = ((hr >= 9) & (hr <= 17)).astype(int)
        
        out["chip"] = (df_part["Use Chip"].astype(str).str.contains("Swipe|Chip", case=False, na=False)).astype(int)
        out["is_online"] = (df_part["Use Chip"].astype(str).str.contains("Online", case=False, na=False)).astype(int)
        out["mcc_n"] = df_part["mcc_n"]
        out["has_zip"] = df_part["Zip"].notna().astype(int)
        out["has_state"] = df_part["Merchant State"].notna().astype(int)
        
        # Merchant features (lookup from train map, default to global mean)
        out["merch_tx_count"] = df_part["Merchant Name"].map(merch_counts).fillna(1)
        out["merch_fraud_rate"] = df_part["Merchant Name"].map(merch_map).fillna(gm_train)
        out["city_fraud_rate"] = df_part["Merchant City"].map(city_map).fillna(gm_train)
        
        # Interactions
        out["amt_x_mcc"] = df_part["amt"] * out["mcc_n"]
        out["amt_x_online"] = df_part["amt"] * out["is_online"]
        
        return out
    
    Xtr = engineer(train_df).values.astype(np.float32)
    ytr = train_df["label"].values
    Xte = engineer(test_df).values.astype(np.float32)
    yte = test_df["label"].values
    
    Xtr = np.nan_to_num(Xtr, nan=0, posinf=100, neginf=-100)
    Xte = np.nan_to_num(Xte, nan=0, posinf=100, neginf=-100)
    
    # Scale on train
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    
    feat_names = [
        "log_amt", "amt_sq", "very_high_amt", "hour_cos", "is_business_hours",
        "chip", "is_online", "mcc_n", "has_zip", "has_state",
        "merch_tx_count", "merch_fraud_rate", "city_fraud_rate",
        "amt_x_mcc", "amt_x_online"
    ]
    print(f"  Features: {len(feat_names)}")
    
    # Train models
    import xgboost as xgb
    import lightgbm as lgb
    
    spw = (len(ytr) - int(ytr.sum())) / max(int(ytr.sum()), 1)
    print(f"  Scale pos weight: {spw:.1f}")
    
    print("\n  Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, gamma=2,
        min_child_weight=5, scale_pos_weight=min(spw, 200),
        tree_method="hist", eval_metric="auc",
        random_state=42, n_jobs=4
    )
    xgb_model.fit(Xtr_s, ytr, verbose=False)
    
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
        scale_pos_weight=min(spw, 200),
        random_state=42, n_jobs=4, verbose=-1
    )
    lgb_model.fit(Xtr_s, ytr)
    
    # Ensemble
    p_xgb = xgb_model.predict_proba(Xte_s)[:, 1]
    p_lgb = lgb_model.predict_proba(Xte_s)[:, 1]
    y_prob = 0.5 * p_xgb + 0.5 * p_lgb
    
    auc = roc_auc_score(yte, y_prob)
    
    # Threshold sweep
    print(f"\n  Temporal ROC-AUC: {auc:.4f}")
    print(f"\n  {'Threshold':>10} | {'FPR':>10} | {'Recall':>10} | {'Prec':>10} | {'F1':>10} | {'TP':>6} {'FP':>6} {'TN':>10} {'FN':>4}")
    print("  " + "-" * 100)
    
    results = []
    for thr in THRESHOLDS:
        m = metrics_at_threshold(yte, y_prob, thr)
        results.append(m)
        print(f"  {thr:>10.2f} | {m['fpr']:>9.4f} | {m['recall']:>9.4f} | {m['precision']:>9.4f} | {m['f1']:>9.4f} | {m['tp']:>6} {m['fp']:>6} {m['tn']:>10} {m['fn']:>4}")
    
    # Best thresholds
    best_fpr1 = max((r for r in results if r["fpr"] <= 0.01), key=lambda r: r["recall"], default=None)
    best_fpr01 = max((r for r in results if r["fpr"] <= 0.001), key=lambda r: r["recall"], default=None)
    best_f1 = max(results, key=lambda r: r["f1"])
    
    print(f"\n  === Summary ===")
    if best_fpr1:
        print(f"  Best @ FPR ≤ 1%:    thr={best_fpr1['threshold']:.2f}  recall={best_fpr1['recall']:.4f}  prec={best_fpr1['precision']:.4f}")
    if best_fpr01:
        print(f"  Best @ FPR ≤ 0.1%:  thr={best_fpr01['threshold']:.2f}  recall={best_fpr01['recall']:.4f}  prec={best_fpr01['precision']:.4f}")
    print(f"  Best F1:            thr={best_f1['threshold']:.2f}  f1={best_f1['f1']:.4f}  prec={best_f1['precision']:.4f}")
    
    # Fine-grained sweep
    print(f"\n  Fine-grained sweep:")
    print(f"  {'Threshold':>10} | {'FPR':>10} | {'Recall':>10} | {'Prec':>10} | {'F1':>10}")
    print("  " + "-" * 60)
    fine_results = []
    for thr in FINE_THR:
        m = metrics_at_threshold(yte, y_prob, thr)
        fine_results.append(m)
        print(f"  {thr:>10.2f} | {m['fpr']:>9.4f} | {m['recall']:>9.4f} | {m['precision']:>9.4f} | {m['f1']:>9.4f}")
    
    # Feature importance
    print("\n  === Feature Importance (XGBoost) ===")
    importances = xgb_model.feature_importances_
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1])[:10]:
        print(f"    {name:>25}: {imp:.4f}")
    
    elapsed = time.time() - t0
    
    out = {
        "dataset": "ibm_altman_honest",
        "split": "temporal_80_20",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fraud_test": int(yte.sum()),
        "roc_auc": round(auc, 6),
        "features": feat_names,
        "feature_importance": {name: round(float(imp), 4) for name, imp in zip(feat_names, importances)},
        "sweep_results": results,
        "fine_sweep": fine_results,
        "elapsed_seconds": round(elapsed, 1),
    }
    
    with open("reports/threshold_sweep_altman_honest.json", "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"\n  Saved: reports/threshold_sweep_altman_honest.json ({elapsed:.1f}s)")
    return out


if __name__ == "__main__":
    main()
