#!/usr/bin/env python3
"""
Deep analysis of what fraud patterns the new features detect.
Provides actionable insight into detection improvement.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data")

DROP_COLS = {
    "label", "new_device_flag", "unusual_location_flag", "failed_auth_count_24h",
    "known_device_count", "shared_device_accounts", "shared_recipient_accounts",
    "mule_ring_score", "is_weekend", "amount_v_correlation", "days_since_last_similar_txn",
}


def analyze_patterns():
    print("=" * 70)
    print("FRAUD PATTERN ANALYSIS — What the new features detect")
    print("=" * 70)

    df = pd.read_csv(DATA_DIR / "transactions_v2.csv")
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    
    fraud = df[df["label"] == 1]
    legit = df[df["label"] == 0]
    
    print(f"\nDataset: {len(df)} rows | Fraud: {len(fraud)} ({len(fraud)/len(df)*100:.2f}%)")
    
    # --- Pattern 1: Amount-based patterns ---
    print("\n" + "=" * 50)
    print("PATTERN 1: AMOUNT DISTRIBUTION")
    print("=" * 50)
    print(f"  Fraud avg amount:   ${fraud.get('amount_ratio', pd.Series([0])).mean() * 3.93:.2f}")
    print(f"  Legit avg amount:   ${legit.get('amount_ratio', pd.Series([0])).mean() * 3.93:.2f}")
    
    amt_quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print("\n  Amount ratio quantiles:")
    for q in amt_quantiles:
        fq = fraud["amount_ratio"].quantile(q) if "amount_ratio" in fraud else 0
        lq = legit["amount_ratio"].quantile(q) if "amount_ratio" in legit else 0
        print(f"    P{int(q*100):2d}: fraud={fq:.3f}  legit={lq:.3f}")
    
    # --- Pattern 2: PCA-based patterns ---
    print("\n" + "=" * 50)
    print("PATTERN 2: PCA VECTOR ANOMALIES")
    print("=" * 50)
    print(f"  Fraud v_magnitude:    mean={fraud['v_magnitude'].mean():.2f}  std={fraud['v_magnitude'].std():.2f}")
    print(f"  Legit v_magnitude:    mean={legit['v_magnitude'].mean():.2f}  std={legit['v_magnitude'].std():.2f}")
    print(f"  Fraud v_asymmetry:    mean={fraud['v_asymmetry'].mean():.3f}  std={fraud['v_asymmetry'].std():.3f}")
    print(f"  Legit v_asymmetry:    mean={legit['v_asymmetry'].mean():.3f}  std={legit['v_asymmetry'].std():.3f}")
    print(f"  Fraud v_extreme_count: mean={fraud['v_extreme_count'].mean():.1f}  std={fraud['v_extreme_count'].std():.1f}")
    print(f"  Legit v_extreme_count: mean={legit['v_extreme_count'].mean():.1f}  std={legit['v_extreme_count'].std():.1f}")
    print(f"  Fraud pca_anomaly:    mean={fraud['pca_anomaly_score'].mean():.2f}  std={fraud['pca_anomaly_score'].std():.2f}")
    print(f"  Legit pca_anomaly:    mean={legit['pca_anomaly_score'].mean():.2f}  std={legit['pca_anomaly_score'].std():.2f}")
    
    # --- Pattern 3: Temporal patterns ---
    print("\n" + "=" * 50)
    print("PATTERN 3: TEMPORAL PATTERNS")
    print("=" * 50)
    fraud_hour = fraud["hour_of_day"]
    legit_hour = legit["hour_of_day"]
    print(f"  Fraud peak hours: {fraud_hour.mode().values[0]:.0f}:00 - {fraud_hour.median():.0f}:00 (median)")
    print(f"  Legit peak hours: {legit_hour.mode().values[0]:.0f}:00 - {legit_hour.median():.0f}:00 (median)")
    print(f"  Fraud unusual time rate: {fraud['txn_time_unusual'].mean()*100:.1f}%")
    print(f"  Legit unusual time rate: {legit['txn_time_unusual'].mean()*100:.1f}%")
    
    # --- Pattern 4: Novelty patterns ---
    print("\n" + "=" * 50)
    print("PATTERN 4: RECIPIENT/NOVELTY PATTERNS")
    print("=" * 50)
    print(f"  Fraud unusual_recipient rate: {fraud['unusual_recipient_flag'].mean()*100:.1f}%")
    print(f"  Legit unusual_recipient rate: {legit['unusual_recipient_flag'].mean()*100:.1f}%")
    print(f"  Fraud recipient_novelty:      {fraud['recipient_novelty'].mean():.3f}")
    print(f"  Legit recipient_novelty:      {legit['recipient_novelty'].mean():.3f}")
    
    # --- Pattern 5: Escalation patterns ---
    print("\n" + "=" * 50)
    print("PATTERN 5: ESCALATION PATTERNS")
    print("=" * 50)
    print(f"  Fraud escalation: {fraud['gradual_escalation_score'].mean():.3f}")
    print(f"  Legit escalation: {legit['gradual_escalation_score'].mean():.3f}")
    
    # --- ROC curve analysis ---
    print("\n" + "=" * 50)
    print("DETECTION THRESHOLDS")
    print("=" * 50)
    
    X = df[feature_cols].values
    y = df["label"].values
    X = np.nan_to_num(X, nan=0.0, posinf=100, neginf=-100)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X, y)
    proba = rf.predict_proba(X)[:, 1]
    
    fpr_arr, tpr_arr, thresholds = roc_curve(y, proba)
    
    # Find thresholds for key FPR levels
    for target_fpr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        valid = fpr_arr <= target_fpr
        if valid.any():
            idx = np.where(valid)[0][-1]
            print(f"  FPR={target_fpr*100:.1f}%: threshold={thresholds[idx]:.4f}  recall={tpr_arr[idx]:.4f}")
        else:
            print(f"  FPR={target_fpr*100:.1f}%: not achievable")
    
    # Cross-validation for honest estimate
    print("\n" + "=" * 50)
    print("5-FOLD CROSS-VALIDATION (honest estimate)")
    print("=" * 50)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    fold_prs = []
    fold_r1s = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        rf_fold = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_fold.fit(X[train_idx], y[train_idx])
        p = rf_fold.predict_proba(X[test_idx])[:, 1]
        
        auc = roc_auc_score(y[test_idx], p)
        pr = average_precision_score(y[test_idx], p)
        fpr_f, tpr_f, _ = roc_curve(y[test_idx], p)
        valid = fpr_f <= 0.01
        r1 = tpr_f[valid][-1] if valid.any() else 0.0
        
        fold_aucs.append(auc)
        fold_prs.append(pr)
        fold_r1s.append(r1)
        print(f"  Fold {fold+1}: AUC={auc:.4f}  PR-AUC={pr:.4f}  R@1%FPR={r1:.4f}")
    
    print(f"\n  Mean ROC-AUC:  {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  Mean PR-AUC:   {np.mean(fold_prs):.4f} ± {np.std(fold_prs):.4f}")
    print(f"  Mean R@1%FPR:  {np.mean(fold_r1s):.4f} ± {np.std(fold_r1s):.4f}")
    
    # --- Feature ablation study ---
    print("\n" + "=" * 50)
    print("FEATURE ABLATION — Impact of each new feature group")
    print("=" * 50)
    
    base_features = [
        "amount_ratio", "txn_time_unusual", "unusual_recipient_flag",
        "gradual_escalation_score", "hour_of_day", "hour_deviation",
        "amount_zscore", "recipient_novelty", "txn_regularity", "amount_log"
    ]
    pca_features = ["v_magnitude", "v_asymmetry", "pca_anomaly_score", "v_extreme_count"]
    amount_features = ["amount_bucket", "amount_time_interaction"]
    temporal_features = ["time_normalized"]
    
    from sklearn.model_selection import cross_val_score
    
    # Base only
    X_base = df[[c for c in base_features if c in df.columns]].values
    X_base = np.nan_to_num(X_base, nan=0, posinf=100, neginf=-100)
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    scores_base = cross_val_score(rf_base, X_base, y, cv=5, scoring="roc_auc")
    print(f"  Base features ({len(base_features)}):        AUC = {scores_base.mean():.4f} ± {scores_base.std():.4f}")
    
    # Base + PCA
    X_base_pca = df[[c for c in base_features + pca_features if c in df.columns]].values
    X_base_pca = np.nan_to_num(X_base_pca, nan=0, posinf=100, neginf=-100)
    rf_bp = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    scores_bp = cross_val_score(rf_bp, X_base_pca, y, cv=5, scoring="roc_auc")
    print(f"  + PCA features ({len(base_features)+len(pca_features)}):    AUC = {scores_bp.mean():.4f} ± {scores_bp.std():.4f}")
    
    # All features
    X_all = df[feature_cols].values
    X_all = np.nan_to_num(X_all, nan=0, posinf=100, neginf=-100)
    rf_all = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    scores_all = cross_val_score(rf_all, X_all, y, cv=5, scoring="roc_auc")
    print(f"  All features ({len(feature_cols)}):         AUC = {scores_all.mean():.4f} ± {scores_all.std():.4f}")
    
    delta_pca = scores_bp.mean() - scores_base.mean()
    delta_all = scores_all.mean() - scores_base.mean()
    print(f"\n  PCA features add: +{delta_pca:.4f} ROC-AUC")
    print(f"  All new features add: +{delta_all:.4f} ROC-AUC")


if __name__ == "__main__":
    analyze_patterns()
