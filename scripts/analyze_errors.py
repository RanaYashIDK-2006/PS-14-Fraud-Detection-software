#!/usr/bin/env python3
"""Analyze model errors to find patterns in missed fraud."""
import warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')


def load_paysim_full():
    """Load PaySim with all features."""
    df = pd.read_csv('data/paysim_1m.csv')
    type_dummies = pd.get_dummies(df['type'], prefix='type').values.astype(np.float32)
    amt = df['amount'].values.astype(np.float32)
    ob = df['oldbalanceOrg'].values.astype(np.float32)
    nb = df['newbalanceOrig'].values.astype(np.float32)
    od = df['oldbalanceDest'].values.astype(np.float32)
    nd = df['newbalanceDest'].values.astype(np.float32)
    flagged = df['isFlaggedFraud'].values.astype(np.float32)
    bal_drain = (ob - nb).astype(np.float32)
    dest_gain = (nd - od).astype(np.float32)
    amt_ratio_ob = np.where(ob > 0, amt / ob, 0).astype(np.float32)
    amt_ratio_dest = np.where(od > 0, amt / od, 0).astype(np.float32)
    zero_dest = (od < 1.0).astype(np.float32)
    zero_orig = (ob < 1.0).astype(np.float32)
    amt_dest_ratio = np.where(nd > 0, amt / nd, 0).astype(np.float32)
    orig_complete_drain = np.where(ob > 0, (ob - nb) / ob, 0).astype(np.float32)
    dest_complete_gain = np.where(nd > 0, (nd - od) / nd, 0).astype(np.float32)
    
    # NEW: Additional ratio features
    amt_vs_avg = amt / (amt.mean() + 1e-6)  # amount vs global average
    ob_log = np.log1p(ob)
    amt_log = np.log1p(amt)
    dest_balance_ratio = np.where(od > 0, nd / od, 0).astype(np.float32)  # dest balance growth
    orig_balance_ratio = np.where(ob > 0, nb / ob, 0).astype(np.float32)  # orig balance remaining
    amt_is_round = (amt % 100 == 0).astype(np.float32)  # round amounts suspicious
    large_amt = (amt > amt.mean() + 2 * amt.std()).astype(np.float32)
    
    X = np.column_stack([
        type_dummies, amt.reshape(-1,1), ob.reshape(-1,1), nb.reshape(-1,1),
        od.reshape(-1,1), nd.reshape(-1,1), flagged.reshape(-1,1),
        bal_drain.reshape(-1,1), dest_gain.reshape(-1,1),
        amt_ratio_ob.reshape(-1,1), amt_ratio_dest.reshape(-1,1),
        zero_dest.reshape(-1,1), zero_orig.reshape(-1,1),
        amt_dest_ratio.reshape(-1,1), orig_complete_drain.reshape(-1,1),
        dest_complete_gain.reshape(-1,1),
        amt_vs_avg.reshape(-1,1), ob_log.reshape(-1,1), amt_log.reshape(-1,1),
        dest_balance_ratio.reshape(-1,1), orig_balance_ratio.reshape(-1,1),
        amt_is_round.reshape(-1,1), large_amt.reshape(-1,1),
    ])
    y = df['isFraud'].values.astype(int)
    return X, y, df


def main():
    print("=" * 70)
    print("ERROR ANALYSIS: PaySim 1M")
    print("=" * 70)
    
    X, y, df = load_paysim_full()
    print(f"Total: {len(y):,}, Fraud: {int(y.sum()):,} ({y.mean()*100:.2f}%)")
    
    # Train with proper split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    imp = SimpleImputer(strategy='median').fit(X_train)
    X_train_c = np.nan_to_num(imp.transform(X_train), nan=0.0, posinf=10.0, neginf=-10.0)
    X_test_c = np.nan_to_num(imp.transform(X_test), nan=0.0, posinf=10.0, neginf=-10.0)
    
    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    scale = max(n_neg / max(n_pos, 1), 1.0)
    model = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
                          eval_metric='auc', random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train_c, y_train)
    
    prob = model.predict_proba(X_test_c)[:, 1]
    roc = roc_auc_score(y_test, prob)
    print(f"ROC-AUC: {roc:.4f}")
    
    # Find threshold for 0.05% FPR
    fpr_arr, tpr_arr, thresholds = roc_curve(y_test, prob)
    for target_fpr in [0.0005, 0.001, 0.005, 0.01]:
        mask = fpr_arr <= target_fpr
        if mask.any():
            idx = np.where(mask)[0][-1]
            print(f"  Recall@{target_fpr*100:.2f}%FPR: {tpr_arr[idx]:.4f} (threshold={thresholds[idx]:.4f})")
    
    # Analyze errors
    test_idx = np.where(np.ones(len(y_test), dtype=bool))[0]  # placeholder
    fraud_mask = y_test == 1
    
    fraud_probs = prob[fraud_mask]
    legit_probs = prob[~fraud_mask]
    
    print(f"\nFraud score distribution:")
    print(f"  Mean: {fraud_probs.mean():.4f}")
    print(f"  Median: {np.median(fraud_probs):.4f}")
    print(f"  Min: {fraud_probs.min():.4f}")
    print(f"  Max: {fraud_probs.max():.4f}")
    print(f"  < 0.5: {(fraud_probs < 0.5).sum()} ({(fraud_probs < 0.5).mean()*100:.1f}%)")
    print(f"  < 0.1: {(fraud_probs < 0.1).sum()} ({(fraud_probs < 0.1).mean()*100:.1f}%)")
    
    print(f"\nLegit score distribution:")
    print(f"  Mean: {legit_probs.mean():.4f}")
    print(f"  > 0.5: {(legit_probs > 0.5).sum()} ({(legit_probs > 0.5).mean()*100:.1f}%)")
    print(f"  > 0.1: {(legit_probs > 0.1).sum()} ({(legit_probs > 0.1).mean()*100:.1f}%)")
    
    # Feature importance
    print(f"\nTop 15 features by importance:")
    imp = model.feature_importances_
    feat_names = [f'f{i}' for i in range(X.shape[1])]
    for i in np.argsort(imp)[::-1][:15]:
        print(f"  {feat_names[i]}: {imp[i]:.4f}")
    
    # Analyze missed fraud patterns
    missed = fraud_mask & (prob < 0.5)
    caught = fraud_mask & (prob >= 0.5)
    print(f"\nMissed fraud (prob < 0.5): {missed.sum()}")
    print(f"Caught fraud (prob >= 0.5): {caught.sum()}")
    
    # Save error analysis
    results = {
        'roc_auc': round(float(roc), 4),
        'missed_count': int(missed.sum()),
        'caught_count': int(caught.sum()),
        'fraud_mean_score': round(float(fraud_probs.mean()), 4),
        'fraud_median_score': round(float(np.median(fraud_probs)), 4),
        'fraud_below_05': int((fraud_probs < 0.5).sum()),
        'fraud_below_01': int((fraud_probs < 0.1).sum()),
        'legit_above_05': int((legit_probs > 0.5).sum()),
        'legit_above_01': int((legit_probs > 0.1).sum()),
        'feature_importance': {feat_names[i]: round(float(imp[i]), 4) for i in np.argsort(imp)[::-1][:15]},
    }
    with open('data/real_data_evaluation/error_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/real_data_evaluation/error_analysis.json")


if __name__ == '__main__':
    main()
