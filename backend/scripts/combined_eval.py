#!/usr/bin/env python3
"""Combined evaluation across all fraud datasets.

Loads every available fraud dataset, maps to ML_FEATURES, runs the
production model, and computes combined metrics for the landing page.
"""

import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import joblib
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk_engine.fusion import ML_FEATURES
N_FEATURES = len(ML_FEATURES)

OUT_DIR = Path("data/real_data_evaluation")


def compute_kaggle_features(df):
    """Map kaggle_fraud columns to 21 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    amt = df['amt'].values.astype(np.float32)
    hours = pd.to_datetime(df['trans_date_trans_time']).dt.hour.values
    dow = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek.values
    unix = df['unix_time'].values.astype(np.float64)
    X[:, ML_FEATURES.index('amount_ratio')] = np.where(df.groupby('cc_num')['amt'].transform('median').values > 0, amt / df.groupby('cc_num')['amt'].transform('median').values, 0)
    X[:, ML_FEATURES.index('txn_freq_last_24h')] = df.groupby('cc_num')['cc_num'].transform('count').values.astype(np.float32)
    X[:, ML_FEATURES.index('txn_time_unusual')] = hours / 23.0
    X[:, ML_FEATURES.index('new_device_flag')] = (df.groupby(['cc_num', 'category']).cumcount() == 0).astype(np.float32)
    lat1, lon1 = df['lat'].values, df['long'].values
    lat2, lon2 = df['merch_lat'].values, df['merch_long'].values
    dist = np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
    card_dist_med = df.assign(dist=dist).groupby('cc_num')['dist'].transform('median').values
    X[:, ML_FEATURES.index('unusual_location_flag')] = (dist > card_dist_med * 2).astype(np.float32)
    X[:, ML_FEATURES.index('unusual_recipient_flag')] = (df.groupby(['cc_num', 'merchant']).cumcount() == 0).astype(np.float32)
    X[:, ML_FEATURES.index('failed_auth_count_24h')] = 0.0
    df_sorted = df.sort_values(['cc_num', 'unix_time'])
    gaps = df_sorted.groupby('cc_num')['unix_time'].diff().fillna(86400).values / 86400.0
    X[:, ML_FEATURES.index('days_since_last_similar_txn')] = np.clip(gaps[:n], 0, 365).astype(np.float32)
    rolling_mean = df.groupby('cc_num')['amt'].transform(lambda x: x.expanding().mean()).values
    X[:, ML_FEATURES.index('gradual_escalation_score')] = np.where(rolling_mean > 0, amt / rolling_mean, 0).astype(np.float32)
    X[:, ML_FEATURES.index('known_device_count')] = df.groupby('cc_num')['merchant'].transform('nunique').values.astype(np.float32)
    first_tx = df.groupby('cc_num')['unix_time'].transform('min').values
    X[:, ML_FEATURES.index('account_tenure_days')] = ((unix - first_tx) / 86400.0).astype(np.float32)
    X[:, ML_FEATURES.index('hour_of_day')] = hours.astype(np.float32)
    X[:, ML_FEATURES.index('is_weekend')] = (dow >= 5).astype(np.float32)
    X[:, ML_FEATURES.index('shared_device_accounts')] = df.groupby('merchant')['cc_num'].transform('nunique').values.astype(np.float32)
    X[:, ML_FEATURES.index('shared_recipient_accounts')] = X[:, ML_FEATURES.index('shared_device_accounts')]
    X[:, ML_FEATURES.index('mule_ring_score')] = df.groupby('cc_num')['city'].transform('nunique').values.astype(np.float32)
    median_hour = df.groupby('cc_num')['trans_date_trans_time'].transform(lambda x: pd.to_datetime(x).dt.hour.median()).values
    X[:, ML_FEATURES.index('hour_deviation')] = np.abs(hours - median_hour).astype(np.float32)
    grp_mean = df.groupby('cc_num')['amt'].transform('mean').values
    grp_std = df.groupby('cc_num')['amt'].transform('std').fillna(1.0).values
    X[:, ML_FEATURES.index('amount_zscore')] = np.where(grp_std > 0, (amt - grp_mean) / grp_std, 0).astype(np.float32)
    X[:, ML_FEATURES.index('velocity_deviation')] = df.groupby('cc_num')['cc_num'].transform('count').values.astype(np.float32) / 24.0
    seen = df.groupby(['cc_num', 'merchant']).cumcount().values + 1
    X[:, ML_FEATURES.index('recipient_novelty')] = (1.0 / seen).astype(np.float32)
    std_gap = df.groupby('cc_num')['unix_time'].transform(lambda x: x.diff().fillna(0).expanding().std()).values / 3600.0
    X[:, ML_FEATURES.index('txn_regularity')] = np.clip(std_gap, 0, 100).astype(np.float32)
    return X


def compute_paysim_features(df):
    """Map paysim columns to 21 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    amt = df['amount'].values.astype(np.float32)
    old_bal = df['oldbalanceOrg'].values.astype(np.float32)
    new_bal = df['newbalanceOrig'].values.astype(np.float32)
    old_dest = df['oldbalanceDest'].values.astype(np.float32)
    new_dest = df['newbalanceDest'].values.astype(np.float32)
    X[:, ML_FEATURES.index('amount_ratio')] = np.where(old_bal > 0, amt / old_bal, 0.0)
    type_map = {'PAYMENT': 1, 'TRANSFER': 1, 'CASH_OUT': 2, 'CASH_IN': 2, 'DEBIT': 1}
    X[:, ML_FEATURES.index('txn_freq_last_24h')] = df['type'].map(type_map).fillna(1).values.astype(np.float32)
    X[:, ML_FEATURES.index('txn_time_unusual')] = 0.5
    X[:, ML_FEATURES.index('new_device_flag')] = (new_bal < 1.0).astype(np.float32)
    X[:, ML_FEATURES.index('unusual_location_flag')] = (np.abs(new_dest - old_dest) > amt * 2).astype(np.float32)
    X[:, ML_FEATURES.index('unusual_recipient_flag')] = (old_dest < 1.0).astype(np.float32)
    X[:, ML_FEATURES.index('failed_auth_count_24h')] = 0.0
    X[:, ML_FEATURES.index('days_since_last_similar_txn')] = 1.0
    X[:, ML_FEATURES.index('gradual_escalation_score')] = np.where(old_bal > 0, amt / old_bal, 0.0)
    X[:, ML_FEATURES.index('known_device_count')] = 1.0
    X[:, ML_FEATURES.index('account_tenure_days')] = 30.0
    X[:, ML_FEATURES.index('hour_of_day')] = 12.0
    X[:, ML_FEATURES.index('is_weekend')] = 0.0
    X[:, ML_FEATURES.index('shared_device_accounts')] = 1.0
    X[:, ML_FEATURES.index('shared_recipient_accounts')] = 1.0
    X[:, ML_FEATURES.index('mule_ring_score')] = (old_bal < 1.0).astype(np.float32) * (new_dest > 10000).astype(np.float32)
    X[:, ML_FEATURES.index('hour_deviation')] = 0.0
    X[:, ML_FEATURES.index('amount_zscore')] = (amt - np.median(amt)) / (np.std(amt) + 1e-6)
    X[:, ML_FEATURES.index('velocity_deviation')] = X[:, ML_FEATURES.index('txn_freq_last_24h')] / 24.0
    X[:, ML_FEATURES.index('recipient_novelty')] = 0.5
    X[:, ML_FEATURES.index('txn_regularity')] = 0.0
    return X


def compute_creditcard_features(df, label_col='Class'):
    """Map ULB/creditcard PCA features to 21 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    pca_cols = [c for c in df.columns if c.startswith('V')]
    pca_data = df[pca_cols].values
    amt = df['Amount'].values.astype(np.float32) if 'Amount' in df.columns else np.zeros(n, dtype=np.float32)
    X[:, ML_FEATURES.index('amount_ratio')] = np.where(np.median(amt) > 0, amt / np.median(amt), 0.0)
    X[:, ML_FEATURES.index('txn_freq_last_24h')] = np.abs(pca_data[:, :5]).mean(axis=1) * 10
    X[:, ML_FEATURES.index('txn_time_unusual')] = (df['Time'].values % 86400) / 86400.0 if 'Time' in df.columns else 0.5
    X[:, ML_FEATURES.index('new_device_flag')] = (np.abs(pca_data[:, 0]) > 2).astype(np.float32)
    X[:, ML_FEATURES.index('unusual_location_flag')] = (np.abs(pca_data[:, 1]) > 2).astype(np.float32)
    X[:, ML_FEATURES.index('unusual_recipient_flag')] = (np.abs(pca_data[:, 2]) > 2).astype(np.float32)
    X[:, ML_FEATURES.index('failed_auth_count_24h')] = 0.0
    X[:, ML_FEATURES.index('days_since_last_similar_txn')] = np.clip(df['Time'].values / 86400.0, 0, 30) if 'Time' in df.columns else 1.0
    X[:, ML_FEATURES.index('gradual_escalation_score')] = np.abs(pca_data[:, 3]).clip(0, 1)
    X[:, ML_FEATURES.index('known_device_count')] = 1.0
    X[:, ML_FEATURES.index('account_tenure_days')] = 30.0
    X[:, ML_FEATURES.index('hour_of_day')] = (df['Time'].values % 86400) / 3600.0 if 'Time' in df.columns else 12.0
    X[:, ML_FEATURES.index('is_weekend')] = 0.0
    X[:, ML_FEATURES.index('shared_device_accounts')] = 1.0
    X[:, ML_FEATURES.index('shared_recipient_accounts')] = 1.0
    X[:, ML_FEATURES.index('mule_ring_score')] = np.abs(pca_data[:, 4]).clip(0, 10)
    X[:, ML_FEATURES.index('hour_deviation')] = np.abs(pca_data[:, 5]).clip(0, 12)
    X[:, ML_FEATURES.index('amount_zscore')] = (amt - amt.mean()) / (amt.std() + 1e-6)
    X[:, ML_FEATURES.index('velocity_deviation')] = np.abs(pca_data[:, 6]).clip(0, 10)
    X[:, ML_FEATURES.index('recipient_novelty')] = np.abs(pca_data[:, 7]).clip(0, 1)
    X[:, ML_FEATURES.index('txn_regularity')] = np.abs(pca_data[:, 8]).clip(0, 100)
    return X


def evaluate(model, scaler, imputer, X, y, name):
    """Evaluate model and return metrics dict."""
    X_clean = imputer.transform(X)
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=10.0, neginf=-10.0)
    X_scaled = scaler.transform(X_clean)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    
    roc = roc_auc_score(y, y_prob)
    pr = average_precision_score(y, y_prob)
    
    fpr_arr, tpr_arr, thresholds = roc_curve(y, y_prob)
    r1, _ = 0.0, 0.0
    valid = fpr_arr <= 0.01
    if valid.any():
        idx = np.where(valid)[0][-1]
        r1 = tpr_arr[idx]
    
    return {
        'name': name, 'n': len(y), 'fraud': int(y.sum()),
        'roc_auc': round(roc, 4), 'pr_auc': round(pr, 4),
        'recall_1pct_fpr': round(r1, 4),
    }


def main():
    model = joblib.load('models/artifacts/xgboost.joblib')
    scaler = joblib.load('models/artifacts/scaler.joblib')
    imputer = joblib.load('models/artifacts/imputer.joblib')
    
    results = []
    
    # 1. kaggle_fraud (train+test combined, evaluate last 30%)
    print("Loading kaggle_fraud...")
    df = pd.concat([pd.read_csv('data/kaggle_fraud/fraudTrain.csv'),
                     pd.read_csv('data/kaggle_fraud/fraudTest.csv')], ignore_index=True)
    df = df.drop(columns=['first','last','street','state','zip','dob','job','trans_num','Unnamed: 0'], errors='ignore')
    X = compute_kaggle_features(df)
    y = df['is_fraud'].values.astype(int)
    split = int(len(df) * 0.7)
    r = evaluate(model, scaler, imputer, X[split:], y[split:], 'kaggle_fraud')
    results.append(r)
    print(f"  kaggle_fraud: n={r['n']:,}, fraud={r['fraud']}, ROC={r['roc_auc']:.4f}, R@1%={r['recall_1pct_fpr']:.4f}")
    
    # 2. paysim_1m
    print("Loading paysim_1m...")
    df = pd.read_csv('data/paysim_1m.csv')
    X = compute_paysim_features(df)
    y = df['isFraud'].values.astype(int)
    r = evaluate(model, scaler, imputer, X, y, 'paysim_1m')
    results.append(r)
    print(f"  paysim_1m: n={r['n']:,}, fraud={r['fraud']}, ROC={r['roc_auc']:.4f}, R@1%={r['recall_1pct_fpr']:.4f}")
    
    # 3. creditcard (ULB)
    print("Loading creditcard...")
    df = pd.read_csv('data/creditcard.csv')
    X = compute_creditcard_features(df)
    y = df['Class'].values.astype(int)
    r = evaluate(model, scaler, imputer, X, y, 'ulb_creditcard')
    results.append(r)
    print(f"  ulb_creditcard: n={r['n']:,}, fraud={r['fraud']}, ROC={r['roc_auc']:.4f}, R@1%={r['recall_1pct_fpr']:.4f}")
    
    # 4. fraud_data
    print("Loading fraud_data...")
    df = pd.read_csv('data/fraud_data.csv')
    X = compute_creditcard_features(df)
    y = df['Class'].values.astype(int)
    r = evaluate(model, scaler, imputer, X, y, 'fraud_data')
    results.append(r)
    print(f"  fraud_data: n={r['n']:,}, fraud={r['fraud']}, ROC={r['roc_auc']:.4f}, R@1%={r['recall_1pct_fpr']:.4f}")
    
    # 5. paysim (small)
    print("Loading paysim (small)...")
    df = pd.read_csv('data/paysim.csv')
    X = compute_paysim_features(df)
    y = df['isFraud'].values.astype(int)
    r = evaluate(model, scaler, imputer, X, y, 'paysim_small')
    results.append(r)
    print(f"  paysim_small: n={r['n']:,}, fraud={r['fraud']}, ROC={r['roc_auc']:.4f}, R@1%={r['recall_1pct_fpr']:.4f}")
    
    # Combined totals
    total_n = sum(r['n'] for r in results)
    total_fraud = sum(r['fraud'] for r in results)
    weighted_roc = sum(r['roc_auc'] * r['n'] for r in results) / total_n
    weighted_r1 = sum(r['recall_1pct_fpr'] * r['n'] for r in results) / total_n
    weighted_pr = sum(r['pr_auc'] * r['n'] for r in results) / total_n
    
    print(f"\n{'='*60}")
    print(f"COMBINED TOTALS:")
    print(f"  Total transactions: {total_n:,}")
    print(f"  Total fraud: {total_fraud:,}")
    print(f"  Weighted ROC-AUC: {weighted_roc:.4f}")
    print(f"  Weighted Recall@1%FPR: {weighted_r1:.4f}")
    print(f"  Weighted PR-AUC: {weighted_pr:.4f}")
    print(f"{'='*60}")
    
    # Save results
    combined = {
        'total_transactions': total_n,
        'total_fraud': total_fraud,
        'weighted_roc_auc': round(weighted_roc, 4),
        'weighted_recall_1pct_fpr': round(weighted_r1, 4),
        'weighted_pr_auc': round(weighted_pr, 4),
        'datasets': results,
    }
    with open(OUT_DIR / 'combined_all_datasets.json', 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"Saved to {OUT_DIR / 'combined_all_datasets.json'}")


if __name__ == '__main__':
    main()
