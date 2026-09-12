#!/usr/bin/env python3
"""Temporal evaluation: pre-compute per-card stats from training, apply to test.

This matches production: per-card statistics are computed from historical data
and cached. New transactions use cached statistics.
"""
import sys, warnings, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("data/real_data_evaluation")


def build_features_from_stats(df, card_stats):
    """Build features using pre-computed per-card statistics."""
    from src.risk_engine.fusion import ML_FEATURES
    N = len(ML_FEATURES); n = len(df)
    X = np.zeros((n, N), dtype=np.float32)
    amt = df['amt'].values.astype(np.float32)
    hours = pd.to_datetime(df['trans_date_trans_time']).dt.hour.values
    dow = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek.values
    unix = df['unix_time'].values.astype(np.float64)
    cards = df['cc_num'].values
    merchs = df['merchant'].values
    cats = df['category'].values
    lats = df['lat'].values; lons = df['long'].values
    merch_lats = df['merch_lat'].values; merch_lons = df['merch_long'].values
    cities = df['city'].values
    
    for i in range(n):
        c = cards[i]; a = amt[i]; h = hours[i]; d = dow[i]; u = unix[i]
        m = merchs[i]; cat = cats[i]
        stats = card_stats.get(c, None)
        
        if stats and stats['count'] > 0:
            med_amt = stats['median_amt']
            mean_amt = stats['mean_amt']
            std_amt = stats['std_amt']
            count = stats['count']
            med_hour = stats['median_hour']
            hist_merchs = stats['merchants']
            hist_cats = stats['categories']
            hist_cities = stats['cities']
            first_tx = stats['first_tx']
            last_tx = stats['last_tx']
            
            X[i, ML_FEATURES.index('amount_ratio')] = a / med_amt if med_amt > 0 else 0
            X[i, ML_FEATURES.index('txn_freq_last_24h')] = count
            X[i, ML_FEATURES.index('txn_time_unusual')] = abs(h - med_hour) / 12.0
            X[i, ML_FEATURES.index('new_device_flag')] = 1.0 if cat not in hist_cats else 0.0
            dist = np.sqrt((lats[i] - merch_lats[i])**2 + (lons[i] - merch_lons[i])**2)
            X[i, ML_FEATURES.index('unusual_location_flag')] = 0.0  # simplified
            X[i, ML_FEATURES.index('unusual_recipient_flag')] = 1.0 if m not in hist_merchs else 0.0
            X[i, ML_FEATURES.index('days_since_last_similar_txn')] = min((u - last_tx) / 86400.0, 365)
            X[i, ML_FEATURES.index('gradual_escalation_score')] = a / mean_amt if mean_amt > 0 else 0
            X[i, ML_FEATURES.index('known_device_count')] = len(hist_merchs)
            X[i, ML_FEATURES.index('account_tenure_days')] = (u - first_tx) / 86400.0
            X[i, ML_FEATURES.index('hour_of_day')] = h
            X[i, ML_FEATURES.index('is_weekend')] = 1.0 if d >= 5 else 0.0
            X[i, ML_FEATURES.index('shared_device_accounts')] = 1.0
            X[i, ML_FEATURES.index('shared_recipient_accounts')] = 1.0
            X[i, ML_FEATURES.index('mule_ring_score')] = len(hist_cities)
            X[i, ML_FEATURES.index('hour_deviation')] = abs(h - med_hour)
            X[i, ML_FEATURES.index('amount_zscore')] = (a - mean_amt) / max(std_amt, 1e-6)
            X[i, ML_FEATURES.index('velocity_deviation')] = count / 24.0
            merch_count = stats['merch_counts'].get(m, 0)
            X[i, ML_FEATURES.index('recipient_novelty')] = 1.0 / (merch_count + 1)
            X[i, ML_FEATURES.index('txn_regularity')] = stats.get('gap_std', 0)
        else:
            # New card — first transaction
            X[i, ML_FEATURES.index('amount_ratio')] = 1.0
            X[i, ML_FEATURES.index('txn_freq_last_24h')] = 0
            X[i, ML_FEATURES.index('txn_time_unusual')] = h / 23.0
            X[i, ML_FEATURES.index('new_device_flag')] = 1.0
            X[i, ML_FEATURES.index('unusual_location_flag')] = 0.0
            X[i, ML_FEATURES.index('unusual_recipient_flag')] = 1.0
            X[i, ML_FEATURES.index('failed_auth_count_24h')] = 0.0
            X[i, ML_FEATURES.index('days_since_last_similar_txn')] = 0.0
            X[i, ML_FEATURES.index('gradual_escalation_score')] = 1.0
            X[i, ML_FEATURES.index('known_device_count')] = 0
            X[i, ML_FEATURES.index('account_tenure_days')] = 0
            X[i, ML_FEATURES.index('hour_of_day')] = h
            X[i, ML_FEATURES.index('is_weekend')] = 1.0 if d >= 5 else 0.0
            X[i, ML_FEATURES.index('shared_device_accounts')] = 1.0
            X[i, ML_FEATURES.index('shared_recipient_accounts')] = 1.0
            X[i, ML_FEATURES.index('mule_ring_score')] = 0
            X[i, ML_FEATURES.index('hour_deviation')] = 0
            X[i, ML_FEATURES.index('amount_zscore')] = 0
            X[i, ML_FEATURES.index('velocity_deviation')] = 0
            X[i, ML_FEATURES.index('recipient_novelty')] = 1.0
            X[i, ML_FEATURES.index('txn_regularity')] = 0
        
        X[i, ML_FEATURES.index('failed_auth_count_24h')] = 0.0
    
    return X


def compute_card_stats(df):
    """Pre-compute per-card statistics from training data."""
    stats = {}
    for card, grp in df.groupby('cc_num'):
        grp = grp.sort_values('unix_time')
        amts = grp['amt'].values
        hours = pd.to_datetime(grp['trans_date_trans_time']).dt.hour.values
        unixs = grp['unix_time'].values
        
        merchs = set(grp['merchant'].values)
        cats = set(grp['category'].values)
        cities = set(grp['city'].values)
        merch_counts = grp['merchant'].value_counts().to_dict()
        
        gaps = np.diff(unixs) / 3600.0 if len(unixs) > 1 else np.array([0])
        
        stats[card] = {
            'median_amt': float(np.median(amts)),
            'mean_amt': float(np.mean(amts)),
            'std_amt': float(np.std(amts)) if len(amts) > 1 else 1.0,
            'count': len(amts),
            'median_hour': float(np.median(hours)),
            'merchants': merchs,
            'categories': cats,
            'cities': cities,
            'merch_counts': merch_counts,
            'first_tx': float(unixs[0]),
            'last_tx': float(unixs[-1]),
            'gap_std': float(np.std(gaps)) if len(gaps) > 1 else 0,
        }
    return stats


def eval_metrics(y_true, y_prob):
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    mets = {}
    for tf, lab in [(0.0001,'r01ppct'),(0.0005,'r05ppct'),(0.001,'r01pct'),(0.005,'r05pct'),(0.01,'r1pct')]:
        mask = fpr_arr <= tf
        mets[lab] = round(float(tpr_arr[np.where(mask)[0][-1]]), 4) if mask.any() else 0.0
    return {'roc_auc': round(float(roc), 4), 'pr_auc': round(float(pr), 4), **mets}


def main():
    print("=" * 70)
    print("TEMPORAL EVALUATION: Train 2019+Q1 → Test Apr-Sep 2020")
    print("Features: pre-computed from training data only (no leakage)")
    print("=" * 70)
    
    df = pd.concat([pd.read_csv('data/kaggle_fraud/fraudTrain.csv'),
                     pd.read_csv('data/kaggle_fraud/fraudTest.csv')], ignore_index=True)
    df = df.drop(columns=['first','last','street','state','zip','dob','job','trans_num','Unnamed: 0'], errors='ignore')
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['month'] = df['trans_date_trans_time'].dt.month
    df['year'] = df['trans_date_trans_time'].dt.year
    df = df.sort_values('unix_time').reset_index(drop=True)
    
    train_mask = (df['year'] == 2019) | ((df['year'] == 2020) & (df['month'] <= 3))
    test_mask = (df['year'] == 2020) & (df['month'] >= 4) & (df['month'] <= 9)
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"\nTrain: {len(df_train):,} rows ({df_train.is_fraud.sum():,} fraud, {df_train.is_fraud.mean()*100:.3f}%)")
    print(f"Test:  {len(df_test):,} rows ({df_test.is_fraud.sum():,} fraud, {df_test.is_fraud.mean()*100:.3f}%)")
    
    # Compute card stats from training data only
    print("\nComputing per-card statistics from training data...")
    t0 = time.time()
    card_stats = compute_card_stats(df_train)
    print(f"  {len(card_stats):,} unique cards, {time.time()-t0:.1f}s")
    
    # Build features
    print("Building training features...")
    X_train = build_features_from_stats(df_train, card_stats)
    y_train = df_train['is_fraud'].values.astype(int)
    print("Building test features (using training card stats only)...")
    X_test = build_features_from_stats(df_test, card_stats)
    y_test = df_test['is_fraud'].values.astype(int)
    
    # Train
    print("Training XGBoost...")
    t0 = time.time()
    imp = SimpleImputer(strategy='median').fit(X_train)
    Xtc = np.nan_to_num(imp.transform(X_train), nan=0.0, posinf=10.0, neginf=-10.0)
    Xec = np.nan_to_num(imp.transform(X_test), nan=0.0, posinf=10.0, neginf=-10.0)
    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    scale = max(n_neg / max(n_pos, 1), 1.0)
    model = XGBClassifier(n_estimators=600, max_depth=10, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
                          eval_metric='auc', random_state=42, n_jobs=-1, verbosity=0,
                          min_child_weight=3, gamma=0.05, reg_alpha=0.1)
    model.fit(Xtc, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    
    # Overall
    prob = model.predict_proba(Xec)[:, 1]
    overall = eval_metrics(y_test, prob)
    print(f"\n{'='*70}")
    print(f"OVERALL (Apr-Sep 2020): {len(y_test):,} rows, {int(y_test.sum()):,} fraud")
    print(f"  ROC-AUC: {overall['roc_auc']:.4f}")
    print(f"  PR-AUC: {overall['pr_auc']:.4f}")
    print(f"  Recall@0.05%FPR: {overall['r05ppct']:.4f}")
    print(f"  Recall@0.1%FPR: {overall['r01pct']:.4f}")
    print(f"  Recall@1%FPR: {overall['r1pct']:.4f}")
    
    # Month-by-month
    print(f"\n{'='*70}")
    print(f"MONTH-BY-MONTH (Apr-Sep 2020)")
    print(f"{'='*70}")
    print(f"{'Month':<12} {'Rows':>8} {'Fraud':>7} {'Prev%':>7} {'ROC':>7} {'PR':>7} {'R@.05%':>7} {'R@.1%':>7} {'R@1%':>7}")
    print("-" * 80)
    
    month_results = []
    month_names = {4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep'}
    for m in range(4, 10):
        mask = df_test['month'].values == m
        if mask.sum() == 0: continue
        y_m = y_test[mask]; p_m = prob[mask]
        if y_m.sum() == 0:
            print(f"{month_names[m]:<12} {mask.sum():>8,} {0:>7} {0:>7.3f} {'N/A':>7}")
            continue
        mets = eval_metrics(y_m, p_m)
        prev = y_m.mean() * 100
        print(f"{month_names[m]:<12} {mask.sum():>8,} {int(y_m.sum()):>7,} {prev:>7.3f} {mets['roc_auc']:>7.4f} {mets['pr_auc']:>7.4f} {mets['r05ppct']:>7.4f} {mets['r01pct']:>7.4f} {mets['r1pct']:>7.4f}")
        month_results.append({'month': month_names[m], 'n': int(mask.sum()), 'fraud': int(y_m.sum()), 'prevalence': round(prev, 3), **mets})
    
    # In-sample
    p_train = model.predict_proba(Xtc)[:, 1]
    train_mets = eval_metrics(y_train, p_train)
    print(f"\nIn-sample (train): ROC-AUC={train_mets['roc_auc']:.4f} R@.05%={train_mets['r05ppct']:.4f} R@1%={train_mets['r1pct']:.4f}")
    
    # Save
    results = {
        'train_period': '2019 all + Jan-Mar 2020', 'test_period': 'Apr-Sep 2020',
        'train_n': len(df_train), 'test_n': len(df_test),
        'train_fraud': int(y_train.sum()), 'test_fraud': int(y_test.sum()),
        'unique_cards_train': len(card_stats),
        'overall': overall, 'in_sample': train_mets, 'monthly': month_results,
        'method': 'pre-computed card stats from training only (no leakage)',
    }
    with open(OUT / 'temporal_proper_eval.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT / 'temporal_proper_eval.json'}")

if __name__ == '__main__':
    main()
