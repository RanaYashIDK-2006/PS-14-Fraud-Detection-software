#!/usr/bin/env python3
"""Temporal evaluation: train on Jan-Mar, test on Apr-Sep (month-by-month)."""
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


def build_features(df):
    """Build 21 ML_FEATURES from kaggle_fraud."""
    from src.risk_engine.fusion import ML_FEATURES
    N = len(ML_FEATURES); n = len(df)
    X = np.zeros((n, N), dtype=np.float32)
    amt = df['amt'].values.astype(np.float32)
    hours = pd.to_datetime(df['trans_date_trans_time']).dt.hour.values
    dow = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek.values
    unix = df['unix_time'].values.astype(np.float64)
    gm = df.groupby('cc_num')['amt'].transform('median').values
    gc = df.groupby('cc_num')['cc_num'].transform('count').values.astype(np.float32)
    X[:, ML_FEATURES.index('amount_ratio')] = np.where(gm > 0, amt / gm, 0)
    X[:, ML_FEATURES.index('txn_freq_last_24h')] = gc
    X[:, ML_FEATURES.index('txn_time_unusual')] = hours / 23.0
    X[:, ML_FEATURES.index('new_device_flag')] = (df.groupby(['cc_num','category']).cumcount() == 0).astype(np.float32)
    lat1, lon1 = df['lat'].values, df['long'].values
    lat2, lon2 = df['merch_lat'].values, df['merch_long'].values
    dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    X[:, ML_FEATURES.index('unusual_location_flag')] = (dist > df.assign(d=dist).groupby('cc_num')['d'].transform('median').values * 2).astype(np.float32)
    X[:, ML_FEATURES.index('unusual_recipient_flag')] = (df.groupby(['cc_num','merchant']).cumcount() == 0).astype(np.float32)
    X[:, ML_FEATURES.index('failed_auth_count_24h')] = 0.0
    df_s = df.sort_values(['cc_num','unix_time'])
    gaps = df_s.groupby('cc_num')['unix_time'].diff().fillna(86400).values / 86400.0
    X[:, ML_FEATURES.index('days_since_last_similar_txn')] = np.clip(gaps, 0, 365).astype(np.float32)
    rm = df.groupby('cc_num')['amt'].transform(lambda x: x.expanding().mean()).values
    X[:, ML_FEATURES.index('gradual_escalation_score')] = np.where(rm > 0, amt / rm, 0).astype(np.float32)
    X[:, ML_FEATURES.index('known_device_count')] = df.groupby('cc_num')['merchant'].transform('nunique').values.astype(np.float32)
    ft = df.groupby('cc_num')['unix_time'].transform('min').values
    X[:, ML_FEATURES.index('account_tenure_days')] = ((unix - ft) / 86400.0).astype(np.float32)
    X[:, ML_FEATURES.index('hour_of_day')] = hours.astype(np.float32)
    X[:, ML_FEATURES.index('is_weekend')] = (dow >= 5).astype(np.float32)
    X[:, ML_FEATURES.index('shared_device_accounts')] = df.groupby('merchant')['cc_num'].transform('nunique').values.astype(np.float32)
    X[:, ML_FEATURES.index('shared_recipient_accounts')] = X[:, ML_FEATURES.index('shared_device_accounts')]
    X[:, ML_FEATURES.index('mule_ring_score')] = df.groupby('cc_num')['city'].transform('nunique').values.astype(np.float32)
    mh = df.groupby('cc_num')['trans_date_trans_time'].transform(lambda x: pd.to_datetime(x).dt.hour.median()).values
    X[:, ML_FEATURES.index('hour_deviation')] = np.abs(hours - mh).astype(np.float32)
    gmean = df.groupby('cc_num')['amt'].transform('mean').values
    gstd = df.groupby('cc_num')['amt'].transform('std').fillna(1.0).values
    X[:, ML_FEATURES.index('amount_zscore')] = np.where(gstd > 0, (amt - gmean) / gstd, 0).astype(np.float32)
    X[:, ML_FEATURES.index('velocity_deviation')] = gc / 24.0
    seen = df.groupby(['cc_num','merchant']).cumcount().values + 1
    X[:, ML_FEATURES.index('recipient_novelty')] = (1.0 / seen).astype(np.float32)
    sg = df.groupby('cc_num')['unix_time'].transform(lambda x: x.diff().fillna(0).expanding().std()).values / 3600.0
    X[:, ML_FEATURES.index('txn_regularity')] = np.clip(sg, 0, 100).astype(np.float32)
    return X


def eval_metrics(y_true, y_prob):
    """Compute comprehensive metrics."""
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
    print("TEMPORAL EVALUATION: Train Jan-Mar → Test Apr-Sep")
    print("=" * 70)
    
    # Load all data
    df = pd.concat([pd.read_csv('data/kaggle_fraud/fraudTrain.csv'),
                     pd.read_csv('data/kaggle_fraud/fraudTest.csv')], ignore_index=True)
    df = df.drop(columns=['first','last','street','state','zip','dob','job','trans_num','Unnamed: 0'], errors='ignore')
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['month'] = df['trans_date_trans_time'].dt.month
    df['year'] = df['trans_date_trans_time'].dt.year
    
    # Create temporal groups
    # Train: all of 2019 + Jan-Mar 2020 (gives model full year of history)
    # Test: Apr-Sep 2020 (the months the user wants)
    train_mask = (df['year'] == 2019) | (df['month'] <= 3)
    test_mask = (df['year'] == 2020) & (df['month'] >= 4) & (df['month'] <= 9)
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"\nTrain: {len(df_train):,} rows (2019 all + Jan-Mar 2020)")
    print(f"  Fraud: {int(df_train.is_fraud.sum()):,} ({df_train.is_fraud.mean()*100:.3f}%)")
    print(f"Test: {len(df_test):,} rows (Apr-Sep 2020)")
    print(f"  Fraud: {int(df_test.is_fraud.sum()):,} ({df_test.is_fraud.mean()*100:.3f}%)")
    
    # Build features
    print("\nBuilding features...")
    t0 = time.time()
    X_train = build_features(df_train)
    y_train = df_train['is_fraud'].values.astype(int)
    X_test = build_features(df_test)
    y_test = df_test['is_fraud'].values.astype(int)
    print(f"  Features built in {time.time()-t0:.1f}s")
    
    # Train model
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
    print(f"  Trained in {time.time()-t0:.1f}s (best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'})")
    
    # Overall test metrics
    prob = model.predict_proba(Xec)[:, 1]
    overall = eval_metrics(y_test, prob)
    print(f"\n{'='*70}")
    print(f"OVERALL (Apr-Sep 2020): {len(y_test):,} rows, {int(y_test.sum()):,} fraud")
    print(f"  ROC-AUC: {overall['roc_auc']:.4f}")
    print(f"  PR-AUC: {overall['pr_auc']:.4f}")
    print(f"  Recall@0.05%FPR: {overall['r05ppct']:.4f}")
    print(f"  Recall@0.1%FPR: {overall['r01pct']:.4f}")
    print(f"  Recall@1%FPR: {overall['r1pct']:.4f}")
    
    # Month-by-month breakdown
    print(f"\n{'='*70}")
    print(f"MONTH-BY-MONTH BREAKDOWN (Apr-Sep 2020)")
    print(f"{'='*70}")
    print(f"{'Month':<12} {'Rows':>8} {'Fraud':>7} {'Prev%':>7} {'ROC':>7} {'PR':>7} {'R@.05%':>7} {'R@.1%':>7} {'R@1%':>7}")
    print("-" * 80)
    
    month_results = []
    month_names = {4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September'}
    for m in range(4, 10):
        mask = df_test['month'] == m
        if mask.sum() == 0:
            continue
        y_m = y_test[mask.values]
        p_m = prob[mask.values]
        if y_m.sum() == 0:
            print(f"{month_names[m]:<12} {mask.sum():>8,} {0:>7} {0:>7.3f} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7}")
            continue
        mets = eval_metrics(y_m, p_m)
        prev = y_m.mean() * 100
        print(f"{month_names[m]:<12} {mask.sum():>8,} {int(y_m.sum()):>7,} {prev:>7.3f} {mets['roc_auc']:>7.4f} {mets['pr_auc']:>7.4f} {mets['r05ppct']:>7.4f} {mets['r01pct']:>7.4f} {mets['r1pct']:>7.4f}")
        month_results.append({'month': month_names[m], 'n': int(mask.sum()), 'fraud': int(y_m.sum()),
                              'prevalence': round(prev, 3), **mets})
    
    # Also evaluate on Jan-Mar 2020 (same year, earlier months) for comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON: Jan-Mar 2020 (same year, earlier months)")
    print(f"{'='*70}")
    jan_mar_mask = (df['year'] == 2020) & (df['month'] <= 3)
    df_jm = df[jan_mar_mask].copy()
    X_jm = build_features(df_jm)
    y_jm = df_jm['is_fraud'].values.astype(int)
    X_jm_c = np.nan_to_num(imp.transform(X_jm), nan=0.0, posinf=10.0, neginf=-10.0)
    p_jm = model.predict_proba(X_jm_c)[:, 1]
    jm_mets = eval_metrics(y_jm, p_jm)
    print(f"  Jan-Mar 2020: {len(y_jm):,} rows, {int(y_jm.sum()):,} fraud")
    print(f"  ROC-AUC: {jm_mets['roc_auc']:.4f}  PR-AUC: {jm_mets['pr_auc']:.4f}")
    print(f"  Recall@0.05%FPR: {jm_mets['r05ppct']:.4f}  Recall@1%FPR: {jm_mets['r1pct']:.4f}")
    
    # Also evaluate on 2019 (training year, in-sample check)
    print(f"\n{'='*70}")
    print(f"IN-SAMPLE: 2019 (training year — should be high)")
    print(f"{'='*70}")
    mask_2019 = df['year'] == 2019
    df_19 = df[mask_2019].copy()
    X_19 = build_features(df_19)
    y_19 = df_19['is_fraud'].values.astype(int)
    X_19_c = np.nan_to_num(imp.transform(X_19), nan=0.0, posinf=10.0, neginf=-10.0)
    p_19 = model.predict_proba(X_19_c)[:, 1]
    m_19 = eval_metrics(y_19, p_19)
    print(f"  2019: {len(y_19):,} rows, {int(y_19.sum()):,} fraud")
    print(f"  ROC-AUC: {m_19['roc_auc']:.4f}  PR-AUC: {m_19['pr_auc']:.4f}")
    print(f"  Recall@0.05%FPR: {m_19['r05ppct']:.4f}  Recall@1%FPR: {m_19['r1pct']:.4f}")
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: Temporal Generalization")
    print(f"{'='*70}")
    print(f"{'Period':<25} {'N':>10} {'Fraud':>7} {'ROC':>7} {'R@.05%':>7} {'R@1%':>7}")
    print("-" * 65)
    print(f"{'2019 (in-sample)':<25} {len(y_19):>10,} {int(y_19.sum()):>7,} {m_19['roc_auc']:>7.4f} {m_19['r05ppct']:>7.4f} {m_19['r1pct']:>7.4f}")
    print(f"{'Jan-Mar 2020':<25} {len(y_jm):>10,} {int(y_jm.sum()):>7,} {jm_mets['roc_auc']:>7.4f} {jm_mets['r05ppct']:>7.4f} {jm_mets['r1pct']:>7.4f}")
    print(f"{'Apr-Sep 2020 (OOD)':<25} {len(y_test):>10,} {int(y_test.sum()):>7,} {overall['roc_auc']:>7.4f} {overall['r05ppct']:>7.4f} {overall['r1pct']:>7.4f}")
    
    # Save results
    results = {
        'train_period': '2019 all + Jan-Mar 2020',
        'test_period': 'Apr-Sep 2020',
        'train_n': len(df_train), 'train_fraud': int(df_train.is_fraud.sum()),
        'test_n': len(df_test), 'test_fraud': int(df_test.is_fraud.sum()),
        'overall': overall,
        'jan_mar_2020': jm_mets,
        'in_sample_2019': m_19,
        'monthly': month_results,
    }
    with open(OUT / 'temporal_apr_sep_eval.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT / 'temporal_apr_sep_eval.json'}")


if __name__ == '__main__':
    main()
