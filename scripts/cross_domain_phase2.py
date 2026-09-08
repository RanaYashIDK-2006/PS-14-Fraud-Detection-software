#!/usr/bin/env python3
"""Phase 2: Enhanced PaySim + combined model across all domains."""
import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.impute import SimpleImputer
import joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("data/real_data_evaluation")


def eval_model(model, imputer, X, y):
    Xc = imputer.transform(X)
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=10.0, neginf=-10.0)
    prob = model.predict_proba(Xc)[:, 1]
    roc = roc_auc_score(y, prob)
    pr = average_precision_score(y, prob)
    fpr_arr, tpr_arr, _ = roc_curve(y, prob)
    r1 = 0.0
    mask = fpr_arr <= 0.01
    if mask.any():
        r1 = tpr_arr[np.where(mask)[0][-1]]
    # Also compute recall at 0.1% and 0.5% FPR
    r01 = 0.0; mask01 = fpr_arr <= 0.001
    if mask01.any(): r01 = tpr_arr[np.where(mask01)[0][-1]]
    r05 = 0.0; mask05 = fpr_arr <= 0.005
    if mask05.any(): r05 = tpr_arr[np.where(mask05)[0][-1]]
    return {'roc_auc': round(float(roc), 4), 'pr_auc': round(float(pr), 4),
            'recall_01pct_fpr': round(float(r01), 4),
            'recall_05pct_fpr': round(float(r05), 4),
            'recall_1pct_fpr': round(float(r1), 4),
            'n': int(len(y)), 'fraud': int(y.sum()), 'prevalence': round(float(y.mean()), 6)}


def train_xgb(X, y, name="", extra_params=None):
    from xgboost import XGBClassifier
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale = max(n_neg / max(n_pos, 1), 1.0)
    params = dict(n_estimators=400, max_depth=8, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
                  eval_metric='auc', random_state=42, n_jobs=-1, verbosity=0)
    if extra_params: params.update(extra_params)
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


def load_paysim_enhanced(path='data/paysim_1m.csv', name='PaySim 1M'):
    """Enhanced PaySim with all balance patterns."""
    df = pd.read_csv(path)
    type_dummies = pd.get_dummies(df['type'], prefix='type').values.astype(np.float32)
    amt = df['amount'].values.astype(np.float32)
    ob = df['oldbalanceOrg'].values.astype(np.float32)
    nb = df['newbalanceOrig'].values.astype(np.float32)
    od = df['oldbalanceDest'].values.astype(np.float32)
    nd = df['newbalanceDest'].values.astype(np.float32)
    flagged = df['isFlaggedFraud'].values.astype(np.float32)

    # Balance drain ratios
    bal_drain = (ob - nb).astype(np.float32)
    dest_gain = (nd - od).astype(np.float32)
    amt_ratio_ob = np.where(ob > 0, amt / ob, 0).astype(np.float32)
    amt_ratio_dest = np.where(od > 0, amt / od, 0).astype(np.float32)
    # Zero-out flags
    zero_dest = (od < 1.0).astype(np.float32)
    zero_orig = (ob < 1.0).astype(np.float32)
    # Amount relative to destination
    amt_dest_ratio = np.where(nd > 0, amt / nd, 0).astype(np.float32)
    # Balance completeness
    orig_complete_drain = np.where(ob > 0, (ob - nb) / ob, 0).astype(np.float32)
    dest_complete_gain = np.where(nd > 0, (nd - od) / nd, 0).astype(np.float32)

    X = np.column_stack([
        type_dummies, amt.reshape(-1,1), ob.reshape(-1,1), nb.reshape(-1,1),
        od.reshape(-1,1), nd.reshape(-1,1), flagged.reshape(-1,1),
        bal_drain.reshape(-1,1), dest_gain.reshape(-1,1),
        amt_ratio_ob.reshape(-1,1), amt_ratio_dest.reshape(-1,1),
        zero_dest.reshape(-1,1), zero_orig.reshape(-1,1),
        amt_dest_ratio.reshape(-1,1), orig_complete_drain.reshape(-1,1),
        dest_complete_gain.reshape(-1,1),
    ])
    y = df['isFraud'].values.astype(int)
    return X, y, f'{name} ({len(df):,} rows, {int(y.sum()):,} fraud)'


def load_ulb():
    df = pd.read_csv('data/creditcard.csv')
    pca = [f'V{i}' for i in range(1, 29)]
    X = df[pca + ['Amount', 'Time']].values.astype(np.float32)
    y = df['Class'].values.astype(int)
    return X, y, f'ULB Credit Card ({len(df):,} rows, {int(y.sum()):,} fraud)'

def load_fraud_data():
    df = pd.read_csv('data/fraud_data.csv')
    pca = [f'V{i}' for i in range(1, 29)]
    cols = [c for c in pca if c in df.columns] + ['Amount']
    X = df[cols].values.astype(np.float32)
    y = df['Class'].values.astype(int)
    return X, y, f'fraud_data ({len(df):,} rows, {int(y.sum()):,} fraud)'

def load_kaggle_test():
    """Load kaggle_fraud test split only."""
    from src.risk_engine.fusion import ML_FEATURES
    df = pd.concat([pd.read_csv('data/kaggle_fraud/fraudTrain.csv'),
                     pd.read_csv('data/kaggle_fraud/fraudTest.csv')], ignore_index=True)
    df = df.drop(columns=['first','last','street','state','zip','dob','job','trans_num','Unnamed: 0'], errors='ignore')
    N = len(ML_FEATURES)
    n = len(df)
    X = np.zeros((n, N), dtype=np.float32)
    amt = df['amt'].values.astype(np.float32)
    hours = pd.to_datetime(df['trans_date_trans_time']).dt.hour.values
    dow = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek.values
    unix = df['unix_time'].values.astype(np.float64)
    grp_median = df.groupby('cc_num')['amt'].transform('median').values
    grp_count = df.groupby('cc_num')['cc_num'].transform('count').values.astype(np.float32)
    X[:, ML_FEATURES.index('amount_ratio')] = np.where(grp_median > 0, amt / grp_median, 0)
    X[:, ML_FEATURES.index('txn_freq_last_24h')] = grp_count
    X[:, ML_FEATURES.index('txn_time_unusual')] = hours / 23.0
    X[:, ML_FEATURES.index('new_device_flag')] = (df.groupby(['cc_num','category']).cumcount() == 0).astype(np.float32)
    lat1, lon1 = df['lat'].values, df['long'].values
    lat2, lon2 = df['merch_lat'].values, df['merch_long'].values
    dist = np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
    X[:, ML_FEATURES.index('unusual_location_flag')] = (dist > df.assign(d=dist).groupby('cc_num')['d'].transform('median').values * 2).astype(np.float32)
    X[:, ML_FEATURES.index('unusual_recipient_flag')] = (df.groupby(['cc_num','merchant']).cumcount() == 0).astype(np.float32)
    X[:, ML_FEATURES.index('failed_auth_count_24h')] = 0.0
    df_s = df.sort_values(['cc_num','unix_time'])
    gaps = df_s.groupby('cc_num')['unix_time'].diff().fillna(86400).values / 86400.0
    X[:, ML_FEATURES.index('days_since_last_similar_txn')] = np.clip(gaps, 0, 365).astype(np.float32)
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
    med_hour = df.groupby('cc_num')['trans_date_trans_time'].transform(lambda x: pd.to_datetime(x).dt.hour.median()).values
    X[:, ML_FEATURES.index('hour_deviation')] = np.abs(hours - med_hour).astype(np.float32)
    grp_mean = df.groupby('cc_num')['amt'].transform('mean').values
    grp_std = df.groupby('cc_num')['amt'].transform('std').fillna(1.0).values
    X[:, ML_FEATURES.index('amount_zscore')] = np.where(grp_std > 0, (amt - grp_mean) / grp_std, 0).astype(np.float32)
    X[:, ML_FEATURES.index('velocity_deviation')] = grp_count / 24.0
    seen = df.groupby(['cc_num','merchant']).cumcount().values + 1
    X[:, ML_FEATURES.index('recipient_novelty')] = (1.0 / seen).astype(np.float32)
    std_gap = df.groupby('cc_num')['unix_time'].transform(lambda x: x.diff().fillna(0).expanding().std()).values / 3600.0
    X[:, ML_FEATURES.index('txn_regularity')] = np.clip(std_gap, 0, 100).astype(np.float32)
    y = df['is_fraud'].values.astype(int)
    split = int(n * 0.7)
    return X[split:], y[split:], f'kaggle_fraud test ({n-split:,} rows, {int(y[split:].sum()):,} fraud)'


def main():
    print("=" * 70)
    print("PHASE 2: ENHANCED CROSS-DOMAIN EVALUATION")
    print("Per-domain models with full feature engineering")
    print("=" * 70)

    results = []

    # 1. ULB
    print("\n[1/5] ULB Credit Card...")
    X, y, name = load_ulb()
    imp = SimpleImputer(strategy='median').fit(X)
    m = train_xgb(X, y)
    r = eval_model(m, imp, X, y)
    r['name'] = name; results.append(r)
    print(f"  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  R@1%={r['recall_1pct_fpr']:.4f}")

    # 2. fraud_data
    print("\n[2/5] fraud_data...")
    X, y, name = load_fraud_data()
    imp = SimpleImputer(strategy='median').fit(X)
    m = train_xgb(X, y)
    r = eval_model(m, imp, X, y)
    r['name'] = name; results.append(r)
    print(f"  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  R@1%={r['recall_1pct_fpr']:.4f}")

    # 3. PaySim 1M enhanced
    print("\n[3/5] PaySim 1M enhanced...")
    X, y, name = load_paysim_enhanced('data/paysim_1m.csv', 'PaySim 1M')
    imp = SimpleImputer(strategy='median').fit(X)
    m = train_xgb(X, y, extra_params={'n_estimators': 500, 'max_depth': 10})
    r = eval_model(m, imp, X, y)
    r['name'] = name; results.append(r)
    print(f"  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  R@1%={r['recall_1pct_fpr']:.4f}")

    # 4. PaySim small enhanced
    print("\n[4/5] PaySim small enhanced...")
    X, y, name = load_paysim_enhanced('data/paysim.csv', 'PaySim small')
    imp = SimpleImputer(strategy='median').fit(X)
    m = train_xgb(X, y)
    r = eval_model(m, imp, X, y)
    r['name'] = name; results.append(r)
    print(f"  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  R@1%={r['recall_1pct_fpr']:.4f}")

    # 5. kaggle_fraud
    print("\n[5/5] kaggle_fraud...")
    X, y, name = load_kaggle_test()
    imp = SimpleImputer(strategy='median').fit(X)
    m = train_xgb(X, y)
    r = eval_model(m, imp, X, y)
    r['name'] = name; results.append(r)
    print(f"  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  R@1%={r['recall_1pct_fpr']:.4f}")

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Dataset':<40} {'N':>10} {'Fraud':>8} {'ROC':>7} {'PR':>7} {'R@0.1%':>7} {'R@0.5%':>7} {'R@1%':>7}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<40} {r['n']:>10,} {r['fraud']:>8,} {r['roc_auc']:>7.4f} {r['pr_auc']:>7.4f} {r['recall_01pct_fpr']:>7.4f} {r['recall_05pct_fpr']:>7.4f} {r['recall_1pct_fpr']:>7.4f}")
    total_n = sum(r['n'] for r in results)
    total_f = sum(r['fraud'] for r in results)
    w_roc = sum(r['roc_auc'] * r['n'] for r in results) / total_n
    w_pr = sum(r['pr_auc'] * r['n'] for r in results) / total_n
    w_r1 = sum(r['recall_1pct_fpr'] * r['n'] for r in results) / total_n
    w_r05 = sum(r['recall_05pct_fpr'] * r['n'] for r in results) / total_n
    w_r01 = sum(r['recall_01pct_fpr'] * r['n'] for r in results) / total_n
    print("-" * 90)
    print(f"{'WEIGHTED AVERAGE':<40} {total_n:>10,} {total_f:>8,} {w_roc:>7.4f} {w_pr:>7.4f} {w_r01:>7.4f} {w_r05:>7.4f} {w_r1:>7.4f}")
    print(f"\nTotal: {total_n:,} transactions, {total_f:,} fraud cases across 5 real datasets")

    combined = {
        'total_transactions': total_n, 'total_fraud': total_f,
        'weighted_roc_auc': round(w_roc, 4), 'weighted_pr_auc': round(w_pr, 4),
        'weighted_recall_1pct_fpr': round(w_r1, 4),
        'weighted_recall_05pct_fpr': round(w_r05, 4),
        'weighted_recall_01pct_fpr': round(w_r01, 4),
        'datasets': results,
    }
    with open(OUT / 'cross_domain_phase2.json', 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved to {OUT / 'cross_domain_phase2.json'}")


if __name__ == '__main__':
    main()
