#!/usr/bin/env python3
"""Train optimized model targeting 99.5% ROC-AUC and <0.05% FPR."""
import sys, warnings, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
warnings.filterwarnings('ignore')

OUT = Path("data/real_data_evaluation")
ARTIFACTS = Path("models/artifacts")


def build_paysim_features(df):
    """Build comprehensive feature set for PaySim."""
    type_dummies = pd.get_dummies(df['type'], prefix='type').values.astype(np.float32)
    amt = df['amount'].values.astype(np.float32)
    ob = df['oldbalanceOrg'].values.astype(np.float32)
    nb = df['newbalanceOrig'].values.astype(np.float32)
    od = df['oldbalanceDest'].values.astype(np.float32)
    nd = df['newbalanceDest'].values.astype(np.float32)
    flagged = df['isFlaggedFraud'].values.astype(np.float32)
    
    # Balance features
    bal_drain = (ob - nb).astype(np.float32)
    dest_gain = (nd - od).astype(np.float32)
    amt_ratio_ob = np.where(ob > 0, amt / ob, 0).astype(np.float32)
    amt_ratio_dest = np.where(od > 0, amt / od, 0).astype(np.float32)
    zero_dest = (od < 1.0).astype(np.float32)
    zero_orig = (ob < 1.0).astype(np.float32)
    amt_dest_ratio = np.where(nd > 0, amt / nd, 0).astype(np.float32)
    orig_complete_drain = np.where(ob > 0, (ob - nb) / ob, 0).astype(np.float32)
    dest_complete_gain = np.where(nd > 0, (nd - od) / nd, 0).astype(np.float32)
    
    # Log transforms
    amt_log = np.log1p(amt).astype(np.float32)
    ob_log = np.log1p(ob).astype(np.float32)
    
    # Balance ratios
    orig_balance_ratio = np.where(ob > 0, nb / ob, 0).astype(np.float32)
    dest_balance_ratio = np.where(od > 0, nd / od, 0).astype(np.float32)
    
    # Amount anomalies
    amt_is_round = (amt % 100 == 0).astype(np.float32)
    large_amt = (amt > amt.mean() + 2 * amt.std()).astype(np.float32)
    
    # Per-type amount stats (non-leaking: only uses amount, not labels)
    type_amt_means = df.groupby('type')['amount'].mean()
    type_amt_stds = df.groupby('type')['amount'].std().fillna(1.0)
    amt_zscore_type = ((amt - df['type'].map(type_amt_means).values) / (df['type'].map(type_amt_stds).values + 1e-6)).astype(np.float32)
    
    # NEW: Balance anomaly features
    ob_mean = df.groupby('type')['oldbalanceOrg'].transform('mean').values.astype(np.float32)
    ob_std = df.groupby('type')['oldbalanceOrg'].transform('std').fillna(1.0).values.astype(np.float32)
    ob_zscore = np.where(ob_std > 0, (ob - ob_mean) / ob_std, 0).astype(np.float32)
    
    # NEW: Amount as fraction of balance
    amt_frac_balance = np.where(ob > 0, amt / (ob + amt), 0).astype(np.float32)
    
    # NEW: Destination balance change
    dest_change_ratio = np.where(od > 0, (nd - od) / (od + 1), 0).astype(np.float32)
    
    # NEW: Origin balance change
    orig_change_ratio = np.where(ob > 0, (ob - nb) / (ob + 1), 0).astype(np.float32)
    
    X = np.column_stack([
        type_dummies,                    # 5
        amt.reshape(-1,1),               # 6
        ob.reshape(-1,1),                # 7
        nb.reshape(-1,1),                # 8
        od.reshape(-1,1),                # 9
        nd.reshape(-1,1),                # 10
        flagged.reshape(-1,1),           # 11
        bal_drain.reshape(-1,1),         # 12
        dest_gain.reshape(-1,1),         # 13
        amt_ratio_ob.reshape(-1,1),      # 14
        amt_ratio_dest.reshape(-1,1),    # 15
        zero_dest.reshape(-1,1),         # 16
        zero_orig.reshape(-1,1),         # 17
        amt_dest_ratio.reshape(-1,1),    # 18
        orig_complete_drain.reshape(-1,1),  # 19
        dest_complete_gain.reshape(-1,1),   # 20
        amt_log.reshape(-1,1),           # 21
        ob_log.reshape(-1,1),            # 22
        orig_balance_ratio.reshape(-1,1),   # 23
        dest_balance_ratio.reshape(-1,1),   # 24
        amt_is_round.reshape(-1,1),      # 25
        large_amt.reshape(-1,1),         # 26
        amt_zscore_type.reshape(-1,1),   # 27
        ob_zscore.reshape(-1,1),         # 29
        amt_frac_balance.reshape(-1,1),  # 30
        dest_change_ratio.reshape(-1,1), # 31
        orig_change_ratio.reshape(-1,1), # 32
    ])
    return X


def build_ulb_features(df):
    """ULB: all 28 PCA + Amount + Time."""
    pca = [f'V{i}' for i in range(1, 29)]
    return df[pca + ['Amount', 'Time']].values.astype(np.float32)


def build_fraud_data_features(df):
    """fraud_data: all 28 PCA + Amount."""
    pca = [f'V{i}' for i in range(1, 29)]
    cols = [c for c in pca if c in df.columns] + ['Amount']
    return df[cols].values.astype(np.float32)


def build_kaggle_features(df):
    """kaggle_fraud: 21 per-card features."""
    from src.risk_engine.fusion import ML_FEATURES
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
    return X


def train_and_eval(X, y, name, test_size=0.3, extra_params=None):
    """Train XGBoost and evaluate with comprehensive metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    imp = SimpleImputer(strategy='median').fit(X_train)
    X_train_c = np.nan_to_num(imp.transform(X_train), nan=0.0, posinf=10.0, neginf=-10.0)
    X_test_c = np.nan_to_num(imp.transform(X_test), nan=0.0, posinf=10.0, neginf=-10.0)
    
    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    scale = max(n_neg / max(n_pos, 1), 1.0)
    params = dict(n_estimators=600, max_depth=10, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
                  eval_metric='auc', random_state=42, n_jobs=-1, verbosity=0,
                  min_child_weight=5, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0)
    if extra_params: params.update(extra_params)
    
    model = XGBClassifier(**params)
    model.fit(X_train_c, y_train)
    
    prob = model.predict_proba(X_test_c)[:, 1]
    roc = roc_auc_score(y_test, prob)
    pr = average_precision_score(y_test, prob)
    
    fpr_arr, tpr_arr, thresholds = roc_curve(y_test, prob)
    metrics = {}
    for target_fpr, label in [(0.0001, 'r01ppct'), (0.0005, 'r05ppct'), (0.001, 'r01pct'), (0.005, 'r05pct'), (0.01, 'r1pct')]:
        mask = fpr_arr <= target_fpr
        if mask.any():
            idx = np.where(mask)[0][-1]
            metrics[label] = round(float(tpr_arr[idx]), 4)
            metrics[f'{label}_thresh'] = round(float(thresholds[idx]), 4)
        else:
            metrics[label] = 0.0
    
    # Find threshold for 0.05% FPR
    mask_005 = fpr_arr <= 0.0005
    if mask_005.any():
        idx_005 = np.where(mask_005)[0][-1]
        metrics['actual_fpr_at_005'] = round(float(fpr_arr[idx_005]), 6)
    
    result = {
        'name': name, 'n_test': int(len(y_test)), 'fraud_test': int(y_test.sum()),
        'roc_auc': round(float(roc), 4), 'pr_auc': round(float(pr), 4),
        **metrics,
        'model': model, 'imputer': imp,
    }
    return result


def main():
    print("=" * 70)
    print("OPTIMIZED TRAINING: Targeting 99.5% ROC-AUC, <0.05% FPR")
    print("=" * 70)
    
    results = []
    
    # 1. PaySim 1M (main target for improvement)
    print("\n[1/5] PaySim 1M (32 features)...")
    t0 = time.time()
    df = pd.read_csv('data/paysim_1m.csv')
    X = build_paysim_features(df)
    y = df['isFraud'].values.astype(int)
    r = train_and_eval(X, y, 'PaySim 1M')
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}  R@0.05%: {r['r05ppct']:.4f}  R@0.1%: {r['r01pct']:.4f}  R@1%: {r['r1pct']:.4f}  ({time.time()-t0:.1f}s)")
    results.append(r)
    
    # 2. PaySim small
    print("\n[2/5] PaySim small (32 features)...")
    t0 = time.time()
    df = pd.read_csv('data/paysim.csv')
    X = build_paysim_features(df)
    y = df['isFraud'].values.astype(int)
    r = train_and_eval(X, y, 'PaySim small')
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}  R@0.05%: {r['r05ppct']:.4f}  R@0.1%: {r['r01pct']:.4f}  R@1%: {r['r1pct']:.4f}  ({time.time()-t0:.1f}s)")
    results.append(r)
    
    # 3. ULB
    print("\n[3/5] ULB Credit Card (30 features)...")
    t0 = time.time()
    df = pd.read_csv('data/creditcard.csv')
    X = build_ulb_features(df)
    y = df['Class'].values.astype(int)
    r = train_and_eval(X, y, 'ULB Credit Card')
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}  R@0.05%: {r['r05ppct']:.4f}  R@0.1%: {r['r01pct']:.4f}  R@1%: {r['r1pct']:.4f}  ({time.time()-t0:.1f}s)")
    results.append(r)
    
    # 4. fraud_data
    print("\n[4/5] fraud_data (29 features)...")
    t0 = time.time()
    df = pd.read_csv('data/fraud_data.csv')
    X = build_fraud_data_features(df)
    y = df['Class'].values.astype(int)
    r = train_and_eval(X, y, 'fraud_data')
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}  R@0.05%: {r['r05ppct']:.4f}  R@0.1%: {r['r01pct']:.4f}  R@1%: {r['r1pct']:.4f}  ({time.time()-t0:.1f}s)")
    results.append(r)
    
    # 5. kaggle_fraud
    print("\n[5/5] kaggle_fraud (21 features)...")
    t0 = time.time()
    df = pd.concat([pd.read_csv('data/kaggle_fraud/fraudTrain.csv'),
                     pd.read_csv('data/kaggle_fraud/fraudTest.csv')], ignore_index=True)
    df = df.drop(columns=['first','last','street','state','zip','dob','job','trans_num','Unnamed: 0'], errors='ignore')
    X = build_kaggle_features(df)
    y = df['is_fraud'].values.astype(int)
    r = train_and_eval(X, y, 'kaggle_fraud')
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}  R@0.05%: {r['r05ppct']:.4f}  R@0.1%: {r['r01pct']:.4f}  R@1%: {r['r1pct']:.4f}  ({time.time()-t0:.1f}s)")
    results.append(r)
    
    # Summary
    print(f"\n{'='*95}")
    print(f"{'Dataset':<25} {'N_test':>8} {'Fraud':>7} {'ROC':>7} {'PR':>7} {'R@.05%':>7} {'R@.1%':>7} {'R@.5%':>7} {'R@1%':>7}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<25} {r['n_test']:>8,} {r['fraud_test']:>7,} {r['roc_auc']:>7.4f} {r['pr_auc']:>7.4f} {r['r05ppct']:>7.4f} {r['r01pct']:>7.4f} {r['r05pct']:>7.4f} {r['r1pct']:>7.4f}")
    total_n = sum(r['n_test'] for r in results)
    total_f = sum(r['fraud_test'] for r in results)
    w_roc = sum(r['roc_auc'] * r['n_test'] for r in results) / total_n
    w_pr = sum(r['pr_auc'] * r['n_test'] for r in results) / total_n
    w_r005 = sum(r['r05ppct'] * r['n_test'] for r in results) / total_n
    w_r1 = sum(r['r1pct'] * r['n_test'] for r in results) / total_n
    print("-" * 95)
    print(f"{'WEIGHTED AVERAGE':<25} {total_n:>8,} {total_f:>7,} {w_roc:>7.4f} {w_pr:>7.4f} {w_r005:>7.4f} {w_r1:>7.4f}")
    
    print(f"\nTarget: ROC-AUC >= 0.995, FPR < 0.0005")
    print(f"Actual: ROC-AUC = {w_roc:.4f}, R@0.05%FPR = {w_r005:.4f}")
    
    # Save
    save_results = [{k: v for k, v in r.items() if k not in ('model', 'imputer')} for r in results]
    combined = {
        'total_test': total_n, 'total_fraud_test': total_f,
        'weighted_roc_auc': round(w_roc, 4), 'weighted_pr_auc': round(w_pr, 4),
        'weighted_recall_005pct_fpr': round(w_r005, 4),
        'weighted_recall_1pct_fpr': round(w_r1, 4),
        'datasets': save_results,
    }
    with open(OUT / 'optimized_results.json', 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved to {OUT / 'optimized_results.json'}")


if __name__ == '__main__':
    main()
