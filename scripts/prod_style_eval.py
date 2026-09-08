#!/usr/bin/env python3
"""Production-style eval: train on first 70%, test on last 30% (temporal-like)."""
import sys, warnings, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def train_eval(df, X, y, name):
    split = int(len(df) * 0.7)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    imp = SimpleImputer(strategy='median').fit(Xtr)
    Xtc = np.nan_to_num(imp.transform(Xtr), nan=0.0, posinf=10.0, neginf=-10.0)
    Xec = np.nan_to_num(imp.transform(Xte), nan=0.0, posinf=10.0, neginf=-10.0)
    n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
    scale = max(n_neg / max(n_pos, 1), 1.0)
    m = XGBClassifier(n_estimators=600, max_depth=10, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
                      eval_metric='auc', random_state=42, n_jobs=-1, verbosity=0,
                      min_child_weight=3, gamma=0.05, reg_alpha=0.1)
    m.fit(Xtc, ytr)
    prob = m.predict_proba(Xte)[:, 1]
    roc = roc_auc_score(yte, prob)
    pr = average_precision_score(yte, prob)
    fpr_arr, tpr_arr, _ = roc_curve(yte, prob)
    mets = {}
    for tf, lab in [(0.0001,'r01ppct'),(0.0005,'r05ppct'),(0.001,'r01pct'),(0.005,'r05pct'),(0.01,'r1pct')]:
        mask = fpr_arr <= tf
        mets[lab] = round(float(tpr_arr[np.where(mask)[0][-1]]), 4) if mask.any() else 0.0
    print(f"  {name}: ROC={roc:.4f} PR={pr:.4f} R@.05%={mets['r05ppct']:.4f} R@.1%={mets['r01pct']:.4f} R@1%={mets['r1pct']:.4f}")
    return {'name': name, 'n_test': int(len(yte)), 'fraud_test': int(yte.sum()),
            'roc_auc': round(float(roc), 4), 'pr_auc': round(float(pr), 4), **mets}

def paysim_features(df):
    td = pd.get_dummies(df['type'], prefix='type').values.astype(np.float32)
    amt = df['amount'].values.astype(np.float32)
    ob = df['oldbalanceOrg'].values.astype(np.float32)
    nb = df['newbalanceOrig'].values.astype(np.float32)
    od = df['oldbalanceDest'].values.astype(np.float32)
    nd = df['newbalanceDest'].values.astype(np.float32)
    fl = df['isFlaggedFraud'].values.astype(np.float32)
    bd = (ob - nb); dg = (nd - od)
    ar_o = np.where(ob > 0, amt / ob, 0); ar_d = np.where(od > 0, amt / od, 0)
    zd = (od < 1.0); zo = (ob < 1.0)
    od2 = np.where(ob > 0, (ob - nb) / ob, 0)
    al = np.log1p(amt); ol = np.log1p(ob)
    obr = np.where(ob > 0, nb / ob, 0); dbr = np.where(od > 0, nd / od, 0)
    dc = np.where(od > 0, (nd - od) / (od + 1), 0)
    oc = np.where(ob > 0, (ob - nb) / (ob + 1), 0)
    return np.column_stack([td, amt.reshape(-1,1), ob.reshape(-1,1), nb.reshape(-1,1),
        od.reshape(-1,1), nd.reshape(-1,1), fl.reshape(-1,1),
        bd.reshape(-1,1), dg.reshape(-1,1), ar_o.reshape(-1,1),
        ar_d.reshape(-1,1), zd.reshape(-1,1), zo.reshape(-1,1),
        od2.reshape(-1,1), al.reshape(-1,1), ol.reshape(-1,1),
        obr.reshape(-1,1), dbr.reshape(-1,1), dc.reshape(-1,1), oc.reshape(-1,1)])

def main():
    results = []
    print("PaySim 1M...")
    df = pd.read_csv('data/paysim_1m.csv')
    results.append(train_eval(df, paysim_features(df), df['isFraud'].values.astype(int), 'PaySim 1M'))
    
    print("PaySim small...")
    df = pd.read_csv('data/paysim.csv')
    results.append(train_eval(df, paysim_features(df), df['isFraud'].values.astype(int), 'PaySim small'))
    
    print("ULB...")
    df = pd.read_csv('data/creditcard.csv')
    pca = [f'V{i}' for i in range(1, 29)]
    results.append(train_eval(df, df[pca + ['Amount', 'Time']].values.astype(np.float32), df['Class'].values.astype(int), 'ULB'))
    
    print("fraud_data...")
    df = pd.read_csv('data/fraud_data.csv')
    pca = [f'V{i}' for i in range(1, 29)]
    cols = [c for c in pca if c in df.columns] + ['Amount']
    results.append(train_eval(df, df[cols].values.astype(np.float32), df['Class'].values.astype(int), 'fraud_data'))
    
    print("kaggle_fraud...")
    from src.risk_engine.fusion import ML_FEATURES
    df = pd.concat([pd.read_csv('data/kaggle_fraud/fraudTrain.csv'), pd.read_csv('data/kaggle_fraud/fraudTest.csv')], ignore_index=True)
    df = df.drop(columns=['first','last','street','state','zip','dob','job','trans_num','Unnamed: 0'], errors='ignore')
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
    lat1, lon1 = df['lat'].values, df['long'].values; lat2, lon2 = df['merch_lat'].values, df['merch_long'].values
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
    results.append(train_eval(df, X, df['is_fraud'].values.astype(int), 'kaggle_fraud'))
    
    print(f"\n{'='*95}")
    print(f"{'Dataset':<20} {'N_test':>8} {'Fraud':>7} {'ROC':>7} {'PR':>7} {'R@.05%':>7} {'R@.1%':>7} {'R@1%':>7}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<20} {r['n_test']:>8,} {r['fraud_test']:>7,} {r['roc_auc']:>7.4f} {r['pr_auc']:>7.4f} {r['r05ppct']:>7.4f} {r['r01pct']:>7.4f} {r['r1pct']:>7.4f}")
    tn = sum(r['n_test'] for r in results)
    wr = sum(r['roc_auc'] * r['n_test'] for r in results) / tn
    wr005 = sum(r['r05ppct'] * r['n_test'] for r in results) / tn
    wr1 = sum(r['r1pct'] * r['n_test'] for r in results) / tn
    print("-" * 95)
    print(f"{'WEIGHTED':<20} {tn:>8,} {'':>7} {wr:>7.4f} {'':>7} {wr005:>7.4f} {'':>7} {wr1:>7.4f}")
    print(f"\nTarget: ROC-AUC >= 0.995, R@0.05%FPR > 0.5")
    print(f"Actual: ROC-AUC = {wr:.4f}, R@0.05%FPR = {wr005:.4f}")
    with open('data/real_data_evaluation/prod_style_eval.json', 'w') as f:
        json.dump({'datasets': results, 'weighted_roc': round(wr, 4), 'weighted_r005': round(wr005, 4)}, f, indent=2)

if __name__ == '__main__':
    main()
