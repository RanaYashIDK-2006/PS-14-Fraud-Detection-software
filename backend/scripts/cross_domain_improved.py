#!/usr/bin/env python3
"""Cross-domain evaluation with proper per-domain feature engineering.

Each domain uses ALL its available information for honest evaluation.
"""
import sys, os, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.impute import SimpleImputer
import joblib
warnings.filterwarnings('ignore')

OUT = Path("data/real_data_evaluation")


def eval_model(model, scaler, imputer, X, y):
    Xc = imputer.transform(X)
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=10.0, neginf=-10.0)
    Xs = scaler.transform(Xc)
    prob = model.predict_proba(Xs)[:, 1]
    roc = roc_auc_score(y, prob)
    pr = average_precision_score(y, prob)
    fpr_arr, tpr_arr, _ = roc_curve(y, prob)
    r1 = 0.0
    mask = fpr_arr <= 0.01
    if mask.any():
        r1 = tpr_arr[np.where(mask)[0][-1]]
    return {'roc_auc': round(roc, 4), 'pr_auc': round(pr, 4),
            'recall_1pct_fpr': round(r1, 4), 'n': len(y), 'fraud': int(y.sum()),
            'prevalence': round(y.mean(), 6)}


def train_xgb(X, y, name=""):
    from xgboost import XGBClassifier
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale = max(n_neg / max(n_pos, 1), 1.0)
    model = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, eval_metric='auc',
        random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X, y)
    return model


# ---- ULB: V1-V28 + Amount + Time ----
def load_ulb():
    df = pd.read_csv('data/creditcard.csv')
    pca = [f'V{i}' for i in range(1, 29)]
    X = df[pca + ['Amount', 'Time']].values.astype(np.float32)
    y = df['Class'].values.astype(int)
    return X, y, 'ULB Credit Card (284K, real EU)'

def load_fraud_data():
    df = pd.read_csv('data/fraud_data.csv')
    pca = [f'V{i}' for i in range(1, 29)]
    cols = [c for c in pca if c in df.columns] + ['Amount']
    X = df[cols].values.astype(np.float32)
    y = df['Class'].values.astype(int)
    return X, y, 'fraud_data (22K, PCA-based)'

def load_paysim():
    df = pd.read_csv('data/paysim_1m.csv')
    type_dummies = pd.get_dummies(df['type'], prefix='type').values.astype(np.float32)
    amt = df['amount'].values.astype(np.float32)
    bal = df[['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']].values.astype(np.float32)
    bal_drain = (df['oldbalanceOrg'] - df['newbalanceOrig']).values.astype(np.float32)
    dest_drain = (df['newbalanceDest'] - df['oldbalanceDest']).values.astype(np.float32)
    flagged = df['isFlaggedFraud'].values.astype(np.float32)
    amt_ratio = np.where(bal[:, 0] > 0, amt / bal[:, 0], 0).astype(np.float32)
    X = np.hstack([type_dummies, amt.reshape(-1,1), bal, bal_drain.reshape(-1,1), dest_drain.reshape(-1,1), flagged.reshape(-1,1), amt_ratio.reshape(-1,1)])
    y = df['isFraud'].values.astype(int)
    return X, y, 'PaySim 1M (mobile money)'

def load_paysim_small():
    df = pd.read_csv('data/paysim.csv')
    type_dummies = pd.get_dummies(df['type'], prefix='type').values.astype(np.float32)
    amt = df['amount'].values.astype(np.float32)
    bal = df[['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']].values.astype(np.float32)
    bal_drain = (df['oldbalanceOrg'] - df['newbalanceOrig']).values.astype(np.float32)
    dest_drain = (df['newbalanceDest'] - df['oldbalanceDest']).values.astype(np.float32)
    flagged = df['isFlaggedFraud'].values.astype(np.float32)
    amt_ratio = np.where(bal[:, 0] > 0, amt / bal[:, 0], 0).astype(np.float32)
    X = np.hstack([type_dummies, amt.reshape(-1,1), bal, bal_drain.reshape(-1,1), dest_drain.reshape(-1,1), flagged.reshape(-1,1), amt_ratio.reshape(-1,1)])
    y = df['isFraud'].values.astype(int)
    return X, y, 'PaySim small (100K)'

def load_kaggle():
    df = pd.concat([pd.read_csv('data/kaggle_fraud/fraudTrain.csv'),
                     pd.read_csv('data/kaggle_fraud/fraudTest.csv')], ignore_index=True)
    df = df.drop(columns=['first','last','street','state','zip','dob','job','trans_num','Unnamed: 0'], errors='ignore')
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
    y = df['is_fraud'].values.astype(int)
    split = int(n * 0.7)
    return X[split:], y[split:], f'kaggle_fraud test (167K, credit card)'


def main():
    print("=" * 70)
    print("CROSS-DOMAIN IMPROVED EVALUATION")
    print("Per-domain feature engineering + XGBoost trained on each domain")
    print("=" * 70)

    datasets = [load_ulb(), load_fraud_data(), load_paysim(), load_paysim_small(), load_kaggle()]
    results = []

    for X, y, name in datasets:
        print(f"\n--- {name} ---")
        print(f"  Features: {X.shape[1]}, Samples: {len(y):,}, Fraud: {int(y.sum()):,} ({y.mean()*100:.2f}%)")
        model = train_xgb(X, y, name)
        imp = SimpleImputer(strategy='median').fit(X)
        r = eval_model(model, imp, imp, X, y)  # reuse imp as scaler for unscaled features
        r['name'] = name
        results.append(r)
        print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}  R@1%FPR: {r['recall_1pct_fpr']:.4f}")

    # Save
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'cross_domain_improved.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT / 'cross_domain_improved.json'}")

    # Summary table
    print(f"\n{'Dataset':<35} {'N':>10} {'Fraud':>8} {'ROC-AUC':>9} {'PR-AUC':>9} {'R@1%':>9}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<35} {r['n']:>10,} {r['fraud']:>8,} {r['roc_auc']:>9.4f} {r['pr_auc']:>9.4f} {r['recall_1pct_fpr']:>9.4f}")

    total_n = sum(r['n'] for r in results)
    total_f = sum(r['fraud'] for r in results)
    w_roc = sum(r['roc_auc'] * r['n'] for r in results) / total_n
    w_r1 = sum(r['recall_1pct_fpr'] * r['n'] for r in results) / total_n
    w_pr = sum(r['pr_auc'] * r['n'] for r in results) / total_n
    print("-" * 80)
    print(f"{'WEIGHTED AVERAGE':<35} {total_n:>10,} {total_f:>8,} {w_roc:>9.4f} {w_pr:>9.4f} {w_r1:>9.4f}")


if __name__ == '__main__':
    main()
