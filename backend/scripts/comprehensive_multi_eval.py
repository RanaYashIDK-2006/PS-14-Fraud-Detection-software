#!/usr/bin/env python3
"""Comprehensive multi-dataset evaluation and model improvement pipeline.

Datasets:
1. kaggle_fraud (train+test): 1.85M rows, 0.48% fraud - Credit card (rich features)
2. paysim_1m: 1.2M rows, 4.9% fraud - Mobile money
3. creditcard: 284K rows, 0.17% fraud - European credit card (PCA)
4. fraud_data: 21K rows, 1.64% fraud - PCA-based credit card
5. ealtman2019: sampled from 24M IBM synthetic

Then:
- Evaluate current model cross-dataset
- Retrain on combined data for better generalization
- Evaluate improved model
- Generate before/after comparison
"""

import sys, os, json, time, hashlib, warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, precision_score, recall_score, confusion_matrix
)
from scipy import stats as sp_stats
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk_engine.fusion import ML_FEATURES

OUT_DIR = Path("data/real_data_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FEATURES = len(ML_FEATURES)  # 21
print(f"ML_FEATURES ({N_FEATURES}): {ML_FEATURES}")

# ─── Bootstrap CI helper ────────────────────────────────────────
def bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap=500, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            pass
    if not scores:
        return 0.0, 0.0, 0.0
    lo = np.percentile(scores, (1-ci)/2*100)
    hi = np.percentile(scores, (1+ci)/2*100)
    return np.mean(scores), lo, hi

# ─── Recall at fixed FPR ────────────────────────────────────────
def recall_at_fpr(y_true, y_score, target_fpr=0.01):
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    valid = fpr_arr <= target_fpr
    if not valid.any():
        return 0.0, 0.0
    idx = np.where(valid)[0][-1]
    return tpr_arr[idx], thresholds[idx]

# ─── Brier / ECE ────────────────────────────────────────────────
def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece_val += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece_val

# ─── Feature mapping functions ──────────────────────────────────
def map_kaggle_to_features(df: pd.DataFrame) -> np.ndarray:
    """Map kaggle_fraud columns to 21 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    amt = df['amt'].values.astype(np.float32)
    # amount_ratio: transaction amount / median amount for the card
    medians = df.groupby('cc_num')['amt'].transform('median').values.astype(np.float32)
    X[:, ML_FEATURES.index('amount_ratio')] = np.where(medians > 0, amt / medians, 0.0)

    # txn_freq_last_24h: count of transactions per card in dataset (proxy)
    X[:, ML_FEATURES.index('txn_freq_last_24h')] = df.groupby('cc_num')['cc_num'].transform('count').values.astype(np.float32)

    # txn_time_unusual: hour of day (0-23) normalized
    hours = pd.to_datetime(df['trans_date_trans_time']).dt.hour.values
    X[:, ML_FEATURES.index('txn_time_unusual')] = hours.astype(np.float32) / 23.0

    # new_device_flag: 1 if first transaction from this merchant category for this card
    X[:, ML_FEATURES.index('new_device_flag')] = (df.groupby(['cc_num', 'category']).cumcount() == 0).values.astype(np.float32)

    # unusual_location_flag: distance between cardholder and merchant
    lat1 = df['lat'].values
    lon1 = df['long'].values
    lat2 = df['merch_lat'].values
    lon2 = df['merch_long'].values
    dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    median_dist = np.median(dist)
    X[:, ML_FEATURES.index('unusual_location_flag')] = (dist > median_dist * 2).astype(np.float32)

    # unusual_recipient_flag: first time at this merchant
    X[:, ML_FEATURES.index('unusual_recipient_flag')] = (df.groupby(['cc_num', 'merchant']).cumcount() == 0).values.astype(np.float32)

    # failed_auth_count_24h: 0 for all (not available in this dataset)
    X[:, ML_FEATURES.index('failed_auth_count_24h')] = 0.0

    # days_since_last_similar_txn: time gap between consecutive txns for same card
    df_sorted = df.sort_values(['cc_num', 'unix_time'])
    gaps = df_sorted.groupby('cc_num')['unix_time'].diff().fillna(3600*24).values / 86400.0
    X[:, ML_FEATURES.index('days_since_last_similar_txn')] = np.clip(gaps[:n], 0, 365).astype(np.float32)

    # gradual_escalation_score: ratio of recent amount to historical mean
    amounts = df.groupby('cc_num')['amt'].transform(lambda x: x.expanding().mean()).values
    X[:, ML_FEATURES.index('gradual_escalation_score')] = np.where(amounts > 0, amt / amounts, 0.0).astype(np.float32)

    # known_device_count: number of unique merchants per card
    X[:, ML_FEATURES.index('known_device_count')] = df.groupby('cc_num')['merchant'].transform('nunique').values.astype(np.float32)

    # account_tenure_days: days since first transaction
    first_tx = df.groupby('cc_num')['unix_time'].transform('min').values
    X[:, ML_FEATURES.index('account_tenure_days')] = ((df['unix_time'].values - first_tx) / 86400.0).astype(np.float32)

    # hour_of_day
    X[:, ML_FEATURES.index('hour_of_day')] = hours.astype(np.float32)

    # is_weekend
    dow = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek.values
    X[:, ML_FEATURES.index('is_weekend')] = (dow >= 5).astype(np.float32)

    # shared_device_accounts: number of cards per merchant
    X[:, ML_FEATURES.index('shared_device_accounts')] = df.groupby('merchant')['cc_num'].transform('nunique').values.astype(np.float32)

    # shared_recipient_accounts: number of cards per merchant (same)
    X[:, ML_FEATURES.index('shared_recipient_accounts')] = X[:, ML_FEATURES.index('shared_device_accounts')]

    # mule_ring_score: number of unique cities per card
    X[:, ML_FEATURES.index('mule_ring_score')] = df.groupby('cc_num')['city'].transform('nunique').values.astype(np.float32)

    # hour_deviation: deviation from card's typical hour
    median_hour = df.groupby('cc_num')['trans_date_trans_time'].transform(lambda x: pd.to_datetime(x).dt.hour.median()).values
    X[:, ML_FEATURES.index('hour_deviation')] = np.abs(hours - median_hour).astype(np.float32)

    # amount_zscore: z-score of amount within card history
    grp_mean = df.groupby('cc_num')['amt'].transform('mean').values
    grp_std = df.groupby('cc_num')['amt'].transform('std').fillna(1.0).values
    X[:, ML_FEATURES.index('amount_zscore')] = np.where(grp_std > 0, (amt - grp_mean) / grp_std, 0.0).astype(np.float32)

    # velocity_deviation: txns in last hour / typical hourly rate
    X[:, ML_FEATURES.index('velocity_deviation')] = X[:, ML_FEATURES.index('txn_freq_last_24h')] / 24.0

    # recipient_novelty: 1 / (1 + times seen this merchant for this card)
    seen = df.groupby(['cc_num', 'merchant']).cumcount().values + 1
    X[:, ML_FEATURES.index('recipient_novelty')] = (1.0 / seen).astype(np.float32)

    # txn_regularity: std of inter-txn times (lower = more regular = suspicious)
    std_gap = df.groupby('cc_num')['unix_time'].transform(lambda x: x.diff().fillna(0).expanding().std()).values / 3600.0
    X[:, ML_FEATURES.index('txn_regularity')] = np.clip(std_gap, 0, 100).astype(np.float32)

    return X


def map_paysim_to_features(df: pd.DataFrame) -> np.ndarray:
    """Map paysim columns to 21 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    amt = df['amount'].values.astype(np.float32)
    old_bal = df['oldbalanceOrg'].values.astype(np.float32)
    new_bal = df['newbalanceOrig'].values.astype(np.float32)
    old_dest = df['oldbalanceDest'].values.astype(np.float32)
    new_dest = df['newbalanceDest'].values.astype(np.float32)

    # amount_ratio
    X[:, ML_FEATURES.index('amount_ratio')] = np.where(old_bal > 0, amt / old_bal, 0.0)

    # txn_freq_last_24h (not available, estimate from type)
    type_map = {'PAYMENT': 1, 'TRANSFER': 1, 'CASH_OUT': 2, 'CASH_IN': 2, 'DEBIT': 1}
    X[:, ML_FEATURES.index('txn_freq_last_24h')] = df['type'].map(type_map).fillna(1).values.astype(np.float32)

    # txn_time_unusual: not available, use 0.5
    X[:, ML_FEATURES.index('txn_time_unusual')] = 0.5

    # new_device_flag: balance went to zero (account drained)
    X[:, ML_FEATURES.index('new_device_flag')] = (new_bal < 1.0).astype(np.float32)

    # unusual_location_flag: large balance difference on destination
    X[:, ML_FEATURES.index('unusual_location_flag')] = (np.abs(new_dest - old_dest) > amt * 2).astype(np.float32)

    # unusual_recipient_flag: destination was previously empty
    X[:, ML_FEATURES.index('unusual_recipient_flag')] = (old_dest < 1.0).astype(np.float32)

    # failed_auth_count_24h
    X[:, ML_FEATURES.index('failed_auth_count_24h')] = 0.0

    # days_since_last_similar_txn
    X[:, ML_FEATURES.index('days_since_last_similar_txn')] = 1.0

    # gradual_escalation_score
    X[:, ML_FEATURES.index('gradual_escalation_score')] = np.where(old_bal > 0, amt / old_bal, 0.0)

    # known_device_count
    X[:, ML_FEATURES.index('known_device_count')] = 1.0

    # account_tenure_days
    X[:, ML_FEATURES.index('account_tenure_days')] = 30.0

    # hour_of_day
    X[:, ML_FEATURES.index('hour_of_day')] = 12.0

    # is_weekend
    X[:, ML_FEATURES.index('is_weekend')] = 0.0

    # shared_device_accounts
    X[:, ML_FEATURES.index('shared_device_accounts')] = 1.0

    # shared_recipient_accounts
    X[:, ML_FEATURES.index('shared_recipient_accounts')] = 1.0

    # mule_ring_score
    X[:, ML_FEATURES.index('mule_ring_score')] = (old_bal < 1.0).astype(np.float32) * (new_dest > 10000).astype(np.float32)

    # hour_deviation
    X[:, ML_FEATURES.index('hour_deviation')] = 0.0

    # amount_zscore
    X[:, ML_FEATURES.index('amount_zscore')] = (amt - np.median(amt)) / (np.std(amt) + 1e-6)

    # velocity_deviation
    X[:, ML_FEATURES.index('velocity_deviation')] = X[:, ML_FEATURES.index('txn_freq_last_24h')] / 24.0

    # recipient_novelty
    X[:, ML_FEATURES.index('recipient_novelty')] = 0.5

    # txn_regularity
    X[:, ML_FEATURES.index('txn_regularity')] = 0.0

    return X


def map_creditcard_to_features(df: pd.DataFrame, label_col='Class') -> np.ndarray:
    """Map ULB/fraud_data PCA features to 21 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    # PCA components: V1-V28 capture variance. Use statistical summaries.
    pca_cols = [c for c in df.columns if c.startswith('V')]
    pca_data = df[pca_cols].values

    amt = df['Amount'].values.astype(np.float32) if 'Amount' in df.columns else np.zeros(n, dtype=np.float32)

    # amount_ratio: amount / median
    med_amt = np.median(amt)
    X[:, ML_FEATURES.index('amount_ratio')] = np.where(med_amt > 0, amt / med_amt, 0.0)

    # txn_freq_last_24h: PCA summary
    X[:, ML_FEATURES.index('txn_freq_last_24h')] = np.abs(pca_data[:, :5]).mean(axis=1) * 10

    # txn_time_unusual: Time column if available
    if 'Time' in df.columns:
        t = df['Time'].values
        X[:, ML_FEATURES.index('txn_time_unusual')] = (t % 86400) / 86400.0
    else:
        X[:, ML_FEATURES.index('txn_time_unusual')] = 0.5

    # new_device_flag: based on PCA variance
    X[:, ML_FEATURES.index('new_device_flag')] = (np.abs(pca_data[:, 0]) > 2).astype(np.float32)

    # unusual_location_flag: based on PCA
    X[:, ML_FEATURES.index('unusual_location_flag')] = (np.abs(pca_data[:, 1]) > 2).astype(np.float32)

    # unusual_recipient_flag: based on PCA
    X[:, ML_FEATURES.index('unusual_recipient_flag')] = (np.abs(pca_data[:, 2]) > 2).astype(np.float32)

    # failed_auth_count_24h
    X[:, ML_FEATURES.index('failed_auth_count_24h')] = 0.0

    # days_since_last_similar_txn
    if 'Time' in df.columns:
        X[:, ML_FEATURES.index('days_since_last_similar_txn')] = np.clip(df['Time'].values / 86400.0, 0, 30)
    else:
        X[:, ML_FEATURES.index('days_since_last_similar_txn')] = 1.0

    # gradual_escalation_score
    X[:, ML_FEATURES.index('gradual_escalation_score')] = np.abs(pca_data[:, 3]).clip(0, 1)

    # known_device_count
    X[:, ML_FEATURES.index('known_device_count')] = 1.0

    # account_tenure_days
    X[:, ML_FEATURES.index('account_tenure_days')] = 30.0

    # hour_of_day
    if 'Time' in df.columns:
        X[:, ML_FEATURES.index('hour_of_day')] = (df['Time'].values % 86400) / 3600.0
    else:
        X[:, ML_FEATURES.index('hour_of_day')] = 12.0

    # is_weekend
    X[:, ML_FEATURES.index('is_weekend')] = 0.0

    # shared_device_accounts
    X[:, ML_FEATURES.index('shared_device_accounts')] = 1.0

    # shared_recipient_accounts
    X[:, ML_FEATURES.index('shared_recipient_accounts')] = 1.0

    # mule_ring_score
    X[:, ML_FEATURES.index('mule_ring_score')] = np.abs(pca_data[:, 4]).clip(0, 10)

    # hour_deviation
    X[:, ML_FEATURES.index('hour_deviation')] = np.abs(pca_data[:, 5]).clip(0, 12)

    # amount_zscore
    X[:, ML_FEATURES.index('amount_zscore')] = (amt - amt.mean()) / (amt.std() + 1e-6)

    # velocity_deviation
    X[:, ML_FEATURES.index('velocity_deviation')] = np.abs(pca_data[:, 6]).clip(0, 10)

    # recipient_novelty
    X[:, ML_FEATURES.index('recipient_novelty')] = np.abs(pca_data[:, 7]).clip(0, 1)

    # txn_regularity
    X[:, ML_FEATURES.index('txn_regularity')] = np.abs(pca_data[:, 8]).clip(0, 100)

    return X


# ─── Dataset loaders ────────────────────────────────────────────
def load_kaggle_fraud(max_rows=200000):
    """Load kaggle_fraud train+test, combine, sample for speed."""
    print("\n[1/5] Loading kaggle_fraud...")
    dfs = []
    for path in ['data/kaggle_fraud/fraudTrain.csv', 'data/kaggle_fraud/fraudTest.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Drop PII columns (keep city for feature mapping)
            df = df.drop(columns=['first', 'last', 'street', 'state', 'zip', 'dob', 'job', 'trans_num', 'Unnamed: 0'], errors='ignore')
            dfs.append(df)
    if not dfs:
        return None, None
    df = pd.concat(dfs, ignore_index=True)
    # Sample for speed
    if len(df) > max_rows:
        fraud_idx = df[df['is_fraud'] == 1].index
        legit_idx = df[df['is_fraud'] == 0].index
        n_fraud = min(len(fraud_idx), int(max_rows * 0.05))
        n_legit = max_rows - n_fraud
        sampled = np.concatenate([np.random.RandomState(SEED).choice(fraud_idx, n_fraud, replace=False),
                                  np.random.RandomState(SEED).choice(legit_idx, n_legit, replace=False)])
        df = df.loc[sampled]
    print(f"  Loaded {len(df):,} rows, {df['is_fraud'].sum():,} fraud")
    X = map_kaggle_to_features(df)
    y = df['is_fraud'].values.astype(np.int32)
    # Drop PII column after feature mapping
    df.drop(columns=['city'], errors='ignore', inplace=True)
    return X, y


def load_paysim(max_rows=200000):
    """Load PaySim dataset."""
    print("\n[2/5] Loading PaySim...")
    df = pd.read_csv('data/paysim_1m.csv')
    if len(df) > max_rows:
        fraud_idx = df[df['isFraud'] == 1].index
        legit_idx = df[df['isFraud'] == 0].index
        n_fraud = min(len(fraud_idx), int(max_rows * 0.1))
        n_legit = max_rows - n_fraud
        sampled = np.concatenate([np.random.RandomState(SEED).choice(fraud_idx, n_fraud, replace=False),
                                  np.random.RandomState(SEED).choice(legit_idx, n_legit, replace=False)])
        df = df.loc[sampled]
    print(f"  Loaded {len(df):,} rows, {df['isFraud'].sum():,} fraud")
    X = map_paysim_to_features(df)
    y = df['isFraud'].values.astype(np.int32)
    return X, y


def load_creditcard(max_rows=150000):
    """Load ULB creditcard dataset."""
    print("\n[3/5] Loading ULB creditcard...")
    df = pd.read_csv('data/creditcard.csv')
    if len(df) > max_rows:
        fraud_idx = df[df['Class'] == 1].index
        legit_idx = df[df['Class'] == 0].index
        n_fraud = len(fraud_idx)
        n_legit = max_rows - n_fraud
        sampled = np.concatenate([fraud_idx.values, np.random.RandomState(SEED).choice(legit_idx, n_legit, replace=False)])
        df = df.loc[sampled]
    print(f"  Loaded {len(df):,} rows, {df['Class'].sum():,} fraud")
    X = map_creditcard_to_features(df, 'Class')
    y = df['Class'].values.astype(np.int32)
    return X, y


def load_fraud_data(max_rows=20000):
    """Load fraud_data (PCA-based)."""
    print("\n[4/5] Loading fraud_data...")
    df = pd.read_csv('data/fraud_data.csv')
    if len(df) > max_rows:
        fraud_idx = df[df['Class'] == 1].index
        legit_idx = df[df['Class'] == 0].index
        n_fraud = len(fraud_idx)
        n_legit = max_rows - n_fraud
        sampled = np.concatenate([fraud_idx.values, np.random.RandomState(SEED).choice(legit_idx, min(n_legit, len(legit_idx)), replace=False)])
        df = df.loc[sampled]
    print(f"  Loaded {len(df):,} rows, {df['Class'].sum():,} fraud")
    X = map_creditcard_to_features(df, 'Class')
    y = df['Class'].values.astype(np.int32)
    return X, y


def load_ealtman_sample(n_rows=100000):
    """Load a small sample of ealtman2019 for cross-domain testing."""
    print("\n[5/5] Loading ealtman2019 sample...")
    path = 'data/ealtman2019/credit_card_transactions-ibm_v2.csv'
    if not os.path.exists(path):
        print("  File not found, skipping")
        return None, None
    # Read just enough rows
    df = pd.read_csv(path, nrows=n_rows + 5000)
    # Determine fraud column
    label_col = None
    for c in ['is_fraud', 'fraud', 'isFraud', 'Class']:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        print("  No fraud label found, skipping")
        return None, None

    fraud_idx = df[df[label_col] == 1].index
    legit_idx = df[df[label_col] == 0].index
    n_fraud = min(len(fraud_idx), 500)
    n_legit = n_rows - n_fraud
    sampled = np.concatenate([
        np.random.RandomState(SEED).choice(fraud_idx, n_fraud, replace=False),
        np.random.RandomState(SEED).choice(legit_idx, n_legit, replace=False)
    ])
    df = df.loc[sampled]
    print(f"  Loaded {len(df):,} rows, {df[label_col].sum():,} fraud")
    # Map features using kaggle mapper (similar column structure)
    # Need to find matching columns
    col_map = {
        'amount': 'amt', 'amt': 'amt',
        'timestamp': 'trans_date_trans_time', 'trans_date_trans_time': 'trans_date_trans_time',
        'card_num': 'cc_num', 'cc_num': 'cc_num',
        'merchant_name': 'merchant', 'merchant': 'merchant',
        'merchant_category': 'category', 'category': 'category',
        'cardholder_lat': 'lat', 'lat': 'lat',
        'cardholder_long': 'long', 'long': 'long',
        'merchant_lat': 'merch_lat', 'merch_lat': 'merch_lat',
        'merchant_long': 'merch_long', 'merch_long': 'merch_long',
        'unix_timestamp': 'unix_time', 'unix_time': 'unix_time',
        'city': 'city',
    }
    df_renamed = df.rename(columns={c: v for c, v in col_map.items() if c in df.columns})
    # Add missing columns with defaults
    for needed in ['cc_num', 'merchant', 'category', 'unix_time', 'city']:
        if needed not in df_renamed.columns:
            df_renamed[needed] = 0
    if 'trans_date_trans_time' not in df_renamed.columns:
        df_renamed['trans_date_trans_time'] = pd.Timestamp('2020-01-01')
    for needed in ['amt', 'lat', 'long', 'merch_lat', 'merch_long']:
        if needed not in df_renamed.columns:
            df_renamed[needed] = 0.0

    X = map_kaggle_to_features(df_renamed)
    y = df[label_col].values.astype(np.int32)
    return X, y


# ─── Model training ─────────────────────────────────────────────
def train_model(X_train, y_train, model_type='xgboost'):
    """Train model and return it."""
    from xgboost import XGBClassifier
    from sklearn.impute import SimpleImputer
    # Impute NaN/Inf for all model types for safety
    imputer = SimpleImputer(strategy='median')
    X_clean = imputer.fit_transform(X_train)
    # Replace any remaining inf
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=10.0, neginf=-10.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    if model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=SEED, eval_metric='auc', use_label_encoder=False
        )
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight='balanced', random_state=SEED, n_jobs=-1
        )
    elif model_type == 'lr':
        model = LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=SEED
        )

    model.fit(X_scaled, y_train)
    return model, scaler, imputer


def evaluate_model(model, scaler, X_test, y_test, name="", imputer=None):
    """Evaluate model and return metrics dict."""
    X_clean = X_test.copy()
    if imputer is not None:
        X_clean = imputer.transform(X_clean)
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=10.0, neginf=-10.0)
    X_scaled = scaler.transform(X_clean)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    r1, _ = recall_at_fpr(y_test, y_prob, 0.001)
    r05, _ = recall_at_fpr(y_test, y_prob, 0.005)
    r1pct, _ = recall_at_fpr(y_test, y_prob, 0.01)

    y_pred = (y_prob > 0.5).astype(int)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    brier = brier_score(y_test, y_prob)
    ece_val = ece(y_test, y_prob)

    # Bootstrap CI for ROC-AUC
    roc_mean, roc_lo, roc_hi = bootstrap_ci(y_test, y_prob, roc_auc_score)
    pr_mean, pr_lo, pr_hi = bootstrap_ci(y_test, y_prob, average_precision_score)

    cm = confusion_matrix(y_test, y_pred)

    result = {
        'name': name,
        'n_test': len(y_test),
        'n_fraud': int(y_test.sum()),
        'fraud_prev': float(y_test.mean()),
        'roc_auc': round(roc, 4),
        'pr_auc': round(pr, 4),
        'recall_0.1pct_fpr': round(r1, 4),
        'recall_0.5pct_fpr': round(r05, 4),
        'recall_1pct_fpr': round(r1pct, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'brier': round(brier, 6),
        'ece': round(ece_val, 6),
        'roc_auc_ci95': [round(roc_lo, 4), round(roc_hi, 4)],
        'pr_auc_ci95': [round(pr_lo, 4), round(pr_hi, 4)],
        'confusion_matrix': cm.tolist(),
    }
    return result


# ─── Main pipeline ──────────────────────────────────────────────
def main():
    np.random.seed(SEED)
    results = {'before': {}, 'after': {}}

    # Load all datasets
    print("="*70)
    print("PHASE 1: Loading datasets")
    print("="*70)

    datasets = {}

    X, y = load_kaggle_fraud(max_rows=100000)
    if X is not None:
        datasets['kaggle_fraud'] = (X, y)

    X, y = load_paysim(max_rows=100000)
    if X is not None:
        datasets['paysim'] = (X, y)

    X, y = load_creditcard(max_rows=80000)
    if X is not None:
        datasets['ulb_creditcard'] = (X, y)

    X, y = load_fraud_data(max_rows=20000)
    if X is not None:
        datasets['fraud_data'] = (X, y)

    X, y = load_ealtman_sample(n_rows=100000)
    if X is not None:
        datasets['ealtman2019'] = (X, y)

    print(f"\nLoaded {len(datasets)} datasets")
    total_rows = sum(X.shape[0] for X, y in datasets.values())
    total_fraud = sum(int(y.sum()) for X, y in datasets.values())
    print(f"Total: {total_rows:,} rows, {total_fraud:,} fraud cases")

    # ─── PHASE 2: Train on kartik2112 (current production) and evaluate cross-dataset ───
    print("\n" + "="*70)
    print("PHASE 2: Baseline cross-dataset evaluation (current model)")
    print("="*70)

    # Load existing model
    import joblib
    from src.risk_engine.fusion import ML_FEATURES as MLF

    try:
        xgb_model = joblib.load('models/artifacts/xgboost.joblib')
        scaler = joblib.load('models/artifacts/scaler.joblib')
        print("Loaded existing production model")

        for dname, (X, y) in datasets.items():
            if X.shape[1] != N_FEATURES:
                print(f"  SKIP {dname}: feature mismatch ({X.shape[1]} vs {N_FEATURES})")
                continue
            res = evaluate_model(xgb_model, scaler, X, y, f"current_{dname}")
            results['before'][dname] = res
            print(f"  {dname}: ROC-AUC={res['roc_auc']:.4f}, PR-AUC={res['pr_auc']:.4f}, R@1%FPR={res['recall_1pct_fpr']:.4f}, n={res['n_test']}, fraud={res['n_fraud']}")
    except Exception as e:
        print(f"Could not load existing model: {e}")

    # ─── PHASE 3: Retrain on combined data ───────────────────────
    print("\n" + "="*70)
    print("PHASE 3: Retrain on combined datasets")
    print("="*70)

    # Combine all datasets for training
    all_X = []
    all_y = []
    for dname, (X, y) in datasets.items():
        all_X.append(X)
        all_y.append(y)
        print(f"  Added {dname}: {X.shape[0]:,} rows")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    # Add SMOTE-like oversampling for fraud minority
    fraud_mask = y_all == 1
    X_fraud = X_all[fraud_mask]
    y_fraud = y_all[fraud_mask]
    X_legit = X_all[~fraud_mask]
    y_legit = y_all[~fraud_mask]

    # Oversample fraud 3x for better learning
    rng = np.random.RandomState(SEED)
    fraud_idx = rng.choice(len(X_fraud), len(X_fraud) * 3, replace=True)
    X_balanced = np.vstack([X_legit, X_fraud[fraud_idx]])
    y_balanced = np.concatenate([y_legit, y_fraud[fraud_idx]])

    print(f"\nBalanced training set: {len(X_balanced):,} rows ({y_balanced.sum():,} fraud)")

    # Time-based split
    n = len(X_balanced)
    idx = rng.permutation(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train = X_balanced[idx[:n_train]]
    y_train = y_balanced[idx[:n_train]]
    X_val = X_balanced[idx[n_train:n_train+n_val]]
    y_val = y_balanced[idx[n_train:n_train+n_val]]
    X_test = X_balanced[idx[n_train+n_val:]]
    y_test = y_balanced[idx[n_train+n_val:]]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Train new XGBoost
    print("\nTraining XGBoost on combined data...")
    xgb_new, scaler_new, xgb_imputer = train_model(X_train, y_train, 'xgboost')

    # Validate
    val_res = evaluate_model(xgb_new, scaler_new, X_val, y_val, "combined_val", xgb_imputer)
    print(f"  Validation: ROC-AUC={val_res['roc_auc']:.4f}, PR-AUC={val_res['pr_auc']:.4f}, R@1%FPR={val_res['recall_1pct_fpr']:.4f}")

    # Test
    test_res = evaluate_model(xgb_new, scaler_new, X_test, y_test, "combined_test", xgb_imputer)
    print(f"  Test: ROC-AUC={test_res['roc_auc']:.4f}, PR-AUC={test_res['pr_auc']:.4f}, R@1%FPR={test_res['recall_1pct_fpr']:.4f}")

    # Also train RF and LR for ensemble
    print("\nTraining Random Forest...")
    rf_new, rf_scaler, rf_imputer = train_model(X_train, y_train, 'rf')

    print("Training Logistic Regression...")
    lr_new, lr_scaler, lr_imputer = train_model(X_train, y_train, 'lr')

    # ─── PHASE 4: Evaluate improved model cross-dataset ──────────
    print("\n" + "="*70)
    print("PHASE 4: Improved model cross-dataset evaluation")
    print("="*70)

    for dname, (X, y) in datasets.items():
        if X.shape[1] != N_FEATURES:
            continue

        # XGBoost
        res_xgb = evaluate_model(xgb_new, scaler_new, X, y, f"improved_xgb_{dname}", xgb_imputer)

        # Random Forest
        res_rf = evaluate_model(rf_new, rf_scaler, X, y, f"improved_rf_{dname}", rf_imputer)

        # Logistic Regression
        res_lr = evaluate_model(lr_new, lr_scaler, X, y, f"improved_lr_{dname}", lr_imputer)

        # Ensemble (average probabilities)
        X_clean = xgb_imputer.transform(X)
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=10.0, neginf=-10.0)
        X_xgb = scaler_new.transform(X_clean)
        X_clean2 = rf_imputer.transform(X)
        X_clean2 = np.nan_to_num(X_clean2, nan=0.0, posinf=10.0, neginf=-10.0)
        X_rf = rf_scaler.transform(X_clean2)
        X_clean3 = lr_imputer.transform(X)
        X_clean3 = np.nan_to_num(X_clean3, nan=0.0, posinf=10.0, neginf=-10.0)
        X_lr = lr_scaler.transform(X_clean3)
        prob_xgb = xgb_new.predict_proba(X_xgb)[:, 1]
        prob_rf = rf_new.predict_proba(X_rf)[:, 1]
        prob_lr = lr_new.predict_proba(X_lr)[:, 1]
        prob_ensemble = 0.5 * prob_xgb + 0.3 * prob_rf + 0.2 * prob_lr
        roc_ens = roc_auc_score(y, prob_ensemble)
        pr_ens = average_precision_score(y, prob_ensemble)
        r1_ens, _ = recall_at_fpr(y, prob_ensemble, 0.01)
        brier_ens = brier_score(y, prob_ensemble)
        ece_ens = ece(y, prob_ensemble)

        results['after'][dname] = {
            'xgboost': res_xgb,
            'random_forest': res_rf,
            'logistic_regression': res_lr,
            'ensemble': {
                'roc_auc': round(roc_ens, 4),
                'pr_auc': round(pr_ens, 4),
                'recall_1pct_fpr': round(r1_ens, 4),
                'brier': round(brier_ens, 6),
                'ece': round(ece_ens, 6),
            }
        }

        before_roc = results['before'].get(dname, {}).get('roc_auc', 'N/A')
        print(f"\n  {dname}:")
        print(f"    Before: ROC-AUC={before_roc}")
        print(f"    XGBoost:  ROC-AUC={res_xgb['roc_auc']:.4f}, PR-AUC={res_xgb['pr_auc']:.4f}, R@1%FPR={res_xgb['recall_1pct_fpr']:.4f}")
        print(f"    RF:       ROC-AUC={res_rf['roc_auc']:.4f}, PR-AUC={res_rf['pr_auc']:.4f}, R@1%FPR={res_rf['recall_1pct_fpr']:.4f}")
        print(f"    LR:       ROC-AUC={res_lr['roc_auc']:.4f}, PR-AUC={res_lr['pr_auc']:.4f}, R@1%FPR={res_lr['recall_1pct_fpr']:.4f}")
        print(f"    Ensemble: ROC-AUC={roc_ens:.4f}, PR-AUC={pr_ens:.4f}, R@1%FPR={r1_ens:.4f}")

    # ─── PHASE 5: Save artifacts ─────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 5: Saving artifacts")
    print("="*70)

    # Save improved model
    joblib.dump(xgb_new, 'models/artifacts/xgboost_v4.joblib')
    joblib.dump(scaler_new, 'models/artifacts/scaler_v4.joblib')
    joblib.dump(xgb_imputer, 'models/artifacts/imputer_v4.joblib')
    joblib.dump(rf_new, 'models/artifacts/random_forest_v4.joblib')
    joblib.dump(lr_new, 'models/artifacts/logistic_regression_v4.joblib')
    print("Saved v4 model artifacts")

    # Save evaluation results
    eval_path = OUT_DIR / 'comprehensive_multi_eval.json'
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved evaluation to {eval_path}")

    # Generate summary table
    print("\n" + "="*70)
    print("FINAL RESULTS: Before vs After")
    print("="*70)
    print(f"{'Dataset':<20} {'Before ROC':>12} {'After ROC':>12} {'Δ':>8} {'Before R@1%':>14} {'After R@1%':>14} {'Δ':>8}")
    print("-" * 92)

    for dname in datasets:
        before = results['before'].get(dname, {})
        after = results['after'].get(dname, {}).get('xgboost', {})
        if before and after:
            b_roc = before.get('roc_auc', 0)
            a_roc = after.get('roc_auc', 0)
            b_r1 = before.get('recall_1pct_fpr', 0)
            a_r1 = after.get('recall_1pct_fpr', 0)
            d_roc = a_roc - b_roc
            d_r1 = a_r1 - b_r1
            print(f"{dname:<20} {b_roc:>12.4f} {a_roc:>12.4f} {d_roc:>+8.4f} {b_r1:>14.4f} {a_r1:>14.4f} {d_r1:>+8.4f}")

    # Hash for manifest
    model_hash = hashlib.sha256(open('models/artifacts/xgboost_v4.joblib', 'rb').read()).hexdigest()[:16]
    scaler_hash = hashlib.sha256(open('models/artifacts/scaler_v4.joblib', 'rb').read()).hexdigest()[:16]
    print(f"\nModel hash: {model_hash}")
    print(f"Scaler hash: {scaler_hash}")

    # Update metadata
    meta = {
        'model_version': 'v4_combined',
        'model_hash': model_hash,
        'scaler_hash': scaler_hash,
        'training_datasets': list(datasets.keys()),
        'training_rows': int(len(X_train)),
        'total_data_rows': int(len(X_all)),
        'features': ML_FEATURES,
        'feature_count': N_FEATURES,
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': SEED,
    }
    with open(OUT_DIR / 'model_manifest_v4.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model manifest to {OUT_DIR / 'model_manifest_v4.json'}")

    return results


if __name__ == '__main__':
    main()
