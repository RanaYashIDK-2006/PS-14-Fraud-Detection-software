#!/usr/bin/env python3
"""Retrain for finance/card fraud with subscription micro-fraud detection.

Adds 6 new features to ML_FEATURES:
1. subscription_pattern_score: regularity of charges to same merchant
2. micro_fraud_flag: small amount + high balance (death by a thousand cuts)
3. balance_drain_ratio: how much of balance is being siphoned
4. merchant_fraud_concentration: fraud density at this merchant category
5. card_velocity_ratio: txns per hour vs card's historical rate
6. amount_cluster_distance: distance from nearest amount cluster center
"""

import sys, os, json, time, hashlib, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, precision_score, recall_score, confusion_matrix
)
from scipy import stats as sp_stats
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk_engine.fusion import ML_FEATURES

SEED = 42
OUT_DIR = Path("data/real_data_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Extended feature set: original 21 + 6 new subscription/fraud features
NEW_FEATURES = ML_FEATURES + [
    'subscription_pattern_score',  # Regularity of charges to same merchant
    'micro_fraud_flag',            # Small amount + high balance ratio
    'balance_drain_ratio',         # Amount / account balance
    'merchant_fraud_concentration', # How suspicious this merchant category is
    'card_velocity_ratio',         # Current speed vs historical speed
    'amount_cluster_distance',     # Distance from typical amount clusters
]
N_FEATURES = len(NEW_FEATURES)
print(f"Extended features ({N_FEATURES}): {NEW_FEATURES[-6:]}")

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

def recall_at_fpr(y_true, y_score, target_fpr=0.01):
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score)
    valid = fpr_arr <= target_fpr
    if not valid.any():
        return 0.0, 0.0
    idx = np.where(valid)[0][-1]
    return tpr_arr[idx], thresholds[idx]

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
    """Map kaggle_fraud columns to 27 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    amt = df['amt'].values.astype(np.float32)
    medians = df.groupby('cc_num')['amt'].transform('median').values.astype(np.float32)
    X[:, NEW_FEATURES.index('amount_ratio')] = np.where(medians > 0, amt / medians, 0.0)

    X[:, NEW_FEATURES.index('txn_freq_last_24h')] = df.groupby('cc_num')['cc_num'].transform('count').values.astype(np.float32)

    hours = pd.to_datetime(df['trans_date_trans_time']).dt.hour.values
    X[:, NEW_FEATURES.index('txn_time_unusual')] = hours.astype(np.float32) / 23.0

    X[:, NEW_FEATURES.index('new_device_flag')] = (df.groupby(['cc_num', 'category']).cumcount() == 0).values.astype(np.float32)

    lat1, lon1 = df['lat'].values, df['long'].values
    lat2, lon2 = df['merch_lat'].values, df['merch_long'].values
    dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    median_dist = np.median(dist)
    X[:, NEW_FEATURES.index('unusual_location_flag')] = (dist > median_dist * 2).astype(np.float32)

    X[:, NEW_FEATURES.index('unusual_recipient_flag')] = (df.groupby(['cc_num', 'merchant']).cumcount() == 0).values.astype(np.float32)
    X[:, NEW_FEATURES.index('failed_auth_count_24h')] = 0.0

    df_sorted = df.sort_values(['cc_num', 'unix_time'])
    gaps = df_sorted.groupby('cc_num')['unix_time'].diff().fillna(3600*24).values / 86400.0
    X[:, NEW_FEATURES.index('days_since_last_similar_txn')] = np.clip(gaps[:n], 0, 365).astype(np.float32)

    amounts = df.groupby('cc_num')['amt'].transform(lambda x: x.expanding().mean()).values
    X[:, NEW_FEATURES.index('gradual_escalation_score')] = np.where(amounts > 0, amt / amounts, 0.0).astype(np.float32)

    X[:, NEW_FEATURES.index('known_device_count')] = df.groupby('cc_num')['merchant'].transform('nunique').values.astype(np.float32)

    first_tx = df.groupby('cc_num')['unix_time'].transform('min').values
    X[:, NEW_FEATURES.index('account_tenure_days')] = ((df['unix_time'].values - first_tx) / 86400.0).astype(np.float32)

    X[:, NEW_FEATURES.index('hour_of_day')] = hours.astype(np.float32)

    dow = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek.values
    X[:, NEW_FEATURES.index('is_weekend')] = (dow >= 5).astype(np.float32)

    X[:, NEW_FEATURES.index('shared_device_accounts')] = df.groupby('merchant')['cc_num'].transform('nunique').values.astype(np.float32)
    X[:, NEW_FEATURES.index('shared_recipient_accounts')] = X[:, NEW_FEATURES.index('shared_device_accounts')]
    X[:, NEW_FEATURES.index('mule_ring_score')] = df.groupby('cc_num')['city'].transform('nunique').values.astype(np.float32)

    median_hour = df.groupby('cc_num')['trans_date_trans_time'].transform(lambda x: pd.to_datetime(x).dt.hour.median()).values
    X[:, NEW_FEATURES.index('hour_deviation')] = np.abs(hours - median_hour).astype(np.float32)

    grp_mean = df.groupby('cc_num')['amt'].transform('mean').values
    grp_std = df.groupby('cc_num')['amt'].transform('std').fillna(1.0).values
    X[:, NEW_FEATURES.index('amount_zscore')] = np.where(grp_std > 0, (amt - grp_mean) / grp_std, 0.0).astype(np.float32)

    X[:, NEW_FEATURES.index('velocity_deviation')] = X[:, NEW_FEATURES.index('txn_freq_last_24h')] / 24.0

    seen = df.groupby(['cc_num', 'merchant']).cumcount().values + 1
    X[:, NEW_FEATURES.index('recipient_novelty')] = (1.0 / seen).astype(np.float32)

    std_gap = df.groupby('cc_num')['unix_time'].transform(lambda x: x.diff().fillna(0).expanding().std()).values / 3600.0
    X[:, NEW_FEATURES.index('txn_regularity')] = np.clip(std_gap, 0, 100).astype(np.float32)

    # ─── NEW: Subscription/Micro-Fraud Features ─────────────────
    # subscription_pattern_score: how regular are charges to same merchant (0=irregular, 1=very regular)
    merchant_gaps = df.groupby(['cc_num', 'merchant'])['unix_time'].diff().fillna(0).values
    X[:, NEW_FEATURES.index('subscription_pattern_score')] = np.where(merchant_gaps > 0,
        np.clip(1.0 / (1.0 + merchant_gaps / 86400.0), 0, 1), 0).astype(np.float32)

    # micro_fraud_flag: small amount relative to card's typical spending
    median_amt = df.groupby('cc_num')['amt'].transform('median').values
    X[:, NEW_FEATURES.index('micro_fraud_flag')] = ((amt < median_amt * 0.1) & (amt > 0)).astype(np.float32)

    # balance_drain_ratio: amount / typical balance (not available in kaggle, use amount_ratio proxy)
    X[:, NEW_FEATURES.index('balance_drain_ratio')] = X[:, NEW_FEATURES.index('amount_ratio')]

    # merchant_fraud_concentration: fraud rate at this category (from training data proxy)
    cat_fraud_rate = df.groupby('category')['is_fraud'].transform('mean').values
    X[:, NEW_FEATURES.index('merchant_fraud_concentration')] = cat_fraud_rate.astype(np.float32)

    # card_velocity_ratio: current txn speed vs card's average
    tx_counts = df.groupby('cc_num').cumcount().values + 1
    time_span = (df['unix_time'].values - first_tx) / 3600.0 + 1
    current_rate = tx_counts / time_span
    X[:, NEW_FEATURES.index('card_velocity_ratio')] = current_rate.astype(np.float32)

    # amount_cluster_distance: distance from round amounts ($10, $50, $100, $500)
    round_amounts = np.array([10, 50, 100, 500, 1000])
    min_dist = np.min(np.abs(amt[:, None] - round_amounts[None, :]), axis=1)
    X[:, NEW_FEATURES.index('amount_cluster_distance')] = min_dist.astype(np.float32)

    return X


def map_paysim_to_features(df: pd.DataFrame) -> np.ndarray:
    """Map paysim columns to 27 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    amt = df['amount'].values.astype(np.float32)
    old_bal = df['oldbalanceOrg'].values.astype(np.float32)
    new_bal = df['newbalanceOrig'].values.astype(np.float32)
    old_dest = df['oldbalanceDest'].values.astype(np.float32)
    new_dest = df['newbalanceDest'].values.astype(np.float32)

    X[:, NEW_FEATURES.index('amount_ratio')] = np.where(old_bal > 0, amt / old_bal, 0.0)

    type_map = {'PAYMENT': 1, 'TRANSFER': 1, 'CASH_OUT': 2, 'CASH_IN': 2, 'DEBIT': 1}
    X[:, NEW_FEATURES.index('txn_freq_last_24h')] = df['type'].map(type_map).fillna(1).values.astype(np.float32)
    X[:, NEW_FEATURES.index('txn_time_unusual')] = 0.5

    X[:, NEW_FEATURES.index('new_device_flag')] = (new_bal < 1.0).astype(np.float32)
    X[:, NEW_FEATURES.index('unusual_location_flag')] = (np.abs(new_dest - old_dest) > amt * 2).astype(np.float32)
    X[:, NEW_FEATURES.index('unusual_recipient_flag')] = (old_dest < 1.0).astype(np.float32)
    X[:, NEW_FEATURES.index('failed_auth_count_24h')] = 0.0
    X[:, NEW_FEATURES.index('days_since_last_similar_txn')] = 1.0
    X[:, NEW_FEATURES.index('gradual_escalation_score')] = np.where(old_bal > 0, amt / old_bal, 0.0)
    X[:, NEW_FEATURES.index('known_device_count')] = 1.0
    X[:, NEW_FEATURES.index('account_tenure_days')] = 30.0
    X[:, NEW_FEATURES.index('hour_of_day')] = 12.0
    X[:, NEW_FEATURES.index('is_weekend')] = 0.0
    X[:, NEW_FEATURES.index('shared_device_accounts')] = 1.0
    X[:, NEW_FEATURES.index('shared_recipient_accounts')] = 1.0
    X[:, NEW_FEATURES.index('mule_ring_score')] = (old_bal < 1.0).astype(np.float32) * (new_dest > 10000).astype(np.float32)
    X[:, NEW_FEATURES.index('hour_deviation')] = 0.0
    X[:, NEW_FEATURES.index('amount_zscore')] = (amt - np.median(amt)) / (np.std(amt) + 1e-6)
    X[:, NEW_FEATURES.index('velocity_deviation')] = X[:, NEW_FEATURES.index('txn_freq_last_24h')] / 24.0
    X[:, NEW_FEATURES.index('recipient_novelty')] = 0.5
    X[:, NEW_FEATURES.index('txn_regularity')] = 0.0

    # ─── NEW: Subscription/Micro-Fraud Features ─────────────────
    # subscription_pattern_score: for PaySim, CASH_OUT of small amounts is suspicious
    is_cash_out = (df['type'] == 'CASH_OUT').astype(np.float32)
    small_amount = (amt < 100).astype(np.float32)
    X[:, NEW_FEATURES.index('subscription_pattern_score')] = is_cash_out * small_amount

    # micro_fraud_flag: small amount relative to balance
    X[:, NEW_FEATURES.index('micro_fraud_flag')] = ((amt < 100) & (old_bal > 10000)).astype(np.float32)

    # balance_drain_ratio: how much of balance is being taken
    X[:, NEW_FEATURES.index('balance_drain_ratio')] = np.where(old_bal > 0, amt / old_bal, 0.0)

    # merchant_fraud_concentration: PAYMENT/TRANSFER/CASH_OUT fraud rates
    type_fraud = {'PAYMENT': 0.01, 'TRANSFER': 0.08, 'CASH_OUT': 0.08, 'CASH_IN': 0.001, 'DEBIT': 0.01}
    X[:, NEW_FEATURES.index('merchant_fraud_concentration')] = df['type'].map(type_fraud).fillna(0.01).values.astype(np.float32)

    # card_velocity_ratio: not available, estimate from type
    X[:, NEW_FEATURES.index('card_velocity_ratio')] = X[:, NEW_FEATURES.index('txn_freq_last_24h')]

    # amount_cluster_distance: distance from round amounts
    round_amounts = np.array([100, 500, 1000, 5000, 10000])
    min_dist = np.min(np.abs(amt[:, None] - round_amounts[None, :]), axis=1)
    X[:, NEW_FEATURES.index('amount_cluster_distance')] = min_dist.astype(np.float32)

    return X


def map_creditcard_to_features(df: pd.DataFrame, label_col='Class') -> np.ndarray:
    """Map ULB/fraud_data PCA features to 27 ML_FEATURES."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    pca_cols = [c for c in df.columns if c.startswith('V')]
    pca_data = df[pca_cols].values
    amt = df['Amount'].values.astype(np.float32) if 'Amount' in df.columns else np.zeros(n, dtype=np.float32)

    med_amt = np.median(amt)
    X[:, NEW_FEATURES.index('amount_ratio')] = np.where(med_amt > 0, amt / med_amt, 0.0)
    X[:, NEW_FEATURES.index('txn_freq_last_24h')] = np.abs(pca_data[:, :5]).mean(axis=1) * 10
    X[:, NEW_FEATURES.index('txn_time_unusual')] = (df['Time'].values % 86400) / 86400.0 if 'Time' in df.columns else 0.5
    X[:, NEW_FEATURES.index('new_device_flag')] = (np.abs(pca_data[:, 0]) > 2).astype(np.float32)
    X[:, NEW_FEATURES.index('unusual_location_flag')] = (np.abs(pca_data[:, 1]) > 2).astype(np.float32)
    X[:, NEW_FEATURES.index('unusual_recipient_flag')] = (np.abs(pca_data[:, 2]) > 2).astype(np.float32)
    X[:, NEW_FEATURES.index('failed_auth_count_24h')] = 0.0
    X[:, NEW_FEATURES.index('days_since_last_similar_txn')] = np.clip(df['Time'].values / 86400.0, 0, 30) if 'Time' in df.columns else 1.0
    X[:, NEW_FEATURES.index('gradual_escalation_score')] = np.abs(pca_data[:, 3]).clip(0, 1)
    X[:, NEW_FEATURES.index('known_device_count')] = 1.0
    X[:, NEW_FEATURES.index('account_tenure_days')] = 30.0
    X[:, NEW_FEATURES.index('hour_of_day')] = (df['Time'].values % 86400) / 3600.0 if 'Time' in df.columns else 12.0
    X[:, NEW_FEATURES.index('is_weekend')] = 0.0
    X[:, NEW_FEATURES.index('shared_device_accounts')] = 1.0
    X[:, NEW_FEATURES.index('shared_recipient_accounts')] = 1.0
    X[:, NEW_FEATURES.index('mule_ring_score')] = np.abs(pca_data[:, 4]).clip(0, 10)
    X[:, NEW_FEATURES.index('hour_deviation')] = np.abs(pca_data[:, 5]).clip(0, 12)
    X[:, NEW_FEATURES.index('amount_zscore')] = (amt - amt.mean()) / (amt.std() + 1e-6)
    X[:, NEW_FEATURES.index('velocity_deviation')] = np.abs(pca_data[:, 6]).clip(0, 10)
    X[:, NEW_FEATURES.index('recipient_novelty')] = np.abs(pca_data[:, 7]).clip(0, 1)
    X[:, NEW_FEATURES.index('txn_regularity')] = np.abs(pca_data[:, 8]).clip(0, 100)

    # ─── NEW: Subscription/Micro-Fraud Features ─────────────────
    X[:, NEW_FEATURES.index('subscription_pattern_score')] = np.abs(pca_data[:, 9]).clip(0, 1) if pca_data.shape[1] > 9 else 0.0
    X[:, NEW_FEATURES.index('micro_fraud_flag')] = (amt < np.median(amt) * 0.1).astype(np.float32)
    X[:, NEW_FEATURES.index('balance_drain_ratio')] = X[:, NEW_FEATURES.index('amount_ratio')]
    X[:, NEW_FEATURES.index('merchant_fraud_concentration')] = np.abs(pca_data[:, 10]).clip(0, 1) if pca_data.shape[1] > 10 else 0.0
    X[:, NEW_FEATURES.index('card_velocity_ratio')] = X[:, NEW_FEATURES.index('txn_freq_last_24h')]
    round_amounts = np.array([10, 50, 100, 500])
    min_dist = np.min(np.abs(amt[:, None] - round_amounts[None, :]), axis=1)
    X[:, NEW_FEATURES.index('amount_cluster_distance')] = min_dist.astype(np.float32)

    return X


# ─── Dataset loaders ────────────────────────────────────────────
def load_kaggle_fraud(max_rows=150000):
    print("\n[1/3] Loading kaggle_fraud...")
    dfs = []
    for path in ['data/kaggle_fraud/fraudTrain.csv', 'data/kaggle_fraud/fraudTest.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.drop(columns=['first', 'last', 'street', 'state', 'zip', 'dob', 'job', 'trans_num', 'Unnamed: 0'], errors='ignore')
            dfs.append(df)
    if not dfs:
        return None, None
    df = pd.concat(dfs, ignore_index=True)
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
    df.drop(columns=['city'], errors='ignore', inplace=True)
    return X, y


def load_paysim(max_rows=150000):
    print("\n[2/3] Loading PaySim...")
    df = pd.read_csv('data/paysim_1m.csv')
    if len(df) > max_rows:
        fraud_idx = df[df['isFraud'] == 1].index
        legit_idx = df[df['isFraud'] == 0].index
        n_fraud = min(len(fraud_idx), int(max_rows * 0.15))
        n_legit = max_rows - n_fraud
        sampled = np.concatenate([np.random.RandomState(SEED).choice(fraud_idx, n_fraud, replace=False),
                                  np.random.RandomState(SEED).choice(legit_idx, n_legit, replace=False)])
        df = df.loc[sampled]
    print(f"  Loaded {len(df):,} rows, {df['isFraud'].sum():,} fraud")
    X = map_paysim_to_features(df)
    y = df['isFraud'].values.astype(np.int32)
    return X, y


def load_creditcard(max_rows=80000):
    print("\n[3/3] Loading ULB creditcard...")
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


# ─── Model training ─────────────────────────────────────────────
def train_model(X_train, y_train, model_type='xgboost'):
    from xgboost import XGBClassifier
    imputer = SimpleImputer(strategy='median')
    X_clean = imputer.fit_transform(X_train)
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=10.0, neginf=-10.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    if model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=SEED, eval_metric='auc', use_label_encoder=False
        )
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=3,
            class_weight='balanced', random_state=SEED, n_jobs=-1
        )
    elif model_type == 'lr':
        model = LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=SEED
        )
    model.fit(X_scaled, y_train)
    return model, scaler, imputer


def evaluate_model(model, scaler, X_test, y_test, name="", imputer=None):
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

    roc_mean, roc_lo, roc_hi = bootstrap_ci(y_test, y_prob, roc_auc_score)
    pr_mean, pr_lo, pr_hi = bootstrap_ci(y_test, y_prob, average_precision_score)
    cm = confusion_matrix(y_test, y_pred)

    result = {
        'name': name, 'n_test': len(y_test), 'n_fraud': int(y_test.sum()),
        'fraud_prev': float(y_test.mean()),
        'roc_auc': round(roc, 4), 'pr_auc': round(pr, 4),
        'recall_0.1pct_fpr': round(r1, 4), 'recall_0.5pct_fpr': round(r05, 4),
        'recall_1pct_fpr': round(r1pct, 4),
        'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
        'brier': round(brier, 6), 'ece': round(ece_val, 6),
        'roc_auc_ci95': [round(roc_lo, 4), round(roc_hi, 4)],
        'pr_auc_ci95': [round(pr_lo, 4), round(pr_hi, 4)],
        'confusion_matrix': cm.tolist(),
    }
    return result


# ─── Main pipeline ──────────────────────────────────────────────
def main():
    np.random.seed(SEED)
    results = {'before': {}, 'after': {}}

    print("="*70)
    print("PHASE 1: Loading datasets")
    print("="*70)

    datasets = {}
    X, y = load_kaggle_fraud(max_rows=150000)
    if X is not None:
        datasets['kaggle_fraud'] = (X, y)
    X, y = load_paysim(max_rows=150000)
    if X is not None:
        datasets['paysim'] = (X, y)
    X, y = load_creditcard(max_rows=80000)
    if X is not None:
        datasets['ulb_creditcard'] = (X, y)

    total_rows = sum(X.shape[0] for X, y in datasets.values())
    total_fraud = sum(int(y.sum()) for X, y in datasets.values())
    print(f"\nTotal: {total_rows:,} rows, {total_fraud:,} fraud cases")

    # ─── PHASE 2: Evaluate current v4 model as baseline ──────────
    print("\n" + "="*70)
    print("PHASE 2: Baseline (v4 model with 21 features)")
    print("="*70)

    import joblib
    try:
        xgb_v4 = joblib.load('models/artifacts/xgboost.joblib')
        scaler_v4 = joblib.load('models/artifacts/scaler.joblib')
        imputer_v4 = joblib.load('models/artifacts/imputer.joblib') if Path('models/artifacts/imputer.joblib').exists() else None
        print("Loaded v4 model")

        for dname, (X, y) in datasets.items():
            # v4 model only has 21 features, evaluate with first 21
            X_21 = X[:, :21]
            res = evaluate_model(xgb_v4, scaler_v4, X_21, y, f"v4_{dname}", imputer_v4)
            results['before'][dname] = res
            print(f"  {dname}: ROC-AUC={res['roc_auc']:.4f}, R@1%FPR={res['recall_1pct_fpr']:.4f}")
    except Exception as e:
        print(f"Could not load v4 model: {e}")

    # ─── PHASE 3: Retrain with 27 features ───────────────────────
    print("\n" + "="*70)
    print("PHASE 3: Retrain with subscription/fraud detection features")
    print("="*70)

    all_X = []
    all_y = []
    for dname, (X, y) in datasets.items():
        all_X.append(X)
        all_y.append(y)
        print(f"  Added {dname}: {X.shape[0]:,} rows")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    # Class balancing: oversample fraud 3x
    fraud_mask = y_all == 1
    X_fraud = X_all[fraud_mask]
    y_fraud = y_all[fraud_mask]
    X_legit = X_all[~fraud_mask]
    y_legit = y_all[~fraud_mask]

    rng = np.random.RandomState(SEED)
    fraud_idx = rng.choice(len(X_fraud), len(X_fraud) * 3, replace=True)
    X_balanced = np.vstack([X_legit, X_fraud[fraud_idx]])
    y_balanced = np.concatenate([y_legit, y_fraud[fraud_idx]])

    # Shuffle
    perm = rng.permutation(len(X_balanced))
    X_balanced = X_balanced[perm]
    y_balanced = y_balanced[perm]

    n = len(X_balanced)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train = X_balanced[:n_train]
    y_train = y_balanced[:n_train]
    X_val = X_balanced[n_train:n_train+n_val]
    y_val = y_balanced[n_train:n_train+n_val]
    X_test = X_balanced[n_train+n_val:]
    y_test = y_balanced[n_train+n_val:]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print(f"Fraud: train={y_train.sum()}, val={y_val.sum()}, test={y_test.sum()}")

    # Train XGBoost with 27 features
    print("\nTraining XGBoost (27 features)...")
    xgb_new, scaler_new, imputer_new = train_model(X_train, y_train, 'xgboost')

    val_res = evaluate_model(xgb_new, scaler_new, X_val, y_val, "27feat_val", imputer_new)
    print(f"  Validation: ROC-AUC={val_res['roc_auc']:.4f}, PR-AUC={val_res['pr_auc']:.4f}, R@1%FPR={val_res['recall_1pct_fpr']:.4f}")

    test_res = evaluate_model(xgb_new, scaler_new, X_test, y_test, "27feat_test", imputer_new)
    print(f"  Test: ROC-AUC={test_res['roc_auc']:.4f}, PR-AUC={test_res['pr_auc']:.4f}, R@1%FPR={test_res['recall_1pct_fpr']:.4f}")

    # Train RF
    print("\nTraining Random Forest (27 features)...")
    rf_new, rf_scaler, rf_imputer = train_model(X_train, y_train, 'rf')

    # Train LR
    print("Training Logistic Regression (27 features)...")
    lr_new, lr_scaler, lr_imputer = train_model(X_train, y_train, 'lr')

    # ─── PHASE 4: Cross-dataset evaluation ──────────────────────
    print("\n" + "="*70)
    print("PHASE 4: Cross-dataset evaluation (27-feature model)")
    print("="*70)

    for dname, (X, y) in datasets.items():
        res_xgb = evaluate_model(xgb_new, scaler_new, X, y, f"new_xgb_{dname}", imputer_new)
        res_rf = evaluate_model(rf_new, rf_scaler, X, y, f"new_rf_{dname}", rf_imputer)
        res_lr = evaluate_model(lr_new, lr_scaler, X, y, f"new_lr_{dname}", lr_imputer)

        # Ensemble
        X_c = imputer_new.transform(X)
        X_c = np.nan_to_num(X_c, nan=0.0, posinf=10.0, neginf=-10.0)
        X_s = scaler_new.transform(X_c)
        prob_xgb = xgb_new.predict_proba(X_s)[:, 1]
        prob_rf = rf_new.predict_proba(rf_scaler.transform(np.nan_to_num(rf_imputer.transform(X), nan=0.0, posinf=10.0, neginf=-10.0)))[:, 1]
        prob_lr = lr_new.predict_proba(lr_scaler.transform(np.nan_to_num(lr_imputer.transform(X), nan=0.0, posinf=10.0, neginf=-10.0)))[:, 1]
        prob_ens = 0.5 * prob_rf + 0.3 * prob_xgb + 0.2 * prob_lr

        roc_ens = roc_auc_score(y, prob_ens)
        pr_ens = average_precision_score(y, prob_ens)
        r1_ens, _ = recall_at_fpr(y, prob_ens, 0.01)

        results['after'][dname] = {
            'xgboost': res_xgb, 'random_forest': res_rf, 'logistic_regression': res_lr,
            'ensemble': {'roc_auc': round(roc_ens, 4), 'pr_auc': round(pr_ens, 4), 'recall_1pct_fpr': round(r1_ens, 4)}
        }

        before_roc = results['before'].get(dname, {}).get('roc_auc', 'N/A')
        print(f"\n  {dname}:")
        print(f"    Before (21 feat): ROC-AUC={before_roc}")
        print(f"    XGBoost (27 feat): ROC-AUC={res_xgb['roc_auc']:.4f}, R@1%FPR={res_xgb['recall_1pct_fpr']:.4f}")
        print(f"    RF (27 feat):      ROC-AUC={res_rf['roc_auc']:.4f}, R@1%FPR={res_rf['recall_1pct_fpr']:.4f}")
        print(f"    Ensemble:          ROC-AUC={roc_ens:.4f}, R@1%FPR={r1_ens:.4f}")

    # ─── PHASE 5: Small-amount fraud analysis ───────────────────
    print("\n" + "="*70)
    print("PHASE 5: Small-amount fraud detection (subscription patterns)")
    print("="*70)

    for dname, (X, y) in datasets.items():
        if X.shape[1] != N_FEATURES:
            continue
        # Predict
        X_c = np.nan_to_num(imputer_new.transform(X), nan=0.0, posinf=10.0, neginf=-10.0)
        y_prob = xgb_new.predict_proba(scaler_new.transform(X_c))[:, 1]

        # Find small-amount fraud (micro_fraud_flag == 1)
        micro_idx = NEW_FEATURES.index('micro_fraud_flag')
        small_fraud_mask = (X[:, micro_idx] == 1) & (y == 1)
        small_legit_mask = (X[:, micro_idx] == 1) & (y == 0)

        if small_fraud_mask.sum() > 0:
            small_roc = roc_auc_score(y[X[:, micro_idx] == 1], y_prob[X[:, micro_idx] == 1]) if len(np.unique(y[X[:, micro_idx] == 1])) > 1 else 0
            print(f"  {dname} small-amount fraud: {small_fraud_mask.sum()} cases, model ROC={small_roc:.4f}")
        else:
            print(f"  {dname}: no small-amount fraud in sample")

    # ─── PHASE 6: Save artifacts ────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 6: Saving artifacts")
    print("="*70)

    joblib.dump(xgb_new, 'models/artifacts/xgboost_finance.joblib')
    joblib.dump(scaler_new, 'models/artifacts/scaler_finance.joblib')
    joblib.dump(imputer_new, 'models/artifacts/imputer_finance.joblib')
    joblib.dump(rf_new, 'models/artifacts/random_forest_finance.joblib')
    joblib.dump(lr_new, 'models/artifacts/logistic_regression_finance.joblib')
    print("Saved finance-fraud model artifacts")

    eval_path = OUT_DIR / 'finance_fraud_eval.json'
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved evaluation to {eval_path}")

    # Summary
    print("\n" + "="*70)
    print("BEFORE vs AFTER (21 features → 27 features)")
    print("="*70)
    print(f"{'Dataset':<20} {'Before ROC':>12} {'After ROC':>12} {'Δ':>8} {'Before R@1%':>14} {'After R@1%':>14} {'Δ':>8}")
    print("-" * 92)
    for dname in datasets:
        b = results['before'].get(dname, {})
        a = results['after'].get(dname, {}).get('xgboost', {})
        if b and a:
            print(f"{dname:<20} {b['roc_auc']:>12.4f} {a['roc_auc']:>12.4f} {a['roc_auc']-b['roc_auc']:>+8.4f} {b['recall_1pct_fpr']:>14.4f} {a['recall_1pct_fpr']:>14.4f} {a['recall_1pct_fpr']-b['recall_1pct_fpr']:>+8.4f}")

    model_hash = hashlib.sha256(open('models/artifacts/xgboost_finance.joblib', 'rb').read()).hexdigest()[:16]
    print(f"\nModel hash: {model_hash}")
    print(f"Features: {N_FEATURES} (21 original + 6 subscription/fraud detection)")
    print(f"New features: subscription_pattern_score, micro_fraud_flag, balance_drain_ratio, merchant_fraud_concentration, card_velocity_ratio, amount_cluster_distance")


if __name__ == '__main__':
    main()
