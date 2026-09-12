#!/usr/bin/env python3
"""Generate fraud report data for the landing page.
Uses the PS-14 model artifacts on kaggle_fraud test data.
"""
from pathlib import Path
import pandas as pd, numpy as np, joblib, json, warnings, os, sys
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

# Use current production models
model = joblib.load(ROOT / 'models' / 'artifacts' / 'xgboost.joblib')
scaler = joblib.load(ROOT / 'models' / 'artifacts' / 'scaler.joblib')
imputer = joblib.load(ROOT / 'models' / 'artifacts' / 'imputer.joblib')

ML_FEATURES = ['amount_ratio','txn_freq_last_24h','txn_time_unusual','new_device_flag','unusual_location_flag','unusual_recipient_flag','failed_auth_count_24h','days_since_last_similar_txn','gradual_escalation_score','known_device_count','account_tenure_days','hour_of_day','is_weekend','shared_device_accounts','shared_recipient_accounts','mule_ring_score','hour_deviation','amount_zscore','velocity_deviation','recipient_novelty','txn_regularity']

# Load kaggle test only (fast)
p = ROOT / 'data' / 'kaggle_fraud' / 'fraudTest.csv'
if not p.exists():
    print("fraudTest.csv not found")
    sys.exit(1)
df = pd.read_csv(p)
df = df.drop(columns=['first', 'last', 'street', 'state', 'zip', 'dob', 'job', 'trans_num', 'Unnamed: 0'], errors='ignore')

# Compute features
N = len(ML_FEATURES)
X = np.zeros((len(df), N), dtype=np.float32)
amt = df['amt'].values.astype(np.float32)
hours = pd.to_datetime(df['trans_date_trans_time']).dt.hour.values
dow = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek.values
unix = df['unix_time'].values.astype(np.float64)

card_median = df.groupby('cc_num')['amt'].transform('median').values
X[:, ML_FEATURES.index('amount_ratio')] = np.where(card_median > 0, amt / card_median, 0)
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
X[:, ML_FEATURES.index('days_since_last_similar_txn')] = np.clip(gaps[:len(df)], 0, 365).astype(np.float32)
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

X = imputer.transform(X)
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
X = scaler.transform(X)
df['ml_score'] = model.predict_proba(X)[:, 1]
df['hour'] = hours

df['is_fraud'] = df['is_fraud'].astype(int)
fraud = df[df.is_fraud == 1].copy()
legit = df[df.is_fraud == 0].copy()

def classify(row):
    s = row['ml_score']
    if s >= 0.9: return 'high_confidence_fraud'
    if s >= 0.5: return 'borderline_fraud'
    if s >= 0.05: return 'subtle_fraud'
    if row['is_fraud'] == 1: return 'missed_fraud'
    if s >= 0.3: return 'false_positive_risk'
    return 'legitimate'

df['category'] = df.apply(classify, axis=1)

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
y_true = df['is_fraud'].values
y_prob = df['ml_score'].values
fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
valid = fpr_arr <= 0.01
r1 = tpr_arr[np.where(valid)[0][-1]] if valid.any() else 0

summary = {
    'total': len(df),
    'fraud_count': int(fraud.shape[0]),
    'legit_count': int(legit.shape[0]),
    'fraud_pct': round(fraud.shape[0] / len(df) * 100, 3),
    'ml_auc': round(roc_auc_score(y_true, y_prob), 4),
    'pr_auc': round(average_precision_score(y_true, y_prob), 4),
    'recall_1pct_fpr': round(r1, 4),
    'categories': df['category'].value_counts().to_dict(),
}

# Feature importance
try:
    importance = model.feature_importances_
    feat_imp = [{'feature': ML_FEATURES[i], 'importance': round(float(importance[i]), 4)} for i in np.argsort(importance)[::-1][:10]]
except Exception:
    feat_imp = []

# Score distribution
bins = [0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
labels = ['0-1%', '1-5%', '5-10%', '10-30%', '30-50%', '50-70%', '70-90%', '90-100%']
df['score_bucket'] = pd.cut(df['ml_score'], bins=bins, labels=labels, right=False)
fraud_dist = df[df.is_fraud == 1]['score_bucket'].value_counts().reindex(labels, fill_value=0).to_dict()
legit_dist = df[df.is_fraud == 0]['score_bucket'].value_counts().reindex(labels, fill_value=0).to_dict()

# Amount distribution
amt_bins = [0, 1, 5, 25, 100, 500, 2500]
amt_labels = ['<$1', '$1-5', '$5-25', '$25-100', '$100-500', '$500+']
df['amt_bucket'] = pd.cut(df['amt'], bins=amt_bins, labels=amt_labels, right=False)
fraud_amt = df[df.is_fraud == 1]['amt_bucket'].value_counts().reindex(amt_labels, fill_value=0).to_dict()
legit_amt = df[df.is_fraud == 0]['amt_bucket'].value_counts().reindex(amt_labels, fill_value=0).to_dict()

# Hour distribution
fraud_hours = fraud['hour'].round(0).value_counts().sort_index().to_dict()
legit_hours = legit['hour'].round(0).value_counts().sort_index().to_dict()

# Top suspects
top_suspect = df.nlargest(10, 'ml_score')[['ml_score', 'amt', 'hour', 'is_fraud', 'category']].copy()
top_suspect.index = top_suspect.index.astype(int).tolist()
top_suspect['idx'] = top_suspect.index
top_records = top_suspect.to_dict('records')

# Missed frauds
missed = fraud[fraud.ml_score < 0.05][['ml_score', 'amt', 'hour']].copy()
missed.index = missed.index.astype(int).tolist()
missed['idx'] = missed.index
missed_records = missed.to_dict('records')

result = {
    'summary': summary,
    'feature_importance': feat_imp,
    'score_distribution': {'fraud': fraud_dist, 'legit': legit_dist, 'labels': labels},
    'amount_distribution': {'fraud': fraud_amt, 'legit': legit_amt, 'labels': amt_labels},
    'hour_distribution': {'fraud': fraud_hours, 'legit': legit_hours},
    'top_suspects': top_records,
    'chameleon_frauds': missed_records,
}

out = ROOT / 'data' / 'fraud_report_data.json'
out.write_text(json.dumps(result, default=str), encoding='utf-8')
print(f"Written {out} ({len(result['summary'])} keys, {summary['total']} rows)")
print(f"AUC={summary['ml_auc']}, R@1%FPR={summary['recall_1pct_fpr']}")
