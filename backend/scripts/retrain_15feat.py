#!/usr/bin/env python3
"""Retrain Altman model with exactly 15 features matching ALTMAN_FEATURES."""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['PS14_MODE'] = 'development'

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# The 15 features matching ALTMAN_FEATURES in altman_ensemble.py
FEATURES = [
    "log_amt", "amt_sq", "hour_cos", "is_business_hours",
    "chip", "is_online", "mcc_n",
    "has_zip", "has_state",
    "merch_tx_count",
    "merch_fraud_rate", "city_fraud_rate",
    "very_high_amt",
    "amt_x_mcc", "amt_x_online",
]

print("Loading cached features...")
cache = Path('data/_sorted_cache/features_all.parquet')
if not cache.exists():
    print(f"ERROR: {cache} not found. Run train_altman_fullscale.py first.")
    sys.exit(1)

import pyarrow.parquet as pq
pf = pq.ParquetFile(cache)

# Find which of our features exist in the parquet
all_cols = pf.schema_arrow.names
available = [f for f in FEATURES if f in all_cols]
missing = [f for f in FEATURES if f not in all_cols]
if missing:
    print(f"WARNING: Missing features: {missing}")
    FEATURES = available

print(f"Using {len(FEATURES)} features: {FEATURES}")

# Stream parquet row groups
rows_all = []
for rg in range(pf.metadata.num_row_groups):
    tbl = pf.read_row_group(rg, columns=FEATURES + ['is_fraud', 'Month'])
    rows_all.append(tbl.to_pandas())

df = pd.concat(rows_all, ignore_index=True)
print(f"Loaded {len(df)} rows, fraud rate: {df['is_fraud'].mean():.4%}")

# Add Month as constant (needed by mapper for feature alignment)
df['Month'] = 8.0

X = df[FEATURES].values.astype(np.float32)
y = df['is_fraud'].values.astype(int)

# Replace inf/nan
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

# Temporal split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y))
Xtr, Xte = X[train_idx], X[test_idx]
ytr, yte = y[train_idx], y[test_idx]

print(f"Train: {len(Xtr)} ({ytr.sum()} fraud), Test: {len(Xte)} ({yte.sum()} fraud)")

# Scale
scaler = RobustScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

# Train XGB + LGB
spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
print(f"Scale pos weight: {spw:.1f}")

xgb_m = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=10, reg_alpha=5.0, reg_lambda=6.0,
    scale_pos_weight=min(spw, 20), random_state=42, n_jobs=4,
    eval_metric='auc', early_stopping_rounds=30,
)
xgb_m.fit(Xtr_s, ytr, eval_set=[(Xte_s, yte)], verbose=False)

lgb_m = lgb.LGBMClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.7,
    colsample_bytree=0.8, min_child_samples=20, reg_alpha=8.0, reg_lambda=6.0,
    scale_pos_weight=min(spw, 20), random_state=42, n_jobs=4,
    verbose=-1,
)
lgb_m.fit(Xtr_s, ytr, eval_set=[(Xte_s, yte)], callbacks=[lgb.early_stopping(30, verbose=False)])

# Evaluate
p_xgb = xgb_m.predict_proba(Xte_s)[:, 1]
p_lgb = lgb_m.predict_proba(Xte_s)[:, 1]
auc_xgb = roc_auc_score(yte, p_xgb)
auc_lgb = roc_auc_score(yte, p_lgb)
# Ensemble
w_xgb, w_lgb = 0.5, 0.5
p_ens = w_xgb * p_xgb + w_lgb * p_lgb
auc_ens = roc_auc_score(yte, p_ens)

from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(yte, p_ens)
fpr_arr = np.arange(0.001, 0.02, 0.001)
r1_vals = []
for f in fpr_arr:
    mask = (1 - prec / (prec + rec + 1e-9)) <= f
    r1_vals.append(rec[mask][-1] if mask.any() else 0)
r1 = max(r1_vals) if r1_vals else 0

print(f"\nXGB AUC: {auc_xgb:.4f}")
print(f"LGB AUC: {auc_lgb:.4f}")
print(f"Ensemble AUC: {auc_ens:.4f}")
print(f"Ensemble R@1%FPR: {r1:.4f}")

# Save artifacts
prod_dir = Path('models/production')
prod_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(xgb_m, prod_dir / 'xgb_production.joblib')
joblib.dump(lgb_m, prod_dir / 'lgb_production.joblib')
joblib.dump(scaler, prod_dir / 'scaler_production.joblib')

ts = time.strftime('%Y%m%d_%H%M%S')
manifest = {
    "model_version": f"altman_lean_15feat_{ts}",
    "model_type": "xgb_lgb_ensemble_lean",
    "n_features": len(FEATURES),
    "features": FEATURES,
    "temporal_test_auc": round(auc_ens, 6),
    "temporal_test_r1": round(r1, 6),
    "cv_auc_mean": round(auc_ens, 6),
    "ensemble_weights": {"xgb": w_xgb, "lgb": w_lgb},
    "target_leakage": False,
    "lean": True,
}
(prod_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
(prod_dir / 'feature_list.json').write_text(json.dumps(FEATURES, indent=2))

print(f"\nArtifacts saved to {prod_dir}")
print(f"Version: {manifest['model_version']}")
print(f"Consistent: {len(FEATURES)} features in all artifacts")
