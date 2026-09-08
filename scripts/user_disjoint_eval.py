#!/usr/bin/env python3
"""User-disjoint evaluation on IBM Altman (honest metrics)."""
import numpy as np, pandas as pd, time, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def recall_at(y, s, t):
    f, tt, _ = roc_curve(y, s)
    v = f <= t
    return float(tt[v][-1]) if v.any() else 0.0

def mk(sub):
    F = pd.DataFrame()
    F['amt_log'] = np.log1p(np.abs(sub['amt'].values))
    g = sub.groupby('User')
    cs = g['amt'].cumsum()
    c = g.cumcount() + 1
    F['um'] = (cs / c).values
    F['us'] = np.sqrt(np.clip(g['amt'].transform(lambda x: (x**2).cumsum()).values / c.values - F['um'].values**2, 0, None))
    F['uz'] = (F['amt_log'] - np.log1p(F['um'].clip(lower=1))) / (F['us'] + 1e-8)
    F['hr'] = sub['dt'].dt.hour.values.astype(np.float32)
    F['night'] = ((F['hr'] >= 22) | (F['hr'] <= 5)).astype(np.float32)
    F['biz'] = ((F['hr'] >= 9) & (F['hr'] <= 17)).astype(np.float32)
    F['hrs'] = np.clip(sub.groupby('User')['dt'].diff().dt.total_seconds().fillna(0).values / 3600, 0, 720).astype(np.float32)
    F['mcc'] = sub['MCC'].values.astype(np.float32)
    F['ms'] = sub.groupby(['User', 'Merchant Name']).cumcount().values.astype(np.float32)
    F['chip'] = sub['Use Chip'].map({'Swipe Transaction': 0, 'Chip Transaction': 1, 'Online Transaction': 2}).fillna(3).values.astype(np.float32)
    F['zip'] = sub['Zip'].notna().values.astype(np.float32)
    F['new_m'] = (F['ms'] == 0).astype(np.float32)
    F['amt_x_night'] = F['amt_log'] * F['night']
    F['amt_x_new'] = F['amt_log'] * F['new_m']
    return F.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

t0 = time.time()
print("Loading 24M rows...")
df = pd.read_csv(
    'data/credit_card_transactions-ibm_v2.csv',
    usecols=['User', 'Year', 'Month', 'Day', 'Time', 'Amount', 'Use Chip',
             'Merchant Name', 'MCC', 'Zip', 'Is Fraud?'],
    low_memory=False,
)
df['amt'] = df['Amount'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
df['is_fraud'] = (df['Is Fraud?'] == 'Yes').astype(int)
df['dt'] = pd.to_datetime(df[['Year', 'Month', 'Day']].assign(
    hour=df['Time'].str.split(':').str[0].astype(int), minute=0))
df = df.sort_values(['User', 'dt']).reset_index(drop=True)
print(f"  {len(df):,} rows, {df['is_fraud'].sum()} fraud in {time.time()-t0:.0f}s")

# User-disjoint split
users = df['User'].unique()
np.random.seed(42)
np.random.shuffle(users)
split = int(0.7 * len(users))
tu = set(users[:split])
te_u = set(users[split:])
tm = df['User'].isin(tu)
em = df['User'].isin(te_u)
print(f"  Train: {tm.sum():,} ({df.loc[tm, 'is_fraud'].sum()} fraud, {len(tu)} users)")
print(f"  Test:  {em.sum():,} ({df.loc[em, 'is_fraud'].sum()} fraud, {len(te_u)} users)")

print("  Features on train...")
td = df[tm].reset_index(drop=True)
Xtr = mk(td)
ytr = td['is_fraud'].values

print("  Features on test...")
ed = df[em].reset_index(drop=True)
Xte = mk(ed)
yte = ed['is_fraud'].values
print(f"  Feature matrix: {Xtr.shape}")

sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)

print(f"\n{'='*60}")
print("USER-DISJOINT EVALUATION (HONEST)")
print(f"{'='*60}")

# LR
t1 = time.time()
lr = LogisticRegression(C=10, class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(Xtr_s, ytr)
p = lr.predict_proba(Xte_s)[:, 1]
auc = roc_auc_score(yte, p)
pr = average_precision_score(yte, p)
r1 = recall_at(yte, p, 0.01)
r05 = recall_at(yte, p, 0.005)
r01 = recall_at(yte, p, 0.001)
print(f"  LR   AUC={auc:.4f}  PR={pr:.4f}  R@1%={r1:.4f}  R@0.5%={r05:.4f}  R@0.1%={r01:.4f}  ({time.time()-t1:.0f}s)")

# XGB
try:
    from xgboost import XGBClassifier
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    t1 = time.time()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
        scale_pos_weight=spw, random_state=42, eval_metric='aucpr',
        use_label_encoder=False, n_jobs=-1
    )
    xgb.fit(Xtr, ytr, verbose=False)
    p = xgb.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    pr = average_precision_score(yte, p)
    r1 = recall_at(yte, p, 0.01)
    r05 = recall_at(yte, p, 0.005)
    r01 = recall_at(yte, p, 0.001)
    print(f"  XGB  AUC={auc:.4f}  PR={pr:.4f}  R@1%={r1:.4f}  R@0.5%={r05:.4f}  R@0.1%={r01:.4f}  ({time.time()-t1:.0f}s)")
    
    # Feature importance
    print("\n  Top features:")
    cols = ['amt_log', 'um', 'us', 'uz', 'hr', 'night', 'biz', 'hrs', 'mcc', 'ms', 'chip', 'zip', 'new_m', 'amt_x_night', 'amt_x_new']
    imp = xgb.feature_importances_
    for i in np.argsort(imp)[::-1][:10]:
        print(f"    {cols[i]:15s} {imp[i]:.4f}")
except Exception as e:
    print(f"  XGB error: {e}")

print(f"\n  Total: {time.time()-t0:.0f}s")
