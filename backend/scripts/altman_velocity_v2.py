#!/usr/bin/env python3
"""Altman velocity v2: fast numpy-based velocity + 25 Optuna trials.

Uses pre-sorted per-user history arrays with binary search for O(log n) lookup.
"""
import json, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, brier_score_loss, average_precision_score
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)
NJ = 4
R = Path("reports"); R.mkdir(exist_ok=True)

def raf(y, p, t):
    fpr, tpr, _ = roc_curve(y, p); m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0

def met(y, p):
    roc = roc_auc_score(y, p); pr = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    r1 = raf(y, p, 0.01); r05 = raf(y, p, 0.005); r01 = raf(y, p, 0.001)
    prec, rec, thrs = precision_recall_curve(y, p)
    f1s = 2*prec*rec/(prec+rec+1e-12); bi = np.argmax(f1s)
    return dict(roc_auc=round(roc,6), pr_auc=round(pr,6), brier=round(brier,6),
                r1=round(r1,6), r05=round(r05,6), r01=round(r01,6),
                f1=round(float(f1s[bi]),6))

# ═══ PASS 1: Sample rows ═══
print("  PASS 1: Sampling...")
t0 = time.time()
rng = np.random.RandomState(42)
rows_all = []; ci = 0
for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
    usecols=["User","Month","Day","Time","Amount","Use Chip","MCC","Errors?",
             "Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"],
    low_memory=False, chunksize=1_000_000):
    chunk["amt"] = chunk["Amount"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float)
    chunk["label"] = (chunk["Is Fraud?"]=="Yes").astype(int)
    chip_map = {"Swipe Transaction":0,"Online Transaction":1,"Chip Transaction":2}
    chunk["chip"] = chunk["Use Chip"].map(chip_map).fillna(-1)
    chunk["mcc_n"] = chunk["MCC"].astype(str).str[:4].astype(float)/10000
    chunk["err"] = (chunk["Errors?"].fillna("")!="").astype(int)
    tp = chunk["Time"].str.split(":",expand=True)
    chunk["hr"] = tp[0].astype(float); chunk["mn"] = tp[1].astype(float)
    chunk["merchant_id"] = chunk["Merchant Name"].astype("category").cat.codes
    chunk["city_id"] = chunk["Merchant City"].astype("category").cat.codes
    chunk["state_id"] = chunk["Merchant State"].fillna("UNK").astype("category").cat.codes
    chunk["card_n"] = chunk["Card"].astype(float)
    chunk["year_n"] = chunk["Year"].astype(float) - 2010
    chunk["is_online"] = (chunk["Merchant City"]=="ONLINE").astype(int)
    chunk["month_n"] = chunk["Month"].astype(float)
    chunk["day_n"] = chunk["Day"].astype(float)
    chunk["hr_bin"] = pd.cut(chunk["hr"], bins=[0,6,12,18,24], labels=[0,1,2,3]).astype(float)
    chunk["amt_log"] = np.log1p(chunk["amt"].clip(upper=1e9))
    chunk["day_decimal"] = chunk["Day"] + chunk["hr"]/24.0 + chunk["mn"]/1440.0
    fm = chunk["label"]==1
    rows_all.append(pd.concat([chunk[fm], chunk[~fm].sample(frac=0.005, random_state=rng)]))
    ci += 1
    if ci >= 16: break

df = pd.concat(rows_all, ignore_index=True)
print(f"  Loaded {len(df):,} rows ({int(df['label'].sum())} fraud) in {time.time()-t0:.1f}s")
sampled_users = set(df["User"].unique())

# ═══ PASS 2: Full histories for sampled users ═══
print(f"\n  PASS 2: Full histories ({len(sampled_users)} users)...")
t1 = time.time()
user_times = defaultdict(list)
user_amounts = defaultdict(list)
user_merchants = defaultdict(list)
ci = 0
for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
    usecols=["User","Day","Time","Amount","Merchant Name"],
    low_memory=False, chunksize=1_000_000):
    mask = chunk["User"].isin(sampled_users)
    sub = chunk[mask]
    if len(sub) == 0: ci += 1; continue
    sub_amt = sub["Amount"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float).values
    tp = sub["Time"].str.split(":",expand=True)
    hr = tp[0].astype(float).values; mn = tp[1].astype(float).values
    dd = sub["Day"].values + hr/24.0 + mn/1440.0
    mid = sub["Merchant Name"].astype("category").cat.codes.values
    uids = sub["User"].values
    for i in range(len(sub)):
        user_times[uids[i]].append(dd[i])
        user_amounts[uids[i]].append(sub_amt[i])
        user_merchants[uids[i]].append(mid[i])
    ci += 1

# Sort each user's history by time
for uid in user_times:
    order = np.argsort(user_times[uid])
    user_times[uid] = np.array(user_times[uid])[order]
    user_amounts[uid] = np.array(user_amounts[uid])[order]
    user_merchants[uid] = np.array(user_merchants[uid])[order]
print(f"  Histories built in {time.time()-t1:.1f}s")

# ═══ Vectorized velocity computation ═══
print(f"\n  Computing velocity features...")
t2 = time.time()
n = len(df)
vel = np.zeros((n, 10), dtype=np.float32)
# Columns: tx_count_1h, tx_count_24h, amount_accel, merchant_diversity_24h,
#          avg_amount_24h, tx_gap_mean, tx_regularity, amount_zscore,
#          tx_freq_accel, merchant_is_new

for i in range(n):
    uid = int(df.iloc[i]["User"])
    dd = df.iloc[i]["day_decimal"]
    t_arr = user_times.get(uid)
    if t_arr is None or len(t_arr) < 3:
        continue
    a_arr = user_amounts[uid]
    m_arr = user_merchants[uid]

    # Binary search: find position
    pos = np.searchsorted(t_arr, dd, side='right')

    # All transactions strictly before this one
    t_before = t_arr[:pos]
    a_before = a_arr[:pos]
    m_before = m_arr[:pos]
    nb = len(t_before)
    if nb < 2:
        continue

    # tx_count_1h
    vel[i, 0] = ((dd - t_before) <= (1.0/24.0)).sum()
    # tx_count_24h
    vel[i, 1] = ((dd - t_before) <= 1.0).sum()
    # amount_accel
    if nb >= 10:
        vel[i, 2] = a_before[-5:].mean() - a_before[-10:-5].mean()
    elif nb >= 5:
        vel[i, 2] = a_before[-5:].mean() - a_before[:-5].mean()
    # merchant_diversity_24h
    last24 = (dd - t_before) <= 1.0
    vel[i, 3] = len(np.unique(m_before[last24]))
    # avg_amount_24h
    vel[i, 4] = a_before[last24].mean() if last24.sum() > 0 else 0
    # tx_gap_mean (hours between last 5)
    gaps = np.diff(t_before[-min(5,nb):]) * 24
    vel[i, 5] = gaps.mean() if len(gaps) > 0 else 0
    # tx_regularity (cv of gaps)
    vel[i, 6] = gaps.std() / (gaps.mean() + 1e-6) if len(gaps) > 1 else 0
    # amount_zscore
    if nb >= 5:
        rmean = a_before[-5:].mean()
        rstd = a_before[-5:].std() + 1e-6
        vel[i, 7] = (a_before[-1] - rmean) / rstd
    # tx_freq_accel
    if nb >= 6:
        f_recent = 1.0 / (gaps[-min(3,len(gaps)):].mean() + 0.01)
        f_prev = 1.0 / (np.diff(t_before[-6:-3]) * 24).mean() + 0.01 if nb >= 6 else f_recent
        vel[i, 8] = f_recent - f_prev
    # merchant_is_new: 1 if this merchant wasn't in previous 10 tx
    if nb >= 2:
        vel[i, 9] = 1.0 if m_before[-1] not in m_before[:-1][-min(10,nb-1):] else 0.0

    if (i+1) % 20000 == 0:
        print(f"    ...{i+1}/{n} ({time.time()-t2:.0f}s)")

print(f"  Velocity done in {time.time()-t2:.1f}s")

# Add to df
vel_names = ["tx_count_1h","tx_count_24h","amount_accel","merchant_diversity_24h",
    "avg_amount_24h","tx_gap_mean","tx_regularity","amount_zscore","tx_freq_accel","merchant_is_new"]
for j, name in enumerate(vel_names):
    df[name] = vel[:, j]

# Interactions
df["amt_x_tx5"] = df["amt"] * df["tx_count_1h"]
df["amt_x_tx24"] = df["amt"] * df["tx_count_24h"]
df["ute_x_tx24"] = df.get("user_te", 0) * df["tx_count_24h"]

# ═══ Build X, y ═══
y = df["label"].values
drop_cols = {"label","User","Time","Amount","Use Chip","MCC","Errors?",
    "Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"}
cols = [c for c in df.columns if c not in drop_cols]
X = np.nan_to_num(df[cols].values.astype(np.float32))
print(f"\n  Features: {len(cols)}")

# ═══ Optuna (25 trials) ═══
print(f"\n  Optuna (25 trials, 3-fold, optimize R@1%FPR)...")
t3 = time.time()

def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_est", 200, 700),
        max_depth=trial.suggest_int("max_depth", 6, 12),
        learning_rate=trial.suggest_float("lr", 0.02, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.65, 1.0),
        colsample_bytree=trial.suggest_float("colsample", 0.4, 0.9),
        min_child_samples=trial.suggest_int("min_child", 3, 25),
        reg_alpha=trial.suggest_float("alpha", 1e-3, 5, log=True),
        reg_lambda=trial.suggest_float("lambda", 0.05, 10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 20, 150),
        verbose=-1, random_state=42, n_jobs=NJ,
    )
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    r1s = []
    for tri, tei in skf.split(X, y):
        m = lgb.LGBMClassifier(**params).fit(X[tri], y[tri])
        p = m.predict_proba(X[tei])[:,1]
        r1s.append(raf(y[tei], p, 0.01))
    return np.mean(r1s)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=25, show_progress_bar=False)
print(f"  Best R@1%FPR: {study.best_value:.6f} ({time.time()-t3:.0f}s)")

# ═══ 5-fold CV ═══
bp = study.best_params
lp = dict(n_estimators=bp["n_est"], max_depth=bp["max_depth"], learning_rate=bp["lr"],
    subsample=bp["subsample"], colsample_bytree=bp["colsample"],
    min_child_samples=bp["min_child"], reg_alpha=bp["alpha"], reg_lambda=bp["lambda"],
    num_leaves=bp["num_leaves"], verbose=-1, random_state=42, n_jobs=NJ)

print(f"\n  5-fold CV...")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
all_p = np.zeros(len(y))
f_aucs = []; f_r1s = []
for fold, (tri, tei) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(**lp).fit(X[tri], y[tri])
    p = m.predict_proba(X[tei])[:,1]
    all_p[tei] = p
    a = roc_auc_score(y[tei], p); r = raf(y[tei], p, 0.01)
    f_aucs.append(a); f_r1s.append(r)
    print(f"    Fold {fold}: AUC={a:.6f} R@1%FPR={r:.4f}")

oa = roc_auc_score(y, all_p)
mm = met(y, all_p)
print(f"\n  CV AUC:     {np.mean(f_aucs):.6f} ± {np.std(f_aucs):.6f}")
print(f"  CV R@1%FPR: {np.mean(f_r1s):.6f} ± {np.std(f_r1s):.6f}")
print(f"  OOF AUC:    {oa:.6f}")
for k,v in mm.items(): print(f"    {k}: {v}")

target = np.mean(f_r1s) >= 0.92
print(f"\n  Target R@1%FPR>92%: {'ACHIEVED' if target else 'NOT MET'} ({np.mean(f_r1s)*100:.2f}%)")

# Feature importance
fi_m = lgb.LGBMClassifier(**lp).fit(X, y)
fi = dict(zip(cols, fi_m.feature_importances_))
top15 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
vel_set = set(vel_names + ["amt_x_tx5","amt_x_tx24","ute_x_tx24"])
print(f"\n  Top 15:")
for nm, imp in top15:
    print(f"    {nm:30s}: {imp:6d}{' ← VEL' if nm in vel_set else ''}")

result = dict(dataset="ALTMAN_VELOCITY_V2", n_rows=len(y), n_features=len(cols),
    n_fraud=int(y.sum()), best_r1=round(study.best_value,6), best_params=bp,
    cv_auc=round(float(np.mean(f_aucs)),6), cv_auc_std=round(float(np.std(f_aucs)),6),
    cv_r1=round(float(np.mean(f_r1s)),6), cv_r1_std=round(float(np.std(f_r1s)),6),
    oof_auc=round(oa,6), metrics=mm, target_r1_92=bool(target),
    top15=[{"n":n,"i":int(v)} for n,v in top15],
    elapsed=round(time.time()-t0),
    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
with open(R/"altman_velocity_v2.json","w") as f: json.dump(result, f, indent=2)
print(f"  Saved: reports/altman_velocity_v2.json")
print(f"  Total: {time.time()-t0:.0f}s")
