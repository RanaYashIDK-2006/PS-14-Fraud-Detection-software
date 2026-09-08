#!/usr/bin/env python3
"""Process ALL 24M Altman rows with velocity features.

Strategy:
  1. Load all 24M rows (24 chunks), keep ALL fraud + sample 2% legit (~700K)
  2. Build full per-user histories from ALL rows (for accurate velocity)
  3. Compute velocity features using numpy binary search on sampled rows
  4. Optuna LGB tuning + 3-fold CV

This measures the true ceiling with the complete dataset.
"""
import json, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)
NJ = 4
R = Path("reports"); R.mkdir(exist_ok=True)

def raf(y, p, t):
    fpr, tpr, _ = roc_curve(y, p); m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0

# ═══ PASS 1: Load ALL 24M rows, sample, build histories ═══
print("=" * 60)
print("  PASS 1: Loading ALL 24M Altman rows")
print("=" * 60)
t0 = time.time()
rng = np.random.RandomState(42)

# We need two things:
#   a) Sampled rows with base features (for training)
#   b) Full user histories (for velocity computation)

# Strategy: single pass through all chunks
#   - Keep ALL fraud rows
#   - Sample 2% of legit rows
#   - Build per-user history for ALL rows (needed for accurate velocity)

rows_sampled = []  # sampled rows with base features
user_times = defaultdict(list)
user_amounts = defaultdict(list)
user_merchants = defaultdict(list)

ci = 0
for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
    usecols=["User","Month","Day","Time","Amount","Use Chip","MCC","Errors?",
             "Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"],
    low_memory=False, chunksize=1_000_000):

    # Base features
    c = chunk.copy()
    c["amt"] = c["Amount"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float)
    c["label"] = (c["Is Fraud?"]=="Yes").astype(int)
    chip_map = {"Swipe Transaction":0,"Online Transaction":1,"Chip Transaction":2}
    c["chip"] = c["Use Chip"].map(chip_map).fillna(-1)
    c["mcc_n"] = c["MCC"].astype(str).str[:4].astype(float)/10000
    c["err"] = (c["Errors?"].fillna("")!="").astype(int)
    tp = c["Time"].str.split(":",expand=True)
    c["hr"] = tp[0].astype(float); c["mn"] = tp[1].astype(float)
    c["merchant_id"] = c["Merchant Name"].astype("category").cat.codes
    c["city_id"] = c["Merchant City"].astype("category").cat.codes
    c["state_id"] = c["Merchant State"].fillna("UNK").astype("category").cat.codes
    c["card_n"] = c["Card"].astype(float)
    c["year_n"] = c["Year"].astype(float)-2010
    c["is_online"] = (c["Merchant City"]=="ONLINE").astype(int)
    c["month_n"] = c["Month"].astype(float)
    c["day_n"] = c["Day"].astype(float)
    c["hr_bin"] = pd.cut(c["hr"], bins=[0,6,12,18,24], labels=[0,1,2,3]).astype(float)
    c["amt_log"] = np.log1p(c["amt"].clip(upper=1e9))
    c["day_decimal"] = c["Day"]+c["hr"]/24.0+c["mn"]/1440.0

    # Sample: keep ALL fraud, sample 2% of legit
    fm = c["label"] == 1
    fraud_df = c[fm]
    legit_sample = c[~fm].sample(frac=0.02, random_state=rng)
    rows_sampled.append(pd.concat([fraud_df, legit_sample]))

    # Build history for ALL rows (not just sampled)
    sub_amt = c["amt"].values
    dd = c["day_decimal"].values
    mid = c["merchant_id"].values
    uids = c["User"].values
    for i in range(len(c)):
        user_times[uids[i]].append(dd[i])
        user_amounts[uids[i]].append(sub_amt[i])
        user_merchants[uids[i]].append(mid[i])

    ci += 1
    if ci % 5 == 0:
        elapsed = time.time() - t0
        n_sampled = sum(len(r) for r in rows_sampled)
        n_hist = sum(len(v) for v in user_times.values())
        print(f"    Chunk {ci}/24: {n_sampled:,} sampled, {n_hist:,} history, {elapsed:.0f}s")
    if ci >= 24:
        break

df = pd.concat(rows_sampled, ignore_index=True)
y = df["label"].values
print(f"\n  Sampled: {len(df):,} rows ({int(y.sum())} fraud, {y.mean()*100:.3f}%)")
print(f"  Users: {len(user_times):,}")
total_hist = sum(len(v) for v in user_times.values())
print(f"  Full history: {total_hist:,} transactions ({time.time()-t0:.1f}s)")

# Sort histories
print("  Sorting histories...")
t1 = time.time()
user_hist = {}
for uid in user_times:
    order = np.argsort(user_times[uid])
    user_hist[uid] = {
        "times": np.array(user_times[uid])[order],
        "amounts": np.array(user_amounts[uid])[order],
        "merchants": np.array(user_merchants[uid])[order],
    }
# Free memory
del user_times, user_amounts, user_merchants
print(f"  Sorted in {time.time()-t1:.1f}s")

# ═══ PASS 2: Target encoding + interactions ═══
print("\n  PASS 2: Feature engineering...")
t2 = time.time()

gm = df["label"].mean()
for ec, gc, p in [("user_te","User",200),("merchant_te","merchant_id",50),("city_te","city_id",20)]:
    oof = np.zeros(len(df))
    for tri, vai in StratifiedShuffleSplit(2, test_size=0.2, random_state=42).split(df, df["label"]):
        stats = df.iloc[tri].groupby(gc)["label"].agg(["mean","count"])
        stats["s"] = (stats["mean"]*stats["count"]+gm*p)/(stats["count"]+p)
        oof[vai] = df.iloc[vai][gc].map(stats["s"].to_dict()).fillna(gm).values
    df[ec] = oof

df["log_amt"] = np.log1p(df["amt"])
df["amt_x_hr"] = df["amt"]*df["hr"]
df["amt_x_mcc"] = df["amt"]*df["mcc_n"]
df["amt_x_chip"] = df["amt"]*df["chip"]
df["amt_x_ute"] = df["amt"]*df["user_te"]
df["amt_x_mte"] = df["amt"]*df["merchant_te"]
df["amt_x_online"] = df["amt"]*df["is_online"]
df["hr_x_chip"] = df["hr"]*df["chip"]
df["ute_x_mte"] = df["user_te"]*df["merchant_te"]
df["chip_x_online"] = df["chip"]*df["is_online"]
df["err_x_amt"] = df["err"]*df["amt"]
df["hr_x_ute"] = df["hr"]*df["user_te"]
df["mcc_x_ute"] = df["mcc_n"]*df["user_te"]
df["amt_sq"] = df["amt"]**2
df["high_amt"] = (df["amt"]>df["amt"].quantile(0.95)).astype(float)
df["night_tx"] = ((df["hr"]>=22)|(df["hr"]<=6)).astype(int)
df["weekend"] = ((df["Day"]%7)>=5).astype(float)

# ═══ PASS 3: Velocity features (numpy binary search) ═══
print(f"\n  PASS 3: Velocity features ({len(df):,} rows)...")
t3 = time.time()
n = len(df)
vel = np.zeros((n, 10), dtype=np.float32)

for i in range(n):
    uid = int(df.iloc[i]["User"])
    dd = df.iloc[i]["day_decimal"]
    h = user_hist.get(uid)
    if h is None or len(h["times"]) < 3:
        continue
    t_arr = h["times"]
    a_arr = h["amounts"]
    m_arr = h["merchants"]
    pos = np.searchsorted(t_arr, dd, side="right")
    t_b = t_arr[:pos]; a_b = a_arr[:pos]; m_b = m_arr[:pos]
    nb = len(t_b)
    if nb < 2:
        continue
    vel[i, 0] = ((dd-t_b)<=(1.0/24.0)).sum()
    vel[i, 1] = ((dd-t_b)<=1.0).sum()
    if nb >= 10:
        vel[i, 2] = a_b[-5:].mean()-a_b[-10:-5].mean()
    elif nb >= 5:
        vel[i, 2] = a_b[-5:].mean()-a_b[:-5].mean()
    last24 = (dd-t_b)<=1.0
    vel[i, 3] = len(np.unique(m_b[last24]))
    vel[i, 4] = a_b[last24].mean() if last24.sum()>0 else 0
    gaps = np.diff(t_b[-min(5,nb):])*24
    vel[i, 5] = gaps.mean() if len(gaps)>0 else 0
    vel[i, 6] = gaps.std()/(gaps.mean()+1e-6) if len(gaps)>1 else 0
    if nb >= 5:
        vel[i, 7] = (a_b[-1]-a_b[-5:].mean())/(a_b[-5:].std()+1e-6)
    if nb >= 6:
        f_rec = 1.0/(gaps[-min(3,len(gaps)):].mean()+0.01)
        f_prev = 1.0/(np.diff(t_b[-6:-3])*24).mean()+0.01
        vel[i, 8] = f_rec-f_prev
    if nb >= 2:
        vel[i, 9] = 1.0 if m_b[-1] not in m_b[:-1][-min(10,nb-1):] else 0.0
    if (i+1)%50000==0:
        print(f"    ...{i+1}/{n} ({time.time()-t3:.0f}s)")

vel_names = ["tx_count_1h","tx_count_24h","amount_accel","merchant_diversity_24h",
    "avg_amount_24h","tx_gap_mean","tx_regularity","amount_zscore","tx_freq_accel","merchant_is_new"]
for j, name in enumerate(vel_names):
    df[name] = vel[:, j]

df["amt_x_tx5"] = df["amt"]*df["tx_count_1h"]
df["amt_x_tx24"] = df["amt"]*df["tx_count_24h"]
df["ute_x_tx24"] = df["user_te"]*df["tx_count_24h"]

print(f"  Velocity done in {time.time()-t3:.1f}s")

# Build X, y
drop_cols = {"label","User","Time","Amount","Use Chip","MCC","Errors?",
    "Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"}
cols = [c for c in df.columns if c not in drop_cols]
X = np.nan_to_num(df[cols].values.astype(np.float32))
print(f"  Features: {len(cols)}")

# ═══ PASS 4: Optuna (20 trials, 3-fold on subsample) ═══
print(f"\n  PASS 4: Optuna (20 trials, 3-fold)...")
t4 = time.time()
spw = (len(y)-int(y.sum()))/max(int(y.sum()),1)

def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("ne",200,600),
        max_depth=trial.suggest_int("md",6,12),
        learning_rate=trial.suggest_float("lr",0.02,0.2,log=True),
        subsample=trial.suggest_float("ss",0.65,1.0),
        colsample_bytree=trial.suggest_float("cs",0.5,0.9),
        min_child_samples=trial.suggest_int("mc",5,25),
        reg_alpha=trial.suggest_float("a",1e-3,5,log=True),
        reg_lambda=trial.suggest_float("l",0.1,10,log=True),
        num_leaves=trial.suggest_int("nl",20,127),
        verbose=-1, random_state=42, n_jobs=NJ,
        scale_pos_weight=min(spw,20),
    )
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    r1s = []
    for tri, tei in skf.split(X, y):
        m = lgb.LGBMClassifier(**params).fit(X[tri], y[tri])
        p = m.predict_proba(X[tei])[:,1]
        r1s.append(raf(y[tei], p, 0.01))
    return np.mean(r1s)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, show_progress_bar=False)
print(f"  Best R@1%FPR: {study.best_value:.6f} ({time.time()-t4:.0f}s)")

# ═══ PASS 5: 3-fold CV with best params ═══
bp = study.best_params
lp = dict(n_estimators=bp["ne"], max_depth=bp["md"], learning_rate=bp["lr"],
    subsample=bp["ss"], colsample_bytree=bp["cs"], min_child_samples=bp["mc"],
    reg_alpha=bp["a"], reg_lambda=bp["l"], num_leaves=bp["nl"],
    verbose=-1, random_state=42, n_jobs=NJ, scale_pos_weight=min(spw,20))

print(f"\n  PASS 5: 3-fold CV with best params...")
skf = StratifiedKFold(3, shuffle=True, random_state=42)
all_p = np.zeros(len(y))
f_aucs = []; f_r1s = []
for fold, (tri, tei) in enumerate(skf.split(X, y)):
    tf = time.time()
    m = lgb.LGBMClassifier(**lp).fit(X[tri], y[tri])
    p = m.predict_proba(X[tei])[:,1]
    all_p[tei] = p
    a = roc_auc_score(y[tei], p); r = raf(y[tei], p, 0.01)
    f_aucs.append(a); f_r1s.append(r)
    print(f"    Fold {fold}: AUC={a:.6f} R@1%FPR={r:.4f} ({time.time()-tf:.0f}s)")

oa = roc_auc_score(y, all_p)
print(f"\n  3-fold CV AUC:     {np.mean(f_aucs):.6f} ± {np.std(f_aucs):.6f}")
print(f"  3-fold CV R@1%FPR: {np.mean(f_r1s):.6f} ± {np.std(f_r1s):.6f}")
print(f"  OOF AUC:           {oa:.6f}")

# ═══ Compare vs 16-chunk baseline ═══
print(f"\n{'='*60}")
print(f"  COMPARISON: 24M (full) vs 16M (sampled)")
print(f"{'='*60}")
print(f"  16-chunk baseline: AUC=0.9948 R@1%FPR=0.9246 (from altman_velocity_v2)")
print(f"  24M full dataset:  AUC={np.mean(f_aucs):.6f} R@1%FPR={np.mean(f_r1s):.6f}")
delta_auc = np.mean(f_aucs) - 0.9948
delta_r1 = np.mean(f_r1s) - 0.9246
print(f"  Delta: AUC={delta_auc:+.4f} R@1%FPR={delta_r1:+.4f}")

result = dict(dataset="ALTMAN_FULL_24M", n_rows=len(y), n_features=len(cols),
    n_fraud=int(y.sum()), fraud_rate=round(float(y.mean()),6),
    full_dataset_chunks=ci, full_dataset_rows_approx=ci*1_000_000,
    history_users=len(user_hist), history_transactions=total_hist,
    optuna_best_r1=round(study.best_value,6), best_params=bp,
    cv_auc=round(float(np.mean(f_aucs)),6), cv_auc_std=round(float(np.std(f_aucs)),6),
    cv_r1=round(float(np.mean(f_r1s)),6), cv_r1_std=round(float(np.std(f_r1s)),6),
    oof_auc=round(oa,6),
    vs_16chunk=dict(delta_auc=round(delta_auc,4), delta_r1=round(delta_r1,4)),
    elapsed=round(time.time()-t0),
    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
with open(R/"altman_full_24m_results.json","w") as f: json.dump(result, f, indent=2)
print(f"\n  Saved: reports/altman_full_24m_results.json")
print(f"  Total: {time.time()-t0:.0f}s")
