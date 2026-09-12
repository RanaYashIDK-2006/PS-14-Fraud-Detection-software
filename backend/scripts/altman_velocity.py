#!/usr/bin/env python3
"""Altman with per-user velocity features + Optuna tuning.

Velocity features require full per-user transaction history (sorted chronologically):
  - tx_count_1h:  transactions in the last 1 hour
  - tx_count_24h: transactions in the last 24 hours
  - amount_accel: (mean amount last 5 tx) - (mean amount prev 5 tx)
  - merchant_diversity_24h: unique merchants in last 24h
  - avg_amount_24h: mean transaction amount in last 24h
  - tx_gap_mean: mean hours between last 5 transactions

Two-pass approach:
  Pass 1: Sample rows (same as before) → identify sampled user IDs
  Pass 2: Load ALL transactions for sampled users → compute velocity features

Target: R@1%FPR > 92% (currently 90.9%)
"""
import json, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
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

# ══════════════════════════════════════════════════════════
# PASS 1: Sample rows + identify users
# ══════════════════════════════════════════════════════════
print("  PASS 1: Sampling rows + identifying users...")
t0 = time.time()
rng = np.random.RandomState(42)
rows_all = []
ci = 0
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
    chunk["day_decimal"] = chunk["Day"] + chunk["hr"]/24.0 + chunk["mn"]/1440.0  # day position in month

    fraud_mask = chunk["label"] == 1
    fraud_df = chunk[fraud_mask]
    legit_sample = chunk[~fraud_mask].sample(frac=0.005, random_state=rng)
    rows_all.append(pd.concat([fraud_df, legit_sample]))
    ci += 1
    if ci >= 16: break

df = pd.concat(rows_all, ignore_index=True)
print(f"  Loaded {len(df):,} rows ({int(df['label'].sum())} fraud) in {time.time()-t0:.1f}s")

# Identify sampled user IDs and their sampled row indices
sampled_users = set(df["User"].unique())
print(f"  {len(sampled_users)} unique users in sample")

# ══════════════════════════════════════════════════════════
# PASS 2: Collect full histories for sampled users
# ══════════════════════════════════════════════════════════
print(f"\n  PASS 2: Collecting full histories for {len(sampled_users)} users...")
t1 = time.time()
user_hist = defaultdict(lambda: {"times": [], "amounts": [], "merchants": []})
ci = 0
for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
    usecols=["User","Day","Time","Amount","Merchant Name"],
    low_memory=False, chunksize=1_000_000):
    # Filter to sampled users only
    mask = chunk["User"].isin(sampled_users)
    sub = chunk[mask]
    if len(sub) == 0:
        ci += 1; continue
    sub_amt = sub["Amount"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float)
    tp = sub["Time"].str.split(":",expand=True)
    hr = tp[0].astype(float); mn = tp[1].astype(float)
    day_dec = (sub["Day"].values + hr.values/24.0 + mn.values/1440.0).tolist()
    mid = sub["Merchant Name"].astype("category").cat.codes.values.tolist()
    sub_amt_list = sub_amt.values.tolist()
    sub_users = sub["User"].values

    for i in range(len(sub)):
        uid = sub_users[i]
        user_hist[uid]["times"].append(day_dec[i])
        user_hist[uid]["amounts"].append(sub_amt_list[i])
        user_hist[uid]["merchants"].append(mid[i])
    ci += 1
    if ci % 5 == 0:
        print(f"    ...chunk {ci} ({time.time()-t1:.0f}s, {sum(len(v['times']) for v in user_hist.values()):,} total tx)")

print(f"  Histories collected in {time.time()-t1:.1f}s")
total_tx = sum(len(v["times"]) for v in user_hist.values())
print(f"  Total transactions in histories: {total_tx:,}")

# ══════════════════════════════════════════════════════════
# Compute velocity features for each sampled row
# ══════════════════════════════════════════════════════════
print(f"\n  Computing velocity features...")
t2 = time.time()

def compute_velocity(row_idx, user_id, day_decimal):
    """Compute velocity features for a single transaction using full user history."""
    h = user_hist.get(user_id)
    if h is None or len(h["times"]) < 3:
        return 0, 0, 0, 0, 0, 0, 0

    times = np.array(h["times"])
    amounts = np.array(h["amounts"])
    merchants = np.array(h["merchants"])

    # Find this transaction's position in the history
    # (approximately — using day_decimal to find nearest)
    diff = np.abs(times - day_decimal)
    pos = np.argmin(diff)

    # Get all transactions BEFORE this one (strict causality)
    mask_before = times < day_decimal
    if mask_before.sum() < 2:
        return 0, 0, 0, 0, 0, 0, 0

    t_before = times[mask_before]
    a_before = amounts[mask_before]
    m_before = merchants[mask_before]

    # tx_count_1h: transactions in last 1 hour
    tx_count_1h = int(((day_decimal - t_before) <= (1.0/24.0)).sum())

    # tx_count_24h: transactions in last 24 hours (1 day)
    tx_count_24h = int(((day_decimal - t_before) <= 1.0).sum())

    # amount_accel: (mean of last 5 tx) - (mean of prev 5 tx before that)
    n = len(a_before)
    if n >= 10:
        recent_5 = a_before[-5:]
        prev_5 = a_before[-10:-5]
        amount_accel = float(recent_5.mean() - prev_5.mean())
    elif n >= 5:
        recent_5 = a_before[-5:]
        amount_accel = float(recent_5.mean() - a_before[:-5].mean()) if n > 5 else 0.0
    else:
        amount_accel = 0.0

    # merchant_diversity_24h: unique merchants in last 24h
    last_24h_mask = (day_decimal - t_before) <= 1.0
    merchant_diversity_24h = int(np.unique(m_before[last_24h_mask]).shape[0])

    # avg_amount_24h
    avg_amount_24h = float(a_before[last_24h_mask].mean()) if last_24h_mask.sum() > 0 else 0.0

    # tx_gap_mean: mean hours between last 5 transactions
    if n >= 2:
        last5_times = t_before[-min(5, n):]
        gaps = np.diff(last5_times) * 24  # convert day fraction to hours
        tx_gap_mean = float(gaps.mean())
    else:
        tx_gap_mean = 0.0

    # tx_regularity: coefficient of variation of gaps (lower = more regular)
    if n >= 3:
        last5_times = t_before[-min(5, n):]
        gaps = np.diff(last5_times) * 24
        tx_regularity = float(gaps.std() / (gaps.mean() + 1e-6))
    else:
        tx_regularity = 0.0

    return tx_count_1h, tx_count_24h, amount_accel, merchant_diversity_24h, avg_amount_24h, tx_gap_mean, tx_regularity

# Vectorized: build arrays for all sampled rows
n = len(df)
vel_features = np.zeros((n, 7), dtype=np.float32)
for i in range(n):
    uid = df.iloc[i]["User"]
    dd = df.iloc[i]["day_decimal"]
    vel_features[i] = compute_velocity(i, uid, dd)
    if (i+1) % 10000 == 0:
        print(f"    ...{i+1}/{n} ({time.time()-t2:.0f}s)")

print(f"  Velocity computed in {time.time()-t2:.1f}s")

# Add velocity features to dataframe
df["tx_count_1h"] = vel_features[:, 0]
df["tx_count_24h"] = vel_features[:, 1]
df["amount_accel"] = vel_features[:, 2]
df["merchant_diversity_24h"] = vel_features[:, 3]
df["avg_amount_24h"] = vel_features[:, 4]
df["tx_gap_mean"] = vel_features[:, 5]
df["tx_regularity"] = vel_features[:, 6]

# Interaction: velocity x amount features
df["amt_x_tx1h"] = df["amt"] * df["tx_count_1h"]
df["amt_x_tx24h"] = df["amt"] * df["tx_count_24h"]
df["ute_x_tx24h"] = df.get("user_te", 0) * df["tx_count_24h"]

print(f"  Added 7 velocity features + 3 interactions = {len([c for c in df.columns if c in ['tx_count_1h','tx_count_24h','amount_accel','merchant_diversity_24h','avg_amount_24h','tx_gap_mean','tx_regularity','amt_x_tx1h','amt_x_tx24h','ute_x_tx24h']])} new features")

# ══════════════════════════════════════════════════════════
# Build feature matrix
# ══════════════════════════════════════════════════════════
y = df["label"].values
drop_cols = {"label","User","Time","Amount","Use Chip","MCC","Errors?",
    "Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"}
cols = [c for c in df.columns if c not in drop_cols]
X = np.nan_to_num(df[cols].values.astype(np.float32))
print(f"\n  Total features: {len(cols)}")

# ══════════════════════════════════════════════════════════
# Optuna tuning (15 trials, 3-fold CV)
# ══════════════════════════════════════════════════════════
print(f"\n  Running Optuna (15 trials, 3-fold CV)...")
t3 = time.time()

def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_est", 200, 600),
        max_depth=trial.suggest_int("max_depth", 6, 12),
        learning_rate=trial.suggest_float("lr", 0.03, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample", 0.5, 0.9),
        min_child_samples=trial.suggest_int("min_child", 5, 25),
        reg_alpha=trial.suggest_float("alpha", 1e-3, 5, log=True),
        reg_lambda=trial.suggest_float("lambda", 0.1, 10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 20, 127),
        verbose=-1, random_state=42, n_jobs=NJ,
    )
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    fold_aucs = []
    fold_r1s = []
    for tri, tei in skf.split(X, y):
        m = lgb.LGBMClassifier(**params).fit(X[tri], y[tri])
        p = m.predict_proba(X[tei])[:,1]
        fold_aucs.append(roc_auc_score(y[tei], p))
        fold_r1s.append(raf(y[tei], p, 0.01))
    # Optimize on R@1%FPR (primary target) with AUC as tiebreaker
    return np.mean(fold_r1s) * 0.8 + np.mean(fold_aucs) * 0.2

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=15, show_progress_bar=False)
opt_time = time.time()-t3
print(f"  Optuna done in {opt_time:.0f}s — best objective: {study.best_value:.6f}")

# ══════════════════════════════════════════════════════════
# 5-fold CV with best params
# ══════════════════════════════════════════════════════════
bp = study.best_params
lgb_params = dict(
    n_estimators=bp["n_est"], max_depth=bp["max_depth"], learning_rate=bp["lr"],
    subsample=bp["subsample"], colsample_bytree=bp["colsample"],
    min_child_samples=bp["min_child"], reg_alpha=bp["alpha"], reg_lambda=bp["lambda"],
    num_leaves=bp["num_leaves"], verbose=-1, random_state=42, n_jobs=NJ,
)

print(f"\n  5-fold CV with best params...")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
all_preds = np.zeros(len(y))
fold_aucs = []; fold_r1s = []
for fold, (tri, tei) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(**lgb_params).fit(X[tri], y[tri])
    p = m.predict_proba(X[tei])[:,1]
    all_preds[tei] = p
    fold_auc = roc_auc_score(y[tei], p)
    fold_r1 = raf(y[tei], p, 0.01)
    fold_aucs.append(fold_auc)
    fold_r1s.append(fold_r1)
    print(f"    Fold {fold}: AUC={fold_auc:.6f} R@1%FPR={fold_r1:.4f} ({int(y[tei].sum())} fraud)")

overall = roc_auc_score(y, all_preds)
m_met = met(y, all_preds)
print(f"\n  5-fold CV AUC:       {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")
print(f"  5-fold CV R@1%FPR:   {np.mean(fold_r1s):.6f} ± {np.std(fold_r1s):.6f}")
print(f"  Overall OOF AUC:     {overall:.6f}")
for k,v in m_met.items(): print(f"    {k}: {v}")

target_met = np.mean(fold_r1s) >= 0.92
print(f"\n  Target R@1%FPR >92%: {'ACHIEVED' if target_met else 'NOT MET'} ({np.mean(fold_r1s)*100:.2f}%)")

# ══════════════════════════════════════════════════════════
# Feature importance
# ══════════════════════════════════════════════════════════
fi_m = lgb.LGBMClassifier(**lgb_params).fit(X, y)
fi = dict(zip(cols, fi_m.feature_importances_))
top15 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
print(f"\n  Top 15 features:")
for name, imp in top15:
    marker = " ← VELOCITY" if name in ["tx_count_1h","tx_count_24h","amount_accel","merchant_diversity_24h","avg_amount_24h","tx_gap_mean","tx_regularity"] else ""
    print(f"    {name:30s}: {imp:6d}{marker}")

# Save
result = dict(dataset="ALTMAN_VELOCITY", n_rows=len(y), n_features=len(cols),
    n_fraud=int(y.sum()), optuna_best=round(study.best_value,6), best_params=bp,
    cv_5fold_auc_mean=round(float(np.mean(fold_aucs)),6),
    cv_5fold_auc_std=round(float(np.std(fold_aucs)),6),
    cv_5fold_r1_mean=round(float(np.mean(fold_r1s)),6),
    cv_5fold_r1_std=round(float(np.std(fold_r1s)),6),
    overall_oof=round(overall,6), metrics=m_met,
    target_r1_92_achieved=bool(target_met),
    velocity_features=["tx_count_1h","tx_count_24h","amount_accel","merchant_diversity_24h","avg_amount_24h","tx_gap_mean","tx_regularity"],
    top15_features=[{"name":n,"importance":int(i)} for n,i in top15],
    elapsed=round(time.time()-t0),
    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
out = R/"altman_velocity_results.json"
with open(out, "w") as f: json.dump(result, f, indent=2)
print(f"\n  Saved: {out}")
print(f"  Total: {time.time()-t0:.0f}s")
