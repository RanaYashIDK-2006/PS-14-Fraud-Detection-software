#!/usr/bin/env python3
"""Evaluate the Optuna-best individual models on Altman — no ensemble blending.

Key finding: Optuna blend AUC=0.9934 but individual XGB=0.9951, LGB=0.9957.
The blend WORSENS performance. Evaluate each individually with 5-fold CV.
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
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

# ── Load Altman with improved features ──
print("  Loading Altman...")
t_load = time.time()
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
    fraud_mask = chunk["label"] == 1
    fraud_df = chunk[fraud_mask]
    legit_sample = chunk[~fraud_mask].sample(frac=0.005, random_state=rng)
    rows_all.append(pd.concat([fraud_df, legit_sample]))
    ci += 1
    if ci >= 8: break

df = pd.concat(rows_all, ignore_index=True)
print(f"  Loaded {len(df):,} rows in {time.time()-t_load:.1f}s ({int(df['label'].sum())} fraud)")

# OOF target encoding
gm = df["label"].mean()
for enc_col, grp_col, prior in [("user_te","User",200), ("merchant_te","merchant_id",50), ("city_te","city_id",20)]:
    oof = np.zeros(len(df))
    for tri, vai in StratifiedShuffleSplit(2, test_size=0.2, random_state=42).split(df, df["label"]):
        stats = df.iloc[tri].groupby(grp_col)["label"].agg(["mean","count"])
        stats["s"] = (stats["mean"]*stats["count"]+gm*prior)/(stats["count"]+prior)
        oof[vai] = df.iloc[vai][grp_col].map(stats["s"].to_dict()).fillna(gm).values
    df[enc_col] = oof

df["log_amt"] = np.log1p(df["amt"])
df["amt_x_hr"] = df["amt"]*df["hr"]
df["amt_x_mcc"] = df["amt"]*df["mcc_n"]
df["amt_x_chip"] = df["amt"]*df["chip"]
df["amt_x_ute"] = df["amt"]*df["user_te"]
df["amt_x_mte"] = df["amt"]*df["merchant_te"]
df["amt_x_online"] = df["amt"]*df["is_online"]
df["hr_x_chip"] = df["hr"]*df["chip"]
df["amt_sq"] = df["amt"]**2
df["high_amt"] = (df["amt"] > df["amt"].quantile(0.95)).astype(float)
df["night_tx"] = ((df["hr"]>=22)|(df["hr"]<=6)).astype(int)
df["weekend"] = ((df["Day"]%7)>=5).astype(float)

y = df["label"].values
drop_cols = {"label","User","Time","Amount","Use Chip","MCC","Errors?",
    "Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"}
cols = [c for c in df.columns if c not in drop_cols]
X = np.nan_to_num(df[cols].values.astype(np.float32))
print(f"  Features: {len(cols)}")

# Optuna-best params
lgb_params = dict(
    n_estimators=399, max_depth=7, learning_rate=0.1648,
    subsample=0.6985, colsample_bytree=0.7348, min_child_samples=10,
    reg_alpha=0.01049, reg_lambda=1.8895,
    verbose=-1, random_state=42, n_jobs=NJ,
)
xgb_params = dict(
    n_estimators=158, max_depth=7, learning_rate=0.1133,
    subsample=0.9472, colsample_bytree=0.7817, min_child_weight=1,
    reg_alpha=1.862, reg_lambda=0.283,
    tree_method="hist", eval_metric="auc", random_state=42, n_jobs=NJ,
)

# Also try defaults for comparison
xgb_default = dict(n_estimators=200, max_depth=7, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.7, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", eval_metric="auc", random_state=42, n_jobs=NJ)
lgb_default = dict(n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.7, min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
    verbose=-1, random_state=42, n_jobs=NJ)

models = [
    ("LGB_Optuna", lgb.LGBMClassifier(**lgb_params)),
    ("XGB_Optuna", xgb.XGBClassifier(**xgb_params)),
    ("LGB_Default", lgb.LGBMClassifier(**lgb_default)),
    ("XGB_Default", xgb.XGBClassifier(**xgb_default)),
    ("RF_30", RandomForestClassifier(n_estimators=30, max_depth=10, class_weight="balanced", n_jobs=NJ, random_state=42)),
    ("LR", LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)),
]

print(f"\n{'='*60}")
print(f"  5-FOLD CV — Altman ({len(df):,} rows, {len(cols)} feat)")
print(f"{'='*60}")

results = {}
for name, model in models:
    t0 = time.time()
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    fold_aucs = []
    all_preds = np.zeros(len(y))
    for tri, tei in skf.split(X, y):
        m = type(model)(**model.get_params())
        m.fit(X[tri], y[tri])
        p = m.predict_proba(X[tei])[:,1]
        all_preds[tei] = p
        fold_aucs.append(roc_auc_score(y[tei], p))
    overall = roc_auc_score(y, all_preds)
    m_met = met(y, all_preds)
    elapsed = time.time()-t0
    results[name] = dict(
        cv_mean=round(float(np.mean(fold_aucs)),6),
        cv_std=round(float(np.std(fold_aucs)),6),
        oof_auc=round(overall,6),
        r1=round(m_met["r1"],6),
        r05=round(m_met["r05"],6),
        pr_auc=round(m_met["pr_auc"],6),
        elapsed=round(elapsed),
    )
    print(f"\n  {name:20s} CV: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}  OOF: {overall:.6f}  R@1%FPR: {m_met['r1']:.4f}  PR-AUC: {m_met['pr_auc']:.4f}  ({elapsed:.0f}s)")

# ── Find best ──
best_name = max(results, key=lambda k: results[k]["cv_mean"])
best = results[best_name]
print(f"\n{'='*60}")
print(f"  BEST: {best_name}")
print(f"    5-fold CV AUC: {best['cv_mean']:.6f} ± {best['cv_std']:.6f}")
print(f"    OOF AUC:       {best['oof_auc']:.6f}")
print(f"    R@1%FPR:       {best['r1']:.6f}")
print(f"    PR-AUC:        {best['pr_auc']:.6f}")
print(f"{'='*60}")

target_met = best["cv_mean"] >= 0.995
print(f"\n  Target >99.5%: {'ACHIEVED' if target_met else 'NOT MET'} ({best['cv_mean']*100:.3f}%)")

# Save
result = dict(dataset="ALTMAN_OPTUNA", best_model=best_name, best_cv_mean=best["cv_mean"],
    best_cv_std=best["cv_std"], best_oof_auc=best["oof_auc"],
    best_r1=best["r1"], best_pr_auc=best["pr_auc"],
    target_995_achieved=target_met,
    all_results=results,
    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
out = R/"altman_optuna_final.json"
with open(out, "w") as f: json.dump(result, f, indent=2)
print(f"  Saved: {out}")
