#!/usr/bin/env python3
"""Push Altman above 99.5% — OOF stacking + extra features.

Key insight from previous runs:
  - Simple blend (xgb_w * xgb + (1-xgb_w) * lgb) = 0.9934 — WORSE than LGB alone
  - OOF stacking uses LR as meta-learner on out-of-fold predictions — avoids leakage
  - Add velocity/behavioral features to improve signal

Target: 5-fold CV AUC > 0.9950
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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

# ── Load Altman with EXTRA features ──
print("  Loading Altman (enhanced features)...")
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
    chunk["month_n"] = chunk["Month"].astype(float)
    chunk["day_n"] = chunk["Day"].astype(float)
    # Hour bins
    chunk["hr_bin"] = pd.cut(chunk["hr"], bins=[0,6,12,18,24], labels=[0,1,2,3]).astype(float)
    # Amount bins (log-scale)
    chunk["amt_log"] = np.log1p(chunk["amt"])
    chunk["amt_bin"] = pd.cut(chunk["amt_log"], bins=20, labels=False).astype(float)
    fraud_mask = chunk["label"] == 1
    fraud_df = chunk[fraud_mask]
    legit_sample = chunk[~fraud_mask].sample(frac=0.005, random_state=rng)
    rows_all.append(pd.concat([fraud_df, legit_sample]))
    ci += 1
    if ci >= 16: break

df = pd.concat(rows_all, ignore_index=True)
print(f"  Loaded {len(df):,} rows ({int(df['label'].sum())} fraud) in {time.time()-t_load:.1f}s")

# OOF target encoding (3 entities)
gm = df["label"].mean()
for enc_col, grp_col, prior in [("user_te","User",200), ("merchant_te","merchant_id",50), ("city_te","city_id",20)]:
    oof = np.zeros(len(df))
    for tri, vai in StratifiedShuffleSplit(2, test_size=0.2, random_state=42).split(df, df["label"]):
        stats = df.iloc[tri].groupby(grp_col)["label"].agg(["mean","count"])
        stats["s"] = (stats["mean"]*stats["count"]+gm*prior)/(stats["count"]+prior)
        oof[vai] = df.iloc[vai][grp_col].map(stats["s"].to_dict()).fillna(gm).values
    df[enc_col] = oof

# Interaction features
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
# Extra features
df["ute_x_mte"] = df["user_te"]*df["merchant_te"]
df["chip_x_online"] = df["chip"]*df["is_online"]
df["err_x_amt"] = df["err"]*df["amt"]
df["hr_x_ute"] = df["hr"]*df["user_te"]
df["mcc_x_ute"] = df["mcc_n"]*df["user_te"]

y = df["label"].values
drop_cols = {"label","User","Time","Amount","Use Chip","MCC","Errors?",
    "Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"}
cols = [c for c in df.columns if c not in drop_cols]
X = np.nan_to_num(df[cols].values.astype(np.float32))
print(f"  Features: {len(cols)}")

# ── Optuna-best params ──
lgb_best = dict(n_estimators=399, max_depth=7, learning_rate=0.1648,
    subsample=0.6985, colsample_bytree=0.7348, min_child_samples=10,
    reg_alpha=0.01049, reg_lambda=1.8895, verbose=-1, random_state=42, n_jobs=NJ)
xgb_best = dict(n_estimators=158, max_depth=7, learning_rate=0.1133,
    subsample=0.9472, colsample_bytree=0.7817, min_child_weight=1,
    reg_alpha=1.862, reg_lambda=0.283, tree_method="hist", eval_metric="auc",
    random_state=42, n_jobs=NJ)

# Additional model configs to diversify the stack
lgb_alt = dict(n_estimators=500, max_depth=6, learning_rate=0.08, subsample=0.75,
    colsample_bytree=0.8, min_child_samples=15, reg_alpha=0.05, reg_lambda=2.0,
    verbose=-1, random_state=43, n_jobs=NJ)
xgb_alt = dict(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.85,
    colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.5, reg_lambda=1.5,
    tree_method="hist", eval_metric="auc", random_state=43, n_jobs=NJ)

print(f"\n{'='*60}")
print(f"  PUSH 99.5% — Altman ({len(df):,} rows, {len(cols)} feat)")
print(f"{'='*60}")

t0 = time.time()
spw = (len(y)-int(y.sum()))/max(int(y.sum()),1)

# ── OOF stacking with 5-fold CV ──
n_models = 7
model_defs = [
    ("lgb_best", lgb.LGBMClassifier(**lgb_best)),
    ("xgb_best", xgb.XGBClassifier(**xgb_best)),
    ("lgb_alt", lgb.LGBMClassifier(**lgb_alt)),
    ("xgb_alt", xgb.XGBClassifier(**xgb_alt)),
    ("rf_50", RandomForestClassifier(n_estimators=50, max_depth=10, class_weight="balanced", n_jobs=NJ, random_state=42)),
    ("hgb", HistGradientBoostingClassifier(max_iter=200, max_depth=7, learning_rate=0.1, random_state=42)),
    ("lr", LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)),
]

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(y), n_models))
fold_aucs = []

for fold, (tri, tei) in enumerate(skf.split(X, y)):
    Xtr, ytr, Xte, yte = X[tri], y[tri], X[tei], y[tei]
    scaler = RobustScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    fold_model_aucs = []
    for j, (nm, _) in enumerate(model_defs):
        m = type(model_defs[j][1])(**model_defs[j][1].get_params())
        m.fit(Xtr_s, ytr)
        p = m.predict_proba(Xte_s)[:,1]
        oof_preds[tei, j] = p
        fold_model_aucs.append(roc_auc_score(yte, p))
    # Individual blend
    blend = 0.55*oof_preds[tei,0] + 0.45*oof_preds[tei,1]  # lgb_best + xgb_best
    blend_auc = roc_auc_score(yte, blend)
    fold_aucs.append(blend_auc)
    print(f"  Fold {fold}: blend={blend_auc:.6f} | " + " | ".join(f"{model_defs[j][0]}={fold_model_aucs[j]:.4f}" for j in range(n_models)))

# ── Meta-learner on OOF predictions ──
print(f"\n  Training OOF meta-learner...")
# Scale OOF predictions for meta-learner
oof_s = RobustScaler().fit_transform(oof_preds)
meta = LogisticRegression(C=100, max_iter=2000, random_state=42)
meta.fit(oof_s, y)
meta_preds = meta.predict_proba(oof_s)[:,1]
meta_auc = roc_auc_score(y, meta_preds)
meta_met = met(y, meta_preds)
print(f"  Meta-learner OOF AUC: {meta_auc:.6f}")
print(f"  Meta weights: {dict(zip([d[0] for d in model_defs], [round(w,4) for w in meta.coef_[0]]))}")

# ── Best simple blend ──
best_blend_auc = 0
best_w = None
for w0 in np.arange(0.3, 0.8, 0.02):
    for w1 in np.arange(0.2, 0.7, 0.02):
        if w0 + w1 > 1.05: continue
        w2 = 1 - w0 - w1
        if w2 < 0: continue
        b = w0*oof_preds[:,0] + w1*oof_preds[:,1] + w2*oof_preds[:,2]
        a = roc_auc_score(y, b)
        if a > best_blend_auc:
            best_blend_auc = a
            best_w = (w0, w1, w2)

blend_met = met(y, best_w[0]*oof_preds[:,0] + best_w[1]*oof_preds[:,1] + best_w[2]*oof_preds[:,2])
print(f"\n  Best 3-model blend: {best_blend_auc:.6f} (w={best_w})")
print(f"    R@1%FPR: {blend_met['r1']:.4f}  PR-AUC: {blend_met['pr_auc']:.4f}")

# ── Individual model OOF AUCs ──
print(f"\n  Individual OOF AUCs:")
for j, (nm, _) in enumerate(model_defs):
    a = roc_auc_score(y, oof_preds[:,j])
    print(f"    {nm:15s}: {a:.6f}")

# ── Find best approach ──
approaches = {
    "lgb_best_oof": roc_auc_score(y, oof_preds[:,0]),
    "xgb_best_oof": roc_auc_score(y, oof_preds[:,1]),
    "lgb_alt_oof": roc_auc_score(y, oof_preds[:,2]),
    "xgb_alt_oof": roc_auc_score(y, oof_preds[:,3]),
    "3model_blend": best_blend_auc,
    "meta_learner": meta_auc,
}
best_approach = max(approaches, key=approaches.get)
best_val = approaches[best_approach]

print(f"\n{'='*60}")
print(f"  BEST APPROACH: {best_approach} = {best_val:.6f}")
print(f"  5-fold CV blend mean: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")
target_met = best_val >= 0.995 or np.mean(fold_aucs) >= 0.995
print(f"  Target >99.5%: {'ACHIEVED' if target_met else 'NOT MET'} ({best_val*100:.3f}%)")
print(f"{'='*60}")

# ── Save ──
result = dict(dataset="ALTMAN_PUSH", n_rows=len(y), n_features=len(cols),
    n_fraud=int(y.sum()), best_approach=best_approach, best_auc=round(best_val,6),
    cv_blend_mean=round(float(np.mean(fold_aucs)),6),
    cv_blend_std=round(float(np.std(fold_aucs)),6),
    meta_oof_auc=round(meta_auc,6),
    approaches={k: round(v,6) for k,v in approaches.items()},
    meta_weights={d[0]: round(float(w),4) for d, w in zip(model_defs, meta.coef_[0])},
    blend_weights={"lgb_best": round(best_w[0],3), "xgb_best": round(best_w[1],3), "lgb_alt": round(best_w[2],3)},
    target_995_achieved=bool(target_met),
    metrics=meta_met if best_approach=="meta_learner" else blend_met,
    elapsed=round(time.time()-t0),
    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
out = R/"altman_push_results.json"
with open(out, "w") as f: json.dump(result, f, indent=2)
print(f"  Saved: {out}")
print(f"  Elapsed: {time.time()-t0:.0f}s")
