#!/usr/bin/env python3
"""Ultra-focused push: deeper LGB + more leaves, 10 trials."""
import json, time, warnings
from pathlib import Path
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

# ── Load (reuse cache if same file, otherwise reload) ──
print("  Loading Altman...")
t_load = time.time()
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

# ── Optuna: focused search around depth 8-12, 400-800 trees ──
print(f"\n  Running Optuna (10 trials, 3-fold CV, deep LGB)...")
t0 = time.time()

def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_est", 300, 800),
        max_depth=trial.suggest_int("max_depth", 8, 12),
        learning_rate=trial.suggest_float("lr", 0.03, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample", 0.5, 0.9),
        min_child_samples=trial.suggest_int("min_child", 5, 20),
        reg_alpha=trial.suggest_float("alpha", 1e-3, 5, log=True),
        reg_lambda=trial.suggest_float("lambda", 0.1, 10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 30, 127),
        verbose=-1, random_state=42, n_jobs=NJ,
    )
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    fold_aucs = []
    for tri, tei in skf.split(X, y):
        m = lgb.LGBMClassifier(**params).fit(X[tri], y[tri])
        p = m.predict_proba(X[tei])[:,1]
        fold_aucs.append(roc_auc_score(y[tei], p))
    return np.mean(fold_aucs)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=10, show_progress_bar=False)
opt_time = time.time()-t0
print(f"  Optuna done in {opt_time:.0f}s — best CV: {study.best_value:.6f}")

# ── 5-fold CV ──
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
fold_aucs = []
for fold, (tri, tei) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(**lgb_params).fit(X[tri], y[tri])
    p = m.predict_proba(X[tei])[:,1]
    all_preds[tei] = p
    fold_aucs.append(roc_auc_score(y[tei], p))
    print(f"    Fold {fold}: AUC={roc_auc_score(y[tei], p):.6f} ({int(y[tei].sum())} fraud)")

overall = roc_auc_score(y, all_preds)
m_met = met(y, all_preds)
print(f"\n  5-fold CV mean: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")
print(f"  Overall OOF:    {overall:.6f}")
for k,v in m_met.items(): print(f"    {k}: {v}")

target_met = np.mean(fold_aucs) >= 0.995 or overall >= 0.995
print(f"\n  Target >99.5%: {'ACHIEVED' if target_met else 'NOT MET'} ({np.mean(fold_aucs)*100:.3f}%)")

result = dict(dataset="ALTMAN_FINAL_995", n_rows=len(y), n_features=len(cols),
    n_fraud=int(y.sum()), optuna_best_cv=round(study.best_value,6), best_params=bp,
    cv_5fold_mean=round(float(np.mean(fold_aucs)),6), cv_5fold_std=round(float(np.std(fold_aucs)),6),
    overall_oof=round(overall,6), metrics=m_met, target_995_achieved=bool(target_met),
    elapsed=round(time.time()-t0),
    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
out = R/"altman_final_995.json"
with open(out, "w") as f: json.dump(result, f, indent=2)
print(f"  Saved: {out}")
