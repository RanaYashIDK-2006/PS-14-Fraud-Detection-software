#!/usr/bin/env python3
"""Fast ULB Optuna: LGB-only, 20 trials, interaction features."""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
                f1=round(float(f1s[bi]),6), opt_thr=round(float(thrs[min(bi,len(thrs)-1)]),6))

# Load ULB
print("  Loading ULB...")
t0 = time.time()
df = pd.read_csv("data/creditcard.csv")
y = df["Class"].values
print(f"  {len(df):,} rows ({int(y.sum())} fraud)")

# Features
df["amount_log"] = np.log1p(df["Amount"])
df["hour"] = (df["Time"] % 86400) / 3600
df["is_night"] = ((df["hour"]>=22)|(df["hour"]<=6)).astype(int)
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
for a,b in [("V1","V2"),("V3","V4"),("V10","V14"),("V12","V14"),("V16","V18"),
            ("V1","V14"),("V4","V10"),("V17","V14"),("V7","V14"),("V9","V14"),
            ("V11","V14"),("V14","V17"),("V1","V4"),("V10","V12")]:
    df[f"{a}x{b}"] = df[a]*df[b]
for c in ["V1","V4","V10","V14","V12","V17","V3","V7","V9","V11","V16"]:
    df[f"{c}_sq"] = df[c]**2
vcols = [f"V{i}" for i in range(1,29)]
df["v_sum"] = df[vcols].sum(axis=1)
df["v_abs_mean"] = df[vcols].abs().mean(axis=1)
df["v_std"] = df[vcols].std(axis=1)
df["v_max"] = df[vcols].max(axis=1)
df["v_min"] = df[vcols].min(axis=1)
df["v_range"] = df["v_max"] - df["v_min"]
df["fraud_cluster_1"] = df["V14"]*df["V17"]
df["fraud_cluster_2"] = df["V12"]*df["V14"]
df["fraud_cluster_3"] = df["V10"]*df["V16"]
df["fraud_cluster_sum"] = df["V14"]+df["V17"]+df["V12"]+df["V10"]

cols = [c for c in df.columns if c not in ("Class","Time")]
X = np.nan_to_num(df[cols].values.astype(np.float32))
print(f"  Features: {len(cols)} ({time.time()-t0:.1f}s)")

# Optuna (20 trials, 3-fold)
print(f"\n  Optuna (20 trials, 3-fold CV)...")
t2 = time.time()

def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_est", 200, 600),
        max_depth=trial.suggest_int("max_depth", 5, 10),
        learning_rate=trial.suggest_float("lr", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.65, 1.0),
        colsample_bytree=trial.suggest_float("colsample", 0.5, 1.0),
        min_child_samples=trial.suggest_int("min_child", 3, 25),
        reg_alpha=trial.suggest_float("alpha", 1e-3, 5, log=True),
        reg_lambda=trial.suggest_float("lambda", 0.1, 10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 20, 100),
        verbose=-1, random_state=42, n_jobs=NJ,
    )
    spw = (len(y)-int(y.sum()))/max(int(y.sum()),1)
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    fold_aucs = []
    for tri, tei in skf.split(X, y):
        m = lgb.LGBMClassifier(**params, scale_pos_weight=min(spw,20))
        m.fit(X[tri], y[tri])
        p = m.predict_proba(X[tei])[:,1]
        fold_aucs.append(roc_auc_score(y[tei], p))
    return np.mean(fold_aucs)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, show_progress_bar=False)
print(f"  Best CV AUC: {study.best_value:.6f} ({time.time()-t2:.0f}s)")

# 5-fold CV
bp = study.best_params
spw = (len(y)-int(y.sum()))/max(int(y.sum()),1)
lp = dict(n_estimators=bp["n_est"], max_depth=bp["max_depth"], learning_rate=bp["lr"],
    subsample=bp["subsample"], colsample_bytree=bp["colsample"],
    min_child_samples=bp["min_child"], reg_alpha=bp["alpha"], reg_lambda=bp["lambda"],
    num_leaves=bp["num_leaves"], scale_pos_weight=min(spw,20), verbose=-1, random_state=42, n_jobs=NJ)

print(f"\n  5-fold CV...")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
all_p = np.zeros(len(y)); f_aucs=[]; f_r1s=[]
for fold, (tri, tei) in enumerate(skf.split(X, y)):
    tf = time.time()
    m = lgb.LGBMClassifier(**lp).fit(X[tri], y[tri])
    p = m.predict_proba(X[tei])[:,1]
    all_p[tei] = p
    a = roc_auc_score(y[tei], p); r = raf(y[tei], p, 0.01)
    f_aucs.append(a); f_r1s.append(r)
    print(f"    Fold {fold}: AUC={a:.6f} R@1%FPR={r:.4f} ({time.time()-tf:.0f}s)")

oa = roc_auc_score(y, all_p)
mm = met(y, all_p)
print(f"\n  CV AUC:     {np.mean(f_aucs):.6f} ± {np.std(f_aucs):.6f}")
print(f"  CV R@1%FPR: {np.mean(f_r1s):.6f} ± {np.std(f_r1s):.6f}")
print(f"  OOF AUC:    {oa:.6f}")
for k,v in mm.items(): print(f"    {k}: {v}")

target = np.mean(f_aucs) >= 0.99
print(f"\n  Target >99%: {'ACHIEVED' if target else 'NOT MET'} ({np.mean(f_aucs)*100:.3f}%)")

# Feature importance
fi_m = lgb.LGBMClassifier(**lp).fit(X, y)
fi = dict(zip(cols, fi_m.feature_importances_))
top15 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
print(f"\n  Top 15:")
for nm, imp in top15:
    print(f"    {nm:25s}: {imp:6d}")

result = dict(dataset="ULB_OPTUNA_FAST", n_rows=len(y), n_features=len(cols),
    n_fraud=int(y.sum()), best_cv=round(study.best_value,6), best_params=bp,
    cv_mean=round(float(np.mean(f_aucs)),6), cv_std=round(float(np.std(f_aucs)),6),
    cv_r1=round(float(np.mean(f_r1s)),6), oof_auc=round(oa,6), metrics=mm,
    target_99=bool(target),
    top15=[{"n":n,"i":int(v)} for n,v in top15],
    elapsed=round(time.time()-t0),
    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
with open(R/"ulb_optuna_fast.json","w") as f: json.dump(result, f, indent=2)
print(f"\n  Saved: reports/ulb_optuna_fast.json")
print(f"  Total: {time.time()-t0:.0f}s")
