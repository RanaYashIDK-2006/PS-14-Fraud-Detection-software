#!/usr/bin/env python3
"""Optuna tuning for ULB dataset — 20 trials, interaction features, target ROC-AUC > 99%.

Current baseline: ROC-AUC 0.9828 (weighted ensemble), R@1%FPR 90.8%
Target: > 0.9900

ULB has PCA features (V1-V28) — limited feature engineering possible.
Focus on: interaction features + Optuna hyperparameter search.
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
import lightgbm as lgb
import xgboost as xgb
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

# ═══ Load ULB with interaction features ═══
print("  Loading ULB...")
t0 = time.time()
df = pd.read_csv("data/creditcard.csv")
y = df["Class"].values
print(f"  {len(df):,} rows ({int(y.sum())} fraud, {y.mean()*100:.4f}%)")

# Existing features
df["amount_log"] = np.log1p(df["Amount"])
df["hour"] = (df["Time"] % 86400) / 3600
df["is_night"] = ((df["hour"]>=22)|(df["hour"]<=6)).astype(int)
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

# Interaction features (proven to help)
for a,b in [("V1","V2"),("V3","V4"),("V10","V14"),("V12","V14"),("V16","V18"),
            ("V1","V14"),("V4","V10"),("V17","V14"),("V7","V14"),("V9","V14"),
            ("V11","V14"),("V14","V17"),("V1","V4"),("V10","V12")]:
    df[f"{a}x{b}"] = df[a]*df[b]

# Squared features
for c in ["V1","V4","V10","V14","V12","V17","V3","V7","V9","V11","V16"]:
    df[f"{c}_sq"] = df[c]**2

# PCA aggregate features
vcols = [f"V{i}" for i in range(1,29)]
df["v_sum"] = df[vcols].sum(axis=1)
df["v_abs_mean"] = df[vcols].abs().mean(axis=1)
df["v_std"] = df[vcols].std(axis=1)
df["v_max"] = df[vcols].max(axis=1)
df["v_min"] = df[vcols].min(axis=1)
df["v_range"] = df["v_max"] - df["v_min"]

# Top fraud-correlated PCA components (V14, V17, V12, V10, V16)
df["fraud_cluster_1"] = df["V14"] * df["V17"]
df["fraud_cluster_2"] = df["V12"] * df["V14"]
df["fraud_cluster_3"] = df["V10"] * df["V16"]
df["fraud_cluster_sum"] = df["V14"] + df["V17"] + df["V12"] + df["V10"]

cols = [c for c in df.columns if c not in ("Class","Time")]
X = np.nan_to_num(df[cols].values.astype(np.float32))
print(f"  Features: {len(cols)} ({time.time()-t0:.1f}s)")

# ═══ Optuna: joint XGB + LGB tuning (20 trials, 3-fold CV) ═══
print(f"\n  Running Optuna (20 trials, 3-fold CV)...")
t2 = time.time()

def objective(trial):
    xgb_params = dict(
        n_estimators=trial.suggest_int("xgb_n_est", 150, 500),
        max_depth=trial.suggest_int("xgb_depth", 5, 10),
        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("xgb_sub", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("xgb_col", 0.5, 1.0),
        min_child_weight=trial.suggest_int("xgb_mcw", 1, 8),
        reg_alpha=trial.suggest_float("xgb_alpha", 1e-3, 5, log=True),
        reg_lambda=trial.suggest_float("xgb_lambda", 0.1, 10, log=True),
        tree_method="hist", eval_metric="auc", random_state=42, n_jobs=NJ,
    )
    lgb_params = dict(
        n_estimators=trial.suggest_int("lgb_n_est", 150, 500),
        max_depth=trial.suggest_int("lgb_depth", 5, 10),
        learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("lgb_sub", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("lgb_col", 0.5, 1.0),
        min_child_samples=trial.suggest_int("lgb_mc", 5, 30),
        reg_alpha=trial.suggest_float("lgb_alpha", 1e-3, 5, log=True),
        reg_lambda=trial.suggest_float("lgb_lambda", 0.1, 10, log=True),
        num_leaves=trial.suggest_int("lgb_leaves", 20, 80),
        verbose=-1, random_state=42, n_jobs=NJ,
    )
    xgb_w = trial.suggest_float("xgb_weight", 0.3, 0.8)

    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    fold_aucs = []
    for tri, tei in skf.split(X, y):
        spw = (len(y[tri])-int(y[tri].sum()))/max(int(y[tri].sum()),1)
        xm = xgb.XGBClassifier(**xgb_params, scale_pos_weight=min(spw,20))
        lm = lgb.LGBMClassifier(**lgb_params, scale_pos_weight=min(spw,20))
        xm.fit(X[tri], y[tri])
        lm.fit(X[tri], y[tri])
        xp = xm.predict_proba(X[tei])[:,1]
        lp = lm.predict_proba(X[tei])[:,1]
        blend = xgb_w * xp + (1-xgb_w) * lp
        fold_aucs.append(roc_auc_score(y[tei], blend))
    return np.mean(fold_aucs)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, show_progress_bar=False)
opt_time = time.time()-t2
print(f"  Optuna done in {opt_time:.0f}s — best CV AUC: {study.best_value:.6f}")
print(f"  Best params: {json.dumps(study.best_params, indent=2)}")

# ═══ 5-fold CV with best params ═══
bp = study.best_params
spw = (len(y)-int(y.sum()))/max(int(y.sum()),1)
xgb_params = dict(
    n_estimators=bp["xgb_n_est"], max_depth=bp["xgb_depth"], learning_rate=bp["xgb_lr"],
    subsample=bp["xgb_sub"], colsample_bytree=bp["xgb_col"], min_child_weight=bp["xgb_mcw"],
    reg_alpha=bp["xgb_alpha"], reg_lambda=bp["xgb_lambda"],
    scale_pos_weight=min(spw,20), tree_method="hist", eval_metric="auc",
    random_state=42, n_jobs=NJ,
)
lgb_params = dict(
    n_estimators=bp["lgb_n_est"], max_depth=bp["lgb_depth"], learning_rate=bp["lgb_lr"],
    subsample=bp["lgb_sub"], colsample_bytree=bp["lgb_col"], min_child_samples=bp["lgb_mc"],
    reg_alpha=bp["lgb_alpha"], reg_lambda=bp["lgb_lambda"], num_leaves=bp["lgb_leaves"],
    scale_pos_weight=min(spw,20), verbose=-1, random_state=42, n_jobs=NJ,
)
xgb_w = bp["xgb_weight"]

print(f"\n  5-fold CV with best params (xgb_weight={xgb_w:.3f})...")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
all_preds = np.zeros(len(y))
fold_aucs = []; fold_r1s = []
for fold, (tri, tei) in enumerate(skf.split(X, y)):
    tf = time.time()
    xm = xgb.XGBClassifier(**xgb_params).fit(X[tri], y[tri])
    lm = lgb.LGBMClassifier(**lgb_params).fit(X[tri], y[tri])
    xp = xm.predict_proba(X[tei])[:,1]
    lp = lm.predict_proba(X[tei])[:,1]
    blend = xgb_w * xp + (1-xgb_w) * lp
    all_preds[tei] = blend
    a = roc_auc_score(y[tei], blend)
    r = raf(y[tei], blend, 0.01)
    fold_aucs.append(a); fold_r1s.append(r)
    print(f"    Fold {fold}: AUC={a:.6f} R@1%FPR={r:.4f} ({time.time()-tf:.0f}s)")

overall = roc_auc_score(y, all_preds)
m_met = met(y, all_preds)
print(f"\n  5-fold CV AUC:       {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")
print(f"  5-fold CV R@1%FPR:   {np.mean(fold_r1s):.6f} ± {np.std(fold_r1s):.6f}")
print(f"  Overall OOF AUC:     {overall:.6f}")
for k,v in m_met.items(): print(f"    {k}: {v}")

target_met = np.mean(fold_aucs) >= 0.99
print(f"\n  Target ROC-AUC >99%: {'ACHIEVED' if target_met else 'NOT MET'} ({np.mean(fold_aucs)*100:.3f}%)")

# Feature importance (from LGB)
fi_m = lgb.LGBMClassifier(**lgb_params).fit(X, y)
fi = dict(zip(cols, fi_m.feature_importances_))
top15 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
print(f"\n  Top 15 features:")
for nm, imp in top15:
    print(f"    {nm:25s}: {imp:6d}")

# Save
result = dict(dataset="ULB_OPTUNA", n_rows=len(y), n_features=len(cols),
    n_fraud=int(y.sum()), optuna_best_cv=round(study.best_value,6), best_params=bp,
    xgb_weight=round(xgb_w,4),
    cv_5fold_mean=round(float(np.mean(fold_aucs)),6),
    cv_5fold_std=round(float(np.std(fold_aucs)),6),
    cv_r1_mean=round(float(np.mean(fold_r1s)),6),
    overall_oof=round(overall,6), metrics=m_met,
    target_99_achieved=bool(target_met),
    top15=[{"n":n,"i":int(v)} for n,v in top15],
    elapsed=round(time.time()-t0),
    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
with open(R/"ulb_optuna_results.json","w") as f: json.dump(result, f, indent=2)
print(f"\n  Saved: reports/ulb_optuna_results.json")
print(f"  Total: {time.time()-t0:.0f}s")
