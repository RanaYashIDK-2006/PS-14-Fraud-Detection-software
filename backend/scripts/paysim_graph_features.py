#!/usr/bin/env python3
"""PaySim flow features: subsampled Optuna (15 trials) + full-data 5-fold CV."""
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

def build_features(df):
    """Build flow-based proxy graph features."""
    # Type one-hot
    for c in pd.get_dummies(df["type"], prefix="t", dtype=float).columns:
        df[c] = pd.get_dummies(df["type"], prefix="t", dtype=float)[c]
    # Balance flow
    df["bal_diff_o"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["bal_diff_d"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amt_ratio_o"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amt_ratio_d"] = df["amount"] / (df["oldbalanceDest"] + 1)
    df["log_amt"] = np.log1p(df["amount"])
    # Consistency
    df["orig_consistency"] = np.abs(df["bal_diff_o"] - df["amount"]) / (df["amount"] + 1)
    df["dest_consistency"] = np.abs(df["bal_diff_d"] - df["amount"]) / (df["amount"] + 1)
    # Wiping
    df["orig_wiped"] = (df["newbalanceOrig"] < 1).astype(float)
    df["dest_empty"] = (df["oldbalanceDest"] < 1).astype(float)
    df["orig_drain"] = df["bal_diff_o"] / (df["oldbalanceOrg"] + 1)
    # Asymmetry
    df["flow_asym"] = np.abs(df["bal_diff_o"] - df["bal_diff_d"]) / (df["amount"] + 1)
    df["amt_vs_orig_change"] = df["amount"] / (np.abs(df["bal_diff_o"]) + 1)
    df["amt_vs_dest_change"] = df["amount"] / (np.abs(df["bal_diff_d"]) + 1)
    # Ratios
    df["bal_ratio"] = (df["oldbalanceOrg"] + 1) / (df["oldbalanceDest"] + 1)
    df["bal_ratio_new"] = (df["newbalanceOrig"] + 1) / (df["newbalanceDest"] + 1)
    # Interactions
    df["amt_sq"] = df["amount"] ** 2
    df["amt_x_flag"] = df["amount"] * df["isFlaggedFraud"]
    df["amt_x_o"] = df["amount"] * df["oldbalanceOrg"]
    df["amt_x_d"] = df["amount"] * df["oldbalanceDest"]
    # Fraud scores
    df["transfer_fraud_score"] = df.get("t_TRANSFER", 0) * df["orig_wiped"] * (df["amt_ratio_o"] > 0.5).astype(float)
    df["cashout_fraud_score"] = df.get("t_CASH_OUT", 0) * df["orig_wiped"] * (df["orig_drain"] > 0.8).astype(float)
    df["large_amt_relative"] = (df["amt_ratio_o"] > 0.9).astype(float)
    df["neg_orig"] = (df["newbalanceOrig"] < 0).astype(float)
    # Cross-account
    df["flow_efficiency"] = df["bal_diff_d"] / (df["bal_diff_o"] + 1)
    df["flow_leakage"] = 1 - df["flow_efficiency"]
    # Row index
    df["row_idx"] = np.arange(len(df)) / len(df)
    # Type stats
    df["type_amt_mean"] = df.groupby("type")["amount"].transform("mean")
    df["amt_vs_type_mean"] = df["amount"] / (df["type_amt_mean"] + 1)
    df["amt_zscore_type"] = (df["amount"] - df["type_amt_mean"]) / (df.groupby("type")["amount"].transform("std") + 1)
    return df

# ═══ Load + build features ═══
print("  Loading PaySim...")
t0 = time.time()
df = pd.read_csv("data/paysim_1m.csv")
y = df["isFraud"].values
print(f"  {len(df):,} rows ({int(y.sum()):,} fraud, {y.mean()*100:.2f}%)")
print("  Building features...")
df = build_features(df)
drop_cols = {"type", "isFraud", "isFlaggedFraud"}
cols = [c for c in df.columns if c not in drop_cols]
X = np.nan_to_num(df[cols].values.astype(np.float32))
print(f"  {len(cols)} features ({time.time()-t0:.1f}s)")

# ═══ Optuna on 200K subsample ═══
print(f"\n  Optuna (15 trials, 3-fold, subsampled 200K)...")
t2 = time.time()
rng = np.random.RandomState(42)
fraud_idx = np.where(y == 1)[0]
legit_idx = np.where(y == 0)[0]
sub_legit = rng.choice(legit_idx, min(180_000, len(legit_idx)), replace=False)
sub_idx = np.concatenate([fraud_idx, sub_legit])
X_sub, y_sub = X[sub_idx], y[sub_idx]
print(f"  Subsample: {len(X_sub):,} ({int(y_sub.sum())} fraud)")

def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_est", 100, 400),
        max_depth=trial.suggest_int("max_depth", 5, 10),
        learning_rate=trial.suggest_float("lr", 0.03, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample", 0.4, 0.9),
        min_child_samples=trial.suggest_int("min_child", 10, 40),
        reg_alpha=trial.suggest_float("alpha", 1e-3, 5, log=True),
        reg_lambda=trial.suggest_float("lambda", 0.1, 10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 20, 80),
        verbose=-1, random_state=42, n_jobs=NJ,
    )
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    r1s = []
    for tri, tei in skf.split(X_sub, y_sub):
        m = lgb.LGBMClassifier(**params).fit(X_sub[tri], y_sub[tri])
        p = m.predict_proba(X_sub[tei])[:, 1]
        r1s.append(raf(y_sub[tei], p, 0.01))
    return np.mean(r1s)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=15, show_progress_bar=False)
print(f"  Best R@1%FPR: {study.best_value:.6f} ({time.time()-t2:.0f}s)")

# ═══ 5-fold CV on FULL data ═══
bp = study.best_params
lp = dict(n_estimators=bp["n_est"], max_depth=bp["max_depth"], learning_rate=bp["lr"],
    subsample=bp["subsample"], colsample_bytree=bp["colsample"],
    min_child_samples=bp["min_child"], reg_alpha=bp["alpha"], reg_lambda=bp["lambda"],
    num_leaves=bp["num_leaves"], verbose=-1, random_state=42, n_jobs=NJ)

print(f"\n  5-fold CV (full {len(X):,} rows)...")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
all_p = np.zeros(len(y))
f_aucs = []; f_r1s = []
for fold, (tri, tei) in enumerate(skf.split(X, y)):
    t3 = time.time()
    m = lgb.LGBMClassifier(**lp).fit(X[tri], y[tri])
    p = m.predict_proba(X[tei])[:, 1]
    all_p[tei] = p
    a = roc_auc_score(y[tei], p); r = raf(y[tei], p, 0.01)
    f_aucs.append(a); f_r1s.append(r)
    print(f"    Fold {fold}: AUC={a:.6f} R@1%FPR={r:.4f} ({int(y[tei].sum())} fraud) ({time.time()-t3:.0f}s)")

oa = roc_auc_score(y, all_p)
mm = met(y, all_p)
print(f"\n  CV AUC:     {np.mean(f_aucs):.6f} ± {np.std(f_aucs):.6f}")
print(f"  CV R@1%FPR: {np.mean(f_r1s):.6f} ± {np.std(f_r1s):.6f}")
print(f"  OOF AUC:    {oa:.6f}")
for k, v in mm.items(): print(f"    {k}: {v}")

target = np.mean(f_r1s) >= 0.60
print(f"\n  Target R@1%FPR>60%: {'ACHIEVED' if target else 'NOT MET'} ({np.mean(f_r1s)*100:.2f}%)")

# Feature importance
fi_m = lgb.LGBMClassifier(**lp).fit(X, y)
fi = dict(zip(cols, fi_m.feature_importances_))
top15 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
flow_set = {"bal_diff_o","bal_diff_d","amt_ratio_o","amt_ratio_d","orig_consistency",
    "dest_consistency","orig_wiped","dest_empty","orig_drain","flow_asym",
    "amt_vs_orig_change","amt_vs_dest_change","bal_ratio","bal_ratio_new",
    "flow_efficiency","flow_leakage","transfer_fraud_score","cashout_fraud_score",
    "large_amt_relative","neg_orig","amt_vs_type_mean","amt_zscore_type"}
print(f"\n  Top 15:")
for nm, imp in top15:
    print(f"    {nm:30s}: {imp:6d}{' ← FLOW' if nm in flow_set else ''}")

result = dict(dataset="PAYSIM_GRAPH", n_rows=len(y), n_features=len(cols),
    n_fraud=int(y.sum()), best_r1=round(study.best_value,6), best_params=bp,
    cv_auc=round(float(np.mean(f_aucs)),6), cv_auc_std=round(float(np.std(f_aucs)),6),
    cv_r1=round(float(np.mean(f_r1s)),6), cv_r1_std=round(float(np.std(f_r1s)),6),
    oof_auc=round(oa,6), metrics=mm, target_r1_60=bool(target),
    top15=[{"n":n,"i":int(v)} for n,v in top15],
    elapsed=round(time.time()-t0),
    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
with open(R/"paysim_graph_results.json","w") as f: json.dump(result, f, indent=2)
print(f"\n  Saved: reports/paysim_graph_results.json")
print(f"  Total: {time.time()-t0:.0f}s")
