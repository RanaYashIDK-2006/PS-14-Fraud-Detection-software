#!/usr/bin/env python3
"""Ultra-fast max performance: 5 trials, 3-fold, 3 models only."""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)
R = Path("reports"); R.mkdir(exist_ok=True)

def raf(y, p, t):
    fpr, tpr, _ = roc_curve(y, p); m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0

def met(y, p):
    roc = roc_auc_score(y, p); pr = average_precision_score(y, p)
    r1 = raf(y, p, 0.01); r05 = raf(y, p, 0.005)
    return dict(roc_auc=round(roc,6), pr_auc=round(pr,6), r1=round(r1,6), r05=round(r05,6))

def load_ulb():
    df = pd.read_csv("data/creditcard.csv")
    y = df["Class"].values
    df["amount_log"] = np.log1p(df["Amount"])
    df["hour"] = (df["Time"] % 86400) / 3600
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    for a,b in [("V1","V2"),("V3","V4"),("V10","V14"),("V12","V14"),("V16","V18"),("V1","V14"),("V4","V10"),("V17","V14")]:
        df[f"{a}x{b}"] = df[a]*df[b]
    for c in ["V1","V4","V10","V14","V12","V17"]:
        df[f"{c}_sq"] = df[c]**2
    vcols = [f"V{i}" for i in range(1,29)]
    df["v_sum"] = df[vcols].sum(axis=1)
    df["v_abs_mean"] = df[vcols].abs().mean(axis=1)
    cols = [c for c in df.columns if c != "Class"]
    return np.nan_to_num(df[cols].values.astype(np.float32)), y, cols

def load_paysim():
    df = pd.read_csv("data/paysim_1m.csv")
    y = df["isFraud"].values
    for c in pd.get_dummies(df["type"],prefix="t",dtype=float).columns:
        df[c] = pd.get_dummies(df["type"],prefix="t",dtype=float)[c]
    df["bal_diff_o"] = df["oldbalanceOrg"]-df["newbalanceOrig"]
    df["bal_diff_d"] = df["oldbalanceDest"]-df["newbalanceDest"]
    df["amt_ratio_o"] = df["amount"]/(df["oldbalanceOrg"]+1)
    df["zero_after"] = (df["newbalanceOrig"]<0.01).astype(float)
    df["log_amt"] = np.log1p(df["amount"])
    df["orig_drain"] = df["bal_diff_o"]/(df["oldbalanceOrg"]+1)
    df["amt_x_flag"] = df["amount"]*df["isFlaggedFraud"]
    df["orig_wiped"] = (df["newbalanceOrig"]<1).astype(float)
    df["dest_empty"] = (df["oldbalanceDest"]<1).astype(float)
    df["amt_sq"] = df["amount"]**2
    df["amt_x_o"] = df["amount"]*df["oldbalanceOrg"]
    cols = [c for c in df.columns if c not in ("type","isFraud","isFlaggedFraud")]
    return np.nan_to_num(df[cols].values.astype(np.float32)), y, cols

def load_altman():
    rows_f, rows_l = [], []
    ci = 0
    for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
        usecols=["User","Month","Day","Time","Amount","Use Chip","MCC","Errors?","Is Fraud?"],
        low_memory=False, chunksize=500_000):
        chunk["amt"] = chunk["Amount"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float)
        chunk["label"] = (chunk["Is Fraud?"]=="Yes").astype(int)
        chip_map = {"Swipe Transaction":0,"Online Transaction":1,"Chip Transaction":2}
        chunk["chip"] = chunk["Use Chip"].map(chip_map).fillna(-1)
        chunk["mcc_n"] = chunk["MCC"].astype(str).str[:4].astype(float)/10000
        chunk["err"] = (chunk["Errors?"].fillna("")!="").astype(int)
        tp = chunk["Time"].str.split(":",expand=True)
        chunk["hr"] = tp[0].astype(float); chunk["mn"] = tp[1].astype(float)
        keep = ["amt","chip","mcc_n","err","hr","mn","Month","label","User"]
        f = chunk[chunk["label"]==1][keep]
        l = chunk[chunk["label"]==0].sample(min(80000,(chunk["label"]==0).sum()),random_state=42)[keep]
        rows_f.append(f); rows_l.append(l)
        ci += 1
        if ci >= 16: break
    df = pd.concat(rows_f+rows_l, ignore_index=True)
    gm = df["label"].mean()
    oof_te = np.zeros(len(df))
    for tri, vai in StratifiedKFold(3,shuffle=True,random_state=42).split(df,df["label"]):
        stats = df.iloc[tri].groupby("User")["label"].agg(["mean","count"])
        stats["s"] = (stats["mean"]*stats["count"]+gm*200)/(stats["count"]+200)
        oof_te[vai] = df.iloc[vai]["User"].map(stats["s"].to_dict()).fillna(gm).values
    df["user_te"] = oof_te
    df["log_amt"] = np.log1p(df["amt"])
    df["amt_x_hr"] = df["amt"]*df["hr"]
    df["amt_x_mcc"] = df["amt"]*df["mcc_n"]
    df["amt_x_chip"] = df["amt"]*df["chip"]
    df["amt_x_ute"] = df["amt"]*df["user_te"]
    df["amt_sq"] = df["amt"]**2
    y = df["label"].values
    cols = [c for c in df.columns if c not in ("label","User")]
    print(f"  Altman: {len(df):,} rows, {int(y.sum())} fraud")
    return np.nan_to_num(df[cols].values.astype(np.float32)), y, cols

def optuna_xgb(X, y, n, spw):
    def obj(tr):
        m = xgb.XGBClassifier(
            n_estimators=tr.suggest_int("n_estimators",200,500), max_depth=tr.suggest_int("max_depth",4,9),
            learning_rate=tr.suggest_float("lr",0.03,0.2,log=True),
            subsample=tr.suggest_float("subsample",0.6,1.0), colsample_bytree=tr.suggest_float("cs",0.5,1.0),
            min_child_weight=tr.suggest_int("mcw",1,8),
            reg_alpha=tr.suggest_float("ra",1e-4,5.0,log=True), reg_lambda=tr.suggest_float("rl",1e-4,5.0,log=True),
            scale_pos_weight=min(spw,tr.suggest_float("spw",1,min(spw,15))),
            random_state=42, n_jobs=-1, tree_method="hist", eval_metric="auc")
        return roc_auc_score(y, cross_val_predict(m,X,y,cv=3,method="predict_proba")[:,1])
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n, show_progress_bar=False)
    bp = s.best_params
    return dict(n_estimators=bp["n_estimators"],max_depth=bp["max_depth"],learning_rate=bp["lr"],
                subsample=bp["subsample"],colsample_bytree=bp["cs"],min_child_weight=bp["mcw"],
                reg_alpha=bp["ra"],reg_lambda=bp["rl"],scale_pos_weight=min(spw,bp["spw"])), s.best_value

def optuna_lgb(X, y, n, spw):
    def obj(tr):
        m = lgb.LGBMClassifier(
            n_estimators=tr.suggest_int("n_estimators",200,500), max_depth=tr.suggest_int("max_depth",4,9),
            learning_rate=tr.suggest_float("lr",0.03,0.2,log=True),
            subsample=tr.suggest_float("subsample",0.6,1.0), colsample_bytree=tr.suggest_float("cs",0.5,1.0),
            min_child_samples=tr.suggest_int("mcs",5,60),
            reg_alpha=tr.suggest_float("ra",1e-4,5.0,log=True), reg_lambda=tr.suggest_float("rl",1e-4,5.0,log=True),
            scale_pos_weight=min(spw,tr.suggest_float("spw",1,min(spw,15))),
            random_state=42, n_jobs=-1, verbose=-1)
        return roc_auc_score(y, cross_val_predict(m,X,y,cv=3,method="predict_proba")[:,1])
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n, show_progress_bar=False)
    bp = s.best_params
    return dict(n_estimators=bp["n_estimators"],max_depth=bp["max_depth"],learning_rate=bp["lr"],
                subsample=bp["subsample"],colsample_bytree=bp["cs"],min_child_samples=bp["mcs"],
                reg_alpha=bp["ra"],reg_lambda=bp["rl"],scale_pos_weight=min(spw,bp["spw"])), s.best_value

def run(name, X, y, trials=10):
    print(f"\n{'='*60}\n  {name}: {X.shape[0]:,} rows, {X.shape[1]} feat, {int(y.sum())} fraud ({y.mean()*100:.3f}%)\n{'='*60}")
    t0 = time.time()
    Xs = RobustScaler().fit_transform(X)
    spw = (len(y)-int(y.sum()))/max(int(y.sum()),1)
    n_sub = min(30_000, len(y))
    sub = np.random.RandomState(42).choice(len(y), n_sub, replace=False)

    print(f"  XGB Optuna ({trials})...")
    xbp, xauc = optuna_xgb(Xs[sub], y[sub], trials, spw)
    print(f"    Best: {xauc:.6f} ({time.time()-t0:.0f}s)")

    print(f"  LGB Optuna ({trials})...")
    lbpa, lauc = optuna_lgb(Xs[sub], y[sub], trials, spw)
    print(f"    Best: {lauc:.6f} ({time.time()-t0:.0f}s)")

    models = [
        ("xgb", xgb.XGBClassifier(**xbp, random_state=42, n_jobs=-1, tree_method="hist", eval_metric="auc")),
        ("lgb", lgb.LGBMClassifier(**lbpa, random_state=42, n_jobs=-1, verbose=-1)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42)),
        ("lr", LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)),
    ]

    # OOF on subsample for speed, then evaluate on held-out
    n_oof = min(80_000, len(y))
    oof_idx = np.random.RandomState(99).choice(len(y), n_oof, replace=False)
    rest_idx = np.setdiff1d(np.arange(len(y)), oof_idx)
    Xo, yo = Xs[oof_idx], y[oof_idx]
    Xr, yr = Xs[rest_idx], y[rest_idx]

    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    print(f"  OOF 3-fold on {n_oof:,} rows...")
    oof = np.zeros((n_oof, 4))
    for j, (nm, m) in enumerate(models):
        oof[:,j] = cross_val_predict(m, Xo, yo, cv=skf, method="predict_proba")[:,1]
        print(f"    {nm}: {roc_auc_score(yo,oof[:,j]):.6f} ({time.time()-t0:.0f}s)")

    # Held-out evaluation
    print("  Held-out eval...")
    oof_held = np.zeros((len(rest_idx), 4))
    for j, (nm, m) in enumerate(models):
        m.fit(Xo, yo)
        oof_held[:,j] = m.predict_proba(Xr)[:,1]
    held_auc = [round(roc_auc_score(yr, oof_held[:,j]),6) for j in range(4)]
    print(f"    Held-out AUCs: {dict(zip(['xgb','lgb','rf','lr'], held_auc))}")

    # Weight search on OOF subsample
    best_auc, best_w = 0, np.ones(4)/4
    for a in np.arange(0.15,0.65,0.05):
        for b in np.arange(0.15,0.65,0.05):
            for c in np.arange(0.0,0.3,0.05):
                d = 1-a-b-c
                if d<0: continue
                w = np.array([a,b,c,d]); bl = oof@w
                a2 = roc_auc_score(yo,bl)
                if a2>best_auc: best_auc=a2; best_w=w.copy()
    w_dict = {models[i][0]:round(float(best_w[i]),3) for i in range(4)}

    # LR stacker on OOF
    meta = LogisticRegression(C=100, max_iter=2000, random_state=42)
    stk = cross_val_predict(meta, oof, yo, cv=skf, method="predict_proba")[:,1]
    stk_auc = roc_auc_score(yo, stk)

    print(f"\n    OOF Weighted: {best_auc:.6f} {w_dict}")
    print(f"    OOF LR stack: {stk_auc:.6f}")

    # Evaluate on held-out
    if stk_auc > best_auc:
        fp_held = oof_held @ np.array([meta.coef_[0][j] for j in range(4)])
        fp_held = 1/(1+np.exp(-fp_held))  # sigmoid since stacker is Ridge-like
        method = "lr_stacker"
    else:
        fp_held = oof_held @ best_w
        method = "weighted"

    # Also try held-out with individual models
    held_best = max(range(4), key=lambda j: roc_auc_score(yr, oof_held[:,j]))

    m_oof = met(yo, oof@best_w if method=="weighted" else stk)
    m_held = met(yr, fp_held)
    print(f"\n  OOF {name} ({method}):")
    for k,v in m_oof.items(): print(f"    {k}: {v}")
    print(f"  HELD-OUT {name} ({method}):")
    for k,v in m_held.items(): print(f"    {k}: {v}")
    print(f"  Best single held model: {models[held_best][0]} AUC={held_auc[held_best]}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    result = dict(dataset=name, n_rows=len(y), n_features=X.shape[1], n_fraud=int(y.sum()),
        fraud_rate=round(float(y.mean()),6), method=method, best_weights=w_dict,
        cv_aucs={models[i][0]:round(roc_auc_score(yo,oof[:,i]),6) for i in range(4)},
        held_out_aucs=dict(zip(['xgb','lgb','rf','lr'], held_auc)),
        stacker_auc=round(stk_auc,6), weighted_auc=round(best_auc,6),
        metrics_oof=m_oof, metrics_held=m_held,
        xgb_params=xbp, lgb_params=lbpa, elapsed_s=round(time.time()-t0),
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out = R/f"push_max_{name.lower()}.json"
    with open(out,"w") as f: json.dump(result,f,indent=2)
    print(f"  Saved: {out}")
    return result

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--dataset",default="all"); p.add_argument("--trials",type=int,default=10)
    a = p.parse_args()
    t0=time.time(); res={}
    for ds in (["ulb","altman","paysim"] if a.dataset=="all" else [a.dataset]):
        X,y,_={"ulb":load_ulb,"paysim":load_paysim,"altman":load_altman}[ds]()
        res[ds]=run(ds.upper(),X,y,a.trials)
    print(f"\n{'='*60}\n  DONE in {time.time()-t0:.0f}s")
    for n,r in res.items(): print(f"  {n}: ROC-AUC={r['metrics_held']['roc_auc']:.6f} R@1%FPR={r['metrics_held']['r1']:.6f}")
    print(f"{'='*60}")
