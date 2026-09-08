#!/usr/bin/env python3
"""Fast performance push: Optuna + 4-model ensemble on ULB, PaySim, Altman.

Strategy:
1. Optuna tunes XGB + LGB on a 40K subsample (fast)
2. RF and LR use good defaults
3. OOF stacking on full data with tuned params
4. Weight search for best blend
"""
import json, hashlib, time, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)
DATA_DIR, REPORTS_DIR = Path("data"), Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

def recall_at_fpr(y, p, fpr_target):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    m = fpr <= fpr_target
    return float(tpr[m].max()) if m.any() else 0.0

def eval_all(y, p):
    roc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    r1 = recall_at_fpr(y, p, 0.01)
    r05 = recall_at_fpr(y, p, 0.005)
    r01 = recall_at_fpr(y, p, 0.001)
    prec, rec, thrs = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    bi = np.argmax(f1s)
    f1 = f1s[bi]
    thr = thrs[min(bi, len(thrs)-1)]
    return dict(roc_auc=round(roc,6), pr_auc=round(pr,6), brier=round(brier,6),
                r1=round(r1,6), r05=round(r05,6), r01=round(r01,6),
                f1=round(f1,6), opt_thr=round(thr,6))

# ── Dataset loaders ──

def load_ulb():
    df = pd.read_csv(DATA_DIR / "creditcard.csv")
    y = df["Class"].values
    df["amount_log"] = np.log1p(df["Amount"])
    df["hour"] = (df["Time"] % 86400) / 3600
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    for a, b in [("V1","V2"),("V3","V4"),("V7","V8"),("V10","V14"),("V11","V17"),("V12","V14"),("V16","V18")]:
        df[f"{a}_{b}"] = df[a] * df[b]
    for c in ["V1","V4","V14"]:
        df[f"{c}_sq"] = df[c] ** 2
    df["amount_x_v1"] = df["Amount"] * df["V1"]
    df["amount_x_v14"] = df["Amount"] * df["V14"]
    vcols = [c for c in df.columns if c.startswith("V")]
    df["v_sum"] = df[vcols].sum(axis=1)
    df["v_abs_mean"] = df[vcols].abs().mean(axis=1)
    cols = [c for c in df.columns if c != "Class"]
    X = np.nan_to_num(df[cols].values.astype(np.float32))
    return X, y, cols

def load_paysim():
    df = pd.read_csv(DATA_DIR / "paysim_1m.csv")
    y = df["isFraud"].values
    for d in pd.get_dummies(df["type"], prefix="t", dtype=float).columns:
        df[d] = pd.get_dummies(df["type"], prefix="t", dtype=float)[d]
    df["bal_diff_o"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["bal_diff_d"] = df["oldbalanceDest"] - df["newbalanceDest"]
    df["amt_ratio_o"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amt_ratio_d"] = df["amount"] / (df["oldbalanceDest"] + 1)
    df["zero_after"] = (df["newbalanceOrig"] < 0.01).astype(float)
    df["log_amt"] = np.log1p(df["amount"])
    df["orig_drain"] = df["bal_diff_o"] / (df["oldbalanceOrg"] + 1)
    df["dest_gain"] = df["bal_diff_d"] / (df["oldbalanceDest"] + 1)
    df["amt_x_flag"] = df["amount"] * df["isFlaggedFraud"]
    cols = [c for c in df.columns if c not in ("type","isFraud","isFlaggedFraud")]
    X = np.nan_to_num(df[cols].values.astype(np.float32))
    return X, y, cols

def load_altman():
    rows_f, rows_l = [], []
    for i, chunk in enumerate(pd.read_csv(
        DATA_DIR / "credit_card_transactions-ibm_v2.csv",
        usecols=["User","Year","Month","Day","Time","Amount","Use Chip","MCC","Errors?","Is Fraud?"],
        low_memory=False, chunksize=500_000)):
        chunk["amt"] = chunk["Amount"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float)
        chunk["label"] = (chunk["Is Fraud?"]=="Yes").astype(int)
        chip_map = {"Swipe Transaction":0,"Online Transaction":1,"Chip Transaction":2}
        chunk["chip"] = chunk["Use Chip"].map(chip_map).fillna(-1)
        chunk["mcc_n"] = chunk["MCC"].astype(str).str[:4].astype(float) / 10000
        chunk["err"] = (chunk["Errors?"].fillna("")!="").astype(int)
        tp = chunk["Time"].str.split(":", expand=True)
        chunk["hr"] = tp[0].astype(float)
        chunk["mn"] = tp[1].astype(float)
        keep = ["amt","chip","mcc_n","err","hr","mn","Month","label","User"]
        f = chunk[chunk["label"]==1][keep]
        l = chunk[chunk["label"]==0].sample(min(100000, (chunk["label"]==0).sum()), random_state=42)[keep]
        rows_f.append(f); rows_l.append(l)
        if i >= 15: break
    df = pd.concat(rows_f + rows_l, ignore_index=True)
    # OOF target encode user
    gm = df["label"].mean()
    oof_te = np.zeros(len(df))
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    for tri, vai in skf.split(df, df["label"]):
        stats = df.iloc[tri].groupby("User")["label"].agg(["mean","count"])
        stats["s"] = (stats["mean"]*stats["count"] + gm*50)/(stats["count"]+50)
        m = stats["s"].to_dict()
        oof_te[vai] = df.iloc[vai]["User"].map(m).fillna(gm).values
    df["user_te"] = oof_te
    df["log_amt"] = np.log1p(df["amt"])
    df["amt_x_hr"] = df["amt"] * df["hr"]
    df["amt_x_mcc"] = df["amt"] * df["mcc_n"]
    y = df["label"].values
    cols = [c for c in df.columns if c not in ("label","User")]
    X = np.nan_to_num(df[cols].values.astype(np.float32))
    print(f"  Altman: {len(df):,} rows, {int(y.sum())} fraud")
    return X, y, cols

# ── Optuna objectives ──

def optuna_xgb(X, y, n_trials, spw):
    def obj(trial):
        m = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("ne",300,600), max_depth=trial.suggest_int("md",5,10),
            learning_rate=trial.suggest_float("lr",0.05,0.2,log=True),
            subsample=trial.suggest_float("ss",0.7,1.0), colsample_bytree=trial.suggest_float("cs",0.6,1.0),
            min_child_weight=trial.suggest_int("mcw",1,8),
            reg_alpha=trial.suggest_float("ra",1e-4,5.0,log=True),
            reg_lambda=trial.suggest_float("rl",1e-4,5.0,log=True),
            scale_pos_weight=min(spw, trial.suggest_float("spw",1,min(spw,20))),
            random_state=42, n_jobs=-1, tree_method="hist", eval_metric="auc")
        p = cross_val_predict(m, X, y, cv=3, method="predict_proba")[:,1]
        return roc_auc_score(y, p)
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = s.best_params
    return dict(n_estimators=bp["ne"], max_depth=bp["md"], learning_rate=bp["lr"],
                subsample=bp["ss"], colsample_bytree=bp["cs"], min_child_weight=bp["mcw"],
                reg_alpha=bp["ra"], reg_lambda=bp["rl"], scale_pos_weight=min(spw,bp["spw"]),
                random_state=42, n_jobs=-1, tree_method="hist", eval_metric="auc"), s.best_value

def optuna_lgb(X, y, n_trials, spw):
    def obj(trial):
        m = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int("ne",300,600), max_depth=trial.suggest_int("md",5,10),
            learning_rate=trial.suggest_float("lr",0.05,0.2,log=True),
            subsample=trial.suggest_float("ss",0.7,1.0), colsample_bytree=trial.suggest_float("cs",0.6,1.0),
            min_child_samples=trial.suggest_int("mcs",10,80),
            reg_alpha=trial.suggest_float("ra",1e-4,5.0,log=True),
            reg_lambda=trial.suggest_float("rl",1e-4,5.0,log=True),
            scale_pos_weight=min(spw, trial.suggest_float("spw",1,min(spw,20))),
            random_state=42, n_jobs=-1, verbose=-1)
        p = cross_val_predict(m, X, y, cv=3, method="predict_proba")[:,1]
        return roc_auc_score(y, p)
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = s.best_params
    return dict(n_estimators=bp["ne"], max_depth=bp["md"], learning_rate=bp["lr"],
                subsample=bp["ss"], colsample_bytree=bp["cs"], min_child_samples=bp["mcs"],
                reg_alpha=bp["ra"], reg_lambda=bp["rl"], scale_pos_weight=min(spw,bp["spw"]),
                random_state=42, n_jobs=-1, verbose=-1), s.best_value

# ── Main optimization ──

def optimize(name, X, y, n_trials=20):
    print(f"\n{'='*60}\n  {name}: {X.shape[0]:,} rows, {X.shape[1]} features, {int(y.sum())} fraud ({y.mean()*100:.3f}%)\n{'='*60}")
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    spw = (len(y)-int(y.sum())) / max(int(y.sum()),1)
    skf = StratifiedKFold(3, shuffle=True, random_state=42)

    # Optuna subsample
    n_sub = min(40_000, len(y))
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(y), n_sub, replace=False)
    Xo, yo = Xs[sub_idx], y[sub_idx]

    print("  Optuna XGB...")
    xgb_p, xgb_auc = optuna_xgb(Xo, yo, n_trials, spw)
    print(f"    Best: {xgb_auc:.6f}")

    print("  Optuna LGB...")
    lgb_p, lgb_auc = optuna_lgb(Xo, yo, n_trials, spw)
    print(f"    Best: {lgb_auc:.6f}")

    # Build models
    models = {
        "xgb": xgb.XGBClassifier(**xgb_p),
        "lgb": lgb.LGBMClassifier(**lgb_p),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42),
        "lr": LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42),
    }

    # OOF predictions
    print("  OOF stacking...")
    oof = np.zeros((len(y), 4))
    cv_aucs = {}
    for j, (nm, m) in enumerate(models.items()):
        oof[:, j] = cross_val_predict(m, Xs, y, cv=skf, method="predict_proba")[:,1]
        cv_aucs[nm] = round(roc_auc_score(y, oof[:,j]),6)
        print(f"    {nm}: {cv_aucs[nm]:.6f}")

    # Weight search
    best_auc, best_w = 0, None
    for wx in np.arange(0.05, 0.6, 0.05):
        for wl in np.arange(0.05, 0.6, 0.05):
            for wr in np.arange(0.0, 0.4, 0.05):
                wlro = 1.0 - wx - wl - wr
                if wlro < 0: continue
                blend = wx*oof[:,0] + wl*oof[:,1] + wr*oof[:,2] + wlro*oof[:,3]
                a = roc_auc_score(y, blend)
                if a > best_auc:
                    best_auc = a
                    best_w = {"xgb":round(wx,3),"lgb":round(wl,3),"rf":round(wr,3),"lr":round(wlro,3)}

    # Stacker
    meta = LogisticRegression(C=100, max_iter=2000, random_state=42)
    stack_oof = cross_val_predict(meta, oof, y, cv=skf, method="predict_proba")[:,1]
    stack_auc = roc_auc_score(y, stack_oof)

    print(f"    Weighted: {best_auc:.6f} {best_w}")
    print(f"    Stacker:  {stack_auc:.6f}")

    # Choose best
    if stack_auc >= best_auc:
        final_p = stack_oof; method = "stacker"
    else:
        final_p = sum(best_w[k]*oof[:,i] for i,k in enumerate(models.keys()))
        method = "weighted"

    # Calibrate
    cal = CalibratedClassifierCV(LogisticRegression(C=10, max_iter=1000), cv=3, method="isotonic")
    cal.fit(oof, y)
    cal_p = cal.predict_proba(oof)[:,1]

    m_raw = eval_all(y, final_p)
    m_cal = eval_all(y, cal_p)

    # Train final models
    for m in models.values():
        m.fit(Xs, y)
    final_all = np.column_stack([m.predict_proba(Xs)[:,1] for m in models.values()])
    if method == "stacker":
        meta.fit(oof, y)
        final_all_p = meta.predict_proba(final_all)[:,1]
    else:
        final_all_p = sum(best_w[k]*final_all[:,i] for i,k in enumerate(models.keys()))

    # Feature importance (top 10 from XGB)
    fi = dict(zip(range(Xs.shape[1]), models["xgb"].feature_importances_))
    top10 = sorted(fi.items(), key=lambda x: -x[1])[:10]

    print(f"\n  FINAL {name}: {method}")
    print(f"    ROC-AUC:  {m_raw['roc_auc']:.6f}")
    print(f"    R@1%FPR:  {m_raw['r1']:.6f}")
    print(f"    R@0.5%FPR:{m_raw['r05']:.6f}")
    print(f"    PR-AUC:   {m_raw['pr_auc']:.6f}")
    print(f"    Cal AUC:  {m_cal['roc_auc']:.6f}")
    print(f"    Cal R@1%: {m_cal['r1']:.6f}")

    result = dict(
        dataset=name, n_rows=len(y), n_features=X.shape[1], n_fraud=int(y.sum()),
        fraud_rate=round(float(y.mean()),6), method=method,
        cv_aucs=cv_aucs, stacker_auc=round(stack_auc,6),
        weighted_auc=round(best_auc,6), best_weights=best_w,
        metrics_raw=m_raw, metrics_calibrated=m_cal,
        optuna_trials=n_trials,
        xgb_params={k:v for k,v in xgb_p.items() if k!="eval_metric"},
        lgb_params={k:v for k,v in lgb_p.items()},
        data_hash=hashlib.md5(str(int(y.sum())).encode()).hexdigest()[:16],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    out = REPORTS_DIR / f"max_perf_{name.lower()}.json"
    with open(out, "w") as f: json.dump(result, f, indent=2)
    print(f"  Saved: {out}")
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ulb","paysim","altman","all"], default="all")
    p.add_argument("--trials", type=int, default=20)
    args = p.parse_args()
    t0 = time.time()
    res = {}
    for ds in (["ulb","paysim","altman"] if args.dataset=="all" else [args.dataset]):
        loader = {"ulb": load_ulb, "paysim": load_paysim, "altman": load_altman}[ds]
        X, y, _ = loader()
        res[ds] = optimize(ds.upper(), X, y, args.trials)
    print(f"\n{'='*60}\n  TOTAL: {time.time()-t0:.0f}s")
    for n, r in res.items():
        print(f"  {n}: ROC-AUC={r['metrics_raw']['roc_auc']:.4f} R@1%FPR={r['metrics_raw']['r1']:.4f}")
    print(f"{'='*60}")
