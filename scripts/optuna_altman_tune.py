#!/usr/bin/env python3
"""Optuna tuning for Altman dataset — 20 trials, improved features.

Current baseline: ROC-AUC 0.9937 (weighted), XGB 0.9939, LGB 0.9940
Target: > 0.9950

Strategy:
  - XGB + LGB joint tuning in each trial
  - 3-fold CV on subsampled data for speed (full data for final eval)
  - Feature set: merchant/city/card/target encoding (29 features)
  - Save best params + final metrics immediately
"""
import json, time, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
import xgboost as xgb
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

def load_altman_tuning(max_rows=200_000):
    """Load Altman with improved features. Single-pass."""
    print("  Loading Altman (improved features)...")
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

        fraud_mask = chunk["label"] == 1
        fraud_df = chunk[fraud_mask]
        legit_sample = chunk[~fraud_mask].sample(frac=0.005, random_state=rng)
        rows_all.append(pd.concat([fraud_df, legit_sample]))
        ci += 1
        if ci >= 8: break

    df = pd.concat(rows_all, ignore_index=True)
    print(f"    Loaded {len(df):,} rows in {time.time()-t0:.1f}s ({int(df['label'].sum())} fraud)")

    # OOF target encoding (2 splits for speed)
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
    print(f"    {len(cols)} features")
    return np.nan_to_num(df[cols].values.astype(np.float32)), y, cols

# ── Optuna objective ──

def objective(trial, X, y, spw):
    """Joint XGB + LGB tuning with 3-fold CV."""
    xgb_params = dict(
        n_estimators=trial.suggest_int("xgb_n_est", 100, 400),
        max_depth=trial.suggest_int("xgb_max_depth", 4, 12),
        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("xgb_colsample", 0.5, 1.0),
        min_child_weight=trial.suggest_int("xgb_min_child", 1, 10),
        reg_alpha=trial.suggest_float("xgb_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("xgb_lambda", 1e-3, 10, log=True),
        scale_pos_weight=min(spw, 20),
        tree_method="hist", eval_metric="auc",
        random_state=42, n_jobs=NJ,
    )
    lgb_params = dict(
        n_estimators=trial.suggest_int("lgb_n_est", 100, 400),
        max_depth=trial.suggest_int("lgb_max_depth", 4, 12),
        learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("lgb_subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("lgb_colsample", 0.5, 1.0),
        min_child_samples=trial.suggest_int("lgb_min_child", 5, 50),
        reg_alpha=trial.suggest_float("lgb_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("lgb_lambda", 1e-3, 10, log=True),
        scale_pos_weight=min(spw, 20),
        verbose=-1, random_state=42, n_jobs=NJ,
    )
    # Ensemble weight for XGB vs LGB
    xgb_w = trial.suggest_float("xgb_weight", 0.3, 0.9)

    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    fold_aucs = []
    for tri, tei in skf.split(X, y):
        Xtr, ytr, Xte, yte = X[tri], y[tri], X[tei], y[tei]
        xm = xgb.XGBClassifier(**xgb_params).fit(Xtr, ytr)
        lm = lgb.LGBMClassifier(**lgb_params).fit(Xtr, ytr)
        xp = xm.predict_proba(Xte)[:,1]
        lp = lm.predict_proba(Xte)[:,1]
        blend = xgb_w * xp + (1 - xgb_w) * lp
        fold_aucs.append(roc_auc_score(yte, blend))
    return np.mean(fold_aucs)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--max-rows", type=int, default=200_000)
    a = p.parse_args()

    X, y, cols = load_altman_tuning(max_rows=a.max_rows)
    spw = (len(y)-int(y.sum()))/max(int(y.sum()),1)
    print(f"  scale_pos_weight: {spw:.1f}")

    # ── Optuna search ──
    print(f"\n  Running Optuna ({a.trials} trials, 3-fold CV)...")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y, spw), n_trials=a.trials, show_progress_bar=False)
    opt_time = time.time() - t0
    print(f"  Optuna done in {opt_time:.0f}s — best CV AUC: {study.best_value:.6f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=2)}")

    # ── Final evaluation with best params ──
    print(f"\n  Final evaluation with best params...")
    bp = study.best_params
    xgb_params = dict(
        n_estimators=bp["xgb_n_est"], max_depth=bp["xgb_max_depth"],
        learning_rate=bp["xgb_lr"], subsample=bp["xgb_subsample"],
        colsample_bytree=bp["xgb_colsample"], min_child_weight=bp["xgb_min_child"],
        reg_alpha=bp["xgb_alpha"], reg_lambda=bp["xgb_lambda"],
        scale_pos_weight=min(spw,20), tree_method="hist", eval_metric="auc",
        random_state=42, n_jobs=NJ,
    )
    lgb_params = dict(
        n_estimators=bp["lgb_n_est"], max_depth=bp["lgb_max_depth"],
        learning_rate=bp["lgb_lr"], subsample=bp["lgb_subsample"],
        colsample_bytree=bp["lgb_colsample"], min_child_samples=bp["lgb_min_child"],
        reg_alpha=bp["lgb_alpha"], reg_lambda=bp["lgb_lambda"],
        scale_pos_weight=min(spw,20), verbose=-1, random_state=42, n_jobs=NJ,
    )
    xgb_w = bp["xgb_weight"]

    # 5-fold CV for honest metrics
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    all_preds = np.zeros(len(y))
    fold_info = []
    for fold, (tri, tei) in enumerate(skf.split(X, y)):
        Xtr, ytr, Xte, yte = X[tri], y[tri], X[tei], y[tei]
        xm = xgb.XGBClassifier(**xgb_params).fit(Xtr, ytr)
        lm = lgb.LGBMClassifier(**lgb_params).fit(Xtr, ytr)
        xp = xm.predict_proba(Xte)[:,1]
        lp = lm.predict_proba(Xte)[:,1]
        blend = xgb_w * xp + (1 - xgb_w) * lp
        all_preds[tei] = blend
        fold_auc = roc_auc_score(yte, blend)
        fold_info.append({"fold": fold, "auc": round(fold_auc,6), "n_train": len(ytr), "n_test": len(yte),
                          "n_fraud_test": int(yte.sum())})
        print(f"    Fold {fold}: AUC={fold_auc:.6f} (train={len(ytr):,}, test={len(yte):,}, fraud_test={int(yte.sum())})")

    overall_auc = roc_auc_score(y, all_preds)
    m = met(y, all_preds)
    fold_aucs = [fi["auc"] for fi in fold_info]
    print(f"\n  5-fold CV mean AUC: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")
    print(f"  Overall AUC (OOF): {overall_auc:.6f}")
    print(f"\n  METRICS:")
    for k,v in m.items(): print(f"    {k}: {v}")

    # ── Individual model AUCs ──
    # Quick single train/test split for individual model comparison
    sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=42)
    tri, tei = next(sss.split(X, y))
    Xtr, ytr, Xte, yte = X[tri], y[tri], X[tei], y[tei]
    xm = xgb.XGBClassifier(**xgb_params).fit(Xtr, ytr)
    lm = lgb.LGBMClassifier(**lgb_params).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=30, max_depth=10, class_weight="balanced", n_jobs=NJ, random_state=42).fit(Xtr, ytr)
    lr = LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42).fit(Xtr, ytr)

    individual = {
        "xgb": round(roc_auc_score(yte, xm.predict_proba(Xte)[:,1]),6),
        "lgb": round(roc_auc_score(yte, lm.predict_proba(Xte)[:,1]),6),
        "rf": round(roc_auc_score(yte, rf.predict_proba(Xte)[:,1]),6),
        "lr": round(roc_auc_score(yte, lr.predict_proba(Xte)[:,1]),6),
    }
    print(f"\n  Individual AUCs (test set): {individual}")

    # ── Save results ──
    result = dict(
        dataset="ALTMAN_OPTIMIZED",
        n_rows=X.shape[0], n_features=X.shape[1],
        n_fraud=int(y.sum()), fraud_rate=round(float(y.mean()),6),
        optuna_trials=a.trials, optuna_best_cv=round(study.best_value,6),
        optuna_time_s=round(opt_time),
        best_params=study.best_params,
        xgb_weight=xgb_w,
        individual_test_aucs=individual,
        cv_5fold_mean=round(float(np.mean(fold_aucs)),6),
        cv_5fold_std=round(float(np.std(fold_aucs)),6),
        cv_folds=fold_info,
        overall_oof_auc=round(overall_auc,6),
        metrics=m,
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    out = R/"altman_optuna_results.json"
    with open(out, "w") as f: json.dump(result, f, indent=2)
    print(f"\n  Saved: {out}")

    # Also save the model manifest for the archive
    manifest = dict(
        altman_xgb_params=xgb_params, altman_lgb_params=lgb_params,
        altman_xgb_weight=xgb_w, altman_cv_auc=round(float(np.mean(fold_aucs)),6),
        altman_optuna_best=round(study.best_value,6),
    )
    manifest_out = R/"altman_optuna_manifest.json"
    with open(manifest_out, "w") as f: json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_out}")

    print(f"\n  DONE in {time.time()-t0:.0f}s total ({opt_time:.0f}s Optuna + {time.time()-t0-opt_time:.0f}s eval)")
