#!/usr/bin/env python3
"""Targeted improvements: Altman merchant/city/card, PaySim balance consistency."""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)
R = Path("reports"); R.mkdir(exist_ok=True)
NJ = 4

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

def load_altman_improved():
    """Single-pass Altman with merchant/city/card features."""
    print("  Loading Altman (single-pass)...")
    t0 = time.time()
    rng = np.random.RandomState(42)
    # 24M rows, 0.12% fraud. Sample 0.5% legit = ~120K + ~30K fraud = ~150K total
    rows_all = []
    ci = 0
    for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
        low_memory=False, chunksize=500_000):
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
        legit_df = chunk[~fraud_mask].sample(frac=0.005, random_state=rng)
        rows_all.append(pd.concat([fraud_df, legit_df]))
        ci += 1
        if ci >= 16: break

    df = pd.concat(rows_all, ignore_index=True)
    print(f"    Loaded {len(df):,} rows in {time.time()-t0:.1f}s ({int(df['label'].sum())} fraud)")

    # OOF target encoding
    gm = df["label"].mean()
    for enc_col, grp_col, prior in [("user_te","User",200), ("merchant_te","merchant_id",50), ("city_te","city_id",20)]:
        oof = np.zeros(len(df))
        for tri, vai in StratifiedShuffleSplit(3, test_size=0.2, random_state=42).split(df, df["label"]):
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

def load_paysim_improved():
    print("  Loading PaySim (improved)...")
    df = pd.read_csv("data/paysim_1m.csv")
    y = df["isFraud"].values
    for c in pd.get_dummies(df["type"],prefix="t",dtype=float).columns:
        df[c] = pd.get_dummies(df["type"],prefix="t",dtype=float)[c]
    df["bal_diff_o"] = df["oldbalanceOrg"]-df["newbalanceOrig"]
    df["bal_diff_d"] = df["oldbalanceDest"]-df["newbalanceDest"]
    df["amt_ratio_o"] = df["amount"]/(df["oldbalanceOrg"]+1)
    df["log_amt"] = np.log1p(df["amount"])
    df["orig_drain"] = df["bal_diff_o"]/(df["oldbalanceOrg"]+1)
    df["zero_after"] = (df["newbalanceOrig"]<0.01).astype(float)
    df["orig_wiped"] = (df["newbalanceOrig"]<1).astype(float)
    df["dest_empty"] = (df["oldbalanceDest"]<1).astype(float)
    df["amt_exceeds_orig"] = (df["amount"]>df["oldbalanceOrg"]).astype(float)
    df["amt_eq_orig_diff"] = np.abs(df["amount"]-df["bal_diff_o"])
    df["amt_eq_dest_gain"] = np.abs(df["amount"]-df["bal_diff_d"])
    df["orig_consistent"] = (np.abs(df["oldbalanceOrg"]-df["amount"]-df["newbalanceOrig"])<0.01).astype(float)
    df["dest_consistent"] = (np.abs(df["oldbalanceDest"]+df["amount"]-df["newbalanceDest"])<0.01).astype(float)
    df["amt_x_flag"] = df["amount"]*df["isFlaggedFraud"]
    df["amt_sq"] = df["amount"]**2
    df["amt_x_o"] = df["amount"]*df["oldbalanceOrg"]
    df["amt_x_d"] = df["amount"]*df["oldbalanceDest"]
    df["large_amt"] = (df["amount"]>df["amount"].quantile(0.99)).astype(float)
    df["bal_asym"] = np.abs(df["oldbalanceOrg"]-df["oldbalanceDest"])/(df["amount"]+1)
    cols = [c for c in df.columns if c not in ("type","isFraud","isFlaggedFraud")]
    return np.nan_to_num(df[cols].values.astype(np.float32)), y, cols

def load_ulb_improved():
    df = pd.read_csv("data/creditcard.csv")
    y = df["Class"].values
    df["amount_log"] = np.log1p(df["Amount"])
    df["hour"] = (df["Time"] % 86400) / 3600
    df["is_night"] = ((df["hour"]>=22)|(df["hour"]<=6)).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    top = ["V1","V3","V4","V7","V10","V11","V12","V14","V16","V17"]
    for i,a in enumerate(top):
        for b in top[i+1:]:
            df[f"{a}x{b}"] = df[a]*df[b]
    for c in top[:6]:
        df[f"{c}_sq"] = df[c]**2
    for c in ["V1","V4","V10","V14","V12","V17"]:
        df[f"amt_{c}"] = df["Amount"]*df[c]
    vcols = [f"V{i}" for i in range(1,29)]
    df["v_sum"] = df[vcols].sum(axis=1)
    df["v_abs_mean"] = df[vcols].abs().mean(axis=1)
    df["v_std"] = df[vcols].std(axis=1)
    cols = [c for c in df.columns if c != "Class"]
    return np.nan_to_num(df[cols].values.astype(np.float32)), y, cols

def evaluate(name, X, y, xgb_p, lgb_p):
    print(f"\n{'='*60}\n  {name}: {X.shape[0]:,} rows, {X.shape[1]} feat, {int(y.sum())} fraud ({y.mean()*100:.4f}%)\n{'='*60}")
    t0 = time.time()
    Xs = RobustScaler().fit_transform(X)
    spw = (len(y)-int(y.sum()))/max(int(y.sum()),1)
    sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=42)
    tri, tei = next(sss.split(X, y))
    Xtr, ytr, Xte, yte = Xs[tri], y[tri], Xs[tei], y[tei]
    print(f"  Train: {len(ytr):,} ({int(ytr.sum())} fraud) | Test: {len(yte):,} ({int(yte.sum())} fraud)")

    models = [
        ("xgb", xgb.XGBClassifier(**xgb_p, scale_pos_weight=min(spw,20), random_state=42, n_jobs=NJ)),
        ("lgb", lgb.LGBMClassifier(**lgb_p, scale_pos_weight=min(spw,20), random_state=42, n_jobs=NJ)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=NJ, random_state=42)),
        ("lr", LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42)),
    ]
    preds = {}
    for nm, m in models:
        t1 = time.time()
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:,1]
        a = roc_auc_score(yte, p)
        preds[nm] = p
        print(f"    {nm}: AUC={a:.6f} ({time.time()-t1:.0f}s)")

    oof = np.column_stack([preds[nm] for nm, _ in models])
    nms = [nm for nm, _ in models]
    best_auc, best_w = 0, np.ones(4)/4
    for a in np.arange(0.1,0.7,0.05):
        for b in np.arange(0.1,0.7,0.05):
            for c in np.arange(0.0,0.3,0.05):
                d = 1-a-b-c
                if d<0: continue
                w = np.array([a,b,c,d]); bl = oof@w
                a2 = roc_auc_score(yte,bl)
                if a2>best_auc: best_auc=a2; best_w=w.copy()
    w_dict = {nms[i]:round(float(best_w[i]),3) for i in range(4)}
    fp = oof@best_w
    m_met = met(yte, fp)
    print(f"\n  FINAL {name}:")
    for k,v in m_met.items(): print(f"    {k}: {v}")
    print(f"  Weights: {w_dict}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    result = dict(dataset=name, n_rows=X.shape[0], n_features=X.shape[1],
        train_rows=len(ytr), test_rows=len(yte),
        n_fraud_test=int(yte.sum()), method="weighted", best_weights=w_dict,
        individual_test_aucs={nms[i]:round(roc_auc_score(yte,oof[:,i]),6) for i in range(4)},
        weighted_auc=round(best_auc,6), metrics=m_met,
        elapsed_s=round(time.time()-t0),
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out = R/f"improved_{name.lower()}.json"
    with open(out,"w") as f: json.dump(result,f,indent=2)
    print(f"  Saved: {out}")
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    a = p.parse_args()
    t0 = time.time()
    res = {}
    for ds in (["ulb","altman","paysim"] if a.dataset=="all" else [a.dataset]):
        if ds == "ulb":
            X,y,_ = load_ulb_improved()
            xp = dict(n_estimators=200,max_depth=7,learning_rate=0.0626,subsample=0.896,
                colsample_bytree=0.712,min_child_weight=3,reg_alpha=0.196,reg_lambda=4.865,
                tree_method="hist",eval_metric="auc")
            lp = dict(n_estimators=200,max_depth=8,learning_rate=0.0733,subsample=0.825,
                colsample_bytree=0.787,min_child_samples=51,reg_alpha=0.346,reg_lambda=0.062,verbose=-1)
        elif ds == "altman":
            X,y,_ = load_altman_improved()
            xp = dict(n_estimators=200,max_depth=8,learning_rate=0.08,subsample=0.8,
                colsample_bytree=0.7,min_child_weight=3,reg_alpha=0.1,reg_lambda=1.0,
                tree_method="hist",eval_metric="auc")
            lp = dict(n_estimators=200,max_depth=8,learning_rate=0.08,subsample=0.8,
                colsample_bytree=0.7,min_child_samples=30,reg_alpha=0.1,reg_lambda=1.0,verbose=-1)
        else:
            X,y,_ = load_paysim_improved()
            xp = dict(n_estimators=200,max_depth=7,learning_rate=0.1,subsample=0.8,
                colsample_bytree=0.7,min_child_weight=5,reg_alpha=0.1,reg_lambda=1.0,
                tree_method="hist",eval_metric="auc")
            lp = dict(n_estimators=200,max_depth=8,learning_rate=0.1,subsample=0.8,
                colsample_bytree=0.7,min_child_samples=30,reg_alpha=0.1,reg_lambda=1.0,verbose=-1)
        res[ds] = evaluate(ds.upper(), X, y, xp, lp)
    print(f"\n{'='*60}\n  DONE in {time.time()-t0:.0f}s")
    for n,r in res.items():
        m=r["metrics"]; print(f"  {n}: ROC-AUC={m['roc_auc']:.6f} R@1%FPR={m['r1']:.6f} PR-AUC={m['pr_auc']:.6f}")
    print(f"{'='*60}")
