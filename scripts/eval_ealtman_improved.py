#!/usr/bin/env python3
"""Improved ealtman2019 — targeted features via vectorized pandas."""

import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SEED = 42
DATA = ROOT / "data" / "credit_card_transactions-ibm_v2.csv"

HIGH_RISK_MCCS = {"5045", "5094", "5932", "5732", "5816", "5712", "4112", "4131", "5311", "5300", "5310"}
HIGH_RISK_CITIES = {"ONLINE", "Rome", "Algiers", "Port au Prince", "Mexico City", "Abuja", "Istanbul", "Strasburg"}
US_STATES = {"TX", "CA", "OH", "MD", "FL", "NY", "PA", "IL", "MI", "GA", "NC", "NJ", "VA", "WA", "AZ",
             "MA", "TN", "IN", "MO", "MN", "CO", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "UT", "IA",
             "NV", "AR", "KS", "MS", "NE", "NM", "ID", "WV", "HI", "NH", "ME", "MT", "RI", "DE", "SD",
             "ND", "AK", "VT", "WY"}


def parse_amt(s):
    return s.str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)


def compute_metrics(yt, yp):
    yt = np.asarray(yt, int); yp = np.asarray(yp, float)
    n = len(yt); nf = int(yt.sum()); nl = n - nf
    prev = nf / n if n else 0.0
    roc = roc_auc_score(yt, yp) if nf and nl else None
    pr = average_precision_score(yt, yp) if nf else None
    br = brier_score_loss(yt, yp) if n else None
    ra = {}
    if nf and nl:
        fpr, tpr, _ = roc_curve(yt, yp)
        for tf, lb in [(1e-4, ".01"), (1e-3, ".1"), (5e-3, ".5"), (1e-2, "1")]:
            m = fpr <= tf
            ra[lb] = round(float(tpr[np.where(m)[0][-1]]), 4) if m.any() else 0.0
    else:
        for lb in (".01", ".1", ".5", "1"): ra[lb] = None
    ci = {}
    if n >= 100 and nf >= 20:
        rng = np.random.default_rng(SEED); ss = min(n, 100000)
        br_, bp_, b1_ = [], [], []
        for _ in range(200):
            idx = rng.choice(n, ss, replace=True)
            yt2, yp2 = yt[idx], yp[idx]
            if yt2.sum() >= 2 and (1-yt2).sum() >= 2:
                br_.append(roc_auc_score(yt2, yp2))
                bp_.append(average_precision_score(yt2, yp2))
                fa, ta, _ = roc_curve(yt2, yp2)
                m = fa <= 1e-2
                b1_.append(float(ta[np.where(m)[0][-1]]) if m.any() else 0.0)
        if br_:
            ci = {"roc_auc": [round(float(np.percentile(br_, 2.5)), 4), round(float(np.percentile(br_, 97.5)), 4)],
                  "pr_auc": [round(float(np.percentile(bp_, 2.5)), 4), round(float(np.percentile(bp_, 97.5)), 4)],
                  "recall_1pct": [round(float(np.percentile(b1_, 2.5)), 4), round(float(np.percentile(b1_, 97.5)), 4)]}
    return {"n": n, "n_fraud": nf, "n_legit": nl, "prevalence": round(prev, 6),
            "roc_auc": round(float(roc), 4) if roc else None,
            "pr_auc": round(float(pr), 4) if pr else None,
            "brier": round(float(br), 6) if br else None,
            "recall_at_0_01pct_fpr": ra.get(".01"), "recall_at_0_1pct_fpr": ra.get(".1"),
            "recall_at_0_5pct_fpr": ra.get(".5"), "recall_at_1pct_fpr": ra.get("1"), "ci": ci}


NAMES = [
    "is_online", "is_swipe", "is_chip",
    "amount", "amt_ratio_med", "amt_ratio_mean", "log_amt", "amt_zscore",
    "card_count", "card_velocity", "tenure_days", "n_merchants", "n_cities",
    "shared_accounts", "online_amt", "online_amt_ratio",
    "is_intl", "is_high_risk_city", "intl_online",
    "mcc_fraud_rate", "is_high_risk_mcc",
    "has_error", "bad_pin", "bad_cvv", "insuff_bal",
    "hour_norm", "is_weekend",
    "recipient_novelty", "days_since_last",
    "amt_micro", "amt_small", "amt_medium", "amt_large",
]


def main():
    print("=" * 70)
    print("PS-14 IMPROVED ealtman2019 (32 targeted features)")
    print("=" * 70)
    T0 = time.time()

    # Load 2M rows
    print("Loading 2M rows...")
    df = pd.read_csv(DATA, nrows=2_000_000, dtype=str)
    n = len(df)

    # Parse
    df["amt"] = parse_amt(df["Amount"]).values
    df["hr"] = df["Time"].str.split(":").str[0].astype(int).values
    df["uc"] = df["User"] + "_" + df["Card"]
    df["ux"] = pd.to_datetime(df["Year"]+"-"+df["Month"].str.zfill(2)+"-"+df["Day"].str.zfill(2)+" "+df["Time"]).astype(np.int64).values // 10**9
    df["dow"] = pd.to_datetime(df["Year"]+"-"+df["Month"].str.zfill(2)+"-"+df["Day"].str.zfill(2)).dt.dayofweek.values
    y = (df["Is Fraud?"]=="Yes").values.astype(int)
    print(f"  {n:,} rows, {int(y.sum()):,} fraud [{time.time()-T0:.0f}s]")

    # Sort by time
    sort_idx = np.argsort(df["ux"].values, kind="mergesort")
    df = df.iloc[sort_idx].reset_index(drop=True)
    y = y[sort_idx]

    # Group stats (vectorized)
    print("Computing group stats...")
    t1 = time.time()
    uc_stats = df.groupby("uc").agg(
        med=("amt","median"), mean=("amt","mean"), std=("amt","std"),
        cnt=("amt","count"), mu=("ux","min"),
        nm=("Merchant Name","nunique"), nc=("Merchant City","nunique"),
        mh=("hr","median"),
    ).reset_index()
    uc_stats["std"] = uc_stats["std"].fillna(1.0)

    mn_users = df.groupby("Merchant Name")["User"].nunique().reset_index()
    mn_users.columns = ["Merchant Name", "n_users"]

    # MCC fraud rates
    mcc_grps = df.groupby("MCC").agg(total=("amt","count"), fraud=("Is Fraud?", lambda x: (x=="Yes").sum())).reset_index()
    mcc_grps["rate"] = mcc_grps["fraud"] / mcc_grps["total"]
    mcc_rate_map = dict(zip(mcc_grps["MCC"], mcc_grps["rate"]))

    print(f"  Stats in {time.time()-t1:.0f}s")

    # Build features (vectorized)
    print("Building features...")
    t2 = time.time()

    m = df[["uc"]].merge(uc_stats, on="uc", how="left")
    mm = df[["Merchant Name"]].merge(mn_users, on="Merchant Name", how="left")

    amt = df["amt"].values
    hr = df["hr"].values.astype(np.float32)
    dow = df["dow"].values
    ux = df["ux"].values

    # MCC rate vectorized
    mcc_rate_arr = df["MCC"].map(mcc_rate_map).fillna(0.0).values.astype(np.float32)

    X = np.zeros((n, len(NAMES)), np.float32)

    # 0-2: Use Chip
    X[:, 0] = (df["Use Chip"]=="Online Transaction").values.astype(np.float32)
    X[:, 1] = (df["Use Chip"]=="Swipe Transaction").values.astype(np.float32)
    X[:, 2] = (df["Use Chip"]=="Chip Transaction").values.astype(np.float32)
    # 3: amount
    X[:, 3] = amt
    # 4-5: amount ratios
    card_med = m["med"].fillna(1.0).values
    card_mean = m["mean"].fillna(1.0).values
    X[:, 4] = np.where(card_med > 0, amt / card_med, 0.0)
    X[:, 5] = np.where(card_mean > 0, amt / card_mean, 0.0)
    # 6: log amount
    X[:, 6] = np.log1p(np.abs(amt))
    # 7: z-score
    card_std = m["std"].fillna(1.0).values
    X[:, 7] = np.where(card_std > 0, (amt - card_mean) / card_std, 0.0)
    # 8-9: card history
    X[:, 8] = m["cnt"].fillna(0).values
    X[:, 9] = m["cnt"].fillna(0).values / 24.0
    # 10: tenure
    X[:, 10] = (ux - m["mu"].fillna(ux.min()).values) / 86400
    # 11-12: merchant/city diversity
    X[:, 11] = m["nm"].fillna(0).values
    X[:, 12] = m["nc"].fillna(0).values
    # 13: shared accounts
    X[:, 13] = mm["n_users"].fillna(1).values
    # 14-15: online interactions
    is_online = X[:, 0]
    X[:, 14] = is_online * amt
    X[:, 15] = is_online * X[:, 4]
    # 16-18: geographic
    X[:, 16] = (~df["Merchant State"].isin(US_STATES)).astype(np.float32)
    X[:, 17] = df["Merchant City"].isin(HIGH_RISK_CITIES).astype(np.float32)
    X[:, 18] = X[:, 16] * is_online
    # 19-20: MCC
    X[:, 19] = mcc_rate_arr
    X[:, 20] = df["MCC"].isin(HIGH_RISK_MCCS).astype(np.float32)
    # 21-24: errors
    err = df["Errors?"].fillna("").values
    X[:, 21] = (err != "").astype(np.float32)
    X[:, 22] = np.array(["Bad PIN" in e for e in err], dtype=np.float32)
    X[:, 23] = np.array(["Bad CVV" in e for e in err], dtype=np.float32)
    X[:, 24] = np.array(["Insufficient Balance" in e for e in err], dtype=np.float32)
    # 25-26: time
    X[:, 25] = hr / 23.0
    X[:, 26] = (dow >= 5).astype(np.float32)
    # 27: recipient novelty
    merch_cc = df.groupby(["uc", "Merchant Name"]).cumcount().values + 1
    X[:, 27] = 1.0 / merch_cc
    # 28: days since last
    si = np.argsort(ux, kind="mergesort")
    inv = np.empty_like(si); inv[si] = np.arange(n)
    su = ux[si]
    gaps = np.diff(su, prepend=su[0]-86400)
    X[:, 28] = np.clip(gaps[inv]/86400.0, 0, 365)
    # 29-32: amount bins
    X[:, 29] = (amt < 10).astype(np.float32)
    X[:, 30] = ((amt >= 10) & (amt < 50)).astype(np.float32)
    X[:, 31] = ((amt >= 50) & (amt < 200)).astype(np.float32)
    X[:, 32] = (amt >= 200).astype(np.float32)

    print(f"  {X.shape[1]} features in {time.time()-t1:.0f}s")

    # Split
    split = int(n * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    print(f"\nTrain: {split:,} ({int(ytr.sum()):,} fraud)")
    print(f"Test:  {n-split:,} ({int(yte.sum()):,} fraud)")

    # Train
    print("\nTraining...")
    t3 = time.time()
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xc_tr = np.nan_to_num(imp.transform(Xtr))
    Xc_te = np.nan_to_num(imp.transform(Xte))
    sc = StandardScaler().fit(Xc_tr)
    Xs_tr, Xs_te = sc.transform(Xc_tr), sc.transform(Xc_te)

    np_ = int(ytr.sum()); nn = split - np_
    model = XGBClassifier(n_estimators=600, max_depth=10, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=max(nn/max(np_,1),1),
        eval_metric="auc", random_state=SEED, n_jobs=-1, verbosity=0,
        min_child_weight=3, gamma=0.05, reg_alpha=0.1)
    model.fit(Xs_tr, ytr)
    print(f"  Trained in {time.time()-t3:.0f}s")

    prob = model.predict_proba(Xs_te)[:, 1]
    mets = compute_metrics(yte, prob)

    # Feature importance
    imp_vals = model.feature_importances_
    top_idx = np.argsort(imp_vals)[::-1][:15]

    print("\n" + "=" * 70)
    print("RESULTS: ealtman2019 IMPROVED")
    print("=" * 70)
    for k in ["roc_auc","pr_auc","recall_at_0_01pct_fpr","recall_at_0_1pct_fpr",
              "recall_at_0_5pct_fpr","recall_at_1pct_fpr","brier"]:
        print(f"  {k}: {mets[k]}")
    if mets.get("ci"):
        print(f"  CI ROC-AUC: {mets['ci'].get('roc_auc')}")
        print(f"  CI PR-AUC:  {mets['ci'].get('pr_auc')}")
        print(f"  CI R@1%:    {mets['ci'].get('recall_1pct')}")
    print(f"\nTop features:")
    for i, idx in enumerate(top_idx):
        fname = NAMES[idx] if idx < len(NAMES) else f"feat[{idx}]"
        print(f"  {i+1:>2}. {fname:<25} importance={imp_vals[idx]:.4f}")
    print(f"\nTotal: {time.time()-T0:.0f}s")

    result = {"experiment_id": "REAL_PUBLIC_ealtman_improved_v1", "category": "REAL_PUBLIC",
              "dataset": "ealtman2019", "data_type": "REAL_PUBLIC_DATASET",
              "n_total": n, "n_fraud": int(y.sum()), "n_train": split, "n_test": n-split,
              "n_features": X.shape[1], **mets}
    out = ROOT / "benchmarks"; out.mkdir(exist_ok=True)
    (out / "ealtman_improved_results.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out / 'ealtman_improved_results.json'}")

if __name__ == "__main__":
    main()
