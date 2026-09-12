#!/usr/bin/env python3
"""Full ealtman2019 evaluation — pure chunked processing, never loads full CSV.

Uses a two-pass approach that works in constant memory:
  Pass 1: Scan all chunks, build per-group stats in dicts (fast)
  Pass 2: Scan all chunks again, build features via vectorized merges per chunk
"""

import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
SEED = 42
CHUNK = 4_000_000
DATA = ROOT / "data" / "credit_card_transactions-ibm_v2.csv"


def parse_amt(s):
    return s.str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)


def metrics(yt, yp):
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
        rng = np.random.default_rng(SEED)
        ss = min(n, 100000)
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
            "recall_at_0_5pct_fpr": ra.get(".5"), "recall_at_1pct_fpr": ra.get("1"),
            "ci": ci}


def main():
    print("=" * 70)
    print("PS-14 FULL ealtman2019 EVALUATION (all rows, chunked)")
    print("=" * 70)
    T0 = time.time()

    # ── PASS 1: per-group stats ───────────────────────────────────────────
    print("\nPASS 1: Group statistics")
    from collections import defaultdict
    uc_sum = defaultdict(float)
    uc_sum2 = defaultdict(float)
    uc_cnt = defaultdict(int)
    uc_min = defaultdict(lambda: 10**18)
    uc_merch = defaultdict(int)
    uc_city = defaultdict(int)
    uc_hr = defaultdict(list)
    mn_usr = defaultdict(set)
    total = 0; tf = 0

    t1 = time.time()
    for ci, ch in enumerate(pd.read_csv(DATA, chunksize=CHUNK, dtype=str)):
        n = len(ch); total += n; tf += int((ch["Is Fraud?"]=="Yes").sum())
        amt = parse_amt(ch["Amount"]).values
        hr = ch["Time"].str.split(":").str[0].astype(int).values
        ux = pd.to_datetime(ch["Year"]+"-"+ch["Month"].str.zfill(2)+"-"+ch["Day"].str.zfill(2)+" "+ch["Time"]).astype(np.int64).values // 10**9
        uc = (ch["User"]+"_"+ch["Card"]).values
        mn = ch["Merchant Name"].values
        ct = ch["Merchant City"].values

        # Vectorized via pandas aggregation per chunk (much faster than Python loop)
        tmp = pd.DataFrame({"uc": uc, "amt": amt, "ux": ux, "hr": hr, "mn": mn, "ct": ct})
        g = tmp.groupby("uc").agg(
            s=("amt","sum"), s2=("amt", lambda x: (x**2).sum()),
            c=("amt","count"), mu=("ux","min"),
            nm=("mn","nunique"), nc=("ct","nunique"), mh=("hr","median")
        ).reset_index()
        for _, r in g.iterrows():
            u = r["uc"]
            uc_sum[u] += r["s"]
            uc_sum2[u] += r["s2"]
            uc_cnt[u] += int(r["c"])
            if r["mu"] < uc_min[u]: uc_min[u] = int(r["mu"])
            uc_merch[u] += int(r["nm"])  # approximate: per-chunk unique
            uc_city[u] += int(r["nc"])
            uc_hr[u].append(r["mh"])

        # Merchant-user counts
        for mn_val in tmp["mn"].unique():
            users_in_chunk = set(tmp.loc[tmp["mn"]==mn_val, "uc"].str.split("_").str[0])
            mn_usr[mn_val].update(users_in_chunk)

        print(f"  Chunk {ci+1}: {total:,} rows ({tf:,} fraud) [{time.time()-t1:.0f}s]")
        del ch, tmp, g

    # Build stats DF
    print("Building stats...")
    rows = []
    for u in uc_sum:
        c = uc_cnt[u]; mean = uc_sum[u]/c
        std = max((uc_sum2[u]/c - mean**2)**0.5, 0.01)
        rows.append({"uc": u, "med": uc_sum[u]/c, "mean": mean, "std": std,
                      "cnt": c, "mu": uc_min[u], "nm": uc_merch[u],
                      "nc": uc_city[u], "mh": np.median(uc_hr[u]) if uc_hr[u] else 12.0})
    uc_df = pd.DataFrame(rows)
    mn_df = pd.DataFrame([{"Merchant Name": k, "n_users": len(v)} for k, v in mn_usr.items()])
    print(f"  Pass 1 done: {time.time()-t1:.0f}s, {len(uc_df):,} user-cards")

    # ── PASS 2: build features ────────────────────────────────────────────
    print("\nPASS 2: Feature matrix (chunked)")
    t2 = time.time()
    all_X = []; all_y = []; all_ux = []
    total = 0

    for ci, ch in enumerate(pd.read_csv(DATA, chunksize=CHUNK, dtype=str)):
        n = len(ch); total += n
        amt = parse_amt(ch["Amount"]).values
        hr = ch["Time"].str.split(":").str[0].astype(int).values
        ux = pd.to_datetime(ch["Year"]+"-"+ch["Month"].str.zfill(2)+"-"+ch["Day"].str.zfill(2)+" "+ch["Time"]).astype(np.int64).values // 10**9
        uc = (ch["User"]+"_"+ch["Card"]).values
        y = (ch["Is Fraud?"]=="Yes").values.astype(int)
        dow = pd.to_datetime(ch["Year"]+"-"+ch["Month"].str.zfill(2)+"-"+ch["Day"].str.zfill(2)).dt.dayofweek.values
        err = ch["Errors?"].fillna("").ne("").astype(np.float32).values

        tmp = pd.DataFrame({"uc": uc, "mn": ch["Merchant Name"].values})
        m = tmp[["uc"]].merge(uc_df, on="uc", how="left")
        mm = tmp[["mn"]].merge(mn_df, left_on="mn", right_on="Merchant Name", how="left")

        X = np.zeros((n, 21), np.float32)
        X[:, 0] = np.where(m["med"]>0, amt/m["med"], 0.0)
        X[:, 1] = m["cnt"].fillna(1).values
        X[:, 2] = hr/23.0
        X[:, 3] = (m["cnt"].fillna(999)<=3).values
        X[:, 5] = (m["nm"].fillna(999)<=2).values
        X[:, 6] = err
        X[:, 8] = np.where(m["mean"]>0, amt/m["mean"], 0.0)
        X[:, 9] = m["nm"].fillna(1).values
        X[:, 10] = ((ux-m["mu"].fillna(ux))/86400).values
        X[:, 11] = hr.astype(np.float32)
        X[:, 12] = (dow>=5).astype(np.float32)
        X[:, 13] = mm["n_users"].fillna(1).values
        X[:, 14] = mm["n_users"].fillna(1).values
        X[:, 15] = m["nc"].fillna(1).values
        X[:, 16] = np.abs(hr-m["mh"].fillna(12.0).values).astype(np.float32)
        X[:, 17] = np.where(m["std"]>0, (amt-m["mean"])/m["std"], 0.0)
        X[:, 18] = m["cnt"].fillna(1).values/24.0

        mc = tmp.groupby(["uc","mn"]).cumcount().values + 1
        X[:, 19] = (1.0/mc).astype(np.float32)

        all_X.append(X); all_y.append(y); all_ux.append(ux)
        print(f"  Chunk {ci+1}: {total:,} rows [{time.time()-t2:.0f}s]")
        del ch, tmp, m, mm

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    ux = np.concatenate(all_ux)
    del all_X, all_y, all_ux

    # Days since last (argsort)
    print("Computing inter-txn gaps...")
    si = np.argsort(ux, kind="mergesort")
    inv = np.empty_like(si); inv[si] = np.arange(len(si))
    su = ux[si]
    gaps = np.diff(su, prepend=su[0]-86400)
    X[:, 7] = np.clip(gaps[inv]/86400.0, 0, 365).astype(np.float32)

    # Txn regularity (global std as rough proxy — true per-card too expensive without full array)
    X[:, 20] = float(np.std(gaps/86400.0))

    # Sort by time
    si2 = np.argsort(ux, kind="mergesort")
    X = X[si2]; y = y[si2]

    print(f"  Pass 2 done: {time.time()-t2:.0f}s, matrix {X.shape}")

    # ── Train / eval ──────────────────────────────────────────────────────
    n = len(X); split = int(n*0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    print(f"\nTrain: {split:,} ({int(ytr.sum()):,} fraud)")
    print(f"Test:  {n-split:,} ({int(yte.sum()):,} fraud)")

    print("\nPreparing + training...")
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
    mets = metrics(yte, prob)
    tt = time.time() - T0

    print("\n" + "=" * 70)
    print("RESULTS: ealtman2019 (FULL dataset)")
    print("=" * 70)
    for k in ["roc_auc","pr_auc","recall_at_0_01pct_fpr","recall_at_0_1pct_fpr",
              "recall_at_0_5pct_fpr","recall_at_1pct_fpr","brier"]:
        print(f"  {k}: {mets[k]}")
    if mets.get("ci"):
        print(f"  CI ROC-AUC: {mets['ci'].get('roc_auc')}")
        print(f"  CI PR-AUC:  {mets['ci'].get('pr_auc')}")
        print(f"  CI R@1%:    {mets['ci'].get('recall_1pct')}")
    print(f"  Time: {tt:.0f}s")

    result = {"experiment_id": "REAL_PUBLIC_ealtman_full_v1", "category": "REAL_PUBLIC",
              "dataset": "ealtman2019_credit_card_transactions", "data_type": "REAL_PUBLIC_DATASET",
              "n_total": n, "n_fraud": int(y.sum()), "n_train": split, "n_test": n-split,
              **mets, "total_time_seconds": round(tt, 1)}
    out = ROOT / "benchmarks"; out.mkdir(exist_ok=True)
    (out / "ealtman_full_results.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out / 'ealtman_full_results.json'}")

if __name__ == "__main__":
    main()
