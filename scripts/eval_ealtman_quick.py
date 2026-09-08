#!/usr/bin/env python3
"""Evaluate on ealtman2019 (2M sample of 24M real credit card transactions)."""
import sys, time, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SEED = 42

print("=" * 70)
print("PS-14 EVALUATION: ealtman2019/credit-card-transactions (2M sample)")
print("=" * 70)

t0 = time.time()
print("Loading 2M rows...")
df = pd.read_csv(ROOT / "data" / "credit_card_transactions-ibm_v2.csv", nrows=2_000_000, dtype=str)
df["Amount_num"] = df["Amount"].str.replace("$", "").str.replace(",", "").astype(float)
df["Hour"] = df["Time"].str.split(":").str[0].astype(int)
df["user_card"] = df["User"] + "_" + df["Card"]
df["unix"] = pd.to_datetime(
    df["Year"] + "-" + df["Month"].str.zfill(2) + "-" + df["Day"].str.zfill(2) + " " + df["Time"]
).astype(np.int64) // 10**9
y = (df["Is Fraud?"] == "Yes").values.astype(int)
n = len(df)
print(f"  Loaded {n:,} rows ({int(y.sum()):,} fraud) in {time.time()-t0:.1f}s")

# ── Features ──────────────────────────────────────────────────────────────
t1 = time.time()
print("Engineering features...")
X = np.zeros((n, 21), dtype=np.float32)
amt = df["Amount_num"].values
hours = df["Hour"].values.astype(np.float32)

uc = df.groupby("user_card").agg(
    med=("Amount_num", "median"), mean=("Amount_num", "mean"),
    cnt=("Amount_num", "count"), std=("Amount_num", "std"),
).reset_index()
uc["std"] = uc["std"].fillna(1.0)
m = df[["user_card"]].merge(uc, on="user_card", how="left")

X[:, 0] = np.where(m["med"] > 0, amt / m["med"], 0.0)
X[:, 1] = m["cnt"].values.astype(np.float32)
X[:, 2] = hours / 23.0
X[:, 3] = (df.groupby(["user_card", "Use Chip"]).cumcount() == 0).values.astype(np.float32)
X[:, 5] = (df.groupby(["user_card", "Merchant Name"]).cumcount() == 0).values.astype(np.float32)
X[:, 6] = (df["Errors?"].fillna("") != "").astype(np.float32)

si = np.argsort(df["unix"].values, kind="mergesort")
inv = np.empty_like(si)
inv[si] = np.arange(n)
su = df["unix"].values[si]
gs = np.diff(su, prepend=su[0] - 86400)
gaps = gs[inv] / 86400.0
X[:, 7] = np.clip(gaps, 0, 365).astype(np.float32)
X[:, 8] = np.where(m["mean"] > 0, amt / m["mean"], 0.0).astype(np.float32)

mc = df.groupby("user_card")["Merchant Name"].nunique().reset_index()
mc.columns = ["user_card", "nm"]
mm = df[["user_card"]].merge(mc, on="user_card", how="left")
X[:, 9] = mm["nm"].values.astype(np.float32)

fu = df.groupby("user_card")["unix"].min().reset_index()
fu.columns = ["user_card", "fu"]
fm = df[["user_card"]].merge(fu, on="user_card", how="left")
X[:, 10] = ((df["unix"].values - fm["fu"].values) / 86400).astype(np.float32)
X[:, 11] = hours
dow = pd.to_datetime(
    df["Year"] + "-" + df["Month"].str.zfill(2) + "-" + df["Day"].str.zfill(2)
).dt.dayofweek.values
X[:, 12] = (dow >= 5).astype(np.float32)

mu = df.groupby("Merchant Name")["User"].nunique().reset_index()
mu.columns = ["Merchant Name", "nu"]
mum = df[["Merchant Name"]].merge(mu, on="Merchant Name", how="left")
X[:, 13] = mum["nu"].values.astype(np.float32)
X[:, 14] = mum["nu"].values.astype(np.float32)

cc = df.groupby("user_card")["Merchant City"].nunique().reset_index()
cc.columns = ["user_card", "nc"]
ccm = df[["user_card"]].merge(cc, on="user_card", how="left")
X[:, 15] = ccm["nc"].values.astype(np.float32)

mh = df.groupby("user_card")["Hour"].median().reset_index()
mh.columns = ["user_card", "mh"]
mhm = df[["user_card"]].merge(mh, on="user_card", how="left")
X[:, 16] = np.abs(hours - mhm["mh"].values).astype(np.float32)
X[:, 17] = np.where(m["std"] > 0, (amt - m["mean"]) / m["std"], 0.0).astype(np.float32)
X[:, 18] = m["cnt"].values.astype(np.float32) / 24.0

mf = df.groupby(["user_card", "Merchant Name"]).cumcount() + 1
X[:, 19] = (1.0 / mf.values).astype(np.float32)

gdf = pd.DataFrame({"uc": df["user_card"].values, "gap": gaps})
gss = gdf.groupby("uc")["gap"].std().reset_index()
gss.columns = ["user_card", "gs"]
gsm = df[["user_card"]].merge(gss, on="user_card", how="left")
X[:, 20] = np.clip(gsm["gs"].fillna(0).values, 0, 100).astype(np.float32)

print(f"  Features in {time.time()-t1:.1f}s")

# ── Split + Train ─────────────────────────────────────────────────────────
split = int(n * 0.8)
Xtr, Xte = X[:split], X[split:]
ytr, yte = y[:split], y[split:]

imp = SimpleImputer(strategy="median").fit(Xtr)
Xc_tr = np.nan_to_num(imp.transform(Xtr))
Xc_te = np.nan_to_num(imp.transform(Xte))
sc = StandardScaler().fit(Xc_tr)
Xs_tr, Xs_te = sc.transform(Xc_tr), sc.transform(Xc_te)

print("Training XGBoost...")
t2 = time.time()
np_ = int(ytr.sum())
nn = len(ytr) - np_
model = XGBClassifier(
    n_estimators=600, max_depth=10, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=max(nn / max(np_, 1), 1.0),
    eval_metric="auc", random_state=SEED, n_jobs=-1, verbosity=0,
    min_child_weight=3, gamma=0.05, reg_alpha=0.1,
)
model.fit(Xs_tr, ytr)
print(f"  Trained in {time.time()-t2:.1f}s")

prob = model.predict_proba(Xs_te)[:, 1]

# ── Metrics ───────────────────────────────────────────────────────────────
roc = roc_auc_score(yte, prob)
pr = average_precision_score(yte, prob)
br = brier_score_loss(yte, prob)
fpr_a, tpr_a, _ = roc_curve(yte, prob)

r_at = {}
for tf, l in [(0.001, "0.1"), (0.005, "0.5"), (0.01, "1")]:
    mask = fpr_a <= tf
    r_at[l] = round(float(tpr_a[np.where(mask)[0][-1]]), 4) if mask.any() else 0.0

print("\n" + "=" * 70)
print("RESULTS: ealtman2019/credit-card-transactions")
print("=" * 70)
print(f"  Dataset: 2,000,000 transactions, {int(y.sum()):,} fraud ({y.sum()/n*100:.4f}%)")
print(f"  Train: {split:,} | Test: {n-split:,}")
print(f"  ROC-AUC:   {roc:.4f}")
print(f"  PR-AUC:    {pr:.4f}")
print(f"  R@0.1%FPR: {r_at['0.1']}")
print(f"  R@0.5%FPR: {r_at['0.5']}")
print(f"  R@1%FPR:   {r_at['1']}")
print(f"  Brier:     {br:.6f}")
print(f"  Time:      {time.time()-t0:.1f}s")

res = {
    "experiment_id": "REAL_PUBLIC_ealtman_v1",
    "category": "REAL_PUBLIC",
    "dataset": "ealtman2019_credit_card_transactions",
    "data_type": "REAL_PUBLIC_DATASET",
    "n": n, "n_fraud": int(y.sum()), "prev": round(float(y.sum()/n), 6),
    "train_n": split, "test_n": n - split,
    "roc_auc": round(float(roc), 4), "pr_auc": round(float(pr), 4),
    "recall_at_0_1pct_fpr": r_at["0.1"], "recall_at_0_5pct_fpr": r_at["0.5"],
    "recall_at_1pct_fpr": r_at["1"], "brier": round(float(br), 6),
}
(ROOT / "benchmarks" / "ealtman_results.json").write_text(json.dumps(res, indent=2))
print(f"\nSaved to benchmarks/ealtman_results.json")
