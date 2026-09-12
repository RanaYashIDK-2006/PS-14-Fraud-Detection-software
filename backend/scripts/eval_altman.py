#!/usr/bin/env python3
"""Phase 2: IBM Altman 24M - optimized for speed."""
import hashlib, json, time, warnings, gc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

def file_hash(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]

def recall_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if fpr[0] > 0:
        fpr = np.concatenate([[0], fpr])
        tpr = np.concatenate([[0], tpr])
    idx = np.searchsorted(fpr, target_fpr)
    return float(tpr[min(idx, len(tpr)-1)])

def ev(y, p, prefix=""):
    roc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    return {
        f"{prefix}roc_auc": round(roc, 6),
        f"{prefix}pr_auc": round(pr, 6),
        f"{prefix}brier": round(brier_score_loss(y, p), 6),
        f"{prefix}r1": round(recall_at_fpr(y, p, 0.01), 6),
        f"{prefix}r05": round(recall_at_fpr(y, p, 0.005), 6),
        f"{prefix}r01": round(recall_at_fpr(y, p, 0.001), 6),
        f"{prefix}n_pos": int(np.sum(y==1)),
        f"{prefix}n_neg": int(np.sum(y==0)),
    }

print("PHASE 2: IBM Altman - Optimized")
csv_path = "data/credit_card_transactions-ibm_v2.csv"
print(f"Hash: {file_hash(csv_path)}")

# Load 3M rows (fast, enough for honest eval)
print("Loading 3M rows...")
df = pd.read_csv(csv_path, nrows=3_000_000)
print(f"Loaded {len(df):,} rows")

df["label"] = df["Is Fraud?"].map(lambda x: 1 if str(x).strip() == "Yes" else 0)
fraud = int(df["label"].sum())
print(f"Fraud: {fraud:,} ({fraud/len(df)*100:.4f}%)")

# Feature engineering
print("Features...")
df["amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace("$","").str.replace(",",""), errors="coerce").fillna(0)
df["amount_log"] = np.log1p(df["amount"])
df["is_negative"] = (df["amount"] < 0).astype(np.float32)
df["hour"] = df["Time"].astype(str).apply(lambda x: int(x.split(":")[0]) if ":" in str(x) else 12)
df["datetime"] = pd.to_datetime(df[["Year","Month","Day"]].rename(columns={"Year":"year","Month":"month","Day":"day"}))
df["dow"] = df["datetime"].dt.dayofweek
df["is_weekend"] = (df["dow"] >= 5).astype(np.float32)
df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(np.float32)
df["user"] = df["User"].astype("category").cat.codes
df["mcc"] = df["MCC"].astype("category").cat.codes
df["merchant_hash"] = df["Merchant Name"].astype("category").cat.codes
df["city_hash"] = df["Merchant City"].astype("category").cat.codes
df["chip"] = df["Use Chip"].map({"Chip":2,"Swipe":1,"Online":0}).fillna(0).astype(np.float32)
df["has_error"] = (df["Errors?"].fillna("") != "").astype(np.float32)

# Sort for expanding features
print("Per-user features...")
df = df.sort_values(["user","datetime"]).reset_index(drop=True)
grp = df.groupby("user", sort=False)
df["user_amt_mean"] = grp["amount"].transform(lambda x: x.expanding().mean())
df["user_amt_std"] = grp["amount"].transform(lambda x: x.expanding().std().fillna(0))
df["user_txn_count"] = grp.cumcount() + 1
df["user_amt_ratio"] = df["amount"] / (df["user_amt_mean"] + 1e-8)
df["user_amt_zscore"] = (df["amount"] - df["user_amt_mean"]) / (df["user_amt_std"] + 1e-8)
df["user_amt_max"] = grp["amount"].transform(lambda x: x.expanding().max())
df["hours_since_last"] = grp["datetime"].diff().dt.total_seconds().fillna(86400) / 3600

mcc_fr = df.groupby("mcc")["label"].mean()
df["mcc_fraud_rate"] = df["mcc"].map(mcc_fr).fillna(0).astype(np.float32)
df["amt_x_night"] = df["amount_log"] * df["is_night"]
df["amt_x_new_merch"] = df["amount_log"] * (1 - df["chip"]/2)

feat_cols = [
    "amount_log","is_negative","user_amt_mean","user_amt_std","user_amt_max",
    "user_amt_zscore","user_amt_ratio","hour","dow","is_weekend","is_night",
    "hours_since_last","user_txn_count","mcc","mcc_fraud_rate",
    "merchant_hash","city_hash","chip","has_error","amt_x_night","amt_x_new_merch",
]
print(f"Features: {len(feat_cols)}")

results = {}

# USER-DISJOINT SPLIT (HONEST)
print("\n--- User-Disjoint (Honest) ---")
all_users = df["user"].unique()
rng = np.random.RandomState(42)
rng.shuffle(all_users)
n_test = int(len(all_users) * 0.2)
test_users = set(all_users[:n_test])
train_users = set(all_users[n_test:])

tr = df[df["user"].isin(train_users)]
te = df[df["user"].isin(test_users)]
print(f"Train: {len(tr):,} users={len(train_users):,} fraud={tr['label'].sum():,}")
print(f"Test:  {len(te):,} users={len(test_users):,} fraud={te['label'].sum():,}")

Xtr = tr[feat_cols].fillna(0).values.astype(np.float32)
ytr = tr["label"].values
Xte = te[feat_cols].fillna(0).values.astype(np.float32)
yte = te["label"].values
sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

# LR
t0=time.time(); m=LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)
m.fit(Xtr_s, ytr); p=m.predict_proba(Xte_s)[:,1]; t=time.time()-t0
results["disjoint_lr"] = ev(yte, p, "d_")
results["disjoint_lr"]["time"] = round(t,1)
print(f"  LR: AUC={results['disjoint_lr']['d_roc_auc']:.4f} R1%={results['disjoint_lr']['d_r1']:.4f} ({t:.1f}s)")

# XGB
t0=time.time()
m=XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", n_jobs=-1)
m.fit(Xtr_s, ytr); p=m.predict_proba(Xte_s)[:,1]; t=time.time()-t0
results["disjoint_xgb"] = ev(yte, p, "d_")
results["disjoint_xgb"]["time"] = round(t,1)
print(f"  XGB: AUC={results['disjoint_xgb']['d_roc_auc']:.4f} R1%={results['disjoint_xgb']['d_r1']:.4f} ({t:.1f}s)")
gc.collect()

# TIME-BASED SPLIT (OPTIMISTIC)
print("\n--- Time-Based (Optimistic, User Overlap) ---")
df_sorted = df.sort_values("datetime").reset_index(drop=True)
si = int(len(df_sorted) * 0.8)
tr_t = df_sorted.iloc[:si]; te_t = df_sorted.iloc[si:]
tr_users_t = set(tr_t["user"].unique()); te_users_t = set(te_t["user"].unique())
overlap = tr_users_t & te_users_t
print(f"User overlap: {len(overlap)}/{len(te_users_t)} ({len(overlap)/max(len(te_users_t),1)*100:.1f}%)")

Xtr_t = tr_t[feat_cols].fillna(0).values.astype(np.float32); ytr_t = tr_t["label"].values
Xte_t = te_t[feat_cols].fillna(0).values.astype(np.float32); yte_t = te_t["label"].values
sc_t = StandardScaler(); Xtr_t_s = sc_t.fit_transform(Xtr_t); Xte_t_s = sc_t.transform(Xte_t)

t0=time.time(); m=LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)
m.fit(Xtr_t_s, ytr_t); p=m.predict_proba(Xte_t_s)[:,1]; t=time.time()-t0
results["time_lr"] = ev(yte_t, p, "t_")
results["time_lr"]["time"] = round(t,1)
print(f"  LR: AUC={results['time_lr']['t_roc_auc']:.4f} R1%={results['time_lr']['t_r1']:.4f} ({t:.1f}s)")

t0=time.time()
m=XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", n_jobs=-1)
m.fit(Xtr_t_s, ytr_t); p=m.predict_proba(Xte_t_s)[:,1]; t=time.time()-t0
results["time_xgb"] = ev(yte_t, p, "t_")
results["time_xgb"]["time"] = round(t,1)
print(f"  XGB: AUC={results['time_xgb']['t_roc_auc']:.4f} R1%={results['time_xgb']['t_r1']:.4f} ({t:.1f}s)")

out = {
    "dataset": "ibm_altman", "sampled": len(df), "total": 24386900,
    "fraud": fraud, "fraud_rate": round(fraud/len(df),6), "hash": file_hash(csv_path),
    "n_features": len(feat_cols),
    "user_disjoint": {"train_users": len(train_users), "test_users": len(test_users)},
    "time_overlap": {"overlap_pct": round(len(overlap)/max(len(te_users_t),1)*100,1)},
    "results": results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open("reports/altman_results_v2.json","w") as f: json.dump(out, f, indent=2)
print(f"\nSaved to reports/altman_results_v2.json")
