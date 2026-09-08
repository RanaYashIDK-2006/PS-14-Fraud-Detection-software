#!/usr/bin/env python3
"""Phase 3: PaySim 1M evaluation."""
import hashlib, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
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

def ev(y, p):
    return {
        "roc_auc": round(roc_auc_score(y, p), 6),
        "pr_auc": round(average_precision_score(y, p), 6),
        "brier": round(brier_score_loss(y, p), 6),
        "r1": round(recall_at_fpr(y, p, 0.01), 6),
        "r05": round(recall_at_fpr(y, p, 0.005), 6),
        "r01": round(recall_at_fpr(y, p, 0.001), 6),
        "n_pos": int(np.sum(y==1)),
        "n_neg": int(np.sum(y==0)),
    }

print("PHASE 3: PaySim 1M")
df = pd.read_csv("data/paysim_1m.csv")
print(f"Hash: {file_hash('data/paysim_1m.csv')}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Find label
label_col = None
for c in ["isFraud","is_fraud","fraud","Class","label"]:
    if c in df.columns:
        label_col = c; break
print(f"Label: {label_col}")
df["label"] = df[label_col].astype(int)
print(f"Fraud: {df['label'].sum():,} ({df['label'].mean()*100:.4f}%)")

# Feature engineering
df["amount_log"] = np.log1p(df["amount"].fillna(0))

# Balance features
for col in ["oldbalanceOrg","oldbalanceDest","newbalanceOrig","newbalanceDest","amount"]:
    if col not in df.columns:
        # Check case variants
        for variant in df.columns:
            if variant.lower() == col.lower():
                df[col] = df[variant]
                break

df["oldbalance"] = df.get("oldbalanceOrg", pd.Series(0, index=df.index)).fillna(0)
df["newbalance"] = df.get("newbalanceOrig", pd.Series(0, index=df.index)).fillna(0)
df["dest_old"] = df.get("oldbalanceDest", pd.Series(0, index=df.index)).fillna(0)
df["dest_new"] = df.get("newbalanceDest", pd.Series(0, index=df.index)).fillna(0)

# KEY FEATURES for PaySim
df["balance_drain"] = np.where(df["oldbalance"]>0, (df["oldbalance"]-df["newbalance"])/df["oldbalance"], 0).clip(-10,10)
df["amt_vs_balance"] = np.where(df["oldbalance"]>0, df["amount"]/df["oldbalance"], 0).clip(0,100)
df["dest_balance_change"] = df["dest_new"] - df["dest_old"]
df["zero_after"] = (df["newbalance"]==0).astype(np.float32)
df["dest_zero_after"] = (df["dest_new"]==0).astype(np.float32)
df["amt_zero_balance"] = ((df["amount"]>0)&(df["oldbalance"]==0)).astype(np.float32)
df["full_drain"] = ((df["balance_drain"]>=0.99)&(df["oldbalance"]>0)).astype(np.float32)
df["small_txn_suspicious"] = ((df["amount"]>0)&(df["amount"]<1)&(df["full_drain"]==1)).astype(np.float32)

# Type encoding
type_col = None
for c in ["type","Type","transaction_type"]:
    if c in df.columns:
        type_col = c; break

feat_cols = ["amount_log","oldbalance","newbalance","dest_old","dest_new",
             "balance_drain","amt_vs_balance","dest_balance_change",
             "zero_after","dest_zero_after","amt_zero_balance","full_drain","small_txn_suspicious"]

if type_col:
    dummies = pd.get_dummies(df[type_col], prefix="type", drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    feat_cols += list(dummies.columns)

if "step" in df.columns:
    df["step_norm"] = df["step"] % 24
    df["is_night"] = ((df["step_norm"]>=22)|(df["step_norm"]<=6)).astype(np.float32)
    feat_cols += ["step_norm","is_night"]

print(f"Features: {len(feat_cols)}")

y = df["label"].values
X = df[feat_cols].fillna(0).values.astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

results = {}

# LR
t0=time.time(); m=LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)
m.fit(Xtr_s, ytr); p=m.predict_proba(Xte_s)[:,1]; t=time.time()-t0
results["lr"] = ev(yte, p); results["lr"]["time"] = round(t,1)
print(f"  LR:  AUC={results['lr']['roc_auc']:.4f} PR={results['lr']['pr_auc']:.4f} R1%={results['lr']['r1']:.4f} ({t:.1f}s)")

# XGB
t0=time.time()
m=XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", n_jobs=-1)
m.fit(Xtr_s, ytr); p=m.predict_proba(Xte_s)[:,1]; t=time.time()-t0
results["xgb"] = ev(yte, p); results["xgb"]["time"] = round(t,1)
print(f"  XGB: AUC={results['xgb']['roc_auc']:.4f} PR={results['xgb']['pr_auc']:.4f} R1%={results['xgb']['r1']:.4f} ({t:.1f}s)")

out = {
    "dataset": "paysim_1m", "n_rows": len(df), "fraud": int(df["label"].sum()),
    "fraud_rate": round(float(df["label"].mean()),6), "hash": file_hash("data/paysim_1m.csv"),
    "n_features": len(feat_cols), "results": results,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open("reports/paysim_results.json","w") as f: json.dump(out, f, indent=2)
print(f"\nSaved to reports/paysim_results.json")
