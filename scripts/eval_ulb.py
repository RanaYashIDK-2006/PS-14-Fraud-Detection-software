#!/usr/bin/env python3
"""Phase 1: ULB Creditcard exhaustive evaluation."""
import hashlib, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
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

def bootstrap_ci(y_true, y_score, fn, n_boot=200, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2: continue
        vals.append(fn(y_true[idx], y_score[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (0,0)

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

print("PHASE 1: ULB Creditcard")
df = pd.read_csv("data/creditcard.csv")
y = df["Class"].values
print(f"Rows: {len(df):,}, Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

# Features: V1-V28 + Amount + Time
feat_cols = [c for c in df.columns if c != "Class"]
X_raw = df[feat_cols].values.astype(np.float32)

# Pattern features
V_cols = [c for c in df.columns if c.startswith("V")]
v_vals = df[V_cols].values
df["v_mag"] = np.sqrt((v_vals**2).sum(axis=1))
df["v_asym"] = v_vals.mean(axis=1) / (v_vals.std(axis=1) + 1e-8)
df["v_extreme"] = (np.abs(v_vals) > 3).sum(axis=1).astype(np.float32)
df["v_energy_lo"] = (v_vals[:,:10]**2).sum(axis=1)
df["v_energy_hi"] = (v_vals[:,10:20]**2).sum(axis=1)
df["v_energy_ratio"] = df["v_energy_lo"] / (df["v_energy_hi"] + 1e-8)
df["amt_x_mag"] = df["Amount"] * df["v_mag"]
df["amt_x_extreme"] = df["Amount"] * df["v_extreme"]
df["hour"] = (df["Time"] / 3600) % 24
df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(np.float32)

X_pat = df[feat_cols + ["v_mag","v_asym","v_extreme","v_energy_lo","v_energy_hi","v_energy_ratio","amt_x_mag","amt_x_extreme","hour","is_night"]].values.astype(np.float32)

results = {}

# Split
Xtr, Xte, ytr, yte = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)
s1 = StandardScaler(); Xtr_s = s1.fit_transform(Xtr); Xte_s = s1.transform(Xte)

Xtr2, Xte2, _, _ = train_test_split(X_pat, y, test_size=0.2, stratify=y, random_state=42)
s2 = StandardScaler(); Xtr2_s = s2.fit_transform(Xtr2); Xte2_s = s2.transform(Xte2)

# Base models
for name, m in [
    ("lr", LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)),
    ("xgb", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", n_jobs=-1)),
]:
    t0=time.time(); m.fit(Xtr_s, ytr); p=m.predict_proba(Xte_s)[:,1]; t=time.time()-t0
    results[f"base_{name}"] = ev(yte, p)
    results[f"base_{name}"]["time"] = round(t,1)
    print(f"  base_{name}: AUC={results[f'base_{name}']['roc_auc']:.4f} PR={results[f'base_{name}']['pr_auc']:.4f} R1%={results[f'base_{name}']['r1']:.4f} ({t:.1f}s)")

# Pattern models
for name, m_fn in [
    ("lr", lambda: LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)),
    ("rf", lambda: RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)),
    ("xgb", lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", n_jobs=-1)),
]:
    t0=time.time(); m=m_fn(); m.fit(Xtr2_s, ytr); p=m.predict_proba(Xte2_s)[:,1]; t=time.time()-t0
    results[f"pattern_{name}"] = ev(yte, p)
    results[f"pattern_{name}"]["time"] = round(t,1)
    print(f"  pat_{name}: AUC={results[f'pattern_{name}']['roc_auc']:.4f} PR={results[f'pattern_{name}']['pr_auc']:.4f} R1%={results[f'pattern_{name}']['r1']:.4f} ({t:.1f}s)")

# Stacker on pattern features
estimators = [
    ("lr", LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)),
    ("xgb", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", n_jobs=-1)),
]
t0=time.time()
stk = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(C=1, max_iter=1000, random_state=42), cv=3, passthrough=False, n_jobs=-1)
stk.fit(Xtr2_s, ytr); p=stk.predict_proba(Xte2_s)[:,1]; t=time.time()-t0
results["pattern_stacker"] = ev(yte, p)
results["pattern_stacker"]["time"] = round(t,1)
print(f"  pat_stacker: AUC={results['pattern_stacker']['roc_auc']:.4f} PR={results['pattern_stacker']['pr_auc']:.4f} R1%={results['pattern_stacker']['r1']:.4f} ({t:.1f}s)")

# 5-fold CV on XGB pattern
print("\n  5-fold CV (XGB pattern features)...")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
cv_aucs, cv_prs, cv_r1s = [], [], []
for fold, (tr,te) in enumerate(skf.split(Xtr2_s, ytr)):
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", n_jobs=-1)
    m.fit(Xtr2_s[tr], ytr[tr])
    p = m.predict_proba(Xtr2_s[te])[:,1]
    cv_aucs.append(roc_auc_score(ytr[te], p))
    cv_prs.append(average_precision_score(ytr[te], p))
    cv_r1s.append(recall_at_fpr(ytr[te], p, 0.01))
    print(f"    Fold {fold+1}: AUC={cv_aucs[-1]:.4f}")

roc_ci = bootstrap_ci(yte, stk.predict_proba(Xte2_s)[:,1], roc_auc_score)
pr_ci = bootstrap_ci(yte, stk.predict_proba(Xte2_s)[:,1], average_precision_score)

results["cv_5fold_xgb"] = {
    "roc_auc_mean": round(float(np.mean(cv_aucs)),6),
    "roc_auc_std": round(float(np.std(cv_aucs)),6),
    "pr_auc_mean": round(float(np.mean(cv_prs)),6),
    "r1_mean": round(float(np.mean(cv_r1s)),6),
}
results["stacker_ci"] = {
    "roc_auc_95ci": [round(roc_ci[0],6), round(roc_ci[1],6)],
    "pr_auc_95ci": [round(pr_ci[0],6), round(pr_ci[1],6)],
}

out = {"dataset": "ulb_creditcard", "n_rows": len(df), "fraud": int(df["Class"].sum()),
       "fraud_rate": round(float(df["Class"].mean()),6), "hash": file_hash("data/creditcard.csv"),
       "n_feat_base": len(feat_cols), "n_feat_pattern": X_pat.shape[1], "results": results,
       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
with open("reports/ulb_results.json","w") as f: json.dump(out, f, indent=2)
print(f"\nSaved to reports/ulb_results.json")
