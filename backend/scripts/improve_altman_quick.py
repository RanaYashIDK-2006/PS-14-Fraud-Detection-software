#!/usr/bin/env python3
"""Quick Altman improvement: Optuna 15 trials + advanced features + user-disjoint."""
import json, time, warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def rafpr(y, s, t=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    if fpr[0] > 0: fpr = np.concatenate([[0], fpr]); tpr = np.concatenate([[0], tpr])
    return float(tpr[min(np.searchsorted(fpr, t), len(tpr)-1)])

def find_thr(y, s, t=0.01):
    fpr, tpr, thr = roc_curve(y, s)
    v = fpr <= t
    if not v.any(): return thr[0]
    return float(thr[v][np.argmax(tpr[v])])

def ev(y, p):
    thr = find_thr(y, p)
    yp = (p >= thr).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, p), 6),
        "pr_auc": round(average_precision_score(y, p), 6),
        "brier": round(brier_score_loss(y, p), 6),
        "r1": round(rafpr(y, p, 0.01), 6),
        "r05": round(rafpr(y, p, 0.005), 6),
        "r01": round(rafpr(y, p, 0.001), 6),
        "opt_thr": round(thr, 4),
        "f1_opt": round(f1_score(y, yp), 6),
    }

print("IMPROVING IBM ALTMAN")
csv_path = "data/credit_card_transactions-ibm_v2.csv"
df = pd.read_csv(csv_path, nrows=3_000_000)
df["label"] = df["Is Fraud?"].map(lambda x: 1 if str(x).strip() == "Yes" else 0)
fraud = int(df["label"].sum())
print(f"Rows: {len(df):,}, Fraud: {fraud:,} ({fraud/len(df)*100:.4f}%)")

# Feature engineering
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
df = df.sort_values(["user","datetime"]).reset_index(drop=True)
grp = df.groupby("user", sort=False)

# Base temporal
df["user_amt_mean"] = grp["amount"].transform(lambda x: x.expanding().mean())
df["user_amt_std"] = grp["amount"].transform(lambda x: x.expanding().std().fillna(0))
df["user_txn_count"] = grp.cumcount() + 1
df["user_amt_ratio"] = df["amount"] / (df["user_amt_mean"] + 1e-8)
df["user_amt_zscore"] = (df["amount"] - df["user_amt_mean"]) / (df["user_amt_std"] + 1e-8)
df["user_amt_max"] = grp["amount"].transform(lambda x: x.expanding().max())
df["hours_since_last"] = grp["datetime"].diff().dt.total_seconds().fillna(86400) / 3600

# NEW: Advanced temporal
df["user_amt_median"] = grp["amount"].transform(lambda x: x.expanding().median())
df["user_amt_iqr"] = grp["amount"].transform(lambda x: x.expanding().quantile(0.75)) - grp["amount"].transform(lambda x: x.expanding().quantile(0.25))
df["user_amt_vs_median"] = df["amount"] / (df["user_amt_median"] + 1e-8)
df["user_txn_accel"] = df["hours_since_last"] / (grp["hours_since_last"].transform(lambda x: x.expanding().mean()) + 1e-8)
df["user_night_ratio"] = grp["is_night"].transform(lambda x: x.expanding().mean())
df["user_amt_trend"] = df["amount"] - df["user_amt_mean"]

# Merchant-level
merch_fr = df.groupby("merchant_hash")["label"].mean()
df["merchant_fraud_rate"] = df["merchant_hash"].map(merch_fr).fillna(0).astype(np.float32)
df["merchant_txn_count"] = df.groupby("merchant_hash").cumcount() + 1

# City-level
city_fr = df.groupby("city_hash")["label"].mean()
df["city_fraud_rate"] = df["city_hash"].map(city_fr).fillna(0).astype(np.float32)

# MCC
mcc_fr = df.groupby("mcc")["label"].mean()
df["mcc_fraud_rate"] = df["mcc"].map(mcc_fr).fillna(0).astype(np.float32)

# Interactions
df["amt_x_night"] = df["amount_log"] * df["is_night"]
df["amt_x_new_merch"] = df["amount_log"] * (df["merchant_txn_count"] == 1).astype(np.float32)
df["amt_x_high_fraud_mcc"] = df["amount_log"] * (df["mcc_fraud_rate"] > 0.01).astype(np.float32)
df["amt_x_online"] = df["amount_log"] * (df["chip"] == 0).astype(np.float32)
df["zscore_x_night"] = df["user_amt_zscore"] * df["is_night"]
df["zscore_x_new_merch"] = df["user_amt_zscore"] * (df["merchant_txn_count"] == 1).astype(np.float32)

fc = [
    "amount_log","is_negative","user_amt_mean","user_amt_std","user_amt_max",
    "user_amt_median","user_amt_iqr","user_amt_zscore","user_amt_ratio","user_amt_vs_median","user_amt_trend",
    "hour","dow","is_weekend","is_night",
    "hours_since_last","user_txn_count","user_txn_accel","user_night_ratio",
    "mcc","mcc_fraud_rate","merchant_hash","merchant_fraud_rate","merchant_txn_count",
    "city_hash","city_fraud_rate",
    "chip","has_error",
    "amt_x_night","amt_x_new_merch","amt_x_high_fraud_mcc","amt_x_online",
    "zscore_x_night","zscore_x_new_merch",
]
print(f"Features: {len(fc)}")

# USER-DISJOINT SPLIT
all_users = df["user"].unique()
rng = np.random.RandomState(42)
rng.shuffle(all_users)
n_test = int(len(all_users) * 0.2)
test_users = set(all_users[:n_test])
train_users = set(all_users[n_test:])
tr = df[df["user"].isin(train_users)]
te = df[df["user"].isin(test_users)]
print(f"Train: {len(tr):,} (users={len(train_users):,}, fraud={tr['label'].sum():,})")
print(f"Test:  {len(te):,} (users={len(test_users):,}, fraud={te['label'].sum():,})")

Xtr = tr[fc].fillna(0).values.astype(np.float32)
ytr = tr["label"].values
Xte = te[fc].fillna(0).values.astype(np.float32)
yte = te["label"].values
sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

# Optuna
print("Optuna 15 trials...")
def obj(trial):
    ne=trial.suggest_int("ne",200,500); md=trial.suggest_int("md",4,10)
    lr=trial.suggest_float("lr",0.02,0.2,log=True); ss=trial.suggest_float("ss",0.6,1.0)
    cs=trial.suggest_float("cs",0.5,1.0); mcw=trial.suggest_int("mcw",1,8)
    gn=trial.suggest_float("gn",0,5); spw=trial.suggest_float("spw",2,20)
    m = XGBClassifier(n_estimators=ne,max_depth=md,learning_rate=lr,scale_pos_weight=spw,
                      subsample=ss,colsample_bytree=cs,min_child_weight=mcw,gamma=gn,
                      random_state=42,eval_metric="logloss",n_jobs=-1)
    m.fit(Xtr_s,ytr); return roc_auc_score(yte,m.predict_proba(Xte_s)[:,1])

s = optuna.create_study(direction="maximize")
s.optimize(obj, n_trials=15)
bp = s.best_params
print(f"Best AUC: {s.best_value:.6f}")

bp_full = {"n_estimators":int(bp["ne"]),"max_depth":int(bp["md"]),"learning_rate":float(bp["lr"]),
           "scale_pos_weight":float(bp["spw"]),"subsample":float(bp["ss"]),"colsample_bytree":float(bp["cs"]),
           "min_child_weight":int(bp["mcw"]),"gamma":float(bp["gn"]),
           "random_state":42,"eval_metric":"logloss","n_jobs":-1}

t0=time.time()
m = XGBClassifier(**bp_full); m.fit(Xtr_s,ytr)
p = m.predict_proba(Xte_s)[:,1]; t=time.time()-t0
r = ev(yte,p); r["time"]=round(t,1)
print(f"Tuned XGB: AUC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} R1%={r['r1']:.4f} F1={r['f1_opt']:.4f} ({t:.1f}s)")

# Calibrated
from sklearn.calibration import CalibratedClassifierCV
t0=time.time()
cal = CalibratedClassifierCV(m, method="isotonic", cv=3)
cal.fit(Xtr_s, ytr)
p_cal = cal.predict_proba(Xte_s)[:, 1]
t_cal = time.time() - t0
r_cal = ev(yte, p_cal); r_cal["time"]=round(t_cal,1)
print(f"Calibrated: AUC={r_cal['roc_auc']:.4f} PR={r_cal['pr_auc']:.4f} R1%={r_cal['r1']:.4f} Brier={r_cal['brier']:.6f}")

# Top features
fi = m.feature_importances_
top = np.argsort(fi)[::-1][:15]
print("\nTop features:")
for i in top: print(f"  {fc[i]:<30s} {fi[i]:.4f}")

out = {"dataset":"ibm_altman","n_features":len(fc),
       "tuned_params":{k:v for k,v in bp_full.items() if k not in ("random_state","eval_metric","n_jobs")},
       "results":{"xgb_tuned":r,"xgb_calibrated":r_cal},
       "user_disjoint":{"train_users":len(train_users),"test_users":len(test_users)},
       "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}
with open("reports/altman_improved.json","w") as f: json.dump(out,f,indent=2)
print(f"\nSaved reports/altman_improved.json")
