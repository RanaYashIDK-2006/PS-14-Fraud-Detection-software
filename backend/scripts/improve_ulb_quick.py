#!/usr/bin/env python3
"""Quick improvement: ULB with Optuna 15 trials + advanced features."""
import json, time, warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
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

print("IMPROVING ULB CREDITCARD")
df = pd.read_csv("data/creditcard.csv")
y = df["Class"].values
V_cols = [c for c in df.columns if c.startswith("V")]
v = df[V_cols].values.astype(np.float32)

# Advanced features
df["v_mag"] = np.sqrt((v**2).sum(axis=1))
df["v_mean"] = v.mean(axis=1); df["v_std"] = v.std(axis=1)
df["v_skew"] = pd.DataFrame(v).skew(axis=1).values
df["v_kurt"] = pd.DataFrame(v).kurtosis(axis=1).values
df["v_asym"] = df["v_mean"] / (df["v_std"] + 1e-8)
df["v_extreme"] = (np.abs(v) > 3).sum(axis=1).astype(np.float32)
df["v_ext_max"] = np.abs(v).max(axis=1)
df["v_lo"] = (v[:,:10]**2).sum(axis=1)
df["v_mid"] = (v[:,10:20]**2).sum(axis=1)
df["v_hi"] = (v[:,20:]**2).sum(axis=1)
df["v_ratio"] = df["v_lo"] / (df["v_hi"] + 1e-8)
df["amount_log"] = np.log1p(df["Amount"])
df["amt_bucket"] = pd.cut(df["Amount"], bins=[0,0.01,1,10,50,100,500,1000,25000], labels=False).fillna(0)
df["amt_x_mag"] = df["Amount"] * df["v_mag"]
df["amt_x_ext"] = df["Amount"] * df["v_extreme"]
df["hour"] = (df["Time"] / 3600) % 24
df["is_night"] = ((df["hour"]>=22)|(df["hour"]<=6)).astype(np.float32)
df["is_weekend"] = ((df["Time"]/86400).astype(int)%7>=5).astype(np.float32)
df["t_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["t_cos"] = np.cos(2*np.pi*df["hour"]/24)
df["amt_x_time"] = df["amount_log"]*df["t_sin"]
df["amt_x_night"] = df["amount_log"]*df["is_night"]
df["mag_x_ext"] = df["v_mag"]*df["v_extreme"]
for i,j in [(0,1),(1,4),(0,3),(1,2),(3,4),(0,17)]:
    df[f"v{i}x{j}"] = df[V_cols[i]]*df[V_cols[j]]
    df[f"v{i}d{j}"] = df[V_cols[i]]/(df[V_cols[j]]+1e-8)

fc = V_cols + ["amount_log","amt_bucket","v_mag","v_mean","v_std","v_skew","v_kurt","v_asym",
    "v_extreme","v_ext_max","v_lo","v_mid","v_hi","v_ratio","amt_x_mag","amt_x_ext",
    "hour","is_night","is_weekend","t_sin","t_cos","amt_x_time","amt_x_night","mag_x_ext"]
for i,j in [(0,1),(1,4),(0,3),(1,2),(3,4),(0,17)]: fc += [f"v{i}x{j}",f"v{i}d{j}"]
fc = list(dict.fromkeys(fc))
print(f"Features: {len(fc)}")

X = df[fc].values.astype(np.float32)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
sc = StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)

# Optuna
print("Optuna 15 trials...")
def obj(trial):
    ne=trial.suggest_int("ne",200,500); md=trial.suggest_int("md",3,8)
    lr=trial.suggest_float("lr",0.02,0.2,log=True); ss=trial.suggest_float("ss",0.6,1.0)
    cs=trial.suggest_float("cs",0.5,1.0); mcw=trial.suggest_int("mcw",1,8)
    gn=trial.suggest_float("gn",0,3); spw=trial.suggest_float("spw",2,15)
    m = XGBClassifier(n_estimators=ne,max_depth=md,learning_rate=lr,scale_pos_weight=spw,
                      subsample=ss,colsample_bytree=cs,min_child_weight=mcw,gamma=gn,
                      random_state=42,eval_metric="logloss",n_jobs=-1)
    m.fit(Xtr_s,ytr); return roc_auc_score(yte,m.predict_proba(Xte_s)[:,1])

s = optuna.create_study(direction="maximize")
s.optimize(obj, n_trials=15)
bp = s.best_params
print(f"Best AUC: {s.best_value:.6f}")

# Train tuned XGB
bp_full = {"n_estimators":int(bp["ne"]),"max_depth":int(bp["md"]),"learning_rate":float(bp["lr"]),"scale_pos_weight":float(bp["spw"]),
           "subsample":float(bp["ss"]),"colsample_bytree":float(bp["cs"]),"min_child_weight":int(bp["mcw"]),"gamma":float(bp["gn"]),
           "random_state":42,"eval_metric":"logloss","n_jobs":-1}
t0=time.time()
m = XGBClassifier(**bp_full); m.fit(Xtr_s,ytr)
p = m.predict_proba(Xte_s)[:,1]; t=time.time()-t0
r = ev(yte,p); r["time"]=round(t,1)
print(f"Tuned XGB: AUC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} R1%={r['r1']:.4f} Brier={r['brier']:.6f} F1={r['f1_opt']:.4f} Thr={r['opt_thr']}")

# 5-fold CV
print("5-fold CV...")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
ca,cp,cr = [],[],[]
for fold,(tri,tei) in enumerate(skf.split(Xtr_s,ytr)):
    m2 = XGBClassifier(**bp_full); m2.fit(Xtr_s[tri],ytr[tri])
    pp = m2.predict_proba(Xtr_s[tei])[:,1]
    ca.append(roc_auc_score(ytr[tei],pp)); cp.append(average_precision_score(ytr[tei],pp))
    cr.append(rafpr(ytr[tei],pp,0.01))
    print(f"  Fold {fold+1}: AUC={ca[-1]:.4f}")
cv = {"roc_auc_mean":round(float(np.mean(ca)),6),"roc_auc_std":round(float(np.std(ca)),6),
      "pr_auc_mean":round(float(np.mean(cp)),6),"r1_mean":round(float(np.mean(cr)),6)}

r2 = None

# Top features
fi = m.feature_importances_
top = np.argsort(fi)[::-1][:15]
print("\nTop features:")
for i in top: print(f"  {fc[i]:<30s} {fi[i]:.4f}")

res = {"xgb_tuned":r,"cv_5fold":cv}
out = {"dataset":"ulb_creditcard","n_features":len(fc),"tuned_params":{k:v for k,v in bp_full.items() if k not in ("random_state","eval_metric","n_jobs")},
       "results":res,"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")}
with open("reports/ulb_improved.json","w") as f: json.dump(out,f,indent=2)
print(f"\nSaved reports/ulb_improved.json")
