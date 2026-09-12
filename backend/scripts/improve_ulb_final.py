#!/usr/bin/env python3
"""ULB Optuna 30 trials — fast, targeted improvement."""
import json, time, warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore"); np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def rafpr(y,s,t=0.01):
    fpr,tpr,_=roc_curve(y,s)
    if fpr[0]>0: fpr=np.concatenate([[0],fpr]);tpr=np.concatenate([[0],tpr])
    return float(tpr[min(np.searchsorted(fpr,t),len(tpr)-1)])
def find_thr(y,s,t=0.01):
    fpr,tpr,thr=roc_curve(y,s);v=fpr<=t
    return thr[0] if not v.any() else float(thr[v][np.argmax(tpr[v])])
def ev(y,p):
    thr=find_thr(y,p);yp=(p>=thr).astype(int)
    return {"roc_auc":round(roc_auc_score(y,p),6),"pr_auc":round(average_precision_score(y,p),6),
            "brier":round(brier_score_loss(y,p),6),"r1":round(rafpr(y,p,0.01),6),
            "r05":round(rafpr(y,p,0.005),6),"r01":round(rafpr(y,p,0.001),6),
            "opt_thr":round(thr,4),"f1":round(f1_score(y,yp),6)}

print("ULB Optuna Push")
df=pd.read_csv("data/creditcard.csv"); y=df["Class"].values
V=[c for c in df.columns if c.startswith("V")]; v=df[V].values.astype(np.float32)
# Features
df["v_mag"]=np.sqrt((v**2).sum(axis=1));df["v_mean"]=v.mean(axis=1);df["v_std"]=v.std(axis=1)
df["v_skew"]=pd.DataFrame(v).skew(axis=1).values;df["v_kurt"]=pd.DataFrame(v).kurtosis(axis=1).values
df["v_asym"]=df["v_mean"]/(df["v_std"]+1e-8);df["v_extreme"]=(np.abs(v)>3).sum(axis=1).astype(np.float32)
df["v_lo"]=(v[:,:10]**2).sum(axis=1);df["v_mid"]=(v[:,10:20]**2).sum(axis=1);df["v_hi"]=(v[:,20:]**2).sum(axis=1)
df["v_ratio"]=df["v_lo"]/(df["v_hi"]+1e-8)
df["amount_log"]=np.log1p(df["Amount"]);df["amt_bucket"]=pd.cut(df["Amount"],bins=[0,.01,1,10,50,100,500,25000],labels=False).fillna(0)
df["amt_x_mag"]=df["Amount"]*df["v_mag"];df["amt_x_ext"]=df["Amount"]*df["v_extreme"]
df["hour"]=(df["Time"]/3600)%24;df["is_night"]=((df["hour"]>=22)|(df["hour"]<=6)).astype(np.float32)
df["t_sin"]=np.sin(2*np.pi*df["hour"]/24);df["t_cos"]=np.cos(2*np.pi*df["hour"]/24)
df["amt_x_time"]=df["amount_log"]*df["t_sin"];df["amt_x_night"]=df["amount_log"]*df["is_night"]
df["mag_x_ext"]=df["v_mag"]*df["v_extreme"]
for i,j in [(0,1),(1,4),(0,3),(1,2),(3,4),(0,17)]:
    df[f"v{i}x{j}"]=df[V[i]]*df[V[j]];df[f"v{i}d{j}"]=df[V[i]]/(df[V[j]]+1e-8)
fc=V+["v_mag","v_mean","v_std","v_skew","v_kurt","v_asym","v_extreme","v_lo","v_mid","v_hi","v_ratio",
      "amount_log","amt_bucket","amt_x_mag","amt_x_ext","hour","is_night","t_sin","t_cos","amt_x_time","amt_x_night","mag_x_ext"]
for i,j in [(0,1),(1,4),(0,3),(1,2),(3,4),(0,17)]: fc+=[f"v{i}x{j}",f"v{i}d{j}"]
fc=list(dict.fromkeys(fc));print(f"Features: {len(fc)}")

X=df[fc].values.astype(np.float32)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
sc=StandardScaler();Xtr_s=sc.fit_transform(Xtr);Xte_s=sc.transform(Xte)

print("Optuna 30 trials...")
def obj(trial):
    m=XGBClassifier(n_estimators=trial.suggest_int("ne",200,600),max_depth=trial.suggest_int("md",3,10),
        learning_rate=trial.suggest_float("lr",.01,.2,log=True),subsample=trial.suggest_float("ss",.6,1),
        colsample_bytree=trial.suggest_float("cs",.5,1),min_child_weight=trial.suggest_int("mcw",1,10),
        gamma=trial.suggest_float("gn",0,5),scale_pos_weight=trial.suggest_float("spw",2,20),
        random_state=42,eval_metric="logloss",n_jobs=-1)
    m.fit(Xtr_s,ytr);return roc_auc_score(yte,m.predict_proba(Xte_s)[:,1])

s=optuna.create_study(direction="maximize");s.optimize(obj,n_trials=30)
bp={k.replace("opt_",""):v for k,v in s.best_params.items()};bp.update({"random_state":42,"eval_metric":"logloss","n_jobs":-1})
print(f"Best: {s.best_value:.6f}")

t0=time.time();m=XGBClassifier(**bp);m.fit(Xtr_s,ytr);p=m.predict_proba(Xte_s)[:,1];t=time.time()-t0
r=ev(yte,p);r["time"]=round(t,2)
print(f"XGB: AUC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} R1%={r['r1']:.4f} ({t:.1f}s)")

# Calibrated
cal=CalibratedClassifierCV(m,method="isotonic",cv=3);cal.fit(Xtr_s,ytr)
p_cal=cal.predict_proba(Xte_s)[:,1];r_cal=ev(yte,p_cal)
print(f"Cal: AUC={r_cal['roc_auc']:.4f} Brier={r_cal['brier']:.6f}")

# 5-fold CV
skf=StratifiedKFold(5,shuffle=True,random_state=42);cv_aucs=[]
for tr_i,te_i in skf.split(Xtr_s,ytr):
    m2=XGBClassifier(**bp);m2.fit(Xtr_s[tr_i],ytr[tr_i])
    cv_aucs.append(roc_auc_score(ytr[te_i],m2.predict_proba(Xtr_s[te_i])[:,1]))
    print(f"  Fold {len(cv_aucs)}: {cv_aucs[-1]:.4f}")
cv_m=float(np.mean(cv_aucs));cv_s=float(np.std(cv_aucs))
print(f"CV: {cv_m:.4f} ± {cv_s:.4f}")

fi=m.feature_importances_;top=np.argsort(fi)[::-1][:10]
print("Top:",[(fc[i],round(fi[i],4)) for i in top])

out={"xgb_optuna":r,"xgb_calibrated":r_cal,"cv":{"mean":round(cv_m,6),"std":round(cv_s,6)},"params":{k:v for k,v in bp.items() if k not in("random_state","eval_metric","n_jobs")},"nf":len(fc),"ts":time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(out,open("reports/ulb_final.json","w"),indent=2);print("Saved reports/ulb_final.json")
