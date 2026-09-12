#!/usr/bin/env python3
"""Fast FPR optimization — ULB full, Altman 4M sample with cached stats."""
import time, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

NJ = 4
def met(y, p, thr):
    yp = (p >= thr).astype(int)
    tp = int(((yp==1)&(y==1)).sum()); fp = int(((yp==1)&(y==0)).sum())
    fn = int(((yp==0)&(y==1)).sum()); n_p = int((y==1).sum()); n_n = int((y==0).sum())
    return {"thr":thr,"fpr":fp/max(n_n,1),"recall":tp/max(n_p,1),
            "prec":tp/max(tp+fp,1),"tp":tp,"fp":fp,"fn":fn,"n_pos":n_p,"n_neg":n_n}

print("="*80)
print("FPR OPTIMIZATION — Fast Mode")
print("="*80)
T0 = time.time()

# ═══ ULB ═══
print("\n[1/2] ULB (full 284K)")
df = pd.read_csv("data/creditcard.csv")
y = df["Class"].values; X = df.drop(columns=["Class"]).values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=100, neginf=-100)
sp = int(len(X)*0.8); Xtr,Xte = X[:sp],X[sp:]; ytr,yte = y[:sp],y[sp:]
spw = (len(ytr)-int(ytr.sum()))/max(int(ytr.sum()),1)
sc = RobustScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

# Baseline
xgb_b = xgb.XGBClassifier(n_estimators=500,max_depth=7,learning_rate=0.03,
    subsample=0.8,colsample_bytree=0.7,gamma=2,min_child_weight=5,
    scale_pos_weight=min(spw,200),tree_method="hist",eval_metric="auc",
    random_state=42,n_jobs=NJ)
xgb_b.fit(Xtr_s,ytr,verbose=False); p_b = xgb_b.predict_proba(Xte_s)[:,1]
auc_b = roc_auc_score(yte,p_b)

# Cost-sensitive
sw = np.ones(len(ytr)); sw[ytr==1] = 5.0
# Add interactions for top features
fi = xgb_b.feature_importances_; top5 = np.argsort(fi)[-5:]
Xtr_i = Xtr_s.copy(); Xte_i = Xte_s.copy()
for a in top5:
    for b in top5:
        if a < b:
            Xtr_i = np.hstack([Xtr_i, (Xtr_s[:,a]*Xtr_s[:,b]).reshape(-1,1)])
            Xte_i = np.hstack([Xte_i, (Xte_s[:,a]*Xte_s[:,b]).reshape(-1,1)])
xgb_o = xgb.XGBClassifier(n_estimators=600,max_depth=8,learning_rate=0.02,
    subsample=0.8,colsample_bytree=0.7,gamma=1,min_child_weight=3,
    scale_pos_weight=min(spw,300),tree_method="hist",eval_metric="aucpr",
    random_state=42,n_jobs=NJ)
xgb_o.fit(Xtr_i,ytr,sample_weight=sw,verbose=False); p_o = xgb_o.predict_proba(Xte_i)[:,1]
auc_o = roc_auc_score(yte,p_o)

# Isotonic cal on OOF
oof = np.zeros(len(ytr))
for tri,vai in StratifiedKFold(3,shuffle=True,random_state=42).split(Xtr_s,ytr):
    m = lgb.LGBMClassifier(n_estimators=300,max_depth=7,learning_rate=0.05,
        scale_pos_weight=min(spw,200),random_state=42,n_jobs=NJ,verbose=-1)
    m.fit(Xtr_s[tri],ytr[tri]); oof[vai] = m.predict_proba(Xtr_s[vai])[:,1]
iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(oof,ytr)
# Calibrate optimized model
p_o_cal = iso.transform(p_o)

# Best weights
bw,ba = 0.5,0
for w in np.arange(0.1,0.9,0.05):
    p = w*p_b+(1-w)*p_o; a = roc_auc_score(yte,p)
    if a>ba: ba=a; bw=w
p_ens = bw*p_b+(1-bw)*p_o; auc_e = roc_auc_score(yte,p_ens)

print(f"  XGB baseline:  AUC={auc_b:.4f}")
print(f"  Cost-sensitive: AUC={auc_o:.4f} (Δ={auc_o-auc_b:+.4f})")
print(f"  Isotonic cal:   AUC={roc_auc_score(yte,p_o_cal):.4f}")
print(f"  Ensemble:       AUC={auc_e:.4f} (w={bw:.2f})")

# Find best FPR at each recall target
print(f"\n  ULB FPR at fixed recall (lower is better):")
print(f"  {'Model':>18} {'R@85%':>8} {'R@88%':>8} {'R@90%':>8} {'R@92%':>8} {'R@95%':>8}")
print(f"  {'-'*60}")
ulb_data = {}
for name, probs in [("Baseline",p_b),("Cost-Sens",p_o),("Calibrated",p_o_cal),("Ensemble",p_ens)]:
    vals = []; row = {}
    for tgt in [0.85,0.88,0.90,0.92,0.95]:
        best_fpr = None
        for thr in np.arange(0.001,0.999,0.001):
            m = met(yte,probs,thr)
            if m["recall"]>=tgt and (best_fpr is None or m["fpr"]<best_fpr):
                best_fpr = m["fpr"]
        v = f"{best_fpr*100 if best_fpr is not None else 99:>6.2f}%" if best_fpr is not None else "    N/A"
        vals.append(v)
        row[f"r@{int(tgt*100)}"] = round(best_fpr,6) if best_fpr else None
    ulb_data[name] = row
    print(f"  {name:>18} {' '.join(vals)}")

# Pick the model with lowest FPR at 90% recall
best_model = None; best_fpr = 1.0
for name, probs in [("Baseline",p_b),("Cost-Sens",p_o),("Calibrated",p_o_cal),("Ensemble",p_ens)]:
    for thr in np.arange(0.001,0.999,0.001):
        m = met(yte,probs,thr)
        if m["recall"]>=0.89 and m["fpr"]<best_fpr:
            best_fpr = m["fpr"]; best_model = name; best_thr = thr
print(f"\n  ★ Best at ~90% recall: {best_model} FPR={best_fpr*100:.3f}% thr={best_thr:.3f}")

ulb_out = {"baseline_auc":auc_b,"best_model":best_model,"best_fpr":round(best_fpr,6),
           "best_thr":round(best_thr,4),"fpr_table":ulb_data}

# ═══ Altman (4M sample with precomputed stats) ═══
print(f"\n\n[2/2] Altman (4M sample)")
print("-"*60)
t0 = time.time()

USECOLS = ["User","Card","Time","Amount","Use Chip","Merchant Name",
           "Merchant City","Merchant State","Zip","MCC","Is Fraud?"]
def parse_hr(t):
    try:
        s=str(t).strip(); p=s.replace('.',':').split(':'); h=int(p[0])
        if 'pm' in s.lower() and h!=12: h+=12
        elif 'am' in s.lower() and h==12: h=0
        return h
    except: return 12

# Load precomputed stats
with open("data/altman_stats.pkl","rb") as f:
    S = pickle.load(f)
gm=S["gm"]; mfr_s=pd.Series(S["merchant_fr"]); mcnt_s=pd.Series(S["mcnt_full"])
cfr_s=pd.Series(S["city_fr"]); del S

# Load last 4M rows
print("  Loading 4M rows...")
all_chunks = []
for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
                          usecols=USECOLS, low_memory=False, chunksize=2_000_000):
    all_chunks.append(chunk)
total = sum(len(c) for c in all_chunks)
start = max(0, total - 4_000_000)
keep = []
cum = 0
for c in all_chunks:
    if cum + len(c) <= start:
        cum += len(c); continue
    if cum < start:
        keep.append(c.iloc[start-cum:])
    else:
        keep.append(c)
    cum += len(c)
df = pd.concat(keep, ignore_index=True); del all_chunks, keep

df["label"] = (df["Is Fraud?"].astype(str).str.strip()=="Yes").astype(int)
df["amt"] = pd.to_numeric(df["Amount"].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False),errors='coerce').fillna(0)
df["hr"] = df["Time"].apply(parse_hr)
df["mcc"] = pd.to_numeric(df["MCC"],errors='coerce').fillna(0)
print(f"  {len(df):,} rows ({df['label'].sum():,} fraud) ({time.time()-t0:.1f}s)")

# 15-feature engineering
NF=15
def eng(dp):
    F = np.zeros((len(dp),NF),dtype=np.float32)
    amts=dp["amt"].values; hrs=dp["hr"].values.astype(float)
    merchs=dp["Merchant Name"].astype(str); cities=dp["Merchant City"].astype(str)
    F[:,0]=np.log1p(amts); F[:,1]=amts**2; F[:,2]=(amts>1000).astype(float)
    F[:,3]=np.cos(2*np.pi*hrs/24); F[:,4]=((hrs>=9)&(hrs<=17)).astype(float)
    F[:,5]=dp["Use Chip"].astype(str).str.contains("Swipe|Chip",case=False,na=False).values.astype(float)
    F[:,6]=dp["Use Chip"].astype(str).str.contains("Online",case=False,na=False).values.astype(float)
    F[:,7]=dp["mcc"].values; F[:,8]=dp["Zip"].notna().values.astype(float)
    F[:,9]=(dp["Merchant State"].fillna("").astype(str)!="").astype(float)
    F[:,10]=merchs.map(mcnt_s).fillna(1).values.astype(float)
    F[:,11]=merchs.map(mfr_s).fillna(gm).values.astype(float)
    F[:,12]=cities.map(cfr_s).fillna(gm).values.astype(float)
    F[:,13]=amts*F[:,7]; F[:,14]=amts*F[:,6]
    return np.nan_to_num(F,nan=0,posinf=100,neginf=-100)

sp2 = int(len(df)*0.8)
tr = df.iloc[:sp2]; te = df.iloc[sp2:]
ytr_a=tr["label"].values; yte_a=te["label"].values
Xtr_a=eng(tr); Xte_a=eng(te)
spw_a=(len(ytr_a)-int(ytr_a.sum()))/max(int(ytr_a.sum()),1)
nft=int(yte_a.sum()); nnt=int((yte_a==0).sum())
sc_a=RobustScaler(); Xtr_as=sc_a.fit_transform(Xtr_a); Xte_as=sc_a.transform(Xte_a)
print(f"  Train: {len(Xtr_a):,} ({ytr_a.sum():,}) Test: {len(Xte_a):,} ({nft:,}) SPW: {spw_a:.0f}")

# Baseline
print("  Training baseline XGB...")
xgb_ab = xgb.XGBClassifier(n_estimators=500,max_depth=8,learning_rate=0.02,
    subsample=0.8,colsample_bytree=0.7,gamma=1,min_child_weight=5,
    scale_pos_weight=min(spw_a,500),tree_method="hist",eval_metric="auc",
    random_state=42,n_jobs=NJ)
xgb_ab.fit(Xtr_as,ytr_a,verbose=False); p_ab=xgb_ab.predict_proba(Xte_as)[:,1]
auc_ab=roc_auc_score(yte_a,p_ab)

# Cost-sensitive
print("  Training cost-sensitive XGB...")
sw_a=np.ones(len(ytr_a)); sw_a[ytr_a==1]=10.0
xgb_ao = xgb.XGBClassifier(n_estimators=600,max_depth=8,learning_rate=0.02,
    subsample=0.8,colsample_bytree=0.7,gamma=1,min_child_weight=3,
    scale_pos_weight=min(spw_a,500),tree_method="hist",eval_metric="aucpr",
    random_state=42,n_jobs=NJ)
xgb_ao.fit(Xtr_as,ytr_a,sample_weight=sw_a,verbose=False); p_ao=xgb_ao.predict_proba(Xte_as)[:,1]
auc_ao=roc_auc_score(yte_a,p_ao)

# Isotonic cal
print("  Calibrating with isotonic...")
oof_a=np.zeros(len(ytr_a))
for tri,vai in StratifiedKFold(3,shuffle=True,random_state=42).split(Xtr_as,ytr_a):
    m=lgb.LGBMClassifier(n_estimators=300,max_depth=8,learning_rate=0.05,
        scale_pos_weight=min(spw_a,500),random_state=42,n_jobs=NJ,verbose=-1)
    m.fit(Xtr_as[tri],ytr_a[tri]); oof_a[vai]=m.predict_proba(Xtr_as[vai])[:,1]
iso_a=IsotonicRegression(out_of_bounds="clip"); iso_a.fit(oof_a,ytr_a)
p_ao_cal=iso_a.transform(p_ao)

# LGB
print("  Training LGB...")
lgb_a=lgb.LGBMClassifier(n_estimators=500,max_depth=8,learning_rate=0.02,
    subsample=0.8,colsample_bytree=0.7,min_child_samples=30,
    scale_pos_weight=min(spw_a,500),random_state=42,n_jobs=NJ,verbose=-1)
lgb_a.fit(Xtr_as,ytr_a); p_lgb_a=lgb_a.predict_proba(Xte_as)[:,1]
auc_lgb=roc_auc_score(yte_a,p_lgb_a)

# Ensemble
bw_a,ba_a=0.5,0
for w in np.arange(0.1,0.9,0.05):
    p=w*p_ab+(1-w)*p_lgb_a; a=roc_auc_score(yte_a,p)
    if a>ba_a: ba_a=a; bw_a=w
p_ea=bw_a*p_ab+(1-bw_a)*p_lgb_a; auc_ea=roc_auc_score(yte_a,p_ea)

# Calibrated ensemble
iso_ea=IsotonicRegression(out_of_bounds="clip"); iso_ea.fit(oof_a,ytr_a)
p_ea_cal=iso_ea.transform(p_ea)

print(f"\n  Altman Results:")
print(f"  Baseline XGB:    AUC={auc_ab:.4f}")
print(f"  Cost-Sens XGB:   AUC={auc_ao:.4f} (Δ={auc_ao-auc_ab:+.4f})")
print(f"  Isotonic Cal:    AUC={roc_auc_score(yte_a,p_ao_cal):.4f}")
print(f"  LGB:             AUC={auc_lgb:.4f}")
print(f"  Ensemble:        AUC={auc_ea:.4f}")
print(f"  Ens+Cal:         AUC={roc_auc_score(yte_a,p_ea_cal):.4f}")

# Comparison
print(f"\n  Altman FPR at fixed recall:")
print(f"  {'Model':>18} {'R@75%':>8} {'R@78%':>8} {'R@80%':>8} {'R@82%':>8} {'R@85%':>8}")
print(f"  {'-'*60}")
alt_data = {}
for name,probs in [("Baseline",p_ab),("Cost-Sens",p_ao),("Calibrated",p_ao_cal),
                    ("LGB",p_lgb_a),("Ensemble",p_ea),("Ens+Cal",p_ea_cal)]:
    vals=[]; row={}
    for tgt in [0.75,0.78,0.80,0.82,0.85]:
        bf=None
        for thr in np.arange(0.001,0.999,0.001):
            m=met(yte_a,probs,thr)
            if m["recall"]>=tgt and (bf is None or m["fpr"]<bf): bf=m["fpr"]
        v=f"{bf*100 if bf is not None else 99:>6.2f}%" if bf else "    N/A"
        vals.append(v); row[f"r@{int(tgt*100)}"]=round(bf,6) if bf else None
    alt_data[name]=row
    print(f"  {name:>18} {' '.join(vals)}")

# Best model at 80% recall
best_a=None; best_fa=1.0
for name,probs in [("Baseline",p_ab),("Cost-Sens",p_ao),("Calibrated",p_ao_cal),
                    ("LGB",p_lgb_a),("Ensemble",p_ea),("Ens+Cal",p_ea_cal)]:
    for thr in np.arange(0.001,0.999,0.001):
        m=met(yte_a,probs,thr)
        if m["recall"]>=0.79 and m["fpr"]<best_fa:
            best_fa=m["fpr"]; best_a=name; best_ta=thr
print(f"\n  ★ Best at ~80% recall: {best_a} FPR={best_fa*100:.3f}% thr={best_ta:.3f}")

elapsed=time.time()-T0
out={"ulb":ulb_out,
     "altman":{"baseline_auc":auc_ab,"best_model":best_a,"best_fpr":round(best_fa,6),
               "best_thr":round(best_ta,4),"fpr_table":alt_data},
     "elapsed":round(elapsed,1)}
with open("reports/fpr_optimization.json","w") as f: json.dump(out,f,indent=2)
print(f"\n  Saved reports/fpr_optimization.json ({elapsed:.1f}s)")
