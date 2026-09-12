#!/usr/bin/env python3
"""Fast 24M Altman: proven params, 1% legit sample, 3-fold CV."""
import json, time, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
warnings.filterwarnings("ignore")
np.random.seed(42); NJ=4; R=Path("reports"); R.mkdir(exist_ok=True)
def raf(y,p,t):
    fpr,tpr,_=roc_curve(y,p); m=fpr<=t
    return float(tpr[m].max()) if m.any() else 0.0

# Load ALL 24M rows
print("Loading ALL 24M Altman rows...")
t0=time.time(); rng=np.random.RandomState(42)
rows_sampled=[]; ut=defaultdict(list); ua=defaultdict(list); um=defaultdict(list)
ci=0
for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
    usecols=["User","Month","Day","Time","Amount","Use Chip","MCC","Errors?",
             "Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"],
    low_memory=False, chunksize=1_000_000):
    c=chunk.copy()
    c["amt"]=c["Amount"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float)
    c["label"]=(c["Is Fraud?"]=="Yes").astype(int)
    chip_map={"Swipe Transaction":0,"Online Transaction":1,"Chip Transaction":2}
    c["chip"]=c["Use Chip"].map(chip_map).fillna(-1)
    c["mcc_n"]=c["MCC"].astype(str).str[:4].astype(float)/10000
    c["err"]=(c["Errors?"].fillna("")!="").astype(int)
    tp=c["Time"].str.split(":",expand=True)
    c["hr"]=tp[0].astype(float); c["mn"]=tp[1].astype(float)
    c["merchant_id"]=c["Merchant Name"].astype("category").cat.codes
    c["city_id"]=c["Merchant City"].astype("category").cat.codes
    c["state_id"]=c["Merchant State"].fillna("UNK").astype("category").cat.codes
    c["card_n"]=c["Card"].astype(float); c["year_n"]=c["Year"].astype(float)-2010
    c["is_online"]=(c["Merchant City"]=="ONLINE").astype(int)
    c["month_n"]=c["Month"].astype(float); c["day_n"]=c["Day"].astype(float)
    c["hr_bin"]=pd.cut(c["hr"],bins=[0,6,12,18,24],labels=[0,1,2,3]).astype(float)
    c["amt_log"]=np.log1p(c["amt"].clip(upper=1e9))
    c["day_decimal"]=c["Day"]+c["hr"]/24.0+c["mn"]/1440.0
    fm=c["label"]==1
    rows_sampled.append(pd.concat([c[fm], c[~fm].sample(frac=0.01,random_state=rng)]))
    # History for ALL rows
    s=c["amt"].values; d=c["day_decimal"].values; m=c["merchant_id"].values; u=c["User"].values
    for i in range(len(c)):
        ut[u[i]].append(d[i]); ua[u[i]].append(s[i]); um[u[i]].append(m[i])
    ci+=1
    if ci%5==0: print(f"  Chunk {ci}/24 ({time.time()-t0:.0f}s)")
    if ci>=24: break

df=pd.concat(rows_sampled,ignore_index=True)
y=df["label"].values
print(f"Sampled: {len(df):,} rows ({int(y.sum())} fraud) in {time.time()-t0:.1f}s")

# Sort histories
print("Sorting histories...")
for uid in ut:
    o=np.argsort(ut[uid])
    ut[uid]=np.array(ut[uid])[o]; ua[uid]=np.array(ua[uid])[o]; um[uid]=np.array(um[uid])[o]
uh={uid:{"times":ut[uid],"amounts":ua[uid],"merchants":um[uid]} for uid in ut}
del ut,ua,um
print(f"  {len(uh):,} users ({time.time()-t0:.1f}s)")

# Target encoding
print("Target encoding...")
gm=df["label"].mean()
for ec,gc,p in [("user_te","User",200),("merchant_te","merchant_id",50),("city_te","city_id",20)]:
    oof=np.zeros(len(df))
    for tri,vai in StratifiedShuffleSplit(2,test_size=0.2,random_state=42).split(df,df["label"]):
        s=df.iloc[tri].groupby(gc)["label"].agg(["mean","count"])
        s["s"]=(s["mean"]*s["count"]+gm*p)/(s["count"]+p)
        oof[vai]=df.iloc[vai][gc].map(s["s"].to_dict()).fillna(gm).values
    df[ec]=oof

# Interactions
df["log_amt"]=np.log1p(df["amt"]); df["amt_x_hr"]=df["amt"]*df["hr"]
df["amt_x_mcc"]=df["amt"]*df["mcc_n"]; df["amt_x_chip"]=df["amt"]*df["chip"]
df["amt_x_ute"]=df["amt"]*df["user_te"]; df["amt_x_mte"]=df["amt"]*df["merchant_te"]
df["amt_x_online"]=df["amt"]*df["is_online"]; df["hr_x_chip"]=df["hr"]*df["chip"]
df["ute_x_mte"]=df["user_te"]*df["merchant_te"]; df["chip_x_online"]=df["chip"]*df["is_online"]
df["err_x_amt"]=df["err"]*df["amt"]; df["hr_x_ute"]=df["hr"]*df["user_te"]
df["mcc_x_ute"]=df["mcc_n"]*df["user_te"]; df["amt_sq"]=df["amt"]**2
df["high_amt"]=(df["amt"]>df["amt"].quantile(0.95)).astype(float)
df["night_tx"]=((df["hr"]>=22)|(df["hr"]<=6)).astype(int)
df["weekend"]=((df["Day"]%7)>=5).astype(float)

# Velocity
print(f"Velocity features ({len(df):,} rows)...")
t3=time.time(); n=len(df); vel=np.zeros((n,10),dtype=np.float32)
for i in range(n):
    uid=int(df.iloc[i]["User"]); dd=df.iloc[i]["day_decimal"]
    h=uh.get(uid)
    if h is None or len(h["times"])<3: continue
    ta=h["times"]; aa=h["amounts"]; ma=h["merchants"]
    pos=np.searchsorted(ta,dd,side="right")
    tb=ta[:pos]; ab=aa[:pos]; mb=ma[:pos]; nb=len(tb)
    if nb<2: continue
    vel[i,0]=((dd-tb)<=(1.0/24.0)).sum()
    vel[i,1]=((dd-tb)<=1.0).sum()
    if nb>=10: vel[i,2]=ab[-5:].mean()-ab[-10:-5].mean()
    elif nb>=5: vel[i,2]=ab[-5:].mean()-ab[:-5].mean()
    l24=(dd-tb)<=1.0
    vel[i,3]=len(np.unique(mb[l24]))
    vel[i,4]=ab[l24].mean() if l24.sum()>0 else 0
    gaps=np.diff(tb[-min(5,nb):])*24
    vel[i,5]=gaps.mean() if len(gaps)>0 else 0
    vel[i,6]=gaps.std()/(gaps.mean()+1e-6) if len(gaps)>1 else 0
    if nb>=5: vel[i,7]=(ab[-1]-ab[-5:].mean())/(ab[-5:].std()+1e-6)
    if nb>=6:
        f_rec=1.0/(gaps[-min(3,len(gaps)):].mean()+0.01)
        f_prev=1.0/(np.diff(tb[-6:-3])*24).mean()+0.01
        vel[i,8]=f_rec-f_prev
    if nb>=2: vel[i,9]=1.0 if mb[-1] not in mb[:-1][-min(10,nb-1):] else 0.0
    if (i+1)%50000==0: print(f"  ...{i+1}/{n} ({time.time()-t3:.0f}s)")

vn=["tx_count_1h","tx_count_24h","amount_accel","merchant_diversity_24h",
    "avg_amount_24h","tx_gap_mean","tx_regularity","amount_zscore","tx_freq_accel","merchant_is_new"]
for j,nm in enumerate(vn): df[nm]=vel[:,j]
df["amt_x_tx5"]=df["amt"]*df["tx_count_1h"]; df["amt_x_tx24"]=df["amt"]*df["tx_count_24h"]
df["ute_x_tx24"]=df["user_te"]*df["tx_count_24h"]
print(f"Velocity done: {time.time()-t3:.0f}s")

drop={"label","User","Time","Amount","Use Chip","MCC","Errors?","Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"}
cols=[c for c in df.columns if c not in drop]
X=np.nan_to_num(df[cols].values.astype(np.float32))
print(f"Features: {len(cols)}")

# 3-fold CV with proven params
print("\n3-fold CV (proven params)...")
spw=(len(y)-int(y.sum()))/max(int(y.sum()),1)
lp=dict(n_estimators=350,max_depth=10,learning_rate=0.127,subsample=0.839,
    colsample_bytree=0.578,min_child_samples=9,reg_alpha=0.00171,reg_lambda=2.915,
    num_leaves=68,verbose=-1,random_state=42,n_jobs=NJ,scale_pos_weight=min(spw,20))
skf=StratifiedKFold(3,shuffle=True,random_state=42)
ap=np.zeros(len(y));fa=[];fr=[]
for fold,(tri,tei) in enumerate(skf.split(X,y)):
    tf=time.time()
    m=lgb.LGBMClassifier(**lp).fit(X[tri],y[tri])
    p=m.predict_proba(X[tei])[:,1]; ap[tei]=p
    a=roc_auc_score(y[tei],p); r=raf(y[tei],p,0.01)
    fa.append(a); fr.append(r)
    print(f"  Fold {fold}: AUC={a:.6f} R@1%FPR={r:.4f} ({time.time()-tf:.0f}s)")

oa=roc_auc_score(y,ap)
print(f"\n3-fold CV AUC:     {np.mean(fa):.6f}+/-{np.std(fa):.6f}")
print(f"3-fold CV R@1%FPR: {np.mean(fr):.6f}+/-{np.std(fr):.6f}")
print(f"OOF AUC:           {oa:.6f}")

# Compare vs 16-chunk
print(f"\n{'='*60}")
print(f"COMPARISON: 24M (full) vs 16M (sampled)")
print(f"{'='*60}")
print(f"16-chunk: AUC=0.9948 R@1%FPR=0.9246 (altman_velocity_v2)")
print(f"24M full: AUC={np.mean(fa):.6f} R@1%FPR={np.mean(fr):.6f}")
d_a=np.mean(fa)-0.9948; d_r=np.mean(fr)-0.9246
print(f"Delta: AUC={d_a:+.4f} R@1%FPR={d_r:+.4f}")
if d_a>0: print("More data IMPROVED AUC")
elif d_a<-0.001: print("More data HURT AUC (overfitting to noise?)")
else: print("More data had negligible effect (already at ceiling)")

result=dict(dataset="ALTMAN_FULL_24M",n_rows=len(y),n_features=len(cols),
    n_fraud=int(y.sum()),fraud_rate=round(float(y.mean()),6),
    chunks=ci,history_users=len(uh),
    cv_auc=round(float(np.mean(fa)),6),cv_auc_std=round(float(np.std(fa)),6),
    cv_r1=round(float(np.mean(fr)),6),cv_r1_std=round(float(np.std(fr)),6),
    oof_auc=round(oa,6),
    vs_16chunk=dict(delta_auc=round(d_a,4),delta_r1=round(d_r,4)),
    elapsed=round(time.time()-t0),
    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
with open(R/"altman_full_24m_results.json","w") as f: json.dump(result,f,indent=2)
print(f"\nSaved: reports/altman_full_24m_results.json")
print(f"Total: {time.time()-t0:.0f}s")
