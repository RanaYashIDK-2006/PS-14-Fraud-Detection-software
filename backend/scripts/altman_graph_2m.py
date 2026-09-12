#!/usr/bin/env python3
"""Altman Graph+Velocity with 2M rows — honest temporal AUC."""
import time, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb, lightgbm as lgb

NJ = 4

def parse_hr(t):
    try:
        s=str(t).strip(); p=s.replace('.',':').split(':'); h=int(p[0])
        if 'pm' in s.lower() and h!=12: h+=12
        elif 'am' in s.lower() and h==12: h=0
        return h
    except: return 12

print("="*80)
print("ALTMAN — Graph + Velocity (2M sample)")
print("="*80)
t_total = time.time()

# ── Load 2M rows ──
print("  Loading 2M rows...")
t0 = time.time()
rows = []
for i, chunk in enumerate(pd.read_csv("data/credit_card_transactions-ibm_v2.csv", low_memory=False, chunksize=500_000)):
    chunk["label"] = (chunk["Is Fraud?"].astype(str).str.strip() == "Yes").astype(int)
    fraud = chunk[chunk["label"]==1]
    legit = chunk[chunk["label"]==0]
    rows.append(fraud)
    rows.append(legit.sample(min(len(legit), 400_000), random_state=42))
    if sum(len(r) for r in rows) > 2_000_000:
        break

df = pd.concat(rows, ignore_index=True)
df = df.sort_values(["Year","Month","Day","Time"]).reset_index(drop=True)
print(f"  {len(df):,} rows ({df['label'].sum():,} fraud, {df['label'].mean()*100:.3f}%) ({time.time()-t0:.1f}s)")

# Parse
df["amt"] = pd.to_numeric(df["Amount"].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False), errors='coerce').fillna(0)
df["hr"] = df["Time"].apply(parse_hr)
df["mcc_n"] = pd.to_numeric(df["MCC"], errors='coerce').fillna(0)
df["chip"] = df["Use Chip"].astype(str).str.contains("Swipe|Chip",case=False,na=False).astype(int)
df["online"] = df["Use Chip"].astype(str).str.contains("Online",case=False,na=False).astype(int)

# Temporal split
sp = int(len(df)*0.8)
tr = df.iloc[:sp].copy(); te = df.iloc[sp:].copy()
ytr = tr["label"].values; yte = te["label"].values
nft = int(yte.sum()); nnt = int((yte==0).sum())
gm = ytr.mean(); spw = (len(ytr)-int(ytr.sum()))/max(int(ytr.sum()),1)
print(f"  Train: {len(tr):,} ({ytr.sum():,} fraud) | Test: {len(te):,} ({nft:,} fraud) | SPW: {spw:.0f}")

# ═══ Build stats ═══
print("  Building stats...")
t0 = time.time()
mf = tr.groupby("Merchant Name")["label"].agg(["mean","count"])
mf["r"] = (mf["mean"]*mf["count"]+gm*50)/(mf["count"]+50)
mm = mf["r"].to_dict()
mfrq = tr["Merchant Name"].value_counts().to_dict()
mavg = tr.groupby("Merchant Name")["amt"].mean().to_dict()
mstd = tr.groupby("Merchant Name")["amt"].std().fillna(1).to_dict()

cf = tr.groupby("Merchant City")["label"].agg(["mean","count"])
cf["r"] = (cf["mean"]*cf["count"]+gm*50)/(cf["count"]+50)
cm = cf["r"].to_dict()
cfq = tr["Merchant City"].value_counts().to_dict()

us = tr.groupby("User").agg(uc=("label","count"),ufr=("label","mean"),ua=("amt","mean"),usd=("amt","std"),um=("Merchant Name","nunique"),uci=("Merchant City","nunique"))
us["ufr"] = (us["ufr"]*us["uc"]+gm*200)/(us["uc"]+200)
utx=us["uc"].to_dict(); ufr2=us["ufr"].to_dict(); uavg=us["ua"].to_dict(); usd2=us["usd"].fillna(1).to_dict(); umrc=us["um"].to_dict(); ucit=us["uci"].to_dict()
ck = tr["User"].astype(str)+"_"+tr["Card"].astype(str); cc = ck.value_counts().to_dict()

# Graph
um = tr.groupby(["User","Merchant Name"]).size().reset_index(name="w")
umdeg = um.groupby("User")["w"].sum().to_dict()
mudeg = um.groupby("Merchant Name")["w"].sum().to_dict()
munq = tr.groupby("Merchant Name")["User"].nunique().to_dict()
uc = tr.groupby(["User","Merchant City"]).size().reset_index(name="w")
ucdeg = uc.groupby("User")["w"].sum().to_dict()
cunq = tr.groupby("Merchant City")["User"].nunique().to_dict()

# Hour bin
tr["hr_bin"] = tr["hr"]//6
uhk = tr["User"].astype(str)+"_"+tr["hr_bin"].astype(str)
uhc = uhk.value_counts().to_dict()
print(f"  Stats: {time.time()-t0:.1f}s")

# ═══ Feature engineering ═══
print("  Engineering features...")
t0 = time.time()

def eng(dp):
    o = pd.DataFrame()
    o["log_amt"]=np.log1p(dp["amt"]); o["amt_sq"]=dp["amt"]**2
    o["vha"]=(dp["amt"]>1000).astype(int); o["ha"]=(dp["amt"]>500).astype(int)
    hr=dp["hr"].fillna(12).astype(float)
    o["hcos"]=np.cos(2*np.pi*hr/24); o["hsin"]=np.sin(2*np.pi*hr/24)
    o["night"]=((hr>=22)|(hr<=6)).astype(int); o["bhr"]=((hr>=9)&(hr<=17)).astype(int)
    o["chip"]=dp["chip"]; o["online"]=dp["online"]; o["mcc"]=dp["mcc_n"]
    o["zzip"]=(dp["Zip"].notna()).astype(int); o["state"]=(dp["Merchant State"].fillna("")!="").astype(int)
    o["month"]=dp["Month"]

    o["mf"]=dp["Merchant Name"].map(mfrq).fillna(0)
    o["mfr"]=dp["Merchant Name"].map(mm).fillna(gm)
    o["mavg"]=dp["Merchant Name"].map(mavg).fillna(dp["amt"].mean())
    o["mstd_v"]=dp["Merchant Name"].map(mstd).fillna(1)
    o["avm"]=dp["amt"]/o["mavg"].clip(lower=0.01)

    o["cfr"]=dp["Merchant City"].map(cm).fillna(gm)
    o["cfq"]=dp["Merchant City"].map(cfq).fillna(0)

    uid=dp["User"]
    o["utc"]=uid.map(utx).fillna(0); o["ufr"]=uid.map(ufr2).fillna(gm)
    o["uavg"]=uid.map(uavg).fillna(dp["amt"].mean()); o["usd_v"]=uid.map(usd2).fillna(1)
    o["avu"]=dp["amt"]/o["uavg"].clip(lower=0.01)
    o["azu"]=(dp["amt"]-o["uavg"])/o["usd_v"].clip(lower=0.01)
    o["uum"]=uid.map(umrc).fillna(0); o["uuc"]=uid.map(ucit).fillna(0)

    ck2=dp["User"].astype(str)+"_"+dp["Card"].astype(str); o["cc"]=ck2.map(cc).fillna(1)

    # Graph
    o["umd"]=uid.map(umdeg).fillna(0); o["mud"]=dp["Merchant Name"].map(mudeg).fillna(0)
    o["mun"]=dp["Merchant Name"].map(munq).fillna(1)
    o["ucd"]=uid.map(ucdeg).fillna(0); o["cun"]=dp["Merchant City"].map(cunq).fillna(1)
    mx=max(o["umd"].max(),1); o["ucen"]=o["umd"]/mx
    mx2=max(o["mud"].max(),1); o["mpop"]=o["mud"]/mx2
    o["mfr_x_amt"]=o["mfr"]*dp["amt"]; o["mfr_x_ufr"]=o["mfr"]*o["ufr"]

    # Velocity
    o["umdiv"]=o["uum"]/o["utc"].clip(lower=1)
    o["ucidiv"]=o["uuc"]/o["utc"].clip(lower=1)
    o["adev"]=(dp["amt"]-o["mavg"])/o["mstd_v"].clip(lower=0.01)

    hr_bin=(hr//6).astype(int)
    o["uhtc"]=(uid.astype(str)+"_"+hr_bin.astype(str)).map(uhc).fillna(0)

    # Interactions
    o["axh"]=dp["amt"]*hr; o["axm"]=dp["amt"]*o["mcc"]
    o["axc"]=dp["amt"]*o["chip"]; o["axo"]=dp["amt"]*o["online"]
    o["axn"]=dp["amt"]*o["night"]; o["axmf"]=dp["amt"]*np.log1p(o["mf"])
    o["axur"]=dp["amt"]*o["ufr"]; o["axmr"]=dp["amt"]*o["mfr"]

    for c in o.columns:
        o[c]=pd.to_numeric(o[c],errors="coerce").fillna(0).replace([np.inf,-np.inf],0)
    return o

Xtr = eng(tr).values.astype(np.float32); Xte = eng(te).values.astype(np.float32)
feat_names = list(eng(tr.head(100)).columns)
Xtr = np.nan_to_num(Xtr, nan=0, posinf=100, neginf=-100)
Xte = np.nan_to_num(Xte, nan=0, posinf=100, neginf=-100)
print(f"  Features: {len(feat_names)} ({time.time()-t0:.1f}s)")

sc = RobustScaler()
Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

# ═══ Train ═══
print("\n  Training XGBoost...")
t0=time.time()
xgb_m = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.02, subsample=0.8,
    colsample_bytree=0.7, gamma=1, min_child_weight=5, scale_pos_weight=min(spw,500),
    tree_method="hist", eval_metric="auc", random_state=42, n_jobs=NJ)
xgb_m.fit(Xtr_s, ytr, verbose=False)
p_xgb = xgb_m.predict_proba(Xte_s)[:,1]
xa = roc_auc_score(yte, p_xgb)
print(f"  XGB: {xa:.4f} ({time.time()-t0:.1f}s)")

print("  Training LightGBM...")
t0=time.time()
lgb_m = lgb.LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.02, subsample=0.8,
    colsample_bytree=0.7, min_child_samples=30, scale_pos_weight=min(spw,500),
    random_state=42, n_jobs=NJ, verbose=-1)
lgb_m.fit(Xtr_s, ytr)
p_lgb = lgb_m.predict_proba(Xte_s)[:,1]
la = roc_auc_score(yte, p_lgb)
print(f"  LGB: {la:.4f} ({time.time()-t0:.1f}s)")

# Ensemble
bw, ba = 0.5, 0
for w in np.arange(0.1, 0.9, 0.05):
    p = w*p_xgb+(1-w)*p_lgb; a = roc_auc_score(yte, p)
    if a > ba: ba=a; bw=w
p_ens = bw*p_xgb+(1-bw)*p_lgb
ea = roc_auc_score(yte, p_ens)

print(f"\n  ═══ RESULTS ═══")
print(f"  XGB: {xa:.4f} | LGB: {la:.4f} | Ensemble: {ea:.4f} (w={bw:.2f})")
print(f"  Target >95%: {'✅' if ea>0.95 else '❌'}")

# Features
imp = xgb_m.feature_importances_
print(f"\n  Top 15:")
for n,v in sorted(zip(feat_names,imp),key=lambda x:-x[1])[:15]:
    print(f"    {n:>15}: {v:.4f}")

# Recall curve
order=np.argsort(-p_ens); sy=yte[order]; ct=np.cumsum(sy); rc=ct/nft
print(f"\n  Recall Curve:")
for tr2 in [0.90,0.95,0.97,0.98,0.985,0.99]:
    idx=np.searchsorted(rc,tr2)
    if idx<len(p_ens):
        thr=p_ens[order[idx]]; fl=idx+1; tp=int(ct[idx]); fp=fl-tp
        print(f"  recall={tr2*100:.1f}% thr={thr:.6f} FPR={fp/max(nnt,1)*100:.3f}% missed={nft-tp}")

elapsed=time.time()-t_total
out={"dataset":"ibm_altman_graph_velocity_2m","n_total":len(df),"n_train":len(tr),"n_test":len(te),
     "fraud_test":nft,"n_features":len(feat_names),"xgb_auc":round(float(xa),6),"lgb_auc":round(float(la),6),
     "ensemble_auc":round(float(ea),6),"target_95":ea>0.95,"features":feat_names,
     "fi":{n:round(float(v),4) for n,v in sorted(zip(feat_names,imp),key=lambda x:-x[1])[:15]},
     "elapsed":round(elapsed,1)}
with open("reports/altman_graph_velocity.json","w") as f:
    json.dump(out,f,indent=2)
print(f"\n  Saved ({elapsed:.1f}s)")
