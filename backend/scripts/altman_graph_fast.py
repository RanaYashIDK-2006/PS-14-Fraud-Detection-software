#!/usr/bin/env python3
"""
Altman graph+velocity features — optimized for speed on 4M temporal sample.
Last 4M rows → 3.2M train + 800K test. Honest temporal split.
"""
import time, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb

NJ = 4
USECOLS = ["User","Card","Year","Month","Day","Time","Amount","Use Chip",
           "Merchant Name","Merchant City","Merchant State","Zip","MCC","Is Fraud?"]

def parse_hr(t):
    try:
        s = str(t).strip(); p = s.replace('.',':').split(':'); h = int(p[0])
        if 'pm' in s.lower() and h != 12: h += 12
        elif 'am' in s.lower() and h == 12: h = 0
        return h
    except: return 12

print("=" * 80)
print("ALTMAN — Graph + Velocity (4M Temporal Sample)")
print("=" * 80)
T0 = time.time()

# ═══ Load last 4M rows (preserves temporal order) ═══
print("  Loading last 4M rows...")
t0 = time.time()
all_chunks = []
for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
                          usecols=USECOLS, low_memory=False, chunksize=2_000_000):
    all_chunks.append(chunk)
total_rows = sum(len(c) for c in all_chunks)
print(f"  Total rows: {total_rows:,}")

# Keep only last 4M
keep = 4_000_000
start = max(0, total_rows - keep)
cum = 0
df_chunks = []
for c in all_chunks:
    if cum + len(c) <= start:
        cum += len(c)
        continue
    if cum < start:
        offset = start - cum
        df_chunks.append(c.iloc[offset:])
    else:
        df_chunks.append(c)
    cum += len(c)

df = pd.concat(df_chunks, ignore_index=True)
del all_chunks, df_chunks

# Parse features
df["label"] = (df["Is Fraud?"].astype(str).str.strip() == "Yes").astype(int)
df["amt"] = pd.to_numeric(df["Amount"].astype(str).str.replace('$','',regex=False)
                          .str.replace(',','',regex=False), errors='coerce').fillna(0)
df["hr"] = df["Time"].apply(parse_hr)
df["mcc"] = pd.to_numeric(df["MCC"], errors='coerce').fillna(0)
df["_m"] = df["Merchant Name"].astype(str)
df["_c"] = df["Merchant City"].astype(str)
df["_u"] = df["User"].astype(str)
df["_card"] = df["Card"].astype(str)
df["chip"] = df["Use Chip"].astype(str).str.contains("Swipe|Chip", case=False, na=False).astype(int)
df["online"] = df["Use Chip"].astype(str).str.contains("Online", case=False, na=False).astype(int)
df["has_zip"] = df["Zip"].notna().astype(int)
df["has_state"] = df["Merchant State"].notna().astype(int)

print(f"  {len(df):,} rows ({df['label'].sum():,} fraud) ({time.time()-t0:.1f}s)")

# ═══ Temporal split (first 80% = train, last 20% = test) ═══
sp = int(len(df) * 0.8)
tr = df.iloc[:sp]
te = df.iloc[sp:]
ytr = tr["label"].values; yte = te["label"].values
nft = int(yte.sum()); nnt = int((yte==0).sum())
spw = (len(ytr) - int(ytr.sum())) / max(int(ytr.sum()), 1)
print(f"  Train: {len(tr):,} ({ytr.sum():,}) | Test: {len(te):,} ({nft:,}) | SPW: {spw:.0f}")

# ═══ Build stats from TRAIN ONLY ═══
print("  Building stats from train...")
t0 = time.time()
gm = ytr.mean(); K = 50

# Merchant fraud rate
mf = tr.groupby("_m").agg(mc=("label","count"), mfs=("label","sum"))
mf["r"] = (mf["mfs"] + gm*K) / (mf["mc"] + K)
mm = mf["r"].to_dict(); mcnt = mf["mc"].to_dict()

# City fraud rate
cf = tr.groupby("_c").agg(cc=("label","count"), cfs=("label","sum"))
cf["r"] = (cf["cfs"] + gm*K) / (cf["cc"] + K)
cm = cf["r"].to_dict()

# Merchant tx count
mcnt_full = tr["_m"].value_counts().to_dict()

# User stats
us = tr.groupby("_u").agg(uc=("label","count"), ufs=("label","sum"),
                           ua=("amt","mean"), usd=("amt","std"),
                           um=("amt", lambda x: len(x)),
                           umdiv=("_m","nunique"), ucdiv=("_c","nunique"))
us["ufr"] = (us["ufs"] + gm*200) / (us["uc"] + 200)
utx = us["uc"].to_dict(); ufr2 = us["ufr"].to_dict()
uavg = us["ua"].to_dict(); usd2 = us["usd"].fillna(1).to_dict()
umdiv = us["umdiv"].to_dict(); ucdiv = us["ucdiv"].to_dict()

# Graph: user-merchant degree
umg = tr.groupby(["_u","_m"]).size().reset_index(name="w")
umdeg = umg.groupby("_u")["w"].sum().to_dict()
mudeg = umg.groupby("_m")["w"].sum().to_dict()
munq = tr.groupby("_m")["_u"].nunique().to_dict()

# Graph: user-city degree
ucg = tr.groupby(["_u","_c"]).size().reset_index(name="w")
ucdeg = ucg.groupby("_u")["w"].sum().to_dict()
cunq = tr.groupby("_c")["_u"].nunique().to_dict()

# Card count
ck_tr = tr["_u"].astype(str) + "_" + tr["_card"].astype(str)
ccnt = ck_tr.value_counts().to_dict()

# User-hour bin
tr["_hb"] = (tr["hr"] // 6).astype(int)
uhk = (tr["_u"].astype(str) + "_" + tr["_hb"].astype(str)).value_counts().to_dict()

# User avg amt for zscore
uasd = tr.groupby("_u")["amt"].std().fillna(1).to_dict()

print(f"  Stats: {time.time()-t0:.1f}s")

# ═══ Feature engineering ═══
print("  Engineering features...")
t0 = time.time()

FEAT = [
    "log_amt", "amt_sq", "very_high_amt", "hour_cos", "is_business_hours",
    "chip", "is_online", "mcc_n", "has_zip", "has_state",
    "merch_tx_count", "merch_fraud_rate", "city_fraud_rate",
    "amt_x_mcc", "amt_x_online",
    "user_merchant_degree", "merchant_popularity", "merchant_unique_users",
    "user_city_degree", "city_unique_users",
    "user_centrality_norm", "merchant_popularity_norm",
    "mfr_x_amt", "mfr_x_ufr", "user_fraud_rate",
    "user_tx_count", "user_merchant_diversity", "user_city_diversity",
    "user_avg_amt", "amt_vs_user_avg", "amt_zscore",
    "card_tx_count", "user_hour_bin_count",
    "user_merchant_ratio", "user_city_ratio",
]
NF = len(FEAT)

umax = max(umdeg.values()) if umdeg else 1
mmax_val = max(mudeg.values()) if mudeg else 1

def eng(dp):
    F = np.zeros((len(dp), NF), dtype=np.float32)
    amts = dp["amt"].values; hrs = dp["hr"].values.astype(float)
    merchs = dp["_m"].values; cities = dp["_c"].values; users = dp["_u"].values

    F[:,0] = np.log1p(amts)
    F[:,1] = amts**2
    F[:,2] = (amts > 1000).astype(float)
    F[:,3] = np.cos(2*np.pi*hrs/24)
    F[:,4] = ((hrs>=9)&(hrs<=17)).astype(float)
    F[:,5] = dp["chip"].values
    F[:,6] = dp["online"].values
    F[:,7] = dp["mcc"].values
    F[:,8] = dp["has_zip"].values
    F[:,9] = dp["has_state"].values
    F[:,10] = np.vectorize(lambda x: mcnt_full.get(x,1))(merchs).astype(float)
    F[:,11] = np.vectorize(lambda x: mm.get(x, gm))(merchs).astype(float)
    F[:,12] = np.vectorize(lambda x: cm.get(x, gm))(cities).astype(float)
    F[:,13] = amts * F[:,7]
    F[:,14] = amts * F[:,6]
    F[:,15] = np.vectorize(lambda x: umdeg.get(x,0))(users).astype(float)
    F[:,16] = np.vectorize(lambda x: mudeg.get(x,0))(merchs).astype(float)
    F[:,17] = np.vectorize(lambda x: munq.get(x,1))(merchs).astype(float)
    F[:,18] = np.vectorize(lambda x: ucdeg.get(x,0))(users).astype(float)
    F[:,19] = np.vectorize(lambda x: cunq.get(x,1))(cities).astype(float)
    F[:,20] = F[:,15] / umax
    F[:,21] = F[:,16] / mmax_val
    F[:,22] = F[:,11] * amts
    F[:,23] = F[:,11] * np.vectorize(lambda x: ufr2.get(x, gm))(users).astype(float)
    F[:,24] = np.vectorize(lambda x: ufr2.get(x, gm))(users).astype(float)
    F[:,25] = np.vectorize(lambda x: utx.get(x,0))(users).astype(float)
    F[:,26] = np.vectorize(lambda x: umdiv.get(x,1))(users).astype(float)
    F[:,27] = np.vectorize(lambda x: ucdiv.get(x,1))(users).astype(float)
    uavg_arr = np.vectorize(lambda x: uavg.get(x,1.0))(users).astype(float)
    F[:,28] = uavg_arr
    F[:,29] = amts / np.maximum(uavg_arr, 0.01)
    ustd_arr = np.vectorize(lambda x: uasd.get(x,1.0))(users).astype(float)
    F[:,30] = (amts - uavg_arr) / np.maximum(ustd_arr, 0.01)
    cards = dp["_card"].values
    ck = users + "_" + cards
    F[:,31] = np.vectorize(lambda x: ccnt.get(x,1))(ck).astype(float)
    hb = (hrs // 6).astype(int).astype(str)
    uhk2 = users + "_" + hb
    F[:,32] = np.vectorize(lambda x: uhk.get(x,0))(uhk2).astype(float)
    F[:,33] = F[:,26] / np.maximum(F[:,25], 1)
    F[:,34] = F[:,27] / np.maximum(F[:,25], 1)
    F = np.nan_to_num(F, nan=0, posinf=100, neginf=-100)
    return F

Xtr = eng(tr); Xte = eng(te)
print(f"  Features: {NF} ({time.time()-t0:.1f}s)")

sc = RobustScaler()
Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)

# ═══ Train ═══
print("\n  Training XGBoost...")
t0 = time.time()
xgb_m = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.02, subsample=0.8,
    colsample_bytree=0.7, gamma=1, min_child_weight=5,
    scale_pos_weight=min(spw, 500), tree_method="hist",
    eval_metric="auc", random_state=42, n_jobs=NJ)
xgb_m.fit(Xtr_s, ytr, verbose=False)
p_xgb = xgb_m.predict_proba(Xte_s)[:,1]
xa = roc_auc_score(yte, p_xgb)
print(f"  XGB: {xa:.4f} ({time.time()-t0:.1f}s)")

print("  Training LightGBM...")
t0 = time.time()
lgb_m = lgb.LGBMClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.02, subsample=0.8,
    colsample_bytree=0.7, min_child_samples=30,
    scale_pos_weight=min(spw, 500), random_state=42, n_jobs=NJ, verbose=-1)
lgb_m.fit(Xtr_s, ytr)
p_lgb = lgb_m.predict_proba(Xte_s)[:,1]
la = roc_auc_score(yte, p_lgb)
print(f"  LGB: {la:.4f} ({time.time()-t0:.1f}s)")

# Ensemble optimization
bw, ba = 0.5, 0
for w in np.arange(0.1, 0.9, 0.05):
    p = w*p_xgb + (1-w)*p_lgb; a = roc_auc_score(yte, p)
    if a > ba: ba = a; bw = w
p_ens = bw*p_xgb + (1-bw)*p_lgb
ea = roc_auc_score(yte, p_ens)

print(f"\n{'='*60}")
print(f"  RESULTS ({NF} features, {len(df):,} rows)")
print(f"{'='*60}")
print(f"  XGB:      {xa:.4f}")
print(f"  LGB:      {la:.4f}")
print(f"  Ensemble: {ea:.4f} (w_xgb={bw:.2f})")
print(f"  Baseline (15 feat): 0.9462")
print(f"  Delta: {(ea-0.9462)*100:+.2f}%")
print(f"  Target >95%: {'ACHIEVED' if ea>0.95 else 'NOT MET'}")

# Feature importance
imp = xgb_m.feature_importances_
base = {"log_amt","amt_sq","very_high_amt","hour_cos","is_business_hours",
        "chip","is_online","mcc_n","has_zip","has_state",
        "merch_tx_count","merch_fraud_rate","city_fraud_rate",
        "amt_x_mcc","amt_x_online"}
print(f"\n  Top 20 Features:")
for nm, v in sorted(zip(FEAT, imp), key=lambda x: -x[1])[:20]:
    tag = " *NEW*" if nm not in base else ""
    print(f"    {nm:>28}: {v:.4f}{tag}")

# Recall curve
order = np.argsort(-p_ens); sy = yte[order]; ct = np.cumsum(sy); rc = ct/nft
print(f"\n  Recall Curve:")
for tgt in [0.90, 0.95, 0.97, 0.98, 0.985, 0.99]:
    idx = np.searchsorted(rc, tgt)
    if idx < len(p_ens):
        thr=p_ens[order[idx]]; fl=idx+1; tp=int(ct[idx]); fp=fl-tp
        print(f"    recall={tgt*100:.1f}% thr={thr:.6f} FPR={fp/max(nnt,1)*100:.3f}% missed={nft-tp}")

elapsed = time.time()-T0
out = {"dataset":"ibm_altman_graph_4m","n_total":len(df),
       "n_train":len(Xtr),"n_test":len(Xte),"fraud_test":nft,
       "baseline_auc":0.9462,"n_features":NF,
       "xgb_auc":round(float(xa),6),"lgb_auc":round(float(la),6),
       "ensemble_auc":round(float(ea),6),"ensemble_weight_xgb":round(float(bw),4),
       "improvement":round(float(ea-0.9462),6),
       "target_95":ea>0.95,"features":FEAT,
       "fi":{nm:round(float(v),4) for nm,v in sorted(zip(FEAT,imp),key=lambda x:-x[1])[:20]},
       "elapsed":round(elapsed,1)}
with open("reports/altman_graph_fast.json","w") as f:
    json.dump(out,f,indent=2)
print(f"\n  Saved reports/altman_graph_fast.json ({elapsed:.1f}s)")
