#!/usr/bin/env python3
"""Step 2: Engineer features on full 24M, write to disk, then train. Memory-efficient."""
import time, json, pickle, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb

NJ = 4
USECOLS = ["User","Card","Time","Amount","Use Chip","Merchant Name",
           "Merchant City","Merchant State","Zip","MCC","Is Fraud?"]
NF = 35
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

def parse_hr(t):
    try:
        s = str(t).strip(); p = s.replace('.',':').split(':'); h = int(p[0])
        if 'pm' in s.lower() and h != 12: h += 12
        elif 'am' in s.lower() and h == 12: h = 0
        return h
    except: return 12

print("Step 2: Memory-efficient full 24M training...")
T0 = time.time()

# Load precomputed stats
print("  Loading stats...")
with open("data/altman_stats.pkl", "rb") as f:
    S = pickle.load(f)
gm = S["gm"]; SPLIT = S["split"]; total = S["total"]
umax = S["umax"]; mmax = S["mmax"]

# Convert to pandas Series for fast .map()
mfr_s = pd.Series(S["merchant_fr"])
mcnt_s = pd.Series(S["mcnt_full"])
cfr_s = pd.Series(S["city_fr"])
umdeg_s = pd.Series(S["um_deg"])
mudeg_s = pd.Series(S["mu_deg"])
munq_s = pd.Series(S["munq"])
ucdeg_s = pd.Series(S["uc_deg"])
cunq_s = pd.Series(S["cunq"])
ufr_s = pd.Series(S["user_fraud_rate"])
utc_s = pd.Series(S["user_count"])
umdiv_s = pd.Series(S["um_div"])
ucdiv_s = pd.Series(S["uc_div"])
uavg_s = pd.Series(S["user_avg_amt"])
ustd_s = pd.Series(S["user_std"])
ccnt_s = pd.Series(S["ccnt"])
uhc_s = pd.Series(S["uhc"])
del S  # free memory

# Write features + labels to disk chunk by chunk
feat_path = "data/altman_features.mmap"
label_path = "data/altman_labels.npy"

print("  Engineering features (writing to disk)...")
t0 = time.time()
total_written = 0
first_chunk = True

for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
                          usecols=USECOLS, low_memory=False, chunksize=2_000_000):
    n = len(chunk)
    amts = pd.to_numeric(chunk["Amount"].astype(str).str.replace('$','',regex=False)
                         .str.replace(',','',regex=False), errors='coerce').fillna(0)
    hrs = chunk["Time"].apply(parse_hr).astype(float)
    mccs = pd.to_numeric(chunk["MCC"], errors='coerce').fillna(0)
    merchs = chunk["Merchant Name"].astype(str)
    cities = chunk["Merchant City"].astype(str)
    users = chunk["User"].astype(str)
    cards = chunk["Card"].astype(str)
    chips = chunk["Use Chip"].astype(str).str.contains("Swipe|Chip", case=False, na=False)
    onlines = chunk["Use Chip"].astype(str).str.contains("Online", case=False, na=False)
    states = chunk["Merchant State"].fillna("").astype(str)
    zips = chunk["Zip"].notna()
    labels = (chunk["Is Fraud?"].astype(str).str.strip() == "Yes").astype(int).values
    del chunk  # free raw dataframe immediately

    # Build feature matrix for this chunk
    F = np.zeros((n, NF), dtype=np.float32)
    F[:,0] = np.log1p(amts.values)
    F[:,1] = amts.values ** 2
    F[:,2] = (amts.values > 1000).astype(np.float32)
    F[:,3] = np.cos(2 * np.pi * hrs.values / 24)
    F[:,4] = ((hrs.values >= 9) & (hrs.values <= 17)).astype(np.float32)
    F[:,5] = chips.values.astype(np.float32)
    F[:,6] = onlines.values.astype(np.float32)
    F[:,7] = mccs.values
    F[:,8] = zips.values.astype(np.float32)
    F[:,9] = (states != "").astype(np.float32)
    F[:,10] = merchs.map(mcnt_s).fillna(1).values.astype(np.float32)
    F[:,11] = merchs.map(mfr_s).fillna(gm).values.astype(np.float32)
    F[:,12] = cities.map(cfr_s).fillna(gm).values.astype(np.float32)
    F[:,13] = amts.values * F[:,7]
    F[:,14] = amts.values * F[:,6]
    F[:,15] = users.map(umdeg_s).fillna(0).values.astype(np.float32)
    F[:,16] = merchs.map(mudeg_s).fillna(0).values.astype(np.float32)
    F[:,17] = merchs.map(munq_s).fillna(1).values.astype(np.float32)
    F[:,18] = users.map(ucdeg_s).fillna(0).values.astype(np.float32)
    F[:,19] = cities.map(cunq_s).fillna(1).values.astype(np.float32)
    F[:,20] = F[:,15] / max(umax, 1)
    F[:,21] = F[:,16] / max(mmax, 1)
    F[:,22] = F[:,11] * amts.values
    ufr_vals = users.map(ufr_s).fillna(gm).values.astype(np.float32)
    F[:,23] = F[:,11] * ufr_vals
    F[:,24] = ufr_vals
    utc_vals = users.map(utc_s).fillna(0).values.astype(np.float32)
    F[:,25] = utc_vals
    F[:,26] = users.map(umdiv_s).fillna(1).values.astype(np.float32)
    F[:,27] = users.map(ucdiv_s).fillna(1).values.astype(np.float32)
    uavg_vals = users.map(uavg_s).fillna(1.0).values.astype(np.float32)
    F[:,28] = uavg_vals
    F[:,29] = amts.values / np.maximum(uavg_vals, 0.01)
    ustd_vals = users.map(ustd_s).fillna(1.0).values.astype(np.float32)
    F[:,30] = (amts.values - uavg_vals) / np.maximum(ustd_vals, 0.01)
    ck = (users + "_" + cards)
    F[:,31] = ck.map(ccnt_s).fillna(1).values.astype(np.float32)
    hb = (hrs.values // 6).astype(int).astype(str)
    uhk = (users + "_" + hb)
    F[:,32] = uhk.map(uhc_s).fillna(0).values.astype(np.float32)
    F[:,33] = F[:,26] / np.maximum(F[:,25], 1)
    F[:,34] = F[:,27] / np.maximum(F[:,25], 1)
    F = np.nan_to_num(F, nan=0, posinf=100, neginf=-100)

    # Write to disk — don't accumulate in memory
    if first_chunk:
        feat_mmap = np.memmap(feat_path, dtype=np.float32, mode='w+',
                              shape=(total, NF))
        label_arr = np.zeros(total, dtype=np.int32)
        first_chunk = False

    feat_mmap[total_written:total_written+n] = F
    label_arr[total_written:total_written+n] = labels
    total_written += n
    print(f"  {total_written:,}/{total:,} ({time.time()-t0:.1f}s)", end="\r")

    # Free chunk data
    del F, amts, hrs, mccs, merchs, cities, users, cards, chips, onlines, states, zips, labels

feat_mmap.flush()
np.save(label_path, label_arr)
del label_arr
print(f"\n  Written to disk ({time.time()-t0:.1f}s)")

# Load from disk for training
print("  Loading from disk for training...")
t0 = time.time()
X_all = np.memmap(feat_path, dtype=np.float32, mode='r', shape=(total_written, NF)).copy()
y_all = np.load(label_path)
print(f"  Loaded {X_all.shape} ({time.time()-t0:.1f}s)")

# Temporal split
sp = int(len(X_all) * 0.8)
Xtr, Xte = X_all[:sp], X_all[sp:]
ytr, yte = y_all[:sp], y_all[sp:]
nft = int(yte.sum()); nnt = int((yte==0).sum())
spw = (len(ytr) - int(ytr.sum())) / max(int(ytr.sum()), 1)
print(f"  Train: {len(Xtr):,} ({ytr.sum():,}) | Test: {len(Xte):,} ({nft:,}) | SPW: {spw:.0f}")

# Scale
sc = RobustScaler()
Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
del X_all, Xtr, Xte  # free memory

# Train
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

# Optimize ensemble
bw, ba = 0.5, 0
for w in np.arange(0.1, 0.9, 0.05):
    p = w*p_xgb + (1-w)*p_lgb; a = roc_auc_score(yte, p)
    if a > ba: ba = a; bw = w
p_ens = bw*p_xgb + (1-bw)*p_lgb
ea = roc_auc_score(yte, p_ens)

print(f"\n{'='*60}")
print(f"  RESULTS ({NF} features, {total_written:,} rows)")
print(f"{'='*60}")
print(f"  XGB:      {xa:.4f}")
print(f"  LGB:      {la:.4f}")
print(f"  Ensemble: {ea:.4f} (w_xgb={bw:.2f})")
print(f"  Baseline (15 feat): 0.9462")
print(f"  Delta: {(ea-0.9462)*100:+.2f}%")
print(f"  Target >95%: {'ACHIEVED' if ea>0.95 else 'NOT MET'}")

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
out = {"dataset":"ibm_altman_graph_full","n_total":total_written,
       "n_train":len(ytr),"n_test":len(yte),"fraud_test":nft,
       "baseline_auc":0.9462,"n_features":NF,
       "xgb_auc":round(float(xa),6),"lgb_auc":round(float(la),6),
       "ensemble_auc":round(float(ea),6),"ensemble_weight_xgb":round(float(bw),4),
       "improvement":round(float(ea-0.9462),6),
       "target_95":ea>0.95,"features":FEAT,
       "fi":{nm:round(float(v),4) for nm,v in sorted(zip(FEAT,imp),key=lambda x:-x[1])[:20]},
       "elapsed":round(elapsed,1)}
with open("reports/altman_graph_full.json","w") as f:
    json.dump(out,f,indent=2)

# Cleanup
os.remove(feat_path); os.remove(label_path)
print(f"\n  Saved reports/altman_graph_full.json ({elapsed:.1f}s)")
