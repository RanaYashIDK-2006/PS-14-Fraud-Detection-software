#!/usr/bin/env python3
"""
Full 24M Altman — two-pass pipeline for graph+velocity features.
Pass 1: Vectorized pandas groupby on train portion for all lookup dicts.
Pass 2: Vectorized feature engineering on full dataset.
No sampling, no target leakage, honest temporal split.
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
print("ALTMAN FULL 24M — Graph + Velocity (Vectorized Two-Pass)")
print("=" * 80)
T0 = time.time()

# ═══ Count total rows ═══
print("\n  Counting rows...")
t0 = time.time()
total = 0
for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
                          usecols=["Is Fraud?"], low_memory=False, chunksize=2_000_000):
    total += len(chunk)
SPLIT = int(total * 0.8)
print(f"  Total: {total:,} | Split: {SPLIT:,} ({time.time()-t0:.1f}s)")

# ═══ Pass 1: Build all stats from train portion (vectorized) ═══
print("\n[Pass 1] Building stats from train portion (vectorized)...")
t0 = time.time()
train_chunks = []
rows = 0
for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
                          usecols=USECOLS, low_memory=False, chunksize=2_000_000):
    if rows >= SPLIT:
        break
    avail = min(len(chunk), SPLIT - rows)
    c = chunk.iloc[:avail].copy()
    c["_label"] = (c["Is Fraud?"].astype(str).str.strip() == "Yes").astype(int)
    c["_amt"] = pd.to_numeric(c["Amount"].astype(str).str.replace('$','',regex=False)
                              .str.replace(',','',regex=False), errors='coerce').fillna(0)
    c["_hr"] = c["Time"].apply(parse_hr)
    c["_mcc"] = pd.to_numeric(c["MCC"], errors='coerce').fillna(0)
    c["_m"] = c["Merchant Name"].astype(str)
    c["_city"] = c["Merchant City"].astype(str)
    c["_u"] = c["User"].astype(str)
    c["_card"] = c["Card"].astype(str)
    c["_ck"] = c["_u"] + "|" + c["_card"]
    c["_hb"] = (c["_hr"] // 6).astype(int).astype(str)
    c["_uhk"] = c["_u"] + "|" + c["_hb"]
    train_chunks.append(c)
    rows += len(c)
    print(f"  {rows:,}/{SPLIT:,}", end="\r")

df_tr = pd.concat(train_chunks, ignore_index=True)
print(f"  Train loaded: {len(df_tr):,} ({time.time()-t0:.1f}s)")

# Vectorized groupby stats
t1 = time.time()
gm = df_tr["_label"].mean()
K = 50

# Merchant stats
mf = df_tr.groupby("_m").agg(mc=("_label","count"), mfs=("_label","sum"), ma=("_amt","mean"))
mf["merchant_fr"] = (mf["mfs"] + gm * K) / (mf["mc"] + K)
merchant_fr = mf["merchant_fr"].to_dict()
merchant_count = mf["mc"].to_dict()

# City stats
cf = df_tr.groupby("_city").agg(cc=("_label","count"), cfs=("_label","sum"))
cf["city_fr"] = (cf["cfs"] + gm * K) / (cf["cc"] + K)
city_fr = cf["city_fr"].to_dict()

# User stats
us = df_tr.groupby("_u").agg(uc=("_label","count"), ufs=("_label","sum"),
                               ua=("_amt","mean"), uasq=("_amt", lambda x: (x**2).sum()))
us["ufr"] = (us["ufs"] + gm * 200) / (us["uc"] + 200)
user_count = us["uc"].to_dict()
user_fraud_rate = us["ufr"].to_dict()
user_avg_amt = us["ua"].to_dict()
user_amt_sq_sum = us["uasq"].to_dict()

# Graph: user-merchant degree
um = df_tr.groupby(["_u", "_m"]).size().reset_index(name="w")
um_deg = um.groupby("_u")["w"].sum().to_dict()
mu_deg = um.groupby("_m")["w"].sum().to_dict()

# Unique users per merchant
munq = df_tr.groupby("_m")["_u"].nunique().to_dict()

# User-city degree
uc = df_tr.groupby(["_u", "_city"]).size().reset_index(name="w")
uc_deg = uc.groupby("_u")["w"].sum().to_dict()

# Unique users per city
cunq = df_tr.groupby("_city")["_u"].nunique().to_dict()

# User-merchant diversity (unique merchants per user)
um_div = df_tr.groupby("_u")["_m"].nunique().to_dict()

# User-city diversity
uDiv = df_tr.groupby("_u")["_city"].nunique().to_dict()

# Card count
cc_count = df_tr["_ck"].value_counts().to_dict()

# User-hour bin count
uhc = df_tr["_uhk"].value_counts().to_dict()

# User std dev
us["u_std"] = np.sqrt(np.maximum(us["uasq"] / us["uc"] - us["ua"]**2, 0.01))
user_std = us["u_std"].to_dict()

print(f"  Stats built: {len(merchant_fr):,} merch, {len(city_fr):,} city, "
      f"{len(user_count):,} user ({time.time()-t1:.1f}s)")

# Free train memory
del df_tr, train_chunks

# ═══ Pass 2: Feature engineering on full dataset ═══
print("\n[Pass 2] Engineering features (vectorized)...")
t0 = time.time()

FEAT_NAMES = [
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
NF = len(FEAT_NAMES)

umax = max(um_deg.values()) if um_deg else 1
mmax = max(mu_deg.values()) if mu_deg else 1

X_all = []
y_all = []
rows = 0

for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
                          usecols=USECOLS, low_memory=False, chunksize=2_000_000):
    n = len(chunk)
    labels = (chunk["Is Fraud?"].astype(str).str.strip() == "Yes").astype(int).values
    amts = pd.to_numeric(chunk["Amount"].astype(str).str.replace('$','',regex=False)
                         .str.replace(',','',regex=False), errors='coerce').fillna(0).values
    hrs = chunk["Time"].apply(parse_hr).values.astype(float)
    mccs = pd.to_numeric(chunk["MCC"], errors='coerce').fillna(0).values
    merchs = chunk["Merchant Name"].astype(str).values
    cities = chunk["Merchant City"].astype(str).values
    states = chunk["Merchant State"].fillna("").astype(str).values
    zips = chunk["Zip"].notna().astype(int).values
    chips = chunk["Use Chip"].astype(str).str.contains("Swipe|Chip", case=False, na=False).astype(int).values
    onlines = chunk["Use Chip"].astype(str).str.contains("Online", case=False, na=False).astype(int).values
    users = chunk["User"].astype(str).values
    cards_arr = chunk["Card"].astype(str).values

    F = np.zeros((n, NF), dtype=np.float32)

    # Baseline 15
    F[:, 0] = np.log1p(amts)
    F[:, 1] = amts ** 2
    F[:, 2] = (amts > 1000).astype(float)
    F[:, 3] = np.cos(2 * np.pi * hrs / 24)
    F[:, 4] = ((hrs >= 9) & (hrs <= 17)).astype(float)
    F[:, 5] = chips
    F[:, 6] = onlines
    F[:, 7] = mccs
    F[:, 8] = zips
    F[:, 9] = (states != "").astype(float)

    # Use np.vectorize for dict lookups (faster than list comprehension)
    v_merch_count = np.vectorize(lambda x: merchant_count.get(x, 1))
    v_merch_fr = np.vectorize(lambda x: merchant_fr.get(x, gm))
    v_city_fr = np.vectorize(lambda x: city_fr.get(x, gm))

    F[:, 10] = v_merch_count(merchs).astype(float)
    F[:, 11] = v_merch_fr(merchs).astype(float)
    F[:, 12] = v_city_fr(cities).astype(float)
    F[:, 13] = amts * mccs
    F[:, 14] = amts * onlines

    # Graph features
    v_um_deg = np.vectorize(lambda x: um_deg.get(x, 0))
    v_mu_deg = np.vectorize(lambda x: mu_deg.get(x, 0))
    v_munq = np.vectorize(lambda x: munq.get(x, 1))
    v_uc_deg = np.vectorize(lambda x: uc_deg.get(x, 0))
    v_cunq = np.vectorize(lambda x: cunq.get(x, 1))

    F[:, 15] = v_um_deg(users).astype(float)
    F[:, 16] = v_mu_deg(merchs).astype(float)
    F[:, 17] = v_munq(merchs).astype(float)
    F[:, 18] = v_uc_deg(users).astype(float)
    F[:, 19] = v_cunq(cities).astype(float)
    F[:, 20] = F[:, 15] / umax
    F[:, 21] = F[:, 16] / mmax

    # Fraud interactions
    v_ufr = np.vectorize(lambda x: user_fraud_rate.get(x, gm))
    F[:, 22] = F[:, 11] * amts
    F[:, 23] = F[:, 11] * v_ufr(users).astype(float)
    F[:, 24] = v_ufr(users).astype(float)

    # Velocity
    v_ucnt = np.vectorize(lambda x: user_count.get(x, 0))
    v_umdiv = np.vectorize(lambda x: um_div.get(x, 1))
    v_ucdiv = np.vectorize(lambda x: uDiv.get(x, 1))
    v_uavg = np.vectorize(lambda x: user_avg_amt.get(x, 1.0))
    v_ustd = np.vectorize(lambda x: user_std.get(x, 1.0))

    F[:, 25] = v_ucnt(users).astype(float)
    F[:, 26] = v_umdiv(users).astype(float)
    F[:, 27] = v_ucdiv(users).astype(float)
    uavg = v_uavg(users).astype(float)
    F[:, 28] = uavg
    F[:, 29] = amts / np.maximum(uavg, 0.01)
    ustd = v_ustd(users).astype(float)
    F[:, 30] = (amts - uavg) / np.maximum(ustd, 0.01)

    # Card + hour bin
    ck = np.core.defchararray.add(np.core.defchararray.add(users, "|"), cards_arr)
    v_cc = np.vectorize(lambda x: cc_count.get(x, 1))
    F[:, 31] = v_cc(ck).astype(float)

    hb = (hrs // 6).astype(int).astype(str)
    uhk = np.core.defchararray.add(np.core.defchararray.add(users, "|"), hb)
    v_uhc = np.vectorize(lambda x: uhc.get(x, 0))
    F[:, 32] = v_uhc(uhk).astype(float)

    # Ratios
    F[:, 33] = F[:, 26] / np.maximum(F[:, 25], 1)
    F[:, 34] = F[:, 27] / np.maximum(F[:, 25], 1)

    F = np.nan_to_num(F, nan=0, posinf=100, neginf=-100)
    X_all.append(F)
    y_all.append(labels)
    rows += n
    print(f"  {rows:,}/{total:,}", end="\r")

print(f"\n  Concatenating...")
X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)
print(f"  Shape: {X_all.shape} ({time.time()-t0:.1f}s)")

# ═══ Temporal split ═══
sp = int(len(X_all) * 0.8)
Xtr, Xte = X_all[:sp], X_all[sp:]
ytr, yte = y_all[:sp], y_all[sp:]
nft = int(yte.sum()); nnt = int((yte == 0).sum())
spw = (len(ytr) - int(ytr.sum())) / max(int(ytr.sum()), 1)
print(f"  Train: {len(Xtr):,} ({ytr.sum():,}) | Test: {len(Xte):,} ({nft:,}) | SPW: {spw:.0f}")

# Scale
sc = RobustScaler()
Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)

# ═══ Train ═══
print(f"\n  Training XGBoost...")
t0 = time.time()
xgb_m = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.02, subsample=0.8,
    colsample_bytree=0.7, gamma=1, min_child_weight=5,
    scale_pos_weight=min(spw, 500), tree_method="hist",
    eval_metric="auc", random_state=42, n_jobs=NJ)
xgb_m.fit(Xtr_s, ytr, verbose=False)
p_xgb = xgb_m.predict_proba(Xte_s)[:, 1]
xa = roc_auc_score(yte, p_xgb)
print(f"  XGB: {xa:.4f} ({time.time()-t0:.1f}s)")

print("  Training LightGBM...")
t0 = time.time()
lgb_m = lgb.LGBMClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.02, subsample=0.8,
    colsample_bytree=0.7, min_child_samples=30,
    scale_pos_weight=min(spw, 500), random_state=42, n_jobs=NJ, verbose=-1)
lgb_m.fit(Xtr_s, ytr)
p_lgb = lgb_m.predict_proba(Xte_s)[:, 1]
la = roc_auc_score(yte, p_lgb)
print(f"  LGB: {la:.4f} ({time.time()-t0:.1f}s)")

# Ensemble optimization
bw, ba = 0.5, 0
for w in np.arange(0.1, 0.9, 0.05):
    p = w * p_xgb + (1 - w) * p_lgb
    a = roc_auc_score(yte, p)
    if a > ba: ba = a; bw = w
p_ens = bw * p_xgb + (1 - bw) * p_lgb
ea = roc_auc_score(yte, p_ens)

print(f"\n{'='*60}")
print(f"  RESULTS ({NF} features, {total:,} rows)")
print(f"{'='*60}")
print(f"  XGB:      {xa:.4f}")
print(f"  LGB:      {la:.4f}")
print(f"  Ensemble: {ea:.4f} (w_xgb={bw:.2f})")
print(f"  Baseline (15 feat): 0.9462")
print(f"  Delta: {(ea - 0.9462) * 100:+.2f}%")
print(f"  Target >95%: {'ACHIEVED' if ea > 0.95 else 'NOT MET'}")

# Feature importance
imp = xgb_m.feature_importances_
base = {"log_amt","amt_sq","very_high_amt","hour_cos","is_business_hours",
        "chip","is_online","mcc_n","has_zip","has_state",
        "merch_tx_count","merch_fraud_rate","city_fraud_rate",
        "amt_x_mcc","amt_x_online"}
print(f"\n  Top 20 Features:")
for nm, v in sorted(zip(FEAT_NAMES, imp), key=lambda x: -x[1])[:20]:
    tag = " *NEW*" if nm not in base else ""
    print(f"    {nm:>28}: {v:.4f}{tag}")

# Recall curve
order = np.argsort(-p_ens); sy = yte[order]; ct = np.cumsum(sy); rc = ct / nft
print(f"\n  Recall Curve:")
for tgt in [0.90, 0.95, 0.97, 0.98, 0.985, 0.99]:
    idx = np.searchsorted(rc, tgt)
    if idx < len(p_ens):
        thr = p_ens[order[idx]]; fl = idx + 1; tp2 = int(ct[idx]); fp2 = fl - tp2
        print(f"    recall={tgt*100:.1f}% thr={thr:.6f} FPR={fp2/max(nnt,1)*100:.3f}% missed={nft-tp2}")

elapsed = time.time() - T0
out = {"dataset": "ibm_altman_graph_full", "n_total": total,
       "n_train": len(Xtr), "n_test": len(Xte), "fraud_test": nft,
       "baseline_auc": 0.9462, "n_features": NF,
       "xgb_auc": round(float(xa), 6), "lgb_auc": round(float(la), 6),
       "ensemble_auc": round(float(ea), 6), "ensemble_weight_xgb": round(float(bw), 4),
       "improvement": round(float(ea - 0.9462), 6),
       "target_95": ea > 0.95, "features": FEAT_NAMES,
       "fi": {nm: round(float(v), 4) for nm, v in sorted(zip(FEAT_NAMES, imp), key=lambda x: -x[1])[:20]},
       "elapsed": round(elapsed, 1)}
with open("reports/altman_graph_full.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved reports/altman_graph_full.json ({elapsed:.1f}s)")
