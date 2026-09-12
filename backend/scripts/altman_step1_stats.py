#!/usr/bin/env python3
"""Step 1: Build lookup dicts from the full 24M Altman train portion."""
import time, json, pickle
import numpy as np
import pandas as pd

USECOLS = ["User","Card","Time","Amount","Use Chip","Merchant Name",
           "Merchant City","Merchant State","Zip","MCC","Is Fraud?"]

def parse_hr(t):
    try:
        s = str(t).strip(); p = s.replace('.',':').split(':'); h = int(p[0])
        if 'pm' in s.lower() and h != 12: h += 12
        elif 'am' in s.lower() and h == 12: h = 0
        return h
    except: return 12

print("Step 1: Building stats from full 24M train...")
T0 = time.time()

# Count total
print("  Counting rows...")
total = 0
for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
                          usecols=["Is Fraud?"], low_memory=False, chunksize=2_000_000):
    total += len(chunk)
SPLIT = int(total * 0.8)
print(f"  Total: {total:,} | Split: {SPLIT:,}")

# Load train portion in chunks and build stats
print("  Building stats...")
rows = 0
chunks_list = []
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
    c["_m"] = c["Merchant Name"].astype(str)
    c["_city"] = c["Merchant City"].astype(str)
    c["_u"] = c["User"].astype(str)
    c["_card"] = c["Card"].astype(str)
    c["_ck"] = c["_u"] + "_" + c["_card"]
    c["_hb"] = (c["_hr"] // 6).astype(int).astype(str)
    c["_uhk"] = c["_u"] + "_" + c["_hb"]
    chunks_list.append(c)
    rows += avail
    print(f"  {rows:,}/{SPLIT:,}", end="\r")

df = pd.concat(chunks_list, ignore_index=True)
del chunks_list
print(f"  Loaded {len(df):,} train rows ({time.time()-T0:.1f}s)")

t1 = time.time()
gm = df["_label"].mean(); K = 50

# Merchant stats
mf = df.groupby("_m").agg(mc=("_label","count"), mfs=("_label","sum"), ma=("_amt","mean"))
mf["fr"] = (mf["mfs"] + gm*K) / (mf["mc"] + K)
merchant_fr = mf["fr"].to_dict()
merchant_count = mf["mc"].to_dict()

# City stats
cf = df.groupby("_city").agg(cc=("_label","count"), cfs=("_label","sum"))
cf["fr"] = (cf["cfs"] + gm*K) / (cf["cc"] + K)
city_fr = cf["fr"].to_dict()

# User stats
us = df.groupby("_u").agg(uc=("_label","count"), ufs=("_label","sum"),
                           ua=("_amt","mean"), uasq=("_amt", lambda x: (x**2).sum()))
us["ufr"] = (us["ufs"] + gm*200) / (us["uc"] + 200)
us["ustd"] = np.sqrt(np.maximum(us["uasq"]/us["uc"] - us["ua"]**2, 0.01))
user_count = us["uc"].to_dict()
user_fraud_rate = us["ufr"].to_dict()
user_avg_amt = us["ua"].to_dict()
user_std = us["ustd"].to_dict()

# Graph features
um = df.groupby(["_u","_m"]).size().reset_index(name="w")
um_deg = um.groupby("_u")["w"].sum().to_dict()
mu_deg = um.groupby("_m")["w"].sum().to_dict()
munq = df.groupby("_m")["_u"].nunique().to_dict()

uc = df.groupby(["_u","_city"]).size().reset_index(name="w")
uc_deg = uc.groupby("_u")["w"].sum().to_dict()
cunq = df.groupby("_city")["_u"].nunique().to_dict()

um_div = df.groupby("_u")["_m"].nunique().to_dict()
uc_div = df.groupby("_u")["_city"].nunique().to_dict()
mcnt_full = df["_m"].value_counts().to_dict()

ccnt = df["_ck"].value_counts().to_dict()
uhc = df["_uhk"].value_counts().to_dict()

umax = max(um_deg.values()) if um_deg else 1
mmax = max(mu_deg.values()) if mu_deg else 1

print(f"  Stats built ({time.time()-t1:.1f}s)")

# Save
stats = {
    "gm": gm, "total": total, "split": SPLIT,
    "merchant_fr": merchant_fr, "merchant_count": merchant_count,
    "city_fr": city_fr,
    "user_count": user_count, "user_fraud_rate": user_fraud_rate,
    "user_avg_amt": user_avg_amt, "user_std": user_std,
    "um_deg": um_deg, "mu_deg": mu_deg, "munq": munq,
    "uc_deg": uc_deg, "cunq": cunq,
    "um_div": um_div, "uc_div": uc_div,
    "mcnt_full": mcnt_full, "ccnt": ccnt, "uhc": uhc,
    "umax": umax, "mmax": mmax,
}

with open("data/altman_stats.pkl", "wb") as f:
    pickle.dump(stats, f)
print(f"  Saved data/altman_stats.pkl ({time.time()-T0:.1f}s total)")
