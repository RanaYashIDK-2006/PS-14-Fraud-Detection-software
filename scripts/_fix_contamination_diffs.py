# -*- coding: utf-8 -*-
"""
Fix for the feature-diff stats in reports/independent_validation.json:
the original comparison mixed units (standardized audit rates vs raw
chronological rates). This recomputes apples-to-apples RAW-vs-RAW diffs.

The SCORING deltas (AUC / recall / FPR) in the JSON are unaffected and remain
as measured.

Audit raw rates are recovered without the model: the file-order merchant/city
state is constant for every test row of a given merchant/city (state = all
pre-test rows), so audit_rate = pre-test totals / pre-test counts.
"""
import gc
import json
import time

import numpy as np
import pandas as pd

T0 = time.time()
CSV = "data/credit_card_transactions-ibm_v2.csv"
TOTAL = 24386900
N_TRAIN = int(TOTAL * 0.60)
N_VAL = int(TOTAL * 0.20)
T0_TEST = N_TRAIN + N_VAL
SAMP = 500_000


def log(m):
    print(f"[{time.time()-T0:5.0f}s] {m}", flush=True)


log("loading 6 cols...")
small = pd.read_csv(CSV, usecols=["Year", "Month", "Day", "Merchant Name", "Merchant City", "Is Fraud?"],
                    low_memory=False)
small["date"] = small["Year"].astype(int) * 10000 + small["Month"].astype(int) * 100 + small["Day"].astype(int)
small["is_f"] = small["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
small["Merchant Name"] = small["Merchant Name"].astype(str)
small["Merchant City"] = small["Merchant City"].astype(str)
small = small[["Merchant Name", "Merchant City", "date", "is_f"]]

# --- audit (file-order) raw rates: pre-test totals per merchant/city ---------
log("building pre-test merchant/city totals (audit state)...")
pretest = small.iloc[:T0_TEST]
m_tot = pretest.groupby("Merchant Name")["is_f"].agg(["size", "sum"])
c_tot = pretest.groupby("Merchant City")["is_f"].agg(["size", "sum"])
m_audit_rate = (m_tot["sum"] / m_tot["size"]).to_dict()
c_audit_rate = (c_tot["sum"] / c_tot["size"]).to_dict()
del pretest
gc.collect()

# --- chronological (strictly-before-date) rates ------------------------------
log("building per-merchant date-cumulative tables...")
mg = small.groupby(["Merchant Name", "date"])["is_f"].agg(["size", "sum"]).reset_index()
mg.columns = ["key", "date", "tx", "fr"]
mg = mg.sort_values(["key", "date"]).reset_index(drop=True)
mg["ctx"] = mg.groupby("key")["tx"].cumsum() - mg["tx"]
mg["cfr_"] = mg.groupby("key")["fr"].cumsum() - mg["fr"]
mg_dates = {k: v["date"].values for k, v in mg.groupby("key")}
mg_ctx = {k: v["ctx"].values for k, v in mg.groupby("key")}
mg_cfr = {k: v["cfr_"].values for k, v in mg.groupby("key")}
del mg
gc.collect()

log("building per-city date-cumulative tables...")
cg = small.groupby(["Merchant City", "date"])["is_f"].agg(["size", "sum"]).reset_index()
cg.columns = ["key", "date", "tx", "fr"]
cg = cg.sort_values(["key", "date"]).reset_index(drop=True)
cg["ctx"] = cg.groupby("key")["tx"].cumsum() - cg["tx"]
cg["cfr_"] = cg.groupby("key")["fr"].cumsum() - cg["fr"]
cg_dates = {k: v["date"].values for k, v in cg.groupby("key")}
cg_ctx = {k: v["ctx"].values for k, v in cg.groupby("key")}
cg_cfr = {k: v["cfr_"].values for k, v in cg.groupby("key")}
del cg
gc.collect()

samp_merch = small["Merchant Name"].values[T0_TEST:T0_TEST + SAMP]
samp_city = small["Merchant City"].values[T0_TEST:T0_TEST + SAMP]
samp_date = small["date"].values[T0_TEST:T0_TEST + SAMP]
del small
gc.collect()


def chrono_rate(keys, dates, tbl_dates, tbl_ctx, tbl_cfr):
    out = np.zeros(len(keys), dtype=np.float64)
    for i in range(len(keys)):
        dd = tbl_dates.get(keys[i])
        if dd is None or len(dd) == 0:
            continue
        idx = int(np.searchsorted(dd, dates[i], side="left")) - 1
        if idx >= 0:
            out[i] = tbl_cfr[keys[i]][idx] / max(tbl_ctx[keys[i]][idx], 1)
    return out


log("computing raw audit + chronological rates for the 500K test window...")
mfr_audit = np.array([m_audit_rate.get(m, 0.0) for m in samp_merch])
cfr_audit = np.array([c_audit_rate.get(c, 0.0) for c in samp_city])
mfr_chrono = chrono_rate(samp_merch, samp_date, mg_dates, mg_ctx, mg_cfr)
cfr_chrono = chrono_rate(samp_city, samp_date, cg_dates, cg_ctx, cg_cfr)

m_diff = np.abs(mfr_audit - mfr_chrono)
c_diff = np.abs(cfr_audit - cfr_chrono)

result = {
    "merch_fraud_rate_raw_diff": {
        "mean_abs_diff": round(float(m_diff.mean()), 6),
        "max_abs_diff": round(float(m_diff.max()), 6),
        "pct_rows_changed": round(float((m_diff > 1e-9).mean() * 100), 2),
        "audit_mean": round(float(mfr_audit.mean()), 6),
        "chrono_mean": round(float(mfr_chrono.mean()), 6),
    },
    "city_fraud_rate_raw_diff": {
        "mean_abs_diff": round(float(c_diff.mean()), 6),
        "max_abs_diff": round(float(c_diff.max()), 6),
        "pct_rows_changed": round(float((c_diff > 1e-9).mean() * 100), 2),
        "audit_mean": round(float(cfr_audit.mean()), 6),
        "chrono_mean": round(float(cfr_chrono.mean()), 6),
    },
}
log(f"merch_fraud_rate RAW: mean|diff|={result['merch_fraud_rate_raw_diff']['mean_abs_diff']} "
    f"max={result['merch_fraud_rate_raw_diff']['max_abs_diff']} "
    f"changed={result['merch_fraud_rate_raw_diff']['pct_rows_changed']}%")
log(f"city_fraud_rate  RAW: mean|diff|={result['city_fraud_rate_raw_diff']['mean_abs_diff']} "
    f"max={result['city_fraud_rate_raw_diff']['max_abs_diff']} "
    f"changed={result['city_fraud_rate_raw_diff']['pct_rows_changed']}%")

path = "reports/independent_validation.json"
r = json.load(open(path))
c = r["chronological_contamination"]
# mark the previously-reported (mixed-unit) numbers as superseded
for feat in ("merch_fraud_rate", "city_fraud_rate"):
    c[feat]["superseded_note"] = "previous values compared standardized audit values vs raw chronological rates (mixed units); replaced by *_raw_diff below"
c.pop("merch_fraud_rate", None)
c.pop("city_fraud_rate", None)
c.update(result)
r["chronological_contamination"] = c
with open(path, "w") as f:
    json.dump(r, f, indent=1, default=str)
log("patched reports/independent_validation.json")