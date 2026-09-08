#!/usr/bin/env python3
"""
PS-14 CHECK #15: SEGMENT-LEVEL PERFORMANCE & BIAS AUDIT

Evaluates the locked model separately across transaction segments
(merchant, city, online/chip/swipe, amount bands, entity history depth)
on the UNTOUCHED temporal test set, reporting sample sizes and CIs so
tiny segments are not over-interpreted.

Identical protocol to the main forensic audit:
  causal expanding features -> temporal 60/20/20 split -> XGB trained with
  early stopping on validation only -> threshold locked from validation
  (FPR < 0.9%) -> evaluate ONCE on the final test.
"""
import json, os, sys, gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
REPORT = ROOT / "reports" / "segment_audit.json"
REPORT.parent.mkdir(parents=True, exist_ok=True)

T0 = time.time()
SEED = 42
np.random.seed(SEED)

def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)

def _cvt(o):
    if isinstance(o, (bool, np.bool_)): return bool(o)
    if isinstance(o, (int, np.integer)): return int(o)
    if isinstance(o, (float, np.floating)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: _cvt(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_cvt(v) for v in o]
    return str(o)

CSV = "data/credit_card_transactions-ibm_v2.csv"
CHUNK = 2_000_000
USECOLS = ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
           "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
           "Errors?", "Is Fraud?"]

FEATURE_NAMES = [
    "log_amt", "amt_sq", "year", "month", "day", "chip", "is_online", "mcc_n",
    "has_zip", "has_state", "user_tx_count", "merch_fraud_rate", "city_fraud_rate",
    "very_high_amt", "amt_x_mcc", "amt_x_online", "merch_popularity",
    "user_fraud_rate", "amt_ratio", "amt_zscore", "amt_acceleration",
    "mfr_x_ufr", "city_popularity", "user_merch_diversity", "amt_x_chip",
]

def expanding_features_fixed(df_chunk, state, pop_state=None):
    n = len(df_chunk)
    amt = pd.to_numeric(df_chunk["Amount"].str.replace("$", "", regex=False), errors="coerce").fillna(0).values.astype(np.float32)
    users = df_chunk["User"].astype(str).values
    merchs = df_chunk["Merchant Name"].astype(str).values
    cities = df_chunk["Merchant City"].astype(str).values
    mccs = df_chunk["MCC"].fillna(0).astype(int).values
    use_chip = df_chunk["Use Chip"].fillna("Online Transaction").values
    year = df_chunk["Year"].fillna(2019).values.astype(float)
    month = df_chunk["Month"].fillna(1).values.astype(float)
    day = df_chunk["Day"].fillna(1).values.astype(float)
    zip_val = df_chunk["Zip"].fillna(0).values
    has_zip = (np.array(zip_val, dtype=float) > 0).astype(float)
    merchant_state = df_chunk["Merchant State"].fillna("").values
    has_state = (np.array(merchant_state) != "").astype(float)
    is_fraud = df_chunk["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int).values

    log_amt = np.log1p(amt)
    amt_sq = amt ** 2
    chip = np.where(np.array(use_chip) == "Chip Transaction", 1.0,
            np.where(np.array(use_chip) == "Swipe Transaction", 0.5, 0.0))
    is_online = np.where(np.array(use_chip) == "Online Transaction", 1.0, 0.0)
    mcc_n = mccs.astype(np.float32) / 6000.0

    utc = np.zeros(n, dtype=np.float32); ufr = np.zeros(n, dtype=np.float32)
    uavg = np.zeros(n, dtype=np.float32); ustd = np.zeros(n, dtype=np.float32)
    mfr = np.zeros(n, dtype=np.float32); cfr = np.zeros(n, dtype=np.float32)
    mtc = np.zeros(n, dtype=np.float32); accel = np.zeros(n, dtype=np.float32)
    ctc = np.zeros(n, dtype=np.float32)

    utc_s, ufc_s, uta_s, uas_s, ula_s = (state[k] for k in
        ["user_tx_count", "user_fraud_count", "user_total_amt", "user_amt_sq", "user_last_amt"])
    mtc_s, mfc_s, ctc_s, cfc_s = (state[k] for k in
        ["merch_tx_count", "merch_fraud_count", "city_tx_count", "city_fraud_count"])

    if pop_state is None:
        pop_state = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0, "total_amt_sq": 0.0, "n_users": 0}
    _ptx, _pfraud, _pamt, _pamtsq, _pn = (pop_state[k] for k in
        ["total_tx", "total_fraud", "total_amt", "total_amt_sq", "n_users"])
    pop_fraud_rate = _pfraud / max(_ptx, 1)
    pop_avg_amt = _pamt / max(_ptx, 1)
    pop_std_amt = max((_pamtsq / max(_ptx, 1)) - pop_avg_amt ** 2, 1e-10) ** 0.5

    for i in range(n):
        u, m, c = users[i], merchs[i], cities[i]
        a = float(amt[i])
        is_new_user = u not in utc_s
        utx = utc_s.get(u, 0); utc[i] = utx
        uf = ufc_s.get(u, 0); ut = uta_s.get(u, 0.0); us = uas_s.get(u, 0.0)
        if is_new_user and _ptx > 0:
            ufr[i] = pop_fraud_rate; uavg[i] = pop_avg_amt; ustd[i] = pop_std_amt; accel[i] = 0.0
        else:
            ufr[i] = uf / max(utx, 1)
            ua = ut / max(utx, 1); uavg[i] = ua
            uv = max(us / max(utx, 1) - ua * ua, 1e-10); ustd[i] = uv ** 0.5
            prev = ula_s.get(u, a)
            accel[i] = abs(a - prev) / max(prev, 0.01) if prev > 0 else 0.0
        mt = mtc_s.get(m, 0); mtc[i] = mt
        mfr[i] = mfc_s.get(m, 0) / max(mt, 1)
        ct = ctc_s.get(c, 0); ctc[i] = ct
        cfr[i] = cfc_s.get(c, 0) / max(ct, 1)
        utc_s[u] = utx + 1; ufc_s[u] = uf + int(is_fraud[i])
        uta_s[u] = ut + a; uas_s[u] = us + a * a; ula_s[u] = a
        mtc_s[m] = mt + 1; mfc_s[m] = mfc_s.get(m, 0) + int(is_fraud[i])
        ctc_s[c] = ct + 1; cfc_s[c] = cfc_s.get(c, 0) + int(is_fraud[i])
        _ptx += 1; _pfraud += int(is_fraud[i]); _pamt += a; _pamtsq += a * a
        if is_new_user: _pn += 1

    pop_state.update(total_tx=_ptx, total_fraud=_pfraud, total_amt=_pamt, total_amt_sq=_pamtsq, n_users=_pn)

    very_high_amt = (amt > 5000).astype(float)
    amt_x_mcc = amt * mcc_n; amt_x_online = amt * is_online; amt_x_chip = amt * chip
    mfr_x_ufr = mfr * ufr
    merch_pop = np.minimum(np.log1p(mtc), 5.0)
    city_pop = np.minimum(np.log1p(ctc), 5.0)
    amt_ratio = amt / np.maximum(uavg, 0.01)
    amt_zscore = (amt - uavg) / np.maximum(ustd, 0.01)
    umdiv = np.minimum(utc / np.maximum(mtc, 1), 10.0)

    F = np.column_stack([
        log_amt, amt_sq, year, month, day, chip, is_online, mcc_n,
        has_zip, has_state, utc, mfr, cfr, very_high_amt, amt_x_mcc,
        amt_x_online, merch_pop, ufr, amt_ratio, amt_zscore, accel,
        mfr_x_ufr, city_pop, umdiv, amt_x_chip,
    ]).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=10.0, neginf=-10.0)


def compute_features(df_slice, state, pop_state):
    chunks = []
    for start in range(0, len(df_slice), CHUNK):
        chunks.append(expanding_features_fixed(df_slice.iloc[start:min(start+CHUNK, len(df_slice))], state, pop_state))
    return np.vstack(chunks)


# ============================================================
log("=" * 70)
log("CHECK #15: SEGMENT PERFORMANCE & BIAS AUDIT")
log("=" * 70)

log("Loading dataset...")
parts = []
for i, chunk in enumerate(pd.read_csv(CSV, usecols=USECOLS, low_memory=False, chunksize=CHUNK)):
    parts.append(chunk)
df = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
total = len(df)
fraud_mask = df["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
log(f"  Total: {total:,} rows, Fraud: {int(fraud_mask.sum()):,} ({fraud_mask.mean()*100:.3f}%)")

n_train = int(total * 0.60); n_val = int(total * 0.20)
n_test = total - n_train - n_val

state = {k: {} for k in ["user_tx_count", "user_fraud_count", "user_total_amt",
                          "user_amt_sq", "user_last_amt", "merch_tx_count",
                          "merch_fraud_count", "city_tx_count", "city_fraud_count"]}
pop = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0, "total_amt_sq": 0.0, "n_users": 0}

t1 = time.time(); log("  Computing train features...")
X_train = compute_features(df.iloc[:n_train], state, pop)
y_train = fraud_mask.iloc[:n_train].values
log(f"  Train: {X_train.shape} ({time.time()-t1:.0f}s)")

log("  Computing val features...")
X_val = compute_features(df.iloc[n_train:n_train+n_val], state, pop)
y_val = fraud_mask.iloc[n_train:n_train+n_val].values

log("  Computing test features...")
X_test = compute_features(df.iloc[n_train+n_val:], state, pop)
y_test = fraud_mask.iloc[n_train+n_val:].values
log(f"  Test: {X_test.shape}")

# Keep raw segment metadata for the test slice
test_raw = df.iloc[n_train+n_val:].copy()
del df; gc.collect()

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_val_s = sc.transform(X_val)
X_test_s = sc.transform(X_test)
del X_train, X_val, X_test; gc.collect()

spw = max(1, int((y_train == 0).sum() / max(int(y_train.sum()), 1)))
model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, scale_pos_weight=min(spw, 30),
    gamma=1, min_child_weight=3, random_state=SEED, n_jobs=4,
    eval_metric="auc", early_stopping_rounds=30)
log("  Training XGB...")
model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=0)

p_val = model.predict_proba(X_val_s)[:, 1]
p_test = model.predict_proba(X_test_s)[:, 1]

n_legit_val = int((y_val == 0).sum()); n_fraud_val = int(y_val.sum())
locked_thr, best_rec = 0.5, -1
for thr in np.arange(0.01, 1.0, 0.001):
    tn, fp, fn, tp = confusion_matrix(y_val, (p_val >= thr).astype(int), labels=[0, 1]).ravel()
    fpr = fp / max(n_legit_val, 1); rec = tp / max(n_fraud_val, 1)
    if fpr < 0.009 and rec > best_rec:
        best_rec = rec; locked_thr = thr
log(f"  Locked threshold: {locked_thr:.4f}")

dec_test = (p_test >= locked_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, dec_test, labels=[0, 1]).ravel()
log(f"  Overall test: AUC={roc_auc_score(y_test, p_test):.4f} "
    f"FPR={fp/max(tn+fp,1)*100:.3f}% Recall={tp/max(tp+fn,1)*100:.1f}% "
    f"Precision={tp/max(tp+fp,1)*100:.1f}% Alerts={tp+fp:,}")

# ============================================================
# SEGMENT DEFINITIONS (from raw test metadata)
# ============================================================
amt = pd.to_numeric(test_raw["Amount"].str.replace("$", "", regex=False), errors="coerce").fillna(0).values
use_chip = test_raw["Use Chip"].fillna("Online Transaction").values
user_str = test_raw["User"].astype(str).values
merch_str = test_raw["Merchant Name"].astype(str).values
city_str = test_raw["Merchant City"].astype(str).values
mcc_arr = test_raw["MCC"].fillna(0).astype(int).values

# feature-derived history depth (from X_test_s is scaled; use raw feature matrix before scaling!)
# Recompute lightweight history proxies from test_raw expansion: use user_tx_count from X_test (pre-scale, col 10)
# We discarded X_test; recompute history cols quickly is expensive. Instead derive from state counts is not
# possible per row now. Use raw proxies available in test_raw + feature vector cols we still have (X_test_s
# scaled). For history depth, rescale is monotone -> thresholds differ. Cleaner: re-derive counts cheaply.
# Simplest robust proxy: per-row ordinal index within (User), (Merchant Name) in the TEST slice only plus
# knowledge that all test entities were seen in train (users are unseen per earlier audit).
# We use test-slice-only cumcounts as "history within evaluation window" proxies and label them as such.
user_cum = pd.Series(user_str).groupby(user_str).cumcount().values
merch_cum = pd.Series(merch_str).groupby(merch_str).cumcount().values
city_cum = pd.Series(city_str).groupby(city_str).cumcount().values

# Channel segments
channel = np.where(np.array(use_chip) == "Online Transaction", "online",
           np.where(np.array(use_chip) == "Chip Transaction", "chip", "swipe"))

# Amount bands (fixed $ bands)
amt_band = np.where(amt < 20, "a_under20",
           np.where(amt < 50, "b_20_50",
           np.where(amt < 100, "c_50_100",
           np.where(amt < 250, "d_100_250",
           np.where(amt < 1000, "e_250_1000", "f_over1000")))))

# MCC top-category bands
mcc_band = np.where(mcc_arr == 0, "mcc_0",
            np.where(mcc_arr < 2000, "mcc_1_2k",
            np.where(mcc_arr < 4000, "mcc_2k_4k",
            np.where(mcc_arr < 6000, "mcc_4k_6k", "mcc_6k_plus"))))

# History depth within test window
user_hist = np.where(user_cum == 0, "u_first_in_test",
            np.where(user_cum < 10, "u_low_history", "u_high_history"))
merch_hist = np.where(merch_cum == 0, "m_first_in_test",
            np.where(merch_cum < 50, "m_low_history", "m_high_history"))

# Per-city and per-merchant: top-N by volume, others aggregated
city_counts = pd.Series(city_str).value_counts()
top_cities = set(city_counts.index[:8])
city_seg = np.where(np.isin(city_str, list(top_cities)), city_str, "other_city")

merch_counts = pd.Series(merch_str).value_counts()
top_merch = set(merch_counts.index[:8])
merch_seg = np.where(np.isin(merch_str, list(top_merch)), merch_str, "other_merchant")

segments = {
    "channel": channel,
    "amount_band": amt_band,
    "mcc_band": mcc_band,
    "user_history_test_window": user_hist,
    "merchant_history_test_window": merch_hist,
    "city_top8": city_seg,
    "merchant_top8": merch_seg,
}

def wilson_ci(k, n, z=1.96):
    if n == 0: return None
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (round(max(centre-half, 0.0), 6), round(min(centre+half, 1.0), 6))

def segment_metrics(seg):
    out = OrderedDict()
    for name in np.unique(seg):
        m = seg == name
        n = int(m.sum())
        if n < 200:  # sample-size floor for reporting (avoid tiny-group conclusions)
            out[name] = {"n": n, "skipped": "n<200"}
            continue
        y = y_test[m]
        p = p_test[m]
        n_fraud = int(y.sum())
        dec = dec_test[m]
        tn, fp, fn, tp = confusion_matrix(y, dec, labels=[0, 1]).ravel()
        fpr = fp / max(tn + fp, 1)
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1) if tp + fp else 0.0
        auc = float(roc_auc_score(y, p)) if n_fraud >= 20 and (n-n_fraud) >= 20 else None
        pr_auc = float(average_precision_score(y, p)) if n_fraud >= 20 else None
        alerts = tp + fp
        out[name] = {
            "n": n, "n_fraud": n_fraud,
            "fraud_rate": round(float(y.mean()), 6),
            "fraud_rate_ci": wilson_ci(n_fraud, n),
            "auc": round(auc, 4) if auc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "recall": round(float(rec), 4), "precision": round(float(prec), 4),
            "fpr": round(float(fpr), 6),
            "fpr_ci": wilson_ci(fp, tn + fp),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "alerts": int(alerts),
            "alerts_per_10k": round(alerts / n * 10000, 1),
        }
    return out

audit = OrderedDict()
audit["protocol"] = {
    "same_as_forensic": True,
    "threshold": round(float(locked_thr), 4),
    "threshold_source": "validation only (FPR<0.9%), locked",
    "test_rows": int(len(y_test)),
    "test_fraud": int(y_test.sum()),
    "segment_note": "history-depth segments use counts WITHIN the test window only; user-level prior "
                    "history is not available for test rows (users are new post-train) - see reports.",
    "metric_note": "AUC/PR-AUC reported only when >=20 fraud AND >=20 legit in segment; segments with "
                   "n<200 skipped to avoid tiny-group conclusions.",
}

for seg_name, seg in segments.items():
    log(f"  Segment: {seg_name}...")
    audit[seg_name] = segment_metrics(seg)

# ============================================================
# BEST / WORST / LARGEST-GAP ANALYSIS
# ============================================================
overall_fpr = fp / max(tn + fp, 1)
overall_rec = tp / max(tp + fn, 1)
overall_prec = tp / max(tp + fp, 1)
overall_auc = roc_auc_score(y_test, p_test)

gaps = OrderedDict()
for seg_name in audit:
    if seg_name in ("protocol", "comparisons"): continue
    for grp, m in audit[seg_name].items():
        if isinstance(m, dict) and "auc" in m and m.get("auc") is not None:
            gaps[f"{seg_name}::{grp}"] = {
                "auc_gap": round(float(m["auc"]) - overall_auc, 4),
                "fpr_gap": round(float(m["fpr"]) - overall_fpr, 6),
                "recall_gap": round(float(m["recall"]) - overall_rec, 4),
                "precision_gap": round(float(m["precision"]) - overall_prec, 4),
                "n": m["n"], "n_fraud": m["n_fraud"],
            }

def minmax(key):
    vals = [(gaps[k][key], k) for k in gaps]
    vals.sort()
    return vals[0], vals[-1]

audit["comparisons"] = {
    "overall": {"auc": round(float(overall_auc), 4), "fpr": round(float(overall_fpr), 6),
                "recall": round(float(overall_rec), 4), "precision": round(float(overall_prec), 4)},
    "best_worst_auc": [minmax("auc_gap")[1], minmax("auc_gap")[0][1]] if gaps else None,
    "best_worst_recall": [minmax("recall_gap")[1], minmax("recall_gap")[0][1]] if gaps else None,
    "best_worst_fpr": [minmax("fpr_gap")[0][1], minmax("fpr_gap")[1][1]] if gaps else None,
    "best_worst_precision": [minmax("precision_gap")[1], minmax("precision_gap")[0][1]] if gaps else None,
}

with open(REPORT, "w") as f:
    json.dump(_cvt(audit), f, indent=2)
log(f"\nReport saved: {REPORT}  (total {time.time()-T0:.0f}s)")
