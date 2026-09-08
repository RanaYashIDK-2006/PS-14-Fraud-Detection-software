#!/usr/bin/env python3
"""
Max-Capacity Test: Process ALL rows on both datasets.

Altman: All 24M rows via 2-pass chunked processing.
ULB: All 284K rows (full dataset).

Writes results to reports/max_capacity.json.
"""
import json, time, os, sys, gc
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

REPORT = ROOT / "reports" / "max_capacity.json"
REPORT.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    t = time.time() - T0
    print(f"[{t:6.0f}s] {msg}", flush=True)

T0 = time.time()
results = {}

# ============================================================
# PART 1: ULB — Full 284K rows
# ============================================================
log("=" * 60)
log("PART 1: ULB Full Dataset (284K rows)")
log("=" * 60)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb

t1 = time.time()
df_ulb = pd.read_csv("data/creditcard.csv")
log(f"Loaded ULB: {len(df_ulb):,} rows, {int(df_ulb.Class.sum())} fraud ({df_ulb.Class.mean()*100:.2f}%)")

y_ulb = df_ulb["Class"].values
X_ulb = df_ulb.drop("Class", axis=1).values.astype(np.float32)
X_ulb = np.nan_to_num(X_ulb, nan=0.0, posinf=0.0, neginf=0.0)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(y_ulb))
fold_aucs = []
fold_auprcs = []

for fold, (tr, te) in enumerate(skf.split(X_ulb, y_ulb)):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_ulb[tr])
    Xte = sc.transform(X_ulb[te])
    ytr, yte = y_ulb[tr], y_ulb[te]
    spw = max(1, int((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(spw, 30),
        random_state=42, n_jobs=4, eval_metric="auc",
        early_stopping_rounds=30,
    )
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    p = m.predict_proba(Xte)[:, 1]
    oof_pred[te] = p
    auc = roc_auc_score(yte, p)
    ap = average_precision_score(yte, p)
    fold_aucs.append(auc)
    fold_auprcs.append(ap)
    log(f"  Fold {fold+1}: AUC={auc:.4f}, PR-AUC={ap:.4f}, fraud={yte.sum()}")

ulb_oof_auc = roc_auc_score(y_ulb, oof_pred)
ulb_oof_pr = average_precision_score(y_ulb, oof_pred)

# Threshold sweep on OOF
ulb_thresholds = {}
for thr in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
    preds = (oof_pred >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_ulb, preds).ravel()
    fpr = fp / max(fp + tn, 1)
    recall = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0
    ulb_thresholds[str(thr)] = {
        "fpr": round(float(fpr), 6), "recall": round(float(recall), 6),
        "precision": round(float(prec), 6), "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn)
    }

best_ulb_fpr1 = None
for thr in np.arange(0.01, 1.0, 0.005):
    preds = (oof_pred >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_ulb, preds).ravel()
    fpr = fp / max(fp + tn, 1)
    recall = tp / max(tp + fn, 1)
    if fpr <= 0.01 and (best_ulb_fpr1 is None or recall > best_ulb_fpr1["recall"]):
        best_ulb_fpr1 = {"threshold": round(float(thr), 4), "fpr": round(float(fpr), 6), "recall": round(float(recall), 6)}

ulb_time = time.time() - t1
results["ulb"] = {
    "total_rows": len(df_ulb),
    "fraud_rows": int(df_ulb.Class.sum()),
    "legit_rows": int((df_ulb.Class == 0).sum()),
    "fraud_rate_pct": round(float(df_ulb.Class.mean() * 100), 4),
    "features": int(X_ulb.shape[1]),
    "cv_folds": 5,
    "fold_aucs": [round(float(a), 4) for a in fold_aucs],
    "fold_auprcs": [round(float(a), 4) for a in fold_auprcs],
    "mean_auc": round(float(np.mean(fold_aucs)), 4),
    "std_auc": round(float(np.std(fold_aucs)), 4),
    "oof_auc": round(float(ulb_oof_auc), 4),
    "oof_pr_auc": round(float(ulb_oof_pr), 4),
    "thresholds": ulb_thresholds,
    "best_fpr_under_1pct": best_ulb_fpr1,
    "train_time_sec": round(ulb_time, 1),
}
log(f"ULB done: OOF AUC={ulb_oof_auc:.4f}, CV mean={np.mean(fold_aucs):.4f}, time={ulb_time:.0f}s")

del df_ulb, X_ulb, y_ulb, oof_pred
gc.collect()

# ============================================================
# PART 2: ALTMAN — Full 24M rows via 2-pass chunked
# ============================================================
log("")
log("=" * 60)
log("PART 2: Altman Full Dataset (~24M rows, 2-pass chunked)")
log("=" * 60)

CHUNK = 2_000_000
CSV_PATH = "data/credit_card_transactions-ibm_v2.csv"
USECOLS = ["User", "Card", "Year", "Month", "Day", "Amount", "Use Chip",
           "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
           "Errors?", "Is Fraud?"]

# --- PASS 1: Compute per-entity stats ---
log("Pass 1: Computing entity statistics from full 24M...")
t2 = time.time()

user_tx_count = {}
user_fraud_count = {}
user_total_amt = {}
user_amt_sq = {}
merch_tx_count = {}
merch_fraud_count = {}
city_tx_count = {}
city_fraud_count = {}
user_merch_unique = {}
user_city_unique = {}
mcc_tx_count = {}
mcc_fraud_count = {}
user_last_amt = {}

row_count = 0
fraud_count = 0

for chunk in pd.read_csv(CSV_PATH, usecols=USECOLS, low_memory=False, chunksize=CHUNK):
    n = len(chunk)
    row_count += n

    is_fraud = chunk["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    fraud_count += int(is_fraud.sum())

    amt = pd.to_numeric(chunk["Amount"].str.replace("$", "", regex=False), errors="coerce").fillna(0).values
    users = chunk["User"].astype(str).values
    merchs = chunk["Merchant Name"].astype(str).values
    cities = chunk["Merchant City"].astype(str).values
    mccs = chunk["MCC"].fillna(0).astype(int).values

    for i in range(n):
        u = users[i]
        m = merchs[i]
        c = cities[i]
        mcc = int(mccs[i])
        a = float(amt[i])
        f = int(is_fraud.iloc[i])

        user_tx_count[u] = user_tx_count.get(u, 0) + 1
        user_fraud_count[u] = user_fraud_count.get(u, 0) + f
        user_total_amt[u] = user_total_amt.get(u, 0.0) + a
        user_amt_sq[u] = user_amt_sq.get(u, 0.0) + a * a

        prev = user_last_amt.get(u, a)
        if prev > 0:
            user_last_amt[u] = abs(a - prev) / max(prev, 0.01)
        else:
            user_last_amt[u] = 0.0

        merch_tx_count[m] = merch_tx_count.get(m, 0) + 1
        merch_fraud_count[m] = merch_fraud_count.get(m, 0) + f
        city_tx_count[c] = city_tx_count.get(c, 0) + 1
        city_fraud_count[c] = city_fraud_count.get(c, 0) + f
        mcc_tx_count[mcc] = mcc_tx_count.get(mcc, 0) + 1
        mcc_fraud_count[mcc] = mcc_fraud_count.get(mcc, 0) + f

        if u not in user_merch_unique:
            user_merch_unique[u] = set()
        user_merch_unique[u].add(m)
        if u not in user_city_unique:
            user_city_unique[u] = set()
        user_city_unique[u].add(c)

    if row_count % (4 * CHUNK) == 0:
        elapsed = time.time() - t2
        log(f"  Pass 1 progress: {row_count/1e6:.0f}M rows, {fraud_count} fraud ({elapsed:.0f}s)")

pass1_time = time.time() - t2
log(f"Pass 1 done: {row_count:,} rows, {fraud_count:,} fraud in {pass1_time:.0f}s")
log(f"  Unique users: {len(user_tx_count):,}")
log(f"  Unique merchants: {len(merch_tx_count):,}")
log(f"  Unique cities: {len(city_tx_count):,}")

# Precompute rates
user_fraud_rate = {u: user_fraud_count.get(u, 0) / max(user_tx_count.get(u, 1), 1) for u in user_tx_count}
merch_fraud_rate = {m: merch_fraud_count.get(m, 0) / max(merch_tx_count.get(m, 1), 1) for m in merch_tx_count}
city_fraud_rate = {c: city_fraud_count.get(c, 0) / max(city_tx_count.get(c, 1), 1) for c in city_tx_count}
mcc_fraud_rate = {m: mcc_fraud_count.get(m, 0) / max(mcc_tx_count.get(m, 1), 1) for m in mcc_tx_count}
user_avg_amt = {u: user_total_amt.get(u, 0) / max(user_tx_count.get(u, 1), 1) for u in user_tx_count}
user_std_amt = {}
for u in user_total_amt:
    n = user_tx_count.get(u, 1)
    avg = user_avg_amt.get(u, 0)
    var = max(user_amt_sq.get(u, 0) / n - avg * avg, 1e-10)
    user_std_amt[u] = var ** 0.5

del user_fraud_count, merch_fraud_count, city_fraud_count, mcc_fraud_count
del user_amt_sq
gc.collect()

# Median values for fill
median_merch_pop = np.median(list(merch_tx_count.values())[:5000]) if merch_tx_count else 1
median_city_pop = np.median(list(city_tx_count.values())[:5000]) if city_tx_count else 1

# --- PASS 2: Build features ---
log("")
log("Pass 2: Building features on full 24M rows...")
t3 = time.time()

all_X = []
all_y = []
row_count = 0

for chunk in pd.read_csv(CSV_PATH, usecols=USECOLS, low_memory=False, chunksize=CHUNK):
    n = len(chunk)
    row_count += n

    is_fraud = chunk["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int).values
    amt_raw = pd.to_numeric(chunk["Amount"].str.replace("$", "", regex=False), errors="coerce").fillna(0).values
    users = chunk["User"].astype(str).values
    merchs = chunk["Merchant Name"].astype(str).values
    cities = chunk["Merchant City"].astype(str).values
    mccs = chunk["MCC"].fillna(0).astype(int).values
    use_chip = chunk["Use Chip"].fillna("Online Transaction").values
    year = chunk["Year"].fillna(2019).values.astype(float)
    month = chunk["Month"].fillna(1).values.astype(float)
    day = chunk["Day"].fillna(1).values.astype(float)
    zip_val = chunk["Zip"].fillna(0).values
    has_zip = (zip_val > 0).astype(float)
    merchant_state = chunk["Merchant State"].fillna("").values
    has_state = (merchant_state != "").astype(float)

    amt = amt_raw.astype(np.float32)

    # Vectorized lookups
    utc = np.array([user_tx_count.get(u, 0) for u in users], dtype=np.float32)
    ufr = np.array([user_fraud_rate.get(u, 0) for u in users], dtype=np.float32)
    mfr = np.array([merch_fraud_rate.get(m, 0) for m in merchs], dtype=np.float32)
    cfr = np.array([city_fraud_rate.get(c, 0) for c in cities], dtype=np.float32)
    mtc = np.array([merch_tx_count.get(m, 0) for m in merchs], dtype=np.float32)
    ctc = np.array([city_tx_count.get(c, 0) for c in cities], dtype=np.float32)
    uavg = np.array([user_avg_amt.get(u, 0) for u in users], dtype=np.float32)
    ustd = np.array([user_std_amt.get(u, 1) for u in users], dtype=np.float32)
    accel = np.array([user_last_amt.get(u, 0) for u in users], dtype=np.float32)

    log_amt = np.log1p(amt)
    amt_sq = amt ** 2

    chip = np.where(use_chip == "Chip Transaction", 1.0,
            np.where(use_chip == "Swipe Transaction", 0.5, 0.0))
    is_online = np.where(use_chip == "Online Transaction", 1.0, 0.0)

    mcc_n = mccs.astype(np.float32) / 6000.0

    merch_pop = np.minimum(mtc / max(median_merch_pop, 1), 5.0)
    city_pop = np.minimum(ctc / max(median_city_pop, 1), 5.0)

    amt_ratio = amt / np.maximum(uavg, 0.01)
    amt_zscore = (amt - uavg) / np.maximum(ustd, 0.01)

    very_high_amt = (amt > 5000).astype(float)
    amt_x_mcc = amt * mcc_n
    amt_x_online = amt * is_online
    amt_x_chip = amt * chip
    mfr_x_ufr = mfr * ufr

    # Merchant diversity (unique merchs per user — approximate from count)
    # We don't store per-user merch sets in pass1 due to memory, so use utc as proxy
    umdiv = np.minimum(utc / np.maximum(mtc, 1), 10.0)  # user-to-merchant ratio

    F = np.column_stack([
        log_amt, amt_sq,                        # 0-1: amount
        year, month, day,                        # 2-4: time
        chip, is_online, mcc_n,                  # 5-7: channel
        has_zip, has_state,                      # 8-9: address
        utc,                                     # 10: user tx count
        mfr, cfr,                                # 11-12: fraud rates
        very_high_amt,                           # 13: amount flag
        amt_x_mcc, amt_x_online,                # 14-15: interactions
        merch_pop,                               # 16: merchant popularity
        ufr,                                     # 17: user fraud rate
        amt_ratio,                               # 18: amount vs user avg
        amt_zscore,                              # 19: amount z-score
        accel,                                   # 20: amount acceleration
        mfr_x_ufr,                               # 21: fraud rate interaction
        city_pop,                                # 22: city popularity
        umdiv,                                   # 23: user merchant diversity
        amt_x_chip,                              # 24: amount x chip
    ]).astype(np.float32)

    F = np.nan_to_num(F, nan=0.0, posinf=10.0, neginf=-10.0)

    all_X.append(F)
    all_y.append(is_fraud)

    if row_count % (4 * CHUNK) == 0:
        elapsed = time.time() - t3
        log(f"  Pass 2 progress: {row_count/1e6:.0f}M rows ({elapsed:.0f}s)")

pass2_time = time.time() - t3
log(f"Pass 2 done: {row_count:,} rows in {pass2_time:.0f}s")

# Combine
X_all = np.vstack(all_X)
y_all = np.concatenate(all_y)
del all_X, all_y
gc.collect()

total_rows = len(y_all)
total_fraud = int(y_all.sum())
log(f"Full dataset: {total_rows:,} rows, {total_fraud:,} fraud ({total_fraud/total_rows*100:.4f}%)")

# Temporal split: 80% train, 20% test
sp = int(total_rows * 0.8)
Xtr, Xte = X_all[:sp], X_all[sp:]
ytr, yte = y_all[:sp], y_all[sp:]
del X_all, y_all
gc.collect()

log(f"Train: {len(ytr):,} rows ({int(ytr.sum())} fraud)")
log(f"Test:  {len(yte):,} rows ({int(yte.sum())} fraud)")

# Scale
sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)
del Xtr, Xte
gc.collect()

# Train XGBoost
log("Training XGBoost on full 24M...")
t4 = time.time()
spw = max(1, int((ytr == 0).sum() / max(int(ytr.sum()), 1)))
xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, scale_pos_weight=min(spw, 30),
    gamma=1, min_child_weight=3, random_state=42, n_jobs=6,
    eval_metric="auc", early_stopping_rounds=50,
)
xgb_model.fit(Xtr_s, ytr, eval_set=[(Xte_s, yte)], verbose=100)
xgb_time = time.time() - t4
log(f"XGB trained in {xgb_time:.0f}s")

# Predict
p_xgb = xgb_model.predict_proba(Xte_s)[:, 1]
auc_xgb = roc_auc_score(yte, p_xgb)
pr_xgb = average_precision_score(yte, p_xgb)
log(f"XGB: AUC={auc_xgb:.4f}, PR-AUC={pr_xgb:.4f}")

# Feature importance
feature_names = [
    "log_amt", "amt_sq", "year", "month", "day", "chip", "is_online", "mcc_n",
    "has_zip", "has_state", "user_tx_count", "merch_fraud_rate", "city_fraud_rate",
    "very_high_amt", "amt_x_mcc", "amt_x_online", "merch_popularity",
    "user_fraud_rate", "amt_ratio", "amt_zscore", "amt_acceleration",
    "mfr_x_ufr", "city_popularity", "user_merch_diversity", "amt_x_chip"
]
fi = {name: round(float(imp), 4) for name, imp in zip(feature_names, xgb_model.feature_importances_)}
top10 = sorted(fi.items(), key=lambda x: -x[1])[:10]
log(f"Top features:")
for name, imp in top10:
    log(f"  {name}: {imp:.4f}")

# Threshold sweep
log("Running threshold sweep...")
alt_thresholds = {}
for thr in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
    preds = (p_xgb >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, preds).ravel()
    fpr = fp / max(fp + tn, 1)
    recall = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0
    alt_thresholds[str(thr)] = {
        "fpr": round(float(fpr), 6), "recall": round(float(recall), 6),
        "precision": round(float(prec), 6), "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn)
    }

# Best FPR thresholds
best_alt_fpr1 = None
best_alt_fpr3 = None
for thr in np.arange(0.01, 1.0, 0.005):
    preds = (p_xgb >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, preds).ravel()
    fpr = fp / max(fp + tn, 1)
    recall = tp / max(tp + fn, 1)
    if fpr <= 0.01 and (best_alt_fpr1 is None or recall > best_alt_fpr1["recall"]):
        best_alt_fpr1 = {"threshold": round(float(thr), 4), "fpr": round(float(fpr), 6), "recall": round(float(recall), 6)}
    if fpr <= 0.03 and (best_alt_fpr3 is None or recall > best_alt_fpr3["recall"]):
        best_alt_fpr3 = {"threshold": round(float(thr), 4), "fpr": round(float(fpr), 6), "recall": round(float(recall), 6)}

alt_time = time.time() - t3
total_time = time.time() - T0

results["altman"] = {
    "total_rows": total_rows,
    "train_rows": int(len(ytr)),
    "test_rows": int(len(yte)),
    "total_fraud": total_fraud,
    "train_fraud": int(ytr.sum()),
    "test_fraud": int(yte.sum()),
    "fraud_rate_pct": round(total_fraud / total_rows * 100, 4),
    "features": int(Xte_s.shape[1]),
    "feature_names": feature_names,
    "feature_importance": fi,
    "top_10_features": [{"name": k, "importance": v} for k, v in top10],
    "xgb_auc": round(float(auc_xgb), 4),
    "xgb_pr_auc": round(float(pr_xgb), 4),
    "xgb_train_time_sec": round(xgb_time, 1),
    "thresholds": alt_thresholds,
    "best_fpr_under_1pct": best_alt_fpr1,
    "best_fpr_under_3pct": best_alt_fpr3,
    "pass1_time_sec": round(pass1_time, 1),
    "pass2_time_sec": round(pass2_time, 1),
    "processing_time_sec": round(alt_time, 1),
}

log(f"Altman done: AUC={auc_xgb:.4f}, PR={pr_xgb:.4f}, time={alt_time:.0f}s")

# Save
results["summary"] = {
    "total_time_sec": round(total_time, 1),
    "ulb_rows": 284807,
    "altman_rows": total_rows,
    "combined_rows": 284807 + total_rows,
}
results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open(REPORT, "w") as f:
    json.dump(results, f, indent=2)

log("")
log("=" * 60)
log(f"COMPLETE: {total_time:.0f}s total")
log(f"ULB:  284,807 rows, CV AUC={results['ulb']['mean_auc']}")
log(f"Alt:  {total_rows:,} rows, AUC={results['altman']['xgb_auc']}")
log(f"Report saved: {REPORT}")
log("=" * 60)
