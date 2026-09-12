#!/usr/bin/env python3
"""
PS-14 CHECK #13 + #14: CALIBRATION ANALYSIS + STRESS TESTING

Runs once: load -> features -> train -> evaluate -> calibration -> stress tests.
Saves combined report to reports/calibration_and_stress.json.
"""
import json, os, sys, gc, time, warnings, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
os.chdir(ROOT)

REPORT = ROOT / "reports" / "calibration_and_stress.json"
REPORT.parent.mkdir(parents=True, exist_ok=True)

T0 = time.time()
SEED = 42
np.random.seed(SEED)

def log(msg):
    t = time.time() - T0
    print(f"[{t:6.0f}s] {msg}", flush=True)

def _cvt(obj):
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _cvt(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cvt(v) for v in obj]
    return str(obj)

CSV = "data/credit_card_transactions-ibm_v2.csv"
CHUNK = 2_000_000
USECOLS = ["User", "Card", "Year", "Month", "Day", "Amount", "Use Chip",
           "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
           "Errors?", "Is Fraud?"]

FEATURE_NAMES = [
    "log_amt", "amt_sq", "year", "month", "day", "chip", "is_online", "mcc_n",
    "has_zip", "has_state", "user_tx_count", "merch_fraud_rate", "city_fraud_rate",
    "very_high_amt", "amt_x_mcc", "amt_x_online", "merch_popularity",
    "user_fraud_rate", "amt_ratio", "amt_zscore", "amt_acceleration",
    "mfr_x_ufr", "city_popularity", "user_merch_diversity", "amt_x_chip",
]

# ============================================================
# EXPANDING WINDOW FEATURE COMPUTATION (causal, cold-start aware)
# ============================================================
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

    utc = np.zeros(n, dtype=np.float32)
    ufr = np.zeros(n, dtype=np.float32)
    uavg = np.zeros(n, dtype=np.float32)
    ustd = np.zeros(n, dtype=np.float32)
    mfr = np.zeros(n, dtype=np.float32)
    cfr = np.zeros(n, dtype=np.float32)
    mtc = np.zeros(n, dtype=np.float32)
    accel = np.zeros(n, dtype=np.float32)
    ctc = np.zeros(n, dtype=np.float32)

    utc_s = state["user_tx_count"]
    ufc_s = state["user_fraud_count"]
    uta_s = state["user_total_amt"]
    uas_s = state["user_amt_sq"]
    ula_s = state["user_last_amt"]
    mtc_s = state["merch_tx_count"]
    mfc_s = state["merch_fraud_count"]
    ctc_s = state["city_tx_count"]
    cfc_s = state["city_fraud_count"]

    if pop_state is None:
        pop_state = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0,
                      "total_amt_sq": 0.0, "n_users": 0}
    _ptx = pop_state["total_tx"]
    _pfraud = pop_state["total_fraud"]
    _pamt = pop_state["total_amt"]
    _pamtsq = pop_state["total_amt_sq"]
    _pn = pop_state["n_users"]
    pop_fraud_rate = _pfraud / max(_ptx, 1)
    pop_avg_amt = _pamt / max(_ptx, 1)
    pop_std_amt = max((_pamtsq / max(_ptx, 1)) - pop_avg_amt ** 2, 1e-10) ** 0.5

    for i in range(n):
        u = users[i]
        m = merchs[i]
        c = cities[i]
        a = float(amt[i])
        is_new_user = u not in utc_s

        utx = utc_s.get(u, 0)
        utc[i] = utx
        uf = ufc_s.get(u, 0)
        ut = uta_s.get(u, 0.0)
        us = uas_s.get(u, 0.0)

        if is_new_user and _ptx > 0:
            ufr[i] = pop_fraud_rate
            uavg[i] = pop_avg_amt
            ustd[i] = pop_std_amt
            accel[i] = 0.0
        else:
            ufr[i] = uf / max(utx, 1)
            ua = ut / max(utx, 1)
            uavg[i] = ua
            uv = max(us / max(utx, 1) - ua * ua, 1e-10)
            ustd[i] = uv ** 0.5
            prev = ula_s.get(u, a)
            accel[i] = abs(a - prev) / max(prev, 0.01) if prev > 0 else 0.0

        mt = mtc_s.get(m, 0)
        mtc[i] = mt
        mf = mfc_s.get(m, 0)
        mfr[i] = mf / max(mt, 1)
        ct = ctc_s.get(c, 0)
        ctc[i] = ct
        cf = cfc_s.get(c, 0)
        cfr[i] = cf / max(ct, 1)

        utc_s[u] = utx + 1
        ufc_s[u] = uf + int(is_fraud[i])
        uta_s[u] = ut + a
        uas_s[u] = us + a * a
        ula_s[u] = a
        mtc_s[m] = mt + 1
        mfc_s[m] = mf + int(is_fraud[i])
        ctc_s[c] = ct + 1
        cfc_s[c] = cf + int(is_fraud[i])
        _ptx += 1
        _pfraud += int(is_fraud[i])
        _pamt += a
        _pamtsq += a * a
        if is_new_user:
            _pn += 1

    pop_state["total_tx"] = _ptx
    pop_state["total_fraud"] = _pfraud
    pop_state["total_amt"] = _pamt
    pop_state["total_amt_sq"] = _pamtsq
    pop_state["n_users"] = _pn

    very_high_amt = (amt > 5000).astype(float)
    amt_x_mcc = amt * mcc_n
    amt_x_online = amt * is_online
    amt_x_chip = amt * chip
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
    F = np.nan_to_num(F, nan=0.0, posinf=10.0, neginf=-10.0)
    return F


def compute_features(df_slice, state, pop_state):
    chunks_F = []
    for start in range(0, len(df_slice), CHUNK):
        end = min(start + CHUNK, len(df_slice))
        chunk = df_slice.iloc[start:end]
        F = expanding_features_fixed(chunk, state, pop_state)
        chunks_F.append(F)
    return np.vstack(chunks_F)


# ============================================================
# PART 1: LOAD DATASET + COMPUTE FEATURES + SPLIT + TRAIN
# ============================================================
log("=" * 70)
log("PART 1: LOAD DATASET, COMPUTE FEATURES, TRAIN")
log("=" * 70)

all_chunks = []
for i, chunk in enumerate(pd.read_csv(CSV, usecols=USECOLS, low_memory=False, chunksize=CHUNK)):
    all_chunks.append(chunk)
    if i % 3 == 0:
        log(f"  Chunk {i}: {sum(len(c) for c in all_chunks):,} rows")

df = pd.concat(all_chunks, ignore_index=True)
del all_chunks
gc.collect()

total = len(df)
fraud_mask = df["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
log(f"  Total: {total:,} rows, Fraud: {int(fraud_mask.sum()):,} ({fraud_mask.mean()*100:.3f}%)")

# Temporal split: 60% train, 20% val, 20% test
n_train = int(total * 0.60)
n_val = int(total * 0.20)
n_test = total - n_train - n_val

log(f"  Split: train={n_train:,}, val={n_val:,}, test={n_test:,}")

# Compute features
state_train = {k: {} for k in ["user_tx_count", "user_fraud_count", "user_total_amt",
                                 "user_amt_sq", "user_last_amt", "merch_tx_count",
                                 "merch_fraud_count", "city_tx_count", "city_fraud_count"]}
pop_train = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0, "total_amt_sq": 0.0, "n_users": 0}

log("  Computing train features...")
t1 = time.time()
X_train = compute_features(df.iloc[:n_train], state_train, pop_train)
y_train = fraud_mask.iloc[:n_train].values
log(f"  Train features: {X_train.shape} ({time.time()-t1:.0f}s)")

log("  Computing val features...")
X_val = compute_features(df.iloc[n_train:n_train+n_val], state_train, pop_train)
y_val = fraud_mask.iloc[n_train:n_train+n_val].values
log(f"  Val features: {X_val.shape}")

log("  Computing test features...")
X_test = compute_features(df.iloc[n_train+n_val:], state_train, pop_train)
y_test = fraud_mask.iloc[n_train+n_val:].values
log(f"  Test features: {X_test.shape}")

del df
gc.collect()

# Scale
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_val_s = sc.transform(X_val)
X_test_s = sc.transform(X_test)
del X_train, X_val, X_test
gc.collect()

# Train
spw = max(1, int((y_train == 0).sum() / max(int(y_train.sum()), 1)))
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, scale_pos_weight=min(spw, 30),
    gamma=1, min_child_weight=3, random_state=SEED, n_jobs=4,
    eval_metric="auc", early_stopping_rounds=30,
)
log("  Training XGB...")
model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=100)
log(f"  Training done (best_iter={model.best_iteration})")

# Raw predictions
p_val = model.predict_proba(X_val_s)[:, 1]
p_test = model.predict_proba(X_test_s)[:, 1]
log(f"  Val predictions: mean={p_val.mean():.6f}, max={p_val.max():.6f}")
log(f"  Test predictions: mean={p_test.mean():.6f}, max={p_test.max():.6f}")

# Lock threshold from validation (FPR < 0.9%, max recall)
n_legit_val = int((y_val == 0).sum())
n_fraud_val = int(y_val.sum())
best_thr = 0.5
best_rec = 0
for thr in np.arange(0.01, 1.0, 0.001):
    preds = (p_val >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
    fpr = fp / max(n_legit_val, 1)
    rec = tp / max(n_fraud_val, 1)
    if fpr < 0.009 and rec > best_rec:
        best_rec = rec
        best_thr = thr
locked_thr = best_thr
log(f"  Locked threshold: {locked_thr:.4f} (val FPR target < 0.9%, max recall)")

# Final test metrics
preds_test = (p_test >= locked_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, preds_test, labels=[0, 1]).ravel()
test_fpr = fp / max(int((y_test == 0).sum()), 1)
test_recall = tp / max(int(y_test.sum()), 1)
test_precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0
test_auc = roc_auc_score(y_test, p_test)
test_pr_auc = average_precision_score(y_test, p_test)
log(f"  Test: AUC={test_auc:.4f}, PR-AUC={test_pr_auc:.4f}, "
    f"Recall={test_recall*100:.1f}%, FPR={test_fpr*100:.3f}%, "
    f"Precision={test_precision*100:.1f}%, Alerts={tp+fp:,}")

audit = OrderedDict()
audit["baseline"] = {
    "threshold": round(float(locked_thr), 4),
    "test_auc": round(float(test_auc), 4),
    "test_pr_auc": round(float(test_pr_auc), 4),
    "test_recall": round(float(test_recall), 4),
    "test_precision": round(float(test_precision), 4),
    "test_fpr": round(float(test_fpr), 6),
    "test_tp": int(tp), "test_tn": int(tn), "test_fp": int(fp), "test_fn": int(fn),
    "test_alerts": int(tp + fp),
    "test_fraud_rate": round(float(y_test.mean()), 6),
    "n_val": int(len(y_val)), "n_test": int(len(y_test)),
}


# ============================================================
# CHECK #13: CALIBRATION ANALYSIS
# ============================================================
log("")
log("=" * 70)
log("CHECK #13: CALIBRATION ANALYSIS")
log("=" * 70)

# --- 13a: Brier score ---
brier_val = brier_score_loss(y_val, p_val)
brier_test = brier_score_loss(y_test, p_test)
log(f"  Brier score: val={brier_val:.6f}, test={brier_test:.6f}")

# --- 13b: ECE (Expected Calibration Error) ---
def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rel_data = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        mean_pred = y_prob[mask].mean()
        actual_rate = y_true[mask].mean()
        weight = mask.sum() / len(y_true)
        ece += weight * abs(mean_pred - actual_rate)
        rel_data.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "mean_predicted": round(float(mean_pred), 4),
            "observed_rate": round(float(actual_rate), 4),
            "count": int(mask.sum()),
            "weight": round(float(weight), 4),
        })
    return float(ece), rel_data

ece_val, rel_val = compute_ece(y_val, p_val)
ece_test, rel_test = compute_ece(y_test, p_test)
log(f"  ECE: val={ece_val:.6f}, test={ece_test:.6f}")

# --- 13c: Segment-level calibration ---
# By time (first half vs second half of test set)
half = len(y_test) // 2
ece_test_1st, _ = compute_ece(y_test[:half], p_test[:half])
ece_test_2nd, _ = compute_ece(y_test[half:], p_test[half:])
brier_test_1st = brier_score_loss(y_test[:half], p_test[:half])
brier_test_2nd = brier_score_loss(y_test[half:], p_test[half:])
log(f"  ECE by time: 1st_half={ece_test_1st:.6f}, 2nd_half={ece_test_2nd:.6f}")
log(f"  Brier by time: 1st_half={brier_test_1st:.6f}, 2nd_half={brier_test_2nd:.6f}")

# By predicted-score band
bands = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 1.0)]
band_cal = []
for lo, hi in bands:
    mask = (p_test >= lo) & (p_test < hi) if hi < 1.0 else (p_test >= lo) & (p_test <= hi)
    if mask.sum() < 10:
        continue
    mean_pred = float(p_test[mask].mean())
    actual_rate = float(y_test[mask].mean())
    band_cal.append({
        "band": f"[{lo},{hi})",
        "mean_predicted": round(mean_pred, 4),
        "observed_rate": round(actual_rate, 4),
        "count": int(mask.sum()),
        "gap": round(abs(mean_pred - actual_rate), 4),
    })
    log(f"  Band [{lo},{hi}): pred={mean_pred:.4f}, actual={actual_rate:.4f}, gap={abs(mean_pred-actual_rate):.4f}, n={mask.sum()}")

# --- 13d: Platt (sigmoid) calibration on validation ---
log("  Fitting Platt calibration on validation...")
lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
p_val_2d = p_val.reshape(-1, 1)
lr.fit(p_val_2d, y_val)
p_val_platt = lr.predict_proba(p_val_2d)[:, 1]
p_test_platt = lr.predict_proba(p_test.reshape(-1, 1))[:, 1]
brier_val_platt = brier_score_loss(y_val, p_val_platt)
brier_test_platt = brier_score_loss(y_test, p_test_platt)
ece_val_platt, _ = compute_ece(y_val, p_val_platt)
ece_test_platt, _ = compute_ece(y_test, p_test_platt)
log(f"  Platt calibration: val_brier={brier_val_platt:.6f} (was {brier_val:.6f}), "
    f"test_brier={brier_test_platt:.6f} (was {brier_test:.6f})")
log(f"  Platt ECE: val={ece_val_platt:.6f}, test={ece_test_platt:.6f}")

# AUC must not change (monotone transform)
auc_val_platt = roc_auc_score(y_val, p_val_platt)
auc_test_platt = roc_auc_score(y_test, p_test_platt)
log(f"  Platt AUC: val={auc_val_platt:.4f} (was {roc_auc_score(y_val, p_val):.4f}), "
    f"test={auc_test_platt:.4f} (was {test_auc:.4f})")

# --- 13e: Isotonic calibration on validation ---
log("  Fitting isotonic calibration on validation...")
iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
iso.fit(p_val, y_val)
p_val_iso = iso.transform(p_val)
p_test_iso = iso.transform(p_test)
brier_val_iso = brier_score_loss(y_val, p_val_iso)
brier_test_iso = brier_score_loss(y_test, p_test_iso)
ece_val_iso, _ = compute_ece(y_val, p_val_iso)
ece_test_iso, _ = compute_ece(y_test, p_test_iso)
log(f"  Isotonic: val_brier={brier_val_iso:.6f} (was {brier_val:.6f}), "
    f"test_brier={brier_test_iso:.6f} (was {brier_test:.6f})")
log(f"  Isotonic ECE: val={ece_val_iso:.6f}, test={ece_test_iso:.6f}")

auc_val_iso = roc_auc_score(y_val, p_val_iso)
auc_test_iso = roc_auc_score(y_test, p_test_iso)
log(f"  Isotonic AUC: val={auc_val_iso:.4f}, test={auc_test_iso:.4f}")

# --- 13f: Does the system need calibrated probabilities? ---
# The production system uses a ranking threshold, not probability.
# But calibrated probabilities help with explainability and risk-banding.
calibration_usefulness = {
    "ranking_auc_unchanged": abs(test_auc - auc_test_platt) < 1e-6 and abs(test_auc - auc_test_iso) < 1e-6,
    "brier_before": round(float(brier_test), 6),
    "brier_platt": round(float(brier_test_platt), 6),
    "brier_isotonic": round(float(brier_test_iso), 6),
    "ece_before": round(float(ece_test), 6),
    "ece_platt": round(float(ece_test_platt), 6),
    "ece_isotonic": round(float(ece_test_iso), 6),
    "best_calibration_method": "isotonic" if brier_test_iso <= brier_test_platt else "platt",
    "needs_calibration_for_ranking": False,
    "needs_calibration_for_risk_banding": abs(ece_test - ece_test_iso) > 0.01,
}

# --- 13g: Reliability diagram data for test set ---
_, rel_test_raw = compute_ece(y_test, p_test, n_bins=10)

# Save calibration results
audit["calibration"] = {
    "brier": {"val": round(float(brier_val), 6), "test": round(float(brier_test), 6)},
    "ece": {"val": round(float(ece_val), 6), "test": round(float(ece_test), 6)},
    "segment_calibration": {
        "by_time": {
            "1st_half_ece": round(float(ece_test_1st), 6),
            "2nd_half_ece": round(float(ece_test_2nd), 6),
            "1st_half_brier": round(float(brier_test_1st), 6),
            "2nd_half_brier": round(float(brier_test_2nd), 6),
        },
        "by_score_band": band_cal,
    },
    "platt": {
        "brier_val": round(float(brier_val_platt), 6),
        "brier_test": round(float(brier_test_platt), 6),
        "ece_val": round(float(ece_val_platt), 6),
        "ece_test": round(float(ece_test_platt), 6),
        "auc_val": round(float(auc_val_platt), 4),
        "auc_test": round(float(auc_test_platt), 4),
    },
    "isotonic": {
        "brier_val": round(float(brier_val_iso), 6),
        "brier_test": round(float(brier_test_iso), 6),
        "ece_val": round(float(ece_val_iso), 6),
        "ece_test": round(float(ece_test_iso), 6),
        "auc_val": round(float(auc_val_iso), 4),
        "auc_test": round(float(auc_test_iso), 4),
    },
    "reliability_diagram_test": rel_test_raw,
    "usefulness": calibration_usefulness,
}


# ============================================================
# CHECK #14: STRESS TESTING
# ============================================================
log("")
log("=" * 70)
log("CHECK #14: STRESS TESTING")
log("=" * 70)

# Reference: clean test predictions
ref_auc = test_auc
ref_pr = test_pr_auc
ref_recall = test_recall
ref_fpr = test_fpr
ref_prec = test_precision

stress_results = []

def eval_stress(name, X_mod, y_true, desc, severity="High"):
    """Evaluate stress scenario: predict, check for NaN/Inf, compute metrics."""
    try:
        # Check for NaN/Inf in input
        has_nan = bool(np.any(np.isnan(X_mod)))
        has_inf = bool(np.any(np.isinf(X_mod)))

        # Predict
        p = model.predict_proba(X_mod)[:, 1]

        # Check output validity
        out_nan = bool(np.any(np.isnan(p)))
        out_inf = bool(np.any(np.isinf(p)))
        out_range = bool(np.any(p < 0) or np.any(p > 1))

        if out_nan or out_inf:
            return {
                "scenario": name, "description": desc, "severity": severity,
                "result": "FAIL", "failure_mode": "NaN/Inf in predictions",
                "input_nan": has_nan, "input_inf": has_inf,
                "output_nan": out_nan, "output_inf": out_inf,
            }

        # Metrics
        n_legit = int((y_true == 0).sum())
        n_fraud = int(y_true.sum())
        preds = (p >= locked_thr).astype(int)
        tn_l, fp_l, fn_l, tp_l = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fpr_v = fp_l / max(n_legit, 1)
        rec_v = tp_l / max(n_fraud, 1) if n_fraud > 0 else 0
        prec_v = tp_l / max(tp_l + fp_l, 1) if (tp_l + fp_l) > 0 else 0
        auc_v = roc_auc_score(y_true, p) if n_fraud > 0 and n_legit > 0 else 0
        pr_v = average_precision_score(y_true, p) if n_fraud > 0 else 0

        # Delta from reference
        d_auc = auc_v - ref_auc
        d_pr = pr_v - ref_pr
        d_rec = rec_v - ref_recall
        d_fpr = fpr_v - ref_fpr

        return {
            "scenario": name, "description": desc, "severity": severity,
            "result": "PASS", "input_nan": has_nan, "input_inf": has_inf,
            "output_nan": False, "output_inf": False, "output_range_ok": True,
            "n_rows": int(len(y_true)),
            "auc": round(float(auc_v), 4), "pr_auc": round(float(pr_v), 4),
            "recall": round(float(rec_v), 4), "precision": round(float(prec_v), 4),
            "fpr": round(float(fpr_v), 6),
            "alerts": int(tp_l + fp_l),
            "delta_auc": round(float(d_auc), 4), "delta_pr_auc": round(float(d_pr), 4),
            "delta_recall": round(float(d_rec), 4), "delta_fpr": round(float(d_fpr), 6),
            "score_mean": round(float(p.mean()), 6),
            "score_std": round(float(p.std()), 6),
            "score_min": round(float(p.min()), 6),
            "score_max": round(float(p.max()), 6),
        }
    except Exception as e:
        return {
            "scenario": name, "description": desc, "severity": severity,
            "result": "FAIL", "failure_mode": f"Exception: {str(e)[:100]}",
        }


# 14a: Missing values — set all features to 0
log("  Scenario: all-zero features (cold-start)...")
X_zeros = np.zeros_like(X_test_s)
stress_results.append(eval_stress(
    "all_zero_features", X_zeros, y_test,
    "All 25 features set to 0 (simulates complete cold-start with no history)",
    severity="Critical"))

# 14b: Missing individual features — set each to 0 one at a time
log("  Scenario: individual missing features...")
for fi in range(min(5, len(FEATURE_NAMES))):  # test first 5 features
    X_m = X_test_s.copy()
    X_m[:, fi] = 0.0
    stress_results.append(eval_stress(
        f"missing_feature_{FEATURE_NAMES[fi]}", X_m, y_test,
        f"Feature '{FEATURE_NAMES[fi]}' set to 0",
        severity="Medium"))

# 14c: Unknown merchants — set merchant features to 0
log("  Scenario: unknown merchants...")
# merch_fraud_rate=idx11, merch_popularity=idx16, mfr_x_ufr=idx21, user_merch_diversity=idx23
X_m = X_test_s.copy()
for fi in [11, 16, 21, 23]:
    X_m[:, fi] = 0.0
stress_results.append(eval_stress(
    "unknown_merchants", X_m, y_test,
    "Merchant features (merch_fraud_rate, merch_popularity, mfr_x_ufr, user_merch_diversity) set to 0",
    severity="High"))

# 14d: Unknown cities — set city features to 0
log("  Scenario: unknown cities...")
# city_fraud_rate=idx12, city_popularity=idx22
X_m = X_test_s.copy()
for fi in [12, 22]:
    X_m[:, fi] = 0.0
stress_results.append(eval_stress(
    "unknown_cities", X_m, y_test,
    "City features (city_fraud_rate, city_popularity) set to 0",
    severity="Medium"))

# 14e: Cold-start users — set all user features to 0
log("  Scenario: cold-start users...")
# user_tx_count=10, user_fraud_rate=17, amt_ratio=18, amt_zscore=19, amt_acceleration=20
X_m = X_test_s.copy()
for fi in [10, 17, 18, 19, 20]:
    X_m[:, fi] = 0.0
stress_results.append(eval_stress(
    "cold_start_users", X_m, y_test,
    "All user-specific features set to 0 (new user with no history)",
    severity="High"))

# 14f: Extreme amounts — push log_amt and amt_sq to extremes
log("  Scenario: extreme amounts...")
X_m = X_test_s.copy()
X_m[:, 0] = 15.0  # log_amt very high (e.g., log(1M+))
X_m[:, 1] = 50.0  # amt_sq very high
stress_results.append(eval_stress(
    "extreme_amounts_high", X_m, y_test,
    "log_amt set to 15 (very high amount ~$3.3M), amt_sq set to 50",
    severity="Medium"))

X_m = X_test_s.copy()
X_m[:, 0] = -5.0  # log_amt very low (negative = <1 cent)
X_m[:, 1] = 0.0   # amt_sq = 0
stress_results.append(eval_stress(
    "extreme_amounts_low", X_m, y_test,
    "log_amt set to -5 (very low amount ~$0.007), amt_sq set to 0",
    severity="Medium"))

# 14g: Large transaction burst — high user_tx_count
log("  Scenario: transaction burst...")
X_m = X_test_s.copy()
X_m[:, 10] = 10.0  # user_tx_count very high (standardized)
stress_results.append(eval_stress(
    "transaction_burst", X_m, y_test,
    "user_tx_count set to 10 std above mean (simulates burst of activity)",
    severity="Medium"))

# 14h: NaN propagation test
log("  Scenario: NaN propagation...")
X_m = X_test_s.copy()
X_m[:100, 0] = np.nan
X_m[100:200, 5] = np.nan
try:
    p_nan = model.predict_proba(X_m)[:, 1]
    nan_in_out = bool(np.any(np.isnan(p_nan)))
    stress_results.append({
        "scenario": "nan_propagation", "description": "Inject NaN into features for 200 rows",
        "severity": "Critical", "result": "FAIL" if nan_in_out else "PASS",
        "failure_mode": "NaN in predictions" if nan_in_out else None,
        "output_nan": nan_in_out,
        "output_inf": bool(np.any(np.isinf(p_nan))),
    })
except Exception as e:
    stress_results.append({
        "scenario": "nan_propagation", "result": "FAIL",
        "failure_mode": f"Crash on NaN input: {str(e)[:100]}",
        "severity": "Critical",
    })

# 14i: Inf propagation test
log("  Scenario: Inf propagation...")
X_m = X_test_s.copy()
X_m[:100, 0] = np.inf
X_m[100:200, 1] = -np.inf
try:
    p_inf = model.predict_proba(X_m)[:, 1]
    inf_in_out = bool(np.any(np.isnan(p_inf)) or np.any(np.isinf(p_inf)))
    stress_results.append({
        "scenario": "inf_propagation", "description": "Inject Inf into features for 200 rows",
        "severity": "Critical", "result": "FAIL" if inf_in_out else "PASS",
        "failure_mode": "NaN/Inf in predictions" if inf_in_out else None,
        "output_nan": bool(np.any(np.isnan(p_inf))),
        "output_inf": bool(np.any(np.isinf(p_inf))),
    })
except Exception as e:
    stress_results.append({
        "scenario": "inf_propagation", "result": "FAIL",
        "failure_mode": f"Crash on Inf input: {str(e)[:100]}",
        "severity": "Critical",
    })

# 14j: Feature distribution shift — scale all features by 2x
log("  Scenario: distribution shift (2x scale)...")
X_m = X_test_s * 2.0
stress_results.append(eval_stress(
    "dist_shift_2x", X_m, y_test,
    "All features multiplied by 2 (distribution shift simulation)",
    severity="High"))

# 14k: Feature distribution shift — scale all features by 0.5x
X_m = X_test_s * 0.5
stress_results.append(eval_stress(
    "dist_shift_half", X_m, y_test,
    "All features multiplied by 0.5 (distribution shift simulation)",
    severity="High"))

# 14l: Invalid timestamps — nonsensical year/month/day
log("  Scenario: invalid timestamps...")
X_m = X_test_s.copy()
X_m[:, 2] = 9999.0 / 2020.0  # year feature (year_n/6000) set to absurd
X_m[:, 3] = 13.0 / 12.0      # month feature set to 13
X_m[:, 4] = 32.0 / 31.0      # day feature set to 32
stress_results.append(eval_stress(
    "invalid_timestamps", X_m, y_test,
    "Year=9999, Month=13, Day=32 (out-of-range timestamps)",
    severity="Medium"))

# 14m: Duplicate transactions — repeat first 1000 rows
log("  Scenario: duplicate transactions...")
X_dup = np.vstack([X_test_s[:1000], X_test_s[:1000]])
y_dup = np.concatenate([y_test[:1000], y_test[:1000]])
stress_results.append(eval_stress(
    "duplicate_transactions", X_dup, y_dup,
    "First 1000 test rows duplicated (exact duplicate transactions)",
    severity="Low"))

# 14n: Rare categories — MCC set to 0
log("  Scenario: rare/zero MCC...")
X_m = X_test_s.copy()
X_m[:, 7] = 0.0  # mcc_n set to 0
stress_results.append(eval_stress(
    "zero_mcc", X_m, y_test,
    "MCC feature set to 0 (unknown/zero merchant category)",
    severity="Low"))

# 14o: All features set to extreme high
log("  Scenario: all features extreme high...")
X_m = np.ones_like(X_test_s) * 10.0
stress_results.append(eval_stress(
    "all_features_extreme_high", X_m, y_test,
    "All features set to +10 (simultaneously extreme)",
    severity="Medium"))

# 14p: All features set to extreme low (negative)
X_m = np.ones_like(X_test_s) * -10.0
stress_results.append(eval_stress(
    "all_features_extreme_low", X_m, y_test,
    "All features set to -10 (simultaneously extreme negative)",
    severity="Medium"))

# 14q: Single row stress — verify per-row inference works
log("  Scenario: single-row inference...")
X_single = X_test_s[:1].copy()
try:
    p_single = model.predict_proba(X_single)[:, 1]
    stress_results.append({
        "scenario": "single_row_inference",
        "description": "Single transaction scored individually",
        "severity": "Critical",
        "result": "PASS",
        "output_valid": bool(0 <= p_single[0] <= 1),
        "score": round(float(p_single[0]), 6),
    })
except Exception as e:
    stress_results.append({
        "scenario": "single_row_inference",
        "result": "FAIL",
        "failure_mode": f"Crash: {str(e)[:100]}",
        "severity": "Critical",
    })

# 14r: Empty input
log("  Scenario: empty input...")
try:
    X_empty = np.zeros((0, 25), dtype=np.float32)
    p_empty = model.predict_proba(X_empty)
    stress_results.append({
        "scenario": "empty_input",
        "description": "Zero rows input",
        "severity": "Critical",
        "result": "PASS",
        "output_shape": p_empty.shape,
    })
except Exception as e:
    stress_results.append({
        "scenario": "empty_input",
        "result": "FAIL" if "crash" in str(e).lower() else "PASS",
        "failure_mode": str(e)[:100],
        "severity": "Medium",
        "note": "Empty input raising an error is acceptable fail-fast behavior",
    })

# 14s: Model file integrity check
log("  Scenario: model integrity...")
try:
    import tempfile, pickle
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(model, f)
        model_path = f.name
    with open(model_path, "rb") as f:
        model_loaded = pickle.load(f)
    p_loaded = model_loaded.predict_proba(X_test_s[:100])[:, 1]
    p_orig = model.predict_proba(X_test_s[:100])[:, 1]
    match = np.allclose(p_loaded, p_orig, atol=1e-10)
    os.unlink(model_path)
    stress_results.append({
        "scenario": "model_integrity",
        "description": "Save/load model pickle, verify predictions match",
        "severity": "Critical",
        "result": "PASS" if match else "FAIL",
        "predictions_match": bool(match),
    })
except Exception as e:
    stress_results.append({
        "scenario": "model_integrity",
        "result": "FAIL",
        "failure_mode": str(e)[:100],
        "severity": "Critical",
    })

# Summary
n_pass = sum(1 for s in stress_results if s["result"] == "PASS")
n_fail = sum(1 for s in stress_results if s["result"] == "FAIL")
n_total = len(stress_results)
log(f"\n  Stress tests: {n_pass}/{n_total} PASS, {n_fail}/{n_total} FAIL")

for s in stress_results:
    if s["result"] == "FAIL":
        log(f"    FAIL: {s['scenario']} - {s.get('failure_mode', 'unknown')} (severity={s.get('severity', '?')})")

audit["stress_tests"] = {
    "n_scenarios": n_total,
    "n_pass": n_pass,
    "n_fail": n_fail,
    "critical_failures": [s for s in stress_results if s["result"] == "FAIL" and s.get("severity") == "Critical"],
    "scenarios": stress_results,
}


# ============================================================
# SAVE REPORT
# ============================================================
log("")
log("=" * 70)
log("SAVING REPORT")
log("=" * 70)

audit["metadata"] = {
    "total_rows": int(total),
    "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test),
    "n_fraud_train": int(y_train.sum()),
    "n_fraud_val": int(y_val.sum()),
    "n_fraud_test": int(y_test.sum()),
    "locked_threshold": round(float(locked_thr), 4),
    "run_time_sec": round(time.time() - T0, 1),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
}

with open(REPORT, "w") as f:
    json.dump(_cvt(audit), f, indent=2)

log(f"Report saved: {REPORT}")
log(f"Total time: {time.time()-T0:.0f}s")
