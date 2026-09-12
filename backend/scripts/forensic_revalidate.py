#!/usr/bin/env python3
"""
PS-14 FORENSIC FIX & RE-VALIDATION

Fixes:
1. Expanding window: no chunk-level statistics (merch_pop/city_pop median bug)
2. Proper train/validation/test split with NO test-set contamination
3. Early stopping on validation ONLY, not test
4. Threshold selection on validation ONLY
5. Correct bootstrap CIs (preserve prevalence)
6. All confusion-matrix identities enforced
7. Full feature-by-feature leakage classification

Protocol:
  TRAIN (60%) -> VALIDATION (20%) -> FINAL TEST (20%)
  Model/threshold/tuning done on train+val
  Final test is UNTouched until one final evaluation
"""
import json, time, os, sys, gc, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
os.chdir(ROOT)

REPORT = ROOT / "reports" / "forensic_revalidation.json"
REPORT.parent.mkdir(parents=True, exist_ok=True)

T0 = time.time()
SEED = 42
np.random.seed(SEED)

def log(msg):
    t = time.time() - T0
    print(f"[{t:6.0f}s] {msg}", flush=True)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, auc as sk_auc
)
import xgboost as xgb

audit = OrderedDict()

# ================================================================
# FEATURE DEFINITIONS & LEAKAGE CLASSIFICATION
# ================================================================
log("=" * 70)
log("SECTION 0: FEATURE LEAKAGE CLASSIFICATION")
log("=" * 70)

feature_audit = [
    {"name": "log_amt",          "type": "raw",       "uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "amt_sq",           "type": "raw",       "uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "year",             "type": "temporal",  "uses_target": False, "uses_timestamp": True,  "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "month",            "type": "temporal",  "uses_target": False, "uses_timestamp": True,  "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "day",              "type": "temporal",  "uses_target": False, "uses_timestamp": True,  "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "chip",             "type": "raw",       "uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "is_online",        "type": "raw",       "uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "mcc_n",            "type": "raw",       "uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "has_zip",          "type": "raw",       "uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "has_state",        "type": "raw",       "uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "very_high_amt",    "type": "raw",       "uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "amt_x_mcc",        "type": "interaction","uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "amt_x_online",     "type": "interaction","uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "amt_x_chip",       "type": "interaction","uses_target": False, "uses_timestamp": False, "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN"},
    {"name": "user_tx_count",    "type": "expanding",  "uses_target": False, "uses_timestamp": True,  "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN (expanding window, past-only; cold-start: =0 for unseen users)"},
    {"name": "merch_fraud_rate", "type": "expanding_target", "uses_target": True, "uses_timestamp": True, "uses_future": False, "uses_current": False, "available_at_inference": "LABEL_LATENCY", "causal": "CONDITIONAL", "verdict": "REQUIRES LABEL LATENCY ANALYSIS"},
    {"name": "city_fraud_rate",  "type": "expanding_target", "uses_target": True, "uses_timestamp": True, "uses_future": False, "uses_current": False, "available_at_inference": "LABEL_LATENCY", "causal": "CONDITIONAL", "verdict": "REQUIRES LABEL LATENCY ANALYSIS"},
    {"name": "user_fraud_rate",  "type": "expanding_target", "uses_target": True, "uses_timestamp": True, "uses_future": False, "uses_current": False, "available_at_inference": "LABEL_LATENCY", "causal": "CONDITIONAL", "verdict": "REQUIRES LABEL LATENCY ANALYSIS"},
    {"name": "mfr_x_ufr",        "type": "interaction_target", "uses_target": True, "uses_timestamp": True, "uses_future": False, "uses_current": False, "available_at_inference": "LABEL_LATENCY", "causal": "CONDITIONAL", "verdict": "REQUIRES LABEL LATENCY ANALYSIS"},
    {"name": "merch_popularity",  "type": "expanding",  "uses_target": False, "uses_timestamp": True,  "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN (after fix: use expanding median)"},
    {"name": "city_popularity",   "type": "expanding",  "uses_target": False, "uses_timestamp": True,  "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN (after fix: use expanding median)"},
    {"name": "amt_ratio",         "type": "expanding",  "uses_target": False, "uses_timestamp": True,  "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN (cold-start: uses pop_avg_amt as denominator for unseen users)"},
    {"name": "amt_zscore",        "type": "expanding",  "uses_target": False, "uses_timestamp": True,  "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN (cold-start: uses pop_mean/std for unseen users)"},
    {"name": "amt_acceleration",  "type": "expanding",  "uses_target": False, "uses_timestamp": True,  "uses_future": False, "uses_current": False, "available_at_inference": True,  "causal": True,  "verdict": "CLEAN (cold-start: =0 for first transaction of unseen users)"},
    {"name": "user_merch_diversity", "type": "expanding", "uses_target": False, "uses_timestamp": True, "uses_future": False, "uses_current": False, "available_at_inference": True, "causal": True, "verdict": "CLEAN (cold-start: =0 for unseen users, uses pop fallback)"},
]

log(f"  Total features: {len(feature_audit)}")
log(f"  CLEAN: {sum(1 for f in feature_audit if 'CLEAN' in f['verdict'])}")
log(f"  REQUIRES LABEL LATENCY: {sum(1 for f in feature_audit if 'LABEL_LATENCY' in str(f['verdict']))}")
audit["feature_leakage_classification"] = feature_audit

# ================================================================
# SECTION 1: DATASET INTEGRITY
# ================================================================
log("")
log("=" * 70)
log("SECTION 1: DATASET INTEGRITY")
log("=" * 70)

CSV_ALTMAN = "data/credit_card_transactions-ibm_v2.csv"
CHUNK = 2_000_000
USECOLS = ["User", "Card", "Year", "Month", "Day", "Amount", "Use Chip",
           "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
           "Errors?", "Is Fraud?"]

log("  Loading full Altman dataset...")
t1 = time.time()
all_chunks = []
for i, chunk in enumerate(pd.read_csv(CSV_ALTMAN, usecols=USECOLS, low_memory=False, chunksize=CHUNK)):
    all_chunks.append(chunk)
    if i % 3 == 0:
        log(f"  Loaded chunk {i}: cumulative {sum(len(c) for c in all_chunks):,} rows")

df = pd.concat(all_chunks, ignore_index=True)
del all_chunks
gc.collect()

total = len(df)
fraud_mask = df["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
n_fraud = int(fraud_mask.sum())
n_legit = total - n_fraud

log(f"  Total: {total:,} rows, Fraud: {n_fraud:,} ({n_fraud/total*100:.3f}%), Legit: {n_legit:,}")
assert n_fraud + n_legit == total, "INTEGRITY FAIL: fraud + legit != total"
log(f"  Integrity check: PASS")

audit["dataset_integrity"] = {
    "total_rows": total,
    "fraud_rows": n_fraud,
    "legit_rows": n_legit,
    "fraud_rate_pct": round(n_fraud / total * 100, 4),
    "sum_check": True,
}

# ================================================================
# SECTION 2: TEMPORAL SORT + 3-WAY SPLIT (60/20/20)
# ================================================================
log("")
log("=" * 70)
log("SECTION 2: TEMPORAL SORT + 3-WAY SPLIT")
log("=" * 70)

# Verify temporal ordering
df["_sort_key"] = df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-" + df["Day"].astype(str).str.zfill(2)
sort_keys = df["_sort_key"].values

# Check if sorted (sample every 100K)
sample_indices = range(0, len(sort_keys), 100000)
is_sorted = all(sort_keys[i] <= sort_keys[i+1] for i in sample_indices if i+1 < len(sort_keys))
log(f"  Dataset temporally sorted: {is_sorted}")

# 3-way split: 60% train, 20% validation, 20% final test
n_train = int(total * 0.60)
n_val = int(total * 0.20)
n_test = total - n_train - n_val

train_end = f"{int(df.iloc[n_train-1]['Year'])}-{int(df.iloc[n_train-1]['Month']):02d}"
val_start = f"{int(df.iloc[n_train]['Year'])}-{int(df.iloc[n_train]['Month']):02d}"
val_end = f"{int(df.iloc[n_train+n_val-1]['Year'])}-{int(df.iloc[n_train+n_val-1]['Month']):02d}"
test_start = f"{int(df.iloc[n_train+n_val]['Year'])}-{int(df.iloc[n_train+n_val]['Month']):02d}"

train_fraud = int(fraud_mask.iloc[:n_train].sum())
val_fraud = int(fraud_mask.iloc[n_train:n_train+n_val].sum())
test_fraud = int(fraud_mask.iloc[n_train+n_val:].sum())

split_info = {
    "method": "Chronological 60/20/20 split",
    "train_rows": n_train,
    "val_rows": n_val,
    "test_rows": n_test,
    "train_end": train_end,
    "val_start": val_start,
    "val_end": val_end,
    "test_start": test_start,
    "train_fraud": train_fraud,
    "val_fraud": val_fraud,
    "test_fraud": test_fraud,
    "train_fraud_rate": round(train_fraud / n_train * 100, 4),
    "val_fraud_rate": round(val_fraud / n_val * 100, 4),
    "test_fraud_rate": round(test_fraud / n_test * 100, 4),
    "chronological_order": train_end <= val_start and val_end <= test_start,
    "test_set_purpose": "FINAL evaluation ONLY - never used for model/threshold selection",
}
log(f"  Train: {n_train:,} rows (through {train_end}), {train_fraud:,} fraud")
log(f"  Val:   {n_val:,} rows ({val_start} to {val_end}), {val_fraud:,} fraud")
log(f"  Test:  {n_test:,} rows (from {test_start}), {test_fraud:,} fraud")
log(f"  Chronological: {split_info['chronological_order']}")
audit["temporal_split"] = split_info

# ================================================================
# SECTION 3: LEAKAGE-FREE EXPANDING WINDOW (FIXED)
# ================================================================
log("")
log("=" * 70)
log("SECTION 3: LEAKAGE-FREE EXPANDING WINDOW (FIXED)")
log("=" * 70)

FEATURE_NAMES = [
    "log_amt", "amt_sq", "year", "month", "day", "chip", "is_online", "mcc_n",
    "has_zip", "has_state", "user_tx_count", "merch_fraud_rate", "city_fraud_rate",
    "very_high_amt", "amt_x_mcc", "amt_x_online", "merch_popularity",
    "user_fraud_rate", "amt_ratio", "amt_zscore", "amt_acceleration",
    "mfr_x_ufr", "city_popularity", "user_merch_diversity", "amt_x_chip",
]

def expanding_features_fixed(df_chunk, state, pop_state=None):
    """Compute features using ONLY data from 'state' (running stats up to current row).
    
    COLD-START FIX: For unseen users (first transaction), features fall back to
    population-level statistics from pop_state instead of zeros/degenerate values.
    This ensures the model gets meaningful features for new users.
    
    No chunk-level statistics are used. Every feature depends only on
    strictly-past rows.
    """
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
    
    # Expanding window features - row by row
    utc = np.zeros(n, dtype=np.float32)
    ufr = np.zeros(n, dtype=np.float32)
    uavg = np.zeros(n, dtype=np.float32)
    ustd = np.zeros(n, dtype=np.float32)
    mfr = np.zeros(n, dtype=np.float32)
    cfr = np.zeros(n, dtype=np.float32)
    mtc = np.zeros(n, dtype=np.float32)
    accel = np.zeros(n, dtype=np.float32)
    ctc = np.zeros(n, dtype=np.float32)
    
    # Local state dicts for speed
    utc_s = state["user_tx_count"]
    ufc_s = state["user_fraud_count"]
    uta_s = state["user_total_amt"]
    uas_s = state["user_amt_sq"]
    ula_s = state["user_last_amt"]
    mtc_s = state["merch_tx_count"]
    mfc_s = state["merch_fraud_count"]
    ctc_s = state["city_tx_count"]
    cfc_s = state["city_fraud_count"]
    
    # Population-level fallback state for cold-start users
    if pop_state is None:
        pop_state = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0,
                      "total_amt_sq": 0.0, "n_users": 0}
    _ptx = pop_state["total_tx"]
    _pfraud = pop_state["total_fraud"]
    _pamt = pop_state["total_amt"]
    _pamtsq = pop_state["total_amt_sq"]
    _pn = pop_state["n_users"]
    # Population fallback values (safe even if _ptx == 0)
    pop_fraud_rate = _pfraud / max(_ptx, 1)
    pop_avg_amt = _pamt / max(_ptx, 1)
    pop_std_amt = max((_pamtsq / max(_ptx, 1)) - pop_avg_amt ** 2, 1e-10) ** 0.5
    pop_avg_tx_per_user = _ptx / max(_pn, 1)
    
    for i in range(n):
        u = users[i]
        m = merchs[i]
        c = cities[i]
        a = float(amt[i])
        is_new_user = u not in utc_s
        
        # Read state BEFORE this row
        utx = utc_s.get(u, 0)
        utc[i] = utx
        uf = ufc_s.get(u, 0)
        ut = uta_s.get(u, 0.0)
        us = uas_s.get(u, 0.0)
        
        if is_new_user and _ptx > 0:
            # COLD-START: unseen user, use population fallbacks
            ufr[i] = pop_fraud_rate
            ua = pop_avg_amt
            uavg[i] = ua
            ustd[i] = pop_std_amt
            accel[i] = 0.0  # no previous transaction
        else:
            # Normal: user has history
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
        
        # UPDATE state AFTER computing features for this row
        utc_s[u] = utx + 1
        ufc_s[u] = uf + int(is_fraud[i])
        uta_s[u] = ut + a
        uas_s[u] = us + a * a
        ula_s[u] = a
        mtc_s[m] = mt + 1
        mfc_s[m] = mf + int(is_fraud[i])
        ctc_s[c] = ct + 1
        cfc_s[c] = cf + int(is_fraud[i])
        # Update population stats
        _ptx += 1
        _pfraud += int(is_fraud[i])
        _pamt += a
        _pamtsq += a * a
        if is_new_user:
            _pn += 1
    
    # Write back pop_state
    pop_state["total_tx"] = _ptx
    pop_state["total_fraud"] = _pfraud
    pop_state["total_amt"] = _pamt
    pop_state["total_amt_sq"] = _pamtsq
    pop_state["n_users"] = _pn
    
    # Derived features
    very_high_amt = (amt > 5000).astype(float)
    amt_x_mcc = amt * mcc_n
    amt_x_online = amt * is_online
    amt_x_chip = amt * chip
    mfr_x_ufr = mfr * ufr
    
    # FIX: Use expanding count directly as popularity metric
    # (no chunk-level median normalization)
    # merch_popularity = log(1 + merch_tx_count) - capped at reasonable range
    # This is strictly causal: count only increases, never depends on future
    merch_pop = np.minimum(np.log1p(mtc), 5.0)
    city_pop = np.minimum(np.log1p(ctc), 5.0)
    
    # For unseen users: amt_ratio = amt/pop_avg_amt (meaningful ratio)
    # For seen users: amt_ratio = amt/user_avg_amt (personal deviation)
    amt_ratio = amt / np.maximum(uavg, 0.01)
    amt_zscore = (amt - uavg) / np.maximum(ustd, 0.01)
    # For unseen users: diversity = 0 (first transaction, no history)
    umdiv = np.minimum(utc / np.maximum(mtc, 1), 10.0)
    
    F = np.column_stack([
        log_amt, amt_sq,
        year, month, day,
        chip, is_online, mcc_n,
        has_zip, has_state,
        utc,
        mfr, cfr,
        very_high_amt,
        amt_x_mcc, amt_x_online,
        merch_pop,
        ufr,
        amt_ratio, amt_zscore, accel,
        mfr_x_ufr,
        city_pop, umdiv, amt_x_chip,
    ]).astype(np.float32)
    
    F = np.nan_to_num(F, nan=0.0, posinf=10.0, neginf=-10.0)
    return F

# Process 3-way split with expanding window
log("  Processing train set (expanding window)...")
state = {
    "user_tx_count": {}, "user_fraud_count": {}, "user_total_amt": {},
    "user_amt_sq": {}, "user_last_amt": {}, "merch_tx_count": {},
    "merch_fraud_count": {}, "city_tx_count": {}, "city_fraud_count": {},
}
pop_state = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0,
             "total_amt_sq": 0.0, "n_users": 0}

# Process train
train_chunks_F = []
train_y_chunks = []
for start in range(0, n_train, CHUNK):
    end = min(start + CHUNK, n_train)
    chunk = df.iloc[start:end]
    F = expanding_features_fixed(chunk, state, pop_state)
    train_chunks_F.append(F)
    train_y_chunks.append(fraud_mask.iloc[start:end].values)
    log(f"  Train chunk {start//CHUNK+1}: {start:,}-{end:,}")

X_train = np.vstack(train_chunks_F)
y_train = np.concatenate(train_y_chunks)
del train_chunks_F, train_y_chunks
gc.collect()
log(f"  Train features: {X_train.shape}")

# Process validation (state continues expanding from train)
log("  Processing validation set (state continues from train)...")
val_chunks_F = []
val_y_chunks = []
for start in range(n_train, n_train + n_val, CHUNK):
    end = min(start + CHUNK, n_train + n_val)
    chunk = df.iloc[start:end]
    F = expanding_features_fixed(chunk, state, pop_state)
    val_chunks_F.append(F)
    val_y_chunks.append(fraud_mask.iloc[start:end].values)
    log(f"  Val chunk: {start:,}-{end:,}")

X_val = np.vstack(val_chunks_F)
y_val = np.concatenate(val_y_chunks)
del val_chunks_F, val_y_chunks
gc.collect()
log(f"  Val features: {X_val.shape}")

# Process final test (state continues expanding from train+val)
# This is the UNTouched test set - features computed but NEVER used for decisions
log("  Processing FINAL TEST set (state continues from train+val)...")
test_chunks_F = []
test_y_chunks = []
for start in range(n_train + n_val, total, CHUNK):
    end = min(start + CHUNK, total)
    chunk = df.iloc[start:end]
    F = expanding_features_fixed(chunk, state, pop_state)
    test_chunks_F.append(F)
    test_y_chunks.append(fraud_mask.iloc[start:end].values)
    log(f"  Test chunk: {start:,}-{end:,}")

X_test = np.vstack(test_chunks_F)
y_test = np.concatenate(test_y_chunks)
del test_chunks_F, test_y_chunks, df
gc.collect()
log(f"  Test features: {X_test.shape}")

# Scale: fit on train ONLY
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_val_s = sc.transform(X_val)
X_test_s = sc.transform(X_test)  # Only transform, never fit on test
del X_train, X_val, X_test
gc.collect()

log("  Scaling: fit on train only, transform val and test")
audit["preprocessing"] = {
    "scaler": "StandardScaler",
    "fit_on": "train only",
    "transform": "val and test",
    "no_test_leakage": True,
}

# ================================================================
# SECTION 4: TRAIN WITH EARLY STOPPING ON VALIDATION (NOT TEST)
# ================================================================
log("")
log("=" * 70)
log("SECTION 4: TRAINING (early stopping on VALIDATION, not test)")
log("=" * 70)

t_train = time.time()
spw = max(1, int((y_train == 0).sum() / max(int(y_train.sum()), 1)))

# XGBoost with early stopping on VALIDATION set
model = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, scale_pos_weight=min(spw, 30),
    gamma=1, min_child_weight=3, random_state=SEED, n_jobs=6,
    eval_metric="auc",
    early_stopping_rounds=50,
)
# CRITICAL: early stopping uses VALIDATION set, NOT test
model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=100)
train_time = time.time() - t_train

log(f"  Training time: {train_time:.0f}s")
log(f"  Best iteration: {model.best_iteration}")
log(f"  Early stopping used validation set: YES")
audit["training"] = {
    "model": "XGBClassifier",
    "params": {
        "n_estimators": 500, "max_depth": 8, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "scale_pos_weight": min(spw, 30), "gamma": 1, "min_child_weight": 3,
    },
    "early_stopping_rounds": 50,
    "early_stopping_on": "validation set (NOT test)",
    "best_iteration": int(model.best_iteration),
    "training_time_sec": round(train_time, 1),
    "test_set_contamination": "NONE - test never used during training",
}

# ================================================================
# SECTION 4b: ISOTONIC CALIBRATION (fit on validation)
# ================================================================
log("")
log("=" * 70)
log("SECTION 4b: CONFORMAL CALIBRATION (fit on validation)")
log("=" * 70)

# Get raw validation and test predictions
p_val_raw = model.predict_proba(X_val_s)[:, 1]
p_test_raw = model.predict_proba(X_test_s)[:, 1]

# Use raw predictions - conformal threshold instead of score recalibration
p_val = p_val_raw
p_test = p_test_raw

# Conformal threshold: find score quantile at target FPR on validation legit txns
val_legit_scores = p_val_raw[y_val == 0]
conformal_thr_1 = float(np.percentile(val_legit_scores, 99.0))
conformal_thr_05 = float(np.percentile(val_legit_scores, 99.5))
conformal_thr_01 = float(np.percentile(val_legit_scores, 99.9))

log(f"  Conformal thresholds (score quantiles on validation legit):")
log(f"    FPR < 1%:   score > {conformal_thr_1:.6f}")
log(f"    FPR < 0.5%: score > {conformal_thr_05:.6f}")
log(f"    FPR < 0.1%: score > {conformal_thr_01:.6f}")

audit["calibration"] = {
    "method": "Conformal prediction threshold (score quantile on validation legit)",
    "fit_on": "validation set only",
    "note": "Threshold estimated from validation legitimate score distribution, not score recalibration.",
    "conformal_threshold_fpr_lt_1pct": round(conformal_thr_1, 6),
    "conformal_threshold_fpr_lt_05pct": round(conformal_thr_05, 6),
    "conformal_threshold_fpr_lt_01pct": round(conformal_thr_01, 6),
}

# ================================================================
# SECTION 5: THRESHOLD SELECTION ON VALIDATION ONLY
# ================================================================
log("")
log("=" * 70)
log("SECTION 5: THRESHOLD SELECTION ON VALIDATION ONLY")
log("=" * 70)

# Get validation predictions

# Find threshold on VALIDATION that maximizes recall subject to FPR < 0.01
best_val_thr = None
best_val_recall = -1
val_n_legit = int((y_val == 0).sum())
val_n_fraud = int(y_val.sum())

for thr in np.arange(0.01, 1.0, 0.001):
    preds = (p_val >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
    fpr = fp / max(val_n_legit, 1)
    recall = tp / max(val_n_fraud, 1)
    if fpr < 0.009 and recall > best_val_recall:
        best_val_recall = recall
        best_val_thr = thr

log(f"  Validation: {val_n_fraud} fraud, {val_n_legit} legit")
log(f"  Best threshold on validation: {best_val_thr}")
log(f"  Validation recall at FPR<1%: {best_val_recall*100:.1f}%")

# Also find thresholds for FPR < 0.5% and FPR < 0.1% on validation
best_val_thr_05 = None
best_val_recall_05 = -1
best_val_thr_01 = None
best_val_recall_01 = -1

for thr in np.arange(0.01, 1.0, 0.001):
    preds = (p_val >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
    fpr = fp / max(val_n_legit, 1)
    recall = tp / max(val_n_fraud, 1)
    if fpr < 0.005 and recall > best_val_recall_05:
        best_val_recall_05 = recall
        best_val_thr_05 = thr
    if fpr < 0.001 and recall > best_val_recall_01:
        best_val_recall_01 = recall
        best_val_thr_01 = thr

log(f"  Validation threshold for FPR<0.5%: {best_val_thr_05} (recall={best_val_recall_05*100:.1f}%)")
log(f"  Validation threshold for FPR<0.1%: {best_val_thr_01} (recall={best_val_recall_01*100:.1f}%)")

locked_thresholds = {
    "fpr_lt_1pct": best_val_thr,
    "fpr_lt_05pct": best_val_thr_05,
    "fpr_lt_01pct": best_val_thr_01,
}
log(f"  LOCKED THRESHOLDS (selected on validation only): {locked_thresholds}")
audit["threshold_selection"] = {
    "method": "Maximize recall subject to FPR < target, on VALIDATION set only",
    "validation_fraud": val_n_fraud,
    "validation_legit": val_n_legit,
    "locked_thresholds": locked_thresholds,
    "validation_recall_at_fpr_lt_1pct": round(float(best_val_recall), 4),
    "test_set_used_for_threshold": "NO",
}

# ================================================================
# SECTION 6: FINAL TEST EVALUATION (ONE PASS, LOCKED THRESHOLD)
# ================================================================
log("")
log("=" * 70)
log("SECTION 6: FINAL TEST EVALUATION (locked threshold, one pass)")
log("=" * 70)

p_test = model.predict_proba(X_test_s)[:, 1]
n_test_total = len(y_test)
n_test_fraud = int(y_test.sum())
n_test_legit = n_test_total - n_test_fraud

log(f"  Final test: {n_test_total:,} rows ({n_test_fraud:,} fraud, {n_test_legit:,} legit)")

# Evaluate at locked threshold (FPR < 1%)
thr = locked_thresholds["fpr_lt_1pct"]
preds = (p_test >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

# Enforce confusion-matrix identities
assert tp + fn == n_test_fraud, f"IDENTITY FAIL: TP+FN={tp+fn} != fraud={n_test_fraud}"
assert tn + fp == n_test_legit, f"IDENTITY FAIL: TN+FP={tn+fp} != legit={n_test_legit}"
assert tp + tn + fp + fn == n_test_total, f"IDENTITY FAIL: total mismatch"
alerts = tp + fp
assert alerts == tp + fp, "IDENTITY FAIL: alerts != TP+FP"

fpr = fp / max(n_test_legit, 1)
recall = tp / max(n_test_fraud, 1)
precision = tp / max(alerts, 1) if alerts > 0 else 0
specificity = tn / max(n_test_legit, 1)
fnr = fn / max(n_test_fraud, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-10)

log(f"  Locked threshold: {thr}")
log(f"  FPR: {fpr*100:.3f}% (target: <1%)")
log(f"  Recall: {recall*100:.1f}%")
log(f"  Precision: {precision*100:.1f}%")
log(f"  TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,}")
log(f"  Alerts: {alerts:,} ({alerts/n_test_total*10000:.1f} per 10K)")

# Verify FPR < 1% (strict inequality)
fpr_pass = fpr < 0.01
log(f"  FPR < 1% (strict): {fpr*100:.4f}% < 1.0000% = {fpr_pass}")

test_results = {
    "threshold": thr,
    "fpr": round(float(fpr), 6),
    "recall": round(float(recall), 4),
    "precision": round(float(precision), 4),
    "specificity": round(float(specificity), 4),
    "fnr": round(float(fnr), 4),
    "f1": round(float(f1), 4),
    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    "alerts": int(alerts),
    "alerts_per_10k": round(alerts / n_test_total * 10000, 1),
    "n_test_total": n_test_total,
    "n_test_fraud": n_test_fraud,
    "n_test_legit": n_test_legit,
    "fpr_strictly_less_than_1pct": fpr_pass,
    "identities_verified": True,
}
audit["final_test_results"] = test_results

# Also compute ROC-AUC and PR-AUC on final test
auc_test = roc_auc_score(y_test, p_test)
pr_test = average_precision_score(y_test, p_test)
log(f"  ROC-AUC: {auc_test:.4f}")
log(f"  PR-AUC:  {pr_test:.4f}")

test_results["roc_auc"] = round(float(auc_test), 4)
test_results["pr_auc"] = round(float(pr_test), 4)

# ================================================================
# SECTION 7: COMPLETE THRESHOLD SWEEP ON FINAL TEST
# ================================================================
log("")
log("=" * 70)
log("SECTION 7: COMPLETE THRESHOLD SWEEP (final test, raw predictions)")
log("=" * 70)

thresholds_grid = np.concatenate([
    np.arange(0.01, 0.10, 0.002),
    np.arange(0.10, 0.50, 0.005),
    np.arange(0.50, 1.01, 0.01),
])

sweep = []
best_fpr1 = None
best_fpr05 = None
best_fpr01 = None

for thr_s in thresholds_grid:
    preds_s = (p_test >= thr_s).astype(int)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_test, preds_s, labels=[0, 1]).ravel()
    alerts_s = tp_s + fp_s
    
    # Enforce identities
    assert tp_s + fn_s == n_test_fraud
    assert tn_s + fp_s == n_test_legit
    assert alerts_s == tp_s + fp_s
    
    fpr_s = fp_s / max(n_test_legit, 1)
    recall_s = tp_s / max(n_test_fraud, 1)
    prec_s = tp_s / max(alerts_s, 1) if alerts_s > 0 else 0
    
    entry = {
        "threshold": round(float(thr_s), 4),
        "fpr": round(float(fpr_s), 6),
        "recall": round(float(recall_s), 6),
        "precision": round(float(prec_s), 6),
        "tp": int(tp_s), "tn": int(tn_s), "fp": int(fp_s), "fn": int(fn_s),
        "alerts": int(alerts_s),
        "alerts_per_10k": round(alerts_s / n_test_total * 10000, 1),
    }
    sweep.append(entry)
    
    if fpr_s < 0.01 and (best_fpr1 is None or recall_s > best_fpr1["recall"]):
        best_fpr1 = entry
    if fpr_s < 0.005 and (best_fpr05 is None or recall_s > best_fpr05["recall"]):
        best_fpr05 = entry
    if fpr_s < 0.001 and (best_fpr01 is None or recall_s > best_fpr01["recall"]):
        best_fpr01 = entry

log(f"  Thresholds evaluated: {len(sweep)}")
if best_fpr1:
    log(f"  Best FPR<1%: thr={best_fpr1['threshold']}, FPR={best_fpr1['fpr']*100:.3f}%, recall={best_fpr1['recall']*100:.1f}%")
    log(f"    Alerts: {best_fpr1['alerts']:,} (TP={best_fpr1['tp']:,}, FP={best_fpr1['fp']:,}, FN={best_fpr1['fn']:,})")

# Key operating points table
key_ops = {}
for label, bp in [("fpr_lt_1pct", best_fpr1), ("fpr_lt_05pct", best_fpr05), ("fpr_lt_01pct", best_fpr01)]:
    if bp:
        key_ops[label] = bp

audit["threshold_sweep"] = {
    "n_thresholds": len(sweep),
    "key_operating_points": key_ops,
    "all_identity_checks_pass": True,
}

# ================================================================
# SECTION 8: BOOTSTRAP CIs (CORRECT - preserve prevalence)
# ================================================================
log("")
log("=" * 70)
log("SECTION 8: BOOTSTRAP CIs (1000 iterations, preserve prevalence)")
log("=" * 70)

t8 = time.time()
n_boot = 1000
rng_boot = np.random.RandomState(SEED)

# Stratified bootstrap: sample fraud and legit SEPARATELY in original proportions
# This preserves prevalence, giving valid PR-AUC CIs
# Stratified bootstrap on a prevalence-preserving subsample
# Full 4.87M x 1000 is too slow; use 500K subsample that preserves 0.118% prevalence
n_boot_test = len(y_test)
fraud_idx_boot = np.where(y_test == 1)[0]
legit_idx_boot = np.where(y_test == 0)[0]
n_fraud_boot = len(fraud_idx_boot)
prevalence = n_fraud_boot / n_boot_test
n_sub_target = 500_000
n_fraud_sub = min(n_fraud_boot, int(n_sub_target * prevalence))
n_legit_sub = n_sub_target - n_fraud_sub
sub_fraud_idx = rng_boot.choice(fraud_idx_boot, size=n_fraud_sub, replace=False)
sub_legit_idx = rng_boot.choice(legit_idx_boot, size=n_legit_sub, replace=False)
sub_idx = np.concatenate([sub_fraud_idx, sub_legit_idx])
p_test_sub = p_test[sub_idx]
y_test_sub = y_test[sub_idx]
n_sub = len(sub_idx)
log(f"  Full test: {n_boot_test:,} rows ({n_fraud_boot} fraud, {n_boot_test - n_fraud_boot:,} legit, prevalence={prevalence*100:.4f}%)")
log(f"  Subsampled for bootstrap: {n_sub:,} rows ({n_fraud_sub} fraud, {n_legit_sub:,} legit, prevalence={n_fraud_sub/n_sub*100:.4f}%)")
log(f"  Stratified bootstrap: sample fraud and legit separately, preserving prevalence")

boot_aucs = []
boot_prs = []
boot_recalls = []
boot_fprs = []

# Use the LOCKED threshold from Section 6
thr_locked = locked_thresholds["fpr_lt_1pct"]

for b in range(n_boot):
    # Stratified: sample with replacement from each class separately
    idx_fraud_b = rng_boot.choice(n_fraud_sub, size=n_fraud_sub, replace=True)
    idx_legit_b = rng_boot.choice(n_legit_sub, size=n_legit_sub, replace=True)
    idx_b = np.concatenate([idx_fraud_b, idx_legit_b])
    
    y_b = y_test_sub[idx_b]
    p_b = p_test_sub[idx_b]
    
    # Skip if no fraud or no legit in bootstrap
    if y_b.sum() == 0 or (y_b == 0).sum() == 0:
        continue
    
    try:
        auc_b = roc_auc_score(y_b, p_b)
        pr_b = average_precision_score(y_b, p_b)
    except:
        continue
    
    preds_b = (p_b >= thr_locked).astype(int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_b, preds_b, labels=[0, 1]).ravel()
    fpr_b = fp_b / max(int((y_b == 0).sum()), 1)
    rec_b = tp_b / max(int(y_b.sum()), 1)
    
    boot_aucs.append(auc_b)
    boot_prs.append(pr_b)
    boot_recalls.append(rec_b)
    boot_fprs.append(fpr_b)

boot_aucs = np.array(boot_aucs)
boot_prs = np.array(boot_prs)
boot_recalls = np.array(boot_recalls)
boot_fprs = np.array(boot_fprs)

log(f"  Bootstrap: {len(boot_aucs)} valid samples (of {n_boot} total), {time.time()-t8:.0f}s")
log(f"  ROC-AUC: {auc_test:.4f} -> 95% CI [{np.percentile(boot_aucs, 2.5):.4f}, {np.percentile(boot_aucs, 97.5):.4f}]")
log(f"  PR-AUC:  {pr_test:.4f} -> 95% CI [{np.percentile(boot_prs, 2.5):.4f}, {np.percentile(boot_prs, 97.5):.4f}]")
log(f"  Recall:  {recall*100:.1f}% -> 95% CI [{np.percentile(boot_recalls, 2.5)*100:.1f}%, {np.percentile(boot_recalls, 97.5)*100:.1f}%]")
log(f"  FPR:     {fpr*100:.3f}% -> 95% CI [{np.percentile(boot_fprs, 2.5)*100:.3f}%, {np.percentile(boot_fprs, 97.5)*100:.3f}%]")

bootstrap_ci = {
    "n_iterations": n_boot,
    "n_valid": len(boot_aucs),
    "method": f"Stratified bootstrap {n_boot} iters on {n_sub:,}-row prevalence-preserving subsample ({n_fraud_sub} fraud + {n_legit_sub:,} legit, prevalence={n_fraud_sub/n_sub*100:.4f}% matching original {prevalence*100:.4f}%)",
    "threshold": thr_locked,
    "threshold_status": "FIXED (locked threshold from validation, not reselected per bootstrap)",
    "recall_log_value": round(float(recall), 4),
    "roc_auc": {
        "point_estimate": round(float(auc_test), 4),
        "bootstrap_mean": round(float(boot_aucs.mean()), 4),
        "ci_95_lower": round(float(np.percentile(boot_aucs, 2.5)), 4),
        "ci_95_upper": round(float(np.percentile(boot_aucs, 97.5)), 4),
    },
    "pr_auc": {
        "point_estimate": round(float(pr_test), 4),
        "bootstrap_mean": round(float(boot_prs.mean()), 4),
        "ci_95_lower": round(float(np.percentile(boot_prs, 2.5)), 4),
        "ci_95_upper": round(float(np.percentile(boot_prs, 97.5)), 4),
    },
    "recall": {
        "point_estimate": round(float(recall), 4),
        "bootstrap_mean": round(float(boot_recalls.mean()), 4),
        "ci_95_lower": round(float(np.percentile(boot_recalls, 2.5)), 4),
        "ci_95_upper": round(float(np.percentile(boot_recalls, 97.5)), 4),
    },
    "fpr": {
        "point_estimate": round(float(fpr), 6),
        "bootstrap_mean": round(float(boot_fprs.mean()), 6),
        "ci_95_lower": round(float(np.percentile(boot_fprs, 2.5)), 6),
        "ci_95_upper": round(float(np.percentile(boot_fprs, 97.5)), 6),
    },
}
audit["bootstrap_ci"] = bootstrap_ci

# ================================================================
# SECTION 9: CAUSALITY TEST (future-row perturbation)
# ================================================================
log("")
log("=" * 70)
log("SECTION 9: CAUSALITY TEST (future-row perturbation)")
log("=" * 70)

# Test: features for first 100 rows should NOT change when we add/remove future rows
log("  Testing: compute features for first 100 rows alone vs with 50 more rows...")

test_state_a = {k: {} for k in ["user_tx_count", "user_fraud_count", "user_total_amt",
    "user_amt_sq", "user_last_amt", "merch_tx_count", "merch_fraud_count",
    "city_tx_count", "city_fraud_count"]}
test_pop_a = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0,
              "total_amt_sq": 0.0, "n_users": 0}

small_df = pd.concat([
    # Use a small slice from train
], ignore_index=True) if False else None

# Rebuild df temporarily for the test
df_reload = pd.read_csv(CSV_ALTMAN, usecols=USECOLS, low_memory=False, nrows=200)
small_F_a = expanding_features_fixed(df_reload.head(100), test_state_a, test_pop_a)

test_state_b = {k: {} for k in test_state_a}
test_pop_b = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0,
              "total_amt_sq": 0.0, "n_users": 0}
extended_F = expanding_features_fixed(df_reload.head(150), test_state_b, test_pop_b)

# Compare first 100 rows
features_match = np.allclose(small_F_a, extended_F[:100], atol=1e-6)
max_diff = float(np.max(np.abs(small_F_a - extended_F[:100])))

log(f"  Features identical: {features_match}")
log(f"  Max absolute difference: {max_diff:.8f}")
log(f"  Verdict: {'PASS' if features_match else 'FAIL'}")

causality_test = {
    "test": "Features for first 100 rows with vs without 50 future rows",
    "features_identical": bool(features_match),
    "max_absolute_difference": max_diff,
    "verdict": "PASS" if features_match else "FAIL",
}
audit["causality_test"] = causality_test

del df_reload
gc.collect()

# ================================================================
# SECTION 10: PERMUTATION TEST
# ================================================================
log("")
log("=" * 70)
log("SECTION 10: PERMUTATION TEST")
log("=" * 70)

rng_perm = np.random.RandomState(SEED)
n_perm = 10
perm_aucs = []

for i in range(n_perm):
    y_perm = rng_perm.permutation(y_train)
    spw_p = max(1, int((y_perm == 0).sum() / max(int(y_perm.sum()), 1)))
    m_p = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        scale_pos_weight=min(spw_p, 30), random_state=SEED, n_jobs=4,
        eval_metric="auc",
    )
    m_p.fit(X_train_s, y_perm, verbose=False)
    pp = m_p.predict_proba(X_val_s)[:, 1]
    try:
        auc_p = roc_auc_score(y_val, pp)
        perm_aucs.append(auc_p)
    except:
        pass
    log(f"  Permutation {i+1}: AUC={auc_p:.4f}")

perm_aucs = np.array(perm_aucs)
log(f"  Mean permuted AUC: {perm_aucs.mean():.4f} (original: {auc_test:.4f})")

permutation_test = {
    "n_permutations": n_perm,
    "mean_auc_permuted": round(float(perm_aucs.mean()), 4),
    "original_auc": round(float(auc_test), 4),
    "performance_drop": round(float(auc_test - perm_aucs.mean()), 4),
    "genuine_signal": perm_aucs.mean() < 0.6,
}
audit["permutation_test"] = permutation_test

# ================================================================
# SECTION 11: FEATURE ABLATION (identical protocol)
# ================================================================
log("")
log("=" * 70)
log("SECTION 11: FEATURE ABLATION (identical protocol)")
log("=" * 70)

target_features = {"merch_fraud_rate", "city_fraud_rate", "user_fraud_rate", "mfr_x_ufr"}
temporal_agg = target_features | {"merch_popularity", "city_popularity", "user_tx_count",
    "user_merch_diversity", "amt_ratio", "amt_zscore", "amt_acceleration"}
simple_feats = {"log_amt", "amt_sq", "year", "month", "day", "chip", "is_online", "mcc_n", "has_zip", "has_state"}

ablation_configs = [
    ("all_25_features", set(FEATURE_NAMES)),
    ("no_target_features", set(FEATURE_NAMES) - target_features),
    ("no_temporal_aggregates", set(FEATURE_NAMES) - temporal_agg),
    ("simple_baseline_10", simple_feats),
]

ablation_results = {}
for config_name, feat_set in ablation_configs:
    feat_idx = [i for i, n in enumerate(FEATURE_NAMES) if n in feat_set]
    if len(feat_idx) == 0:
        continue
    
    Xtr = X_train_s[:, feat_idx]
    Xv = X_val_s[:, feat_idx]
    Xte = X_test_s[:, feat_idx]
    
    spw_a = max(1, int((y_train == 0).sum() / max(int(y_train.sum()), 1)))
    n_est = 300 if len(feat_idx) <= 10 else 500
    md = 6 if len(feat_idx) <= 10 else 8
    
    m_a = xgb.XGBClassifier(
        n_estimators=n_est, max_depth=md, learning_rate=0.05 if len(feat_idx) <= 10 else 0.03,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(spw_a, 30),
        random_state=SEED, n_jobs=4, eval_metric="auc", early_stopping_rounds=30,
    )
    # Early stopping on VALIDATION, not test
    m_a.fit(Xtr, y_train, eval_set=[(Xv, y_val)], verbose=False)
    
    # Select threshold on VALIDATION
    p_v = m_a.predict_proba(Xv)[:, 1]
    best_thr_a = None
    best_rec_a = -1
    for thr_a in np.arange(0.01, 1.0, 0.005):
        preds_a = (p_v >= thr_a).astype(int)
        tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_val, preds_a, labels=[0, 1]).ravel()
        fpr_a = fp_a / max(int((y_val == 0).sum()), 1)
        rec_a = tp_a / max(int(y_val.sum()), 1)
        if fpr_a < 0.01 and rec_a > best_rec_a:
            best_rec_a = rec_a
            best_thr_a = thr_a
    
    # Evaluate on FINAL TEST with locked threshold
    p_te = m_a.predict_proba(Xte)[:, 1]
    if best_thr_a is not None:
        preds_te = (p_te >= best_thr_a).astype(int)
        tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, preds_te, labels=[0, 1]).ravel()
        fpr_te = fp_te / max(n_test_legit, 1)
        rec_te = tp_te / max(n_test_fraud, 1)
        prec_te = tp_te / max(tp_te + fp_te, 1) if (tp_te + fp_te) > 0 else 0
    else:
        fpr_te = rec_te = prec_te = 0
    
    auc_a = roc_auc_score(y_test, p_te)
    pr_a = average_precision_score(y_test, p_te)
    
    ablation_results[config_name] = {
        "n_features": len(feat_idx),
        "auc": round(float(auc_a), 4),
        "pr_auc": round(float(pr_a), 4),
        "threshold": best_thr_a,
        "recall_at_locked_fpr": round(float(rec_te), 4),
        "precision_at_locked_fpr": round(float(prec_te), 4),
        "fpr_at_locked_fpr": round(float(fpr_te), 6),
    }
    log(f"  {config_name}: {len(feat_idx)} feat, AUC={auc_a:.4f}, PR-AUC={pr_a:.4f}, recall@FPR<1%={rec_te*100:.1f}%")

# Compute deltas
baseline_auc = ablation_results["all_25_features"]["auc"]
baseline_pr = ablation_results["all_25_features"]["pr_auc"]
for k in ablation_results:
    r = ablation_results[k]
    r["auc_delta"] = round((r["auc"] - baseline_auc) * 100, 2)
    r["pr_auc_delta"] = round((r["pr_auc"] - baseline_pr) * 100, 2)

audit["feature_ablation"] = ablation_results

# ================================================================
# SECTION 12: TEMPORAL STABILITY (on final test)
# ================================================================
log("")
log("=" * 70)
log("SECTION 12: TEMPORAL STABILITY")
log("=" * 70)

n_windows = 4
window_size = n_test_total // n_windows
windows = []

for w in range(n_windows):
    start = w * window_size
    end = (w + 1) * window_size if w < n_windows - 1 else n_test_total
    p_w = p_test[start:end]
    y_w = y_test[start:end]
    
    if y_w.sum() == 0:
        continue
    
    auc_w = roc_auc_score(y_w, p_w)
    pr_w = average_precision_score(y_w, p_w)
    n_fraud_w = int(y_w.sum())
    
    preds_w = (p_w >= thr_locked).astype(int)
    tn_w, fp_w, fn_w, tp_w = confusion_matrix(y_w, preds_w, labels=[0, 1]).ravel()
    n_legit_w = int((y_w == 0).sum())
    fpr_w = fp_w / max(n_legit_w, 1)
    rec_w = tp_w / max(n_fraud_w, 1)
    prec_w = tp_w / max(tp_w + fp_w, 1) if (tp_w + fp_w) > 0 else 0
    
    windows.append({
        "window": w + 1,
        "rows": int(end - start),
        "n_fraud": n_fraud_w,
        "auc": round(float(auc_w), 4),
        "pr_auc": round(float(pr_w), 4),
        "fpr": round(float(fpr_w), 6),
        "recall": round(float(rec_w), 4),
        "precision": round(float(prec_w), 4),
        "alerts": int(tp_w + fp_w),
    })
    log(f"  Window {w+1}: AUC={auc_w:.4f}, recall={rec_w*100:.1f}%, FPR={fpr_w*100:.3f}%, fraud={n_fraud_w}")

if len(windows) >= 2:
    auc_vals = [w["auc"] for w in windows]
    recall_vals = [w["recall"] for w in windows]
    fpr_vals = [w["fpr"] for w in windows]
    
    auc_range = max(auc_vals) - min(auc_vals)
    recall_range = max(recall_vals) - min(recall_vals)
    fpr_range = max(fpr_vals) - min(fpr_vals)
    windows_above_fpr1 = sum(1 for f in fpr_vals if f > 0.01)
    
    stability = {
        "windows": windows,
        "auc_range": round(float(auc_range), 4),
        "recall_range_pct_pts": round(float(recall_range * 100), 1),
        "fpr_range_pct_pts": round(float(fpr_range * 100), 3),
        "windows_exceeding_fpr_1pct": windows_above_fpr1,
        "honest_assessment": (
            f"AUC varies by {auc_range:.4f} ({auc_range*100:.2f}pp), "
            f"recall varies by {recall_range*100:.1f}pp "
            f"({min(recall_vals)*100:.1f}% to {max(recall_vals)*100:.1f}%). "
            f"{windows_above_fpr1} of {len(windows)} windows exceed FPR 1%. "
            f"No formal statistical drift test performed."
        ),
    }
    audit["temporal_stability"] = stability
    log(f"  AUC range: {auc_range:.4f}, Recall range: {recall_range*100:.1f}pp")
    log(f"  Windows exceeding FPR 1%: {windows_above_fpr1}/{len(windows)}")

# ================================================================
# SECTION 13: LABEL LATENCY ANALYSIS
# ================================================================
log("")
log("=" * 70)
log("SECTION 13: LABEL LATENCY ANALYSIS")
log("=" * 70)

label_latency = {
    "dataset": "IBM Altman synthetic credit card transactions",
    "label_column": "Is Fraud? (Yes/No)",
    "label_availability_in_dataset": "Static - all labels known at dataset creation",
    "real_world_label_latency": "UNKNOWN - dataset does not contain label-confirmation timestamps",
    "impact_on_fraud_rate_features": (
        "Features merch_fraud_rate, city_fraud_rate, user_fraud_rate, mfr_x_ufr "
        "depend on fraud labels. In production, fraud labels are confirmed hours to days "
        "after the transaction. The expanding-window computation assumes labels are "
        "available instantly, which is unrealistic. These features would have additional "
        "latency in production."
    ),
    "recommendation": (
        "In production, fraud rate features should use only labels confirmed before "
        "the transaction timestamp. Consider a label-availability buffer (e.g., 24h) "
        "or use only pre-confirmation features."
    ),
}
log(f"  Dataset: {label_latency['dataset']}")
log(f"  Real-world label latency: {label_latency['real_world_label_latency']}")
log(f"  Impact: {label_latency['impact_on_fraud_rate_features'][:80]}...")
audit["label_latency_analysis"] = label_latency

# ================================================================
# SECTION 14: DUPLICATE / ENTITY LEAKAGE
# ================================================================
log("")
log("=" * 70)
log("SECTION 14: DUPLICATE / ENTITY LEAKAGE")
log("=" * 70)

# Reload a sample for duplicate check
df_check = pd.read_csv(CSV_ALTMAN, low_memory=False, nrows=1_000_000)
full_dups = int(df_check.duplicated().sum())
del df_check
gc.collect()

entity_leakage = {
    "full_duplicate_rows_in_1M_sample": full_dups,
    "sample_size": 1_000_000,
    "note": "Entity overlap (same users in train/val/test) is expected and intended for temporal split. Users appear across time periods.",
}
log(f"  Full duplicates in 1M sample: {full_dups}")
audit["entity_leakage"] = entity_leakage

# ================================================================
# SECTION 15: FEATURE IMPORTANCE (method-defined)
# ================================================================
log("")
log("=" * 70)
log("SECTION 15: FEATURE IMPORTANCE (method: XGB gain)")
log("=" * 70)

fi = {name: round(float(imp), 4) for name, imp in zip(FEATURE_NAMES, model.feature_importances_)}
top10 = sorted(fi.items(), key=lambda x: -x[1])[:10]
log(f"  Method: XGB built-in gain importance (NOT permutation, NOT SHAP)")
for name, imp in top10:
    log(f"    {name}: {imp:.4f}")

audit["feature_importance"] = {
    "method": "XGBoost built-in gain importance",
    "note": "Gain importance measures total reduction in loss from splits using this feature. NOT permutation importance. NOT SHAP.",
    "top10": top10,
    "all_features": fi,
}

# ================================================================
# SECTION 16: SAVE FINAL REPORT
# ================================================================
log("")
log("=" * 70)
log("SAVING FORENSIC RE-VALIDATION REPORT")
log("=" * 70)

def _cvt(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (bool,)):
        return obj
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

# Final verdict
critical_failures = []
if not causality_test["features_identical"]:
    critical_failures.append(f"Causality test FAILED: max diff = {causality_test['max_absolute_difference']}")
if not fpr_pass:
    critical_failures.append(f"FPR {fpr*100:.4f}% >= 1% (strict)")
if not permutation_test["genuine_signal"]:
    critical_failures.append("Permutation test: model does not learn genuine signal")

verdict = "VALIDATED" if len(critical_failures) == 0 else "NOT VALIDATED"
log(f"  VERDICT: {verdict}")
if critical_failures:
    for f in critical_failures:
        log(f"  BLOCKING: {f}")

audit["final_verdict"] = {
    "verdict": verdict,
    "critical_failures": critical_failures,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "total_time_sec": round(time.time() - T0, 1),
}

with open(REPORT, "w") as f:
    json.dump(_cvt(audit), f, indent=2)

log(f"  Report saved: {REPORT}")
log(f"  Total time: {time.time()-T0:.0f}s")

# Print summary
log("")
log("=" * 70)
log("FINAL SUMMARY")
log("=" * 70)
log(f"  VERDICT: {verdict}")
log(f"  ROC-AUC: {auc_test:.4f} (95% CI: [{bootstrap_ci['roc_auc']['ci_95_lower']}, {bootstrap_ci['roc_auc']['ci_95_upper']}])")
log(f"  PR-AUC:  {pr_test:.4f} (95% CI: [{bootstrap_ci['pr_auc']['ci_95_lower']}, {bootstrap_ci['pr_auc']['ci_95_upper']}])")
log(f"  Recall @ FPR<1%: {recall*100:.1f}% (95% CI: [{bootstrap_ci['recall']['ci_95_lower']*100:.1f}%, {bootstrap_ci['recall']['ci_95_upper']*100:.1f}%])")
log(f"  Precision: {precision*100:.1f}%")
log(f"  FPR: {fpr*100:.3f}% (strict < 1%: {fpr_pass})")
log(f"  Alerts: {alerts:,} ({alerts/n_test_total*10000:.1f} per 10K)")
log(f"  Causality test: {causality_test['verdict']} (max diff: {causality_test['max_absolute_difference']:.8f})")
log(f"  Permutation: genuine signal = {permutation_test['genuine_signal']}")
log(f"  Test set: FINAL TEST (never used for model/threshold selection)")
