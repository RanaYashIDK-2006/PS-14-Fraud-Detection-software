#!/usr/bin/env python3
"""
PS-14 ROLLING FORWARD VALIDATION + POST-DEPLOYMENT MONITORING

1. Rolling forward validation: expanding train windows, fixed validation
2. Feature drift detection against training reference
3. Data-quality monitoring
4. Model-health report
5. Retraining rules and safety protocol
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
# Windows redirects to files default to cp1252, which crashes on non-ASCII
# characters (e.g. arrows). Force UTF-8 on the actual stdout stream.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
os.chdir(ROOT)

REPORT = ROOT / "reports" / "rolling_validation_and_monitoring.json"
REPORT.parent.mkdir(parents=True, exist_ok=True)

T0 = time.time()
SEED = 42
np.random.seed(SEED)

def log(msg):
    t = time.time() - T0
    print(f"[{t:6.0f}s] {msg}", flush=True)

def _cvt(obj):
    if isinstance(obj, (np.bool_, bool)):
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
    """Process a dataframe slice through the expanding window."""
    chunks_F = []
    for start in range(0, len(df_slice), CHUNK):
        end = min(start + CHUNK, len(df_slice))
        chunk = df_slice.iloc[start:end]
        F = expanding_features_fixed(chunk, state, pop_state)
        chunks_F.append(F)
    return np.vstack(chunks_F)


def _select_threshold(p_sel, y_sel, fpr_target=0.009):
    """Select threshold maximizing recall subject to FPR < fpr_target.
    Used ONLY on the window-1 calibration period; the result is locked for
    all subsequent windows."""
    n_legit = int((y_sel == 0).sum())
    n_fraud = int(y_sel.sum())
    best_thr = None
    best_rec = -1
    for thr in np.arange(0.01, 1.0, 0.001):
        preds = (p_sel >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_sel, preds, labels=[0, 1]).ravel()
        fpr = fp / max(n_legit, 1)
        rec = tp / max(n_fraud, 1)
        if fpr < fpr_target and rec > best_rec:
            best_rec = rec
            best_thr = thr
    return 0.5 if best_thr is None else best_thr


def train_and_evaluate(X_fit, y_fit, X_early, y_early, X_eval, y_eval,
                       locked_thr=None, label=""):
    """Train XGB with early stopping on a temporal INTERNAL holdout
    (last 25% of the training period), never on the forward eval period.

    Threshold: window 1 selects on its own (calibration) period and LOCKS it;
    windows 2+ evaluate the forward period with the locked threshold.
    """
    spw = max(1, int((y_fit == 0).sum() / max(int(y_fit.sum()), 1)))
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, scale_pos_weight=min(spw, 30),
        gamma=1, min_child_weight=3, random_state=SEED, n_jobs=4,
        eval_metric="auc", early_stopping_rounds=30,
    )
    # Early stopping uses ONLY the internal holdout (rows inside the train
    # period); the forward validation period never influences training.
    model.fit(X_fit, y_fit, eval_set=[(X_early, y_early)], verbose=False)

    p_eval = model.predict_proba(X_eval)[:, 1]

    if locked_thr is None:
        # Window 1 = calibration period: select + lock the threshold here.
        best_thr = _select_threshold(p_eval, y_eval, fpr_target=0.009)
        thr_source = "selected_on_calibration_period (window 1)"
    else:
        best_thr = locked_thr
        thr_source = "locked_from_window_1"

    preds = (p_eval >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_eval, preds, labels=[0, 1]).ravel()
    fpr = fp / max(int((y_eval == 0).sum()), 1)
    recall = tp / max(int(y_eval.sum()), 1)
    precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0
    auc = roc_auc_score(y_eval, p_eval)
    pr_auc = average_precision_score(y_eval, p_eval)
    alerts = tp + fp

    return {
        "label": label,
        "threshold": round(float(best_thr), 4),
        "threshold_source": thr_source,
        "n_fit": int(len(y_fit)),
        "n_earlystop_holdout": int(len(y_early)),
        "n_eval": int(len(y_eval)),
        "n_fraud_fit": int(y_fit.sum()),
        "n_fraud_eval": int(y_eval.sum()),
        "fraud_rate_eval": round(float(y_eval.mean() * 100), 4),
        "auc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "fpr": round(float(fpr), 6),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "alerts": int(alerts),
        "alerts_per_10k": round(alerts / max(len(y_eval), 1) * 10000, 1),
        "best_iteration": int(model.best_iteration),
    }


# ============================================================
# MAIN
# ============================================================
audit = OrderedDict()

# ============================================================
# PART 1: LOAD DATASET
# ============================================================
log("=" * 70)
log("LOADING DATASET")
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

# ============================================================
# PART 2: ROLLING FORWARD VALIDATION
# ============================================================
log("")
log("=" * 70)
log("ROLLING FORWARD VALIDATION (5 expanding windows)")
log("=" * 70)

# Create 5 windows: each expands training, fixed-size validation
n_windows = 5
val_size = total // (n_windows + 1)  # ~4M per validation window
train_sizes = [val_size * (i + 1) for i in range(n_windows)]

# ---- Resume support: load partial report if present ----
windows = []
locked_thr = None
if REPORT.exists():
    try:
        with open(REPORT) as f:
            prev = json.load(f)
        done = prev.get("rolling_validation", [])
        if done:
            windows = done
            locked_thr = prev.get("locked_threshold")
            log(f"  RESUMING: {len(windows)}/{n_windows} windows already completed "
                f"(locked_thr={locked_thr})")
    except Exception as e:
        log(f"  Partial report unreadable ({e}); starting fresh")
        windows = []

def save_partial():
    """Persist completed windows so a restart resumes instead of redoing work."""
    audit["rolling_validation"] = windows
    audit["locked_threshold"] = locked_thr
    audit["partial"] = {"completed_windows": len(windows), "n_windows": n_windows}
    with open(REPORT, "w") as f:
        json.dump(_cvt(audit), f, indent=2)
    log(f"  [partial saved: {len(windows)}/{n_windows} windows]")

for w in range(len(windows), n_windows):
    n_tr = train_sizes[w]
    n_val = val_size
    n_val_end = min(n_tr + n_val, total)
    n_fit = int(n_tr * 0.75)  # internal temporal holdout = last 25% of train period

    log(f"\n--- Window {w+1}: train=0-{n_tr:,} (fit 0-{n_fit:,}, "
        f"early-stop {n_fit:,}-{n_tr:,}), val={n_tr:,}-{n_val_end:,} ---")

    # Reset state for this window
    state = {k: {} for k in ["user_tx_count", "user_fraud_count", "user_total_amt",
                               "user_amt_sq", "user_last_amt", "merch_tx_count",
                               "merch_fraud_count", "city_tx_count", "city_fraud_count"]}
    pop_state = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0,
                 "total_amt_sq": 0.0, "n_users": 0}

    # Process train (features are causal; fit/early split is row-contiguous)
    log(f"  Computing train features...")
    t1 = time.time()
    X_train = compute_features(df.iloc[:n_tr], state, pop_state)
    y_train = fraud_mask.iloc[:n_tr].values
    log(f"  Train features: {X_train.shape} ({time.time()-t1:.0f}s)")

    X_fit, y_fit = X_train[:n_fit], y_train[:n_fit]
    X_early, y_early = X_train[n_fit:], y_train[n_fit:]

    # Process validation (forward period, completely untouched by training)
    log(f"  Computing val features...")
    X_val = compute_features(df.iloc[n_tr:n_val_end], state, pop_state)
    y_val = fraud_mask.iloc[n_tr:n_val_end].values
    log(f"  Val features: {X_val.shape}")

    # Scale (fit scaler on the fit portion only)
    sc = StandardScaler()
    X_fit_s = sc.fit_transform(X_fit)
    X_early_s = sc.transform(X_early)
    X_val_s = sc.transform(X_val)
    del X_train, X_fit, X_early, X_val
    gc.collect()

    # Train and evaluate (threshold locked after window 1 calibration)
    log(f"  Training XGB...")
    result = train_and_evaluate(X_fit_s, y_fit, X_early_s, y_early,
                                X_val_s, y_val, locked_thr=locked_thr,
                                label=f"Window {w+1}")
    if locked_thr is None:
        locked_thr = result["threshold"]
        log(f"  Locked threshold = {locked_thr:.4f} (window 1 calibration period)")
    windows.append(result)
    save_partial()

    log(f"  Window {w+1}: AUC={result['auc']:.4f}, PR-AUC={result['pr_auc']:.4f}, "
        f"Recall={result['recall']*100:.1f}%, FPR={result['fpr']*100:.3f}%, "
        f"Alerts={result['alerts']:,} ({result['alerts_per_10k']:.1f}/10K), "
        f"thr={result['threshold']:.4f} ({result['threshold_source']})")

audit["rolling_validation"] = windows
audit["locked_threshold"] = locked_thr

# ============================================================
# PART 3: FEATURE DRIFT DETECTION
# ============================================================
log("")
log("=" * 70)
log("FEATURE DRIFT DETECTION")
log("=" * 70)

# Compute features on first 20% (reference) and last 20% (production)
ref_size = total // 5
prod_start = total - ref_size

state_ref = {k: {} for k in ["user_tx_count", "user_fraud_count", "user_total_amt",
                               "user_amt_sq", "user_last_amt", "merch_tx_count",
                               "merch_fraud_count", "city_tx_count", "city_fraud_count"]}
pop_ref = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0, "total_amt_sq": 0.0, "n_users": 0}

log("  Computing reference features (first 20%)...")
X_ref = compute_features(df.iloc[:ref_size], state_ref, pop_ref)
y_ref = fraud_mask.iloc[:ref_size].values

state_prod = {k: {} for k in state_ref}
pop_prod = {"total_tx": 0, "total_fraud": 0, "total_amt": 0.0, "total_amt_sq": 0.0, "n_users": 0}

log("  Computing production features (last 20%)...")
# Process everything up to prod_start to build state
_ = compute_features(df.iloc[:prod_start], state_prod, pop_prod)
X_prod = compute_features(df.iloc[prod_start:], state_prod, pop_prod)
y_prod = fraud_mask.iloc[prod_start:].values

# PSI-style drift detection
drift_results = []
for i, fname in enumerate(FEATURE_NAMES):
    ref_vals = X_ref[:, i]
    prod_vals = X_prod[:, i]

    ref_mean = float(np.mean(ref_vals))
    prod_mean = float(np.mean(prod_vals))
    ref_std = float(np.std(ref_vals))
    prod_std = float(np.std(prod_vals))

    # Population Stability Index (simplified)
    ref_hist, edges = np.histogram(ref_vals, bins=20, range=(-10, 10))
    prod_hist, _ = np.histogram(prod_vals, bins=20, range=(-10, 10))
    ref_pct = np.maximum(ref_hist / max(ref_hist.sum(), 1), 1e-6)
    prod_pct = np.maximum(prod_hist / max(prod_hist.sum(), 1), 1e-6)
    psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

    # Z-test for mean shift
    n_ref = len(ref_vals)
    n_prod = len(prod_vals)
    se = np.sqrt(ref_std**2 / max(n_ref, 1) + prod_std**2 / max(n_prod, 1))
    z_score = abs(prod_mean - ref_mean) / max(se, 1e-10)

    # Drift level
    if psi > 0.25:
        drift_level = "CRITICAL"
    elif psi > 0.10:
        drift_level = "WARNING"
    elif psi > 0.02:
        drift_level = "INFO"
    else:
        drift_level = "HEALTHY"

    drift_results.append({
        "feature": fname,
        "ref_mean": round(ref_mean, 4),
        "prod_mean": round(prod_mean, 4),
        "ref_std": round(ref_std, 4),
        "prod_std": round(prod_std, 4),
        "psi": round(psi, 6),
        "z_score": round(float(z_score), 2),
        "drift_level": drift_level,
    })

    if drift_level in ("CRITICAL", "WARNING"):
        log(f"  {fname}: PSI={psi:.4f} ({drift_level}) z={z_score:.1f}")

n_critical = sum(1 for d in drift_results if d["drift_level"] == "CRITICAL")
n_warning = sum(1 for d in drift_results if d["drift_level"] == "WARNING")
n_healthy = sum(1 for d in drift_results if d["drift_level"] == "HEALTHY")

log(f"  Drift summary: {n_healthy} HEALTHY, {n_warning} WARNING, {n_critical} CRITICAL")
audit["feature_drift"] = {
    "reference_window": f"rows 0-{ref_size:,}",
    "production_window": f"rows {prod_start:,}-{total:,}",
    "n_features": len(FEATURE_NAMES),
    "n_healthy": n_healthy,
    "n_warning": n_warning,
    "n_critical": n_critical,
    "features": drift_results,
}

# ============================================================
# PART 4: DATA QUALITY MONITORING
# ============================================================
log("")
log("=" * 70)
log("DATA QUALITY MONITORING")
log("=" * 70)

# Check the last 100K rows for data quality issues
dq_start = max(0, total - 100_000)
dq_df = df.iloc[dq_start:]

dq_issues = []

# Missing value rates
for col in ["Amount", "Use Chip", "Merchant Name", "Merchant City", "MCC", "Zip"]:
    miss_rate = dq_df[col].isna().mean()
    if miss_rate > 0.05:
        dq_issues.append({"check": f"Missing rate: {col}", "value": f"{miss_rate*100:.1f}%", "level": "WARNING"})
    elif miss_rate > 0.01:
        dq_issues.append({"check": f"Missing rate: {col}", "value": f"{miss_rate*100:.2f}%", "level": "INFO"})

# Invalid timestamps
invalid_ts = ((dq_df["Year"] < 2000) | (dq_df["Year"] > 2030) |
              (dq_df["Month"] < 1) | (dq_df["Month"] > 12) |
              (dq_df["Day"] < 1) | (dq_df["Day"] > 31)).mean()
if invalid_ts > 0:
    dq_issues.append({"check": "Invalid timestamps", "value": f"{invalid_ts*100:.2f}%", "level": "CRITICAL"})

# Unexpected categories
valid_chips = {"Chip Transaction", "Swipe Transaction", "Online Transaction"}
unexpected_chip = (~dq_df["Use Chip"].isin(valid_chips) & dq_df["Use Chip"].notna()).mean()
if unexpected_chip > 0:
    dq_issues.append({"check": "Unexpected Use Chip categories", "value": f"{unexpected_chip*100:.2f}%", "level": "WARNING"})

# Fraud rate change
ref_fraud_rate = fraud_mask.iloc[:ref_size].mean()
prod_fraud_rate = fraud_mask.iloc[prod_start:].mean()
fraud_rate_ratio = prod_fraud_rate / max(ref_fraud_rate, 1e-10)
if fraud_rate_ratio > 2.0 or fraud_rate_ratio < 0.5:
    dq_issues.append({"check": "Fraud rate change", "value": f"ref={ref_fraud_rate*100:.3f}% prod={prod_fraud_rate*100:.3f}% (ratio={fraud_rate_ratio:.2f})", "level": "WARNING"})

# Amount extremes
extreme_amounts = (dq_df["Amount"].str.replace("$", "", regex=False).astype(float) > 50000).mean()
if extreme_amounts > 0.001:
    dq_issues.append({"check": "Extreme amounts (>$50K)", "value": f"{extreme_amounts*100:.2f}%", "level": "INFO"})

# Overall health
if any(d["level"] == "CRITICAL" for d in dq_issues):
    overall_health = "CRITICAL"
elif any(d["level"] == "WARNING" for d in dq_issues):
    overall_health = "WARNING"
else:
    overall_health = "HEALTHY"

log(f"  Data quality: {overall_health}")
for issue in dq_issues:
    log(f"    {issue['level']}: {issue['check']} = {issue['value']}")

audit["data_quality"] = {
    "overall_health": overall_health,
    "sample_window": f"rows {dq_start:,}-{total:,}",
    "issues": dq_issues,
    "ref_fraud_rate": round(float(ref_fraud_rate * 100), 4),
    "prod_fraud_rate": round(float(prod_fraud_rate * 100), 4),
}

# ============================================================
# PART 5: RETRAINING RULES
# ============================================================
log("")
log("=" * 70)
log("RETRAINING RULES")
log("=" * 70)

retraining_rules = {
    "description": "PS-14 operational thresholds for triggering model retraining",
    "rules": [
        {
            "name": "Recall degradation",
            "condition": "Rolling recall falls below 85%",
            "current_value": f"{windows[-1]['recall']*100:.1f}%",
            "triggered": windows[-1]["recall"] < 0.85,
            "action": "Investigate feature drift, retrain if persistent",
        },
        {
            "name": "FPR exceedance",
            "condition": "Rolling FPR exceeds 1.5%",
            "current_value": f"{windows[-1]['fpr']*100:.3f}%",
            "triggered": windows[-1]["fpr"] > 0.015,
            "action": "Retrain with updated threshold selection",
        },
        {
            "name": "PR-AUC drop",
            "condition": "PR-AUC drops below 0.60 (from ~0.80 baseline)",
            "current_value": f"{windows[-1]['pr_auc']:.4f}",
            "triggered": windows[-1]["pr_auc"] < 0.60,
            "action": "Full retrain with fresh data",
        },
        {
            "name": "Feature drift (CRITICAL)",
            "condition": "Any feature PSI > 0.25",
            "current_value": f"{n_critical} features",
            "triggered": n_critical > 0,
            "action": "Investigate feature pipeline, retrain if drift is genuine",
        },
        {
            "name": "Feature drift (WARNING)",
            "condition": "More than 5 features PSI > 0.10",
            "current_value": f"{n_warning} features",
            "triggered": n_warning > 5,
            "action": "Monitor closely, prepare for retrain",
        },
        {
            "name": "Fraud distribution shift",
            "condition": "Fraud rate ratio > 2x or < 0.5x vs reference",
            "current_value": f"{fraud_rate_ratio:.2f}x",
            "triggered": fraud_rate_ratio > 2.0 or fraud_rate_ratio < 0.5,
            "action": "Retrain with updated class weights",
        },
        {
            "name": "Data quality degradation",
            "condition": "Overall health = CRITICAL",
            "current_value": overall_health,
            "triggered": overall_health == "CRITICAL",
            "action": "Fix data pipeline before retraining",
        },
        {
            "name": "Scheduled retrain",
            "condition": "Every 90 days or after 10M new transactions",
            "current_value": "Scheduled",
            "triggered": False,
            "action": "Proactive retrain to prevent gradual degradation",
        },
    ],
}

n_triggered = sum(1 for r in retraining_rules["rules"] if r["triggered"])
log(f"  Retraining rules: {len(retraining_rules['rules'])} defined, {n_triggered} triggered")
for rule in retraining_rules["rules"]:
    status = "TRIGGERED" if rule["triggered"] else "OK"
    log(f"    [{status}] {rule['name']}: {rule['condition']} (current: {rule['current_value']})")

audit["retraining_rules"] = retraining_rules

# ============================================================
# PART 6: MODEL HEALTH REPORT
# ============================================================
log("")
log("=" * 70)
log("MODEL HEALTH REPORT")
log("=" * 70)

latest = windows[-1]
model_health = {
    "current_model_status": "OPERATIONAL" if n_triggered == 0 else "RETRAINING RECOMMENDED",
    "last_validation_date": "2026-09-02",
    "latest_performance": {
        "window": latest["label"],
        "auc": latest["auc"],
        "pr_auc": latest["pr_auc"],
        "recall": latest["recall"],
        "precision": latest["precision"],
        "fpr": latest["fpr"],
        "alerts_per_10k": latest["alerts_per_10k"],
    },
    "drift_status": f"{n_critical} CRITICAL, {n_warning} WARNING, {n_healthy} HEALTHY",
    "data_quality_status": overall_health,
    "production_parity_status": "PARTIAL (expanding window offline vs sliding window production)",
    "retraining_recommendation": "RETRAIN" if n_triggered > 0 else "MONITOR",
    "reason_for_recommendation": (
        f"{n_triggered} retraining rules triggered: " +
        ", ".join(r["name"] for r in retraining_rules["rules"] if r["triggered"])
    ) if n_triggered > 0 else "All metrics within operational thresholds. Continue monitoring.",
    "retraining_safety_protocol": {
        "steps": [
            "1. Current model -> Candidate model (retrained on fresh data)",
            "2. Same untouched evaluation protocol (causal features, val-only threshold)",
            "3. Compare performance + stability across rolling windows",
            "4. Security/data-quality checks",
            "5. Leakage verification (causality test, permutation test)",
            "6. Human approval required before deployment",
            "7. Canary deployment (10% traffic) -> full rollout",
        ],
        "never_auto_deploy": True,
        "require_same_audit": True,
    },
}

for k, v in model_health.items():
    if isinstance(v, dict):
        log(f"  {k}:")
        for kk, vv in v.items():
            log(f"    {kk}: {vv}")
    else:
        log(f"  {k}: {v}")

audit["model_health"] = model_health

# ============================================================
# SAVE REPORT
# ============================================================
audit["metadata"] = {
    "total_rows": total,
    "n_windows": n_windows,
    "run_time_sec": round(time.time() - T0, 1),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
}

with open(REPORT, "w") as f:
    json.dump(_cvt(audit), f, indent=2)

log(f"\nReport saved: {REPORT}")
log(f"Total time: {time.time()-T0:.0f}s")
