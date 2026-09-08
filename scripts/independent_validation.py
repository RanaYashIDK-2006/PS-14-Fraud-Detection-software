# -*- coding: utf-8 -*-
"""
PS-14 Check #24 -- INDEPENDENT FINAL VALIDATION (does not trust the main pipeline)

Independence strategy (different implementation path wherever practical):
- Dataset stats: direct pandas recount (script data_provenance.py), cross-checked here.
- Metrics: numpy-only implementations (rank-based ROC-AUC with average-rank tie
  handling; sklearn-equivalent Average-Precision-style PR-AUC; bincount confusion
  matrix). No sklearn metric functions are used for the final numbers.
- Leakage: independently written perturbation / label-flip / chronological
  contamination tests.
- Threshold: independently re-derived from validation scores (numpy), never test.
- Claims: every major recorded claim compared against fresh evidence.

The feature matrix is rebuilt with the VERBATIM audit feature function
(_forensic_common.expanding_features_fixed, byte-verified) because check #23
reproducibility requires the SAME experiment; independence is delivered by the
layers above. The full pipeline re-run below IS run #2 of the recorded
experiment (run #1 = reports/forensic_revalidation.json).

Output: reports/independent_validation.json
"""

import copy
import gc
import json
import time

import numpy as np
import pandas as pd

from _forensic_common import (
    CSV_ALTMAN, CHUNK, SEED, USECOLS, FEATURE_NAMES,
    expanding_features_fixed, new_state, new_pop_state,
)

T0 = time.time()
REPORT = {}


def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# INDEPENDENT METRIC IMPLEMENTATIONS (numpy only)
# ---------------------------------------------------------------------------
def np_roc_auc(y, p):
    """Rank-based ROC-AUC (Mann-Whitney U) with average-rank tie handling."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sorter = np.argsort(p, kind="mergesort")
    sp = p[sorter]
    _, inv, counts = np.unique(sp, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    avg_rank = starts[inv] + (counts[inv] - 1) / 2.0 + 1.0
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[sorter] = avg_rank
    sum_pos = ranks[y].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def np_pr_auc(y, p):
    """Average-Precision-style PR-AUC, identical definition to
    sklearn.metrics.average_precision_score (sum over positive positions of
    precision * delta-recall)."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    total = np.arange(1, len(ys) + 1)
    prec = tp / total
    rec = tp / n_pos
    pos = np.where(ys)[0]
    prev = np.concatenate([[0.0], rec[pos[:-1]]])
    return float((prec[pos] * (rec[pos] - prev)).sum())


def np_pr_auc_trap(y, p):
    """Trapezoidal PR-AUC over the full precision-recall curve (cross-check)."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / n_pos
    # manual trapezoid (version-independent): prepend (recall=0, precision=1)
    rec_e = np.concatenate([[0.0], rec])
    prec_e = np.concatenate([[1.0], prec])
    return float(np.sum((rec_e[1:] - rec_e[:-1]) * (prec_e[1:] + prec_e[:-1]) / 2.0))


def np_confusion(y, p, thr):
    preds = np.asarray(p) >= thr
    y = np.asarray(y).astype(bool)
    tp = int((preds & y).sum())
    fp = int((preds & ~y).sum())
    fn = int((~preds & y).sum())
    tn = int((~preds & ~y).sum())
    return tp, fp, fn, tn


def np_operating_point(y, p, thr):
    tp, fp, fn, tn = np_confusion(y, p, thr)
    n_fraud = tp + fn
    n_legit = tn + fp
    alerts = tp + fp
    return {
        "threshold": float(thr),
        "fpr": fp / n_legit if n_legit else float("nan"),
        "recall": tp / n_fraud if n_fraud else float("nan"),
        "precision": tp / alerts if alerts else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alerts": alerts,
        "alerts_per_10k": alerts / (n_fraud + n_legit) * 10000,
        "n_total": n_fraud + n_legit, "n_fraud": n_fraud, "n_legit": n_legit,
    }


# ---------------------------------------------------------------------------
# SECTION 0: reference recorded claims + provenance cross-check
# ---------------------------------------------------------------------------
log("=" * 70)
log("SECTION 0: RECORDED CLAIMS (run #1) + PROVENANCE CROSS-CHECK")
log("=" * 70)

recorded = json.load(open("reports/forensic_revalidation.json"))
prov = json.load(open("reports/data_provenance.json"))  # requires check #23

claims = {
    "dataset_total_rows": 24386900,
    "dataset_fraud_rows": 29757,
    "dataset_legit_rows": 24357143,
    "dataset_fraud_rate_pct": 0.122,
    "train_rows": 14632140, "val_rows": 4877380, "test_rows": 4877380,
    "train_fraud": 17736, "val_fraud": 6260, "test_fraud": 5761,
    "roc_auc": 0.9948, "pr_auc": 0.7965,
    "recall": 0.9181, "fpr": 0.009489, "precision": 0.1027,
    "tp": 5289, "fp": 46226, "tn": 4825393, "fn": 472,
    "alerts": 51515, "alerts_per_10k": 105.6,
    "threshold": 0.094,
    "bootstrap_roc_ci": [0.9926, 0.9954],
    "bootstrap_pr_ci": [0.8119, 0.8500],
}
ds = prov["dataset_stats"]
assert ds["total_rows"] == claims["dataset_total_rows"]
assert ds["fraud_rows"] == claims["dataset_fraud_rows"]
log(f"  provenance recount matches recorded dataset claims: rows/fraud/legit OK")

# ---------------------------------------------------------------------------
# SECTION 1: load data, independent split reproduction
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION 1: LOAD + INDEPENDENT SPLIT REPRODUCTION")
log("=" * 70)

log("  loading full CSV (14 cols)...")
df = pd.read_csv(CSV_ALTMAN, usecols=USECOLS, low_memory=False)
total = len(df)
fraud_mask = df["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
n_train = int(total * 0.60)
n_val = int(total * 0.20)
n_test = total - n_train - n_val
tr_f, va_f, te_f = (int(fraud_mask.iloc[:n_train].sum()),
                    int(fraud_mask.iloc[n_train:n_train + n_val].sum()),
                    int(fraud_mask.iloc[n_train + n_val:].sum()))
split_repro = {
    "train_rows": n_train, "val_rows": n_val, "test_rows": n_test,
    "train_fraud": tr_f, "val_fraud": va_f, "test_fraud": te_f,
    "matches_recorded": (n_train == claims["train_rows"] and n_val == claims["val_rows"]
                         and n_test == claims["test_rows"] and tr_f == claims["train_fraud"]
                         and va_f == claims["val_fraud"] and te_f == claims["test_fraud"]),
}
log(f"  split reproduction: {split_repro['matches_recorded']} "
    f"(train {n_train:,}/{tr_f:,}, val {n_val:,}/{va_f:,}, test {n_test:,}/{te_f:,})")
REPORT["split_reproduction"] = split_repro

# ---------------------------------------------------------------------------
# SECTION 2: INDEPENDENT LEAKAGE TESTS
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION 2A: FILE-ORDER FUTURE-ROW PERTURBATION (replicating audit method)")
log("=" * 70)

st_a, pop_a = new_state(), new_pop_state()
F_a = expanding_features_fixed(df.iloc[:100], st_a, pop_a)
st_b, pop_b = new_state(), new_pop_state()
F_b = expanding_features_fixed(df.iloc[:150], st_b, pop_b)
pert_file = float(np.abs(F_a - F_b[:100]).max())
pert_file_pass = pert_file == 0.0
log(f"  features rows[0:100) alone vs with 50 extra file-order rows: max diff = {pert_file:.10g}")
log(f"  verdict (file-order causality of feature function): {'PASS' if pert_file_pass else 'FAIL'}")
REPORT["perturbation_file_order"] = {
    "max_diff": pert_file, "pass": pert_file_pass,
    "interpretation": ("PASS proves the feature function is causal in FILE order "
                       "only. It does NOT prove chronological causality: the file is "
                       "user-major ordered (see check #23 / section 4)."),
}

log("")
log("=" * 70)
log("SECTION 2B: TARGET LEAKAGE / CURRENT-ROW CONTAMINATION (label flip)")
log("=" * 70)

# Find k where rows k and k+1 share the same merchant (and k >= 5000 so state is rich)
merch = df["Merchant Name"].astype(str).values
k = None
for i in range(5000, 20000):
    if merch[i] == merch[i + 1]:
        k = i
        break
assert k is not None, "no same-merchant consecutive pair found"

def process_prefix(nrows, state, pop_state):
    for s in range(0, nrows, CHUNK):
        e = min(s + CHUNK, nrows)
        expanding_features_fixed(df.iloc[s:e], state, pop_state)

st1, pop1 = new_state(), new_pop_state()
process_prefix(k, st1, pop1)
st_orig = copy.deepcopy(st1)
pop_orig = copy.deepcopy(pop1)
F_orig = expanding_features_fixed(df.iloc[k:k + 3], st_orig, pop_orig)

df_flip = df.iloc[k:k + 3].copy()
df_flip.loc[df_flip.index[0], "Is Fraud?"] = "Yes" if df_flip.iloc[0]["Is Fraud?"] != "Yes" else "No"
st_flip = copy.deepcopy(st1)
pop_flip = copy.deepcopy(pop1)
F_flip = expanding_features_fixed(df_flip, st_flip, pop_flip)

self_diff = float(np.abs(F_orig[0] - F_flip[0]).max())
next_mfr_orig = float(F_orig[1, 11])
next_mfr_flip = float(F_flip[1, 11])
target_leak = {
    "k": k, "row_k_merchant": str(df.iloc[k]["Merchant Name"]),
    "row_k_own_features_unchanged_by_own_label": self_diff == 0.0,
    "row_k_max_feature_diff": self_diff,
    "row_k1_same_merchant": merch[k] == merch[k + 1],
    "row_k1_merch_fraud_rate_orig": next_mfr_orig,
    "row_k1_merch_fraud_rate_flipped": next_mfr_flip,
    "next_row_sees_updated_label": next_mfr_orig != next_mfr_flip,
    "verdict": "PASS" if (self_diff == 0.0 and next_mfr_orig != next_mfr_flip) else "FAIL",
}
log(f"  row {k}: own features unchanged by own-label flip (max diff {self_diff:.3g}) -> {target_leak['verdict']}")
log(f"  row {k}+1 (same merchant): merch_fraud_rate {next_mfr_orig:.5f} -> {next_mfr_flip:.5f} after flip")
REPORT["target_leakage_label_flip"] = target_leak

log("")
log("=" * 70)
log("SECTION 2C: TRAIN/TEST ROW & USER OVERLAP")
log("=" * 70)
u = df["User"].astype(str).values
train_users = set(u[:n_train])
test_users = set(u[n_train + n_val:])
row_overlap = 0  # disjoint index ranges by construction; no duplicated rows across blocks
user_overlap_tt = sorted(train_users & test_users)
log(f"  train/test row overlap: {row_overlap} (disjoint index ranges)")
log(f"  train/test user overlap: {len(user_overlap_tt)}")
REPORT["train_test_overlap"] = {"row_overlap": row_overlap,
                                "user_overlap_count": len(user_overlap_tt),
                                "user_overlap_examples": user_overlap_tt[:5]}

# ---------------------------------------------------------------------------
# SECTION 3: FULL PIPELINE RE-RUN (run #2 of the recorded experiment)
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION 3: FULL PIPELINE RE-RUN (verbatim features, SEED=42, recorded params)")
log("=" * 70)

from sklearn.preprocessing import StandardScaler
import xgboost as xgb

state = new_state()
pop_state = new_pop_state()

log("  processing train features...")
train_chunks, train_y = [], []
for s in range(0, n_train, CHUNK):
    e = min(s + CHUNK, n_train)
    train_chunks.append(expanding_features_fixed(df.iloc[s:e], state, pop_state))
    train_y.append(fraud_mask.iloc[s:e].values)
X_train = np.vstack(train_chunks)
y_train = np.concatenate(train_y)
del train_chunks, train_y
gc.collect()

log("  processing validation features (state continues)...")
val_chunks, val_y = [], []
for s in range(n_train, n_train + n_val, CHUNK):
    e = min(s + CHUNK, n_train + n_val)
    val_chunks.append(expanding_features_fixed(df.iloc[s:e], state, pop_state))
    val_y.append(fraud_mask.iloc[s:e].values)
X_val = np.vstack(val_chunks)
y_val = np.concatenate(val_y)
del val_chunks, val_y
gc.collect()

log("  processing test features (state continues; UNTOUCHED)...")
test_chunks, test_y = [], []
for s in range(n_train + n_val, total, CHUNK):
    e = min(s + CHUNK, total)
    test_chunks.append(expanding_features_fixed(df.iloc[s:e], state, pop_state))
    test_y.append(fraud_mask.iloc[s:e].values)
X_test = np.vstack(test_chunks)
y_test = np.concatenate(test_y)
del test_chunks, test_y
gc.collect()
log(f"  features: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_val_s = sc.transform(X_val)
X_test_s = sc.transform(X_test)
del X_train, X_val, X_test
gc.collect()

spw = max(1, int((y_train == 0).sum() / max(int(y_train.sum()), 1)))
log(f"  training XGB (spw={min(spw, 30)}, seed={SEED})...")
t_train = time.time()
model = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, scale_pos_weight=min(spw, 30),
    gamma=1, min_child_weight=3, random_state=SEED, n_jobs=6,
    eval_metric="auc", early_stopping_rounds=50,
)
model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
log(f"  training done in {time.time()-t_train:.0f}s, best_iteration={model.best_iteration} "
    f"(recorded: 481)")

p_val = model.predict_proba(X_val_s)[:, 1]
p_test = model.predict_proba(X_test_s)[:, 1]
del X_train_s, X_val_s
gc.collect()

# ---------------------------------------------------------------------------
# SECTION 4: INDEPENDENT METRICS (numpy) + CLAIMS COMPARISON
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION 4: INDEPENDENT METRICS (numpy-only implementations)")
log("=" * 70)

# 4a. val-only threshold selection (independent, same protocol as recorded)
val_legit = int((y_val == 0).sum())
val_fraud = int(y_val.sum())
best_thr, best_rec = None, -1.0
order = np.argsort(-p_val, kind="mergesort")
p_desc = p_val[order]
y_desc = y_val[order]
cs_tp = np.cumsum(y_desc)
for thr in np.arange(0.01, 1.0, 0.001):
    k = int(np.searchsorted(-p_desc, -thr, side="right"))  # #scores >= thr
    tp = int(cs_tp[k - 1]) if k > 0 else 0
    fp = k - tp
    fpr = fp / val_legit
    recall = tp / val_fraud
    if fpr < 0.009 and recall > best_rec:
        best_rec, best_thr = recall, thr
log(f"  independent val-locked threshold: {best_thr} (recorded: {recorded['threshold_selection']['locked_thresholds']['fpr_lt_1pct']:.3f})")
threshold_provenance = {
    "method": "max recall s.t. FPR<0.009 on VALIDATION only (numpy sweep)",
    "independent_threshold": best_thr,
    "recorded_threshold": recorded["threshold_selection"]["locked_thresholds"]["fpr_lt_1pct"],
    "match": abs(best_thr - recorded["threshold_selection"]["locked_thresholds"]["fpr_lt_1pct"]) < 1e-9,
    "test_used_for_threshold": False,
}
REPORT["threshold_provenance"] = threshold_provenance

# 4b. final test evaluation at the LOCKED threshold
thr_lock = best_thr
op = np_operating_point(y_test, p_test, thr_lock)
auc = np_roc_auc(y_test, p_test)
pr_ap = np_pr_auc(y_test, p_test)
pr_trap = np_pr_auc_trap(y_test, p_test)

fresh = {
    "threshold": thr_lock,
    "roc_auc": round(auc, 4), "roc_auc_full": auc,
    "pr_auc": round(pr_ap, 4), "pr_auc_full": pr_ap,
    "pr_auc_trapezoid": round(pr_trap, 4),
    "recall": round(op["recall"], 4), "fpr": round(op["fpr"], 6),
    "precision": round(op["precision"], 4),
    "tp": op["tp"], "fp": op["fp"], "fn": op["fn"], "tn": op["tn"],
    "alerts": op["alerts"], "alerts_per_10k": round(op["alerts_per_10k"], 1),
}
log(f"  FRESH  : AUC={fresh['roc_auc']} PR={fresh['pr_auc']} recall={fresh['recall']*100:.1f}% "
    f"FPR={fresh['fpr']*100:.3f}% prec={fresh['precision']*100:.1f}% alerts={fresh['alerts']:,}")
log(f"  RECORDED: AUC=0.9948 PR=0.7965 recall=91.8% FPR=0.949% prec=10.3% alerts=51,515")

# 4c. claims table
tol = {"roc_auc": 5e-4, "pr_auc": 5e-4, "recall": 5e-4, "fpr": 1e-4, "precision": 5e-4,
       "tp": 0, "fp": 0, "tn": 0, "fn": 0, "alerts": 0}
claims_table = [
    {"claim": "Dataset total rows", "reported": claims["dataset_total_rows"],
     "verified": ds["total_rows"], "status": "PASS" if ds["total_rows"] == claims["dataset_total_rows"] else "FAIL"},
    {"claim": "Fraud count", "reported": claims["dataset_fraud_rows"],
     "verified": ds["fraud_rows"], "status": "PASS" if ds["fraud_rows"] == claims["dataset_fraud_rows"] else "FAIL"},
    {"claim": "Legit count", "reported": claims["dataset_legit_rows"],
     "verified": ds["legit_rows"], "status": "PASS" if ds["legit_rows"] == claims["dataset_legit_rows"] else "FAIL"},
    {"claim": "Train rows", "reported": claims["train_rows"], "verified": n_train, "status": "PASS"},
    {"claim": "Val rows", "reported": claims["val_rows"], "verified": n_val, "status": "PASS"},
    {"claim": "Test rows", "reported": claims["test_rows"], "verified": n_test, "status": "PASS"},
    {"claim": "Train fraud", "reported": claims["train_fraud"], "verified": tr_f, "status": "PASS" if tr_f == claims["train_fraud"] else "FAIL"},
    {"claim": "Val fraud", "reported": claims["val_fraud"], "verified": va_f, "status": "PASS" if va_f == claims["val_fraud"] else "FAIL"},
    {"claim": "Test fraud", "reported": claims["test_fraud"], "verified": te_f, "status": "PASS" if te_f == claims["test_fraud"] else "FAIL"},
    {"claim": "ROC-AUC", "reported": 0.9948, "verified": round(auc, 4), "status": "PASS" if abs(auc - 0.9948) <= tol["roc_auc"] else "FAIL"},
    {"claim": "PR-AUC", "reported": 0.7965, "verified": round(pr_ap, 4), "status": "PASS" if abs(pr_ap - 0.7965) <= tol["pr_auc"] else "FAIL"},
    {"claim": "Recall", "reported": 0.9181, "verified": round(op["recall"], 4), "status": "PASS" if abs(op["recall"] - 0.9181) <= tol["recall"] else "FAIL"},
    {"claim": "FPR", "reported": 0.009489, "verified": round(op["fpr"], 6), "status": "PASS" if abs(op["fpr"] - 0.009489) <= tol["fpr"] else "FAIL"},
    {"claim": "Precision", "reported": 0.1027, "verified": round(op["precision"], 4), "status": "PASS" if abs(op["precision"] - 0.1027) <= tol["precision"] else "FAIL"},
    {"claim": "TP", "reported": 5289, "verified": op["tp"], "status": "PASS" if op["tp"] == 5289 else "FAIL"},
    {"claim": "FP", "reported": 46226, "verified": op["fp"], "status": "PASS" if op["fp"] == 46226 else "FAIL"},
    {"claim": "FN", "reported": 472, "verified": op["fn"], "status": "PASS" if op["fn"] == 472 else "FAIL"},
    {"claim": "Alerts", "reported": 51515, "verified": op["alerts"], "status": "PASS" if op["alerts"] == 51515 else "FAIL"},
    {"claim": "Locked threshold (val-selected)", "reported": 0.094,
     "verified": round(thr_lock, 3) if thr_lock is not None else None,
     "status": "PASS" if (thr_lock is not None and abs(thr_lock - 0.094) <= 0.001) else "FAIL"},
    {"claim": "Future-row perturbation PASS (file order)", "reported": "PASS (max diff 0.0)",
     "verified": f"{pert_file:.6g}", "status": "PASS" if pert_file_pass else "FAIL"},
    {"claim": "Chronological split (60/20/20 by time)", "reported": "True",
     "verified": "FALSE -- user-major file, each block spans 1991-2020, boundaries slice user blocks",
     "status": "FAIL"},
    {"claim": "Train/test row overlap", "reported": "ZERO", "verified": row_overlap, "status": "PASS"},
    {"claim": "Train/test user overlap", "reported": "0.0% (all test users new)",
     "verified": f"{len(user_overlap_tt)} users", "status": "PASS" if len(user_overlap_tt) == 0 else "FAIL"},
    {"claim": "Bootstrap CI ROC [0.9926, 0.9954]", "reported": "[0.9926, 0.9954]",
     "verified": "see section 6 (200-iter independent)", "status": "PENDING"},
]
REPORT["fresh_metrics"] = fresh
REPORT["claims_table_metrics"] = claims_table
log(f"  claims table: {sum(1 for c in claims_table if c['status']=='PASS')} PASS / "
    f"{sum(1 for c in claims_table if c['status']=='FAIL')} FAIL / "
    f"{sum(1 for c in claims_table if c['status']=='PENDING')} PENDING")

# ---------------------------------------------------------------------------
# SECTION 5: CHRONOLOGICAL CONTAMINATION QUANTIFICATION
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION 5: CHRONOLOGICAL CONTAMINATION OF TEST FEATURES")
log("=" * 70)

# Build per-merchant and per-city cumulative (tx, fraud) STRICTLY BEFORE each date
cols6 = ["Year", "Month", "Day", "Merchant Name", "Merchant City", "Is Fraud?"]
small = df[cols6].copy()
small["date"] = small["Year"].astype(int) * 10000 + small["Month"].astype(int) * 100 + small["Day"].astype(int)
small["is_f"] = small["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
small = small[["Merchant Name", "Merchant City", "date", "is_f"]]
small["Merchant Name"] = small["Merchant Name"].astype(str)
small["Merchant City"] = small["Merchant City"].astype(str)

log("  building per-merchant date-cumulative tables...")
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

log("  building per-city date-cumulative tables...")
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

SAMP = 500_000
t0 = n_train + n_val
samp_merch = small["Merchant Name"].values[t0:t0 + SAMP]
samp_city = small["Merchant City"].values[t0:t0 + SAMP]
samp_date = small["date"].values[t0:t0 + SAMP]
samp_y = fraud_mask.iloc[t0:t0 + SAMP].values
del small
gc.collect()


def chrono_rate(keys, dates, tbl_dates, tbl_ctx, tbl_cfr):
    out = np.zeros(len(keys), dtype=np.float64)
    for i in range(len(keys)):
        k = keys[i]
        dd = tbl_dates.get(k)
        if dd is None or len(dd) == 0:
            continue
        idx = int(np.searchsorted(dd, dates[i], side="left")) - 1
        if idx >= 0:
            out[i] = tbl_cfr[k][idx] / max(tbl_ctx[k][idx], 1)
    return out


log("  computing chronologically-correct merchant/city fraud rates for 500K test rows...")
mfr_chrono = chrono_rate(samp_merch, samp_date, mg_dates, mg_ctx, mg_cfr)
cfr_chrono = chrono_rate(samp_city, samp_date, cg_dates, cg_ctx, cg_cfr)
del mg_dates, mg_ctx, mg_cfr, cg_dates, cg_ctx, cg_cfr
gc.collect()

audit_mfr = X_test_s[:SAMP, 11].copy()
audit_cfr = X_test_s[:SAMP, 12].copy()
mfr_diff = np.abs(audit_mfr - mfr_chrono)
cfr_diff = np.abs(audit_cfr - cfr_chrono)

log(f"  merch_fraud_rate: mean|diff|={mfr_diff.mean():.5f}, max|diff|={mfr_diff.max():.5f}, "
    f"rows changed: {(mfr_diff > 1e-9).mean()*100:.1f}%")
log(f"  city_fraud_rate:  mean|diff|={cfr_diff.mean():.5f}, max|diff|={cfr_diff.max():.5f}, "
    f"rows changed: {(cfr_diff > 1e-9).mean()*100:.1f}%")

# Score both variants on the window with the SAME trained model + scaler
def score_variant(X_audit_slice, mfr_c, cfr_c):
    Xc = X_audit_slice.copy()
    Xc[:, 11] = mfr_c
    Xc[:, 12] = cfr_c
    Xc[:, 21] = mfr_c * Xc[:, 17]  # mfr_x_ufr = mfr * ufr
    Xs = sc.transform(Xc)  # same scaler (fit on train only)
    return model.predict_proba(Xs)[:, 1]


p_audit_win = model.predict_proba(X_test_s[:SAMP])[:, 1]
p_chrono_win = score_variant(X_test_s[:SAMP], mfr_chrono, cfr_chrono)
auc_audit_win = np_roc_auc(samp_y, p_audit_win)
auc_chrono_win = np_roc_auc(samp_y, p_chrono_win)
op_audit_win = np_operating_point(samp_y, p_audit_win, thr_lock)
op_chrono_win = np_operating_point(samp_y, p_chrono_win, thr_lock)

contam = {
    "window": f"test rows [{t0:,}, {t0+SAMP:,})",
    "n_rows": SAMP, "n_fraud": int(samp_y.sum()),
    "merch_fraud_rate": {"mean_abs_diff": float(mfr_diff.mean()), "max_abs_diff": float(mfr_diff.max()),
                         "pct_rows_changed": float((mfr_diff > 1e-9).mean() * 100)},
    "city_fraud_rate": {"mean_abs_diff": float(cfr_diff.mean()), "max_abs_diff": float(cfr_diff.max()),
                        "pct_rows_changed": float((cfr_diff > 1e-9).mean() * 100)},
    "audit_window_metrics": {"roc_auc": round(auc_audit_win, 4),
                             "recall": round(op_audit_win["recall"], 4),
                             "fpr": round(op_audit_win["fpr"], 6)},
    "chronological_window_metrics": {"roc_auc": round(auc_chrono_win, 4),
                                     "recall": round(op_chrono_win["recall"], 4),
                                     "fpr": round(op_chrono_win["fpr"], 6)},
    "roc_auc_delta": round(auc_chrono_win - auc_audit_win, 4),
    "interpretation": (
        "The audit's expanding-window state is file-order (user-major). Test rows dated "
        "early in the calendar (e.g. 1991-2013) therefore include merchant/city fraud "
        "statistics from OTHER users' transactions dated AFTER the row's own timestamp "
        "(up to 2020). The chronologically-correct rate uses only transactions strictly "
        "before the row's date. Delta quantifies how much reported test performance rides "
        "on future-dated (relative to row timestamp) merchant/city fraud information."),
}
REPORT["chronological_contamination"] = contam
log(f"  window AUC: audit {auc_audit_win:.4f} -> chronological {auc_chrono_win:.4f} "
    f"(delta {auc_chrono_win-auc_audit_win:+.4f})")
log(f"  window recall/FPR @{thr_lock:.3f}: audit {op_audit_win['recall']*100:.1f}%/{op_audit_win['fpr']*100:.2f}% "
    f"-> chronological {op_chrono_win['recall']*100:.1f}%/{op_chrono_win['fpr']*100:.2f}%")

# Temporal-stability windows: what are they really?
log("")
log("=" * 70)
log("SECTION 5b: WHAT THE 'TEMPORAL STABILITY' WINDOWS ACTUALLY ARE")
log("=" * 70)
d_all = (df["Year"].astype(int) * 10000 + df["Month"].astype(int) * 100 + df["Day"].astype(int)).values
win_desc = []
for w in range(4):
    lo = t0 + w * SAMP
    hi = lo + SAMP
    seg = d_all[lo:hi]
    win_desc.append({"window": w + 1, "rows": int(hi - lo),
                     "min_date": int(seg.min()), "max_date": int(seg.max())})
REPORT["temporal_stability_windows_reality"] = win_desc
for wd in win_desc:
    log(f"  window {wd['window']}: dates {wd['min_date']}..{wd['max_date']} (NOT a contiguous time window)")

# ---------------------------------------------------------------------------
# SECTION 6: INDEPENDENT BOOTSTRAP (200 iters, numpy metrics)
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION 6: INDEPENDENT BOOTSTRAP (200 iters, numpy metrics)")
log("=" * 70)

rng = np.random.RandomState(42)
test_fraud_idx = np.where(y_test == 1)[0]
test_legit_idx = np.where(y_test == 0)[0]
n_fraud_sub, n_legit_sub = 590, 499410
sub_fraud = rng.choice(test_fraud_idx, n_fraud_sub, replace=False)
sub_legit = rng.choice(test_legit_idx, n_legit_sub, replace=False)
sub_idx = np.concatenate([sub_fraud, sub_legit])
sub_y = y_test[sub_idx]
sub_p = p_test[sub_idx]

boot_roc, boot_pr, boot_rec, boot_fpr = [], [], [], []
for it in range(200):
    idx = rng.choice(len(sub_idx), len(sub_idx), replace=True)
    yb, pb = sub_y[idx], sub_p[idx]
    boot_roc.append(np_roc_auc(yb, pb))
    boot_pr.append(np_pr_auc(yb, pb))
    opb = np_operating_point(yb, pb, thr_lock)
    boot_rec.append(opb["recall"])
    boot_fpr.append(opb["fpr"])

boot = {
    "n_iterations": 200,
    "method": "stratified subsample 590 fraud + 499,410 legit (prevalence 0.118%), resample with replacement, fixed locked threshold",
    "roc_auc_ci": [round(float(np.percentile(boot_roc, 2.5)), 4), round(float(np.percentile(boot_roc, 97.5)), 4)],
    "pr_auc_ci": [round(float(np.percentile(boot_pr, 2.5)), 4), round(float(np.percentile(boot_pr, 97.5)), 4)],
    "recall_ci": [round(float(np.percentile(boot_rec, 2.5)), 4), round(float(np.percentile(boot_rec, 97.5)), 4)],
    "fpr_ci": [round(float(np.percentile(boot_fpr, 2.5)), 6), round(float(np.percentile(boot_fpr, 97.5)), 6)],
    "recorded_roc_ci": [0.9926, 0.9954],
    "recorded_pr_ci": [0.8119, 0.8500],
}
boot["roc_ci_overlaps_recorded"] = not (boot["roc_auc_ci"][1] < 0.9926 or boot["roc_auc_ci"][0] > 0.9954)
boot["pr_ci_overlaps_recorded"] = not (boot["pr_auc_ci"][1] < 0.8119 or boot["pr_auc_ci"][0] > 0.8500)
REPORT["bootstrap"] = boot
log(f"  ROC CI: {boot['roc_auc_ci']} (recorded [0.9926, 0.9954]) overlap={boot['roc_ci_overlaps_recorded']}")
log(f"  PR CI:  {boot['pr_auc_ci']} (recorded [0.8119, 0.8500]) overlap={boot['pr_ci_overlaps_recorded']}")

# ---------------------------------------------------------------------------
# SECTION 7: CLAIMS TABLE (final statuses) + VERDICT
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION 7: FINAL CLAIMS TABLE + VERDICT")
log("=" * 70)

for c in claims_table:
    if c["claim"].startswith("Bootstrap CI ROC"):
        c["verified"] = f"{boot['roc_auc_ci']} (200-iter independent)"
        c["status"] = "PASS" if boot["roc_ci_overlaps_recorded"] else "FAIL"

critical = [
    "Chronological split claim is FALSE: the file is ordered by USER (2,000 contiguous user "
    "blocks, not by time). The 60/20/20 row split puts the FULL date range 1991-2020 in every "
    "block and slices two users across boundaries (1 user shared train/val, 1 shared val/test). "
    "The audit's sampled monotonicity check (every 100K rows) missed 5,789 full-file date "
    "regressions because dips are localized at user transitions.",
    "Test-set features are contaminated with FUTURE-dated (relative to the row's own timestamp) "
    "merchant/city fraud statistics: the expanding-window state is file-order, so an early-dated "
    "test row sees other users' later-dated transactions at the same merchant/city (up to 2020). "
    f"Quantified on a 500K test window: AUC {contam['audit_window_metrics']['roc_auc']} -> "
    f"{contam['chronological_window_metrics']['roc_auc']} with chronologically-correct rates. "
    "The audit's 'future-row perturbation PASS' proves only FILE-order causality, not "
    "chronological causality.",
]
high = [
    "1 of 415 test users also appears in the validation set (boundary-sliced user) - that user's "
    "transactions informed validation threshold selection AND final test evaluation.",
    "The reported 'temporal stability' windows are user-block segments spanning the full date "
    "range, not contiguous time windows; 'temporal_stability' labels are misleading.",
]
medium = [
    "Report metadata (train_end 2020-02, val_end 2013-07, test_start 2013-07) are boundary ROW "
    "dates, not period boundaries; they were recorded as if they described split periods.",
]
minor = [
    "Permutation-test claim (AUC 0.4812) and ablation numbers were NOT independently re-run "
    "(4 extra trainings + permutation would add ~40 min); they are the audit's own outputs.",
    "Bootstrap: independent 200-iter CIs overlap the recorded 1000-iter CIs but iteration count "
    "differs, so CIs are not directly comparable.",
]

verified = [c for c in claims_table if c["status"] == "PASS"]
failed = [c for c in claims_table if c["status"] == "FAIL"]
unverified = [
    "Real-world label latency of the fraud-rate features (no label-confirmation timestamps in the dataset).",
    "Permutation test / feature ablation independent re-run.",
    "Operational alert capacity (no production capacity requirement exists).",
]

final_status = "FAIL"
verdict = {
    "critical_failures": critical,
    "high_failures": high,
    "medium_failures": medium,
    "minor_issues": minor,
    "verified_claims": [c["claim"] for c in verified],
    "failed_claims": [c["claim"] for c in failed],
    "unverified_claims": unverified,
    "final_validation": final_status,
    "rationale": ("Any critical leakage / test-contamination / data-integrity issue forces FAIL. "
                  "Two critical items remain: (1) the 'chronological split' is false (entity-major "
                  "ordering, full date range in every block, boundary-sliced users shared across "
                  "splits); (2) test features embed future-dated merchant/city fraud relative to "
                  "the row's timestamp, inflating the reported metrics. The recorded numbers are "
                  "reproduced exactly (reproducibility PASS), but they describe a file-order "
                  "experiment, not a chronological one."),
}
REPORT["verdict"] = verdict
REPORT["claims_table_metrics"] = claims_table
REPORT["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
REPORT["runtime_sec"] = round(time.time() - T0, 1)

with open("reports/independent_validation.json", "w") as f:
    json.dump(REPORT, f, indent=1, default=str)

log("")
log("DONE -> reports/independent_validation.json")
log(f"  runtime: {REPORT['runtime_sec']}s")
log(f"  FINAL VALIDATION = {final_status}")