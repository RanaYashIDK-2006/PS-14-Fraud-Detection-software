# -*- coding: utf-8 -*-
"""
PS-14 Check #23 -- DATA PROVENANCE, DATASET INTEGRITY & EXPERIMENT REPRODUCIBILITY

Independent implementation: direct pandas/numpy reads of the raw dataset; no
metric/feature functions imported from the audit pipeline (except the VERBATIM
feature function in _forensic_common.py, which is byte-verified against
forensic_revalidate.py -- reuse is required by the reproducibility mandate).

Output: reports/data_provenance.json
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from _forensic_common import (
    CSV_ALTMAN, CHUNK, SEED, USECOLS, FEATURE_NAMES,
    expanding_features_fixed, new_state, new_pop_state, verify_verbatim_copy,
)

T0 = time.time()
REPORT = {}


def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# SECTION A: FILE HASHES (provenance record)
# ---------------------------------------------------------------------------
log("=" * 70)
log("SECTION A: FILE HASHES")
log("=" * 70)

files_to_hash = [
    ("raw_dataset_csv", CSV_ALTMAN),
    ("cached_altman_sorted_parquet", "data/_sorted_cache/altman_sorted.parquet"),
    ("cached_features_all_parquet", "data/_sorted_cache/features_all.parquet"),
    ("audit_experiment_metadata", "reports/forensic_revalidation.json"),
    ("audit_locked_threshold_record", "models/model_records/audit_25feat_xgb_inline_threshold.json"),
    ("deployed_model_record", "models/model_records/altman_lean_15feat_20260830_200346.json"),
]

hashes = {}
for name, path in files_to_hash:
    if os.path.exists(path):
        size = os.path.getsize(path)
        log(f"  hashing {name} ({size/1e6:.0f} MB)...")
        hashes[name] = {"path": path, "size": size, "sha256": sha256_file(path)}
    else:
        hashes[name] = {"path": path, "error": "MISSING"}
        log(f"  MISSING: {path}")

# Deployed artifact set (from the governance record)
gov = json.load(open("reports/model_governance.json"))
rec = json.load(open("models/model_records/altman_lean_15feat_20260830_200346.json"))
artifact_checks = []
for fname, meta in rec.get("artifact_files", {}).items():
    # locate the artifact: models/production/<fname> or models/artifacts/<fname>
    cand = [f"models/production/{fname}", f"models/artifacts/{fname}"]
    found = next((c for c in cand if os.path.exists(c)), None)
    if found:
        h = sha256_file(found)
        ok = h == meta["sha256"]
        artifact_checks.append({"file": fname, "path": found,
                                "recorded_sha256": meta["sha256"],
                                "actual_sha256": h, "match": ok})
    else:
        artifact_checks.append({"file": fname, "path": None,
                                "recorded_sha256": meta["sha256"],
                                "actual_sha256": None, "match": False})
n_art_match = sum(1 for a in artifact_checks if a["match"])
log(f"  deployed artifacts verified vs governance record: {n_art_match}/{len(artifact_checks)} match")
REPORT["file_hashes"] = hashes
REPORT["deployed_artifact_hash_verification"] = {
    "n_artifacts": len(artifact_checks),
    "n_match": n_art_match,
    "checks": artifact_checks,
}

# ---------------------------------------------------------------------------
# SECTION B: DATASET STATS (independent pandas read)
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION B: DATASET STATS")
log("=" * 70)

log("  loading full raw CSV...")
df = pd.read_csv(CSV_ALTMAN, low_memory=False)
total = len(df)
log(f"  rows: {total:,}")

schema = [{"column": c, "dtype": str(df[c].dtype)} for c in df.columns]

is_fraud = df["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
n_fraud = int(is_fraud.sum())
n_legit = total - n_fraud

# Timestamp range
dates = df["Year"].astype(int) * 10000 + df["Month"].astype(int) * 100 + df["Day"].astype(int)
date_min, date_max = int(dates.min()), int(dates.max())

# Missing values
missing = {c: int(df[c].isna().sum()) for c in df.columns}

# Full-row duplicates
dup_mask = df.duplicated(keep="first")
n_dups = int(dup_mask.sum())
dup_example = df[dup_mask].head(3).to_dict("records") if n_dups else []

# Label distribution (including NaN counts)
label_counts = df["Is Fraud?"].value_counts(dropna=False).to_dict()
n_label_nan = int(df["Is Fraud?"].isna().sum())

dataset_stats = {
    "total_rows": total,
    "n_columns": len(df.columns),
    "schema": schema,
    "fraud_rows": n_fraud,
    "legit_rows": n_legit,
    "fraud_rate_pct": round(n_fraud / total * 100, 4),
    "date_min": date_min,
    "date_max": date_max,
    "missing_values": missing,
    "full_row_duplicates": n_dups,
    "duplicate_examples": dup_example,
    "label_value_counts": {str(k): int(v) for k, v in label_counts.items()},
    "label_missing_count": n_label_nan,
}
REPORT["dataset_stats"] = dataset_stats
log(f"  fraud={n_fraud:,} legit={n_legit:,} rate={n_fraud/total*100:.4f}%")
log(f"  date range: {date_min} .. {date_max}; dup rows: {n_dups:,}; missing labels: {n_label_nan:,}")

# ---------------------------------------------------------------------------
# SECTION C: SPLIT VERIFICATION + ENTITY-ORDER ANALYSIS
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION C: SPLIT VERIFICATION & ENTITY-ORDER ANALYSIS")
log("=" * 70)

n_train = int(total * 0.60)
n_val = int(total * 0.20)
n_test = total - n_train - n_val

# Boundary dates
def date_at(i):
    r = df.iloc[i]
    return f"{int(r['Year'])}-{int(r['Month']):02d}-{int(r['Day']):02d}"

boundary = {
    "train_end_row": n_train - 1, "train_end_date": date_at(n_train - 1),
    "val_start_date": date_at(n_train),
    "val_end_row": n_train + n_val - 1, "val_end_date": date_at(n_train + n_val - 1),
    "test_start_date": date_at(n_train + n_val),
}

train_fraud = int(is_fraud.iloc[:n_train].sum())
val_fraud = int(is_fraud.iloc[n_train:n_train + n_val].sum())
test_fraud = int(is_fraud.iloc[n_train + n_val:].sum())

# Entity (user) structure
u = df["User"].astype(str).values
trans = u[:-1] != u[1:]
n_blocks = int(trans.sum()) + 1
block_starts = np.concatenate([[0], np.where(trans)[0] + 1, [total]])
block_users = [u[block_starts[i]] for i in range(len(block_starts) - 1)]
block_sizes = np.diff(block_starts)

# Which users straddle the split boundaries?
train_users = set(u[:n_train])
val_users = set(u[n_train:n_train + n_val])
test_users = set(u[n_train + n_val:])
straddle_train_val = sorted(train_users & val_users)
straddle_val_test = sorted(val_users & test_users)
overlap_train_test = sorted(train_users & test_users)

# Date monotonicity
d = (df["Year"].astype(int) * 10000 + df["Month"].astype(int) * 100 + df["Day"].astype(int)).values
viol_full = int((d[:-1] > d[1:]).sum())
# The audit's sampled check (every 100K rows)
sk = (df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-" + df["Day"].astype(str).str.zfill(2)).values
viol_sampled = int(sum(1 for i in range(0, len(sk), 100000) if i + 1 < len(sk) and sk[i] > sk[i + 1]))

# Per-block date ranges
block_ranges = {}
for nm, lo, hi in [("train", 0, n_train), ("val", n_train, n_train + n_val), ("test", n_train + n_val, total)]:
    seg = d[lo:hi]
    block_ranges[nm] = {"rows": int(hi - lo), "min_date": int(seg.min()), "max_date": int(seg.max()),
                        "n_users": len(set(u[lo:hi]))}

recorded = {
    "train_rows": 14632140, "val_rows": 4877380, "test_rows": 4877380,
    "train_fraud": 17736, "val_fraud": 6260, "test_fraud": 5761,
}
actual = {"train_rows": n_train, "val_rows": n_val, "test_rows": n_test,
          "train_fraud": train_fraud, "val_fraud": val_fraud, "test_fraud": test_fraud}
split_match = {k: (recorded[k] == actual[k]) for k in recorded}

split_analysis = {
    "method_recorded": "Chronological 60/20/20 split",
    "boundary_dates": boundary,
    "recorded": recorded,
    "actual": actual,
    "split_match": split_match,
    "entity_structure": {
        "n_user_blocks": int(n_blocks),
        "n_distinct_users": len(set(u)),
        "file_is_user_major": True,
        "split_slices_user_blocks": bool(straddle_train_val or straddle_val_test),
        "users_straddling_train_val": straddle_train_val[:5],
        "users_straddling_val_test": straddle_val_test[:5],
        "user_overlap_train_test": overlap_train_test[:5],
    },
    "chronology": {
        "full_file_date_regressions": viol_full,
        "audit_sampled_check_regressions": viol_sampled,
        "per_block_date_ranges": block_ranges,
        "claim_chronological_split": False,
        "why": ("File is ordered by USER (2000 contiguous user blocks), not by time. "
                "Each split block spans the FULL date range (1991-2020). Split boundaries "
                "slice through user blocks. The audit's sampled monotonicity check (every "
                "100K rows) found 0 regressions only because date dips are localized at "
                "user transitions; the FULL check finds 5,789 regressions."),
    },
}
REPORT["split_analysis"] = split_analysis
log(f"  recorded vs actual split: {split_match}")
log(f"  user blocks: {n_blocks:,}; file is user-major; boundaries slice users: {bool(straddle_train_val or straddle_val_test)}")
log(f"  full-file date regressions: {viol_full:,} (audit sampled check: {viol_sampled})")
for nm, rng in block_ranges.items():
    log(f"  {nm}: rows {rng['rows']:,}, dates {rng['min_date']}..{rng['max_date']}, users {rng['n_users']}")

# ---------------------------------------------------------------------------
# SECTION D: LABEL AUDIT
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION D: LABEL AUDIT")
log("=" * 70)

label_audit = {
    "label_column": "Is Fraud?",
    "label_1_meaning": "Is Fraud? == 'Yes' -- transaction flagged as fraudulent in the source dataset",
    "label_0_meaning": "Is Fraud? == 'No' -- legitimate transaction",
    "label_created_when": "At dataset generation time (pre-labeled source file); no creation pipeline exists in this repo",
    "labels_change": "No mechanism in the dataset to update labels after the fact (static file)",
    "labels_delayed": "UNVERIFIED -- the dataset carries no label-confirmation timestamp; the audit's own feature audit marks fraud-rate features as LABEL_LATENCY risk",
    "missing_labels": n_label_nan,
    "missing_label_handling": "Pipeline maps 'Yes'->1, 'No'->0, fillna(0): any missing label silently becomes legitimate (0)",
    "uncertain_labels": "UNVERIFIED -- no label-quality/confidence field exists",
    "label_used_in_features": "4 fraud-rate features (merch/city/user_fraud_rate + mfr_x_ufr) update state AFTER feature computation for the current row (past-only, verified independently in check #24); the current row's own label never feeds its own features",
    "label_leakage_into_features_verdict": "No current-row label leakage in the feature function (state read before update); CHRONOLOGICAL ordering issue documented in split_analysis",
}
REPORT["label_audit"] = label_audit
log("  label=1: 'Is Fraud?'=='Yes' (pre-labeled source); delay/mutability UNVERIFIED")

# ---------------------------------------------------------------------------
# SECTION E: REPRODUCIBILITY -- double-run of a fixed experiment
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("SECTION E: REPRODUCIBILITY (double-run of fixed seeded experiment)")
log("=" * 70)

# 0) Verify the verbatim copy first
ok_verbatim, detail_verbatim = verify_verbatim_copy()
REPORT["verbatim_copy_verified"] = {"ok": ok_verbatim, "detail": detail_verbatim}
log(f"  verbatim feature-function copy: {ok_verbatim} ({detail_verbatim})")

import xgboost as xgb

EXPERIMENT_ROWS = 2_000_000
VAL_SPLIT_AT = 1_600_000  # last 400K rows of the slice act as a mini-validation

def run_fixed_experiment():
    """Deterministic mini-experiment: features on rows [0, 2M) (seed-free),
    XGB train on [0, 1.6M), eval AUC on [1.6M, 2M). Returns metrics."""
    st = new_state()
    pst = new_pop_state()
    F = expanding_features_fixed(df.iloc[:EXPERIMENT_ROWS], st, pst)
    y = is_fraud.iloc[:EXPERIMENT_ROWS].values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xs = sc.fit_transform(F)
    del F
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.7, random_state=SEED,
                              n_jobs=4, eval_metric="auc", early_stopping_rounds=20)
    model.fit(Xs[:VAL_SPLIT_AT], y[:VAL_SPLIT_AT], eval_set=[(Xs[VAL_SPLIT_AT:], y[VAL_SPLIT_AT:])], verbose=False)
    p = model.predict_proba(Xs[VAL_SPLIT_AT:])[:, 1]
    from sklearn.metrics import roc_auc_score
    return {
        "val_auc": float(roc_auc_score(y[VAL_SPLIT_AT:], p)),
        "best_iteration": int(model.best_iteration),
        "n_est": int(model.n_estimators),
    }

run1 = run_fixed_experiment()
log(f"  run #1: {run1}")
run2 = run_fixed_experiment()
log(f"  run #2: {run2}")

repro = {
    "experiment": f"rows [0,{EXPERIMENT_ROWS:,}) features + XGB(seed={SEED}) train [0,{VAL_SPLIT_AT:,}) eval [{VAL_SPLIT_AT:,},{EXPERIMENT_ROWS:,})",
    "run_1": run1,
    "run_2": run2,
    "identical": run1 == run2,
    "note": "Full-pipeline reproducibility (run #1 = recorded forensic_revalidation.json) is verified in independent_validation.py (check #24).",
}
REPORT["reproducibility"] = repro
log(f"  runs identical: {run1 == run2}")

del df
import gc
gc.collect()

REPORT["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
REPORT["runtime_sec"] = round(time.time() - T0, 1)

with open("reports/data_provenance.json", "w") as f:
    json.dump(REPORT, f, indent=1, default=str)
log("")
log("DONE -> reports/data_provenance.json")
log(f"  runtime: {REPORT['runtime_sec']}s")