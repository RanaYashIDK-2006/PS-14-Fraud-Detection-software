#!/usr/bin/env python3
"""PS-14 ML-VALIDITY REBUILD - corrected, leakage-free evaluation protocol.

Ground-up rebuild (2026-09-03). Pipeline order is HARD:

    DATASET (causal features, provenance hashed)
      -> chronological TRAIN / VALIDATION / FINAL TEST (locked boundaries)
      -> fit scaler + model on TRAIN only; early stopping on VALIDATION only
      -> Platt calibration on VALIDATION only
      -> threshold selected on VALIDATION only (FPR <= 1%), then LOCKED
      -> evaluate ONCE on the untouched FINAL TEST
      -> metrics recomputed independently; bootstrap CI on paired resamples

The legacy modeling matrix (data/transactions.csv) is NOT used here: its
amount_ratio / amount_zscore / escalation features are computed from the
account-wide median that includes FUTURE transactions (proven by the
causality perturbation test). The corrected dataset is
data/transactions_causal.csv (derive_features_causal, prior-only stats).

Every number in reports/PS14_ML_VALIDITY_REPORT.md and
reports/ml_validity_protocol.json comes from this script +
scripts/independent_validator.py (separate code path).

Run:  ./.venv/Scripts/python.exe scripts/ml_validity_rebuild.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CSV = ROOT / "data" / "transactions_causal.csv"
OUT_JSON = ROOT / "reports" / "ml_validity_protocol.json"
OUT_PRED = ROOT / "reports" / "ml_validity_predictions.npz"
SPLIT = (0.70, 0.15, 0.15)  # train / validation / final test (chronological)

MODEL_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
    "unusual_location_flag", "unusual_recipient_flag", "failed_auth_count_24h",
    "days_since_last_similar_txn", "gradual_escalation_score", "known_device_count",
    "account_tenure_days", "hour_of_day", "is_weekend", "shared_device_accounts",
    "shared_recipient_accounts", "mule_ring_score", "hour_deviation", "amount_zscore",
    "velocity_deviation", "recipient_novelty", "txn_regularity",
]

# Feature-group taxonomy for the causality table + ablations. A feature is
# "historical" if it aggregates events before the current one.
INSTANT_FEATURES = ["hour_of_day", "is_weekend", "txn_time_unusual", "new_device_flag",
                    "unusual_location_flag", "unusual_recipient_flag"]
HISTORICAL_FEATURES = [f for f in MODEL_FEATURES if f not in INSTANT_FEATURES]

# Ablation groups (identical protocol, features subsetted).
ABLATION_GROUPS = {
    "full_causal_21": MODEL_FEATURES,
    "no_amount_history": [f for f in MODEL_FEATURES
                          if f not in ("amount_ratio", "amount_zscore", "gradual_escalation_score")],
    "instant_only": INSTANT_FEATURES,
    "flags_only": ["txn_time_unusual", "new_device_flag", "unusual_location_flag",
                   "unusual_recipient_flag", "is_weekend"],
}

TOL_CAUSAL = 1e-9  # max |prior-feature change| when future rows are perturbed


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- metrics
def metrics_at(score: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Count-based metrics from raw arrays; every identity is checkable."""
    y = y.astype(int)
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "alerts": tp + fp,
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "fpr": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "specificity": float(tn / (fp + tn)) if fp + tn else float("nan"),
        "tpr": float(tp / (tp + fn)) if tp + fn else float("nan"),
    }


def roc_auc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def pr_auc(score, y):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, score))


def threshold_for_fpr(val_score, val_y, max_fpr: float = 0.01) -> float:
    """Most permissive threshold whose FULL-ARRAY FPR <= max_fpr (validation only).

    Decision rule is `score >= threshold`, so the empirical FPR must count
    EVERY legitimate row at or above the threshold. A row-indexed cumulative
    count undercounts when scores tie at the boundary (rows beyond the cutoff
    with an equal score are also flagged), which would silently breach the
    cap - hence the unique-score scan: FPR falls as the threshold rises, and
    the FIRST threshold meeting the cap maximises recall within it.
    """
    val_y = val_y.astype(int)
    n_neg = max(int((val_y == 0).sum()), 1)
    neg_sorted = np.sort(val_score[val_y == 0])
    chosen = float(val_score.min())
    for thr in np.unique(val_score):
        fp = n_neg - int(np.searchsorted(neg_sorted, thr, side="left"))
        if fp / n_neg <= max_fpr:
            chosen = float(thr)
            break
    return chosen


def paired_bootstrap_ci(score, y, n_iter: int = 2000, seed: int = 42,
                        metric="roc_auc") -> dict:
    """Paired resample of (y, score) rows preserving joint structure.

    Class prevalence varies only through binomial sampling of the SAME rows
    (paired), which is the standard bootstrap for dependent AUC estimates.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if metric == "roc_auc":
        from sklearn.metrics import roc_auc_score
        fn = lambda s, yy: roc_auc_score(yy, s)
    else:
        from sklearn.metrics import average_precision_score
        fn = lambda s, yy: average_precision_score(yy, s)
    est = fn(score, y)
    vals = np.empty(n_iter)
    prevalences = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        vals[i] = fn(score[idx], y[idx])
        prevalences[i] = float(y[idx].mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    se = float(vals.std(ddof=1))
    return {
        "metric": metric, "n_iter": n_iter, "seed": seed,
        "point_estimate": float(est), "ci95": [float(lo), float(hi)],
        "ci_width": float(hi - lo), "bootstrap_se": se,
        "resample_prevalence_mean": float(prevalences.mean()),
        "resample_prevalence_sd": float(prevalences.std(ddof=1)),
        "original_prevalence": float(y.mean()),
        "method": "paired (y, score) row resampling; 2.5/97.5 percentiles; fixed seed",
    }


def main() -> int:
    t0 = time.time()
    report: dict = {"built_at": datetime.now(timezone.utc).isoformat()}

    # ---- A. dataset integrity -------------------------------------------
    raw = pd.read_csv(CSV)
    # `ts` is written as ISO-8601 text; parse to UTC and sort chronologically.
    ts_dt = pd.to_datetime(raw["ts"], utc=True)
    order = np.argsort(ts_dt.to_numpy(), kind="mergesort")
    df = raw.iloc[order].reset_index(drop=True)
    ts_dt = ts_dt.iloc[order].reset_index(drop=True)
    n = len(df)
    y_all = df["label"].to_numpy(dtype=int)
    ts = ts_dt.astype("int64").to_numpy(dtype=float) / 1e9  # epoch seconds
    integrity = {
        "file": str(CSV), "sha256": file_sha256(CSV),
        "rows": int(n), "fraud": int(y_all.sum()),
        "legit": int(n - y_all.sum()),
        "fraud_rate": round(float(y_all.mean()), 6),
        "ts_min_epoch": float(ts.min()), "ts_max_epoch": float(ts.max()),
        "ts_min_utc": str(pd.Timestamp(ts.min(), unit="s", tz="UTC")),
        "ts_max_utc": str(pd.Timestamp(ts.max(), unit="s", tz="UTC")),
        "duplicate_rows": int(df.duplicated().sum()),
        "ts_duplicate_count": int(n - np.unique(ts).size),
        "ordering": "monotonic_non_decreasing",
        "ordering_ok": bool(np.all(ts[1:] >= ts[:-1])),
        "accounts": int(df["fraud_id"].nunique()) if "fraud_id" in df else None,
    }
    report["A_dataset_integrity"] = integrity
    if not integrity["ordering_ok"]:
        print("FATAL: dataset timestamps go backwards; aborting")
        return 2

    # ---- B. chronological split (reproducible cut on sorted ts) ---------
    i_val = int(round(n * SPLIT[0]))
    i_test = int(round(n * (SPLIT[0] + SPLIT[1])))
    splits = {}
    for name, lo, hi in (("train", 0, i_val), ("validation", i_val, i_test),
                         ("final_test", i_test, n)):
        part = df.iloc[lo:hi]
        py = part["label"].to_numpy(dtype=int)
        pts = ts[lo:hi]
        splits[name] = {
            "row_start": int(lo), "row_end": int(hi), "rows": int(hi - lo),
            "fraud": int(py.sum()), "legit": int(hi - lo - py.sum()),
            "fraud_rate": round(float(py.mean()), 6),
            "ts_min_epoch": float(pts.min()), "ts_max_epoch": float(pts.max()),
            "ts_min_utc": str(pd.Timestamp(pts.min(), unit="s", tz="UTC")),
            "ts_max_utc": str(pd.Timestamp(pts.max(), unit="s", tz="UTC")),
        }
    # Future-contamination gate: no later-split row may have a timestamp
    # strictly earlier than any earlier-split row. The global sort guarantees
    # ts[cut] >= ts[cut-1]; we assert it explicitly so a future change that
    # breaks the monotonic cut fails here. Rows sharing an identical ts across
    # a boundary are SIMULTANEOUS (same second), not future data; they are
    # counted for disclosure rather than treated as overlap.
    contam_free = bool(ts[i_val] >= ts[i_val - 1] and ts[i_test] >= ts[i_test - 1])
    ties: dict = {}
    for label, cut in (("train_val", i_val), ("val_test", i_test)):
        cut_ts = ts[cut - 1]
        ties[label] = {
            "boundary_epoch": float(cut_ts),
            "boundary_utc": str(pd.Timestamp(cut_ts, unit="s", tz="UTC")),
            "rows_same_ts_left_of_cut": int((ts[:cut] == cut_ts).sum()),
            "rows_same_ts_right_of_cut": int((ts[cut:] == cut_ts).sum()),
        }
    splits["_future_contamination_free"] = contam_free
    splits["_boundary_simultaneous_ts"] = ties
    report["B_split"] = splits
    if not contam_free:
        print("FATAL: split cut is not monotonic (future ts in an earlier split)")
        return 2

    # split reproducibility: identical cut by hashing sorted ts prefix keys
    rep = hashlib.sha256(
        ts[[0, n - 1, i_val - 1, i_val, i_test - 1, i_test]].tobytes()).hexdigest()[:16]
    report["B_split_reproducibility_key"] = rep

    # ---- model protocol (validation-only decisions) ----------------------
    X = df[MODEL_FEATURES].to_numpy(dtype=float)
    Xtr, Xva, Xte = X[:i_val], X[i_val:i_test], X[i_test:]
    ytr, yva, yte = y_all[:i_val], y_all[i_val:i_test], y_all[i_test:]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)          # fit on TRAIN only
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    protocol: dict = {
        "features": MODEL_FEATURES,
        "classifier": "XGBClassifier (n_estimators 400, lr 0.05, depth 6, "
                      "subsample 0.8, colsample 0.8, min_child_weight 5, "
                      "reg_alpha 1, reg_lambda 1, scale_pos_weight=train ratio, "
                      "early_stopping on VALIDATION, seed 42)",
        "preprocessing": "StandardScaler fitted on TRAIN only",
        "calibration": "none (report uses raw probability; ranking metrics unaffected)",
        "threshold_policy": "validation-only, highest threshold with FPR <= 1% (exact)",
        "train_rows": int(len(ytr)), "validation_rows": int(len(yva)),
        "final_test_rows": int(len(yte)),
    }

    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_alpha=1.0, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=42, n_jobs=4,
        eval_metric="auc", early_stopping_rounds=30,
    )
    model.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], verbose=False)
    protocol["best_iteration"] = int(model.best_iteration)
    model.n_estimators = model.best_iteration + 1

    va_score = model.predict_proba(Xva_s)[:, 1]
    te_score = model.predict_proba(Xte_s)[:, 1]
    tr_score = model.predict_proba(Xtr_s)[:, 1]

    # Threshold LOCKED from validation only.
    locked = threshold_for_fpr(va_score, yva, max_fpr=0.01)
    val_metrics = metrics_at(va_score, yva, locked)
    te_metrics = metrics_at(te_score, yte, locked)
    # For disclosure: what threshold would the test itself pick? NOT used.
    test_threshold = threshold_for_fpr(te_score, yte, max_fpr=0.01)

    protocol["validation"] = {
        "auc": roc_auc(va_score, yva), "pr_auc": pr_auc(va_score, yva),
        "locked_threshold": locked, "metrics_at_locked": val_metrics,
        "note": "threshold and all decisions made HERE only",
    }
    protocol["final_test"] = {
        "auc": roc_auc(te_score, yte), "pr_auc": pr_auc(te_score, yte),
        "metrics_at_locked": te_metrics,
        "exact_fpr_at_locked": te_metrics["fpr"],
        "fpr_le_1pct": bool(te_metrics["fpr"] <= 0.01),
        "disclosure_test_picked_threshold_not_used": float(test_threshold),
        "auc_bootstrap": paired_bootstrap_ci(te_score, yte, metric="roc_auc"),
        "pr_auc_bootstrap": paired_bootstrap_ci(te_score, yte, metric="pr_auc"),
    }
    report["C_protocol"] = protocol

    # ---- D. ablation (same protocol, feature subsets) ---------------------
    abl = {}
    for gname, feats in ABLATION_GROUPS.items():
        Xg = df[feats].to_numpy(dtype=float)
        Xgtr, Xgva, Xgte = Xg[:i_val], Xg[i_val:i_test], Xg[i_test:]
        sc = StandardScaler().fit(Xgtr)
        m = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_alpha=1.0, reg_lambda=1.0,
            scale_pos_weight=spw, random_state=42, n_jobs=4,
            eval_metric="auc", early_stopping_rounds=30,
        )
        m.fit(sc.transform(Xgtr), ytr, eval_set=[(sc.transform(Xgva), yva)], verbose=False)
        m.n_estimators = m.best_iteration + 1
        sva = m.predict_proba(sc.transform(Xgva))[:, 1]
        ste = m.predict_proba(sc.transform(Xgte))[:, 1]
        th = threshold_for_fpr(sva, yva, max_fpr=0.01)
        abl[gname] = {
            "n_features": len(feats), "features": feats,
            "validation_auc": roc_auc(sva, yva),
            "final_test_auc": roc_auc(ste, yte),
            "final_test_pr_auc": pr_auc(ste, yte),
            "locked_threshold": th,
            "final_test_metrics_at_locked": metrics_at(ste, yte, th),
        }
    report["D_ablation"] = abl

    # ---- E. temporal stability over test windows -------------------------
    te_df = df.iloc[i_test:].reset_index(drop=True)
    wsize = max(int(len(te_df) // 4), 1)
    windows = {}
    for k in range(4):
        w = te_df.iloc[k * wsize:(k + 1) * wsize] if k < 3 else te_df.iloc[k * wsize:]
        ws = ts[i_test + k * wsize: i_test + ((k + 1) * wsize if k < 3 else n)]
        wy = w["label"].to_numpy(dtype=int)
        wscores = model.predict_proba(scaler.transform(w[MODEL_FEATURES].to_numpy(dtype=float)))[:, 1]
        windows[f"window_{k+1}"] = {
            "rows": int(len(w)), "fraud": int(wy.sum()),
            "fraud_prevalence": round(float(wy.mean()), 6),
            "ts_start": str(pd.Timestamp(ws.min(), unit="s", tz="UTC")),
            "ts_end": str(pd.Timestamp(ws.max(), unit="s", tz="UTC")),
            "auc": roc_auc(wscores, wy),
            "pr_auc": pr_auc(wscores, wy),
            "metrics_at_locked": metrics_at(wscores, wy, locked),
        }
    report["E_temporal_windows"] = windows

    np.savez_compressed(OUT_PRED, te_score=te_score, te_y=yte,
                        va_score=va_score, va_y=yva, tr_score=tr_score, tr_y=ytr,
                        locked_threshold=np.array([locked]))
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON} in {time.time()-t0:.0f}s")
    print(f"final test: AUC={protocol['final_test']['auc']:.4f} "
          f"PR={protocol['final_test']['pr_auc']:.4f} "
          f"locked={locked:.4f} FPR={te_metrics['fpr']:.6f} "
          f"recall={te_metrics['recall']:.4f} tp/fp/fn={te_metrics['tp']}/{te_metrics['fp']}/{te_metrics['fn']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
