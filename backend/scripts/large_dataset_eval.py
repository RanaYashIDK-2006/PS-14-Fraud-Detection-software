#!/usr/bin/env python3
"""Optimized evaluation for large datasets.

Uses subsampled training (200K max), batched prediction, and
lightweight models to handle datasets with 1M+ rows.

Also includes the previously completed 4-dataset results.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DATA = ROOT / "data"
RESULTS = DATA / "real_data_evaluation"
RESULTS.mkdir(exist_ok=True)

ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "hour_of_day", "is_weekend", "shared_device_accounts",
    "shared_recipient_accounts", "mule_ring_score", "hour_deviation",
    "amount_zscore", "velocity_deviation", "recipient_novelty", "txn_regularity",
]
N_F = len(ML_FEATURES)


# ═══════════════ FEATURE MAPPERS (same as real_data_evaluation.py) ═══════

def map_kartik2112(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    rng = np.random.RandomState(42)
    features = np.zeros((n, N_F))
    labels = df["is_fraud"].values.astype(int)

    unix = df["unix_time"].values.astype(float)
    hour_raw = (unix % 86400) / 3600
    features[:, ML_FEATURES.index("hour_of_day")] = hour_raw
    features[:, ML_FEATURES.index("is_weekend")] = ((unix % 604800) / 86400 > 5).astype(float)

    amt = df["amt"].values.astype(float)
    amt_mean = max(amt.mean(), 1)
    features[:, ML_FEATURES.index("amount_ratio")] = amt / amt_mean

    clat = df["lat"].values.astype(float)
    clon = df["long"].values.astype(float)
    mlat = df["merch_lat"].values.astype(float)
    mlon = df["merch_long"].values.astype(float)
    dist = np.sqrt((clat - mlat)**2 + (clon - mlon)**2)
    features[:, ML_FEATURES.index("new_device_flag")] = (dist > np.percentile(dist, 90)).astype(float)
    features[:, ML_FEATURES.index("unusual_location_flag")] = (dist > np.percentile(dist, 95)).astype(float)

    card_counts = df.groupby("cc_num").cumcount().values
    features[:, ML_FEATURES.index("txn_freq_last_24h")] = np.clip(card_counts, 0, 50).astype(float)
    features[:, ML_FEATURES.index("txn_time_unusual")] = hour_raw / 24.0

    pop = df["city_pop"].values.astype(float)
    features[:, ML_FEATURES.index("failed_auth_count_24h")] = (pop < np.percentile(pop, 10)).astype(float)

    sorted_idx = np.argsort(unix)
    td = np.zeros(n)
    td[sorted_idx[1:]] = np.diff(unix[sorted_idx])
    features[:, ML_FEATURES.index("days_since_last_similar_txn")] = np.clip(td / 86400, 0, 365)

    card_amt_max = df.groupby("cc_num")["amt"].cummax().values
    features[:, ML_FEATURES.index("gradual_escalation_score")] = amt / np.maximum(card_amt_max, 1)

    features[:, ML_FEATURES.index("known_device_count")] = np.clip(
        df.groupby("cc_num")["merchant"].transform("nunique").values, 0, 20
    )

    card_first = df.groupby("cc_num")["unix_time"].transform("min").values
    features[:, ML_FEATURES.index("account_tenure_days")] = (unix - card_first) / 86400

    card_seen = df.groupby("cc_num")["merchant"].transform(lambda x: pd.factorize(x)[0]).values
    features[:, ML_FEATURES.index("unusual_recipient_flag")] = (card_seen < 2).astype(float)

    features[:, ML_FEATURES.index("shared_device_accounts")] = rng.poisson(0.1, n).astype(float)
    features[:, ML_FEATURES.index("shared_recipient_accounts")] = rng.poisson(0.2, n).astype(float)
    features[:, ML_FEATURES.index("mule_ring_score")] = rng.poisson(0.05, n).astype(float)
    features[:, ML_FEATURES.index("hour_deviation")] = np.abs(hour_raw - 14) / 14
    features[:, ML_FEATURES.index("amount_zscore")] = (amt - amt_mean) / max(amt.std(), 1e-6)
    features[:, ML_FEATURES.index("velocity_deviation")] = np.clip(card_counts / max(np.median(card_counts), 1), 0, 5)
    features[:, ML_FEATURES.index("recipient_novelty")] = np.clip(card_seen / max(card_seen.max(), 1), 0, 1)
    features[:, ML_FEATURES.index("txn_regularity")] = rng.uniform(0, 1, n)
    return features, labels


def map_elliptic() -> tuple[np.ndarray, np.ndarray]:
    ep = DATA / "elliptic_bitcoin_dataset"
    df_feat = pd.read_csv(ep / "elliptic_txs_features.csv", header=None)
    df_cls = pd.read_csv(ep / "elliptic_txs_classes.csv")
    n_cols = df_feat.shape[1]
    df_feat.columns = ["txId", "time_step"] + [f"f{i}" for i in range(n_cols - 2)]
    df = df_feat.merge(df_cls, left_on="txId", right_on="txId", how="left")
    df["label_num"] = df["class"].map({1: 1, "1": 1, "illicit": 1, 2: 0, "2": 0, "licit": 0}).fillna(0)

    n = len(df)
    rng = np.random.RandomState(42)
    features = np.zeros((n, N_F))
    labels = df["label_num"].values.astype(int)

    ts = df["time_step"].values.astype(float)
    features[:, ML_FEATURES.index("hour_of_day")] = ts % 24
    features[:, ML_FEATURES.index("is_weekend")] = (ts % 7 >= 5).astype(float)

    feat_cols = [c for c in df_feat.columns if c not in ("txId", "time_step")]
    fd = df[feat_cols].values.astype(float)
    agg = fd[:, -1]
    agg_mean = max(np.mean(np.abs(agg)), 1e-6)
    features[:, ML_FEATURES.index("amount_ratio")] = np.abs(agg) / agg_mean

    fam = np.mean(np.abs(fd), axis=1)
    fsd = np.std(fd, axis=1)
    fmx = np.max(np.abs(fd), axis=1)

    features[:, ML_FEATURES.index("new_device_flag")] = (fmx > np.percentile(fmx, 85)).astype(float)
    features[:, ML_FEATURES.index("unusual_location_flag")] = (fam > np.percentile(fam, 90)).astype(float)
    features[:, ML_FEATURES.index("unusual_recipient_flag")] = (fsd > np.percentile(fsd, 85)).astype(float)
    features[:, ML_FEATURES.index("txn_freq_last_24h")] = np.clip(ts / 5, 0, 10)
    features[:, ML_FEATURES.index("txn_time_unusual")] = ts / 49.0
    features[:, ML_FEATURES.index("days_since_last_similar_txn")] = rng.exponential(1, n)
    features[:, ML_FEATURES.index("gradual_escalation_score")] = np.clip(fam / max(fam.max(), 1e-6), 0, 1)
    features[:, ML_FEATURES.index("known_device_count")] = 2.0
    features[:, ML_FEATURES.index("account_tenure_days")] = ts * 10
    features[:, ML_FEATURES.index("failed_auth_count_24h")] = (fmx > np.percentile(fmx, 95)).astype(float)
    features[:, ML_FEATURES.index("shared_device_accounts")] = rng.poisson(0.3, n).astype(float)
    features[:, ML_FEATURES.index("shared_recipient_accounts")] = rng.poisson(0.4, n).astype(float)
    features[:, ML_FEATURES.index("mule_ring_score")] = (fam > np.percentile(fam, 95)).astype(float) * 2
    features[:, ML_FEATURES.index("hour_deviation")] = np.abs(features[:, ML_FEATURES.index("hour_of_day")] - 14) / 14
    amt_r = features[:, ML_FEATURES.index("amount_ratio")]
    features[:, ML_FEATURES.index("amount_zscore")] = (amt_r - 1) / max(amt_r.std(), 1e-6)
    features[:, ML_FEATURES.index("velocity_deviation")] = np.clip(features[:, ML_FEATURES.index("txn_freq_last_24h")] / 2, 0, 5)
    features[:, ML_FEATURES.index("recipient_novelty")] = np.clip(fam / max(fam.max(), 1e-6), 0, 1)
    features[:, ML_FEATURES.index("txn_regularity")] = np.clip(fsd / max(fsd.mean(), 1e-6), 0, 1)
    return features, labels


# ═══════════════ FAST METRICS (no bootstrap for speed) ═══════

def fast_metrics(y_true, y_prob, name=""):
    n = len(y_true)
    n_fraud = int(y_true.sum())
    n_legit = n - n_fraud
    if n_fraud == 0 or n_legit == 0:
        return {"name": name, "status": "INVALID", "n": n, "n_fraud": n_fraud}

    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)

    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    def r_at(target):
        idx = np.searchsorted(fpr_arr, target, side="left")
        return float(tpr_arr[min(idx, len(tpr_arr) - 1)])

    # Quick bootstrap (50 samples for speed)
    boot_roc, boot_pr = [], []
    rng = np.random.default_rng(42)
    for _ in range(50):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_roc.append(roc_auc_score(y_true[idx], y_prob[idx]))
        boot_pr.append(average_precision_score(y_true[idx], y_prob[idx]))

    ci_roc = (round(float(np.percentile(boot_roc, 2.5)), 4), round(float(np.percentile(boot_roc, 97.5)), 4)) if boot_roc else (0, 0)
    ci_pr = (round(float(np.percentile(boot_pr, 2.5)), 4), round(float(np.percentile(boot_pr, 97.5)), 4)) if boot_pr else (0, 0)

    brier = brier_score_loss(y_true, y_prob)
    return {
        "name": name, "n_samples": n, "n_fraud": n_fraud, "n_legit": n_legit,
        "prevalence": round(n_fraud / n, 4),
        "roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
        "recall_at_01pct_fpr": round(r_at(0.001), 4),
        "recall_at_05pct_fpr": round(r_at(0.005), 4),
        "recall_at_1pct_fpr": round(r_at(0.01), 4),
        "brier": round(brier, 4),
        "ci_roc_auc_95": ci_roc, "ci_pr_auc_95": ci_pr,
    }


# ═══════════════ OPTIMIZED TRAIN+EVAL ═══════

def optimized_train_eval(name: str, features: np.ndarray, labels: np.ndarray,
                         max_train: int = 200_000, max_test: int = 100_000):
    """Train on subsampled data, evaluate on held-out test."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Total: {len(labels):,} rows, {int(labels.sum()):,} fraud ({labels.mean():.4%})")
    print(f"{'='*60}")

    n = len(labels)
    n_fraud = int(labels.sum())
    if n_fraud < 10:
        return {"name": name, "status": "SKIP"}

    # Temporal split
    split_train = min(int(n * 0.70), max_train)
    split_val = int(n * 0.85)
    split_test = min(n, split_val + max_test)

    X_train = features[:split_train]
    y_train = labels[:split_train]
    X_val = features[split_train:split_val]
    y_val = labels[split_train:split_val]
    X_test = features[split_val:split_test]
    y_test = labels[split_val:split_test]

    print(f"  Train: {len(y_train):,} ({int(y_train.sum())} fraud)")
    print(f"  Val:   {len(y_val):,} ({int(y_val.sum())} fraud)")
    print(f"  Test:  {len(y_test):,} ({int(y_test.sum())} fraud)")

    if len(np.unique(y_test)) < 2:
        return {"name": name, "status": "SKIP - single class in test"}

    t0 = time.time()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    # Logistic Regression (fast)
    lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    lr.fit(X_tr, y_train)
    lr_te = lr.predict_proba(X_te)[:, 1]

    # Random Forest (limited trees for speed)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_train)
    rf_te = rf.predict_proba(X_te)[:, 1]

    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                            scale_pos_weight=max(n_fraud / max(n - n_fraud, 1), 1),
                            random_state=42, eval_metric="aucpr", verbosity=0,
                            tree_method="hist")  # hist = fast on large data
        xgb.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)
        xgb_te = xgb.predict_proba(X_te)[:, 1]
    except ImportError:
        xgb_te = (lr_te + rf_te) / 2

    # Stacker
    lr_v = lr.predict_proba(X_v)[:, 1]
    rf_v = rf.predict_proba(X_v)[:, 1]
    try:
        xgb_v = xgb.predict_proba(X_v)[:, 1]
    except Exception:
        xgb_v = (lr_v + rf_v) / 2

    stk_X = np.column_stack([lr_v, rf_v, xgb_v])
    stk_X_te = np.column_stack([lr_te, rf_te, xgb_te])
    stk = LogisticRegression(max_iter=500, random_state=42)
    stk.fit(stk_X, y_val)
    fused_te = stk.predict_proba(stk_X_te)[:, 1]

    train_s = time.time() - t0
    print(f"  Training: {train_s:.1f}s")

    results = {"training_time_s": round(train_s, 1)}
    for mname, scores in [("lr", lr_te), ("rf", rf_te), ("xgb", xgb_te), ("stacker", fused_te)]:
        results[mname] = fast_metrics(y_test, scores, mname)
        m = results[mname]
        print(f"  {mname:8s}  ROC={m.get('roc_auc','?')}  PR={m.get('pr_auc','?')}  R@1%={m.get('recall_at_1pct_fpr','?')}")

    # Feature importance
    if hasattr(xgb, "feature_importances_"):
        imp = xgb.feature_importances_
        top = np.argsort(imp)[::-1][:5]
        print(f"  Top 5: {', '.join(ML_FEATURES[i] for i in top)}")

    return results


# ═══════════════ MAIN ═══════

def main():
    all_results = {}
    t_start = time.time()

    # --- kartik2112 (1.85M rows) ---
    kpath = DATA / "kaggle_fraud"
    if (kpath / "fraudTrain.csv").exists():
        print("Loading kartik2112...")
        df = pd.concat([
            pd.read_csv(kpath / "fraudTrain.csv"),
            pd.read_csv(kpath / "fraudTest.csv"),
        ], ignore_index=True)
        X, y = map_kartik2112(df)
        all_results["kartik2112"] = optimized_train_eval("kartik2112 (1.85M)", X, y)

    # --- Elliptic Bitcoin (203K rows) ---
    epath = DATA / "elliptic_bitcoin_dataset"
    if (epath / "elliptic_txs_features.csv").exists():
        print("\nLoading Elliptic Bitcoin...")
        X, y = map_elliptic()
        all_results["elliptic_btc"] = optimized_train_eval("Elliptic Bitcoin (203K)", X, y)

    # --- Load previous 4-dataset results ---
    prev_path = RESULTS / "real_data_benchmark.json"
    if prev_path.exists():
        with open(prev_path) as f:
            prev = json.load(f)
        for k in ["ulb_creditcard", "paysim_1m", "paysim_100k", "synthetic_500k"]:
            if k in prev and "calibrated" in prev[k]:
                all_results[k] = prev[k]

    total = time.time() - t_start

    # --- Save ---
    out = RESULTS / "full_benchmark.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # --- Print summary table ---
    print(f"\n{'='*130}")
    print("PS-14 COMPLETE REAL-DATA BENCHMARK")
    print(f"{'='*130}")
    print(f"Total time: {total:.0f}s\n")

    header = f"{'Dataset':25s} {'N':>10s} {'Fraud':>8s} {'Prev':>7s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'R@0.1%':>8s} {'R@0.5%':>8s} {'R@1%':>8s} {'Brier':>8s}"
    print(header)
    print("-" * len(header))

    order = ["ulb_creditcard", "paysim_1m", "paysim_100k", "kartik2112", "elliptic_btc", "synthetic_500k"]
    for key in order:
        if key not in all_results:
            continue
        r = all_results[key]
        if "status" in r:
            print(f"{key:25s}  {r['status']}")
            continue

        # Pick best model (stacker if available, else calibrated)
        best = r.get("stacker", r.get("calibrated", {}))
        if "roc_auc" not in best:
            best = r.get("xgb", r.get("xgboost", {}))
        if "roc_auc" not in best:
            print(f"{key:25s}  NO VALID RESULTS")
            continue

        ci = best.get("ci_roc_auc_95", ("?", "?"))
        print(
            f"{best.get('name', key):25s} "
            f"{best['n_samples']:>10,d} "
            f"{best['n_fraud']:>8,d} "
            f"{best.get('prevalence',0):>6.2%} "
            f"{best['roc_auc']:>9.4f} "
            f"{best['pr_auc']:>9.4f} "
            f"{best.get('recall_at_01pct_fpr','?'):>8} "
            f"{best.get('recall_at_05pct_fpr','?'):>8} "
            f"{best.get('recall_at_1pct_fpr','?'):>8} "
            f"{best.get('brier','?'):>8}"
        )
        print(f"{'':25s}  95% CI ROC [{ci[0]}, {ci[1]}]")

    print(f"\n{'='*130}")
    print("DONE")


if __name__ == "__main__":
    main()
