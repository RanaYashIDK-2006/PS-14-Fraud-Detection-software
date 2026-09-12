#!/usr/bin/env python3
"""ULB Velocity Features + Optuna XGBoost tuning.

Goal: push ROC-AUC > 99%, R@1%FPR > 93%.

ULB dataset is 284K transactions, time-sorted, PCA V1-V28 + Amount + Time.
Only 492 fraud cases (0.17%) — overfitting is the main risk.

Velocity features use the time-ordered sequence:
- Inter-transaction time gaps and rolling stats
- Amount rolling statistics (mean, std, z-score, acceleration)
- PCA component rolling dynamics (V14, V17, V12 are dominant)
- Burst detection patterns
- Time-of-day cyclical encoding
"""
from __future__ import annotations
import json, time, warnings, sys, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

NJ = 4  # Windows thread limit

# ── Feature Engineering ──────────────────────────────────────────────────────

def build_velocity_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build velocity/time-series features from the time-sorted ULB data."""
    df = df.copy()

    # --- Time gap features ---
    df["time_gap"] = df["Time"].diff().fillna(0).clip(lower=0)
    df["time_gap_log"] = np.log1p(df["time_gap"])
    df["time_gap_ratio"] = df["time_gap"] / df["time_gap"].rolling(20, min_periods=1).mean().clip(lower=1)

    # --- Rolling time window counts (1h=3600s) ---
    for window_s in [3600, 7200, 14400]:  # 1h, 2h, 4h
        wlabel = window_s // 3600
        # Number of transactions in the last W seconds
        window_count = (
            df["time_gap"]
            .rolling(window_s, min_periods=1)
            .apply(lambda x: np.sum(x <= window_s), raw=True)
            .fillna(0)
        )
        df[f"tx_count_{wlabel}h"] = window_count

    # --- Amount rolling features ---
    for w in [50, 200, 1000]:
        roll_amt = df["Amount"].rolling(w, min_periods=1)
        df[f"amt_roll_mean_{w}"] = roll_amt.mean()
        df[f"amt_roll_std_{w}"] = roll_amt.std().fillna(0)
        df[f"amt_roll_max_{w}"] = roll_amt.max()

    # Amount z-scores relative to rolling windows
    for w in [200, 1000]:
        roll_mean = df["Amount"].rolling(w, min_periods=1).mean()
        roll_std = df["Amount"].rolling(w, min_periods=1).std().fillna(1)
        df[f"amt_zscore_{w}"] = (df["Amount"] - roll_mean) / roll_std.clip(lower=0.01)

    # Amount acceleration: difference between short and long rolling means
    amt_s = df["Amount"].rolling(50, min_periods=1).mean()
    amt_l = df["Amount"].rolling(500, min_periods=1).mean()
    df["amt_acceleration"] = amt_s - amt_l
    df["amt_acceleration_pct"] = (amt_s - amt_l) / amt_l.clip(lower=0.01)

    # --- PCA component rolling dynamics ---
    # Focus on the most important fraud signals: V14, V17, V12, V10, V4, V11
    key_pcs = ["V14", "V17", "V12", "V10", "V4", "V11"]
    for v in key_pcs:
        if v in df.columns:
            roll_v = df[v].rolling(200, min_periods=1)
            df[f"{v}_roll_mean"] = roll_v.mean()
            df[f"{v}_roll_std"] = roll_v.std().fillna(0)
            df[f"{v}_deviation"] = df[v] - df[f"{v}_roll_mean"]

    # --- PCA anomaly scores ---
    # Sum of absolute PCA deviations from rolling mean
    dev_cols = [c for c in df.columns if c.endswith("_deviation")]
    df["pca_anomaly_sum"] = df[dev_cols].abs().sum(axis=1)
    df["pca_anomaly_max"] = df[dev_cols].abs().max(axis=1)

    # --- Burst detection ---
    # How many txns happened very recently (< 60s, < 300s)
    df["burst_60s"] = (df["time_gap"] <= 60).astype(int)
    df["burst_300s"] = (df["time_gap"] <= 300).astype(int)
    df["burst_count_60s"] = df["burst_60s"].rolling(10, min_periods=1).sum()
    df["burst_count_300s"] = df["burst_300s"].rolling(50, min_periods=1).sum()

    # --- Time-of-day cyclical ---
    # Time is seconds since start; map to hour of day (48h span)
    df["hour_of_day"] = (df["Time"] % 86400) / 3600
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["minute_of_hour"] = ((df["Time"] % 3600) / 60)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int)

    # --- Transaction position features ---
    df["tx_position"] = np.arange(len(df))
    df["tx_position_pct"] = df["tx_position"] / len(df)
    df["tx_position_day"] = df["Time"] / 86400  # fraction of day

    # --- Amount x PCA interactions (top fraud signals) ---
    if "V14" in df.columns:
        df["amt_x_V14"] = df["Amount"] * df["V14"]
    if "V17" in df.columns:
        df["amt_x_V17"] = df["Amount"] * df["V17"]
    if "V12" in df.columns:
        df["amt_x_V12"] = df["Amount"] * df["V12"]

    # --- PCA rolling cross-correlations ---
    if "V14" in df.columns and "V17" in df.columns:
        v14_roll = df["V14"].rolling(200, min_periods=1)
        v17_roll = df["V17"].rolling(200, min_periods=1)
        df["V14xV17_mean"] = v14_roll.mean() * v17_roll.mean()

    # Collect feature columns (exclude label)
    exclude = {"Class", "Time"}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Clean
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df, feature_cols


# ── Optuna + Evaluation ──────────────────────────────────────────────────────

def run_optuna_ulb(df: pd.DataFrame, feature_cols: list[str],
                   n_trials: int = 40) -> dict:
    """Run Optuna on XGBoost for ULB velocity features."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = df[feature_cols].values.astype(np.float32)
    y = df["Class"].values.astype(int)

    print(f"  Features: {len(feature_cols)}, Rows: {len(X)}, Fraud: {y.sum()}")

    # Time-based split for Optuna (first 80% train, last 20% test)
    split_idx = int(len(X) * 0.8)
    Xtr, Xte = X[:split_idx], X[split_idx:]
    ytr, yte = y[:split_idx], y[split_idx:]
    n_fraud_train = ytr.sum()
    n_fraud_test = yte.sum()
    print(f"  Train: {len(Xtr)} ({n_fraud_train} fraud), Test: {len(Xte)} ({n_fraud_test} fraud)")

    import xgboost as xgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": min(n_fraud_train / max(len(ytr) - n_fraud_train, 1) * 50, 200),
            "random_state": 42,
            "n_jobs": NJ,
            "eval_metric": "auc",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(Xtr, ytr, verbose=False)
        preds = model.predict_proba(Xte)[:, 1]
        return roc_auc_score(yte, preds)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Optuna best AUC: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=4)}")

    # Now evaluate best model with 5-fold time-based CV
    print("\n  Evaluating best model with 5-fold stratified CV...")
    best_params = study.best_params.copy()
    best_params["scale_pos_weight"] = min(n_fraud_train / max(len(ytr) - n_fraud_train, 1) * 50, 200)
    best_params["random_state"] = 42
    best_params["n_jobs"] = NJ
    best_params["eval_metric"] = "auc"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        Xtr_f, Xte_f = X[tr_idx], X[te_idx]
        ytr_f, yte_f = y[tr_idx], y[te_idx]

        model = xgb.XGBClassifier(**best_params)
        model.fit(Xtr_f, ytr_f, verbose=False)
        preds = model.predict_proba(Xte_f)[:, 1]

        fold_auc = roc_auc_score(yte_f, preds)

        # Recall at 1% FPR
        fpr_arr, tpr_arr, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(yte_f, preds)
        r1 = tpr_arr[np.searchsorted(fpr_arr, 0.01, side="right") - 1] if np.any(fpr_arr <= 0.01) else 0.0
        r05 = tpr_arr[np.searchsorted(fpr_arr, 0.005, side="right") - 1] if np.any(fpr_arr <= 0.005) else 0.0

        precision_arr, recall_arr, _ = precision_recall_curve(yte_f, preds)
        pr_auc = auc(recall_arr, precision_arr)

        fold_metrics.append({
            "fold": fold_idx,
            "auc": round(fold_auc, 6),
            "recall_1pct": round(r1, 6),
            "recall_05pct": round(r05, 6),
            "pr_auc": round(pr_auc, 6),
            "n_test": len(te_idx),
            "n_fraud_test": int(yte_f.sum()),
        })
        print(f"    Fold {fold_idx}: AUC={fold_auc:.4f} R@1%FPR={r1:.4f} R@0.5%FPR={r05:.4f} PR-AUC={pr_auc:.4f}")

    # Aggregate
    aucs = [f["auc"] for f in fold_metrics]
    r1s = [f["recall_1pct"] for f in fold_metrics]
    r05s = [f["recall_05pct"] for f in fold_metrics]

    results = {
        "dataset": "ULB",
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "n_fraud": int(y.sum()),
        "fraud_rate": round(float(y.mean()), 6),
        "optuna_trials": n_trials,
        "optuna_best_auc": round(study.best_value, 6),
        "optuna_best_params": study.best_params,
        "cv_mean_auc": round(float(np.mean(aucs)), 6),
        "cv_std_auc": round(float(np.std(aucs)), 6),
        "cv_mean_r1pct": round(float(np.mean(r1s)), 6),
        "cv_std_r1pct": round(float(np.std(r1s)), 6),
        "cv_mean_r05pct": round(float(np.mean(r05s)), 6),
        "cv_std_r05pct": round(float(np.std(r05s)), 6),
        "cv_folds": fold_metrics,
        "feature_cols": feature_cols,
        "velocity_features": [c for c in feature_cols if c not in
            [f"V{i}" for i in range(1, 29)] + ["Amount", "Class", "Time"]],
    }

    return results


def main():
    print("=" * 72)
    print("  ULB Velocity Features + Optuna XGBoost Tuning")
    print("=" * 72)

    # Load data
    print("\n[1] Loading ULB dataset...")
    t0 = time.time()
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Build velocity features
    print("\n[2] Building velocity features...")
    t0 = time.time()
    df, feature_cols = build_velocity_features(df)
    velocity_features = [c for c in feature_cols if c not in
        [f"V{i}" for i in range(1, 29)] + ["Amount", "Class", "Time"]]
    print(f"  Built {len(velocity_features)} velocity features in {time.time()-t0:.1f}s")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Velocity features: {velocity_features}")

    # Run Optuna
    print("\n[3] Running Optuna XGBoost tuning (40 trials)...")
    t0 = time.time()
    results = run_optuna_ulb(df, feature_cols, n_trials=40)
    results["elapsed_s"] = round(time.time() - t0, 1)

    # Save
    out = REPORTS / "ulb_velocity_optuna.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved: {out}")

    # Summary
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Features: {results['n_features']} ({results['n_features'] - len(results['velocity_features'])} original + {len(results['velocity_features'])} velocity)")
    print(f"  Optuna best AUC: {results['optuna_best_auc']:.4f}")
    print(f"  5-fold CV AUC:   {results['cv_mean_auc']:.4f} ± {results['cv_std_auc']:.4f}")
    print(f"  5-fold CV R@1%FPR: {results['cv_mean_r1pct']:.4f} ± {results['cv_std_r1pct']:.4f}")
    print(f"  5-fold CV R@0.5%FPR: {results['cv_mean_r05pct']:.4f} ± {results['cv_std_r05pct']:.4f}")

    target_auc = 0.99
    target_r1 = 0.93
    auc_status = '✅ ACHIEVED' if results['cv_mean_auc'] >= target_auc else f'❌ NOT MET ({results["cv_mean_auc"]:.4f})'
    r1_status = '✅ ACHIEVED' if results['cv_mean_r1pct'] >= target_r1 else f'❌ NOT MET ({results["cv_mean_r1pct"]:.4f})'
    print(f"\n  Target AUC > {target_auc:.0%}: {auc_status}")
    print(f"  Target R@1%FPR > {target_r1:.0%}: {r1_status}")

    # Feature importance from last fold
    import xgboost as xgb
    X = df[feature_cols].values.astype(np.float32)
    y = df["Class"].values
    model = xgb.XGBClassifier(**{**results["optuna_best_params"],
        "scale_pos_weight": min(492 / max(284315, 1) * 50, 200),
        "random_state": 42, "n_jobs": NJ, "eval_metric": "auc"})
    model.fit(X, y, verbose=False)
    imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    print("\n  Top 20 features by importance:")
    for name, score in imp[:20]:
        marker = " [VELOCITY]" if name in velocity_features else ""
        print(f"    {score:.4f}  {name}{marker}")

    print("\n" + "=" * 72)
    print("  COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
