#!/usr/bin/env python3
"""ULB focused push: minimal features + heavy regularization.

Key insight from previous run: PCA deviations (V12_dev, V10_dev, V17_dev, V14_dev)
are the strongest velocity features (5 of top 20). But 84 features with 492 fraud
cases causes overfitting. Strategy:

1. Start with original 31 features (PCA + Time + Amount)
2. Add ONLY the top 5-8 velocity features (PCA deviations + anomaly sum)
3. Use Optuna with extremely strong regularization
4. Evaluate with 5-fold stratified CV
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
NJ = 4


def build_focused_velocity(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build only the highest-signal velocity features."""
    df = df.copy()

    velocity_features = []

    # --- PCA rolling deviations (the top signals from previous run) ---
    key_pcs = ["V14", "V17", "V12", "V10", "V4", "V11"]
    for v in key_pcs:
        if v in df.columns:
            roll_mean = df[v].rolling(200, min_periods=1).mean()
            df[f"{v}_dev"] = df[v] - roll_mean
            velocity_features.append(f"{v}_dev")

    # --- PCA anomaly sum ---
    dev_cols = [f"{v}_dev" for v in key_pcs if f"{v}_dev" in df.columns]
    df["pca_anomaly_sum"] = df[dev_cols].abs().sum(axis=1)
    velocity_features.append("pca_anomaly_sum")

    # --- Amount z-score (short vs long window) ---
    amt_s = df["Amount"].rolling(50, min_periods=1).mean()
    amt_l = df["Amount"].rolling(500, min_periods=1).mean()
    df["amt_zscore_short"] = (df["Amount"] - amt_s) / df["Amount"].rolling(50, min_periods=1).std().fillna(1).clip(lower=0.01)
    velocity_features.append("amt_zscore_short")

    # --- Time gap ---
    df["time_gap"] = df["Time"].diff().fillna(0).clip(lower=0)
    velocity_features.append("time_gap")

    # --- Amount x V14 interaction (fraud amplification) ---
    df["amt_x_V14"] = df["Amount"] * df["V14"]
    velocity_features.append("amt_x_V14")

    # Clean
    original_features = [c for c in df.columns if c not in set(["Class", "Time"] + velocity_features)]
    all_features = original_features + velocity_features
    df[all_features] = df[all_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df, all_features, velocity_features


def main():
    print("=" * 72)
    print("  ULB Focused Push: Minimal Velocity + Heavy Regularization")
    print("=" * 72)

    # Load
    print("\n[1] Loading data...")
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    print(f"  {len(df):,} rows, {df.Class.sum()} fraud")

    # Build focused velocity
    print("\n[2] Building focused velocity features...")
    df, feature_cols, velocity_features = build_focused_velocity(df)
    print(f"  Total features: {len(feature_cols)} (31 original + {len(velocity_features)} velocity)")
    print(f"  Velocity: {velocity_features}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["Class"].values.astype(int)

    # ── Step 1: Optuna on time-based split ──
    print("\n[3] Optuna tuning (time-based split, 60 trials)...")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import xgboost as xgb

    split_idx = int(len(X) * 0.8)
    Xtr, Xte = X[:split_idx], X[split_idx:]
    ytr, yte = y[:split_idx], y[split_idx:]
    n_pos = int(ytr.sum())
    n_neg = int(len(ytr) - n_pos)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0.5, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 10, 500, log=True),
            "random_state": 42, "n_jobs": NJ, "eval_metric": "auc",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(Xtr, ytr, verbose=False)
        preds = model.predict_proba(Xte)[:, 1]
        return roc_auc_score(yte, preds)

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    print(f"  Optuna best: {study.best_value:.4f} ({time.time()-t0:.0f}s)")

    # ── Step 2: 5-fold CV with best params ──
    print("\n[4] 5-fold stratified CV with best params...")
    best = study.best_params.copy()
    best["random_state"] = 42
    best["n_jobs"] = NJ
    best["eval_metric"] = "auc"

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fi, (tr_i, te_i) in enumerate(skf.split(X, y)):
        Xtr_f, Xte_f = X[tr_i], X[te_i]
        ytr_f, yte_f = y[tr_i], y[te_i]

        model = xgb.XGBClassifier(**best)
        model.fit(Xtr_f, ytr_f, verbose=False)
        preds = model.predict_proba(Xte_f)[:, 1]

        fold_auc = roc_auc_score(yte_f, preds)

        fpr_arr, tpr_arr, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(yte_f, preds)
        # Recall at 1% FPR: find the first index where fpr > 0.01, use previous tpr
        idx_1pct = np.searchsorted(fpr_arr, 0.01, side="right")
        r1 = tpr_arr[idx_1pct - 1] if idx_1pct > 0 else 0.0
        idx_05pct = np.searchsorted(fpr_arr, 0.005, side="right")
        r05 = tpr_arr[idx_05pct - 1] if idx_05pct > 0 else 0.0

        prec, rec, _ = precision_recall_curve(yte_f, preds)
        pr_auc = auc(rec, prec)

        fold_results.append({
            "fold": fi, "auc": round(fold_auc, 6), "r1": round(r1, 6),
            "r05": round(r05, 6), "pr_auc": round(pr_auc, 6),
            "n_fraud": int(yte_f.sum()),
        })
        print(f"    Fold {fi}: AUC={fold_auc:.4f} R@1%FPR={r1:.4f} R@0.5%FPR={r05:.4f} (fraud={int(yte_f.sum())})")

    aucs = [f["auc"] for f in fold_results]
    r1s = [f["r1"] for f in fold_results]
    r05s = [f["r05"] for f in fold_results]

    print(f"\n  CV Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  CV Mean R@1%FPR: {np.mean(r1s):.4f} ± {np.std(r1s):.4f}")
    print(f"  CV Mean R@0.5%FPR: {np.mean(r05s):.4f} ± {np.std(r05s):.4f}")

    # ── Step 3: Full-data model for feature importance ──
    print("\n[5] Full-data model for feature importance...")
    full_model = xgb.XGBClassifier(**best)
    full_model.fit(X, y, verbose=False)
    imp = sorted(zip(feature_cols, full_model.feature_importances_), key=lambda x: -x[1])

    print("  Top 20 features:")
    for name, score in imp[:20]:
        marker = " [VEL]" if name in velocity_features else ""
        print(f"    {score:.4f}  {name}{marker}")

    # ── Save ──
    results = {
        "dataset": "ULB",
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "n_fraud": int(y.sum()),
        "optuna_best_auc": round(study.best_value, 6),
        "optuna_best_params": study.best_params,
        "cv_mean_auc": round(float(np.mean(aucs)), 6),
        "cv_std_auc": round(float(np.std(aucs)), 6),
        "cv_mean_r1pct": round(float(np.mean(r1s)), 6),
        "cv_std_r1pct": round(float(np.std(r1s)), 6),
        "cv_mean_r05pct": round(float(np.mean(r05s)), 6),
        "cv_std_r05pct": round(float(np.std(r05s)), 6),
        "cv_folds": fold_results,
        "velocity_features": velocity_features,
        "feature_cols": feature_cols,
        "top_features": [{"name": n, "importance": round(s, 6)} for n, s in imp[:20]],
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = REPORTS / "ulb_focused_push.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out}")

    # ── Summary ──
    print("\n" + "=" * 72)
    auc_achieved = np.mean(aucs) >= 0.99
    r1_achieved = np.mean(r1s) >= 0.93
    print(f"  Target AUC > 99%:     {'✅ ACHIEVED' if auc_achieved else f'❌ NOT MET ({np.mean(aucs):.4f})'}")
    print(f"  Target R@1%FPR > 93%: {'✅ ACHIEVED' if r1_achieved else f'❌ NOT MET ({np.mean(r1s):.4f})'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
