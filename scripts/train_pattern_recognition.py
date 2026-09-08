#!/usr/bin/env python3
"""
Train and evaluate fraud detection with pattern recognition features.

Uses the new transactions_v2.csv with 31 features.
Trains LR + RF + XGB ensemble with proper stratified split.
Reports honest metrics with confidence intervals.
"""

import numpy as np
import pandas as pd
import hashlib
import json
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")

# Features to use (drop broken and zero-variance features)
DROP_COLS = {
    "label",                          # target
    "new_device_flag",                # all zeros
    "unusual_location_flag",          # all zeros
    "failed_auth_count_24h",          # all zeros
    "known_device_count",             # all zeros
    "shared_device_accounts",         # all zeros
    "shared_recipient_accounts",      # all zeros
    "mule_ring_score",                # all zeros
    "is_weekend",                     # all zeros
    "amount_v_correlation",           # ~0 for all
    "days_since_last_similar_txn",    # very small values, noisy
}


def compute_hash(filepath: Path) -> str:
    """SHA256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    if not scores:
        return (float("nan"), float("nan"))
    alpha = (1 - ci) / 2
    return (np.percentile(scores, alpha * 100), np.percentile(scores, (1 - alpha) * 100))


def recall_at_fpr(y_true, y_score, fpr_target=0.01):
    """Recall at a fixed false positive rate."""
    fpr, tpr, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(y_true, y_score)
    # Find threshold where FPR <= target
    valid = fpr <= fpr_target
    if not valid.any():
        return 0.0
    return float(tpr[valid][-1])


def train_and_evaluate():
    print("=" * 70)
    print("PATTERN RECOGNITION ENHANCED MODEL TRAINING")
    print("=" * 70)

    # Load data
    df = pd.read_csv(DATA_DIR / "transactions_v2.csv")
    print(f"Dataset: {len(df)} rows, fraud rate: {df['label'].mean()*100:.3f}%")

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    # Replace any remaining inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

    # Hash the dataset
    data_hash = compute_hash(DATA_DIR / "transactions_v2.csv")
    print(f"Dataset hash: {data_hash}")

    # Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} (fraud: {y_train.sum()})")
    print(f"Test:  {len(X_test)} (fraud: {y_test.sum()})")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    all_models = {}

    # --- Logistic Regression ---
    print("\n--- Logistic Regression ---")
    t0 = time.time()
    lr = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000,
        solver="lbfgs", random_state=42
    )
    lr.fit(X_train_s, y_train)
    lr_time = time.time() - t0

    lr_proba = lr.predict_proba(X_test_s)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_pr = average_precision_score(y_test, lr_proba)
    lr_brier = brier_score_loss(y_test, lr_proba)
    lr_r1 = recall_at_fpr(y_test, lr_proba, 0.01)
    lr_cm = confusion_matrix(y_test, lr_pred)
    print(f"  ROC-AUC: {lr_auc:.4f}")
    print(f"  PR-AUC:  {lr_pr:.4f}")
    print(f"  Recall@1%FPR: {lr_r1:.4f}")
    print(f"  Brier:   {lr_brier:.6f}")
    print(f"  Time:    {lr_time:.1f}s")
    print(f"  CM:\n{lr_cm}")

    results["lr"] = {
        "roc_auc": lr_auc, "pr_auc": lr_pr, "brier": lr_brier,
        "recall_at_1pct_fpr": lr_r1, "time": lr_time,
        "cm": lr_cm.tolist()
    }
    all_models["lr"] = lr

    # --- Random Forest ---
    print("\n--- Random Forest ---")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0

    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    rf_auc = roc_auc_score(y_test, rf_proba)
    rf_pr = average_precision_score(y_test, rf_proba)
    rf_brier = brier_score_loss(y_test, rf_proba)
    rf_r1 = recall_at_fpr(y_test, rf_proba, 0.01)
    rf_cm = confusion_matrix(y_test, rf_pred)
    print(f"  ROC-AUC: {rf_auc:.4f}")
    print(f"  PR-AUC:  {rf_pr:.4f}")
    print(f"  Recall@1%FPR: {rf_r1:.4f}")
    print(f"  Brier:   {rf_brier:.6f}")
    print(f"  Time:    {rf_time:.1f}s")
    print(f"  CM:\n{rf_cm}")

    results["rf"] = {
        "roc_auc": rf_auc, "pr_auc": rf_pr, "brier": rf_brier,
        "recall_at_1pct_fpr": rf_r1, "time": rf_time,
        "cm": rf_cm.tolist()
    }
    all_models["rf"] = rf

    # Feature importance from RF
    print("\n  Top 15 features (RF importance):")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    for i, idx in enumerate(indices):
        print(f"  {i+1:2d}. {feature_cols[idx]:30s} {importances[idx]:.4f}")

    # --- XGBoost ---
    print("\n--- XGBoost ---")
    try:
        from xgboost import XGBClassifier
        t0 = time.time()
        xgb = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1),
            random_state=42, eval_metric="aucpr",
            use_label_encoder=False, n_jobs=-1
        )
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_time = time.time() - t0

        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        xgb_pred = (xgb_proba >= 0.5).astype(int)
        xgb_auc = roc_auc_score(y_test, xgb_proba)
        xgb_pr = average_precision_score(y_test, xgb_proba)
        xgb_brier = brier_score_loss(y_test, xgb_proba)
        xgb_r1 = recall_at_fpr(y_test, xgb_proba, 0.01)
        xgb_cm = confusion_matrix(y_test, xgb_pred)
        print(f"  ROC-AUC: {xgb_auc:.4f}")
        print(f"  PR-AUC:  {xgb_pr:.4f}")
        print(f"  Recall@1%FPR: {xgb_r1:.4f}")
        print(f"  Brier:   {xgb_brier:.6f}")
        print(f"  Time:    {xgb_time:.1f}s")
        print(f"  CM:\n{xgb_cm}")

        results["xgb"] = {
            "roc_auc": xgb_auc, "pr_auc": xgb_pr, "brier": xgb_brier,
            "recall_at_1pct_fpr": xgb_r1, "time": xgb_time,
            "cm": xgb_cm.tolist()
        }
        all_models["xgb"] = xgb

        # XGB feature importance
        print("\n  Top 15 features (XGB importance):")
        xgb_imp = xgb.feature_importances_
        xgb_idx = np.argsort(xgb_imp)[::-1][:15]
        for i, idx in enumerate(xgb_idx):
            print(f"  {i+1:2d}. {feature_cols[idx]:30s} {xgb_imp[idx]:.4f}")

    except ImportError:
        print("  XGBoost not available, skipping")

    # --- Ensemble (Stacker) ---
    print("\n--- Stacker Ensemble ---")
    # Build meta-features from training predictions via CV
    n_models = len(all_models)
    meta_train = np.zeros((len(X_train), n_models))
    meta_test = np.zeros((len(X_test), n_models))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        for j, (name, model) in enumerate(all_models.items()):
            if name == "lr":
                model.fit(X_train_s[tr_idx], y_train[tr_idx])
                meta_train[va_idx, j] = model.predict_proba(X_train_s[va_idx])[:, 1]
                meta_test[:, j] += model.predict_proba(X_test_s)[:, 1] / 5
            else:
                model.fit(X_train[tr_idx], y_train[tr_idx])
                meta_train[va_idx, j] = model.predict_proba(X_train[va_idx])[:, 1]
                meta_test[:, j] += model.predict_proba(X_test)[:, 1] / 5

    # Retrain base models on full training data for final meta_test
    for name, model in all_models.items():
        if name == "lr":
            model.fit(X_train_s, y_train)
            meta_test[:, list(all_models.keys()).index(name)] = model.predict_proba(X_test_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            meta_test[:, list(all_models.keys()).index(name)] = model.predict_proba(X_test)[:, 1]

    # Meta-learner
    t0 = time.time()
    meta_lr = LogisticRegression(C=10, max_iter=1000, random_state=42)
    meta_lr.fit(meta_train, y_train)
    stack_time = time.time() - t0

    stack_proba = meta_lr.predict_proba(meta_test)[:, 1]
    stack_pred = (stack_proba >= 0.5).astype(int)
    stack_auc = roc_auc_score(y_test, stack_proba)
    stack_pr = average_precision_score(y_test, stack_proba)
    stack_brier = brier_score_loss(y_test, stack_proba)
    stack_r1 = recall_at_fpr(y_test, stack_proba, 0.01)
    stack_cm = confusion_matrix(y_test, stack_pred)
    print(f"  ROC-AUC: {stack_auc:.4f}")
    print(f"  PR-AUC:  {stack_pr:.4f}")
    print(f"  Recall@1%FPR: {stack_r1:.4f}")
    print(f"  Brier:   {stack_brier:.6f}")
    print(f"  Time:    {stack_time:.1f}s")
    print(f"  CM:\n{stack_cm}")

    results["stacker"] = {
        "roc_auc": stack_auc, "pr_auc": stack_pr, "brier": stack_brier,
        "recall_at_1pct_fpr": stack_r1, "time": stack_time,
        "cm": stack_cm.tolist()
    }

    # --- Bootstrap CIs for best model ---
    print("\n--- Bootstrap Confidence Intervals (best model) ---")
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_proba = {
        "lr": lr_proba, "rf": rf_proba,
        "xgb": xgb_proba if "xgb_proba" in dir() else rf_proba,
        "stacker": stack_proba
    }[best_name]

    ci_auc = bootstrap_ci(y_test, best_proba, roc_auc_score)
    ci_pr = bootstrap_ci(y_test, best_proba, average_precision_score)
    print(f"  Best model: {best_name}")
    print(f"  ROC-AUC: {results[best_name]['roc_auc']:.4f}  95% CI: [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
    print(f"  PR-AUC:  {results[best_name]['pr_auc']:.4f}  95% CI: [{ci_pr[0]:.4f}, {ci_pr[1]:.4f}]")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY — Pattern Recognition Enhanced Results")
    print("=" * 70)
    print(f"{'Model':<12} {'ROC-AUC':>10} {'PR-AUC':>10} {'R@1%FPR':>10} {'Brier':>10}")
    print("-" * 52)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["roc_auc"]):
        print(f"{name:<12} {r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f} {r['recall_at_1pct_fpr']:>10.4f} {r['brier']:>10.6f}")

    # Save results
    output = {
        "dataset": "creditcard_v2",
        "dataset_hash": data_hash,
        "feature_version": "v2_pattern_recognition",
        "features_used": feature_cols,
        "n_features": len(feature_cols),
        "n_rows": len(df),
        "fraud_count": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "best_model": best_name,
        "confidence_intervals": {
            "roc_auc_95ci": list(ci_auc),
            "pr_auc_95ci": list(ci_pr),
        },
        "data_hash": data_hash,
        "model_hashes": {}
    }

    # Hash model artifacts
    import joblib, tempfile
    for name, model in all_models.items():
        tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        joblib.dump(model, tmp.name)
        output["model_hashes"][name] = compute_hash(Path(tmp.name))
        tmp.close()
        Path(tmp.name).unlink()

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "pattern_recognition_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")

    return results


if __name__ == "__main__":
    train_and_evaluate()
