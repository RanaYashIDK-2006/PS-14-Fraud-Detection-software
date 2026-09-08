#!/usr/bin/env python3
"""Large-scale training and evaluation with temporal splits and OOD holdout.

Trains on the 1.5M dataset with strict temporal splits:
  - Train: months 1-18 (2024-01 to 2025-06)
  - Validation: months 19-21 (2025-07 to 2025-09)
  - Test: months 22-24 (2025-10 to 2025-12)

Also creates OOD holdout:
  - merchant_fraud held out from training entirely
  - Evaluated as unseen fraud archetype

Produces the final pitch-quality benchmark table.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from privacy_layer.features import ML_FEATURES

# ── Dataset paths ────────────────────────────────────────────────────────

LARGE_DATA = ROOT / "data" / "transactions_large.csv"
ARTIFACTS_DIR = ROOT / "models" / "production"
EXPERIMENTS_DIR = ROOT / "models" / "experiments"


def load_dataset() -> pd.DataFrame:
    """Load the large dataset."""
    if not LARGE_DATA.exists():
        print(f"ERROR: Dataset not found at {LARGE_DATA}")
        print("Run: python scripts/generate_large_dataset.py")
        sys.exit(1)
    return pd.read_csv(LARGE_DATA)


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict temporal split — no shuffling.

    Train: months 1-18 (2024-01 to 2025-06)
    Validation: months 19-21 (2025-07 to 2025-09)
    Test: months 22-24 (2025-10 to 2025-12)
    """
    df["ts"] = pd.to_datetime(df["ts"])

    train_end = pd.Timestamp("2025-06-30 23:59:59")
    val_end = pd.Timestamp("2025-09-30 23:59:59")

    train = df[df["ts"] <= train_end].copy()
    val = df[(df["ts"] > train_end) & (df["ts"] <= val_end)].copy()
    test = df[df["ts"] > val_end].copy()

    return train, val, test


def ood_holdout(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    ood_archetype: str = "merchant_fraud",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out one archetype entirely from train/val for OOD evaluation.

    Returns: (train_ood, val_ood, test_ood, test_ood_with_legit)
    """
    # Remove OOD archetype from train and val
    train_ood = train[train["archetype"] != ood_archetype].copy()
    val_ood = val[val["archetype"] != ood_archetype].copy()

    # Test OOD: ONLY the held-out archetype + some legitimate samples
    test_ood_fraud = test[test["archetype"] == ood_archetype].copy()
    test_ood_legit = test[test["archetype"] == "legit"].sample(
        n=min(len(test_ood_fraud) * 3, (test["archetype"] == "legit").sum()),
        random_state=42,
    )
    test_ood = pd.concat([test_ood_fraud, test_ood_legit]).sort_values("ts").reset_index(drop=True)

    return train_ood, val_ood, test_ood_fraud, test_ood


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    prevalence: float | None = None,
) -> dict:
    """Compute comprehensive metrics with bootstrap CIs."""
    # Basic metrics
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": 0.0, "pr_auc": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "recall_at_01pct_fpr": 0.0, "recall_at_05pct_fpr": 0.0,
            "recall_at_1pct_fpr": 0.0, "brier": 0.0, "ece": 0.0,
            "threshold": threshold, "n_samples": len(y_true),
            "n_fraud": int(y_true.sum()), "n_legit": int((y_true == 0).sum()),
            "prevalence": round(float(y_true.mean()), 4) if len(y_true) > 0 else 0,
            "ci_roc_auc": (0.0, 0.0), "ci_pr_auc": (0.0, 0.0),
            "ci_recall": (0.0, 0.0),
        }
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)

    # Recall at fixed FPR levels
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    def recall_at_fpr(target_fpr: float) -> float:
        idx = np.searchsorted(fpr_arr, target_fpr, side="left")
        if idx >= len(tpr_arr):
            return 0.0
        return float(tpr_arr[idx])

    recall_01 = recall_at_fpr(0.001)
    recall_05 = recall_at_fpr(0.005)
    recall_1 = recall_at_fpr(0.01)

    # At operating threshold
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # ECE (Expected Calibration Error)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)

    # Bootstrap CI (200 samples for speed)
    n_boot = 200
    boot_roc = []
    boot_pr = []
    boot_recall = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_roc.append(roc_auc_score(y_true[idx], y_prob[idx]))
        boot_pr.append(average_precision_score(y_true[idx], y_prob[idx]))
        y_pred_boot = (y_prob[idx] >= threshold).astype(int)
        cm_boot = confusion_matrix(y_true[idx], y_pred_boot)
        if cm_boot.shape == (2, 2):
            tn_b, fp_b, fn_b, tp_b = cm_boot.ravel()
            boot_recall.append(tp_b / max(tp_b + fn_b, 1))

    ci_roc = (np.percentile(boot_roc, 2.5), np.percentile(boot_roc, 97.5)) if boot_roc else (0, 0)
    ci_pr = (np.percentile(boot_pr, 2.5), np.percentile(boot_pr, 97.5)) if boot_pr else (0, 0)
    ci_recall = (np.percentile(boot_recall, 2.5), np.percentile(boot_recall, 97.5)) if boot_recall else (0, 0)

    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "recall_at_01pct_fpr": round(recall_01, 4),
        "recall_at_05pct_fpr": round(recall_05, 4),
        "recall_at_1pct_fpr": round(recall_1, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_fraud": int(y_true.sum()),
        "n_legit": int((y_true == 0).sum()),
        "prevalence": round(float(y_true.mean()), 4),
        "ci_roc_auc": (round(ci_roc[0], 4), round(ci_roc[1], 4)),
        "ci_pr_auc": (round(ci_pr[0], 4), round(ci_pr[1], 4)),
        "ci_recall": (round(ci_recall[0], 4), round(ci_recall[1], 4)),
    }


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Train the model ensemble and return fitted models."""
    print("Training models...")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Logistic Regression
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_s, y_train)
    print(f"  Logistic Regression: {time.time()-t0:.1f}s")

    # Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)
    print(f"  Random Forest: {time.time()-t0:.1f}s")

    # XGBoost (using GradientBoosting as fallback)
    t0 = time.time()
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, scale_pos_weight=(y_train == 0).sum() / max(y_train.sum(), 1),
            eval_metric="aucpr", early_stopping_rounds=20,
        )
        xgb.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    except ImportError:
        xgb = GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42,
        )
        xgb.fit(X_train_s, y_train)
    print(f"  XGBoost: {time.time()-t0:.1f}s")

    # Compute validation scores for stacking
    lr_val = lr.predict_proba(X_val_s)[:, 1]
    rf_val = rf.predict_proba(X_val_s)[:, 1]
    xgb_val = xgb.predict_proba(X_val_s)[:, 1]

    # Simple average stacker (no extra fitting on val)
    stack_val = (lr_val + rf_val + xgb_val) / 3.0

    return {
        "scaler": scaler,
        "lr": lr,
        "rf": rf,
        "xgb": xgb,
        "val_scores": {"lr": lr_val, "rf": rf_val, "xgb": xgb_val, "stack": stack_val},
    }


def evaluate_split(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    archetype_col: np.ndarray | None = None,
) -> dict:
    """Evaluate models on a split and return comprehensive metrics."""
    scaler = models["scaler"]
    X_s = scaler.transform(X)

    results = {}
    for name, model in [("lr", models["lr"]), ("rf", models["rf"]), ("xgb", models["xgb"])]:
        prob = model.predict_proba(X_s)[:, 1]
        metrics = compute_metrics(y, prob)
        results[name] = metrics

    # Stacker
    lr_prob = models["lr"].predict_proba(X_s)[:, 1]
    rf_prob = models["rf"].predict_proba(X_s)[:, 1]
    xgb_prob = models["xgb"].predict_proba(X_s)[:, 1]
    stack_prob = (lr_prob + rf_prob + xgb_prob) / 3.0
    results["stack"] = compute_metrics(y, stack_prob)

    # Per-archetype breakdown (for test set)
    if archetype_col is not None:
        archetypes = {}
        for arch in np.unique(archetype_col):
            mask = archetype_col == arch
            if mask.sum() > 0:
                arch_metrics = compute_metrics(y[mask], stack_prob[mask])
                archetypes[str(arch)] = arch_metrics
        results["per_archetype"] = archetypes

    return results


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 on validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-10)
    best_idx = np.argmax(f1)
    return float(thresholds[min(best_idx, len(thresholds) - 1)])


def main():
    print("=" * 80)
    print("PS-14 LARGE-SCALE TRAINING AND EVALUATION")
    print("=" * 80)

    # Load dataset
    df = load_dataset()
    print(f"\nDataset: {len(df):,} transactions, {int(df.label.sum()):,} fraud")

    # Temporal split
    train_full, val_full, test_full = temporal_split(df)
    print(f"\nTemporal split:")
    print(f"  Train:    {len(train_full):,} rows ({train_full.ts.min().date()} to {train_full.ts.max().date()})")
    print(f"  Val:      {len(val_full):,} rows ({val_full.ts.min().date()} to {val_full.ts.max().date()})")
    print(f"  Test:     {len(test_full):,} rows ({test_full.ts.min().date()} to {test_full.ts.max().date()})")

    # Feature matrices
    available_features = [f for f in ML_FEATURES if f in train_full.columns]
    print(f"\nFeatures: {len(available_features)}/{len(ML_FEATURES)}")

    X_train = train_full[available_features].values
    y_train = train_full["label"].values
    X_val = val_full[available_features].values
    y_val = val_full["label"].values
    X_test = test_full[available_features].values
    y_test = test_full["label"].values

    # ── OOD Holdout ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OOD HOLDOUT: merchant_fraud held out from training")
    print("=" * 80)

    train_ood, val_ood, test_ood_fraud, test_ood = ood_holdout(
        train_full, val_full, test_full, "merchant_fraud"
    )
    print(f"  Train (no merchant_fraud): {len(train_ood):,}")
    print(f"  Val (no merchant_fraud):   {len(val_ood):,}")
    print(f"  OOD test fraud:            {len(test_ood_fraud):,}")
    print(f"  OOD test total:            {len(test_ood):,}")

    X_train_ood = train_ood[available_features].values
    y_train_ood = train_ood["label"].values
    X_val_ood = val_ood[available_features].values
    y_val_ood = val_ood["label"].values
    X_test_ood = test_ood[available_features].values
    y_test_ood = test_ood["label"].values

    # ── Train full model ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING: Full model (all archetypes)")
    print("=" * 80)

    models_full = train_models(X_train, y_train, X_val, y_val)

    # Find optimal threshold on validation
    threshold = find_optimal_threshold(y_val, models_full["val_scores"]["stack"])
    print(f"\nOptimal threshold (val F1): {threshold:.4f}")

    # ── Train OOD model ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING: OOD model (merchant_fraud excluded)")
    print("=" * 80)

    models_ood = train_models(X_train_ood, y_train_ood, X_val_ood, y_val_ood)
    threshold_ood = find_optimal_threshold(y_val_ood, models_ood["val_scores"]["stack"])
    print(f"\nOptimal threshold (val F1): {threshold_ood:.4f}")

    # ── Evaluate all splits ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    all_results = {}

    # 1. In-domain (temporal test)
    print("\n--- In-Domain (temporal test) ---")
    res_in = evaluate_split(
        models_full, X_test, y_test, "in_domain",
        test_full["archetype"].values,
    )
    all_results["in_domain"] = res_in
    s = res_in["stack"]
    print(f"  Stack: ROC-AUC={s['roc_auc']:.4f}  PR-AUC={s['pr_auc']:.4f}  "
          f"Recall@1%FPR={s['recall_at_1pct_fpr']:.4f}  "
          f"Brier={s['brier']:.4f}  ECE={s['ece']:.4f}")
    print(f"         95% CI ROC-AUC={s['ci_roc_auc']}  95% CI PR-AUC={s['ci_pr_auc']}")

    # 2. OOD fraud-type
    print("\n--- OOD (merchant_fraud unseen) ---")
    res_ood = evaluate_split(
        models_ood, X_test_ood, y_test_ood, "ood_fraud_type",
    )
    all_results["ood_fraud_type"] = res_ood
    s = res_ood["stack"]
    print(f"  Stack: ROC-AUC={s['roc_auc']:.4f}  PR-AUC={s['pr_auc']:.4f}  "
          f"Recall@1%FPR={s['recall_at_1pct_fpr']:.4f}  "
          f"Brier={s['brier']:.4f}  ECE={s['ece']:.4f}")

    # 3. Per-archetype breakdown
    print("\n--- Per-Archetype Breakdown (in-domain test) ---")
    if "per_archetype" in res_in:
        for arch, metrics in sorted(res_in["per_archetype"].items()):
            n_f = metrics["n_fraud"]
            n_l = metrics["n_legit"]
            valid = "VALID" if n_l > 0 else "INVALID (no legit samples)"
            print(f"  {arch:25s} ROC={metrics['roc_auc']:.4f}  PR={metrics['pr_auc']:.4f}  "
                  f"n_fraud={n_f}  n_legit={n_l}  {valid}")

    # ── Model artifacts ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SAVING ARTIFACTS")
    print("=" * 80)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save production artifacts
    joblib.dump(models_full["scaler"], ARTIFACTS_DIR / "scaler.joblib")
    joblib.dump(models_full["xgb"], ARTIFACTS_DIR / "model.joblib")
    joblib.dump(models_full["lr"], ARTIFACTS_DIR / "logistic_regression.joblib")
    joblib.dump(models_full["rf"], ARTIFACTS_DIR / "random_forest.joblib")

    # Save OOD model
    joblib.dump(models_ood["scaler"], EXPERIMENTS_DIR / "scaler_ood.joblib")
    joblib.dump(models_ood["xgb"], EXPERIMENTS_DIR / "model_ood.joblib")

    # Compute hashes
    def file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # Manifest
    manifest = {
        "model_version": "v2.0-large",
        "feature_version": "v2.0-deviation",
        "dataset_version": "large-1.5M",
        "dataset_hash": "5d611945564f3414",
        "n_features": len(available_features),
        "features": available_features,
        "threshold": threshold,
        "training_rows": len(train_full),
        "validation_rows": len(val_full),
        "test_rows": len(test_full),
        "training_period": f"{train_full.ts.min().date()} to {train_full.ts.max().date()}",
        "validation_period": f"{val_full.ts.min().date()} to {val_full.ts.max().date()}",
        "test_period": f"{test_full.ts.min().date()} to {test_full.ts.max().date()}",
        "seed": 42,
        "model_hashes": {
            "model": file_hash(ARTIFACTS_DIR / "model.joblib"),
            "scaler": file_hash(ARTIFACTS_DIR / "scaler.joblib"),
            "lr": file_hash(ARTIFACTS_DIR / "logistic_regression.joblib"),
            "rf": file_hash(ARTIFACTS_DIR / "random_forest.joblib"),
        },
    }
    with open(ARTIFACTS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Production artifacts: {ARTIFACTS_DIR}")
    print(f"  OOD model: {EXPERIMENTS_DIR}")

    # ── Save results ─────────────────────────────────────────────────────
    results_path = ROOT / "models" / "large_scale_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Generate benchmark table ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK TABLE")
    print("=" * 80)

    header = f"{'Dataset':<30} {'N':>8} {'Fraud':>8} {'Prev':>8} {'ROC-AUC':>10} {'PR-AUC':>10} {'R@0.1%':>8} {'R@0.5%':>8} {'R@1%':>8} {'Brier':>8} {'ECE':>8}"
    print(header)
    print("-" * len(header))

    for split_name, metrics in all_results.items():
        s = metrics["stack"]
        name = split_name.replace("_", " ").title()
        print(f"{name:<30} {s['n_samples']:>8,} {s['n_fraud']:>8,} {s['prevalence']:>8.2%} "
              f"{s['roc_auc']:>10.4f} {s['pr_auc']:>10.4f} "
              f"{s['recall_at_01pct_fpr']:>8.4f} {s['recall_at_05pct_fpr']:>8.4f} "
              f"{s['recall_at_1pct_fpr']:>8.4f} {s['brier']:>8.4f} {s['ece']:>8.4f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
