#!/usr/bin/env python3
"""Financial-only fraud detection evaluation.

Evaluates the PS-14 system on real financial fraud datasets:
1. Kaggle ULB Credit Card Fraud (284K transactions, 492 fraud)
2. PS-14 Synthetic Transactions (9.8K transactions, 181 fraud)

Reports precision, recall, FPR, ROC-AUC, and cross-dataset generalization.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import json, warnings
warnings.filterwarnings("ignore")

DATA = Path("data")


def load_kaggle_creditcard() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Kaggle ULB Credit Card Fraud dataset."""
    df = pd.read_csv(DATA / "creditcard.csv")
    label_col = "Class"
    feat_cols = [c for c in df.columns if c not in [label_col, "Time"]]
    X = df[feat_cols].values
    y = df[label_col].values
    return X, y, feat_cols


def load_ps14_synthetic() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load PS-14 synthetic transaction dataset."""
    df = pd.read_csv(DATA / "transactions.csv")
    # Encode string columns
    bucket_map = {"micro": 0, "small": 0.25, "typical": 0.5, "large": 0.75, "unusual": 1.0}
    if "txn_amount_bucket" in df.columns:
        df["txn_amount_bucket"] = df["txn_amount_bucket"].map(bucket_map).fillna(0.5)
    # Drop non-feature columns
    drop_cols = ["event_id", "fraud_id", "ts", "archetype", "label"]
    # Drop any remaining string columns
    str_cols = [c for c in df.columns if df[c].dtype == object and c not in drop_cols]
    drop_cols.extend(str_cols)
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].values.astype(float)
    y = df["label"].values
    return X, y, feat_cols


def evaluate_model(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Train XGBoost with 5-fold CV and evaluate."""
    from xgboost import XGBClassifier

    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    params = {
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": spw, "min_child_weight": 5,
        "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1, "verbosity": 0,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, precs, recs, f1s = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = XGBClassifier(**params)
        model.fit(X_tr_s, y_tr)
        probs = model.predict_proba(X_val_s)[:, 1]

        aucs.append(roc_auc_score(y_val, probs))
        preds = (probs >= 0.5).astype(int)
        precs.append(precision_score(y_val, preds, zero_division=0))
        recs.append(recall_score(y_val, preds, zero_division=0))
        f1s.append(f1_score(y_val, preds, zero_division=0))

    return {
        "name": name,
        "roc_auc": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "f1": float(np.mean(f1s)),
        "n_samples": len(y),
        "n_fraud": int(y.sum()),
        "fraud_rate": float(y.mean() * 100),
    }


def cross_dataset_evaluate(X_source: np.ndarray, y_source: np.ndarray,
                            X_target: np.ndarray, y_target: np.ndarray,
                            source_name: str, target_name: str) -> dict:
    """Train on source, evaluate on target (cross-dataset transfer)."""
    from xgboost import XGBClassifier

    spw = float((y_source == 0).sum() / max((y_source == 1).sum(), 1))
    params = {
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": spw, "min_child_weight": 5,
        "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1, "verbosity": 0,
    }

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_source)
    X_te_s = scaler.transform(X_target)

    model = XGBClassifier(**params)
    model.fit(X_tr_s, y_source)
    probs = model.predict_proba(X_te_s)[:, 1]

    roc = roc_auc_score(y_target, probs) if len(np.unique(y_target)) > 1 else 0.0
    preds = (probs >= 0.5).astype(int)
    prec = precision_score(y_target, preds, zero_division=0)
    rec = recall_score(y_target, preds, zero_division=0)

    return {
        "source": source_name,
        "target": target_name,
        "roc_auc": float(roc),
        "precision": float(prec),
        "recall": float(rec),
        "n_target_fraud": int(y_target.sum()),
    }


def main():
    print("=" * 70)
    print("FINANCIAL FRAUD DETECTION — EVALUATION")
    print("=" * 70)

    # Load datasets
    print("\n[1] Loading financial datasets...")
    X_kaggle, y_kaggle, cols_kaggle = load_kaggle_creditcard()
    X_ps14, y_ps14, cols_ps14 = load_ps14_synthetic()

    print(f"  Kaggle Credit Card: {X_kaggle.shape[0]:,} samples, {y_kaggle.sum()} fraud ({y_kaggle.mean()*100:.2f}%)")
    print(f"  PS-14 Synthetic:    {X_ps14.shape[0]:,} samples, {y_ps14.sum()} fraud ({y_ps14.mean()*100:.1f}%)")

    # In-domain evaluation
    print("\n[2] In-domain evaluation (5-fold CV)...")
    kaggle_result = evaluate_model(X_kaggle, y_kaggle, "Kaggle Credit Card")
    ps14_result = evaluate_model(X_ps14, y_ps14, "PS-14 Synthetic")

    print(f"\n  {'Dataset':<25s} | {'ROC-AUC':>8s} | {'Precision':>9s} | {'Recall':>7s} | {'F1':>5s} | {'Fraud':>6s}")
    print("  " + "-" * 75)
    for r in [kaggle_result, ps14_result]:
        print(f"  {r['name']:<25s} | {r['roc_auc']:8.4f} | {r['precision']:8.1%} | {r['recall']:6.1%} | {r['f1']:.3f} | {r['n_fraud']:5d}")

    # Cross-dataset evaluation (skip if feature dimensions differ)
    print("\n[3] Cross-dataset transfer...")
    transfer_results = []
    if X_kaggle.shape[1] == X_ps14.shape[1]:
        t1 = cross_dataset_evaluate(X_kaggle, y_kaggle, X_ps14, y_ps14, "Kaggle", "PS-14")
        t2 = cross_dataset_evaluate(X_ps14, y_ps14, X_kaggle, y_kaggle, "PS-14", "Kaggle")
        transfer_results = [t1, t2]
        print(f"\n  {'Source -> Target':<30s} | {'ROC-AUC':>8s} | {'Precision':>9s} | {'Recall':>7s}")
        print("  " + "-" * 65)
        for r in transfer_results:
            print(f"  {r['source'] + ' -> ' + r['target']:<30s} | {r['roc_auc']:8.4f} | {r['precision']:8.1%} | {r['recall']:6.1%}")
    else:
        print(f"  Skipped: feature dimensions differ (Kaggle={X_kaggle.shape[1]}, PS-14={X_ps14.shape[1]})")

    # Summary
    avg_roc = (kaggle_result["roc_auc"] + ps14_result["roc_auc"]) / 2
    print(f"\n{'='*70}")
    print(f"FINANCIAL DATASET SUMMARY:")
    print(f"  Average ROC-AUC:     {avg_roc:.4f}")
    print(f"  Kaggle:              {kaggle_result['roc_auc']:.4f} (precision={kaggle_result['precision']:.1%})")
    print(f"  PS-14:               {ps14_result['roc_auc']:.4f} (precision={ps14_result['precision']:.1%})")
    print(f"{'='*70}")

    # Save results
    results = {
        "kaggle_creditcard": kaggle_result,
        "ps14_synthetic": ps14_result,
        "cross_transfer": transfer_results,
    }
    with open(DATA / "financial_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to data/financial_evaluation.json")


if __name__ == "__main__":
    main()
