#!/usr/bin/env python3
"""Cross-Domain Model Comparison.

Trains XGBoost on three source domains (Kaggle, UCI, Combined) and
evaluates each model on both target domains. Reports ROC-AUC, PR-AUC,
precision@recall, and confusion matrices.

Run: python scripts/cross_domain_compare.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORT_PATH = DATA / "cross_domain_results.json"

ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "hour_of_day", "is_weekend",
    "shared_device_accounts", "shared_recipient_accounts", "mule_ring_score",
]


def load_kaggle_ps14() -> tuple[pd.DataFrame, np.ndarray]:
    """Load Kaggle data that's already been transformed to PS-14 features."""
    df = pd.read_csv(DATA / "validation_kaggle.csv")
    available = [f for f in ML_FEATURES if f in df.columns]
    X = df[available].values
    y = df["label"].values
    print(f"  Kaggle: {X.shape[0]:,} rows, {y.sum():,} fraud ({y.mean()*100:.2f}%)")
    return df, X, y, available


def load_uci_ps14() -> tuple[pd.DataFrame, np.ndarray]:
    """Load UCI data that's already been transformed to PS-14 features."""
    df = pd.read_csv(DATA / "validation_uci_default.csv")
    available = [f for f in ML_FEATURES if f in df.columns]
    X = df[available].values
    y = df["label"].values
    print(f"  UCI:    {X.shape[0]:,} rows, {y.sum():,} fraud ({y.mean()*100:.2f}%)")
    return df, X, y, available


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  name: str) -> dict:
    """Train XGBoost with precision-focused hyperparameters."""
    scale_pos = max(1, int((y_train == 0).sum() / max((y_train == 1).sum(), 1)))
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        random_state=42,
        use_label_encoder=False,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    
    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    # Precision at various thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Find precision at 50% recall
    idx_50r = np.argmin(np.abs(recalls - 0.5))
    prec_at_50r = precisions[idx_50r]
    
    # Find recall at 90% precision
    idx_90p = np.argmin(np.abs(precisions - 0.9))
    recall_at_90 = recalls[idx_90p]
    
    # Best F1 threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_f1_idx, len(thresholds) - 1)]
    best_f1 = f1_scores[best_f1_idx]
    
    # Classification report at optimal threshold
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred_opt, output_dict=True, zero_division=0)
    
    # Confusion matrix components
    tp = int(((y_pred_opt == 1) & (y_test == 1)).sum())
    fp = int(((y_pred_opt == 1) & (y_test == 0)).sum())
    tn = int(((y_pred_opt == 0) & (y_test == 0)).sum())
    fn = int(((y_pred_opt == 0) & (y_test == 1)).sum())
    
    return {
        "model_name": name,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_at_50pct_recall": round(prec_at_50r, 4),
        "recall_at_90pct_precision": round(recall_at_90, 4),
        "best_f1": round(best_f1, 4),
        "best_threshold": round(best_threshold, 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": round(report["1"]["precision"], 4),
        "recall": round(report["1"]["recall"], 4),
        "f1_score": round(report["1"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
        "feature_importance": dict(sorted(
            zip(ML_FEATURES[:len(model.feature_importances_)],
                [round(float(x), 4) for x in model.feature_importances_]),
            key=lambda x: -x[1]
        )[:8]),
    }


def main():
    print("=" * 70)
    print("CROSS-DOMAIN MODEL COMPARISON")
    print("=" * 70)
    
    # Load datasets
    print("\n[1] Loading datasets...")
    _, X_kaggle, y_kaggle, feats_k = load_kaggle_ps14()
    _, X_uci, y_uci, feats_uci = load_uci_ps14()
    
    # Ensure same features
    common_feats = [f for f in ML_FEATURES if f in feats_k and f in feats_uci]
    print(f"\n  Common features: {len(common_feats)}")
    
    # Use common features only
    feat_idx_k = [feats_k.index(f) for f in common_feats]
    feat_idx_u = [feats_uci.index(f) for f in common_feats]
    X_k = X_kaggle[:, feat_idx_k]
    X_u = X_uci[:, feat_idx_u]
    
    # Scale features
    scaler_k = StandardScaler().fit(X_k)
    scaler_u = StandardScaler().fit(X_u)
    X_k_scaled = scaler_k.transform(X_k)
    X_u_scaled = scaler_u.transform(X_u)
    
    # Combine for joint training
    X_combined = np.vstack([X_k_scaled, X_u_scaled])
    y_combined = np.concatenate([y_kaggle, y_uci])
    
    results = {}
    
    # ---- Scenario 1: Train on Kaggle, Test on Kaggle (in-domain) ----
    print("\n[2] Scenario 1: Train Kaggle → Test Kaggle (in-domain baseline)")
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_k_scaled, y_kaggle, test_size=0.2, stratify=y_kaggle, random_state=42
    )
    results["kaggle_to_kaggle"] = train_xgboost(X_tr, y_tr, X_te, y_te,
                                                 "Kaggle → Kaggle (in-domain)")
    r = results["kaggle_to_kaggle"]
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}")
    print(f"  Precision: {r['precision']:.4f}  Recall: {r['recall']:.4f}")
    print(f"  F1: {r['f1_score']:.4f}  Acc: {r['accuracy']:.4f}")
    
    # ---- Scenario 2: Train on Kaggle, Test on UCI (transfer) ----
    print("\n[3] Scenario 2: Train Kaggle → Test UCI (cross-domain transfer)")
    results["kaggle_to_uci"] = train_xgboost(X_k_scaled, y_kaggle,
                                              X_u_scaled, y_uci,
                                              "Kaggle → UCI (transfer)")
    r = results["kaggle_to_uci"]
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}")
    print(f"  Precision: {r['precision']:.4f}  Recall: {r['recall']:.4f}")
    print(f"  F1: {r['f1_score']:.4f}  Acc: {r['accuracy']:.4f}")
    
    # ---- Scenario 3: Train on UCI, Test on UCI (in-domain) ----
    print("\n[4] Scenario 3: Train UCI → Test UCI (in-domain)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_u_scaled, y_uci, test_size=0.2, stratify=y_uci, random_state=42
    )
    results["uci_to_uci"] = train_xgboost(X_tr, y_tr, X_te, y_te,
                                           "UCI → UCI (in-domain)")
    r = results["uci_to_uci"]
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}")
    print(f"  Precision: {r['precision']:.4f}  Recall: {r['recall']:.4f}")
    print(f"  F1: {r['f1_score']:.4f}  Acc: {r['accuracy']:.4f}")
    
    # ---- Scenario 4: Train on UCI, Test on Kaggle (reverse transfer) ----
    print("\n[5] Scenario 4: Train UCI → Test Kaggle (reverse transfer)")
    results["uci_to_kaggle"] = train_xgboost(X_u_scaled, y_uci,
                                              X_k_scaled, y_kaggle,
                                              "UCI → Kaggle (reverse transfer)")
    r = results["uci_to_kaggle"]
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}")
    print(f"  Precision: {r['precision']:.4f}  Recall: {r['recall']:.4f}")
    print(f"  F1: {r['f1_score']:.4f}  Acc: {r['accuracy']:.4f}")
    
    # ---- Scenario 5: Train on Combined, Test on each ----
    print("\n[6] Scenario 5: Train Combined → Test Kaggle")
    results["combined_to_kaggle"] = train_xgboost(X_combined, y_combined,
                                                   X_k_scaled, y_kaggle,
                                                   "Combined → Kaggle")
    r = results["combined_to_kaggle"]
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}")
    print(f"  Precision: {r['precision']:.4f}  Recall: {r['recall']:.4f}")
    
    print("\n[7] Scenario 6: Train Combined → Test UCI")
    results["combined_to_uci"] = train_xgboost(X_combined, y_combined,
                                                X_u_scaled, y_uci,
                                                "Combined → UCI")
    r = results["combined_to_uci"]
    print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}")
    print(f"  Precision: {r['precision']:.4f}  Recall: {r['recall']:.4f}")
    
    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("SUMMARY: Cross-Domain ROC-AUC Comparison")
    print("=" * 70)
    print(f"{'Scenario':<40} {'ROC-AUC':>8} {'PR-AUC':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 80)
    for key, r in results.items():
        print(f"{r['model_name']:<40} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1_score']:>8.4f}")
    
    # ---- Transfer drop analysis ----
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING ANALYSIS")
    print("=" * 70)
    
    kg_in = results["kaggle_to_kaggle"]["roc_auc"]
    kg_xfer = results["kaggle_to_uci"]["roc_auc"]
    uci_in = results["uci_to_uci"]["roc_auc"]
    uci_xfer = results["uci_to_kaggle"]["roc_auc"]
    comb_kg = results["combined_to_kaggle"]["roc_auc"]
    comb_uci = results["combined_to_uci"]["roc_auc"]
    
    print(f"  Kaggle in-domain:    {kg_in:.4f}")
    print(f"  Kaggle→UCI transfer: {kg_xfer:.4f}  (drop: {kg_in - kg_xfer:+.4f})")
    print(f"  UCI in-domain:       {uci_in:.4f}")
    print(f"  UCI→Kaggle transfer: {uci_xfer:.4f}  (drop: {uci_in - uci_xfer:+.4f})")
    print(f"  Combined→Kaggle:     {comb_kg:.4f}  (vs in-domain: {comb_kg - kg_in:+.4f})")
    print(f"  Combined→UCI:        {comb_uci:.4f}  (vs in-domain: {comb_uci - uci_in:+.4f})")
    
    best_transfer = max(kg_xfer, uci_xfer, comb_kg, comb_uci)
    print(f"\n  Best transfer ROC-AUC: {best_transfer:.4f}")
    
    # ---- Feature importance comparison ----
    print("\n" + "=" * 70)
    print("TOP FEATURES BY TRAINING DOMAIN")
    print("=" * 70)
    for key in ["kaggle_to_kaggle", "uci_to_uci", "combined_to_kaggle"]:
        fi = results[key]["feature_importance"]
        top3 = list(fi.items())[:3]
        print(f"  {results[key]['model_name']:<40}: {', '.join(f'{k}={v:.3f}' for k,v in top3)}")
    
    # Save report
    report = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "datasets": {
            "kaggle": {"rows": int(y_kaggle.shape[0]), "fraud": int(y_kaggle.sum())},
            "uci": {"rows": int(y_uci.shape[0]), "fraud": int(y_uci.sum())},
        },
        "scenarios": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport saved -> {REPORT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
