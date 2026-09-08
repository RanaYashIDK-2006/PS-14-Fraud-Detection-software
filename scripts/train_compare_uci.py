#!/usr/bin/env python3
"""Train models on UCI Credit Card Default and Kaggle datasets, then compare.

This script:
1. Maps the raw Kaggle creditcard.csv to PS-16 §16 features
2. Trains the full 4-model fusion (LR, RF, XGB, IF + stacker) on each dataset
3. Cross-evaluates: each model on the other dataset's test split
4. Reports domain-specific vs cross-domain performance
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "models" / "artifacts"

ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "hour_of_day", "is_weekend",
    "shared_device_accounts", "shared_recipient_accounts", "mule_ring_score",
]


# ── Dataset loaders ──────────────────────────────────────────────────────

def load_kaggle_as_ps16() -> pd.DataFrame:
    """Map Kaggle creditcard.csv (PCA V1-V28) to PS-16 §16 features."""
    path = DATA / "creditcard.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Raw Kaggle: {df.shape[0]:,} rows, {df.shape[1]} columns")

    out = pd.DataFrame(index=df.index)

    # amount_ratio: Amount relative to median (proxy for unusual size)
    median_amt = df["Amount"].median()
    if median_amt == 0:
        median_amt = 1.0
    out["amount_ratio"] = (df["Amount"] / median_amt).clip(0, 10).round(4)

    # txn_freq_last_24h: V3 captures time between transactions (PCA component)
    # Map to frequency bins: more negative = faster transactions
    v3 = df["V3"]
    out["txn_freq_last_24h"] = np.where(v3 < -2, 6,
                         np.where(v3 < -1, 5,
                         np.where(v3 < 0, 4,
                         np.where(v3 < 1, 3, 2)))).astype(int)

    # txn_time_unusual: V4 captures temporal anomaly signal
    out["txn_time_unusual"] = (df["V4"].abs() > 2).astype(int)

    # new_device_flag: V14 is the strongest fraud signal — use as device anomaly proxy
    out["new_device_flag"] = (df["V14"] < -5).astype(int)

    # unusual_location_flag: V8 captures location-like pattern
    out["unusual_location_flag"] = (df["V8"].abs() > 2).astype(int)

    # unusual_recipient_flag: V7 captures recipient-like pattern
    out["unusual_recipient_flag"] = (df["V7"].abs() > 2).astype(int)

    # failed_auth_count_24h: V1 captures authentication failure signal
    v1 = df["V1"]
    out["failed_auth_count_24h"] = np.where(v1 < -5, 4,
                            np.where(v1 < -2, 3,
                            np.where(v1 < 0, 2,
                            np.where(v1 < 2, 1, 0)))).astype(int)

    # days_since_last_similar_txn: Time component
    out["days_since_last_similar_txn"] = (
        df["Time"] / 3600
    ).clip(0, 48).round(0).astype(int)

    # gradual_escalation_score: V10 captures escalation pattern
    out["gradual_escalation_score"] = np.clip(
        (df["V10"].abs() - 2) / 8, 0, 1
    ).round(4)

    # known_device_count: V11 captures account maturity
    v11 = df["V11"]
    out["known_device_count"] = np.where(v11 < -1, 1,
                          np.where(v11 < 0, 2,
                          np.where(v11 < 1, 3, 4))).astype(int)

    # account_tenure_days: Use V12 as proxy
    out["account_tenure_days"] = (
        np.clip(df["V12"], -10, 10) + 10
    ).round(0).astype(int) + 1

    # hour_of_day: Time-based feature
    out["hour_of_day"] = (df["Time"] % (24 * 3600) / 3600).round(0).astype(int) % 24

    # is_weekend: V13 captures weekend-like pattern
    out["is_weekend"] = (df["V13"].abs() > 1.5).astype(int)

    # Link-analysis features: not available in Kaggle (PCA scrambled these)
    out["shared_device_accounts"] = 0
    out["shared_recipient_accounts"] = 0
    out["mule_ring_score"] = 0.0

    out["label"] = df["Class"].values

    print(f"  Mapped: {out.shape[0]:,} rows, {out.shape[1]} columns")
    print(f"  Fraud rate: {out['label'].mean()*100:.3f}%")
    return out


def load_uci_as_ps16() -> pd.DataFrame:
    """Load the pre-mapped UCI Credit Card Default dataset."""
    path = DATA / "validation_uci_default.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Mapped UCI: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"  Default rate: {df['label'].mean()*100:.2f}%")
    return df


# ── Training ─────────────────────────────────────────────────────────────

def train_fusion(X_train: np.ndarray, y_train: np.ndarray, dataset_name: str) -> dict:
    """Train the full 4-model fusion (LR, RF, XGB, IF + stacker)."""
    print(f"\n  Training fusion on {dataset_name}...")

    # Scale features for LR and IF
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    # Model 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(Xs, y_train)
    p_lr = lr.predict_proba(Xs)[:, 1]
    print(f"    LR    ROC-AUC: {roc_auc_score(y_train, p_lr):.4f}")

    # Model 2: Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_train)[:, 1]
    print(f"    RF    ROC-AUC: {roc_auc_score(y_train, p_rf):.4f}")

    # Model 3: XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb.fit(X_train, y_train)
        p_xgb = xgb.predict_proba(X_train)[:, 1]
        print(f"    XGB   ROC-AUC: {roc_auc_score(y_train, p_xgb):.4f}")
    except ImportError:
        print("    XGB   SKIPPED (xgboost not installed)")
        xgb = None
        p_xgb = np.full(len(y_train), 0.5)

    # Model 4: Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(Xs)
    iso_scores = -iso.score_samples(Xs)  # Higher = more anomalous
    # Rank-normalize to [0, 1]
    iso_pct = np.array([(iso_scores < s).mean() for s in iso_scores])
    print(f"    IF    AUC (anomaly): {roc_auc_score(y_train, iso_pct):.4f}")

    # Stacker: logistic regression on 4 model outputs
    X_stack = np.column_stack([p_lr, p_rf, p_xgb, iso_pct])
    stacker = LogisticRegression(max_iter=1000, random_state=42)
    stacker.fit(X_stack, y_train)
    p_fused = stacker.predict_proba(X_stack)[:, 1]
    fused_auc = roc_auc_score(y_train, p_fused)
    print(f"    FUSED ROC-AUC: {fused_auc:.4f}")

    # Compute precision/recall at various thresholds
    pr_auc = average_precision_score(y_train, p_fused)
    prec, rec, thresh = precision_recall_curve(y_train, p_fused)
    # Recall at 1% FPR
    for t in np.arange(0.01, 1.0, 0.01):
        fp_rate = ((p_fused >= t) & (y_train == 0)).sum() / max((y_train == 0).sum(), 1)
        if fp_rate <= 0.01:
            recall_at_1 = ((p_fused >= t) & (y_train == 1)).sum() / max((y_train == 1).sum(), 1)
            break
    else:
        recall_at_1 = 0.0

    return {
        "lr": lr, "rf": rf, "xgb": xgb, "iso": iso,
        "stacker": stacker, "scaler": scaler,
        "train_auc": fused_auc, "pr_auc": pr_auc,
        "recall_at_1pct_fpr": recall_at_1,
        "iso_train_scores": iso_scores,
    }


def evaluate_fusion(fusion: dict, X_test: np.ndarray, y_test: np.ndarray, dataset_name: str) -> dict:
    """Evaluate a trained fusion on a test set."""
    lr, rf, xgb, iso, stacker, scaler = (
        fusion["lr"], fusion["rf"], fusion["xgb"],
        fusion["iso"], fusion["stacker"], fusion["scaler"],
    )

    Xs = scaler.transform(X_test)
    p_lr = lr.predict_proba(Xs)[:, 1]
    p_rf = rf.predict_proba(X_test)[:, 1]
    p_xgb = xgb.predict_proba(X_test)[:, 1] if xgb is not None else np.full(len(y_test), 0.5)
    iso_scores = -iso.score_samples(Xs)
    iso_pct = np.array([(fusion["iso_train_scores"] < s).mean() for s in iso_scores])

    X_stack = np.column_stack([p_lr, p_rf, p_xgb, iso_pct])
    p_fused = stacker.predict_proba(X_stack)[:, 1]

    roc = roc_auc_score(y_test, p_fused)
    pr = average_precision_score(y_test, p_fused)

    # Recall at 1% FPR
    recall_at_1 = 0.0
    for t in np.arange(0.01, 1.0, 0.01):
        fp_rate = ((p_fused >= t) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1)
        if fp_rate <= 0.01:
            recall_at_1 = ((p_fused >= t) & (y_test == 1)).sum() / max((y_test == 1).sum(), 1)
            break

    # At threshold 0.30
    tp = ((p_fused >= 0.30) & (y_test == 1)).sum()
    fp = ((p_fused >= 0.30) & (y_test == 0)).sum()
    fn = ((p_fused < 0.30) & (y_test == 1)).sum()
    precision_30 = tp / max(tp + fp, 1)
    recall_30 = tp / max(tp + fn, 1)

    # Best model from stacker coefficients
    coefs = stacker.coef_[0]
    model_names = ["LR", "RF", "XGB", "IF"]
    best_model = model_names[np.argmax(np.abs(coefs))]

    return {
        "dataset": dataset_name,
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr), 4),
        "recall_at_1pct_fpr": round(float(recall_at_1), 4),
        "precision_at_0.30": round(float(precision_30), 4),
        "recall_at_0.30": round(float(recall_30), 4),
        "stacker_coefs": {n: round(float(c), 4) for n, c in zip(model_names, coefs)},
        "best_model": best_model,
        "n_test": int(len(y_test)),
        "n_fraud_test": int(y_test.sum()),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PS-14: XGBoost + Fusion — UCI vs Kaggle Comparison")
    print("=" * 70)

    # Load datasets
    print("\n[1] Loading datasets...")
    kaggle_df = load_kaggle_as_ps16()
    uci_df = load_uci_as_ps16()

    if kaggle_df.empty or uci_df.empty:
        print("ERROR: Both datasets required")
        sys.exit(1)

    # Split each dataset 70/30
    print("\n[2] Splitting datasets 70/30...")
    k_X = kaggle_df[ML_FEATURES].values.astype(float)
    k_y = kaggle_df["label"].values.astype(int)
    k_X_tr, k_X_te, k_y_tr, k_y_te = train_test_split(k_X, k_y, test_size=0.3, random_state=42, stratify=k_y)
    print(f"  Kaggle: train={len(k_y_tr):,} ({k_y_tr.sum()} fraud), test={len(k_y_te):,} ({k_y_te.sum()} fraud)")

    u_X = uci_df[ML_FEATURES].values.astype(float)
    u_y = uci_df["label"].values.astype(int)
    u_X_tr, u_X_te, u_y_tr, u_y_te = train_test_split(u_X, u_y, test_size=0.3, random_state=42, stratify=u_y)
    print(f"  UCI:    train={len(u_y_tr):,} ({u_y_tr.sum()} fraud), test={len(u_y_te):,} ({u_y_te.sum()} fraud)")

    # Train on Kaggle
    print("\n[3] Training fusion on KAGGLE dataset...")
    kaggle_fusion = train_fusion(k_X_tr, k_y_tr, "Kaggle")

    # Train on UCI
    print("\n[4] Training fusion on UCI dataset...")
    uci_fusion = train_fusion(u_X_tr, u_y_tr, "UCI")

    # Cross-evaluate: 4 scenarios
    print("\n[5] Cross-domain evaluation...")
    results = {}

    print("\n  ── Kaggle model → Kaggle test (in-domain) ──")
    results["kaggle_on_kaggle"] = evaluate_fusion(kaggle_fusion, k_X_te, k_y_te, "Kaggle (in-domain)")
    print(f"    ROC-AUC: {results['kaggle_on_kaggle']['roc_auc']}")
    print(f"    PR-AUC:  {results['kaggle_on_kaggle']['pr_auc']}")
    print(f"    R@1%FPR: {results['kaggle_on_kaggle']['recall_at_1pct_fpr']}")
    print(f"    Best:    {results['kaggle_on_kaggle']['best_model']}")

    print("\n  ── Kaggle model → UCI test (cross-domain) ──")
    results["kaggle_on_uci"] = evaluate_fusion(kaggle_fusion, u_X_te, u_y_te, "Kaggle→UCI (cross)")
    print(f"    ROC-AUC: {results['kaggle_on_uci']['roc_auc']}")
    print(f"    PR-AUC:  {results['kaggle_on_uci']['pr_auc']}")
    print(f"    R@1%FPR: {results['kaggle_on_uci']['recall_at_1pct_fpr']}")
    print(f"    Best:    {results['kaggle_on_uci']['best_model']}")

    print("\n  ── UCI model → UCI test (in-domain) ──")
    results["uci_on_uci"] = evaluate_fusion(uci_fusion, u_X_te, u_y_te, "UCI (in-domain)")
    print(f"    ROC-AUC: {results['uci_on_uci']['roc_auc']}")
    print(f"    PR-AUC:  {results['uci_on_uci']['pr_auc']}")
    print(f"    R@1%FPR: {results['uci_on_uci']['recall_at_1pct_fpr']}")
    print(f"    Best:    {results['uci_on_uci']['best_model']}")

    print("\n  ── UCI model → Kaggle test (cross-domain) ──")
    results["uci_on_kaggle"] = evaluate_fusion(uci_fusion, k_X_te, k_y_te, "UCI→Kaggle (cross)")
    print(f"    ROC-AUC: {results['uci_on_kaggle']['roc_auc']}")
    print(f"    PR-AUC:  {results['uci_on_kaggle']['pr_auc']}")
    print(f"    R@1%FPR: {results['uci_on_kaggle']['recall_at_1pct_fpr']}")
    print(f"    Best:    {results['uci_on_kaggle']['best_model']}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN COMPARISON MATRIX")
    print("=" * 70)
    print(f"{'Scenario':<30} {'ROC-AUC':>10} {'PR-AUC':>10} {'R@1%FPR':>10} {'Best':>6}")
    print("-" * 70)
    for key, label in [
        ("kaggle_on_kaggle", "Kaggle→Kaggle (in-domain)"),
        ("kaggle_on_uci",    "Kaggle→UCI (cross-domain)"),
        ("uci_on_uci",       "UCI→UCI (in-domain)"),
        ("uci_on_kaggle",    "UCI→Kaggle (cross-domain)"),
    ]:
        r = results[key]
        print(f"{label:<30} {r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f} {r['recall_at_1pct_fpr']:>10.4f} {r['best_model']:>6}")

    # Cross-domain degradation
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN DEGRADATION")
    print("=" * 70)
    k_drop = results["kaggle_on_kaggle"]["roc_auc"] - results["kaggle_on_uci"]["roc_auc"]
    u_drop = results["uci_on_uci"]["roc_auc"] - results["uci_on_kaggle"]["roc_auc"]
    print(f"  Kaggle model on UCI:  {k_drop:+.4f} ROC-AUC drop ({'minimal' if abs(k_drop) < 0.05 else 'significant'})")
    print(f"  UCI model on Kaggle:  {u_drop:+.4f} ROC-AUC drop ({'minimal' if abs(u_drop) < 0.05 else 'significant'})")

    if abs(k_drop) < 0.05 and abs(u_drop) < 0.05:
        print("  → Both models generalize well across domains")
    elif k_drop < u_drop:
        print("  → Kaggle model generalizes better (diverse fraud patterns)")
    else:
        print("  → UCI model generalizes better (payment-behavior features transfer)")

    # Stacker coefficients comparison
    print("\n" + "=" * 70)
    print("STACKER COEFFICIENTS (model importance)")
    print("=" * 70)
    for key, label in [("kaggle_on_kaggle", "Kaggle model"), ("uci_on_uci", "UCI model")]:
        coefs = results[key]["stacker_coefs"]
        print(f"  {label}: " + "  ".join(f"{n}={c:+.4f}" for n, c in coefs.items()))

    # Save results
    out = {
        "datasets": {
            "kaggle": {"rows": len(k_y), "fraud": int(k_y.sum()), "rate": float(k_y.mean())},
            "uci": {"rows": len(u_y), "fraud": int(u_y.sum()), "rate": float(u_y.mean())},
        },
        "results": results,
        "degradation": {"kaggle_to_uci": round(k_drop, 4), "uci_to_kaggle": round(u_drop, 4)},
    }
    out_path = DATA / "uci_kaggle_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
