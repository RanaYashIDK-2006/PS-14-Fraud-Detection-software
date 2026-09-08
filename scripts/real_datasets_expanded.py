#!/usr/bin/env python3
"""Expanded real-dataset evaluation: 4 public datasets mapped to PS-16 features.

Datasets:
1. Kaggle ULB Credit Card Fraud (284K rows, 0.17% fraud)
2. UCI Credit Card Default (30K rows, 22.1% default)
3. UCI Bank Marketing (41K rows, 11.3% subscription)
4. UCI Diabetes 130-US Hospitals (101K rows, 11.2% 30-day readmission)

Each dataset is mapped to the 16-dim PS-16 feature space using domain-specific
heuristics. Models are trained on each dataset and cross-evaluated on all others.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "hour_of_day", "is_weekend",
    "shared_device_accounts", "shared_recipient_accounts", "mule_ring_score",
]


# ── Dataset loaders ──────────────────────────────────────────────────────

def load_kaggle() -> tuple[np.ndarray, np.ndarray]:
    """Kaggle Credit Card Fraud — PCA features → PS-16."""
    df = pd.read_csv(DATA / "creditcard.csv")
    n = len(df)
    X = np.column_stack([
        (df["Amount"] / max(df["Amount"].median(), 1)).clip(0, 10).values,  # amount_ratio
        np.clip((-df["V3"]).rank(pct=True) * 6, 1, 6).astype(int).values,  # txn_freq
        (df["V4"].abs() > 2).astype(int).values,  # txn_time_unusual
        (df["V14"] < -5).astype(int).values,  # new_device_flag
        (df["V8"].abs() > 2).astype(int).values,  # unusual_location
        (df["V7"].abs() > 2).astype(int).values,  # unusual_recipient
        np.clip((-df["V1"]).rank(pct=True) * 4, 0, 4).astype(int).values,  # failed_auth
        np.clip(df["Time"] / 3600, 0, 48).astype(int).values,  # days_since
        np.clip((df["V10"].abs() - 2) / 8, 0, 1).values,  # escalation
        np.clip(df["V11"].rank(pct=True) * 4, 1, 4).astype(int).values,  # known_device
        np.clip(df["V12"].rank(pct=True) * 10, 1, 10).astype(int).values,  # tenure
        (df["Time"] % (24 * 3600) / 3600).astype(int).values % 24,  # hour
        (df["V13"].abs() > 1.5).astype(int).values,  # is_weekend
        np.zeros(n, dtype=int),  # shared_device
        np.zeros(n, dtype=int),  # shared_recipient
        np.zeros(n),  # mule_ring
    ])
    y = df["Class"].values.astype(int)
    print(f"  Kaggle: {n:,} rows, fraud={y.sum()} ({y.mean()*100:.3f}%)")
    return X, y


def load_uci_default() -> tuple[np.ndarray, np.ndarray]:
    """UCI Credit Card Default — already mapped to PS-16."""
    df = pd.read_csv(DATA / "validation_uci_default.csv")
    X = df[ML_FEATURES].values.astype(float)
    y = df["label"].values.astype(int)
    print(f"  UCI Default: {len(y):,} rows, default={y.sum()} ({y.mean()*100:.2f}%)")
    return X, y


def load_bank_marketing() -> tuple[np.ndarray, np.ndarray]:
    """UCI Bank Marketing — map to PS-16 features."""
    df = pd.read_csv(DATA / "bank_raw/bank-additional/bank-additional-full.csv", sep=";")
    n = len(df)

    # Encode categoricals
    job_map = {j: i for i, j in enumerate(df["job"].unique())}
    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    contact_map = {"telephone": 0, "cellular": 1}
    pout_map = {"nonexistent": 0, "failure": 1, "success": 2}

    X = np.column_stack([
        np.clip(df["duration"] / max(df["duration"].median(), 1), 0, 10).values,  # amount_ratio (duration proxy)
        np.clip(df["campaign"], 1, 6).values,  # txn_freq (campaign contacts)
        (df["day_of_week"].isin(["fri", "mon"])).astype(int).values,  # txn_time_unusual (start of week)
        (df["pdays"] == 999).astype(int).values,  # new_device_flag (not previously contacted)
        (df["emp.var.rate"] < -1).astype(int).values,  # unusual_location (economic stress)
        (df["previous"] > 0).astype(int).values,  # unusual_recipient (prior campaign contact)
        np.clip(df["previous"], 0, 4).values,  # failed_auth (prior rejections)
        np.clip(df["pdays"], 0, 48).values,  # days_since_last_similar
        np.clip(df["campaign"].rank(pct=True), 0, 1).values,  # escalation
        df["job"].map(job_map).fillna(0).values.astype(int),  # known_device (job category)
        np.clip(df["age"] / 10, 1, 8).astype(int).values,  # tenure (age proxy)
        df["month"].map(month_map).fillna(5).values % 24,  # hour (month as proxy)
        (df["day_of_week"].isin(["sat", "sun"])).astype(int).values,  # weekend
        np.zeros(n, dtype=int),  # shared_device
        np.zeros(n, dtype=int),  # shared_recipient
        np.zeros(n),  # mule_ring
    ])
    y = (df["y"] == "yes").astype(int).values
    print(f"  Bank Marketing: {n:,} rows, subscribe={y.sum()} ({y.mean()*100:.2f}%)")
    return X, y


def load_diabetes() -> tuple[np.ndarray, np.ndarray]:
    """UCI Diabetes 130-US Hospitals — map to PS-16 features."""
    df = pd.read_csv(DATA / "diabetes_raw/diabetic_data.csv", na_values="?", low_memory=False)
    n = len(df)

    # Encode key features
    age_map = {"[0-10)": 1, "[10-20)": 2, "[20-30)": 3, "[30-40)": 4, "[40-50)": 5,
               "[50-60)": 6, "[60-70)": 7, "[70-80)": 8, "[80-90)": 9, "[90-100)": 10}

    X = np.column_stack([
        np.clip(df["num_medications"] / max(df["num_medications"].median(), 1), 0, 10).values,  # amount_ratio
        np.clip(df["num_lab_procedures"] / 10, 0, 6).astype(int).values,  # txn_freq
        (df["time_in_hospital"] > 7).astype(int).values,  # txn_time_unusual (long stay)
        (df["number_inpatient"] > 0).astype(int).values,  # new_device_flag (prior inpatient)
        (df["number_emergency"] > 0).astype(int).values,  # unusual_location (ER visits)
        (df["num_procedures"] > 3).astype(int).values,  # unusual_recipient
        np.clip(df["number_inpatient"], 0, 4).values,  # failed_auth
        np.clip(df["number_outpatient"], 0, 48).values,  # days_since
        np.clip(df["num_medications"].rank(pct=True), 0, 1).values,  # escalation
        df["age"].map(age_map).fillna(5).values.astype(int),  # known_device (age tier)
        df["age"].map(age_map).fillna(5).values.astype(int),  # tenure (same)
        np.zeros(n, dtype=int),  # hour (no time data)
        (df["discharge_disposition_id"].isin([1, 6, 8])).astype(int).values,  # weekend (discharge to home/transfer)
        np.zeros(n, dtype=int),  # shared_device
        np.zeros(n, dtype=int),  # shared_recipient
        np.zeros(n),  # mule_ring
    ])
    y = (df["readmitted"] == "<30").astype(int).values
    print(f"  Diabetes: {n:,} rows, readmit_30d={y.sum()} ({y.mean()*100:.2f}%)")
    return X, y


# ── Training ─────────────────────────────────────────────────────────────

def train_fusion(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Train 4-model fusion + stacker."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(Xs, y)
    p_lr = lr.predict_proba(Xs)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X, y)
    p_rf = rf.predict_proba(X)[:, 1]

    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb.fit(X, y)
        p_xgb = xgb.predict_proba(X)[:, 1]
    except ImportError:
        xgb = None
        p_xgb = np.full(len(y), 0.5)

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(Xs)
    iso_scores = -iso.score_samples(Xs)
    iso_pct = np.array([(iso_scores < s).mean() for s in iso_scores])

    X_stack = np.column_stack([p_lr, p_rf, p_xgb, iso_pct])
    stacker = LogisticRegression(max_iter=1000, random_state=42)
    stacker.fit(X_stack, y)
    p_fused = stacker.predict_proba(X_stack)[:, 1]

    train_auc = roc_auc_score(y, p_fused)
    coefs = stacker.coef_[0]
    print(f"  {name} train: ROC-AUC={train_auc:.4f}  coefs={dict(zip(['LR','RF','XGB','IF'], [f'{c:+.3f}' for c in coefs]))}")

    return {
        "lr": lr, "rf": rf, "xgb": xgb, "iso": iso,
        "stacker": stacker, "scaler": scaler,
        "iso_train_scores": iso_scores,
    }


def evaluate(fusion: dict, X: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Evaluate fusion on a test set."""
    lr, rf, xgb, iso, stacker, scaler = (
        fusion["lr"], fusion["rf"], fusion["xgb"],
        fusion["iso"], fusion["stacker"], fusion["scaler"],
    )
    Xs = scaler.transform(X)
    p_lr = lr.predict_proba(Xs)[:, 1]
    p_rf = rf.predict_proba(X)[:, 1]
    p_xgb = xgb.predict_proba(X)[:, 1] if xgb is not None else np.full(len(y), 0.5)
    iso_scores = -iso.score_samples(Xs)
    iso_pct = np.array([(fusion["iso_train_scores"] < s).mean() for s in iso_scores])

    X_stack = np.column_stack([p_lr, p_rf, p_xgb, iso_pct])
    p_fused = stacker.predict_proba(X_stack)[:, 1]

    roc = roc_auc_score(y, p_fused)
    pr = average_precision_score(y, p_fused)

    # Recall at 1% FPR
    recall_at_1 = 0.0
    for t in np.arange(0.01, 1.0, 0.005):
        fp_rate = ((p_fused >= t) & (y == 0)).sum() / max((y == 0).sum(), 1)
        if fp_rate <= 0.01:
            recall_at_1 = ((p_fused >= t) & (y == 1)).sum() / max((y == 1).sum(), 1)
            break

    # Precision/recall at 0.30
    tp = ((p_fused >= 0.30) & (y == 1)).sum()
    fp = ((p_fused >= 0.30) & (y == 0)).sum()
    fn = ((p_fused < 0.30) & (y == 1)).sum()

    return {
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr), 4),
        "recall_at_1pct_fpr": round(float(recall_at_1), 4),
        "precision_at_0.30": round(float(tp / max(tp + fp, 1)), 4),
        "recall_at_0.30": round(float(tp / max(tp + fn, 1)), 4),
        "n_test": int(len(y)),
        "n_fraud": int(y.sum()),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("PS-14: Expanded Real-Dataset Evaluation (4 datasets)")
    print("=" * 72)

    # Load all datasets
    print("\n[1] Loading datasets...")
    datasets = {
        "Kaggle CC Fraud": load_kaggle(),
        "UCI Default": load_uci_default(),
        "Bank Marketing": load_bank_marketing(),
        "Diabetes Readmit": load_diabetes(),
    }

    # Split 70/30
    print("\n[2] Splitting 70/30...")
    splits = {}
    for name, (X, y) in datasets.items():
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42,
                                                     stratify=y if y.sum() > 10 else None)
        splits[name] = (X_tr, X_te, y_tr, y_te)
        print(f"  {name}: train={len(y_tr):,} ({y_tr.sum()} pos), test={len(y_te):,} ({y_te.sum()} pos)")

    # Train on each dataset
    print("\n[3] Training models...")
    fusions = {}
    for name in datasets:
        X_tr, _, y_tr, _ = splits[name]
        fusions[name] = train_fusion(X_tr, y_tr, name)

    # Cross-evaluate: 4×4 matrix
    print("\n[4] Cross-domain evaluation (4×4 matrix)...")
    names = list(datasets.keys())
    matrix = {}

    for train_name in names:
        for test_name in names:
            X_tr, X_te, y_tr, y_te = splits[test_name]
            tag = f"{train_name} → {test_name}"
            in_domain = train_name == test_name
            r = evaluate(fusions[train_name], X_te, y_te, tag)
            matrix[tag] = r
            marker = "" if in_domain else "  [cross]"
            print(f"  {tag:<35} ROC={r['roc_auc']:.4f}  PR={r['pr_auc']:.4f}  R@1%={r['recall_at_1pct_fpr']:.4f}{marker}")

    # Summary table
    print("\n" + "=" * 72)
    print("CROSS-DOMAIN MATRIX (ROC-AUC)")
    print("=" * 72)
    header = f"{'Train \\ Test':<25}" + "".join(f"{n[:12]:>13}" for n in names)
    print(header)
    print("-" * len(header))
    for train_name in names:
        row = f"{train_name:<25}"
        for test_name in names:
            tag = f"{train_name} → {test_name}"
            val = matrix[tag]["roc_auc"]
            marker = "*" if train_name == test_name else " "
            row += f"{val:>12.4f}{marker}"
        print(row)

    # Degradation analysis
    print("\n" + "=" * 72)
    print("CROSS-DOMAIN DEGRADATION (ROC-AUC drop from in-domain)")
    print("=" * 72)
    for train_name in names:
        in_domain = matrix[f"{train_name} → {train_name}"]["roc_auc"]
        for test_name in names:
            if train_name != test_name:
                cross = matrix[f"{train_name} → {test_name}"]["roc_auc"]
                drop = in_domain - cross
                print(f"  {train_name} → {test_name:<20} {drop:+.4f} ({'ok' if abs(drop) < 0.1 else 'LARGE'})")

    # Best model per test domain
    print("\n" + "=" * 72)
    print("BEST TRAINING DOMAIN PER TEST DOMAIN")
    print("=" * 72)
    for test_name in names:
        best_auc = 0
        best_train = ""
        for train_name in names:
            auc = matrix[f"{train_name} → {test_name}"]["roc_auc"]
            if auc > best_auc:
                best_auc = auc
                best_train = train_name
        in_domain = matrix[f"{test_name} → {test_name}"]["roc_auc"]
        print(f"  {test_name:<25} best={best_train:<20} AUC={best_auc:.4f} (in-domain={in_domain:.4f})")

    # Save
    out = {
        "datasets": {name: {"rows": len(y), "positive": int(y.sum()), "rate": round(float(y.mean()), 4)}
                     for name, (X, y) in datasets.items()},
        "matrix": matrix,
    }
    out_path = DATA / "expanded_real_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
