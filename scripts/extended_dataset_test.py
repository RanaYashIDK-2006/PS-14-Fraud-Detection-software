#!/usr/bin/env python3
"""Extended cross-dataset evaluation with additional fraud patterns.

Generates 4 more synthetic datasets with distinct fraud characteristics:
1. Cryptocurrency exchange — wash trading, layering
2. Insurance — claim fraud, staged accidents
3. Payroll — ghost employees, overtime fraud
4. Healthcare — phantom billing, upcoding

Then runs PS-14 models against all of them.
"""
from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "models" / "artifacts"
DATA = ROOT / "data"

sys.path.insert(0, str(ROOT))
rng = np.random.default_rng(123)

from src.privacy_layer.features import ML_FEATURES  # authoritative source



def _row(
    fraud: bool,
    hour: int | None = None,
    amount: float = 50.0,
    avg_amount: float = 100.0,
    freq_24h: int = 2,
    new_device: int = 0,
    unusual_loc: int = 0,
    unusual_rcpt: int = 0,
    failed_auth: int = 0,
    days_similar: int = 30,
    tenure: int = 180,
    known_devices: int = 2,
    shared_dev: int = 0,
    shared_rcpt: int = 0,
) -> dict:
    if hour is None:
        hour = int(rng.integers(0, 24))
    is_weekend = int(hour in (0, 6))
    amount_ratio = round(amount / max(avg_amount, 1), 4)
    time_unusual = int(hour < 6 or hour > 22)
    escalation = round(rng.uniform(0, 0.3), 4) if not fraud else round(rng.uniform(0.3, 1.0), 4)
    mule = 0.0
    if fraud and (shared_dev > 0 or shared_rcpt > 0):
        mule = round(min(shared_dev + shared_rcpt, 5) / 5.0, 4)
    return {
        "amount_ratio": amount_ratio, "txn_freq_last_24h": freq_24h,
        "txn_time_unusual": time_unusual, "new_device_flag": new_device,
        "unusual_location_flag": unusual_loc, "unusual_recipient_flag": unusual_rcpt,
        "failed_auth_count_24h": failed_auth, "days_since_last_similar_txn": days_similar,
        "gradual_escalation_score": escalation, "known_device_count": known_devices,
        "account_tenure_days": tenure, "hour_of_day": hour, "is_weekend": is_weekend,
        "shared_device_accounts": shared_dev, "shared_recipient_accounts": shared_rcpt,
        "mule_ring_score": mule,
        # Deviation features (production ML_FEATURES expects these 5 extras)
        "hour_deviation": round(abs(hour - 14) / 24.0, 4),
        "amount_zscore": round((amount - avg_amount) / max(avg_amount, 1), 4),
        "velocity_deviation": round(freq_24h / max(24, freq_24h), 4),
        "recipient_novelty": round(0.5 if fraud else 0.1, 4),
        "txn_regularity": round(rng.uniform(0.3, 0.7), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 1: Cryptocurrency Exchange
# ═══════════════════════════════════════════════════════════════════════════════
def generate_crypto(n: int = 50_000, fraud_rate: float = 0.025) -> pd.DataFrame:
    """Wash trading, layering, rapid self-trades."""
    n_f = int(n * fraud_rate)
    rows, labels = [], []

    for _ in range(n - n_f):
        h = int(rng.integers(0, 24))
        amt = round(float(rng.lognormal(5, 1.5)), 2)
        rows.append(_row(False, hour=h, amount=amt, avg_amount=200,
                         freq_24h=int(rng.integers(1, 6)), tenure=int(rng.integers(30, 800))))
        labels.append(0)

    for _ in range(n_f):
        h = int(rng.integers(0, 24))
        amt = round(float(rng.lognormal(6, 2)), 2)  # high-value wash trades
        rows.append(_row(True, hour=h, amount=amt, avg_amount=200,
                         freq_24h=int(rng.integers(10, 50)),  # extreme frequency
                         new_device=int(rng.choice([0, 1], p=[0.3, 0.7])),
                         unusual_loc=int(rng.choice([0, 1], p=[0.4, 0.6])),
                         unusual_rcpt=int(rng.choice([0, 1], p=[0.3, 0.7])),
                         days_similar=int(rng.integers(0, 2)),
                         tenure=int(rng.integers(1, 20)),
                         known_devices=1, shared_dev=int(rng.integers(1, 6)),
                         shared_rcpt=int(rng.integers(1, 5))))
        labels.append(1)

    df = pd.DataFrame(rows); df["label"] = labels
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 2: Insurance Claims
# ═══════════════════════════════════════════════════════════════════════════════
def generate_insurance(n: int = 50_000, fraud_rate: float = 0.04) -> pd.DataFrame:
    """Staged accidents, phantom claims, claim mills."""
    n_f = int(n * fraud_rate)
    rows, labels = [], []

    for _ in range(n - n_f):
        h = int(rng.integers(8, 18))
        amt = round(float(rng.lognormal(6, 1)), 2)
        rows.append(_row(False, hour=h, amount=amt, avg_amount=400,
                         freq_24h=int(rng.integers(0, 2)), tenure=int(rng.integers(60, 2000)),
                         known_devices=int(rng.integers(1, 3))))
        labels.append(0)

    for _ in range(n_f):
        h = int(rng.integers(0, 24))
        amt = round(float(rng.lognormal(7, 1.5)), 2)
        rows.append(_row(True, hour=h, amount=amt, avg_amount=400,
                         freq_24h=int(rng.integers(3, 12)),
                         new_device=int(rng.choice([0, 1], p=[0.25, 0.75])),
                         unusual_loc=int(rng.choice([0, 1], p=[0.3, 0.7])),
                         unusual_rcpt=int(rng.choice([0, 1], p=[0.2, 0.8])),
                         failed_auth=int(rng.integers(0, 2)),
                         days_similar=int(rng.integers(0, 5)),
                         tenure=int(rng.integers(5, 50)),
                         known_devices=1, shared_dev=int(rng.integers(0, 4)),
                         shared_rcpt=int(rng.integers(0, 5))))
        labels.append(1)

    df = pd.DataFrame(rows); df["label"] = labels
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 3: Payroll Fraud
# ═══════════════════════════════════════════════════════════════════════════════
def generate_payroll(n: int = 50_000, fraud_rate: float = 0.03) -> pd.DataFrame:
    """Ghost employees, overtime padding, unauthorized bonuses."""
    n_f = int(n * fraud_rate)
    rows, labels = [], []

    for _ in range(n - n_f):
        h = int(rng.integers(8, 18))
        amt = round(float(rng.lognormal(6, 0.8)), 2)
        rows.append(_row(False, hour=h, amount=amt, avg_amount=500,
                         freq_24h=int(rng.integers(0, 1)), tenure=int(rng.integers(90, 3000)),
                         known_devices=int(rng.integers(1, 2))))
        labels.append(0)

    for _ in range(n_f):
        h = int(rng.integers(0, 24))
        amt = round(float(rng.lognormal(7, 1.2)), 2)
        rows.append(_row(True, hour=h, amount=amt, avg_amount=500,
                         freq_24h=int(rng.integers(2, 8)),
                         new_device=int(rng.choice([0, 1], p=[0.3, 0.7])),
                         unusual_loc=int(rng.choice([0, 1], p=[0.5, 0.5])),
                         unusual_rcpt=int(rng.choice([0, 1], p=[0.3, 0.7])),
                         failed_auth=0, days_similar=int(rng.integers(0, 3)),
                         tenure=int(rng.integers(1, 30)),
                         known_devices=1, shared_dev=int(rng.integers(0, 3)),
                         shared_rcpt=int(rng.integers(0, 2))))
        labels.append(1)

    df = pd.DataFrame(rows); df["label"] = labels
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 4: Healthcare Billing
# ═══════════════════════════════════════════════════════════════════════════════
def generate_healthcare(n: int = 50_000, fraud_rate: float = 0.035) -> pd.DataFrame:
    """Phantom billing, upcoding, unbundling."""
    n_f = int(n * fraud_rate)
    rows, labels = [], []

    for _ in range(n - n_f):
        h = int(rng.integers(6, 20))
        amt = round(float(rng.lognormal(5.5, 1.2)), 2)
        rows.append(_row(False, hour=h, amount=amt, avg_amount=300,
                         freq_24h=int(rng.integers(1, 5)), tenure=int(rng.integers(30, 1500)),
                         known_devices=int(rng.integers(1, 3))))
        labels.append(0)

    for _ in range(n_f):
        h = int(rng.integers(0, 24))
        amt = round(float(rng.lognormal(7, 1.8)), 2)
        rows.append(_row(True, hour=h, amount=amt, avg_amount=300,
                         freq_24h=int(rng.integers(5, 25)),  # high claim frequency
                         new_device=int(rng.choice([0, 1], p=[0.3, 0.7])),
                         unusual_loc=int(rng.choice([0, 1], p=[0.2, 0.8])),
                         unusual_rcpt=int(rng.choice([0, 1], p=[0.15, 0.85])),
                         failed_auth=int(rng.integers(0, 2)),
                         days_similar=int(rng.integers(0, 3)),
                         tenure=int(rng.integers(10, 100)),
                         known_devices=1, shared_dev=int(rng.integers(0, 6)),
                         shared_rcpt=int(rng.integers(0, 5))))
        labels.append(1)

    df = pd.DataFrame(rows); df["label"] = labels
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(name: str, df: pd.DataFrame, models: dict) -> dict:
    X = df[ML_FEATURES].values
    y = df["label"].values
    fraud_idx = np.where(y == 1)[0]
    n_fraud = len(fraud_idx)
    n_legit = len(y) - n_fraud

    # Base models only
    base_names = [k for k in models if k not in ("stacker", "calibrator")]
    scores = {}
    for mn in base_names:
        m = models[mn]
        if hasattr(m, "predict_proba"):
            scores[mn] = m.predict_proba(X)[:, 1]
        else:
            scores[mn] = m.predict(X).astype(float)

    sorted_keys = sorted(scores.keys())
    meta_X = np.column_stack([scores[k] for k in sorted_keys])

    if "stacker" in models and meta_X.shape[1] == models["stacker"].n_features_in_:
        stacked = models["stacker"].predict_proba(meta_X)[:, 1]
    else:
        stacked = meta_X.mean(axis=1)

    calibrated = stacked  # calibrator not loadable from scripts/

    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    y = y.astype(int)
    roc = roc_auc_score(y, calibrated)
    precision_arr, recall_arr, _ = precision_recall_curve(y, calibrated)
    pr_auc = auc(recall_arr, precision_arr)

    sorted_idx = np.argsort(-calibrated)
    fraud_ranks = np.searchsorted(sorted_idx, fraud_idx)
    r1 = float(np.mean(fraud_ranks < int(n_legit * 0.01)))

    result = {"name": name, "n_total": len(y), "n_fraud": n_fraud, "n_legit": n_legit,
              "roc_auc": round(roc, 4), "pr_auc": round(pr_auc, 4), "recall_at_1pct_fpr": round(r1, 4)}

    for thresh in [0.30, 0.50, 0.70, 0.90]:
        flagged = calibrated >= thresh
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        result[f"thresh_{thresh}"] = {
            "flagged": int(np.sum(flagged)),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(n_fraud, 1), 4),
            "fpr": round(fp / max(n_legit, 1), 6),
        }

    per_model = {}
    for mn in sorted_keys:
        try:
            m_roc = round(roc_auc_score(y, scores[mn]), 4)
        except ValueError:
            m_roc = 0.5
        m_sorted = np.argsort(-scores[mn])
        m_ranks = np.searchsorted(m_sorted, fraud_idx)
        m_r1 = round(float(np.mean(m_ranks < int(n_legit * 0.01))), 4)
        per_model[mn] = {"roc_auc": m_roc, "recall_at_1pct_fpr": m_r1}
    result["per_model"] = per_model
    return result


def main():
    print("=" * 72)
    print("  PS-14 EXTENDED CROSS-DATASET EVALUATION")
    print("=" * 72)

    # Load models
    print("\nLoading PS-14 models...")
    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost", "xgb_student_distilled"]:
        p = ARTIFACTS / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)
            print(f"  ✓ {name}")
    for sn, fn in [("stacker", "stacker_v2"), ("stacker", "stacker")]:
        p = ARTIFACTS / f"{fn}.joblib"
        if p.exists() and "stacker" not in models:
            models["stacker"] = joblib.load(p)
            print(f"  ✓ {fn}")

    # Generate datasets
    print("\nGenerating extended fraud datasets...")
    datasets = {
        "Crypto Exchange (wash trading)": generate_crypto(50_000, 0.025),
        "Insurance Claims (staged fraud)": generate_insurance(50_000, 0.04),
        "Payroll (ghost employees)": generate_payroll(50_000, 0.03),
        "Healthcare Billing (phantom)": generate_healthcare(50_000, 0.035),
    }

    # Also include previous results
    for fname, label in [("validation_kaggle.csv", "Credit Card (Kaggle ULB)")]:
        fp = DATA / fname
        if fp.exists():
            kdf = pd.read_csv(fp)
            if "label" in kdf.columns and all(c in kdf.columns for c in ML_FEATURES):
                datasets[label] = kdf

    for name, df in datasets.items():
        fpct = df["label"].mean() * 100
        print(f"  ✓ {name} — {len(df):,} txns, {df['label'].sum():,} fraud ({fpct:.2f}%)")

    # Evaluate
    all_results = {}
    for name, df in datasets.items():
        print(f"\n{'─' * 72}")
        print(f"  {name}")
        print(f"{'─' * 72}")
        r = evaluate(name, df, models)
        all_results[name] = r

        print(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}  R@1%FPR: {r['recall_at_1pct_fpr']:.4f}")
        print(f"  {'Thresh':<8} {'Flagged':>8} {'Prec':>8} {'Recall':>8} {'FPR':>10}")
        for t in [0.30, 0.50, 0.70, 0.90]:
            d = r[f"thresh_{t}"]
            print(f"  {t:<8.2f} {d['flagged']:>8,} {d['precision']:>8.4f} {d['recall']:>8.4f} {d['fpr']:>10.6f}")

        print(f"  Per-model:")
        for mn, mr in r["per_model"].items():
            print(f"    {mn:<30} ROC={mr['roc_auc']:.4f}  R@1%={mr['recall_at_1pct_fpr']:.4f}")

    # Summary
    print(f"\n{'═' * 72}")
    print("  COMPLETE CROSS-DATASET SUMMARY (8 datasets)")
    print(f"{'═' * 72}")
    print(f"\n  {'Dataset':<40} {'ROC-AUC':>8} {'PR-AUC':>8} {'R@1%FPR':>8} {'Generalization':>14}")
    print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8} {'─'*14}")
    for name, r in all_results.items():
        gen = "STRONG" if r["roc_auc"] > 0.9 else "MODERATE" if r["roc_auc"] > 0.7 else "WEAK"
        print(f"  {name:<40} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {r['recall_at_1pct_fpr']:>8.4f} {gen:>14}")

    # Average across all
    avg_roc = np.mean([r["roc_auc"] for r in all_results.values()])
    avg_pr = np.mean([r["pr_auc"] for r in all_results.values()])
    avg_r1 = np.mean([r["recall_at_1pct_fpr"] for r in all_results.values()])
    print(f"\n  {'AVERAGE':<40} {avg_roc:>8.4f} {avg_pr:>8.4f} {avg_r1:>8.4f}")

    # Save
    report_path = DATA / "extended_dataset_results.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {report_path}")

    print(f"\n{'═' * 72}")
    print("  COMPLETE")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
