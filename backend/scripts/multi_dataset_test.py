#!/usr/bin/env python3
"""Cross-dataset fraud detection evaluation.

Generates 3 synthetic datasets mimicking different fraud domains:
1. Mobile money (PaySim-style) — account-to-account transfers
2. E-commerce (IEEE-CIS-style) — card-not-present transactions
3. Banking (SWIFT-style) — high-value wire transfers

Runs PS-14's trained models against each and reports metrics.
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
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
ARTIFACTS = ROOT / "models" / "artifacts"
DATA = ROOT / "data"

# ── RNG ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / "backend"))
rng = np.random.default_rng(42)

# ── PS-14 feature columns (matching src/privacy_layer/features.py ML_FEATURES) ──
from src.privacy_layer.features import ML_FEATURES as PS16_COLS



def _ps16_row(
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
    shared_device_accts: int = 0,
    shared_rcpt_accts: int = 0,
) -> dict:
    """Build a §16 feature row (16 features matching ML_FEATURES)."""
    if hour is None:
        hour = rng.integers(0, 24)
    is_weekend = int(hour in (0, 6))
    amount_ratio = round(amount / max(avg_amount, 1), 4)
    time_unusual = int(hour < 6 or hour > 22)
    escalation = round(rng.uniform(0, 0.3), 4) if not fraud else round(rng.uniform(0.3, 1.0), 4)
    # Mule ring score: 0 for normal, elevated for fraud with shared signals
    mule_score = 0.0
    if fraud and (shared_device_accts > 0 or shared_rcpt_accts > 0):
        mule_score = round(min(shared_device_accts + shared_rcpt_accts, 5) / 5.0, 4)
    return {
        "amount_ratio": amount_ratio,
        "txn_freq_last_24h": freq_24h,
        "txn_time_unusual": time_unusual,
        "new_device_flag": new_device,
        "unusual_location_flag": unusual_loc,
        "unusual_recipient_flag": unusual_rcpt,
        "failed_auth_count_24h": failed_auth,
        "days_since_last_similar_txn": days_similar,
        "gradual_escalation_score": escalation,
        "known_device_count": known_devices,
        "account_tenure_days": tenure,
        "hour_of_day": hour,
        "is_weekend": is_weekend,
        "shared_device_accounts": shared_device_accts,
        "shared_recipient_accounts": shared_rcpt_accts,
        "mule_ring_score": mule_score,
        # Deviation features (production ML_FEATURES expects these 5 extras)
        "hour_deviation": round(abs(hour - 14) / 24.0, 4),
        "amount_zscore": round((amount - avg_amount) / max(avg_amount, 1), 4),
        "velocity_deviation": round(freq_24h / max(24, freq_24h), 4),
        "recipient_novelty": round(0.5 if fraud else 0.1, 4),
        "txn_regularity": round(rng.uniform(0.3, 0.7), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 1: Mobile Money (PaySim-style)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_mobile_money(n: int = 50_000, fraud_rate: float = 0.02) -> pd.DataFrame:
    """Simulate mobile money transfers (agent-to-agent, P2P, cash-in/out).

    Fraud patterns:
    - Rapid cash-out after cash-in (money laundering)
    - Transfer to new recipient at night
    - Agent manipulation (transfer to self)
    """
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    rows = []
    labels = []

    for _ in range(n_legit):
        hour = int(rng.integers(6, 22))
        amount = round(float(rng.lognormal(4, 1)), 2)  # median ~$55
        rows.append(_ps16_row(
            fraud=False, hour=hour, amount=amount, avg_amount=80,
            freq_24h=int(rng.integers(1, 5)), tenure=int(rng.integers(30, 1000)),
        ))
        labels.append(0)

    for _ in range(n_fraud):
        hour = int(rng.integers(0, 24))
        amount = round(float(rng.lognormal(5, 1.5)), 2)  # higher amounts
        rows.append(_ps16_row(
            fraud=True, hour=hour, amount=amount, avg_amount=80,
            freq_24h=int(rng.integers(3, 15)),  # rapid burst
            new_device=int(rng.choice([0, 1], p=[0.3, 0.7])),
            unusual_loc=int(rng.choice([0, 1], p=[0.2, 0.8])),
            unusual_rcpt=int(rng.choice([0, 1], p=[0.15, 0.85])),
            failed_auth=int(rng.integers(0, 3)),
            days_similar=int(rng.integers(0, 3)),
            tenure=int(rng.integers(1, 30)),  # new accounts
            known_devices=1,
            shared_device_accts=int(rng.integers(0, 4)),
            shared_rcpt_accts=int(rng.integers(0, 3)),
        ))
        labels.append(1)

    df = pd.DataFrame(rows)
    df["label"] = labels
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 2: E-commerce (IEEE-CIS-style)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_ecommerce(n: int = 50_000, fraud_rate: float = 0.03) -> pd.DataFrame:
    """Simulate card-not-present e-commerce transactions.

    Fraud patterns:
    - High-value purchase on new device
    - Multiple rapid purchases (card testing)
    - Unusual time/location combo
    - Account takeover (new device + high amount)
    """
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    rows = []
    labels = []

    for _ in range(n_legit):
        hour = int(rng.integers(7, 23))
        amount = round(float(rng.lognormal(3.5, 0.8)), 2)  # median ~$33
        rows.append(_ps16_row(
            fraud=False, hour=hour, amount=amount, avg_amount=40,
            freq_24h=int(rng.integers(1, 4)),
            tenure=int(rng.integers(60, 2000)),
            known_devices=int(rng.integers(1, 4)),
        ))
        labels.append(0)

    for _ in range(n_fraud):
        fraud_type = rng.choice(["card_test", "ato", "high_value", "rapid_fire"])
        if fraud_type == "card_test":
            # Micro-transactions to test stolen card
            amount = round(float(rng.uniform(0.5, 5.0)), 2)
            hour = int(rng.integers(0, 24))
            freq = int(rng.integers(5, 20))
        elif fraud_type == "ato":
            # Account takeover: new device, high amount
            amount = round(float(rng.lognormal(5, 1.5)), 2)
            hour = int(rng.integers(0, 24))
            freq = 1
        elif fraud_type == "high_value":
            # Expensive single purchase
            amount = round(float(rng.lognormal(7, 1)), 2)
            hour = int(rng.integers(0, 24))
            freq = 1
        else:  # rapid_fire
            # Multiple rapid purchases
            amount = round(float(rng.lognormal(4, 1)), 2)
            hour = int(rng.integers(0, 24))
            freq = int(rng.integers(10, 30))

        rows.append(_ps16_row(
            fraud=True, hour=hour, amount=amount, avg_amount=40,
            freq_24h=freq,
            new_device=int(rng.choice([0, 1], p=[0.2, 0.8])),
            unusual_loc=int(rng.choice([0, 1], p=[0.25, 0.75])),
            unusual_rcpt=int(rng.choice([0, 1], p=[0.2, 0.8])),
            failed_auth=int(rng.integers(0, 4)),
            days_similar=int(rng.integers(0, 5)),
            tenure=int(rng.integers(1, 60)),
            known_devices=1,
            shared_device_accts=int(rng.integers(0, 5)),
            shared_rcpt_accts=int(rng.integers(0, 4)),
        ))
        labels.append(1)

    df = pd.DataFrame(rows)
    df["label"] = labels
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 3: Banking (SWIFT-style)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_banking(n: int = 50_000, fraud_rate: float = 0.015) -> pd.DataFrame:
    """Simulate banking wire transfers.

    Fraud patterns:
    - High-value transfer to new recipient
    - Off-hours transfer from new device
    - Gradual escalation (increase amounts over days)
    - First-time international transfer
    """
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    rows = []
    labels = []

    for _ in range(n_legit):
        hour = int(rng.integers(8, 18))
        amount = round(float(rng.lognormal(6, 1.5)), 2)  # median ~$400
        rows.append(_ps16_row(
            fraud=False, hour=hour, amount=amount, avg_amount=500,
            freq_24h=int(rng.integers(0, 3)),
            tenure=int(rng.integers(90, 5000)),
            known_devices=int(rng.integers(1, 3)),
        ))
        labels.append(0)

    for _ in range(n_fraud):
        fraud_type = rng.choice(["high_value", "escalation", "off_hours"])
        if fraud_type == "high_value":
            amount = round(float(rng.lognormal(9, 1.5)), 2)  # $5K-$50K
            hour = int(rng.integers(8, 18))
        elif fraud_type == "escalation":
            amount = round(float(rng.lognormal(7, 1)), 2)  # growing amounts
            hour = int(rng.integers(8, 18))
        else:  # off_hours
            amount = round(float(rng.lognormal(7, 1.5)), 2)
            hour = int(rng.integers(0, 6))

        rows.append(_ps16_row(
            fraud=True, hour=hour, amount=amount, avg_amount=500,
            freq_24h=int(rng.integers(2, 8)),
            new_device=int(rng.choice([0, 1], p=[0.3, 0.7])),
            unusual_loc=int(rng.choice([0, 1], p=[0.3, 0.7])),
            unusual_rcpt=int(rng.choice([0, 1], p=[0.2, 0.8])),
            failed_auth=int(rng.integers(0, 2)),
            days_similar=int(rng.integers(0, 10)),
            tenure=int(rng.integers(10, 100)),
            known_devices=1,
            shared_device_accts=int(rng.integers(0, 3)),
            shared_rcpt_accts=int(rng.integers(0, 3)),
        ))
        labels.append(1)

    df = pd.DataFrame(rows)
    df["label"] = labels
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(name: str, df: pd.DataFrame, models: dict) -> dict:
    """Run all PS-14 models against a dataset and compute metrics."""
    X = df[PS16_COLS].values
    y = df["label"].values
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    n_fraud = len(fraud_idx)
    n_legit = len(legit_idx)

    # Base models only (not stacker or calibrator)
    base_model_names = [k for k in models if k not in ("stacker", "calibrator")]
    scores = {}
    for model_name in base_model_names:
        model = models[model_name]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = model.predict(X).astype(float)
        scores[model_name] = proba

    # Stacked score (meta_X columns must match stacker's training order)
    # The stacker was trained on: logistic_regression, random_forest, xgboost, isolation_forest
    # but we may not have all — sort by name for consistent ordering
    sorted_keys = sorted(scores.keys())
    meta_X = np.column_stack([scores[k] for k in sorted_keys])
    if "stacker" in models and meta_X.shape[1] == models["stacker"].n_features_in_:
        stacked = models["stacker"].predict_proba(meta_X)[:, 1]
    else:
        stacked = meta_X.mean(axis=1)

    # Calibrated
    if "calibrator" in models:
        calibrated = models["calibrator"].predict(stacked.reshape(-1, 1))
    else:
        calibrated = stacked

    # Metrics for calibrated score
    sorted_idx = np.argsort(-calibrated)
    fraud_ranks = np.searchsorted(sorted_idx, fraud_idx)
    recall_at_1pct = np.mean(fraud_ranks < int(n_legit * 0.01))

    # ROC-AUC
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    roc = roc_auc_score(y, calibrated)
    precision, recall, thresholds = precision_recall_curve(y, calibrated)
    pr_auc = auc(recall, precision)

    # Threshold analysis
    results = {"name": name, "n_total": len(y), "n_fraud": n_fraud, "n_legit": n_legit,
               "roc_auc": round(roc, 4), "pr_auc": round(pr_auc, 4),
               "recall_at_1pct_fpr": round(float(recall_at_1pct), 4)}

    for thresh in [0.30, 0.50, 0.70, 0.90]:
        flagged = calibrated >= thresh
        tp = np.sum(flagged & (y == 1))
        fp = np.sum(flagged & (y == 0))
        precision_at = tp / max(tp + fp, 1)
        recall_at = tp / max(n_fraud, 1)
        fpr_at = fp / max(n_legit, 1)
        results[f"thresh_{thresh}"] = {
            "flagged": int(np.sum(flagged)),
            "precision": round(float(precision_at), 4),
            "recall": round(float(recall_at), 4),
            "fpr": round(float(fpr_at), 6),
        }

    # Per-model breakdown
    per_model = {}
    for model_name, proba in scores.items():
        try:
            m_roc = roc_auc_score(y, proba)
        except ValueError:
            m_roc = 0.5
        m_sorted = np.argsort(-proba)
        m_ranks = np.searchsorted(m_sorted, fraud_idx)
        m_recall = np.mean(m_ranks < int(n_legit * 0.01))
        per_model[model_name] = {"roc_auc": round(m_roc, 4), "recall_at_1pct_fpr": round(float(m_recall), 4)}
    results["per_model"] = per_model
    results["stacked_roc_auc"] = round(float(roc), 4)

    return results


def main():
    print("=" * 72)
    print("  PS-14 CROSS-DATASET EVALUATION")
    print("=" * 72)

    # Load PS-14 models
    print("\nLoading PS-14 models...")
    models = {}
    model_files = {
        "logistic_regression": "logistic_regression",
        "random_forest": "random_forest",
        "xgboost": "xgboost",
        "xgb_student_distilled": "xgb_student_distilled",
    }
    for display_name, file_stem in model_files.items():
        path = ARTIFACTS / f"{file_stem}.joblib"
        if path.exists():
            models[display_name] = joblib.load(path)
            print(f"  ✓ {display_name}")

    stacker_path = ARTIFACTS / "stacker_v2.joblib"
    if stacker_path.exists():
        models["stacker"] = joblib.load(stacker_path)
        print(f"  ✓ stacker_v2")
    else:
        stacker_path = ARTIFACTS / "stacker.joblib"
        if stacker_path.exists():
            models["stacker"] = joblib.load(stacker_path)
            print(f"  ✓ stacker")

    calibrator_path = ARTIFACTS / "calibrator_v2.joblib"
    if not calibrator_path.exists():
        calibrator_path = ARTIFACTS / "calibrator.joblib"
    if calibrator_path.exists():
        try:
            models["calibrator"] = joblib.load(calibrator_path)
            print(f"  ✓ calibrator")
        except Exception as e:
            print(f"  ⚠ calibrator load failed ({e.__class__.__name__}), using stacked scores")

    # Generate datasets
    print("\nGenerating cross-domain datasets...")
    datasets = {
        "Mobile Money (PaySim-style)": generate_mobile_money(50_000, 0.02),
        "E-commerce (IEEE-CIS-style)": generate_ecommerce(50_000, 0.03),
        "Banking (SWIFT-style)": generate_banking(50_000, 0.015),
    }

    # Also include existing Kaggle data
    kaggle_path = DATA / "validation_kaggle.csv"
    if kaggle_path.exists():
        kdf = pd.read_csv(kaggle_path)
        if "label" in kdf.columns and all(c in kdf.columns for c in PS16_COLS):
            datasets["Credit Card (Kaggle ULB)"] = kdf
            print(f"  ✓ Credit Card (Kaggle ULB) — {len(kdf)} rows from file")
        else:
            print(f"  ⚠ Kaggle data missing columns, skipping")

    for name, df in datasets.items():
        fraud_pct = df["label"].mean() * 100
        print(f"  ✓ {name} — {len(df):,} txns, {df['label'].sum():,} fraud ({fraud_pct:.2f}%)")

    # Evaluate each dataset
    all_results = {}
    for name, df in datasets.items():
        print(f"\n{'─' * 72}")
        print(f"  Evaluating: {name}")
        print(f"{'─' * 72}")
        result = evaluate(name, df, models)
        all_results[name] = result

        print(f"  ROC-AUC:           {result['roc_auc']:.4f}")
        print(f"  PR-AUC:            {result['pr_auc']:.4f}")
        print(f"  Recall @ 1% FPR:   {result['recall_at_1pct_fpr']:.4f}")
        print()
        print(f"  {'Threshold':<12} {'Flagged':>8} {'Precision':>10} {'Recall':>8} {'FPR':>10}")
        print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*8} {'─'*10}")
        for thresh in [0.30, 0.50, 0.70, 0.90]:
            t = result[f"thresh_{thresh}"]
            print(f"  {thresh:<12.2f} {t['flagged']:>8,} {t['precision']:>10.4f} {t['recall']:>8.4f} {t['fpr']:>10.6f}")

        print(f"\n  Per-model ROC-AUC:")
        for mn, mr in result["per_model"].items():
            print(f"    {mn:<30} ROC={mr['roc_auc']:.4f}  R@1%={mr['recall_at_1pct_fpr']:.4f}")

    # ── Cross-dataset summary ────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  CROSS-DATASET COMPARISON SUMMARY")
    print(f"{'═' * 72}")
    print(f"\n  {'Dataset':<32} {'ROC-AUC':>8} {'PR-AUC':>8} {'R@1%FPR':>8} {'Best Thresh':>12}")
    print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8} {'─'*12}")
    for name, r in all_results.items():
        # Find best threshold (highest F1)
        best_f1 = 0
        best_t = 0.50
        for thresh in [0.30, 0.50, 0.70, 0.90]:
            t = r[f"thresh_{thresh}"]
            f1 = 2 * t["precision"] * t["recall"] / max(t["precision"] + t["recall"], 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_t = thresh
        print(f"  {name:<32} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {r['recall_at_1pct_fpr']:>8.4f} {best_t:>12.2f}")

    # ── Key insights ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  KEY INSIGHTS")
    print(f"{'═' * 72}")

    for name, r in all_results.items():
        best_recall = max(r[f"thresh_{t}"]["recall"] for t in [0.30, 0.50, 0.70, 0.90])
        best_fp = min(r[f"thresh_{t}"]["fpr"] for t in [0.30, 0.50, 0.70, 0.90])
        print(f"\n  {name}:")
        print(f"    Best recall: {best_recall:.1%} (at lowest threshold)")
        print(f"    Lowest FPR:  {best_fp:.4%} (at highest threshold)")
        print(f"    Generalization: {'STRONG' if r['roc_auc'] > 0.8 else 'MODERATE' if r['roc_auc'] > 0.6 else 'WEAK'}")

    # Save results
    report_path = DATA / "cross_dataset_results.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {report_path}")

    # Save as text report
    txt_path = DATA / "cross_dataset_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("PS-14 Cross-Dataset Evaluation Report\n")
        f.write("=" * 72 + "\n\n")
        for name, r in all_results.items():
            f.write(f"Dataset: {name}\n")
            f.write(f"  Transactions: {r['n_total']:,}  Fraud: {r['n_fraud']:,}  ({r['n_fraud']/r['n_total']*100:.2f}%)\n")
            f.write(f"  ROC-AUC: {r['roc_auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}\n")
            f.write(f"  Recall @ 1% FPR: {r['recall_at_1pct_fpr']:.4f}\n\n")
    print(f"  Text report saved to {txt_path}")

    print(f"\n{'═' * 72}")
    print("  EVALUATION COMPLETE")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
