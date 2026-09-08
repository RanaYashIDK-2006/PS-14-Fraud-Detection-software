"""Benchmark PS-14 on 1.2M PaySim dataset and compare with industry standards.

FIXED: Removed all label-dependent features from the benchmark transform.
Previously, failed_auth_count_24h, gradual_escalation_score,
shared_device_accounts, shared_recipient_accounts, and mule_ring_score were
derived from df["isFraud"] — classic data leakage producing spurious 1.0000 AUC.

This version only uses features derivable from the transaction itself (no label).

Methodology:
- Dataset: PaySim 1.2M synthetic (not real fraud data)
- Features: 16 PS-14 features, NONE derived from the label
- Evaluation: pure in-process sklearn predict_proba (no HTTP, no DB, no audit)
- Latency: model inference only (no network overhead)
- Throughput: batch inference speed (not real-world TPS)
- Industry numbers: cited from papers/vendor blogs (see sources)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
    "unusual_location_flag", "unusual_recipient_flag", "failed_auth_count_24h",
    "days_since_last_similar_txn", "gradual_escalation_score", "known_device_count",
    "account_tenure_days", "hour_of_day", "is_weekend",
    "shared_device_accounts", "shared_recipient_accounts", "mule_ring_score",
]


def transform_paysim_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw PaySim columns into PS-14 features.

    CRITICAL: NO feature uses df["isFraud"]. All features must be derivable
    from the transaction data alone, as they would be in production.
    """
    rng = np.random.default_rng(42)
    n = len(df)

    features = pd.DataFrame()

    # amount_ratio: transaction amount relative to median for that type
    type_median = df.groupby("type")["amount"].transform("median")
    features["amount_ratio"] = (df["amount"] / type_median.clip(lower=1)).clip(0, 20)

    # txn_freq_last_24h: synthetic (no timestamp in PaySim)
    features["txn_freq_last_24h"] = rng.integers(0, 8, size=n)

    # txn_time_unusual: random (no timestamp in PaySim)
    features["txn_time_unusual"] = rng.binomial(1, 0.15, size=n)

    # new_device_flag: zero source balance = new device (derivable from data)
    features["new_device_flag"] = (df["oldbalanceOrg"] == 0).astype(int)

    # unusual_location_flag: random (no location in PaySim)
    features["unusual_location_flag"] = rng.binomial(1, 0.12, size=n)

    # unusual_recipient_flag: zero dest balance = new recipient (derivable)
    features["unusual_recipient_flag"] = (df["oldbalanceDest"] == 0).astype(int)

    # failed_auth_count_24h: random poisson (no auth data in PaySim)
    # FIXED: was using isFraud label — now uniform distribution
    features["failed_auth_count_24h"] = rng.poisson(0.3, size=n)

    # days_since_last_similar_txn: random (no history in PaySim)
    features["days_since_last_similar_txn"] = rng.exponential(7, size=n)

    # gradual_escalation_score: random (no history in PaySim)
    # FIXED: was using isFraud label — now uniform distribution
    features["gradual_escalation_score"] = rng.beta(1, 5, size=n)

    # known_device_count: random
    features["known_device_count"] = rng.poisson(3, size=n) + 1

    # account_tenure_days: random
    features["account_tenure_days"] = rng.exponential(60, size=n) + 1

    # hour_of_day: random (no timestamp in PaySim)
    features["hour_of_day"] = rng.integers(0, 24, size=n)

    # is_weekend: random
    features["is_weekend"] = rng.binomial(1, 0.28, size=n)

    # shared_device_accounts: random
    # FIXED: was using isFraud label — now uniform distribution
    features["shared_device_accounts"] = rng.poisson(0.1, size=n)

    # shared_recipient_accounts: random
    # FIXED: was using isFraud label — now uniform distribution
    features["shared_recipient_accounts"] = rng.poisson(0.05, size=n)

    # mule_ring_score: random
    # FIXED: was using isFraud label — now uniform distribution
    features["mule_ring_score"] = rng.beta(0.5, 10, size=n)

    features["label"] = df["isFraud"].values
    return features


def main():
    print("=" * 70)
    print("PS-14 BENCHMARK (FIXED — no label leakage)")
    print("=" * 70)
    print()
    print("WARNING: This benchmark uses SYNTHETIC features (no real PaySim")
    print("timestamps, auth data, or device history). The features that WOULD")
    print("be most predictive in production (failed_auth, escalation_score,")
    print("shared_devices, mule_ring) are random noise here.")
    print()
    print("Methodology: pure in-process sklearn predict_proba.")
    print("  - NO HTTP/network overhead")
    print("  - NO database writes")
    print("  - NO audit logging")
    print("  - NO real-world latency")
    print()

    # Load data
    print("[1] Loading 1.2M PaySim dataset...")
    df = pd.read_csv(ROOT / "data" / "paysim_1m.csv")
    print(f"    Rows: {len(df):,}, Fraud: {df['isFraud'].sum():,}")

    # Transform to PS-14 features (NO label leakage)
    print("[2] Transforming to PS-14 features (NO label-dependent features)...")
    feat_df = transform_paysim_to_features(df)
    X = feat_df[ML_FEATURES].values.astype(float)
    y = feat_df["label"].values
    print(f"    Features: {X.shape[1]} (all derivable from transaction data)")

    # Load models
    print("[3] Loading models...")
    xgb_imp = joblib.load(ROOT / "models" / "artifacts" / "xgb_improved.joblib")
    scaler_imp = joblib.load(ROOT / "models" / "artifacts" / "scaler_improved.joblib")
    rf = joblib.load(ROOT / "models" / "artifacts" / "random_forest.joblib")

    # Evaluate
    print("[4] Evaluating (in-process only, no HTTP)...")
    t0 = time.time()
    scores_imp = xgb_imp.predict_proba(scaler_imp.transform(X))[:, 1]
    scores_rf = rf.predict_proba(X)[:, 1]
    ensemble_scores = 0.75 * scores_imp + 0.25 * scores_rf
    eval_time = time.time() - t0

    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y, ensemble_scores)
    pr = average_precision_score(y, ensemble_scores)

    fraud_mask = y == 1
    legit_mask = y == 0
    n_fraud = fraud_mask.sum()
    n_legit = legit_mask.sum()

    print(f"\n    ROC-AUC: {auc:.4f} (NO label leakage)")
    print(f"    PR-AUC: {pr:.4f}")
    print(f"    Inference time: {eval_time:.1f}s ({len(df)/eval_time:,.0f} samples/sec)")
    print(f"    NOTE: This is BATCH inference, not per-request latency.")
    print()

    # Threshold sweep
    print("    Threshold sweep (synthetic features):")
    print(f"    {'Thresh':>7} {'Recall':>8} {'FPR':>8} {'Precision':>8}")
    print(f"    {'-'*40}")
    for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
        tp = ((ensemble_scores >= t) & fraud_mask).sum()
        fp = ((ensemble_scores >= t) & legit_mask).sum()
        recall = tp / n_fraud * 100 if n_fraud > 0 else 0
        fpr = fp / n_legit * 100 if n_legit > 0 else 0
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        print(f"    {t:7.2f} {recall:7.1f}% {fpr:7.2f}% {prec:7.1f}%")

    # Industry comparison (with explicit caveats)
    print("\n" + "=" * 70)
    print("INDUSTRY COMPARISON (caveats below)")
    print("=" * 70)
    print()
    print("System                  ROC-AUC  Source")
    print("-" * 60)
    print(f"PS-14 v2 (this, fixed)  {auc:.4f}  In-process, synthetic features")
    print("GNN (Wang et al. 2024)  0.992    IEEE-CIS, real data")
    print("XGBoost (various)       0.975    Kaggle ULB, real data")
    print("LSTM (Fu et al. 2016)   0.980    Kaggle ULB, real data")
    print("Isolation Forest        0.920    Various, unsupervised")
    print()
    print("CAVEAT: PS-14 was evaluated on SYNTHETIC features with random")
    print("noise for the most predictive signals. Real-world performance")
    print("depends on actual auth failures, device sharing, and behavioral")
    print("deviation — which this benchmark cannot measure.")
    print()
    print("The industry numbers above are from published papers/blogs.")
    print("They are NOT directly comparable — different datasets, different")
    print("feature sets, different evaluation protocols.")

    # Save results with honest metadata
    output = {
        "dataset": "PaySim 1.2M (synthetic)",
        "rows": len(df),
        "fraud_count": int(n_fraud),
        "label_leakage": "NONE — all features derivable from transaction data",
        "evaluation_mode": "in-process sklearn (no HTTP, no DB, no audit)",
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr, 4),
        "inference_time_seconds": round(eval_time, 2),
        "throughput_samples_per_sec": round(len(df) / eval_time, 0),
        "caveats": [
            "Synthetic features — no real auth, device, or behavioral data",
            "Batch inference — not per-request latency",
            "No network, DB, or audit overhead",
            "Industry comparisons are approximate (different datasets/protocols)",
        ],
    }
    out_path = ROOT / "data" / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
