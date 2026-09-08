#!/usr/bin/env python3
"""Evaluate PS-14 models against all available REAL fraud datasets.

Datasets:
1. Kaggle ULB Credit Card Fraud (284,807 txns, 492 fraud, 0.17%)
2. UCI Credit Card Default (30,000 clients, 6,636 defaults, 22.1%)

Both are publicly available, legally licensed datasets.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "models" / "artifacts"
DATA = ROOT / "data"

ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "hour_of_day", "is_weekend",
    "shared_device_accounts", "shared_recipient_accounts", "mule_ring_score",
]

# The offline artifacts (models/artifacts/*.joblib) were trained by
# train_compare on the 21-feature vector: the 16 history/context features
# above plus 5 history-derived aggregates. The stored real-dataset CSVs
# predate that vector and carry only the 16. History-derived features have
# a documented neutral default of 0 when no history exists (the same
# convention the runtime FeatureVector and train_compare's snapshot
# backfill use), so pad in canonical metadata order before scoring.
ARTIFACT_FEATURES_FILE = ROOT / "models" / "artifacts" / "metadata.json"
_HISTORY_AGGREGATES = [
    "hour_deviation", "amount_zscore", "velocity_deviation",
    "recipient_novelty", "txn_regularity",
]


def _artifact_feature_order() -> list[str]:
    """Canonical 21-feature order the artifacts were trained in."""
    if ARTIFACT_FEATURES_FILE.exists():
        try:
            meta = json.loads(ARTIFACT_FEATURES_FILE.read_text(encoding="utf-8"))
            feats = meta.get("features")
            if isinstance(feats, list) and len(feats) >= len(ML_FEATURES):
                return list(feats)
        except Exception:
            pass
    return ML_FEATURES + _HISTORY_AGGREGATES


FEATURE_ORDER = _artifact_feature_order()


def load_models() -> dict:
    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost", "xgb_student_distilled"]:
        p = ARTIFACTS / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)
    # Try stacker v2 first
    for fn in ["stacker_v2", "stacker"]:
        p = ARTIFACTS / f"{fn}.joblib"
        if p.exists() and "stacker" not in models:
            models["stacker"] = joblib.load(p)
    return models


def evaluate_dataset(name: str, df: pd.DataFrame, models: dict) -> dict:
    # Score in the artifacts' canonical feature order; columns absent from the
    # real-dataset CSV (history-derived aggregates) take their neutral 0.
    X = np.column_stack([
        df[f].to_numpy(dtype=float) if f in df.columns
        else np.zeros(len(df), dtype=float)
        for f in FEATURE_ORDER
    ])
    y = df["label"].values
    fraud_idx = np.where(y == 1)[0]
    n_fraud = len(fraud_idx)
    n_legit = len(y) - n_fraud

    base_names = [k for k in models if k not in ("stacker", "calibrator")]
    scores = {}
    skipped = []
    for mn in base_names:
        m = models[mn]
        if getattr(m, "n_features_in_", X.shape[1]) != X.shape[1]:
            skipped.append(f"{mn}({getattr(m, 'n_features_in_', '?' )}f != {X.shape[1]}f)")
            continue
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

    calibrated = stacked

    roc = roc_auc_score(y, calibrated)
    prec_arr, rec_arr, _ = precision_recall_curve(y, calibrated)
    pr_auc = auc(rec_arr, prec_arr)

    sorted_idx = np.argsort(-calibrated)
    fraud_ranks = np.searchsorted(sorted_idx, fraud_idx)
    r1 = float(np.mean(fraud_ranks < int(n_legit * 0.01)))

    result = {
        "name": name, "n_total": len(y), "n_fraud": n_fraud, "n_legit": n_legit,
        "roc_auc": round(roc, 4), "pr_auc": round(pr_auc, 4),
        "recall_at_1pct_fpr": round(r1, 4),
    }

    for thresh in [0.10, 0.20, 0.30, 0.50, 0.70]:
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

        # Per-model threshold analysis
        m_th = {}
        try:
            for t in [0.30, 0.50]:
                m_flagged = scores[mn] >= t
                m_tp = int(np.sum(m_flagged & (y == 1)))
                m_fp = int(np.sum(m_flagged & (y == 0)))
                m_th[f"t{t}"] = {
                    "flagged": int(np.sum(m_flagged)),
                    "precision": round(m_tp / max(m_tp + m_fp, 1), 4),
                    "recall": round(m_tp / max(n_fraud, 1), 4),
                }
        except Exception:
            pass

        per_model[mn] = {"roc_auc": m_roc, "recall_at_1pct_fpr": m_r1, "thresholds": m_th}
    result["per_model"] = per_model
    result["skipped_schema_mismatch"] = skipped
    return result


def main():
    print("=" * 72)
    print("  PS-14 REAL DATASET EVALUATION")
    print("  Only legally public, real-world fraud datasets")
    print("=" * 72)

    models = load_models()
    print(f"\nModels loaded: {', '.join(models.keys())}")

    datasets = {}

    # Dataset 1: Kaggle ULB Credit Card Fraud
    kaggle_path = DATA / "validation_kaggle.csv"
    if kaggle_path.exists():
        kdf = pd.read_csv(kaggle_path)
        if "label" in kdf.columns and all(c in kdf.columns for c in ML_FEATURES):
            datasets["Kaggle ULB Credit Card Fraud"] = kdf
            print(f"\n✓ Kaggle ULB Credit Card Fraud:")
            print(f"  Source: Dal Pozzolo et al., IEEE CIDM 2015")
            print(f"  License: CC BY 4.0 (Kaggle) / Academic use")
            print(f"  Size: {len(kdf):,} transactions")
            print(f"  Fraud: {kdf['label'].sum():,} ({kdf['label'].mean()*100:.3f}%)")
            print(f"  Features: 28 PCA components mapped to §16")

    # Dataset 2: UCI Credit Card Default
    uci_path = DATA / "validation_uci_default.csv"
    if uci_path.exists():
        udf = pd.read_csv(uci_path)
        if "label" in udf.columns and all(c in udf.columns for c in ML_FEATURES):
            datasets["UCI Credit Card Default"] = udf
            print(f"\n✓ UCI Credit Card Default:")
            print(f"  Source: Yeh & Lien, 2009 (UCI ML Repository #350)")
            print(f"  License: CC BY 4.0")
            print(f"  Size: {len(udf):,} clients")
            print(f"  Default: {udf['label'].sum():,} ({udf['label'].mean()*100:.2f}%)")
            print(f"  Features: 24 payment/bill features mapped to §16")

    if not datasets:
        print("\nERROR: No datasets found. Run import scripts first.")
        return

    # Evaluate
    all_results = {}
    for name, df in datasets.items():
        print(f"\n{'─' * 72}")
        print(f"  Evaluating: {name}")
        print(f"{'─' * 72}")
        r = evaluate_dataset(name, df, models)
        all_results[name] = r

        print(f"  ROC-AUC:       {r['roc_auc']:.4f}")
        print(f"  PR-AUC:        {r['pr_auc']:.4f}")
        print(f"  Recall@1%FPR:  {r['recall_at_1pct_fpr']:.4f}")
        print()
        print(f"  {'Thresh':<8} {'Flagged':>8} {'Precision':>10} {'Recall':>8} {'FPR':>10}")
        print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*10}")
        for t in [0.10, 0.20, 0.30, 0.50, 0.70]:
            d = r[f"thresh_{t}"]
            print(f"  {t:<8.2f} {d['flagged']:>8,} {d['precision']:>10.4f} {d['recall']:>8.4f} {d['fpr']:>10.6f}")

        print(f"\n  Per-model breakdown:")
        for mn, mr in r["per_model"].items():
            t30 = mr["thresholds"].get("t30", {"flagged": 0, "precision": 0, "recall": 0})
            t50 = mr["thresholds"].get("t50", {"flagged": 0, "precision": 0, "recall": 0})
            print(f"    {mn:<30} ROC={mr['roc_auc']:.4f}  R@1%={mr['recall_at_1pct_fpr']:.4f}")
            print(f"      @0.30: {t30['flagged']:>6} flagged, P={t30['precision']:.4f}, R={t30['recall']:.4f}")
            print(f"      @0.50: {t50['flagged']:>6} flagged, P={t50['precision']:.4f}, R={t50['recall']:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  REAL DATASET COMPARISON SUMMARY")
    print(f"{'═' * 72}")
    print(f"\n  {'Dataset':<35} {'Rows':>8} {'Fraud':>8} {'ROC-AUC':>8} {'PR-AUC':>8} {'R@1%FPR':>8}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name, r in all_results.items():
        print(f"  {name:<35} {r['n_total']:>8,} {r['n_fraud']:>8,} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {r['recall_at_1pct_fpr']:>8.4f}")

    if len(all_results) > 1:
        avg_roc = np.mean([r["roc_auc"] for r in all_results.values()])
        avg_pr = np.mean([r["pr_auc"] for r in all_results.values()])
        avg_r1 = np.mean([r["recall_at_1pct_fpr"] for r in all_results.values()])
        print(f"\n  {'AVERAGE':<35} {'':>8} {'':>8} {avg_roc:>8.4f} {avg_pr:>8.4f} {avg_r1:>8.4f}")

    # ── Insights ─────────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  KEY INSIGHTS")
    print(f"{'═' * 72}")

    for name, r in all_results.items():
        print(f"\n  {name}:")
        if r["roc_auc"] > 0.9:
            gen = "EXCELLENT"
        elif r["roc_auc"] > 0.8:
            gen = "GOOD"
        elif r["roc_auc"] > 0.7:
            gen = "MODERATE"
        else:
            gen = "WEAK"
        print(f"    Generalization: {gen} (ROC-AUC {r['roc_auc']:.4f})")
        t30 = r.get('thresh_0.3', r.get('thresh_0.30', {}))
        if t30:
            print(f"    At threshold 0.30: {t30['precision']:.1%} precision, {t30['recall']:.1%} recall")

        # Best model
        best_mn = max(r["per_model"].items(), key=lambda x: x[1]["roc_auc"])
        print(f"    Best individual model: {best_mn[0]} (ROC-AUC {best_mn[1]['roc_auc']:.4f})")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = DATA / "real_dataset_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Dataset provenance ───────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  DATASET PROVENANCE")
    print(f"{'═' * 72}")
    print("""
  1. Kaggle ULB Credit Card Fraud
     Source: Machine Learning Group - ULB (Université Libre de Bruxelles)
     Paper: Dal Pozzolo et al., "Calibrating Probability with Undersampling
            for Unbalanced Classification," IEEE CIDM 2015
     URL: https://www.kaggle.com/mlg-ulb/creditcardfraud
     License: CC BY 4.0 (Kaggle) / Academic use (original)
     Size: 284,807 transactions, 492 frauds (0.17%)
     Features: Time, Amount, V1-V28 (PCA), Class (fraud label)
     Note: PCA components preserve privacy; V14, V12, V4 are strongest signals

  2. UCI Credit Card Default
     Source: I-C. Yeh and C.-H. Lien, "The comparisons of data mining
            techniques for the predictive accuracy of probability of default
            of credit card clients," Expert Systems with Applications, 2009
     URL: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
     License: CC BY 4.0
     Size: 30,000 clients, 24 months history, 6,636 defaults (22.1%)
     Features: Credit limit, payment history (6 months), bill amounts,
               payment amounts, demographics
     Note: Target is credit default (not fraud), but shares similar
           behavioral signals — delinquency patterns, payment anomalies
""")

    print("  EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
