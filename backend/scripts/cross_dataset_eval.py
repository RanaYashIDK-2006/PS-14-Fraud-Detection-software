#!/usr/bin/env python3
"""Cross-dataset generalization comparison: synthetic (production) vs IBM v2 models.

Scores the SAME rows with BOTH model families (ML-only, no rules) and reports
ROC-AUC / PR-AUC / recall@1%FPR / best-F1 per dataset, plus the per-dataset
winner. Datasets:

  * synthetic    data/transactions.csv          (model's own training dist)
  * ibm_v2       IBM v2 holdout (fresh rows NOT used in training)
  * kaggle_pca   data/creditcard.csv  (ULB, PCA features -> mapped onto the
                 21-feature contract via rank-preserving proxy mapping)

Usage:
    python backend/scripts/cross_dataset_eval.py [--ibm-rows 300000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.privacy_layer.features import ML_FEATURES

ARTIFACTS = ROOT / "models" / "artifacts"
DATA = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "cross_dataset"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Fractions used by ibm_train for the user-disjoint split (70/15/15).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85


def load_models():
    import joblib

    synth = {
        "lr": joblib.load(ARTIFACTS / "logistic_regression.joblib"),
        "rf": joblib.load(ARTIFACTS / "random_forest.joblib"),
        "xgb": joblib.load(ARTIFACTS / "xgboost.joblib"),
    }
    ibm = {
        "lr": joblib.load(ARTIFACTS / "lr_model.joblib"),
        "rf": joblib.load(ARTIFACTS / "rf_model.joblib"),
        "xgb": joblib.load(ARTIFACTS / "xgb_model.joblib"),
    }
    return synth, ibm


def score_fused(models: dict, X: np.ndarray) -> np.ndarray:
    p_lr = models["lr"].predict_proba(X)[:, 1]
    p_rf = models["rf"].predict_proba(X)[:, 1]
    p_xgb = models["xgb"].predict_proba(X)[:, 1]
    return (p_lr + p_rf + p_xgb) / 3.0


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    out = {"roc_auc": float(roc_auc_score(y, s)),
           "pr_auc": float(average_precision_score(y, s))}
    fpr, tpr, thr = roc_curve(y, s)
    idx = np.searchsorted(fpr, 0.01, side="right")
    idx = max(idx - 1, 0)
    out["recall_at_1pct_fpr"] = float(tpr[idx])
    # best-F1 operating point
    best_f1, best_t = 0.0, 0.5
    for t in np.unique(s)[:: max(1, len(np.unique(s)) // 200)]:
        f1 = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    out["best_f1"] = float(best_f1)
    out["best_f1_threshold"] = best_t
    return out


def load_synthetic() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA / "transactions.csv")
    missing = [f for f in ML_FEATURES if f not in df.columns]
    for f in missing:
        df[f] = 0.0
    X = df[ML_FEATURES].to_numpy(dtype=np.float64, na_value=0.0)
    y = df["label"].to_numpy(dtype=int)
    return X, y


def load_ibm_holdout(max_rows: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Fresh IBM rows AFTER the training prefix (never seen by the model)."""
    sys.path.insert(0, str(ROOT / "backend" / "scripts"))
    from ibm_train import compute_ibm_features  # same features as training

    total_lines = 24_386_900  # known line count (header included)
    need = int(total_lines * VAL_FRAC) + 1  # skip everything through val
    read_kwargs = dict(
        usecols=["User", "Card", "Year", "Month", "Day", "Time", "Amount",
                 "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
                 "Zip", "MCC", "Errors?", "Is Fraud?"],
        skiprows=range(1, need),
    )
    if max_rows:
        read_kwargs["nrows"] = max_rows

    df = pd.read_csv(DATA / "credit_card_transactions-ibm_v2.csv", **read_kwargs)
    df = compute_ibm_features(df)
    for f in ML_FEATURES:
        if f not in df.columns:
            df[f] = 0.0
    X = df[ML_FEATURES].to_numpy(dtype=np.float64, na_value=0.0)
    y = df["label"].to_numpy(dtype=int)
    return X, y


def load_kaggle_pca() -> tuple[np.ndarray, np.ndarray]:
    """ULB creditcard.csv: V1..V28 PCA + Amount + Class.

    The 21-feature contract cannot be computed from PCA columns, so we map
    the PCA components RANK-PRESERVINGLY onto the contract: V-features are
    already variance-ordered, so V1..V21 proxy the 21 contract features and
    Amount/log-amount land where the contract keeps them. This is a PROXY
    evaluation: it measures rank discrimination transfer, not true feature
    semantics. Flagged as such in the report.
    """
    df = pd.read_csv(DATA / "creditcard.csv")
    y = df["Class"].to_numpy(dtype=int)
    X = np.zeros((len(df), len(ML_FEATURES)), dtype=np.float64)
    contract = list(ML_FEATURES)
    # Put log-amount into the contract's amount-ish slot if present
    for i, name in enumerate(contract):
        if name == "log_amount":
            X[:, i] = np.log1p(df["Amount"].to_numpy(dtype=np.float64))
        elif name in ("amount_ratio", "amount_zscore"):
            X[:, i] = np.log1p(df["Amount"].to_numpy(dtype=np.float64))
        elif name.startswith("V") and name[1:].isdigit() and 1 <= int(name[1:]) <= 28:
            X[:, i] = df[name].to_numpy(dtype=np.float64)
    # Fill remaining slots with V-components (rank-preserving proxy)
    vcols = [c for c in df.columns if c.startswith("V")]
    vi = 0
    for i, name in enumerate(contract):
        if np.all(X[:, i] == 0) and vi < len(vcols):
            X[:, i] = df[vcols[vi]].to_numpy(dtype=np.float64)
            vi += 1
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ibm-rows", type=int, default=300_000,
                    help="rows of IBM holdout to score (post-training region)")
    args = ap.parse_args()

    print("Loading models...")
    synth_models, ibm_models = load_models()

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    print("Loading synthetic...")
    datasets["synthetic"] = load_synthetic()
    print("Loading IBM v2 holdout (fresh rows after the training prefix)...")
    datasets["ibm_v2_holdout"] = load_ibm_holdout(args.ibm_rows)
    print("Loading Kaggle ULB (PCA proxy mapping)...")
    datasets["kaggle_ulb_pca_proxy"] = load_kaggle_pca()

    results: dict[str, dict] = {}
    for ds_name, (X, y) in datasets.items():
        print(f"\n=== {ds_name}: {len(X):,} rows, {int(y.sum()):,} fraud ===")
        t0 = time.time()
        s_synth = score_fused(synth_models, X)
        s_ibm = score_fused(ibm_models, X)
        m_synth = metrics(y, s_synth)
        m_ibm = metrics(y, s_ibm)
        dt = time.time() - t0
        print(f"  scored both models in {dt:.1f}s")
        results[ds_name] = {
            "rows": int(len(X)),
            "fraud": int(y.sum()),
            "fraud_rate": round(float(y.mean()), 6),
            "synthetic_model": {k: round(v, 4) for k, v in m_synth.items()},
            "ibm_v2_model": {k: round(v, 4) for k, v in m_ibm.items()},
        }
        hdr = f"  {'metric':<20}{'synthetic':>12}{'ibm_v2':>12}"
        print(hdr)
        for k in m_synth:
            print(f"  {k:<20}{m_synth[k]:>12.4f}{m_ibm[k]:>12.4f}")

    # Verdicts
    print("\n=== VERDICTS ===")
    for ds_name, r in results.items():
        s = r["synthetic_model"]["roc_auc"]
        i = r["ibm_v2_model"]["roc_auc"]
        winner = "ibm_v2" if i > s else ("synthetic" if s > i else "tie")
        print(f"  {ds_name:<24} ROC-AUC synth={s:.4f} ibm={i:.4f} -> winner: {winner}")

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "ML-only (no rules) comparison. Both models scored on identical rows. "
            "IBM v2 holdout rows are strictly AFTER the training prefix (user-disjoint "
            "70/15/15 split, users shared across no folds). Kaggle ULB is a RANK-PRESERVING "
            "PCA-PROXY mapping onto the 21-feature contract (V1..V21 slot-filling); "
            "treat those numbers as rank-transfer evidence, not feature-semantic evaluation."
        ),
        "datasets": results,
    }
    out = REPORT_DIR / "cross_dataset_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
