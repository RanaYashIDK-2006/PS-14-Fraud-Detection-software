#!/usr/bin/env python3
"""Full-dataset evaluation against the validation_kaggle.csv dataset.

Scores every transaction using the production model (FusionEngine + RulesEngine)
and computes comprehensive metrics: ROC-AUC, recall, FPR, precision, throughput.

Usage:
    python scripts/full_eval.py
    python scripts/full_eval.py --input data/validation_kaggle.csv
    python scripts/full_eval.py --max-rows 100000  # quick test
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

from src.privacy_layer.features import ML_FEATURES
from src.risk_engine.fusion import FusionEngine
from src.risk_engine.rules_engine import RulesEngine

import yaml

ARTIFACTS_DIR = ROOT / "models" / "artifacts"
RULES_PATH = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
RESULTS_PATH = ROOT / "db" / "full_eval_results.json"


def load_rules_engine() -> tuple[RulesEngine, int, dict]:
    cfg = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    rules = RulesEngine(cfg["rules"], severity_scale=float(cfg.get("severity_scale", 1.0)))
    severity_floor = int(cfg.get("severity_floor", 80))
    velocity_limits_cfg = cfg.get("velocity_limits")
    return rules, severity_floor, velocity_limits_cfg


def band_of(score: int) -> str:
    if score < 30:
        return "allow"
    elif score < 70:
        return "step_up"
    return "verify"


def full_eval(input_path: Path, chunk_size: int = 50_000, max_rows: int | None = None) -> dict:
    """Run full evaluation on a validation dataset with ML_FEATURES + labels."""
    print(f"Loading {input_path}...")
    t0 = time.time()

    # Load data — use nrows for quick test, otherwise full file
    read_kwargs = {}
    if max_rows:
        read_kwargs["nrows"] = max_rows

    df = pd.read_csv(input_path, **read_kwargs)

    # Check for required columns
    has_label = "label" in df.columns
    if not has_label:
        sys.exit("Input CSV must contain a 'label' column for evaluation.")

    # Compute general features from raw columns where available
    # These improve cross-dataset generalization
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], errors="coerce")
        df["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
        df["month"] = ts.dt.month.fillna(1).astype(int)
    if "hour_of_day" in df.columns:
        h = df["hour_of_day"]
        df["is_night"] = (h < 6).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * h / 24).round(4)
        df["hour_cos"] = np.cos(2 * np.pi * h / 24).round(4)
    if "amount_ratio" in df.columns:
        df["log_amount"] = np.log1p(df["amount_ratio"].clip(lower=0)).round(4)
    elif "Amount" in df.columns:
        amt = pd.to_numeric(df["Amount"].astype(str).str.replace("$", "", regex=False), errors="coerce").fillna(0)
        df["log_amount"] = np.log1p(amt.clip(lower=0)).round(4)
    # Global z-score for amount
    if "log_amount" in df.columns:
        mean_amt = df["log_amount"].mean()
        std_amt = df["log_amount"].std()
        if std_amt > 0:
            df["amount_zscore_global"] = ((df["log_amount"] - mean_amt) / std_amt).round(4)
        else:
            df["amount_zscore_global"] = 0.0

    # Zero-fill missing ML features
    missing = [f for f in ML_FEATURES if f not in df.columns]
    if missing:
        print(f"  Zero-filling {len(missing)} missing features: {missing}")
        for f in missing:
            df[f] = 0.0

    # Ensure correct column order
    X = df[ML_FEATURES].to_numpy(dtype=np.float64, na_value=0.0)
    labels = df["label"].to_numpy(dtype=int)

    print(f"  Loaded {len(df):,} rows ({int(labels.sum()):,} fraud) in {time.time()-t0:.1f}s")
    print(f"  Feature matrix: {X.shape}")

    # Load models
    print(f"Loading models from {ARTIFACTS_DIR}...")
    fusion = FusionEngine(ARTIFACTS_DIR)
    rules_engine, severity_floor, velocity_limits_cfg = load_rules_engine()
    print(f"  Models loaded in {time.time()-t0:.1f}s")

    # Score in chunks (for progress reporting and memory)
    total_rows = len(df)
    all_ml_scores = np.zeros(total_rows)
    all_risk_scores = np.zeros(total_rows, dtype=int)
    all_decisions = []

    t_start = time.time()
    n_chunks = max(1, (total_rows + chunk_size - 1) // chunk_size)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_rows)
        X_chunk = X[start:end]
        feature_dicts = df.iloc[start:end][ML_FEATURES].to_dict(orient="records")

        t_chunk = time.time()

        # ML scores via matrix path (fastest)
        ml_scores = fusion.predict_matrix(X_chunk)

        # Rules evaluation
        rule_scores = np.zeros(len(X_chunk))
        critical_flags = np.zeros(len(X_chunk), dtype=int)
        for j, fd in enumerate(feature_dicts):
            rule = rules_engine.evaluate(fd)
            rule_scores[j] = rule["score"]
            critical_flags[j] = 1 if rule["critical"] else 0

        # Combine ML + rules
        combined = np.maximum(ml_scores, rule_scores)
        risk_raw = np.round(np.clip(combined * 100, 0, 100)).astype(int)

        # Apply severity floor for critical rules
        crit_mask = critical_flags == 1
        risk_raw[crit_mask] = np.maximum(risk_raw[crit_mask], severity_floor)
        risk_raw = np.clip(risk_raw, 0, 100)

        all_ml_scores[start:end] = ml_scores
        all_risk_scores[start:end] = risk_raw
        all_decisions.extend([band_of(int(s)) for s in risk_raw])

        chunk_ms = (time.time() - t_chunk) * 1000
        elapsed = time.time() - t_start
        throughput = (end) / elapsed if elapsed > 0 else 0
        print(f"  Chunk {i+1}/{n_chunks}: {end-start:,} rows in {chunk_ms:.0f}ms "
              f"| {throughput:,.0f} txn/s cumulative")

        del X_chunk, feature_dicts
        gc.collect()

    # Compute metrics
    total_time = time.time() - t_start
    throughput = total_rows / total_time if total_time > 0 else 0
    predictions = (all_risk_scores >= 50).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    try:
        roc_auc = roc_auc_score(labels, all_ml_scores)
    except ValueError:
        roc_auc = 0.0

    n_fraud = int(labels.sum())
    n_legit = int((labels == 0).sum())
    n_allow = sum(1 for d in all_decisions if d == "allow")
    n_step = sum(1 for d in all_decisions if d == "step_up")
    n_verify = sum(1 for d in all_decisions if d == "verify")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(input_path.name),
        "total_rows": total_rows,
        "total_fraud": n_fraud,
        "total_legit": n_legit,
        "fraud_rate_pct": round(n_fraud / total_rows * 100, 3) if total_rows else 0,
        "total_time_s": round(total_time, 2),
        "throughput_txn_s": round(throughput, 0),
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "precision": round(precision, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
        "decisions": {
            "allow": n_allow,
            "step_up": n_step,
            "verify": n_verify,
        },
    }

    print(f"\n{'='*60}")
    print(f"FULL EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Dataset:      {input_path.name}")
    print(f"  Total rows:   {total_rows:,}")
    print(f"  Fraud:        {n_fraud:,} ({results['fraud_rate_pct']}%)")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"  Throughput:   {throughput:,.0f} txn/s")
    print(f"  ROC-AUC:      {roc_auc:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  FPR:          {fpr:.4f}")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Decisions:    allow={n_allow:,} step_up={n_step:,} verify={n_verify:,}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(ROOT / "data" / "validation_kaggle.csv"),
                     help="Input CSV with ML_FEATURES + label columns")
    ap.add_argument("--chunk-size", type=int, default=50_000)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    results = full_eval(Path(args.input), args.chunk_size, args.max_rows)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")
