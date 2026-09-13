#!/usr/bin/env python3
"""Evaluate the production model against the IBM v2 dataset (24M rows).

Computes ML_FEATURES from raw IBM columns (real timestamps, amounts, MCCs)
and evaluates recall/FPR with the production model + rules engine.

Usage:
    python scripts/ibm_eval.py                    # full 24M rows
    python scripts/ibm_eval.py --max-rows 100000  # quick test
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.privacy_layer.features import ML_FEATURES
from src.risk_engine.fusion import FusionEngine
from src.risk_engine.rules_engine import RulesEngine

import yaml

ARTIFACTS_DIR = ROOT / "models" / "artifacts"
RULES_PATH = ROOT / "backend" / "src" / "risk_engine" / "rules.yaml"
RESULTS_PATH = ROOT / "db" / "ibm_eval_results.json"

# MCC high-risk categories (based on industry data)
HIGH_RISK_MCC = {5967, 5966, 5964, 5962, 7995, 6051, 6540}  # direct marketing, crypto, money order


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


def compute_features_from_ibm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ML_FEATURES from raw IBM v2 columns.

    IBM columns: User, Card, Year, Month, Day, Time, Amount, Use Chip,
    Merchant Name, Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?
    """
    # Parse timestamp
    df["ts_dt"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-" +
        df["Day"].astype(str).str.zfill(2) + " " + df["Time"],
        errors="coerce"
    )
    hour = df["ts_dt"].dt.hour.fillna(12).astype(int)
    df["hour_of_day"] = hour

    # Parse amount
    amt = pd.to_numeric(
        df["Amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0)
    df["amount"] = amt

    # Weekend
    dow = df["ts"] = df["ts"] if "ts" in df.columns else pd.to_datetime(
        df[["Year", "Month", "Day"]].assign(hour=df["hour_of_day"])
    )
    if not isinstance(df["ts"].dtype, pd.DatetimeTZDtype) and df["ts"].dtype != "object":
        pass
    else:
        df["ts"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
        df["ts"] = df["ts"] + pd.to_timedelta(df["hour_of_day"], unit="h")
    df["is_weekend"] = df["ts"].dt.dayofweek.isin([5, 6]).astype(int)

    # Per-account running stats
    df = df.sort_values(["User", "ts"]).reset_index(drop=True)

    # Account tenure = days since first transaction
    first_ts = df.groupby("User")["ts"].transform("min")
    df["account_tenure_days"] = ((df["ts"] - first_ts).dt.total_seconds() / 86400.0).clip(upper=365)

    # Amount ratio = amount / account median
    acct_median = df.groupby("User")["amount"].transform("median")
    df["amount_ratio"] = (df["amount"] / acct_median.clip(lower=0.01)).clip(upper=20)

    # Txn freq last 24h
    df["txn_freq_last_24h"] = 0
    for user, grp in df.groupby("User"):
        idx = grp.index
        ts_arr = grp["ts"].values
        freq = np.zeros(len(grp), dtype=int)
        for i in range(len(grp)):
            window_start = ts_arr[i] - np.timedelta64(24, "h")
            freq[i] = int(((ts_arr[:i] >= window_start) & (ts_arr[:i] <= ts_arr[i])).sum())
        df.loc[idx, "txn_freq_last_24h"] = freq

    # Txn time unusual: 1 if hour < 6 or hour >= 22
    df["txn_time_unusual"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)

    # New device flag: 1 if first N transactions on this device
    df["new_device_flag"] = 0
    df["unusual_location_flag"] = 0
    df["unusual_recipient_flag"] = 0

    # Failed auth: derived from Errors column
    df["failed_auth_count_24h"] = df["Errors?"].notna().astype(int)

    # Days since last similar
    df["days_since_last_similar_txn"] = 0.0

    # Gradual escalation score
    df["gradual_escalation_score"] = 0.0

    # Known device count
    df["known_device_count"] = 1

    # Shared device/recipient (not available in IBM data)
    df["shared_device_accounts"] = 0
    df["shared_recipient_accounts"] = 0
    df["mule_ring_score"] = 0.0

    # Deviation features
    df["hour_deviation"] = 0.5
    df["amount_zscore"] = 0.0
    df["velocity_deviation"] = 0.0
    df["recipient_novelty"] = 0.5
    df["txn_regularity"] = 0.0

    # Txn amount bucket
    df["txn_amount_bucket"] = "typical"

    # Label
    df["label"] = (df["Is Fraud?"] == "Yes").astype(int)

    return df


def ibm_eval(input_path: Path, chunk_size: int = 200_000, max_rows: int | None = None) -> dict:
    """Compute features from raw IBM v2 data and evaluate."""
    print(f"Loading {input_path}...")
    t0 = time.time()

    read_kwargs = {"usecols": ["User", "Card", "Year", "Month", "Day", "Time",
                                "Amount", "Use Chip", "Merchant Name", "Merchant City",
                                "Merchant State", "Zip", "MCC", "Errors?", "Is Fraud?"]}
    if max_rows:
        read_kwargs["nrows"] = max_rows

    df = pd.read_csv(input_path, **read_kwargs)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Compute features from raw data
    print("  Computing features from raw columns...")
    t_feat = time.time()
    df = compute_ibm_features(df)
    print(f"  Features computed in {time.time()-t_feat:.1f}s")

    # Ensure all ML_FEATURES present
    for f in ML_FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    X = df[ML_FEATURES].to_numpy(dtype=np.float64, na_value=0.0)
    labels = df["label"].to_numpy(dtype=int)
    n_fraud = int(labels.sum())
    print(f"  Feature matrix: {X.shape}, fraud: {n_fraud} ({n_fraud/len(df)*100:.3f}%)")

    # Load models
    print(f"Loading models from {ARTIFACTS_DIR}...")
    fusion = FusionEngine(ARTIFACTS_DIR)
    rules_engine, severity_floor, velocity_limits_cfg = load_rules_engine()
    print(f"  Models loaded in {time.time()-t0:.1f}s")

    # Score in chunks
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
        ml_scores = fusion.predict_matrix(X_chunk)

        rule_scores = np.zeros(len(X_chunk))
        critical_flags = np.zeros(len(X_chunk), dtype=int)
        for j, fd in enumerate(feature_dicts):
            rule = rules_engine.evaluate(fd)
            rule_scores[j] = rule["score"]
            critical_flags[j] = 1 if rule["critical"] else 0

        combined = np.maximum(ml_scores, rule_scores)
        risk_raw = np.round(np.clip(combined * 100, 0, 100)).astype(int)
        crit_mask = critical_flags == 1
        risk_raw[crit_mask] = np.maximum(risk_raw[crit_mask], severity_floor)
        risk_raw = np.clip(risk_raw, 0, 100)

        all_ml_scores[start:end] = ml_scores
        all_risk_scores[start:end] = risk_raw
        all_decisions.extend([band_of(int(s)) for s in risk_raw])

        chunk_ms = (time.time() - t_chunk) * 1000
        elapsed = time.time() - t_start
        throughput = end / elapsed if elapsed > 0 else 0
        print(f"  Chunk {i+1}/{n_chunks}: {end-start:,} rows in {chunk_ms:.0f}ms | {throughput:,.0f} txn/s")

        del X_chunk, feature_dicts
        gc.collect()

    # Metrics
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

    n_allow = sum(1 for d in all_decisions if d == "allow")
    n_step = sum(1 for d in all_decisions if d == "step_up")
    n_verify = sum(1 for d in all_decisions if d == "verify")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(input_path.name),
        "total_rows": total_rows,
        "total_fraud": n_fraud,
        "total_legit": int((labels == 0).sum()),
        "fraud_rate_pct": round(n_fraud / total_rows * 100, 3) if total_rows else 0,
        "total_time_s": round(total_time, 2),
        "throughput_txn_s": round(throughput, 0),
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "precision": round(precision, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        },
        "decisions": {"allow": n_allow, "step_up": n_step, "verify": n_verify},
    }

    print(f"\n{'='*60}")
    print(f"IBM V2 EVALUATION COMPLETE")
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


def compute_ibm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ML_FEATURES from raw IBM v2 columns."""
    # Parse timestamp
    df["ts"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
    time_parts = df["Time"].astype(str).str.split(":", expand=True)
    df["ts"] = df["ts"] + pd.to_timedelta(time_parts[0].astype(int), unit="h") + \
               pd.to_timedelta(time_parts[1].astype(int), unit="m")

    # Parse amount
    amt = pd.to_numeric(
        df["Amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0)
    df["amount"] = amt
    df["hour_of_day"] = df["ts"].dt.hour
    df["is_weekend"] = df["ts"].dt.dayofweek.isin([5, 6]).astype(int)

    # Sort by user and timestamp for running stats
    df = df.sort_values(["User", "ts"]).reset_index(drop=True)

    # Account tenure
    first_ts = df.groupby("User")["ts"].transform("min")
    df["account_tenure_days"] = ((df["ts"] - first_ts).dt.total_seconds() / 86400.0).clip(upper=365)

    # Amount ratio
    acct_median = df.groupby("User")["amount"].transform("median")
    df["amount_ratio"] = (df["amount"] / acct_median.clip(lower=0.01)).clip(upper=20)

    # Txn freq last 24h (vectorized within groups)
    df["txn_freq_last_24h"] = 0
    for user, grp in df.groupby("User"):
        if len(grp) < 2:
            continue
        idx = grp.index
        ts_arr = grp["ts"].values
        amt_arr = grp["amount"].values
        n = len(grp)
        freq = np.zeros(n, dtype=int)
        days_since = np.zeros(n)
        escalation = np.zeros(n)
        for j in range(n):
            if j > 0:
                window_start = ts_arr[j] - np.timedelta64(24, "h")
                mask = (ts_arr[:j] >= window_start)
                freq[j] = int(mask.sum())
                # Days since last similar (amount >= 80%)
                for k in range(j-1, -1, -1):
                    if amt_arr[k] >= 0.8 * amt_arr[j]:
                        days_since[j] = (ts_arr[j] - ts_arr[k]) / np.timedelta64(1, "D")
                        break
                else:
                    days_since[j] = (ts_arr[j] - ts_arr[0]) / np.timedelta64(1, "D")
                # Escalation
                window = amt_arr[max(0, j-9):j+1] / max(acct_median.iloc[idx[j]], 0.01)
                if len(window) >= 3:
                    x = np.arange(len(window), dtype=float)
                    y = np.log(np.maximum(window, 1e-6))
                    slope = float(np.polyfit(x, y, 1)[0])
                    escalation[j] = float(np.clip(slope * len(window) * 0.5, 0.0, 1.0))
        df.loc[idx, "txn_freq_last_24h"] = freq
        df.loc[idx, "days_since_last_similar_txn"] = days_since
        df.loc[idx, "gradual_escalation_score"] = escalation

    # Binary flags (simplified for IBM data — no device/recipient tracking)
    df["txn_time_unusual"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
    df["new_device_flag"] = 0
    df["unusual_location_flag"] = 0
    df["unusual_recipient_flag"] = 0
    df["failed_auth_count_24h"] = df["Errors?"].notna().astype(int)
    df["known_device_count"] = 1
    df["shared_device_accounts"] = 0
    df["shared_recipient_accounts"] = 0
    df["mule_ring_score"] = 0.0
    df["hour_deviation"] = 0.5
    df["amount_zscore"] = 0.0
    df["velocity_deviation"] = 0.0
    df["recipient_novelty"] = 0.5
    df["txn_regularity"] = 0.0
    df["txn_amount_bucket"] = "typical"
    df["label"] = (df["Is Fraud?"] == "Yes").astype(int)

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(ROOT / "data" / "credit_card_transactions-ibm_v2.csv"))
    ap.add_argument("--chunk-size", type=int, default=200_000)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    results = ibm_eval(Path(args.input), args.chunk_size, args.max_rows)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")
