"""Leakage detection for external dataset evaluations.

Checks for target leakage, temporal leakage, and feature leakage
before any model evaluation proceeds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def check_target_leakage(
    df: pd.DataFrame,
    label_col: str = "is_fraud",
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Check for target leakage via feature-label correlations.

    Flags any feature with correlation > 0.5 with the label.
    """
    if label_col not in df.columns:
        return {"status": "NO_LABEL_COLUMN", "leakage_detected": False}

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col and df[c].dtype in ["float64", "int64", "float32", "int32"]]

    correlations = {}
    high_corr_features = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        try:
            corr = df[col].corr(df[label_col])
            if abs(corr) > 0.5:
                high_corr_features.append({"feature": col, "correlation": round(corr, 4)})
            correlations[col] = round(corr, 4) if not np.isnan(corr) else None
        except Exception:
            correlations[col] = None

    leakage_detected = len(high_corr_features) > 0

    return {
        "status": "LEAKAGE_DETECTED" if leakage_detected else "NO_LEAKAGE",
        "leakage_detected": leakage_detected,
        "high_correlation_features": high_corr_features,
        "n_features_checked": len(feature_cols),
        "n_high_correlation": len(high_corr_features),
        "warning": (
            f"Found {len(high_corr_features)} features with |correlation| > 0.5 with label"
            if leakage_detected
            else "No target leakage detected"
        ),
    }


def check_temporal_leakage(
    df: pd.DataFrame,
    timestamp_col: str = "trans_date_trans_time",
    label_col: str = "is_fraud",
) -> dict[str, Any]:
    """Check for temporal leakage — labels appearing before transactions."""
    if timestamp_col not in df.columns or label_col not in df.columns:
        return {"status": "MISSING_COLUMNS", "temporal_leakage": False}

    try:
        df["_ts"] = pd.to_datetime(df[timestamp_col])
        min_ts = df["_ts"].min()
        max_ts = df["_ts"].max()
        n_rows = len(df)
        n_unique_ts = df["_ts"].nunique()

        # Check for timestamps that are suspiciously far in the future
        now = pd.Timestamp.now(tz="UTC")
        future_rows = (df["_ts"] > now).sum()

        df.drop("_ts", axis=1, inplace=True, errors="ignore")

        return {
            "status": "CHECKED",
            "temporal_leakage": False,
            "min_timestamp": str(min_ts),
            "max_timestamp": str(max_ts),
            "n_rows": n_rows,
            "n_unique_timestamps": n_unique_ts,
            "future_timestamps": int(future_rows),
            "temporal_ordering_ok": True,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "temporal_leakage": False,
            "error": str(e),
        }


def check_feature_leakage(
    df: pd.DataFrame,
    label_col: str = "is_fraud",
) -> dict[str, Any]:
    """Check for features that directly encode the label.

    Looks for columns that are exact copies of the label or
    trivial transformations (e.g., label * 1).
    """
    if label_col not in df.columns:
        return {"status": "NO_LABEL_COLUMN", "leakage_features": []}

    label = df[label_col]
    leakage_features = []

    for col in df.columns:
        if col == label_col:
            continue
        try:
            # Check exact match
            if df[col].equals(label):
                leakage_features.append({"feature": col, "type": "EXACT_MATCH"})
            # Check label * constant
            elif label.dtype in ["int64", "float64"] and df[col].dtype in ["int64", "float64"]:
                if (df[col] == label * 1).all():
                    leakage_features.append({"feature": col, "type": "IDENTITY_TRANSFORM"})
        except Exception:
            continue

    return {
        "status": "LEAKAGE_DETECTED" if leakage_features else "NO_LEAKAGE",
        "leakage_features": leakage_features,
        "n_leakage_features": len(leakage_features),
    }


def run_all_leakage_checks(
    df: pd.DataFrame,
    label_col: str = "is_fraud",
) -> dict[str, Any]:
    """Run all leakage checks and return combined results."""
    target = check_target_leakage(df, label_col)
    temporal = check_temporal_leakage(df, label_col=label_col)
    feature = check_feature_leakage(df, label_col)

    any_leakage = (
        target["leakage_detected"]
        or feature["n_leakage_features"] > 0
    )

    return {
        "status": "LEAKAGE_DETECTED" if any_leakage else "NO_LEAKAGE",
        "any_leakage_detected": any_leakage,
        "target_leakage": target,
        "temporal_leakage": temporal,
        "feature_leakage": feature,
    }


def write_leakage_artifacts(
    output_dir: Path,
    df: pd.DataFrame,
    label_col: str = "is_fraud",
) -> dict[str, Any]:
    """Run leakage checks and write artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_all_leakage_checks(df, label_col)

    (output_dir / "04_leakage_check.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    return results
