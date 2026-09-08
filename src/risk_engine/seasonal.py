"""Seasonal decomposition for fraud pattern analysis and training weighting.

Decomposes fraud rates across temporal dimensions (month, day-of-week, hour)
to detect seasonal shifts.  Produces per-sample weights that upweight
under-represented seasonal fraud patterns so the model doesn't bias toward
the majority-season fraud signature.

Standard approach:
  1. Compute fraud rate per temporal bucket (month, dow, hour).
  2. Compute global average fraud rate.
  3. Weight = (global_avg / bucket_fraud_rate) for fraud samples,
     1.0 for legitimate samples.
  4. This prevents the model from overfitting to the most common fraud season.

The decomposition also identifies:
  - Which months have the highest/lowest fraud rates
  - Which hours are most dangerous
  - Day-of-week patterns
  - Trend (fraud rate increasing/decreasing over time)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def decompose_monthly(df: pd.DataFrame, ts_col: str = "trans_date_trans_time",
                      label_col: str = "is_fraud") -> dict[str, Any]:
    """Monthly fraud rate decomposition with trend analysis.

    Returns:
        monthly_rates: dict mapping "YYYY-MM" -> fraud_rate
        trend: slope of fraud rate over time (positive = increasing)
        high_risk_months: list of months with above-average fraud
        low_risk_months: list of months with below-average fraud
        overall_rate: global fraud rate
    """
    ts = pd.to_datetime(df[ts_col])
    year_month = ts.dt.to_period("M")
    grouped = df.groupby(year_month)

    monthly = {}
    rates = []
    periods = []
    for period, group in grouped:
        key = str(period)
        rate = group[label_col].mean()
        monthly[key] = {
            "n_transactions": int(len(group)),
            "n_fraud": int(group[label_col].sum()),
            "fraud_rate": round(float(rate), 6),
            "fraud_rate_pct": round(float(rate) * 100, 3),
        }
        rates.append(rate)
        periods.append(period.ordinal)

    overall_rate = float(df[label_col].mean())

    # Linear trend
    if len(periods) >= 2:
        coeffs = np.polyfit(np.array(periods, dtype=float), np.array(rates), 1)
        trend_slope = float(coeffs[0])
    else:
        trend_slope = 0.0

    # Classify months
    high_risk = [k for k, v in monthly.items() if v["fraud_rate"] > overall_rate * 1.2]
    low_risk = [k for k, v in monthly.items() if v["fraud_rate"] < overall_rate * 0.8]

    return {
        "monthly_rates": monthly,
        "trend_slope": round(trend_slope, 8),
        "trend_direction": "increasing" if trend_slope > 1e-6 else "decreasing" if trend_slope < -1e-6 else "stable",
        "high_risk_months": high_risk,
        "low_risk_months": low_risk,
        "overall_rate": round(overall_rate, 6),
        "overall_rate_pct": round(overall_rate * 100, 3),
    }


def decompose_hourly(df: pd.DataFrame, ts_col: str = "trans_date_trans_time",
                     label_col: str = "is_fraud") -> dict[str, Any]:
    """Hourly fraud rate decomposition.

    Identifies dangerous hours (typically late night / early morning).
    """
    ts = pd.to_datetime(df[ts_col])
    hour = ts.dt.hour
    grouped = df.groupby(hour)

    hourly = {}
    rates = []
    hours = []
    for h, group in grouped:
        rate = group[label_col].mean()
        hourly[int(h)] = {
            "n_transactions": int(len(group)),
            "n_fraud": int(group[label_col].sum()),
            "fraud_rate": round(float(rate), 6),
            "fraud_rate_pct": round(float(rate) * 100, 3),
        }
        rates.append(rate)
        hours.append(h)

    overall_rate = float(df[label_col].mean())
    rates_arr = np.array(rates)

    # Peak danger hours (top 3 by fraud rate)
    sorted_indices = np.argsort(rates_arr)[::-1]
    peak_hours = [int(hours[i]) for i in sorted_indices[:3]]
    safe_hours = [int(hours[i]) for i in sorted_indices[-3:]]

    # Night vs day fraud rate
    night_mask = hour.isin([22, 23, 0, 1, 2, 3])
    day_mask = ~night_mask
    night_rate = float(df.loc[night_mask, label_col].mean()) if night_mask.any() else 0.0
    day_rate = float(df.loc[day_mask, label_col].mean()) if day_mask.any() else 0.0

    return {
        "hourly_rates": hourly,
        "overall_rate": round(overall_rate, 6),
        "peak_danger_hours": peak_hours,
        "safest_hours": safe_hours,
        "night_fraud_rate": round(night_rate, 6),
        "day_fraud_rate": round(day_rate, 6),
        "night_vs_day_ratio": round(night_rate / max(day_rate, 1e-8), 2),
    }


def decompose_dow(df: pd.DataFrame, ts_col: str = "trans_date_trans_time",
                  label_col: str = "is_fraud") -> dict[str, Any]:
    """Day-of-week fraud rate decomposition."""
    ts = pd.to_datetime(df[ts_col])
    dow = ts.dt.dayofweek
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped = df.groupby(dow)

    dow_rates = {}
    rates = []
    for d, group in grouped:
        rate = group[label_col].mean()
        dow_rates[day_names[d]] = {
            "n_transactions": int(len(group)),
            "n_fraud": int(group[label_col].sum()),
            "fraud_rate": round(float(rate), 6),
            "fraud_rate_pct": round(float(rate) * 100, 3),
        }
        rates.append(rate)

    overall_rate = float(df[label_col].mean())
    rates_arr = np.array(rates)
    peak_idx = int(np.argmax(rates_arr))
    low_idx = int(np.argmin(rates_arr))

    return {
        "dow_rates": dow_rates,
        "overall_rate": round(overall_rate, 6),
        "peak_day": day_names[peak_idx],
        "safest_day": day_names[low_idx],
        "weekday_vs_weekend": round(
            float(np.mean(rates_arr[:5])) / max(float(np.mean(rates_arr[5:])), 1e-8), 2
        ),
    }


def compute_sample_weights(df: pd.DataFrame, ts_col: str = "trans_date_trans_time",
                           label_col: str = "is_fraud",
                           method: str = "inverse_rate") -> np.ndarray:
    """Compute per-sample weights for seasonal balancing.

    Methods:
        inverse_rate: weight = (global_avg / bucket_rate) for fraud samples.
                     Legitimate samples get weight 1.0.
                     This prevents the model from overfitting to the most
                     common fraud season.

        balanced: same as inverse_rate but also adjusts legitimate samples
                  to balance the class ratio within each season.

    Returns:
        Array of weights, one per row.
    """
    ts = pd.to_datetime(df[ts_col])
    labels = df[label_col].values.astype(int)

    # Create temporal buckets: month + hour combination
    month = ts.dt.month.values
    hour = ts.dt.hour.values
    buckets = month * 100 + hour  # e.g., 100 = Jan-00h, 1222 = Dec-22h

    overall_rate = float(labels.mean())
    if overall_rate <= 0:
        return np.ones(len(df), dtype=np.float32)

    weights = np.ones(len(df), dtype=np.float32)

    if method == "inverse_rate":
        # Compute fraud rate per bucket
        unique_buckets = np.unique(buckets)
        for b in unique_buckets:
            mask = buckets == b
            fraud_mask = mask & (labels == 1)
            n_fraud = fraud_mask.sum()
            n_total = mask.sum()
            if n_total < 10:
                continue  # skip tiny buckets
            bucket_rate = n_fraud / n_total
            if bucket_rate > 0:
                # Upweight fraud samples in low-fraud buckets
                weight = overall_rate / bucket_rate
                weight = min(weight, 10.0)  # cap at 10x
                weights[fraud_mask] = weight
            # Legitimate samples in high-fraud buckets get slightly downweighted
            legit_mask = mask & (labels == 0)
            if bucket_rate > overall_rate * 1.5:
                weights[legit_mask] = 0.8

    elif method == "balanced":
        # Within each bucket, balance fraud/legit weights
        unique_buckets = np.unique(buckets)
        for b in unique_buckets:
            mask = buckets == b
            fraud_mask = mask & (labels == 1)
            legit_mask = mask & (labels == 0)
            n_fraud = fraud_mask.sum()
            n_legit = legit_mask.sum()
            n_total = n_fraud + n_legit
            if n_total < 10:
                continue
            # Weight so fraud and legit contribute equally to each bucket
            if n_fraud > 0:
                weights[fraud_mask] = n_legit / n_fraud
            if n_legit > 0:
                weights[legit_mask] = 1.0

    return weights


def full_decomposition(df: pd.DataFrame, ts_col: str = "trans_date_trans_time",
                       label_col: str = "is_fraud") -> dict[str, Any]:
    """Run complete seasonal decomposition across all temporal dimensions."""
    monthly = decompose_monthly(df, ts_col, label_col)
    hourly = decompose_hourly(df, ts_col, label_col)
    dow = decompose_dow(df, ts_col, label_col)

    # Cross-tabulation: month × hour interaction
    ts = pd.to_datetime(df[ts_col])
    month = ts.dt.month
    hour = ts.dt.hour
    labels = df[label_col].values.astype(int)

    # Find the most dangerous month-hour combinations
    interaction_rates = {}
    for m in range(1, 13):
        for h in [0, 3, 6, 9, 12, 15, 18, 21]:
            mask = (month == m) & (hour == h)
            if mask.sum() < 5:
                continue
            rate = labels[mask.values].mean()
            if rate > float(df[label_col].mean()) * 2:
                interaction_rates[f"{m:02d}-{h:02d}h"] = round(float(rate) * 100, 3)

    return {
        "monthly": monthly,
        "hourly": hourly,
        "day_of_week": dow,
        "dangerous_interactions": interaction_rates,
        "n_transactions": len(df),
        "n_fraud": int(labels.sum()),
        "overall_fraud_rate_pct": round(float(labels.mean()) * 100, 3),
    }


def save_decomposition_report(decomp: dict, path: Path) -> None:
    """Save decomposition report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decomp, indent=2, default=str))
