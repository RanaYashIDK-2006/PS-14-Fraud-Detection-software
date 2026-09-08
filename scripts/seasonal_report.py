#!/usr/bin/env python3
"""Generate seasonal decomposition report for the landing page.

Run: .venv/Scripts/python.exe scripts/seasonal_report.py

Outputs data/real_data_evaluation/seasonal_report.json consumed by the
fraud-report frontend.
"""

import sys, json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk_engine.seasonal import full_decomposition, compute_sample_weights

OUT = Path("data/real_data_evaluation")


def main():
    df = pd.concat([
        pd.read_csv('data/kaggle_fraud/fraudTrain.csv'),
        pd.read_csv('data/kaggle_fraud/fraudTest.csv'),
    ], ignore_index=True)
    df = df.drop(columns=['first', 'last', 'street', 'state', 'zip', 'dob', 'job', 'trans_num', 'Unnamed: 0'], errors='ignore')
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df = df.sort_values('unix_time').reset_index(drop=True)

    print("Running seasonal decomposition...")
    decomp = full_decomposition(df)

    print("Computing sample weights...")
    weights = compute_sample_weights(df, method="inverse_rate")
    fraud_mask = df['is_fraud'].values.astype(bool)

    decomp['weight_analysis'] = {
        "method": "inverse_rate",
        "mean_fraud_weight": round(float(weights[fraud_mask].mean()), 3),
        "max_fraud_weight": round(float(weights[fraud_mask].max()), 3),
        "weight_distribution": {
            "low_fraud_months_upweighted": True,
            "description": "Fraud samples in low-fraud months get higher weight to prevent seasonal bias",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "seasonal_report.json"
    out_path.write_text(json.dumps(decomp, indent=2))
    print(f"Saved to {out_path}")
    print(f"\nKey findings:")
    print(f"  Overall fraud rate: {decomp['overall_fraud_rate_pct']:.3f}%")
    print(f"  Trend: {decomp['monthly']['trend_direction']}")
    print(f"  Peak danger hours: {decomp['hourly']['peak_danger_hours']}")
    print(f"  Night/day ratio: {decomp['hourly']['night_vs_day_ratio']}x")
    print(f"  High-risk months: {decomp['monthly']['high_risk_months']}")
    print(f"  Peak day: {decomp['day_of_week']['peak_day']}")
    print(f"  Dangerous hour-month combos: {len(decomp['dangerous_interactions'])}")


if __name__ == '__main__':
    main()
