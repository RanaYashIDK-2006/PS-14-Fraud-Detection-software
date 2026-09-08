"""Test false-positive reduction strategies for the Kaggle dataset.

Strategy: micro-transaction discount for borderline ML scores.
When ML is uncertain (0.50-0.90) and the transaction is <= $1,
require at least one context flag to maintain the score.
Without context flags, the transaction is likely a false positive.
"""
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "creditcard.csv"))
    model = joblib.load(os.path.join(ROOT, "models", "artifacts", "xgb_kaggle.joblib"))
    scaler = joblib.load(os.path.join(ROOT, "models", "artifacts", "scaler_kaggle.joblib"))

    features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = scaler.transform(df[features].values)
    df["ml_score"] = model.predict_proba(X)[:, 1]

    cols = [f"V{i}" for i in range(1, 29)]
    df["anomaly_count"] = (df[cols].abs() > 1.0).sum(axis=1)
    df["energy"] = df[cols].abs().sum(axis=1)

    fraud = df[df.Class == 1]
    legit = df[df.Class == 0]

    print("=" * 90)
    print("  FALSE POSITIVE REDUCTION — Strategy Analysis")
    print("=" * 90)
    print()

    # Baseline: ML threshold 0.50
    baseline_caught = (fraud.ml_score >= 0.50).sum()
    baseline_fp = (legit.ml_score >= 0.50).sum()
    print(f"  BASELINE (ML threshold 0.50):")
    print(f"    Fraud caught: {baseline_caught}/492 ({baseline_caught/492*100:.1f}%)")
    print(f"    False positives: {baseline_fp} ({baseline_fp/len(legit)*100:.4f}%)")
    print()

    # Strategy 1: Raise threshold to 0.70
    s1_caught = (fraud.ml_score >= 0.70).sum()
    s1_fp = (legit.ml_score >= 0.70).sum()
    print(f"  STRATEGY 1: Raise ML threshold to 0.70")
    print(f"    Fraud caught: {s1_caught}/492 ({s1_caught/492*100:.1f}%, delta={s1_caught-baseline_caught})")
    print(f"    False positives: {s1_fp} (delta={s1_fp-baseline_fp})")
    print()

    # Strategy 2: Raise threshold to 0.90
    s2_caught = (fraud.ml_score >= 0.90).sum()
    s2_fp = (legit.ml_score >= 0.90).sum()
    print(f"  STRATEGY 2: Raise ML threshold to 0.90")
    print(f"    Fraud caught: {s2_caught}/492 ({s2_caught/492*100:.1f}%, delta={s2_caught-baseline_caught})")
    print(f"    False positives: {s2_fp} (delta={s2_fp-baseline_fp})")
    print()

    # Strategy 3: Micro-transaction discount
    # For transactions <= $1 with ML score 0.50-0.90, apply a discount factor
    # Only keep flagged if energy > 50 (genuinely anomalous, not just random)
    df["adjusted_score"] = df["ml_score"].copy()
    micro_mask = (df["Amount"] <= 1.0) & (df["ml_score"] >= 0.50) & (df["ml_score"] < 0.90)
    # Discount: reduce score by 0.3 if energy is moderate (20-60)
    moderate_energy = micro_mask & (df["energy"] >= 20) & (df["energy"] < 60)
    df.loc[moderate_energy, "adjusted_score"] = df.loc[moderate_energy, "ml_score"] - 0.3
    # Heavy discount: reduce by 0.5 if energy is low (<20)
    low_energy = micro_mask & (df["energy"] < 20)
    df.loc[low_energy, "adjusted_score"] = df.loc[low_energy, "ml_score"] - 0.5
    df["adjusted_score"] = df["adjusted_score"].clip(0, 1)

    s3_caught = ((df["Class"] == 1) & (df.adjusted_score >= 0.50)).sum()
    s3_fp = ((df["Class"] == 0) & (df.adjusted_score >= 0.50)).sum()
    print(f"  STRATEGY 3: Micro-txn discount (score 0.50-0.90 + Amount<=1)")
    print(f"    Fraud caught: {s3_caught}/492 ({s3_caught/492*100:.1f}%, delta={s3_caught-baseline_caught})")
    print(f"    False positives: {s3_fp} (delta={s3_fp-baseline_fp})")
    print()

    # Strategy 4: Combined — raise base threshold to 0.70 + micro discount
    df["adjusted4"] = df["ml_score"].copy()
    micro4 = (df["Amount"] <= 1.0) & (df["ml_score"] >= 0.70) & (df["ml_score"] < 0.95)
    moderate4 = micro4 & (df["energy"] >= 20) & (df["energy"] < 60)
    df.loc[moderate4, "adjusted4"] = df.loc[moderate4, "ml_score"] - 0.25
    low4 = micro4 & (df["energy"] < 20)
    df.loc[low4, "adjusted4"] = df.loc[low4, "ml_score"] - 0.5
    df["adjusted4"] = df["adjusted4"].clip(0, 1)

    s4_caught = ((df["Class"] == 1) & (df.adjusted4 >= 0.50)).sum()
    s4_fp = ((df["Class"] == 0) & (df.adjusted4 >= 0.50)).sum()
    print(f"  STRATEGY 4: Base 0.70 + micro discount")
    print(f"    Fraud caught: {s4_caught}/492 ({s4_caught/492*100:.1f}%, delta={s4_caught-baseline_caught})")
    print(f"    False positives: {s4_fp} (delta={s4_fp-baseline_fp})")
    print()

    # Strategy 5: Pure threshold — 0.90 (best single-threshold trade-off)
    # Then add back micro-discount for the 0.90+ range
    df["adjusted5"] = df["ml_score"].copy()
    micro5 = (df["Amount"] <= 1.0) & (df["ml_score"] >= 0.90)
    moderate5 = micro5 & (df["energy"] >= 30) & (df["energy"] < 70)
    df.loc[moderate5, "adjusted5"] = df.loc[moderate5, "ml_score"] - 0.45
    low5 = micro5 & (df["energy"] < 30)
    df.loc[low5, "adjusted5"] = df.loc[low5, "ml_score"] - 0.6
    df["adjusted5"] = df["adjusted5"].clip(0, 1)

    s5_caught = ((df["Class"] == 1) & (df.adjusted5 >= 0.50)).sum()
    s5_fp = ((df["Class"] == 0) & (df.adjusted5 >= 0.50)).sum()
    print(f"  STRATEGY 5: Threshold 0.90 + micro discount at 0.90+")
    print(f"    Fraud caught: {s5_caught}/492 ({s5_caught/492*100:.1f}%, delta={s5_caught-baseline_caught})")
    print(f"    False positives: {s5_fp} (delta={s5_fp-baseline_fp})")
    print()

    # Show which FPs each strategy eliminates
    fp_mask = (df["Class"] == 0) & (df["ml_score"] >= 0.50)
    fp_indices = df[fp_mask].index.tolist()
    print(f"  FP ELIMINATION DETAILS (strategy 5):")
    for idx in fp_indices:
        row = df.loc[idx]
        elim = row.adjusted5 < 0.50
        flag = "ELIMINATED" if elim else "STILL FP"
        print(f"    idx={idx:6d} ml={row.ml_score:.4f} adj={row.adjusted5:.4f} amt=${row.Amount:.2f} energy={row.energy:.0f} -> {flag}")

    # Final comparison
    print()
    print("=" * 90)
    print("  RECOMMENDED STRATEGY: 0.90 base + micro-transaction discount")
    print("=" * 90)
    print(f"    Before: {baseline_caught} fraud caught, {baseline_fp} false positives")
    print(f"    After:  {s5_caught} fraud caught, {s5_fp} false positives")
    print(f"    FP reduction: {baseline_fp - s5_fp} eliminated ({(baseline_fp-s5_fp)/baseline_fp*100:.0f}%)")
    print(f"    Fraud cost:   {baseline_caught - s5_caught} missed ({(baseline_caught-s5_caught)/baseline_caught*100:.1f}%)")
    print(f"    Precision at threshold: {s5_caught/(s5_caught+s5_fp)*100:.1f}% (was {baseline_caught/(baseline_caught+baseline_fp)*100:.1f}%)")


if __name__ == "__main__":
    main()
