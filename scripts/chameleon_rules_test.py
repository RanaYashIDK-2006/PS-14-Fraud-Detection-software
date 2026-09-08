"""Chameleon fraud detection rules — catches frauds that ML misses by spreading
weak signals across many features instead of one strong anomalous spike.

These rules are designed as a SECONDARY signal layer. They augment (not replace)
the ML model. A transaction flagged by rules gets a risk score boost that feeds
into the existing fusion engine.

Design principle: high recall on the 14 missed frauds with < 50 new false positives
(out of 284,315 legit transactions — < 0.018% FPR increase).
"""
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count_anomalies(row, feature_cols, threshold=1.0):
    """Count how many features deviate beyond threshold (in any direction)."""
    return sum(abs(row[f]) > threshold for f in feature_cols)


def max_composite(row, feature_cols, weights=None):
    """Compute a weighted composite anomaly score from multiple features."""
    if weights is None:
        weights = {f: 1.0 for f in feature_cols}
    return sum(abs(row[f]) * weights.get(f, 1.0) for f in feature_cols)


def opposite_direction_score(row, fraud_means, feature_cols):
    """Score features that deviate in the OPPOSITE direction from typical fraud.
    Typical fraud: V14 negative, V12 negative, V10 negative, V17 negative,
                   V4 positive, V11 positive.
    Missed fraud often has these flipped or weakened."""
    score = 0.0
    for f in feature_cols:
        fraud_mean = fraud_means.get(f, 0)
        if fraud_mean == 0:
            continue
        val = row[f]
        # If value is on the SAME side as fraud but weaker, add partial credit
        if fraud_mean < 0 and val < 0:
            # Weak negative (closer to 0 than fraud mean)
            ratio = abs(val) / abs(fraud_mean) if abs(fraud_mean) > 0.1 else 0
            if 0.2 < ratio < 0.8:
                score += 0.3 * (1 - abs(ratio - 0.5))
        elif fraud_mean > 0 and val > 0:
            ratio = abs(val) / abs(fraud_mean) if abs(fraud_mean) > 0.1 else 0
            if 0.2 < ratio < 0.8:
                score += 0.3 * (1 - abs(ratio - 0.5))
    return score


def chameleon_rules(row, feature_cols, fraud_means):
    """Apply chameleon fraud rules to a single transaction.

    Returns (flagged: bool, rule_scores: dict, combined_score: float)
    """
    scores = {}

    # Rule 1: Multi-feature anomaly count
    # Missed frauds have >7 features with |value| > 1.0
    anomaly_count = count_anomalies(row, feature_cols, threshold=1.0)
    scores["multi_feature_anomaly"] = 1.0 if anomaly_count >= 7 else 0.0

    # Rule 2: High-feature energy
    # Even though no single feature is extreme, the sum of |values| is large
    total_energy = sum(abs(row[f]) for f in feature_cols)
    scores["high_energy"] = 1.0 if total_energy > 25.0 else 0.0

    # Rule 3: V4 + V12 combination
    # V4 positive + V12 negative is a fraud signature, even when each is moderate
    if row["V4"] > 1.5 and row["V12"] < -1.0:
        scores["v4_v12_combo"] = 1.0
    else:
        scores["v4_v12_combo"] = 0.0

    # Rule 4: V14 + V17 combination
    # Both moderately negative is a fraud signal
    if row["V14"] < -1.0 and row["V17"] < -1.0:
        scores["v14_v17_combo"] = 1.0
    elif row["V14"] < -1.0 and row["V17"] > 1.0:
        # V17 flipped — still suspicious (some missed frauds have this)
        scores["v14_v17_combo"] = 0.5
    else:
        scores["v14_v17_combo"] = 0.0

    # Rule 5: High V5 + low V8 (or vice versa) — unusual internal consistency
    if abs(row["V5"]) > 2.0 and abs(row["V8"]) > 2.0:
        scores["v5_v8_anomaly"] = 1.0
    else:
        scores["v5_v8_anomaly"] = 0.0

    # Rule 6: Amount > $200 combined with any moderate feature anomaly
    amount_flag = 1.0 if row["Amount"] > 200 else 0.0
    mod_anomaly = 1.0 if anomaly_count >= 4 else 0.0
    scores["high_amount_plus_anomaly"] = amount_flag * mod_anomaly

    # Rule 7: Low-value micro-transaction ($0-$5) with multi-feature signal
    # Many missed frauds are small test transactions
    if row["Amount"] <= 5.0 and anomaly_count >= 5:
        scores["micro_test_tx"] = 1.0
    else:
        scores["micro_test_tx"] = 0.0

    # Rule 8: Unusual hour + moderate feature deviation
    hour = (row["Time"] / 3600) % 24
    unusual_time = 1.0 if (hour < 5 or hour > 22) else 0.0
    scores["night_tx_plus_anomaly"] = unusual_time * (1.0 if anomaly_count >= 3 else 0.0)

    # Combined score: majority vote with weighted confidence
    rule_hits = sum(1 for v in scores.values() if v >= 0.5)
    total_rules = len(scores)
    combined = sum(scores.values()) / total_rules

    # Flag if at least 3 rules fire (majority)
    flagged = rule_hits >= 3

    return flagged, scores, combined


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "creditcard.csv"))
    model = joblib.load(os.path.join(ROOT, "models", "artifacts", "xgb_kaggle.joblib"))
    scaler = joblib.load(os.path.join(ROOT, "models", "artifacts", "scaler_kaggle.joblib"))

    features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X_full = scaler.transform(df[features].values)
    df["ml_score"] = model.predict_proba(X_full)[:, 1]

    # Vectorized rules for speed on 284K rows
    rule_features = [f"V{i}" for i in range(1, 29)]

    # Rule 1: count features with |value| > 1.0
    anomaly_matrix = df[rule_features].abs() > 1.0
    df["anomaly_count"] = anomaly_matrix.sum(axis=1)

    # Rule 2: total energy
    df["total_energy"] = df[rule_features].abs().sum(axis=1)

    # Rule scores
    df["r_multi_feature"] = (df["anomaly_count"] >= 7).astype(int)
    df["r_high_energy"] = (df["total_energy"] > 25.0).astype(int)
    df["r_v4_v12"] = ((df["V4"] > 1.5) & (df["V12"] < -1.0)).astype(int)
    df["r_v14_v17"] = (
        ((df["V14"] < -1.0) & (df["V17"] < -1.0)).astype(float)
        + 0.5 * ((df["V14"] < -1.0) & (df["V17"] > 1.0)).astype(float)
    )
    df["r_v5_v8"] = ((df["V5"].abs() > 2.0) & (df["V8"].abs() > 2.0)).astype(int)
    df["r_high_amt"] = ((df["Amount"] > 200) & (df["anomaly_count"] >= 4)).astype(int)
    df["r_micro_test"] = ((df["Amount"] <= 5.0) & (df["anomaly_count"] >= 5)).astype(int)
    hour = (df["Time"] / 3600) % 24
    night = ((hour < 5) | (hour > 22)).astype(int)
    df["r_night"] = (night & (df["anomaly_count"] >= 3)).astype(int)

    rule_cols = [
        "r_multi_feature", "r_high_energy", "r_v4_v12",
        "r_v14_v17", "r_v5_v8", "r_high_amt",
        "r_micro_test", "r_night",
    ]
    df["rule_hits"] = df[rule_cols].apply(lambda row: (row >= 0.5).sum(), axis=1)
    df["rule_score"] = df[rule_cols].sum(axis=1) / len(rule_cols)
    df["flagged"] = df["rule_hits"] >= 3
    df["combined"] = df["rule_score"]

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]
    missed_by_ml = fraud[fraud["ml_score"] < 0.05]

    print("=" * 90)
    print("  CHAMELEON RULES — Effectiveness Analysis")
    print("=" * 90)
    print()

    # ML-only performance
    print("  ML-ONLY (threshold 0.50):")
    ml_caught = (fraud.ml_score >= 0.50).sum()
    ml_fp = (legit.ml_score >= 0.50).sum()
    print(f"    Fraud caught: {ml_caught}/492 ({ml_caught/492*100:.1f}%)")
    print(f"    False positives: {ml_fp}/{len(legit)} ({ml_fp/len(legit)*100:.4f}%)")
    print()

    # Rules-only performance
    rules_flagged_fraud = fraud["flagged"]
    rules_flagged_legit = legit["flagged"]
    print("  RULES-ONLY (3+ rules fire):")
    print(f"    Fraud caught: {rules_flagged_fraud.sum()}/492 ({rules_flagged_fraud.sum()/492*100:.1f}%)")
    print(f"    False positives: {rules_flagged_legit.sum()}/{len(legit)} ({rules_flagged_legit.sum()/len(legit)*100:.4f}%)")
    print()

    # Combined ML OR Rules
    combined_fraud = fraud[(fraud.ml_score >= 0.50) | (fraud["flagged"])]
    combined_legit = legit[(legit.ml_score >= 0.50) | (legit["flagged"])]
    print("  COMBINED (ML threshold 0.50 OR rules 3+ fires):")
    print(f"    Fraud caught: {len(combined_fraud)}/492 ({len(combined_fraud)/492*100:.1f}%)")
    print(f"    False positives: {len(combined_legit)}/{len(legit)} ({len(combined_legit)/len(legit)*100:.4f}%)")
    print()

    # Impact on missed frauds specifically
    print("  IMPACT ON ML'S 14 MISSED FRAUDS:")
    missed_caught = missed_by_ml[missed_by_ml["flagged"]]
    missed_still = missed_by_ml[~missed_by_ml["flagged"]]
    print(f"    Now caught by rules: {len(missed_caught)}/14 ({len(missed_caught)/14*100:.0f}%)")
    print(f"    Still missed:        {len(missed_still)}/14 ({len(missed_still)/14*100:.0f}%)")
    print()

    if len(missed_caught) > 0:
        print("  MISSED FRAUDS NOW CAUGHT BY RULES:")
        for _, row in missed_caught.iterrows():
            active = [c for c in rule_cols if row[c] >= 0.5]
            print(f"    #{int(row.name):6d} amount=${row.Amount:.2f} ml={row.ml_score:.4f} rules={row.combined:.3f}  " + " | ".join(active))
        print()

    if len(missed_still) > 0:
        print("  STILL MISSED:")
        for _, row in missed_still.iterrows():
            energy = row["total_energy"]
            print(f"    #{int(row.name):6d} amount=${row.Amount:.2f} ml={row.ml_score:.4f} energy={energy:.1f}")
        print()

    # Break down which rules catch what
    print("  RULE EFFECTIVENESS (on fraud):")
    for rule in rule_cols:
        hf = fraud[rule].sum()
        hl = legit[rule].sum()
        prec = hf / (hf + hl) if (hf + hl) > 0 else 0
        print(f"    {rule:25s}: {hf:3.0f} fraud | {hl:5.0f} legit  (precision={prec:.3f})")
    print()

    # Threshold sweep
    print("  THRESHOLD SWEEP (combined OR):")
    print(f"    {'ML_thresh':>10s} {'Rules':>6s} {'Fraud':>6s} {'FP':>8s} {'F1':>6s}")
    print("    " + "-" * 42)
    for ml_t in [0.90, 0.70, 0.50, 0.30, 0.10]:
        for r_t in [2, 3, 4, 5]:
            c_fraud = fraud[(fraud.ml_score >= ml_t) | (fraud["rule_hits"] >= r_t)].shape[0]
            c_fp = legit[(legit.ml_score >= ml_t) | (legit["rule_hits"] >= r_t)].shape[0]
            ml_only = (fraud.ml_score >= ml_t).sum()
            prec = c_fraud / (c_fraud + c_fp) if (c_fraud + c_fp) > 0 else 0
            rec = c_fraud / 492
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if c_fraud > ml_only:
                print(f"    {ml_t:>10.2f} {r_t:>6d} {c_fraud:>6d} {c_fp:>8d} {f1:>6.3f}")


if __name__ == "__main__":
    main()
