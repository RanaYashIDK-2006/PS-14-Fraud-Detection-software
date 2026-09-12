#!/usr/bin/env python3
"""Comprehensive fraud taxonomy evaluation with 15+ behavioral categories.

Evaluates the PS-14 model against diverse fraud archetypes and reports
honest ROC-AUC, PR-AUC, and per-category recall metrics.

Fraud Categories:
1.  Account Takeover (ATO) - credential compromise, rapid burst
2.  Card-Not-Present (CNP) - small rapid transactions, new merchants
3.  Mule Activity - shared devices/recipients, money convergence
4.  Payment Fraud - stolen card, unauthorized transactions
5.  Credential Abuse - password stuffing, brute force
6.  Velocity Fraud - rapid-fire transactions, velocity limits
7.  Device Compromise - malware, session hijacking
8.  Merchant Fraud - collusion, fake merchants
9.  Synthetic Identity - fabricated identity, thin file
10. First-Party Fraud - friendly fraud, chargeback abuse
11. Gradual Escalation - boiling frog, slow ramp
12. Micro-Fraud - testing stolen cards with tiny amounts
13. Geographic Anomaly - impossible travel, location spoofing
14. Time Anomaly - unusual hours, pattern deviation
15. Recipient Anomaly - new payees, unusual destinations
16. Amount Anomaly - unusual transaction sizes
17. Behavioral Deviation - pattern break from normal activity
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

from src.privacy_layer.features import ML_FEATURES, mule_ring_score


# ============================================================================
# FRAUD TAXONOMY - 17 behavioral categories
# ============================================================================

FRAUD_CATEGORIES = {
    "ATO": "Account Takeover - credential compromise, rapid burst of txns",
    "CNP": "Card-Not-Present - small rapid transactions to new merchants",
    "MULE": "Mule Activity - shared devices/recipients, money convergence",
    "PAYMENT_FRAUD": "Payment Fraud - stolen card, unauthorized transactions",
    "CREDENTIAL_ABUSE": "Credential Abuse - password stuffing, brute force attempts",
    "VELOCITY_FRAUD": "Velocity Fraud - rapid-fire transactions exceeding limits",
    "DEVICE_COMPROMISE": "Device Compromise - malware, session hijacking patterns",
    "MERCHANT_FRAUD": "Merchant Fraud - collusion, fake merchant patterns",
    "SYNTHETIC_ID": "Synthetic Identity - fabricated identity, thin file",
    "FIRST_PARTY": "First-Party Fraud - friendly fraud, chargeback abuse",
    "GRADUAL_ESCALATION": "Gradual Escalation - boiling frog, slow amount ramp",
    "MICRO_FRAUD": "Micro-Fraud - testing stolen cards with tiny amounts",
    "GEO_ANOMALY": "Geographic Anomaly - impossible travel, location spoofing",
    "TIME_ANOMALY": "Time Anomaly - unusual hours, pattern deviation",
    "RECIPIENT_ANOMALY": "Recipient Anomaly - new payees, unusual destinations",
    "AMOUNT_ANOMALY": "Amount Anomaly - unusual transaction sizes",
    "BEHAVIORAL_DEVIATION": "Behavioral Deviation - pattern break from normal activity",
}


def generate_category_samples(
    category: str,
    n_samples: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Generate feature vectors for a specific fraud category."""
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples):
        if category == "ATO":
            # Account takeover: high failed auth, new device, unusual location/time
            samples.append({
                "amount_ratio": rng.uniform(1.5, 4.0),
                "txn_freq_last_24h": int(rng.integers(5, 15)),
                "txn_time_unusual": 1,
                "new_device_flag": 1,
                "unusual_location_flag": 1,
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": int(rng.integers(3, 10)),
                "days_since_last_similar_txn": rng.uniform(0.01, 1.0),
                "gradual_escalation_score": rng.uniform(0.0, 0.3),
                "known_device_count": 1,
                "account_tenure_days": rng.uniform(30, 365),
                "hour_of_day": int(rng.choice([1, 2, 3, 4, 5])),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "CNP":
            # Card-not-present: small amounts, many new merchants, night hours
            samples.append({
                "amount_ratio": rng.uniform(0.3, 1.2),
                "txn_freq_last_24h": int(rng.integers(8, 25)),
                "txn_time_unusual": 1,
                "new_device_flag": rng.choice([0, 1]),
                "unusual_location_flag": rng.choice([0, 1]),
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(0.01, 0.5),
                "gradual_escalation_score": 0.0,
                "known_device_count": rng.integers(1, 4),
                "account_tenure_days": rng.uniform(1, 365),
                "hour_of_day": int(rng.choice([0, 1, 2, 3, 23])),
                "is_weekend": rng.choice([0, 1]),
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "MULE":
            # Mule: shared devices/recipients, moderate amounts
            shared = int(rng.integers(2, 6))
            samples.append({
                "amount_ratio": rng.uniform(0.8, 1.5),
                "txn_freq_last_24h": int(rng.integers(1, 5)),
                "txn_time_unusual": 0,
                "new_device_flag": 1,
                "unusual_location_flag": rng.choice([0, 1]),
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(1.0, 7.0),
                "gradual_escalation_score": 0.0,
                "known_device_count": 1,
                "account_tenure_days": rng.uniform(5, 60),
                "hour_of_day": int(rng.integers(8, 20)),
                "is_weekend": 0,
                "shared_device_accounts": shared,
                "shared_recipient_accounts": shared,
                "mule_ring_score": round(min(1.0, shared / 4.0), 4),
            })
        elif category == "PAYMENT_FRAUD":
            # Stolen card: large amounts, new device, unusual location
            samples.append({
                "amount_ratio": rng.uniform(3.0, 8.0),
                "txn_freq_last_24h": int(rng.integers(1, 4)),
                "txn_time_unusual": rng.choice([0, 1]),
                "new_device_flag": 1,
                "unusual_location_flag": 1,
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": rng.choice([0, 0, 1]),
                "days_since_last_similar_txn": rng.uniform(5.0, 30.0),
                "gradual_escalation_score": 0.0,
                "known_device_count": 1,
                "account_tenure_days": rng.uniform(30, 365),
                "hour_of_day": int(rng.integers(10, 22)),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "CREDENTIAL_ABUSE":
            # Password stuffing: many failed auths, rapid attempts
            samples.append({
                "amount_ratio": rng.uniform(0.5, 2.0),
                "txn_freq_last_24h": int(rng.integers(10, 30)),
                "txn_time_unusual": 1,
                "new_device_flag": 1,
                "unusual_location_flag": 1,
                "unusual_recipient_flag": 0,
                "failed_auth_count_24h": int(rng.integers(8, 20)),
                "days_since_last_similar_txn": rng.uniform(0.001, 0.1),
                "gradual_escalation_score": 0.0,
                "known_device_count": 1,
                "account_tenure_days": rng.uniform(0.001, 1.0),
                "hour_of_day": int(rng.choice([0, 1, 2, 3, 4])),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "VELOCITY_FRAUD":
            # Velocity: rapid-fire transactions exceeding normal patterns
            samples.append({
                "amount_ratio": rng.uniform(0.5, 2.0),
                "txn_freq_last_24h": int(rng.integers(15, 40)),
                "txn_time_unusual": 0,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 0,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(0.001, 0.05),
                "gradual_escalation_score": 0.0,
                "known_device_count": rng.integers(2, 5),
                "account_tenure_days": rng.uniform(30, 365),
                "hour_of_day": int(rng.integers(8, 20)),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "DEVICE_COMPROMISE":
            # Device compromise: new device but familiar patterns otherwise
            samples.append({
                "amount_ratio": rng.uniform(0.8, 2.5),
                "txn_freq_last_24h": int(rng.integers(2, 8)),
                "txn_time_unusual": 0,
                "new_device_flag": 1,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": rng.choice([0, 0, 1]),
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(1.0, 5.0),
                "gradual_escalation_score": rng.uniform(0.0, 0.5),
                "known_device_count": 2,
                "account_tenure_days": rng.uniform(60, 365),
                "hour_of_day": int(rng.integers(8, 22)),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "MERCHANT_FRAUD":
            # Merchant fraud: collusion patterns, repeated same merchant
            samples.append({
                "amount_ratio": rng.uniform(1.0, 3.0),
                "txn_freq_last_24h": int(rng.integers(3, 10)),
                "txn_time_unusual": 0,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(0.5, 3.0),
                "gradual_escalation_score": rng.uniform(0.0, 0.3),
                "known_device_count": rng.integers(1, 3),
                "account_tenure_days": rng.uniform(10, 180),
                "hour_of_day": int(rng.integers(9, 18)),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "SYNTHETIC_ID":
            # Synthetic identity: very new account, thin history
            samples.append({
                "amount_ratio": rng.uniform(0.5, 2.0),
                "txn_freq_last_24h": int(rng.integers(1, 4)),
                "txn_time_unusual": 0,
                "new_device_flag": 1,
                "unusual_location_flag": rng.choice([0, 1]),
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(0.001, 1.0),
                "gradual_escalation_score": 0.0,
                "known_device_count": 1,
                "account_tenure_days": rng.uniform(0.001, 5.0),
                "hour_of_day": int(rng.integers(8, 22)),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "FIRST_PARTY":
            # First-party fraud: intentional chargeback, normal-looking txns
            samples.append({
                "amount_ratio": rng.uniform(1.0, 3.0),
                "txn_freq_last_24h": int(rng.integers(1, 3)),
                "txn_time_unusual": 0,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 0,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(1.0, 7.0),
                "gradual_escalation_score": 0.0,
                "known_device_count": rng.integers(1, 3),
                "account_tenure_days": rng.uniform(30, 365),
                "hour_of_day": int(rng.integers(9, 18)),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "GRADUAL_ESCALATION":
            # Boiling frog: slow amount increase over time
            samples.append({
                "amount_ratio": rng.uniform(2.0, 5.0),
                "txn_freq_last_24h": int(rng.integers(1, 4)),
                "txn_time_unusual": 0,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 0,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(2.0, 10.0),
                "gradual_escalation_score": rng.uniform(0.6, 1.0),
                "known_device_count": rng.integers(1, 3),
                "account_tenure_days": rng.uniform(30, 180),
                "hour_of_day": int(rng.integers(9, 18)),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "MICRO_FRAUD":
            # Testing stolen cards with tiny amounts
            samples.append({
                "amount_ratio": rng.uniform(0.01, 0.1),
                "txn_freq_last_24h": int(rng.integers(5, 15)),
                "txn_time_unusual": 1,
                "new_device_flag": 1,
                "unusual_location_flag": 1,
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(0.001, 0.1),
                "gradual_escalation_score": 0.0,
                "known_device_count": 1,
                "account_tenure_days": rng.uniform(0.001, 1.0),
                "hour_of_day": int(rng.choice([1, 2, 3, 4])),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "GEO_ANOMALY":
            # Geographic anomaly: impossible travel patterns
            samples.append({
                "amount_ratio": rng.uniform(0.5, 3.0),
                "txn_freq_last_24h": int(rng.integers(2, 6)),
                "txn_time_unusual": rng.choice([0, 1]),
                "new_device_flag": rng.choice([0, 1]),
                "unusual_location_flag": 1,
                "unusual_recipient_flag": rng.choice([0, 1]),
                "failed_auth_count_24h": rng.choice([0, 1]),
                "days_since_last_similar_txn": rng.uniform(0.1, 2.0),
                "gradual_escalation_score": 0.0,
                "known_device_count": rng.integers(1, 3),
                "account_tenure_days": rng.uniform(30, 365),
                "hour_of_day": int(rng.integers(0, 23)),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "TIME_ANOMALY":
            # Time anomaly: transactions at unusual hours
            samples.append({
                "amount_ratio": rng.uniform(0.5, 2.0),
                "txn_freq_last_24h": int(rng.integers(1, 4)),
                "txn_time_unusual": 1,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 0,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(1.0, 7.0),
                "gradual_escalation_score": 0.0,
                "known_device_count": rng.integers(1, 3),
                "account_tenure_days": rng.uniform(30, 365),
                "hour_of_day": int(rng.choice([0, 1, 2, 3, 4, 23])),
                "is_weekend": rng.choice([0, 1]),
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "RECIPIENT_ANOMALY":
            # Recipient anomaly: new payees, unusual destinations
            samples.append({
                "amount_ratio": rng.uniform(0.5, 3.0),
                "txn_freq_last_24h": int(rng.integers(1, 5)),
                "txn_time_unusual": 0,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(2.0, 14.0),
                "gradual_escalation_score": 0.0,
                "known_device_count": rng.integers(1, 3),
                "account_tenure_days": rng.uniform(30, 365),
                "hour_of_day": int(rng.integers(8, 20)),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "AMOUNT_ANOMALY":
            # Amount anomaly: unusual transaction sizes
            samples.append({
                "amount_ratio": rng.uniform(5.0, 15.0),
                "txn_freq_last_24h": int(rng.integers(1, 3)),
                "txn_time_unusual": 0,
                "new_device_flag": 0,
                "unusual_location_flag": 0,
                "unusual_recipient_flag": 0,
                "failed_auth_count_24h": 0,
                "days_since_last_similar_txn": rng.uniform(7.0, 30.0),
                "gradual_escalation_score": 0.0,
                "known_device_count": rng.integers(1, 3),
                "account_tenure_days": rng.uniform(30, 365),
                "hour_of_day": int(rng.integers(9, 18)),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })
        elif category == "BEHAVIORAL_DEVIATION":
            # General behavioral deviation: multiple subtle signals
            samples.append({
                "amount_ratio": rng.uniform(1.5, 4.0),
                "txn_freq_last_24h": int(rng.integers(3, 8)),
                "txn_time_unusual": 1,
                "new_device_flag": 1,
                "unusual_location_flag": 1,
                "unusual_recipient_flag": 1,
                "failed_auth_count_24h": int(rng.integers(1, 4)),
                "days_since_last_similar_txn": rng.uniform(0.5, 3.0),
                "gradual_escalation_score": rng.uniform(0.2, 0.6),
                "known_device_count": 1,
                "account_tenure_days": rng.uniform(10, 180),
                "hour_of_day": int(rng.choice([2, 3, 4, 5])),
                "is_weekend": 0,
                "shared_device_accounts": 0,
                "shared_recipient_accounts": 0,
                "mule_ring_score": 0.0,
            })

    # Add deviation features to all samples
    for s in samples:
        _add_deviation_features(s, rng)

    return samples


def _add_deviation_features(s: dict, rng: np.random.Generator) -> None:
    """Add the 5 new deviation features to a sample dict in-place."""
    typical_hours = list(range(8, 22))
    hour = s["hour_of_day"]
    min_dist = min(abs(hour - h) if abs(hour - h) <= 12 else 24 - abs(hour - h) for h in typical_hours)
    s["hour_deviation"] = round(min(min_dist / 12.0, 1.0), 4)
    # amount_zscore: how unusual is the amount relative to typical
    median_amt = 100.0  # assumed typical
    s["amount_zscore"] = round(float(np.clip((s["amount_ratio"] * median_amt - median_amt) / (median_amt * 0.5), -3.0, 3.0)), 4)
    # velocity_deviation: sigmoid of how unusual the frequency is
    typical_freq = 1.5
    z = (s["txn_freq_last_24h"] - typical_freq) / max(typical_freq ** 0.5, 1.0)
    s["velocity_deviation"] = round(float(1.0 / (1.0 + np.exp(-z))), 4)
    # recipient_novelty: based on unusual_recipient_flag
    s["recipient_novelty"] = float(s["unusual_recipient_flag"])
    # txn_regularity: velocity fraud = very regular (low CV), legit = irregular
    if s["txn_freq_last_24h"] > 10:
        # Many txns in 24h → likely rapid-fire → low regularity score (bot-like)
        s["txn_regularity"] = round(float(rng.uniform(0.1, 0.3)), 4)
    elif s["txn_freq_last_24h"] <= 2:
        # Few txns → can't determine regularity → mid score
        s["txn_regularity"] = round(float(rng.uniform(0.4, 0.6)), 4)
    else:
        s["txn_regularity"] = round(float(rng.uniform(0.5, 0.8)), 4)


def generate_legitimate_samples(
    n_samples: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Generate legitimate transaction feature vectors."""
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples):
        samples.append({
            "amount_ratio": rng.lognormal(0, 0.4),
            "txn_freq_last_24h": int(rng.poisson(1.5)),
            "txn_time_unusual": int(rng.random() < 0.1),
            "new_device_flag": int(rng.random() < 0.08),
            "unusual_location_flag": int(rng.random() < 0.1),
            "unusual_recipient_flag": int(rng.random() < 0.07),
            "failed_auth_count_24h": int(rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 1, 2], p=[0.92, 0, 0, 0, 0, 0, 0, 0, 0.06, 0.02])),
            "days_since_last_similar_txn": rng.exponential(3.0),
            "gradual_escalation_score": rng.beta(1, 8),
            "known_device_count": int(rng.poisson(2)),
            "account_tenure_days": rng.uniform(1, 365),
            "hour_of_day": int(rng.choice(24, p=np.array([0, 0, 0, 0, 0, 0, 1, 3, 8, 12, 14, 12, 10, 10, 11, 12, 14, 16, 18, 15, 9, 4, 1, 0], dtype=float) / 170.0)),
            "is_weekend": int(rng.random() < 0.28),
            "shared_device_accounts": 0,
            "shared_recipient_accounts": 0,
            "mule_ring_score": 0.0,
        })

    # Add deviation features to all legitimate samples
    for s in samples:
        _add_deviation_features(s, rng)

    return samples


def evaluate_model(
    model_path: str = "models/artifacts",
    n_per_category: int = 200,
    n_legitimate: int = 2000,
) -> dict:
    """Evaluate the model against each fraud category."""
    import joblib

    artifacts_dir = Path(ROOT / model_path)
    if not artifacts_dir.exists():
        print(f"ERROR: Model artifacts not found at {artifacts_dir}")
        sys.exit(1)

    # Load model components
    scaler = joblib.load(artifacts_dir / "scaler.joblib")
    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost", "isolation_forest"]:
        path = artifacts_dir / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    stacker = joblib.load(artifacts_dir / "stacker.joblib")
    iso_train = joblib.load(artifacts_dir / "iso_train_scores.joblib")
    cal_path = artifacts_dir / "calibrator.joblib"
    calibrator = joblib.load(cal_path) if cal_path.exists() else None

    # Generate legitimate samples
    legit_samples = generate_legitimate_samples(n_legitimate)
    legit_features = np.array([[s[f] for f in ML_FEATURES] for s in legit_samples])

    # Evaluate each category
    results = {}
    all_scores = []
    all_labels = []

    print("=" * 80)
    print("PS-14 COMPREHENSIVE FRAUD TAXONOMY EVALUATION")
    print("=" * 80)
    print()

    for cat_name, cat_desc in FRAUD_CATEGORIES.items():
        # Generate fraud samples for this category
        fraud_samples = generate_category_samples(cat_name, n_per_category)
        fraud_features = np.array([[s[f] for f in ML_FEATURES] for s in fraud_samples])

        # Combine legitimate + fraud
        X = np.vstack([legit_features, fraud_features])
        y = np.concatenate([np.zeros(len(legit_features)), np.ones(len(fraud_features))])

        # Scale features
        X_scaled = scaler.transform(X)

        # Get predictions from each model
        probs = {}
        if "logistic_regression" in models:
            probs["lr"] = models["logistic_regression"].predict_proba(X_scaled)[:, 1]
        if "random_forest" in models:
            probs["rf"] = models["random_forest"].predict_proba(X)[:, 1]
        if "xgboost" in models:
            probs["xgb"] = models["xgboost"].predict_proba(X)[:, 1]
        if "isolation_forest" in models:
            iso_score = -models["isolation_forest"].decision_function(X_scaled)
            probs["iso"] = np.array([(iso_train < s).mean() for s in iso_score])

        # Stack predictions
        X_stack = np.column_stack([probs.get("lr", np.zeros(len(X))),
                                    probs.get("rf", np.zeros(len(X))),
                                    probs.get("xgb", np.zeros(len(X))),
                                    probs.get("iso", np.zeros(len(X)))])
        raw = stacker.predict_proba(X_stack)[:, 1]

        # Calibrate
        if calibrator is not None:
            scores = calibrator.predict(raw)
        else:
            scores = raw

        # Compute metrics
        try:
            roc_auc = roc_auc_score(y, scores)
        except ValueError:
            roc_auc = 0.5

        try:
            pr_auc = average_precision_score(y, scores)
        except ValueError:
            pr_auc = 0.0

        # Per-category recall at 1% FPR
        fpr, tpr, _ = roc_curve(y, scores)
        recall_at_1fpr = tpr[np.searchsorted(fpr, 0.01)] if len(tpr) > 1 else 0.0

        # Confusion matrix at optimal threshold
        threshold = np.percentile(scores[y == 0], 99)  # 1% FPR threshold
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        results[cat_name] = {
            "description": cat_desc,
            "n_fraud": int(n_per_category),
            "n_legit": int(n_legitimate),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "recall_at_1pct_fpr": round(recall_at_1fpr, 4),
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "detection_rate": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
        }

        all_scores.extend(scores[y == 1].tolist())
        all_labels.extend([1] * int(y.sum()))

        # Print category result
        status = "PASS" if roc_auc > 0.7 else "WEAK" if roc_auc > 0.5 else "FAIL"
        print(f"  [{status}] {cat_name:25s} ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  Recall@1%FPR={recall_at_1fpr:.4f}  Detect={tp}/{tp+fn}")

    # Overall metrics (all fraud categories combined)
    print()
    print("=" * 80)
    print("OVERALL METRICS (all fraud categories combined)")
    print("=" * 80)

    # Rebuild overall dataset
    all_fraud = []
    for cat_name in FRAUD_CATEGORIES:
        all_fraud.extend(generate_category_samples(cat_name, n_per_category))

    fraud_features = np.array([[s[f] for f in ML_FEATURES] for s in all_fraud])
    X_all = np.vstack([legit_features, fraud_features])
    y_all = np.concatenate([np.zeros(len(legit_features)), np.ones(len(fraud_features))])

    X_scaled = scaler.transform(X_all)
    probs = {}
    if "logistic_regression" in models:
        probs["lr"] = models["logistic_regression"].predict_proba(X_scaled)[:, 1]
    if "random_forest" in models:
        probs["rf"] = models["random_forest"].predict_proba(X_all)[:, 1]
    if "xgboost" in models:
        probs["xgb"] = models["xgboost"].predict_proba(X_all)[:, 1]
    if "isolation_forest" in models:
        iso_score = -models["isolation_forest"].decision_function(X_scaled)
        probs["iso"] = np.array([(iso_train < s).mean() for s in iso_score])

    X_stack = np.column_stack([probs.get("lr", np.zeros(len(X_all))),
                                probs.get("rf", np.zeros(len(X_all))),
                                probs.get("xgb", np.zeros(len(X_all))),
                                probs.get("iso", np.zeros(len(X_all)))])
    raw = stacker.predict_proba(X_stack)[:, 1]
    scores = calibrator.predict(raw) if calibrator else raw

    overall_roc = roc_auc_score(y_all, scores)
    overall_pr = average_precision_score(y_all, scores)
    fpr, tpr, _ = roc_curve(y_all, scores)
    overall_recall = tpr[np.searchsorted(fpr, 0.01)] if len(tpr) > 1 else 0.0

    print(f"  Overall ROC-AUC:        {overall_roc:.4f}")
    print(f"  Overall PR-AUC:         {overall_pr:.4f}")
    print(f"  Recall at 1% FPR:       {overall_recall:.4f}")
    print(f"  Total fraud samples:    {len(all_fraud)}")
    print(f"  Total legit samples:    {len(legit_features)}")
    print(f"  Fraud categories:       {len(FRAUD_CATEGORIES)}")
    print()

    # Category breakdown
    print("=" * 80)
    print("CATEGORY BREAKDOWN (sorted by ROC-AUC)")
    print("=" * 80)
    sorted_cats = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
    for cat_name, metrics in sorted_cats:
        bar = "#" * int(metrics["roc_auc"] * 20)
        print(f"  {cat_name:25s} {bar:20s} {metrics['roc_auc']:.4f}  (PR-AUC={metrics['pr_auc']:.4f})")

    print()
    print("=" * 80)
    print("WEAK CATEGORIES (ROC-AUC < 0.7 — need improvement)")
    print("=" * 80)
    weak = [(k, v) for k, v in results.items() if v["roc_auc"] < 0.7]
    if weak:
        for cat_name, metrics in weak:
            print(f"  {cat_name:25s} ROC-AUC={metrics['roc_auc']:.4f} — {metrics['description']}")
    else:
        print("  None — all categories above 0.7 threshold")

    print()
    print("=" * 80)
    print("FAILED CATEGORIES (ROC-AUC < 0.5 — worse than random)")
    print("=" * 80)
    failed = [(k, v) for k, v in results.items() if v["roc_auc"] < 0.5]
    if failed:
        for cat_name, metrics in failed:
            print(f"  {cat_name:25s} ROC-AUC={metrics['roc_auc']:.4f} — {metrics['description']}")
    else:
        print("  None — all categories above random baseline")

    # Save results
    output_path = ROOT / "models" / "fraud_taxonomy_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "overall": {
                "roc_auc": round(overall_roc, 4),
                "pr_auc": round(overall_pr, 4),
                "recall_at_1pct_fpr": round(overall_recall, 4),
                "n_fraud": len(all_fraud),
                "n_legit": len(legit_features),
                "n_categories": len(FRAUD_CATEGORIES),
            },
            "categories": results,
            "categories_description": FRAUD_CATEGORIES,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-category", type=int, default=200)
    parser.add_argument("--n-legitimate", type=int, default=2000)
    parser.add_argument("--model-path", type=str, default="models/artifacts")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.n_per_category, args.n_legitimate)
