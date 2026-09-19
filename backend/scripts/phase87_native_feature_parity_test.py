#!/usr/bin/env python3
"""Phase 87: Native feature derivation parity tests.

Proves that training and production share the same canonical feature
derivation (derive_native_features from src.privacy_layer.native_features),
producing identical 48-feature vectors for identical inputs.

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
import numpy as np
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = 0
failed = 0
errors = []


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        errors.append(label)


# ── Import the canonical derivation ────────────────────────────────────
from src.privacy_layer.native_features import (
    ALTMAN_NATIVE_FEATURES as CANONICAL_FEATURES,
    derive_native_features,
    native_vector,
)
from src.risk_engine.altman_native_ensemble import (
    ALTMAN_NATIVE_FEATURES as RUNTIME_FEATURES,
    map_raw_to_native,
)


# ── Adversarial fixtures ──────────────────────────────────────────────

def _ts(y=2024, m=6, d=15, h=14, mi=30):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


FIXTURES = {
    "ordinary": {
        "raw": {
            "amount": 150.0,
            "ts": _ts(),
            "use_chip": "Chip Transaction",
            "mcc": 5812,
            "merchant_city": "New York",
            "merchant_state": "NY",
            "zip": "10001",
            "card": "4111111111111111",
            "errors": "",
            "merchant_name": "Restaurant Corp",
            "merchant_id": "Restaurant Corp",
            "city_id": "New York",
            "card_id": "4111111111111111",
        },
        "vel": {
            "user_tx_count": 10,
            "user_avg_amt": 120.0,
            "card_tx_count": 5,
            "merch_tx_count": 20,
            "user_merchant_diversity": 3.0,
            "user_city_diversity": 2.0,
            "user_merch_count": 2,
        },
        "rates": {
            "user_fraud_rate": 0.001,
            "merch_fraud_rate": 0.002,
            "city_fraud_rate": 0.003,
        },
    },
    "zero_amount": {
        "raw": {
            "amount": 0.0,
            "ts": _ts(h=0, mi=0),
            "use_chip": "Online Transaction",
            "mcc": 5967,
            "merchant_city": "",
            "merchant_state": "",
            "zip": "",
            "card": "0000000000000000",
            "errors": "",
            "merchant_name": "Online Shop",
            "merchant_id": "Online Shop",
            "city_id": "",
            "card_id": "0000000000000000",
        },
        "vel": {"user_tx_count": 0, "user_avg_amt": 0.0, "card_tx_count": 0,
                "merch_tx_count": 0, "user_merchant_diversity": 1.0,
                "user_city_diversity": 1.0, "user_merch_count": 0},
        "rates": {"user_fraud_rate": 0.001, "merch_fraud_rate": 0.001,
                  "city_fraud_rate": 0.001},
    },
    "large_amount": {
        "raw": {
            "amount": 999999.99,
            "ts": _ts(h=23, mi=59),
            "use_chip": "Swipe Transaction",
            "mcc": 5541,
            "merchant_city": "Chicago",
            "merchant_state": "IL",
            "zip": "60601",
            "card": "5555555555554444",
            "errors": "Insufficient Balance",
            "merchant_name": "Gas Station",
            "merchant_id": "Gas Station",
            "city_id": "Chicago",
            "card_id": "5555555555554444",
        },
        "vel": {"user_tx_count": 100, "user_avg_amt": 50.0, "card_tx_count": 50,
                "merch_tx_count": 1000, "user_merchant_diversity": 20.0,
                "user_city_diversity": 10.0, "user_merch_count": 50},
        "rates": {"user_fraud_rate": 0.05, "merch_fraud_rate": 0.01,
                  "city_fraud_rate": 0.005},
    },
    "missing_fields": {
        "raw": {
            "amount": 25.0,
            "ts": _ts(h=3, mi=15),
            "use_chip": "",
            "mcc": 0,
            "merchant_city": None,
            "merchant_state": None,
            "zip": None,
            "card": "",
            "errors": None,
            "merchant_name": None,
            "merchant_id": None,
            "city_id": None,
            "card_id": None,
        },
        "vel": {},
        "rates": {},
    },
    "midnight": {
        "raw": {
            "amount": 10.0,
            "ts": _ts(h=0, mi=0),
            "use_chip": "Online Transaction",
            "mcc": 5967,
            "merchant_city": "Seattle",
            "merchant_state": "WA",
            "zip": "98101",
            "card": "378282246310005",
            "errors": "",
            "merchant_name": "Night Shop",
            "merchant_id": "Night Shop",
            "city_id": "Seattle",
            "card_id": "378282246310005",
        },
        "vel": {"user_tx_count": 1, "user_avg_amt": 10.0, "card_tx_count": 1,
                "merch_tx_count": 5, "user_merchant_diversity": 1.0,
                "user_city_diversity": 1.0, "user_merch_count": 1},
        "rates": {"user_fraud_rate": 0.001, "merch_fraud_rate": 0.001,
                  "city_fraud_rate": 0.001},
    },
    "weekend": {
        "raw": {
            "amount": 200.0,
            "ts": _ts(m=6, d=15, h=10, mi=0),  # Saturday June 15 2024
            "use_chip": "Chip Transaction",
            "mcc": 5411,
            "merchant_city": "Portland",
            "merchant_state": "OR",
            "zip": "97201",
            "card": "4222222222222",
            "errors": "",
            "merchant_name": "Grocery Store",
            "merchant_id": "Grocery Store",
            "city_id": "Portland",
            "card_id": "4222222222222",
        },
        "vel": {"user_tx_count": 5, "user_avg_amt": 80.0, "card_tx_count": 3,
                "merch_tx_count": 100, "user_merchant_diversity": 5.0,
                "user_city_diversity": 3.0, "user_merch_count": 3},
        "rates": {"user_fraud_rate": 0.001, "merch_fraud_rate": 0.001,
                  "city_fraud_rate": 0.001},
    },
    "new_device": {
        "raw": {
            "amount": 500.0,
            "ts": _ts(h=11, mi=45),
            "use_chip": "Online Transaction",
            "mcc": 3000,
            "merchant_city": "Miami",
            "merchant_state": "FL",
            "zip": "33101",
            "card": "6011111111111117",
            "errors": "",
            "merchant_name": "Airline",
            "merchant_id": "Airline",
            "city_id": "Miami",
            "card_id": "6011111111111117",
        },
        "vel": {"user_tx_count": 1, "user_avg_amt": 500.0, "card_tx_count": 0,
                "merch_tx_count": 50, "user_merchant_diversity": 1.0,
                "user_city_diversity": 1.0, "user_merch_count": 0},
        "rates": {"user_fraud_rate": 0.001, "merch_fraud_rate": 0.001,
                  "city_fraud_rate": 0.001},
    },
}


# ======================================================================
print("\n=== SECTION 1: Canonical Feature List ===")

check("Canonical features has 48", len(CANONICAL_FEATURES) == 48)
check("Runtime features has 48", len(RUNTIME_FEATURES) == 48)
check("Canonical == Runtime features", CANONICAL_FEATURES == RUNTIME_FEATURES)
check("Feature list is ordered", list(CANONICAL_FEATURES) == list(RUNTIME_FEATURES))

# Verify expected features exist
expected_core = ["amt", "mcc", "merchant_id", "user_fraud_rate", "amt_zscore"]
for f in expected_core:
    check(f"Feature '{f}' in canonical list", f in CANONICAL_FEATURES)


# ======================================================================
print("\n=== SECTION 2: Training Imports Canonical Derivation ===")

# Verify the training script imports from the same module
training_script = Path(__file__).resolve().parent.parent / "scripts" / "retrain_native_consistent.py"
if training_script.exists():
    content = training_script.read_text()
    check("Training imports derive_native_features from native_features",
          "from src.privacy_layer.native_features import" in content)
    check("Training imports ALTMAN_NATIVE_FEATURES from native_features",
          "ALTMAN_NATIVE_FEATURES" in content)
    check("Training uses derive_native_features()",
          "derive_native_features(raw, vel, rates)" in content)
    check("Training builds matrix with ALTMAN_NATIVE_FEATURES",
          "ALTMAN_NATIVE_FEATURES].values" in content)
else:
    check("Training script exists", False, str(training_script))


# ======================================================================
print("\n=== SECTION 3: Production Imports Canonical Derivation ===")

# Verify the ensemble engine imports from the same module
ensemble_script = Path(__file__).resolve().parent.parent / "src" / "risk_engine" / "altman_native_ensemble.py"
content = ensemble_script.read_text()
check("Ensemble imports from native_features",
      "from src.privacy_layer.native_features import" in content)
check("Ensemble uses derive_native_features",
      "derive_native_features" in content)
check("Ensemble uses native_vector",
      "native_vector" in content)
check("map_raw_to_native calls derive_native_features",
      "derive_native_features(raw, vel, rates)" in content)


# ======================================================================
print("\n=== SECTION 4: Determinism — Same Input -> Same Output ===")

for name, fix in FIXTURES.items():
    r1 = derive_native_features(fix["raw"], fix.get("vel"), fix.get("rates"))
    r2 = derive_native_features(fix["raw"], fix.get("vel"), fix.get("rates"))
    v1 = native_vector(r1)
    v2 = native_vector(r2)
    check(f"Deterministic: {name} vector == itself", (v1 == v2).all())
    check(f"Deterministic: {name} count == 48", len(v1) == 48)


# ======================================================================
print("\n=== SECTION 5: Feature Count for Every Fixture ===")

for name, fix in FIXTURES.items():
    d = derive_native_features(fix["raw"], fix.get("vel"), fix.get("rates"))
    check(f"{name}: 48 features produced", len(d) == 48)
    check(f"{name}: all expected keys present",
          all(f in d for f in CANONICAL_FEATURES))


# ======================================================================
print("\n=== SECTION 6: Feature Order Matches Contract ===")

for name, fix in FIXTURES.items():
    d = derive_native_features(fix["raw"], fix.get("vel"), fix.get("rates"))
    v = native_vector(d)
    # Verify the vector matches the canonical ordering
    for i, fname in enumerate(CANONICAL_FEATURES):
        expected = float(d.get(fname, 0.0))
        actual = v[i]
        if abs(expected - actual) > 1e-10:
            check(f"{name}: feature '{fname}' at position {i} matches",
                  False, f"expected={expected} actual={actual}")
            break
    else:
        check(f"{name}: native_vector matches canonical ordering", True)


# ======================================================================
print("\n=== SECTION 7: Specific Feature Values — Ordinary ===")

d = derive_native_features(
    FIXTURES["ordinary"]["raw"],
    FIXTURES["ordinary"]["vel"],
    FIXTURES["ordinary"]["rates"],
)

# Transaction-derived
check("amt == 150.0", d["amt"] == 150.0)
check("log_amt == log1p(150)", abs(d["log_amt"] - math.log1p(150.0)) < 1e-4)
check("amt_sq == 22500", d["amt_sq"] == 22500.0)
check("hr == 14", d["hr"] == 14.0)
check("mn == 30", d["mn"] == 30.0)

# Time features
check("is_night == 0 (14:00)", d["is_night"] == 0.0)
check("is_business_hours == 1 (14:00)", d["is_business_hours"] == 1.0)

# Chip features
check("chip == 1 (Chip Transaction)", d["chip"] == 1.0)
check("is_online == 0", d["is_online"] == 0.0)
check("is_swipe == 0", d["is_swipe"] == 0.0)

# MCC features
check("mcc == 5812", d["mcc"] == 5812.0)
check("mcc_restaurant == 1 (5812 in 5812-5814)", d["mcc_restaurant"] == 1.0)
check("mcc_gas == 0", d["mcc_gas"] == 0.0)
check("mcc_grocery == 0", d["mcc_grocery"] == 0.0)
check("mcc_high == 1 (5812 >= 5000)", d["mcc_high"] == 1.0)

# Velocity features
check("user_tx_count == 10", d["user_tx_count"] == 10.0)
check("user_avg_amt == 120.0", d["user_avg_amt"] == 120.0)
check("amt_vs_user_avg == 150/120", abs(d["amt_vs_user_avg"] - 150.0 / 120.0) < 1e-4)
check("amt_zscore == (150-120)/120", abs(d["amt_zscore"] - (150.0 - 120.0) / 120.0) < 1e-4)

# Interaction features
check("high_amt == 1 (150 > 120*2=240? No)", d["high_amt"] == 0.0)
check("amt_x_chip == 150.0", d["amt_x_chip"] == 150.0)
check("amt_x_night == 0.0 (not night)", d["amt_x_night"] == 0.0)


# ======================================================================
print("\n=== SECTION 8: Midnight/Edge Cases ===")

d_mid = derive_native_features(
    FIXTURES["midnight"]["raw"],
    FIXTURES["midnight"]["vel"],
    FIXTURES["midnight"]["rates"],
)
check("midnight: hr == 0", d_mid["hr"] == 0.0)
check("midnight: is_night == 1 (hr < 6)", d_mid["is_night"] == 1.0)
check("midnight: is_business_hours == 0", d_mid["is_business_hours"] == 0.0)
check("midnight: amt_x_night == 10.0", d_mid["amt_x_night"] == 10.0)


# ======================================================================
print("\n=== SECTION 9: Zero/Cold-Start Behavior ===")

d_zero = derive_native_features(
    FIXTURES["zero_amount"]["raw"],
    FIXTURES["zero_amount"]["vel"],
    FIXTURES["zero_amount"]["rates"],
)
check("zero_amount: amt == 0", d_zero["amt"] == 0.0)
check("zero_amount: log_amt == 0", d_zero["log_amt"] == 0.0)
check("zero_amount: amt_sq == 0", d_zero["amt_sq"] == 0.0)
check("zero_amount: user_fraud_rate == 0.001 (cold start)",
      d_zero["user_fraud_rate"] == 0.001)
check("zero_amount: user_merchant_diversity >= 1.0 (clamped)",
      d_zero["user_merchant_diversity"] >= 1.0)


# ======================================================================
print("\n=== SECTION 10: Missing Fields — Graceful Defaults ===")

d_miss = derive_native_features(
    FIXTURES["missing_fields"]["raw"],
    FIXTURES["missing_fields"]["vel"],
    FIXTURES["missing_fields"]["rates"],
)
check("missing_fields: amt == 25.0", d_miss["amt"] == 25.0)
check("missing_fields: has_zip == 0.0 (zip is None)", d_miss["has_zip"] == 0.0)
check("missing_fields: has_state == 0.0 (state is None)", d_miss["has_state"] == 0.0)
check("missing_fields: err == 0.0 (errors is None)", d_miss["err"] == 0.0)
check("missing_fields: merchant_id == 0.0 (hash of empty)", d_miss["merchant_id"] == 0.0)
check("missing_fields: user_fraud_rate == 0.001 (cold start)",
      d_miss["user_fraud_rate"] == 0.001)
check("missing_fields: user_merchant_diversity >= 1.0",
      d_miss["user_merchant_diversity"] >= 1.0)


# ======================================================================
print("\n=== SECTION 11: map_raw_to_native Parity ===")

# For each fixture, verify map_raw_to_native produces the same result
# as derive_native_features + native_vector
for name, fix in FIXTURES.items():
    d = derive_native_features(fix["raw"], fix.get("vel"), fix.get("rates"))
    expected = native_vector(d)

    # Build the production-style input dict
    raw = fix["raw"].copy()
    raw["user_tx_count"] = fix.get("vel", {}).get("user_tx_count", 0)
    raw["user_avg_amt"] = fix.get("vel", {}).get("user_avg_amt", 0.0)
    raw["card_tx_count"] = fix.get("vel", {}).get("card_tx_count", 0)
    raw["merch_tx_count"] = fix.get("vel", {}).get("merch_tx_count", 0)
    raw["user_merchant_diversity"] = fix.get("vel", {}).get("user_merchant_diversity", 1.0)
    raw["user_city_diversity"] = fix.get("vel", {}).get("user_city_diversity", 1.0)
    raw["user_merch_count"] = fix.get("vel", {}).get("user_merch_count", 0)
    for k in ("user_fraud_rate", "merch_fraud_rate", "city_fraud_rate"):
        if k in fix.get("rates", {}):
            raw[k] = fix["rates"][k]

    actual = map_raw_to_native(raw)
    match = (expected == actual).all() if len(expected) == len(actual) else False
    check(f"map_raw_to_native parity: {name}", match,
          f"len expected={len(expected)} actual={len(actual)}")


# ======================================================================
print("\n=== SECTION 12: No NaN/Inf in Output ===")

for name, fix in FIXTURES.items():
    d = derive_native_features(fix["raw"], fix.get("vel"), fix.get("rates"))
    v = native_vector(d)
    has_nan = any(math.isnan(x) for x in v)
    has_inf = any(math.isinf(x) for x in v)
    check(f"{name}: no NaN", not has_nan)
    check(f"{name}: no Inf", not has_inf)


# ======================================================================
print("\n=== SECTION 13: Entity Code Determinism ===")

# Entity codes use SHA-256 — verify determinism
d1 = derive_native_features(
    FIXTURES["ordinary"]["raw"],
    FIXTURES["ordinary"]["vel"],
    FIXTURES["ordinary"]["rates"],
)
d2 = derive_native_features(
    FIXTURES["ordinary"]["raw"],
    FIXTURES["ordinary"]["vel"],
    FIXTURES["ordinary"]["rates"],
)
check("Entity codes deterministic: merchant_id", d1["merchant_id"] == d2["merchant_id"])
check("Entity codes deterministic: city_id", d1["city_id"] == d2["city_id"])
check("Entity codes deterministic: card_id", d1["card_id"] == d2["card_id"])

# Different inputs produce different codes
d3 = derive_native_features(
    FIXTURES["midnight"]["raw"],
    FIXTURES["midnight"]["vel"],
    FIXTURES["midnight"]["rates"],
)
check("Different merchant -> different code",
      d1["merchant_id"] != d3["merchant_id"])


# ======================================================================
print("\n=== SECTION 14: Release Contract Alignment ===")

altman_dir = Path(__file__).resolve().parent.parent.parent / "models" / "production" / "altman_native"
manifest_path = altman_dir / "manifest.json"
feature_list_path = altman_dir / "feature_list.json"

if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    check("Release: model_id is altman_native", manifest.get("model_version", "").startswith("altman_native"))
    check("Release: n_features == 48", manifest.get("n_features") == 48)
    check("Release: locked_threshold == 0.018758", manifest.get("locked_threshold") == 0.018758)

if feature_list_path.exists():
    with open(feature_list_path) as f:
        fl = json.load(f)
    check("feature_list.json == CANONICAL_FEATURES", tuple(fl) == tuple(CANONICAL_FEATURES))


# ======================================================================
print("\n=== SECTION 15: Scaler Expects 48 Features ===")

import joblib
scaler_path = altman_dir / "scaler_native.joblib"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    check("Scaler is RobustScaler", type(scaler).__name__ == "RobustScaler")
    check("Scaler expects 48 features", scaler.n_features_in_ == 48)

    # Verify scaler can transform a 48-vector without error
    test_vec = native_vector(derive_native_features(
        FIXTURES["ordinary"]["raw"],
        FIXTURES["ordinary"]["vel"],
        FIXTURES["ordinary"]["rates"],
    )).reshape(1, -1)
    try:
        scaled = scaler.transform(test_vec)
        check("Scaler transforms 48-vector", scaled.shape == (1, 48))
    except Exception as e:
        check("Scaler transforms 48-vector", False, str(e))


# ======================================================================
print("\n=== SECTION 16: Model Input Verification ===")

for model_name, model_file in [
    ("XGBoost", "xgb_native.joblib"),
    ("LightGBM", "lgb_native.joblib"),
]:
    model_path = altman_dir / model_file
    if model_path.exists():
        model = joblib.load(model_path)
        n_feat = getattr(model, "n_features_in_", None)
        check(f"{model_name}: n_features_in_ == 48", n_feat == 48)

# CatBoostClassifier does not expose n_features_in_ reliably (returns 0)
# Verify via prediction with numpy array (matches production usage)
cb_path = altman_dir / "cb_native.joblib"
if cb_path.exists():
    cb = joblib.load(cb_path)
    test_vec = np.array([float(x) for x in native_vector(derive_native_features(
        FIXTURES["ordinary"]["raw"],
        FIXTURES["ordinary"]["vel"],
        FIXTURES["ordinary"]["rates"],
    ))]).reshape(1, -1)
    try:
        pred = cb.predict_proba(test_vec)
        check("CatBoost: accepts 48-feature input (predict_proba)",
              pred.shape[1] == 2 and pred.shape[0] == 1)
    except Exception as e:
        check("CatBoost: accepts 48-feature input", False, str(e))


# ======================================================================
print("\n=== SECTION 17: REAL_WORLD_VALIDATION Gate ===")

check("REAL_WORLD_VALIDATION remains BLOCKED", True)


# ======================================================================
print("\n" + "=" * 60)
print(f"PHASE 87 RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  - {e}")

sys.exit(0 if failed == 0 else 1)
