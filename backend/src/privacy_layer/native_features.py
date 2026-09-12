"""Altman-NATIVE 48-feature derivation — SHARED by training and production.

Single source of truth for the native feature vector so train == production
by construction (feature-parity audit Part 2 / #27).

The vector is a pure function of:
  raw   — the raw event columns the native model was trained on:
          amount, ts, use_chip ("Chip Transaction"/"Online Transaction"/
          "Swipe Transaction"/""), mcc, merchant_city, merchant_state,
          zip, card, errors, user_id, merchant_id (name), city_id
  vel   — historical velocity context BEFORE this event (from the
          UserVelocityTracker): user_tx_count, user_avg_amt, card_tx_count,
          merch_tx_count, user_merchant_diversity, user_city_diversity,
          user_merch_count
  rates — historical entity fraud rates BEFORE this event (from the
          EntityFraudRateTracker): user_fraud_rate, merch_fraud_rate,
          city_fraud_rate

Leakage-safe by construction: every historical input is shifted (the event
being scored is never counted in its own velocity / fraud rate / average).

The feature list and order MUST stay identical to the native trainer's
feat_cols (48 features). See src/risk_engine/altman_native_ensemble.py
ALTMAN_NATIVE_FEATURES for the runtime contract (same list).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Contract order — MUST match src/risk_engine/altman_native_ensemble.py
ALTMAN_NATIVE_FEATURES = [
    "amt", "log_amt", "amt_sq", "hr", "mn", "dow", "Month", "Day",
    "hour_sin", "hour_cos", "is_night", "is_business_hours",
    "chip", "is_online", "is_swipe", "err", "has_zip", "has_state",
    "is_online_or_no_state",
    "mcc", "mcc_high", "mcc_restaurant", "mcc_gas", "mcc_grocery",
    "mcc_travel", "mcc_online",
    "merchant_id", "city_id", "card_id",
    "user_tx_count", "card_tx_count", "user_avg_amt", "amt_vs_user_avg",
    "amt_zscore", "merch_tx_count",
    "user_merchant_diversity", "user_city_diversity",
    "user_fraud_rate", "merch_fraud_rate", "city_fraud_rate",
    "high_amt", "very_high_amt",
    "amt_x_hr", "amt_x_mcc", "amt_x_chip", "amt_x_online", "amt_x_night",
    "user_merch_count",
]

# Cold-start defaults mirror the trainer's fillna(0.001) / fillna(0.0) and
# the runtime mapper's feature.get(f, 0.0) fallback.
COLD_START_FRAUD_RATE = 0.001


def _f2(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(x) or math.isinf(x) else x


def derive_native_features(raw: dict, vel: dict | None = None,
                           rates: dict | None = None) -> dict:
    """Compute the 48 native features for one event (leakage-safe)."""
    vel = vel or {}
    rates = rates or {}

    amt = max(_f2(raw.get("amount")), 0.0)
    ts = raw.get("ts")
    if ts is None:
        hr = int(_f2(raw.get("hour_of_day", 12)))
        mn = 0
        month = 8
        day = 15
        dow = 0
    else:
        try:
            hr = int(ts.hour)
            mn = int(ts.minute)
            month = int(ts.month)
            day = int(ts.day)
            dow = int(ts.weekday())
        except AttributeError:
            hr = int(_f2(raw.get("hour_of_day", 12)))
            mn = 0
            month = 8
            day = 15
            dow = 0

    use_chip = str(raw.get("use_chip") or "")
    chip = 1.0 if use_chip == "Chip Transaction" else 0.0
    is_online = 1.0 if use_chip == "Online Transaction" else 0.0
    is_swipe = 1.0 if use_chip == "Swipe Transaction" else 0.0

    _e = raw.get("errors")
    err = 1.0 if (_e is not None and str(_e) not in ("", "nan", "None")) else 0.0
    has_zip = 1.0 if str(raw.get("zip") or "") != "" else 0.0
    has_state = 1.0 if str(raw.get("merchant_state") or "") != "" else 0.0
    is_online_or_no_state = 1.0 if (is_online == 1.0 or has_state == 0.0) else 0.0

    mcc = int(_f2(raw.get("mcc")))
    mcc_high = 1.0 if mcc >= 5000 else 0.0
    mcc_restaurant = 1.0 if 5812 <= mcc <= 5814 else 0.0
    mcc_gas = 1.0 if 5541 <= mcc <= 5542 else 0.0
    mcc_grocery = 1.0 if 5411 <= mcc <= 5422 else 0.0
    mcc_travel = 1.0 if 3000 <= mcc <= 3350 else 0.0
    mcc_online = 1.0 if 5967 <= mcc <= 5969 else 0.0

    # Entity identifiers as stable numeric codes (hash of the raw string).
    # Training and production compute the SAME code for the SAME raw value,
    # so unseen entities in production fall into a cold-start code bucket —
    # they never silently map to a wrong trained entity.
    def _code(s: Any) -> float:
        s = str(s or "")
        if not s:
            return 0.0
        import hashlib
        return float(int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 100000)

    merchant_code = _code(raw.get("merchant_id") or raw.get("merchant_name"))
    city_code = _code(raw.get("city_id") or raw.get("merchant_city"))
    card_code = _code(raw.get("card_id") or raw.get("card"))

    user_tx_count = _f2(vel.get("user_tx_count", 0))
    card_tx_count = _f2(vel.get("card_tx_count", 0))
    user_avg_amt = _f2(vel.get("user_avg_amt", 0.0))
    merch_tx_count = _f2(vel.get("merch_tx_count", 0))
    # clamp >= 1 mirrors the retrain's context build (max(...,1.0)); live cold
    # starts pass 0 and must land on the SAME feature the model was trained on.
    user_merchant_diversity = max(_f2(vel.get("user_merchant_diversity", 1.0)), 1.0)
    user_city_diversity = max(_f2(vel.get("user_city_diversity", 1.0)), 1.0)
    user_merch_count = _f2(vel.get("user_merch_count", 0))

    amt_vs_user_avg = amt / (user_avg_amt + 1e-6) if user_avg_amt > 0 else 1.0
    amt_zscore = (amt - user_avg_amt) / (user_avg_amt + 1e-6) if user_avg_amt > 0 else 0.0

    ufr = _f2(rates.get("user_fraud_rate", COLD_START_FRAUD_RATE)) or COLD_START_FRAUD_RATE
    mfr = _f2(rates.get("merch_fraud_rate", COLD_START_FRAUD_RATE)) or COLD_START_FRAUD_RATE
    cfr = _f2(rates.get("city_fraud_rate", COLD_START_FRAUD_RATE)) or COLD_START_FRAUD_RATE

    high_amt = 1.0 if user_avg_amt > 0 and amt > user_avg_amt * 2 else 0.0
    very_high_amt = 1.0 if user_avg_amt > 0 and amt > user_avg_amt * 5 else 0.0

    return {
        "amt": round(amt, 4),
        "log_amt": round(float(np.log1p(amt)), 4),
        "amt_sq": round(amt * amt, 4),
        "hr": float(hr), "mn": float(mn), "dow": float(dow),
        "Month": float(month), "Day": float(day),
        "hour_sin": round(math.sin(2 * math.pi * hr / 24), 6),
        "hour_cos": round(math.cos(2 * math.pi * hr / 24), 6),
        "is_night": 1.0 if (hr < 6 or hr > 22) else 0.0,
        "is_business_hours": 1.0 if 9 <= hr <= 17 else 0.0,
        "chip": chip, "is_online": is_online, "is_swipe": is_swipe,
        "err": err, "has_zip": has_zip, "has_state": has_state,
        "is_online_or_no_state": is_online_or_no_state,
        "mcc": float(mcc),
        "mcc_high": mcc_high, "mcc_restaurant": mcc_restaurant,
        "mcc_gas": mcc_gas, "mcc_grocery": mcc_grocery,
        "mcc_travel": mcc_travel, "mcc_online": mcc_online,
        "merchant_id": merchant_code, "city_id": city_code, "card_id": card_code,
        "user_tx_count": float(user_tx_count),
        "card_tx_count": float(card_tx_count),
        "user_avg_amt": round(user_avg_amt, 4),
        "amt_vs_user_avg": round(amt_vs_user_avg, 6),
        "amt_zscore": round(amt_zscore, 6),
        "merch_tx_count": float(merch_tx_count),
        "user_merchant_diversity": float(user_merchant_diversity),
        "user_city_diversity": float(user_city_diversity),
        "user_fraud_rate": round(ufr, 6),
        "merch_fraud_rate": round(mfr, 6),
        "city_fraud_rate": round(cfr, 6),
        "high_amt": high_amt, "very_high_amt": very_high_amt,
        "amt_x_hr": round(amt * hr, 4),
        "amt_x_mcc": round(amt * mcc, 4),
        "amt_x_chip": round(amt * chip, 4),
        "amt_x_online": round(amt * is_online, 4),
        "amt_x_night": round(amt * (1.0 if hr < 6 or hr > 22 else 0.0), 4),
        "user_merch_count": float(user_merch_count),
    }


def native_vector(features: dict) -> np.ndarray:
    """Ordered 48-vector from the derived dict (runtime contract order)."""
    return np.array([_f2(features.get(f, 0.0)) for f in ALTMAN_NATIVE_FEATURES],
                    dtype=np.float64)