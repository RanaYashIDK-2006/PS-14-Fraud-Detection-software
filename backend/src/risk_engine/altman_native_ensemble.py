"""Altman-NATIVE Ensemble Engine — full feature set from raw columns.

Unlike the mapped ensemble (which proxies ML_FEATURES → 35 Altman features),
this engine accepts the FULL 48-feature Altman-native feature set directly:
  - Raw MCC codes and MCC category flags
  - Entity IDs encoded as integer codes
  - User merchant/city diversity counts
  - Per-user merchant interaction count

Deployed alongside the mapped ensemble for A/B comparison.

Architecture:
  Raw Altman features → RobustScaler → [XGB, LGB, CB] → weighted avg → prob
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
NATIVE_DIR = ROOT / "models" / "production" / "altman_native"

# The 48 native Altman features
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

ENSEMBLE_WEIGHTS = {"xgb": 0.34, "lgb": 0.33, "cb": 0.33}


def map_raw_to_native(features: dict, entity_tracker=None) -> np.ndarray:
    """Map raw Altman features to the 48-feature native vector.

    Accepts (in priority order):
    1. A dict with RAW native columns (amount/ts/use_chip/mcc/merchant_*
       /zip/card/errors) — production path: the SHARED derivation module
       (src/privacy_layer.native_features) computes the 48-vector exactly
       as the retrain did, so train == production by construction
       (feature-parity audit Part 2 / #27).
    2. A dict with native feature keys directly (already-derived vector).
    3. A dict with ML_FEATURES keys + entity IDs (legacy proxy path,
       documented fallback — raw columns unknown).
    """
    # 1. Raw native columns present -> shared derivation (parity path)
    if "amount" in features or "use_chip" in features or "mcc" in features:
        return _from_raw_native(features, entity_tracker)

    # 2. Already-derived native vector
    if "amt" in features and "mcc" in features and "merchant_id" in features:
        return _from_native_dict(features, entity_tracker)

    # 3. Legacy ML_FEATURES proxy
    return _from_ml_features(features, entity_tracker)


def _from_raw_native(features: dict, entity_tracker=None) -> np.ndarray:
    """Shared-derivation path: raw columns -> 48-vector (parity with retrain)."""
    from src.privacy_layer.native_features import derive_native_features, native_vector

    vel = {
        "user_tx_count": features.get("user_tx_count", 0),
        "user_avg_amt": features.get("user_avg_amt", 0.0),
        "card_tx_count": features.get("card_tx_count", 0),
        "merch_tx_count": features.get("merch_tx_count", 0),
        "user_merchant_diversity": features.get("user_merchant_diversity", 1.0),
        "user_city_diversity": features.get("user_city_diversity", 1.0),
        "user_merch_count": features.get("user_merch_count", 0),
    }
    rates = {}
    for k in ("user_fraud_rate", "merch_fraud_rate", "city_fraud_rate"):
        if k in features:
            rates[k] = features[k]
    if len(rates) < 3 and entity_tracker is not None:
        try:
            r = entity_tracker.get_rates(
                user_id=str(features.get("user_id", "")),
                merchant_id=str(features.get("merchant_id", "")),
                city_id=str(features.get("city_id", "")),
            )
            rates.update(r)
        except Exception:
            pass
    ts = features.get("ts")
    if isinstance(ts, str) and ts:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            ts = None
    _amt = features.get("amount")
    if _amt is None or _amt == "":
        _amt = features.get("amount_ratio", 1.0) * 100.0
    raw = {
        "amount": _amt,  # 0.0 is a legit amount — never replaced by the ratio default
        "ts": ts,
        "hour_of_day": features.get("hour_of_day", 12),
        "use_chip": features.get("use_chip", ""),
        "mcc": features.get("mcc", 0),
        "merchant_city": features.get("merchant_city", ""),
        "merchant_state": features.get("merchant_state", ""),
        "zip": features.get("zip", ""),
        "card": features.get("card", ""),
        "errors": features.get("errors", ""),
        "merchant_name": features.get("merchant_id", ""),
        "merchant_id": features.get("merchant_id", ""),
        "city_id": features.get("city_id", ""),
        "card_id": features.get("card_id", ""),
    }
    return native_vector(derive_native_features(raw, vel, rates))


def _from_native_dict(features: dict, entity_tracker=None) -> np.ndarray:
    """Build vector from a dict that already has native feature keys."""
    vec = []
    for f in ALTMAN_NATIVE_FEATURES:
        val = features.get(f, 0.0)
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        vec.append(float(val))
    return np.array(vec, dtype=np.float64)


def _from_ml_features(features: dict, entity_tracker=None) -> np.ndarray:
    """Map ML_FEATURES dict to native 48-feature vector."""
    amt_ratio = float(features.get("amount_ratio", 1.0))
    hour_of_day = float(features.get("hour_of_day", 12.0))
    is_weekend = int(features.get("is_weekend", 0))
    chip = int(features.get("new_device_flag", 0))
    txn_freq = float(features.get("txn_freq_last_24h", 5))
    known_devices = float(features.get("known_device_count", 5))
    failed_auth = float(features.get("failed_auth_count_24h", 0))
    amt_zscore_ml = float(features.get("amount_zscore", 0.0))

    ref_amt = 100.0
    amt = amt_ratio * ref_amt
    hr = hour_of_day
    mn = 0.0

    # Entity fraud rates
    user_id = features.get("user_id", "")
    merchant_id = features.get("merchant_id", "")
    city_id = features.get("city_id", "")
    user_fraud_rate = features.get("user_fraud_rate", None)
    merch_fraud_rate = features.get("merch_fraud_rate", None)
    city_fraud_rate = features.get("city_fraud_rate", None)

    if user_fraud_rate is None or merch_fraud_rate is None or city_fraud_rate is None:
        try:
            from src.risk_engine.entity_fraud_rates import get_tracker
            tracker = entity_tracker or get_tracker()
            rates = tracker.get_rates(user_id, merchant_id, city_id)
            user_fraud_rate = user_fraud_rate or rates["user_fraud_rate"]
            merch_fraud_rate = merch_fraud_rate or rates["merch_fraud_rate"]
            city_fraud_rate = city_fraud_rate or rates["city_fraud_rate"]
        except Exception:
            user_fraud_rate = user_fraud_rate or 0.001
            merch_fraud_rate = merch_fraud_rate or 0.001
            city_fraud_rate = city_fraud_rate or 0.001

    vec = np.array([
        amt, np.log1p(amt), amt ** 2, hr, mn,
        float(is_weekend), 8.0, 15.0,  # Month, Day defaults
        np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
        1.0 if hr < 6 or hr > 22 else 0.0,
        1.0 if 9 <= hr <= 17 else 0.0,
        float(chip), float(chip), 0.0,  # is_online, is_swipe
        1.0 if failed_auth > 0 else 0.0,
        0.0, 0.0, 0.0,  # has_zip, has_state, is_online_or_no_state
        0.0,  # mcc
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # MCC category flags
        0.0, 0.0, 0.0,  # merchant_id, city_id, card_id (unknown in ML_FEATURES)
        txn_freq, txn_freq * 0.8, ref_amt, amt_ratio,
        amt_zscore_ml, known_devices,
        known_devices * 0.8, txn_freq * 0.5,  # merchant/city diversity proxies
        float(user_fraud_rate), float(merch_fraud_rate), float(city_fraud_rate),
        1.0 if amt_ratio > 2.0 else 0.0,
        1.0 if amt_ratio > 5.0 else 0.0,
        amt * hr, amt * 0.0, amt * chip, amt * chip, amt * (1.0 if hr < 6 or hr > 22 else 0.0),
        0.0,  # user_merch_count
    ], dtype=np.float64)

    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec


class AltmanNativeEnsembleEngine:
    """48-feature Altman-native ensemble — drop-in FusionEngine replacement."""

    def __init__(self, model_dir: Path = None):
        if model_dir is None:
            model_dir = NATIVE_DIR
        self._model_dir = model_dir

        manifest_path = model_dir / "manifest.json"
        self._manifest = {}
        if manifest_path.exists():
            import json
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.xgb = joblib.load(model_dir / "xgb_native.joblib")
        self.lgb = joblib.load(model_dir / "lgb_native.joblib")
        self.cb = joblib.load(model_dir / "cb_native.joblib")
        self.scaler = joblib.load(model_dir / "scaler_native.joblib")

        cal_path = model_dir.parent / "artifacts" / "calibrator.joblib"
        self.calibrator = joblib.load(cal_path) if cal_path.exists() else None

        self._version = self._manifest.get("model_version", "unknown")
        self._n_features = self._manifest.get("n_features", 48)
        self.locked_threshold = float(self._manifest.get("locked_threshold", 0.0))

    def predict(self, features: dict) -> tuple[float, dict]:
        # float32 parity: training cast the 48-vector to float32 BEFORE scaling
        # (retrain .values.astype(np.float32)); feeding float64 moves razor-thin
        # memorized splits on large interaction features (amt_x_mcc ~1e6) across
        # the boundary and flips scores. Live inference must mirror float32.
        vec = map_raw_to_native(features).reshape(1, -1).astype(np.float32)
        X = self.scaler.transform(vec)
        p_xgb = float(self.xgb.predict_proba(X)[0, 1])
        p_lgb = float(self.lgb.predict_proba(X)[0, 1])
        p_cb = float(self.cb.predict_proba(X)[0, 1])
        raw = ENSEMBLE_WEIGHTS["xgb"] * p_xgb + ENSEMBLE_WEIGHTS["lgb"] * p_lgb + ENSEMBLE_WEIGHTS["cb"] * p_cb
        prob = float(np.clip(raw, 0.0, 1.0))
        members = [p_xgb, p_lgb, p_cb]
        return prob, {
            "model_variance": round(float(np.var(members)), 6),
            "model_disagreement": round(float(np.ptp(members)), 6),
            "individual_outputs": {"xgboost": round(p_xgb, 4), "lightgbm": round(p_lgb, 4), "catboost": round(p_cb, 4)},
            "ensemble_raw": round(raw, 6),
            "model_version": self._version,
            "model_type": "altman_native",
        }

    def predict_many(self, rows: list[dict]) -> np.ndarray:
        vecs = np.array([map_raw_to_native(r) for r in rows]).astype(np.float32)
        X = self.scaler.transform(vecs)
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]
        p_cb = self.cb.predict_proba(X)[:, 1]
        return ENSEMBLE_WEIGHTS["xgb"] * p_xgb + ENSEMBLE_WEIGHTS["lgb"] * p_lgb + ENSEMBLE_WEIGHTS["cb"] * p_cb

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def model_type(self) -> str:
        return "altman_native_xgb_lgb_cb"

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def _has_finance_model(self) -> bool:
        return False

    def predict_combined(self, features: dict) -> tuple[float, float, dict]:
        prob, uncertainty = self.predict(features)
        return prob, prob, uncertainty

    def predict_with_finance(self, features: dict) -> tuple[float, dict]:
        prob, uncertainty = self.predict(features)
        return prob, {"base_score": prob, "finance_score": 0.0, "finance_boost": 0.0,
                      "is_micro_fraud_suspect": False, "has_finance_data": False}

    def odds_of(self, p: float) -> float:
        return float("inf") if p >= 1.0 else float(p / (1.0 - p))

    def components(self, features: dict) -> dict:
        vec = map_raw_to_native(features).reshape(1, -1).astype(np.float32)
        X = self.scaler.transform(vec)
        p_xgb = float(self.xgb.predict_proba(X)[0, 1])
        p_lgb = float(self.lgb.predict_proba(X)[0, 1])
        p_cb = float(self.cb.predict_proba(X)[0, 1])
        return {
            "outputs": {"xgboost": p_xgb, "lightgbm": p_lgb, "catboost": p_cb},
            "coefficients": {"xgboost": 0.34, "lightgbm": 0.33, "catboost": 0.33},
            "intercept": 0.0, "X": vec,
        }
