"""Altman Production Ensemble Engine — with entity-level fraud rates.

Loads the XGB + LightGBM + CatBoost ensemble trained on 24M Altman rows.
Accepts the standard ML_FEATURES dict from the privacy layer plus optional
entity identifiers (user_id, merchant_id, city_id) for entity-level fraud
rate features.

Entity fraud rate features are LEAKAGE-SAFE because:
- Computed from historical data only (sliding window of past events)
- The tracker maintains per-entity deque of (label, timestamp) pairs
- Rates are computed BEFORE the current event is scored
- Window is per-entity, not global

Architecture:
  ML_FEATURES dict + entity IDs → feature mapper → entity fraud rate lookup
  → RobustScaler → [XGB, LGB, CB] → weighted average → calibrated prob
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.risk_engine.kill_switch import KillSwitchActiveError, is_armed

ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTION_DIR = ROOT / "models" / "production"

# The 15 Altman features — auto-generated lean model.
# This MUST match models/production/manifest.json features exactly.
ALTMAN_FEATURES = [
    "log_amt", "amt_sq", "hour_cos", "is_business_hours",
    "chip", "is_online", "mcc_n",
    "has_zip", "has_state",
    "merch_tx_count",
    "merch_fraud_rate", "city_fraud_rate",
    "very_high_amt",
    "amt_x_mcc", "amt_x_online",
]

# The 21 causal §16 features — the full vector the Privacy Layer actually
# computes at ingest (derive_event_features), prior-only by construction
# (Part-1 causality audit, `--causal` generator). The Part-1 protocol model
# trained on exactly this set reached 97.95% test recall at 1% FPR, while the
# deployed 15-feature projection (7/15 constant: no MCC/ZIP/state source,
# cold-start zeros) collapsed recall to ~36%. Schema `altman_runtime_v3`
# retrains the production ensemble on THIS set so train == prod by
# construction AND the model sees the informative features.
CAUSAL_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "hour_of_day", "is_weekend", "shared_device_accounts",
    "shared_recipient_accounts", "mule_ring_score", "hour_deviation",
    "amount_zscore", "velocity_deviation", "recipient_novelty",
    "txn_regularity",
]

# §16 feature order must match the privacy layer's derive_event_features dict.
# Fallbacks mirror the privacy layer's zero-state defaults (cold start).
_CAUSAL_DEFAULTS = {
    "amount_ratio": 1.0, "txn_freq_last_24h": 0.0, "txn_time_unusual": 0.0,
    "new_device_flag": 0.0, "unusual_location_flag": 0.0,
    "unusual_recipient_flag": 0.0, "failed_auth_count_24h": 0.0,
    "days_since_last_similar_txn": 0.0, "gradual_escalation_score": 0.0,
    "known_device_count": 0.0, "account_tenure_days": 1.0,
    "hour_of_day": 12.0, "is_weekend": 0.0, "shared_device_accounts": 0.0,
    "shared_recipient_accounts": 0.0, "mule_ring_score": 0.0,
    "hour_deviation": 0.0, "amount_zscore": 0.0, "velocity_deviation": 0.0,
    "recipient_novelty": 0.0, "txn_regularity": 0.0,
}


def map_causal_features(features: dict) -> np.ndarray:
    """Map a ML_FEATURES dict to the 21 causal §16 feature vector.

    This is the FEATURE-CONTRACT binding for schema altman_runtime_v3: it
    reads the exact keys the privacy layer's derive_event_features returns,
    in contract order, with cold-start fallbacks for missing keys (identical
    to the privacy layer's defaults). No derived/reprojected columns - the
    vector the model sees in production IS the vector the privacy layer
    computes at ingest, and IS the vector the retrain mapped from the CSV
    (which was produced by the same causal derivation).
    """
    vec = np.array([
        float(features.get(name, _CAUSAL_DEFAULTS[name]))
        for name in CAUSAL_FEATURES
    ], dtype=np.float64)
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec


# Default ensemble weights (near-equal, calibrated by OOF).
# The LIVE source of truth is manifest.json `ensemble_weights`: the engine
# reads membership + weights from the manifest at load so a deploy can never
# silently blend an undeclared member or weight set (model-release audit #28).
# This constant is only a fallback for manifests that predate the field.
ENSEMBLE_WEIGHTS = {"xgb": 0.34, "lgb": 0.33, "cb": 0.33}


def map_ml_features_to_altman(features: dict) -> np.ndarray:
    """Map a ML_FEATURES dict + entity IDs to 28 Altman lean features.

    Entity fraud rates are looked up from the EntityFraudRateTracker
    singleton. If entity IDs are not present, uses baseline rate (0.001).
    """
    amt_ratio = float(features.get("amount_ratio", 1.0))
    hour_of_day = float(features.get("hour_of_day", 12.0))
    is_weekend = int(features.get("is_weekend", 0))
    chip = int(features.get("new_device_flag", 0))  # 1 = online (new device proxy)
    txn_freq = float(features.get("txn_freq_last_24h", 5))
    known_devices = float(features.get("known_device_count", 5))
    account_tenure = float(features.get("account_tenure_days", 180))
    failed_auth = float(features.get("failed_auth_count_24h", 0))
    amt_zscore_ml = float(features.get("amount_zscore", 0.0))
    vel_dev = float(features.get("velocity_deviation", 0.3))

    # Entity fraud rates — look up from tracker or use pre-computed values
    user_id = features.get("user_id", "")
    merchant_id = features.get("merchant_id", "")
    city_id = features.get("city_id", "")

    # Try to get from the entity tracker (live path)
    user_fraud_rate = features.get("user_fraud_rate", None)
    merch_fraud_rate = features.get("merch_fraud_rate", None)
    city_fraud_rate = features.get("city_fraud_rate", None)

    if user_fraud_rate is None or merch_fraud_rate is None or city_fraud_rate is None:
        # Lookup from tracker
        try:
            from src.risk_engine.entity_fraud_rates import get_tracker
            tracker = get_tracker()
            rates = tracker.get_rates(user_id, merchant_id, city_id)
            if user_fraud_rate is None:
                user_fraud_rate = rates["user_fraud_rate"]
            if merch_fraud_rate is None:
                merch_fraud_rate = rates["merch_fraud_rate"]
            if city_fraud_rate is None:
                city_fraud_rate = rates["city_fraud_rate"]
        except Exception:
            # Tracker not available — use baseline
            if user_fraud_rate is None:
                user_fraud_rate = 0.001
            if merch_fraud_rate is None:
                merch_fraud_rate = 0.001
            if city_fraud_rate is None:
                city_fraud_rate = 0.001

    user_fraud_rate = float(user_fraud_rate)
    merch_fraud_rate = float(merch_fraud_rate)
    city_fraud_rate = float(city_fraud_rate)

    # Derive raw-ish values from ratios
    ref_amt = 100.0
    amt = amt_ratio * ref_amt
    hr = hour_of_day
    mn = 0.0

    # Clean Altman features
    log_amt = np.log1p(amt)
    amt_sq = amt ** 2
    dow = float(is_weekend)
    mcc_n = 0.0  # MCC not in ML_FEATURES
    err = 1.0 if failed_auth > 0 else 0.0
    is_online = chip
    day = 15.0

    # Velocity — use real values from Privacy Layer velocity tracker
    user_tx_count = float(features.get("user_tx_count", txn_freq))
    card_tx_count = float(features.get("card_tx_count", txn_freq * 0.8))
    # merch_tx_count: the velocity tracker reports 24h merchant-transaction
    # counts and returns 0 for merchants with no window history (feature
    # parity #2 fix). Training rows carry no merchant context and must see
    # the SAME cold-start value - NOT known_devices (a device-count proxy
    # that never equals the merchant velocity count). 0 is the canonical
    # cold-start value on both the offline and the live path.
    merch_tx_count = float(features.get("merch_tx_count", 0.0))

    # Amount features — use real user average from velocity tracker
    user_avg_amt = float(features.get("user_avg_amt", ref_amt))
    amt_vs_user_avg = amt / (user_avg_amt + 1e-6)
    amt_zscore = amt_zscore_ml
    very_high_amt = 1.0 if amt_ratio > 5.0 else 0.0

    # Time features
    hour_sin = np.sin(2 * np.pi * hr / 24.0)
    hour_cos = np.cos(2 * np.pi * hr / 24.0)
    is_business_hours = 1.0 if 9 <= hr <= 17 else 0.0

    # Interactions
    amt_x_hr = amt * hr
    amt_x_mcc = amt * mcc_n
    amt_x_chip = amt * chip
    amt_x_online = amt * is_online

    # Location
    has_zip = 0
    has_state = 0
    very_high_amt = 1.0 if amt_ratio > 5.0 else 0.0

    vec = np.array([
        log_amt, amt_sq, hour_cos, is_business_hours,
        chip, is_online, mcc_n,
        has_zip, has_state,
        merch_tx_count,
        merch_fraud_rate, city_fraud_rate,
        very_high_amt,
        amt_x_mcc, amt_x_online,
    ], dtype=np.float64)

    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec


class AltmanEnsembleEngine:
    """Altman XGB+LGB+CB ensemble — NO target leakage, 32 features.

    Drop-in compatible with FusionEngine.predict() API.
    """

    def __init__(self, prod_dir: Path = None, verify_integrity: bool = True):
        if prod_dir is None:
            prod_dir = PRODUCTION_DIR

        self._prod_dir = prod_dir
        self._manifest = {}

        # Load manifest
        manifest_path = prod_dir / "manifest.json"
        if manifest_path.exists():
            import json
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Ensemble membership + weights are governed by the manifest (single
        # source of truth). Members default to "any file present" for legacy
        # manifests that predate the field; weights default to the constants
        # above. A CatBoost file the manifest does NOT declare is never
        # blended in - it is logged and ignored (release audit #28).
        import logging
        _log = logging.getLogger("risk_engine.altman")
        _mw = self._manifest.get("ensemble_weights") or {}
        self._members = self._manifest.get("ensemble_members")
        self._weights = {
            "xgb": float(_mw.get("xgb", ENSEMBLE_WEIGHTS["xgb"])),
            "lgb": float(_mw.get("lgb", ENSEMBLE_WEIGHTS["lgb"])),
            "cb": float(_mw.get("cb", ENSEMBLE_WEIGHTS["cb"])),
        }

        # Load models
        self.xgb = joblib.load(prod_dir / "xgb_production.joblib")
        self.lgb = joblib.load(prod_dir / "lgb_production.joblib")
        self.scaler = joblib.load(prod_dir / "scaler_production.joblib")
        cb_path = prod_dir / "cb_production.joblib"
        cb_declared = self._members is None or "cb" in self._members
        if cb_path.exists() and cb_declared:
            self.cb = joblib.load(cb_path)
        elif cb_path.exists():
            self.cb = None
            _log.warning(
                "cb_production.joblib present but manifest does not declare a "
                "'cb' member - ignoring it (release audit #28). Add it to "
                "manifest.json ensemble_members/ensemble_weights to blend it."
            )
        else:
            self.cb = None

        # Load calibrator: prefer a calibrator shipped with the release
        # (calibrator_production.joblib) over the shared artifacts one, so a
        # model release can carry its own calibration (release audit #28 / #27
        # runtime-consistent retrain). Backwards compatible - older releases
        # fall back to models/artifacts/calibrator.joblib.
        cal_path = prod_dir / "calibrator_production.joblib"
        if not cal_path.exists():
            cal_path = prod_dir.parent / "artifacts" / "calibrator.joblib"
        self.calibrator = joblib.load(cal_path) if cal_path.exists() else None

        self._version = self._manifest.get("model_version", "unknown")
        self._n_features = self._manifest.get("n_features", 32)

        # Model governance: if a permanent record exists for this model, verify
        # artifact hashes + feature schema BEFORE serving. A mismatch raises so
        # the caller (main.py lifespan) degrades to rules-only instead of
        # silently serving a wrong/corrupt artifact.
        if verify_integrity:
            self._verify_schema(prod_dir)
            self._verify_against_record(prod_dir)

    def _schema_features(self) -> list[str]:
        """Resolve the feature list for the loaded manifest.

        A release declares its feature vector in manifest.json `features`.
        Two schemas exist: altman_runtime_v1/v2 (15-feature Altman projection)
        and altman_runtime_v3 (21 causal §16 features). The mapper, contract
        check, and column probe all key off this resolved list so both schema
        families remain loadable (rollback to a v2 release must still work).
        """
        mf = self._manifest.get("features")
        if mf == CAUSAL_FEATURES:
            return list(CAUSAL_FEATURES)
        if mf == ALTMAN_FEATURES:
            return list(ALTMAN_FEATURES)
        # Fall back to the module constant (legacy manifests w/o feature list)
        return list(ALTMAN_FEATURES)

    def _verify_schema(self, prod_dir: Path) -> None:
        """Feature-contract binding: refuse to serve a schema mismatch.

        Three independent declarations of the feature vector must agree, in
        order: the manifest's `features`, the shipped feature_list.json, and
        the resolved schema list (21 causal or 15 Altman). The scaler and
        every learner must also have been fitted on exactly that many columns.
        Any disagreement means a wrong/corrupt release - raise so the caller
        fails closed.
        """
        import json as _json
        feats = self._schema_features()
        mf = self._manifest.get("features")
        fl_path = prod_dir / "feature_list.json"
        fl = _json.loads(fl_path.read_text(encoding="utf-8"))["features"] \
            if fl_path.exists() else None
        # Functional column probe: a N-column vector must score on every
        # learner (CatBoost does not expose sklearn's n_features_in_, so a
        # predict_proba probe is the portable dimensional check).
        try:
            _probe = self.scaler.transform(np.zeros((1, len(feats))))
            _ok = all(m.predict_proba(_probe).shape[1] == 2
                      for m in (self.xgb, self.lgb) if m is not None)
            if self.cb is not None:
                _ok = _ok and self.cb.predict_proba(_probe).shape[1] == 2
        except Exception:
            _ok = False
        _sc = self.scaler
        _center = getattr(_sc, "center_", None)
        _scale = getattr(_sc, "scale_", None)
        _sc_n = (_center if _center is not None else _scale)
        _sc_n = _sc_n.shape[0] if _sc_n is not None else None
        checks = {
            "manifest.features == resolved schema": mf == feats,
            "feature_list.json == resolved schema": fl == feats,
            "scaler fitted on N columns": _sc_n == len(feats),
            "learners score an N-column probe": bool(_ok),
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise RuntimeError(
                "Feature-schema mismatch - refusing to serve. "
                + "; ".join(bad)
                + f" (manifest: {mf}, feature_list: {fl})")

    def _verify_against_record(self, prod_dir: Path) -> None:
        """Check loaded artifacts against models/model_records/<model_id>.json."""
        try:
            from src.risk_engine.model_governance import ModelRecord, verify
        except Exception:
            return  # governance module unavailable -> skip (non-fatal)
        if self._version == "unknown":
            return
        records_dir = (ROOT / "models" / "model_records")
        record_path = records_dir / f"{self._version}.json"
        if not record_path.exists():
            # No record on disk: hash the artifact now and warn loudly instead of
            # failing the load (record generation is a deployment-step concern).
            import warnings
            warnings.warn(
                f"Model {self._version} has NO governance record "
                f"({record_path}). Run scripts/model_governance.py to register it."
            )
            return
        try:
            record = ModelRecord.load(records_dir, self._version)
            verify(record, prod_dir, loaded_features=self._schema_features())
        except Exception as e:
            import logging
            logging.getLogger("risk_engine").error(
                "Model integrity verification FAILED for %s: %s", self._version, e
            )
            raise

    def _map(self, features: dict) -> np.ndarray:
        """Schema-aware mapper: 21 causal (§16) or 15 Altman projection."""
        if self._schema_features() == CAUSAL_FEATURES:
            return map_causal_features(features)
        return map_ml_features_to_altman(features)

    def predict(self, features: dict) -> tuple[float, dict]:
        """Predict fraud probability for a single ML_FEATURES dict."""
        if is_armed():
            raise KillSwitchActiveError(
                "ML scoring is quarantined by the kill switch; use the "
                "rules-only fallback (see models/kill_switch.json)"
            )
        vec = self._map(features).reshape(1, -1)
        X = self.scaler.transform(vec)

        p_xgb = float(self.xgb.predict_proba(X)[0, 1])
        p_lgb = float(self.lgb.predict_proba(X)[0, 1])

        w_xgb = self._weights["xgb"]
        w_lgb = self._weights["lgb"]
        if self.cb is not None:
            p_cb = float(self.cb.predict_proba(X)[0, 1])
            w_cb = self._weights["cb"]
            # Renormalize
            total_w = w_xgb + w_lgb + w_cb
            raw = (w_xgb * p_xgb + w_lgb * p_lgb + w_cb * p_cb) / total_w
        else:
            raw = w_xgb * p_xgb + w_lgb * p_lgb

        if self.calibrator is not None:
            prob = float(np.clip(
                self.calibrator.predict(np.array([[raw]]))[0], 0.0, 1.0
            ))
        else:
            prob = float(np.clip(raw, 0.0, 1.0))

        outputs = [p_xgb, p_lgb]
        names = ["xgboost", "lightgbm"]
        if self.cb is not None:
            outputs.append(p_cb)
            names.append("catboost")
        uncertainty = {
            "model_variance": round(float(np.var(outputs)), 6),
            "model_disagreement": round(float(max(outputs) - min(outputs)), 6),
            "individual_outputs": {n: round(p, 4) for n, p in zip(names, outputs)},
            "ensemble_raw": round(raw, 6),
            "calibrated": prob != raw,
            "model_version": self._version,
            "target_leakage": False,
        }
        return prob, uncertainty

    def predict_many(self, rows: list[dict]) -> np.ndarray:
        """Batch predict — maps all rows, scales, and predicts with each model."""
        if is_armed():
            raise KillSwitchActiveError(
                "ML scoring is quarantined by the kill switch; use the "
                "rules-only fallback (see models/kill_switch.json)"
            )
        vecs = np.array([self._map(r) for r in rows])
        X = self.scaler.transform(vecs)

        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]

        w_xgb = self._weights["xgb"]
        w_lgb = self._weights["lgb"]
        if self.cb is not None:
            p_cb = self.cb.predict_proba(X)[:, 1]
            w_cb = self._weights["cb"]
            total_w = w_xgb + w_lgb + w_cb
            raw = (w_xgb * p_xgb + w_lgb * p_lgb + w_cb * p_cb) / total_w
        else:
            raw = w_xgb * p_xgb + w_lgb * p_lgb

        if self.calibrator is not None:
            calibrated = self.calibrator.predict(raw.reshape(-1, 1))
            return np.clip(calibrated, 0.0, 1.0)
        return np.clip(raw, 0.0, 1.0)

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def model_type(self) -> str:
        return "altman_xgb_lgb_cb_clean"

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def _has_finance_model(self) -> bool:
        return False

    def predict_combined(self, features: dict) -> tuple[float, float, dict]:
        """Single-pass calibrated + weighted prediction (FusionEngine API compat)."""
        prob, uncertainty = self.predict(features)
        return prob, prob, uncertainty

    def predict_with_finance(self, features: dict) -> tuple[float, dict]:
        """Finance-enhanced prediction — delegates to predict (no finance model)."""
        prob, uncertainty = self.predict(features)
        return prob, {"base_score": prob, "finance_score": 0.0,
                      "finance_boost": 0.0, "is_micro_fraud_suspect": False,
                      "has_finance_data": False}

    def odds_of(self, p: float) -> float:
        """Odds for a calibrated probability: p / (1 - p)."""
        if p >= 1.0:
            return float("inf")
        return float(p / (1.0 - p))

    def components(self, features: dict) -> dict:
        """Per-model outputs + weights for attribution (FusionEngine API compat)."""
        vec = self._map(features).reshape(1, -1)
        X = self.scaler.transform(vec)
        p_xgb = float(self.xgb.predict_proba(X)[0, 1])
        p_lgb = float(self.lgb.predict_proba(X)[0, 1])
        outputs = {"xgboost": p_xgb, "lightgbm": p_lgb}
        coefficients = {"xgboost": self._weights["xgb"], "lightgbm": self._weights["lgb"]}
        if self.cb is not None:
            p_cb = float(self.cb.predict_proba(X)[0, 1])
            outputs["catboost"] = p_cb
            coefficients["catboost"] = self._weights["cb"]
        return {
            "outputs": outputs,
            "coefficients": coefficients,
            "intercept": 0.0,
            "X": vec,
        }
