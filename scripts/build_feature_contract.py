#!/usr/bin/env python3
"""PS-14 FEATURE CONTRACT (Part 2 - production parity, schema-aware).

Generates the single canonical machine-readable feature contract for the
DEPLOYED runtime schema from the ONE implementation that both the offline
training path and the live scoring path consume. Two schema families are
supported and dispatched on the deployed manifest:
  - altman_runtime_v1/v2: the 15-feature Altman projection
    (map_ml_features_to_altman)
  - altman_runtime_v3: the 21 causal section-16 features (Part-5 recall fix,
    map_causal_features - the exact vector the Privacy Layer computes at
    ingest, prior-only by construction)
The contract is written to models/feature_contract.json; every entry documents
name, type, units, formula, source keys, causal/label rules, missing-value
behavior and cold-start fallback.

Invariant enforced here: contract features == resolved schema == the deployed
manifest features == the shipped feature_list.json, in order. The risk engine
enforces the same binding at load time (AltmanEnsembleEngine._verify_schema).

Usage:
    ./.venv/Scripts/python.exe scripts/build_feature_contract.py [prod_dir]
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.risk_engine.altman_ensemble import (  # noqa: E402
    ALTMAN_FEATURES, CAUSAL_FEATURES)
from src.privacy_layer.native_features import (  # noqa: E402
    ALTMAN_NATIVE_FEATURES)


def _f(name, ftype, units, formula, source, causal, missing):
    return {"name": name, "type": ftype, "units": units, "formula": formula,
            "source_keys": source, "causal_rule": causal,
            "missing_value_behavior": missing}


def _causal_entries() -> list[dict]:
    """Contract entries for the 21 causal section-16 features (schema v3).

    These are the exact keys derive_event_features returns at ingest; the
    causal rule for every one is the Part-1 audited prior-only derivation
    (no future data, no label use). Source key == feature name.
    """
    return [
        _f("amount_ratio", "float", "ratio [0,inf)",
           "amount / account prior median amount (expanding prior-only median)",
           ["amount_ratio"], "prior-only expanding window; no future rows",
           "absent -> 1.0 (ratio to self)"),
        _f("txn_freq_last_24h", "float", "txns/24h",
           "count of prior events in the 24h window anchored at event ts",
           ["txn_freq_last_24h"], "prior-only 24h window (anchor=min(now,ts), "
           "both bounds - causal derivation)", "absent -> 0.0"),
        _f("txn_time_unusual", "int {0,1}", "bool",
           "hour_of_day not in account typical hours", ["txn_time_unusual"],
           "instant field; profile prior-only", "absent -> 0"),
        _f("new_device_flag", "int {0,1}", "bool",
           "device not in account known-device set (prior-only)",
           ["new_device_flag"], "prior-only device history", "absent -> 0"),
        _f("unusual_location_flag", "int {0,1}", "bool",
           "location not in usual locations (prior-only)",
           ["unusual_location_flag"], "prior-only location history", "absent -> 0"),
        _f("unusual_recipient_flag", "int {0,1}", "bool",
           "recipient not in usual recipients (prior-only)",
           ["unusual_recipient_flag"], "prior-only recipient history", "absent -> 0"),
        _f("failed_auth_count_24h", "float", "count",
           "prior failed authentications in 24h", ["failed_auth_count_24h"],
           "prior-only window", "absent -> 0.0"),
        _f("days_since_last_similar_txn", "float", "days",
           "days since the most recent similar prior transaction",
           ["days_since_last_similar_txn"], "prior-only; never the current event",
           "absent -> 0.0"),
        _f("gradual_escalation_score", "float", "ratio [0,1] clipped",
           "escalation of recent amount ratios (prior window + current ratio)",
           ["gradual_escalation_score"], "prior window only; label never consumed",
           "absent -> 0.0"),
        _f("known_device_count", "float", "count",
           "number of prior known devices for the account",
           ["known_device_count"], "prior-only device history", "absent -> 0.0"),
        _f("account_tenure_days", "float", "days",
           "account age at event time", ["account_tenure_days"],
           "instant; no label dependency", "absent -> 1.0"),
        _f("hour_of_day", "float", "hour [0,24)", "event hour (0-23)",
           ["hour_of_day"], "instant event field", "absent -> 12.0"),
        _f("is_weekend", "int {0,1}", "bool", "event on Sat/Sun",
           ["is_weekend"], "instant event field", "absent -> 0"),
        _f("shared_device_accounts", "float", "count (capped)",
           "other accounts seen on this device (link analysis, prior graph)",
           ["shared_device_accounts"], "prior graph; no future links", "absent -> 0.0"),
        _f("shared_recipient_accounts", "float", "count (capped)",
           "other accounts sharing this recipient (prior graph)",
           ["shared_recipient_accounts"], "prior graph; no future links", "absent -> 0.0"),
        _f("mule_ring_score", "float", "normalized [0,1]",
           "composite of shared-device/recipient counts (link analysis)",
           ["mule_ring_score"], "prior graph only", "absent -> 0.0"),
        _f("hour_deviation", "float", "z-like score",
           "how unusual the event hour is vs account typical hours",
           ["hour_deviation"], "prior profile only", "absent -> 0.0"),
        _f("amount_zscore", "float", "z-score",
           "amount vs account prior amount distribution", ["amount_zscore"],
           "prior-only expanding stats", "absent -> 0.0"),
        _f("velocity_deviation", "float", "z-like score",
           "txn freq vs account typical frequency", ["velocity_deviation"],
           "prior-only", "absent -> 0.0"),
        _f("recipient_novelty", "float", "fraction [0,1]",
           "fraction of recent recipients that are new", ["recipient_novelty"],
           "prior recipients only", "absent -> 0.0"),
        _f("txn_regularity", "float", "CV of inter-arrival hours",
           "coefficient of variation of prior inter-arrival times",
           ["txn_regularity"], "prior inter-arrivals only", "absent -> 0.0"),
    ]


def _native_entries() -> list[dict]:
    """Contract entries for the 48 Altman-NATIVE features (schema v2).

    Every historical feature (counts, averages, rates) is derived by
    src/privacy_layer/native_features.py from velocity/fraud-rate context
    that is SHIFTED before the event being scored - the event itself is
    never counted in its own velocity, average, or entity fraud rate.
    """
    return [
        _f("amt", "float", "dollars", "raw amount (clamped >= 0)",
           ["amount"], "instant event field", "absent -> 0.0"),
        _f("log_amt", "float", "log dollars", "log1p(amt)", ["amount"],
           "derived from instant field", "absent -> 0.0"),
        _f("amt_sq", "float", "dollars^2", "amt ** 2", ["amount"],
           "derived from instant field", "absent -> 0.0"),
        _f("hr", "float", "hour [0,24)", "event hour", ["ts", "hour_of_day"],
           "instant event field", "absent -> 12.0"),
        _f("mn", "float", "minute [0,60)", "event minute", ["ts"],
           "instant event field", "absent -> 0.0"),
        _f("dow", "float", "day-of-week [0,6]", "event weekday", ["ts"],
           "instant event field", "absent -> 0.0"),
        _f("Month", "float", "month [1,12]", "event month", ["ts"],
           "instant event field", "absent -> 8.0"),
        _f("Day", "float", "day-of-month [1,31]", "event day", ["ts"],
           "instant event field", "absent -> 15.0"),
        _f("hour_sin", "float", "sin(2*pi*hour/24)", "sin transform of hr",
           ["ts", "hour_of_day"], "derived from instant field", "absent -> 0.0"),
        _f("hour_cos", "float", "cos(2*pi*hour/24)", "cos transform of hr",
           ["ts", "hour_of_day"], "derived from instant field", "absent -> 0.0"),
        _f("is_night", "int {0,1}", "bool", "1 if hr < 6 or hr >= 22",
           ["ts"], "instant event field", "absent -> 0"),
        _f("is_business_hours", "int {0,1}", "bool", "1 if 9 <= hr <= 17",
           ["ts"], "instant event field", "absent -> 1"),
        _f("chip", "int {0,1}", "bool", "use_chip == 'Chip Transaction'",
           ["use_chip"], "instant event field", "absent -> 0"),
        _f("is_online", "int {0,1}", "bool",
           "use_chip == 'Online Transaction'", ["use_chip"],
           "instant event field", "absent -> 0"),
        _f("is_swipe", "int {0,1}", "bool",
           "use_chip == 'Swipe Transaction'", ["use_chip"],
           "instant event field", "absent -> 0"),
        _f("err", "int {0,1}", "bool", "errors field non-empty",
           ["errors"], "instant event field", "absent -> 0"),
        _f("has_zip", "int {0,1}", "bool", "zip field non-empty", ["zip"],
           "instant event field", "absent -> 0"),
        _f("has_state", "int {0,1}", "bool",
           "merchant_state field non-empty", ["merchant_state"],
           "instant event field", "absent -> 0"),
        _f("is_online_or_no_state", "int {0,1}", "bool",
           "is_online OR has_state == 0", ["use_chip", "merchant_state"],
           "derived from instant fields", "absent -> 1"),
        _f("mcc", "float", "merchant category code", "raw mcc", ["mcc"],
           "instant event field", "absent -> 0.0"),
        _f("mcc_high", "int {0,1}", "bool", "mcc >= 5000", ["mcc"],
           "instant event field", "absent -> 0"),
        _f("mcc_restaurant", "int {0,1}", "bool", "5812 <= mcc <= 5814",
           ["mcc"], "instant event field", "absent -> 0"),
        _f("mcc_gas", "int {0,1}", "bool", "5541 <= mcc <= 5542", ["mcc"],
           "instant event field", "absent -> 0"),
        _f("mcc_grocery", "int {0,1}", "bool", "5411 <= mcc <= 5422",
           ["mcc"], "instant event field", "absent -> 0"),
        _f("mcc_travel", "int {0,1}", "bool", "3000 <= mcc <= 3350",
           ["mcc"], "instant event field", "absent -> 0"),
        _f("mcc_online", "int {0,1}", "bool", "5967 <= mcc <= 5969",
           ["mcc"], "instant event field", "absent -> 0"),
        _f("merchant_id", "float", "stable code [0,100000)",
           "sha256(merchant name) mod 100000; same code train==prod",
           ["merchant_id"], "entity identifier; code stable across split",
           "absent -> 0.0 (cold-start bucket)"),
        _f("city_id", "float", "stable code [0,100000)",
           "sha256(merchant_city) mod 100000", ["city_id"],
           "entity identifier; code stable across split",
           "absent -> 0.0 (cold-start bucket)"),
        _f("card_id", "float", "stable code [0,100000)",
           "sha256(card) mod 100000", ["card_id"],
           "entity identifier; code stable across split",
           "absent -> 0.0 (cold-start bucket)"),
        _f("user_tx_count", "float", "count",
           "prior events for user_id (velocity tracker, shifted)",
           ["user_tx_count"], "prior-only history; event excluded", "absent -> 0.0"),
        _f("card_tx_count", "float", "count",
           "prior events for card (velocity tracker, shifted)",
           ["card_tx_count"], "prior-only history; event excluded", "absent -> 0.0"),
        _f("user_avg_amt", "float", "dollars",
           "prior mean amount for user (shifted average)", ["user_avg_amt"],
           "prior-only; event excluded", "absent -> 0.0"),
        _f("amt_vs_user_avg", "float", "ratio",
           "amt / user_avg_amt (prior mean)", ["amount", "user_avg_amt"],
           "prior-only average; event excluded", "absent -> 0.0"),
        _f("amt_zscore", "float", "z-score",
           "(amt - prior mean) / prior std (shifted)",
           ["amount", "user_avg_amt"], "prior-only stats; event excluded",
           "absent -> 0.0"),
        _f("merch_tx_count", "float", "count",
           "prior events for merchant (velocity tracker, shifted)",
           ["merch_tx_count"], "prior-only history; event excluded", "absent -> 0.0"),
        _f("user_merchant_diversity", "float", "count",
           "distinct prior merchants for user", ["user_merchant_diversity"],
           "prior-only", "absent -> 0.0"),
        _f("user_city_diversity", "float", "count",
           "distinct prior cities for user", ["user_city_diversity"],
           "prior-only", "absent -> 0.0"),
        _f("user_fraud_rate", "float", "rate [0,1]",
           "prior fraud rate for user (entity tracker, shifted; labels "
           "recorded only after the scored event)", ["user_fraud_rate"],
           "prior-only; label can never precede the scored event",
           "absent -> 0.001 cold-start baseline"),
        _f("merch_fraud_rate", "float", "rate [0,1]",
           "prior fraud rate for merchant (shifted)",
           ["merch_fraud_rate"], "prior-only; label post-event",
           "absent -> 0.001 cold-start baseline"),
        _f("city_fraud_rate", "float", "rate [0,1]",
           "prior fraud rate for city (shifted)", ["city_fraud_rate"],
           "prior-only; label post-event", "absent -> 0.001 cold-start baseline"),
        _f("high_amt", "int {0,1}", "bool", "amt > 2 * user_avg_amt",
           ["amount", "user_avg_amt"], "prior-only average", "absent -> 0"),
        _f("very_high_amt", "int {0,1}", "bool", "amt > 5 * user_avg_amt",
           ["amount", "user_avg_amt"], "prior-only average", "absent -> 0"),
        _f("amt_x_hr", "float", "dollars*hour", "amt * hr", ["amount", "ts"],
           "derived from instant fields", "absent -> 0.0"),
        _f("amt_x_mcc", "float", "dollars*code", "amt * mcc",
           ["amount", "mcc"], "derived from instant fields", "absent -> 0.0"),
        _f("amt_x_chip", "float", "dollars*bool", "amt * chip",
           ["amount", "use_chip"], "derived from instant fields", "absent -> 0.0"),
        _f("amt_x_online", "float", "dollars*bool", "amt * is_online",
           ["amount", "use_chip"], "derived from instant fields", "absent -> 0.0"),
        _f("amt_x_night", "float", "dollars*bool", "amt * is_night",
           ["amount", "ts"], "derived from instant fields", "absent -> 0.0"),
        _f("user_merch_count", "float", "count",
           "prior merchant count for user (velocity tracker, shifted)",
           ["user_merch_count"], "prior-only; event excluded", "absent -> 0.0"),
    ]


def contract_for(prod_dir: Path) -> dict:
    manifest = json.loads((prod_dir / "manifest.json").read_text(encoding="utf-8"))
    mf = manifest.get("features")
    if manifest.get("is_native") or manifest.get("model_type", "").endswith("_native"):
        # native artifacts live in the altman_native subdir; its
        # feature_list.json is the plain 48-name list.
        flp = prod_dir / "altman_native" / "feature_list.json"
        features = ALTMAN_NATIVE_FEATURES
        mapper_name = "derive_native_features"
        entries = _native_entries()
    else:
        flp = prod_dir / "feature_list.json"
    fl = json.loads(flp.read_text(encoding="utf-8"))
    if isinstance(fl, dict):
        fl = fl.get("features") or fl["features"]
    if not (manifest.get("is_native") or manifest.get("model_type", "").endswith("_native")):
        if mf == CAUSAL_FEATURES:
            features = CAUSAL_FEATURES
            mapper_name = "map_causal_features"
            entries = _causal_entries()
        elif mf == ALTMAN_FEATURES:
            features = ALTMAN_FEATURES
            mapper_name = "map_ml_features_to_altman"
            entries = None
        else:
            raise AssertionError(f"unknown schema features: {mf}")
    assert mf is None or mf == features, (
        "feature declarations disagree - refusing to build a contract")
    assert fl == features, ("feature_list.json does not match resolved schema - "
                            "refusing to build a contract")
    schema = manifest.get("feature_schema_version", "altman_runtime_v1")
    # model artifact hash (xgb/lgb/cb/scaler/calibrator). For the native
    # family the artifacts live in the altman_native subdir with native
    # names - hash those (the actually-served files), not the legacy
    # root-level causal artifacts.
    hashes = {}
    if manifest.get("is_native") or manifest.get("model_type", "").endswith("_native"):
        base = prod_dir / "altman_native"
        names = ("xgb_native.joblib", "lgb_native.joblib", "cb_native.joblib",
                 "scaler_native.joblib", "feature_list.json")
    else:
        base = prod_dir
        names = ("xgb_production.joblib", "lgb_production.joblib",
                 "cb_production.joblib", "scaler_production.joblib",
                 "calibrator_production.joblib")
    for f in names:
        p = base / f
        if p.exists():
            hashes[f] = hashlib.sha256(p.read_bytes()).hexdigest()
    feat_list = entries if entries is not None else [
        # --- amount features. Raw amounts never leave the privacy layer;
        # the section-16 amount proxy is amount_ratio (prior-amount ratio).
        # amt := amount_ratio * 100 (reference $100). log_amt/amt_sq are
        # built on that SAME canonical amt - never on a different quantity.
        _f("log_amt", "float", "log dollars (log1p)",
           "log1p(amount_ratio * 100)", ["amount_ratio"],
           "amount_ratio is prior-only (causal dataset + live prior profile); "
           "no future/label info", "absent amount_ratio -> 1.0 (ratio to self)"),
        _f("amt_sq", "float", "dollars^2",
           "(amount_ratio * 100) ** 2", ["amount_ratio"],
           "same causal source as log_amt", "absent amount_ratio -> 1.0"),
        _f("hour_cos", "float", "cos(2*pi*hour/24) in [-1,1]",
           "cos(2*pi*hour_of_day/24)", ["hour_of_day"],
           "instant event field", "absent hour -> 12.0"),
        _f("is_business_hours", "int {0,1}", "bool",
           "1 if 9 <= hour_of_day <= 17 else 0", ["hour_of_day"],
           "instant event field", "absent hour -> 12.0 -> 1"),
        _f("chip", "int {0,1}", "bool (proxy)",
           "new_device_flag (documented proxy: no chip channel exists; a "
           "real chip input would need a new raw field + retrain)",
           ["new_device_flag"], "instant flag", "absent -> 0"),
        _f("is_online", "int {0,1}", "bool (proxy)",
           "same as chip (new_device_flag); documented proxy",
           ["new_device_flag"], "instant flag", "absent -> 0"),
        _f("mcc_n", "float", "constant 0.0",
           "0.0 - no merchant-category source in the synthetic event "
           "model; constant on BOTH train and inference by construction",
           [], "n/a", "constant 0.0 (documented no-source feature)"),
        _f("has_zip", "int {0,1}", "constant 0",
           "0 - no postal-code source; constant both sides", [], "n/a",
           "constant 0 (documented no-source feature)"),
        _f("has_state", "int {0,1}", "constant 0",
           "0 - no state source; constant both sides", [], "n/a",
           "constant 0 (documented no-source feature)"),
        _f("merch_tx_count", "float", "txns in 24h window (int-valued)",
           "velocity tracker merchant count for the event's merchant; 0 "
           "when no 24h history OR no merchant context (canonical "
           "cold-start). Fallback default is 0 on BOTH paths",
           ["merch_tx_count"], "prior-only 24h window anchored at event ts",
           "absent/unknown merchant -> 0.0 (never a device-count proxy)"),
        _f("merch_fraud_rate", "float", "rate [0,1]",
           "entity tracker rolling fraud rate for merchant_id; baseline "
           "0.001 only for entities with < min_events history",
           ["merch_fraud_rate", "merchant_id"], "labels recorded AFTER the "
           "event is scored (tracker.record post-evaluation); window is "
           "prior-only", "unseen merchant / absent id -> baseline 0.001 "
           "(documented cold-start fallback, distinct from real rates)"),
        _f("city_fraud_rate", "float", "rate [0,1]",
           "entity tracker rolling fraud rate for city_id; baseline 0.001 "
           "only below min_events", ["city_fraud_rate", "city_id"],
           "prior-only rolling window; labels post-evaluation",
           "unseen city / absent id -> baseline 0.001"),
        _f("very_high_amt", "int {0,1}", "bool",
           "1 if amount_ratio > 5.0 else 0", ["amount_ratio"],
           "causal amount_ratio source", "absent -> 0"),
        _f("amt_x_mcc", "float", "dollars (amt * mcc_n)",
           "(amount_ratio * 100) * mcc_n = 0.0 (mcc_n constant 0)",
           ["amount_ratio"], "derived; same sources as factors", "0.0"),
        _f("amt_x_online", "float", "dollars (amt * is_online)",
           "(amount_ratio * 100) * is_online", ["amount_ratio",
                                                "new_device_flag"],
           "causal amount_ratio; instant flag", "0.0 when online proxy 0"),
    ]
    return {
        "contract_name": "ps14_altman_runtime_feature_contract",
        "schema_version": schema,
        "model_version": manifest["model_version"],
        "model_type": manifest.get("model_type"),
        "feature_count": len(features),
        "feature_order": features,
        "model_artifact_sha256": hashes,
        "shared_implementation": (
            ("src/privacy_layer/native_features." + mapper_name
             if (manifest.get("is_native") or manifest.get("model_type", "").endswith("_native"))
             else "src/risk_engine/altman_ensemble." + mapper_name)
            + " (single derivation used by offline training AND live scoring)"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": feat_list,
        "historical_feature_rule": ("for event t only events with ts < t "
                                    "contribute; no future rows/labels; "
                                    "verified by the Part-1 causality gates"),
        "label_latency_rule": ("entity fraud rates are recorded only after an "
                               "evaluation outcome/label exists (tracker.record "
                               "is called post-score), so a label can never "
                               "influence an earlier or concurrent score"),
    }


def main() -> int:
    prod_dir = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "models/production")
    contract = contract_for(prod_dir)
    out = ROOT / "models" / "feature_contract.json"
    out.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(contract['features'])} features, "
          f"schema {contract['schema_version']}, model {contract['model_version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
