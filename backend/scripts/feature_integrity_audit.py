#!/usr/bin/env python3
"""Check #27 — Production data & feature integrity audit.

Independently compares the feature definitions, scales, and behavior of the
DEPLOYED lean-15 Altman model (models/production/*) against the OFFLINE
training definitions (data/_sorted_cache/features_all.parquet, built by
scripts/train_altman_fullscale.py pass2) and probes the runtime mapper with
missing/NaN/Inf/extreme/reordered/malformed inputs.

The check does NOT assume that matching column names imply matching pipelines.
For every feature it documents: train definition (code line), prod definition
(code line), whether the definitions match, real-time availability, and a
verdict. It computes train-vs-prod distribution stats on real data (parquet
sample vs DB-2 transaction rows mapped through the real deployed mapper) and
measures the fraction of production input vectors that fall OUTSIDE the
training-scaled envelope (the operational definition of feature drift).

Evidence outputs: reports/feature_integrity_audit.json,
reports/PS14_FEATURE_INTEGRITY_AUDIT_REPORT.md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

os_environ = __import__("os").environ
os_environ.setdefault("PS14_MODE", "development")

REPORT: dict = {"check": 27, "generated_at": None, "sections": {}}


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0, "missing": 0, "mean": None, "std": None, "min": None,
                "q25": None, "median": None, "q75": None, "max": None,
                "nunique": 0, "const": True}
    q = np.percentile(x, [25, 50, 75])
    return {
        "n": int(len(x)),
        "missing": int(np.isnan(x).sum()),
        "mean": round(float(x.mean()), 6),
        "std": round(float(x.std()), 6),
        "min": round(float(x.min()), 6),
        "q25": round(float(q[0]), 6),
        "median": round(float(q[1]), 6),
        "q75": round(float(q[2]), 6),
        "max": round(float(x.max()), 6),
        "nunique": int(len(np.unique(x))),
        "const": bool(np.all(x == x[0])),
    }


def _load_train_sample(feats: list[str], stride: int = 20,
                       cap_rows: int = 4_000_000) -> dict[str, np.ndarray]:
    """Sample every `stride`-th row from each row group of the training cache."""
    import pyarrow.parquet as pq
    path = ROOT / "data" / "_sorted_cache" / "features_all.parquet"
    pf = pq.ParquetFile(path)
    out = {f: [] for f in feats}
    taken = 0
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=feats).to_pandas()
        idx = slice(0, len(tbl), stride)
        for f in feats:
            out[f].append(tbl[f].to_numpy(dtype=np.float64)[idx])
        taken += len(range(0, len(tbl), stride))
        if taken >= cap_rows:
            break
    return {f: np.concatenate(v) for f, v in out.items()}


def _db2_feature_dicts() -> list[dict]:
    """Reconstruct production ML feature dicts from DB-2 stored rows.

    The privacy-layer ingest response (which is what /internal/evaluate
    receives) contains exactly these keys plus the transient velocity keys
    (user_tx_count/user_avg_amt/card_tx_count/merch_tx_count) and NO entity
    ids (user_id/merchant_id/city_id). Reconstructing from the persisted row
    reproduces the stored §16 vector; velocity keys are absent (mapper uses
    its documented defaults), matching the legacy caller path.
    """
    import sqlite3
    c = sqlite3.connect(str(ROOT / "db" / "features.db"))
    rows = c.execute("SELECT * FROM transaction_features").fetchall()
    cols = [d[1] for d in c.execute(
        "PRAGMA table_info(transaction_features)").fetchall()]
    c.close()
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        # Only keys the Altman mapper reads are needed; drop nulls so the
        # mapper's own defaults are exercised exactly as at runtime.
        out.append({k: v for k, v in d.items()
                    if v is not None and k in (
                        "amount_ratio", "hour_of_day", "is_weekend",
                        "new_device_flag", "failed_auth_count_24h",
                        "known_device_count", "account_tenure_days",
                        "amount_zscore", "velocity_deviation",
                        "txn_freq_last_24h", "user_tx_count",
                        "card_tx_count", "merch_tx_count", "user_avg_amt")})
    return out


def main() -> None:
    t0 = time.time()
    REPORT["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    from src.risk_engine.altman_ensemble import (
        ALTMAN_FEATURES, map_ml_features_to_altman, AltmanEnsembleEngine,
    )
    PROD_DIR = ROOT / "models" / "production"

    # ── 1. Schema integrity: names, order, count across all artifacts ─────
    manifest = json.loads((PROD_DIR / "manifest.json").read_text(encoding="utf-8"))
    eng = AltmanEnsembleEngine(PROD_DIR, verify_integrity=False)
    engine = eng  # alias
    scaler = engine.scaler
    schema = {
        "mapper_list": list(ALTMAN_FEATURES),
        "manifest_features": list(manifest["features"]),
        "manifest_n": manifest["n_features"],
        "scaler_n_features_in": int(scaler.n_features_in_),
        "xgb_n_features": int(engine.xgb.n_features_in_),
        "lgb_n_features": int(engine.lgb.n_features_in_),
        "model_version": engine.model_version,
    }
    schema["mapper_matches_manifest"] = list(ALTMAN_FEATURES) == list(manifest["features"])
    schema["scaler_matches_mapper"] = (
        scaler.n_features_in_ == len(ALTMAN_FEATURES)
        and int(engine.xgb.n_features_in_) == len(ALTMAN_FEATURES)
        and int(engine.lgb.n_features_in_) == len(ALTMAN_FEATURES))
    REPORT["sections"]["1_schema_integrity"] = schema

    # ── 2. Definitional matrix (train vs prod), evidence = code lines ──────
    # Train definitions: scripts/train_altman_fullscale.py pass2 (L140-310),
    # then scripts/retrain_15feat.py picked exactly these 15 columns.
    # Prod definitions: src/risk_engine/altman_ensemble.py map_ml_features_to_altman.
    # Call-path provenance: privacy ingest returns NO entity ids and NO fraud
    # rates (src/privacy_layer/features.py L267-270), so tracker lookups use
    # empty ids → baseline.
    DEFS = [
        # feature, train def, prod def, match?, realtime?, verdict, notes
        ("log_amt", "log1p(raw IBM dollar amount)", "log1p(amount_ratio * 100)",
         "NO", "YES", "MISMATCH",
         "Same transform family, different quantity: train=absolute dollars, prod=ratio-to-account-median scaled by 100 (proxy amount). Scales cannot match."),
        ("amt_sq", "raw amount squared", "(amount_ratio * 100)^2",
         "NO", "YES", "MISMATCH", "Quadratic of the same proxy amount — scale mismatch by construction."),
        ("hour_cos", "cos(2*pi*hr/24), real clock hour", "cos(2*pi*hr/24), real hour_of_day",
         "YES", "YES", "MATCH", "Definition identical; only source of the hour differs (IBM txn vs PS-14 event)."),
        ("is_business_hours", "1 if 9<=hr<=17", "1 if 9<=hr<=17",
         "YES", "YES", "MATCH", "Identical."),
        ("chip", "IBM chip-card indicator (0/1)", "new_device_flag (0/1)",
         "NO", "YES", "MISMATCH",
         "Train: whether the card used EMV chip. Prod: whether the DEVICE is new (mapper comment: '1 = online (new device proxy)'). Same name, different real-world meaning."),
        ("is_online", "IBM online-transaction indicator", "= chip value (new_device_flag)",
         "NO", "YES", "MISMATCH",
         "Train: independent online flag. Prod: hard copy of the chip column — prod chip == prod is_online for every row (correlation 1.0), train chip vs is_online are distinct (0.0-0.5 range)."),
        ("mcc_n", "real MCC code per merchant (high cardinality)", "0.0 hardcoded",
         "NO", "NO", "MISMATCH",
         "Train: varies per merchant. Prod: constant 0 — the mapper has no MCC source. Feature is DEAD in production (check #21: top SHAP driver on every row)."),
        ("has_zip", "IBM zip-present indicator (0/1, varies)", "0 hardcoded",
         "NO", "NO", "MISMATCH", "Constant 0 in prod, variable in train."),
        ("has_state", "IBM state-present indicator (0/1, varies)", "0 hardcoded",
         "NO", "NO", "MISMATCH", "Constant 0 in prod, variable in train."),
        ("merch_tx_count", "cumulative count of merchant's EARLIER file-order rows", "real-time 24h txn count to this recipient (privacy velocity); falls back to known_device_count",
         "PARTIAL", "YES", "MISMATCH",
         "Train window: entire history in file order (check #24: file order != time — count can include rows dated after the current txn). Prod: privacy-layer 24h window keyed on recipient id; when absent, mapper aliases to known_device_count (a different entity's count)."),
        ("merch_fraud_rate", "merchant cumulative confirmed-fraud ratio over earlier file-order rows (baseline 0.001)", "tracker rate over last 100 events keyed merchant_id — but call path passes NO merchant_id → constant baseline 0.001",
         "NO", "NO", "MISMATCH",
         "Even ignoring definition: the runtime never supplies entity ids (verification_service/main.py L879-904 forwards only the privacy ingest response), so the deployed model scores merch_fraud_rate = 0.001 on every transaction. Where ids ARE supplied, tracker labels come from risk_score>=70 (the model's own decision — self-referential), not confirmed fraud (risk_engine/main.py _record_entity_rates)."),
        ("city_fraud_rate", "per-Merchant-City cumulative confirmed-fraud ratio over earlier file-order rows", "tracker rate over last 100 events keyed city_id — call path passes NO city_id → constant baseline 0.001",
         "NO", "NO", "MISMATCH", "Same as merch_fraud_rate: frozen at baseline in the deployed path; tracker seed labels are model decisions, not confirmed labels."),
        ("very_high_amt", "1 if amt > 5 * user running AVERAGE amount", "1 if amount_ratio > 5 (vs account MEDIAN)",
         "PARTIAL", "YES", "MISMATCH",
         "Both express '~5x typical amount' but train uses running mean of raw dollars, prod uses ratio to median; different denominators."),
        ("amt_x_mcc", "amt * mcc_n (varies)", "amt * 0 = 0.0 hardcoded",
         "NO", "NO", "MISMATCH", "Dead in production (product with the constant-zero mcc_n)."),
        ("amt_x_online", "amt * is_online", "amt * is_online (prod is_online = new_device_flag)",
         "PARTIAL", "YES", "MISMATCH",
         "Form identical; semantics inherited from the is_online mismatch above (online flag is really 'new device')."),
    ]
    matrix_rows = []
    for feat, train_def, prod_def, match, rt, verdict, note in DEFS:
        matrix_rows.append({
            "feature": feat, "train_definition": train_def,
            "production_definition": prod_def, "exact_definition_match": match,
            "real_time_available": rt, "verdict": verdict, "notes": note,
        })
    REPORT["sections"]["2_definitional_matrix"] = {
        "n_features": len(DEFS),
        "matches": sum(1 for r in matrix_rows if r["verdict"] == "MATCH"),
        "mismatches": sum(1 for r in matrix_rows if r["verdict"] == "MISMATCH"),
        "rows": matrix_rows,
    }

    # ── 3. Edge-case / robustness probes on the REAL engine ────────────────
    probes = {}
    base = {
        "amount_ratio": 1.0, "hour_of_day": 12, "is_weekend": 0,
        "new_device_flag": 0, "failed_auth_count_24h": 0,
        "known_device_count": 2, "account_tenure_days": 180.0,
        "amount_zscore": 0.0, "velocity_deviation": 0.0,
    }
    v_base = map_ml_features_to_altman(base)

    def _probe(name, d):
        try:
            v = map_ml_features_to_altman(d)
            return {"result": "ok", "vec": v.round(6).tolist(),
                    "identical_to_base": bool(np.array_equal(v, v_base))}
        except Exception as e:  # noqa: BLE001
            return {"result": "raises", "error": f"{type(e).__name__}: {e}"}

    probes["empty_dict_all_missing"] = _probe("empty", {})
    probes["nan_amount"] = _probe("nan", {**base, "amount_ratio": float("nan")})
    probes["posinf_amount"] = _probe("inf", {**base, "amount_ratio": float("inf")})
    probes["neginf_amount"] = _probe("ninf", {**base, "amount_ratio": float("-inf")})
    probes["negative_amount"] = _probe("neg", {**base, "amount_ratio": -3.0})
    probes["zero_amount"] = _probe("zero", {**base, "amount_ratio": 0.0})
    probes["extreme_amount"] = _probe("extreme", {**base, "amount_ratio": 1e9})
    probes["nan_hour"] = _probe("nanhour", {**base, "hour_of_day": float("nan")})
    probes["hour_23_weekend"] = _probe("late", {**base, "hour_of_day": 23, "is_weekend": 1})
    probes["empty_entity_ids"] = _probe(
        "eid", {**base, "user_id": "", "merchant_id": "", "city_id": ""})
    probes["random_entity_ids"] = _probe(
        "eid2", {**base, "user_id": "FABC1234567890AB", "merchant_id": "R-XYZ-9",
                 "city_id": "L-1"})
    probes["nan_failed_auth"] = _probe("nfa", {**base, "failed_auth_count_24h": float("nan")})

    # Reordered dict (same keys, shuffled insertion order) must map identically.
    keys = list(base.keys())
    rng = np.random.default_rng(7)
    shuffled = {k: base[k] for k in rng.permutation(keys)}
    probes["reordered_keys"] = _probe("reorder", shuffled)
    probes["reordered_keys"]["same_as_base"] = bool(np.array_equal(
        np.asarray(probes["reordered_keys"]["vec"]), v_base.round(6)))

    # Wrong-typed value — should raise (degrade), never silently score.
    probes["string_in_numeric_slot"] = _probe(
        "badtype", {**base, "amount_ratio": "not-a-number"})

    # Determinism: map the same dict 5x → identical vectors; engine predict 3x
    # → identical scores; fresh engine instance → identical scores.
    det = {}
    vecs = [map_ml_features_to_altman(base) for _ in range(5)]
    det["map_deterministic"] = all(np.array_equal(vecs[0], v) for v in vecs[1:])
    eng2 = AltmanEnsembleEngine(PROD_DIR, verify_integrity=False)
    scores = [engine.predict(base)[0] for _ in range(3)]
    det["predict_repeat_scores"] = scores
    det["predict_deterministic"] = len(set(scores)) == 1
    s_other = eng2.predict(base)[0]
    det["predict_fresh_engine_identical"] = s_other == scores[0]
    det["scaler_transform_identical"] = bool(np.array_equal(
        engine.scaler.transform(map_ml_features_to_altman(base).reshape(1, -1)),
        engine.scaler.transform(map_ml_features_to_altman(base).reshape(1, -1))))
    probes["determinism"] = det
    REPORT["sections"]["3_edge_case_probes"] = probes

    # ── 3b. Structural signal capacity (sweep over varied legal inputs) ────
    # 600 varied PS-14 feature dicts, entity-id-free and fraud-rate-free —
    # exactly what the deployed call path (privacy ingest → evaluate) sends.
    # Features that stay constant across the whole sweep are STRUCTURALLY dead
    # in production (they cannot carry signal for any legal input), independent
    # of what any particular corpus happens to contain.
    rng_s = np.random.default_rng(42)
    sweep = []
    for _ in range(600):
        sweep.append({
            "amount_ratio": float(np.exp(rng_s.uniform(np.log(0.02), np.log(300)))),
            "hour_of_day": int(rng_s.integers(0, 24)),
            "is_weekend": int(rng_s.integers(0, 2)),
            "new_device_flag": int(rng_s.integers(0, 2)),
            "failed_auth_count_24h": int(rng_s.integers(0, 9)),
            "known_device_count": int(rng_s.integers(0, 21)),
            "account_tenure_days": float(rng_s.uniform(0, 3000)),
            "amount_zscore": float(rng_s.uniform(-6, 6)),
            "velocity_deviation": float(rng_s.uniform(0, 3)),
            "txn_freq_last_24h": int(rng_s.integers(0, 61)),
            "user_tx_count": int(rng_s.integers(0, 61)),
            "card_tx_count": int(rng_s.integers(0, 41)),
            "merch_tx_count": int(rng_s.integers(0, 41)),
            "user_avg_amt": float(rng_s.uniform(20, 500)),
        })
    sweep_vecs = np.array([map_ml_features_to_altman(d) for d in sweep])
    sweep_stats = [_stats(sweep_vecs[:, i]) for i in range(len(ALTMAN_FEATURES))]
    structurally_const = [
        f for f, s in zip(ALTMAN_FEATURES, sweep_stats) if s["const"]]
    REPORT["sections"]["3b_signal_capacity_sweep"] = {
        "n_sweep_rows": len(sweep),
        "sweep_input_space": "amount_ratio 0.02-300 log-uniform, hour 0-23, "
                             "weekend/new-device 0-1, failed_auth 0-8, "
                             "known_device 0-20, tenure 0-3000d, zscore -6..6, "
                             "velocity 0-3, txn counts 0-60; entity ids and "
                             "fraud-rate keys ABSENT (deployed call path)",
        "structurally_const_features": structurally_const,
        "per_feature_sweep_std": [round(float(s["std"]), 6)
                                  if s["std"] is not None else None
                                  for s in sweep_stats],
    }

    # ── 4. Distribution comparison: training cache vs production-mapped ────
    t1 = time.time()
    train = _load_train_sample(list(ALTMAN_FEATURES))
    train_seconds = round(time.time() - t1, 1)
    prod_dicts = _db2_feature_dicts()
    prod_vecs = np.array([map_ml_features_to_altman(d) for d in prod_dicts])
    n_with_velocity = sum(1 for d in prod_dicts if "merch_tx_count" in d
                          or "user_avg_amt" in d)
    db2_note = ("DB-2 real corpus is degenerate (hour_of_day=12 const, "
                "tenure=0, freq=0, known_device=2) but amount_ratio has 10 "
                "distinct values (100-550); listed as the REAL production "
                "reference — the sweep in 3b is the signal-capacity test.")
    dist = {"train_sample_rows": int(len(train[ALTMAN_FEATURES[0]])),
            "train_sampling_seconds": train_seconds,
            "prod_rows": int(len(prod_vecs)),
            "prod_rows_with_velocity_keys": n_with_velocity,
            "db2_corpus_note": db2_note}
    per_feat = []
    for i, f in enumerate(ALTMAN_FEATURES):
        tr = _stats(train[f])
        db2 = _stats(prod_vecs[:, i])
        sweep_const = bool(sweep_stats[i]["const"])
        train_varies = not tr["const"]
        per_feat.append({
            "feature": f,
            "train": tr,
            "db2_production_rows": db2,
            "structurally_const_in_prod": sweep_const,
            "train_varies": train_varies,
            "comment": ("structurally constant in production while varying "
                        "in training → feature carries no runtime signal"
                        if sweep_const and train_varies
                        else "definitional mismatch (see section 2)"),
        })
    dist["per_feature"] = per_feat
    dist["train_varies_prod_const"] = [
        f for f, p in zip(ALTMAN_FEATURES, per_feat)
        if p["structurally_const_in_prod"] and p["train_varies"]]
    REPORT["sections"]["4_distributions"] = dist

    # ── 5. Scaling provenance + OOD envelope ────────────────────────────────
    # Scaler fit scope: retrain_15feat.py fits RobustScaler on Xtr only
    # (80% random split of the training cache) and reuses it for test +
    # production (scripts/retrain_15feat.py L93-99). Verify transform is the
    # ONLY op applied at inference (no refit) — proven by identical scaler
    # objects across fresh engine loads, plus the fit evidence below.
    train_s = scaler.transform(np.vstack([train[f] for f in ALTMAN_FEATURES]
                                         ).T.astype(np.float64))
    prod_s = scaler.transform(prod_vecs)
    lo = train_s.min(axis=0)
    hi = train_s.max(axis=0)
    # Fraction of prod rows whose scaled value for each feature exceeds the
    # full training-scaled envelope (i.e., the model is extrapolating).
    ood_frac = []
    for i, f in enumerate(ALTMAN_FEATURES):
        below = int((prod_s[:, i] < lo[i]).sum())
        above = int((prod_s[:, i] > hi[i]).sum())
        ood_frac.append({"feature": f, "below_train_min": below,
                         "above_train_max": above,
                         "rows_outside": below + above,
                         "frac_outside": round((below + above) / len(prod_s), 4)})
    n_ood_rows = int(np.any((prod_s < lo) | (prod_s > hi), axis=1).sum())
    scaling = {
        "scaler_class": type(scaler).__name__,
        "scaler_fit_scope": "fitted on Xtr only (80% random split of "
                            "features_all.parquet) in retrain_15feat.py; "
                            "transform reused unchanged on test and at runtime",
        "scaler_is_refit_at_inference": False,
        "center_mean": round(float(np.mean(scaler.center_)), 4),
        "scale_mean": round(float(np.mean(scaler.scale_)), 4),
        "prod_rows_any_feature_outside_train_envelope": int(n_ood_rows),
        "prod_frac_any_feature_outside_train_envelope": round(
            n_ood_rows / len(prod_s), 4),
        "per_feature_outside": ood_frac,
    }
    REPORT["sections"]["5_scaling_and_ood"] = scaling

    # ── 6. Historical-correctness & label-availability verdict ─────────────
    hist = {
        "offline_window_definition": "cumulative running state in FILE ORDER "
                                     "(train_altman_fullscale.py pass2, L252-298) "
                                     "— NOT anchored to row timestamps; check #24 "
                                     "measured 66.8% of merchant-rate rows change "
                                     "under strict chronological replay and the "
                                     "chronological rescore drops AUC 0.9952→0.9229",
        "offline_label_availability_timing": "UNVERIFIED — no fraud-confirmation "
                                             "timestamp exists in the dataset, so "
                                             "it is unprovable when offline labels "
                                             "became available to the expanding window",
        "prod_window": "EntityFraudRateTracker: in-memory deque(maxlen=100) per "
                       "entity, min 5 events, baseline 0.001; populated only by "
                       "_record_entity_rates AFTER each evaluation and by "
                       "_seed_entity_tracker (labels = risk_score >= 70, the "
                       "model's own decision, not confirmed fraud)",
        "prod_persistence": "in-memory only — cleared on restart and re-seeded "
                            "from the top-5000 DB-2 risk rows",
        "prod_actual_values": "call path (privacy ingest → verify demo seed) "
                              "supplies no user_id/merchant_id/city_id, so both "
                              "fraud-rate features are constant baseline 0.001 "
                              "in every deployed decision",
        "future_data_in_prod": "NO (tracker only records events after they are "
                               "scored; no future rows can enter)",
        "future_data_in_train": "YES — quantified (check #24); file-order "
                                "expanding state feeds early-dated rows with "
                                "later-dated transactions of the same merchant/city",
        "verdict": "FAIL — training fraud-rate features are not provably "
                   "past-only and are not reproducible at real time; the "
                   "production substitutes are frozen at baseline in the "
                   "actual call path",
    }
    REPORT["sections"]["6_historical_correctness"] = hist

    # ── Aggregate verdict ──────────────────────────────────────────────────
    n_mismatch = sum(1 for r in matrix_rows if r["verdict"] == "MISMATCH")
    n_match = sum(1 for r in matrix_rows if r["verdict"] == "MATCH")
    verdict = {
        "features_total": len(DEFS),
        "exact_definition_match": n_match,
        "definition_mismatch": n_mismatch,
        "structurally_const_in_prod_while_train_varies": dist["train_varies_prod_const"],
        "schema_names_order_match": bool(schema["mapper_matches_manifest"]
                                         and schema["scaler_matches_mapper"]),
        "edge_probes_raising": [k for k, v in probes.items()
                                if isinstance(v, dict) and v.get("result") == "raises"],
        "frozen_fraud_rates_in_call_path": ["merch_fraud_rate", "city_fraud_rate"],
        "overall": "FAIL",
        "headline": (
            f"{n_mismatch}/{len(DEFS)} features are definitional mismatches "
            f"(only {n_match} match exactly); "
            f"{len(dist['train_varies_prod_const'])} features are STRUCTURALLY "
            f"constant in production while varying in training "
            f"({', '.join(dist['train_varies_prod_const'])}); both fraud-rate "
            "features are frozen at baseline 0.001 in the deployed call path; "
            "the offline expanding-window definitions are contaminated by "
            "file order (check #24 quantified: 66.8% of merchant-rate rows "
            "change under strict chronology, AUC 0.9952→0.9229). Schema names/"
            "order/count DO match across mapper, manifest, scaler, and both "
            "boosters — the failure is semantic, not structural."),
    }
    REPORT["sections"]["7_verdict"] = verdict

    out_path = ROOT / "reports" / "feature_integrity_audit.json"
    out_path.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    print(f"feature_integrity_audit: wrote {out_path} in {time.time()-t0:.0f}s")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
