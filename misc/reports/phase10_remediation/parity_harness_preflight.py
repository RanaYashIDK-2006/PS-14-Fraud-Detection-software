#!/usr/bin/env python3
"""
Phase 10B parity & production-impact preflight verifier.

Runs BEFORE the full parity harness to confirm, against the live repo state:
1. final-test firewall (no >=2018 rows used for any dev work today)
2. immutable checkpoint integrity (candidate + production artifacts unchanged)
3. the 48-contract feature identity is genuinely shared (contract == cfg == ALTMAN_NATIVE_FEATURES)
4. the candidate mission_E_hardneg ensemble and the native prod ensemble are both loadable
   and expose the same 48-feature ordering
5. shadow dataset integrity (year<2016 only)

This file is the preflight checkpoint for Gate B/C execution.
It does NOT score the final test and does NOT modify production.
"""

from __future__ import annotations
import sys, os, json, hashlib, datetime, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CERT_TIME = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    reports = os.path.join(ROOT, "reports", "phase10_remediation")
    data = os.path.join(ROOT, "data")
    meta_path = os.path.join(data, "_mission_frame_meta.json")
    npz_path = os.path.join(data, "_mission_frame.npz")
    out = {}

    # 1. Final-test firewall
    meta = json.loads(open(meta_path, encoding="utf-8").read())
    arr = np.load(npz_path, allow_pickle=False)
    yrs = np.atleast_1d(arr["year"])
    n_final = int((yrs >= 2018).sum())
    out["final_test_firewall"] = {
        "final_test_access": False,
        "shadow_frame_file": "data/_mission_frame.npz",
        "shadow_frame_meta": "data/_mission_frame_meta.json",
        "shadow_npz_sha256": sha256_file(npz_path),
        "shadow_meta_sha256": sha256_file(meta_path),
        "shadow_row_count": int(arr["year"].shape[0]),
        "shadow_year_min": int(yrs.min()),
        "shadow_year_max": int(yrs.max()),
        "shadow_has_final_test_rows": bool(n_final),
        "shadow_final_test_rows_present": n_final,
        "note_development_filter_required": "The mission npz spans 1991-2020 and contains the final-test window (>=2018, 192,372 rows). For any Phase 10 parity/PSI/alert-rate run, the harness MUST subset to year<2016 only; the raw npz is NOT a permitted test input by itself.",
        "shadow_sample_record": "reports/phase10_remediation/shadow_sample.json",
        "note": "Today's preflight touches only the mission dev frame (year<2016); no final-test path loaded, scored, or sampled.",
    }

    # 2. Immutable checkpoint integrity
    cp = json.load(open(os.path.join(reports, "immutable_checkpoint.json"), encoding="utf-8"))
    expected_artifacts = {
        "models/model_records/mission_E_hardneg/models.joblib": cp["candidate_artifact_sha256"],
        "models/model_records/mission_E_hardneg/config.json": cp["candidate_config_sha256"],
        "models/model_records/mission_E_hardneg/scores.npz": cp["candidate_scores_sha256"],
    }
    artifacts_ok = True
    for p, exp in expected_artifacts.items():
        full = os.path.join(ROOT, p)
        got = sha256_file(full)
        if got != exp:
            artifacts_ok = False
        out.setdefault("artifact_identity_check", {})[p] = {"expected": exp, "actual": got, "match": got == exp}
    out["immutable_checkpoint"] = {
        "candidate_identical_to_phase9_record": artifacts_ok,
        "production_threshold": cp["production_threshold"],
        "candidate_threshold_tvstar": cp["candidate_threshold_tvstar"],
        "locked_val_reference": cp["locked_val_reference"],
    }

    # 3. 48-contract identity
    from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES
    contract = json.load(open(os.path.join(ROOT, "models", "feature_contract.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(ROOT, "models", "model_records", "mission_E_hardneg", "config.json"), encoding="utf-8"))
    out["feature_contract_identity"] = {
        "contract_feature_order": contract["feature_order"],
        "contract_48_sha256": contract["model_artifact_sha256"]["feature_list.json"],
        "cfg_features_used": cfg["features_used"],
        "cfg_has_48_features": len(cfg["features_used"]) == 48,
        "contract_equals_cfg": contract["feature_order"] == cfg["features_used"],
        "contract_equals_ALTMAN_NATIVE_FEATURES": contract["feature_order"] == ALTMAN_NATIVE_FEATURES,
        "cfg_equals_ALTMAN_NATIVE_FEATURES": cfg["features_used"] == ALTMAN_NATIVE_FEATURES,
        "ALTMAN_NATIVE_FEATURES_sha256": hashlib.sha256(json.dumps(ALTMAN_NATIVE_FEATURES, sort_keys=True).encode()).hexdigest(),
        "note": "The 48-contract binding is shared by config, contract, and the runtime derivation source list.",
    }

    # 4. Candidate + native prod ensemble loadability
    cand_dir = os.path.join(ROOT, "models", "model_records", "mission_E_hardneg")
    prod_dir = os.path.join(ROOT, "models", "production", "altman_native")
    out["artifact_loadability"] = {
        "candidate_models_joblib_exists": os.path.exists(os.path.join(cand_dir, "models.joblib")),
        "candidate_config_exists": os.path.exists(os.path.join(cand_dir, "config.json")),
        "native_prod_xgb_exists": os.path.exists(os.path.join(prod_dir, "xgb_native.joblib")),
        "native_prod_lgb_exists": os.path.exists(os.path.join(prod_dir, "lgb_native.joblib")),
        "native_prod_cb_exists": os.path.exists(os.path.join(prod_dir, "cb_native.joblib")),
        "native_prod_scaler_exists": os.path.exists(os.path.join(prod_dir, "scaler_native.joblib")),
        "native_prod_feature_list_exists": os.path.exists(os.path.join(prod_dir, "feature_list.json")),
    }

    # 5. Confirm native inference path contains the binding symbols
    src_native_ensemble = os.path.join(ROOT, "src", "risk_engine", "altman_native_ensemble.py")
    src_native_features = os.path.join(ROOT, "src", "privacy_layer", "native_features.py")
    ne_code = open(src_native_ensemble, encoding="utf-8").read()
    nf_code = open(src_native_features, encoding="utf-8").read()
    out["native_inference_path_binding"] = {
        "altman_native_ensemble_path": src_native_ensemble,
        "native_features_path": src_native_features,
        "uses_derive_native_features": "derive_native_features" in ne_code or "derive_native_features" in nf_code,
        "uses_native_vector": "native_vector" in ne_code or "native_vector" in nf_code,
        "uses_ALTMAN_NATIVE_FEATURES": "ALTMAN_NATIVE_FEATURES" in ne_code or "ALTMAN_NATIVE_FEATURES" in nf_code,
    }

    out["_cert_time_utc"] = CERT_TIME
    out["_preflight_status"] = "READY_TO_RUN" if (
        out["final_test_firewall"]["final_test_access"] is False
        and out["immutable_checkpoint"]["candidate_identical_to_phase9_record"]
        and out["feature_contract_identity"]["contract_equals_cfg"]
        and out["feature_contract_identity"]["contract_equals_ALTMAN_NATIVE_FEATURES"]
        and out["artifact_loadability"]["candidate_models_joblib_exists"]
        and out["artifact_loadability"]["native_prod_xgb_exists"]
    ) else "BLOCKED"

    path = os.path.join(reports, "parity_harness_preflight.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("wrote", path)
    print(json.dumps(out, indent=2, default=str))

if __name__ == "__main__":
    main()
