#!/usr/bin/env python3
"""Check #28 — Model release, rollback & disaster-recovery drill.

Exercises the FULL model lifecycle against a SANDBOX copy of the deployed
artifact set (models/_drill_<ts>/) — the real models/production directory is
never modified:

  Phase 1  Inventory & integrity of the ACTIVE artifact set (record gaps,
           undeclared cb member, engine-vs-manifest weight divergence).
  Phase 2  Baseline: known-good engine outputs on a fixed canary set.
  Phase 3  Atomicity: partial activation must be caught (record-backed
           integrity verify) and must never serve a mixed set.
  Phase 4  Rollback drills (governance.rollback actually exercised):
           corrupt artifact / incompatible feature schema / bad model /
           abnormal score distribution / severe latency increase.
           Each: activate -> canary -> detection -> rollback -> verify outputs
           equal the known-good baseline.
  Phase 5  Disaster recovery: storage unavailable, cold restart (RTO),
           temp-state loss (tracker cold start), backup restoration
           (scripts/backup_restore_test.py run separately).
  Phase 6  Verdict + matrix.

Evidence: reports/model_release_drill.json, reports/PS14_MODEL_RECOVERY_AUDIT_REPORT.md
"""
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

PROD = ROOT / "models" / "production"
ARTIFACTS = ROOT / "models" / "artifacts"

REPORT: dict = {"check": 28, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sandbox": None, "phases": {}}

# Engine files that make up a release (what AltmanEnsembleEngine reads).
ENGINE_FILES = ["manifest.json", "feature_list.json", "xgb_production.joblib",
                "lgb_production.joblib", "cb_production.joblib",
                "scaler_production.joblib"]
CALIBRATOR = "calibrator.joblib"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _ConstModel:  # module-level so joblib can pickle it
    """Deliberately broken model: constant predict_proba (drill only)."""
    def __init__(self, prob: float):
        self.prob = prob
        self.n_features_in_ = 15

    def predict_proba(self, X):
        return np.tile([1.0 - self.prob, self.prob], (len(X), 1)).astype(float)


class _SlowLGB:  # module-level so joblib can pickle it
    """Deliberately slow wrapper over the real lgb booster (drill only)."""
    def __init__(self, real_lgb, delay_s: float = 0.25):
        self._inner = real_lgb
        self._delay = delay_s
        self.n_features_in_ = real_lgb.n_features_in_

    def predict_proba(self, X):
        time.sleep(self._delay)
        return self._inner.predict_proba(X)


class _NoiseModel:  # module-level so joblib can pickle it
    """Deliberately broken model: random scores uncorrelated with input."""
    def __init__(self):
        self.n_features_in_ = 15
        self._rng = np.random.default_rng(99)

    def predict_proba(self, X):
        return np.column_stack([1.0 - self._rng.random(len(X)),
                                self._rng.random(len(X))])


def _canary_rows(n: int = 24, seed: int = 123) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append({
            "amount_ratio": float(np.exp(rng.uniform(np.log(0.05), np.log(200)))),
            "hour_of_day": int(rng.integers(0, 24)),
            "is_weekend": int(rng.integers(0, 2)),
            "new_device_flag": int(rng.integers(0, 2)),
            "failed_auth_count_24h": int(rng.integers(0, 9)),
            "known_device_count": int(rng.integers(0, 21)),
            "account_tenure_days": float(rng.uniform(0, 3000)),
            "amount_zscore": float(rng.uniform(-6, 6)),
            "velocity_deviation": float(rng.uniform(0, 3)),
            "user_tx_count": int(rng.integers(0, 61)),
            "card_tx_count": int(rng.integers(0, 41)),
            "merch_tx_count": int(rng.integers(0, 41)),
            "user_avg_amt": float(rng.uniform(20, 500)),
        })
    return rows


def _run_canary(eng, rows: list[dict]) -> dict:
    scores, raws, lats = [], [], []
    t0 = time.perf_counter()
    for r in rows:
        s = time.perf_counter()
        p, u = eng.predict(r)
        lats.append((time.perf_counter() - s) * 1000)
        scores.append(p)
        raws.append(u.get("ensemble_raw", p))
    return {"scores": scores, "raws": raws, "latencies_ms": lats,
            "wall_ms": (time.perf_counter() - t0) * 1000}


def _score_stats(res: dict, key: str = "scores") -> dict:
    s = np.array(res[key])
    l = np.array(res["latencies_ms"])
    return {"n": int(len(s)), "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6), "max": round(float(s.max()), 6),
            "p50_lat": round(float(np.percentile(l, 50)), 3),
            "p95_lat": round(float(np.percentile(l, 95)), 3),
            "p99_lat": round(float(np.percentile(l, 99)), 3)}


def _copy_engine_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ENGINE_FILES:
        shutil.copy2(src / name, dst / name)


def load_engine(active_dir: Path, verify: bool) -> "object":
    from src.risk_engine.altman_ensemble import AltmanEnsembleEngine
    return AltmanEnsembleEngine(active_dir, verify_integrity=verify)


def main() -> None:
    # ── Phase 1: inventory & integrity of the ACTIVE set ──────────────────
    manifest = json.loads((PROD / "manifest.json").read_text(encoding="utf-8"))
    active = {
        "model_version": manifest["model_version"],
        "manifest_model_type": manifest["model_type"],
        "manifest_ensemble_weights": manifest["ensemble_weights"],
        "lean": manifest.get("lean"),
        "engine_files_present": {f: (PROD / f).exists()
                                 for f in ENGINE_FILES},
    }
    # Engine weight constant & live third member
    from src.risk_engine.altman_ensemble import ENSEMBLE_WEIGHTS, ALTMAN_FEATURES
    active["engine_weight_constant"] = ENSEMBLE_WEIGHTS
    cb = None
    if (PROD / "cb_production.joblib").exists():
        import joblib
        cb = joblib.load(PROD / "cb_production.joblib")
        active["cb_loaded_by_engine"] = True
        active["cb_declared_in_manifest"] = "cb" in manifest.get(
            "ensemble_weights", {}) or manifest["model_type"] != "xgb_lgb_ensemble_lean"
        active["cb_feature_names"] = list(cb.feature_names_ or [])
        active["cb_feature_names_are_altman"] = list(cb.feature_names_ or []) == ALTMAN_FEATURES
    else:
        active["cb_loaded_by_engine"] = False

    # Governance record completeness for the ACTIVE model
    rec_path = ROOT / "models" / "model_records" / f"{manifest['model_version']}.json"
    record_gaps = []
    record_fields = {}
    if rec_path.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        record_fields = rec
        for field in ["selected_threshold", "training_period", "validation_period",
                      "final_test_period"]:
            if not rec.get(field):
                record_gaps.append(f"record.{field} empty/missing (deployed model "
                                   "has no locked threshold / period provenance)")
        if "hash" not in rec.get("code_version", "").lower() and "commit" not in rec.get("code_version", "").lower():
            record_gaps.append(f"record.code_version is a description, not a hash: "
                               f"'{rec.get('code_version')}'")
        hp = rec.get("hyperparameters", {})
        if not hp or "could not introspect" in json.dumps(hp):
            record_gaps.append("record.hyperparameters not captured (introspection failed)")
        gates = rec.get("promotion", {}).get("gates", {})
        for g, v in gates.items():
            if not (str(v).upper().startswith("PASS") or str(v).upper() in ("TRUE", "APPROVED")):
                record_gaps.append(f"promotion gate '{g}' = {v}")
        if not gates:
            record_gaps.append("no promotion gate record")
        # record lists extra/stale artifacts -> what the record verified
        extra = [n for n in rec.get("artifact_files", {})
                 if n not in ENGINE_FILES and n != "latest"]
        if extra:
            active["record_covers_stale_artifacts"] = extra
    else:
        record_gaps.append("NO governance record for the active model version")
    active["governance_record_exists"] = rec_path.exists()
    active["record_gaps"] = record_gaps
    REPORT["phases"]["1_active_inventory"] = active

    # ── Phase 2: sandbox + baseline ───────────────────────────────────────
    sandbox = Path(tempfile.mkdtemp(prefix="ps14_release_drill_",
                                    dir=str(ROOT / "models")))
    REPORT["sandbox"] = str(sandbox)
    active_dir = sandbox / "production"
    artifacts_dir = sandbox / "artifacts"
    rel_dir = sandbox / "releases"
    recs_dir = sandbox / "records"
    rel_dir.mkdir(); recs_dir.mkdir(); artifacts_dir.mkdir()
    try:
        _copy_engine_files(PROD, active_dir)
        if (ARTIFACTS / CALIBRATOR).exists():
            shutil.copy2(ARTIFACTS / CALIBRATOR, artifacts_dir / CALIBRATOR)
        # governance record for the drill's known-good (engine files only)
        from src.risk_engine.model_governance import ModelRecord, rollback, verify_artifacts
        good_id = "drill_known_good_v1"
        # Record scope = the engine-file set (calibrator is loaded from
        # <prod_dir>/../artifacts and is out of rollback-archive scope here;
        # its provenance gap is documented in phase 1).
        files = {}
        for name in ENGINE_FILES:
            p = PROD / name
            if p.exists():
                files[name] = {"sha256": sha256_file(p), "size": p.stat().st_size}
        good = ModelRecord(
            model_id=good_id,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            training_dataset_path=str(PROD),
            feature_schema_version="altman_lean_v1",
            features=list(ALTMAN_FEATURES),
            algorithm="xgb_lgb_cb_ensemble (drill sandbox)",
            artifact_files=files,
            code_version="release-drill sandbox",
        )
        good.save(recs_dir)

        t0 = time.perf_counter()
        good_eng = load_engine(active_dir, verify=False)
        REPORT["phases"]["2_good_load_seconds"] = round(
            time.perf_counter() - t0, 3)
        canary = _canary_rows()
        base_res = _run_canary(good_eng, canary)
        base = _score_stats(base_res)
        base_raw = _score_stats(base_res, key="raws")
        REPORT["phases"]["2_baseline"] = {"calibrated": base, "raw": base_raw}

        # ── Phase 3: atomicity / partial-activation test ──────────────────
        # Current deploy style (retrain_15feat): overwrite in place, manifest
        # last. Simulate an interrupted deploy: NEW lgb bytes + OLD manifest.
        # With a governance record present, the load-time verify (what
        # main.py enables: verify_integrity=True) MUST refuse the mixed set.
        atomic = {}
        backup_dir = sandbox / "atomic_backup"
        _copy_engine_files(active_dir, backup_dir)
        # write garbage into lgb (as if the new model landed first)
        (active_dir / "lgb_production.joblib").write_bytes(b"PARTIAL" * 100)
        t0 = time.perf_counter()
        try:
            load_engine(active_dir, verify=True)
            atomic["detected"] = False
            atomic["detail"] = "mixed set LOADED silently (FAIL)"
        except Exception as e:  # noqa: BLE001
            atomic["detected"] = True
            atomic["detect_seconds"] = round(time.perf_counter() - t0, 3)
            atomic["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        # restore
        for name in ENGINE_FILES:
            shutil.copy2(backup_dir / name, active_dir / name)
        e2 = load_engine(active_dir, verify=False)
        s2 = _score_stats(_run_canary(e2, canary))
        atomic["restored_outputs_equal_baseline"] = (
            s2["mean"] == base["mean"] and s2["std"] == base["std"])
        REPORT["phases"]["3_atomicity_partial_activation"] = atomic

        # ── Phase 4: rollback drills ───────────────────────────────────────
        drills = []
        import joblib as _jl

        # 4b: incompatible schema — tiny LGB trained on 3 features
        import lightgbm as lgb
        rng = np.random.default_rng(5)
        X3 = rng.normal(size=(300, 3))
        y3 = (X3[:, 0] > 0.5).astype(int)
        bad3 = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
        bad3.fit(X3, y3)

        # 4e: slow wrapper around the real lgb
        real_lgb = _jl.load(PROD / "lgb_production.joblib")

        scenario_files = {
            "corrupt_artifact": ("lgb_production.joblib",
                                 lambda: b"GARBAGE" * 2000, None),
            "incompatible_schema": ("lgb_production.joblib", lambda: bad3, None),
            "bad_model": ("lgb_production.joblib", lambda: _NoiseModel(), None),
            "abnormal_score_dist": ("lgb_production.joblib",
                                    lambda: _ConstModel(0.001), None),
            "severe_latency": ("lgb_production.joblib",
                               lambda: _SlowLGB(real_lgb), None),
        }

        for label, (fname, factory, _det) in scenario_files.items():
            d = {"scenario": label, "activated": label}
            # stage the bad artifact over a fresh copy of the good active dir
            for name in ENGINE_FILES:
                shutil.copy2(backup_dir / name, active_dir / name)
            obj = factory()
            if isinstance(obj, bytes):
                (active_dir / fname).write_bytes(obj)
            else:
                _jl.dump(obj, active_dir / fname)

            # activation check (what a release gate would do: load + canary)
            t0 = time.perf_counter()
            try:
                eng_bad = load_engine(active_dir, verify=False)
                res_bad = _run_canary(eng_bad, canary)
                load_error = None
            except Exception as e:  # noqa: BLE001
                res_bad = None
                load_error = f"{type(e).__name__}: {str(e)[:160]}"
            d["activation_load_seconds"] = round(time.perf_counter() - t0, 3)

            # detection. Use the RAW ensemble (pre-calibrator) for shift tests:
            # the deployed cross-family Platt calibrator compresses score
            # movement and can mask a bad model at the calibrated layer.
            if load_error is not None:
                d["detected"] = True
                d["detector"] = f"engine load/predict raised ({load_error})"
            else:
                st_cal = _score_stats(res_bad)
                st_raw = _score_stats(res_bad, key="raws")
                shift = abs(st_raw["mean"] - base_raw["mean"])
                if label == "severe_latency":
                    ok = st_raw["p95_lat"] <= 100.0  # canary SLO
                    d["detected"] = not ok
                    d["detector"] = (f"canary p95 {st_raw['p95_lat']}ms vs "
                                     f"100ms SLO")
                else:
                    # bad model / abnormal distribution: raw ensemble-mean
                    # shift. A constant-high member is NOT detectable here
                    # because the deployed trio itself saturates near 1.0 raw
                    # (documented finding); genuine damage (noise / collapse
                    # of one member) moves the raw mean materially.
                    d["detected"] = shift > 0.10
                    d["detector"] = (f"|raw mean shift| {shift:.4f} vs "
                                     f"baseline {base_raw['mean']:.4f}")
                d["bad_calibrated"] = st_cal
                d["bad_raw"] = st_raw

            # rollback via governance.rollback (real helper, real files)
            t0 = time.perf_counter()
            try:
                rb = rollback(recs_dir, backup_dir, active_dir,
                              bad_model_id=label, good_model_id=good_id)
                d["rollback"] = {"ok": True,
                                 "seconds": round(time.perf_counter() - t0, 3),
                                 "files_copied": rb["files_copied"],
                                 "post_rollback_verify_pass": rb["post_rollback_verify_pass"]}
            except Exception as e:  # noqa: BLE001
                d["rollback"] = {"ok": False,
                                 "seconds": round(time.perf_counter() - t0, 3),
                                 "error": f"{type(e).__name__}: {str(e)[:160]}"}

            # post-rollback verification: outputs equal baseline exactly
            t0 = time.perf_counter()
            eng_good2 = load_engine(active_dir, verify=False)
            d["post_rollback_reload_seconds"] = round(
                time.perf_counter() - t0, 3)
            s_after = _score_stats(_run_canary(eng_good2, canary))
            s_after_raw = _score_stats(_run_canary(eng_good2, canary),
                                       key="raws")
            d["rollback_correct"] = (
                s_after["mean"] == base["mean"]
                and s_after["std"] == base["std"]
                and s_after_raw["mean"] == base_raw["mean"]
                and s_after_raw["std"] == base_raw["std"])
            drills.append(d)

        REPORT["phases"]["4_rollback_drills"] = drills

        # ── Phase 5: disaster recovery ─────────────────────────────────────
        dr = {}
        # 5a model storage unavailable
        stash = sandbox / "stash"
        stash.mkdir()
        for name in ENGINE_FILES:
            shutil.move(str(active_dir / name), str(stash / name))
        t0 = time.perf_counter()
        try:
            load_engine(active_dir, verify=False)
            dr["storage_unavailable"] = {"detected": False,
                                         "detail": "engine loaded with missing files"}
        except Exception as e:  # noqa: BLE001
            dr["storage_unavailable"] = {
                "detected": True,
                "detail": (f"load raises {type(e).__name__} — no silent fallback "
                           "to stale/other artifacts; the app lifespan catches "
                           "this and degrades to rules-only with ML_UNAVAILABLE "
                           "tagging (src/risk_engine/main.py L157-170)"),
                "seconds_to_fail": round(time.perf_counter() - t0, 3)}
        # recovery from archive (verify against the DRILL record; the engine's
        # verify_integrity=True path would use the real 14-file record whose
        # non-engine extras are not in this sandbox — the real record rejecting
        # a partial set is itself the fail-safe behavior, noted in the report)
        t0 = time.perf_counter()
        for name in ENGINE_FILES:
            shutil.copy2(backup_dir / name, active_dir / name)
        rec_eng = load_engine(active_dir, verify=False)
        v_after = verify_artifacts(good, active_dir)
        dr["recovery_from_archive"] = {
            "seconds": round(time.perf_counter() - t0, 3),
            "verify_pass_on_reload": all(c["pass"] for c in v_after),
            "outputs_equal_baseline": (
                _score_stats(_run_canary(rec_eng, canary))["mean"]
                == base["mean"])}
        # 5b cold restart RTO (fresh process-equivalent load + record verify)
        t0 = time.perf_counter()
        eng_cold = load_engine(active_dir, verify=False)
        v_cold = verify_artifacts(good, active_dir)
        dr["cold_restart_engine_rto_seconds"] = round(
            time.perf_counter() - t0, 3)
        dr["cold_restart_verify_pass"] = all(c["pass"] for c in v_cold)
        dr["cold_restart_outputs_equal"] = (
            _score_stats(_run_canary(eng_cold, canary))["mean"] == base["mean"])
        # 5c temp-state loss: fresh entity tracker behaves as after restart
        from src.risk_engine.entity_fraud_rates import EntityFraudRateTracker
        ft = EntityFraudRateTracker()
        dr["temp_state_loss_tracker"] = {
            "fresh_tracker_user_rate": ft.get_rates("U1", "M1", "C1")["user_fraud_rate"],
            "fresh_tracker_merch_rate": ft.get_rates("U1", "M1", "C1")["merch_fraud_rate"],
            "detail": ("in-memory state lost on restart; rates return to "
                       "baseline 0.001 until re-seeded from DB-2 at startup "
                       "(risk_engine/main.py _seed_entity_tracker) — model "
                       "scores unchanged, fraud-rate features cold for ~first "
                       "events")}
        REPORT["phases"]["5_disaster_recovery"] = dr

        # ── Phase 6: verdict ───────────────────────────────────────────────
        d_ok = all(x["rollback"].get("ok") and x["rollback_correct"]
                   for x in drills)
        REPORT["phases"]["6_verdict"] = {
            "rollback_drills_all_pass": d_ok,
            "atomicity_partial_activation_detected": atomic["detected"],
            "n_drills": len(drills),
            "scenario_detected": {x["scenario"]: x["detected"] for x in drills},
            "overall": "FAIL" if record_gaps or not d_ok or not atomic["detected"] else "PASS",
        }
        out = ROOT / "reports" / "model_release_drill.json"
        out.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
        print(f"model_release_drill: wrote {out}")
        print(json.dumps(REPORT["phases"]["6_verdict"], indent=2))
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


if __name__ == "__main__":
    main()
