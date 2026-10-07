#!/usr/bin/env python3
"""PS-14 PHASE 7 - NATIVE PRODUCTION-ENGINE CI FIXTURE TEST.

Closes the Phase 6 limitation: `models/production/` is gitignored, so a clean
checkout previously exercised only the fallback engine and the native
production path never ran in CI.

Verification chain exercised here (all REAL code, no mocks, no stubs):

    fixture build (scripts/native_fixture_build.py, seeded, tiny)
        -> AltmanNativeEnsembleEngine loader (4x real joblib.load)
        -> all three native members predict (XGB + LGB + CB)
        -> ensemble composition (weighted average, exact)
        -> determinism (repeat + regeneration)
        -> Phase 5 batch contract (predict == predict_combined_many)
        -> main.py lifespan engine selection (PRODUCTION_DIR path-injected
           to the fixture root - no function is stubbed)
        -> /internal/evaluate end-to-end inference

EVIDENCE CLASSIFICATION: this is software-path verification (loader,
dependency set, engine execution, schema compatibility, deterministic
output, CI integration). The fixture is a tiny synthetic TEST FIXTURE - it
is NOT the production model and NOT evidence of fraud-detection accuracy,
generalization, or real-world effectiveness.

Run from the project root:
  python backend/scripts/native_fixture_test.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

# Hermetic env BEFORE any src.* import (mirrors risk_engine_test.py).
TMP = tempfile.mkdtemp(prefix="ps14-native-fixture-")
os.environ["DB_DIR"] = TMP
os.environ["PS14_MODE"] = "development"
os.environ["JWT_SECRET"] = "smoke-test-secret-0123456789abcdef"
os.environ["INTERNAL_TOKEN"] = "smoke-internal-token"

import numpy as np  # noqa: E402

from src.risk_engine.altman_native_ensemble import (  # noqa: E402
    ENSEMBLE_WEIGHTS, ALTMAN_NATIVE_FEATURES, AltmanNativeEnsembleEngine,
    map_raw_to_native,
)

BUILDER = ROOT / "backend" / "scripts" / "native_fixture_build.py"
SEED = 42
# Committed contract literals - a builder/test mismatch is a test failure.
EXPECTED_MODEL_VERSION = "native_fixture_ci_v1"
EXPECTED_ENGINE_MODEL_TYPE = "altman_native_xgb_lgb_cb"
FRAUD_ID = "F24BRMMMBJWYMTDW"  # same shape risk_engine_test uses
TOKEN = "smoke-internal-token"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(tag: str) -> Path:
    out = Path(TMP) / tag
    r = subprocess.run(
        [sys.executable, str(BUILDER), "--out", str(out), "--seed", str(SEED)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
    )
    check(f"fixture '{tag}' builder exit 0", r.returncode == 0,
          (r.stdout + r.stderr)[-300:] if r.returncode else "")
    return out


def native_raw(**over) -> dict:
    """A raw native-column row -> map_raw_to_native path 1 (the shared
    derive_native_features derivation, i.e. the production train==prod path).
    All keys the derivation reads are set explicitly so the same dict scores
    identically whether it comes from this test or from FeatureVector dumps.
    """
    d = {
        "amount": 120.0, "ts": "2019-03-05T14:30:00",
        "use_chip": "Chip Transaction", "mcc": 5411,
        "merchant_city": "New York", "merchant_state": "NY",
        "zip": "10001", "card": "C-1", "errors": "", "hour_of_day": 14,
        "user_id": "U-1", "merchant_id": "M-1", "city_id": "New York",
        "user_tx_count": 40, "user_avg_amt": 95.0, "card_tx_count": 30,
        "merch_tx_count": 12, "user_merchant_diversity": 9.0,
        "user_city_diversity": 3.0, "user_merch_count": 4,
        "user_fraud_rate": 0.004, "merch_fraud_rate": 0.012,
        "city_fraud_rate": 0.006,
    }
    d.update(over)
    return d


def endpoint_row(**over) -> dict:
    """A full /internal/evaluate payload features dict: the required base
    FeatureVector fields + the native raw columns (what the Privacy Layer
    sends the deployed native engine)."""
    d = {
        # required FeatureVector fields
        "amount_ratio": 1.2, "txn_freq_last_24h": 3, "txn_time_unusual": 0,
        "new_device_flag": 0, "unusual_location_flag": 0,
        "unusual_recipient_flag": 0, "failed_auth_count_24h": 0,
        "days_since_last_similar_txn": 2.5, "gradual_escalation_score": 0.1,
        "known_device_count": 2, "account_tenure_days": 300.0,
        "is_weekend": 0,
    }
    d.update(native_raw())  # hour_of_day comes from native_raw (required too)
    d.update(over)
    return d


def corpus() -> list[dict]:
    """Deterministic literal rows covering normal / night / high-amount /
    cold-start / empty-edge cases (no RNG - values are committed)."""
    return [
        native_raw(),
        native_raw(amount=15.5, ts="2021-06-01T03:12:00",
                   use_chip="Swipe Transaction", hour_of_day=3, mcc=5999,
                   merchant_city="Austin", merchant_state="TX", zip="73301",
                   user_fraud_rate=0.02, merch_fraud_rate=0.03,
                   city_fraud_rate=0.01),
        native_raw(amount=4200.0, ts="2020-11-27T02:10:00",
                   use_chip="Online Transaction", errors="Bad PIN",
                   hour_of_day=2, mcc=4899, merchant_city="Chicago",
                   merchant_state="IL", card="C-9",
                   user_tx_count=3, user_avg_amt=60.0),
        native_raw(user_id="U-new", merchant_id="M-new",
                   city_id="Nowhere", user_tx_count=0, user_avg_amt=0.0,
                   card_tx_count=0, merch_tx_count=0,
                   user_merchant_diversity=0.0, user_city_diversity=0.0,
                   user_merch_count=0, user_fraud_rate=0.001,
                   merch_fraud_rate=0.001, city_fraud_rate=0.001),
        native_raw(amount=0.0, ts="", use_chip="", mcc=0,
                   merchant_city="", merchant_state="", zip="", card="",
                   errors="", hour_of_day=0),
    ]


def member_checks(eng: AltmanNativeEnsembleEngine, rows: list[dict],
                  tag: str) -> np.ndarray:
    """Per-member (XGB/LGB/CB) load + predict + range/shape assertions (§7)."""
    vecs = np.array([map_raw_to_native(dict(r)) for r in rows]).astype(np.float32)
    X = eng.scaler.transform(vecs)
    for name, m in (("xgboost", eng.xgb), ("lightgbm", eng.lgb),
                    ("catboost", eng.cb)):
        p = m.predict_proba(X)
        check(f"{tag}: {name} predict_proba shape (n,2)",
              p.shape == (len(rows), 2), str(p.shape))
        check(f"{tag}: {name} outputs finite", bool(np.isfinite(p).all()))
        check(f"{tag}: {name} outputs in [0,1]",
              bool((p >= 0.0).all() and (p <= 1.0).all()))
        check(f"{tag}: {name} rows sum to 1",
              bool(np.allclose(p.sum(axis=1), 1.0, atol=1e-6)))
    return X


def main() -> int:
    print("== PS-14 Phase 7: native production-engine fixture test ==")
    prod_present = (ROOT / "models" / "production").is_dir()
    print(f"  fixture root: {TMP}")
    print(f"  models/production present in this checkout: {prod_present} "
          "(informational - this test never reads it)")

    # ---- §3: deterministic fixture generation (Option A) ------------------
    print("\n-- fixture build --")
    fix_a = build_fixture("fixture_a")
    fix_b = build_fixture("fixture_b")  # second build = regeneration check

    expected_files = [
        fix_a / "manifest.json",
        fix_a / "altman_native" / "manifest.json",
        fix_a / "altman_native" / "xgb_native.joblib",
        fix_a / "altman_native" / "lgb_native.joblib",
        fix_a / "altman_native" / "cb_native.joblib",
        fix_a / "altman_native" / "scaler_native.joblib",
    ]
    check("fixture layout complete (selection+engine manifests, 4 joblibs)",
          all(p.is_file() for p in expected_files),
          str([p.name for p in expected_files if not p.is_file()]))
    sel = json.loads((fix_a / "manifest.json").read_text(encoding="utf-8"))
    check("selection manifest declares native model_type",
          sel.get("model_type") == "xgb_lgb_cb_native",
          str(sel.get("model_type")))
    check("fixture artifacts are labeled TEST FIXTURE",
          sel.get("fixture") is True and "TEST FIXTURE" in sel.get("purpose", ""))

    # ---- §5: real loader on fixture A ------------------------------------
    print("\n-- native loader + engine identity --")
    eng = AltmanNativeEnsembleEngine(fix_a / "altman_native")
    check("loader returned AltmanNativeEnsembleEngine",
          type(eng).__name__ == "AltmanNativeEnsembleEngine",
          type(eng).__name__)
    check("engine model_type property",
          eng.model_type == EXPECTED_ENGINE_MODEL_TYPE, eng.model_type)
    check("engine model_version == fixture contract",
          eng.model_version == EXPECTED_MODEL_VERSION, eng.model_version)
    check("engine n_features == 48", eng.n_features == 48, str(eng.n_features))
    check("engine reads the FIXTURE dir (not models/production)",
          Path(eng._model_dir).resolve().is_relative_to(Path(TMP).resolve()),
          str(eng._model_dir))
    check("members are the real native estimators",
          type(eng.xgb).__name__ == "XGBClassifier"
          and type(eng.lgb).__name__ == "LGBMClassifier"
          and type(eng.cb).__name__ == "CatBoostClassifier",
          f"{type(eng.xgb).__name__}/{type(eng.lgb).__name__}/"
          f"{type(eng.cb).__name__}")
    check("scaler is a fitted RobustScaler(48)",
          type(eng.scaler).__name__ == "RobustScaler"
          and getattr(eng.scaler, "n_features_in_", None) == 48,
          f"{type(eng.scaler).__name__}({eng.scaler.n_features_in_})")

    rows = corpus()
    member_checks(eng, rows, "fixture_a")

    # ---- §7/§8: ensemble + deterministic expected results -----------------
    print("\n-- ensemble + determinism --")
    prob, unc = eng.predict(dict(rows[0]))
    check("ensemble prob finite in [0,1]",
          isinstance(prob, float) and math.isfinite(prob) and 0.0 <= prob <= 1.0,
          repr(prob))
    check("uncertainty marks native model_type",
          unc.get("model_type") == "altman_native", str(unc.get("model_type")))
    check("individual_outputs covers all three members",
          set(unc.get("individual_outputs", {}))
          == {"xgboost", "lightgbm", "catboost"},
          str(unc.get("individual_outputs")))
    check("uncertainty carries fixture model_version",
          unc.get("model_version") == EXPECTED_MODEL_VERSION,
          str(unc.get("model_version")))

    comp = eng.components(dict(rows[0]))["outputs"]
    expected = float(np.clip(
        ENSEMBLE_WEIGHTS["xgb"] * comp["xgboost"]
        + ENSEMBLE_WEIGHTS["lgb"] * comp["lightgbm"]
        + ENSEMBLE_WEIGHTS["cb"] * comp["catboost"], 0.0, 1.0))
    check("ensemble == exact weighted member combination", prob == expected,
          f"{prob!r} vs {expected!r}")

    prob2, unc2 = eng.predict(dict(rows[0]))
    check("repeated prediction bit-identical", prob == prob2 and unc == unc2,
          f"{prob!r} vs {prob2!r}")

    # ---- §9: Phase 5 batch contract ---------------------------------------
    print("\n-- batch consistency (Phase 5 contract) --")
    loop = [eng.predict_combined(dict(r)) for r in rows]
    many = eng.predict_combined_many([dict(r) for r in rows])
    check("predict_combined_many scores EXACTLY equal per-event loop",
          all(a[0] == b[0] and a[1] == b[1] for a, b in zip(loop, many)),
          f"max_d={max(abs(a[0] - b[0]) for a, b in zip(loop, many))}")
    check("predict_combined_many uncertainty dicts EXACTLY equal",
          all(a[2] == b[2] for a, b in zip(loop, many)))
    check("batched scores finite",
          bool(np.isfinite([b[0] for b in many]).all()))

    # ---- §8: regeneration reproduces the expected results -----------------
    print("\n-- regeneration determinism --")
    eng_b = AltmanNativeEnsembleEngine(fix_b / "altman_native")
    probs_a = [eng.predict(dict(r)) for r in rows]
    probs_b = [eng_b.predict(dict(r)) for r in rows]
    check("regenerated fixture reproduces predictions exactly",
          probs_a == probs_b)
    check("regenerated batch output exactly equal",
          eng.predict_combined_many([dict(r) for r in rows])
          == eng_b.predict_combined_many([dict(r) for r in rows]))
    byte_stable = ["xgb_native.joblib", "lgb_native.joblib",
                   "scaler_native.joblib"]
    for fn in byte_stable:
        check(f"{fn} byte-identical across builds",
              sha256(fix_a / "altman_native" / fn)
              == sha256(fix_b / "altman_native" / fn))
    for mf in ["manifest.json", "altman_native/manifest.json"]:
        check(f"{mf} byte-identical across builds",
              sha256(fix_a / mf) == sha256(fix_b / mf))
    # cb joblib carries one non-deterministic serialization byte; its member
    # predictions are the determinism contract.
    Xa = eng.scaler.transform(
        np.array([map_raw_to_native(dict(r)) for r in rows]).astype(np.float32))
    Xb = eng_b.scaler.transform(
        np.array([map_raw_to_native(dict(r)) for r in rows]).astype(np.float32))
    check("catboost member predictions identical after regeneration",
          np.array_equal(eng.cb.predict_proba(Xa), eng_b.cb.predict_proba(Xb)))

    # ---- §6: main.py selection + end-to-end endpoint ----------------------
    print("\n-- risk-engine selection + /internal/evaluate --")
    import src.risk_engine.main as risk_main  # noqa: PLC0415
    from fastapi.testclient import TestClient  # noqa: PLC0415

    orig_prod_dir = risk_main.PRODUCTION_DIR
    risk_main.PRODUCTION_DIR = fix_a  # path injection: selection code and
    # loader stay REAL; only the artifact root points at the fixture.
    try:
        with TestClient(risk_main.app) as c:
            fusion = risk_main.fusion
            check("service selected the NATIVE engine class",
                  type(fusion).__name__ == "AltmanNativeEnsembleEngine",
                  type(fusion).__name__)
            check("service engine model_type (fallback substitution detectable)",
                  getattr(fusion, "model_type", None)
                  == EXPECTED_ENGINE_MODEL_TYPE,
                  str(getattr(fusion, "model_type", None)))
            check("service engine model_version == fixture (production "
                  "artifacts not used)",
                  getattr(fusion, "model_version", None)
                  == EXPECTED_MODEL_VERSION,
                  str(getattr(fusion, "model_version", None)))
            check("service engine reads the FIXTURE dir",
                  Path(fusion._model_dir).resolve()
                  .is_relative_to(Path(TMP).resolve()),
                  str(fusion._model_dir))
            check("service engine exposes native-only predict_combined_many",
                  hasattr(fusion, "predict_combined_many"))
            check("runtime state READY after fixture load",
                  risk_main.runtime_state.name == "READY",
                  risk_main.runtime_state.name)

            feats = endpoint_row()
            r1 = c.post("/internal/evaluate",
                        headers={"X-Internal-Token": TOKEN},
                        json={"event_id": "ev-nfx-00001",
                              "fraud_id": FRAUD_ID, "features": feats})
            r2 = c.post("/internal/evaluate",
                        headers={"X-Internal-Token": TOKEN},
                        json={"event_id": "ev-nfx-00002",
                              "fraud_id": FRAUD_ID, "features": feats})
            check("endpoint evaluate 200 (both calls)",
                  r1.status_code == 200 and r2.status_code == 200,
                  f"{r1.status_code}/{r2.status_code} "
                  f"{r1.text[:150] if r1.status_code != 200 else ''}")
            if r1.status_code == 200 and r2.status_code == 200:
                m1 = r1.json()["ml_score"]
                m2 = r2.json()["ml_score"]
                check("endpoint ml_score deterministic across calls",
                      m1 == m2, f"{m1} vs {m2}")
                check("endpoint ml_score finite in [0,1]",
                      isinstance(m1, float) and math.isfinite(m1)
                      and 0.0 <= m1 <= 1.0, repr(m1))
                check("endpoint not degraded (native ML ran)",
                      r1.json().get("degraded") is False,
                      str(r1.json().get("degraded")))
                direct, _ = fusion.predict(dict(feats))
                check("endpoint ml_score == native engine prediction "
                      "(round 4dp)",
                      m1 == round(direct, 4), f"{m1} vs {round(direct, 4)}")
    finally:
        risk_main.PRODUCTION_DIR = orig_prod_dir

    print("\n" + ("ALL CHECKS PASSED" if not failures
                  else f"{len(failures)} CHECK(S) FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
