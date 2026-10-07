#!/usr/bin/env python3
"""Build the Phase 7 NATIVE-ENGINE CI TEST FIXTURE (tiny deterministic models).

*** THIS IS A TEST FIXTURE, NOT A PRODUCTION MODEL. ***

Purpose: let a clean checkout (CI) exercise the REAL native production code
path — `AltmanNativeEnsembleEngine`'s joblib loader, all three native members
(XGBoost / LightGBM / CatBoost), the RobustScaler preprocessing step, the
ensemble composition and the inference contract — without committing any
binary model artifacts. `models/production/` stays gitignored (Phase 6
limitation); this fixture lives outside it, is generated at test time into a
caller-chosen directory (tests use the OS temp dir), and is never used for
any accuracy, generalization or real-world-effectiveness claim.

Determinism controls (Phase 14):
  - single numpy RNG stream seeded with `--seed` (default 42): training data
    AND the label-noise stream come from it in fixed order;
  - explicit model seeds (`random_state` / `random_seed` = `--seed`);
  - single-threaded training everywhere (`n_jobs=1`, `num_threads=1`,
    `thread_count=1`) so no reduction order depends on core count;
  - fixed feature order: `ALTMAN_NATIVE_FEATURES` (the production 48-feature
    contract, imported from the engine module — not re-typed);
  - no timestamps, hostnames or absolute paths are written into any output
    file; manifests are serialized with sort_keys for byte-stable JSON;
  - joblib dumps use a fixed compression level.

Output layout (mirrors models/production/altman_native/):
    <out>/manifest.json                 selection manifest (model_type gate)
    <out>/altman_native/manifest.json   engine manifest (version/n/threshold)
    <out>/altman_native/xgb_native.joblib
    <out>/altman_native/lgb_native.joblib
    <out>/altman_native/cb_native.joblib
    <out>/altman_native/scaler_native.joblib

Usage (invoked by scripts/native_fixture_test.py, runnable standalone):
    python backend/scripts/native_fixture_build.py --out <dir> [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

import joblib  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.risk_engine.altman_native_ensemble import ALTMAN_NATIVE_FEATURES  # noqa: E402

# Contract constants — native_fixture_test.py asserts against these literals;
# a mismatch between builder and test is a test failure, by design.
FIXTURE_MODEL_VERSION = "native_fixture_ci_v1"
FIXTURE_MODEL_TYPE = "xgb_lgb_cb_native"  # exact string main.py selects on
N_FEATURES = 48
LOCKED_THRESHOLD = 0.5
N_ROWS = 256
PURPOSE = (
    "TEST FIXTURE - software-path verification only. NOT a production model, "
    "NOT production weights, NOT model-performance evidence."
)

# Tiny, shared hyperparameters: 8 trees/iterations each. Large enough that
# the members are real fitted tree ensembles with non-constant outputs,
# small enough to build in well under a second per member.
HParams = dict(n_estimators=8, max_depth=3, learning_rate=0.5)


def make_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed synthetic data: 256 rows x 48 features, learnable binary label.

    Entirely from the seeded RNG — no repository data, no network, no
    production training rows. Both classes are guaranteed present.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_ROWS, N_FEATURES)).astype(np.float32)
    noise = rng.standard_normal(N_ROWS)  # second draw from the same stream
    score = (1.1 * X[:, 0] - 0.9 * X[:, 3] + 0.7 * X[:, 22]
             - 0.6 * X[:, 31] + 0.5 * noise)
    y = (score > 0).astype(np.int64)
    assert 0 < int(y.sum()) < N_ROWS, "fixture labels must contain both classes"
    return X, y


def build_fixture(out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)  # CatBoost needs the parent of train_dir
    X, y = make_dataset(seed)
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)

    xgb = XGBClassifier(
        **HParams, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
        random_state=seed, n_jobs=1, tree_method="hist",
        eval_metric="logloss",
    )
    xgb.fit(Xs, y)

    lgb = LGBMClassifier(
        **HParams, min_child_samples=1, random_state=seed, num_threads=1,
        deterministic=True, verbose=-1,
    )
    lgb.fit(Xs, y)

    cb_dir = out / "cb_train"  # keep CatBoost's train_dir out of the cwd
    cb = CatBoostClassifier(
        iterations=8, depth=3, learning_rate=0.5, random_seed=seed,
        thread_count=1, verbose=False, train_dir=str(cb_dir),
    )
    cb.fit(Xs, y)
    # Drop CatBoost's training-log directory: the fixture output must be the
    # four joblibs + two manifests only.
    shutil.rmtree(cb_dir, ignore_errors=True)

    model_dir = out / "altman_native"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb, model_dir / "xgb_native.joblib", compress=3)
    joblib.dump(lgb, model_dir / "lgb_native.joblib", compress=3)
    joblib.dump(cb, model_dir / "cb_native.joblib", compress=3)
    joblib.dump(scaler, model_dir / "scaler_native.joblib", compress=3)

    # Engine manifest — exactly the keys AltmanNativeEnsembleEngine reads.
    (model_dir / "manifest.json").write_text(
        json.dumps({
            "model_version": FIXTURE_MODEL_VERSION,
            "model_type": FIXTURE_MODEL_TYPE,
            "n_features": N_FEATURES,
            "locked_threshold": LOCKED_THRESHOLD,
            "features": list(ALTMAN_NATIVE_FEATURES),
            "fixture": True,
            "seed": seed,
            "purpose": PURPOSE,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    # Selection manifest — main.py's lifespan picks the native engine iff
    # model_type == "xgb_lgb_cb_native" (read from PRODUCTION_DIR/manifest.json).
    (out / "manifest.json").write_text(
        json.dumps({
            "model_type": FIXTURE_MODEL_TYPE,
            "model_version": FIXTURE_MODEL_VERSION,
            "n_features": N_FEATURES,
            "fixture": True,
            "seed": seed,
            "purpose": PURPOSE,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory (fixture root; must NOT be under "
                         "models/production/)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out: Path = args.out
    if "production" in out.parts:
        print("REFUSED: fixture must not be written under a production path",
              file=sys.stderr)
        return 2
    if any(out.glob("*")):
        print(f"REFUSED: output directory not empty: {out}", file=sys.stderr)
        return 2

    build_fixture(out, args.seed)

    sizes = {p.name: p.stat().st_size
             for p in sorted((out / "altman_native").glob("*.joblib"))}
    print(f"native fixture built at {out}")
    print(f"  model_version={FIXTURE_MODEL_VERSION} seed={args.seed} "
          f"features={N_FEATURES} rows={N_ROWS}")
    print(f"  members: XGBClassifier({sizes['xgb_native.joblib']}B) "
          f"LGBMClassifier({sizes['lgb_native.joblib']}B) "
          f"CatBoostClassifier({sizes['cb_native.joblib']}B) "
          f"RobustScaler({sizes['scaler_native.joblib']}B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
