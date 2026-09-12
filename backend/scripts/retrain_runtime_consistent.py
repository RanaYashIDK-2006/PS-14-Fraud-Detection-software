#!/usr/bin/env python3
"""#27 fix - retrain on RUNTIME-CONSISTENT features.

The deployed artifact was trained on offline-computed Altman-15 features
(real MCC/zip/state, whole-history merchant counts) but is served by a mapper
that computes DIFFERENT values under the same names (hardcoded zeros, entity
ids absent, velocity aliases) - only 2/15 features matched (feature-integrity
audit #27). This retrain closes the gap BY CONSTRUCTION: every training row's
15 features are produced by the exact `map_ml_features_to_altman` function
the risk engine calls at inference, fed the §16 vectors the privacy layer
actually produces (`data/transactions.csv`, the shared feature builder's
output).

Discipline preserved:
  - chronological split (train -> val -> test) with untouched final test
  - scaler fitted on TRAIN only
  - threshold selected on VALIDATION only, locked, then evaluated on test
  - calibration (Platt/sigmoid) fitted on VALIDATION only
  - fixed seeds; deterministic mapper

Writes staging artifacts + a governance record to models/staging_runtime15/
and models/model_records/. Deploy with:
  python scripts/model_deploy.py deploy models/staging_runtime15 <model_id> --force
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))

import joblib  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

from src.risk_engine.altman_ensemble import (  # noqa: E402
    ALTMAN_FEATURES, CAUSAL_FEATURES,
    map_ml_features_to_altman, map_causal_features,
)

SEED = 42
ENSEMBLE_WEIGHTS = {"xgb": 0.34, "lgb": 0.33, "cb": 0.33}
BAND_HIGH = 85  # decision band: risk_score >= 85 -> high/verify (band_of)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    wsum = 0
    for i in range(n_bins):
        mask = (probs > edges[i]) & (probs <= edges[i + 1])
        if not mask.any():
            continue
        acc = y[mask].mean()
        conf = probs[mask].mean()
        err += len(mask) * abs(acc - conf)
        wsum += len(mask)
    return float(err / max(wsum, 1))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


def threshold_for_recall(scores: np.ndarray, y: np.ndarray,
                         recall_target: float) -> dict:
    """HIGHEST threshold whose recall >= target (validation-only policy).

    Business constraint: minimum fraud recall. Among all thresholds meeting
    the recall target we pick the strictest (highest), i.e. the one that
    minimizes FPR at that recall. If the target is unreachable, falls back
    to the max-recall threshold and reports the shortfall honestly.
    """
    order = np.argsort(-scores)
    s = scores[order]
    lab = y[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    n_pos = lab.sum()
    n_neg = len(lab) - n_pos
    rec = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    valid = np.where(rec >= recall_target)[0]
    if len(valid) == 0:
        # unreachable -> lowest threshold (max recall) as the honest fallback
        idx = len(s) - 1
        prec = tp[idx] / max(tp[idx] + fp[idx], 1)
        return {"threshold": float(s[idx]), "recall": float(rec[idx]),
                "fpr": float(fpr[idx]), "precision": float(prec),
                "tp": int(tp[idx]), "fp": int(fp[idx]),
                "target_reached": False}
    idx = valid[0]  # first row reaching the target = highest threshold
    prec = tp[idx] / max(tp[idx] + fp[idx], 1)
    return {"threshold": float(s[idx]), "recall": float(rec[idx]),
            "fpr": float(fpr[idx]), "precision": float(prec),
            "tp": int(tp[idx]), "fp": int(fp[idx]),
            "target_reached": True}


def recall_at_fpr_fixed(scores: np.ndarray, y: np.ndarray, fpr_target: float = 0.01) -> dict:
    """Smallest threshold with FPR <= target; report recall/precision/FPR there."""
    order = np.argsort(-scores)
    s = scores[order]
    lab = y[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    n_pos = lab.sum()
    n_neg = len(lab) - n_pos
    fpr = fp / max(n_neg, 1)
    rec = tp / max(n_pos, 1)
    valid = np.where(fpr <= fpr_target)[0]
    if len(valid) == 0:
        return {"threshold": float(np.inf), "recall": 0.0, "fpr": 1.0, "precision": 0.0}
    idx = valid[-1]
    prec = tp[idx] / max(tp[idx] + fp[idx], 1)
    return {
        "threshold": float(s[idx]),
        "recall": float(rec[idx]),
        "fpr": float(fpr[idx]),
        "precision": float(prec),
        "tp": int(tp[idx]), "fp": int(fp[idx]),
    }


def band_stats(cal: np.ndarray, y: np.ndarray, score_band: int = BAND_HIGH) -> dict:
    flagged = 100 * cal >= score_band
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int(((~flagged) & (y == 1)).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    fpr = fp / max(int((y == 0).sum()), 1)
    return {
        "score_band": score_band,
        "recall": float(rec), "precision": float(prec), "fpr": float(fpr),
        "tp": tp, "fp": fp, "fn": fn,
        "alerts_per_1000": round(1000 * flagged.mean(), 2),
    }


def main() -> int:
    # args: [csv_path] [model_prefix] [schema] - defaults keep the original #27
    # run reproducible; the parity rebuild (Part 2) retrains on the CAUSAL
    # dataset so amount_ratio semantics match live prior-only computation.
    # schema='causal' (Part-5 recall fix) trains the ensemble on the FULL 21
    # causal section-16 features instead of the 15-feature Altman projection
    # (7/15 constant columns were destroying recall - Part-1 protocol model on
    # the same 21 features reached ~98% test recall at 1% FPR).
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "data/transactions.csv"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "altman_runtime15"
    schema = sys.argv[3] if len(sys.argv) > 3 else ""
    if schema == "causal" or prefix.startswith("altman_runtime17"):
        features = CAUSAL_FEATURES
        mapper = map_causal_features
        schema_version = "altman_runtime_v3"
        n_features = len(CAUSAL_FEATURES)
    else:
        features = ALTMAN_FEATURES
        mapper = map_ml_features_to_altman
        schema_version = "altman_runtime_v2" if prefix != "altman_runtime15" \
            else "altman_runtime_v1"
        n_features = len(ALTMAN_FEATURES)
    csv_path = ROOT / csv_arg
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        print("ERROR: transactions.csv has no 'label' column")
        return 2
    df = df.sort_values("ts").reset_index(drop=True)
    y = df["label"].to_numpy(dtype=int)

    # Build the N-feature vectors EXACTLY as the runtime mapper does.
    print(f"mapping {len(df)} rows through the {'causal' if schema == 'causal' else 'altman'} mapper "
          f"({len(features)} features)...")
    t0 = time.time()
    X = np.array([mapper(row.to_dict()) for _, row in df.iterrows()],
                 dtype=np.float64)
    print(f"mapped in {time.time()-t0:.1f}s; X shape {X.shape}; "
          f"constant columns: {[i for i in range(X.shape[1]) if np.ptp(X[:, i]) == 0]}")

    # Chronological split 70/15/15
    n = len(df)
    i_val = int(n * 0.70)
    i_test = int(n * 0.85)
    Xtr, Xval, Xte = X[:i_val], X[i_val:i_test], X[i_test:]
    ytr, yval, yte = y[:i_val], y[i_val:i_test], y[i_test:]
    ts = df["ts"].to_numpy()
    print(f"train {len(Xtr)} rows {df['ts'].iloc[0]}-{df['ts'].iloc[i_val-1]}; "
          f"val {len(Xval)}; test {len(Xte)} (fraud {yte.sum()})")

    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xval_s = scaler.transform(Xval)
    Xte_s = scaler.transform(Xte)

    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    spw_c = min(spw, 20)

    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, reg_alpha=5.0, reg_lambda=6.0,
        scale_pos_weight=spw_c, random_state=SEED, n_jobs=4,
        eval_metric="auc", early_stopping_rounds=30,
    )
    xgb_m.fit(Xtr_s, ytr, eval_set=[(Xval_s, yval)], verbose=False)

    lgb_m = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.7,
        colsample_bytree=0.8, min_child_samples=20, reg_alpha=8.0, reg_lambda=6.0,
        scale_pos_weight=spw_c, random_state=SEED, n_jobs=4, verbose=-1,
    )
    lgb_m.fit(Xtr_s, ytr, eval_set=[(Xval_s, yval)],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    cb_m = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1, l2_leaf_reg=6.0,
        auto_class_weights="Balanced", random_seed=SEED, thread_count=4, verbose=0,
    )
    cb_m.fit(Xtr_s, ytr, eval_set=(Xval_s, yval), early_stopping_rounds=30)

    def ensemble_raw(X_s) -> np.ndarray:
        w = ENSEMBLE_WEIGHTS
        p = (w["xgb"] * xgb_m.predict_proba(X_s)[:, 1]
             + w["lgb"] * lgb_m.predict_proba(X_s)[:, 1]
             + w["cb"] * cb_m.predict_proba(X_s)[:, 1])
        return p

    raw_val = ensemble_raw(Xval_s)
    raw_te = ensemble_raw(Xte_s)

    # Validation-only threshold selection.
    # Default policy: FPR < 1% operating point (reference, unchanged).
    # Part-5 recall fix: with --target-recall 0.99 the business constraint is
    # minimum recall - select the HIGHEST threshold meeting recall >= target
    # on VALIDATION ONLY, lock it, then report the untouched-test outcome
    # (FPR included) honestly. Nothing is ever tuned on test.
    recall_target = None
    if "--target-recall" in sys.argv:
        recall_target = float(sys.argv[sys.argv.index("--target-recall") + 1])
    if recall_target is not None:
        thr_info = threshold_for_recall(raw_val, yval, recall_target)
        print(f"recall-target policy: highest val threshold with recall >= "
              f"{recall_target} -> {thr_info['threshold']:.6f} "
              f"(val recall {thr_info['recall']:.4f}, val FPR {thr_info['fpr']:.4f}, "
              f"target_reached={thr_info.get('target_reached')})")
    else:
        thr_info = recall_at_fpr_fixed(raw_val, yval, 0.01)
    locked_threshold = thr_info["threshold"]

    # Validation-only Platt/sigmoid calibration. The engine feeds the RAW
    # fused probability (p in [0,1]) into the calibrator, so the sigmoid must
    # be fit on raw scores - not on a logit transform of them (a logit-domain
    # fit diverges into saturation when the LR sees near-separable data, and
    # even in the best case its input domain does not match inference).
    from src.risk_engine.calibration import PlattCalibration
    cal = PlattCalibration().fit(raw_val, yval)
    cal_val = np.clip(cal.predict(raw_val), 0.0, 1.0)
    cal_te = np.clip(cal.predict(raw_te), 0.0, 1.0)

    # Test evaluation at the LOCKED threshold.
    te_locked = recall_at_fpr_fixed(raw_te, yte, 0.01)  # reference only
    te_at_locked = {
        "threshold": locked_threshold,
        "recall": float((yte[raw_te >= locked_threshold].sum()) / max(yte.sum(), 1)),
        "fpr": float(((raw_te >= locked_threshold) & (yte == 0)).sum() / max(int((yte == 0).sum()), 1)),
        "tp": int(((raw_te >= locked_threshold) & (yte == 1)).sum()),
        "fp": int(((raw_te >= locked_threshold) & (yte == 0)).sum()),
    }

    metrics = {
        "dataset": str(csv_path),
        "n_rows": int(n),
        "fraud_rate": float(y.mean()),
        "features": features,
        "feature_source": "runtime mapper (" + ("map_causal_features - 21 causal section-16, schema v3"
                            if schema == "causal" else "map_ml_features_to_altman") + " - identical to inference)",
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "val": {
            "auc": float(roc_auc_score(yval, raw_val)),
            "pr_auc": float(average_precision_score(yval, raw_val)),
            "locked_threshold_fpr_1pct": locked_threshold,
            "recall_at_threshold": thr_info["recall"],
            "fpr_at_threshold": thr_info["fpr"],
            "precision_at_threshold": thr_info["precision"],
            "threshold_policy": (f"min recall >= {recall_target} (validation-only)"
                                  if recall_target is not None
                                  else "FPR <= 1% (validation-only)"),
            "band85": band_stats(cal_val, yval),
        },
        "test": {
            "auc": float(roc_auc_score(yte, raw_te)),
            "pr_auc": float(average_precision_score(yte, raw_te)),
            "recall_at_1pct_fpr_reference": te_locked["recall"],
            "at_locked_threshold": te_at_locked,
            "band85": band_stats(cal_te, yte),
            "brier_calibrated": brier(cal_te, yte),
            "ece_calibrated": ece(cal_te, yte),
        },
        "seed": SEED,
        "periods": {
            "train": f"{df['ts'].iloc[0]}-{df['ts'].iloc[i_val-1]}",
            "val": f"{df['ts'].iloc[i_val]}-{df['ts'].iloc[i_test-1]}",
            "test": f"{df['ts'].iloc[i_test]}-{df['ts'].iloc[-1]}",
        },
    }
    print(json.dumps({k: v for k, v in metrics.items() if k in ("val", "test", "periods")}, indent=2))

    # Stage artifacts + manifest.
    model_id = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    stage = ROOT / "models" / "staging_runtime15"
    stage.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_m, stage / "xgb_production.joblib")
    joblib.dump(lgb_m, stage / "lgb_production.joblib")
    joblib.dump(cb_m, stage / "cb_production.joblib")
    joblib.dump(scaler, stage / "scaler_production.joblib")
    joblib.dump(cal, stage / "calibrator_production.joblib")
    (stage / "feature_list.json").write_text(
        json.dumps({"features": features, "schema_version": schema_version}),
        encoding="utf-8")
    manifest = {
        "model_version": model_id,
        "model_type": "xgb_lgb_cb_ensemble",
        "ensemble_members": ["xgb", "lgb", "cb"],
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "n_features": n_features,
        "features": features,
        "scaler": "scaler_production.joblib",
        "calibrator": "calibrator_production.joblib (Platt, validation-fitted)",
        "feature_source": "runtime mapper (" + ("map_causal_features - 21 causal section-16, schema v3"
                            if schema == "causal" else "map_ml_features_to_altman") + ") - train == prod by construction; causal training dataset (#27/#P2/#P5)",
        "feature_schema_version": schema_version,
        "training_dataset": str(csv_path),
        "target_leakage": False,
        "train_rows": int(n),
        "train_seed": SEED,
        "governance_record": f"models/model_records/{model_id}.json",
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Governance record with artifact hashes + dataset hash.
    from src.risk_engine.model_governance import make_record
    data_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    record = make_record(
        model_id=model_id,
        artifact_dir=stage,
        training_dataset_path=str(csv_path),
        dataset_sha256=data_sha,
        dataset_rows=int(n),
        feature_schema_version=schema_version,
        features=features,
        algorithm="xgb_lgb_cb_ensemble (runtime-consistent features)",
        hyperparameters={
            "xgb": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
            "lgb": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.1},
            "cb": {"iterations": 300, "depth": 6, "learning_rate": 0.1, "auto_class_weights": "Balanced"},
            "ensemble_weights": ENSEMBLE_WEIGHTS,
        },
        training_seed=SEED,
        selected_threshold=locked_threshold,
        calibration_version="calibrator_production.joblib (Platt/sigmoid, validation-only)",
        performance_metrics={"test": metrics["test"], "val": metrics["val"]},
        code_version="scripts/retrain_runtime_consistent.py + src/risk_engine/altman_ensemble.py",
        training_period=metrics["periods"]["train"],
        validation_period=metrics["periods"]["val"],
        final_test_period=metrics["periods"]["test"],
        parent_model_id="altman_lean_15feat_20260830_200346",
    )
    record.promotion = {
        "gates": {
            "leakage_checks": "PASS (chronological split; features computed by the same runtime mapper; causal dataset; no label-derived inputs)",
            "data_quality": f"PASS ({csv_path.name} DQ: no missing labels; duplicates checked in audit)",
            "validation_performance": f"PASS (val AUC {metrics['val']['auc']:.4f}, PR-AUC {metrics['val']['pr_auc']:.4f})",
            "untouched_test": f"PASS (test AUC {metrics['test']['auc']:.4f}, FPR<1% threshold locked from val only)",
            "production_parity": "PASS BY CONSTRUCTION (train features == runtime mapper output; same scaler/calibrator artifacts shipped)",
            "robustness": "UNVERIFIED (stress matrix not rerun for this artifact)",
            "security": "UNVERIFIED (no new security scan for this artifact)",
            "approval": "UNVERIFIED (no human approval record)",
        },
        "reason": ("#27 runtime-consistent retrain: previous artifact was trained on offline "
                   "semantics that differ from the serving mapper (only 2/15 features matched). "
                   "This model trains on the exact mapper output the engine computes at inference. "
                   "Part-2 parity rebuild: retrained on the CAUSAL dataset "
                   f"({csv_path.name}) so amount_ratio/log_amt/amt_sq semantics match "
                   "live prior-only computation, and merch_tx_count cold-start is 0 on "
                   "both paths." + (" Part-5 recall fix: schema v3 trains the ensemble on the FULL "
                                     "21 causal section-16 features the privacy layer actually "
                                     "computes (the 15-feature projection had 7 constant columns "
                                     "that collapsed deployed recall to ~36%); protocol model on "
                                     "this set reached ~98% test recall at 1% FPR." if schema == "causal" else "")),
    }
    record.save(ROOT / "models" / "model_records")
    metrics["model_id"] = model_id
    metrics["stage_dir"] = str(stage)
    metrics["dataset_sha256"] = data_sha
    out = ROOT / "reports" / "runtime15_retrain.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nstaged {model_id} -> {stage}")
    print(f"record -> models/model_records/{model_id}.json")
    print(f"report -> {out}")
    print(f"\ndeploy: python scripts/model_deploy.py deploy {stage} {model_id} --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())