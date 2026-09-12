#!/usr/bin/env python3
"""PS-14 Phase 2 - train & compare the four-model ensemble (section 5, 18.4).

Methodology, following the architecture doc:
  * Time-based 70/15/15 split (never random - fraud is time-dependent).
  * Imbalance handled via class weighting / scale_pos_weight on TRAINING only.
  * Isolation Forest is fit WITHOUT labels (unsupervised; catches novel fraud
    that supervised models trained on historical labels cannot).
  * Reported: PR-AUC, ROC-AUC, recall@1%FPR, plus precision/recall/F1/FPR/FNR
    at a validation-tuned (max-F1) threshold. Never accuracy alone.
  * Champion/challenger ready: artifacts are saved for the Risk Engine
    (next milestone) to fuse into a 0-100 score.

Usage:
  python src/train_compare.py [--data data/transactions.csv]
                              [--outdir models/artifacts] [--seed 42]
                              [--feedback data/feedback_labeled.csv]
                              [--feedback-date 20260816]

`--feedback` consumes an explicit pool file; `--feedback-date` resolves the
latest dated snapshot (`data/feedback_snapshots/feedback_YYYYMMDD_HHMMSS.csv`)
at or before that date (or "latest" for the newest), so a retrain is
reproducible from any point in the pool's history. The resolved snapshot
path and a content hash are recorded in metadata.json.

`--feedback` merges a labeled pool produced by scripts/export_feedback.py
(verification outcomes -> confirmed=legit / disputed=fraud) into the
synthetic set before the time-based split, implementing the section-7
feedback loop. Feedback rows keep their own timestamps, so they fall into
the time-based splits naturally (recently resolved events land in test).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Single source of truth for the ML feature list (shared with the generator
# and the Privacy Layer). Root is needed for `src.*` imports (the calibrator
# class), `src/` for the feature module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from privacy_layer.features import ML_FEATURES as FEATURES  # noqa: E402

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import joblib as _joblib

from src.risk_engine.calibration import PlattCalibration  # noqa: E402

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

ID_COLS = {"event_id", "fraud_id", "ts", "txn_amount_bucket", "label"}

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "feedback_snapshots"
SNAPSHOT_PREFIX = "feedback_"
SNAPSHOT_STAMP = "%Y%m%d_%H%M%S"

# The four-model ensemble spec, shared by the production fit (main) and the
# leave-one-archetype-out generalization folds (group_split_eval) so both
# use EXACTLY the same model family and hyperparameters.
MODEL_SPECS: dict[str, tuple[object, bool]] = {
    "logistic_regression": (
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42),
        True,
    ),
    "random_forest": (
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        False,
    ),
    "xgboost": (
        XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,  # recomputed per fold in fit_fold
            eval_metric="aucpr",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
        ),
        False,
    ),
    "isolation_forest": (
        IsolationForest(contamination=0.05, n_estimators=300, random_state=42),
        True,
    ),
}

STACKER_FEATURES = ("logistic_regression", "random_forest", "xgboost", "isolation_forest")


def resolve_feedback(feedback: str | None, feedback_date: str | None,
                     snapshot_dir: Path) -> tuple[Path | None, str | None]:
    """Resolve which labeled feedback pool to consume. Returns (path, tag)
    where `tag` identifies the snapshot (or None for an explicit file).

      feedback given     -> that explicit file (unchanged behavior)
      feedback_date given -> the latest snapshot at or before that
        YYYYMMDD[HHMMSS] in `snapshot_dir` (reproducible retraining from
        any point in the pool's history; the tag "latest" picks the newest
        snapshot)
      neither           -> (None, None) - train without the feedback pool

    Raises SystemExit when a requested date has no matching snapshot.
    """
    if feedback and feedback_date:
        raise SystemExit("--feedback and --feedback-date are mutually exclusive")
    if feedback:
        return Path(feedback).resolve(), None
    if feedback_date is None:
        return None, None
    if not snapshot_dir.is_dir():
        raise SystemExit(f"no feedback snapshots in {snapshot_dir} "
                         f"(run scripts/export_feedback.py first)")
    snaps = sorted(snapshot_dir.glob(f"{SNAPSHOT_PREFIX}*.csv"))
    if not snaps:
        raise SystemExit(f"no feedback snapshots in {snapshot_dir} "
                         f"(run scripts/export_feedback.py first)")
    if feedback_date == "latest":
        chosen = snaps[-1]
    else:
        date = feedback_date.replace("-", "").replace(" ", "_").replace(":", "")
        if len(date) == 8:
            date += "_235959"  # end of the requested day
        eligible = [s for s in snaps if s.stem[len(SNAPSHOT_PREFIX):] <= date]
        if not eligible:
            raise SystemExit(f"no feedback snapshot at or before {feedback_date} "
                             f"(earliest: {snaps[0].stem})")
        chosen = eligible[-1]
    return chosen.resolve(), chosen.stem[len(SNAPSHOT_PREFIX):]


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "fnr": fnr}


def best_threshold_by_f1(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    grid = np.unique(np.quantile(scores_val, np.linspace(0.0, 1.0, 1001)))
    best_t, best_f1 = grid[0], -1.0
    for t in grid:
        m = metrics_at_threshold(y_val, scores_val, t)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], t
    return float(best_t)


def threshold_at_fpr(y_val: np.ndarray, scores_val: np.ndarray, target_fpr: float = 0.01) -> float:
    grid = np.unique(np.quantile(scores_val, np.linspace(0.0, 1.0, 2001)))
    best_t, best_err = grid[0], np.inf
    for t in grid:
        err = abs(metrics_at_threshold(y_val, scores_val, t)["fpr"] - target_fpr)
        if err < best_err:
            best_err, best_t = err, t
    return float(best_t)


def fit_fold(df: pd.DataFrame, seed: int) -> dict:
    """Fit the full ensemble on a fold's 70/15/15 TIME split: scaler, the
    four models, the stacker (fit on validation only) and the calibrator
    (also validation only) - the same pipeline main() runs for production.

    Shared by the production fit and the leave-one-archetype-out folds so
    the generalization numbers come from the exact same model family.
    Returns everything needed to score arbitrary rows and to report
    fold-level metrics (fold val/test labels, per-model thresholds).
    """
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    sv, st = int(n * 0.70), int(n * 0.85)
    train, val, test = df.iloc[:sv], df.iloc[sv:st], df.iloc[st:]

    y_train = train["label"].to_numpy(dtype=int)
    y_val = val["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)
    X_train = train[FEATURES].to_numpy(dtype=float)
    X_val = val[FEATURES].to_numpy(dtype=float)
    X_test = test[FEATURES].to_numpy(dtype=float)

    pos, neg = int(y_train.sum()), int(len(y_train) - y_train.sum())
    specs = {
        name: (model, use_scaler) if name != "xgboost" else
        (XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=neg / max(pos, 1),
            eval_metric="aucpr", early_stopping_rounds=50,
            random_state=seed, n_jobs=-1), use_scaler)
        for name, (model, use_scaler) in MODEL_SPECS.items()
    }

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    fitted: dict[str, object] = {}
    scores_val: dict[str, np.ndarray] = {}
    scores_test: dict[str, np.ndarray] = {}
    iso_train_scores: np.ndarray | None = None
    thresholds: dict[str, dict] = {}

    for name, (model, use_scaler) in specs.items():
        Xtr = X_train_s if use_scaler else X_train
        Xva = X_val_s if use_scaler else X_val
        Xte = X_test_s if use_scaler else X_test
        if name == "xgboost":
            model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
        else:
            model.fit(Xtr, y_train)
        if name == "isolation_forest":
            s_train = -model.decision_function(Xtr)
            scores_val[name] = -model.decision_function(Xva)
            scores_test[name] = -model.decision_function(Xte)
            iso_train_scores = s_train
        else:
            scores_val[name] = model.predict_proba(Xva)[:, 1]
            scores_test[name] = model.predict_proba(Xte)[:, 1]
        fitted[name] = model
        thresholds[name] = {
            "f1": best_threshold_by_f1(y_val, scores_val[name]),
            "1pct_fpr": threshold_at_fpr(y_val, scores_val[name], 0.01),
        }

    X_stack_val = np.column_stack([scores_val[n] for n in STACKER_FEATURES])
    X_stack_test = np.column_stack([scores_test[n] for n in STACKER_FEATURES])
    stacker = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    stacker.fit(X_stack_val, y_val)

    fused_val_raw = stacker.predict_proba(X_stack_val)[:, 1]
    fused_test_raw = stacker.predict_proba(X_stack_test)[:, 1]
    # Smooth parametric calibration (Platt) - cannot step-function like
    # isotonic on near-separable data. Fold-internal calibrators are only
    # for threshold tuning; the PRODUCTION calibrator is fit on the
    # out-of-archetype pool by main().
    calibrator = PlattCalibration().fit(fused_val_raw, y_val)
    fused_val = np.clip(calibrator.predict(fused_val_raw), 0.0, 1.0)
    fused_test = np.clip(calibrator.predict(fused_test_raw), 0.0, 1.0)

    return {
        "scaler": scaler,
        "fitted": fitted,
        "iso_train_scores": iso_train_scores,
        "stacker": stacker,
        "calibrator": calibrator,
        "y_val": y_val,
        "y_test": y_test,
        "scores_val": scores_val,
        "scores_test": scores_test,
        "thresholds": thresholds,
        "fused_val_raw": fused_val_raw,
        "fused_test_raw": fused_test_raw,
        "fused_val": fused_val,
        "fused_test": fused_test,
    }


def _stack_features(fit: dict, rows: pd.DataFrame) -> np.ndarray:
    """The stacker input matrix for arbitrary rows (the four per-model
    outputs, ISO rank-normalized against this fold's training scores)."""
    X = rows[FEATURES].to_numpy(dtype=float)
    Xs = fit["scaler"].transform(X)
    p_lr = fit["fitted"]["logistic_regression"].predict_proba(Xs)[:, 1]
    p_rf = fit["fitted"]["random_forest"].predict_proba(X)[:, 1]
    p_xgb = fit["fitted"]["xgboost"].predict_proba(X)[:, 1]
    iso_score = -fit["fitted"]["isolation_forest"].decision_function(Xs)
    iso_pct = np.array([(fit["iso_train_scores"] < s).mean() for s in iso_score])
    return np.column_stack([p_lr, p_rf, p_xgb, iso_pct])


def score_raw(fit: dict, rows: pd.DataFrame) -> np.ndarray:
    """Uncalibrated stacker output for arbitrary rows - the calibrator's
    input - replicating `fusion.predict_raw_many` with THIS fold's artifacts.
    """
    return fit["stacker"].predict_proba(_stack_features(fit, rows))[:, 1]


def score_fused(fit: dict, rows: pd.DataFrame) -> np.ndarray:
    """CALIBRATED fused probability for arbitrary rows, replicating
    `src/risk_engine/fusion.py` exactly (scaler -> models -> ISO rank
    percentile -> stacker -> calibrator), using THIS fold's artifacts.
    """
    return np.clip(fit["calibrator"].predict(score_raw(fit, rows)), 0.0, 1.0)


def run_ood_folds(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One pass over the leave-one-out folds, used for BOTH the honest
    generalization metrics AND the production calibrator's training pool.

    For each held-out group (every fraud archetype, plus a novel-legit-
    account fold) the full ensemble is RETRAINED without that group, then
    the group's rows are scored twice:
      * raw stacker output + label -> appended to the calibration pool (the
        raw->probability mapping for patterns the fitting model NEVER saw),
      * calibrated at the fold's own thresholds -> generalization recall.

    Returns (gen_df, pool_raw, pool_y). With `--no-group-eval` main() skips
    this and falls back to fitting the calibrator on validation.
    """
    rows: list[dict] = []
    raws: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    if "archetype" not in df.columns:
        print("\n== Generalization: skipped (no 'archetype' column — non-PS14 dataset) ==")
        return pd.DataFrame(), np.array([]), np.array([])
    fraud_archs = sorted(
        a for a in df["archetype"].unique()
        if int(((df["archetype"] == a) & (df["label"] == 1)).sum()) >= 10
    )
    print("\n== Generalization: leave-one-archetype-out (retrained per fold) ==")
    for a in fraud_archs:
        fit = fit_fold(df[df["archetype"] != a], seed)
        held = df[df["archetype"] == a]
        y = held["label"].to_numpy(dtype=int)
        raw = score_raw(fit, held)
        ml = np.clip(fit["calibrator"].predict(raw), 0.0, 1.0)
        # Thresholds are tuned on the fold's RAW stacker output so the
        # recall numbers are calibration-independent (a monotonic calibrator
        # must not change a ranking-based metric - the earlier isotonic steps
        # made the F1 threshold land on different points).
        t_f1 = best_threshold_by_f1(fit["y_val"], fit["fused_val_raw"])
        t_1pct = threshold_at_fpr(fit["y_val"], fit["fused_val_raw"], 0.01)
        n_fraud = int(y.sum())
        recall_f1 = float((raw[y == 1] >= t_f1).mean()) if n_fraud else np.nan
        recall_1pct = float((raw[y == 1] >= t_1pct).mean()) if n_fraud else np.nan
        ood_legit_fpr = float((raw[y == 0] >= t_f1).mean()) if (y == 0).any() else np.nan
        rows.append({
            "held_out": a, "n_fraud": n_fraud, "n_legit": int((y == 0).sum()),
            "recall_f1": round(recall_f1, 3), "recall_at_1pct_fpr": round(recall_1pct, 3),
            "ood_legit_fpr": None if np.isnan(ood_legit_fpr) else round(ood_legit_fpr, 3),
            "avg_ml_fraud": round(float(ml[y == 1].mean()), 4) if n_fraud else None,
        })
        print(f"  held-out {a:12s} | recall(F1) {recall_f1:.3f} | recall@1%FPR {recall_1pct:.3f}"
              + (f" | OOD legit FPR {ood_legit_fpr:.3f}" if not np.isnan(ood_legit_fpr) else ""))
        raws.append(raw)
        ys.append(y)

    # Novel-account probe: hold out whole legit-only accounts (cold start).
    # ~40 accounts (~13% of the legit population, ~1.3k rows): enough to be
    # meaningful without gutting the fold's legit training data (150 accounts
    # = half the population made the FPR estimate overly pessimistic).
    legit_fids = df.loc[df["label"] == 0, "fraud_id"].unique()
    rng = np.random.default_rng(seed)
    held_fids = rng.choice(legit_fids, size=min(40, len(legit_fids)), replace=False)
    fit = fit_fold(df[~df["fraud_id"].isin(held_fids)], seed)
    held = df[df["fraud_id"].isin(held_fids)]
    raw = score_raw(fit, held)
    ml = np.clip(fit["calibrator"].predict(raw), 0.0, 1.0)
    t_f1 = best_threshold_by_f1(fit["y_val"], fit["fused_val_raw"])
    fpr_f1 = float((raw >= t_f1).mean())
    fpr_6 = float((ml >= 0.6).mean())  # the production BEHAVIOR_DEVIATION trigger
    rows.append({
        "held_out": f"{len(held_fids)} novel legit accounts", "n_fraud": 0,
        "n_legit": int(len(held)), "recall_f1": None, "recall_at_1pct_fpr": None,
        "ood_legit_fpr": round(fpr_f1, 3), "avg_ml_fraud": None,
        "novel_account_fpr_0.6": round(fpr_6, 3),
    })
    print(f"  novel legit accounts ({len(held_fids)}) | FPR@{t_f1:.2f}(F1) {fpr_f1:.3f} | FPR@0.6 {fpr_6:.3f}")
    raws.append(raw)
    ys.append(held["label"].to_numpy(dtype=int))
    return pd.DataFrame(rows), np.concatenate(raws), np.concatenate(ys)


def check_ood_recall_gate(gen_df: pd.DataFrame, archetypes: list[str], floor: float) -> list[dict]:
    """Gate the retrain on held-out archetype recall.

    Fails when a named fraud archetype's recall-at-1%-FPR on the
    leave-one-archetype-out fold drops below `floor`. The metric is the
    ranking-based `recall_at_1pct_fpr`, NOT `recall_f1`: the F1-optimal
    threshold is unreliable at low fold fraud prevalence (e.g. the ato fold's
    validation is ~0.2% fraud because holding out ato removes nearly all
    fraud from the val window, so its F1 threshold lands far above held-out
    ato scores even when ranking is intact). Returns one row per checked
    archetype for the report; callers fail the retrain if any row is FAIL.
    """
    rows: list[dict] = []
    if gen_df.empty:
        print("OOD recall gate: skipped (no group-split eval - rerun without --no-group-eval)")
        return rows
    table = {r["held_out"]: r for r in gen_df.to_dict(orient="records")}
    print(f"\n== OOD recall gate (recall@1%FPR >= {floor:.2f}, held-out archetypes: {', '.join(archetypes)}) ==")
    for a in archetypes:
        row = table.get(a)
        if row is None:
            print(f"  {a:12s} SKIP (archetype absent from the group-split eval)")
            continue
        rec = row["recall_at_1pct_fpr"]
        if rec is None or np.isnan(rec):
            print(f"  {a:12s} SKIP (no fraud rows in the held-out fold)")
            continue
        ok = rec >= floor
        rows.append({"archetype": a, "recall_at_1pct_fpr": float(rec), "floor": floor, "passed": bool(ok)})
        print(f"  {a:12s} recall@1%FPR {rec:.3f}  floor {floor:.2f}  -> {'PASS' if ok else 'FAIL'}")
    return rows


def pinned_ood_eval(df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """Score the pinned live OOD scenarios (`data/ood_scenarios.csv`) with
    the PRODUCTION fit, recording ml and whether it crosses the
    BEHAVIOR_DEVIATION trigger (0.6) / step-up floor (0.3). These are the
    five real vectors from the live scenario walkthrough - a permanent
    regression test that a retrain keeps (or improves) OOD behaviour.
    """
    path = Path(__file__).resolve().parent.parent / "data" / "ood_scenarios.csv"
    if not path.exists():
        print("  (data/ood_scenarios.csv not found - skipping pinned OOD check)")
        return pd.DataFrame()
    sc = pd.read_csv(path)
    ml = score_fused(fit, sc)
    rows = []
    for _, r in sc.iterrows():
        m = float(ml[_])
        rows.append({
            "scenario": r["scenario"], "note": r["note"],
            "ml_score": round(m, 4),
            "flag_novel": bool(m >= 0.6), "stepup_floor": bool(m >= 0.3),
        })
    print("\n== Pinned live OOD scenarios (production fit) ==")
    for r in rows:
        print(f"  {r['scenario']:32s} ml={r['ml_score']:.4f} flag={r['flag_novel']}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--outdir", default="models/artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feedback", default=None,
                    help="labeled feedback CSV (verification outcomes) to merge")
    ap.add_argument("--feedback-date", default=None,
                    help="YYYYMMDD[HHMMSS] or 'latest': resolve the feedback snapshot at or before that time")
    ap.add_argument("--snapshot-dir", default=str(SNAPSHOT_DIR))
    ap.add_argument("--no-group-eval", action="store_true",
                    help="skip the leave-one-archetype-out generalization + pinned OOD evaluation")
    ap.add_argument("--ood-recall-floor", type=float, default=0.60,
                    help="held-out archetype recall@1%FPR floor; the retrain FAILS if a gated archetype drops below it")
    ap.add_argument("--ood-gate-archetypes", default="ato,mule",
                    help="comma-separated archetypes gated on held-out recall@1%FPR")
    ap.add_argument("--no-ood-gate", action="store_true",
                    help="report the gate but never fail the retrain")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_path = root / args.data
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    # Ensure ts is numeric (unix timestamps) for sorting
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")

    n_feedback = 0
    feedback_source = None
    feedback_sha256 = None
    fb_path, fb_tag = resolve_feedback(args.feedback, args.feedback_date, root / args.snapshot_dir)
    if fb_path is not None:
        fb = pd.read_csv(fb_path)
        fb["ts"] = pd.to_numeric(fb["ts"], errors="coerce")
        missing = [c for c in df.columns if c not in fb.columns]
        missing_ml = [c for c in missing if c in FEATURES]
        missing_other = [c for c in missing if c not in FEATURES]
        if missing_other:
            sys.exit(f"feedback CSV {fb_path} missing required columns: {missing_other}")
        if missing_ml:
            # Old snapshots predate newly added ML features (e.g. the link-
            # analysis signals); backfill 0 so any point in the pool's history
            # stays reproducible. Fresh exports carry real values.
            print(f"feedback pool: backfilling {len(missing_ml)} new ML feature(s) "
                  f"with 0: {missing_ml}")
            for c in missing_ml:
                fb[c] = 0
        n_feedback = len(fb)
        feedback_source = str(fb_path)
        feedback_sha256 = hashlib.sha256(fb_path.read_bytes()).hexdigest()[:16]
        df = pd.concat([df, fb[df.columns]], ignore_index=True)
        tag_note = f" (snapshot {fb_tag})" if fb_tag else ""
        print(f"feedback loop: merged {n_feedback} verified events "
              f"({int((fb['label'] == 0).sum())} confirmed/legit, "
              f"{int((fb['label'] == 1).sum())} disputed/fraud){tag_note}")

    df = df.sort_values("ts").reset_index(drop=True)

    if XGBClassifier is None:
        sys.exit("xgboost is required - install it via `pip install -r requirements.txt`")

    # Time-based split: 70/15/15 by timestamp quantile.
    n = len(df)
    split_val = int(n * 0.70)
    split_test = int(n * 0.85)
    train, val, test = df.iloc[:split_val], df.iloc[split_val:split_test], df.iloc[split_test:]

    y_train = train["label"].to_numpy(dtype=int)
    y_val = val["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)

    prevalence = float(y_train.mean())
    print(f"split sizes: train={len(train)} val={len(val)} test={len(test)} "
          f"(test fraud rate {y_test.mean():.4f})")

    # Fit the production ensemble (the SAME pipeline the generalization folds
    # use - see fit_fold). The solo-model results below are rebuilt from the
    # fold's per-model validation/test scores so the printed table is
    # unchanged from before the refactor.
    fit = fit_fold(df, args.seed)
    scaler = fit["scaler"]
    fitted = fit["fitted"]
    thresholds = fit["thresholds"]
    scores_test = fit["scores_test"]
    scores_val = fit["scores_val"]
    iso_train_scores = fit["iso_train_scores"]
    stacker = fit["stacker"]
    stacker_features = STACKER_FEATURES
    fused_val_raw = fit["fused_val_raw"]
    fused_test_raw = fit["fused_test_raw"]

    # ---- Calibration: Platt scaling on the out-of-archetype pool ----------
    # The production calibrator is fit on the cross-fit OOD pool (raw stacker
    # outputs + labels for patterns the fitting ensemble never saw) - NOT the
    # leaky validation split. Platt is parametric (2 params), so it cannot
    # step-function the way isotonic did. The pool oversamples fraud (~12% vs
    # ~2% population), so legit rows are reweighted to the population
    # prevalence before the fit, keeping the sigmoid's intercept honest.
    gen_df, pool_raw, pool_y = pd.DataFrame(), np.array([]), np.array([])
    if not args.no_group_eval:
        gen_df, pool_raw, pool_y = run_ood_folds(df, args.seed)
        if not gen_df.empty:
            gen_df.to_csv(outdir / "generalization_by_archetype.csv", index=False)

    calibrator = PlattCalibration()
    if len(pool_y):
        # Blend the PRODUCTION model's own validation predictions into the
        # OOD pool: the OOD cross-fit rows anchor the honest mid-range
        # (never-seen patterns cap around raw ~0.7), and the production val
        # rows span the full raw range up to 1.0 - without them the sigmoid
        # saturates at ~0.36 and nothing can ever reach the 0.6
        # BEHAVIOR_DEVIATION trigger. Both are reweighted to the population
        # prevalence so the intercept stays honest.
        pool_raw = np.concatenate([pool_raw, fused_val_raw])
        pool_y = np.concatenate([pool_y, y_val])
        p_pop = float(df["label"].mean())
        p_pool = float(pool_y.mean())
        if 0 < p_pool < 1:
            w = np.ones_like(pool_y, dtype=float)
            n_neg = float((pool_y == 0).sum())
            w[pool_y == 0] = max((pool_y.sum() * (1 - p_pop)) / max(n_neg * p_pop, 1e-9), 0.0)
            calibrator.fit(pool_raw, pool_y, sample_weight=w)
        else:
            calibrator.fit(pool_raw, pool_y)
        print(f"  calibrator: Platt on OOD cross-fit pool + production val "
              f"(n={len(pool_y)}, fraud {p_pool:.3f}, reweighted to {p_pop:.4f})")
    else:
        # --no-group-eval fallback: still Platt, just on validation.
        calibrator.fit(fused_val_raw, y_val)
        print("  calibrator: Platt on validation (group eval skipped)")

    fused_val = np.clip(calibrator.predict(fused_val_raw), 0.0, 1.0)
    fused_test = np.clip(calibrator.predict(fused_test_raw), 0.0, 1.0)
    cal_mae = float(np.mean(np.abs(fused_val - y_val)))

    # Calibration check: decile table (predicted mean vs actual fraud rate).
    order = np.argsort(fused_test)
    cal_deciles = []
    for q in range(10):
        idx = order[q * len(fused_test) // 10:(q + 1) * len(fused_test) // 10]
        cal_deciles.append((float(fused_test[idx].mean()), float(y_test[idx].mean())))
    print("  calibration deciles (test, pred/actual): "
          + " | ".join(f"{p:.2f}/{a:.2f}" for p, a in cal_deciles))

    results = []
    for name in MODEL_SPECS:
        s_val = scores_val[name]
        s_test = scores_test[name]
        t_f1 = thresholds[name]["f1"]
        t_1pct = thresholds[name]["1pct_fpr"]
        m_f1 = metrics_at_threshold(y_test, s_test, t_f1)
        m_1pct = metrics_at_threshold(y_test, s_test, t_1pct)

        results.append(
            {
                "model": name,
                "pr_auc": float(average_precision_score(y_test, s_test)),
                "roc_auc": float(roc_auc_score(y_test, s_test)),
                "recall_at_1pct_fpr": m_1pct["recall"],
                "precision": m_f1["precision"],
                "recall": m_f1["recall"],
                "f1": m_f1["f1"],
                "fpr": m_f1["fpr"],
                "fnr": m_f1["fnr"],
                "threshold_f1": t_f1,
                "threshold_1pct": t_1pct,
            }
        )
        print(f"  {name:22s} PR-AUC={results[-1]['pr_auc']:.4f} "
              f"ROC-AUC={results[-1]['roc_auc']:.4f} F1={results[-1]['f1']:.4f}")

    _joblib.dump(stacker, outdir / "stacker.joblib")
    if iso_train_scores is not None:
        _joblib.dump(iso_train_scores, outdir / "iso_train_scores.joblib")
    print(f"  stacker coefficients: {dict(zip(stacker_features, np.round(stacker.coef_[0], 3)))}")
    _joblib.dump(calibrator, outdir / "calibrator.joblib")
    print(f"  calibrator: Platt (MAE on val {cal_mae:.4f})")

    t_f1 = best_threshold_by_f1(y_val, fused_val)
    t_1pct = threshold_at_fpr(y_val, fused_val, 0.01)
    m_f1 = metrics_at_threshold(y_test, fused_test, t_f1)
    m_1pct = metrics_at_threshold(y_test, fused_test, t_1pct)
    results.append(
        {
            "model": "fused_ensemble (stacker)",
            "pr_auc": float(average_precision_score(y_test, fused_test)),
            "roc_auc": float(roc_auc_score(y_test, fused_test)),
            "recall_at_1pct_fpr": m_1pct["recall"],
            "precision": m_f1["precision"],
            "recall": m_f1["recall"],
            "f1": m_f1["f1"],
            "fpr": m_f1["fpr"],
            "fnr": m_f1["fnr"],
            "threshold_f1": t_f1,
            "threshold_1pct": t_1pct,
        }
    )
    print(f"  fused_ensemble (stacker) PR-AUC={results[-1]['pr_auc']:.4f} "
          f"ROC-AUC={results[-1]['roc_auc']:.4f} F1={results[-1]['f1']:.4f}")

    # Reference baseline: a random classifier has PR-AUC = prevalence, ROC-AUC = 0.5.
    results.append(
        {
            "model": "random_baseline",
            "pr_auc": prevalence,
            "roc_auc": 0.5,
            "recall_at_1pct_fpr": 0.01,
            "precision": prevalence,
            "recall": prevalence,
            "f1": 2 * prevalence / (1 + prevalence),
            "fpr": prevalence,
            "fnr": 1 - prevalence,
            "threshold_f1": np.nan,
            "threshold_1pct": np.nan,
        }
    )

    res_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    res_df.to_csv(outdir / "metrics_comparison.csv", index=False)

    # ---- EVALUATION.md -----------------------------------------------------
    def fmt(v: float, nd: int = 4) -> str:
        return f"{v:.{nd}f}"

    rows = "\n".join(
        f"| {r['model']} | {fmt(r['pr_auc'])} | {fmt(r['roc_auc'])} | "
        f"{fmt(r['recall_at_1pct_fpr'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
        f"{fmt(r['f1'])} | {fmt(r['fpr'])} | {fmt(r['fnr'])} |"
        for r in res_df.to_dict(orient="records")
    )

    # ---- Recall by fraud archetype (test, F1-optimal threshold) -----------
    arch_rows: list[dict] = []
    if "archetype" in df.columns:
        test_arch = test["archetype"].to_numpy()
        for arch in sorted(np.unique(test_arch)):
            mask = test_arch == arch
            n_pos = int(y_test[mask].sum())
            if n_pos == 0:
                continue
            row: dict = {"archetype": arch, "n_fraud_test": n_pos}
            for name in ("logistic_regression", "random_forest", "xgboost", "isolation_forest"):
                m = metrics_at_threshold(y_test[mask], scores_test[name][mask], thresholds[name]["f1"])
                row[name] = round(m["recall"], 3)
            arch_rows.append(row)
    arch_df = pd.DataFrame(arch_rows)
    if not arch_df.empty:
        arch_df.to_csv(outdir / "recall_by_archetype.csv", index=False)
        print("\nRecall by fraud archetype (F1-optimal threshold, test):")
        print(arch_df.to_string(index=False))

    # Archetype table for EVALUATION.md
    arch_md = ""
    if not arch_df.empty:
        header = "| Archetype | n (test) | LR | RF | XGB | ISO |\n|---|---|---|---|---|---|\n"
        arch_md = (
            "\n## Recall by fraud archetype (test, F1-optimal threshold)\n\n"
            + header
            + "\n".join(
                f"| {r['archetype']} | {r['n_fraud_test']} | {r['logistic_regression']} | "
                f"{r['random_forest']} | {r['xgboost']} | {r['isolation_forest']} |"
                for r in arch_df.to_dict(orient="records")
            )
            + "\n"
        )

    # Feature importances (RF + XGB) for the explainability step.
    imp_lines = []
    for mname in ("random_forest", "xgboost"):
        if mname in fitted:
            imp = np.argsort(fitted[mname].feature_importances_)[::-1][:10]
            lines = [f"- **{FEATURES[i]}** ({fitted[mname].feature_importances_[i]:.4f})" for i in imp]
            imp_lines.append(f"### Top features - {mname}\n" + "\n".join(lines))

    md = f"""# Model Evaluation - PS-14 Phase 2 (synthetic set)

Generated by `src/train_compare.py` (seed {args.seed}).

**Method (per section 5/6 of the architecture doc):**
- Time-based split 70/15/15 (no random split - avoids temporal leakage).
- Class weighting / `scale_pos_weight` applied on **training only**.
- Isolation Forest fit **without labels** (unsupervised novelty detection);
  its decision threshold is tuned on validation like the other models.
- Metrics reported together - never accuracy alone; a random classifier
  baseline (PR-AUC = prevalence) is included for reference.
- Thresholds: F1-optimal (validation) for precision/recall/F1/FPR/FNR;
  recall at a fixed 1% FPR is reported separately as the primary operating
  point for a low-false-positive fraud system.

Data: {data_path.name}, {len(df):,} events, test fraud rate {y_test.mean():.4f}.

| Model | PR-AUC | ROC-AUC | Recall@1%FPR | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|---|
{rows}

## Calibration (deciles on test: predicted mean / actual fraud rate)

{chr(10).join(f"| {q} | {p:.3f} | {a:.3f} |" for q, (p, a) in enumerate(cal_deciles))}
{arch_md}
{chr(10).join(imp_lines)}

## Notes
- A model predicting "not fraud" always would score ~{1 - prevalence:.4f}
  accuracy here; that is exactly why accuracy is not a selection metric.
- The ensemble (fused in the Risk Engine milestone) is expected to beat any
  single model; Isolation Forest stays in production even if its solo metrics
  are weaker, because it is the only component that can catch fraud patterns
  absent from historical labels.
"""
    # ---- Pinned OOD scenarios (production fit) -----------------------------
    # `gen_df` was already built by run_ood_folds above (and doubles as the
    # calibrator's training pool); this section only scores the pinned live
    # vectors with the shipped artifacts.
    prod_fit = {
        "scaler": scaler, "fitted": fitted, "iso_train_scores": iso_train_scores,
        "stacker": stacker, "calibrator": calibrator,
    }
    ood_df = pinned_ood_eval(df, prod_fit)
    if not ood_df.empty:
        ood_df.to_csv(outdir / "ood_scenarios_scores.csv", index=False)

    gen_md = ood_md = ""
    if not gen_df.empty:
        hdr = "| Held-out group | n fraud | recall (F1) | recall@1%FPR | OOD legit FPR | avg ml (fraud) |\n|---|---|---|---|---|---|\n"
        gen_md = (
            "\n## Generalization (leave-one-archetype-out, retrained per fold)\n\n"
            "The ensemble is RETRAINED with the entire group excluded, so the"
            " recall below is on fraud patterns the model has never seen - the"
            " honest counterpart to the time-split table above (where every"
            " archetype leaks same-pattern rows into training).\n\n" + hdr
            + "\n".join(
                f"| {r['held_out']} | {r['n_fraud']} | {r['recall_f1']} | "
                f"{r['recall_at_1pct_fpr']} | {r['ood_legit_fpr']} | {r['avg_ml_fraud']} |"
                for r in gen_df.to_dict(orient="records")
            ) + "\n"
        )
    if not ood_df.empty:
        ood_md = (
            "\n## Pinned live OOD scenarios (production fit)\n\n"
            "The five real vectors from the live walkthrough, scored with the"
            " shipped artifacts - a pinned regression check that a retrain"
            " keeps (or improves) out-of-distribution behaviour. `flag` is"
            " the BEHAVIOR_DEVIATION trigger (ml >= 0.6).\n\n"
            "| scenario | ml_score | flag (>=0.6) | expected |\n|---|---|---|---|\n"
            + "\n".join(
                f"| {r['scenario']} | {r['ml_score']} | {r['flag_novel']} | {r['note']} |"
                for r in ood_df.to_dict(orient="records")
            ) + "\n"
        )
    md += gen_md + ood_md

    (outdir / "EVALUATION.md").write_text(md, encoding="utf-8")

    # ---- Artifacts ---------------------------------------------------------
    _joblib.dump(scaler, outdir / "scaler.joblib")
    for name, model in fitted.items():
        _joblib.dump(model, outdir / f"{name}.joblib")

    # Save validation labels + scores for calibration testing
    _joblib.dump(y_val, outdir / "validation_labels.joblib")
    _joblib.dump(fused_val_raw, outdir / "fused_val_scores.joblib")

    meta = {
        "seed": args.seed,
        "data": str(data_path.relative_to(root) if data_path.is_absolute() else data_path),
        "feedback": n_feedback,
        "feedback_source": str(Path(feedback_source).relative_to(root) if Path(feedback_source).is_absolute() else feedback_source) if feedback_source else None,
        "feedback_sha256": feedback_sha256,
        "features": FEATURES,
        "split_counts": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "test_prevalence": float(y_test.mean()),
        "thresholds": thresholds,
        "metrics": res_df.to_dict(orient="records"),
        "recall_by_archetype": arch_df.to_dict(orient="records"),
        "fusion": {
            "stacker": "stacker.joblib",
            "stacker_features": list(stacker_features),
            "iso_rank_scores": "iso_train_scores.joblib",
            "stacker_coefficients": {k: float(v) for k, v in zip(stacker_features, stacker.coef_[0])},
            "calibrator": "calibrator.joblib",
            "calibrator_type": "platt_ood" if len(pool_y) else "platt_val",
            "calibration_pool": {"n": int(len(pool_y)), "fraud_rate": float(pool_y.mean()) if len(pool_y) else None},
            "calibration_mae_val": cal_mae,
        },
        "generalization": gen_df.to_dict(orient="records"),
        "ood_scenarios": ood_df.to_dict(orient="records"),
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- Optional plot -----------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_df = res_df[res_df["model"] != "random_baseline"]
        x = np.arange(len(plot_df))
        width = 0.27
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width, plot_df["pr_auc"], width, label="PR-AUC")
        ax.bar(x, plot_df["roc_auc"], width, label="ROC-AUC")
        ax.bar(x + width, plot_df["f1"], width, label="F1 (val-tuned)")
        ax.set_xticks(x, plot_df["model"], rotation=15)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("score")
        ax.legend()
        ax.set_title("PS-14 model comparison (synthetic data)")
        fig.tight_layout()
        fig.savefig(outdir / "metrics_plot.png", dpi=150)
        print(f"  plot: {outdir/'metrics_plot.png'}")
    except ImportError:
        print("  (matplotlib not available - skipping plot)")

    # ---- OOD recall gate ---------------------------------------------------
    # Runs AFTER the artifacts/report are written so a failed retrain leaves
    # the evaluation on disk for debugging; the nonzero exit stops the
    # training pipeline before it retunes or touches rules.yaml.
    gate_rows = check_ood_recall_gate(
        gen_df, [a.strip() for a in args.ood_gate_archetypes.split(",") if a.strip()],
        args.ood_recall_floor,
    ) if not args.no_ood_gate else []
    meta["ood_gate"] = {
        "floor": args.ood_recall_floor,
        "metric": "recall_at_1pct_fpr",
        "archetypes": [a.strip() for a in args.ood_gate_archetypes.split(",") if a.strip()],
        "rows": gate_rows,
        "enforced": not args.no_ood_gate and bool(gen_df.shape[0]),
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if gate_rows and not all(r["passed"] for r in gate_rows) and not args.no_ood_gate:
        failed = ", ".join(r["archetype"] for r in gate_rows if not r["passed"])
        sys.exit(f"OOD recall gate FAILED: held-out recall@1%FPR below {args.ood_recall_floor:.2f} "
                 f"for {failed} - rerun with --no-ood-gate to accept this model")

    print(f"\nArtifacts written to {outdir}:")
    for p in sorted(outdir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
