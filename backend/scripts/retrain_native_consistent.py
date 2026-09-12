#!/usr/bin/env python3
"""Retrain the Altman-NATIVE ensemble with a GENUINELY chronological split.

Fixes the two blockers the forensic audits found in the legacy native
trainer (train_altman_native.py):

1. SPLIT VALIDITY — the old trainer used `Month <= 9` vs `Month > 9`, which
   pools 26 years of data (1995-2020): 2020 Jan-Feb sits in TRAIN while 1995
   Oct-Dec is "TEST". This retrain sorts by full datetime and splits
   chronologically:  train <= 2015 | validation 2016-2017 | final test 2018-2020.
2. FEATURE PARITY — every row's 48 native features come from the SHARED
   derivation (src/privacy_layer/native_features.derive_native_features),
   the same module the production privacy layer uses at ingest, so
   train == production by construction (feature-parity audit Part 2 / #27).

Methodology (mirrors the causal runtime retrain):
  - Stream all 24.4M rows; keep ALL fraud + ~1% legit (rng 42) — the native
    corpus is 0.12% fraud, so 1% legit sampling is the documented approach.
  - Expanding velocity/fraud-rate context computed leakage-safe (shifted:
    the event itself is never counted in its own history).
  - Validation-only threshold selection (FPR<=1% policy, locked from val).
  - Governance record (model_records/) with artifact hashes, dataset hash,
    split boundaries, locked threshold, and performance metrics.
  - Untouched final test evaluated at the LOCKED threshold.

Usage:
  python scripts/retrain_native_consistent.py [--out DIR] [--val-start 2016] ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root.parent  # repo root
sys.path.insert(0, str(ROOT / "backend"))

from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES, derive_native_features  # noqa: E402

DATA_CSV = ROOT / "data" / "credit_card_transactions-ibm_v2.csv"
OUT_DIR = ROOT / "models" / "production" / "altman_native_v2"
RECORDS_DIR = ROOT / "models" / "model_records"
SEED = 42

USECOLS = ["User", "Card", "Year", "Month", "Day", "Time", "Amount",
           "Use Chip", "MCC", "Merchant Name", "Merchant City",
           "Merchant State", "Zip", "Errors?", "Is Fraud?"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recall_at_fpr(y_true, y_score, target=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target, side="right")
    return float(tpr[idx - 1]) if idx > 0 else 0.0


def load_sample() -> pd.DataFrame:
    """Stream all rows, keep all fraud + ~1% legit, return chronological frame."""
    print("[1/7] Streaming 24.4M rows (keep all fraud + 1% legit)...", flush=True)
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    chunks = []
    n_total = 0
    n_fraud = 0
    for chunk in pd.read_csv(DATA_CSV, usecols=USECOLS, low_memory=False,
                             chunksize=1_000_000):
        n_total += len(chunk)
        fr = chunk["Is Fraud?"].eq("Yes")
        n_fraud += int(fr.sum())
        keep = fr | (rng.random(len(chunk)) < 0.01)
        chunks.append(chunk[keep].copy())
    df = pd.concat(chunks, ignore_index=True)
    print(f"  scanned {n_total:,} | fraud {n_fraud:,} | sampled {len(df):,} "
          f"({df['Is Fraud?'].eq('Yes').sum():,} fraud) | {time.time()-t0:.0f}s",
          flush=True)

    print("[2/7] Parsing raw columns + sorting chronologically...", flush=True)
    df["amt"] = (df["Amount"].str.replace("$", "", regex=False)
                 .str.replace(",", "", regex=False).astype(float))
    df["is_fraud"] = df["Is Fraud?"].eq("Yes").astype(int)
    df["ts"] = pd.to_datetime(
        df[["Year", "Month", "Day"]].assign(
            hour=df["Time"].str.split(":").str[0].astype(int),
            minute=df["Time"].str.split(":").str[1].astype(int)),
    )
    # Genuine chronological order (leakage-safe expanding context needs it)
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"  ts range {df['ts'].min()} .. {df['ts'].max()}", flush=True)
    return df


def build_context_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe expanding context per entity, then shared 48-vector."""
    print("[3/7] Expanding context (shifted, leakage-safe)...", flush=True)
    t0 = time.time()
    g = df.groupby("User")
    # per-user velocity (count BEFORE this event)
    df["user_tx_count"] = g.cumcount()
    df["user_avg_amt"] = (g["amt"].cumsum() - df["amt"]) / df["user_tx_count"].clip(lower=1)
    df["user_avg_amt"] = df["user_avg_amt"].where(df["user_tx_count"] > 0, 0.0)
    # per-card / per-merchant velocity
    df["card_tx_count"] = df.groupby("Card").cumcount()
    df["merch_tx_count"] = df.groupby("Merchant Name").cumcount()
    # per-user merchant/city diversity (distinct seen BEFORE this event)
    first_m = ~df.duplicated(subset=["User", "Merchant Name"], keep="first")
    df["user_merchant_diversity"] = (first_m.groupby(df["User"]).cumsum() - first_m).clip(lower=0)
    first_c = ~df.duplicated(subset=["User", "Merchant City"], keep="first")
    df["user_city_diversity"] = (first_c.groupby(df["User"]).cumsum() - first_c).clip(lower=0)
    # per-user-merchant interaction count (BEFORE this event)
    df["user_merch_count"] = df.groupby(["User", "Merchant Name"]).cumcount()
    # entity fraud rates (expanding mean of PAST labels, shifted)
    df["user_fraud_rate"] = (
        (g["is_fraud"].cumsum() - df["is_fraud"]) / df["user_tx_count"].clip(lower=1)
    ).where(df["user_tx_count"] > 0, 0.001)
    gm = df.groupby("Merchant Name")
    df["merch_tx_count_all"] = gm.cumcount()
    df["merch_fraud_rate"] = (
        (gm["is_fraud"].cumsum() - df["is_fraud"]) / df["merch_tx_count_all"].clip(lower=1)
    ).where(df["merch_tx_count_all"] > 0, 0.001)
    gc = df.groupby("Merchant City")
    df["city_tx_count_all"] = gc.cumcount()
    df["city_fraud_rate"] = (
        (gc["is_fraud"].cumsum() - df["is_fraud"]) / df["city_tx_count_all"].clip(lower=1)
    ).where(df["city_tx_count_all"] > 0, 0.001)

    print("[4/7] Shared 48-feature derivation per row...", flush=True)
    rows = []
    for i, r in df.iterrows():
        vel = {
            "user_tx_count": int(r["user_tx_count"]),
            "user_avg_amt": float(r["user_avg_amt"]),
            "card_tx_count": int(r["card_tx_count"]),
            "merch_tx_count": int(r["merch_tx_count"]),
            "user_merchant_diversity": max(float(r["user_merchant_diversity"]), 1.0),
            "user_city_diversity": max(float(r["user_city_diversity"]), 1.0),
            "user_merch_count": int(r["user_merch_count"]),
        }
        rates = {
            "user_fraud_rate": float(r["user_fraud_rate"]),
            "merch_fraud_rate": float(r["merch_fraud_rate"]),
            "city_fraud_rate": float(r["city_fraud_rate"]),
        }
        raw = {
            "amount": float(r["amt"]),
            "ts": r["ts"],
            "use_chip": str(r["Use Chip"]),
            "mcc": int(r["MCC"]) if pd.notna(r["MCC"]) else 0,
            "merchant_city": str(r["Merchant City"] or ""),
            # NaN-safe: pandas NaN is truthy, so str(x or "") would yield "nan"
            # for missing states (same bug class as the err fix) and silently
            # turn has_state into constant-1 / poison is_online_or_no_state.
            # Live payloads send "" for missing state; training must match.
            "merchant_state": "" if pd.isna(r["Merchant State"]) else str(r["Merchant State"]),
            "zip": "" if pd.isna(r["Zip"]) else str(r["Zip"]),
            "card": str(r["Card"]),
            "errors": ("" if pd.isna(r["Errors?"])
                        else str(r["Errors?"])),
            "merchant_name": str(r["Merchant Name"]),
            "merchant_id": str(r["Merchant Name"]),
            "city_id": str(r["Merchant City"] or ""),
            "card_id": str(r["Card"]),
        }
        rows.append(derive_native_features(raw, vel, rates))
    F = pd.DataFrame(rows)
    F["is_fraud"] = df["is_fraud"].values
    F["ts"] = df["ts"].values
    print(f"  features done ({time.time()-t0:.0f}s)", flush=True)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--val-start", type=int, default=2016,
                    help="first year of validation window")
    ap.add_argument("--test-start", type=int, default=2018,
                    help="first year of final test window")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_hash = file_sha256(DATA_CSV)
    df = load_sample()
    F = build_context_and_features(df)

    # ── Chronological split ───────────────────────────────────────────────
    print("[5/7] Chronological split "
          f"(train<{args.val_start}, val {args.val_start}-{args.test_start-1}, "
          f"test>={args.test_start})...", flush=True)
    yrs = F["ts"].dt.year
    tr_mask = yrs < args.val_start
    va_mask = (yrs >= args.val_start) & (yrs < args.test_start)
    te_mask = yrs >= args.test_start
    Xtr, ytr = F.loc[tr_mask, ALTMAN_NATIVE_FEATURES].values.astype(np.float32), F.loc[tr_mask, "is_fraud"].values
    Xva, yva = F.loc[va_mask, ALTMAN_NATIVE_FEATURES].values.astype(np.float32), F.loc[va_mask, "is_fraud"].values
    Xte, yte = F.loc[te_mask, ALTMAN_NATIVE_FEATURES].values.astype(np.float32), F.loc[te_mask, "is_fraud"].values
    print(f"  train {len(Xtr):,} ({ytr.sum():,} fraud) | val {len(Xva):,} "
          f"({yva.sum():,}) | test {len(Xte):,} ({yte.sum():,})", flush=True)
    if ytr.sum() == 0 or yva.sum() == 0 or yte.sum() == 0:
        print("FATAL: a window has zero fraud — adjust --val-start/--test-start", flush=True)
        sys.exit(2)

    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # ── Train ensemble (same family as the causal runtime model) ──────────
    print("[6/7] Training XGB+LGB+CB ensemble...", flush=True)
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    t0 = time.time()
    spw = (1 - ytr.mean()) / ytr.mean()
    spw = min(spw, 20.0)
    m_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                              scale_pos_weight=spw, random_state=SEED, n_jobs=4,
                              eval_metric="auc")
    m_xgb.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], verbose=False)
    m_lgb = lgb.LGBMClassifier(n_estimators=400, max_depth=7, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                               scale_pos_weight=spw, random_state=SEED, n_jobs=4, verbose=-1)
    m_lgb.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)])
    m_cb = cb.CatBoostClassifier(iterations=400, depth=7, learning_rate=0.05,
                                 l2_leaf_reg=3, random_seed=SEED, verbose=0,
                                 auto_class_weights="Balanced")
    m_cb.fit(Xtr_s, ytr, eval_set=(Xva_s, yva))
    print(f"  trained in {time.time()-t0:.0f}s", flush=True)

    def ens(X):
        return (0.34 * m_xgb.predict_proba(X)[:, 1]
                + 0.33 * m_lgb.predict_proba(X)[:, 1]
                + 0.33 * m_cb.predict_proba(X)[:, 1])

    # ── Validation-only threshold (FPR<=1% policy, locked from VAL) ───────
    print("[7/7] Validation-only threshold + untouched final test...", flush=True)
    p_va = ens(Xva_s)
    fpr_va, tpr_va, thr_va = roc_curve(yva, p_va)
    valid = fpr_va <= 0.01
    if valid.any():
        idx = int(np.argmax(tpr_va[valid]))  # highest recall within FPR<=1%
        locked = float(thr_va[idx])
        val_fpr = float(fpr_va[idx])
        val_recall = float(tpr_va[idx])
    else:
        idx = int(np.argmin(np.abs(fpr_va - 0.01)))
        locked = float(thr_va[idx])
        val_fpr = float(fpr_va[idx])
        val_recall = float(tpr_va[idx])
    print(f"  LOCKED threshold (val FPR<=1%): {locked:.6f} "
          f"(val FPR={val_fpr:.5f}, val recall={val_recall:.5f})", flush=True)

    p_te = ens(Xte_s)
    te_auc = float(roc_auc_score(yte, p_te))
    te_apr = float(average_precision_score(yte, p_te))
    te_r1 = recall_at_fpr(yte, p_te, 0.01)
    te_pred = (p_te >= locked).astype(int)
    tp = int(((te_pred == 1) & (yte == 1)).sum())
    fp = int(((te_pred == 1) & (yte == 0)).sum())
    fn = int(((te_pred == 0) & (yte == 1)).sum())
    tn = int(((te_pred == 0) & (yte == 0)).sum())
    te_fpr = fp / max(fp + tn, 1)
    te_recall = tp / max(tp + fn, 1)
    te_prec = tp / max(tp + fp, 1)
    print(f"  TEST (untouched, locked thr {locked:.6f}): "
          f"AUC={te_auc:.4f} PR-AUC={te_apr:.4f} R@1%={te_r1:.4f} "
          f"recall={te_recall:.4f} FPR={te_fpr:.5f} prec={te_prec:.4f} "
          f"TP={tp} FP={fp} FN={fn} TN={tn}", flush=True)

    # ── Save artifacts + governance record ────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"altman_native_v2_{ts}"
    for name, obj in [("xgb_native.joblib", m_xgb), ("lgb_native.joblib", m_lgb),
                      ("cb_native.joblib", m_cb), ("scaler_native.joblib", scaler)]:
        joblib.dump(obj, out_dir / name)
    (out_dir / "feature_list.json").write_text(
        json.dumps(ALTMAN_NATIVE_FEATURES, indent=2))

    artifact_hashes = {}
    for name in ["xgb_native.joblib", "lgb_native.joblib", "cb_native.joblib",
                 "scaler_native.joblib", "feature_list.json"]:
        artifact_hashes[name] = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()

    manifest = {
        "model_version": model_id,
        "model_type": "xgb_lgb_cb_native",
        "feature_schema_version": "altman_native_v2",
        "feature_source": ("shared src/privacy_layer/native_features.py — "
                           "train == production by construction"),
        "dataset": str(DATA_CSV),
        "dataset_sha256": data_hash,
        "dataset_rows_scanned": 24_386_900,
        "training_rows": int(len(Xtr)),
        "n_fraud_train": int(ytr.sum()),
        "n_features": len(ALTMAN_NATIVE_FEATURES),
        "split": {"train_end": f"{args.val_start-1}-12-31",
                  "val": f"{args.val_start}-01-01..{args.test_start-1}-12-31",
                  "test_start": f"{args.test_start}-01-01",
                  "method": "chronological by ts (legacy native used Month<=9 across 26 years)"},
        "locked_threshold": locked,
        "val_fpr_at_lock": round(val_fpr, 6),
        "val_recall_at_lock": round(val_recall, 6),
        "test_metrics_at_lock": {
            "auc": round(te_auc, 6), "pr_auc": round(te_apr, 6),
            "recall_at_1pct_fpr_reference": round(te_r1, 6),
            "recall": round(te_recall, 6), "fpr": round(te_fpr, 6),
            "precision": round(te_prec, 6), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "test_rows": int(len(Xte)),
        "test_fraud": int(yte.sum()),
        "ensemble_weights": {"xgb": 0.34, "lgb": 0.33, "cb": 0.33},
        "seed": SEED,
        "legacy_reference": "altman_native_20260830_135348 (invalid Month<=9 split)",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "is_native": True,
        "artifacts": artifact_hashes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "model_id": model_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "training_dataset_path": str(DATA_CSV),
        "training_dataset_sha256": data_hash,
        "code_version": "scripts/retrain_native_consistent.py + src/privacy_layer/native_features.py",
        "feature_schema_version": "altman_native_v2",
        "features": ALTMAN_NATIVE_FEATURES,
        "algorithm": "xgb_lgb_cb_ensemble (48 native features)",
        "hyperparameters": {
            "xgb": str(m_xgb.get_params()), "lgb": str(m_lgb.get_params()),
            "cb": str(m_cb.get_params()),
            "ensemble_weights": {"xgb": 0.34, "lgb": 0.33, "cb": 0.33}},
        "training_seed": SEED,
        "training_period": f"1995-06-01..{args.val_start-1}-12-31",
        "validation_period": f"{args.val_start}-01-01..{args.test_start-1}-12-31",
        "final_test_period": f"{args.test_start}-01-01..2020-02-29",
        "selected_threshold": locked,
        "threshold_policy": "validation-only: max recall within val FPR<=1%",
        "calibration_version": "none (ensemble proba; calibration deferred to deploy step)",
        "performance_metrics": {
            "val": {"fpr_at_lock": round(val_fpr, 6), "recall_at_lock": round(val_recall, 6)},
            "test": manifest["test_metrics_at_lock"],
        },
        "artifact_files": artifact_hashes,
        "promotion": {"gates": "leakage-safe shared derivation; chronological split; "
                               "validation-only threshold; untouched final test",
                      "reason": "user request: deploy the Altman-native (real 24.4M-row) model"},
        "parent_model_id": "altman_native_20260830_135348",
    }
    (RECORDS_DIR / f"{model_id}.json").write_text(json.dumps(record, indent=2))

    print("\n" + "=" * 72, flush=True)
    print(f"SAVED: {out_dir}  (model_id={model_id})", flush=True)
    print(f"  dataset sha256: {data_hash[:16]}...", flush=True)
    print(f"  locked threshold: {locked:.6f} (val FPR {val_fpr:.5f})", flush=True)
    print(f"  TEST AUC={te_auc:.4f} PR-AUC={te_apr:.4f} R@1%={te_r1:.4f} "
          f"recall={te_recall:.4f} FPR={te_fpr:.5f} precision={te_prec:.4f}",
          flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()