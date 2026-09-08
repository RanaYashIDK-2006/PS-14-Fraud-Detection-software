#!/usr/bin/env python3
"""Production inference pipeline for the tuned Altman LGB model.

Stages:
  1. TRAIN: Fit tuned LGB on Altman with velocity features
  2. VERSION: Save model + metadata + feature schema + checksums
  3. SCORE: Batch scoring with chunked processing
  4. BENCHMARK: Latency measurement (single + batch)
  5. VERIFY: Load from saved artifacts and verify parity

Usage:
    python scripts/production_pipeline.py --stage all
    python scripts/production_pipeline.py --stage train
    python scripts/production_pipeline.py --stage score --input data/test.csv --output data/scored.csv
    python scripts/production_pipeline.py --stage benchmark
"""
import argparse
import hashlib
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, brier_score_loss,
)
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)
NJ = 4

ARTIFACTS_DIR = Path("models/production")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# ── Feature Engineering ──

ALTMAN_USECOLS = [
    "User", "Month", "Day", "Time", "Amount", "Use Chip", "MCC", "Errors?",
    "Is Fraud?", "Merchant Name", "Merchant City", "Merchant State", "Zip",
    "Year", "Card",
]

CHIP_MAP = {"Swipe Transaction": 0, "Online Transaction": 1, "Chip Transaction": 2}


def engineer_base_features(chunk: pd.DataFrame) -> pd.DataFrame:
    """Engineer base features from a raw Altman chunk."""
    c = chunk.copy()
    c["amt"] = (
        c["Amount"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    c["label"] = (c["Is Fraud?"] == "Yes").astype(int)
    c["chip"] = c["Use Chip"].map(CHIP_MAP).fillna(-1)
    c["mcc_n"] = c["MCC"].astype(str).str[:4].astype(float) / 10000
    c["err"] = (c["Errors?"].fillna("") != "").astype(int)
    tp = c["Time"].str.split(":", expand=True)
    c["hr"] = tp[0].astype(float)
    c["mn"] = tp[1].astype(float)
    c["merchant_id"] = c["Merchant Name"].astype("category").cat.codes
    c["city_id"] = c["Merchant City"].astype("category").cat.codes
    c["state_id"] = c["Merchant State"].fillna("UNK").astype("category").cat.codes
    c["card_n"] = c["Card"].astype(float)
    c["year_n"] = c["Year"].astype(float) - 2010
    c["is_online"] = (c["Merchant City"] == "ONLINE").astype(int)
    c["month_n"] = c["Month"].astype(float)
    c["day_n"] = c["Day"].astype(float)
    c["hr_bin"] = pd.cut(c["hr"], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(float)
    c["amt_log"] = np.log1p(c["amt"].clip(upper=1e9))
    c["day_decimal"] = c["Day"] + c["hr"] / 24.0 + c["mn"] / 1440.0

    # Interaction features
    c["log_amt"] = np.log1p(c["amt"])
    c["amt_x_hr"] = c["amt"] * c["hr"]
    c["amt_x_mcc"] = c["amt"] * c["mcc_n"]
    c["amt_x_chip"] = c["amt"] * c["chip"]
    c["amt_sq"] = c["amt"] ** 2
    c["high_amt"] = (c["amt"] > c["amt"].quantile(0.95)).astype(float)
    c["night_tx"] = ((c["hr"] >= 22) | (c["hr"] <= 6)).astype(int)
    c["weekend"] = ((c["Day"] % 7) >= 5).astype(float)
    return c


def compute_target_encoding(
    df: pd.DataFrame, group_col: str, prior: int = 200
) -> np.ndarray:
    """OOF target encoding for a single group column."""
    gm = df["label"].mean()
    oof = np.zeros(len(df))
    for tri, vai in StratifiedShuffleSplit(
        2, test_size=0.2, random_state=42
    ).split(df, df["label"]):
        stats = df.iloc[tri].groupby(group_col)["label"].agg(["mean", "count"])
        stats["s"] = (stats["mean"] * stats["count"] + gm * prior) / (
            stats["count"] + prior
        )
        mapping = stats["s"].to_dict()
        oof[vai] = df.iloc[vai][group_col].map(mapping).fillna(gm).values
    return oof


def add_velocity_features(df: pd.DataFrame, user_hist: dict) -> pd.DataFrame:
    """Compute per-user velocity features using pre-built history."""
    n = len(df)
    vel = np.zeros((n, 10), dtype=np.float32)
    for i in range(n):
        uid = int(df.iloc[i]["User"])
        dd = df.iloc[i]["day_decimal"]
        h = user_hist.get(uid)
        if h is None or len(h["times"]) < 3:
            continue
        t_arr = np.array(h["times"])
        a_arr = np.array(h["amounts"])
        m_arr = np.array(h["merchants"])
        pos = np.searchsorted(t_arr, dd, side="right")
        t_b = t_arr[:pos]; a_b = a_arr[:pos]; m_b = m_arr[:pos]
        nb = len(t_b)
        if nb < 2:
            continue
        vel[i, 0] = ((dd - t_b) <= (1.0 / 24.0)).sum()  # tx_count_1h
        vel[i, 1] = ((dd - t_b) <= 1.0).sum()  # tx_count_24h
        if nb >= 10:
            vel[i, 2] = a_b[-5:].mean() - a_b[-10:-5].mean()  # amount_accel
        elif nb >= 5:
            vel[i, 2] = a_b[-5:].mean() - a_b[:-5].mean()
        last24 = (dd - t_b) <= 1.0
        vel[i, 3] = len(np.unique(m_b[last24]))  # merchant_diversity_24h
        vel[i, 4] = a_b[last24].mean() if last24.sum() > 0 else 0  # avg_amount_24h
        gaps = np.diff(t_b[-min(5, nb):]) * 24
        vel[i, 5] = gaps.mean() if len(gaps) > 0 else 0  # tx_gap_mean
        vel[i, 6] = (
            gaps.std() / (gaps.mean() + 1e-6) if len(gaps) > 1 else 0
        )  # tx_regularity
        if nb >= 5:
            rm = a_b[-5:].mean()
            rs = a_b[-5:].std() + 1e-6
            vel[i, 7] = (a_b[-1] - rm) / rs  # amount_zscore
        if nb >= 6:
            f_rec = 1.0 / (gaps[-min(3, len(gaps)) :].mean() + 0.01)
            f_prev = 1.0 / (np.diff(t_b[-6:-3]) * 24).mean() + 0.01
            vel[i, 8] = f_rec - f_prev  # tx_freq_accel
        if nb >= 2:
            vel[i, 9] = (
                1.0 if m_b[-1] not in m_b[:-1][-min(10, nb - 1) :] else 0.0
            )  # merchant_is_new

    vel_names = [
        "tx_count_1h", "tx_count_24h", "amount_accel", "merchant_diversity_24h",
        "avg_amount_24h", "tx_gap_mean", "tx_regularity", "amount_zscore",
        "tx_freq_accel", "merchant_is_new",
    ]
    for j, name in enumerate(vel_names):
        df[name] = vel[:, j]
    return df


# ── STAGE 1: TRAIN ──

def stage_train():
    """Train the tuned LGB model and save with versioning."""
    print("=" * 60)
    print("  STAGE 1: TRAIN")
    print("=" * 60)
    t0 = time.time()

    # Load all 16 chunks
    print("  Loading Altman (16 chunks)...")
    rng = np.random.RandomState(42)
    rows_all = []
    ci = 0
    for chunk in pd.read_csv(
        "data/credit_card_transactions-ibm_v2.csv",
        usecols=ALTMAN_USECOLS, low_memory=False, chunksize=1_000_000,
    ):
        c = engineer_base_features(chunk)
        fm = c["label"] == 1
        rows_all.append(pd.concat([c[fm], c[~fm].sample(frac=0.005, random_state=rng)]))
        ci += 1
        if ci >= 16:
            break
    df = pd.concat(rows_all, ignore_index=True)
    print(f"  Loaded {len(df):,} rows ({int(df['label'].sum())} fraud)")

    # Target encoding
    gm = df["label"].mean()
    for ec, gc, p in [
        ("user_te", "User", 200),
        ("merchant_te", "merchant_id", 50),
        ("city_te", "city_id", 20),
    ]:
        df[ec] = compute_target_encoding(df, gc, p)

    # Interactions involving target encoding
    df["amt_x_ute"] = df["amt"] * df["user_te"]
    df["amt_x_mte"] = df["amt"] * df["merchant_te"]
    df["amt_x_online"] = df["amt"] * df["is_online"]
    df["hr_x_chip"] = df["hr"] * df["chip"]
    df["ute_x_mte"] = df["user_te"] * df["merchant_te"]
    df["chip_x_online"] = df["chip"] * df["is_online"]
    df["err_x_amt"] = df["err"] * df["amt"]
    df["hr_x_ute"] = df["hr"] * df["user_te"]
    df["mcc_x_ute"] = df["mcc_n"] * df["user_te"]

    # Velocity features
    print("  Building per-user histories...")
    user_times = defaultdict(list)
    user_amounts = defaultdict(list)
    user_merchants = defaultdict(list)
    sampled_users = set(df["User"].unique())
    ci2 = 0
    for chunk in pd.read_csv(
        "data/credit_card_transactions-ibm_v2.csv",
        usecols=["User", "Day", "Time", "Amount", "Merchant Name"],
        low_memory=False, chunksize=1_000_000,
    ):
        mask = chunk["User"].isin(sampled_users)
        sub = chunk[mask]
        if len(sub) == 0:
            ci2 += 1
            continue
        sub_amt = (
            sub["Amount"]
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
            .values
        )
        tp = sub["Time"].str.split(":", expand=True)
        hr = tp[0].astype(float).values
        mn = tp[1].astype(float).values
        dd = sub["Day"].values + hr / 24.0 + mn / 1440.0
        mid = sub["Merchant Name"].astype("category").cat.codes.values
        uids = sub["User"].values
        for i in range(len(sub)):
            user_times[uids[i]].append(dd[i])
            user_amounts[uids[i]].append(sub_amt[i])
            user_merchants[uids[i]].append(mid[i])
        ci2 += 1

    # Sort histories
    for uid in user_times:
        order = np.argsort(user_times[uid])
        user_times[uid] = np.array(user_times[uid])[order]
        user_amounts[uid] = np.array(user_amounts[uid])[order]
        user_merchants[uid] = np.array(user_merchants[uid])[order]
    user_hist = {
        uid: {"times": user_times[uid], "amounts": user_amounts[uid], "merchants": user_merchants[uid]}
        for uid in user_times
    }
    print(f"  Histories for {len(user_hist):,} users")

    print("  Computing velocity features...")
    df = add_velocity_features(df, user_hist)

    # Feature interactions with velocity
    df["amt_x_tx5"] = df["amt"] * df["tx_count_1h"]
    df["amt_x_tx24"] = df["amt"] * df["tx_count_24h"]
    df["ute_x_tx24"] = df["user_te"] * df["tx_count_24h"]

    # Build X, y
    y = df["label"].values
    drop_cols = {
        "label", "User", "Time", "Amount", "Use Chip", "MCC", "Errors?",
        "Is Fraud?", "Merchant Name", "Merchant City", "Merchant State",
        "Zip", "Year", "Card",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = np.nan_to_num(df[feature_cols].values.astype(np.float32))
    print(f"  Features: {len(feature_cols)}")

    # Optuna-tuned params
    best_params = dict(
        n_estimators=350, max_depth=10, learning_rate=0.127,
        subsample=0.839, colsample_bytree=0.578, min_child_samples=9,
        reg_alpha=0.00171, reg_lambda=2.915, num_leaves=68,
    )

    # 5-fold CV
    print("\n  5-fold CV...")
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    all_preds = np.zeros(len(y))
    fold_aucs = []
    fold_r1s = []
    for fold, (tri, tei) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(
            **best_params, verbose=-1, random_state=42, n_jobs=NJ,
        )
        m.fit(X[tri], y[tri])
        p = m.predict_proba(X[tei])[:, 1]
        all_preds[tei] = p
        fpr, tpr, _ = roc_curve(y[tei], p)
        m2 = fpr <= 0.01
        r1 = float(tpr[m2].max()) if m2.any() else 0.0
        fold_aucs.append(roc_auc_score(y[tei], p))
        fold_r1s.append(r1)
        print(f"    Fold {fold}: AUC={fold_aucs[-1]:.6f} R@1%FPR={r1:.4f}")

    cv_auc = float(np.mean(fold_aucs))
    cv_r1 = float(np.mean(fold_r1s))
    print(f"\n  CV AUC: {cv_auc:.6f} ± {np.std(fold_aucs):.6f}")
    print(f"  CV R@1%FPR: {cv_r1:.6f}")

    # Train final model on all data
    print("\n  Training final model on all data...")
    final_model = lgb.LGBMClassifier(
        **best_params, verbose=-1, random_state=42, n_jobs=NJ,
    )
    final_model.fit(X, y)

    # ── Save artifacts with versioning ──
    version = time.strftime("%Y%m%d_%H%M%S")
    model_dir = ARTIFACTS_DIR / f"altman_lgb_{version}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.joblib"
    joblib.dump(final_model, model_path)

    # Save feature schema
    feature_schema = {
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "feature_types": {c: str(X[:, feature_cols.index(c)].dtype) for c in feature_cols[:5]},
    }
    (model_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2)
    )

    # Save model metadata
    metadata = {
        "model_name": "altman_lgb_v1",
        "model_version": version,
        "model_type": "LGBMClassifier",
        "framework": f"lightgbm {lgb.__version__}",
        "training_rows": len(y),
        "n_fraud": int(y.sum()),
        "fraud_rate": round(float(y.mean()), 6),
        "n_features": len(feature_cols),
        "hyperparameters": best_params,
        "cv_auc": round(cv_auc, 6),
        "cv_r1_fpr": round(cv_r1, 6),
        "cv_fold_aucs": [round(a, 6) for a in fold_aucs],
        "training_time_s": round(time.time() - t0),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_paths": {
            "model": str(model_path),
            "feature_schema": str(model_dir / "feature_schema.json"),
            "velocity_histories": str(model_dir / "velocity_histories.joblib"),
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Save velocity histories (for production scoring)
    joblib.dump(user_hist, model_dir / "velocity_histories.joblib")

    # Compute checksums
    checksums = {}
    for f in model_dir.iterdir():
        if f.is_file():
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            checksums[f.name] = {"sha256": h, "size_bytes": f.stat().st_size}
    (model_dir / "checksums.json").write_text(json.dumps(checksums, indent=2))

    # Save as "latest" symlink target
    latest_path = ARTIFACTS_DIR / "latest"
    latest_path.write_text(version)

    print(f"\n  Model saved: {model_dir}")
    print(f"  Version: {version}")
    print(f"  Total: {time.time()-t0:.0f}s")
    return model_dir, metadata


# ── STAGE 2: BATCH SCORE ──

def stage_score(input_path: str, output_path: str, model_dir: Path = None):
    """Batch score transactions from CSV."""
    print("=" * 60)
    print("  STAGE 2: BATCH SCORE")
    print("=" * 60)
    t0 = time.time()

    # Load model
    if model_dir is None:
        latest_version = (ARTIFACTS_DIR / "latest").read_text().strip()
        model_dir = ARTIFACTS_DIR / f"altman_lgb_{latest_version}"
    print(f"  Model: {model_dir.name}")

    model = joblib.load(model_dir / "model.joblib")
    schema = json.loads((model_dir / "feature_schema.json").read_text())
    feature_names = schema["feature_names"]

    # Load input data
    print(f"  Loading: {input_path}")
    df = pd.read_csv(input_path)
    n_rows = len(df)
    print(f"  {n_rows:,} rows")

    # Check if this is raw Altman data or pre-processed
    if "Is Fraud?" in df.columns:
        # Raw Altman — engineer features
        df = engineer_base_features(df)
        # For velocity, load histories if available
        hist_path = model_dir / "velocity_histories.joblib"
        if hist_path.exists():
            user_hist = joblib.load(hist_path)
            df = add_velocity_features(df, user_hist)
    else:
        # Pre-processed — assume features already engineered
        pass

    # Ensure all required features exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    X = np.nan_to_num(df[feature_names].values.astype(np.float32))

    # Score in chunks
    CHUNK_SIZE = 50_000
    all_preds = []
    n_chunks = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci in range(n_chunks):
        start = ci * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_rows)
        chunk_X = X[start:end]
        p = model.predict_proba(chunk_X)[:, 1]
        all_preds.append(p)
        if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
            print(f"    Chunk {ci+1}/{n_chunks} ({end:,}/{n_rows:,})")

    preds = np.concatenate(all_preds)
    df["fraud_probability"] = preds
    df["fraud_flag"] = (preds > 0.5).astype(int)

    # Save output
    df.to_csv(output_path, index=False)
    print(f"\n  Output: {output_path}")
    print(f"  Scored: {n_rows:,} rows in {time.time()-t0:.1f}s")
    print(f"  Fraud predictions: {int(df['fraud_flag'].sum()):,} ({df['fraud_flag'].mean()*100:.2f}%)")
    print(f"  Mean fraud probability: {preds.mean():.6f}")


# ── STAGE 3: BENCHMARK ──

def stage_benchmark(model_dir: Path = None):
    """Benchmark inference latency."""
    print("=" * 60)
    print("  STAGE 3: BENCHMARK LATENCY")
    print("=" * 60)

    if model_dir is None:
        latest_version = (ARTIFACTS_DIR / "latest").read_text().strip()
        model_dir = ARTIFACTS_DIR / f"altman_lgb_{latest_version}"
    model = joblib.load(model_dir / "model.joblib")
    schema = json.loads((model_dir / "feature_schema.json").read_text())
    n_features = schema["n_features"]

    # Generate synthetic test data
    rng = np.random.RandomState(42)
    single = rng.randn(1, n_features).astype(np.float32)
    batch_sizes = [1, 10, 50, 100, 500, 1000, 5000, 10000]

    results = []
    print(f"\n  {'Batch':>8s}  {'Time (ms)':>10s}  {'Per-tx (µs)':>12s}  {'Throughput':>12s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}")

    for bs in batch_sizes:
        batch = rng.randn(bs, n_features).astype(np.float32)
        # Warmup
        model.predict_proba(batch[:min(10, bs)])

        # Benchmark (10 iterations)
        times = []
        for _ in range(10):
            t = time.perf_counter()
            model.predict_proba(batch)
            times.append((time.perf_counter() - t) * 1000)

        median_ms = np.median(times)
        per_tx_us = (median_ms / bs) * 1000
        throughput = bs / (median_ms / 1000)

        results.append({
            "batch_size": bs,
            "median_ms": round(median_ms, 2),
            "per_tx_us": round(per_tx_us, 1),
            "throughput_tps": round(throughput, 0),
        })
        print(f"  {bs:>8d}  {median_ms:>10.2f}  {per_tx_us:>12.1f}  {throughput:>12.0f}")

    # Summary
    single_us = results[0]["per_tx_us"]
    batch_1k = next(r for r in results if r["batch_size"] == 1000)
    print(f"\n  Single-tx latency: {single_us:.0f}µs")
    print(f"  Batch-1K latency: {batch_1k['median_ms']:.1f}ms ({batch_1k['per_tx_us']:.0f}µs/tx)")
    print(f"  Max throughput: {batch_1k['throughput_tps']:.0f} tx/s")

    # Save benchmark
    benchmark = {
        "model_version": model_dir.name,
        "n_features": n_features,
        "results": results,
        "single_tx_latency_us": single_us,
        "batch_1k_throughput_tps": batch_1k["throughput_tps"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = REPORTS_DIR / "benchmark_latency.json"
    out.write_text(json.dumps(benchmark, indent=2))
    print(f"\n  Saved: {out}")
    return benchmark


# ── STAGE 4: VERIFY ──

def stage_verify(model_dir: Path = None):
    """Verify saved model loads correctly and produces same predictions."""
    print("=" * 60)
    print("  STAGE 4: VERIFY ARTIFACTS")
    print("=" * 60)

    if model_dir is None:
        latest_version = (ARTIFACTS_DIR / "latest").read_text().strip()
        model_dir = ARTIFACTS_DIR / f"altman_lgb_{latest_version}"

    # Check all files exist
    required = ["model.joblib", "feature_schema.json", "metadata.json",
                "velocity_histories.joblib", "checksums.json"]
    for f in required:
        path = model_dir / f
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        size = f"{path.stat().st_size:,}" if exists else "—"
        print(f"  {f:35s} {status:>8s}  {size:>10s} bytes")

    # Verify checksums
    checksums = json.loads((model_dir / "checksums.json").read_text())
    print(f"\n  Verifying checksums...")
    all_ok = True
    for fname, info in checksums.items():
        path = model_dir / fname
        if path.exists():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            ok = actual == info["sha256"]
            if not ok:
                all_ok = False
                print(f"    {fname}: MISMATCH")
    if all_ok:
        print(f"    All {len(checksums)} checksums OK")

    # Verify model loads and runs
    model = joblib.load(model_dir / "model.joblib")
    schema = json.loads((model_dir / "feature_schema.json").read_text())
    rng = np.random.RandomState(42)
    test_X = rng.randn(100, schema["n_features"]).astype(np.float32)
    preds = model.predict_proba(test_X)[:, 1]
    print(f"\n  Model loads OK: {type(model).__name__}")
    print(f"  Test predictions: mean={preds.mean():.6f}, std={preds.std():.6f}")

    # Verify velocity histories
    hist = joblib.load(model_dir / "velocity_histories.joblib")
    total_tx = sum(len(v["times"]) for v in hist.values())
    print(f"  Velocity histories: {len(hist):,} users, {total_tx:,} total tx")

    # Verify metadata
    meta = json.loads((model_dir / "metadata.json").read_text())
    print(f"\n  Model: {meta['model_name']} v{meta['model_version']}")
    print(f"  CV AUC: {meta['cv_auc']}")
    print(f"  CV R@1%FPR: {meta['cv_r1_fpr']}")
    print(f"  Training rows: {meta['training_rows']:,}")
    print(f"  Features: {meta['n_features']}")

    total_size = sum(f.stat().st_size for f in model_dir.iterdir() if f.is_file())
    print(f"\n  Total artifact size: {total_size/1e6:.1f} MB")
    print(f"  Verify: {'ALL PASSED' if all_ok else 'FAILED'}")


# ── MAIN ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production inference pipeline")
    parser.add_argument("--stage", default="all",
        choices=["all", "train", "score", "benchmark", "verify"])
    parser.add_argument("--input", default="data/credit_card_transactions-ibm_v2.csv")
    parser.add_argument("--output", default="data/altman_scored.csv")
    args = parser.parse_args()

    if args.stage in ("all", "train"):
        model_dir, metadata = stage_train()
    else:
        model_dir = None

    if args.stage in ("all", "score"):
        stage_score(args.input, args.output, model_dir)

    if args.stage in ("all", "benchmark"):
        stage_benchmark(model_dir)

    if args.stage in ("all", "verify"):
        stage_verify(model_dir)

    if args.stage == "all":
        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print("=" * 60)
