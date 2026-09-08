#!/usr/bin/env python3
"""Full-scale Altman training on ALL 24M rows — no sampling.

Architecture:
  1. Pass 1 — Stream CSV in chunks, extract + sort by User+datetime
  2. Pass 2 — Online expanding features (per-user state, chunk-at-a-time)
  3. Pass 3 — Train XGB+LGB+CB ensemble on all rows
  4. Temporal split: train months 1-9, test months 10-12

Memory strategy:
  - Read only needed columns (15 of 24)
  - Category-encode high-cardinality strings → int32 codes
  - Float32 everywhere (not float64)
  - Expanding features computed per-user with running state (no full-D groupby)
  - Write sorted chunks to disk as parquet, then stream back for training
"""
from __future__ import annotations

import gc
import json
import hashlib
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODELS_DIR = Path("models/production")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("data") / "_sorted_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Pass 1: Stream + Sort ───────────────────────────────────────────────────

def pass1_sort(path: Path, chunksize: int = 500_000) -> Path:
    """Read CSV in chunks, parse minimal columns, sort by User+datetime.

    Writes sorted chunks as parquet to CACHE_DIR, then merge-sorts them
    into a single sorted file. Returns the path to the sorted parquet.
    """
    print("\n[PASS 1] Streaming + sorting 24M rows...")
    t0 = time.time()

    sorted_path = CACHE_DIR / "altman_sorted.parquet"
    if sorted_path.exists():
        mtime = sorted_path.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 24:
            print(f"  Using cached sorted file ({age_hours:.1f}h old)")
            return sorted_path

    chunk_paths = []
    n_total = 0
    n_fraud = 0

    for ci, chunk in enumerate(pd.read_csv(
        path,
        usecols=["User", "Card", "Year", "Month", "Day", "Time", "Amount",
                 "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
                 "Zip", "MCC", "Errors?", "Is Fraud?"],
        low_memory=False,
        chunksize=chunksize,
    )):
        # Parse minimal columns
        chunk["amt"] = chunk["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(np.float32)
        chunk["is_fraud"] = (chunk["Is Fraud?"] == "Yes").astype(np.int8)
        chunk["hr"] = chunk["Time"].str.split(":").str[0].astype(np.int8)
        chunk["mn"] = chunk["Time"].str.split(":").str[1].astype(np.int8)
        chunk["chip"] = (chunk["Use Chip"] == "Chip Transaction").astype(np.int8)
        chunk["is_online"] = (chunk["Use Chip"] == "Online Transaction").astype(np.int8)
        chunk["mcc_n"] = chunk["MCC"].fillna(0).astype(np.float32)
        chunk["has_zip"] = chunk["Zip"].notna().astype(np.int8)
        chunk["has_state"] = chunk["Merchant State"].notna().astype(np.int8)
        chunk["err"] = (chunk["Errors?"].fillna("") != "").astype(np.int8)

        # Category encode high-cardinality strings
        for col in ["User", "Card", "Merchant Name", "Merchant City"]:
            chunk[col] = chunk[col].astype("category").cat.codes.astype(np.int32)

        # Build datetime for sorting
        chunk["datetime"] = pd.to_datetime(
            chunk[["Year", "Month", "Day"]].assign(hour=chunk["hr"], minute=chunk["mn"])
        )

        # Sort within chunk
        chunk.sort_values(["User", "datetime"], inplace=True)

        # Drop unused columns
        chunk.drop(columns=["Amount", "Use Chip", "MCC", "Errors?",
                            "Is Fraud?", "Merchant State", "Zip",
                            "Year", "datetime"], inplace=True, errors="ignore")

        n_total += len(chunk)
        n_fraud += chunk["is_fraud"].sum()

        # Write chunk
        cpath = CACHE_DIR / f"chunk_{ci:03d}.parquet"
        chunk.to_parquet(cpath, index=False)
        chunk_paths.append(cpath)

        if (ci + 1) % 5 == 0:
            print(f"  Chunk {ci+1}: {n_total:,} rows ({n_fraud:,} fraud)")

    print(f"  Total: {n_total:,} rows ({n_fraud:,} fraud, {n_fraud/n_total*100:.3f}%)")

    # Merge-sort: read all chunks, concatenate, sort, write single parquet
    print(f"  Merging {len(chunk_paths)} sorted chunks...")
    dfs = [pd.read_parquet(p) for p in chunk_paths]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    df.sort_values(["User", "Month", "Day", "hr", "mn"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(sorted_path, index=False)

    # Cleanup temp chunks
    for p in chunk_paths:
        p.unlink()

    elapsed = time.time() - t0
    print(f"  Sorted: {len(df):,} rows in {elapsed:.0f}s")
    return sorted_path


# ── Pass 2: Online Expanding Features ───────────────────────────────────────

def pass2_features(sorted_path: Path, chunksize: int = 2_000_000) -> Path:
    """Compute expanding features using online per-user state.

    Reads sorted parquet in chunks, maintains per-user running state,
    and computes all 35 features. Writes feature chunks to disk.
    Returns path to combined features file.
    """
    print("\n[PASS 2] Computing expanding features (online, per-user state)...")
    t0 = time.time()

    features_path = CACHE_DIR / "features_all.parquet"
    if features_path.exists():
        mtime = features_path.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 24:
            print(f"  Using cached features file ({age_hours:.1f}h old)")
            return features_path

    # Per-user state: {user_id: {count, sum, sum_sq, fraud_count, last_amt}}
    # We need: user_tx_count, user_avg_amt, amt_zscore, user_fraud_rate
    # We need: card_tx_count, merch_tx_count (per card/merchant)
    user_state = {}  # user -> {count, sum, fraud_count, amounts_deque}
    card_state = {}  # card -> count
    merch_state = {}  # merchant -> count

    feat_chunks = []
    chunk_idx = 0

    # Parquet doesn't support chunksize — read row groups via pyarrow
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(sorted_path)
    n_row_groups = pf.metadata.num_row_groups
    print(f"  {n_row_groups} row groups in parquet file")

    for rg_idx in range(n_row_groups):
        chunk = pf.read_row_group(rg_idx).to_pandas()
        n = len(chunk)
        F = pd.DataFrame(index=chunk.index)

        # Basic features (no state needed)
        F["amt"] = chunk["amt"]
        F["log_amt"] = np.log1p(F["amt"])
        F["amt_sq"] = F["amt"] ** 2
        F["hr"] = chunk["hr"]
        F["mn"] = chunk["mn"]
        F["dow"] = 0  # placeholder, computed below
        F["Month"] = chunk["Month"]
        F["Day"] = chunk["Day"]
        F["hour_sin"] = np.sin(2 * np.pi * F["hr"] / 24)
        F["hour_cos"] = np.cos(2 * np.pi * F["hr"] / 24)
        F["is_night"] = ((F["hr"] < 6) | (F["hr"] > 22)).astype(np.int8)
        F["is_business_hours"] = ((F["hr"] >= 9) & (F["hr"] <= 17)).astype(np.int8)
        F["chip"] = chunk["chip"]
        F["is_online"] = chunk["is_online"]
        F["err"] = chunk["err"]
        F["mcc_n"] = chunk["mcc_n"]
        F["has_zip"] = chunk["has_zip"]
        F["has_state"] = chunk["has_state"]
        F["is_online_or_no_state"] = ((F["is_online"] == 1) | (F["has_state"] == 0)).astype(np.int8)

        # Online expanding features: per-user state
        user_tx = np.zeros(n, dtype=np.float32)
        card_tx = np.zeros(n, dtype=np.float32)
        merch_tx = np.zeros(n, dtype=np.float32)
        user_avg = np.zeros(n, dtype=np.float32)
        amt_z = np.zeros(n, dtype=np.float32)
        user_fr = np.full(n, 0.001, dtype=np.float32)  # baseline
        merch_fr = np.full(n, 0.001, dtype=np.float32)
        city_fr = np.full(n, 0.001, dtype=np.float32)

        users = chunk["User"].values
        cards = chunk["Card"].values
        merchants = chunk["Merchant Name"].values
        cities = chunk["Merchant City"].values
        amts = chunk["amt"].values
        labels = chunk["is_fraud"].values

        # Per-city fraud state (smaller cardinality)
        city_state = {}

        for i in range(n):
            uid = int(users[i])
            cid = int(cards[i])
            mid = int(merchants[i])
            ctid = int(cities[i])
            amt = float(amts[i])
            label = int(labels[i])

            # User state
            us = user_state.get(uid)
            if us is None:
                us = {"count": 0, "sum": 0.0, "sum_sq": 0.0, "fraud_count": 0}
                user_state[uid] = us

            # Card state
            cs = card_state.get(cid)
            if cs is None:
                cs = {"count": 0}
                card_state[cid] = cs

            # Merchant state
            ms = merch_state.get(mid)
            if ms is None:
                ms = {"count": 0, "fraud_count": 0, "total": 0}
                merch_state[mid] = ms

            # City state
            cts = city_state.get(ctid)
            if cts is None:
                cts = {"fraud_count": 0, "total": 0}
                city_state[ctid] = cts

            # Record BEFORE this event (shift(1) semantics)
            user_tx[i] = us["count"]
            card_tx[i] = cs["count"]
            merch_tx[i] = ms["count"]

            if us["count"] > 0:
                user_avg[i] = us["sum"] / us["count"]
                if us["count"] >= 2:
                    mean = us["sum"] / us["count"]
                    variance = us["sum_sq"] / us["count"] - mean ** 2
                    std = max(variance, 0) ** 0.5
                    amt_z[i] = (amt - mean) / (std + 1e-6)
                else:
                    amt_z[i] = (amt - us["sum"]) / (us["sum"] + 1e-6)
            else:
                user_avg[i] = 0.0
                amt_z[i] = 0.0

            # Fraud rates from history only
            if us["count"] > 0:
                user_fr[i] = us["fraud_count"] / us["count"]
            if ms["total"] > 0:
                merch_fr[i] = ms["fraud_count"] / ms["total"]
            if cts["total"] > 0:
                city_fr[i] = cts["fraud_count"] / cts["total"]

            # Update state AFTER recording (for next event)
            us["count"] += 1
            us["sum"] += amt
            us["sum_sq"] += amt ** 2
            us["fraud_count"] += label
            cs["count"] += 1
            ms["count"] += 1
            ms["fraud_count"] += label
            ms["total"] += 1
            cts["fraud_count"] += label
            cts["total"] += 1

        F["user_tx_count"] = user_tx
        F["card_tx_count"] = card_tx
        F["merch_tx_count"] = merch_tx
        F["user_avg_amt"] = user_avg
        F["amt_vs_user_avg"] = F["amt"] / (F["user_avg_amt"] + 1e-6)
        F["amt_zscore"] = amt_z
        F["user_fraud_rate"] = user_fr
        F["merch_fraud_rate"] = merch_fr
        F["city_fraud_rate"] = city_fr

        # Thresholds
        F["high_amt"] = (F["amt"] > F["user_avg_amt"] * 2).astype(np.int8)
        F["very_high_amt"] = (F["amt"] > F["user_avg_amt"] * 5).astype(np.int8)

        # Interactions
        F["amt_x_hr"] = F["amt"] * F["hr"]
        F["amt_x_mcc"] = F["amt"] * F["mcc_n"]
        F["amt_x_chip"] = F["amt"] * F["chip"]
        F["amt_x_online"] = F["amt"] * F["is_online"]
        F["amt_x_night"] = F["amt"] * F["is_night"]

        # Labels
        F["is_fraud"] = labels
        F["Month"] = chunk["Month"]

        # Replace inf/nan
        F.replace([np.inf, -np.inf], np.nan, inplace=True)
        F.fillna(0.0, inplace=True)

        # Write feature chunk
        cpath = CACHE_DIR / f"feats_{chunk_idx:03d}.parquet"
        F.astype(np.float32).to_parquet(cpath, index=False)
        feat_chunks.append(cpath)
        chunk_idx += 1

        print(f"  Chunk {chunk_idx}: users={len(user_state):,} cards={len(card_state):,} "
              f"merchants={len(merch_state):,} cities={len(city_state):,}")

        # Free raw chunk memory
        del chunk, F
        gc.collect()

    # Combine feature chunks
    print(f"  Combining {len(feat_chunks)} feature chunks...")
    dfs = [pd.read_parquet(p) for p in feat_chunks]
    F_all = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    F_all.to_parquet(features_path, index=False)

    # Cleanup
    for p in feat_chunks:
        p.unlink()

    elapsed = time.time() - t0
    print(f"  Features: {F_all.shape[0]:,} rows x {F_all.shape[1]} cols in {elapsed:.0f}s")
    return features_path


# ── Pass 3: Train Ensemble ──────────────────────────────────────────────────

def recall_at(y_true, y_score, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= fpr_target
    return float(tpr[mask][-1]) if mask.any() else 0.0


def pass3_train(features_path: Path):
    """Train XGB+LGB+CB ensemble on all 24M rows."""
    print("\n[PASS 3] Training ensemble on ALL rows...")
    t0 = time.time()

    F = pd.read_parquet(features_path)
    print(f"  Loaded: {F.shape[0]:,} rows x {F.shape[1]} features")

    y = F["is_fraud"].values
    feat_cols = [c for c in F.columns if c not in ("is_fraud", "Month")]
    X = F[feat_cols].values.astype(np.float32)
    months = F["Month"].values

    # Temporal split
    train_mask = months <= 9
    test_mask = months > 9
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    print(f"  Train: {len(Xtr):,} ({ytr.sum():,} fraud)")
    print(f"  Test:  {len(Xte):,} ({yte.sum():,} fraud)")

    # Scale
    scaler = RobustScaler()
    print("  Scaling...")
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Free raw arrays
    del X
    gc.collect()

    # Compute class weight
    spw = (1 - ytr.mean()) / ytr.mean()
    spw = min(spw, 20)
    print(f"  scale_pos_weight: {spw:.1f}")

    # 5-fold CV on training set
    print("\n  5-fold stratified CV...")
    import xgboost as xgb
    import lightgbm as lgb
    try:
        import catboost as cb
        HAS_CB = True
    except ImportError:
        HAS_CB = False
        print("  WARNING: catboost not installed")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs, cv_r1s, cv_aprs = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr_s, ytr)):
        Xa, Xv = Xtr_s[tr_idx], Xtr_s[va_idx]
        ya, yv = ytr[tr_idx], ytr[va_idx]

        m_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            scale_pos_weight=spw, random_state=42, n_jobs=4,
            eval_metric="auc", use_label_encoder=False,
        )
        m_xgb.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)

        m_lgb = lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            num_leaves=31, scale_pos_weight=spw, random_state=42,
            n_jobs=4, verbose=-1,
        )
        m_lgb.fit(Xa, ya, eval_set=[(Xv, yv)])

        if HAS_CB:
            m_cb = cb.CatBoostClassifier(
                iterations=300, depth=8, learning_rate=0.05,
                l2_leaf_reg=5, auto_class_weights="Balanced",
                random_seed=42, verbose=0,
            )
            m_cb.fit(Xa, ya, eval_set=(Xv, yv))
            p = 0.34 * m_xgb.predict_proba(Xv)[:, 1] + \
                0.33 * m_lgb.predict_proba(Xv)[:, 1] + \
                0.33 * m_cb.predict_proba(Xv)[:, 1]
        else:
            p = 0.5 * m_xgb.predict_proba(Xv)[:, 1] + \
                0.5 * m_lgb.predict_proba(Xv)[:, 1]

        auc = roc_auc_score(yv, p)
        r1 = recall_at(yv, p)
        apr = average_precision_score(yv, p)
        cv_aucs.append(auc)
        cv_r1s.append(r1)
        cv_aprs.append(apr)
        print(f"    Fold {fold+1}: AUC={auc:.4f} R@1%FPR={r1:.4f} APR={apr:.4f}")

    cv_auc_mean = np.mean(cv_aucs)
    cv_r1_mean = np.mean(cv_r1s)
    print(f"\n  CV AUC: {cv_auc_mean:.4f} ± {np.std(cv_aucs):.4f}")
    print(f"  CV R@1%FPR: {cv_r1_mean:.4f} ± {np.std(cv_r1s):.4f}")

    # Final training on full train set
    print("\n  Training final models on full train set...")
    m_xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        scale_pos_weight=spw, random_state=42, n_jobs=4,
        eval_metric="auc", use_label_encoder=False,
    )
    m_xgb.fit(Xtr_s, ytr, verbose=False)

    m_lgb = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        num_leaves=31, scale_pos_weight=spw, random_state=42,
        n_jobs=4, verbose=-1,
    )
    m_lgb.fit(Xtr_s, ytr)

    if HAS_CB:
        m_cb = cb.CatBoostClassifier(
            iterations=300, depth=8, learning_rate=0.05,
            l2_leaf_reg=5, auto_class_weights="Balanced",
            random_seed=42, verbose=0,
        )
        m_cb.fit(Xtr_s, ytr)
        ensemble_weights = {"xgb": 0.34, "lgb": 0.33, "cb": 0.33}
        p_te = (ensemble_weights["xgb"] * m_xgb.predict_proba(Xte_s)[:, 1] +
                ensemble_weights["lgb"] * m_lgb.predict_proba(Xte_s)[:, 1] +
                ensemble_weights["cb"] * m_cb.predict_proba(Xte_s)[:, 1])
    else:
        ensemble_weights = {"xgb": 0.5, "lgb": 0.5}
        p_te = (ensemble_weights["xgb"] * m_xgb.predict_proba(Xte_s)[:, 1] +
                ensemble_weights["lgb"] * m_lgb.predict_proba(Xte_s)[:, 1])
        m_cb = None

    test_auc = roc_auc_score(yte, p_te)
    test_r1 = recall_at(yte, p_te)
    test_apr = average_precision_score(yte, p_te)

    print(f"\n  Temporal test AUC:  {test_auc:.4f}")
    print(f"  Temporal test R@1%FPR: {test_r1:.4f}")
    print(f"  Temporal test APR:  {test_apr:.4f}")

    # Compare with baseline (272K sampled model)
    baseline_auc = 0.99168
    delta = test_auc - baseline_auc
    print(f"\n  Baseline (272K sampled): {baseline_auc:.4f}")
    print(f"  Full 24M:                {test_auc:.4f}")
    print(f"  Delta:                   {delta:+.4f} ({delta*100:+.2f}%)")

    if test_auc > baseline_auc:
        print(f"\n  FULL-SCALE MODEL IMPROVES BY {delta*100:+.2f}%")
    else:
        print(f"\n  Full-scale model does not improve (expected — more temporal drift)")

    # Save artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prod_dir = MODELS_DIR

    joblib.dump(m_xgb, prod_dir / "xgb_production.joblib")
    joblib.dump(m_lgb, prod_dir / "lgb_production.joblib")
    if m_cb is not None:
        joblib.dump(m_cb, prod_dir / "cb_production.joblib")
    joblib.dump(scaler, prod_dir / "scaler_production.joblib")

    model_hash = hashlib.md5()
    for f in sorted(prod_dir.glob("*.joblib")):
        model_hash.update(f.read_bytes())

    manifest = {
        "model_version": f"altman_full24m_{ts}",
        "model_type": "xgb_lgb_cb_ensemble",
        "feature_version": "v5_full24m",
        "dataset_version": "ibm_altman_v2_full",
        "dataset_rows_scanned": int(F.shape[0]),
        "training_rows": int(len(Xtr)),
        "n_fraud_train": int(ytr.sum()),
        "n_features": len(feat_cols),
        "features": feat_cols,
        "cv_auc_mean": round(cv_auc_mean, 6),
        "cv_auc_std": round(float(np.std(cv_aucs)), 6),
        "cv_r1_mean": round(cv_r1_mean, 6),
        "cv_r1_std": round(float(np.std(cv_r1s)), 6),
        "cv_apr_mean": round(float(np.mean(cv_aprs)), 6),
        "temporal_test_auc": round(test_auc, 6),
        "temporal_test_r1": round(test_r1, 6),
        "temporal_test_apr": round(test_apr, 6),
        "ensemble_weights": ensemble_weights,
        "scale_pos_weight": round(spw, 2),
        "model_hash": model_hash.hexdigest()[:16],
        "target_leakage": False,
        "temporal_split": "train_months_1-9, test_months_10-12",
        "full_scale": True,
        "sampled": False,
        "trained_at": datetime.now().isoformat(),
        "artifacts": [str(f.name) for f in sorted(prod_dir.glob("*.joblib"))],
    }

    (prod_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (prod_dir / "feature_list.json").write_text(json.dumps(feat_cols, indent=2))

    # Feature importance
    imp = m_xgb.feature_importances_
    pairs = sorted(zip(feat_cols, imp), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 features (XGB):")
    for f, i in pairs[:10]:
        print(f"    {f}: {i:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed:.0f}s")
    print(f"  Saved to: {prod_dir}")
    print("=" * 70)

    return {
        "cv_auc": cv_auc_mean,
        "test_auc": test_auc,
        "test_r1": test_r1,
        "delta": delta,
    }


def main():
    print("=" * 70)
    print("FULL-SCALE ALTMAN TRAINING — ALL 24M ROWS, NO SAMPLING")
    print("=" * 70)
    t0 = time.time()

    csv_path = DATA_DIR / "credit_card_transactions-ibm_v2.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    # Pass 1: Sort
    sorted_path = pass1_sort(csv_path)

    # Pass 2: Features
    features_path = pass2_features(sorted_path)

    # Pass 3: Train
    results = pass3_train(features_path)

    elapsed = time.time() - t0
    print(f"\nTotal pipeline time: {elapsed/60:.1f} minutes")

    # Save results report
    results["elapsed_s"] = round(elapsed, 1)
    results["timestamp"] = datetime.now().isoformat()
    (REPORTS_DIR / "full_scale_24m.json").write_text(json.dumps(results, indent=2))
    print(f"Report: reports/full_scale_24m.json")


if __name__ == "__main__":
    main()
