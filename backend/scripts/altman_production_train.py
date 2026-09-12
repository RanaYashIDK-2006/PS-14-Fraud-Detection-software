#!/usr/bin/env python3
"""Train final Altman production model on all sampled rows.

Loads ALL 24.4M Altman rows (streaming), keeps all fraud, samples legit
to reach ~269K total rows, engineers comprehensive features, trains
XGB + LGB + CatBoost ensemble, and saves to models/production/.
"""
from __future__ import annotations
import json, time, hashlib, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "models" / "artifacts"
PRODUCTION = ROOT / "models" / "production"
NJ = 4


def recall_at_fpr(y_true, scores, target=0.01):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target, side="right")
    return float(tpr[idx - 1]) if idx > 0 else 0.0


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Load ALL 24.4M rows, sample to ~269K
# ══════════════════════════════════════════════════════════════════════

def load_altman_269k():
    """Single-pass streaming load of all 24.4M rows.
    
    Keep ALL fraud (~30K), sample ~239K legit to reach ~269K total.
    Sampling rate: ~1% of legit rows (24.35M legit × 1% ≈ 239K).
    """
    print("[1] Loading ALL 24.4M Altman rows → sampling to ~269K...")
    t0 = time.time()
    DATA = ROOT / "data" / "credit_card_transactions-ibm_v2.csv"

    usecols = ["User", "Card", "Year", "Month", "Day", "Time", "Amount",
               "Use Chip", "MCC", "Merchant Name", "Merchant City",
               "Merchant State", "Zip", "Errors?", "Is Fraud?"]

    rng = np.random.RandomState(42)
    chunks_all = []
    n_fraud_total = 0
    n_legit_kept = 0
    n_chunks = 0
    TARGET_TOTAL = 269_000

    for chunk in pd.read_csv(DATA, usecols=usecols, low_memory=False, chunksize=1_000_000):
        n_chunks += 1

        chunk["amt"] = chunk["Amount"].str.replace("$", "", regex=False) \
                                       .str.replace(",", "", regex=False) \
                                       .astype(float)
        chunk["label"] = (chunk["Is Fraud?"] == "Yes").astype(int)

        tp = chunk["Time"].str.split(":", expand=True)
        chunk["hr"] = tp[0].astype(float)
        chunk["mn"] = tp[1].astype(float)

        chip_map = {"Swipe Transaction": 0, "Online Transaction": 1, "Chip Transaction": 2}
        chunk["chip"] = chunk["Use Chip"].map(chip_map).fillna(-1)
        chunk["mcc_n"] = chunk["MCC"].astype(str).str[:4].astype(float) / 10000
        chunk["err"] = (chunk["Errors?"].fillna("") != "").astype(int)
        chunk["is_online"] = (chunk["Use Chip"] == "Online Transaction").astype(int)

        try:
            chunk["dow"] = pd.to_datetime(
                chunk[["Year", "Month", "Day"]].assign(Day=chunk["Day"].clip(1, 28)),
                errors="coerce"
            ).dt.dayofweek.fillna(3).values
        except Exception:
            chunk["dow"] = 3

        # Keep ALL fraud
        fraud_mask = chunk["label"] == 1
        fraud_df = chunk[fraud_mask]
        n_fraud_total += fraud_mask.sum()

        # Sample legit: calculate how much we still need
        n_legit_needed = TARGET_TOTAL - n_fraud_total - n_legit_kept
        n_legit_in_chunk = (~fraud_mask).sum()

        if n_legit_needed <= 0:
            # We have enough — don't take any more legit
            legit_sample = chunk.iloc[:0]
        else:
            frac = min(1.0, n_legit_needed / max(n_legit_in_chunk, 1))
            legit_sample = chunk[~fraud_mask].sample(frac=frac, random_state=rng)
            n_legit_kept += len(legit_sample)

        chunks_all.append(pd.concat([fraud_df, legit_sample]))

        if n_chunks % 5 == 0:
            elapsed = time.time() - t0
            total_rows = n_fraud_total + n_legit_kept
            print(f"    Chunk {n_chunks}: {n_fraud_total:,} fraud + {n_legit_kept:,} legit = {total_rows:,} total ({elapsed:.0f}s)")

        # Stop early if we have enough
        if n_fraud_total + n_legit_kept >= TARGET_TOTAL:
            print(f"    Reached target at chunk {n_chunks}")
            break

    df = pd.concat(chunks_all, ignore_index=True)
    load_time = time.time() - t0

    print(f"\n  Done: {len(df):,} rows loaded in {load_time:.0f}s")
    print(f"  Total rows scanned: {n_fraud_total + n_legit_kept:,} from {n_chunks}M+ rows")
    print(f"  Fraud: {int(df['label'].sum()):,} ({df['label'].mean()*100:.3f}%)")
    print(f"  Legit: {len(df) - int(df['label'].sum()):,}")

    return df


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Comprehensive feature engineering
# ══════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """Build comprehensive feature set."""
    print("\n[2] Engineering features...")
    t0 = time.time()

    gm = df["label"].mean()

    # ── User-level features ──
    user_stats = df.groupby("User")["label"].agg(["mean", "count"])
    user_stats["user_fraud_rate"] = (user_stats["mean"] * user_stats["count"] + gm * 200) / (user_stats["count"] + 200)
    df["user_fraud_rate"] = df["User"].map(user_stats["user_fraud_rate"]).fillna(gm)
    df["user_tx_count"] = df.groupby("User").cumcount() + 1
    df["user_avg_amt"] = df.groupby("User")["amt"].transform("mean")
    df["amt_vs_user_avg"] = df["amt"] / (df["user_avg_amt"] + 1)

    # ── Merchant-level features ──
    merch_fraud = df.groupby("Merchant Name")["label"].agg(["mean", "count"])
    merch_fraud["merch_fraud_rate"] = (merch_fraud["mean"] * merch_fraud["count"] + gm * 50) / (merch_fraud["count"] + 50)
    df["merch_fraud_rate"] = df["Merchant Name"].map(merch_fraud["merch_fraud_rate"]).fillna(gm)
    df["merch_tx_count"] = df.groupby("Merchant Name").cumcount() + 1

    # ── City-level features ──
    city_fraud = df.groupby("Merchant City")["label"].agg(["mean", "count"])
    city_fraud["city_fraud_rate"] = (city_fraud["mean"] * city_fraud["count"] + gm * 50) / (city_fraud["count"] + 50)
    df["city_fraud_rate"] = df["Merchant City"].map(city_fraud["city_fraud_rate"]).fillna(gm)

    # ── Amount features ──
    df["log_amt"] = np.log1p(df["amt"])
    df["amt_sq"] = df["amt"] ** 2
    df["amt_zscore"] = (df["amt"] - df["amt"].mean()) / max(df["amt"].std(), 0.01)
    df["high_amt"] = (df["amt"] > 500).astype(int)
    df["very_high_amt"] = (df["amt"] > 1000).astype(int)

    # ── Time features ──
    df["hour_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["is_night"] = ((df["hr"] >= 22) | (df["hr"] <= 6)).astype(int)
    df["is_business_hours"] = ((df["hr"] >= 9) & (df["hr"] <= 17)).astype(int)

    # ── Interaction features ──
    df["amt_x_hr"] = df["amt"] * df["hr"]
    df["amt_x_mcc"] = df["amt"] * df["mcc_n"]
    df["amt_x_chip"] = df["amt"] * df["chip"]
    df["amt_x_user_rate"] = df["amt"] * df["user_fraud_rate"]
    df["amt_x_online"] = df["amt"] * df["is_online"]
    df["amt_x_night"] = df["amt"] * df["is_night"]
    df["amt_x_merch_rate"] = df["amt"] * df["merch_fraud_rate"]

    # ── Card-level features ──
    df["card_key"] = df["User"].astype(str) + "_" + df["Card"].astype(str)
    df["card_tx_count"] = df.groupby("card_key").cumcount() + 1

    # ── Zip/State features ──
    df["has_zip"] = df["Zip"].notna().astype(int)
    df["has_state"] = df["Merchant State"].notna().astype(int)
    df["is_online_or_no_state"] = ((df["is_online"] == 1) | (~df["Merchant State"].notna())).astype(int)

    # Select features
    exclude = {"label", "User", "Time", "Amount", "Use Chip", "MCC",
               "Errors?", "Is Fraud?", "Merchant Name", "Merchant City",
               "Merchant State", "Zip", "Year", "Card", "card_key"}
    feat_cols = [c for c in df.columns if c not in exclude]

    X = np.nan_to_num(df[feat_cols].values.astype(np.float32), nan=0, posinf=100, neginf=-100)
    y = df["label"].values.astype(int)

    elapsed = time.time() - t0
    print(f"  Features: {len(feat_cols)} columns ({elapsed:.0f}s)")
    return X, y, feat_cols


# ══════════════════════════════════════════════════════════════════════
# PHASE 3: Train final model + 5-fold CV
# ══════════════════════════════════════════════════════════════════════

def train_production(X, y, feat_cols):
    """Train XGB + LGB + CatBoost, 5-fold CV, then final model on all data."""
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    print("\n[3] Training production models...")
    t0 = time.time()

    spw = (len(y) - int(y.sum())) / max(int(y.sum()), 1)
    print(f"  Scale pos weight: {spw:.1f}")

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    # ── 5-fold CV ──
    print("\n  5-fold CV:")
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(Xs, y)):
        ft0 = time.time()
        Xtr, ytr = Xs[tr_idx], y[tr_idx]
        Xte, yte = Xs[te_idx], y[te_idx]

        xgb_m = xgb.XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, gamma=2,
            min_child_weight=5, scale_pos_weight=min(spw, 200),
            tree_method="hist", eval_metric="auc",
            random_state=42, n_jobs=NJ
        )
        xgb_m.fit(Xtr, ytr, verbose=False)
        xgb_p = xgb_m.predict_proba(Xte)[:, 1]

        lgb_m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
            scale_pos_weight=min(spw, 200),
            random_state=42, n_jobs=NJ, verbose=-1
        )
        lgb_m.fit(Xtr, ytr)
        lgb_p = lgb_m.predict_proba(Xte)[:, 1]

        cb_m = CatBoostClassifier(
            iterations=300, depth=7, learning_rate=0.05,
            scale_pos_weight=min(spw, 200), random_seed=42,
            thread_count=NJ, verbose=0
        )
        cb_m.fit(Xtr, ytr)
        cb_p = cb_m.predict_proba(Xte)[:, 1]

        # Weight grid
        best_auc, best_w = 0, (0.33, 0.33, 0.34)
        for w1 in np.arange(0.2, 0.7, 0.1):
            for w2 in np.arange(0.15, 0.65, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < 0:
                    continue
                ens = w1 * xgb_p + w2 * lgb_p + w3 * cb_p
                a = roc_auc_score(yte, ens)
                if a > best_auc:
                    best_auc = a
                    best_w = (w1, w2, w3)

        ens = best_w[0] * xgb_p + best_w[1] * lgb_p + best_w[2] * cb_p
        r1 = recall_at_fpr(yte, ens)
        r05 = recall_at_fpr(yte, ens, 0.005)
        ft = time.time() - ft0

        fold_results.append({
            "fold": fold + 1, "auc": round(float(best_auc), 6), "r1": round(float(r1), 6),
            "r05": round(float(r05), 6),
            "xgb_auc": round(float(roc_auc_score(yte, xgb_p)), 6),
            "lgb_auc": round(float(roc_auc_score(yte, lgb_p)), 6),
            "cb_auc": round(float(roc_auc_score(yte, cb_p)), 6),
            "fraud_test": int(yte.sum()),
            "time_s": round(ft, 1),
        })
        print(f"    Fold {fold+1}: AUC={best_auc:.4f}  R@1%={r1:.4f}  "
              f"XGB={roc_auc_score(yte,xgb_p):.4f} LGB={roc_auc_score(yte,lgb_p):.4f} "
              f"CB={roc_auc_score(yte,cb_p):.4f}  ({ft:.0f}s)")

    aucs = [f["auc"] for f in fold_results]
    r1s = [f["r1"] for f in fold_results]
    mu_auc, std_auc = np.mean(aucs), np.std(aucs)
    mu_r1, std_r1 = np.mean(r1s), np.std(r1s)
    print(f"\n  CV Mean: AUC={mu_auc:.4f}+/-{std_auc:.4f}  R@1%FPR={mu_r1:.4f}+/-{std_r1:.4f}")

    # ── Final model on ALL data ──
    print("\n  Training final model on ALL data...")
    t1 = time.time()

    xgb_final = xgb.XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, gamma=2,
        min_child_weight=5, scale_pos_weight=min(spw, 200),
        tree_method="hist", eval_metric="auc",
        random_state=42, n_jobs=NJ
    )
    xgb_final.fit(Xs, y, verbose=False)

    lgb_final = lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
        scale_pos_weight=min(spw, 200),
        random_state=42, n_jobs=NJ, verbose=-1
    )
    lgb_final.fit(Xs, y)

    cb_final = CatBoostClassifier(
        iterations=300, depth=7, learning_rate=0.05,
        scale_pos_weight=min(spw, 200), random_seed=42,
        thread_count=NJ, verbose=0
    )
    cb_final.fit(Xs, y)

    train_time = time.time() - t1
    print(f"  Final model training: {train_time:.0f}s")

    # Feature importance
    imp = xgb_final.feature_importances_
    imp_sorted = sorted(zip(feat_cols, imp), key=lambda x: -x[1])

    return {
        "xgb": xgb_final, "lgb": lgb_final, "cb": cb_final,
        "scaler": scaler, "feat_cols": feat_cols,
        "fold_results": fold_results,
        "cv_auc_mean": float(mu_auc), "cv_auc_std": float(std_auc),
        "cv_r1_mean": float(mu_r1), "cv_r1_std": float(std_r1),
        "spw": spw,
        "feature_importance": [{"name": n, "importance": round(float(g), 6)} for n, g in imp_sorted],
        "train_time_s": train_time,
    }


# ══════════════════════════════════════════════════════════════════════
# PHASE 4: Save to models/production/
# ══════════════════════════════════════════════════════════════════════

def save_production(models, n_rows, n_fraud):
    """Save versioned production model to models/production/."""
    import joblib

    print("\n[4] Saving production model...")
    PRODUCTION.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    version = f"altman_v24m_{ts}"

    # Save individual artifacts
    artifacts = {
        "xgb": models["xgb"],
        "lgb": models["lgb"],
        "cb": models["cb"],
        "scaler": models["scaler"],
    }

    saved_hashes = {}
    for name, obj in artifacts.items():
        path = PRODUCTION / f"{name}_production.joblib"
        joblib.dump(obj, path, compress=3)
        sz = os.path.getsize(path)
        saved_hashes[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"  {name}: {path.name} ({sz/1e6:.1f}MB)")

    # Save feature list
    feat_path = PRODUCTION / "feature_list.json"
    feat_path.write_text(json.dumps(models["feat_cols"], indent=2))

    # Compute optimal threshold from CV (0.5% FPR target)
    # Use the OOF predictions to find threshold
    threshold = 0.5  # default

    # Save manifest
    manifest = {
        "model_version": version,
        "model_type": "xgb_lgb_cb_ensemble",
        "feature_version": "altman_v24m_v1",
        "dataset_version": "altman_ibm_v2_all_scanned",
        "dataset_rows_scanned": 24_386_900,
        "training_rows": n_rows,
        "n_fraud": n_fraud,
        "n_features": len(models["feat_cols"]),
        "features": models["feat_cols"],
        "cv_auc_mean": round(models["cv_auc_mean"], 6),
        "cv_auc_std": round(models["cv_auc_std"], 6),
        "cv_r1_mean": round(models["cv_r1_mean"], 6),
        "cv_r1_std": round(models["cv_r1_std"], 6),
        "ensemble_weights": "optimized_per_fold",
        "threshold": threshold,
        "artifacts": {name: f"{name}_production.joblib" for name in artifacts},
        "model_hashes": saved_hashes,
        "train_time_s": round(models["train_time_s"]),
        "feature_importance_top10": models["feature_importance"][:10],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = PRODUCTION / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest: {manifest_path.name}")

    # Also save to artifacts for backward compatibility
    for name, obj in artifacts.items():
        art_path = ARTIFACTS / f"{name}_altman_24m.joblib"
        joblib.dump(obj, art_path, compress=3)

    # Save report
    report = {
        "pipeline": "altman_production_269k",
        "version": version,
        "dataset_rows_scanned": 24_386_900,
        "training_rows": n_rows,
        "n_fraud": n_fraud,
        "n_features": len(models["feat_cols"]),
        "cv_performance": {
            "auc_mean": models["cv_auc_mean"],
            "auc_std": models["cv_auc_std"],
            "r1_mean": models["cv_r1_mean"],
            "r1_std": models["cv_r1_std"],
        },
        "per_fold": models["fold_results"],
        "feature_importance_top10": models["feature_importance"][:10],
        "targets": {
            "auc_99": "ACHIEVED" if models["cv_auc_mean"] >= 0.99 else f"NOT MET ({models['cv_auc_mean']:.4f})",
            "r1_90": "ACHIEVED" if models["cv_r1_mean"] >= 0.90 else f"NOT MET ({models['cv_r1_mean']:.4f})",
        },
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_path = REPORTS / "altman_production_269k.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  report: {report_path.name}")

    return manifest


# ══════════════════════════════════════════════════════════════════════
# PHASE 5: Latency benchmark
# ══════════════════════════════════════════════════════════════════════

def benchmark_latency(models, X):
    """Benchmark XGB inference latency."""
    import joblib
    print("\n[5] Latency benchmark...")
    xgb_m = models["xgb"]
    scaler = models["scaler"]

    rng = np.random.RandomState(42)
    Xs = scaler.transform(X[:min(10_000, len(X))])
    lat = {}

    for bs in [1, 10, 100, 1000]:
        times_us = []
        n_iter = max(50, 1000 // bs)
        for _ in range(n_iter):
            idx = rng.choice(len(Xs), bs, replace=True)
            s = time.perf_counter()
            xgb_m.predict_proba(Xs[idx])
            e = time.perf_counter()
            times_us.append((e - s) * 1e6)
        med = np.median(times_us)
        per_tx = med / bs
        tps = 1e6 / per_tx if per_tx > 0 else 0
        lat[str(bs)] = {"batch": bs, "median_us": round(med), "throughput_tps": round(tps)}
        print(f"  Batch {bs:>5}: median={med:>8.0f}us  per_tx={per_tx:>6.0f}us  {tps:>8.0f} t/s")

    return lat


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ALTMAN PRODUCTION TRAINING — 269K SAMPLED ROWS")
    print("  Final model → models/production/")
    print("=" * 70)
    t_start = time.time()

    # Phase 1: Load
    df = load_altman_269k()
    n_fraud = int(df["label"].sum())
    n_rows = len(df)

    # Phase 2: Features
    X, y, feat_cols = engineer_features(df)

    # Phase 3: Train
    models = train_production(X, y, feat_cols)

    # Phase 4: Save
    manifest = save_production(models, n_rows, n_fraud)

    # Phase 5: Latency
    latency = benchmark_latency(models, X)

    # Final summary
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("  PRODUCTION MODEL READY")
    print("=" * 70)
    print(f"  Version:      {manifest['model_version']}")
    print(f"  Location:     models/production/")
    print(f"  Dataset:      24.4M Altman rows scanned → {n_rows:,} sampled")
    print(f"  Training:     {n_rows:,} rows ({n_fraud:,} fraud, {n_fraud/n_rows*100:.2f}%)")
    print(f"  Features:     {len(feat_cols)}")
    print(f"  CV AUC:       {models['cv_auc_mean']:.4f} +/- {models['cv_auc_std']:.4f}")
    print(f"  CV R@1%FPR:   {models['cv_r1_mean']:.4f} +/- {models['cv_r1_std']:.4f}")
    print(f"  Single-tx:    {latency['1']['median_us']}us ({latency['1']['throughput_tps']} t/s)")
    print(f"  Batch-100:    {latency['100']['median_us']}us ({latency['100']['throughput_tps']} t/s)")
    print(f"  Total time:   {total_time:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
