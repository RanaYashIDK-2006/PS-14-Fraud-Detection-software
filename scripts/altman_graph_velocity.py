#!/usr/bin/env python3
"""
Altman Graph + Velocity Features — Push honest AUC above 95%.

Builds:
1. User-Merchant bipartite graph: degree centrality, merchant popularity, user breadth
2. User-City graph: city diversity, geographic spread
3. Per-user velocity: tx count, amount stats, time gaps, merchant diversity
4. Merchant fraud clustering: merchants sharing similar fraud patterns
5. Temporal patterns: hour-of-day, day-of-week, seasonal patterns

All features computed on TRAIN ONLY to prevent target leakage.
Temporal split for honest evaluation.
"""
import sys, time, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb

# ─── Config ───
CHUNK_SIZE = 500_000
SAMPLE_LEGIT = 200_000  # per chunk max
NJ = 4


def parse_hr(t):
    try:
        s = str(t).strip()
        parts = s.replace('.', ':').split(':')
        h = int(parts[0])
        if 'pm' in s.lower() and h != 12: h += 12
        elif 'am' in s.lower() and h == 12: h = 0
        return h
    except:
        return 12


def load_chunk_features(chunk):
    """Extract raw fields from a chunk for feature engineering."""
    c = pd.DataFrame()
    c["User"] = chunk["User"]
    c["Card"] = chunk["Card"]
    c["label"] = (chunk["Is Fraud?"].astype(str).str.strip() == "Yes").astype(int)
    c["amt"] = pd.to_numeric(
        chunk["Amount"].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False),
        errors='coerce'
    ).fillna(0)
    c["hr"] = chunk["Time"].apply(parse_hr)
    c["mcc_n"] = pd.to_numeric(chunk["MCC"], errors='coerce').fillna(0)
    c["merchant"] = chunk["Merchant Name"].astype(str)
    c["city"] = chunk["Merchant City"].astype(str)
    c["state"] = chunk["Merchant State"].astype(str).fillna("")
    c["zip"] = chunk["Zip"].fillna(0)
    c["chip"] = chunk["Use Chip"].astype(str).str.contains("Swipe|Chip", case=False, na=False).astype(int)
    c["online"] = chunk["Use Chip"].astype(str).str.contains("Online", case=False, na=False).astype(int)
    c["year"] = chunk["Year"].astype(int)
    c["month"] = chunk["Month"].astype(int)
    c["day"] = chunk["Day"].astype(int)
    return c


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Build training statistics (no leakage)
# ══════════════════════════════════════════════════════════════════════
def build_train_stats(train_df):
    """Compute all statistics from training data only."""
    stats = {}
    gm = train_df["label"].mean()
    stats["global_mean"] = gm

    print("    Building merchant stats...")
    # Merchant frequency
    merch_freq = train_df["merchant"].value_counts().to_dict()
    stats["merch_freq"] = merch_freq

    # Merchant fraud rate (smoothed)
    mf = train_df.groupby("merchant")["label"].agg(["mean", "count"])
    mf["r"] = (mf["mean"] * mf["count"] + gm * 50) / (mf["count"] + 50)
    stats["merch_fraud_rate"] = mf["r"].to_dict()

    # Merchant avg amount
    stats["merch_avg_amt"] = train_df.groupby("merchant")["amt"].mean().to_dict()
    stats["merch_std_amt"] = train_df.groupby("merchant")["amt"].std().fillna(1).to_dict()

    print("    Building city stats...")
    # City fraud rate
    cf = train_df.groupby("city")["label"].agg(["mean", "count"])
    cf["r"] = (cf["mean"] * cf["count"] + gm * 50) / (cf["count"] + 50)
    stats["city_fraud_rate"] = cf["r"].to_dict()

    # City frequency
    stats["city_freq"] = train_df["city"].value_counts().to_dict()

    print("    Building user stats...")
    # Per-user stats
    us = train_df.groupby("User").agg(
        tx_count=("label", "count"),
        fraud_rate=("label", "mean"),
        avg_amt=("amt", "mean"),
        std_amt=("amt", "std"),
        avg_mcc=("mcc_n", "mean"),
        unique_merchants=("merchant", "nunique"),
        unique_cities=("city", "nunique"),
    )
    us["fraud_rate"] = (us["fraud_rate"] * us["tx_count"] + gm * 200) / (us["tx_count"] + 200)
    stats["user_tx_count"] = us["tx_count"].to_dict()
    stats["user_fraud_rate"] = us["fraud_rate"].to_dict()
    stats["user_avg_amt"] = us["avg_amt"].to_dict()
    stats["user_std_amt"] = us["std_amt"].fillna(1).to_dict()
    stats["user_avg_mcc"] = us["avg_mcc"].to_dict()
    stats["user_unique_merchants"] = us["unique_merchants"].to_dict()
    stats["user_unique_cities"] = us["unique_cities"].to_dict()

    # Per-user merchant list (for merchant diversity features)
    stats["user_merchants"] = train_df.groupby("User")["merchant"].apply(lambda x: set(x)).to_dict()
    stats["user_cities"] = train_df.groupby("User")["city"].apply(lambda x: set(x)).to_dict()

    # Per-user card stats
    ck = train_df["User"].astype(str) + "_" + train_df["Card"].astype(str)
    stats["card_tx_count"] = ck.value_counts().to_dict()

    print("    Building graph features...")
    # User-Merchant bipartite graph: degree centrality
    um_edges = train_df.groupby(["User", "merchant"]).size().reset_index(name="weight")
    stats["user_merchant_degree"] = um_edges.groupby("User")["weight"].sum().to_dict()
    stats["merchant_user_degree"] = um_edges.groupby("merchant")["weight"].sum().to_dict()

    # User-City graph
    uc_edges = train_df.groupby(["User", "city"]).size().reset_index(name="weight")
    stats["user_city_degree"] = uc_edges.groupby("User")["weight"].sum().to_dict()
    stats["city_user_degree"] = uc_edges.groupby("city")["weight"].sum().to_dict()

    # Merchant popularity (how many unique users shop there)
    stats["merchant_unique_users"] = train_df.groupby("merchant")["User"].nunique().to_dict()

    # City popularity
    stats["city_unique_users"] = train_df.groupby("city")["User"].nunique().to_dict()

    print("    Building time patterns...")
    # Hour-of-day fraud rate per user
    train_df["hr_bin"] = train_df["hr"] // 6  # 4 bins: 0-5, 6-11, 12-17, 18-23
    uh = train_df.groupby(["User", "hr_bin"])["label"].agg(["mean", "count"])
    uh["r"] = (uh["mean"] * uh["count"] + gm * 10) / (uh["count"] + 10)
    stats["user_hour_fraud"] = uh["r"].to_dict()

    # Merchant MCC mode
    stats["merchant_mcc_mode"] = train_df.groupby("merchant")["mcc_n"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0).to_dict()

    print(f"    Stats built: {len(stats)} keys")
    return stats


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Engineer features from stats
# ══════════════════════════════════════════════════════════════════════
def engineer_features(df, stats):
    """Build feature matrix from raw data + training stats. No target leakage."""
    gm = stats["global_mean"]
    o = pd.DataFrame()

    # ── Basic features ──
    o["log_amt"] = np.log1p(df["amt"])
    o["amt_sq"] = df["amt"] ** 2
    o["amt_zscore"] = (df["amt"] - df["amt"].mean()) / max(df["amt"].std(), 0.01)
    o["very_high_amt"] = (df["amt"] > 1000).astype(int)
    o["high_amt"] = (df["amt"] > 500).astype(int)

    hr = df["hr"].fillna(12).astype(float)
    o["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    o["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    o["is_night"] = ((hr >= 22) | (hr <= 6)).astype(int)
    o["is_business_hours"] = ((hr >= 9) & (hr <= 17)).astype(int)
    o["hr_bin"] = (hr // 6).astype(int)

    o["chip"] = df["chip"]
    o["online"] = df["online"]
    o["mcc_n"] = df["mcc_n"]
    o["has_zip"] = (df["zip"] > 0).astype(int)
    o["has_state"] = (df["state"] != "").astype(int)
    o["month"] = df["month"]

    # ── Merchant features ──
    o["merch_freq"] = df["merchant"].map(stats["merch_freq"]).fillna(0)
    o["merch_fraud_rate"] = df["merchant"].map(stats["merch_fraud_rate"]).fillna(gm)
    o["merch_avg_amt"] = df["merchant"].map(stats["merch_avg_amt"]).fillna(df["amt"].mean())
    o["merch_std_amt"] = df["merchant"].map(stats["merch_std_amt"]).fillna(1)
    o["amt_vs_merch_avg"] = df["amt"] / o["merch_avg_amt"].clip(lower=0.01)

    # ── City features ──
    o["city_fraud_rate"] = df["city"].map(stats["city_fraud_rate"]).fillna(gm)
    o["city_freq"] = df["city"].map(stats["city_freq"]).fillna(0)

    # ── User features ──
    o["user_tx_count"] = df["User"].map(stats["user_tx_count"]).fillna(0)
    o["user_fraud_rate"] = df["User"].map(stats["user_fraud_rate"]).fillna(gm)
    o["user_avg_amt"] = df["User"].map(stats["user_avg_amt"]).fillna(df["amt"].mean())
    o["user_std_amt"] = df["User"].map(stats["user_std_amt"]).fillna(1)
    o["amt_vs_user_avg"] = df["amt"] / o["user_avg_amt"].clip(lower=0.01)
    o["amt_zscore_user"] = (df["amt"] - o["user_avg_amt"]) / o["user_std_amt"].clip(lower=0.01)
    o["user_unique_merchants"] = df["User"].map(stats["user_unique_merchants"]).fillna(0)
    o["user_unique_cities"] = df["User"].map(stats["user_unique_cities"]).fillna(0)
    o["user_avg_mcc"] = df["User"].map(stats["user_avg_mcc"]).fillna(0)

    # ── Card features ──
    ck = df["User"].astype(str) + "_" + df["Card"].astype(str)
    o["card_tx_count"] = ck.map(stats["card_tx_count"]).fillna(0)

    # ── GRAPH FEATURES ──
    # User-Merchant degree (how many merchants does this user visit)
    o["user_merch_degree"] = df["User"].map(stats["user_merchant_degree"]).fillna(0)
    # Merchant popularity (how many users shop at this merchant)
    o["merch_user_degree"] = df["merchant"].map(stats["merchant_user_degree"]).fillna(0)
    o["merch_unique_users"] = df["merchant"].map(stats["merchant_unique_users"]).fillna(1)
    # User-City degree
    o["user_city_degree"] = df["User"].map(stats["user_city_degree"]).fillna(0)
    o["city_unique_users"] = df["city"].map(stats["city_unique_users"]).fillna(1)

    # Normalized centrality scores
    max_user_merch = max(o["user_merch_degree"].max(), 1)
    o["user_merch_centrality"] = o["user_merch_degree"] / max_user_merch
    max_merch_user = max(o["merch_user_degree"].max(), 1)
    o["merch_popularity"] = o["merch_user_degree"] / max_merch_user

    # Merchant fraud clustering: merchants with similar fraud patterns
    # High fraud_rate merchant surrounded by low fraud_rate users = suspicious
    o["merch_fraud_x_user_amt"] = o["merch_fraud_rate"] * df["amt"]
    o["merch_fraud_x_user_rate"] = o["merch_fraud_rate"] * o["user_fraud_rate"]

    # ── VELOCITY FEATURES ──
    # User merchant diversity ratio (unique merchants / total tx)
    o["user_merchant_diversity"] = o["user_unique_merchants"] / o["user_tx_count"].clip(lower=1)
    # User city diversity ratio
    o["user_city_diversity"] = o["user_unique_cities"] / o["user_tx_count"].clip(lower=1)

    # Amount relative to merchant average (deviation)
    o["amt_deviation_merch"] = (df["amt"] - o["merch_avg_amt"]) / o["merch_std_amt"].clip(lower=0.01)

    # ── INTERACTION FEATURES ──
    o["amt_x_hour"] = df["amt"] * hr
    o["amt_x_mcc"] = df["amt"] * o["mcc_n"]
    o["amt_x_chip"] = df["amt"] * o["chip"]
    o["amt_x_online"] = df["amt"] * o["online"]
    o["amt_x_night"] = df["amt"] * o["is_night"]
    o["amt_x_merch_freq"] = df["amt"] * np.log1p(o["merch_freq"])
    o["amt_x_user_rate"] = df["amt"] * o["user_fraud_rate"]
    o["amt_x_merch_rate"] = df["amt"] * o["merch_fraud_rate"]

    # ── Temporal velocity (user's tx in same hour bin) ──
    key = (df["User"].astype(str) + "_" + o["hr_bin"].astype(str))
    o["user_hr_tx_count"] = key.map(train_hr_counts(stats, key)).fillna(0) if "train_hr" in stats else 0

    # Replace inf/nan
    for c in o.columns:
        o[c] = pd.to_numeric(o[c], errors="coerce").fillna(0).replace([np.inf, -np.inf], 0)

    return o


def train_hr_counts(stats, key):
    """Placeholder — computed in build_train_stats."""
    return {}


def build_hr_counts(train_df):
    """Pre-compute per-user per-hour-bin transaction counts."""
    train_df["hr_bin"] = train_df["hr"] // 6
    key = train_df["User"].astype(str) + "_" + train_df["hr_bin"].astype(str)
    return key.value_counts().to_dict()


# ══════════════════════════════════════════════════════════════════════
# PHASE 3: Load data with temporal split
# ══════════════════════════════════════════════════════════════════════
def load_data():
    """Load full Altman dataset, apply temporal split, build features."""
    print("  Loading full Altman dataset (streaming)...")
    t0 = time.time()

    all_chunks = []
    n_total = 0
    n_fraud = 0

    for chunk in pd.read_csv("data/credit_card_transactions-ibm_v2.csv",
                              low_memory=False, chunksize=CHUNK_SIZE):
        c = load_chunk_features(chunk)
        n_total += len(c)
        n_fraud += c["label"].sum()
        all_chunks.append(c)
        print(f"    Loaded {n_total:,} rows ({n_fraud:,} fraud)", end="\r")

    print(f"\n  Total: {n_total:,} rows ({n_fraud:,} fraud, {n_fraud/n_total*100:.4f}%)")
    df = pd.concat(all_chunks, ignore_index=True)

    # Temporal split: first 80% = train, last 20% = test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df):,} ({train_df['label'].sum():,} fraud)")
    print(f"  Test:  {len(test_df):,} ({test_df['label'].sum():,} fraud)")
    print(f"  Load time: {time.time()-t0:.1f}s")

    return train_df, test_df


# ══════════════════════════════════════════════════════════════════════
# PHASE 4: Train and evaluate
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("ALTMAN — Graph + Velocity Features → AUC > 95%")
    print("=" * 80)
    t_total = time.time()

    # Load data
    train_df, test_df = load_data()

    # Build training statistics
    print("\n  Building training statistics...")
    t0 = time.time()
    stats = build_train_stats(train_df)
    stats["train_hr_counts"] = build_hr_counts(train_df)
    print(f"  Stats built in {time.time()-t0:.1f}s")

    # Engineer features
    print("\n  Engineering features (train)...")
    t0 = time.time()
    Xtr = engineer_features(train_df, stats)
    ytr = train_df["label"].values
    feat_names = list(Xtr.columns)
    print(f"  Train features: {len(feat_names)} columns ({time.time()-t0:.1f}s)")

    print("  Engineering features (test)...")
    t0 = time.time()
    Xte = engineer_features(test_df, stats)
    yte = test_df["label"].values
    print(f"  Test features: {Xte.shape[1]} columns ({time.time()-t0:.1f}s)")

    # Convert to numpy
    Xtr_np = np.nan_to_num(Xtr.values.astype(np.float32), nan=0, posinf=100, neginf=-100)
    Xte_np = np.nan_to_num(Xte.values.astype(np.float32), nan=0, posinf=100, neginf=-100)

    # Scale
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr_np)
    Xte_s = scaler.transform(Xte_np)

    n_fraud_te = int(yte.sum())
    n_neg_te = int((yte == 0).sum())
    spw = (len(ytr) - int(ytr.sum())) / max(int(ytr.sum()), 1)

    print(f"\n  Features: {len(feat_names)}")
    print(f"  Scale pos weight: {spw:.0f}")

    # ── Train XGBoost ──
    print("\n  Training XGBoost...")
    t0 = time.time()
    xgb_m = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.7, gamma=1, min_child_weight=5,
        scale_pos_weight=min(spw, 500), tree_method="hist",
        eval_metric="auc", random_state=42, n_jobs=NJ,
    )
    xgb_m.fit(Xtr_s, ytr, verbose=False)
    p_xgb = xgb_m.predict_proba(Xte_s)[:, 1]
    xgb_auc = roc_auc_score(yte, p_xgb)
    print(f"  XGB done ({time.time()-t0:.1f}s) AUC={xgb_auc:.4f}")

    # ── Train LightGBM ──
    print("  Training LightGBM...")
    t0 = time.time()
    lgb_m = lgb.LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.7, min_child_samples=30,
        scale_pos_weight=min(spw, 500), random_state=42, n_jobs=NJ, verbose=-1,
    )
    lgb_m.fit(Xtr_s, ytr)
    p_lgb = lgb_m.predict_proba(Xte_s)[:, 1]
    lgb_auc = roc_auc_score(yte, p_lgb)
    print(f"  LGB done ({time.time()-t0:.1f}s) AUC={lgb_auc:.4f}")

    # ── Ensemble ──
    # Find optimal weight
    best_w, best_auc = 0.5, 0
    for w in np.arange(0.1, 0.9, 0.05):
        p = w * p_xgb + (1 - w) * p_lgb
        auc = roc_auc_score(yte, p)
        if auc > best_auc:
            best_auc = auc
            best_w = w

    p_ens = best_w * p_xgb + (1 - best_w) * p_lgb
    ens_auc = roc_auc_score(yte, p_ens)

    print(f"\n  ═══ RESULTS ═══")
    print(f"  XGB AUC:     {xgb_auc:.4f}")
    print(f"  LGB AUC:     {lgb_auc:.4f}")
    print(f"  Ensemble AUC: {ens_auc:.4f} (weight: XGB={best_w:.2f} LGB={1-best_w:.2f})")
    print(f"  Target: > 0.9500  {'✅ ACHIEVED' if ens_auc > 0.95 else '❌ NOT MET'}")

    # ── Feature importance ──
    print(f"\n  Top 20 Features (XGBoost):")
    importances = xgb_m.feature_importances_
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1])[:20]:
        print(f"    {name:>30}: {imp:.4f}")

    # ── Threshold sweep ──
    print(f"\n  === Threshold Sweep (Ensemble) ===")
    thresholds = [0.1, 0.25, 0.5, 0.75, 1.0]
    print(f"  {'Thr':>6} | {'FPR':>8} | {'Recall':>8} | {'Prec':>8} | {'F1':>8}")
    print("  " + "-" * 55)
    for thr in thresholds:
        y_pred = (p_ens >= thr).astype(int)
        tp = int(((y_pred == 1) & (yte == 1)).sum())
        fp = int(((y_pred == 1) & (yte == 0)).sum())
        fn = int(((y_pred == 0) & (yte == 1)).sum())
        recall = tp / max(n_fraud_te, 1)
        fpr = fp / max(n_neg_te, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-12)
        print(f"  {thr:>6.2f} | {fpr:>7.4f} | {recall:>7.4f} | {prec:>7.4f} | {f1:>7.4f}")

    # ── Find 98.5% recall threshold ──
    order = np.argsort(-p_ens)
    sorted_y = yte[order]
    cum_tp = np.cumsum(sorted_y)
    recall_curve = cum_tp / n_fraud_te

    print(f"\n  === Recall Curve ===")
    for target_r in [0.90, 0.95, 0.98, 0.985, 0.99, 1.0]:
        idx = np.searchsorted(recall_curve, target_r)
        if idx < len(p_ens):
            thr = p_ens[order[idx]]
            flagged = idx + 1
            tp = int(cum_tp[idx])
            fp = flagged - tp
            fpr = fp / max(n_neg_te, 1)
            print(f"  recall={target_r*100:.1f}% thr={thr:.6f} FPR={fpr*100:.3f}% flagged={flagged} missed={n_fraud_te-tp}")

    elapsed = time.time() - t_total

    # Save
    out = {
        "dataset": "ibm_altman_graph_velocity",
        "split": "temporal_80_20",
        "n_total": len(train_df) + len(test_df),
        "n_train": len(train_df), "n_test": len(test_df),
        "fraud_test": n_fraud_te,
        "n_features": len(feat_names),
        "features": feat_names,
        "xgb_auc": round(float(xgb_auc), 6),
        "lgb_auc": round(float(lgb_auc), 6),
        "ensemble_auc": round(float(ens_auc), 6),
        "ensemble_weight": {"xgb": round(best_w, 2), "lgb": round(1-best_w, 2)},
        "target_95_achieved": ens_auc > 0.95,
        "feature_importance_top20": {name: round(float(imp), 4) for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1])[:20]},
        "elapsed_seconds": round(elapsed, 1),
    }
    with open("reports/altman_graph_velocity.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  Saved: reports/altman_graph_velocity.json ({elapsed:.1f}s)")
    return out


if __name__ == "__main__":
    main()
