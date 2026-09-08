#!/usr/bin/env python3
"""Cross-Domain Transfer Evaluation — ULB ↔ Altman.

Trains on one domain, tests on the other, using a shared feature space:
- Amount-based features (log_amt, amt_ratio, amt_zscore)
- Time-based features (hour, is_night, is_business_hours)
- Velocity features (user_tx_count, merch_tx_count — where available)
- Transaction-level features (err flag, is_online proxy)

Produces:
  1. Transfer matrix (4 cells: ULB→ULB, ULB→Altman, Altman→ULB, Altman→Altman)
  2. Feature importance comparison across domains
  3. Domain gap analysis (feature distribution shift)
"""
from __future__ import annotations

import gc
import json
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
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# ── Shared Feature Space ────────────────────────────────────────────────────
# Both ULB and Altman map to these 15 features
SHARED_FEATURES = [
    "log_amt",          # log(1 + amount)
    "amt_zscore",       # z-score of amount vs user mean
    "hour",             # hour of day (0-23)
    "hour_sin",         # cyclic encoding
    "hour_cos",         # cyclic encoding
    "is_night",         # 1 if 0-5 or 22-23
    "is_business_hours", # 1 if 9-17
    "is_weekend",       # 1 if Sat/Sun
    "user_tx_count",    # cumulative tx count for user
    "merch_tx_count",   # cumulative tx count for merchant
    "card_tx_count",    # cumulative tx count for card
    "user_avg_amt",     # running average amount
    "amt_vs_user_avg",  # amount / user_avg
    "err_flag",         # 1 if error/auth failure
    "txn_freq_24h",     # transactions in last 24h
]


def load_ulb(max_rows: int = 100_000) -> pd.DataFrame:
    """Load ULB Creditcard and map to shared features."""
    print("  Loading ULB...")
    df = pd.read_csv(DATA_DIR / "creditcard.csv", nrows=max_rows * 3)
    # Keep all fraud, sample legit
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(min(max_rows, len(fraud) * 20), random_state=42)
    df = pd.concat([fraud, legit]).reset_index(drop=True)

    # Map to shared features
    F = pd.DataFrame()
    F["log_amt"] = np.log1p(df["Amount"])
    # Amount z-score per pseudo-user (use Time as proxy for user session)
    F["amt_zscore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-6)
    F["hour"] = (df["Time"] % 86400) / 3600  # seconds → hour
    F["hour_sin"] = np.sin(2 * np.pi * F["hour"] / 24)
    F["hour_cos"] = np.cos(2 * np.pi * F["hour"] / 24)
    F["is_night"] = ((F["hour"] < 6) | (F["hour"] > 22)).astype(int)
    F["is_business_hours"] = ((F["hour"] >= 9) & (F["hour"] <= 17)).astype(int)
    F["is_weekend"] = 0  # ULB doesn't have day-of-week
    F["user_tx_count"] = 0  # ULB doesn't have user IDs
    F["merch_tx_count"] = 0
    F["card_tx_count"] = 0
    F["user_avg_amt"] = df["Amount"].mean()
    F["amt_vs_user_avg"] = df["Amount"] / (F["user_avg_amt"] + 1e-6)
    F["err_flag"] = 0  # ULB doesn't have error flags
    F["txn_freq_24h"] = df.groupby((df["Time"] // 86400)).cumcount()

    F["label"] = df["Class"].values
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"  ULB: {len(F):,} rows ({int(F['label'].sum()):,} fraud)")
    return F


def load_altman_shared(max_rows: int = 200_000) -> pd.DataFrame:
    """Load Altman and map to shared features."""
    print("  Loading Altman...")
    rng = np.random.RandomState(42)
    chunks = []
    for chunk in pd.read_csv(
        DATA_DIR / "credit_card_transactions-ibm_v2.csv",
        usecols=["User", "Card", "Year", "Month", "Day", "Time", "Amount",
                 "Use Chip", "Merchant Name", "Errors?", "Is Fraud?"],
        low_memory=False, chunksize=500_000,
    ):
        fraud_mask = chunk["Is Fraud?"] == "Yes"
        is_legit = ~fraud_mask
        sample = rng.random(len(chunk)) < 0.01
        chunks.append(chunk[fraud_mask | (is_legit & sample)].copy())
        if sum(len(c) for c in chunks) >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True)
    df["amt"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["is_fraud"] = (df["Is Fraud?"] == "Yes").astype(int)
    df["hr"] = df["Time"].str.split(":").str[0].astype(int)
    df["chip"] = (df["Use Chip"] == "Chip Transaction").astype(int)
    df["is_online"] = (df["Use Chip"] == "Online Transaction").astype(int)
    df["err"] = (df["Errors?"].fillna("") != "").astype(int)
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]].assign(
        hour=df["hr"], minute=df["Time"].str.split(":").str[1].astype(int)))
    df = df.sort_values(["User", "datetime"]).reset_index(drop=True)

    F = pd.DataFrame()
    F["log_amt"] = np.log1p(df["amt"])
    # Per-user z-score
    user_mean = df.groupby("User")["amt"].transform("mean")
    user_std = df.groupby("User")["amt"].transform("std").fillna(1.0)
    F["amt_zscore"] = (df["amt"] - user_mean) / (user_std + 1e-6)
    F["hour"] = df["hr"].astype(float)
    F["hour_sin"] = np.sin(2 * np.pi * F["hour"] / 24)
    F["hour_cos"] = np.cos(2 * np.pi * F["hour"] / 24)
    F["is_night"] = ((F["hour"] < 6) | (F["hour"] > 22)).astype(int)
    F["is_business_hours"] = ((F["hour"] >= 9) & (F["hour"] <= 17)).astype(int)
    dow = pd.to_datetime(df[["Year", "Month", "Day"]]).dt.dayofweek
    F["is_weekend"] = (dow >= 5).astype(int)
    F["user_tx_count"] = df.groupby("User").cumcount()
    F["merch_tx_count"] = df.groupby("Merchant Name").cumcount()
    F["card_tx_count"] = df.groupby("Card").cumcount()
    F["user_avg_amt"] = df.groupby("User")["amt"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(df["amt"].mean())
    F["amt_vs_user_avg"] = df["amt"] / (F["user_avg_amt"] + 1e-6)
    F["err_flag"] = df["err"]
    F["txn_freq_24h"] = df.groupby("User").cumcount()  # simplified

    F["label"] = df["is_fraud"].values
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"  Altman: {len(F):,} rows ({int(F['label'].sum()):,} fraud)")
    return F


def train_and_evaluate(Xtr, ytr, Xte, yte, domain_name: str) -> dict:
    """Train XGB+LGB ensemble and evaluate."""
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    spw = min((1 - ytr.mean()) / ytr.mean(), 20)

    # 3-fold CV on train
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_aucs = []
    for tr_i, va_i in skf.split(Xtr_s, ytr):
        Xa, Xv = Xtr_s[tr_i], Xtr_s[va_i]
        ya, yv = ytr[tr_i], ytr[va_i]
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            scale_pos_weight=spw, random_state=42, n_jobs=4,
            eval_metric="auc", use_label_encoder=False)
        m.fit(Xa, ya, verbose=False)
        p = m.predict_proba(Xv)[:, 1]
        cv_aucs.append(roc_auc_score(yv, p))
    cv_auc = np.mean(cv_aucs)

    # Final train
    m_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        scale_pos_weight=spw, random_state=42, n_jobs=4,
        eval_metric="auc", use_label_encoder=False)
    m_xgb.fit(Xtr_s, ytr, verbose=False)

    m_lgb = lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        num_leaves=31, scale_pos_weight=spw, random_state=42, n_jobs=4, verbose=-1)
    m_lgb.fit(Xtr_s, ytr)

    # Evaluate
    p_te = 0.5 * m_xgb.predict_proba(Xte_s)[:, 1] + 0.5 * m_lgb.predict_proba(Xte_s)[:, 1]
    test_auc = roc_auc_score(yte, p_te)
    fpr, tpr, _ = roc_curve(yte, p_te)
    test_r1 = float(tpr[fpr <= 0.01][-1]) if (fpr <= 0.01).any() else 0.0
    test_apr = average_precision_score(yte, p_te)

    # Feature importance (XGB)
    imp = dict(zip(SHARED_FEATURES, m_xgb.feature_importances_.tolist()))

    return {
        "cv_auc": round(cv_auc, 4),
        "test_auc": round(test_auc, 4),
        "test_r1": round(test_r1, 4),
        "test_apr": round(test_apr, 4),
        "train_size": len(ytr),
        "test_size": len(yte),
        "train_fraud": int(ytr.sum()),
        "test_fraud": int(yte.sum()),
        "importance": {k: round(v, 4) for k, v in sorted(imp.items(), key=lambda x: -x[1])},
    }


def main():
    print("=" * 70)
    print("CROSS-DOMAIN TRANSFER MATRIX — ULB ↔ Altman")
    print("=" * 70)
    t0 = time.time()

    # Load both domains in shared feature space
    print("\n[1/5] Loading datasets in shared feature space...")
    ulb = load_ulb(max_rows=50_000)
    alt = load_altman_shared(max_rows=200_000)

    feat_cols = SHARED_FEATURES
    X_ulb = ulb[feat_cols].values.astype(np.float32)
    y_ulb = ulb["label"].values
    X_alt = alt[feat_cols].values.astype(np.float32)
    y_alt = alt["label"].values
    del ulb, alt
    gc.collect()

    # Split: 70% train, 30% test (stratified)
    from sklearn.model_selection import train_test_split
    X_ulb_tr, X_ulb_te, y_ulb_tr, y_ulb_te = train_test_split(
        X_ulb, y_ulb, test_size=0.3, stratify=y_ulb, random_state=42)
    X_alt_tr, X_alt_te, y_alt_tr, y_alt_te = train_test_split(
        X_alt, y_alt, test_size=0.3, stratify=y_alt, random_state=42)

    print(f"\n  ULB: train={len(y_ulb_tr):,} ({int(y_ulb_tr.sum()):,} fraud), "
          f"test={len(y_ulb_te):,} ({int(y_ulb_te.sum()):,} fraud)")
    print(f"  Altman: train={len(y_alt_tr):,} ({int(y_alt_tr.sum()):,} fraud), "
          f"test={len(y_alt_te):,} ({int(y_alt_te.sum()):,} fraud)")

    # Transfer matrix: 4 cells
    matrix = {}

    # Cell 1: ULB → ULB (in-domain baseline)
    print("\n[2/5] ULB → ULB (in-domain)...")
    matrix["ulb_to_ulb"] = train_and_evaluate(X_ulb_tr, y_ulb_tr, X_ulb_te, y_ulb_te, "ULB")

    # Cell 2: ULB → Altman (cross-domain)
    print("[3/5] ULB → Altman (cross-domain)...")
    matrix["ulb_to_altman"] = train_and_evaluate(X_ulb_tr, y_ulb_tr, X_alt_te, y_alt_te, "ULB→Altman")

    # Cell 3: Altman → ULB (cross-domain)
    print("[4/5] Altman → ULB (cross-domain)...")
    matrix["altman_to_ulb"] = train_and_evaluate(X_alt_tr, y_alt_tr, X_ulb_te, y_ulb_te, "Altman→ULB")

    # Cell 4: Altman → Altman (in-domain baseline)
    print("[5/5] Altman → Altman (in-domain)...")
    matrix["altman_to_altman"] = train_and_evaluate(X_alt_tr, y_alt_tr, X_alt_te, y_alt_te, "Altman")

    # Domain gap analysis
    print("\n[Analysis] Feature distribution shift (KS test)...")
    from scipy import stats as sp_stats
    ks_results = {}
    for f in feat_cols:
        ulb_vals = X_ulb[:, feat_cols.index(f)]
        alt_vals = X_alt[:, feat_cols.index(f)]
        ks_stat, pval = sp_stats.ks_2samp(ulb_vals, alt_vals)
        ks_results[f] = {"ks_stat": round(ks_stat, 4), "p_value": round(pval, 4),
                         "shift": "HIGH" if ks_stat > 0.3 else "MEDIUM" if ks_stat > 0.1 else "LOW"}
    del X_ulb, X_alt; gc.collect()

    # Print transfer matrix
    print("\n" + "=" * 70)
    print("TRANSFER MATRIX")
    print("=" * 70)
    print(f"{'':>20} {'Test: ULB':>15} {'Test: Altman':>15}")
    print(f"{'Train: ULB':>20} {matrix['ulb_to_ulb']['test_auc']:>14.4f}  {matrix['ulb_to_altman']['test_auc']:>14.4f}")
    print(f"{'Train: Altman':>20} {matrix['altman_to_ulb']['test_auc']:>14.4f}  {matrix['altman_to_altman']['test_auc']:>14.4f}")

    print(f"\n{'':>20} {'Test: ULB':>15} {'Test: Altman':>15}")
    print(f"{'Recall@1%FPR':>20}")
    print(f"{'Train: ULB':>20} {matrix['ulb_to_ulb']['test_r1']:>14.4f}  {matrix['ulb_to_altman']['test_r1']:>14.4f}")
    print(f"{'Train: Altman':>20} {matrix['altman_to_ulb']['test_r1']:>14.4f}  {matrix['altman_to_altman']['test_r1']:>14.4f}")

    # Generalization gap
    print("\n" + "=" * 70)
    print("GENERALIZATION GAP")
    print("=" * 70)
    ulb_gap_auc = matrix["ulb_to_ulb"]["test_auc"] - matrix["ulb_to_altman"]["test_auc"]
    alt_gap_auc = matrix["altman_to_altman"]["test_auc"] - matrix["altman_to_ulb"]["test_auc"]
    ulb_gap_r1 = matrix["ulb_to_ulb"]["test_r1"] - matrix["ulb_to_altman"]["test_r1"]
    alt_gap_r1 = matrix["altman_to_altman"]["test_r1"] - matrix["altman_to_ulb"]["test_r1"]
    print(f"  ULB → Altman AUC gap:    {ulb_gap_auc:+.4f} ({'GOOD' if ulb_gap_auc < 0.05 else 'LARGE'})")
    print(f"  Altman → ULB AUC gap:    {alt_gap_auc:+.4f} ({'GOOD' if alt_gap_auc < 0.05 else 'LARGE'})")
    print(f"  ULB → Altman R@1% gap:   {ulb_gap_r1:+.4f}")
    print(f"  Altman → ULB R@1% gap:   {alt_gap_r1:+.4f}")

    # Feature importance comparison
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE COMPARISON")
    print("=" * 70)
    print(f"{'Feature':>20} {'ULB model':>12} {'Altman model':>12} {'Shift':>8}")
    for f in feat_cols:
        ulb_imp = matrix["ulb_to_ulb"]["importance"].get(f, 0)
        alt_imp = matrix["altman_to_altman"]["importance"].get(f, 0)
        shift = ks_results.get(f, {}).get("shift", "?")
        print(f"{f:>20} {ulb_imp:>12.4f} {alt_imp:>12.4f} {shift:>8}")

    # Domain gap (feature distributions)
    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTION SHIFT (ULB vs Altman)")
    print("=" * 70)
    print(f"{'Feature':>20} {'KS Stat':>10} {'P-value':>10} {'Shift':>8}")
    for f in feat_cols:
        ks = ks_results[f]
        print(f"{f:>20} {ks['ks_stat']:>10.4f} {ks['p_value']:>10.4f} {ks['shift']:>8}")

    elapsed = time.time() - t0

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": round(elapsed, 1),
        "transfer_matrix": matrix,
        "domain_gap": {
            "ulb_to_altman_auc_gap": round(ulb_gap_auc, 4),
            "altman_to_ulb_auc_gap": round(alt_gap_auc, 4),
            "ulb_to_altman_r1_gap": round(ulb_gap_r1, 4),
            "altman_to_ulb_r1_gap": round(alt_gap_r1, 4),
        },
        "ks_distribution_shift": ks_results,
    }
    (REPORTS / "cross_domain_transfer.json").write_text(json.dumps(report, indent=2))

    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Report: reports/cross_domain_transfer.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
