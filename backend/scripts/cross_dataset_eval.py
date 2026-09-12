#!/usr/bin/env python3
"""Cross-Dataset Evaluation with Shared Semantic Feature Space.

The three datasets have completely different column names:
  - ULB: V1-V28, Amount, Time
  - Altman: amt, hr, merchant_id, chip, mcc, ...
  - PaySim: amount, type, oldbalanceOrg, ...

Strategy: map each dataset to a SHARED SEMANTIC FEATURE SPACE where
features represent the same concept (amount_log, hour_sin, etc.),
enabling cross-domain transfer on a common representation.

Then also test in-domain performance with domain-specific features.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
REPORTS = ROOT / "reports"
NJ = 4


def recall_at_fpr(y_true, scores, target=0.01):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target, side="right")
    return float(tpr[idx - 1]) if idx > 0 else 0.0


def quick_cv(X, y, name="", n_splits=3):
    """Quick 3-fold CV evaluation."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, r1s, r05s = [], [], []
    for tr, te in skf.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        np_ = int(ytr.sum())
        m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, gamma=2,
                          min_child_weight=5,
                          scale_pos_weight=min(np_/max(len(ytr)-np_,1)*50, 200),
                          random_state=42, n_jobs=NJ, eval_metric="auc")
        m.fit(Xtr, ytr, verbose=False)
        p = m.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, p))
        r1s.append(recall_at_fpr(yte, p, 0.01))
        r05s.append(recall_at_fpr(yte, p, 0.005))
    return {"auc": round(np.mean(aucs), 6), "r1": round(np.mean(r1s), 6),
            "r05": round(np.mean(r05s), 6), "n_features": X.shape[1]}


def train_test(train_X, train_y, test_X, test_y):
    """Train on train, evaluate on test (single split)."""
    train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n_pos = int(train_y.sum())
    if n_pos < 10:
        return {"auc": 0, "r1": 0, "r05": 0, "error": "too_few_pos"}
    m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, gamma=2,
                      min_child_weight=5,
                      scale_pos_weight=min(n_pos/max(len(train_y)-n_pos,1)*50, 200),
                      random_state=42, n_jobs=NJ, eval_metric="auc")
    m.fit(train_X, train_y, verbose=False)
    p = m.predict_proba(test_X)[:, 1]
    return {"auc": round(roc_auc_score(test_y, p), 6),
            "r1": round(recall_at_fpr(test_y, p, 0.01), 6),
            "r05": round(recall_at_fpr(test_y, p, 0.005), 6),
            "n_features": train_X.shape[1]}


# ── Shared Feature Space ──

SHARED_FEATURES = [
    "amount_log", "amount_zscore", "hour_sin", "hour_cos",
    "is_night", "is_weekend", "amount_x_hour", "amount_sq",
    "high_amount", "amount_bucket",
]


def build_ulb_shared(df):
    """Map ULB to shared feature space."""
    out = pd.DataFrame()
    out["amount_log"] = np.log1p(df["Amount"].clip(upper=1e6))
    out["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / max(df["Amount"].std(), 0.01)
    # Map Time (seconds since first txn) to hour of day
    hour = (df["Time"] % 86400) / 3600.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_night"] = ((hour >= 22) | (hour <= 6)).astype(float)
    out["is_weekend"] = 0.0  # unknown from Time alone
    out["amount_x_hour"] = df["Amount"] * hour / 24.0
    out["amount_sq"] = df["Amount"] ** 2
    out["high_amount"] = (df["Amount"] > 100).astype(float)
    out["amount_bucket"] = pd.cut(df["Amount"], bins=[0, 10, 50, 200, 1000, 1e9], labels=False).fillna(0).astype(float)
    return out


def build_ulb_shared_with_velocity(df):
    """Map ULB to shared space + PCA-derived velocity."""
    out = build_ulb_shared(df)
    # PCA deviations as anomaly features
    for v in ["V14", "V17", "V12", "V10"]:
        rm = df[v].rolling(200, min_periods=1).mean()
        out[f"{v}_dev"] = df[v] - rm
    devs = [c for c in out.columns if c.endswith("_dev")]
    out["pca_anomaly_sum"] = out[devs].abs().sum(axis=1)
    out["amt_x_V14"] = df["Amount"] * df["V14"]
    out["amt_x_V17"] = df["Amount"] * df["V17"]
    return out


def build_altman_shared(df):
    """Map Altman to shared feature space."""
    out = pd.DataFrame()
    out["amount_log"] = np.log1p(df["amt"].clip(upper=1e6))
    out["amount_zscore"] = (df["amt"] - df["amt"].mean()) / max(df["amt"].std(), 0.01)
    out["hour_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    out["is_night"] = ((df["hr"] >= 22) | (df["hr"] <= 6)).astype(float)
    out["is_weekend"] = 0.0  # unknown without day-of-week
    out["amount_x_hour"] = df["amt"] * df["hr"] / 24.0
    out["amount_sq"] = df["amt"] ** 2
    out["high_amount"] = (df["amt"] > 100).astype(float)
    out["amount_bucket"] = pd.cut(df["amt"], bins=[0, 10, 50, 200, 1000, 1e9], labels=False).fillna(0).astype(float)
    return out


def build_altman_shared_with_velocity(df):
    """Map Altman to shared space + velocity."""
    out = build_altman_shared(df)
    out["merchant_tx_count"] = df.groupby("merchant_id").cumcount().values / 1000.0
    out["city_tx_count"] = df.groupby("city_id").cumcount().values / 1000.0
    out["is_online"] = df["is_online"].astype(float)
    out["chip_type"] = df["chip"].astype(float)
    out["night_deviation"] = df["night_tx"].rolling(100, min_periods=1).mean().fillna(0)
    return out


def build_paysim_shared(df):
    """Map PaySim to shared feature space."""
    out = pd.DataFrame()
    out["amount_log"] = np.log1p(df["amount"].clip(upper=1e6))
    out["amount_zscore"] = (df["amount"] - df["amount"].mean()) / max(df["amount"].std(), 0.01)
    # PaySim has no timestamps — use synthetic uniform hour
    rng = np.random.RandomState(42)
    out["hour_sin"] = rng.uniform(-1, 1, len(df))
    out["hour_cos"] = rng.uniform(-1, 1, len(df))
    out["is_night"] = rng.uniform(0, 1, len(df))
    out["is_weekend"] = rng.uniform(0, 1, len(df))
    out["amount_x_hour"] = out["amount_log"] * out["hour_sin"]
    out["amount_sq"] = df["amount"] ** 2
    out["high_amount"] = (df["amount"] > 100).astype(float)
    out["amount_bucket"] = pd.cut(df["amount"], bins=[0, 10, 50, 200, 1000, 1e9], labels=False).fillna(0).astype(float)
    return out


def build_paysim_shared_with_velocity(df):
    """Map PaySim to shared space + velocity."""
    out = build_paysim_shared(df)
    out["is_cashout"] = (df["type"] == "CASH_OUT").astype(float)
    out["is_transfer"] = (df["type"] == "TRANSFER").astype(float)
    out["orig_wiped"] = (df["newbalanceOrig"] == 0).astype(float)
    out["drain_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    out["balance_wipe_score"] = out["orig_wiped"] * out["is_transfer"]
    return out


# ── Domain-Specific Feature Spaces ──

def build_ulb_domain(df):
    """ULB domain-specific: all PCA + Amount + velocity."""
    out = df[[f"V{i}" for i in range(1, 29)] + ["Amount"]].copy()
    for v in ["V14", "V17", "V12", "V10", "V4", "V11"]:
        rm = df[v].rolling(200, min_periods=1).mean()
        out[f"{v}_dev"] = df[v] - rm
    devs = [c for c in out.columns if c.endswith("_dev")]
    out["pca_anomaly_sum"] = out[devs].abs().sum(axis=1)
    out["amt_x_V14"] = df["Amount"] * df["V14"]
    return out


def build_altman_domain(df):
    """Altman domain-specific: merchant + card + velocity."""
    return df[["amt", "hr", "mn", "mcc_n", "chip", "err", "merchant_id", "city_id",
               "is_online", "day_decimal", "amt_log", "amt_x_hr", "amt_x_mcc",
               "amt_x_chip", "amt_sq", "night_tx", "Month", "Year", "Card"]].copy()


def build_paysim_domain(df):
    """PaySim domain-specific: balance + type features."""
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    out = pd.concat([df[["amount", "oldbalanceOrg", "newbalanceOrig",
                          "oldbalanceDest", "newbalanceDest"]], type_dummies], axis=1)
    out["bal_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    out["bal_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    out["amt_ratio_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    out["orig_wiped"] = (df["newbalanceOrig"] == 0).astype(int)
    out["dest_zero"] = (df["oldbalanceDest"] == 0).astype(int)
    out["flow_asym"] = abs(out["bal_diff_orig"] + out["bal_diff_dest"])
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  CROSS-DATASET EVALUATION")
    print("  Shared Semantic Feature Space + Domain-Specific + Velocity Transfer")
    print("=" * 80)
    t_total = time.time()

    # ── Load and build ──
    print("\n[1] Loading datasets...")
    t0 = time.time()

    ulb = pd.read_csv(ROOT / "data" / "creditcard.csv")
    print(f"  ULB: {len(ulb):,} rows, {int(ulb.Class.sum())} fraud")
    ulb_y = ulb["Class"].values.astype(int)

    # Load Altman (sampled)
    rng = np.random.RandomState(42)
    rows_all = []
    ci = 0
    for chunk in pd.read_csv(ROOT / "data" / "credit_card_transactions-ibm_v2.csv",
        usecols=["User", "Month", "Day", "Time", "Amount", "Use Chip", "MCC", "Errors?",
                 "Is Fraud?", "Merchant Name", "Merchant City", "Merchant State", "Zip", "Year", "Card"],
        low_memory=False, chunksize=1_000_000):
        fm = chunk["Is Fraud?"] == "Yes"
        rows_all.append(pd.concat([chunk[fm], chunk[~fm].sample(frac=0.01, random_state=rng)]))
        ci += 1
        if ci >= 16: break
    alt_raw = pd.concat(rows_all, ignore_index=True)
    chip_map = {"Swipe Transaction": 0, "Online Transaction": 1, "Chip Transaction": 2}
    alt_raw["chip"] = alt_raw["Use Chip"].map(chip_map).fillna(-1)
    alt_raw["mcc_n"] = pd.to_numeric(alt_raw["MCC"], errors="coerce").fillna(0) / 10000
    alt_raw["err"] = (alt_raw["Errors?"].fillna("") != "").astype(int)
    tp = alt_raw["Time"].str.split(":", expand=True)
    alt_raw["hr"] = pd.to_numeric(tp[0], errors="coerce").fillna(12)
    alt_raw["mn"] = pd.to_numeric(tp[1], errors="coerce").fillna(0)
    alt_raw["merchant_id"] = alt_raw["Merchant Name"].astype("category").cat.codes
    alt_raw["city_id"] = alt_raw["Merchant City"].astype("category").cat.codes
    alt_raw["is_online"] = (alt_raw["Merchant City"] == "ONLINE").astype(int)
    alt_raw["day_decimal"] = alt_raw["Day"] + alt_raw["hr"] / 24.0 + alt_raw["mn"] / 1440.0
    alt_raw["amt"] = pd.to_numeric(alt_raw["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce").fillna(0)
    alt_raw["amt_log"] = np.log1p(alt_raw["amt"].clip(upper=1e9))
    alt_raw["amt_x_hr"] = alt_raw["amt"] * alt_raw["hr"]
    alt_raw["amt_x_mcc"] = alt_raw["amt"] * alt_raw["mcc_n"]
    alt_raw["amt_x_chip"] = alt_raw["amt"] * alt_raw["chip"]
    alt_raw["amt_sq"] = alt_raw["amt"] ** 2
    alt_raw["night_tx"] = ((alt_raw["hr"] >= 22) | (alt_raw["hr"] <= 6)).astype(int)
    alt_raw["label"] = (alt_raw["Is Fraud?"] == "Yes").astype(int)
    alt_y = alt_raw["label"].values.astype(int)
    print(f"  Altman: {len(alt_raw):,} rows, {int(alt_raw.label.sum())} fraud")

    ps = pd.read_csv(ROOT / "data" / "paysim_1m.csv")
    ps["label"] = ps["isFraud"]
    ps_y = ps["label"].values.astype(int)
    print(f"  PaySim: {len(ps):,} rows, {int(ps.label.sum())} fraud")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Build feature matrices ──
    print("\n[2] Building feature matrices...")
    # Shared space (no velocity)
    ulb_shared = build_ulb_shared(ulb).values
    alt_shared = build_altman_shared(alt_raw).values
    ps_shared = build_paysim_shared(ps).values

    # Shared space + velocity
    ulb_shared_vel = build_ulb_shared_with_velocity(ulb).values
    alt_shared_vel = build_altman_shared_with_velocity(alt_raw).values
    ps_shared_vel = build_paysim_shared_with_velocity(ps).values

    # Domain-specific
    ulb_domain = build_ulb_domain(ulb).values
    alt_domain = build_altman_domain(alt_raw).values
    ps_domain = build_paysim_domain(ps).values

    # Standardize shared features
    from sklearn.preprocessing import StandardScaler
    for name, arr in [("ulb_shared", ulb_shared), ("alt_shared", alt_shared), ("ps_shared", ps_shared),
                       ("ulb_shared_vel", ulb_shared_vel), ("alt_shared_vel", alt_shared_vel),
                       ("ps_shared_vel", ps_shared_vel)]:
        scaler = StandardScaler()
        globals()[name] = scaler.fit_transform(arr)

    print(f"  Shared features: {ulb_shared.shape[1]}")
    print(f"  Shared+velocity: {ulb_shared_vel.shape[1]}")
    print(f"  ULB domain: {ulb_domain.shape[1]}")
    print(f"  Altman domain: {alt_domain.shape[1]}")
    print(f"  PaySim domain: {ps_domain.shape[1]}")

    # ── Experiments ──
    print("\n[3] Running experiments...")

    results = {"experiments": [], "in_domain": {}, "transfer_matrix": {}}

    # A. In-domain baselines
    print("\n  === IN-DOMAIN BASELINES ===")
    for name, X, y in [("ULB", ulb_domain, ulb_y), ("Altman", alt_domain, alt_y), ("PaySim", ps_domain, ps_y)]:
        r = quick_cv(X, y, name)
        results["in_domain"][name] = r
        print(f"  {name:>8} (domain): AUC={r['auc']:.4f}  R@1%={r['r1']:.4f}  ({r['n_features']} feats)")

    for name, X, y in [("ULB", ulb_shared_vel, ulb_y), ("Altman", alt_shared_vel, alt_y), ("PaySim", ps_shared_vel, ps_y)]:
        r = quick_cv(X, y, name)
        print(f"  {name:>8} (shared+vel): AUC={r['auc']:.4f}  R@1%={r['r1']:.4f}  ({r['n_features']} feats)")

    # B. Cross-domain transfer on SHARED space
    print("\n  === CROSS-DOMAIN TRANSFER (shared features) ===")
    transfer_pairs = [("ULB", "Altman"), ("ULB", "PaySim"), ("Altman", "ULB"),
                      ("Altman", "PaySim"), ("PaySim", "ULB"), ("PaySim", "Altman")]

    shared_data = {"ULB": ulb_shared, "Altman": alt_shared, "PaySim": ps_shared}
    shared_vel_data = {"ULB": ulb_shared_vel, "Altman": alt_shared_vel, "PaySim": ps_shared_vel}
    domain_data = {"ULB": ulb_domain, "Altman": alt_domain, "PaySim": ps_domain}
    labels = {"ULB": ulb_y, "Altman": alt_y, "PaySim": ps_y}

    # Train/test split per dataset (80/20 stratified)
    splits = {}
    for name in ["ULB", "Altman", "PaySim"]:
        y = labels[name]
        idx = np.arange(len(y))
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
        splits[name] = (tr, te)

    for mode, data_dict in [("shared", shared_data), ("shared+vel", shared_vel_data), ("domain", domain_data)]:
        print(f"\n  --- Mode: {mode} ---")
        for src, tgt in transfer_pairs:
            tr_idx = splits[src][0]  # train on source's training split
            train_X = data_dict[src][tr_idx]
            train_y = labels[src][tr_idx]
            test_X = data_dict[tgt]  # test on ALL of target
            test_y = labels[tgt]

            # Filter to common features
            n_feat = min(train_X.shape[1], test_X.shape[1])
            train_X_f = train_X[:, :n_feat]
            test_X_f = test_X[:, :n_feat]

            r = train_test(train_X_f, train_y, test_X_f, test_y)
            key = f"{src}->{tgt}"
            if key not in results["transfer_matrix"]:
                results["transfer_matrix"][key] = {}
            results["transfer_matrix"][key][mode] = r
            print(f"    {src:>8} -> {tgt:<8}: AUC={r.get('auc',0):.4f}  R@1%={r.get('r1',0):.4f}")

    # C. Calibrated transfer
    print("\n  === CALIBRATED TRANSFER (500-sample isotonic on target) ===")
    from sklearn.isotonic import IsotonicRegression
    for src, tgt in transfer_pairs:
        tr_idx, te_idx = splits[src]
        train_X = shared_data[src][tr_idx]
        train_y = labels[src][tr_idx]

        te_all = np.arange(len(labels[tgt]))
        cal_n = min(500, len(te_all) // 5)
        rng_cal = np.random.RandomState(42)
        cal_idx = rng_cal.choice(te_all, cal_n, replace=False)
        eval_idx = np.setdiff1d(te_all, cal_idx)

        n_feat = min(train_X.shape[1], shared_data[tgt].shape[1])
        train_X_f = train_X[:, :n_feat]

        # Fit source model
        n_pos = int(train_y.sum())
        m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, gamma=2, min_child_weight=5,
                          scale_pos_weight=min(n_pos/max(len(train_y)-n_pos,1)*50, 200),
                          random_state=42, n_jobs=NJ, eval_metric="auc")
        m.fit(np.nan_to_num(train_X_f).astype(np.float32), train_y, verbose=False)

        # Predict on target calibration + eval
        all_preds = m.predict_proba(np.nan_to_num(shared_data[tgt][:, :n_feat]).astype(np.float32))[:, 1]

        # Calibrate on calibration set
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(all_preds[cal_idx], labels[tgt][cal_idx])
        calibrated = iso.transform(all_preds[eval_idx])

        a = roc_auc_score(labels[tgt][eval_idx], calibrated)
        r1 = recall_at_fpr(labels[tgt][eval_idx], calibrated, 0.01)

        # Also evaluate uncalibrated on same eval set
        raw = all_preds[eval_idx]
        a_raw = roc_auc_score(labels[tgt][eval_idx], raw)
        r1_raw = recall_at_fpr(labels[tgt][eval_idx], raw, 0.01)

        print(f"    {src:>8} -> {tgt:<8}: raw AUC={a_raw:.4f} R@1%={r1_raw:.4f} -> cal AUC={a:.4f} R@1%={r1:.4f} (delta={a-a_raw:+.4f})")

    # ── Final Summary Matrix ──
    print("\n" + "=" * 80)
    print("  DOMAIN GENERALIZATION MATRIX")
    print("=" * 80)
    print(f"\n  {'Source':>8} -> {'Target':<8}  {'Shared':>8}  {'S+Vel':>8}  {'Domain':>8}  {'Calib':>8}  {'In-Dom':>8}")
    print(f"  {'-'*8}    {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for src, tgt in transfer_pairs:
        key = f"{src}->{tgt}"
        tm = results["transfer_matrix"].get(key, {})
        shared_auc = tm.get("shared", {}).get("auc", 0)
        sv_auc = tm.get("shared+vel", {}).get("auc", 0)
        dom_auc = tm.get("domain", {}).get("auc", 0)
        # Get calibrated AUC from the last experiment output
        print(f"  {src:>8} -> {tgt:<8}  {shared_auc:>8.4f}  {sv_auc:>8.4f}  {dom_auc:>8.4f}  {'see above':>8}  {'---':>8}")

    print(f"\n  In-domain baselines:")
    for name in ["ULB", "Altman", "PaySim"]:
        r = results["in_domain"][name]
        print(f"    {name:>8}: AUC={r['auc']:.4f}  R@1%={r['r1']:.4f}")

    # ── Insights ──
    print("\n" + "=" * 80)
    print("  KEY FINDINGS")
    print("=" * 80)

    # Find best/worst transfer
    all_transfer = []
    for key, modes in results["transfer_matrix"].items():
        for mode, r in modes.items():
            if r.get("auc", 0) > 0:
                all_transfer.append({**r, "pair": key, "mode": mode})
    if all_transfer:
        best = max(all_transfer, key=lambda x: x["auc"])
        worst = min(all_transfer, key=lambda x: x["auc"])
        print(f"\n  Best transfer:  {best['pair']} ({best['mode']}) AUC={best['auc']:.4f}")
        print(f"  Worst transfer: {worst['pair']} ({worst['mode']}) AUC={worst['auc']:.4f}")

    # Velocity impact
    print("\n  Velocity impact on transfer (shared vs shared+vel):")
    for src, tgt in transfer_pairs:
        key = f"{src}->{tgt}"
        tm = results["transfer_matrix"].get(key, {})
        s = tm.get("shared", {}).get("auc", 0)
        v = tm.get("shared+vel", {}).get("auc", 0)
        if s > 0:
            print(f"    {src:>8} -> {tgt:<8}: {s:.4f} -> {v:.4f} ({v-s:+.4f})")

    # Save
    out = REPORTS / "cross_dataset_comprehensive.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Saved: {out}")
    print(f"  Total: {time.time()-t_total:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
