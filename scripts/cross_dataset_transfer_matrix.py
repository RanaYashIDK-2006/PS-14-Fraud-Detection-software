#!/usr/bin/env python3
"""Complete Cross-Dataset Transfer Matrix.

Trains fraud models on one dataset and evaluates on all others.
Measures: in-domain baseline, cross-domain transfer (base features),
cross-domain transfer (+velocity), and calibrated transfer.

All datasets use the same XGB hyperparameters for fair comparison.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
NJ = 4


def recall_at_fpr(y_true, scores, target=0.01):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target, side="right")
    return float(tpr[idx - 1]) if idx > 0 else 0.0


def train_evaluate(Xtr, ytr, Xte, yte, label=""):
    """Train XGB, return metrics."""
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n_pos = int(ytr.sum())
    if n_pos < 5:
        return {"auc": 0, "r1": 0, "r05": 0, "n_features": Xtr.shape[1], "error": "few_pos"}

    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, gamma=2,
                      min_child_weight=5,
                      scale_pos_weight=min(n_pos / max(len(ytr) - n_pos, 1) * 50, 200),
                      random_state=42, n_jobs=NJ, eval_metric="auc")
    m.fit(Xtr, ytr, verbose=False)
    p = m.predict_proba(Xte)[:, 1]
    a = roc_auc_score(yte, p)
    r1 = recall_at_fpr(yte, p, 0.01)
    r05 = recall_at_fpr(yte, p, 0.005)
    return {"auc": round(a, 6), "r1": round(r1, 6), "r05": round(r05, 6),
            "n_features": Xtr.shape[1]}


# ── Dataset Loaders ──

def load_ulb():
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    y = df["Class"].values.astype(int)
    # Domain features
    domain = {}
    for i in range(1, 29):
        domain[f"V{i}"] = df[f"V{i}"].values
    domain["Amount"] = df["Amount"].values
    # Velocity features
    for v in ["V14", "V17", "V12", "V10", "V4", "V11"]:
        rm = pd.Series(domain[v]).rolling(200, min_periods=1).mean().values
        domain[f"{v}_dev"] = domain[v] - rm
    devs = [f"{v}_dev" for v in ["V14", "V17", "V12", "V10", "V4", "V11"]]
    domain["pca_anomaly_sum"] = np.sum(np.abs([domain[d] for d in devs]), axis=0)
    domain["amt_x_V14"] = domain["Amount"] * domain["V14"]
    domain["amt_x_V17"] = domain["Amount"] * domain["V17"]

    # Shared semantic features
    shared = {}
    shared["amount_log"] = np.log1p(np.clip(domain["Amount"], 0, 1e6))
    hour = (df["Time"].values % 86400) / 3600.0
    shared["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    shared["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    shared["is_night"] = ((hour >= 22) | (hour <= 6)).astype(float)
    shared["amount_sq"] = domain["Amount"] ** 2
    shared["high_amount"] = (domain["Amount"] > 100).astype(float)
    shared["amount_bucket"] = np.nan_to_num(pd.cut(domain["Amount"], bins=[0,10,50,200,1000,1e9], labels=False).astype(float), nan=0.0)

    return domain, shared, y, "ULB"


def load_altman():
    rng = np.random.RandomState(42)
    rows_all = []
    ci = 0
    for chunk in pd.read_csv(ROOT / "data" / "credit_card_transactions-ibm_v2.csv",
        usecols=["User","Month","Day","Time","Amount","Use Chip","MCC","Errors?",
                 "Is Fraud?","Merchant Name","Merchant City","Merchant State","Zip","Year","Card"],
        low_memory=False, chunksize=1_000_000):
        fm = chunk["Is Fraud?"] == "Yes"
        rows_all.append(pd.concat([chunk[fm], chunk[~fm].sample(frac=0.01, random_state=rng)]))
        ci += 1
        if ci >= 16: break
    raw = pd.concat(rows_all, ignore_index=True)

    y = (raw["Is Fraud?"] == "Yes").values.astype(int)

    chip_map = {"Swipe Transaction": 0, "Online Transaction": 1, "Chip Transaction": 2}
    chip = raw["Use Chip"].map(chip_map).fillna(-1).values.astype(float)
    mcc_n = pd.to_numeric(raw["MCC"], errors="coerce").fillna(0).values / 10000
    err = (raw["Errors?"].fillna("") != "").values.astype(float)
    tp = raw["Time"].str.split(":", expand=True)
    hr = pd.to_numeric(tp[0], errors="coerce").fillna(12).values.astype(float)
    mn = pd.to_numeric(tp[1], errors="coerce").fillna(0).values.astype(float)
    merchant_id = raw["Merchant Name"].astype("category").cat.codes.values.astype(float)
    city_id = raw["Merchant City"].astype("category").cat.codes.values.astype(float)
    is_online = (raw["Merchant City"] == "ONLINE").values.astype(float)
    day_decimal = raw["Day"].values.astype(float) + hr/24 + mn/1440
    amt = pd.to_numeric(raw["Amount"].str.replace("$","",regex=False)
                        .str.replace(",","",regex=False), errors="coerce").fillna(0).values

    domain = {"amt": amt, "hr": hr, "mn": mn, "mcc_n": mcc_n, "chip": chip, "err": err,
              "merchant_id": merchant_id, "city_id": city_id, "is_online": is_online,
              "day_decimal": day_decimal, "Month": raw["Month"].values.astype(float),
              "Year": raw["Year"].values.astype(float), "Card": raw["Card"].values.astype(float),
              "amt_log": np.log1p(np.clip(amt, 0, 1e9)),
              "amt_x_hr": amt * hr, "amt_x_mcc": amt * mcc_n, "amt_x_chip": amt * chip,
              "amt_sq": amt ** 2,
              "night_tx": ((hr >= 22) | (hr <= 6)).astype(float)}

    # Velocity
    df_tmp = pd.DataFrame(domain)
    df_tmp["merchant_tx_count"] = df_tmp.groupby("merchant_id").cumcount().values / 1000
    df_tmp["city_tx_count"] = df_tmp.groupby("city_id").cumcount().values / 1000
    df_tmp["night_deviation"] = df_tmp["night_tx"].rolling(100, min_periods=1).mean().fillna(0)
    domain["merchant_tx_count"] = df_tmp["merchant_tx_count"].values
    domain["city_tx_count"] = df_tmp["city_tx_count"].values
    domain["night_deviation"] = df_tmp["night_deviation"].values

    # Shared semantic
    shared = {}
    shared["amount_log"] = np.log1p(np.clip(amt, 0, 1e6))
    shared["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    shared["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    shared["is_night"] = ((hr >= 22) | (hr <= 6)).astype(float)
    shared["amount_sq"] = amt ** 2
    shared["high_amount"] = (amt > 100).astype(float)
    shared["amount_bucket"] = np.nan_to_num(pd.cut(amt, bins=[0,10,50,200,1000,1e9], labels=False).astype(float), nan=0.0)

    return domain, shared, y, "Altman"


def load_paysim():
    df = pd.read_csv(ROOT / "data" / "paysim_1m.csv")
    y = df["isFraud"].values.astype(int)

    domain = {"amount": df["amount"].values,
              "oldbalanceOrg": df["oldbalanceOrg"].values,
              "newbalanceOrig": df["newbalanceOrig"].values,
              "oldbalanceDest": df["oldbalanceDest"].values,
              "newbalanceDest": df["newbalanceDest"].values}
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    for c in type_dummies.columns:
        domain[c] = type_dummies[c].values.astype(float)
    domain["bal_diff_orig"] = domain["newbalanceOrig"] - domain["oldbalanceOrg"]
    domain["bal_diff_dest"] = domain["newbalanceDest"] - domain["oldbalanceDest"]
    domain["amt_ratio_orig"] = df["amount"].values / (domain["oldbalanceOrg"] + 1)
    domain["orig_wiped"] = (domain["newbalanceOrig"] == 0).astype(float)
    domain["dest_zero"] = (domain["oldbalanceDest"] == 0).astype(float)
    domain["flow_asym"] = np.abs(domain["bal_diff_orig"] + domain["bal_diff_dest"])
    domain["is_cashout"] = (df["type"] == "CASH_OUT").values.astype(float)
    domain["is_transfer"] = (df["type"] == "TRANSFER").values.astype(float)
    domain["balance_wipe_score"] = domain["orig_wiped"] * domain["is_transfer"]
    domain["drain_ratio"] = df["amount"].values / (domain["oldbalanceOrg"] + 1)

    # Shared semantic (PaySim has no time — use synthetic for fairness)
    rng = np.random.RandomState(42)
    shared = {}
    shared["amount_log"] = np.log1p(np.clip(df["amount"].values, 0, 1e6))
    shared["hour_sin"] = rng.uniform(-1, 1, len(df))
    shared["hour_cos"] = rng.uniform(-1, 1, len(df))
    shared["is_night"] = rng.uniform(0, 1, len(df))
    shared["amount_sq"] = df["amount"].values ** 2
    shared["high_amount"] = (df["amount"].values > 100).astype(float)
    shared["amount_bucket"] = np.nan_to_num(pd.cut(df["amount"].values, bins=[0,10,50,200,1000,1e9], labels=False).astype(float), nan=0.0)

    return domain, shared, y, "PaySim"


def to_matrix(feat_dict, feature_names=None):
    """Convert dict of arrays to matrix, selecting only requested features."""
    if feature_names is None:
        feature_names = sorted(feat_dict.keys())
    cols = [f for f in feature_names if f in feat_dict]
    if not cols:
        return np.zeros((len(next(iter(feat_dict.values()))), 0)), []
    X = np.column_stack([feat_dict[c] for c in cols])
    return X, cols


def standardize(Xtr, Xte):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    return Xtr_s, Xte_s


# ── Main ──

def main():
    print("=" * 80)
    print("  COMPLETE CROSS-DATASET TRANSFER MATRIX")
    print("  ULB x Altman x PaySim")
    print("=" * 80)
    t0 = time.time()

    # Load
    print("\n[1] Loading datasets...")
    ulb_dom, ulb_sh, ulb_y, _ = load_ulb()
    alt_dom, alt_sh, alt_y, _ = load_altman()
    ps_dom, ps_sh, ps_y, _ = load_paysim()
    print(f"  ULB:    {len(ulb_y):>8,} rows, {int(ulb_y.sum()):>5,} fraud ({ulb_y.mean()*100:.3f}%)")
    print(f"  Altman: {len(alt_y):>8,} rows, {int(alt_y.sum()):>5,} fraud ({alt_y.mean()*100:.3f}%)")
    print(f"  PaySim: {len(ps_y):>8,} rows, {int(ps_y.sum()):>5,} fraud ({ps_y.mean()*100:.3f}%)")

    # Prepare feature sets
    print("\n[2] Preparing feature matrices...")

    # Build feature name lists
    ulb_dom_names = sorted(ulb_dom.keys())
    alt_dom_names = sorted(alt_dom.keys())
    ps_dom_names = sorted(ps_dom.keys())
    shared_names = sorted(set(list(ulb_sh.keys()) + list(alt_sh.keys()) + list(ps_sh.keys())))

    # Velocity-only features (added on top of shared)
    ulb_vel_names = [f for f in ulb_dom_names if "dev" in f or "anomaly" in f or "amt_x_V" in f]
    alt_vel_names = [f for f in alt_dom_names if "tx_count" in f or "deviation" in f]
    ps_vel_names = [f for f in ps_dom_names if "cashout" in f or "transfer" in f or "wipe" in f or "drain" in f]

    shared_vel_names = shared_names + ulb_vel_names + alt_vel_names + ps_vel_names

    print(f"  Shared features: {len(shared_names)}")
    print(f"  Shared+velocity: {len(shared_vel_names)}")
    print(f"  ULB domain: {len(ulb_dom_names)}")
    print(f"  Altman domain: {len(alt_dom_names)}")
    print(f"  PaySim domain: {len(ps_dom_names)}")

    # Build matrices
    datasets = {
        "ULB": {"domain": ulb_dom, "shared": ulb_sh, "y": ulb_y,
                "dom_names": ulb_dom_names},
        "Altman": {"domain": alt_dom, "shared": alt_sh, "y": alt_y,
                   "dom_names": alt_dom_names},
        "PaySim": {"domain": ps_dom, "shared": ps_sh, "y": ps_y,
                   "dom_names": ps_dom_names},
    }

    # ── Experiment 1: In-domain 5-fold CV ──
    print("\n[3] In-domain baselines (5-fold CV)...")
    in_domain = {}
    for name, d in datasets.items():
        # Domain features
        X, feat_names = to_matrix(d["domain"], d["dom_names"])
        y = d["y"]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs, r1s = [], []
        for tr, te in skf.split(X, y):
            Xtr, Xte = X[tr], X[te]
            Xtr = np.nan_to_num(Xtr).astype(np.float32)
            Xte = np.nan_to_num(Xte).astype(np.float32)
            ytr, yte = y[tr], y[te]
            r = train_evaluate(Xtr, ytr, Xte, yte)
            aucs.append(r["auc"])
            r1s.append(r["r1"])
        in_domain[name] = {"auc": round(np.mean(aucs), 4), "auc_std": round(np.std(aucs), 4),
                            "r1": round(np.mean(r1s), 4), "n_features": len(feat_names)}
        print(f"  {name:>8} (domain, {len(feat_names)} feats):  AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}  R@1%={np.mean(r1s):.4f}")

        # Shared features
        Xs, sn = to_matrix(d["shared"], shared_names)
        aucs2, r1s2 = [], []
        for tr, te in skf.split(Xs, y):
            r = train_evaluate(Xs[tr], y[tr], Xs[te], y[te])
            aucs2.append(r["auc"])
            r1s2.append(r["r1"])
        in_domain[f"{name}_shared"] = {"auc": round(np.mean(aucs2), 4), "r1": round(np.mean(r1s2), 4), "n_features": len(sn)}
        print(f"  {name:>8} (shared,  {len(sn)} feats):   AUC={np.mean(aucs2):.4f}  R@1%={np.mean(r1s2):.4f}")

    # ── Experiment 2-4: Cross-domain transfer ──
    print("\n[4] Cross-domain transfer matrix...")
    pairs = [("ULB", "Altman"), ("ULB", "PaySim"), ("Altman", "ULB"),
             ("Altman", "PaySim"), ("PaySim", "ULB"), ("PaySim", "Altman")]

    modes = [
        ("shared", shared_names),
        ("shared+vel", shared_vel_names),
        ("domain", None),  # use each dataset's own domain features (aligned)
    ]

    transfer = {}
    for src, tgt in pairs:
        transfer[f"{src}->{tgt}"] = {}
        src_d, tgt_d = datasets[src], datasets[tgt]
        ytr, yte = src_d["y"], tgt_d["y"]

        for mode, feat_list in modes:
            # Merge domain + shared dicts for feature lookup
            src_all = {**src_d["domain"], **src_d["shared"]}
            tgt_all = {**tgt_d["domain"], **tgt_d["shared"]}
            if feat_list is None:
                # Domain mode: align on common domain features
                src_X, src_fn = to_matrix(src_d["domain"], src_d["dom_names"])
                tgt_X, tgt_fn = to_matrix(tgt_d["domain"], tgt_d["dom_names"])
                common = [f for f in src_fn if f in tgt_fn]
                src_X, _ = to_matrix(src_d["domain"], common)
                tgt_X, _ = to_matrix(tgt_d["domain"], common)
                n_feat = len(common)
            else:
                src_X, src_fn = to_matrix(src_all, feat_list)
                tgt_X, tgt_fn = to_matrix(tgt_all, feat_list)
                # Align on common features present in both
                common = [f for f in src_fn if f in tgt_fn]
                src_X, _ = to_matrix(src_all, common)
                tgt_X, _ = to_matrix(tgt_all, common)
                n_feat = len(common)

            # Standardize
            src_X = np.nan_to_num(src_X).astype(np.float32)
            tgt_X = np.nan_to_num(tgt_X).astype(np.float32)
            if src_X.shape[1] == 0:
                r = {"auc": 0.5, "r1": 0.0, "r05": 0.0, "n_features": 0, "error": "no_common_features"}
            else:
                sc = StandardScaler()
                src_Xs = sc.fit_transform(src_X)
                tgt_Xs = sc.transform(tgt_X)
                r = train_evaluate(src_Xs, ytr, tgt_Xs, yte)
            transfer[f"{src}->{tgt}"][mode] = r

        print(f"  {src:>8} -> {tgt:<8}:  shared={transfer[f'{src}->{tgt}']['shared']['auc']:.4f}"
              f"  s+vel={transfer[f'{src}->{tgt}']['shared+vel']['auc']:.4f}"
              f"  domain={transfer[f'{src}->{tgt}']['domain']['auc']:.4f}")

    # ── Experiment 5: Calibrated transfer ──
    print("\n[5] Calibrated transfer (500-sample isotonic on target)...")
    calibrated = {}
    for src, tgt in pairs:
        src_d, tgt_d = datasets[src], datasets[tgt]
        ytr, yte = src_d["y"], tgt_d["y"]

        src_X, _ = to_matrix(src_d["shared"], shared_names)
        tgt_X, _ = to_matrix(tgt_d["shared"], shared_names)
        n_feat = min(src_X.shape[1], tgt_X.shape[1])
        src_X, tgt_X = src_X[:, :n_feat], tgt_X[:, :n_feat]

        src_X = np.nan_to_num(src_X).astype(np.float32)
        tgt_X = np.nan_to_num(tgt_X).astype(np.float32)
        sc = StandardScaler()
        src_Xs = sc.fit_transform(src_X)
        tgt_Xs = sc.transform(tgt_X)

        # Split target: 500 for calibration, rest for evaluation
        n_cal = min(500, len(yte) // 5)
        rng_c = np.random.RandomState(42)
        cal_idx = rng_c.choice(len(yte), n_cal, replace=False)
        eval_idx = np.setdiff1d(np.arange(len(yte)), cal_idx)

        # Train on source
        n_pos = int(ytr.sum())
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, gamma=2,
                          min_child_weight=5,
                          scale_pos_weight=min(n_pos/max(len(ytr)-n_pos,1)*50, 200),
                          random_state=42, n_jobs=NJ, eval_metric="auc")
        m.fit(src_Xs, ytr, verbose=False)
        all_preds = m.predict_proba(tgt_Xs)[:, 1]

        # Calibrate
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(all_preds[cal_idx], yte[cal_idx])
        cal_preds = iso.transform(all_preds[eval_idx])

        raw_auc = roc_auc_score(yte[eval_idx], all_preds[eval_idx])
        raw_r1 = recall_at_fpr(yte[eval_idx], all_preds[eval_idx])
        cal_auc = roc_auc_score(yte[eval_idx], cal_preds)
        cal_r1 = recall_at_fpr(yte[eval_idx], cal_preds)

        calibrated[f"{src}->{tgt}"] = {"raw_auc": round(raw_auc, 4), "raw_r1": round(raw_r1, 4),
                                         "cal_auc": round(cal_auc, 4), "cal_r1": round(cal_r1, 4),
                                         "delta_auc": round(cal_auc - raw_auc, 4)}

        print(f"  {src:>8} -> {tgt:<8}: raw AUC={raw_auc:.4f} -> cal AUC={cal_auc:.4f} (delta={cal_auc-raw_auc:+.4f})")

    # ── Summary Matrix ──
    print("\n" + "=" * 80)
    print("  COMPLETE TRANSFER MATRIX")
    print("=" * 80)

    print(f"\n  {'Source':>8} -> {'Target':<8}  {'In-Dom':>7}  {'Shared':>7}  {'S+Vel':>7}  {'Domain':>7}  {'Calib':>7}  {'Cal.R1':>7}")
    print(f"  {'-'*8}    {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    for src, tgt in pairs:
        id_auc = in_domain[src]["auc"]
        s = transfer[f"{src}->{tgt}"]["shared"]["auc"]
        sv = transfer[f"{src}->{tgt}"]["shared+vel"]["auc"]
        d = transfer[f"{src}->{tgt}"]["domain"]["auc"]
        c = calibrated[f"{src}->{tgt}"]["cal_auc"]
        cr = calibrated[f"{src}->{tgt}"]["cal_r1"]
        print(f"  {src:>8} -> {tgt:<8}  {id_auc:>7.4f}  {s:>7.4f}  {sv:>7.4f}  {d:>7.4f}  {c:>7.4f}  {cr:>7.4f}")

    # Diagonal (in-domain)
    for name in ["ULB", "Altman", "PaySim"]:
        r = in_domain[name]
        print(f"  {name:>8} -> {name:<8}  {r['auc']:>7.4f}  {'---':>7}  {'---':>7}  {'---':>7}  {'---':>7}  {'---':>7}  [IN-DOMAIN]")

    # ── Key Insights ──
    print("\n" + "=" * 80)
    print("  KEY INSIGHTS")
    print("=" * 80)

    # Best/worst transfer
    all_t = []
    for key, modes in transfer.items():
        for mode, r in modes.items():
            if r.get("auc", 0) > 0:
                all_t.append({**r, "pair": key, "mode": mode})
    best = max(all_t, key=lambda x: x["auc"])
    worst = min(all_t, key=lambda x: x["auc"])
    print(f"\n  Best transfer:  {best['pair']} ({best['mode']}) AUC={best['auc']:.4f}")
    print(f"  Worst transfer: {worst['pair']} ({worst['mode']}) AUC={worst['auc']:.4f}")

    # Velocity impact
    print("\n  Velocity impact:")
    for src, tgt in pairs:
        s = transfer[f"{src}->{tgt}"]["shared"]["auc"]
        sv = transfer[f"{src}->{tgt}"]["shared+vel"]["auc"]
        d = sv - s
        arrow = "+" if d > 0 else ""
        print(f"    {src:>8} -> {tgt:<8}: {s:.4f} -> {sv:.4f} ({arrow}{d:.4f})")

    # Calibration impact
    print("\n  Calibration impact:")
    for src, tgt in pairs:
        raw = calibrated[f"{src}->{tgt}"]["raw_auc"]
        cal = calibrated[f"{src}->{tgt}"]["cal_auc"]
        d = cal - raw
        arrow = "+" if d > 0 else ""
        print(f"    {src:>8} -> {tgt:<8}: {raw:.4f} -> {cal:.4f} ({arrow}{d:.4f})")

    # Transfer potential (overlap of feature importances)
    print("\n  Transfer potential (how much of target's fraud signal source captures):")
    for src, tgt in pairs:
        s = transfer[f"{src}->{tgt}"]["shared"]["auc"]
        id_s = in_domain[f"{tgt}_shared"]["auc"]
        if id_s > 0:
            pct = s / id_s * 100
            print(f"    {src:>8} -> {tgt:<8}: {s:.4f}/{id_s:.4f} = {pct:.1f}% of in-domain shared performance")

    # Save
    results = {
        "in_domain": in_domain,
        "transfer": transfer,
        "calibrated": calibrated,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = REPORTS / "cross_dataset_transfer_matrix.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out}")
    print(f"  Total: {time.time()-t0:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
