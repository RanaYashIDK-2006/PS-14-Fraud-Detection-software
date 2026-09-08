#!/usr/bin/env python3
"""PaySim v2 feature engineering — interactions, binning, step-based temporal signals."""

import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true); nf = int(y_true.sum()); nl = n - nf
    prev = nf / n if n else 0.0
    roc = roc_auc_score(y_true, y_prob) if nf and nl else None
    pr = average_precision_score(y_true, y_prob) if nf else None
    br = brier_score_loss(y_true, y_prob) if n else None
    ra = {}
    if nf and nl:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        for tf, lb in [(0.001, "0.1"), (0.005, "0.5"), (0.01, "1")]:
            m = fpr <= tf
            ra[lb] = round(float(tpr[np.where(m)[0][-1]]), 4) if m.any() else 0.0
    else:
        for lb in ("0.1", "0.5", "1"): ra[lb] = None
    return {"n": n, "n_fraud": nf, "prevalence": round(prev, 6),
            "roc_auc": round(float(roc), 4) if roc else None,
            "pr_auc": round(float(pr), 4) if pr else None,
            "brier": round(float(br), 6) if br else None,
            "recall_at_0_1pct_fpr": ra.get("0.1"), "recall_at_0_5pct_fpr": ra.get("0.5"),
            "recall_at_1pct_fpr": ra.get("1")}


def features_v2(df):
    """Advanced PaySim features with interactions, binning, step-based temporal."""
    X = []
    amt = df["amount"].values.astype(np.float64)
    obO = df["oldbalanceOrg"].values.astype(np.float64)
    nbO = df["newbalanceOrig"].values.astype(np.float64)
    obD = df["oldbalanceDest"].values.astype(np.float64)
    nbD = df["newbalanceDest"].values.astype(np.float64)

    is_co = (df["type"] == "CASH_OUT").values.astype(np.float32)
    is_tf = (df["type"] == "TRANSFER").values.astype(np.float32)
    is_ci = (df["type"] == "CASH_IN").values.astype(np.float32)
    is_pm = (df["type"] == "PAYMENT").values.astype(np.float32)
    is_db = (df["type"] == "DEBIT").values.astype(np.float32)
    X.append(np.column_stack([is_co, is_tf, is_ci, is_pm, is_db]))

    X.append(amt.reshape(-1, 1).astype(np.float32))

    # ── Core balance features ─────────────────────────────────────────────
    drain_org = np.where(obO > 0, (obO - nbO) / obO, 0.0)
    drain_dest = np.where(obD > 0, (nbD - obD) / obD, 0.0)
    amt_bal_ratio = np.where(obO > 0, amt / obO, 0.0)
    recv_ratio = np.where(amt > 0, (nbD - obD) / amt, 0.0)
    exhausted = (nbO < 0.01).astype(np.float32)
    delta_org = (obO - nbO).astype(np.float32)
    delta_dest = (nbD - obD).astype(np.float32)
    X.append(np.column_stack([
        drain_org.astype(np.float32), drain_dest.astype(np.float32),
        amt_bal_ratio.astype(np.float32),
        np.clip(recv_ratio, -10, 10).astype(np.float32),
        exhausted, delta_org, delta_dest,
    ]))

    # ── Interaction features ──────────────────────────────────────────────
    # CASH_OUT + full drain = very suspicious
    X.append((is_co * exhausted).reshape(-1, 1))
    # TRANSFER + new dest = mule-like
    X.append((is_tf * (obD < 0.01).astype(np.float32)).reshape(-1, 1))
    # High amount + CASH_OUT
    X.append((is_co * (amt > 10000).astype(np.float32)).reshape(-1, 1))
    # Transfer to experienced dest
    X.append((is_tf * (obD > 10000).astype(np.float32)).reshape(-1, 1))
    # Amount * drain ratio interaction
    X.append((amt * drain_org).reshape(-1, 1).astype(np.float32))
    # CASH_OUT + high drain
    X.append((is_co * np.clip(drain_org, 0, 1)).reshape(-1, 1))
    # TRANSFER + amount > balance
    X.append((is_tf * (amt > obO).astype(np.float32)).reshape(-1, 1))

    # ── Log transforms ────────────────────────────────────────────────────
    X.append(np.log1p(amt).reshape(-1, 1).astype(np.float32))
    X.append(np.log1p(obO).reshape(-1, 1).astype(np.float32))
    X.append(np.log1p(obD).reshape(-1, 1).reshape(-1, 1).astype(np.float32))

    # ── Step-based features (temporal within the simulation) ───────────────
    if "step" in df.columns:
        step = df["step"].values.astype(np.float32)
        # Step mod 24 (hour of day in simulation)
        X.append((step % 24).reshape(-1, 1))
        # Is night step (simulated)
        X.append(((step % 24) >= 22).astype(np.float32).reshape(-1, 1))
        # Step / 24 (day number)
        X.append((step / 24).reshape(-1, 1))

    # ── Amount binning ────────────────────────────────────────────────────
    # Discretize amount into risk-relevant bins
    amt_bins = np.zeros((len(amt), 4), dtype=np.float32)
    amt_bins[amt < 100, 0] = 1       # micro
    amt_bins[(amt >= 100) & (amt < 1000), 1] = 1   # small
    amt_bins[(amt >= 1000) & (amt < 10000), 2] = 1  # medium
    amt_bins[amt >= 10000, 3] = 1     # large
    X.append(amt_bins)

    # ── Balance inconsistency score ───────────────────────────────────────
    # Fraud often has balance inconsistencies
    expected_org_after = obO - amt
    org_inconsistency = np.abs(nbO - expected_org_after)
    expected_dest_after = obD + amt
    dest_inconsistency = np.abs(nbD - expected_dest_after)
    X.append(org_inconsistency.reshape(-1, 1).astype(np.float32))
    X.append(dest_inconsistency.reshape(-1, 1).astype(np.float32))

    # ── Is FlaggedFraud (supervised signal — include only for feature analysis) ──
    if "isFlaggedFraud" in df.columns:
        X.append(df["isFlaggedFraud"].values.astype(np.float32).reshape(-1, 1))

    return np.column_stack(X).astype(np.float32)


FEATURE_NAMES = [
    "is_CO", "is_TF", "is_CI", "is_PM", "is_DB",
    "amount",
    "obO", "nbO", "obD", "nbD",
    "drain_org", "drain_dest", "amt_bal_ratio", "recv_ratio",
    "exhausted", "delta_org", "delta_dest",
    "co_exhausted", "tf_new_dest", "co_high_amt", "tf_exp_dest",
    "amt_x_drain", "co_x_drain", "tf_overdraft",
    "log_amt", "log_obO", "log_obD",
    "step_mod24", "is_night", "day_num",
    "amt_micro", "amt_small", "amt_medium", "amt_large",
    "org_inconsistency", "dest_inconsistency",
    "isFlaggedFraud",
]


def eval_split(X, y, name):
    n = len(X); split = int(n * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xc_tr = np.nan_to_num(imp.transform(Xtr))
    Xc_te = np.nan_to_num(imp.transform(Xte))
    sc = StandardScaler().fit(Xc_tr)
    Xs_tr, Xs_te = sc.transform(Xc_tr), sc.transform(Xc_te)

    np_ = int(ytr.sum()); nn = split - np_
    model = XGBClassifier(
        n_estimators=600, max_depth=10, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=max(nn / max(np_, 1), 1.0),
        eval_metric="auc", random_state=SEED, n_jobs=-1, verbosity=0,
        min_child_weight=3, gamma=0.05, reg_alpha=0.1,
    )
    model.fit(Xs_tr, ytr)
    prob = model.predict_proba(Xs_te)[:, 1]
    mets = compute_metrics(yte, prob)

    imp_vals = model.feature_importances_
    top_idx = np.argsort(imp_vals)[::-1][:15]

    print(f"\n=== {name} ({X.shape[1]} features) ===")
    print(f"  ROC-AUC:  {mets['roc_auc']}  |  PR-AUC: {mets['pr_auc']}")
    print(f"  R@0.1%:   {mets['recall_at_0_1pct_fpr']}  |  R@0.5%: {mets['recall_at_0_5pct_fpr']}  |  R@1%: {mets['recall_at_1pct_fpr']}")
    print(f"  Brier:    {mets['brier']}")
    print(f"  Top features:")
    for i, idx in enumerate(top_idx):
        fname = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat[{idx}]"
        print(f"    {i+1:>2}. {fname:<25} importance={imp_vals[idx]:.4f}")

    return mets, model, imp_vals


def main():
    print("=" * 70)
    print("PaySim v2 FEATURE ENGINEERING")
    print("=" * 70)
    t0 = time.time()

    print("Loading 1.2M rows...")
    df = pd.read_csv(ROOT / "data" / "paysim_1m.csv")
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
    y = df["isFraud"].values.astype(int)
    print(f"  {len(df):,} rows, {int(y.sum()):,} fraud")

    # Run enhanced features
    print("\nBuilding enhanced features...")
    t1 = time.time()
    X = features_v2(df)
    print(f"  {X.shape[1]} features built in {time.time()-t1:.1f}s")

    mets, model, imp_vals = eval_split(X, y, "PaySim v2 Enhanced")

    # Also run baseline for comparison
    print("\n--- Baseline comparison ---")
    type_dummies = pd.get_dummies(df["type"], prefix="type").values.astype(np.float32)
    amt = df["amount"].values.astype(np.float32).reshape(-1, 1)
    bal = df[["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].values.astype(np.float32)
    X_base = np.hstack([type_dummies, amt, bal])
    m_base, _, _ = eval_split(X_base, y, "Baseline (10 features)")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>10} {'Enhanced':>10} {'Delta':>10}")
    print("-" * 55)
    for k in ["roc_auc", "pr_auc", "recall_at_0_1pct_fpr", "recall_at_0_5pct_fpr", "recall_at_1pct_fpr", "brier"]:
        bv = m_base.get(k); ev = mets.get(k)
        if bv is not None and ev is not None:
            d = ev - bv
            print(f"  {k:<23} {bv:>10.4f} {ev:>10.4f} {d:>+10.4f}")

    result = {"baseline": m_base, "enhanced": mets,
              "n_features": X.shape[1], "total_time": round(time.time()-t0, 1)}
    (ROOT / "benchmarks" / "paysim_v2.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved to benchmarks/paysim_v2.json")


if __name__ == "__main__":
    main()
