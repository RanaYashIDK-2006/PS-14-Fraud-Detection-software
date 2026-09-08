#!/usr/bin/env python3
"""PaySim feature engineering + evaluation — targeted balance/velocity/exhaustion features.

Current baseline: 93.8% ROC-AUC with 9 features (type dummies + amount + 4 balances).
Goal: boost detection by engineering richer features from the available columns.
"""

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
    n = len(y_true)
    nf = int(y_true.sum()); nl = n - nf
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
        for lb in (".01", ".1", ".5", "1"): ra[lb] = None

    return {
        "n": n, "n_fraud": nf, "prevalence": round(prev, 6),
        "roc_auc": round(float(roc), 4) if roc else None,
        "pr_auc": round(float(pr), 4) if pr else None,
        "brier": round(float(br), 6) if br else None,
        "recall_at_0_1pct_fpr": ra.get("0.1"),
        "recall_at_0_5pct_fpr": ra.get("0.5"),
        "recall_at_1pct_fpr": ra.get("1"),
    }


def features_paysim_baseline(df):
    """Baseline: type dummies + amount + 4 balances = 9 features."""
    type_dummies = pd.get_dummies(df["type"], prefix="type").values.astype(np.float32)
    amt = df["amount"].values.astype(np.float32).reshape(-1, 1)
    bal_cols = []
    for c in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        if c in df.columns:
            bal_cols.append(df[c].values.astype(np.float32).reshape(-1, 1))
    return np.hstack([type_dummies, amt] + bal_cols)


def features_paysim_enhanced(df):
    """Enhanced: balance drain, exhaustion, velocity, novelty features."""
    X = []

    # 1. Type dummies (5 types)
    type_dummies = pd.get_dummies(df["type"], prefix="type").values.astype(np.float32)
    X.append(type_dummies)

    # 2. Raw amount
    amt = df["amount"].values.astype(np.float32)
    X.append(amt.reshape(-1, 1))

    # 3. Balance columns (4)
    obO = df["oldbalanceOrg"].values.astype(np.float32)
    nbO = df["newbalanceOrig"].values.astype(np.float32)
    obD = df["oldbalanceDest"].values.astype(np.float32)
    nbD = df["newbalanceDest"].values.astype(np.float32)
    X.append(np.column_stack([obO, nbO, obD, nbD]))

    # 4. BALANCE DRAIN RATIO (originator) — key fraud signal
    # How much of the sender's balance was drained
    drain_org = np.where(obO > 0, (obO - nbO) / obO, 0.0)
    X.append(drain_org.reshape(-1, 1))

    # 5. BALANCE DRAIN RATIO (destination) — money received
    drain_dest = np.where(obD > 0, (nbD - obD) / obD, 0.0)
    X.append(drain_dest.reshape(-1, 1))

    # 6. AMOUNT / OLD BALANCE — is the amount a large fraction of available funds?
    amt_bal_ratio = np.where(obO > 0, amt / obO, 0.0)
    X.append(amt_bal_ratio.reshape(-1, 1))

    # 7. DESTINATION RECEIVED RATIO — how much of the transfer actually arrived
    received_ratio = np.where(amt > 0, (nbD - obD) / amt, 0.0)
    X.append(np.clip(received_ratio, -10, 10).reshape(-1, 1))

    # 8. ACCOUNT EXHAUSTED FLAG — balance goes to zero after transaction
    exhausted = (nbO < 0.01).astype(np.float32)
    X.append(exhausted.reshape(-1, 1))

    # 9. ORIGINATOR BALANCE DELTA — raw change in sender balance
    bal_delta_org = obO - nbO
    X.append(bal_delta_org.reshape(-1, 1))

    # 10. DESTINATION BALANCE DELTA — raw change in receiver balance
    bal_delta_dest = nbD - obD
    X.append(bal_delta_dest.reshape(-1, 1))

    # 11. IS CASH_OUT or TRANSFER (high-risk types)
    is_high_risk = df["type"].isin(["CASH_OUT", "TRANSFER"]).astype(np.float32).values
    X.append(is_high_risk.reshape(-1, 1))

    # 12. AMOUNT * IS_HIGH_RISK — interaction feature
    X.append((amt * is_high_risk).reshape(-1, 1))

    # 13. NEW BALANCE ZERO AFTER CASH_OUT (very suspicious)
    is_cash_out = (df["type"] == "CASH_OUT").values.astype(np.float32)
    cashout_zero = is_cash_out * exhausted
    X.append(cashout_zero.reshape(-1, 1))

    # 14. TRANSFER TO ZERO-BALANCE DESTINATION (mule-like)
    is_transfer = (df["type"] == "TRANSFER").values.astype(np.float32)
    transfer_new_dest = is_transfer * (obD < 0.01).astype(np.float32)
    X.append(transfer_new_dest.reshape(-1, 1))

    # 15. ORIGINATOR NEVER HAD BALANCE (synthetic account signal)
    no_prior_bal = (obO < 0.01).astype(np.float32)
    X.append(no_prior_bal.reshape(-1, 1))

    # 16. AMOUNT EXCEEDS ORIGINATOR BALANCE (overdraft)
    overdraft = (amt > obO).astype(np.float32)
    X.append(overdraft.reshape(-1, 1))

    # 17. DESTINATION HAD LARGE BALANCE BEFORE (experienced account)
    dest_experienced = (obD > 10000).astype(np.float32)
    X.append(dest_experienced.reshape(-1, 1))

    # 18. LOG AMOUNT (reduce skew)
    log_amt = np.log1p(amt)
    X.append(log_amt.reshape(-1, 1))

    # 19. LOG ORIGINATOR BALANCE
    log_obO = np.log1p(obO)
    X.append(log_obO.reshape(-1, 1))

    # 20. LOG DESTINATION BALANCE
    log_obD = np.log1p(obD)
    X.append(log_obD.reshape(-1, 1))

    # 21. BALANCE CONSISTENCY CHECK — does amount match balance delta?
    # (fraud often has inconsistencies)
    expected_drain = np.abs(obO - nbO)
    drain_match = np.where(amt > 0, np.abs(amt - expected_drain) / amt, 0.0)
    X.append(np.clip(drain_match, 0, 10).reshape(-1, 1))

    return np.hstack(X)


def eval_split(X, y, name, split_ratio=0.8):
    """Train/eval on temporal split."""
    n = len(X)
    split = int(n * split_ratio)
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

    # Feature importance
    imp_vals = model.feature_importances_
    top_idx = np.argsort(imp_vals)[::-1][:10]

    print(f"\n=== {name} ({X.shape[1]} features) ===")
    print(f"  Train: {split:,} | Test: {n-split:,}")
    print(f"  ROC-AUC:  {mets['roc_auc']}")
    print(f"  PR-AUC:   {mets['pr_auc']}")
    print(f"  R@0.1%:   {mets['recall_at_0_1pct_fpr']}")
    print(f"  R@0.5%:   {mets['recall_at_0_5pct_fpr']}")
    print(f"  R@1%:     {mets['recall_at_1pct_fpr']}")
    print(f"  Brier:    {mets['brier']}")
    print(f"  Top features by importance:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. feat[{idx}] importance={imp_vals[idx]:.4f}")

    return mets


def main():
    print("=" * 70)
    print("PaySim FEATURE ENGINEERING EXPERIMENT")
    print("=" * 70)

    # Load full 1M dataset
    t0 = time.time()
    print("Loading 1.2M rows...")
    df = pd.read_csv(ROOT / "data" / "paysim_1m.csv")
    y = df["isFraud"].values.astype(int)
    print(f"  Loaded {len(df):,} rows ({int(y.sum()):,} fraud) in {time.time()-t0:.1f}s")

    # Sort by step for temporal split (PaySim has a 'step' column representing hour)
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
        y = df["isFraud"].values.astype(int)
        print(f"  Sorted by step (temporal order)")

    # Experiment 1: Baseline
    print("\n" + "=" * 70)
    t1 = time.time()
    X_base = features_paysim_baseline(df)
    m_base = eval_split(X_base, y, "BASELINE (9 features)")
    print(f"  Time: {time.time()-t1:.1f}s")

    # Experiment 2: Enhanced
    t2 = time.time()
    X_enh = features_paysim_enhanced(df)
    m_enh = eval_split(X_enh, y, "ENHANCED (21 features)")
    print(f"  Time: {time.time()-t2:.1f}s")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>10} {'Enhanced':>10} {'Delta':>10}")
    print("-" * 55)
    for k in ["roc_auc", "pr_auc", "recall_at_0_1pct_fpr", "recall_at_0_5pct_fpr", "recall_at_1pct_fpr"]:
        bv = m_base.get(k)
        ev = m_enh.get(k)
        if bv is not None and ev is not None:
            delta = ev - bv
            print(f"  {k:<23} {bv:>10.4f} {ev:>10.4f} {delta:>+10.4f}")
        else:
            print(f"  {k:<23} {str(bv):>10} {str(ev):>10}")

    # Save
    result = {
        "baseline": m_base,
        "enhanced": m_enh,
        "n_features_baseline": X_base.shape[1],
        "n_features_enhanced": X_enh.shape[1],
        "total_time": round(time.time() - t0, 1),
    }
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    (out / "paysim_features.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out / 'paysim_features.json'}")


if __name__ == "__main__":
    main()
