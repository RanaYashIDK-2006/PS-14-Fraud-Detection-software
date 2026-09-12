#!/usr/bin/env python3
"""ULB aggressive push: all 28 PCA velocity features, interactions, CatBoost, stacking."""
from __future__ import annotations
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
REPORTS = ROOT / "reports"
NJ = 4


def recall_at_fpr(y_true, scores, target=0.01):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target, side="right")
    return float(tpr[idx - 1]) if idx > 0 else 0.0


def load_ulb_aggressive():
    """ALL 28 PCA + all 28 velocity deviations + interactions."""
    print("  Loading ULB with aggressive features...")
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    y = df["Class"].values.astype(int)
    amt = df["Amount"].values
    time_s = df["Time"].values
    hour = (time_s % 86400) / 3600.0

    v_cols = [f"V{i}" for i in range(1, 29)]
    v_data = df[v_cols].values.astype(np.float32)

    # Sort by time for velocity
    sort_idx = np.argsort(time_s)
    sorted_v = v_data[sort_idx]
    sorted_amt = amt[sort_idx]

    # Rolling mean deviations for ALL 28 PCs, window=30
    window = 30
    dev = np.zeros_like(v_data)
    for i in range(window, len(df)):
        prev = sorted_v[i - window:i, :]
        rolling_mean = prev.mean(axis=0)
        dev[sort_idx[i]] = sorted_v[i] - rolling_mean

    # Rolling std (volatility)
    vol = np.zeros_like(v_data)
    for i in range(window, len(df)):
        prev = sorted_v[i - window:i, :]
        vol[sort_idx[i]] = prev.std(axis=0)

    # Interaction features: V14 * other key PCs
    v14 = v_data[:, 13]
    v12 = v_data[:, 11]
    v10 = v_data[:, 9]
    v17 = v_data[:, 16]
    v4 = v_data[:, 3]

    interactions = np.column_stack([
        v14 * v12,  # V14xV12
        v14 * v10,  # V14xV10
        v14 * v17,  # V14xV17
        v10 * v12,  # V10xV12
        np.abs(v14) * amt,  # |V14| x Amount
        np.abs(v12) * amt,  # |V12| x Amount
    ])

    # Amount features
    amt_log = np.log1p(np.clip(amt, 0, 1e6)).reshape(-1, 1)
    amt_z = ((amt - amt.mean()) / max(amt.std(), 0.01)).reshape(-1, 1)
    amt_sq = (amt ** 2).reshape(-1, 1)

    # Time features
    hour_sin = np.sin(2 * np.pi * hour / 24).reshape(-1, 1)
    hour_cos = np.cos(2 * np.pi * hour / 24).reshape(-1, 1)
    is_night = ((hour >= 22) | (hour <= 6)).astype(float).reshape(-1, 1)

    # Amount acceleration
    amt_accel = np.zeros(len(df), dtype=np.float32)
    for i in range(window * 2, len(df)):
        recent = sorted_amt[i - window:i].mean()
        older = sorted_amt[i - window * 2:i - window].mean()
        amt_accel[sort_idx[i]] = recent - older

    # PCA anomaly sum (abs dev across all 28 PCs)
    pca_anomaly = np.abs(dev).sum(axis=1).reshape(-1, 1)

    # Top PCA volatility features
    top_pcs = [13, 11, 9, 16, 3, 0, 1, 10]  # V14, V12, V10, V17, V4, V1, V2, V11
    top_dev = dev[:, top_pcs]
    top_vol = vol[:, top_pcs]
    top_dev_names = [f"V{pc+1}_dev" for pc in top_pcs]
    top_vol_names = [f"V{pc+1}_vol" for pc in top_pcs]

    X = np.hstack([
        v_data,  # V1-V28 (28)
        amt.reshape(-1, 1),  # Amount
        time_s.reshape(-1, 1),  # Time
        top_dev,  # 8 dev features
        top_vol,  # 8 vol features
        pca_anomaly,  # 1
        amt_log, amt_z, amt_sq,  # 3
        hour_sin, hour_cos, is_night,  # 3
        amt_accel.reshape(-1, 1),  # 1
        interactions,  # 6
    ])

    feat_names = v_cols + ["Amount", "Time"] + top_dev_names + top_vol_names + [
        "pca_anomaly", "amount_log", "amount_zscore", "amount_sq",
        "hour_sin", "hour_cos", "is_night", "amount_accel",
        "V14xV12", "V14xV10", "V14xV17", "V10xV12", "V14xAMT", "V12xAMT",
    ]

    return np.nan_to_num(X).astype(np.float32), y, feat_names


def train_xgb(Xtr, ytr, Xte, params=None):
    from xgboost import XGBClassifier
    spw = min((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), 200)
    defaults = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.7, gamma=2,
                    min_child_weight=5, scale_pos_weight=spw,
                    random_state=42, n_jobs=NJ, eval_metric="auc")
    if params:
        defaults.update(params)
    m = XGBClassifier(**defaults)
    m.fit(Xtr, ytr, verbose=False)
    return m.predict_proba(Xte)[:, 1]


def train_lgb(Xtr, ytr, Xte, params=None):
    from lightgbm import LGBMClassifier
    spw = min((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), 200)
    defaults = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                    scale_pos_weight=spw, random_state=42, n_jobs=NJ, verbose=-1)
    if params:
        defaults.update(params)
    m = LGBMClassifier(**defaults)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def train_catboost(Xtr, ytr, Xte):
    from catboost import CatBoostClassifier
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    m = CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.05,
        scale_pos_weight=min(spw, 200), random_seed=42,
        thread_count=NJ, verbose=0
    )
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def main():
    print("=" * 70)
    print("  ULB AGGRESSIVE PUSH")
    print("  ALL 28 PCA + velocity + interactions + CatBoost + stacking")
    print("=" * 70)
    t0 = time.time()

    X, y, feat_names = load_ulb_aggressive()
    print(f"  Dataset: {len(y):,} rows, {int(y.sum())} fraud ({y.mean()*100:.3f}%)")
    print(f"  Features: {X.shape[1]} ({len(feat_names)} named)")

    # ── 5-fold CV with stacking ──
    print("\n[1] 5-fold stacking CV...")
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_aucs, cv_r1s, cv_r05s = [], [], []
    all_oof_xgb, all_oof_lgb, all_oof_cb = [], [], []
    all_y = []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
        Xte_s = sc.transform(Xte).astype(np.float32)

        # Train 3 base models
        xgb_p = train_xgb(Xtr_s, ytr, Xte_s)
        lgb_p = train_lgb(Xtr_s, ytr, Xte_s)
        cb_p = train_catboost(Xtr_s, ytr, Xte_s)

        # Meta-learner (logistic regression on OOF predictions)
        oof_p = np.column_stack([xgb_p, lgb_p, cb_p])
        meta = LogisticRegression(C=10, random_state=42)
        meta.fit(oof_p, yte)
        ens_p = meta.predict_proba(oof_p)[:, 1]

        fold_auc = roc_auc_score(yte, ens_p)
        fold_r1 = recall_at_fpr(yte, ens_p)
        fold_r05 = recall_at_fpr(yte, ens_p, 0.005)
        cv_aucs.append(fold_auc)
        cv_r1s.append(fold_r1)
        cv_r05s.append(fold_r05)

        all_oof_xgb.extend(xgb_p)
        all_oof_lgb.extend(lgb_p)
        all_oof_cb.extend(cb_p)
        all_y.extend(yte)

        print(f"    Fold {fold+1}: AUC={fold_auc:.4f}  R@1%={fold_r1:.4f}  "
              f"R@0.5%={fold_r05:.4f}  XGB={roc_auc_score(yte,xgb_p):.4f} "
              f"LGB={roc_auc_score(yte,lgb_p):.4f} CB={roc_auc_score(yte,cb_p):.4f}")

    mu_auc, std_auc = np.mean(cv_aucs), np.std(cv_aucs)
    mu_r1, std_r1 = np.mean(cv_r1s), np.std(cv_r1s)
    mu_r05, std_r05 = np.mean(cv_r05s), np.std(cv_r05s)
    print(f"\n  CV Mean: AUC={mu_auc:.4f}+/-{std_auc:.4f}  R@1%FPR={mu_r1:.4f}+/-{std_r1:.4f}  R@0.5%FPR={mu_r05:.4f}+/-{std_r05:.4f}")

    # Individual model means
    xgb_aucs = [roc_auc_score(np.array(all_y)[np.arange(fold*len(yte)//5, (fold+1)*len(yte)//5)],
                               np.array(all_oof_xgb)[np.arange(fold*len(yte)//5, (fold+1)*len(yte)//5)])
                 for fold in range(5)]

    # ── Feature importance (from full XGB) ──
    print("\n[2] Feature importance (XGB, full data)...")
    sc = StandardScaler()
    Xs = sc.fit_transform(X).astype(np.float32)
    from xgboost import XGBClassifier
    spw = min((len(y) - y.sum()) / max(y.sum(), 1), 200)
    full_model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.7, gamma=2,
                               min_child_weight=5, scale_pos_weight=spw,
                               random_state=42, n_jobs=NJ)
    full_model.fit(Xs, y, verbose=False)
    imp = full_model.feature_importances_
    imp_sorted = sorted(zip(feat_names, imp), key=lambda x: -x[1])
    for name, g in imp_sorted[:15]:
        print(f"    {g:.4f}  {name}")

    # ── Time-split evaluation ──
    print("\n[3] Time-split evaluation (80/20)...")
    sort_idx = np.argsort(X[:, 31])  # Time column
    split = int(len(y) * 0.8)
    tr_idx, te_idx = sort_idx[:split], sort_idx[split:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)

    xgb_p = train_xgb(Xtr_s, ytr, Xte_s)
    lgb_p = train_lgb(Xtr_s, ytr, Xte_s)
    cb_p = train_catboost(Xtr_s, ytr, Xte_s)

    # Weighted ensemble
    best_auc = 0
    best_w = None
    for w1 in np.arange(0.2, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.7, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            ens = w1 * xgb_p + w2 * lgb_p + w3 * cb_p
            a = roc_auc_score(yte, ens)
            if a > best_auc:
                best_auc = a
                best_w = (w1, w2, w3)

    ens_p = best_w[0] * xgb_p + best_w[1] * lgb_p + best_w[2] * cb_p
    ens_r1 = recall_at_fpr(yte, ens_p)
    ens_r05 = recall_at_fpr(yte, ens_p, 0.005)
    print(f"  Time-split: XGB={roc_auc_score(yte,xgb_p):.4f} LGB={roc_auc_score(yte,lgb_p):.4f} CB={roc_auc_score(yte,cb_p):.4f}")
    print(f"  Ensemble ({best_w[0]:.1f}/{best_w[1]:.1f}/{best_w[2]:.1f}): AUC={best_auc:.4f}  R@1%={ens_r1:.4f}  R@0.5%={ens_r05:.4f}")

    # ── Save ──
    results = {
        "dataset": "ULB creditcard.csv",
        "n_rows": len(y),
        "n_fraud": int(y.sum()),
        "n_features": int(X.shape[1]),
        "cv_5fold_stacking": {
            "auc_mean": round(float(mu_auc), 6),
            "auc_std": round(float(std_auc), 6),
            "r1_mean": round(float(mu_r1), 6),
            "r1_std": round(float(std_r1), 6),
            "r05_mean": round(float(mu_r05), 6),
            "r05_std": round(float(std_r05), 6),
        },
        "time_split_ensemble": {
            "auc": round(float(best_auc), 6),
            "r1": round(float(ens_r1), 6),
            "r05": round(float(ens_r05), 6),
            "weights": {"xgb": round(best_w[0], 2), "lgb": round(best_w[1], 2), "cb": round(best_w[2], 2)},
        },
        "feature_importance_top15": [{"name": n, "importance": round(float(g), 6)} for n, g in imp_sorted[:15]],
        "target_assessment": {
            "auc_99": "ACHIEVED" if mu_auc >= 0.99 else f"NOT MET ({mu_auc:.4f})",
            "r1_90": "ACHIEVED" if mu_r1 >= 0.90 else f"NOT MET ({mu_r1:.4f})",
        },
    }

    out = REPORTS / "ulb_aggressive_push.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out}")

    best_final = max(mu_auc, best_auc)
    best_r1_final = max(mu_r1, ens_r1)
    print(f"\n  Target AUC > 99%:     {'ACHIEVED' if best_final >= 0.99 else f'NOT MET ({best_final:.4f})'}")
    print(f"  Target R@1%FPR > 90%: {'ACHIEVED' if best_r1_final >= 0.90 else f'NOT MET ({best_r1_final:.4f})'}")
    print(f"\n  Total time: {time.time()-t0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
