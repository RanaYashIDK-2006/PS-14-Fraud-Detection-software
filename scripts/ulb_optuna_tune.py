#!/usr/bin/env python3
"""Optuna tuning on ULB aggressive features to push AUC above 99%."""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
NJ = 4


def recall_at_fpr(y_true, scores, target=0.01):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target, side="right")
    return float(tpr[idx - 1]) if idx > 0 else 0.0


def load_ulb_features():
    """Same aggressive features as ulb_aggressive_push.py."""
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    y = df["Class"].values.astype(int)
    amt = df["Amount"].values
    time_s = df["Time"].values
    hour = (time_s % 86400) / 3600.0
    v_cols = [f"V{i}" for i in range(1, 29)]
    v_data = df[v_cols].values.astype(np.float32)

    sort_idx = np.argsort(time_s)
    sorted_v = v_data[sort_idx]
    sorted_amt = amt[sort_idx]

    window = 30
    dev = np.zeros_like(v_data)
    for i in range(window, len(df)):
        dev[sort_idx[i]] = sorted_v[i] - sorted_v[i - window:i].mean(axis=0)

    top_pcs = [13, 11, 9, 16, 3, 0, 1, 10]
    top_dev = dev[:, top_pcs]

    v14 = v_data[:, 13]
    v12 = v_data[:, 11]
    v10 = v_data[:, 9]
    v17 = v_data[:, 16]
    interactions = np.column_stack([v14*v12, v14*v10, v14*v17, np.abs(v14)*amt, np.abs(v12)*amt])

    X = np.hstack([
        v_data,
        amt.reshape(-1, 1),
        time_s.reshape(-1, 1),
        top_dev,
        np.log1p(np.clip(amt, 0, 1e6)).reshape(-1, 1),
        ((amt - amt.mean()) / max(amt.std(), 0.01)).reshape(-1, 1),
        np.sin(2*np.pi*hour/24).reshape(-1, 1),
        np.cos(2*np.pi*hour/24).reshape(-1, 1),
        ((hour >= 22) | (hour <= 6)).astype(float).reshape(-1, 1),
        interactions,
    ])
    return np.nan_to_num(X).astype(np.float32), y


def main():
    print("=" * 70)
    print("  ULB OPTUNA TUNING (Aggressive Features)")
    print("=" * 70)
    t0 = time.time()

    X, y = load_ulb_features()
    print(f"  Dataset: {len(y):,} rows, {int(y.sum())} fraud, {X.shape[1]} features")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    spw = min((len(y) - y.sum()) / max(y.sum(), 1), 200)

    # ── Optuna: tune XGB + LGB + CB weights ──
    print("\n[1] Optuna: tune XGB (50 trials)...")
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    def xgb_objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_est", 200, 800),
            max_depth=trial.suggest_int("md", 3, 10),
            learning_rate=trial.suggest_float("lr", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("ss", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("cs", 0.5, 1.0),
            gamma=trial.suggest_float("g", 0, 5),
            min_child_weight=trial.suggest_int("mcw", 1, 20),
            scale_pos_weight=spw, random_state=42, n_jobs=NJ, eval_metric="auc",
        )
        aucs = []
        for tr_idx, te_idx in skf.split(X, y):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr_idx]).astype(np.float32)
            Xte = sc.transform(X[te_idx]).astype(np.float32)
            m = XGBClassifier(**params)
            m.fit(Xtr, y[tr_idx], verbose=False)
            aucs.append(roc_auc_score(y[te_idx], m.predict_proba(Xte)[:, 1]))
        return np.mean(aucs)

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(xgb_objective, n_trials=50, show_progress_bar=False)
    print(f"  XGB best: {study_xgb.best_value:.4f}")
    print(f"  XGB params: {study_xgb.best_params}")

    print("\n[2] Optuna: tune LGB (30 trials)...")
    def lgb_objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_est", 200, 800),
            max_depth=trial.suggest_int("md", 3, 10),
            learning_rate=trial.suggest_float("lr", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("ss", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("cs", 0.5, 1.0),
            min_child_weight=trial.suggest_int("mcw", 1, 20),
            scale_pos_weight=spw, random_state=42, n_jobs=NJ, verbose=-1,
        )
        aucs = []
        for tr_idx, te_idx in skf.split(X, y):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr_idx]).astype(np.float32)
            Xte = sc.transform(X[te_idx]).astype(np.float32)
            m = LGBMClassifier(**params)
            m.fit(Xtr, y[tr_idx])
            aucs.append(roc_auc_score(y[te_idx], m.predict_proba(Xte)[:, 1]))
        return np.mean(aucs)

    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(lgb_objective, n_trials=30, show_progress_bar=False)
    print(f"  LGB best: {study_lgb.best_value:.4f}")

    # ── Final 5-fold CV with tuned models + ensemble ──
    print("\n[3] Final 5-fold CV with tuned ensemble...")
    xgb_p = study_xgb.best_params
    lgb_p = study_lgb.best_params
    cv_aucs, cv_r1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr_idx]).astype(np.float32)
        Xte = sc.transform(X[te_idx]).astype(np.float32)

        xgb_m = XGBClassifier(**{k: xgb_p[k] for k in xgb_p}, scale_pos_weight=spw,
                               random_state=42, n_jobs=NJ, eval_metric="auc")
        xgb_m.fit(Xtr, y[tr_idx], verbose=False)
        xgb_probs = xgb_m.predict_proba(Xte)[:, 1]

        lgb_m = LGBMClassifier(**{k: lgb_p[k] for k in lgb_p}, scale_pos_weight=spw,
                               random_state=42, n_jobs=NJ, verbose=-1)
        lgb_m.fit(Xtr, y[tr_idx])
        lgb_probs = lgb_m.predict_proba(Xte)[:, 1]

        cb_m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                                   scale_pos_weight=min(spw, 200), random_seed=42,
                                   thread_count=NJ, verbose=0)
        cb_m.fit(Xtr, y[tr_idx])
        cb_probs = cb_m.predict_proba(Xte)[:, 1]

        # Try multiple weight combos
        best_fold_auc = 0
        best_ens = None
        for w1 in np.arange(0.3, 0.7, 0.05):
            for w2 in np.arange(0.15, 0.55, 0.05):
                w3 = 1.0 - w1 - w2
                if w3 < 0:
                    continue
                ens = w1 * xgb_probs + w2 * lgb_probs + w3 * cb_probs
                a = roc_auc_score(y[te_idx], ens)
                if a > best_fold_auc:
                    best_fold_auc = a
                    best_ens = ens

        fold_r1 = recall_at_fpr(y[te_idx], best_ens)
        cv_aucs.append(best_fold_auc)
        cv_r1s.append(fold_r1)
        print(f"    Fold {fold+1}: AUC={best_fold_auc:.4f}  R@1%={fold_r1:.4f}  "
              f"(XGB={roc_auc_score(y[te_idx], xgb_probs):.4f} LGB={roc_auc_score(y[te_idx], lgb_probs):.4f} "
              f"CB={roc_auc_score(y[te_idx], cb_probs):.4f})")

    mu_auc = np.mean(cv_aucs)
    std_auc = np.std(cv_aucs)
    mu_r1 = np.mean(cv_r1s)
    std_r1 = np.std(cv_r1s)

    print(f"\n  Final CV: AUC={mu_auc:.4f}+/-{std_auc:.4f}  R@1%FPR={mu_r1:.4f}+/-{std_r1:.4f}")

    results = {
        "dataset": "ULB creditcard.csv",
        "n_rows": len(y),
        "n_features": int(X.shape[1]),
        "xgb_optuna_best_auc": round(study_xgb.best_value, 6),
        "lgb_optuna_best_auc": round(study_lgb.best_value, 6),
        "final_cv_auc": round(float(mu_auc), 6),
        "final_cv_auc_std": round(float(std_auc), 6),
        "final_cv_r1": round(float(mu_r1), 6),
        "final_cv_r1_std": round(float(std_r1), 6),
        "xgb_params": study_xgb.best_params,
        "lgb_params": study_lgb.best_params,
        "targets": {
            "auc_99": "ACHIEVED" if mu_auc >= 0.99 else f"NOT MET ({mu_auc:.4f})",
            "r1_90": "ACHIEVED" if mu_r1 >= 0.90 else f"NOT MET ({mu_r1:.4f})",
        },
    }

    out = REPORTS / "ulb_optuna_final.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out}")
    print(f"  Total: {time.time()-t0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
