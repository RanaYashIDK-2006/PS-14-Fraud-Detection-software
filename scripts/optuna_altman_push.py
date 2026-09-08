#!/usr/bin/env python3
"""Optuna hyperparameter search on Altman 35-feature model.

Objective: maximize temporal test AUC (train months 1-9, test months 10-12).
Searches: XGBoost, LightGBM, CatBoost hyperparameters + ensemble weights.
"""
import numpy as np
import pandas as pd
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")


def recall_at(y_true, y_score, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= fpr_target
    return float(tpr[mask][-1]) if mask.any() else 0.0


def load_and_engineer():
    """Load Altman data and engineer the 35 clean features."""
    rng = np.random.RandomState(42)
    chunks = []
    for chunk in pd.read_csv(
        DATA_DIR / "credit_card_transactions-ibm_v2.csv",
        low_memory=False, chunksize=500_000,
    ):
        fraud_mask = chunk["Is Fraud?"] == "Yes"
        is_legit = ~fraud_mask
        sample_legit = rng.random(len(chunk)) < 0.01
        chunks.append(chunk[fraud_mask | (is_legit & sample_legit)].copy())

    df = pd.concat(chunks, ignore_index=True)

    df["amt"] = df["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df["is_fraud"] = (df["Is Fraud?"] == "Yes").astype(int)
    df["hr"] = df["Time"].str.split(":").str[0].astype(int)
    df["mn"] = df["Time"].str.split(":").str[1].astype(int)
    df["dow"] = pd.to_datetime(df[["Year", "Month", "Day"]]).dt.dayofweek
    df["chip"] = (df["Use Chip"] == "Chip Transaction").astype(int)
    df["is_online"] = (df["Use Chip"] == "Online Transaction").astype(int)
    df["mcc_n"] = df["MCC"].fillna(0).astype(float)
    df["err"] = (df["Errors?"].fillna("") != "").astype(int)
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]].assign(
        hour=df["hr"], minute=df["mn"]))
    df = df.sort_values(["User", "datetime"]).reset_index(drop=True)

    F = pd.DataFrame()
    F["amt"] = df["amt"]
    F["log_amt"] = np.log1p(F["amt"])
    F["amt_sq"] = F["amt"] ** 2
    F["hr"] = df["hr"]
    F["mn"] = df["mn"]
    F["dow"] = df["dow"]
    F["Month"] = df["Month"]
    F["Day"] = df["Day"]
    F["hour_sin"] = np.sin(2 * np.pi * F["hr"] / 24)
    F["hour_cos"] = np.cos(2 * np.pi * F["hr"] / 24)
    F["is_night"] = ((F["hr"] < 6) | (F["hr"] > 22)).astype(int)
    F["is_business_hours"] = ((F["hr"] >= 9) & (F["hr"] <= 17)).astype(int)
    F["chip"] = df["chip"]
    F["is_online"] = df["is_online"]
    F["err"] = (df["err"] != 0).astype(int)
    F["mcc_n"] = df["mcc_n"]
    F["user_tx_count"] = df.groupby("User").cumcount()
    F["card_tx_count"] = df.groupby("Card").cumcount()
    F["user_avg_amt"] = df.groupby("User")["amt"].transform(lambda x: x.expanding().mean().shift(1))
    F["amt_vs_user_avg"] = F["amt"] / (F["user_avg_amt"] + 1e-6)
    F["amt_zscore"] = (F["amt"] - F["user_avg_amt"]) / (
        df.groupby("User")["amt"].transform(lambda x: x.expanding().std().shift(1)) + 1e-6)
    F["merch_tx_count"] = df.groupby("Merchant Name").cumcount()
    F["user_fraud_rate"] = df.groupby("User")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(0.001)
    F["merch_fraud_rate"] = df.groupby("Merchant Name")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(0.001)
    F["city_fraud_rate"] = df.groupby("Merchant City")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1)).fillna(0.001)
    F["has_zip"] = df["Zip"].notna().astype(int)
    F["has_state"] = df["Merchant State"].notna().astype(int)
    F["is_online_or_no_state"] = ((F["is_online"] == 1) | (F["has_state"] == 0)).astype(int)
    F["high_amt"] = (F["amt"] > F["user_avg_amt"] * 2).astype(int)
    F["very_high_amt"] = (F["amt"] > F["user_avg_amt"] * 5).astype(int)
    F["amt_x_hr"] = F["amt"] * F["hr"]
    F["amt_x_mcc"] = F["amt"] * F["mcc_n"]
    F["amt_x_chip"] = F["amt"] * F["chip"]
    F["amt_x_online"] = F["amt"] * F["is_online"]
    F["amt_x_night"] = F["amt"] * F["is_night"]
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Temporal split
    train_mask = df["Month"] <= 9
    test_mask = df["Month"] > 9
    Xtr = F.loc[train_mask].values.astype(np.float32)
    ytr = df.loc[train_mask, "is_fraud"].values
    Xte = F.loc[test_mask].values.astype(np.float32)
    yte = df.loc[test_mask, "is_fraud"].values

    return Xtr, ytr, Xte, yte


def main():
    import optuna
    from optuna.samplers import TPESampler
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    from sklearn.model_selection import StratifiedKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 70)
    print("OPTUNA HYPERPARAMETER SEARCH — ALTMAN 35-FEATURE MODEL")
    print("=" * 70)
    t0 = time.time()

    print("\n[1/3] Loading data...")
    Xtr, ytr, Xte, yte = load_and_engineer()
    print(f"  Train: {len(Xtr):,} ({ytr.sum():,} fraud)")
    print(f"  Test:  {len(Xte):,} ({yte.sum():,} fraud)")

    spw = (1 - ytr.mean()) / ytr.mean()
    print(f"  scale_pos_weight: {spw:.1f}")

    # Subsample for Optuna speed: keep all fraud, sample 50K legit
    fraud_idx = np.where(ytr == 1)[0]
    legit_idx = np.where(ytr == 0)[0]
    rng_sub = np.random.RandomState(42)
    n_sub = min(50_000, len(legit_idx))
    sub_legit = rng_sub.choice(legit_idx, n_sub, replace=False)
    sub_idx = np.concatenate([fraud_idx, sub_legit])
    Xtr_sub = Xtr[sub_idx]
    ytr_sub = ytr[sub_idx]
    print(f"  Optuna subsample: {len(Xtr_sub):,} ({ytr_sub.sum():,} fraud)")

    def objective(trial):
        """Optimize XGB + LGB ensemble (2 models, faster)."""
        # XGBoost params
        xgb_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 400),
            "max_depth": trial.suggest_int("xgb_max_depth", 4, 8),
            "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("xgb_min_child", 1, 15),
            "reg_alpha": trial.suggest_float("xgb_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("xgb_lambda", 0.0, 5.0),
            "scale_pos_weight": min(spw, 20),
            "random_state": 42, "n_jobs": 4,
            "eval_metric": "auc", "use_label_encoder": False,
        }

        # LightGBM params
        lgb_params = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 400),
            "max_depth": trial.suggest_int("lgb_max_depth", 4, 10),
            "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("lgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("lgb_min_child", 1, 15),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 63),
            "reg_alpha": trial.suggest_float("lgb_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("lgb_lambda", 0.0, 5.0),
            "scale_pos_weight": min(spw, 20),
            "random_state": 42, "n_jobs": 4, "verbose": -1,
        }

        # Ensemble weight
        w_xgb = trial.suggest_float("w_xgb", 0.2, 0.8)
        w_lgb = 1.0 - w_xgb

        # 2-fold CV on subsample (fast)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        cv_aucs = []

        for tr_idx, va_idx in skf.split(Xtr_sub, ytr_sub):
            Xa, Xv = Xtr_sub[tr_idx], Xtr_sub[va_idx]
            ya, yv = ytr_sub[tr_idx], ytr_sub[va_idx]

            scaler = RobustScaler()
            Xa_s = scaler.fit_transform(Xa)
            Xv_s = scaler.transform(Xv)

            m_xgb = xgb.XGBClassifier(**xgb_params)
            m_xgb.fit(Xa_s, ya, verbose=False)

            m_lgb = lgb.LGBMClassifier(**lgb_params)
            m_lgb.fit(Xa_s, ya)

            p = w_xgb * m_xgb.predict_proba(Xv_s)[:, 1] + \
                w_lgb * m_lgb.predict_proba(Xv_s)[:, 1]

            cv_aucs.append(roc_auc_score(yv, p))

        return np.mean(cv_aucs)

    print("\n[2/3] Running Optuna search (30 trials, subsampled)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=8),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    print(f"\n  Best CV AUC: {study.best_value:.6f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=2)}")

    # Retrain with best params on full training set and evaluate on temporal test
    print("\n[3/3] Retraining with best params on temporal test...")
    bp = study.best_params
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    m_xgb = xgb.XGBClassifier(
        n_estimators=bp["xgb_n_estimators"], max_depth=bp["xgb_max_depth"],
        learning_rate=bp["xgb_lr"], subsample=bp["xgb_subsample"],
        colsample_bytree=bp["xgb_colsample"], min_child_weight=bp["xgb_min_child"],
        reg_alpha=bp["xgb_alpha"], reg_lambda=bp["xgb_lambda"],
        scale_pos_weight=min(spw, 20), random_state=42, n_jobs=4,
        eval_metric="auc", use_label_encoder=False,
    )
    m_xgb.fit(Xtr_s, ytr, verbose=False)

    m_lgb = lgb.LGBMClassifier(
        n_estimators=bp["lgb_n_estimators"], max_depth=bp["lgb_max_depth"],
        learning_rate=bp["lgb_lr"], subsample=bp["lgb_subsample"],
        colsample_bytree=bp["lgb_colsample"], min_child_weight=bp["lgb_min_child"],
        num_leaves=bp["lgb_num_leaves"], reg_alpha=bp["lgb_alpha"], reg_lambda=bp["lgb_lambda"],
        scale_pos_weight=min(spw, 20), random_state=42, n_jobs=4, verbose=-1,
    )
    m_lgb.fit(Xtr_s, ytr)

    w_xgb = bp["w_xgb"]
    w_lgb = 1.0 - w_xgb
    p_te = w_xgb * m_xgb.predict_proba(Xte_s)[:, 1] + \
           w_lgb * m_lgb.predict_proba(Xte_s)[:, 1]

    test_auc = roc_auc_score(yte, p_te)
    test_r1 = recall_at(yte, p_te, 0.01)

    print(f"\n  Temporal test: AUC={test_auc:.6f} R@1%FPR={test_r1:.4f}")
    print(f"  Weights: XGB={w_xgb:.3f} LGB={w_lgb:.3f}")

    # Compare vs baseline
    baseline_auc = 0.99168  # from manifest
    delta = test_auc - baseline_auc
    print(f"\n  Baseline:  {baseline_auc:.6f}")
    print(f"  Optimized: {test_auc:.6f}")
    print(f"  Delta:     {delta:+.6f} ({delta*100:+.4f}%)")

    if test_auc >= 0.995:
        print(f"\n  ✅ TARGET MET: {test_auc:.4f} >= 99.5%")
    elif test_auc > baseline_auc:
        print(f"\n  📈 IMPROVED: {test_auc:.4f} > {baseline_auc:.4f}")
    else:
        print(f"\n  ⚠️  No improvement over baseline")

    # Save report
    elapsed = time.time() - t0
    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_auc": baseline_auc,
        "optimized_auc": round(test_auc, 6),
        "delta": round(delta, 6),
        "test_r1": round(test_r1, 6),
        "weights": {"xgb": round(w_xgb, 4), "lgb": round(w_lgb, 4)},
        "best_params": study.best_params,
        "n_trials": 30,
        "elapsed_s": round(elapsed, 1),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "optuna_altman_results.json").write_text(json.dumps(report, indent=2))

    print(f"\n  Report: reports/optuna_altman_results.json")
    print(f"  Time: {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
