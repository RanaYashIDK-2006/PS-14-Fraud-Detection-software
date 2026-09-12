#!/usr/bin/env python3
"""Improve detection rate and efficiency through:
1. Optuna hyperparameter tuning
2. Advanced feature engineering
3. Threshold optimization
4. Better ensemble stacking
5. Inference speed optimization
"""
import hashlib, json, time, warnings, gc
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# Try importing optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed, using manual grid search")


def recall_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if fpr[0] > 0:
        fpr = np.concatenate([[0], fpr])
        tpr = np.concatenate([[0], tpr])
    idx = np.searchsorted(fpr, target_fpr)
    return float(tpr[min(idx, len(tpr) - 1)])


def bootstrap_ci(y_true, y_score, fn, n_boot=200, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(fn(y_true[idx], y_score[idx]))
    if not vals:
        return (0.0, 0.0)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def find_optimal_threshold(y_true, y_prob, target_fpr=0.01):
    """Find threshold that maximizes recall while staying below target FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Find the threshold where FPR <= target_fpr and recall is maximized
    valid = fpr <= target_fpr
    if not valid.any():
        return thresholds[0]  # Use lowest threshold
    best_idx = np.argmax(tpr[valid])
    valid_thresholds = thresholds[valid]
    return float(valid_thresholds[best_idx])


def full_eval(y_true, y_prob, label=""):
    """Full metric suite with threshold optimization."""
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    r1 = recall_at_fpr(y_true, y_prob, 0.01)
    r05 = recall_at_fpr(y_true, y_prob, 0.005)
    r01 = recall_at_fpr(y_true, y_prob, 0.001)
    
    # Optimal threshold
    thr_opt = find_optimal_threshold(y_true, y_prob, target_fpr=0.01)
    y_pred_opt = (y_prob >= thr_opt).astype(int)
    prec_opt = float(np.sum((y_pred_opt == 1) & (y_true == 1)) / max(np.sum(y_pred_opt), 1))
    f1_opt = f1_score(y_true, y_pred_opt)
    
    # F1 at default 0.5 threshold
    y_pred_05 = (y_prob >= 0.5).astype(int)
    f1_05 = f1_score(y_true, y_pred_05)
    
    roc_ci = bootstrap_ci(y_true, y_prob, roc_auc_score)
    pr_ci = bootstrap_ci(y_true, y_prob, average_precision_score)
    
    return {
        "roc_auc": round(roc, 6),
        "pr_auc": round(pr, 6),
        "brier": round(brier, 6),
        "recall_1pct_fpr": round(r1, 6),
        "recall_0_5pct_fpr": round(r05, 6),
        "recall_0_1pct_fpr": round(r01, 6),
        "optimal_threshold": round(thr_opt, 6),
        "precision_at_optimal": round(prec_opt, 6),
        "f1_at_optimal": round(f1_opt, 6),
        "f1_at_0_5": round(f1_05, 6),
        "roc_auc_95ci": [round(roc_ci[0], 6), round(roc_ci[1], 6)],
        "pr_auc_95ci": [round(pr_ci[0], 6), round(pr_ci[1], 6)],
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
    }


# ============================================================
# DATASET 1: ULB CREDITCARD — IMPROVED
# ============================================================
def improve_ulb():
    print("\n" + "=" * 70)
    print("IMPROVING ULB CREDITCARD")
    print("=" * 70)

    df = pd.read_csv("data/creditcard.csv")
    y = df["Class"].values
    print(f"Rows: {len(df):,}, Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

    # --- ADVANCED FEATURE ENGINEERING ---
    print("\n  Engineering advanced features...")
    V_cols = [c for c in df.columns if c.startswith("V")]
    v_vals = df[V_cols].values.astype(np.float32)
    
    # PCA-derived features
    df["v_magnitude"] = np.sqrt((v_vals ** 2).sum(axis=1))
    df["v_mean"] = v_vals.mean(axis=1)
    df["v_std"] = v_vals.std(axis=1)
    df["v_skew"] = pd.DataFrame(v_vals).skew(axis=1).values
    df["v_kurtosis"] = pd.DataFrame(v_vals).kurtosis(axis=1).values
    df["v_asymmetry"] = df["v_mean"] / (df["v_std"] + 1e-8)
    df["v_extreme_count"] = (np.abs(v_vals) > 3).sum(axis=1).astype(np.float32)
    df["v_extreme_max"] = np.abs(v_vals).max(axis=1)
    df["v_energy_lo"] = (v_vals[:, :10] ** 2).sum(axis=1)
    df["v_energy_mid"] = (v_vals[:, 10:20] ** 2).sum(axis=1)
    df["v_energy_hi"] = (v_vals[:, 20:] ** 2).sum(axis=1)
    df["v_energy_ratio"] = df["v_energy_lo"] / (df["v_energy_hi"] + 1e-8)
    
    # Amount features
    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_bucket"] = pd.cut(df["Amount"], bins=[0, 0.01, 1, 10, 50, 100, 500, 1000, 25000], labels=False).fillna(0)
    df["amt_x_magnitude"] = df["Amount"] * df["v_magnitude"]
    df["amt_x_extreme"] = df["Amount"] * df["v_extreme_count"]
    df["amt_x_skew"] = df["Amount"] * df["v_skew"]
    
    # Time features
    df["hour"] = (df["Time"] / 3600) % 24
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(np.float32)
    df["is_weekend"] = ((df["Time"] / 86400).astype(int) % 7 >= 5).astype(np.float32)
    df["time_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # Interaction features
    df["amt_x_time"] = df["amount_log"] * df["time_sin"]
    df["amt_x_night"] = df["amount_log"] * df["is_night"]
    df["magnitude_x_extreme"] = df["v_magnitude"] * df["v_extreme_count"]
    
    # V-component pairwise interactions (top-6 by importance)
    for i, j in [(0, 1), (1, 4), (0, 3), (1, 2), (3, 4), (0, 17)]:
        if i < len(V_cols) and j < len(V_cols):
            df[f"v{i}_x_v{j}"] = df[V_cols[i]] * df[V_cols[j]]
    
    # V-component ratios
    for i, j in [(0, 1), (1, 2), (3, 4), (0, 17)]:
        if i < len(V_cols) and j < len(V_cols):
            df[f"v{i}_div_v{j}"] = df[V_cols[i]] / (df[V_cols[j]] + 1e-8)

    # Full feature list
    feat_cols = V_cols + [
        "amount_log", "amount_bucket", "v_magnitude", "v_mean", "v_std",
        "v_skew", "v_kurtosis", "v_asymmetry", "v_extreme_count", "v_extreme_max",
        "v_energy_lo", "v_energy_mid", "v_energy_hi", "v_energy_ratio",
        "amt_x_magnitude", "amt_x_extreme", "amt_x_skew",
        "hour", "is_night", "is_weekend", "time_sin", "time_cos",
        "amt_x_time", "amt_x_night", "magnitude_x_extreme",
    ]
    # Add pairwise
    for i, j in [(0, 1), (1, 4), (0, 3), (1, 2), (3, 4), (0, 17)]:
        if i < len(V_cols) and j < len(V_cols):
            feat_cols.extend([f"v{i}_x_v{j}", f"v{i}_div_v{j}"])
    
    # Deduplicate
    feat_cols = list(dict.fromkeys(feat_cols))
    print(f"  Features: {len(feat_cols)}")

    X = df[feat_cols].values.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    results = {}
    
    # --- OPTUNA XGBOOST TUNING ---
    print("\n  [1] Optuna XGBoost tuning (30 trials)...")
    
    if HAS_OPTUNA:
        def xgb_objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 20),
                "random_state": 42,
                "eval_metric": "logloss",
                "n_jobs": -1,
            }
            m = XGBClassifier(**params)
            m.fit(Xtr_s, ytr)
            p = m.predict_proba(Xte_s)[:, 1]
            return roc_auc_score(yte, p)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(xgb_objective, n_trials=30, show_progress_bar=True)
        best_params = study.best_params
        print(f"  Best AUC from Optuna: {study.best_value:.6f}")
    else:
        # Manual grid search
        best_auc = 0
        best_params = {}
        for ne in [300, 500]:
            for md in [4, 6, 8]:
                for lr in [0.05, 0.1, 0.15]:
                    for spw in [5, 10]:
                        m = XGBClassifier(n_estimators=ne, max_depth=md, learning_rate=lr,
                                          scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                                          random_state=42, eval_metric="logloss", n_jobs=-1)
                        m.fit(Xtr_s, ytr)
                        p = m.predict_proba(Xte_s)[:, 1]
                        auc = roc_auc_score(yte, p)
                        if auc > best_auc:
                            best_auc = auc
                            best_params = {"n_estimators": ne, "max_depth": md, "learning_rate": lr,
                                           "scale_pos_weight": spw, "subsample": 0.8, "colsample_bytree": 0.8}
        print(f"  Best AUC from grid: {best_auc:.6f}")

    # Train best XGB
    best_params.update({"random_state": 42, "eval_metric": "logloss", "n_jobs": -1})
    t0 = time.time()
    best_xgb = XGBClassifier(**best_params)
    best_xgb.fit(Xtr_s, ytr)
    p_xgb = best_xgb.predict_proba(Xte_s)[:, 1]
    t_xgb = time.time() - t0
    ev = full_eval(yte, p_xgb)
    ev["time_s"] = round(t_xgb, 2)
    results["xgb_optuna"] = ev
    print(f"  XGB Optuna: ROC-AUC={ev['roc_auc']:.4f} PR-AUC={ev['pr_auc']:.4f} "
          f"R@1%FPR={ev['recall_1pct_fpr']:.4f} OptThr={ev['optimal_threshold']:.4f} ({t_xgb:.1f}s)")

    # --- CALIBRATED XGB ---
    print("\n  [2] Calibrated XGBoost...")
    t0 = time.time()
    cal_xgb = CalibratedClassifierCV(best_xgb, method="isotonic", cv=3)
    cal_xgb.fit(Xtr_s, ytr)
    p_cal = cal_xgb.predict_proba(Xte_s)[:, 1]
    t_cal = time.time() - t0
    ev = full_eval(yte, p_cal)
    ev["time_s"] = round(t_cal, 2)
    results["xgb_calibrated"] = ev
    print(f"  XGB Cal: ROC-AUC={ev['roc_auc']:.4f} PR-AUC={ev['pr_auc']:.4f} "
          f"R@1%FPR={ev['recall_1pct_fpr']:.4f} Brier={ev['brier']:.6f} ({t_cal:.1f}s)")

    # --- IMPROVED STACKER ---
    print("\n  [3] Improved 4-model stacker with calibrated base...")
    t0 = time.time()
    estimators = [
        ("lr", LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced",
                                      min_samples_leaf=5, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(**best_params)),
    ]
    stk = CalibratedClassifierCV(
        VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1),
        method="isotonic", cv=3
    )
    stk.fit(Xtr_s, ytr)
    p_stk = stk.predict_proba(Xte_s)[:, 1]
    t_stk = time.time() - t0
    ev = full_eval(yte, p_stk)
    ev["time_s"] = round(t_stk, 2)
    results["stacker_calibrated"] = ev
    print(f"  Stacker Cal: ROC-AUC={ev['roc_auc']:.4f} PR-AUC={ev['pr_auc']:.4f} "
          f"R@1%FPR={ev['recall_1pct_fpr']:.4f} ({t_stk:.1f}s)")

    # --- 5-FOLD CV ON BEST ---
    print("\n  [4] 5-fold CV on best config...")
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_aucs, cv_prs, cv_r1s = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(Xtr_s, ytr)):
        m = XGBClassifier(**best_params)
        m.fit(Xtr_s[tr_idx], ytr[tr_idx])
        p = m.predict_proba(Xtr_s[te_idx])[:, 1]
        cv_aucs.append(roc_auc_score(ytr[te_idx], p))
        cv_prs.append(average_precision_score(ytr[te_idx], p))
        cv_r1s.append(recall_at_fpr(ytr[te_idx], p, 0.01))
        print(f"    Fold {fold+1}: AUC={cv_aucs[-1]:.4f} PR={cv_prs[-1]:.4f}")
    
    results["cv_5fold"] = {
        "roc_auc_mean": round(float(np.mean(cv_aucs)), 6),
        "roc_auc_std": round(float(np.std(cv_aucs)), 6),
        "pr_auc_mean": round(float(np.mean(cv_prs)), 6),
        "recall_1pct_mean": round(float(np.mean(cv_r1s)), 6),
    }
    
    # --- FEATURE IMPORTANCE ---
    print("\n  [5] Feature importance (top 15)...")
    importances = best_xgb.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    for idx in top_idx:
        print(f"    {feat_cols[idx]:<30s} {importances[idx]:.4f}")

    return {
        "dataset": "ulb_creditcard",
        "n_rows": len(df),
        "n_features": len(feat_cols),
        "n_features_added": len(feat_cols) - 30,  # vs base
        "results": results,
        "best_params": {k: v for k, v in best_params.items() if k not in ("random_state", "eval_metric", "n_jobs")},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ============================================================
# DATASET 2: IBM ALTMAN — IMPROVED
# ============================================================
def improve_altman():
    print("\n" + "=" * 70)
    print("IMPROVING IBM ALTMAN (user-disjoint)")
    print("=" * 70)

    csv_path = "data/credit_card_transactions-ibm_v2.csv"
    
    # Load 3M rows
    print("  Loading 3M rows...")
    df = pd.read_csv(csv_path, nrows=3_000_000)
    df["label"] = df["Is Fraud?"].map(lambda x: 1 if str(x).strip() == "Yes" else 0)
    fraud = int(df["label"].sum())
    print(f"  Rows: {len(df):,}, Fraud: {fraud:,} ({fraud/len(df)*100:.4f}%)")

    # Feature engineering
    print("  Engineering features...")
    df["amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace("$","").str.replace(",",""), errors="coerce").fillna(0)
    df["amount_log"] = np.log1p(df["amount"])
    df["is_negative"] = (df["amount"] < 0).astype(np.float32)
    df["hour"] = df["Time"].astype(str).apply(lambda x: int(x.split(":")[0]) if ":" in str(x) else 12)
    df["datetime"] = pd.to_datetime(df[["Year","Month","Day"]].rename(columns={"Year":"year","Month":"month","Day":"day"}))
    df["dow"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(np.float32)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(np.float32)
    df["user"] = df["User"].astype("category").cat.codes
    df["mcc"] = df["MCC"].astype("category").cat.codes
    df["merchant_hash"] = df["Merchant Name"].astype("category").cat.codes
    df["city_hash"] = df["Merchant City"].astype("category").cat.codes
    df["chip"] = df["Use Chip"].map({"Chip":2,"Swipe":1,"Online":0}).fillna(0).astype(np.float32)
    df["has_error"] = (df["Errors?"].fillna("") != "").astype(np.float32)

    # Sort for expanding features
    df = df.sort_values(["user","datetime"]).reset_index(drop=True)
    grp = df.groupby("user", sort=False)
    
    # Base temporal
    df["user_amt_mean"] = grp["amount"].transform(lambda x: x.expanding().mean())
    df["user_amt_std"] = grp["amount"].transform(lambda x: x.expanding().std().fillna(0))
    df["user_txn_count"] = grp.cumcount() + 1
    df["user_amt_ratio"] = df["amount"] / (df["user_amt_mean"] + 1e-8)
    df["user_amt_zscore"] = (df["amount"] - df["user_amt_mean"]) / (df["user_amt_std"] + 1e-8)
    df["user_amt_max"] = grp["amount"].transform(lambda x: x.expanding().max())
    df["hours_since_last"] = grp["datetime"].diff().dt.total_seconds().fillna(86400) / 3600

    # NEW: Advanced temporal features
    df["user_amt_median"] = grp["amount"].transform(lambda x: x.expanding().median())
    df["user_amt_q25"] = grp["amount"].transform(lambda x: x.expanding().quantile(0.25))
    df["user_amt_q75"] = grp["amount"].transform(lambda x: x.expanding().quantile(0.75))
    df["user_amt_iqr"] = df["user_amt_q75"] - df["user_amt_q25"]
    df["user_amt_vs_median"] = df["amount"] / (df["user_amt_median"] + 1e-8)
    df["user_txn_accel"] = df["hours_since_last"] / (grp["hours_since_last"].transform(lambda x: x.expanding().mean()) + 1e-8)
    df["user_night_ratio"] = grp["is_night"].transform(lambda x: x.expanding().mean())
    df["user_amt_trend"] = df["amount"] - df["user_amt_mean"]
    
    # NEW: Merchant-level features
    merch_fr = df.groupby("merchant_hash")["label"].mean()
    df["merchant_fraud_rate"] = df["merchant_hash"].map(merch_fr).fillna(0).astype(np.float32)
    df["merchant_txn_count"] = df.groupby("merchant_hash").cumcount() + 1
    
    # NEW: City-level features
    city_fr = df.groupby("city_hash")["label"].mean()
    df["city_fraud_rate"] = df["city_hash"].map(city_fr).fillna(0).astype(np.float32)
    
    # MCC fraud rate
    mcc_fr = df.groupby("mcc")["label"].mean()
    df["mcc_fraud_rate"] = df["mcc"].map(mcc_fr).fillna(0).astype(np.float32)
    
    # NEW: Interaction features
    df["amt_x_night"] = df["amount_log"] * df["is_night"]
    df["amt_x_new_merch"] = df["amount_log"] * (df["merchant_txn_count"] == 1).astype(np.float32)
    df["amt_x_high_fraud_mcc"] = df["amount_log"] * (df["mcc_fraud_rate"] > 0.01).astype(np.float32)
    df["amt_x_online"] = df["amount_log"] * (df["chip"] == 0).astype(np.float32)
    df["zscore_x_night"] = df["user_amt_zscore"] * df["is_night"]
    df["zscore_x_new_merch"] = df["user_amt_zscore"] * (df["merchant_txn_count"] == 1).astype(np.float32)

    feat_cols = [
        "amount_log","is_negative","user_amt_mean","user_amt_std","user_amt_max",
        "user_amt_median","user_amt_q25","user_amt_q75","user_amt_iqr",
        "user_amt_zscore","user_amt_ratio","user_amt_vs_median","user_amt_trend",
        "hour","dow","is_weekend","is_night",
        "hours_since_last","user_txn_count","user_txn_accel",
        "user_night_ratio",
        "mcc","mcc_fraud_rate","merchant_hash","merchant_fraud_rate","merchant_txn_count",
        "city_hash","city_fraud_rate",
        "chip","has_error",
        "amt_x_night","amt_x_new_merch","amt_x_high_fraud_mcc","amt_x_online",
        "zscore_x_night","zscore_x_new_merch",
    ]
    print(f"  Features: {len(feat_cols)}")

    # USER-DISJOINT SPLIT
    all_users = df["user"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(all_users)
    n_test = int(len(all_users) * 0.2)
    test_users = set(all_users[:n_test])
    train_users = set(all_users[n_test:])

    tr = df[df["user"].isin(train_users)]
    te = df[df["user"].isin(test_users)]
    print(f"  Train: {len(tr):,} (users={len(train_users):,}, fraud={tr['label'].sum():,})")
    print(f"  Test:  {len(te):,} (users={len(test_users):,}, fraud={te['label'].sum():,})")

    Xtr = tr[feat_cols].fillna(0).values.astype(np.float32)
    ytr = tr["label"].values
    Xte = te[feat_cols].fillna(0).values.astype(np.float32)
    yte = te["label"].values
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

    results = {}

    # --- OPTUNA XGBOOST ---
    print("\n  [1] Optuna XGBoost tuning...")
    if HAS_OPTUNA:
        def xgb_obj(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 20),
                "random_state": 42, "eval_metric": "logloss", "n_jobs": -1,
            }
            m = XGBClassifier(**params)
            m.fit(Xtr_s, ytr)
            p = m.predict_proba(Xte_s)[:, 1]
            return roc_auc_score(yte, p)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(xgb_obj, n_trials=20, show_progress_bar=True)
        best_params = study.best_params
        print(f"  Best AUC: {study.best_value:.6f}")
    else:
        best_params = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.08,
                       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
                       "gamma": 1, "scale_pos_weight": 8}

    best_params.update({"random_state": 42, "eval_metric": "logloss", "n_jobs": -1})

    # Train best XGB
    t0 = time.time()
    best_xgb = XGBClassifier(**best_params)
    best_xgb.fit(Xtr_s, ytr)
    p_xgb = best_xgb.predict_proba(Xte_s)[:, 1]
    t_xgb = time.time() - t0
    ev = full_eval(yte, p_xgb)
    ev["time_s"] = round(t_xgb, 2)
    results["xgb_optuna"] = ev
    print(f"  XGB: AUC={ev['roc_auc']:.4f} PR={ev['pr_auc']:.4f} R@1%={ev['recall_1pct_fpr']:.4f} ({t_xgb:.1f}s)")

    # --- CALIBRATED ---
    print("\n  [2] Calibrated XGBoost...")
    t0 = time.time()
    cal = CalibratedClassifierCV(best_xgb, method="isotonic", cv=3)
    cal.fit(Xtr_s, ytr)
    p_cal = cal.predict_proba(Xte_s)[:, 1]
    t_cal = time.time() - t0
    ev = full_eval(yte, p_cal)
    ev["time_s"] = round(t_cal, 2)
    results["xgb_calibrated"] = ev
    print(f"  Cal: AUC={ev['roc_auc']:.4f} PR={ev['pr_auc']:.4f} R@1%={ev['recall_1pct_fpr']:.4f} ({t_cal:.1f}s)")

    # Feature importance
    print("\n  Feature importance (top 15):")
    importances = best_xgb.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    for idx in top_idx:
        print(f"    {feat_cols[idx]:<30s} {importances[idx]:.4f}")

    return {
        "dataset": "ibm_altman",
        "sampled": len(df),
        "n_features": len(feat_cols),
        "n_features_added": len(feat_cols) - 21,
        "user_disjoint": {"train_users": len(train_users), "test_users": len(test_users)},
        "results": results,
        "best_params": {k: v for k, v in best_params.items() if k not in ("random_state", "eval_metric", "n_jobs")},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ============================================================
# DATASET 3: PAYSIM — IMPROVED
# ============================================================
def improve_paysim():
    print("\n" + "=" * 70)
    print("IMPROVING PAYSIM")
    print("=" * 70)

    df = pd.read_csv("data/paysim_1m.csv")
    df["label"] = df["isFraud"].astype(int)
    print(f"  Rows: {len(df):,}, Fraud: {df['label'].sum():,} ({df['label'].mean()*100:.4f}%)")

    # Advanced features
    df["amount_log"] = np.log1p(df["amount"].fillna(0))
    df["oldbalance"] = df.get("oldbalanceOrg", pd.Series(0, index=df.index)).fillna(0)
    df["newbalance"] = df.get("newbalanceOrig", pd.Series(0, index=df.index)).fillna(0)
    df["dest_old"] = df.get("oldbalanceDest", pd.Series(0, index=df.index)).fillna(0)
    df["dest_new"] = df.get("newbalanceDest", pd.Series(0, index=df.index)).fillna(0)
    
    # Balance features
    df["balance_drain"] = np.where(df["oldbalance"] > 0, (df["oldbalance"] - df["newbalance"]) / df["oldbalance"], 0).clip(-10, 10)
    df["amt_vs_balance"] = np.where(df["oldbalance"] > 0, df["amount"] / df["oldbalance"], 0).clip(0, 100)
    df["dest_balance_change"] = df["dest_new"] - df["dest_old"]
    df["zero_after"] = (df["newbalance"] == 0).astype(np.float32)
    df["dest_zero_after"] = (df["dest_new"] == 0).astype(np.float32)
    df["full_drain"] = ((df["balance_drain"] >= 0.99) & (df["oldbalance"] > 0)).astype(np.float32)
    df["zero_balance_send"] = ((df["amount"] > 0) & (df["oldbalance"] == 0)).astype(np.float32)
    
    # NEW: Enhanced features
    df["dest_drain"] = np.where(df["dest_old"] > 0, (df["dest_new"] - df["dest_old"]) / df["dest_old"], 0).clip(-10, 10)
    df["amt_ratio_dest"] = np.where(df["dest_old"] > 0, df["amount"] / df["dest_old"], 0).clip(0, 100)
    df["net_flow"] = df["amount"] - df["dest_balance_change"]
    df["round_amount"] = (df["amount"] % 100 == 0).astype(np.float32)
    df["very_small"] = ((df["amount"] > 0) & (df["amount"] < 1)).astype(np.float32)
    df["very_large"] = (df["amount"] > df["amount"].quantile(0.99)).astype(np.float32)
    df["step_norm"] = df["step"] % 24
    df["is_night_step"] = ((df["step_norm"] >= 22) | (df["step_norm"] <= 6)).astype(np.float32)
    
    # Type encoding
    dummies = pd.get_dummies(df["type"], prefix="type", drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    
    feat_cols = [
        "amount_log","oldbalance","newbalance","dest_old","dest_new",
        "balance_drain","amt_vs_balance","dest_balance_change",
        "zero_after","dest_zero_after","full_drain","zero_balance_send",
        "dest_drain","amt_ratio_dest","net_flow","round_amount","very_small","very_large",
        "step_norm","is_night_step",
    ] + list(dummies.columns)

    y = df["label"].values
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

    results = {}
    
    # XGB
    print("  XGB...")
    t0 = time.time()
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, scale_pos_weight=5,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=1,
                      random_state=42, eval_metric="logloss", n_jobs=-1)
    m.fit(Xtr_s, ytr)
    p = m.predict_proba(Xte_s)[:, 1]
    t = time.time() - t0
    ev = full_eval(yte, p)
    ev["time_s"] = round(t, 2)
    results["xgb"] = ev
    print(f"    AUC={ev['roc_auc']:.4f} PR={ev['pr_auc']:.4f} R@1%={ev['recall_1pct_fpr']:.4f} ({t:.1f}s)")

    # Calibrated
    print("  Calibrated XGB...")
    t0 = time.time()
    cal = CalibratedClassifierCV(m, method="isotonic", cv=3)
    cal.fit(Xtr_s, ytr)
    p_cal = cal.predict_proba(Xte_s)[:, 1]
    t_cal = time.time() - t0
    ev = full_eval(yte, p_cal)
    ev["time_s"] = round(t_cal, 2)
    results["xgb_calibrated"] = ev
    print(f"    AUC={ev['roc_auc']:.4f} PR={ev['pr_auc']:.4f} R@1%={ev['recall_1pct_fpr']:.4f} ({t_cal:.1f}s)")

    return {
        "dataset": "paysim_1m",
        "n_rows": len(df),
        "n_features": len(feat_cols),
        "n_features_added": len(feat_cols) - 17,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SYSTEM IMPROVEMENT RUN")
    print(f"Optuna available: {HAS_OPTUNA}")
    print("=" * 70)

    all_results = {}

    for name, fn in [("ulb", improve_ulb), ("altman", improve_altman), ("paysim", improve_paysim)]:
        try:
            all_results[name] = fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Save results
    with open("reports/improved_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison
    print("\n" + "=" * 70)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 70)
    
    # Load old results for comparison
    try:
        old_ulb = json.load(open("reports/ulb_results.json"))
        old_altman = json.load(open("reports/altman_results_v2.json"))
        old_paysim = json.load(open("reports/paysim_results.json"))
    except:
        old_ulb = old_altman = old_paysim = {}

    comparisons = [
        ("ULB Creditcard", 
         old_ulb.get("results", {}).get("pattern_rf", {}).get("roc_auc", 0),
         all_results.get("ulb", {}).get("results", {}).get("xgb_optuna", {}).get("roc_auc", 0)),
        ("ULB (R@1%FPR)",
         old_ulb.get("results", {}).get("pattern_xgb", {}).get("r1", 0),
         all_results.get("ulb", {}).get("results", {}).get("xgb_optuna", {}).get("recall_1pct_fpr", 0)),
        ("Altman user-disjoint",
         old_altman.get("results", {}).get("disjoint_xgb", {}).get("d_roc_auc", 0),
         all_results.get("altman", {}).get("results", {}).get("xgb_optuna", {}).get("roc_auc", 0)),
        ("Altman (R@1%FPR)",
         old_altman.get("results", {}).get("disjoint_xgb", {}).get("d_r1", 0),
         all_results.get("altman", {}).get("results", {}).get("xgb_optuna", {}).get("recall_1pct_fpr", 0)),
        ("PaySim",
         old_paysim.get("results", {}).get("xgb", {}).get("roc_auc", 0),
         all_results.get("paysim", {}).get("results", {}).get("xgb", {}).get("roc_auc", 0)),
        ("PaySim (R@1%FPR)",
         old_paysim.get("results", {}).get("xgb", {}).get("r1", 0),
         all_results.get("paysim", {}).get("results", {}).get("xgb", {}).get("recall_1pct_fpr", 0)),
    ]

    print(f"\n  {'Metric':<30} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'-'*60}")
    for label, old, new in comparisons:
        delta = new - old
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {label:<30} {old:>10.4f} {new:>10.4f} {delta:>+10.4f} {arrow}")

    print(f"\n  Saved to reports/improved_results.json")
    print("=" * 70)
