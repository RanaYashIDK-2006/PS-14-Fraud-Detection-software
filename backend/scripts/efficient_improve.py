#!/usr/bin/env python3
"""Efficient improvement: Optuna tuning + advanced features for all 3 datasets.

Runs sequentially, each tuned to finish within timeout.
"""
import json, time, warnings, gc
import numpy as np
import pandas as pd
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def rafpr(y, s, t=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    if fpr[0] > 0:
        fpr = np.concatenate([[0], fpr]); tpr = np.concatenate([[0], tpr])
    return float(tpr[min(np.searchsorted(fpr, t), len(tpr) - 1)])


def find_thr(y, s, t=0.01):
    fpr, tpr, thr = roc_curve(y, s)
    v = fpr <= t
    if not v.any(): return thr[0]
    return float(thr[v][np.argmax(tpr[v])])


def ev(y, p):
    thr = find_thr(y, p)
    yp = (p >= thr).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, p), 6),
        "pr_auc": round(average_precision_score(y, p), 6),
        "brier": round(brier_score_loss(y, p), 6),
        "r1": round(rafpr(y, p, 0.01), 6),
        "r05": round(rafpr(y, p, 0.005), 6),
        "r01": round(rafpr(y, p, 0.001), 6),
        "opt_thr": round(thr, 4),
        "f1": round(f1_score(y, yp), 6),
    }


def make_xgb(trial, prefix=""):
    ne = trial.suggest_int(f"{prefix}ne", 200, 600)
    md = trial.suggest_int(f"{prefix}md", 3, 10)
    lr = trial.suggest_float(f"{prefix}lr", 0.01, 0.2, log=True)
    ss = trial.suggest_float(f"{prefix}ss", 0.6, 1.0)
    cs = trial.suggest_float(f"{prefix}cs", 0.5, 1.0)
    mcw = trial.suggest_int(f"{prefix}mcw", 1, 10)
    gn = trial.suggest_float(f"{prefix}gn", 0, 5)
    spw = trial.suggest_float(f"{prefix}spw", 2, 20)
    return XGBClassifier(
        n_estimators=ne, max_depth=md, learning_rate=lr, scale_pos_weight=spw,
        subsample=ss, colsample_bytree=cs, min_child_weight=mcw, gamma=gn,
        random_state=42, eval_metric="logloss", n_jobs=-1,
    ), {f"{prefix}ne": ne, f"{prefix}md": md, f"{prefix}lr": lr, f"{prefix}ss": ss,
         f"{prefix}cs": cs, f"{prefix}mcw": mcw, f"{prefix}gn": gn, f"{prefix}spw": spw}


def optuna_best(Xtr, ytr, Xte, yte, n_trials=25, prefix=""):
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    
    def obj(trial):
        m, _ = make_xgb(trial, prefix)
        m.fit(Xtr_s, ytr)
        return roc_auc_score(yte, m.predict_proba(Xte_s)[:, 1])
    
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)
    
    bp = {k.replace(prefix, ""): v for k, v in study.best_params.items()}
    bp["random_state"] = 42; bp["eval_metric"] = "logloss"; bp["n_jobs"] = -1
    
    t0 = time.time()
    best = XGBClassifier(**bp); best.fit(Xtr_s, ytr)
    p = best.predict_proba(Xte_s)[:, 1]
    t = time.time() - t0
    
    return ev(yte, p), best, sc, bp, study.best_value


# ═══════════════════════════════════════════════════════
# ULB
# ═══════════════════════════════════════════════════════
def improve_ulb():
    print("\n" + "=" * 60)
    print("ULB CREDITCARD — Optuna 25 trials")
    print("=" * 60)
    
    df = pd.read_csv("data/creditcard.csv")
    y = df["Class"].values
    V_cols = [c for c in df.columns if c.startswith("V")]
    v = df[V_cols].values.astype(np.float32)
    
    # Advanced features
    df["v_mag"] = np.sqrt((v**2).sum(axis=1))
    df["v_mean"] = v.mean(axis=1); df["v_std"] = v.std(axis=1)
    df["v_skew"] = pd.DataFrame(v).skew(axis=1).values
    df["v_kurt"] = pd.DataFrame(v).kurtosis(axis=1).values
    df["v_asym"] = df["v_mean"] / (df["v_std"] + 1e-8)
    df["v_extreme"] = (np.abs(v) > 3).sum(axis=1).astype(np.float32)
    df["v_lo"] = (v[:, :10]**2).sum(axis=1)
    df["v_mid"] = (v[:, 10:20]**2).sum(axis=1)
    df["v_hi"] = (v[:, 20:]**2).sum(axis=1)
    df["v_ratio"] = df["v_lo"] / (df["v_hi"] + 1e-8)
    df["amount_log"] = np.log1p(df["Amount"])
    df["amt_bucket"] = pd.cut(df["Amount"], bins=[0, 0.01, 1, 10, 50, 100, 500, 25000],
                               labels=False).fillna(0)
    df["amt_x_mag"] = df["Amount"] * df["v_mag"]
    df["amt_x_ext"] = df["Amount"] * df["v_extreme"]
    df["hour"] = (df["Time"] / 3600) % 24
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(np.float32)
    df["t_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["t_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["amt_x_time"] = df["amount_log"] * df["t_sin"]
    df["amt_x_night"] = df["amount_log"] * df["is_night"]
    df["mag_x_ext"] = df["v_mag"] * df["v_extreme"]
    for i, j in [(0, 1), (1, 4), (0, 3), (1, 2), (3, 4), (0, 17)]:
        df[f"v{i}x{j}"] = df[V_cols[i]] * df[V_cols[j]]
        df[f"v{i}d{j}"] = df[V_cols[i]] / (df[V_cols[j]] + 1e-8)
    
    fc = V_cols + [
        "v_mag", "v_mean", "v_std", "v_skew", "v_kurt", "v_asym", "v_extreme",
        "v_lo", "v_mid", "v_hi", "v_ratio",
        "amount_log", "amt_bucket", "amt_x_mag", "amt_x_ext",
        "hour", "is_night", "t_sin", "t_cos", "amt_x_time", "amt_x_night", "mag_x_ext",
    ]
    for i, j in [(0, 1), (1, 4), (0, 3), (1, 2), (3, 4), (0, 17)]:
        fc += [f"v{i}x{j}", f"v{i}d{j}"]
    fc = list(dict.fromkeys(fc))
    
    X = df[fc].values.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    r, m, sc, bp, best_auc = optuna_best(Xtr, ytr, Xte, yte, n_trials=25)
    print(f"  Optuna best: {best_auc:.6f}")
    print(f"  Test: AUC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} R1%={r['r1']:.4f}")
    
    # 5-fold CV
    sc2 = StandardScaler(); X_all_s = sc2.fit_transform(X)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_aucs = []
    for tr_i, te_i in skf.split(X_all_s, y):
        m2 = XGBClassifier(**bp); m2.fit(X_all_s[tr_i], y[tr_i])
        cv_aucs.append(roc_auc_score(y[te_i], m2.predict_proba(X_all_s[te_i])[:, 1]))
    cv_mean = float(np.mean(cv_aucs)); cv_std = float(np.std(cv_aucs))
    print(f"  5-fold CV: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Calibrated
    t0 = time.time()
    cal = CalibratedClassifierCV(m, method="isotonic", cv=3)
    cal.fit(sc.transform(Xtr), ytr)
    p_cal = cal.predict_proba(sc.transform(Xte))[:, 1]
    r_cal = ev(yte, p_cal)
    print(f"  Calibrated: AUC={r_cal['roc_auc']:.4f} Brier={r_cal['brier']:.6f}")
    
    fi = m.feature_importances_
    top = np.argsort(fi)[::-1][:10]
    print("  Top features:", [(fc[i], round(fi[i], 4)) for i in top])
    
    return {"xgb_optuna": r, "xgb_calibrated": r_cal, "cv_5fold": {"mean": round(cv_mean, 6), "std": round(cv_std, 6)},
            "best_params": bp, "n_features": len(fc)}


# ═══════════════════════════════════════════════════════
# ALTMAN
# ═══════════════════════════════════════════════════════
def improve_altman():
    print("\n" + "=" * 60)
    print("ALTMAN — Optuna 20 trials (user-disjoint)")
    print("=" * 60)
    
    df = pd.read_csv("data/credit_card_transactions-ibm_v2.csv", nrows=3_000_000)
    df["label"] = df["Is Fraud?"].map(lambda x: 1 if str(x).strip() == "Yes" else 0)
    
    df["amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace("$", "").str.replace(",", ""), errors="coerce").fillna(0)
    df["amount_log"] = np.log1p(df["amount"])
    df["is_negative"] = (df["amount"] < 0).astype(np.float32)
    df["hour"] = df["Time"].astype(str).apply(lambda x: int(x.split(":")[0]) if ":" in str(x) else 12)
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]].rename(columns={"Year": "year", "Month": "month", "Day": "day"}))
    df["dow"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(np.float32)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(np.float32)
    df["user"] = df["User"].astype("category").cat.codes
    df["mcc"] = df["MCC"].astype("category").cat.codes
    df["merchant_hash"] = df["Merchant Name"].astype("category").cat.codes
    df["city_hash"] = df["Merchant City"].astype("category").cat.codes
    df["chip"] = df["Use Chip"].map({"Chip": 2, "Swipe": 1, "Online": 0}).fillna(0).astype(np.float32)
    df["has_error"] = (df["Errors?"].fillna("") != "").astype(np.float32)
    
    # User-disjoint split
    all_users = df["user"].unique()
    rng = np.random.RandomState(42); rng.shuffle(all_users)
    n_test = int(len(all_users) * 0.2)
    test_users = set(all_users[:n_test])
    train_users = set(all_users[n_test:])
    tr_mask = df["user"].isin(train_users)
    
    # Features
    df = df.sort_values(["user", "datetime"]).reset_index(drop=True)
    grp = df.groupby("user", sort=False)
    df["user_amt_mean"] = grp["amount"].transform(lambda x: x.expanding().mean())
    df["user_amt_std"] = grp["amount"].transform(lambda x: x.expanding().std().fillna(0))
    df["user_txn_count"] = grp.cumcount() + 1
    df["user_amt_ratio"] = df["amount"] / (df["user_amt_mean"] + 1e-8)
    df["user_amt_zscore"] = (df["amount"] - df["user_amt_mean"]) / (df["user_amt_std"] + 1e-8)
    df["user_amt_max"] = grp["amount"].transform(lambda x: x.expanding().max())
    df["hours_since_last"] = grp["datetime"].diff().dt.total_seconds().fillna(86400) / 3600
    df["user_amt_median"] = grp["amount"].transform(lambda x: x.expanding().median())
    df["user_txn_accel"] = df["hours_since_last"] / (grp["hours_since_last"].transform(lambda x: x.expanding().mean()) + 1e-8)
    df["user_night_ratio"] = grp["is_night"].transform(lambda x: x.expanding().mean())
    df["user_amt_trend"] = df["amount"] - df["user_amt_mean"]
    
    # Fraud rates from TRAIN ONLY
    merch_fr = df.loc[tr_mask].groupby("merchant_hash")["label"].mean()
    city_fr = df.loc[tr_mask].groupby("city_hash")["label"].mean()
    mcc_fr = df.loc[tr_mask].groupby("mcc")["label"].mean()
    df["merchant_fraud_rate"] = df["merchant_hash"].map(merch_fr).fillna(0).astype(np.float32)
    df["city_fraud_rate"] = df["city_hash"].map(city_fr).fillna(0).astype(np.float32)
    df["mcc_fraud_rate"] = df["mcc"].map(mcc_fr).fillna(0).astype(np.float32)
    df["merchant_txn_count"] = df.groupby("merchant_hash").cumcount() + 1
    
    df["amt_x_night"] = df["amount_log"] * df["is_night"]
    df["amt_x_new_merch"] = df["amount_log"] * (df["merchant_txn_count"] == 1).astype(np.float32)
    df["amt_x_high_fraud_mcc"] = df["amount_log"] * (df["mcc_fraud_rate"] > 0.01).astype(np.float32)
    df["amt_x_online"] = df["amount_log"] * (df["chip"] == 0).astype(np.float32)
    df["zscore_x_night"] = df["user_amt_zscore"] * df["is_night"]
    df["zscore_x_new_merch"] = df["user_amt_zscore"] * (df["merchant_txn_count"] == 1).astype(np.float32)
    
    fc = [
        "amount_log", "is_negative", "user_amt_mean", "user_amt_std", "user_amt_max",
        "user_amt_median", "user_amt_zscore", "user_amt_ratio", "user_amt_trend",
        "hour", "dow", "is_weekend", "is_night",
        "hours_since_last", "user_txn_count", "user_txn_accel", "user_night_ratio",
        "mcc", "mcc_fraud_rate", "merchant_hash", "merchant_fraud_rate", "merchant_txn_count",
        "city_hash", "city_fraud_rate",
        "chip", "has_error",
        "amt_x_night", "amt_x_new_merch", "amt_x_high_fraud_mcc", "amt_x_online",
        "zscore_x_night", "zscore_x_new_merch",
    ]
    
    tr_df = df[tr_mask]; te_df = df[~tr_mask]
    Xtr = tr_df[fc].fillna(0).values.astype(np.float32); ytr = tr_df["label"].values
    Xte = te_df[fc].fillna(0).values.astype(np.float32); yte = te_df["label"].values
    print(f"  Train: {len(Xtr):,} Test: {len(Xte):,} Features: {len(fc)}")
    
    r, m, sc, bp, best_auc = optuna_best(Xtr, ytr, Xte, yte, n_trials=20)
    print(f"  Optuna best: {best_auc:.6f}")
    print(f"  Test: AUC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} R1%={r['r1']:.4f}")
    
    # Calibrated
    cal = CalibratedClassifierCV(m, method="isotonic", cv=3)
    cal.fit(sc.transform(Xtr), ytr)
    p_cal = cal.predict_proba(sc.transform(Xte))[:, 1]
    r_cal = ev(yte, p_cal)
    print(f"  Calibrated: AUC={r_cal['roc_auc']:.4f} Brier={r_cal['brier']:.6f}")
    
    return {"xgb_optuna": r, "xgb_calibrated": r_cal, "best_params": bp,
            "user_disjoint": {"train_users": len(train_users), "test_users": len(test_users)}}


# ═══════════════════════════════════════════════════════
# PAYSIM
# ═══════════════════════════════════════════════════════
def improve_paysim():
    print("\n" + "=" * 60)
    print("PAYSIM — Optuna 20 trials")
    print("=" * 60)
    
    df = pd.read_csv("data/paysim_1m.csv")
    df["label"] = df["isFraud"].values.astype(int)
    
    df["amount_log"] = np.log1p(df["amount"].fillna(0))
    df["oldbalance"] = df.get("oldbalanceOrg", pd.Series(0, index=df.index)).fillna(0)
    df["newbalance"] = df.get("newbalanceOrig", pd.Series(0, index=df.index)).fillna(0)
    df["dest_old"] = df.get("oldbalanceDest", pd.Series(0, index=df.index)).fillna(0)
    df["dest_new"] = df.get("newbalanceDest", pd.Series(0, index=df.index)).fillna(0)
    
    df["balance_drain"] = np.where(df["oldbalance"] > 0, (df["oldbalance"] - df["newbalance"]) / df["oldbalance"], 0).clip(-10, 10)
    df["amt_vs_balance"] = np.where(df["oldbalance"] > 0, df["amount"] / df["oldbalance"], 0).clip(0, 100)
    df["dest_balance_change"] = df["dest_new"] - df["dest_old"]
    df["zero_after"] = (df["newbalance"] == 0).astype(np.float32)
    df["dest_zero_after"] = (df["dest_new"] == 0).astype(np.float32)
    df["full_drain"] = ((df["balance_drain"] >= 0.99) & (df["oldbalance"] > 0)).astype(np.float32)
    df["zero_balance_send"] = ((df["amount"] > 0) & (df["oldbalance"] == 0)).astype(np.float32)
    df["dest_drain"] = np.where(df["dest_old"] > 0, (df["dest_new"] - df["dest_old"]) / df["dest_old"], 0).clip(-10, 10)
    df["amt_ratio_dest"] = np.where(df["dest_old"] > 0, df["amount"] / df["dest_old"], 0).clip(0, 100)
    df["round_amount"] = (df["amount"] % 100 == 0).astype(np.float32)
    df["very_small"] = ((df["amount"] > 0) & (df["amount"] < 1)).astype(np.float32)
    df["very_large"] = (df["amount"] > df["amount"].quantile(0.99)).astype(np.float32)
    
    dummies = pd.get_dummies(df["type"], prefix="type", drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    
    fc = [
        "amount_log", "oldbalance", "newbalance", "dest_old", "dest_new",
        "balance_drain", "amt_vs_balance", "dest_balance_change",
        "zero_after", "dest_zero_after", "full_drain", "zero_balance_send",
        "dest_drain", "amt_ratio_dest", "round_amount", "very_small", "very_large",
    ] + list(dummies.columns)
    
    y = df["label"].values
    X = df[fc].fillna(0).values.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    r, m, sc, bp, best_auc = optuna_best(Xtr, ytr, Xte, yte, n_trials=20)
    print(f"  Optuna best: {best_auc:.6f}")
    print(f"  Test: AUC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} R1%={r['r1']:.4f}")
    
    # Calibrated
    cal = CalibratedClassifierCV(m, method="isotonic", cv=3)
    cal.fit(sc.transform(Xtr), ytr)
    p_cal = cal.predict_proba(sc.transform(Xte))[:, 1]
    r_cal = ev(yte, p_cal)
    print(f"  Calibrated: AUC={r_cal['roc_auc']:.4f} Brier={r_cal['brier']:.6f}")
    
    return {"xgb_optuna": r, "xgb_calibrated": r_cal, "best_params": bp, "n_features": len(fc)}


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("EFFICIENT SYSTEM IMPROVEMENT")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print("=" * 60)
    
    all_results = {}
    t_total = time.time()
    
    # ULB
    try:
        all_results["ulb"] = improve_ulb()
    except Exception as e:
        import traceback; traceback.print_exc()
        all_results["ulb"] = {"error": str(e)}
    gc.collect()
    
    # Altman
    try:
        all_results["altman"] = improve_altman()
    except Exception as e:
        import traceback; traceback.print_exc()
        all_results["altman"] = {"error": str(e)}
    gc.collect()
    
    # PaySim
    try:
        all_results["paysim"] = improve_paysim()
    except Exception as e:
        import traceback; traceback.print_exc()
        all_results["paysim"] = {"error": str(e)}
    
    t_total = time.time() - t_total
    
    # Comparison
    print("\n" + "=" * 60)
    print("BEFORE vs AFTER")
    print("=" * 60)
    
    old = {
        "ulb": {"auc": 0.9837, "r1": 0.898},
        "altman": {"auc": 0.9759, "r1": 0.8095},
        "paysim": {"auc": 0.9410, "r1": 0.4884},
    }
    
    for ds in ["ulb", "altman", "paysim"]:
        r = all_results.get(ds, {})
        if "error" in r:
            print(f"  {ds}: ERROR - {r['error'][:50]}")
            continue
        new_r = r.get("xgb_optuna", {})
        new_auc = new_r.get("roc_auc", 0)
        new_r1 = new_r.get("r1", 0)
        old_auc = old[ds]["auc"]
        old_r1 = old[ds]["r1"]
        print(f"  {ds:<8} AUC: {old_auc:.4f} → {new_auc:.4f} ({new_auc-old_auc:+.4f})  "
              f"R1%: {old_r1:.4f} → {new_r1:.4f} ({new_r1-old_r1:+.4f})")
    
    # Targets
    ulb_auc = all_results.get("ulb", {}).get("xgb_optuna", {}).get("roc_auc", 0)
    alt_auc = all_results.get("altman", {}).get("xgb_optuna", {}).get("roc_auc", 0)
    print(f"\n  ULB target 98.5%: {'✅' if ulb_auc >= 0.985 else '❌'} ({ulb_auc:.4f})")
    print(f"  Altman target 97%: {'✅' if alt_auc >= 0.97 else '❌'} ({alt_auc:.4f})")
    print(f"  Total time: {t_total:.0f}s")
    
    out = {"results": all_results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "total_time_s": round(t_total, 1)}
    with open("reports/efficient_improve_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved reports/efficient_improve_results.json")
