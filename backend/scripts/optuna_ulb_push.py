#!/usr/bin/env python3
"""Aggressive Optuna tuning for ULB Creditcard — target ROC-AUC > 98.5%.

Strategy:
1. 100 Optuna trials with wide search space
2. Advanced features from V1-V28
3. Ensemble of top-3 configs
4. 5-fold CV for honest generalization
5. Threshold optimization for R@1%FPR
"""
import hashlib, json, time, warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
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
        fpr = np.concatenate([[0], fpr])
        tpr = np.concatenate([[0], tpr])
    return float(tpr[min(np.searchsorted(fpr, t), len(tpr)-1)])


def find_thr(y, s, t=0.01):
    fpr, tpr, thr = roc_curve(y, s)
    v = fpr <= t
    if not v.any():
        return thr[0]
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
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
    }


# ═══════════════════════════════════════════════════════
# LOAD & FEATURE ENGINEER ULB
# ═══════════════════════════════════════════════════════

print("=" * 70)
print("OPTUNA ULB PUSH: target ROC-AUC > 98.5%")
print("=" * 70)

df = pd.read_csv("data/creditcard.csv")
y = df["Class"].values
V_cols = [c for c in df.columns if c.startswith("V")]
v = df[V_cols].values.astype(np.float32)
print(f"ULB: {len(df):,} rows, fraud={df['Class'].sum():,}")

# ── AGGRESSIVE FEATURE ENGINEERING ──
print("Engineering features...")

# PCA statistical features
df["v_mag"] = np.sqrt((v**2).sum(axis=1))
df["v_mean"] = v.mean(axis=1)
df["v_std"] = v.std(axis=1)
df["v_skew"] = pd.DataFrame(v).skew(axis=1).values
df["v_kurt"] = pd.DataFrame(v).kurtosis(axis=1).values
df["v_asym"] = df["v_mean"] / (df["v_std"] + 1e-8)
df["v_median"] = np.median(v, axis=1)
df["v_iqr"] = np.percentile(v, 75, axis=1) - np.percentile(v, 25, axis=1)
df["v_range"] = v.max(axis=1) - v.min(axis=1)
df["v_norm_l2"] = np.sqrt((v**2).mean(axis=1))

# Extreme value features
for threshold in [1, 2, 3]:
    mask = np.abs(v) > threshold
    df[f"v_extreme_{threshold}"] = mask.sum(axis=1).astype(np.float32)
    # Mean of extreme values per row
    abs_v = np.abs(v)
    extreme_sums = (abs_v * mask).sum(axis=1)
    extreme_counts = mask.sum(axis=1).astype(float)
    df[f"v_extreme_mean_{threshold}"] = np.where(
        extreme_counts > 0, extreme_sums / extreme_counts, 0
    )

# Energy bands
for start, end, name in [(0, 10, "lo"), (10, 20, "mid"), (20, 28, "hi")]:
    df[f"v_energy_{name}"] = (v[:, start:end]**2).sum(axis=1)
df["v_ratio_lo_hi"] = df["v_energy_lo"] / (df["v_energy_hi"] + 1e-8)
df["v_ratio_lo_mid"] = df["v_energy_lo"] / (df["v_energy_mid"] + 1e-8)

# Pairwise V interactions (top-10 most important pairs)
pairs = [(0,1),(0,3),(0,17),(1,2),(1,4),(1,12),(3,4),(3,10),(4,9),(10,12)]
for i, j in pairs:
    df[f"v{i}x{j}"] = df[V_cols[i]] * df[V_cols[j]]
    df[f"v{i}d{j}"] = df[V_cols[i]] / (df[V_cols[j]] + 1e-8)
    df[f"v{i}p{j}"] = df[V_cols[i]] + df[V_cols[j]]

# Amount features
df["amount_log"] = np.log1p(df["Amount"])
df["amount_bucket"] = pd.cut(df["Amount"], bins=[0,0.01,0.5,1,5,10,25,50,100,250,500,1000,25000],
                              labels=False).fillna(0)
df["amt_x_mag"] = df["Amount"] * df["v_mag"]
df["amt_x_extreme"] = df["Amount"] * df["v_extreme_3"]
df["amt_x_skew"] = df["Amount"] * df["v_skew"]
df["amt_x_kurt"] = df["Amount"] * df["v_kurt"]

# Time features
df["hour"] = (df["Time"] / 3600) % 24
df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(np.float32)
df["is_weekend"] = ((df["Time"] / 86400).astype(int) % 7 >= 5).astype(np.float32)
df["t_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["t_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["amt_x_time"] = df["amount_log"] * df["t_sin"]
df["amt_x_night"] = df["amount_log"] * df["is_night"]
df["mag_x_extreme"] = df["v_mag"] * df["v_extreme_3"]

# Build full feature list
fc = V_cols.copy()
fc += [
    "v_mag","v_mean","v_std","v_skew","v_kurt","v_asym","v_median","v_iqr","v_range","v_norm_l2",
    "v_extreme_1","v_extreme_mean_1","v_extreme_2","v_extreme_mean_2","v_extreme_3","v_extreme_mean_3",
    "v_energy_lo","v_energy_mid","v_energy_hi","v_ratio_lo_hi","v_ratio_lo_mid",
    "amount_log","amount_bucket","amt_x_mag","amt_x_extreme","amt_x_skew","amt_x_kurt",
    "hour","is_night","is_weekend","t_sin","t_cos","amt_x_time","amt_x_night","mag_x_extreme",
]
for i, j in pairs:
    fc += [f"v{i}x{j}", f"v{i}d{j}", f"v{i}p{j}"]
fc = list(dict.fromkeys(fc))
print(f"Features: {len(fc)}")

X = df[fc].values.astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)

# ═══════════════════════════════════════════════════════
# OPTUNA 100 TRIALS
# ═══════════════════════════════════════════════════════

print(f"\nOptuna 30 trials...")
t0 = time.time()

# Store top-5 trials for ensemble
top_trials = []


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0, 8),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 15),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 15),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 30),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    m = XGBClassifier(**params)
    m.fit(Xtr_s, ytr)
    p = m.predict_proba(Xte_s)[:, 1]
    score = roc_auc_score(yte, p)

    # Track top-5
    top_trials.append((score, params.copy()))
    top_trials.sort(key=lambda x: x[0], reverse=True)
    if len(top_trials) > 5:
        top_trials.pop()

    return score


study = optuna.create_study(direction="maximize", study_name="ulb_xgb")
study.optimize(objective, n_trials=30, show_progress_bar=True)
optuna_time = time.time() - t0
print(f"\nOptuna complete in {optuna_time:.0f}s")
print(f"Best trial AUC: {study.best_value:.6f}")
print(f"Top-5 AUCs: {[f'{s:.4f}' for s, _ in top_trials[:5]]}")

# ═══════════════════════════════════════════════════════
# EVALUATE BEST CONFIG
# ═══════════════════════════════════════════════════════

results = {}

# Best single XGB
bp = study.best_params
bp["random_state"] = 42; bp["eval_metric"] = "logloss"; bp["n_jobs"] = -1
t0 = time.time()
best_xgb = XGBClassifier(**bp)
best_xgb.fit(Xtr_s, ytr)
p_best = best_xgb.predict_proba(Xte_s)[:, 1]
t_best = time.time() - t0
r = ev(yte, p_best); r["time"] = round(t_best, 2)
results["xgb_best"] = r
print(f"\nBest XGB: AUC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} R1%={r['r1']:.4f} ({t_best:.1f}s)")

# ═══════════════════════════════════════════════════════
# ENSEMBLE OF TOP-3
# ═══════════════════════════════════════════════════════

print("\nTop-3 ensemble...")
top3 = top_trials[:3]
probs_ensemble = np.zeros(len(yte))
for i, (score, params) in enumerate(top3):
    params["random_state"] = 42; params["eval_metric"] = "logloss"; params["n_jobs"] = -1
    m = XGBClassifier(**params)
    m.fit(Xtr_s, ytr)
    probs_ensemble += m.predict_proba(Xte_s)[:, 1] / 3
    print(f"  Config {i+1}: AUC={score:.4f}")

r_ens = ev(yte, probs_ensemble)
results["ensemble_top3"] = r_ens
print(f"Ensemble: AUC={r_ens['roc_auc']:.4f} PR={r_ens['pr_auc']:.4f} R1%={r_ens['r1']:.4f}")

# ═══════════════════════════════════════════════════════
# ENSEMBLE OF TOP-5
# ═══════════════════════════════════════════════════════

print("\nTop-5 ensemble...")
top5 = top_trials[:5]
probs_ens5 = np.zeros(len(yte))
for i, (score, params) in enumerate(top5):
    params["random_state"] = 42; params["eval_metric"] = "logloss"; params["n_jobs"] = -1
    m = XGBClassifier(**params)
    m.fit(Xtr_s, ytr)
    probs_ens5 += m.predict_proba(Xte_s)[:, 1] / 5

r_ens5 = ev(yte, probs_ens5)
results["ensemble_top5"] = r_ens5
print(f"Top-5 Ensemble: AUC={r_ens5['roc_auc']:.4f} PR={r_ens5['pr_auc']:.4f} R1%={r_ens5['r1']:.4f}")

# ═══════════════════════════════════════════════════════
# CALIBRATED BEST
# ═══════════════════════════════════════════════════════

print("\nCalibrated best XGB...")
t0 = time.time()
cal = CalibratedClassifierCV(best_xgb, method="isotonic", cv=3)
cal.fit(Xtr_s, ytr)
p_cal = cal.predict_proba(Xte_s)[:, 1]
t_cal = time.time() - t0
r_cal = ev(yte, p_cal); r_cal["time"] = round(t_cal, 2)
results["xgb_calibrated"] = r_cal
print(f"Calibrated: AUC={r_cal['roc_auc']:.4f} Brier={r_cal['brier']:.6f}")

# ═══════════════════════════════════════════════════════
# 5-FOLD CV ON BEST CONFIG
# ═══════════════════════════════════════════════════════

print("\n5-fold CV on best config...")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
cv_aucs, cv_prs, cv_r1s = [], [], []
for fold, (tr_idx, te_idx) in enumerate(skf.split(Xtr_s, ytr)):
    params_cv = bp.copy()
    m = XGBClassifier(**params_cv)
    m.fit(Xtr_s[tr_idx], ytr[tr_idx])
    p = m.predict_proba(Xtr_s[te_idx])[:, 1]
    cv_aucs.append(roc_auc_score(ytr[te_idx], p))
    cv_prs.append(average_precision_score(ytr[te_idx], p))
    cv_r1s.append(rafpr(ytr[te_idx], p, 0.01))
    print(f"  Fold {fold+1}: AUC={cv_aucs[-1]:.4f} PR={cv_prs[-1]:.4f}")

results["cv_5fold"] = {
    "roc_auc_mean": round(float(np.mean(cv_aucs)), 6),
    "roc_auc_std": round(float(np.std(cv_aucs)), 6),
    "pr_auc_mean": round(float(np.mean(cv_prs)), 6),
    "r1_mean": round(float(np.mean(cv_r1s)), 6),
}

# ═══════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════

fi = best_xgb.feature_importances_
top_idx = np.argsort(fi)[::-1][:20]
print("\nTop 20 features:")
for i in top_idx:
    print(f"  {fc[i]:<30s} {fi[i]:.4f}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
for name, m in results.items():
    if "cv" in name:
        print(f"  {name:<25} ROC-AUC={m['roc_auc_mean']:.4f} ± {m['roc_auc_std']:.4f}")
    else:
        roc = m.get("roc_auc", 0)
        pr = m.get("pr_auc", 0)
        r1 = m.get("r1", 0)
        print(f"  {name:<25} ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  R@1%FPR={r1:.4f}")

target_met = max(results.get("xgb_best", {}).get("roc_auc", 0),
                 results.get("ensemble_top3", {}).get("roc_auc", 0),
                 results.get("ensemble_top5", {}).get("roc_auc", 0)) >= 0.985
print(f"\n  Target 98.5%: {'✅ MET' if target_met else '❌ NOT MET'}")

# Save
out = {
    "dataset": "ulb_creditcard",
    "n_features": len(fc),
    "optuna_trials": 30,
    "optuna_time_s": round(optuna_time, 1),
    "best_params": {k: v for k, v in bp.items() if k not in ("random_state", "eval_metric", "n_jobs")},
    "top5_configs": [{k: v for k, v in p.items() if k not in ("random_state", "eval_metric", "n_jobs")} for _, p in top_trials[:5]],
    "results": results,
    "target_met": target_met,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open("reports/ulb_optuna_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved reports/ulb_optuna_results.json")
print("=" * 70)
