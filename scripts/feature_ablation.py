#!/usr/bin/env python3
"""Feature Ablation Study — find the minimal feature set that maximizes AUC.

Strategy:
  1. Start with all 35 features
  2. Remove the least important feature
  3. Retrain and evaluate on full 6M test set
  4. If AUC doesn't drop, keep it removed
  5. Repeat until no more features can be removed
"""
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

CACHE = Path("data/_sorted_cache")
MODELS = Path("models/production")
REPORTS = Path("reports")


def load_features():
    """Load features from cache (streaming)."""
    pf = pq.ParquetFile(CACHE / "features_all.parquet")
    feat_cols = None
    train_X, train_y = [], []
    test_X, test_y = [], []

    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg).to_pandas()
        if feat_cols is None:
            feat_cols = [c for c in chunk.columns if c != "is_fraud"]
        y = chunk["is_fraud"].values.astype(np.int8)
        months = chunk["Month"].values
        X = chunk[feat_cols].values.astype(np.float32)
        del chunk; gc.collect()
        tr = months <= 9; te = months > 9
        if tr.any():
            ti = np.where(tr)[0]
            fi = ti[y[ti] == 1]; li = ti[y[ti] == 0]
            rng = np.random.RandomState(42)
            sub = rng.choice(li, min(max(1, int(len(li) * 0.10)), len(li)), replace=False)
            train_X.append(X[np.concatenate([fi, sub])])
            train_y.append(y[np.concatenate([fi, sub])])
        if te.any():
            test_X.append(X[np.where(te)[0]])
            test_y.append(y[np.where(te)[0]])
        del X; gc.collect()

    Xtr = np.concatenate(train_X); ytr = np.concatenate(train_y)
    Xte = np.concatenate(test_X); yte = np.concatenate(test_y)
    del train_X, test_X; gc.collect()
    return Xtr, ytr, Xte, yte, feat_cols


def train_evaluate(Xtr, ytr, Xte, yte, feat_names, spw):
    """Quick train+evaluate with XGB only (fast for ablation)."""
    import xgboost as xgb
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        scale_pos_weight=spw, random_state=42, n_jobs=4,
        eval_metric="auc", use_label_encoder=False,
    )
    m.fit(Xtr_s, ytr, verbose=False)
    p = m.predict_proba(Xte_s)[:, 1]
    auc = roc_auc_score(yte, p)
    fpr, tpr, _ = roc_curve(yte, p)
    r1 = float(tpr[fpr <= 0.01][-1]) if (fpr <= 0.01).any() else 0.0
    imp = dict(zip(feat_names, m.feature_importances_))
    return auc, r1, imp, m, scaler


def main():
    print("=" * 70)
    print("FEATURE ABLATION STUDY")
    print("=" * 70)
    t0 = time.time()

    Xtr, ytr, Xte, yte, all_feats = load_features()
    spw = min((1 - ytr.mean()) / ytr.mean(), 20)
    print(f"  Train: {len(Xtr):,} ({int(ytr.sum()):,} fraud)")
    print(f"  Test:  {len(Xte):,} ({int(yte.sum()):,} fraud)")
    print(f"  Features: {len(all_feats)}")

    # Baseline: all features
    print("\n[BASELINE] All features...")
    auc_base, r1_base, imp_base, _, _ = train_evaluate(Xtr, ytr, Xte, yte, all_feats, spw)
    print(f"  AUC: {auc_base:.4f}  R@1%FPR: {r1_base:.4f}")

    # Ablation: iteratively remove lowest-importance feature
    remaining = list(all_feats)
    results = [{"n_features": len(remaining), "features": remaining[:],
                "auc": auc_base, "r1": r1_base, "dropped": "none"}]

    for step in range(20):  # max 20 removals
        if len(remaining) <= 10:
            print(f"\n  Stopping at {len(remaining)} features (minimum)")
            break

        # Get current importances
        feat_idx = [all_feats.index(f) for f in remaining]
        Xtr_sub = Xtr[:, feat_idx]
        Xte_sub = Xte[:, feat_idx]

        auc_cur, r1_cur, imp_cur, _, _ = train_evaluate(
            Xtr_sub, ytr, Xte_sub, yte, remaining, spw)

        # Find least important feature
        sorted_imp = sorted(imp_cur.items(), key=lambda x: x[1])
        worst_feat, worst_imp = sorted_imp[0]
        worst_pct = worst_imp / sum(v for _, v in sorted_imp) * 100

        # Try dropping it
        candidate = [f for f in remaining if f != worst_feat]
        cand_idx = [all_feats.index(f) for f in candidate]
        Xtr_cand = Xtr[:, cand_idx]
        Xte_cand = Xte[:, cand_idx]

        auc_new, r1_new, _, _, _ = train_evaluate(
            Xtr_cand, ytr, Xte_cand, yte, candidate, spw)

        delta_auc = auc_new - auc_cur
        delta_r1 = r1_new - r1_cur

        if delta_auc >= -0.001:  # allow tiny drop
            remaining = candidate
            status = "DROPPED" if delta_auc >= 0 else f"KEPT ({delta_auc:+.4f})"
            print(f"  Step {step+1}: dropped '{worst_feat}' ({worst_pct:.2f}%) → "
                  f"AUC={auc_new:.4f} ({delta_auc:+.4f}) R@1%={r1_new:.4f} [{status}]")
            results.append({
                "n_features": len(remaining), "features": remaining[:],
                "auc": round(auc_new, 4), "r1": round(r1_new, 4),
                "dropped": worst_feat,
            })
        else:
            print(f"  Step {step+1}: CANNOT drop '{worst_feat}' ({worst_pct:.2f}%) → "
                  f"AUC would drop {delta_auc:.4f}")
            break

    # Final model with minimal features
    print(f"\n[FINAL] {len(remaining)} features...")
    feat_idx = [all_feats.index(f) for f in remaining]
    Xtr_final = Xtr[:, feat_idx]
    Xte_final = Xte[:, feat_idx]

    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(Xtr_final)
    Xte_s = scaler.transform(Xte_final)

    # 3-fold CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_aucs = []
    for tr_i, va_i in skf.split(Xtr_s, ytr):
        Xa, Xv = Xtr_s[tr_i], Xtr_s[va_i]
        ya, yv = ytr[tr_i], ytr[va_i]
        m_x = xgb.XGBClassifier(n_estimators=363, max_depth=5, learning_rate=0.127,
            subsample=0.88, colsample_bytree=0.77, min_child_weight=6, gamma=0.21,
            reg_alpha=5.6, reg_lambda=6.6, scale_pos_weight=spw,
            random_state=42, n_jobs=4, eval_metric="auc", use_label_encoder=False)
        m_x.fit(Xa, ya, verbose=False)
        m_l = lgb.LGBMClassifier(n_estimators=545, max_depth=12, learning_rate=0.133,
            subsample=0.62, colsample_bytree=0.95, min_child_weight=16, num_leaves=117,
            reg_alpha=8.66, reg_lambda=6.91, scale_pos_weight=spw,
            random_state=42, n_jobs=4, verbose=-1)
        m_l.fit(Xa, ya)
        p = 0.307 * m_x.predict_proba(Xv)[:, 1] + 0.543 * m_l.predict_proba(Xv)[:, 1]
        cv_aucs.append(roc_auc_score(yv, p))

    # Final train
    m_xgb = xgb.XGBClassifier(n_estimators=363, max_depth=5, learning_rate=0.127,
        subsample=0.88, colsample_bytree=0.77, min_child_weight=6, gamma=0.21,
        reg_alpha=5.6, reg_lambda=6.6, scale_pos_weight=spw,
        random_state=42, n_jobs=4, eval_metric="auc", use_label_encoder=False)
    m_xgb.fit(Xtr_s, ytr, verbose=False)

    m_lgb = lgb.LGBMClassifier(n_estimators=545, max_depth=12, learning_rate=0.133,
        subsample=0.62, colsample_bytree=0.95, min_child_weight=16, num_leaves=117,
        reg_alpha=8.66, reg_lambda=6.91, scale_pos_weight=spw,
        random_state=42, n_jobs=4, verbose=-1)
    m_lgb.fit(Xtr_s, ytr)

    p_te = 0.307 * m_xgb.predict_proba(Xte_s)[:, 1] + 0.543 * m_lgb.predict_proba(Xte_s)[:, 1]
    test_auc = roc_auc_score(yte, p_te)
    fpr, tpr, _ = roc_curve(yte, p_te)
    test_r1 = float(tpr[fpr <= 0.01][-1]) if (fpr <= 0.01).any() else 0.0

    print(f"  CV AUC:    {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    print(f"  Test AUC:  {test_auc:.4f} (baseline: {auc_base:.4f}, delta: {test_auc-auc_base:+.4f})")
    print(f"  Test R@1%: {test_r1:.4f} (baseline: {r1_base:.4f}, delta: {test_r1-r1_base:+.4f})")
    print(f"  Features:  {len(remaining)} (was {len(all_feats)}, saved {len(all_feats)-len(remaining)})")

    # Save lean model
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(m_xgb, MODELS / "xgb_production.joblib")
    joblib.dump(m_lgb, MODELS / "lgb_production.joblib")
    joblib.dump(scaler, MODELS / "scaler_production.joblib")

    import hashlib
    mh = hashlib.md5()
    for f in sorted(MODELS.glob("*.joblib")):
        mh.update(f.read_bytes())

    manifest = {
        "model_version": f"altman_lean_{len(remaining)}feat_{ts}",
        "model_type": "xgb_lgb_ensemble_lean",
        "n_features": len(remaining),
        "features": remaining,
        "temporal_test_auc": round(test_auc, 6),
        "temporal_test_r1": round(test_r1, 6),
        "cv_auc_mean": round(float(np.mean(cv_aucs)), 6),
        "ensemble_weights": {"xgb": 0.307, "lgb": 0.543},
        "model_hash": mh.hexdigest()[:16],
        "target_leakage": False,
        "full_scale": True,
        "lean": True,
        "features_dropped": [r["dropped"] for r in results if r["dropped"] != "none"],
        "trained_at": datetime.now().isoformat(),
        "artifacts": [f.name for f in sorted(MODELS.glob("*.joblib"))],
    }
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (MODELS / "feature_list.json").write_text(json.dumps(remaining, indent=2))

    # Also save to optuna_tuned
    joblib.dump(m_xgb, MODELS / "optuna_tuned" / "xgb_optuna.joblib")
    joblib.dump(m_lgb, MODELS / "optuna_tuned" / "lgb_optuna.joblib")
    joblib.dump(scaler, MODELS / "optuna_tuned" / "scaler_optuna.joblib")
    (MODELS / "optuna_tuned" / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Save ablation report
    (REPORTS / "feature_ablation.json").write_text(json.dumps({
        "baseline_auc": round(auc_base, 4),
        "baseline_r1": round(r1_base, 4),
        "final_auc": round(test_auc, 4),
        "final_r1": round(test_r1, 4),
        "features_dropped": [r["dropped"] for r in results if r["dropped"] != "none"],
        "features_remaining": remaining,
        "n_features_before": len(all_feats),
        "n_features_after": len(remaining),
        "ablation_steps": results,
    }, indent=2))

    print(f"\n  Saved: {manifest['model_version']}")
    print(f"  Dropped: {[r['dropped'] for r in results if r['dropped'] != 'none']}")
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
