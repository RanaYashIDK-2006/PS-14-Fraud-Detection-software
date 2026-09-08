#!/usr/bin/env python3
"""Phase 1: Exhaustive model sweep on ULB creditcard (fast)."""
import numpy as np, pandas as pd, json, time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

DATA_DIR, REPORTS_DIR = Path("data"), Path("reports")
DROP = {"label","new_device_flag","unusual_location_flag","failed_auth_count_24h",
        "known_device_count","shared_device_accounts","shared_recipient_accounts",
        "mule_ring_score","is_weekend","amount_v_correlation","days_since_last_similar_txn"}

def recall_at(y_true, y_score, fpr_target):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    v = fpr <= fpr_target
    return float(tpr[v][-1]) if v.any() else 0.0

def main():
    df = pd.read_csv(DATA_DIR / "transactions_v2.csv")
    cols = [c for c in df.columns if c not in DROP]
    X = np.nan_to_num(df[cols].values.astype(np.float32), nan=0, posinf=100, neginf=-100)
    y = df["label"].values
    print(f"ULB: {len(df):,} rows, {y.sum()} fraud, {len(cols)} features")

    configs = [
        ("LR_C10", LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)),
        ("RF_200", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)),
    ]
    try:
        from xgboost import XGBClassifier
        spw = (y==0).sum() / max((y==1).sum(), 1)
        configs += [
            ("XGB_300d8", XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8, scale_pos_weight=spw, random_state=42, eval_metric="aucpr", use_label_encoder=False, n_jobs=-1)),
            ("XGB_500d6", XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.85, scale_pos_weight=spw, min_child_weight=5, random_state=42, eval_metric="aucpr", use_label_encoder=False, n_jobs=-1)),
        ]
    except ImportError:
        pass

    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    results = {}
    print(f"\n{'Config':<15} {'AUC':>8} {'PR-AUC':>8} {'R@1%':>8} {'R@0.5%':>8} {'R@0.1%':>8}")
    print("-" * 58)

    for name, model in configs:
        t0 = time.time()
        aucs, prs, r1s, r5s, r1s_ = [], [], [], [], []
        for tr, va in skf.split(X, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xva = scaler.transform(X[va])
            model.fit(Xtr, y[tr])
            p = model.predict_proba(Xva)[:, 1]
            aucs.append(roc_auc_score(y[va], p))
            prs.append(average_precision_score(y[va], p))
            r1s.append(recall_at(y[va], p, 0.01))
            r5s.append(recall_at(y[va], p, 0.005))
            r1s_.append(recall_at(y[va], p, 0.001))
        dt = time.time() - t0
        m = {"roc_auc": np.mean(aucs), "pr_auc": np.mean(prs),
             "recall_1pct": np.mean(r1s), "recall_0_5pct": np.mean(r5s), "recall_0_1pct": np.mean(r1s_),
             "std_auc": np.std(aucs)}
        results[name] = m
        print(f"  {name:<13} {m['roc_auc']:.4f}  {m['pr_auc']:.4f}  {m['recall_1pct']:.4f}  {m['recall_0_5pct']:.4f}  {m['recall_0_1pct']:.4f}  ({dt:.0f}s)")

    # Best model
    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest: {best} AUC={results[best]['roc_auc']:.4f}")

    # Quick stacker: use train/test split with top models
    print("\n--- Stacker Ensemble ---")
    top3 = sorted(results, key=lambda k: results[k]["roc_auc"], reverse=True)[:3]
    print(f"  Top 3: {top3}")
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler_s = StandardScaler()
    Xtr_s = scaler_s.fit_transform(Xtr)
    Xte_s = scaler_s.transform(Xte)
    # Get test preds from top models
    test_preds = []
    for name in top3:
        for n, m in configs:
            if n == name:
                m.fit(Xtr_s, ytr)
                test_preds.append(m.predict_proba(Xte_s)[:, 1])
                break
    # Also get train OOF preds via CV
    oof_preds = []
    for name in top3:
        for n, m in configs:
            if n == name:
                oof_preds.append(cross_val_predict(m, Xtr_s, ytr, cv=3, method="predict_proba")[:, 1])
                break
    oof_mat = np.column_stack(oof_preds)
    test_mat = np.column_stack(test_preds)
    meta = LogisticRegression(C=10, max_iter=1000, random_state=42)
    meta.fit(oof_mat, ytr)
    p_stack = meta.predict_proba(test_mat)[:, 1]
    stack_auc = roc_auc_score(yte, p_stack)
    stack_pr = average_precision_score(yte, p_stack)
    stack_r1 = recall_at(yte, p_stack, 0.01)
    stack_r5 = recall_at(yte, p_stack, 0.005)
    stack_r01 = recall_at(yte, p_stack, 0.001)
    print(f"  Stacker: AUC={stack_auc:.4f}  PR-AUC={stack_pr:.4f}  R@1%={stack_r1:.4f}  R@0.5%={stack_r5:.4f}  R@0.1%={stack_r01:.4f}")
    results["stacker"] = {"roc_auc": stack_auc, "pr_auc": stack_pr, "recall_1pct": stack_r1, "recall_0_5pct": stack_r5, "recall_0_1pct": stack_r01}

    # Save
    with open(REPORTS_DIR / "phase1_ulb_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to reports/phase1_ulb_results.json")

if __name__ == "__main__":
    main()
