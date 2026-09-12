#!/usr/bin/env python3
"""Domain-adaptive meta-ensemble: combine 4 domain-specific fusions.

Strategy:
1. Train 4 domain-specific fusions (Kaggle, UCI Default, Bank, Diabetes)
2. Level-1: each fusion produces a prediction for every sample
3. Level-2: a meta-learner combines the 4 predictions + original features
4. The meta-learner learns which domain model to trust for each input pattern

This should generalize better than any single-domain model because it
sees patterns from multiple fraud/abuse domains.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DATA = ROOT / "data"

ML_FEATURES = [
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "hour_of_day", "is_weekend",
    "shared_device_accounts", "shared_recipient_accounts", "mule_ring_score",
]


# ── Dataset loaders (same as expanded script) ────────────────────────────

def load_kaggle() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA / "creditcard.csv")
    n = len(df)
    X = np.column_stack([
        (df["Amount"] / max(df["Amount"].median(), 1)).clip(0, 10).values,
        np.clip((-df["V3"]).rank(pct=True) * 6, 1, 6).astype(int).values,
        (df["V4"].abs() > 2).astype(int).values,
        (df["V14"] < -5).astype(int).values,
        (df["V8"].abs() > 2).astype(int).values,
        (df["V7"].abs() > 2).astype(int).values,
        np.clip((-df["V1"]).rank(pct=True) * 4, 0, 4).astype(int).values,
        np.clip(df["Time"] / 3600, 0, 48).astype(int).values,
        np.clip((df["V10"].abs() - 2) / 8, 0, 1).values,
        np.clip(df["V11"].rank(pct=True) * 4, 1, 4).astype(int).values,
        np.clip(df["V12"].rank(pct=True) * 10, 1, 10).astype(int).values,
        (df["Time"] % (24 * 3600) / 3600).astype(int).values % 24,
        (df["V13"].abs() > 1.5).astype(int).values,
        np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n),
    ])
    return X, df["Class"].values.astype(int)


def load_uci_default() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA / "validation_uci_default.csv")
    return df[ML_FEATURES].values.astype(float), df["label"].values.astype(int)


def load_bank_marketing() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA / "bank_raw/bank-additional/bank-additional-full.csv", sep=";")
    n = len(df)
    job_map = {j: i for i, j in enumerate(df["job"].unique())}
    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    X = np.column_stack([
        np.clip(df["duration"] / max(df["duration"].median(), 1), 0, 10).values,
        np.clip(df["campaign"], 1, 6).values,
        (df["day_of_week"].isin(["fri", "mon"])).astype(int).values,
        (df["pdays"] == 999).astype(int).values,
        (df["emp.var.rate"] < -1).astype(int).values,
        (df["previous"] > 0).astype(int).values,
        np.clip(df["previous"], 0, 4).values,
        np.clip(df["pdays"], 0, 48).values,
        np.clip(df["campaign"].rank(pct=True), 0, 1).values,
        df["job"].map(job_map).fillna(0).values.astype(int),
        np.clip(df["age"] / 10, 1, 8).astype(int).values,
        df["month"].map(month_map).fillna(5).values % 24,
        (df["day_of_week"].isin(["sat", "sun"])).astype(int).values,
        np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n),
    ])
    return X, (df["y"] == "yes").astype(int).values


def load_diabetes() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA / "diabetes_raw/diabetic_data.csv", na_values="?", low_memory=False)
    n = len(df)
    age_map = {"[0-10)": 1, "[10-20)": 2, "[20-30)": 3, "[30-40)": 4, "[40-50)": 5,
               "[50-60)": 6, "[60-70)": 7, "[70-80)": 8, "[80-90)": 9, "[90-100)": 10}
    X = np.column_stack([
        np.clip(df["num_medications"] / max(df["num_medications"].median(), 1), 0, 10).values,
        np.clip(df["num_lab_procedures"] / 10, 0, 6).astype(int).values,
        (df["time_in_hospital"] > 7).astype(int).values,
        (df["number_inpatient"] > 0).astype(int).values,
        (df["number_emergency"] > 0).astype(int).values,
        (df["num_procedures"] > 3).astype(int).values,
        np.clip(df["number_inpatient"], 0, 4).values,
        np.clip(df["number_outpatient"], 0, 48).values,
        np.clip(df["num_medications"].rank(pct=True), 0, 1).values,
        df["age"].map(age_map).fillna(5).values.astype(int),
        df["age"].map(age_map).fillna(5).values.astype(int),
        np.zeros(n, dtype=int),
        (df["discharge_disposition_id"].isin([1, 6, 8])).astype(int).values,
        np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n),
    ])
    return X, (df["readmitted"] == "<30").astype(int).values


# ── Fusion trainer ───────────────────────────────────────────────────────

def train_one_fusion(X: np.ndarray, y: np.ndarray) -> dict:
    """Train a lightweight fusion: LR + XGB + stacker."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    lr.fit(Xs, y)

    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.2,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb.fit(X, y)
    except ImportError:
        xgb = None

    # Stacker: LR + XGB only
    p_lr = lr.predict_proba(Xs)[:, 1]
    p_xgb = xgb.predict_proba(X)[:, 1] if xgb is not None else np.full(len(y), 0.5)

    X_stack = np.column_stack([p_lr, p_xgb])
    stacker = LogisticRegression(max_iter=500, random_state=42)
    stacker.fit(X_stack, y)

    return {
        "lr": lr, "xgb": xgb,
        "stacker": stacker, "scaler": scaler,
    }


def predict_fusion(fusion: dict, X: np.ndarray) -> np.ndarray:
    """Get fused prediction from a trained fusion."""
    Xs = fusion["scaler"].transform(X)
    p_lr = fusion["lr"].predict_proba(Xs)[:, 1]
    p_xgb = fusion["xgb"].predict_proba(X)[:, 1] if fusion["xgb"] is not None else np.full(len(X), 0.5)

    X_stack = np.column_stack([p_lr, p_xgb])
    return fusion["stacker"].predict_proba(X_stack)[:, 1]


# ── Evaluation ───────────────────────────────────────────────────────────

def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute standard metrics."""
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)

    recall_at_1 = 0.0
    for t in np.arange(0.01, 1.0, 0.005):
        fp_rate = ((y_prob >= t) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
        if fp_rate <= 0.01:
            recall_at_1 = ((y_prob >= t) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)
            break

    return {
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr), 4),
        "recall_at_1pct_fpr": round(float(recall_at_1), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("PS-14: Domain-Adaptive Meta-Ensemble")
    print("=" * 72)

    # Load datasets
    print("\n[1] Loading 4 real datasets...")
    raw = {
        "Kaggle Fraud": load_kaggle(),
        "UCI Default": load_uci_default(),
        "Bank Marketing": load_bank_marketing(),
        "Diabetes Readmit": load_diabetes(),
    }

    # Sample large datasets to keep runtime manageable
    print("\n[2] Sampling large datasets (max 30K each)...")
    MAX_N = 30000
    for name in list(raw.keys()):
        X, y = raw[name]
        if len(y) > MAX_N:
            idx = np.random.RandomState(42).choice(len(y), MAX_N, replace=False)
            # Ensure we keep enough positive samples
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos = min(len(pos_idx), MAX_N // 10)
            n_neg = MAX_N - n_pos
            chosen = np.concatenate([
                np.random.RandomState(42).choice(pos_idx, n_pos, replace=False),
                np.random.RandomState(42).choice(neg_idx, n_neg, replace=False),
            ])
            raw[name] = (X[chosen], y[chosen])
            print(f"  {name}: sampled to {len(chosen):,} ({y[chosen].sum()} pos)")

    # Split 70/30
    print("\n[3] Splitting 70/30...")
    splits = {}
    for name, (X, y) in raw.items():
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42,
                                                     stratify=y if y.sum() > 10 else None)
        splits[name] = (X_tr, X_te, y_tr, y_te)
        print(f"  {name}: train={len(y_tr):,} ({y_tr.sum()} pos), test={len(y_te):,} ({y_te.sum()} pos)")

    # Train 4 domain-specific fusions
    print("\n[4] Training 4 domain-specific fusions...")
    fusions = {}
    for name in raw:
        X_tr, _, y_tr, _ = splits[name]
        fusions[name] = train_one_fusion(X_tr, y_tr)
        pred = predict_fusion(fusions[name], X_tr)
        auc = roc_auc_score(y_tr, pred)
        print(f"  {name}: train AUC={auc:.4f}")

    # Generate level-1 predictions for all test sets
    print("\n[5] Generating level-1 predictions (4 models × 4 test sets)...")
    L1 = {}  # L1[domain_name] = np.array of shape (n_test, 4)
    for test_name in raw:
        _, X_te, _, y_te = splits[test_name]
        preds = np.column_stack([
            predict_fusion(fusions[m], X_te) for m in raw
        ])
        L1[test_name] = preds
        print(f"  {test_name}: shape={preds.shape}, range=[{preds.min():.3f}, {preds.max():.3f}]")

    # ── Strategy 1: Simple average ──
    print("\n[6] Strategy 1: Simple average of all 4 fusions...")
    avg_results = {}
    for test_name in raw:
        _, X_te, _, y_te = splits[test_name]
        avg_pred = L1[test_name].mean(axis=1)
        avg_results[test_name] = eval_metrics(y_te, avg_pred)
        print(f"  {test_name}: ROC={avg_results[test_name]['roc_auc']:.4f}")

    # ── Strategy 2: Weighted average (weights from cross-domain in-domain AUC) ──
    print("\n[7] Strategy 2: Weighted average (weights ∝ in-domain AUC)...")
    # Compute in-domain AUCs for weighting
    weights = {}
    for name in raw:
        _, X_te, _, y_te = splits[name]
        pred = predict_fusion(fusions[name], X_te)
        weights[name] = roc_auc_score(y_te, pred)
    # Normalize
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}
    print(f"  Weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")

    weighted_results = {}
    for test_name in raw:
        _, X_te, _, y_te = splits[test_name]
        w = np.array([weights[m] for m in raw])
        weighted_pred = (L1[test_name] * w).sum(axis=1)
        weighted_results[test_name] = eval_metrics(y_te, weighted_pred)
        print(f"  {test_name}: ROC={weighted_results[test_name]['roc_auc']:.4f}")

    # ── Strategy 3: Best-of (pick highest-confidence model per sample) ──
    print("\n[8] Strategy 3: Best-of (argmax per sample)...")
    bestof_results = {}
    for test_name in raw:
        _, X_te, _, y_te = splits[test_name]
        l1 = L1[test_name]
        # For fraud detection: pick the model with highest score (most suspicious)
        best_pred = l1.max(axis=1)
        bestof_results[test_name] = eval_metrics(y_te, best_pred)
        print(f"  {test_name}: ROC={bestof_results[test_name]['roc_auc']:.4f}")

    # ── Strategy 4: Meta-learner (XGBoost on L1 predictions) ──
    print("\n[9] Strategy 4: Meta-learner (XGBoost on L1 predictions)...")

    # Train meta-learner: use leave-one-out cross-domain training
    # For each domain, train the meta-learner on the OTHER 3 domains' L1 predictions
    meta_learners = {}
    for holdout_name in raw:
        # Collect L1 predictions from the other 3 domains
        meta_X_parts = []
        meta_y_parts = []
        for train_name in raw:
            if train_name == holdout_name:
                continue
            X_tr, _, y_tr, _ = splits[train_name]
            # Get L1 predictions from all 4 fusions on this training domain
            l1_preds = np.column_stack([
                predict_fusion(fusions[m], X_tr) for m in raw
            ])
            meta_X_parts.append(l1_preds)
            meta_y_parts.append(y_tr)

        meta_X = np.vstack(meta_X_parts)
        meta_y = np.concatenate(meta_y_parts)

        # Train meta-learner
        try:
            from xgboost import XGBClassifier
            meta_xgb = XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.2,
                scale_pos_weight=(meta_y == 0).sum() / max((meta_y == 1).sum(), 1),
                eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0,
            )
            meta_xgb.fit(meta_X, meta_y)
            meta_learners[holdout_name] = meta_xgb
        except ImportError:
            meta_lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
            meta_lr.fit(meta_X, meta_y)
            meta_learners[holdout_name] = meta_lr

    # Evaluate meta-learner on each test set
    meta_results = {}
    for test_name in raw:
        _, X_te, _, y_te = splits[test_name]
        l1 = L1[test_name]
        meta_model = meta_learners[test_name]
        meta_pred = meta_model.predict_proba(l1)[:, 1]
        meta_results[test_name] = eval_metrics(y_te, meta_pred)
        print(f"  {test_name}: ROC={meta_results[test_name]['roc_auc']:.4f}")

    # ── Strategy 5: Universal meta-learner (trained on ALL L1 predictions) ──
    print("\n[10] Strategy 5: Universal meta-learner (all domains pooled)...")

    # Build pooled training data: each domain's L1 predictions + labels
    pool_X_parts = []
    pool_y_parts = []
    for name in raw:
        X_tr, _, y_tr, _ = splits[name]
        l1_preds = np.column_stack([
            predict_fusion(fusions[m], X_tr) for m in raw
        ])
        pool_X_parts.append(l1_preds)
        pool_y_parts.append(y_tr)

    pool_X = np.vstack(pool_X_parts)
    pool_y = np.concatenate(pool_y_parts)

    try:
        from xgboost import XGBClassifier
        universal_meta = XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.2,
            scale_pos_weight=(pool_y == 0).sum() / max((pool_y == 1).sum(), 1),
            eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0,
        )
        universal_meta.fit(pool_X, pool_y)
    except ImportError:
        universal_meta = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        universal_meta.fit(pool_X, pool_y)

    universal_results = {}
    for test_name in raw:
        _, X_te, _, y_te = splits[test_name]
        l1 = L1[test_name]
        universal_pred = universal_meta.predict_proba(l1)[:, 1]
        universal_results[test_name] = eval_metrics(y_te, universal_pred)
        print(f"  {test_name}: ROC={universal_results[test_name]['roc_auc']:.4f}")

    # ── Comparison table ──
    print("\n" + "=" * 72)
    print("STRATEGY COMPARISON (ROC-AUC)")
    print("=" * 72)
    header = f"{'Strategy':<30}" + "".join(f"{n[:14]:>15}" for n in raw) + f"{'Average':>10}"
    print(header)
    print("-" * len(header))

    # Single-domain best: for each test domain, pick the model trained on that domain
    single_results = {}
    for test_name in raw:
        _, X_te, _, y_te = splits[test_name]
        pred = predict_fusion(fusions[test_name], X_te)
        single_results[test_name] = eval_metrics(y_te, pred)

    strategies = [
        ("Single-domain (best)", single_results),
        ("Simple average", avg_results),
        ("Weighted average", weighted_results),
        ("Best-of (max)", bestof_results),
        ("Meta-learner (LOO)", meta_results),
        ("Universal meta", universal_results),
    ]

    # Pre-compute column bests
    col_bests = {}
    for test_name in raw:
        col_bests[test_name] = max(results[test_name]["roc_auc"] for _, results in strategies)

    for strat_name, results in strategies:
        row = f"{strat_name:<30}"
        aucs = []
        for test_name in raw:
            auc = results[test_name]["roc_auc"]
            aucs.append(auc)
            marker = " ★" if auc >= col_bests[test_name] - 0.001 else "  "
            row += f"{auc:>13.4f}{marker}"
        avg = np.mean(aucs)
        row += f"{avg:>10.4f}"
        print(row)

    # ── Detailed per-strategy breakdown ──
    print("\n" + "=" * 72)
    print("DETAILED METRICS (ROC-AUC / PR-AUC / Recall@1%FPR)")
    print("=" * 72)
    for strat_name, results in strategies:
        print(f"\n  {strat_name}:")
        for test_name in raw:
            r = results[test_name]
            print(f"    {test_name:<25} ROC={r['roc_auc']:.4f}  PR={r['pr_auc']:.4f}  R@1%={r['recall_at_1pct_fpr']:.4f}")

    # ── Improvement analysis ──
    print("\n" + "=" * 72)
    print("IMPROVEMENT OVER SINGLE-DOMAIN")
    print("=" * 72)
    strat_dict = {name: res for name, res in strategies}
    for strat_name in ["Simple average", "Weighted average", "Best-of (max)", "Meta-learner (LOO)", "Universal meta"]:
        gains = []
        for test_name in raw:
            single = single_results[test_name]["roc_auc"]
            multi = strat_dict[strat_name][test_name]["roc_auc"]
            gain = multi - single
            gains.append(gain)
        avg_gain = np.mean(gains)
        print(f"  {strat_name:<30} avg gain: {avg_gain:+.4f} ({'improved' if avg_gain > 0 else 'decreased'})")

    # ── Meta-learner feature importance ──
    print("\n" + "=" * 72)
    print("META-LEARNER: Which domain model matters most?")
    print("=" * 72)
    domain_names = list(raw.keys())
    if hasattr(universal_meta, "feature_importances_"):
        importances = universal_meta.feature_importances_
        for name, imp in sorted(zip(domain_names, importances), key=lambda x: -x[1]):
            bar = "█" * int(imp * 50)
            print(f"  {name:<25} {imp:.4f}  {bar}")
    elif hasattr(universal_meta, "coef_"):
        coefs = universal_meta.coef_[0]
        for name, c in sorted(zip(domain_names, coefs), key=lambda x: -abs(x[1])):
            bar = "█" * int(abs(c) * 5)
            print(f"  {name:<25} {c:+.4f}  {bar}")

    # Save results
    out = {
        "datasets": {name: {"rows": len(y), "positive": int(y.sum())}
                     for name, (X, y) in raw.items()},
        "strategies": {
            strat: {test: metrics for test, metrics in results.items()}
            for strat, results in strategies
        },
    }
    out_path = DATA / "meta_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
