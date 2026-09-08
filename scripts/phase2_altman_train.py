#!/usr/bin/env python3
"""Phase 2+3: IBM Altman 24M — feature engineering + train + evaluate."""
import numpy as np, pandas as pd, time, json, os, hashlib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

DATA_DIR, REPORTS_DIR = Path("data"), Path("reports")

def recall_at(y_true, y_score, fpr_target):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    v = fpr <= fpr_target
    return float(tpr[v][-1]) if v.any() else 0.0

def main():
    print("=" * 60)
    print("IBM ALTMAN 24M — FULL PIPELINE")
    print("=" * 60)
    t0 = time.time()

    # Load full dataset
    print("Loading 24M rows...")
    df = pd.read_csv(
        DATA_DIR / "credit_card_transactions-ibm_v2.csv",
        usecols=["User","Card","Year","Month","Day","Time","Amount","Use Chip",
                  "Merchant Name","Merchant City","Merchant State","Zip","MCC","Errors?","Is Fraud?"],
        low_memory=False,
    )
    print(f"  Loaded {len(df):,} in {time.time()-t0:.0f}s")

    # Parse
    df["amt"] = df["Amount"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float)
    df["is_fraud"] = (df["Is Fraud?"] == "Yes").astype(int)
    df["datetime"] = pd.to_datetime(df[["Year","Month","Day"]].assign(
        hour=df["Time"].str.split(":").str[0].astype(int),
        minute=df["Time"].str.split(":").str[1].astype(int),
    ))
    df = df.sort_values(["User","datetime"]).reset_index(drop=True)

    fraud_n = df["is_fraud"].sum()
    print(f"  Fraud: {fraud_n:,} ({fraud_n/len(df)*100:.3f}%)")

    # === FAST FEATURE ENGINEERING ===
    print("\n  Feature engineering (fast, vectorized)...")
    F = pd.DataFrame()

    # Amount
    F["amount"] = df["amt"].values
    F["amount_log"] = np.log1p(np.abs(df["amt"].values))
    F["is_negative"] = (df["amt"].values < 0).astype(np.float32)

    # User amount stats (cumulative, fast)
    g = df.groupby("User")
    F["user_amt_mean"] = g["amt"].cumsum() / (g.cumcount() + 1)
    F["user_amt_std"] = np.sqrt(np.clip(
        g["amt"].transform(lambda x: (x**2).cumsum()) / (g.cumcount()+1) - F["user_amt_mean"]**2,
        0, None
    ))
    F["user_amt_max"] = g["amt"].cummax()
    F["user_amt_zscore"] = (F["amount"] - F["user_amt_mean"]) / (F["user_amt_std"] + 1e-8)
    F["user_amt_ratio"] = F["amount"] / (F["user_amt_mean"].clip(lower=1))

    # Temporal
    F["hour"] = df["datetime"].dt.hour.values.astype(np.float32)
    F["dow"] = df["datetime"].dt.dayofweek.values.astype(np.float32)
    F["is_weekend"] = (F["dow"] >= 5).astype(np.float32)
    F["is_night"] = ((F["hour"] >= 22) | (F["hour"] <= 5)).astype(np.float32)
    F["is_business"] = ((F["hour"] >= 9) & (F["hour"] <= 17)).astype(np.float32)
    F["hours_since_last"] = np.clip(
        df.groupby("User")["datetime"].diff().dt.total_seconds().fillna(0).values / 3600.0,
        0, 720
    ).astype(np.float32)

    # Velocity
    F["txn_count"] = (g.cumcount() + 1).values.astype(np.float32)

    # MCC
    F["mcc"] = df["MCC"].values.astype(np.float32)
    F["mcc_fraud_rate"] = df.groupby("MCC")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    ).values.astype(np.float32)

    # Merchant
    F["merchant_code"] = df["Merchant Name"].astype("category").cat.codes.values.astype(np.float32)
    F["merchant_fraud_rate"] = df.groupby("Merchant Name")["is_fraud"].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    ).values.astype(np.float32)

    # Seen-before counts (fast cumulative groupby)
    F["merchant_seen_before"] = df.groupby(["User","Merchant Name"]).cumcount().values.astype(np.float32)
    F["city_seen_before"] = df.groupby(["User","Merchant City"]).cumcount().values.astype(np.float32)
    F["mcc_seen_before"] = df.groupby(["User","MCC"]).cumcount().values.astype(np.float32)

    # Card/Chip/Location
    F["card_num"] = df["Card"].values.astype(np.float32)
    chip_map = {"Swipe Transaction": 0, "Chip Transaction": 1, "Online Transaction": 2}
    F["chip_type"] = df["Use Chip"].map(chip_map).fillna(3).values.astype(np.float32)
    F["has_error"] = df["Errors?"].notna().values.astype(np.float32)
    F["city_code"] = df["Merchant City"].astype("category").cat.codes.values.astype(np.float32)
    F["state_code"] = df["Merchant State"].fillna("UNK").astype("category").cat.codes.values.astype(np.float32)
    F["has_zip"] = df["Zip"].notna().values.astype(np.float32)

    # Interactions
    F["amt_x_night"] = F["is_night"] * F["amount_log"]
    F["amt_x_new_merchant"] = (F["merchant_seen_before"] == 0).astype(np.float32) * F["amount_log"]

    # New merchant flag
    F["is_new_merchant"] = (F["merchant_seen_before"] == 0).astype(np.float32)
    F["is_new_city"] = (F["city_seen_before"] == 0).astype(np.float32)
    F["is_new_mcc"] = (F["mcc_seen_before"] == 0).astype(np.float32)

    # Clean
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0)
    F["label"] = df["is_fraud"].values

    print(f"  Features: {F.shape[0]:,} x {F.shape[1]}")
    print(f"  Feature engineering: {time.time()-t0:.0f}s total")

    # Feature separation
    fraud_mask = F["label"] == 1
    print("\n  Feature separation (fraud vs legit):")
    for col in F.columns:
        if col == "label":
            continue
        fm = F.loc[fraud_mask, col].mean()
        lm = F.loc[~fraud_mask, col].mean()
        fs = F.loc[fraud_mask, col].std() + 1e-8
        sep = abs(fm - lm) / fs
        if sep > 0.1:
            print(f"    {col:30s}: fraud={fm:10.3f}  legit={lm:10.3f}  sep={sep:.3f}")

    # === TRAIN + EVALUATE ===
    feat_cols = [c for c in F.columns if c != "label"]
    X = F[feat_cols].values.astype(np.float32)
    y = F["label"].values

    # Subsample: keep ALL fraud, sample 500K legit for training speed
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    n_legit_sub = min(500_000, len(legit_idx))
    sub_legit = np.random.RandomState(42).choice(legit_idx, n_legit_sub, replace=False)
    sub_idx = np.concatenate([fraud_idx, sub_legit])
    np.random.RandomState(42).shuffle(sub_idx)
    X_sub = X[sub_idx]
    y_sub = y[sub_idx]
    print(f"  Subsample: {len(X_sub):,} rows ({y_sub.sum()} fraud)")

    Xtr, Xte, ytr, yte = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    print(f"  Train: {len(Xtr):,} ({ytr.sum()} fraud), Test: {len(Xte):,} ({yte.sum()} fraud)")

    results = {}

    # LR
    t1 = time.time()
    lr = LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(Xtr_s, ytr)
    p_lr = lr.predict_proba(Xte_s)[:, 1]
    t_lr = time.time() - t1
    auc = roc_auc_score(yte, p_lr)
    pr = average_precision_score(yte, p_lr)
    r1 = recall_at(yte, p_lr, 0.01)
    r05 = recall_at(yte, p_lr, 0.005)
    r01 = recall_at(yte, p_lr, 0.001)
    print(f"  LR:     AUC={auc:.4f}  PR={pr:.4f}  R@1%={r1:.4f}  R@0.5%={r05:.4f}  R@0.1%={r01:.4f}  ({t_lr:.0f}s)")
    results["lr"] = {"roc_auc": auc, "pr_auc": pr, "recall_1pct": r1, "recall_0_5pct": r05, "recall_0_1pct": r01}

    # RF
    t1 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    p_rf = rf.predict_proba(Xte)[:, 1]
    t_rf = time.time() - t1
    auc = roc_auc_score(yte, p_rf)
    pr = average_precision_score(yte, p_rf)
    r1 = recall_at(yte, p_rf, 0.01)
    r05 = recall_at(yte, p_rf, 0.005)
    r01 = recall_at(yte, p_rf, 0.001)
    print(f"  RF:     AUC={auc:.4f}  PR={pr:.4f}  R@1%={r1:.4f}  R@0.5%={r05:.4f}  R@0.1%={r01:.4f}  ({t_rf:.0f}s)")
    results["rf"] = {"roc_auc": auc, "pr_auc": pr, "recall_1pct": r1, "recall_0_5pct": r05, "recall_0_1pct": r01}

    # XGB
    try:
        from xgboost import XGBClassifier
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        t1 = time.time()
        xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8, scale_pos_weight=spw, random_state=42, eval_metric="aucpr", use_label_encoder=False, n_jobs=-1)
        xgb.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        p_xgb = xgb.predict_proba(Xte)[:, 1]
        t_xgb = time.time() - t1
        auc = roc_auc_score(yte, p_xgb)
        pr = average_precision_score(yte, p_xgb)
        r1 = recall_at(yte, p_xgb, 0.01)
        r05 = recall_at(yte, p_xgb, 0.005)
        r01 = recall_at(yte, p_xgb, 0.001)
        print(f"  XGB:    AUC={auc:.4f}  PR={pr:.4f}  R@1%={r1:.4f}  R@0.5%={r05:.4f}  R@0.1%={r01:.4f}  ({t_xgb:.0f}s)")
        results["xgb"] = {"roc_auc": auc, "pr_auc": pr, "recall_1pct": r1, "recall_0_5pct": r05, "recall_0_1pct": r01}

        # Feature importance
        print("\n  Top 15 features (XGB):")
        imp = xgb.feature_importances_
        for i in np.argsort(imp)[::-1][:15]:
            print(f"    {feat_cols[i]:35s} {imp[i]:.4f}")
    except Exception as e:
        print(f"  XGB error: {e}")

    # Stacker
    print("\n--- Stacker Ensemble ---")
    preds_dict = {"lr": p_lr, "rf": p_rf}
    if "xgb" in results:
        preds_dict["xgb"] = p_xgb
    meta_names = list(preds_dict.keys())
    meta_test = np.column_stack([preds_dict[n] for n in meta_names])

    # OOF for meta training
    from xgboost import XGBClassifier
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    meta_oof = np.zeros((len(ytr), len(meta_names)))
    for j, name in enumerate(meta_names):
        if name == "lr":
            meta_oof[:, j] = cross_val_predict(LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42), Xtr_s, ytr, cv=skf, method="predict_proba")[:, 1]
        elif name == "rf":
            meta_oof[:, j] = cross_val_predict(RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1), Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
        elif name == "xgb":
            spw_meta = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            meta_oof[:, j] = cross_val_predict(XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=spw_meta, random_state=42, eval_metric="aucpr", use_label_encoder=False, n_jobs=-1), Xtr, ytr, cv=skf, method="predict_proba")[:, 1]

    meta = LogisticRegression(C=10, max_iter=1000, random_state=42)
    meta.fit(meta_oof, ytr)
    p_stack = meta.predict_proba(meta_test)[:, 1]
    auc = roc_auc_score(yte, p_stack)
    pr = average_precision_score(yte, p_stack)
    r1 = recall_at(yte, p_stack, 0.01)
    r05 = recall_at(yte, p_stack, 0.005)
    r01 = recall_at(yte, p_stack, 0.001)
    print(f"  Stacker: AUC={auc:.4f}  PR={pr:.4f}  R@1%={r1:.4f}  R@0.5%={r05:.4f}  R@0.1%={r01:.4f}")
    results["stacker"] = {"roc_auc": auc, "pr_auc": pr, "recall_1pct": r1, "recall_0_5pct": r05, "recall_0_1pct": r01}

    # Bootstrap CI for best
    best = max(results, key=lambda k: results[k]["roc_auc"])
    bp = preds_dict.get(best, p_stack)
    if best == "stacker":
        bp = p_stack
    ci_lo = []
    ci_hi = []
    rng = np.random.RandomState(42)
    for _ in range(200):
        idx = rng.choice(len(yte), len(yte), replace=True)
        if len(np.unique(yte[idx])) < 2:
            continue
        ci_lo.append(roc_auc_score(yte[idx], bp[idx]))
    ci_lo_sorted = sorted(ci_lo)
    ci = (ci_lo_sorted[int(len(ci_lo_sorted)*0.025)], ci_lo_sorted[int(len(ci_lo_sorted)*0.975)])
    print(f"\n  Best: {best} AUC={results[best]['roc_auc']:.4f}  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Save
    output = {
        "dataset": "ibm_altman_v2",
        "n_rows": int(len(df)),
        "n_fraud": int(fraud_n),
        "fraud_rate": float(fraud_n/len(df)),
        "n_features": len(feat_cols),
        "features": feat_cols,
        "results": results,
        "best_model": best,
        "ci_95": list(ci),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = REPORTS_DIR / "altman_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")
    print(f"  Total time: {time.time()-t0:.0f}s")

    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("BRUTALLY TRUE RESULTS — IBM ALTMAN 24M")
    print("=" * 70)
    print(f"{'Model':<12} {'ROC-AUC':>10} {'PR-AUC':>10} {'R@1%FPR':>10} {'R@0.5%FPR':>10} {'R@0.1%FPR':>10}")
    print("-" * 64)
    for k, v in sorted(results.items(), key=lambda x: -x[1]["roc_auc"]):
        print(f"  {k:<10} {v['roc_auc']:>10.4f} {v['pr_auc']:>10.4f} {v['recall_1pct']:>10.4f} {v['recall_0_5pct']:>10.4f} {v['recall_0_1pct']:>10.4f}")


if __name__ == "__main__":
    from sklearn.model_selection import cross_val_predict
    main()
