#!/usr/bin/env python3
"""Unified Cross-Domain Fraud Detection

Maps ULB (PCA), Altman (merchant/user), and PaySim (balance) into a shared
15-dimensional domain-agnostic feature space, then evaluates:
1. In-domain performance on unified features
2. Cross-domain transfer (train on one, test on others)
3. Combined training across all three datasets
"""
import hashlib, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
np.random.seed(42)


# ═══════════════════════════════════════════════════════
# UNIFIED FEATURE SCHEMA (15 domain-agnostic features)
# ═══════════════════════════════════════════════════════
UNIFIED_FEATURES = [
    # Magnitude
    "amount_log",          # Log transaction size
    "amount_zscore",       # Deviation from typical amount
    
    # Timing
    "hour_sin",            # Cyclical hour encoding
    "hour_cos",
    "is_night",            # Late-night flag
    "is_weekend",          # Weekend flag
    
    # PCA Anomaly (for ULB; 0 for others)
    "pca_magnitude",       # PCA vector magnitude
    "pca_skew",            # PCA distribution skewness
    "pca_extreme_count",   # Count of extreme PCA components
    
    # Balance (for PaySim; 0 for others)
    "balance_drain_ratio", # Fraction of balance drained
    "dest_balance_change", # Destination balance delta
    
    # Network (for Altman; 0 for others)
    "entity_seen_before",  # Merchant/account seen before
    "chip_transaction",    # Chip vs online
    "is_new_entity",       # First-time merchant flag
    
    # Universal
    "has_error",           # Transaction error flag
]


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


def full_ev(y, p, prefix=""):
    thr = find_thr(y, p)
    yp = (p >= thr).astype(int)
    return {
        f"{prefix}roc_auc": round(roc_auc_score(y, p), 6),
        f"{prefix}pr_auc": round(average_precision_score(y, p), 6),
        f"{prefix}brier": round(brier_score_loss(y, p), 6),
        f"{prefix}r1": round(rafpr(y, p, 0.01), 6),
        f"{prefix}r05": round(rafpr(y, p, 0.005), 6),
        f"{prefix}r01": round(rafpr(y, p, 0.001), 6),
        f"{prefix}opt_thr": round(thr, 4),
        f"{prefix}f1": round(f1_score(y, yp), 6),
        f"{prefix}n_pos": int(np.sum(y == 1)),
        f"{prefix}n_neg": int(np.sum(y == 0)),
    }


# ═══════════════════════════════════════════════════════
# EXTRACT UNIFIED FEATURES FROM EACH DATASET
# ═══════════════════════════════════════════════════════

def extract_ulb():
    """ULB Creditcard: PCA components → unified features."""
    print("  Loading ULB...")
    df = pd.read_csv("data/creditcard.csv")
    y = df["Class"].values
    n = len(df)

    # Amount features
    df["amount_log"] = np.log1p(df["Amount"])
    amt_mean = df["Amount"].mean()
    amt_std = df["Amount"].std()
    df["amount_zscore"] = (df["Amount"] - amt_mean) / (amt_std + 1e-8)

    # Time features (ULB Time is seconds from start)
    df["hour"] = (df["Time"] / 3600) % 24
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(np.float32)
    df["is_weekend"] = ((df["Time"] / 86400).astype(int) % 7 >= 5).astype(np.float32)

    # PCA features from V1-V28
    V_cols = [c for c in df.columns if c.startswith("V")]
    v = df[V_cols].values.astype(np.float32)
    df["pca_magnitude"] = np.sqrt((v ** 2).sum(axis=1))
    df["pca_skew"] = pd.DataFrame(v).skew(axis=1).values
    df["pca_extreme_count"] = (np.abs(v) > 3).sum(axis=1).astype(np.float32)

    # No balance/network features for ULB
    df["balance_drain_ratio"] = 0.0
    df["dest_balance_change"] = 0.0
    df["entity_seen_before"] = 1.0  # all entities "seen" in PCA space
    df["chip_transaction"] = 0.0     # unknown
    df["is_new_entity"] = 0.0
    df["has_error"] = 0.0

    X = df[UNIFIED_FEATURES].values.astype(np.float32)
    print(f"  ULB: {n:,} rows, fraud={y.sum():,}")
    return X, y, "ulb_creditcard"


def extract_altman():
    """IBM Altman: merchant/user features → unified features."""
    print("  Loading Altman (3M rows)...")
    df = pd.read_csv("data/credit_card_transactions-ibm_v2.csv", nrows=3_000_000)
    df["label"] = df["Is Fraud?"].map(lambda x: 1 if str(x).strip() == "Yes" else 0)
    y = df["label"].values

    if "Amount" in df.columns:
        df["amount"] = pd.to_numeric(
            df["Amount"].astype(str).str.replace("$", "").str.replace(",", ""),
            errors="coerce"
        ).fillna(0)
    else:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["amount_log"] = np.log1p(df["amount"])
    amt_mean = df["amount"].mean()
    amt_std = df["amount"].std()
    df["amount_zscore"] = (df["amount"] - amt_mean) / (amt_std + 1e-8)

    # Time
    df["hour"] = df["Time"].astype(str).apply(
        lambda x: int(x.split(":")[0]) if ":" in str(x) else 12
    )
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(np.float32)
    df["datetime"] = pd.to_datetime(
        df[["Year", "Month", "Day"]].rename(columns={"Year": "year", "Month": "month", "Day": "day"})
    )
    df["dow"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(np.float32)

    # No PCA for Altman
    df["pca_magnitude"] = 0.0
    df["pca_skew"] = 0.0
    df["pca_extreme_count"] = 0.0

    # No balance for Altman
    df["balance_drain_ratio"] = 0.0
    df["dest_balance_change"] = 0.0

    # Network features
    df["merchant_hash"] = df["Merchant Name"].astype("category").cat.codes
    df["user_hash"] = df["User"].astype("category").cat.codes
    # Seen before: cumulative merchant count per user
    df = df.sort_values(["user_hash", "datetime"]).reset_index(drop=True)
    df["merchant_cumcount"] = df.groupby("user_hash").cumcount()
    df["entity_seen_before"] = (df["merchant_cumcount"] > 0).astype(np.float32)
    df["chip_transaction"] = df["Use Chip"].map({"Chip": 1, "Swipe": 0.5, "Online": 0}).fillna(0).astype(np.float32)
    df["is_new_entity"] = (df["merchant_cumcount"] == 0).astype(np.float32)
    df["has_error"] = (df["Errors?"].fillna("") != "").astype(np.float32)

    X = df[UNIFIED_FEATURES].values.astype(np.float32)
    print(f"  Altman: {len(df):,} rows, fraud={y.sum():,}")
    return X, y, "ibm_altman"


def extract_paysim():
    """PaySim: balance/transaction features → unified features."""
    print("  Loading PaySim...")
    df = pd.read_csv("data/paysim_1m.csv")
    y = df["isFraud"].values.astype(int)

    df["amount_log"] = np.log1p(df["amount"].fillna(0))
    amt_mean = df["amount"].mean()
    amt_std = df["amount"].std()
    df["amount_zscore"] = (df["amount"] - amt_mean) / (amt_std + 1e-8)

    # No temporal info in paysim_1m (no step column)
    df["hour"] = 12.0  # neutral default
    df["hour_sin"] = 0.0
    df["hour_cos"] = 1.0
    df["is_night"] = 0.0
    df["is_weekend"] = 0.0

    # No PCA
    df["pca_magnitude"] = 0.0
    df["pca_skew"] = 0.0
    df["pca_extreme_count"] = 0.0

    # Balance features (the key PaySim signal)
    df["oldbalance"] = df.get("oldbalanceOrg", pd.Series(0, index=df.index)).fillna(0)
    df["newbalance"] = df.get("newbalanceOrig", pd.Series(0, index=df.index)).fillna(0)
    df["dest_old"] = df.get("oldbalanceDest", pd.Series(0, index=df.index)).fillna(0)
    df["dest_new"] = df.get("newbalanceDest", pd.Series(0, index=df.index)).fillna(0)
    df["balance_drain_ratio"] = np.where(
        df["oldbalance"] > 0,
        (df["oldbalance"] - df["newbalance"]) / df["oldbalance"],
        0
    ).clip(-10, 10).astype(np.float32)
    df["dest_balance_change"] = (df["dest_new"] - df["dest_old"]).astype(np.float32)

    # No network for PaySim
    df["entity_seen_before"] = 0.0
    df["chip_transaction"] = 0.0
    df["is_new_entity"] = 1.0
    df["has_error"] = 0.0

    X = df[UNIFIED_FEATURES].values.astype(np.float32)
    print(f"  PaySim: {len(df):,} rows, fraud={y.sum():,}")
    return X, y, "paysim_1m"


# ═══════════════════════════════════════════════════════
# MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════

def train_and_eval(Xtr, ytr, Xte, yte, label=""):
    """Train XGB on unified features, return full metrics."""
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    m = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=1,
        random_state=42, eval_metric="logloss", n_jobs=-1
    )
    t0 = time.time()
    m.fit(Xtr_s, ytr)
    p = m.predict_proba(Xte_s)[:, 1]
    elapsed = time.time() - t0

    ev = full_ev(yte, p, prefix=f"{label}_")
    ev[f"{label}_time"] = round(elapsed, 2)
    return ev, m, sc


def cross_domain_eval(Xsrc, ysrc, Xtgt, ytgt, src_name, tgt_name, sc_src=None):
    """Train on source, test on target. Uses source scaler."""
    if sc_src is None:
        sc_src = StandardScaler()
        sc_src.fit(Xsrc)
    Xsrc_s = sc_src.transform(Xsrc)
    Xtgt_s = sc_src.transform(Xtgt)

    m = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=1,
        random_state=42, eval_metric="logloss", n_jobs=-1
    )
    t0 = time.time()
    m.fit(Xsrc_s, ysrc)
    p = m.predict_proba(Xtgt_s)[:, 1]
    elapsed = time.time() - t0

    label = f"{src_name}_to_{tgt_name}"
    ev = full_ev(ytgt, p, prefix=f"{label}_")
    ev[f"{label}_time"] = round(elapsed, 2)
    return ev


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED CROSS-DOMAIN FRAUD DETECTION")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Unified features: {len(UNIFIED_FEATURES)}")
    print("=" * 70)

    # ── Step 1: Extract unified features from all 3 datasets ──
    print("\n── STEP 1: Extract unified features ──")
    X_ulb, y_ulb, name_ulb = extract_ulb()
    X_alt, y_alt, name_alt = extract_altman()
    X_pay, y_pay, name_pay = extract_paysim()

    # ── Step 2: In-domain evaluation (each dataset alone) ──
    print("\n── STEP 2: In-domain evaluation ──")
    in_domain = {}

    # ULB
    Xtr, Xte, ytr, yte = train_test_split(X_ulb, y_ulb, test_size=0.2, stratify=y_ulb, random_state=42)
    ev, _, sc_ulb = train_and_eval(Xtr, ytr, Xte, yte, "ulb")
    in_domain["ulb"] = ev
    print(f"  ULB:  AUC={ev['ulb_roc_auc']:.4f} R1%={ev['ulb_r1']:.4f}")

    # Altman (user-disjoint)
    rng = np.random.RandomState(42)
    all_idx = np.arange(len(X_alt))
    # Use last 20% of rows as test (rough temporal split)
    split = int(len(X_alt) * 0.8)
    Xtr_a, Xte_a = X_alt[:split], X_alt[split:]
    ytr_a, yte_a = y_alt[:split], y_alt[split:]
    ev, _, sc_alt = train_and_eval(Xtr_a, ytr_a, Xte_a, yte_a, "alt")
    in_domain["altman"] = ev
    print(f"  Altman: AUC={ev['alt_roc_auc']:.4f} R1%={ev['alt_r1']:.4f}")

    # PaySim
    Xtr, Xte, ytr, yte = train_test_split(X_pay, y_pay, test_size=0.2, stratify=y_pay, random_state=42)
    ev, _, sc_pay = train_and_eval(Xtr, ytr, Xte, yte, "pay")
    in_domain["paysim"] = ev
    print(f"  PaySim: AUC={ev['pay_roc_auc']:.4f} R1%={ev['pay_r1']:.4f}")

    # ── Step 3: Cross-domain transfer (train on one, test on others) ──
    print("\n── STEP 3: Cross-domain transfer ──")
    cross_domain = {}

    # Use full datasets for cross-domain (train on all of source, test on all of target)
    pairs = [
        (X_ulb, y_ulb, "ulb", sc_ulb),
        (X_alt, y_alt, "altman", sc_alt),
        (X_pay, y_pay, "paysim", sc_pay),
    ]

    for Xsrc, ysrc, src, sc_src in pairs:
        for Xtgt, ytgt, tgt, _ in pairs:
            if src == tgt:
                continue
            ev = cross_domain_eval(Xsrc, ysrc, Xtgt, ytgt, src, tgt, sc_src)
            key = f"{src}_to_{tgt}"
            cross_domain[key] = ev
            auc = ev.get(f"{key}_roc_auc", 0)
            r1 = ev.get(f"{key}_r1", 0)
            print(f"  {key}: AUC={auc:.4f} R1%={r1:.4f}")

    # ── Step 4: Combined training (all 3 datasets together) ──
    print("\n── STEP 4: Combined training (all 3) ──")
    X_all = np.vstack([X_ulb, X_alt, X_pay])
    y_all = np.concatenate([y_ulb, y_alt, y_pay])
    print(f"  Combined: {len(X_all):,} rows, fraud={y_all.sum():,}")

    # Train on combined, test on each dataset's holdout
    # Resplit each dataset for held-out test
    Xtr_u, Xte_u, ytr_u, yte_u = train_test_split(X_ulb, y_ulb, test_size=0.2, stratify=y_ulb, random_state=42)
    Xtr_a, Xte_a, ytr_a, yte_a = X_alt[:split], X_alt[split:], y_alt[:split], y_alt[split:]
    Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(X_pay, y_pay, test_size=0.2, stratify=y_pay, random_state=42)

    # Train on all training splits combined
    X_train_combined = np.vstack([Xtr_u, Xtr_a, Xtr_p])
    y_train_combined = np.concatenate([ytr_u, ytr_a, ytr_p])

    sc_comb = StandardScaler()
    X_train_combined_s = sc_comb.fit_transform(X_train_combined)

    m_combined = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=1,
        random_state=42, eval_metric="logloss", n_jobs=-1
    )
    t0 = time.time()
    m_combined.fit(X_train_combined_s, y_train_combined)
    t_train = time.time() - t0
    print(f"  Training time: {t_train:.1f}s")

    combined_results = {}
    for name, Xte, yte in [("ulb", Xte_u, yte_u), ("altman", Xte_a, yte_a), ("paysim", Xte_p, yte_p)]:
        Xte_s = sc_comb.transform(Xte)
        p = m_combined.predict_proba(Xte_s)[:, 1]
        ev = full_ev(yte, p, prefix=f"combined_{name}_")
        combined_results[name] = ev
        auc = ev.get(f"combined_{name}_roc_auc", 0)
        r1 = ev.get(f"combined_{name}_r1", 0)
        print(f"  Combined→{name}: AUC={auc:.4f} R1%={r1:.4f}")

    # ── Step 5: Print summary ──
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN RESULTS SUMMARY")
    print("=" * 70)

    print("\n  IN-DOMAIN (unified features):")
    for ds, m in in_domain.items():
        roc_key = next(k for k in m if k.endswith("roc_auc"))
        r1_key = next(k for k in m if k.endswith("r1"))
        pr_key = next(k for k in m if k.endswith("pr_auc"))
        print(f"    {ds:<12} ROC-AUC={m[roc_key]:.4f}  PR-AUC={m[pr_key]:.4f}  R@1%FPR={m[r1_key]:.4f}")

    print("\n  CROSS-DOMAIN TRANSFER:")
    for pair, m in cross_domain.items():
        roc_key = next(k for k in m if k.endswith("roc_auc"))
        r1_key = next(k for k in m if k.endswith("r1"))
        print(f"    {pair:<28} ROC-AUC={m[roc_key]:.4f}  R@1%FPR={m[r1_key]:.4f}")

    print("\n  COMBINED TRAINING → test on each:")
    for ds, m in combined_results.items():
        roc_key = next(k for k in m if k.endswith("roc_auc"))
        r1_key = next(k for k in m if k.endswith("r1"))
        pr_key = next(k for k in m if k.endswith("pr_auc"))
        print(f"    Combined→{ds:<14} ROC-AUC={m[roc_key]:.4f}  PR-AUC={m[pr_key]:.4f}  R@1%FPR={m[r1_key]:.4f}")

    # ── Save ──
    out = {
        "title": "Unified Cross-Domain Evaluation",
        "unified_features": UNIFIED_FEATURES,
        "n_features": len(UNIFIED_FEATURES),
        "in_domain": in_domain,
        "cross_domain": cross_domain,
        "combined_training": combined_results,
        "datasets": {
            "ulb": {"rows": len(X_ulb), "fraud": int(y_ulb.sum()), "features": "PCA V1-V28 → unified"},
            "altman": {"rows": len(X_alt), "fraud": int(y_alt.sum()), "features": "merchant/user → unified"},
            "paysim": {"rows": len(X_pay), "fraud": int(y_pay.sum()), "features": "balance/transaction → unified"},
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open("reports/cross_domain_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved reports/cross_domain_results.json")
    print("=" * 70)
