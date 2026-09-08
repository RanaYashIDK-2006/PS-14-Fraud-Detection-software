#!/usr/bin/env python3
"""Pattern Transfer Analysis: Which fraud patterns are universal vs domain-specific.

Maps features from all three datasets to semantic concepts, then compares
feature importance rankings to identify:
  1. Universal fraud signals (high importance across all domains)
  2. Domain-specific signals (high importance in one domain only)
  3. Negative transfer signals (important in source, harmful on target)
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
NJ = 4


# ── Semantic Concept Mapping ──

CONCEPT_MAP = {
    # Amount patterns (universal)
    "Amount": "amount_magnitude",
    "amount": "amount_magnitude",
    "amt": "amount_magnitude",
    "amount_log": "amount_magnitude",
    "amt_log": "amount_magnitude",
    "amt_sq": "amount_magnitude",
    "amount_sq": "amount_magnitude",
    "high_amount": "amount_magnitude",
    "amount_bucket": "amount_magnitude",
    "amount_zscore": "amount_deviation",
    "amt_ratio_orig": "amount_deviation",
    "drain_ratio": "amount_deviation",

    # Time patterns
    "Time": "time_pattern",
    "hr": "time_pattern",
    "hour_sin": "time_pattern",
    "hour_cos": "time_pattern",
    "is_night": "time_pattern",
    "night_tx": "time_pattern",
    "is_weekend": "time_pattern",
    "day_decimal": "time_pattern",
    "Day": "time_pattern",
    "mn": "time_pattern",
    "Month": "time_pattern",
    "Year": "time_pattern",
    "amount_x_hour": "amount_x_time",

    # PCA components (ULB-specific fraud signals)
    "V1": "pca_component", "V2": "pca_component", "V3": "pca_component",
    "V4": "pca_component", "V5": "pca_component", "V6": "pca_component",
    "V7": "pca_component", "V8": "pca_component", "V9": "pca_component",
    "V10": "pca_component", "V11": "pca_component", "V12": "pca_component",
    "V13": "pca_component", "V14": "pca_component", "V15": "pca_component",
    "V16": "pca_component", "V17": "pca_component", "V18": "pca_component",
    "V19": "pca_component", "V20": "pca_component", "V21": "pca_component",
    "V22": "pca_component", "V23": "pca_component", "V24": "pca_component",
    "V25": "pca_component", "V26": "pca_component", "V27": "pca_component",
    "V28": "pca_component",

    # PCA velocity (ULB-specific)
    "V14_dev": "pca_velocity", "V17_dev": "pca_velocity",
    "V12_dev": "pca_velocity", "V10_dev": "pca_velocity",
    "V4_dev": "pca_velocity", "V11_dev": "pca_velocity",
    "pca_anomaly_sum": "pca_velocity",
    "amt_x_V14": "pca_x_amount", "amt_x_V17": "pca_x_amount",

    # Merchant/location patterns (Altman-specific)
    "merchant_id": "merchant_identity",
    "city_id": "location_identity",
    "Merchant Name": "merchant_identity",
    "Merchant City": "location_identity",
    "is_online": "channel_type",
    "chip": "channel_type",
    "chip_type": "channel_type",
    "mcc_n": "merchant_category",
    "MCC": "merchant_category",
    "err": "transaction_error",
    "Card": "card_identity",
    "User": "user_identity",

    # Merchant/location velocity (Altman-specific)
    "merchant_tx_count": "merchant_velocity",
    "city_tx_count": "location_velocity",
    "night_deviation": "time_deviation",

    # Balance patterns (PaySim-specific)
    "oldbalanceOrg": "balance_state",
    "newbalanceOrig": "balance_state",
    "oldbalanceDest": "balance_state",
    "newbalanceDest": "balance_state",
    "bal_diff_orig": "balance_change",
    "bal_diff_dest": "balance_change",
    "orig_wiped": "balance_wipe",
    "dest_zero": "balance_wipe",
    "flow_asym": "balance_asymmetry",

    # Transaction type (PaySim-specific)
    "is_cashout": "transaction_type",
    "is_transfer": "transaction_type",
    "type_CASH_IN": "transaction_type",
    "type_CASH_OUT": "transaction_type",
    "type_DEBIT": "transaction_type",
    "type_PAYMENT": "transaction_type",
    "type_TRANSFER": "transaction_type",

    # Balance velocity (PaySim-specific)
    "balance_wipe_score": "balance_velocity",
    "amt_ratio_orig": "amount_deviation",

    # Interaction features
    "amt_x_hr": "amount_x_time",
    "amt_x_mcc": "amount_x_category",
    "amt_x_chip": "amount_x_channel",
    "amount_x_hour": "amount_x_time",
}


def get_concept(feature_name: str) -> str:
    """Map a feature name to its semantic concept."""
    if feature_name in CONCEPT_MAP:
        return CONCEPT_MAP[feature_name]
    # Fallback heuristics
    if feature_name.startswith("V") and feature_name[1:].isdigit():
        return "pca_component"
    if "dev" in feature_name.lower():
        return "velocity_other"
    if "roll" in feature_name.lower():
        return "rolling_stat"
    return "other"


# ── Dataset Loading (same as cross_dataset_eval.py) ──

def load_ulb():
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    # Build domain features
    feats = {}
    feats["Amount"] = df["Amount"]
    feats["Time"] = df["Time"]
    for v in range(1, 29):
        feats[f"V{v}"] = df[f"V{v}"]
    # Velocity
    for v in ["V14", "V17", "V12", "V10", "V4", "V11"]:
        rm = df[v].rolling(200, min_periods=1).mean()
        feats[f"{v}_dev"] = df[v] - rm
    devs = [f"{v}_dev" for v in ["V14", "V17", "V12", "V10", "V4", "V11"]]
    feat_df = pd.DataFrame(feats)
    feat_df["pca_anomaly_sum"] = feat_df[devs].abs().sum(axis=1)
    feat_df["amt_x_V14"] = df["Amount"] * df["V14"]
    feat_df["amt_x_V17"] = df["Amount"] * df["V17"]
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feat_df, df["Class"].values.astype(int), "ULB"


def load_altman():
    rng = np.random.RandomState(42)
    rows_all = []
    ci = 0
    for chunk in pd.read_csv(ROOT / "data" / "credit_card_transactions-ibm_v2.csv",
        usecols=["User", "Month", "Day", "Time", "Amount", "Use Chip", "MCC", "Errors?",
                 "Is Fraud?", "Merchant Name", "Merchant City", "Merchant State", "Zip", "Year", "Card"],
        low_memory=False, chunksize=1_000_000):
        fm = chunk["Is Fraud?"] == "Yes"
        rows_all.append(pd.concat([chunk[fm], chunk[~fm].sample(frac=0.01, random_state=rng)]))
        ci += 1
        if ci >= 16: break
    raw = pd.concat(rows_all, ignore_index=True)

    feats = {}
    chip_map = {"Swipe Transaction": 0, "Online Transaction": 1, "Chip Transaction": 2}
    feats["chip"] = raw["Use Chip"].map(chip_map).fillna(-1).astype(float)
    feats["mcc_n"] = pd.to_numeric(raw["MCC"], errors="coerce").fillna(0) / 10000
    feats["err"] = (raw["Errors?"].fillna("") != "").astype(float)
    tp = raw["Time"].str.split(":", expand=True)
    feats["hr"] = pd.to_numeric(tp[0], errors="coerce").fillna(12)
    feats["mn"] = pd.to_numeric(tp[1], errors="coerce").fillna(0)
    feats["merchant_id"] = raw["Merchant Name"].astype("category").cat.codes.astype(float)
    feats["city_id"] = raw["Merchant City"].astype("category").cat.codes.astype(float)
    feats["is_online"] = (raw["Merchant City"] == "ONLINE").astype(float)
    feats["day_decimal"] = raw["Day"] + feats["hr"] / 24 + feats["mn"] / 1440
    feats["Month"] = raw["Month"].astype(float)
    feats["Year"] = raw["Year"].astype(float)
    feats["Card"] = raw["Card"].astype(float)
    amt = pd.to_numeric(raw["Amount"].str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce").fillna(0)
    feats["amt"] = amt
    feats["amt_log"] = np.log1p(amt.clip(upper=1e9))
    feats["amt_x_hr"] = amt * feats["hr"]
    feats["amt_x_mcc"] = amt * feats["mcc_n"]
    feats["amt_x_chip"] = amt * feats["chip"]
    feats["amt_sq"] = amt ** 2
    feats["night_tx"] = ((feats["hr"] >= 22) | (feats["hr"] <= 6)).astype(float)

    # Velocity
    feat_df = pd.DataFrame(feats)
    feat_df["merchant_tx_count"] = feat_df.groupby("merchant_id").cumcount().values / 1000
    feat_df["city_tx_count"] = feat_df.groupby("city_id").cumcount().values / 1000
    feat_df["night_deviation"] = feat_df["night_tx"].rolling(100, min_periods=1).mean().fillna(0)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = (raw["Is Fraud?"] == "Yes").values.astype(int)
    return feat_df, y, "Altman"


def load_paysim():
    df = pd.read_csv(ROOT / "data" / "paysim_1m.csv")
    feats = {}
    feats["amount"] = df["amount"]
    feats["oldbalanceOrg"] = df["oldbalanceOrg"]
    feats["newbalanceOrig"] = df["newbalanceOrig"]
    feats["oldbalanceDest"] = df["oldbalanceDest"]
    feats["newbalanceDest"] = df["newbalanceDest"]
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    for c in type_dummies.columns:
        feats[c] = type_dummies[c].values

    feat_df = pd.DataFrame(feats)
    feat_df["bal_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    feat_df["bal_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    feat_df["amt_ratio_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    feat_df["orig_wiped"] = (df["newbalanceOrig"] == 0).astype(float)
    feat_df["dest_zero"] = (df["oldbalanceDest"] == 0).astype(float)
    feat_df["flow_asym"] = abs(feat_df["bal_diff_orig"] + feat_df["bal_diff_dest"])
    feat_df["is_cashout"] = (df["type"] == "CASH_OUT").astype(float)
    feat_df["is_transfer"] = (df["type"] == "TRANSFER").astype(float)
    feat_df["balance_wipe_score"] = feat_df["orig_wiped"] * feat_df["is_transfer"]
    feat_df["drain_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return feat_df, df["isFraud"].values.astype(int), "PaySim"


# ── Analysis ──

def get_feature_importance(df, y, name):
    """Train XGB and return sorted feature importances."""
    X = df.values.astype(np.float32)
    X[~np.isfinite(X)] = 0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    n_pos = int(ytr.sum())
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, gamma=2,
                      min_child_weight=5,
                      scale_pos_weight=min(n_pos/max(len(ytr)-n_pos,1)*50, 200),
                      random_state=42, n_jobs=NJ, eval_metric="auc")
    m.fit(Xtr, ytr, verbose=False)
    preds = m.predict_proba(Xte)[:, 1]
    auc_val = roc_auc_score(yte, preds)

    importances = dict(zip(df.columns, m.feature_importances_))
    # Normalize to sum=1
    total = sum(importances.values())
    if total > 0:
        importances = {k: v/total for k, v in importances.items()}

    return importances, auc_val


def concept_importance(importances):
    """Aggregate feature importances by semantic concept."""
    concept_scores = defaultdict(float)
    concept_features = defaultdict(list)
    for feat, imp in importances.items():
        concept = get_concept(feat)
        concept_scores[concept] += imp
        concept_features[concept].append((feat, imp))
    return dict(concept_scores), dict(concept_features)


def main():
    print("=" * 80)
    print("  PATTERN TRANSFER ANALYSIS")
    print("  Which fraud patterns are universal vs domain-specific?")
    print("=" * 80)
    t0 = time.time()

    # Load datasets
    print("\n[1] Loading datasets...")
    ulb_df, ulb_y, _ = load_ulb()
    alt_df, alt_y, _ = load_altman()
    ps_df, ps_y, _ = load_paysim()
    print(f"  ULB: {len(ulb_df):,} rows, {int(ulb_y.sum())} fraud, {len(ulb_df.columns)} features")
    print(f"  Altman: {len(alt_df):,} rows, {int(alt_y.sum())} fraud, {len(alt_df.columns)} features")
    print(f"  PaySim: {len(ps_df):,} rows, {int(ps_y.sum())} fraud, {len(ps_df.columns)} features")

    # Get feature importances
    print("\n[2] Training models and extracting importances...")
    ulb_imp, ulb_auc = get_feature_importance(ulb_df, ulb_y, "ULB")
    alt_imp, alt_auc = get_feature_importance(alt_df, alt_y, "Altman")
    ps_imp, ps_auc = get_feature_importance(ps_df, ps_y, "PaySim")
    print(f"  ULB: AUC={ulb_auc:.4f}")
    print(f"  Altman: AUC={alt_auc:.4f}")
    print(f"  PaySim: AUC={ps_auc:.4f}")

    # Concept-level importance
    ulb_concepts, ulb_concept_feats = concept_importance(ulb_imp)
    alt_concepts, alt_concept_feats = concept_importance(alt_imp)
    ps_concepts, ps_concept_feats = concept_importance(ps_imp)

    # ── Feature-level importance ranking ──
    print("\n" + "=" * 80)
    print("  FEATURE IMPORTANCE RANKINGS (top 15 per domain)")
    print("=" * 80)

    for name, imp in [("ULB", ulb_imp), ("Altman", alt_imp), ("PaySim", ps_imp)]:
        ranked = sorted(imp.items(), key=lambda x: -x[1])
        print(f"\n  {name}:")
        for i, (feat, score) in enumerate(ranked[:15]):
            concept = get_concept(feat)
            print(f"    {i+1:>2}. {score:.4f}  {feat:<25}  [{concept}]")

    # ── Concept-level importance comparison ──
    print("\n" + "=" * 80)
    print("  CONCEPT-LEVEL IMPORTANCE COMPARISON")
    print("=" * 80)

    # Collect all concepts
    all_concepts = sorted(set(list(ulb_concepts.keys()) + list(alt_concepts.keys()) + list(ps_concepts.keys())))

    print(f"\n  {'Concept':<25} {'ULB':>8} {'Altman':>8} {'PaySim':>8}  {'Max Domain':>12}  {'Transfer?':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}  {'-'*12}  {'-'*10}")

    concept_analysis = {}
    for concept in all_concepts:
        u = ulb_concepts.get(concept, 0)
        a = alt_concepts.get(concept, 0)
        p = ps_concepts.get(concept, 0)
        max_val = max(u, a, p)
        domains = []
        if u > 0.01: domains.append("ULB")
        if a > 0.01: domains.append("Altman")
        if p > 0.01: domains.append("PaySim")

        if max_val == u: max_dom = "ULB"
        elif max_val == a: max_dom = "Altman"
        else: max_dom = "PaySim"

        n_domains = len(domains)
        if n_domains >= 3:
            transfer = "UNIVERSAL"
        elif n_domains == 2:
            transfer = "SHARED"
        else:
            transfer = "DOMAIN-SPEC"

        marker = " ***" if n_domains >= 3 else " **" if n_domains == 2 else ""
        print(f"  {concept:<25} {u:>8.4f} {a:>8.4f} {p:>8.4f}  {max_dom:>12}  {transfer:>10}{marker}")

        concept_analysis[concept] = {
            "ulb": round(u, 6), "altman": round(a, 6), "paysim": round(p, 6),
            "max_domain": max_dom, "transfer": transfer, "n_domains": n_domains,
        }

    # ── Universal patterns (present in all 3 domains) ──
    print("\n" + "=" * 80)
    print("  UNIVERSAL FRAUD PATTERNS (high importance in ALL 3 domains)")
    print("=" * 80)

    universal = {k: v for k, v in concept_analysis.items() if v["n_domains"] >= 3}
    for concept, info in sorted(universal.items(), key=lambda x: -sum([x[1]["ulb"], x[1]["altman"], x[1]["paysim"]])):
        total = info["ulb"] + info["altman"] + info["paysim"]
        print(f"\n  {concept} (total importance: {total:.4f})")
        print(f"    ULB: {info['ulb']:.4f}  Altman: {info['altman']:.4f}  PaySim: {info['paysim']:.4f}")

        # Show which specific features contribute
        for name, concept_feats in [("ULB", ulb_concept_feats), ("Altman", alt_concept_feats), ("PaySim", ps_concept_feats)]:
            if concept in concept_feats:
                top = sorted(concept_feats[concept], key=lambda x: -x[1])[:3]
                feats_str = ", ".join(f"{f}({s:.3f})" for f, s in top)
                print(f"    {name}: {feats_str}")

    # ── Domain-specific patterns ──
    print("\n" + "=" * 80)
    print("  DOMAIN-SPECIFIC FRAUD PATTERNS (unique to one domain)")
    print("=" * 80)

    for domain in ["ULB", "Altman", "PaySim"]:
        dk = domain.lower()
        specific = {k: v for k, v in concept_analysis.items()
                   if v["max_domain"] == domain and v["n_domains"] == 1 and v[dk] > 0.05}
        if specific:
            print(f"\n  {domain}-specific patterns:")
            for concept, info in sorted(specific.items(), key=lambda x: -x[1][dk]):
                print(f"    {concept}: {info[dk]:.4f}")
                for name, concept_feats in [("ULB", ulb_concept_feats), ("Altman", alt_concept_feats), ("PaySim", ps_concept_feats)]:
                    if concept in concept_feats:
                        top = sorted(concept_feats[concept], key=lambda x: -x[1])[:2]
                        feats_str = ", ".join(f"{f}({s:.3f})" for f, s in top)
                        print(f"      {name}: {feats_str}")

    # ── Cross-domain feature correlation ──
    print("\n" + "=" * 80)
    print("  CROSS-DOMAIN IMPORTANCE CORRELATION")
    print("=" * 80)

    # Map all features to concepts, then compare concept importance vectors
    concept_vec = {}
    for concept in all_concepts:
        concept_vec[concept] = [
            ulb_concepts.get(concept, 0),
            alt_concepts.get(concept, 0),
            ps_concepts.get(concept, 0),
        ]

    # Spearman rank correlation between pairs
    from scipy.stats import spearmanr
    pairs = [("ULB", "Altman"), ("ULB", "PaySim"), ("Altman", "PaySim")]
    for d1, d2 in pairs:
        idx = {"ULB": 0, "Altman": 1, "PaySim": 2}
        v1 = [concept_vec[c][idx[d1]] for c in all_concepts]
        v2 = [concept_vec[c][idx[d2]] for c in all_concepts]
        corr, pval = spearmanr(v1, v2)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  {d1:>8} <-> {d2:<8}: Spearman rho = {corr:.4f} (p={pval:.4f}) {sig}")

    # ── Transfer potential scoring ──
    print("\n" + "=" * 80)
    print("  TRANSFER POTENTIAL SCORE")
    print("  (How much of domain B's fraud signal is captured by domain A's model)")
    print("=" * 80)

    for src, tgt in [("ULB", "Altman"), ("Altman", "ULB"), ("ULB", "PaySim"),
                      ("PaySim", "ULB"), ("Altman", "PaySim"), ("PaySim", "Altman")]:
        src_key = src.lower()
        tgt_key = tgt.lower()

        # Score = sum of min(src_importance, tgt_importance) / sum of tgt_importance
        overlap = 0
        tgt_total = 0
        for concept in all_concepts:
            src_imp = concept_analysis[concept][src_key.lower()]
            tgt_imp = concept_analysis[concept][tgt_key.lower()]
            overlap += min(src_imp, tgt_imp)
            tgt_total += tgt_imp

        score = overlap / tgt_total if tgt_total > 0 else 0
        print(f"  {src:>8} -> {tgt:<8}: transfer potential = {score:.1%}")

    # Save
    results = {
        "concept_analysis": concept_analysis,
        "feature_importances": {
            "ULB": {k: round(v, 6) for k, v in ulb_imp.items()},
            "Altman": {k: round(v, 6) for k, v in alt_imp.items()},
            "PaySim": {k: round(v, 6) for k, v in ps_imp.items()},
        },
        "domain_aucs": {"ULB": round(ulb_auc, 6), "Altman": round(alt_auc, 6), "PaySim": round(ps_auc, 6)},
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = REPORTS / "pattern_transfer_analysis.json"
    out.write_text(json.dumps(results, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o)))
    print(f"\n  Saved: {out}")
    print(f"  Total: {time.time()-t0:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
