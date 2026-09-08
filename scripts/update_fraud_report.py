#!/usr/bin/env python3
"""Analyze chameleon frauds and update the fraud report data for the dashboard."""
import json, sys, yaml
from pathlib import Path
import pandas as pd, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk_engine.fusion import FusionEngine
from src.risk_engine.rules_engine import RulesEngine
from src.privacy_layer.features import ML_FEATURES

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "models" / "artifacts"
RULES_PATH = ROOT / "src" / "risk_engine" / "rules.yaml"


def main():
    # Load data
    df = pd.read_csv(ROOT / "data" / "transactions.csv")
    labels = df["label"].values

    # Load models
    fusion = FusionEngine(ARTIFACTS)
    cfg = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    rules = RulesEngine(cfg["rules"], severity_scale=float(cfg.get("severity_scale", 1.0)))

    # Build feature dicts
    feature_dicts = []
    for i in range(len(df)):
        feat = {k: float(df.iloc[i][k]) for k in ML_FEATURES if k in df.columns}
        feature_dicts.append(feat)

    # Batched ML prediction
    ml_scores = fusion.predict_many(feature_dicts)

    # Rules evaluation (also vectorized where possible)
    rules_scores = np.zeros(len(df))
    for i in range(len(df)):
        rule_result = rules.evaluate(feature_dicts[i])
        rules_scores[i] = rule_result.get("score", 0.0)

    combined_scores = np.maximum(ml_scores, rules_scores)

    threshold = 0.30

    n_fraud = int(labels.sum())
    n_legit = len(labels) - n_fraud
    fraud_idx = np.where(labels == 1)[0]
    legit_idx = np.where(labels == 0)[0]

    # ML-only
    ml_tp = int((ml_scores[fraud_idx] >= threshold).sum())
    ml_fp = int((ml_scores[legit_idx] >= threshold).sum())

    # Combined
    comb_tp = int((combined_scores[fraud_idx] >= threshold).sum())
    comb_fp = int((combined_scores[legit_idx] >= threshold).sum())

    # Chameleons
    chameleon_mask = ml_scores[fraud_idx] < threshold
    chameleon_indices = fraud_idx[chameleon_mask]
    chameleon_caught = int((combined_scores[chameleon_indices] >= threshold).sum())
    chameleon_missed = len(chameleon_indices) - chameleon_caught

    # Chameleon detail
    chameleon_detail = []
    for ci in chameleon_indices:
        row = df.iloc[ci]
        feat = feature_dicts[ci]
        deviations = {k: round(feat[k], 3) for k in ML_FEATURES if abs(feat[k]) > 1.0}
        chameleon_detail.append({
            "idx": int(ci),
            "amount": float(row.get("amount_ratio", 0)),
            "hour": int(row.get("hour_of_day", 0)),
            "ml_score": round(float(ml_scores[ci]), 6),
            "rules_score": round(float(rules_scores[ci]), 6),
            "combined": round(float(combined_scores[ci]), 6),
            "caught": bool(combined_scores[ci] >= threshold),
            "top_features": dict(sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]),
            "archetype": str(row.get("archetype", "")),
        })

    # Top false positives
    fp_mask = combined_scores[legit_idx] >= threshold
    fp_indices = legit_idx[fp_mask]
    fp_sorted = np.argsort(-combined_scores[fp_indices])[:20]
    top_fps = []
    for rank, oi in enumerate(fp_sorted):
        fi = fp_indices[oi]
        top_fps.append({
            "rank": rank + 1,
            "idx": int(fi),
            "amount": float(df.iloc[fi].get("amount_ratio", 0)),
            "ml_score": round(float(ml_scores[fi]), 6),
            "rules_score": round(float(rules_scores[fi]), 6),
            "combined": round(float(combined_scores[fi]), 6),
        })

    success_rate = round(comb_tp / n_fraud * 100, 2) if n_fraud > 0 else 0
    precision = round(comb_tp / (comb_tp + comb_fp) * 100, 2) if (comb_tp + comb_fp) > 0 else 0
    fpr = round(comb_fp / n_legit * 100, 4) if n_legit > 0 else 0

    results = {
        "summary": {
            "total_transactions": len(df),
            "total_fraud": n_fraud,
            "total_legit": n_legit,
            "threshold": threshold,
            "success_rate_pct": success_rate,
            "ml_only_tp": ml_tp,
            "ml_only_fp": ml_fp,
            "combined_tp": comb_tp,
            "combined_fp": comb_fp,
            "combined_fn": n_fraud - comb_tp,
            "precision_pct": precision,
            "fpr_pct": fpr,
            "chameleons_total": len(chameleon_indices),
            "chameleons_caught": chameleon_caught,
            "chameleons_missed": chameleon_missed,
        },
        "chameleons": chameleon_detail,
        "top_false_positives": top_fps,
    }

    out_path = ROOT / "data" / "updated_chameleon_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"=== CHAMELEON ANALYSIS ===")
    print(f"Total fraud: {n_fraud} | Caught (ML only): {ml_tp} | Caught (combined): {comb_tp}")
    print(f"Success rate: {success_rate}% (target: 97.5%)")
    print(f"Precision: {precision}% | FPR: {fpr}%")
    print(f"Chameleons: {len(chameleon_indices)} total | {chameleon_caught} caught by rules | {chameleon_missed} still missed")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
