"""Phase 2: distribution, dependency, leakage-attack and distinguishability tests.

Compares the 50M synthetic benchmark against the REAL reference matrix built by
`build_real_reference_features.py` (same 48-feature runtime contract, same
past-only discipline), so differences are attributable to generation rather
than to featurisation.

Covers phase-2 sections 6, 7, 11 and 12.

Usage:
    ./.venv/Scripts/python.exe scripts/compare_real_synthetic.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BENCH = "data/synthetic_50m"
REAL = "misc/reports/real_reference_features.parquet"
CONTRACT = "models/production/manifest.json"
OUT = "misc/reports/phase2_comparison.json"

N_SYNTH = 400_000      # sampled from the head of the benchmark
N_REAL = 400_000
SEED = 20261005

# History-depth-dependent features. The real side is built from a STRIDED
# sample of a user-sorted file, so cumulative counts/velocities on the real
# side are not comparable to the benchmark's full-history values. Reporting a
# head-to-head number for these would be misleading, so they are EXCLUDED from
# the distribution/dependency comparison and reported descriptively only.
HISTORY_FEATURES = [
    "user_tx_count", "card_tx_count", "merch_tx_count",
    "user_merchant_diversity", "user_city_diversity", "user_merch_count",
    "user_avg_amt", "amt_vs_user_avg", "amt_zscore", "high_amt",
    "very_high_amt", "user_fraud_rate", "merch_fraud_rate", "city_fraud_rate",
]


def load_synth(n: int) -> pd.DataFrame:
    files = sorted(str(p) for p in Path(BENCH, "data").rglob("*.parquet"))
    parts, total = [], 0
    for fp in files:
        df = pq.ParquetFile(fp).read().to_pandas()
        parts.append(df)
        total += len(df)
        if total >= n:
            break
    return pd.concat(parts, ignore_index=True).head(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--synth", type=int, default=N_SYNTH)
    ap.add_argument("--real", type=int, default=N_REAL)
    args = ap.parse_args()

    features = json.loads(Path(CONTRACT).read_text(encoding="utf-8"))["features"]
    intrinsic = [f for f in features if f not in HISTORY_FEATURES]
    res: dict = {"compared_features": intrinsic,
                 "excluded_history_features": HISTORY_FEATURES,
                 "exclusion_reason":
                     "real reference is a strided sample of a user-sorted file; "
                     "cumulative history features are not comparable"}

    syn = load_synth(args.synth)
    rea = pd.read_parquet(REAL).head(args.real)
    print(f"synthetic sample {len(syn):,}  real sample {len(rea):,}", flush=True)

    # ---- section 6: distribution -----------------------------------------
    dist = []
    for f in intrinsic:
        s = syn[f].to_numpy(dtype=np.float64)
        r = rea[f].to_numpy(dtype=np.float64)
        row = {"feature": f}
        for tag, x in (("syn", s), ("rea", r)):
            qs = np.percentile(x, [1, 25, 50, 75, 99])
            row[f"{tag}_mean"] = float(np.mean(x))
            row[f"{tag}_std"] = float(np.std(x))
            row[f"{tag}_min"] = float(np.min(x))
            row[f"{tag}_max"] = float(np.max(x))
            for q, v in zip(("p1", "p25", "p50", "p75", "p99"), qs):
                row[f"{tag}_{q}"] = float(v)
            row[f"{tag}_nuniq"] = int(np.unique(x).size)
        # standardised mean difference: comparable across feature scales
        sd = np.sqrt((row["syn_std"] ** 2 + row["rea_std"] ** 2) / 2)
        row["smd"] = float(abs(row["syn_mean"] - row["rea_mean"]) / sd) if sd > 1e-12 else 0.0
        dist.append(row)
    res["distribution"] = dist
    big = sorted(dist, key=lambda r: -r["smd"])[:8]
    print("\nLargest standardised mean differences (synthetic vs real):")
    for r in big:
        print(f"  {r['feature']:26} SMD={r['smd']:.4f} "
              f"mean syn={r['syn_mean']:.3f} real={r['rea_mean']:.3f}")

    # ---- section 7: dependency -------------------------------------------
    cs = syn[intrinsic].corr(method="spearman")
    cr = rea[intrinsic].corr(method="spearman")
    delta = (cs - cr).abs()
    # pandas 3 copy-on-write: to_numpy(copy=True) gives a writable buffer
    arr = delta.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    delta = pd.DataFrame(arr, index=delta.index, columns=delta.columns)
    flat = delta.stack().sort_values(ascending=False)
    top_pairs = []
    seen = set()
    for (a, b), v in flat.items():
        if a > b and (a, b) not in seen:
            seen.add((a, b))
            top_pairs.append({"a": a, "b": b, "abs_delta": float(v),
                              "syn_rho": float(cs.loc[a, b]), "real_rho": float(cr.loc[a, b])})
        if len(top_pairs) >= 15:
            break
    offdiag = delta.values[np.triu_indices(len(intrinsic), 1)]
    res["dependency"] = {
        "method": "spearman",
        "mean_abs_delta": float(np.nanmean(offdiag)),
        "max_abs_delta": float(np.nanmax(offdiag)),
        "p95_abs_delta": float(np.nanpercentile(offdiag, 95)),
        "pairs_above_0.10": int(np.nansum(offdiag > 0.10)),
        "top_degraded_pairs": top_pairs,
    }
    print(f"\nDependency: mean|delta rho|={np.nanmean(offdiag):.4f} "
          f"max={np.nanmax(offdiag):.4f} pairs>0.10={int(np.nansum(offdiag>0.10))}")
    for t in top_pairs[:6]:
        print(f"  {t['a']:24} x {t['b']:22} syn={t['syn_rho']:+.3f} "
              f"real={t['real_rho']:+.3f} d={t['abs_delta']:.3f}")

    # fraud-vs-predictor AUC on each side (section 7 dependency)
    from sklearn.metrics import roc_auc_score

    def auc_table(df):
        y = df["label"].to_numpy()
        out = {}
        for f in features:
            x = df[f].to_numpy(dtype=np.float64)
            try:
                a = roc_auc_score(y, x)
                out[f] = float(max(a, 1 - a))
            except Exception:
                out[f] = None
        return out

    as_, ar_ = auc_table(syn), auc_table(rea)
    fa = sorted(((f, as_[f], ar_[f]) for f in features if as_[f] is not None),
                key=lambda t: -abs(t[1] - t[2]))[:10]
    res["fraud_predictor_auc"] = {
        "synthetic": {f: as_[f] for f in features},
        "real": {f: ar_[f] for f in features},
        "largest_gaps": [{"feature": f, "syn": a, "real": b} for f, a, b in fa],
    }
    print("\nFraud-predictor AUC gaps (|syn-real|):")
    for f, a, b in fa[:6]:
        print(f"  {f:26} syn={a:.4f} real={b:.4f} gap={abs(a-b):.4f}")

    # ---- section 11: leakage attacks -------------------------------------
    y = syn["label"].to_numpy()
    atk = {}
    atk["label_exact_auc"] = float(roc_auc_score(y, y.astype(float)))
    ym = pd.factorize(syn["year_month"].astype(str))[0].astype(float)
    atk["year_month_auc"] = float(max(roc_auc_score(y, ym),
                                      1 - roc_auc_score(y, ym)))
    uid = syn["user_id"].to_numpy().astype(float)
    atk["user_id_auc"] = float(max(roc_auc_score(y, uid), 1 - roc_auc_score(y, uid)))
    rowpos = np.arange(len(syn), dtype=float)
    atk["row_position_auc"] = float(max(roc_auc_score(y, rowpos),
                                       1 - roc_auc_score(y, rowpos)))
    # can split membership be predicted from features alone? (contamination)
    from sklearn.linear_model import LogisticRegression

    sp = (syn["split"] == "test").to_numpy().astype(int)
    if len(np.unique(sp)) > 1:
        X = syn[features].to_numpy(dtype=np.float32)
        lr = LogisticRegression(max_iter=300, n_jobs=-1).fit(X, sp)
        atk["split_predictability_auc"] = float(
            max(roc_auc_score(sp, lr.predict_proba(X)[:, 1]),
                1 - roc_auc_score(sp, lr.predict_proba(X)[:, 1])))
    else:
        atk["split_predictability_auc"] = None
    res["leakage_attacks"] = atk
    print("\nLeakage attacks:")
    for k, v in atk.items():
        print(f"  {k:28} {v:.4f}" if v is not None else f"  {k:28} n/a")

    # ---- section 12: real vs synthetic distinguishability ----------------
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    ns = min(args.synth, 300_000)
    nr = min(args.real, 300_000)
    Xs = syn[features].head(ns).to_numpy(dtype=np.float32)
    Xr = rea[features].head(nr).to_numpy(dtype=np.float32)
    X = np.vstack([Xs, Xr])
    d = np.concatenate([np.zeros(len(Xs), dtype=int), np.ones(len(Xr), dtype=int)])
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
    cv = cross_val_score(clf, X, d, cv=3, scoring="roc_auc", n_jobs=-1)
    aucs = float(cv.mean())
    clf.fit(X, d)
    imp = sorted(zip(features, clf.feature_importances_ if hasattr(clf, "feature_importances_")
                     else np.zeros(len(features))), key=lambda t: -t[1])[:10]
    res["distinguishability"] = {
        "method": "HistGradientBoosting, 3-fold CV ROC-AUC",
        "n_synth": int(len(Xs)), "n_real": int(len(Xr)),
        "cv_auc_mean": aucs, "cv_auc_std": float(cv.std()),
        "top_features": [{"feature": f, "importance": float(v)} for f, v in imp],
        "interpretation": None,
    }
    print(f"\nReal-vs-synthetic distinguishability: CV AUC = {aucs:.4f} "
          f"(+/- {cv.std():.4f})")
    for f, v in imp[:6]:
        print(f"  {f:26} importance={v:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
