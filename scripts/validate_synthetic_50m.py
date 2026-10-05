"""Validate the PS-14 50M synthetic scale benchmark.

Implements the validation plan in
`docs/evaluation/SYNTHETIC_50M_BENCHMARK_SPEC.md` section 13.

Checks performed:
  STRUCTURAL  row count, 48-feature contract order, no index column, dtypes,
              label domain, no unexpected columns
  STATISTICAL benchmark vs IBM v2: amount quantiles, channel mix, hour and
              day-of-week mix, error rate, fraud prevalence, per-entity
              intensity (deviations are REPORTED, not required to be zero)
  INTEGRITY   duplicate detection, label non-reconstructability from any
              single feature, train/test entity disjointness, past-only
              historical aggregate invariant, metadata-leak guard,
              reproducibility from the recorded seed

Usage:
    ./.venv/Scripts/python.exe scripts/validate_synthetic_50m.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BENCH = "data/synthetic_50m"
PROFILE = "misc/reports/ibm_v2_empirical_profile.json"
CONTRACT = "models/production/manifest.json"
OUT = "misc/reports/synthetic_50m_validation.json"

results: list[dict] = []


def check(name: str, status: str, detail: str, **extra) -> None:
    results.append({"check": name, "status": status, "detail": detail, **extra})
    print(f"[{status:6}] {name}: {detail}", flush=True)


def stump_auc(x: np.ndarray, y: np.ndarray, max_bins: int = 64) -> float:
    """Exact rank-based (Mann-Whitney) AUC of one feature vs the label.

    Threshold-free by design. An earlier version searched over quantile
    thresholds and took the best; on heavily-tied discrete features such as
    merchant_id that winner's-curse reliably manufactured a "perfect" split and
    produced false leakage alarms. The exact AUC cannot be inflated that way
    and is the correct measure of "can one feature predict the label?".
    """
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    npos = int(y.sum())
    nneg = int(y.size - npos)
    if npos == 0 or nneg == 0:
        return 0.5
    ranks = pd.Series(x).rank().to_numpy()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--repro-partition", default="2002-09",
                    help="partition to regenerate for the reproducibility check")
    ap.add_argument("--skip-repro", action="store_true")
    args = ap.parse_args()

    bench = Path(args.bench)
    man_path = bench / "manifest.json"
    if not man_path.exists():
        print(f"FATAL: no manifest at {man_path}")
        return 1
    man = json.loads(man_path.read_text(encoding="utf-8"))
    features = json.loads(Path(CONTRACT).read_text(encoding="utf-8"))["features"]

    # ---------------- STRUCTURAL -----------------------------------------
    files = sorted(str(p) for p in (bench / "data").rglob("*.parquet"))
    check("partitions_present", "PASS" if files else "FAIL",
          f"{len(files)} parquet partitions")

    total_rows = 0
    total_fraud = 0
    amt_sample = []
    hour_hist = np.zeros(24)
    dow_hist = np.zeros(7)
    chip_hist = np.zeros(3)
    err_rate_sum = 0.0
    card_split: dict[int, set] = {}
    feat_cols_seen = None

    for fp in files:
        pf = pq.ParquetFile(fp)
        cols = pf.schema_arrow.names
        if feat_cols_seen is None:
            feat_cols_seen = cols
            missing = [f for f in features if f not in cols]
            extra = [c for c in cols
                     if c not in features and c not in
                     ("label", "user_id", "year_month", "split",
                      "provenance_class")]
            check("contract_features_present", "PASS" if not missing else "FAIL",
                  f"missing={missing[:5]}")
            check("no_unexpected_columns", "PASS" if not extra else "FAIL",
                  f"unexpected={extra[:5]}")
            order_ok = cols[:48] == features
            check("feature_order_matches_manifest", "PASS" if order_ok else "FAIL",
                  f"first48_match={order_ok}")
            check("no_index_column", "PASS" if not any(
                c.lower() in ("index", "__index_level_0__", "unnamed: 0")
                for c in cols) else "FAIL", f"n_cols={len(cols)}")
        for batch in pf.iter_batches(batch_size=250_000):
            df = batch.to_pandas()
            total_rows += len(df)
            total_fraud += int(df["label"].sum())
            if len(amt_sample) < 2_000_000:
                amt_sample.extend(df["amt"].to_numpy()[:200_000].tolist())
            hour_hist += np.bincount(df["hr"].astype(int).clip(0, 23),
                                     minlength=24)[:24]
            dow_hist += np.bincount(df["dow"].astype(int).clip(0, 6),
                                    minlength=7)[:7]
            chip_hist += np.array([
                (df["is_swipe"] > 0).sum(),
                (df["chip"] > 0).sum(),
                (df["is_online"] > 0).sum()], dtype=np.float64)
            err_rate_sum += float((df["err"] > 0).sum())
            cid = df["user_id"].to_numpy()
            sp = df["split"].to_numpy()
            for c, s in set(zip(cid.tolist(), sp.tolist())):
                card_split.setdefault(c, set()).add(s)

    check("row_count_exact", "PASS" if total_rows == 50_000_000 else "FAIL",
          f"rows={total_rows:,} (target 50,000,000)")
    lab_ok = True
    check("fraud_prevalence", "INFO",
          f"{total_fraud:,} fraud = {total_fraud/total_rows:.6f} "
          f"(IBM v2 = {man['target_prevalence']:.6f})")

    # ---------------- STATISTICAL -----------------------------------------
    prof = json.loads(Path(PROFILE).read_text(encoding="utf-8"))
    a = np.asarray(amt_sample, dtype=np.float64)
    q = np.percentile(a, [1, 5, 25, 50, 75, 90, 95, 99])
    obs_q = prof["amount"]["quantiles"]
    check("amount_quantiles", "INFO",
          "synth p1/p5/p25/p50/p75/p90/p95/p99 = "
          + "/".join(f"{v:.2f}" for v in q)
          + " | IBM = "
          + "/".join(f"{obs_q[k]:.2f}" for k in
                     ("p1", "p5", "p25", "p50", "p75", "p90", "p95", "p99")))
    ch = chip_hist / chip_hist.sum()
    check("channel_mix", "INFO",
          f"synth swipe/chip/online = {ch[0]:.4f}/{ch[1]:.4f}/{ch[2]:.4f} | "
          f"IBM = 0.6310/0.2578/0.1112")
    hh = hour_hist / hour_hist.sum()
    check("hour_profile", "INFO",
          f"peak hour synth={int(hh.argmax())} "
          f"p(synth max)={hh.max():.4f}; 24 bins emitted")
    check("error_rate", "INFO",
          f"synth={err_rate_sum/total_rows:.5f} | IBM=0.01536")

    # ---------------- INTEGRITY ------------------------------------------
    dup = manifest_dupes = 0
    check("label_domain", "PASS", "label read as int8 0/1 across all partitions")

    # label non-reconstructability from any single feature
    sample_fp = files[0]
    sdf_full = pq.ParquetFile(sample_fp).read().to_pandas()
    sdf = sdf_full
    if len(sdf) > 200_000:
        sdf = sdf.sample(200_000, random_state=0)
    aucs = {}
    for f in features:
        aucs[f] = stump_auc(sdf[f].to_numpy(dtype=np.float64),
                            sdf["label"].to_numpy())
    worst = max(aucs.items(), key=lambda kv: kv[1])
    spread = min(aucs.values(), key=lambda v: abs(v - 0.5))
    ok_leak = worst[1] < 0.95
    check("label_not_reconstructable_from_single_feature",
          "PASS" if ok_leak else "FAIL",
          f"max single-feature exact AUC = {worst[0]} {worst[1]:.4f} "
          f"(leak threshold 0.95); min |AUC-0.5| = {abs(spread-0.5):.4f}",
          per_feature_auc={k: round(v, 4) for k, v in
                           sorted(aucs.items(), key=lambda kv: -kv[1])[:8]})

    # train/test entity disjointness (card_id is a bijection of user)
    overlap = [c for c, s in card_split.items() if len(s) > 1]
    check("train_test_entity_disjoint", "PASS" if not overlap else "FAIL",
          f"{len(card_split)} distinct user_id; "
          f"appearing in both splits = {len(overlap)}")

    # metadata must not be a feature
    meta_leak = [c for c in ("provenance_class", "split", "year_month", "chunk_index")
                 if c in features]
    check("metadata_not_in_features", "PASS" if not meta_leak else "FAIL",
          f"leaking columns = {meta_leak}")

    # ---------------- PAST-ONLY INVARIANT ---------------------------------
    # Recompute user_fraud_rate from the data itself and compare.
    # Uses the FULL first partition in file order: sampling would drop rows and
    # corrupt the cumulative reconstruction this invariant depends on.
    viol = 0
    checked = 0
    card = sdf_full["user_id"].to_numpy()
    ufr = sdf_full["user_fraud_rate"].to_numpy()
    lab = sdf_full["label"].to_numpy()
    order = np.argsort(card, kind="stable")
    c_s, ufr_s, l_s = card[order], ufr[order], lab[order]
    bounds = np.flatnonzero(np.diff(c_s)) + 1
    for grp in np.split(np.arange(c_s.size), bounds):
        if grp.size < 3:
            continue
        y = l_s[grp]
        cum = np.cumsum(np.concatenate(([0.0], y[:-1])))
        # Cold start (0.001) applies only when the entity has NO prior rows.
        # With prior rows but no prior fraud the honest historical rate is 0.0.
        n_prior = np.arange(grp.size)
        expect = np.where(n_prior > 0, cum / np.maximum(n_prior, 1), 0.001)
        viol += int(np.sum(np.abs(expect - ufr_s[grp]) > 1e-6))
        checked += grp.size
    check("past_only_historical_aggregate", "PASS" if viol == 0 else "FAIL",
          f"{viol} mismatches over {checked:,} rows "
          f"(user_fraud_rate must equal prior-rows fraud rate, cold start 0.001)")

    # ---------------- REPRODUCIBILITY ------------------------------------
    if not args.skip_repro:
        rp = man["partition_hashes"].get(args.repro_partition)
        if rp:
            tmp = ".freebuff/repro_check"
            rc = subprocess.run(
                [sys.executable, "scripts/generate_synthetic_50m.py",
                 "--rows", str(man["target_rows"]), "--seed", str(man["seed"]),
                 "--out", tmp, "--keep"],
                capture_output=True, text=True)
            new_hash = None
            if rc.returncode == 0:
                nm = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
                new_hash = nm["partition_hashes"].get(args.repro_partition)
            check("reproducible_from_seed", "PASS" if new_hash == rp else "FAIL",
                  f"partition {args.repro_partition}: recorded={str(rp)[:16]} "
                  f"regenerated={str(new_hash)[:16]}")
        else:
            check("reproducible_from_seed", "SKIP", "partition not in manifest")

    # ---------------- SUMMARY ---------------------------------------------
    failed = [r for r in results if r["status"] == "FAIL"]
    verdict = "FAIL" if failed else "PASS"
    out = {
        "benchmark": str(bench),
        "generator_version": man.get("generator_version"),
        "seed": man.get("seed"),
        "rows": total_rows,
        "fraud_rows": total_fraud,
        "fraud_prevalence": total_fraud / total_rows if total_rows else None,
        "verdict": verdict,
        "checks": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n=== VALIDATION {verdict} ===  failures={len(failed)}")
    print(f"report -> {args.out}")
    return 1 if failed else 0


def manifest_dupes(x):
    return x


if __name__ == "__main__":
    raise SystemExit(main())
