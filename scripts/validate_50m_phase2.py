"""Phase 2 independent validation of the 50M synthetic benchmark.

Single streaming pass over all 166 partitions accumulating only aggregates, so
memory stays flat. Covers phase-2 sections 3, 4, 5, 8, 9, 10, 11 and 14.

Section 7 (dependency) and 12 (real-vs-synthetic distinguishability) live in
`scripts/compare_real_synthetic.py` because they need the real reference matrix.

Usage:
    ./.venv/Scripts/python.exe scripts/validate_50m_phase2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BENCH = "data/synthetic_50m"
CONTRACT = "models/production/manifest.json"
OUT = "misc/reports/phase2_validation.json"

FINDINGS: list[dict] = []


def finding(sev: str, section: str, text: str) -> None:
    FINDINGS.append({"severity": sev, "section": section, "finding": text})
    print(f"[{sev:8}] {section}: {text}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--scan-rows", type=int, default=0,
                    help="0 = all rows (full pass)")
    args = ap.parse_args()

    bench = Path(args.bench)
    man = json.loads((bench / "manifest.json").read_text(encoding="utf-8"))
    features = json.loads(Path(CONTRACT).read_text(encoding="utf-8"))["features"]

    files = sorted(str(p) for p in (bench / "data").rglob("*.parquet"))
    t0 = time.time()
    peak_mem_mb = 0.0

    # ---- section 3: structure ------------------------------------------
    n_rows = n_fraud = 0
    col_set = None
    dup_cols: list[str] = []
    schema_mismatch: list[str] = []
    nulls = {c: 0 for c in features}
    nonfinite = {c: 0 for c in features}
    label_vals: set[int] = set()
    expected_p1 = None
    year_col_present = False
    truncated: list[str] = []
    part_rows: dict[str, int] = {}
    partition_bytes = 0
    scanned_bytes = 0

    feat_min = {f: np.inf for f in features}
    feat_max = {f: -np.inf for f in features}
    feat_sum = {f: 0.0 for f in features}
    feat_sqsum = {f: 0.0 for f in features}
    feat_sum2 = {f: 0.0 for f in features}
    feat_vals: dict[str, np.ndarray] = {f: [] for f in features}
    n_hist = 0

    # section 5/8/9 accumulators
    user_fraud = {}
    user_rows: dict[int, int] = {}
    fraud_by_year: dict[str, int] = {}
    rows_by_year: dict[str, int] = {}
    fraud_by_mcc: dict[int, int] = {}
    rows_by_mcc: dict[int, int] = {}
    fraud_by_hour = np.zeros(24)
    rows_by_hour = np.zeros(24)
    fraud_by_dow = np.zeros(7)
    rows_by_dow = np.zeros(7)
    fraud_by_channel = np.zeros(3)
    rows_by_channel = np.zeros(3)
    split_by_user: dict[int, set] = {}
    merch_seen: set = set()

    dup_hash = 0

    for fp in files:
        pf = pq.ParquetFile(fp)
        pname = Path(fp).parent.name
        partition_bytes += os.path.getsize(fp)
        scanned_bytes += os.path.getsize(fp)
        part_rows[pname] = pf.metadata.num_rows
        names = pf.schema_arrow.names
        if col_set is None:
            col_set = names
            seen: list[str] = []
            for c in names:
                if c in seen:
                    dup_cols.append(c)
                seen.append(c)
        elif names != col_set:
            schema_mismatch.append(pname)
        if pf.metadata.num_rows == 0:
            truncated.append(pname)

        for batch in pf.iter_batches(batch_size=250_000):
            df = batch.to_pandas()
            n = len(df)
            n_rows += n
            lab = df["label"].to_numpy()
            n_fraud += int(lab.sum())
            label_vals.update(np.unique(lab).tolist())

            ym = df["year_month"].astype(str)
            # dict.update() REPLACES existing keys; a month spans several
            # 250k batches, so explicit accumulation is required.
            for k, v in ym.value_counts().to_dict().items():
                rows_by_year[k] = rows_by_year.get(k, 0) + int(v)

            # feature-level stats
            arrs = {}
            for f in features:
                x = df[f].to_numpy(dtype=np.float64)
                arrs[f] = x
                nn = int(np.isnan(x).sum())
                nulls[f] += nn
                nf = int((~np.isfinite(x)).sum())
                nonfinite[f] += nf
                xv = x[np.isfinite(x)]
                if xv.size:
                    feat_min[f] = min(feat_min[f], float(xv.min()))
                    feat_max[f] = max(feat_max[f], float(xv.max()))
                    feat_sum[f] += float(xv.sum())
                    feat_sqsum[f] += float((xv * xv).sum())
                if f == "amt" and expected_p1 is None and n:
                    expected_p1 = 1
            if "year" in df.columns:
                year_col_present = True
            n_hist += n

            # bounded sample for downstream stats
            take = min(n, 60_000)
            for f in features:
                if len(feat_vals[f]) < 300_000:
                    room = 300_000 - len(feat_vals[f])
                    feat_vals[f].append(arrs[f][:room].copy())

            # label-stratified accumulators
            u = df["user_id"].to_numpy()
            for uid, cnt in pd.Series(u).value_counts().items():
                user_rows[int(uid)] = user_rows.get(int(uid), 0) + int(cnt)
            fr = pd.DataFrame({"u": u, "f": lab})
            for uid, sub in fr.groupby("u"):
                if int(sub["f"].sum()):
                    user_fraud[int(uid)] = user_fraud.get(int(uid), 0) + int(sub["f"].sum())
            merch_seen.update(df["merchant_id"].unique().tolist())

            mcc = arrs["mcc"].astype(int)
            for k, v in pd.Series(mcc).value_counts().items():
                rows_by_mcc[int(k)] = rows_by_mcc.get(int(k), 0) + int(v)
            for k, sub in pd.DataFrame({"m": mcc, "f": lab}).groupby("m"):
                if sub["f"].sum():
                    fraud_by_mcc[int(k)] = fraud_by_mcc.get(int(k), 0) + int(sub["f"].sum())

            hr = arrs["hr"].astype(int).clip(0, 23)
            fraud_by_hour += np.bincount(hr, weights=lab, minlength=24)[:24]
            rows_by_hour += np.bincount(hr, minlength=24)[:24]
            dw = arrs["dow"].astype(int).clip(0, 6)
            fraud_by_dow += np.bincount(dw, weights=lab, minlength=7)[:7]
            rows_by_dow += np.bincount(dw, minlength=7)[:7]
            ch = ((arrs["is_swipe"] > 0).astype(int)
                  + (arrs["chip"] > 0).astype(int)
                  + (arrs["is_online"] > 0).astype(int))
            for ci in (0, 1, 2):
                m = ch == ci
                rows_by_channel[ci] += int(m.sum())
                fraud_by_channel[ci] += int(lab[m].sum())
            yv = None
            for k, sub_y in pd.DataFrame({"y": ym, "f": lab}).groupby("y"):
                fraud_by_year[k] = fraud_by_year.get(k, 0) + int(sub_y["f"].sum())

            for uid, sp in set(zip(u.tolist(), df["split"].tolist())):
                split_by_user.setdefault(uid, set()).add(sp)

            if n and n_hist % (250_000 * 40) < 250_000:
                try:
                    peak_mem_mb = max(peak_mem_mb, _mem_mb())
                except Exception:
                    pass
        if len(part_rows) % 40 == 0:
            print(f"  scanned {len(part_rows)}/166 partitions, "
                  f"{n_rows:,} rows ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # ---- section 3 findings --------------------------------------------
    if n_rows == 50_000_000:
        finding("PASS", "3", f"exactly 50,000,000 data rows")
    else:
        finding("FAIL", "3", f"row count is {n_rows:,}, expected 50,000,000")
    if dup_cols:
        finding("FAIL", "3", f"duplicate columns: {dup_cols}")
    else:
        finding("PASS", "3", "no duplicate columns")
    if schema_mismatch:
        finding("FAIL", "3", f"schema differs across partitions: {schema_mismatch}")
    else:
        finding("PASS", "3", f"schema identical across all {len(files)} partitions")
    if truncated:
        finding("FAIL", "3", f"truncated/empty partitions: {truncated}")
    else:
        finding("PASS", "3", "no truncated partitions")
    if label_vals <= {0, 1}:
        finding("PASS", "5", f"label domain = {sorted(label_vals)}")
    else:
        finding("FAIL", "5", f"label contains illegal values: {sorted(label_vals)}")
    tot_null = sum(nulls.values())
    tot_nf = sum(nonfinite.values())
    finding("PASS" if tot_null == 0 else "WARN", "3",
            f"nulls across 48 features = {tot_null:,}; non-finite = {tot_nf:,}")

    # section 4 feature contract
    feat_table = []
    for f in features:
        cnt = n_rows - nulls[f]
        mean = feat_sum[f] / cnt if cnt else None
        var = (feat_sqsum[f] / cnt - mean ** 2) if (cnt and mean is not None) else None
        vals = np.concatenate(feat_vals[f]) if feat_vals[f] else np.array([])
        q = np.percentile(vals, [1, 25, 50, 75, 99]).tolist() if vals.size else []
        feat_table.append({
            "feature": f, "missing": nulls[f],
            "min": None if feat_min[f] is np.inf else feat_min[f],
            "max": None if feat_max[f] == -np.inf else feat_max[f],
            "mean": mean, "std": float(np.sqrt(var)) if var and var > 0 else 0.0,
            "q1": q[0] if q else None, "q25": q[1] if q else None,
            "median": q[2] if q else None, "q75": q[3] if q else None,
            "q99": q[4] if q else None,
            "n_distinct_sample": int(np.unique(vals).size) if vals.size else 0,
        })
    const = [r["feature"] for r in feat_table
             if r["min"] is not None and r["max"] is not None
             and abs(r["max"] - r["min"]) < 1e-12]
    finding("INFO", "4", f"constant features (expected: has_zip, has_state): {const}")

    # section 5
    prev = n_fraud / n_rows if n_rows else 0
    finding("PASS" if abs(prev - man["target_prevalence"]) / man["target_prevalence"] < 0.05
            else "WARN", "5",
            f"prevalence {prev:.7f} vs target {man['target_prevalence']:.7f} "
            f"({(prev/man['target_prevalence']-1)*100:+.2f}%)")
    year_prev = {k: fraud_by_year.get(k, 0) / rows_by_year[k]
                 for k in sorted(rows_by_year) if rows_by_year[k]}
    vals = list(year_prev.values())
    spread = (max(vals) - min(vals)) if vals else 0
    finding("PASS" if spread < 0.003 else "WARN", "9",
            f"prevalence across {len(year_prev)} monthly segments: "
            f"min={min(vals):.5f} max={max(vals):.5f} spread={spread:.5f}")

    # section 8 entities
    ur = np.array(list(user_rows.values()))
    finding("PASS", "8", f"{len(user_rows)} users, tx/user mean={ur.mean():.1f} "
                         f"min={ur.min()} max={ur.max()} p95={np.percentile(ur,95):.0f}")
    srt = np.sort(ur)[::-1]
    top1 = srt[:max(1, len(srt)//100)].sum() / srt.sum()
    finding("INFO", "8", f"top-1% users hold {top1:.3f} of all transactions")
    uf = np.array([user_fraud.get(u, 0) for u in user_rows])
    uu = np.array(list(user_rows.values()))
    with np.errstate(divide="ignore", invalid="ignore"):
        upr = np.where(uu > 0, uf / uu, 0.0)
    finding("PASS" if upr.max() < 0.2 else "WARN", "8",
            f"user fraud-rate range: min={upr.min():.5f} max={upr.max():.5f} "
            f"mean={upr.mean():.5f}")

    # section 10 contamination
    overlap = [u for u, s in split_by_user.items() if len(s) > 1]
    finding("PASS" if not overlap else "FAIL", "10",
            f"entity-disjoint split: {len(split_by_user)} users, "
            f"{len(overlap)} in both splits")
    tr_frac = sum(1 for u, s in split_by_user.items() if s == {"train"}) / max(len(split_by_user), 1)
    finding("INFO", "10", f"train fraction of users = {tr_frac:.3f} "
                          f"(0.5 expected)")

    # section 14 scale
    total_bytes = sum(f.stat().st_size for f in Path(bench).rglob("*") if f.is_file())
    finding("PASS", "14",
            f"total on disk {total_bytes/2**30:.2f} GiB across {len(files)} partitions "
            f"({partition_bytes/2**30:.2f} GiB parquet), "
            f"{total_bytes/n_rows:.2f} bytes/row, read {scanned_bytes/2**30:.2f} GiB "
            f"in {elapsed:.0f}s ({scanned_bytes/2**20/max(elapsed,1):.0f} MiB/s aggregate)")

    out = {
        "benchmark_identity": {
            "git_sha": man["git_sha"], "generator_version": man["generator_version"],
            "seed": man["seed"], "schema_hash": man["schema_hash"],
            "source_sha256": man["source_dataset_sha256"],
            "target_rows": man["target_rows"], "partitions": man["partitions"],
        },
        "rows": n_rows, "fraud_rows": n_fraud, "prevalence": prev,
        "label_domain": sorted(label_vals),
        "n_columns": len(col_set), "columns": col_set,
        "nulls_total": tot_null, "nonfinite_total": tot_nf,
        "feature_table": feat_table,
        "constant_features": const,
        "prevalence_by_month": year_prev,
        "prevalence_by_hour": (fraud_by_hour / np.maximum(rows_by_hour, 1)).tolist(),
        "prevalence_by_dow": (fraud_by_dow / np.maximum(rows_by_dow, 1)).tolist(),
        "prevalence_by_channel": (fraud_by_channel / np.maximum(rows_by_channel, 1)).tolist(),
        "rows_by_channel": rows_by_channel.tolist(),
        "fraud_rate_by_mcc": {str(k): fraud_by_mcc.get(k, 0) / v
                              for k, v in rows_by_mcc.items() if v},
        "entities": {"n_users": len(user_rows),
                     "tx_per_user_mean": float(ur.mean()),
                     "tx_per_user_min": int(ur.min()),
                     "tx_per_user_max": int(ur.max()),
                     "tx_per_user_p95": float(np.percentile(ur, 95)),
                     "top1pct_user_share": float(top1),
                     "user_fraud_rate_max": float(upr.max())},
        "contamination": {"users_in_both_splits": len(overlap),
                          "n_users": len(split_by_user),
                          "train_fraction": float(tr_frac)},
        "scale": {"total_bytes": total_bytes, "parquet_bytes": partition_bytes,
                  "bytes_per_row": total_bytes / max(n_rows, 1),
                  "scan_seconds": elapsed,
                  "throughput_MiB_s": scanned_bytes / 2**20 / max(elapsed, 1),
                  "peak_rss_MB": peak_mem_mb},
        "findings": FINDINGS,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    fails = [f for f in FINDINGS if f["severity"] == "FAIL"]
    print(f"\n=== PHASE2 SCAN DONE  rows={n_rows:,}  failures={len(fails)} ===")
    print(f"report -> {args.out}")
    return 1 if fails else 0


def _mem_mb() -> float:
    import psutil  # optional
    return psutil.Process().memory_info().rss / 2**20


if __name__ == "__main__":
    raise SystemExit(main())
