"""Phase 3 large-scale baseline runner: inference over the validated 50M benchmark.

Runs the CURRENT production/native ensemble exactly as it exists (no changes):
    native 48-vector (contract order) -> float32 -> RobustScaler
    -> {xgb, lgb, cb}.predict_proba -> 0.34/0.33/0.33 weighted mean -> clip[0,1]
    -> decision vs locked_threshold 0.7847116291110687

No calibrator: AltmanNativeEnsembleEngine loads it from
`models/production/artifacts/calibrator.joblib`, which does not exist, so
`self.calibrator is None` in production. This runner reproduces that.

Every stage is timed separately (load / validate / scale / infer / fuse /
threshold) so the report does not quote only the fastest component.

Partitions are processed in CHRONOLOGICAL order (year_month ascending). All
operations are STATELESS across batches: the benchmark's historical features
were already computed by the generator, and this runner never refactorizes
ids or recomputes aggregates, so batching cannot alter semantics.

Usage:
    ./.venv/Scripts/python.exe scripts/large_scale_runner.py --rows 50000000 \
        --out misc/reports/phase3_inference_50m.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

BENCH = "data/synthetic_50m"
MODEL_DIR = Path("models/production/altman_native")
CONTRACT = "models/production/manifest.json"
ENSEMBLE_WEIGHTS = {"xgb": 0.34, "lgb": 0.33, "cb": 0.33}


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def psutil_mem_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return float("nan")


def cpu_model() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def total_ram_gb() -> float:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "[math]::Round((Get-CimInstance Win32_ComputerSystem)"
                              ".TotalPhysicalMemory/1GB,1)"],
                             capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000_000)
    ap.add_argument("--batch", type=int, default=250_000)
    ap.add_argument("--out", default="misc/reports/phase3_inference_50m.json")
    ap.add_argument("--label", default="inference")
    ap.add_argument("--keep-scores", action="store_true")
    ap.add_argument("--bench", default=BENCH,
                    help="benchmark root override; Phase 3 §12 fault injection "
                         "points this at COPIES of partitions, never the original")
    args = ap.parse_args()

    t_start = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    manifest = json.loads(Path(CONTRACT).read_text(encoding="utf-8"))
    features = manifest["features"]
    threshold = float(manifest["locked_threshold"])
    bench_man = json.loads((Path(args.bench) / "manifest.json").read_text(encoding="utf-8"))
    ph = bench_man["partition_hashes"]
    comb = hashlib.sha256()
    for k in sorted(ph):
        comb.update(k.encode())
        comb.update(ph[k].encode())

    # ---- model artifacts (read-only; never written) --------------------
    scaler = joblib.load(MODEL_DIR / "scaler_native.joblib")
    xgb = joblib.load(MODEL_DIR / "xgb_native.joblib")
    lgb = joblib.load(MODEL_DIR / "lgb_native.joblib")
    cb = joblib.load(MODEL_DIR / "cb_native.joblib")
    cal_path = MODEL_DIR.parent / "artifacts" / "calibrator.joblib"
    calibrator = joblib.load(cal_path) if cal_path.exists() else None

    # ---- partitions in chronological order -----------------------------
    files = sorted((str(p) for p in (Path(args.bench) / "data").rglob("*.parquet")),
                   key=lambda s: s.split("year_month=")[-1])
    order = [f.split("year_month=")[-1].split(os.sep)[0] for f in files]

    cap = args.rows
    scores = np.empty(cap, dtype=np.float32)
    member = {k: np.empty(cap, dtype=np.float32) for k in ENSEMBLE_WEIGHTS}
    labels = np.empty(cap, dtype=np.int8)
    splits = np.empty(cap, dtype=np.int8)      # 0=train 1=test

    t = {k: 0.0 for k in ("load", "validate", "scale", "infer", "fuse",
                          "threshold", "store")}
    lat: list[float] = []            # per-batch end-to-end latency
    n = 0
    n_fraud = 0
    n_alert = 0
    tp = fp = fn = tn = 0
    n_nonfinite = 0
    n_missing_col = 0
    partitions_done = 0
    bytes_read = 0

    for fpath in files:
        if n >= cap:
            break
        pf = pq.ParquetFile(fpath)
        bytes_read += os.path.getsize(fpath)
        for batch in pf.iter_batches(batch_size=args.batch,
                                     columns=features + ["label", "split"]):
            if n >= cap:
                break                        # cap reached inside this partition:
                                             # a further batch would be an empty
                                             # frame (RobustScaler rejects 0 rows)
            tb = time.perf_counter()
            df = batch.to_pandas()
            if n + len(df) > cap:
                df = df.iloc[: cap - n]
            t["load"] += time.perf_counter() - tb

            tb = time.perf_counter()
            missing = [c for c in features if c not in df.columns]
            if missing:
                n_missing_col += 1
                raise SystemExit(f"FATAL: partition missing features: {missing}")
            X = df[features].to_numpy(dtype=np.float32)
            bad = int((~np.isfinite(X)).sum())
            n_nonfinite += bad
            if bad:
                raise SystemExit(f"FATAL: {bad} non-finite feature values")
            lab = df["label"].to_numpy().astype(np.int8)
            sp = df["split"].astype(str).to_numpy()
            t["validate"] += time.perf_counter() - tb

            tb = time.perf_counter()
            Xs = scaler.transform(X)
            t["scale"] += time.perf_counter() - tb

            tb = time.perf_counter()
            p_x = xgb.predict_proba(Xs)[:, 1]
            p_l = lgb.predict_proba(Xs)[:, 1]
            p_c = cb.predict_proba(Xs)[:, 1]
            t["infer"] += time.perf_counter() - tb

            tb = time.perf_counter()
            raw = (ENSEMBLE_WEIGHTS["xgb"] * p_x + ENSEMBLE_WEIGHTS["lgb"] * p_l
                   + ENSEMBLE_WEIGHTS["cb"] * p_c)
            prob = np.clip(raw, 0.0, 1.0)
            t["fuse"] += time.perf_counter() - tb

            tb = time.perf_counter()
            pred = (prob >= threshold).astype(np.int8)
            t["threshold"] += time.perf_counter() - tb

            tb = time.perf_counter()
            k = len(df)
            scores[n:n + k] = prob
            member["xgb"][n:n + k] = p_x
            member["lgb"][n:n + k] = p_l
            member["cb"][n:n + k] = p_c
            labels[n:n + k] = lab
            splits[n:n + k] = np.where(sp == "test", 1, 0)
            n += k
            t["store"] += time.perf_counter() - tb

            n_fraud += int(lab.sum())
            n_alert += int(pred.sum())
            tp += int(((pred == 1) & (lab == 1)).sum())
            fp += int(((pred == 1) & (lab == 0)).sum())
            fn += int(((pred == 0) & (lab == 1)).sum())
            tn += int(((pred == 0) & (lab == 0)).sum())
            lat.append(time.perf_counter() - tb)
        partitions_done += 1

    wall = time.time() - t_start
    s = scores[:n]
    y = labels[:n]
    lat_ms = np.array(lat) * 1000.0

    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)
    res = {
        "label": args.label,
        "evidence_class": "SELF-TESTED — SYNTHETIC 50M — SCALE EXPERIMENT",
        "rows_processed": n,
        "partitions_processed": partitions_done,
        "partition_order": "chronological (year_month ascending)",
        "batch_size": args.batch,
        "workers": 1,
        "fraud_rows": n_fraud,
        "fraud_prevalence": n_fraud / n if n else None,
        "wall_seconds": wall,
        "rows_per_second": n / wall if wall else None,
        "stage_seconds": t,
        "stage_fraction": {k: (v / wall if wall else None) for k, v in t.items()},
        "bytes_read": bytes_read,
        "latency_ms_per_batch": {
            "mean": float(lat_ms.mean()) if lat_ms.size else None,
            "p50": float(np.percentile(lat_ms, 50)) if lat_ms.size else None,
            "p95": float(np.percentile(lat_ms, 95)) if lat_ms.size else None,
            "p99": float(np.percentile(lat_ms, 99)) if lat_ms.size else None,
        },
        "peak_rss_MB": psutil_mem_mb(),
        "threshold": threshold,
        "confusion_at_threshold": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
        "alert_rate": n_alert / n if n else None,
        "roc_auc": float(roc_auc_score(y, s)) if n else None,
        "pr_auc": float(average_precision_score(y, s)) if n else None,
        "brier": float(brier_score_loss(y, s)) if n else None,
        "member_roc_auc": {k: float(roc_auc_score(y, v[:n])) for k, v in member.items()},
        "member_pr_auc": {k: float(average_precision_score(y, v[:n]))
                          for k, v in member.items()},
        "member_precision": {},
        "nonfinite_values": n_nonfinite,
        "missing_feature_partitions": n_missing_col,
        "calibrator_loaded": calibrator is not None,
        "experiment_manifest": {
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
            "benchmark_id": "synthetic_50m",
            "benchmark_combined_sha256": comb.hexdigest(),
            "benchmark_schema_hash": bench_man["schema_hash"],
            "generator_version": bench_man["generator_version"],
            "benchmark_seed": bench_man["seed"],
            "benchmark_source_sha256": bench_man["source_dataset_sha256"],
            "model_version": manifest["model_version"],
            "model_type": manifest["model_type"],
            "ensemble_weights": ENSEMBLE_WEIGHTS,
            "n_features": manifest["n_features"],
            "feature_schema_version": manifest["feature_schema_version"],
            "production_manifest_sha256": sha256_file(CONTRACT),
            "feature_list_sha256": sha256_file(MODEL_DIR / "feature_list.json"),
            "scaler_sha256": sha256_file(MODEL_DIR / "scaler_native.joblib"),
            "xgb_sha256": sha256_file(MODEL_DIR / "xgb_native.joblib"),
            "lgb_sha256": sha256_file(MODEL_DIR / "lgb_native.joblib"),
            "cb_sha256": sha256_file(MODEL_DIR / "cb_native.joblib"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu": cpu_model(),
            "ram_gb": total_ram_gb(),
            "batch_size": args.batch,
            "workers": 1,
            "started_utc": started_utc,
            "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    # per-member precision at the same locked threshold
    for k, v in member.items():
        pv = (v[:n] >= threshold).astype(np.int8)
        tpk = int(((pv == 1) & (y == 1)).sum())
        fpk = int(((pv == 1) & (y == 0)).sum())
        res["member_precision"][k] = tpk / (tpk + fpk) if (tpk + fpk) else None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"rows={n:,} fraud={n_fraud:,} wall={wall:.1f}s "
          f"throughput={n/wall:,.0f} rows/s")
    print(f"ROC-AUC={res['roc_auc']:.6f} PR-AUC={res['pr_auc']:.6f} "
          f"Brier={res['brier']:.6f}")
    print(f"member ROC-AUC: {res['member_roc_auc']}")
    print(f"latency ms p50/p95/p99: {res['latency_ms_per_batch']['p50']:.0f}/"
          f"{res['latency_ms_per_batch']['p95']:.0f}/"
          f"{res['latency_ms_per_batch']['p99']:.0f}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
