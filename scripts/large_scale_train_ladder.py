"""Phase 3 §7 — TRAINING SCALABILITY LADDER on the validated 50M benchmark.

Retrains the CURRENT production/native methodology at FIXED canonical
hyperparameters on progressively larger prefixes of the benchmark, in
chronological (year_month ascending) order.

Hyperparameters are copied verbatim from the canonical native retrain
(`.freebuff/retrain_native_tuned.py`), which produced the production members:

    xgb : n_estimators=400 max_depth=7 lr=0.05 subsample=0.8
          colsample_bytree=0.7 min_child_weight=5 scale_pos_weight=SPW
          random_state=42 n_jobs=4 eval_metric=auc
    lgb : n_estimators=400 max_depth=7 lr=0.05 subsample=0.8
          colsample_bytree=0.7 min_child_weight=5 scale_pos_weight=SPW
          random_state=42 n_jobs=4
    cb  : iterations=400 depth=7 lr=0.05 l2_leaf_reg=3 random_seed=42
          auto_class_weights=Balanced
    scaler: RobustScaler fit on the training rows only

NO hyperparameter search is performed (§7 forbids unlimited search). SPW is the
single value the canonical script's own default resolves to on this benchmark:
min((1 - prevalence) / prevalence, 20.0) = 20.0.

Each rung records: wall seconds, effective training rows/s, peak RSS, artifact
size on disk, boosting rounds completed, convergence status. Rungs are appended
to a JSONL immediately so a truncated ladder still yields measured evidence.

NOTHING IS EVER WRITTEN TO models/production. Models land in a scratch dir.

Usage:
    ./.venv/Scripts/python.exe scripts/large_scale_train_ladder.py \
        --ladder 1000000,5000000,10000000,25000000 --out misc/reports/phase3_train_ladder.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

BENCH = Path("data/synthetic_50m")
MODEL_DIR = Path("models/production/altman_native")
CONTRACT = Path("models/production/manifest.json")
SCRATCH = Path("misc/reports/phase3_train_ladder_artifacts")
SPW = 20.0
SEED = 42


def sha256_file(p) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def peak_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return float("nan")


def cpu_model() -> str:
    try:
        import subprocess

        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def total_ram_gb() -> float:
    try:
        import subprocess

        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "[math]::Round((Get-CimInstance Win32_ComputerSystem)"
                              ".TotalPhysicalMemory/1GB,1)"],
                             capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return float("nan")


def partitions() -> list[str]:
    """Chronological partition order: year_month ascending (matches the runner)."""
    return sorted((str(p) for p in BENCH.joinpath("data").rglob("*.parquet")),
                  key=lambda s: s.split("year_month=")[-1])


def load_prefix(n_rows: int, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read the first n_rows rows in chronological partition order."""
    X = np.empty((n_rows, len(features)), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.int8)
    n = 0
    for fp in partitions():
        if n >= n_rows:
            break
        pf = pq.ParquetFile(fp)
        for b in pf.iter_batches(batch_size=500_000, columns=features + ["label"]):
            df = b.to_pandas()
            k = min(len(df), n_rows - n)
            X[n:n + k] = df[features].to_numpy(dtype=np.float32)[:k]
            y[n:n + k] = df["label"].to_numpy().astype(np.int8)[:k]
            n += k
            if n >= n_rows:
                break
    return X[:n], y[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="1000000,5000000,10000000,25000000")
    ap.add_argument("--out", default="misc/reports/phase3_train_ladder.jsonl")
    ap.add_argument("--max-ram-gb", type=float, default=6.0,
                    help="refuse to materialise a rung whose float32 X alone "
                         "would exceed this (measured host has ~7 GB free)")
    ap.add_argument("--spw", type=float, default=SPW)
    args = ap.parse_args()

    manifest = json.loads(CONTRACT.read_text(encoding="utf-8"))
    features = list(manifest["features"])
    bench_man = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))
    ph = bench_man["partition_hashes"]
    import hashlib

    comb = hashlib.sha256()
    for k in sorted(ph):
        comb.update(k.encode())
        comb.update(ph[k].encode())

    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler

    env = {
        "cpu": cpu_model(),
        "ram_gb": total_ram_gb(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    for n_rows in [int(x) for x in args.ladder.split(",") if x.strip()]:
        rung: dict = {
            "rows": n_rows,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence_class": "SELF-TESTED — SYNTHETIC 50M — SCALE EXPERIMENT",
            "experiment_manifest": {
                "git_sha": __import__("subprocess").run(
                    ["git", "rev-parse", "HEAD"], capture_output=True,
                    text=True).stdout.strip(),
                "benchmark_combined_sha256": comb.hexdigest(),
                "benchmark_schema_hash": bench_man["schema_hash"],
                "benchmark_seed": bench_man["seed"],
                "generator_version": bench_man["generator_version"],
                "benchmark_source_sha256": bench_man["source_dataset_sha256"],
                "xgb_sha256": sha256_file(MODEL_DIR / "xgb_native.joblib"),
                "lgb_sha256": sha256_file(MODEL_DIR / "lgb_native.joblib"),
                "cb_sha256": sha256_file(MODEL_DIR / "cb_native.joblib"),
                "production_manifest_sha256": sha256_file(CONTRACT),
                "hyperparameter_source": ".freebuff/retrain_native_tuned.py",
                "hyperparameter_search": "NONE (fixed canonical values)",
                "scale_pos_weight": args.spw,
                "random_seed": SEED,
                "environment": env,
            },
        }

        need_gb = n_rows * len(features) * 4 / 2**30
        if need_gb > args.max_ram_gb:
            rung["status"] = "BLOCKED BY COMPUTE"
            rung["blocker"] = (f"float32 feature matrix alone needs {need_gb:.2f} GB "
                               f"> {args.max_ram_gb:.2f} GB available headroom")
            rung["rows_per_second"] = None
            rung["wall_seconds"] = None
            rung["peak_rss_MB"] = None
            print(f"[{n_rows:,}] BLOCKED BY COMPUTE — {rung['blocker']}", flush=True)
            _append(args.out, rung)
            continue

        print(f"[{n_rows:,}] loading prefix ...", flush=True)
        t0 = time.time()
        try:
            X, y = load_prefix(n_rows, features)
        except MemoryError as e:
            rung.update(status="BLOCKED BY COMPUTE", blocker=f"MemoryError: {e}",
                        rows=n_rows)
            print(f"[{n_rows:,}] MemoryError during load -> BLOCKED BY COMPUTE", flush=True)
            _append(args.out, rung)
            continue
        t_load = time.time() - t0
        n = len(X)
        pos = int(y.sum())
        rung["rows_materialised"] = n
        rung["fraud_rows"] = pos
        rung["fraud_prevalence"] = pos / n if n else None
        rung["load_seconds"] = t_load
        rung["load_rows_per_second"] = n / t_load if t_load else None
        rung["load_peak_rss_MB"] = peak_rss_mb()
        print(f"[{n_rows:,}] loaded {n:,} rows ({pos:,} fraud) in {t_load:.1f}s "
              f"peak_rss={rung['load_peak_rss_MB']:.0f}MB", flush=True)

        t1 = time.time()
        try:
            scaler = RobustScaler().fit(X)
            Xs = scaler.transform(X)
        except MemoryError as e:
            rung.update(status="BLOCKED BY COMPUTE", blocker=f"MemoryError scaling: {e}")
            _append(args.out, rung)
            continue
        rung["scale_seconds"] = time.time() - t1
        del X
        rung["scale_peak_rss_MB"] = peak_rss_mb()

        per_member = {}
        arts = {}
        try:
            tx = time.time()
            mx = xgb.XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.7,
                                   min_child_weight=5, scale_pos_weight=args.spw,
                                   random_state=SEED, n_jobs=4, eval_metric="auc")
            mx.fit(Xs, y, verbose=False)
            per_member["xgb"] = {
                "seconds": time.time() - tx,
                "rounds_requested": 400,
                "rounds_built": int(getattr(mx, "n_estimators", 0) or 0),
                "peak_rss_MB": peak_rss_mb(),
            }
            arts["xgb"] = joblib.dump(mx, SCRATCH / f"xgb_{n}.joblib", compress=3)
            print(f"  xgb {per_member['xgb']['seconds']:.1f}s", flush=True)

            tx = time.time()
            ml = lgb.LGBMClassifier(n_estimators=400, max_depth=7, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.7,
                                    min_child_weight=5, scale_pos_weight=args.spw,
                                    random_state=SEED, n_jobs=4, verbose=-1)
            ml.fit(Xs, y)
            per_member["lgb"] = {
                "seconds": time.time() - tx,
                "rounds_requested": 400,
                "rounds_built": int(ml.n_estimators_),
                "peak_rss_MB": peak_rss_mb(),
            }
            arts["lgb"] = joblib.dump(ml, SCRATCH / f"lgb_{n}.joblib", compress=3)
            print(f"  lgb {per_member['lgb']['seconds']:.1f}s", flush=True)

            tx = time.time()
            mc = cb.CatBoostClassifier(iterations=400, depth=7, learning_rate=0.05,
                                       l2_leaf_reg=3, random_seed=SEED, verbose=0,
                                       auto_class_weights="Balanced")
            mc.fit(Xs, y)
            per_member["cb"] = {
                "seconds": time.time() - tx,
                "rounds_requested": 400,
                "rounds_built": int(mc.tree_count_),
                "converged": "no early stopping configured (fixed 400 rounds)",
                "peak_rss_MB": peak_rss_mb(),
            }
            arts["cb"] = joblib.dump(mc, SCRATCH / f"cb_{n}.joblib", compress=3)
            print(f"  cb {per_member['cb']['seconds']:.1f}s", flush=True)
        except MemoryError as e:
            rung.update(status="BLOCKED BY COMPUTE",
                        blocker=f"MemoryError during fit: {e}",
                        per_member=per_member)
            print(f"[{n_rows:,}] MemoryError during fit -> BLOCKED BY COMPUTE", flush=True)
            _append(args.out, rung)
            continue

        train_s = sum(v["seconds"] for v in per_member.values())
        rung["status"] = "MEASURED"
        rung["per_member"] = per_member
        rung["wall_seconds"] = t_load + rung["scale_seconds"] + train_s
        rung["fit_seconds_total"] = train_s
        rung["rows_per_second"] = n / train_s if train_s else None
        rung["end_to_end_rows_per_second"] = n / rung["wall_seconds"] if rung["wall_seconds"] else None
        rung["peak_rss_MB"] = max(v["peak_rss_MB"] for v in per_member.values())
        rung["artifact_bytes"] = {k: int(os.path.getsize(p[0])) for k, p in arts.items()}
        rung["artifact_bytes_total"] = sum(rung["artifact_bytes"].values())
        rung["temporary_storage_bytes"] = rung["artifact_bytes_total"]
        rung["convergence_status"] = ("completed all 400 rounds per member; "
                                      "no early stopping configured")
        rung["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rung["wall_seconds"] = time.time() - t0
        rung["end_to_end_rows_per_second"] = n / rung["wall_seconds"]
        del Xs, y
        print(f"[{n_rows:,}] DONE fit={train_s:.1f}s wall={rung['wall_seconds']:.1f}s "
              f"fit_rows/s={rung['rows_per_second']:,.0f} peak_rss={rung['peak_rss_MB']:.0f}MB",
              flush=True)
        _append(args.out, rung)
    return 0


def _append(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, default=str) + "\n")
    print(f"  -> appended {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())