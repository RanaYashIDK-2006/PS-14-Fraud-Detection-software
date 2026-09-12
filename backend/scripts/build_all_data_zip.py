#!/usr/bin/env python3
"""Rebuild the curated PS14_ALL_DATA.zip.

Scope (matches the Round-8 curated bundle):
  - Full  scripts/ , src/ , reports/ , docs/, config/, benchmarks/
  - models/ : everything (joblib via LZMA, incl. production + model_records)
  - data/   : all files < 5 MB (configs/results) + provenance-critical core
              datasets (transactions_causal.csv, transactions.csv,
              altman_stats.pkl)
  - .freebuff/ : experiment scripts and JSON records (no logs/caches)
  - Root files: README, MODEL_CARD, Caddyfile*, docker-compose*, Makefile,
    DATA_GOVERNANCE.md, *_AUDIT.md
  Excluded: .env*, *.zip, db/, data giant corpora (IBM csv/parquet caches,
  kaggle/elliptic/paysim raw, uci/bank/diabetes raw), logs, caches.
"""
import os
import zipfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
OUT = ROOT / "PS14_ALL_DATA.zip"

SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", ".idea", "db",
             "certs", "cache", "data/_sorted_cache", "data/raw", ".pytest_cache"}
SKIP_EXTS = {".pyc", ".pyo", ".db", ".db-shm", ".db-wal", ".db-journal",
             ".log", ".zip", ".enc", ".pem", ".key", ".crt", ".npy", ".npz"}
SKIP_FILES = {".env", ".env.local", "PS14_ALL_DATA.zip"}

# Provenance-critical core datasets always included regardless of size.
CORE_DATA = {"transactions_causal.csv", "transactions.csv", "altman_stats.pkl"}
DATA_MAX = 5 * 1024 * 1024  # 5 MB for the curated data/ subset

MODEL_LZMA = {".joblib", ".pkl", ".bin", ".onnx"}
USE_LZMA = {".joblib"}  # LZMA only for model binaries (matches Round-8)

count = 0
total_bytes = 0


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    top = rel.parts[0] if len(rel.parts) > 1 else "."
    fname = path.name
    if fname in SKIP_FILES or any(fname.endswith(e) for e in SKIP_EXTS):
        return False
    if top == "data":
        if fname in CORE_DATA:
            return True
        return path.stat().st_size <= DATA_MAX
    return True


with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and
                   not any(d.startswith(s) for s in ("_sorted", "_cov_frame"))]
        for fname in files:
            fp = Path(root) / fname
            if not include(fp):
                continue
            rel = fp.relative_to(ROOT).as_posix()
            compress = zipfile.ZIP_LZMA if any(
                fname.endswith(e) for e in USE_LZMA) else zipfile.ZIP_DEFLATED
            with open(fp, "rb") as fh:
                data = fh.read()
            zf.writestr(zipfile.ZipInfo(rel), data, compress_type=compress)
            count += 1
            total_bytes += len(data)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "files": count,
        "uncompressed_bytes": total_bytes,
        "scope": "scripts+src+reports+docs+models(records+production)+curated data(<5MB)+core datasets+.freebuff experiments+root docs",
    }
    zf.writestr("ARCHIVE_MANIFEST.json",
                __import__("json").dumps(manifest, indent=2))

sz = OUT.stat().st_size
print(f"OK {OUT.name}: {count} files, {sz/1e6:.1f} MB "
      f"({total_bytes/1e6:.1f} MB uncompressed)")