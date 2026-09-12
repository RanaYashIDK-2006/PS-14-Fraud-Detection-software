#!/usr/bin/env python3
"""Phase-10B — build the TRAIN-window (<2016) raw replay cache.

Firewall: this script only emits `tr_*` arrays (year < 2016). It does NOT
save va (2016-2017) or te (>=2018) rows. The existing data/_raw_parity_cache.npz
only contains va/te; te contains the forbidden final-test window, so it is
NEVER loaded by the Gate B harness. Output: data/_raw_parity_cache_tr.npz.

Identical sampling to mission_cache.py: all fraud + 5% legit, rng 42,
chronological sort, then the shared build_context_and_features (expanding,
shifted context) + derive_native_features per row -> X (float32).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import scripts.retrain_native_consistent as rt  # noqa: E402
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES  # noqa: E402

OUT = ROOT / "data" / "_raw_parity_cache_tr.npz"
OUT_META = ROOT / "data" / "_raw_parity_cache_tr_meta.json"
SEED = 42
LEGIT_FRAC = 0.05


def main() -> int:
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    chunks = []
    for chunk in pd.read_csv(rt.DATA_CSV, usecols=rt.USECOLS, low_memory=False,
                             chunksize=1_000_000):
        fr = chunk["Is Fraud?"].eq("Yes")
        keep = fr | (rng.random(len(chunk)) < LEGIT_FRAC)
        chunks.append(chunk[keep].copy())
    df = pd.concat(chunks, ignore_index=True)
    df["amt"] = (df["Amount"].str.replace("$", "", regex=False)
                 .str.replace(",", "", regex=False).astype(float))
    df["is_fraud"] = df["Is Fraud?"].eq("Yes").astype(int)
    df["ts"] = pd.to_datetime(
        df[["Year", "Month", "Day"]].assign(
            hour=df["Time"].str.split(":").str[0].astype(int),
            minute=df["Time"].str.split(":").str[1].astype(int)))
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"sampled {len(df):,} | {time.time()-t0:.0f}s", flush=True)

    yrs = df["ts"].dt.year.to_numpy()
    tr = yrs < 2016
    n_tr = int(tr.sum())
    n_tr_fr = int(df.loc[tr, "is_fraud"].sum())
    print(f"train window rows={n_tr:,} fraud={n_tr_fr:,} | {time.time()-t0:.0f}s",
          flush=True)

    F = rt.build_context_and_features(df)
    print(f"features done | {time.time()-t0:.0f}s", flush=True)

    def _e(s):
        s = str(s)
        return "" if s in ("nan", "None") else s

    d = df.loc[tr].reset_index(drop=True)
    ts_ = d["ts"]
    rec = {
        "amount": d["amt"].to_numpy(np.float64),
        "iso": (ts_.dt.strftime("%Y-%m-%dT%H:%M:%S")).to_numpy().astype("S19"),
        "use_chip": d["Use Chip"].fillna("").map(str).to_numpy().astype("S24"),
        "mcc": d["MCC"].fillna(0).astype(int).to_numpy(),
        "merchant_name": d["Merchant Name"].map(str).to_numpy().astype("S48"),
        "merchant_city": d["Merchant City"].fillna("").map(str).to_numpy().astype("S40"),
        "merchant_state": d["Merchant State"].fillna("").map(str).to_numpy().astype("S4"),
        "zip": d["Zip"].map(lambda v: "" if pd.isna(v) else str(v)).to_numpy().astype("S12"),
        "card": d["Card"].map(str).to_numpy().astype("S24"),
        "errors": d["Errors?"].map(_e).to_numpy().astype("S32"),
        "user_id": d["User"].map(str).to_numpy().astype("S32"),
        "user_tx_count": d["user_tx_count"].to_numpy(np.float64),
        "user_avg_amt": d["user_avg_amt"].to_numpy(np.float64),
        "card_tx_count": d["card_tx_count"].to_numpy(np.float64),
        "merch_tx_count": d["merch_tx_count"].to_numpy(np.float64),
        "user_merchant_diversity": d["user_merchant_diversity"].to_numpy(np.float64),
        "user_city_diversity": d["user_city_diversity"].to_numpy(np.float64),
        "user_merch_count": d["user_merch_count"].to_numpy(np.float64),
        "user_fraud_rate": d["user_fraud_rate"].to_numpy(np.float64),
        "merch_fraud_rate": d["merch_fraud_rate"].to_numpy(np.float64),
        "city_fraud_rate": d["city_fraud_rate"].to_numpy(np.float64),
        "X": np.asarray(F.loc[tr, ALTMAN_NATIVE_FEATURES].values, dtype=np.float32),
        "y": d["is_fraud"].to_numpy(np.int8),
        "ts": (ts_.values.astype("datetime64[ns]").astype("int64") // 10**9).astype(np.int64),
        "year": d["Year"].to_numpy(np.int16),
    }
    arr = {f"tr_{k}": v for k, v in rec.items()}
    np.savez_compressed(OUT, **arr)

    import hashlib
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    meta = {
        "path": str(OUT),
        "sha256": h,
        "rows": n_tr,
        "fraud": n_tr_fr,
        "year_min": int(yrs.min()),
        "year_max": int(yrs[tr].max()),
        "sampling": f"all fraud + {LEGIT_FRAC:.0%} legit (rng {SEED})",
        "window": "train <2016 ONLY — va/te windows never saved",
        "final_test_rows_present": 0,
        "built_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Firewall: only tr_* arrays emitted; te/va excluded by construction.",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=1), flush=True)
    print(f"saved {OUT} | {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())