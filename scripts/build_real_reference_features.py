"""Build a REAL reference feature matrix from IBM v2 on the native 48 contract.

Phase 2 validation needs a like-for-like comparison: dependency structure
(phase 2 section 7) and real-vs-synthetic distinguishability (section 12) are
only meaningful if the real side is featurised with the SAME contract and the
SAME strictly-past-only discipline as the benchmark.

This re-implements the runtime contract of
`backend/src/privacy_layer/native_features.py` for the real CSV:
  * amt = max(amount, 0)                      (runtime clamp)
  * is_night = hr < 6 or hr > 22              (runtime form, not the trainer's)
  * entity ids = sha256(raw)[:8] % 100000     (runtime _code)
  * historical aggregates strictly exclude the current row

Read-only. Never writes to the source dataset.

Usage:
    ./.venv/Scripts/python.exe scripts/build_real_reference_features.py \
        --rows 1200000 --out misc/reports/real_reference_features.parquet
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "data/credit_card_transactions-ibm_v2.csv"
CONTRACT = "models/production/manifest.json"
OUT = "misc/reports/real_reference_features.parquet"
META = "misc/reports/real_reference_features_meta.json"
COLD_START = 0.001


def _code(s: str) -> float:
    if not s:
        return 0.0
    return float(int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 100000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--meta", default=META)
    ap.add_argument("--rows", type=int, default=1_200_000)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--stride", type=int, default=0,
                    help="0 = auto (every Nth chunk, spanning all users)")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f"FATAL: source missing: {args.src}")
        return 1
    features = json.loads(open(CONTRACT, encoding="utf-8").read())["features"]

    reader = pd.read_csv(args.src, chunksize=args.chunk, dtype=str,
                         keep_default_na=False, na_values=[""])
    out_parts: list[pd.DataFrame] = []
    state: dict = {}
    seen = 0
    t0 = time.time()
    # IBM v2 is sorted by User, so a PREFIX of the file covers only the first
    # ~5% of users. Reading every Nth chunk spans the whole user population
    # while keeping each kept block internally time-ordered.
    total_chunks = None
    try:
        import pyarrow.csv as _pacsv  # noqa: F401
        total_chunks = 24_386_900 // args.chunk
    except Exception:
        total_chunks = 120
    stride = args.stride if args.stride > 0 else max(1, total_chunks // 24)
    chunk_i = -1

    for chunk in reader:
        chunk_i += 1
        if chunk_i % stride:
            continue
        if seen >= args.rows:
            break
        if seen + len(chunk) > args.rows:
            chunk = chunk.iloc[: args.rows - seen]
        seen += len(chunk)

        amt_raw = pd.to_numeric(chunk["Amount"].astype(str)
                                .str.replace(r"[$,]", "", regex=True),
                                errors="coerce")
        amt = amt_raw.fillna(0.0).to_numpy()
        amt = np.maximum(amt, 0.0)                      # runtime clamp

        hr = pd.to_numeric(chunk["Time"].str.slice(0, 2), errors="coerce").fillna(12).astype(int).to_numpy()
        mn = pd.to_numeric(chunk["Time"].str.slice(3, 5), errors="coerce").fillna(0).astype(int).to_numpy()
        month = pd.to_numeric(chunk["Month"], errors="coerce").fillna(1).astype(int).to_numpy()
        day = pd.to_numeric(chunk["Day"], errors="coerce").fillna(1).astype(int).to_numpy()
        yr = pd.to_numeric(chunk["Year"], errors="coerce").fillna(2002).astype(int).to_numpy()
        dt = pd.to_datetime(dict(year=yr, month=month, day=day), errors="coerce")
        dow = dt.dt.dayofweek.fillna(0).astype(int).to_numpy()
        label = (chunk["Is Fraud?"].str.strip() == "Yes").to_numpy().astype(np.int8)
        labelf = label.astype(np.float64)

        use_chip = chunk["Use Chip"].astype(str)
        is_chip = (use_chip == "Chip Transaction").to_numpy().astype(np.float64)
        is_online = (use_chip == "Online Transaction").to_numpy().astype(np.float64)
        is_swipe = (use_chip == "Swipe Transaction").to_numpy().astype(np.float64)
        # NaN must be treated as "no error". Naive astype(str) yields the
        # literal "nan", which would mark EVERY row as an error (mean err=1.0).
        # The runtime contract guards exactly this: not in ("", "nan", "None").
        raw_err = chunk["Errors?"]
        errs = raw_err.astype(str).where(raw_err.notna(), "")
        err = (~errs.isin(["", "nan", "None"])).to_numpy().astype(np.float64)
        has_zip = (chunk["Zip"].astype(str) != "").to_numpy().astype(np.float64)
        has_state = (chunk["Merchant State"].astype(str) != "").to_numpy().astype(np.float64)
        mcc = pd.to_numeric(chunk["MCC"], errors="coerce").fillna(0).to_numpy()

        u_raw = chunk["User"].astype(str)
        c_raw = chunk["Card"].astype(str)
        m_raw = chunk["Merchant Name"].astype(str)
        city_raw = chunk["Merchant City"].astype(str)
        card_key = (u_raw + "_" + c_raw).to_numpy()

        # Global entity maps: factorize() is per-chunk, so keys would NOT be
        # consistent across chunks and the running state would be corrupted.
        # These dictionaries persist for the whole build.
        def to_key(series: pd.Series, name: str) -> np.ndarray:
            mp = state.setdefault(f"map_{name}", {})
            idx = {v: i for i, v in enumerate(mp)}
            base = len(mp)
            out = series.map(lambda v: idx.get(v, -1)).to_numpy()
            need = out < 0
            if need.any():
                newvals = series[need].drop_duplicates().tolist()
                for v in newvals:
                    idx[v] = len(mp)
                    mp[v] = len(mp) - 1
                out = series.map(lambda v: idx[v]).to_numpy()
            state.setdefault(f"n_{name}", base)
            return out.astype(np.int64)

        u_key = to_key(u_raw, "user")
        card_key2 = to_key(pd.Series(card_key), "card")
        m_key = to_key(m_raw, "merch")
        city_key = to_key(city_raw, "city")
        n_user = len(state["map_user"])
        n_card = len(state["map_card"])
        n_merch = len(state["map_merch"])
        n_city = len(state["map_city"])
        m_uni = np.arange(n_merch)
        city_uni = np.arange(n_city)
        u_uni = np.arange(n_user)

        for name, n_ent in (("user", n_user), ("card", n_card),
                            ("merch", n_merch), ("city", n_city)):
            for k in ("n", "amt_sum", "fraud_n"):
                st = state.setdefault(f"{name}_{k}", np.zeros(n_ent))
                if len(st) < n_ent:                    # grow if new entities appear
                    state[f"{name}_{k}"] = np.pad(st, (0, n_ent - len(st)))
            if name == "city":                         # city amount sum not used
                state.setdefault("city_amt_sum", np.zeros(n_city))

        def before(prior_n, prior_sum, values, keys):
            df = pd.DataFrame({"k": keys, "v": values})
            cum_n = df.groupby("k").cumcount().to_numpy()
            cum_s = (df.groupby("k")["v"].cumsum() - df["v"]).to_numpy()
            cnt = prior_n[keys] + cum_n
            tot = prior_sum[keys] + cum_s
            mean = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
            return cnt.astype(np.float64), mean

        user_tx_count, user_avg_amt = before(state["user_n"], state["user_amt_sum"], amt, u_key)
        card_tx_count, _ = before(state["card_n"], state["card_amt_sum"], amt, card_key2)
        merch_tx_count, _ = before(state["merch_n"], state["merch_amt_sum"], amt, m_key)
        city_n_before, _ = before(state["city_n"], state["city_amt_sum"], amt, city_key)

        def rate_before(name, keys, counts):
            cum = (pd.DataFrame({"k": keys, "v": labelf}).groupby("k")["v"]
                   .cumsum() - labelf).to_numpy()
            tot = state[f"{name}_fraud_n"][keys] + cum
            return np.where(counts > 0, tot / np.maximum(counts, 1.0), COLD_START)

        ufr = rate_before("user", u_key, user_tx_count)
        mfr = rate_before("merch", m_key, merch_tx_count)
        cfr = rate_before("city", city_key, city_n_before)

        # Distinct merchant / city per user, STRICTLY BEFORE this row.
        # A global first-occurrence row for (user, entity) is the only one that
        # increments the user's diversity; its increment lands on the rows that
        # FOLLOW it. Applying the end-of-chunk final count to every row (as an
        # earlier version did) leaks future information and inflates the feature.
        def diversity_before(entity_keys, n_ent, tag):
            prior = state.setdefault(f"{tag}_div", np.zeros(n_user))
            if len(prior) < n_user:
                state[f"{tag}_div"] = np.pad(prior, (0, n_user - len(prior)))
                prior = state[f"{tag}_div"]
            pair = u_key.astype(np.int64) * (n_ent + 1) + entity_keys
            seen = state.setdefault(f"{tag}_seen", set())
            first_mask = np.array([p not in seen for p in pair.tolist()], dtype=bool)
            inc = np.zeros(len(u_key), dtype=np.float64)
            if first_mask.any():
                sub = pd.DataFrame({"u": u_key[first_mask],
                                    "pos": np.flatnonzero(first_mask)})
                sub = sub.sort_values(["u", "pos"])
                rank = sub.groupby("u").cumcount().to_numpy().astype(np.float64)
                inc[first_mask] = rank
                seen.update(pair[first_mask].tolist())
            return np.maximum(prior[u_key] + inc, 1.0), inc

        user_merchant_diversity, inc_um = diversity_before(m_key, n_merch, "um")
        user_city_diversity, inc_uc = diversity_before(city_key, n_city, "uc")

        night = ((hr < 6) | (hr > 22)).astype(np.float64)
        mccf = mcc.astype(np.float64)
        feat = pd.DataFrame({
            "amt": np.round(amt, 4),
            "log_amt": np.round(np.log1p(amt), 4),
            "amt_sq": np.round(amt * amt, 4),
            "hr": hr.astype(np.float64), "mn": mn.astype(np.float64),
            "dow": dow.astype(np.float64),
            "Month": month.astype(np.float64), "Day": day.astype(np.float64),
            "hour_sin": np.round(np.sin(2 * np.pi * hr / 24), 6),
            "hour_cos": np.round(np.cos(2 * np.pi * hr / 24), 6),
            "is_night": night,
            "is_business_hours": ((hr >= 9) & (hr <= 17)).astype(np.float64),
            "chip": is_chip, "is_online": is_online, "is_swipe": is_swipe,
            "err": err, "has_zip": has_zip, "has_state": has_state,
            "is_online_or_no_state": ((is_online > 0) | (has_state == 0)).astype(np.float64),
            "mcc": mccf,
            "mcc_high": (mccf >= 5000).astype(np.float64),
            "mcc_restaurant": ((mccf >= 5812) & (mccf <= 5814)).astype(np.float64),
            "mcc_gas": ((mccf >= 5541) & (mccf <= 5542)).astype(np.float64),
            "mcc_grocery": ((mccf >= 5411) & (mccf <= 5422)).astype(np.float64),
            "mcc_travel": ((mccf >= 3000) & (mccf <= 3350)).astype(np.float64),
            "mcc_online": ((mccf >= 5967) & (mccf <= 5969)).astype(np.float64),
            "merchant_id": np.array([_code(v) for v in m_raw]),
            "city_id": np.array([_code(v) for v in city_raw]),
            "card_id": np.array([_code(v) for v in card_key]),
            "user_tx_count": user_tx_count,
            "card_tx_count": card_tx_count,
            "user_avg_amt": np.round(user_avg_amt, 4),
            "amt_vs_user_avg": np.round(np.where(user_avg_amt > 0,
                                                amt / (user_avg_amt + 1e-6), 1.0), 6),
            "amt_zscore": np.round(np.where(user_avg_amt > 0,
                                            (amt - user_avg_amt) / (user_avg_amt + 1e-6), 0.0), 6),
            "merch_tx_count": merch_tx_count,
            "user_merchant_diversity": user_merchant_diversity,
            "user_city_diversity": user_city_diversity,
            "user_fraud_rate": np.round(ufr, 6),
            "merch_fraud_rate": np.round(mfr, 6),
            "city_fraud_rate": np.round(cfr, 6),
            "high_amt": ((user_avg_amt > 0) & (amt > user_avg_amt * 2)).astype(np.float64),
            "very_high_amt": ((user_avg_amt > 0) & (amt > user_avg_amt * 5)).astype(np.float64),
            "amt_x_hr": np.round(amt * hr, 4),
            "amt_x_mcc": np.round(amt * mccf, 4),
            "amt_x_chip": np.round(amt * is_chip, 4),
            "amt_x_online": np.round(amt * is_online, 4),
            "amt_x_night": np.round(amt * night, 4),
            "user_merch_count": user_merchant_diversity,
        })
        feat = feat[features]
        feat["label"] = label
        feat["split"] = "reference"
        feat["provenance_class"] = "DERIVED_FROM_OBSERVED"
        out_parts.append(feat)

        np.add.at(state["user_n"], u_key, 1)
        np.add.at(state["user_amt_sum"], u_key, amt)
        np.add.at(state["user_fraud_n"], u_key, labelf)
        np.add.at(state["card_n"], card_key2, 1)
        np.add.at(state["card_amt_sum"], card_key2, amt)
        np.add.at(state["merch_n"], m_key, 1)
        np.add.at(state["merch_amt_sum"], m_key, amt)
        np.add.at(state["merch_fraud_n"], m_key, labelf)
        np.add.at(state["city_n"], city_key, 1)
        np.add.at(state["city_amt_sum"], city_key, amt)
        np.add.at(state["city_fraud_n"], city_key, labelf)
        if "um_div" in state:
            np.add.at(state["um_div"], u_key, inc_um)
        if "uc_div" in state:
            np.add.at(state["uc_div"], u_key, inc_uc)
        print(f"  processed {seen:,} rows ({time.time()-t0:.0f}s)", flush=True)

    df = pd.concat(out_parts, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), args.out,
                   compression="zstd")
    meta = {
        "source": args.src,
        "rows": int(len(df)),
        "fraud_rows": int(df["label"].sum()),
        "fraud_prevalence": float(df["label"].mean()),
        "feature_contract": CONTRACT,
        "semantics": "runtime contract native_features.py, past-only",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(args.meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {args.out} rows={len(df):,} fraud={int(df['label'].sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
