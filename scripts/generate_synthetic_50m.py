"""Deterministic generator for the PS-14 50M synthetic scale benchmark.

Implements `docs/evaluation/SYNTHETIC_50M_BENCHMARK_SPEC.md` (v1.0).

Design constraints taken from the spec and from the repository:
  * Feature semantics follow the RUNTIME contract in
    `backend/src/privacy_layer/native_features.py` (which the production
    manifest names as the feature source), NOT the training script.
  * Every historical aggregate is strictly PAST-ONLY: a row's own amount and
    label are never folded into its own features.
  * The production model is never imported, executed, or referenced. Labels
    come from a declared logistic mechanism over generator-internal latents.
  * The real dataset is opened read-only and is never written, renamed, or
    appended to. The output directory is refused if it resolves to the source.

Usage:
    ./.venv/Scripts/python.exe scripts/generate_synthetic_50m.py --rows 50000000
    ./.venv/Scripts/python.exe scripts/generate_synthetic_50m.py --rows 20000 --smoke
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 20261005
SOURCE = "data/credit_card_transactions-ibm_v2.csv"
PROFILE = "misc/reports/ibm_v2_empirical_profile.json"
CONTRACT = "models/production/manifest.json"
OUT_ROOT = "data/synthetic_50m"

# Runtime cold-start default (native_features.COLD_START_FRAUD_RATE)
COLD_START_FRAUD_RATE = 0.001

# Declared fraud-mechanism coefficients (spec section 7). Generator-internal
# latents; no model is consulted.
FRAUD_B = {
    "b0": None,          # solved by bisection to hit target prevalence
    "b_amt": 0.85,
    "b_online": 0.55,
    "b_night": 0.30,
    "b_err": 0.45,
    "b_new_merchant": 0.70,
    "b_velocity": 0.25,
    "b_far": 0.40,
}
TARGET_PREVALENCE = 0.0012202042900081602  # observed IBM v2 prevalence

CHANNELS = ["Swipe Transaction", "Chip Transaction", "Online Transaction"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def entity_code(s: str) -> float:
    """Identical to native_features._code: sha256(s)[:8] as int, mod 100000."""
    if not s:
        return 0.0
    return float(int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 100000)


def load_contract() -> list[str]:
    with open(CONTRACT, encoding="utf-8") as fh:
        return json.load(fh)["features"]


def entity_code_table(n: int, prefix: str) -> np.ndarray:
    """Precompute hash-based codes for n synthetic entity ids (vectorised)."""
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = entity_code(f"{prefix}{i}")
    return out


def refuse_unsafe_output(out_root: Path, source: Path) -> None:
    """Fail safely: never write to, or clobber, the real dataset."""
    src = source.resolve()
    out = out_root.resolve()
    if out == src or src in out.parents or out in src.parents:
        raise SystemExit(
            f"FATAL: output {out} overlaps the real dataset {src}. Refusing."
        )
    if str(out).replace("\\", "/").lower().endswith(".csv"):
        raise SystemExit("FATAL: refusing to write the benchmark as a .csv file.")


def solve_b0(rng: np.random.Generator, z_amt, online, night, err,
             new_merch, vel, far, prevalence: float) -> float:
    """Bisect the intercept so mean(sigmoid(logit)) == target prevalence."""
    rest = (FRAUD_B["b_amt"] * z_amt + FRAUD_B["b_online"] * online
            + FRAUD_B["b_night"] * night + FRAUD_B["b_err"] * err
            + FRAUD_B["b_new_merchant"] * new_merch
            + FRAUD_B["b_velocity"] * vel + FRAUD_B["b_far"] * far)

    def mean_p(b0: float) -> float:
        return float(np.mean(1.0 / (1.0 + np.exp(-(b0 + rest)))))

    lo, hi = -30.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_p(mid) > prevalence:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def build_month(args, prof, state, rng, year, month, n_rows, codes, cal):
    """Generate one monthly partition with strictly past-only aggregates."""
    days_in_month = calendar.monthrange(year, month)[1]

    # ---- timestamps (within the month), sorted -----------------------------
    day = rng.integers(1, days_in_month + 1, n_rows)
    hour = rng.choice(24, size=n_rows, p=cal["hour_p"])
    minute = rng.integers(0, 60, n_rows)
    ts = pd.to_datetime(
        dict(year=np.full(n_rows, year), month=np.full(n_rows, month),
             day=day, hour=hour, minute=minute)
    ).sort_values().reset_index(drop=True)
    day, hour, minute = ts.dt.day.to_numpy(), ts.dt.hour.to_numpy(), ts.dt.minute.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()

    # ---- entities ---------------------------------------------------------
    n_users = args.n_users
    user = rng.integers(0, n_users, n_rows)
    # users are unevenly active (Pareto-like), matching observed intensity spread
    user = np.clip((user.astype(np.float64) ** 1.0).astype(np.int64), 0, n_users - 1)

    card = (user * 2 + rng.integers(0, 2, n_rows)) % args.n_cards

    # merchant: mostly from the user's persistent pool, occasionally new
    pool_size = state["user_pool_size"]
    offset = rng.integers(0, pool_size[user])
    merch = (state["user_anchor"][user] + offset) % args.n_merchants
    is_new_merch = (offset == 0) & (state["user_merch_n"][user] == 0)
    # a small share of genuinely new merchants
    new_draw = rng.random(n_rows) < 0.012
    merch = np.where(new_draw, rng.integers(0, args.n_merchants, n_rows), merch)
    is_new_merch = np.where(new_draw, True, is_new_merch)

    city = state["merch_city"][merch]

    # ---- amount: quantile map + per-user tendency + channel factor --------
    u = rng.random(n_rows)
    base = np.interp(u, cal["amt_q"], cal["amt_v"])
    tendency = state["user_tendency"][user]
    amt = base * tendency

    # ---- channel ----------------------------------------------------------
    chan_idx = rng.choice(3, size=n_rows, p=cal["channel_p"])
    is_online = (chan_idx == 2).astype(np.float64)
    is_chip = (chan_idx == 1).astype(np.float64)
    is_swipe = (chan_idx == 0).astype(np.float64)
    amt = amt * np.where(is_online > 0, 1.15, np.where(is_chip > 0, 1.0, 0.95))

    # ---- errors: card-not-present inflated --------------------------------
    err_p = np.where((is_online > 0) | (is_chip > 0), cal["err_p_cnp"], cal["err_p"])
    has_err = (rng.random(n_rows) < err_p).astype(np.float64)

    mcc = state["merch_mcc"][merch]

    # ---- runtime amount clamp: amt = max(amt, 0) --------------------------
    amt_feat = np.maximum(amt, 0.0)

    # ================= PAST-ONLY HISTORICAL AGGREGATES ====================
    n_merch_seen = state["user_merch_n"]      # distinct merchants used so far
    n_city_seen = state["user_city_n"]
    ufr_prior = state["user_fraud_n"] / np.maximum(state["user_n"], 1)
    mfr_prior = state["merch_fraud_n"] / np.maximum(state["merch_n"], 1)
    cfr_prior = state["city_fraud_n"] / np.maximum(state["city_n"], 1)
    ufr_prior = np.where(state["user_n"] > 0, ufr_prior, COLD_START_FRAUD_RATE)
    mfr_prior = np.where(state["merch_n"] > 0, mfr_prior, COLD_START_FRAUD_RATE)
    cfr_prior = np.where(state["city_n"] > 0, cfr_prior, COLD_START_FRAUD_RATE)

    # per-row "before" values via stable group ordering (rows already ts-sorted)
    def before(prior_n, prior_sum, values, keys):
        """Return (count_before, mean_before) with the row itself excluded."""
        df = pd.DataFrame({"k": keys, "v": values})
        cum_n = df.groupby("k").cumcount()
        cum_s = df.groupby("k")["v"].cumsum() - df["v"]  # strictly before
        cnt = prior_n[keys] + cum_n.to_numpy()
        tot = prior_sum[keys] + cum_s.to_numpy()
        mean = np.divide(tot, cnt, out=np.zeros_like(tot, dtype=np.float64),
                         where=cnt > 0)
        return cnt.astype(np.float64), mean

    user_tx_count, user_avg_amt = before(
        state["user_n"].astype(np.float64), state["user_amt_sum"],
        amt_feat, user)
    card_tx_count, _ = before(
        state["card_n"].astype(np.float64), state["card_amt_sum"],
        amt_feat, card)
    merch_tx_count, _ = before(
        state["merch_n"].astype(np.float64), state["merch_amt_sum"],
        amt_feat, merch)
    city_n_before, _ = before(
        state["city_n"].astype(np.float64), np.zeros(args.n_cities), amt_feat, city)

    # fraud rates are computed after the label, above.

    # ---- distinct merchant / city counts per user, strictly before this row
    pair_first = ~pd.DataFrame({"k": user.astype(np.int64) * args.n_merchants + merch})["k"].duplicated()
    pool_rank = np.zeros(n_rows, dtype=np.float64)
    if pair_first.any():
        sub = pd.DataFrame({"u": user[pair_first], "t": ts[pair_first].astype("int64")})
        sub = sub.sort_values(["u", "t"])
        grp_rank = sub.groupby("u").cumcount().to_numpy()
        pool_rank[pair_first] = grp_rank.astype(np.float64)
    user_merchant_diversity = (n_merch_seen[user] + pool_rank).clip(min=1.0)
    user_merch_count = user_merchant_diversity
    user_city_diversity = n_city_seen[user].astype(np.float64).clip(min=1.0)

    # ================= LABEL (declared mechanism, no model) =================
    # Computed BEFORE the fraud-rate features. The mechanism deliberately does
    # NOT consume user/merchant/city fraud rates, so ordering is not circular.
    z_amt = np.log1p(np.abs(amt_feat))
    vel = np.log1p(user_tx_count)
    far = np.abs(city - state["user_home_city"][user]).astype(np.float64) / args.n_cities
    z_amt = (z_amt - z_amt.mean()) / max(z_amt.std(), 1e-6)
    vel = (vel - vel.mean()) / max(vel.std(), 1e-6)
    night_f = ((hour < 6) | (hour > 22)).astype(np.float64)

    b0 = solve_b0(rng, z_amt, is_online, night_f, has_err,
                  is_new_merch.astype(np.float64), vel, far, TARGET_PREVALENCE)
    logit = (b0 + FRAUD_B["b_amt"] * z_amt + FRAUD_B["b_online"] * is_online
             + FRAUD_B["b_night"] * night_f + FRAUD_B["b_err"] * has_err
             + FRAUD_B["b_new_merchant"] * is_new_merch.astype(np.float64)
             + FRAUD_B["b_velocity"] * vel + FRAUD_B["b_far"] * far)
    p = 1.0 / (1.0 + np.exp(-logit))
    label = (rng.random(n_rows) < p).astype(np.int8)
    label_f = label.astype(np.float64)

    # ---- fraud rates: prior-month state PLUS this month's strictly-earlier rows
    def rate_before(prior_fraud, prior_n, keys, counts_before):
        cum = (pd.DataFrame({"k": keys, "v": label_f}).groupby("k")["v"]
               .cumsum() - label_f).to_numpy()
        tot = prior_fraud[keys] + cum
        cnt = counts_before
        return np.where(cnt > 0, tot / np.maximum(cnt, 1.0), COLD_START_FRAUD_RATE)

    ufr = rate_before(state["user_fraud_n"], state["user_n"], user, user_tx_count)
    mfr = rate_before(state["merch_fraud_n"], state["merch_n"], merch, merch_tx_count)
    cfr = rate_before(state["city_fraud_n"], state["city_n"], city, city_n_before)

    # ================= 48 FEATURES (runtime semantics) ======================
    amt_vs_user_avg = np.where(user_avg_amt > 0, amt_feat / (user_avg_amt + 1e-6), 1.0)
    amt_zscore = np.where(user_avg_amt > 0,
                          (amt_feat - user_avg_amt) / (user_avg_amt + 1e-6), 0.0)
    high_amt = ((user_avg_amt > 0) & (amt_feat > user_avg_amt * 2)).astype(np.float64)
    very_high_amt = ((user_avg_amt > 0) & (amt_feat > user_avg_amt * 5)).astype(np.float64)
    is_night = ((hour < 6) | (hour > 22)).astype(np.float64)
    mcc_f = mcc.astype(np.float64)

    feats = pd.DataFrame({
        "amt": np.round(amt_feat, 4),
        "log_amt": np.round(np.log1p(amt_feat), 4),
        "amt_sq": np.round(amt_feat * amt_feat, 4),
        "hr": hour.astype(np.float64), "mn": minute.astype(np.float64),
        "dow": dow.astype(np.float64),
        "Month": np.full(n_rows, float(month)), "Day": day.astype(np.float64),
        "hour_sin": np.round(np.sin(2 * np.pi * hour / 24), 6),
        "hour_cos": np.round(np.cos(2 * np.pi * hour / 24), 6),
        "is_night": is_night,
        "is_business_hours": ((hour >= 9) & (hour <= 17)).astype(np.float64),
        "chip": is_chip, "is_online": is_online, "is_swipe": is_swipe,
        "err": has_err, "has_zip": np.ones(n_rows), "has_state": np.ones(n_rows),
        "is_online_or_no_state": is_online,
        "mcc": mcc_f,
        "mcc_high": (mcc_f >= 5000).astype(np.float64),
        "mcc_restaurant": ((mcc_f >= 5812) & (mcc_f <= 5814)).astype(np.float64),
        "mcc_gas": ((mcc_f >= 5541) & (mcc_f <= 5542)).astype(np.float64),
        "mcc_grocery": ((mcc_f >= 5411) & (mcc_f <= 5422)).astype(np.float64),
        "mcc_travel": ((mcc_f >= 3000) & (mcc_f <= 3350)).astype(np.float64),
        "mcc_online": ((mcc_f >= 5967) & (mcc_f <= 5969)).astype(np.float64),
        "merchant_id": codes["merch"][merch],
        "city_id": codes["city"][city],
        "card_id": codes["card"][card],
        "user_tx_count": user_tx_count,
        "card_tx_count": card_tx_count,
        "user_avg_amt": np.round(user_avg_amt, 4),
        "amt_vs_user_avg": np.round(amt_vs_user_avg, 6),
        "amt_zscore": np.round(amt_zscore, 6),
        "merch_tx_count": merch_tx_count,
        "user_merchant_diversity": user_merchant_diversity,
        "user_city_diversity": user_city_diversity,
        "user_fraud_rate": np.round(ufr, 6),
        "merch_fraud_rate": np.round(mfr, 6),
        "city_fraud_rate": np.round(cfr, 6),
        "high_amt": high_amt, "very_high_amt": very_high_amt,
        "amt_x_hr": np.round(amt_feat * hour, 4),
        "amt_x_mcc": np.round(amt_feat * mcc_f, 4),
        "amt_x_chip": np.round(amt_feat * is_chip, 4),
        "amt_x_online": np.round(amt_feat * is_online, 4),
        "amt_x_night": np.round(amt_feat * is_night, 4),
        "user_merch_count": user_merch_count,
    })
    feats["label"] = label
    feats["user_id"] = user.astype(np.int32)
    feats["year_month"] = f"{year:04d}-{month:02d}"
    feats["split"] = np.where(state["user_split"][user] == 0, "train", "test")
    feats["provenance_class"] = "SYNTHETIC"

    # ---- fold this month into the running state (AFTER features) ----------
    np.add.at(state["user_n"], user, 1)
    np.add.at(state["user_amt_sum"], user, amt_feat)
    np.add.at(state["user_fraud_n"], user, label.astype(np.float64))
    np.add.at(state["card_n"], card, 1)
    np.add.at(state["card_amt_sum"], card, amt_feat)
    np.add.at(state["merch_n"], merch, 1)
    np.add.at(state["merch_amt_sum"], merch, amt_feat)
    np.add.at(state["merch_fraud_n"], merch, label.astype(np.float64))
    np.add.at(state["city_n"], city, 1)
    np.add.at(state["city_fraud_n"], city, label.astype(np.float64))
    # Distinct user-merchant / user-city counts. Done with C-speed set algebra
    # on (user, entity) keys so the 50M-row run stays linear and fast.
    mkey = np.unique(user.astype(np.int64) * args.n_merchants + merch)
    new_m = set(mkey.tolist()) - state["seen_merch_pairs"]
    state["seen_merch_pairs"].update(new_m)
    if new_m:
        arr = np.fromiter(new_m, dtype=np.int64, count=len(new_m))
        np.add.at(state["user_merch_n"], arr // args.n_merchants, 1.0)

    ckey = np.unique(user.astype(np.int64) * args.n_cities + city)
    new_c = set(ckey.tolist()) - state["seen_city_pairs"]
    state["seen_city_pairs"].update(new_c)
    if new_c:
        arr = np.fromiter(new_c, dtype=np.int64, count=len(new_c))
        np.add.at(state["user_city_n"], arr // args.n_cities, 1.0)

    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000_000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--smoke", action="store_true",
                    help="include 1 OBSERVED row to exercise the provenance class")
    ap.add_argument("--keep", action="store_true", help="do not wipe an existing output dir")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"FATAL: source dataset missing: {src}")
        return 1
    out_root = Path(args.out)
    refuse_unsafe_output(out_root, src)

    with open(args.profile, encoding="utf-8") as fh:
        prof = json.load(fh)
    features = load_contract()
    if len(features) != 48:
        print(f"FATAL: contract has {len(features)} features, expected 48")
        return 1

    # --- calibration from the observed profile ----------------------------
    obs_n = prof["volume"]["n_rows"]
    scale = args.rows / obs_n
    args.n_users = max(2, int(round(prof["entities"]["n_users"] * scale)))
    args.n_merchants = max(4, int(round(prof["entities"]["n_merchants"] * scale)))
    args.n_cities = max(2, int(round(prof["entities"]["n_cities"] * scale)))
    args.n_cards = args.n_users * 2

    uc = prof["categorical"]["use_chip"]
    tot = sum(uc.values())
    channel_p = np.array([uc.get(c, 0) / tot for c in CHANNELS])

    err_rate = sum(prof["categorical"]["errors"].values()) / obs_n
    err_p = float(err_rate)
    err_p_cnp = float(min(err_rate * 2.4, 0.9))

    amt_q = np.array([0.0, .01, .05, .25, .5, .75, .90, .95, .99, 1.0])
    qv = prof["amount"]["quantiles"]
    amt_v = np.array([-96.0, qv["p1"], qv["p5"], qv["p25"], qv["p50"],
                      qv["p75"], qv["p90"], qv["p95"], qv["p99"], qv["p99.9"]])
    hour_hist = prof["temporal"]["hours"]
    hour_p = np.array([hour_hist.get(str(h), 0) for h in range(24)], dtype=np.float64)
    hour_p = hour_p / hour_p.sum()

    mcc_top = prof["categorical"]["mcc_top20"]
    mcc_vals = np.array(sorted(int(k) for k in mcc_top), dtype=np.int64)
    mcc_w = np.array([mcc_top[str(k)] for k in mcc_vals], dtype=np.float64)
    mcc_w = mcc_w / mcc_w.sum()

    cal = dict(channel_p=channel_p, err_p=err_p, err_p_cnp=err_p_cnp,
               amt_q=amt_q, amt_v=amt_v, hour_p=hour_p)

    # --- entity state ------------------------------------------------------
    ss = np.random.SeedSequence(args.seed)
    n_months = 166
    month_seeds = ss.spawn(n_months)

    init = np.random.SeedSequence(args.seed + 1)
    ri = np.random.default_rng(init)
    state = {
        "user_n": np.zeros(args.n_users, dtype=np.float64),
        "user_amt_sum": np.zeros(args.n_users),
        "user_fraud_n": np.zeros(args.n_users),
        "card_n": np.zeros(args.n_cards),
        "card_amt_sum": np.zeros(args.n_cards),
        "merch_n": np.zeros(args.n_merchants),
        "merch_amt_sum": np.zeros(args.n_merchants),
        "merch_fraud_n": np.zeros(args.n_merchants),
        "city_n": np.zeros(args.n_cities),
        "city_fraud_n": np.zeros(args.n_cities),
        "merch_city": ri.integers(0, args.n_cities, args.n_merchants),
        "merch_mcc": ri.choice(mcc_vals, size=args.n_merchants, p=mcc_w),
        "user_anchor": ri.integers(0, args.n_merchants, args.n_users),
        "user_pool_size": ri.integers(3, 40, args.n_users),
        "user_tendency": np.exp(ri.normal(0.0, 0.35, args.n_users)),
        "user_home_city": ri.integers(0, args.n_cities, args.n_users),
        "user_split": ri.integers(0, 2, args.n_users),
        "seen_merch_pairs": set(),
        "seen_city_pairs": set(),
        "user_merch_n": np.zeros(args.n_users),
        "user_city_n": np.zeros(args.n_users),
    }
    codes = {
        "merch": entity_code_table(args.n_merchants, "SYN_MERCH_"),
        "city": entity_code_table(args.n_cities, "SYN_CITY_"),
        "card": entity_code_table(args.n_cards, "SYN_CARD_"),
    }

    if out_root.exists() and not args.keep:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_dir = out_root / "data"
    data_dir.mkdir(exist_ok=True)

    start = pd.Timestamp("2002-09-01")
    per_month = args.rows // n_months
    extra = args.rows - per_month * n_months

    t0 = time.time()
    total = 0
    total_fraud = 0
    part_hashes: dict[str, str] = {}

    for mi in range(n_months):
        m = start + pd.DateOffset(months=mi)
        n_rows = per_month + (1 if mi < extra else 0)
        if n_rows <= 0:
            continue
        rng = np.random.default_rng(month_seeds[mi])
        feats = build_month(args, prof, state, rng, m.year, m.month, n_rows,
                            codes, cal)
        total += len(feats)
        total_fraud += int(feats["label"].sum())

        cols = features + ["label", "user_id", "year_month", "split",
                           "provenance_class"]
        feats = feats[cols]
        part = data_dir / f"year_month={m.year:04d}-{m.month:02d}"
        part.mkdir(parents=True, exist_ok=True)
        path = part / "part-0.parquet"
        table = pa.Table.from_pandas(feats, preserve_index=False)
        pq.write_table(table, path, compression=args.compression)
        part_hashes[f"{m.year:04d}-{m.month:02d}"] = sha256_file(path)

        if mi % 10 == 0 or mi == n_months - 1:
            el = time.time() - t0
            print(f"[{mi+1:3}/{n_months}] {total:,} rows  "
                  f"fraud={total_fraud:,}  {el:.0f}s", flush=True)

    # ---- schema + manifest ------------------------------------------------
    schema_str = ",".join(f"{c}:{t}" for c, t in
                          zip(cols, [str(table.schema.field(c).type) for c in cols]))
    schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "git_sha": os.popen("git rev-parse HEAD").read().strip(),
        "seed": args.seed,
        "source_dataset": args.source,
        "source_dataset_sha256": sha256_file(args.source),
        "profile_sha256": sha256_file(args.profile),
        "target_rows": args.rows,
        "actual_rows": total,
        "fraud_rows": total_fraud,
        "fraud_prevalence": (total_fraud / total) if total else None,
        "target_prevalence": TARGET_PREVALENCE,
        "schema": schema_str,
        "schema_hash": schema_hash,
        "n_features": len(features),
        "entities": {"users": args.n_users, "merchants": args.n_merchants,
                     "cities": args.n_cities, "cards": args.n_cards},
        "partitions": len(part_hashes),
        "partition_hashes": part_hashes,
        "provenance_classes": ["SYNTHETIC"],
        "feature_semantics": "runtime contract: backend/src/privacy_layer/native_features.py",
        "historical_aggregates": "strictly past-only (row's own amt/label excluded)",
        "label_mechanism": "declared logistic over generator-internal latents; no model consulted",
        "compression": args.compression,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    mpath = out_root / "manifest.json"
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nrows={total:,} fraud={total_fraud:,} "
          f"prevalence={total_fraud/total:.6f}" if total else "no rows")
    print(f"manifest -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
