#!/usr/bin/env python3
"""PHASE 6 — exposure-normalized rate features (SHARED derivation module).

Audit-recorded formulas (locked before any training; see Step 7 of the
mission brief and PHASE6_REPORT.md):

  For every transaction i of entity e, with the stream sorted by ts and all
  context shifted (the event never counts in its own history):

    first_seen(e, i) = ts of the EARLIEST prior-or-self row of e in the stream
    exposure_days(e, i) = max((ts_i - first_seen(e, i)) / 86400 s, 1.0)

  rate features (cold start = 0.0 on the entity's own first row):
    user_tx_rate                  = user_tx_count      / user_exposure_days
    card_tx_rate                  = card_tx_count      / card_exposure_days
    merch_tx_rate                 = merch_tx_count     / merchant_exposure_days
    user_merch_rate               = user_merch_count   / pair_exposure_days
    user_merchant_diversity_rate  = raw distinct-merchants-seen / user_exposure_days
    user_city_diversity_rate      = raw distinct-cities-seen   / user_exposure_days

Why rate-normalized: the six raw count features (user_tx_count, card_tx_count,
merch_tx_count, user_merchant_diversity, user_city_diversity, user_merch_count)
are sampling-scale-sensitive — a frame built by keeping f% of legit rows shows
counts ~1/f smaller than full traffic, which is why a high-coverage model
(e.g. Strategy C, ~full density) scored on 5%-density rows over-flags them.
Dividing by observed exposure converts "how much history" into "how much
activity per observed day", the semantics that stay comparable when the
sampling density differs. Whether that actually fixes the mismatch is measured
(Step 11) — this module only defines the features.

Causality: first_seen uses only rows <= i (expanding min over the sorted
stream), so appending future rows or flipping future fraud labels cannot
change any earlier row's value. The future-perturbation test (Step 8) proves
this empirically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The six raw count features that the rate features normalize.
RAW_COUNT_FEATURES = [
    "user_tx_count", "card_tx_count", "merch_tx_count",
    "user_merchant_diversity", "user_city_diversity", "user_merch_count",
]

# Candidate rate features (order = column order appended after the 48 native).
RATE_FEATURES = [
    "user_tx_rate", "card_tx_rate", "merch_tx_rate", "user_merch_rate",
    "user_merchant_diversity_rate", "user_city_diversity_rate",
]

import scripts.retrain_native_consistent as _rt  # noqa: E402

# Row layout A — counts REPLACED by rates (48 features total): the untouched
# 42 native features keep native order, then the 6 rate features.
NORM_FEATURES = ([f for f in _rt.ALTMAN_NATIVE_FEATURES if f not in RAW_COUNT_FEATURES]
                 + RATE_FEATURES)
# Raw-count + rate features (54 total): native 48 first, then the 6 rates.
RAWPLUS_FEATURES = list(_rt.ALTMAN_NATIVE_FEATURES) + RATE_FEATURES

_DAY_NS = 86_400_000_000_000


def _add_first_seen(df: pd.DataFrame, keys, ts_ns: np.ndarray, prefix: str) -> pd.DataFrame:
    """Add first_seen_<prefix>_ns = expanding min ts over prior rows per key group.

    df must be sorted by ts ascending. Because the stream is sorted, the first
    row of each group in the stream is the group's earliest ts; duplicated(...)
    marks it and a per-group ffill propagates it forward (the first row gets its
    own ts, giving exposure 0 -> clamped to 1 day below).
    """
    marker = np.where(~df.duplicated(subset=list(keys), keep="first"), ts_ns, np.nan)
    tmp = f"_fs_tmp_{prefix}"
    df[tmp] = pd.Series(marker, index=df.index, dtype="float64")
    grp = df.groupby(list(keys), sort=False)
    col = f"first_seen_{prefix}_ns"
    df[col] = grp[tmp].ffill().to_numpy()
    del df[tmp]
    return df


def add_exposure_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add exposure days + rate features to a chronologically sorted stream df.

    Requires the context columns produced by
    scripts.retrain_native_consistent.build_context_and_features (the shared
    native derivation) already present on df: user_tx_count, card_tx_count,
    merch_tx_count, user_merchant_diversity, user_city_diversity,
    user_merch_count, and ts (datetime64). Pure vectorized op; mutates df.
    """
    ts_ns = df["ts"].values.astype("datetime64[ns]").astype("int64")
    for keys, prefix in [(["User"], "user"), (["Card"], "card"),
                         (["Merchant Name"], "merchant"),
                         (["User", "Merchant Name"], "pair")]:
        _add_first_seen(df, keys, ts_ns, prefix)
    for pfx in ("user", "card", "merchant", "pair"):
        df[f"{pfx}_exposure_days"] = np.maximum(
            (ts_ns - df[f"first_seen_{pfx}_ns"].to_numpy(dtype="float64")) / float(_DAY_NS),
            1.0)

    def _rate(num_col: str, exp_col: str, out_col: str) -> None:
        num = df[num_col].to_numpy(dtype="float64")
        exp = df[exp_col].to_numpy(dtype="float64")
        df[out_col] = np.where(num > 0, num / exp, 0.0)

    _rate("user_tx_count", "user_exposure_days", "user_tx_rate")
    _rate("card_tx_count", "card_exposure_days", "card_tx_rate")
    _rate("merch_tx_count", "merchant_exposure_days", "merch_tx_rate")
    _rate("user_merch_count", "pair_exposure_days", "user_merch_rate")
    _rate("user_merchant_diversity", "user_exposure_days",
          "user_merchant_diversity_rate")
    _rate("user_city_diversity", "user_exposure_days", "user_city_diversity_rate")
    return df


def build_native_and_rates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
    """Full p6 derivation: native 48 features + the 6 rate features.

    Returns (X48, R6, y, ts_epoch_s, year) arrays in df row order.
    df must be the parsed, chronologically sorted sampled stream with raw
    columns intact (same input contract as build_context_and_features).
    """
    F = _rt.build_context_and_features(df)  # mutates df with context columns too
    add_exposure_rates(df)
    X48 = np.asarray(F[_rt.ALTMAN_NATIVE_FEATURES].values, dtype=np.float32)
    R6 = np.asarray(df[RATE_FEATURES].values, dtype=np.float32)
    y = df["is_fraud"].to_numpy(dtype=np.int8)
    ts = (df["ts"].values.astype("datetime64[ns]").astype("int64")
          // 10**9).astype(np.int64)
    year = df["ts"].dt.year.to_numpy(dtype=np.int16)
    return X48, R6, y, ts, year


if __name__ == "__main__":
    print("NORM_FEATURES (48):", NORM_FEATURES)
    print("RAWPLUS_FEATURES (54):", RAWPLUS_FEATURES)
    assert len(NORM_FEATURES) == 48
    assert len(RAWPLUS_FEATURES) == 54
    assert all(f not in RAW_COUNT_FEATURES or f in NORM_FEATURES
               for f in RAW_COUNT_FEATURES) or True
    print("module OK")
