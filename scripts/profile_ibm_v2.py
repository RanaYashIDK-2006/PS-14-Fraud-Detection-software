"""Streaming empirical profile of the IBM v2 transaction dataset.

Produces `misc/reports/ibm_v2_empirical_profile.json`, the statistical
foundation for the 50M synthetic benchmark specification (§1, §3, §5).

Read-only. Never writes to the source dataset. Streams the 2.3 GB CSV in
chunks so it fits in RAM.

Usage:
    ./.venv/Scripts/python.exe scripts/profile_ibm_v2.py [--rows N] [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

SRC = "data/credit_card_transactions-ibm_v2.csv"
OUT = "misc/reports/ibm_v2_empirical_profile.json"

COLUMNS = [
    "User", "Card", "Year", "Month", "Day", "Time", "Amount",
    "Use Chip", "Merchant Name", "Merchant City", "Merchant State", "Zip",
    "MCC", "Errors?", "Is Fraud?",
]


def parse_amount(s: pd.Series) -> pd.Series:
    """'$134.09' -> 134.09 ; NaN/blank -> NaN."""
    return pd.to_numeric(
        s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rows", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f"FATAL: source dataset not found: {args.src}")
        return 1

    # ---- source identity (§8: source dataset hash) --------------------
    h = hashlib.sha256()
    with open(args.src, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    src_sha = h.hexdigest()

    n_rows = 0
    n_fraud = 0
    users: set = set()
    merchants: set = set()
    cities: set = set()
    states: Counter = Counter()
    chip: Counter = Counter()
    mcc: Counter = Counter()
    errors: Counter = Counter()
    fraud_flag: Counter = Counter()
    year: Counter = Counter()
    hour: Counter = Counter()
    dow = Counter()
    amt_sum = 0.0
    amt_sq = 0.0
    amt_n = 0
    amt_vals: list[float] = []
    log_amt_sum = 0.0
    zip_missing = 0
    state_missing = 0
    # joint: mean amount + fraud rate by chip type and by mcc group
    j_chip_amt: dict = {}
    j_mcc_fraud: dict = {}

    reader = pd.read_csv(
        args.src,
        chunksize=500_000,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    for chunk in reader:
        if args.rows and n_rows >= args.rows:
            break
        if args.rows and n_rows + len(chunk) > args.rows:
            chunk = chunk.iloc[: args.rows - n_rows]

        n_rows += len(chunk)

        fraud_flag.update(chunk["Is Fraud?"].value_counts().to_dict())
        is_fraud = (chunk["Is Fraud?"].str.strip() == "Yes")
        n_fraud += int(is_fraud.sum())

        users.update(chunk["User"].unique().tolist())
        merchants.update(chunk["Merchant Name"].unique().tolist())
        cities.update(chunk["Merchant City"].unique().tolist())
        states.update(chunk["Merchant State"].value_counts().to_dict())
        chip.update(chunk["Use Chip"].value_counts().to_dict())
        errors.update(chunk["Errors?"].value_counts().to_dict())
        year.update(chunk["Year"].value_counts().to_dict())

        zip_missing += int((chunk["Zip"].str.strip() == "").sum())
        state_missing += int((chunk["Merchant State"].str.strip() == "").sum())

        amt = parse_amount(chunk["Amount"])
        ok = amt.notna()
        a = amt[ok].to_numpy(dtype=np.float64)
        amt_n += a.size
        if a.size:
            amt_sum += float(a.sum())
            amt_sq += float((a * a).sum())
            if len(amt_vals) < 2_000_000:  # bounded reservoir for quantiles
                amt_vals.extend(a[: 2_000_000 - len(amt_vals)].tolist())

        # hour / dow from Time + Year/Month/Day
        hh = pd.to_numeric(chunk["Time"].str.slice(0, 2), errors="coerce")
        hour.update(hh.value_counts().to_dict())
        dt = pd.to_datetime(
            dict(
                year=pd.to_numeric(chunk["Year"], errors="coerce"),
                month=pd.to_numeric(chunk["Month"], errors="coerce"),
                day=pd.to_numeric(chunk["Day"], errors="coerce"),
            ),
            errors="coerce",
        )
        dow.update(dt.dt.dayofweek.value_counts().to_dict())

        mcc_v = chunk["MCC"].str.strip()
        mcc.update(mcc_v.value_counts().to_dict())
        for k, v in mcc_v.value_counts().items():
            j_mcc_fraud.setdefault(k, [0, 0])
        grp = pd.DataFrame({"mcc": mcc_v, "fraud": is_fraud.to_numpy()})
        for k, sub in grp.groupby("mcc"):
            j_mcc_fraud.setdefault(k, [0, 0])
            j_mcc_fraud[k][0] += int(sub["fraud"].sum())
            j_mcc_fraud[k][1] += len(sub)

        chip_v = chunk["Use Chip"].str.strip()
        grp2 = pd.DataFrame({"c": chip_v, "a": amt.to_numpy(), "f": is_fraud.to_numpy()})
        for k, sub in grp2.groupby("c"):
            e = j_chip_amt.setdefault(k, [0, 0.0, 0])
            e[0] += int(sub["f"].sum())
            e[1] += float(np.nansum(sub["a"].to_numpy(dtype=np.float64)))
            e[2] += int(sub["a"].notna().sum())

    av = np.asarray(amt_vals, dtype=np.float64)
    qs = np.percentile(av, [1, 5, 25, 50, 75, 90, 95, 99, 99.9]) if av.size else []

    profile = {
        "source": {
            "path": args.src,
            "sha256": src_sha,
            "columns": COLUMNS,
            "n_columns": len(COLUMNS),
        },
        "volume": {
            "n_rows": n_rows,
            "n_fraud": n_fraud,
            "fraud_prevalence": (n_fraud / n_rows) if n_rows else None,
        },
        "entities": {
            "n_users": len(users),
            "n_merchants": len(merchants),
            "n_cities": len(cities),
            "n_states": len(states),
            "rows_per_user": (n_rows / len(users)) if users else None,
            "rows_per_merchant": (n_rows / len(merchants)) if merchants else None,
        },
        "amount": {
            "n": amt_n,
            "mean": (amt_sum / amt_n) if amt_n else None,
            "std": float(np.sqrt(max(amt_sq / amt_n - (amt_sum / amt_n) ** 2, 0.0))) if amt_n else None,
            "quantiles": {str(q): float(v) for q, v in zip(
                ["p1", "p5", "p25", "p50", "p75", "p90", "p95", "p99", "p99.9"], qs)},
            "share_negative_or_zero": None,
        },
        "temporal": {
            "years": dict(sorted(year.items())),
            "hours": {str(k): v for k, v in sorted(hour.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else -1)},
            "dow": {str(k): v for k, v in sorted(dow.items())},
        },
        "categorical": {
            "use_chip": dict(chip.most_common()),
            "errors": dict(errors.most_common()),
            "is_fraud": dict(fraud_flag.most_common()),
            "mcc_top20": dict(mcc.most_common(20)),
            "n_distinct_mcc": len(mcc),
        },
        "missingness": {
            "zip_blank": zip_missing,
            "zip_blank_rate": (zip_missing / n_rows) if n_rows else None,
            "state_blank": state_missing,
            "state_blank_rate": (state_missing / n_rows) if n_rows else None,
        },
        "joint": {
            "amount_and_fraud_by_chip": {
                k: {
                    "n_with_amount": v[2],
                    "mean_amount": (v[1] / v[2]) if v[2] else None,
                    "fraud": v[0],
                }
                for k, v in j_chip_amt.items()
            },
            "fraud_rate_by_mcc_top20": {
                k: {"n": v[1], "fraud": v[0], "rate": (v[0] / v[1]) if v[1] else None}
                for k, v in sorted(
                    j_mcc_fraud.items(), key=lambda x: -x[1][1]
                )[:20]
            },
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, default=str)
    print(f"wrote {args.out}")
    print(f"rows={n_rows:,} fraud={n_fraud:,} "
          f"prevalence={(n_fraud / n_rows):.6f}" if n_rows else "no rows")
    print(f"users={len(users):,} merchants={len(merchants):,} cities={len(cities):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
