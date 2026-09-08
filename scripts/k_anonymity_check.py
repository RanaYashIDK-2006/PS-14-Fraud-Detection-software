#!/usr/bin/env python3
"""k-anonymity gate for exported feature-store datasets (threat scenario D).

Rejects any release where a pseudonymous behavioral profile is uniquely
identifiable: every combination of quasi-identifier values must occur in at
least k rows drawn from at least k distinct accounts.

  python scripts/k_anonymity_check.py --csv data/transactions.csv --k 5
  python scripts/k_anonymity_check.py --csv data/feedback_labeled.csv \
      --report out/release_report.json --suppress out/safe_release.csv
  python scripts/k_anonymity_check.py --db --days 30        # live DB-2 export

Exit codes: 0 = k-anonymous (release safe), 1 = violations (REJECT the
export), 2 = input error. With --suppress, the violating rows are removed
into a k-anonymous release file (the §16-compatible fix: publish less, do
not collect more); the exit code still reflects the ORIGINAL dataset, so a
re-check of the release should be run to confirm it passes.

Quasi-identifiers default to the store's coarse, purpose-limited behavioral
attributes (txn_amount_bucket + the boolean flags); pass --qid to override.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.k_anonymity.checker import (  # noqa: E402
    ACCOUNT_COL,
    DEFAULT_QIDS,
    CheckResult,
    check_k_anonymity,
    release,
)
from src.settings import get_settings  # noqa: E402

EXIT_PASS, EXIT_REJECT, EXIT_ERROR = 0, 1, 2


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ACCOUNT_COL not in df.columns:
        raise ValueError(f"{path}: missing account column '{ACCOUNT_COL}' (not a feature-store export)")
    return df


def export_from_db(settings, days: int | None) -> pd.DataFrame:
    """Pull derived feature vectors from DB-2 (optionally the last N days)."""
    con = sqlite3.connect(settings.features_db_path)
    try:
        sql = "SELECT * FROM transaction_features"
        params: tuple = ()
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            sql += " WHERE created_at >= ?"
            params = (cutoff,)
        df = pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()
    if df.empty:
        return df
    if "is_weekend" not in df.columns:
        df["is_weekend"] = 0  # absent on very old rows; default weekday
    return df


def print_report(result: CheckResult, source: str) -> None:
    print(f"source: {source}")
    print(f"k={result.k} over {len(result.qids)} QIDs "
          f"({'distinct-accounts required' if result.require_distinct_accounts else 'row-only'})")
    print(f"rows={result.n_rows} accounts={result.n_accounts} "
          f"qid-combos={result.n_qid_combos} "
          f"(group sizes min={result.min_group_size} max={result.max_group_size})")
    for w in result.warnings:
        print(f"  ! {w}")
    if result.passed:
        print("PASS: every QID combination has >= k rows from >= k accounts")
        return
    print(f"REJECT: {result.n_violating_rows} rows in {result.n_violating_groups} "
          f"violating group(s) - uniquely identifiable profiles")
    for g in result.violating_groups[:10]:
        vals = ", ".join(f"{q}={v}" for q, v in g["qid_values"].items())
        print(f"  x {{{vals}}}  rows={g['n_rows']} accounts={g['n_distinct_accounts']} "
              f"({g['reason']})")
    if result.n_violating_groups > len(result.violating_groups):
        print(f"  ... and {result.n_violating_groups - len(result.violating_groups)} more group(s) "
              f"(use --limit to see more)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="feature-store export CSV to check")
    src.add_argument("--db", action="store_true", help="check a live DB-2 export instead")
    ap.add_argument("--days", type=int, default=None, help="with --db: only the last N days")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--qid", action="append", default=None, help="QID column (repeatable; "
                    f"default: {' '.join(DEFAULT_QIDS)})")
    ap.add_argument("--no-distinct-accounts", action="store_true",
                    help="classic row k-anonymity only (drop the >= k accounts requirement)")
    ap.add_argument("--limit", type=int, default=20, help="max violating groups in the report")
    ap.add_argument("--report", default=None, help="write a JSON compliance report here")
    ap.add_argument("--suppress", default=None,
                    help="write the suppressed k-anonymous release here (violating rows removed)")
    args = ap.parse_args(argv)

    if args.k < 1:
        print(f"k must be >= 1, got {args.k}")
        return EXIT_ERROR

    try:
        if args.csv:
            df = load_csv(ROOT / args.csv)
            source = args.csv
        else:
            df = export_from_db(get_settings(), args.days)
            source = f"db-2 export (last {args.days} days)" if args.days else "db-2 export (all)"
        result = check_k_anonymity(
            df, qids=args.qid, k=args.k,
            require_distinct_accounts=not args.no_distinct_accounts,
            detail_limit=args.limit,
        )
    except ValueError as e:
        print(f"error: {e}")
        return EXIT_ERROR

    print_report(result, source)

    if args.report:
        out = ROOT / args.report
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"report written: {out}")

    if args.suppress and not result.passed:
        safe = release(df, result)
        out = ROOT / args.suppress
        out.parent.mkdir(parents=True, exist_ok=True)
        safe.to_csv(out, index=False)
        print(f"suppressed release written: {out} "
              f"({len(safe)} of {result.n_rows} rows kept, "
              f"{result.n_rows - len(safe)} removed)")
    elif args.suppress:
        print("no suppression needed — dataset already k-anonymous")

    return EXIT_PASS if result.passed else EXIT_REJECT


if __name__ == "__main__":
    sys.exit(main())
