#!/usr/bin/env python3
"""Smoke test for the threat-scenario-D k-anonymity checker.

Checks the core math (passing release, uniquely identifiable profile,
distinct-account requirement, near-unique-QID warning, input guards) and the
CLI gate (exit codes 0/1/2, JSON report, suppressed k-anonymous release,
live DB-2 export mode).

Run from the project root:
  python scripts/k_anonymity_test.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = tempfile.mkdtemp(prefix="ps14-kanon-")
os.environ["DB_DIR"] = TMP
os.environ["JWT_SECRET"] = "kanon-test-secret-0123456789abcdef"

import pandas as pd  # noqa: E402

from scripts import k_anonymity_check as cli  # noqa: E402
from src.k_anonymity.checker import (  # noqa: E402
    DEFAULT_QIDS,
    check_k_anonymity,
    release,
)
from src.settings import get_settings  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


# Four QID combinations (over the default 6 QIDs) used across accounts.
_COMBOS = [
    ("typical", 0, 0, 0, 0, 0),
    ("high_relative_to_avg", 1, 0, 0, 0, 1),
    ("low_relative_to_avg", 0, 1, 0, 0, 0),
    ("extreme_relative_to_avg", 1, 1, 1, 1, 0),
]


def combo_row(c: tuple, **kw) -> dict:
    """QID combo tuple -> row dict (optionally overridden fields)."""
    row = {"txn_amount_bucket": c[0], "txn_time_unusual": c[1], "new_device_flag": c[2],
           "unusual_location_flag": c[3], "unusual_recipient_flag": c[4], "is_weekend": c[5]}
    row.update(kw)
    return row


def make_df(n_accounts: int = 8, events_per: int = 6, extra: list[dict] | None = None) -> pd.DataFrame:
    """Every base combo appears in >= 5 rows from >= 5 accounts (k=5 pass)."""
    rows: list[dict] = []
    for a in range(n_accounts):
        for i in range(events_per):
            c = _COMBOS[i % len(_COMBOS)]
            rows.append({"event_id": f"e{a:03d}-{i}", "fraud_id": f"ACCT{a:03d}",
                         **combo_row(c)})
    if extra:
        rows.extend(extra)
    return pd.DataFrame(rows)


def main() -> int:
    print("== Threat-D k-anonymity smoke test ===")

    # ---- core: passing release --------------------------------------------
    df = make_df()
    r = check_k_anonymity(df, k=5)
    check("passing release accepted (k=5)", r.passed, f"{r.n_violating_rows} violating rows")
    check("group stats sane", r.n_rows == 48 and r.n_accounts == 8 and r.max_group_size >= 5,
          f"{r.n_rows}/{r.n_accounts}/{r.max_group_size}")
    check("k=1 always passes", check_k_anonymity(df, k=1).passed)

    # ---- core: uniquely identifiable profile ------------------------------
    unique = {"event_id": "e-UNIQ", "fraud_id": "ACCT999",
              "txn_amount_bucket": "typical", "txn_time_unusual": 1,
              "new_device_flag": 1, "unusual_location_flag": 1,
              "unusual_recipient_flag": 1, "is_weekend": 1}
    dfu = make_df(extra=[unique])
    ru = check_k_anonymity(dfu, k=5)
    check("unique profile rejected", not ru.passed and ru.n_violating_rows == 1, str(ru.n_violating_rows))
    check("violating group carries QID values + reason",
          ru.violating_groups and ru.violating_groups[0]["reason"] == "fewer than k rows"
          and ru.violating_groups[0]["qid_values"]["new_device_flag"] == 1,
          json.dumps(ru.violating_groups[0] if ru.violating_groups else {}))

    # ---- core: k rows but one account still reveals the account -----------
    # A combo NOT in the base set, so its only rows are 5 from one account.
    same_acct = [combo_row(("typical", 0, 0, 0, 0, 1), event_id=f"dup-{i}",
                           fraud_id="ACCT000") for i in range(5)]
    dfs = make_df(extra=same_acct)
    rs = check_k_anonymity(dfs, k=5)
    check("5 rows / 1 account rejected (profile identifiable)", not rs.passed,
          str(rs.violating_groups[0]["reason"] if rs.violating_groups else ""))
    rs_rowonly = check_k_anonymity(dfs, k=5, require_distinct_accounts=False)
    check("row-only k-anonymity accepts same-account group", rs_rowonly.passed)
    check("distinct-account reason reported",
          any(g["reason"] == "rows from fewer than k accounts" for g in rs.violating_groups))

    # ---- core: near-unique QID warning ------------------------------------
    dfw = df.copy()
    dfw["hour_of_day"] = list(range(len(dfw)))  # fully unique column
    rw = check_k_anonymity(dfw, qids=DEFAULT_QIDS + ["hour_of_day"], k=5)
    check("near-unique QID warns", any("near-unique" in w for w in rw.warnings),
          str(rw.warnings))

    # ---- core: guards ------------------------------------------------------
    try:
        check_k_anonymity(df, qids=["not_a_column"])
        check("missing QID column raises", False)
    except ValueError:
        check("missing QID column raises", True)
    try:
        check_k_anonymity(df, k=0)
        check("k<1 raises", False)
    except ValueError:
        check("k<1 raises", True)
    re_ = check_k_anonymity(df.head(0), k=5)
    check("empty dataset -> pass with warning", re_.passed and bool(re_.warnings))

    # ---- suppression -------------------------------------------------------
    rel = release(dfu, ru)
    check("suppression removes only violating rows", len(rel) == len(dfu) - 1, str(len(rel)))
    check("suppressed release passes a fresh check", check_k_anonymity(rel, k=5).passed)

    # ---- CLI gate ----------------------------------------------------------
    td = Path(tempfile.mkdtemp(prefix="ps14-kanon-cli-"))
    df.to_csv(td / "ok.csv", index=False)
    dfu.to_csv(td / "bad.csv", index=False)
    dfs.to_csv(td / "same_acct.csv", index=False)

    rc = cli.main(["--csv", str(td / "ok.csv"), "--k", "5"])
    check("CLI: safe export exits 0", rc == 0, f"rc={rc}")
    rc = cli.main(["--csv", str(td / "bad.csv"), "--k", "5"])
    check("CLI: unique profile exits 1 (REJECT)", rc == 1, f"rc={rc}")
    rc = cli.main(["--csv", str(td / "same_acct.csv"), "--k", "5"])
    check("CLI: same-account group exits 1", rc == 1, f"rc={rc}")
    rc = cli.main(["--csv", str(td / "same_acct.csv"), "--k", "5", "--no-distinct-accounts"])
    check("CLI: --no-distinct-accounts accepts", rc == 0, f"rc={rc}")

    report = td / "report.json"
    rc = cli.main(["--csv", str(td / "bad.csv"), "--k", "5", "--report", str(report)])
    rep = json.loads(report.read_text())
    check("CLI: JSON report written", rc == 1 and rep["passed"] is False
          and rep["check"] == "k_anonymity" and rep["n_violating_rows"] == 1,
          f"rc={rc} {rep.get('n_violating_rows')}")

    safe = td / "safe.csv"
    rc = cli.main(["--csv", str(td / "bad.csv"), "--k", "5", "--suppress", str(safe)])
    kept = pd.read_csv(safe)
    check("CLI: suppressed release written + passes re-check",
          rc == 1 and len(kept) == len(dfu) - 1
          and check_k_anonymity(kept, k=5).passed, f"rc={rc} kept={len(kept)}")

    (td / "no_acct.csv").write_text("a,b\n1,2\n3,4\n")
    rc = cli.main(["--csv", str(td / "no_acct.csv")])
    check("CLI: non-export CSV exits 2", rc == 2, f"rc={rc}")

    # ---- live DB-2 export mode --------------------------------------------
    con = sqlite3.connect(get_settings().features_db_path)
    con.execute("""CREATE TABLE transaction_features (
        event_id TEXT PRIMARY KEY, fraud_id TEXT, amount_ratio REAL,
        txn_amount_bucket TEXT, txn_freq_last_24h INTEGER, txn_time_unusual INTEGER,
        new_device_flag INTEGER, unusual_location_flag INTEGER,
        unusual_recipient_flag INTEGER, failed_auth_count_24h INTEGER,
        days_since_last_similar_txn REAL, gradual_escalation_score REAL,
        known_device_count INTEGER, account_tenure_days REAL,
        hour_of_day INTEGER, is_weekend INTEGER, created_at TEXT)""")
    for a in range(5):
        for i, c in enumerate(_COMBOS[:2]):  # 5 accounts x 2 combos = k=5 rows/account
            con.execute(
                "INSERT INTO transaction_features (event_id, fraud_id, amount_ratio, "
                "txn_amount_bucket, txn_freq_last_24h, txn_time_unusual, new_device_flag, "
                "unusual_location_flag, unusual_recipient_flag, failed_auth_count_24h, "
                "days_since_last_similar_txn, gradual_escalation_score, known_device_count, "
                "account_tenure_days, hour_of_day, is_weekend, created_at) "
                "VALUES (?, ?, 1.0, ?, 1, ?, ?, ?, ?, 0, 2.0, 0.0, 2, 60.0, 12, ?, ?)",
                (f"db-{a}-{i}", f"ACCT{a:03d}", c[0], c[1], c[2], c[3], c[4], c[5],
                 datetime_now()))
    con.commit()
    con.close()

    exported = cli.export_from_db(get_settings(), days=None)
    check("DB export reads stored vectors", len(exported) == 10, str(len(exported)))
    rc = cli.main(["--db"])
    check("CLI: DB-2 export passes k=5", rc == 0, f"rc={rc}")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


def datetime_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
