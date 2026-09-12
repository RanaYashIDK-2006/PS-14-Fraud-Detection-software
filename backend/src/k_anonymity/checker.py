"""k-anonymity checks for exported feature-store datasets (threat scenario D).

Threat scenario D: an attacker with auxiliary knowledge about a person's
transaction behavior (approximate amount, time of day, device, payee
pattern) matches it against a *released* pseudonymous feature-store dataset
and re-identifies the account behind a fraud_id — defeating
pseudonymization.

The mitigation checked here: the released dataset must be **k-anonymous**
over its quasi-identifiers (QIDs) — every combination of QID values must
occur at least k times, so a victim's known behavior maps to at least k
candidate rows, not one.

On top of classic row k-anonymity, the checker by default requires each
group's rows to span at least k DISTINCT accounts: k rows all belonging to
one account still reveal that account (and therefore the person), even
though they are k rows. This is the "no pseudonymous behavioral profile is
uniquely identifiable" property.

The check runs on the §16-derived attributes the store already carries
(coarse buckets and flags) — no new raw data is collected to protect it.

    python scripts/k_anonymity_check.py --csv data/transactions.csv --k 5
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Default quasi-identifiers: coarse, purpose-limited behavioral attributes
# (already generalized at the Privacy Layer). Continuous features must be
# bucketed before release — the store carries txn_amount_bucket, and
# hour_of_day is represented by the txn_time_unusual flag.
DEFAULT_QIDS = [
    "txn_amount_bucket",
    "txn_time_unusual",
    "new_device_flag",
    "unusual_location_flag",
    "unusual_recipient_flag",
    "is_weekend",
]

ACCOUNT_COL = "fraud_id"

# A QID column with more distinct values than this share of rows is
# effectively unique per row — warn (it may still pass if values repeat,
# but it is a strong linking attribute).
NEAR_UNIQUE_RATIO = 0.5


def _jsonable(v) -> object:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    return v


@dataclass
class CheckResult:
    """Outcome of a k-anonymity check on one dataset release."""

    k: int
    qids: list[str]
    require_distinct_accounts: bool
    n_rows: int
    n_accounts: int
    n_qid_combos: int
    passed: bool
    min_group_size: int
    max_group_size: int
    n_violating_groups: int
    n_violating_rows: int
    violating_groups: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return {
            "check": "k_anonymity",
            "k": self.k,
            "qids": self.qids,
            "require_distinct_accounts": self.require_distinct_accounts,
            "n_rows": self.n_rows,
            "n_accounts": self.n_accounts,
            "n_qid_combos": self.n_qid_combos,
            "passed": self.passed,
            "min_group_size": self.min_group_size,
            "max_group_size": self.max_group_size,
            "n_violating_groups": self.n_violating_groups,
            "n_violating_rows": self.n_violating_rows,
            "violating_groups": self.violating_groups,
            "warnings": self.warnings,
            "checked_at": self.checked_at,
        }


def check_k_anonymity(
    df: pd.DataFrame,
    qids: list[str] | None = None,
    k: int = 5,
    require_distinct_accounts: bool = True,
    detail_limit: int = 20,
) -> CheckResult:
    """Check that every QID combination occurs in >= k rows from >= k accounts.

    Raises ValueError for missing QID/account columns or invalid k.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    qids = list(qids or DEFAULT_QIDS)
    missing = [q for q in qids if q not in df.columns]
    if missing:
        raise ValueError(f"dataset missing quasi-identifier column(s): {missing}")
    if ACCOUNT_COL not in df.columns:
        raise ValueError(f"dataset missing account column '{ACCOUNT_COL}'")

    warnings: list[str] = []
    n_rows = int(len(df))
    if n_rows == 0:
        return CheckResult(
            k=k, qids=qids, require_distinct_accounts=require_distinct_accounts,
            n_rows=0, n_accounts=0, n_qid_combos=0, passed=True,
            min_group_size=0, max_group_size=0, n_violating_groups=0,
            n_violating_rows=0, warnings=["empty dataset: nothing identifiable, but also nothing released"],
        )

    # Near-unique QID columns are strong linking attributes - surface them.
    for q in qids:
        nunique = int(df[q].nunique(dropna=False))
        if nunique >= NEAR_UNIQUE_RATIO * n_rows:
            warnings.append(
                f"QID '{q}' is near-unique ({nunique}/{n_rows} distinct values) - "
                "a strong quasi-identifier; consider generalizing before release"
            )

    grp = df.groupby(qids, dropna=False)
    sizes = grp.size()
    n_accounts_in = grp[ACCOUNT_COL].nunique()

    violating_groups: list[dict] = []
    n_violating_rows = 0
    for combo, size in sizes.items():
        key = combo
        if not isinstance(combo, tuple):
            combo = (combo,)
        na = int(n_accounts_in[key])
        violates = size < k or (require_distinct_accounts and na < k)
        if violates:
            n_violating_rows += int(size)
            if len(violating_groups) < detail_limit:
                violating_groups.append({
                    "qid_values": {q: _jsonable(v) for q, v in zip(qids, combo)},
                    "n_rows": int(size),
                    "n_distinct_accounts": na,
                    "reason": ("fewer than k rows" if size < k else "rows from fewer than k accounts"),
                })

    return CheckResult(
        k=k, qids=qids, require_distinct_accounts=require_distinct_accounts,
        n_rows=n_rows, n_accounts=int(df[ACCOUNT_COL].nunique()),
        n_qid_combos=int(len(sizes)),
        passed=n_violating_rows == 0,
        min_group_size=int(sizes.min()), max_group_size=int(sizes.max()),
        n_violating_groups=len(violating_groups),
        n_violating_rows=n_violating_rows,
        violating_groups=violating_groups,
        warnings=warnings,
    )


def suppression_mask(df: pd.DataFrame, result: CheckResult) -> np.ndarray:
    """Boolean mask of rows that are NOT in a violating group.

    `df[result.mask]` is the k-anonymous release (violating rows removed,
    the §16-compatible way to publish safely rather than collect more data).
    """
    grp = df.groupby(result.qids, dropna=False)
    sizes = grp.size()
    n_accounts_in = grp[ACCOUNT_COL].nunique()
    violating: set[tuple] = set()
    for combo, size in sizes.items():
        key = combo
        if not isinstance(combo, tuple):
            combo = (combo,)
        na = int(n_accounts_in[key])
        if size < result.k or (result.require_distinct_accounts and na < result.k):
            violating.add(combo)
    keys = df[result.qids].apply(lambda r: tuple(r.values), axis=1)
    return ~keys.isin(violating).to_numpy()


def release(df: pd.DataFrame, result: CheckResult) -> pd.DataFrame:
    """The suppressed, k-anonymous release (violating rows removed)."""
    return df.loc[suppression_mask(df, result)].copy()
