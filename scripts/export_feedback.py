#!/usr/bin/env python3
"""Export verification outcomes into labeled training rows (feedback loop,
architecture section 7: "verification outcome -> labeled data -> retraining
queue").

Reads DB-3 `verification_outcomes` (outcome, event_id, fraud_id) and joins
each to its stored derived feature vector in DB-2 `transaction_features`,
then writes the labeled pool in the trainer's schema:

  label: confirmed -> 0 (legit), disputed -> 1 (fraud)

The pool is ACCUMULATIVE - each run rebuilds it from every verification
outcome in DB-3 (idempotent, no double counting) - and VERSIONED: every
export also writes a dated snapshot
`data/feedback_snapshots/feedback_YYYYMMDD_HHMMSS.csv` so any point in the
pool's history is reproducible (`src/train_compare.py --feedback-date`
resolves the latest snapshot at or before a date). The trainer consumes the
latest snapshot by default via `scripts/training_pipeline.py`. Rows carry
`archetype` = feedback_legit / feedback_fraud so the per-archetype recall
report shows them as their own group.

Section-6 queue trigger + gate:
  * GATE - an export is skipped (exit code 2, nothing written) while the
    number of REAL verified outcomes is below `--min-outcomes` (default
    10). The pool must be big enough to train on; `--force` bypasses.
  * TRIGGER - once the pool's real DISPUTED events reach
    `--disputed-threshold` (default 5), the export appends a hash-chained
    `retrain_trigger` event to the DB-4 audit trail (mirrors the drift
    monitor's `drift_alert`) - the signal that enough fraud-positive
    feedback has accumulated to justify a retraining round.
  * A successful export that leaves the pool unchanged writes no new
    snapshot (versions only track pool state changes).

`--simulate N` appends N rows sampled from an existing labeled CSV (default
data/transactions.csv) with archetype `feedback_sim`, standing in for a large
verification wave. This is a MECHANICS simulation - sampled from the same
distribution the models trained on, so it is circular by construction and
must not be read as evidence of real-world improvement (simulated rows never
satisfy the pool-size gate). The real signal comes from genuinely new events
whose user outcomes confirm or contradict model calls.

Run from the project root:
  python scripts/export_feedback.py                  # real pool + snapshot
  python scripts/export_feedback.py --simulate 500 --force   # mechanics wave
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.settings import get_settings  # noqa: E402

SNAPSHOT_DIR = ROOT / "data" / "feedback_snapshots"

SNAPSHOT_PREFIX = "feedback_"
SNAPSHOT_STAMP = "%Y%m%d_%H%M%S"

# Exit codes (stable contract for cron/CI): 0 = exported, 2 = gated.
EXIT_OK, EXIT_GATED = 0, 2

DEFAULT_MIN_OUTCOMES = 10        # real verified outcomes required to export
DEFAULT_DISPUTED_THRESHOLD = 5   # disputed pool events -> retrain-queue alert

# Pseudonymous actor for pool-level (non-account) audit events.
POPULATION_ACTOR = "POPULATION-LEVEL"


# Exact training schema (matches data/transactions.csv columns).
COLUMNS = [
    "event_id", "fraud_id", "ts", "hour_of_day", "is_weekend",
    "amount_ratio", "txn_amount_bucket", "txn_freq_last_24h", "txn_time_unusual",
    "new_device_flag", "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count", "account_tenure_days",
    "archetype", "label",
]

_FEAT_SQL = (
    "SELECT amount_ratio, txn_amount_bucket, txn_freq_last_24h, txn_time_unusual, "
    "new_device_flag, unusual_location_flag, unusual_recipient_flag, "
    "failed_auth_count_24h, days_since_last_similar_txn, gradual_escalation_score, "
    "known_device_count, account_tenure_days, hour_of_day, is_weekend, created_at "
    "FROM transaction_features WHERE event_id = ?"
)


def export_real(settings) -> pd.DataFrame:
    """All verification outcomes joined to their stored feature vectors."""
    con3 = sqlite3.connect(settings.risk_db_path)
    outcomes = con3.execute(
        "SELECT event_id, fraud_id, outcome FROM verification_outcomes "
        "ORDER BY resolved_at, verification_id"
    ).fetchall()
    con3.close()

    if not outcomes:
        return pd.DataFrame(columns=COLUMNS)

    con2 = sqlite3.connect(settings.features_db_path)
    rows: list[dict] = []
    seen: set[str] = set()
    for event_id, fraud_id, outcome in outcomes:
        if event_id in seen:  # latest outcome wins for a duplicated event
            continue
        seen.add(event_id)
        feat = con2.execute(_FEAT_SQL, (event_id,)).fetchone()
        if feat is None:
            print(f"  ! skipping {event_id}: no feature vector in DB-2")
            continue
        (ratio, bucket, freq, time_u, dev, loc, recip, auth,
         days_since, esc, dev_cnt, tenure, hour, weekend, created_at) = feat
        label = 0 if outcome == "confirmed" else 1
        rows.append({
            "event_id": event_id,
            "fraud_id": fraud_id,
            "ts": created_at,
            "hour_of_day": hour,
            "is_weekend": int(weekend),
            "amount_ratio": ratio,
            "txn_amount_bucket": bucket,
            "txn_freq_last_24h": freq,
            "txn_time_unusual": int(time_u),
            "new_device_flag": int(dev),
            "unusual_location_flag": int(loc),
            "unusual_recipient_flag": int(recip),
            "failed_auth_count_24h": auth,
            "days_since_last_similar_txn": days_since,
            "gradual_escalation_score": esc,
            "known_device_count": dev_cnt,
            "account_tenure_days": tenure,
            "archetype": "feedback_legit" if label == 0 else "feedback_fraud",
            "label": label,
        })
    con2.close()
    return pd.DataFrame(rows, columns=COLUMNS)


def simulate(source: Path, n: int, seed: int) -> pd.DataFrame:
    """Sample n labeled events from an existing labeled CSV as a stand-in for
    a large verification wave (archetype feedback_sim). Circular by
    construction - see module docstring."""
    src = pd.read_csv(source)
    need = [c for c in COLUMNS if c in src.columns]
    if len(need) != len(COLUMNS):
        missing = [c for c in COLUMNS if c not in src.columns]
        raise SystemExit(f"simulate source {source} missing columns: {missing}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(src), size=min(n, len(src)), replace=False)
    out = src.iloc[idx][COLUMNS].copy()
    out["archetype"] = "feedback_sim"
    return out


def latest_snapshot(snapshot_dir: Path) -> Path | None:
    """The newest dated snapshot in the directory, or None."""
    if not snapshot_dir.is_dir():
        return None
    snaps = sorted(snapshot_dir.glob(f"{SNAPSHOT_PREFIX}*.csv"))
    return snaps[-1] if snaps else None


def pool_unchanged(pool: pd.DataFrame, snapshot_dir: Path) -> Path | None:
    """Return the latest snapshot path when the pool content equals it (no
    new state to version), else None."""
    latest = latest_snapshot(snapshot_dir)
    if latest is None:
        return None
    if pool.to_csv(index=False) == latest.read_bytes().decode("utf-8"):
        return latest
    return None


def emit_retrain_trigger(real: pd.DataFrame, snapshot_tag: str,
                         disputed_threshold: int, emit: bool = True) -> dict | None:
    """Section-6 retrain-queue signal: when the pool's real disputed events
    reach `disputed_threshold`, append a hash-chained `retrain_trigger`
    event to the DB-4 audit trail (mirrors the drift monitor's
    `drift_alert`). Returns the payload, or None when not triggered or
    emission is disabled. The payload is pseudonymous - counts and
    thresholds only, never event ids or vectors."""
    if len(real) == 0:
        return None
    disputed = int((real["label"] == 1).sum())
    confirmed = int((real["label"] == 0).sum())
    if disputed < disputed_threshold:
        return None
    payload = {
        "monitor": "feedback_queue",
        "pool_rows": int(len(real)),
        "confirmed": confirmed,
        "disputed": disputed,
        "threshold": {"disputed": disputed_threshold},
        "snapshot": snapshot_tag,
        "action": "retrain_queue_per_section_6",
    }
    if emit:
        from src.audit_service.writer import append_audit_event
        ev = append_audit_event(
            fraud_id=POPULATION_ACTOR,
            event_type="retrain_trigger",
            payload=payload,
        )
        print(f"[RETRAIN QUEUE] retrain_trigger appended to audit chain (DB-4): "
              f"seq {ev.seq} entry {ev.entry_hash[:12]}… — {disputed} disputed "
              f">= threshold {disputed_threshold}")
    return payload


def write_snapshot(pool: pd.DataFrame, snapshot_dir: Path, stamp: str | None = None) -> Path:
    """Write the pool as a dated snapshot and return its path. `stamp` is an
    injectable `YYYYMMDD_HHMMSS` string (deterministic tests); defaults to
    now. The snapshot is the exact CSV the trainer consumes - content is
    identical to the pool file at export time."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime(SNAPSHOT_STAMP)
    path = snapshot_dir / f"{SNAPSHOT_PREFIX}{stamp}.csv"
    pool.to_csv(path, index=False)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulate", type=int, default=0,
                    help="append N rows sampled from --simulate-source as a simulated verification wave")
    ap.add_argument("--simulate-source", default="data/transactions.csv")
    ap.add_argument("--simulate-seed", type=int, default=42)
    ap.add_argument("--out", default="data/feedback_labeled.csv")
    ap.add_argument("--snapshot-dir", default=str(SNAPSHOT_DIR))
    ap.add_argument("--min-outcomes", type=int, default=DEFAULT_MIN_OUTCOMES,
                    help="minimum REAL verified outcomes to export (exit 2 when below)")
    ap.add_argument("--disputed-threshold", type=int, default=DEFAULT_DISPUTED_THRESHOLD,
                    help="disputed pool events that justify a retrain-queue alert")
    ap.add_argument("--force", action="store_true", help="bypass the pool-size gate")
    ap.add_argument("--no-alert", action="store_true", help="do not append the retrain_trigger audit event")
    args = ap.parse_args()

    settings = get_settings()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = ROOT / args.snapshot_dir

    real = export_real(settings)
    parts = [real]
    n_real = len(real)
    print(f"verification outcomes exported: {n_real} "
          f"({int((real['label'] == 0).sum())} confirmed / "
          f"{int((real['label'] == 1).sum())} disputed)" if n_real else
          "verification outcomes exported: 0 (no outcomes yet)")

    # ---- section-6 gate: minimum REAL pool size ---------------------------
    if n_real < args.min_outcomes and not args.force:
        print(f"[GATED] feedback export skipped: {n_real} real verified outcomes "
              f"< minimum pool size {args.min_outcomes} (exit {EXIT_GATED}) - "
              f"run again once more outcomes accumulate, or --force to export anyway")
        return EXIT_GATED

    if args.simulate:
        sim = simulate(ROOT / args.simulate_source, args.simulate, args.simulate_seed)
        parts.append(sim)
        print(f"simulated verification wave: +{len(sim)} rows "
              f"({int((sim['label'] == 0).sum())} legit / {int((sim['label'] == 1).sum())} fraud, "
              f"archetype=feedback_sim)")

    pool = pd.concat(parts, ignore_index=True)
    pool["label"] = pool["label"].astype(int)
    # Normalize timestamps to one format: sources differ in precision
    # (sqlite second-precision vs generator nanosecond-precision), and a
    # mixed column defeats pandas' CSV date parsing downstream.
    pool["ts"] = (
        pd.to_datetime(pool["ts"].astype(str), format="mixed", errors="coerce")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    # ---- versioned write: skip when the pool state did not change ---------
    unchanged = pool_unchanged(pool, snapshot_dir)
    if unchanged is not None:
        print(f"pool unchanged since snapshot {unchanged.name} - no new snapshot")
        tag = unchanged.stem[len(SNAPSHOT_PREFIX):]
    else:
        pool.to_csv(out_path, index=False)
        print(f"feedback pool written: {out_path} ({len(pool)} rows, "
              f"fraud rate {pool['label'].mean():.4f})")
        snap = write_snapshot(pool, snapshot_dir)
        tag = snap.stem[len(SNAPSHOT_PREFIX):]
        print(f"snapshot written: {snap} ({len(pool)} rows) - "
              f"reproduce via src/train_compare.py --feedback-date {tag}")

    # ---- section-6 retrain-queue trigger ----------------------------------
    emit_retrain_trigger(real, tag, args.disputed_threshold, emit=not args.no_alert)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
