#!/usr/bin/env python3
"""Section-6 drift monitoring CLI (population stability index).

Detects week-over-week feature-distribution drift against the distribution
the models were trained on. The training baseline is stored once as bin
edges + expected proportions (never the raw training rows), and each check
compares only the current window's values — section 6: drift *triggers* a
retraining review, it does not require retaining raw data.

Commands:

  python scripts/drift_monitor.py build-baseline \
      [--input data/transactions.csv] [--out data/drift_baseline.json]

  python scripts/drift_monitor.py check \
      [--baseline data/drift_baseline.json] [--days 7]
      [--window-csv path.csv]        # optional explicit window file
      # default: last 7 days from DB-2 transaction_features (live path)

Exit codes: 0 = ok, 1 = warn (PSI >= 0.10), 2 = alert (PSI >= 0.25),
3 = insufficient window data. An alert-level breach automatically appends a
hash-chained `drift_alert` event to the DB-4 audit trail (section 13).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.drift_monitor import psi  # noqa: E402
from src.privacy_layer.features import ML_FEATURES  # noqa: E402
from src.settings import get_settings  # noqa: E402

DEFAULT_BASELINE = ROOT / "data" / "drift_baseline.json"

# A window smaller than this cannot support a meaningful 10-bin comparison.
MIN_WINDOW_EVENTS = 30

# Exit codes (stable contract for cron/CI).
EXIT_OK, EXIT_WARN, EXIT_ALERT, EXIT_INSUFFICIENT = 0, 1, 2, 3

# Pseudonymous actor for population-level (non-account) audit events.
POPULATION_ACTOR = "POPULATION-LEVEL"


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def build_baseline(df: pd.DataFrame, features: list[str] | None = None,
                   n_bins: int = 10) -> dict:
    features = features or ML_FEATURES
    bl = psi.build_baseline(df, features, n_bins=n_bins)
    bl["meta"] = {
        "n_events": int(len(df)),
        "date_min": str(df["ts"].min()) if "ts" in df.columns else None,
        "date_max": str(df["ts"].max()) if "ts" in df.columns else None,
        "features": features,
        "bins": n_bins,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return bl


# ---------------------------------------------------------------------------
# Window loading
# ---------------------------------------------------------------------------

def load_window_csv(path: Path, days: int) -> pd.DataFrame:
    """Read a window CSV. If it carries a time column, keep only the last
    `days` days relative to the newest row; otherwise all rows are the
    window (deterministic for tests/offline batches)."""
    df = pd.read_csv(path)
    missing = [c for c in ML_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"window {path} missing feature columns: {missing}")
    ts_col = next((c for c in ("ts", "created_at") if c in df.columns), None)
    if ts_col is not None and len(df) > 0:
        ts = pd.to_datetime(df[ts_col], format="mixed", errors="coerce")
        newest = ts.max()
        df = df[ts >= newest - pd.Timedelta(days=days)].copy()
    return df[ML_FEATURES].copy()


def load_window_db(settings, days: int) -> pd.DataFrame:
    """Live path: feature vectors ingested by the Privacy Layer in the last
    `days` days (DB-2 transaction_features)."""
    con = sqlite3.connect(settings.features_db_path)
    try:
        cols = ", ".join(ML_FEATURES)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = con.execute(
            f"SELECT {cols}, created_at FROM transaction_features "
            "WHERE created_at >= ? ORDER BY created_at",
            (cutoff,),
        ).fetchall()
    finally:
        con.close()
    return pd.DataFrame(rows, columns=ML_FEATURES + ["created_at"])


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def compute_report(df_window: pd.DataFrame, baseline: dict,
                   features: list[str] | None = None) -> tuple[dict, str, float]:
    """Per-feature PSI, overall level, and max PSI for a window."""
    features = features or ML_FEATURES
    per = psi.check_window(baseline, df_window, features)
    levels = [r.get("level") for r in per.values()]
    overall = "alert" if "alert" in levels else ("warn" if "warn" in levels else "ok")
    max_psi = max((r.get("psi") or 0.0) for r in per.values())
    return per, overall, float(max_psi)


def _alert_payload(per: dict, df_window: pd.DataFrame, days: int) -> dict:
    """Pseudonymous alert payload — feature-level PSI only, no raw data."""
    alerting = {f: r["psi"] for f, r in per.items() if r.get("level") == "alert"}
    ts_col = next((c for c in ("ts", "created_at") if c in df_window.columns), None)
    return {
        "monitor": "psi",
        "window_days": days,
        "window_start": str(df_window[ts_col].min()) if ts_col else None,
        "window_end": str(df_window[ts_col].max()) if ts_col else None,
        "n_events": int(len(df_window)),
        "thresholds": {"warn": psi.PSI_WARN, "alert": psi.PSI_ALERT},
        "max_psi": round(max((r.get("psi") or 0.0) for r in per.values()), 4),
        "features_alerting": alerting,
        "action": "retraining_review_per_section_6",
    }


def run_check(df_window: pd.DataFrame, baseline: dict, days: int = 7,
              emit_alert: bool = True) -> int:
    """Run a window check; append a hash-chained drift_alert on breach."""
    if len(df_window) < MIN_WINDOW_EVENTS:
        print(f"insufficient window data: {len(df_window)} events "
              f"(need >= {MIN_WINDOW_EVENTS}) — check skipped")
        return EXIT_INSUFFICIENT

    per, overall, max_psi = compute_report(df_window, baseline)
    print(f"window: {len(df_window)} events, last {days} days")
    print(f"{'feature':<28}{'psi':>10}  level")
    for f, r in per.items():
        psi_v = r.get("psi")
        print(f"{f:<28}{psi_v if psi_v is not None else float('nan'):>10.4f}  {r.get('level')}")
    print(f"overall: {overall.upper()} (max PSI {max_psi:.4f}; "
          f"warn >= {psi.PSI_WARN}, alert >= {psi.PSI_ALERT})")

    if overall == "alert" and emit_alert:
        from src.audit_service.writer import append_audit_event
        ev = append_audit_event(
            fraud_id=POPULATION_ACTOR,
            event_type="drift_alert",
            payload=_alert_payload(per, df_window, days),
        )
        print(f"[ALERT] drift_alert appended to audit chain (DB-4): "
              f"seq {ev.seq} entry {ev.entry_hash[:12]}…")
        return EXIT_ALERT
    return EXIT_OK if overall == "ok" else EXIT_WARN


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-baseline", help="build training-distribution baseline")
    b.add_argument("--input", default="data/transactions.csv")
    b.add_argument("--out", default=str(DEFAULT_BASELINE))
    b.add_argument("--bins", type=int, default=10)

    c = sub.add_parser("check", help="check the current window against the baseline")
    c.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    c.add_argument("--days", type=int, default=7)
    c.add_argument("--window-csv", default=None,
                   help="explicit window file (default: last N days from DB-2)")
    args = ap.parse_args(argv)

    if args.cmd == "build-baseline":
        src = ROOT / args.input
        df = pd.read_csv(src)
        bl = build_baseline(df, n_bins=args.bins)
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(bl, indent=2))
        n_const = sum(1 for k, v in bl.items() if k != "meta" and v.get("constant"))
        print(f"baseline written: {out} ({bl['meta']['n_events']} events, "
              f"{len(ML_FEATURES) - n_const} binned features, {n_const} constant)")
        return 0

    # check
    baseline_path = ROOT / args.baseline
    if not baseline_path.exists():
        print(f"baseline not found: {baseline_path} (run build-baseline first)")
        return 3
    baseline = json.loads(baseline_path.read_text())

    settings = get_settings()
    if args.window_csv:
        df_window = load_window_csv(ROOT / args.window_csv, args.days)
    else:
        df_window = load_window_db(settings, args.days)
    return run_check(df_window, baseline, days=args.days)


if __name__ == "__main__":
    sys.exit(main())
