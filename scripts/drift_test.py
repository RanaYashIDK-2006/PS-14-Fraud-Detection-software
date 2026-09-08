#!/usr/bin/env python3
"""Smoke test for the section-6 PSI drift monitor.

Checks PSI math (identical distributions -> 0, shifted -> alert), baseline
binning (constant features handled), the check/report pipeline, and the
automatic alert path: a drifted window exits 2 AND appends a hash-chained
`drift_alert` event to DB-4, while an unchanged window exits 0 and appends
nothing. Also guards the insufficient-data path (no false alert, exit 3).

Run from the project root:
  python scripts/drift_test.py
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

TMP = tempfile.mkdtemp(prefix="ps14-drift-")
os.environ["DB_DIR"] = TMP
os.environ["JWT_SECRET"] = "drift-test-secret-0123456789abcdef"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import drift_monitor  # noqa: E402
from src.audit_service.writer import GENESIS_HASH, flush_audit_queue, verify_chain  # noqa: E402
from src.drift_monitor import psi  # noqa: E402
from src.privacy_layer.features import ML_FEATURES  # noqa: E402
from src.settings import get_settings  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def make_df(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Realistic-shaped feature matrix over all ML features (all non-constant)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "amount_ratio": np.maximum(0.05, rng.lognormal(0, 0.5, n)),
        "txn_freq_last_24h": rng.poisson(2, n),
        "txn_time_unusual": rng.binomial(1, 0.15, n),
        "new_device_flag": rng.binomial(1, 0.08, n),
        "unusual_location_flag": rng.binomial(1, 0.10, n),
        "unusual_recipient_flag": rng.binomial(1, 0.07, n),
        "failed_auth_count_24h": rng.poisson(0.2, n),
        "days_since_last_similar_txn": np.maximum(0.2, rng.exponential(5, n)),
        "gradual_escalation_score": np.clip(rng.beta(0.5, 5, n), 0.0, 1.0),
        "known_device_count": rng.poisson(2, n) + 1,
        "account_tenure_days": np.maximum(1.0, rng.lognormal(3, 1, n)),
        "hour_of_day": rng.integers(0, 24, n),
        "is_weekend": rng.binomial(1, 0.3, n),
        "shared_device_accounts": rng.poisson(0.2, n),
        "shared_recipient_accounts": rng.poisson(0.1, n),
        "mule_ring_score": np.clip(rng.beta(0.3, 8, n), 0.0, 1.0),
        "hour_deviation": np.clip(rng.beta(1, 4, n), 0.0, 1.0),
        "amount_zscore": rng.normal(0, 1, n),
        "velocity_deviation": np.clip(rng.beta(2, 5, n), 0.0, 1.0),
        "recipient_novelty": rng.binomial(1, 0.07, n).astype(float),
        "txn_regularity": np.clip(rng.beta(2, 3, n), 0.0, 1.0),
    })


def audit_rows() -> list[tuple]:
    con = sqlite3.connect(get_settings().audit_db_path)
    rows = con.execute(
        "SELECT seq, event_type, prev_hash, entry_hash FROM audit_events ORDER BY seq"
    ).fetchall()
    con.close()
    return rows


def main() -> int:
    print("== PSI drift monitor smoke test ===")

    # ---- PSI math ----------------------------------------------------------
    x = np.linspace(0.1, 5.0, 500)
    edges = [0.1, 1.0, 2.0, 3.0, 5.0]
    counts, _ = np.histogram(x, bins=edges)
    exp = counts / counts.sum()
    p_same = psi.check_feature({"edges": edges, "expected": exp}, x)
    check("identical distribution -> psi 0 / ok", p_same["psi"] == 0.0 and p_same["level"] == "ok",
          str(p_same))
    shifted = psi.check_feature({"edges": edges, "expected": exp}, x * 10)
    check("shifted distribution -> alert", shifted["level"] == "alert", str(shifted))
    check("level thresholds (0.10 warn / 0.25 alert)",
          psi.level(0.05) == "ok" and psi.level(0.10) == "warn" and psi.level(0.25) == "alert",
          f"{psi.level(0.05)}/{psi.level(0.10)}/{psi.level(0.25)}")

    # ---- baseline binning --------------------------------------------------
    train = make_df()
    baseline = drift_monitor.build_baseline(train)
    check("baseline covers all ML features",
          all(f in baseline for f in ML_FEATURES), str(len(baseline) - 1))
    const = pd.DataFrame({"c": np.ones(50)})
    bl_const = psi.build_baseline(const, ["c"])
    check("constant feature marked constant", bl_const["c"].get("constant") is True, str(bl_const))
    r_const = psi.check_feature(bl_const["c"], np.ones(20))
    check("constant feature check -> ok", r_const["level"] == "ok", str(r_const))

    # ---- report pipeline ---------------------------------------------------
    per, overall, max_psi = drift_monitor.compute_report(train, baseline)
    check("same-data window -> overall ok", overall == "ok", f"max psi {max_psi:.4f}")

    drifted = train.copy()
    drifted["amount_ratio"] = drifted["amount_ratio"] * 5.0
    per2, overall2, max_psi2 = drift_monitor.compute_report(drifted, baseline)
    check("drifted window -> overall alert", overall2 == "alert", f"max psi {max_psi2:.4f}")
    check("alerting feature named in report",
          per2["amount_ratio"]["level"] == "alert", str(per2["amount_ratio"]))

    # ---- CLI end-to-end (temp DB-4 + files) --------------------------------
    td = Path(tempfile.mkdtemp(prefix="ps14-drift-cli-"))
    train.to_csv(td / "train.csv", index=False)
    base_out = td / "baseline.json"
    rc = drift_monitor.main(["build-baseline", "--input", str(td / "train.csv"),
                             "--out", str(base_out)])
    check("build-baseline exits 0", rc == 0)
    bl_file = json.loads(base_out.read_text())
    check("baseline json written with meta", bl_file["meta"]["n_events"] == len(train),
          str(bl_file["meta"]["n_events"]))

    train.to_csv(td / "window_ok.csv", index=False)
    drifted.to_csv(td / "window_drifted.csv", index=False)
    tiny = train.head(10)
    tiny.to_csv(td / "window_tiny.csv", index=False)

    before = len(audit_rows())
    rc_ok = drift_monitor.main(["check", "--baseline", str(base_out),
                                "--window-csv", str(td / "window_ok.csv")])
    check("ok window exits 0", rc_ok == 0, f"rc={rc_ok}")
    check("ok window appends nothing to audit", len(audit_rows()) == before,
          f"{before} -> {len(audit_rows())}")

    rc_alert = drift_monitor.main(["check", "--baseline", str(base_out),
                                   "--window-csv", str(td / "window_drifted.csv")])
    check("drifted window exits 2", rc_alert == 2, f"rc={rc_alert}")
    flush_audit_queue()
    rows = audit_rows()
    check("drift_alert appended to audit chain", len(rows) == before + 1
          and rows[-1][1] == "drift_alert", str(rows[-1][:2]))
    linked = rows[-1][2] == (GENESIS_HASH if len(rows) == 1 else rows[-2][3])
    check("drift_alert hash-chained (prev == previous entry)", linked,
          f"{rows[-1][2][:12]}…")
    con = sqlite3.connect(get_settings().audit_db_path)
    payload = json.loads(con.execute(
        "SELECT payload_summary FROM audit_events WHERE seq = ?", (rows[-1][0],)
    ).fetchone()[0])
    con.close()
    check("alert payload actionable + pseudonymous (feature PSI only, no raw rows)",
          payload["features_alerting"].get("amount_ratio") is not None
          and payload["action"] == "retraining_review_per_section_6"
          and payload.get("n_events") == len(drifted)
          and "ts" not in payload and "fraud_id" not in payload,
          str(payload["features_alerting"]))
    check("chain integrity verified (writer)", verify_chain(audit_rows_as_objects(rows))["ok"] is True)

    rc_tiny = drift_monitor.main(["check", "--baseline", str(base_out),
                                  "--window-csv", str(td / "window_tiny.csv")])
    check("insufficient data -> exit 3, no alert", rc_tiny == 3
          and len(audit_rows()) == before + 1, f"rc={rc_tiny}")

    rc_missing = drift_monitor.main(["check", "--baseline", str(td / "nope.json")])
    check("missing baseline -> exit 3", rc_missing == 3, f"rc={rc_missing}")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    return 1 if failures else 0


def audit_rows_as_objects(rows: list[tuple]):
    """Re-wrap raw audit rows into AuditEvent-like objects for verify_chain."""
    from types import SimpleNamespace
    con = sqlite3.connect(get_settings().audit_db_path)
    payloads = {row[0]: row[1] for row in con.execute(
        "SELECT seq, payload_summary FROM audit_events ORDER BY seq")}
    con.close()
    return [SimpleNamespace(seq=r[0], event_type=r[1], prev_hash=r[2],
                            entry_hash=r[3], payload_summary=payloads[r[0]])
            for r in rows]


if __name__ == "__main__":
    sys.exit(main())
