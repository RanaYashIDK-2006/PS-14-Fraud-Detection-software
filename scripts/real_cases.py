#!/usr/bin/env python3
"""30-case real-world batch test against the live PS-14 stack.

Uses human-readable labels: device (new/usual), location (usual/unusual/abnormal),
and amount categories. Runs through the live Risk Engine and writes a report for
the front page's /real-cases endpoint.

Run from the project root with the stack up (ports 8003 + 8005):
  python scripts/real_cases.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RISK_URL = "http://127.0.0.1:8003"
AUDIT_URL = "http://127.0.0.1:8005"
INTERNAL_TOKEN = "ps14-dev-internal-token-change-me"
FRAUD_ID = "FA7K2Q9X4M6B3Z5C"
DEFAULT_REPORT = str(Path(os.environ.get("DB_DIR", "db")) / "real_cases_report.json")


def _post(url: str, payload: dict, headers: dict | None = None) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def _get(url: str, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def base_vector() -> dict:
    return {
        "amount_ratio": 1.0,
        "txn_freq_last_24h": 2,
        "txn_time_unusual": 0,
        "new_device_flag": 0,
        "unusual_location_flag": 0,
        "unusual_recipient_flag": 0,
        "failed_auth_count_24h": 0,
        "days_since_last_similar_txn": 3.0,
        "gradual_escalation_score": 0.0,
        "known_device_count": 2,
        "account_tenure_days": 400.0,
        "hour_of_day": 13,
        "is_weekend": 0,
        "shared_device_accounts": 0,
        "shared_recipient_accounts": 0,
        "mule_ring_score": 0.0,
        "account_daily_spend_ratio": 0.8,
        "device_daily_count": 3,
    }


# 30 real-world cases: device (new/usual) × location (usual/unusual/abnormal) × amount
CASES: list[tuple[str, dict]] = [
    # --- NEW DEVICE cases ---
    ("New device · Usual location · Small amount ($45)", {
        **base_vector(), "new_device_flag": 1, "amount_ratio": 0.5,
    }),
    ("New device · Usual location · Medium amount ($180)", {
        **base_vector(), "new_device_flag": 1, "amount_ratio": 1.2,
    }),
    ("New device · Usual location · Large amount ($1,200)", {
        **base_vector(), "new_device_flag": 1, "amount_ratio": 8.0,
    }),
    ("New device · Usual location · Huge amount ($4,500)", {
        **base_vector(), "new_device_flag": 1, "amount_ratio": 30.0,
    }),
    ("New device · Unusual location · Small amount ($35)", {
        **base_vector(), "new_device_flag": 1, "unusual_location_flag": 1, "amount_ratio": 0.4,
    }),
    ("New device · Unusual location · Medium amount ($250)", {
        **base_vector(), "new_device_flag": 1, "unusual_location_flag": 1, "amount_ratio": 1.5,
    }),
    ("New device · Unusual location · Large amount ($800)", {
        **base_vector(), "new_device_flag": 1, "unusual_location_flag": 1, "amount_ratio": 5.0,
    }),
    ("New device · Unusual location · Huge amount ($3,000)", {
        **base_vector(), "new_device_flag": 1, "unusual_location_flag": 1, "amount_ratio": 20.0,
    }),
    ("New device · Abnormal location · Medium amount ($200)", {
        **base_vector(), "new_device_flag": 1, "unusual_location_flag": 1, "unusual_recipient_flag": 1,
        "amount_ratio": 1.3,
    }),
    ("New device · Abnormal location · Large amount ($1,500)", {
        **base_vector(), "new_device_flag": 1, "unusual_location_flag": 1, "unusual_recipient_flag": 1,
        "amount_ratio": 10.0,
    }),

    # --- USUAL DEVICE cases ---
    ("Usual device · Usual location · Small amount ($25)", {
        **base_vector(), "amount_ratio": 0.3,
    }),
    ("Usual device · Usual location · Medium amount ($150)", {
        **base_vector(), "amount_ratio": 1.0,
    }),
    ("Usual device · Usual location · Large amount ($900)", {
        **base_vector(), "amount_ratio": 6.0,
    }),
    ("Usual device · Usual location · Huge amount ($5,000)", {
        **base_vector(), "amount_ratio": 33.0,
    }),
    ("Usual device · Unusual location · Small amount ($40)", {
        **base_vector(), "unusual_location_flag": 1, "amount_ratio": 0.5,
    }),
    ("Usual device · Unusual location · Medium amount ($220)", {
        **base_vector(), "unusual_location_flag": 1, "amount_ratio": 1.4,
    }),
    ("Usual device · Unusual location · Large amount ($1,100)", {
        **base_vector(), "unusual_location_flag": 1, "amount_ratio": 7.0,
    }),
    ("Usual device · Unusual location · Huge amount ($6,000)", {
        **base_vector(), "unusual_location_flag": 1, "amount_ratio": 40.0,
    }),
    ("Usual device · Abnormal location · Medium amount ($180)", {
        **base_vector(), "unusual_location_flag": 1, "unusual_recipient_flag": 1, "amount_ratio": 1.2,
    }),
    ("Usual device · Abnormal location · Large amount ($2,000)", {
        **base_vector(), "unusual_location_flag": 1, "unusual_recipient_flag": 1, "amount_ratio": 13.0,
    }),

    # --- EDGE CASES (mixed signals) ---
    ("New device · Usual location · 3 AM · $400", {
        **base_vector(), "new_device_flag": 1, "amount_ratio": 2.5, "txn_time_unusual": 1,
        "hour_of_day": 3,
    }),
    ("Usual device · Usual location · 2 AM · $15", {
        **base_vector(), "amount_ratio": 0.2, "txn_time_unusual": 1, "hour_of_day": 2,
    }),
    ("New device · Unusual location · 11 PM · $600", {
        **base_vector(), "new_device_flag": 1, "unusual_location_flag": 1, "amount_ratio": 4.0,
        "hour_of_day": 23,
    }),
    ("Usual device · Unusual location · Known recipient · $50", {
        **base_vector(), "unusual_location_flag": 1, "unusual_recipient_flag": 0, "amount_ratio": 0.6,
    }),

    # --- HIGH-VELOCITY / MULE cases ---
    ("Usual device · Usual location · 15 txns today · $30", {
        **base_vector(), "amount_ratio": 0.4, "txn_freq_last_24h": 15, "device_daily_count": 15,
    }),
    ("New device · Unusual location · Mule ring score 0.8 · $500", {
        **base_vector(), "new_device_flag": 1, "unusual_location_flag": 1,
        "mule_ring_score": 0.8, "amount_ratio": 3.0,
        "shared_device_accounts": 8, "shared_recipient_accounts": 6,
    }),
    ("Usual device · Usual location · Failed auth 5x · $100", {
        **base_vector(), "amount_ratio": 0.7, "failed_auth_count_24h": 5,
    }),
    ("New device · Abnormal location · 30 txns · $1,200", {
        **base_vector(), "new_device_flag": 1, "unusual_location_flag": 1, "unusual_recipient_flag": 1,
        "amount_ratio": 8.0, "txn_freq_last_24h": 30, "device_daily_count": 30,
    }),

    # --- LOW-RISK / CONFIDENCE cases ---
    ("Usual device · Usual location · Known recipient · $12", {
        **base_vector(), "amount_ratio": 0.15, "unusual_recipient_flag": 0,
    }),
    ("Usual device · Usual location · $80 · mature account", {
        **base_vector(), "amount_ratio": 0.5, "account_tenure_days": 1500.0,
    }),
]

assert len(CASES) == 30, f"expected 30, got {len(CASES)}"


def main() -> int:
    # Reachability check
    try:
        _get(f"{RISK_URL}/health")
        _get(f"{AUDIT_URL}/health")
    except Exception as e:
        print(f"ERROR: stack unreachable: {e}", file=sys.stderr)
        return 2

    before = _get(f"{AUDIT_URL}/audit/integrity", {"X-Internal-Token": INTERNAL_TOKEN})
    print(f"== 30-case real-world test == ({datetime.now(timezone.utc):%H:%M:%S} UTC)")

    results = []
    for i, (name, v) in enumerate(CASES):
        res = _post(f"{RISK_URL}/internal/evaluate", {
            "event_id": f"real-{i:02d}-00000000",
            "fraud_id": FRAUD_ID,
            "features": v,
        }, {"X-Internal-Token": INTERNAL_TOKEN})
        results.append((name, v, res))

    after = _get(f"{AUDIT_URL}/audit/integrity", {"X-Internal-Token": INTERNAL_TOKEN})

    # Print table
    print(f"\n{'#':>3} {'Scenario':<52} {'Score':>5} {'Band':<8} {'Decision':<9} Reasons")
    print("-" * 100)
    bands = {"low": 0, "medium": 0, "high": 0}
    for i, (name, _v, r) in enumerate(results, 1):
        bands[r["risk_band"]] += 1
        reasons = ", ".join(r["reason_codes"][:3])
        if len(r["reason_codes"]) > 3:
            reasons += f" +{len(r['reason_codes']) - 3}"
        print(f"{i:>3} {name:<52} {r['risk_score']:>5} {r['risk_band']:<8} "
              f"{r['decision']:<9} {reasons}")

    print(f"\nDistribution: LOW {bands['low']} · MEDIUM {bands['medium']} · HIGH {bands['high']}")
    print(f"Audit chain: {before.get('n_entries')} -> {after.get('n_entries')} entries")

    # Write report for the front page
    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_cases": 30,
        "distribution": {"low": bands["low"], "medium": bands["medium"], "high": bands["high"]},
        "chain": {
            "before": before.get("n_entries"),
            "after": after.get("n_entries"),
            "grew": after.get("n_entries", 0) - before.get("n_entries", 0),
        },
        "cases": [
            {"name": name, "score": r["risk_score"], "band": r["risk_band"],
             "decision": r["decision"], "reasons": r["reason_codes"]}
            for name, _v, r in results
        ],
    }
    report_path = Path(DEFAULT_REPORT)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, report_path)
    print(f"\nWrote report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
