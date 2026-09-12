#!/usr/bin/env python3
"""Fairness / bias review for PS-14 risk scores.

Reviews the §16 feature set for plausible proxy variables and checks
whether risk outcomes differ unexpectedly across account segments the
data already supports (tenure, device count, time-of-day patterns) —
without collecting any new sensitive attributes.

The §16 features are intentionally limited:
  - amount_ratio (transaction amount / median)
  - txn_freq_last_24h
  - txn_time_unusual (boolean)
  - new_device_flag (boolean)
  - unusual_location_flag (boolean)
  - unusual_recipient_flag (boolean)
  - failed_auth_count_24h
  - days_since_last_similar_txn
  - gradual_escalation_score
  - known_device_count
  - account_tenure_days
  - hour_of_day
  - is_weekend
  - shared_device_accounts
  - shared_recipient_accounts
  - mule_ring_score

Proxy variable analysis: which features could correlate with protected
characteristics?

  - hour_of_day: correlates with work schedules (socioeconomic proxy)
  - is_weekend: correlates with employment type
  - account_tenure_days: correlates with age/loyalty
  - known_device_count: correlates with tech access
  - unusual_location_flag: could proxy for travel patterns (geography)

Usage:
    python scripts/fairness_review.py [--db-dir db] [--json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _load_scores(db_path: Path) -> list[dict]:
    """Load all risk scores with their reason codes."""
    if not db_path.exists():
        return []
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT fraud_id, risk_score, risk_band, reason_codes, ml_score, model_version FROM risk_scores"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def _load_features(db_path: Path) -> dict[str, list[dict]]:
    """Load feature vectors grouped by fraud_id."""
    if not db_path.exists():
        return {}
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT fraud_id, account_tenure_days, known_device_count, hour_of_day, is_weekend, "
        "new_device_flag, unusual_location_flag FROM transaction_features"
    ).fetchall()
    con.close()
    by_fid: dict[str, list[dict]] = {}
    for r in rows:
        d = dict(r)
        fid = d.pop("fraud_id")
        by_fid.setdefault(fid, []).append(d)
    return by_fid


def _segment_scores(scores: list[dict], features: dict[str, list[dict]]) -> dict:
    """Segment scores by account characteristics."""
    segments: dict[str, dict] = {
        "by_tenure": {"new_0_7d": [], "mid_8_30d": [], "mature_30d_plus": []},
        "by_device_count": {"no_device": [], "single_device": [], "multi_device": []},
        "by_time_pattern": {"business_hours": [], "off_hours": [], "weekend": []},
    }

    for s in scores:
        fid = s["fraud_id"]
        score = s["risk_score"]
        feats = features.get(fid, [{}])
        avg_tenure = sum(f.get("account_tenure_days", 0) for f in feats) / max(len(feats), 1)
        avg_devices = sum(f.get("known_device_count", 0) for f in feats) / max(len(feats), 1)
        avg_hour = sum(f.get("hour_of_day", 12) for f in feats) / max(len(feats), 1)
        is_weekend = any(f.get("is_weekend", 0) for f in feats)

        # Tenure segments
        if avg_tenure <= 7:
            segments["by_tenure"]["new_0_7d"].append(score)
        elif avg_tenure <= 30:
            segments["by_tenure"]["mid_8_30d"].append(score)
        else:
            segments["by_tenure"]["mature_30d_plus"].append(score)

        # Device segments
        if avg_devices == 0:
            segments["by_device_count"]["no_device"].append(score)
        elif avg_devices == 1:
            segments["by_device_count"]["single_device"].append(score)
        else:
            segments["by_device_count"]["multi_device"].append(score)

        # Time pattern segments
        if is_weekend:
            segments["by_time_pattern"]["weekend"].append(score)
        elif 9 <= avg_hour <= 17:
            segments["by_time_pattern"]["business_hours"].append(score)
        else:
            segments["by_time_pattern"]["off_hours"].append(score)

    return segments


def _compute_segment_stats(segment_scores: list[int]) -> dict:
    """Compute mean, median, and high-risk rate for a segment."""
    if not segment_scores:
        return {"count": 0, "mean": 0, "median": 0, "high_risk_pct": 0}
    s = sorted(segment_scores)
    return {
        "count": len(s),
        "mean": round(sum(s) / len(s), 1),
        "median": s[len(s) // 2],
        "high_risk_pct": round(100 * sum(1 for x in s if x >= 71) / len(s), 1),
    }


def _proxy_analysis() -> list[dict]:
    """Analyze potential proxy variables in the §16 feature set."""
    return [
        {
            "feature": "hour_of_day",
            "proxy_risk": "medium",
            "reason": "Correlates with work schedules (socioeconomic proxy). Unusual hours trigger txn_time_unusual but the feature itself is necessary for fraud detection.",
            "recommendation": "Keep — necessary for temporal anomaly detection. Monitor for disparate impact across segments.",
        },
        {
            "feature": "is_weekend",
            "proxy_risk": "low",
            "reason": "Correlates with employment type. Weekend transactions are slightly less common, triggering frequency anomalies.",
            "recommendation": "Keep — weak proxy, strong fraud signal. No action needed.",
        },
        {
            "feature": "account_tenure_days",
            "proxy_risk": "medium",
            "reason": "Correlates with age and loyalty. New accounts naturally score higher (cold start), but the cold-start fix has reduced this effect.",
            "recommendation": "Keep — essential for behavioral baseline. Monitor new-account false positive rates.",
        },
        {
            "feature": "known_device_count",
            "proxy_risk": "low",
            "reason": "Correlates with tech access. Accounts with zero known devices flag new_device_flag on first transaction.",
            "recommendation": "Keep — strong fraud signal (device registration). No action needed.",
        },
        {
            "feature": "unusual_location_flag",
            "proxy_risk": "medium",
            "reason": "Could proxy for travel patterns (geography, immigration status). The flag is coarse (boolean) not precise geo.",
            "recommendation": "Keep — coarse location is necessary for fraud. Monitor for geographic bias.",
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="PS-14 fairness/bias review")
    parser.add_argument("--db-dir", type=Path, default=Path("db"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scores = _load_scores(args.db_dir / "risk.db")
    features = _load_features(args.db_dir / "features.db")

    if not scores:
        print("No risk scores found. Run the demo seed or batch test first.")
        sys.exit(1)

    segments = _segment_scores(scores, features)
    proxy_analysis = _proxy_analysis()

    report = {
        "total_scores": len(scores),
        "overall": _compute_segment_stats([s["risk_score"] for s in scores]),
        "segments": {},
        "proxy_analysis": proxy_analysis,
        "findings": [],
    }

    for category, segs in segments.items():
        report["segments"][category] = {}
        for name, seg_scores in segs.items():
            report["segments"][category][name] = _compute_segment_stats(seg_scores)

    # Check for significant disparities
    for category, segs in segments.items():
        means = {name: _compute_segment_stats(scores)["mean"] for name, scores in segs.items() if scores}
        if means:
            max_mean = max(means.values())
            min_mean = min(means.values())
            if max_mean - min_mean > 20:
                report["findings"].append(
                    f"DISPARITY in {category}: max mean {max_mean} vs min mean {min_mean} (gap {max_mean - min_mean})"
                )

    if not args.json:
        print(f"Total risk scores: {report['total_scores']}")
        print(f"Overall: mean={report['overall']['mean']} high_risk={report['overall']['high_risk_pct']}%")
        print()
        for category, segs in report["segments"].items():
            print(f"--- {category} ---")
            for name, stats in segs.items():
                print(f"  {name}: n={stats['count']} mean={stats['mean']} high={stats['high_risk_pct']}%")
        print()
        print("--- Proxy variable analysis ---")
        for p in proxy_analysis:
            print(f"  [{p['proxy_risk'].upper()}] {p['feature']}: {p['reason']}")
        print()
        if report["findings"]:
            print("FINDINGS:")
            for f in report["findings"]:
                print(f"  - {f}")
        else:
            print("No significant disparities found.")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
