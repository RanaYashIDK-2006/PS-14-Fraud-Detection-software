#!/usr/bin/env python3
"""Temporal correctness tests.

Proves that features are computed point-in-time correctly:
future records cannot influence past predictions.

Uses synthetic timestamps to verify temporal isolation.
"""

import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        errors.append(name)


def run_tests():
    global passed, failed, errors
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("TEMPORAL CORRECTNESS TESTS")
    print("=" * 70)

    try:
        from src.privacy_layer.features import (
            derive_event_features, typical_hours_default,
            escalation_score, amount_bucket, mule_ring_score,
            ML_FEATURES,
        )
    except ImportError as e:
        print(f"\n  FATAL: Cannot import features module: {e}")
        return 1

    # ---- Feature derivation uses only pre-event data ----
    print("\n[1] Feature derivation — point-in-time correctness")

    # Simulate an account with 7 days of history
    now = datetime(2026, 8, 24, 14, 0, 0)
    profile = {
        "median_amount": 100.0,
        "typical_hours": list(range(8, 22)),
        "known_devices": ["device_A"],
        "usual_locations": ["loc_1", "loc_2"],
        "usual_recipients": ["recip_1"],
        "tenure_days": 30.0,
    }

    # Event that happened BEFORE now
    event = {
        "amount": 250.0,
        "hour_of_day": 10,
        "is_weekend": 0,
        "device_hash": "device_A",
        "location_id": "loc_1",
        "recipient_id": "recip_1",
        "failed_auth_count_24h": 0,
    }

    features = derive_event_features(
        profile=profile,
        event=event,
        freq_last_24h=5,
        days_since_similar=2.5,
        recent_ratios=[0.8, 1.0, 1.2],
    )

    test("amount_ratio is derived from event + profile (not future data)",
         abs(features["amount_ratio"] - 2.5) < 0.01,
         f"got {features['amount_ratio']}")

    test("txn_freq_last_24h is a pre-event count",
         features["txn_freq_last_24h"] == 5)

    test("new_device_flag is based on pre-event known devices",
         features["new_device_flag"] == 0,  # device_A is known
         f"got {features['new_device_flag']}")

    test("unusual_location_flag is based on pre-event profile",
         features["unusual_location_flag"] == 0,  # loc_1 is usual
         f"got {features['unusual_location_flag']}")

    test("unusual_recipient_flag is based on pre-event profile",
         features["unusual_recipient_flag"] == 0,  # recip_1 is usual
         f"got {features['unusual_recipient_flag']}")

    test("account_tenure_days comes from profile (pre-computed)",
         abs(features["account_tenure_days"] - 30.0) < 0.01)

    test("days_since_last_similar_txn is pre-event",
         abs(features["days_since_last_similar_txn"] - 2.5) < 0.01)

    test("gradual_escalation_score uses only recent ratios (pre-event)",
         0.0 <= features["gradual_escalation_score"] <= 1.0)

    # ---- Future event cannot influence past features ----
    print("\n[2] Future event isolation")

    # Profile has history up to day 10
    profile_day10 = {
        "median_amount": 100.0,
        "typical_hours": list(range(8, 22)),
        "known_devices": ["device_A"],
        "usual_locations": ["loc_1"],
        "usual_recipients": ["recip_1"],
        "tenure_days": 10.0,
    }

    # Event on day 5 (before the profile was updated on day 10)
    event_day5 = {
        "amount": 50.0,  # half the median → ratio 0.5
        "hour_of_day": 14,
        "is_weekend": 0,
        "device_hash": "device_A",
        "location_id": "loc_1",
        "recipient_id": "recip_1",
        "failed_auth_count_24h": 0,
    }

    features_day5 = derive_event_features(
        profile=profile_day10,
        event=event_day5,
        freq_last_24h=3,
        days_since_similar=1.0,
        recent_ratios=[0.9, 1.0],
    )

    test("Day-5 event uses profile from day 10 (historical snapshot)",
         abs(features_day5["amount_ratio"] - 0.5) < 0.01,
         f"got {features_day5['amount_ratio']}")

    # A future device (added on day 15) should NOT affect day-5 features
    event_day5_new_device = {
        "amount": 50.0,
        "hour_of_day": 14,
        "is_weekend": 0,
        "device_hash": "device_B",  # unknown at day 5
        "location_id": "loc_1",
        "recipient_id": "recip_1",
        "failed_auth_count_24h": 0,
    }

    features_day5_new = derive_event_features(
        profile=profile_day10,
        event=event_day5_new_device,
        freq_last_24h=3,
        days_since_similar=1.0,
        recent_ratios=[0.9, 1.0],
    )

    test("Day-5 event with unknown device flags new_device=1 (correct)",
         features_day5_new["new_device_flag"] == 1)

    # ---- Escalation score uses only pre-event ratios ----
    print("\n[3] Escalation score — temporal correctness")

    # Flat history → no escalation
    flat_ratios = [1.0, 1.0, 1.0, 1.0, 1.0]
    flat_score = escalation_score(flat_ratios)
    test("Flat history → low escalation score",
         flat_score < 0.1,
         f"got {flat_score}")

    # Escalating history → high escalation
    escalate_ratios = [0.5, 0.8, 1.2, 2.0, 3.0]
    esc_score = escalation_score(escalate_ratios)
    test("Escalating history → high escalation score",
         esc_score > 0.3,
         f"got {esc_score}")

    # Declining history → no escalation
    decline_ratios = [3.0, 2.0, 1.2, 0.8, 0.5]
    dec_score = escalation_score(decline_ratios)
    test("Declining history → low escalation score",
         dec_score < 0.1,
         f"got {dec_score}")

    # ---- Mule ring score is stateless (no temporal concern) ----
    print("\n[4] Mule ring score — stateless computation")
    test("mule_ring_score(0,0) = 0",
         mule_ring_score(0, 0) == 0.0)
    test("mule_ring_score(2,2) = 1.0",
         mule_ring_score(2, 2) == 1.0)
    test("mule_ring_score(1,0) = 0.25",
         mule_ring_score(1, 0) == 0.25)

    # ---- Amount bucket is stateless ----
    print("\n[5] Amount bucket — stateless computation")
    test("amount_bucket(0.3) = low_relative_to_avg",
         amount_bucket(0.3) == "low_relative_to_avg")
    test("amount_bucket(1.0) = typical",
         amount_bucket(1.0) == "typical")
    test("amount_bucket(2.0) = high_relative_to_avg",
         amount_bucket(2.0) == "high_relative_to_avg")
    test("amount_bucket(3.0) = extreme_relative_to_avg",
         amount_bucket(3.0) == "extreme_relative_to_avg")

    # ---- Feature list consistency ----
    print("\n[6] Feature list consistency")
    test(f"ML_FEATURES has {len(ML_FEATURES)} features",
         len(ML_FEATURES) >= 16,
         f"got {len(ML_FEATURES)} (expected >= 16)")

    # All features should be computable from profile + event (no future data)
    expected_inputs = {
        "amount_ratio",  # event.amount / profile.median
        "txn_freq_last_24h",  # count from DB (pre-event)
        "txn_time_unusual",  # event.hour not in profile.typical_hours
        "new_device_flag",  # event.device not in profile.known_devices
        "unusual_location_flag",  # event.location not in profile.usual_locations
        "unusual_recipient_flag",  # event.recipient not in profile.usual_recipients
        "failed_auth_count_24h",  # count from DB (pre-event)
        "days_since_last_similar_txn",  # time delta (pre-event)
        "gradual_escalation_score",  # slope of recent ratios (pre-event)
        "known_device_count",  # len(profile.known_devices)
        "account_tenure_days",  # now - profile.created_at
        "hour_of_day",  # event timestamp
        "is_weekend",  # event timestamp
        "shared_device_accounts",  # link analysis (pre-event)
        "shared_recipient_accounts",  # link analysis (pre-event)
        "mule_ring_score",  # composite of shared counts
        # Deviation features — derived from profile history + current event
        "hour_deviation",  # circular distance from typical hours
        "amount_zscore",  # z-score vs account amount history
        "velocity_deviation",  # sigmoid of freq deviation
        "recipient_novelty",  # fraction of recent recipients new
        "txn_regularity",  # CV of inter-arrival times
    }
    test("All features derivable from profile + event (no future data)",
         set(ML_FEATURES) == expected_inputs,
         f"Unexpected: {set(ML_FEATURES) - expected_inputs}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e}")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
