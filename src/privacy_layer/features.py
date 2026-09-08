"""Feature derivation logic — shared by the Privacy Layer (production path)
and the synthetic data generator (training path), so training features and
live features stay consistent.

Only derived, purpose-limited values are produced (section 16): no raw
amounts, no exact geo, no device IDs, no timestamps finer than hour-of-day.
"""

from __future__ import annotations

import numpy as np

# Features consumed by the ML pipeline. One source of truth for the
# generator, the trainer, and the Privacy Layer.
ML_FEATURES = [
    "amount_ratio",
    "txn_freq_last_24h",
    "txn_time_unusual",
    "new_device_flag",
    "unusual_location_flag",
    "unusual_recipient_flag",
    "failed_auth_count_24h",
    "days_since_last_similar_txn",
    "gradual_escalation_score",
    "known_device_count",
    "account_tenure_days",
    "hour_of_day",
    "is_weekend",
    # Cross-account link-analysis signals (mule rings, shared devices): the
    # count of OTHER accounts seen on this device / recipient, plus a
    # normalized composite. Computed dataset-wide in training and against the
    # live store in the Privacy Layer.
    "shared_device_accounts",
    "shared_recipient_accounts",
    "mule_ring_score",
    # Deviation features: continuous scores that capture HOW UNUSUAL each
    # signal is relative to this account's baseline. These complement the
    # binary flags (which only say "different" vs "same") and help detect
    # subtle fraud patterns that the binary features miss.
    "hour_deviation",        # circular distance from typical hours [0,1]
    "amount_zscore",         # signed z-score of amount vs account history
    "velocity_deviation",    # how unusual is the current frequency [0,1]
    "recipient_novelty",     # fraction of recent recipients that are new [0,1]
    "txn_regularity",        # coefficient of variation of inter-arrival times
]


def typical_hours_default() -> list[int]:
    """Used before an account has enough history to learn its own pattern."""
    return list(range(8, 22))  # 8:00–21:59


def amount_bucket(ratio: float) -> str:
    if ratio < 0.5:
        return "low_relative_to_avg"
    if ratio < 1.3:
        return "typical"
    if ratio < 2.5:
        return "high_relative_to_avg"
    return "extreme_relative_to_avg"


def escalation_score(ratios: list[float]) -> float:
    """Slope of log(amount_ratio) over the recent window, mapped to [0, 1].

    Zero for flat/declining histories; large for a "boiling frog" ramp
    (attack scenario A: gradual escalation over weeks).
    """
    if len(ratios) < 3:
        return 0.0
    x = np.arange(len(ratios), dtype=float)
    y = np.log(np.maximum(ratios, 1e-6))
    slope = float(np.polyfit(x, y, 1)[0])
    return float(np.clip(slope * len(ratios) * 0.5, 0.0, 1.0))


GRAPH_CAP = 8  # cap shared-account counts so the model sees stable magnitudes


def _circular_distance(h1: int, h2: int) -> float:
    """Shortest angular distance between two hours on a 24h clock, in [0, 12]."""
    diff = abs(h1 - h2)
    return min(diff, 24 - diff)


def hour_deviation_score(current_hour: int, typical_hours: list[int]) -> float:
    """Circular distance from current hour to the nearest typical hour.

    Returns a value in [0, 1] where 0 = perfectly normal, 1 = maximally unusual
    (e.g. 3 AM for an account that only transacts 9-17).
    """
    if not typical_hours:
        return 0.0
    min_dist = min(_circular_distance(current_hour, h) for h in typical_hours)
    return round(min(min_dist / 12.0, 1.0), 4)


def amount_zscore(amount: float, median_amount: float,
                   recent_amounts: list[float] | None = None) -> float:
    """Signed z-score of amount relative to account history.

    Positive = above normal, negative = below normal.
    If no history, returns 0.0.
    """
    if recent_amounts and len(recent_amounts) >= 3:
        arr = np.array(recent_amounts + [amount])
    else:
        # No history — fall back to ratio from median
        if median_amount > 0:
            return round(float(np.clip((amount / median_amount) - 1.0, -3.0, 3.0)), 4)
        return 0.0
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        return 0.0
    return round(float(np.clip((amount - mean) / std, -3.0, 3.0)), 4)


def velocity_deviation_score(current_freq: int, typical_freq: float) -> float:
    """How unusual the current 24h frequency is relative to account baseline.

    Returns [0, 1] where 0 = normal, 1 = highly unusual.
    Uses a simple z-score with sigmoid squashing.
    """
    if typical_freq <= 0:
        # No history — use absolute threshold
        return round(float(np.clip(current_freq / 10.0, 0.0, 1.0)), 4)
    z = (current_freq - typical_freq) / max(typical_freq ** 0.5, 1.0)
    return round(float(1.0 / (1.0 + np.exp(-z))), 4)


def recipient_novelty_score(
    current_recipient: str,
    usual_recipients: list[str],
    recent_recipients: list[str] | None = None,
) -> float:
    """Fraction of recent recipients that are new to this account.

    If recent_recipients is provided, computes novelty over that window.
    Otherwise, just returns 1.0 if the current recipient is new.
    """
    if recent_recipients:
        new_count = sum(1 for r in recent_recipients if r not in usual_recipients)
        return round(new_count / max(len(recent_recipients), 1), 4)
    return round(1.0 if current_recipient not in usual_recipients else 0.0, 4)


def txn_regularity_score(inter_arrival_hours: list[float]) -> float:
    """Coefficient of variation of inter-arrival times.

    Low CV = regular (suspicious bot-like), high CV = irregular (normal human).
    Returns [0, 1] where 0 = very regular (bot-like), 1 = very irregular (normal).
    """
    if len(inter_arrival_hours) < 2:
        return 0.5  # insufficient data
    arr = np.array(inter_arrival_hours)
    mean = arr.mean()
    if mean < 1e-6:
        return 0.0
    cv = arr.std() / mean
    # Sigmoid transform: CV=0 -> 0.0 (bot), CV=1 -> 0.73 (normal), CV>2 -> ~1.0
    return round(float(1.0 / (1.0 + np.exp(-(cv - 0.5)))), 4)


def mule_ring_score(shared_device: int, shared_recipient: int) -> float:
    """Normalized composite of cross-account sharing in [0, 1].

    A mule ring typically shares BOTH a device and a converging recipient
    (2 accounts -> 0.5, 3+ -> 1.0); a family phone shared by two legit
    accounts contributes only 1/4. Pure device sharing by itself stays low.
    """
    return min(1.0, (shared_device + shared_recipient) / 4.0)


def derive_event_features(
    *,
    profile: dict,
    event: dict,
    freq_last_24h: int,
    days_since_similar: float,
    recent_ratios: list[float],
    shared_device_accounts: int = 0,
    shared_recipient_accounts: int = 0,
    velocity: dict | None = None,
) -> dict:
    """Derive the section-16 feature vector for one event.

    `profile` must provide: median_amount, typical_hours, known_devices,
    usual_locations, usual_recipients, tenure_days.
    `event` must provide: amount, hour_of_day, is_weekend, device_hash,
    location_id, recipient_id, failed_auth_count_24h.
    `shared_device_accounts` / `shared_recipient_accounts` are the counts of
    OTHER accounts seen on this device / recipient (link analysis); they
    default to 0 when no cross-account context is available.
    `velocity` is an optional dict with real-time running stats from the
    UserVelocityTracker: user_tx_count, user_avg_amt, card_tx_count,
    merch_tx_count. These replace proxy approximations in the Altman ensemble.
    """
    median = profile.get("median_amount") or event["amount"]
    ratio = event["amount"] / median
    typical = profile.get("typical_hours") or typical_hours_default()
    # Support both pre-computed new_device flag (production path, avoids
    # an extra DB query for the full known_devices list) and the legacy
    # known_devices list path (synthetic data generator, training).
    if "new_device" in event:
        new_device = bool(event["new_device"])
    else:
        new_device = event["device_hash"] not in profile.get("known_devices", [])
    usual_locations = profile.get("usual_locations", [])
    usual_recipients = profile.get("usual_recipients", [])
    s_dev = int(min(GRAPH_CAP, shared_device_accounts))
    s_recip = int(min(GRAPH_CAP, shared_recipient_accounts))

    # Deviation features — how unusual each signal is relative to this account.
    _hour_dev = hour_deviation_score(event["hour_of_day"], typical)
    _amt_z = amount_zscore(
        event["amount"],
        profile.get("median_amount", event["amount"]),
        profile.get("recent_amounts"),
    )
    _vel_dev = velocity_deviation_score(
        freq_last_24h,
        profile.get("typical_freq_24h", float(freq_last_24h)),
    )
    _recip_novelty = recipient_novelty_score(
        event["recipient_id"],
        usual_recipients,
        profile.get("recent_recipients"),
    )
    _txn_reg = txn_regularity_score(
        profile.get("inter_arrival_hours", []),
    )

    # Real-time velocity features from UserVelocityTracker (computed at ingest).
    _vel = velocity or {}
    _user_tx_count = int(_vel.get("user_tx_count", 0))
    _user_avg_amt = float(_vel.get("user_avg_amt", 0.0))
    _card_tx_count = int(_vel.get("card_tx_count", 0))
    _merch_tx_count = int(_vel.get("merch_tx_count", 0))

    return {
        "amount_ratio": round(ratio, 4),
        "txn_amount_bucket": amount_bucket(ratio),
        "txn_freq_last_24h": int(freq_last_24h),
        "txn_time_unusual": int(event["hour_of_day"] not in typical),
        "new_device_flag": int(new_device),
        "unusual_location_flag": int(event["location_id"] not in usual_locations),
        "unusual_recipient_flag": int(event["recipient_id"] not in usual_recipients),
        "failed_auth_count_24h": int(event["failed_auth_count_24h"]),
        "days_since_last_similar_txn": round(days_since_similar, 2),
        "gradual_escalation_score": round(escalation_score(recent_ratios + [ratio]), 4),
        # Prefer pre-computed count (production); fall back to list length (training).
        "known_device_count": profile.get("known_device_count", len(profile.get("known_devices", [])) + (1 if new_device else 0)),
        "account_tenure_days": round(profile.get("tenure_days", 1.0), 2),
        "hour_of_day": int(event["hour_of_day"]),
        "is_weekend": int(event["is_weekend"]),
        "shared_device_accounts": s_dev,
        "shared_recipient_accounts": s_recip,
        "mule_ring_score": round(mule_ring_score(s_dev, s_recip), 4),
        # Deviation scores — continuous measures of how unusual each signal is.
        "hour_deviation": _hour_dev,
        "amount_zscore": _amt_z,
        "velocity_deviation": _vel_dev,
        "recipient_novelty": _recip_novelty,
        "txn_regularity": _txn_reg,
        # Real-time velocity features (replaces proxy approximations).
        "user_tx_count": _user_tx_count,
        "user_avg_amt": round(_user_avg_amt, 4),
        "card_tx_count": _card_tx_count,
        "merch_tx_count": _merch_tx_count,
    }
