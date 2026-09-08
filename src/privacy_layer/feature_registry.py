"""Feature registry — structural leakage prevention.

Every feature used by the ML pipeline must be registered here with metadata
that proves it is safe for online scoring.  Training should fail if any
registered feature has uses_label=True or uses_future_data=True.

This makes leakage prevention structural rather than relying on developer
discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PrivacyClass(Enum):
    SAFE = "safe"             # derived from pre-event data only
    DERIVED = "derived"       # computed from other safe features
    AGGREGATE = "aggregate"   # historical aggregate (point-in-time)


@dataclass(frozen=True)
class FeatureMeta:
    """Metadata for one feature in the §16 vector."""
    name: str
    source: str                    # where the data comes from
    event_time_dependency: str     # "before_event" | "at_event" | "aggregated"
    allowed_at_scoring_time: bool  # must be True for online scoring
    uses_label: bool               # True = LEAKAGE — must not appear in scoring
    uses_future_data: bool         # True = LEAKAGE — must not appear in scoring
    uses_post_event_outcome: bool  # True = LEAKAGE — must not appear in scoring
    privacy_class: PrivacyClass = PrivacyClass.SAFE


# ============================================================
# SECTION 16 FEATURE REGISTRY
# ============================================================
# All features must satisfy:
#   allowed_at_scoring_time == True
#   uses_label == False
#   uses_future_data == False
#   uses_post_event_outcome == False
#
# Training MUST reject any feature that violates these constraints.
# ============================================================

FEATURE_REGISTRY: list[FeatureMeta] = [
    FeatureMeta(
        name="amount_ratio",
        source="event.amount / profile.median_amount",
        event_time_dependency="at_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    FeatureMeta(
        name="txn_freq_last_24h",
        source="count of committed events in 24h before this event",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.AGGREGATE,
    ),
    FeatureMeta(
        name="txn_time_unusual",
        source="event.hour_of_day not in profile.typical_hours",
        event_time_dependency="at_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    FeatureMeta(
        name="new_device_flag",
        source="event.device_hash not in profile.known_devices (pre-event)",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    FeatureMeta(
        name="unusual_location_flag",
        source="event.location_id not in profile.usual_locations",
        event_time_dependency="at_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    FeatureMeta(
        name="unusual_recipient_flag",
        source="event.recipient_id not in profile.usual_recipients",
        event_time_dependency="at_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    FeatureMeta(
        name="failed_auth_count_24h",
        source="count of failed authentications in 24h before event",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.AGGREGATE,
    ),
    FeatureMeta(
        name="days_since_last_similar_txn",
        source="time delta to most recent similar transaction before this event",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.AGGREGATE,
    ),
    FeatureMeta(
        name="gradual_escalation_score",
        source="slope of log(amount_ratio) over recent window",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    FeatureMeta(
        name="known_device_count",
        source="count of devices registered to account before this event",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.AGGREGATE,
    ),
    FeatureMeta(
        name="account_tenure_days",
        source="days since profile creation (before this event)",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.AGGREGATE,
    ),
    FeatureMeta(
        name="hour_of_day",
        source="event timestamp hour",
        event_time_dependency="at_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.SAFE,
    ),
    FeatureMeta(
        name="is_weekend",
        source="event timestamp day-of-week >= 5",
        event_time_dependency="at_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.SAFE,
    ),
    FeatureMeta(
        name="shared_device_accounts",
        source="count of OTHER accounts seen on this device (link analysis)",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.AGGREGATE,
    ),
    FeatureMeta(
        name="shared_recipient_accounts",
        source="count of OTHER accounts seen on this recipient (link analysis)",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.AGGREGATE,
    ),
    FeatureMeta(
        name="mule_ring_score",
        source="normalized composite of shared_device + shared_recipient",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    # ── Deviation features ─────────────────────────────────────────────────
    # Continuous scores that complement the binary flags. Derived from
    # profile history and the current event — no future data or labels.
    FeatureMeta(
        name="hour_deviation",
        source="circular distance from event.hour to nearest typical hour",
        event_time_dependency="at_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    FeatureMeta(
        name="amount_zscore",
        source="signed z-score of amount vs account history",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    FeatureMeta(
        name="velocity_deviation",
        source="sigmoid of z-score of freq vs account baseline",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    FeatureMeta(
        name="recipient_novelty",
        source="fraction of recent recipients new to account",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
    FeatureMeta(
        name="txn_regularity",
        source="coefficient of variation of inter-arrival times",
        event_time_dependency="before_event",
        allowed_at_scoring_time=True,
        uses_label=False,
        uses_future_data=False,
        uses_post_event_outcome=False,
        privacy_class=PrivacyClass.DERIVED,
    ),
]

# ============================================================
# ILLEGAL FEATURES — these must NEVER appear in the feature vector
# ============================================================
# If any of these names appear in the feature list, training must fail.

ILLEGAL_FEATURES = {
    # Label-derived
    "label", "fraud_label", "is_fraud", "target", "y",
    # Post-event outcomes
    "chargeback", "dispute_outcome", "verification_result",
    "investigation_result", "confirmed_fraud", "was_fraud",
    # Future data
    "future_transaction_count", "next_transaction_amount",
    "subsequent_fraud_flag",
    # Raw PII
    "email", "phone", "address", "full_name", "user_id",
    "account_number", "ssn", "ip_address",
    # Raw financial (not ratio-based)
    "raw_amount", "transaction_amount", "actual_amount",
    "raw_median_amount",
}


def validate_feature_list(feature_names: list[str]) -> list[str]:
    """Validate a feature list against the registry.

    Returns a list of error messages (empty = valid).
    """
    errors = []
    registry_names = {f.name for f in FEATURE_REGISTRY}

    for name in feature_names:
        if name in ILLEGAL_FEATURES:
            errors.append(
                f"LEAKAGE: Feature '{name}' is in the ILLEGAL_FEATURES set "
                f"and must not appear in the feature vector."
            )
        if name not in registry_names:
            errors.append(
                f"UNREGISTERED: Feature '{name}' is not in the FEATURE_REGISTRY. "
                f"All features must be registered with metadata."
            )

    # Check for features in registry that are marked unsafe
    for feat in FEATURE_REGISTRY:
        if feat.name in feature_names:
            if feat.uses_label:
                errors.append(
                    f"LEAKAGE: Feature '{feat.name}' has uses_label=True."
                )
            if feat.uses_future_data:
                errors.append(
                    f"LEAKAGE: Feature '{feat.name}' has uses_future_data=True."
                )
            if feat.uses_post_event_outcome:
                errors.append(
                    f"LEAKAGE: Feature '{feat.name}' has uses_post_event_outcome=True."
                )
            if not feat.allowed_at_scoring_time:
                errors.append(
                    f"SCORING: Feature '{feat.name}' is not allowed at scoring time."
                )

    return errors


def get_scoring_features() -> list[str]:
    """Return only features that are safe for online scoring."""
    return [
        f.name for f in FEATURE_REGISTRY
        if f.allowed_at_scoring_time
        and not f.uses_label
        and not f.uses_future_data
        and not f.uses_post_event_outcome
    ]
