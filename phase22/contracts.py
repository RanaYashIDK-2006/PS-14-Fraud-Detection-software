"""Data contracts, schema mapping, and metadata validation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Real-world data contract — minimum metadata for a validation-eligible dataset
# ---------------------------------------------------------------------------

REQUIRED_METADATA_KEYS = [
    "dataset_name",
    "source",
    "owner",
    "real_world_status",      # "REAL_WORLD" | "SYNTHETIC" | "UNKNOWN"
    "authorization_status",   # "AUTHORIZED" | "UNAUTHORIZED" | "UNVERIFIED"
    "label_definition",
    "label_source",
    "transaction_time_field",
]

OPTIONAL_METADATA_KEYS = [
    "collection_period",
    "label_available_time_field",
    "label_finalization_time_field",
    "decision_time_field",
    "channel_field",
    "merchant_id_field",
    "user_id_field",
    "mcc_field",
    "amount_field",
]


@dataclass
class MetadataContract:
    """Validated metadata for a real-world dataset."""
    dataset_name: str
    source: str
    owner: str
    real_world_status: str
    authorization_status: str
    label_definition: str
    label_source: str
    transaction_time_field: str
    label_available_time_field: Optional[str] = None
    label_finalization_time_field: Optional[str] = None
    decision_time_field: Optional[str] = None
    channel_field: Optional[str] = None
    merchant_id_field: Optional[str] = None
    user_id_field: Optional[str] = None
    mcc_field: Optional[str] = None
    amount_field: Optional[str] = None
    collection_period: Optional[dict] = None

    @classmethod
    def from_json(cls, path: Path) -> "MetadataContract":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if self.real_world_status not in ("REAL_WORLD",):
            errors.append(f"real_world_status={self.real_world_status!r} — must be REAL_WORLD")
        if self.authorization_status not in ("AUTHORIZED",):
            errors.append(f"authorization_status={self.authorization_status!r} — must be AUTHORIZED")
        if not self.label_definition:
            errors.append("label_definition is empty")
        if not self.label_source:
            errors.append("label_source is empty")
        return errors


# ---------------------------------------------------------------------------
# Schema mapping — real-world field → PS-14 canonical field
# ---------------------------------------------------------------------------

# Default mapping (IBM v2 columns → PS-14 canonical)
DEFAULT_SCHEMA_MAP: dict[str, str] = {
    "timestamp": "ts",
    "amount": "amount",
    "merchant_id": "merchant_id",
    "user_id": "account_id",
    "channel": "channel",
    "mcc": "mcc",
    "fraud_label": "fraud",
    "city": "city",
    "country": "country",
}

P20_REQUIRED_FEATURES = [
    "amt", "log_amt", "amt_sq", "hr", "mn", "is_weekend",
    "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
    "unusual_location_flag", "unusual_recipient_flag",
    "failed_auth_count_24h", "days_since_last_similar_txn",
    "gradual_escalation_score", "known_device_count",
    "account_tenure_days", "hour_of_day", "is_weekend",
    "shared_device_accounts", "shared_recipient_accounts",
    "mule_ring_score", "hour_deviation", "amount_zscore",
    "velocity_deviation", "recipient_novelty", "txn_regularity",
    # Additional features from the 45-feature contract
    "txn_amount_bucket", "amount_ratio", "category_encoded",
    "merchant_risk_score", "device_age_days", "location_distance",
    "amount_pct_of_avg", "txn_count_7d", "txn_count_30d",
    "avg_amount_7d", "avg_amount_30d", "max_amount_7d",
    "unique_merchants_7d", "unique_merchants_30d",
    "night_txn_ratio", "weekend_txn_ratio",
    "same_merchant_count_24h", "cross_border_flag",
    "card_age_days", "prev_fraud_flag",
    "merchant_fraud_history_30d", "city_population_bucket",
]

E_HARDNEG_EXTRA_FEATURES = [
    "user_fraud_rate",
    "merch_fraud_rate",
    "city_fraud_rate",
]


# ---------------------------------------------------------------------------
# Dataset hashing
# ---------------------------------------------------------------------------

def compute_file_hash(path: Path, chunk_size: int = 8192) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def compute_string_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
