"""External dataset intake contract — metadata schema for dataset evaluation.

Defines the minimum information required before external evaluation can begin.
Every field is documented; unknown fields are preserved as UNKNOWN rather than
silently treated as valid.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class ProvenanceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    UNKNOWN = "UNKNOWN"
    SUSPECTED_SYNTHETIC = "SUSPECTED_SYNTHETIC"
    CONFIRMED_SYNTHETIC = "CONFIRMED_SYNTHETIC"


class DatasetEligibility(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    CONDITIONALLY_ELIGIBLE = "CONDITIONALLY_ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNKNOWN = "UNKNOWN"
    BLOCKED = "BLOCKED"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class FeatureProvenance:
    """Provenance metadata for a single feature."""
    feature_name: str
    description: str = ""
    source: str = ""
    availability_time: str = ""  # e.g., "at_event_time", "post_event", "unknown"
    event_time_relation: str = ""  # e.g., "before", "after", "at", "unknown"
    post_event_possible: bool = False
    allowed_for_scoring: bool = True
    reason: str = ""


@dataclass
class ExternalDatasetMetadata:
    """Complete metadata contract for an external dataset.

    Every field must be explicitly provided or marked UNKNOWN.
    Fields marked UNKNOWN prevent ELIGIBLE classification.
    """
    # Dataset identity
    name: str = ""
    source: str = ""
    original_url: str = ""
    license: str = ""
    collection_description: str = ""
    owner_publisher: str = ""

    # Provenance
    provenance_status: str = ProvenanceStatus.UNKNOWN.value
    provenance_notes: str = ""

    # Temporal
    date_acquired: str = ""
    date_range_start: str = ""
    date_range_end: str = ""
    geography: str = ""

    # Transaction context
    transaction_channel_type: str = ""  # e.g., "card", "wire", "mixed", "unknown"

    # Labels
    label_definition: str = ""
    fraud_label_timing: str = ""  # e.g., "at_transaction", "post_investigation", "unknown"
    label_column: str = "is_fraud"
    positive_class_value: Any = 1
    negative_class_value: Any = 0

    # Features
    feature_definitions: list[FeatureProvenance] = field(default_factory=list)
    feature_availability_at_decision_time: str = "unknown"

    # Dataset statistics
    num_rows: int = 0
    num_positive: int = 0
    num_negative: int = 0
    missing_value_pct: float = 0.0
    duplicate_pct: float = 0.0

    # Schema
    columns: list[str] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)

    # Known transformations
    known_transformations: str = ""
    preprocessing_performed: str = ""

    # Classification flags
    is_synthetic: bool = False
    is_derived: bool = False
    identifiers_anonymized: bool = False
    train_test_splits_exist: bool = False

    # Evaluation configuration
    feature_mapping_version: str = "1.0"
    notes: str = ""

    def compute_hash(self) -> str:
        """Compute a deterministic hash of the metadata for reproducibility."""
        d = asdict(self)
        # Remove non-deterministic fields
        for k in ["notes", "date_acquired"]:
            d.pop(k, None)
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if not self.name:
            errors.append("dataset name is required")
        if not self.source:
            errors.append("dataset source is required")
        if self.provenance_status == ProvenanceStatus.UNKNOWN.value:
            errors.append("provenance status is UNKNOWN")
        if not self.label_definition:
            errors.append("label definition is required")
        if not self.label_column:
            errors.append("label column name is required")
        if self.num_rows <= 0:
            errors.append("num_rows must be positive")
        if self.num_positive <= 0:
            errors.append("num_positive must be positive (at least one fraud case)")
        if not self.columns:
            errors.append("column list is required")
        return errors

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, data: str | dict) -> ExternalDatasetMetadata:
        """Deserialize from JSON string or dict."""
        if isinstance(data, str):
            data = json.loads(data)
        # Convert nested FeatureProvenance dicts
        features = []
        for f in data.pop("feature_definitions", []):
            if isinstance(f, dict):
                features.append(FeatureProvenance(**f))
            else:
                features.append(f)
        data["feature_definitions"] = features
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> ExternalDatasetMetadata:
        """Load metadata from a JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        """Save metadata to a JSON file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")


# Default schema for the PS-14 §16 feature contract
PS14_ML_FEATURES = [
    "hour_of_day",
    "is_weekend",
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
    "account_daily_spend_ratio",
    "device_daily_count",
    "shared_device_accounts",
    "shared_recipient_accounts",
    "mule_ring_score",
]
