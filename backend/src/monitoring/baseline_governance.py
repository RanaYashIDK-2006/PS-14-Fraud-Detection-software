"""Baseline governance — explicit metadata, approval, and source classification for drift baselines.

A drift baseline MUST have metadata. Baseline sources must be classified.
Baselines cannot silently switch to unapproved datasets.

Allowed baseline sources:
  - approved training window
  - approved validation/production-valid reference window
  - explicitly approved monitoring reference window

Forbidden baseline sources:
  - external test-set-only baseline
  - future data
  - label-derived features
  - post-outcome information
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class BaselineSource(str, Enum):
    TRAINING_WINDOW = "training_window"
    VALIDATION_WINDOW = "validation_window"
    APPROVED_REFERENCE = "approved_reference"
    UNKNOWN = "unknown"
    FORBIDDEN = "forbidden"  # external test set, future data, etc.


class BaselineStatus(str, Enum):
    ACTIVE = "active"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass
class BaselineMetadata:
    """Complete metadata for a drift baseline."""
    baseline_id: str
    model_version: str
    feature_schema_version: str
    created_at: float = field(default_factory=time.time)
    source_type: BaselineSource = BaselineSource.UNKNOWN
    source_description: str = ""
    approved_by: str = ""
    approved_at: float | None = None
    status: BaselineStatus = BaselineStatus.APPROVED
    sample_count: int = 0
    feature_definitions: dict[str, str] = field(default_factory=dict)
    binning_config: dict[str, Any] = field(default_factory=dict)
    threshold_config: dict[str, Any] = field(default_factory=dict)
    code_version: str = ""
    dataset_identifier: str = ""
    dataset_hash: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        d["status"] = self.status.value
        return d

    def save(self, path: Path) -> None:
        """Save baseline metadata to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BaselineMetadata:
        """Load baseline metadata from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        data["source_type"] = BaselineSource(data.get("source_type", "unknown"))
        data["status"] = BaselineStatus(data.get("status", "approved"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def validate_baseline_source(
    source_type: BaselineSource,
    source_description: str = "",
) -> tuple[bool, str]:
    """Validate that a baseline source is allowed.

    Returns (is_valid, reason).
    """
    if source_type == BaselineSource.FORBIDDEN:
        return False, "forbidden source type (external test set, future data, etc.)"

    if source_type == BaselineSource.UNKNOWN:
        return False, "source type is unknown — must be explicitly classified"

    if source_type == BaselineSource.TRAINING_WINDOW:
        if not source_description:
            return False, "training window baseline must include source description"
        return True, "training window baseline is allowed"

    if source_type == BaselineSource.VALIDATION_WINDOW:
        return True, "validation window baseline is allowed"

    if source_type == BaselineSource.APPROVED_REFERENCE:
        if not source_description:
            return False, "approved reference must include source description"
        return True, "approved reference baseline is allowed"

    return False, f"unhandled source type: {source_type}"


def create_baseline(
    baseline_id: str,
    model_version: str,
    feature_schema_version: str,
    source_type: BaselineSource,
    source_description: str,
    sample_count: int,
    approved_by: str = "",
    feature_definitions: dict[str, str] | None = None,
    binning_config: dict[str, Any] | None = None,
    threshold_config: dict[str, Any] | None = None,
    code_version: str = "",
    dataset_identifier: str = "",
    dataset_hash: str = "",
) -> BaselineMetadata:
    """Create a new baseline with validated metadata.

    Raises ValueError if the source is forbidden.
    """
    is_valid, reason = validate_baseline_source(source_type, source_description)
    if not is_valid:
        raise ValueError(f"baseline creation rejected: {reason}")

    now = time.time()
    metadata = BaselineMetadata(
        baseline_id=baseline_id,
        model_version=model_version,
        feature_schema_version=feature_schema_version,
        created_at=now,
        source_type=source_type,
        source_description=source_description,
        approved_by=approved_by,
        approved_at=now if approved_by else None,
        status=BaselineStatus.APPROVED if approved_by else BaselineStatus.ACTIVE,
        sample_count=sample_count,
        feature_definitions=feature_definitions or {},
        binning_config=binning_config or {},
        threshold_config=threshold_config or {},
        code_version=code_version,
        dataset_identifier=dataset_identifier,
        dataset_hash=dataset_hash,
    )
    return metadata


def supersede_baseline(
    old_metadata: BaselineMetadata,
    new_metadata: BaselineMetadata,
) -> tuple[BaselineMetadata, BaselineMetadata]:
    """Mark an old baseline as superseded and return both."""
    old_metadata.status = BaselineStatus.SUPERSEDED
    return old_metadata, new_metadata
