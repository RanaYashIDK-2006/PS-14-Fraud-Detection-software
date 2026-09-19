"""Phase 84: Dataset ingestion, inventory, classification & candidate construction.

Minimal, governed path for turning external real-world fraud datasets into
DatasetCandidate descriptors that can be evaluated by the Phase 82/83
admission system.

Does NOT train models, modify production, or bypass trust governance.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.monitoring.dataset_admission import (
    DatasetCandidate,
    FeatureSchemaDescriptor,
    FeatureCompatibilityState,
    evaluate_feature_compatibility,
    make_dataset_candidate,
)
from src.monitoring.feature_contract import ML_FEATURE_ORDER, ML_FEATURE_VERSION


# ── Dataset classification ────────────────────────────────────────────

class DatasetClassification(str, Enum):
    SYNTHETIC = "synthetic"
    RESEARCH_DERIVED = "research_derived"
    PUBLIC_REAL_WORLD = "public_real_world"
    UNKNOWN = "unknown"


class LabelProvenance(str, Enum):
    INDEPENDENT_HUMAN = "independent_human"
    INDEPENDENT_INSTITUTIONAL = "independent_institutional"
    MODEL_DERIVED = "model_derived"
    SYNTHETIC_GENERATED = "synthetic_generated"
    UNKNOWN = "unknown"


# ── Dataset metadata ──────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetProvenance:
    """Immutable provenance record for an external dataset."""
    dataset_name: str
    classification: str  # DatasetClassification value
    source_url: str = ""
    source_doi: str = ""
    license: str = "unknown"
    label_field: str = ""
    label_provenance: str = LabelProvenance.UNKNOWN.value
    label_vocabulary: tuple[str, ...] = ()
    timestamp_field: str = ""
    has_timestamps: bool = False
    row_count: int = 0
    source_hash: str = ""  # SHA-256 of source file
    schema_hash: str = ""  # SHA-256 of header row
    schema_columns: tuple[str, ...] = ()
    source_description: str = ""
    acquisition_date: str = ""
    is_committed: bool = False  # whether data is in the repo

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "classification": self.classification,
            "source_url": self.source_url,
            "license": self.license,
            "label_field": self.label_field,
            "label_provenance": self.label_provenance,
            "label_vocabulary": list(self.label_vocabulary),
            "timestamp_field": self.timestamp_field,
            "has_timestamps": self.has_timestamps,
            "row_count": self.row_count,
            "source_hash": self.source_hash,
            "schema_hash": self.schema_hash,
            "schema_columns": list(self.schema_columns),
            "source_description": self.source_description,
        }


# ── Source hashing ────────────────────────────────────────────────────

def compute_file_hash(filepath: str | Path, chunk_size: int = 65536) -> str:
    """Deterministic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_header_hash(filepath: str | Path) -> str:
    """SHA-256 hash of just the header row (deterministic schema fingerprint)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
    return hashlib.sha256(header.encode("utf-8")).hexdigest()


def count_rows(filepath: str | Path) -> int:
    """Count data rows (excluding header)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        total = sum(1 for _ in f)
    return max(0, total - 1)  # subtract header, floor at 0


def extract_columns(filepath: str | Path) -> tuple[str, ...]:
    """Extract column names from CSV header.

    Handles quoted CSV headers by stripping surrounding quotes.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
    # Use csv module to properly handle quoted headers
    reader = csv.reader(io.StringIO(header))
    for row in reader:
        return tuple(col.strip() for col in row)
    return tuple(header.split(","))


# ── Schema extraction ─────────────────────────────────────────────────

def extract_feature_schema(
    columns: tuple[str, ...],
    label_field: str = "",
    timestamp_field: str = "",
) -> FeatureSchemaDescriptor:
    """Extract a FeatureSchemaDescriptor from raw CSV columns.

    Does NOT modify columns. Identifies feature columns by excluding
    label and timestamp fields.
    """
    # Exclude label and timestamp fields
    exclude = {label_field, timestamp_field, ""}
    feature_names = tuple(c for c in columns if c not in exclude)

    # Infer types from first data row (simple heuristic)
    feature_types = tuple("float" for _ in feature_names)

    return FeatureSchemaDescriptor(
        schema_id="external_csv",
        feature_count=len(feature_names),
        feature_names=feature_names,
        feature_types=feature_types,
        feature_version="",
        source_description="Extracted from CSV columns",
    )


# ── Known dataset inventory ───────────────────────────────────────────

KNOWN_DATASETS: dict[str, DatasetProvenance] = {
    "ulb_creditcard": DatasetProvenance(
        dataset_name="ULB Credit Card Fraud Detection",
        classification=DatasetClassification.SYNTHETIC.value,
        source_url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        license="unknown (Kaggle dataset)",
        label_field="Class",
        label_provenance=LabelProvenance.UNKNOWN.value,
        label_vocabulary=("0", "1"),  # 0=legit, 1=fraud
        timestamp_field="Time",
        has_timestamps=True,
        source_description=(
            "ULB/Machine Learning Group credit card fraud dataset. "
            "PCA-transformed V1-V28 features. Not directly interpretable. "
            "INCOMPATIBLE with PS-14 feature contract."
        ),
    ),
    "ibm_v2": DatasetProvenance(
        dataset_name="IBM Credit Card Transactions v2",
        classification=DatasetClassification.SYNTHETIC.value,
        source_url="https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions",
        license="unknown (Kaggle dataset)",
        label_field="Is Fraud?",
        label_provenance=LabelProvenance.SYNTHETIC_GENERATED.value,
        label_vocabulary=("No", "Yes"),
        timestamp_field="",
        has_timestamps=True,  # Year,Month,Day,Time columns
        source_description=(
            "IBM-generated synthetic credit card transactions. "
            "24M+ rows. Labels are synthetically generated, not from real fraud investigations."
        ),
    ),
    "paysim": DatasetProvenance(
        dataset_name="PaySim Mobile Money Simulator",
        classification=DatasetClassification.SYNTHETIC.value,
        source_url="https://www.kaggle.com/datasets/ntnu-testimon/paysim1",
        license="unknown",
        label_field="isFraud",
        label_provenance=LabelProvenance.SYNTHETIC_GENERATED.value,
        label_vocabulary=("0", "1"),
        timestamp_field="",
        has_timestamps=False,
        source_description=(
            "PaySim: mobile money transaction simulator based on real dataset "
            "from a mobile money service. Fraud labels are simulated."
        ),
    ),
    "kaggle_fraud_train": DatasetProvenance(
        dataset_name="Kaggle Fraud Detection (train)",
        classification=DatasetClassification.SYNTHETIC.value,
        source_url="https://www.kaggle.com/datasets/ datasets/credit-card-fraud-detection",
        license="unknown",
        label_field="is_fraud",
        label_provenance=LabelProvenance.MODEL_DERIVED.value,
        label_vocabulary=("0", "1"),
        timestamp_field="trans_date_trans_time",
        has_timestamps=True,
        source_description=(
            "Kaggle fraudTrain.csv. Derived from ULB-style PCA features. "
            "Labels appear to be model-derived (Isolation Forest or similar). "
            "NOT independent ground truth."
        ),
    ),
}


# ── Ingestion adapter ─────────────────────────────────────────────────

class DatasetIngestionError(Exception):
    """Raised when dataset ingestion fails."""
    pass


@dataclass
class IngestionResult:
    """Result of ingesting an external dataset."""
    provenance: DatasetProvenance
    candidate: DatasetCandidate
    feature_schema: FeatureSchemaDescriptor
    feature_compatibility: dict[str, Any]
    ingestion_time: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "candidate": {
                "dataset_id": self.candidate.dataset_id,
                "dataset_version": self.candidate.dataset_version,
                "feature_schema_id": self.candidate.feature_schema_id,
            },
            "feature_schema": self.feature_schema.to_dict(),
            "feature_compatibility": self.feature_compatibility,
            "ingestion_time": self.ingestion_time,
        }


def ingest_dataset(
    filepath: str | Path,
    provenance: DatasetProvenance | None = None,
    dataset_name: str = "",
) -> IngestionResult:
    """Minimal deterministic ingestion of a CSV dataset.

    Reads the file, extracts schema, computes hashes, builds a
    DatasetCandidate, and evaluates feature compatibility.

    Does NOT create OutcomeRecords. Does NOT bypass trust governance.
    Does NOT transform features.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise DatasetIngestionError(f"file not found: {filepath}")

    # 1. Compute source hash
    source_hash = compute_file_hash(filepath)

    # 2. Extract columns and schema
    columns = extract_columns(filepath)
    schema_hash = compute_header_hash(filepath)
    row_count = count_rows(filepath)

    # 3. Build or update provenance
    if provenance is None:
        provenance = DatasetProvenance(
            dataset_name=dataset_name or filepath.stem,
            classification=DatasetClassification.UNKNOWN.value,
            row_count=row_count,
            source_hash=source_hash,
            schema_hash=schema_hash,
            schema_columns=columns,
        )
    else:
        # Update computed fields
        provenance = DatasetProvenance(
            dataset_name=provenance.dataset_name,
            classification=provenance.classification,
            source_url=provenance.source_url,
            source_doi=provenance.source_doi,
            license=provenance.license,
            label_field=provenance.label_field,
            label_provenance=provenance.label_provenance,
            label_vocabulary=provenance.label_vocabulary,
            timestamp_field=provenance.timestamp_field,
            has_timestamps=provenance.has_timestamps,
            row_count=row_count,
            source_hash=source_hash,
            schema_hash=schema_hash,
            schema_columns=columns,
            source_description=provenance.source_description,
            acquisition_date=provenance.acquisition_date,
            is_committed=provenance.is_committed,
        )

    # 4. Extract feature schema
    feature_schema = extract_feature_schema(
        columns,
        label_field=provenance.label_field,
        timestamp_field=provenance.timestamp_field,
    )

    # 5. Evaluate feature compatibility
    feature_compat = evaluate_feature_compatibility(feature_schema)

    # 6. Build candidate
    candidate = make_dataset_candidate(
        dataset_version=f"1.0_{source_hash[:8]}",
        feature_schema_id=provenance.dataset_name.lower().replace(" ", "_"),
    )

    return IngestionResult(
        provenance=provenance,
        candidate=candidate,
        feature_schema=feature_schema,
        feature_compatibility=feature_compat,
        ingestion_time=datetime.now(timezone.utc).isoformat(),
    )


# ── Inventory scanner ─────────────────────────────────────────────────

def scan_data_directory(data_dir: str | Path) -> list[DatasetProvenance]:
    """Scan a data directory and classify discovered datasets."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []

    results: list[DatasetProvenance] = []

    for csv_file in sorted(data_dir.glob("*.csv")):
        try:
            columns = extract_columns(csv_file)
            row_count = count_rows(csv_file)
            source_hash = compute_file_hash(csv_file)
            schema_hash = compute_header_hash(csv_file)

            # Classify based on schema
            classification = _classify_by_schema(columns, csv_file.name)
            label_field = _detect_label_field(columns)

            provenance = DatasetProvenance(
                dataset_name=csv_file.stem,
                classification=classification.value,
                label_field=label_field,
                label_provenance=_infer_label_provenance(classification, csv_file.name).value,
                row_count=row_count,
                source_hash=source_hash,
                schema_hash=schema_hash,
                schema_columns=columns,
                source_description=f"Auto-classified: {classification.value}",
            )
            results.append(provenance)
        except Exception:
            continue

    return results


def _classify_by_schema(columns: tuple[str, ...], filename: str) -> DatasetClassification:
    """Classify dataset by schema heuristics."""
    col_set = set(c.lower() for c in columns)

    # ULB PCA pattern
    if "v1" in col_set and "v28" in col_set and "class" in col_set:
        return DatasetClassification.SYNTHETIC

    # PaySim
    if "isfraud" in col_set and "type" in col_set and "oldbalanceorg" in col_set:
        return DatasetClassification.SYNTHETIC

    # IBM v2
    if "is fraud?" in col_set or "is_fraud" in col_set:
        if "user" in col_set and "card" in col_set:
            return DatasetClassification.SYNTHETIC

    # PS-14 synthetic features
    ps14_features = {"amount_ratio", "txn_freq_last_24h", "new_device_flag"}
    if ps14_features.issubset(col_set):
        return DatasetClassification.SYNTHETIC

    # Kaggle fraud with is_fraud
    if "is_fraud" in col_set and "cc_num" in col_set:
        return DatasetClassification.SYNTHETIC

    return DatasetClassification.UNKNOWN


def _detect_label_field(columns: tuple[str, ...]) -> str:
    """Detect the label column."""
    label_names = {"class", "isfraud", "is_fraud", "is fraud?", "fraud", "label"}
    for c in columns:
        if c.lower() in label_names:
            return c
    return ""


def _infer_label_provenance(
    classification: DatasetClassification, filename: str
) -> LabelProvenance:
    """Infer label provenance from classification."""
    if classification == DatasetClassification.SYNTHETIC:
        return LabelProvenance.SYNTHETIC_GENERATED
    if classification == DatasetClassification.RESEARCH_DERIVED:
        return LabelProvenance.UNKNOWN
    return LabelProvenance.UNKNOWN
