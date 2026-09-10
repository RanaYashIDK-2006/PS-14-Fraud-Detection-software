"""Provenance and governance checks for external datasets.

Verifies dataset origin, authorization, label definition, and
label timing before any evaluation proceeds.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dataset provenance contract
# ---------------------------------------------------------------------------

KNOWN_SOURCES = {
    "kaggle_fraud": {
        "source_name": "Kaggle Credit Card Fraud Detection",
        "provider": "DV (Kaggle, 2024)",
        "owner": "Unknown — published on Kaggle",
        "collection_method": "Unclear — may be real or simulated",
        "real_or_synthetic": "UNCERTAIN",
        "geographic_scope": "US",
        "collection_period": "2019-01-01 to 2020-06-21",
        "license_or_authorization": "CC0 (public domain)",
        "privacy_constraints": "PII present (names, addresses, DOB)",
        "provenance_verdict": "UNVERIFIED",
        "label_definition": "is_fraud — definition undisclosed",
        "label_source": "Unknown — column present in dataset",
        "label_creation_process": "Unknown",
        "label_first_available_time": "Unknown",
        "label_finalization_time": "Unknown",
        "label_revision_process": "Unknown",
    }
}


def verify_dataset_provenance(
    dataset_name: str,
    data_dir: Path,
) -> dict[str, Any]:
    """Verify dataset provenance against known sources.

    Returns a provenance record with status fields.
    """
    source = KNOWN_SOURCES.get(dataset_name)
    if source is None:
        return {
            "status": "UNKNOWN_SOURCE",
            "dataset_name": dataset_name,
            "provenance_verdict": "UNVERIFIED",
            "authorization": "UNVERIFIED",
            "label_definition": "UNKNOWN",
            "label_latency": "UNVERIFIED",
            "real_world_status": "UNKNOWN",
            "proceed": False,
            "reason": f"Dataset '{dataset_name}' not in known sources registry",
        }

    # Compute file hashes
    hashes = {}
    for csv_file in sorted(data_dir.glob("*.csv")):
        h = hashlib.sha256(csv_file.read_bytes()).hexdigest()
        hashes[csv_file.name] = h

    return {
        "status": "VERIFIED_KNOWN_SOURCE",
        "dataset_name": dataset_name,
        **source,
        "file_hashes": hashes,
        "proceed": source["provenance_verdict"] == "VERIFIED",
        "proceed_reason": (
            "Provenance UNVERIFIED — proceeding with explicit caveats"
            if source["provenance_verdict"] != "VERIFIED"
            else "Provenance verified"
        ),
    }


# ---------------------------------------------------------------------------
# Label governance
# ---------------------------------------------------------------------------

def check_label_governance(dataset_name: str) -> dict[str, Any]:
    """Check label governance for the dataset.

    Determines whether labels can be trusted for evaluation.
    """
    source = KNOWN_SOURCES.get(dataset_name, {})
    label_def = source.get("label_definition", "UNKNOWN")
    label_source = source.get("label_source", "Unknown")
    label_latency = source.get("label_first_available_time", "Unknown")

    governance_ok = label_def != "UNKNOWN" and label_latency != "Unknown"

    return {
        "label_name": "is_fraud",
        "label_definition": label_def,
        "label_source": label_source,
        "label_creation_process": source.get("label_creation_process", "Unknown"),
        "label_first_available_time": label_latency,
        "label_finalization_time": source.get("label_finalization_time", "Unknown"),
        "label_revision_process": source.get("label_revision_process", "Unknown"),
        "governance_status": "VERIFIED" if governance_ok else "UNVERIFIED",
        "can_use_as_ground_truth": governance_ok,
        "warning": (
            "Labels are from an unverified source. "
            "Do NOT claim these represent real-world fraud confirmation."
        ),
    }


# ---------------------------------------------------------------------------
# Label latency
# ---------------------------------------------------------------------------

def check_label_latency(dataset_name: str) -> dict[str, Any]:
    """Check whether labels were available at decision time.

    For external datasets, this is typically UNVERIFIED.
    """
    source = KNOWN_SOURCES.get(dataset_name, {})

    return {
        "transaction_time": "present (trans_date_trans_time)",
        "decision_time": "present (unix_time)",
        "label_available_time": source.get("label_first_available_time", "Unknown"),
        "label_finalization_time": source.get("label_finalization_time", "Unknown"),
        "label_latency_status": "UNVERIFIED",
        "causal_validity": "NOT_ESTABLISHED",
        "warning": (
            "Cannot confirm labels were available at transaction decision time. "
            "This evaluation is a benchmark, not a causal production simulation."
        ),
    }


# ---------------------------------------------------------------------------
# Write provenance artifacts
# ---------------------------------------------------------------------------

def write_provenance_artifacts(
    output_dir: Path,
    dataset_name: str,
    data_dir: Path,
) -> dict[str, Any]:
    """Write all provenance/governance artifacts to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = verify_dataset_provenance(dataset_name, data_dir)
    governance = check_label_governance(dataset_name)
    latency = check_label_latency(dataset_name)

    # Write artifacts
    (output_dir / "01_provenance_audit.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "02_label_governance.json").write_text(
        json.dumps(governance, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "03_label_latency.json").write_text(
        json.dumps(latency, indent=2, default=str), encoding="utf-8"
    )

    return {
        "provenance": provenance,
        "governance": governance,
        "latency": latency,
    }
