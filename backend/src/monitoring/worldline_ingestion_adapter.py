"""Phase 90: Dormant Worldline ingestion adapter.

Accepts a local authorized dataset path and produces a pre-admission
report.  Does NOT connect to Worldline, download data, authenticate,
modify models, or change REAL_WORLD_VALIDATION.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class IngestionStatus(str, Enum):
    NOT_RECEIVED = "not_received"
    FILE_NOT_FOUND = "file_not_found"
    SCHEMA_VALIDATED = "schema_validated"
    SCHEMA_INVALID = "schema_invalid"
    PRE_ADMITTED = "pre_admitted"


class FieldCheckResult(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IngestionReport:
    """Immutable pre-admission report from dormant adapter."""
    adapter_id: str
    adapter_version: str
    source_path: str
    source_hash: str
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    ingestion_status: str
    field_checks: dict[str, str]
    identity_match: str
    schema_compatibility: str
    timestamp_fields: tuple[str, ...]
    has_numeric_amount: bool
    has_identifier_fields: bool
    has_label_field: bool
    pre_admission_result: str
    blockers: tuple[str, ...]
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["columns"] = list(d["columns"])
        d["timestamp_fields"] = list(d["timestamp_fields"])
        d["field_checks"] = dict(d["field_checks"])
        d["blockers"] = list(d["blockers"])
        return d


def _compute_file_hash(path: str) -> str:
    """Deterministic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_columns(header_line: str) -> tuple[str, ...]:
    """Parse CSV header into column names."""
    cols = [c.strip().strip('"').strip("'") for c in header_line.split(",")]
    return tuple(c for c in cols if c)


# Known Worldline NAG column names (from published research, approximate)
WORLDLINE_NAG_EXPECTED_COLUMNS = frozenset({
    "TransactionID", "CustomerID", "TerminalID",
    "TransactionAmount", "TransactionDate",
    "Fraud", "fraud", "isFraud", "label",
})


def ingest_local_dataset(
    file_path: str,
    expected_dataset_id: str = "WORLDLINE_ECOM_2017_NAG",
) -> IngestionReport:
    """Dormant ingestion adapter: inspect a local file without connecting
    to any external service.

    Does NOT download, authenticate, or connect to Worldline.
    Accepts only a local file path supplied by the user/operator.
    """
    adapter_id = "WORLDLINE-DORMANT-INGESTION-90"
    adapter_version = "phase90_v1"
    blockers: list[str] = []
    field_checks: dict[str, str] = {}

    # File existence check
    if not os.path.isfile(file_path):
        return IngestionReport(
            adapter_id=adapter_id, adapter_version=adapter_version,
            source_path=file_path, source_hash="", row_count=0,
            column_count=0, columns=(), ingestion_status=IngestionStatus.FILE_NOT_FOUND.value,
            field_checks={}, identity_match="file_not_found",
            schema_compatibility="not_evaluated", timestamp_fields=(),
            has_numeric_amount=False, has_identifier_fields=False,
            has_label_field=False, pre_admission_result=IngestionStatus.FILE_NOT_FOUND.value,
            blockers=("File not found",),
            report_hash="",
        )

    # Hash the file
    source_hash = _compute_file_hash(file_path)

    # Read header
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline().strip()
            columns = _detect_columns(header_line)
            # Count rows
            row_count = sum(1 for _ in f)
    except Exception as e:
        return IngestionReport(
            adapter_id=adapter_id, adapter_version=adapter_version,
            source_path=file_path, source_hash=source_hash, row_count=0,
            column_count=0, columns=(), ingestion_status=IngestionStatus.SCHEMA_INVALID.value,
            field_checks={}, identity_match="read_error",
            schema_compatibility="error", timestamp_fields=(),
            has_numeric_amount=False, has_identifier_fields=False,
            has_label_field=False, pre_admission_result=IngestionStatus.SCHEMA_INVALID.value,
            blockers=(f"Read error: {e}",),
            report_hash="",
        )

    cols_lower = {c.lower(): c for c in columns}

    # Field detection
    def _check(name_variants: list[str]) -> FieldCheckResult:
        for v in name_variants:
            if v.lower() in cols_lower:
                return FieldCheckResult.PRESENT
        return FieldCheckResult.ABSENT

    # Timestamp fields
    ts_candidates = ["transactiondate", "timestamp", "date", "time", "ts",
                     "transactiondt", "trans_date_trans_time"]
    timestamp_fields = tuple(
        cols_lower[t] for t in ts_candidates if t in cols_lower
    )
    has_timestamps = len(timestamp_fields) > 0
    field_checks["timestamps"] = "present" if has_timestamps else "absent"

    # Amount
    amt_check = _check(["transactionamount", "amount", "amt", "transaction_amt"])
    has_amount = amt_check == FieldCheckResult.PRESENT
    field_checks["amount"] = amt_check.value

    # Identifiers
    customer_check = _check(["customerid", "customer_id", "user_id", "userid",
                             "card1", "card_id"])
    merchant_check = _check(["merchantid", "merchant_id", "terminalid", "terminal_id"])
    city_check = _check(["city_id", "city", "merchant_city", "addr1"])
    card_check = _check(["card_id", "card", "card1"])
    mcc_check = _check(["mcc", "merchant_category", "category_code"])
    channel_check = _check(["use_chip", "channel", "channel_type", "transaction_type"])

    field_checks["customer_id"] = customer_check.value
    field_checks["merchant_id"] = merchant_check.value
    field_checks["city_id"] = city_check.value
    field_checks["card_id"] = card_check.value
    field_checks["mcc"] = mcc_check.value
    field_checks["channel"] = channel_check.value

    has_identifiers = customer_check == FieldCheckResult.PRESENT

    # Labels
    label_check = _check(["fraud", "isfraud", "label", "is_fraud", "target"])
    has_labels = label_check == FieldCheckResult.PRESENT
    field_checks["fraud_label"] = label_check.value

    # Identity match
    identity_match = "unverified"
    if expected_dataset_id in ("WORLDLINE_ECOM_2017_NAG",):
        # Can only match by column pattern, not actual content
        if has_timestamps and has_amount:
            identity_match = "pattern_compatible"
        else:
            identity_match = "pattern_incompatible"

    # Schema compatibility
    schema_compat = "unknown"
    if has_timestamps and has_amount and has_identifiers:
        schema_compat = "potentially_compatible"
    elif has_timestamps and has_amount:
        schema_compat = "partial"
    else:
        schema_compat = "incompatible"

    # Blockers
    if not has_timestamps:
        blockers.append("No timestamp fields detected")
    if not has_amount:
        blockers.append("No amount fields detected")
    if not has_identifiers:
        blockers.append("No customer/user identifier detected")
    if not has_labels:
        blockers.append("No fraud label field detected")

    # Determine status
    if schema_compat == "potentially_compatible" and has_labels:
        status = IngestionStatus.PRE_ADMITTED.value
        pre_admission = "eligible_for_phase85_admission_pending_verification"
    elif schema_compat in ("partial",) and has_labels:
        status = IngestionStatus.SCHEMA_VALIDATED.value
        pre_admission = "partial_schema_compatible_pending_verification"
    else:
        status = IngestionStatus.SCHEMA_INVALID.value
        pre_admission = "not_eligible_schema_incompatible"

    # Build report hash
    report_content = {
        "adapter_id": adapter_id,
        "source_hash": source_hash,
        "row_count": row_count,
        "column_count": len(columns),
        "identity_match": identity_match,
        "schema_compatibility": schema_compat,
        "pre_admission_result": pre_admission,
    }
    canonical = json.dumps(report_content, sort_keys=True, separators=(",", ":"))
    report_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return IngestionReport(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        source_path=file_path,
        source_hash=source_hash,
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
        ingestion_status=status,
        field_checks=field_checks,
        identity_match=identity_match,
        schema_compatibility=schema_compat,
        timestamp_fields=timestamp_fields,
        has_numeric_amount=has_amount,
        has_identifier_fields=has_identifiers,
        has_label_field=has_labels,
        pre_admission_result=pre_admission,
        blockers=tuple(blockers),
        report_hash=report_hash,
    )
