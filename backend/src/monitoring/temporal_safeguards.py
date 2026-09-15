"""Temporal leakage safeguards — ensures features don't use future information.

Verifies that:
  - feature(t) uses only information available at or before transaction time t
  - future events cannot change historical feature values
  - label/outcome-derived features never enter online inference
  - timestamps are consistent (timezone-aware, monotonic, reasonable)

This module provides checks, not fixes. It is designed to be run as
part of testing and monitoring, not on every transaction (too expensive).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any


class TemporalCheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    UNKNOWN = "unknown"


@dataclass
class TemporalCheckResult:
    """Result of a temporal safety check."""
    check_name: str
    status: TemporalCheckStatus
    detail: str
    feature: str | None = None


def check_timestamp_safety(
    event_timestamp: datetime,
    current_time: datetime | None = None,
    max_future_seconds: float = 300,  # 5 minutes tolerance for clock skew
    max_past_days: float = 365,  # 1 year
) -> list[TemporalCheckResult]:
    """Check that an event timestamp is safe (not in the future, not too old).

    Args:
        event_timestamp: the transaction timestamp
        current_time: current time (defaults to UTC now)
        max_future_seconds: maximum allowed future clock skew
        max_past_days: maximum age for an event
    """
    results: list[TemporalCheckResult] = []

    if current_time is None:
        current_time = datetime.now(timezone.utc)

    # Ensure both are timezone-aware
    if event_timestamp.tzinfo is None:
        event_timestamp = event_timestamp.replace(tzinfo=timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    # Check 1: Future timestamp
    future_diff = (event_timestamp - current_time).total_seconds()
    if future_diff > max_future_seconds:
        results.append(TemporalCheckResult(
            check_name="future_timestamp",
            status=TemporalCheckStatus.FAIL,
            detail=f"event is {future_diff:.0f}s in the future (max {max_future_seconds:.0f}s)",
        ))
    elif future_diff > 0:
        results.append(TemporalCheckResult(
            check_name="future_timestamp",
            status=TemporalCheckStatus.WARNING,
            detail=f"event is {future_diff:.0f}s in the future (within tolerance)",
        ))
    else:
        results.append(TemporalCheckResult(
            check_name="future_timestamp",
            status=TemporalCheckStatus.PASS,
            detail="timestamp is in the past",
        ))

    # Check 2: Too old
    past_diff = (current_time - event_timestamp).total_seconds()
    if past_diff > max_past_days * 86400:
        results.append(TemporalCheckResult(
            check_name="stale_timestamp",
            status=TemporalCheckStatus.WARNING,
            detail=f"event is {past_diff / 86400:.0f} days old (max {max_past_days:.0f})",
        ))
    else:
        results.append(TemporalCheckResult(
            check_name="stale_timestamp",
            status=TemporalCheckStatus.PASS,
            detail=f"event age {past_diff / 86400:.1f} days is acceptable",
        ))

    # Check 3: Timezone awareness
    if event_timestamp.tzinfo is None:
        results.append(TemporalCheckResult(
            check_name="timezone_aware",
            status=TemporalCheckStatus.WARNING,
            detail="event timestamp is timezone-naive (assumed UTC)",
        ))
    else:
        results.append(TemporalCheckResult(
            check_name="timezone_aware",
            status=TemporalCheckStatus.PASS,
            detail="event timestamp is timezone-aware",
        ))

    return results


def check_timestamp_ordering(
    timestamps: list[datetime],
) -> list[TemporalCheckResult]:
    """Check that timestamps are monotonically non-decreasing.

    Used to verify that historical features don't include future events.
    """
    results: list[TemporalCheckResult] = []

    if len(timestamps) < 2:
        results.append(TemporalCheckResult(
            check_name="timestamp_ordering",
            status=TemporalCheckStatus.PASS,
            detail="single timestamp — no ordering to check",
        ))
        return results

    violations = 0
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            violations += 1

    if violations == 0:
        results.append(TemporalCheckResult(
            check_name="timestamp_ordering",
            status=TemporalCheckStatus.PASS,
            detail=f"all {len(timestamps)} timestamps are monotonically ordered",
        ))
    else:
        results.append(TemporalCheckResult(
            check_name="timestamp_ordering",
            status=TemporalCheckStatus.FAIL,
            detail=f"{violations}/{len(timestamps)-1} timestamp ordering violations",
        ))

    return results


def check_rolling_feature_causality(
    feature_name: str,
    event_timestamp: datetime,
    contributing_timestamps: list[datetime],
    contributing_labels: list[int | None] | None = None,
) -> list[TemporalCheckResult]:
    """Verify that a rolling/aggregate feature only uses data from before the event.

    Args:
        feature_name: name of the feature being checked
        event_timestamp: the transaction timestamp
        contributing_timestamps: timestamps of events that contributed to this feature
        contributing_labels: optional labels of contributing events (must be None for online use)
    """
    results: list[TemporalCheckResult] = []

    # Check 1: No future events included
    future_events = [ts for ts in contributing_timestamps if ts > event_timestamp]
    if future_events:
        results.append(TemporalCheckResult(
            check_name="no_future_events",
            status=TemporalCheckStatus.FAIL,
            feature=feature_name,
            detail=f"{len(future_events)} future events in rolling window for '{feature_name}'",
        ))
    else:
        results.append(TemporalCheckResult(
            check_name="no_future_events",
            status=TemporalCheckStatus.PASS,
            feature=feature_name,
            detail="all contributing events are at or before the event time",
        ))

    # Check 2: No labels in contributing events (for online inference)
    if contributing_labels is not None:
        labeled = [l for l in contributing_labels if l is not None]
        if labeled:
            results.append(TemporalCheckResult(
                check_name="no_labels_in_window",
                status=TemporalCheckStatus.FAIL,
                feature=feature_name,
                detail=f"{len(labeled)} labeled events in window for '{feature_name}' — labels are not available at decision time",
            ))
        else:
            results.append(TemporalCheckResult(
                check_name="no_labels_in_window",
                status=TemporalCheckStatus.PASS,
                feature=feature_name,
                detail="no labels in contributing events",
            ))

    return results


def check_feature_causality(
    feature_name: str,
    feature_value: Any,
    feature_timestamp: datetime | None,
    event_timestamp: datetime,
    feature_depends_on_label: bool = False,
    feature_depends_on_future: bool = False,
) -> list[TemporalCheckResult]:
    """Check if a feature is causally valid for the given event.

    This is a high-level check combining timestamp safety and dependency checks.
    """
    results: list[TemporalCheckResult] = []

    # Check dependency flags
    if feature_depends_on_label:
        results.append(TemporalCheckResult(
            check_name="label_dependency",
            status=TemporalCheckStatus.FAIL,
            feature=feature_name,
            detail=f"feature '{feature_name}' depends on labels — not available at decision time",
        ))

    if feature_depends_on_future:
        results.append(TemporalCheckResult(
            check_name="future_dependency",
            status=TemporalCheckStatus.FAIL,
            feature=feature_name,
            detail=f"feature '{feature_name}' depends on future information",
        ))

    # Check timestamp if provided
    if feature_timestamp is not None:
        if feature_timestamp > event_timestamp:
            results.append(TemporalCheckResult(
                check_name="feature_before_event",
                status=TemporalCheckStatus.FAIL,
                feature=feature_name,
                detail=f"feature computed after the event it describes",
            ))
        else:
            results.append(TemporalCheckResult(
                check_name="feature_before_event",
                status=TemporalCheckStatus.PASS,
                feature=feature_name,
                detail="feature was computed at or before the event time",
            ))

    if not results:
        results.append(TemporalCheckResult(
            check_name="causality_check",
            status=TemporalCheckStatus.PASS,
            feature=feature_name,
            detail="no causality issues detected",
        ))

    return results
