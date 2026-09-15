"""Numerical robustness and missing-data policy.

Handles NaN, Inf, overflow, underflow, division-by-zero, and missing values
in model input features. Every transformation is documented and deterministic.

The key distinction this module enforces:
  MISSING != ZERO != NOT_APPLICABLE != UNKNOWN

These must not automatically collapse to the same value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.monitoring.feature_contract import (
    ML_FEATURE_CONTRACT,
    MissingPolicy,
    FeatureSpec,
)


@dataclass
class RobustnessResult:
    """Result of robustness check on a single feature value."""
    feature: str
    original_value: Any
    processed_value: float
    was_missing: bool
    was_non_finite: bool
    was_out_of_bounds: bool
    action_taken: str  # "pass", "clamped", "imputed", "rejected"
    detail: str = ""


def safe_float(value: Any, default: float = 0.0) -> tuple[float, str]:
    """Convert a value to float safely, handling NaN/Inf/None/strings.

    Returns (converted_value, detail_string).
    """
    if value is None:
        return default, "null -> default"
    if isinstance(value, bool):
        return float(int(value)), "bool -> float"
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f):
            return default, f"NaN -> {default}"
        if math.isinf(f):
            if f > 0:
                return default, f"+Inf -> {default}"
            return default, f"-Inf -> {default}"
        return f, "ok"
    # String or other type
    try:
        f = float(value)
        if math.isnan(f):
            return default, f"NaN from string -> {default}"
        if math.isinf(f):
            return default, f"Inf from string -> {default}"
        return f, "ok"
    except (TypeError, ValueError):
        return default, f"{type(value).__name__} -> default {default}"


def clamp_value(value: float, spec: FeatureSpec) -> tuple[float, bool, str]:
    """Clamp a value to the feature's allowed range.

    Returns (clamped_value, was_clamped, detail).
    """
    clamped = value
    was_clamped = False
    parts = []

    if spec.min_value is not None and value < spec.min_value:
        clamped = spec.min_value
        was_clamped = True
        parts.append(f"clamped from {value} to min {spec.min_value}")

    if spec.max_value is not None and clamped > spec.max_value:
        clamped = spec.max_value
        was_clamped = True
        parts.append(f"clamped from {clamped} to max {spec.max_value}")

    return clamped, was_clamped, "; ".join(parts)


def process_feature(name: str, value: Any) -> RobustnessResult:
    """Apply the full robustness pipeline to a single feature.

    Pipeline:
    1. safe_float (handles NaN/Inf/None/wrong type)
    2. clamp to valid range (if defined)
    3. Apply missing policy (reject/use_neutral/use_imputed/fail_closed)
    """
    spec = ML_FEATURE_CONTRACT.get(name)
    if spec is None:
        # No contract — pass through with NaN/Inf cleanup
        v, detail = safe_float(value)
        return RobustnessResult(
            feature=name, original_value=value, processed_value=v,
            was_missing=value is None, was_non_finite=False,
            was_out_of_bounds=False, action_taken="pass", detail=detail,
        )

    # Step 1: Handle missing
    if value is None:
        if spec.missing_policy == MissingPolicy.REJECT:
            return RobustnessResult(
                feature=name, original_value=None,
                processed_value=spec.neutral_value,
                was_missing=True, was_non_finite=False, was_out_of_bounds=False,
                action_taken="rejected", detail=f"required feature is null",
            )
        elif spec.missing_policy == MissingPolicy.FAIL_CLOSED:
            return RobustnessResult(
                feature=name, original_value=None,
                processed_value=spec.max_value or 1.0,
                was_missing=True, was_non_finite=False, was_out_of_bounds=False,
                action_taken="fail_closed", detail="null -> fail-closed value",
            )
        elif spec.missing_policy == MissingPolicy.USE_IMPUTED:
            # Model-compatible imputation
            return RobustnessResult(
                feature=name, original_value=None,
                processed_value=spec.neutral_value,
                was_missing=True, was_non_finite=False, was_out_of_bounds=False,
                action_taken="imputed", detail=f"null -> imputed {spec.neutral_value}",
            )
        else:  # USE_NEUTRAL
            return RobustnessResult(
                feature=name, original_value=None,
                processed_value=spec.neutral_value,
                was_missing=True, was_non_finite=False, was_out_of_bounds=False,
                action_taken="imputed", detail=f"null -> neutral {spec.neutral_value}",
            )

    # Step 2: Safe float conversion
    v, conv_detail = safe_float(value, spec.neutral_value)
    was_non_finite = conv_detail != "ok"

    # Step 3: Clamp to range
    v, was_clamped, clamp_detail = clamp_value(v, spec)

    # Step 4: Cast to int if needed
    if spec.datatype == "int":
        v = float(round(v))

    action = "pass"
    if was_clamped:
        action = "clamped"
    elif was_non_finite:
        action = "imputed"

    return RobustnessResult(
        feature=name, original_value=value, processed_value=v,
        was_missing=False, was_non_finite=was_non_finite,
        was_out_of_bounds=was_clamped,
        action_taken=action,
        detail=conv_detail if was_non_finite else (clamp_detail if was_clamped else "ok"),
    )


def process_feature_vector(features: dict[str, Any]) -> tuple[dict[str, float], list[RobustnessResult]]:
    """Process an entire feature vector through the robustness pipeline.

    Returns (processed_features, list_of_results).
    """
    processed: dict[str, float] = {}
    results: list[RobustnessResult] = []

    for name in ML_FEATURE_CONTRACT:
        value = features.get(name)
        r = process_feature(name, value)
        processed[name] = r.processed_value
        results.append(r)

    return processed, results


def has_rejections(results: list[RobustnessResult]) -> bool:
    """Check if any features were rejected (critical missing required fields)."""
    return any(r.action_taken == "rejected" for r in results)


def get_quality_summary(results: list[RobustnessResult]) -> dict[str, int]:
    """Summarize robustness actions taken."""
    summary: dict[str, int] = {}
    for r in results:
        summary[r.action_taken] = summary.get(r.action_taken, 0) + 1
    return summary
