"""Schema mapping layer — controlled mapping between external and internal schemas.

Provides explicit mapping, missing required feature detection, incompatible type
detection, and deterministic preprocessing. No silent column guessing.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metadata_schema import ExternalDatasetMetadata, PS14_ML_FEATURES


@dataclass
class MappingEntry:
    """A single feature mapping."""
    external_feature: str
    internal_feature: str
    transformation: str  # "direct", "rename", "computed", "imputed_zero", "missing"
    validation_status: str  # "compatible", "incompatible", "missing", "type_mismatch"
    notes: str = ""


@dataclass
class SchemaMappingReport:
    """Result of schema mapping."""
    compatible_features: list[MappingEntry] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)
    extra_features: list[str] = field(default_factory=list)
    incompatible_features: list[MappingEntry] = field(default_factory=list)
    all_required_present: bool = True
    schema_compatible: bool = True

    def to_dict(self) -> dict:
        return {
            "compatible_count": len(self.compatible_features),
            "missing_count": len(self.missing_features),
            "extra_count": len(self.extra_features),
            "incompatible_count": len(self.incompatible_features),
            "all_required_present": self.all_required_present,
            "schema_compatible": self.schema_compatible,
            "missing": self.missing_features,
            "incompatible": [
                {"external": m.external_feature, "internal": m.internal_feature, "reason": m.notes}
                for m in self.incompatible_features
            ],
        }


def _check_type_compatibility(ext_type: str, expected_type: str) -> bool:
    """Check if an external column type is compatible with expected type."""
    numeric_types = {"float", "float64", "float32", "int", "int64", "int32", "number"}
    if expected_type == "numeric":
        return ext_type.lower() in numeric_types
    return ext_type.lower() == expected_type.lower()


def map_schema(
    meta: ExternalDatasetMetadata,
    feature_mapping: dict[str, str] | None = None,
) -> SchemaMappingReport:
    """Map external dataset columns to the PS-14 feature schema.

    Args:
        meta: External dataset metadata with column information.
        feature_mapping: Optional explicit mapping from external -> internal feature names.
            If None, attempts direct name matching.

    Returns:
        SchemaMappingReport with compatible, missing, extra, and incompatible features.
    """
    report = SchemaMappingReport()
    mapping = feature_mapping or {}

    # Build set of external columns (excluding label)
    ext_columns = set(meta.columns) - {meta.label_column}

    # Check each required internal feature
    for internal_feat in PS14_ML_FEATURES:
        # Find the external column that maps to this internal feature
        ext_col = None
        for ext_name, int_name in mapping.items():
            if int_name == internal_feat:
                ext_col = ext_name
                break

        # Try direct name match if no explicit mapping
        if ext_col is None and internal_feat in ext_columns:
            ext_col = internal_feat

        if ext_col is None:
            report.missing_features.append(internal_feat)
            report.all_required_present = False
            report.compatible_features.append(
                MappingEntry(
                    external_feature="",
                    internal_feature=internal_feat,
                    transformation="missing",
                    validation_status="missing",
                    notes=f"Required feature '{internal_feat}' not found in external dataset",
                )
            )
        elif ext_col in ext_columns:
            # Check type compatibility
            ext_type = meta.column_types.get(ext_col, "unknown")
            if _check_type_compatibility(ext_type, "numeric"):
                transformation = "direct" if ext_col == internal_feat else f"rename:{ext_col}"
                report.compatible_features.append(
                    MappingEntry(
                        external_feature=ext_col,
                        internal_feature=internal_feat,
                        transformation=transformation,
                        validation_status="compatible",
                    )
                )
            else:
                report.incompatible_features.append(
                    MappingEntry(
                        external_feature=ext_col,
                        internal_feature=internal_feat,
                        transformation="incompatible",
                        validation_status="type_mismatch",
                        notes=f"External type '{ext_type}' not compatible with expected numeric",
                    )
                )
                report.schema_compatible = False

    # Identify extra features (present in external but not required)
    mapped_ext = {m.external_feature for m in report.compatible_features if m.external_feature}
    mapped_ext.update({m.external_feature for m in report.incompatible_features if m.external_feature})
    report.extra_features = sorted(ext_columns - mapped_ext)

    return report


def apply_mapping(
    df: "Any",  # pandas DataFrame
    mapping_report: SchemaMappingReport,
    excluded_features: list[str] | None = None,
) -> np.ndarray:
    """Apply the schema mapping to produce a feature matrix.

    Args:
        df: External dataset DataFrame.
        mapping_report: The schema mapping report.
        excluded_features: Features excluded by causality audit.

    Returns:
        numpy array of shape (n_samples, n_features) in PS14_ML_FEATURES order.
        Missing features are filled with 0.0.

    Raises:
        ValueError: If the schema is incompatible.
    """
    if not mapping_report.schema_compatible:
        incompatible = [m.external_feature for m in mapping_report.incompatible_features]
        raise ValueError(
            f"Schema incompatible — cannot apply mapping. Incompatible features: {incompatible}"
        )

    excluded = set(excluded_features or [])
    n_samples = len(df)
    n_features = len(PS14_ML_FEATURES)
    X = np.zeros((n_samples, n_features), dtype=np.float64)

    for entry in mapping_report.compatible_features:
        if entry.validation_status == "missing":
            continue  # already zero-filled
        if entry.internal_feature in excluded:
            continue  # excluded by causality audit — keep zero
        if entry.external_feature and entry.external_feature in df.columns:
            idx = PS14_ML_FEATURES.index(entry.internal_feature)
            X[:, idx] = df[entry.external_feature].to_numpy(dtype=np.float64, na_error="ignore")

    return X
