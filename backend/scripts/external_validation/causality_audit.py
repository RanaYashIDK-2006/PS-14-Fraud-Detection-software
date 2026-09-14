"""Feature causality audit — verifies features are available at decision time.

For every external feature, checks whether it could have existed at the
moment the fraud decision was made. Flags potentially post-event variables.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .metadata_schema import ExternalDatasetMetadata, FeatureProvenance


# Known suspicious feature patterns (post-event indicators)
POST_EVENT_PATTERNS = [
    "chargeback",
    "disputed",
    "investigation",
    "confirmed_fraud",
    "claim",
    "refund",
    "reversal",
    "resolution",
    "outcome",
    "label",
    "is_fraud",  # the label itself
    "fraud_status",
    "account_closed",
    "write_off",
]

SUSPICIOUS_CATEGORIES = {
    "chargeback_outcome": "Chargeback is a post-event resolution",
    "confirmed_fraud_investigation_result": "Investigation result is post-event",
    "future_account_behavior": "Future behavior is post-event",
    "manually_assigned_fraud_status": "Manual assignment is post-event",
    "post_transaction_resolution": "Resolution is post-event",
    "features_calculated_using_future_observations": "Future-dependent features are post-event",
}


@dataclass
class CausalityFlag:
    """A flagged feature with explanation."""
    feature_name: str
    flag_type: str  # "post_event", "suspicious", "unknown_availability"
    reason: str
    excluded_from_evaluation: bool = False
    excluded_reason: str = ""


@dataclass
class CausalityReport:
    """Result of the causality audit."""
    total_features: int = 0
    flagged_features: list[CausalityFlag] = field(default_factory=list)
    excluded_features: list[str] = field(default_factory=list)
    clean_features: list[str] = field(default_factory=list)
    all_clear: bool = True

    def to_dict(self) -> dict:
        return {
            "total_features": self.total_features,
            "flagged_count": len(self.flagged_features),
            "excluded_count": len(self.excluded_features),
            "all_clear": self.all_clear,
            "flagged": [
                {
                    "feature": f.feature_name,
                    "type": f.flag_type,
                    "reason": f.reason,
                    "excluded": f.excluded_from_evaluation,
                }
                for f in self.flagged_features
            ],
        }


def _check_feature_name_suspicion(name: str) -> str | None:
    """Check if a feature name matches known post-event patterns."""
    name_lower = name.lower()
    for pattern in POST_EVENT_PATTERNS:
        if pattern in name_lower:
            return f"Feature name contains '{pattern}' — possible post-event variable"
    return None


def audit_feature_provenance(feature: FeatureProvenance) -> CausalityFlag | None:
    """Audit a single feature's provenance for causality issues."""
    # Check post-event flag
    if feature.post_event_possible:
        return CausalityFlag(
            feature_name=feature.feature_name,
            flag_type="post_event",
            reason=f"Feature is flagged as potentially post-event: {feature.reason or 'no reason given'}",
            excluded_from_evaluation=True,
            excluded_reason="Post-event feature — cannot be used for real-time scoring",
        )

    # Check name-based suspicion
    suspicion = _check_feature_name_suspicion(feature.feature_name)
    if suspicion:
        return CausalityFlag(
            feature_name=feature.feature_name,
            flag_type="suspicious",
            reason=suspicion,
            excluded_from_evaluation=False,  # flag but don't auto-exclude
        )

    # Check availability time
    if feature.availability_time in ("", "unknown"):
        return CausalityFlag(
            feature_name=feature.feature_name,
            flag_type="unknown_availability",
            reason="Feature availability time not specified — cannot verify causality",
            excluded_from_evaluation=False,
        )

    return None


def run_causality_audit(meta: ExternalDatasetMetadata) -> CausalityReport:
    """Run the full causality audit on all features in the metadata.

    Returns a report listing flagged, excluded, and clean features.
    """
    report = CausalityReport()
    report.total_features = len(meta.columns)

    if not meta.feature_definitions:
        # No feature provenance provided — flag all features as unknown
        report.all_clear = False
        for col in meta.columns:
            if col != meta.label_column:
                report.flagged_features.append(
                    CausalityFlag(
                        feature_name=col,
                        flag_type="unknown_availability",
                        reason="No feature provenance metadata provided — cannot verify causality",
                    )
                )
        return report

    for feature in meta.feature_definitions:
        flag = audit_feature_provenance(feature)
        if flag:
            report.all_clear = False
            report.flagged_features.append(flag)
            if flag.excluded_from_evaluation:
                report.excluded_features.append(flag.feature_name)
        else:
            report.clean_features.append(feature.feature_name)

    return report
