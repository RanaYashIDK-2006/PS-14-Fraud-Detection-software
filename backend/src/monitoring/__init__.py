"""Model monitoring, drift detection, data quality, and safeguards (Phases 40-41)."""
from src.monitoring.drift_detector import DriftDetector, ReferenceDistribution, DriftReport
from src.monitoring.drift_monitor import DriftMonitor, MonitoringStatus
from src.monitoring.schema_drift import (
    SchemaDriftReport,
    SchemaStatus,
    FeatureAvailability,
    check_schema,
    check_feature_availability,
)
from src.monitoring.prediction_drift import (
    PredictionDriftReport,
    check_prediction_drift,
)
from src.monitoring.alert_manager import AlertManager, Alert, AlertState, AlertSeverity
from src.monitoring.safeguards import PromotionSafeguard, SafeguardStatus, SafeguardResult
from src.monitoring.label_awareness import (
    LabelAvailability,
    OutcomeMonitoringReport,
    check_label_availability,
    check_outcome_drift,
)
from src.monitoring.baseline_governance import (
    BaselineMetadata,
    BaselineSource,
    BaselineStatus,
    validate_baseline_source,
    create_baseline,
)
from src.monitoring.feature_contract import (
    FeatureCategory,
    FeatureStatus,
    MissingPolicy,
    FeatureSpec,
    ML_FEATURE_CONTRACT,
    ML_FEATURE_VERSION,
    ML_FEATURE_ORDER,
    validate_feature,
    validate_feature_vector,
    check_feature_ordering,
    classify_decision_time_availability,
)
from src.monitoring.numerical_robustness import (
    RobustnessResult,
    safe_float,
    clamp_value,
    process_feature,
    process_feature_vector,
    has_rejections,
    get_quality_summary,
)
from src.monitoring.feature_freshness import (
    FreshnessStatus,
    FreshnessResult,
    FRESHNESS_REQUIREMENTS,
    check_feature_freshness,
    check_vector_freshness,
    get_stale_features,
)
from src.monitoring.temporal_safeguards import (
    TemporalCheckStatus,
    TemporalCheckResult,
    check_timestamp_safety,
    check_timestamp_ordering,
    check_rolling_feature_causality,
    check_feature_causality,
)
from src.monitoring.data_quality import (
    DataQualityStatus,
    DataQualityReport,
    assess_data_quality,
)
from src.monitoring.runtime_enforcement import (
    EnforcementVerdict,
    EnforcementResult,
    enforce_before_inference,
    get_feature_contract_version,
    get_expected_feature_count,
    get_expected_feature_names,
)

__all__ = [
    # Existing
    "DriftDetector",
    "ReferenceDistribution",
    "DriftReport",
    # Phase 40 orchestrator
    "DriftMonitor",
    "MonitoringStatus",
    # Schema drift
    "SchemaDriftReport",
    "SchemaStatus",
    "FeatureAvailability",
    "check_schema",
    "check_feature_availability",
    # Prediction drift
    "PredictionDriftReport",
    "check_prediction_drift",
    # Alerts
    "AlertManager",
    "Alert",
    "AlertState",
    "AlertSeverity",
    # Safeguards
    "PromotionSafeguard",
    "SafeguardStatus",
    "SafeguardResult",
    # Label awareness
    "LabelAvailability",
    "OutcomeMonitoringReport",
    "check_label_availability",
    "check_outcome_drift",
    # Baseline governance
    "BaselineMetadata",
    "BaselineSource",
    "BaselineStatus",
    "validate_baseline_source",
    "create_baseline",
    # Phase 41: feature contract
    "FeatureCategory",
    "FeatureStatus",
    "MissingPolicy",
    "FeatureSpec",
    "ML_FEATURE_CONTRACT",
    "ML_FEATURE_VERSION",
    "ML_FEATURE_ORDER",
    "validate_feature",
    "validate_feature_vector",
    "check_feature_ordering",
    "classify_decision_time_availability",
    # Phase 41: numerical robustness
    "RobustnessResult",
    "safe_float",
    "clamp_value",
    "process_feature",
    "process_feature_vector",
    "has_rejections",
    "get_quality_summary",
    # Phase 41: feature freshness
    "FreshnessStatus",
    "FreshnessResult",
    "FRESHNESS_REQUIREMENTS",
    "check_feature_freshness",
    "check_vector_freshness",
    "get_stale_features",
    # Phase 41: temporal safeguards
    "TemporalCheckStatus",
    "TemporalCheckResult",
    "check_timestamp_safety",
    "check_timestamp_ordering",
    "check_rolling_feature_causality",
    "check_feature_causality",
    # Phase 41: data quality gates
    "DataQualityStatus",
    "DataQualityReport",
    "assess_data_quality",
    # Phase 42: runtime enforcement
    "EnforcementVerdict",
    "EnforcementResult",
    "enforce_before_inference",
    "get_feature_contract_version",
    "get_expected_feature_count",
    "get_expected_feature_names",
]
