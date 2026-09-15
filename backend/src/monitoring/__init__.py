"""Model monitoring, drift detection, and safeguards (Phase 40)."""
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
]
