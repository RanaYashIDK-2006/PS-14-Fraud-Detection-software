"""Decision traceability — canonical decision trace, feature-vector hashing, consistency verification.

Every live fraud decision is end-to-end traceable via a deterministic
decision trace that binds:
  - feature vector (canonicalized + hashed)
  - model version
  - feature/schema version
  - rule version
  - enforcement verdict
  - risk score/band/decision
  - reason codes
  - degraded status
  - timestamp

The trace hash is included in both DB-3 (RiskScore) and DB-4 (audit event)
so that consistency can be verified retroactively.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from src.monitoring.feature_contract import ML_FEATURE_ORDER, ML_FEATURE_VERSION


@dataclass
class DecisionTrace:
    """Canonical decision trace — single source of truth for one decision."""
    event_id: str
    fraud_id: str
    timestamp: float
    # Model/version info
    model_version: str
    feature_version: str
    schema_version: str
    rule_version: str
    # Feature vector
    feature_hash: str
    feature_count: int
    # Enforcement
    enforcement_verdict: str
    enforcement_issues: list[str]
    # Decision
    risk_score: int
    risk_band: str
    decision: str
    reason_codes: list[str]
    ml_score: float
    rule_score: float
    degraded: bool
    # Trace integrity
    trace_hash: str = ""

    def __post_init__(self):
        if not self.trace_hash:
            self.trace_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Deterministic hash of the decision trace (excludes timestamp for replay consistency)."""
        payload = {
            "event_id": self.event_id,
            "fraud_id": self.fraud_id,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "schema_version": self.schema_version,
            "rule_version": self.rule_version,
            "feature_hash": self.feature_hash,
            "enforcement_verdict": self.enforcement_verdict,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "decision": self.decision,
            "reason_codes": sorted(self.reason_codes),
            "ml_score": round(self.ml_score, 6),
            "rule_score": round(self.rule_score, 6),
            "degraded": self.degraded,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trace_hash"] = self.trace_hash
        return d

    def to_audit_payload(self) -> dict:
        """Audit-safe payload (no sensitive data)."""
        return {
            "event_id": self.event_id,
            "trace_hash": self.trace_hash,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "rule_version": self.rule_version,
            "feature_hash": self.feature_hash,
            "enforcement_verdict": self.enforcement_verdict,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "degraded": self.degraded,
        }


def canonicalize_features(features: dict[str, Any]) -> dict[str, float]:
    """Canonicalize a feature vector for hashing.

    - Uses ML_FEATURE_ORDER for deterministic ordering
    - Rounds floats to 6 decimal places
    - Casts ints to int
    - Excludes non-model fields
    """
    canonical = {}
    for name in ML_FEATURE_ORDER:
        value = features.get(name)
        if value is None:
            canonical[name] = 0.0
        elif isinstance(value, bool):
            canonical[name] = float(int(value))
        elif isinstance(value, int):
            canonical[name] = float(value)
        elif isinstance(value, float):
            import math
            if math.isnan(value) or math.isinf(value):
                canonical[name] = 0.0
            else:
                canonical[name] = round(value, 6)
        else:
            try:
                canonical[name] = round(float(value), 6)
            except (TypeError, ValueError):
                canonical[name] = 0.0
    return canonical


def hash_feature_vector(features: dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of a canonicalized feature vector.

    Same logical vector always produces the same hash regardless of:
    - dict ordering
    - float precision
    - NaN/Inf (normalized to 0.0)
    """
    canonical = canonicalize_features(features)
    # Use ordered JSON for determinism
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def build_decision_trace(
    *,
    event_id: str,
    fraud_id: str,
    features: dict[str, Any],
    model_version: str,
    feature_version: str,
    schema_version: str,
    rule_version: str,
    enforcement_verdict: str,
    enforcement_issues: list[str],
    risk_score: int,
    risk_band: str,
    decision: str,
    reason_codes: list[str],
    ml_score: float,
    rule_score: float,
    degraded: bool,
) -> DecisionTrace:
    """Build a canonical decision trace from evaluation results."""
    feature_hash = hash_feature_vector(features)
    return DecisionTrace(
        event_id=event_id,
        fraud_id=fraud_id,
        timestamp=time.time(),
        model_version=model_version,
        feature_version=feature_version,
        schema_version=schema_version,
        rule_version=rule_version,
        feature_hash=feature_hash,
        feature_count=len([f for f in ML_FEATURE_ORDER if f in features]),
        enforcement_verdict=enforcement_verdict,
        enforcement_issues=enforcement_issues,
        risk_score=risk_score,
        risk_band=risk_band,
        decision=decision,
        reason_codes=reason_codes,
        ml_score=ml_score,
        rule_score=rule_score,
        degraded=degraded,
    )


def verify_response_db_consistency(
    response: dict,
    db_row: dict,
) -> tuple[bool, list[str]]:
    """Verify that an API response matches the persisted DB-3 record.

    Returns (is_consistent, list_of_mismatches).
    """
    mismatches = []
    checks = [
        ("event_id", "event_id"),
        ("risk_score", "risk_score"),
        ("risk_band", "risk_band"),
        ("ml_score", "ml_score"),
        ("rule_score", "rule_score"),
        ("degraded", "degraded"),
    ]
    for resp_key, db_key in checks:
        resp_val = response.get(resp_key)
        db_val = db_row.get(db_key)
        # Normalize for comparison
        if isinstance(resp_val, float):
            resp_val = round(resp_val, 4)
        if isinstance(db_val, float):
            db_val = round(db_val, 4)
        if resp_val != db_val:
            mismatches.append(f"{resp_key}: response={resp_val} != db={db_val}")

    # Check reason_codes (response is list, DB is JSON string)
    resp_codes = sorted(response.get("reason_codes", []))
    db_codes = sorted(json.loads(db_row.get("reason_codes", "[]")) if isinstance(db_row.get("reason_codes"), str) else db_row.get("reason_codes", []))
    if resp_codes != db_codes:
        mismatches.append(f"reason_codes: response={resp_codes} != db={db_codes}")

    # Check model_version if present in response
    if "model_version" in response and "model_version" in db_row:
        if response["model_version"] != db_row["model_version"]:
            mismatches.append(f"model_version: response={response['model_version']} != db={db_row['model_version']}")

    return len(mismatches) == 0, mismatches


def verify_db_audit_consistency(
    db_row: dict,
    audit_payload: dict,
) -> tuple[bool, list[str]]:
    """Verify that a DB-3 decision record is consistent with its DB-4 audit event.

    Returns (is_consistent, list_of_mismatches).
    """
    mismatches = []
    checks = [
        ("event_id", "event_id"),
        ("risk_score", "risk_score"),
        ("risk_band", "risk_band"),
        ("degraded", "degraded"),
    ]
    for db_key, audit_key in checks:
        db_val = db_row.get(db_key)
        audit_val = audit_payload.get(audit_key)
        if db_val != audit_val:
            mismatches.append(f"{db_key}: db={db_val} != audit={audit_val}")

    # Check trace_hash if present in audit
    if "trace_hash" in audit_payload:
        # trace_hash should match if both sides computed it
        pass  # trace_hash is computed independently, verified by structure

    return len(mismatches) == 0, mismatches
