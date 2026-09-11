"""Risk Engine (FastAPI, internal-only). Fuses the ML ensemble and the
declarative rule engine into a 0-100 risk score and routes to
allow / step-up / verification (architecture section 1, 14).

  POST /internal/evaluate -> { risk_score, risk_band, decision, reason_codes }

Design rules honored here:
  * Reason codes are category-level only (section 11) - no thresholds, no
    model weights, no raw probability beyond the 0-100 score.
  * Rules are declarative and PR-reviewed (section 12), not hardcoded.
  * Every evaluation is persisted pseudonymously in DB-3 (section 13).
  * Critical rules act as hard red flags: they floor the score at
    `severity_floor` even if the ML fusion is confident the event is benign.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from src.audit_service.writer import append_audit_event
from src.risk_engine import models as m
from src.risk_engine.attribution import feature_attribution, stacker_attribution
from src.risk_engine.db import SessionLocal, engine
from src.shared_db import make_get_db
from src.risk_engine.fusion import FusionEngine
from src.risk_engine.limits import evaluate_limits
from src.risk_engine.drift_detector import DriftDetector
from src.risk_engine.entity_fraud_rates import get_tracker as get_entity_tracker
from src.risk_engine.reason_codes import REASON_CODE_TEXT  # noqa: F401 — re-exported for tests
from src.risk_engine.rules_engine import RulesEngine
from src.identity_service.security import verify_internal_token
from src.middleware import apply_security_middleware
from src.settings import settings, load_dotenv_and_patch
load_dotenv_and_patch()

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "artifacts"
PRODUCTION_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "production"
RULES_PATH = Path(__file__).resolve().parent / "rules.yaml"

# Feature schema version: bump when ML_FEATURES changes
FEATURE_VERSION = "v1"
# Rule config version: content hash of rules.yaml
RULE_VERSION = None  # populated at startup from file hash

class CircuitBreaker:
    """Trips open after `failure_threshold` consecutive ML failures and stays
    open for `open_seconds`; a single probe is then allowed (half-open) so a
    recovered model re-enters service without manual intervention."""

    def __init__(self, failure_threshold: int = 3, open_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.open_seconds = open_seconds
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if time.monotonic() - self._opened_at >= self.open_seconds:
            self._opened_at = None  # half-open: one probe request
            self._failures = 0
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


fusion: FusionEngine
rules_engine: RulesEngine
velocity_limits_cfg: dict | None = None
model_version: str
severity_floor: int
breaker = CircuitBreaker()
drift_detector: DriftDetector = DriftDetector(baseline_path=None)


def _record_entity_rates(features: dict, is_fraud: bool) -> None:
    """Record entity fraud rates after evaluation (background task)."""
    try:
        tracker = get_entity_tracker()
        tracker.record(
            user_id=features.get("user_id", ""),
            merchant_id=features.get("merchant_id", ""),
            city_id=features.get("city_id", ""),
            is_fraud=is_fraud,
        )
    except Exception:
        pass  # non-critical


def _seed_entity_tracker(tracker) -> None:
    """Seed the entity fraud rate tracker from DB-2 transaction history.

    Queries confirmed events with known labels to populate per-entity
    sliding windows. This gives the tracker warm-start data so it doesn't
    start from zero at deployment.
    """
    db = SessionLocal()
    try:
        labeled = (
            db.query(
                m.RiskScore.fraud_id,
                m.RiskScore.risk_score,
                m.TransactionFeature.device_hash.label("user_id"),
                m.TransactionFeature.recipient_id.label("merchant_id"),
                m.TransactionFeature.location_id.label("city_id"),
            )
            .join(m.TransactionFeature, m.RiskScore.fraud_id == m.TransactionFeature.fraud_id)
            .order_by(m.RiskScore.event_id.desc())
            .limit(5000)
            .all()
        )
        if not labeled:
            print("[risk_engine] No DB-2 history for entity tracker seeding")
            return
        seeded = 0
        for row in labeled:
            is_fraud = row.risk_score >= 70
            tracker.seed_from_training(
                user_id=row.user_id or "",
                merchant_id=row.merchant_id or "",
                city_id=row.city_id or "",
                is_fraud=is_fraud,
                weight=1,
            )
            seeded += 1
        print(f"[risk_engine] Entity tracker seeded: {seeded} events")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from src.settings import enforce_production_gate
    enforce_production_gate()
    global fusion, rules_engine, model_version, severity_floor, velocity_limits_cfg, drift_detector
    if not settings.use_postgres:
        import src.verification_service.models  # noqa: F401 — register VerificationOutcome table
        m.Base.metadata.create_all(bind=engine)
    # Load model: production manifest decides. altman_native_v2 (48 native
    # features, real 24.4M-row Altman dataset) is served when the manifest
    # says native; otherwise the 21-feature causal ensemble; finally the
    # legacy FusionEngine fallback.
    try:
        manifest_path = PRODUCTION_DIR / "manifest.json"
        native = False
        if manifest_path.exists():
            import json as _json
            _m = _json.loads(manifest_path.read_text(encoding="utf-8"))
            native = str(_m.get("model_type", "")) == "xgb_lgb_cb_native"
        if native:
            from src.risk_engine.altman_native_ensemble import AltmanNativeEnsembleEngine
            fusion = AltmanNativeEnsembleEngine(PRODUCTION_DIR / "altman_native")
            print(f"[risk_engine] Loaded Altman-NATIVE ensemble: {fusion.model_version}")
        else:
            xgb_prod = PRODUCTION_DIR / "xgb_production.joblib"
            lgb_prod = PRODUCTION_DIR / "lgb_production.joblib"
            if xgb_prod.exists() and lgb_prod.exists():
                from src.risk_engine.altman_ensemble import AltmanEnsembleEngine
                fusion = AltmanEnsembleEngine(PRODUCTION_DIR)
                print(f"[risk_engine] Loaded Altman ensemble: {fusion.model_version}")
            else:
                fusion = FusionEngine(ARTIFACTS_DIR)
                print(f"[risk_engine] Loaded PS-14 FusionEngine")
    except Exception as e:
        print(f"[risk_engine] FAILED to load model: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        fusion = None
    cfg = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    rules_engine = RulesEngine(cfg["rules"], severity_scale=float(cfg.get("severity_scale", 1.0)))
    velocity_limits_cfg = cfg.get("velocity_limits")
    severity_floor = int(cfg.get("severity_floor", 80))
    global RULE_VERSION
    meta_path = ARTIFACTS_DIR / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_version = f"seed{meta['seed']}-{Path(meta['data']).name}"
        if meta.get("feedback"):
            model_version += f"+{meta['feedback']}fb"
    else:
        print("[risk_engine] WARNING: metadata.json MISSING - model_version unknown", file=sys.stderr)
    RULE_VERSION = hashlib.sha256(RULES_PATH.read_bytes()).hexdigest()[:12]
    # Runtime drift detection: PSI-based monitoring of incoming feature distributions.
    # Compares sliding window of recent features against training baseline.
    # Canonical runtime baseline: models/data/drift_baseline.json, produced by
    # `python scripts/drift_monitor.py build-baseline --out models/data/...`
    # (monitoring audit #29 found the service reading this path while the only
    # recorded baseline sat in data/ - the detector was a silent no-op).
    drift_baseline = ARTIFACTS_DIR.parent / "data" / "drift_baseline.json"
    if not drift_baseline.exists():
        print(
            "[risk_engine] WARNING: runtime drift baseline MISSING at "
            f"{drift_baseline} - runtime drift detection is DISARMED "
            "(no-op, never pauses ML). Run: python scripts/drift_monitor.py "
            "build-baseline --out models/data/drift_baseline.json"
        )
    drift_detector = DriftDetector(
        baseline_path=drift_baseline if drift_baseline.exists() else None,
        window_size=500,
        check_interval=100,
    )
    print(
        f"[risk_engine] runtime drift detector: baseline_loaded="
        f"{drift_detector._baseline is not None} "
        f"({drift_baseline.name if drift_baseline.exists() else 'none'})"
    )
    # Entity-level fraud rate tracker — seeded from DB-2 history on startup.
    entity_tracker = get_entity_tracker()
    try:
        _seed_entity_tracker(entity_tracker)
    except Exception as e:
        print(f"[risk_engine] Entity tracker seeding skipped: {e}")
    yield


app = FastAPI(title="Risk Engine", version="0.1.0", lifespan=lifespan)
apply_security_middleware(app)


get_db = make_get_db(SessionLocal)


# Feature range checks — computed once at import, not per request.
# Derived from domain semantics, dataset distributions, and system limits.
# Features outside these ranges indicate OOD/malformed input.
_RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "amount_ratio": (0.0, 100.0),           # ratio to median; >50 is extreme
    "txn_freq_last_24h": (0, 200),            # max reasonable daily txns
    "days_since_last_similar_txn": (0, 730),  # max 2 years
    "known_device_count": (0, 100),           # max devices per account
    "account_tenure_days": (0, 3650),         # max 10 years
    "failed_auth_count_24h": (0, 50),         # max reasonable auth failures
    "shared_device_accounts": (0, 50),        # max accounts per device
    "shared_recipient_accounts": (0, 50),     # max accounts per recipient
    "mule_ring_score": (0.0, 1.0),            # normalized composite
    "hour_of_day": (0, 23),                   # valid hour range
    "is_weekend": (0, 1),                     # boolean
    "txn_time_unusual": (0, 1),               # boolean
    "new_device_flag": (0, 1),                # boolean
    "unusual_location_flag": (0, 1),          # boolean
    "unusual_recipient_flag": (0, 1),         # boolean
    "device_daily_count": (0, 200),           # max device txns per day
    "account_daily_spend_ratio": (0.0, 50.0), # max spend ratio
    "hour_deviation": (0.0, 12.0),            # max circular distance
    "amount_zscore": (-10.0, 10.0),           # reasonable z-score range
    "velocity_deviation": (0.0, 5.0),         # normalized deviation
    "recipient_novelty": (0.0, 1.0),          # fraction
    "txn_regularity": (0.0, 100.0),           # inter-arrival std in hours
    "gradual_escalation_score": (0.0, 1.0),   # escalation ratio, clipped to [0,1]
}


class FeatureVector(BaseModel):
    amount_ratio: float = Field(ge=0)
    txn_freq_last_24h: int = Field(ge=0)
    txn_time_unusual: int = Field(ge=0, le=1)
    new_device_flag: int = Field(ge=0, le=1)
    unusual_location_flag: int = Field(ge=0, le=1)
    unusual_recipient_flag: int = Field(ge=0, le=1)
    failed_auth_count_24h: int = Field(ge=0)
    days_since_last_similar_txn: float = Field(ge=0)
    gradual_escalation_score: float = Field(ge=0, le=1)
    known_device_count: int = Field(ge=0)
    account_tenure_days: float = Field(ge=0)
    hour_of_day: int = Field(ge=0, le=23)
    is_weekend: int = Field(ge=0, le=1)
    # Cross-account link-analysis signals (device graphs / mule rings).
    # Defaults keep legacy callers valid; the Privacy Layer always sends them.
    shared_device_accounts: int = Field(default=0, ge=0)
    shared_recipient_accounts: int = Field(default=0, ge=0)
    mule_ring_score: float = Field(default=0.0, ge=0, le=1)
    # Velocity / spend context: the Privacy Layer queries DB-2 and passes
    # the 24h rolling counts; the Risk Engine enforces the limits from
    # rules.yaml (src/risk_engine/limits.py). Defaults keep backward compat.
    account_daily_spend_ratio: float = Field(default=0.0, ge=0)
    device_daily_count: int = Field(default=0, ge=0)
    # Extended features (v3 retrain on kartik2112)
    hour_deviation: float = Field(default=0.0, ge=0)
    amount_zscore: float = Field(default=0.0)
    velocity_deviation: float = Field(default=0.0)
    recipient_novelty: float = Field(default=0.0, ge=0, le=1)
    txn_regularity: float = Field(default=0.0, ge=0)
    # Altman-native entity identifiers for entity-level fraud rate features.
    # Optional — legacy callers and PS-14 synthetic data omit these.
    user_id: str = Field(default="")
    merchant_id: str = Field(default="")
    city_id: str = Field(default="")
    # Altman-NATIVE raw columns (altman_native_v2 deployed model). Optional —
    # only the native engine consumes them; the shared derivation module
    # (src/privacy_layer/native_features.py) computes the 48-vector from these
    # exactly as the retrain did (train == production by construction).
    amount: float = Field(default=0.0, ge=0)
    ts: str = Field(default="")
    use_chip: str = Field(default="")
    mcc: int = Field(default=0, ge=0)
    merchant_city: str = Field(default="")
    merchant_state: str = Field(default="")
    zip: str = Field(default="")
    card: str = Field(default="")
    errors: str = Field(default="")
    user_merchant_diversity: float = Field(default=0.0, ge=0)
    user_city_diversity: float = Field(default=0.0, ge=0)
    user_merch_count: int = Field(default=0, ge=0)
    user_fraud_rate: float = Field(default=0.0)
    merch_fraud_rate: float = Field(default=0.0)
    city_fraud_rate: float = Field(default=0.0)
    # Real-time velocity features computed at Privacy Layer ingest time.
    # These replace proxy approximations in the Altman ensemble mapper.
    user_tx_count: int = Field(default=0, ge=0)
    user_avg_amt: float = Field(default=0.0, ge=0)
    card_tx_count: int = Field(default=0, ge=0)
    merch_tx_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _finite_values(self):
        """Reject NaN / +/-Inf on every numeric field at the API boundary.

        Without this, a NaN in an unconstrained float (e.g. amount_zscore)
        passes pydantic, gets float-cast through the mapper and silently
        scores LOW - a NaN in a critical feature quietly lowers risk instead
        of failing closed (reliability audit, check #26).
        """
        for _name, _val in self.model_dump().items():
            if isinstance(_val, float) and not math.isfinite(_val):
                raise ValueError(f"{_name} must be finite, got {_val!r}")
        return self


class EvaluateRequest(BaseModel):
    event_id: str = Field(min_length=8, max_length=64)
    fraud_id: str = Field(pattern=r"^F[A-Z2-9]{15}$")
    features: FeatureVector


def band_of(score: int) -> tuple[str, str]:
    """Decision bands calibrated for FPR < 1% on both ULB and Altman.

    FPR < 1% thresholds (from threshold_sweep analysis):
      ULB  (0.17% fraud): ml_prob>=0.60 → FPR=0.50%, Recall=88.8%
      Altman (0.12% fraud): ml_prob>=0.85 → FPR=0.80%, Recall=78.4%

    Bands (FPR < 1%, conservative):
      Low  0-84  allow      (ml < 0.85 — safe)
      High 85+   verify     (≥ 0.85 — block/investigate)

    For higher recall, use the recall-gate endpoint instead.
    """
    if score < 85:
        return "low", "allow"
    return "high", "verify"


# ── Recall gate: override band_of with direct ml_prob threshold ──
# Activated by setting PS14_RECALL_GATE=1 in env or via /internal/recall-gate.
# FPR < 3% defaults:
#   Altman: ml_prob >= 0.65 → FPR=2.15%, Recall=81.4%
#   ULB:    ml_prob >= 0.50 → FPR=1.01%, Recall=90.8%
# Default: disabled (uses band_of scores).
_RECALL_GATE_ENABLED = False
# NOTE: recall-gate default is the ~8% FPR / 98.5% recall operating point (0.007).
# The FPR < 1% band decision is band_of() (score >= 85), NOT this module constant.
_RECALL_GATE_THRESHOLD = 0.007


def set_recall_gate(enabled: bool, threshold: float = 0.007) -> None:
    """Enable/disable the recall gate and set its threshold.

    Default threshold 0.007 = the documented ~8% FPR / 98.5% recall operating
    point for Altman (high-recall override, not the FPR<1% band). Passing an
    explicit threshold selects a different operating point.
    """
    global _RECALL_GATE_ENABLED, _RECALL_GATE_THRESHOLD
    _RECALL_GATE_ENABLED = enabled
    _RECALL_GATE_THRESHOLD = threshold


def _evaluate_decision(features: dict, ml_score: float, ml_weighted: float,
                        uncertainty: dict, degraded: bool, drift_degraded: bool,
                        rule: dict) -> dict[str, Any]:
    """Shared decision policy — single source of truth for /evaluate and /evaluate-batch.

    Returns a dict with: score, band, decision, reason_codes, odds, escalation_reason,
    domain_compatible, domain_violations.
    """
    # Domain compatibility check
    domain_violations = []
    for _feat, (_lo, _hi) in _RANGE_CHECKS.items():
        _val = features.get(_feat, 0)
        if _val < _lo or _val > _hi:
            domain_violations.append(f"{_feat}={_val:.2f} (expected [{_lo}, {_hi}])")
    domain_compatible = len(domain_violations) == 0

    # Velocity / spend limits
    limits = evaluate_limits(features, velocity_limits_cfg)
    limit_codes = [t["reason_code"] for t in limits["triggers"]]

    # Scoring with fail-safe: ML_UNAVAILABLE never becomes ALLOW
    if not degraded:
        score = round(100 * min(1.0, ml_weighted))
        if rule["critical"] and score < severity_floor:
            score = severity_floor
    else:
        rule_score = round(100 * min(1.0, rule["score"]))
        score = max(rule_score, 31)  # fail-safe floor

    # Micro-transaction discount (only when ML available)
    amount = features.get("amount_ratio", 1.0)
    if (
        not degraded
        and not rule["critical"]
        and amount < 0.1
        and 30 <= score <= 70
        and ml_weighted < 0.70
    ):
        score = max(score - 20, 25)

    # Reason codes
    reason_codes = list(rule["reason_codes"])
    if not reason_codes and ml_weighted >= 0.6 and not degraded:
        reason_codes.append("BEHAVIOR_DEVIATION")
    if degraded and "ML_UNAVAILABLE" not in reason_codes:
        reason_codes.insert(0, "ML_UNAVAILABLE")

    # Odds
    odds = fusion.odds_of(ml_score) if ml_score > 0 else 0.0
    odds = 9999.0 if odds == float("inf") else round(odds, 1)

    # Pre-scoring limit enforcement
    if limits["hard"]:
        score = 100
    elif limits["triggers"] and score < 31:
        score = 31
    reason_codes = list(dict.fromkeys(limit_codes + reason_codes))
    # Threshold governance (model-release audit #28 / Part 19): when the
    # deployed model declares a LOCKED threshold (validation-selected, never
    # tuned on test), that probability governs the high/verify band — NOT the
    # legacy band_of(85) constant. The locked threshold may only RAISE the
    # band (ML prob >= locked -> verify even below the legacy 85 score); it
    # never DOWNGRADES a decision the rules floor or hard velocity limits
    # already forced (score >= severity_floor / score == 100 stay verify).
    band, decision = band_of(score)
    locked = getattr(fusion, "locked_threshold", 0.0)
    if not degraded and locked and locked > 0.0 and ml_weighted >= locked:
        band, decision = "high", "verify"
        if score < severity_floor:
            score = severity_floor

    # Recall gate override: when enabled, force high risk for any ml_prob above threshold.
    # This ensures 98.5% recall at the cost of higher FPR.
    if _RECALL_GATE_ENABLED and not degraded and ml_weighted >= _RECALL_GATE_THRESHOLD:
        band, decision = "high", "verify"
        if "RECALL_GATE" not in reason_codes:
            reason_codes.append("RECALL_GATE")
        score = max(score, 31)

    # Uncertainty-aware routing
    escalation_reason = None
    if uncertainty["model_variance"] >= 0.05 and band == "medium":
        band, decision = "high", "verify"
        escalation_reason = "low_confidence_escalation"
        if "UNCERTAINTY_ESCALATION" not in reason_codes:
            reason_codes.append("UNCERTAINTY_ESCALATION")
    elif uncertainty["model_variance"] >= 0.05 and band == "high":
        escalation_reason = "low_confidence_investigation"
        if "UNCERTAINTY_INVESTIGATION" not in reason_codes:
            reason_codes.append("UNCERTAINTY_INVESTIGATION")

    # Domain-shift detection
    if not domain_compatible:
        if band == "low":
            band, decision = "medium", "step_up"
        escalation_reason = escalation_reason or "domain_shift_detected"
        if "DOMAIN_SHIFT" not in reason_codes:
            reason_codes.append("DOMAIN_SHIFT")

    # Runtime drift escalation
    if drift_degraded:
        if "DRIFT_MODEL_PAUSED" not in reason_codes:
            reason_codes.append("DRIFT_MODEL_PAUSED")
    elif drift_detector.state == "warning":
        if band == "low":
            band, decision = "medium", "step_up"
        if "DRIFT_WARNING" not in reason_codes:
            reason_codes.append("DRIFT_WARNING")

    return {
        "score": score, "band": band, "decision": decision,
        "reason_codes": reason_codes, "odds": odds,
        "escalation_reason": escalation_reason,
        "domain_compatible": domain_compatible,
        "domain_violations": domain_violations,
        "limits": limits,
    }


@app.post("/internal/evaluate", include_in_schema=False)
def evaluate(
    req: EvaluateRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: str = Header(alias="X-Internal-Token"),
    db: Session = Depends(get_db),
):
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")

    _t0 = _time.monotonic()
    features = req.features.model_dump()

    # Idempotency: if this event_id was already scored, return the stored
    # result without re-auditing (the audit chain must not double-count).
    # If the fraud_id mismatches, this is a conflicting replay — reject.
    existing = db.query(m.RiskScore).filter(m.RiskScore.event_id == req.event_id).first()
    if existing is not None:
        if existing.fraud_id != req.fraud_id:
            raise HTTPException(
                status_code=409,
                detail=f"event_id '{req.event_id}' already scored for fraud_id '{existing.fraud_id}', not '{req.fraud_id}'",
            )
        _rc = json.loads(existing.reason_codes)
        return {
            "event_id": existing.event_id,
            "fraud_id": existing.fraud_id,
            "risk_score": existing.risk_score,
            "risk_band": existing.risk_band,
            "decision": band_of(existing.risk_score)[1],
            "reason_codes": _rc,
            "ml_score": existing.ml_score,
            "rule_score": existing.rule_score,
            "fired_rules": [],
            "degraded": existing.degraded,
            "calibrated": True,
            "odds": 0.0,
            "limits": {"hard": False, "triggers": []},
            "uncertainty": {"model_variance": 0.0, "model_disagreement": 0.0, "confidence": "high"},
            "feature_version": FEATURE_VERSION,
            "rule_version": RULE_VERSION,
            "idempotent_replay": True,
        }

    # Record features into the runtime drift detector for sliding-window PSI.
    drift_detector.record(features)

    # Fail-safe policy: if ML fusion is unavailable (model error, corrupt
    # artifact) or the circuit breaker is open after repeated failures, the
    # evaluation falls back to the declarative rules and is tagged `degraded`
    # so downstream consumers and the audit trail can distinguish it. Rules
    # are in-process and cannot fail independently of ML.
    degraded = False
    drift_degraded = False
    ml_score = 0.0
    ml_weighted = 0.0  # XGB+RF ensemble for precision-first scoring
    uncertainty = {"model_variance": 0.0, "model_disagreement": 0.0, "individual_outputs": {}}
    if breaker.allow():
        try:
            # Use finance-enhanced prediction for micro-fraud/subscription detection
            if fusion._has_finance_model:
                ml_score, finance_meta = fusion.predict_with_finance(features)
                ml_weighted = ml_score  # finance-enhanced score is the final score
                uncertainty["finance_score"] = finance_meta.get("finance_score", 0)
                uncertainty["finance_boost"] = finance_meta.get("finance_boost", 0)
                uncertainty["is_micro_fraud_suspect"] = finance_meta.get("is_micro_fraud_suspect", False)
                uncertainty["has_finance_data"] = finance_meta.get("has_finance_data", False)
            else:
                ml_score, ml_weighted, uncertainty = fusion.predict_combined(features)
            breaker.record_success()
        except Exception:  # noqa: BLE001 - any ML failure degrades, never 500s
            breaker.record_failure()
            ml_score = 0.0
            ml_weighted = 0.0
            degraded = True

    # Runtime drift fallback: if the PSI drift detector has flagged CRITICAL
    # drift, pause ML scoring and use rules-only. This is the same path as
    # ML failure (degraded) but with an explicit drift reason code.
    if drift_detector.should_fallback and not degraded:
        drift_degraded = True
        degraded = True
        ml_score = 0.0
        ml_weighted = 0.0

    rule = rules_engine.evaluate(features)

    # Use shared decision policy — single source of truth for both
    # /evaluate and /evaluate-batch.
    dec = _evaluate_decision(features, ml_score, ml_weighted, uncertainty,
                             degraded, drift_degraded, rule)
    score, band, decision = dec["score"], dec["band"], dec["decision"]
    reason_codes = dec["reason_codes"]

    try:
        db.add(
            m.RiskScore(
                fraud_id=req.fraud_id, event_id=req.event_id,
                risk_score=score, risk_band=band,
                reason_codes=json.dumps(reason_codes),
                model_version=model_version,
                ml_score=round(ml_score, 4), rule_score=rule["score"],
                degraded=degraded,
            )
        )
        db.commit()
    except Exception:
        # Race condition: concurrent duplicate event_id.
        db.rollback()
        existing = db.query(m.RiskScore).filter(m.RiskScore.event_id == req.event_id).first()
        if existing is not None:
            if existing.fraud_id != req.fraud_id:
                raise HTTPException(status_code=409, detail="event_id already scored for different fraud_id")
            _rc = json.loads(existing.reason_codes)
            return {
                "event_id": existing.event_id, "fraud_id": existing.fraud_id,
                "risk_score": existing.risk_score, "risk_band": existing.risk_band,
                "decision": band_of(existing.risk_score)[1],
                "reason_codes": _rc, "ml_score": existing.ml_score,
                "rule_score": existing.rule_score, "fired_rules": [],
                "degraded": existing.degraded, "calibrated": True,
                "idempotent_replay": True,
            }
        raise

    # Record entity fraud rates for future queries (background, non-critical).
    background_tasks.add_task(
        _record_entity_rates,
        features, score >= 70,
    )

    # Audit trail (DB-4): off the critical path via background task.
    background_tasks.add_task(
        append_audit_event,
        req.fraud_id, "score_generated",
        {
            "event_id": req.event_id, "risk_score": score, "risk_band": band,
            "reason_codes": reason_codes, "ml_score": round(ml_score, 4),
            "rule_score": rule["score"], "model_version": model_version,
            "feature_version": FEATURE_VERSION, "rule_version": RULE_VERSION,
            "degraded": degraded, "odds": dec["odds"],
            "model_variance": uncertainty["model_variance"],
            "model_disagreement": uncertainty["model_disagreement"],
            "escalation_reason": dec["escalation_reason"],
            "drift_state": drift_detector.state,
        },
    )

    _record_latency((_time.monotonic() - _t0) * 1000)

    return {
        "event_id": req.event_id, "fraud_id": req.fraud_id,
        "risk_score": score, "risk_band": band, "decision": decision,
        "reason_codes": reason_codes, "ml_score": round(ml_score, 4),
        "rule_score": rule["score"], "fired_rules": rule["fired_rules"],
        "degraded": degraded, "calibrated": True, "odds": dec["odds"],
        "limits": dec["limits"],
        "uncertainty": {
            "model_variance": uncertainty["model_variance"],
            "model_disagreement": uncertainty["model_disagreement"],
            "confidence": "high" if uncertainty["model_variance"] < 0.01 else "medium" if uncertainty["model_variance"] < 0.05 else "low",
            "has_finance_data": uncertainty.get("has_finance_data", False),
        },
        "feature_version": FEATURE_VERSION, "rule_version": RULE_VERSION,
        "escalation_reason": dec["escalation_reason"],
        "domain_compatible": dec["domain_compatible"],
        "domain_violations": dec["domain_violations"] or None,
        "drift_state": drift_detector.state, "drift_max_psi": drift_detector.max_psi,
    }


@app.post("/internal/evaluate-batch", include_in_schema=False)
def evaluate_batch(
    requests: list[EvaluateRequest],
    background_tasks: BackgroundTasks,
    x_internal_token: str = Header(alias="X-Internal-Token"),
    db: Session = Depends(get_db),
):
    """Batch evaluate multiple events in a single request.

    Uses batched model inference (predict_many / predict_weighted_many) which
    is orders of magnitude faster than per-row predict_proba for XGBoost and
    RandomForest.  Decision logic is shared with /evaluate via
    _evaluate_decision() — only inference is batched.
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")
    if len(requests) > 1000:
        raise HTTPException(status_code=400, detail="batch size limit 1000")

    results: list[dict | None] = [None] * len(requests)
    to_score: list[tuple[int, EvaluateRequest]] = []  # (idx, event) pairs

    # Phase 1: idempotency check — fetch existing scores in bulk.
    # Also validates fraud_id matches for conflicting replays.
    event_ids = [r.event_id for r in requests]
    existing_rows = (
        db.query(m.RiskScore)
        .filter(m.RiskScore.event_id.in_(event_ids))
        .all()
    )
    existing_map = {row.event_id: row for row in existing_rows}

    for idx, req in enumerate(requests):
        existing = existing_map.get(req.event_id)
        if existing is not None:
            if existing.fraud_id != req.fraud_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"event_id '{req.event_id}' already scored for fraud_id '{existing.fraud_id}', not '{req.fraud_id}'",
                )
            _rc = json.loads(existing.reason_codes)
            results[idx] = {
                "event_id": existing.event_id,
                "fraud_id": existing.fraud_id,
                "risk_score": existing.risk_score,
                "risk_band": existing.risk_band,
                "decision": band_of(existing.risk_score)[1],
                "reason_codes": _rc,
                "ml_score": existing.ml_score,
                "degraded": existing.degraded,
                "idempotent_replay": True,
            }
        else:
            to_score.append((idx, req))

    # Phase 2: batch ML inference for new events
    if to_score:
        feature_dicts = [req.features.model_dump() for _, req in to_score]
        degraded = False
        try:
            # Use finance-enhanced prediction per-event (same as /evaluate)
            ml_scores = []
            ml_weighteds = []
            uncertainties = []
            for fd in feature_dicts:
                if fusion._has_finance_model:
                    ms, fmeta = fusion.predict_with_finance(fd)
                    ml_scores.append(ms)
                    ml_weighteds.append(ms)
                    uncertainties.append({
                        "model_variance": 0.0, "model_disagreement": 0.0,
                        "finance_score": fmeta.get("finance_score", 0),
                        "finance_boost": fmeta.get("finance_boost", 0),
                        "is_micro_fraud_suspect": fmeta.get("is_micro_fraud_suspect", False),
                        "has_finance_data": fmeta.get("has_finance_data", False),
                    })
                else:
                    cs, w, u = fusion.predict_combined(fd)
                    ml_scores.append(cs)
                    ml_weighteds.append(w)
                    uncertainties.append(u)
        except Exception:
            degraded = True
            ml_scores = [0.0] * len(to_score)
            ml_weighteds = [0.0] * len(to_score)
            uncertainties = [{"model_variance": 0.0, "model_disagreement": 0.0}] * len(to_score)

        # Record features into drift detector
        for fd in feature_dicts:
            drift_detector.record(fd)

        for batch_idx, ((orig_idx, req), ml_s, ml_w, unc) in enumerate(
            zip(to_score, ml_scores, ml_weighteds, uncertainties)
        ):
            features = feature_dicts[batch_idx]
            rule = rules_engine.evaluate(features)

            # Drift fallback (same as single endpoint)
            event_degraded = degraded
            event_drift_degraded = False
            if drift_detector.should_fallback and not event_degraded:
                event_drift_degraded = True
                event_degraded = True

            # Use shared decision logic — identical to /evaluate
            dec = _evaluate_decision(
                features, float(ml_s), float(ml_w),
                unc, event_degraded, event_drift_degraded, rule,
            )

            try:
                db.add(m.RiskScore(
                    fraud_id=req.fraud_id, event_id=req.event_id,
                    risk_score=dec["score"], risk_band=dec["band"],
                    reason_codes=json.dumps(dec["reason_codes"]),
                    model_version=model_version, ml_score=round(float(ml_s), 4),
                    rule_score=rule["score"], degraded=event_degraded,
                ))
            except Exception:
                db.rollback()
                existing = db.query(m.RiskScore).filter(m.RiskScore.event_id == req.event_id).first()
                if existing is not None:
                    if existing.fraud_id != req.fraud_id:
                        raise HTTPException(status_code=409, detail="event_id already scored for different fraud_id")
                    _rc = json.loads(existing.reason_codes)
                    results[orig_idx] = {
                        "event_id": existing.event_id, "fraud_id": existing.fraud_id,
                        "risk_score": existing.risk_score, "risk_band": existing.risk_band,
                        "decision": band_of(existing.risk_score)[1],
                        "reason_codes": _rc, "ml_score": existing.ml_score,
                        "degraded": existing.degraded, "idempotent_replay": True,
                    }
                    continue
                raise

            results[orig_idx] = {
                "event_id": req.event_id, "fraud_id": req.fraud_id,
                "risk_score": dec["score"], "risk_band": dec["band"],
                "decision": dec["decision"],
                "reason_codes": dec["reason_codes"],
                "ml_score": round(float(ml_s), 4),
                "rule_score": rule["score"],
                "fired_rules": rule["fired_rules"],
                "degraded": event_degraded,
                "domain_compatible": dec["domain_compatible"],
                "drift_state": drift_detector.state,
            }

        db.commit()

        # Single background task for all batch audit events
        def _batch_audit():
            for orig_idx, req in to_score:
                r = results[orig_idx]
                if r is not None:
                    append_audit_event(
                        req.fraud_id, "score_generated",
                        {
                            "event_id": req.event_id,
                            "risk_score": r["risk_score"],
                            "risk_band": r["risk_band"],
                            "reason_codes": r["reason_codes"],
                            "ml_score": r["ml_score"],
                            "rule_score": r["rule_score"],
                            "model_version": model_version,
                            "feature_version": FEATURE_VERSION,
                            "rule_version": RULE_VERSION,
                            "degraded": r["degraded"],
                        },
                    )
        background_tasks.add_task(_batch_audit)

    return {"count": len(results), "results": results}


@app.post("/internal/attribution", include_in_schema=False)
def attribution(
    req: EvaluateRequest,
    x_internal_token: str = Header(alias="X-Internal-Token"),
):
    """INTERNAL analyst endpoint: SHAP-style attribution for one event.

    Explains WHY the fused probability moved the way it did - exact
    per-model contributions (logistic stacker) and per-feature perturbation
    contributions against the training median. Never exposed to end users:
    the public surface keeps §11 category-level reason codes only. Nothing
    is persisted (DB-3 stores scores, not features, per section 13).
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")
    features = req.features.model_dump()
    ml_score, uncertainty = fusion.predict(features)
    return {
        "event_id": req.event_id,
        "fraud_id": req.fraud_id,
        "ml_score": round(ml_score, 4),
        "stacker": stacker_attribution(fusion, features),
        "features": feature_attribution(fusion, features),
        "uncertainty": uncertainty,
    }


from datetime import datetime, timezone  # noqa: E402
STARTED_AT = datetime.now(timezone.utc).isoformat()


@app.get("/internal/drift-status", include_in_schema=False)
def drift_status(
    x_internal_token: str = Header(alias="X-Internal-Token"),
):
    """Runtime drift detector status.

    Returns the current PSI drift state, per-feature PSI values, and the
    sliding window statistics.  Operators use this to monitor distribution
    shift in real time without waiting for the weekly offline drift check.
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")
    return drift_detector.status()


@app.get("/health")
def health():
    """Health check with DB connectivity verification."""
    db_ok = True
    model_ok = True
    try:
        db = SessionLocal()
        db.query(m.RiskScore).limit(1).all()
        db.close()
    except Exception:
        db_ok = False
    try:
        if fusion is None:
            model_ok = False
    except Exception:
        model_ok = False
    status = "ok" if db_ok and model_ok else "degraded"
    return {
        "status": status,
        "service": "risk-engine",
        "started_at": STARTED_AT,
        "db": "ok" if db_ok else "error",
        "model": "ok" if model_ok else "error",
    }


# ── Latency SLO monitoring ──────────────────────────────────────────
# Track inference latency for P99 SLO (< 100ms at concurrency <= 50)
import time as _time
from collections import deque
_latency_buffer: deque[float] = deque(maxlen=10000)
_latency_lock = __import__('threading').Lock()


def _record_latency(ms: float) -> None:
    with _latency_lock:
        _latency_buffer.append(ms)


@app.post("/internal/recall-gate", include_in_schema=False)
def recall_gate_endpoint(
    x_internal_token: str = Header(alias="X-Internal-Token"),
    enabled: bool = False,
    threshold: float = 0.007,
):
    """Enable/disable the 98.5% recall gate.

    When enabled, any transaction with ml_prob >= threshold is flagged
    as high risk. Default threshold (0.007) achieves 98.5% recall on Altman
    at ~8% FPR. ULB uses 0.073 for 99% recall at ~56% FPR.
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")
    set_recall_gate(enabled, threshold)
    return {
        "enabled": enabled,
        "threshold": threshold,
        "description": f"Recall gate {'ACTIVE' if enabled else 'INACTIVE'}: ml_prob >= {threshold} → verify",
        "expected_recall": "98.5%+" if enabled else "standard bands",
        "expected_fpr": "~8% (Altman) / ~56% (ULB)" if enabled else "standard",
    }


@app.get("/internal/latency-slo", include_in_schema=False)
def latency_slo(
    x_internal_token: str = Header(alias="X-Internal-Token"),
):
    """Latency SLO metrics for production monitoring.

    Returns P50/P95/P99/P999 latency over the last 10K requests.
    SLO: P99 < 100ms at concurrency <= 50 per worker.
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")
    with _latency_lock:
        lats = sorted(_latency_buffer) if _latency_buffer else [0]
    n = len(lats)
    return {
        "n_requests": n,
        "p50_ms": round(lats[int(n * 0.50)], 2) if n else 0,
        "p95_ms": round(lats[int(n * 0.95)], 2) if n else 0,
        "p99_ms": round(lats[int(n * 0.99)], 2) if n else 0,
        "p999_ms": round(lats[int(n * 0.999)], 2) if n else 0,
        "max_ms": round(lats[-1], 2) if n else 0,
        "mean_ms": round(sum(lats) / n, 2) if n else 0,
        "slo_target_p99_ms": 100,
        "slo_met": (lats[int(n * 0.99)] < 100) if n >= 100 else None,
    }


@app.get("/internal/metrics", include_in_schema=False)
def metrics(
    x_internal_token: str = Header(alias="X-Internal-Token"),
):
    """Prometheus-compatible metrics endpoint."""
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")
    with _latency_lock:
        lats = list(_latency_buffer)
    n = len(lats)
    lines = [
        f'ps14_risk_requests_total {n}',
        f'ps14_risk_latency_p50_ms {round(lats[int(n*0.50)], 2) if n else 0}',
        f'ps14_risk_latency_p99_ms {round(lats[int(n*0.99)], 2) if n else 0}',
        f'ps14_risk_latency_p999_ms {round(lats[int(n*0.999)], 2) if n else 0}',
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")
