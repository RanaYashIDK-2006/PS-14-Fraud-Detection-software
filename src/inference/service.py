"""Real-time Fraud Scoring Inference Service (FastAPI, port 8006).

Loads the tuned LGB Altman model and computes velocity features from a
Redis-backed sliding window of recent transactions. Sub-100ms per-transaction
latency with horizontal scaling across multiple workers.

Endpoints:
  POST /score           — score a single transaction
  POST /score-batch     — score a batch of transactions
  GET  /health          — health check with model + Redis info
  GET  /stats           — state manager statistics
  DELETE /state/{user}  — reset a user's sliding window
  GET  /                — interactive docs

State backend:
  - Default: Redis (redis://localhost:6379/0) for horizontal scaling
  - Fallback: in-process dict if Redis is unavailable
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

# --- Environment setup before imports that read settings ---
from src.settings import load_dotenv_and_patch
load_dotenv_and_patch()

import os

from src.inference.realtime_scorer import (  # noqa: E402
    RealtimeScorer, Transaction, ScoredTransaction, build_feature_vector,
    StateManager as InProcessStateManager,
)
from src.inference.redis_state import RedisStateManager  # noqa: E402
from src.inference.rate_limiter import RedisRateLimiter  # noqa: E402
from src.inference.model_registry import ModelRegistry  # noqa: E402
from src.inference.ab_testing import TrafficSplitter  # noqa: E402

# ── Pydantic Models ──

class ScoreRequest(BaseModel):
    """Score a single transaction."""
    user_id: str = Field(..., description="Unique user/account identifier")
    amount: float = Field(..., ge=0, description="Transaction amount")
    hour: float = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    minute: float = Field(..., ge=0, le=59, description="Minute (0-59)")
    day: float = Field(..., ge=1, le=31, description="Day of month")
    merchant_id: int = Field(0, description="Merchant identifier")
    city_id: int = Field(0, description="City identifier")
    chip: int = Field(0, ge=0, le=2, description="Chip type (0=swipe,1=online,2=chip)")
    mcc: float = Field(0.0, description="Merchant category code (raw)")
    is_online: int = Field(0, ge=0, le=1, description="Is online transaction")
    is_error: int = Field(0, ge=0, le=1, description="Is error transaction")
    card: float = Field(0.0, description="Card type identifier")
    year: float = Field(2024.0, description="Transaction year")
    month: float = Field(1.0, description="Transaction month")


class ScoreResponse(BaseModel):
    """Scored transaction result."""
    user_id: str
    fraud_probability: float
    fraud_flag: bool
    risk_level: str
    velocity_features: dict
    latency_ms: float
    model_version: str


class BatchScoreRequest(BaseModel):
    """Score a batch of transactions."""
    transactions: list[ScoreRequest]


class BatchScoreResponse(BaseModel):
    """Batch scoring result."""
    results: list[ScoreResponse]
    total_latency_ms: float
    per_transaction_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    model_loaded: bool
    model_version: str
    model_type: str
    active_users: int
    state_backend: str
    redis_connected: bool | None = None
    redis_memory_mb: float | None = None
    uptime_s: float
    registered_models: int = 0
    ab_experiments: int = 0


class StatsResponse(BaseModel):
    """Pipeline statistics."""
    active_users: int
    window_size: int
    max_users: int
    model_version: str
    model_type: str
    uptime_s: float
    total_scored: int


# ── A/B Testing Pydantic Models ──

class ABScoreRequest(BaseModel):
    """Score with A/B routing."""
    user_id: str
    amount: float
    hour: float
    minute: float = 0.0
    day: float = 15.0
    merchant_id: int = 0
    city_id: int = 0
    chip: int = 0
    mcc: float = 0.0
    is_online: int = 0
    is_error: int = 0
    card: float = 0.0
    year: float = 2024.0
    month: float = 1.0
    experiment: str = Field("default", description="Experiment name")
    force_version: str | None = Field(None, description="Override: force a specific model version")


class ABScoreResponse(BaseModel):
    """A/B scored result with routing info."""
    user_id: str
    fraud_probability: float
    fraud_flag: bool
    risk_level: str
    model_version: str
    experiment: str
    route: str  # "control" or "treatment"
    latency_ms: float
    velocity_features: dict


class CreateExperimentRequest(BaseModel):
    """Create a new A/B experiment."""
    name: str
    versions: dict[str, float]  # version → traffic percentage
    strategy: str = "percentage"  # "percentage" or "user_deterministic"
    description: str = ""


class UpdateTrafficRequest(BaseModel):
    """Update traffic split for an experiment."""
    versions: dict[str, float]


# ── Lifespan ──

_scorer: RealtimeScorer | None = None
_redis_state: RedisStateManager | None = None
_registry: ModelRegistry | None = None
_ab: TrafficSplitter | None = None
_start_time: float = 0.0
_total_scored: int = 0
_state_backend: str = "unknown"


def _build_scorer() -> tuple[RealtimeScorer, str]:
    """Build scorer with Redis (standalone or Sentinel) or in-process fallback."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        rs = RedisStateManager(redis_url=redis_url)
        if rs._connected:
            backend = f"redis:{rs._mode}"
            scorer = RealtimeScorer(state_manager=rs)
            return scorer, backend
        else:
            raise ConnectionError("Redis not responding")
    except Exception as e:
        print(f"[inference] Redis unavailable ({e}), using in-process state")
        ism = InProcessStateManager(max_users=100_000, window_size=50)
        scorer = RealtimeScorer(state_manager=ism)
        return scorer, "in-process"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _scorer, _redis_state, _registry, _ab, _start_time, _total_scored, _state_backend
    _start_time = time.time()

    # Build primary scorer (Redis or in-process)
    try:
        _scorer, _state_backend = _build_scorer()
        model_ver = _scorer.get_stats().get("model_version", "unknown")
        print(f"[inference] Model: {model_ver} | State: {_state_backend}")
    except Exception as e:
        print(f"[inference] WARNING: Model load failed ({e}), running in degraded mode")
        _scorer = None
        _state_backend = "degraded"

    # Initialize model registry and A/B testing
    _registry = ModelRegistry()
    discovered = _registry.auto_discover()
    print(f"[inference] Registry: {len(_registry.versions)} models loaded: {_registry.versions}")

    _ab = TrafficSplitter()
    # Create default experiment if 2+ models are registered
    if len(_registry.versions) >= 2:
        default_versions = {v: 1.0 / len(_registry.versions) for v in _registry.versions}
        _ab.create_experiment("default", default_versions, description="Default equal-split experiment")
        print(f"[inference] A/B: default experiment with {len(_registry.versions)} versions")

    # Auto-load batch scorer on startup
    try:
        from src.inference.batch_scorer import get_batch_scorer
        batch = get_batch_scorer()
        bv = batch.load()
        print(f"[inference] Batch scorer: {bv.version} ({bv.n_features} features)")
    except Exception as e:
        print(f"[inference] WARNING: Batch scorer load failed ({e})")

    # Eagerly load unified scorer so /unified-status returns loaded=true
    global _unified_scorer
    try:
        from src.inference.unified_scorer import UnifiedScorer
        _unified_scorer = UnifiedScorer()
        status = _unified_scorer.get_status()
        print(f"[inference] Unified scorer: loaded={status.get('loaded')} | subsystems={list(status.get('subsystems', {}).keys())}")
    except Exception as e:
        print(f"[inference] WARNING: Unified scorer load failed ({e})")

    # Initialize Redis-backed rate limiter
    rate_limiter = None
    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        rl = RedisStateManager(redis_url=redis_url)
        if rl._connected:
            rate_limiter = RedisRateLimiter(
                r=rl.r,
                per_user_limit=int(os.environ.get("RATE_LIMIT_PER_USER", "100")),
                global_limit=int(os.environ.get("RATE_LIMIT_GLOBAL", "10000")),
                window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW", "60")),
            )
            print(f"[inference] Rate limiter: per_user={rate_limiter.per_user_limit}, "
                  f"global={rate_limiter.global_limit}, window={rate_limiter.window_seconds}s")
    except Exception as e:
        print(f"[inference] Rate limiter init failed (non-fatal): {e}")
    _app.state.rate_limiter = rate_limiter

    yield
    # Cleanup
    if _redis_state is not None:
        _redis_state.close()


# ── App ──

app = FastAPI(
    title="PS-14 Real-Time Fraud Scoring",
    version="1.0.0",
    lifespan=lifespan,
)

from src.settings import get_settings
_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.allowed_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──

def _to_txn(req: ScoreRequest) -> Transaction:
    """Convert Pydantic request to internal Transaction."""
    return Transaction(
        user_id=req.user_id,
        amount=req.amount,
        hour=req.hour,
        minute=req.minute,
        day=req.day,
        merchant_id=req.merchant_id,
        city_id=req.city_id,
        chip=req.chip,
        mcc=req.mcc,
        is_online=req.is_online,
        is_error=req.is_error,
        card=req.card,
        year=req.year,
        month=req.month,
    )


def _risk_level(prob: float) -> str:
    """Map probability to risk level."""
    if prob >= 0.7:
        return "critical"
    elif prob >= 0.4:
        return "high"
    elif prob >= 0.15:
        return "medium"
    else:
        return "low"


def _to_response(scored: ScoredTransaction) -> ScoreResponse:
    """Convert internal scored transaction to response."""
    return ScoreResponse(
        user_id=scored.user_id,
        fraud_probability=scored.fraud_probability,
        fraud_flag=scored.fraud_flag,
        risk_level=_risk_level(scored.fraud_probability),
        velocity_features=scored.velocity_features,
        latency_ms=scored.latency_ms,
        model_version=scored.model_version,
    )


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with model + Redis info."""
    # Check batch scorer (always available)
    from src.inference.batch_scorer import get_batch_scorer
    batch = get_batch_scorer()
    batch_loaded = batch.is_loaded
    batch_version = batch.version.version if batch.version else "none"

    if _scorer is None and not batch_loaded:
        return HealthResponse(
            status="degraded",
            service="inference-service",
            model_loaded=False,
            model_version="none",
            model_type="none",
            active_users=0,
            state_backend="degraded",
            redis_connected=False,
            uptime_s=round(time.time() - _start_time, 1),
        )
    if _scorer is None:
        # Batch scorer is loaded — report OK
        return HealthResponse(
            status="ok",
            service="inference-service",
            model_loaded=True,
            model_version=batch_version,
            model_type="xgb_lgb_batch",
            active_users=0,
            state_backend="batch-only",
            redis_connected=False,
            uptime_s=round(time.time() - _start_time, 1),
            registered_models=len(_registry.versions) if _registry else 0,
        )
    stats = _scorer.get_stats()
    return HealthResponse(
        status="ok",
        service="inference-service",
        model_loaded=True,
        model_version=stats.get("model_version", "unknown"),
        model_type=stats.get("model_type", "unknown"),
        active_users=stats.get("active_users", 0),
        state_backend=_state_backend,
        redis_connected=stats.get("redis_connected", None),
        redis_memory_mb=stats.get("redis_memory_mb", None),
        uptime_s=round(time.time() - _start_time, 1),
        registered_models=len(_registry.versions) if _registry else 0,
        ab_experiments=len(_ab.list_experiments()) if _ab else 0,
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Pipeline statistics."""
    if _scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    s = _scorer.get_stats()
    return StatsResponse(
        active_users=s.get("active_users", 0),
        window_size=s.get("window_size", 50),
        max_users=s.get("max_users", 100_000),
        model_version=s.get("model_version", "unknown"),
        model_type=s.get("model_type", "unknown"),
        uptime_s=round(time.time() - _start_time, 1),
        total_scored=_total_scored,
    )


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    """Score a single transaction for fraud probability.

    The endpoint:
    1. Updates the user's sliding window with the new transaction
    2. Computes velocity features (tx_count, amount_zscore, merchant_diversity, etc.)
    3. Builds the 52-feature vector matching the production model schema
    4. Runs LGB inference and returns the fraud probability + risk level

    Target latency: <100ms per transaction.
    """
    global _total_scored

    if _scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    txn = _to_txn(req)
    scored = _scorer.score(txn)
    _total_scored += 1

    return _to_response(scored)


@app.post("/score-batch", response_model=BatchScoreResponse)
async def score_batch(req: BatchScoreRequest):
    """Score a batch of transactions efficiently.

    Batches the LGB prediction for throughput while still computing
    per-transaction velocity features from the sliding window.
    """
    global _total_scored

    if _scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.transactions:
        raise HTTPException(status_code=400, detail="Empty transaction list")

    if len(req.transactions) > 10_000:
        raise HTTPException(status_code=400, detail="Batch too large (max 10,000)")

    txns = [_to_txn(t) for t in req.transactions]
    t0 = time.perf_counter()
    scored_list = _scorer.score_batch(txns)
    total_ms = (time.perf_counter() - t0) * 1000
    _total_scored += len(txns)

    return BatchScoreResponse(
        results=[_to_response(s) for s in scored_list],
        total_latency_ms=round(total_ms, 2),
        per_transaction_ms=round(total_ms / len(txns), 3),
        batch_size=len(txns),
    )


@app.delete("/state/{user_id}")
async def reset_state(user_id: str):
    """Reset a user's sliding window state."""
    if _scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Try Redis reset first
    if hasattr(_scorer.state, 'reset_user'):
        deleted = _scorer.state.reset_user(user_id)
        action = "state_reset" if deleted else "no_state_found"
        return {"status": "ok", "user_id": user_id, "action": action, "backend": _state_backend}

    # Fallback to in-process reset
    with _scorer.state._lock:
        if user_id in _scorer.state._windows:
            del _scorer.state._windows[user_id]
            try:
                _scorer.state._access_order.remove(user_id)
            except ValueError:
                pass
            return {"status": "ok", "user_id": user_id, "action": "state_reset", "backend": _state_backend}
        else:
            return {"status": "ok", "user_id": user_id, "action": "no_state_found", "backend": _state_backend}


# ── A/B Testing Endpoints ──

@app.post("/ab/score", response_model=ABScoreResponse)
async def ab_score(req: ABScoreRequest):
    """Score a transaction through A/B routing.

    Routes the transaction to a model version based on the experiment config.
    Records metrics for both the routed version and the experiment.
    """
    global _total_scored

    if _registry is None or _ab is None:
        raise HTTPException(status_code=503, detail="A/B framework not initialized")

    t0 = time.perf_counter()

    # Determine which version to use
    if req.force_version:
        version = req.force_version
        route = "forced"
    else:
        version = _ab.get_version(req.experiment, user_id=req.user_id)
        route = "experiment"

    # Get the scorer for this version
    scorer_fn = _registry.get_scorer(version)
    mv = _registry.get_model(version)

    # Build a feature vector from the raw Altman fields
    features = np.array([
        req.amount, np.log1p(req.amount), req.amount ** 2,
        req.hour, req.minute, 0.0, req.month, req.day,
        np.sin(2 * np.pi * req.hour / 24), np.cos(2 * np.pi * req.hour / 24),
        1.0 if req.hour < 6 or req.hour > 22 else 0.0,
        1.0 if 9 <= req.hour <= 17 else 0.0,
        float(req.chip), float(req.is_online), float(req.is_error),
        0.0,  # has_zip
        1.0 if req.is_online else 0.0,  # has_state
        0.0,  # is_online_or_no_state
        float(req.mcc),
        1.0 if req.mcc >= 5000 else 0.0,
        1.0 if 5812 <= req.mcc <= 5814 else 0.0,
        1.0 if 5541 <= req.mcc <= 5542 else 0.0,
        1.0 if 5411 <= req.mcc <= 5422 else 0.0,
        1.0 if 3000 <= req.mcc <= 3350 else 0.0,
        1.0 if 5967 <= req.mcc <= 5969 else 0.0,
        float(req.merchant_id), float(req.city_id), float(req.card),
        0.0, 0.0, 0.0, 0.0,  # user_tx_count, card_tx_count, user_avg_amt, amt_vs_user_avg
        0.0, 0.0,  # amt_zscore, merch_tx_count
        0.0, 0.0,  # user_merchant_diversity, user_city_diversity
        0.001, 0.001, 0.001,  # entity fraud rates (baseline)
        1.0 if req.amount > 200 else 0.0,  # high_amt
        1.0 if req.amount > 500 else 0.0,  # very_high_amt
        req.amount * req.hour, req.amount * req.mcc,
        req.amount * req.chip, req.amount * req.is_online,
        req.amount * (1.0 if req.hour < 6 or req.hour > 22 else 0.0),
        0.0,  # user_merch_count
    ], dtype=np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    # Score with the routed model
    scored_ok = False
    proba = 0.0
    if scorer_fn is not None and mv is not None:
        try:
            n_feat = features.shape[0]
            expected = mv.n_features
            if n_feat == expected:
                proba = float(scorer_fn(features.reshape(1, -1))[0])
                scored_ok = True
            elif n_feat > expected:
                proba = float(scorer_fn(features[:expected].reshape(1, -1))[0])
                scored_ok = True
            elif n_feat < expected:
                padded = np.zeros(expected, dtype=np.float64)
                padded[:n_feat] = features
                proba = float(scorer_fn(padded.reshape(1, -1))[0])
                scored_ok = True
        except Exception:
            pass

    if not scored_ok:
        proba = 0.0
        route = "fallback"
    else:
        route = "experiment"

    fraud_flag = proba > 0.5
    latency_ms = (time.perf_counter() - t0) * 1000
    _total_scored += 1

    # Record A/B metrics
    _ab.record(req.experiment, version, proba, fraud_flag, latency_ms)

    return ABScoreResponse(
        user_id=req.user_id,
        fraud_probability=round(proba, 6),
        fraud_flag=fraud_flag,
        risk_level=_risk_level(proba),
        model_version=version,
        experiment=req.experiment,
        route=route,
        latency_ms=round(latency_ms, 3),
        velocity_features={},
    )


@app.get("/ab/experiments")
async def list_experiments():
    """List all A/B experiments and their metrics."""
    if _ab is None:
        raise HTTPException(status_code=503, detail="A/B framework not initialized")
    return {"experiments": _ab.list_experiments()}


@app.post("/ab/experiments")
async def create_experiment(req: CreateExperimentRequest):
    """Create a new A/B experiment."""
    if _ab is None:
        raise HTTPException(status_code=503, detail="A/B framework not initialized")
    if _registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")

    # Validate versions exist in registry
    for v in req.versions:
        if v not in _registry.versions:
            raise HTTPException(
                status_code=400,
                detail=f"Version '{v}' not in registry. Available: {_registry.versions}"
            )

    try:
        exp = _ab.create_experiment(
            req.name, req.versions, req.strategy, req.description
        )
        return {"status": "ok", "experiment": exp.name, "versions": exp.versions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/ab/experiments/{name}/traffic")
async def update_traffic(name: str, req: UpdateTrafficRequest):
    """Update traffic split for an experiment."""
    if _ab is None:
        raise HTTPException(status_code=503, detail="A/B framework not initialized")

    success = _ab.update_traffic(name, req.versions)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found or invalid split")
    return {"status": "ok", "experiment": name, "versions": req.versions}


@app.post("/ab/experiments/{name}/deactivate")
async def deactivate_experiment(name: str):
    """Deactivate an experiment (all traffic goes to default)."""
    if _ab is None:
        raise HTTPException(status_code=503, detail="A/B framework not initialized")

    success = _ab.deactivate_experiment(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    return {"status": "ok", "experiment": name, "active": False}


@app.get("/ab/metrics/{experiment}")
async def get_ab_metrics(experiment: str):
    """Get metrics for all versions in an experiment."""
    if _ab is None:
        raise HTTPException(status_code=503, detail="A/B framework not initialized")

    metrics = _ab.get_metrics(experiment)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment}' not found")
    return {"experiment": experiment, "metrics": metrics}


@app.get("/ab/versions")
async def list_versions():
    """List all registered model versions."""
    if _registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    return {"versions": _registry.list_versions(), "default": _registry.default_version}


# ── Unified Detection Endpoint ─────────────────────────────────────────────
_unified_scorer = None

class UnifiedScoreRequest(BaseModel):
    """Unified scoring request — accepts raw features OR ML_FEATURES directly."""
    model_config = {"extra": "allow"}  # pass through ML_FEATURES fields
    user_id: str = Field("", description="User/account identifier")
    transaction_id: str = Field("", description="Transaction identifier")
    amount: float = Field(0.0, ge=0, description="Transaction amount")
    timestamp: float = Field(0.0, description="Timestamp")


@app.post("/unified-score")
async def unified_score(tx: UnifiedScoreRequest):
    """Score through ALL detection systems in one pass: ML fusion + rules + velocity."""
    global _unified_scorer
    if _unified_scorer is None:
        try:
            from src.inference.unified_scorer import UnifiedScorer
            _unified_scorer = UnifiedScorer()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Unified scorer init failed: {e}")

    features = tx.model_dump()
    result = _unified_scorer.score(features, user_id=tx.user_id)
    return {
        "fraud_probability": result.fraud_probability,
        "risk_score": result.risk_score,
        "decision": result.decision,
        "risk_level": result.risk_level,
        "ml_score": result.ml_score,
        "rules_score": result.rules_score,
        "rules_fired": result.rules_fired,
        "reason_codes": result.reason_codes,
        "velocity_triggered": result.velocity_triggered,
        "critical_fired": result.critical_fired,
        "degraded": result.degraded,
        "latency_ms": result.latency_ms,
        "model_version": result.model_version,
    }


@app.post("/unified-score-batch")
async def unified_score_batch(txns: list[UnifiedScoreRequest]):
    """Score a batch through ALL detection systems with batched ML."""
    global _unified_scorer
    if _unified_scorer is None:
        try:
            from src.inference.unified_scorer import UnifiedScorer
            _unified_scorer = UnifiedScorer()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Unified scorer init failed: {e}")

    features_list = [tx.model_dump() for tx in txns]
    results = _unified_scorer.score_batch(
        features_list,
        user_ids=[tx.user_id for i, tx in enumerate(txns)],
    )
    return [{
        "fraud_probability": r.fraud_probability,
        "risk_score": r.risk_score,
        "decision": r.decision,
        "risk_level": r.risk_level,
        "ml_score": r.ml_score,
        "rules_score": r.rules_score,
        "rules_fired": r.rules_fired,
        "latency_ms": r.latency_ms,
    } for r in results]


@app.get("/unified-status")
async def unified_status():
    """Status of all unified detection subsystems."""
    global _unified_scorer
    if _unified_scorer is None:
        return {"loaded": False}
    return _unified_scorer.get_status()


# ── Drift Detection Endpoints ──────────────────────────────────────────────

@app.get("/drift/status")
async def drift_status():
    """PSI drift detection status — feature distribution monitoring."""
    global _unified_scorer
    if _unified_scorer is None:
        return {"drift_detector_ready": False, "error": "scorer not loaded"}
    return _unified_scorer.get_drift_status()


@app.get("/drift/history")
async def drift_history(limit: int = 20):
    """Recent drift check history."""
    global _unified_scorer
    if _unified_scorer is None:
        return {"error": "scorer not loaded"}
    return {"history": _unified_scorer.get_drift_history(limit)}


@app.post("/drift/check")
async def drift_check():
    """Force an immediate drift check on the current buffer."""
    global _unified_scorer
    if _unified_scorer is None:
        return {"error": "scorer not loaded"}
    result = _unified_scorer.force_drift_check()
    if result is None:
        return {"error": "no data in buffer or drift detector not ready"}
    return result


@app.get("/drift/alerts")
async def drift_alerts():
    """Recent drift alerts."""
    global _unified_scorer
    if _unified_scorer is None:
        return {"alerts": []}
    status = _unified_scorer.get_drift_status()
    return {"alerts": status.get("recent_alerts", []),
            "n_alerts": status.get("n_alerts", 0)}


# ── Rate Limiting Endpoints ────────────────────────────────────────────────

@app.get("/rate-limit/status")
async def rate_limit_status():
    """Rate limiter status and statistics."""
    limiter = getattr(app.state, "rate_limiter", None)
    if limiter is None:
        return {"enabled": False, "message": "Rate limiter not available (Redis required)"}
    return {"enabled": True, **limiter.get_stats()}


@app.get("/rate-limit/check/{user_id}")
async def rate_limit_check(user_id: str):
    """Check rate limit for a user without recording."""
    limiter = getattr(app.state, "rate_limiter", None)
    if limiter is None:
        return {"enabled": False}
    result = limiter.check_only(user_id)
    return {"enabled": True, **result.to_dict()}


@app.delete("/rate-limit/reset/{user_id}")
async def rate_limit_reset_user(user_id: str):
    """Reset a user's rate limit window."""
    limiter = getattr(app.state, "rate_limiter", None)
    if limiter is None:
        return {"error": "Rate limiter not available"}
    limiter.reset_user(user_id)
    return {"ok": True, "user_id": user_id, "message": "Rate limit reset"}


@app.delete("/rate-limit/reset-global")
async def rate_limit_reset_global():
    """Reset the global rate limit window."""
    limiter = getattr(app.state, "rate_limiter", None)
    if limiter is None:
        return {"error": "Rate limiter not available"}
    limiter.reset_global()
    return {"ok": True, "message": "Global rate limit reset"}


# ── Batch Scorer Endpoints ──

from src.inference.batch_scorer import get_batch_scorer  # noqa: E402


class BatchScorerRequest(BaseModel):
    """Batch scoring request for the optimized XGB+LGB ensemble."""
    transactions: list[dict]
    model_version: str | None = None


class BatchScorerResponse(BaseModel):
    """Batch scoring result with latency breakdown."""
    results: list[dict]
    batch_size: int
    total_latency_ms: float
    per_txn_ms: float
    model_version: str
    features_used: int


class BenchmarkRequest(BaseModel):
    """Latency benchmark request."""
    n_transactions: int = 1000
    n_iterations: int = 10
    batch_sizes: list[int] | None = None


class BenchmarkResponse(BaseModel):
    """Latency benchmark results."""
    benchmarks: list[dict]
    model_version: str
    summary: dict


@app.get("/batch/model-info")
async def batch_model_info():
    """Get metadata for the batch scoring model."""
    scorer = get_batch_scorer()
    if not scorer.is_loaded:
        try:
            scorer.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model load failed: {e}")
    return {
        "status": "ok",
        "model": scorer.version.to_dict(),
        "stats": scorer.stats,
    }


@app.post("/batch/score", response_model=BatchScorerResponse)
async def batch_score(req: BatchScorerRequest):
    """Score a batch of transactions using the optimized ensemble.

    Accepts ML_FEATURES dicts (from Privacy Layer or synthetic data).
    Returns per-transaction scores with latency breakdown.
    """
    scorer = get_batch_scorer()
    if not scorer.is_loaded:
        try:
            scorer.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model load failed: {e}")

    if not req.transactions:
        raise HTTPException(status_code=400, detail="Empty transaction list")

    if len(req.transactions) > 50_000:
        raise HTTPException(status_code=400, detail="Batch too large (max 50,000)")

    t0 = time.perf_counter()
    results = scorer.score_batch(req.transactions)
    total_ms = (time.perf_counter() - t0) * 1000

    return BatchScorerResponse(
        results=results,
        batch_size=len(req.transactions),
        total_latency_ms=round(total_ms, 2),
        per_txn_ms=round(total_ms / len(req.transactions), 3),
        model_version=scorer.version.version,
        features_used=scorer.version.n_features,
    )


@app.post("/batch/benchmark", response_model=BenchmarkResponse)
async def batch_benchmark(req: BenchmarkRequest):
    """Run latency benchmark across multiple batch sizes.

    Generates synthetic transactions and measures throughput.
    Results include p50/p95/p99 latencies and TPS.
    """
    scorer = get_batch_scorer()
    if not scorer.is_loaded:
        try:
            scorer.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model load failed: {e}")

    benchmarks = scorer.benchmark(
        n_transactions=req.n_transactions,
        n_iterations=req.n_iterations,
        batch_sizes=req.batch_sizes,
    )

    # Summary
    best = max(benchmarks, key=lambda b: b.throughput_tps)
    smallest = min(benchmarks, key=lambda b: b.batch_size)
    summary = {
        "peak_throughput_tps": round(best.throughput_tps, 1),
        "peak_batch_size": best.batch_size,
        "min_latency_per_txn_ms": round(smallest.per_txn_ms, 3),
        "min_latency_batch_size": smallest.batch_size,
        "model_version": scorer.version.version,
    }

    return BenchmarkResponse(
        benchmarks=[b.to_dict() for b in benchmarks],
        model_version=scorer.version.version,
        summary=summary,
    )


@app.get("/batch/stats")
async def batch_stats():
    """Get batch scorer cumulative statistics."""
    scorer = get_batch_scorer()
    return {
        "status": "ok",
        **scorer.stats,
    }
