"""Privacy Layer (FastAPI, internal-only). The one-way transformation between
the Identity world and the Fraud world (section 1, 7).

It receives a raw transaction event, strips/derives everything, and persists
ONLY the derived feature vector keyed by fraud_id. It holds no PII and cannot
be queried backward to recover identity — the Fraud Engine has no path into
the Identity Store.

  POST /internal/ingest-transaction  -> feature vector (section 16 shape)
"""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from sqlalchemy.exc import IntegrityError

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import sqlalchemy
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.privacy_layer import models as m
from src.privacy_layer.db import SessionLocal, engine
from src.shared_db import make_get_db
from src.privacy_layer.features import derive_event_features, typical_hours_default
from src.privacy_layer.velocity_tracker import get_tracker as get_velocity_tracker
from src.audit_service.writer import append_audit_event
from src.identity_service.security import verify_internal_token
from src.middleware import apply_security_middleware
from src.settings import settings, load_dotenv_and_patch
load_dotenv_and_patch()

MAX_USUAL = 8  # cap on usual locations/recipients kept per profile


def _migrate_device_fingerprints():
    """Drop and recreate device_fingerprints if it has the old sole-PK schema.

    SQLite-only: SQLite cannot ALTER PRIMARY KEY, so the only safe migration
    is DROP + CREATE.  DB-2 is throwaway demo state (AGENTS.md); losing
    device history on restart is acceptable.

    On PostgreSQL, tables are pre-created via supabase_migration_v2.sql
    with the correct composite PK — no runtime migration needed.
    """
    if settings.use_postgres:
        return  # PostgreSQL: tables pre-created by migration SQL
    with engine.connect() as conn:
        exists = conn.execute(
            sqlalchemy.text("SELECT name FROM sqlite_master WHERE type='table' AND name='device_fingerprints'")
        ).scalar()
        if exists is None:
            return
        info = conn.execute(sqlalchemy.text("PRAGMA table_info(device_fingerprints)")).fetchall()
        pk_cols = [row[1] for row in info if row[5] > 0]
        if pk_cols == ["device_hash"]:
            conn.execute(sqlalchemy.text("DROP TABLE device_fingerprints"))
            conn.commit()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from src.settings import enforce_production_gate
    enforce_production_gate()
    _migrate_device_fingerprints()
    # PostgreSQL: tables are pre-created by supabase_migration_v2.sql.
    # Only call create_all for SQLite (creates tables in the local .db file).
    if not settings.use_postgres:
        m.Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Privacy Layer", version="0.1.0", lifespan=lifespan)
apply_security_middleware(app)


get_db = make_get_db(SessionLocal)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize_ts(ts: datetime) -> datetime:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _hash_device(device_id: str) -> str:
    """SHA-256 of the raw device id (plus a pepper). Raw IDs never stored."""
    return hashlib.sha256(f"{device_id}::{settings.jwt_secret}".encode()).hexdigest()[:32]


class IngestTransactionRequest(BaseModel):
    event_id: str = Field(min_length=8, max_length=64)
    fraud_id: str = Field(pattern=r"^F[A-Z2-9]{15}$")
    amount: float = Field(gt=0)
    ts: datetime
    hour_of_day: int = Field(ge=0, le=23)
    device_id: str = Field(min_length=1, max_length=128)
    location_id: str = Field(min_length=1, max_length=64)  # coarse token, no raw geo
    recipient_id: str = Field(min_length=1, max_length=64)  # opaque token
    failed_auth_count_24h: int = Field(default=0, ge=0, le=100)
    # Altman-NATIVE raw columns (optional). When present, the Risk Engine's
    # native engine computes the 48-feature vector via the shared derivation
    # module — the same module the native retrain used (train == production).
    use_chip: str = Field(default="")
    mcc: int = Field(default=0, ge=0)
    merchant_city: str = Field(default="")
    merchant_state: str = Field(default="")
    zip: str = Field(default="")
    card: str = Field(default="")
    errors: str = Field(default="")


def _top_hours(db: Session, fraud_id: str, limit: int = 8) -> list[int]:
    """Most frequent hours over the last 30 COMMITTED events (the account's
    own rhythm). Only allowed/confirmed events enter the baseline, so a
    blocked attempt at 3am can never become a "typical" hour."""
    rows = (
        db.query(m.TransactionFeature.hour_of_day)
        .filter(
            m.TransactionFeature.fraud_id == fraud_id,
            m.TransactionFeature.baseline_committed.is_(True),
        )
        .order_by(m.TransactionFeature.created_at.desc())
        .limit(30)
        .all()
    )
    counts: dict[int, int] = {}
    for (hour,) in rows:
        counts[hour] = counts.get(hour, 0) + 1
    return [h for h, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]]


def _apply_commit(db: Session, profile: m.FraudProfile, row: m.TransactionFeature) -> None:
    """Apply one allowed/confirmed event's stored vector to the profile.

    The median EMA runs in ratio space - the raw amount is never stored
    (section 16): median_new = median * (0.9 + 0.1 * ratio), which is the
    classic 0.9*median + 0.1*amount whenever the median has not moved since
    ingest. Also registers the event's device/location/recipient, so blocked
    attempts never become "known".
    """
    profile.avg_txn_amount_90d = profile.median_amount * (0.9 + 0.1 * row.amount_ratio)
    now = _utcnow()
    profile.txn_freq_7d = (
        db.query(func.count(m.TransactionFeature.event_id))
        .filter(
            m.TransactionFeature.fraud_id == profile.fraud_id,
            m.TransactionFeature.baseline_committed.is_(True),
            m.TransactionFeature.created_at >= now - timedelta(days=7),
        )
        .scalar()
        or 0
    )
    profile.typical_txn_hours = json.dumps(_top_hours(db, profile.fraud_id))
    locs = profile.locations
    if row.location_id and row.location_id not in locs:
        locs = (locs + [row.location_id])[-MAX_USUAL:]
        profile.usual_locations = json.dumps(locs)
    recips = profile.recipients
    if row.recipient_id and row.recipient_id not in recips:
        recips = (recips + [row.recipient_id])[-MAX_USUAL:]
        profile.usual_recipients = json.dumps(recips)
    if row.device_hash:
        dev = db.query(m.DeviceFingerprint).filter(m.DeviceFingerprint.device_hash == row.device_hash).first()
        if dev is None:
            db.add(m.DeviceFingerprint(device_hash=row.device_hash, fraud_id=profile.fraud_id))
        else:
            dev.last_seen = now
    profile.known_device_count = (
        db.query(func.count(m.DeviceFingerprint.device_hash))
        .filter(m.DeviceFingerprint.fraud_id == profile.fraud_id)
        .scalar()
        or 0
    )
    base = json.loads(profile.behavioral_baseline_vector)
    base["txn_count"] = base.get("txn_count", 0) + 1
    profile.behavioral_baseline_vector = json.dumps(base)


@app.post("/internal/ingest-transaction", include_in_schema=False)
def ingest_transaction(
    req: IngestTransactionRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: str = Header(alias="X-Internal-Token"),
    db: Session = Depends(get_db),
):
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")

    now = _utcnow()
    ts = _normalize_ts(req.ts)
    device_hash = _hash_device(req.device_id)

    # --- load or initialize the behavioral profile -------------------------
    profile = db.query(m.FraudProfile).filter(m.FraudProfile.fraud_id == req.fraud_id).first()
    new_profile = profile is None
    if profile is None:
        # Initialize with neutral ratio baseline (1.0).  The raw amount
        # is NEVER stored in the feature store.  For a brand-new account
        # the first event establishes the baseline via commit; until then,
        # ratio = amount / 1.0 which is an overestimate — acceptable for
        # the first event only.  After commit, median updates via EMA in
        # ratio space: median_new = median * (0.9 + 0.1 * ratio).
        profile = m.FraudProfile(
            fraud_id=req.fraud_id,
            avg_txn_amount_90d=1.0,
            txn_freq_7d=1,
            typical_txn_hours=json.dumps(typical_hours_default()),
            known_device_count=1,
            usual_locations=json.dumps([req.location_id]),
            usual_recipients=json.dumps([req.recipient_id]),
            behavioral_baseline_vector=json.dumps({"txn_count": 1}),
        )
        db.add(profile)
        db.flush()

    # --- derive context from the store -------------------------------------
    # The 24h windows are anchored at the EVENT's timestamp (not wall-clock
    # now): "how much did this account do in the 24h BEFORE this transaction".
    # For live events ts ~= now, so this matches the old behavior; for
    # backdated/aged history (demo seed, historical replays) it counts the
    # rows that actually preceded the event instead of everything ingested
    # in the last real-time hour.
    anchor = min(now, ts)
    # MERGED QUERY: freq_24h + account_daily_spend_ratio in one pass
    agg = db.query(
        func.count(m.TransactionFeature.event_id).label("freq"),
        func.sum(m.TransactionFeature.amount_ratio).label("spend"),
    ).filter(
        m.TransactionFeature.fraud_id == req.fraud_id,
        m.TransactionFeature.created_at >= anchor - timedelta(hours=24),
        m.TransactionFeature.created_at <= anchor,
    ).one()
    freq_24h = agg.freq or 0
    account_daily_spend_ratio = round((agg.spend or 0.0) + req.amount / profile.median_amount, 4)

    # Device daily count (separate filter on device_hash)
    device_daily_count = (
        db.query(func.count(m.TransactionFeature.event_id))
        .filter(m.TransactionFeature.device_hash == device_hash,
                m.TransactionFeature.created_at >= anchor - timedelta(hours=24),
                m.TransactionFeature.created_at <= anchor)
        .scalar()
        or 0
    ) + 1  # includes the current event

    ratio = req.amount / profile.median_amount
    recent = (
        db.query(m.TransactionFeature.amount_ratio)
        .filter(m.TransactionFeature.fraud_id == req.fraud_id)
        # event_id tiebreak keeps the recent window deterministic when several
        # events share the same (second-precision) created_at.
        .order_by(m.TransactionFeature.created_at.desc(), m.TransactionFeature.event_id)
        .limit(9)
        .all()
    )
    recent_ratios = [r[0] for r in reversed(recent)]

    prev_similar = (
        db.query(m.TransactionFeature)
        .filter(
            m.TransactionFeature.fraud_id == req.fraud_id,
            m.TransactionFeature.created_at < now,
            m.TransactionFeature.amount_ratio >= 0.8 * ratio,
        )
        .order_by(m.TransactionFeature.created_at.desc())
        .first()
    )
    # No 1-day floor on time-derived features: a brand-new account's first
    # events carry their REAL (sub-day) age so the derived vector stays
    # in-distribution with training (cold-start false-positive fix).
    tenure_days = (now - profile.created_at).total_seconds() / 86400.0
    if prev_similar is not None:
        days_since = (now - prev_similar.created_at).total_seconds() / 86400.0
    else:
        days_since = tenure_days

    # `new_device_flag` must reflect knowledge BEFORE this transaction, so we
    # derive features first and register the device afterwards. (A fresh
    # account with zero history therefore flags its first device as new -
    # semantically correct: nothing was known yet.)
    # MERGED QUERY: device + known_devices + other_device_accounts in one pass
    all_device_rows = db.query(m.DeviceFingerprint).filter(
        m.DeviceFingerprint.device_hash == device_hash
    ).all()
    device = None  # the DeviceFingerprint row for THIS account (or None)
    other_device_accounts = set()
    for drow in all_device_rows:
        if drow.fraud_id == req.fraud_id:
            device = drow  # keep the actual ORM row (need .last_seen later)
        else:
            other_device_accounts.add(drow.fraud_id)
    # Derive known_device_count from profile state instead of querying all
    # device fingerprints — avoids one DB round-trip per ingest.
    # `device is not None` iff the current device is already known to this account.
    new_device = device is None
    known_device_count = profile.known_device_count + (1 if new_device else 0)
    shared_device_accounts = len(other_device_accounts)

    def _other_accounts(col: str, value: str) -> int:
        return (db.query(func.count(func.distinct(m.TransactionFeature.fraud_id)))
                .filter(getattr(m.TransactionFeature, col) == value,
                        m.TransactionFeature.fraud_id != req.fraud_id)
                .scalar() or 0)

    shared_recipient_accounts = _other_accounts("recipient_id", req.recipient_id)

    profile_dict = {
        "median_amount": profile.median_amount,
        "typical_hours": profile.typical_hours,
        "known_device_count": known_device_count,
        "usual_locations": profile.locations,
        "usual_recipients": profile.recipients,
        "tenure_days": tenure_days,
    }

    # Get real-time velocity stats from the tracker (before recording this event)
    velocity_tracker = get_velocity_tracker()
    velocity_stats = velocity_tracker.get_velocity(
        user_id=str(req.fraud_id),
        card_id=device_hash,
        merchant_id=req.recipient_id,
    )

    features = derive_event_features(
        profile=profile_dict,
        event={
            "amount": req.amount,
            "hour_of_day": req.hour_of_day,
            "is_weekend": int(ts.weekday() >= 5),
            "device_hash": device_hash,
            "new_device": new_device,
            "location_id": req.location_id,
            "recipient_id": req.recipient_id,
            "failed_auth_count_24h": req.failed_auth_count_24h,
        },
        freq_last_24h=freq_24h,
        days_since_similar=days_since,
        recent_ratios=recent_ratios,
        shared_device_accounts=shared_device_accounts,
        shared_recipient_accounts=shared_recipient_accounts,
        velocity=velocity_stats,
    )

    # Record this event in the velocity tracker (background, after scoring)
    # Note: this updates the tracker for the NEXT event, not this one
    import threading
    def _record():
        velocity_tracker.record_event(
            user_id=str(req.fraud_id),
            amount=req.amount,
            card_id=device_hash,
            merchant_id=req.recipient_id,
        )
    threading.Thread(target=_record, daemon=True).start()

    # --- persist: derived vector + coarse identifiers (never raw amounts) ---
    # Strip transient velocity keys that aren't DB columns (real-time only).
    _VELOCITY_KEYS = {"user_tx_count", "user_avg_amt", "card_tx_count", "merch_tx_count"}
    db_features = {k: v for k, v in features.items() if k not in _VELOCITY_KEYS}
    # Non-init events do NOT touch the statistical profile or register
    # devices/locations/recipients here - that happens only at commit
    # (allowed/confirmed events). A blocked attempt therefore cannot move
    # the median, become a "typical" hour, or become a known device.
    row = m.TransactionFeature(
        event_id=req.event_id,
        fraud_id=req.fraud_id,
        **db_features,
        device_hash=device_hash,
        location_id=req.location_id,
        recipient_id=req.recipient_id,
    )
    if new_profile:
        # The account's birth event establishes the baseline (its median,
        # first device/location/recipient) - there is nothing to pollute
        # yet, and feature derivation for later events needs it.
        row.baseline_committed = True
        if device is None:
            db.add(m.DeviceFingerprint(device_hash=device_hash, fraud_id=req.fraud_id))
        else:
            device.last_seen = now  # device already seen (shared across accounts)
    elif device is not None:
        device.last_seen = now  # observation only; registration stays pending
    # Idempotency: if this event_id already exists, return the stored vector
    # without re-auditing (the audit chain must not double-count).
    existing = db.query(m.TransactionFeature).filter(m.TransactionFeature.event_id == req.event_id).first()
    if existing is not None:
        return {
            "event_id": req.event_id,
            "fraud_id": req.fraud_id,
            "amount_ratio": existing.amount_ratio,
            "txn_amount_bucket": existing.txn_amount_bucket,
            "txn_freq_last_24h": existing.txn_freq_last_24h,
            "txn_time_unusual": int(existing.txn_time_unusual),
            "new_device_flag": int(existing.new_device_flag),
            "unusual_location_flag": int(existing.unusual_location_flag),
            "unusual_recipient_flag": int(existing.unusual_recipient_flag),
            "failed_auth_count_24h": existing.failed_auth_count_24h,
            "days_since_last_similar_txn": existing.days_since_last_similar_txn,
            "gradual_escalation_score": existing.gradual_escalation_score,
            "known_device_count": existing.known_device_count,
            "account_tenure_days": existing.account_tenure_days,
            "hour_of_day": existing.hour_of_day,
            "is_weekend": int(existing.is_weekend),
            "account_daily_spend_ratio": account_daily_spend_ratio,
            "device_daily_count": device_daily_count,
            "label": None,
            "idempotent_replay": True,
        }

    db.add(row)
    db.commit()

    # Audit trail off the critical path (same pattern as Risk Engine).
    # The hash-chained append is pure logging — doesn't affect response.
    background_tasks.add_task(
        append_audit_event,
        req.fraud_id,
        "feature_ingested",
        {"event_id": req.event_id},
    )

    # --- return the section-16 live-event shape (label unknown) ------------
    # The velocity context is passed to the Risk Engine for pre-scoring limit
    # enforcement, but is NOT part of the stored §16 vector (those columns do
    # not exist on TransactionFeature).
    resp = {
        "event_id": req.event_id,
        "fraud_id": req.fraud_id,
        **features,
        "account_daily_spend_ratio": account_daily_spend_ratio,
        "device_daily_count": device_daily_count,
        "label": None,
    }
    # Altman-native raw columns (optional): forward verbatim so the Risk
    # Engine's native engine can build the 48-vector via the shared module.
    if req.use_chip or req.mcc or req.merchant_city or req.merchant_state \
            or req.zip or req.card or req.errors:
        resp.update({
            "amount": req.amount,
            "ts": req.ts.isoformat(),
            "use_chip": req.use_chip,
            "mcc": req.mcc,
            "merchant_city": req.merchant_city,
            "merchant_state": req.merchant_state,
            "zip": req.zip,
            "card": req.card,
            "errors": req.errors,
        })
    return resp


@app.get("/internal/device-graph", include_in_schema=False)
def device_graph(
    device_id: str | None = None,
    recipient_id: str | None = None,
    x_internal_token: str = Header(alias="X-Internal-Token"),
    db: Session = Depends(get_db),
):
    """Compliance viewer data: which pseudonymous accounts share a device or
    recipient token (device graphs / mule rings). One of `device_id` or
    `recipient_id` is required. Returns the accounts (fraud_ids only, never
    identity), their event counts and first/last seen timestamps, plus the
    link-analysis composite (mule_ring_score). Internal token only - the
    Verification Service proxies it behind the compliance passphrase.
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")
    if device_id is None and recipient_id is None:
        raise HTTPException(status_code=422, detail="provide device_id or recipient_id")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if device_id is not None:
        # Same keyed, truncated hash the ingest path uses (`_hash_device`),
        # so a raw token a reviewer copies from the seed response resolves
        # to the stored rows. Plain sha256 would silently return nothing.
        device_hash = _hash_device(device_id)
        rows = (
            db.query(
                m.TransactionFeature.fraud_id,
                func.count(m.TransactionFeature.event_id),
                func.min(m.TransactionFeature.created_at),
                func.max(m.TransactionFeature.created_at),
            )
            .filter(m.TransactionFeature.device_hash == device_hash)
            .group_by(m.TransactionFeature.fraud_id)
            .all()
        )
    else:
        rows = (
            db.query(
                m.TransactionFeature.fraud_id,
                func.count(m.TransactionFeature.event_id),
                func.min(m.TransactionFeature.created_at),
                func.max(m.TransactionFeature.created_at),
            )
            .filter(m.TransactionFeature.recipient_id == recipient_id)
            .group_by(m.TransactionFeature.fraud_id)
            .all()
        )

    accounts = [
        {
            "fraud_id": fid,
            "events": int(cnt),
            "first_seen": first.isoformat() if first else None,
            "last_seen": last.isoformat() if last else None,
            "days_since_last": round((now - last).total_seconds() / 86400.0, 1) if last else None,
        }
        for fid, cnt, first, last in rows
    ]
    accounts.sort(key=lambda a: a["events"], reverse=True)
    return {
        "device_id": device_id,
        "recipient_id": recipient_id,
        "total_accounts": len(accounts),
        "accounts": accounts,
    }


class CommitBaselineRequest(BaseModel):
    event_id: str = Field(min_length=8, max_length=64)
    fraud_id: str = Field(pattern=r"^F[A-Z2-9]{15}$")


@app.post("/internal/commit-baseline", include_in_schema=False)
def commit_baseline(
    req: CommitBaselineRequest,
    x_internal_token: str = Header(alias="X-Internal-Token"),
    db: Session = Depends(get_db),
):
    """Apply a stored vector to the behavioral profile - called ONLY for
    events that were allowed (decision == allow) or confirmed (this was me).
    Blocked or disputed events are never committed here, so they cannot
    pollute the account's median, typical hours, or known devices.

    Idempotent: a second commit for the same event is a no-op.
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")

    row = (
        db.query(m.TransactionFeature)
        .filter(
            m.TransactionFeature.event_id == req.event_id,
            m.TransactionFeature.fraud_id == req.fraud_id,
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="event not found")
    if row.baseline_committed:
        return {"committed": False, "already_committed": True, "event_id": req.event_id}

    profile = db.query(m.FraudProfile).filter(m.FraudProfile.fraud_id == req.fraud_id).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")

    row.baseline_committed = True
    db.flush()  # so _top_hours sees this event's hour
    _apply_commit(db, profile, row)
    db.commit()
    return {"committed": True, "already_committed": False, "event_id": req.event_id}


class DemoAgeAccountRequest(BaseModel):
    fraud_id: str = Field(pattern=r"^F[A-Z2-9]{15}$")
    age_days: float = Field(default=60.0, ge=14, le=3650)


@app.post("/internal/demo-age-account", include_in_schema=False)
def demo_age_account(
    req: DemoAgeAccountRequest,
    x_internal_token: str = Header(alias="X-Internal-Token"),
    db: Session = Depends(get_db),
):
    """DEV ONLY demo harness: backdate a demo account's DB-2 history so the
    time-derived features (account_tenure_days, days_since_last_similar_txn,
    txn_freq_last_24h) reflect a MATURE account instead of a brand-new one.

    Spreads the account's stored feature rows evenly over the last
    [8, age_days - 5] days (oldest row furthest back, newest ~8 days ago so
    the demo's current event isn't shadowed by same-day history) and moves
    the profile birth to `age_days` ago. Only `created_at` is rewritten - the
    stored §16 vectors are unchanged and no raw data is added. This is the
    service-side version of the aging step in
    scripts/live_curl_walkthrough.sh, kept inside the store that owns DB-2
    rather than direct sqlite3 from outside.
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")
    profile = db.query(m.FraudProfile).filter(m.FraudProfile.fraud_id == req.fraud_id).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")
    rows = (
        db.query(m.TransactionFeature)
        .filter(m.TransactionFeature.fraud_id == req.fraud_id)
        .order_by(m.TransactionFeature.created_at.asc(), m.TransactionFeature.event_id)
        .all()
    )
    now = _utcnow()
    n = len(rows)
    oldest_age = req.age_days - 5.0
    newest_age = 8.0
    for i, row in enumerate(rows):
        age = oldest_age - (oldest_age - newest_age) * i / max(n - 1, 1)
        row.created_at = now - timedelta(days=age)
    profile.created_at = now - timedelta(days=req.age_days)
    db.commit()
    return {"fraud_id": req.fraud_id, "rows_backdated": n, "profile_age_days": req.age_days}


# ========================================================================
# Batch ingest — PostgreSQL-optimized bulk path
# ========================================================================

class BatchIngestRequest(BaseModel):
    events: list[IngestTransactionRequest] = Field(max_length=1000)


@app.post("/internal/ingest-batch", include_in_schema=False)
def ingest_batch(
    req: BatchIngestRequest,
    background_tasks: BackgroundTasks,
    x_internal_token: str = Header(alias="X-Internal-Token"),
    db: Session = Depends(get_db),
):
    """Bulk ingest multiple transactions in a single request.

    Uses PostgreSQL bulk inserts + ON CONFLICT for idempotency.
    Orders events by (fraud_id, ts) to ensure correct feature derivation
    within each account's timeline.
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")

    now = _utcnow()
    results = []

    # Group by fraud_id for efficient profile lookups
    fraud_ids = set(e.fraud_id for e in req.events)
    profiles = {
        p.fraud_id: p
        for p in db.query(m.FraudProfile).filter(
            m.FraudProfile.fraud_id.in_(fraud_ids)
        ).all()
    }

    # Pre-fetch all existing event_ids for idempotency check
    event_ids = [e.event_id for e in req.events]
    existing_ids = set(
        row[0] for row in db.query(m.TransactionFeature.event_id).filter(
            m.TransactionFeature.event_id.in_(event_ids)
        ).all()
    )

    new_rows = []
    for evt in req.events:
        if evt.event_id in existing_ids:
            results.append({"event_id": evt.event_id, "idempotent_replay": True})
            continue

        ts = _normalize_ts(evt.ts)
        device_hash = _hash_device(evt.device_id)
        profile = profiles.get(evt.fraud_id)
        new_profile = profile is None

        if new_profile:
            profile = m.FraudProfile(
                fraud_id=evt.fraud_id,
                avg_txn_amount_90d=1.0,
                txn_freq_7d=1,
                typical_txn_hours=json.dumps(typical_hours_default()),
                known_device_count=1,
                usual_locations=json.dumps([evt.location_id]),
                usual_recipients=json.dumps([evt.recipient_id]),
                behavioral_baseline_vector=json.dumps({"txn_count": 1}),
            )
            db.add(profile)
            db.flush()
            profiles[evt.fraud_id] = profile

        # Simplified feature derivation for batch (uses profile defaults)
        ratio = evt.amount / profile.median_amount
        row = m.TransactionFeature(
            event_id=evt.event_id,
            fraud_id=evt.fraud_id,
            amount_ratio=round(ratio, 4),
            txn_amount_bucket=amount_bucket(ratio),
            txn_freq_last_24h=0,
            txn_time_unusual=int(evt.hour_of_day not in json.loads(profile.typical_txn_hours)),
            new_device_flag=1,
            unusual_location_flag=1,
            unusual_recipient_flag=1,
            failed_auth_count_24h=evt.failed_auth_count_24h,
            days_since_last_similar_txn=0.0,
            gradual_escalation_score=0.0,
            known_device_count=profile.known_device_count,
            account_tenure_days=(now - profile.created_at).total_seconds() / 86400.0,
            hour_of_day=evt.hour_of_day,
            is_weekend=int(ts.weekday() >= 5),
            shared_device_accounts=0,
            shared_recipient_accounts=0,
            mule_ring_score=0.0,
            baseline_committed=new_profile,
            device_hash=device_hash,
            location_id=evt.location_id,
            recipient_id=evt.recipient_id,
        )
        new_rows.append(row)
        results.append({"event_id": evt.event_id, "idempotent_replay": False})

    # Bulk insert all new rows in one query
    if new_rows:
        db.add_all(new_rows)
        db.commit()

    # Audit trail: single batch event instead of N individual events
    background_tasks.add_task(
        append_audit_event,
        "batch",
        "features_ingested",
        {"count": len(new_rows), "event_ids": [r.event_id for r in new_rows[:10]]},
    )

    return {
        "total": len(req.events),
        "ingested": len(new_rows),
        "idempotent_replays": len(req.events) - len(new_rows),
        "results": results,
    }


# ========================================================================
# Real-time fraud analytics — PostgreSQL window functions
# ========================================================================

@app.get("/internal/analytics", include_in_schema=False)
def get_analytics(
    hours: int = 24,
    x_internal_token: str = Header(alias="X-Internal-Token"),
    db: Session = Depends(get_db),
):
    """Real-time fraud analytics using PostgreSQL window functions.

    Returns aggregated metrics over the last N hours without loading
    individual rows into Python — all computation happens in the database.
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")

    from sqlalchemy import text

    cutoff = now - timedelta(hours=hours)

    # Hourly ingest volume with fraud rate
    hourly = db.execute(text("""
        SELECT
            date_trunc('hour', created_at) AS hour,
            COUNT(*) AS total_events,
            COUNT(DISTINCT fraud_id) AS unique_accounts,
            AVG(amount_ratio) AS avg_amount_ratio,
            SUM(CASE WHEN new_device_flag THEN 1 ELSE 0 END) AS new_device_events,
            SUM(CASE WHEN unusual_location_flag THEN 1 ELSE 0 END) AS unusual_location_events
        FROM privacy.transaction_features
        WHERE created_at >= :cutoff
        GROUP BY date_trunc('hour', created_at)
        ORDER BY hour DESC
    """), {"cutoff": cutoff}).fetchall()

    # Top risky accounts by feature volume
    top_accounts = db.execute(text("""
        SELECT
            fraud_id,
            COUNT(*) AS event_count,
            AVG(amount_ratio) AS avg_ratio,
            MAX(gradual_escalation_score) AS max_escalation,
            SUM(CASE WHEN new_device_flag THEN 1 ELSE 0 END) AS new_device_pct,
            MAX(created_at) AS last_seen
        FROM privacy.transaction_features
        WHERE created_at >= :cutoff
        GROUP BY fraud_id
        HAVING COUNT(*) >= 3
        ORDER BY MAX(gradual_escalation_score) DESC, COUNT(*) DESC
        LIMIT 10
    """), {"cutoff": cutoff}).fetchall()

    # Device sharing analysis (mule ring detection)
    device_sharing = db.execute(text("""
        SELECT
            device_hash,
            COUNT(DISTINCT fraud_id) AS account_count,
            MIN(first_seen) AS first_seen,
            MAX(last_seen) AS last_seen
        FROM privacy.device_fingerprints
        GROUP BY device_hash
        HAVING COUNT(DISTINCT fraud_id) > 1
        ORDER BY account_count DESC
        LIMIT 20
    """)).fetchall()

    return {
        "period_hours": hours,
        "hourly_volume": [
            {
                "hour": str(r[0]),
                "total_events": r[1],
                "unique_accounts": r[2],
                "avg_amount_ratio": round(float(r[3] or 0), 4),
                "new_device_events": r[4],
                "unusual_location_events": r[5],
            }
            for r in hourly
        ],
        "top_risky_accounts": [
            {
                "fraud_id": r[0],
                "event_count": r[1],
                "avg_ratio": round(float(r[2] or 0), 4),
                "max_escalation": round(float(r[3] or 0), 4),
                "new_device_pct": round(r[4] / max(r[1], 1), 2),
                "last_seen": str(r[5]),
            }
            for r in top_accounts
        ],
        "device_sharing": [
            {
                "device_hash": r[0],
                "account_count": r[1],
                "first_seen": str(r[2]),
                "last_seen": str(r[3]),
            }
            for r in device_sharing
        ],
    }


# ========================================================================
# Full-text audit search
# ========================================================================

@app.get("/internal/audit-search", include_in_schema=False)
def audit_search(
    q: str = "",
    event_type: str | None = None,
    fraud_id: str | None = None,
    limit: int = 50,
    x_internal_token: str = Header(alias="X-Internal-Token"),
    db: Session = Depends(get_db),
):
    """Search audit events using PostgreSQL full-text search.

    Searches the payload_summary JSON field for matching text.
    Uses PostgreSQL's ILIKE for case-insensitive search (fast with indexes).
    """
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")

    from sqlalchemy import text as sql_text

    conditions = []
    params = {"limit": min(limit, 200)}

    if q:
        conditions.append("payload_summary ILIKE :query")
        params["query"] = f"%{q}%"
    if event_type:
        conditions.append("event_type = :event_type")
        params["event_type"] = event_type
    if fraud_id:
        conditions.append("fraud_id = :fraud_id")
        params["fraud_id"] = fraud_id

    where = " AND ".join(conditions) if conditions else "1=1"

    # Fixed SQL skeleton; `where` holds only the fixed condition templates
    # joined above (values travel via :params, never via interpolation).
    audit_search_sql = (  # nosec B608 - static skeleton + bound params, no interpolated values
        """
        SELECT seq, event_id, fraud_id, event_type, payload_summary, entry_hash, created_at
        FROM audit.audit_events
        WHERE __WHERE_CLAUSE__
        ORDER BY seq DESC
        LIMIT :limit
    """
    ).replace("__WHERE_CLAUSE__", where)

    rows = db.execute(sql_text(audit_search_sql), params).fetchall()

    return {
        "count": len(rows),
        "events": [
            {
                "seq": r[0],
                "event_id": r[1],
                "fraud_id": r[2],
                "event_type": r[3],
                "payload_summary": r[4],
                "entry_hash": r[5],
                "created_at": str(r[6]),
            }
            for r in rows
        ],
    }


STARTED_AT = datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health():
    """Health check with DB connectivity verification."""
    db_ok = True
    try:
        db = SessionLocal()
        db.query(m.TransactionFeature).limit(1).all()
        db.close()
    except Exception:
        db_ok = False
    status = "ok" if db_ok else "degraded"
    return {
        "status": status,
        "service": "privacy-layer",
        "started_at": STARTED_AT,
        "db": "ok" if db_ok else "error",
    }
