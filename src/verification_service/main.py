"""Verification Service (FastAPI, user-facing). The "AI recommends,
verification decides" loop (architecture section 1, 8, 14).

  GET  /                        - the static verification UI
  GET  /alerts                  - high-risk alerts for the logged-in user
  POST /alerts/{event_id}/confirm { outcome: this_was_me | this_wasnt_me }
  GET  /alerts/{event_id}/reason - category-level explanation (section 11)
  POST /demo/seed               - DEV ONLY: chain Privacy -> Risk to create
                                   demo alerts for the logged-in user
  GET  /compliance/events       - pseudonymous decision trail (compliance
                                   role, proxied from the Audit Service)
  GET  /compliance/integrity    - hash-chain integrity verification

Session auth: Bearer JWT issued by the Identity Service, carrying only the
`fraud_id` claim. Alerts are scoped to the caller's own fraud_id. The
compliance endpoints are gated by a separate compliance passphrase (dev
replacement for real RBAC + mTLS, section 3) and proxy to the Audit
Service, which logs every read - the audit of the audit.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import os

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import desc, func, literal_column
from sqlalchemy.orm import Session
from starlette.responses import Response

from src.audit_service.writer import append_audit_event
from src.identity_service.security import decode_access_token
from src.risk_engine.models import RiskScore
from src.risk_engine.reason_codes import REASON_CODE_TEXT
from src.settings import settings, load_dotenv_and_patch
load_dotenv_and_patch()
from src.verification_service import models as m
from src.verification_service.db import SessionLocal, engine
from src.shared_db import make_get_db
from src.middleware import apply_security_middleware

STATIC_DIR = Path(__file__).resolve().parent / "static"


class NoCacheStaticFiles(StaticFiles):
    """Serve the UI assets with `Cache-Control: no-store`.

    The UI is a single-file dev app that changes in place; a browser cache
    (etag revalidation the preview webview skips) keeps serving stale
    index.html/api.js after an edit, which looks like broken/old behavior.
    No-store guarantees every load reflects the current code.
    """

    async def get_response(self, path: str, scope):
        response: Response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response

RECOVERY_STEPS = [
    "Freeze the affected payment channel",
    "Force re-authentication on all sessions",
    "Review recent devices and sessions",
    "Notify the Identity Service for a secure recovery flow",
    "A case has been opened in the audit trail",
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from src.settings import enforce_production_gate
    enforce_production_gate()
    if not settings.use_postgres:
        m.RiskBase.metadata.create_all(bind=engine)
        # Investigator cases live in the SHARED risk store next to the
        # risk_scores they reference (audit #30 moved the workflow off the
        # separate verify.db, where score_id FK targets never existed).
        # Ensure the shared store has the tables + the per-event dedup index.
        from src.risk_engine.db import engine as risk_store_engine
        m.RiskBase.metadata.create_all(bind=risk_store_engine)
        try:
            from sqlalchemy import text as _text
            with risk_store_engine.begin() as _conn:
                _dup = _conn.execute(_text(
                    "SELECT event_id, COUNT(*) c FROM investigator_cases "
                    "GROUP BY event_id HAVING c > 1 LIMIT 1"
                )).fetchone()
                if _dup:
                    print(f"[verification] WARNING: {_dup[1]} duplicate "
                          f"investigator_cases for event {_dup[0]} - dedup "
                          "index NOT created (alert-quality audit #30)")
                else:
                    _conn.execute(_text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS "
                        "uq_investigator_cases_event_id "
                        "ON investigator_cases(event_id)"
                    ))
        except Exception as _e:  # noqa: BLE001 - never block startup on DDL
            print(f"[verification] WARNING: could not ensure case dedup index: {_e}")
    yield


# Stable, readable operationIds in the OpenAPI schema (default embeds the
# path: `confirm_alert_alerts__event_id__confirm_post`). The UI builds its
# API client from the schema, so operationId == the generated helper name.
def _clean_operation_id(route):
    return route.name


app = FastAPI(
    title="Verification Service",
    version="0.1.0",
    lifespan=lifespan,
    generate_unique_id_function=_clean_operation_id,
)
apply_security_middleware(app)
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

# ── Identity-service proxy (same-origin, avoids CSP cross-port blocks) ────
import httpx as _httpx

_IDENTITY_BASE = "http://127.0.0.1:8001"


@app.get("/identity/openapi.json")
async def proxy_identity_openapi():
    """Proxy identity OpenAPI spec (same-origin for CSP)."""
    async with _httpx.AsyncClient() as client:
        resp = await client.get(f"{_IDENTITY_BASE}/openapi.json", timeout=5.0)
    from fastapi.responses import JSONResponse
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/identity/auth/login")
async def proxy_identity_login(body: dict):
    """Proxy identity login."""
    async with _httpx.AsyncClient() as client:
        resp = await client.post(f"{_IDENTITY_BASE}/auth/login", json=body, timeout=10.0)
    from fastapi.responses import JSONResponse
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/identity/auth/register")
async def proxy_identity_register(body: dict):
    """Proxy identity register."""
    async with _httpx.AsyncClient() as client:
        resp = await client.post(f"{_IDENTITY_BASE}/auth/register", json=body, timeout=10.0)
    from fastapi.responses import JSONResponse
    return JSONResponse(content=resp.json(), status_code=resp.status_code)

get_db = make_get_db(SessionLocal)

# RiskScore lives in the risk engine's DB (DB-3), not the verification DB (DB-5).
# Wire a second session factory for cross-service RiskScore reads.
from src.risk_engine.db import SessionLocal as RiskSessionLocal
get_risk_db = make_get_db(RiskSessionLocal)


def current_fraud_id(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        payload = decode_access_token(authorization.removeprefix("Bearer "))
        return payload["sub"]  # extract fraud_id, not the whole dict
    except Exception:
        raise HTTPException(status_code=401, detail="invalid or expired token")


def require_compliance_token(x_compliance_token: str | None = Header(default=None, alias="X-Compliance-Token")) -> None:
    """Compliance-role gate for the audit viewer. Production: real RBAC +
    mTLS (section 3); the dev passphrase only exists for the prototype."""
    if not x_compliance_token or x_compliance_token != settings.compliance_token:
        raise HTTPException(status_code=401, detail="compliance role required")


def _audit_get(path: str, actor: str = "compliance-ui") -> tuple[int, dict]:
    """Proxy a read to the Audit Service (which logs every read)."""
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(
                f"{settings.audit_url}{path}",
                headers={"X-Internal-Token": settings.internal_token, "X-Audit-Actor": actor},
            )
            return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text})
    except httpx.HTTPError:
        return 503, {"detail": "audit service unreachable"}


@app.get("/compliance/integrity")
def compliance_integrity(_role=Depends(require_compliance_token)):
    """Hash-chain integrity: recomputed by the Audit Service (section 13)."""
    status, data = _audit_get("/audit/integrity")
    if status != 200:
        raise HTTPException(status_code=status, detail=data.get("detail", "audit service unavailable"))
    return data


@app.get("/compliance/events")
def compliance_events(
    limit: int = 100,
    offset: int = 0,
    _role=Depends(require_compliance_token),
):
    """Pseudonymous decision trail: scores, reason codes, outcomes, case IDs."""
    status, data = _audit_get(f"/audit/events?limit={limit}&offset={offset}")
    if status != 200:
        raise HTTPException(status_code=status, detail=data.get("detail", "audit service unavailable"))
    # Add category-level reason texts for the UI (section 11) - never
    # thresholds, weights, or raw probabilities.
    for ev in data.get("events", []):
        payload = ev.get("payload", {})
        if ev.get("event_type") == "score_generated" and payload.get("reason_codes"):
            payload["reason_texts"] = [REASON_CODE_TEXT.get(c, c) for c in payload["reason_codes"]]
    return data


@app.get("/compliance/overview")
def compliance_overview(
    limit: int = 100,
    offset: int = 0,
    _role=Depends(require_compliance_token),
):
    """BFF aggregate for the compliance view: chain integrity, the latest
    verification case, and a paged slice of events with the chain-wide
    aggregates - one round trip instead of integrity + events.
    """
    status, data = _audit_get(f"/audit/overview?limit={limit}&offset={offset}")
    if status != 200:
        raise HTTPException(status_code=status, detail=data.get("detail", "audit service unavailable"))
    return data


@app.get("/compliance/device-graph")
def compliance_device_graph(
    device_id: str | None = None,
    recipient_id: str | None = None,
    _role=Depends(require_compliance_token),
):
    """Compliance viewer: which pseudonymous accounts share a device or
    recipient token. Proxied from the Privacy Layer (DB-2 owner) behind the
    compliance passphrase - the Privacy Layer itself stays internal-only.
    """
    if device_id is None and recipient_id is None:
        raise HTTPException(status_code=422, detail="provide device_id or recipient_id")
    params = "&" .join(f"{k}={v}" for k, v in {"device_id": device_id, "recipient_id": recipient_id}.items() if v)
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(
                f"{settings.privacy_url}/internal/device-graph?{params}",
                headers={"X-Internal-Token": settings.internal_token},
            )
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=r.json().get("detail", "privacy layer error"))
            return r.json()
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="privacy layer unreachable")


@app.get("/", include_in_schema=False)
def index():
    resp = FileResponse(STATIC_DIR / "index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


class ConfirmRequest(BaseModel):
    outcome: Literal["this_was_me", "this_wasnt_me"]


def _alert_dict(score: RiskScore) -> dict:
    codes = json.loads(score.reason_codes) if score.reason_codes else []
    return {
        "event_id": score.event_id,
        "risk_score": score.risk_score,
        "risk_band": score.risk_band,
        "reason_codes": codes,
        "reason_texts": [REASON_CODE_TEXT.get(c, c) for c in codes],
        "scored_at": score.scored_at.isoformat() if score.scored_at else None,
    }


@app.get("/alerts")
def alerts(fraud_id: str = Depends(current_fraud_id), db: Session = Depends(get_db), risk_db: Session = Depends(get_risk_db)):
    rows = (
        risk_db.query(RiskScore)
        .filter(RiskScore.fraud_id == fraud_id, RiskScore.risk_band == "high")
        .order_by(desc(RiskScore.scored_at))
        .all()
    )
    out = []
    for s in rows:
        resolved = risk_db.query(m.VerificationOutcome).filter(m.VerificationOutcome.score_id == s.score_id).first()
        if resolved is None:
            out.append(_alert_dict(s))
    return {"fraud_id": fraud_id, "alerts": out}


@app.get("/alerts/{event_id}/reason")
def alert_reason(event_id: str, fraud_id: str = Depends(current_fraud_id), risk_db: Session = Depends(get_risk_db)):
    score = (
        risk_db.query(RiskScore).filter(RiskScore.event_id == event_id, RiskScore.fraud_id == fraud_id).first()
    )
    if score is None:
        raise HTTPException(status_code=404, detail="alert not found")
    return _alert_dict(score)


@app.post("/alerts/{event_id}/confirm")
def confirm_alert(
    event_id: str,
    req: ConfirmRequest,
    fraud_id: str = Depends(current_fraud_id),
    db: Session = Depends(get_db),
    risk_db: Session = Depends(get_risk_db),
):
    score = (
        risk_db.query(RiskScore).filter(RiskScore.event_id == event_id, RiskScore.fraud_id == fraud_id).first()
    )
    if score is None:
        raise HTTPException(status_code=404, detail="alert not found")
    if score.risk_band != "high":
        raise HTTPException(status_code=409, detail="no verification required for this event")
    existing = risk_db.query(m.VerificationOutcome).filter(m.VerificationOutcome.score_id == score.score_id).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="alert already resolved")

    outcome = "confirmed" if req.outcome == "this_was_me" else "disputed"
    case_id = f"PS14-{uuid.uuid4().hex[:8].upper()}"
    record = m.VerificationOutcome(
        score_id=score.score_id,
        fraud_id=fraud_id,
        event_id=event_id,
        outcome=outcome,
        case_id=case_id,
    )
    risk_db.add(record)
    risk_db.commit()

    # Hash-chained audit trail: human verification outcomes are logged
    # pseudonymously (section 13) - the label source for the retraining queue.
    append_audit_event(
        fraud_id,
        "verification_resolved",
        {"event_id": event_id, "outcome": outcome, "case_id": case_id, "verification_id": record.verification_id},
    )

    # The outcome (and the chain) changed: drop cached /overview payloads so
    # the next read shows the new case, feedback counts, and chain length.
    _invalidate_overview()

    # Baseline discipline: a CONFIRMED event is legitimate, so it enters the
    # account's behavioral profile (median, typical hours, device). A
    # DISPUTED event never does - it was blocked and stays out of the
    # baseline. Best-effort over HTTP with the internal token; verification
    # itself must not fail because the Privacy Layer is unreachable.
    baseline_updated = False
    if outcome == "confirmed":
        try:
            with httpx.Client(timeout=10.0) as client:
                cr = client.post(
                    f"{settings.privacy_url}/internal/commit-baseline",
                    json={"event_id": event_id, "fraud_id": fraud_id},
                    headers={"X-Internal-Token": settings.internal_token},
                )
                baseline_updated = cr.status_code == 200 and cr.json().get("committed") is True
        except httpx.HTTPError:
            baseline_updated = False

    body = {
        "verification_id": record.verification_id,
        "case_id": case_id,
        "outcome": outcome,
        "event_id": event_id,
        "message": "Transaction confirmed." if outcome == "confirmed" else "Dispute logged - recovery flow initiated.",
        "recovery_steps": [] if outcome == "confirmed" else RECOVERY_STEPS,
        "baseline_updated": baseline_updated,
    }
    return body


# --------------------------------------------------------------------------
# Post-verification: case history + feedback-pool status
# --------------------------------------------------------------------------

# Mirrors scripts/export_feedback.py's --disputed-threshold default: the
# section-6 signal that enough disputed outcomes justify a retraining round.
RETRAIN_DISPUTED_THRESHOLD = 5


def _case_dict(vo: "m.VerificationOutcome", score: RiskScore) -> dict:
    """A resolved case joined with its risk score (including reason codes)."""
    codes = json.loads(score.reason_codes) if score.reason_codes else []
    return {
        "case_id": vo.case_id,
        "outcome": vo.outcome,
        "event_id": vo.event_id,
        "risk_score": score.risk_score,
        "risk_band": score.risk_band,
        "reason_codes": codes,
        "reason_texts": [REASON_CODE_TEXT.get(c, c) for c in codes],
        "resolved_at": vo.resolved_at.isoformat() if vo.resolved_at else None,
    }


def _feedback_dict(db: Session) -> dict:
    """Feedback-pool counts feeding the retraining gate (section 6)."""
    resolved = db.query(func.count(m.VerificationOutcome.verification_id)).scalar() or 0
    disputed = (
        db.query(func.count(m.VerificationOutcome.verification_id))
        .filter(m.VerificationOutcome.outcome == "disputed")
        .scalar()
        or 0
    )
    return {
        "resolved": resolved,
        "confirmed": resolved - disputed,
        "disputed": disputed,
        "retrain_disputed_threshold": RETRAIN_DISPUTED_THRESHOLD,
    }


@app.get("/cases")
def cases(fraud_id: str = Depends(current_fraud_id), risk_db: Session = Depends(get_risk_db)):
    """Resolved verification cases for the caller, newest first.

    Each case joins the outcome (DB-3) to its risk score so the history
    shows what was decided and how it was scored - case id, outcome,
    event, band/score, reason codes, and when it was resolved.
    """
    rows = (
        risk_db.query(m.VerificationOutcome, RiskScore)
        .join(RiskScore, RiskScore.score_id == m.VerificationOutcome.score_id)
        .filter(m.VerificationOutcome.fraud_id == fraud_id)
        .order_by(desc(m.VerificationOutcome.resolved_at))
        .all()
    )
    return {"fraud_id": fraud_id, "cases": [_case_dict(vo, score) for vo, score in rows]}


@app.get("/feedback-status")
def feedback_status(risk_db: Session = Depends(get_risk_db)):
    """Feedback-pool counts feeding the retraining gate (section 6).

    Every confirmed/disputed outcome is labeled training data consumed by
    scripts/export_feedback.py; the disputed count is the trigger for a
    retraining round. Read-only over DB-3 - no PII beyond the counts.
    """
    return _feedback_dict(risk_db)


# --------------------------------------------------------------------------
# Investigator case workflow (§2.5)
# --------------------------------------------------------------------------
# Case state machine: NEW → REVIEWING → USER_VERIFICATION →
# CONFIRMED_SUSPICIOUS / CONFIRMED_LEGITIMATE → CLOSED
# Each transition is audited. Investigators see pseudonymous ID + score +
# confidence + reason codes + prior related alerts.

def _confidence_from_uncertainty(ml_score: float) -> str:
    """Map ML score to confidence level for case priority.

    High confidence = model is sure (score near 0 or 1).
    Low confidence = model is uncertain (score near 0.5).
    """
    if ml_score >= 0.85 or ml_score <= 0.15:
        return "high"
    if ml_score >= 0.65 or ml_score <= 0.35:
        return "medium"
    return "low"


def _compute_priority(risk_score: int, confidence: str, ml_score: float) -> float:
    """Priority = risk_score * confidence_weight.

    Low confidence cases get a priority boost so they surface for human
    review (the model is uncertain, so a human should decide).
    """
    conf_weight = {"high": 1.0, "medium": 1.2, "low": 1.5}
    return round(risk_score * conf_weight.get(confidence, 1.0), 2)


@app.get("/investigator/cases")
def investigator_cases(
    status: str | None = None,
    min_priority: float = 0.0,
    x_compliance_token: str = Header(alias="X-Compliance-Token"),
    db: Session = Depends(get_risk_db),
):
    """List investigator cases for compliance/analyst role.

    Filterable by status and minimum priority. Returns cases sorted by
    priority descending (highest priority first). Investigators see
    pseudonymous ID + score + confidence + reason codes. Cases live in the
    shared risk store (DB-3) next to the risk_scores they reference.
    """
    if x_compliance_token != settings.compliance_token:
        raise HTTPException(status_code=401, detail="invalid compliance token")

    q = db.query(m.InvestigatorCase)
    if status:
        q = q.filter(m.InvestigatorCase.status == status)
    if min_priority > 0:
        q = q.filter(m.InvestigatorCase.priority >= min_priority)
    rows = q.order_by(desc(m.InvestigatorCase.priority)).all()

    return {
        "cases": [
            {
                "case_id": c.case_id,
                "fraud_id": c.fraud_id,
                "event_id": c.event_id,
                "status": c.status,
                "confidence": c.confidence,
                "priority": c.priority,
                "risk_score": c.risk_score,
                "reason_codes": json.loads(c.reason_codes),
                "investigator_id": c.investigator_id,
                "notes": c.notes,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            }
            for c in rows
        ],
        "total": len(rows),
        "by_status": {
            s: sum(1 for c in rows if c.status == s)
            for s in m.CASE_STATES
        },
    }


@app.post("/investigator/cases")
def create_investigator_case(
    event_id: str,
    fraud_id: str,
    risk_score: int,
    ml_score: float,
    reason_codes: list[str],
    x_compliance_token: str = Header(alias="X-Compliance-Token"),
    db: Session = Depends(get_risk_db),
):
    """Create a new investigator case for a high-risk event.

    The case is created in the shared risk store against the event's actual
    RiskScore row: score_id is resolved from that row (it is NOT NULL in the
    schema but the pre-fix endpoint never supplied it -> 500 on every call,
    alert-quality audit #30), and risk/confidence/priority are derived from
    the STORED score, not from client-supplied numbers, so case ranking is
    consistent with the decision that produced the alert.

    Manual/analyst creation only (compliance role). Automatic case creation
    from score_generated events is intentionally NOT wired: alerts themselves
    are the queue and auto-casing every high-band event would flood the
    backlog (see audit #30 operational-burden findings).
    """
    if x_compliance_token != settings.compliance_token:
        raise HTTPException(status_code=401, detail="invalid compliance token")

    # Resolve the authoritative score row (must exist for traceability).
    score = db.query(RiskScore).filter(
        RiskScore.event_id == event_id,
        RiskScore.fraud_id == fraud_id,
    ).first()
    if score is None:
        raise HTTPException(
            status_code=404,
            detail="no risk score exists for this event_id/fraud_id - a case "
            "must reference a scored event",
        )
    risk_score = score.risk_score
    if score.ml_score is not None:
        ml_score = float(score.ml_score)

    confidence = _confidence_from_uncertainty(ml_score)
    priority = _compute_priority(risk_score, confidence, ml_score)

    # Dedup: one case per event (DB unique index enforces it race-free).
    existing = db.query(m.InvestigatorCase).filter(
        m.InvestigatorCase.event_id == event_id
    ).first()
    if existing:
        return {"case_id": existing.case_id, "status": existing.status,
                "already_exists": True, "score_id": existing.score_id}

    case = m.InvestigatorCase(
        fraud_id=fraud_id,
        event_id=event_id,
        score_id=score.score_id,
        status="NEW",
        confidence=confidence,
        priority=priority,
        risk_score=risk_score,
        reason_codes=json.dumps(reason_codes),
    )
    db.add(case)
    try:
        db.commit()
    except Exception:
        # Concurrent duplicate create (unique index) - return the winner.
        db.rollback()
        existing = db.query(m.InvestigatorCase).filter(
            m.InvestigatorCase.event_id == event_id
        ).first()
        if existing is not None:
            return {"case_id": existing.case_id, "status": existing.status,
                    "already_exists": True, "score_id": existing.score_id}
        raise

    append_audit_event(
        fraud_id,
        "case_created",
        {
            "case_id": case.case_id,
            "event_id": event_id,
            "risk_score": risk_score,
            "confidence": confidence,
            "priority": priority,
            "score_id": case.score_id,
        },
    )

    return {"case_id": case.case_id, "status": "NEW", "priority": priority,
            "score_id": case.score_id}


@app.post("/investigator/cases/{case_id}/transition")
def transition_case(
    case_id: str,
    new_status: str,
    notes: str = "",
    investigator_id: str = "",
    x_compliance_token: str = Header(alias="X-Compliance-Token"),
    db: Session = Depends(get_risk_db),
):
    """Transition a case to a new state. Validates the transition is legal.

    Each transition is audited to the hash chain. When an investigator
    CONFIRMS the case (CONFIRMED_SUSPICIOUS / CONFIRMED_LEGITIMATE) a
    VerificationOutcome is recorded exactly once, so investigator verdicts
    become labels in the retraining pool the same way user confirmations do
    (audit #30 finding: verdicts previously wrote no outcome).
    """
    if x_compliance_token != settings.compliance_token:
        raise HTTPException(status_code=401, detail="invalid compliance token")

    case = db.query(m.InvestigatorCase).filter(m.InvestigatorCase.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="case not found")

    allowed = m.CASE_TRANSITIONS.get(case.status, ())
    if new_status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"cannot transition from {case.status} to {new_status}; allowed: {allowed}",
        )

    # Capture BEFORE updating — audit must record the actual transition.
    old_status = case.status
    case.status = new_status
    case.investigator_id = investigator_id or case.investigator_id
    if notes:
        case.notes = (case.notes + "\n" + notes).strip()
    if new_status in ("CONFIRMED_SUSPICIOUS", "CONFIRMED_LEGITIMATE", "CLOSED"):
        case.closed_at = datetime.now(timezone.utc).replace(tzinfo=None)

    # Investigator verdict -> label (exactly once per score). Same store and
    # semantics as user confirmations: CONFIRMED_SUSPICIOUS = disputed,
    # CONFIRMED_LEGITIMATE = confirmed. Skipped if the event already has an
    # outcome (e.g. the user resolved it first).
    label_written = False
    if new_status in ("CONFIRMED_SUSPICIOUS", "CONFIRMED_LEGITIMATE"):
        resolved = db.query(m.VerificationOutcome).filter(
            m.VerificationOutcome.score_id == case.score_id
        ).first()
        if resolved is None:
            outcome = "disputed" if new_status == "CONFIRMED_SUSPICIOUS" else "confirmed"
            db.add(m.VerificationOutcome(
                score_id=case.score_id,
                fraud_id=case.fraud_id,
                event_id=case.event_id,
                outcome=outcome,
                case_id=case.case_id,
            ))
            label_written = True
    db.commit()

    append_audit_event(
        case.fraud_id,
        "case_transition",
        {
            "case_id": case_id,
            "from_status": old_status,
            "to_status": new_status,
            "investigator_id": investigator_id,
            "notes": notes[:200],
            "label_recorded": label_written,
        },
    )

    return {"case_id": case_id, "status": new_status, "priority": case.priority,
            "label_recorded": label_written}


# --------------------------------------------------------------------------
# /overview server-side cache
# --------------------------------------------------------------------------
# Short-lived in-process cache so repeat viewers don't re-run the DB
# queries + audit-proxy call on every read (the frontend re-fetches on
# login, refresh, and after each decision). Invalidated on every write
# (confirm / demo seed) because feedback counts and the chain are global -
# a write by any account changes them for everyone. The TTL is a backstop
# for anything that isn't a write path.
OVERVIEW_TTL_SECONDS = 10
_OVERVIEW_CACHE: dict[str, dict] = {}  # fraud_id -> {"expires": float, "payload": dict}


def _overview_cached(fraud_id: str) -> dict | None:
    entry = _OVERVIEW_CACHE.get(fraud_id)
    if entry is None or time.monotonic() >= entry["expires"]:
        return None
    return entry["payload"]


def _store_overview(fraud_id: str, payload: dict) -> None:
    if len(_OVERVIEW_CACHE) >= 100:  # bound growth: drop expired entries first
        for k in [k for k, v in _OVERVIEW_CACHE.items() if time.monotonic() >= v["expires"]]:
            del _OVERVIEW_CACHE[k]
    _OVERVIEW_CACHE[fraud_id] = {"expires": time.monotonic() + OVERVIEW_TTL_SECONDS, "payload": payload}


def _invalidate_overview() -> None:
    _OVERVIEW_CACHE.clear()


@app.get("/history")
def history(fraud_id: str = Depends(current_fraud_id), risk_db: Session = Depends(get_risk_db)):
    """Account history: every scored event for the caller, newest first,
    joined with its verification outcome, plus an abnormality analysis
    (repeated reason codes, dispute patterns, score escalation). All cases
    share the same fraud_id, so this is the trail a reviewer uses to spot
    patterns across them.
    """
    rows = (
        risk_db.query(RiskScore)
        .filter(RiskScore.fraud_id == fraud_id)
        # scored_at has second precision; the rowid tiebreak keeps same-second
        # events in true insertion order (score_id is a UUID - arbitrary).
        .order_by(desc(RiskScore.scored_at), desc(literal_column("rowid")))
        .all()
    )
    outcomes = {
        o.score_id: o
        for o in risk_db.query(m.VerificationOutcome).filter(m.VerificationOutcome.fraud_id == fraud_id).all()
    }
    reason_counts: dict[str, int] = {}
    events: list[dict] = []
    for s in rows:
        codes = json.loads(s.reason_codes) if s.reason_codes else []
        for c in codes:
            reason_counts[c] = reason_counts.get(c, 0) + 1
        o = outcomes.get(s.score_id)
        events.append(
            {
                "event_id": s.event_id,
                "risk_score": s.risk_score,
                "risk_band": s.risk_band,
                "reason_codes": codes,
                "reason_texts": [REASON_CODE_TEXT.get(c, c) for c in codes],
                "scored_at": s.scored_at.isoformat() if s.scored_at else None,
                "outcome": o.outcome if o else None,
            }
        )
    bands: dict[str, int] = {}
    for e in events:
        bands[e["risk_band"]] = bands.get(e["risk_band"], 0) + 1
    disputed = sum(1 for e in events if e["outcome"] == "disputed")
    confirmed = sum(1 for e in events if e["outcome"] == "confirmed")
    repeated = [
        {"code": c, "count": n, "text": REASON_CODE_TEXT.get(c, c)}
        for c, n in sorted(reason_counts.items(), key=lambda kv: -kv[1])
        if n >= 2
    ]

    flags: list[str] = []
    if disputed >= 2:
        flags.append(f"{disputed} disputed cases - repeated fraud pattern")
    if disputed and confirmed:
        flags.append("Mixed outcomes: some events confirmed, others disputed")
    if len(events) >= 3 and all(
        events[i]["risk_score"] >= events[i + 1]["risk_score"] for i in range(min(2, len(events) - 1))
    ):
        flags.append("Score escalation: recent scores are rising")
    for r in repeated:
        flags.append(f"{r['text']} raised in {r['count']} events")
    if not events:
        flags.append("No scored events yet")

    return {
        "fraud_id": fraud_id,
        "events": events,
        "stats": {
            "total": len(events),
            "bands": bands,
            "disputed": disputed,
            "confirmed": confirmed,
            "repeated_reasons": repeated,
            "flags": flags,
        },
    }


@app.get("/overview")
def overview(fraud_id: str = Depends(current_fraud_id), risk_db: Session = Depends(get_risk_db)):
    """One round trip for the whole alerts screen: open alerts, resolved
    cases, feedback-pool counts, and live chain status.

    The frontend renders everything from this single payload instead of
    chaining /alerts + /cases + /feedback-status + /chain-status fetches
    (BFF pattern); the sub-endpoints stay for direct callers/tests.
    Served from a short-lived in-process cache; invalidated on writes.
    """
    cached = _overview_cached(fraud_id)
    if cached is not None:
        return cached
    alert_rows = (
        risk_db.query(RiskScore)
        .filter(RiskScore.fraud_id == fraud_id, RiskScore.risk_band == "high")
        .order_by(desc(RiskScore.scored_at))
        .all()
    )
    resolved_outcomes = {
        vo.score_id
        for vo in risk_db.query(m.VerificationOutcome).filter(m.VerificationOutcome.fraud_id == fraud_id).all()
    }
    case_rows = (
        risk_db.query(m.VerificationOutcome, RiskScore)
        .join(RiskScore, RiskScore.score_id == m.VerificationOutcome.score_id)
        .filter(m.VerificationOutcome.fraud_id == fraud_id)
        .order_by(desc(m.VerificationOutcome.resolved_at))
        .all()
    )
    status, chain = _audit_get("/audit/integrity", actor="verification-ui-overview")
    payload = {
        "fraud_id": fraud_id,
        "alerts": [
            _alert_dict(s) for s in alert_rows if s.score_id not in resolved_outcomes
        ],
        "cases": [_case_dict(vo, score) for vo, score in case_rows],
        "feedback": _feedback_dict(risk_db),
        "chain": chain if status == 200 else {"available": False},
    }
    _store_overview(fraud_id, payload)
    return payload


# --------------------------------------------------------------------------
# DEV ONLY: end-to-end demo seeding (Privacy -> Risk -> alert)
# --------------------------------------------------------------------------

class DemoSeedResponse(BaseModel):
    seeded: int
    high_risk: int
    alerts: list[dict]
    baseline_committed: int = 0
    age_days: float = 0.0  # warm-up: account aged to this maturity before the attack
    device_id: str = ""  # the account's own device, used by the warm-up history


@app.post("/demo/seed", response_model=DemoSeedResponse)
def demo_seed(fraud_id: str = Depends(current_fraud_id)):
    """Creates demo activity for the caller by chaining the Privacy Layer and
    Risk Engine over HTTP. DEV ONLY - gated by PS14_MODE != production.

    WARM-UP FIRST: the first transaction bootstraps the profile, then the
    account is AGED in DB-2 (profile birth backdated to 60 days ago) BEFORE
    the remaining warm-ups are ingested/evaluated/committed - so every
    warm-up feature derives against a MATURE profile (tenure ~60d, last
    similar txn weeks ago, no 24h spike), exactly like a real account's
    history. Only then is the attack event evaluated. Evaluating warm-ups
    before aging would score the whole history as a brand-new account -
    cold-start false positives in the preview (fresh-account features
    dominate, not the attack).
    """
    if os.environ.get("PS14_MODE", "production").lower() == "production":
        raise HTTPException(status_code=403, detail="Demo seed endpoint disabled in production")

    now = datetime.now(timezone.utc)
    prefix = uuid.uuid4().hex[:10]  # unique per call - event_ids are global
    # Each demo account gets its OWN device, location, and recipient: the
    # fingerprint/location/recipient stores are GLOBAL (one row per token), so
    # a constant demo token shared across accounts would look like a mule ring
    # to the link-analysis features (device/recipient shared with other
    # accounts) and every warm-up event would score high.
    uid = uuid.uuid4().hex[:6]
    device_id = f"demo-device-{uid}"
    location_id = f"L-DEMO-{uid}"
    recipient_id = f"R-DEMO-{uid}"
    warm_days = [49, 42, 35, 28, 21, 14]
    warmup = [
        {
            "fraud_id": fraud_id,
            "event_id": f"{prefix}-hist-{i:04d}",
            "amount": 120.0,  # flat normal spend - keeps the median stable
            "ts": (now - timedelta(days=d)).isoformat(),
            "hour_of_day": 12,
            "device_id": device_id,
            "location_id": location_id,
            "recipient_id": recipient_id,
            "failed_auth_count_24h": 0,
        }
        for i, d in enumerate(warm_days)
    ]
    attack = {
        "fraud_id": fraud_id,
        "event_id": f"{prefix}-attack-0001",
        "amount": 4500.0,
        "ts": now.isoformat(),
        "hour_of_day": 3,
        "device_id": f"demo-attacker-{uuid.uuid4().hex[:6]}",
        "location_id": f"L-ATK-{uuid.uuid4().hex[:6]}",
        "recipient_id": f"R-ATK-{uuid.uuid4().hex[:6]}",
        "failed_auth_count_24h": 5,
    }

    results: list[dict] = []
    committed = 0
    headers = {"X-Internal-Token": settings.internal_token}
    age_days = 60.0
    with httpx.Client(timeout=10.0) as client:
        # 1) bootstrap: the first warm-up creates the DB-2 profile (the
        #    account's birth) but is NOT evaluated - scoring it against zero
        #    history would be a cold-start false positive.
        first = warmup[0]
        r = client.post(
            f"{settings.privacy_url}/internal/ingest-transaction", json=first, headers=headers
        )
        r.raise_for_status()

        # 2) age the account IMMEDIATELY: the profile is born 60 days ago, so
        #    every warm-up feature below derives against a mature account
        #    (tenure ~60d, last similar txn weeks ago, no 24h spike).
        ar = client.post(
            f"{settings.privacy_url}/internal/demo-age-account",
            json={"fraud_id": fraud_id, "age_days": age_days},
            headers=headers,
        )
        ar.raise_for_status()

        # 3) warm-up history: ingest + evaluate the remaining warm-ups,
        #    committing each ALLOWED event to the baseline (blocked/verify
        #    events never are). The first committed event registers the
        #    account's own device, so the later warm-ups are fully known.
        for ev in warmup[1:]:
            r = client.post(
                f"{settings.privacy_url}/internal/ingest-transaction", json=ev, headers=headers
            )
            r.raise_for_status()
            features = r.json()
            r2 = client.post(
                f"{settings.risk_url}/internal/evaluate",
                json={"event_id": ev["event_id"], "fraud_id": fraud_id, "features": features},
                headers=headers,
            )
            r2.raise_for_status()
            results.append(r2.json())
            if r2.json().get("decision") == "allow":
                try:
                    cr = client.post(
                        f"{settings.privacy_url}/internal/commit-baseline",
                        json={"event_id": ev["event_id"], "fraud_id": fraud_id},
                        headers=headers,
                    )
                    if cr.status_code == 200 and cr.json().get("committed"):
                        committed += 1
                except httpx.HTTPError:
                    pass

        # 4) the attack, evaluated against the mature history (ts = now, so it
        #    is the only event in the current window).
        r = client.post(
            f"{settings.privacy_url}/internal/ingest-transaction", json=attack, headers=headers
        )
        r.raise_for_status()
        r2 = client.post(
            f"{settings.risk_url}/internal/evaluate",
            json={"event_id": attack["event_id"], "fraud_id": fraud_id, "features": r.json()},
            headers=headers,
        )
        r2.raise_for_status()
        results.append(r2.json())

    # New scores landed (and the chain grew): cached /overview payloads are
    # stale for every account - feedback counts and the chain are global.
    _invalidate_overview()

    return {
        "seeded": len(warmup) + 1,
        "high_risk": sum(1 for x in results if x["risk_band"] == "high"),
        "alerts": [x for x in results if x["risk_band"] == "high"],
        "baseline_committed": committed,
        "age_days": age_days,
        "device_id": device_id,
    }


@app.get("/chain-status")
def chain_status():
    """Live chain-integrity status for the UI footer badge.

    A system-level health read (ok / entry count / genesis / first bad
    seq) proxied to the Audit Service with the internal token - the
    compliance passphrase stays private to the trail view. Every poll is
    logged by the Audit Service (`audit_access_log`, actor
    verification-ui-footer), so the badge itself is audited.
    """
    status, data = _audit_get("/audit/integrity", actor="verification-ui-footer")
    if status != 200:
        raise HTTPException(status_code=status, detail=data.get("detail", "audit service unavailable"))
    return data


STARTED_AT = datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health():
    return {"status": "ok", "service": "verification-service", "started_at": STARTED_AT}
