"""Audit Service (FastAPI, compliance-role only). Reads the append-only,
hash-chained audit trail and verifies its integrity (architecture section 4,
13, 14).

  GET /                   - static compliance viewer UI
  GET /audit/events       - pseudonymous event trail (RBAC: internal token)
  GET /audit/integrity    - recompute the whole chain; report first bad link
  GET /audit/export       - full-chain signed export for regulators (HMAC)
  GET /compliance/events  - same trail, gated by the compliance passphrase
  GET /compliance/integrity - same integrity check (browser-usable)

Every read is itself logged to `audit_access_log` (the audit of the audit).
The audit store holds only pseudonyms - never plaintext PII.
"""

from __future__ import annotations

import hmac
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from src.audit_service import models as m
from src.audit_service.db import SessionLocal, engine
from src.shared_db import make_get_db
from src.audit_service.export import sign_export
from src.audit_service.writer import GENESIS_HASH, verify_chain
from src.identity_service.security import verify_internal_token
from src.middleware import apply_security_middleware
from src.risk_engine.reason_codes import REASON_CODE_TEXT
from src.settings import settings, load_dotenv_and_patch
load_dotenv_and_patch()

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from src.settings import enforce_production_gate
    enforce_production_gate()
    if not settings.use_postgres:
        m.Base.metadata.create_all(bind=engine)
    yield


# Stable, readable operationIds in the OpenAPI schema (UI clients build
# from the schema - operationId == helper name).
def _clean_operation_id(route):
    return route.name


app = FastAPI(
    title="Audit Service",
    version="0.1.0",
    lifespan=lifespan,
    generate_unique_id_function=_clean_operation_id,
)
apply_security_middleware(app)


get_db = make_get_db(SessionLocal)


def require_compliance(x_internal_token: str | None = Header(default=None, alias="X-Internal-Token")) -> None:
    """Elevated role gate (service-to-service). Production: compliance role +
    mTLS, per section 3."""
    if not x_internal_token or not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="compliance role required")


def require_compliance_token(
    x_compliance_token: str | None = Header(default=None, alias="X-Compliance-Token"),
) -> str:
    """Browser-facing gate for the compliance viewer UI.

    Accepts either:
    1. Legacy shared compliance token (for backward compatibility)
    2. Per-analyst credentials: X-Compliance-Token header contains 'username:passphrase'

    Returns the analyst identifier for audit logging.
    """
    from src.audit_service.analyst_store import AnalystStore
    analyst_db = Path(__file__).resolve().parent.parent.parent.parent / "db" / "analysts.json"
    store = AnalystStore(analyst_db)

    if not x_compliance_token:
        raise HTTPException(status_code=401, detail="compliance role required")

    # Check for per-analyst format: 'username:passphrase'
    if ":" in x_compliance_token:
        username, passphrase = x_compliance_token.split(":", 1)
        if store.verify(username, passphrase):
            return username

    # Fallback: legacy shared token (backward compatibility)
    if hmac.compare_digest(x_compliance_token, settings.compliance_token):
        return "legacy-shared-token"

    raise HTTPException(status_code=401, detail="invalid compliance credentials")


def _log_access(db: Session, actor: str, action: str, query_summary: dict) -> None:
    db.add(m.AuditAccessLog(actor=actor, action=action, query_summary=json.dumps(query_summary)))
    db.commit()


def _serialize(row: m.AuditEvent) -> dict:
    try:
        payload = json.loads(row.payload_summary)
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = {"corrupted_payload": row.payload_summary}  # flagged by integrity check
    return {
        "seq": row.seq,
        "event_id": row.event_id,
        "fraud_id": row.fraud_id,
        "event_type": row.event_type,
        "prev_hash": row.prev_hash,
        "entry_hash": row.entry_hash,
        "payload": payload,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _score_stats(db: Session, event_type: str | None) -> tuple[dict, int]:
    """Chain-wide summary for the viewer's chips: {event_type: count} across
    the whole chain (respecting an active filter) plus how many score events
    carry no flagged reasons. Window-independent, so the numbers are stable
    no matter how many pages the user has loaded."""
    by_type_q = db.query(m.AuditEvent.event_type, func.count()).group_by(m.AuditEvent.event_type)
    if event_type:
        by_type_q = by_type_q.filter(m.AuditEvent.event_type == event_type)
    by_type = dict(by_type_q.all())
    no_flag = 0
    if not event_type or event_type == "score_generated":
        for (ps,) in db.query(m.AuditEvent.payload_summary).filter(
                m.AuditEvent.event_type == "score_generated").all():
            try:
                if not (json.loads(ps).get("reason_codes") or []):
                    no_flag += 1
            except (json.JSONDecodeError, TypeError, ValueError):
                pass  # corrupt payloads are flagged by integrity, not counted here
    return by_type, no_flag


def _list_events(db: Session, fraud_id: str | None, event_type: str | None, limit: int, offset: int = 0) -> list[dict]:
    q = db.query(m.AuditEvent)
    if fraud_id:
        q = q.filter(m.AuditEvent.fraud_id == fraud_id)
    if event_type:
        q = q.filter(m.AuditEvent.event_type == event_type)
    rows = q.order_by(desc(m.AuditEvent.seq)).offset(offset).limit(min(limit, 1000)).all()
    return [_serialize(r) for r in rows]


def _integrity(db: Session) -> dict:
    rows = db.query(m.AuditEvent).order_by(m.AuditEvent.seq.asc()).all()
    result = verify_chain(rows)
    return {
        "ok": result["ok"],
        "n_entries": len(rows),
        "first_bad_seq": result.get("first_bad_seq"),
        "genesis_hash": GENESIS_HASH,
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/static/app.js", include_in_schema=False)
def app_js():
    resp = FileResponse(STATIC_DIR / "app.js")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.get("/audit/events")
def audit_events(
    fraud_id: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
    x_audit_actor: str | None = Header(default=None, alias="X-Audit-Actor"),
    _compliance=Depends(require_compliance),
    db: Session = Depends(get_db),
):
    events = _list_events(db, fraud_id, event_type, limit, offset)
    total = db.query(func.count()).select_from(m.AuditEvent)
    if fraud_id:
        total = total.filter(m.AuditEvent.fraud_id == fraud_id)
    if event_type:
        total = total.filter(m.AuditEvent.event_type == event_type)
    _log_access(db, x_audit_actor or "compliance", "list_audit_events",
                {"fraud_id": fraud_id, "event_type": event_type, "limit": limit, "offset": offset})
    return {"count": len(events), "total": total.scalar() or 0, "events": events}


@app.get("/audit/integrity")
def audit_integrity(
    x_audit_actor: str | None = Header(default=None, alias="X-Audit-Actor"),
    _compliance=Depends(require_compliance),
    db: Session = Depends(get_db),
):
    result = _integrity(db)
    _log_access(db, x_audit_actor or "compliance", "verify_chain_integrity",
                {"n_entries": result["n_entries"], "ok": result["ok"]})
    return result


def _latest_case(db: Session) -> dict | None:
    """Newest verification_resolved, joined with its own score_generated
    (same event_id) so the compliance card tells the whole story - alert ->
    score -> outcome -> case -> chain entry. Derived from DB-4 directly, so
    it stays correct even when the newest case falls outside the paged
    window of events."""
    row = (
        db.query(m.AuditEvent)
        .filter(m.AuditEvent.event_type == "verification_resolved")
        .order_by(desc(m.AuditEvent.seq))
        .first()
    )
    if row is None:
        return None
    try:
        payload = json.loads(row.payload_summary)
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = {}
    event_id = payload.get("event_id")
    score = None
    if event_id:
        for s in (
            db.query(m.AuditEvent)
            .filter(
                m.AuditEvent.event_type == "score_generated",
                m.AuditEvent.fraud_id == row.fraud_id,
            )
            .order_by(desc(m.AuditEvent.seq))
            .all()
        ):
            try:
                sp = json.loads(s.payload_summary)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if sp.get("event_id") == event_id:
                score = {"risk_band": sp.get("risk_band"), "risk_score": sp.get("risk_score")}
                break
    return {
        "case_id": payload.get("case_id"),
        "outcome": payload.get("outcome"),
        "event_id": event_id,
        "resolved_at": row.created_at.isoformat() if row.created_at else None,
        "seq": row.seq,
        "entry_hash": row.entry_hash,
        "risk_band": score["risk_band"] if score else None,
        "risk_score": score["risk_score"] if score else None,
    }


@app.get("/audit/overview")
def audit_overview(
    limit: int = 100,
    offset: int = 0,
    x_audit_actor: str | None = Header(default=None, alias="X-Audit-Actor"),
    _compliance=Depends(require_compliance),
    db: Session = Depends(get_db),
):
    """BFF aggregate for the compliance viewer: chain integrity, the latest
    verification case (joined with its score), and a paged slice of events
    with the chain-wide aggregates - one round trip instead of three.

    Every read is logged (audit of the audit).
    """
    events = _list_events(db, None, None, limit, offset)
    # Section 11: category-level reason texts for the UI - never thresholds,
    # weights, or raw probabilities.
    for ev in events:
        payload = ev.get("payload", {})
        if ev.get("event_type") == "score_generated" and payload.get("reason_codes"):
            payload["reason_texts"] = [REASON_CODE_TEXT.get(c, c) for c in payload["reason_codes"]]
    integrity = _integrity(db)
    by_type, no_flag_scores = _score_stats(db, None)
    total = db.query(func.count()).select_from(m.AuditEvent).scalar() or 0
    _log_access(db, x_audit_actor or "compliance-ui", "compliance_overview",
                {"limit": limit, "offset": offset, "n_entries": integrity["n_entries"]})
    return {
        "integrity": integrity,
        "latest_case": _latest_case(db),
        "events": events,
        "count": len(events),
        "total": total,
        "by_type": by_type,
        "no_flag_scores": no_flag_scores,
    }


@app.get("/audit/export")
def audit_export(
    x_audit_actor: str | None = Header(default=None, alias="X-Audit-Actor"),
    _compliance=Depends(require_compliance),
    db: Session = Depends(get_db),
):
    """Full-chain compliance export for an external regulator (section 13).

    Every event (pseudonymous), the integrity report, and an HMAC-SHA256
    signature over the whole body (settings.export_signing_key). A regulator
    can verify authenticity with the signature and recompute the chain
    independently with scripts/verify_export.py - without trusting this
    service's own integrity field.

    The export read is itself logged to audit_access_log (audit of the
    audit). Production: asymmetric KMS signing, and hand the document out
    of band (or over a separate channel) rather than through this API.
    """
    rows = db.query(m.AuditEvent).order_by(m.AuditEvent.seq.asc()).all()
    events = [_serialize(r) for r in rows]
    integrity = _integrity(db)
    # The signature covers everything except itself: the algorithm label is
    # part of the signed body, so a verifier that only strips `signature`
    # recomputes over exactly what was signed.
    body = {
        "format": "ps14-audit-export-v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "genesis_hash": GENESIS_HASH,
        "integrity": integrity,
        "events": events,
        "signature_algorithm": "hmac-sha256",
    }
    doc = {**body, "signature": sign_export(body, settings.export_signing_key)}
    _log_access(db, x_audit_actor or "compliance", "export_audit_chain",
                {"n_entries": len(events), "ok": integrity["ok"]})
    return doc


# --------------------------------------------------------------------------
# Compliance viewer endpoints (browser-usable, passphrase-gated)
# --------------------------------------------------------------------------

@app.get("/compliance/events")
def compliance_events(
    limit: int = 100,
    offset: int = 0,
    event_type: str | None = None,
    analyst: str = Depends(require_compliance_token),
    db: Session = Depends(get_db),
):
    events = _list_events(db, None, event_type, limit, offset)
    # Section 11: category-level reason texts for the UI - never thresholds,
    # weights, or raw probabilities.
    for ev in events:
        payload = ev.get("payload", {})
        if ev.get("event_type") == "score_generated" and payload.get("reason_codes"):
            payload["reason_texts"] = [REASON_CODE_TEXT.get(c, c) for c in payload["reason_codes"]]
    total = db.query(func.count()).select_from(m.AuditEvent)
    if event_type:
        total = total.filter(m.AuditEvent.event_type == event_type)
    by_type, no_flag_scores = _score_stats(db, event_type)
    _log_access(db, f"analyst:{analyst}", "list_audit_events",
                {"limit": limit, "offset": offset, "event_type": event_type})
    return {"count": len(events), "total": total.scalar() or 0,
            "no_flag_scores": no_flag_scores, "by_type": by_type, "events": events}


@app.get("/compliance/integrity")
def compliance_integrity(
    analyst: str = Depends(require_compliance_token),
    db: Session = Depends(get_db),
):
    result = _integrity(db)
    _log_access(db, f"analyst:{analyst}", "verify_chain_integrity",
                {"n_entries": result["n_entries"], "ok": result["ok"]})
    return result


STARTED_AT = datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health():
    return {"status": "ok", "service": "audit-service", "started_at": STARTED_AT}
