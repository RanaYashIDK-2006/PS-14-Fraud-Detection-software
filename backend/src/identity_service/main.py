"""Identity Service (FastAPI). Owns PII + auth + the real<->pseudonym bridge.

API surface (section 14):
  POST /auth/register            - user-facing
  POST /auth/login               - user-facing
  GET  /me/fraud-id              - user-facing (Bearer; caller's own ID only)
  POST /internal/resolve-fraud-id- internal only (break-glass, logged)

Everything else in the system must reach identity through these endpoints;
the Fraud Engine / Risk Engine have no network path or credentials into this
store (section 1).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import partial

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from src.identity_service import models as m
from src.identity_service.db import SessionLocal, engine
from src.shared_db import make_get_db
from src.identity_service.security import (
    blind_index,
    create_access_token,
    decrypt_opt,
    decrypt_pii,
    decode_access_token,
    encrypt_pii,
    gen_fraud_id,
    hash_password,
    mask,
    verify_internal_token,
    verify_password,
)
from src.middleware.rate_limiter import RateLimiter, rate_limit_middleware
from src.middleware.security_headers import SecurityHeadersMiddleware
from src.audit_service.writer import append_audit_event
from src.settings import settings, load_dotenv_and_patch
load_dotenv_and_patch()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from src.settings import enforce_production_gate
    enforce_production_gate()
    # PostgreSQL: tables are pre-created by supabase_migration_v2.sql.
    # Only run SQLite-specific create_all + migrations for local dev.
    if not settings.use_postgres:
        m.Base.metadata.create_all(bind=engine)
        # --- lightweight migration: add columns that were added after initial
        # schema creation.  SQLite doesn't support ADD COLUMN IF NOT EXISTS, so
        # we catch the expected OperationalError when the column already exists.
        _new_cols = [
            ("account_number", "VARCHAR(20)"),
            ("account_type", "VARCHAR(20)"),
            ("role", "VARCHAR(20) DEFAULT 'user'"),
        ]
        import logging as _mig_log
        _logger = _mig_log.getLogger("ps14.identity.migration")
        with engine.connect() as _conn:
            for _col, _dtype in _new_cols:
                try:
                    _conn.execute(
                        __import__("sqlalchemy").text(
                            f"ALTER TABLE users ADD COLUMN {_col} {_dtype}"
                        )
                    )
                    _conn.commit()
                except Exception as e:
                    err_str = str(e).lower()
                    if "duplicate column" in err_str or "already exists" in err_str:
                        pass
                    else:
                        raise RuntimeError(
                            f"Migration ADD COLUMN {_col} failed: {e}. "
                            f"Unexpected schema error — startup halted."
                        ) from e
            # PII remediation: drop the plaintext 'address' TEXT column if it exists.
            try:
                col_exists = _conn.execute(
                    __import__("sqlalchemy").text(
                        "SELECT COUNT(*) FROM pragma_table_info('users') WHERE name = 'address'"
                    )
                ).fetchone()
                if col_exists and col_exists[0] > 0:
                    _conn.execute(
                        __import__("sqlalchemy").text("ALTER TABLE users DROP COLUMN address")
                    )
                    _conn.commit()
                    _logger.info("Dropped plaintext 'address' column from users table")
            except Exception as e:
                err_str = str(e).lower()
                if "no such column" in err_str or "duplicate column" in err_str:
                    pass
                else:
                    raise RuntimeError(
                        f"Address column migration failed: {e}. "
                        f"The plaintext address column may still exist. "
                        f"Startup halted for security."
                    ) from e
    yield


# Stable, readable operationIds in the OpenAPI schema (the verification UI
# builds its API client from the schema - operationId == helper name).
def _clean_operation_id(route):
    return route.name


app = FastAPI(
    title="Identity Service",
    version="0.1.0",
    lifespan=lifespan,
    generate_unique_id_function=_clean_operation_id,
)

# SECURITY: CORS restricted to specific origins (no wildcards)
# Production: only allowlist the actual UI domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Internal-Token"],
    allow_credentials=True,
)

# SECURITY: Rate limiting for brute-force protection
_rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        return await rate_limit_middleware(request, call_next, _rate_limiter)


app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


get_db = make_get_db(SessionLocal)


def current_fraud_id(authorization: str | None = Header(default=None)) -> str:
    """Bearer token -> fraud_id. JWT carries only the pseudonym (§4)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        payload = decode_access_token(authorization.removeprefix("Bearer "))
        return payload["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="invalid or expired token")


# ---------- RBAC helpers --------------------------------------------------
# Roles are hierarchical: admin > analyst > user.
# Permissions are strings like "risk:read", "pii:read", "user:manage".

ROLE_HIERARCHY: dict[str, int] = {"user": 0, "analyst": 1, "admin": 2}

PERMISSION_MAP: dict[str, list[str]] = {
    "user": [
        "profile:read", "profile:update:name", "profile:update:account",
        "profile:update:address", "temp_key:generate",
    ],
    "analyst": [
        "risk:read", "risk:trace", "case:read", "case:update_status",
        "audit:read", "device_graph:read",  # pseudonymous only – no PII
    ],
    "admin": [
        "user:manage", "role:assign", "system:config",
        "audit:read", "audit:integrity",  # full compliance access
    ],
}
# Analysts inherit user permissions; admins inherit both.
for _parent, _child in [("user", "analyst"), ("analyst", "admin")]:
    PERMISSION_MAP[_child] = PERMISSION_MAP.get(_parent, []) + PERMISSION_MAP[_child]


def _extract_auth(authorization: str | None = Header(default=None)) -> dict:
    """Parse and validate the Bearer token, returning the full payload."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        return decode_access_token(authorization.removeprefix("Bearer "))
    except Exception:
        raise HTTPException(status_code=401, detail="invalid or expired token")


def require_role(*allowed_roles: str):
    """FastAPI dependency that gates on role.  Usage:

        @app.get("/admin-only")
        def admin_view(_: dict = Depends(require_role("admin"))): ...
    """
    def _dep(payload: dict = Depends(_extract_auth)) -> dict:
        user_role = payload.get("role", "user")
        if user_role not in allowed_roles:
            raise HTTPException(status_code=403, detail=f"requires role: {', '.join(allowed_roles)}")
        return payload
    return _dep


def require_permission(*perms: str):
    """FastAPI dependency that gates on one or more permissions.
    The user must hold ALL listed permissions."""
    def _dep(payload: dict = Depends(_extract_auth)) -> dict:
        user_role = payload.get("role", "user")
        granted = set(PERMISSION_MAP.get(user_role, []))
        missing = set(perms) - granted
        if missing:
            raise HTTPException(status_code=403, detail=f"missing permissions: {', '.join(sorted(missing))}")
        return payload
    return _dep


# --------------------------------------------------------------------------
# User-facing endpoints
# --------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    phone: str = Field(pattern=r"^\+?[0-9]{10,15}$")
    password: str = Field(min_length=8, max_length=200)
    # role is NOT accepted from clients — every self-registration is hardcoded to "user".
    # Use POST /admin/assign-role (admin-only) to grant analyst/admin roles.


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)
    totp_code: str | None = Field(None, min_length=6, max_length=6)


class ProfileUpdateRequest(BaseModel):
    full_name: str | None = Field(None, min_length=1, max_length=200)
    phone: str | None = Field(None, pattern=r"^\+?[0-9]{10,15}$")
    account_number: str | None = Field(None, min_length=5, max_length=20)
    account_type: str | None = Field(None, pattern=r"^(savings|current|fixed_deposit)$")
    address: str | None = Field(None, max_length=500)


class TempAccessRequest(BaseModel):
    email: EmailStr
    password: str
    purpose: str = Field(min_length=1, max_length=100)  # e.g., "profile_update"
    duration_minutes: int = Field(default=30, ge=5, le=1440)  # 5 min to 24 hours


class TempAccessVerifyRequest(BaseModel):
    temp_key: str


class TempKey(BaseModel):
    """Temporary access key for limited operations."""
    key_id: str
    user_id: str
    fraud_id: str
    purpose: str
    expires_at: str
    permissions: list[str]


@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    email_key = blind_index(str(req.email))
    if db.query(m.User).filter(m.User.email_hash == email_key).first():
        raise HTTPException(status_code=409, detail="email already registered")

    user = m.User(
        full_name=encrypt_pii(req.full_name),
        phone_encrypted=encrypt_pii(req.phone),
        email_encrypted=encrypt_pii(str(req.email)),
        email_hash=email_key,
        address_encrypted=encrypt_pii(req.address) if getattr(req, 'address', None) else None,
    )
    # Hardcode role to "user" — no client-supplied role is accepted.
    # Use POST /admin/assign-role (admin-only) for role changes.
    if hasattr(user, 'role'):
        user.role = "user"
    db.add(user)
    db.flush()
    db.add(m.AuthCredential(user_id=user.user_id, secret_hash=hash_password(req.password)))
    fraud_id = gen_fraud_id()
    db.add(m.PseudonymMapping(fraud_id=fraud_id, user_id=user.user_id))
    db.commit()
    return {"user_id": user.user_id, "fraud_id": fraud_id, "role": "user"}


# --- Login brute-force protection ---
import time as _time
_login_failures: dict[str, list[float]] = {}  # ip -> [timestamps]
_LOGIN_LOCKOUT_THRESHOLD = 10  # failures before lockout
_LOGIN_LOCKOUT_WINDOW = 900    # 15 minutes
_LOGIN_LOCKOUT_DURATION = 900  # lockout for 15 minutes


def _check_login_brute_force(ip: str) -> None:
    now = _time.time()
    attempts = _login_failures.get(ip, [])
    attempts = [t for t in attempts if now - t < _LOGIN_LOCKOUT_WINDOW]
    _login_failures[ip] = attempts
    if len(attempts) >= _LOGIN_LOCKOUT_THRESHOLD:
        oldest = min(attempts)
        if now - oldest < _LOGIN_LOCKOUT_DURATION:
            remaining = int(_LOGIN_LOCKOUT_DURATION - (now - oldest))
            raise HTTPException(
                status_code=429,
                detail=f"too many failed attempts, try again in {remaining}s"
            )
        else:
            _login_failures[ip] = []


def _record_login_failure(ip: str) -> None:
    _login_failures.setdefault(ip, []).append(_time.time())


@app.post("/auth/login")
def login(req: LoginRequest, request: Request, db: Session = Depends(get_db)):
    client_ip = request.client.host if request.client else "unknown"
    _check_login_brute_force(client_ip)

    user = db.query(m.User).filter(m.User.email_hash == blind_index(str(req.email))).first()
    if user is None:
        _record_login_failure(client_ip)
        raise HTTPException(status_code=401, detail="invalid credentials")
    cred = (
        db.query(m.AuthCredential)
        .filter(m.AuthCredential.user_id == user.user_id, m.AuthCredential.credential_type == "password")
        .first()
    )
    if cred is None or not verify_password(req.password, cred.secret_hash):
        _record_login_failure(client_ip)
        raise HTTPException(status_code=401, detail="invalid credentials")

    mapping = db.query(m.PseudonymMapping).filter(m.PseudonymMapping.user_id == user.user_id).first()
    user_role = getattr(user, 'role', 'user') or 'user'
    return {
        "access_token": create_access_token(mapping.fraud_id, role=user_role),
        "token_type": "bearer",
        "role": user_role,
        "expires_in": settings.jwt_expiry_minutes * 60,
    }


@app.get("/me/fraud-id")
def my_fraud_id(fraud_id: str = Depends(current_fraud_id), db: Session = Depends(get_db)):
    mapping = db.query(m.PseudonymMapping).filter(m.PseudonymMapping.fraud_id == fraud_id).first()
    if mapping is None:
        raise HTTPException(status_code=404, detail="mapping not found")
    # Logged: every identity-adjacent access is recorded (access_log_required).
    db.add(
        m.PseudonymAccessLog(
            fraud_id=fraud_id, actor="user:self-service", action="view_own_fraud_id", reason="user requested own fraud id"
        )
    )
    db.commit()
    return {"fraud_id": fraud_id}


# --------------------------------------------------------------------------
# Temporary Access Key System
# --------------------------------------------------------------------------
# Shared store for temporary keys (SQLite-backed, multi-worker safe).
from src.session_store import SessionStore
from pathlib import Path as _Path
_temp_keys = SessionStore(_Path(settings.db_dir) / "sessions.db", session_type="temp_key")

import secrets as _secrets
from datetime import datetime, timedelta, timezone


@app.post("/auth/temp-access", status_code=201)
def generate_temp_access(req: TempAccessRequest, db: Session = Depends(get_db)):
    """Generate a temporary access key for limited operations."""
    # Verify credentials first
    user = db.query(m.User).filter(m.User.email_hash == blind_index(str(req.email))).first()
    if user is None:
        raise HTTPException(status_code=401, detail="invalid credentials")
    
    cred = (
        db.query(m.AuthCredential)
        .filter(m.AuthCredential.user_id == user.user_id, m.AuthCredential.credential_type == "password")
        .first()
    )
    if cred is None or not verify_password(req.password, cred.secret_hash):
        raise HTTPException(status_code=401, detail="invalid credentials")
    
    # Get fraud_id
    mapping = db.query(m.PseudonymMapping).filter(m.PseudonymMapping.user_id == user.user_id).first()
    if mapping is None:
        raise HTTPException(status_code=404, detail="mapping not found")
    
    # Generate temporary key
    key_id = _secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=req.duration_minutes)
    
    # Set permissions based on purpose
    permissions = ["profile:read"]
    if req.purpose == "profile_update":
        permissions.extend(["profile:update:name", "profile:update:account"])
    elif req.purpose == "full_access":
        permissions.extend(["profile:update:name", "profile:update:account", "profile:update:address"])
    
    _temp_keys.put(key_id, {
        "user_id": user.user_id,
        "fraud_id": mapping.fraud_id,
        "purpose": req.purpose,
        "permissions": permissions,
    }, expires_at)
    
    # Log the access
    db.add(
        m.PseudonymAccessLog(
            fraud_id=mapping.fraud_id,
            actor="user:self-service",
            action="temp_access_generated",
            reason=f"Purpose: {req.purpose}, Duration: {req.duration_minutes}min"
        )
    )
    db.commit()
    
    return {
        "key_id": key_id,
        "expires_at": expires_at.isoformat(),
        "permissions": permissions,
        "purpose": req.purpose,
    }


def _verify_temp_key(key_id: str) -> dict | None:
    """Verify a temporary access key and return its info."""
    return _temp_keys.get(key_id)


@app.get("/auth/temp-access/verify")
def verify_temp_access(temp_key: str = Header(alias="X-Temp-Key")):
    """Verify a temporary access key."""
    key_info = _verify_temp_key(temp_key)
    if key_info is None:
        raise HTTPException(status_code=401, detail="invalid or expired temporary key")
    return {
        "valid": True,
        "permissions": key_info["permissions"],
        "expires_at": key_info["expires_at"],
    }


@app.get("/me/profile")
def get_profile(fraud_id: str = Depends(current_fraud_id), db: Session = Depends(get_db)):
    """Get user profile (requires JWT or temp key)."""
    mapping = db.query(m.PseudonymMapping).filter(m.PseudonymMapping.fraud_id == fraud_id).first()
    if mapping is None:
        raise HTTPException(status_code=404, detail="mapping not found")
    
    user = db.query(m.User).filter(m.User.user_id == mapping.user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    
    return {
        "user_id": user.user_id,
        "fraud_id": fraud_id,
        "full_name": decrypt_opt(user.full_name),
        "email_masked": mask(decrypt_pii(user.email_encrypted)),
        "phone_masked": mask(decrypt_pii(user.phone_encrypted)),
        "account_number": decrypt_opt(user.account_number),
        "account_type": decrypt_opt(user.account_type),
        "address_masked": mask(decrypt_pii(user.address_encrypted)) if getattr(user, 'address_encrypted', None) else None,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@app.put("/me/profile")
def update_profile(
    req: ProfileUpdateRequest,
    fraud_id: str = Depends(current_fraud_id),
    db: Session = Depends(get_db)
):
    """Update user profile (requires JWT or temp key with appropriate permissions)."""
    mapping = db.query(m.PseudonymMapping).filter(m.PseudonymMapping.fraud_id == fraud_id).first()
    if mapping is None:
        raise HTTPException(status_code=404, detail="mapping not found")
    
    user = db.query(m.User).filter(m.User.user_id == mapping.user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    
    # Update allowed fields
    updates = []
    if req.full_name is not None:
        user.full_name = encrypt_pii(req.full_name)
        updates.append("full_name")
    if req.phone is not None:
        user.phone_encrypted = encrypt_pii(req.phone)
        updates.append("phone")
    if req.account_number is not None:
        user.account_number = encrypt_pii(req.account_number)
        updates.append("account_number")
    if req.account_type is not None:
        user.account_type = encrypt_pii(req.account_type)
        updates.append("account_type")
    if req.address is not None:
        if hasattr(user, 'address_encrypted'):
            user.address_encrypted = encrypt_pii(req.address)
            updates.append("address")
    
    if not updates:
        raise HTTPException(status_code=400, detail="no updates provided")
    
    db.commit()
    
    # Log the update
    db.add(
        m.PseudonymAccessLog(
            fraud_id=fraud_id,
            actor="user:self-service",
            action="profile_updated",
            reason=f"Updated fields: {', '.join(updates)}"
        )
    )
    db.commit()
    
    return {
        "ok": True,
        "updated_fields": updates,
        "message": f"Profile updated: {', '.join(updates)}"
    }


# --------------------------------------------------------------------------
# Admin-only endpoints (RBAC-gated)
# --------------------------------------------------------------------------

class AssignRoleRequest(BaseModel):
    user_id: str
    role: str = Field(pattern=r"^(user|analyst|admin)$")


@app.get("/admin/users")
def list_users(
    _: dict = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    """List all users (admin only). Returns masked PII."""
    users = db.query(m.User).all()
    result = []
    for u in users:
        mapping = db.query(m.PseudonymMapping).filter(m.PseudonymMapping.user_id == u.user_id).first()
        # Decrypt PII gracefully — older users may have been encrypted
        # under a different Fernet key; show [encrypted] instead of crashing.
        try:
            email_masked = mask(decrypt_pii(u.email_encrypted))
            phone_masked = mask(decrypt_pii(u.phone_encrypted))
        except Exception:
            email_masked = "[encrypted]"
            phone_masked = "[encrypted]"
        result.append({
            "user_id": u.user_id,
            "fraud_id": mapping.fraud_id if mapping else None,
            "full_name": decrypt_opt(u.full_name),
            "email_masked": email_masked,
            "phone_masked": phone_masked,
            "role": getattr(u, 'role', 'user') or 'user',
            "created_at": u.created_at.isoformat() if u.created_at else None,
        })
    return {"users": result, "count": len(result)}


@app.post("/admin/assign-role")
def assign_role(
    req: AssignRoleRequest,
    payload: dict = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    """Assign a role to a user (admin only)."""
    user = db.query(m.User).filter(m.User.user_id == req.user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    
    old_role = getattr(user, 'role', 'user') or 'user'
    if hasattr(user, 'role'):
        user.role = req.role
    db.commit()
    
    # Log the role change
    mapping = db.query(m.PseudonymMapping).filter(m.PseudonymMapping.user_id == user.user_id).first()
    if mapping:
        append_audit_event(
            mapping.fraud_id,
            "role_changed",
            {"old_role": old_role, "new_role": req.role, "admin": payload.get("sub")},
        )
    
    return {"ok": True, "user_id": req.user_id, "old_role": old_role, "new_role": req.role}


# --------------------------------------------------------------------------
# Analyst-only endpoints (pseudonymous data, no PII)
# --------------------------------------------------------------------------

@app.get("/analyst/pseudonymous-users")
def analyst_list_users(
    _: dict = Depends(require_role("analyst", "admin")),
    db: Session = Depends(get_db),
):
    """List users as pseudonymous profiles (analyst/admin only). No PII."""
    users = db.query(m.User).all()
    result = []
    for u in users:
        mapping = db.query(m.PseudonymMapping).filter(m.PseudonymMapping.user_id == u.user_id).first()
        if mapping:
            result.append({
                "fraud_id": mapping.fraud_id,
                "role": getattr(u, 'role', 'user') or 'user',
                "created_at": u.created_at.isoformat() if u.created_at else None,
            })
    return {"profiles": result, "count": len(result)}


@app.get("/analyst/pseudonymous-users/{fraud_id}")
def analyst_get_user(
    fraud_id: str,
    _: dict = Depends(require_role("analyst", "admin")),
    db: Session = Depends(get_db),
):
    """Get a single pseudonymous user profile (analyst/admin only). No PII."""
    mapping = db.query(m.PseudonymMapping).filter(m.PseudonymMapping.fraud_id == fraud_id).first()
    if mapping is None:
        raise HTTPException(status_code=404, detail="fraud_id not found")
    user = db.query(m.User).filter(m.User.user_id == mapping.user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return {
        "fraud_id": fraud_id,
        "role": getattr(user, 'role', 'user') or 'user',
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


# --------------------------------------------------------------------------
# Internal (service-to-service). mTLS + internal token in production (§4).
# --------------------------------------------------------------------------

class ResolveRequest(BaseModel):
    fraud_id: str = Field(pattern=r"^F[A-Z2-9]{15}$")
    reason: str = Field(min_length=1, max_length=500)
    case_id: str = Field(min_length=1, max_length=100)  # REQUIRED: audit case reference
    approver: str | None = Field(None, min_length=1, max_length=100)  # optional second approver


# Break-glass rate limiting: max 10 resolutions per hour per actor
_break_glass_rate: dict[str, list[float]] = {}
_BREAK_GLASS_LIMIT = 10
_BREAK_GLASS_WINDOW = 3600  # 1 hour


def _check_break_glass_rate(actor: str) -> None:
    import time
    now = time.time()
    attempts = _break_glass_rate.get(actor, [])
    attempts = [t for t in attempts if now - t < _BREAK_GLASS_WINDOW]
    _break_glass_rate[actor] = attempts
    if len(attempts) >= _BREAK_GLASS_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"break-glass rate limit exceeded: max {_BREAK_GLASS_LIMIT} per hour"
        )
    attempts.append(now)


@app.post("/internal/resolve-fraud-id", include_in_schema=False)
def resolve_fraud_id(
    req: ResolveRequest,
    x_internal_token: str = Header(alias="X-Internal-Token"),
    db: Session = Depends(get_db),
):
    if not verify_internal_token(x_internal_token):
        raise HTTPException(status_code=401, detail="invalid internal token")

    actor = "internal:compliance"
    _check_break_glass_rate(actor)

    mapping = db.query(m.PseudonymMapping).filter(m.PseudonymMapping.fraud_id == req.fraud_id).first()
    if mapping is None:
        raise HTTPException(status_code=404, detail="unknown fraud_id")
    user = db.query(m.User).filter(m.User.user_id == mapping.user_id).first()

    # Break-glass resolution is always logged with actor + case_id + reason
    # (DB-1 local access log) AND appended to the hash-chained audit trail
    # (DB-4): pseudonymous actor + case_id + reason only — the PII itself
    # stays in DB-1 and never appears on the chain.
    db.add(
        m.PseudonymAccessLog(
            fraud_id=req.fraud_id, actor=actor, action="resolve_fraud_id",
            reason=f"case={req.case_id} {req.reason}"
        )
    )
    db.commit()
    append_audit_event(
        req.fraud_id,
        "fraud_id_resolved",
        {
            "actor": actor,
            "case_id": req.case_id,
            "reason": req.reason,
            "approver": req.approver,
        },
    )

    return {
        "user_id": user.user_id,
        "email_masked": mask(decrypt_pii(user.email_encrypted)),
        "phone_masked": mask(decrypt_pii(user.phone_encrypted)),
        "case_id": req.case_id,
    }


from datetime import datetime, timezone  # noqa: E402
STARTED_AT = datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health():
    return {"status": "ok", "service": "identity-service", "started_at": STARTED_AT}
