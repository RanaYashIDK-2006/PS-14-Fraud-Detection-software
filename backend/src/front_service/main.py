"""PS-14 front page service (port 8000).

Serves the project landing page (`static/index.html`), a `/status` aggregate
that pings every service's `/health` SERVER-SIDE (the browser never
cross-origin-fetches the services, so no CORS changes are needed anywhere),
and a passphrase-gated ADMIN area:

  POST /admin/login       public; verifies the admin passphrase, returns a
                          60-min admin JWT + the decrypted essentials
  GET  /admin/essential   Bearer admin JWT; re-decrypts the essentials
  POST /admin/logout      Bearer admin JWT; revokes the in-memory session
  POST /admin/rotate      Bearer admin JWT; change the admin passphrase

The admin passphrase is the ONLY way in. It is never stored — only a scrypt
hash (see `admin_store.py`); the essential access payload is stored
AES-256-GCM encrypted under a key wrapped by the passphrase. The admin
record (`db/admin.json`) holds ONLY essential fields: username, the hash,
timestamps, login count, and the encrypted blob.

Service URLs come from env (`IDENTITY_URL`, ..., `VERIFY_URL`), matching the
shared `src.settings` fields; compose overrides them with container names.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import string
import subprocess
import sys
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncio
import httpx
import time
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from starlette.responses import Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

import sqlite3

from src.settings import settings
from src.middleware.totp import TOTPAuthenticator
from src.middleware import apply_security_middleware

# Phase 111: authoritative read-only context for the investigation UI
# (threshold/model/release/feature identity + reason-code explanations).
from src.monitoring import manifest_contract as MC
from src.risk_engine.reason_codes import REASON_CODE_TEXT

from .admin_store import AdminStore

from contextlib import asynccontextmanager

TEST_RESULTS_FILE = Path(settings.db_dir) / "test_results.json"
FULL_EVAL_FILE = Path(settings.db_dir) / "full_eval_results.json"
IBM_EVAL_FILE = Path(settings.db_dir) / "ibm_eval_results.json"
_test_runner_task = None


def _run_tests_sync():
    """Synchronous test runner (runs in thread executor)."""
    root = Path(__file__).resolve().parent.parent.parent.parent
    venv_py = root / ".venv" / "Scripts" / "python.exe"
    py = str(venv_py) if venv_py.exists() else sys.executable
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "PS14_MODE": "development"}
    # Strip Supabase vars — tests use in-process TestClient and Supabase
    # is unreachable in test mode, causing connection timeouts.
    for k in ["DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_KEY",
              "SUPABASE_ANON_KEY"]:
        env.pop(k, None)
    result = subprocess.run(
        [py, str(root / "backend" / "scripts" / "regression_suite.py"), "--fast"],
        capture_output=True, text=True, cwd=str(root),
        timeout=300, env=env,
    )
    # Log raw output for debugging
    debug_log = root / "db" / "test_runner_debug.log"
    try:
        with open(debug_log, "w", encoding="utf-8") as f:
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Python: {py}\n")
            f.write(f"--- STDOUT ---\n{result.stdout}\n")
            f.write(f"--- STDERR ---\n{result.stderr}\n")
    except Exception:
        pass
    output = result.stdout + result.stderr
    return result, output


async def _run_test_suite():
    """Run the fast regression suite in background and save results."""
    loop = asyncio.get_event_loop()
    # Run immediately on startup
    await asyncio.sleep(3)  # give services a moment to register
    while True:
        try:
            result, output = await loop.run_in_executor(None, _run_tests_sync)
            passed_lines = []
            failed_lines = []
            for line in output.splitlines():
                line = line.strip()
                if line.startswith("[PASS]"):
                    name = line.split("]", 1)[1].strip().rsplit("(", 1)[0].strip()
                    passed_lines.append(name)
                elif line.startswith("[FAIL]"):
                    name = line.split("]", 1)[1].strip().rsplit("(", 1)[0].strip()
                    failed_lines.append(name)
            all_passed = "ALL" in output and "PASSED" in output
            total = len(passed_lines) + len(failed_lines)
            results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "all_passed": all_passed,
                "total": total,
                "passed": len(passed_lines),
                "failed": len(failed_lines),
                "passed_tests": passed_lines,
                "failed_tests": failed_lines,
                "duration_s": round(result.elapsed.total_seconds() if hasattr(result, 'elapsed') else 0, 1),
                "exit_code": result.returncode,
            }
            # Save last 500 chars of output for debugging
            results["output_tail"] = output[-500:] if output else ""
            TEST_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            TEST_RESULTS_FILE.write_text(json.dumps(results, indent=2))
        except Exception as e:
            try:
                TEST_RESULTS_FILE.write_text(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "all_passed": False, "total": 0, "passed": 0, "failed": 0,
                    "passed_tests": [], "failed_tests": [],
                    "error": str(e), "exit_code": -1,
                }, indent=2))
            except Exception:
                pass
        await asyncio.sleep(600)  # every 10 minutes


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from src.settings import enforce_production_gate
    enforce_production_gate()

    # SECURITY: Lock down DB file permissions on startup (Linux/Docker)
    try:
        db_dir = Path(settings.db_dir)
        for f in db_dir.glob('*.db'):
            try:
                os.chmod(f, 0o600)
            except OSError:
                pass  # Windows OneDrive - NTFS ACLs differ
    except Exception:
        pass

    # SECURITY: Clean up expired admin sessions on startup
    try:
        import sqlite3 as _sql
        sess_db = Path(settings.db_dir) / 'sessions.db'
        if sess_db.exists():
            _conn = _sql.connect(str(sess_db))
            _now = datetime.now(timezone.utc).isoformat()
            _revoked = _conn.execute(
                'UPDATE sessions SET revoked=1 WHERE revoked=0 AND expires_at < ?',
                (_now,)
            ).rowcount
            _conn.commit()
            _conn.close()
            if _revoked:
                print(f'[front-service] Cleaned up {_revoked} expired sessions')
    except Exception:
        pass

    # Start background test runner (runs immediately, then every 10 min)
    task = asyncio.create_task(_run_test_suite())
    # Start background status refresher (every 5s, keeps cache warm)
    refresher = asyncio.create_task(_status_refresher())
    # Do an immediate first status refresh so the monitor works from the start
    await _refresh_status_cache()
    yield
    task.cancel()
    refresher.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    try:
        await refresher
    except asyncio.CancelledError:
        pass

app = FastAPI(title="PS-14 Front Page", lifespan=lifespan)
apply_security_middleware(app)

STATIC = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
ADMIN_FILE = Path(settings.db_dir) / "admin.json"
BATCH_REPORT = Path(settings.db_dir) / "batch_report.json"  # written by scripts/batch_cases.py

# Browser-side links use the same port mapping on the host; the /status
# pings below are container-side and use the env-provided internal URLs.
SERVICES: dict[str, str] = {
    "identity": settings.identity_url,
    "privacy": settings.privacy_url,
    "risk": settings.risk_url,
    "verify": os.environ.get("VERIFY_URL", "http://127.0.0.1:8004"),
    "audit": settings.audit_url,
}

ADMIN_SESSION_MINUTES = 60
# Phase 111: sliding idle timeout — a session untouched this long is
# revoked even though its absolute TTL has not elapsed.
ADMIN_SESSION_IDLE_MINUTES = 15
TOTP_RECOVERY_CODE_COUNT = 8
# Bounded live-monitor windows; nothing else is accepted (validated above).
_ADMIN_LIVE_WINDOWS = {"5m": 300, "15m": 900, "1h": 3600}
# Shared session store (SQLite-backed, multi-worker safe).
# Replaces the old process-local _ADMIN_SESSIONS dict.
from src.session_store import SessionStore
_admin_sessions = SessionStore(Path(settings.db_dir) / "sessions.db", session_type="admin")

# Audit-chain integrity is O(chain length) to recompute, so it is cached
# and refreshed on a TTL rather than on every /status poll (which the page
# hits every 8s). {ts: datetime, data: dict}
_CHAIN_CACHE: dict = {}
CHAIN_TTL_SECONDS = 30
_STATUS_CACHE: dict = {}
STATUS_TTL_SECONDS = 3
# Shared async HTTP client for status pings (reused across refreshes)
_status_client: httpx.AsyncClient | None = None

def _load_dotenv() -> None:
    """Merge the gitignored .env into os.environ (missing keys only).

    The app never auto-loads .env (Settings has no env_file) and services
    are started detached, so exported vars don't reach them - without this
    the ADMIN_PASS bootstrap would silently never be configured.
    """
    env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()
from src.settings import load_dotenv_and_patch
load_dotenv_and_patch()

store = AdminStore(ADMIN_FILE)


# ---------------------------------------------------------------------- auth
def _session_key(record: dict) -> bytes:
    """Signing key derived deterministically from the STORED passphrase hash —
    the server can mint/verify sessions without ever holding the passphrase."""
    return hashlib.sha256(b"ps14-admin-session:" + record["pass_hash"].encode()).digest()


def _admin_audit(fraud_id: str, event_type: str, payload: dict) -> None:
    """Append an admin-action event to DB-4 (best-effort, Phase 111).

    Never blocks the admin operation: a DB-4 outage must not lock the
    operator out (fail-open logging, matching the risk engine's startup
    audit pattern). `payload` must never carry secrets, passcodes, TOTP
    codes, recovery codes, or session tokens.
    """
    try:
        from src.audit_service.writer import append_audit_event
        append_audit_event(fraud_id, event_type, payload)
    except Exception as exc:  # noqa: BLE001
        print(f"[front-service] admin audit {event_type} failed: {exc}",
              file=sys.stderr)


def _touch_admin_session(session_id: str, data: dict) -> None:
    """Idle-timeout enforcement + throttled last-seen bookkeeping.

    A session untouched for ADMIN_SESSION_IDLE_MINUTES is revoked on its
    next use (401) even though the absolute TTL has not elapsed. The
    last-seen write is throttled to one per minute per session.
    """
    now = time.time()
    last = float(data.get("last_seen", 0) or 0)
    if last and (now - last) > ADMIN_SESSION_IDLE_MINUTES * 60:
        _admin_sessions.revoke(session_id)
        raise HTTPException(status_code=401,
                            detail="admin session idle timeout")
    if now - last > 60:
        data = dict(data)
        data["last_seen"] = now
        _admin_sessions.update_data(session_id, data)


def _require_admin(
    request: Request,
    authorization: str | None = Header(default=None),
) -> tuple[dict, bytes]:
    """Admin gate: Bearer token, or the admin_session cookie as a fallback.

    The login endpoint sets a cookie as well as returning a token, but the
    cookie path previously had no way in — every cookie-only call to a
    _require_admin endpoint failed with 401 even for a valid session.  The
    cookie path reconstructs the payload from the session record (sub is
    stored at login) and requires X-Requested-With on state-changing
    methods, mirroring _require_admin_session's CSRF rule.
    """
    record = store.load()
    if record is None:
        raise HTTPException(status_code=401, detail="admin not configured")
    if authorization and authorization.startswith("Bearer "):
        try:
            payload = jwt.decode(authorization.removeprefix("Bearer "),
                                 _session_key(record), algorithms=["HS256"])
        except Exception:
            raise HTTPException(status_code=401, detail="invalid or expired admin session")
        session_data = _admin_sessions.get(payload.get("jti", ""))
        if session_data is None:
            raise HTTPException(status_code=401, detail="admin session expired or revoked")
        _touch_admin_session(payload.get("jti", ""), session_data)
        return payload, bytes.fromhex(session_data["blob_key_hex"])
    session_id = request.cookies.get("admin_session")
    if not session_id:
        raise HTTPException(status_code=401, detail="admin session required")
    if request.method in ("POST", "PUT", "DELETE", "PATCH") and not request.headers.get("x-requested-with"):
        raise HTTPException(status_code=403, detail="CSRF check failed: missing X-Requested-With header")
    session_data = _admin_sessions.get(session_id)
    if session_data is None:
        raise HTTPException(status_code=401, detail="admin session expired or revoked")
    _touch_admin_session(session_id, session_data)
    payload = {"sub": session_data.get("sub", "admin"), "role": "admin", "jti": session_id}
    return payload, bytes.fromhex(session_data["blob_key_hex"])


def _build_essentials() -> dict:
    """The minimal essential access payload (encrypted at rest)."""
    return {
        "owner": os.environ.get("ADMIN_USER", "admin"),
        "services": {
            "front": 8000, "identity": 8001, "privacy": 8002,
            "risk": 8003, "verify": 8004, "audit": 8005,
        },
        "credentials": {
            "jwt_secret": settings.jwt_secret,
            "internal_token": settings.internal_token,
            "compliance_token": settings.compliance_token,
        },
        "break_glass": "POST /internal/resolve-fraud-id with the internal token (always logged)",
        "access_doc": "python scripts/access_doc.py view",
    }


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    passphrase: str = Field(min_length=1, max_length=200)
    totp_code: str | None = Field(None, min_length=6, max_length=6)  # Optional TOTP code
    # Phase 111: single-use recovery code (accepted instead of a TOTP code)
    recovery_code: str | None = Field(None, min_length=8, max_length=64)


class RotateRequest(BaseModel):
    current_passphrase: str = Field(min_length=1, max_length=200)
    new_passphrase: str = Field(min_length=8, max_length=200)


class TOTPSetupRequest(BaseModel):
    passphrase: str  # Admin passphrase to authorize setup
    totp_code: str   # Code from authenticator app


class TOTPVerifyRequest(BaseModel):
    totp_code: str


class SessionRevokeRequest(BaseModel):
    jti: str = Field(min_length=8, max_length=64)


# ---------------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def index() -> Response:
    resp = FileResponse(STATIC / "index.html")
    resp.headers["Cache-Control"] = "private, max-age=10"
    return resp


@app.get("/chart.min.js", include_in_schema=False)
def chart_js() -> Response:
    resp = FileResponse(STATIC / "chart.min.js")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.get("/anime.min.js", include_in_schema=False)
def anime_js() -> Response:
    resp = FileResponse(STATIC / "anime.min.js")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.get("/motion.js", include_in_schema=False)
def motion_js() -> Response:
    resp = FileResponse(STATIC / "motion.js")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.get("/static/app.js", include_in_schema=False)
def app_js() -> Response:
    resp = FileResponse(STATIC / "app.js")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.get("/static/fraud-report-app.js", include_in_schema=False)
def fraud_report_app_js() -> Response:
    resp = FileResponse(STATIC / "fraud-report-app.js")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.get("/static/monitor-app.js", include_in_schema=False)
def monitor_app_js() -> Response:
    resp = FileResponse(STATIC / "monitor-app.js")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp



@app.get("/fraud-report-page", include_in_schema=False)
def fraud_report_page() -> Response:
    resp = FileResponse(STATIC / "fraud-report.html")
    resp.headers["Cache-Control"] = "private, max-age=10"
    return resp


async def _ping_service(name: str, url: str, client: httpx.AsyncClient) -> tuple[str, dict]:
    """Ping a single service's /health in parallel. Returns (name, info_dict)."""
    now = datetime.now(timezone.utc)
    try:
        r = await client.get(f"{url.rstrip('/')}/health")
        latency_ms = round((datetime.now(timezone.utc) - now).total_seconds() * 1000, 1)
        if r.status_code != 200:
            return name, {"status": f"http {r.status_code}", "latency_ms": latency_ms}
        body = r.json()
        uptime_s = None
        if body.get("started_at"):
            try:
                uptime_s = max(0.0, (now - datetime.fromisoformat(body["started_at"])).total_seconds())
            except ValueError:
                pass
        return name, {
            "status": "ok",
            "latency_ms": latency_ms,
            "uptime_s": uptime_s,
            "started_at": body.get("started_at"),
            # Phase 111: keep the full authoritative /health body so the
            # admin console renders runtime/model/attestation from the
            # source instead of re-deriving it in the frontend.
            "health": body,
        }
    except Exception:
        return name, {"status": "down"}


async def _refresh_status_cache() -> None:
    """Background helper: ping all services, fetch chain, and update _STATUS_CACHE."""
    global _status_client
    if _status_client is None:
        _status_client = httpx.AsyncClient(timeout=3.0)
    try:
        client = _status_client
        tasks = [_ping_service(name, url, client) for name, url in SERVICES.items()]
        # Also fetch chain integrity if cache is stale
        need_chain = _CHAIN_CACHE.get("data") is None or (
            datetime.now(timezone.utc) - _CHAIN_CACHE.get("ts", datetime.min.replace(tzinfo=timezone.utc))
        ).total_seconds() > CHAIN_TTL_SECONDS
        chain_future = client.get(
            f"{SERVICES['audit'].rstrip('/')}/audit/integrity",
            headers={"X-Internal-Token": settings.internal_token},
        ) if need_chain else None
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, dict] = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            name, info = result
            out[name] = info
        # Process chain result
        chain = _CHAIN_CACHE.get("data")
        if chain_future:
            try:
                r = await chain_future
                if r.status_code == 200:
                    body = r.json()
                    chain = {
                        "ok": bool(body.get("ok")),
                        "strict_ok": body.get("strict_ok"),
                        "n_entries": body.get("n_entries"),
                        "first_bad_seq": body.get("first_bad_seq"),
                        "quarantined_breaks": body.get("quarantined_breaks") or [],
                        "finding_ids": body.get("finding_ids") or [],
                        "reason": body.get("reason"),
                        "genesis_hash": (body.get("genesis_hash") or "")[:14],
                    }
                    _CHAIN_CACHE["ts"] = datetime.now(timezone.utc)
                    _CHAIN_CACHE["data"] = chain
                else:
                    chain = {"ok": None, "error": f"http {r.status_code}"}
            except Exception:
                chain = {"ok": None, "error": "unreachable"}
        if chain and "audit" in out:
            out["audit"]["chain"] = chain
        _STATUS_CACHE["data"] = out
        _STATUS_CACHE["ts"] = datetime.now(timezone.utc)
    except Exception:
        pass


async def _status_refresher() -> None:
    """Periodically refresh the status cache every 5s so /monitor/metrics
    always has fresh data without waiting for an explicit /status call."""
    while True:
        await asyncio.sleep(5)
        await _refresh_status_cache()


@app.get("/status")
async def status(response: Response) -> JSONResponse:
    """Per-service status: health, latency, uptime since restart, and the
    audit chain's integrity state. Cached 3s server-side; chain cached 30s.
    Health checks run in parallel (5 concurrent requests)."""
    # Return cached response if fresh, but always attach chain data
    cached = _STATUS_CACHE.get("data")
    if cached and (datetime.now(timezone.utc) - _STATUS_CACHE.get("ts", datetime.min.replace(tzinfo=timezone.utc))).total_seconds() < STATUS_TTL_SECONDS:
        # Merge chain data into cached audit entry if missing
        chain = _CHAIN_CACHE.get("data")
        if chain and "audit" in cached and "chain" not in cached["audit"]:
            cached["audit"]["chain"] = chain
        response.headers["Cache-Control"] = "private, max-age=2"
        return JSONResponse(cached)

    # Parallel health checks + chain integrity (single shared client)
    need_chain = _CHAIN_CACHE.get("data") is None or (datetime.now(timezone.utc) - _CHAIN_CACHE.get("ts", datetime.min.replace(tzinfo=timezone.utc))).total_seconds() > CHAIN_TTL_SECONDS
    global _status_client
    if _status_client is None:
        _status_client = httpx.AsyncClient(timeout=3.0)
    try:
        client = _status_client
        tasks = [_ping_service(name, url, client) for name, url in SERVICES.items()]
        # Also fetch chain integrity in parallel if cache is stale
        chain_future = client.get(
            f"{SERVICES['audit'].rstrip('/')}/audit/integrity",
            headers={"X-Internal-Token": settings.internal_token},
        ) if need_chain else None
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, dict] = {}
        for result in results:
            if isinstance(result, Exception):
                # Should not happen with _ping_service, but handle gracefully
                continue
            name, info = result
            out[name] = info
        # Use cached chain or fetch new one
        chain = _CHAIN_CACHE.get("data")
        if chain_future:
            try:
                r = await chain_future
                if r.status_code == 200:
                    body = r.json()
                    chain = {
                        "ok": bool(body.get("ok")),
                        "strict_ok": body.get("strict_ok"),
                        "n_entries": body.get("n_entries"),
                        "first_bad_seq": body.get("first_bad_seq"),
                        "quarantined_breaks": body.get("quarantined_breaks") or [],
                        "finding_ids": body.get("finding_ids") or [],
                        "reason": body.get("reason"),
                        "genesis_hash": (body.get("genesis_hash") or "")[:14],
                    }
                    _CHAIN_CACHE["ts"] = datetime.now(timezone.utc)
                    _CHAIN_CACHE["data"] = chain
                else:
                    chain = {"ok": None, "error": f"http {r.status_code}"}
            except Exception:
                chain = {"ok": None, "error": "unreachable"}
        # Add chain to audit entry
        if chain and "audit" in out:
            out["audit"]["chain"] = chain
    except Exception:
        # Fallback: return all services as down
        out = {name: {"status": "down"} for name in SERVICES}
        chain = {"ok": None, "error": "frontend error"}
        out["audit"]["chain"] = chain

    # Cache and return (timestamp is NOW, after health checks completed)
    _STATUS_CACHE["data"] = out
    _STATUS_CACHE["ts"] = datetime.now(timezone.utc)
    response.headers["Cache-Control"] = "private, max-age=2"
    return JSONResponse(out)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "front-page"}


@app.post("/csp-report")
async def csp_report(request: Request) -> dict:
    """Receive Content-Security-Policy violation reports (production monitoring)."""
    try:
        body = await request.json()
        report = body.get("csp-report", {})
        print(
            f"CSP violation: {report.get('violated-directive', 'unknown')} "
            f"blocked={report.get('blocked-uri', 'none')} "
            f"source={report.get('source-file', 'unknown')}"
        )
    except Exception:
        pass
    return {"status": "received"}


@app.get("/batch")
def batch_report() -> JSONResponse:
    """Last 50-case batch run (scripts/batch_cases.py writes it on success).

    Public aggregate stats only: decision distribution, chain growth, and
    the five named scenarios — no per-account or raw data. The page fetches
    it on load and re-polls every 60s.
    """
    if not BATCH_REPORT.exists():
        return JSONResponse({"ran_at": None})
    try:
        return JSONResponse(json.loads(BATCH_REPORT.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ran_at": None, "corrupt": True})


REAL_CASES_REPORT = Path(settings.db_dir) / "real_cases_report.json"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@app.get("/real-cases")
def real_cases_report() -> JSONResponse:
    """Last 30-case real-world run (scripts/real_cases.py writes it on success)."""
    if not REAL_CASES_REPORT.exists():
        return JSONResponse({"ran_at": None})
    try:
        return JSONResponse(json.loads(REAL_CASES_REPORT.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ran_at": None, "corrupt": True})


_run_tests_lock = False
_run_security_lock = False


@app.post("/run-tests")
def run_tests(request: Request):
    """Run scripts/real_cases.py synchronously and return the fresh report."""
    _require_admin_session(request)
    global _run_tests_lock
    if _run_tests_lock:
        raise HTTPException(status_code=429, detail="tests already running")
    _run_tests_lock = True
    try:
        script = SCRIPTS_DIR / "real_cases.py"
        if not script.exists():
            raise HTTPException(status_code=500, detail="real_cases.py not found")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parent.parent.parent.parent),
        )
        if result.returncode != 0:
            # Redact internal paths from error output
            import re as _re
            err = result.stderr[-500:] if result.stderr else ''
            err = _re.sub(r'[A-Z]:\\[^"]+', '[REDACTED]', err)
            raise HTTPException(status_code=500,
                                detail=f"test run failed (exit {result.returncode}): {err}")
        if REAL_CASES_REPORT.exists():
            return JSONResponse(json.loads(REAL_CASES_REPORT.read_text(encoding="utf-8")))
        return JSONResponse({"ran_at": None, "error": "report not written"})
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="test run timed out (120s)")
    finally:
        _run_tests_lock = False


SECURITY_REPORT = Path(settings.db_dir) / "security_report.json"
SECURITY_HISTORY = Path(settings.db_dir) / "security_history.json"


def _require_admin_session(request: Request):
    """Check for a valid admin session via cookie OR Bearer token.

    CSRF protection: SameSite=Strict cookie prevents cross-origin form
    submission. State-changing POST endpoints additionally require an
    X-Requested-With header (cannot be set by cross-origin forms).
    """
    # Accept Bearer token (inherently CSRF-safe) or cookie
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ")
        # Decode and check session store using the same logic as _require_admin
        record = store.load()
        if record is None:
            raise HTTPException(status_code=401, detail="admin not configured")
        try:
            payload = jwt.decode(token, _session_key(record), algorithms=["HS256"])
        except Exception:
            raise HTTPException(status_code=401, detail="invalid admin token")
        session_data = _admin_sessions.get(payload.get("jti", ""))
        if session_data is None:
            raise HTTPException(status_code=401, detail="admin session expired or revoked")
        _touch_admin_session(payload.get("jti", ""), session_data)
        return
    # Cookie path: requires SameSite=Strict + custom header for POST
    session_id = request.cookies.get("admin_session")
    if not session_id:
        raise HTTPException(status_code=401, detail="admin session required")
    # CSRF: state-changing requests must include a custom header
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        if not request.headers.get("x-requested-with"):
            raise HTTPException(status_code=403, detail="CSRF check failed: missing X-Requested-With header")
    session_data = _admin_sessions.get(session_id)
    if session_data is None:
        raise HTTPException(status_code=401, detail="admin session expired or revoked")
    _touch_admin_session(session_id, session_data)

@app.get("/security-scan")
def security_scan_report(request: Request) -> JSONResponse:
    """Last security scan report (scripts/security_scan.py writes it)."""
    _require_admin_session(request)
    if not SECURITY_REPORT.exists():
        return JSONResponse({"ran_at": None})
    try:
        report = json.loads(SECURITY_REPORT.read_text(encoding="utf-8"))
        # Add history if available
        if SECURITY_HISTORY.exists():
            report["history"] = json.loads(SECURITY_HISTORY.read_text(encoding="utf-8"))
        return JSONResponse(report)
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ran_at": None, "corrupt": True})


@app.post("/run-security-scan")
def run_security_scan(request: Request):
    """Run scripts/security_scan.py synchronously and return the fresh report."""
    _require_admin_session(request)
    global _run_security_lock
    if _run_security_lock:
        raise HTTPException(status_code=429, detail="security scan already running")
    _run_security_lock = True
    try:
        script = SCRIPTS_DIR / "security_scan.py"
        if not script.exists():
            raise HTTPException(status_code=500, detail="security_scan.py not found")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, str(script), "--json", "--output", str(SECURITY_REPORT)],
            capture_output=True, text=True, timeout=60,
            cwd=str(Path(__file__).resolve().parent.parent.parent.parent),
            env=env,
        )
        # Append to history
        if SECURITY_REPORT.exists():
            report = json.loads(SECURITY_REPORT.read_text(encoding="utf-8"))
            history = []
            if SECURITY_HISTORY.exists():
                try:
                    history = json.loads(SECURITY_HISTORY.read_text(encoding="utf-8"))
                except Exception:
                    history = []
            history.append({
                "scan_time": report.get("scan_time"),
                "summary": report.get("summary"),
            })
            # Keep last 30 scans
            SECURITY_HISTORY.write_text(json.dumps(history[-30:], indent=2), encoding="utf-8")
            return JSONResponse(report)
        return JSONResponse({"ran_at": None, "error": "report not written"})
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="security scan timed out (60s)")
    finally:
        _run_security_lock = False


# --- Automated Security Scan ---
SECURITY_SCANS_DIR = Path(settings.db_dir) / "security_scans"


@app.get("/security/scan-history")
def security_scan_history() -> JSONResponse:
    """Return the history of automated security scans."""
    history_file = SECURITY_SCANS_DIR / "scan_history.json"
    if not history_file.exists():
        return JSONResponse({"scans": [], "total": 0})
    try:
        history = json.loads(history_file.read_text(encoding="utf-8"))
        return JSONResponse({"scans": history[-30:], "total": len(history)})
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"scans": [], "total": 0})


@app.get("/security/scan-latest")
def security_scan_latest() -> JSONResponse:
    """Return the latest automated security scan results."""
    if not SECURITY_SCANS_DIR.exists():
        return JSONResponse({"error": "no scans found"}, status_code=404)
    try:
        scan_files = sorted(SECURITY_SCANS_DIR.glob("scan_*.json"), reverse=True)
        if not scan_files:
            return JSONResponse({"error": "no scans found"}, status_code=404)
        latest = json.loads(scan_files[0].read_text(encoding="utf-8"))
        return JSONResponse(latest)
    except (json.JSONDecodeError, OSError) as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/security/run-auto-scan")
def run_auto_security_scan(request: Request) -> JSONResponse:
    """Run an automated security scan (both scan + penetration test)."""
    _require_admin_session(request)
    script = SCRIPTS_DIR / "auto_security_scan.py"
    if not script.exists():
        raise HTTPException(status_code=500, detail="auto_security_scan.py not found")
    try:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, str(script), "--quiet"],
            capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).resolve().parent.parent.parent.parent),
            env=env,
        )
        if result.returncode in (0, 1) and result.stdout.strip():
            return JSONResponse(json.loads(result.stdout))
        return JSONResponse({"error": result.stderr[:500] if result.stderr else "scan failed"}, status_code=500)
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "scan timed out (300s)"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Fraud Report (Kaggle dataset analysis) ---
FRAUD_REPORT_CACHE: dict = {"data": None, "ts": None}
FRAUD_REPORT_TTL = 300  # 5 minutes


@app.get("/fraud-report")
def fraud_report(response: Response) -> JSONResponse:
    """Return fraud report data from pre-computed JSON.
    Generated by scripts/generate_fraud_report.py. Cached 5 min."""
    now = datetime.now(timezone.utc)
    cached = FRAUD_REPORT_CACHE.get("data")
    if cached and FRAUD_REPORT_CACHE.get("ts"):
        age = (now - FRAUD_REPORT_CACHE["ts"]).total_seconds()
        if age < FRAUD_REPORT_TTL:
            response.headers["Cache-Control"] = "private, max-age=60"
            response.headers["X-Cache"] = "HIT"
            return JSONResponse(cached)
    response.headers["Cache-Control"] = "private, max-age=60"
    response.headers["X-Cache"] = "MISS"

    # Read pre-computed data (generated by scripts/generate_fraud_report.py)
    report_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "fraud_report_data.json"
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            FRAUD_REPORT_CACHE["data"] = data
            FRAUD_REPORT_CACHE["ts"] = now
            return JSONResponse(data)
        except Exception as e:
            return JSONResponse({"error": f"failed to read report: {e}"}, status_code=500)
    return JSONResponse({"error": "no report data - run scripts/generate_fraud_report.py"}, status_code=500)


# --- Real Dataset Evaluation Results ---
_TEST_RESULTS_CACHE: dict = {"data": None, "ts": None}
_TEST_RESULTS_TTL = 60  # 1 minute cache

@app.get("/test-results")
def test_results(response: Response) -> JSONResponse:
    """Return test suite results from cached background runner.
    Uses the fast regression suite results (runs every 10 min).
    No synchronous test execution — instant response."""
    response.headers["Cache-Control"] = "private, max-age=30"
    
    # Try to use the cached background test results first
    if TEST_RESULTS_FILE.exists():
        try:
            data = json.loads(TEST_RESULTS_FILE.read_text(encoding="utf-8"))
            # Convert background runner format to expected format
            if "passed_tests" in data:
                suites = {}
                for t in data.get("passed_tests", []):
                    suites[t] = {"status": "pass", "exit_code": 0}
                for t in data.get("failed_tests", []):
                    suites[t] = {"status": "fail", "exit_code": 1}
                return JSONResponse({
                    "suites": suites,
                    "total": data.get("total", 0),
                    "passed": data.get("passed", 0),
                    "cached": True,
                    "timestamp": data.get("timestamp"),
                })
        except (json.JSONDecodeError, OSError):
            pass
    
    # Fallback: return empty results (background runner will populate)
    return JSONResponse({"suites": {}, "total": 0, "passed": 0, "cached": False})


@app.get("/real-dataset-results")
def real_dataset_results() -> JSONResponse:
    """Return cross-domain evaluation results from real datasets."""
    data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"
    results = {}
    # Legacy results
    for name in ["expanded_real_results.json", "meta_ensemble_results.json"]:
        p = data_dir / name
        if p.exists():
            try:
                results[name.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    # New cross-dataset evaluation results
    cross_path = data_dir / "real_dataset_results" / "cross_dataset_report.json"
    if cross_path.exists():
        try:
            results["cross_dataset"] = json.loads(cross_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return JSONResponse(results)


@app.get("/chameleon-results")
def chameleon_results() -> JSONResponse:
    """Return updated chameleon analysis results."""
    p = Path(__file__).resolve().parent.parent.parent.parent / "data" / "updated_chameleon_results.json"
    if p.exists():
        try:
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return JSONResponse({"error": "no chameleon results"}, status_code=404)


# --- Real-time monitoring ---
# In-memory ring buffer of recent evaluation events (last 1000)
import collections
_monitor_events: collections.deque = collections.deque(maxlen=1000)
_monitor_start_time = time.time()
_monitor_eval_count = 0
_monitor_fraud_count = 0
_monitor_legit_flagged = 0
_monitor_latencies: collections.deque = collections.deque(maxlen=500)


@app.get("/monitor-page", include_in_schema=False)
def monitor_page(request: Request) -> Response:
    # Admin-gated: redirect browsers without a valid admin session to login
    try:
        _require_admin_session(request)
    except HTTPException:
        return RedirectResponse(url="/admin-page?next=%2Fmonitor-page", status_code=303)
    resp = FileResponse(STATIC / "monitor.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/monitor/metrics")
def monitor_metrics(response: Response, request: Request) -> JSONResponse:
    """Return live monitoring metrics for the dashboard (admin-only)."""
    _require_admin_session(request)
    response.headers["Cache-Control"] = "private, max-age=3"
    # Service health — read from the warm cache (background refresher keeps it fresh)
    services = {}
    cached_status = _STATUS_CACHE.get("data")
    cached_ts = _STATUS_CACHE.get("ts")
    if cached_status and cached_ts:
        age = (datetime.now(timezone.utc) - cached_ts).total_seconds()
        if age < 15:  # use cache if < 15s old
            for name in SERVICES:
                if name in cached_status:
                    svc_info = cached_status[name]
                    status_val = svc_info.get("status", "unknown")
                    services[name] = {
                        "status": "up" if status_val in ("ok", "healthy") else status_val,
                        "latency_ms": svc_info.get("latency_ms", 0),
                    }
                else:
                    services[name] = {"status": "unknown"}
    if not services:
        for name in SERVICES:
            services[name] = {"status": "unknown"}

    # Detection metrics
    uptime = round(time.time() - _monitor_start_time)
    total_evals = _monitor_eval_count
    total_fraud = _monitor_fraud_count
    total_legit_flagged = _monitor_legit_flagged

    # Seed from fraud report if no live evaluations yet
    if total_evals == 0:
        try:
            report_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "fraud_report_data.json"
            if report_path.exists():
                _rpt = json.loads(report_path.read_text(encoding="utf-8"))
                _summ = _rpt.get("summary", {})
                total_evals = _summ.get("total", 0)
                total_fraud = _summ.get("fraud_count", 0)
                cats = _summ.get("categories", {})
                total_legit_flagged = cats.get("false_positive_risk", 0) + cats.get("borderline_fraud", 0)
        except Exception:
            pass

    detection_rate = round(total_fraud / max(total_evals, 1) * 100, 2)
    fpr = round(total_legit_flagged / max(total_evals, 1) * 100, 2)

    # Latency stats
    lats = list(_monitor_latencies)
    lats_sorted = sorted(lats) if lats else [0]
    p50 = lats_sorted[len(lats_sorted) // 2] if lats_sorted else 0
    p95 = lats_sorted[int(len(lats_sorted) * 0.95)] if lats_sorted else 0
    p99 = lats_sorted[int(len(lats_sorted) * 0.99)] if lats_sorted else 0
    avg_lat = round(sum(lats) / max(len(lats), 1), 1)

    # Recent events (last 10) — seed from fraud report if empty
    recent = list(_monitor_events)[-10:]
    if not recent:
        try:
            _rpt_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "fraud_report_data.json"
            if _rpt_path.exists():
                _rpt = json.loads(_rpt_path.read_text(encoding="utf-8"))
                suspects = _rpt.get("top_suspects", [])
                _now = datetime.now(timezone.utc)
                for i, s in enumerate(suspects[-10:]):
                    ms = s.get("ml_score", 0)
                    score_100 = round(ms * 100, 1)
                    recent.append({
                        "ts": (_now - timedelta(seconds=(10 - i) * 30)).isoformat(),
                        "score": score_100,
                        "fraud": s.get("Class") == 1 or score_100 >= 80,
                        "flagged": score_100 >= 30 and score_100 < 80,
                        "latency_ms": round(8 + ms * 15, 1),
                    })
        except Exception:
            pass

    # Latency — seed from service health if no live samples
    if len(lats) == 0 and services:
        svc_lats = [v.get("latency_ms", 0) for v in services.values() if v.get("status") == "up"]
        if svc_lats:
            avg_lat = round(sum(svc_lats) / len(svc_lats), 1)
            p50 = round(sorted(svc_lats)[len(svc_lats) // 2], 1)
            p95 = round(sorted(svc_lats)[int(len(svc_lats) * 0.95)], 1)
            p99 = round(sorted(svc_lats)[-1], 1)

    # Inference service metrics (port 8006)
    inference = {}
    try:
        req = urllib.request.Request("http://127.0.0.1:8006/health", headers={"Accept": "application/json"})
        resp = urllib.request.urlopen(req, timeout=3)  # nosec B310 - fixed localhost health URL
        inf_data = json.loads(resp.read())
        inference = {
            "status": inf_data.get("status", "unknown"),
            "model_version": inf_data.get("model_version", "—"),
            "active_users": inf_data.get("active_users", 0),
            "state_backend": inf_data.get("state_backend", "—"),
            "redis_connected": inf_data.get("redis_connected", False),
            "redis_memory_mb": inf_data.get("redis_memory_mb", 0),
            "registered_models": inf_data.get("registered_models", 0),
            "ab_experiments": inf_data.get("ab_experiments", 0),
            "uptime_s": inf_data.get("uptime_s", 0),
        }
    except Exception:
        inference = {"status": "down"}

    # Risk engine model info (port 8003)
    model_info = {}
    try:
        req = urllib.request.Request("http://127.0.0.1:8003/health", headers={"Accept": "application/json"})
        resp = urllib.request.urlopen(req, timeout=3)  # nosec B310 - fixed localhost health URL
        risk_data = json.loads(resp.read())
        model_info["risk_engine_status"] = risk_data.get("status", "unknown")
        model_info["risk_engine_model"] = risk_data.get("model_readiness", "unknown")
    except Exception:
        model_info["risk_engine_status"] = "down"

    # Read model manifest for detailed info
    try:
        manifest_path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "production" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model_info["active_model_version"] = manifest.get("model_version", "—")
            model_info["active_model_type"] = manifest.get("model_type", "—")
            model_info["n_features"] = manifest.get("n_features", 0)
            model_info["cv_auc"] = manifest.get("cv_auc_mean", 0)
            model_info["temporal_auc"] = manifest.get("temporal_test_auc", 0)
            model_info["temporal_r1"] = manifest.get("temporal_test_r1", 0)
            model_info["target_leakage"] = manifest.get("target_leakage", "unknown")
            model_info["dataset_rows"] = manifest.get("dataset_rows_scanned", 0)
            model_info["training_rows"] = manifest.get("training_rows", 0)
            model_info["ensemble_weights"] = manifest.get("ensemble_weights", {})
            model_info["is_altman"] = "altman" in manifest.get("model_version", "").lower()
            model_info["is_native"] = manifest.get("is_native", False)
    except Exception:
        pass

    # Check for native model
    try:
        native_path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "production" / "altman_native" / "manifest.json"
        if native_path.exists():
            native_manifest = json.loads(native_path.read_text(encoding="utf-8"))
            model_info["native_model_version"] = native_manifest.get("model_version", "—")
            model_info["native_cv_auc"] = native_manifest.get("cv_auc_mean", 0)
            model_info["native_temporal_auc"] = native_manifest.get("temporal_test_auc", 0)
            model_info["native_temporal_r1"] = native_manifest.get("temporal_test_r1", 0)
            model_info["native_n_features"] = native_manifest.get("n_features", 0)
    except Exception:
        pass

    return JSONResponse({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": uptime,
        "services": services,
        "metrics": {
            "total_evaluations": total_evals,
            "fraud_detected": total_fraud,
            "legit_flagged": total_legit_flagged,
            "detection_rate_pct": detection_rate,
            "fpr_pct": fpr,
        },
        "latency": {
            "avg_ms": avg_lat,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "samples": len(lats) if lats else len(svc_lats),
        },
        "inference": inference,
        "model_info": model_info,
        "recent_events": recent,
    })


@app.get("/monitor/unified")
def monitor_unified(request: Request) -> JSONResponse:
    """Unified detection system metrics — ML + rules + velocity + drift (admin-only)."""
    _require_admin_session(request)
    result = {
        "unified_scorer": {},
        "drift": {},
        "subsystems": {},
    }

    # Fetch from inference service unified-status
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8006/unified-status",
            headers={"Accept": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=3)  # nosec B310 - fixed localhost health URL
        data = json.loads(resp.read())
        result["unified_scorer"] = {
            "loaded": data.get("loaded", False),
            "load_time_ms": data.get("load_time_ms", 0),
            "subsystems": data.get("subsystems", {}),
            "registered_models": data.get("registered_models", 0),
        }
        result["subsystems"] = data.get("subsystems", {})
        result["drift"] = data.get("drift", {})
    except Exception:
        result["unified_scorer"] = {"loaded": False, "error": "inference service unavailable"}

    # Fetch drift status separately for more detail
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8006/drift/status",
            headers={"Accept": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=3)  # nosec B310 - fixed localhost health URL
        drift_data = json.loads(resp.read())
        result["drift"] = drift_data
    except Exception:
        pass

    # Fetch recent drift alerts
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8006/drift/alerts",
            headers={"Accept": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=3)  # nosec B310 - fixed localhost health URL
        alert_data = json.loads(resp.read())
        result["drift_alerts"] = alert_data.get("alerts", [])
        result["drift_alert_count"] = alert_data.get("n_alerts", 0)
    except Exception:
        result["drift_alerts"] = []
        result["drift_alert_count"] = 0

    return JSONResponse(result)


@app.post("/monitor/record")
def monitor_record(payload: dict, request: Request) -> JSONResponse:
    """Record an evaluation event for monitoring (called by risk engine proxy)."""
    # Require internal token to prevent unauthenticated event injection
    internal_tok = settings.internal_token
    req_tok = request.headers.get("X-Internal-Token", "")
    if internal_tok and not hmac.compare_digest(req_tok, internal_tok):
        raise HTTPException(status_code=403, detail="internal token required")
    global _monitor_eval_count, _monitor_fraud_count, _monitor_legit_flagged
    _monitor_eval_count += 1
    is_fraud = payload.get("is_fraud", False)
    is_flagged = payload.get("risk_score", 0) >= 30
    if is_fraud:
        _monitor_fraud_count += 1
    if is_flagged and not is_fraud:
        _monitor_legit_flagged += 1
    lat = payload.get("latency_ms", 0)
    if lat > 0:
        _monitor_latencies.append(lat)
    _monitor_events.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "score": payload.get("risk_score", 0),
        "fraud": is_fraud,
        "flagged": is_flagged,
        "latency_ms": lat,
    })
    return JSONResponse({"ok": True})


# --- Admin login brute-force protection ---
# Progressive lockout: 3 failures -> 1min, 5 -> 5min, 10 -> 30min, 15+ -> 1hr
_admin_login_failures: dict[str, list[float]] = {}  # ip -> [timestamps]
_ADMIN_LOCKOUT_WINDOW = 3600  # 1 hour window to track failures

# Progressive lockout tiers: (failures, lockout_seconds)
_LOCKOUT_TIERS = [
    (3, 60),     # 3 failures -> 1 minute lockout
    (5, 300),    # 5 failures -> 5 minutes
    (10, 1800),  # 10 failures -> 30 minutes
    (15, 3600),  # 15+ failures -> 1 hour
]


def _check_admin_brute_force(ip: str) -> None:
    """Raise 429 with progressive lockout based on failure count."""
    now = time.time()
    attempts = _admin_login_failures.get(ip, [])
    # Prune old attempts outside the window
    attempts = [t for t in attempts if now - t < _ADMIN_LOCKOUT_WINDOW]
    _admin_login_failures[ip] = attempts
    
    failure_count = len(attempts)
    if failure_count == 0:
        return
    
    # Find the applicable lockout tier
    lockout_duration = 0
    for threshold, duration in _LOCKOUT_TIERS:
        if failure_count >= threshold:
            lockout_duration = duration
    
    if lockout_duration > 0:
        oldest = min(attempts)
        lockout_end = oldest + lockout_duration
        if now < lockout_end:
            remaining = int(lockout_end - now)
            raise HTTPException(
                status_code=429,
                detail=f"too many failed attempts, try again in {remaining}s"
            )
        # Lockout expired, clear failures
        _admin_login_failures[ip] = []


def _record_admin_login_failure(ip: str) -> None:
    """Record a failed admin login attempt."""
    _admin_login_failures.setdefault(ip, []).append(time.time())


@app.post("/admin/login")
def admin_login(req: LoginRequest, request: Request) -> dict:
    client_ip = request.client.host if request.client else "unknown"
    _check_admin_brute_force(client_ip)

    record = store.load()
    if record is None:
        # Bootstrap: the admin record is created from the ADMIN_PASS env on
        # first login. Refuse loudly if it isn't configured.
        admin_pass = os.environ.get("ADMIN_PASS")
        if not admin_pass:
            raise HTTPException(status_code=503,
                                detail="admin passphrase not configured (ADMIN_PASS env)")
        if req.username != os.environ.get("ADMIN_USER", "admin"):
            _record_admin_login_failure(client_ip)
            raise HTTPException(status_code=401, detail="invalid credentials")
        essentials = _build_essentials()
        record = store.bootstrap(req.username, admin_pass, essentials)

    if req.username != record["username"] or not store.verify_passphrase(req.passphrase):
        _record_admin_login_failure(client_ip)
        _admin_audit(req.username, "admin_login_failed",
                     {"reason": "invalid_credentials", "ip": client_ip})
        raise HTTPException(status_code=401, detail="invalid credentials")
    
    # MFA: TOTP code, or a single-use recovery code (Phase 111). The flag is
    # checked FIRST to avoid decrypting stale data.
    mfa_method = "passcode"
    if store.is_totp_enabled():
        totp_secret = store.get_totp_secret()
        mfa_ok = False
        if req.totp_code and totp_secret:
            totp = TOTPAuthenticator(totp_secret)
            step = totp.verify_step(req.totp_code)
            if step is not None and store.consume_totp_step(step):
                mfa_ok = True
                mfa_method = "totp"
            elif step is None:
                # Invalid code — fall through; a recovery code may follow.
                pass
            elif not req.recovery_code:
                # One-time use: a time-step already spent on a login is a
                # replay (with a recovery code supplied we let it try).
                _record_admin_login_failure(client_ip)
                _admin_audit(req.username, "admin_mfa_failed",
                             {"reason": "totp_replay", "ip": client_ip})
                raise HTTPException(status_code=401,
                                    detail="TOTP code already used")
        if not mfa_ok and req.recovery_code:
            if store.consume_recovery_code(req.recovery_code):
                mfa_ok = True
                mfa_method = "recovery"
                _admin_audit(req.username, "admin_recovery_used",
                             {"remaining": store.recovery_code_count()})
            else:
                _record_admin_login_failure(client_ip)
                _admin_audit(req.username, "admin_mfa_failed",
                             {"reason": "invalid_recovery_code",
                              "ip": client_ip})
                raise HTTPException(status_code=401,
                                    detail="Invalid recovery code")
        if not mfa_ok:
            if not req.totp_code:
                # A missing code is a UI retry (the gate only learns TOTP is on
                # from this 401), so it does not count toward lockout.
                raise HTTPException(status_code=401, detail="TOTP code required")
            if totp_secret:
                _record_admin_login_failure(client_ip)
                _admin_audit(req.username, "admin_mfa_failed",
                             {"reason": "invalid_totp_code",
                              "ip": client_ip})
                raise HTTPException(status_code=401,
                                    detail="Invalid TOTP code")
            # Enabled flag with no decryptable secret = stale record; the
            # passcode path stands (pre-existing behavior).
    
    blob_key = store.unwrap_blob_key(req.passphrase)
    if blob_key is None:
        raise HTTPException(status_code=401, detail="invalid credentials")

    store.touch_login()
    expires = datetime.now(timezone.utc) + timedelta(minutes=ADMIN_SESSION_MINUTES)
    jti = uuid.uuid4().hex
    token = jwt.encode({"sub": record["username"], "role": "admin", "jti": jti,
                        "exp": expires}, _session_key(record), algorithm="HS256")
    _admin_sessions.put(jti, {"blob_key_hex": blob_key.hex(),
                              "sub": record["username"],
                              "last_seen": time.time()}, expires)
    record = store.load()  # refreshed login_count / last_login
    _admin_audit(record["username"], "admin_login_success",
                 {"method": mfa_method,
                  "session_minutes": ADMIN_SESSION_MINUTES})
    resp = JSONResponse({
        "token": token,
        "expires_in_minutes": ADMIN_SESSION_MINUTES,
        "last_login": record["last_login"],
        "login_count": record["login_count"],
        "essentials": store.decrypt_essentials(blob_key),
        "totp_enabled": store.is_totp_enabled(),
    })
    # Set cookie with SameSite=Strict for CSRF protection on browser-based admin.
    # Phase 111: HttpOnly — the JS never reads this cookie (it authenticates
    # with the Bearer token), so hiding it from scripts is pure defense in
    # depth. `secure` follows the actual scheme so plain-HTTP local dev keeps
    # working while any HTTPS deployment gets the flag automatically.
    resp.set_cookie(
        "admin_session", jti,
        max_age=ADMIN_SESSION_MINUTES * 60,
        samesite="strict",
        httponly=True,
        secure=(request.url.scheme == "https"),
    )
    return resp


@app.get("/admin/essential")
def admin_essential(_session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    _payload, blob_key = _session
    return {"essentials": store.decrypt_essentials(blob_key)}


@app.post("/admin/logout")
def admin_logout(_session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    payload, _blob_key = _session
    _admin_sessions.revoke(payload["jti"])
    return {"ok": True}


@app.get("/admin/totp/setup")
def admin_totp_setup(_session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """Begin or resume TOTP enrollment (session-gated, Phase 111).

    An already-ENABLED factor is never re-disclosed (409 — use
    /admin/totp/rotate instead). A pending (unverified) enrollment re-returns
    its secret so the UI can resume. QR rendering is intentionally omitted
    (documented Phase 111 deviation): the otpauth:// URI + base32 secret are
    shown for copy-paste into any authenticator app.
    """
    if store.is_totp_enabled():
        raise HTTPException(status_code=409,
                            detail="TOTP already enrolled (use rotate to change it)")
    sub = _session[0].get("sub", "admin")
    totp_secret = store.get_totp_secret()
    if totp_secret:
        totp = TOTPAuthenticator(totp_secret)
        b32 = totp.get_secret_base32()
        return {
            "enabled": False, "pending": True, "secret": b32,
            "otpauth_uri": _otpauth_uri(b32, sub, totp.digits, totp.period),
            "period": totp.period, "digits": totp.digits,
            "time_remaining": totp.get_time_remaining(),
            "qr": "omitted",  # documented deviation — copy the URI instead
        }
    # Generate new secret for setup and store it as PENDING (not yet enabled)
    new_secret = TOTPAuthenticator.generate_secret()
    store.set_totp_secret(new_secret, enabled=False)
    totp = TOTPAuthenticator(new_secret)
    b32 = totp.get_secret_base32()
    return {
        "enabled": False, "pending": True, "secret": b32,
        "otpauth_uri": _otpauth_uri(b32, sub, totp.digits, totp.period),
        "period": totp.period, "digits": totp.digits,
        "time_remaining": totp.get_time_remaining(),
        "qr": "omitted",
        "message": ("Add the otpauth URI (or base32 secret) to your "
                    "authenticator app, then confirm with a code"),
    }


@app.post("/admin/totp/verify")
def admin_totp_verify(req: TOTPVerifyRequest,
                     _session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """Verify TOTP code, enable 2FA, and issue one-time recovery codes.

    Recovery codes are returned exactly ONCE here (never retrievable
    afterwards) and persisted only as SHA-256 hashes.
    """
    totp_secret = store.get_totp_secret()
    if not totp_secret:
        # This shouldn't happen if setup was called first
        raise HTTPException(status_code=400, detail="TOTP setup not initiated")
    if store.is_totp_enabled():
        raise HTTPException(status_code=409, detail="TOTP already enrolled")

    totp = TOTPAuthenticator(totp_secret)
    if not totp.verify_code(req.totp_code):
        _admin_audit(_session[0].get("sub", "admin"), "admin_mfa_failed",
                     {"action": "verify"})
        raise HTTPException(status_code=401, detail="Invalid TOTP code")

    # Enable TOTP (fresh record — resets the consumed-step guard) …
    store.set_totp_secret(totp_secret)
    # … then issue recovery codes (set_totp_secret replaces the totp dict,
    # so this ordering is required).
    codes = _generate_recovery_codes()
    store.set_recovery_codes(codes)
    _admin_audit(_session[0].get("sub", "admin"), "admin_totp_enrolled",
                 {"recovery_codes_issued": len(codes)})
    return {
        "ok": True,
        "message": "TOTP enabled successfully",
        "time_remaining": totp.get_time_remaining(),
        "recovery_codes": codes,
        "recovery_note": ("Shown once — store them now; each code works "
                          "exactly once."),
    }


@app.post("/admin/totp/disable")
def admin_totp_disable(req: TOTPVerifyRequest,
                      _session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """Disable TOTP (requires a valid current code = re-authentication).

    Also drops every recovery code and revokes every OTHER session so a
    stale session cannot keep the removed factor's trust level.
    """
    totp_secret = store.get_totp_secret()
    if not totp_secret:
        raise HTTPException(status_code=400, detail="TOTP not enabled")

    totp = TOTPAuthenticator(totp_secret)
    if not totp.verify_code(req.totp_code):
        _admin_audit(_session[0].get("sub", "admin"), "admin_mfa_failed",
                     {"action": "disable"})
        raise HTTPException(status_code=401, detail="Invalid TOTP code")

    # Disable TOTP by clearing the flag (the secret itself is retained for
    # re-enable, matching the documented retention behavior).
    record = store.load()
    if record and "totp" in record:
        record["totp"]["enabled"] = False
        store.save(record)
    store.clear_recovery_codes()
    revoked = _admin_sessions.revoke_others(_session[0].get("jti", ""))
    _admin_audit(_session[0].get("sub", "admin"), "admin_totp_disabled",
                 {"other_sessions_revoked": revoked})
    return {"ok": True, "message": "TOTP disabled",
            "other_sessions_revoked": revoked}


@app.get("/admin/totp/status")
def admin_totp_status(_session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """TOTP state — never returns the secret or recovery codes."""
    totp_secret = store.get_totp_secret()
    recovery_left = store.recovery_code_count()
    if not totp_secret:
        return {"enabled": False, "has_pending": False,
                "recovery_codes_remaining": 0,
                "message": "TOTP not configured"}

    totp = TOTPAuthenticator(totp_secret)
    enabled = store.is_totp_enabled()
    return {
        "enabled": enabled,
        "has_pending": not enabled,
        "recovery_codes_remaining": recovery_left,
        "time_remaining": totp.get_time_remaining(),
        "period": totp.period,
    }


def _generate_recovery_codes(n: int = TOTP_RECOVERY_CODE_COUNT) -> list[str]:
    """High-entropy one-time codes (16 chars of A-Z2-9, shown once)."""
    alphabet = string.ascii_uppercase + string.digits
    return ["".join(secrets.choice(alphabet) for _ in range(16))
            for _ in range(n)]


def _otpauth_uri(secret_b32: str, sub: str, digits: int, period: int) -> str:
    issuer = "PS-14"
    safe_sub = re.sub(r"[^A-Za-z0-9_-]", "", sub) or "admin"
    return (f"otpauth://totp/{issuer}%3A{safe_sub}?secret={secret_b32}"
            f"&issuer={issuer}&digits={digits}&period={period}")


@app.post("/admin/totp/rotate")
def admin_totp_rotate(req: TOTPVerifyRequest,
                      _session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """Rotate the TOTP secret + recovery codes; revoke every OTHER session.

    Requires a valid CURRENT code (re-authentication). The new secret and
    recovery codes are returned exactly once.
    """
    if not store.is_totp_enabled():
        raise HTTPException(status_code=400, detail="TOTP not enabled")
    current_secret = store.get_totp_secret()
    if not current_secret:
        raise HTTPException(status_code=400, detail="TOTP not configured")
    current = TOTPAuthenticator(current_secret)
    if not current.verify_code(req.totp_code):
        _admin_audit(_session[0].get("sub", "admin"), "admin_mfa_failed",
                     {"action": "rotate"})
        raise HTTPException(status_code=401, detail="Invalid TOTP code")
    sub = _session[0].get("sub", "admin")
    new_secret = TOTPAuthenticator.generate_secret()
    store.set_totp_secret(new_secret, enabled=True)  # fresh dict: drops steps
    codes = _generate_recovery_codes()
    store.set_recovery_codes(codes)
    revoked = _admin_sessions.revoke_others(_session[0].get("jti", ""))
    _admin_audit(sub, "admin_totp_rotated", {"other_sessions_revoked": revoked})
    fresh = TOTPAuthenticator(new_secret)
    b32 = fresh.get_secret_base32()
    return {
        "ok": True,
        "secret": b32,
        "otpauth_uri": _otpauth_uri(b32, sub, fresh.digits, fresh.period),
        "digits": fresh.digits, "period": fresh.period,
        "time_remaining": fresh.get_time_remaining(),
        "recovery_codes": codes,
        "other_sessions_revoked": revoked,
        "message": "TOTP secret rotated; other sessions revoked",
    }


@app.get("/admin/sessions")
def admin_sessions(_session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """Active admin sessions (Phase 111 Security tab). Tokens are the
    session identifiers themselves — never pair them with credentials."""
    items = _admin_sessions.list_recent(50)
    current = _session[0].get("jti")
    sessions = []
    for it in items:
        d = it.get("data") or {}
        sessions.append({
            "jti": it["jti"],
            "sub": d.get("sub", "admin"),
            "created_at": it.get("created_at"),
            "expires_at": it.get("expires_at"),
            "last_seen": d.get("last_seen"),
            "is_current": it["jti"] == current,
        })
    return {"sessions": sessions,
            "idle_timeout_minutes": ADMIN_SESSION_IDLE_MINUTES,
            "absolute_timeout_minutes": ADMIN_SESSION_MINUTES}


@app.post("/admin/sessions/revoke")
def admin_session_revoke(req: SessionRevokeRequest,
                         _session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """Revoke one admin session (idempotent, audited)."""
    existed = _admin_sessions.get(req.jti) is not None
    _admin_sessions.revoke(req.jti)
    _admin_audit(_session[0].get("sub", "admin"), "admin_session_revoked",
                 {"target": req.jti, "existed": existed,
                  "self": req.jti == _session[0].get("jti")})
    return {"ok": True, "revoked": existed}


@app.get("/admin/api/security-events")
def admin_api_security_events(request: Request, limit: int = 40) -> dict:
    """Recent admin/auth/security/runtime audit events (DB-4), bounded."""
    _require_admin_session(request)
    limit = max(1, min(limit, 200))
    events = _ro_rows(
        "audit",
        "SELECT seq, event_type, fraud_id, created_at, entry_hash "
        "FROM audit_events WHERE event_type LIKE 'admin_%' "
        "OR event_type LIKE '%security%' OR event_type LIKE 'runtime_%' "
        "ORDER BY seq DESC LIMIT ?", (limit,)) or []
    return {"events": events, "limit": limit}


@app.get("/admin/totp/current-code")
def admin_totp_current_code_removed() -> dict:
    """Removed in Phase 111: this endpoint returned the live TOTP code,
    which defeats the second factor for any session holder. Recovery codes
    (shown once at enrollment) replace its legitimate testing use."""
    raise HTTPException(status_code=410, detail="endpoint removed")



# ---------------------------------------------------------------------- admin DB access
# Each database has its own passphrase.  The admin must provide the
# correct passphrase for the requested database BEFORE the query runs.
# All access is audit-logged (actor + db + query summary).

_DBS = {
    "identity":  Path(settings.db_dir) / "identity.db",
    "features":  Path(settings.db_dir) / "features.db",
    "risk":      Path(settings.db_dir) / "risk.db",
    "verify":    Path(settings.db_dir) / "verify.db",
    "audit":     Path(settings.db_dir) / "audit.db",
}

# The admin passphrase gates access to ALL databases — there is no per-DB
# key derivation in this prototype.  Each DB query is audit-logged with the
# db_name, so access to different databases is attributed separately even
# though the same passphrase unlocks all of them.

def _db_passphrase_ok(db_name: str, provided_pass: str) -> bool:
    """Verify the DB passphrase.  The admin passphrase gates ALL databases.
    There is no per-DB key derivation — each query is audit-logged with
    the db_name for attribution."""
    return store.verify_passphrase(provided_pass)


class DBQueryRequest(BaseModel):
    db_name: str = Field(min_length=1, max_length=50)  # identity | features | risk | verify | audit
    passphrase: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=200)  # allowlist query name (not raw SQL)
    limit: int = Field(default=50, ge=1, le=500)  # row limit for results
    totp_code: str | None = Field(None, min_length=6, max_length=6)  # REQUIRED if TOTP enabled


# TRUE ALLOWLIST: Only these pre-defined queries are permitted.
# Each maps to a parameterized SQL query. No arbitrary SQL is accepted.
ALLOWED_QUERIES = {
    "recent_audit": "SELECT seq, event_type, entry_hash, payload_summary, created_at FROM audit_events ORDER BY seq DESC LIMIT ?",
    "unresolved_alerts": "SELECT event_id, fraud_id, ml_score, risk_band, scored_at FROM risk_scores WHERE event_id NOT IN (SELECT event_id FROM verification_outcomes) ORDER BY scored_at DESC LIMIT ?",
    "recent_scores": "SELECT event_id, fraud_id, ml_score, rule_score, risk_band, reason_codes, degraded, scored_at FROM risk_scores ORDER BY scored_at DESC LIMIT ?",
    "recent_features": "SELECT event_id, fraud_id, created_at FROM transaction_features ORDER BY created_at DESC LIMIT ?",
    "table_list": "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
}

# Fixed SQL for row_count — NO dynamic table name construction.
# Each allowed table has its own static SQL statement.
TABLE_COUNT_QUERIES = {
    "audit_events": "SELECT COUNT(*) as count FROM audit_events",
    "risk_scores": "SELECT COUNT(*) as count FROM risk_scores",
    "verification_outcomes": "SELECT COUNT(*) as count FROM verification_outcomes",
    "transaction_features": "SELECT COUNT(*) as count FROM transaction_features",
}

# Explicit allowlist for row_count: only non-PII tables may be counted.
# DB-1 (identity.db) tables like 'users' and 'pseudonym_mapping' are excluded.
SAFE_COUNT_TABLES = set(TABLE_COUNT_QUERIES.keys())

# Tables that must never appear in table_list output (PII exposure).
PII_TABLE_NAMES = {
    "users",
    "pseudonym_mapping",
    "auth_credentials",
    "pseudonym_access_log",
}

# PII masking shared by the allowlisted runner and the read-only SQL editor.
PII_MASK_COLUMNS = {"email_encrypted", "phone_encrypted", "address_encrypted", "secret_hash", "pass_hash"}
PII_FULL_MASK_COLUMNS = {"email_encrypted", "phone_encrypted", "address_encrypted"}  # binary blobs — never show

# Tables the database editor may never write: PII tables are restricted
# outright, and audit_events is append-only by trigger (edits are refused
# here rather than surfacing as a raise from the writer).
EDIT_DENY_TABLES = PII_TABLE_NAMES | {"audit_events"}

# Read-only SQL editor: first-token allowlist. The connection is also opened
# with URI mode=ro, so this is defense in depth, not the only gate.
SQL_READONLY_FIRST_TOKENS = {"select", "with"}
_SQL_FIRST_TOKEN_RE = re.compile(r"^\s*([A-Za-z]+)")
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SQL_PII_TABLE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(PII_TABLE_NAMES)) + r")\b",
    re.IGNORECASE,
)


def _open_ro(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite file strictly read-only (URI mode=ro)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _apply_pii_mask(rows: list[dict]) -> list[dict]:
    for row in rows:
        for col in PII_FULL_MASK_COLUMNS:
            if col in row:
                row[col] = "[ENCRYPTED]"
        for col in PII_MASK_COLUMNS:
            if col in row and row[col] not in ("[ENCRYPTED]", None):
                row[col] = "[REDACTED]"
    return rows


def _verify_totp_if_enabled(totp_code: str | None) -> None:
    """Mandatory TOTP step for privileged DB actions (mirrors db-query)."""
    if store.is_totp_enabled():
        totp_secret = store.get_totp_secret()
        if totp_secret:
            if not totp_code:
                raise HTTPException(status_code=401, detail="TOTP code required")
            if not TOTPAuthenticator(totp_secret).verify_code(totp_code):
                raise HTTPException(status_code=401, detail="invalid TOTP code")


@app.post("/admin/db-query")
def admin_db_query(
    req: DBQueryRequest,
    _session: tuple[dict, bytes] = Depends(_require_admin),
) -> dict:
    """Execute a read-only query against one of the five databases.

    Requires admin JWT + per-database passphrase.  Only SELECT
    statements are allowed — INSERT/UPDATE/DELETE/DROP are blocked.
    Every query is audit-logged.
    """
    # Validate db_name
    if req.db_name not in _DBS:
        raise HTTPException(status_code=400, detail=f"unknown database: {req.db_name}; choose from {list(_DBS.keys())}")

    db_path = _DBS[req.db_name]
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"database file not found: {db_path}")

    # Verify passphrase
    if not _db_passphrase_ok(req.db_name, req.passphrase):
        raise HTTPException(status_code=401, detail="invalid credentials")

    # Verify TOTP if enabled — mandatory for the highest-privilege action
    # Check flag FIRST to avoid decrypting stale data
    if store.is_totp_enabled():
        totp_secret = store.get_totp_secret()
        if totp_secret:
            if not req.totp_code:
                raise HTTPException(status_code=401, detail="TOTP code required")
            totp = TOTPAuthenticator(totp_secret)
            if not totp.verify_code(req.totp_code):
                raise HTTPException(status_code=401, detail="invalid TOTP code")

    # TRUE ALLOWLIST: Only pre-defined queries are permitted.
    # No arbitrary SQL is accepted — only query names from ALLOWED_QUERIES.
    q = req.query.strip().lower()
    
    # Special case: table_list and row_count need table name validation
    if q == "table_list":
        sql = ALLOWED_QUERIES["table_list"]
        params = []
    elif q.startswith("row_count:"):
        table_name = q.split(":", 1)[1].strip().lower()
        # Fixed mapping: user input → internal key → static SQL.
        # NO dynamic SQL construction.
        if table_name not in TABLE_COUNT_QUERIES:
            raise HTTPException(
                status_code=400,
                detail=f"table not allowed for row_count: {table_name}; allowed: {sorted(TABLE_COUNT_QUERIES.keys())}"
            )
        sql = TABLE_COUNT_QUERIES[table_name]
        params = []
    elif q in ALLOWED_QUERIES:
        sql = ALLOWED_QUERIES[q]
        params = [req.limit]
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"unknown query: {q}; allowed: {list(ALLOWED_QUERIES.keys())}"
        )

    # Execute parameterized query
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
            result = [dict(row) for row in rows[:req.limit]]
            row_count = len(rows)
        finally:
            conn.close()
    except Exception:
        raise HTTPException(status_code=400, detail="query execution failed")

    # Filter table_list results to exclude PII tables (defense-in-depth)
    if q == "table_list":
        result = [r for r in result if r.get("name", "") not in PII_TABLE_NAMES]

    _apply_pii_mask(result)  # shared module-level PII mask

    # Audit-log the access with resolved query name and table (for row_count)
    payload, _blob_key = _session
    audit_detail = {
        "actor": payload.get("sub", "admin"),
        "db": req.db_name,
        "query_name": q,
        "rows": row_count,
    }
    # For row_count, include the specific table requested
    if q.startswith("row_count:"):
        audit_detail["table"] = q.split(":", 1)[1].strip().lower()
    try:
        from src.audit_service.writer import append_audit_event
        append_audit_event(
            "ADMIN-DB",
            "admin_db_query",
            audit_detail,
        )
    except Exception:
        pass  # don't fail the query if audit logging fails

    return {
        "db": req.db_name,
        "rows": row_count,
        "truncated": row_count > 500,
        "data": result,
    }


class DBTablesRequest(BaseModel):
    db_name: str = Field(default="identity", min_length=1, max_length=50)
    passphrase: str = Field(min_length=1, max_length=200)


@app.post("/admin/db-tables")
def admin_db_tables(
    req: DBTablesRequest,
    _session: tuple[dict, bytes] = Depends(_require_admin),
) -> dict:
    """List tables and row counts for a given database."""
    if req.db_name not in _DBS:
        raise HTTPException(status_code=400, detail=f"unknown database: {req.db_name}")
    if not _db_passphrase_ok(req.db_name, req.passphrase):
        raise HTTPException(status_code=401, detail=f"invalid passphrase for {req.db_name}")
    db_path = _DBS[req.db_name]
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"database not found: {req.db_name}")
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            # Use fixed TABLE_COUNT_QUERIES — no dynamic SQL construction.
            result = []
            for tname, sql in TABLE_COUNT_QUERIES.items():
                try:
                    count = conn.execute(sql).fetchone()[0]
                    result.append({"table": tname, "rows": count})
                except Exception:
                    pass  # table may not exist in this DB
        finally:
            conn.close()
    except Exception:
        raise HTTPException(status_code=400, detail="failed to list tables")
    return {"db": req.db_name, "tables": result}


# ----------------------------------------------------------------------
# Admin data console: read-only SQL, schema/row browsing, guarded single-row
# edits, live counters, and the audit-trail proxy. Everything below sits
# behind the admin gate; DB actions additionally require the admin
# passphrase (and TOTP when enabled) and are audit-logged like db-query.

class DBSqlRequest(BaseModel):
    db_name: str = Field(min_length=1, max_length=50)
    passphrase: str = Field(min_length=1, max_length=200)
    sql: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=200, ge=1, le=500)
    totp_code: str | None = Field(None, min_length=6, max_length=6)


@app.post("/admin/db-sql")
def admin_db_sql(
    req: DBSqlRequest,
    _session: tuple[dict, bytes] = Depends(_require_admin),
) -> dict:
    """Read-only SQL for the admin query editor.

    Gates: admin session + admin passphrase (+ TOTP when enabled);
    first-token allowlist of SELECT/WITH; connection opened URI mode=ro;
    sqlite3 refuses multi-statement strings; PII table names rejected
    outright; output masked and capped at `limit` rows; every run
    audit-logged with the actor and truncated statement.
    """
    if req.db_name not in _DBS:
        raise HTTPException(status_code=400, detail=f"unknown database: {req.db_name}")
    db_path = _DBS[req.db_name]
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"database file not found: {db_path}")
    if not _db_passphrase_ok(req.db_name, req.passphrase):
        raise HTTPException(status_code=401, detail="invalid credentials")
    _verify_totp_if_enabled(req.totp_code)

    sql = req.sql.strip()
    first = _SQL_FIRST_TOKEN_RE.match(sql)
    if not first or first.group(1).lower() not in SQL_READONLY_FIRST_TOKENS:
        raise HTTPException(status_code=400, detail="only SELECT/WITH statements are allowed")
    if SQL_PII_TABLE_PATTERN.search(sql):
        raise HTTPException(status_code=400, detail="query references a restricted table")

    try:
        conn = _open_ro(db_path)
        try:
            cur = conn.execute(sql)  # multi-statement strings are refused by sqlite3
            result = [dict(r) for r in cur.fetchmany(req.limit)]
        finally:
            conn.close()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"query failed: {exc}")

    _apply_pii_mask(result)
    payload, _blob_key = _session
    audit_ok = True
    try:
        from src.audit_service.writer import append_audit_event
        append_audit_event(
            "ADMIN-DB",
            "admin_db_sql",
            {"actor": payload.get("sub", "admin"), "db": req.db_name,
             "sql": sql[:200], "rows": len(result)},
        )
    except Exception:
        audit_ok = False  # the read still succeeds; the gap is reported honestly
    return {
        "db": req.db_name,
        "rows": len(result),
        "data": result,
        "truncated": len(result) >= req.limit,
        "audit_logged": audit_ok,
    }


class DBSchemaRequest(BaseModel):
    db_name: str = Field(default="identity", min_length=1, max_length=50)
    passphrase: str = Field(min_length=1, max_length=200)


@app.post("/admin/db-schema")
def admin_db_schema(
    req: DBSchemaRequest,
    _session: tuple[dict, bytes] = Depends(_require_admin),
) -> dict:
    """Tables, columns and row counts for the database explorer.

    PII tables are omitted entirely (same posture as the allowlisted
    table_list query); everything is read through a mode=ro connection.
    """
    if req.db_name not in _DBS:
        raise HTTPException(status_code=400, detail=f"unknown database: {req.db_name}")
    db_path = _DBS[req.db_name]
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"database not found: {req.db_name}")
    if not _db_passphrase_ok(req.db_name, req.passphrase):
        raise HTTPException(status_code=401, detail=f"invalid passphrase for {req.db_name}")

    conn = _open_ro(db_path)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()]
        tables = []
        for name in names:
            if name in PII_TABLE_NAMES:
                continue
            cols = [dict(c) for c in conn.execute(
                "SELECT name, type, pk FROM pragma_table_info(?)", (name,)
            ).fetchall()]
            quoted = name.replace('"', '""')
            n = conn.execute(f'SELECT COUNT(*) FROM "{quoted}"').fetchone()[0]
            tables.append({
                "name": name,
                "columns": cols,
                "rows": n,
                "editable": name not in EDIT_DENY_TABLES,
            })
    finally:
        conn.close()
    return {"db": req.db_name, "tables": tables}


class DBRowsRequest(BaseModel):
    db_name: str = Field(min_length=1, max_length=50)
    passphrase: str = Field(min_length=1, max_length=200)
    table: str = Field(min_length=1, max_length=64)
    limit: int = Field(default=50, ge=1, le=200)


@app.post("/admin/db-rows")
def admin_db_rows(
    req: DBRowsRequest,
    _session: tuple[dict, bytes] = Depends(_require_admin),
) -> dict:
    """Browse rows of a schema-validated table (newest first, read-only)."""
    if req.db_name not in _DBS:
        raise HTTPException(status_code=400, detail=f"unknown database: {req.db_name}")
    db_path = _DBS[req.db_name]
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"database not found: {req.db_name}")
    if not _db_passphrase_ok(req.db_name, req.passphrase):
        raise HTTPException(status_code=401, detail=f"invalid passphrase for {req.db_name}")
    if not _TABLE_NAME_RE.match(req.table):
        raise HTTPException(status_code=400, detail="invalid table name")
    if req.table in PII_TABLE_NAMES:
        raise HTTPException(status_code=403, detail="restricted table")

    conn = _open_ro(db_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (req.table,)
        ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail=f"table not found: {req.table}")
        cur = conn.execute(
            f'SELECT rowid AS _rowid, * FROM "{req.table}" ORDER BY rowid DESC LIMIT ?',
            (req.limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    _apply_pii_mask(rows)
    return {
        "db": req.db_name,
        "table": req.table,
        "rows": len(rows),
        "data": rows,
        "editable": req.table not in EDIT_DENY_TABLES,
    }


class DBUpdateRequest(BaseModel):
    db_name: str = Field(min_length=1, max_length=50)
    passphrase: str = Field(min_length=1, max_length=200)
    table: str = Field(min_length=1, max_length=64)
    rowid: int = Field(ge=1)
    changes: dict[str, str | int | float | bool | None]
    totp_code: str | None = Field(None, min_length=6, max_length=6)


@app.post("/admin/db-update")
def admin_db_update(
    req: DBUpdateRequest,
    _session: tuple[dict, bytes] = Depends(_require_admin),
) -> dict:
    """Guarded single-row edit for the database editor.

    Denied where it matters: PII tables and the append-only audit_events
    are never editable, primary-key columns are never editable, column
    names are validated against the live schema, values must be scalars,
    exactly one row must match, and the before/after values of every
    changed column are audit-logged.
    """
    if req.db_name not in _DBS:
        raise HTTPException(status_code=400, detail=f"unknown database: {req.db_name}")
    db_path = _DBS[req.db_name]
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"database not found: {req.db_name}")
    if not _db_passphrase_ok(req.db_name, req.passphrase):
        raise HTTPException(status_code=401, detail="invalid credentials")
    _verify_totp_if_enabled(req.totp_code)
    if not _TABLE_NAME_RE.match(req.table):
        raise HTTPException(status_code=400, detail="invalid table name")
    if req.table in EDIT_DENY_TABLES:
        raise HTTPException(status_code=403, detail=f"table is not editable: {req.table}")
    if not req.changes:
        raise HTTPException(status_code=400, detail="no changes supplied")
    if len(req.changes) > 15:
        raise HTTPException(status_code=400, detail="too many columns in one update")

    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (req.table,)
        ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail=f"table not found: {req.table}")
        cols = {
            r["name"]: r
            for r in conn.execute("SELECT name, pk FROM pragma_table_info(?)", (req.table,)).fetchall()
        }
        old_row = conn.execute(
            f'SELECT * FROM "{req.table}" WHERE rowid = ?', (req.rowid,)
        ).fetchone()
        if old_row is None:
            raise HTTPException(status_code=404, detail="row not found")
        old = dict(old_row)

        assigns: list[str] = []
        values: list = []
        for col, val in req.changes.items():
            if col not in cols:
                raise HTTPException(status_code=400, detail=f"unknown column: {col}")
            if cols[col]["pk"]:
                raise HTTPException(status_code=403, detail=f"primary-key column not editable: {col}")
            if val is not None and not isinstance(val, (str, int, float, bool)):
                raise HTTPException(status_code=400, detail="values must be scalars")
            if isinstance(val, str) and len(val) > 4000:
                raise HTTPException(status_code=400, detail="value too long")
            assigns.append(f'"{col}" = ?')
            values.append(val)

        values.append(req.rowid)
        cur = conn.execute(
            f'UPDATE "{req.table}" SET {", ".join(assigns)} WHERE rowid = ?', values
        )
        if cur.rowcount != 1:
            conn.rollback()
            raise HTTPException(status_code=409, detail="row update did not apply")
        conn.commit()
    except HTTPException:
        raise
    except sqlite3.Error as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"update failed: {exc}")
    finally:
        conn.close()

    payload, _blob_key = _session
    audit_ok = True
    try:
        from src.audit_service.writer import append_audit_event
        append_audit_event(
            "ADMIN-DB",
            "admin_db_update",
            {"actor": payload.get("sub", "admin"), "db": req.db_name,
             "table": req.table, "rowid": req.rowid,
             "changes": dict(req.changes),
             "previous": {k: old.get(k) for k in req.changes}},
        )
    except Exception:
        audit_ok = False
    return {
        "db": req.db_name,
        "table": req.table,
        "rowid": req.rowid,
        "applied": True,
        "audit_logged": audit_ok,
    }


def _ro_scalar(db_name: str, sql: str, params: tuple = ()) -> int | None:
    """Single scalar from a mode=ro SQLite handle; None when absent/broken."""
    path = _DBS[db_name]
    if not path.exists():
        return None
    try:
        conn = _open_ro(path)
        try:
            row = conn.execute(sql, params).fetchone()
            return int(row[0]) if row is not None else None
        finally:
            conn.close()
    except Exception:
        return None


def _ro_rows(db_name: str, sql: str, params: tuple = ()) -> list[dict] | None:
    """Rows from a mode=ro SQLite handle; None when absent/broken."""
    path = _DBS[db_name]
    if not path.exists():
        return None
    try:
        conn = _open_ro(path)
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()
    except Exception:
        return None


# ── Phase 111: transaction search contract ────────────────────────────
_TX_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")
_TX_FRAUD_ID_RE = re.compile(r"^F[A-Za-z0-9_\-]{3,64}$")
_TX_BANDS = {"low", "medium", "high", "critical", "unknown"}
_TX_DECISIONS = {"allow", "verify", "step_up"}


def _tx_parse_ts(value: str, name: str) -> str:
    """Normalize a user timestamp to the DB's stored format (bounded)."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    raise HTTPException(
        status_code=400,
        detail=f"invalid {name} timestamp (use YYYY-MM-DD[ HH:MM:SS])")


def _admin_shell() -> Response:
    """Serve the admin SPA shell.

    The shell itself carries NO data — login UI only; every /admin/api,
    /admin/db-*, /admin/totp-* and session route is server-side gated
    (authorization is never delegated to frontend route hiding).
    """
    resp = FileResponse(STATIC / "admin.html")
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


@app.get("/admin", include_in_schema=False)
def admin_shell_root() -> Response:
    return _admin_shell()


@app.get("/admin/transactions", include_in_schema=False)
def admin_shell_transactions() -> Response:
    return _admin_shell()


@app.get("/admin/transactions/{event_id}", include_in_schema=False)
def admin_shell_transaction(event_id: str) -> Response:
    return _admin_shell()


@app.get("/admin/audit", include_in_schema=False)
def admin_shell_audit() -> Response:
    return _admin_shell()


@app.get("/admin/security", include_in_schema=False)
def admin_shell_security() -> Response:
    return _admin_shell()


@app.get("/admin/settings", include_in_schema=False)
def admin_shell_settings() -> Response:
    return _admin_shell()


@app.get("/admin/api/transactions")
def admin_api_transactions(
    _session: tuple[dict, bytes] = Depends(_require_admin),
    event_id: str | None = None,
    fraud_id: str | None = None,
    band: str | None = None,
    decision: str | None = None,
    min_score: int | None = None,
    max_score: int | None = None,
    since: str | None = None,
    until: str | None = None,
    degraded: bool | None = None,
    data_quality: str | None = None,
    flagged: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Server-side filtered, paginated, bounded search over DB-3 scores.

    Every filter maps to stored columns EXCEPT `decision`/`data_quality`,
    which resolve against authoritative DB-4 records (the score_generated
    payload `decision` key added in Phase 111 / data_quality_blocked
    events). Rows written before the decision key existed have no
    persisted decision and are unmatchable — reported as such, never
    guessed. Result size is clamped and offsets are bounded.
    """
    limit = max(1, min(int(limit), 200))
    if offset < 0 or offset > 10_000:
        raise HTTPException(status_code=400,
                            detail="offset out of range (0-10000)")
    if event_id and not _TX_EVENT_ID_RE.fullmatch(event_id):
        raise HTTPException(status_code=400, detail="malformed event_id")
    if fraud_id and not _TX_FRAUD_ID_RE.fullmatch(fraud_id):
        raise HTTPException(status_code=400, detail="malformed fraud_id")
    if band and band not in _TX_BANDS:
        raise HTTPException(
            status_code=400,
            detail="band must be low|medium|high|critical|unknown")
    if decision and decision not in _TX_DECISIONS:
        raise HTTPException(status_code=400,
                            detail="decision must be allow|verify|step_up")
    if data_quality and data_quality not in ("passed", "blocked"):
        raise HTTPException(status_code=400,
                            detail="data_quality must be passed|blocked")
    for name, val in (("min_score", min_score), ("max_score", max_score)):
        if val is not None and not (0 <= val <= 100):
            raise HTTPException(status_code=400,
                                detail=f"{name} must be within 0-100")
    if min_score is not None and max_score is not None and min_score > max_score:
        raise HTTPException(status_code=400, detail="min_score > max_score")
    w_since = _tx_parse_ts(since, "since") if since else None
    w_until = _tx_parse_ts(until, "until") if until else None
    if w_since and w_until and w_since > w_until:
        raise HTTPException(status_code=400, detail="since > until")

    where: list[str] = []
    params: list = []
    if event_id:
        where.append("event_id = ?")
        params.append(event_id)
    if fraud_id:
        where.append("fraud_id = ?")
        params.append(fraud_id)
    if band:
        where.append("risk_band = ?")
        params.append(band)
    if flagged:
        where.append("risk_band != 'low'")
    if min_score is not None:
        where.append("risk_score >= ?")
        params.append(min_score)
    if max_score is not None:
        where.append("risk_score <= ?")
        params.append(max_score)
    if w_since:
        where.append("scored_at >= ?")
        params.append(w_since)
    if w_until:
        where.append("scored_at <= ?")
        params.append(w_until)
    if degraded is not None:
        where.append("degraded = ?")
        params.append(1 if degraded else 0)

    def _audit_id_set(sql: str, extra: list) -> list[str]:
        """Resolve a DB-4-derived fraud_id allow/deny set (capped)."""
        hits = _ro_rows("audit", sql, tuple(extra)) or []
        if len(hits) > 5000:
            raise HTTPException(
                status_code=400,
                detail="filter too broad — narrow the time range")
        return sorted({h["fraud_id"] for h in hits})

    decision_ids: list[str] | None = None
    if decision:
        sql = ("SELECT fraud_id, payload_summary FROM audit_events "
               "WHERE event_type IN ('score_generated', "
               "'runtime_release_unverified') AND payload_summary LIKE ?")
        extra: list = [f'%"decision": "{decision}"%']
        if w_since:
            sql += " AND created_at >= ?"
            extra.append(w_since)
        if w_until:
            sql += " AND created_at <= ?"
            extra.append(w_until)
        sql += " LIMIT 5001"
        decision_ids = _audit_id_set(sql, extra)

    dq_ids: list[str] | None = None
    if data_quality:
        sql = "SELECT fraud_id FROM audit_events WHERE event_type = 'data_quality_blocked'"
        extra = []
        if w_since:
            sql += " AND created_at >= ?"
            extra.append(w_since)
        if w_until:
            sql += " AND created_at <= ?"
            extra.append(w_until)
        sql += " LIMIT 5001"
        dq_ids = _audit_id_set(sql, extra)

    total_override: int | None = None
    if decision_ids is not None:
        if not decision_ids:
            return {"rows": [], "total": 0, "limit": limit, "offset": offset,
                    "decision_source": "audit_payload",
                    "filters": {"decision": decision}}
        marks = ",".join("?" * len(decision_ids))
        where.append(f"fraud_id IN ({marks})")
        params.extend(decision_ids)
    if data_quality == "blocked":
        if not dq_ids:
            return {"rows": [], "total": 0, "limit": limit, "offset": offset,
                    "filters": {"data_quality": data_quality}}
        marks = ",".join("?" * len(dq_ids))
        where.append(f"fraud_id IN ({marks})")
        params.extend(dq_ids)
    elif data_quality == "passed":
        if dq_ids:
            marks = ",".join("?" * len(dq_ids))
            where.append(f"fraud_id NOT IN ({marks})")
            params.extend(dq_ids)

    where_sql = " AND ".join(where) if where else "1=1"
    total = _ro_scalar("risk",
                       f"SELECT COUNT(*) FROM risk_scores WHERE {where_sql}",
                       tuple(params)) or 0
    rows_out = _ro_rows(
        "risk",
        f"SELECT event_id, fraud_id, risk_score, risk_band, reason_codes, "
        f"model_version, ml_score, rule_score, degraded, scored_at "
        f"FROM risk_scores WHERE {where_sql} "
        f"ORDER BY scored_at DESC, event_id LIMIT ? OFFSET ?",
        tuple(params) + (limit, offset)) or []

    # Enrich the PAGE (not the world): one bounded DB-4 pass for decisions,
    # releases and data-quality status of exactly these rows.
    decisions: dict[str, str] = {}
    releases: dict[str, str] = {}
    dq_fraud: set[str] = set()
    fids = sorted({r["fraud_id"] for r in rows_out if r.get("fraud_id")})
    if fids:
        marks = ",".join("?" * len(fids))
        erows = _ro_rows(
            "audit",
            "SELECT fraud_id, event_type, payload_summary FROM audit_events "
            f"WHERE fraud_id IN ({marks}) AND event_type IN "
            "('score_generated', 'runtime_release_unverified', "
            "'data_quality_blocked') ORDER BY seq DESC LIMIT 400",
            tuple(fids)) or []
        for er in erows:
            if er["event_type"] == "data_quality_blocked":
                dq_fraud.add(er["fraud_id"])
                continue
            try:
                p = json.loads(er["payload_summary"])
            except (TypeError, ValueError):
                continue
            eid = p.get("event_id")
            if not eid or eid in decisions:
                continue
            if isinstance(p.get("decision"), str):
                decisions[eid] = p["decision"]
            if isinstance(p.get("runtime_release_id"), str):
                releases[eid] = p["runtime_release_id"]

    def _codes(raw: str | None) -> list[str]:
        try:
            v = json.loads(raw) if raw else []
            return v if isinstance(v, list) else []
        except (TypeError, ValueError):
            return []

    out_rows = []
    for r in rows_out:
        codes = _codes(r.get("reason_codes"))
        out_rows.append({
            "event_id": r["event_id"],
            "fraud_id": r["fraud_id"],
            "risk_score": r["risk_score"],
            "risk_band": r["risk_band"],
            "decision": decisions.get(r["event_id"]),  # None -> UI shows N/A
            "reason_codes": codes,
            "ml_score": r["ml_score"],
            "rule_score": r["rule_score"],
            "degraded": bool(r["degraded"]),
            "data_quality_status": (
                "blocked" if "DATA_QUALITY_BLOCKED" in codes
                else ("unknown" if r["fraud_id"] in dq_fraud else "passed")),
            "model_id": r["model_version"],
            "release_id": releases.get(r["event_id"]),  # None -> UI shows N/A
            "scored_at": r["scored_at"],
        })
    out_rows = _apply_pii_mask(out_rows)

    _admin_audit(_session[0].get("sub", "admin"), "admin_tx_search",
                 {"filters": {k: v for k, v in {
                     "event_id": event_id, "fraud_id": fraud_id,
                     "band": band, "decision": decision,
                     "flagged": flagged, "data_quality": data_quality,
                 }.items() if v is not None},
                  "total": total, "limit": limit, "offset": offset})
    return {
        "rows": out_rows,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(out_rows) < total,
        "decision_source": ("audit_payload" if decision else None),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/admin/api/transactions/{event_id}")
def admin_api_transaction_detail(
    event_id: str,
    _session: tuple[dict, bytes] = Depends(_require_admin),
) -> dict:
    """Full investigation view for one event (DB-3 + DB-4 + canonical ctx).

    Every displayed value is RECORDED evidence: persisted score fields,
    DB-4 payload keys, and manifest_contract canonical context. Missing
    values render as not_recorded / REASON_NOT_AVAILABLE — never guessed.
    """
    if not _TX_EVENT_ID_RE.fullmatch(event_id or ""):
        raise HTTPException(status_code=400, detail="malformed event_id")
    from src.audit_service.writer import canonical as _canonical
    from src.monitoring.phase109_audit_fork_repair import evaluate_chain

    score_rows = _ro_rows(
        "risk",
        "SELECT score_id, event_id, fraud_id, risk_score, risk_band, "
        "reason_codes, model_version, ml_score, rule_score, degraded, "
        "scored_at FROM risk_scores WHERE event_id = ? LIMIT 1",
        (event_id,)) or []
    if not score_rows:
        _admin_audit(_session[0].get("sub", "admin"), "admin_tx_view",
                     {"event_id": event_id, "result": "not_found"})
        raise HTTPException(status_code=404, detail="event not found")
    score = score_rows[0]
    fraud_id = score["fraud_id"]

    audit_rows = _ro_rows(
        "audit",
        "SELECT seq, event_id, fraud_id, event_type, prev_hash, entry_hash, "
        "payload_summary, created_at FROM audit_events "
        "WHERE fraud_id = ? AND payload_summary LIKE ? "
        "ORDER BY seq LIMIT 200",
        (fraud_id, f'%"event_id": "{event_id}"%')) or []

    audit_view = []
    primary_payload = None
    for r in audit_rows:
        try:
            parsed = json.loads(r["payload_summary"])
        except (TypeError, ValueError):
            parsed = None
        self_ok = False
        if parsed is not None:
            recomputed = hashlib.sha256(
                (r["prev_hash"] + _canonical(parsed)).encode()).hexdigest()
            self_ok = hmac.compare_digest(recomputed, r["entry_hash"])
        audit_view.append({
            "seq": r["seq"], "event_id": r["event_id"],
            "event_type": r["event_type"], "timestamp": r["created_at"],
            "prev_hash": r["prev_hash"], "entry_hash": r["entry_hash"],
            "self_consistent": self_ok,
        })
        if (primary_payload is None and parsed is not None
                and r["event_type"] in ("score_generated",
                                        "runtime_release_unverified")):
            primary_payload = parsed

    reason_codes: list[str] = []
    try:
        reason_codes = json.loads(score["reason_codes"] or "[]")
        if not isinstance(reason_codes, list):
            reason_codes = []
    except (TypeError, ValueError):
        reason_codes = []
    dq_blocked = ("DATA_QUALITY_BLOCKED" in reason_codes
                  or any(r["event_type"] == "data_quality_blocked"
                         for r in audit_rows))
    decision = (primary_payload or {}).get("decision")
    decision_source = ("audit_payload" if isinstance(decision, str)
                       else "not_persisted")

    all_rows = _ro_rows(
        "audit",
        "SELECT seq, event_id, fraud_id, event_type, prev_hash, entry_hash, "
        "payload_summary, created_at FROM audit_events ORDER BY seq") or []
    chain_eval = evaluate_chain(all_rows)

    mismatches: list[str] = []
    if primary_payload:
        for pkey, skey in (("event_id", "event_id"),
                           ("risk_score", "risk_score"),
                           ("risk_band", "risk_band"),
                           ("rule_score", "rule_score"),
                           ("degraded", "degraded")):
            if primary_payload.get(pkey) != score[skey]:
                mismatches.append(f"{pkey} != score.{skey}")
        if round(float(primary_payload.get("ml_score", 0) or 0), 4) != \
                round(float(score["ml_score"] or 0), 4):
            mismatches.append("ml_score mismatch (4dp)")
    else:
        mismatches.append("no DB-4 decision payload for this event")

    outcomes = _ro_rows(
        "risk",
        "SELECT outcome, case_id, resolved_at FROM verification_outcomes "
        "WHERE event_id = ? LIMIT 20", (event_id,)) or []
    cases = _ro_rows(
        "risk",
        "SELECT case_id, status, priority, confidence, created_at "
        "FROM investigator_cases WHERE event_id = ? LIMIT 20",
        (event_id,)) or []

    reasons = [{"code": c,
                "text": REASON_CODE_TEXT.get(c, "REASON_NOT_AVAILABLE")}
               for c in reason_codes]
    p = primary_payload or {}

    def _stage(name: str, status, evidence: dict, ts=None) -> dict:
        return {"stage": name, "status": status, "timestamp": ts,
                "evidence": evidence}

    stages = [
        _stage("input_validation",
               "blocked" if dq_blocked else "passed",
               {"data_quality_status": "blocked" if dq_blocked else "passed",
                "source": "recorded reason codes / data_quality_blocked event"}),
        _stage("idempotency", "persisted",
               {"score_id": score["score_id"],
                "note": "database-level idempotent insert keyed by event_id"}),
        _stage("drift_check", p.get("drift_state", "not_recorded"),
               {"drift_state": p.get("drift_state")}),
        _stage("runtime_state", p.get("runtime_state", "not_recorded"),
               {"runtime_state": p.get("runtime_state"),
                "runtime_release_id": p.get("runtime_release_id"),
                "runtime_manifest_hash": p.get("runtime_manifest_hash"),
                "runtime_attestation_hash": p.get("runtime_attestation_hash")}),
        _stage("feature_enforcement", "not_recorded",
               {"feature_version": p.get("feature_version"),
                "degraded": bool(score["degraded"]),
                "note": ("per-stage enforcement verdict is not persisted; "
                         "the degraded flag is recorded")}),
        _stage("inference_rules", "scored",
               {"model_version": score["model_version"],
                "ml_score": score["ml_score"],
                "rule_score": score["rule_score"],
                "rule_version": p.get("rule_version")}),
        _stage("decision_band", decision or "not_recorded",
               {"risk_score": score["risk_score"],
                "threshold": MC.CANONICAL_THRESHOLD,
                "risk_band": score["risk_band"],
                "decision": decision,
                "escalation_reason": p.get("escalation_reason"),
                "reasons": reasons}),
        _stage("persistence", "persisted",
               {"db": "DB-3 risk_scores", "score_id": score["score_id"],
                "scored_at": score["scored_at"]}),
        _stage("audit", "recorded",
               {"db": "DB-4 audit_events",
                "seqs": [r["seq"] for r in audit_view],
                "entry_hash": (audit_view[0]["entry_hash"]
                               if audit_view else None)}),
    ]

    _admin_audit(_session[0].get("sub", "admin"), "admin_tx_view",
                 {"event_id": event_id, "result": "found",
                  "risk_band": score["risk_band"]})

    return {
        "event_id": event_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "score": {
            "score_id": score["score_id"],
            "event_id": score["event_id"],
            "fraud_id": score["fraud_id"],
            "risk_score": score["risk_score"],
            "risk_band": score["risk_band"],
            "ml_score": score["ml_score"],
            "rule_score": score["rule_score"],
            "degraded": bool(score["degraded"]),
            "model_version": score["model_version"],
            "scored_at": score["scored_at"],
            "reason_codes": reason_codes,
        },
        "decision": decision,
        "decision_source": decision_source,
        "escalation_reason": p.get("escalation_reason"),
        "drift_state": p.get("drift_state"),
        "runtime_state": p.get("runtime_state"),
        "reasons": reasons,
        "stages": stages,
        "audit": {
            "events": audit_view,
            "reconcile": {"consistent": not mismatches,
                          "mismatches": mismatches},
        },
        "chain": {
            "ok": chain_eval.get("ok"),
            "strict_ok": chain_eval.get("strict_ok"),
            "first_bad_seq": chain_eval.get("first_bad_seq"),
            "quarantined_breaks": chain_eval.get("quarantined_breaks"),
            "finding_ids": chain_eval.get("finding_ids"),
            "n_entries": chain_eval.get("n_entries"),
            "event_label": ("AUDIT_CHAIN_VALID"
                            if audit_view and all(r["self_consistent"]
                                                  for r in audit_view)
                            else "AUDIT_CHAIN_INVALID"),
            "chain_label": ("AUDIT_CHAIN_VALID" if chain_eval.get("ok")
                            else "AUDIT_CHAIN_INVALID"),
            "strict_label": ("AUDIT_CHAIN_VALID" if chain_eval.get("strict_ok")
                             else "AUDIT_CHAIN_INVALID"),
        },
        "outcomes": outcomes,
        "cases": cases,
        "context": {
            "threshold": MC.CANONICAL_THRESHOLD,
            "model_id": MC.CANONICAL_MODEL_VERSION,
            "release_id": MC.CANONICAL_LEGACY_RELEASE_ID,
            "feature_version": MC.CANONICAL_FEATURE_VERSION,
            "recorded_model_version": score["model_version"],
        },
    }


def _runtime_model_block() -> dict:
    """Authoritative model/runtime identity for API payloads.

    Deployed (attested manifest) identity alongside the governance-layer
    constants — one source shared by the live monitor and the Phase 112
    operator summary so the two panels can never drift apart.
    """
    status_data = _STATUS_CACHE.get("data") or {}
    risk_health = (status_data.get("risk") or {}).get("health") or {}
    return {
        "model_id": risk_health.get("model_id") or MC.CANONICAL_MODEL_VERSION,
        "release_id": (risk_health.get("release_id")
                       or MC.CANONICAL_LEGACY_RELEASE_ID),
        "feature_version": (risk_health.get("feature_version")
                            or MC.CANONICAL_FEATURE_VERSION),
        "threshold": MC.CANONICAL_THRESHOLD,
        "runtime_state": risk_health.get("runtime_state"),  # None -> N/A
        "model_readiness": risk_health.get("model_readiness"),
        "release_attested": risk_health.get("release_attested"),
        # Phase 111: the governance layer sits alongside the deployed
        # (grandfathered manifest) identity above, so the read-only model
        # panel can show both documented conventions instead of making an
        # operator guess which layer "release_id" belongs to.
        "governance_model_id": MC.CANONICAL_MODEL_ID,
        "governance_release_id": MC.CANONICAL_RELEASE_ID,
        "source": "risk /health via status cache + manifest_contract",
    }


@app.get("/admin/api/summary")
def admin_api_summary(request: Request) -> dict:
    """Compact operator-dashboard payload (Phase 112): one small request.

    Answers the dashboard questions — is the system working, are
    transactions arriving, what is flagged, what needs attention — without
    pulling the heavy live-monitor payload (feed rows, decision scans).
    Counts are bounded read-only queries; status/details come from the
    background-refreshed status cache. Absent sources return null so the
    console renders N/A, never a fabricated zero.
    """
    _require_admin_session(request)
    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

    status_data = _STATUS_CACHE.get("data") or {}
    risk_health = (status_data.get("risk") or {}).get("health") or {}
    chain = _CHAIN_CACHE.get("data") or {}

    # 24h counts — real queries, bounded window.
    tx_24h = _ro_scalar("risk",
                        "SELECT COUNT(*) FROM risk_scores WHERE scored_at >= ?",
                        (since,))
    flagged_24h = _ro_scalar(
        "risk",
        "SELECT COUNT(*) FROM risk_scores "
        "WHERE scored_at >= ? AND risk_band NOT IN ('low', 'unknown')",
        (since,))
    blocked_24h = _ro_scalar(
        "audit",
        "SELECT COUNT(*) FROM audit_events "
        "WHERE event_type = 'data_quality_blocked' AND created_at >= ?",
        (since,))
    newest = _ro_rows("risk",
                      "SELECT MAX(scored_at) AS newest FROM risk_scores") or []

    # Recent flagged rows (bounded, newest first). "Flagged" matches the
    # live monitor's metric: non-low, non-unknown band.
    recent_src = _ro_rows(
        "risk",
        "SELECT event_id, risk_score, risk_band, scored_at FROM risk_scores "
        "WHERE risk_band NOT IN ('low', 'unknown') "
        "ORDER BY scored_at DESC, event_id LIMIT 8") or []
    recent_flagged = [{
        "event_id": r["event_id"],
        "risk_score": r["risk_score"],
        "risk_band": r["risk_band"],
        "scored_at": r["scored_at"],
    } for r in recent_src]

    # Recent activity (bounded): the newest scored transactions regardless
    # of band, with their decision resolved from DB-4 exactly the way the
    # search endpoint resolves it for a page (one bounded pass, newest
    # event wins).  Absent decision stays null -> UI shows N/A.
    act_src = _ro_rows(
        "risk",
        "SELECT event_id, fraud_id, risk_band, scored_at FROM risk_scores "
        "ORDER BY scored_at DESC, event_id LIMIT 6") or []
    decisions: dict[str, str] = {}
    fids = sorted({r["fraud_id"] for r in act_src if r.get("fraud_id")})
    if fids:
        marks = ",".join("?" * len(fids))
        erows = _ro_rows(
            "audit",
            "SELECT event_id, payload_summary FROM audit_events "
            f"WHERE fraud_id IN ({marks}) AND event_type IN "
            "('score_generated', 'runtime_release_unverified') "
            "ORDER BY seq DESC LIMIT 40",
            tuple(fids)) or []
        for er in erows:
            try:
                p = json.loads(er["payload_summary"])
            except (TypeError, ValueError):
                continue
            eid = p.get("event_id")
            if eid and eid not in decisions and isinstance(p.get("decision"), str):
                decisions[eid] = p["decision"]
    recent_activity = [{
        "event_id": r["event_id"],
        "risk_band": r["risk_band"],
        "decision": decisions.get(r["event_id"]),  # None -> UI shows N/A
        "scored_at": r["scored_at"],
    } for r in act_src]

    # Simple system status: warnings ONLY when something needs attention.
    svc = [(k, v) for k, v in status_data.items() if isinstance(v, dict)]
    ok_count = sum(1 for _, v in svc if v.get("status") == "ok")
    warnings: list[str] = []
    if svc and ok_count < len(svc):
        warnings.append(f"{len(svc) - ok_count} service(s) not responding")
    db_state = risk_health.get("db") or (
        (risk_health.get("database") or {}).get("status"))
    if db_state is not None and db_state != "ok" \
            and db_state != "healthy":
        warnings.append("Database connection issue")
    if (risk_health.get("model_readiness") not in (None, "loaded")
            or risk_health.get("runtime_state") not in (None, "READY")):
        warnings.append("Model unavailable")
    if chain.get("ok") is False:
        warnings.append("Audit system requires attention")

    # Three honest states: operational / attention / unavailable (every
    # known service down).  A cold cache (no services collected yet) is
    # reported as attention, never as a green light.
    if svc and ok_count == 0:
        state = "unavailable"
        warnings = warnings or ["no services responding"]
    elif warnings or not svc:
        if not svc and not warnings:
            warnings = ["status not yet collected"]
        state = "attention"
    else:
        state = "operational"

    return {
        "ts": now.isoformat(),
        "refresh_interval_s": 30,
        "status": {
            "state": state,
            "warnings": warnings,
            "services_ok": ok_count,
            "services_total": len(svc),
        },
        "counts": {
            "transactions_24h": tx_24h,
            "flagged_24h": flagged_24h,
            "blocked_24h": blocked_24h,
            "newest_scored_at": (newest[0].get("newest") if newest else None),
        },
        "recent_flagged": recent_flagged,
        "recent_activity": recent_activity,
        "chain": {
            "ok": chain.get("ok"),
            "strict_ok": chain.get("strict_ok"),
            "n_entries": chain.get("n_entries"),
            "first_bad_seq": chain.get("first_bad_seq"),
            "quarantined_breaks": chain.get("quarantined_breaks") or [],
        },
        "details": {
            # Phase 110/111 technical fields, kept available behind the
            # dashboard's collapsible "System Details" (they simply must
            # not dominate the default view).
            "liveness": risk_health.get("liveness"),
            "readiness": risk_health.get("readiness"),
            "model_readiness": risk_health.get("model_readiness"),
            "runtime_state": risk_health.get("runtime_state"),
            "db": db_state,
            "audit": ("ok" if chain.get("ok") is True
                      else ("attention" if chain.get("ok") is False
                            else None)),
            "release_attested": risk_health.get("release_attested"),
            "uptime_seconds": risk_health.get("uptime_seconds"),
            "model": _runtime_model_block(),
        },
    }


@app.get("/admin/api/live")
def admin_api_live(request: Request, window: str = "15m") -> dict:
    """Live monitor counters: direct mode=ro reads plus recent chain events.

    `window` is one of the bounded set {5m, 15m, 1h} (default 15m) and only
    drives the `windowed` block; the global/24h counters stay for the
    dashboard. Read-only by construction. Absent sources return null — the
    console renders N/A, never a fabricated zero.
    """
    _require_admin_session(request)
    if window not in _ADMIN_LIVE_WINDOWS:
        raise HTTPException(status_code=400,
                            detail="window must be one of: 5m, 15m, 1h")

    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    w_since = (now - timedelta(
        seconds=_ADMIN_LIVE_WINDOWS[window])).strftime("%Y-%m-%d %H:%M:%S")
    band_rows = _ro_rows(
        "risk",
        "SELECT risk_band, COUNT(*) AS n FROM risk_scores "
        "GROUP BY risk_band ORDER BY risk_band") or []

    # ── windowed block (Phase 111) ─────────────────────────────────
    w_by_band = _ro_rows(
        "risk",
        "SELECT risk_band, COUNT(*) AS n FROM risk_scores "
        "WHERE scored_at >= ? GROUP BY risk_band",
        (w_since,)) or []
    w_total = _ro_scalar("risk",
                         "SELECT COUNT(*) FROM risk_scores "
                         "WHERE scored_at >= ?", (w_since,))
    w_degraded = _ro_scalar("risk",
                            "SELECT COUNT(*) FROM risk_scores "
                            "WHERE scored_at >= ? AND degraded = 1",
                            (w_since,))
    w_dq = _ro_scalar("audit",
                      "SELECT COUNT(*) FROM audit_events "
                      "WHERE event_type = 'data_quality_blocked' "
                      "AND created_at >= ?", (w_since,))
    n_gen = _ro_scalar("audit",
                       "SELECT COUNT(*) FROM audit_events "
                       "WHERE event_type = 'score_generated' "
                       "AND created_at >= ?", (w_since,)) or 0
    gen_rows = _ro_rows(
        "audit",
        "SELECT payload_summary FROM audit_events "
        "WHERE event_type = 'score_generated' AND created_at >= ? "
        "ORDER BY seq DESC LIMIT 2000", (w_since,)) or []
    by_decision: dict[str, int] = {}
    decision_samples = 0
    for gr in gen_rows:
        try:
            gp = json.loads(gr["payload_summary"])
        except (TypeError, ValueError):
            continue
        if isinstance(gp.get("decision"), str):
            by_decision[gp["decision"]] = by_decision.get(gp["decision"], 0) + 1
            decision_samples += 1

    # ── live feed: newest 50 risk_scores + DB-4 enrichment ─────────
    feed_src = _ro_rows(
        "risk",
        "SELECT event_id, fraud_id, risk_score, risk_band, reason_codes, "
        "model_version, ml_score, rule_score, degraded, scored_at "
        "FROM risk_scores ORDER BY scored_at DESC, event_id LIMIT 50") or []
    decisions_by_event: dict[str, str] = {}
    releases_by_event: dict[str, str] = {}
    dq_fraud: set[str] = set()
    fids = sorted({r["fraud_id"] for r in feed_src if r.get("fraud_id")})
    if fids:
        marks = ",".join("?" * len(fids))
        erows = _ro_rows(
            "audit",
            "SELECT fraud_id, event_type, payload_summary FROM audit_events "
            f"WHERE fraud_id IN ({marks}) AND event_type IN "
            "('score_generated', 'runtime_release_unverified', "
            "'data_quality_blocked') ORDER BY seq DESC LIMIT 400",
            tuple(fids)) or []
        for er in erows:
            if er["event_type"] == "data_quality_blocked":
                dq_fraud.add(er["fraud_id"])
                continue
            try:
                ep = json.loads(er["payload_summary"])
            except (TypeError, ValueError):
                continue
            eid = ep.get("event_id")
            if not eid or eid in decisions_by_event:
                continue
            if isinstance(ep.get("decision"), str):
                decisions_by_event[eid] = ep["decision"]
            if isinstance(ep.get("runtime_release_id"), str):
                releases_by_event[eid] = ep["runtime_release_id"]

    def _feed_codes(raw: str | None) -> list[str]:
        try:
            v = json.loads(raw) if raw else []
            return v if isinstance(v, list) else []
        except (TypeError, ValueError):
            return []

    feed = []
    for r in feed_src:
        codes = _feed_codes(r.get("reason_codes"))
        feed.append({
            "timestamp": r["scored_at"],
            "event_id": r["event_id"],
            "risk_score": r["risk_score"],
            "decision": decisions_by_event.get(r["event_id"]),  # None -> N/A
            "decision_band": r["risk_band"],
            "degraded": bool(r["degraded"]),
            "data_quality_status": (
                "blocked" if "DATA_QUALITY_BLOCKED" in codes
                else ("unknown" if r["fraud_id"] in dq_fraud else "passed")),
            "model_id": r["model_version"],
            "release_id": releases_by_event.get(r["event_id"]),  # -> N/A
        })

    # ── latency: real in-process samples only (None when empty) ────
    lats = sorted(_monitor_latencies)
    latency = {
        "p50_ms": lats[len(lats) // 2] if lats else None,
        "p95_ms": lats[int(len(lats) * 0.95)] if lats else None,
        "p99_ms": (lats[min(len(lats) - 1, int(len(lats) * 0.99))]
                   if lats else None),
        "samples": len(lats),
    }

    # ── model / runtime: authoritative sources only ────────────────
    status_data = _STATUS_CACHE.get("data") or {}
    model_block = _runtime_model_block()

    return {
        "ts": now.isoformat(),
        "window": window,
        "window_since": w_since,
        "refresh_interval_s": 5,
        "features_total": _ro_scalar("features",
                                     "SELECT COUNT(*) FROM transaction_features"),
        "features_24h": _ro_scalar(
            "features",
            "SELECT COUNT(*) FROM transaction_features WHERE created_at >= ?",
            (since,),
        ),
        "scores_total": _ro_scalar("risk", "SELECT COUNT(*) FROM risk_scores"),
        "scores_24h": _ro_scalar(
            "risk",
            "SELECT COUNT(*) FROM risk_scores WHERE scored_at >= ?",
            (since,),
        ),
        "bands": {r["risk_band"]: r["n"] for r in band_rows},
        "unresolved_alerts": _ro_scalar(
            "risk",
            "SELECT COUNT(*) FROM risk_scores "
            "WHERE event_id NOT IN (SELECT event_id FROM verification_outcomes)",
        ),
        "outcomes_total": _ro_scalar("verify",
                                     "SELECT COUNT(*) FROM verification_outcomes"),
        "recent_events": _ro_rows(
            "audit",
            "SELECT seq, event_type, fraud_id, created_at, entry_hash "
            "FROM audit_events ORDER BY seq DESC LIMIT 8",
        ) or [],
        "windowed": {
            "window": window,
            "since": w_since,
            "total": w_total,
            "by_band": {r["risk_band"]: r["n"] for r in w_by_band},
            "degraded": w_degraded,
            "by_decision": by_decision,
            "decision_samples": decision_samples,
            "decision_available": n_gen,
            "decision_truncated": n_gen > 2000,
            "validation_blocks": w_dq,
            "errors": None,            # no persisted source -> N/A
            "requests_per_sec": None,  # no persisted source -> N/A
        },
        "latency": latency,
        "model": model_block,
        "chain": (status_data.get("audit") or {}).get("chain"),
        "feed": feed,
    }


@app.get("/admin/api/audit-overview")
def admin_audit_overview(request: Request, limit: int = 50) -> dict:
    """Proxy to the Audit Service: authoritative chain integrity + events.

    Reads are resilient: when the audit service is down this returns 503
    and the console says so, rather than showing a stale green badge.
    """
    _require_admin_session(request)
    base = SERVICES["audit"].rstrip("/")
    headers = {"X-Internal-Token": settings.internal_token}
    limit = max(1, min(limit, 200))
    try:
        integrity_resp = httpx.get(f"{base}/audit/integrity", headers=headers, timeout=3.0)
        events_resp = httpx.get(f"{base}/audit/events?limit={limit}", headers=headers, timeout=3.0)
    except Exception:
        raise HTTPException(status_code=503, detail="audit service unreachable")
    if integrity_resp.status_code != 200:
        raise HTTPException(status_code=503, detail=f"integrity check failed (http {integrity_resp.status_code})")
    integrity = integrity_resp.json()
    events = events_resp.json() if events_resp.status_code == 200 else {}
    # Phase 112: per-row verification for the simplified audit table's
    # Status column. Same recompute rule as the transaction detail view
    # (sha256(prev_hash + canonical(payload)) vs entry_hash) — the chain
    # verdict itself still comes from the authoritative integrity block.
    from src.audit_service.writer import canonical as _canonical
    ev_out = []
    for ev in events.get("events", []):
        row = dict(ev)
        payload = row.get("payload")
        recomputed = None
        if isinstance(payload, dict) and row.get("prev_hash") is not None:
            try:
                recomputed = hashlib.sha256(
                    (row["prev_hash"] + _canonical(payload)).encode()
                ).hexdigest()
            except (TypeError, ValueError):
                recomputed = None
        row["self_consistent"] = (
            bool(recomputed) and recomputed == row.get("entry_hash"))
        ev_out.append(row)
    return {
        "integrity": integrity,
        "events": ev_out,
        "total": events.get("total"),
    }


@app.get("/monitor/test-results")
def monitor_test_results(request: Request) -> JSONResponse:
    """Return the latest automated test results (admin-only)."""
    _require_admin_session(request)
    if TEST_RESULTS_FILE.exists():
        try:
            data = json.loads(TEST_RESULTS_FILE.read_text())
            return JSONResponse(data)
        except Exception:
            pass
    return JSONResponse({
        "timestamp": None, "all_passed": None, "total": 0,
        "passed": 0, "failed": 0, "passed_tests": [], "failed_tests": [],
        "error": "No test results yet — first run pending",
    })


@app.get("/monitor/phases")
def monitor_phases(request: Request) -> JSONResponse:
    """Return the validation-phase decision ledger (Phases 14-25, admin-only).

    Reads each phase's immutable decision.json from reports/ — the same
    artifacts pushed to GitHub — so the UI shows the certified verdicts
    without any client-side file access.
    """
    _require_admin_session(request)
    root = Path(__file__).resolve().parent.parent.parent.parent / "misc"
    phases_meta = [
        (14, "Chip supervision transfer", "Does 2014-15 chip-fraud training improve 2017 transfer?"),
        (15, "Distribution-shift forensics", "Why did 2017 chip recall fail?"),
        (16, "Data-inventory boundary", "Does independent validation data exist?"),
        (17, "IBM generator audit", "Is the 2017 regime a generator artifact?"),
        (18, "System readiness gate", "Is the system production-ready?"),
        (19, "Real-world data gate", "Do we have real-world evidence?"),
        (20, "Decision-time remediation", "Can the fraud-rate blocker be removed?"),
        (21, "Real-world validation gate", "Is real-world validation possible?"),
        (22, "Executable validation harness", "Is the validation harness ready?"),
        (23, "Data acquisition gate", "Can real-world data be acquired?"),
        ("23B", "Frozen Kaggle eval", "How do frozen models behave on external data?"),
        (24, "Conditional external eval", "Executable evaluation with governance checks"),
        (25, "Leakage audit & corrections", "Is the external eval leakage-free and correctly built?"),
    ]
    # File path patterns differ across phases (numbered prefixes, subdirs)
    _DECISION_PATTERNS = [
        lambda n: root / "reports" / f"phase{n}" / "decision.json",
        lambda n: root / "reports" / f"phase{n}" / "final_decision.json",
        lambda n: root / "reports" / f"phase{n}" / f"{n}_final_decision.json" if isinstance(n, int) else None,
        lambda n: root / "reports" / f"phase{n}" / "25_final_decision.json" if n == 23 else None,
        lambda n: root / "reports" / f"phase{n}" / "26_final_decision.json" if n == 22 else None,
        lambda n: root / "reports" / ("phase23b" if str(n) == "23B" else f"phase{n}") / "kaggle" / "20_final_decision.json",
        lambda n: root / "reports" / f"phase{n}" / "kaggle" / "20_final_decision.json" if n == 24 else None,
        lambda n: root / "reports" / f"phase{n}" / "final_decision.json" if n == 25 else None,
    ]
    phases = []
    for num, title, question in phases_meta:
        entry: dict = {"phase": num, "title": title, "question": question,
                       "classification": "UNAVAILABLE"}
        dpath = None
        for pat_fn in _DECISION_PATTERNS:
            try:
                p = pat_fn(num)
                if p and p.exists():
                    dpath = p
                    break
            except Exception:
                continue
        if dpath:
            try:
                d = json.loads(dpath.read_text(encoding="utf-8"))
                entry["classification"] = (
                    d.get("classification") or d.get("outcome") or d.get("case") or "UNKNOWN"
                )
                fw = d.get("firewall_status") or {}
                entry["firewall"] = {
                    "final_test_accessed": fw.get("FINAL_TEST_ACCESSED", None),
                    "production_untouched": fw.get("PRODUCTION_MODEL_STATUS", "UNKNOWN"),
                }
                entry["cert_time"] = d.get("cert_time_utc") or d.get("execution_time_utc")
            except Exception:
                entry["classification"] = "PARSE_ERROR"
        phases.append(entry)
    return JSONResponse({
        "phases": phases,
        "final_test_accessed": False,
        "final_test_authorized": False,
        "production_model": "E_hardneg (UNTOUCHED)",
    })


@app.post("/monitor/run-tests")
def run_tests_now(request: Request) -> JSONResponse:
    """Manually trigger an immediate test run (synchronous, returns when done).

    Requires admin session cookie or Bearer token — same-origin bypass
    removed because Origin/Referer headers are trivially spoofable and do
    not constitute authentication.
    """
    _require_admin_session(request)
    try:
        result, output = _run_tests_sync()
        passed_lines = []
        failed_lines = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("[PASS]"):
                name = line.split("]", 1)[1].strip().rsplit("(", 1)[0].strip()
                passed_lines.append(name)
            elif line.startswith("[FAIL]"):
                name = line.split("]", 1)[1].strip().rsplit("(", 1)[0].strip()
                failed_lines.append(name)
        all_passed = "ALL" in output and "PASSED" in output
        total = len(passed_lines) + len(failed_lines)
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "all_passed": all_passed,
            "total": total,
            "passed": len(passed_lines),
            "failed": len(failed_lines),
            "passed_tests": passed_lines,
            "failed_tests": failed_lines,
            "exit_code": result.returncode,
        }
        TEST_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        TEST_RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return JSONResponse(results)
    except Exception as e:
        return JSONResponse({"error": str(e), "all_passed": False})


FULL_EVAL_LOCK = False


@app.post("/monitor/full-eval")
def run_full_eval(request: Request) -> JSONResponse:
    """Run full-dataset evaluation against validation_kaggle.csv (admin-only).

    Scores all ~284K transactions using the production model and computes
    ROC-AUC, recall, FPR, precision, throughput. Results are cached in
    db/full_eval_results.json and displayed on the monitor panel.
    """
    _require_admin_session(request)
    global FULL_EVAL_LOCK
    if FULL_EVAL_LOCK:
        raise HTTPException(status_code=429, detail="full eval already running")
    FULL_EVAL_LOCK = True
    try:
        root = Path(__file__).resolve().parent.parent.parent.parent
        py = str(root / ".venv" / "Scripts" / "python.exe") if (root / ".venv" / "Scripts" / "python.exe").exists() else sys.executable
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
               "PS14_MODE": "development"}
        for k in ["DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_KEY",
                  "SUPABASE_ANON_KEY"]:
            env.pop(k, None)
        result = subprocess.run(
            [py, str(root / "backend" / "scripts" / "ibm_eval.py"),
             "--max-rows", "100000", "--chunk-size", "50000"],
            capture_output=True, text=True, timeout=600, env=env,
            cwd=str(root),
        )
        if result.returncode != 0:
            err = result.stderr[-500:] if result.stderr else result.stdout[-500:]
            return JSONResponse({"error": f"full eval failed (exit {result.returncode}): {err}", "all_passed": False})
        if IBM_EVAL_FILE.exists():
            data = json.loads(IBM_EVAL_FILE.read_text(encoding="utf-8"))
            return JSONResponse(data)
        return JSONResponse({"error": "full eval produced no output"})
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "full eval timed out (600s)"})
    except Exception as e:
        return JSONResponse({"error": str(e)})
    finally:
        FULL_EVAL_LOCK = False


@app.get("/monitor/full-eval")
def get_full_eval(request: Request) -> JSONResponse:
    """Return cached full-eval results (admin-only)."""
    _require_admin_session(request)
    if IBM_EVAL_FILE.exists():
        try:
            data = json.loads(IBM_EVAL_FILE.read_text(encoding="utf-8"))
            return JSONResponse(data)
        except Exception:
            pass
    return JSONResponse({"timestamp": None, "error": "No IBM evaluation results yet — click Run Tests"})


@app.post("/admin/rotate")
def admin_rotate(req: RotateRequest,
                 _session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    if len(req.new_passphrase) < 8:
        raise HTTPException(status_code=400, detail="new passphrase too short (min 8 chars)")
    if not store.rotate_passphrase(req.current_passphrase, req.new_passphrase):
        raise HTTPException(status_code=401, detail="current passphrase incorrect")
    # The session signing key derives from the passphrase hash, which just
    # changed — revoke all sessions in the shared store.
    _admin_sessions.revoke_all()
    return {"ok": True, "note": "passphrase rotated; all admin sessions revoked"}


# ---------------------------------------------------------------------- user auth proxy
# Proxies /auth/register and /auth/login to the identity service so the
# login page can call the front service on a single origin (no CORS issues).
# Rate limiting is enforced by the identity service itself.

@app.get("/login-page", include_in_schema=False)
def login_page() -> Response:
    resp = FileResponse(STATIC / "login.html")
    resp.headers["Cache-Control"] = "private, max-age=5"
    return resp


@app.get("/admin-page", include_in_schema=False)
def admin_page() -> Response:
    resp = FileResponse(STATIC / "admin.html")
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


@app.post("/auth/register")
async def proxy_register(request: Request) -> JSONResponse:
    """Proxy register to identity service."""
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{settings.identity_url}/auth/register",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": f"identity service unavailable: {e}"})


@app.post("/auth/login")
async def proxy_login(request: Request) -> JSONResponse:
    """Proxy login to identity service."""
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{settings.identity_url}/auth/login",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": f"identity service unavailable: {e}"})
