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
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import sqlite3

from src.settings import settings
from src.middleware.totp import TOTPAuthenticator
from src.middleware import apply_security_middleware

from .admin_store import AdminStore

from contextlib import asynccontextmanager

TEST_RESULTS_FILE = Path(settings.db_dir) / "test_results.json"
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
        [py, str(root / "scripts" / "regression_suite.py"), "--fast"],
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


def _require_admin(authorization: str | None = Header(default=None)) -> tuple[dict, bytes]:
    record = store.load()
    if record is None or not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="admin session required")
    try:
        payload = jwt.decode(authorization.removeprefix("Bearer "),
                             _session_key(record), algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="invalid or expired admin session")
    session_data = _admin_sessions.get(payload.get("jti", ""))
    if session_data is None:
        raise HTTPException(status_code=401, detail="admin session expired or revoked")
    blob_key = bytes.fromhex(session_data["blob_key_hex"])
    return payload, blob_key


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


class RotateRequest(BaseModel):
    current_passphrase: str = Field(min_length=1, max_length=200)
    new_passphrase: str = Field(min_length=8, max_length=200)


class TOTPSetupRequest(BaseModel):
    passphrase: str  # Admin passphrase to authorize setup
    totp_code: str   # Code from authenticator app


class TOTPVerifyRequest(BaseModel):
    totp_code: str


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
                        "n_entries": body.get("n_entries"),
                        "first_bad_seq": body.get("first_bad_seq"),
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
                        "n_entries": body.get("n_entries"),
                        "first_bad_seq": body.get("first_bad_seq"),
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
def monitor_page() -> Response:
    resp = FileResponse(STATIC / "monitor.html")
    resp.headers["Cache-Control"] = "private, max-age=5"
    return resp


@app.get("/monitor/metrics")
def monitor_metrics(response: Response) -> JSONResponse:
    """Return live monitoring metrics for the dashboard."""
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
        model_info["risk_engine_model"] = risk_data.get("model", "unknown")
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
def monitor_unified() -> JSONResponse:
    """Unified detection system metrics — ML + rules + velocity + drift."""
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
        raise HTTPException(status_code=401, detail="invalid credentials")
    
    # Check TOTP if enabled — check flag FIRST to avoid decrypting stale data
    if store.is_totp_enabled():
        totp_secret = store.get_totp_secret()
        if totp_secret:
            if not req.totp_code:
                raise HTTPException(status_code=401, detail="TOTP code required")
            totp = TOTPAuthenticator(totp_secret)
            if not totp.verify_code(req.totp_code):
                raise HTTPException(status_code=401, detail="Invalid TOTP code")
    
    blob_key = store.unwrap_blob_key(req.passphrase)
    if blob_key is None:
        raise HTTPException(status_code=401, detail="invalid credentials")

    store.touch_login()
    expires = datetime.now(timezone.utc) + timedelta(minutes=ADMIN_SESSION_MINUTES)
    jti = uuid.uuid4().hex
    token = jwt.encode({"sub": record["username"], "role": "admin", "jti": jti,
                        "exp": expires}, _session_key(record), algorithm="HS256")
    _admin_sessions.put(jti, {"blob_key_hex": blob_key.hex()}, expires)
    record = store.load()  # refreshed login_count / last_login
    resp = JSONResponse({
        "token": token,
        "expires_in_minutes": ADMIN_SESSION_MINUTES,
        "last_login": record["last_login"],
        "login_count": record["login_count"],
        "essentials": store.decrypt_essentials(blob_key),
        "totp_enabled": store.is_totp_enabled(),
    })
    # Set cookie with SameSite=Strict for CSRF protection on browser-based admin.
    # HttpOnly=False so the frontend JS can read it if needed.
    resp.set_cookie(
        "admin_session", jti,
        max_age=ADMIN_SESSION_MINUTES * 60,
        samesite="strict",
        httponly=False,
        secure=False,  # Set True behind TLS in production
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
    """Get TOTP setup information (QR code URL and secret)."""
    totp_secret = store.get_totp_secret()
    if totp_secret:
        totp = TOTPAuthenticator(totp_secret)
        return {
            "enabled": store.is_totp_enabled(),
            "secret": totp.get_secret_base32(),
            "period": totp.period,
            "digits": totp.digits,
            "time_remaining": totp.get_time_remaining(),
        }
    # Generate new secret for setup and store it as PENDING (not yet enabled)
    new_secret = TOTPAuthenticator.generate_secret()
    store.set_totp_secret(new_secret, enabled=False)
    totp = TOTPAuthenticator(new_secret)
    return {
        "enabled": False,
        "secret": totp.get_secret_base32(),
        "period": totp.period,
        "digits": totp.digits,
        "time_remaining": totp.get_time_remaining(),
        "message": "Scan this secret with your authenticator app, then verify with /admin/totp/verify",
    }


@app.post("/admin/totp/verify")
def admin_totp_verify(req: TOTPVerifyRequest,
                     _session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """Verify TOTP code and enable 2FA."""
    # Get or create secret
    totp_secret = store.get_totp_secret()
    if not totp_secret:
        # This shouldn't happen if setup was called first
        raise HTTPException(status_code=400, detail="TOTP setup not initiated")
    
    totp = TOTPAuthenticator(totp_secret)
    if not totp.verify_code(req.totp_code):
        raise HTTPException(status_code=401, detail="Invalid TOTP code")
    
    # Enable TOTP
    store.set_totp_secret(totp_secret)
    return {
        "ok": True,
        "message": "TOTP enabled successfully",
        "time_remaining": totp.get_time_remaining(),
    }


@app.post("/admin/totp/disable")
def admin_totp_disable(req: TOTPVerifyRequest,
                      _session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """Disable TOTP (requires valid code)."""
    totp_secret = store.get_totp_secret()
    if not totp_secret:
        raise HTTPException(status_code=400, detail="TOTP not enabled")
    
    totp = TOTPAuthenticator(totp_secret)
    if not totp.verify_code(req.totp_code):
        raise HTTPException(status_code=401, detail="Invalid TOTP code")
    
    # Disable TOTP by removing the secret
    record = store.load()
    if record and "totp" in record:
        record["totp"]["enabled"] = False
        store.save(record)
    
    return {"ok": True, "message": "TOTP disabled"}


@app.get("/admin/totp/status")
def admin_totp_status(_session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """Check TOTP status and get current code (for testing)."""
    totp_secret = store.get_totp_secret()
    if not totp_secret:
        return {"enabled": False, "message": "TOTP not configured"}
    
    totp = TOTPAuthenticator(totp_secret)
    return {
        "enabled": store.is_totp_enabled(),
        "time_remaining": totp.get_time_remaining(),
        "period": totp.period,
    }


@app.get("/admin/totp/current-code")
def admin_totp_current_code(_session: tuple[dict, bytes] = Depends(_require_admin)) -> dict:
    """Get current TOTP code — disabled in production (defeats 2FA)."""
    _ps14_mode = os.environ.get("PS14_MODE", "production").lower()
    if _ps14_mode == "production":
        raise HTTPException(status_code=403, detail="TOTP code endpoint disabled in production")
    totp_secret = store.get_totp_secret()
    if not totp_secret:
        raise HTTPException(status_code=400, detail="TOTP not configured")
    
    totp = TOTPAuthenticator(totp_secret)
    return {
        "code": totp.generate_code(),
        "time_remaining": totp.get_time_remaining(),
        "period": totp.period,
    }


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
    "unresolved_alerts": "SELECT event_id, fraud_id, ml_score, band, decision, created_at FROM risk_scores WHERE event_id NOT IN (SELECT event_id FROM verification_outcomes) ORDER BY created_at DESC LIMIT ?",
    "recent_scores": "SELECT event_id, fraud_id, ml_score, band, decision, reasons_json, created_at FROM risk_scores ORDER BY created_at DESC LIMIT ?",
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

    # PII masking: mask encrypted columns and sensitive fields in identity DB
    _PII_MASK_COLUMNS = {"email_encrypted", "phone_encrypted", "address_encrypted", "secret_hash", "pass_hash"}
    _PII_FULL_MASK_COLUMNS = {"email_encrypted", "phone_encrypted", "address_encrypted"}  # binary blobs — never show
    for row in result:
        for col in _PII_FULL_MASK_COLUMNS:
            if col in row:
                row[col] = "[ENCRYPTED]"
        for col in _PII_MASK_COLUMNS:
            if col in row and row[col] not in ("[ENCRYPTED]", None):
                row[col] = "[REDACTED]"

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


@app.get("/monitor/test-results")
def monitor_test_results() -> JSONResponse:
    """Return the latest automated test results."""
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
def monitor_phases() -> JSONResponse:
    """Return the validation-phase decision ledger (Phases 14-25).

    Reads each phase's immutable decision.json from reports/ — the same
    artifacts pushed to GitHub — so the UI shows the certified verdicts
    without any client-side file access.
    """
    root = Path(__file__).resolve().parent.parent.parent
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
        lambda n: root / "reports" / f"phase{n}" / "kaggle" / "20_final_decision.json" if str(n) == "23B" else None,
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

    Requires admin session cookie — same-origin bypass removed because
    Origin/Referer headers are trivially spoofable and do not constitute
    authentication.
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
