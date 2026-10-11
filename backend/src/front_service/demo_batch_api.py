"""Phase 20 — batch demo + reports API for the front service.

Endpoints (all same-origin, no token ever leaves the server):

  POST /demo/batch/run                    start a 25,000-row batch demo run
  GET  /demo/batch/progress               progress of the active/latest run
  GET  /demo/reports                      persisted report list (newest first)
  GET  /demo/reports/{report_id}          one full report
  GET  /demo/reports/{report_id}/export   JSON or CSV export (keeps metadata)

The heavy lifting happens in `backend/scripts/demo_batch_runner.py`, spawned as
one bounded subprocess per run. That keeps the page service light (it never
loads a model into its own process), gives real progress through a progress
file, and needs no queue or extra infrastructure. A run id is minted per run;
duplicate submissions are refused while a run is active, and a crashed run is
recorded as `failed` — never as a successful report.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "backend" / "scripts" / "demo_batch_runner.py"
OUT_DIR = ROOT / "db" / "demo_batch"
RUNS_DIR = OUT_DIR / "runs"

router = APIRouter()

DEMO_BATCH_RATE_WINDOW_S = 300
DEMO_BATCH_RATE_MAX = 6
_batch_hits: dict[str, list[float]] = {}

_active: dict | None = None  # {report_id, proc, started_at, mode, rows}


def _rate_limited(key: str) -> bool:
    now = time.time()
    hits = [t for t in _batch_hits.get(key, []) if now - t < DEMO_BATCH_RATE_WINDOW_S]
    if len(hits) >= DEMO_BATCH_RATE_MAX:
        _batch_hits[key] = hits
        return True
    hits.append(now)
    _batch_hits[key] = hits
    return False


class BatchRunRequest(BaseModel):
    """Batch demo launch parameters (strictly bounded allowlists)."""
    mode: str = Field(default="offline", pattern="^(offline|live)$")
    rows: int = Field(default=25_000, ge=100, le=100_000)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _report_path(report_id: str) -> Path:
    # report ids are minted by the runner (RPT-<stamp>-<hex>); reject anything
    # that could escape the runs directory.
    if not report_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=422, detail="invalid report id")
    return RUNS_DIR / report_id / "report.json"


def _active_snapshot() -> dict | None:
    if _active is None:
        return None
    progress = _read_json(RUNS_DIR / _active["report_id"] / "progress.json") or {}
    return {
        "report_id": _active["report_id"],
        "mode": _active["mode"],
        "rows": _active["rows"],
        "started_at": _active["started_at"],
        "state": progress.get("state", "running"),
        "processed": progress.get("processed", 0),
        "total": progress.get("total", _active["rows"]),
        "pid": _active["proc"].pid if _active.get("proc") else None,
    }


async def _watch(proc: asyncio.subprocess.Process, report_id: str, mode: str, rows: int,
                  log_handle=None) -> None:
    global _active
    code = await proc.wait()
    if log_handle is not None:
        try:
            log_handle.close()
        except Exception:  # noqa: BLE001
            pass
    progress_path = RUNS_DIR / report_id / "progress.json"
    progress = _read_json(progress_path) or {}
    if code != 0 and progress.get("state") not in ("completed", "failed"):
        # The runner died before writing a terminal state: record an honest
        # failed run so it can never be mistaken for a completed report.
        progress.update({"state": "failed", "exit_code": code,
                         "updated_at": datetime.now(timezone.utc).isoformat()})
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    if _active and _active["report_id"] == report_id:
        _active = None


@router.post("/demo/batch/run", status_code=202)
async def demo_batch_run(req: BatchRunRequest, request: Request) -> JSONResponse:
    global _active
    client_key = request.client.host if request.client else "unknown"
    if _rate_limited(client_key):
        raise HTTPException(status_code=429,
                            detail="Too many demo runs in the last few minutes — wait a bit and retry.")
    if _active is not None and _active.get("proc") and _active["proc"].returncode is None:
        raise HTTPException(
            status_code=409,
            detail=f"A batch demo run is already in progress ({_active['report_id']}).")

    report_id = await _mint_report_id()
    cmd = [
        sys.executable, str(RUNNER), "--mode", req.mode,
        "--rows", str(req.rows), "--seed", str(req.seed),
        "--noise", "0.15", "--out-dir", str(OUT_DIR), "--report-id", report_id,
    ]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": str(ROOT / "backend")}
    # Local SQLite demo mode: .env may carry Postgres/Supabase settings (the
    # local stack launcher strips them for the same reason). Without this the
    # runner would try to reach Postgres and die before scoring a single row.
    for key in ("DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_KEY",
                "SUPABASE_ANON_KEY", "DB_SCHEMA"):
        env.pop(key, None)
    run_dir = RUNS_DIR / report_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "runner.log"
    log_handle = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - closed in _watch
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(ROOT), env=env,
        stdout=log_handle, stderr=asyncio.subprocess.STDOUT)
    started = datetime.now(timezone.utc).isoformat()
    _active = {"report_id": report_id, "proc": proc, "started_at": started,
               "mode": req.mode, "rows": req.rows}
    asyncio.ensure_future(_watch(proc, report_id, req.mode, req.rows, log_handle))
    return JSONResponse({"report_id": report_id, "state": "running", "mode": req.mode,
                         "rows": req.rows, "started_at": started}, status_code=202)


async def _mint_report_id() -> str:
    """Mint a runner-style id that does not collide with an existing run."""
    from uuid import uuid4
    for _ in range(5):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = f"RPT-{stamp}-{uuid4().hex[:6]}"
        if not (RUNS_DIR / candidate).exists():
            return candidate
    raise HTTPException(status_code=503, detail="Could not mint a unique report id")


@router.get("/demo/batch/progress")
async def demo_batch_progress() -> JSONResponse:
    snap = _active_snapshot()
    if snap:
        return JSONResponse({"active": snap, "latest": None})
    latest = _latest_report_summary()
    return JSONResponse({"active": None, "latest": latest})


def _latest_report_summary() -> dict | None:
    reports = _list_reports(limit=1)
    return reports[0] if reports else None


def _list_reports(limit: int = 20) -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    out: list[dict] = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        report = _read_json(run_dir / "report.json")
        if report is None:
            progress = _read_json(run_dir / "progress.json")
            if progress and progress.get("state") in ("running", "failed"):
                out.append({"report_id": run_dir.name,
                            "state": progress.get("state"),
                            "started_at": progress.get("updated_at"),
                            "partial": True})
            continue
        out.append({
            "report_id": report.get("report_id", run_dir.name),
            "state": report.get("state"),
            "execution_mode": report.get("execution_mode"),
            "started_at": report.get("started_at"),
            "completed_at": report.get("completed_at"),
            "elapsed_s": report.get("elapsed_s"),
            "rows_selected": (report.get("dataset") or {}).get("selection", {}).get("rows_selected"),
            "rows_scored": (report.get("summary") or {}).get("rows_scored"),
            "decision_counts": (report.get("summary") or {}).get("decision_counts"),
        })
        if len(out) >= limit:
            break
    return out


@router.get("/demo/reports")
async def demo_reports() -> JSONResponse:
    return JSONResponse({"reports": _list_reports(),
                         "active": _active_snapshot()})


@router.get("/demo/reports/{report_id}")
async def demo_report(report_id: str) -> JSONResponse:
    path = _report_path(report_id)
    report = _read_json(path)
    if report is None:
        progress = _read_json(path.parent / "progress.json")
        if progress and progress.get("state") == "running":
            raise HTTPException(status_code=409, detail="Report is still being generated")
        raise HTTPException(status_code=404, detail="Report not found")
    return JSONResponse(report)


def _flatten(prefix: str, obj, rows: list[tuple[str, str]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, rows)
    elif isinstance(obj, list):
        rows.append((prefix, json.dumps(obj)))
    else:
        rows.append((prefix, "" if obj is None else str(obj)))


@router.get("/demo/reports/{report_id}/export")
async def demo_report_export(report_id: str, format: str = "json") -> Response:
    report = _read_json(_report_path(report_id))
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    if format == "json":
        return Response(
            content=json.dumps(report, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{report_id}.json"'})
    if format == "csv":
        rows: list[tuple[str, str]] = []
        _flatten("", report, rows)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["field", "value"])
        writer.writerows(rows)
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{report_id}.csv"'})
    raise HTTPException(status_code=422, detail="format must be json or csv")
