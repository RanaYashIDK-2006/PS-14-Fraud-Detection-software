#!/usr/bin/env python3
"""Phase 21 — batch demo failure-recovery & repeatability test (hermetic).

The Phase 20 usability suite covers the single-transaction demo path; this
suite covers the BATCH surface (`demo_batch_api`) — the parts where a user
could be misled by a half-finished run:

  1. invalid launch parameters are rejected (422) before any subprocess starts;
  2. a duplicate submission while a run is active is refused (409) and names
     the active report id;
  3. progress/status retrieval reflects the real active/latest run;
  4. a runner that dies without writing a terminal state is recorded as
     FAILED — never as a completed report (the false-success invariant);
  5. a failed run never appears as `completed` in the report list, and its
     report id is an honest not-found (no partial report is served);
  6. report retrieval + CSV export agree with the stored report; unknown ids
     404, unknown export formats 422, path-traversal ids 422;
  7. error responses carry no traceback, no internal token, and no absolute
     filesystem path;
  8. the per-client batch rate limit trips with an understandable 429.

No live stack and no model: the subprocess boundary is a fake process, and
runs live in a temp directory. Run from the repo root:

  PYTHONPATH=backend python backend/scripts/phase21_batch_recovery_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

# Hermetic stores BEFORE src.settings is imported (front service writes an
# admin/session store on import).
_TMP = tempfile.mkdtemp(prefix="ps14_phase21_")
os.environ["DB_DIR"] = _TMP
os.environ["PS14_MODE"] = "development"
os.environ.setdefault("INTERNAL_TOKEN", "phase21-batch-test-token")
os.environ.setdefault("JWT_SECRET", "phase21-batch-test-jwt-secret-value-32b")

from fastapi.testclient import TestClient  # noqa: E402

import src.front_service.demo_batch_api as api  # noqa: E402
import src.front_service.main as front  # noqa: E402

passed = failed = 0
errors: list[str] = []

# All batch state lives in a temp runs dir for this suite.
RUNS = Path(tempfile.mkdtemp(prefix="ps14_phase21_runs_"))
api.RUNS_DIR = RUNS
api.OUT_DIR = RUNS


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        errors.append(f"{name}{' — ' + detail if detail else ''}")
        print(f"  [FAIL] {name}{' — ' + detail if detail else ''}")


class FakeProc:
    """Minimal asyncio.subprocess.Process stand-in."""

    def __init__(self, exit_code: int | None, pid: int = 4242):
        self.returncode = exit_code
        self.pid = pid

    async def wait(self) -> int:
        return self.returncode


def write_run(report_id: str, *, progress: dict | None = None,
              report: dict | None = None) -> Path:
    d = RUNS / report_id
    d.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        (d / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    if report is not None:
        (d / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return d


def fake_completed_report(report_id: str, rows: int = 25000) -> dict:
    return {
        "report_id": report_id, "state": "completed", "execution_mode": "offline",
        "started_at": "2026-10-11T00:00:00+00:00",
        "completed_at": "2026-10-11T00:00:07+00:00", "elapsed_s": 7.0,
        "dataset": {"type": "synthetic",
                    "selection": {"rows_selected": rows, "seed": 42,
                                  "source_csv": "db/demo_batch/transactions_causal.csv"},
                    "sha256": "0" * 64, "schema_version": "ps14-features-v1"},
        "summary": {"rows_scored": rows,
                    "decision_counts": {"allow": rows - 10, "verify": 10, "step_up": 0}},
        "metrics": {"available": False, "reason": "fixture"},
        "limitations": ["fixture report for hermetic tests"],
    }


def main() -> int:
    global passed, failed
    client = TestClient(front.app)  # no context manager -> stays hermetic

    print("== Phase 21 batch failure-recovery test ==")
    print(f"temp runs dir: {RUNS}")
    print()

    # ── 1. invalid launch parameters rejected before any subprocess ────
    print("-- 1) invalid launch parameters --")
    spawned: list[list[str]] = []
    real_exec = api.asyncio.create_subprocess_exec

    async def spy_exec(*cmd, **kw):
        spawned.append(list(cmd))
        return FakeProc(None)

    api.asyncio.create_subprocess_exec = spy_exec
    try:
        for bad, label in [
            ({"mode": "turbo"}, "bad mode"),
            ({"rows": 10}, "rows below floor"),
            ({"rows": 1_000_000}, "rows above ceiling"),
            ({"seed": -1}, "negative seed"),
        ]:
            r = client.post("/demo/batch/run", json=bad)
            check(f"{label} -> 422", r.status_code == 422,
                  f"status={r.status_code} body={r.text[:120]}")
        check("no subprocess was spawned for invalid input", spawned == [])
    finally:
        api._active = None

    # ── 2. duplicate submission while active -> 409 ────────────────────
    print("-- 2) duplicate submission --")
    try:
        api._active = {"report_id": "RPT-FAKEACTIVE-aaaaaa",
                       "proc": FakeProc(None), "started_at": "2026-10-11T00:00:00+00:00",
                       "mode": "offline", "rows": 25000}
        r = client.post("/demo/batch/run", json={"mode": "offline"})
        check("second run while active -> 409", r.status_code == 409,
              f"status={r.status_code}")
        check("409 names the active report id", "RPT-FAKEACTIVE-aaaaaa" in r.text)
        check("no second subprocess spawned", spawned == [])
        check("409 leaks nothing",
              "Traceback" not in r.text and api.__name__ not in r.text)
    finally:
        api._active = None

    # ── 3. progress reflects real state ────────────────────────────────
    print("-- 3) progress retrieval --")
    r = client.get("/demo/batch/progress")
    body = r.json()
    check("progress 200", r.status_code == 200)
    check("no active run reported when idle", body.get("active") is None)
    check("latest is None with no reports", body.get("latest") is None)

    write_run("RPT-PROGRESS-111111",
              progress={"state": "running", "processed": 9_000, "total": 25_000})
    api._active = {"report_id": "RPT-PROGRESS-111111", "proc": FakeProc(None),
                   "started_at": "2026-10-11T00:00:00+00:00",
                   "mode": "offline", "rows": 25000}
    body = client.get("/demo/batch/progress").json()
    snap = body.get("active") or {}
    check("active run reported", snap.get("report_id") == "RPT-PROGRESS-111111")
    check("progress processed is real", snap.get("processed") == 9_000,
          str(snap.get("processed")))
    check("progress total is real", snap.get("total") == 25_000)
    check("progress state is running", snap.get("state") == "running")
    api._active = None

    # ── 4. runner death without terminal state -> FAILED, not completed ──
    print("-- 4) false-success invariant --")
    crashed_id = "RPT-CRASHED-222222"
    write_run(crashed_id, progress={"state": "running", "processed": 3, "total": 25_000})
    asyncio.run(api._watch(FakeProc(exit_code=1), crashed_id, "offline", 25000, None))
    prog = json.loads((RUNS / crashed_id / "progress.json").read_text(encoding="utf-8"))
    check("crashed run recorded as failed", prog.get("state") == "failed",
          str(prog.get("state")))
    check("exit code preserved", prog.get("exit_code") == 1)
    check("crashed run has no report.json",
          not (RUNS / crashed_id / "report.json").exists())

    # A runner that dies AFTER writing a completed report keeps completed.
    ok_id = "RPT-OKAFTER-333333"
    write_run(ok_id, progress={"state": "completed", "processed": 25000, "total": 25000},
              report=fake_completed_report(ok_id))
    asyncio.run(api._watch(FakeProc(exit_code=0), ok_id, "offline", 25000, None))
    prog = json.loads((RUNS / ok_id / "progress.json").read_text(encoding="utf-8"))
    check("completed run stays completed", prog.get("state") == "completed")

    # ── 5. failed runs never listed as completed; report is not-found ──
    print("-- 5) failed runs are honest --")
    listing = client.get("/demo/reports").json()["reports"]
    by_id = {r["report_id"]: r for r in listing}
    check("failed run appears in the list with its real state",
          by_id.get(crashed_id, {}).get("state") == "failed",
          str(by_id.get(crashed_id)))
    check("failed run is marked partial",
          by_id.get(crashed_id, {}).get("partial") is True)
    check("completed fixture listed as completed",
          by_id.get(ok_id, {}).get("state") == "completed")
    check("no listing entry claims a failed run completed",
          all(not (x["report_id"] == crashed_id and x["state"] == "completed")
              for x in listing))
    r = client.get(f"/demo/reports/{crashed_id}")
    check("failed run report id -> 404 (no partial report served)",
          r.status_code == 404, f"status={r.status_code}")
    check("404 body is the honest not-found", "not found" in r.text.lower())

    # ── 6. retrieval + export fidelity ─────────────────────────────────
    print("-- 6) retrieval and export --")
    stored = json.loads((RUNS / ok_id / "report.json").read_text(encoding="utf-8"))
    r = client.get(f"/demo/reports/{ok_id}")
    check("report retrieval returns the stored document", r.json() == stored)

    csv_resp = client.get(f"/demo/reports/{ok_id}/export?format=csv")
    check("csv export 200", csv_resp.status_code == 200)
    # csv.writer emits RFC-4180 CRLF line endings; strip per line.
    lines = [ln.rstrip("\r") for ln in csv_resp.text.strip().split("\n") if ln]
    check("csv header is field,value", lines[0] == "field,value", repr(lines[0]))
    check("csv carries the report id", f"report_id,{ok_id}" in csv_resp.text)
    check("csv rows_scored agrees with the report",
          "summary.rows_scored,25000" in csv_resp.text)
    check("csv decision counts agree with the report",
          "summary.decision_counts.allow,24990" in csv_resp.text)
    check("csv preserves the limitations block",
          "limitations," in csv_resp.text)

    json_resp = client.get(f"/demo/reports/{ok_id}/export?format=json")
    check("json export equals the stored report",
          json_resp.json() == stored)
    check("export sets a download filename",
          f'{ok_id}.csv' in csv_resp.headers.get("content-disposition", ""))

    check("unknown export format -> 422",
          client.get(f"/demo/reports/{ok_id}/export?format=pdf").status_code == 422)
    check("unknown report id -> 404",
          client.get("/demo/reports/RPT-DOES-NOT-EXIST").status_code == 404)
    # Encoded-slash traversal never matches the route (Starlette rejects it
    # before the handler); either rejection is safe as long as nothing is
    # served and the server does not error.
    traversal = client.get("/demo/reports/..%2F..%2Fetc")
    check("path traversal report id rejected (404/422, nothing served)",
          traversal.status_code in (404, 422),
          f"status={traversal.status_code} body={traversal.text[:80]}")
    check("traversal rejection serves no file content",
          "report_id" not in traversal.text)
    check("malformed report id -> 422",
          client.get("/demo/reports/not-a-valid-id!").status_code == 422)

    # ── 7. error responses leak nothing ───────────────────────────────
    print("-- 7) error hygiene --")
    for resp in (csv_resp, json_resp,
                 client.get("/demo/reports/RPT-NOPE"),
                 client.get(f"/demo/reports/{crashed_id}")):
        text = resp.text
        check("no internal token in response",
              os.environ["INTERNAL_TOKEN"] not in text)
        check("no traceback in response", "Traceback (most recent call last)" not in text)
        check("no absolute local path in response",
              ":\\" not in text and "/home/" not in text and "/Users/" not in text)
    check("stored fixture report has no absolute paths",
          ":\\" not in json.dumps(stored))

    # ── 8. batch rate limit ────────────────────────────────────────────
    print("-- 8) batch rate limit --")
    api._batch_hits.clear()
    api._active = None
    statuses = []
    for _ in range(api.DEMO_BATCH_RATE_MAX + 2):
        statuses.append(client.post("/demo/batch/run", json={"rows": 100}).status_code)
    check("requests under the limit accepted (202)",
          all(s == 202 for s in statuses[:api.DEMO_BATCH_RATE_MAX]),
          str(statuses[:api.DEMO_BATCH_RATE_MAX]))
    check("requests over the limit are 429",
          all(s == 429 for s in statuses[api.DEMO_BATCH_RATE_MAX:]),
          str(statuses[api.DEMO_BATCH_RATE_MAX:]))
    check("429 detail is understandable",
          "wait" in client.post("/demo/batch/run", json={"rows": 100}).text.lower())
    api._batch_hits.clear()
    # Let any spawned fake watchers settle, then clear active state.
    api._active = None
    print()

    print("== result ==")
    print(f"checks passed={passed} failed={failed}")
    if errors:
        print("failures:")
        for e in errors:
            print(f"  - {e}")
    print("NOTE: hermetic response-contract test for the batch demo surface — "
          "software behavior only, not real-world fraud-detection performance.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
