#!/usr/bin/env python3
"""PS-14 Phase 9 -- request-path performance & latency baseline (LIVE HTTP).

Evidence class: DEMONSTRATED (real uvicorn listener, real HTTP requests),
except the delayed/unavailable audit runs whose *fault* is SIMULATED (the
Phase 8 injection applied inside a live process by `ps14_p9_fault_boot.py`).

What this harness does, in order:

  1. starts the actual risk-engine server (uvicorn, 1 worker) on a dedicated
     port with an isolated DB_DIR, and waits for /health;
  2. measures cold start (spawn -> readiness), the first request after
     readiness, then warms up;
  3. runs the deterministic probe sequence (identical in every condition, so
     the decision payloads can be compared field-by-field across healthy /
     delayed / unavailable DB-4);
  4. runs the measured segments: two 500-request serial baselines
     (repeatability), a 1/2/4/8/16/32 concurrency ramp, the audit-load
     segment, a single-vs-batch comparison, and a deliberate rate-limit probe;
  5. snapshots the audit backlog (audit.db rows + audit_pending.jsonl),
     fetches the server's own latency SLO view, shuts the server down, and
     verifies the port is free;
  6. writes a machine-readable record to .freebuff/p9/.

Rate-limit pacing (why the harness sleeps):
  `/internal/evaluate` calls `check_access()` (Phase 73), which allows 500
  requests / 60 s sliding window per (client_ip, role) in EVERY PS14_MODE --
  the client_ip is the hardcoded 127.0.0.1. Without pacing, a 500-request
  baseline would trip the limiter mid-run and measure 429s instead of the
  request path. The harness therefore keeps every 60 s window under the
  budget (<= 500 evaluate calls) and sleeps 61 s between segments. The
  limiter itself is measured separately by the final `rate_limit_probe`.

Usage (repo root):

    python backend/scripts/ps14_p9_live_bench.py --profile full
    python backend/scripts/ps14_p9_live_bench.py --profile audit --audit-mode delayed
    python backend/scripts/ps14_p9_live_bench.py --profile audit --audit-mode unavailable
    python backend/scripts/ps14_p9_live_bench.py --profile smoke   # mechanics check ONLY

`--profile smoke` skips the pacing sleeps and is NOT evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import platform
import random
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
BACKEND = ROOT / "backend"
OUT_ROOT = ROOT / ".freebuff" / "p9"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

HOST = "127.0.0.1"
TOKEN = "ps14-p9-bench-token"           # fixture; passed to the server as INTERNAL_TOKEN
FRAUD_ID = "F24BRMMMBJWYMTDW"           # must match ^F[A-Z2-9]{15}$

WINDOW_GAP_S = 61.0                     # > 60 s sliding window of the Phase 73 limiter
EVAL_LIMIT_PER_WINDOW = 500             # Phase 73 /internal/evaluate budget
BATCH_LIMIT_PER_WINDOW = 100            # Phase 73 /internal/evaluate-batch budget

# ---------------------------------------------------------------------------
# Deterministic fixture (all keys valid for FeatureVector, all values inside
# the _RANGE_CHECKS domain so DOMAIN_SHIFT does not fire)
# ---------------------------------------------------------------------------
EVAL_FEATURES = {
    "amount_ratio": 0.95,
    "txn_freq_last_24h": 1,
    "txn_time_unusual": 0,
    "new_device_flag": 0,
    "unusual_location_flag": 0,
    "unusual_recipient_flag": 0,
    "failed_auth_count_24h": 0,
    "days_since_last_similar_txn": 3.0,
    "gradual_escalation_score": 0.0,
    "known_device_count": 3,
    "account_tenure_days": 90.0,
    "hour_of_day": 12,
    "is_weekend": 0,
    "shared_device_accounts": 0,
    "shared_recipient_accounts": 0,
    "mule_ring_score": 0.0,
    "hour_deviation": 0.5,
    "amount_zscore": 0.0,
    "velocity_deviation": 0.0,
    "recipient_novelty": 0.0,
    "txn_regularity": 0.0,
    "account_daily_spend_ratio": 0.2,
    "device_daily_count": 1,
}

# ATO-like vector: exercises reason codes; still inside the domain.
ATO_FEATURES = {
    **EVAL_FEATURES,
    "amount_ratio": 1.1,
    "failed_auth_count_24h": 4,
    "new_device_flag": 1,
    "unusual_location_flag": 1,
    "unusual_recipient_flag": 1,
    "hour_of_day": 2,
}

PROBE_VECTORS = [
    ("probe-01", EVAL_FEATURES), ("probe-02", EVAL_FEATURES), ("probe-03", EVAL_FEATURES),
    ("probe-04", ATO_FEATURES), ("probe-05", ATO_FEATURES), ("probe-06", ATO_FEATURES),
]

# ---------------------------------------------------------------------------
# Deterministic IN-DISTRIBUTION load fixture
#
# A single repeated vector is a degenerate distribution: the PSI drift
# detector (window capacity 500, checked every 100 events) fires at exactly
# request 100 (max_psi ~12.9 vs an alert threshold of 0.25), latches to
# `critical`, and pauses ML (`ML_UNAVAILABLE | DRIFT_MODEL_PAUSED`) for the
# rest of the process lifetime -- measured live on this machine, recorded as
# a Phase 9 finding. To benchmark the NORMAL request path (ML + rules) the
# load segments therefore draw feature vectors from the same table the drift
# baseline was built from (`data/transactions.csv`), shuffled with a fixed
# seed: deterministic, reproducible, and in-distribution.
# Cold start and the cross-condition probes keep the fixed vectors above so
# the probe payloads stay byte-identical across healthy/delayed/unavailable.
# ---------------------------------------------------------------------------
WORKLOAD_CSV = ROOT / "data" / "transactions.csv"
WORKLOAD_SEED = 20261009
_INT_FIELDS = {
    "txn_freq_last_24h", "txn_time_unusual", "new_device_flag", "unusual_location_flag",
    "unusual_recipient_flag", "failed_auth_count_24h", "known_device_count", "hour_of_day",
    "is_weekend", "shared_device_accounts", "shared_recipient_accounts",
}
_FLOAT_FIELDS = {
    "amount_ratio", "days_since_last_similar_txn", "gradual_escalation_score",
    "account_tenure_days", "mule_ring_score", "hour_deviation", "amount_zscore",
    "velocity_deviation", "recipient_novelty", "txn_regularity",
}
_workload: list[dict] | None = None
_workload_meta: dict = {}


def load_workload() -> list[dict]:
    global _workload, _workload_meta
    if _workload is not None:
        return _workload
    rows: list[dict] = []
    with open(WORKLOAD_CSV, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            feat: dict = {}
            for key in _INT_FIELDS:
                val = rec.get(key)
                if val not in (None, ""):
                    feat[key] = int(float(val))
            for key in _FLOAT_FIELDS:
                val = rec.get(key)
                if val not in (None, ""):
                    feat[key] = float(val)
            rows.append(feat)
    order = list(range(len(rows)))
    random.Random(WORKLOAD_SEED).shuffle(order)
    _workload = [rows[i] for i in order]
    _workload_meta = {
        "source": str(WORKLOAD_CSV.relative_to(ROOT)),
        "sha256": hashlib.sha256(WORKLOAD_CSV.read_bytes()).hexdigest(),
        "rows": len(rows),
        "shuffle_seed": WORKLOAD_SEED,
        "fields_taken": sorted(_INT_FIELDS | _FLOAT_FIELDS),
        "index_rule": "(crc32(segment_prefix) + i) % rows",
        "purpose": ("in-distribution deterministic fixture so sustained load does not "
                    "self-trigger the PSI drift gate (see findings)"),
    }
    return _workload


def feat_at(prefix: str, i: int) -> dict:
    wl = load_workload()
    return wl[(zlib.crc32(prefix.encode("utf-8")) + i) % len(wl)]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def pct(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return {"n": 0}

    def p(q: float):
        return round(xs[min(n - 1, int(round(q / 100.0 * (n - 1))))], 3)

    return {
        "n": n,
        "min": round(xs[0], 3),
        "mean": round(sum(xs) / n, 3),
        "p50": p(50),
        "p95": p(95),
        "p99": p(99),
        "max": round(xs[-1], 3),
    }


def tally(items, key) -> dict:
    out: dict[str, int] = {}
    for it in items:
        k = it.get(key) if isinstance(it, dict) else None
        out[str(k)] = out.get(str(k), 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# HTTP client (one persistent connection per thread -- no per-request TCP
# handshake, no ephemeral-port churn at concurrency 32)
# ---------------------------------------------------------------------------
class HttpClient:
    def __init__(self, port: int, token: str = TOKEN, timeout: float = 60.0):
        self.host = HOST
        self.port = port
        self.token = token
        self.timeout = timeout
        self.conn = http.client.HTTPConnection(HOST, port, timeout=timeout)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def _reconnect(self) -> None:
        self.close()
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    # Public alias: segments always start on a fresh connection because uvicorn
    # drops keep-alive connections after `timeout_keep_alive` (default 5 s) and
    # every paced segment is preceded by a 61 s idle gap. Without this, the
    # first request of each segment died with RemoteDisconnected (observed
    # once per paced segment before the reconnect was added).
    reconnect = _reconnect

    def request(self, method: str, path: str, payload=None) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"X-Internal-Token": self.token, "Content-Type": "application/json"}
        t0 = time.perf_counter()
        try:
            self.conn.request(method, path, body=body, headers=headers)
            resp = self.conn.getresponse()
            raw = resp.read()
            status = resp.status
            err = None
        except Exception as exc:  # noqa: BLE001 - transport failure is evidence too
            status = -1
            raw = b""
            err = f"{type(exc).__name__}: {exc}"
            self._reconnect()
        ms = (time.perf_counter() - t0) * 1000.0
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else None
        except Exception:  # noqa: BLE001
            parsed = {"_raw": raw[:200].decode("utf-8", "replace")}
        return {"status": status, "ms": round(ms, 3), "body": parsed, "error": err}


# ---------------------------------------------------------------------------
# Phase 73 rate-limit budget gate
# ---------------------------------------------------------------------------
class Gate:
    """Keeps every 60 s sliding window under the Phase 73 evaluate budget."""

    def __init__(self):
        self.last_eval: float | None = None
        self.last_batch: float | None = None
        self.waits: list[dict] = []
        self.eval_sent = 0          # total /internal/evaluate requests sent
        self.batch_sent = 0
        self._lock = threading.Lock()

    def note(self, kind: str = "eval") -> None:
        with self._lock:
            if kind == "eval":
                self.last_eval = time.monotonic()
                self.eval_sent += 1
            else:
                self.last_batch = time.monotonic()
                self.batch_sent += 1

    def _wait(self, last: float | None, gap: float, label: str) -> float:
        if last is None:
            return 0.0
        need = last + gap - time.monotonic()
        if need > 0:
            self.waits.append({"before": label, "seconds": round(need, 1)})
            time.sleep(need)
            return need
        return 0.0

    def wait_clear(self, label: str) -> float:
        return self._wait(self.last_eval, WINDOW_GAP_S, label)

    def wait_batch_clear(self, label: str) -> float:
        return self._wait(self.last_batch, WINDOW_GAP_S, label)


# ---------------------------------------------------------------------------
# Server process lifecycle
# ---------------------------------------------------------------------------
def netstat_lines() -> str:
    return subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout


def port_listening(port: int) -> list[int]:
    pids = set()
    for line in netstat_lines().splitlines():
        if f":{port}" in line and "LISTENING" in line.upper():
            parts = line.split()
            try:
                pids.add(int(parts[-1]))
            except (ValueError, IndexError):
                continue
    return sorted(pids)


class StartupFailure(RuntimeError):
    pass


class Server:
    def __init__(self, port: int, audit_mode: str, db_dir: Path, log_path: Path):
        self.port = port
        self.audit_mode = audit_mode
        self.db_dir = db_dir
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.cmd: list[str] = []
        self.env_extra: dict[str, str] = {}
        self.spawn_monotonic = 0.0
        self.spawn_utc = ""
        self.ready_s: float | None = None

    def start(self, timeout_s: float = 180.0) -> None:
        if audit_app(self.audit_mode):
            app = "ps14_p9_fault_boot:app"
            app_dir = str(BACKEND / "scripts")
        else:
            app = "src.risk_engine.main:app"
            app_dir = str(BACKEND)
        self.cmd = [str(PY), "-m", "uvicorn", app, "--app-dir", app_dir,
                    "--host", HOST, "--port", str(self.port), "--workers", "1"]
        env = {k: v for k, v in os.environ.items()
               if k not in {"DATABASE_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY",
                            "SUPABASE_SERVICE_KEY", "DB_SCHEMA", "SERVICE_NAME"}}
        env.update({
            "DB_DIR": str(self.db_dir),
            "PS14_MODE": "development",      # .env + preview-stack convention
            "INTERNAL_TOKEN": TOKEN,
            "PYTHONPATH": str(BACKEND),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PS14_P9_AUDIT_MODE": self.audit_mode,
            "PS14_P9_AUDIT_DELAY": "0.25",
        })
        self.env_extra = {k: env[k] for k in
                          ("DB_DIR", "PS14_MODE", "INTERNAL_TOKEN", "PYTHONPATH",
                           "PS14_P9_AUDIT_MODE", "PS14_P9_AUDIT_DELAY")}
        self.db_dir.mkdir(parents=True, exist_ok=True)
        logf = open(self.log_path, "w", encoding="utf-8", buffering=1)
        self.spawn_monotonic = time.monotonic()
        self.spawn_utc = datetime.now(timezone.utc).isoformat()
        self.proc = subprocess.Popen(
            self.cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._log_file = logf

    def log_tail(self, n: int = 25) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except Exception:  # noqa: BLE001
            return "<no log>"

    def wait_ready(self, timeout_s: float = 180.0) -> float:
        """Poll /health until 200. Returns seconds from spawn to readiness."""
        client = HttpClient(self.port, timeout=3.0)
        deadline = time.monotonic() + timeout_s
        last_err = ""
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise StartupFailure(
                    f"server exited with code {self.proc.returncode} before readiness\n"
                    f"--- log tail ---\n{self.log_tail()}"
                )
            try:
                r = client.request("GET", "/health")
                if r["status"] == 200:
                    self.ready_s = round(time.monotonic() - self.spawn_monotonic, 3)
                    client.close()
                    return self.ready_s
                last_err = f"status={r['status']} body={str(r['body'])[:200]}"
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
            time.sleep(0.25)
        client.close()
        raise StartupFailure(
            f"server not ready within {timeout_s}s (last: {last_err})\n"
            f"--- log tail ---\n{self.log_tail()}"
        )

    def stop(self) -> dict:
        """Best-effort graceful shutdown, then force; verify the port is free."""
        out: dict = {"method": None, "exited": None, "port_free": None}
        if self.proc is None:
            return out
        listeners = port_listening(self.port)
        try:
            self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            out["method"] = "CTRL_BREAK"
        except Exception as exc:  # noqa: BLE001
            out["method"] = f"CTRL_BREAK unavailable ({type(exc).__name__})"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None and not port_listening(self.port):
                break
            time.sleep(0.3)
        # Force-kill whatever is still holding the port. NOTE: this runs as a
        # real Windows process (not through a shell), so the flags are the
        # native "/PID" / "/F" forms -- the "//PID" spelling is only needed
        # when typing taskkill inside Git Bash.
        forced = []
        for pid in port_listening(self.port):
            res = subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                                 capture_output=True, text=True)
            forced.append({"pid": pid, "rc": res.returncode,
                           "out": (res.stdout + res.stderr).strip()[:200]})
            out["method"] = f"{out['method']} + taskkill(listener {pid})"
        if self.proc.poll() is None:
            res = subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/F"],
                                 capture_output=True, text=True)
            forced.append({"pid": self.proc.pid, "rc": res.returncode,
                           "out": (res.stdout + res.stderr).strip()[:200]})
            out["method"] = f"{out['method']} + taskkill(spawner)"
        out["taskkill_results"] = forced
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        out["exited"] = self.proc.poll() is not None
        for _ in range(20):
            if not port_listening(self.port):
                break
            time.sleep(0.25)
        out["port_free"] = not port_listening(self.port)
        try:
            self._log_file.close()
        except Exception:  # noqa: BLE001
            pass
        return out


def audit_app(audit_mode: str) -> bool:
    return audit_mode in ("delayed", "unavailable")


def reconcile_server_log(log_path: Path, eval_sent: int, batch_sent: int) -> dict:
    """Cross-check the server's own access log against what this client sent.

    A mismatch means traffic the harness did not generate (a stray client) or
    requests the harness sent that never reached this listener -- either way the
    run's latency numbers would not be attributable to this client alone.
    """
    import re

    out: dict = {"harness_eval_sent": eval_sent, "harness_batch_sent": batch_sent}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return {**out, "error": f"{type(exc).__name__}: {exc}"}
    eval_codes = re.findall(r'POST /internal/evaluate HTTP/1\.1" (\d+)', text)
    batch_codes = re.findall(r'POST /internal/evaluate-batch HTTP/1\.1" (\d+)', text)
    health = len(re.findall(r'GET /health HTTP/1\.1" (\d+)', text))
    from collections import Counter
    out.update({
        "server_logged_eval_total": len(eval_codes),
        "server_logged_eval_by_status": dict(sorted(Counter(eval_codes).items())),
        "server_logged_batch_total": len(batch_codes),
        "server_logged_health": health,
        "eval_match": len(eval_codes) == eval_sent,
        "batch_match": len(batch_codes) == batch_sent,
        "foreign_client_traffic_suspected": len(eval_codes) != eval_sent,
    })
    return out


# ---------------------------------------------------------------------------
# Resource sampling (psutil; listener pid found via netstat)
# ---------------------------------------------------------------------------
def listener_tree(port: int):
    import psutil

    procs = []
    for pid in port_listening(port):
        try:
            p = psutil.Process(pid)
            procs.append(p)
            procs.extend(p.children(recursive=True))
        except Exception:  # noqa: BLE001
            continue
    seen, uniq = set(), []
    for p in procs:
        if p.pid not in seen:
            seen.add(p.pid)
            uniq.append(p)
    return uniq


def cpu_seconds(procs) -> float:
    total = 0.0
    for p in procs:
        try:
            ct = p.cpu_times()
            total += ct.user + ct.system
        except Exception:  # noqa: BLE001
            continue
    return total


class ResSampler:
    def __init__(self, port: int):
        self.port = port
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.peak_rss = 0
        self.samples = 0

    def _loop(self) -> None:
        while not self.stop_flag.is_set():
            rss = 0
            for p in listener_tree(self.port):
                try:
                    rss += p.memory_info().rss
                except Exception:  # noqa: BLE001
                    continue
            self.peak_rss = max(self.peak_rss, rss)
            self.samples += 1
            self.stop_flag.wait(0.5)

    def start(self) -> None:
        import psutil
        self.stop_flag.clear()
        self.peak_rss = 0
        self.samples = 0
        self._procs = listener_tree(self.port)
        self._cpu0 = cpu_seconds(self._procs)
        self._t0 = time.monotonic()
        psutil.cpu_percent(None)  # prime
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        import psutil
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=3)
        wall = time.monotonic() - self._t0
        procs = listener_tree(self.port)
        cpu1 = cpu_seconds(procs)
        try:
            sys_cpu = psutil.cpu_percent(None)
        except Exception:  # noqa: BLE001
            sys_cpu = None
        threads = 0
        for p in procs:
            try:
                threads += p.num_threads()
            except Exception:  # noqa: BLE001
                pass
        rss_end = 0
        for p in procs:
            try:
                rss_end += p.memory_info().rss
            except Exception:  # noqa: BLE001
                pass
        core_s = max(0.0, cpu1 - self._cpu0)
        return {
            "wall_s": round(wall, 3),
            "cpu_core_seconds": round(core_s, 3),
            "avg_cores_busy": round(core_s / wall, 3) if wall > 0 else None,
            "system_cpu_pct_during": sys_cpu,
            "rss_peak_bytes": self.peak_rss,
            "rss_end_bytes": rss_end,
            "listener_processes": [p.pid for p in procs],
            "threads_total": threads,
            "samples": self.samples,
        }


# ---------------------------------------------------------------------------
# Audit backlog observation (file-level: audit.db rows + pending store)
# ---------------------------------------------------------------------------
def audit_snapshot(db_dir: Path) -> dict:
    snap: dict = {"db_dir": str(db_dir)}
    db = db_dir / "audit.db"
    if db.exists():
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            cur = con.execute("SELECT COUNT(*), COALESCE(MAX(seq), 0) FROM audit_events")
            rows, max_seq = cur.fetchone()
            con.close()
            snap["audit_events_rows"] = rows
            snap["audit_events_max_seq"] = max_seq
        except Exception as exc:  # noqa: BLE001
            snap["audit_events_error"] = f"{type(exc).__name__}: {exc}"
    else:
        snap["audit_events_rows"] = None
    pending = db_dir / "audit_pending.jsonl"
    if pending.exists():
        statuses: dict[str, int] = {}
        n = 0
        for line in pending.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                st = str(json.loads(line).get("status"))
            except Exception:  # noqa: BLE001
                st = "<unparseable>"
            statuses[st] = statuses.get(st, 0) + 1
        snap["pending_lines"] = n
        snap["pending_statuses"] = dict(sorted(statuses.items()))
    else:
        snap["pending_lines"] = 0
        snap["pending_statuses"] = {}
    risk_db = db_dir / "risk.db"
    if risk_db.exists():
        try:
            con = sqlite3.connect(f"file:{risk_db}?mode=ro", uri=True, timeout=5)
            snap["risk_scores_rows"] = con.execute(
                "SELECT COUNT(*) FROM risk_scores").fetchone()[0]
            con.close()
        except Exception as exc:  # noqa: BLE001
            snap["risk_scores_error"] = f"{type(exc).__name__}: {exc}"
    return snap


# ---------------------------------------------------------------------------
# Segment runners
# ---------------------------------------------------------------------------
def new_event_id(prefix: str, runid: str, i: int) -> str:
    return f"{prefix}-{runid}-{i:05d}"


def run_seq(client: HttpClient, gate: Gate, specs: list[tuple[str, dict]],
            name: str, sampler: ResSampler | None = None,
            keep_bodies: bool = False, keep_latencies: bool = False) -> dict:
    """specs: [(event_id, features), ...] sent serially."""
    client.reconnect()  # fresh connection (see HttpClient.reconnect)
    lat, bodies, errors = [], [], []
    status_counts: dict[str, int] = {}
    if sampler:
        sampler.start()
    wall0 = time.perf_counter()
    for event_id, feats in specs:
        r = client.request("POST", "/internal/evaluate", {
            "event_id": event_id, "fraud_id": FRAUD_ID, "features": feats,
        })
        gate.note("eval")
        key = str(r["status"])
        status_counts[key] = status_counts.get(key, 0) + 1
        if r["status"] == 200:
            lat.append(r["ms"])
            bodies.append(r["body"] or {})
        else:
            errors.append({"event_id": event_id, "status": r["status"],
                           "body": r["body"], "error": r["error"]})
    wall = time.perf_counter() - wall0
    out = {
        "name": name,
        "requests": len(specs),
        "success": len(lat),
        "errors": len(errors),
        "status_counts": dict(sorted(status_counts.items())),
        "error_samples": errors[:3],
        "wall_s": round(wall, 3),
        "rps": round(len(specs) / wall, 1) if wall > 0 else None,
        "latency_ms": pct(lat),
    }
    if bodies:
        out["decision_counts"] = tally(bodies, "decision")
        out["degraded_count"] = sum(1 for b in bodies if b.get("degraded"))
        out["risk_band_counts"] = tally(bodies, "risk_band")
        out["drift_state_counts"] = tally(bodies, "drift_state")
    if keep_bodies:
        if len(lat) == len(specs) == len(bodies):
            out["bodies"] = [
                {"event_id": ev, "latency_ms": ms, "body": body}
                for (ev, _), ms, body in zip(specs, lat, bodies)
            ]
        else:
            out["bodies_unavailable"] = (
                f"alignment mismatch: specs={len(specs)} lat={len(lat)} "
                f"bodies={len(bodies)}"
            )
    if keep_latencies:
        out["latencies_ms"] = [round(x, 3) for x in lat]
    if sampler:
        out["server_resources"] = sampler.stop()
    return out


def wspecs(prefix: str, runid: str, n: int) -> list[tuple[str, dict]]:
    """Deterministic in-distribution specs: unique event ids + workload rows."""
    return [(new_event_id(prefix, runid, i), feat_at(prefix, i)) for i in range(n)]


def run_concurrent(port: int, gate: Gate, workers: int, total: int,
                   prefix: str, runid: str, feat_fn,
                   sampler: ResSampler | None = None) -> dict:
    per = total // workers
    rem = total - per * workers
    barrier = threading.Barrier(workers)
    slots: list[dict] = [{} for _ in range(workers)]

    def worker(k: int) -> None:
        client = HttpClient(port)
        start = k * per
        end = start + per + (rem if k == workers - 1 else 0)
        lat, errors, statuses, bodies = [], [], {}, []
        barrier.wait()
        for i in range(start, end):
            r = client.request("POST", "/internal/evaluate", {
                "event_id": new_event_id(prefix, runid, i),
                "fraud_id": FRAUD_ID, "features": feat_fn(i),
            })
            gate.note("eval")
            key = str(r["status"])
            statuses[key] = statuses.get(key, 0) + 1
            if r["status"] == 200:
                lat.append(r["ms"])
                bodies.append(r["body"] or {})
            else:
                errors.append({"i": i, "status": r["status"], "body": r["body"],
                               "error": r["error"]})
        client.close()
        slots[k] = {"lat": lat, "errors": errors, "statuses": statuses, "bodies": bodies}

    if sampler:
        sampler.start()
    wall0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    wall = time.perf_counter() - wall0

    lat, errors, bodies = [], [], []
    status_counts: dict[str, int] = {}
    finished = 0
    for s in slots:
        if not s:
            continue
        finished += 1
        lat.extend(s["lat"])
        errors.extend(s["errors"])
        bodies.extend(s["bodies"])
        for k, v in s["statuses"].items():
            status_counts[k] = status_counts.get(k, 0) + v
    out = {
        "name": f"concurrency_{workers}",
        "concurrency": workers,
        "threads_finished": finished,
        "requests": total,
        "success": len(lat),
        "errors": len(errors),
        "status_counts": dict(sorted(status_counts.items())),
        "error_samples": errors[:3],
        "wall_s": round(wall, 3),
        "rps": round(len(lat) / wall, 1) if wall > 0 else None,
        "latency_ms": pct(lat),
    }
    if bodies:
        out["decision_counts"] = tally(bodies, "decision")
        out["degraded_count"] = sum(1 for b in bodies if b.get("degraded"))
        out["drift_state_counts"] = tally(bodies, "drift_state")
    if sampler:
        out["server_resources"] = sampler.stop()
    return out


def run_batch(client: HttpClient, gate: Gate, sizes: list[int], runid: str,
              name: str = "batch") -> dict:
    gate.wait_batch_clear(name)
    client.reconnect()
    out_sizes = []
    for size in sizes:
        payloads = [
            {"event_id": f"batch{size}-{runid}-{i:05d}", "fraud_id": FRAUD_ID,
             "features": feat_at(f"batch{size}", i)}
            for i in range(size)
        ]
        r = client.request("POST", "/internal/evaluate-batch", payloads)
        gate.note("batch")
        body = r["body"] if isinstance(r["body"], dict) else {}
        results = body.get("results") if isinstance(body.get("results"), list) else []
        ok_items = sum(1 for x in results if isinstance(x, dict) and x.get("event_id"))
        out_sizes.append({
            "size": size,
            "status": r["status"],
            "total_ms": r["ms"],
            "per_item_ms": round(r["ms"] / size, 3) if size else None,
            "response_count": body.get("count"),
            "items_returned": len(results),
            "items_valid": ok_items,
            "items_failed": size - ok_items if r["status"] == 200 else size,
            "error": r["error"],
            "decision_counts": tally([x for x in results if isinstance(x, dict)], "decision")
            if results else {},
        })
    return {"name": name, "sizes": out_sizes}


def fetch_json(client: HttpClient, path: str) -> dict:
    client.reconnect()  # segment gaps can exceed uvicorn's 5 s keep-alive timeout
    r = client.request("GET", path)
    return {"path": path, "status": r["status"], "body": r["body"], "ms": r["ms"]}


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------
def git_sha() -> dict:
    def rev(expr: str) -> str:
        try:
            return subprocess.run(["git", "rev-parse", expr], cwd=str(ROOT),
                                  capture_output=True, text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return "<unavailable>"
    def dirty() -> str:
        try:
            return subprocess.run(["git", "status", "--short"], cwd=str(ROOT),
                                  capture_output=True, text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return "<unavailable>"
    return {"head": rev("HEAD"), "origin_main": rev("origin/main"), "status_short": dirty()}


def env_record(server: Server, port: int) -> dict:
    rec = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "logical_cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        rec["memory_total_bytes"] = vm.total
        rec["memory_available_bytes"] = vm.available
        rec["system_cpu_count"] = psutil.cpu_count()
        rec["system_cpu_percent_before"] = psutil.cpu_percent(None)
    except Exception:  # noqa: BLE001
        pass
    try:
        import uvicorn
        rec["uvicorn"] = getattr(uvicorn, "__version__", "<unknown>")
    except Exception:  # noqa: BLE001
        pass
    rec["server_command"] = " ".join(server.cmd)
    rec["server_workers"] = 1
    rec["server_env"] = server.env_extra
    rec["port"] = port
    rec["client_transport"] = "http.client.HTTPConnection, keep-alive, 1 conn/thread"
    rec["git"] = git_sha()
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_ml_cycles(args, record: dict, out_dir: Path, db_dir: Path, runid: str,
                   stamp: str, gate: Gate, sampler: ResSampler) -> int:
    """Profile `ml`: fresh-process cycles for the ML-path baseline.

    The PSI drift gate gives a process only 99 pre-flip requests (see the
    fixture's `drift_gate` block), and the Phase 73 rate limiter is
    process-local too -- so each cycle starts a fresh server, measures
    `--ml-per-cycle` requests on the ML path, records the drift state, and
    stops. Repeating the cycle multiplies the ML-path sample without ever
    crossing event 100 inside a process.
    """
    if args.ml_per_cycle >= 100:
        print("--ml-per-cycle must stay under 100 (the drift gate flips at event 100)",
              flush=True)
        return 2
    print(f"== profile ml: {args.ml_cycles} cycles x {args.ml_per_cycle} requests ==",
          flush=True)
    cycles: list[dict] = []
    srv_ref: Server | None = None
    ready_times: list[float] = []
    for c in range(args.ml_cycles):
        log_c = out_dir / f"server_ml_c{c}_{stamp}.log"
        srv = Server(args.port, "healthy", db_dir, log_c)
        srv_ref = srv
        srv.start()
        try:
            ready_s = srv.wait_ready()
        except StartupFailure as exc:
            record["startup"] = {"ok": False, "cycle": c, "error": str(exc)}
            record["shutdown"] = srv.stop()
            break
        ready_times.append(ready_s)
        client = HttpClient(args.port)
        specs = wspecs(f"ml{c}", runid, args.ml_per_cycle)
        seg = run_seq(client, gate, specs, f"ml_cycle_{c}", sampler,
                      keep_latencies=True)
        seg["cycle"] = c
        seg["spawn_to_ready_s"] = ready_s
        drift = client.request("GET", "/internal/drift-status")
        seg["drift_status"] = drift["body"]
        client.close()
        seg["shutdown"] = srv.stop()
        cycles.append(seg)
        # The Phase 73 limiter is process-local: a fresh process resets its budget.
        gate.last_eval = None
        lat = seg["latency_ms"]
        print(f"  cycle {c}: ready={ready_s:.2f}s ok={seg['success']} err={seg['errors']} "
              f"degraded={seg.get('degraded_count')} p50={lat.get('p50')} "
              f"p95={lat.get('p95')} p99={lat.get('p99')} max={lat.get('max')}ms",
              flush=True)
    record["ml_cycles"] = cycles
    pooled: list[float] = [x for seg in cycles for x in seg.get("latencies_ms", [])]
    total_req = sum(seg["requests"] for seg in cycles)
    total_ok = sum(seg["success"] for seg in cycles)
    total_err = sum(seg["errors"] for seg in cycles)
    total_degraded = sum(seg.get("degraded_count") or 0 for seg in cycles)
    record["ml_baseline"] = {
        "cycles_completed": len(cycles),
        "requests": total_req,
        "success": total_ok,
        "errors": total_err,
        "degraded_count": total_degraded,
        "drift_states": sorted({s for seg in cycles
                                for s in (seg.get("drift_state_counts") or {})}),
        "spawn_to_ready_s": {
            "min": round(min(ready_times), 3) if ready_times else None,
            "max": round(max(ready_times), 3) if ready_times else None,
        },
        "pooled_latency_ms": pct(pooled),
        "interpretation": (
            "ML-path request latency (drift state normal, ML active): every "
            "request measured before a process reaches event 100, pooled over "
            "fresh processes"
        ),
    }
    if srv_ref is not None:
        record["environment"] = env_record(srv_ref, args.port)
    print(f"  POOLED n={len(pooled)} p50={record['ml_baseline']['pooled_latency_ms'].get('p50')} "
          f"p95={record['ml_baseline']['pooled_latency_ms'].get('p95')} "
          f"p99={record['ml_baseline']['pooled_latency_ms'].get('p99')} "
          f"degraded_total={total_degraded}", flush=True)
    return _finish(record, out_dir, srv_ref, None, gate)


def main() -> int:
    ap = argparse.ArgumentParser(description="PS-14 Phase 9 live request-path benchmark")
    ap.add_argument("--profile", choices=["full", "audit", "smoke", "ml"],
                    default="full")
    ap.add_argument("--audit-mode", choices=["healthy", "delayed", "unavailable"],
                    default="healthy")
    ap.add_argument("--ml-cycles", type=int, default=5,
                    help="fresh-process cycles for the ML-path baseline (profile=ml)")
    ap.add_argument("--ml-per-cycle", type=int, default=98,
                    help="requests per ML cycle; must stay under event 100 (drift gate)")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    runid = uuid.uuid4().hex[:6]
    load_workload()  # deterministic in-distribution fixture (see module docstring)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%z")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_dir = ROOT / "db" / f"p9_{args.audit_mode}_{runid}"
    log_path = out_dir / f"server_{args.audit_mode}_{stamp}.log"

    print(f"== PS-14 Phase 9 live benchmark ==", flush=True)
    print(f"  profile={args.profile} audit_mode={args.audit_mode} port={args.port}", flush=True)
    print(f"  db_dir={db_dir}", flush=True)
    if args.profile == "smoke":
        print("  *** SMOKE RUN -- pacing sleeps skipped -- NOT EVIDENCE ***", flush=True)

    gate = Gate()
    sampler = ResSampler(args.port)
    server = Server(args.port, args.audit_mode, db_dir, log_path)

    record: dict = {
        "phase": "9",
        "runid": runid,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "audit_mode": args.audit_mode,
        "evidence_class": ("SMOKE (not evidence)" if args.profile == "smoke"
                           else "DEMONSTRATED live HTTP"
                           + ("" if args.audit_mode == "healthy"
                              else " with SIMULATED DB-4 fault")),
        "fixture": {
            "endpoint": f"http://{HOST}:{args.port}/internal/evaluate",
            "internal_token": TOKEN,
            "fraud_id": FRAUD_ID,
            "fraud_id_regex_ok": True,
            "workload": _workload_meta,
            "drift_gate": {
                "detector": "backend/src/risk_engine/drift_detector.py",
                "baseline": "models/data/drift_baseline.json",
                "window_size": 500,
                "check_interval": 100,
                "thresholds": {"warn": 0.1, "alert": 0.25},
                "observed_flip_event": 100,
                "observed_max_psi": 12.88,
                "effect": ("state->critical, reason codes ML_UNAVAILABLE | "
                           "DRIFT_MODEL_PAUSED, ML paused for the rest of the "
                           "process (rules-only, score floor 31)"),
                "why": ("the baseline was built from a 200,000-event dataset "
                        "(2012 dates) that is NOT the serving table "
                        "data/transactions.csv (9,799 rows) the deployed model "
                        "was trained on -- PSI explodes on ANY traffic"),
                "reproduced_by": ".freebuff/p9/diagnose_degraded.py + verify_workload.py",
                "segment_budget": ("events 1..99 are the ML path; segments after "
                                   "event 100 are the post-flip steady state"),
            },
            "feature_count": len(EVAL_FEATURES),
            "feature_keys": sorted(EVAL_FEATURES),
            "fixed_vectors_used_for": ["cold start", "6 cross-condition probes"],
            "eval_vector": EVAL_FEATURES,
            "ato_vector": ATO_FEATURES,
            "probe_sequence": [pid for pid, _ in PROBE_VECTORS],
            "rate_limit_pacing": {
                "window_s": 60,
                "evaluate_budget_per_window": EVAL_LIMIT_PER_WINDOW,
                "inter_segment_gap_s": WINDOW_GAP_S,
                "source": "backend/src/monitoring/access_control.py check_access()",
            },
        },
        "segments": {},
    }

    if args.profile == "ml":
        return _run_ml_cycles(args, record, out_dir, db_dir, runid, stamp, gate, sampler)

    try:
        # -- 1. start + readiness -------------------------------------------
        server.start()
        try:
            ready_s = server.wait_ready()
        except StartupFailure as exc:
            record["startup"] = {"ok": False, "error": str(exc)}
            record["shutdown"] = server.stop()
            record["environment"] = env_record(server, args.port)
            _finish(record, out_dir, server, None, gate)
            print(f"STARTUP FAILED\n{exc}", flush=True)
            return 2
        record["startup"] = {
            "ok": True,
            "spawn_utc": server.spawn_utc,
            "spawn_to_ready_s": ready_s,
            "log": str(log_path),
        }
        print(f"  ready in {ready_s:.2f}s", flush=True)

        client = HttpClient(args.port)

        # -- 2. cold first request -----------------------------------------
        sampler.start()
        r = client.request("POST", "/internal/evaluate", {
            "event_id": "coldstart-0001", "fraud_id": FRAUD_ID, "features": EVAL_FEATURES,
        })
        gate.note("eval")
        cold_res = sampler.stop()
        record["cold_start"] = {
            "spawn_to_ready_s": ready_s,
            "first_request": {"status": r["status"], "latency_ms": r["ms"],
                              "body": r["body"], "error": r["error"]},
            "server_resources_during_first_request": cold_res,
        }
        print(f"  first request: {r['status']} {r['ms']:.1f}ms", flush=True)

        # -- 3. warm-up (in-distribution workload) ---------------------------
        # 20, not 40: the PSI drift gate flips at event 100 (check_interval),
        # so cold(1) + warmup(20) + probes(6) + audit_load(60) + ml_tail(12)
        # = 99 keeps EVERY pre-flip segment on the ML path in every condition.
        warm_specs = wspecs("warmup", runid, 20)
        warm = run_seq(client, gate, warm_specs, "warmup_20")
        record["segments"]["warmup_20"] = warm
        print(f"  warmup: ok={warm['success']} err={warm['errors']} "
              f"p50={warm['latency_ms'].get('p50')}ms", flush=True)

        # -- 4. probes (identical sequence in every condition; full response
        #       payloads kept for field-by-field healthy vs fault comparison) --
        probe_specs = [(pid, feats) for pid, feats in PROBE_VECTORS]
        probes = run_seq(client, gate, probe_specs, "probes_6", keep_bodies=True)
        probes["note"] = (
            "sent once, in this order, in EVERY condition (healthy/delayed/unavailable) "
            "after the same cold + warmup sequence; bodies are the decision payloads "
            "compared field-by-field in the closeout"
        )
        record["segments"]["probes_6"] = probes
        record["probe_payloads"] = probes.get("bodies", [])
        print(f"  probes: ok={probes['success']} err={probes['errors']}", flush=True)

        if args.profile == "smoke":
            b = run_batch(client, gate, [5], runid, "smoke_batch")
            record["segments"]["smoke_batch"] = b
            record["segments"]["smoke_eval"] = run_seq(
                client, gate, [(f"smoke-{runid}-{i:03d}", EVAL_FEATURES) for i in range(3)],
                "smoke_eval")
            record["server_side"] = [fetch_json(client, "/internal/latency-slo")]
            record["audit_backlog"] = {"pre_shutdown": audit_snapshot(db_dir)}
            client.close()
            record["shutdown"] = server.stop()
            record["audit_backlog"]["post_shutdown"] = audit_snapshot(db_dir)
            record["environment"] = env_record(server, args.port)
            return _finish(record, out_dir, server, None, gate)

        # -- 5a. audit-load segment (identical sequence in every condition) --
        if args.profile == "audit" or args.profile == "full":
            waited = gate.wait_clear("audit_load_60")
            audit_specs = wspecs("auditload", runid, 60)
            seg = run_seq(client, gate, audit_specs, "audit_load_60", sampler)
            seg["pacing_wait_s"] = round(waited, 1)
            record["segments"]["audit_load_60"] = seg
            print(f"  audit_load_60: ok={seg['success']} err={seg['errors']} "
                  f"p50={seg['latency_ms'].get('p50')} p95={seg['latency_ms'].get('p95')}ms",
                  flush=True)

        if args.profile == "audit":
            # let the background audit queue drain before snapshotting
            time.sleep(5)
            record["audit_backlog"] = {"pre_shutdown": audit_snapshot(db_dir)}
            record["server_side"] = [
                fetch_json(client, "/internal/latency-slo"),
                fetch_json(client, "/internal/metrics"),
            ]
            client.close()
            record["shutdown"] = server.stop()
            record["audit_backlog"]["post_shutdown"] = audit_snapshot(db_dir)
            record["environment"] = env_record(server, args.port)
            record["rate_limit_pacing_log"] = gate.waits
            return _finish(record, out_dir, server, None, gate)

        # -- 5b. serial baselines (repeatability) ---------------------------
        # The PSI drift gate flips at event 100 (check_interval=100, window=500):
        # cold(1) + warmup(20) + probes(6) + audit_load(60) + ml_tail(12) = 99,
        # so everything from here on is measured in the system's post-flip
        # steady state (ML paused -> rules-only), which is what this checkout
        # actually serves after 100 evaluations. Segments self-describe via
        # their `drift_state_counts` / `degraded_count` fields.
        ml_tail = run_seq(client, gate, wspecs("mltail", runid, 12), "ml_tail_12")
        record["segments"]["ml_tail_12"] = ml_tail
        print(f"  ml_tail_12: ok={ml_tail['success']} err={ml_tail['errors']} "
              f"degraded={ml_tail.get('degraded_count')} "
              f"drift={ml_tail.get('drift_state_counts')}", flush=True)

        # NOTE: each segment gets its OWN event-id prefix. Reusing a prefix
        # would make the second run hit the idempotent-replay path (event_id
        # already scored) and measure the stored-result fast path instead of
        # the full request path -- caught in the first full run, where the
        # repeated segment reported p50 3.4 ms against 29.7 ms for the real one.
        for label, prefix, n in (("single_500_run1", "mono1", 500),
                                 ("single_500_run2", "mono2", 500)):
            waited = gate.wait_clear(label)
            specs = wspecs(prefix, runid, n)
            seg = run_seq(client, gate, specs, label, sampler)
            seg["pacing_wait_s"] = round(waited, 1)
            record["segments"][label] = seg
            print(f"  {label}: ok={seg['success']} err={seg['errors']} rps={seg['rps']} "
                  f"p50={seg['latency_ms'].get('p50')} p95={seg['latency_ms'].get('p95')} "
                  f"p99={seg['latency_ms'].get('p99')} max={seg['latency_ms'].get('max')}ms",
                  flush=True)

        # -- 5c. concurrency ramp ------------------------------------------
        ramp = {}
        for workers, total in ((1, 150), (2, 200), (4, 300), (8, 300), (16, 300), (32, 300)):
            waited = gate.wait_clear(f"concurrency_{workers}")
            seg = run_concurrent(args.port, gate, workers, total, f"ramp{workers}",
                                 runid,
                                 (lambda i, p=f"ramp{workers}": feat_at(p, i)),
                                 sampler)
            seg["pacing_wait_s"] = round(waited, 1)
            ramp[str(workers)] = seg
            record["segments"][seg["name"]] = seg
            print(f"  c={workers}: ok={seg['success']} err={seg['errors']} rps={seg['rps']} "
                  f"p50={seg['latency_ms'].get('p50')} p95={seg['latency_ms'].get('p95')} "
                  f"p99={seg['latency_ms'].get('p99')} max={seg['latency_ms'].get('max')}ms",
                  flush=True)
        record["concurrency_ramp"] = ramp

        # -- 5d. single vs batch -------------------------------------------
        waited = gate.wait_clear("mono20_vs_batch")
        mono20 = run_seq(client, gate, wspecs("mono20", runid, 20),
                         "single_20_for_batch_compare")
        mono20["pacing_wait_s"] = round(waited, 1)
        record["segments"]["single_20_for_batch_compare"] = mono20
        batch = run_batch(client, gate, [10, 50, 100, 500], runid)
        record["batch"] = batch
        print(f"  batch: " + " ".join(
            f"[{s['size']}] {s['total_ms']:.0f}ms({s['per_item_ms']:.1f}/item,"
            f"ok={s['items_valid']})" for s in batch["sizes"]), flush=True)

        # -- 5e. server-side view + backlog ---------------------------------
        record["server_side"] = [
            fetch_json(client, "/internal/latency-slo"),
            fetch_json(client, "/internal/metrics"),
        ]
        record["audit_backlog"] = {"pre_shutdown": audit_snapshot(db_dir)}

        # -- 5f. deliberate rate-limit probe (LAST: it blocks the client) ---
        waited = gate.wait_clear("rate_limit_probe")
        probe = run_seq(client, gate,
                        wspecs("rlprobe", runid, EVAL_LIMIT_PER_WINDOW + 5),
                        "rate_limit_probe_505")
        probe["pacing_wait_s"] = round(waited, 1)
        first_429 = next((e["event_id"] for e in probe.get("error_samples", [])
                          if e["status"] == 429), None)
        probe["first_429_event_id"] = first_429
        probe["note"] = ("deliberate over-budget burst: measures the Phase 73 limiter, "
                         "not request-path latency; expected ~500x200 then 429 + 30s block")
        record["segments"]["rate_limit_probe"] = probe
        rl_status = probe["status_counts"]
        print(f"  rate_limit_probe: {rl_status} (expected ~500 x 200, then 429)", flush=True)

        client.close()
        record["shutdown"] = server.stop()
        record["audit_backlog"]["post_shutdown"] = audit_snapshot(db_dir)
        record["environment"] = env_record(server, args.port)
        record["rate_limit_pacing_log"] = gate.waits
        return _finish(record, out_dir, server, None, gate)

    except Exception as exc:  # noqa: BLE001
        record["fatal_error"] = f"{type(exc).__name__}: {exc}"
        record["server_log_tail"] = server.log_tail()
        _finish(record, out_dir, server, None, gate)
        print(f"FATAL: {type(exc).__name__}: {exc}", flush=True)
        print(server.log_tail(), flush=True)
        return 1


def _finish(record: dict, out_dir: Path, server: Server, _unused, gate: Gate) -> int:
    record["integrity"] = {
        "server_log_reconciliation": reconcile_server_log(
            server.log_path, gate.eval_sent, gate.batch_sent
        )
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%z")
    out = out_dir / f"ps14_p9_{record.get('audit_mode','x')}_{record.get('profile','x')}_{ts}.json"
    try:
        out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"could not write record: {exc}", flush=True)
        return 1
    print(f"\n== record written: {out} ==", flush=True)
    record["_record_path"] = str(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
