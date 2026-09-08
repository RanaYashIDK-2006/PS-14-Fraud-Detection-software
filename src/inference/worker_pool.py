"""Auto-scaling worker pool manager for inference workers.

Monitors Redis load metrics and dynamically starts/stops uvicorn workers
to maintain target latency and throughput.

Architecture:
  WorkerPoolManager
    ├─ RedisMetricsCollector (reads Redis stats every N seconds)
    ├─ AutoScaler (decides scale-up/down based on thresholds)
    └─ ProcessManager (starts/stops uvicorn workers)

Scaling triggers:
  - Scale UP:   avg_latency > target_latency OR error_rate > error_threshold
  - Scale DOWN: avg_latency < target_latency * 0.5 AND workers > min_workers
  - Cooldown:   60s between scaling events (prevents thrashing)

Worker lifecycle:
  - Start: uvicorn --port {port} --workers 1 (detached)
  - Health: HTTP GET /health every 10s
  - Stop: SIGTERM → 10s grace → SIGKILL

Usage:
    pool = WorkerPoolManager(min_workers=2, max_workers=8)
    pool.start()  # begins monitoring and auto-scaling
    ...
    pool.stop()   # graceful shutdown

    # Or as a CLI tool:
    # python -m src.inference.worker_pool --min 2 --max 8 --target-latency 50
"""
from __future__ import annotations

import os
import time
import json
import signal
import socket
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import redis


ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class WorkerInfo:
    """Information about a single inference worker."""
    port: int
    pid: int
    started_at: float
    last_health_check: float = 0.0
    healthy: bool = True
    request_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0

    @property
    def uptime(self) -> float:
        return time.time() - self.started_at

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def to_dict(self) -> dict:
        return {
            "port": self.port,
            "pid": self.pid,
            "uptime_s": round(self.uptime, 1),
            "healthy": self.healthy,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class RedisMetricsCollector:
    """Collects load metrics from Redis for auto-scaling decisions.

    Metrics tracked:
      - Active users (from sliding window keys)
      - Request rate (from rate limiter ZSETs)
      - Redis memory usage
      - Connected clients
      - Evicted keys (pressure indicator)
    """

    def __init__(self, r: redis.Redis, prefix: str = "rl:"):
        self.r = r
        self.prefix = prefix
        self._history: list[dict] = []
        self._max_history = 60  # keep 60 data points (10 min at 10s intervals)

    def collect(self) -> dict:
        """Collect current metrics from Redis."""
        metrics = {
            "timestamp": time.time(),
            "active_users": 0,
            "request_rate_1m": 0,
            "redis_memory_mb": 0.0,
            "connected_clients": 0,
            "evicted_keys": 0,
            "total_commands": 0,
            "keyspace_hits": 0,
            "keyspace_misses": 0,
        }

        try:
            # Count active sliding window keys
            cursor = 0
            while True:
                cursor, keys = self.r.scan(cursor, match="fw:window:*", count=1000)
                metrics["active_users"] += len(keys)
                if cursor == 0:
                    break

            # Count requests in last minute from global rate limiter
            global_key = f"{self.prefix}global"
            if self.r.exists(global_key):
                one_min_ago = time.time() - 60
                metrics["request_rate_1m"] = self.r.zcount(
                    global_key, one_min_ago, "+inf"
                )

            # Redis info
            info = self.r.info("memory")
            metrics["redis_memory_mb"] = round(
                info.get("used_memory", 0) / (1024 * 1024), 2
            )

            clients = self.r.info("clients")
            metrics["connected_clients"] = clients.get("connected_clients", 0)

            stats = self.r.info("stats")
            metrics["evicted_keys"] = stats.get("evicted_keys", 0)
            metrics["total_commands"] = stats.get("total_commands_processed", 0)

            keyspace = self.r.info("keyspace")
            for db_info in keyspace.values():
                if isinstance(db_info, dict):
                    metrics["keyspace_hits"] += db_info.get("hits", 0)
                    metrics["keyspace_misses"] += db_info.get("misses", 0)

        except Exception as e:
            metrics["error"] = str(e)

        self._history.append(metrics)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return metrics

    def get_trend(self, metric: str, window: int = 6) -> dict:
        """Get trend for a metric over the last N data points."""
        if len(self._history) < 2:
            return {"trend": "insufficient_data", "current": 0, "avg": 0, "max": 0}

        recent = [h.get(metric, 0) for h in self._history[-window:]]
        current = recent[-1]
        avg = sum(recent) / len(recent)
        maximum = max(recent)

        if len(recent) >= 2:
            slope = (recent[-1] - recent[0]) / len(recent)
            if slope > avg * 0.1:
                trend = "increasing"
            elif slope < -avg * 0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {"trend": trend, "current": current, "avg": round(avg, 2), "max": maximum}

    @property
    def history(self) -> list[dict]:
        return list(self._history)


class AutoScaler:
    """Decides when to scale workers up or down based on metrics.

    Scaling logic:
      - Scale UP if:
        1. avg_latency > target_latency (sustained for cooldown_period)
        2. OR error_rate > error_threshold
        3. AND current_workers < max_workers
        4. AND cooldown has elapsed

      - Scale DOWN if:
        1. avg_latency < target_latency * 0.5
        2. AND current_workers > min_workers
        3. AND cooldown has elapsed
    """

    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 8,
        target_latency_ms: float = 50.0,
        error_threshold: float = 0.05,
        cooldown_seconds: float = 60.0,
        scale_up_consecutive: int = 3,
        scale_down_consecutive: int = 5,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_latency_ms = target_latency_ms
        self.error_threshold = error_threshold
        self.cooldown_seconds = cooldown_seconds
        self.scale_up_consecutive = scale_up_consecutive
        self.scale_down_consecutive = scale_down_consecutive

        self._last_scale_time = 0.0
        self._high_latency_count = 0
        self._low_latency_count = 0
        self._last_action = "none"

    def evaluate(
        self,
        metrics: dict,
        current_workers: int,
        worker_latencies: list[float],
        worker_errors: list[int],
        worker_requests: list[int],
    ) -> dict:
        """Evaluate whether to scale up, down, or hold.

        Returns dict with action, reason, and target worker count.
        """
        now = time.time()
        cooldown_ok = (now - self._last_scale_time) >= self.cooldown_seconds

        # Aggregate worker metrics
        total_requests = sum(worker_requests) if worker_requests else 0
        total_errors = sum(worker_errors) if worker_errors else 0
        error_rate = total_errors / max(total_requests, 1)

        avg_latency = (
            sum(worker_latencies) / len(worker_latencies)
            if worker_latencies
            else 0.0
        )

        # Check scaling conditions
        high_latency = avg_latency > self.target_latency_ms
        low_latency = avg_latency < self.target_latency_ms * 0.5
        high_errors = error_rate > self.error_threshold

        if high_latency or high_errors:
            self._high_latency_count += 1
            self._low_latency_count = 0
        elif low_latency:
            self._low_latency_count += 1
            self._high_latency_count = 0
        else:
            self._high_latency_count = max(0, self._high_latency_count - 1)
            self._low_latency_count = max(0, self._low_latency_count - 1)

        # Decision
        action = "hold"
        target = current_workers
        reason = "within thresholds"

        if (
            self._high_latency_count >= self.scale_up_consecutive
            and current_workers < self.max_workers
            and cooldown_ok
        ):
            action = "scale_up"
            target = min(current_workers + 1, self.max_workers)
            reason = f"high latency ({avg_latency:.1f}ms > {self.target_latency_ms}ms) for {self._high_latency_count} checks"
            self._last_scale_time = now
            self._high_latency_count = 0

        elif (
            self._low_latency_count >= self.scale_down_consecutive
            and current_workers > self.min_workers
            and cooldown_ok
            and not high_errors
        ):
            action = "scale_down"
            target = max(current_workers - 1, self.min_workers)
            reason = f"low latency ({avg_latency:.1f}ms < {self.target_latency_ms * 0.5}ms) for {self._low_latency_count} checks"
            self._last_scale_time = now
            self._low_latency_count = 0

        if action != "hold":
            self._last_action = action

        return {
            "action": action,
            "target_workers": target,
            "reason": reason,
            "current_workers": current_workers,
            "avg_latency_ms": round(avg_latency, 1),
            "error_rate": round(error_rate, 4),
            "total_requests": total_requests,
            "redis_active_users": metrics.get("active_users", 0),
            "redis_memory_mb": metrics.get("redis_memory_mb", 0),
        }


class WorkerPoolManager:
    """Manages a pool of inference workers with auto-scaling.

    Lifecycle:
      1. Start with min_workers on consecutive ports
      2. Monitor health every 10s
      3. Collect Redis metrics every 10s
      4. Evaluate scaling every 30s
      5. Start/stop workers as needed

    Args:
        base_port: Starting port for workers (default: 8006)
        min_workers: Minimum workers to maintain
        max_workers: Maximum workers allowed
        target_latency_ms: Target per-request latency
        redis_url: Redis URL for metrics collection
        check_interval: Seconds between health checks
        scale_interval: Seconds between scaling evaluations
    """

    def __init__(
        self,
        base_port: int = 8006,
        min_workers: int = 2,
        max_workers: int = 8,
        target_latency_ms: float = 50.0,
        redis_url: str = "redis://localhost:6379/0",
        check_interval: float = 10.0,
        scale_interval: float = 30.0,
    ):
        self.base_port = base_port
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.check_interval = check_interval
        self.scale_interval = scale_interval

        # Redis
        self._redis = redis.Redis.from_url(redis_url, protocol=2, socket_timeout=3)
        self._metrics = RedisMetricsCollector(self._redis)
        self._scaler = AutoScaler(
            min_workers=min_workers,
            max_workers=max_workers,
            target_latency_ms=target_latency_ms,
        )

        # Workers
        self._workers: dict[int, WorkerInfo] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._scaling_events: list[dict] = []
        self._total_starts = 0
        self._total_stops = 0

    def start(self) -> None:
        """Start the worker pool manager."""
        if self._running:
            return

        self._running = True

        # Start initial workers
        for i in range(self.min_workers):
            port = self.base_port + i
            if not self._is_port_in_use(port):
                self._start_worker(port)
            else:
                # Port already in use — adopt it
                pid = self._find_pid_on_port(port)
                if pid:
                    with self._lock:
                        self._workers[port] = WorkerInfo(
                            port=port, pid=pid, started_at=time.time()
                        )
                    print(f"[pool] Adopted existing worker on port {port} (PID {pid})")

        # Start monitoring thread
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[pool] Started: {len(self._workers)} workers, ports {self.base_port}-{self.base_port + len(self._workers) - 1}")

    def stop(self) -> None:
        """Gracefully stop all workers and the manager."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        # Stop all workers
        with self._lock:
            ports = list(self._workers.keys())

        for port in ports:
            self._stop_worker(port)

        print(f"[pool] Stopped: {self._total_starts} starts, {self._total_stops} stops")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        last_scale_check = 0.0

        while self._running:
            try:
                # Collect metrics
                metrics = self._metrics.collect()

                # Health check workers
                self._health_check_workers()

                # Scale evaluation
                now = time.time()
                if (now - last_scale_check) >= self.scale_interval:
                    self._evaluate_scaling(metrics)
                    last_scale_check = now

            except Exception as e:
                print(f"[pool] Monitor error: {e}")

            time.sleep(self.check_interval)

    def _health_check_workers(self) -> None:
        """Check health of all workers."""
        import urllib.request

        with self._lock:
            ports = list(self._workers.keys())

        for port in ports:
            worker = self._workers.get(port)
            if not worker:
                continue

            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/health",
                    headers={"Accept": "application/json"},
                )
                resp = urllib.request.urlopen(req, timeout=3)
                worker.healthy = True
                worker.last_health_check = time.time()
            except Exception:
                worker.healthy = False
                # If worker is dead, restart it
                if not self._is_port_in_use(port):
                    print(f"[pool] Worker on port {port} died, restarting...")
                    self._stop_worker(port)
                    self._start_worker(port)

    def _evaluate_scaling(self, metrics: dict) -> None:
        """Evaluate whether to scale up or down."""
        with self._lock:
            n_workers = len(self._workers)
            ports = list(self._workers.keys())

        # Gather worker metrics
        latencies = []
        errors = []
        requests = []
        for port in ports:
            worker = self._workers.get(port)
            if worker and worker.healthy:
                latencies.append(worker.avg_latency_ms or 50.0)  # default if unknown
                errors.append(worker.error_count)
                requests.append(worker.request_count)

        decision = self._scaler.evaluate(
            metrics, n_workers, latencies, errors, requests
        )

        if decision["action"] == "scale_up":
            self._scaling_events.append({**decision, "timestamp": time.time()})
            new_port = self._find_free_port()
            if new_port:
                self._start_worker(new_port)
                print(f"[pool] SCALE UP → {decision['target_workers']} workers (port {new_port})")

        elif decision["action"] == "scale_down":
            self._scaling_events.append({**decision, "timestamp": time.time()})
            # Stop the least healthy worker
            with self._lock:
                worst_port = min(
                    self._workers.keys(),
                    key=lambda p: (
                        not self._workers[p].healthy,
                        self._workers[p].request_count,
                    ),
                )
            self._stop_worker(worst_port)
            print(f"[pool] SCALE DOWN → {decision['target_workers']} workers (stopped port {worst_port})")

    def _start_worker(self, port: int) -> bool:
        """Start a new inference worker on the given port."""
        try:
            log_out = str(ROOT / ".freebuff" / f"inference_{port}.log")
            log_err = str(ROOT / ".freebuff" / f"inference_{port}.log.err")

            proc = subprocess.Popen(
                [
                    str(ROOT / ".venv" / "Scripts" / "python.exe"),
                    "-m", "uvicorn",
                    "src.inference.service:app",
                    "--host", "0.0.0.0",
                    "--port", str(port),
                ],
                stdout=open(log_out, "w"),
                stderr=open(log_err, "w"),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                cwd=str(ROOT),
            )

            # Wait briefly for startup
            time.sleep(2)

            if self._is_port_in_use(port):
                with self._lock:
                    self._workers[port] = WorkerInfo(
                        port=port,
                        pid=proc.pid,
                        started_at=time.time(),
                    )
                self._total_starts += 1
                print(f"[pool] Started worker on port {port} (PID {proc.pid})")
                return True
            else:
                proc.kill()
                print(f"[pool] Failed to start worker on port {port}")
                return False

        except Exception as e:
            print(f"[pool] Error starting worker on port {port}: {e}")
            return False

    def _stop_worker(self, port: int) -> bool:
        """Gracefully stop a worker."""
        with self._lock:
            worker = self._workers.pop(port, None)

        if not worker:
            return False

        try:
            os.kill(worker.pid, signal.SIGTERM)
            time.sleep(2)
            try:
                os.kill(worker.pid, signal.SIGTERM)
            except OSError:
                pass
            self._total_stops += 1
            print(f"[pool] Stopped worker on port {port} (PID {worker.pid})")
            return True
        except OSError:
            # Process already dead
            self._total_stops += 1
            return True

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.settimeout(1)
                s.connect(("127.0.0.1", port))
                return True
            except (ConnectionRefusedError, OSError, socket.timeout):
                return False

    def _find_pid_on_port(self, port: int) -> Optional[int]:
        """Find the PID of the process listening on a port."""
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        return int(parts[-1])
        except Exception:
            pass
        return None

    def _find_free_port(self) -> Optional[int]:
        """Find the next free port starting from base_port."""
        for offset in range(self.max_workers):
            port = self.base_port + offset
            if port not in self._workers and not self._is_port_in_use(port):
                return port
        return None

    def get_status(self) -> dict:
        """Get current pool status."""
        with self._lock:
            workers = {p: w.to_dict() for p, w in self._workers.items()}

        metrics = self._metrics.history[-1] if self._metrics.history else {}
        trend = self._metrics.get_trend("request_rate_1m")

        return {
            "running": self._running,
            "workers": workers,
            "n_workers": len(workers),
            "n_healthy": sum(1 for w in workers.values() if w.get("healthy")),
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "total_starts": self._total_starts,
            "total_stops": self._total_stops,
            "metrics": metrics,
            "request_rate_trend": trend,
            "recent_scaling": self._scaling_events[-5:],
        }


# ── CLI ──

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference worker pool manager")
    parser.add_argument("--min", type=int, default=2, help="Min workers")
    parser.add_argument("--max", type=int, default=8, help="Max workers")
    parser.add_argument("--base-port", type=int, default=8006, help="Base port")
    parser.add_argument("--target-latency", type=float, default=50.0, help="Target latency (ms)")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--status", action="store_true", help="Print status and exit")
    args = parser.parse_args()

    pool = WorkerPoolManager(
        base_port=args.base_port,
        min_workers=args.min,
        max_workers=args.max,
        target_latency_ms=args.target_latency,
        redis_url=args.redis_url,
    )

    if args.status:
        status = pool.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        # Install signal handlers
        def handle_signal(sig, frame):
            print("\n[pool] Shutting down...")
            pool.stop()
            exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        pool.start()
        print(f"[pool] Running (min={args.min}, max={args.max}, target={args.target_latency}ms)")
        print("[pool] Press Ctrl+C to stop")

        # Keep alive
        try:
            while pool._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pool.stop()
