"""Runtime drift detection — sliding-window PSI against the training baseline.

Unlike the offline drift monitor (scripts/drift_monitor.py) which checks
weekly windows from the database, this module runs in-process on every
evaluate() call.  It maintains a circular buffer of recent feature vectors,
periodically computes per-feature PSI against the training baseline, and
exposes a simple predicate that the Risk Engine uses to decide whether to
fall back to rules-only scoring.

Thresholds (standard industry PSI bands):
  PSI < 0.10  -> NORMAL   (model continues)
  0.10-0.25   -> WARNING  (log, model continues but alert emitted)
  PSI >= 0.25 -> CRITICAL (model paused, rules-only fallback)

Design:
  - Thread-safe via a simple lock (FastAPI is sync by default here).
  - Re-checks PSI every ``check_interval`` events, not every event.
  - The sliding window is a fixed-size deque; old events are evicted.
  - If no baseline is available the detector is a no-op (passthrough).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from src.drift_monitor.psi import PSI_ALERT, PSI_WARN, bin_edges, psi_proportions

logger = logging.getLogger(__name__)

_EPS = 1e-4  # smoothing for empty bins (same as psi.py)

# Drift states the Risk Engine can query.
NORMAL = "normal"
WARNING = "warning"
CRITICAL = "critical"


class DriftDetector:
    """In-process sliding-window drift detector.

    Parameters
    ----------
    baseline_path : Path | None
        Path to drift_baseline.json (produced by scripts/drift_monitor.py
        build-baseline).  If None or file missing, detector is a no-op.
    window_size : int
        Max events kept in the sliding window (default 500).
    check_interval : int
        Recompute PSI every N events (default 100 — amortises cost).
    features : list[str] | None
        Feature names to monitor; defaults to ML_FEATURES.
    """

    def __init__(
        self,
        baseline_path: Path | None = None,
        window_size: int = 500,
        check_interval: int = 100,
        features: list[str] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._window: deque[dict[str, float]] = deque(maxlen=window_size)
        self._check_interval = check_interval
        self._events_since_check = 0
        self._baseline: dict | None = None
        self._state: str = NORMAL
        self._max_psi: float = 0.0
        self._per_feature_psi: dict[str, float] = {}
        self._last_check_ts: float = 0.0
        self._total_events: int = 0
        self._critical_since: float | None = None

        # Import here to avoid circular imports at module level.
        from src.privacy_layer.features import ML_FEATURES
        self._features = features or ML_FEATURES

        if baseline_path and Path(baseline_path).exists():
            try:
                raw = json.loads(baseline_path.read_text(encoding="utf-8"))
                # Strip the "meta" key if present (it's metadata, not feature baselines).
                self._baseline = {k: v for k, v in raw.items() if k != "meta"}
                logger.info(
                    "drift detector loaded baseline from %s (%d features)",
                    baseline_path,
                    len(self._baseline),
                )
            except Exception:
                logger.warning("drift detector: failed to load baseline from %s", baseline_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, features: dict[str, float]) -> None:
        """Record one event's feature vector into the sliding window."""
        with self._lock:
            row = {f: float(features.get(f, 0.0)) for f in self._features}
            self._window.append(row)
            self._events_since_check += 1
            self._total_events += 1
            if self._events_since_check >= self._check_interval:
                self._run_check()

    @property
    def state(self) -> str:
        """Current drift state: normal / warning / critical."""
        return self._state

    @property
    def is_critical(self) -> bool:
        """True when drift exceeds the alert threshold — model should pause."""
        return self._state == CRITICAL

    @property
    def should_fallback(self) -> bool:
        """Alias used by the Risk Engine to decide rules-only mode."""
        return self.is_critical

    @property
    def total_events(self) -> int:
        return self._total_events

    @property
    def max_psi(self) -> float:
        return self._max_psi

    @property
    def per_feature_psi(self) -> dict[str, float]:
        return dict(self._per_feature_psi)

    def status(self) -> dict[str, Any]:
        """Snapshot for the /internal/drift-status endpoint."""
        with self._lock:
            alerting = {
                f: psi_v
                for f, psi_v in self._per_feature_psi.items()
                if psi_v >= PSI_WARN
            }
            return {
                "state": self._state,
                "max_psi": round(self._max_psi, 4),
                "total_events": self._total_events,
                "window_size": len(self._window),
                "window_capacity": self._window.maxlen,
                "check_interval": self._check_interval,
                "features_alerting": alerting,
                "last_check_ts": self._last_check_ts,
                "critical_since": self._critical_since,
                "baseline_loaded": self._baseline is not None,
                "thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
            }

    def _reset_unlocked(self) -> None:
        """Reset state without acquiring the lock (caller must hold it)."""
        self._window.clear()
        self._events_since_check = 0
        self._state = NORMAL
        self._max_psi = 0.0
        self._per_feature_psi = {}
        self._critical_since = None
        self._total_events = 0

    def reset(self) -> None:
        """Reset detector state (e.g. after a successful retrain)."""
        with self._lock:
            self._reset_unlocked()
            logger.info("drift detector reset")

    def reload_baseline(self, path: Path) -> None:
        """Hot-reload a new baseline (e.g. after retrain)."""
        with self._lock:
            try:
                raw = json.loads(Path(path).read_text(encoding="utf-8"))
                self._baseline = {k: v for k, v in raw.items() if k != "meta"}
                self._reset_unlocked()
                logger.info("drift detector baseline reloaded from %s", path)
            except Exception:
                logger.warning("drift detector: failed to reload baseline from %s", path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_check(self) -> None:
        """Compute PSI for the current window against the baseline.

        Must be called with self._lock held.
        """
        self._events_since_check = 0
        self._last_check_ts = time.time()

        if self._baseline is None or len(self._window) < 30:
            return  # not enough data or no baseline

        # Build a temporary array from the window for fast histogram.
        rows = np.array(
            [[row[f] for f in self._features] for row in self._window],
            dtype=np.float64,
        )

        max_psi = 0.0
        per_feature: dict[str, float] = {}

        for idx, feat in enumerate(self._features):
            b = self._baseline.get(feat)
            if b is None or b.get("constant"):
                continue

            vals = rows[:, idx]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 10:
                continue

            edges = np.asarray(b["edges"], dtype=float)
            counts, _ = np.histogram(vals, bins=edges)
            nb = len(edges) - 1
            ep = np.asarray(b["expected"], dtype=float)
            ep = ep / ep.sum()
            ap = (counts + _EPS) / (counts.sum() + _EPS * nb)
            psi_val = psi_proportions(ep, ap)

            per_feature[feat] = round(float(psi_val), 4)
            if psi_val > max_psi:
                max_psi = psi_val

        self._max_psi = round(max_psi, 4)
        self._per_feature_psi = per_feature

        # Determine state.
        alerting = [f for f, p in per_feature.items() if p >= PSI_ALERT]
        warn_features = [f for f, p in per_feature.items() if PSI_WARN <= p < PSI_ALERT]

        if alerting:
            new_state = CRITICAL
        elif warn_features:
            new_state = WARNING
        else:
            new_state = NORMAL

        # State transition logging.
        if new_state != self._state:
            if new_state == CRITICAL:
                self._critical_since = time.time()
                logger.warning(
                    "DRIFT CRITICAL: %d features alerting (PSI>=%.2f), "
                    "max PSI=%.4f — model paused, rules-only fallback",
                    len(alerting),
                    PSI_ALERT,
                    max_psi,
                )
            elif new_state == WARNING:
                logger.warning(
                    "DRIFT WARNING: %d features drifting (PSI>=%.1f), "
                    "max PSI=%.4f",
                    len(warn_features),
                    PSI_WARN,
                    max_psi,
                )
            else:
                logger.info(
                    "DRIFT NORMAL: was %s, now recovered (max PSI=%.4f)",
                    self._state,
                    max_psi,
                )
        self._state = new_state
