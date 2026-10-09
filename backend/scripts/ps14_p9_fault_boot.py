#!/usr/bin/env python3
"""PS-14 Phase 9 -- live-server adapter for the Phase 8 audit fault injection.

Phase 8's DB-4 fault injection lives in the in-process test harness
(`scripts/audit_resilience_test.py`): it swaps `writer.SessionLocal` for a
failing or deliberately slow factory and asserts that the decision path never
blocks on, or raises because of, DB-4.

Phase 9 measures the SAME behavior over real HTTP, so the identical swap has
to happen inside a real uvicorn process. This module performs that swap at
import time and then re-exports the unmodified risk-engine app. No service
source file is changed; this bootstrap exists only to make the measurement
possible.

Modes (env `PS14_P9_AUDIT_MODE`):

  healthy      (default) no patch -- `src.risk_engine.main:app` verbatim
  delayed      every writer-side session creation sleeps
               `PS14_P9_AUDIT_DELAY` seconds (default 0.25) before returning
               the real session, i.e. persistence is slowed but healthy
  unavailable  writer-side session creation raises `OperationalError`,
               exactly like `fail_sessions()` in the Phase 8 suite

Run (repo root, PYTHONPATH=backend):

    python -m uvicorn ps14_p9_fault_boot:app --app-dir backend/scripts --port 8009
"""

from __future__ import annotations

import os
import time

MODE = os.environ.get("PS14_P9_AUDIT_MODE", "healthy").lower()

if MODE in ("delayed", "unavailable"):
    from sqlalchemy.exc import OperationalError

    import src.audit_service.writer as w

    if MODE == "unavailable":
        def _boom(*_a, **_k):
            raise OperationalError(
                "audit_events",
                {},
                OSError("phase9 injected: DB-4 unavailable"),
            )
        w.SessionLocal = _boom  # type: ignore[assignment]
    else:
        _real_session = w.SessionLocal
        _delay = float(os.environ.get("PS14_P9_AUDIT_DELAY", "0.25"))

        def _slow_session(*_a, **_k):
            time.sleep(_delay)
            return _real_session(*_a, **_k)
        w.SessionLocal = _slow_session  # type: ignore[assignment]

# Import AFTER the swap: the app's background tasks call the writer module's
# globals at run time, so either order works -- kept last for clarity.
from src.risk_engine.main import app  # noqa: E402,F401

__all__ = ["app", "MODE"]
