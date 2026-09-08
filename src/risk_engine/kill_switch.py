"""Kill switch (check #22): controlled quarantine of the ML scoring path.

A file-backed switch (`models/kill_switch.json`) that, when armed, makes the
Altman ensemble engine refuse to serve ML scores (raises KillSwitchActiveError)
so callers degrade to the documented safe state (rules-only / block) instead of
silently continuing with a suspect model.

Guarantees:
- Only authorized operators (explicit key strings) can arm/disarm.
- Every activation/deactivation is appended to the hash-chained audit trail
  (event_type="kill_switch") with actor, reason and incident id.
- The armed state is checked by the engine on every predict (cheap stat with a
  short TTL cache), so containment is immediate.
- Disarming restores the previous known-good behavior (engine serves again);
  prediction parity before/after is verified by the incident-response test.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SWITCH_PATH = ROOT / "models" / "kill_switch.json"

# Authorized operator keys. In production this would be a signed identity /
# role check; the prototype uses explicit keys so authorization can be tested.
AUTHORIZED_OPERATORS = {"ops-oncall", "platform-admin", "fraud-desk-lead"}

_SAFE_STATE = "rules_only"  # documented safe state when armed: no ML scoring

_armed_cache: dict = {"mtime": 0.0, "value": False}
_TTL = 2.0  # seconds; re-stat the switch at most every 2s per process


class KillSwitchActiveError(RuntimeError):
    """Raised when the model scoring path is quarantined by the kill switch."""


def _read_state() -> dict:
    if not SWITCH_PATH.exists():
        return {}
    try:
        return json.loads(SWITCH_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict) -> None:
    SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    SWITCH_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _audit(action: str, by: str, reason: str, incident_id: str) -> None:
    """Append to the hash-chained audit trail (best-effort; never blocks)."""
    try:
        from src.audit_service.writer import append_audit_event, flush_audit_queue
        append_audit_event(
            incident_id,
            "kill_switch",
            {"action": action, "operator": by, "reason": reason,
             "incident_id": incident_id, "safe_state": _SAFE_STATE,
             "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        )
        flush_audit_queue(timeout=2.0)
    except Exception:
        pass  # audit write failure must never crash the kill-switch path


def arm(reason: str, by: str, incident_id: str) -> dict:
    """Arm the kill switch. Raises PermissionError for unauthorized operators."""
    if by not in AUTHORIZED_OPERATORS:
        # Log the ATTEMPT too (unauthorized attempts are security-relevant).
        _audit("arm_denied", by, reason, incident_id)
        raise PermissionError(
            f"operator '{by}' is not authorized to arm the kill switch"
        )
    state = {
        "armed": True,
        "armed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "armed_by": by,
        "reason": reason,
        "incident_id": incident_id,
        "safe_state": _SAFE_STATE,
    }
    _write_state(state)
    _audit("arm", by, reason, incident_id)
    return state


def disarm(by: str, incident_id: str) -> dict:
    """Disarm the kill switch (authorized operators only)."""
    if by not in AUTHORIZED_OPERATORS:
        _audit("disarm_denied", by, "", incident_id)
        raise PermissionError(
            f"operator '{by}' is not authorized to disarm the kill switch"
        )
    if SWITCH_PATH.exists():
        SWITCH_PATH.unlink()
    _audit("disarm", by, "incident resolved, model re-enabled", incident_id)
    return {"armed": False, "disarmed_by": by, "incident_id": incident_id}


def status() -> dict:
    return _read_state()


def is_armed(ttl: float = _TTL) -> bool:
    """Cheap TTL-cached check; re-stats the switch file at most every `ttl`s."""
    global _armed_cache
    try:
        mtime = SWITCH_PATH.stat().st_mtime
    except FileNotFoundError:
        if _armed_cache["value"]:
            _armed_cache = {"mtime": 0.0, "value": False}
        return False
    now = time.monotonic()
    if _armed_cache["mtime"] == mtime and now - _armed_cache.get("t", 0) < ttl:
        return _armed_cache["value"]
    state = _read_state()
    _armed_cache = {"mtime": mtime, "value": bool(state.get("armed")), "t": now}
    return _armed_cache["value"]