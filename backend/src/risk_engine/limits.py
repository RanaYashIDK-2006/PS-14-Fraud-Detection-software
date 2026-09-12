"""Pre-scoring velocity and spend limits (production hardening: section 7).

Per-account and per-device hard/soft caps evaluated BEFORE the ML+rules
scoring in the Risk Engine. Hard caps auto-block (score forced to 100,
decision = verify); soft caps trigger step-up with reason codes.

Configured via the ``velocity_limits`` section in ``rules.yaml``::

    velocity_limits:
      per_account:
        daily_count: 10          # hard cap: max txns per account in 24h
        daily_spend_ratio: 3.0   # soft cap: cumulative spend vs median
      per_device:
        daily_count: 20          # hard cap: max txns across ALL accounts on this device in 24h

All fields are optional; omitted fields mean no limit is enforced.
The limits query the feature store (DB-2) for the account's own prior events
and the global DeviceFingerprint store for device-level counts.
"""

from __future__ import annotations

from typing import Any


def _check_daily_count(count: int, limit: int, label: str, level: str) -> dict[str, Any] | None:
    if count > limit:
        return {
            "reason_code": label,
            "level": level,
            "detail": f"{count} events in 24h (limit {limit})",
        }
    return None


def _check_daily_spend(ratio: float, limit: float) -> dict[str, Any] | None:
    """ratio = cumulative daily spend / median_amount (computed in the
    Privacy Layer, which owns DB-2 - the Risk Engine never sees raw
    amounts)."""
    if ratio > limit:
        return {
            "reason_code": "DAILY_SPEND_EXCEEDED",
            "level": "soft",
            "detail": f"daily spend ratio {ratio:.2f}x (limit {limit:.1f}x)",
        }
    return None


def evaluate_limits(features: dict, config: dict[str, Any] | None) -> dict[str, Any]:
    """Evaluate velocity limits from the derived feature vector.

    All inputs come from the §16 feature vector the Privacy Layer built
    (``txn_freq_last_24h`` = account 24h count, ``account_daily_spend_ratio``
    = daily spend / median, ``device_daily_count`` = 24h device count across
    accounts) - no raw amounts or DB-2 queries reach the Risk Engine.

    Returns ``{"pass": True/False, "triggers": [...]}`` where each trigger
    is a reason-code + level dict. A hard cap failure means the score will
    be forced to 100 (verify) regardless of ML+rules.
    """
    if not config:
        return {"pass": True, "triggers": []}

    triggers: list[dict[str, Any]] = []
    acfg = config.get("per_account", {})
    dcfg = config.get("per_device", {})

    daily_count = acfg.get("daily_count")
    daily_spend = acfg.get("daily_spend_ratio")
    device_count = dcfg.get("daily_count")

    if daily_count is not None:
        # txn_freq_last_24h counts events BEFORE this one; +1 makes the
        # current event part of the total so the transaction that breaks the
        # cap is the one that trips it.
        t = _check_daily_count(
            int(features.get("txn_freq_last_24h", 0)) + 1, daily_count, "ACCOUNT_DAILY_LIMIT", "hard"
        )
        if t:
            triggers.append(t)

    if daily_spend is not None:
        t = _check_daily_spend(float(features.get("account_daily_spend_ratio", 0.0)), daily_spend)
        if t:
            triggers.append(t)

    if device_count is not None:
        t = _check_daily_count(
            int(features.get("device_daily_count", 0)), device_count, "DEVICE_DAILY_LIMIT", "hard"
        )
        if t:
            triggers.append(t)

    any_hard = any(t["level"] == "hard" for t in triggers)
    return {"pass": len(triggers) == 0, "triggers": triggers, "hard": any_hard}
