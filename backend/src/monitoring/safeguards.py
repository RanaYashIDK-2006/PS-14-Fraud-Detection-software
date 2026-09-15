"""Monitoring safeguards — prevents drift detection from auto-retraining or auto-promoting.

Core rules:
  DRIFT DETECTED != AUTOMATIC RETRAINING
  DRIFT DETECTED != AUTOMATIC PROMOTION

This module provides a gate that must be explicitly satisfied before
retraining or promotion can proceed. It checks:
  - drift state is not CRITICAL (or retrain is explicitly requested)
  - promotion gates are all satisfied
  - no active critical alerts
  - real-world validation status
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class SafeguardStatus(str, Enum):
    PASS = "pass"
    BLOCKED = "blocked"
    CONDITIONAL = "conditional"


@dataclass
class SafeguardResult:
    """Result of a safeguard check."""
    check_name: str
    status: SafeguardStatus
    detail: str
    blocking: bool = False  # if True, prevents the action

    def to_dict(self) -> dict:
        return {
            "check": self.check_name,
            "status": self.status.value,
            "detail": self.detail,
            "blocking": self.blocking,
        }


@dataclass
class PromotionSafeguard:
    """Evaluates whether a model promotion should proceed.

    Drift detection alone NEVER authorizes promotion. This safeguard
    requires all existing gates to be satisfied.
    """

    def check_retrain_eligible(
        self,
        drift_state: str,
        has_active_critical_alerts: bool = False,
        monitoring_healthy: bool = True,
    ) -> list[SafeguardResult]:
        """Check if retraining is eligible given current drift state.

        Drift DETECTED does NOT mean retraining should happen automatically.
        This check ensures retraining is a deliberate decision, not automatic.
        """
        results = []

        # Check 1: Drift state
        if drift_state == "critical":
            results.append(SafeguardResult(
                check_name="drift_state",
                status=SafeguardStatus.CONDITIONAL,
                detail="drift is CRITICAL — retraining may be considered but must be explicitly requested",
                blocking=False,  # drift CRITICAL suggests retrain but doesn't auto-trigger
            ))
        else:
            results.append(SafeguardResult(
                check_name="drift_state",
                status=SafeguardStatus.PASS,
                detail=f"drift state is {drift_state} — no automatic retrain needed",
            ))

        # Check 2: Active critical alerts
        if has_active_critical_alerts:
            results.append(SafeguardResult(
                check_name="active_alerts",
                status=SafeguardStatus.BLOCKED,
                detail="active critical alerts must be acknowledged before retraining",
                blocking=True,
            ))
        else:
            results.append(SafeguardResult(
                check_name="active_alerts",
                status=SafeguardStatus.PASS,
                detail="no active critical alerts",
            ))

        # Check 3: Monitoring health
        if not monitoring_healthy:
            results.append(SafeguardResult(
                check_name="monitoring_health",
                status=SafeguardStatus.BLOCKED,
                detail="monitoring system is unhealthy — cannot verify drift state",
                blocking=True,
            ))
        else:
            results.append(SafeguardResult(
                check_name="monitoring_health",
                status=SafeguardStatus.PASS,
                detail="monitoring system is healthy",
            ))

        return results

    def check_promotion_eligible(
        self,
        drift_state: str,
        leakage_checks_pass: bool = False,
        data_quality_pass: bool = False,
        validation_performance_pass: bool = False,
        untouched_test_pass: bool = False,
        robustness_pass: bool = False,
        production_parity_pass: bool = False,
        security_pass: bool = False,
        approval_pass: bool = False,
        real_world_validation_status: str = "BLOCKED",
        has_active_critical_alerts: bool = False,
    ) -> list[SafeguardResult]:
        """Check if promotion should proceed.

        ALL gates must pass. Drift state alone is never sufficient.
        Real-world validation status is checked but does not auto-block
        prototype promotions.
        """
        results = []

        # Gate checks (all must pass)
        gates = {
            "leakage_checks": leakage_checks_pass,
            "data_quality": data_quality_pass,
            "validation_performance": validation_performance_pass,
            "untouched_test": untouched_test_pass,
            "robustness": robustness_pass,
            "production_parity": production_parity_pass,
            "security": security_pass,
            "approval": approval_pass,
        }

        for gate_name, passed in gates.items():
            if passed:
                results.append(SafeguardResult(
                    check_name=f"gate_{gate_name}",
                    status=SafeguardStatus.PASS,
                    detail=f"{gate_name} passed",
                ))
            else:
                results.append(SafeguardResult(
                    check_name=f"gate_{gate_name}",
                    status=SafeguardStatus.BLOCKED,
                    detail=f"{gate_name} not verified — promotion blocked",
                    blocking=True,
                ))

        # Drift state check
        if drift_state == "critical":
            results.append(SafeguardResult(
                check_name="drift_state",
                status=SafeguardStatus.BLOCKED,
                detail="drift is CRITICAL — promotion blocked until drift resolves",
                blocking=True,
            ))
        elif drift_state == "warning":
            results.append(SafeguardResult(
                check_name="drift_state",
                status=SafeguardStatus.CONDITIONAL,
                detail="drift is WARNING — promotion allowed but monitor closely",
            ))
        else:
            results.append(SafeguardResult(
                check_name="drift_state",
                status=SafeguardStatus.PASS,
                detail=f"drift state is {drift_state}",
            ))

        # Active alerts
        if has_active_critical_alerts:
            results.append(SafeguardResult(
                check_name="active_alerts",
                status=SafeguardStatus.BLOCKED,
                detail="active critical alerts must be resolved before promotion",
                blocking=True,
            ))

        # Real-world validation status
        if real_world_validation_status == "BLOCKED":
            results.append(SafeguardResult(
                check_name="real_world_validation",
                status=SafeguardStatus.CONDITIONAL,
                detail="real-world validation is BLOCKED — prototype promotion allowed, production promotion blocked",
            ))
        else:
            results.append(SafeguardResult(
                check_name="real_world_validation",
                status=SafeguardStatus.PASS,
                detail=f"real-world validation status: {real_world_validation_status}",
            ))

        return results

    def summarize(self, results: list[SafeguardResult]) -> dict:
        """Summarize safeguard results into a promotion decision."""
        blocking = [r for r in results if r.blocking and r.status == SafeguardStatus.BLOCKED]
        conditional = [r for r in results if r.status == SafeguardStatus.CONDITIONAL]

        if blocking:
            return {
                "eligible": False,
                "status": "BLOCKED",
                "blocking_checks": [r.check_name for r in blocking],
                "conditional_checks": [r.check_name for r in conditional],
                "detail": f"blocked by: {', '.join(r.check_name for r in blocking)}",
            }
        elif conditional:
            return {
                "eligible": True,
                "status": "CONDITIONAL",
                "blocking_checks": [],
                "conditional_checks": [r.check_name for r in conditional],
                "detail": f"conditional: {', '.join(r.check_name for r in conditional)}",
            }
        else:
            return {
                "eligible": True,
                "status": "PASS",
                "blocking_checks": [],
                "conditional_checks": [],
                "detail": "all checks passed",
            }
