"""Validation gates — every critical check that must pass before evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class GateResult:
    name: str
    passed: bool
    reason: str = ""
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gate": self.name,
            "passed": self.passed,
            "reason": self.reason,
            "evidence": self.evidence,
        }


class ValidationGates:
    """Collects and evaluates all validation gates."""

    def __init__(self):
        self.gates: list[GateResult] = []

    def record(self, name: str, passed: bool, reason: str = "", evidence: dict | None = None):
        self.gates.append(GateResult(name, passed, reason, evidence or {}))

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed]

    def summary(self) -> dict:
        return {
            "total": len(self.gates),
            "passed": sum(1 for g in self.gates if g.passed),
            "failed": sum(1 for g in self.gates if not g.passed),
            "all_passed": self.all_passed,
            "gates": [g.to_dict() for g in self.gates],
        }

    def write(self, path: Path):
        path.write_text(json.dumps(self.summary(), indent=2, default=str), encoding="utf-8")


def gate_firewall(final_test_accessed: bool, production_modified: bool,
                  e_hardneg_modified: bool, p20_modified: bool) -> GateResult:
    """§1 — Absolute firewall."""
    ok = (not final_test_accessed and not production_modified
          and not e_hardneg_modified and not p20_modified)
    return GateResult("firewall", ok,
                      "PASS" if ok else "METHODOLOGY_FAILURE — protected artifact changed",
                      {"final_test_accessed": final_test_accessed,
                       "production_modified": production_modified,
                       "e_hardneg_modified": e_hardneg_modified,
                       "p20_modified": p20_modified})


def gate_real_world_data(metadata_status: str) -> GateResult:
    """§5 — Real-world data acceptance."""
    ok = metadata_status == "REAL_WORLD"
    return GateResult("real_world_data", ok,
                      "PASS" if ok else f"BLOCKED — dataset classified as {metadata_status}")


def gate_authorization(auth_status: str) -> GateResult:
    ok = auth_status == "AUTHORIZED"
    return GateResult("authorization", ok,
                      "PASS" if ok else f"BLOCKED — authorization={auth_status}")


def gate_label_definition(label_def: str) -> GateResult:
    ok = bool(label_def) and label_def != "UNVERIFIED"
    return GateResult("label_definition", ok,
                      "PASS" if ok else "BLOCKED — label definition unverified")


def gate_label_latency(label_latency_known: bool) -> GateResult:
    return GateResult("label_latency", label_latency_known,
                      "PASS" if label_latency_known else "BLOCKED — LABEL_LATENCY=UNVERIFIED")


def gate_schema_compatibility(n_matched: int, n_required: int) -> GateResult:
    ok = n_matched >= n_required
    return GateResult("schema_compatibility", ok,
                      f"PASS ({n_matched}/{n_required})" if ok
                      else f"FAIL ({n_matched}/{n_required} required fields matched)",
                      {"matched": n_matched, "required": n_required})


def gate_feature_reconstruction(p20_ok: bool, e_hardneg_ok: Optional[bool] = None) -> GateResult:
    ok = p20_ok
    evidence = {"p20": p20_ok}
    if e_hardneg_ok is not None:
        evidence["e_hardneg"] = e_hardneg_ok
    return GateResult("feature_reconstruction", ok,
                      "PASS" if ok else "BLOCKED — P20 feature reconstruction failed",
                      evidence)


def gate_temporal_ordering(sorted_correctly: bool) -> GateResult:
    return GateResult("temporal_ordering", sorted_correctly,
                      "PASS" if sorted_correctly else "BLOCKED — temporal ordering violation")


def gate_threshold_frozen(p20_frozen: bool, e_hardneg_frozen: bool) -> GateResult:
    ok = p20_frozen and e_hardneg_frozen
    return GateResult("threshold_frozen", ok,
                      "PASS" if ok else "BLOCKED — threshold not frozen",
                      {"p20": p20_frozen, "e_hardneg": e_hardneg_frozen})


def gate_final_holdout(model_frozen: bool, threshold_frozen: bool,
                       feature_contract_frozen: bool, protocol_frozen: bool) -> GateResult:
    ok = all([model_frozen, threshold_frozen, feature_contract_frozen, protocol_frozen])
    return GateResult("final_holdout", ok,
                      "PASS" if ok else "BLOCKED — holdout authorization failed",
                      {"model_frozen": model_frozen, "threshold_frozen": threshold_frozen,
                       "feature_contract_frozen": feature_contract_frozen,
                       "protocol_frozen": protocol_frozen})


def gate_promotion(gates: ValidationGates, performance_met: Optional[bool] = None) -> GateResult:
    """§32 — Promotion gate — all applicable gates must pass."""
    all_ok = gates.all_passed
    if performance_met is not None:
        all_ok = all_ok and performance_met
    blocking = [g.name for g in gates.failures]
    return GateResult("promotion", all_ok,
                      "ALLOWED" if all_ok else f"BLOCKED by: {', '.join(blocking)}",
                      {"blocking_gates": blocking})
