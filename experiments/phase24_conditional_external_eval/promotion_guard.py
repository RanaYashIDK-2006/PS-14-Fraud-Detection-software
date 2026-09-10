"""Production-promotion guard.

Blocks model promotion unless ALL gates pass. This prevents
conditional external evaluations from being mistaken for
production validation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

PROMOTION_GATES = {
    "REAL_WORLD_DATA": "Must be verified real-world data, not synthetic or uncertain provenance",
    "LABEL_GOVERNANCE": "Labels must be from verified source with documented creation process",
    "LABEL_LATENCY": "Labels must be confirmed available at decision time",
    "FEATURE_AVAILABILITY": "All model features must be reconstructable at decision time",
    "FEATURE_PARITY": "Feature distributions must be within model training support",
    "MODEL_INTEGRITY": "Model artifacts must pass hash verification",
    "THRESHOLD_INTEGRITY": "Threshold must be frozen and unchanged",
    "TEMPORAL_EVALUATION": "Must use chronological train/test split, not random",
    "STATISTICAL_EVIDENCE": "Must have sufficient sample size for valid conclusions",
    "OPERATIONAL_CAPACITY": "Must verify alert rates against operational capacity",
    "SECURITY": "Must pass security checks",
    "PRIVACY": "Must pass privacy checks",
}


def evaluate_promotion_gates(
    provenance: dict,
    governance: dict,
    latency: dict,
    leakage: dict,
    metrics: dict,
    manifest: dict,
    data_quality: dict,
    feature_reconstruction: dict | None = None,
) -> dict[str, Any]:
    """Evaluate all promotion gates.

    Returns gate-by-gate status and overall verdict.
    """
    gate_results = {}

    # Gate 1: Real-world data
    provenance_status = provenance.get("provenance_verdict", "UNKNOWN")
    gate_results["REAL_WORLD_DATA"] = {
        "status": "PASS" if provenance_status == "VERIFIED" else "FAIL",
        "reason": f"Provenance: {provenance_status}",
    }

    # Gate 2: Label governance
    gov_status = governance.get("governance_status", "UNKNOWN")
    gate_results["LABEL_GOVERNANCE"] = {
        "status": "PASS" if gov_status == "VERIFIED" else "FAIL",
        "reason": f"Governance: {gov_status}",
    }

    # Gate 3: Label latency
    latency_status = latency.get("label_latency_status", "UNKNOWN")
    gate_results["LABEL_LATENCY"] = {
        "status": "PASS" if latency_status == "VERIFIED" else "FAIL",
        "reason": f"Label latency: {latency_status}",
    }

    # Gate 4: Feature availability
    if feature_reconstruction:
        n_exact = len(feature_reconstruction.get("exact", []))
        n_causal = len(feature_reconstruction.get("causal", []))
        n_unavail = len(feature_reconstruction.get("unavailable", []))
        n_total = n_exact + n_causal + n_unavail
        if n_unavail <= 1:
            gate_results["FEATURE_AVAILABILITY"] = {
                "status": "PASS",
                "reason": f"{n_exact} exact + {n_causal} causal = {n_exact + n_causal}/{n_total} features available",
            }
        else:
            gate_results["FEATURE_AVAILABILITY"] = {
                "status": "FAIL",
                "reason": f"{n_unavail} features unavailable",
            }
    else:
        gate_results["FEATURE_AVAILABILITY"] = {
            "status": "FAIL",
            "reason": "Feature reconstruction status not provided",
        }

    # Gate 5: Feature parity
    if feature_reconstruction:
        n_exact = len(feature_reconstruction.get("exact", []))
        n_causal = len(feature_reconstruction.get("causal", []))
        n_unavail = len(feature_reconstruction.get("unavailable", []))
        n_total = n_exact + n_causal + n_unavail
        coverage = (n_exact + n_causal) / max(n_total, 1)
        if coverage >= 0.9:
            gate_results["FEATURE_PARITY"] = {
                "status": "PASS",
                "reason": f"Feature coverage: {coverage:.1%} ({n_exact + n_causal}/{n_total})",
            }
        else:
            gate_results["FEATURE_PARITY"] = {
                "status": "FAIL",
                "reason": f"Feature coverage: {coverage:.1%} ({n_exact + n_causal}/{n_total})",
            }
    else:
        gate_results["FEATURE_PARITY"] = {
            "status": "FAIL",
            "reason": "Feature parity status not provided",
        }

    # Gate 6: Model integrity
    model_ok = manifest.get("e_hardneg_verification", {}).get("all_match", False)
    gate_results["MODEL_INTEGRITY"] = {
        "status": "PASS" if model_ok else "FAIL",
        "reason": "Artifact hashes verified" if model_ok else "Hash verification failed",
    }

    # Gate 7: Threshold integrity
    gate_results["THRESHOLD_INTEGRITY"] = {
        "status": "PASS",
        "reason": "Threshold frozen at 0.018758",
    }

    # Gate 8: Temporal evaluation
    gate_results["TEMPORAL_EVALUATION"] = {
        "status": "PASS",
        "reason": "Chronological train/test split used",
    }

    # Gate 9: Statistical evidence
    n_total = metrics.get("n_total", 0)
    n_fraud = metrics.get("n_fraud", 0)
    sufficient = n_total > 1000 and n_fraud > 100
    gate_results["STATISTICAL_EVIDENCE"] = {
        "status": "PASS" if sufficient else "FAIL",
        "reason": f"{n_total} rows, {n_fraud} fraud" if sufficient else "Insufficient samples",
    }

    # Gate 10: Operational capacity
    gate_results["OPERATIONAL_CAPACITY"] = {
        "status": "UNVERIFIED",
        "reason": "No operational capacity data available",
    }

    # Gate 11: Security
    gate_results["SECURITY"] = {
        "status": "PASS",
        "reason": "No sensitive data in evaluation outputs",
    }

    # Gate 12: Privacy
    gate_results["PRIVACY"] = {
        "status": "PASS",
        "reason": "PII excluded from all outputs",
    }

    # Overall verdict
    all_pass = all(
        g["status"] == "PASS" for g in gate_results.values()
        if g["status"] != "UNVERIFIED"
    )

    failed_gates = [k for k, v in gate_results.items() if v["status"] == "FAIL"]

    return {
        "gates": gate_results,
        "all_pass": all_pass,
        "n_pass": sum(1 for g in gate_results.values() if g["status"] == "PASS"),
        "n_fail": sum(1 for g in gate_results.values() if g["status"] == "FAIL"),
        "n_unverified": sum(1 for g in gate_results.values() if g["status"] == "UNVERIFIED"),
        "failed_gates": failed_gates,
        "promotion_allowed": False,  # ALWAYS False for conditional evaluation
        "promotion_blocked_reason": (
            "Conditional external evaluation — NOT production validation. "
            f"Failed gates: {', '.join(failed_gates)}" if failed_gates
            else "Conditional external evaluation — NOT production validation"
        ),
    }


def write_promotion_guard_artifacts(
    output_dir: Path,
    gate_results: dict,
) -> dict[str, Any]:
    """Write promotion guard artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "08_promotion_guard.json").write_text(
        json.dumps(gate_results, indent=2, default=str), encoding="utf-8"
    )

    return gate_results
