"""Phase 39: Cross-dataset result matrix — one standardized comparison table.

Builds the standardized evaluation matrix (Dataset | Domain | Training
relationship | Samples | Fraud rate | ROC-AUC | PR-AUC | Recall | FPR | Status)
from evaluation records in an EvaluationLedger or from explicit result dicts.

Design rules enforced here:
* Rows are never averaged into a single aggregate score — fundamentally
  different datasets (in-domain vs cross-domain vs external vs synthetic vs
  derived vs provenance-uncertain) are kept as separate, labeled rows.
* Every row carries its ``training_relationship`` so a reader can see at a
  glance whether the evaluation was in-domain, same-generator-family, or
  independent.
* Failed/low evaluations are rows like any other — failure-preserving.

STATUS: IMPLEMENTED
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
"""

from __future__ import annotations

from typing import Any

# Relationship taxonomy (closed set — unknown stays UNKNOWN, never upgraded).
TRAINING_RELATIONSHIPS = {
    "IN_DOMAIN": "same distribution as the training data",
    "SAME_GENERATOR_FAMILY": "different dataset, same synthetic generator family",
    "CROSS_DOMAIN": "different domain, public dataset",
    "EXTERNAL": "independently sourced dataset, not used in training",
    "UNKNOWN": "relationship not established",
}

DOMAINS = {"SYNTHETIC", "DERIVED", "REAL_WORLD_ORIGIN", "PROVENANCE_UNCERTAIN", "UNKNOWN"}

# Distribution-shift linkage (Phase 38 integration): language guidance for
# reports. Cautious wording is mandatory — shift is an observation, not a
# proven cause of poor performance.
SHIFT_CAUSATION_NOTE = (
    "Distribution-shift findings are descriptive. Where poor external "
    "performance coincides with large shift, reports say the result is "
    "'consistent with distribution shift', NOT that shift proven-ly caused it."
)


def classify_row(dataset_name: str,
                 training_relationship: str | None = None,
                 domain: str | None = None) -> dict[str, str]:
    """Classify a matrix row; unknown inputs stay UNKNOWN (never upgraded)."""
    rel = (training_relationship or "").upper()
    if rel not in TRAINING_RELATIONSHIPS:
        rel = "UNKNOWN"
    dom = (domain or "").upper()
    if dom not in DOMAINS:
        dom = "UNKNOWN"
    return {"training_relationship": rel, "domain": dom}


def build_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the standardized matrix from row dicts.

    Each row dict supports: dataset, domain, training_relationship, samples,
    fraud_rate, roc_auc, pr_auc, recall, fpr, status, evaluation_id,
    model_hash (short), source (report/artifact path). Missing metrics stay
    None — they are never imputed.
    """
    matrix_rows: list[dict[str, Any]] = []
    for r in rows:
        cls = classify_row(r.get("dataset", ""), r.get("training_relationship"),
                           r.get("domain"))
        matrix_rows.append({
            "dataset": r.get("dataset", "unknown"),
            "domain": cls["domain"],
            "training_relationship": cls["training_relationship"],
            "samples": r.get("samples"),
            "fraud_rate": r.get("fraud_rate"),
            "roc_auc": r.get("roc_auc"),
            "pr_auc": r.get("pr_auc"),
            "recall": r.get("recall"),
            "fpr": r.get("fpr"),
            "status": r.get("status", "EVALUATED"),
            "evaluation_id": r.get("evaluation_id"),
            "model_hash_short": (r.get("model_hash") or "")[:12] or None,
            "source": r.get("source"),
        })
    return {
        "matrix_version": "1.0",
        "note": ("Rows are NOT aggregated into a single score. In-domain, "
                 "cross-domain, external, synthetic, derived and "
                 "provenance-uncertain evaluations are reported separately."),
        "relationships": TRAINING_RELATIONSHIPS,
        "rows": matrix_rows,
    }


def matrix_to_markdown(matrix: dict[str, Any]) -> str:
    """Human-readable table matching README conventions."""
    hdr = ("| Dataset | Domain | Training relationship | Samples | Fraud rate "
           "| ROC-AUC | PR-AUC | Recall | FPR | Status |\n"
           "|---|---|---|---|---|---|---|---|---|---|\n")

    def fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 1000 else f"{v:.0f}"
        return str(v)

    lines = []
    for r in matrix["rows"]:
        lines.append(
            f"| {r['dataset']} | {r['domain']} | {r['training_relationship']} "
            f"| {fmt(r['samples'])} | {fmt(r['fraud_rate'])} | {fmt(r['roc_auc'])} "
            f"| {fmt(r['pr_auc'])} | {fmt(r['recall'])} | {fmt(r['fpr'])} "
            f"| {r['status']} |"
        )
    return hdr + "\n".join(lines) + "\n"
