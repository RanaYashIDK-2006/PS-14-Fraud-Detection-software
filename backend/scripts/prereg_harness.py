#!/usr/bin/env python3
"""NR-04 — preregistered benchmark harness (Gate-2 execution machinery).

Translates the frozen semantics of the preregistered protocol
(`docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md`, v0.1.2-draft) and
`docs/PHASE_NR03_EVALUATION_SEMANTICS.md` (§C matrix = experiment registry
source of truth) into executable, provenance-complete evaluation procedures.

Design principle (NR-04 §2): three states are strictly separated —
FROZEN (executable), PENDING REVIEW (refused, never silently valued),
BLOCKED (refused, never substituted). Neither PENDING REVIEW nor BLOCKED can
ever become a MEASURED result. Evidence-first (§9): a result is MEASURED only
after its v1.1 ledger record is created and validated.

This module never executes a model itself: `execute()` takes an injected
`runner`. The CLI exposes only registry/guard/matrix/validate subcommands —
there is no flag that overrides the reviewer-decision guard (§6); the only
reviewer-authorized flip points are the module-level *_APPROVED constants
below, which a reviewer decision changes in code review with a protocol
change-history note.

Deliverable doc: docs/PHASE_NR04_PREREGISTERED_HARNESS.md
Tests: backend/scripts/prereg_harness_test.py (determinism + fixtures).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_record as er                      # schema v1.1 evidence records
import metric_definitions as md               # v1.0 metric semantics
from calibration_test import compute_brier_score, compute_ece  # pinned Brier/ECE
import eval_ulb                               # pinned ranking recall@1%FPR (import-safe since NR-04)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_LEDGER = REPO_ROOT / "reports" / "evaluation_runs" / "eval_ledger.jsonl"
PROTOCOL_PATH = REPO_ROOT / "docs" / "evaluation" / "PRE_REGISTERED_EVALUATION_PROTOCOL.md"
NR03_PATH = REPO_ROOT / "docs" / "PHASE_NR03_EVALUATION_SEMANTICS.md"
CONTRACT_PATH = REPO_ROOT / "backend" / "research" / "public_feature_contract.json"

HARNESS_VERSION = "0.1"
PROTOCOL_VERSION = "0.1.2-draft"                              # bound to protocol file (test-asserted)
EVIDENCE_SCHEMA_VERSION = er.EVALUATION_RECORD_VERSION        # "1.1"
METRIC_DEFINITIONS_VERSION = md.METRIC_DEFINITIONS_VERSION    # "1.0"

# ── Reviewer-decision gates (§6): the ONLY flip points in the codebase ───────
# A reviewer-approved decision is recorded by changing the constant here AND
# in the protocol's Change history BEFORE any affected run (protocol §9.10).
# No command-line flag or caller input can set these.
PROTOCOL_SIGNED_OFF = False            # 4-role sign-off block still _pending_
SEED_POLICY_APPROVED = False           # §5 marker / T-3 (proposal 42–46 NOT approved)
SEED_AGGREGATION_APPROVED = False      # T-4 (NR-02 D-5: mean/median/pooled unset)
BOOTSTRAP_COUNT_APPROVED: int | None = None   # §2 marker / T-5 (200/1000/2000 unset)
ECE_BINNING_APPROVED: int | None = None       # §2 marker / T-6 (equal-mass vs width unset)
PAIRED_CI_SPEC_APPROVED = False        # paired-difference construction is a
                                       # statistical-review item (NR-02 §H); unset

# ── Cell statuses (§8/§18): exactly six, never collapsed ─────────────────────
MEASURED = "MEASURED"
FAILED = "FAILED"
BLOCKED = "BLOCKED"
NOT_ESTABLISHED = "NOT ESTABLISHED"
NOT_APPLICABLE = "NOT APPLICABLE"
PENDING_REVIEW = "PENDING REVIEW"
CELL_STATUSES = (MEASURED, FAILED, BLOCKED, NOT_ESTABLISHED, NOT_APPLICABLE, PENDING_REVIEW)

# Refusal codes (diagnostics — distinct from cell statuses, §5)
CODE_OK = "OK"
CODE_CONFIG = "CONFIGURATION ERROR"
CODE_PENDING = "DO NOT EXECUTE — PENDING REVIEW"
CODE_VIOLATION = "PROTOCOL VIOLATION"
CODE_EVIDENCE = "EVIDENCE FAILURE"

# Frozen NR-03 §G nine-row failed-run policy: class, rerun rule, cell outcome
FAILURE_CLASSES: dict[str, dict[str, Any]] = {
    "crash": {"class": "implementation", "rerun": True, "cell": FAILED},
    "invalid_output": {"class": "implementation/data", "rerun": True, "cell": FAILED},
    "feature_contract_violation": {"class": "invalid experiment", "rerun": False,
                                   "cell": NOT_ESTABLISHED,
                                   "note": "rerun only after contract fix, new experiment ID"},
    "evidence_failure": {"class": "evidence failure", "rerun": True, "cell": NOT_ESTABLISHED,
                         "note": "one evidence-only rerun, identical frozen config; persistent -> NOT ESTABLISHED"},
    "missing_data": {"class": "data quality", "rerun": False, "cell": FAILED},
    "nan_inf": {"class": "implementation/data", "rerun": True, "cell": FAILED},
    "resource_limit": {"class": "implementation/infra", "rerun": True, "cell": FAILED},
    "reproducibility": {"class": "evidence failure", "rerun": True, "cell": NOT_ESTABLISHED},
    "metric_domain": {"class": "data quality", "rerun": False, "cell": FAILED,
                      "note": "e.g. ROC-AUC on single-class partition — split design issue, back to reviewer"},
}


class HarnessRefusal(Exception):
    """Execution refused with a machine-readable code (never a result)."""
    code = CODE_CONFIG
    status = NOT_ESTABLISHED
    def __init__(self, message: str, *, detail: Any = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigurationError(HarnessRefusal):
    """A required frozen field is missing/unusable (§5)."""
    code = CODE_CONFIG
    status = NOT_ESTABLISHED


class PendingReview(HarnessRefusal):
    """A reviewer-owned decision is unresolved for this experiment (§6)."""
    code = CODE_PENDING
    status = PENDING_REVIEW


class Blocked(HarnessRefusal):
    """External dependency unavailable — never substituted (§15/§16)."""
    code = BLOCKED
    status = BLOCKED


class ProtocolViolation(HarnessRefusal):
    """Final-test protection tripped (§17): confirmatory status void."""
    code = CODE_VIOLATION
    status = NOT_ESTABLISHED


class EvidenceFailure(HarnessRefusal):
    """Evidence record could not be created/validated (§9) — not MEASURED."""
    code = CODE_EVIDENCE
    status = NOT_ESTABLISHED


class MetricDomainError(Exception):
    """Metric inputs outside the mathematical domain (§G rows 6/9)."""
    def __init__(self, failure_kind: str, message: str):
        super().__init__(message)
        self.failure_kind = failure_kind   # key into FAILURE_CLASSES


# ── Reviewer-decision registry (§6) ─────────────────────────────────────────
# `kind == "marker"` entries are exactly the 11 protocol
# [REQUIRES DECISION/APPROVAL] markers (NR-03 §T list; the protocol's literal
# count is verified by the test suite). Sign-off/implementation entries are
# the remaining NR-02/NR-03 open items the guard must also hold back.
# `affects`: experiment ids, or ["*"] when every experiment is affected.

PENDING_DECISIONS: list[dict[str, Any]] = [
    {"id": "M-0", "kind": "marker", "label": "§0 scope: which candidate system(s) are under evaluation (S1/S2)",
     "refs": ["protocol §0", "NR-03 §T"], "affects": ["*"]},
    {"id": "M-1", "kind": "marker", "label": "§1 split axis per dataset (temporal / entity-disjoint / both)",
     "refs": ["protocol §1", "NR-03 §T", "NR-02 D-3"], "affects": ["*"]},
    {"id": "M-2", "kind": "marker", "label": "§2 ECE bin count and binning type (equal-mass vs equal-width)",
     "refs": ["protocol §2", "NR-03 §T", "NR-03 §I"], "affects": ["E1", "E2", "E3", "E4"]},
    {"id": "M-3", "kind": "marker", "label": "§2 bootstrap CI replicate count and seed (200/1000/2000 unresolved)",
     "refs": ["protocol §2", "NR-03 §J", "NR-03 §T", "T-5"], "affects": ["*"]},
    {"id": "M-4", "kind": "marker", "label": "§2 multiplicity correction (Holm–Bonferroni vs FDR)",
     "refs": ["protocol §2", "NR-03 §T"], "affects": ["E1", "E2", "E3", "E7"]},
    {"id": "M-5", "kind": "marker", "label": "§3 all proposed MMD magnitudes",
     "refs": ["protocol §3", "NR-03 §K", "NR-03 §T"], "affects": ["E1", "E2", "E3"]},
    {"id": "M-6", "kind": "marker", "label": "§4 A/B/C conclusion wording and publication terms",
     "refs": ["protocol §4", "NR-03 §T"], "affects": ["*"]},
    {"id": "M-7", "kind": "marker", "label": "§5 seed count and seed values (proposal 42–46 NOT approved)",
     "refs": ["protocol §5", "NR-03 §F", "NR-03 §T", "T-3"], "affects": ["*"]},
    {"id": "M-8", "kind": "marker", "label": "§5 operating point (fixed FPR vs fixed alert volume)",
     "refs": ["protocol §5", "NR-03 §T"], "affects": ["E1", "E2", "E5", "E6"]},
    {"id": "M-9", "kind": "marker", "label": "§5 whether S1's locked threshold carries into external evaluation untouched",
     "refs": ["protocol §5.5", "NR-03 §T"], "affects": ["E1", "E2", "E3", "E4"]},
    {"id": "M-10", "kind": "marker", "label": "§7 analyst-capacity number (domain assumption; illustrative until bank input)",
     "refs": ["protocol §7", "NR-03 §T"], "affects": ["E6"]},
    {"id": "PD-SEED-AGG", "kind": "signoff", "label": "Across-seed primary aggregation (mean vs median vs pooled) undecided (T-4)",
     "refs": ["NR-02 D-5", "NR-03 §F", "T-4"], "affects": ["E1", "E2", "E3", "E4", "E7"]},
    {"id": "PD-PAIRED", "kind": "implementation",
     "label": "Paired-difference CI construction not implemented/approved — prerequisite for E1/E2/E3",
     "refs": ["NR-02 §H", "NR-03 §J"], "affects": ["E1", "E2", "E3"]},
    {"id": "PD-D1", "kind": "signoff", "label": "D-1 dataset for criteria 1–2 unnamed (tied to §0/§1)",
     "refs": ["NR-02 D-1", "NR-03 §D", "T-1"], "affects": ["E1"]},
    {"id": "PD-D1B", "kind": "signoff", "label": "D-1b dataset for criteria 3–4 (7 gating conditions) unnamed",
     "refs": ["NR-03 §D", "T-2"], "affects": ["E2"]},
    {"id": "PD-T7", "kind": "signoff", "label": "Wrong-confident decile population unspecified (C3 input)",
     "refs": ["NR-03 §I", "T-7"], "affects": ["E2"]},
    {"id": "PD-T8", "kind": "signoff", "label": "External set confirmatory status given prior promotion exposure",
     "refs": ["NR-03 §L", "T-8"], "affects": ["E3", "E4"]},
    {"id": "PD-T9", "kind": "signoff", "label": "External split boundaries (temporal + entity-safe) not specified anywhere",
     "refs": ["NR-03 §D", "T-9"], "affects": ["E3", "E4"]},
    {"id": "PD-T10", "kind": "signoff", "label": "In-domain reference identity for C5 retention undecided (tracks D-1)",
     "refs": ["NR-03 §K", "T-10"], "affects": ["E3"]},
    {"id": "PD-T11", "kind": "signoff", "label": "Preprocessing-freeze / external-set-selection rules acknowledgment (§E/§L)",
     "refs": ["NR-03 §E/§L", "T-11"], "affects": ["*"]},
    {"id": "PD-SIGNOFF", "kind": "signoff", "label": "Protocol is DRAFT/NOT APPROVED — 4-role sign-off block all _pending_",
     "refs": ["protocol header", "NR-02 §N"], "affects": ["*"]},
    {"id": "PD-FREEZE", "kind": "implementation",
     "label": "Final-test freeze record does not exist yet (protocol §9.2) — required before final tiers",
     "refs": ["protocol §9", "NR-03 §S"], "affects": ["*"],
     "tiers": ["final_test", "final_external"]},
]

MARKER_IDS = [d["id"] for d in PENDING_DECISIONS if d["kind"] == "marker"]


def pending_for(experiment_id: str, *, tier: str = "development") -> list[dict[str, Any]]:
    """Decisions that gate this experiment (and this data tier)."""
    out = []
    for d in PENDING_DECISIONS:
        if experiment_id not in d["affects"] and "*" not in d["affects"]:
            continue
        if "tiers" in d and tier not in d["tiers"]:
            continue
        out.append(d)
    return out


# ── Experiment registry (§4): NR-03 §C matrix is the source of truth ────────
# Statuses: F=FROZEN, P=PENDING REVIEW, B=BLOCKED, N=NOT APPLICABLE.
# The test suite parses the actual table out of docs/PHASE_NR03_EVALUATION_
# SEMANTICS.md and asserts this transcription matches it cell-for-cell.
MATRIX_FIELD_ORDER = [
    "Experiment ID", "Research question", "Dataset", "Dataset version/hash",
    "Population", "Split", "Train population", "Validation population",
    "Final-test population", "Seed policy", "Model", "Feature contract",
    "Preprocessing", "Calibration", "Threshold policy", "Primary metric",
    "Secondary metrics", "Uncertainty method", "Statistical test (if applicable)",
    "Selection data", "Locked data", "Evidence artifact", "Failure rule",
]
NR03_GRID: dict[str, list[str]] = {
    #            E1  E2  E3  E4  E5  E6  E7   (NR-03 §C table, verbatim)
    "Experiment ID":                 ["F", "F", "F", "F", "F", "F", "F"],
    "Research question":             ["F", "F", "F", "F", "F", "F", "F"],
    "Dataset":                       ["P", "P", "F", "F", "P", "P", "P"],
    "Dataset version/hash":          ["P", "P", "F", "F", "P", "P", "P"],
    "Population":                    ["P", "P", "F", "F", "P", "P", "P"],
    "Split":                         ["P", "P", "P", "P", "P", "P", "P"],
    "Train population":              ["P", "P", "P", "N", "N", "N", "P"],
    "Validation population":         ["P", "P", "N", "N", "N", "N", "P"],
    "Final-test population":         ["P", "P", "F", "F", "P", "P", "P"],
    "Seed policy":                   ["P", "P", "P", "P", "P", "P", "P"],
    "Model":                         ["F", "F", "F", "F", "F", "F", "F"],
    "Feature contract":              ["P", "P", "P", "P", "P", "P", "P"],
    "Preprocessing":                 ["F", "F", "F", "F", "F", "F", "F"],
    "Calibration":                   ["F", "F", "F", "F", "N", "N", "N"],
    "Threshold policy":              ["F", "F", "F", "F", "F", "F", "F"],
    "Primary metric":                ["F", "F", "F", "P", "F", "F", "F"],
    "Secondary metrics":             ["F", "F", "F", "P", "F", "F", "F"],
    "Uncertainty method":            ["P", "P", "P", "P", "P", "P", "P"],
    "Statistical test (if applicable)": ["P", "P", "P", "P", "N", "N", "P"],
    "Selection data":                ["F", "F", "F", "F", "F", "F", "F"],
    "Locked data":                   ["F", "F", "F", "F", "F", "F", "F"],
    "Evidence artifact":             ["F", "F", "F", "F", "F", "F", "F"],
    "Failure rule":                  ["F", "F", "F", "F", "F", "F", "F"],
}
EXPERIMENT_IDS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]

_EXPERIMENT_META: dict[str, dict[str, str]] = {
    "E1": {"purpose": "Ensemble contribution vs best single constituent (criteria 1/2)",
           "criteria": ["C1", "C2"]},
    "E2": {"purpose": "Gating effectiveness across the 7 pre-defined conditions (criteria 3/4)",
           "criteria": ["C3", "C4"]},
    "E3": {"purpose": "External-transfer evaluation of the frozen candidate (criteria 5/6)",
           "criteria": ["C5", "C6"]},
    "E4": {"purpose": "External calibration (Brier/ECE/reliability on the external split)",
           "criteria": []},
    "E5": {"purpose": "Subgroup/segment descriptive analysis (protocol §6)",
           "criteria": []},
    "E6": {"purpose": "Business metrics — recall@top-k, burden, cost-ratio curves (protocol §7)",
           "criteria": []},
    "E7": {"purpose": "Baselines & ablations (reference lines supporting C1/C2)",
           "criteria": []},
}

# Frozen values for FROZEN cells (P/N cells deliberately have NO value —
# "Do not invent values for cells marked PENDING REVIEW"). Sources are the
# protocol/NR-03 sections named in each string.
_FROZEN_VALUES: dict[str, dict[str, str]] = {
    "E1": {
        "Primary metric": "PR-AUC = average precision (metric_definitions.pr_auc, NR-03 §I semantic pin)",
        "Secondary metrics": "recall@1%FPR (ranking variant, eval_ulb.recall_at_fpr pin), Brier/ECE (provisional per protocol §2)",
    },
    "E2": {
        "Primary metric": "wrong-confident reduction at matched coverage (C3; decile population pending T-7 — see PD-T7)",
        "Secondary metrics": "selective risk curves, FP/FN counts, coverage (protocol C3/§6)",
    },
    "E3": {
        "Dataset": "kaggle_fraud_test (data/kaggle_fraud/fraudTest.csv — final external, protocol §9)",
        "Dataset version/hash": "12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0",
        "Population": "final external tier: evaluation only, never selection (protocol §9 data tiers)",
        "Primary metric": "external ROC-AUC + retention ratio (C5; NR-03 §K)",
        "Secondary metrics": "PR-AUC, Brier/ECE (provisional), degradation narrative",
    },
    "E4": {
        "Dataset": "kaggle_fraud_test (data/kaggle_fraud/fraudTest.csv — final external, protocol §9)",
        "Dataset version/hash": "12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0",
        "Population": "final external tier: evaluation only, never selection (protocol §9 data tiers)",
    },
    "E5": {
        "Primary metric": "none (descriptive) — per-segment PR-AUC / recall@1%FPR / alert rate (protocol §6)",
        "Secondary metrics": "bootstrap CIs per segment; small-n segments flagged (NR-03 §P)",
    },
    "E6": {
        "Primary metric": "none (descriptive) — recall@top-k curves (protocol §7)",
        "Secondary metrics": "false-positive burden, cost-ratio parameterized curves (never realized benefit)",
    },
    "E7": {
        "Primary metric": "reference lines supporting E1/C1 (majority/random baselines, metric_definitions)",
        "Secondary metrics": "baseline/ablation tables (NR-03 §P)",
    },
}

_COMMON_FROZEN = {
    "Model": "pre-registered candidate(s); all weights frozen with the freeze record (protocol §9.1); candidate scope = §0 marker M-0",
    "Preprocessing": ("fit-train-only, fit-once-per-split (NR-03 §H); vocabulary: eval_ulb=standardscaler_fit_on_train/"
                      "ulb_pca30_research, train_compare=standardscaler_fit_on_train/ps14_synthetic_48_native, "
                      "cross_dataset=embedded_in_artifacts/production_48_native, calibration=none_fixed_artifacts/none_fixed_scores"),
    "Calibration": "Platt fitted on validation only (protocol §5.2; NR-03 §H)",
    "Threshold policy": ("train→validation→freeze→single-use test (protocol §5); ranking primaries are threshold-free; "
                         "test/external/holdout threshold sources refused (evaluate.py:65 refuse_test_tuning)"),
    "Selection data": "validation only — calibration, threshold, constituent and hyperparameter selection (protocol §1)",
    "Final-test population": ("designated final-test partition of the preregistered dataset, "
                              "locked by the freeze record (protocol §9.1) — partition identity "
                              "resolves when D-1/§1 are decided"),
    "Locked data": ("final-test partition (single use, protocol §9) + final external fraudTest.csv "
                    "(evaluation only, never selection)"),
    "Evidence artifact": ("v1.1 EvaluationRecord appended to reports/evaluation_runs/eval_ledger.jsonl + "
                          "record_<id>.json sidecar (schema per eval_record_test 22/22)"),
    "Failure rule": "NR-03 §G nine-row policy; protocol §9.6–9.8 — failures preserved append-only, rerun identical-config only",
    "Experiment ID": "",
    "Research question": "",
}


def build_experiments() -> dict[str, dict[str, Any]]:
    """Assemble the 7-experiment registry from grid + frozen values."""
    exps: dict[str, dict[str, Any]] = {}
    for i, eid in enumerate(EXPERIMENT_IDS):
        meta = _EXPERIMENT_META[eid]
        vals = dict(_COMMON_FROZEN)
        vals["Experiment ID"] = eid
        vals["Research question"] = meta["purpose"]
        vals.update(_FROZEN_VALUES.get(eid, {}))
        fields: dict[str, dict[str, Any]] = {}
        for fname in MATRIX_FIELD_ORDER:
            st = NR03_GRID[fname][i]
            status = {"F": "FROZEN", "P": "PENDING REVIEW",
                      "B": "BLOCKED", "N": "NOT APPLICABLE"}[st]
            value = vals.get(fname) if st == "F" else None
            if st == "N":
                value = "not applicable to this experiment (NR-03 §C)"
            fields[fname] = {"status": status, "value": value}
        exps[eid] = {"experiment_id": eid, "purpose": meta["purpose"],
                     "criteria": meta["criteria"], "fields": fields}
    return exps


EXPERIMENTS = build_experiments()

# ── Harness contract (§3): one authoritative contract, deterministic sources ─
# (name, source) — versions are pinned here and must equal caller values;
# git SHA is captured by the harness from the repository, never caller-typed.
CONTRACT_FIELDS: list[tuple[str, str]] = [
    ("protocol_version", "docs/evaluation/PRE_REGISTERED_EVALUATION_PROTOCOL.md version note (must equal %s)" % PROTOCOL_VERSION),
    ("experiment_id", "experiment registry (this file, from NR-03 §C)"),
    ("dataset_name", "dataset registry (this file)"),
    ("dataset_path", "repository path of the dataset file"),
    ("dataset_expected_sha256", "pinned dataset hash (protocol §9 / NR-03 §D / BASELINE_REPRODUCTION §2)"),
    ("dataset_role", "dataset registry role (candidate / final_external / development)"),
    ("split", "preregistered split spec {name, axis, role, ...} (§1 marker — PENDING until signed)"),
    ("population", "experiment registry field 'Population' / preregistered population statement"),
    ("seed", "SeedPlan seed for this replicate (§5 marker — PENDING until approved)"),
    ("model_identity", "human label of the frozen candidate system (§0 marker selects scope)"),
    ("artifact_paths", "model artifact files hashed by eval_record.artifact_hashes"),
    ("feature_contract_path", "contract file (backend/research/public_feature_contract.json etc.) + hash"),
    ("preprocessing_identity", "NR-03 §H per-experiment preprocessing vocabulary"),
    ("preprocessing_fit_population", "NR-03 §H: must be 'train'"),
    ("calibration_config", "{procedure, fit_population} — Platt on validation (protocol §5.2)"),
    ("threshold_config", "ThresholdSpec (value/source/selected_on) — protocol §5 freeze states"),
    ("metric_definitions_version", "metric_definitions.METRIC_DEFINITIONS_VERSION (must equal %s)" % METRIC_DEFINITIONS_VERSION),
    ("bootstrap_config", "{n_bootstrap, seed} — explicit; approval tracked by BOOTSTRAP_COUNT_APPROVED"),
    ("statistical_test_config", "{paired_difference_ci: ...} — PD-PAIRED until approved"),
    ("acceptance_criterion_config", "MMD rule ids from protocol §3 / NR-03 §K (C1/C3/C5...)"),
    ("evidence_schema_version", "eval_record.EVALUATION_RECORD_VERSION (must equal %s)" % EVIDENCE_SCHEMA_VERSION),
    ("execution_command", "exact invocation string recorded in the evidence record (v1.1 command)"),
    ("data_tier", "development | validation | final_test | final_external (protocol §9 tiers)"),
    ("deviations", "list of dated deviation records (protocol §9.10); empty list is valid"),
]
REQUIRED_CONTRACT_KEYS = [k for k, _ in CONTRACT_FIELDS]


def validate_contract(contract: dict[str, Any]) -> list[str]:
    """Frozen-field validation (§5). Returns CONFIGURATION ERROR diagnostics.

    Missing field / version drift → configuration error (execution stops).
    Present-but-pending fields are NOT errors here — they surface through the
    reviewer-decision guard as PENDING REVIEW (a different state, §2).
    """
    errors: list[str] = []
    for key, source in CONTRACT_FIELDS:
        if key not in contract or contract[key] is None:
            errors.append(f"missing frozen field '{key}' (deterministic source: {source})")
    if contract.get("protocol_version") not in (None, PROTOCOL_VERSION):
        errors.append(
            f"protocol_version drift: contract={contract.get('protocol_version')!r} "
            f"harness={PROTOCOL_VERSION!r} — protocol edits require a Change-history entry")
    if contract.get("metric_definitions_version") not in (None, METRIC_DEFINITIONS_VERSION):
        errors.append("metric_definitions_version drift vs pinned %s" % METRIC_DEFINITIONS_VERSION)
    if contract.get("evidence_schema_version") not in (None, EVIDENCE_SCHEMA_VERSION):
        errors.append("evidence_schema_version drift vs pinned %s" % EVIDENCE_SCHEMA_VERSION)
    exp = contract.get("experiment_id")
    if exp is not None and exp not in EXPERIMENTS:
        errors.append(f"unknown experiment_id {exp!r} — registry holds {EXPERIMENT_IDS}")
    seeds = contract.get("seed")
    if isinstance(seeds, list):
        if not seeds or any(not isinstance(s, int) for s in seeds) or len(set(seeds)) != len(seeds):
            errors.append("seed list must be non-empty unique integers")
    elif seeds is not None and not isinstance(seeds, int):
        errors.append("seed must be an int or a list of ints (SeedPlan)")
    return errors


@dataclass
class Decision:
    """Guard verdict for one requested execution (§6)."""
    experiment_id: str
    code: str
    config_errors: list[str] = dc_field(default_factory=list)
    blocked: list[str] = dc_field(default_factory=list)
    pending: list[dict[str, Any]] = dc_field(default_factory=list)
    violations: list[str] = dc_field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.code == CODE_OK

    @property
    def message(self) -> str:
        if self.allowed:
            return f"{self.experiment_id}: CLEAR TO EXECUTE (frozen fields validated)"
        if self.code == CODE_CONFIG:
            return f"{self.experiment_id}: {CODE_CONFIG} — " + "; ".join(self.config_errors)
        if self.code == BLOCKED:
            return f"{self.experiment_id}: {self.blocked[0]}"
        if self.code == CODE_VIOLATION:
            return f"{self.experiment_id}: {CODE_VIOLATION} — " + "; ".join(self.violations)
        ids = ", ".join(d["id"] for d in self.pending)
        return f"{self.experiment_id}: {CODE_PENDING} — pending: {ids}"

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "code": self.code,
                "allowed": self.allowed, "message": self.message,
                "config_errors": self.config_errors, "blocked": self.blocked,
                "pending": [d["id"] for d in self.pending],
                "violations": self.violations}


def guard(experiment_id: str, contract: dict[str, Any] | None = None,
          *, tier: str | None = None, ledger_records: list[dict] | None = None) -> Decision:
    """Reviewer-decision guard + frozen-field validation (§5/§6/§17).

    Precedence: CONFIGURATION ERROR > BLOCKED > PROTOCOL VIOLATION >
    PENDING REVIEW > CLEAR — each state stays distinct, never merged.
    """
    if experiment_id not in EXPERIMENTS:
        raise ConfigurationError(f"unknown experiment_id {experiment_id!r}")
    contract = contract or {}
    eff_tier = tier or contract.get("data_tier") or "development"
    cfg = validate_contract(contract) if contract else []
    blocked: list[str] = []
    violations: list[str] = []
    if contract:
        # dataset identity (§15) — unavailable/hash-mismatch/contract failures
        if contract.get("dataset_name") is not None:
            try:
                validate_dataset(contract["dataset_name"],
                                 path=contract.get("dataset_path"),
                                 expected_sha256=contract.get("dataset_expected_sha256"))
            except Blocked as b:
                blocked.append(b.message)
        # feature contract (§15 / N-02 pre-run check)
        if contract.get("feature_contract_path") is not None:
            try:
                validate_feature_contract(contract["feature_contract_path"])
            except Blocked as b:
                blocked.append(b.message)
        # split provability (§16) — structural checks only; identity is pending M-1
        if contract.get("split") is not None:
            try:
                validate_split(contract["split"], contract.get("partitions") or {})
            except Blocked as b:
                blocked.append(b.message)
        # final-test protection (§17)
        violations.extend(final_test_protection(contract, tier=eff_tier,
                                                ledger_records=ledger_records))
    pending = pending_for(experiment_id, tier=eff_tier)
    if cfg:
        return Decision(experiment_id, CODE_CONFIG, config_errors=cfg)
    if blocked:
        return Decision(experiment_id, BLOCKED, blocked=blocked, pending=pending)
    if violations:
        return Decision(experiment_id, CODE_VIOLATION, violations=violations, pending=pending)
    if pending:
        return Decision(experiment_id, CODE_PENDING, pending=pending)
    return Decision(experiment_id, CODE_OK)


# ── Seed harness (§7) ───────────────────────────────────────────────────────
PROPOSED_SEEDS = (42, 43, 44, 45, 46)   # NR-03 §F proposal — NOT approved


class SeedPlan:
    """Explicit seed list with deterministic propagation (§7).

    The proposed 42–46 five-seed policy stays PENDING REVIEW until
    SEED_POLICY_APPROVED flips; the plan itself never decides aggregation
    (SEED_AGGREGATION_APPROVED stays false — NR-02 D-5 / T-4).
    """

    def __init__(self, seeds: Iterable[int]):
        self.seeds: tuple[int, ...] = tuple(int(s) for s in seeds)
        if not self.seeds:
            raise ConfigurationError("seed list must be non-empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigurationError(f"duplicate seeds not allowed: {self.seeds}")

    @property
    def approved(self) -> bool:
        return SEED_POLICY_APPROVED

    def pending_decisions(self) -> list[dict[str, Any]]:
        if self.approved and SEED_AGGREGATION_APPROVED:
            return []
        return [d for d in PENDING_DECISIONS if d["id"] in ("M-7", "PD-SEED-AGG")]

    def to_config(self) -> dict[str, Any]:
        return {"seeds": list(self.seeds), "policy_approved": self.approved,
                "aggregation_approved": SEED_AGGREGATION_APPROVED,
                "proposal_not_approved": list(PROPOSED_SEEDS)}


# ── Dataset registry (§15) ─────────────────────────────────────────────────
# Hashes pinned by NR-03 §D / protocol §9 / BASELINE_REPRODUCTION §2.
DATASETS: dict[str, dict[str, Any]] = {
    "ulb": {"path": "data/creditcard.csv", "label_column": "Class",
            "sha256": "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89",
            "role": "in_domain_candidate"},
    "ibm_v2": {"path": "data/credit_card_transactions-ibm_v2.csv", "label_column": "Is Fraud?",
               "sha256": "b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15",
               "role": "entity_disjoint_candidate"},
    "kaggle_fraud_test": {"path": "data/kaggle_fraud/fraudTest.csv", "label_column": "is_fraud",
                          "sha256": "12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0",
                          "role": "final_external_locked"},
    "kaggle_fraud_train": {"path": "data/kaggle_fraud/fraudTrain.csv", "label_column": "is_fraud",
                           "sha256": "fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d",
                           "role": "external_train_development_only"},
    "synthetic_ps14": {"path": "data/transactions.csv", "label_column": "label",
                       "sha256": "9f0f56bf0549fccf0c6335ece1514d1ae515a57f809a72658ce5fcaf48525d63",
                       "role": "synthetic_development"},
    "baf": {"path": "data/external_benchmark/", "sha256": None, "status": BLOCKED,
            "blocker": "NeurIPS 2022 BAF not acquired — download approval + license/semantics review (NR-16)"},
}


def validate_dataset(name: str, *, path: str | Path | None = None,
                     expected_sha256: str | None = None,
                     check_label: bool = True) -> dict[str, Any]:
    """Identity/hash/schema validation (§15). Never substitutes a dataset.

    Raises Blocked with one of the mandated diagnostics:
    BLOCKED — DATASET UNAVAILABLE / DATASET IDENTITY MISMATCH (plus schema
    failure reported under the unavailable status — no silent downgrade).
    """
    spec = DATASETS.get(name)
    if spec is None and path is None:
        raise ConfigurationError(
            f"dataset '{name}' unknown to the registry and no path supplied")
    spec = spec or {}
    if spec.get("status") == BLOCKED:
        raise Blocked(f"BLOCKED — DATASET UNAVAILABLE: {name} ({spec.get('blocker')})")
    fpath = Path(path) if path is not None else REPO_ROOT / spec["path"]
    pin = expected_sha256 or spec.get("sha256")
    if not fpath.exists():
        raise Blocked(f"BLOCKED — DATASET UNAVAILABLE: {name} — file missing: {fpath}")
    actual = er.sha256_file(fpath) if fpath.is_file() else None
    if pin and actual != pin:
        raise Blocked(
            f"DATASET IDENTITY MISMATCH: {name} — pinned sha256 {pin[:16]}… "
            f"but file hashes to {str(actual)[:16]}… — refusing to proceed")
    label_col = spec.get("label_column")
    if check_label and label_col and fpath.is_file():
        import csv
        with open(fpath, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
        if label_col not in header:
            raise Blocked(
                f"BLOCKED — DATASET UNAVAILABLE: {name} schema invalid — "
                f"label column {label_col!r} not found in header")
    return {"name": name, "path": str(fpath), "sha256": actual,
            "role": spec.get("role"), "label_column": label_col}


# ── Feature-contract validation (§15, incl. the N-02 pre-run check) ─────────
def validate_feature_contract(path: str | Path = CONTRACT_PATH) -> dict[str, Any]:
    """Contract identity + internal consistency. Failure = BLOCKED — FEATURE
    CONTRACT (never a silent downgrade, never an invented mapping)."""
    p = Path(path)
    if not p.exists():
        raise Blocked(f"BLOCKED — FEATURE CONTRACT: contract file missing: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Blocked(f"BLOCKED — FEATURE CONTRACT: {p.name} does not parse: {e}") from None
    feats = doc.get("features")
    if not isinstance(feats, (dict, list)):
        raise Blocked(f"BLOCKED — FEATURE CONTRACT: {p.name} has no 'features' map")
    declared = doc.get("public_features_count")
    if declared != len(feats):
        raise Blocked(
            f"BLOCKED — FEATURE CONTRACT: {p.name} declares public_features_count={declared} "
            f"but lists {len(feats)} features (N-02 invariant)")
    missing = doc.get("missing_from_public") or {}
    mlist = missing.get("features")
    if isinstance(mlist, list) and missing.get("count") != len(mlist):
        raise Blocked(
            f"BLOCKED — FEATURE CONTRACT: {p.name} declares missing count={missing.get('count')} "
            f"but lists {len(mlist)} missing features")
    required_keys = {"description", "available_in", "type"}
    items = feats.items() if isinstance(feats, dict) else enumerate(feats)
    for fname, entry in items:
        if not isinstance(entry, dict) or not required_keys <= set(entry):
            raise Blocked(
                f"BLOCKED — FEATURE CONTRACT: feature {fname!r} missing "
                f"{sorted(required_keys - set(entry if isinstance(entry, dict) else {}))}")
    return {"path": str(p), "name": p.name, "sha256": er.sha256_file(p),
            "version": doc.get("research_feature_version"), "count": declared}


# ── Split validation (§16) ──────────────────────────────────────────────────
SPLIT_AXES = ("temporal", "entity", "random")


def validate_split(split_spec: dict[str, Any],
                   partitions: dict[str, Iterable[Any]] | None = None,
                   *, timestamps: dict[str, Iterable] | None = None,
                   entities: dict[str, Iterable] | None = None) -> dict[str, Any]:
    """Structural proof that the requested split satisfies its declaration.

    Checks: identity fields present; partition membership pairwise disjoint
    (no train/test or validation/test overlap); chronology when axis=temporal;
    entity separation when axis=entity. Unprovable claims raise
    BLOCKED — SPLIT VALIDATION; the run never executes as measured.
    """
    if not isinstance(split_spec, dict) or not split_spec.get("name"):
        raise Blocked("BLOCKED — SPLIT VALIDATION: split identity (name) missing")
    axis = split_spec.get("axis")
    if axis not in SPLIT_AXES:
        raise Blocked(
            f"BLOCKED — SPLIT VALIDATION: axis must be one of {SPLIT_AXES}, got {axis!r}")
    checks: dict[str, Any] = {"name": split_spec["name"], "axis": axis,
                              "disjoint": None, "chronology": None,
                              "entity_separation": None}
    parts = {k: list(v) for k, v in (partitions or {}).items()}
    if parts:
        names = sorted(parts)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = set(parts[a]) & set(parts[b])
                if overlap:
                    raise Blocked(
                        f"BLOCKED — SPLIT VALIDATION: partitions {a!r} and {b!r} overlap "
                        f"on {len(overlap)} rows (train/test or validation/test leakage)")
        checks["disjoint"] = True
        checks["partitions"] = {k: len(v) for k, v in parts.items()}
    if axis == "temporal":
        ts = {k: list(v) for k, v in (timestamps or {}).items()}
        if not {"train", "test"} <= set(ts):
            raise Blocked(
                "BLOCKED — SPLIT VALIDATION: temporal axis claimed but per-partition "
                "timestamps not provided — chronology cannot be proven")
        if max(ts["train"]) > min(ts["test"]):
            raise Blocked(
                "BLOCKED — SPLIT VALIDATION: temporal ordering violated "
                "(max(train) > min(test))")
        if "validation" in ts and max(ts["validation"]) > min(ts["test"]):
            raise Blocked(
                "BLOCKED — SPLIT VALIDATION: temporal ordering violated "
                "(max(validation) > min(test))")
        checks["chronology"] = True
    if axis == "entity":
        ent = {k: set(v) for k, v in (entities or {}).items()}
        if not {"train", "test"} <= set(ent):
            raise Blocked(
                "BLOCKED — SPLIT VALIDATION: entity axis claimed but per-partition "
                "entity ids not provided — entity separation cannot be proven")
        names = sorted(ent)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if ent[a] & ent[b]:
                    raise Blocked(
                        f"BLOCKED — SPLIT VALIDATION: entity leakage between {a!r} and {b!r} "
                        f"({len(ent[a] & ent[b])} shared entities)")
        checks["entity_separation"] = True
    if split_spec.get("role") and split_spec.get("expected_role") and \
            split_spec["role"] != split_spec["expected_role"]:
        raise Blocked(
            f"BLOCKED — SPLIT VALIDATION: split role {split_spec['role']!r} does not match "
            f"the preregistered role {split_spec['expected_role']!r}")
    return checks


# ── Threshold freeze (§14) ─────────────────────────────────────────────────
THRESHOLD_STATES = ("frozen", "validation_selected", "exploratory", "pending")
_FORBIDDEN_SOURCES = ("test", "external", "holdout")   # parity with evaluate.py:65


@dataclass
class ThresholdSpec:
    """Four-state threshold provenance (§14)."""
    value: float | None
    source: str            # "frozen" | "validation" | "exploratory" | "pending"
    selected_on: str = "none"   # "train"|"validation"|"final_test"|"external"|"none"|"unknown"

    @property
    def state(self) -> str:
        if self.source == "pending" or self.selected_on == "unknown":
            return "pending"
        if self.source == "frozen":
            return "frozen"
        if self.source == "validation":
            return "validation_selected"
        if self.source == "exploratory":
            return "exploratory"
        return "pending"


def validate_threshold(thr: ThresholdSpec | dict[str, Any] | None, *,
                       tier: str = "development", confirmatory: bool = False) -> str:
    """Refuse test-derived thresholds; classify threshold state (§14).

    - selected_on in {final_test, external} → PROTOCOL VIOLATION (§17).
    - source in {test, external, holdout} → PROTOCOL VIOLATION (parity with
      refuse_test_tuning — the harness never tunes against evaluation data).
    - exploratory threshold on a final tier → PROTOCOL VIOLATION.
    - pending threshold on a confirmatory preregistered run → PENDING REVIEW.
    The harness contains no threshold optimizer of any kind (§14/§28).
    """
    if thr is None:
        return "none"
    if isinstance(thr, dict):
        thr = ThresholdSpec(value=thr.get("value"), source=thr.get("source", "pending"),
                            selected_on=thr.get("selected_on", "unknown"))
    state = thr.state
    src = (thr.source or "").lower()
    if src in _FORBIDDEN_SOURCES:
        raise ProtocolViolation(
            f"{CODE_VIOLATION}: threshold_source={thr.source!r} would tune the operating "
            "threshold on the evaluation data itself (parity with refuse_test_tuning)")
    if thr.selected_on in ("final_test", "external"):
        raise ProtocolViolation(
            f"{CODE_VIOLATION}: threshold selected on {thr.selected_on!r} observations — "
            "final-test protection forbids test-derived threshold selection (§17)")
    if state == "exploratory" and tier in ("final_test", "final_external"):
        raise ProtocolViolation(
            f"{CODE_VIOLATION}: exploratory threshold cannot be used on tier {tier!r} — "
            "only frozen/validation-selected thresholds on final tiers")
    if state == "pending" and confirmatory:
        raise PendingReview(
            "threshold selection semantics unresolved (pending threshold) — "
            "operating-point marker M-8 / §5 discipline")
    return state


# ── Final-test protection (§17) ─────────────────────────────────────────────
def final_test_protection(contract: dict[str, Any], *, tier: str = "development",
                          ledger_records: list[dict] | None = None) -> list[str]:
    """Return violation diagnostics (empty = clean). Detects: test-derived
    threshold/calibration/preprocessing, selective seed removal, post-hoc
    dataset selection, and final-test execution before the freeze exists."""
    v: list[str] = []
    thr = contract.get("threshold_config")
    if thr is not None:
        try:
            validate_threshold(thr, tier=tier)
        except ProtocolViolation as pv:
            v.append(pv.message)
    fit = contract.get("preprocessing_fit_population")
    if fit is None:
        v.append("preprocessing fit population not declared — cannot prove no test-derived preprocessing")
    elif fit != "train":
        v.append(f"preprocessing fit population {fit!r} != 'train' (NR-03 §H freeze)")
    cal = contract.get("calibration_config") or {}
    cal_fit = cal.get("fit_population") if isinstance(cal, dict) else None
    if tier in ("final_test", "final_external"):
        if cal_fit not in ("validation", "none", None):
            v.append(f"calibration fit population {cal_fit!r} != 'validation' (protocol §5.2)")
        declared = contract.get("declared_seeds")
        seeds = contract.get("seed")
        if declared and seeds is not None:
            got = set(seeds) if isinstance(seeds, (list, tuple, set)) else {seeds}
            if set(declared) != got:
                v.append(
                    f"seed set {sorted(got)} differs from pre-declared {sorted(set(declared))} "
                    "— selective seed removal/addition is forbidden (§9.9)")
        frozen_ds = contract.get("preregistered_dataset_name")
        if frozen_ds and contract.get("dataset_name") and \
                frozen_ds != contract.get("dataset_name"):
            v.append(
                f"post-hoc dataset selection: requested {contract.get('dataset_name')!r} but "
                f"preregistered dataset is {frozen_ds!r} (§9.4)")
        if not any((r.get("evaluation_config") or {}).get("final_test_freeze")
                   for r in (ledger_records or [])):
            v.append("final-test freeze record does not exist — executing a final test before "
                     "the freeze is a PROTOCOL VIOLATION (protocol §9.2/§9.5)")
    return v


# ── Metric harness (§13): one authoritative path, NR-03 §I semantics ────────
def validate_metric_inputs(y: np.ndarray, scores: np.ndarray) -> None:
    y = np.asarray(y)
    scores = np.asarray(scores)
    if y.shape != scores.shape:
        raise MetricDomainError("metric_domain",
                                f"labels {y.shape} vs scores {scores.shape} shape mismatch")
    if not np.all(np.isfinite(scores)):
        raise MetricDomainError("nan_inf", "NaN/Inf in scores — FAILED per NR-03 §G row 6")
    uniq = np.unique(y)
    if not set(uniq.tolist()) <= {0, 1}:
        raise MetricDomainError("metric_domain", f"labels must be 0/1, found {uniq.tolist()}")
    if len(uniq) < 2:
        raise MetricDomainError(
            "metric_domain",
            "single-class partition — ROC-AUC/PR-AUC undefined (NR-03 §G row 9: FAILED, "
            "split design issue -> back to reviewer)")


def compute_preregistered_metrics(y: np.ndarray, scores: np.ndarray, *,
                                  threshold: float | None = None,
                                  ci: bool = False,
                                  n_bootstrap: int | None = None,
                                  bootstrap_seed: int = 42,
                                  mode: str = "development") -> dict[str, Any]:
    """The authoritative preregistered metric path (§13).

    Exact frozen semantics: ROC-AUC/PR-AUC via metric_definitions (PR-AUC =
    average precision, NOT trapezoidal); recall@1%FPR via eval_ulb's pinned
    ranking variant (prepend-(0,0) + first FPR >= 0.01 via searchsorted —
    never the threshold variant at metric_definitions:161); Brier/ECE via the
    governing calibration_test implementations (ECE binning still PENDING —
    results labeled provisional). CI plumbing requires an explicit
    n_bootstrap (§11); preregistered mode additionally requires approval.
    """
    validate_metric_inputs(y, scores)
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    out: dict[str, Any] = {
        "roc_auc": float(md.roc_auc(y, scores)),
        "pr_auc": float(md.pr_auc(y, scores)),
        "recall_at_1pct_fpr": float(eval_ulb.recall_at_fpr(y, scores, 0.01)),
        "brier": float(compute_brier_score(y, scores)),
        "ece": {"value": float(compute_ece(y, scores, n_bins=10)),
                "provisional": True,
                "implementation": "calibration_test.compute_ece (10 equal-width bins)",
                "binning_status": "PENDING REVIEW (protocol §2 marker M-2; ECE_BINNING_APPROVED unset)"},
        "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
    }
    if threshold is not None:
        tp, fp, tn, fn = md.confusion_counts(y, scores, float(threshold))
        n_pos, n_neg = tp + fn, tn + fp
        out["at_threshold"] = {
            "threshold": float(threshold),
            "recall": tp / n_pos if n_pos else None,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "fpr": fp / n_neg if n_neg else None,
        }
    if ci:
        cfg = resolve_bootstrap(n_bootstrap=n_bootstrap, seed=bootstrap_seed, mode=mode)
        out["ci"] = {
            "roc_auc": md.bootstrap_ci(md.roc_auc, y, scores,
                                       n_bootstrap=cfg["n_bootstrap"], seed=cfg["seed"]).to_dict(),
            "pr_auc": md.bootstrap_ci(md.pr_auc, y, scores,
                                      n_bootstrap=cfg["n_bootstrap"], seed=cfg["seed"]).to_dict(),
            **cfg,
        }
    return out


# ── Bootstrap reconciliation (§11): three implementations, no silent default ─
BOOTSTRAP_IMPLEMENTATIONS: list[dict[str, Any]] = [
    {"id": "metric_definitions.bootstrap_ci", "default_replicates": md.BOOTSTRAP_DEFAULT_SAMPLES,
     "seed_default": 42, "structure": "percentile, class-stratified, withhold if <50% valid",
     "called_by": "metric_definitions.FullEvaluationMetrics, evaluate.py (--bootstrap), eval_record_test"},
    {"id": "eval_ulb.bootstrap_ci", "default_replicates": 200,
     "seed_default": 42, "structure": "percentile, class-stratified (single-class resamples skipped)",
     "called_by": "eval_ulb.py only (lines 135-136, stacker CIs)"},
    {"id": "protocol §2 proposal", "default_replicates": 2000,
     "seed_default": None, "structure": "stratified (proposal)",
     "called_by": "nothing — proposal only; marker M-3/T-5 unresolved",
     "status": "PENDING REVIEW — not implemented, not approved"},
]


def resolve_bootstrap(*, n_bootstrap: int | None, seed: int = 42,
                      mode: str = "development") -> dict[str, Any]:
    """Explicit replicate count or refusal — an implicit default can never
    silently determine a preregistered result (§11)."""
    if mode == "preregistered":
        if BOOTSTRAP_COUNT_APPROVED is None:
            raise PendingReview(
                "bootstrap replicate count unresolved: implementations exist with defaults "
                f"{[b['default_replicates'] for b in BOOTSTRAP_IMPLEMENTATIONS[:2]]} and the "
                "protocol proposes 2000 — approval (M-3/T-5) required; passing n_bootstrap "
                "explicitly does NOT constitute approval")
        if n_bootstrap is None or n_bootstrap != BOOTSTRAP_COUNT_APPROVED:
            raise PendingReview(
                f"preregistered runs require n_bootstrap == approved value "
                f"({BOOTSTRAP_COUNT_APPROVED}), got {n_bootstrap}")
        return {"n_bootstrap": n_bootstrap, "seed": seed, "explicit": True,
                "implementation": "metric_definitions.bootstrap_ci", "approved": True}
    if n_bootstrap is None:
        return {"n_bootstrap": md.BOOTSTRAP_DEFAULT_SAMPLES, "seed": seed, "explicit": False,
                "implementation": "metric_definitions.bootstrap_ci",
                "warning": "development default used — replicate count is a pending reviewer decision (M-3)"}
    return {"n_bootstrap": int(n_bootstrap), "seed": seed, "explicit": True,
            "implementation": "metric_definitions.bootstrap_ci", "approved": False,
            "warning": "development run — explicit count recorded, approval still pending (M-3)"}


def paired_difference_ci(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                         y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, *,
                         n_bootstrap: int | None, seed: int | None,
                         level: float = 0.95) -> None:
    """Paired-difference CI — INTERFACE BOUNDARY ONLY (§12).

    Schema contract (inputs must satisfy this before approval):
      metric_fn(y, scores) -> float on identical rows; scores_a/scores_b are
      paired per-row systems on the same y; n_bootstrap and seed are explicit
      (no defaults); result would be {point_delta, ci_low, ci_high, level,
      n_bootstrap, seed, withheld_reason} with percentile class-stratified
      shared-row resampling per NR-03 §J — BUT the estimator/parameters are a
      listed statistical-review item (NR-02 §H; T-5), so no CI is generated
      here. Raising PendingReview marks E1/E2/E3 PENDING REVIEW; a false or
      invented interval can never be produced (§12/§28).
    """
    y = np.asarray(y)
    for nm, arr in (("scores_a", scores_a), ("scores_b", scores_b)):
        if np.asarray(arr).shape != y.shape:
            raise ConfigurationError(
                f"paired_difference_ci: {nm} shape {np.asarray(arr).shape} != labels "
                f"{y.shape} — pairing requires identical rows")
    if n_bootstrap is None or seed is None:
        raise ConfigurationError(
            "paired_difference_ci: n_bootstrap and seed must be explicit (no implicit defaults)")
    if not PAIRED_CI_SPEC_APPROVED:
        raise PendingReview(
            "paired-difference CI construction is not approved (NR-02 §H statistical-review "
            "item; T-5) — interface/schema boundary only; E1/E2/E3 remain PENDING REVIEW "
            "(PD-PAIRED). No interval is generated.")
    raise NotImplementedError("estimator intentionally absent until an approved specification exists")


# ── MMD decision rules (§K): thresholds unchanged, mechanics only ──────────
MMD_RULES: dict[str, dict[str, Any]] = {
    "C1": {"quantity": "(ens - single)/single on PR-AUC", "threshold": 0.02,
           "direction": "higher", "equality": "met", "ci_excludes": 0.0},
    "C3": {"quantity": "wrong-confident reduction (rel.) + coverage loss (abs.)",
           "reduction_threshold": 0.20, "coverage_loss_max": 0.02,
           "direction": "higher", "equality": "met", "ci_excludes": 0.0},
    "C5": {"quantity": "external ROC-AUC + retention ratio", "roc_auc_threshold": 0.70,
           "retention_threshold": 0.80, "direction": "higher", "equality": "met",
           "ci_excludes": 0.5},
}


def evaluate_mmd(rule_id: str, observed: dict[str, float],
                 ci: dict[str, float] | None) -> dict[str, Any]:
    """MMD gate mechanics on already-observed numbers (never selects them).
    Equality = MET (NR-03 §K); an interval touching the excluded value does
    NOT exclude it. Descriptive point readings may accompany, never replace,
    the gate."""
    if rule_id not in MMD_RULES:
        raise ConfigurationError(f"unknown MMD rule {rule_id!r}; known: {sorted(MMD_RULES)}")
    rule = MMD_RULES[rule_id]
    if rule_id == "C1":
        point_met = observed.get("relative_delta", float("-inf")) >= rule["threshold"]
        excl = ci is not None and (ci.get("lo", 0.0) > rule["ci_excludes"] or
                                   ci.get("hi", 0.0) < rule["ci_excludes"])
    elif rule_id == "C3":
        point_met = (observed.get("reduction_rel", float("-inf")) >= rule["reduction_threshold"]
                     and observed.get("coverage_loss", 1.0) <= rule["coverage_loss_max"])
        excl = ci is not None and (ci.get("lo", 0.0) > rule["ci_excludes"] or
                                   ci.get("hi", 0.0) < rule["ci_excludes"])
    else:  # C5
        point_met = (observed.get("external_roc_auc", float("-inf")) >= rule["roc_auc_threshold"]
                     and observed.get("retention", float("-inf")) >= rule["retention_threshold"])
        excl = ci is not None and (ci.get("lo", 1.0) > rule["ci_excludes"] or
                                   ci.get("hi", -1.0) < rule["ci_excludes"])
    return {"rule": rule_id, "point_met": bool(point_met), "ci_excludes": bool(excl),
            "gate_met": bool(point_met and excl),
            "note": "MMD magnitudes themselves are pending reviewer approval (M-5) — "
                    "this function only applies the recorded rule mechanically"}


# ── Provenance population (§10): the 14 NR-03 §O items on one record ───────
PROVENANCE_ITEMS: list[tuple[str, str]] = [
    ("experiment_id", "evaluation_config.experiment_id"),
    ("dataset_hash", "dataset.sha256"),
    ("split", "evaluation_config.split"),
    ("seed", "seed (or evaluation_config.seed_not_applicable)"),
    ("command", "command"),
    ("model_config_identity", "model_hash + artifact_file_hashes"),
    ("feature_contract_identity", "feature_schema_version"),
    ("preprocessing_identity", "preprocessing_version"),
    ("threshold_state", "threshold + threshold_source"),
    ("metric_definitions_identity", "metric_definitions_version"),
    ("protocol_linkage", "evaluation_config.protocol_version"),
    ("deviations", "evaluation_config.deviations"),
    ("artifact_identity", "artifact_file_hashes"),
    ("git_sha", "git_commit"),
]


def provenance_report(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Population status of all 14 items on an evidence record (§10)."""
    cfg = record.get("evaluation_config") or {}
    seed_ok = record.get("seed") is not None or bool(cfg.get("seed_not_applicable"))
    values = {
        "experiment_id": cfg.get("experiment_id"),
        "dataset_hash": (record.get("dataset") or {}).get("sha256"),
        "split": cfg.get("split"),
        "seed": record.get("seed") if seed_ok else None,
        "command": record.get("command"),
        "model_config_identity": record.get("model_hash"),
        "feature_contract_identity": record.get("feature_schema_version"),
        "preprocessing_identity": record.get("preprocessing_version"),
        "threshold_state": (record.get("threshold"), record.get("threshold_source")),
        "metric_definitions_identity": record.get("metric_definitions_version"),
        "protocol_linkage": cfg.get("protocol_version"),
        "deviations": cfg.get("deviations"),
        "artifact_identity": record.get("artifact_file_hashes"),
        "git_sha": record.get("git_commit"),
    }
    out: dict[str, dict[str, Any]] = {}
    for item, field_path in PROVENANCE_ITEMS:
        v = values.get(item)
        populated = v is not None and v != "" and v != {} and v != ()
        if item == "seed":
            populated = seed_ok
        if item == "deviations":
            populated = isinstance(v, list)   # an explicit empty list is populated
        out[item] = {"field": field_path, "status": "POPULATED" if populated else "MISSING",
                     "value": v if item != "artifact_identity" else
                     (sorted(v) if isinstance(v, dict) else v)}
    return out


def validate_evidence_record(record: dict[str, Any], *,
                             require_preregistered: bool) -> tuple[bool, list[str]]:
    """Evidence-first validation (§9): MEASURED requires a fully populated,
    resolvable v1.1 record. Returns (ok, errors)."""
    errors: list[str] = []
    if record.get("status") != "COMPLETED":
        errors.append(f"record status {record.get('status')!r} != COMPLETED")
    if record.get("record_version") != EVIDENCE_SCHEMA_VERSION:
        errors.append(f"record_version {record.get('record_version')!r} != {EVIDENCE_SCHEMA_VERSION}")
    if not (record.get("dataset") or {}).get("sha256"):
        errors.append("dataset.sha256 missing")
    if not record.get("git_commit"):
        errors.append("git_commit missing")
    if not record.get("command"):
        errors.append("command missing (v1.1)")
    cfg = record.get("evaluation_config") or {}
    if record.get("seed") is None and not cfg.get("seed_not_applicable"):
        errors.append("seed and seed_not_applicable both absent")
    prov = provenance_report(record)
    for item, info in prov.items():
        if info["status"] != "POPULATED":
            errors.append(f"provenance item '{item}' not populated ({info['field']})")
    if require_preregistered:
        if not cfg.get("preregistered"):
            errors.append("record not marked preregistered — cannot back a MEASURED cell")
        if cfg.get("fixture_test"):
            errors.append("fixture record must never back a MEASURED benchmark cell")
    return (not errors, errors)


@dataclass
class RunResult:
    """Outcome of one requested execution attempt (§8/§9)."""
    experiment_id: str
    seed: int | None
    status: str                       # one of CELL_STATUSES
    code: str                          # OK / CONFIGURATION ERROR / ... (§5 diagnostics)
    measured: bool = False
    preregistered: bool = False
    failure_class: str | None = None
    diagnostic: str | None = None
    rerun_permitted: bool | None = None
    evaluation_id: str | None = None
    config_digest: str | None = None
    metrics: dict[str, Any] | None = None
    pending: list[str] = dc_field(default_factory=list)
    blocked: list[str] = dc_field(default_factory=list)
    violations: list[str] = dc_field(default_factory=list)
    timestamp: str = ""
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["metrics_is_fixture"] = self.metrics is not None and \
            bool((self.metrics or {}).get("_fixture"))
        return d


def config_digest(contract: dict[str, Any], *, include_seed: bool = True) -> str:
    """Deterministic digest over the frozen contract (§19 determinism)."""
    keys = [k for k, _ in CONTRACT_FIELDS if k != "git_commit"]
    payload = {k: contract.get(k) for k in keys if k != "seed" or include_seed}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ── Evidence-first execution (§8/§9) ───────────────────────────────────────
def _refusal_result(experiment_id: str, seed: int | None, exc: HarnessRefusal,
                    contract: dict[str, Any]) -> RunResult:
    pending_ids: list[str] = []
    blocked: list[str] = []
    violations: list[str] = []
    if isinstance(exc, PendingReview):
        pending_ids = [d["id"] for d in pending_for(experiment_id,
                        tier=contract.get("data_tier") or "development")]
    elif isinstance(exc, Blocked):
        blocked = [exc.message]
    elif isinstance(exc, ProtocolViolation):
        violations = [exc.message]
    return RunResult(
        experiment_id=experiment_id, seed=seed, status=exc.status, code=exc.code,
        measured=False, preregistered=False, diagnostic=exc.message,
        config_digest=config_digest(contract) if contract else None,
        pending=pending_ids, blocked=blocked, violations=violations,
        timestamp=er.datetime.now(er.timezone.utc).isoformat(timespec="seconds"),
        git_commit=er.git_commit())


def execute(contract: dict[str, Any],
            runner: Callable[[dict[str, Any], int], dict[str, Any]],
            *, mode: str = "preregistered",
            ledger_path: str | Path | None = None,
            ledger_records: list[dict] | None = None) -> RunResult:
    """Run the 8-step evidence-first pipeline (§9):
    validate protocol -> validate dataset -> validate configuration ->
    execute evaluation -> calculate metrics -> generate evidence ->
    validate evidence -> only then expose as MEASURED.

    Refusals (config/pending/blocked/violation) never execute the runner.
    Development mode is the explicitly-labeled non-preregistered override
    (§6): it may proceed past the guard for testing, but its results can
    never be MEASURED (records carry preregistered=false + a warning).
    """
    if mode not in ("preregistered", "development"):
        raise ConfigurationError(f"mode must be 'preregistered' or 'development', got {mode!r}")
    prereg = mode == "preregistered"
    experiment_id = contract.get("experiment_id")
    if experiment_id is None:
        return RunResult(experiment_id="<missing>", seed=contract.get("seed"),
                         status=NOT_ESTABLISHED, code=CODE_CONFIG,
                         diagnostic="missing frozen field 'experiment_id'",
                         timestamp=er.datetime.now(er.timezone.utc).isoformat(timespec="seconds"),
                         git_commit=er.git_commit())
    seed = contract.get("seed") if isinstance(contract.get("seed"), int) else None
    digest = config_digest(contract)

    # 1. validate protocol/contract fields
    cfg_errors = validate_contract(contract)
    if cfg_errors:
        return RunResult(experiment_id, seed, NOT_ESTABLISHED, CODE_CONFIG, measured=False,
                         preregistered=False, diagnostic="; ".join(cfg_errors),
                         config_digest=digest,
                         timestamp=er.datetime.now(er.timezone.utc).isoformat(timespec="seconds"),
                         git_commit=er.git_commit())
    # 2. guard (§6) — pending/blocked; final-test protection (§17)
    decision = guard(experiment_id, contract,
                     tier=contract.get("data_tier"), ledger_records=ledger_records)
    dev_pending: list[str] = []
    if not decision.allowed:
        if decision.code == CODE_PENDING and not prereg:
            # §6 explicitly-labeled development override: PENDING REVIEW may be
            # proceeded past ONLY in development mode; the pending decisions are
            # stamped onto the record, which can never be MEASURED.
            dev_pending = [d["id"] for d in decision.pending]
        else:
            if decision.code == CODE_CONFIG:
                exc: HarnessRefusal = ConfigurationError("; ".join(decision.config_errors))
            elif decision.code == BLOCKED:
                exc = Blocked(decision.blocked[0])
            elif decision.code == CODE_VIOLATION:
                exc = ProtocolViolation("; ".join(decision.violations))
            else:
                exc = PendingReview(decision.message)
            return _refusal_result(experiment_id, seed, exc, contract)
    # Development-mode note: when mode == 'development', the record written
    # below carries preregistered=false + a NON-PREREGISTERED warning, so the
    # result can never be exposed as MEASURED (§6).

    # 3. execute evaluation (injected runner; harness never scores datasets itself)
    try:
        out = runner(contract, seed if seed is not None else 0)
        y = np.asarray(out["y"])
        scores = np.asarray(out["scores"])
    except MetricDomainError as e:
        fc = FAILURE_CLASSES[e.failure_kind]
        return RunResult(experiment_id, seed, fc["cell"], CODE_OK, measured=False,
                         preregistered=False, failure_class=fc["class"],
                         diagnostic=str(e), rerun_permitted=fc["rerun"],
                         config_digest=digest,
                         timestamp=er.datetime.now(er.timezone.utc).isoformat(timespec="seconds"),
                         git_commit=er.git_commit())
    except Exception as e:  # crash → FAILED, preserved (§8)
        fc = FAILURE_CLASSES["crash"]
        res = RunResult(experiment_id, seed, FAILED, CODE_OK, measured=False,
                        preregistered=False, failure_class=fc["class"],
                        diagnostic=f"{type(e).__name__}: {e}", rerun_permitted=fc["rerun"],
                        config_digest=digest,
                        timestamp=er.datetime.now(er.timezone.utc).isoformat(timespec="seconds"),
                        git_commit=er.git_commit())
        _try_record_failed_run(contract, res, ledger_path)
        return res

    # 4. calculate metrics (authoritative path; bootstrap params explicit)
    thr_cfg = contract.get("threshold_config")
    thr_value = None
    _thr_src = "none"
    if isinstance(thr_cfg, dict):
        thr_value = thr_cfg.get("value")
        _thr_src = thr_cfg.get("source") or "none"
    elif isinstance(thr_cfg, ThresholdSpec):
        thr_value = thr_cfg.value
        _thr_src = thr_cfg.source
    thr_source = {"frozen": "fixed", "validation": "validation",
                  "exploratory": "exploratory"}.get(_thr_src, "none")
    try:
        boot = contract.get("bootstrap_config") or {}
        metrics = compute_preregistered_metrics(
            y, scores, threshold=thr_value,
            ci=bool(boot.get("ci")),
            n_bootstrap=boot.get("n_bootstrap"),
            bootstrap_seed=boot.get("seed") or 42, mode=mode)
    except (PendingReview, ConfigurationError) as e:
        return _refusal_result(experiment_id, seed, e, contract)
    except MetricDomainError as e:
        fc = FAILURE_CLASSES[e.failure_kind]
        return RunResult(experiment_id, seed, fc["cell"], CODE_OK, measured=False,
                         preregistered=False, failure_class=fc["class"],
                         diagnostic=str(e), rerun_permitted=fc["rerun"],
                         config_digest=digest,
                         timestamp=er.datetime.now(er.timezone.utc).isoformat(timespec="seconds"),
                         git_commit=er.git_commit())

    # 5–7. generate + validate evidence (§9): a number is not MEASURED until
    # its record exists, resolves, and carries all 14 provenance items.
    ledger = Path(ledger_path) if ledger_path is not None else CANONICAL_LEDGER
    warnings: list[str] = []
    if not prereg:
        warnings.append("NON-PREREGISTERED development run — not eligible for confirmatory "
                        "benchmark claims (explicit development override, NR-04 §6)")
    try:
        fc_ident = validate_feature_contract(contract.get("feature_contract_path") or CONTRACT_PATH)
        led_obj = er.EvaluationLedger(ledger)
        # attempt salt: identical reruns in the same UTC second must still get
        # distinct evaluation_ids (content digest would otherwise collide — §22)
        attempt = 1 + sum(
            1 for r in led_obj.records()
            if (r.get("evaluation_config") or {}).get("config_digest") == digest)
        rec = er.create_evaluation_record(
            model_identifier=f"prereg_{experiment_id}__{contract.get('model_identity', 'unspecified')}",
            artifact_paths=[Path(p) for p in (contract.get("artifact_paths") or [])],
            dataset_path=Path(contract.get("dataset_path")),
            preprocessing_version=contract.get("preprocessing_identity"),
            feature_schema_version=f"{fc_ident['name']}@{fc_ident['sha256'][:12]}",
            seed=seed, threshold=thr_value,
            threshold_source=thr_source,
            evaluation_config={
                "experiment_id": experiment_id,
                "protocol_version": contract.get("protocol_version"),
                "split": (contract.get("split") or {}).get("name")
                         if isinstance(contract.get("split"), dict) else contract.get("split"),
                "population": contract.get("population"),
                "data_tier": contract.get("data_tier"),
                "deviations": contract.get("deviations") or [],
                "preregistered": prereg,
                "fixture_test": bool(contract.get("fixture_test")),
                "harness": f"prereg_harness {HARNESS_VERSION}",
                "guard_pending_at_execution": dev_pending,
                "execution_attempt": attempt,
                "config_digest": digest,
                "bootstrap": contract.get("bootstrap_config") or {},
                "statistical_tests": contract.get("statistical_test_config"),
                "acceptance_criteria": contract.get("acceptance_criterion_config"),
                "metric_provisional_labels": ["ece_binning_pending_M-2"],
                **({"seed_not_applicable": contract["seed_not_applicable"]}
                   if contract.get("seed_not_applicable") else {}),
            },
            metrics=metrics, command=contract.get("execution_command"),
            warnings=warnings,
            status="COMPLETED")
        led = led_obj
        led.append(rec)
        try:
            (ledger.parent / f"record_{rec.evaluation_id}.json").write_text(
                json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
        except OSError as side_err:
            warnings.append(f"sidecar write failed (ledger record is authoritative): {side_err}")
        # validate by re-reading the persisted record (not the in-memory object)
        persisted = [r for r in led.records() if r.get("evaluation_id") == rec.evaluation_id]
        if not persisted:
            raise EvidenceFailure("record not resolvable in the ledger after append")
        ok, errs = validate_evidence_record(persisted[0], require_preregistered=prereg)
        if not ok:
            raise EvidenceFailure("; ".join(errs))
    except (EvidenceFailure, HarnessRefusal) as e:
        fc = FAILURE_CLASSES["evidence_failure"]
        return RunResult(experiment_id, seed, NOT_ESTABLISHED, CODE_EVIDENCE, measured=False,
                         preregistered=False, failure_class=fc["class"],
                         diagnostic=f"{CODE_EVIDENCE}: {e.message if isinstance(e, HarnessRefusal) else e} "
                                    "— result cannot become a measured benchmark claim (§9)",
                         rerun_permitted=True, config_digest=digest,
                         timestamp=er.datetime.now(er.timezone.utc).isoformat(timespec="seconds"),
                         git_commit=er.git_commit())
    except Exception as e:
        fc = FAILURE_CLASSES["evidence_failure"]
        return RunResult(experiment_id, seed, NOT_ESTABLISHED, CODE_EVIDENCE, measured=False,
                         preregistered=False, failure_class=fc["class"],
                         diagnostic=f"{CODE_EVIDENCE}: {type(e).__name__}: {e} "
                                    "— result cannot become a measured benchmark claim (§9)",
                         rerun_permitted=True, config_digest=digest,
                         timestamp=er.datetime.now(er.timezone.utc).isoformat(timespec="seconds"),
                         git_commit=er.git_commit())

    # 8. expose as MEASURED — only now (§9)
    if prereg:
        return RunResult(experiment_id, seed, MEASURED, CODE_OK, measured=True,
                         preregistered=True, evaluation_id=rec.evaluation_id,
                         config_digest=digest, metrics=metrics,
                         rerun_permitted=False,
                         timestamp=rec.timestamp_utc, git_commit=rec.git_commit)
    return RunResult(experiment_id, seed, NOT_ESTABLISHED, "NON-PREREGISTERED (development)",
                     measured=False, preregistered=False,
                     diagnostic="development/fixture run — evidence recorded outside the "
                                "confirmatory path; cell status NOT ESTABLISHED (invalid as a "
                                "preregistered experiment), never MEASURED (§6)",
                     evaluation_id=rec.evaluation_id, config_digest=digest, metrics=metrics,
                     rerun_permitted=False, timestamp=rec.timestamp_utc,
                     git_commit=rec.git_commit, pending=dev_pending)


def record_failed_run(contract: dict[str, Any], *, failure_kind: str,
                      diagnostic: str, seed: int | None = None,
                      ledger_path: str | Path | None = None) -> RunResult:
    """Persist a FAILED attempt with full identity (§8): experiment id, seed,
    configuration, git SHA, timestamp, failure class, diagnostic, rerun flag.
    Append-only — never deleted, never silently replaced."""
    if failure_kind not in FAILURE_CLASSES:
        raise ConfigurationError(f"unknown failure kind {failure_kind!r}; "
                                 f"known: {sorted(FAILURE_CLASSES)}")
    fc = FAILURE_CLASSES[failure_kind]
    res = RunResult(contract.get("experiment_id", "<missing>"), seed, fc["cell"], CODE_OK,
                    measured=False, preregistered=False, failure_class=fc["class"],
                    diagnostic=diagnostic, rerun_permitted=fc["rerun"],
                    config_digest=config_digest(contract),
                    timestamp=er.datetime.now(er.timezone.utc).isoformat(timespec="seconds"),
                    git_commit=er.git_commit())
    _try_record_failed_run(contract, res, ledger_path)
    return res


def _try_record_failed_run(contract: dict[str, Any], res: RunResult,
                           ledger_path: str | Path | None) -> None:
    try:
        ledger = Path(ledger_path) if ledger_path is not None else CANONICAL_LEDGER
        led_obj = er.EvaluationLedger(ledger)
        attempt = 1 + sum(
            1 for r in led_obj.records()
            if (r.get("evaluation_config") or {}).get("config_digest") == res.config_digest)
        rec = er.create_evaluation_record(
            model_identifier=f"prereg_{contract.get('experiment_id', '<missing>')}__FAILED",
            artifact_paths=[Path(p) for p in (contract.get("artifact_paths") or [])],
            dataset_path=Path(contract.get("dataset_path") or "data/transactions.csv"),
            seed=res.seed, threshold=None, threshold_source="none",
            preprocessing_version=contract.get("preprocessing_identity") or "undeclared",
            feature_schema_version=contract.get("feature_contract_path") or "undeclared",
            evaluation_config={
                "experiment_id": contract.get("experiment_id"),
                "protocol_version": contract.get("protocol_version"),
                "split": (contract.get("split") or {}).get("name")
                         if isinstance(contract.get("split"), dict) else contract.get("split"),
                "deviations": contract.get("deviations") or [],
                "preregistered": False, "fixture_test": bool(contract.get("fixture_test")),
                "harness": f"prereg_harness {HARNESS_VERSION}",
                "config_digest": res.config_digest,
                "execution_attempt": attempt,
                "failure_class": res.failure_class,
                "diagnostic": res.diagnostic,
                "rerun_permitted": res.rerun_permitted,
                "attempt_timestamp": res.timestamp,
            },
            metrics={}, command=contract.get("execution_command"),
            warnings=[f"FAILED run preserved append-only: {res.failure_class}"],
            status="FAILED")
        led_obj.append(rec)
        res.evaluation_id = rec.evaluation_id
    except Exception as e:
        res.diagnostic = f"{res.diagnostic} | (failed-run ledger append also failed: {e})"


def run_seed_plan(contract_template: dict[str, Any],
                  runner: Callable[[dict[str, Any], int], dict[str, Any]],
                  seeds: Iterable[int], *, mode: str = "preregistered",
                  ledger_path: str | Path | None = None,
                  ledger_records: list[dict] | None = None) -> list[RunResult]:
    """Per-seed execution with per-seed evidence records (§7): each seed is
    one independent replicate with its own record; failed seeds are retained;
    reruns use the identical frozen configuration (same config digest minus
    nothing — seed identity is part of the digest)."""
    results: list[RunResult] = []
    plan = SeedPlan(seeds)
    for s in plan.seeds:
        contract = dict(contract_template)
        contract["seed"] = s
        results.append(execute(contract, runner, mode=mode, ledger_path=ledger_path,
                               ledger_records=ledger_records))
    return results


# ── Full-matrix reporting (§18): canonical reporting surface ───────────────
_CONDITIONS_E2 = [
    "clean", "missing_features", "invalid_features", "drift", "corruption",
    "adversarially_unusual_valid", "combinations",
]


def _cell(experiment_id: str, axes: dict[str, str], statuses: list[str],
          note: str = "") -> dict[str, Any]:
    if BLOCKED in statuses:
        st = BLOCKED
    elif PENDING_REVIEW in statuses:
        st = PENDING_REVIEW
    elif all(s == NOT_APPLICABLE for s in statuses):
        st = NOT_APPLICABLE
    else:
        st = NOT_ESTABLISHED   # design resolved, never executed (§24: no run yet)
    return {"experiment": experiment_id, "axes": axes, "status": st, "note": note,
            "evaluation_id": None}


def build_matrix() -> dict[str, Any]:
    """Complete experiment × dataset × seed × contract × condition × transfer
    tensor (protocol §9.11). Every cell carries exactly one of the six
    statuses; unresolved axes are single explicit PENDING slots (no invented
    values); blocked tracks get their own BLOCKED cells — never collapsed
    into missing data and never zero-filled."""
    cells: list[dict[str, Any]] = []
    design: dict[str, dict[str, str]] = {}
    for eid in EXPERIMENT_IDS:
        fields = EXPERIMENTS[eid]["fields"]
        design[eid] = {f: info["status"] for f, info in fields.items()}
        # axis slots, derived from the frozen/pending/blocked design cells
        dataset_axis = (["kaggle_fraud_test@12d553ab19440c75…"]
                        if fields["Dataset"]["status"] == "FROZEN"
                        else ["PENDING:dataset (D-1/D-1b/§1 — unresolved)"])
        dataset_status = (["ok"] if fields["Dataset"]["status"] == "FROZEN"
                           else [PENDING_REVIEW])
        seed_axis = ([NOT_APPLICABLE] if eid in ("E5", "E6")
                     else ["PENDING:seed policy (§5/T-3 — 42–46 proposal not approved)"])
        seed_status = ([NOT_APPLICABLE] if eid in ("E5", "E6") else [PENDING_REVIEW])
        contract_axis = ["PENDING:scope (§0 — candidate/contract track unresolved)"]
        contract_status = [PENDING_REVIEW]
        if eid in ("E3", "E4"):   # §N: 21+7 full-contract claim is blocked by design
            contract_axis.append("21+7_full_bank_contract")
            contract_status.append(BLOCKED)
        condition_axis = _CONDITIONS_E2 if eid == "E2" else ["default"]
        direction_axis = (["in_domain_to_external"] if eid in ("E3", "E4")
                          else [NOT_APPLICABLE])
        direction_status = ([PENDING_REVIEW] if eid in ("E3", "E4") else [NOT_APPLICABLE])
        for dsn, dstat in zip(dataset_axis, dataset_status):
            for sd, sstat in zip(seed_axis, seed_status):
                for cn, cstat in zip(contract_axis, contract_status):
                    for cond in condition_axis:
                        for dr, rstat in zip(direction_axis, direction_status):
                            axes = {"dataset": dsn, "seed": sd, "contract": cn,
                                    "condition": cond, "transfer_direction": dr}
                            sts = [dstat, sstat, cstat, rstat]
                            note = ""
                            if cstat == BLOCKED:
                                note = ("BLOCKED — FEATURE CONTRACT: the 7 bank features are "
                                        "absent from every public dataset (G-20); no substitute")
                            cells.append(_cell(eid, axes, sts, note))
    counts = {s: sum(1 for c in cells if c["status"] == s) for s in CELL_STATUSES}
    return {
        "harness_version": HARNESS_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "statuses": list(CELL_STATUSES),
        "design_matrix": design,           # NR-03 §C transcription (F/P/B/N -> mapped words)
        "tensor_cells": cells,
        "cell_counts": counts,
        "total_cells": len(cells),
        "measured_cells": counts[MEASURED],
        "note": "No cell may be omitted, zero-filled, or selectively presented; "
                "MEASURED requires a validated preregistered ledger record (§9/§18).",
    }


# ── Existing pipeline integration (§21) ────────────────────────────────────
PIPELINES: dict[str, dict[str, Any]] = {
    "calibration_test": {
        "script": "backend/scripts/calibration_test.py",
        "role": "supporting: secondary calibration metrics on validation (not a confirmatory run)",
        "record": "canonical ledger (COMPLETED, command v1.1)",
        "import_safe": True,
        "metric_pins": "compute_brier_score:41 / compute_ece:46 (10 equal-width — governing implementation)",
        "notes": "harness imports the pinned Brier/ECE functions — single authoritative path",
    },
    "eval_ulb": {
        "script": "backend/scripts/eval_ulb.py",
        "role": "baseline reproduction (feeds NR-07 baselines), not a confirmatory run",
        "record": "canonical ledger (record block appended at end of script)",
        "import_safe": True,
        "metric_pins": "recall_at_fpr:24 (ranking variant — pinned by NR-03 §I), bootstrap_ci:32 (default 200)",
        "notes": "NR-04 refactor: script body moved under main() + __main__ guard so the pinned "
                 "functions are importable; line numbers of all cited functions unchanged; "
                 "scientific procedure untouched",
    },
    "cross_dataset_eval": {
        "script": "backend/scripts/cross_dataset_eval.py",
        "role": "supporting/provisional transfer evidence (proxy mapping is explicitly NOT "
                 "feature-semantic evaluation)",
        "record": "canonical ledger, but append is try/except-wrapped (non-fatal) — flagged in "
                  "NR-01 §flag for harness design",
        "import_safe": True,
        "notes": "metrics() uses searchsorted side='right' then idx-1 for r@1%FPR — a DIFFERENT "
                 "implementation from the eval_ulb pin; harness metric path uses the eval_ulb pin "
                 "(NR-03 §I semantic pin), this script's numbers are not interchangeable",
    },
    "train_compare": {
        "script": "backend/src/train_compare.py",
        "role": "closest executable analogue of C1/C2, on synthetic data (unnamed in protocol §1)",
        "record": "canonical ledger (record_from_run; gate_rows append bug fixed in Phase 105)",
        "import_safe": True,
        "notes": "harness-invoked runs MUST pin --outdir (default models/artifacts overwrites "
                 "production artifacts — NR-02 §K carry-over)",
        "outdir_required": True,
    },
    "evaluate": {
        "script": "backend/scripts/evaluate.py",
        "role": "generic frozen evaluation with refuse_test_tuning guard",
        "record": "writes output-dir-local eval_ledger.jsonl (args.output_dir), NOT the canonical "
                  "ledger unless output_dir points at reports/evaluation_runs",
        "import_safe": True,
        "notes": "threshold_source choices fixed/validation only; harness parity-tests the guard",
    },
    "ibm_train": {
        "script": "backend/scripts/ibm_train.py",
        "role": "IBM v2 21-feature fusion (R-04 remainder reproduction target)",
        "record": "NONE — produces no ledger record today",
        "import_safe": True,
        "notes": "documented incompatibility (§21): re-executing it would overwrite "
                 "reports/ibm_train/training_report.json, the artifact pinned by claims "
                 "C-004…C-009, and takes tens of minutes on 1.2M rows — reproduction + "
                 "record attachment is deferred, NOT done this phase; harness integration "
                 "point = attach a v1.1 record post-run with pinned outdir",
        "deferred": True,
    },
}


# ── CLI (§25: an entry point inside the existing evaluation location) ───────
def _cmd_registry(_args: argparse.Namespace) -> int:
    print(json.dumps({"experiments": EXPERIMENTS, "marker_ids": MARKER_IDS}, indent=2))
    return 0


def _cmd_pending(_args: argparse.Namespace) -> int:
    print(json.dumps({"pending_decisions": PENDING_DECISIONS,
                      "count": len(PENDING_DECISIONS),
                      "markers": len(MARKER_IDS),
                      "gates": {"PROTOCOL_SIGNED_OFF": PROTOCOL_SIGNED_OFF,
                                "SEED_POLICY_APPROVED": SEED_POLICY_APPROVED,
                                "SEED_AGGREGATION_APPROVED": SEED_AGGREGATION_APPROVED,
                                "BOOTSTRAP_COUNT_APPROVED": BOOTSTRAP_COUNT_APPROVED,
                                "ECE_BINNING_APPROVED": ECE_BINNING_APPROVED,
                                "PAIRED_CI_SPEC_APPROVED": PAIRED_CI_SPEC_APPROVED}},
                     indent=2))
    return 0


def _cmd_guard(args: argparse.Namespace) -> int:
    d = guard(args.experiment_id, tier=args.tier)
    print(json.dumps(d.to_dict(), indent=2))
    return 0 if d.allowed else 3


def _cmd_matrix(args: argparse.Namespace) -> int:
    m = build_matrix()
    text = json.dumps(m, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out} ({m['total_cells']} cells, measured={m['measured_cells']})")
    else:
        print(text)
    return 0


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    try:
        info = validate_dataset(args.name)
    except HarnessRefusal as e:
        print(e.message)
        return 4
    print(json.dumps(info, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prereg_harness",
                                 description="NR-04 preregistered benchmark harness "
                                             "(guard/matrix/validation — no execution entry point)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("registry", help="print the 7-experiment registry (NR-03 §C)")
    sub.add_parser("pending", help="print every reviewer-decision gate and its state")
    g = sub.add_parser("guard", help="emit the guard verdict for one experiment")
    g.add_argument("experiment_id")
    g.add_argument("--tier", default=None)
    m = sub.add_parser("matrix", help="emit the full status matrix (§18)")
    m.add_argument("--out", default=None)
    d = sub.add_parser("validate-dataset", help="validate a registered dataset identity")
    d.add_argument("name")
    args = ap.parse_args(argv)
    handlers = {"registry": _cmd_registry, "pending": _cmd_pending,
                "guard": _cmd_guard, "matrix": _cmd_matrix,
                "validate-dataset": _cmd_validate_dataset}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
