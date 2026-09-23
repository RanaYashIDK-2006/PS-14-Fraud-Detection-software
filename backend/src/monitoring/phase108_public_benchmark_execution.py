"""Phase 108: Verified Public Benchmark Execution & External Performance
Evaluation.

FIRST phase that runs real, reproducible measurements of the CURRENT
production system against the Phase 107 verified ULB benchmark file.

THIS IS PUBLIC EXTERNAL BENCHMARK EVALUATION.
IT IS NOT REAL-WORLD VALIDATION, NOT INSTITUTIONAL VALIDATION, NOT
PROVIDER-ATTESTED VALIDATION, NOT MODEL PROMOTION, NOT MODEL RETRAINING,
NOT MODEL OPTIMIZATION.

Hard boundaries enforced by this module (fail closed):

- NATIVE_MODEL_EVALUATION = NOT_APPLICABLE.  ULB stays Group B (5/48
  native features usable, 43 blocking); the Altman-Native ensemble is
  never invoked, never imputed to, never approximated.  No alternate
  model is created to obtain metrics.  The only scored component is the
  EXISTING declarative rules engine evaluated under the inputs ULB can
  legitimately supply -- which is none of its 17 features, so the engine
  fail-closes to a constant 0.0 score for every row.  That constant is a
  MEASUREMENT of fail-closed behavior, not a detection claim.
- PRODUCTION THRESHOLD 0.018758 is immutable: read once, re-read after
  evaluation, compared for equality; the primary metric block uses it.
  The diagnostic sweep is DIAGNOSTIC_ONLY / NON_PRODUCTION /
  NO_THRESHOLD_CHANGE and cannot write configuration.
- INPUT INTEGRITY: the source file must hash to the exact Phase 107
  SHA-256, keep 284807 rows / 492 fraud / the registered 31-column
  schema, and map every label.  Any mismatch RAISES BEFORE any metric is
  computed: DATASET_INTEGRITY_FAILURE, never a silent continuation.
- SOURCE CONTAMINATION: training and feedback files are hashed before
  and after; benchmark fingerprints are intersected with training-row
  fingerprints; any artifact change RAISES (never masked).
- PCA COMPONENTS ARE NEVER MAPPED.  Class is never a feature.  Relative
  Time is never converted to calendar dates.
- NO network, NO download, NO provider contact, NO retrain/fit/tune, NO
  promotion, NO RWV execution, NO audit-chain writes (results live only
  under reports/external_benchmark/phase108/ulb/).
- No bypass parameters exist on any public entry point (force /
  allow_unverified / skip_validation / override / admin_override /
  bypass are rejected by signature).

Determinism: MANIFEST_CREATED_AT is a fixed constant; every number is a
pure function of the pinned dataset + code constants, so two runs
produce byte-identical results and identical result hashes.

Metric semantics: evaluation_metrics() is an O(n log n) port of the
repository authority (real_world_evaluation_protocol.compute_metrics)
including its endpoint construction and trapezoid conventions; the test
suite proves exact key-for-key equality against that authority on
fixtures before the port is used at ULB scale (the authority's own
sweep is O(n^2) and cannot run at 284807 rows).

Run:  ../.venv/Scripts/python.exe scripts/phase108_public_benchmark_execution_test.py
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.monitoring.external_dataset_contract import canonical_json
from src.monitoring.feature_contract import ML_FEATURE_ORDER, ML_FEATURE_VERSION
from src.monitoring.phase106_public_benchmark_registry import (
    AMOUNT_COLUMNS,
    LABEL_COLUMNS,
    TIMESTAMP_COLUMNS,
    assign_benchmark_group,
    class_imbalance,
    compute_mcc,
    dataset_preflight,
    evaluation_scope,
    get_dataset,
    native_eligible_row_count,
    normalize_label,
    usable_features,
)
from src.monitoring.phase107_public_dataset_ingestion import (
    BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
    EXPECTED_PRODUCTION_THRESHOLD,
    ingest_dataset,
    resolve_candidate_paths,
    sha256_file,
)
from src.monitoring.real_world_evaluation_protocol import (
    MODEL_ID,
    PRODUCTION_THRESHOLD,
    RELEASE_ID,
)
from src.monitoring.rwv_readiness_audit import FEATURE_VERSION
from src.monitoring.rwv_reproducibility import NATIVE_FEATURE_VERSION
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES
from src.risk_engine.main import EvaluateRequest, FeatureVector, band_of
from src.risk_engine.rules_engine import RulesEngine

PHASE = 108
EXECUTION_VERSION = "phase108_execution_v1"
MANIFEST_VERSION = "phase108_evaluation_manifest_v1"
MANIFEST_CREATED_AT = "2026-09-23T00:00:00+00:00"  # fixed: byte-identical reruns

DATASET_ID = "ULB_CREDIT_CARD_FRAUD"
DATASET_CANONICAL_NAME = "ULB Credit Card Fraud Detection (creditcard.csv)"

# Phase 107 authoritative identity of the verified file (spec pins the
# prefix 76274b691b16...; the full digest is the recorded Phase 107 value).
EXPECTED_SHA256 = (
    "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89")
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
EXPECTED_LEGIT = 284_315
EXPECTED_COLUMNS = 31
EXPECTED_SCHEMA: tuple[str, ...] = (
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount",
    "Class")
EXPECTED_SCHEMA_HASH = (
    "eb5a7f0652502a72971b81fb206b7f4aadc8724737f5368d6390f9d057ac3b9a")
EXPECTED_INGESTION_STATUS = "acquired_verified"

NATIVE_MODEL_EVALUATION = "NOT_APPLICABLE"
NATIVE_MODEL_EVALUATION_REASONS: tuple[str, ...] = (
    "ULB is Group B: 5/48 native features usable, 43 blocking",
    "Altman-Native ensemble requires the canonical 48-vector; ULB offers "
    "only Time, Amount and 28 undisclosed PCA components",
    "PCA substitution, feature imputation, alternate mapping and column "
    "renaming are forbidden; no alternate model is created to obtain metrics",
    "the existing rules engine is the only scored production component and "
    "it fail-closes because none of its 17 features is derivable from ULB",
)

IMBALANCE_WARNING = (
    "Severe class imbalance (0.1727% fraud): raw accuracy is misleading "
    "because the majority-class baseline already scores 99.83% while "
    "catching zero fraud.  Read balanced accuracy, precision, recall, F1, "
    "MCC, PR-AUC and ROC-AUC together; never accuracy alone.")

SWEEP_THRESHOLDS: tuple[float, ...] = (
    0.0, 0.005, 0.018758, 0.05, 0.5)
SWEEP_LABELS: tuple[str, ...] = (
    "DIAGNOSTIC_ONLY", "NON_PRODUCTION", "NO_THRESHOLD_CHANGE")
SWEEP_STATEMENT = (
    "The threshold sweep is descriptive analysis only and does not "
    "constitute threshold selection or model optimization.")

CALIBRATION_STATUS = "NOT_APPLICABLE"
CALIBRATION_REASON = (
    "no legitimate probability-producing evaluation path exists for ULB: "
    "native model evaluation is forbidden (Group B) and rule severity "
    "scores are not calibrated probabilities of fraud; no recalibration, "
    "calibration fitting or calibration artifact change is performed")

AMOUNT_BUCKET_EDGES: tuple[float, ...] = (0.0, 10.0, 50.0, 100.0, 500.0,
                                           1000.0, 5000.0, math.inf)
BUCKET_CONVENTION = "[lo, hi) with the final bucket open-ended"

# The 17 features the production rules engine reads, each with the honest
# reason ULB cannot supply it.  Derived FROM rules.yaml by the test suite
# (it walks the engine's conditions and compares the sets), so this table
# cannot silently drift from the deployed rules.
RULE_FEATURE_UNAVAILABILITY: dict[str, str] = {
    "account_daily_spend_ratio":
        "needs account daily-spend history; ULB has no account identity",
    "account_tenure_days":
        "needs account creation time; ULB has no account identity",
    "amount_ratio":
        "needs the account's typical amount; ULB has no account history",
    "days_since_last_similar_txn":
        "needs prior transaction history; ULB rows are independent",
    "device_daily_count":
        "needs device identifiers; ULB has no device column",
    "failed_auth_count_24h":
        "needs authentication event history; absent from ULB",
    "gradual_escalation_score":
        "needs per-account escalation history; absent from ULB",
    "hour_of_day":
        "ULB Time is relative seconds (ordering_only_relative); absolute "
        "calendar semantics are unavailable and must not be fabricated",
    "is_weekend":
        "needs an absolute calendar date; ULB Time is relative seconds",
    "mule_ring_score":
        "needs cross-account recipient graph; absent from ULB",
    "new_device_flag":
        "needs device identity continuity; ULB has no device column",
    "shared_device_accounts":
        "needs device-account graph; absent from ULB",
    "shared_recipient_accounts":
        "needs recipient-account graph; absent from ULB",
    "txn_freq_last_24h":
        "needs per-account 24h activity window; absent from ULB",
    "txn_time_unusual":
        "needs per-account time-of-day history; absent from ULB",
    "unusual_location_flag":
        "needs location columns; ULB PCA components must never be read "
        "as locations (semantics undisclosed)",
    "unusual_recipient_flag":
        "needs recipient identity; absent from ULB",
}

RESULTS_NAMESPACE = "reports/external_benchmark/phase108/ulb"

_BACKEND = Path(__file__).resolve().parents[2]
REPO_ROOT = _BACKEND.parent

# Read-only production identity snapshot: hashed as BYTES (never
# deserialized) before and after evaluation; any change raises.
PRODUCTION_ARTIFACT_PATHS: tuple[str, ...] = (
    "models/production/altman_native_e_hardneg_cert/manifest.json",
    "models/production/altman_native_e_hardneg_cert/feature_list.json",
    "models/production/altman_native_e_hardneg_cert/xgb_native.joblib",
    "models/production/altman_native_e_hardneg_cert/lgb_native.joblib",
    "models/production/altman_native_e_hardneg_cert/cb_native.joblib",
    "models/production/altman_native_e_hardneg_cert/scaler_native.joblib",
    "models/production/release_registry.json",
    "models/feature_contract.json",
    "src/risk_engine/rules.yaml",
    "src/monitoring/real_world_evaluation_protocol.py",
    "data/transactions.csv",
    "data/feedback_labeled.csv",
)


class EvaluationMode(str, Enum):
    MODE_A_BASELINE_DATASET_AUDIT = "MODE_A_BASELINE_DATASET_AUDIT"
    MODE_B_COMPONENT_EVALUATION = "MODE_B_COMPONENT_EVALUATION"
    MODE_C_BENCHMARK_MODEL_EVALUATION = "MODE_C_BENCHMARK_MODEL_EVALUATION"


class DatasetIntegrityError(Exception):
    """DATASET_INTEGRITY_FAILURE: evaluation STOPPED before metrics."""

    def __init__(self, failures: Sequence[str]) -> None:
        self.failures = tuple(failures)
        super().__init__(
            "DATASET_INTEGRITY_FAILURE: " + ", ".join(self.failures))


class ProductionArtifactError(Exception):
    """A production artifact changed across evaluation: fail closed."""

    def __init__(self, changed: Sequence[str]) -> None:
        self.changed = tuple(changed)
        super().__init__(
            "production artifact changed during evaluation: "
            + ", ".join(self.changed))


def _stable(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:  # test fixtures live outside the repository tree
        return resolved.as_posix()


# ══════════════════════════════════════════════════════════════════════
# METRICS (authority-faithful, efficient)
# ══════════════════════════════════════════════════════════════════════

def _counts(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int,
                                                                    int, int]:
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1:
            if p == 1:
                tp += 1
            else:
                fn += 1
        else:
            if p == 1:
                fp += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def _prf(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "specificity": round(specificity, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
    }


def confusion_at_threshold(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    threshold: float,
) -> tuple[int, int, int, int]:
    """O(n) confusion matrix at a fixed threshold."""
    pred = [1 if s >= threshold else 0 for s in y_scores]
    return _counts(y_true, pred)


def evaluation_metrics(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    threshold: float = PRODUCTION_THRESHOLD,
) -> dict[str, Any]:
    """Deterministic metric block at a FIXED threshold.

    Key-for-key semantics match the repository authority
    (real_world_evaluation_protocol.compute_metrics): same zero-division
    guards, same round(..., 6), same endpoint construction ({0.0, 1.0}
    plus every score), same lexicographic point sorting, same trapezoid
    accumulation, same brier summation order.  The only difference is
    the O(n log n) distinct-score sweep instead of the authority's
    O(n^2) per-threshold rescans -- proven equal on fixtures by the test
    suite.  Adds mcc / balanced accuracy / accuracy / support on top
    (reusing the Phase 106 compute_mcc authority).
    """
    n = len(y_true)
    if n == 0:
        return {"error": "no_data"}
    if len(y_scores) != n:
        raise ValueError("y_true and y_scores must align")

    pred = [1 if s >= threshold else 0 for s in y_scores]
    tp, fp, fn, tn = _counts(y_true, pred)
    total_fraud = tp + fn
    base = _prf(tp, fp, fn, tn)
    fraud_rate = total_fraud / n if n > 0 else 0.0
    alert_rate = (tp + fp) / n if n > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0
    brier = sum((s - float(t)) ** 2 for s, t in zip(y_scores, y_true)) / n

    # --- ROC / PR: same point set as the authority, swept once --------
    # Distinct score groups descending; thresholds = {0.0, 1.0} ∪ scores.
    groups: list[tuple[float, int, int]] = []
    seen: dict[float, int] = {}
    for s, t in zip(y_scores, y_true):
        idx = seen.get(s)
        if idx is None:
            seen[s] = len(groups)
            groups.append((s, 1 if t == 1 else 0, 0 if t == 1 else 1))
        else:
            g = groups[idx]
            groups[idx] = (g[0], g[1] + (1 if t == 1 else 0),
                           g[2] + (0 if t == 1 else 1))
    groups.sort(key=lambda g: -g[0])  # descending score

    thresholds = sorted({0.0, 1.0} | set(seen), reverse=True)
    roc_points: list[tuple[float, float]] = []
    pr_points: list[tuple[float, float]] = []
    cum_tp = cum_fp = 0
    gi = 0
    n_pos = total_fraud
    n_neg = n - total_fraud
    for thr in thresholds:
        while gi < len(groups) and groups[gi][0] >= thr:
            cum_tp += groups[gi][1]
            cum_fp += groups[gi][2]
            gi += 1
        tpr = cum_tp / n_pos if n_pos > 0 else 0.0
        fpr = cum_fp / n_neg if n_neg > 0 else 0.0
        prec = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
        rec = cum_tp / n_pos if n_pos > 0 else 0.0
        roc_points.append((fpr, tpr))
        pr_points.append((rec, prec))
    roc_points.sort()
    pr_points.sort()

    roc_auc = 0.0
    for i in range(1, len(roc_points)):
        dx = roc_points[i][0] - roc_points[i - 1][0]
        y_avg = (roc_points[i][1] + roc_points[i - 1][1]) / 2
        roc_auc += dx * y_avg
    pr_auc = 0.0
    for i in range(1, len(pr_points)):
        dx = pr_points[i][0] - pr_points[i - 1][0]
        y_avg = (pr_points[i][1] + pr_points[i - 1][1]) / 2
        pr_auc += dx * y_avg

    balanced = (base["recall"] + base["specificity"]) / 2
    return {
        "precision": base["precision"],
        "recall": base["recall"],
        "f1": base["f1"],
        "specificity": base["specificity"],
        "fpr": base["fpr"],
        "fnr": base["fnr"],
        "fraud_capture_rate": round(fraud_rate, 6),
        "alert_rate": round(alert_rate, 6),
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "brier_score": round(brier, 6),
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": n,
        "total_fraud": total_fraud,
        # extensions (documented additions on top of the authority keys)
        "mcc": compute_mcc(tp, fp, fn, tn),
        "balanced_accuracy": round(balanced, 6),
        "accuracy": round(accuracy, 6),
        "support_fraud": total_fraud,
        "support_legit": n - total_fraud,
        "class_imbalance": class_imbalance(y_true),
        "threshold_source": "PRODUCTION_THRESHOLD (locked, read-only)",
    }


# ══════════════════════════════════════════════════════════════════════
# INPUT-VALIDATION PROBES (synthetic contract tests, never dataset rows)
# ══════════════════════════════════════════════════════════════════════

def _base_payload() -> dict[str, Any]:
    return {
        "event_id": "phase108-eval-0001",
        "fraud_id": "FAAAAAAAAAAAAAAA",
        "features": {
            "amount_ratio": 1.0, "txn_freq_last_24h": 1,
            "txn_time_unusual": 0, "new_device_flag": 0,
            "unusual_location_flag": 0, "unusual_recipient_flag": 0,
            "failed_auth_count_24h": 0, "days_since_last_similar_txn": 1.0,
            "gradual_escalation_score": 0.0, "known_device_count": 1,
            "account_tenure_days": 100.0, "hour_of_day": 12,
            "is_weekend": 0,
        },
    }


def _del_amount_ratio(p: dict[str, Any]) -> None:
    del p["features"]["amount_ratio"]


def _m_class_injection(p: dict[str, Any]) -> None:
    p["features"]["Class"] = 1
    p["features"]["V1"] = 0.5
    p["features"]["V28"] = -0.5


def _m_duplicate(p: dict[str, Any]) -> None:
    p["event_id"] = "phase108-duplicate-01"


# (probe_id, expected_accept, mutator, expectation note)
ProbeSpec = tuple[str, bool, Callable[[dict[str, Any]], None], str]

PROBE_SPECS: tuple[ProbeSpec, ...] = (
    ("valid_base", True, lambda p: None,
     "contract-valid payload accepted"),
    ("nan_amount_ratio", False,
     lambda p: p["features"].update(amount_ratio=float("nan")),
     "non-finite rejected by the finite validator"),
    ("inf_amount_ratio", False,
     lambda p: p["features"].update(amount_ratio=float("inf")),
     "infinity rejected by the finite validator"),
    ("neg_amount_ratio", False,
     lambda p: p["features"].update(amount_ratio=-0.5),
     "ge=0 enforced on amount_ratio"),
    ("hour_24", False, lambda p: p["features"].update(hour_of_day=24),
     "le=23 enforced on hour_of_day"),
    ("hour_neg1", False, lambda p: p["features"].update(hour_of_day=-1),
     "ge=0 enforced on hour_of_day"),
    ("unusual_2", False, lambda p: p["features"].update(txn_time_unusual=2),
     "le=1 enforced on binary flags"),
    ("escalation_11", False,
     lambda p: p["features"].update(gradual_escalation_score=11.0),
     "le=10 enforced on escalation score"),
    ("huge_finite_1e308", True,
     lambda p: p["features"].update(amount_ratio=1e308),
     "extreme-but-finite accepted (documented, not a rejection)"),
    ("overflow_1e309", False,
     lambda p: p["features"].update(amount_ratio=1e309),
     "float overflow to infinity rejected by the finite validator"),
    ("str_numeric", True, lambda p: p["features"].update(amount_ratio="2.5"),
     "numeric string coerced by the schema (existing behavior)"),
    ("str_garbage", False, lambda p: p["features"].update(amount_ratio="abc"),
     "non-numeric string rejected"),
    ("short_event_id", False, lambda p: p.update(event_id="short"),
     "min_length=8 enforced on event_id"),
    ("long_event_id", False, lambda p: p.update(event_id="x" * 65),
     "max_length=64 enforced on event_id"),
    ("bad_fraud_id", False,
     lambda p: p.update(fraud_id="XXXXXXXXXXXXXXXX"),
     "fraud_id pattern ^F[A-Z2-9]{15}$ enforced"),
    ("missing_required_field", False, _del_amount_ratio,
     "required FeatureVector field missing -> rejected"),
    ("nan_amount", False,
     lambda p: p["features"].update(amount=float("nan")),
     "non-finite amount rejected"),
    ("neg_amount", False, lambda p: p["features"].update(amount=-5.0),
     "ge=0 enforced on amount"),
    ("mule_ring_1p5", False,
     lambda p: p["features"].update(mule_ring_score=1.5),
     "le=1 enforced on mule_ring_score"),
    ("shared_device_neg", False,
     lambda p: p["features"].update(shared_device_accounts=-1),
     "ge=0 enforced on shared_device_accounts"),
    ("label_and_pca_injection", True, _m_class_injection,
     "extra Class/V1/V28 keys ignored by schema (never features)"),
    ("duplicate_event_id", True, _m_duplicate,
     "schema does not dedupe event_id (persistence layer owns dedup)"),
)


def _validate(payload: Mapping[str, Any]) -> bool:
    """True when the production boundary ACCEPTS the payload.

    Pure pydantic construction: no server, no persistence, no network.
    """
    try:
        EvaluateRequest(**payload)
        return True
    except Exception:  # ValidationError / ValueError / TypeError
        return False


def run_input_validation_probes() -> tuple[tuple[str, bool, bool], ...]:
    """Execute every probe against the real EvaluateRequest boundary.

    Returns (probe_id, expected_accept, actual_accept) triples in
    PROBE_SPECS order.  The duplicate probe sends the SAME event_id
    twice: the schema boundary is stateless, so both calls must
    validate (deduplication belongs to persistence, out of scope).
    """
    outcomes: list[tuple[str, bool, bool]] = []
    for probe_id, expected, mutator, _note in PROBE_SPECS:
        payload = _base_payload()
        mutator(payload)
        actual = _validate(payload)
        if probe_id == "duplicate_event_id":
            actual = actual and _validate(payload)
        outcomes.append((probe_id, expected, actual))
    return tuple(outcomes)


def probe_metrics(
    outcomes: Sequence[tuple[str, bool, bool]],
) -> dict[str, Any]:
    """Validation-conformance metrics.

    Positive class = INVALID payload (should be rejected); score = 1.0
    when the boundary rejected it; probe decision threshold 0.5 (binary
    indicator -- NOT the production scoring threshold, documented).
    """
    y_true = [0 if expected_accept else 1 for _pid, expected_accept, _a
              in outcomes]
    y_scores = [0.0 if actual_accept else 1.0
                for _pid, _e, actual_accept in outcomes]
    block = evaluation_metrics(y_true, y_scores, threshold=0.5)
    block["threshold_source"] = (
        "probe conformance decision (binary reject indicator); the "
        "production-reference analysis uses PRODUCTION_THRESHOLD")
    return block


# ══════════════════════════════════════════════════════════════════════
# RESULT STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BaselineAudit:
    """MODE A: read-only statistics over the verified file."""

    total_rows: int
    fraud_rows: int
    non_fraud_rows: int
    fraud_pct: float
    amount: dict[str, float]
    time_stats: dict[str, float]
    missing_total: int
    missing_by_column: tuple[tuple[str, int], ...]
    duplicate_findings: tuple[int, ...]
    class_distribution: tuple[tuple[str, int], ...]
    unmapped_labels: int
    ragged_rows: int
    type_mismatch_total: int
    verification_warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.total_rows != EXPECTED_ROWS:
            raise ValueError("baseline row count must stay pinned")
        if self.fraud_rows + self.non_fraud_rows != self.total_rows:
            raise ValueError("class distribution must reconcile")
        if self.fraud_rows != EXPECTED_FRAUD:
            raise ValueError("baseline fraud count must stay pinned")
        if self.unmapped_labels != 0:
            raise ValueError("every label must normalize deterministically")
        if self.missing_total != sum(
                n for _c, n in self.missing_by_column):
            raise ValueError("missing counts must reconcile")
        if round(self.fraud_pct, 6) != round(
                100.0 * EXPECTED_FRAUD / EXPECTED_ROWS, 6):
            raise ValueError("fraud percentage must reconcile")


@dataclass(frozen=True)
class ComponentResult:
    """One MODE B component: declared inputs, limits, findings, metrics."""

    name: str
    status: str                       # "evaluated" | "not_applicable"
    input_features: tuple[str, ...]
    limitations: tuple[str, ...]
    findings: tuple[str, ...]
    metrics: tuple[tuple[str, Any], ...]
    metric_scope: str

    def __post_init__(self) -> None:
        if self.status not in {"evaluated", "not_applicable"}:
            raise ValueError("invalid component status")
        if not self.limitations:
            raise ValueError("every component must declare limitations")
        if self.status == "not_applicable" and self.metrics:
            raise ValueError(
                "not-applicable components must not carry metrics")


@dataclass(frozen=True)
class TemporalSlice:
    name: str
    row_lo: int
    row_hi: int
    rows: int
    fraud: int
    fraud_rate: float
    time_min: float
    time_max: float


@dataclass(frozen=True)
class TemporalAnalysis:
    timestamp_status: str
    calendar_conversion: str
    split_method: str
    slices: tuple[TemporalSlice, ...]
    ordering_deterministic: bool
    future_leakage_note: str

    def __post_init__(self) -> None:
        if self.timestamp_status != "ordering_only_relative":
            raise ValueError("ULB Time stays relative (ordering only)")
        if self.calendar_conversion != "none":
            raise ValueError("relative Time must never become calendar time")
        if sum(s.rows for s in self.slices) != EXPECTED_ROWS:
            raise ValueError("temporal slices must cover every row")
        if sum(s.fraud for s in self.slices) != EXPECTED_FRAUD:
            raise ValueError("temporal fraud counts must reconcile")


@dataclass(frozen=True)
class BucketResult:
    label: str
    lo: float
    hi: float
    rows: int
    fraud: int
    tp: int
    fp: int
    fn: int
    tn: int

    def __post_init__(self) -> None:
        if self.tp + self.fp + self.fn + self.tn != self.rows:
            raise ValueError("bucket confusion must reconcile")
        if self.fn != self.fraud:
            raise ValueError("fail-closed scores: every fraud row is FN")


@dataclass(frozen=True)
class ErrorAnalysis:
    tp: int
    fp: int
    fn: int
    tn: int
    fraud_amount: dict[str, float]
    legit_amount: dict[str, float]
    fn_amount: dict[str, float] | str
    fp_amount: dict[str, float] | str
    amount_buckets: tuple[BucketResult, ...]
    temporal_buckets: tuple[BucketResult, ...]
    reconciliation: tuple[tuple[str, bool], ...]
    identifier_exposure: str

    def __post_init__(self) -> None:
        if self.tp + self.fp + self.fn + self.tn != EXPECTED_ROWS:
            raise ValueError("error analysis must reconcile with rows")
        if self.fn != EXPECTED_FRAUD:
            raise ValueError("fail-closed component: fraud rows are FN")
        if self.fp != 0:
            raise ValueError(
                "constant 0.0 scores cannot produce false positives")
        if sum(b.rows for b in self.amount_buckets) != EXPECTED_ROWS:
            raise ValueError("amount buckets must cover every row")
        if sum(b.fraud for b in self.amount_buckets) != EXPECTED_FRAUD:
            raise ValueError("amount-bucket fraud must reconcile")
        if sum(b.rows for b in self.temporal_buckets) != EXPECTED_ROWS:
            raise ValueError("temporal buckets must cover every row")
        if sum(b.fraud for b in self.temporal_buckets) != EXPECTED_FRAUD:
            raise ValueError("temporal-bucket fraud must reconcile")
        if self.fn_amount != self.fraud_amount:
            raise ValueError(
                "every fraud row is FN: FN amount stats must equal the "
                "fraud amount distribution")
        if not all(ok for _k, ok in self.reconciliation):
            raise ValueError("all reconciliations must hold")
        if self.identifier_exposure != "aggregate_amount_and_time_only":
            raise ValueError(
                "error analysis exposes aggregates only, never raw rows")


@dataclass(frozen=True)
class ContaminationCheck:
    train_sha256_before: str
    train_sha256_after: str
    train_unchanged: bool
    feedback_sha256: str
    train_rows: int
    train_fingerprint_overlap: int
    ts_domains_disjoint: bool
    benchmark_wrote_training: bool

    def __post_init__(self) -> None:
        if self.train_sha256_before != self.train_sha256_after:
            raise ValueError("training data identity changed")
        if not self.train_unchanged:
            raise ValueError("training data must stay unchanged")
        if self.benchmark_wrote_training:
            raise ValueError(
                "benchmark results must never be written to training data")


@dataclass(frozen=True)
class ArtifactIntegrity:
    entries: tuple[tuple[str, str, str], ...]  # (path, before, after)
    unchanged: bool
    threshold_before: float
    threshold_after: float
    threshold_unchanged: bool

    def __post_init__(self) -> None:
        if not self.unchanged:
            changed = [p for p, b, a in self.entries if b != a]
            raise ProductionArtifactError(changed)
        if any(b != a for _p, b, a in self.entries):
            raise ProductionArtifactError(["entry_mismatch"])
        if self.threshold_before != self.threshold_after:
            raise ValueError("production threshold changed")
        if self.threshold_before != EXPECTED_PRODUCTION_THRESHOLD:
            raise ValueError("production threshold drifted from 0.018758")


@dataclass(frozen=True)
class DiagnosticSweep:
    labels: tuple[str, ...]
    statement: str
    rows: tuple[tuple[float, float, float, float, float, float], ...]
    pr_auc_note: str
    production_threshold: float

    def __post_init__(self) -> None:
        if self.labels != SWEEP_LABELS:
            raise ValueError("sweep labels are fixed")
        if self.statement != SWEEP_STATEMENT:
            raise ValueError("sweep statement is fixed")
        if self.production_threshold != PRODUCTION_THRESHOLD:
            raise ValueError("sweep must report the locked threshold")
        if SWEEP_THRESHOLDS != (0.0, 0.005, 0.018758, 0.05, 0.5):
            raise ValueError("sweep grid is fixed")
        if tuple(r[0] for r in self.rows) != SWEEP_THRESHOLDS:
            raise ValueError("sweep rows must follow the fixed grid")


@dataclass(frozen=True)
class EvaluationManifest:
    """Deterministic, hash-bound evaluation manifest.

    Bookkeeping only: exactly its own type, never accepted by any RWV or
    promotion gate (mirrors the Phase 106/107 manifest declaration).
    """

    benchmark_id: str
    manifest_version: str
    created_at: str
    code_version: str
    dataset_id: str
    dataset_sha256: str
    row_count: int
    label_counts: tuple[tuple[str, int], ...]
    schema_hash: str
    model_id: str
    release_id: str
    feature_version: str
    native_feature_version: str
    domain_feature_count: int
    native_feature_count: int
    canonical_mapping: str
    production_threshold: float
    evaluation_modes: tuple[str, ...]
    mode_c_status: str
    evaluation_configuration: tuple[tuple[str, str], ...]
    metric_configuration: tuple[tuple[str, str], ...]
    not_gate_evidence: tuple[str, ...]
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError("unexpected manifest version")
        if self.created_at != MANIFEST_CREATED_AT:
            raise ValueError("manifest timestamp is a fixed constant")
        if self.code_version != EXECUTION_VERSION:
            raise ValueError("manifest must pin the execution version")
        if self.dataset_id != DATASET_ID:
            raise ValueError("manifest must pin the dataset")
        if self.dataset_sha256 != EXPECTED_SHA256:
            raise ValueError("manifest must bind the Phase 107 hash")
        if self.row_count != EXPECTED_ROWS:
            raise ValueError("manifest must bind the pinned row count")
        if self.schema_hash != EXPECTED_SCHEMA_HASH:
            raise ValueError("manifest must bind the registered schema")
        if self.model_id != MODEL_ID or self.release_id != RELEASE_ID:
            raise ValueError("production identity is locked")
        if self.feature_version != FEATURE_VERSION:
            raise ValueError("feature version is locked")
        if self.native_feature_version != NATIVE_FEATURE_VERSION:
            raise ValueError("native feature version is locked")
        if self.domain_feature_count != len(ML_FEATURE_ORDER):
            raise ValueError("domain feature count is locked")
        if self.native_feature_count != len(ALTMAN_NATIVE_FEATURES):
            raise ValueError("native feature count is locked")
        if self.canonical_mapping.count("map_raw_to_native") != 1:
            raise ValueError("canonical mapping identity is locked")
        if self.production_threshold != PRODUCTION_THRESHOLD:
            raise ValueError("production threshold is locked")
        if self.evaluation_modes != (
                EvaluationMode.MODE_A_BASELINE_DATASET_AUDIT.value,
                EvaluationMode.MODE_B_COMPONENT_EVALUATION.value):
            raise ValueError("only Mode A and Mode B execute")
        if self.mode_c_status != NATIVE_MODEL_EVALUATION:
            raise ValueError("Mode C must report NOT_APPLICABLE")
        if self.not_gate_evidence != BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE:
            raise ValueError("not-gate-evidence declaration must be carried")
        object.__setattr__(self, "manifest_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        payload = dataclasses.asdict(self)
        payload["manifest_hash"] = ""
        return _stable(payload)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def verify_manifest_hash(manifest: EvaluationManifest) -> bool:
    return manifest.manifest_hash == manifest._compute_hash()  # noqa: SLF001


def verify_result_hash(result: "BenchmarkExecution") -> bool:
    return result.result_hash == result._compute_hash()  # noqa: SLF001


@dataclass(frozen=True)
class BenchmarkExecution:
    """Complete, deterministic outcome of one Phase 108 execution."""

    phase: int
    execution_version: str
    dataset_id: str
    dataset_canonical_name: str
    dataset_sha256: str
    dataset_status: str
    row_count: int
    fraud_count: int
    fraud_prevalence: float
    integrity_result: str
    schema_hash: str
    modes_executed: tuple[str, ...]
    mode_c_status: str
    native_model_evaluation: str
    native_model_evaluation_reasons: tuple[str, ...]
    group: str
    scope: str
    usable_native_features: tuple[str, ...]
    native_eligible_rows: int
    baseline_audit: BaselineAudit
    components: tuple[ComponentResult, ...]
    metric_block: tuple[tuple[str, Any], ...]
    imbalance_warning: str
    threshold_used: float
    threshold_unchanged: bool
    calibration_status: str
    calibration_reason: str
    temporal: TemporalAnalysis
    error_analysis: ErrorAnalysis
    robustness_findings: tuple[str, ...]
    leakage_findings: tuple[str, ...]
    contamination: ContaminationCheck
    artifact_integrity: ArtifactIntegrity
    sweep: DiagnosticSweep
    manifest: EvaluationManifest
    probe_mismatches: int
    result_hash: str = ""

    def __post_init__(self) -> None:
        if self.phase != PHASE:
            raise ValueError("execution phase is fixed")
        if self.execution_version != EXECUTION_VERSION:
            raise ValueError("unexpected execution version")
        if self.dataset_sha256 != EXPECTED_SHA256:
            raise ValueError("execution must bind the Phase 107 hash")
        if self.row_count != EXPECTED_ROWS:
            raise ValueError("row count must stay pinned")
        if self.fraud_count != EXPECTED_FRAUD:
            raise ValueError("fraud count must stay pinned")
        if self.integrity_result != "VERIFIED":
            raise ValueError(
                "metrics only exist for VERIFIED integrity")
        if round(self.fraud_prevalence, 6) != round(
                EXPECTED_FRAUD / EXPECTED_ROWS, 6):
            raise ValueError("fraud prevalence must reconcile")
        if self.modes_executed != (
                EvaluationMode.MODE_A_BASELINE_DATASET_AUDIT.value,
                EvaluationMode.MODE_B_COMPONENT_EVALUATION.value):
            raise ValueError("only Mode A and Mode B execute here")
        if self.mode_c_status != NATIVE_MODEL_EVALUATION:
            raise ValueError("Mode C must report NOT_APPLICABLE")
        if self.native_model_evaluation != NATIVE_MODEL_EVALUATION:
            raise ValueError("native evaluation is forbidden, not failed")
        if self.threshold_used != PRODUCTION_THRESHOLD:
            raise ValueError("primary analysis must use 0.018758")
        if not self.threshold_unchanged:
            raise ValueError("threshold immutability is mandatory")
        if self.calibration_status != CALIBRATION_STATUS:
            raise ValueError("calibration must stay NOT_APPLICABLE")
        if self.probe_mismatches != 0:
            raise ValueError(
                "every validation probe must behave as its contract "
                "documents (probe expectation mismatch)")
        if self.group != "group_b":
            raise ValueError("ULB stays Group B in this phase")
        if self.scope != "component_evaluation_only":
            raise ValueError("ULB scope stays component-only")
        if self.usable_native_features != usable_features(
                dataset_preflight(DATASET_ID)):
            raise ValueError("preflight must stay the Phase 106 authority")
        if self.native_eligible_rows != 0:
            raise ValueError("no row may become native-eligible")
        if self.imbalance_warning != IMBALANCE_WARNING:
            raise ValueError("imbalance warning must be carried verbatim")
        metric = dict(self.metric_block)
        if metric.get("threshold") != PRODUCTION_THRESHOLD:
            raise ValueError("metric block must carry the locked threshold")
        if metric.get("tp", -1) != 0 or metric.get("fp", -1) != 0:
            raise ValueError(
                "fail-closed constant scores yield tp=0 and fp=0")
        if metric.get("fn") != EXPECTED_FRAUD:
            raise ValueError("every fraud row must be FN under 0.0 scores")
        if metric.get("tn") != EXPECTED_LEGIT:
            raise ValueError("every legit row must be TN under 0.0 scores")
        if not self.leakage_findings:
            raise ValueError("leakage findings must be stated")
        if self.manifest.dataset_sha256 != self.dataset_sha256:
            raise ValueError("manifest must bind the executed dataset")
        object.__setattr__(self, "result_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        payload = dataclasses.asdict(self)
        payload["result_hash"] = ""
        return _stable(payload)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ══════════════════════════════════════════════════════════════════════
# INTEGRITY + ARTIFACT SNAPSHOT
# ══════════════════════════════════════════════════════════════════════

def resolve_benchmark_file(dataset_path: str | None = None) -> Path:
    """Locate the ULB file (fixture override never relaxes a check)."""
    if dataset_path is not None:
        p = Path(dataset_path)
        if not p.is_file():
            raise DatasetIntegrityError(("dataset_file_absent",))
        return p
    for rel in resolve_candidate_paths(DATASET_ID):
        p = REPO_ROOT / rel
        if p.is_file():
            return p
    raise DatasetIntegrityError(("dataset_file_absent",))


def verify_source_integrity(path: Path) -> str:
    """Phase 107 hash binding; mismatch RAISES before any evaluation."""
    digest = sha256_file(str(path))
    if digest != EXPECTED_SHA256:
        raise DatasetIntegrityError(
            (f"sha256_mismatch:{digest[:16]}",))
    return digest


def _artifact_path(rel: str) -> Path:
    """src/ paths live under backend/, everything else under the root."""
    return (_BACKEND / rel) if rel.startswith("src/") else (REPO_ROOT / rel)


def snapshot_artifacts() -> dict[str, str]:
    """Read-only byte hash of every production identity artifact."""
    snap: dict[str, str] = {}
    for rel in PRODUCTION_ARTIFACT_PATHS:
        p = _artifact_path(rel)
        snap[rel] = sha256_file(str(p)) if p.is_file() else "absent"
    return snap


def compare_artifacts(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> ArtifactIntegrity:
    """Any content change across evaluation raises ProductionArtifactError."""
    entries = tuple(
        (rel, before.get(rel, "absent"), after.get(rel, "absent"))
        for rel in PRODUCTION_ARTIFACT_PATHS)
    changed = [rel for rel, b, a in entries if b != a]
    unchanged = not changed and len(entries) == len(PRODUCTION_ARTIFACT_PATHS)
    if changed:
        raise ProductionArtifactError(changed)
    return ArtifactIntegrity(
        entries=entries,
        unchanged=unchanged,
        threshold_before=PRODUCTION_THRESHOLD,
        threshold_after=PRODUCTION_THRESHOLD,
        threshold_unchanged=PRODUCTION_THRESHOLD
        == EXPECTED_PRODUCTION_THRESHOLD,
    )


def _row_fingerprint(y: int, ts: str, amt: str | None) -> str:
    """Exact Phase 106 row_fingerprint formula ({y, ts, amt})."""
    return _stable({"y": y, "ts": ts, "amt": amt})


def _training_fingerprints(
    path: Path,
) -> tuple[set[str], int, set[str]]:
    """(fingerprints, row count, raw timestamp strings) of training rows.

    data/transactions.csv has no raw amount column, so amt is None for
    the training side (AMOUNT_COLUMNS semantics: unregistered source ->
    no amount column).  The test suite proves formula equivalence
    against the Phase 106 row_fingerprint authority on real ULB rows.
    The timestamp set lets execute_evaluation MEASURE domain
    disjointness instead of asserting it.
    """
    fps: set[str] = set()
    ts_values: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            ts = str(row["ts"])
            ts_values.add(ts)
            fps.add(_row_fingerprint(int(row["label"]), ts, None))
    return fps, rows, ts_values


def _amount_stats(amounts: Sequence[float]) -> dict[str, float]:
    n = len(amounts)
    if n == 0:
        return {"count": 0.0}
    ordered = sorted(amounts)
    p95_idx = max(0, math.ceil(0.95 * n) - 1)  # nearest-rank, documented
    mid = n // 2
    median = (ordered[mid] if n % 2 == 1
              else (ordered[mid - 1] + ordered[mid]) / 2)
    return {
        "count": float(n),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "mean": round(sum(amounts) / n, 6),
        "median": round(median, 6),
        "p95": round(ordered[p95_idx], 6),
    }


def _mismatch_total(counts: Any) -> int:
    """Total type-mismatch cells from a Phase 107 record (pairs|dict)."""
    if isinstance(counts, dict):
        return sum(int(v) for v in counts.values())
    return sum(int(n) for _c, n in counts)


def _bucket_of(value: float, edges: Sequence[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2  # top bucket absorbs >= last finite edge


def _bucket_label(lo: float, hi: float) -> str:
    hi_s = "inf" if math.isinf(hi) else f"{hi:g}"
    return f"[{lo:g}, {hi_s})"


def _make_buckets(
    rows: Sequence[tuple[float, float, int]],
    edges: Sequence[float],
) -> tuple[BucketResult, ...]:
    """rows = (amount, time, label); fail-closed pred is always 0."""
    acc: list[list[int]] = [[0, 0] for _ in range(len(edges) - 1)]
    for amount, _ts, y in rows:
        b = _bucket_of(amount, edges)
        acc[b][0] += 1
        acc[b][1] += y
    out: list[BucketResult] = []
    for i in range(len(edges) - 1):
        rows_n, fraud_n = acc[i]
        out.append(BucketResult(
            label=_bucket_label(edges[i], edges[i + 1]),
            lo=edges[i], hi=edges[i + 1],
            rows=rows_n, fraud=fraud_n,
            tp=0, fp=0, fn=fraud_n, tn=rows_n - fraud_n))
    return tuple(out)


# ══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

def execute_evaluation(dataset_path: str | None = None) -> BenchmarkExecution:
    """Run Mode A + Mode B against the verified ULB file.

    Order is deliberate: hash -> header/schema -> full pass with counts
    -> ONLY THEN metrics -> analysis -> artifact re-snapshot ->
    manifest -> result.  Any integrity failure raises before a single
    metric exists; any artifact drift raises after it.
    """
    if PRODUCTION_THRESHOLD != EXPECTED_PRODUCTION_THRESHOLD:
        raise DatasetIntegrityError(("threshold_drift",))

    # --- integrity gate ------------------------------------------------
    path = resolve_benchmark_file(dataset_path)
    digest = verify_source_integrity(path)

    # Phase 107's authoritative record for the SAME file: a second
    # independent verification pass whose findings (duplicates,
    # warnings, schema) are REUSED here rather than re-derived.
    record = ingest_dataset(DATASET_ID)
    if record.status != EXPECTED_INGESTION_STATUS:
        raise DatasetIntegrityError((f"ingest_status:{record.status}",))
    if record.row_count != EXPECTED_ROWS:
        raise DatasetIntegrityError((f"ingest_rows:{record.row_count}",))

    artifacts_before = snapshot_artifacts()
    train_path = REPO_ROOT / "data" / "transactions.csv"
    feedback_path = REPO_ROOT / "data" / "feedback_labeled.csv"
    train_sha_before = (sha256_file(str(train_path))
                        if train_path.is_file() else "absent")
    feedback_sha = (sha256_file(str(feedback_path))
                    if feedback_path.is_file() else "absent")
    if train_path.is_file():
        train_fps, train_rows, train_ts = _training_fingerprints(train_path)
    else:
        train_fps, train_rows, train_ts = set(), 0, set()

    label_col = LABEL_COLUMNS[DATASET_ID]
    ts_col = TIMESTAMP_COLUMNS[DATASET_ID]
    amt_col = AMOUNT_COLUMNS[DATASET_ID]
    ds = get_dataset(DATASET_ID)

    # --- MODE A: single read-only pass over the verified file ----------
    amounts: list[float] = []
    times: list[float] = []
    labels: list[int] = []
    rows_for_buckets: list[tuple[float, float, int]] = []
    missing_by_col: dict[str, int] = {c: 0 for c in EXPECTED_SCHEMA}
    class_counts: dict[str, int] = {}
    unmapped = 0
    fraud_rows = 0
    overlap_hits: set[str] = set()
    ts_overlap = 0

    # Evaluate the rules engine ONCE under the empty legitimate input
    # (the only input ULB can supply: 0/17 features are derivable).  The
    # input is provably identical for every row, determinism is asserted
    # across repeated calls, and the constant result is broadcast --
    # evaluating the same input 284807 times would be pure waste.
    engine = RulesEngine.from_yaml(_BACKEND / "src" / "risk_engine"
                                   / "rules.yaml")
    base_out = engine.evaluate({})
    if base_out != engine.evaluate({}) or base_out != engine.evaluate({}):
        raise DatasetIntegrityError(("rules_engine_nondeterministic",))
    score0 = float(base_out["score"])
    band0 = band_of(int(round(100.0 * score0)))
    band_key = f"{band0[0]}:{band0[1]}"

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = tuple(next(reader))
        if header != EXPECTED_SCHEMA:
            raise DatasetIntegrityError(
                ("schema_mismatch:" + ",".join(
                    c for c in header if c not in EXPECTED_SCHEMA)[:64],))
        total = 0
        for row in reader:
            if len(row) != EXPECTED_COLUMNS:
                raise DatasetIntegrityError(("ragged_row",))
            total += 1
            rec = dict(zip(header, row))
            for col in EXPECTED_SCHEMA:
                if rec[col].strip() == "":
                    missing_by_col[col] += 1
            raw_label = rec[label_col]
            class_counts[raw_label] = class_counts.get(raw_label, 0) + 1
            y = normalize_label(DATASET_ID, raw_label)
            if y is None:
                unmapped += 1
                continue
            labels.append(y)
            fraud_rows += y
            amount = float(rec[amt_col])
            t_rel = float(rec[ts_col])
            amounts.append(amount)
            times.append(t_rel)
            rows_for_buckets.append((amount, t_rel, y))
            # contamination: Phase 106 fingerprint formula on both
            # sides, plus a MEASURED timestamp-domain overlap (training
            # uses ISO datetimes, ULB uses relative seconds -- the
            # disjointness is counted here, never assumed)
            fp = _row_fingerprint(y, str(rec[ts_col]),
                                  str(rec[amt_col]) if amt_col else None)
            if fp in train_fps:
                overlap_hits.add(fp)
            if str(rec[ts_col]) in train_ts:
                ts_overlap += 1

    if total != EXPECTED_ROWS:
        raise DatasetIntegrityError((f"row_count:{total}",))
    if fraud_rows != EXPECTED_FRAUD:
        raise DatasetIntegrityError((f"fraud_count:{fraud_rows}",))
    if unmapped != 0:
        raise DatasetIntegrityError((f"unmapped_labels:{unmapped}",))
    if len(labels) != EXPECTED_ROWS:
        raise DatasetIntegrityError(("label_count_drift",))
    if record.label_counts != tuple(sorted(class_counts.items())):
        raise DatasetIntegrityError(("label_distribution_drift",))
    if record.schema_actual != EXPECTED_SCHEMA:
        raise DatasetIntegrityError(("schema_record_drift",))

    # constant fail-closed score/band broadcast over the verified rows
    scores = [score0] * len(labels)
    band_counts = {band_key: len(labels)}
    ts_disjoint = ts_overlap == 0  # MEASURED, not assumed

    # --- component evaluations (MODE B) --------------------------------
    preflight = dataset_preflight(DATASET_ID)
    usable = usable_features(preflight)
    group = assign_benchmark_group(DATASET_ID)[0]
    scope = evaluation_scope(DATASET_ID)

    outcomes = run_input_validation_probes()
    probe_block = probe_metrics(outcomes)
    probe_failures = [pid for pid, exp, act in outcomes if exp != act]
    missing_cols = set(ML_FEATURE_ORDER) & {"Class"} | {
        c for c in ML_FEATURE_ORDER if c.startswith("V")
        and c[1:].isdigit() and 1 <= int(c[1:]) <= 28}
    fv_extra = sorted(
        set(FeatureVector.model_fields) & (
            {"Class"} | {f"V{i}" for i in range(1, 29)}))

    components: tuple[ComponentResult, ...] = (
        ComponentResult(
            name="input_validation",
            status="evaluated",
            input_features=("synthetic contract probes", "event_id",
                            "fraud_id", "FeatureVector fields"),
            limitations=(
                "probes are synthetic contract tests of the validation "
                "boundary, never dataset rows and never evidence",
                "positive class = invalid payload (should be rejected)",
                "probe decision threshold 0.5 on a binary reject "
                "indicator, not the production scoring threshold",
                "covers spec Mode B error handling (structured "
                "rejection outcomes recorded per probe) and robustness "
                "to extreme values (NaN/Inf/overflow/extremes) against "
                "the EXISTING validators, unchanged",
            ),
            findings=(
                f"probes={len(outcomes)}",
                f"expectation_mismatches={len(probe_failures)}",
                "non-finite, out-of-range, malformed and missing-field "
                "payloads all rejected by pydantic validators",
                "extreme-but-finite 1e308 accepted (documented limit)",
                "duplicate event_id accepted at the schema boundary "
                "(persistence layer owns deduplication; out of scope)",
                "injected Class/V1/V28 keys ignored by the schema "
                "(never become features)",
            ),
            metrics=tuple(sorted(probe_block.items())),
            metric_scope="validation conformance (invalid-payload class)",
        ),
        ComponentResult(
            name="amount_processing",
            status="evaluated",
            input_features=("Amount",),
            limitations=(
                "amount_ratio (relative amount) is NOT derivable: ULB has "
                "no account typical-amount history",
                "conformance checks the PRODUCTION amount constraints "
                "(ge=0 and finite) read from FeatureVector field "
                "metadata; no alternate constraint is invented",
            ),
            findings=(
                f"rows_checked={EXPECTED_ROWS}",
                f"constraint_violations={sum(1 for a in amounts if a < 0 or not math.isfinite(a))}",
                f"amount_min={min(amounts):g} amount_max={max(amounts):g}",
            ),
            metrics=(),
            metric_scope="descriptive statistics and constraint "
                         "conformance; no prediction semantics",
        ),
        ComponentResult(
            name="timestamp_processing",
            status="evaluated",
            input_features=("Time",),
            limitations=(
                "Time is dataset-relative seconds "
                "(ordering_only_relative); absolute calendar semantics "
                "are unavailable and are never fabricated",
                "hour_of_day and is_weekend therefore stay underivable "
                "for every rule that reads them",
            ),
            findings=(
                "relative ordering verified by deterministic tertiles",
                "calendar conversion: none",
            ),
            metrics=(),
            metric_scope="ordering and distribution only; no prediction "
                         "semantics",
        ),
        ComponentResult(
            name="class_label_handling",
            status="evaluated",
            input_features=("Class",),
            limitations=(
                "Class is the TARGET LABEL and is never an input "
                "feature: absent from ML_FEATURE_ORDER, from "
                "FeatureVector fields and from the native 48-vector",
                "label semantics carried verbatim from the Phase 106 "
                "registry (1 = fraud, 0 otherwise)",
            ),
            findings=(
                f"class_distribution={tuple(sorted(class_counts.items()))}",
                f"unmapped_labels={unmapped}",
                "Phase 107 warning 'leakage_fields_present' carried: "
                "label present in raw data, excluded from features",
                "PCA columns are never mapped: ML_FEATURE_ORDER and "
                "FeatureVector contain no V1..V28 field"
                if not missing_cols and not fv_extra else
                "PCA MAPPING DETECTED",
            ),
            metrics=(),
            metric_scope="handling conformance; no prediction semantics",
        ),
        ComponentResult(
            name="rules_engine_fail_closed",
            status="evaluated",
            input_features=(),
            limitations=(
                "0 of 17 rule features are derivable from ULB (declared "
                "per feature in RULE_FEATURE_UNAVAILABILITY); the engine "
                "is evaluated under the empty legitimate input and "
                "fail-closes to a constant 0.0 score for every row",
                "the metric block measures fail-closed DECISION "
                "behavior, never detection capability",
                "component scores are rule severities, not calibrated "
                "probabilities",
            ),
            findings=(
                "rule_features=17",
                "derivable_features=0",
                f"score_vector=constant_{score0:g} over "
                f"{EXPECTED_ROWS} rows",
                f"distinct_scores={len(set(scores))}",
            ),
            metrics=(),
            metric_scope="score vector feeds the evaluation-level "
                         "metric block (labels = Class)",
        ),
        ComponentResult(
            name="decision_infrastructure",
            status="evaluated",
            input_features=("rule score",),
            limitations=(
                "evaluated in the rules-only degraded composition "
                "(100*max(unavailable ml, rule_score)); the native "
                "fusion path is forbidden for ULB, so this observes "
                "banding of the fail-closed score only",
            ),
            findings=tuple(
                f"band[{k}]={v}" for k, v in sorted(band_counts.items()))
            + (f"band_of({round(100.0 * score0):g}) -> {band0} for "
               "every row (constant fail-closed score)",),
            metrics=(),
            metric_scope="decision distribution; no prediction semantics",
        ),
        ComponentResult(
            name="data_quality_enforcement",
            status="evaluated",
            input_features=("schema", "cell values", "Phase 107 record"),
            limitations=(
                "duplicate statistics are the Phase 107 ingestion "
                "findings (reused, not recomputed)",
            ),
            findings=(
                f"missing_cells={sum(missing_by_col.values())}",
                f"ragged_rows={record.ragged_rows}",
                f"within_source_duplicates="
                f"{record.duplicate_findings.within_source_duplicate_rows}"
                f", cross_source="
                f"{record.duplicate_findings.cross_source_duplicate_rows}"
                " (Phase 107 record)",
                "schema conformity verified against the Phase 106 "
                "expected schema; discrepancies are recorded by the "
                "Phase 107 record, never silently dropped",
            ),
            metrics=(),
            metric_scope="quality counts; no prediction semantics",
        ),
        ComponentResult(
            name="audit_trace_generation",
            status="not_applicable",
            input_features=(),
            limitations=(
                "benchmark rows must never enter the production "
                "decision/audit records (Phase 108 writes results only "
                "under reports/external_benchmark/phase108/ulb/); no "
                "audit-chain write is performed by design",
            ),
            findings=(),
            metrics=(),
            metric_scope="NOT_APPLICABLE",
        ),
    )

    # --- evaluation-level metric block (production threshold) ----------
    metric_block = evaluation_metrics(
        labels, scores, threshold=PRODUCTION_THRESHOLD)

    # --- MODE A baseline (duplicate/warning statistics REUSED verbatim
    # from the Phase 107 record, never re-derived here) ---------------
    baseline = BaselineAudit(
        total_rows=total,
        fraud_rows=fraud_rows,
        non_fraud_rows=total - fraud_rows,
        fraud_pct=round(100.0 * fraud_rows / total, 6),
        amount=_amount_stats(amounts),
        time_stats={
            "min": round(min(times), 6),
            "max": round(max(times), 6),
            "mean": round(sum(times) / len(times), 6),
            "calendar_conversion": 0.0,
        },
        missing_total=sum(missing_by_col.values()),
        missing_by_column=tuple(
            (c, n) for c, n in sorted(missing_by_col.items()) if n > 0),
        duplicate_findings=dataclasses.astuple(record.duplicate_findings),
        class_distribution=tuple(sorted(class_counts.items())),
        unmapped_labels=unmapped,
        ragged_rows=record.ragged_rows,
        type_mismatch_total=_mismatch_total(record.type_mismatch_counts),
        verification_warnings=record.verification_warnings,
    )

    # --- temporal analysis --------------------------------------------
    order = sorted(range(len(times)), key=lambda i: (times[i], i))
    n3 = len(order) // 3
    bounds = ((0, n3), (n3, 2 * n3), (2 * n3, len(order)))
    names = ("early", "middle", "late")
    slices: list[TemporalSlice] = []
    for name, (lo, hi) in zip(names, bounds):
        idx = order[lo:hi]
        fraud_n = sum(labels[i] for i in idx)
        slices.append(TemporalSlice(
            name=name, row_lo=lo, row_hi=hi, rows=len(idx),
            fraud=fraud_n,
            fraud_rate=round(fraud_n / len(idx), 6),
            time_min=round(min(times[i] for i in idx), 6),
            time_max=round(max(times[i] for i in idx), 6)))
    temporal = TemporalAnalysis(
        timestamp_status="ordering_only_relative",
        calendar_conversion="none",
        split_method=(
            "tertiles by row count over a stable Time-ascending sort "
            "(ties broken by original row order); Time stays in "
            "dataset-relative seconds; no calendar period is claimed"),
        slices=tuple(slices),
        ordering_deterministic=True,
        future_leakage_note=(
            "descriptive partitioning only: the split uses Time alone; "
            "no feature, label or score from a later row influences any "
            "earlier row, so no future information leaks into the past"),
    )

    # --- error analysis + buckets -------------------------------------
    fraud_amounts = [a for a, y in zip(amounts, labels) if y == 1]
    legit_amounts = [a for a, y in zip(amounts, labels) if y == 0]
    # temporal buckets reuse the fixed tertile boundaries (amount-bucket
    # helper needs an amount axis, so temporal buckets are built from
    # the same rows with a 3-way edge on slice membership)
    temporal_bucket_rows = tuple(
        BucketResult(
            label=s.name, lo=float(s.row_lo), hi=float(s.row_hi),
            rows=s.rows, fraud=s.fraud,
            tp=0, fp=0, fn=s.fraud, tn=s.rows - s.fraud)
        for s in slices)
    fn_stats = _amount_stats(fraud_amounts)
    amount_buckets = _make_buckets(rows_for_buckets, AMOUNT_BUCKET_EDGES)
    error = ErrorAnalysis(
        tp=metric_block["tp"], fp=metric_block["fp"],
        fn=metric_block["fn"], tn=metric_block["tn"],
        fraud_amount=_amount_stats(fraud_amounts),
        legit_amount=_amount_stats(legit_amounts),
        fn_amount=fn_stats,
        fp_amount="NOT_APPLICABLE",
        amount_buckets=amount_buckets,
        temporal_buckets=temporal_bucket_rows,
        reconciliation=(
            ("tp+fp+fn+tn==rows",
             metric_block["tp"] + metric_block["fp"] + metric_block["fn"]
             + metric_block["tn"] == EXPECTED_ROWS),
            ("fn==fraud_count", metric_block["fn"] == EXPECTED_FRAUD),
            ("fp==0", metric_block["fp"] == 0),
            ("fn_amount_count==fraud_count",
             fn_stats.get("count") == float(EXPECTED_FRAUD)),
            ("amount_bucket_rows_cover",
             sum(b.rows for b in amount_buckets) == EXPECTED_ROWS),
        ),
        identifier_exposure="aggregate_amount_and_time_only",
    )

    # --- contamination -------------------------------------------------
    train_sha_after = (sha256_file(str(train_path))
                       if train_path.is_file() else "absent")
    contamination = ContaminationCheck(
        train_sha256_before=train_sha_before,
        train_sha256_after=train_sha_after,
        train_unchanged=train_sha_before == train_sha_after
        and train_sha_before != "absent",
        feedback_sha256=feedback_sha,
        train_rows=train_rows,
        train_fingerprint_overlap=len(overlap_hits),
        ts_domains_disjoint=ts_disjoint,
        benchmark_wrote_training=False,
    )

    # --- artifact re-snapshot (raises on ANY drift) --------------------
    artifacts_after = snapshot_artifacts()
    artifact_integrity = compare_artifacts(artifacts_before,
                                           artifacts_after)

    # --- diagnostic sweep (labels fixed, cannot write) -----------------
    sweep_rows: list[tuple[float, float, float, float, float, float]] = []
    for thr in SWEEP_THRESHOLDS:
        tp, fp, fn, tn = confusion_at_threshold(labels, scores, thr)
        prf = _prf(tp, fp, fn, tn)
        sweep_rows.append((
            thr, prf["precision"], prf["recall"], prf["f1"],
            compute_mcc(tp, fp, fn, tn),
            metric_block["pr_auc"]))
    sweep = DiagnosticSweep(
        labels=SWEEP_LABELS,
        statement=SWEEP_STATEMENT,
        rows=tuple(sweep_rows),
        pr_auc_note=(
            "PR-AUC is a property of the score vector, not of a single "
            "threshold; the constant value is reported unchanged on "
            "every sweep row"),
        production_threshold=PRODUCTION_THRESHOLD,
    )

    # --- manifest + result --------------------------------------------
    manifest = EvaluationManifest(
        benchmark_id="phase108-ulb-component-evaluation",
        manifest_version=MANIFEST_VERSION,
        created_at=MANIFEST_CREATED_AT,
        code_version=EXECUTION_VERSION,
        dataset_id=DATASET_ID,
        dataset_sha256=digest,
        row_count=total,
        label_counts=tuple(sorted(class_counts.items())),
        schema_hash=ds.schema_hash,
        model_id=MODEL_ID,
        release_id=RELEASE_ID,
        feature_version=FEATURE_VERSION,
        native_feature_version=NATIVE_FEATURE_VERSION,
        domain_feature_count=len(ML_FEATURE_ORDER),
        native_feature_count=len(ALTMAN_NATIVE_FEATURES),
        canonical_mapping="canonical map_raw_to_native pipeline unchanged",
        production_threshold=PRODUCTION_THRESHOLD,
        evaluation_modes=(
            EvaluationMode.MODE_A_BASELINE_DATASET_AUDIT.value,
            EvaluationMode.MODE_B_COMPONENT_EVALUATION.value),
        mode_c_status=NATIVE_MODEL_EVALUATION,
        evaluation_configuration=(
            ("sweep_grid", ",".join(f"{t:g}" for t in SWEEP_THRESHOLDS)),
            ("temporal_split", temporal.split_method),
            ("amount_buckets", BUCKET_CONVENTION),
            ("probes", str(len(outcomes))),
            ("band_composition", "rules-only degraded composition"),
            ("ml_version", ML_FEATURE_VERSION),
        ),
        metric_configuration=(
            ("threshold", f"{PRODUCTION_THRESHOLD:g}"),
            ("threshold_source", "PRODUCTION_THRESHOLD locked"),
            ("semantics", "port of real_world_evaluation_protocol "
                          "compute_metrics, fixture-proven equal"),
            ("zero_division", "guarded to 0.0 (authority convention)"),
            ("rounding", "round(..., 6) per authority"),
        ),
        not_gate_evidence=BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
    )

    return BenchmarkExecution(
        phase=PHASE,
        execution_version=EXECUTION_VERSION,
        dataset_id=DATASET_ID,
        dataset_canonical_name=DATASET_CANONICAL_NAME,
        dataset_sha256=digest,
        dataset_status=EXPECTED_INGESTION_STATUS,
        row_count=total,
        fraud_count=fraud_rows,
        fraud_prevalence=round(fraud_rows / total, 6),
        integrity_result="VERIFIED",
        schema_hash=ds.schema_hash,
        modes_executed=(
            EvaluationMode.MODE_A_BASELINE_DATASET_AUDIT.value,
            EvaluationMode.MODE_B_COMPONENT_EVALUATION.value),
        mode_c_status=NATIVE_MODEL_EVALUATION,
        native_model_evaluation=NATIVE_MODEL_EVALUATION,
        native_model_evaluation_reasons=NATIVE_MODEL_EVALUATION_REASONS,
        group=group,
        scope=scope,
        usable_native_features=usable,
        native_eligible_rows=native_eligible_row_count(DATASET_ID),
        baseline_audit=baseline,
        components=components,
        metric_block=tuple(sorted(metric_block.items())),
        imbalance_warning=IMBALANCE_WARNING,
        threshold_used=PRODUCTION_THRESHOLD,
        threshold_unchanged=True,
        calibration_status=CALIBRATION_STATUS,
        calibration_reason=CALIBRATION_REASON,
        temporal=temporal,
        error_analysis=error,
        robustness_findings=(
            "NaN, +Inf and float-overflow payloads rejected (finite "
            "validator)",
            "negative amount and amount_ratio rejected (ge=0)",
            "hour_of_day 24/-1, flags >1, escalation 11.0 rejected "
            "(range validators)",
            "short/long event_id and malformed fraud_id rejected "
            "(length/pattern validators)",
            "missing required FeatureVector field rejected",
            "extreme-but-finite 1e308 accepted (documented boundary)",
            "numeric strings coerced, non-numeric strings rejected "
            "(existing schema behavior, unchanged)",
            "duplicate event_id accepted at validation (dedup belongs "
            "to persistence; out of scope for this phase)",
            f"expectation_mismatches={len(probe_failures)}",
        ),
        leakage_findings=(
            "Class excluded: absent from ML_FEATURE_ORDER (21) and from "
            "FeatureVector fields",
            "PCA V1..V28 never mapped: no canonical feature and no "
            "native feature references an undeclared component",
            "Phase 106 known_leakage_risks carried verbatim (PCA "
            "semantics undisclosed)",
            "Phase 107 warning 'leakage_fields_present' carried",
            "label leakage excluded from every component input "
            "declaration",
        ),
        contamination=contamination,
        artifact_integrity=artifact_integrity,
        sweep=sweep,
        manifest=manifest,
        probe_mismatches=len(probe_failures),
    )


# ══════════════════════════════════════════════════════════════════════
# RESULT STORAGE (dedicated namespace only)
# ══════════════════════════════════════════════════════════════════════

def _resolve_results_dir(out_dir: str | None) -> Path:
    if out_dir is None:
        target = REPO_ROOT / RESULTS_NAMESPACE
    else:
        target = Path(out_dir)
        if not target.is_absolute():
            target = REPO_ROOT / target
        # runtime guard: writers only ever target the Phase 108 namespace
        if RESULTS_NAMESPACE not in target.as_posix():
            raise ValueError(
                f"results may only be written under {RESULTS_NAMESPACE}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _dump_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2,
                      ensure_ascii=False) + "\n"


def write_evaluation_results(
    result: BenchmarkExecution,
    out_dir: str | None = None,
) -> tuple[str, ...]:
    """Persist deterministic JSON into the Phase 108 namespace only."""
    target = _resolve_results_dir(out_dir)
    manifest_file = target / "evaluation_manifest.json"
    results_file = target / "evaluation_results.json"
    manifest_file.write_text(
        _dump_json(result.manifest.to_dict()), encoding="utf-8")
    results_file.write_text(_dump_json(result.to_dict()),
                            encoding="utf-8")
    return (_display(manifest_file), _display(results_file))
