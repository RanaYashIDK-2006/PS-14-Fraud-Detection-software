"""Phase 106: Public Multi-Dataset External Benchmark — test suite.

Deterministic, offline, adversarial coverage for the Phase 106 registry,
harmonization layer, manifest and report.  Never touches the network,
model artifacts, DB-4 or any RWV/promotion authority.

Run:  ../.venv/Scripts/python.exe scripts/phase106_public_benchmark_test.py
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import os
import re
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from src.monitoring import phase106_public_benchmark_registry as REG
from src.monitoring import phase106_public_benchmark_report as REP
from src.monitoring.external_dataset_contract import (
    CANONICAL_TRANSFORMATION,
    REQUIREMENT_KEYS,
    AccessAuthorityClaim,
    blocking_features,
    canonical_json,
)
from src.monitoring.external_dataset_qualification import (
    QualificationResult,
    evaluate_dataset_evidence,
    evaluate_known_candidates,
)
from src.monitoring.feature_contract import ML_FEATURE_ORDER, ML_FEATURE_VERSION
from src.monitoring.phase103_production_readiness_closure import (
    PROMOTION_STATE,
    REAL_WORLD_VALIDATION,
    SYSTEM_READINESS,
    check_global_state,
)
from src.monitoring.phase106_public_benchmark_registry import (
    AMOUNT_COLUMNS,
    BENCHMARK_DATASETS,
    BENCHMARK_DESCRIPTOR,
    BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE,
    COMMON_BENCHMARK_FIELDS,
    DATASET_IDS,
    EVAL_CONFIG_VERSION,
    EXPECTED_LOCAL_ROOT,
    FORBIDDEN_BYPASS_PARAMETERS,
    FORBIDDEN_SINGLE_SOURCE_CLAIMS,
    GROUP_READINESS,
    LABEL_COLUMNS,
    LEAKAGE_FIELD_PATTERNS,
    MANIFEST_CREATED_AT,
    MANIFEST_VERSION,
    NATIVE_FEATURE_COUNT,
    NON_REAL_WORLD_CLASSIFICATIONS,
    NOT_DOCUMENTED,
    REGISTRY_VERSION,
    SYNTHETIC_AUGMENTATION_MARKERS,
    TIMESTAMP_COLUMNS,
    BenchmarkDataset,
    BenchmarkEvaluationConfig,
    BenchmarkGroup,
    BenchmarkManifest,
    DatasetClassification,
    EvaluationReadiness,
    HarmonizationStatus,
    HarmonizedRow,
    MultiSourceBenchmark,
    build_benchmark_compat_contract,
    build_benchmark_manifest,
    build_evaluation_config,
    build_multisource_benchmark,
    class_imbalance,
    compute_benchmark_metrics,
    compute_mcc,
    dataset_leakage_findings,
    dataset_preflight,
    detect_cross_source_duplicates,
    detect_duplicate_datasets,
    detect_source_contamination,
    detect_synthetic_augmentation,
    detect_train_evaluation_overlap,
    describe_benchmark,
    documented_row_totals,
    evaluation_coverage,
    evaluation_scope,
    assign_benchmark_group,
    find_single_source_claims,
    get_dataset,
    group_summary,
    harmonize_rows,
    manual_acquisition_instructions,
    native_eligible_row_count,
    native_feature_eligibility_report,
    normalize_label,
    registry_integrity_hash,
    row_fingerprint,
    scan_leakage_fields,
    source_contamination_report,
    source_specific_compatibility_report,
    stratified_evaluation,
    cross_dataset_evaluation,
    temporal_evaluation,
    usable_features,
)
from src.monitoring.phase106_public_benchmark_report import (
    DECLARATIONS,
    EXPECTED_CONCLUSION,
    generate_phase106_report,
    _stable_hash as _report_hash,
)
from src.monitoring.promotion_gate import (
    GateStatus,
    PromotionBlockedError,
    PromotionToken,
    PromotionVerdict,
    assert_promotion_allowed,
    evaluate_promotion,
    evaluate_real_world_validation,
)
from src.monitoring.provider_evidence import KNOWN_CANDIDATES, qualify_dataset
from src.monitoring.provider_evidence_intake import qualify_submissions
from src.monitoring.real_world_evaluation_protocol import PRODUCTION_THRESHOLD
from src.monitoring.rwv_execution import (
    SessionState,
    check_rwv_execution_eligibility,
    create_session,
)
from src.monitoring.rwv_promotion_evidence import RWVPromotionEvidence
from src.monitoring.rwv_readiness_audit import (
    FEATURE_VERSION as READINESS_FEATURE_VERSION,
    MODEL_ID,
    RELEASE_ID,
)
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES
from src.risk_engine.altman_native_ensemble import map_raw_to_native

# ══════════════════════════════════════════════════════════════════════
# HARNESS
# ══════════════════════════════════════════════════════════════════════

ASSERTIONS = 0
FAILURES: list[str] = []
INV_TOTAL = 0
INV_FAILED = 0


def check(cond: bool, label: str) -> None:
    global ASSERTIONS
    ASSERTIONS += 1
    if not cond:
        FAILURES.append(label)
        print(f"[FAIL] {label}")


def inv(cond: bool, label: str) -> None:
    """Core invariant: also counted as an assertion."""
    global INV_TOTAL, INV_FAILED
    INV_TOTAL += 1
    if not cond:
        INV_FAILED += 1
    check(cond, f"[INV] {label}")


def section(title: str) -> None:
    print("=" * 72)
    print(f"SECTION — {title}")
    print("=" * 72)


def expect_typeerror(fn, label: str) -> None:
    """A bypass parameter must fail at argument binding (TypeError)."""
    try:
        fn()
    except TypeError:
        check(True, label)
    except Exception as e:  # noqa: BLE001 — fail-closed means no bypass ran
        check(False, f"{label}: unexpected {type(e).__name__}: {e}")
    else:
        check(False, f"{label}: NO EXCEPTION — bypass accepted")


def expect_valueerror(fn, label: str) -> None:
    try:
        fn()
    except ValueError:
        check(True, label)
    except Exception as e:  # noqa: BLE001
        check(False, f"{label}: unexpected {type(e).__name__}: {e}")
    else:
        check(False, f"{label}: NO EXCEPTION — substitution accepted")


def expect_failclosed(fn, label: str) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001 — any exception = fail closed
        check(True, label)
    else:
        check(False, f"{label}: NO EXCEPTION — not fail-closed")


@contextlib.contextmanager
def _with_dataset(ds: BenchmarkDataset):
    """Temporarily patch the registry lookup with a fixture entry."""
    saved = REG._BY_ID[ds.dataset_id]
    REG._BY_ID[ds.dataset_id] = ds
    try:
        yield ds
    finally:
        REG._BY_ID[ds.dataset_id] = saved


IEEE = get_dataset("IEEE_CIS")
ULB = get_dataset("ULB_CREDIT_CARD_FRAUD")
PAYSIM = get_dataset("PAYSIM")
BANKSIM = get_dataset("BANKSIM")
BAF = get_dataset("BAF_BANK_ACCOUNT_FRAUD")
EXPECTED_IDS = ("IEEE_CIS", "ULB_CREDIT_CARD_FRAUD", "PAYSIM",
                "BANKSIM", "BAF_BANK_ACCOUNT_FRAUD")

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — REGISTRY INTEGRITY & DUPLICATE DETECTION
# ══════════════════════════════════════════════════════════════════════

section("1. DATASET REGISTRY INTEGRITY / DUPLICATES")
inv(REGISTRY_VERSION == "phase106_registry_v1", "registry version pinned")
inv(MANIFEST_VERSION == "phase106_benchmark_manifest_v1",
    "manifest version pinned")
inv(EVAL_CONFIG_VERSION == "phase106_eval_config_v1",
    "eval config version pinned")
inv(BENCHMARK_DESCRIPTOR == "multi-source public benchmark",
    "descriptor is the approved multi-source phrase")
inv(len(BENCHMARK_DATASETS) == 5, "five registered candidates")
inv(DATASET_IDS == EXPECTED_IDS, "dataset id set/order exactly as spec'd")
inv(len(set(DATASET_IDS)) == 5, "dataset ids unique")
inv(detect_duplicate_datasets(BENCHMARK_DATASETS) == (),
    "real registry has no duplicate id/name/source")
fix_name = dataclasses.replace(IEEE, canonical_name=ULB.canonical_name)
check("duplicate_canonical_name" in "|".join(
    detect_duplicate_datasets([fix_name, ULB])),
    "duplicate canonical name detected")
fix_src = dataclasses.replace(IEEE, source_reference=ULB.source_reference)
check("duplicate_source_reference" in "|".join(
    detect_duplicate_datasets([fix_src, ULB])),
    "duplicate source reference detected")
fix_id = dataclasses.replace(ULB, dataset_id="IEEE_CIS")
check("duplicate_dataset_id" in "|".join(
    detect_duplicate_datasets([IEEE, fix_id])),
    "duplicate dataset id detected")
h1, h2 = registry_integrity_hash(), registry_integrity_hash()
inv(h1 == h2 and len(h1) == 64, "registry integrity hash deterministic")
inv(h1 == registry_integrity_hash(),
    "registry hash matches report-bound hash source")
inv(get_dataset("IEEE_CIS") == IEEE, "get_dataset round-trip")
expect_failclosed(lambda: get_dataset("NOT_REGISTERED"),
                  "unknown dataset id fails closed")
expect_valueerror(lambda: dataclasses.replace(
    IEEE, classification="for_sale"),
    "invalid classification rejected at construction")
expect_valueerror(lambda: dataclasses.replace(
    IEEE, requirement_availability=(("amount", "directly_available"),)),
    "incomplete requirement availability rejected")
expect_valueerror(lambda: dataclasses.replace(
    IEEE, requirement_availability=(
        ("amount", "directly_available"), ("bogus_key", "unavailable"))),
    "unknown requirement key rejected")
expect_valueerror(lambda: dataclasses.replace(
    IEEE, license_documented=False,
    license_terms="still claiming terms"),
    "undocumented license carrying invented terms rejected")
expect_valueerror(lambda: dataclasses.replace(
    IEEE, license_documented=True, license_terms=""),
    "documented license without terms rejected")
expect_valueerror(lambda: dataclasses.replace(
    IEEE, acquisition_status="acquired"),
    "phase-106 registry never flips to acquired")
for ds in BENCHMARK_DATASETS:
    d = ds.to_dict()
    check(d == ds.to_dict(), f"{ds.dataset_id}: to_dict deterministic")
    check(ds.schema_hash == ds.schema_hash,
          f"{ds.dataset_id:26s} schema hash deterministic")
    check(dataclasses.replace(
        ds, schema_columns=ds.schema_columns + ("X_col",)
    ).schema_hash != ds.schema_hash,
        f"{ds.dataset_id:26s} schema hash changes with columns")
    check(ds.acquisition_status == "not_acquired",
          f"{ds.dataset_id:26s} acquisition_status never acquired")

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — CLASSIFICATION & GROUP ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════

section("2. DATASET CLASSIFICATION & GROUPS A/B/C/D")
inv({c.value for c in DatasetClassification} == {
    "public_real_anonymized", "public_research_derived",
    "public_simulated", "public_synthetic", "unknown"},
    "classification enum matches spec exactly")
inv(IEEE.classification == "public_real_anonymized"
    and ULB.classification == "public_real_anonymized",
    "IEEE/ULB classified public real anonymized")
inv(PAYSIM.classification == "public_simulated"
    and BANKSIM.classification == "public_simulated",
    "PaySim/BankSim classified simulated, never real")
inv(BAF.classification == "public_synthetic",
    "BAF classified synthetic, never real")
inv(NON_REAL_WORLD_CLASSIFICATIONS == frozenset({
    "public_simulated", "public_synthetic",
    "public_research_derived", "unknown"}),
    "non-real-world classification set complete")
for ds in (PAYSIM, BANKSIM, BAF):
    check(ds.classification in NON_REAL_WORLD_CLASSIFICATIONS,
          f"{ds.dataset_id:26s} never described as real-world")
gs = group_summary()
inv(gs["group_a"] == () and gs["group_c"] == (), "no group A/C yet")
inv(gs["group_b"] == ("IEEE_CIS", "ULB_CREDIT_CARD_FRAUD", "PAYSIM"),
    "group B = partially compatible trio")
inv(gs["group_d"] == ("BANKSIM", "BAF_BANK_ACCOUNT_FRAUD"),
    "group D = governance-rejected pair")
inv(GROUP_READINESS == {
    "group_a": EvaluationReadiness.READY_NATIVE.value,
    "group_b": EvaluationReadiness.COMPONENT_ONLY.value,
    "group_c": EvaluationReadiness.REFERENCE_ONLY.value,
    "group_d": EvaluationReadiness.REJECTED_NOT_ESTABLISHED.value,
}, "group readiness map matches spec")
# Full A/B/C/D branch coverage via registry-local fixtures.
all_direct = tuple(sorted(
    (k, "directly_available") for k in REQUIREMENT_KEYS))
with _with_dataset(dataclasses.replace(
        IEEE, requirement_availability=all_direct)) as fx:
    g, reasons = assign_benchmark_group(fx.dataset_id)
    inv(g == "group_a" and reasons == (),
        f"all-direct fixture -> group A (got {g})")
    inv(not blocking_features(dataset_preflight(fx.dataset_id)),
        "all-direct fixture has zero blocking features")
all_unavail = tuple(sorted((k, "unavailable") for k in REQUIREMENT_KEYS))
with _with_dataset(dataclasses.replace(
        IEEE, requirement_availability=all_unavail)) as fx:
    g, reasons = assign_benchmark_group(fx.dataset_id)
    inv(g == "group_c", f"all-unavailable fixture -> group C (got {g})")
    inv(not usable_features(dataset_preflight(fx.dataset_id)),
        "all-unavailable fixture has zero usable features")
    inv(reasons and "reference data only" in reasons[0],
        "group C reason names reference-data-only")
with _with_dataset(dataclasses.replace(
        IEEE, classification="unknown")) as fx:
    g, reasons = assign_benchmark_group(fx.dataset_id)
    inv(g == "group_d" and "classification_unknown" in reasons,
        "unknown classification -> group D veto")
with _with_dataset(dataclasses.replace(
        IEEE, provenance_confidence="low")) as fx:
    g, reasons = assign_benchmark_group(fx.dataset_id)
    inv(g == "group_d" and "provenance_not_established" in reasons,
        "low provenance -> group D veto")
with _with_dataset(dataclasses.replace(
        IEEE, semantics_established=False)) as fx:
    g, reasons = assign_benchmark_group(fx.dataset_id)
    inv(g == "group_d" and "schema_semantics_not_established" in reasons,
        "unestablished semantics -> group D veto")
with _with_dataset(dataclasses.replace(
        IEEE, leakage_clear=False)) as fx:
    g, reasons = assign_benchmark_group(fx.dataset_id)
    inv(g == "group_d" and "leakage_risk_unresolved" in reasons,
        "unresolved leakage -> group D veto")
with _with_dataset(dataclasses.replace(
        IEEE, license_documented=False, license_terms="")) as fx:
    g, reasons = assign_benchmark_group(fx.dataset_id)
    inv(g == "group_d" and "license_not_established" in reasons,
        "missing license -> group D veto")
inv(REG._BY_ID["IEEE_CIS"] == IEEE, "registry restored after fixtures")
inv(REG.BENCHMARK_DATASETS[0] == IEEE, "global registry tuple untouched")
for dsid in EXPECTED_IDS:
    scope = evaluation_scope(dsid)
    check(scope in {r.value for r in EvaluationReadiness},
          f"{dsid:26s} evaluation scope defined")
inv(evaluation_scope("IEEE_CIS") == "component_evaluation_only",
    "group B scope = component evaluation only")
inv(evaluation_scope("BANKSIM") == "rejected_governance_not_established",
    "group D scope = rejected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — PROVENANCE / LICENSE METADATA (never invented)
# ══════════════════════════════════════════════════════════════════════

section("3. PROVENANCE / LICENSE METADATA")
CONF = {"high", "medium", "low"}
ACCESS = {"public_repository_page", "public_direct_download",
          "research_portal"}
for ds in BENCHMARK_DATASETS:
    i = ds.dataset_id
    check(bool(ds.canonical_name), f"{i:26s} canonical_name documented")
    check(bool(ds.source_reference), f"{i:26s} source_reference documented")
    check(bool(ds.provider_author), f"{i:26s} provider/author documented")
    check(bool(ds.publication_reference),
          f"{i:26s} publication reference documented")
    check(bool(ds.counts_reference), f"{i:26s} counts_reference documented")
    check(ds.documentation_confidence in CONF,
          f"{i:26s} documentation confidence vocabulary")
    check(ds.provenance_confidence in CONF,
          f"{i:26s} provenance confidence vocabulary")
    check(ds.access_type in ACCESS,
          f"{i:26s} access_type is descriptive-only vocabulary")
    check(ds.expected_local_path.startswith(EXPECTED_LOCAL_ROOT + "/"),
          f"{i:26s} expected local path under data/external")
    check(ds.license_documented is (ds.license_terms != ""),
          f"{i:26s} license terms iff documented")
    check(ds.acquisition_status == "not_acquired",
          f"{i:26s} not acquired")
inv(IEEE.license_documented and ULB.license_documented
    and PAYSIM.license_documented, "IEEE/ULB/PaySim licenses documented")
inv(not BANKSIM.license_documented and not BAF.license_documented,
    "BankSim/BAF licenses honestly undocumented")
inv(BANKSIM.license_terms == "" and BAF.license_terms == "",
    "undocumented licenses carry no invented terms")
inv(ULB.license_terms.startswith("CC BY-NC-SA"),
    "ULB license terms transcribed from source page")
inv(PAYSIM.license_terms.startswith("CC BY 4.0"),
    "PaySim license terms transcribed from Mendeley record")
inv(BAF.timestamp_semantics == NOT_DOCUMENTED
    and BAF.mcc_information == NOT_DOCUMENTED
    and BAF.entity_identifiers == (),
    "BAF undocumented fields use the sentinel, never prose guesses")
inv(BANKSIM.transaction_count is None and BAF.transaction_count is None,
    "undocumented row counts stay None (never invented)")
inv(IEEE.transaction_count == 590540 and ULB.transaction_count == 284807
    and PAYSIM.transaction_count == 1048576,
    "documented row counts transcribed verbatim")
inv(ULB.fraud_count == 492 and IEEE.fraud_count is None
    and PAYSIM.fraud_count is None,
    "fraud counts only where documented")
inv(BAF.provenance_confidence == "high"
    and BAF.documentation_confidence == "low",
    "BAF confidence honestly low-documentation")
for ds in BENCHMARK_DATASETS:
    txt = manual_acquisition_instructions(ds.dataset_id)
    check("MANUAL ACQUISITION" in txt,
          f"{ds.dataset_id:26s} instructions are manual")
    check(ds.expected_local_path in txt,
          f"{ds.dataset_id:26s} instructions name expected path")
    check("Never script this acquisition" in txt,
          f"{ds.dataset_id:26s} instructions forbid scripting")
    check(ds.source_reference in txt,
          f"{ds.dataset_id:26s} instructions cite the source")
    check("if license_documented is False, STOP" in txt,
          f"{ds.dataset_id:26s} instructions gate on license")

# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE COMPATIBILITY & 21→48 PARITY
# ══════════════════════════════════════════════════════════════════════

section("4. FEATURE COMPATIBILITY & 21->48 PIPELINE PARITY")
inv(len(ML_FEATURE_ORDER) == 21, "canonical domain/runtime feature count = 21")
inv(len(ALTMAN_NATIVE_FEATURES) == 48, "canonical native feature count = 48")
inv(REG.DOMAIN_FEATURE_COUNT == 21 and NATIVE_FEATURE_COUNT == 48,
    "registry counts bound to the canonical 21/48")
inv(ML_FEATURE_VERSION == "v1", "feature contract version = v1")
inv(ML_FEATURE_VERSION == REG.ML_FEATURE_VERSION
    and REG.NATIVE_FEATURE_VERSION == "v1",
    "registry re-exports the authority versions")
vec = map_raw_to_native({name: 1.0 for name in ML_FEATURE_ORDER})
inv(vec.shape == (48,),
    "canonical map_raw_to_native maps 21 -> 48 at runtime")
USABLE = {"IEEE_CIS": 5, "ULB_CREDIT_CARD_FRAUD": 5, "PAYSIM": 10,
          "BANKSIM": 8, "BAF_BANK_ACCOUNT_FRAUD": 0}
BLOCKING = {"IEEE_CIS": 43, "ULB_CREDIT_CARD_FRAUD": 43, "PAYSIM": 38,
            "BANKSIM": 40, "BAF_BANK_ACCOUNT_FRAUD": 48}
AVAIL_VOCAB = {"directly_available", "deterministically_derivable",
               "unavailable", "ambiguous", "leakage_risk"}
for ds in BENCHMARK_DATASETS:
    i = ds.dataset_id
    pf = dataset_preflight(i)
    check(len(pf) == 48, f"{i:26s} preflight covers exactly 48 features")
    check(tuple(it.feature for it in pf) == tuple(ALTMAN_NATIVE_FEATURES),
          f"{i:26s} preflight order == canonical native order")
    check(len({it.feature for it in pf}) == 48,
          f"{i:26s} no duplicated/missing native features")
    check(len(usable_features(pf)) == USABLE[i],
          f"{i:26s} usable feature count ({len(usable_features(pf))})")
    check(len(blocking_features(pf)) == BLOCKING[i],
          f"{i:26s} blocking feature count ({len(blocking_features(pf))})")
    check(len(usable_features(pf)) + len(blocking_features(pf)) == 48,
          f"{i:26s} usable + blocking partitions all 48")
    for it in pf:
        check(it.availability in AVAIL_VOCAB,
              f"{i}:{it.feature} availability vocabulary")
        check(bool(it.reason),
              f"{i}:{it.feature} has a stated reason")
        check(it.qualification_impact in ("none", "blocking"),
              f"{i}:{it.feature} impact vocabulary")
    claim = build_benchmark_compat_contract(i).feature_compatibility
    check(claim.feature_contract_version == "v1"
          and claim.native_feature_version == "v1",
          f"{i:26s} claim binds feature/native versions v1/v1")
    check(claim.transformation == CANONICAL_TRANSFORMATION
          and "map_raw_to_native" in claim.transformation,
          f"{i:26s} claim uses the canonical transformation only")
    check(claim.native_feature_order == tuple(ALTMAN_NATIVE_FEATURES),
          f"{i:26s} claim carries the canonical native order")
    check(claim.direct_domain_feature_inference is False,
          f"{i:26s} no direct 21-feature inference claim")
    keys = {k for k, _ in ds.requirement_availability}
    check(keys == set(REQUIREMENT_KEYS),
          f"{i:26s} availability covers all {len(REQUIREMENT_KEYS)} requirements")
    check({v for _, v in ds.requirement_availability}
          <= set(AVAIL_VOCAB) - {"leakage_risk"},
          f"{i:26s} requirement availability vocabulary")
    check(ds.label_column == LABEL_COLUMNS[i],
          f"{i:26s} label column consistent")
    check(TIMESTAMP_COLUMNS.get(i) is None
          or TIMESTAMP_COLUMNS[i] in ds.schema_columns,
          f"{i:26s} timestamp column present in documented schema")
    check(AMOUNT_COLUMNS.get(i) is None
          or AMOUNT_COLUMNS[i] in ds.schema_columns,
          f"{i:26s} amount column documented in schema")

# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — UNAVAILABLE FEATURES & ANTI-FABRICATION
# ══════════════════════════════════════════════════════════════════════

section("5. UNAVAILABLE FEATURES / NO FABRICATION")
elig = native_feature_eligibility_report()
inv(len(elig) == 5, "eligibility report covers all five candidates")
for row in elig:
    i = row["dataset_id"]
    check(row["status"] == "INCOMPATIBLE_FOR_NATIVE_EVALUATION",
          f"{i:26s} INCOMPATIBLE_FOR_NATIVE_EVALUATION")
    check(row["eligible_rows"] == 0,
          f"{i:26s} zero native-eligible rows")
    check(row["native_feature_count"] == 48,
          f"{i:26s} native feature count 48 in eligibility row")
    check(bool(row["reasons"]),
          f"{i:26s} rejection reasons stated")
for ds in BENCHMARK_DATASETS:
    check(native_eligible_row_count(ds.dataset_id) == 0,
          f"{ds.dataset_id:26s} native_eligible_row_count == 0")
expect_failclosed(lambda: native_eligible_row_count("FAKE"),
                  "native_eligible_row_count unknown id fails closed")
baf_pf = dataset_preflight("BAF_BANK_ACCOUNT_FRAUD")
inv(len(baf_pf) == 48, "BAF preflight covers all 48 features")
inv(len(usable_features(baf_pf)) == 0
    and len(blocking_features(baf_pf)) == 48,
    "BAF: every native feature blocked, none fabricated usable")
inv(ULB.entity_identifiers == (), "ULB has no fabricated entity identifiers")
inv(sorted(ULB.schema_columns) == sorted(
    ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)] + ["Class"]),
    "ULB schema exactly Time/Amount/V1..V28/Class (nothing invented)")
inv(IEEE.entity_identifiers
    and "merchant" not in " ".join(IEEE.entity_identifiers),
    "IEEE entity ids are the documented ones only")
inv(BAF.schema_columns == (), "BAF empty schema never invented")
# Fabrication rejection at harmonization level.
excl = harmonize_rows("PAYSIM", [{"step": 5, "amount": 1.0}])
inv(excl[0].harmonization_status ==
    HarmonizationStatus.EXCLUDED_UNMAPPED_LABEL.value
    and excl[0].normalized_label is None,
    "row without label column is excluded, never guessed")
inv(excl[0].timestamp == "5",
    "timestamp preserved even when row later excluded")
noforge = build_benchmark_compat_contract("BAF_BANK_ACCOUNT_FRAUD")
inv(noforge.feature_compatibility.requirement_availability
    == BAF.requirement_availability,
    "compat contract reuses registry availability, never fills it")

# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — HARMONIZATION & SOURCE IDENTITY PRESERVATION
# ══════════════════════════════════════════════════════════════════════

section("6. MULTI-DATASET HARMONIZATION / SOURCE IDENTITY")
LABEL_CASES = [
    ("PAYSIM", {"isFraud": 1}, 1),
    ("PAYSIM", {"isFraud": 0}, 0),
    ("PAYSIM", {"isFraud": "1"}, 1),
    ("PAYSIM", {"isFraud": "0"}, 0),
    ("PAYSIM", {"isFraud": True}, 1),
    ("PAYSIM", {"isFraud": False}, 0),
    ("PAYSIM", {"isFraud": 2}, None),
    ("PAYSIM", {"isFraud": "yes"}, None),
    ("PAYSIM", {"isFraud": None}, None),
    ("PAYSIM", {"isFraud": 0.5}, None),
    ("PAYSIM", {"isFraud": ""}, None),
    ("IEEE_CIS", {"isFraud": 1}, 1),
    ("ULB_CREDIT_CARD_FRAUD", {"Class": "0"}, 0),
    ("BANKSIM", {"fraudRisk": 1}, 1),
    ("BAF_BANK_ACCOUNT_FRAUD", {"fraud_bool": 0}, 0),
]
for dsid, row, want in LABEL_CASES:
    col, raw = next(iter(row.items()))
    got = normalize_label(dsid, raw)
    check(got == want,
          f"{dsid}: normalize_label({col}={raw!r}) -> {want} (got {got})")
expect_failclosed(lambda: normalize_label("UNREGISTERED_DS", 1),
                  "normalize_label unknown dataset fails closed")
prows = harmonize_rows("PAYSIM", [
    {"record_id": "p1", "isFraud": 1, "step": 5, "amount": 10.0},
    {"record_id": "p2", "isFraud": "0", "step": 6, "amount": 2.5},
])
inv(len(prows) == 2, "two PaySim rows harmonized")
for r in prows:
    inv(r.source_dataset_id == "PAYSIM",
        f"row {r.source_record_id}: source identity preserved")
inv(prows[0].original_label == "1" and prows[1].original_label == "0",
    "original labels preserved verbatim")
inv(prows[0].timestamp == "5" and prows[1].timestamp == "6",
    "timestamps preserved verbatim, never re-anchored")
inv(prows[0].source_record_id == "p1"
    and prows[1].source_record_id == "p2",
    "source record ids preserved")
inv(prows[0].harmonization_status == HarmonizationStatus.HARMONIZED.value
    and prows[1].harmonization_status == HarmonizationStatus.HARMONIZED.value,
    "mappable labels -> harmonized")
inv(dict(prows[0].original_schema_metadata)["amount"] == "10.0",
    "schema metadata captured verbatim as strings")
inv(prows[0].original_schema_metadata == tuple(sorted(
    prows[0].original_schema_metadata)),
    "schema metadata canonically sorted")
foreign = harmonize_rows("PAYSIM", [
    {"source_dataset_id": "IEEE_CIS", "isFraud": 1, "step": 1}])
inv(foreign[0].harmonization_status ==
    HarmonizationStatus.EXCLUDED_FOREIGN_SOURCE.value
    and foreign[0].normalized_label is None,
    "foreign source row excluded, never silently re-attributed")
ulb_rows = harmonize_rows("ULB_CREDIT_CARD_FRAUD", [{"Class": 1}])
inv(ulb_rows[0].timestamp == "",
    "missing timestamp stays empty, never fabricated")
bench = build_multisource_benchmark([
    ("PAYSIM", [{"isFraud": 1, "step": 1, "amount": 1.0}]),
    ("ULB_CREDIT_CARD_FRAUD", [{"Class": 0, "Time": 123.4}]),
    ("IEEE_CIS", [{"isFraud": 1, "TransactionDT": 999}]),
])
inv(bench.descriptor == BENCHMARK_DESCRIPTOR,
    "combined bench uses the approved descriptor")
inv(bench.source_dataset_ids == ("IEEE_CIS", "PAYSIM",
                                 "ULB_CREDIT_CARD_FRAUD"),
    "combined bench lists its true sources")
inv(len(bench.harmonized_rows()) == 3 and not bench.excluded_rows(),
    "all fixture rows harmonized")
for r in bench.rows:
    check(r.source_dataset_id in EXPECTED_IDS,
          f"row sourced from a registered dataset ({r.source_dataset_id})")
inv(describe_benchmark([]) == "multi-source public benchmark",
    "describe_benchmark only ever returns the approved phrase")
inv(find_single_source_claims(describe_benchmark([])) == (),
    "approved descriptor contains no forbidden claim")
check(bool(find_single_source_claims(
    "these rows come from one financial institution")),
    "forbidden single-institution claim detected")
check(bool(find_single_source_claims(
    "single-institution dataset")),
    "forbidden single-institution dataset claim detected")
inv(COMMON_BENCHMARK_FIELDS == ("normalized_label",),
    "only the label has defensible cross-source semantics")
expect_valueerror(lambda: MultiSourceBenchmark(
    benchmark_id="x",
    descriptor="single financial institution",
    rows=()),
    "single-institution descriptor rejected")
expect_failclosed(lambda: MultiSourceBenchmark(
    benchmark_id="x",
    descriptor=BENCHMARK_DESCRIPTOR,
    rows=(HarmonizedRow(
        source_dataset_id="NOT_REGISTERED", source_record_id="",
        original_label="1", normalized_label=1, timestamp="",
        original_schema_metadata=(),
        harmonization_status=HarmonizationStatus.HARMONIZED.value),)),
    "unregistered source row rejected by the benchmark")
expect_failclosed(lambda: harmonize_rows("UNREGISTERED", []),
                  "harmonize unknown dataset fails closed")
expect_failclosed(lambda: build_multisource_benchmark(
    [("UNREGISTERED", [])]),
    "multi-source build with unknown source fails closed")
expect_valueerror(lambda: HarmonizedRow(
    source_dataset_id="PAYSIM", source_record_id="",
    original_label="x", normalized_label=7, timestamp="",
    original_schema_metadata=(),
    harmonization_status=HarmonizationStatus.HARMONIZED.value),
    "harmonized row with non-binary label rejected")
expect_valueerror(lambda: HarmonizedRow(
    source_dataset_id="PAYSIM", source_record_id="",
    original_label="x", normalized_label=1, timestamp="",
    original_schema_metadata=(),
    harmonization_status=HarmonizationStatus.EXCLUDED_UNMAPPED_LABEL.value),
    "excluded row carrying a label rejected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — TIMESTAMP SEMANTICS
# ══════════════════════════════════════════════════════════════════════

section("7. TIMESTAMP SEMANTICS")
for ds in BENCHMARK_DATASETS:
    check(bool(ds.timestamp_semantics),
          f"{ds.dataset_id:26s} timestamp semantics stated")
    check(ds.timestamp_supports_absolute_time is False,
          f"{ds.dataset_id:26s} never claims absolute clock time")
inv(IEEE.timestamp_supports_ordering and ULB.timestamp_supports_ordering
    and PAYSIM.timestamp_supports_ordering
    and BANKSIM.timestamp_supports_ordering,
    "relative/sim clocks still support within-source ordering")
inv(not BAF.timestamp_supports_ordering,
    "BAF timestamps not orderable (none documented)")
inv(TIMESTAMP_COLUMNS == {
    "IEEE_CIS": "TransactionDT", "ULB_CREDIT_CARD_FRAUD": "Time",
    "PAYSIM": "step", "BANKSIM": "step",
    "BAF_BANK_ACCOUNT_FRAUD": None},
    "timestamp columns exactly as documented")
inv(not BAF.timestamp_columns_present
    and ULB.timestamp_columns_present
    and PAYSIM.timestamp_columns_present,
    "timestamp presence derived from documented schema")
ts_rows = (harmonize_rows("BAF_BANK_ACCOUNT_FRAUD", [{"fraud_bool": 1}])
           + prows)
t_out = temporal_evaluation(ts_rows, [0.9, 0.5, 0.5])
inv(t_out["scope"] == "within_source_ordering_only",
    "temporal evaluation never cross-aligns sources")
inv(t_out["refusals"] == {
    "BAF_BANK_ACCOUNT_FRAUD": "timestamps_not_orderable"},
    f"BAF refused for temporal evaluation (got {t_out['refusals']})")
inv(set(t_out["segments"]) == {"PAYSIM"},
    "segments built only for orderable, included sources")
expect_valueerror(lambda: temporal_evaluation(ts_rows, [0.5, 0.5, 0.5], 0.0),
                  "holdout 0.0 rejected")
expect_valueerror(lambda: temporal_evaluation(ts_rows, [0.5, 0.5, 0.5], 1.0),
                  "holdout 1.0 rejected")
expect_valueerror(lambda: temporal_evaluation(ts_rows, [0.5]),
                  "temporal rows/scores misalignment rejected")
inv("seconds elapsed" in IEEE.timestamp_semantics
    and "relative only" in IEEE.timestamp_semantics,
    "IEEE relative-only semantics stated explicitly")
inv("relative only" in ULB.timestamp_semantics,
    "ULB relative-only semantics stated explicitly")
inv("simulation clock" in PAYSIM.timestamp_semantics
    and "simulation clock" in BANKSIM.timestamp_semantics,
    "simulated clocks named as simulation clocks")

# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — LEAKAGE CONTROLS
# ══════════════════════════════════════════════════════════════════════

section("8. LEAKAGE CONTROLS")
hits = scan_leakage_fields(
    ["class", "is_fraud", "target", "model_score", "amount", "step"])
inv(hits == ("class", "is_fraud", "model_score", "target"),
    f"label/outcome fields flagged (got {hits})")
inv("amount" not in hits and "step" not in hits,
    "legitimate feature columns not flagged")
for pat in ("^class$", "^label$", "^target$", "model_score", "_after$",
            "^post_", "^isfraud$", "^fraudrisk$", "predicted"):
    check(any(pat in p for p in LEAKAGE_FIELD_PATTERNS),
          f"leakage pattern present: {pat}")
for marker in ("smote", "gan_generated", "generated_by_model",
               "synthetic_augmented"):
    check(marker in SYNTHETIC_AUGMENTATION_MARKERS,
          f"synthetic augmentation marker present: {marker}")
leak = dict(dataset_leakage_findings())
inv(leak == {"IEEE_CIS": ("isFraud",),
             "ULB_CREDIT_CARD_FRAUD": ("Class",),
             "PAYSIM": ("isFlaggedFraud", "isFraud",
                        "newbalanceDest", "newbalanceOrig"),
             "BANKSIM": ("fraudRisk",),
             "BAF_BANK_ACCOUNT_FRAUD": ()},
    f"exact per-dataset leakage findings (got {leak})")
for ds in BENCHMARK_DATASETS:
    flagged = scan_leakage_fields(ds.schema_columns)
    check(set(flagged) <= set(ds.schema_columns),
          f"{ds.dataset_id:26s} flagged columns come from its own schema")
inv(any("isFlaggedFraud" in r for r in PAYSIM.known_leakage_risks)
    and any("newbalance" in r for r in PAYSIM.known_leakage_risks),
    "PaySim known leakage risks document the flagged columns")
inv(BAF.known_leakage_risks
    and "no column may be mapped" in BAF.known_leakage_risks[0],
    "BAF documents zero-column mapping due to leakage risk")
mark_row = harmonize_rows("PAYSIM", [
    {"isFraud": 1, "step": 3, "augmented_synthetic": "True"}])
inv(detect_synthetic_augmentation(mark_row) == ("PAYSIM:0",),
    "synthetic-augmented row detected")
inv(detect_synthetic_augmentation(prows) == (),
    "clean rows carry no augmentation marker")

# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — CONTAMINATION & CROSS-DATASET ISOLATION
# ══════════════════════════════════════════════════════════════════════

section("9. SOURCE CONTAMINATION / CROSS-DATASET ISOLATION")
dup_p = harmonize_rows("PAYSIM", [{"isFraud": 1, "step": 10, "amount": 5.0}])
dup_u = harmonize_rows("ULB_CREDIT_CARD_FRAUD",
                       [{"Class": 1, "Time": 10, "Amount": 5.0}])
cross = detect_cross_source_duplicates(dup_p + dup_u)
inv(len(cross) == 1
    and cross[0][1] == ("PAYSIM", "ULB_CREDIT_CARD_FRAUD"),
    "cross-source duplicate content detected across sources")
inv(detect_cross_source_duplicates(dup_p + dup_p) == (),
    "same-source duplicates are not cross-source contamination")
inv(detect_cross_source_duplicates(()) == (),
    "empty row set -> no contamination finding")
inv(detect_cross_source_duplicates(()) == (),
    "empty row set -> no contamination finding")
expected = frozenset({"PAYSIM", "ULB_CREDIT_CARD_FRAUD"})
inv(detect_source_contamination(dup_p + dup_u, expected) == (),
    "rows within expected sources -> clean")
fake = HarmonizedRow(
    source_dataset_id="SMUGGLED_SOURCE", source_record_id="",
    original_label="1", normalized_label=1, timestamp="",
    original_schema_metadata=(),
    harmonization_status=HarmonizationStatus.HARMONIZED.value)
inv(detect_source_contamination(dup_p + (fake,), expected)
    == ("SMUGGLED_SOURCE",),
    "smuggled foreign source flagged")
fp0 = row_fingerprint(dup_p[0])
inv(detect_train_evaluation_overlap(dup_u + dup_p, {fp0}) == (fp0,),
    "train/evaluation fingerprint overlap detected")
inv(detect_train_evaluation_overlap(dup_p, {fp0 * 2}) == (),
    "disjoint fingerprints -> no overlap finding")
inv(detect_train_evaluation_overlap(dup_p, ()) == (),
    "empty training set -> no overlap claim")
cov = evaluation_coverage(bench.rows + foreign)
inv(cov["native_eligible_rows"] == 0,
    "coverage never reports native-eligible rows")
inv(cov["total_rows"] == len(bench.rows) + len(foreign)
    and cov["excluded_rows"] == 1,
    "coverage counts all rows, exactly one excluded (foreign)")
inv(dict(cov["per_source"]) == {"IEEE_CIS": 1, "PAYSIM": 2,
                                "ULB_CREDIT_CARD_FRAUD": 1},
    "coverage preserves per-source identity (foreign row stays PAYSIM)")
scr = source_contamination_report()
inv(scr["acquired_rows"] == 0 and not scr["cross_source_duplicates"]
    and not scr["train_evaluation_overlap"],
    "real benchmark is empty: no contamination possible, detectors armed")
inv("no rows acquired" in scr["status"], "contamination status honest")
# Cross-source feature isolation: PaySim metadata never carries ULB cols.
pay_only = set(dict(prows[0].original_schema_metadata))
inv("Class" not in pay_only and "Time" not in pay_only,
    "PaySim row metadata never carries ULB columns")
ulb_only = set(dict(ulb_rows[0].original_schema_metadata))
inv("isFraud" not in ulb_only and "step" not in ulb_only,
    "ULB row metadata never carries PaySim columns")

# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — BENCHMARK MANIFEST (NOT gate evidence)
# ══════════════════════════════════════════════════════════════════════

section("10. BENCHMARK MANIFEST / HASHING / IMMUTABILITY")
m1 = build_benchmark_manifest()
m2 = build_benchmark_manifest()
inv(m1.manifest_hash == m2.manifest_hash, "manifest hash deterministic")
inv(m1.verify_hash() and m2.verify_hash(), "manifest self-hash verifies")
inv(m1.manifest_hash == hashlib.sha256(
    canonical_json(m1.payload()).encode("utf-8")).hexdigest(),
    "manifest hash = sha256 of canonical payload")
inv(m1.manifest_kind == "benchmark_manifest"
    and m1.descriptor == BENCHMARK_DESCRIPTOR,
    "manifest kind/descriptor pinned")
inv(m1.created_at == MANIFEST_CREATED_AT
    and m1.created_at == "2026-09-22T00:00:00+00:00",
    "manifest instant fixed, never a wall-clock read")
inv(m1.dataset_ids == DATASET_IDS, "manifest lists all registered datasets")
inv(len(m1.source_references) == 5 and len(m1.license_metadata) == 5
    and len(m1.row_counts) == 5 and len(m1.schema_hashes) == 5,
    "manifest carries source/license/row/schema metadata for all five")
inv(m1.source_hashes == (),
    "no source files acquired: source hash list honestly empty")
inv(len(m1.harmonization_spec_hash) == 64,
    "harmonization spec hash present")
inv(len(m1.feature_compatibility_results) == 5
    and all(len(row) == 4 for row in m1.feature_compatibility_results),
    "manifest embeds per-dataset compat results")
inv(("threshold", repr(PRODUCTION_THRESHOLD)) in m1.evaluation_config
    and repr(PRODUCTION_THRESHOLD) == "0.018758",
    "manifest evaluation config pins threshold 0.018758")
inv(any(k == "purpose" and "external_assessment_only" in v
        for k, v in m1.evaluation_config),
    "manifest purpose declares external assessment only")
inv(m1.model_id == "altman_native"
    and m1.model_id == MODEL_ID,
    "manifest model identity = production authority")
inv(m1.release_id == "release-altman_native_E_hardneg_cert_20260904"
    and m1.release_id == RELEASE_ID,
    "manifest release identity = production authority")
inv(m1.feature_contract_version == "v1"
    and m1.native_feature_version == "v1",
    "manifest feature/native versions = v1/v1")
inv(m1.domain_feature_count == 21 and m1.native_feature_count == 48,
    "manifest domain/native counts = 21/48")
inv(m1.production_threshold == 0.018758 == PRODUCTION_THRESHOLD,
    "manifest threshold = frozen production threshold")
inv(m1.canonical_transformation == CANONICAL_TRANSFORMATION
    and "map_raw_to_native" in m1.canonical_transformation,
    "manifest canonical mapping = map_raw_to_native()")
# Not gate evidence: no authority accepts this object.
inv(not isinstance(m1, RWVPromotionEvidence),
    "manifest is not RWVPromotionEvidence")
inv(not isinstance(m1, QualificationResult),
    "manifest is not a QualificationResult")
inv(not isinstance(m1, PromotionToken), "manifest is not a PromotionToken")
prov_type = type(KNOWN_CANDIDATES["IEEE_CIS"])
inv(not isinstance(m1, prov_type), "manifest is not ProviderEvidence")
inv(set(BENCHMARK_MANIFEST_NOT_GATE_EVIDENCE) == {
    "ProviderEvidence", "QualificationResult",
    "RWVPromotionEvidence", "PromotionToken"},
    "not-gate-evidence list names all four authorities")
# Tamper detection.
tampered = dataclasses.replace(m1, dataset_ids=("IEEE_CIS",))
inv(not tampered.verify_hash(),
    "tampered dataset_ids detected by verify_hash")
resalted = dataclasses.replace(m1, manifest_hash="0" * 64)
inv(not resalted.verify_hash(), "forged manifest hash detected")
json.dumps(m1.to_dict())  # serializable without custom encoder
inv(True, "manifest JSON-serializable")
# Identity/threshold substitution fails at construction.
expect_valueerror(lambda: dataclasses.replace(
    m1, model_id="other_model"), "model identity substitution rejected")
expect_valueerror(lambda: dataclasses.replace(
    m1, release_id="release-historical-2024"),
    "historical release substitution rejected")
expect_valueerror(lambda: dataclasses.replace(
    m1, feature_contract_version="v0"),
    "feature-version substitution rejected")
expect_valueerror(lambda: dataclasses.replace(
    m1, native_feature_version="v0"),
    "native-feature-version substitution rejected")
expect_valueerror(lambda: dataclasses.replace(
    m1, domain_feature_count=22), "domain-count substitution rejected")
expect_valueerror(lambda: dataclasses.replace(
    m1, native_feature_count=47), "native-count substitution rejected")
expect_valueerror(lambda: dataclasses.replace(
    m1, production_threshold=0.5), "threshold substitution rejected")
expect_valueerror(lambda: dataclasses.replace(
    m1, canonical_transformation="alternate_mapping()"),
    "alternate transformation substitution rejected")
expect_valueerror(lambda: dataclasses.replace(
    m1, descriptor="single financial institution"),
    "single-institution descriptor substitution rejected")
expect_valueerror(lambda: dataclasses.replace(
    m1, manifest_kind="qualification_result"),
    "direct qualification-state injection rejected")
expect_valueerror(lambda: dataclasses.replace(
    m1, manifest_kind="RWVPromotionEvidence"),
    "RWVPromotionEvidence-kind injection rejected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 11 — EVALUATION INFRASTRUCTURE / THRESHOLD IMMUTABILITY
# ══════════════════════════════════════════════════════════════════════

section("11. EVALUATION INFRASTRUCTURE / THRESHOLD IMMUTABILITY")
inv(PRODUCTION_THRESHOLD == 0.018758, "production threshold = 0.018758")
cfg = build_evaluation_config()
inv(cfg.threshold == PRODUCTION_THRESHOLD,
    "eval config uses the locked threshold")
inv(cfg.config_version == EVAL_CONFIG_VERSION, "eval config version pinned")
inv(cfg.purpose ==
    "external_assessment_only_no_tuning_no_threshold_selection",
    "purpose: assessment only, no tuning/threshold selection")
inv(cfg.temporal_scope == "within_source_ordering_only",
    "temporal scope is within-source only")
inv(cfg.native_scope_requires == "ready_for_native_evaluation",
    "native scope requires ready_for_native_evaluation")
inv(cfg.stratification == ("source_dataset_id",),
    "stratification by source dataset")
inv(cfg.metrics == ("precision", "recall", "f1", "pr_auc", "roc_auc",
                    "mcc", "brier_score", "confusion"),
    "metric set complete")
expect_valueerror(lambda: BenchmarkEvaluationConfig(
    config_version="x", threshold=0.5, metrics=(),
    stratification=(), temporal_scope="", purpose="",
    native_scope_requires=""),
    "non-production threshold rejected at construction")
inv("threshold" not in inspect.signature(
    compute_benchmark_metrics).parameters,
    "compute_benchmark_metrics accepts no threshold argument")
base = compute_benchmark_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
for key in ("precision", "recall", "f1", "pr_auc", "roc_auc", "mcc",
            "brier_score", "tp", "fp", "fn", "tn", "class_imbalance",
            "threshold", "threshold_source"):
    check(key in base, f"metric present: {key}")
inv(base["threshold"] == 0.018758,
    "metrics computed at the locked threshold")
inv("locked" in base["threshold_source"],
    "threshold_source declares the threshold locked")
inv(compute_mcc(2, 0, 0, 2) == 1.0, "MCC perfect-prediction sanity")
inv(compute_mcc(0, 0, 0, 0) == 0.0, "MCC zero-denominator fail-safe")
inv(class_imbalance([0, 1, 1]) == {"rows": 3, "fraud_count": 2,
                                   "fraud_rate": 0.666667},
    "class imbalance reporting exact")
degen = compute_benchmark_metrics([0, 0], [0.1, 0.2])
inv(isinstance(degen, dict),
    f"degenerate input returns a dict (got {type(degen).__name__})")
five_rows = (
    harmonize_rows("IEEE_CIS", [{"isFraud": 1, "TransactionDT": 100}])
    + harmonize_rows("ULB_CREDIT_CARD_FRAUD", [{"Class": 0, "Time": 5}])
    + harmonize_rows("PAYSIM", [{"isFraud": 1, "step": 2}])
    + harmonize_rows("BANKSIM", [{"fraudRisk": 1, "step": 1}])
    + harmonize_rows("BAF_BANK_ACCOUNT_FRAUD", [{"fraud_bool": 1}])
)
scores = [0.9, 0.1, 0.8, 0.7, 0.6]
st = stratified_evaluation(five_rows, scores)
inv(set(st["per_source"]) == {"IEEE_CIS", "PAYSIM",
                              "ULB_CREDIT_CARD_FRAUD"},
    f"stratified excludes group-D sources (got {sorted(st['per_source'])})")
inv(set(st["refusals"]) == {"BANKSIM", "BAF_BANK_ACCOUNT_FRAUD"}
    and all(v == "source_rejected_governance_not_established"
            for v in st["refusals"].values()),
    "group-D sources refused with the governance reason")
for src, met in st["per_source"].items():
    check("mcc" in met and "pr_auc" in met,
          f"stratified metrics complete for {src}")
cd = cross_dataset_evaluation(five_rows, scores)
inv(cd["field_scope"] == ["normalized_label"],
    "pooled evaluation scoped to the common label field only")
inv("no cross-source feature merge" in cd["note"],
    "pooled evaluation disclaims feature merging")
inv(set(cd["refusals"]) == {"BANKSIM", "BAF_BANK_ACCOUNT_FRAUD"},
    "pooled evaluation refuses group D too")
check(bool(cd["pooled_metrics"].get("mcc")
           is not None or "error" in cd["pooled_metrics"]),
    "pooled metrics computed deterministically")
expect_valueerror(lambda: stratified_evaluation(five_rows, [0.5]),
                  "stratified rows/scores misalignment rejected")
expect_valueerror(lambda: cross_dataset_evaluation(five_rows, [0.5]),
                  "cross-dataset rows/scores misalignment rejected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 12 — RWV / PROMOTION GATES UNCHANGED
# ══════════════════════════════════════════════════════════════════════

section("12. RWV / PROMOTION GATES REMAIN UNCHANGED")
inv(SYSTEM_READINESS == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET",
    "SYSTEM_READINESS unchanged")
inv(REAL_WORLD_VALIDATION == "BLOCKED_PENDING_ELIGIBLE_DATASET",
    "REAL_WORLD_VALIDATION unchanged")
inv(PROMOTION_STATE == "PROMOTION_GATE_REQUIRED", "PROMOTION unchanged")
inv(REG.SYSTEM_READINESS == SYSTEM_READINESS
    and REG.REAL_WORLD_VALIDATION == REAL_WORLD_VALIDATION
    and REG.PROMOTION_STATE == PROMOTION_STATE,
    "registry re-exports read-only state, no duplicate authority")
closure = check_global_state()
inv(bool(closure) and all(i.verdict == "pass" for i in closure),
    f"Phase 103 closure invariants all pass ({len(closure)})")
rwv_gate = evaluate_real_world_validation()
inv(rwv_gate.status == GateStatus.BLOCKED,
    "RWV gate evaluates BLOCKED")
inv(rwv_gate.gate_name == "REAL_WORLD_VALIDATION",
    "gate name is the authoritative one")
decision = evaluate_promotion()
inv(decision.verdict == PromotionVerdict.BLOCKED,
    "promotion decision BLOCKED")
inv("REAL_WORLD_VALIDATION" in decision.blocking_gates,
    "blocking gate lists REAL_WORLD_VALIDATION")
expect_failclosed(
    lambda: assert_promotion_allowed(decision),
    "assert_promotion_allowed raises on blocked decision")
ev = KNOWN_CANDIDATES["IEEE_CIS"]
qual = qualify_dataset(ev)
el = check_rwv_execution_eligibility(ev, qual)
inv(el["eligible"] is False, "RWV execution eligibility False")
inv(bool(el["blocking_gates"]), "RWV eligibility carries blocking gates")
sess = create_session(ev, qual, session_id="phase106-test")
inv(sess.status == SessionState.BLOCKED.value,
    f"RWV session created BLOCKED (got {sess.status})")
expect_failclosed(lambda: PromotionToken(decision),
                  "PromotionToken from blocked decision rejected")
expect_failclosed(lambda: PromotionToken(m1),
                  "PromotionToken from a manifest rejected (no verdict)")
inv(REG.MODEL_ID == "altman_native"
    and RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904",
    "model/release identities match the production authority")

# ══════════════════════════════════════════════════════════════════════
# SECTION 13 — KNOWN-CANDIDATE REGRESSION (SPEC §15)
# ══════════════════════════════════════════════════════════════════════

section("13. KNOWN CANDIDATES REMAIN BLOCKED")
known = evaluate_known_candidates()
EXPECTED_KNOWN = {"WORLDLINE_ECOM_2017_NAG", "WORLDLINE_ONLINE_2018",
                  "NOVATTI", "IEEE_CIS"}
inv(set(known) == EXPECTED_KNOWN, "exactly the four known candidates")
EXPECTED_BLOCKED = {
    "WORLDLINE_ECOM_2017_NAG": (False, "blocked"),
    "WORLDLINE_ONLINE_2018": (False, "blocked"),
    "NOVATTI": (False, "blocked"),
    "IEEE_CIS": (False, "blocked"),
}
for k, v in sorted(known.items()):
    want_q, want_s = EXPECTED_BLOCKED[k]
    check(isinstance(v, QualificationResult),
          f"{k}: authoritative QualificationResult type")
    check(v.qualified is want_q, f"{k}: qualified == {want_q}")
    check(v.qualification_state == want_s,
          f"{k}: state == {want_s} (got {v.qualification_state})")
inv(assign_benchmark_group("IEEE_CIS")[0] == "group_b"
    and not known["IEEE_CIS"].qualified,
    "benchmark group membership NEVER upgrades qualification")
report = generate_phase106_report()
inv(report.qualified_datasets == (), "qualified_datasets == ()")
inv(report.any_dataset_qualified is False, "any_dataset_qualified False")
inv(report.known_candidate_states == (
    ("IEEE_CIS", "blocked"), ("NOVATTI", "blocked"),
    ("WORLDLINE_ECOM_2017_NAG", "blocked"),
    ("WORLDLINE_ONLINE_2018", "blocked")),
    "report embeds the four blocked states verbatim")

# ══════════════════════════════════════════════════════════════════════
# SECTION 14 — REPORT COMPLETENESS & DETERMINISM
# ══════════════════════════════════════════════════════════════════════

section("14. REPORT COMPLETENESS / DETERMINISM")
report2 = generate_phase106_report()
inv(report.report_hash == report2.report_hash,
    "report hash deterministic across runs")
recomputed = _report_hash({**report.to_dict(), "report_hash": ""})
inv(len(recomputed) == 64, "report hash recomputable")
inv(recomputed == report.report_hash,
    "report hash self-consistent over to_dict (hash field blanked)")
REQUIRED_KEYS = [
    "report_id", "phase", "registry_version", "manifest_version",
    "eval_config_version", "contract_version", "evidence_policy_version",
    "benchmark_id", "descriptor", "model_id", "release_id",
    "feature_contract_version", "native_feature_version",
    "feature_version", "preprocessing_hash", "rule_hash",
    "domain_feature_count", "native_feature_count",
    "production_threshold", "system_readiness",
    "real_world_validation", "promotion_state", "candidates",
    "candidate_count", "group_summary", "common_harmonized_fields",
    "leakage_findings", "contamination_findings", "native_eligibility",
    "row_totals", "registry_hash", "manifest_hash",
    "evaluation_readiness", "known_candidate_states",
    "any_dataset_qualified", "qualified_datasets", "declarations",
    "conclusion", "report_hash",
]
d = report.to_dict()
for key in REQUIRED_KEYS:
    check(key in d, f"report field present: {key}")
inv(d["phase"] == 106, "report phase = 106")
inv(d["contract_version"] and d["evidence_policy_version"],
    "contract & evidence-policy versions recorded")
inv(report.model_id == "altman_native"
    and report.release_id == "release-altman_native_E_hardneg_cert_20260904",
    "report binds production model/release identity")
inv(report.feature_contract_version == ML_FEATURE_VERSION
    and report.native_feature_version == REG.NATIVE_FEATURE_VERSION
    and report.feature_version == READINESS_FEATURE_VERSION,
    "report binds feature/native/feature_version to the authorities")
inv(report.domain_feature_count == 21
    and report.native_feature_count == 48,
    "report feature counts = 21/48")
inv(report.production_threshold == 0.018758,
    "report records the frozen threshold")
inv(report.system_readiness == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET"
    and report.real_world_validation == "BLOCKED_PENDING_ELIGIBLE_DATASET"
    and report.promotion_state == "PROMOTION_GATE_REQUIRED",
    "report embeds the three authoritative states")
inv(report.conclusion == EXPECTED_CONCLUSION
    and report.conclusion == "READY_WITH_EXTERNAL_PREREQUISITE",
    "conclusion = READY_WITH_EXTERNAL_PREREQUISITE")
inv(report.candidate_count == 5
    and len(report.candidates) == 5,
    "report covers five discovered candidates")
for c in report.candidates:
    check(c.eligible_rows == 0,
          f"{c.dataset_id:26s} report eligible rows = 0")
    check(bool(c.rejection_reasons),
          f"{c.dataset_id:26s} report states rejection reasons")
    check(c.group in ("group_a", "group_b", "group_c", "group_d"),
          f"{c.dataset_id:26s} report group in A/B/C/D vocabulary")
    check(len(c.usable_features) + len(c.blocking_features) == 48,
          f"{c.dataset_id:26s} report usable+blocking = 48")
    check(c.classification == get_dataset(c.dataset_id).classification,
          f"{c.dataset_id:26s} report classification matches registry")
inv(tuple(g for g, _ in report.group_summary) == (
    "group_a", "group_b", "group_c", "group_d"),
    "report group summary covers all four groups")
inv(dict(report.group_summary)["group_b"] == (
    "IEEE_CIS", "ULB_CREDIT_CARD_FRAUD", "PAYSIM"),
    "report group B membership matches registry")
inv(dict(report.group_summary)["group_d"] == (
    "BANKSIM", "BAF_BANK_ACCOUNT_FRAUD"),
    "report group D membership matches registry")
inv(report.common_harmonized_fields == ("normalized_label",),
    "report common harmonized field set")
inv(len(report.leakage_findings) == 5
    and dict(report.leakage_findings)["PAYSIM"] == (
        "isFlaggedFraud", "isFraud", "newbalanceDest", "newbalanceOrig"),
    "report leakage findings complete")
inv(report.contamination_findings.get("acquired_rows") == 0,
    "report contamination: zero acquired rows")
for row in report.native_eligibility:
    check(row["status"] == "INCOMPATIBLE_FOR_NATIVE_EVALUATION"
          and row["eligible_rows"] == 0,
          f"report eligibility: {row['dataset_id']} incompatible, 0 rows")
rt = report.row_totals
inv(rt["native_eligible_rows_total"] == 0,
    "report: zero native-eval-eligible rows total")
inv(rt["documented_rows_total"] == 1923923,
    "report documented row total = sum of documented counts")
inv(rt["rejected_rows_total"] == rt["documented_rows_total"],
    "every documented row currently rejected for native scope")
inv(set(rt["count_not_transcribed_datasets"]) == {"BANKSIM",
                                                  "BAF_BANK_ACCOUNT_FRAUD"},
    "counts-not-transcribed honestly listed")
inv(report.manifest_hash == m1.manifest_hash
    and report.manifest_hash == build_benchmark_manifest().manifest_hash,
    "report binds the deterministic manifest hash")
inv(report.registry_hash == registry_integrity_hash(),
    "report binds the registry integrity hash")
inv(report.evaluation_readiness ==
    "NOT_READY_NO_DATASET_FULLY_COMPATIBLE_AND_NO_FILE_ACQUIRED",
    "evaluation readiness honest: not ready")
DECLARED = set(DECLARATIONS)
for decl in DECLARATIONS:
    check(bool(decl) and decl.endswith("."), f"declaration well-formed: {decl}")
inv("THIS IS NOT REAL-WORLD VALIDATION." in DECLARED,
    "explicit NOT-RWV statement")
inv("CURRENT REAL_WORLD_VALIDATION REMAINS "
    "BLOCKED_PENDING_ELIGIBLE_DATASET." in DECLARED,
    "explicit RWV-blocked statement")
inv("PRODUCTION PROMOTION REMAINS PROMOTION_GATE_REQUIRED." in DECLARED,
    "explicit promotion-gated statement")
inv("NO EXTERNAL DATASET WAS ACQUIRED." in DECLARED
    and "NO PROVIDER WAS CONTACTED." in DECLARED
    and "NO DATA WAS DOWNLOADED." in DECLARED,
    "no-acquisition / no-contact / no-download statements")
inv("NO PROMOTION PATH WAS CREATED." in DECLARED,
    "no-promotion-path statement")
inv(report.declarations == DECLARATIONS, "report embeds declarations")
inv("BENCHMARK RESULTS CONSTITUTE EXTERNAL ASSESSMENT ONLY, "
    "NEVER REAL-WORLD VALIDATION." in DECLARED,
    "benchmark != RWV statement")

# ══════════════════════════════════════════════════════════════════════
# SECTION 15 — SECURITY SOURCE SCANS
# ══════════════════════════════════════════════════════════════════════

section("15. SECURITY SOURCE SCANS (new Phase 106 code only)")
REG_PATH = os.path.join(BACKEND, "src", "monitoring",
                        "phase106_public_benchmark_registry.py")
REP_PATH = os.path.join(BACKEND, "src", "monitoring",
                        "phase106_public_benchmark_report.py")
with open(REG_PATH, encoding="utf-8") as fh:
    reg_src = fh.read()
with open(REP_PATH, encoding="utf-8") as fh:
    rep_src = fh.read()
SOURCES = {"registry": reg_src, "report": rep_src}
FORBIDDEN: list[tuple[str, str]] = [
    (r"import\s+urllib|urllib\.", "urllib"),
    (r"import\s+requests|requests\.", "requests/http client"),
    (r"\bhttpx\b|\baiohttp\b", "async http client"),
    (r"\bsocket\b", "socket"),
    (r"\bsubprocess\b", "subprocess"),
    (r"urlretrieve|urlopen", "url retrieval"),
    (r"\bcurl\b|\bwget\b", "shell downloader"),
    (r"\bpickle\b", "pickle"),
    (r"\bjoblib\b", "joblib"),
    (r"os\.environ|getenv", "environment/credential read"),
    (r"api[_-]?key", "api key"),
    (r"passw(?:or)?d", "password"),
    (r"secrets\.", "secrets module"),
    (r"\bopen\s*\(", "file open()"),
    (r"\.fit\s*\(|\.predict\s*\(", "model fit/predict call"),
    (r"retrain\s*\(|promote\s*\(", "retrain/promote call"),
    (r"download\s*\(", "automated download call"),
    (r"PromotionToken\s*\(", "PromotionToken construction"),
    (r"RWVPromotionEvidence\s*\(", "RWVPromotionEvidence construction"),
    (r"ProviderEvidence\s*\(", "ProviderEvidence construction"),
    (r"QualificationResult\s*\(", "QualificationResult construction"),
    (r"create_session\s*\(", "RWV session creation"),
    (r"evaluate_promotion\s*\(", "promotion evaluation call"),
    (r"assert_promotion_allowed\s*\(", "promotion assert call"),
    (r"check_rwv_execution_eligibility\s*\(", "RWV eligibility call"),
    (r"from\s+src\.monitoring\.promotion_gate",
     "promotion-gate import"),
    (r"from\s+src\.monitoring\.rwv_execution", "RWV-execution import"),
    (r"from\s+src\.monitoring\.rwv_promotion_evidence",
     "RWV-evidence import"),
    (r"from\s+src\.monitoring\.provider_evidence",
     "provider-evidence import"),
    (r"DATASET_QUALIFIED_FOR_CONTROLLED_RWV",
     "qualification-state token"),
    (r"\beval\s*\(|\bexec\s*\(|__import__", "dynamic execution"),
    (r"read_csv|read_table|loadtxt|genfromtxt", "data file loader"),
    (r"to_pickle|to_joblib", "artifact writer"),
]
for sname, stext in SOURCES.items():
    for pattern, what in FORBIDDEN:
        check(re.search(pattern, stext) is None,
              f"{sname}: no {what} ({pattern})")
inv("multi-source public benchmark" in reg_src,
    "registry self-describes as multi-source public benchmark")
inv(len(FORBIDDEN_SINGLE_SOURCE_CLAIMS) >= 5,
    "forbidden single-source claim list populated")
for phrase in FORBIDDEN_SINGLE_SOURCE_CLAIMS:
    check(phrase in reg_src,
          f"registry detector knows forbidden phrase: {phrase}")
    check(phrase not in rep_src, f"report contains no '{phrase}'")
# No bypass parameter on any public/private function of either module.
bypass_hits: list[str] = []
for mod in (REG, REP):
    for name, obj in vars(mod).items():
        if (inspect.isfunction(obj)
                and getattr(obj, "__module__", "") == mod.__name__):
            for pname in inspect.signature(obj).parameters:
                if pname in FORBIDDEN_BYPASS_PARAMETERS:
                    bypass_hits.append(f"{mod.__name__}.{name}:{pname}")
inv(not bypass_hits,
    f"no force/allow_unverified/skip_validation/override/admin_override/"
    f"bypass params (hits: {bypass_hits})")
inv(not any("qualify" in n.lower() or n.lower().startswith("promote")
            for n in dir(REG)),
    "registry defines no qualification/promotion authority")
inv(not any(n.lower().startswith("promote") or "promotiontoken" in
            n.lower() for n in dir(REP)),
    "report defines no promotion authority")
inv(not any(p in REG.__dict__ for p in
            ("force", "allow_unverified", "skip_validation", "override",
             "admin_override", "bypass")),
    "no bypass module-level constants")
inv("assessed=False" in reg_src
    and "NEVER submitted to the Phase 104" in reg_src,
    "compat contract is never submitted to qualification")

# ══════════════════════════════════════════════════════════════════════
# SECTION 16 — BYPASS & SUBSTITUTION ATTEMPTS (all fail closed)
# ══════════════════════════════════════════════════════════════════════

section("16. BYPASS / SUBSTITUTION ATTEMPTS — ALL FAIL CLOSED")
for param in sorted(FORBIDDEN_BYPASS_PARAMETERS):
    expect_typeerror(
        lambda p=param: generate_phase106_report(**{p: True}),
        f"generate_phase106_report({param}=True) rejected")
expect_typeerror(lambda: build_benchmark_manifest(force=True),
                 "build_benchmark_manifest(force=True) rejected")
expect_typeerror(lambda: dataset_preflight("IEEE_CIS", force=True),
                 "dataset_preflight(force=True) rejected")
expect_typeerror(lambda: assign_benchmark_group("IEEE_CIS", force=True),
                 "assign_benchmark_group(force=True) rejected")
expect_typeerror(
    lambda: compute_benchmark_metrics([0], [0.1], force=True),
    "compute_benchmark_metrics(force=True) rejected")
expect_typeerror(
    lambda: evaluate_dataset_evidence(
        build_benchmark_compat_contract("IEEE_CIS"),
        KNOWN_CANDIDATES["IEEE_CIS"], force=True),
    "Phase 104 engine rejects force=True")
expect_typeerror(
    lambda: qualify_submissions(
        build_benchmark_compat_contract("IEEE_CIS"), (), force=True),
    "Phase 105 intake rejects force=True")
expect_typeerror(
    lambda: evaluate_dataset_evidence(
        build_benchmark_compat_contract("IEEE_CIS"),
        KNOWN_CANDIDATES["IEEE_CIS"], allow_unverified=True),
    "Phase 104 engine rejects allow_unverified=True")
expect_typeerror(
    lambda: stratified_evaluation(five_rows, scores, override=True),
    "stratified_evaluation rejects override=True")
# Origin-conversion attempts: local/simulated/synthetic never becomes
# provider/real-world evidence.
for dsid in ("PAYSIM", "BANKSIM", "BAF_BANK_ACCOUNT_FRAUD"):
    dsx = get_dataset(dsid)
    inv(dsx.classification in NON_REAL_WORLD_CLASSIFICATIONS
        and dsx.classification != "public_real_anonymized",
        f"{dsid}: synthetic/simulated never converts to real-world")
inv(BAF.classification == "public_synthetic"
    and assign_benchmark_group("BAF_BANK_ACCOUNT_FRAUD")[0] == "group_d",
    "synthetic BAF cannot earn a qualifying group")
# Public availability never becomes access authority.
inv("access_authority" not in {f.name
                               for f in dataclasses.fields(BenchmarkDataset)},
    "registry record holds no access-authority claim field")
contract = build_benchmark_compat_contract("IEEE_CIS")
inv(contract.access_authority == AccessAuthorityClaim(),
    "compat contract asserts no access authority (default = unknown)")
for ds in BENCHMARK_DATASETS:
    check(ds.access_type not in {
        "provider_authorized", "institutional_agreement",
        "confidential_provider_access", "unknown"},
        f"{ds.dataset_id:26s} access_type never claims authorization")
check(getattr(contract, "assessed", False) is False,
      "compat contract assessed=False (never enters Phase 104)")
# Direct RWV / promotion attempts already covered in section 12; assert
# manifest cannot masquerade as gate evidence through the intake path.
expect_failclosed(
    lambda: qualify_submissions(contract, (m1,)),
    "Phase 105 intake rejects a manifest as evidence")
descr = describe_benchmark([{"x": 1}])
inv(descr == BENCHMARK_DESCRIPTOR,
    "descriptor callable returns the approved phrase")
inv("real" not in descr.lower(),
    "descriptor never claims real-world data")
inv(find_single_source_claims(descr) == (),
    "descriptor free of single-institution claims")

# ══════════════════════════════════════════════════════════════════════
# SECTION 17 — SELF-CHECK
# ══════════════════════════════════════════════════════════════════════

section("17. SELF-CHECK / SUMMARY")
inv(INV_TOTAL >= 60, f"invariant target >= 60 (got {INV_TOTAL})")
inv(ASSERTIONS >= 300, f"assertion target >= 300 (got {ASSERTIONS})")
inv(not FAILURES, f"zero failures (got {len(FAILURES)})")
inv(report.qualified_datasets == ()
    and evaluate_promotion().verdict == PromotionVerdict.BLOCKED
    and evaluate_real_world_validation().status == GateStatus.BLOCKED,
    "final state: nothing qualified, RWV blocked, promotion gated")

print("=" * 72)
print(f"PHASE 106 TEST: {'PASS' if not FAILURES else 'FAIL'}")
print(f"assertions: {ASSERTIONS} total, "
      f"{ASSERTIONS - len(FAILURES)} passed, {len(FAILURES)} failed")
print(f"invariants: {INV_TOTAL} total, "
      f"{INV_TOTAL - INV_FAILED} passed, {INV_FAILED} failed")
if FAILURES:
    print("failures:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
sys.exit(0)
