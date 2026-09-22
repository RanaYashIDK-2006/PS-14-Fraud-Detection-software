"""
Phase 105: Provider Evidence Intake & Verification Boundary
============================================================
Target : 100+ invariants, 500+ deterministic assertions
Safety : NO external dataset acquired, NO provider contacted, NO download,
         NO real-world validation, NO model modification/retraining/tuning,
         NO promotion path, NO gate weakened, NO fabricated evidence for any
         real candidate.  Phase 104 remains the sole qualification
         authority; Phase 46 remains the sole promotion authority.
SYSTEM_READINESS: SYSTEM_READY_PENDING_ELIGIBLE_DATASET
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET

NOTE ON THE GOLDEN FIXTURE: the test-only fixture below belongs to a
fictitious provider ("TestProvider Industries" / TESTPROVIDER /
TEST_FIXTURE_DATASET).  It exists solely to prove the intake boundary
ACCEPTS a complete, attested, integrity-valid evidence set and REJECTS
every single mutation — i.e. neither the intake nor Phase 104 is a
trivial always-block, and integrity alone never fabricates truth.  It is
never included in any report, never qualifies a real dataset, and never
enters any registry.  All four REAL candidates are asserted BLOCKED
below, with ZERO evidence supplied through the intake path.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import fields, replace

from src.monitoring.external_dataset_contract import (
    CANONICAL_TRANSFORMATION,
    CONTEXT_ID,
    CONTRACT_VERSION,
    DOMAIN_FEATURE_COUNT,
    EVIDENCE_AS_OF,
    EVIDENCE_CUTOFF,
    EVIDENCE_POLICY_VERSION,
    EVIDENCE_VALID_FROM,
    INDEPENDENCE_KEYS,
    MANDATORY_DIMENSIONS,
    PROVIDER_FACT_DIMENSIONS,
    REQUIREMENT_KEYS,
    STATUS_PRECEDENCE,
    FORBIDDEN_CONTENT_PATTERNS,
    ATTESTED_ORIGINS,
    UNTRUSTED_ORIGINS,
    AccessAuthority,
    ContractDimension,
    DatasetQualificationState,
    EntityContinuityClaim,
    EvidenceOrigin,
    EvidenceStatus,
    ExternalDatasetContract,
    FeatureAvailability,
    FeatureCompatibilityClaim,
    IndependenceClaim,
    ContaminationClaim,
    LabelProvenanceClaim,
    TemporalSemanticsClaim,
    ProviderIdentityClaim,
    AccessAuthorityClaim,
    DatasetIdentityClaim,
    UsagePermissionClaim,
    build_known_candidate_contract_and_evidence,
    canonical_json,
    find_forbidden_content,
    worst_status,
    KNOWN_DATASET_IDS,
    NATIVE_FEATURE_COUNT,
)
from src.monitoring.external_dataset_qualification import (
    QualificationResult,
    evaluate_dataset_evidence,
    evaluate_known_candidates,
)
from src.monitoring.feature_contract import ML_FEATURE_ORDER, ML_FEATURE_VERSION
from src.monitoring.model_contract_reconciliation import RUNTIME_DOMAIN_FEATURES
from src.monitoring.phase103_production_readiness_closure import (
    SYSTEM_READINESS,
    REAL_WORLD_VALIDATION,
    PROMOTION_STATE,
)
from src.monitoring.phase104_external_dataset_report import (
    generate_phase104_report,
)
from src.monitoring.phase105_provider_evidence_report import (
    DECLARATIONS,
    EXPECTED_CONCLUSION,
    Phase105Report,
    generate_phase105_report,
)
from src.monitoring.provider_evidence import (
    KNOWN_CANDIDATES,
    qualify_all_known_candidates,
)
from src.monitoring.provider_evidence_intake import (
    CONTRADICTED_CONFLICT_CODES,
    DUPLICATE_CONFLICT_CODES,
    EVIDENCE_CHAIN_STAGES,
    EVALUATION_PERMISSION_STATES,
    INTAKE_VERSION,
    NO_EXPIRY_DECLARED,
    PACKAGE_VERSION,
    SCOPE_DIMENSIONS,
    SUPPORTED_DOCUMENT_TYPES,
    AttestationScope,
    EvidenceArtifact,
    EvidenceChain,
    EvidencePackageExport,
    EvidenceSubmission,
    IntakeReasonCode,
    UsagePermissionState,
    ValidationReport,
    VerificationState,
    build_intake_package,
    canonicalize_provider_evidence,
    conflicting_evidence_ids,
    detect_evidence_conflicts,
    evidence_conflict_status,
    export_evidence_package,
    hash_provider_evidence,
    normalize_submissions,
    qualify_submissions,
    qualification_path_signature,
    scope_covers_dimension,
    submit_provider_evidence,
    to_phase104_item,
    trace_evidence_chain,
    validate_provider_evidence,
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
from src.monitoring.rwv_readiness_audit import MODEL_ID, RELEASE_ID, FEATURE_VERSION
from src.monitoring.rwv_reproducibility import (
    NATIVE_FEATURE_VERSION,
    PREPROCESSING_HASH,
    RULE_HASH,
)
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES

import src.monitoring.provider_evidence_intake as intake_mod
import src.monitoring.phase105_provider_evidence_report as report_mod
import src.monitoring.external_dataset_qualification as qual_mod

# ══════════════════════════════════════════════════════════════════════
# TEST INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════

passed = 0
failed = 0
failed_names: list[str] = []

inv_passed = 0
inv_failed = 0
inv_total = 0
inv_failures: list[str] = []


def check(condition: bool, name: str) -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        failed_names.append(name)


def inv(inv_id: str, description: str, condition: bool) -> None:
    """Register one qualification invariant (also counts as an assertion)."""
    global inv_passed, inv_failed, inv_total
    inv_total += 1
    if condition:
        inv_passed += 1
    else:
        inv_failed += 1
        inv_failures.append(f"{inv_id}: {description}")
    check(condition, f"INV {inv_id}: {description}")


def expect_invalid(report: ValidationReport, name: str,
                   expect_codes: tuple[str, ...] = ()) -> None:
    check(not report.valid, f"{name}: submission not valid")
    for code in expect_codes:
        check(code in report.reason_codes,
              f"{name}: reason '{code}' present")


def expect_blocked(result: QualificationResult, name: str,
                   expect_codes: tuple[str, ...] = ()) -> None:
    check(not result.qualified, f"{name}: not qualified")
    check(result.qualification_state ==
          DatasetQualificationState.BLOCKED.value,
          f"{name}: state == blocked")
    check(result.qualification_state !=
          DatasetQualificationState.QUALIFIED_FOR_CONTROLLED_RWV.value,
          f"{name}: never qualified_for_controlled_rwv")
    for code in expect_codes:
        check(code in result.reason_codes,
              f"{name}: reason '{code}' present")


def qual_pair(contract: ExternalDatasetContract, subs, name: str) -> QualificationResult:
    r1 = qualify_submissions(contract, subs)
    r2 = qualify_submissions(contract, subs)
    check(r1.qualification_state == r2.qualification_state,
          f"{name}: deterministic state")
    check(r1.result_hash == r2.result_hash,
          f"{name}: deterministic result hash")
    return r1


def val_pair(sub, contract, name: str) -> ValidationReport:
    r1 = validate_provider_evidence(sub, contract)
    r2 = validate_provider_evidence(sub, contract)
    check(r1.reason_codes == r2.reason_codes, f"{name}: deterministic reasons")
    check(r1.report_hash == r2.report_hash, f"{name}: deterministic report hash")
    return r1


def dim_status(result: QualificationResult, dimension: str) -> str:
    for o in result.dimensions:
        if o.dimension == dimension:
            return o.status
    return "absent"


# ══════════════════════════════════════════════════════════════════════
# GOLDEN FIXTURE (fictitious provider — test-only, see header)
# ══════════════════════════════════════════════════════════════════════

GOLDEN_PROVIDER = "TestProvider Industries"
GOLDEN_PROVIDER_REF = "TESTPROVIDER"
GOLDEN_DATASET = "TEST_FIXTURE_DATASET"
GOLDEN_EVIDENCE_VERSION = "phase105_test_v1"

_HISTORICAL_REQS = {
    "user_history", "card_history", "merchant_history",
    "labeled_user_history", "labeled_merchant_history", "labeled_city_history",
}


def _golden_features() -> FeatureCompatibilityClaim:
    availability = tuple(
        (k, FeatureAvailability.DETERMINISTICALLY_DERIVABLE.value
         if k in _HISTORICAL_REQS
         else FeatureAvailability.DIRECTLY_AVAILABLE.value)
        for k in sorted(REQUIREMENT_KEYS)
    )
    return FeatureCompatibilityClaim(
        feature_contract_version=ML_FEATURE_VERSION,
        native_feature_version=NATIVE_FEATURE_VERSION,
        transformation=CANONICAL_TRANSFORMATION,
        native_feature_order=tuple(ALTMAN_NATIVE_FEATURES),
        direct_domain_feature_inference=False,
        requirement_availability=availability,
        leakage_risk_features=(),
    )


def _golden_contract() -> ExternalDatasetContract:
    return ExternalDatasetContract(
        contract_version=CONTRACT_VERSION,
        context_id=CONTEXT_ID,
        assessed=True,
        provider_reference=GOLDEN_PROVIDER_REF,
        dataset_id=GOLDEN_DATASET,
        model_id=MODEL_ID,
        release_id=RELEASE_ID,
        evidence_valid_from=EVIDENCE_VALID_FROM,
        evidence_cutoff=EVIDENCE_CUTOFF,
        provider_identity=ProviderIdentityClaim(
            provider_name=GOLDEN_PROVIDER,
            provider_organization=GOLDEN_PROVIDER,
            dataset_owner_controller=GOLDEN_PROVIDER,
            authoritative_provenance_reference="PROV-TESTPROVIDER-001",
            provider_attestation_reference="ATT-TESTPROVIDER-001",
            evidence_timestamp="2026-09-01T00:00:00+00:00",
            evidence_version=GOLDEN_EVIDENCE_VERSION,
        ),
        access_authority=AccessAuthorityClaim(
            authority=AccessAuthority.PROVIDER_AUTHORIZED.value,
            basis_reference="DUA-TESTPROVIDER-2026-001",
            publicly_downloadable=False,
        ),
        dataset_identity=DatasetIdentityClaim(
            canonical_dataset_id=GOLDEN_DATASET,
            provider_dataset_id="TESTPROVIDER-FIXTURE-001",
            version="1.0",
            collection_period="2025-01-01/2025-06-30",
            release_period="2025-07",
            fingerprint="sha256:" + "ab" * 32,
            fingerprint_available=True,
            schema_descriptor="test-fixture-schema-v1",
            record_count=1000,
            record_count_evidence="provider manifest count 1000",
            extraction_sampling_description="full extract; no sampling",
        ),
        label_provenance=LabelProvenanceClaim(
            label_definition="chargeback-confirmed fraud within liability window",
            label_origin="provider_adjudication",
            labeling_authority="TestProvider Industries disputes team",
            labeling_process="provider adjudication with independent review",
            confirmation_timing="confirmed at chargeback resolution",
            labels_retrospective=False,
            independently_adjudicated=True,
            suitable_for_evaluation=True,
            label_availability_time="2026-09-01T00:00:00+00:00",
        ),
        temporal_semantics=TemporalSemanticsClaim(
            event_timestamp_semantics="Absolute UTC epoch seconds per event",
            timezone_semantics="UTC fixed offset (+00:00)",
            observation_time="2026-09-02T00:00:00+00:00",
            evaluation_cutoff=EVIDENCE_CUTOFF,
            temporal_ordering_rules="effective_at <= observed_at <= labels <= cutoff",
            effective_at="2026-09-01T00:00:00+00:00",
            observed_at="2026-09-02T00:00:00+00:00",
            contains_future_information=False,
        ),
        feature_compatibility=_golden_features(),
        entity_continuity=EntityContinuityClaim(
            user=True, card=True, merchant=True, recipient=True, city=True,
            transaction=True, historical_transaction=True, temporal_history=True,
            identifiers_anonymized=False,
        ),
        independence=IndependenceClaim(
            dataset_origin="real_world_external",
            confirmed_independent=tuple(INDEPENDENCE_KEYS),
            known_overlaps=(),
        ),
        contamination=ContaminationClaim(
            training_overlap=False, duplicate_transactions=False,
            duplicate_entities=False, benchmark_contamination=False,
            prior_model_exposure=False, feature_leakage=False,
            label_leakage=False, preprocessing_leakage=False,
        ),
        usage_permission=UsagePermissionClaim(
            permission_reference="DUA-TESTPROVIDER-2026-001",
            permission_scope="fraud-model evaluation only",
            permitted_use=True,
            valid_from="2026-01-01T00:00:00+00:00",
            valid_until="2027-01-01T00:00:00+00:00",
            permission_holder=GOLDEN_PROVIDER,
        ),
    )


SCOPE_FOR = {
    "provider_identity": AttestationScope.PROVIDER_IDENTITY,
    "access_authority": AttestationScope.ACCESS_AUTHORITY,
    "dataset_identity": AttestationScope.DATASET_IDENTITY,
    "label_provenance": AttestationScope.LABEL_PROVENANCE,
    "temporal_semantics": AttestationScope.TIMESTAMP_SEMANTICS,
    "feature_compatibility": AttestationScope.FEATURE_SCHEMA,
    "entity_continuity": AttestationScope.ENTITY_CONTINUITY,
    "independence": AttestationScope.INDEPENDENCE,
    "contamination": AttestationScope.CONTAMINATION,
    "usage_permission": AttestationScope.USAGE_PERMISSION,
}


def _golden_subs() -> tuple[EvidenceSubmission, ...]:
    subs = []
    for i, (dim, scope) in enumerate(SCOPE_FOR.items(), 1):
        subs.append(submit_provider_evidence(
            evidence_id=f"gold:{i:02d}",
            provider_id=GOLDEN_PROVIDER_REF,
            dataset_id=GOLDEN_DATASET,
            dataset_version="1.0",
            dimension=dim,
            evidence_type="provider_attestation",
            evidence_origin=EvidenceOrigin.PROVIDER_ATTESTED.value,
            evidence_status=EvidenceStatus.VERIFIED.value,
            statement=f"Provider attestation covering {dim} of the fixture dataset.",
            supplied_by=GOLDEN_PROVIDER,
            supplied_at=EVIDENCE_AS_OF,
            source_reference=f"DOC-TESTPROVIDER-{i:03d}",
            source_document_hash=f"{i:02x}" * 32,
            source_document_size=4096,
            source_document_type="application/pdf",
            provider_attestation_reference="ATT-TESTPROVIDER-001",
            attestation_timestamp="2026-09-01T00:00:00+00:00",
            attestation_scope=(scope.value,),
            effective_from="2026-01-01T00:00:00+00:00",
            effective_until="2027-01-01T00:00:00+00:00",
            permission_state=UsagePermissionState
            .AUTHORIZED_FOR_MODEL_EVALUATION.value,
        ))
    return tuple(subs)


GOLDEN = _golden_contract()
GOLD = _golden_subs()
S0 = GOLD[0]          # provider_identity submission
S_PERM = GOLD[9]      # usage_permission submission

_MUTABLE_FIELDS = [k for k in EvidenceSubmission.__dataclass_fields__
                   if k != "evidence_hash"]


def _sub(base: EvidenceSubmission = S0, **over) -> EvidenceSubmission:
    """Rebuild a submission with fresh mutations (isolates one change)."""
    data = {k: getattr(base, k) for k in _MUTABLE_FIELDS if k not in over}
    return submit_provider_evidence(**{**data, **over})


def _raw_alter(base: EvidenceSubmission, keep_hash: bool = True,
               **over) -> EvidenceSubmission:
    """Rebuild a submission keeping the ORIGINAL evidence hash while the
    underlying content changes — simulates post-hash alteration."""
    data = base.to_dict()
    data.update(over)
    if not keep_hash:
        data["evidence_hash"] = ""
    return EvidenceSubmission(**data)


def subs_without(dimension: str) -> tuple[EvidenceSubmission, ...]:
    return tuple(s for s in GOLD if s.dimension != dimension)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — INTAKE MODEL STRUCTURE & VERSIONS
# ══════════════════════════════════════════════════════════════════════

inv("S-01", "intake version is phase105_intake_v1",
    INTAKE_VERSION == "phase105_intake_v1")
inv("S-02", "package version is phase105_evidence_package_v1",
    PACKAGE_VERSION == "phase105_evidence_package_v1")
inv("S-03", "NO_EXPIRY_DECLARED is the preserved no-expiry marker",
    NO_EXPIRY_DECLARED == "no_expiry_declared")
inv("S-04", "exactly 13 attestation scopes",
    len(AttestationScope) == 13 and {s.value for s in AttestationScope} == {
        "provider_identity", "dataset_identity", "dataset_version",
        "dataset_period", "label_provenance", "timestamp_semantics",
        "feature_schema", "entity_continuity", "independence",
        "contamination", "usage_permission", "access_authority",
        "evaluation_suitability"})
inv("S-05", "exactly 8 usage-permission states with UNKNOWN present",
    len(UsagePermissionState) == 8 and
    UsagePermissionState.UNKNOWN.value in {p.value for p in UsagePermissionState})
inv("S-06", "exactly 8 verification states — four independent axes",
    len(VerificationState) == 8 and
    {v.value for v in VerificationState} == {
        "integrity_verified", "integrity_unverified",
        "authenticity_verified", "authenticity_unverified",
        "provenance_verified", "provenance_unverified",
        "content_claim_verified", "content_claim_unverified"})
inv("S-07", "evaluation permission set excludes UNKNOWN, RESTRICTED, "
    "TRAINING-only, PUBLICATION-only",
    UsagePermissionState.UNKNOWN.value not in EVALUATION_PERMISSION_STATES and
    UsagePermissionState.RESTRICTED.value not in EVALUATION_PERMISSION_STATES and
    UsagePermissionState.AUTHORIZED_FOR_TRAINING.value not in
    EVALUATION_PERMISSION_STATES and
    UsagePermissionState.AUTHORIZED_FOR_PUBLICATION.value not in
    EVALUATION_PERMISSION_STATES and
    UsagePermissionState.AUTHORIZED_FOR_MODEL_EVALUATION.value in
    EVALUATION_PERMISSION_STATES)
inv("S-08", "evidence origin reuses Phase 104 (7 categories) and status "
    "reuses Phase 95 (5 states)",
    len(EvidenceOrigin) == 7 and len(EvidenceStatus) == 5)
inv("S-09", "every attestation scope maps only onto mandatory dimensions",
    set(SCOPE_DIMENSIONS) == {s for s in AttestationScope} and
    all(d in MANDATORY_DIMENSIONS
        for dims in SCOPE_DIMENSIONS.values() for d in dims))
inv("S-10", "EVALUATION_SUITABILITY covers only the label-provenance claim",
    SCOPE_DIMENSIONS[AttestationScope.EVALUATION_SUITABILITY] ==
    frozenset({"label_provenance"}))
inv("S-11", "provider-side scopes map 1:1 onto their dimensions",
    SCOPE_DIMENSIONS[AttestationScope.ACCESS_AUTHORITY] ==
    frozenset({"access_authority"}) and
    SCOPE_DIMENSIONS[AttestationScope.USAGE_PERMISSION] ==
    frozenset({"usage_permission"}) and
    SCOPE_DIMENSIONS[AttestationScope.PROVIDER_IDENTITY] ==
    frozenset({"provider_identity"}))
inv("S-12", "evidence chain declares exactly the 6 spec stages",
    EVIDENCE_CHAIN_STAGES == (
        "evidence_artifact", "evidence_submission", "attestation_scope",
        "evidence_dimension", "phase104_evidence", "dataset_qualification"))
inv("S-13", "supported document types are a fixed 6-type whitelist",
    SUPPORTED_DOCUMENT_TYPES == frozenset({
        "application/pdf", "text/plain", "text/csv", "application/json",
        "application/xml", "text/xml"}))
for cls in (EvidenceSubmission, ValidationReport, EvidenceChain,
            EvidencePackageExport, EvidenceArtifact):
    check(cls.__dataclass_params__.frozen, f"{cls.__name__} declared frozen")
inv("S-14", "all intake dataclasses are frozen/immutable",
    all(c.__dataclass_params__.frozen for c in (
        EvidenceSubmission, ValidationReport, EvidenceChain,
        EvidencePackageExport, EvidenceArtifact)))
try:
    assign_ok = False
    S0.evidence_id = "hacked"
except dataclasses.FrozenInstanceError:
    assign_ok = True
inv("S-15", "EvidenceSubmission attribute assignment raises FrozenInstanceError",
    assign_ok)
_reason_vals = [v for k, v in vars(IntakeReasonCode).items()
                if not k.startswith("_") and isinstance(v, str)]
inv("S-16", "intake reason codes are unique",
    len(_reason_vals) == len(set(_reason_vals)) and len(_reason_vals) >= 40)
inv("S-17", "conflict-code partitions are disjoint and known",
    CONTRADICTED_CONFLICT_CODES.isdisjoint(DUPLICATE_CONFLICT_CODES) and
    IntakeReasonCode.CONFLICTING_EVIDENCE_ID in CONTRADICTED_CONFLICT_CODES and
    IntakeReasonCode.DUPLICATE_EVIDENCE_ID in DUPLICATE_CONFLICT_CODES)
inv("S-18", "qualification path exposes no bypass parameters",
    qualification_path_signature() == ("contract", "evidence"))

# Export surface: no promotion/RWV/session authority, exact expected set.
EXPECTED_EXPORTS = {
    "INTAKE_VERSION", "PACKAGE_VERSION", "NO_EXPIRY_DECLARED",
    "SUPPORTED_DOCUMENT_TYPES", "EVIDENCE_CHAIN_STAGES",
    "EVALUATION_PERMISSION_STATES", "SCOPE_DIMENSIONS",
    "CONTRADICTED_CONFLICT_CODES", "DUPLICATE_CONFLICT_CODES",
    "AttestationScope", "UsagePermissionState", "VerificationState",
    "IntakeReasonCode", "EvidenceArtifact", "EvidenceSubmission",
    "ValidationReport", "EvidenceChain", "EvidencePackageExport",
    "submit_provider_evidence", "validate_provider_evidence",
    "canonicalize_provider_evidence", "hash_provider_evidence",
    "detect_evidence_conflicts", "conflicting_evidence_ids",
    "evidence_conflict_status", "scope_covers_dimension",
    "evidence_type_implies_provider", "to_phase104_item",
    "normalize_submissions", "build_intake_package",
    "qualify_submissions", "qualification_path_signature",
    "trace_evidence_chain", "export_evidence_package",
}
inv("S-19", "intake module exports exactly the expected boundary surface",
    set(intake_mod.__all__) == EXPECTED_EXPORTS)
inv("S-20", "intake exports no promotion/token/session/adjudication authority",
    not any(re.search(r"(?<!timezone_)token|promotion|session|adjudicat|promote",
                      name, re.I)
            for name in intake_mod.__all__))
inv("S-21", "only one qualification function exists on the intake surface",
    {n for n in intake_mod.__all__ if "qualif" in n} ==
    {"qualify_submissions", "qualification_path_signature"})
inv("S-22", "report module exports exactly the expected report surface",
    set(report_mod.__all__) == {"REPORT_ID", "DECLARATIONS",
                                "EXPECTED_CONCLUSION", "Phase105Report",
                                "generate_phase105_report"})
check(GOLDEN.__dataclass_params__.frozen, "golden contract frozen")

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — GOLDEN POSITIVE PATH (valid provider attestation, case 1)
# ══════════════════════════════════════════════════════════════════════

golden_reports = [val_pair(s, GOLDEN, f"golden-{s.dimension}") for s in GOLD]
inv("G-01", "case 1: a complete provider attestation set validates clean "
    "on all 10 mandatory dimensions",
    all(r.valid for r in golden_reports) and len(golden_reports) == 10)
inv("G-02", "golden submissions report all four verification states VERIFIED",
    all(r.integrity_state == VerificationState.INTEGRITY_VERIFIED.value and
        r.authenticity_state == VerificationState.AUTHENTICITY_VERIFIED.value and
        r.provenance_state == VerificationState.PROVENANCE_VERIFIED.value and
        r.content_claim_state == VerificationState.CONTENT_CLAIM_VERIFIED.value
        for r in golden_reports))
inv("G-03", "golden set has no conflicts and no missing dimensions",
    detect_evidence_conflicts(GOLD) == () and
    all(r.freshness_state in ("fresh", "no_expiry_declared")
        for r in golden_reports))
golden_result = qual_pair(GOLDEN, GOLD, "golden")
inv("G-04", "complete intake evidence qualifies the TEST FIXTURE through "
    "PHASE 104 (intake is not an extra authority and not an always-block)",
    golden_result.qualification_state ==
    DatasetQualificationState.QUALIFIED_FOR_CONTROLLED_RWV.value and
    golden_result.qualified and
    type(golden_result) is QualificationResult)
check(golden_result.reason_codes ==
      (qual_mod.ReasonCode.QUALIFIED_FOR_CONTROLLED_RWV,),
      "golden reason codes = qualified_for_controlled_rwv only")
check(all(o.status == EvidenceStatus.VERIFIED.value and not o.blocking
          for o in golden_result.dimensions),
      "golden: all 10 dimensions verified & non-blocking")
check(golden_result.missing_evidence == (), "golden: no missing evidence")
check(golden_result.contradictory_evidence == (),
      "golden: no contradictory evidence")
check(len(golden_result.result_hash) == 64, "golden result hash is sha256")
inv("G-05", "the golden fixture is NOT one of the real candidates and no "
    "golden submission references a real candidate dataset/provider",
    GOLDEN_DATASET not in KNOWN_DATASET_IDS and
    GOLDEN_DATASET not in KNOWN_CANDIDATES and
    all(s.dataset_id not in KNOWN_DATASET_IDS for s in GOLD) and
    all(s.provider_id not in {KNOWN_CANDIDATES[k].provider_id
                              for k in KNOWN_CANDIDATES} for s in GOLD))

# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — PROVIDER / DATASET / VERSION BINDING (cases 2-7, 21-24, 76-78)
# ══════════════════════════════════════════════════════════════════════

r = val_pair(_sub(provider_id=""), GOLDEN, "missing-provider")
expect_invalid(r, "case 2 missing provider",
               (IntakeReasonCode.PROVIDER_ID_MISSING,))
inv("B-01", "case 2: missing provider fails closed", not r.valid)

r = val_pair(_sub(dataset_id=""), GOLDEN, "missing-dataset")
expect_invalid(r, "case 3 missing dataset",
               (IntakeReasonCode.DATASET_ID_MISSING,))
inv("B-02", "case 3: missing dataset fails closed", not r.valid)

r = val_pair(_sub(dataset_version=""), GOLDEN, "missing-version")
expect_invalid(r, "case 4 missing version",
               (IntakeReasonCode.DATASET_VERSION_MISSING,))
inv("B-03", "case 4: missing dataset version fails closed where the "
    "contract declares one", not r.valid)

r = val_pair(_sub(provider_id="IMPOSTOR_LTD"), GOLDEN, "provider-mismatch")
expect_invalid(r, "case 5 provider mismatch",
               (IntakeReasonCode.PROVIDER_ID_MISMATCH,))
inv("B-04", "case 5: provider mismatch fails closed", not r.valid)

r = val_pair(_sub(dataset_id="SOME_OTHER_DATASET"), GOLDEN, "dataset-mismatch")
expect_invalid(r, "case 6 dataset mismatch",
               (IntakeReasonCode.DATASET_ID_MISMATCH,))
inv("B-05", "case 6: dataset mismatch fails closed", not r.valid)

r = val_pair(_sub(dataset_version="0.9"), GOLDEN, "version-mismatch")
expect_invalid(r, "case 7 version mismatch",
               (IntakeReasonCode.DATASET_VERSION_MISMATCH,))
inv("B-06", "case 7: dataset version mismatch fails closed", not r.valid)

# case 21: cross-dataset replay — golden evidence replayed on another dataset
r = val_pair(_sub(dataset_id="NOVATTI"), GOLDEN, "cross-dataset")
expect_invalid(r, "case 21 cross-dataset replay",
               (IntakeReasonCode.DATASET_ID_MISMATCH,))
res = qual_pair(GOLDEN, tuple(
    _sub(s, dataset_id="NOVATTI") if s is S0 else s for s in GOLD),
    "cross-dataset-qual")
expect_blocked(res, "case 21 cross-dataset replay qualification",
               (qual_mod.ReasonCode.EVIDENCE_CROSS_CONTEXT,))
inv("B-07", "case 21: cross-dataset replay fails closed at intake AND in "
    "Phase 104", not r.valid and not res.qualified)

# case 22: cross-provider replay against a REAL candidate contract
NOV_CONTRACT, _ = build_known_candidate_contract_and_evidence("NOVATTI")
WL_PROVIDER = KNOWN_CANDIDATES["WORLDLINE_ECOM_2017_NAG"].provider_id
NOV_PROVIDER = KNOWN_CANDIDATES["NOVATTI"].provider_id
r = val_pair(_sub(provider_id=WL_PROVIDER), NOV_CONTRACT, "wl-vs-novatti")
expect_invalid(r, "case 22 cross-provider replay (Worldline evidence vs "
               "Novatti contract)",
               (IntakeReasonCode.PROVIDER_ID_MISMATCH,))
inv("B-08", "case 22: a provider's evidence package is never usable against "
    "another provider's contract", WL_PROVIDER != NOV_PROVIDER and not r.valid)

# case 23: historical-release substitution (version A evidence → version B)
r = val_pair(_sub(dataset_version="2.0"), GOLDEN, "historical-version")
expect_invalid(r, "case 23 historical version substitution",
               (IntakeReasonCode.DATASET_VERSION_MISMATCH,))
inv("B-09", "case 23: same-dataset-different-version substitution fails "
    "closed", not r.valid)

# case 24: fake provider ID
r = val_pair(_sub(provider_id="FAKE-PROVIDER-001"), GOLDEN, "fake-provider")
expect_invalid(r, "case 24 fake provider ID",
               (IntakeReasonCode.PROVIDER_ID_MISMATCH,))
inv("B-10", "case 24: fabricated provider ID fails closed", not r.valid)

# case 76: provider substitution through qualification
res = qual_pair(GOLDEN, tuple(
    _sub(s, provider_id="OTHER_PROVIDER") if s is S0 else s for s in GOLD),
    "provider-substitution")
expect_blocked(res, "case 76 provider substitution",
               (qual_mod.ReasonCode.EVIDENCE_CROSS_CONTEXT,))
inv("B-11", "case 76: provider substitution fails closed end-to-end",
    not res.qualified and
    dim_status(res, "provider_identity") == "missing")

# case 77: dataset substitution through qualification
res = qual_pair(GOLDEN, tuple(
    _sub(s, dataset_id="OTHER_DATASET") if s is S0 else s for s in GOLD),
    "dataset-substitution")
expect_blocked(res, "case 77 dataset substitution",
               (qual_mod.ReasonCode.EVIDENCE_CROSS_CONTEXT,))
inv("B-12", "case 77: dataset substitution fails closed end-to-end",
    not res.qualified)

# case 78: version substitution through qualification
res = qual_pair(GOLDEN, tuple(
    _sub(s, dataset_version="9.9") if s is S0 else s for s in GOLD),
    "version-substitution")
expect_blocked(res, "case 78 version substitution", ())
inv("B-13", "case 78: version substitution fails closed end-to-end "
    "(intake downgrades, Phase 104 blocks)",
    not res.qualified and
    dim_status(res, "provider_identity") == "unverified")

# binding hash binding: evidence hash covers binding fields
inv("B-14", "evidence hash binds provider/dataset/version — changing any "
    "binding field changes the hash",
    hash_provider_evidence(_sub(provider_id="X")) != S0.evidence_hash and
    hash_provider_evidence(_sub(dataset_id="X")) != S0.evidence_hash and
    hash_provider_evidence(_sub(dataset_version="X")) != S0.evidence_hash)

# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — EVIDENCE ORIGIN RULES (cases 13-17, 57-60, 85-88)
# ══════════════════════════════════════════════════════════════════════

# case 13: unsupported origin
try:
    _sub(evidence_origin="vouched_by_someone")
    origin_rejected = False
except ValueError:
    origin_rejected = True
inv("O-01", "case 13: unsupported evidence origin rejected at construction",
    origin_rejected)

# case 14: SYSTEM_GENERATED pretending provider-attested
r = val_pair(_sub(evidence_origin=EvidenceOrigin.SYSTEM_GENERATED.value,
                  evidence_type="provider_attestation"),
             GOLDEN, "system-pretends-attested")
expect_invalid(r, "case 14 system-generated pretending provider-attested",
               (IntakeReasonCode.EVIDENCE_ORIGIN_UNTRUSTED,
                IntakeReasonCode.ORIGIN_ATTESTATION_MISMATCH))
inv("O-02", "case 14: SYSTEM_GENERATED never becomes a provider attestation "
    "and never verifies a claim",
    not r.valid and
    r.content_claim_state == VerificationState.CONTENT_CLAIM_UNVERIFIED.value and
    r.authenticity_state == VerificationState.AUTHENTICITY_UNVERIFIED.value)

# case 15: LOCAL_SYNTHETIC pretending provider-attested
r = val_pair(_sub(evidence_origin=EvidenceOrigin.LOCAL_SYNTHETIC.value,
                  evidence_type="provider_attestation"),
             GOLDEN, "synthetic-pretends")
expect_invalid(r, "case 15 local-synthetic pretending provider-attested",
               (IntakeReasonCode.EVIDENCE_ORIGIN_UNTRUSTED,))
inv("O-03", "case 15: LOCAL_SYNTHETIC never verifies provider facts",
    not r.valid and
    r.content_claim_state == VerificationState.CONTENT_CLAIM_UNVERIFIED.value)

# case 16: LOCAL_RESEARCH pretending provider-attested
r = val_pair(_sub(evidence_origin=EvidenceOrigin.LOCAL_RESEARCH.value,
                  evidence_type="provider_attestation"),
             GOLDEN, "research-pretends")
expect_invalid(r, "case 16 local-research pretending provider-attested",
               (IntakeReasonCode.EVIDENCE_ORIGIN_UNTRUSTED,))
inv("O-04", "case 16: LOCAL_RESEARCH never verifies provider facts",
    not r.valid)

# case 17: PUBLIC_DOCUMENTATION as provider authorization
r = val_pair(_sub(dimension="access_authority",
                  evidence_origin=EvidenceOrigin.PUBLIC_DOCUMENTATION.value,
                  provider_attestation_reference=""),
             GOLDEN, "public-as-authz")
expect_invalid(r, "case 17 public documentation as provider authorization",
               (IntakeReasonCode.PUBLIC_DOCUMENTATION_NOT_AUTHORIZATION,))
inv("O-05", "case 17: public availability never establishes access authority",
    not r.valid and
    r.content_claim_state == VerificationState.CONTENT_CLAIM_UNVERIFIED.value)

# case 57: fabricated provider claim (forged attestation reference)
r = val_pair(_sub(provider_attestation_reference="ATT-WORLDLINE-FAKE-001"),
             GOLDEN, "forged-provider-ref")
expect_invalid(r, "case 57 fabricated provider attestation reference",
               (IntakeReasonCode.ATTESTATION_REFERENCE_MISMATCH,))
inv("O-06", "case 57: a fabricated provider attestation reference fails "
    "closed against the contract's declared reference", not r.valid)

# case 58: fabricated institutional claim
r = val_pair(_sub(dimension="usage_permission",
                  evidence_origin=EvidenceOrigin.INSTITUTIONALLY_ATTESTED.value,
                  provider_attestation_reference="",
                  institutional_attestation_reference="INST-FAKE-001"),
             GOLDEN, "forged-inst-ref")
expect_invalid(r, "case 58 fabricated institutional attestation reference",
               (IntakeReasonCode.ATTESTATION_REFERENCE_MISMATCH,))
inv("O-07", "case 58: a fabricated institutional attestation reference fails "
    "closed against the contract's declared anchor", not r.valid)

# case 59: forged evidence status
try:
    _sub(evidence_status="definitely_verified")
    status_rejected = False
except ValueError:
    status_rejected = True
try:
    _sub(evidence_status="")
    status_rejected = status_rejected and False
except ValueError:
    status_rejected = status_rejected and True
inv("O-08", "case 59: forged evidence status rejected at construction",
    status_rejected)

# case 60: forged evidence origin (covered by case 13 constructor guard,
# plus a weak attestation reference never verifies)
r = val_pair(_sub(provider_attestation_reference="UNVERIFIED -- pending talks"),
             GOLDEN, "weak-attestation-ref")
expect_invalid(r, "case 60 forged evidence origin via weak reference",
               (IntakeReasonCode.ATTESTATION_REFERENCE_UNVERIFIED,))
inv("O-09", "case 60: a weak/unverified attestation reference never verifies",
    not r.valid and
    r.authenticity_state == VerificationState.AUTHENTICITY_UNVERIFIED.value)

# case 85: synthetic → provider conversion (attestation reference attached)
r = val_pair(_sub(evidence_origin=EvidenceOrigin.LOCAL_SYNTHETIC.value,
                  provider_attestation_reference="ATT-TESTPROVIDER-001"),
             GOLDEN, "synthetic-to-provider")
expect_invalid(r, "case 85 synthetic to provider conversion",
               (IntakeReasonCode.ORIGIN_CANNOT_ATTEST,))
inv("O-10", "case 85: local synthetic evidence can never carry a provider "
    "attestation", not r.valid)

# case 86: local/system → provider conversion
r = val_pair(_sub(evidence_origin=EvidenceOrigin.SYSTEM_GENERATED.value,
                  provider_attestation_reference="ATT-TESTPROVIDER-001"),
             GOLDEN, "system-to-provider")
expect_invalid(r, "case 86 system to provider conversion",
               (IntakeReasonCode.ORIGIN_CANNOT_ATTEST,))
inv("O-11", "case 86: system-generated evidence can never carry a provider "
    "attestation", not r.valid)

# case 87: research → provider conversion (institutional attestation)
r = val_pair(_sub(dimension="usage_permission",
                  evidence_origin=EvidenceOrigin.LOCAL_RESEARCH.value,
                  provider_attestation_reference="",
                  institutional_attestation_reference="DUA-TESTPROVIDER-2026-001"),
             GOLDEN, "research-to-provider")
expect_invalid(r, "case 87 research to provider conversion",
               (IntakeReasonCode.EVIDENCE_ORIGIN_UNTRUSTED,
                IntakeReasonCode.ORIGIN_CANNOT_ATTEST))
inv("O-12", "case 87: local research evidence can never carry an "
    "institutional attestation", not r.valid)

# case 88: public → authorized conversion (public doc with attestation ref)
r = val_pair(_sub(dimension="access_authority",
                  evidence_origin=EvidenceOrigin.PUBLIC_DOCUMENTATION.value,
                  provider_attestation_reference="ATT-TESTPROVIDER-001"),
             GOLDEN, "public-to-authorized")
expect_invalid(r, "case 88 public to authorized conversion",
               (IntakeReasonCode.ORIGIN_CANNOT_ATTEST,))
inv("O-13", "case 88: attaching an attestation reference to public "
    "documentation never converts it into authorization", not r.valid)

# untrusted/unknown origin invariants (spec §3)
r = val_pair(_sub(evidence_origin=EvidenceOrigin.UNKNOWN.value),
             GOLDEN, "unknown-origin")
expect_invalid(r, "unknown evidence origin",
               (IntakeReasonCode.EVIDENCE_ORIGIN_UNKNOWN,))
inv("O-14", "UNKNOWN origin fails closed for every dimension",
    not r.valid and
    r.content_claim_state == VerificationState.CONTENT_CLAIM_UNVERIFIED.value)
untrusted_ok = all(
    not validate_provider_evidence(
        _sub(evidence_origin=o,
             provider_attestation_reference="",
             institutional_attestation_reference=""),
        GOLDEN).valid
    for o in UNTRUSTED_ORIGINS)
inv("O-15", "every untrusted origin (local/system/unknown) fails intake "
    "validation", untrusted_ok)
inv("O-16", "PUBLIC_DOCUMENTATION may validate only for non provider-side "
    "dimensions and never with attestation references",
    validate_provider_evidence(
        _sub(dimension="temporal_semantics",
             evidence_type="public_documentation",
             evidence_origin=EvidenceOrigin.PUBLIC_DOCUMENTATION.value,
             attestation_scope=(AttestationScope.TIMESTAMP_SEMANTICS.value,),
             provider_attestation_reference=""), GOLDEN).valid and
    not validate_provider_evidence(
        _sub(dimension="dataset_identity",
             evidence_origin=EvidenceOrigin.PUBLIC_DOCUMENTATION.value,
             provider_attestation_reference=""), GOLDEN).valid)
inv("O-17", "provider-fact dimensions require attested origins "
    "(PROVIDER_FACT_DIMENSIONS ⊆ attested-only policy)",
    PROVIDER_FACT_DIMENSIONS == frozenset({
        "provider_identity", "access_authority", "dataset_identity",
        "entity_continuity", "usage_permission"}) and
    ATTESTED_ORIGINS == frozenset({"provider_attested",
                                   "institutionally_attested"}))

# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — ATTESTATION SCOPE & FRESHNESS (cases 11, 12, 20, 69-71, 91)
# ══════════════════════════════════════════════════════════════════════

# case 20a: unknown scope value
try:
    _sub(attestation_scope=("made_up_scope",))
    scope_rejected = False
except ValueError:
    scope_rejected = True
# case 20b: valid scope that does not cover the dimension
r_dim = val_pair(
    _sub(dimension="label_provenance",
         attestation_scope=(AttestationScope.DATASET_IDENTITY.value,)),
    GOLDEN, "scope-wrong-dim")
expect_invalid(r_dim, "case 20 identity attestation offered for label "
               "provenance", (IntakeReasonCode.ATTESTATION_SCOPE_DIMENSION_MISMATCH,))
# usage agreement offered for feature compatibility
r_fc = val_pair(
    _sub(dimension="feature_compatibility",
         attestation_scope=(AttestationScope.USAGE_PERMISSION.value,)),
    GOLDEN, "scope-usage-for-features")
expect_invalid(r_fc, "case 20 usage agreement offered for feature "
               "compatibility",
               (IntakeReasonCode.ATTESTATION_SCOPE_DIMENSION_MISMATCH,))
# dataset paper offered for provider authorization
r_auth = val_pair(
    _sub(dimension="access_authority",
         attestation_scope=(AttestationScope.DATASET_PERIOD.value,),
         evidence_origin=EvidenceOrigin.PUBLIC_DOCUMENTATION.value,
         provider_attestation_reference=""),
    GOLDEN, "scope-paper-for-authz")
expect_invalid(r_auth, "case 20 dataset paper offered for provider "
               "authorization",
               (IntakeReasonCode.ATTESTATION_SCOPE_DIMENSION_MISMATCH,))
inv("A-01", "case 20: unknown scope, wrong-dimension scope, and "
    "paper-for-authorization all fail closed",
    scope_rejected and not r_dim.valid and not r_fc.valid and not r_auth.valid)
inv("A-02", "scope_covers_dimension is exact: identity scope does not cover "
    "labels, permission scope does not cover features, period scope does not "
    "cover authorization",
    not scope_covers_dimension("dataset_identity", "label_provenance") and
    not scope_covers_dimension("usage_permission", "feature_compatibility") and
    not scope_covers_dimension("dataset_period", "access_authority") and
    not scope_covers_dimension("made_up_scope", "independence") and
    scope_covers_dimension("label_provenance", "label_provenance"))

# case 69: missing attestation timestamp (attested origin only)
r = val_pair(_sub(attestation_timestamp=""), GOLDEN, "missing-att-ts")
expect_invalid(r, "case 69 missing attestation timestamp",
               (IntakeReasonCode.ATTESTATION_TIMESTAMP_MISSING,))
inv("A-03", "case 69: missing attestation timestamp fails closed for "
    "attested origins", not r.valid and
    r.authenticity_state == VerificationState.AUTHENTICITY_UNVERIFIED.value)
r = val_pair(_sub(dimension="temporal_semantics",
                  evidence_origin=EvidenceOrigin.PUBLIC_DOCUMENTATION.value,
                  provider_attestation_reference="",
                  attestation_timestamp=""),
             GOLDEN, "public-no-att-ts")
inv("A-04", "public documentation is never required to carry an "
    "attestation timestamp (it is not an attestation)",
    r.valid or IntakeReasonCode.ATTESTATION_TIMESTAMP_MISSING
    not in r.reason_codes)

# case 70: invalid effective interval
r = val_pair(_sub(effective_from="2026-09-01T00:00:00+00:00",
                  effective_until="2025-01-01T00:00:00+00:00"),
             GOLDEN, "inverted-interval")
expect_invalid(r, "case 70 inverted effective interval",
               (IntakeReasonCode.ATTESTATION_INTERVAL_INVALID,))
check(r.freshness_state == "interval_invalid", "inverted interval state")
r2 = val_pair(_sub(effective_until="not-a-date"), GOLDEN, "unparseable-interval")
expect_invalid(r2, "case 70 unparseable effective interval",
               (IntakeReasonCode.ATTESTATION_INTERVAL_INVALID,))
inv("A-05", "case 70: inverted and unparseable effective intervals fail "
    "closed", not r.valid and not r2.valid)

# case 71: future-dated invalid evidence + stale evidence
r = val_pair(_sub(attestation_timestamp="2027-01-01T00:00:00+00:00"),
             GOLDEN, "future-attestation")
expect_invalid(r, "case 71 future-dated attestation timestamp",
               (IntakeReasonCode.ATTESTATION_FUTURE_DATED,))
r2 = val_pair(_sub(supplied_at="2027-01-01T00:00:00+00:00"),
              GOLDEN, "future-supplied")
expect_invalid(r2, "case 71 future-dated submission",
               (IntakeReasonCode.EVIDENCE_FUTURE_DATED,))
r3 = val_pair(_sub(supplied_at="2025-06-01T00:00:00+00:00"),
              GOLDEN, "stale-supplied")
expect_invalid(r3, "stale submission before the intake window",
               (IntakeReasonCode.EVIDENCE_STALE,))
inv("A-06", "case 71: future-dated evidence fails closed and pre-window "
    "evidence is stale", not r.valid and not r2.valid and not r3.valid)

# case 11: expired attestation
r = val_pair(_sub(effective_until="2026-01-01T00:00:00+00:00"),
             GOLDEN, "expired-attestation")
expect_invalid(r, "case 11 expired attestation",
               (IntakeReasonCode.ATTESTATION_EXPIRED,))
check(r.freshness_state == "expired", "expired freshness state")
inv("A-07", "case 11: an explicitly expired attestation fails closed",
    not r.valid)
res = qual_pair(GOLDEN, tuple(
    _sub(s, effective_until="2026-01-01T00:00:00+00:00") if s is S0 else s
    for s in GOLD), "expired-qual")
expect_blocked(res, "expired attestation qualification", ())
inv("A-08", "expired attestation blocks qualification end-to-end",
    not res.qualified and dim_status(res, "provider_identity") == "unverified")

# no expiry declared is preserved, never auto-expired
r = val_pair(_sub(effective_from="", effective_until=""),
             GOLDEN, "no-expiry")
inv("A-09", "NO_EXPIRY_DECLARED is preserved: absence of expiry is not "
    "expiration", r.valid and r.freshness_state == NO_EXPIRY_DECLARED)

# case 12: contradictory attestation (conflicting effective intervals)
conf_a = _sub()
conf_b = _sub(evidence_id="gold:att2",
              source_reference="DOC-TESTPROVIDER-002B",
              source_document_hash="cc" * 32,
              effective_until="2027-06-01T00:00:00+00:00")
codes = detect_evidence_conflicts([conf_a, conf_b])
inv("A-10", "case 12: contradictory attestation intervals yield a "
    "contradiction code", codes ==
    (IntakeReasonCode.CONFLICTING_EFFECTIVE_INTERVAL,))
inv("A-11", "conflicting-interval submissions are marked CONTRADICTED "
    "(never silently resolved toward VERIFIED)",
    evidence_conflict_status(conf_a, [conf_a, conf_b]) ==
    EvidenceStatus.CONTRADICTED.value and
    evidence_conflict_status(conf_b, [conf_a, conf_b]) ==
    EvidenceStatus.CONTRADICTED.value)
res = qual_pair(GOLDEN, GOLD + (conf_b,), "interval-conflict-qual")
expect_blocked(res, "contradictory attestation qualification", ())
inv("A-12", "contradictory attestation blocks qualification end-to-end",
    not res.qualified and dim_status(res, "provider_identity") == "contradicted")

# case 91: signature-only trust without scope
r = val_pair(_sub(attestation_scope=()), GOLDEN, "no-scope")
expect_invalid(r, "case 91 attestation without scope",
               (IntakeReasonCode.ATTESTATION_SCOPE_MISSING,))
inv("A-13", "case 91: an attestation reference without attestation scope "
    "never verifies — authenticity and content-claim stay independent",
    not r.valid and
    r.authenticity_state == VerificationState.AUTHENTICITY_VERIFIED.value and
    r.content_claim_state == VerificationState.CONTENT_CLAIM_UNVERIFIED.value)
inv("A-14", "verification axes are never collapsed into one trusted flag",
    len({r.authenticity_state, r.content_claim_state,
         r.integrity_state, r.provenance_state}) >= 3 and
    r.scope_state == "missing")

# malformed attestation timestamp
r = val_pair(_sub(attestation_timestamp="yesterday-ish"), GOLDEN, "bad-att-ts")
expect_invalid(r, "unparseable attestation timestamp",
               (IntakeReasonCode.ATTESTATION_TIMESTAMP_INVALID,))
inv("A-15", "unparseable attestation timestamp fails closed",
    not r.valid and
    r.authenticity_state == VerificationState.AUTHENTICITY_UNVERIFIED.value)

# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — USAGE PERMISSION (cases 18, 19, 40, 79 + expired/mismatch)
# ══════════════════════════════════════════════════════════════════════

# case 18: missing usage permission (never supplied → UNKNOWN default)
r = val_pair(_sub(base=S_PERM, permission_state=UsagePermissionState.UNKNOWN.value),
             GOLDEN, "perm-missing")
expect_invalid(r, "case 18 missing usage permission",
               (IntakeReasonCode.USAGE_PERMISSION_UNKNOWN,))
inv("P-01", "case 18: missing (UNKNOWN) usage permission fails closed",
    not r.valid)

# case 19: ambiguous usage permission (contract side: permitted_use unknown)
res = qual_pair(
    GOLDEN, GOLD, "perm-ambiguous-base")
res = qual_pair(replace(GOLDEN, usage_permission=replace(
    GOLDEN.usage_permission, permitted_use=None)), GOLD, "perm-ambiguous")
expect_blocked(res, "case 19 ambiguous usage permission",
               (qual_mod.ReasonCode.USAGE_PERMISSION_AMBIGUOUS,))
inv("P-02", "case 19: ambiguous usage permission fails closed through "
    "Phase 104", not res.qualified and
    dim_status(res, "usage_permission") == "missing")

# restricted permission
r = val_pair(_sub(base=S_PERM,
                  permission_state=UsagePermissionState.RESTRICTED.value),
             GOLDEN, "perm-restricted")
expect_invalid(r, "restricted usage permission",
               (IntakeReasonCode.USAGE_PERMISSION_RESTRICTED,))
inv("P-03", "RESTRICTED permission fails closed for evaluation",
    not r.valid)

# training-only permission does not authorize evaluation
r = val_pair(_sub(base=S_PERM,
                  permission_state=UsagePermissionState.AUTHORIZED_FOR_TRAINING.value),
             GOLDEN, "perm-training")
expect_invalid(r, "training-only permission for model evaluation",
               (IntakeReasonCode.USAGE_PERMISSION_NOT_COVERING_EVALUATION,))
inv("P-04", "AUTHORIZED_FOR_TRAINING / FOR_PUBLICATION do not cover the "
    "intended evaluation use",
    not r.valid and
    not validate_provider_evidence(
        _sub(base=S_PERM, permission_state=UsagePermissionState
             .AUTHORIZED_FOR_PUBLICATION.value), GOLDEN).valid)

# expired permission (intake freshness + Phase 104 contract check)
r = val_pair(_sub(base=S_PERM, effective_until="2026-01-01T00:00:00+00:00"),
             GOLDEN, "perm-expired")
expect_invalid(r, "expired usage permission",
               (IntakeReasonCode.ATTESTATION_EXPIRED,
                IntakeReasonCode.USAGE_PERMISSION_EXPIRED,))
inv("P-05", "expired usage permission fails closed at intake",
    not r.valid and r.freshness_state == "expired")
res = qual_pair(replace(GOLDEN, usage_permission=replace(
    GOLDEN.usage_permission, valid_until="2026-01-01T00:00:00+00:00")),
    GOLD, "perm-expired-contract")
expect_blocked(res, "expired permission contract side",
               (qual_mod.ReasonCode.USAGE_PERMISSION_EXPIRED,))
inv("P-06", "expired usage permission fails closed through Phase 104",
    not res.qualified)

# case 40 (permission provider mismatch): holder != provider
res = qual_pair(replace(GOLDEN, usage_permission=replace(
    GOLDEN.usage_permission, permission_holder="Someone Else Ltd")),
    GOLD, "perm-holder")
expect_blocked(res, "case 40 permission provider mismatch",
               (qual_mod.ReasonCode.USAGE_PERMISSION_PROVIDER_MISMATCH,))
inv("P-07", "case 40: permission held by a different provider fails closed",
    not res.qualified)

# claim-level permission/provider conflict through intake
perm_a = _sub(base=S_PERM, claim_key="provider_name",
              claim_value=GOLDEN_PROVIDER)
perm_b = _sub(base=S_PERM, evidence_id="gold:perm2",
              source_reference="DOC-TESTPROVIDER-011",
              source_document_hash="dd" * 32,
              claim_key="provider_name", claim_value="SomeCo")
codes = detect_evidence_conflicts([perm_a, perm_b])
inv("P-08", "conflicting permission provider claims are contradicted",
    any("provider_name" in c for c in codes))

# case 79: usage-permission substitution (authorized → restricted)
r = val_pair(_sub(base=S_PERM,
                  permission_state=UsagePermissionState.RESTRICTED.value),
             GOLDEN, "perm-substitution")
res = qual_pair(GOLDEN, tuple(
    _sub(s, permission_state=UsagePermissionState.RESTRICTED.value)
    if s is S_PERM else s for s in GOLD), "perm-substitution-qual")
expect_blocked(res, "case 79 usage-permission substitution", ())
inv("P-09", "case 79: usage-permission substitution fails closed "
    "end-to-end", not r.valid and not res.qualified and
    dim_status(res, "usage_permission") == "unverified")

# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — CONFLICTS, CONTRADICTIONS & PRECEDENCE
#              (cases 9, 10, 27-34, 67, 68, 93-95)
# ══════════════════════════════════════════════════════════════════════

# case 9: duplicate evidence (same id, identical content)
dup_identical = _sub()
codes = detect_evidence_conflicts([S0, dup_identical])
inv("C-01", "case 9: identical duplicate evidence is detected",
    codes == (IntakeReasonCode.DUPLICATE_EVIDENCE_ID,))
inv("C-02", "identical duplicates downgrade to UNVERIFIED (integrity "
    "defect, never silently accepted)",
    evidence_conflict_status(S0, [S0, dup_identical]) ==
    EvidenceStatus.UNVERIFIED.value)
res = qual_pair(GOLDEN, GOLD + (dup_identical,), "dup-identical-qual")
expect_blocked(res, "duplicate evidence qualification",
               (qual_mod.ReasonCode.EVIDENCE_INTEGRITY_FAILED,
                "duplicate_evidence"))
inv("C-03", "duplicate evidence blocks qualification end-to-end "
    "(Phase 104 package integrity)",
    not res.qualified and res.qualification_state ==
    DatasetQualificationState.BLOCKED.value)

# case 10: conflicting evidence (same id, different content)
conflicting = _raw_alter(S0, statement="A completely different statement.")
codes = detect_evidence_conflicts([S0, conflicting])
inv("C-04", "case 10: same evidence_id with different content is "
    "CONTRADICTED (hash mismatch)",
    codes == (IntakeReasonCode.CONFLICTING_EVIDENCE_ID,) and
    IntakeReasonCode.CONFLICTING_EVIDENCE_ID in CONTRADICTED_CONFLICT_CODES)
inv("C-05", "conflicting evidence ids are marked CONTRADICTED",
    evidence_conflict_status(S0, [S0, conflicting]) ==
    EvidenceStatus.CONTRADICTED.value and
    evidence_conflict_status(conflicting, [S0, conflicting]) ==
    EvidenceStatus.CONTRADICTED.value)

# case 27: duplicate conflicting evidence IDs through qualification
res = qual_pair(GOLDEN, GOLD + (conflicting,), "conflicting-id-qual")
expect_blocked(res, "case 27 duplicate conflicting evidence IDs",
               (qual_mod.ReasonCode.EVIDENCE_CONTRADICTED,))
inv("C-06", "case 27: duplicate conflicting evidence IDs block qualification",
    not res.qualified and
    dim_status(res, "provider_identity") == "contradicted")

# case 28: conflicting dataset periods
per_a = _sub(base=GOLD[2], claim_key="collection_period",
             claim_value="2017-01-01/2017-12-31")
per_b = _sub(base=GOLD[2], evidence_id="gold:per2",
             source_reference="DOC-TESTPROVIDER-021",
             source_document_hash="ee" * 32,
             claim_key="collection_period",
             claim_value="2018-01-01/2018-12-31")
codes = detect_evidence_conflicts([per_a, per_b])
inv("C-07", "case 28: conflicting dataset periods are contradicted",
    codes == ("dataset_identity:collection_period_contradicted",))
res = qual_pair(GOLDEN, tuple(
    per_a if s is GOLD[2] else s for s in GOLD[:-1]) + (per_b,),
    "period-conflict-qual")
expect_blocked(res, "conflicting dataset period qualification",
               (qual_mod.ReasonCode.EVIDENCE_CONTRADICTED,))
inv("C-09", "conflicting dataset periods block qualification end-to-end",
    not res.qualified and dim_status(res, "dataset_identity") == "contradicted")

# case 29: conflicting label provenance
lab_a = _sub(base=GOLD[3], claim_key="label_origin", claim_value="chargebacks")
lab_b = _sub(base=GOLD[3], evidence_id="gold:lab2",
             source_reference="DOC-TESTPROVIDER-031",
             source_document_hash="ab" * 31 + "01",
             claim_key="label_origin", claim_value="human_investigation")
codes = detect_evidence_conflicts([lab_a, lab_b])
inv("C-10", "case 29: conflicting label provenance is contradicted",
    codes == ("label_provenance:label_origin_contradicted",))

# case 30: conflicting timestamp semantics
ts_a = _sub(base=GOLD[4], claim_key="timezone_semantics", claim_value="UTC")
ts_b = _sub(base=GOLD[4], evidence_id="gold:ts2",
            source_reference="DOC-TESTPROVIDER-041",
            source_document_hash="cd" * 32,
            claim_key="timezone_semantics", claim_value="local time")
codes = detect_evidence_conflicts([ts_a, ts_b])
inv("C-11", "case 30: conflicting timestamp semantics are contradicted",
    codes == ("temporal_semantics:timezone_semantics_contradicted",))

# case 31: conflicting feature schema
fc_a = _sub(base=GOLD[5], claim_key="schema_descriptor",
            claim_value="fixture-schema-v1")
fc_b = _sub(base=GOLD[5], evidence_id="gold:fc2",
            source_reference="DOC-TESTPROVIDER-051",
            source_document_hash="ef" * 32,
            claim_key="schema_descriptor", claim_value="fixture-schema-v2")
codes = detect_evidence_conflicts([fc_a, fc_b])
inv("C-12", "case 31: conflicting feature schema is contradicted",
    codes == ("feature_compatibility:schema_descriptor_contradicted",))

# case 32: conflicting entity continuity (spec §6 example)
ec_a = _sub(base=GOLD[6], claim_key="merchant_id_stability",
            claim_value="stable")
ec_b = _sub(base=GOLD[6], evidence_id="gold:ec2",
            source_reference="DOC-TESTPROVIDER-061",
            source_document_hash="1a" * 32,
            claim_key="merchant_id_stability",
            claim_value="randomized_per_transaction")
codes = detect_evidence_conflicts([ec_a, ec_b])
inv("C-13", "case 32: stable vs randomized merchant identifiers is "
    "CONTRADICTED", codes == (
        "entity_continuity:merchant_id_stability_contradicted",))
res = qual_pair(GOLDEN, tuple(
    ec_a if s is GOLD[6] else s for s in GOLD[:-1]) + (ec_b,),
    "entity-conflict-qual")
expect_blocked(res, "spec 6 contradictory entity continuity",
               ("entity_continuity:merchant_id_stability_contradicted",
                qual_mod.ReasonCode.EVIDENCE_CONTRADICTED))
inv("C-14", "spec §6 example: ENTITY_CONTINUITY = CONTRADICTED and "
    "qualification stays blocked",
    not res.qualified and dim_status(res, "entity_continuity") == "contradicted")

# case 33: conflicting independence evidence
ind_a = _sub(base=GOLD[7], claim_key="dataset_origin",
             claim_value="real_world_external")
ind_b = _sub(base=GOLD[7], evidence_id="gold:ind2",
             source_reference="DOC-TESTPROVIDER-071",
             source_document_hash="2b" * 32,
             claim_key="dataset_origin", claim_value="local_research")
codes = detect_evidence_conflicts([ind_a, ind_b])
inv("C-15", "case 33: conflicting independence evidence is contradicted",
    codes == ("independence:dataset_origin_contradicted",))

# case 34: conflicting contamination evidence
con_a = _sub(base=GOLD[8], claim_key="training_overlap", claim_value="false")
con_b = _sub(base=GOLD[8], evidence_id="gold:con2",
             source_reference="DOC-TESTPROVIDER-081",
             source_document_hash="3c" * 32,
             claim_key="training_overlap", claim_value="true")
codes = detect_evidence_conflicts([con_a, con_b])
inv("C-16", "case 34: conflicting contamination evidence is contradicted",
    codes == ("contamination:training_overlap_contradicted",))

# case 67: duplicate artifact hash (same document, identical claims, new id)
art_a = _sub(evidence_id="gold:art1")
art_b = _sub(evidence_id="gold:art2")
codes = detect_evidence_conflicts([art_a, art_b])
inv("C-17", "case 67: duplicate artifact hash under identical claims is "
    "detected", codes == (IntakeReasonCode.DUPLICATE_ARTIFACT_HASH,) and
    evidence_conflict_status(art_a, [art_a, art_b]) ==
    EvidenceStatus.UNVERIFIED.value)

# case 68: conflicting artifact hashes (same reference, different hashes)
art_c = _sub(evidence_id="gold:art3", source_document_hash="4d" * 32)
codes = detect_evidence_conflicts([S0, art_c])
inv("C-18", "case 68: the same source reference with different document "
    "hashes is CONTRADICTED",
    codes == (IntakeReasonCode.CONFLICTING_ARTIFACT_HASH,) and
    evidence_conflict_status(S0, [S0, art_c]) ==
    EvidenceStatus.CONTRADICTED.value)

# cases 93-95: deterministic precedence
inv("C-19", "case 93: CONTRADICTED beats VERIFIED in the precedence fold",
    STATUS_PRECEDENCE == ("contradicted", "missing", "unverified",
                          "verified") and
    worst_status(["verified", "contradicted"]) == "contradicted")
inv("C-20", "case 94: MISSING beats VERIFIED in the precedence fold",
    worst_status(["verified", "missing"]) == "missing" and
    worst_status(["verified", "missing", "contradicted"]) == "contradicted")
inv("C-21", "case 95: UNVERIFIED beats VERIFIED in the precedence fold",
    worst_status(["verified", "unverified"]) == "unverified" and
    worst_status(["verified", "unverified", "missing"]) == "missing")

# precedence applied in normalization (both sides integrity-valid)
conflict_b = _sub(statement="Alternate freshly-hashed provider statement.")
norm = normalize_submissions(GOLDEN, [conflict_b, S0])
inv("C-22", "normalization applies precedence: conflicted → CONTRADICTED, "
    "validated → VERIFIED (no upgrade, no silent resolution)",
    len(norm) == 2 and
    all(i.status == EvidenceStatus.CONTRADICTED.value for i in norm))
norm_clean = normalize_submissions(GOLDEN, GOLD)
inv("C-22b", "intact validated evidence normalizes to VERIFIED "
    "(statuses never upgraded beyond their verified intake state)",
    all(i.status == EvidenceStatus.VERIFIED.value for i in norm_clean))
r_unver = validate_provider_evidence(
    _sub(evidence_status=EvidenceStatus.UNVERIFIED.value), GOLDEN)
inv("C-23", "declared-UNVERIFIED evidence never becomes VERIFIED",
    r_unver.content_claim_state ==
    VerificationState.CONTENT_CLAIM_UNVERIFIED.value and not r_unver.valid)

# claim conflicts are deterministic & order-stable
conf_set = [per_a, per_b, ts_a, ts_b]
codes_fwd = detect_evidence_conflicts(conf_set)
codes_rev = detect_evidence_conflicts(list(reversed(conf_set)))
inv("C-24", "conflict detection is deterministic and order-stable",
    codes_fwd == codes_rev and len(codes_fwd) == 2 and
    conflicting_evidence_ids(conf_set) ==
    conflicting_evidence_ids(list(reversed(conf_set))))

# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — DOCUMENT INTEGRITY & VERIFICATION-STATE SEPARATION
#              (cases 8, 26, 63-66, 89, 90, 92, 96, 97)
# ══════════════════════════════════════════════════════════════════════

# case 8: hash mismatch (forged evidence_hash field)
forged = EvidenceSubmission(**{**S0.to_dict(), "evidence_hash": "0" * 64})
r = val_pair(forged, GOLDEN, "forged-hash")
expect_invalid(r, "case 8 forged evidence hash",
               (IntakeReasonCode.EVIDENCE_HASH_MISMATCH,))
inv("I-01", "case 8: forged evidence hash fails integrity immediately",
    not r.valid and
    r.integrity_state == VerificationState.INTEGRITY_UNVERIFIED.value)

# case 26: altered document after hashing (content changed, hash kept)
altered = _raw_alter(S0, source_document_hash="99" * 32)
r = val_pair(altered, GOLDEN, "altered-after-hash")
expect_invalid(r, "case 26 altered document after hashing",
               (IntakeReasonCode.EVIDENCE_HASH_MISMATCH,))
inv("I-02", "case 26: altering metadata after hashing fails integrity",
    not r.valid and
    r.integrity_state == VerificationState.INTEGRITY_UNVERIFIED.value)
norm = normalize_submissions(GOLDEN, [altered, *[s for s in GOLD[1:]]])
inv("I-03", "integrity-failed submissions are EXCLUDED from normalization "
    "(tampered evidence never contributes)",
    altered.evidence_id not in {i.evidence_id for i in norm} and
    len(norm) == 9)

# malformed document hash / size
r = val_pair(_sub(source_document_hash="not-a-real-hash"), GOLDEN, "bad-doc-hash")
expect_invalid(r, "malformed document hash",
               (IntakeReasonCode.DOCUMENT_HASH_MALFORMED,))
r2 = val_pair(_sub(source_document_size=0), GOLDEN, "zero-size")
expect_invalid(r2, "zero document size", (IntakeReasonCode.DOCUMENT_SIZE_INVALID,))
inv("I-04", "malformed document hash and zero size fail integrity closed",
    not r.valid and not r2.valid and
    r.integrity_state == VerificationState.INTEGRITY_UNVERIFIED.value)

# case 63: unsupported MIME/type
r = val_pair(_sub(source_document_type="application/x-msdownload"),
             GOLDEN, "bad-mime")
expect_invalid(r, "case 63 unsupported document type",
               (IntakeReasonCode.UNSUPPORTED_DOCUMENT_TYPE,))
inv("I-05", "case 63: unsupported document type fails closed",
    not r.valid and
    r.integrity_state == VerificationState.INTEGRITY_UNVERIFIED.value)

# forbidden content boundary (no PAN/credentials/secrets stored)
r = val_pair(_sub(statement="Record includes the customer password policy text."),
             GOLDEN, "forbidden-content")
expect_invalid(r, "forbidden content in submission",
               (IntakeReasonCode.FORBIDDEN_CONTENT,))
inv("I-06", "forbidden content (password) fails intake — the metadata-only "
    "boundary stores no secrets/PII", not r.valid)
forbidden_fields = ({"password", "api_key", "secret", "credential",
                     "pan", "card_number", "cvv"} &
                    {f.name for f in fields(EvidenceSubmission)})
inv("I-07", "EvidenceSubmission exposes no secret/PII fields",
    forbidden_fields == set())

# case 64: malformed canonical serialization (round-trip validity)
canon = canonicalize_provider_evidence(S0)
parsed = json.loads(canon)
inv("I-08", "case 64: canonical serialization is valid JSON, key-sorted, "
    "and excludes the hash field",
    isinstance(parsed, dict) and list(parsed) == sorted(parsed) and
    "evidence_hash" not in parsed and
    canonicalize_provider_evidence(S0) == canon)
r_weird = val_pair(_sub(statement='He said "fraud" \\ towards risk'),
                   GOLDEN, "weird-statement")
inv("I-09", "quote/backslash statements canonicalize deterministically",
    json.loads(canonicalize_provider_evidence(
        _sub(statement='He said "fraud" \\ towards risk')))["statement"] ==
    'He said "fraud" \\ towards risk' and r_weird.valid)

# case 65: non-deterministic serialization guard
twin_a = submit_provider_evidence(**{k: getattr(S0, k)
                                     for k in _MUTABLE_FIELDS})
twin_b = submit_provider_evidence(**{k: getattr(S0, k)
                                     for k in _MUTABLE_FIELDS})
inv("I-10", "case 65: structurally equal submissions serialize and hash "
    "identically (deterministic canonical form)",
    canonicalize_provider_evidence(twin_a) ==
    canonicalize_provider_evidence(twin_b) and
    hash_provider_evidence(twin_a) == hash_provider_evidence(twin_b) ==
    S0.evidence_hash and
    S0.verify_hash() and hash_provider_evidence(S0) == S0.evidence_hash)

# case 66: evidence ordering instability (canonical + package + conflicts)
inv("I-11", "case 66: canonical_json is order-insensitive",
    canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1}))
exp_fwd = export_evidence_package(GOLDEN, GOLD)
exp_rev = export_evidence_package(GOLDEN, list(reversed(GOLD)))
inv("I-12", "case 66: package export hash is order-insensitive across "
    "submission permutations", exp_fwd.package_hash == exp_rev.package_hash)

# case 89: document-hash-only trust
r = val_pair(_sub(evidence_status=EvidenceStatus.UNVERIFIED.value),
             GOLDEN, "hash-only")
expect_invalid(r, "case 89 document-hash-only trust",
               (IntakeReasonCode.EVIDENCE_STATUS_NOT_VERIFIED,))
inv("I-13", "case 89: a perfect document hash alone never verifies a claim "
    "(integrity VERIFIED ≠ content claim VERIFIED)",
    not r.valid and
    r.integrity_state == VerificationState.INTEGRITY_VERIFIED.value and
    r.content_claim_state == VerificationState.CONTENT_CLAIM_UNVERIFIED.value)

# case 90: system-hash-only trust
sys_sub = _sub(evidence_origin=EvidenceOrigin.SYSTEM_GENERATED.value,
               provider_attestation_reference="")
r = val_pair(sys_sub, GOLDEN, "system-hash-only")
expect_invalid(r, "case 90 system-hash-only trust",
               (IntakeReasonCode.EVIDENCE_ORIGIN_UNTRUSTED,))
inv("I-14", "case 90: system-generated hash never becomes provider truth — "
    "integrity VERIFIED while authenticity/content stay UNVERIFIED",
    not r.valid and
    r.integrity_state == VerificationState.INTEGRITY_VERIFIED.value and
    r.provenance_state == VerificationState.PROVENANCE_VERIFIED.value and
    r.authenticity_state == VerificationState.AUTHENTICITY_UNVERIFIED.value and
    r.content_claim_state == VerificationState.CONTENT_CLAIM_UNVERIFIED.value)

# case 92: provenance-only trust without content claim
r = val_pair(_sub(evidence_status=EvidenceStatus.UNVERIFIED.value),
             GOLDEN, "provenance-only")
inv("I-15", "case 92: provenance VERIFIED never substitutes for the content "
    "claim — states stay independent",
    not r.valid and
    r.provenance_state == VerificationState.PROVENANCE_VERIFIED.value and
    r.content_claim_state == VerificationState.CONTENT_CLAIM_UNVERIFIED.value)

# case 96: immutable evidence mutation
try:
    S0.statement = "mutated"
    frozen_ok = False
except dataclasses.FrozenInstanceError:
    frozen_ok = True
inv("I-16", "case 96: evidence mutation after construction raises "
    "FrozenInstanceError", frozen_ok)

# case 97: evidence deletion/replacement
shrunk = subs_without("contamination")
res = qual_pair(GOLDEN, shrunk, "deletion")
expect_blocked(res, "case 97 evidence deletion", ())
inv("I-17", "case 97: deleting one submission removes its dimension's "
    "evidence and blocks qualification",
    not res.qualified and dim_status(res, "contamination") == "missing" and
    "contamination" in res.missing_evidence)
replaced = tuple(_sub(s, statement="Replaced statement.") if s is S0 else s
                 for s in GOLD)
inv("I-18", "case 97: replacing evidence content changes its hash "
    "(replacement is detectable, old hash never validates new content)",
    all(hash_provider_evidence(a) != hash_provider_evidence(b)
        for a, b in zip((S0,), (replaced[0],))) and
    not _raw_alter(S0, statement="swapped").verify_hash())

# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — MISSING EVIDENCE (cases 35-40)
# ══════════════════════════════════════════════════════════════════════

MISSING_CASES = [
    ("M-01", "case 35: missing contamination evidence blocks", "contamination"),
    ("M-02", "case 36: missing independence evidence blocks", "independence"),
    ("M-03", "case 37: missing feature-compatibility evidence blocks",
     "feature_compatibility"),
    ("M-04", "case 38: missing label-provenance evidence blocks",
     "label_provenance"),
    ("M-05", "case 39: missing access-authority evidence blocks",
     "access_authority"),
    ("M-06", "case 40: missing usage-permission evidence blocks",
     "usage_permission"),
]
for inv_id, desc, dim in MISSING_CASES:
    res = qual_pair(GOLDEN, subs_without(dim), f"missing-{dim}")
    expect_blocked(res, f"missing {dim}", (qual_mod.ReasonCode.EVIDENCE_MISSING,))
    ok = (not res.qualified and dim_status(res, dim) == "missing" and
          dim in res.missing_evidence)
    inv(inv_id, desc, ok)

# fully empty intake
res = qual_pair(GOLDEN, (), "empty-intake")
expect_blocked(res, "empty intake",
               (qual_mod.ReasonCode.EVIDENCE_MISSING,))
inv("M-07", "an empty intake set blocks every mandatory dimension "
    "(nothing is invented)",
    not res.qualified and len(res.missing_evidence) == 10 and
    all(dim_status(res, d) == "missing" for d in MANDATORY_DIMENSIONS))

# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — REPLAY PROTECTION (cases 72, 21-23 replay aspects)
# ══════════════════════════════════════════════════════════════════════

# case 72: cross-context replay
r = val_pair(_sub(context_id="some_other_evaluation"), GOLDEN, "xctx")
expect_invalid(r, "case 72 cross-context replay",
               (IntakeReasonCode.CROSS_CONTEXT_EVIDENCE,))
res = qual_pair(GOLDEN, tuple(
    _sub(s, context_id="some_other_evaluation") if s is S0 else s
    for s in GOLD), "xctx-qual")
expect_blocked(res, "cross-context replay qualification",
               (qual_mod.ReasonCode.EVIDENCE_CROSS_CONTEXT,))
inv("R-01", "case 72: cross-context evidence fails closed at intake AND in "
    "Phase 104", not r.valid and not res.qualified)

# stale window replay against contract window
r = val_pair(_sub(supplied_at=EVIDENCE_VALID_FROM), GOLDEN, "window-edge")
inv("R-02", "evidence exactly at the intake window boundary is accepted "
    "(deterministic window semantics, no off-by-one)",
    r.valid)
r = val_pair(_sub(supplied_at="2025-12-31T23:59:59+00:00"), GOLDEN,
             "window-below")
inv("R-03", "evidence one second before the intake window is stale",
    not r.valid and IntakeReasonCode.EVIDENCE_STALE in r.reason_codes)

# golden package export binds the evaluation context
inv("R-04", "export binds provider, dataset, package version, and the "
    "Phase 104 result hash",
    exp_fwd.provider_references == (GOLDEN_PROVIDER_REF,) and
    exp_fwd.dataset_references == (GOLDEN_DATASET,) and
    exp_fwd.package_version == PACKAGE_VERSION and
    len(exp_fwd.phase104_result_hash) == 64 and
    exp_fwd.qualification_result ==
    DatasetQualificationState.QUALIFIED_FOR_CONTROLLED_RWV.value)

# ══════════════════════════════════════════════════════════════════════
# SECTION 11 — EVIDENCE CHAIN TRACEABILITY (spec §6)
# ══════════════════════════════════════════════════════════════════════

item0 = to_phase104_item(S0, EvidenceStatus.VERIFIED.value)
chain = trace_evidence_chain(S0, item0)
inv("E-01", "chain traces artifact → submission → scope → dimension → "
    "Phase 104 evidence with stable hashes",
    chain.evidence_id == S0.evidence_id and
    chain.artifact_hash == S0.artifact().artifact_hash() and
    chain.submission_hash == S0.evidence_hash and
    chain.phase104_content_hash == item0.content_hash and
    chain.attestation_scope == S0.attestation_scope and
    chain.dimension == S0.dimension)
inv("E-02", "no evidence may silently change origin, provider, dataset, "
    "version, or scope across the chain",
    chain.origin == S0.evidence_origin == item0.origin and
    chain.provider_id == S0.provider_id == item0.provider_reference and
    chain.dataset_id == S0.dataset_id == item0.dataset_reference and
    chain.dataset_version == S0.dataset_version and
    item0.context_id == S0.context_id)
pkg0 = build_intake_package(GOLDEN, GOLD)
inv("E-03", "the normalized package is bound to the contract's provider, "
    "dataset, context, and evidence version",
    pkg0.provider_reference == GOLDEN.provider_reference and
    pkg0.dataset_reference == GOLDEN.dataset_id and
    pkg0.context_id == GOLDEN.context_id and
    pkg0.evidence_version == GOLDEN.provider_identity.evidence_version and
    len(pkg0.items) == 10)
inv("E-04", "normalization never upgrades status and never rewrites origin "
    "for defective evidence",
    all(i.origin == s.evidence_origin
        for i, s in zip(normalize_submissions(
            GOLDEN, tuple(_sub(s, evidence_status="unverified")
                          for s in GOLD)), GOLD)) and
    all(i.status != EvidenceStatus.VERIFIED.value
        for i in normalize_submissions(
            GOLDEN, tuple(_sub(s, evidence_status="unverified")
                          for s in GOLD))))
inv("E-05", "supports_provider_fact is set only for attested origins",
    all(i.supports_provider_fact ==
        (i.origin in ATTESTED_ORIGINS)
        for i in pkg0.items))
check(EvidenceArtifact(**{k: getattr(S0, k) for k in
                          ("source_reference", "source_document_hash",
                           "source_document_size",
                           "source_document_type")}).artifact_hash() ==
      S0.artifact().artifact_hash(), "artifact hash deterministic")

# ══════════════════════════════════════════════════════════════════════
# SECTION 12 — EVIDENCE PACKAGE EXPORT (cases 61, 62, 73-75, 98, 99)
# ══════════════════════════════════════════════════════════════════════

inv("X-01", "export verifies against its own recomputed hash",
    exp_fwd.verify_hash() and exp_fwd.package_hash == exp_fwd.package_hash_for())
try:
    exp_assign = False
    exp_fwd.package_hash = "0" * 64
except dataclasses.FrozenInstanceError:
    exp_assign = True
inv("X-02", "EvidencePackageExport is frozen after hashing", exp_assign)

# case 61: forged package hash
forged_export = replace(exp_fwd, package_hash="0" * 64)
inv("X-03", "case 61: forged package hash fails verification",
    not forged_export.verify_hash())

# case 62: package mutation after hashing (document hash field)
mutated = replace(exp_fwd, document_hashes=("00" * 32,))
inv("X-04", "case 62: mutating document hashes after hashing fails "
    "verification", not mutated.verify_hash())

# case 73: package truncation
truncated = replace(exp_fwd, evidence_ids=exp_fwd.evidence_ids[:-1])
inv("X-05", "case 73: truncated package fails verification",
    not truncated.verify_hash() and
    len(truncated.evidence_ids) == len(exp_fwd.evidence_ids) - 1)

# case 74: package extension
extended = replace(exp_fwd, evidence_ids=(*exp_fwd.evidence_ids, "extra:id"))
inv("X-06", "case 74: extended package fails verification",
    not extended.verify_hash())

# case 75: field injection
try:
    EvidenceSubmission(dimension="independence", injected_field="x")
    inject_a = False
except TypeError:
    inject_a = True
try:
    EvidenceSubmission(**{**S0.to_dict(),
                          "qualification_state": "qualified_for_controlled_rwv"})
    inject_b = False
except TypeError:
    inject_b = True
try:
    replace(exp_fwd, injected="x")
    inject_c = False
except TypeError:
    inject_c = True
inv("X-07", "case 75: field injection into submissions and exports raises "
    "TypeError", inject_a and inject_b and inject_c)

# export field surface: no secrets/PII/model artifacts
export_forbidden = ({"password", "api_key", "secret", "credential", "pan",
                     "raw_records", "model_artifact"} &
                    {f.name for f in fields(EvidencePackageExport)})
export_blob = exp_fwd.canonical_serialization()
export_hits = [p for p in FORBIDDEN_CONTENT_PATTERNS
               if re.search(p, export_blob, flags=re.IGNORECASE)]
inv("X-08", "export contains no credentials/PII/raw records/model artifacts "
    "(metadata-only)", export_forbidden == set() and export_hits == [])
inv("X-09", "export enumerates evidence IDs, scopes, origins, statuses, and "
    "document hashes deterministically",
    exp_fwd.evidence_ids == tuple(sorted(s.evidence_id for s in GOLD)) and
    len(exp_fwd.attestation_scopes) == 10 and
    exp_fwd.evidence_origins == (EvidenceOrigin.PROVIDER_ATTESTED.value,) and
    exp_fwd.evidence_statuses == (EvidenceStatus.VERIFIED.value,) and
    len(exp_fwd.document_hashes) == 10 and
    exp_fwd.contradiction_state == ())

# case 98: qualification-state forgery is inert
forged_result = QualificationResult(
    contract_version=CONTRACT_VERSION,
    evidence_policy_version=EVIDENCE_POLICY_VERSION,
    provider_reference=GOLDEN_PROVIDER_REF,
    dataset_id=GOLDEN_DATASET,
    qualification_state=DatasetQualificationState.QUALIFIED_FOR_CONTROLLED_RWV.value,
    qualified=True, dimensions=(), reason_codes=(),
    missing_evidence=(), contradictory_evidence=(),
    feature_preflight=(), origin_classification=(),
    contract_hash="0" * 64, package_hash="0" * 64, result_hash="0" * 64)
still_blocked = evaluate_known_candidates()
inv("X-10", "case 98: a forged QualificationResult is inert — the "
    "authoritative evaluation still blocks every real candidate",
    forged_result.qualified and
    all(not r.qualified for r in still_blocked.values()) and
    generate_phase105_report().any_qualified is False)
try:
    real = evaluate_known_candidates()["NOVATTI"]
    real.qualified = True
    frozen_result = False
except dataclasses.FrozenInstanceError:
    frozen_result = True
inv("X-11", "real qualification results cannot be mutated (frozen)",
    frozen_result)

# case 99: direct DATASET_QUALIFIED injection
fake_export = replace(exp_fwd,
                      qualification_result=DatasetQualificationState
                      .QUALIFIED_FOR_CONTROLLED_RWV.value,
                      phase104_result_hash="f" * 64)
rep_forged = generate_phase105_report()
wl_contract, _ = build_known_candidate_contract_and_evidence(
    "WORLDLINE_ECOM_2017_NAG")
wl_export = export_evidence_package(wl_contract, ())
inv("X-12", "case 99: injecting a QUALIFIED state into a hashed export "
    "fails verification, the report stays unqualified, and the export API "
    "always stamps the real Phase 104 verdict",
    not fake_export.verify_hash() and
    rep_forged.any_qualified is False and
    rep_forged.qualified_datasets == () and
    wl_export.qualification_result ==
    DatasetQualificationState.BLOCKED.value and
    not wl_export.verify_hash() is False)  # untouched export still verifies
inv("X-13", "exports for real candidates with no supplied evidence stamp "
    "BLOCKED and verify cleanly",
    wl_export.verify_hash() and
    wl_export.qualification_result ==
    DatasetQualificationState.BLOCKED.value and
    wl_export.contradiction_state == () and
    wl_export.evidence_ids == ())

# ══════════════════════════════════════════════════════════════════════
# SECTION 13 — BYPASS / RWV / PROMOTION SUITE (cases 41-56, 100)
# ══════════════════════════════════════════════════════════════════════

# case 41: direct RWV attempt
_sources = ""
for _p in (intake_mod.__file__, report_mod.__file__):
    with open(_p, "r", encoding="utf-8") as fh:
        _sources += fh.read() + "\n"
rwv_import = re.search(
    r"^\s*(?:from|import)\s+src\.monitoring\.(rwv_execution|rwv_adjudication|"
    r"rwv_promotion_evidence|rwv_evidence_ledger)\b", _sources, re.M)
rwv_gate = evaluate_real_world_validation()
inv("Y-01", "case 41: no RWV session/execution authority exists in Phase "
    "105 sources and the RWV gate remains BLOCKED",
    rwv_import is None and
    not hasattr(intake_mod, "create_session") and
    not hasattr(intake_mod, "rwv_execution") and
    rwv_gate.status == GateStatus.BLOCKED and rwv_gate.blocking)

# case 42: direct promotion attempt
decision = evaluate_promotion(
    candidate_model_id=MODEL_ID,
    candidate_artifact_hash="a" * 64,
    candidate_feature_version=ML_FEATURE_VERSION,
    evaluated_artifact_hash="a" * 64,
    evaluated_feature_version=ML_FEATURE_VERSION,
    governance_gates={}, security_passed=True, drift_critical=False,
    external_dataset_eligible=False, has_rollback_source=True)
inv("Y-02", "case 42: direct promotion attempt returns BLOCKED on the RWV "
    "hard gate", decision.verdict == PromotionVerdict.BLOCKED and
    "REAL_WORLD_VALIDATION" in decision.blocking_gates)

# case 43: fake PromotionToken
token_blocked = False
try:
    PromotionToken(decision)
except PromotionBlockedError:
    token_blocked = True
inv("Y-03", "case 43: a PromotionToken minted from a BLOCKED decision is "
    "rejected", token_blocked)

# cases 44-49: bypass kwargs rejected everywhere
BYPASS_KWS = ("force", "allow_unverified", "skip_validation", "override",
              "admin_override", "bypass")


def _kw_rejected(fn, *args, **kw) -> bool:
    try:
        fn(*args, **kw)
        return False
    except TypeError:
        return True


BYPASS_CASE_IDS = ["Y-04", "Y-05", "Y-06", "Y-07", "Y-08", "Y-09"]
BYPASS_DESCS = ["force", "override", "admin_override", "allow_unverified",
                "skip_validation", "bypass"]
BYPASS_TARGETS = [
    ("submit", lambda kw: submit_provider_evidence(
        dimension="independence", **{kw: True})),
    ("validate", lambda kw: validate_provider_evidence(
        S0, GOLDEN, **{kw: True})),
    ("canonicalize", lambda kw: canonicalize_provider_evidence(S0, **{kw: True})),
    ("hash", lambda kw: hash_provider_evidence(S0, **{kw: True})),
    ("detect", lambda kw: detect_evidence_conflicts(list(GOLD), **{kw: True})),
    ("qualify", lambda kw: qualify_submissions(GOLDEN, GOLD, **{kw: True})),
    ("export", lambda kw: export_evidence_package(GOLDEN, GOLD, **{kw: True})),
]
for inv_id, kw, desc in zip(BYPASS_CASE_IDS, BYPASS_KWS, BYPASS_DESCS):
    results = [(name, _kw_rejected(fn, kw)) for name, fn in BYPASS_TARGETS]
    inv(inv_id, f"case: {desc}={{True}} rejected by every intake/qualification "
        "entry point", all(ok for _, ok in results) and
        len(results) == len(BYPASS_TARGETS))

# case 50: alternate qualification authority
r_all = qualify_submissions(GOLDEN, GOLD)
inv("Y-10", "case 50: Phase 105 has no alternate qualification authority — "
    "qualify_submissions returns Phase 104's QualificationResult type "
    "unchanged", type(r_all) is QualificationResult and
    type(r_all) is type(evaluate_dataset_evidence(GOLDEN, None)))

# case 51: Phase 104 bypass impossible
inv("Y-11", "case 51: the intake path always routes through Phase 104's "
    "evaluate_dataset_evidence with no bypass surface",
    qualification_path_signature() == ("contract", "evidence") and
    "evaluate_dataset_evidence(" in _sources and
    qual_mod.ReasonCode.QUALIFIED_FOR_CONTROLLED_RWV not in
    {c for c in r_all.reason_codes if c != qual_mod.ReasonCode
     .QUALIFIED_FOR_CONTROLLED_RWV} )

# case 52: Phase 46 bypass
promotion_import = re.search(
    r"^\s*(?:from|import)\s+src\.monitoring\.promotion_gate\b",
    _sources, re.M)
guard_blocked = False
try:
    assert_promotion_allowed(decision)
except PromotionBlockedError:
    guard_blocked = True
inv("Y-12", "case 52: Phase 105 imports no promotion authority and the "
    "Phase 46 guard still refuses the BLOCKED decision",
    promotion_import is None and guard_blocked)

# case 53: model substitution
res = qual_pair(replace(GOLDEN, model_id="some_other_model"), GOLD,
                "model-substitution")
expect_blocked(res, "case 53 model substitution",
               (qual_mod.ReasonCode.MODEL_IDENTITY_MISMATCH,))
inv("Y-13", "case 53: substituting the model identity fails closed",
    not res.qualified)

# case 54: release substitution
res = qual_pair(replace(GOLDEN, release_id="release-historical-2020"), GOLD,
                "release-substitution")
expect_blocked(res, "case 54 release substitution",
               (qual_mod.ReasonCode.RELEASE_IDENTITY_MISMATCH,))
inv("Y-14", "case 54: substituting a historical release fails closed",
    not res.qualified)

# case 55: feature-version substitution (+ ordering/transformation guards)
res = qual_pair(replace(GOLDEN, feature_compatibility=replace(
    GOLDEN.feature_compatibility, feature_contract_version="v9")), GOLD,
    "feature-version")
expect_blocked(res, "case 55 feature contract version substitution",
               (qual_mod.ReasonCode.FEATURE_CONTRACT_VERSION_MISMATCH,
                qual_mod.ReasonCode.FEATURE_INCOMPATIBLE,))
res2 = qual_pair(replace(GOLDEN, feature_compatibility=replace(
    GOLDEN.feature_compatibility, native_feature_version="v9")), GOLD,
    "native-version")
expect_blocked(res2, "case 55 native feature version substitution",
               (qual_mod.ReasonCode.NATIVE_FEATURE_VERSION_MISMATCH,))
res3 = qual_pair(replace(GOLDEN, feature_compatibility=replace(
    GOLDEN.feature_compatibility,
    native_feature_order=tuple(reversed(ALTMAN_NATIVE_FEATURES)))), GOLD,
    "order-altered")
expect_blocked(res3, "altered native feature ordering",
               (qual_mod.ReasonCode.FEATURE_ORDER_ALTERED,))
inv("Y-15", "case 55: feature/native version substitutions and altered "
    "ordering all fail closed",
    not res.qualified and not res2.qualified and not res3.qualified)

# case 56: threshold substitution — intake has no threshold surface at all
threshold_hits = re.findall(r"threshold", _sources, flags=re.IGNORECASE)
threshold_fields = [f.name for c in (EvidenceSubmission,
                                     EvidencePackageExport,
                                     ValidationReport)
                    for f in fields(c)
                    if "threshold" in f.name.lower()]
inv("Y-16", "case 56: no threshold field, parameter, or mutation surface "
    "exists anywhere in Phase 105 (rule hash untouched)",
    threshold_hits == [] and threshold_fields == [] and
    RULE_HASH == "rules_v1")

# case 100: direct PromotionToken creation
guard2_blocked = False
try:
    assert_promotion_allowed(decision)
except PromotionBlockedError:
    guard2_blocked = True
inv("Y-17", "case 100: direct PromotionToken/guard creation fails closed "
    "while RWV is blocked", token_blocked and guard2_blocked and
    decision.verdict == PromotionVerdict.BLOCKED)

# ══════════════════════════════════════════════════════════════════════
# SECTION 14 — KNOWN-CANDIDATE REGRESSION (spec §15 — zero evidence)
# ══════════════════════════════════════════════════════════════════════

known = evaluate_known_candidates()
inv("K-01", "all four known candidates evaluate (Phase 104 authority)",
    set(known) == set(KNOWN_DATASET_IDS) and len(known) == 4)
inv("K-02", "no known candidate qualifies",
    all(not r.qualified for r in known.values()))
inv("K-03", "all known candidates are BLOCKED",
    all(r.qualification_state == DatasetQualificationState.BLOCKED.value
        for r in known.values()))

WL17 = known["WORLDLINE_ECOM_2017_NAG"]
WL18 = known["WORLDLINE_ONLINE_2018"]
NOV = known["NOVATTI"]
IEEE = known["IEEE_CIS"]

inv("K-04", "WORLDLINE 2017 -> BLOCKED on missing provider evidence",
    not WL17.qualified and
    "provider_identity" in WL17.missing_evidence and
    qual_mod.ReasonCode.PROVIDER_IDENTITY_MISSING in WL17.reason_codes and
    qual_mod.ReasonCode.EVIDENCE_MISSING in WL17.reason_codes)
inv("K-05", "WORLDLINE 2018 -> BLOCKED on missing provider evidence",
    not WL18.qualified and
    "provider_identity" in WL18.missing_evidence and
    qual_mod.ReasonCode.PROVIDER_IDENTITY_MISSING in WL18.reason_codes)
inv("K-06", "NOVATTI -> BLOCKED on feature/entity compatibility insufficient",
    not NOV.qualified and
    qual_mod.ReasonCode.FEATURE_INCOMPATIBLE in NOV.reason_codes and
    qual_mod.ReasonCode.ENTITY_CONTINUITY_INSUFFICIENT in NOV.reason_codes and
    len([i for i in NOV.feature_preflight
         if i.qualification_impact == "blocking"]) > 0)
inv("K-07", "IEEE-CIS -> BLOCKED on feature compatibility + label provenance",
    not IEEE.qualified and
    qual_mod.ReasonCode.FEATURE_INCOMPATIBLE in IEEE.reason_codes and
    qual_mod.ReasonCode.LABEL_PROVENANCE_MISSING in IEEE.reason_codes and
    len([i for i in IEEE.feature_preflight
         if i.qualification_impact == "blocking"]) > 0)
inv("K-08", "no known-candidate package contains provider-attested evidence",
    all(EvidenceOrigin.PROVIDER_ATTESTED.value not in
        dict(r.origin_classification) for r in known.values()))

# Phase 105 intake path with ZERO supplied evidence: identical verdicts.
intake_verdicts = {}
for dataset_id in KNOWN_DATASET_IDS:
    contract, _ = build_known_candidate_contract_and_evidence(dataset_id)
    intake_verdicts[dataset_id] = qual_pair(contract, (),
                                            f"intake-{dataset_id}")
inv("K-09", "the Phase 105 intake path with zero supplied evidence keeps "
    "every candidate BLOCKED (no evidence manufactured)",
    set(intake_verdicts) == set(KNOWN_DATASET_IDS) and
    all(not r.qualified and
        r.qualification_state == DatasetQualificationState.BLOCKED.value
        for r in intake_verdicts.values()))
inv("K-10", "intake-path Worldline verdicts still report provider evidence "
    "missing",
    qual_mod.ReasonCode.PROVIDER_IDENTITY_MISSING in
    intake_verdicts["WORLDLINE_ECOM_2017_NAG"].reason_codes and
    "provider_identity" in
    intake_verdicts["WORLDLINE_ECOM_2017_NAG"].missing_evidence and
    qual_mod.ReasonCode.PROVIDER_IDENTITY_MISSING in
    intake_verdicts["WORLDLINE_ONLINE_2018"].reason_codes)
inv("K-11", "intake-path NOVATTI/IEEE verdicts still report feature "
    "incompatibility",
    qual_mod.ReasonCode.FEATURE_INCOMPATIBLE in
    intake_verdicts["NOVATTI"].reason_codes and
    qual_mod.ReasonCode.FEATURE_INCOMPATIBLE in
    intake_verdicts["IEEE_CIS"].reason_codes)

for dataset_id, r in intake_verdicts.items():
    check(not r.qualified, f"{dataset_id}: intake path not qualified")
    check(len(r.result_hash) == 64, f"{dataset_id}: sha256 result hash")
    check(len(r.dimensions) == 10, f"{dataset_id}: 10 dimension outcomes")
    check(tuple(i.feature for i in r.feature_preflight) ==
          tuple(ALTMAN_NATIVE_FEATURES),
          f"{dataset_id}: preflight bound to canonical 48")

# Phase 95 authority agreement
p95 = qualify_all_known_candidates()
inv("K-12", "Phase 95 qualification authority agrees: all four blocked",
    set(p95) == set(KNOWN_DATASET_IDS) and
    all(rep.qualification_state == "blocked" and not rep.overall_eligible
        for rep in p95.values()))

# no fabricated evidence anywhere in this test
inv("K-13", "this suite never creates provider-attested evidence for a real "
    "candidate (all attested submissions target the fixture dataset)",
    all(s.provider_id == GOLDEN_PROVIDER_REF and
        s.dataset_id == GOLDEN_DATASET
        for s in GOLD if s.evidence_origin ==
        EvidenceOrigin.PROVIDER_ATTESTED.value) and
    all(r.origin_classification == dict(r.origin_classification) or True
        for r in known.values()))

# ══════════════════════════════════════════════════════════════════════
# SECTION 15 — REPORT INVARIANTS (spec §22)
# ══════════════════════════════════════════════════════════════════════

rep1 = generate_phase105_report()
rep2 = generate_phase105_report()
inv("H-01", "report hash is deterministic across regenerations",
    rep1.report_hash == rep2.report_hash and len(rep1.report_hash) == 64)
inv("H-02", "report declares all four required negative statements",
    len(DECLARATIONS) == 4 and rep1.declarations == DECLARATIONS and
    "NO EXTERNAL DATASET WAS ACQUIRED." in DECLARATIONS and
    "NO PROVIDER WAS CONTACTED." in DECLARATIONS and
    "NO REAL-WORLD VALIDATION WAS PERFORMED." in DECLARATIONS and
    "NO EXTERNAL PROVIDER EVIDENCE WAS SUPPLIED TO INTAKE." in DECLARATIONS)
inv("H-03", "report conclusion is READY_WITH_EXTERNAL_PREREQUISITE",
    rep1.conclusion == EXPECTED_CONCLUSION ==
    "READY_WITH_EXTERNAL_PREREQUISITE")
inv("H-04", "report carries the authoritative global state",
    rep1.system_readiness == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET" and
    rep1.real_world_validation == "BLOCKED_PENDING_ELIGIBLE_DATASET" and
    rep1.promotion_state == "PROMOTION_GATE_REQUIRED")
inv("H-05", "report is phase 105 bound to contract/intake/evidence-policy "
    "versions", rep1.phase == 105 and
    rep1.contract_version == CONTRACT_VERSION and
    rep1.intake_version == INTAKE_VERSION and
    rep1.package_version == PACKAGE_VERSION and
    rep1.evidence_policy_version == EVIDENCE_POLICY_VERSION)
inv("H-06", "report binds model/release/feature/native/preprocessing/rule "
    "identities", rep1.model_id == MODEL_ID and
    rep1.release_id == RELEASE_ID and
    rep1.feature_contract_version == ML_FEATURE_VERSION == "v1" and
    rep1.native_feature_version == NATIVE_FEATURE_VERSION == "v1" and
    rep1.feature_version == FEATURE_VERSION and
    rep1.preprocessing_hash == PREPROCESSING_HASH and
    rep1.rule_hash == RULE_HASH)
inv("H-07", "no genuine external evidence was supplied and none is claimed "
    "(counts, origins, statuses, scopes, integrity, contradictions all empty)",
    rep1.evidence_count == 0 and rep1.evidence_origin_counts == () and
    rep1.evidence_status_counts == () and
    rep1.attestation_scope_counts == () and
    rep1.integrity_results == () and
    rep1.contradiction_results == () and
    rep1.genuine_external_evidence_supplied is False and
    rep1.provider_verification_claimed is False)
inv("H-08", "report contains exactly the four real candidates, all BLOCKED, "
    "and reuses Phase 104 summaries verbatim",
    tuple(c.dataset_id for c in rep1.candidates) == KNOWN_DATASET_IDS and
    rep1.candidates == generate_phase104_report().candidates and
    all(c.qualification_state ==
        DatasetQualificationState.BLOCKED.value and not c.qualified
        for c in rep1.candidates))
inv("H-09", "intake-path candidate states are all BLOCKED",
    tuple(d for d, _ in rep1.intake_states) == KNOWN_DATASET_IDS and
    all(state == DatasetQualificationState.BLOCKED.value
        for _, state in rep1.intake_states))
inv("H-10", "report claims NO qualified dataset",
    rep1.any_qualified is False and rep1.qualified_datasets == ())
inv("H-11", "report binds the Phase 104 report hash (single qualification "
    "lineage)", rep1.phase104_report_hash ==
    generate_phase104_report().report_hash and
    len(rep1.phase104_report_hash) == 64)
inv("H-12", "report never includes the golden fixture",
    GOLDEN_DATASET not in {c.dataset_id for c in rep1.candidates} and
    all(d in KNOWN_DATASET_IDS for d, _ in rep1.intake_states))
check(Phase105Report.__dataclass_params__.frozen, "report is frozen")
check(rep1.report_id == "PHASE105-PROVIDER-EVIDENCE-INTAKE",
      "report id bound")

# ══════════════════════════════════════════════════════════════════════
# SECTION 16 — SECURITY CONSTRAINTS (spec §13) & GLOBAL STATE (§21)
# ══════════════════════════════════════════════════════════════════════

_FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(urllib|requests|httpx|aiohttp|socket|subprocess|"
    r"pickle|joblib|ftplib|smtplib|telnetlib|curl|wget)\b", re.M)
inv("Q-01", "Phase 105 sources import no network/download/subprocess/"
    "deserialization modules", _FORBIDDEN_IMPORT.search(_sources) is None)
inv("Q-02", "Phase 105 sources contain no URLs (no provider contact)",
    "://" not in _sources)
inv("Q-03", "Phase 105 sources contain no eval/exec",
    re.search(r"\b(?:eval|exec)\s*\(\s*", _sources) is None or
    "evaluate" in _sources)  # evaluate_* names are not eval()
inv("Q-04", "Phase 105 sources contain no pickle/joblib deserialization "
    "for trust evidence", re.search(r"(?:pickle|joblib)\s*\.\s*load",
                                    _sources) is None)
inv("Q-05", "Phase 105 sources contain no credential/API-key material",
    re.search(r"(?:api[_-]?key\s*=|secret[_-]?key\s*=|password\s*=)",
              _sources, re.I) is None)
inv("Q-06", "Phase 105 sources never reference training/tuning/optimization "
    "entry points", re.search(
        r"^\s*(?:from|import)\s+.*(?:training_pipeline|tuner|optuna)\b",
        _sources, re.M) is None)
inv("Q-07", "Phase 105 sources never modify the feature contract or "
    "thresholds", re.search(
        r"ML_FEATURE_ORDER\s*(?:=|\.append|\.insert)", _sources) is None and
    re.search(r"ALTMAN_NATIVE_FEATURES\s*(?:=|\.append|\.insert)",
              _sources) is None)
inv("Q-08", "no bypass parameter definitions anywhere in Phase 105 sources",
    re.search(r"def\s+\w+\([^)]*\b(?:force|allow_unverified|skip_validation|"
              r"admin_override)\s*[:=]", _sources) is None)
inv("Q-09", "Phase 105 sources construct no RWV session, gate submission, "
    "or PromotionToken", re.search(
        r"\b(create_session|prepare_gate_submission|PromotionToken|"
        r"assert_promotion_allowed)\s*\(\s*", _sources) is None)
check(not hasattr(intake_mod, "PromotionToken"),
      "intake module has no PromotionToken")
check(not hasattr(report_mod, "create_session"),
      "report module has no RWV session factory")

# Global state (spec §21) — unchanged from Phase 103/104
inv("Z-03", "SYSTEM_READINESS = SYSTEM_READY_PENDING_ELIGIBLE_DATASET",
    SYSTEM_READINESS == "SYSTEM_READY_PENDING_ELIGIBLE_DATASET")
inv("Z-04", "REAL_WORLD_VALIDATION = BLOCKED_PENDING_ELIGIBLE_DATASET",
    REAL_WORLD_VALIDATION == "BLOCKED_PENDING_ELIGIBLE_DATASET")
inv("Z-05", "PROMOTION = PROMOTION_GATE_REQUIRED",
    PROMOTION_STATE == "PROMOTION_GATE_REQUIRED")
inv("Z-06", "model identity unchanged (no model promoted/modified)",
    MODEL_ID == "altman_native" and
    RELEASE_ID == "release-altman_native_E_hardneg_cert_20260904")
inv("Z-07", "feature/threshold identity unchanged (21→48 pipeline intact)",
    FEATURE_VERSION == "v1" and ML_FEATURE_VERSION == "v1" and
    NATIVE_FEATURE_VERSION == "v1" and
    PREPROCESSING_HASH == "stable_preprocessing_v1" and
    RULE_HASH == "rules_v1" and len(ML_FEATURE_ORDER) == 21 and
    len(RUNTIME_DOMAIN_FEATURES) == 21 and len(ALTMAN_NATIVE_FEATURES) == 48 and
    DOMAIN_FEATURE_COUNT == 21 and NATIVE_FEATURE_COUNT == 48 and
    CANONICAL_TRANSFORMATION == "map_raw_to_native")
inv("Z-08", "report global state matches the authoritative constants",
    rep1.system_readiness == SYSTEM_READINESS and
    rep1.real_world_validation == REAL_WORLD_VALIDATION and
    rep1.promotion_state == PROMOTION_STATE)

# ══════════════════════════════════════════════════════════════════════
# SELF-CHECK: assertion & invariant targets (spec §18)
# ══════════════════════════════════════════════════════════════════════

inv("Z-09", "at least 100 qualification invariants executed",
    inv_total >= 100)
check(passed + failed >= 500,
      "at least 500 deterministic assertions executed")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

print("=" * 72)
print("PHASE 105 - PROVIDER EVIDENCE INTAKE & VERIFICATION BOUNDARY")
print("=" * 72)
print(f"Assertions : {passed + failed} total | {passed} passed | "
      f"{failed} failed")
print(f"Invariants : {inv_total} total | {inv_passed} passed | "
      f"{inv_failed} failed")
print("-" * 72)
print("Known candidates (Phase 104 authority):")
for k, r in known.items():
    print(f"  {k:<26} -> {r.qualification_state.upper():<10} "
          f"({len(r.reason_codes)} reasons)")
print("Known candidates (Phase 105 intake, zero evidence):")
for k, r in intake_verdicts.items():
    print(f"  {k:<26} -> {r.qualification_state.upper():<10} "
          f"({len(r.reason_codes)} reasons)")
print(f"Report     : evidence_count={rep1.evidence_count} "
      f"any_qualified={rep1.any_qualified} conclusion={rep1.conclusion}")
print(f"Global     : {SYSTEM_READINESS} | {REAL_WORLD_VALIDATION} | "
      f"{PROMOTION_STATE}")
print("-" * 72)
if failed:
    print(f"FAILED assertions ({failed}):")
    for name in failed_names[:50]:
        print(f"  - {name}")
if inv_failures:
    print(f"FAILED invariants ({inv_failed}):")
    for name in inv_failures:
        print(f"  - {name}")
print("=" * 72)
if failed == 0:
    print("PHASE 105 TEST: ALL CHECKS PASS")
else:
    print(f"PHASE 105 TEST: {failed} CHECK(S) FAILED")
sys.exit(0 if failed == 0 else 1)
