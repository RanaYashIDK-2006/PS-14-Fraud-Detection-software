"""Phase 105: Provider Evidence Intake Report.

Deterministic, metadata-only report over the Phase 105 provider evidence
intake boundary and the Phase 104 qualification results it feeds.

NO EXTERNAL DATASET WAS ACQUIRED.
NO PROVIDER WAS CONTACTED.
NO REAL-WORLD VALIDATION WAS PERFORMED.
NO EXTERNAL PROVIDER EVIDENCE WAS SUPPLIED TO INTAKE.

No genuine provider/institution evidence exists for any candidate
dataset, so this report never claims provider verification occurred.
Candidate states come from Phase 104 (the sole qualification authority)
and from the Phase 105 intake path with zero supplied evidence — both
must agree: all four candidates BLOCKED.

Reuses the authoritative global state from Phase 103 and the
authoritative model/release/feature identities from Phases 94–96 — no
duplicate promotion, RWV, or trust-policy authority is created here.

STATUS: IMPLEMENTED
SYSTEM_READINESS: SYSTEM_READY_PENDING_ELIGIBLE_DATASET
REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET
PROMOTION: PROMOTION_GATE_REQUIRED
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.monitoring.external_dataset_contract import (
    CONTRACT_VERSION,
    EVIDENCE_POLICY_VERSION,
    KNOWN_DATASET_IDS,
    build_known_candidate_contract_and_evidence,
)
from src.monitoring.feature_contract import ML_FEATURE_VERSION
from src.monitoring.phase103_production_readiness_closure import (
    PROMOTION_STATE as GLOBAL_PROMOTION_STATE,
    REAL_WORLD_VALIDATION as GLOBAL_REAL_WORLD_VALIDATION,
    SYSTEM_READINESS as GLOBAL_SYSTEM_READINESS,
)
from src.monitoring.phase104_external_dataset_report import (
    CandidateQualificationSummary,
    generate_phase104_report,
)
from src.monitoring.provider_evidence_intake import (
    INTAKE_VERSION,
    PACKAGE_VERSION,
    qualify_submissions,
)
from src.monitoring.rwv_readiness_audit import MODEL_ID, RELEASE_ID, FEATURE_VERSION
from src.monitoring.rwv_reproducibility import (
    NATIVE_FEATURE_VERSION,
    PREPROCESSING_HASH,
    RULE_HASH,
)

REPORT_ID = "PHASE105-PROVIDER-EVIDENCE-INTAKE"

DECLARATIONS: tuple[str, ...] = (
    "NO EXTERNAL DATASET WAS ACQUIRED.",
    "NO PROVIDER WAS CONTACTED.",
    "NO REAL-WORLD VALIDATION WAS PERFORMED.",
    "NO EXTERNAL PROVIDER EVIDENCE WAS SUPPLIED TO INTAKE.",
)

EXPECTED_CONCLUSION = "READY_WITH_EXTERNAL_PREREQUISITE"


def _stable_hash(d: Any) -> str:
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Phase105Report:
    """Deterministic Phase 105 provider evidence intake report.

    Evidence counts/origins/statuses describe GENUINELY SUPPLIED
    evidence only — the test fixtures are never counted, and
    provider_verification_claimed is False unless actual provider or
    institution evidence was accepted under the defined verification
    conditions (none has been).
    """
    report_id: str
    phase: int
    contract_version: str
    intake_version: str
    package_version: str
    evidence_policy_version: str
    model_id: str
    release_id: str
    feature_contract_version: str
    native_feature_version: str
    feature_version: str
    preprocessing_hash: str
    rule_hash: str
    system_readiness: str
    real_world_validation: str
    promotion_state: str
    evidence_count: int
    evidence_origin_counts: tuple[tuple[str, int], ...]
    evidence_status_counts: tuple[tuple[str, int], ...]
    attestation_scope_counts: tuple[tuple[str, int], ...]
    integrity_results: tuple[tuple[str, int], ...]
    contradiction_results: tuple[str, ...]
    candidates: tuple[CandidateQualificationSummary, ...]
    intake_states: tuple[tuple[str, str], ...]
    phase104_report_hash: str
    any_qualified: bool
    qualified_datasets: tuple[str, ...]
    genuine_external_evidence_supplied: bool
    provider_verification_claimed: bool
    declarations: tuple[str, ...]
    conclusion: str
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["candidates"] = [c.to_dict() for c in self.candidates]
        for key in ("evidence_origin_counts", "evidence_status_counts",
                    "attestation_scope_counts", "integrity_results",
                    "contradiction_results", "intake_states",
                    "qualified_datasets", "declarations"):
            d[key] = [list(x) if isinstance(x, tuple) else x for x in d[key]]
        return d


def generate_phase105_report() -> Phase105Report:
    """Generate the deterministic Phase 105 intake report.

    Pure offline evaluation over metadata — no network, no provider
    contact, no external data, no model artifacts, no RWV, no
    promotion.  Every count reflects genuinely supplied evidence (zero)
    and candidate states are read back from Phase 104 authority plus the
    Phase 105 intake path with zero submissions.
    """
    p104 = generate_phase104_report()

    # Phase 105 intake path with NO supplied evidence: every candidate
    # must remain BLOCKED — intake never manufactures evidence.
    intake_states: list[tuple[str, str]] = []
    for dataset_id in KNOWN_DATASET_IDS:
        contract, _ = build_known_candidate_contract_and_evidence(dataset_id)
        result = qualify_submissions(contract, ())
        intake_states.append((dataset_id, result.qualification_state))

    qualified = tuple(s.dataset_id for s in p104.candidates if s.qualified)

    report = Phase105Report(
        report_id=REPORT_ID,
        phase=105,
        contract_version=CONTRACT_VERSION,
        intake_version=INTAKE_VERSION,
        package_version=PACKAGE_VERSION,
        evidence_policy_version=EVIDENCE_POLICY_VERSION,
        model_id=MODEL_ID,
        release_id=RELEASE_ID,
        feature_contract_version=ML_FEATURE_VERSION,
        native_feature_version=NATIVE_FEATURE_VERSION,
        feature_version=FEATURE_VERSION,
        preprocessing_hash=PREPROCESSING_HASH,
        rule_hash=RULE_HASH,
        system_readiness=GLOBAL_SYSTEM_READINESS,
        real_world_validation=GLOBAL_REAL_WORLD_VALIDATION,
        promotion_state=GLOBAL_PROMOTION_STATE,
        evidence_count=0,
        evidence_origin_counts=(),
        evidence_status_counts=(),
        attestation_scope_counts=(),
        integrity_results=(),
        contradiction_results=(),
        candidates=p104.candidates,
        intake_states=tuple(intake_states),
        phase104_report_hash=p104.report_hash,
        any_qualified=bool(qualified),
        qualified_datasets=qualified,
        genuine_external_evidence_supplied=False,
        provider_verification_claimed=False,
        declarations=DECLARATIONS,
        conclusion=EXPECTED_CONCLUSION,
        report_hash="",
    )
    report_hash = _stable_hash(report.to_dict())
    return Phase105Report(**{**report.__dict__, "report_hash": report_hash})


__all__ = [
    "REPORT_ID", "DECLARATIONS", "EXPECTED_CONCLUSION",
    "Phase105Report", "generate_phase105_report",
]
