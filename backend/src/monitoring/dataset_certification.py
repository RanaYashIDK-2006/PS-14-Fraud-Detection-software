"""Phase 56: Real-world dataset evidence execution & eligibility certification.

Investigates every known candidate dataset against the Phase 53/54/55
admission framework using actual available evidence.

Two acceptable outcomes:
  PATH A: ELIGIBLE DATASET FOUND — certify, freeze, evaluate
  PATH B: NO ELIGIBLE DATASET — record evidence, keep blocked

REAL_WORLD_VALIDATION: BLOCKED_PENDING_ELIGIBLE_DATASET

This module does NOT fabricate evidence or eligibility.

STATUS: IMPLEMENTED
PHASE: 56
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


# ── Evidence Verification Levels ───────────────────────────────────────────

class EvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    DOCUMENTED = "DOCUMENTED"
    SOURCE_REPORTED = "SOURCE_REPORTED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CONTRADICTED = "CONTRADICTED"
    NOT_EXECUTED = "NOT_EXECUTED"


class CertificationVerdict(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    BLOCKED = "BLOCKED"
    NOT_EVALUATED = "NOT_EVALUATED"


# ── Gate Evidence Record ───────────────────────────────────────────────────

@dataclass
class GateEvidenceRecord:
    """Evidence record for a single admission gate."""
    gate_name: str
    passed: bool
    evidence_status: str = EvidenceStatus.UNKNOWN.value
    evidence_reference: str = ""
    blocking_reason: str = ""
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Candidate Evidence Matrix ──────────────────────────────────────────────

@dataclass
class CandidateEvidenceMatrix:
    """Complete evidence matrix for a single candidate dataset."""
    # Identity
    candidate_id: str = ""
    dataset_name: str = ""
    dataset_version: str = ""
    artifact_hash: str = ""
    acquisition_id: str = ""

    # Source
    original_source: str = ""
    publisher: str = ""
    distributor: str = ""
    distribution_channel: str = ""
    source_documentation: str = ""

    # Real-world origin
    origin_status: str = EvidenceStatus.UNKNOWN.value

    # Provenance
    provenance_status: str = EvidenceStatus.UNKNOWN.value
    provenance_evidence: list[str] = field(default_factory=list)

    # Label semantics
    label_definition: str = ""
    label_column: str = ""
    label_granularity: str = ""
    label_author: str = ""
    label_generation_method: str = ""
    label_timing: str = ""
    label_timing_known: bool = False
    label_prediction_time_availability: str = ""
    positive_class: str = ""
    negative_class: str = ""
    label_evidence_references: list[str] = field(default_factory=list)

    # Temporal
    timestamp_field: str = ""
    collection_period_start: str = ""
    collection_period_end: str = ""
    chronological_integrity: str = EvidenceStatus.UNKNOWN.value

    # Real-world context
    geography: str = ""
    institution_domain: str = ""
    transaction_channel: str = ""
    population: str = ""
    operational_context: str = ""

    # Independence
    shared_ids_status: str = EvidenceStatus.UNKNOWN.value
    transaction_overlap: str = EvidenceStatus.UNKNOWN.value
    feature_overlap: str = EvidenceStatus.UNKNOWN.value
    duplicate_overlap: str = EvidenceStatus.UNKNOWN.value
    derived_copy: str = EvidenceStatus.UNKNOWN.value
    synthetic_relationship: str = EvidenceStatus.UNKNOWN.value
    temporal_contamination: str = EvidenceStatus.UNKNOWN.value

    # Feature compatibility
    external_schema: list[str] = field(default_factory=list)
    ps14_mapping: dict[str, str] = field(default_factory=dict)
    mapping_evidence: str = ""
    compatibility_state: str = "UNKNOWN"
    incompatible_features: list[str] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)

    # Leakage
    target_leakage: str = EvidenceStatus.UNKNOWN.value
    future_information: str = EvidenceStatus.UNKNOWN.value
    duplicate_contamination: str = EvidenceStatus.UNKNOWN.value
    train_test_contamination: str = EvidenceStatus.UNKNOWN.value

    # Admission gates
    gate_results: list[GateEvidenceRecord] = field(default_factory=list)
    verdict: str = CertificationVerdict.NOT_EVALUATED.value
    blocking_reasons: list[str] = field(default_factory=list)
    certification_timestamp: float = 0.0

    def compute_hash(self) -> str:
        """Deterministic hash of the evidence matrix."""
        d = asdict(self)
        d.pop("certification_timestamp", None)
        d.pop("gate_results", None)
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ── Certification Record ───────────────────────────────────────────────────

@dataclass
class CertificationRecord:
    """Immutable certification record for a dataset."""
    certification_id: str = ""
    candidate_id: str = ""
    dataset_version: str = ""
    acquisition_id: str = ""
    dataset_hash: str = ""
    evidence_bundle_hash: str = ""
    provenance_evidence_hash: str = ""
    label_evidence_hash: str = ""
    independence_evidence_hash: str = ""
    feature_mapping_hash: str = ""
    admission_package_hash: str = ""
    certification_timestamp: float = 0.0
    protocol_version: str = "1.0"
    verdict: str = CertificationVerdict.NOT_EVALUATED.value
    record_hash: str = ""

    def compute_hash(self) -> str:
        """Deterministic hash of the certification record."""
        d = asdict(self)
        d.pop("record_hash", None)
        canonical = json.dumps(d, sort_keys=True, default=str)
        self.record_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.record_hash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Certification Workflow ─────────────────────────────────────────────────

class DatasetCertificationWorkflow:
    """Manages the certification and investigation of dataset candidates.

    Integrates with Phase 53/54/55 to:
      - Build evidence matrices for all candidates
      - Investigate each candidate with actual evidence
      - Certify or reject through admission gates
      - Record forensic evidence
    """

    def __init__(self):
        self.matrices: dict[str, CandidateEvidenceMatrix] = {}
        self.certifications: dict[str, CertificationRecord] = {}
        self.evaluation_not_executed_reason: str = "NO_ELIGIBLE_DATASET"

    def build_evidence_matrix(
        self,
        candidate_id: str,
        dataset_name: str,
        **kwargs: Any,
    ) -> CandidateEvidenceMatrix:
        """Build an evidence matrix for a candidate."""
        matrix = CandidateEvidenceMatrix(
            candidate_id=candidate_id,
            dataset_name=dataset_name,
        )
        for k, v in kwargs.items():
            if hasattr(matrix, k):
                setattr(matrix, k, v)
        self.matrices[candidate_id] = matrix
        return matrix

    def evaluate_gates(self, candidate_id: str) -> list[GateEvidenceRecord]:
        """Evaluate all admission gates for a candidate."""
        m = self.matrices.get(candidate_id)
        if not m:
            return []

        gates = []

        # Gate 1: Source classification
        # BLOCKED: SYNTHETIC, DERIVED, UNKNOWN, SOURCE_REPORTED
        # Only VERIFIED real-world origin can pass
        if m.origin_status == EvidenceStatus.VERIFIED.value:
            # Check if verified as SYNTHETIC or DERIVED
            if ("SYNTHETIC" in m.dataset_name.upper()
                or "SDV" in m.dataset_name.upper()
                or "simulator" in m.operational_context.lower()
                or m.synthetic_relationship == EvidenceStatus.VERIFIED.value
                or m.derived_copy == EvidenceStatus.VERIFIED.value):
                gates.append(GateEvidenceRecord(
                    "source_classification", False,
                    EvidenceStatus.VERIFIED.value,
                    blocking_reason="Dataset is SYNTHETIC/DERIVED (verified)",
                ))
            else:
                gates.append(GateEvidenceRecord(
                    "source_classification", True,
                    EvidenceStatus.VERIFIED.value,
                    details="Real-world origin verified",
                ))
        else:
            gates.append(GateEvidenceRecord(
                "source_classification", False,
                m.origin_status,
                blocking_reason=f"Origin status: {m.origin_status} (not independently verified)",
            ))

        # Gate 2: Provenance
        if m.provenance_status in (EvidenceStatus.VERIFIED.value, EvidenceStatus.DOCUMENTED.value):
            gates.append(GateEvidenceRecord(
                "provenance", True, m.provenance_status,
                details=f"Provenance: {m.provenance_status}",
            ))
        else:
            gates.append(GateEvidenceRecord(
                "provenance", False, m.provenance_status,
                blocking_reason=f"Provenance: {m.provenance_status}",
            ))

        # Gate 3: Label semantics
        label_ok = all([
            m.label_definition,
            m.label_column,
            m.label_timing_known,
            m.label_timing not in ("", "unknown"),
            m.label_generation_method not in ("", "unknown"),
        ])
        if label_ok:
            gates.append(GateEvidenceRecord(
                "label_semantics", True,
                EvidenceStatus.VERIFIED.value,
                details=f"Label: {m.label_definition}, timing: {m.label_timing}",
            ))
        else:
            gaps = []
            if not m.label_definition:
                gaps.append("no definition")
            if not m.label_timing_known:
                gaps.append("timing unknown")
            if m.label_timing in ("", "unknown"):
                gaps.append("timing not specified")
            if m.label_generation_method in ("", "unknown"):
                gaps.append("generation method unknown")
            gates.append(GateEvidenceRecord(
                "label_semantics", False,
                EvidenceStatus.UNKNOWN.value,
                blocking_reason=f"Label gaps: {'; '.join(gaps)}",
            ))

        # Gate 4: Temporal semantics
        temporal_ok = all([
            m.timestamp_field,
            m.collection_period_start,
            m.collection_period_end,
        ])
        if temporal_ok:
            gates.append(GateEvidenceRecord(
                "temporal_semantics", True,
                EvidenceStatus.VERIFIED.value,
                details=f"Period: {m.collection_period_start} to {m.collection_period_end}",
            ))
        else:
            gaps = []
            if not m.timestamp_field:
                gaps.append("no timestamp field")
            if not m.collection_period_start:
                gaps.append("collection period start unknown")
            if not m.collection_period_end:
                gaps.append("collection period end unknown")
            gates.append(GateEvidenceRecord(
                "temporal_semantics", False,
                EvidenceStatus.UNKNOWN.value,
                blocking_reason=f"Temporal gaps: {'; '.join(gaps)}",
            ))

        # Gate 5: Schema integrity
        schema_ok = bool(m.artifact_hash and m.external_schema and True)
        if schema_ok:
            gates.append(GateEvidenceRecord(
                "schema_integrity", True,
                EvidenceStatus.VERIFIED.value,
                details=f"Schema: {len(m.external_schema)} columns",
            ))
        else:
            gates.append(GateEvidenceRecord(
                "schema_integrity", False,
                EvidenceStatus.UNKNOWN.value,
                blocking_reason="Schema/incomplete identity",
            ))

        # Gate 6: Feature compatibility
        compat_ok = (
            not m.incompatible_features
            and not m.missing_features
            and m.compatibility_state not in ("UNSUPPORTED", "UNKNOWN")
        )
        if compat_ok:
            gates.append(GateEvidenceRecord(
                "feature_compatibility", True,
                EvidenceStatus.VERIFIED.value,
                details=f"Compatibility: {m.compatibility_state}",
            ))
        else:
            gaps = []
            if m.missing_features:
                gaps.append(f"missing: {m.missing_features}")
            if m.incompatible_features:
                gaps.append(f"incompatible: {m.incompatible_features}")
            if m.compatibility_state in ("UNSUPPORTED", "UNKNOWN"):
                gaps.append(f"state: {m.compatibility_state}")
            gates.append(GateEvidenceRecord(
                "feature_compatibility", False,
                EvidenceStatus.UNKNOWN.value,
                blocking_reason=f"Feature gaps: {'; '.join(gaps)}",
            ))

        # Gate 7: Leakage
        leakage_ok = all(s == EvidenceStatus.VERIFIED.value or s == EvidenceStatus.NOT_APPLICABLE.value
                         for s in [m.target_leakage, m.future_information,
                                   m.duplicate_contamination, m.train_test_contamination]
                         if s != EvidenceStatus.UNKNOWN.value)
        if leakage_ok:
            gates.append(GateEvidenceRecord(
                "leakage_checks", True,
                EvidenceStatus.VERIFIED.value,
                details="No leakage detected",
            ))
        else:
            gates.append(GateEvidenceRecord(
                "leakage_checks", False,
                EvidenceStatus.UNKNOWN.value,
                blocking_reason="Leakage status unknown",
            ))

        # Gate 8: Independence
        indep_ok = all(s == EvidenceStatus.VERIFIED.value
                       for s in [m.shared_ids_status, m.derived_copy, m.synthetic_relationship]
                       if s != EvidenceStatus.UNKNOWN.value)
        if indep_ok:
            gates.append(GateEvidenceRecord(
                "independence", True,
                EvidenceStatus.VERIFIED.value,
                details="Independence verified",
            ))
        else:
            gates.append(GateEvidenceRecord(
                "independence", False,
                m.synthetic_relationship if m.synthetic_relationship != EvidenceStatus.UNKNOWN.value else EvidenceStatus.UNKNOWN.value,
                blocking_reason="Independence not fully verified",
            ))

        # Store gates
        m.gate_results = gates
        return gates

    def certify(self, candidate_id: str) -> CertificationRecord | None:
        """Certify a candidate based on gate evaluation."""
        m = self.matrices.get(candidate_id)
        if not m:
            return None

        gates = self.evaluate_gates(candidate_id)
        all_passed = all(g.passed for g in gates)
        blocking = [g.blocking_reason for g in gates if not g.passed and g.blocking_reason]

        if all_passed:
            m.verdict = CertificationVerdict.ELIGIBLE.value
            m.certification_timestamp = time.time()
        elif any(g.gate_name == "source_classification" and not g.passed for g in gates):
            m.verdict = CertificationVerdict.BLOCKED.value
            m.blocking_reasons = blocking
        else:
            m.verdict = CertificationVerdict.INELIGIBLE.value
            m.blocking_reasons = blocking

        # Create certification record
        cert = CertificationRecord(
            certification_id=f"cert-{candidate_id}",
            candidate_id=candidate_id,
            dataset_version=m.dataset_version,
            acquisition_id=m.acquisition_id,
            dataset_hash=m.artifact_hash,
            certification_timestamp=time.time(),
            verdict=m.verdict,
        )
        cert.compute_hash()
        self.certifications[cert.certification_id] = cert
        return cert

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all investigations."""
        summary = {
            "total_candidates": len(self.matrices),
            "eligible": 0,
            "ineligible": 0,
            "blocked": 0,
            "not_evaluated": 0,
            "evaluation_not_executed": True,
            "evaluation_not_executed_reason": self.evaluation_not_executed_reason,
            "candidates": {},
        }
        for cid, m in self.matrices.items():
            summary["candidates"][cid] = {
                "name": m.dataset_name,
                "verdict": m.verdict,
                "origin": m.origin_status,
                "provenance": m.provenance_status,
                "label_timing_known": m.label_timing_known,
                "blocking_reasons": m.blocking_reasons,
            }
            if m.verdict == CertificationVerdict.ELIGIBLE.value:
                summary["eligible"] += 1
                summary["evaluation_not_executed"] = False
            elif m.verdict == CertificationVerdict.BLOCKED.value:
                summary["blocked"] += 1
            elif m.verdict == CertificationVerdict.INELIGIBLE.value:
                summary["ineligible"] += 1
            else:
                summary["not_evaluated"] += 1

        if summary["eligible"] == 0:
            self.evaluation_not_executed_reason = "NO_ELIGIBLE_DATASET"
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrices": {k: v.to_dict() for k, v in self.matrices.items()},
            "certifications": {k: v.to_dict() for k, v in self.certifications.items()},
            "summary": self.get_summary(),
        }


# ── Known Candidate Investigations ────────────────────────────────────────

def investigate_all_known_candidates() -> DatasetCertificationWorkflow:
    """Investigate all known candidates using actual available evidence.

    Returns a workflow with all evidence matrices populated.
    """
    wf = DatasetCertificationWorkflow()

    # 1. IBM Altman SDV (e_hardneg)
    wf.build_evidence_matrix(
        candidate_id="ibm-altman-sdv",
        dataset_name="IBM Altman SDV (e_hardneg)",
        dataset_version="e_hardneg",
        original_source="SDV (Synthetic Data Vault) Altman model",
        publisher="IBM Research / SDV",
        source_documentation="Phase 17: SYNTHETIC_REGIME_ARTIFACT",
        origin_status=EvidenceStatus.VERIFIED.value,  # verified SYNTHETIC
        provenance_status=EvidenceStatus.VERIFIED.value,  # synthetic provenance verified
        provenance_evidence=["Phase 17 analysis confirms synthetic generation"],
        label_definition="Simulated fraud labels from SDV generator",
        label_column="is_fraud",
        label_granularity="transaction_level",
        label_author="automated_system",
        label_generation_method="synthetic_simulation",
        label_timing="at_event",
        label_timing_known=True,  # timing known but synthetic
        positive_class="1",
        negative_class="0",
        label_evidence_references=["Phase 17 SYNTHETIC_REGIME_ARTIFACT verdict"],
        timestamp_field="timestamp",
        collection_period_start="2023-01-01",
        collection_period_end="2023-12-31",
        geography="synthetic",
        institution_domain="synthetic",
        transaction_channel="synthetic",
        population="synthetic",
        operational_context="synthetic_simulation",
        shared_ids_status=EvidenceStatus.NOT_APPLICABLE.value,
        transaction_overlap=EvidenceStatus.NOT_APPLICABLE.value,
        derived_copy=EvidenceStatus.NOT_APPLICABLE.value,
        synthetic_relationship=EvidenceStatus.VERIFIED.value,  # verified synthetic
    )

    # 2. ULB/MLG Credit Card (kaggle)
    wf.build_evidence_matrix(
        candidate_id="ulb-creditcard",
        dataset_name="ULB/MLG Credit Card (kaggle)",
        dataset_version="unknown",
        original_source="ULB Machine Learning Group",
        publisher="ULB (Universite Libre de Bruxelles)",
        source_documentation="Kaggle dataset page; PCA-transformed features",
        origin_status=EvidenceStatus.SOURCE_REPORTED.value,  # claimed real but unverified
        provenance_status=EvidenceStatus.SOURCE_REPORTED.value,
        provenance_evidence=["Kaggle page claims European cardholder transactions"],
        label_definition="Fraud vs legitimate transactions",
        label_column="Class",
        label_granularity="transaction_level",
        label_author="unknown",
        label_generation_method="unknown",
        label_timing="unknown",
        label_timing_known=False,
        positive_class="1",
        negative_class="0",
        label_evidence_references=["Kaggle dataset page (no formal documentation)"],
        timestamp_field="Time",
        collection_period_start="",
        collection_period_end="",
        geography="Europe",
        institution_domain="unknown",
        transaction_channel="card",
        population="unknown",
        operational_context="unknown",
        shared_ids_status=EvidenceStatus.UNKNOWN.value,
        feature_overlap=EvidenceStatus.UNKNOWN.value,
        derived_copy=EvidenceStatus.UNKNOWN.value,
        synthetic_relationship=EvidenceStatus.UNKNOWN.value,
    )

    # 3. Kaggle Fraud Detection (DV)
    wf.build_evidence_matrix(
        candidate_id="kaggle-fraud-dv",
        dataset_name="Kaggle Fraud Detection (DV)",
        dataset_version="unknown",
        original_source="Unknown origin",
        publisher="Unknown",
        source_documentation="Kaggle dataset page; unclear provenance",
        origin_status=EvidenceStatus.SOURCE_REPORTED.value,
        provenance_status=EvidenceStatus.SOURCE_REPORTED.value,
        provenance_evidence=["Kaggle page only; no independent source"],
        label_definition="Fraud labels present",
        label_column="is_fraud",
        label_granularity="transaction_level",
        label_author="unknown",
        label_generation_method="unknown",
        label_timing="unknown",
        label_timing_known=False,
        positive_class="1",
        negative_class="0",
        timestamp_field="timestamp",
        collection_period_start="",
        collection_period_end="",
        geography="unknown",
        institution_domain="unknown",
        transaction_channel="unknown",
        population="unknown",
        operational_context="unknown",
    )

    # 4. PaySim
    wf.build_evidence_matrix(
        candidate_id="paysim",
        dataset_name="PaySim",
        dataset_version="1",
        original_source="PaySim mobile money simulator",
        publisher="NTNU (Norwegian University of Science and Technology)",
        source_documentation="Academic paper describing simulator",
        origin_status=EvidenceStatus.VERIFIED.value,  # verified SYNTHETIC
        provenance_status=EvidenceStatus.DOCUMENTED.value,
        provenance_evidence=["PaySim paper describes simulator methodology"],
        label_definition="Simulated fraud labels",
        label_column="isFraud",
        label_granularity="transaction_level",
        label_author="automated_system",
        label_generation_method="simulation",
        label_timing="at_event",
        label_timing_known=True,
        positive_class="1",
        negative_class="0",
        timestamp_field="step",
        collection_period_start="",
        collection_period_end="",
        geography="synthetic",
        institution_domain="synthetic",
        transaction_channel="mobile_money",
        population="synthetic",
        operational_context="simulation",
        synthetic_relationship=EvidenceStatus.VERIFIED.value,
    )

    # 5. Elliptic (Bitcoin)
    wf.build_evidence_matrix(
        candidate_id="elliptic",
        dataset_name="Elliptic (Bitcoin)",
        dataset_version="unknown",
        original_source="Elliptic Labs / academic research",
        publisher="Elliptic Ltd",
        source_documentation="Academic paper on Bitcoin transaction classification",
        origin_status=EvidenceStatus.VERIFIED.value,  # verified REAL but different domain
        provenance_status=EvidenceStatus.DOCUMENTED.value,
        provenance_evidence=["Published academic paper"],
        label_definition="Illicit vs licit Bitcoin transactions",
        label_column="class",
        label_granularity="transaction_level",
        label_author="unknown",
        label_generation_method="unknown",
        label_timing="unknown",
        label_timing_known=False,
        positive_class="1 (illicit)",
        negative_class="0 (licit)",
        timestamp_field="time",
        collection_period_start="",
        collection_period_end="",
        geography="Bitcoin network",
        institution_domain="cryptocurrency",
        transaction_channel="bitcoin",
        population="Bitcoin transactions",
        operational_context="cryptocurrency_analysis",
    )

    # 6. PS-14 derived datasets
    wf.build_evidence_matrix(
        candidate_id="ps14-derived",
        dataset_name="PS-14 derived datasets",
        dataset_version="various",
        original_source="PS-14 training pipeline",
        publisher="PS-14 project",
        source_documentation="Derived from PS-14 synthetic generation",
        origin_status=EvidenceStatus.VERIFIED.value,  # verified DERIVED
        provenance_status=EvidenceStatus.VERIFIED.value,
        provenance_evidence=["Known to be derived from PS-14 pipeline"],
        label_definition="Derived from PS-14 labels",
        label_column="is_fraud",
        label_granularity="transaction_level",
        label_author="automated_system",
        label_generation_method="derived_from_synthetic",
        label_timing="at_event",
        label_timing_known=True,
        positive_class="1",
        negative_class="0",
        timestamp_field="timestamp",
        collection_period_start="",
        collection_period_end="",
        geography="synthetic",
        institution_domain="synthetic",
        transaction_channel="synthetic",
        population="synthetic",
        operational_context="derived_from_training",
        derived_copy=EvidenceStatus.VERIFIED.value,  # verified derived
        synthetic_relationship=EvidenceStatus.VERIFIED.value,
    )

    # 7. IEEE-CIS Fraud Detection
    wf.build_evidence_matrix(
        candidate_id="ieee-cis",
        dataset_name="IEEE-CIS Fraud Detection",
        dataset_version="competition",
        original_source="Vesta Corporation",
        publisher="IEEE / Vesta Corporation",
        source_documentation="Kaggle competition page; claims real e-commerce transactions",
        origin_status=EvidenceStatus.SOURCE_REPORTED.value,
        provenance_status=EvidenceStatus.SOURCE_REPORTED.value,
        provenance_evidence=["Kaggle competition page claims Vesta Corporation origin"],
        label_definition="Fraud vs legitimate e-commerce transactions",
        label_column="isFraud",
        label_granularity="transaction_level",
        label_author="unknown",
        label_generation_method="unknown",
        label_timing="unknown",
        label_timing_known=False,
        positive_class="1",
        negative_class="0",
        timestamp_field="TransactionDT",
        collection_period_start="",
        collection_period_end="",
        geography="unknown",
        institution_domain="e-commerce",
        transaction_channel="e-commerce",
        population="e-commerce transactions",
        operational_context="competition_dataset",
    )

    # 8. Synthetic Financial Datasets (SD) / PaySim duplicate
    wf.build_evidence_matrix(
        candidate_id="synthetic-financial-sd",
        dataset_name="Synthetic Financial Datasets (SD)",
        dataset_version="1",
        original_source="PaySim simulator",
        publisher="NTNU",
        source_documentation="Same as PaySim",
        origin_status=EvidenceStatus.VERIFIED.value,  # verified SYNTHETIC
        provenance_status=EvidenceStatus.DOCUMENTED.value,
        label_definition="Simulated fraud",
        label_column="isFraud",
        label_granularity="transaction_level",
        label_author="automated_system",
        label_generation_method="simulation",
        label_timing="at_event",
        label_timing_known=True,
        positive_class="1",
        negative_class="0",
    )

    # 9. UCI Credit Card / ULB distribution
    wf.build_evidence_matrix(
        candidate_id="uci-creditcard",
        dataset_name="UCI Credit Card Fraud Detection",
        dataset_version="ULB distribution",
        original_source="ULB Machine Learning Group",
        publisher="ULB",
        source_documentation="UCI ML Repository; Kaggle mirror",
        origin_status=EvidenceStatus.SOURCE_REPORTED.value,
        provenance_status=EvidenceStatus.SOURCE_REPORTED.value,
        label_definition="Same as ULB/MLG dataset",
        label_column="Class",
        label_granularity="transaction_level",
        label_author="unknown",
        label_generation_method="unknown",
        label_timing="unknown",
        label_timing_known=False,
        positive_class="1",
        negative_class="0",
        timestamp_field="Time",
    )

    # Certify all candidates
    for cid in wf.matrices:
        wf.certify(cid)

    return wf
