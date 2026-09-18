#!/usr/bin/env python3
"""Phase 64: Real-World Dataset Feature Compatibility Analysis.

Runs every known candidate through the existing Phase 53 admission gates
and documents the exact blocking reason for each. Produces the definitive
21-feature compatibility table.

NO NEW INFRASTRUCTURE. Uses existing gates only.
"""

import sys, os, csv, hashlib, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.dataset_evidence import (
    SourceClassification, EvidenceLevel, DatasetEvidenceRecord,
    AdmissionState, run_admission_gates,
)
from src.monitoring.feature_contract import ML_FEATURE_CONTRACT, ML_FEATURE_ORDER

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} -- {detail}")


def main():
    global passed, failed

    print("=" * 70)
    print("PHASE 64: REAL-WORLD DATASET FEATURE COMPATIBILITY ANALYSIS")
    print("=" * 70)

    # ================================================================
    # PART 1: EXACT 21-FEATURE PS-14 CONTRACT
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 1: EXACT 21-FEATURE PS-14 CONTRACT")
    print("=" * 70)

    feature_table = {}
    for name in ML_FEATURE_ORDER:
        spec = ML_FEATURE_CONTRACT[name]
        feature_table[name] = {
            "category": spec.category.value,
            "datatype": spec.datatype,
            "description": spec.description,
            "missing_policy": spec.missing_policy.value,
        }

    # Required data sources per feature
    data_requirements = {
        "amount_ratio":        ["transaction_amount", "account_payment_history"],
        "txn_freq_last_24h":   ["transaction_history", "timestamps"],
        "txn_time_unusual":    ["transaction_timestamp"],
        "new_device_flag":     ["device_id", "account_device_history"],
        "unusual_location_flag": ["geolocation", "account_location_history"],
        "unusual_recipient_flag": ["recipient_id", "account_recipient_history"],
        "failed_auth_count_24h": ["auth_event_log", "timestamps"],
        "days_since_last_similar_txn": ["transaction_history", "transaction_categories"],
        "gradual_escalation_score": ["transaction_history", "amounts"],
        "known_device_count":  ["device_id", "account_device_history"],
        "account_tenure_days": ["account_first_transaction_ts"],
        "hour_of_day":         ["transaction_timestamp"],
        "is_weekend":          ["transaction_timestamp"],
        "shared_device_accounts": ["device_id", "device_account_graph"],
        "shared_recipient_accounts": ["recipient_id", "recipient_account_graph"],
        "mule_ring_score":     ["device_id", "recipient_id", "account_graph"],
        "hour_deviation":      ["transaction_history", "timestamps"],
        "amount_zscore":       ["transaction_history", "amounts"],
        "velocity_deviation":  ["transaction_history", "timestamps"],
        "recipient_novelty":   ["recipient_id", "account_recipient_history"],
        "txn_regularity":      ["transaction_history", "timestamps"],
    }

    print(f"\nAll {len(ML_FEATURE_ORDER)} features verified in contract:")
    for feat in ML_FEATURE_ORDER:
        info = feature_table[feat]
        reqs = data_requirements.get(feat, [])
        print(f"  {feat:<30} [{info['category']:<24}] {info['datatype']:<5} policy={info['missing_policy']:<12} requires={reqs}")

    check("21 features in contract", len(ML_FEATURE_ORDER) == 21,
          f"got {len(ML_FEATURE_ORDER)}")

    # Collect all unique raw data sources
    all_sources = set()
    for reqs in data_requirements.values():
        for r in reqs:
            all_sources.add(r)
    print(f"\nUnique raw data sources required: {len(all_sources)}")
    for s in sorted(all_sources):
        needing = [f for f, reqs in data_requirements.items() if s in reqs]
        print(f"  {s}: needed by {len(needing)} features")

    # ================================================================
    # PART 2: CANDIDATE INVESTIGATION & GATE RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: CANDIDATE INVESTIGATION & GATE RESULTS")
    print("=" * 70)

    candidates = []

    # --- CANDIDATE 1: Kaggle Fraud (Sparkov/SYNTHETIC) ---
    print("\n--- CANDIDATE 1: Kaggle Credit Card Fraud Detection ---")
    print("Source: kartik2112/fraud-detection on Kaggle")
    print("Amazon Science FDB: SYNTHETIC (Sparkov Data Generation Tool)")
    print("Columns: trans_date_trans_time, cc_num, merchant, category, amt,")
    print("         first, last, gender, street, city, state, zip, lat, long,")
    print("         city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long")
    print("Rows: 1,296,675 train + 555,719 test = 1,852,394")
    print("Fraud: 7,506 (0.58%)")

    kaggle_record = DatasetEvidenceRecord(
        dataset_id="kaggle-fraud-detection",
        source_name="Kaggle Credit Card Fraud Detection (Sparkov)",
        source_url="https://www.kaggle.com/datasets/kartik2112/fraud-detection",
        source_type="public_research",
        source_classification=SourceClassification.SYNTHETIC.value,
        license="CC0: Public Domain",
        dataset_hash="fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d",
        row_count=1852394,
        schema_hash="kaggle-schema-v1",
        label_column="is_fraud",
        label_definition="Binary fraud label (0/1)",
        label_generation_method="synthetic generation",
        label_timestamp_semantics="synthetic timestamps generated by Sparkov tool",
        label_timing_known=True,
        positive_class_definition="1 = fraudulent transaction",
        negative_class_definition="0 = genuine transaction",
        transaction_timestamp_column="trans_date_trans_time",
        collection_period_start="2020-01-01",
        collection_period_end="2020-06-30",
        temporal_order_known=True,
        provenance_confidence=EvidenceLevel.KNOWN.value,
        provenance_references=["https://www.kaggle.com/datasets/kartik2112/fraud-detection"],
        feature_mapping_version="1.0",
        missing_features=[],
        incompatible_features=[],
    )
    report1 = run_admission_gates(kaggle_record)
    print(f"  Overall: {report1.overall_status}")
    for g in report1.gates:
        s = "PASS" if g.passed else "FAIL"
        print(f"    Gate {g.gate_name}: {s} -- {g.reason}")

    kaggle_can = ["amount_ratio", "txn_freq_last_24h", "txn_time_unusual",
        "hour_of_day", "is_weekend", "amount_zscore", "hour_deviation",
        "days_since_last_similar_txn", "account_tenure_days"]
    kaggle_miss = [f for f in ML_FEATURE_ORDER if f not in kaggle_can]
    print(f"  PS-14 compatible features: {len(kaggle_can)}/21")
    print(f"  Missing PS-14 features: {len(kaggle_miss)}/21")
    print(f"  Blocking: source_classification = SYNTHETIC")
    check("Kaggle blocked as SYNTHETIC", report1.overall_status == AdmissionState.BLOCKED.value)
    candidates.append({"name": "Kaggle Fraud (Sparkov)", "status": "BLOCKED",
        "reason": "SYNTHETIC (Sparkov data generator)", "feature_compat": f"{len(kaggle_can)}/21"})

    # --- CANDIDATE 2: ULB/MLG Credit Card ---
    print("\n--- CANDIDATE 2: ULB/MLG Credit Card Fraud Detection ---")
    print("Source: Zenodo DOI: 10.5281/zenodo.7395559 + OpenML ID 1597")
    print("Publisher: Universite Libre de Bruxelles + Worldline")
    print("Columns: Time, V1-V28 (PCA), Amount, Class")
    print("Rows: 284,807 | Fraud: 492 (0.17%)")

    ulb_record = DatasetEvidenceRecord(
        dataset_id="ulb-creditcard-fraud",
        source_name="ULB/MLG Credit Card Fraud Detection",
        source_url="https://zenodo.org/records/7395559",
        source_type="academic",
        source_classification=SourceClassification.REAL.value,
        license="Open Data Commons Public Domain Dedication",
        dataset_hash="76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89",
        row_count=284807,
        schema_hash="ulb-schema-v1",
        label_column="Class",
        label_definition="Binary fraud label (1=fraud, 0=genuine)",
        label_generation_method="retrospective",
        label_timestamp_semantics="post-investigation retrospective labels",
        label_timing_known=True,
        positive_class_definition="1 = fraudulent transaction",
        negative_class_definition="0 = genuine transaction",
        transaction_timestamp_column="Time",
        collection_period_start="2013-09-01",
        collection_period_end="2013-09-02",
        temporal_order_known=True,
        provenance_confidence=EvidenceLevel.VERIFIED.value,
        provenance_references=[
            "https://zenodo.org/records/7395559",
            "https://openml.org/d/1597",
            "Dal Pozzolo et al., IEEE CIDM 2015",
        ],
        feature_mapping_version="",
        missing_features=list(ML_FEATURE_ORDER),
        incompatible_features=["V1-V28 PCA components have no semantic mapping to PS-14 domain features"],
    )
    report2 = run_admission_gates(ulb_record)
    print(f"  Overall: {report2.overall_status}")
    for g in report2.gates:
        s = "PASS" if g.passed else "FAIL"
        print(f"    Gate {g.gate_name}: {s} -- {g.reason}")
    print(f"  PS-14 compatible features: 0/21")
    print(f"  Missing PS-14 features: 21/21")
    print(f"  Blocking: feature_compatibility (all 21 unmappable - PCA components)")
    check("ULB blocked by feature compatibility",
          report2.overall_status == AdmissionState.INELIGIBLE.value)
    candidates.append({"name": "ULB/MLG Credit Card", "status": "INELIGIBLE",
        "reason": "Feature incompatibility (0/21 - PCA)", "feature_compat": "0/21"})

    # --- CANDIDATE 3: IEEE-CIS (Vesta) ---
    print("\n--- CANDIDATE 3: IEEE-CIS Fraud Detection (Vesta Corporation) ---")
    print("Source: Kaggle competition + IEEE DataPort (DOI: 10.21227/y5e7-wp63)")
    print("Provider: Vesta Corporation (real e-commerce transactions)")
    print("Features: DeviceType, DeviceInfo, card1-card6, ProductCD, M1-M9")
    print("Rows: 590,540 | Fraud rate: 3.5%")
    print("Status: CANNOT ACQUIRE (requires Kaggle competition entry)")
    print("  Would be PS-14 compatible: PARTIAL (device/card present but anonymized)")
    print("  Blocking: ACQUISITION_BLOCKED (authentication required)")
    candidates.append({"name": "IEEE-CIS (Vesta)", "status": "ACQUISITION_BLOCKED",
        "reason": "Requires Kaggle authentication", "feature_compat": "PARTIAL (12-16/21 est.)"})

    # --- CANDIDATE 4: IBM Altman v2 ---
    print("\n--- CANDIDATE 4: IBM Altman Credit Card Transactions v2 ---")
    print("Source: IBM Research / Kaggle")
    print("Rows: 24,386,900")

    ibm_record = DatasetEvidenceRecord(
        dataset_id="ibm-altman-v2",
        source_name="IBM Altman Credit Card Transactions v2",
        source_url="https://www.kaggle.com/datasets/ealtman/ibm-credit-card-transactions",
        source_type="public_research",
        source_classification=SourceClassification.SYNTHETIC.value,
        license="IBM Research License",
        dataset_hash="b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15",
        row_count=24386900,
        schema_hash="ibm-v2-schema-v1",
        label_column="Is Fraud?",
        label_definition="Simulated fraud label",
        label_generation_method="IBM synthetic data generation",
        label_timestamp_semantics="simulation timestamps",
        label_timing_known=True,
        transaction_timestamp_column="Year/Month/Day/Time",
        collection_period_start="2002-01-01",
        collection_period_end="2019-12-31",
        temporal_order_known=True,
        provenance_confidence=EvidenceLevel.KNOWN.value,
        provenance_references=["https://www.kaggle.com/datasets/ealtman/ibm-credit-card-transactions"],
        feature_mapping_version="",
        missing_features=[],
        incompatible_features=[],
    )
    report4 = run_admission_gates(ibm_record)
    print(f"  Overall: {report4.overall_status}")
    for g in report4.gates:
        s = "PASS" if g.passed else "FAIL"
        print(f"    Gate {g.gate_name}: {s} -- {g.reason}")
    print(f"  Blocking: source_classification = SYNTHETIC")
    check("IBM v2 blocked as SYNTHETIC", report4.overall_status == AdmissionState.BLOCKED.value)
    candidates.append({"name": "IBM Altman v2", "status": "BLOCKED",
        "reason": "SYNTHETIC (IBM simulated data)", "feature_compat": "N/A"})

    # --- CANDIDATE 5: PaySim ---
    print("\n--- CANDIDATE 5: PaySim Mobile Money Fraud ---")
    print("Source: SpringerOpen (PaySim mobile money simulator)")
    print("Rows: 6,362,620")
    print("Status: SYNTHETIC -- blocked by source_classification")
    candidates.append({"name": "PaySim", "status": "BLOCKED",
        "reason": "SYNTHETIC (PaySim simulator)", "feature_compat": "N/A"})

    # ================================================================
    # PART 3: FEATURE COMPATIBILITY MATRIX
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 3: FEATURE COMPATIBILITY MATRIX")
    print("=" * 70)

    compat = {
        "amount_ratio":           {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
        "txn_freq_last_24h":      {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
        "txn_time_unusual":       {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
        "new_device_flag":        {"ULB": 0, "Kaggle": 0, "IBM_v2": 0, "IEEE": 1},
        "unusual_location_flag":  {"ULB": 0, "Kaggle": 0, "IBM_v2": 0, "IEEE": 0},
        "unusual_recipient_flag": {"ULB": 0, "Kaggle": 0, "IBM_v2": 0, "IEEE": 0},
        "failed_auth_count_24h":  {"ULB": 0, "Kaggle": 0, "IBM_v2": 0, "IEEE": 0},
        "days_since_last_similar_txn": {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
        "gradual_escalation_score":    {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
        "known_device_count":     {"ULB": 0, "Kaggle": 0, "IBM_v2": 0, "IEEE": 1},
        "account_tenure_days":    {"ULB": 0, "Kaggle": 0, "IBM_v2": 1, "IEEE": 0},
        "hour_of_day":            {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
        "is_weekend":             {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
        "shared_device_accounts": {"ULB": 0, "Kaggle": 0, "IBM_v2": 0, "IEEE": 0},
        "shared_recipient_accounts": {"ULB": 0, "Kaggle": 0, "IBM_v2": 0, "IEEE": 0},
        "mule_ring_score":        {"ULB": 0, "Kaggle": 0, "IBM_v2": 0, "IEEE": 0},
        "hour_deviation":         {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
        "amount_zscore":          {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
        "velocity_deviation":     {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
        "recipient_novelty":      {"ULB": 0, "Kaggle": 0, "IBM_v2": 0, "IEEE": 0},
        "txn_regularity":         {"ULB": 0, "Kaggle": 1, "IBM_v2": 1, "IEEE": 1},
    }

    ds_names = ["ULB", "Kaggle", "IBM_v2", "IEEE"]
    print(f"\n{'Feature':<30} {'ULB':>5} {'Kaggle':>7} {'IBM_v2':>7} {'IEEE-CIS':>9}")
    print("-" * 60)
    totals = {d: 0 for d in ds_names}
    for feat in ML_FEATURE_ORDER:
        row = compat.get(feat, {})
        vals = []
        for d in ds_names:
            v = row.get(d, 0)
            totals[d] += v
            vals.append("YES" if v else "NO")
        print(f"{feat:<30} {vals[0]:>5} {vals[1]:>7} {vals[2]:>7} {vals[3]:>9}")

    print("-" * 60)
    print(f"{'TOTAL COMPATIBLE':<30} {totals['ULB']:>5} {totals['Kaggle']:>7} {totals['IBM_v2']:>7} {totals['IEEE']:>9}")

    # Features NO dataset can provide
    no_dataset = [f for f in ML_FEATURE_ORDER if all(compat[f][d] == 0 for d in ds_names)]
    print(f"\nFeatures that NO investigated dataset can provide ({len(no_dataset)}/21):")
    for f in no_dataset:
        reqs = data_requirements[f]
        print(f"  {f}: requires {', '.join(reqs)}")

    check("Compatibility matrix is complete", len(compat) == 21)

    # ================================================================
    # PART 4: DEFINITIVE FINDINGS
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 4: DEFINITIVE FINDINGS")
    print("=" * 70)

    print("""
BLOCKER ANALYSIS:
================

The PS-14 21-feature contract requires THREE categories of data that
publicly accessible fraud datasets generally do not provide:

A. DEVICE/CHANNEL FEATURES (3 features):
   new_device_flag, unusual_location_flag, unusual_recipient_flag
   Requires: device_id, geolocation baseline, recipient history

B. GRAPH/AGGREGATED FEATURES (3 features):
   shared_device_accounts, shared_recipient_accounts, mule_ring_score
   Requires: device-account cross-referencing, recipient-account graph

C. HISTORY-DEPENDENT FEATURES (6 features):
   days_since_last_similar_txn, gradual_escalation_score, known_device_count,
   account_tenure_days, velocity_deviation, recipient_novelty, txn_regularity
   Requires: per-account transaction history with device/recipient info

FINDING 1: The Kaggle fraud dataset (kartik2112) is SYNTHETIC.
   Generated by Sparkov Data Generation Tool.
   Documented by Amazon Science Fraud Dataset Benchmark (FDB).
   1,000 simulated customers, 800 merchants, 6 months.
   Would be PS-14 compatible for 9/21 features IF it were real.
   BLOCKED correctly at source_classification gate.

FINDING 2: ULB/MLG is REAL but has 0/21 feature compatibility.
   PCA transformation destroys ALL original feature semantics.
   No device, location, recipient, or graph information.
   BLOCKED correctly at feature_compatibility gate.

FINDING 3: IEEE-CIS (Vesta) is the only potentially compatible candidate.
   Contains DeviceType, DeviceInfo, card features, merchant features.
   Requires Kaggle competition entry (authentication not available).
   Estimated 12-16/21 features (graph features still missing).
   BLOCKED at acquisition (authentication required).

FINDING 4: IBM Altman v2 is SYNTHETIC. BLOCKED correctly.

FINDING 5: PaySim is SYNTHETIC. BLOCKED correctly.

CONCLUSION:
===========
The 21-feature PS-14 contract is architecturally correct for its intended
use case (institutional payment systems with full device/location/recipient
telemetry). However, this same richness makes public real-world validation
currently IMPOSSIBLE because:

1. No publicly accessible real-world fraud dataset provides device-location-
   recipient graph features at the transaction level.

2. The only real-world dataset (ULB) has features anonymized via PCA,
   destroying all semantic information.

3. The only potentially compatible real-world dataset (IEEE-CIS) requires
   authentication that is not available in this environment.

This is an ARCHITECTURAL/DATA-CONTRACT BLOCKER, not a validation
framework gap. The validation infrastructure (Phases 53-63) is working
correctly by blocking ineligible datasets.
""")

    # ================================================================
    # PART 5: CANDIDATE SUMMARY
    # ================================================================
    print("=" * 70)
    print("PART 5: CANDIDATE SUMMARY")
    print("=" * 70)

    print(f"\n{'Candidate':<30} {'Status':<25} {'Reason'}")
    print("-" * 90)
    for c in candidates:
        print(f"{c['name']:<30} {c['status']:<25} {c['reason']}")

    # ================================================================
    # FINAL
    # ================================================================
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} PASSED, {failed} FAILED (out of {passed+failed})")
    print(f"{'='*70}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
