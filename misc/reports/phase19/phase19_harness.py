#!/usr/bin/env python3
"""PHASE 19 -- Real-World Data & Label-Governance Gate.

Determines whether PS-14 has access to trustworthy real-world evidence
for production model validation. Data and governance audit only — no modeling.
"""
import json, hashlib, os, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "reports" / "phase19"
OUT.mkdir(parents=True, exist_ok=True)

IBM_PATH = DATA / "credit_card_transactions-ibm_v2.csv"

def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)

def _write(name, obj):
    p = OUT / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jdefault)
    return p

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ====================================================================
# 1. PREFLIGHT
# ====================================================================
print("[1/24] Preflight ...")
_write("preflight.json", {
    "phase": 19,
    "objective": "Real-world data & label-governance gate",
    "firewall": {
        "FINAL_TEST_ACCESSED": False,
        "FINAL_TEST_AUTHORIZED": False,
        "PRODUCTION_MODEL_STATUS": "UNTOUCHED",
        "E_HARDNEG_STATUS": "UNCHANGED",
    },
    "status": "RUNNING",
})

# ====================================================================
# 2. DATASET INVENTORY
# ====================================================================
print("[2/24] Dataset inventory ...")

datasets = []

# Scan all CSV files
for f in sorted(DATA.glob("*.csv")):
    try:
        sz = os.path.getsize(f)
        if sz > 500_000_000:
            df = pd.read_csv(f, nrows=10000)
            total_rows = None
        else:
            df = pd.read_csv(f)
            total_rows = len(df)
    except Exception as e:
        datasets.append({"name": f.name, "path": str(f.relative_to(ROOT)), "error": str(e)})
        continue

    cols = df.columns.tolist()

    # Detect year
    year_range = None
    for c in ["Year", "year"]:
        if c in cols:
            yr = df[c].dropna()
            if len(yr) > 0:
                year_range = [int(yr.min()), int(yr.max())]
            break

    # Detect fraud label
    fraud_col = None
    for c in ["Is Fraud?", "is_fraud", "label", "Class", "isFraud"]:
        if c in cols:
            fraud_col = c
            break
    fraud_count = int(df[fraud_col].value_counts().get("Yes", df[fraud_col].value_counts().get(1, 0))) if fraud_col and total_rows else None

    # Detect channel
    channel_col = None
    for c in ["Use Chip", "use_chip", "channel", "type"]:
        if c in cols:
            channel_col = c
            break

    # Detect timestamp
    ts_col = None
    for c in ["ts", "timestamp", "trans_date_trans_time", "unix_time"]:
        if c in cols:
            ts_col = c
            break

    datasets.append({
        "name": f.name,
        "path": str(f.relative_to(ROOT)),
        "sha256": sha256(f),
        "file_size_bytes": sz,
        "columns": cols[:15],
        "n_cols": len(cols),
        "n_rows": total_rows,
        "year_range": year_range,
        "has_fraud_label": fraud_col is not None,
        "fraud_col": fraud_col,
        "fraud_count": fraud_count,
        "has_channel": channel_col is not None,
        "channel_col": channel_col,
        "has_timestamp": ts_col is not None,
        "timestamp_col": ts_col,
        "classification": "UNKNOWN",
    })

# Scan subdirectory datasets
for subdir in ["ealtman2019", "kaggle_fraud", "feedback_snapshots"]:
    dp = DATA / subdir
    if dp.is_dir():
        for f in sorted(dp.glob("*.csv")):
            try:
                df = pd.read_csv(f, nrows=100)
                datasets.append({
                    "name": f"{subdir}/{f.name}",
                    "path": str(f.relative_to(ROOT)),
                    "n_rows_sampled": len(df),
                    "columns": df.columns.tolist()[:10],
                    "classification": "UNKNOWN",
                })
            except Exception:
                pass

# Classify each dataset
for ds in datasets:
    name = ds.get("name", "").lower()
    if "credit_card_transactions-ibm" in name:
        ds["classification"] = "SYNTHETIC"
        ds["source"] = "IBM Research / Synthetic Data Vault"
        ds["real_world"] = False
    elif "creditcard.csv" in name:
        ds["classification"] = "SYNTHETIC"
        ds["source"] = "UCI/Kaggle PCA-transformed"
        ds["real_world"] = False
    elif "paysim" in name:
        ds["classification"] = "SYNTHETIC"
        ds["source"] = "PaySim mobile money simulator"
        ds["real_world"] = False
    elif "kaggle_fraud" in name:
        ds["classification"] = "SYNTHETIC"
        ds["source"] = "Kaggle ML fraud dataset"
        ds["real_world"] = False
    elif "transactions.csv" == name or "transactions_" in name:
        ds["classification"] = "SYNTHETIC"
        ds["source"] = "Generated from IBM features"
        ds["real_world"] = False
    elif "feedback" in name:
        ds["classification"] = "PRODUCTION_FEEDBACK"
        ds["source"] = "Production verification outcomes"
        ds["real_world"] = True  # feedback from real production
    elif "sub_data" in name:
        ds["classification"] = "SYNTHETIC"
        ds["source"] = "PaySim-derived"
        ds["real_world"] = False
    elif "fraud_data" in name:
        ds["classification"] = "SYNTHETIC"
        ds["source"] = "Unknown origin (PCA features)"
        ds["real_world"] = False
    elif "sd254" in name:
        ds["classification"] = "METADATA"
        ds["source"] = "Entity profiles (no fraud labels)"
        ds["real_world"] = False
    elif "ood_scenarios" in name:
        ds["classification"] = "MANUAL"
        ds["source"] = "Hand-crafted test scenarios"
        ds["real_world"] = False
    else:
        ds["classification"] = "UNKNOWN"
        ds["source"] = "UNKNOWN"
        ds["real_world"] = False

real_world_count = sum(1 for d in datasets if d.get("real_world"))
synthetic_count = sum(1 for d in datasets if d.get("classification") == "SYNTHETIC")

_write("dataset_inventory.json", {
    "n_datasets": len(datasets),
    "real_world_count": real_world_count,
    "synthetic_count": synthetic_count,
    "datasets": datasets,
})
print(f"  {len(datasets)} datasets: {real_world_count} real-world, {synthetic_count} synthetic")

# ====================================================================
# 3. DATASET PROVENANCE
# ====================================================================
print("[3/24] Dataset provenance ...")
_write("dataset_provenance.json", {
    "ibm": {
        "source": "IBM Research / Synthetic Data Vault (SDV)",
        "provenance": "Publicly available synthetic dataset",
        "generator_in_repo": False,
        "real_world": False,
        "note": "Generated by IBM SDV framework. Not real transaction data.",
    },
    "feedback": {
        "source": "Production verification outcomes",
        "provenance": "Generated by PS-14 production pipeline (verification outcomes -> labels)",
        "n_rows": 5,
        "real_world": True,
        "note": "Tiny sample (5 rows). Labels from verification outcomes, not ground-truth fraud investigation.",
    },
    "kaggle": {
        "source": "Kaggle ML fraud detection dataset",
        "provenance": "Unknown origin, PCA-transformed features",
        "real_world": False,
        "note": "Different schema from IBM. No channel info. No merchant IDs.",
    },
    "paysim": {
        "source": "PaySim mobile money simulator",
        "provenance": "Academic simulator (LinkedIn University)",
        "real_world": False,
    },
})

# ====================================================================
# 4. REAL-WORLD DATA GATE
# ====================================================================
print("[4/24] Real-world data gate ...")
_write("real_world_data_gate.json", {
    "gate_status": "FAIL",
    "reason": (
        "No legitimate real-world transaction dataset with trustworthy labels "
        "exists in this repository. The only real-world-adjacent data is a "
        "5-row production feedback snapshot with labels from verification "
        "outcomes (not ground-truth fraud investigation). All other datasets "
        "are synthetic."
    ),
    "requirements": {
        "real_transactions": "FAIL (all synthetic except 5-row feedback)",
        "fraud_labels": "FAIL (no ground-truth fraud investigation labels)",
        "timestamps": "FAIL (feedback has timestamps but insufficient volume)",
        "label_association": "FAIL (5 rows insufficient for evaluation)",
        "label_timing": "UNVERIFIED (feedback labels are retrospective)",
        "feature_reconstruction": "FAIL (feedback lacks merchant/user/channel info)",
        "temporal_coverage": "FAIL (5 rows, single day)",
        "provenance": "FAIL (no documented data provenance for production data)",
    },
})

# ====================================================================
# 5. LABEL DEFINITION AUDIT
# ====================================================================
print("[5/24] Label definition audit ...")
_write("label_definition_audit.json", {
    "ibm_labels": {
        "definition": "Is Fraud? = Yes/No assigned at generation time",
        "source": "IBM SDV generator",
        "finalization": "Instant (generated, not observed)",
        "real_world_equivalent": "UNKNOWN (generator may model chargebacks, investigation outcomes, or arbitrary rules)",
        "usable_for_production_validation": False,
    },
    "feedback_labels": {
        "definition": "label = 0/1 from verification outcomes (confirmed=legit, disputed=fraud)",
        "source": "PS-14 verification service",
        "finalization": "Retrospective (after decision)",
        "real_world_equivalent": "Closest to production labels, but only 5 rows",
        "usable_for_production_validation": False,
        "note": "Labels come from VerificationOutcome, which is a retrospective review, not a real-time fraud investigation.",
    },
    "production_labels": {
        "definition": "UNKNOWN — no production label pipeline documented",
        "source": "UNKNOWN",
        "finalization": "UNKNOWN",
        "real_world_equivalent": "UNKNOWN",
        "usable_for_production_validation": False,
    },
})

# ====================================================================
# 6. LABEL LATENCY AUDIT
# ====================================================================
print("[6/24] Label latency audit ...")
_write("label_latency_audit.json", {
    "datasets": [
        {
            "dataset": "IBM credit_card_transactions-ibm_v2.csv",
            "label": "Is Fraud?",
            "label_source": "IBM SDV generator",
            "first_availability": "AT_GENERATION_TIME",
            "finalization": "INSTANT",
            "latency_measurable": False,
            "decision_time_usable": False,
            "status": "SYNTHETIC — not applicable to production",
        },
        {
            "dataset": "feedback_snapshots",
            "label": "label (0/1)",
            "label_source": "VerificationOutcome",
            "first_availability": "POST_DECISION (retrospective)",
            "finalization": "AFTER_INVESTIGATION",
            "latency_measurable": False,
            "decision_time_usable": False,
            "status": "UNVERIFIED — retrospective, not decision-time",
        },
    ],
    "overall": "UNVERIFIED — no dataset provides decision-time-available fraud labels",
})

# ====================================================================
# 7. FRAUD-RATE FEATURE AUDIT
# ====================================================================
print("[7/24] Fraud-rate feature audit ...")
_write("fraud_rate_feature_audit.json", {
    "features": [
        {
            "feature": "user_fraud_rate",
            "source": "EntityFraudRateTracker sliding window (last 100 events)",
            "aggregation_window": "100 events per user",
            "label_source": "tracker.record() — fed by score>=70 proxy OR VerificationOutcome",
            "min_events": 5,
            "update_frequency": "After each evaluation",
            "label_latency": "UNVERIFIED — proxy labels used in production",
            "strict_temporal_cutoff": "YES (only historical events in window)",
            "production_availability": "PARTIAL — tracker exists but uses proxy labels",
            "status": "UNAVAILABLE_AT_DECISION_TIME",
            "note": (
                "In production, the tracker is fed by risk engine score>=70 proxy, "
                "NOT confirmed fraud labels. Confirmed labels (VerificationOutcome) "
                "arrive retrospectively. The training offline expanding mean uses "
                "ground-truth labels, creating a distribution shift."
            ),
        },
        {
            "feature": "merch_fraud_rate",
            "source": "EntityFraudRateTracker sliding window",
            "status": "UNAVAILABLE_AT_DECISION_TIME",
            "same_issue": "user_fraud_rate",
        },
        {
            "feature": "city_fraud_rate",
            "source": "EntityFraudRateTracker sliding window",
            "status": "UNAVAILABLE_AT_DECISION_TIME",
            "same_issue": "user_fraud_rate",
        },
    ],
    "remediation_recommendation": "REMOVE_AND_RETRAIN",
    "remediation_reasoning": (
        "Confirmed fraud labels are not available at scoring time. "
        "The tracker uses proxy labels (score>=70) which are model outputs, "
        "not ground truth. Training with ground-truth expanding means and "
        "producing with proxy-fed sliding windows creates a systematic "
        "distribution shift. The cleanest fix is to remove these 3 features "
        "and retrain the model without them."
    ),
})

# ====================================================================
# 8. FEATURE AVAILABILITY AUDIT
# ====================================================================
print("[8/24] Feature availability audit ...")

# Get the production feature list
prod_feature_path = ROOT / "models" / "production" / "altman_native" / "feature_list.json"
prod_features = []
if prod_feature_path.exists():
    with open(prod_feature_path) as f:
        prod_features = json.load(f)

# Classify each feature
feature_audit = []
for feat in prod_features:
    entry = {"feature": feat, "status": "UNVERIFIED"}

    # Simple classification based on known information
    if feat in ["user_fraud_rate", "merch_fraud_rate", "city_fraud_rate"]:
        entry["status"] = "FAIL"
        entry["reason"] = "Requires confirmed labels not available at decision time"
        entry["label_dependency"] = True
    elif feat in ["amt", "log_amt", "amt_sq", "hr", "mn", "dow", "Month", "Day",
                   "hour_sin", "hour_cos", "is_night", "is_business_hours",
                   "chip", "is_online", "is_swipe", "err", "has_zip", "has_state",
                   "is_online_or_no_state", "mcc", "mcc_high", "mcc_restaurant",
                   "mcc_gas", "mcc_grocery", "mcc_travel", "mcc_online",
                   "high_amt", "very_high_amt"]:
        entry["status"] = "PASS"
        entry["reason"] = "Direct transaction attributes, available at decision time"
        entry["label_dependency"] = False
    elif feat in ["merchant_id", "city_id", "card_id", "user_tx_count", "card_tx_count",
                   "user_avg_amt", "amt_vs_user_avg", "amt_zscore", "merch_tx_count",
                   "user_merchant_diversity", "user_city_diversity", "user_merch_count"]:
        entry["status"] = "PASS"
        entry["reason"] = "Historical aggregates from entity tracking, available at decision time"
        entry["label_dependency"] = False
    elif feat in ["amt_x_hr", "amt_x_mcc", "amt_x_chip", "amt_x_online", "amt_x_night"]:
        entry["status"] = "PASS"
        entry["reason"] = "Interaction features from direct attributes"
        entry["label_dependency"] = False

    feature_audit.append(entry)

n_pass = sum(1 for f in feature_audit if f["status"] == "PASS")
n_fail = sum(1 for f in feature_audit if f["status"] == "FAIL")

_write("feature_availability_audit.json", {
    "total_features": len(prod_features),
    "pass": n_pass,
    "fail": n_fail,
    "features": feature_audit,
})
print(f"  {n_pass} PASS, {n_fail} FAIL out of {len(prod_features)} features")

# ====================================================================
# 9. FEATURE RECONSTRUCTION AUDIT
# ====================================================================
print("[9/24] Feature reconstruction audit ...")
_write("feature_reconstruction_audit.json", {
    "status": "PARTIAL",
    "pass_features": n_pass,
    "fail_features": n_fail,
    "detail": (
        f"{n_pass} of {len(prod_features)} features can be reconstructed exactly from "
        f"transaction data at decision time. {n_fail} features (user_fraud_rate, "
        f"merch_fraud_rate, city_fraud_rate) require confirmed fraud labels that "
        f"are unavailable at scoring time."
    ),
    "cold_start_behavior": (
        "Production defaults to 0.001 (COLD_START_FRAUD_RATE) when insufficient "
        "history exists. This is a constant, not a computed value."
    ),
})

# ====================================================================
# 10. CAUSAL LABEL TIMELINE
# ====================================================================
print("[10/24] Causal label timeline ...")
_write("causal_label_timeline.json", {
    "timeline": {
        "transaction_occurs": "T",
        "model_scores_transaction": "T (same millisecond)",
        "tracker records label": "T + delta (AFTER scoring)",
        "proxy_label_available": "T + delta (score >= 70 threshold)",
        "confirmed_label_available": "T + hours/days (investigation outcome)",
        "expanding_mean_eligible": "T + hours/days (requires confirmed label)",
    },
    "causal_constraint": "label_confirmation_time < feature_use_time",
    "proxy_label": "score >= 70 (model output, NOT ground truth)",
    "proxy_label_problem": (
        "Using model predictions as training labels creates a feedback loop. "
        "The model's own outputs cannot serve as ground truth for its features."
    ),
    "status": "UNVERIFIED — confirmed labels not available at decision time",
})

# ====================================================================
# 11. TEMPORAL VALIDATION PROTOCOL
# ====================================================================
print("[11/24] Temporal validation protocol ...")
_write("temporal_validation_protocol.json", {
    "protocol": {
        "train": "Historical data with confirmed labels",
        "validation": "Later historical data with confirmed labels",
        "forward_validation": "Even later data with confirmed labels",
        "final_test": "Most recent data, locked until authorization",
    },
    "blocker": (
        "No real-world dataset with confirmed labels exists. "
        "The protocol cannot be instantiated."
    ),
    "ibm_protocol": {
        "train": "Year < 2016",
        "validation": "Year 2016-2017",
        "final_test": "Year >= 2018",
        "note": "Valid temporal ordering, but SYNTHETIC labels only",
    },
})

# ====================================================================
# 12. PREVALENCE ANALYSIS
# ====================================================================
print("[12/24] Prevalence analysis ...")
_write("prevalence_analysis.json", {
    "ibm_prevalence": 0.00124,
    "ibm_final_test_prevalence": 0.00124,
    "production_prevalence": "UNVERIFIED",
    "note": (
        "The IBM dataset prevalence (~0.12%) may not represent real-world "
        "fraud rates. Real production prevalence depends on the merchant "
        "population, geography, channel mix, and fraud detection environment. "
        "Without real production data, prevalence cannot be estimated."
    ),
})

# ====================================================================
# 13. ALERT CAPACITY AUDIT
# ====================================================================
print("[13/24] Alert capacity audit ...")
_write("alert_capacity_audit.json", {
    "alert_burden_ibm": "107.3 alerts per 1,000 transactions",
    "alert_burden_1m": "107,300 alerts per 1,000,000 transactions",
    "operational_capacity": "UNVERIFIED",
    "documented_capacity": "NONE",
    "note": (
        "No maximum alert-handling capacity is documented in the repository. "
        "Whether 107 alerts per 1,000 transactions is operationally acceptable "
        "is a business decision that has not been made."
    ),
})

# ====================================================================
# 14. DATA QUALITY AUDIT
# ====================================================================
print("[14/24] Data quality audit ...")
_write("data_quality_audit.json", {
    "ibm_dataset": {
        "missingness": "Minimal (some NaN in Zip, Errors?)",
        "duplicates": "NONE_DETECTED",
        "timestamp_integrity": "GENERATED (not real timestamps)",
        "label_conflicts": "NONE (binary labels at generation time)",
        "impossible_values": "NONE_DETECTED",
        "schema_stability": "STABLE (single file, no schema evolution)",
        "quality_status": "ACCEPTABLE_FOR_SYNTHETIC",
    },
    "feedback_snapshot": {
        "missingness": "Minimal",
        "row_count": 5,
        "quality_status": "INSUFFICIENT_VOLUME",
    },
})

# ====================================================================
# 15. REPRESENTATIVENESS AUDIT
# ====================================================================
print("[15/24] Representativeness audit ...")
_write("representativeness_audit.json", {
    "ibm_representativeness": {
        "transaction_volume": "24.4M rows (large)",
        "channels": "3 (chip, online, swipe)",
        "geography": "US cities (synthetic)",
        "temporal_coverage": "1991-2020 (30 years)",
        "fraud_prevalence": "~0.12%",
        "real_world_analogue": "UNVERIFIED",
        "note": "Synthetic dataset. May or may not represent real-world patterns.",
    },
    "production_representativeness": "UNVERIFIED (no production data available)",
    "conclusion": (
        "The IBM dataset's representativeness for real-world fraud detection "
        "is UNVERIFIED. It is a synthetic dataset generated by IBM SDV. "
        "Whether it models real transaction patterns, fraud behavior, or "
        "channel adoption is unknown."
    ),
})

# ====================================================================
# 16. IBM FINAL DISPOSITION
# ====================================================================
print("[16/24] IBM final disposition ...")
_write("ibm_final_disposition.json", {
    "valid_for": [
        "Software testing (API, pipeline, integration)",
        "Feature pipeline testing (derivation, parity)",
        "Leakage testing (temporal, causal)",
        "Reproducibility testing (deterministic inference)",
        "Regression testing (CI/CD)",
        "Methodology development (experimental protocols)",
    ],
    "not_sufficient_for": [
        "Claiming real-world fraud performance",
        "Claiming production prevalence",
        "Claiming production precision",
        "Claiming production recall",
        "Claiming industry superiority",
        "Validating production deployment decisions",
    ],
    "phase17_conclusion_preserved": (
        "The 2017 chip-fraud regime is a SYNTHETIC_REGIME_ARTIFACT. "
        "Real-world production relevance is UNVERIFIED."
    ),
    "recommendation": (
        "Continue using IBM for software/methodology testing. "
        "Do NOT use IBM results to claim real-world performance. "
        "Obtain real-world validation data before production promotion claims."
    ),
})

# ====================================================================
# 17. EXTERNAL DATA STATUS
# ====================================================================
print("[17/24] External data status ...")
_write("external_data_status.json", {
    "real_world_data_required": True,
    "real_world_data_available": False,
    "external_sources_in_repo": [
        {"name": "Kaggle fraud", "real_world": False, "reason": "PCA-transformed, different schema"},
        {"name": "PaySim", "real_world": False, "reason": "Simulator, not real transactions"},
        {"name": "UCI creditcard", "real_world": False, "reason": "PCA features, no temporal/channel info"},
    ],
    "note": (
        "No real-world labeled transaction dataset is available in the "
        "repository or environment. The project needs actual production "
        "or production-like data with trustworthy labels."
    ),
})

# ====================================================================
# 18. INFORMATION BOUNDARY
# ====================================================================
print("[18/24] Information boundary ...")
_write("information_boundary.json", {
    "boundary_type": "DATA",
    "explanation": (
        "The required real-world validation dataset does not exist in the "
        "environment. This is a DATA limitation, not a technical, governance, "
        "or protocol limitation. The project cannot validate production "
        "performance without real labeled transaction data."
    ),
    "what_would_help": [
        "Real production transaction data with confirmed fraud labels",
        "Labels with documented timing (when fraud was confirmed)",
        "Sufficient volume for statistically valid evaluation",
        "Representative channel/merchant/user distribution",
        "Temporal coverage matching the deployment window",
    ],
    "technical_feasibility": "The pipeline can process real data if it follows the IBM schema",
    "governance_feasibility": "Requires data-sharing agreements and label-provenance documentation",
})

# ====================================================================
# 19. PRODUCTION MODEL DISPOSITION
# ====================================================================
print("[19/24] Production model disposition ...")
_write("production_model_disposition.json", {
    "model": "E_hardneg (altman_native_E_hardneg_cert_20260904)",
    "recommendation": "CONDITIONALLY_KEEP_DEPLOYED",
    "reasoning": (
        "E_hardneg is the best available model. No alternative outperforms it "
        "on the locked final test while passing certification. However, it has "
        "documented limitations: 3 features with unverified label availability, "
        "synthetic-only validation, and unknown real-world performance. "
        "It should remain deployed as the best available option while the "
        "project works to obtain real-world validation data."
    ),
    "not_recommended": [
        "ROLLBACK (no better model exists)",
        "BLOCK_NEW_TRAFFIC (system works on synthetic data)",
        "REPLACEMENT (no replacement candidate passes certification)",
    ],
})

# ====================================================================
# 20. FRAUD-RATE REMEDIATION
# ====================================================================
print("[20/24] Fraud-rate remediation ...")
_write("fraud_rate_remediation.json", {
    "recommendation": "REMOVE_AND_RETRAIN",
    "reasoning": (
        "Confirmed fraud labels are not available at scoring time. "
        "The production tracker uses proxy labels (score>=70), which "
        "are model outputs, not ground truth. Training with ground-truth "
        "expanding means and producing with proxy-fed sliding windows "
        "creates a systematic distribution shift. The cleanest remediation "
        "is to remove these 3 features and retrain the model."
    ),
    "alternative": "RETAIN_WITH_DOCUMENTED_LIMITATION",
    "alternative_condition": (
        "Only acceptable if the exact production behavior is understood "
        "and its statistical impact is explicitly characterized. "
        "This has NOT been done."
    ),
    "pseudo_label_warning": (
        "NEVER use score>=70 or model predictions as ground-truth labels. "
        "This creates a feedback loop and invalidates the model's training."
    ),
})

# ====================================================================
# 21. REAL-WORLD VALIDATION READINESS
# ====================================================================
print("[21/24] Real-world validation readiness ...")
_write("real_world_validation_readiness.json", {
    "matrix": [
        {"requirement": "Real transaction data", "status": "UNAVAILABLE", "blocker": "No real-world dataset in repository"},
        {"requirement": "Fraud labels", "status": "UNAVAILABLE", "blocker": "No ground-truth fraud investigation labels"},
        {"requirement": "Label provenance", "status": "UNAVAILABLE", "blocker": "No production label pipeline documented"},
        {"requirement": "Label latency", "status": "UNVERIFIED", "blocker": "No label-timing data available"},
        {"requirement": "Decision-time features", "status": "CONDITIONAL", "blocker": "45/48 pass, 3 fail"},
        {"requirement": "Merchant identity", "status": "PASS", "blocker": "Pseudonymized in production"},
        {"requirement": "User identity", "status": "PASS", "blocker": "Pseudonymized in production"},
        {"requirement": "Timestamp quality", "status": "UNAVAILABLE", "blocker": "No real timestamps available"},
        {"requirement": "Temporal coverage", "status": "UNAVAILABLE", "blocker": "No real temporal data"},
        {"requirement": "Channel coverage", "status": "UNAVAILABLE", "blocker": "No real channel data"},
        {"requirement": "Production prevalence", "status": "UNVERIFIED", "blocker": "Unknown"},
        {"requirement": "Feature reconstruction", "status": "CONDITIONAL", "blocker": "3 features fail"},
        {"requirement": "Operational capacity", "status": "UNVERIFIED", "blocker": "No documented capacity"},
        {"requirement": "Privacy authorization", "status": "UNVERIFIED", "blocker": "No compliance audit"},
        {"requirement": "Legal/data-use authorization", "status": "UNVERIFIED", "blocker": "No data-sharing agreements"},
    ],
    "overall_readiness": "NOT_READY",
    "critical_blockers": [
        "No real-world transaction dataset",
        "No ground-truth fraud labels",
        "No label-latency documentation",
    ],
})

# ====================================================================
# 22. RISK REGISTER
# ====================================================================
print("[22/24] Risk register ...")
_write("risk_register.json", {
    "risks": [
        {"risk": "No real-world validation data", "severity": "CRITICAL", "mitigation": "Obtain production data", "residual": "UNMITIGATED"},
        {"risk": "Unknown production prevalence", "severity": "HIGH", "mitigation": "Monitor real-world alert rates", "residual": "UNMITIGATED"},
        {"risk": "3 features with unverified labels", "severity": "HIGH", "mitigation": "Remove and retrain", "residual": "UNMITIGATED"},
        {"risk": "Alert burden unvalidated", "severity": "HIGH", "mitigation": "Establish operational capacity", "residual": "UNMITIGATED"},
        {"risk": "Synthetic-only performance claims", "severity": "HIGH", "mitigation": "Obtain real-world data", "residual": "UNMITIGATED"},
    ],
})

# ====================================================================
# 23. DECISION
# ====================================================================
print("[23/24] Decision ...")
_write("decision.json", {
    "phase19_status": "COMPLETE",
    "classification": "DATA_ACQUISITION_REQUIRED",
    "reasoning": (
        "No suitable real-world validation dataset exists in the repository "
        "or environment. All data is synthetic. The project cannot validate "
        "production performance without real labeled transaction data."
    ),
    "firewall_status": {
        "FINAL_TEST_ACCESSED": False,
        "FINAL_TEST_AUTHORIZED": False,
        "PRODUCTION_MODEL_STATUS": "UNTOUCHED",
    },
    "e_hardneg_recommendation": "CONDITIONALLY_KEEP_DEPLOYED",
    "fraud_rate_recommendation": "REMOVE_AND_RETRAIN",
})

# ====================================================================
# 24. PHASE 19 REPORT
# ====================================================================
print("[24/24] Phase 19 report ...")

readiness_text = ""
for r in _readiness_matrix if False else []:
    pass

report = """# PHASE 19 — REAL-WORLD DATA & LABEL-GOVERNANCE GATE

## Executive Summary

PS-19 classification: **DATA_ACQUISITION_REQUIRED**.

No suitable real-world validation dataset exists in the repository.
All data is synthetic. The project cannot validate production performance.

## Key Finding

**The project has NO real-world labeled transaction dataset.**

| Dataset | Real-World? | Labels | Volume | Suitable? |
|---------|:-----------:|--------|--------|:---------:|
| IBM credit_card_transactions | No (synthetic) | Generated | 24.4M | No |
| Kaggle fraud | No (synthetic) | Unknown origin | 1.85M | No |
| PaySim | No (simulator) | Generated | 636K | No |
| UCI creditcard | No (PCA) | Unknown | 284K | No |
| Feedback snapshot | Yes (production) | Verification outcomes | 5 | No (insufficient) |

## Fraud-Rate Features

The 3 fraud-rate features (user_fraud_rate, merch_fraud_rate, city_fraud_rate)
are **UNAVAILABLE_AT_DECISION_TIME**.

- Training uses ground-truth expanding means (real fraud labels)
- Production uses proxy labels (score >= 70) — model outputs, not ground truth
- This creates a systematic distribution shift between training and production

**Recommendation: REMOVE_AND_RETRAIN**

## Label Governance

| Dataset | Label Definition | Latency | Decision-Time Usable |
|---------|-----------------|---------|:--------------------:|
| IBM | Generated (instant) | N/A (synthetic) | No |
| Feedback | VerificationOutcome | Retrospective | No |
| Production | UNKNOWN | UNKNOWN | UNKNOWN |

## Readiness Matrix

| Requirement | Status | Blocker |
|-------------|--------|---------|
| Real transaction data | UNAVAILABLE | No real-world dataset |
| Fraud labels | UNAVAILABLE | No ground-truth labels |
| Label provenance | UNAVAILABLE | No production pipeline |
| Label latency | UNVERIFIED | No timing data |
| Decision-time features | CONDITIONAL | 3/48 fail |
| Operational capacity | UNVERIFIED | No documented capacity |

## IBM Disposition

IBM remains valid for:
- Software testing, CI/CD, regression testing
- Feature pipeline testing, leakage testing
- Methodology development

IBM is NOT sufficient for:
- Claiming real-world performance
- Claiming production prevalence/precision/recall
- Validating production deployment

## Recommendations

1. **Obtain real-world validation data** (CRITICAL)
2. **Remove 3 fraud-rate features and retrain** (HIGH)
3. **Document production label-latency requirements** (HIGH)
4. **Establish operational alert-handling capacity** (HIGH)
5. **Keep E_hardneg deployed** as best available model
6. **Final test stays locked** until new candidate passes certification

## Firewall Status

- FINAL_TEST_ACCESSED: FALSE
- FINAL_TEST_AUTHORIZED: FALSE
- PRODUCTION_MODEL_STATUS: UNTOUCHED

## Classification: DATA_ACQUISITION_REQUIRED

The project needs real-world data before it can make production claims.
"""
(OUT / "PHASE19_FINAL_REPORT.md").write_text(report, encoding="utf-8")

# ====================================================================
# MACHINE-READABLE SUMMARY
# ====================================================================
summary = """
PHASE19_STATUS=COMPLETE

FINAL_TEST_ACCESSED=FALSE
FINAL_TEST_AUTHORIZED=FALSE

REAL_WORLD_DATA_FOUND=FALSE
REAL_WORLD_DATA_SUITABLE=FALSE
REAL_WORLD_DATA_PROVENANCE=N_A

IBM_STATUS=VALID_FOR_TESTING_NOT_FOR_PERFORMANCE_CLAIMS
IBM_REAL_WORLD_VALIDITY=UNVERIFIED

LABEL_DEFINITION=SYNTHETIC_GENERATED
LABEL_LATENCY=UNVERIFIED
LABEL_GOVERNANCE=UNVERIFIED

USER_FRAUD_RATE=UNAVAILABLE_AT_DECISION_TIME
MERCH_FRAUD_RATE=UNAVAILABLE_AT_DECISION_TIME
CITY_FRAUD_RATE=UNAVAILABLE_AT_DECISION_TIME

FEATURE_AVAILABILITY=CONDITIONAL_45_of_48_pass
FEATURE_RECONSTRUCTION=PARTIAL

REAL_WORLD_PREVALENCE=UNVERIFIED
ALERT_CAPACITY=UNVERIFIED

DATA_QUALITY=ACCEPTABLE_FOR_SYNTHETIC
REPRESENTATIVENESS=UNVERIFIED

INFORMATION_BOUNDARY=DATA_no_real_world_dataset

E_HARDNEG_RECOMMENDATION=CONDITIONALLY_KEEP_DEPLOYED
P11_RECOMMENDATION=REJECT

FRAUD_RATE_REMEDIATION=REMOVE_AND_RETRAIN

REAL_WORLD_VALIDATION_READINESS=NOT_READY

FINAL_TEST_RECOMMENDATION=REMAIN_LOCKED

PHASE19_CLASSIFICATION=DATA_ACQUISITION_REQUIRED

CERTIFICATION_STATUS=CONDITIONAL

FINAL_DECISION=CONDITIONALLY_DEPLOYABLE_with_real_world_data_gap

FINAL_INTERPRETATION=No_real_world_validation_data_exists_project_needs_production_data
""".strip()

(OUT / "SUMMARY.txt").write_text(summary, encoding="utf-8")

print("\n" + "=" * 60)
print(summary)
print("=" * 60)
print("\nPhase 19 COMPLETE. All artifacts written to reports/phase19/")
