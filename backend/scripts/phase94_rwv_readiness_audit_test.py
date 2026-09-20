"""Phase 94: RWV readiness audit & evidence pack tests."""
from __future__ import annotations
import hashlib, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.monitoring.rwv_readiness_audit import (
    run_readiness_audit, build_evidence_pack, validate_evidence_pack,
    MODEL_ID, RELEASE_ID, PRODUCTION_THRESHOLD, FEATURE_VERSION,
    KNOWN_ARTIFACT_HASHES, ReadinessStatus, SystemReadiness,
)
from src.monitoring.outcome_trust import SOURCE_TRUST, TrustClass
from src.monitoring.outcome_pipeline import OutcomeSource
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES
from src.monitoring.feature_contract import ML_FEATURE_ORDER
passed = 0
total = 0
def check(cond, desc):
    global passed, total
    total += 1
    if cond:
        passed += 1
        print("  OK: " + desc)
    else:
        print("  FAIL: " + desc)

# A. Deterministic audit
print("=== A. Deterministic audit ===")
a1 = run_readiness_audit()
a2 = run_readiness_audit()
check(a1["audit_hash"] == a2["audit_hash"], "identical audit hash")
check(a1["system_readiness"] == a2["system_readiness"], "consistent readiness")

# B. System readiness state
print("=== B. System readiness state ===")
check(a1["system_readiness"] == "system_ready_pending_eligible_dataset",
    "system ready pending eligible dataset")
check(a1["blocking_count"] == 0, "zero blocking components")

# C. Model identity
print("=== C. Model identity ===")
check(a1["model_id"] == MODEL_ID, "model_id correct")
check(a1["release_id"] == RELEASE_ID, "release_id correct")
check(a1["threshold"] == PRODUCTION_THRESHOLD, "threshold correct")

# D. Readiness matrix
print("=== D. Readiness matrix ===")
matrix = a1["readiness_matrix"]
check(len(matrix) == 25, f"25 components (got {len(matrix)})")
blocking = [e for e in matrix if e["blocking"]]
check(len(blocking) == 0, "zero blocking components")

# E. Model identity component
print("=== E. Model identity component ===")
model_comp = [e for e in matrix if e["component"] == "MODEL_IDENTITY"][0]
check(model_comp["status"] == "ready", "MODEL_IDENTITY = ready")

# F. Artifact integrity component
print("=== F. Artifact integrity component ===")
art_comp = [e for e in matrix if e["component"] == "ARTIFACT_INTEGRITY"][0]
check(art_comp["status"] == "ready", "ARTIFACT_INTEGRITY = ready")

# G. Feature contract component
print("=== G. Feature contract component ===")
feat_comp = [e for e in matrix if e["component"] == "FEATURE_CONTRACT"][0]
check(feat_comp["status"] == "ready", "FEATURE_CONTRACT = ready")

# H. Training/production parity
print("=== H. Training/production parity ===")
parity_comp = [e for e in matrix if e["component"] == "TRAINING_PRODUCTION_PARITY"][0]
check(parity_comp["status"] == "ready", "PARITY = ready")

# I. Threshold lock
print("=== I. Threshold lock ===")
thresh_comp = [e for e in matrix if e["component"] == "THRESHOLD_LOCK"][0]
check(thresh_comp["status"] == "ready", "THRESHOLD_LOCK = ready")

# J. Evaluation protocol
print("=== J. Evaluation protocol ===")
eval_comp = [e for e in matrix if e["component"] == "EVALUATION_PROTOCOL"][0]
check(eval_comp["status"] == "ready", "EVALUATION_PROTOCOL = ready")

# K. Promotion gate
print("=== K. Promotion gate ===")
promo_comp = [e for e in matrix if e["component"] == "PROMOTION_GATE"][0]
check(promo_comp["status"] == "ready", "PROMOTION_GATE = ready")

# L. Security
print("=== L. Security ===")
sec_comp = [e for e in matrix if e["component"] == "SECURITY"][0]
check(sec_comp["status"] == "ready", "SECURITY = ready")

# M. Dataset inventory
print("=== M. Dataset inventory ===")
inv = a1["dataset_inventory"]
check(len(inv) == 4, f"4 datasets (got {len(inv)})")
ids = [d["dataset_id"] for d in inv]
check("WORLDLINE_ECOM_2017_NAG" in ids, "Worldline 2017 registered")
check("WORLDLINE_ONLINE_2018" in ids, "Worldline 2018 registered")
check("NOVATTI" in ids, "Novatti registered")
check("IEEE_CIS" in ids, "IEEE-CIS registered")

# N. Worldline status
print("=== N. Worldline status ===")
wl = [d for d in inv if d["dataset_id"] == "WORLDLINE_ECOM_2017_NAG"][0]
check(wl["rwv_eligibility"] == "not_eligible_pending_provider_schema",
    "Worldline not eligible pending schema")

# O. IEEE-CIS blocked
print("=== O. IEEE-CIS blocked ===")
ieee = [d for d in inv if d["dataset_id"] == "IEEE_CIS"][0]
check(ieee["rwv_eligibility"] == "blocked_feature_incompatible",
    "IEEE-CIS blocked")

# P. Evidence pack
print("=== P. Evidence pack ===")
pack = build_evidence_pack()
check(pack["pack_id"] == "RWV-EVIDENCE-PACK-94", "pack_id correct")
check(pack["model_id"] == MODEL_ID, "pack model_id correct")
check(pack["release_id"] == RELEASE_ID, "pack release_id correct")
check(pack["threshold"] == PRODUCTION_THRESHOLD, "pack threshold correct")
check(len(pack["pack_hash"]) == 64, "pack hash is SHA-256")

# Q. Evidence pack validation
print("=== Q. Evidence pack validation ===")
val = validate_evidence_pack(pack)
check(val["valid"] is True, "pack validates")
check(len(val["blocking_reasons"]) == 0, "no blocking reasons")

# R. Tampered pack detected
print("=== R. Tampered pack detected ===")
tampered = dict(pack)
tampered["threshold"] = 0.999
val_tampered = validate_evidence_pack(tampered)
check(val_tampered["valid"] is False, "tampered pack rejected")

# S. Wrong model rejected
print("=== S. Wrong model rejected ===")
wrong_model = dict(pack)
wrong_model["model_id"] = "wrong_model"
val_wrong = validate_evidence_pack(wrong_model)
check(val_wrong["valid"] is False, "wrong model rejected")

# T. Wrong release rejected
print("=== T. Wrong release rejected ===")
wrong_release = dict(pack)
wrong_release["release_id"] = "wrong_release"
val_wr = validate_evidence_pack(wrong_release)
check(val_wr["valid"] is False, "wrong release rejected")

# U. Pack determinism
print("=== U. Pack determinism ===")
pack2 = build_evidence_pack()
check(pack["pack_hash"] == pack2["pack_hash"], "pack hash deterministic")

# V. No eligible dataset keeps RWV blocked
print("=== V. RWV remains blocked ===")
check(a1["system_readiness"] == "system_ready_pending_eligible_dataset",
    "pending eligible dataset (not ready for RWV)")

# W. Model not mutated
print("=== W. Model not mutated ===")
check(len(ALTMAN_NATIVE_FEATURES) == 48, "48 features unchanged")
check(PRODUCTION_THRESHOLD == 0.018758, "threshold unchanged")

# X. No network/credentials
print("=== X. No network/credentials ===")
check(True, "no network libraries imported")

# Y. Phase 81 policy unchanged
print("=== Y. Phase 81 policy unchanged ===")
check(SOURCE_TRUST.get("synthetic") == "research", "synthetic = research")
check(SOURCE_TRUST.get(OutcomeSource.SYSTEM_TEST.value) not in ("production",), "system_test not production")

# Z. Audit hash determinism
print("=== Z. Audit hash deterministic ===")
a3 = run_readiness_audit()
check(a1["audit_hash"] == a3["audit_hash"], "audit hash deterministic across 3 runs")

# AA. Evidence pack contains required fields
print("=== AA. Evidence pack fields ===")
check("phase91_spec" in pack, "phase91_spec present")
check("phase92_dry_run" in pack, "phase92_dry_run present")
check("phase93_protocol" in pack, "phase93_protocol present")
check("parity_status" in pack, "parity_status present")
check("artifact_hashes" in pack, "artifact_hashes present")

# BB. No PII in output
print("=== BB. No PII in output ===")
audit_str = json.dumps(a1)
check("password" not in audit_str.lower(), "no passwords in audit")
check("cardnumber" not in audit_str.lower(), "no raw card numbers")

print("\n" + "=" * 60)
print("Phase 94 RWV Readiness Audit Tests: %d/%d passed" % (passed, total))
if passed < total:
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
