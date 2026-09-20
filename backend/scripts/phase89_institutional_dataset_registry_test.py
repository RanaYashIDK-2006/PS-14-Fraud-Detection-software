"""Phase 89: Institutional dataset registry tests."""
from __future__ import annotations
import hashlib, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.monitoring.institutional_dataset_registry import (
    run_registry, get_preflight_summary, CANONICAL_48,
    WORLDLINE_RECORD, NOVATTI_RECORD, _build_worldline_preflights,
    _build_novatti_preflights, WORLDLINE_ENTITIES, NOVATTI_ENTITIES,
)
from src.monitoring.outcome_pipeline import OutcomeSource, SOURCE_TRUST
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

print("=== A. Deterministic output ===")
r1 = run_registry()
r2 = run_registry()
check(r1.manifest_hash == r2.manifest_hash, "identical manifest hash")
check(r1.registry_id == "INSTITUTIONAL-DATASET-REGISTRY-89", "constant registry_id")
check(r1.to_dict() == r2.to_dict(), "full result deterministic")

print("=== B. Canonical 48-feature list ===")
check(len(CANONICAL_48) == 48, "48 canonical features")
check(len(set(CANONICAL_48)) == 48, "no duplicates")
wl_pf = _build_worldline_preflights()
nv_pf = _build_novatti_preflights()
wl_features = [f.feature_name for f in wl_pf]
nv_features = [f.feature_name for f in nv_pf]
check(wl_features == list(CANONICAL_48), "Worldline preflights match canonical order")
check(nv_features == list(CANONICAL_48), "Novatti preflights match canonical order")
check(len(wl_pf) == 48, "Worldline has 48 preflights")
check(len(nv_pf) == 48, "Novatti has 48 preflights")

print("=== C. Record existence ===")
check(WORLDLINE_RECORD.dataset_id == "INST-WORLDLINE-001", "Worldline record exists")
check(NOVATTI_RECORD.dataset_id == "INST-NOVATTI-001", "Novatti record exists")
check(WORLDLINE_RECORD.approximate_row_count > 50_000_000, "Worldline >50M rows")
check(NOVATTI_RECORD.approximate_row_count == 126_184, "Novatti 126184 rows")
check(NOVATTI_RECORD.fraud_count == 394, "Novatti 394 fraud cases")

print("=== D. Provenance states ===")
check(WORLDLINE_RECORD.provenance_status in ("verified", "partially_verified", "unknown"), "Worldline provenance valid")
check(NOVATTI_RECORD.provenance_status in ("verified", "partially_verified", "unknown"), "Novatti provenance valid")

print("=== E. Access states ===")
check(WORLDLINE_RECORD.public_access_status == "confidential", "Worldline is confidential")
check(WORLDLINE_RECORD.institutional_access_required is True, "Worldline requires institutional access")
check(NOVATTI_RECORD.public_access_status == "confidential", "Novatti is confidential")
check(NOVATTI_RECORD.institutional_access_required is True, "Novatti requires institutional access")

print("=== F. Label provenance ===")
check(WORLDLINE_RECORD.label_type == "human_investigator", "Worldline: human_investigator")
check(NOVATTI_RECORD.label_type == "chargeback", "Novatti: chargeback")

print("=== G. Entity continuity ===")
wl_ent = [e for e in WORLDLINE_ENTITIES if e.entity_type == "user"]
check(len(wl_ent) == 1 and wl_ent[0].availability == "available", "Worldline user entity available")
nv_ent = [e for e in NOVATTI_ENTITIES if e.entity_type == "user"]
check(len(nv_ent) == 1 and nv_ent[0].availability == "removed", "Novatti user entity removed")

print("=== H. Preflight derivability ===")
wl_s = get_preflight_summary("INST-WORLDLINE-001", wl_pf)
nv_s = get_preflight_summary("INST-NOVATTI-001", nv_pf)
wl_total = sum(wl_s.values())
nv_total = sum(nv_s.values())
check(wl_total == 48, "Worldline 48 features classified")
check(nv_total == 48, "Novatti 48 features classified")
check(wl_s.get("direct", 0) >= 1, "Worldline has at least 1 direct")
check(nv_s.get("direct", 0) >= 1, "Novatti has at least 1 direct")
check(wl_s.get("not_derivable", 0) >= 1, "Worldline has not-derivable features")
check(nv_s.get("not_derivable", 0) >= 20, "Novatti has many not-derivable")

print("=== I. Worldline promising but unverified ===")
check(WORLDLINE_RECORD.feature_compatibility_status == "promising_unverified", "Worldline: promising_unverified")
check("unknown" in [f.classification for f in wl_pf], "Worldline has UNKNOWN features")

print("=== J. Novatti has removed entities ===")
check(NOVATTI_RECORD.feature_compatibility_status == "partial", "Novatti: partial")
nv_removed = sum(1 for e in NOVATTI_ENTITIES if e.availability == "removed")
check(nv_removed >= 2, "Novatti has >=2 removed entity types")

print("=== K. Contamination ===")
r = run_registry()
check(r.contamination_assessment["worldline"] == "no_contamination_evidence_found", "No Worldline contamination")
check(r.contamination_assessment["novatti"] == "no_contamination_evidence_found", "No Novatti contamination")

print("=== L. RWV gate preservation ===")
check(r.rwv_gate_status == "BLOCKED_PENDING_ELIGIBLE_DATASET", "RWV remains BLOCKED")
check(WORLDLINE_RECORD.acquisition_status != "data_received", "Worldline not yet received")
check(NOVATTI_RECORD.acquisition_status != "data_received", "Novatti not yet received")

print("=== M. Manifest determinism ===")
check(len(r.manifest_hash) == 64, "manifest hash is SHA-256")
r3 = run_registry()
check(r.manifest_hash == r3.manifest_hash, "manifest hash deterministic")

print("=== N. No PII in output ===")
r_str = json.dumps(r.to_dict())
check("password" not in r_str.lower(), "no passwords")
check("credit card" not in r_str.lower(), "no credit card numbers")

print("=== O. Registration does not confer eligibility ===")
check(WORLDLINE_RECORD.feature_compatibility_status != "verified", "Worldline not verified")
check(NOVATTI_RECORD.feature_compatibility_status != "verified", "Novatti not verified")
check(WORLDLINE_RECORD.acquisition_status == "access_request_required", "Worldline access not yet requested")
check(NOVATTI_RECORD.acquisition_status == "access_request_required", "Novatti access not yet requested")

print("=== P. Temporal requirements ===")
for pf in wl_pf:
    if pf.feature_name == "user_tx_count":
        check("history" in pf.temporal_requirement or "leakage" in pf.classification or "not_derivable" in pf.classification,
            "Worldline user_tx_count has temporal requirement")
        break

print("=== Q. Fraud-rate features ===")
fraud_features = {"user_fraud_rate", "merch_fraud_rate", "city_fraud_rate"}
for pf in wl_pf:
    if pf.feature_name in fraud_features:
        check(pf.classification in ("leakage_risk", "not_derivable", "unknown"),
            "Worldline " + pf.feature_name + ": " + pf.classification)
for pf in nv_pf:
    if pf.feature_name in fraud_features:
        check(pf.classification in ("leakage_risk", "not_derivable", "unknown"),
            "Novatti " + pf.feature_name + ": " + pf.classification)

print("=== R. Evidence sources ===")
check(len(WORLDLINE_RECORD.evidence_sources) >= 1, "Worldline has evidence sources")
check(len(NOVATTI_RECORD.evidence_sources) >= 1, "Novatti has evidence sources")

print("=== S. Blockers documented ===")
check(len(WORLDLINE_RECORD.blockers) >= 3, "Worldline has >=3 blockers")
check(len(NOVATTI_RECORD.blockers) >= 3, "Novatti has >=3 blockers")

print("=== T. Next action ===")
check("ACCESS" in WORLDLINE_RECORD.next_action.upper() or "REQUEST" in WORLDLINE_RECORD.next_action.upper(), "Worldline next action access-related")
check("ACCESS" in NOVATTI_RECORD.next_action.upper() or "REQUEST" in NOVATTI_RECORD.next_action.upper(), "Novatti next action access-related")

print("=== U. No production impact ===")
from src.privacy_layer.native_features import ALTMAN_NATIVE_FEATURES
check(len(ALTMAN_NATIVE_FEATURES) == 48, "ALTMAN_NATIVE_FEATURES unchanged")

print("\n" + "=" * 60)
print("Phase 89 Institutional Registry Tests: %d/%d passed" % (passed, total))
if passed < total:
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
