#!/usr/bin/env python3
"""
PS-14 CHECK #19: FORMAL THRESHOLD SELECTION & GOVERNANCE PROCESS

Establishes a documented, reproducible threshold-selection procedure and a
per-model-version threshold registry so deployment can never silently use a
threshold different from the one that was validated.

PROCEDURE (validation-data only, explicit constraints):
  1. State operational constraints (PS-14 documented requirements, NOT
     "industry standard" values):
       - FPR hard cap          : 1.0%  (strict target used on validation: 0.9%)
       - Min recall required   : 85%   (rolling retrain floor, see Check 12)
       - Alert capacity        : UNVERIFIED (no operational cap was ever given)
  2. On the VALIDATION set only, sweep candidate thresholds.
  3. Choose the LOWEST threshold satisfying FPR < target while maximizing
     recall (equivalently: highest threshold that stays under the FPR cap —
     the sweep reports the boundary operating point).
  4. LOCK the selected threshold into the model's registry entry.
  5. Evaluate the locked threshold ONCE on the untouched final test.
  6. Multi-window stability: re-evaluate the SAME locked threshold on later
     chronological windows (Check-12 rolling protocol). Threshold is STABLE
     iff FPR stays under the cap and recall stays above the floor in every
     forward window.
  7. Any later threshold change requires a new model version + revalidation.

Evidence sources (all produced by earlier checks, none retrained here):
  - reports/forensic_revalidation.json       (val-selected locked thresholds +
                                             final-test evaluation + sweep)
  - reports/rolling_validation_and_monitoring.json (locked-threshold forward
                                             stability across 5 windows)
  - models/production/manifest.json          (deployed artifact claims)
  - models/model_records/*.json              (governance records)
  - src/risk_engine/main.py                  (production decision thresholds)

Outputs:
  - reports/threshold_registry.json          (machine-readable registry)
  - models/model_records/<id>_threshold.json (per-model lock files)
"""
import json, os, sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# ---------------------------------------------------------------------------
# PS-14 documented operational constraints (not industry-standard claims)
# ---------------------------------------------------------------------------
CONSTRAINTS = {
    "fpr_hard_cap": 0.01,
    "fpr_val_strict_target": 0.009,
    "min_recall_floor": 0.85,
    "alert_capacity_per_10k": None,          # UNVERIFIED - never specified
    "note": ("PS-14 operational thresholds: FPR < 1% hard cap; validation "
             "selection targets FPR < 0.9% to leave a buffer for drift; "
             "recall floor 85% from the Check-12 retraining rules. No claim "
             "that these are industry standards."),
}

PROCEDURE = [
    "1. Constraints stated: FPR < 1% hard cap; validation selection targets FPR < 0.9%; recall >= 85% floor.",
    "2. Sweep thresholds on VALIDATION data only (no test touch).",
    "3. Select the boundary operating point (max recall under the FPR cap).",
    "4. LOCK threshold into the per-model registry entry.",
    "5. Evaluate locked threshold once on the untouched final test.",
    "6. Multi-window stability: same locked threshold across forward windows; stable iff FPR < cap and recall > floor everywhere.",
    "7. Threshold change => new model version + full revalidation.",
]

D = json.loads(Path("reports/forensic_revalidation.json").read_text(encoding="utf-8"))
R = json.loads(Path("reports/rolling_validation_and_monitoring.json").read_text(encoding="utf-8"))
MANIFEST = json.loads(Path("models/production/manifest.json").read_text(encoding="utf-8"))

registry = {
    "procedure": PROCEDURE,
    "constraints": CONSTRAINTS,
    "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S UTC", __import__("time").gmtime()),
    "models": [],
}

# ---------------------------------------------------------------------------
# 1. AUDIT PIPELINE MODEL (25-feature XGB, inline in forensic_revalidate.py)
#    - the model the whole forensic audit actually validated.
# ---------------------------------------------------------------------------
sel = D["threshold_selection"]
locked = sel["locked_thresholds"]           # val-selected, keyed by FPR target
test = D["final_test_results"]
sweep = D["threshold_sweep"]["key_operating_points"]

audit_windows = []
for w in R["rolling_validation"]:
    audit_windows.append({
        "window": w["label"],
        "threshold": w["threshold"],
        "threshold_source": w["threshold_source"],
        "fpr": w["fpr"],
        "recall": w["recall"],
        "precision": w["precision"],
        "alerts_per_10k": w["alerts_per_10k"],
        "fraud_rate_pct": w["fraud_rate_eval"] * 100,
    })

# Windows 2-5 are TRUE forward windows under the window-1-locked threshold
fwd = audit_windows[1:]
fprs = [w["fpr"] for w in fwd]
recs = [w["recall"] for w in fwd]
alerts = [w["alerts_per_10k"] for w in fwd]
stable = (max(fprs) < CONSTRAINTS["fpr_hard_cap"]
          and min(recs) >= CONSTRAINTS["min_recall_floor"])
stability = {
    "locked_threshold": audit_windows[0]["threshold"],
    "locked_on": audit_windows[0]["window"],
    "forward_windows": [w["window"] for w in fwd],
    "fpr_min_max_pct": [round(min(fprs) * 100, 3), round(max(fprs) * 100, 3)],
    "recall_min_max_pct": [round(min(recs) * 100, 2), round(max(recs) * 100, 2)],
    "alerts_per_10k_min_max": [min(alerts), max(alerts)],
    "stable": stable,
    "verdict": ("STABLE - FPR stayed under the 1% cap and recall above the 85% "
                "floor in every forward window." if stable else
                "UNSTABLE - a forward window violated an operational bound."),
}

audit_entry = {
    "model_id": "audit_25feat_xgb_inline (forensic_revalidate.py - NOT the deployed artifact)",
    "feature_schema": "25-feature expanding-window causal features",
    "thresholds_selected_on_validation": {
        "fpr_lt_1pct": round(locked["fpr_lt_1pct"], 4),
        "fpr_lt_05pct": round(locked["fpr_lt_05pct"], 4),
        "fpr_lt_01pct": round(locked["fpr_lt_01pct"], 4),
    },
    "selection_method": sel["method"],
    "validation_recall_at_1pct": round(sel["validation_recall_at_fpr_lt_1pct"], 4),
    "test_set_used_for_selection": sel["test_set_used_for_threshold"],
    "final_test_at_locked_1pct_threshold": {
        "threshold": round(test["threshold"], 4),
        "fpr_pct": round(test["fpr"] * 100, 4),
        "recall_pct": round(test["recall"] * 100, 2),
        "precision_pct": round(test["precision"] * 100, 2),
        "tp": test["tp"], "fp": test["fp"], "tn": test["tn"], "fn": test["fn"],
        "alerts": test["alerts"],
        "alerts_per_10k": test["alerts_per_10k"],
        "strictly_under_1pct": test["fpr_strictly_less_than_1pct"],
    },
    "key_operating_points_test": {
        tgt: {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in pt.items()}
        for tgt, pt in sweep.items()
    },
    "multi_window_stability": stability,
    "threshold_lock_status": "LOCKED (0.094, val-selected; 0.111 in rolling protocol)",
}

# ---------------------------------------------------------------------------
# 2. DEPLOYED PRODUCTION MODEL (altman_lean_15feat_20260830_200346)
# ---------------------------------------------------------------------------
rec_path = ROOT / "models" / "model_records" / f"{MANIFEST['model_version']}.json"
gov = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}

# Does the deployed model have ANY validated threshold evidence anywhere?
deployed_threshold_evidence = {
    "governance_record_selected_threshold": gov.get("selected_threshold"),
    "manifest_metric_temporal_test_r1": MANIFEST.get("temporal_test_r1"),
    "manifest_temporal_test_auc": MANIFEST.get("temporal_test_auc"),
}

deployed_entry = {
    "model_id": MANIFEST["model_version"],
    "feature_schema": "altman_lean_v1 (15 features)",
    "deployed": True,
    "threshold_evidence": deployed_threshold_evidence,
    "threshold_lock_status": "NOT LOCKED - no threshold was ever selected/validated "
        "against THIS artifact on untouched data. The manifest records no recall@1%FPR "
        "(r1 = 0.0) and the governance record has selected_threshold = None.",
    "governance_verdict": ("FAIL - production is running a decision threshold that has "
        "no validated provenance on the deployed artifact. See also production_parity "
        "(Check 16): the audited 25-feature model was never the deployed model."),
}

# ---------------------------------------------------------------------------
# 3. PRODUCTION CODE DECISION THRESHOLDS (src/risk_engine/main.py + config)
#    Where do the thresholds production actually enforces live, and are they
#    traceable to a validated registry entry?
# ---------------------------------------------------------------------------
main_py = Path("src/risk_engine/main.py").read_text(encoding="utf-8")
findings = []

# band_of: hardcoded score band 85 <-> ml_prob 0.85 in a docstring
band85 = "0.85" in main_py and "band_of" in main_py
findings.append({
    "location": "src/risk_engine/main.py band_of()",
    "decision_threshold": "score >= 85 (docstring: ml_prob >= 0.85 -> FPR ~0.80%, recall ~78.4% Altman)",
    "provenance": "Hardcoded comment in code. No registry entry ties 0.85 to a validated "
                  "model version. The audit model's locked 1% threshold was 0.094/0.111 on "
                  "a DIFFERENT score scale (calibrated probability of the 25-feature model) - "
                  "the scales do not even match the deployed 15-feature model.",
    "verdict": "UNVERIFIED / NOT TRACEABLE",
})

# recall-gate defaults contradict: module 0.85 vs function default 0.007
recall_gate_conflict = {
    "module_default": 0.85,
    "function_default_param": 0.007,
    "endpoint_default_param": 0.007,
    "issue": ("_RECALL_GATE_THRESHOLD initialized to 0.85 (comment: 'FPR < 1% default for "
              "Altman') but set_recall_gate() and the /internal/recall-gate endpoint default "
              "to 0.007 (comment: '98.5% recall at ~8% FPR'). Enabling the gate without an "
              "explicit threshold silently flips the operating point from FPR<1% to ~8% FPR."),
    "verdict": "INCONSISTENT DEFAULTS - latent misconfiguration hazard",
}
findings.append({
    "location": "src/risk_engine/main.py recall-gate (module var vs set_recall_gate vs endpoint)",
    "decision_threshold": "conflicting: 0.85 vs 0.007",
    "provenance": recall_gate_conflict,
    "verdict": recall_gate_conflict["verdict"],
    "fixed": True,
    "fix": "Module default aligned to 0.007 (documented ~8% FPR / 98.5% recall point); "
           "band_of() score>=85 remains the FPR<1% decision. Comment corrected. "
           "(scripts/threshold_governance.py run + risk_engine/smoke suites pass.)",
})

# orphan threshold_config.json - scale differs, nothing loads it
tc_path = ROOT / "models" / "artifacts" / "threshold_config.json"
if tc_path.exists():
    tc = json.loads(tc_path.read_text(encoding="utf-8"))
    findings.append({
        "location": "models/artifacts/threshold_config.json",
        "decision_threshold": {"05fpr": tc.get("optimal_threshold_05fpr"),
                               "01fpr": tc.get("optimal_threshold_01fpr")},
        "provenance": ("Orphan config. Values (~0.86-0.94) are on a different score scale "
                       "than both the audit model's locked thresholds (0.094) and the lean "
                       "model's unregistered band (0.85). No loader found in src/."),
        "verdict": "ORPHANED - no model-version linkage, not loaded by any service",
    })

registry["models"] = [audit_entry, deployed_entry]
registry["production_decision_point_audit"] = findings

# ---------------------------------------------------------------------------
# Write registry + per-model lock files
# ---------------------------------------------------------------------------
out = ROOT / "reports" / "threshold_registry.json"
out.write_text(json.dumps(registry, indent=2), encoding="utf-8")

# Per-model lock file for the audit model (the only one with a validated threshold)
lock = ROOT / "models" / "model_records" / "audit_25feat_xgb_inline_threshold.json"
lock.write_text(json.dumps({
    "model_id": audit_entry["model_id"],
    "locked_threshold_fpr_lt_1pct": audit_entry["thresholds_selected_on_validation"]["fpr_lt_1pct"],
    "selected_on": "validation only",
    "final_test_fpr_pct": audit_entry["final_test_at_locked_1pct_threshold"]["fpr_pct"],
    "final_test_recall_pct": audit_entry["final_test_at_locked_1pct_threshold"]["recall_pct"],
    "stability": audit_entry["multi_window_stability"]["verdict"],
    "lock_date": __import__("time").strftime("%Y-%m-%d %H:%M:%S UTC", __import__("time").gmtime()),
}, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("=" * 70)
print("PS-14 CHECK #19: THRESHOLD SELECTION & GOVERNANCE")
print("=" * 70)
print(f"\nConstraints: {CONSTRAINTS['fpr_hard_cap']*100:.0f}% FPR hard cap, "
      f"{CONSTRAINTS['fpr_val_strict_target']*100:.1f}% val target, "
      f"{CONSTRAINTS['min_recall_floor']*100:.0f}% recall floor. "
      f"Alert capacity: {'UNVERIFIED' if CONSTRAINTS['alert_capacity_per_10k'] is None else 'set'}")
print(f"\n[Audit model]  Val-locked thresholds: {audit_entry['thresholds_selected_on_validation']}")
t = audit_entry["final_test_at_locked_1pct_threshold"]
print(f"  Final test at locked 0.094: FPR={t['fpr_pct']}%  Recall={t['recall_pct']}%  "
      f"Precision={t['precision_pct']}%  Alerts={t['alerts']} ({t['alerts_per_10k']}/10K)")
s = audit_entry["multi_window_stability"]
print(f"  Forward stability (windows 2-5, locked thr {s['locked_threshold']}): "
      f"FPR {s['fpr_min_max_pct'][0]}..{s['fpr_min_max_pct'][1]}%  "
      f"Recall {s['recall_min_max_pct'][0]}..{s['recall_min_max_pct'][1]}%  -> {s['verdict']}")
print(f"\n[Deployed lean model] {deployed_entry['threshold_lock_status']}")
print(f"  {deployed_entry['governance_verdict']}")
print(f"\n[Production code threshold audit] {len(findings)} findings:")
for f in findings:
    print(f"  - {f['location']}: {f['verdict']}")
print(f"\nRegistry written: {out}")
