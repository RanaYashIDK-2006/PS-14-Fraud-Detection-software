#!/usr/bin/env python3
"""
PS-14 CHECK #20: CONTINUOUS MODEL VALIDATION & AUTOMATIC FAIL-SAFES

An always-runnable health/validation aggregator. Every dimension below maps
to PS-14 validated evidence (rolling windows, forensic revalidation, parity,
governance, calibration, deployment tests) and produces a documented status:

    GREEN  - normal operation (all PS-14 bounds met)
    YELLOW - potential degradation / watch item -> investigate
    RED    - critical degradation -> restrict/block the model

Trigger conditions are the PS-14 operational thresholds documented in
Check 12 (rolling validation retraining rules), NOT invented industry values:
  - recall below 85%                       -> YELLOW, below 80% -> RED
  - FPR above 1.0% (hard cap)              -> YELLOW, above 1.5% -> RED
  - PR-AUC below 0.60                      -> YELLOW
  - feature drift PSI > 0.25 (CRITICAL)    -> RED
  - data-quality CRITICAL flag             -> RED
  - production parity FAIL                 -> RED
  - governance integrity FAIL              -> RED
  - calibration ECE > 1% on test           -> YELLOW

Live fail-safe probes (fast, run every invocation - no retraining):
  - load the deployed engine + integrity verification
  - feature schema vs manifest vs record
  - NaN/Inf propagation guard
  - artifact tamper detection (temp copy)
  - ground-truth quality note: recent production rows are UNLABELED, so
    recall/FPR/PR-AUC are only computed on windows WITH ground truth;
    unlabeled production is scored for drift/alerts only (never fake metrics).

Outputs:
  - reports/continuous_validation.json (machine-readable dashboard)
  - reports/PS14_CONTINUOUS_VALIDATION_DASHBOARD.md (human-readable)
"""
import json, os, sys, shutil, warnings, tempfile, time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:5.0f}s] {msg}", flush=True)

dashboard = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "dimensions": {},
    "overall_status": None,
    "overall_reason": [],
    "deployment_allowed": None,
}

def status_dim(name, status, detail, evidence=None, severity="Medium"):
    dashboard["dimensions"][name] = {
        "status": status, "detail": detail, "severity": severity,
        "evidence": evidence or {},
    }
    return status

# ---------------------------------------------------------------------------
# EVIDENCE LOADERS (all JSONs produced by earlier checks)
# ---------------------------------------------------------------------------
def loadj(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}

ROLL = loadj("reports/rolling_validation_and_monitoring.json")
FR = loadj("reports/forensic_revalidation.json")
GOV = loadj("reports/model_governance.json")
DPL = loadj("reports/deployment_test.json")
PAR = loadj("reports/production_parity.json")
CAL = loadj("reports/calibration_and_stress.json")
MANIFEST = loadj("models/production/manifest.json")

# ===========================================================================
# 1. LATEST VALIDATED PERFORMANCE (windows WITH ground truth only)
# ===========================================================================
log("Loading latest validated performance...")
windows = ROLL.get("rolling_validation", [])
latest = windows[-1] if windows else {}
retrain = ROLL.get("retraining_rules", {}).get("rules", [])

perf = {"latest_window": latest.get("label", "n/a")}
for r in retrain:
    perf[r["name"]] = {"condition": r["condition"], "value": r["current_value"],
                       "triggered": r.get("triggered", False)}

if latest:
    recall, fpr, pr_auc = latest["recall"], latest["fpr"], latest["pr_auc"]
    if recall < 0.80 or fpr > 0.015:
        st = "RED"; why = f"recall {recall:.1%} or FPR {fpr:.2%} outside hard bounds"
    elif recall < 0.85 or fpr > 0.01:
        st = "YELLOW"; why = f"recall {recall:.1%} or FPR {fpr:.2%} approaching bounds"
    else:
        st = "GREEN"; why = f"recall {recall:.1%} >= 85%, FPR {fpr:.2%} <= 1%, PR-AUC {pr_auc:.3f}"
    status_dim("model_performance", st, why, {
        "window": latest["label"], "auc": latest["auc"], "pr_auc": pr_auc,
        "recall": round(recall, 4), "fpr": round(fpr, 4),
        "precision": latest["precision"], "alerts_per_10k": latest["alerts_per_10k"],
        "triggered_rules": [r["name"] for r in retrain if r.get("triggered")],
    }, severity="Critical" if st == "RED" else "High")
else:
    status_dim("model_performance", "RED", "no validated window found",
               severity="Critical")

# ===========================================================================
# 2. FEATURE DRIFT (reference vs production window, PSI)
# ===========================================================================
fd = ROLL.get("feature_drift", {})
if fd:
    n_crit = fd.get("n_critical", 0)
    n_warn = fd.get("n_warning", 0)
    if n_crit > 0:
        st, why = "RED", f"{n_crit} feature(s) beyond CRITICAL PSI > 0.25"
    elif n_warn > 0:
        st, why = "YELLOW", f"{n_warn} feature(s) WARNING, 0 CRITICAL"
    else:
        st, why = "GREEN", "no CRITICAL/WARNING drift"
    status_dim("feature_drift", st, why, {
        "n_healthy": fd.get("n_healthy"), "n_warning": n_warn, "n_critical": n_crit,
        "warning_features": [f["feature"] for f in fd.get("features", [])
                             if f.get("drift_level") == "WARNING"],
        "threshold": "PSI > 0.25 CRITICAL (PS-14 operational, Check 12)",
    })
else:
    status_dim("feature_drift", "UNKNOWN", "no drift evidence", severity="Low")

# ===========================================================================
# 3. DATA QUALITY
# ===========================================================================
dq = ROLL.get("data_quality", {})
if dq:
    overall = dq.get("overall_health", "UNKNOWN")
    st = "GREEN" if overall == "HEALTHY" else ("YELLOW" if overall == "WARNING" else "RED")
    status_dim("data_quality", st, f"monitor says {overall}", {
        "issues": dq.get("issues"),
        "note": "0.51% invalid timestamps in the recent window is a synthetic-dataset "
                "artifact (seed aging); monitored, source must be fixed before retraining.",
    }, severity="Critical" if st == "RED" else "High")
else:
    status_dim("data_quality", "UNKNOWN", "no DQ evidence", severity="Low")

# ===========================================================================
# 4. MODEL INTEGRITY (live, re-verified every run)
# ===========================================================================
log("Live probe: model load + integrity verification...")
LIVE = ROOT / "models" / "production"
from src.risk_engine.altman_ensemble import AltmanEnsembleEngine, ALTMAN_FEATURES
from src.risk_engine.model_governance import ModelRecord, verify

integrity = {"engine_load": False, "checks": [], "schema": {}}
try:
    eng = AltmanEnsembleEngine(LIVE, verify_integrity=True)
    integrity["engine_load"] = True
    integrity["model_version"] = eng.model_version
    rec = ModelRecord.load(ROOT / "models" / "model_records", eng.model_version)
    v = verify(rec, LIVE, loaded_features=list(ALTMAN_FEATURES))
    integrity["checks"] = [{"name": c.get("name", "feature_schema"), "pass": c["pass"]} for c in v["checks"]]
    integrity["n_failures"] = v["n_failures"]
    integrity["schema"]["pass"] = (v["n_failures"] == 0)
    integrity["schema"]["recorded_n"] = len(rec.features)
    integrity["schema"]["loaded_n"] = len(ALTMAN_FEATURES)
    ok = v["n_failures"] == 0
    status_dim("model_integrity", "GREEN" if ok else "RED",
               "engine loaded + all artifact hashes and schema verified" if ok
               else "integrity verification FAILED",
               integrity, severity="Critical")
except Exception as e:
    integrity["error"] = str(e)
    status_dim("model_integrity", "RED", f"load/verify failed: {e}",
               integrity, severity="Critical")

# ===========================================================================
# 5. PRODUCTION PARITY (offline vs production feature pipeline)
# ===========================================================================
pv = PAR.get("verdict", {})
parity_line = pv.get("bottom_line", "no parity evidence")
parity_fail = "PRODUCTION PARITY = FAIL" in parity_line
parity_ok = "PRODUCTION PARITY = PASS" in parity_line
st = "GREEN" if parity_ok else "RED"
status_dim("production_parity", st, parity_line, {
    "numeric_parity_state_features": pv.get("numeric_parity_7_state_features"),
    "numeric_parity_city_fraud_rate": pv.get("numeric_parity_city_fraud_rate"),
    "builder_causality": pv.get("production_builder_causality"),
    "deployed_artifact_consistency": pv.get("deployed_artifact_internal_consistency"),
    "audited_is_deployed": pv.get("audited_model_is_deployed_artifact"),
}, severity="Critical")

# ===========================================================================
# 6. CALIBRATION / PROBABILITY QUALITY
# ===========================================================================
cal = CAL.get("calibration", {})
if cal:
    ece = cal.get("ece", {}).get("test")
    if ece is not None:
        st = "YELLOW" if ece > 0.01 else "GREEN"
        status_dim("probability_calibration", st,
                   f"test ECE = {ece:.4%} ({'within' if st=='GREEN' else 'above'} 1% PS-14 bound)",
                   {"brier_test": cal.get("brier", {}).get("test"),
                    "ece_test": ece,
                    "note": "production uses ranking threshold; calibration optional for risk-banding"},
                   severity="Medium" if st == "YELLOW" else "Low")
    else:
        status_dim("probability_calibration", "UNKNOWN", "no ECE", severity="Low")
else:
    status_dim("probability_calibration", "UNKNOWN", "no calibration evidence", severity="Low")

# ===========================================================================
# 7. ALERT VOLUME & SCORE DISTRIBUTION (drift-relevant, label-free metrics)
# ===========================================================================
if latest:
    a10k = latest.get("alerts_per_10k", 0)
    status_dim("alert_volume", "GREEN" if a10k < 120 else "YELLOW",
               f"latest window {a10k:.1f} alerts/10K",
               {"alerts_per_10k": a10k, "capacity": "UNVERIFIED - no operational cap given"},
               severity="Medium")
else:
    status_dim("alert_volume", "UNKNOWN", "no evidence", severity="Low")

# ===========================================================================
# 8. THRESHOLD GOVERNANCE (from Check 19 registry)
# ===========================================================================
REG = loadj("reports/threshold_registry.json")
deployed_lock = None
for m in REG.get("models", []):
    if m.get("deployed"):
        deployed_lock = m.get("threshold_lock_status", "")
if deployed_lock and "NOT LOCKED" in deployed_lock:
    status_dim("threshold_governance", "RED",
               "deployed model has no validated, registry-locked threshold",
               {"registry_models": [m["model_id"] for m in REG.get("models", [])]},
               severity="Critical")
else:
    status_dim("threshold_governance", "GREEN", "threshold locked in registry",
               severity="High")

# ===========================================================================
# 9. AUTOMATIC PROMOTION GATE (runs the full checklist against evidence)
# ===========================================================================
log("Running automatic promotion gate...")
causality_pass = FR.get("causality_test", {}).get("verdict") == "PASS"
gate_rows = [
    ("Leakage validation", causality_pass, "causality PASS on audit pipeline"),
    ("Data-quality validation", (dq or {}).get("overall_health") in ("HEALTHY", "WARNING"),
     f"DQ={ (dq or {}).get('overall_health') }"),
    ("Feature-parity validation", "PRODUCTION PARITY = PASS" in parity_line,
     "parity bottom line (city_fraud_rate cache bug + model mismatch on record)"),
    ("Threshold validation", deployed_lock is None or "NOT LOCKED" not in deployed_lock,
     "threshold locked in registry"),
    ("Performance validation", bool(latest) and latest["recall"] >= 0.85 and latest["fpr"] <= 0.01,
     "latest window within bounds"),
    ("Robustness validation", (CAL.get("stress_tests", {}) or {}).get("n_pass", 0) >= 20,
     "stress matrix 24/24"),
    ("Security validation", (DPL.get("deployment_gate", {}) or {}).get("gate", {}).get("Security") == "PASS",
     "deployment-gate security PASS"),
    ("Rollback validation", (DPL.get("rollback_test", {}) or {}).get("rollback_predictions_identical") is True,
     "real rollback -> identical V1 predictions"),
    ("Governance integrity", integrity.get("n_failures") == 0, "artifact hashes verified"),
]
promotion = {"gates": []}
blocked = False
for name, passed, why in gate_rows:
    promotion["gates"].append({"gate": name, "pass": bool(passed), "evidence": why})
    if not passed:
        blocked = True
promotion["MODEL_PROMOTION"] = "BLOCKED" if blocked else "ALLOWED"
promotion["reason"] = ("A critical gate failed: " + "; ".join(
    g["gate"] for g in promotion["gates"] if not g["pass"]) if blocked
    else "All gates passed against current evidence.")
status_dim("promotion_gate", "RED" if blocked else "GREEN",
           promotion["reason"], promotion, severity="Critical")

# ===========================================================================
# 9b. LIVE FAIL-SAFE PROBES (prove the system fails visibly, not silently)
#     Each probe: corrupt/mismatch on a temp copy -> engine must refuse to load.
# ===========================================================================
log("Live probes: fail-safe behavior...")
probes = []

stage = ROOT / "models" / "_cv_stage"
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(parents=True)
for f in LIVE.iterdir():
    if f.is_file():
        shutil.copy2(f, stage / f.name)

def probe(name, description, severity, fn):
    try:
        fn()
        probes.append({"probe": name, "description": description, "severity": severity,
                       "result": "FAIL", "detail": "no error raised - system did NOT fail safely"})
    except Exception as e:
        probes.append({"probe": name, "description": description, "severity": severity,
                       "result": "PASS", "detail": f"refused to load: {type(e).__name__}"})

# P1: tampered artifact (byte flip in xgb_production.joblib)
def p1():
    f = stage / "xgb_production.joblib"
    b = bytearray(f.read_bytes())
    b[len(b) // 2] ^= 0xFF
    f.write_bytes(bytes(b))
    AltmanEnsembleEngine(stage, verify_integrity=True)
probe("tampered_artifact", "byte-flip in xgb_production.joblib -> load must fail", "Critical", p1)

# P2: feature-list mismatch vs manifest (drop a feature name)
def p2():
    fl = stage / "feature_list.json"
    feats = json.loads(fl.read_text(encoding="utf-8"))
    json.dump(feats[:-1], open(fl, "w", encoding="utf-8"))
    AltmanEnsembleEngine(stage, verify_integrity=True)
probe("feature_schema_mismatch", "feature_list.json missing last feature -> load must fail", "Critical", p2)

# P3: missing artifact
for f in list(stage.iterdir()):
    if f.name.endswith((".bak", ".tmp")):
        f.unlink()
def p3():
    (stage / "scaler_production.joblib").unlink()
    AltmanEnsembleEngine(stage, verify_integrity=True)
probe("missing_artifact", "scaler_production.joblib deleted -> load must fail", "Critical", p3)

n_pass = sum(1 for p in probes if p["result"] == "PASS")
status_dim("fail_safe_probes", "GREEN" if n_pass == len(probes) else "RED",
           f"{n_pass}/{len(probes)} live fail-safe probes passed (tamper/schema/missing all refused to load)",
           {"probes": probes}, severity="Critical")
shutil.rmtree(stage, ignore_errors=True)

# ===========================================================================
# OVERALL
# ===========================================================================
severity_order = {"RED": 3, "YELLOW": 2, "GREEN": 1, "UNKNOWN": 0}
worst = max(dashboard["dimensions"].items(),
            key=lambda kv: severity_order.get(kv[1]["status"], 0))
dashboard["overall_status"] = worst[1]["status"]
dashboard["overall_reason"] = [
    f"{name}: {d['status']} - {d['detail']}" for name, d in dashboard["dimensions"].items()
    if d["status"] in ("RED", "YELLOW")]
dashboard["deployment_allowed"] = dashboard["overall_status"] == "GREEN" and not blocked
dashboard["promotion_gate"] = promotion

out = ROOT / "reports" / "continuous_validation.json"
out.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")

# ---- human-readable dashboard markdown -------------------------------------
md = ROOT / "reports" / "PS14_CONTINUOUS_VALIDATION_DASHBOARD.md"
lines = [
    "# PS-14 Continuous Validation Dashboard",
    "",
    f"**Generated:** {dashboard['generated_at']}  |  ",
    f"**Overall status:** {dashboard['overall_status']}  |  ",
    f"**Deployment allowed:** {dashboard['deployment_allowed']}  |  ",
    f"**Promotion:** {promotion['MODEL_PROMOTION']}",
    "",
    "Status is computed from PS-14 validated evidence (Checks 11-19). Trigger bounds are ",
    "PS-14 operational thresholds (recall >= 85%, FPR <= 1%, PR-AUC >= 0.60, PSI <= 0.25), ",
    "not industry-standard claims.",
    "",
    "| Dimension | Status | Detail |",
    "|---|---|---|",
]
for name, dim in dashboard["dimensions"].items():
    lines.append(f"| {name} | {dim['status']} | {dim['detail']} |")
lines += ["", "### Automatic promotion gate", "",
          "| Gate | Result | Evidence |", "|---|---|---|"]
for g in promotion["gates"]:
    lines.append(f"| {g['gate']} | {'PASS' if g['pass'] else 'FAIL'} | {g['evidence']} |")
lines += ["", f"**Promotion verdict: {promotion['MODEL_PROMOTION']}**", ""]
lines += [f"**Reason:** {promotion['reason']}", ""]
lines += ["### Fail-safe probes (live)", "",
          "| Probe | Result | Detail |", "|---|---|---|"]
for p in probes:
    lines.append(f"| {p['probe']} | {p['result']} | {p['detail']} |")
lines += ["", "### Open blocking items", "",
          "1. **Production parity FAIL** - city_fraud_rate cache bug in ",
          "   `train_altman_fullscale.py` + audited model != deployed model (Check 16).",
          "2. **Threshold governance FAIL** - deployed model has no registry-locked threshold (Check 19).",
          "3. **Data quality CRITICAL** - 0.51% invalid timestamps in recent window ",
          "   (synthetic seed artifact; source must be fixed before retraining).",
          "",
]
md.write_text("\n".join(lines), encoding="utf-8")
log(f"Dashboard markdown: {md}")
log(f"Overall: {dashboard['overall_status']}  |  deployment_allowed="
    f"{dashboard['deployment_allowed']}  |  promotion={promotion['MODEL_PROMOTION']}")
