#!/usr/bin/env python3
"""PS-14 Fraud Quality Scorecard.

Consolidates metrics across model performance, operations, fraud detection,
privacy, security, and model lifecycle into a single report.

Usage:
    python scripts/quality_scorecard.py [--json] [--db-dir db]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_metrics_comparison() -> dict:
    """Load ML metrics from the training artifacts."""
    path = ROOT / "models" / "artifacts" / "metrics_comparison.csv"
    if not path.exists():
        return {"status": "NOT MEASURED", "reason": "models/artifacts/metrics_comparison.csv not found"}
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"status": "NOT MEASURED", "reason": "empty metrics file"}
    # Find the fusion row or the best model
    fusion = None
    for r in rows:
        if "fusion" in r.get("model", "").lower() or "stacker" in r.get("model", "").lower():
            fusion = r
            break
    if not fusion:
        fusion = rows[0]
    return {
        "model": fusion.get("model", "unknown"),
        "pr_auc": float(fusion.get("pr_auc", 0)),
        "roc_auc": float(fusion.get("roc_auc", 0)),
        "recall_at_1pct_fpr": float(fusion.get("recall_at_1pct_fpr", 0)),
        "f1": float(fusion.get("f1", 0)),
        "fpr": float(fusion.get("fpr", 0)),
        "fnr": float(fusion.get("fnr", 0)),
    }


def _load_calibration() -> dict:
    """Check calibration status."""
    cal_path = ROOT / "models" / "artifacts" / "calibrator.joblib"
    if not cal_path.exists():
        return {"status": "NOT MEASURED", "reason": "no calibrator artifact"}
    return {"status": "PLATT SCALING", "artifact": "calibrator.joblib"}


def _load_model_info() -> dict:
    """Load model metadata."""
    meta_path = ROOT / "models" / "artifacts" / "metadata.json"
    if not meta_path.exists():
        return {"status": "NOT MEASURED"}
    import json as _json
    meta = _json.loads(meta_path.read_text())
    return {
        "model_version": f"seed{meta.get('seed', '?')}-{Path(meta.get('data', '?')).name}",
        "feedback_count": meta.get("feedback", 0),
        "has_ood_gate": "ood_gate" in meta,
    }


def _load_drift_status() -> dict:
    """Check drift monitoring status."""
    baseline = ROOT / "data" / "drift_baseline.json"
    if not baseline.exists():
        return {"status": "NOT MEASURED", "reason": "no drift baseline"}
    return {"status": "BASELINE EXISTS", "path": str(baseline.relative_to(ROOT))}


def _load_audit_stats(db_dir: Path) -> dict:
    """Load audit trail statistics."""
    db = db_dir / "audit.db"
    if not db.exists():
        return {"status": "NOT MEASURED"}
    con = sqlite3.connect(db)
    total = con.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    by_type = dict(con.execute(
        "SELECT event_type, COUNT(*) FROM audit_events GROUP BY event_type"
    ).fetchall())
    access_count = con.execute("SELECT COUNT(*) FROM audit_access_log").fetchone()[0]
    con.close()
    return {
        "total_events": total,
        "by_type": by_type,
        "access_log_entries": access_count,
    }


def _load_risk_stats(db_dir: Path) -> dict:
    """Load risk score statistics."""
    db = db_dir / "risk.db"
    if not db.exists():
        return {"status": "NOT MEASURED"}
    con = sqlite3.connect(db)
    total = con.execute("SELECT COUNT(*) FROM risk_scores").fetchone()[0]
    bands = dict(con.execute(
        "SELECT risk_band, COUNT(*) FROM risk_scores GROUP BY risk_band"
    ).fetchall())
    avg_score = con.execute("SELECT AVG(risk_score) FROM risk_scores").fetchone()[0]
    high_risk = con.execute("SELECT COUNT(*) FROM risk_scores WHERE risk_band='high'").fetchone()[0]
    degraded = con.execute("SELECT COUNT(*) FROM risk_scores WHERE degraded=1").fetchone()[0]
    outcomes = con.execute("SELECT COUNT(*) FROM verification_outcomes").fetchone()[0]
    disputed = con.execute("SELECT COUNT(*) FROM verification_outcomes WHERE outcome='disputed'").fetchone()[0]
    con.close()
    return {
        "total_scores": total,
        "bands": bands,
        "avg_score": round(avg_score or 0, 1),
        "high_risk_count": high_risk,
        "degraded_count": degraded,
        "verification_outcomes": outcomes,
        "disputed_count": disputed,
    }


def _load_identity_stats(db_dir: Path) -> dict:
    """Load identity statistics."""
    db = db_dir / "identity.db"
    if not db.exists():
        return {"status": "NOT MEASURED"}
    con = sqlite3.connect(db)
    users = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    pseudonyms = con.execute("SELECT COUNT(*) FROM pseudonym_mapping").fetchone()[0]
    break_glass = con.execute(
        "SELECT COUNT(*) FROM break_glass_log"
    ).fetchone()[0] if "break_glass_log" in [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()] else 0
    con.close()
    return {
        "users": users,
        "pseudonyms": pseudonyms,
        "break_glass_events": break_glass,
    }


def _load_feature_stats(db_dir: Path) -> dict:
    """Load feature store statistics."""
    db = db_dir / "features.db"
    if not db.exists():
        return {"status": "NOT MEASURED"}
    con = sqlite3.connect(db)
    events = con.execute("SELECT COUNT(*) FROM transaction_features").fetchone()[0]
    profiles = con.execute("SELECT COUNT(*) FROM fraud_profiles").fetchone()[0]
    devices = con.execute("SELECT COUNT(*) FROM device_fingerprints").fetchone()[0]
    con.close()
    return {
        "total_events": events,
        "profiles": profiles,
        "known_devices": devices,
    }


def _load_security_tests() -> dict:
    """Run security tests and report results."""
    result = os.popen(f"{sys.executable} scripts/security_test.py 2>&1").read()
    passed = result.count("[PASS]")
    failed = result.count("[FAIL]")
    return {
        "passed": passed,
        "failed": failed,
        "status": "ALL PASSED" if failed == 0 else f"{failed} FAILED",
    }


def _load_k_anonymity() -> dict:
    """Check k-anonymity status."""
    report = ROOT / "data" / "kanon_report.json"
    if report.exists():
        import json as _json
        data = _json.loads(report.read_text())
        return {"status": "CHECKED", "k": data.get("k", "?"), "violations": data.get("violations", "?")}
    return {"status": "NOT MEASURED", "reason": "no k-anonymity report"}


def main():
    parser = argparse.ArgumentParser(description="PS-14 Quality Scorecard")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--db-dir", type=Path, default=ROOT / "db")
    args = parser.parse_args()

    scorecard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "metrics": _load_metrics_comparison(),
            "calibration": _load_calibration(),
            "info": _load_model_info(),
            "drift": _load_drift_status(),
        },
        "operations": {
            "note": "NOT MEASURED — requires live load test (scripts/load_test.py)",
        },
        "fraud_detection": _load_risk_stats(args.db_dir),
        "privacy": {
            "k_anonymity": _load_k_anonymity(),
            "feature_store": _load_feature_stats(args.db_dir),
        },
        "security": {
            "security_tests": _load_security_tests(),
        },
        "audit": _load_audit_stats(args.db_dir),
        "identity": _load_identity_stats(args.db_dir),
        "model_lifecycle": {
            "registry": "file-based (models/artifacts/registry.json)",
            "shadow": "opt-in (mode: shadow)",
            "canary": "opt-in (mode: canary, configurable %)",
            "rollback": "auto-rollback on breach",
        },
    }

    if args.json:
        print(json.dumps(scorecard, indent=2))
    else:
        print("=" * 60)
        print("PS-14 FRAUD QUALITY SCORECARD")
        print("=" * 60)
        print(f"Generated: {scorecard['generated_at']}")
        print()

        # Model
        m = scorecard["model"]["metrics"]
        if "pr_auc" in m:
            print(f"MODEL: {m['model']}")
            print(f"  PR-AUC: {m['pr_auc']:.4f}  ROC-AUC: {m['roc_auc']:.4f}")
            print(f"  Recall@1%FPR: {m['recall_at_1pct_fpr']:.4f}  F1: {m['f1']:.4f}")
            print(f"  FPR: {m['fpr']:.4f}  FNR: {m['fnr']:.4f}")
        else:
            print(f"MODEL: {m.get('status', 'unknown')}")
        print(f"  Calibration: {scorecard['model']['calibration'].get('status', '?')}")
        print(f"  Version: {scorecard['model']['info'].get('model_version', '?')}")
        print(f"  Drift: {scorecard['model']['drift'].get('status', '?')}")
        print()

        # Fraud detection
        f = scorecard["fraud_detection"]
        if "total_scores" in f:
            print(f"FRAUD DETECTION: {f['total_scores']} scores")
            print(f"  Bands: {f['bands']}")
            print(f"  Avg score: {f['avg_score']}  High-risk: {f['high_risk_count']}")
            print(f"  Degraded: {f['degraded_count']}  Outcomes: {f['verification_outcomes']}  Disputed: {f['disputed_count']}")
        else:
            print(f"FRAUD DETECTION: {f.get('status', 'unknown')}")
        print()

        # Privacy
        p = scorecard["privacy"]
        print(f"PRIVACY:")
        print(f"  K-anonymity: {p['k_anonymity'].get('status', '?')}")
        fs = p["feature_store"]
        if "total_events" in fs:
            print(f"  Feature store: {fs['total_events']} events, {fs['profiles']} profiles, {fs['known_devices']} devices")
        print()

        # Security
        s = scorecard["security"]["security_tests"]
        print(f"SECURITY: {s['status']} ({s['passed']} passed, {s['failed']} failed)")
        print()

        # Audit
        a = scorecard["audit"]
        if "total_events" in a:
            print(f"AUDIT: {a['total_events']} events")
            print(f"  By type: {a['by_type']}")
            print(f"  Access log: {a['access_log_entries']} entries")
        print()

        # Identity
        i = scorecard["identity"]
        if "users" in i:
            print(f"IDENTITY: {i['users']} users, {i['pseudonyms']} pseudonyms")
            print(f"  Break-glass events: {i['break_glass_events']}")
        print()

        # Model lifecycle
        print(f"MODEL LIFECYCLE:")
        for k, v in scorecard["model_lifecycle"].items():
            print(f"  {k}: {v}")
        print()

        # Operations
        print(f"OPERATIONS: {scorecard['operations']['note']}")


if __name__ == "__main__":
    main()
