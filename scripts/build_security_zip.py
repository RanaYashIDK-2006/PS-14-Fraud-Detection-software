"""Build a focused security audit zip for PS-14."""
import zipfile
import os

zip_name = "ps14_security_audit.zip"

# Security-relevant files only
security_files = [
    # Security tests
    "scripts/penetration_test.py",
    "scripts/security_scan.py",
    "scripts/security_test.py",
    "scripts/security_ci_gate.py",
    "scripts/sql_injection_test.py",
    "scripts/privacy_test.py",
    "scripts/leakage_test.py",
    "scripts/leakage_structural_test.py",
    "scripts/temporal_test.py",
    "scripts/calibration_test.py",
    "scripts/adversarial_test.py",
    "scripts/negative_test.py",

    # Identity service (PII boundary)
    "src/identity_service/main.py",
    "src/identity_service/security.py",
    "src/identity_service/models.py",

    # Privacy layer (feature derivation)
    "src/privacy_layer/main.py",
    "src/privacy_layer/features.py",
    "src/privacy_layer/models.py",
    "src/privacy_layer/feature_registry.py",

    # Risk engine (scoring)
    "src/risk_engine/main.py",
    "src/risk_engine/fusion.py",
    "src/risk_engine/calibration.py",
    "src/risk_engine/models.py",
    "src/risk_engine/reason_codes.py",
    "src/risk_engine/rules.yaml",

    # Audit service
    "src/audit_service/main.py",
    "src/audit_service/writer.py",
    "src/audit_service/analyst_store.py",

    # Front service (admin endpoints)
    "src/front_service/main.py",

    # Middleware
    "src/middleware/security_headers.py",
    "src/middleware/totp.py",
    "src/middleware/rate_limiter.py",

    # Config
    "src/settings.py",
    ".env.example",
    "requirements.txt",
    "docker-compose.yml",

    # Training pipeline
    "src/train_compare.py",
    "scripts/build_model_artifacts.py",
    "scripts/generate_reports.py",

    # Model artifacts (required for validation)
    "models/artifacts/calibrator.joblib",
    "models/artifacts/scaler.joblib",
    "models/artifacts/xgboost.joblib",
    "models/artifacts/logistic_regression.joblib",
    "models/artifacts/random_forest.joblib",
    "models/artifacts/isolation_forest.joblib",
    "models/artifacts/stacker.joblib",
    "models/artifacts/iso_train_scores.joblib",
    "models/artifacts/validation_labels.joblib",
    "models/artifacts/fused_val_scores.joblib",

    # Model manifest
    "models/artifacts/metadata.json",

    # Tests
    "scripts/regression_suite.py",
    "scripts/pseudonym_separation_test.py",
    "scripts/production_gate_test.py",
    "scripts/smoke_test.py",
    "scripts/risk_engine_test.py",
    "scripts/drift_test.py",
    "scripts/ood_gate_test.py",
    "scripts/rules_gate_test.py",
    "scripts/tune_test.py",
    "scripts/pipeline_test.py",
    "scripts/feedback_test.py",
    "scripts/verification_test.py",
    "scripts/audit_test.py",
    "scripts/front_service_test.py",
    "scripts/resilience_test.py",

    # Reports
    "README.md",
    "SECURITY_REMEDIATION_REPORT.md",
    "ML_VALIDATION_REPORT.md",
    "MODEL_CARD.md",
    "DATA_GOVERNANCE.md",


    # Verification service (case handling, auth)
    "src/verification_service/main.py",

    # Frontend JS (CSP-compliant)
    "src/front_service/static/app.js",
    "src/front_service/static/fraud-report-app.js",
    "src/front_service/static/monitor-app.js",
    "src/verification_service/static/app.js",
    "src/audit_service/static/app.js",
]

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    count = 0
    missing = []
    for f in security_files:
        if os.path.exists(f):
            zf.write(f, f)
            count += 1
        else:
            missing.append(f)

    # Add MANIFEST
    lines = [
        f"PS-14 Security Audit Zip - {count} files",
        "",
        f"Missing: {len(missing)}",
    ]
    for m in missing:
        lines.append(f"  - {m}")
    lines.append("")
    lines.append(f"{'Size':>8s}  Path")
    lines.append("")
    for info in zf.infolist():
        if info.filename != "MANIFEST.txt":
            lines.append(f"  {info.file_size:>8d}  {info.filename}")
    zf.writestr("MANIFEST.txt", "\n".join(lines))

sz = os.path.getsize(zip_name)
print(f"Created {zip_name}: {count} files, {sz / 1024:.0f} KB")
if missing:
    print(f"Missing: {missing}")
