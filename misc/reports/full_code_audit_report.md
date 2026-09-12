# PS-14 Full Code Audit Report
Generated: 2026-08-27T20:53:13.225556

## Executive Summary

- Total source files: 191
- Total lines of code: 62965
- Regression tests: 18/18 PASS
- Penetration tests: 42/42 PASS
- Security scan: 100/100 (Grade A)
- Claims verified: 14

## Code Metrics by Language

| Language | Files | Lines |
|----------|-------|-------|
| .py | 190 | 50672 |
| .md | 24 | 4331 |
| .js | 10 | 2191 |
| .json | 18 | 2010 |
| .html | 5 | 1647 |
| .yaml | 3 | 1338 |
| .yml | 4 | 776 |

## Code Metrics by Directory

| Directory | Files | Lines |
|-----------|-------|-------|
| ./ | 174 | 46751 |
| src/ | 64 | 14389 |
| dist/ | 8 | 940 |
| models/ | 5 | 458 |
| .github/ | 2 | 419 |
| .agents/ | 1 | 8 |

## Security Results

- Scan Score: 100/100 (Grade A)
- Penetration Tests: 42/42 PASS
- Regression Suite: 18/18 PASS
- Sql Injection: All payloads blocked
- Auth Bypass: All attempts blocked
- Xss: All attempts blocked
- Privilege Escalation: All attempts blocked
- Race Conditions: All blocked
- Header Injection: All blocked
- Info Disclosure: No stack traces in errors

## Test Coverage

- Regression Suites: 18
- Regression Passed: 18
- Penetration Tests: 42
- Penetration Passed: 42
- Security Scanner Checks: 54
- Adversarial Tests: 42
- Leakage Tests: structural + temporal + label
- Calibration Tests: Platt scaler validated

## Claims Audit

- Verified: 14
- Partially verified: 3
- Unsupported: 12
- False: 6

Key corrections:
- tamper-proof -> tamper-EVIDENT (hash chain detects but does not prevent)
- industry-grade -> UNSUPPORTED (only synthetic/cross-domain eval)
- enterprise-ready -> UNSUPPORTED (prototype system)
- production-ready -> UNSUPPORTED (pre-production prototype)

## Services Architecture

| Service | Port | Purpose |
|---------|------|---------|
| Front Service | 8000 | Landing page, admin, monitoring |
| Identity Service | 8001 | Auth, accounts, pseudonymization |
| Privacy Layer | 8002 | Feature extraction, PII protection |
| Risk Engine | 8003 | ML + rules scoring, decisions |
| Verification Service | 8004 | Alert verification, human review |
| Audit Service | 8005 | Hash-chained audit trail |

## ML Model

- Ensemble: LogisticRegression + RandomForest + XGBoost
- Calibration: Platt scaling
- Training Data: kaggle_fraud (real credit card transactions)
- Feature Count: 21
- Batch Throughput: 35K txn/s (2M rows in 57s)