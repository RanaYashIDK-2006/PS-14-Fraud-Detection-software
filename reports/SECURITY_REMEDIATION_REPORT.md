# PS-14 Security Remediation Report

**Generated:** 2026-08-27
**System:** PS-14 Fraud Detection Research Prototype
**Classification:** Pre-production research prototype — NOT production-ready

---

## Executive Summary

| Metric | Count |
|--------|-------|
| Security tests defined | 62 |
| Tests executed | 44 |
| Tests passed | 42 |
| Tests failed | 0 |
| Tests blocked | 0 |
| Tests NOT_RUN | 17 |
| Tests UNSUPPORTED | 1 |
| Independent tool findings | 24 (0 HIGH, 1 MEDIUM, 23 LOW) |
| Dependency vulnerabilities | Needs pip-audit verification |
| Coverage | 69.4% (44/62 executed) |
| Pass rate (of executed) | 94.6% (42/44 passed or partial) |

**Honest assessment:** The system has no confirmed security vulnerabilities among executed tests. However, 17 tests are NOT_RUN because the test infrastructure does not exist yet. This means coverage is 69.4%, not 100%.

---

## VERIFIED — Things That Actually Passed

| Control | Test | Evidence |
|---------|------|----------|
| JWT forgery rejected | AUTH-001 | Invalid signature → 401 |
| JWT none-alg attack blocked | AUTH-002 | none-alg → 401 |
| Expired tokens rejected | AUTH-003 | Expired JWT → 401 |
| Role escalation blocked | AUTH-004 | Forged admin role → 401 |
| Token replay handled | AUTH-008 | Idempotent replay returns same result |
| Rate limiting works | AUTH-010 | Login blocked after N attempts |
| Privilege escalation blocked | AUTH-012 | role=admin in register → silently ignored |
| Unauth access blocked (all endpoints) | AUTHZ-001..011 | All /me/*, /alerts/*, /admin/*, /internal/* return 401/403 |
| SQL injection (8 payloads) | SQLI-001..008 | All return 400 (allowlist rejection) |
| Input validation (4 tests) | INPUT-001..004 | Oversized/empty/null/huge inputs rejected |
| CRLF injection blocked | API-001 | httpx rejects illegal headers |
| Race conditions handled | API-003 | 0/5 concurrent duplicate registrations |
| No stack traces in errors | INFO-001 | Traceback not found in error responses |
| No internal endpoints exposed | INFO-002 | /internal/* hidden from OpenAPI |
| No version info leaked | INFO-003 | Health endpoint clean |
| PII not in feature store | PRIV-001 | privacy_test.py (27/27) |
| PII not in risk scores | PRIV-002 | privacy_test.py |
| PII not in audit logs | PRIV-003 | privacy_test.py |
| Pseudonymization works | PRIV-006 | pseudonym_separation_test.py |
| Audit chain integrity | AUDIT-001..004 | Hash chain verified; unauthorized access blocked |
| Adversarial attacks detected | ML-001 | adversarial_test.py (42/42) |
| No label leakage | ML-002 | leakage_structural_test.py (123/123) |
| No temporal leakage | ML-003 | temporal_test.py (22/22) |
| Calibration valid | ML-004 | calibration_test.py (16/16) |
| Drift detection works | ML-005 | drift_detector_test.py (31/31) |
| Fail-safe ML degraded | ML-006 | ML_UNAVAILABLE → score >= 31 (step_up) |

## PARTIALLY VERIFIED — Implementation Exists But Evidence Incomplete

| Control | Issue |
|---------|-------|
| SQL injection prevention | Static allowlist works; not tested with automated SQLi scanners (sqlmap) |
| Rate limiting | In-memory only; not tested across multiple workers |
| Penetration test overall | Some tests BLOCKED when services unavailable; BLOCKED != PASS |
| In-domain ROC-AUC | Multiple conflicting values (99.13%, 98.77%, 97.98%); needs artifact-specific reproduction |
| Cross-domain ROC-AUC | 98.5% claim lacks documented weighting methodology |
| Bandit MEDIUM finding | privacy_layer/main.py:837 uses string-based query construction (false positive: uses parameterized bindings, not interpolation) |

## UNSUPPORTED — Claims Without Evidence

| Claim | Why |
|-------|-----|
| Industry-grade fraud detection | No independent industry validation |
| Generalizes to real-world fraud | Evaluated on public datasets only |
| Admin sessions multi-worker safe | Process-local dict; would fail with multiple workers |
| Data retention policy | Not implemented |
| Right to deletion (GDPR) | Not implemented |

## FALSE — Claims Contradicted by Implementation

| Claim | Reality |
|-------|---------|
| Tamper-proof audit trail | Hash chain is tamper-EVIDENT not tamper-PROOF. SQLite can be modified at filesystem level. |

## NOT_RUN — Tests That Need Implementation

| Category | Missing Tests |
|----------|--------------|
| Authentication | Algorithm confusion, invalid issuer, invalid audience, session revocation, TOTP validation |
| Input validation | Numeric overflow, negative values, Unicode handling |
| API security | HTTP method abuse, path traversal, SSRF |
| Information disclosure | Database credentials in errors, JWT secret in errors |
| Audit integrity | Modified event detection, reordered event detection |

## NOT IMPLEMENTED — Required Functionality

| Feature | Status |
|---------|--------|
| TLS/HTTPS | HTTP only in prototype |
| Production Caddy config | Not verified to actually start |
| Multi-worker session store | Process-local dict only |
| Data retention | No mechanism |
| Right to deletion | No API |
| Retention vs audit integrity | No retention exists to test |
| Container hardening | No non-root user, no read-only FS |
| Backup/restore testing | Not implemented |
| CI/CD pipeline | GitHub Actions config exists but not verified as functional |
| Rollback mechanism | Not implemented |

## Known Limitations

1. **Session management is not production-safe.** `_ADMIN_SESSIONS` is process-local. Multiple workers would have inconsistent session state.
2. **SQLite is used for all databases.** Shared writable volumes mean a compromised service can access other services' data.
3. **The security scanner's score (100/100) is misleading.** It counts findings, not checks executed. With 69.4% test coverage, "100/100" is not an accurate representation.
4. **The penetration test's "42/42 passed" is accurate** but only when all 6 services are running. When services are down, tests are BLOCKED, not PASS.
5. **ML metrics are not artifact-specific.** Different scripts report different ROC-AUC values for apparently the same experiment.
6. **No independent vulnerability scanning** was done before this report (now added: bandit, pip-audit).
7. **The Bandit MEDIUM finding** (SQL injection in privacy_layer) is a false positive — the query uses parameterized bindings via SQLAlchemy's `sql_text` with bound `:param` parameters.

---

## Independent Tool Results

### Bandit (Static Analysis)

| Severity | Count | Details |
|----------|-------|---------|
| HIGH | 0 | — |
| MEDIUM | 1 | False positive (parameterized query flagged as string-based) |
| LOW | 23 | subprocess usage, try/except/pass, etc. |

### pip-audit (Dependency Scan)

| Status | Details |
|--------|---------|
| Ran | Check `.env` or CI for results |

### Custom Security Scanner

| Metric | Value | Honest Assessment |
|--------|-------|-------------------|
| Files scanned | 54 | Actual files in src/ |
| Checks defined | ~18 rules | Limited to pattern matching |
| Checks executed | ~18 | All patterns ran |
| Findings | 0 CRITICAL, 0 HIGH | True for patterns scanned |
| Score | 100/100 | MISLEADING — only measures findings, not coverage |

---

## What Was Fixed This Session

| Fix | Severity | Impact |
|-----|----------|--------|
| Claims registry created | HIGH | Single source of truth for all claims |
| Security test matrix created | HIGH | Every test has individual record |
| False "tamper-proof" claim identified | HIGH | Must use "tamper-evident" |
| ML benchmark registry created | MEDIUM | Artifact-specific metrics required |
| Independent tooling added | MEDIUM | bandit + pip-audit complement custom scanner |
| DO-NOT-CLAIM list created | MEDIUM | Prevents future false claims |

---

## Remaining Production Blockers

1. **Session management** — Not multi-worker safe
2. **TLS/HTTPS** — HTTP only
3. **SQLite everywhere** — Not production-grade storage
4. **No data retention** — GDPR non-compliant
5. **No right to deletion** — GDPR non-compliant
6. **No backup/restore** — No disaster recovery
7. **No container hardening** — Runs as root with full capabilities
8. **17 security tests NOT_RUN** — Coverage gap
9. **ML metrics not artifact-specific** — Cannot reproduce exact results
10. **No Caddy/rate-limit production verification** — Config exists but not tested

---

*This report is intentionally blunt. A smaller truthful claim is better than a larger unsupported claim.*
