# Security Remediation Report

**Generated:** 2026-08-24 17:24 UTC
**Test Results:** 8/8 tests passed

## Test Suite Results

| Test Suite | Exit Code | Status | Details |
|------------|-----------|--------|---------|
| privacy | 0 | PASS | 27 passed, 0 failed |
| leakage_structural | 0 | PASS | 97 passed, 0 failed |
| temporal | 0 | PASS | 22 passed, 0 failed |
| calibration | 0 | PASS | 16 passed, 0 failed |
| adversarial | 0 | PASS | 94 passed, 0 failed |
| production_gate | 0 | PASS | exit 0 |
| smoke | 0 | PASS | exit 0 |
| risk_engine | 0 | PASS | exit 0 |

## Security Findings

| Finding | Severity | Status | Evidence |
|---------|----------|--------|----------|
| Plaintext address column | HIGH | FIXED | privacy_test: address isolation verified |
| Raw SQL execution | MEDIUM | NOT VERIFIED | sql_injection_test: NOT RUN or FAILED |
| SQL identifier concatenation | MEDIUM | NOT VERIFIED | sql_injection_test: NOT RUN or FAILED |
| Raw amounts in feature store | MEDIUM | FIXED | privacy_test: no raw amounts in schema |

## Acceptance Criteria

- [x] No raw transaction amount is persisted in feature/risk storage
- [x] Address cannot leak outside the identity boundary
- [ ] Arbitrary SQL execution is impossible
- [ ] SQL identifiers use fixed internal mappings
- [x] Privacy tests pass from a clean environment
- [ ] SQL injection tests pass