# PS-14 — CHECK #25: PRIVACY, ACCESS CONTROL & DATA GOVERNANCE AUDIT

**Date:** 2026-09-03
**Method:** Static implementation inspection of every route handler in all six services (risk 8003, privacy 8002, identity 8001, verification 8004, audit 8005, front 8000) + SQLite schema review of all four stores + masked regex secret scan over the whole tree + live `pip-audit` vulnerability scan. No live HTTP probes (services were not running); server-side auth was verified by reading the actual handlers, not docs.
**Verdict: FAIL** — but the failure is *not* where a first-pass guess would put it.

---

## What is genuinely strong (verified by code, not config)

- **All 15 `/internal/*` endpoints** (risk ×7, privacy ×7, identity ×1) require the `X-Internal-Token` header and verify it with `hmac.compare_digest` against `settings.internal_token`. No internal endpoint is open.
- **Server-side authorization everywhere**: identity uses a hierarchical `PERMISSION_MAP` (user < analyst < admin) with `require_role`/`require_permission` FastAPI dependencies; self-registration hardcodes `role="user"` — clients cannot grant themselves roles. Front admin endpoints all depend on `_require_admin` (session JWT bound to the stored scrypt passphrase hash, revocable server-side). `/admin/db-query` (raw SQL) is admin-gated and audit-logged.
- **No live secrets in the repository.** Secrets default to empty and are randomized per process; every service loads `.env`; `.env`, `db/`, `certs/` are gitignored. The production gate (`PS14_MODE=production`, the *default*) aborts startup if any secret is missing from the environment or CORS is still localhost. The actual `.env` secret values were compared by SHA-256 against the documented dev defaults and **differ** — the "change-me" strings are not live.
- **Pseudonymous audit chain**: all nine `append_audit_event` call sites carry only `fraud_id`, `event_id`, scores, decisions and IDs — no amounts, cards, emails or names. DB-2 stores derived features (device hashed keyed+truncated; no raw amounts); DB-3 stores scores only.
- **Query parameterization**: ORM everywhere; the one raw-SQL search (`audit_search`) uses bound `:params`.
- **Brute-force protection**: rate limiting middleware on identity + login lockout after 10 failures; front admin login has its own IP-based throttle.
- **Security headers/CSP** (production-strict) + 10 MB body cap applied to every service.

## Findings (matrix in `reports/privacy_access_governance.json`)

| # | Finding | Severity |
|---|---------|----------|
| 1 | **DB-1 plaintext PII at rest.** `identity.db.users` encrypts `phone/email/address` (Fernet `LargeBinary`) but stores `full_name`, `account_number`, `account_type`, `kyc_doc_ref` **plaintext** — contradicting the code comment *"PII is column-encrypted"* | HIGH |
| 2 | **No data-retention policy or deletion path** for identity/features/risk stores; demo state, reports and `db/*.db.bak` accumulate indefinitely (audit chain is append-only by design and exempt, but nothing documents the rest) | MEDIUM |
| 3 | **Anonymous operational endpoints on the public front service** (`:8000`): `/security/scan-history`, `/security/scan-latest` (inconsistent with admin-gated `/security-scan`), `/monitor/metrics`, `/monitor/unified`, plus scenario reports with amounts (`/batch`, `/real-cases`, `/fraud-report`) | MEDIUM |
| 4 | **setuptools 78.1.0 → 4 known CVEs** (`PYSEC-2025-49`, `PYSEC-2026-3447`; fixes 78.1.1/83.0.0). Dev/build-time, not a runtime service dependency — but unpatched. **Scanner completeness:** 141/142 packages audited; `torch==2.13.0+cpu` could not be audited (off-PyPI local build) and is reported, not hidden | MEDIUM |
| 5 | Dev-default `ps14-dev-*-change-me` literals hardcoded in `scripts/real_cases.py` + walkthrough scripts and documented in `.freebuff/security_audit.md` (inert today, but a trap if an operator copies them) | MEDIUM |
| 6 | Legacy **shared compliance token** fallback still accepted by audit_service's compliance gate alongside per-analyst passphrases | LOW |
| 7 | uvicorn default access logs record query strings (contains `fraud_id`) | LOW |
| 8 | Unused dev TLS private key at `certs/ps14.key` (gitignored, referenced by no code) | LOW |
| 9 | `db/security_report.json` (2026-08-22): **`checks_performed: 0` yet `passed: true`** — a zero-check scan persisted as a pass | MEDIUM |

## Honest non-claims

- **TLS in transit: NOT TESTED.** The stack is loopback HTTP by design; no in-process TLS exists and no proxy config is in this repo. Production transport security is unverifiable here and is not claimed.
- **PostgreSQL least-privilege: PARTIAL.** SQLite gives per-file isolation for free; the Postgres path uses a single shared `DATABASE_URL` (per-service *schemas* are enforced by the production gate when `SERVICE_NAME` is set, but per-service *roles/credentials* do not exist).

## Bottom line

Authentication, authorization, secrets handling and audit-data pseudonymization are implemented properly and survived real inspection — no bypass, no live secret, no UI-only gate. The FAIL is data-governance, exactly the category that matters when this stops being a prototype: **PII at rest is only partially encrypted (fix #1), there is no retention regime (fix #2), the public front page exposes operational endpoints (fix #3), and the dependency tree carries known CVEs (fix #4).** Each has a concrete, small fix; none is a redesign.

**Files:** `reports/privacy_access_governance.json`, `reports/dependency_scan.txt`, this report. Prior live penetration evidence: `reports/PENTEST_RESULTS.json`, `reports/SECURITY_TEST_MATRIX.json`.
