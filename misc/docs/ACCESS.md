# PS-14 — Private Access Information

**This document is encrypted at rest.** The committed copy is
`docs/ACCESS.md.enc` (Fernet ciphertext). Open it with:

```bash
python scripts/access_doc.py view            # prompts for the passphrase
ACCESS_DOC_KEY=... python scripts/access_doc.py view -o ACCESS.md
```

Rotate the document's key with `python scripts/access_doc.py rekey`.

> These are the **prototype dev values** (also the defaults in
> `src/settings.py`, env-overridable via `.env`). Production must replace
> every shared token/passphrase below with per-service mTLS, short-lived
> RBAC credentials, and KMS-managed keys (architecture section 3) — the
> prototype's shared tokens are for local development only.

---

## 1. Service map

| # | Service | Port | URL | Role | Reachable from |
|---|---|---|---|---|---|
| 1 | Identity Service | 8001 | http://127.0.0.1:8001 | PII owner, JWT issuer | user-facing |
| 2 | Privacy Layer | 8002 | http://127.0.0.1:8002 | pseudonymous feature store | internal only |
| 3 | Risk Engine | 8003 | http://127.0.0.1:8003 | scoring + rules + limits | internal only |
| 4 | Verification Service | 8004 | http://127.0.0.1:8004 | alerts UI + BFF + compliance proxy | user-facing |
| 5 | Audit Service | 8005 | http://127.0.0.1:8005 | hash-chained trail + compliance viewer | compliance role |

Start all five: `powershell -ExecutionPolicy Bypass -File .freebuff/start_stack.ps1`
(Windows) or five `uvicorn` commands from the README. Health: `GET /health`
on every service.

## 2. Credentials (dev)

| Secret | Value (dev) | Where it lives | Used by |
|---|---|---|---|
| JWT secret | `ps14-dev-secret-change-me` | `JWT_SECRET` env / `settings.jwt_secret` | Identity + Verification (Bearer tokens) |
| Internal (service-to-service) token | `ps14-dev-internal-token-change-me` | `INTERNAL_TOKEN` env / `settings.internal_token` | every `/internal/*` endpoint |
| Compliance passphrase | `ps14-dev-compliance-token-change-me` | `COMPLIANCE_TOKEN` env / `settings.compliance_token` | `/compliance/*` viewers on :8004/:8005 |
| Break-glass resolution | internal token (above) + `X-Audit-Actor` header | — | `POST /internal/resolve-fraud-id` on :8001 |
| Export signing key | derived from `JWT_SECRET` (HMAC-SHA256) | `settings.export_signing_key` | `GET /audit/export` signature |
| JWT expiry | 15 minutes | `JWT_EXPIRY_MINUTES` | login sessions |

The export signature key is **derived from the JWT secret** — rotating the
JWT secret invalidates previously issued export signatures and re-keys the
Fernet PII encryption (`settings.fernet_key`), so it must be a coordinated
rotation, not a solo change.

## 3. Data-store access matrix (every read passage)

| Store | DB file | Owned by | Holds | Passage | Credential |
|---|---|---|---|---|---|
| DB-1 | `db/identity.db` | Identity (:8001) | PII, credentials, `pseudonym_mapping` | `POST /auth/register` · `/auth/login` | none |
| DB-1 | “ | “ | “ | `GET /me/fraud-id` | Bearer JWT (own pseudonym only) |
| DB-1 | “ | “ | “ | `POST /internal/resolve-fraud-id` (break-glass, logged) | internal token — the ONLY pseudonym→person passage |
| DB-2 | `db/features.db` | Privacy (:8002) | feature vectors, profiles, device fingerprints | `POST /internal/ingest-transaction` · `/internal/commit-baseline` · `/internal/demo-age-account` | internal token |
| DB-2 | “ | “ | “ | `GET /internal/device-graph` (also `:8004/compliance/device-graph`) | internal token / compliance passphrase |
| DB-3 | `db/risk.db` | Risk (:8003) + Verify (shared, §13) | `risk_scores`, `verification_outcomes` | `POST /internal/evaluate` · `/internal/attribution` | internal token |
| DB-3 | “ | “ | “ | `GET /alerts` · `/alerts/{id}/confirm` · `/alerts/{id}/reason` · `/cases` · `/history` · `/overview` | Bearer JWT (own rows only) |
| DB-3 | “ | “ | “ | `POST /demo/seed` | Bearer JWT (DEV ONLY) |
| DB-4 | `db/audit.db` | Audit (:8005) | hash-chained `audit_events` | `GET /audit/events` · `/audit/integrity` · `/audit/overview` · `/audit/export` | internal token (+ `X-Audit-Actor` on reads) |
| DB-4 | “ | “ | “ | `GET /compliance/events` · `/compliance/integrity` (+ :8004 `/compliance/*` proxies) | compliance passphrase |

**Three human passages:** account holder via **JWT** (own rows), compliance
role via the **passphrase** (pseudonymous trail), and **break-glass** as the
single pseudonym↔person resolution. Nothing else reads PII or raw amounts.

## 4. How to get in (quick procedures)

- **Watch the live UI:** open http://127.0.0.1:8004/ (alerts) and
  http://127.0.0.1:8005/ (compliance trail). Register/login on :8004 to get
  a Bearer JWT; the compliance views ask for the compliance passphrase.
- **Seed a demo account:** on :8004, after login click the demo button →
  "Seeded 7 events · baseline committed 5/6 · 1 high-risk alert".
- **Compliance trail with hash verification:** :8005 (or :8004 compliance
  view) — per-event hash linkage, chain-integrity banner, device-graph
  lookup, signed export at `GET /audit/export` (internal token).
- **Break-glass (pseudonym → person):** `POST /internal/resolve-fraud-id`
  with the internal token; always audit-logged; returns masked PII.
- **Full walkthrough:** `scripts/live_walkthrough.sh` boots all five
  services and prints the register-to-audit trail.

## 5. Rotation & incident notes

- **Tokens are shared secrets** (dev prototype): treat `INTERNAL_TOKEN` and
  `COMPLIANCE_TOKEN` as blast-radius-critical — a leaked compliance
  passphrase exposes the whole pseudonymous trail; a leaked internal token
  exposes every store's service API.
- **Audit chain is tamper-evident:** `GET /audit/integrity` recomputes the
  chain from genesis and reports the first bad seq — use it before trusting
  any export or after any suspected compromise.
- **DB files are not encrypted at rest** in the prototype (only the access
  document is). Production: per-store KMS keys + encrypted volumes.
- Rotating `JWT_SECRET` cascades to Fernet PII keys and export signatures —
  coordinate (see §2).
