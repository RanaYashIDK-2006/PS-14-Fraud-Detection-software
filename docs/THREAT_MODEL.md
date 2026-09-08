# PS-14 Threat Model

## Scope

This document models attackers against the PS-14 Privacy-First AI Fraud
Detection System and documents existing protections, remaining risks, and
mitigations.

**System boundary**: Six services (Front, Identity, Privacy, Risk,
Verification, Audit) communicating over HTTP with shared `X-Internal-Token`.
Four SQLite databases (DB-1 through DB-4) on a shared volume.

**Trust boundaries**:
- External users → Front page / Verification UI / Identity Service
- Internal services → Privacy / Risk / Audit (service-to-service via token)
- Compliance role → Audit trail viewer (passphrase-gated)
- Admin role → DB Explorer (JWT + passphrase + TOTP)

---

## Threat 1: Steal the fraud database (DB-2/DB-3)

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker gains read access to `features.db` or `risk.db` |
| **Impact** | Exposes pseudonymous feature vectors and risk scores — not PII |
| **Likelihood** | Medium (shared volume in prototype; network-isolated in production) |
| **Existing protection** | DB-2 contains only derived §16 features (no raw amounts, no device IDs, no PII). DB-3 contains only risk scores + pseudonymous fraud IDs. `pseudonym_separation_test.py` proves DB-2 compromise doesn't reveal identity. |
| **Remaining risk** | Cross-referencing pseudonymous feature vectors with external behavioral data could enable re-identification if k-anonymity is weak. |
| **Mitigation** | k-anonymity gate on exports (`k_anonymity_check.py`); production PostgreSQL with per-store credentials + TLS + network isolation |

---

## Threat 2: Steal the identity database (DB-1)

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker gains read access to `identity.db` |
| **Impact** | Exposes encrypted PII + pseudonym_mapping (fraud_id ↔ user) |
| **Likelihood** | Low (highest-protected store; break-glass only) |
| **Existing protection** | PII encrypted with AES-256 (Fernet) via `fernet_key` derived from `JWT_SECRET`. Login uses blind index (SHA-256 HMAC) so plaintext email is never queried. Argon2id password hashing. |
| **Remaining risk** | If `JWT_SECRET` is compromised, all PII is decryptable. KMS envelope encryption would limit blast radius. |
| **Mitigation** | Production: KMS-backed envelope encryption with per-tenant keys; rotate `JWT_SECRET` as coordinated event; keep DB-1 on isolated network segment |

---

## Threat 3: Compromise an internal service

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker gains code execution on one service (e.g., Privacy Layer) |
| **Impact** | Can call other services using the shared `X-Internal-Token` |
| **Likelihood** | Low (container isolation in production) |
| **Existing protection** | Services have least-privilege data access (each owns its DB). Privacy Layer can't read PII. Risk Engine can't write to DB-1. Audit writes are append-only with hash chain. |
| **Remaining risk** | Shared `INTERNAL_TOKEN` means a compromised service can call any internal endpoint. No per-service credential isolation. |
| **Mitigation** | **Production**: per-service mTLS + short-lived RBAC credentials (architecture §3/§4). Currently the top remaining backend-security gap. |

---

## Threat 4: Compromise an administrator

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker obtains admin passphrase |
| **Impact** | Can query all five databases via `/admin/db-query`, view encrypted essentials |
| **Likelihood** | Low (passphrase-gated, TOTP now mandatory for DB queries) |
| **Existing protection** | Admin passphrase stored as scrypt hash only. TOTP 2FA mandatory for DB queries. All queries audit-logged to hash chain. Write operations blocked. 500-row cap. |
| **Remaining risk** | If both passphrase AND TOTP secret are compromised, attacker has read access to all DBs for as long as the session lives. |
| **Mitigation** | Short admin session expiry; admin passphrase rotation revokes all sessions; audit trail provides forensic evidence |

---

## Threat 5: Manipulate transactions

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker submits crafted transactions to influence risk scores |
| **Impact** | Could cause false negatives (missed fraud) or false positives (blocked legit users) |
| **Likelihood** | Medium (ongoing in production) |
| **Existing protection** | Privacy Layer strips raw data, stores only derived features. Velocity limits enforce hard caps. Idempotent on `event_id` (replays rejected). Rules engine + ML fusion. |
| **Remaining risk** | Adversarial feature manipulation (e.g., submitting transactions that produce benign-looking §16 features while hiding fraud signals) |
| **Mitigation** | Model uncertainty monitoring; OOD recall gate; drift detection; rules backtesting |

---

## Threat 6: Replay requests

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker captures and replays a valid request |
| **Impact** | Double-processing, hash chain pollution |
| **Likelihood** | Low (HTTPS in production; token expiry) |
| **Existing protection** | Idempotent on `event_id` — duplicates return stored result with `idempotent_replay: true` without re-auditing. JWT tokens expire in 15 minutes. |
| **Remaining risk** | If event_id is predictable, an attacker could pre-claim an event_id |
| **Mitigation** | event_id is 8-64 chars from the caller; production should use UUIDs |

---

## Threat 7: Manipulate model inputs

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker submits feature vectors that cause the model to misclassify |
| **Impact** | False negatives (missed fraud) |
| **Likelihood** | Low (features derived server-side, not user-supplied) |
| **Existing protection** | §16 features are derived by the Privacy Layer from transaction context, not supplied by the caller. Rules engine provides deterministic backup. |
| **Remaining risk** | If Privacy Layer derivation has bugs, attacker-controlled inputs could skew features |
| **Mitigation** | `smoke_test.py` asserts feature derivation; `feature_compatibility_test.py` prevents training/serving skew |

---

## Threat 8: Poison feedback / training data

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker submits false "this was me" / "this wasn't me" responses to corrupt training labels |
| **Impact** | Degraded model performance over time |
| **Likelihood** | Low-Medium (users can confirm/dispute their own alerts) |
| **Existing protection** | Feedback pool gated on minimum outcomes. Versioned snapshots. OOD recall gate blocks regressed models. No auto-deploy from individual feedback. |
| **Remaining risk** | A sophisticated attacker could systematically poison feedback for their own account |
| **Mitigation** | `--min-outcomes` threshold; `disputed` events never enter behavioral baseline; retrain requires human approval |

---

## Threat 9: Enumerate Fraud IDs

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker guesses or brute-forces fraud IDs |
| **Impact** | Could access another user's alerts or trigger false verification |
| **Likelihood** | Very Low (80-bit CSPRNG space = ~1.2 × 10²⁴ possibilities) |
| **Existing protection** | 16-char base32-style IDs from CSPRNG (`secrets.choice`). Not derived from PII. Rate limiting on all endpoints. |
| **Remaining risk** | 80-bit space is smaller than ideal 128-bit (self-flagged limitation) |
| **Mitigation** | Production: 128-bit IDs (schema change); rate limiting; account lockout on repeated 404s |

---

## Threat 10: Abuse graph queries

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker uses device-graph or relationship queries to map the social network |
| **Impact** | Re-identification via relationship patterns |
| **Likelihood** | Low (compliance-token-gated) |
| **Existing protection** | `/compliance/device-graph` requires compliance passphrase. Returns only pseudonymous accounts + event counts. No raw device/recipient IDs exposed. |
| **Remaining risk** | No traversal-depth limits or query-rate limiting on graph endpoints |
| **Mitigation** | Add depth limiting and rate limiting on graph traversal; production RBAC |

---

## Threat 11: Abuse break-glass access

| Attribute | Assessment |
|---|---|
| **Attack** | Authorized user abuses break-glass to resolve pseudonymous IDs |
| **Impact** | Privacy violation — linking pseudonymous data to real identity |
| **Likelihood** | Low (internal token required, always logged) |
| **Existing protection** | Requires `X-Internal-Token`. Logged to both DB-1 local access log AND DB-4 hash chain. Actor + reason recorded. Masked PII returned. |
| **Remaining risk** | No automatic expiration, no case-ID requirement, no two-person approval |
| **Mitigation** | **Production**: add case-ID requirement, auto-expiry on resolutions, two-person approval for high-sensitivity lookups |

---

## Threat 12: Obtain logs / audit trail

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker gains access to audit logs |
| **Impact** | Could reveal pseudonymous decision patterns |
| **Likelihood** | Low (audit DB is append-only with hash chain) |
| **Existing protection** | Hash-chained, tamper-evident. `/audit/integrity` verifies chain. Export signed with HMAC-SHA256. `verify_export.py` independently re-verifies. |
| **Remaining risk** | Audit trail contains pseudonymous fraud_ids + decision patterns — could enable behavioral fingerprinting |
| **Mitigation** | Retention policy (365d via redaction-in-place); production: encrypt at rest, access-controlled |

---

## Threat 13: Intentionally trigger degraded mode

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker causes ML failures to force rules-only mode |
| **Impact** | Reduced detection quality (rules-only is weaker than fusion) |
| **Likelihood** | Low (circuit breaker has 3-failure threshold + 30s recovery) |
| **Existing protection** | Circuit breaker: 3 consecutive failures → open 30s → half-open probe. Degraded decisions tagged `degraded: true` in audit. Rules-only path still catches hard red flags. |
| **Remaining risk** | If attacker can reliably cause ML failures, they get a weaker detector |
| **Mitigation** | Monitor `degraded` flag frequency; alert on sustained degradation; rules backtesting ensures baseline detection |

---

## Threat 14: Obtain backups

| Attribute | Assessment |
|---|---|
| **Attack** | Attacker gains access to database backups |
| **Impact** | Same as Threats 1+2 depending on which backup |
| **Likelihood** | Low (backups are operational artifacts) |
| **Existing protection** | Backups contain the same encrypted/hashed data as live DBs |
| **Remaining risk** | Backup encryption not explicitly confirmed |
| **Mitigation** | Production: encrypt backups at rest; test restore procedure; off-site storage |

---

## Summary

| Threat | Likelihood | Impact | Protection Level |
|---|---|---|---|
| Steal fraud DB | Medium | Low | 🟢 Strong (pseudonymous only) |
| Steal identity DB | Low | High | 🟡 Good (encrypted, needs KMS) |
| Compromise service | Low | High | 🟡 Good (needs mTLS) |
| Compromise admin | Low | High | 🟢 Strong (TOTP mandatory) |
| Manipulate transactions | Medium | Medium | 🟢 Strong (derived features + rules) |
| Replay requests | Low | Low | 🟢 Strong (idempotent + JWT expiry) |
| Manipulate model inputs | Low | Medium | 🟢 Strong (server-side derivation) |
| Poison feedback | Low-Med | Medium | 🟢 Strong (gated + versioned) |
| Enumerate fraud IDs | Very Low | Low | 🟢 Strong (80-bit CSPRNG) |
| Abuse graph queries | Low | Medium | 🟡 Good (needs depth limits) |
| Abuse break-glass | Low | High | 🟡 Good (needs case-ID + expiry) |
| Obtain logs | Low | Medium | 🟢 Strong (hash-chained) |
| Trigger degraded mode | Low | Medium | 🟢 Strong (circuit breaker) |
| Obtain backups | Low | High | 🟡 Good (needs encryption confirm) |

---

## Production Dependencies (external, not yet wired)

1. **mTLS / per-service credentials** — replace shared `INTERNAL_TOKEN`
2. **KMS/HSM envelope encryption** — replace Fernet key derivation from `JWT_SECRET`
3. **Real TLS certificates** — replace self-signed for production
4. **External ML validation dataset** — replace synthetic-only evaluation
5. **Monitoring / alerting** — Prometheus/Grafana for service health + audit chain
6. **Backup encryption** — encrypt at rest, test restore
