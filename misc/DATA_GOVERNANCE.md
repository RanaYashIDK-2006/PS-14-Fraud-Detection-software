# PS-14 Data Governance

**Date:** 2026-08-24
**Compliance:** DPDP Act 2023, RBI Guidelines

---

## 1. PII Classification

### PII Fields (Identity DB Only)
| Field | Storage | Encryption | Access |
|-------|---------|------------|--------|
| full_name | Text | Plaintext (identity boundary) | Owner, Admin |
| phone_encrypted | LargeBinary | AES-256 (Fernet) | Owner, Admin |
| email_encrypted | LargeBinary | AES-256 (Fernet) | Owner, Admin |
| email_hash | String(64) | SHA-256 (blind index) | Identity service only |
| address_encrypted | LargeBinary | AES-256 (Fernet) | Owner, Admin (break-glass) |
| account_number | VARCHAR(20) | Plaintext | Owner, Admin |
| account_type | VARCHAR(20) | Plaintext | Owner, Admin |

### PII Rules
1. **Never leaves Identity Service** — PII is not sent to Privacy Layer, Risk Engine, Verification, or Audit
2. **Never in features** — ML features contain no PII
3. **Never in audit events** — Audit chain stores fraud_id, not user_id
4. **Never in logs** — PII is masked in all log output
5. **Never in model artifacts** — Training data contains no PII
6. **Break-glass only** — Identity resolution requires admin auth, reason code, rate limiting

---

## 2. Sensitive Financial Data

### Raw Transaction Amounts
- **NOT stored** in feature DB (only ratio-based values)
- **NOT stored** in risk DB (only scores)
- **NOT stored** in audit events (only pseudonymous event_id)
- **Ratio-based features:** `amount_ratio`, `avg_txn_amount_90d` (EMA of ratios)

### Schema Tests
`scripts/privacy_test.py` verifies:
- ✓ No `raw_amount` column in feature store
- ✓ No `transaction_amount` column in feature store
- ✓ `avg_txn_amount_90d` stores ratio-based values
- ✓ No raw amounts in fraud profiles

---

## 3. Feature Classifications

### SAFE DERIVED FEATURES (in ML vector)
| Feature | Classification | Rationale |
|---------|---------------|-----------|
| amount_ratio | DERIVED | Ratio of event amount to historical median |
| txn_freq_last_24h | AGGREGATE | Count of pre-event transactions |
| txn_time_unusual | DERIVED | Boolean flag based on typical hours |
| new_device_flag | DERIVED | Boolean flag based on known devices |
| unusual_location_flag | DERIVED | Boolean flag based on usual locations |
| unusual_recipient_flag | DERIVED | Boolean flag based on usual recipients |
| failed_auth_count_24h | AGGREGATE | Count of pre-event auth failures |
| days_since_last_similar_txn | AGGREGATE | Time delta to most recent similar txn |
| gradual_escalation_score | DERIVED | Slope of log(amount_ratio) trend |
| known_device_count | AGGREGATE | Count of registered devices |
| account_tenure_days | AGGREGATE | Days since profile creation |
| hour_of_day | SAFE | Transaction timestamp hour |
| is_weekend | SAFE | Transaction timestamp day-of-week |
| shared_device_accounts | AGGREGATE | Link analysis: other accounts on device |
| shared_recipient_accounts | AGGREGATE | Link analysis: other accounts with recipient |
| mule_ring_score | DERIVED | Composite of sharing signals |

### PROHIBITED RAW SENSITIVE FIELDS
- `raw_amount`, `transaction_amount`, `actual_amount`
- `email`, `phone`, `address`, `full_name`
- `user_id`, `account_number`, `ssn`
- `chargeback`, `dispute_outcome`, `verification_result`
- `label`, `is_fraud`, `was_fraud`

---

## 4. Retention Policy

### Data Retention
| Store | Retention | Deletion |
|-------|-----------|----------|
| identity.db (DB-1) | Indefinite | Manual (admin) |
| features.db (DB-2) | 90 days | Automated (data_retention.py) |
| risk.db (DB-3) | 90 days | Automated |
| audit.db (DB-4) | 1 year | Append-only; export before deletion |
| verify.db (DB-5) | 90 days | Automated |

### Deletion Policy
- User requests deletion → Identity service marks user as deleted
- PII is encrypted at rest; deletion = key destruction
- Pseudonym mapping retained for audit trail
- Feature vectors retained for 90 days (pseudonymous only)

---

## 5. Access Controls

### Role-Based Access Control (RBAC)
| Role | Permissions |
|------|-------------|
| user | own profile, own fraud_id |
| analyst | pseudonymous risk data, cases, audit trail |
| admin | user management, role assignment, system config |
| internal service | inter-service communication via internal token |
| compliance | audit trail read, break-glass (with logging) |

### Endpoint Authorization Matrix
| Endpoint | user | analyst | admin | internal | compliance |
|----------|------|---------|-------|----------|------------|
| POST /auth/register | ✅ | ✅ | ✅ | ❌ | ❌ |
| POST /auth/login | ✅ | ✅ | ✅ | ❌ | ❌ |
| GET /me/profile | ✅(own) | ❌ | ✅ | ❌ | ❌ |
| PUT /me/profile | ✅(own) | ❌ | ❌ | ❌ | ❌ |
| GET /admin/users | ❌ | ❌ | ✅ | ❌ | ❌ |
| POST /admin/assign-role | ❌ | ❌ | ✅ | ❌ | ❌ |
| POST /admin/db-query | ❌ | ❌ | ✅ | ❌ | ❌ |
| POST /internal/resolve-fraud-id | ❌ | ❌ | ❌ | ✅ | ❌ |
| GET /compliance/trail | ❌ | ❌ | ❌ | ❌ | ✅ |
| GET /analyst/pseudonymous-users | ❌ | ✅ | ✅ | ❌ | ❌ |

---

## 6. Pseudonymization

### How It Works
1. User registers → `user_id` (UUID) created
2. `fraud_id` (CSPRNG, 16 chars) generated — not derived from any PII
3. `pseudonym_mapping` table links fraud_id ↔ user_id
4. All downstream systems use fraud_id only
5. Identity resolution requires break-glass authorization

### Fraud ID Properties
- **CSPRNG-generated** — not a hash of PII
- **Rainbow-table resistant** — random, not deterministic
- **Rotation groups** — fraud_ids can be rotated periodically
- **No reverse path** — without access to identity.db, fraud_id cannot be resolved

---

## 7. Audit Requirements

### Audit Trail (DB-4)
Every privacy-sensitive action is logged:
- `feature_ingested` — Privacy Layer ingest (event_id only)
- `score_generated` — Risk Engine scoring (fraud_id, event_id, score)
- `verification_resolved` — Verification outcome (fraud_id, event_id, outcome)
- `fraud_id_resolved` — Break-glass identity resolution (actor, reason, case_id)
- `role_changed` — Admin role assignment
- `admin_db_query` — Admin DB access (actor, db, query_name, row count)

### Hash Chain Integrity
- Each entry includes: seq, prev_hash, entry_hash, timestamp
- Chain verification: recompute hashes from genesis
- Tamper detection: any modification breaks the chain
- Export: HMAC-SHA256 signed, independently verifiable

---

## 8. Deletion Rights (DPDP Act 2023)

### Right to Erasure
1. User requests deletion via admin
2. Identity service marks user as deleted
3. PII fields are zeroized (encryption key for that user destroyed)
4. Pseudonym mapping retained for audit trail (pseudonymous only)
5. Feature vectors retained for 90 days (pseudonymous only)
6. Audit events retained for 1 year (pseudonymous only)

### What Cannot Be Deleted
- Audit trail entries (append-only, legally required)
- Pseudonym mapping (needed for audit trail integrity)
- Model training data that has already been used (model cannot be "untrained")

---

## 9. Security Controls

### Encryption at Rest
- PII: AES-256 via Fernet (envelope encryption)
- Database files: OS-level encryption (recommended)
- Model artifacts: No encryption (no PII)

### Encryption in Transit
- All internal services: HTTP (localhost only)
- Production: HTTPS with TLS 1.3
- Inter-service: Internal token authentication

### Authentication
- User: Argon2id password hashing, JWT tokens
- Admin: Passphrase + optional TOTP
- Internal: Shared token (constant-time comparison)
- Compliance: Passphrase-gated

### Rate Limiting
- Login: 10 failures / 15 min lockout
- Break-glass: 10 resolutions / hour
- Admin DB queries: TOTP required if enabled
