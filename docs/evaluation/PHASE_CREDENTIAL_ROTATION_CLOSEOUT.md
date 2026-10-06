# PS-14 — Phase 4A (Credential Rotation and PII Re-Encryption) Closeout

**Phase:** 4A — Credential Rotation and PII Re-Encryption
**Status:** rotation and PII migration complete for every locally held credential; **one provider credential remains an owner action**
**Primary report:** [CREDENTIAL_ROTATION_AND_PII_MIGRATION.md](CREDENTIAL_ROTATION_AND_PII_MIGRATION.md)
**Machine-readable evidence:** [PHASE4A_CREDENTIAL_ROTATION_EVIDENCE.json](PHASE4A_CREDENTIAL_ROTATION_EVIDENCE.json)
**Predecessor:** [PHASE_STRIX_SECURITY_CLOSEOUT.md](PHASE_STRIX_SECURITY_CLOSEOUT.md) (Phase 4)

## Commit / SHA ledger

| Item | SHA |
|---|---|
| Branch | `main` |
| HEAD at phase start (Phase-4 push tip) | `0f2914c95078b1529f5f2159666c7bc4e98dab4f` |
| **Phase-4A code + evidence commit** (versioned PII cipher, migration tool and its tests, rotated configuration contract, report, evidence bundle) | **`cb504ea`** (pushed with the commit below as `0f2914c..28d0935`) |
| Phase-4A closeout commit (this document) | **`28d0935fb4b1ae0b4a4bd51999425d9bbbb94220`** |
| Final Git SHA of the phase | `28d0935fb4b1ae0b4a4bd51999425d9bbbb94220`; the CI outcome below was appended afterwards in a **documentation-only** commit, because a commit cannot contain its own CI result |
| Remote | `https://github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software.git` |

Git history was **not** rewritten. The previously exposed values remain recoverable from history, and that is
recorded below as part of the project's security history rather than fixed.

## Credentials affected (by type only)

No value is reproduced anywhere in this closeout, the report, the evidence bundle, any log or any test fixture.

| Credential type | Exposure | Rotation mechanism | Result |
|---|---|---|---|
| Authentication signing secret (HS256 JWT) | historically committed | application secret rotation | **ROTATED** |
| Internal service token (`X-Internal-Token`, all five services) | historically committed (a different literal than the live value) | application rotation | **ROTATED** |
| PII data-encryption key material (Fernet) | never committed | versioned key rotation + re-encryption | **ROTATED AND PII MIGRATED** |
| Supabase `service_role` key + project URL | historically committed | provider dashboard/management API, **account owner only** | **OWNER ACTION REQUIRED** |
| Export signing key, compliance token, admin passphrase, `ACCESS_DOC_KEY` | never committed | out of scope | UNCHANGED |

Types only, plus scope of action: the phase touched exactly one encryption library (Fernet/AES-CBC+HMAC), one key
derivation rule (`base64(sha256(material))`), three ciphertext stores, and no scientific artifact.

## Migration version and scheme

| Item | Value |
|---|---|
| Target scheme | versioned key map, ciphertext tagged `vN:` (untagged = the legacy `v1` form) |
| Migration version (own key material) | `PII_KEY_VERSION=v2`, material in `PII_KEY_CURRENT` |
| Legacy version kept for migration reads only | `v1`, material in `PII_KEY_LEGACY` (plus the historical `PII_ENCRYPTION_KEY` entry) |
| Key fingerprint — `v1` (retired, non-secret, sha256 prefix of the derived key) | `4101a113988fad4c` |
| Key fingerprint — `v2` (current) | `248f5b3d3a0f00af` |
| Coupling to authentication | **none** — measured, not assumed: the JWT-derived key decrypts 0 of 2487 stored tokens while the PII-derived key decrypts all of them |

## Records processed

| Metric | Value |
|---|---|
| Stores surveyed | 3 (`db/identity.db`, `db/walkthrough/identity.db`, `db/wt-scenarios/identity.db`) |
| Records (rows) surveyed | 835 |
| Fernet tokens discovered | 2500 |
| Tokens requiring migration (dry run) | 2500 |
| **Tokens successfully migrated** | **2490** |
| Migrated with a failed read-back or failed verify | **0** |
| **Tokens requiring manual attention** | **10** — all in the pre-existing `wt-scenarios` fixture store |
| Tokens skipped because already current | 0 (first run); re-runs migrate nothing (idempotent) |
| Migration problems (`--apply` exit 0) | 0 |

The 10 manual-attention tokens are **not** an outcome of this phase: they were written under a per-process random
development key by the scenario fixture generator before Phase 4A, are readable by no key in this repository, and
were left **byte-identical** (quarantined, never overwritten, never deleted). They are excluded from every live
store and from every evidence record; if that fixture is needed again it must be regenerated.

## Authentication verification

Stack restarted with the rotated configuration; all six `/health` endpoints returned 200.

| Check | Result |
|---|---|
| Login with the rotated signing secret | 200 |
| Token signed with the **old** signing secret | **401 (rejected)** |
| Same claims signed with the **new** secret (control) | 200 |
| Admin passphrase login and an authenticated admin API call | 200 / 200 |
| Token contents/claims unchanged after rotation (verification only, no re-issue) | unchanged |

## Database verification

| Check | Result |
|---|---|
| `db/identity.db` — every PII token read with the current key | **2595 / 2595** |
| `db/identity.db` — tokens still readable **only** by the retired key | **0** |
| `db/walkthrough/identity.db` — current key / legacy key | 3 / 0 |
| `db/wt-scenarios/identity.db` | 0 / 0 — the 10 quarantined tokens, unreadable by any available key |
| Records lost or silently dropped | 0 (row counts unchanged apart from rows legitimately created afterwards) |
| Records duplicated | 0 |
| New writes after the migration | tagged `v2:` — 108 tokens created by the running stack after the migration are all already on the new key |
| Provider (Supabase) database operations | **not exercised** — no local component reads that credential; nothing to verify on this host |

## Old-credential invalidation result

| Old credential | Test | Result |
|---|---|---|
| JWT signing secret | token signed with the retired value presented to the identity service | **invalid (401)** |
| Internal service token | retired value on risk `/internal/evaluate` and privacy `/internal/ingest-transaction` | **rejected (401)** on both |
| PII data key (v1) | retired key used to read every migrated token | **0 of 2487** readable; the live service writes and reads only v2 |
| Supabase `service_role` key | **not tested** — rotating or probing the provider credential needs account ownership | recorded as owner action, not as a pass |

## Regression-test result

| Suite | Result |
|---|---|
| `backend/scripts/pii_key_rotation_test.py` (**new**, hermetic, in the CI fast list) | **31 / 31 checks pass, 0 fail** — round-trip, legacy read, migration + read-back, reverse migration (rollback path), corrupted ciphertext rejected, unknown key version refused, idempotent duplicate run, blocked-write rollback, JWT↔PII independence in both directions, fail-closed on missing keys, no plaintext or key material in logs/errors/report |
| `backend/scripts/regression_suite.py --fast` | **23 / 23 PASSED** (38.1 s; includes the new suite) |
| `backend/scripts/secret_hygiene_test.py` | **16 / 16 PASS** |
| Bandit `-r backend/src --severity-level medium` (the CI command) | **exit 0** |

The new suite found two real defects in the migration tool before it touched live data (tagged tokens not recognised
as ciphertext; tag not stripped before decryption) and one after the live run (backup file names collided across the
three same-named stores — fixed, and the rollback path was changed to not depend on those files). All three are
documented in the report; none affected migrated data.

## Repository verification

| Check | Result |
|---|---|
| Regression battery `bash .freebuff/p114_battery.sh` | **84 / 84 PASS, 0 FAIL, 0 MISSING** (ended 2026-10-06T11:08:40+05:30); the pre-fix run had one FAIL, whose cause was fixed in the supply-chain suite's own exclusion list |
| `scripts/claim_evidence_check.py` | PASS (22 claims) |
| `backend/scripts/eval_record_test.py` | PASS (22 / 22) |
| `backend/scripts/review_resolution_check.py` | PASS (14 / 14 decisions; no freeze record or tag) |
| `backend/scripts/review_package_check.py` | **PASS** (13 / 13 sections; no licence cleared, no reviewer assigned, no 50M claim, freeze state unchanged) |
| `scripts/check_freeze.py` | **rc 1 / 78 placeholder failures — expected by design**, unchanged by this phase |
| Targeted scan of all 15 live `.env` values **and** all retired values over 1741 tracked/to-be-committed files | **no hits** (only the non-secret `CORS_ORIGINS` is present by design) |
| `git status` on `models/`, `data/`, `backend/src/risk_engine/rules.yaml`, Research Plan, preregistration, review and dataset documents | **clean — no scientific artifact modified** |
| CI on the final commit | **both workflows green** on `28d0935` — CI/CD run `37420876940` (all five jobs) and Security Scan run `37420876927` (see *CI result* below) |

## Repository invariants

This phase changed application and security code, its tests, and evidence documentation only. Not modified: the
Research Plan, preregistration, statistical and domain review records, datasets, synthetic benchmark specification,
model artifacts, thresholds, calibration, and the fraud-detection methodology. No new evaluation metric, split or
claim was introduced, and no previously recorded result was altered.

## CI result

Verified through the GitHub Actions API against the phase tip `28d0935` (not inferred from a badge — the
in-progress run was excluded until it reported a conclusion):

| Workflow | Run | Conclusion | Jobs |
|---|---|---|---|
| CI/CD | [#138 / run `37420876940`](https://github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software/actions/runs/37420876940) | **success** | 5 / 5 — Test Suite, Security Scan, Docker Build, Integration Test, Deploy Image |
| Security Scan | [#142 / run `37420876927`](https://github.com/RanaYashIDK-2006/PS-14-Fraud-Detection-software/actions/runs/37420876927) | **success** | 1 / 1 |

The Security Scan workflow is the one that enforces the new hygiene suite and the Bandit gate, so the secret-scan
and SAST steps executed on the rotated configuration rather than only locally. Both runs started from the same push
at 2026-10-06T05:54:19Z and completed successfully (CI/CD at 06:05:10Z).

The documentation-only commit that carries this record is pushed to the same branch and therefore runs the same two
workflows; its outcome is the last CI result reported with this hand-off. Every commit after `28d0935` is
non-code — this CI record plus hook-generated evaluation-run records — so the verified tip and the phase tip differ
only in documentation.

## Owner action still required

| Item | What is needed |
|---|---|
| Supabase `service_role` key + project URL (historically exposed in a tracked script) | Roll the credential in the Supabase dashboard / management API and replace the value in the deployment secret store. Until then the interim state is: the value is **not** committed, **no** local service reads it (the front service only strips `SUPABASE_*` from child environments), and the tracked script is env-driven with fail-fast behaviour. |
| Retirement of `PII_KEY_LEGACY` / `PII_ENCRYPTION_KEY` and the pre-rotation `.env` copy | Delete once the rollback window closes; every store already reports nothing left to migrate |
| Rewriting git history to erase the historical exposure | Separate operational decision with collaborator/CI consequences; deliberately not performed |

## Final decision

> ## `REQUIRES OWNER ACTION`
>
> `OWNER ACTION REQUIRED — PROVIDER CREDENTIAL ROTATION`

Every locally held credential is rotated, the retired values are demonstrably rejected, and the PII encryption
material has been migrated to the versioned replacement key with zero failures. The single unfinished item — the
Supabase `service_role` credential — cannot be completed from this environment because it needs the account owner.
Phase 5 work must not treat that item as closed.
